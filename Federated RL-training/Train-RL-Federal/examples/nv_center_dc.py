#!/usr/bin/env python3
from typing import Literal, Callable, List, Optional, \
    Tuple

import tensorflow as tf
from tensorflow import cast, stop_gradient, ones, \
    gather, concat, reshape, norm, expand_dims, \
        broadcast_to, constant, \
            Variable, Tensor
from tensorflow.math import exp, log, cos, abs, minimum, \
    reciprocal_no_nan, round
from tensorflow.linalg import trace, tensor_diag, matmul
from tensorflow.random import stateless_uniform, uniform, \
    Generator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.config.experimental import get_memory_info
from numpy import ceil, sqrt, zeros, savetxt, loadtxt
from numpy.random import rand
from itertools import product
from argparse import ArgumentParser
from os.path import join
from math import pi

from src.qsensoropt import StatelessPhysicalModel, \
    StatelessMetrology,  ParticleFilter, \
        Parameter, Control, SimulationParameters
from src.qsensoropt.utils import train, \
     get_seed

class NVCenter(StatelessPhysicalModel):

    def __init__(
        self, batchsize: int, params: List[Parameter],
        prec: Literal["float64", "float32"] = "float64",
        res: Literal["meas", "time"] = "time",
        control_phase: bool = False,
        ):

        self.control_phase = control_phase

        if not res in ("meas", "time"):
            raise ValueError("The allowed values of \
                             res are time and res.")
        
        if self.control_phase:
            # Controls the time and the phase.
            controls=[
                Control(name="EvolutionTime"),
                Control(name="Phase"),
            ]
        else:
            # The only control is the free precession
            # time of the NV center
            # in the magnetic field, between the two pi/2-pulses.
            controls=[
                Control(name="EvolutionTime"),
            ]
        
        super().__init__(
            batchsize, controls, params,
            prec=prec,
        )

        self.res = res

    def perform_measurement(
        self, idxN:int,controls: Tensor, parameters: Tensor,
        meas_step: Tensor,
        rangen: Generator,
        ):

        list_plus = ones((self.bs, 1, 1), dtype=self.prec)
        prob_plus = self.model(idxN,
            list_plus, controls, parameters, meas_step,
            )
        # Extraction of the actual outcomes
        seed = get_seed(rangen)
        outcomes = 2*cast((stateless_uniform((self.bs, 1),
                        seed, dtype=self.prec) <
                        stop_gradient(abs(prob_plus))),
                        dtype="int8") - 1
        outcomes = cast(
            expand_dims(outcomes, axis=1),
            dtype=self.prec, name="outcomes",
            )
        prob_outcomes = self.model(idxN,
            outcomes, controls, parameters, meas_step,
            )
        log_prob = cast(
            log(prob_outcomes),
            dtype=self.prec, name="log_prob",
            )
        return outcomes, log_prob

    def count_resources(
        self, resources: Tensor, outcomes: Tensor,
        controls: Tensor, true_values: Tensor,
        meas_step: Tensor,
        ):
        if self.res == "time":
            return resources+abs(controls[:, 0:1])+240 #5#25.0
        return resources+240 #5#25.0#1.0


class NVCenterDCMagnetometry(NVCenter):

    def __init__(
        self, batchsize: int, params: List[Parameter],
        prec: Literal["float64", "float32"] = "float64",
        res: Literal["meas", "time"] = "meas",
        invT2: Optional[float] = None,
        ):

        super().__init__(
            batchsize, params, prec=prec, res=res,
        )

        self.invT2 = invT2


    
class Magnetometry(StatelessMetrology):

    def __init__(
            self, particle_filter: ParticleFilter,
            phys_model: NVCenter,
            control_strategy: Callable,
            simpars: SimulationParameters,
            cov_weight_matrix=None,
            eta_flag: bool = False,
            extraction_flag: bool = False,
            cov_flag: bool = False,
    ):

        super().__init__(
            particle_filter, phys_model, control_strategy,
            simpars, cov_weight_matrix,
            )

        d = self.pf.d
        self.eta_flag = eta_flag
        self.extraction_flag = extraction_flag
        self.cov_flag = cov_flag

        if self.extraction_flag:
            self.input_size += 2*d
            self.input_name +=  [f"Ext1_{par.name}" \
                                 for par in self.pf.phys_model.params] + \
                [f"Ext2_{par.name}" for par in self.pf.phys_model.params]
        if self.cov_flag:
            self.input_size += d**2
            self.input_name += [f"Cov_{par1.name}_{par2.name}" \
                                for par1, par2 \
                    in product(self.pf.phys_model.params,
                               self.pf.phys_model.params)]

    def generate_input(
        self, weights: Tensor,
        particles: Tensor,
        meas_step: Tensor,
        used_resources: Tensor,
        rangen: Generator,
        ):

        d = self.pf.d
        input_tensor = super().generate_input(
            weights, particles, meas_step, used_resources,
            rangen,
            )

        if self.extraction_flag:
            two_particles = self.pf.extract_particles(
                weights, particles, 2, rangen,
                )
            input_tensor = concat(
                [input_tensor, reshape(two_particles,
                                       (self.bs, 2*d))], 1,
                )
        if self.cov_flag:
            cov = self.pf.compute_covariance(weights, particles)
            input_tensor = concat(
                [input_tensor, reshape(cov, (self.bs, d**2))], 1,
            )
        return input_tensor
        
    def loss_function(
            self, weights: Tensor, particles: Tensor,
            true_values: Tensor, used_resources: Tensor,
            meas_step: Tensor,
    ):

        pars = self.simpars
        loss_values = super().loss_function(
                weights, particles, true_values,
                used_resources, meas_step,
            )
        d = len(self.pf.phys_model.params)
        if self.eta_flag:
            init_var = []
            for param in self.pf.phys_model.params:
                bounds = param.bounds
                init_var.append(1/12*(bounds[1]-bounds[0])**2)
            initial_loss = broadcast_to(reshape(
                tensor_diag(constant(init_var, dtype=pars.prec)),
                (1, d, d),
                ), (self.bs, d, d),
            )
            initial_loss_scalar = expand_dims(
                trace(matmul(
                    self.cov_weight_matrix_tensor, initial_loss),
                    ), axis=1,
                    )
            eta = minimum(
                reciprocal_no_nan(used_resources),
                initial_loss_scalar,
            )
            loss_values = loss_values/eta
        return loss_values

