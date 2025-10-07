"""Module containing some utility functions
for training the control strategy and visualizing the
results.
"""
from typing import Tuple, TypedDict, Callable, \
    List

from tensorflow import GradientTape, function, zeros, transpose, zeros_like, print, Variable
from tensorflow import reshape, broadcast_to, \
    linspace, Tensor, dtypes
from tensorflow import range as tfrange
from tensorflow.linalg import svd, diag, matmul
from tensorflow.math import log, lgamma, abs, tanh
from tensorflow.math import sqrt as tfsqrt
from tensorflow.profiler.experimental import Trace, start, stop
from tensorflow.summary import trace_on, create_file_writer, \
    trace_export
from tensorflow.random import stateless_uniform, Generator
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from numpy import repeat, concatenate, savetxt, loadtxt, \
    linspace, inf
from numpy import sqrt as npsqrt
from numpy import floor as npfloor
from numpy import tanh as nptanh
from numpy import exp as npexp
import scipy.integrate as integrate
from pandas import DataFrame, Series
from os.path import join
from os import makedirs
from math import pi, floor
from datetime import datetime
from typing import Optional
from progressbar import progressbar



def normalize(
        input_tensor: Tensor,
        bounds: Tuple[float, float],
) -> Tensor:
    r"""Normalizes the entries of
    `input_tensor` in the interval `[-1, 1]`.
    The `input_tensor` entries must lay within the
    given tuple of upper and lower bounds.

    Parameters
    ----------
    input_tensor: Tensor
        `Tensor` of unnormalized values.
    bound: List[float, float]
        Min and max of the admissible entries
        of `input_tensor`.

    Returns
    -------
    Tensor:
        Version of `input_tensor` with entries
        normalized in the interval `[-1, 1]`.
        Each entry :math:`x` of the input becomes

        .. math::
            y = \frac{2(x-\text{bound}[0])}{\text{bound}[1]-
            \text{bound}[0]} - 1 \;

        in the returned `Tensor`, which has the same
        type and shape of `input_tensor`.
    """
    step_1 = (input_tensor-bounds[0])/(bounds[1]-bounds[0])
    return 2*step_1-1


def denormalize(
        input_tensor: Tensor,
        bounds: Tuple[float, float],
) -> Tensor:
    r"""Given `input_tensor` with values in `[-1, 1]`
    this function rescales it, so that its entries take
    values within the given tuple of extrema.

    Parameters
    ----------
    input_tensor: Tensor
        `Tensor` having entries normalized in `[-1, 1]`.
    bound: Tuple[float, float]
        Min and max of the admissible entries
        of the returned `Tensor`.

    Returns
    -------
    Tensor:
        Version of `input_tensor` normalized in
        the interval delimited by the new extrema.
        Each entry :math:`y` of `input_tensor` becomes

        .. math::
            x = \frac{y (\text{bound}[1]-
            \text{bound}[0])}{2}+\text{bounds}[0] \; .

        It is a `Tensor` with the same type and shape
        of `input_tensor`.
    """
    step_1 = (input_tensor+1)/2
    return (step_1*(bounds[1]-bounds[0]))+bounds[0]


def get_seed(
        random_generator: Generator,
) -> Tensor:
    """Generate a random seed from `random_generator`.

    Extracts uniformly a couple of integers from
    `random_generator` to be used as seed in the stateless
    functions of the :py:mod:`tensorflow.random` module.

    Parameters
    ----------
    random_generator: Generator
        Random number generator from the module
        :py:mod:`tensorflow.random`.

    Returns
    -------
    seed: Tensor
        `Tensor` of shape (2, ) and of type `int32`
        containing two random seeds.
    """
    return random_generator.uniform(
        [2, ], minval=0, maxval=dtypes.int32.max,
        dtype="int32", name="seed",
    )


def random_uniform(
        batchsize: int, prec: str,
        min_value: float, max_value: float,
        seed: Tensor,
):
    """Extracts `batchsize` random number of type `prec` between
    the `min_value` and the `max_value` from a uniform
    distribution.

    Parameters
    ----------
    batchsize: int
        Number of random values to be extracted.
    prec: str
        Type of the extracted numbers.
        It is typically `float32` or `float64`.
    min_values: float
        Lower extremum of the uniform distribution
        from which the values are extracted.
    max_values: float
        Upper extremum of the uniform distribution
        from which the values are extracted.
    seed: Tensor
        Seed of the random number generator used in this
        function. It is a `Tensor` of type `int32` and of shape
        (`2`, ). This is the kind of seed that is accepted by
        the stateless random
        function of the module :py:mod:`tensorflow.random`.
        It can be generated with the function
        :py:func:`~.utils.get_seed` from a
        :py:obj:`Generator` object.

    Returns
    -------
    Tensor:
        `Tensor` of shape (`batchsize`, ) and of
        type `prec` with uniformly extracted entries.
    """
    random_values = stateless_uniform(
        (batchsize, ), seed, maxval=1, dtype=prec,
        name="stateless_uniform",
    )
    return (max_value - min_value)*random_values + min_value


def train(
        simulation,
        optimizer: Optimizer,
        iterations: int,
        save_path: str,
        interval_save: int = 16,
        networks: List[Optional[Model]]= None ,
        custom_controls: Optional[Variable] = None,
        load_best: bool = True,
        gradient_accumulation: int = 1,
        numN: int = 10,
        xla_compile: bool = True,
        rangen: Generator = Generator.from_seed(0xdeadd0d0),
):
    """Training routine for the neural network
    or the custom controls.

    This function contains a cycle that
    performs a gradient descent optimization
    of the training variables of the control strategy.

    Parameters
    ----------
    simulation: :py:obj:`~.Simulation`
        Contains the complete description of the
        estimation task, and how the various components
        of it (particle filter, probe system, and control)
        interact to produce the estimator and the loss.
    optimizer: :py:obj:`Optimizer`
        This object applies the gradient to the trainable
        variables of the control strategy and realized therefore
        the update of the gradient descent step.
        The default optimizer to be chosen
        is `Adam` with the variable learning
        rate described in the
        class :py:obj:`~.InverseSqrtDecay`.
    iterations: int
        Number of update steps in the training of the
        strategy. It is the number of complete executions
        of a batch of estimation tasks for training the controls.
    save_path: str
        Path containing the intermediate trained variables,
        that are saved each `interval_save` iterations,
        and the history of the loss at the end of the
        training.
    interval_save: int = 128
        Number of iterations of the training cycle
        between two consecutive saving of the training
        variables on disk.
    network: :py:obj:`Model`, optional
        Neural network model used in the `simulation`
        object to parametrize the control strategy.
        In `simulation` it is wrapped inside the
        `control_strategy` attribute and there the
        :py:obj:`~.Simulation` object has no direct access
        to its trainable variables. If no neural network
        is used in the estimation, for example
        because a static strategy is being trained,
        this parameter can be omitted.
    custom_controls: :py:obj:`Variable`, optional
        It is possible to train a control strategy that
        goes beyond what can be done with a neural network,
        by writing a custom `control_strategy` function
        which is not only a wrapper for the neural
        network. The trainable variables of this custom
        strategy are contained in the object object
        `custom_controls`.

        The typical application is that of a **static strategy**,
        that is a control strategy that depends only
        on the measurement step and not on the
        posterior distribution moments. This is a
        non-adaptive strategy, that therefore
        doesn't need to be
        computed in real time, but can be hard coded before
        the beginning of the estimation.
        Such approach will save memory, because the
        Bayesian analysis can be postponed
        and performed on a more powerful device,
        and it will also be faster, because no
        computation is necessary between two
        measurements, since the approach is
        non-adaptive. Of course one can't expect
        in general the same performances of the
        adaptive strategy.
    load_best: bool = True
        Keeps track of the loss of the control strategy
        during the training and picks the best model
        instead of the last. To pick the best model
        the loss is averaged over
        `interval_save` update steps.
    gradient_accumulation: int = 1
        This controls the "effective" batchsize of the
        training. In the training loop for each
        update of the variables the method
        :py:meth:`~.Simulation.execute` runs
        a number `gradient_accumulation` of times
        and the gradients computed from each run
        are averaged before the update step
        is performed.
        In this way, given `bs` the true batchsize of the
        `simulation` object, the "effective" batchsize
        of the training is
        `gradient_accumulation*bs`, which is the number of
        independent simulations that contribute to a single
        update step of the training variables. This feature
        can be used to increase the batchsize, when the
        memory on the machine is limited,
        since the maximum amount of memory required
        by the training is determine by
        `bs` and is independent on
        `gradient_accumulation`.
    xla_compile: bool = True
        Just-in-time XLA (Accelerated Linear Algebra)
        compilation for the strategy training.
        It should reduce the memory and training
        time on a GPU.


        **Achtung!** Not all the Tensorflow
        operations can be compiled with XLA.

        **Achtung!** It might not perform well on a CPU.
    rangen: Generator = Generator.from_seed(0xdeadd0d0)
        Random number generator from the module
        :py:mod:`tensorflow.random`.

    After the return of the :py:func:`~.utils.train`
    routine
    either `network` or the `custom_controls`
    will be trained
    and in `save_path` there will be
    a directory containing the partially trained
    variables (with name ending in "_weights")
    and a csv file with the history
    of the loss (with name ending in "_history.csv"),
    averaged over a window of
    `interval_save` iterations.

    """
    if networks[0] is not None:
        variabless = [networks[i].trainable_variables for i in range(numN)]
        custom = False
    elif custom_controls is not None:
        variables = [custom_controls]
        custom = True
    else:
        print("No NN or static_controls given!")
        return
    
    def _single_iteration(idxN, rangen):
        with GradientTape() as tape:
            loss_diff, loss = simulation.execute(idxN,rangen)
        print("\nLoss: ", loss)
        grads = tape.gradient(loss_diff, variabless[idxN])
        return grads, loss
    
    single_iteration = function(jit_compile=False)(_single_iteration)
    
        

    def update_nn(rangen):
        Grads = []
        accLoss = []
        for idxN in range(numN):
            #print('Train--',idxN)
            acc_loss = zeros((1,), simulation.simpars.prec)
            acc_gradient = [zeros_like(this_var)
                            for this_var in variabless[idxN]]
            for _ in range(gradient_accumulation):
                grads, loss = single_iteration(idxN,rangen)
                acc_loss += loss
                acc_gradient = [(acc_grad+grad)
                            for acc_grad, grad in
                            zip(acc_gradient, grads)]
                acc_gradient = [this_grad /
                        gradient_accumulation for
                        this_grad in acc_gradient]
            Grads.append(acc_gradient)
            accLoss.append(acc_loss)
        for idxN in range(1,numN):
            #print('Grads[idxN][key]',Grads[idxN])
            Grads[0] += Grads[idxN]

        Grads[0] = [this_grad/numN for this_grad in Grads[0]]
        for idxN in range(numN):
            optimizer.apply_gradients(zip(Grads[0], variabless[idxN]))
            accLoss[idxN] /= gradient_accumulation
        acc_Loss = zeros((1,), simulation.simpars.prec)
        for idxN in range(numN):
            acc_Loss += accLoss[idxN]/numN
        return acc_Loss


    loss_list = []
    for j in progressbar(range(iterations)):
        loss = update_nn(rangen)
        print("\nFL-Loss: ", loss)
        loss_list.append(loss.numpy())

        if (j+1) % interval_save == 0:  # Save the NN weights
            dir_name = save_path+ \
                "_history_weights/"

            # Save model weights
            print(dir_name)
            if custom:
                makedirs(dir_name, exist_ok=True)
                savetxt(dir_name+str(floor((j+1)/interval_save)), custom_controls.numpy())
            else:
                for i in range(numN):
                    # networks[i].save_weights(dir_name+str(floor((j+1)/interval_save))+str(i))
                    networks[i].save_weights(f"{dir_name}{floor((j + 1) / interval_save)}_{i}.weights.h5")
    # The last training steps,
    # that would not fit in the block
    # average are trimmed out.
            loss_array = concatenate(loss_list, axis=0)
            num_nn_saved = floor(iterations/interval_save)
            loss_df = DataFrame(
                {'Loss': loss_array[:num_nn_saved*interval_save]},
            )
            groups_block_mean = Series(
                repeat(range(num_nn_saved), interval_save),
            )
            mean_loss = loss_df.groupby(groups_block_mean).agg('mean')
            mean_loss.to_csv(
                save_path+'_history.csv',
                index=False,
                float_format='%.4e',
            )
    if load_best:
        index_best = mean_loss.idxmin().values[0] + 1
        file_path = save_path + \
            "_history_weights/"
        file_name  = file_path+str(index_best)
        print(file_name,str(index_best),file_name+str(2))
        if custom:
            loaded_controls = loadtxt(file_name)
            if len(loaded_controls.shape) == 1:
                loaded_controls = loaded_controls[:, None]
            custom_controls.assign(loaded_controls)

        else:
            # Loading of the best model
            for i in range(numN):
                networks[i].load_weights(file_name+str(i))
                networks[i].save(
                    join(save_path, str(i)),
                )




class FitSpecifics(TypedDict):
    """This dictionary specifies the hyperparameters of the
    precision fit operated by the
    function :py:func:`~.utils.performance_evaluation`."""
    num_points: int
    """After the fit the neural network
    representing the relation between the resources
    and the average precision is evaluated
    on `num_points` resources values
    equally spaced in the
    interval [0, `max_resources`], with
    `max_resources` being the attribute
    of :py:obj:`~.SimulationParameters`."""
    batchsize: int
    """Batchsize of the training of the
    neural network to fit the
    Precision/Resources relation. The
    data cloud is divided in minibatches
    and each of them is used for a sigle
    iteration of the training loop."""
    epochs: int
    """The number of trainings on the
    same data."""
    direct_func: Callable
    r"""Callable object that takes in input
    the precision and the consumed resources
    and outputs a values :math:`x`
    that is of order one. In symbols

    .. math::
        f(\text{Resources}, \text{Precision})
        = x \sim \mathcal{O}(1) \; .

    This requires having some knowledge of the
    expected precision given the resources, which
    could be for example a Cramér-Rao bound
    on the precision."""
    inverse_func: Callable
    r"""Inverse of the function defined
    by `direct_func`, that is

    .. math::
        g(\text{Resources}, x) = \text{Precision} \; ."""





def standard_model(
    input_size: int = 1,
    controls_size: int = 1,
    neurons_per_layer: int = 64,
    num_mid_layers: int = 5,
    prec: str = "float64",
    normalize_activation: bool = True,
    sigma_input: float = 0.33,
    last_layer_activation: str = "tanh",
) -> Model:
    """Returns a dense neural network to fit the optimal
    control strategy.

    Parameters
    ----------
    input_size: int = 1
        Number of scalars given in input to the neural
        network.
    controls_size: int = 1
        Number of scalars outputted by the neural network.
        It corresponds to the number of controls in the
        estimation task.
    neurons_per_layer: int = 64
        Number of units (neurons) in each intermediate
        layer of the neural network. The intermediate
        layers of the neural network
        are all identical.
    num_mid_layers: int = 5
        Number of layers between the input and the output
        of the neural network. Should be greater then one.
    prec: str = "float64"
        Type of the weights and biases of the neural network.
    normalize_activation: bool = True
        If this flag is active the activation function (`tanh`)
        is normalized to preserve the variance
        of the input after each layer of neurons. This
        should speed up the convergence.
    sigma_input: float = 0.33
        Approximate variance of the input of the neural
        network. The default value is the variance of
        a uniform distribution in `[-1, +1]`.
    last_layer_activation: str = "tanh"
        Activation function of the last neural
        network layer.
    Returns
    -------
    Model:
        Sequential model made up of
        `num_mid_layers` intermediate layers, each with
        `neurons_per_layer` units. The initialization
        of the weight and biases is done with
        *glorot uniform* and the activation function
        is `tanh`, possibly normalized to reduce
        the saturation.

        .. image:: ../docs/_static/nn_definition.png
            :width: 500
            :alt: definition_nn
    """
    # There should always be at least 2 intermediate layers.
    if num_mid_layers <= 1:
        raise ValueError("num_mid_layers must be > 1.")

    if normalize_activation:
        sigma_z_input = npsqrt(
            2*input_size/(input_size+neurons_per_layer))*sigma_input
        sigma_z_mid = sigma_input
        sigma_z_output = npsqrt(2*neurons_per_layer /
                                (controls_size+neurons_per_layer))*sigma_input

        C_input = integrate.quad(
            lambda x: nptanh(x)**2*npgauss_pdf(x, sigma_z_input), -inf, inf,
        )[0]
        C_mid = integrate.quad(
            lambda x: nptanh(x)**2*npgauss_pdf(x, sigma_z_mid), -inf, inf,
        )[0]
        C_output = integrate.quad(
            lambda x: nptanh(x)**2*npgauss_pdf(x, sigma_z_output), -inf, inf)[0]

        input_norm, mid_norm, output_norm = sigma_input/npsqrt(C_input), \
            sigma_input/npsqrt(C_mid), sigma_input/npsqrt(C_output)

    else:
        input_norm, mid_norm, output_norm = 1.0, 1.0, 1.0

    layer_list = [
        Dense(neurons_per_layer,
              activation=lambda x: input_norm*tanh(x),
              dtype=prec, input_shape=(input_size, )),
    ]

    for _ in range(num_mid_layers-2):
        layer_list.append(
            Dense(neurons_per_layer,
                  activation=lambda x: mid_norm*tanh(x),
                  dtype=prec),
        )

    layer_list.append(
        Dense(neurons_per_layer,
              activation=lambda x: output_norm*tanh(x),
              dtype=prec),
    )

    layer_list.append(
        Dense(controls_size,
              activation=last_layer_activation,
              dtype=prec)
    )

    return Sequential(layer_list)


def npgauss_pdf(x, dev):
    """Logarithm of a batch of
    1D-Gaussian probability densities (compatible with
    Numpy)"""
    return 1/(npsqrt(2*pi)*dev)*npexp(-0.5*x**2/dev**2)


def loggauss(x, mean, dev):
    r"""Logarithm of a batch of
    1D-Gaussian probability densities.

    Parameters
    ----------
    x: Tensor
        Values extracted from the Gaussian distributions.
        Must have the same type and size of `mean` and `dev`.
    mean: Tensor
        Means of the Gaussian distributions.
        Must have the same type and size of `x` and `dev`.
    dev:
        Standard deviation of the Gaussian distributions.
        Must have the same type and size of `mean` and `x`.

    Returns
    -------
    Tensor:
        Logarithm of the probability densities for
        extracting the entries of `x` from the Gaussian
        distributions defined by `mean` and `dev`. It has
        the same shape and type of `x`, `mean`, and `dev`.

        Calling :math:`x`, :math:`\mu`, and :math:`\sigma`
        respectively an entry of the tensor `x`, `mean`, and `dev`,
        the corresponding entry of the returned tensor is

        .. math::
            -\log \left( \sqrt{2 \pi} \sigma \right) -
            \frac{(x-\mu)^2}{2 \sigma^2} \; .

    """
    return -log(npsqrt(2*pi)*dev)+(-0.5*((x-mean)/dev)**2)


def logpoisson(mean, k):
    r"""Logarithm of the probability densities
    of a batch of Poissonian distributions.

    Parameters
    ----------
    mean: Tensor
        Mean values defining the Poissonian distributions.
        Must have the same type and shape of `k`.
    k: Tensor
        Observed outcomes of the sampling from the
        Poissonian distributions.
        Must have the same type and shape of `k`.

    Returns
    -------
    Tensor:
        `Tensor` having the same type and shape
        of `mean` and `k`, whose entries are defined
        by

        .. math::
            k \log (\mu) - \mu - \log (k !) \; ,

        where :math:`k` and :math:`\mu` are respectively
        the entries of `k` and `mean`.
    """
    return k*log(mean) - mean - lgamma(k + 1)


def sqrt_hmatrix(matrix: Tensor) -> Tensor:
    """Square root of the absolute value of
    a symmetric (hermitian) matrix.

    The default matrix square root algorithm
    implemented in Tensorflow [7]_
    doesn't work for matrices with very small entries,
    this implementation does, and must be
    always preferred.

    Parameters
    ----------
    matrix: Tensor
        Batch of symmetric (hermitian) square matrices.

    Returns
    -------
    Tensor:
        Matrix square root of `matrix`.

    Examples
    --------
    ``A = constant([[1e-16, 1e-15], [1e-15, 1e-16]],
    dtype="float64", )``

    ``print(tf.sqrtm(A))``

    Output:

    ``[[-nan -nan], [-nan -nan]]``

    While ``print(sqrt_hmatrix(A))`` outputs

    ``[[1.58312395e-09, 3.15831240e-08],
    [3.15831240e-08, 1.58312395e-09]]``

    .. [7] N. J. Higham, "Computing real square
           roots of a real matrix", Linear
           Algebra Appl., 1987.
    """
    s, _, v = svd(matrix)
    return matmul(
        matmul(v, diag(tfsqrt(abs(s)))), v, adjoint_b=True,
        name="square_root",
    )
