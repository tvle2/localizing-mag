import jax
import jax.numpy as jnp
from queso.sensors import Sensor
from queso.estimators import BayesianDNNEstimator
from dataclasses import dataclass, field, fields, asdict
import yaml
from queso.configs import Configuration
import argparse

import jax

from queso.io import IO
parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="tmp")
args = parser.parse_args()
folder = args.folder

io = IO(folder=f"{folder}")
print(io)

def to_yaml(cnf, file):
    data = asdict(cnf)
    # for key, val in data.items():
    #     if isinstance(val, jnp.ndarray):
    #         data[key] = val.tolist()
    with open(file, "w") as fid:
        yaml.dump(data, fid)

config = Configuration(n_epochs = 80)#phi_fi=1.157
to_yaml(config,io.path.joinpath('config.yaml'))