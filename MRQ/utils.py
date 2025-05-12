# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import pprint
from typing import Union

import numpy as np


def enforce_dataclass_type(dataclass: dataclasses.dataclass):
    for field in dataclasses.fields(dataclass):
        setattr(dataclass, field.name, field.type(getattr(dataclass, field.name)))


def set_instance_vars(hp: dataclasses.dataclass, c: object):
    for field in dataclasses.fields(hp):
        c.__dict__[field.name] = getattr(hp, field.name)


class Logger:
    def __init__(self, log_file: str):
        self.log_file = log_file


    def log_print(self, x: Union[str, object]):
        with open(self.log_file, 'a') as f:
            if isinstance(x, str):
                print(x)
                f.write(x+'\n')
            else:
                pprint.pprint(x)
                pprint.pprint(x, f)


    def title(self, text: str):
        self.log_print('-'*40)
        self.log_print(text)
        self.log_print('-'*40)


# Takes the formatted results and returns a dictionary of env -> (timesteps, seed).
def results_to_numpy(file: str='../results/gym_results.txt'):
    results = {}

    for line in open(file):
        if '----' in line:
            continue
        if 'Timestep' in line:
            continue
        if 'Env:' in line:
            env = line.split(' ')[1][:-1]
            results[env] = []
        else:
            timestep = []
            for seed in line.split('\t')[1:]:
                if seed != '':
                    seed = seed.replace('\n', '')
                    timestep.append(float(seed))
            results[env].append(timestep)

    for k in results:
        results[k] = np.array(results[k])
        print(k, results[k].shape)

    return results

air_hockey_envs = [
    '3dof',
    '3dof-hit',
    '3dof-defend',
    '7dof',
    '7dof-hit',
    '7dof-defend',
    '7dof-prepare',
    'tournament',
]
