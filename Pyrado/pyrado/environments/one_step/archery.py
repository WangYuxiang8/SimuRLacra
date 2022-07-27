from copy import deepcopy

import numpy as np
import torch as to
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.pysim.base import SimPyEnv
from pyrado.spaces import BoxSpace

from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.final_reward import FinalRewMode, FinalRewTask
from pyrado.tasks.reward_functions import AbsErrRewFcn


class Archery(SimPyEnv, Serializable):
    """
    DARC, Archery environment.

    [1]. OFF-DYNAMICS REINFORCEMENT LEARNING: TRAINING FOR TRANSFER WITH DOMAIN CLASSIFIERS
    """

    name: str = "archery"

    def __init__(self):
        """Constructor"""
        Serializable._init(self, locals())

        # Call SimEnv's constructor
        super().__init__(dt=1.0 / 200, max_steps=20)

        # Initialize the domain parameters and the derived constants
        self._domain_param = self.get_nominal_domain_param()

    def _create_spaces(self):
        self.max_obs = np.array([40.])
        self.max_act = np.array([np.pi / 20])

        self._state_space = BoxSpace(-self.max_obs, self.max_obs, labels=["pos"])
        self._obs_space = BoxSpace(-self.max_obs, self.max_obs, labels=["pos"])
        self._init_space = BoxSpace(-self.max_obs / self.max_obs[0], self.max_obs / self.max_obs[0], labels=["pos"])
        self._act_space = BoxSpace(-self.max_act, self.max_act, labels=["theta"])

    def _create_task(self, task_args: dict) -> Task:
        state_des = task_args.get("state_des", np.array([0.0]))
        Q = task_args.get("Q", np.diag([1.]))
        R = task_args.get("R", np.diag([0.]))

        return FinalRewTask(
            DesStateTask(self.spec, state_des, AbsErrRewFcn(Q, R)),
            mode=FinalRewMode(state_dependent=True, time_dependent=True),
        )

    def observe(self, state) -> np.ndarray:
        return np.array([state[0]])

    def _calc_constants(self):
        self.wf = self.domain_param["wf"]

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        """
        param = dict(
            wf: 0.0, # wind force
        )
        """
        return dict(
            wf=0.0
        )

    def _step_dynamics(self, act: np.ndarray):
        # print(self.wf)
        state = 70 * np.sin(act) + 5 * self.wf / np.power(np.cos(act), 2)
        max_obs = self.max_obs[0]
        state = np.clip(state, -max_obs, max_obs)
        self.state[0] = state
