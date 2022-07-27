import pyrado
from copy import deepcopy
from pyrado.environments.pysim.openai_classical_control import GymEnv
from pyrado.spaces.discrete import DiscreteSpace
from init_args_serializer import Serializable
import numpy as np


class LunarLanderRado(GymEnv, Serializable):

    name: str = "lunar-lander"

    def __init__(self):
        super(LunarLanderRado, self).__init__("LunarLander-v2", True)
        self._domain_param = self.get_nominal_domain_param()
        self._set_domain_param_attrs(self.get_nominal_domain_param())
        self._calc_constants()

    @classmethod
    def get_nominal_domain_param(cls):
        return dict(
            me=13.0,      # main engine
            se=0.6        # side engine
        )

    @property
    def domain_param(self) -> dict:
        return deepcopy(self._domain_param)

    @domain_param.setter
    def domain_param(self, domain_param):
        if not isinstance(domain_param, dict):
            raise pyrado.TypeErr(given=domain_param, expected_type=dict)
        # Update the parameters
        self._domain_param.update(domain_param)
        self._calc_constants()

    def _calc_constants(self):
        # need to modify the source gym file
        self._gym_env.set_rado_params(self.domain_param["me"], self.domain_param["se"])

    def _set_domain_param_attrs(self, domain_param: dict):
        """
        Set all key value pairs of the given domain parameter dict to the state dict (i.e. make them an attribute).

        :param domain_param: dict of domain parameters to save a private attributes
        """
        for name in self.supported_domain_param:
            dp = domain_param.get(name, None)
            if dp is not None:
                setattr(self, name, dp)

    def reset(self, init_state=None, domain_param=None):
        # Reset the domain parameters
        if domain_param is not None:
            self.domain_param = domain_param
        self.state = self._gym_env.reset()
        self._gym_env.close()
        return self.state

    def step(self, act) -> tuple:
        if isinstance(self.act_space, DiscreteSpace):
            act = act.astype(dtype=np.int64)  # PyTorch policies operate on doubles but discrete gym envs want integers
            act = act.item()  # discrete gym envs want integers or scalar arrays
        self.state, reward, done, info = self._gym_env.step(act)
        return self.state, reward, done, info


