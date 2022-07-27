import gym
from gym import spaces
import numpy as np
from gym.utils import seeding
import torch
import torch.nn as nn
import torch.nn.functional as F


class VesselModel(nn.Module):
    def __init__(self):
        super(VesselModel, self).__init__()
        self.fc1 = nn.Linear(10, 30)
        self.fc2 = nn.Linear(30, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 1)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练

    def forward(self, x):
        #x = F.relu(self.dropout(self.fc1(x)))
        #x = F.relu(self.dropout(self.fc2(x)))
        #x = F.relu(self.dropout(self.fc3(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class VesselEnvNN(gym.Env):
    """
    Args:
        conn: 连接套接字
    """

    name: str = "vessel_nn"

    def __init__(self):
        self.model = self._load_model('/home/wyx/workspace/SimuRLacra/Pyrado/pyrado/environments/pysim/vessel_net_1e6.pth')      # nn模型
        self.max_steps = 1000 # 10000
        self.max_episode_steps = self.max_steps
        self.reward_freq = 1
        self.const_a = -1.6e-5
        self.mode = 0

        self.obs_dim = 3
        self.act_dim = 2

        obs_high = np.array([np.inf] * self.obs_dim , dtype=np.float32)
        act_high = np.array([1.0] * self.act_dim, dtype=np.float32)

        self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps = 0

        self.param = {
            "wave_height": 3.0
        }
        self.reset()

    def _load_model(self, path):
        net = VesselModel()
        net.load_state_dict(torch.load(path))
        net.eval()
        return net

    def set_rado_params(self, wave_height):
        self.param["wave_height"] = wave_height
        self.state[6] = wave_height

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, reset_step=True):
        self.steps = 0
        self.mode = 0 #np.random.randint(0, 4)
        # TODO: 固定参数空间
        # TODO: 例如只改变浪向

        wind_velocity = 0 # np.random.uniform(0, 13.8)
        wind_direction = 0 # int(np.random.uniform(0, 360) / 10) * 10
        flow_velocity = 0 # np.random.uniform(0.25, 0.78)
        flow_direction = 0 # int(np.random.uniform(25, 75) / 10) * 10
        depth = 500
        wave_height = self.param["wave_height"] # np.random.uniform(0, 6)
        wave_direction = 180 # int(np.random.uniform(0, 360) / 10) * 10
        wave_period = 8 # np.random.uniform(2.1, 13)

        vessel_velocity = np.random.uniform(0, 30)
        vessel_direction = int(np.random.uniform(0, 360) / 10) * 10
        self.state = [
            wind_direction,
            wind_velocity,
            flow_direction,
            flow_velocity,
            depth,
            wave_direction,
            wave_height,
            wave_period,
            vessel_direction,
            vessel_velocity
        ]
        return np.array([0, self.state[-1], self.state[-2]])

    def step(self, action: list):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.steps += 1

        self.state[-2] += action[0]
        self.state[-1] += action[1]
        self.state[-2] %= 360
        if self.state[-1] < 0:
            self.state[-1] = 0
        elif self.state[-1] > 30:
            self.state[-1] = 30

        inputs = torch.tensor(self.state, dtype=torch.float32)
        score = self.model(inputs).detach().item()

        reward = -0.01 * (0.5 * abs(action[0]) + 0.5 * abs(action[1]))
        if self.steps % self.reward_freq == 0:
            inputs = torch.tensor(self.state, dtype=torch.float32)
            reward += -self.model(inputs).detach().item()

        done = False
        if self.steps > self.max_steps:
            done = True

        info = {}

        return np.array([score, self.state[-1], self.state[-2]]), reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass