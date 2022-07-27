"""
2d环境测试
"""

from pyrado.environments.one_step.two_dim_gaussian import TwoDimGaussian

if __name__ == '__main__':
    env = TwoDimGaussian()

    d1 = 0
    d2 = 0
    iter = 1000

    for _ in range(iter):
        obs = env.reset()
        while True:
            act = 1
            _obs, rew, done, info = env.step(act)
            for i in range(len(_obs)):
                if i % 2 == 0:
                    d1 += _obs[i]
                else:
                    d2 += _obs[i]
            print("Obs: {0}".format(_obs))
            obs = _obs
            if done:
                break

    print("d1: {0}, d2: {1}".format(d1 / (iter * 4), d2 / (iter * 4)))