import os.path as osp
import sbi.utils as sbiutils
import torch
import torch as to
from sbi.inference import SNPE_C, SNLE
from sbi.utils.get_nn_models import likelihood_nn

import pyrado
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.environments.one_step.two_dim_gaussian import TwoDimGaussian
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_forward.dummy import IdlePolicy
from pyrado.sampling.sbi_embeddings import LastStepEmbedding
from pyrado.sampling.sbi_rollout_sampler import SimRolloutSamplerForSBI, RolloutSamplerForSBI
from pyrado.utils.argparser import get_argparser
from sbi import analysis as analysis
from sbi.inference import prepare_for_sbi, simulate_for_sbi
import time

# 超参数
dp_mapping = {0: "m_1", 1: "m_2", 2: "s_1", 3: "s_2", 4: "rho"}
num_segments = 20
vi_parameters = {}

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    pyrado.set_seed(args.seed, verbose=True)

    env = TwoDimGaussian()  # 环境，基于gym接口
    policy = IdlePolicy(env.spec)  # 策略，这里是固定策略
    prior = sbiutils.BoxUniform(low=-3 * to.ones((5,)), high=3 * to.ones((5,)))  # 先验

    # 定义embedding，用于将rollouts转化为固定的embedding。
    embedding = LastStepEmbedding(env.spec, RolloutSamplerForSBI.get_dim_data(env.spec))

    # 创建基于sbi的模拟器和先验
    rollout_sampler = SimRolloutSamplerForSBI(
        env,
        policy,
        dp_mapping,
        embedding,
        num_segments=num_segments
    )
    simulator, prior = prepare_for_sbi(rollout_sampler, prior)

    # 定义推断算法
    """
    density_estimator_fun = likelihood_nn(
        model='maf',
        hidden_features=50
    )
    """
    inference = SNLE(prior=prior)

    # 多轮推断，Multi-round inference
    num_rounds = 3

    posteriors = []
    proposal = prior

    # 设置固定的params，传入模拟器，并输出一个观测
    theta = torch.tensor([-1, 1, 0, 0, 0])
    x_o = simulator(theta)
    print("Real data: {0}".format(x_o))

    for i in range(num_rounds):

        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=500)
        # print("Step {0}, params: {1}, data: {2}".format(i, theta, x))

        # 如果build_posterior改为mcmc方式，第二轮训练时就会出现错误： File
        # "/Users/4paradigm/anaconda3/envs/pyrado/lib/python3.7/site-packages/sbi/inference/snpe/snpe_c.py",
        # line 163, in train isinstance(proposal.posterior_estimator._distribution, mdn) AttributeError:
        # 'MCMCPosterior' object has no attribute 'posterior_estimator'
        _ = inference.append_simulations(theta, x).train()

        # 通过神经密度估计函数（神经网络）生成一个后验采样器，这里可以指定采样的方式，三种 - rejection/mcmc/vi
        # 一般rejection效率可能比较低，代码会提示说拒绝样本过多，推荐使用mcmc
        posterior = inference.build_posterior(sample_with="vi", vi_parameters=vi_parameters)
        posterior = posterior.set_default_x(x_o)

        # vi后验器需要训练一个后验分布q
        posterior.train(**vi_parameters)
        proposal = posterior.set_default_x(x_o)

        posteriors.append(posterior)

        start = time.time()
        posterior_samples = posterior.sample((100,), x=x_o)
        posterior_samples_mean = [posterior_samples[:, i].mean() for i in range(len(posterior_samples[0]))]
        end = time.time()
        print("Posterior samples: {0}, time cost: {1}".format(posterior_samples_mean, end - start))

        # 保存后验,出错AttributeError: Can't pickle local object 'make_object_deepcopy_compatible.<locals>.__deepcopy__'
        posterior._optimizer = None
        posterior.__deepcopy__ = None
        posterior._q_build_fn = None
        posterior._q.__deepcopy__ = None
        # pyrado.save(posterior, f"snvi_posterior_{0}.pt".format(i), "./2dg")
        torch.save(posterior, f"./2dg/snvi_posterior_{i}.pt")

    # torch.save(posteriors[-1], f"./2dg/snvi_posterior_{0}.pt".format(i))


    # 验证
    # 1. 验证环境，基于sbi设置的环境，输入domain params，输出rollouts
    # 2. 验证后验，与真实分布之间的差距
