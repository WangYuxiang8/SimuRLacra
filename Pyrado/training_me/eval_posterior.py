import pyrado
import numpy as np
import time
import sbi.utils as sbiutils
import torch
from pyrado.environments.one_step.archery import Archery
from pyrado.environments.pysim.lunar_lander import LunarLanderRado
from pyrado.environments.pysim.vessel_rado import VesselRado
from pyrado.policies.feed_forward.dummy import DummyPolicy
from pyrado.sampling.sbi_rollout_sampler import SimRolloutSamplerForSBI
from sbi.inference import prepare_for_sbi
import time
from pyrado.utils.sbi import create_embedding
from pyrado.sampling.sbi_embeddings import LastStepEmbedding, BayesSimEmbedding
from pyrado.sampling.sbi_rollout_sampler import SimRolloutSamplerForSBI, RolloutSamplerForSBI
from pyrado.environments.mujoco.openai_half_cheetah import HalfCheetahSim
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment


def get_mujoco_true_data(name: str):
    parser = get_argparser()
    args = parser.parse_args()
    if name == "cth":
        dp_mapping = {0: "reset_noise_halfspan", 1: "total_mass", 2: "tangential_friction_coeff",
                      3: "torsional_friction_coeff",
                      4: "rolling_friction_coeff"}
        num_segments = 20
        prior_min = [0., 10., 0., 0., 0.]
        prior_max = [1, 20., 1., 1., 1.]
        true_theta = torch.tensor([0., 14., 0.4, 0.1, 0.1])

        env = HalfCheetahSim()  # 环境，基于gym接口
        ex_dir = ask_for_experiment(hparam_list=args.show_hparams) if args.dir is None else args.dir
        env_sim, policy, _ = load_experiment(ex_dir, args)

        prior = sbiutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

        # 定义embedding，用于将rollouts转化为固定的embedding。
        embedding_hparam = dict(downsampling_factor=1)
        embedding = create_embedding(BayesSimEmbedding.name, env.spec, **embedding_hparam)

        # 创建基于sbi的模拟器和先验
        rollout_sampler = SimRolloutSamplerForSBI(
            env,
            policy,
            dp_mapping,
            embedding,
            num_segments=num_segments
        )
        simulator, prior = prepare_for_sbi(rollout_sampler, prior)

        # 设置固定的params，传入模拟器，并输出一个观测
        x_o = simulator(true_theta)

        return true_theta, x_o


def get_lunar_lander_true_data():
    num_segments = 20
    dp_mapping = {0: "me", 1: "se"}

    env = LunarLanderRado()  # 环境，基于gym接口
    policy = DummyPolicy(env.spec)  # 策略，这里是固定策略
    prior_min = [8., 0.3]
    prior_max = [20, 1.]
    prior = sbiutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

    # 定义embedding，用于将rollouts转化为固定的embedding。
    embedding_hparam = dict(downsampling_factor=1)
    embedding = create_embedding(BayesSimEmbedding.name, env.spec, **embedding_hparam)

    # 创建基于sbi的模拟器和先验
    rollout_sampler = SimRolloutSamplerForSBI(
        env,
        policy,
        dp_mapping,
        embedding,
        num_segments=num_segments
    )
    simulator, prior = prepare_for_sbi(rollout_sampler, prior)

    theta = torch.tensor([18.0, 0.6])
    x_o = simulator(theta)
    print("Parameters: {0}, Real data: {1}".format(theta, x_o))

    return theta, x_o


def get_archery_true_data():
    num_segments = 20
    dp_mapping = {0: "wf"}

    env = Archery()  # 环境，基于gym接口
    policy = DummyPolicy(env.spec)  # 策略，这里是固定策略
    prior_min = [-5.]
    prior_max = [5.]
    prior = sbiutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

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

    x = []
    theta = torch.tensor([4.98])
    for r in range(100):
        x.append(simulator(theta).detach().item())
    x_o = np.array(x).mean()
    print("Parameters: {0}, Real data: {1}".format(theta, x_o))

    return theta, x_o


def get_vessel_true_data():
    num_segments = 20
    dp_mapping = {0: "wave_height"}

    env = VesselRado()  # 环境，基于gym接口
    policy = DummyPolicy(env.spec)  # 策略，这里是固定策略
    prior_min = [0.]
    prior_max = [6.]
    prior = sbiutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

    # 定义embedding，用于将rollouts转化为固定的embedding。
    embedding_hparam = dict(downsampling_factor=1)
    embedding = create_embedding(BayesSimEmbedding.name, env.spec, **embedding_hparam)

    # 创建基于sbi的模拟器和先验
    rollout_sampler = SimRolloutSamplerForSBI(
        env,
        policy,
        dp_mapping,
        embedding,
        num_segments=num_segments
    )
    simulator, prior = prepare_for_sbi(rollout_sampler, prior)

    theta = torch.tensor([5.0])
    x_o = simulator(theta)
    print("Parameters: {0}, Real data: {1}".format(theta, x_o))

    return theta, x_o


def get_true_data(env_name: str = "2dg"):
    if env_name == "2dg":
        pass
    elif env_name == "archery":
        return get_archery_true_data()
    elif env_name == "lunar_lander":
        return get_lunar_lander_true_data()
    elif env_name == "vessel":
        return get_vessel_true_data()
    elif env_name in ["cth"]:
        return get_mujoco_true_data(env_name)


def eval_vi_posterior(ex_dir, file_name, sample_num, x_o):
    posterior = pyrado.load(name=file_name, load_dir=ex_dir)
    posterior = posterior.set_default_x(x_o)
    posterior.train()

    start = time.time()
    samples = posterior.sample((sample_num,), x=x_o)
    posterior_samples = np.array(samples)
    mean = [posterior_samples[:, i].mean() for i in range(len(posterior_samples[0]))]
    end = time.time()
    print("Posterior mean: {0}, time cost: {1}".format(mean, end - start))

    return posterior_samples


def eval_rejection_posterior(ex_dir, file_name, sample_num, x_o):
    posterior = pyrado.load(name=file_name, load_dir=ex_dir)

    start = time.time()
    samples = posterior.sample((sample_num,), x=x_o)
    posterior_samples = np.array(samples)
    mean = [posterior_samples[:, i].mean() for i in range(len(posterior_samples[0]))]
    end = time.time()
    print("Posterior mean: {0}, time cost: {1}".format(mean, end - start))

    return posterior_samples


if __name__ == '__main__':
    ex_dir = "/training_me/lunar_lander"
    sample_num = 100000
    name = "cth"
    _, x_o = get_true_data(name)
    x_o_path = "/Users/joseph/workspace/SimuRLacra/Pyrado/posterior_data/" + name + "/true_x.pt"
    print(x_o)
    torch.save(x_o, x_o_path)
    #eval_vi_posterior(ex_dir, f"snvi_posterior_2.pt", sample_num, x_o)
    #eval_rejection_posterior(ex_dir, f"rejection_posterior_2.pt", sample_num, x_o)
