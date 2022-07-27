"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

#env_name = "CartPole-v1"
env_name = "LunarLanderContinuous-v2"
#env = gym.make(env_name)
env = gym.make(env_name)


def train_expert():
    print("Training a expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=100,
        n_steps=64,
    )
    expert.learn(100000)  # Note: change this to 100000 to trian a decent expert.
    return expert


def sample_expert_transitions():
    expert = train_expert()

    print("Sampling expert transitions.")
    rollouts = rollout.generate_trajectories(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(n_timesteps=None, n_episodes=50),
    )
    return rollout.flatten_trajectories(rollouts)


transitions = sample_expert_transitions()
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    expert_data=transitions,
)

reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=10, render=True)
print(f"Reward before training: {reward}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=10)

reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=10, render=True)
print(f"Reward after training: {reward}")

bc_trainer.save_policy("./{0}_policy.pt".format(env_name))
