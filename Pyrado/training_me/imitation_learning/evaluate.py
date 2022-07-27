from imitation.algorithms.bc import reconstruct_policy
from imitation.data.wrappers import RolloutInfoWrapper
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

env_name = "HalfCheetah-v2"
env = gym.make(env_name)
policy = reconstruct_policy("./HalfCheetah-v2_dagger.pt")

reward, _ = evaluate_policy(policy, env, n_eval_episodes=10, render=True)
print(f"Reward before training: {reward}")
