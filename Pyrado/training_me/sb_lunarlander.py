from pyrado.environments.pysim.lunar_lander import LunarLanderRado
from pyrado.policies.feed_forward.dummy import DummyPolicy
from pyrado.sampling.rollout import after_rollout_query, rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt

if __name__ == '__main__':
    env = LunarLanderRado()
    policy = DummyPolicy(env.spec)

    done, param, state = False, {"me": 20, "se": 1}, None
    while not done:
        ro = rollout(
            env,
            policy,
            max_steps=5,
            render_mode=RenderMode(text=False, video=False, render=True),
            eval=True,
            reset_kwargs=dict(domain_param=param, init_state=state),
        )
        # print_cbt(f"State: {ro.get_data_values('state')}, length: {ro.length}", "g", bright=True)
        print_cbt(f"Observation: {ro.get_data_values('observations')}, length: {ro.length}", "g", bright=True)
        print_cbt(f"Action: {ro.get_data_values('actions')}", "g", bright=True)
        print_cbt(f"Reward: {ro.get_data_values('rewards')}", "g", bright=True)
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, state, param = after_rollout_query(env, policy, ro)
