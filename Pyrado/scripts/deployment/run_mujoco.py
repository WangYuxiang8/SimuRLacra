import pyrado
from pyrado.domain_randomization.utils import wrap_like_other_env
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSim
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.environments.quanser.quanser_ball_balancer import QBallBalancerReal
from pyrado.environments.quanser.quanser_cartpole import QCartPoleReal
from pyrado.environments.quanser.quanser_qube import QQubeSwingUpReal
from pyrado.environments.mujoco.openai_half_cheetah import HalfCheetahSim
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import after_rollout_query, rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment(hparam_list=args.show_hparams) if args.dir is None else args.dir

    # Load the policy (trained in simulation) and the environment (for constructing the real-world counterpart)
    env_sim, policy, _ = load_experiment(ex_dir, args)

    # Detect the correct real-world counterpart and create it
    env_hparam = dict()
    env_real = HalfCheetahSim(**env_hparam)

    # Wrap the real environment in the same way as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run on device
    done = False
    print_cbt("Running loaded policy ...", "c", bright=True)
    while not done:
        ro = rollout(
            env_real,
            policy,
            eval=True,
            record_dts=True,
            render_mode=RenderMode(text=args.verbose, video=args.animation),
        )
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, _, _ = after_rollout_query(env_real, policy, ro)
