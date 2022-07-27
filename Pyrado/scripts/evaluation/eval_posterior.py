import pyrado
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment

if __name__ == '__main__':
    args = get_argparser().parse_args()
    # Get the experiment's directory to load from
    ex_dir = f"{pyrado.TEMP_DIR}/2dg/npdr/2022-04-14_20-12-36--observation_1"

    env_sim, policy, kwout = load_experiment(ex_dir, args)
    env_real = pyrado.load("env_real.pkl", ex_dir)
    prior = kwout["prior"]
    # posterior = kwout["posterior"]
    data_real = kwout["data_real"]
    posterior = pyrado.load(name=f"posterior.pt", load_dir=ex_dir)
    # 后验获取不到potential_fn
    ref_post_sam = pyrado.load(name=f"reference_posterior_samples.pt", load_dir=ex_dir, obj=None, verbose=True)
    true_param = pyrado.load(name=f"true_parameters.pt", load_dir=ex_dir, obj=None, verbose=True)

    posterior_samples = posterior.sample((1000,), x=data_real)
    print("env real: {0}".format(env_real))
    print("prior: {0}, prior samples: {1}".format(prior, prior.sample()))
    print("policy: {0}".format(policy))
    print("data real: {0}".format(data_real))
    print("posterior: {0}, posterior samples: {1}".format(posterior, posterior_samples))
    print("reference_posterior_samples: {0}".format(ref_post_sam))
    print("true_parameters: {0}".format(true_param))

    import numpy as np

    posterior_samples = np.array(posterior_samples)
    mean = [posterior_samples[:, i].mean() for i in range(5)]
    print("posterior mean: {0}".format(mean))
    # plot posterior samples
    from sbi import analysis

    _ = analysis.pairplot(posterior_samples, limits=[[-3, 3], [-3, 3], [-3, 3], [-3, 3], [-3, 3]],
                          figsize=(5, 5))
