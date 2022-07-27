import torch
import pyrado
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi import analysis as analysis

_ = torch.manual_seed(0)

num_dim = 3
prior = utils.BoxUniform(low=-2 * torch.ones(num_dim),
                         high=2 * torch.ones(num_dim))


def linear_gaussian(theta):
    return theta + 1.0 + torch.randn_like(theta) * 0.1


inference = SNPE(prior=prior)

simulator, prior = prepare_for_sbi(linear_gaussian, prior)

num_rounds = 2
x_o = torch.zeros(3, )

posteriors = []
proposal = prior

for _ in range(num_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=500)
    density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
    posterior = inference.build_posterior(density_estimator)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(x_o)

posterior_samples = posterior.sample((10000,), x=x_o)

# plot posterior samples
_ = analysis.pairplot(posterior_samples, limits=[[-2, 2], [-2, 2], [-2, 2]],
                      figsize=(5, 5))

pyrado.save(posterior, "posterior.pt", "./")