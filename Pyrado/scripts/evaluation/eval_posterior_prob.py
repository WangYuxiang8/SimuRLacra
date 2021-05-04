# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt, nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Script to evaluate a posterior obtained using the sbi package.
By default (args.iter = -1), the most recent iteration is evaluated.
"""
import os

import seaborn as sns
import torch as to
from matplotlib import pyplot as plt

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.meta.bayessim import BayesSim
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.algorithms.meta.sbi_base import SBIBase
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.distribution import (
    draw_posterior_distr_1d,
    draw_posterior_distr_2d,
    draw_posterior_distr_pairwise_heatmap,
    draw_posterior_distr_pairwise_scatter,
)
from pyrado.plotting.utils import num_rows_cols_from_length
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt
from pyrado.utils.ordering import remove_none_from_list


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    args = parser.parse_args()
    plt.rc("text", usetex=args.use_tex)
    if not isinstance(args.num_samples, int) or args.num_samples < 1:
        raise pyrado.ValueErr(given=args.num_samples, ge_constraint="1")

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment(hparam_list=args.show_hparams) if args.dir is None else args.dir

    # Load the algorithm
    algo = Algorithm.load_snapshot(ex_dir)
    if not isinstance(algo, (NPDR, BayesSim)):
        raise pyrado.TypeErr(given=algo, expected_type=(NPDR, BayesSim))

    # Load the environments, the policy, and the posterior
    env_sim, policy, kwout = load_experiment(ex_dir, args)
    env_real = pyrado.load("env_real.pkl", ex_dir)
    prior = kwout["prior"]
    posterior = kwout["posterior"]
    data_real = kwout["data_real"]

    if args.mode.lower() == "evolution-round" and args.iter == -1:
        args.iter = algo.curr_iter
        print_cbt("Set the evaluation iteration to the latest iteration of the algorithm.", "y")

    # Load the sequence of posteriors if desired
    if args.mode.lower() == "evolution-iter":
        posterior = [SBIBase.load_posterior(ex_dir, idx_iter=i, verbose=True) for i in range(algo.max_iter)]
        posterior = remove_none_from_list(posterior)  # in case the algorithm terminated early
    elif args.mode.lower() == "evolution-round":
        posterior = [SBIBase.load_posterior(ex_dir, idx_round=i, verbose=True) for i in range(algo.num_sbi_rounds)]
        posterior = remove_none_from_list(posterior)  # in case the algorithm terminated early

    if "evolution" in args.mode.lower() and data_real.shape[0] > len(posterior):
        print_cbt(
            f"Found {data_real.shape[0]} data sets but {len(posterior)} posteriors. Truncated the superfluous data.",
            "y",
        )
        data_real = data_real[: len(posterior), :]

    if args.mode.lower() == "evolution-round":
        # Artificially repeat the data (which was the same for every round) to later be able to use the same code
        data_real = data_real.repeat(len(posterior), 1)
        assert data_real.shape[0] == len(posterior)

    # Select the domain parameters to plot
    if len(algo.dp_mapping) == 1:
        idcs_dp = (0,)
    elif len(algo.dp_mapping) == 2:
        idcs_dp = (0, 1)
    elif args.idcs is not None:
        idcs_dp = args.idcs
    elif "pairwise" not in args.mode.lower():
        usr_inp = input(
            f"Found the domain parameter mapping {algo.dp_mapping}. Select 1 or 2 domain parameter by index "
            f"to be plotted (format: separated by a whitespace):\n"
        )
        idcs_dp = tuple(map(int, usr_inp.split()))
    else:
        # We are using all dims for pairwise plot. We only set this here to jump over the len(idcs_dp) == 1 case later.
        idcs_dp = tuple(i for i in algo.dp_mapping.keys())

    # Set the condition if necessary (always necessary for the pairwise plots)
    if (2 >= len(algo.dp_mapping) == len(idcs_dp)) and "pairwise" not in args.mode.lower():
        # No condition necessary since dim(posterior) = dim(grid)
        condition = None
    else:
        if args.mode.lower() == "pairwise-scatter":
            # Use the latest posterior to sample domain parameters to obtain a condition
            domain_params, log_probs = SBIBase.eval_posterior(
                posterior[-1] if args.mode.lower() == "evolution-iter" else posterior,
                data_real,
                args.num_samples,
                normalize_posterior=False,  # not necessary here
                subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=args.use_mcmc),
            )
            domain_params_posterior = domain_params.reshape(1, -1, domain_params.shape[-1]).squeeze()
            # Reconstruct ground truth domain parameters if they exist
            if typed_env(env_real, DomainRandWrapperBuffer):
                dp_gt = to.stack(
                    [to.stack(list(d.values())) for d in env_real.randomizer.get_params(-1, "list", "torch")]
                )
            elif isinstance(env_real, (SimEnv, EnvWrapper)):
                dp_gt = to.tensor([env_real.domain_param[v] for v in algo.dp_mapping.values()])
                dp_gt = to.atleast_2d(dp_gt)
            else:
                dp_gt = None
            if isinstance(env_sim, (SimEnv, EnvWrapper)):
                dp_nom = to.tensor([env_sim.domain_param[v] for v in algo.dp_mapping.values()])
                dp_nom = to.atleast_2d(dp_nom)
            else:
                dp_nom = None
            reference_posterior_samples = None
            for dirpath, dirnames, filenames in os.walk(ex_dir):
                for f in filenames:
                    if "reference_posterior_samples" in f:
                        reference_posterior_samples = pyrado.load(f, ex_dir)
            condition = None
        else:
            # Get the most likely domain parameters per iteration
            condition, _ = SBIBase.get_ml_posterior_samples(
                algo.dp_mapping,
                posterior[-1] if "evolution" in args.mode.lower() else posterior,
                data_real,
                args.num_samples,
                num_ml_samples=1,
                normalize_posterior=False,
                subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=args.use_mcmc),
                return_as_tensor=True,
            )

    # Plot the posterior distribution, the true parameters / their distribution
    if len(idcs_dp) == 1:
        fig, axs = plt.subplots(figsize=(14, 7), tight_layout=True)
        _ = draw_posterior_distr_1d(
            axs,
            posterior[-1] if "evolution" in args.mode.lower() else posterior,  # ignore plotting mode
            data_real,
            algo.dp_mapping,
            idcs_dp,
            prior,
            env_real,
            condition,
            normalize_posterior=args.normalize,
            rescale_posterior=args.rescale,
            # x_label=None,
            # y_label=None,
        )

    else:
        if "pairwise" in args.mode.lower():
            if args.layout == "inside":
                num_rows, num_cols = len(algo.dp_mapping), len(algo.dp_mapping)
            elif args.layout == "outside":
                num_rows, num_cols = len(algo.dp_mapping) + 1, len(algo.dp_mapping) + 1
            else:
                raise pyrado.ValueErr(given=args.mode.lower(), eq_constraint="inside or outside")

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 16), tight_layout=False)
            if args.mode.lower() == "pairwise-scatter":
                axis_limits = to.stack((prior.base_dist.low, prior.base_dist.high))
                dp_samples = [domain_params_posterior]
                legend_labels = ["Sim"]
                c_palette = sns.color_palette()[1:]
                if dp_gt is not None:
                    legend_labels.append("Real")
                    c_palette.insert(1, (0.0, 0.0, 0.0))
                    dp_samples.append(dp_gt)
                if dp_nom is not None:
                    legend_labels.append("Nominal")
                    c_palette.insert(2, sns.color_palette()[0])
                    dp_samples.append(dp_nom)
                if reference_posterior_samples is not None:
                    # If there are more reference samples than num_samples, short the reference samples
                    if args.num_samples < reference_posterior_samples.shape[0]:
                        randperm = to.randperm(reference_posterior_samples.shape[0])
                        reference_posterior_samples = reference_posterior_samples[randperm, :]
                        reference_posterior_samples = reference_posterior_samples[: args.num_samples, :]
                    dp_samples.append(reference_posterior_samples)
                _ = draw_posterior_distr_pairwise_scatter(
                    axs,
                    dp_samples,
                    algo.dp_mapping,
                    marginal_layout=args.layout,
                    legend_labels=legend_labels,
                    color_palette=c_palette,
                    set_alpha=0.2,
                    axis_limits=axis_limits,
                )
            else:
                _ = draw_posterior_distr_pairwise_heatmap(
                    axs,
                    posterior,
                    data_real,
                    algo.dp_mapping,
                    condition,
                    prior,
                    env_real,
                    marginal_layout=args.layout,
                    grid_res=100,
                    normalize_posterior=args.normalize,
                    rescale_posterior=args.rescale,
                    # x_labels=None,
                    # y_labels=None,
                )

        else:
            if args.mode.lower() == "joint":
                num_rows, num_cols = 1, 1
            elif args.mode.lower() in ["separate", "evolution-iter", "evolution-round"]:
                num_rows, num_cols = num_rows_cols_from_length(data_real.shape[0])
            else:
                raise pyrado.ValueErr(
                    given=args.mode,
                    eq_constraint="joint, separate, evolution-iter, evolution-round, pairwise-density, "
                    "or pairwise-scatter",
                )

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 7), tight_layout=True)
            draw_posterior_distr_2d(
                axs,
                args.mode,
                posterior,
                data_real,
                algo.dp_mapping,
                idcs_dp,
                prior,
                env_real,
                condition,
                grid_res=200,
                normalize_posterior=args.normalize,
                rescale_posterior=args.rescale,
                add_sep_colorbar=False,
                x_label=None,
                y_label=None,
            )

    if args.save:
        for fmt in ["pdf", "pgf", "png"]:
            os.makedirs(os.path.join(ex_dir, "plots"), exist_ok=True)
            rnd = f"_round_{args.round}" if args.round != -1 else ""
            fig.savefig(
                os.path.join(ex_dir, "plots", f"posterior_prob_iter_{args.iter}{rnd}_{args.mode}.{fmt}"),
                dpi=500,
            )

    plt.show()