import numpy as np
import torch

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# sbi
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer

# remove top and right axis from plots
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

from HH_helper_functions import syn_current
from HH_helper_functions import HHsimulator
from HH_helper_functions import calculate_summary_statistics


def run_HH_model(params):
    params = np.asarray(params)

    # input current, time step
    I, t_on, t_off, dt, t, A_soma = syn_current()

    t = np.arange(0, len(I), 1) * dt

    # initial voltage
    V0 = -70

    states = HHsimulator(V0, params.reshape(1, -1), dt, t, I)

    # print(states)
    return dict(data=states.reshape(-1), time=t, dt=dt, I=I.reshape(-1))


def simulation_wrapper(params):
    """
    Returns summary statistics from conductance values in `params`.

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    obs = run_HH_model(params)
    summstats = torch.as_tensor(calculate_summary_statistics(obs))
    return summstats


if __name__ == '__main__':
    # I表示电流列表，对应每个时刻的电流量大小
    # t_on表示开始输入电流的时刻
    # t_off表示结束输入电流的时刻
    # t表示时间点列表
    I, t_on, t_off, dt, t, A_soma = syn_current()
    print(I, t_on, t_off, dt, t, A_soma)

    # 构造三组参数，测试一下模拟器，输入参数，输出一个观测
    # three sets of (g_Na, g_K)
    params = np.array([[50., 1.], [4., 1.5], [20., 15.]])

    num_samples = len(params[:, 0])
    sim_samples = np.zeros((num_samples, len(I)))
    for i in range(num_samples):
        sim_samples[i, :] = run_HH_model(params=params[i, :])['data']

    # 可视化
    # colors for traces
    col_min = 2
    num_colors = num_samples + col_min
    cm1 = mpl.cm.Blues
    # 获取不同深浅的蓝色
    col1 = [cm1(1. * i / num_colors) for i in range(col_min, num_colors)]

    fig = plt.figure(figsize=(7, 5))
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax = plt.subplot(gs[0])
    for i in range(num_samples):
        plt.plot(t, sim_samples[i, :], color=col1[i], lw=2)
    plt.ylabel('voltage (mV)')
    ax.set_xticks([])
    ax.set_yticks([-80, -20, 40])

    ax = plt.subplot(gs[1])
    plt.plot(t, I * A_soma * 1e3, 'k', lw=2)
    plt.xlabel('time (ms)')
    plt.ylabel('input (nA)')

    ax.set_xticks([0, max(t) / 2, max(t)])
    ax.set_yticks([0, 1.1 * np.max(I * A_soma * 1e3)])
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    plt.show()

    # 定义先验分布
    prior_min = [.5, 1e-4]
    prior_max = [80., 15.]
    prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),
                                        high=torch.as_tensor(prior_max))

    # 定义推断算法
    # 这是一个简单的推断接口，用一行代码就可以执行完训练
    posterior = infer(simulation_wrapper, prior, method='SNPE',
                      num_simulations=300, num_workers=4)

    # 进行评估，定义一个真实参数，并根据这个真实参数生成样本
    # 用后验生成样本，并比较这两者之间的差距
    # true parameters and respective labels
    true_params = np.array([50., 5.])
    labels_params = [r'$g_{Na}$', r'$g_{K}$']
    observation_trace = run_HH_model(true_params)
    observation_summary_statistics = calculate_summary_statistics(observation_trace)

    fig = plt.figure(figsize=(7, 5))
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax = plt.subplot(gs[0])
    plt.plot(observation_trace['time'], observation_trace['data'])
    plt.ylabel('voltage (mV)')
    plt.title('observed data')
    plt.setp(ax, xticks=[], yticks=[-80, -20, 40])

    ax = plt.subplot(gs[1])
    plt.plot(observation_trace['time'], I * A_soma * 1e3, 'k', lw=2)
    plt.xlabel('time (ms)')
    plt.ylabel('input (nA)')

    ax.set_xticks([0, max(observation_trace['time']) / 2, max(observation_trace['time'])])
    ax.set_yticks([0, 1.1 * np.max(I * A_soma * 1e3)])
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

    samples = posterior.sample((10000,),
                               x=observation_summary_statistics)

    fig, axes = analysis.pairplot(samples,
                                  limits=[[.5, 80], [1e-4, 15.]],
                                  ticks=[[.5, 80], [1e-4, 15.]],
                                  figsize=(5, 5),
                                  points=true_params,
                                  points_offdiag={'markersize': 6},
                                  points_colors='r');

    # Draw a sample from the posterior and convert to numpy for plotting.
    posterior_sample = posterior.sample((1,),
                                        x=observation_summary_statistics).numpy()

    fig = plt.figure(figsize=(7, 5))

    # plot observation
    t = observation_trace['time']
    y_obs = observation_trace['data']
    plt.plot(t, y_obs, lw=2, label='observation')

    # simulate and plot samples
    x = run_HH_model(posterior_sample)
    plt.plot(t, x['data'], '--', lw=2, label='posterior sample')

    plt.xlabel('time (ms)')
    plt.ylabel('voltage (mV)')

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.3, 1),
              loc='upper right')

    ax.set_xticks([0, 60, 120])
    ax.set_yticks([-80, -20, 40]);
