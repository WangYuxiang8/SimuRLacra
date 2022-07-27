"""
Evaluate pickle file to see what is it.
"""
import pickle

if __name__ == '__main__':
    # ../../data/temp/2dg/npdr/2022-04-07_17-33-36--observation_1/env_sim.pkl
    # ../../data/perma/evaluation/qq_chrip_10to0Hz_-1.5V_250Hz_10s/rollout_real_2021-04-14_18-56-57.pkl
    filename = "../../data/temp/qcp-su/npdr_qq-sub/2022-04-08_22-38-01--numsegs-20/iter_0_ml_domain_param.pkl"
    with open(filename, "rb") as reader:
        data = pickle.load(reader)
        print(data)
