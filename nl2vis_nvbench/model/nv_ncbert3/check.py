from nl2vis_nvbench import root
from nl2vis_nvbench.common import read_pickle
import os.path as osp

res = read_pickle(osp.join(root(), "model", "ncnet_no_af", "result", "train_results.pkl"))
