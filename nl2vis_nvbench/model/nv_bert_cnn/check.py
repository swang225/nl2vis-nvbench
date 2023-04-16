from nl2vis_nvbench import root
from nl2vis_nvbench.common import read_pickle
import os.path as osp

res = read_pickle(osp.join(root(), "model", "nv_bert_cnn", "result", "train_results.pkl"))
