from nebula import root
from nebula.common import read_pickle
import os.path as osp

res = read_pickle(osp.join(root(), "model", "nv_bert_cnn", "result", "train_results.pkl"))
