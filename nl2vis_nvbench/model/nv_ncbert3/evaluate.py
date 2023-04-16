from nl2vis_nvbench.model.nv_ncbert3 import nvncBert3
import torch
import torch.nn as nn
import os.path as osp
import pickle

from nl2vis_nvbench.common import Counter, read_pickle, write_pickle

import numpy as np
import random
import time
import math


def evaluate(model, iterator):
    model.eval()

    correct = 0
    total = 0

    nl_correct = 0
    nl_total = 0

    wc_correct = 0
    wc_total = 0

    counter = Counter(total=len(iterator))
    counter.start()

    print("- start evaluating")
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0]
            trg = batch[1]
            tok_types = batch[2]
            bert_src = batch[3]

            trg_len = trg.shape[-1]

            output, _ = model(src, trg[:, :-1], tok_types, bert_src)

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            res = output.argmax(dim=1) == trg
            correct += res.sum().item()
            total += len(res)

            counter.update()

            if batch[4][0]:
                wc_correct += res.sum().item()
                wc_total += len(res)
            else:
                nl_correct += res.sum().item()
                nl_total += len(res)

            if nl_total > 0 and wc_total > 0 and total > 0:
                print(f"current accuracy: {correct / total}")
                print(f"current accuracy (nl only): {nl_correct / nl_total}")
                print(f"current accuracy (with chart): {wc_correct / wc_total}")

            del src
            del trg
            del output
            del output_dim

    print(f"final accuracy: {correct / total}")
    print(f"final accuracy (nl only): {nl_correct / nl_total}")
    print(f"final accuracy (with chart): {wc_correct / wc_total}")


if __name__ == '__main__':
    from nebula import root

    temp_dataset_path = "C:/Users/aphri/Documents/t0002/pycharm/data/ncnet/temp_data"
    batch_size = 1

    model = nvncBert3(
        trained_model_path=osp.join(root(), "model", "nv_ncbert3", "result", "model_best_nvncbert.pt"),
        temp_dataset_path=temp_dataset_path,
        batch_size=batch_size
    )
    # # load the current best model
    # from nebula import root
    # model.load_model(osp.join(root(), "model", "nv_ncbert3", "result", "model_best_nvncbert.pt"))

    evaluate(model.ncNet, model.test_iterator)
