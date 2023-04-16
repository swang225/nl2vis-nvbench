import pandas as pd
from nl2vis_nvbench.model.nv_bert import nvBert
import torch
import torch.nn as nn
import os.path as osp
import pickle

from nl2vis_nvbench.common import Counter, read_pickle, write_pickle

import numpy as np
import random
import time
import math


def idx_to_words(mod, idx):

    itos = mod.label_vocab.get_itos()
    return np.array([itos[i] for i in idx])


def evaluate(model, iterator, mod):
    model.eval()

    correct = 0
    total = 0

    nl_correct = 0
    nl_total = 0

    wc_correct = 0
    wc_total = 0

    counter = Counter(total=len(iterator))
    counter.start()

    eval_res = []

    print("- start evaluating")
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0]
            trg = batch[1]
            source = batch[3]

            trg_len = trg.shape[-1]

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            prd = output.argmax(dim=1)

            res = prd == trg
            correct += res.sum().item()
            total += len(res)

            prd_words  = idx_to_words(mod, prd)
            trg_words = idx_to_words(mod, trg)

            incorrect_prd_words = prd_words[~res]
            incorrect_trg_words = trg_words[~res]

            correct_prd_words = prd_words[res]
            correct_trg_words = trg_words[res]

            cur_res = [
                source,
                " ".join(trg_words), " ".join(prd_words),
                " ".join(incorrect_trg_words), " ".join(incorrect_prd_words),
                " ".join(correct_trg_words), " ".join(correct_prd_words),
                res.sum().item(), len(res)
            ]
            eval_res.append(cur_res)

            counter.update()

            if batch[2][0]:
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

    eval_res_df = pd.DataFrame(
        data=dict(zip(
            [
                "source", "target", "predicted",
                "incorrect target", "incorrect prediction",
                "correct target", "correct prediction",
                "correct count", "total count"
            ],
            np.array(eval_res).transpose()
        ))
    )

    eval_res_df.to_csv("C:/Users/aphri/Documents/t0002/pycharm/data/ncnet/nv_bert_eval_res_df2.csv")

    print(f"final accuracy: {correct / total}")
    print(f"final accuracy (nl only): {nl_correct / nl_total}")
    print(f"final accuracy (with chart): {wc_correct / wc_total}")


if __name__ == '__main__':
    from nebula import root

    temp_dataset_path = "C:/Users/aphri/Documents/t0002/pycharm/data/ncnet/temp_data"
    batch_size = 1

    model = nvBert(
        temp_dataset_path=temp_dataset_path,
        batch_size=batch_size
    )
    # # load the current best model
    from nebula import root
    model.load_model(osp.join(root(), "model", "nv_bert", "result", "model_best.pt"))

    evaluate(model.model, model.test_dl, model)
