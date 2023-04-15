import torch
import torch.nn as nn
import os
import os.path as osp
import pickle

from nebula.model.nv_ncbert3 import nvncBert3
from nebula.common import Counter

import numpy as np
import random
import time
import math


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    epoch_count = 0

    counter = Counter(total=len(iterator))
    counter.start()
    print("- start training")
    for i, batch in enumerate(iterator):
        src = batch[0]
        trg = batch[1]
        tok_types = batch[2]
        bert_src = batch[3]

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1], tok_types, bert_src)

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        epoch_count += 1

        counter.update()
        print(f"current loss: {epoch_loss / epoch_count}")

        del src
        del trg
        del output
        del output_dim
        del loss

    print("- end training")
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0]
            trg = batch[1]
            tok_types = batch[2]
            bert_src = batch[3]

            output, _ = model(src, trg[:, :-1], tok_types, bert_src)

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run_train(
        opt,
        seed=1234,
        testing=False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    m1 = nvncBert3(batch_size=opt.batch_size)

    print("initialize weights")
    m1.ncNet.apply(initialize_weights)

    LEARNING_RATE = opt.learning_rate
    optimizer = torch.optim.Adam(m1.ncNet.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=m1.TRG_PAD_IDX)

    N_EPOCHS = opt.epoch
    CLIP = 1

    train_loss_list, valid_loss_list = list(), list()
    best_valid_loss = float('inf')

    train_iterator = m1.train_iterator if not testing else m1.train_iterator_small

    print("start training")
    for epoch in range(N_EPOCHS):

        print(f"epoch: {epoch}")

        start_time = time.time()

        train_loss = train(m1.ncNet, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(m1.ncNet, m1.valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # save the best trained model
        if valid_loss < best_valid_loss:
            print(f"saving best models with validation loss: {valid_loss}")
            best_valid_loss = valid_loss
            torch.save(
                m1.ncNet.state_dict(),
                str(osp.join(opt.output_dir, 'model_best.pt'))
            )

        # save model on each epoch
        print(f"saving mode for epoch: {epoch + 1}")
        torch.save(
            m1.ncNet.state_dict(),
            str(osp.join(opt.output_dir, 'model_' + str(epoch + 1) + '.pt'))
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        res = {
            "epoch": epoch + 1,
            "train_loss": train_loss_list,
            "valid_loss": valid_loss_list,
        }

        with open(str(osp.join(opt.output_dir, 'train_results.pkl')), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    from argparse import Namespace
    from nebula import root

    opt = Namespace()
    opt.data_dir = osp.join(root(), "data", "nvbench", "dataset", "dataset_final")
    opt.db_info = osp.join(root(), "data", "nvbench", "dataset", "database_information.csv")
    opt.output_dir = "/home/ubuntu/data/ncnet/output_models_nvncbert3"
    opt.epoch = 5
    opt.learning_rate = 0.0005
    opt.batch_size = 32
    opt.max_input_length = 128

    if not osp.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    run_train(opt=opt, testing=False)
