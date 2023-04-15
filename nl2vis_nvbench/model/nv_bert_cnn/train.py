import torch
import torch.nn as nn
import os.path as osp
import os
import pickle

from nebula.model.nv_bert_cnn import nvBertCNN
from nebula.common import Counter, read_pickle, write_pickle

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

        optimizer.zero_grad()

        # notice how the training is done here
        # if the label is [1, 2, 3, 4]
        # we feed [1, 2, 3] into the model
        # get the prob vector for [2, 3, 4] (each of size 826)
        # and do cross entropy loss check for [2, 3, 4]
        output, _ = model(src, trg[:, :-1])

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

    counter = Counter(total=len(iterator))
    counter.start()

    print("- start evaluating")
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0]
            trg = batch[1]

            trg_len = trg.shape[-1]

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

            counter.update()

            del src
            del trg
            del output
            del output_dim
            del loss

    print("- end evaluating")
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run_train(
        model,
        opt,
        seed=1234,
        testing=False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    print("initialize weights")
    model.model.apply(initialize_weights)

    LEARNING_RATE = opt.learning_rate
    optimizer = torch.optim.Adam(model.model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=model.TRG_PAD_IDX)

    N_EPOCHS = opt.epoch
    CLIP = 1

    train_loss_list, valid_loss_list = list(), list()
    best_valid_loss = float('inf')

    train_dl = model.train_dl if not testing else model.train_dl_small

    print("start training")
    for epoch in range(N_EPOCHS):

        print(f"epoch: {epoch}")

        epoch_model_path = osp.join(opt.output_dir, 'model_' + str(epoch + 1) + '.pt')
        res_path = osp.join(opt.output_dir, 'train_results.pkl')
        if osp.exists(epoch_model_path):
            print(f"model already exist")
            model.load_model(epoch_model_path)

            res = read_pickle(res_path)
            train_loss_list = res["train_loss"]
            valid_loss_list = res["valid_loss"]

            continue

        start_time = time.time()

        train_loss = train(model.model, train_dl, optimizer, criterion, CLIP)
        valid_loss = evaluate(model.model, model.validation_dl, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # save the best trained model
        if valid_loss < best_valid_loss:
            print(f"saving best models with validation loss: {valid_loss}")
            best_valid_loss = valid_loss
            torch.save(
                model.model.state_dict(),
                str(osp.join(opt.output_dir, 'model_best.pt'))
            )

        # save model on each epoch
        print(f"saving mode for epoch: {epoch + 1}")
        torch.save(
            model.model.state_dict(),
            epoch_model_path
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
        write_pickle(res_path, res)


if __name__ == '__main__':
    from argparse import Namespace
    from nebula import root

    opt = Namespace()
    opt.data_dir = osp.join(root(), "data", "nvbench", "dataset", "dataset_final")
    opt.db_info = osp.join(root(), "data", "nvbench", "dataset", "database_information.csv")
    opt.output_dir = "/home/ubuntu/data/ncnet/output_models_bert_cnn_test1"
    opt.temp_dataset_path = "/home/ubuntu/data/ncnet/temp_data"
    opt.epoch = 5
    opt.learning_rate = 0.0005
    opt.batch_size = 32
    opt.max_input_length = 128

    if not osp.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    model = nvBertCNN(
        temp_dataset_path=opt.temp_dataset_path,
        batch_size=opt.batch_size
    )
    run_train(model=model, opt=opt, testing=True)
