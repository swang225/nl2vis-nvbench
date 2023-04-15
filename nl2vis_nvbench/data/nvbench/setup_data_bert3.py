import torch
import pandas as pd
import os.path as osp
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ... import root
from transformers import BertTokenizer


class NvBenchDataset(Dataset):
    def __init__(
            self,
            data,
            bert_tokenizer,
            tokenizer,
            vocab,
            types,
    ):

        data = data.reset_index(drop=True)
        data.columns = ["source", "label", "token_type"]
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.types = types
        self.sos = vocab.get_stoi()['<sos>']
        self.eos = vocab.get_stoi()['<eos>']

        self.bert_tokenizer = bert_tokenizer

    def string_to_ids(self, input, vocab):

        stoi = vocab.get_stoi()
        res = [stoi[t] for t in self.tokenizer(input)]

        res = [self.sos] + res + [self.eos]
        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        cur_data = self.data.loc[idx]

        src = self.string_to_ids(cur_data["source"], self.vocab)
        lbl = self.string_to_ids(cur_data["label"], self.vocab)
        tok = self.string_to_ids(cur_data["token_type"], self.types)

        src_bert = self.bert_tokenizer(cur_data["source"])
        src_bert = src_bert["input_ids"]

        has_chart_type = '[T]' not in cur_data["source"]

        return src, lbl, tok, src_bert, has_chart_type


class Tokenizer:
    def __init__(
            self,
            tokenizer=None,
            lower=True,
    ):
        self._tok = get_tokenizer(tokenizer) \
            if tokenizer is not None \
            else self.tokenize
        self.lower = lower

    def tokenize(self, text):
        return text.split(' ')

    def __call__(self, data):
        return self._tok(
            data.lower()
            if self.lower
            else data
        )


class VocabBuilder:
    def __init__(
            self,
            tokenizer,
    ):

        self._tok = tokenizer

    def build_vocab_from_dfs(self, dfs):
        res = []

        for df in dfs:
            for col in df.columns:
                data = df[col].astype(str)
                cur_res = [self._tok(d.rstrip('\n')) for d in data]
                res += cur_res

        return build_vocab_from_iterator(
            res,
            specials=['<unk>', '<pad>', '<sos>', '<eos>'],
            min_freq=2
        )


class DataPadder:
    def __init__(self, vocab, types, bert_pad_id):
        self._pad_id = vocab.get_stoi()['<pad>']
        self._tok_pad_id = types.get_stoi()['<pad>']
        self._bert_pad_id = bert_pad_id

    @staticmethod
    def batch_data(data_zip, pad_id):
        data_len_max = max([len(s) for s in data_zip])
        data_batch = torch.tensor([
            data + [pad_id] * (data_len_max - len(data))
            for data
            in data_zip
        ])

        return data_batch

    def __call__(self, data):

        data_zip = list(zip(*data))

        src_batch = self.batch_data(data_zip[0], self._pad_id)
        lbl_batch = self.batch_data(data_zip[1], self._pad_id)
        tok_batch = self.batch_data(data_zip[2], self._tok_pad_id)

        src_bert_batch = self.batch_data(data_zip[3], self._bert_pad_id)

        has_chart_type_batch = data_zip[4]

        return src_batch, lbl_batch, tok_batch, src_bert_batch, has_chart_type_batch


def read_data(
        data_dir=osp.join(root(), "data", "nvbench", "dataset", "dataset_final"),
        db_info=osp.join(root(), "data", "nvbench", "dataset", "database_information.csv"),
):

    train_path = osp.join(data_dir, "train.csv")
    validation_path = osp.join(data_dir, "dev.csv")
    test_path = osp.join(data_dir, "test.csv")
    info_path = db_info

    train_df = pd.read_csv(train_path, keep_default_na=False, dtype=str)
    validation_df = pd.read_csv(validation_path, keep_default_na=False, dtype=str)
    test_df = pd.read_csv(test_path, keep_default_na=False, dtype=str)
    info_df = pd.read_csv(info_path, keep_default_na=False, dtype=str)

    test_df = test_df[["source", "labels", "token_types"]]
    validation_df = validation_df[["source", "labels", "token_types"]]
    train_df = train_df[["source", "labels", "token_types"]]
    info_df = info_df[["table", "column", "value"]]

    return test_df, validation_df, train_df, info_df


def get_bert_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    return tokenizer


def setup_data(
        data_dir=osp.join(root(), "data", "nvbench", "dataset", "dataset_final"),
        db_info=osp.join(root(), "data", "nvbench", "dataset", "database_information.csv"),
        batch_size=128,
        max_input_length=128
):

    test_df, validation_df, train_df, info_df = read_data(data_dir, db_info)

    bert_tokenizer = get_bert_tokenizer()
    tk = Tokenizer()
    vb = VocabBuilder(tokenizer=tk)

    vocab_dfs = [
        test_df[["source", "labels"]],
        validation_df[["source", "labels"]],
        train_df[["source", "labels"]],
        info_df[["table", "column", "value"]]
    ]
    vocab = vb.build_vocab_from_dfs(vocab_dfs)
    vocab.set_default_index(0)

    type_dfs = [
        test_df[["token_types"]],
        validation_df[["token_types"]],
        train_df[["token_types"]],
    ]
    types = vb.build_vocab_from_dfs(type_dfs)
    types.set_default_index(0)

    train_ds = NvBenchDataset(train_df, bert_tokenizer, tk, vocab, types)
    train_dl = DataLoader(train_ds, batch_size=batch_size, collate_fn=DataPadder(vocab, types, bert_pad_id=bert_tokenizer.pad_token_id))

    train_ds_small = NvBenchDataset(train_df.head(100), bert_tokenizer, tk, vocab, types)
    train_dl_small = DataLoader(train_ds_small, batch_size=batch_size, collate_fn=DataPadder(vocab, types, bert_pad_id=bert_tokenizer.pad_token_id))

    validation_ds = NvBenchDataset(validation_df, bert_tokenizer, tk, vocab, types)
    validation_dl = DataLoader(validation_ds, batch_size=batch_size, collate_fn=DataPadder(vocab, types, bert_pad_id=bert_tokenizer.pad_token_id))

    test_ds = NvBenchDataset(test_df, bert_tokenizer, tk, vocab, types)
    test_dl = DataLoader(test_ds, batch_size=batch_size, collate_fn=DataPadder(vocab, types, bert_pad_id=bert_tokenizer.pad_token_id))

    return vocab, vocab, types, batch_size, train_dl, validation_dl, test_dl, max_input_length, train_dl_small
