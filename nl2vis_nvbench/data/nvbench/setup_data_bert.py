from nl2vis_nvbench.data.nvbench.build_vocab import read_data, Tokenizer, VocabBuilder
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os.path as osp
from nl2vis_nvbench import root
import torch


class NvBenchBertDataset(Dataset):
    def __init__(
            self,
            data,
            bert_tokenizer,
            label_tokenizer,
            label_vocab,
    ):

        data = data.reset_index(drop=True)
        data.columns = ["source", "label", "token_type"]
        self.data = data
        self.bert_tokenizer = bert_tokenizer
        self.label_tokenzier = label_tokenizer
        self.label_vocab = label_vocab
        self.sos = self.label_vocab.get_stoi()['<sos>']
        self.eos = self.label_vocab.get_stoi()['<eos>']

    def string_to_ids(self, input, tokenizer, vocab):

        stoi = vocab.get_stoi()
        res = [stoi[t] for t in tokenizer(input)]

        res = [self.sos] + res + [self.eos]
        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        cur_data = self.data.loc[idx]

        src_res = self.bert_tokenizer(cur_data["source"])
        src = src_res["input_ids"]

        lbl = self.string_to_ids(cur_data["label"], self.label_tokenzier, self.label_vocab)

        has_chart_type = '[T]' not in cur_data["source"]

        return src, lbl, has_chart_type, cur_data["source"]


class DataPadder:
    def __init__(self, src_pad_id, trg_pad_id):
        self._src_pad_id = src_pad_id
        self._trg_pad_id = trg_pad_id

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

        src_batch = self.batch_data(data_zip[0], self._src_pad_id)
        lbl_batch = self.batch_data(data_zip[1], self._trg_pad_id)

        has_chart_type_batch = data_zip[2]

        source_batch = data_zip[3]

        return src_batch, lbl_batch, has_chart_type_batch, source_batch


def get_bert_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    return tokenizer


# TODO: should probably use the pre modified nl query (not the set in dataset_final)
def setup_data(
        data_dir=osp.join(root(), "data", "nvbench", "dataset", "dataset_final"),
        db_info=osp.join(root(), "data", "nvbench", "dataset", "database_information.csv"),
        batch_size=128
):
    test_df, validation_df, train_df, info_df = read_data(data_dir, db_info)

    label_tokenizer = Tokenizer()
    vb = VocabBuilder(tokenizer=label_tokenizer)

    vocab_dfs = [
        test_df[["labels"]],
        validation_df[["labels"]],
        train_df[["labels"]],
        # info_df[["table", "column", "value"]] # should not be needed
    ]
    label_vocab = vb.build_vocab_from_dfs(vocab_dfs)
    label_vocab.set_default_index(0)

    bert_tokenizer = get_bert_tokenizer()

    train_ds = NvBenchBertDataset(train_df, bert_tokenizer, label_tokenizer, label_vocab)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=DataPadder(
            src_pad_id=bert_tokenizer.pad_token_id,
            trg_pad_id=label_vocab.get_stoi()['<pad>']
        )
    )

    train_ds_small = NvBenchBertDataset(train_df.head(100), bert_tokenizer, label_tokenizer, label_vocab)
    train_dl_small = DataLoader(
        train_ds_small,
        batch_size=batch_size,
        collate_fn=DataPadder(
            src_pad_id=bert_tokenizer.pad_token_id,
            trg_pad_id=label_vocab.get_stoi()['<pad>']
        )
    )

    validation_ds = NvBenchBertDataset(validation_df, bert_tokenizer, label_tokenizer, label_vocab)
    validation_dl = DataLoader(
        validation_ds,
        batch_size=batch_size,
        collate_fn=DataPadder(
            src_pad_id=bert_tokenizer.pad_token_id,
            trg_pad_id=label_vocab.get_stoi()['<pad>']
        )
    )

    test_ds = NvBenchBertDataset(test_df, bert_tokenizer, label_tokenizer, label_vocab)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=DataPadder(
            src_pad_id=bert_tokenizer.pad_token_id,
            trg_pad_id=label_vocab.get_stoi()['<pad>']
        )
    )

    return train_dl, validation_dl, test_dl, train_dl_small, label_vocab
