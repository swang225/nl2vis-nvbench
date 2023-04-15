import torch


from nebula.common import Counter, is_matching
from nebula.model.ncnet_no_af2.translate import (
    postprocessing,
    get_all_table_columns
)
from nebula.model.nv_ncbert3 import nvncBert3
from nebula import root

import random
import numpy as np
import pandas as pd
import os.path as osp


if __name__ == "__main__":

    from argparse import Namespace

    opt = Namespace()
    base_dir = root()
    opt.model = osp.join(root(), "model/nv_ncbert3/result/model_best_nvncbert.pt")
    opt.data_dir = osp.join(root(), "data/nvbench/dataset/dataset_final")
    opt.db_info = osp.join(root(), "data/nvbench/dataset/database_information.csv")
    opt.test_data = osp.join(root(), "data/nvbench/dataset/dataset_final/test.csv")
    opt.db_schema = osp.join(root(), "data/nvbench/dataset/db_tables_columns.json")
    opt.db_tables_columns_types = osp.join(root(), "data/nvbench/dataset/db_tables_columns_types.json")
    opt.batch_size = 128
    opt.max_input_length = 128
    opt.show_progress = False

    print("the input parameters: ", opt)

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    m1 = nvncBert3(
        trained_model_path=opt.model,
    )

    ncNet = m1.ncNet
    db_tables_columns = get_all_table_columns(opt.db_schema)
    db_tables_columns_types = get_all_table_columns(opt.db_tables_columns_types)

    test_df = pd.read_csv(opt.test_data)

    only_nl_cnt = 0
    only_nl_match = 0
    nl_template_cnt = 0
    nl_template_match = 0

    counter = Counter(total=len(test_df))
    counter.start()

    for index, row in test_df.iterrows():

        try:
            gold_query = row['labels'].lower()
            src = row['source'].lower()
            tok_types = row['token_types']
            db_id = row['db_id']
            table_name = gold_query.split(' ')[gold_query.split(' ').index('data') + 1]

            pred_query, attention, enc_attention = m1.translate(
                input_src=src,
                token_types=tok_types,
                visualization_aware_translation=True,
                show_progress=False,
                db_id=db_id,
                table_name=table_name,
                db_tables_columns=db_tables_columns,
                db_tables_columns_types=db_tables_columns_types,
            )

            old_pred_query = pred_query

            if '[t]' not in src:
                # with template
                pred_query = postprocessing(gold_query, pred_query, True, src)

                nl_template_cnt += 1
                if is_matching(gold_query, pred_query):
                    nl_template_match += 1
            else:
                # without template
                pred_query = postprocessing(gold_query, pred_query, False, src)

                only_nl_cnt += 1
                if is_matching(gold_query, pred_query):
                    only_nl_match += 1

        except Exception as e:
            print(f'error {e}')

        if only_nl_cnt > 0 and nl_template_cnt > 0:
            print("--")
            print('nv bert w/o chart template:', only_nl_match / only_nl_cnt)
            print('nv bert with chart template:', nl_template_match / nl_template_cnt)
            print('nv bert overall:', (only_nl_match + nl_template_match) / (only_nl_cnt + nl_template_cnt))

        counter.update()
        # if index > 100:
        #     break

    print("--")
    print('nvncBert3 w/o chart template:', only_nl_match / only_nl_cnt)
    print('nvncBert3 with chart template:', nl_template_match / nl_template_cnt)
    print('nvncBert3 overall:', (only_nl_match + nl_template_match) / (only_nl_cnt + nl_template_cnt))
