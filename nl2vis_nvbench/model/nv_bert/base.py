import pandas as pd
import torch
import sqlite3
import re
import os
import os.path as osp

from nl2vis_nvbench.common import get_device
from nl2vis_nvbench.data.nvbench.process_dataset import ProcessData4Training
from nl2vis_nvbench.data.nvbench.setup_data_bert import get_bert_tokenizer, setup_data
from nl2vis_nvbench.model.nv_bert.component.bert_encoder import BertEncoder, EMBEDDING_SIZE
from nl2vis_nvbench.model.nv_bert.component.decoder import Decoder
from nl2vis_nvbench.model.nv_bert.component.seq2seq import Seq2Seq
from nl2vis_nvbench.model.nv_bert.translate import get_token_types, fix_chart_template, postprocessing


def get_candidate_columns(src):
    col_list = re.findall('<col>.*</col>', src)[0].lower().split(' ')
    return col_list[1:-1] # remove <col> </col>


def get_template(src):
    col_list = re.findall('<c>.*</c>', src)[0].lower().split(' ')
    return col_list[1:-1] # remove <template> </template>


def get_chart_type(pred_tokens_list):
    return pred_tokens_list[pred_tokens_list.index('mark') + 1]


def get_agg_func(pred_tokens_list):
    return pred_tokens_list[pred_tokens_list.index('aggregate') + 1]


def get_x(pred_tokens_list):
    return pred_tokens_list[pred_tokens_list.index('x') + 1]


def get_y(pred_tokens_list):
    return pred_tokens_list[pred_tokens_list.index('aggregate') + 2]


def guide_decoder_by_candidates(
        db_id,
        table_id,
        trg_field,
        input_source,
        table_columns,
        db_tables_columns_types,
        topk_ids,
        topk_tokens,
        current_token_type,
        pred_tokens_list
):
    '''
    get the current token types (X, Y,...),
    we use the topk tokens from the decoder and the candidate columns to inference the "best" pred_token.
    table_columns: all columns in this table.
    topk_tokens: the top-k candidate predicted tokens
    current_token_type = x|y|groupby-axis|bin x|  if_template:[orderby-axis, order-type, chart_type]
    pred_tokens_list: the predicted tokens list
    '''
    # candidate columns mentioned by the NL query
    candidate_columns = get_candidate_columns(input_source)

    best_token = topk_tokens[0]
    best_id = topk_ids[0]

    if current_token_type == 'x_axis':
        mark_type = get_chart_type(pred_tokens_list)

        if best_token not in table_columns and '(' not in best_token:
            is_in_topk = False
            for tok in topk_tokens:
                if tok in candidate_columns and tok in table_columns:
                    # get column's type
                    if mark_type in ['bar', 'line'] and db_tables_columns_types!=None and db_tables_columns_types[db_id][table_id][tok] != 'numeric':
                        best_token = tok
                        best_id = trg_field[best_token]
                        is_in_topk = True
                        break
                    if mark_type == 'point' and db_tables_columns_types!=None and db_tables_columns_types[db_id][table_id][tok] == 'numeric':
                        best_token = tok
                        best_id = trg_field[best_token]
                        is_in_topk = True
                        break
                    if mark_type == 'arc' and db_tables_columns_types!=None and db_tables_columns_types[db_id][table_id][tok] != 'numeric':
                        best_token = tok
                        best_id = trg_field[best_token]
                        is_in_topk = True
                        break

            if is_in_topk == False and len(candidate_columns) > 0:
                for col in candidate_columns:
                    if col != '':
                        if mark_type in ['bar', 'line'] and db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][col] != 'numeric':
                            best_token = col
                            best_id = trg_field[best_token]
                            break

                        if mark_type == 'point' and db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][col] == 'numeric':
                            best_token = col
                            best_id = trg_field[best_token]
                            break
                        if mark_type == 'arc' and db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][col] != 'numeric':
                            best_token = col
                            best_id = trg_field[best_token]
                            break

    if current_token_type == 'y_axis':
        mark_type = get_chart_type(pred_tokens_list)
        agg_function = get_agg_func(pred_tokens_list)
        selected_x = get_x(pred_tokens_list)

        y = best_token

        if y not in table_columns and y != 'distinct':
            is_in_topk = False
            for tok in topk_tokens:
                if tok in candidate_columns and tok in table_columns:
                    if mark_type in ['bar', 'arc', 'line'] and agg_function == 'count':
                        best_token = tok
                        best_id = trg_field[best_token]
                        is_in_topk = True
                        break
                    if mark_type in ['bar', 'arc', 'line'] and agg_function != 'count' and \
                            db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][tok] == 'numeric':
                        best_token = tok
                        best_id = trg_field[best_token]
                        is_in_topk = True
                        break
                    if mark_type == 'point' and tok != selected_x:
                        best_token = tok
                        best_id = trg_field[best_token]
                        break

            if is_in_topk == False and len(candidate_columns) > 0:
                for col in candidate_columns:
                    if col != '':
                        if mark_type in ['bar', 'arc', 'line'] and agg_function == 'count':
                            best_token = col
                            best_id = trg_field[best_token]
                            is_in_topk = True
                            break
                        if mark_type in ['bar', 'arc', 'line'] and agg_function != 'count' and \
                                db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][col] == 'numeric':
                            best_token = col
                            best_id = trg_field[best_token]
                            break
                        if mark_type == 'point' and col != selected_x:
                            best_token = col
                            best_id = trg_field[best_token]
                            break

        # TODO!
        if (y in table_columns and y not in candidate_columns) and ('(' not in y):
            for tok in topk_tokens:
                if tok in candidate_columns and tok in table_columns:
                    best_token = tok
                    best_id = trg_field[best_token]
                    is_in_topk = True
                    break

    if current_token_type == 'z_axis':
        selected_x = get_x(pred_tokens_list)
        selected_y = get_y(pred_tokens_list)

        if best_token not in table_columns or best_token == selected_x or best_token == selected_y:
            is_in_topk = False
            for tok in topk_tokens:
                if tok in candidate_columns and tok in table_columns:
                    # get column's type
                    if selected_x != tok and selected_y != tok and db_tables_columns_types !=None and db_tables_columns_types[db_id][table_id][tok] == 'categorical':
                        best_token = tok
                        best_id = trg_field[best_token]
                        is_in_topk = True
                        break

            if is_in_topk == False and len(candidate_columns) > 0:
                for col in candidate_columns:
                    if col != selected_x and col != selected_y and db_tables_columns_types!=None and db_tables_columns_types[db_id][table_id][
                        col] == 'categorical':
                        best_token = col
                        best_id = trg_field[best_token]
                        break

        if selected_x == best_token or selected_y == best_token or db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][
            best_token] != 'categorical':
            for tok in topk_tokens:
                if tok in candidate_columns and tok in table_columns:
                    # get column's type
                    if selected_x != tok and selected_y != tok and db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][
                        tok] == 'categorical':
                        best_token = tok
                        best_id = trg_field[best_token]
                        break

    if current_token_type == 'topk':  # bin [x] by ..
        is_in_topk = False
        if best_token.isdigit() == False:
            for tok in topk_tokens:
                if tok.isdigit():
                    best_token = tok
                    is_in_topk = True
        if is_in_topk == False:
            best_token = '3'  # default
        best_id = trg_field[best_token]

    if current_token_type == 'groupby_axis':
        if best_token != 'x':
            if best_token not in table_columns or db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][best_token] == 'numeric':
                is_in_topk = False
                for tok in topk_tokens:
                    if tok in candidate_columns and tok in table_columns:
                        # get column's type
                        if db_tables_columns_types != None and db_tables_columns_types[db_id][table_id][tok] == 'categorical':
                            best_token = tok
                            best_id = trg_field[best_token]
                            is_in_topk = True
                            break

                if is_in_topk == False:
                    best_token = get_x(pred_tokens_list)
                    best_id = trg_field[best_token]

    if current_token_type == 'bin_axis':  # bin [x] by ..
        best_token = 'x'
        best_id = trg_field[best_token]

    template_list = get_template(input_source)

    if '[t]' not in template_list:  # have the chart template
        if current_token_type == 'chart_type':
            best_token = template_list[template_list.index('mark') + 1]
            best_id = trg_field[best_token]

        if current_token_type == 'orderby_axis':
            #   print('Case-3')
            if template_list[template_list.index('sort') + 1] == '[x]':
                best_token = 'x'
                best_id = trg_field[best_token]

            elif template_list[template_list.index('sort') + 1] == '[y]':
                best_token = 'y'
                best_id = trg_field[best_token]
            else:
                pass
                # print('Let me know this issue!')

        if current_token_type == 'orderby_type':
            best_token = template_list[template_list.index('sort') + 2]
            best_id = trg_field[best_token]

    return best_id, best_token


# TODO: incomplete, finish translation with guidance
def translate(
        input_src,
        model,
        label_vocab,
        device,
        db_id,
        table_id,
        db_tables_columns,
        db_tables_columns_types,
        max_len=128
):
    model.eval()

    tokenizer = get_bert_tokenizer()
    res = tokenizer(input_src, return_tensors="pt")
    src_tensor = res["input_ids"]
    src_mask = res["attention_mask"]

    with torch.no_grad():
        enc_src = \
            model.encoder(src_tensor, src_mask)

    trg_indexes = [label_vocab['<sos>']]
    trg_tokens = []
    current_token_type = None

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = \
                model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        table_columns = []
        try:  # get all columns in a table
            table_columns = db_tables_columns[db_id][table_id]
        except:
            print('[Fail] get all columns in a table')
            table_columns = []

        if current_token_type == 'table_name':
            '''
            only for single table !!!
            '''
            pred_token = table_id
            pred_id = label_vocab[pred_token]
            # print('-------------------\nCurrent Token Type: Table Name , top-3 tokens: [{}]'.format(
            #     current_token_type, pred_token))

        else:
            topk_ids = torch.topk(output, k=5, dim=2, sorted=True).indices[:, -1, :].tolist()[0]
            topk_tokens = [label_vocab.get_itos()[tok_id] for tok_id in topk_ids]

            '''
            apply guide_decoder_by_candidates
            '''
            pred_id, pred_token = guide_decoder_by_candidates(
                db_id, table_id, label_vocab, input_src, table_columns, db_tables_columns_types, topk_ids,
                topk_tokens, current_token_type, trg_tokens
            )

            # if current_token_type == None:
            #     print('-------------------\nCurrent Token Type: Query Sketch Part , top-3 tokens: [{}]'.format(
            #         ', '.join(topk_tokens)))
            # else:
            #     print(
            #         '-------------------\nCurrent Token Type: {} , original top-3 tokens: [{}] , the final tokens by VisAwareTranslation: {}'.format(
            #             current_token_type, ', '.join(topk_tokens), pred_token))

        current_token_type = None

        trg_indexes.append(pred_id)
        trg_tokens.append(pred_token)

        # update the current_token_type and pred_aix here
        # mark bar data apartments encoding x apt_type_code y aggregate count apt_type_code transform group x sort y desc
        if i == 0:
            current_token_type = 'chart_type'

        if i > 1:
            if trg_tokens[-1] == 'data' and trg_tokens[-2] in ['bar', 'arc', 'line', 'point']:
                current_token_type = 'table_name'

        if i > 2:
            if trg_tokens[-1] == 'x' and trg_tokens[-2] == 'encoding':
                current_token_type = 'x_axis'

            if trg_tokens[-1] == 'aggregate' and trg_tokens[-2] == 'y':
                current_token_type = 'aggFunction'

            if trg_tokens[-2] == 'aggregate' and trg_tokens[-1] in ['count', 'sum', 'mean', 'avg', 'max', 'min']:
                current_token_type = 'y_axis'

            if trg_tokens[-3] == 'aggregate' and trg_tokens[-2] in ['count', 'sum', 'mean', 'avg', 'max', 'min'] and \
                    trg_tokens[-1] == 'distinct':
                current_token_type = 'y_axis'

            # mark [T] data photos encoding x [X] y aggregate [AggFunction] [Y] color [Z] transform filter [F] group [G] bin [B] sort [S] topk [K]
            if trg_tokens[-1] == 'color' and trg_tokens[-4] == 'aggregate':
                current_token_type = 'z_axis'

            if trg_tokens[-1] == 'bin':
                current_token_type = 'bin_axis'

            if trg_tokens[-1] == 'group':
                current_token_type = 'groupby_axis'

            if trg_tokens[-1] == 'sort':
                current_token_type = 'orderby_axis'

            if trg_tokens[-2] == 'sort' and trg_tokens[-1] in ['x', 'y']:
                current_token_type = 'orderby_type'

            if trg_tokens[-1] == 'topk':
                current_token_type = 'topk'

        if pred_id == label_vocab['<eos>']:
            break

    return trg_tokens, attention


class nvBert:
    def __init__(
            self,
            temp_dataset_path=".",
            batch_size=128,
    ):
        self.temp_dataset_path = temp_dataset_path
        self.device = get_device()

        (
            self.train_dl,
            self.validation_dl,
            self.test_dl,
            self.train_dl_small,
            self.label_vocab
        ) = setup_data(batch_size=batch_size)

        OUTPUT_DIM = len(self.label_vocab.vocab)
        HID_DIM = EMBEDDING_SIZE  # it equals to embedding dimension
        DEC_LAYERS = 3
        DEC_HEADS = 8
        DEC_PF_DIM = 512
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1
        MAX_LENGTH = 128

        enc = BertEncoder(dropout=ENC_DROPOUT)

        dec = Decoder(OUTPUT_DIM,
                      HID_DIM,
                      DEC_LAYERS,
                      DEC_HEADS,
                      DEC_PF_DIM,
                      DEC_DROPOUT,
                      self.device,
                      MAX_LENGTH
                      )

        bert_tokenizer = get_bert_tokenizer()
        self.SRC_PAD_IDX = bert_tokenizer.pad_token_id
        self.TRG_PAD_IDX = self.label_vocab.get_stoi()["<pad>"]

        self.model = Seq2Seq(
            enc,
            dec,
            self.SRC_PAD_IDX,
            self.TRG_PAD_IDX,
            self.device
        ).to(self.device)

    def load_model(self, trained_model_path):
        self.model.load_state_dict(
            torch.load(
                trained_model_path,
                map_location=self.device
            )
        )

    def translate(
            self,
            input_src,
            token_types,
            db_id=None,
            table_id=None,
            db_tables_columns=None,
            db_tables_columns_types=None
    ):

        db_id = db_id or self.db_id
        table_id = table_id or self.table_id
        db_tables_columns = db_tables_columns or self.db_tables_columns
        db_tables_columns_types = db_tables_columns_types or self.db_tables_columns_types

        res, attention = translate(
            input_src,
            self.model,
            self.label_vocab,
            self.device,
            db_id,
            table_id,
            db_tables_columns,
            db_tables_columns_types
        )

        res = ' '.join(res).replace(' <eos>', '').lower()

        return res

    def predict(
            self,
            nl_question,
            chart_template=None,
            show_progress=None,
            visualization_aware_translation=True
    ):

        input_src, token_types = self.process_input(nl_question, chart_template)

        res = self.translate(
            input_src=input_src,
            token_types=token_types
        )

        if chart_template != None:
            res = postprocessing(res, res, True, input_src)
        else:
            res = postprocessing(res, res, False, input_src)

        res = ' '.join(res.replace('"', "'").split())

        print('[NL Question]:', nl_question)
        print('[Chart Template]:', chart_template)
        print('[Predicted VIS Query]:', res)

        return res

    def process_input(self, nl_question, chart_template):

        query_template = fix_chart_template(chart_template)

        # get a list of mentioned values in the NL question
        col_names, value_names = self.data_processor.get_mentioned_values_in_NL_question(
            self.db_id, self.table_id, nl_question, db_table_col_val_map=self.db_table_col_val_map
        )
        col_names = ' '.join(str(e) for e in col_names)
        value_names = ' '.join(str(e) for e in value_names)

        input_src = (
            f"<N> {nl_question} </N> " \
            f"<C> {query_template} </C> " \
            f"<D> {self.table_id} <COL> {col_names} </COL> <VAL> {value_names} </VAL> </D>").lower()
        token_types = get_token_types(input_src)

        return input_src, token_types

    def specify_dataset(
            self,
            data_type,
            db_url = None,
            table_name = None,
            data = None,
            data_url = None
    ):
        '''
        this function creates a temporary save db for the input data
        the save db is a sqlite db in ./dataset/database
        the db name is temp_<table_name>, there is one table in it called <table_name>

        :param data_type: sqlite3, csv, json
        :param db_url: db path for sqlite3 database,
                       e.g., './dataset/database/flight/flight.sqlite'
        :param table_name: the table name in a sqlite3
        :param data: DataFrame for csv
        :param data_url: data path for csv or json
        :return: save the DataFrame in the self.data
        '''
        self.db_id = 'temp_' + table_name
        self.table_id = table_name

        # read in data as dataframe
        if data_type == 'csv':
            if data != None and data_url == None:
                self.data = data
            elif data == None and data_url != None:
                self.data = pd.read_csv(data_url)
            else:
                raise ValueError('Please only specify one of the data or data_url')
        elif data_type == 'json':
            if data == None and data_url != None:
                self.data = pd.read_json(data_url)
            else:
                raise ValueError(
                    'Read JSON from the json file, ' 
                    'please only specify the "data_type" or "data_url"'
                )
        elif data_type == 'sqlite3':
            # Create your connection.
            try:
                cnx = sqlite3.connect(db_url)
                self.data = pd.read_sql_query("SELECT * FROM " + table_name, cnx)
            except:
                raise ValueError(
                    f'Errors in read table from sqlite3 database. \n' 
                    f'db_url: {db_url}\n'
                    f' table_name : {table_name} '
                )
        else:
            if data != None and type(data) == pd.core.frame.DataFrame:
                self.data = data
            else:
                raise ValueError(
                    'The data type must be one of the '
                    'csv, json, sqlite3, or a DataFrame object.'
                )

        # same data column name and types
        self.db_tables_columns_types = dict()
        self.db_tables_columns_types[self.db_id] = dict()
        self.db_tables_columns_types[self.db_id][table_name] = dict()
        for col, _type in self.data.dtypes.items():
            # print(col, _type)
            if 'int' in str(_type).lower() or 'float' in str(_type).lower():
                _type = 'numeric'
            else:
                _type = 'categorical'
            self.db_tables_columns_types[self.db_id][table_name][col.lower()] = _type

        # convert all columns in data df to string lower case
        self.data.columns = self.data.columns.str.lower()

        # a dictionary of table column names
        self.db_tables_columns = {
            self.db_id:{
                self.table_id: list(self.data.columns)
            }
        }

        # saves the input data to a storage place in .'dataset/database
        # to be used by data processor
        if data_type == 'json' or data_type == 'sqlite3':
            # write to sqlite3 database
            dir = osp.join(self.temp_dataset_path, self.db_id)
            if not os.path.exists(dir):
                os.makedirs(dir)
            conn = sqlite3.connect(osp.join(dir, self.db_id+'.sqlite'))
            self.data.to_sql(self.table_id, conn, if_exists='replace', index=False)

        # create data processor and retrieve
        # all data from db and save to db_table_col_val_map
        self.data_processor = ProcessData4Training(db_url=self.temp_dataset_path)
        self.db_table_col_val_map = {}
        table_cols = self.data_processor.get_table_columns(self.db_id)
        self.db_table_col_val_map[self.db_id] = {}
        for table, cols in table_cols.items():
            col_val_map = self.data_processor.get_values_in_columns(
                self.db_id,
                table,
                cols,
                conditions='remove'
            )
            self.db_table_col_val_map[self.db_id][table] = col_val_map

    def show_dataset(self, top_rows=5):
        return self.data[:top_rows]