import pandas as pd
import sqlite3
import re
import os
import os.path as osp

import torch
from nl2vis_nvbench.common import get_device
from nl2vis_nvbench.model.nv_ncbert3.translate import (
    translate_sentence_with_guidance,
    translate_sentence,
    postprocessing
)
from nl2vis_nvbench.model.nv_ncbert3.component.seq2seq import Seq2Seq
from nl2vis_nvbench.model.nv_ncbert3.component.encoder import Encoder
from nl2vis_nvbench.model.nv_ncbert3.component.decoder import Decoder
from nl2vis_nvbench.data.nvbench.setup_data_bert3 import setup_data, get_bert_tokenizer
from nl2vis_nvbench.data.nvbench.process_dataset import ProcessData4Training
from nl2vis_nvbench import root


class nvncBert3:
    def __init__(
            self,
            trained_model_path=None,
            temp_dataset_path=".",
            batch_size=128,
    ):
        self.temp_dataset_path = temp_dataset_path

        self.device = get_device()

        # state variable to dataset specification
        self.data = None
        self.db_id = ''
        self.table_id = ''
        self.db_tables_columns = None
        self.db_tables_columns_types = None
        self.data_processor = None
        self.db_table_col_val_map = None

        # SRC - vocab for source
        # TRG - vocab for target
        # TOK_TYPES - vocab for token types
        self.SRC, \
        self.TRG, \
        self.TOK_TYPES, \
        self.BATCH_SIZE, \
        self.train_iterator, \
        self.valid_iterator, \
        self.test_iterator, \
        self.my_max_length, \
        self.train_iterator_small, \
            = setup_data(batch_size=batch_size)

        INPUT_DIM = len(self.SRC.vocab)
        OUTPUT_DIM = len(self.TRG.vocab)
        HID_DIM = 256  # it equals to embedding dimension
        ENC_LAYERS = 3
        DEC_LAYERS = 3
        ENC_HEADS = 4
        DEC_HEADS = 4
        ENC_PF_DIM = 512
        DEC_PF_DIM = 512
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1

        enc = Encoder(INPUT_DIM,
                      HID_DIM,
                      ENC_LAYERS,
                      ENC_HEADS,
                      ENC_PF_DIM,
                      ENC_DROPOUT,
                      self.device,
                      self.TOK_TYPES,
                      self.my_max_length
                      )

        dec = Decoder(OUTPUT_DIM,
                      HID_DIM,
                      DEC_LAYERS,
                      DEC_HEADS,
                      DEC_PF_DIM,
                      DEC_DROPOUT,
                      self.device,
                      self.my_max_length
                      )

        bert_tokenizer = get_bert_tokenizer()
        BERT_SRC_PAD_IDX = bert_tokenizer.pad_token_id
        self.SRC_PAD_IDX = self.SRC.get_stoi()["<pad>"]
        self.TRG_PAD_IDX = self.SRC.get_stoi()["<pad>"]

        self.ncNet = Seq2Seq(
            enc,
            dec,
            self.SRC,
            self.SRC_PAD_IDX,
            self.TRG_PAD_IDX,
            BERT_SRC_PAD_IDX,
            self.device
        ).to(self.device)

        self.trained_model_path = trained_model_path
        if self.trained_model_path is not None:
            self.ncNet.load_state_dict(
                torch.load(
                    trained_model_path,
                    map_location=self.device
                )
            )

    def vocab_data(self):

        return (
            self.SRC,
            self.TRG,
            self.TOK_TYPES,
            self.BATCH_SIZE,
            self.train_iterator,
            self.valid_iterator,
            self.test_iterator,
            self.my_max_length
        )

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
                    f'db_url: {data_url}\n'
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

    def translate(
            self,
            input_src,
            token_types,
            visualization_aware_translation=True,
            show_progress=False,
            db_id=None,
            table_name=None,
            db_tables_columns=None,
            db_tables_columns_types=None
    ):

        db_id = db_id or self.db_id
        table_name = table_name or self.table_id
        db_tables_columns = db_tables_columns or self.db_tables_columns
        db_tables_columns_types = db_tables_columns_types or self.db_tables_columns_types

        if visualization_aware_translation == True:
            pred_query, attention, enc_attention = translate_sentence_with_guidance(
                db_id,
                table_name,
                input_src,
                self.SRC, self.TRG, self.TOK_TYPES,
                token_types, self.SRC, self.ncNet,
                db_tables_columns, db_tables_columns_types,
                self.device, self.my_max_length, show_progress
            )
        else:
            pred_query, attention, enc_attention = translate_sentence(
                input_src,
                self.SRC, self.TRG, self.TOK_TYPES,
                token_types, self.ncNet,
                self.device, self.my_max_length
            )

        pred_query = ' '.join(pred_query).replace(' <eos>', '').lower()
        return pred_query, attention, enc_attention

    def predict(
            self,
            nl_question,
            chart_template=None,
            show_progress=None,
            visualization_aware_translation=True
    ):
        # process and the nl_question and the chart template as input.
        # call the model to perform prediction
        # render the predicted query

        input_src, token_types = self.process_input(nl_question, chart_template)

        pred_query, attention, enc_attention = self.translate(
            input_src=input_src,
            token_types=token_types,
            visualization_aware_translation=visualization_aware_translation,
            show_progress=show_progress
        )
        if chart_template != None:
            pred_query = postprocessing(pred_query, pred_query, True, input_src)
        else:
            pred_query = postprocessing(pred_query, pred_query, False, input_src)

        pred_query = ' '.join(pred_query.replace('"', "'").split())

        print('[NL Question]:', nl_question)
        print('[Chart Template]:', chart_template)
        print('[Predicted VIS Query]:', pred_query)

        return pred_query

    def process_input(self, nl_question, chart_template):

        def get_token_types(input_source):
            # print('input_source:', input_src)

            token_types = ''

            for ele in re.findall('<n>.*</n>', input_source)[0].split(' '):
                token_types += ' nl'

            for ele in re.findall('<c>.*</c>', input_source)[0].split(' '):
                token_types += ' template'

            token_types += ' table table'

            for ele in re.findall('<col>.*</col>', input_source)[0].split(' '):
                token_types += ' col'

            for ele in re.findall('<val>.*</val>', input_source)[0].split(' '):
                token_types += ' value'

            token_types += ' table'

            token_types = token_types.strip()
            return token_types

        def fix_chart_template(
                chart_template=None
        ):
            query_template = \
                'mark [T] ' \
                'data [D] '\
                'encoding x [X] y aggregate [AggFunction] [Y] ' \
                'color [Z] transform filter [F] ' \
                'group [G] ' \
                'bin [B] ' \
                'sort [S] ' \
                'topk [K]'

            if chart_template != None:
                try:
                    query_template = query_template.replace('[T]', chart_template['chart'])
                except:
                    raise ValueError('Error at settings of chart type!')

                try:
                    if 'sorting_options' in chart_template and chart_template['sorting_options'] != None:
                        order_xy = '[O]'
                        if 'axis' in chart_template['sorting_options']:
                            if chart_template['sorting_options']['axis'].lower() == 'x':
                                order_xy = '[X]'
                            elif chart_template['sorting_options']['axis'].lower() == 'y':
                                order_xy = '[Y]'
                            else:
                                order_xy = '[O]'

                        order_type = 'ASC'
                        if 'type' in chart_template['sorting_options']:
                            if chart_template['sorting_options']['type'].lower() == 'desc':
                                order_type = 'DESC'
                            elif chart_template['sorting_options']['type'].lower() == 'asc':
                                order_type = 'ASC'
                            else:
                                raise ValueError('Unknown order by settings, the order-type must be "desc", or "asc"')
                        query_template = query_template.replace('sort [S]', 'sort '+order_xy+' '+order_type)
                except:
                    raise ValueError('Error at settings of sorting!')

                return query_template
            else:
                return query_template

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


if __name__ == '__main__':
    ncNet = ncNet(
        trained_model_path='./save_models/trained_model.pt'
    )
    ncNet.specify_dataset(
        data_type='sqlite3',
        db_url='./dataset/database/car_1/car_1.sqlite',
        table_name='cars_data'
    )
    ncNet.nl2vis(
        nl_question='What is the average weight and year for each year. Plot them as line chart.',
        chart_template=None
    )
