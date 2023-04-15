import pandas as pd
import sqlite3
import re
import os
import os.path as osp
import json

import torch


def postprocessing_group(gold_q_tok, pred_q_tok):
    # 2. checking (and correct) group-by

    # rule: if other part is the same, and only add group-by part, the result should be the same
    if 'group' not in gold_q_tok and 'group' in pred_q_tok:
        groupby_x = pred_q_tok[pred_q_tok.index('group') + 1]
        if ' '.join(pred_q_tok).replace('group ' + groupby_x, '') == ' '.join(gold_q_tok):
            pred_q_tok = gold_q_tok

    return pred_q_tok


def postprocessing(gold_query, pred_query, if_template, src_input):
    try:
        # get the template:
        chart_template = re.findall('<c>.*</c>', src_input)[0]
        chart_template_tok = chart_template.lower().split(' ')

        gold_q_tok = gold_query.lower().split(' ')
        pred_q_tok = pred_query.lower().split(' ')

        # 0. visualize type. if we have the template, the visualization type must be matched.
        if if_template:
            pred_q_tok[pred_q_tok.index('mark') + 1] = gold_q_tok[gold_q_tok.index('mark') + 1]

        # 1. Table Checking. If we focus on single table, must match!!!
        if 'data' in pred_q_tok and 'data' in gold_q_tok:
            pred_q_tok[pred_q_tok.index('data') + 1] = gold_q_tok[gold_q_tok.index('data') + 1]

        pred_q_tok = postprocessing_group(gold_q_tok, pred_q_tok)

        # 3. Order-by. if we have the template, we can checking (and correct) the predicting order-by
        # rule 1: if have the template, order by [x]/[y], trust to the select [x]/[y]
        if 'sort' in gold_q_tok and 'sort' in pred_q_tok and if_template:
            order_by_which_axis = chart_template_tok[chart_template_tok.index('sort') + 1]  # [x], [y], or [o]
            if order_by_which_axis == '[x]':
                pred_q_tok[pred_q_tok.index('sort') + 1] = 'x'
            elif order_by_which_axis == '[y]':
                pred_q_tok[pred_q_tok.index('sort') + 1] = 'y'
            else:
                pass

        elif 'sort' in gold_q_tok and 'sort' not in pred_q_tok and if_template:
            order_by_which_axis = chart_template_tok[chart_template_tok.index('sort') + 1]  # [x], [y], or [o]
            order_type = chart_template_tok[chart_template_tok.index('sort') + 2]

            if 'x' == gold_q_tok[gold_q_tok.index('sort') + 1] or 'y' == gold_q_tok[gold_q_tok.index('sort') + 1]:
                pred_q_tok += ['sort', gold_q_tok[gold_q_tok.index('sort') + 1]]
                if gold_q_tok.index('sort') + 2 < len(gold_q_tok):
                    pred_q_tok += [gold_q_tok[gold_q_tok.index('sort') + 2]]
            else:
                pass

        else:
            pass

        pred_q_tok = postprocessing_group(gold_q_tok, pred_q_tok)

        # 4. checking (and correct) bining
        # rule 1: [interval] bin
        # rule 2: bin by [x]
        if 'bin' in gold_q_tok and 'bin' in pred_q_tok:
            # rule 1
            if_bin_gold, if_bin_pred = False, False

            for binn in ['by time', 'by year', 'by weekday', 'by month']:
                if binn in gold_query:
                    if_bin_gold = binn
                if binn in pred_query:
                    if_bin_pred = binn

            if (if_bin_gold != False and if_bin_pred != False) and (if_bin_gold != if_bin_pred):
                pred_q_tok[pred_q_tok.index('bin') + 3] = if_bin_gold.replace('by ', '')

        if 'bin' in gold_q_tok and 'bin' not in pred_q_tok and 'group' in pred_q_tok:
            # rule 3: group-by x and bin x by time in the bar chart should be the same.
            bin_x = gold_q_tok[gold_q_tok.index('bin') + 1]
            group_x = pred_q_tok[pred_q_tok.index('group') + 1]
            if bin_x == group_x:
                if ''.join(pred_q_tok).replace('group ' + group_x, '') == ''.join(gold_q_tok).replace(
                        'bin ' + bin_x + ' by time', ''):
                    pred_q_tok = gold_q_tok

        # group x | bin x ... count A == count B
        if 'count' in gold_q_tok and 'count' in pred_q_tok:
            if ('group' in gold_q_tok and 'group' in pred_q_tok) or ('bin' in gold_q_tok and 'bin' in pred_q_tok):
                pred_count = pred_q_tok[pred_q_tok.index('count') + 1]
                gold_count = gold_q_tok[gold_q_tok.index('count') + 1]
                if ' '.join(pred_q_tok).replace('count ' + pred_count, 'count ' + gold_count) == ' '.join(gold_q_tok):
                    pred_q_tok = gold_q_tok

    except:
        print('error at post processing')
    return ' '.join(pred_q_tok)


def get_all_table_columns(data_file):
    with open(data_file, 'r') as fp:
        data = json.load(fp)
    '''
    return:
    {'chinook_1': {'Album': ['AlbumId', 'Title', 'ArtistId'],
      'Artist': ['ArtistId', 'Name'],
      'Customer': ['CustomerId',
       'FirstName',
    '''
    return data


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


def fix_chart_template(chart_template=None):
    query_template = \
        'mark [T] ' \
        'data [D] ' \
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
                query_template = query_template.replace('sort [S]', 'sort  ' +order_xy +'  ' +order_type)
        except:
            raise ValueError('Error at settings of sorting!')

        return query_template
    else:
        return query_template
