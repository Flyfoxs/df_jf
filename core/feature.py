import sys
import os

from core.config import *
import pandas as pd
from file_cache.utils.util_pandas import *
import matplotlib.pyplot as plot
from file_cache.cache import file_cache
import numpy as np
from functools import lru_cache

@file_cache()
def get_input_analysis(gp_type='missing'):
    df = pd.DataFrame()

    for wtid in range(1, 34):
        wtid = str(wtid)
        train = pd.read_csv(f"./input/{wtid.rjust(3,'0')}/201807.csv") 
        logger.debug(f'================{wtid}================')
        logger.debug(f'{train.shape}, {train.wtid.min()}, {train.wtid.max()}')
        summary = {}
        for col in train:
            if gp_type == 'missing':
                summary[col] = int(pd.isna(train[col]).sum())
            elif gp_type =='max':
                summary[col] = train[col].max()
            elif gp_type == 'min':
                summary[col] = train[col].min()
            elif gp_type == 'nunique':
                summary[col] = train[col].nunique()
            else:
                raise Exception('Unknown gp_type:%s' % gp_type)

        summary['wtid'] = wtid
        summary['total'] =  len(train) 
        #logger.debug(summary)
        df = df.append(summary, ignore_index=True)
    return df


def get_analysis_enum():
    col_list = ['wtid','var053','var066','var016','var020','var047',  ]

    train_list = []
    for wtid in range(1, 34):
        wtid = str(wtid)
        train = pd.read_csv(f"./input/{wtid.rjust(3,'0')}/201807.csv", usecols=col_list)
        train = train.groupby(col_list).agg({'wtid':'count'})
        train.rename(index=str, columns={"wtid": "count"}, inplace=True)
        train = train.reset_index()
        print(train.shape)
        train_list.append(train)

    all = pd.concat(train_list)
    return all


@lru_cache()
@file_cache()
def get_sub_template():
    template = pd.read_csv('./input/template_submit_result.csv')
    template = template.set_index(['ts', 'wtid'])

    for wtid in range(1, 34):
        wtid = str(wtid)
        train = pd.read_csv(f"./input/{wtid.rjust(3,'0')}/201807.csv")
        #train = pd.read_csv('./input/001/201807.csv')
        train['sn'] = train.index
        train = train.set_index(['ts', 'wtid'])
        train = train[train.index.isin(template.index)]
        template = template.combine_first(train)
        logger.debug(f'wtid={wtid}, {template.shape}, {train.shape},')
    template = template.reset_index()
    template = template.sort_values(['wtid', 'ts', ])
    return template


@lru_cache(maxsize=2)
def get_train_ex(wtid):
    wtid = str(wtid)
    train = pd.read_csv(f"./input/{wtid.rjust(3,'0')}/201807.csv", parse_dates=['ts'])
    old_shape = train.shape

    train.set_index('ts', inplace=True)

    template = get_sub_template()
    template = template[template.wtid == int(wtid)]
    template.set_index('ts', inplace=True)
    template.drop(columns='sn', errors='ignore', inplace=True)

    logger.debug(f'template={template.shape}')

    train = train.combine_first(template)

    logger.debug(f'Convert train#{wtid} from {old_shape} to {train.shape}')

    train.reset_index(inplace=True)

    train.sort_values('ts', inplace=True)

    train.reset_index(inplace=True, drop=True)

    train['time_sn'] = (train.ts - pd.to_datetime('2018-07-01')).astype(int) / 1000000000

    return train


def get_result(wtid, col, window=100):
    train = get_train_ex(wtid)
    missing_list = train[pd.isna(train[col])].index

    block_count = 0
    last_missing = 0
    for missing in missing_list:
        if missing <= last_missing:
            continue

        block_count += 1
        begin, end, train = get_missing_block(wtid, col, missing, window)
        last_missing = end


        msg =f'wtid={wtid:3},{col}#{block_count:2},length={1+end-begin:4},' \
                      f'begin={begin},' \
                      f'end={end},' \
                      f'missing={missing},'
        logger.debug(msg)




def get_missing_block(wtid, col, cur_missing, window=100):
    train = get_train_ex(wtid)
    begin = train[col].loc[:cur_missing].dropna().index.max() + 1
    end   = train[col].loc[cur_missing:].dropna().index.min() - 1

    train_col_list = [col, 'time_sn' ]
    train_before = train[train_col_list].iloc[:begin].dropna(how='any').iloc[-window:]

    train_after = train[train_col_list].iloc[end+1:].dropna(how='any').iloc[:window]

    train = pd.concat([train_before, train_after])

    return begin, end, train





if __name__ == '__main__':

    columns = list(date_type.keys())
    columns.remove('wtid')
    columns = sorted(columns)
    for col in columns:
        for wtid in range(1, 34):
            get_result(wtid, col, 100)
