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


@lru_cache()
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
    train['loop'] = train.index

    missing_list = train[pd.isna(train[col])].index

    train = train[[col, 'time_sn', 'loop']]

    #train = train[~train.index.isin(missing_list)]

    train = train.dropna(how='any')

    train.reset_index(inplace=True, drop=True)

    for missing in missing_list:


        previous_index = train[(train.loop < missing) & pd.notnull(train[col])].index.max()

        previous_index_original = train.iloc[previous_index]['loop']

        train_begin = previous_index - window
        train_end =   previous_index + window
        train_sample = train.iloc[train_begin:train_end]

        original_begin = int(train.iloc[train_begin]["loop"])
        original_end = int(train.iloc[train_end]["loop"])
        gap  = int(missing-previous_index_original)

        if gap == 1 and missing!= missing_list[0]:
            logger.debug(previous_msg)

        previous_msg =f'{col}, begin={original_begin},' \
                      f'previous={int(previous_index_original)}/{previous_index}, ' \
                      f'missing={missing},  gap={gap}, ' \
                      f'end={original_end}/{original_end - original_begin}, {train_sample.shape}'
    logger.debug(previous_msg)






def get_original_sn(df, sn):

    return df.iloc[sn]['loop']





if __name__ == '__main__':

    get_input_analysis('max')
    get_input_analysis('min')
    get_input_analysis('nunique')
    get_input_analysis('missing')