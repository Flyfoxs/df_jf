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

    #logger.debug(f'template={template.shape}')

    train = train.combine_first(template)

    logger.debug(f'Convert train#{wtid} from {old_shape} to {train.shape}')

    train.reset_index(inplace=True)

    train.sort_values('ts', inplace=True)

    train.reset_index(inplace=True, drop=True)

    train['time_sn'] = (train.ts - pd.to_datetime('2018-07-01')).astype(int) / 1000000000

    return train



@file_cache()
def get_missing_block_all():
    """
    wtid, col, begin, end
    :return:
    """
    df = pd.DataFrame(columns=['wtid', 'col', 'begin', 'end'])
    columns = list(date_type.keys())
    columns.remove('wtid')
    columns = sorted(columns)
    for wtid in sorted(range(1, 34), reverse=True):
        for col in columns:
            for begin, end in get_missing_block_for_col(wtid, col):
                df = df.append({'wtid':wtid, 'col':col,
                                'begin':begin, 'end':end
                                }, ignore_index=True)
    return df

@file_cache()
def get_train_block_all():
    missing = get_missing_block_all()
    df_list = []
    for wtid in range(1, 34):
        for col in missing.col.drop_duplicates():
            df_tmp = missing.loc[(missing.wtid == wtid) & (missing.col == col)]
            missing_end = df_tmp.end.max()
            df_tmp.sort_values('begin', inplace=True)
            df_tmp['begin'], df_tmp['end'] = (df_tmp.end.shift(1) + 1).fillna(0), df_tmp.begin - 1

            df_tmp['min'], df_tmp['max'], df_tmp['distinct'] = None, None, None

            train = get_train_ex(wtid)
            train_len = len(train)

            df_tmp = df_tmp.append({'wtid': wtid, 'col': col,
                                    'begin': missing_end + 1,
                                    'end': train_len - 1,
                                    }, ignore_index=True)

            for index, row in df_tmp.iterrows():
                df_tmp.loc[index, 'distinct'] = train.loc[row.begin:row.end, row.col].nunique()
                if 'float' in date_type[row.col].__name__:
                    df_tmp.loc[index, 'min'] = train.loc[row.begin:row.end, row.col].min()
                    df_tmp.loc[index, 'max'] = train.loc[row.begin:row.end, row.col].max()

            # df_tmp.length = df_tmp.end - df_tmp.begin

            df_list.append(df_tmp)
    return pd.concat(df_list)


@lru_cache()
def get_blocks():
    train = get_train_block_all()

    missing = get_missing_block_all()

    train['kind'] = 'train'
    missing['kind'] = 'missing'

    all = pd.concat([train, missing])

    all['length'] = all.end - all.begin +1
    all.sort_values(['wtid','col','begin'], inplace=True)

    all['data_type'] = all.col.apply(lambda val: date_type[val].__name__)
    return all.reset_index(drop=True)



def get_missing_block_for_col(wtid, col, window=100):
    train = get_train_ex(wtid)
    missing_list = train[pd.isna(train[col])].index

    block_count = 0
    last_missing = 0
    block_list = []
    for missing in missing_list:
        if missing <= last_missing:
            continue

        block_count += 1
        begin, end, train = get_missing_block_single(wtid, col, missing, window)
        block_list.append((begin, end))

        last_missing = end


        msg =f'wtid={wtid:2},{col}#{block_count:2},length={1+end-begin:4},' \
                      f'begin={begin},' \
                      f'end={end},' \
                      f'missing={missing},'
        logger.debug(msg)
    logger.debug(f'get {block_count:2} blocks for wtid:{wtid:2}#{col}, type:{date_type[col]}')

    return block_list



def get_missing_block_single(wtid, col, cur_missing, window=100):
    train = get_train_ex(wtid)
    begin = train[col].loc[:cur_missing].dropna().index.max() + 1
    end   = train[col].loc[cur_missing:].dropna().index.min() - 1

    train_col_list = [col, 'time_sn' ]
    train_before = train[train_col_list].iloc[:begin].dropna(how='any').iloc[-window:]

    train_after = train[train_col_list].iloc[end+1:].dropna(how='any').iloc[:window]

    train = pd.concat([train_before, train_after])

    return begin, end, train


def get_train_feature(wtid, col):
    feature_list = []

    block = get_blocks()

    train_block = block.loc[(block.wtid == wtid) & (block.col == col) & (block.kind == 'train')]

    missing_block = block.loc[(block.wtid == wtid) & (block.col == col) & (block.kind == 'missing')]

    for missing_length in missing_block['length'].sort_values().values:
        cur_windows = round(missing_length * 0.7)
        at_least_len = missing_length + 2 * cur_windows
        logger.debug(
            f'at_least_len={at_least_len}, window={cur_windows}, missing_len={missing_length} {train_block[train_block["length"]>=at_least_len].shape}')
        for index, cur_block in (train_block[train_block['length'] >= at_least_len]).iterrows():
            train = get_train_ex(wtid)
            begin, end = cur_block.begin, cur_block.end
            # Get the data without missing
            block = train.iloc[begin:end + 1][['time_sn', col]]

            block = block.reset_index(drop=True)

            # only pick the latest data closed to training
            block = block.iloc[-at_least_len:]

            logger.debug(
                f'wtid:{wtid}, col:{col}, len:{len(block)}, std:{block[col].std():2.2f}, block:[{end-at_least_len},{end}]')
            train_feature = pd.concat([block.iloc[:cur_windows], block.iloc[-cur_windows:]])
            val_feature = block.iloc[cur_windows: -cur_windows]

            time_gap = val_feature.time_sn.max() - val_feature.time_sn.min()
            time_begin = val_feature.time_sn.min() - 2 * time_gap
            time_end = val_feature.time_sn.max() + 2 * time_gap
            # Make the train closed to validate
            train_feature = train_feature[(train_feature.time_sn >= time_begin) & (train_feature.time_sn <= time_end)]

            feature_list.append((train_feature, val_feature, index))
            # logger.debug(f'Train:{train_feature.shape}, Val:{val_feature.shape}')

    return feature_list


def get_submit_feature_by_block_id(blockid):
    cur_block = get_blocks().iloc[blockid]
    col_name = cur_block['col']
    wtid = cur_block['wtid']
    missing_length = cur_block['length']

    cur_windows = round(missing_length * 0.7)

    train = get_train_ex(wtid)

    begin, end = cur_block.begin, cur_block.end
    # Get the data without missing
    block = train.iloc[begin - cur_windows:end + cur_windows + 1][['time_sn', col_name]]

    block = block.reset_index(drop=True)

    logger.debug(f'wtid:{wtid}, col:{col_name}, len:{len(block)}, std:{block[col_name].std():2.2f}, blockid:{blockid}')
    train_feature = pd.concat([block.iloc[:cur_windows], block.iloc[-cur_windows:]])
    val_feature = block.iloc[cur_windows: -cur_windows]

    logger.debug(f'original: {train_feature.shape}, {val_feature.shape}')

    time_gap = max(30, val_feature.time_sn.max() - val_feature.time_sn.min())
    time_begin = val_feature.time_sn.min() - 5 * time_gap
    time_end = val_feature.time_sn.max() + 5 * time_gap
    # Make the train closed to validate
    train_feature = train_feature[(train_feature.time_sn >= time_begin) & (train_feature.time_sn <= time_end)]

    logger.debug(f'new(filter by time): {train_feature.shape}, {val_feature.shape}')

    return train_feature, val_feature


@file_cache()
def get_std_all():
    df = pd.DataFrame(columns=['wtid', 'col', 'mean', 'min', 'max', 'std'])
    columns = list(date_type.keys())
    columns.remove('wtid')
    columns = sorted(columns)
    for wtid in sorted(range(1, 34), reverse=True):
        for col in columns:
            std_sample =  check_std(wtid,col)
            df = df.append(std_sample,ignore_index=True)

    df['data_type'] = df.col.apply(lambda val: date_type[val].__name__)
    return df

def check_std(wtid, col, windows=100):
    feature_list = []
    std_list = []
    block = get_blocks()

    train_block = block.loc[(block.wtid == wtid) & (block.col == col) & (block.kind == 'train')]

    missing_block = block.loc[(block.wtid == wtid) & (block.col == col) & (block.kind == 'missing')]

    for missing in [missing_block['length'].sort_values().max()]:
        cur_windows = max(windows, missing)
        at_least_len = missing + 2 * cur_windows
        logger.debug(
            f'at_least_len={at_least_len}, window={cur_windows}, missing_len={missing} {train_block[train_block["length"]>=at_least_len].shape}')
        for index, cur_block in (train_block[train_block['length'] >= at_least_len]).iterrows():
            train = get_train_ex(wtid)
            begin, end = cur_block.begin, cur_block.end
            # Get the data without missing
            block = train.iloc[begin:end + 1][['time_sn', col]]

            block = block.reset_index(drop=True)

            # only pick the latest data closed to training
            block = block.iloc[-at_least_len:]

            std_list.append(round(block[col].std(),3))

            logger.debug(f'wtid:{wtid}, col:{col}, len:{len(block)}, std:{block[col].std():2.2f}, block:[{end-at_least_len},{end}]')
            train_feature = pd.concat([block.iloc[:cur_windows], block.iloc[-cur_windows:]])
            val_feature = block.iloc[cur_windows: -cur_windows]

            feature_list.append((train_feature, val_feature))
            #logger.debug(f'Train:{train_feature.shape}, Val:{val_feature.shape}')
    std_list = np.array(std_list)
    summary_map = {
        'wtid':wtid, 'col':col,
        'mean':round(std_list.mean(),3),
        'min':std_list.min(),
        'max':std_list.max(),
        'std':round(std_list.std(), 3),
    }
    logger.debug( f'Summary: {summary_map}')


    return summary_map


def get_predict_fun(blockid, train):
    block = get_blocks().iloc[blockid]

    col_name = block['col']

    wtid = block['wtid']

    is_enum = True if 'int' in date_type[col_name].__name__ else False

    if is_enum:
        fn = lambda val: np.full_like(val, train[col_name].value_counts().index[0])
    else:
        w = np.polyfit(train.time_sn, train[col_name], 1)
        fn = np.poly1d(w)
    return fn


def score(val1, val2, enum=False):
    loss = 0
    for real, predict in zip(val1, val2):
        if enum:
            loss += 1 if real == predict else 0
        else:
            loss += np.exp(-100 * abs(real - predict) / max(abs(real), 1e-15))
    return len(val1), round(loss, 4)

if __name__ == '__main__':

    # columns = list(date_type.keys())
    # columns.remove('wtid')
    # columns = sorted(columns)
    # for wtid in sorted(range(1, 34), reverse=True):
    #     for col in columns:
    #         get_result(wtid, col, 100)

    get_std_all()