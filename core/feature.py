import sys
import os

from tqdm import tqdm
from core.config import *
import pandas as pd
from file_cache.utils.util_pandas import *
import matplotlib.pyplot as plot
from file_cache.cache import file_cache
import numpy as np
from functools import lru_cache
from munch import *

def get_predict_col():
    col_list = [col for col in list(date_type.keys()) if 'var' in col]
    return sorted(col_list)
#
# @file_cache()
# def get_input_analysis(gp_type='missing'):
#     df = pd.DataFrame()
#
#     for wtid in range(1, 34):
#         wtid = str(wtid)
#         train = pd.read_csv(f"./input/{wtid.rjust(3,'0')}/201807.csv")
#         logger.debug(f'================{wtid}================')
#         logger.debug(f'{train.shape}, {train.wtid.min()}, {train.wtid.max()}')
#         summary = {}
#         for col in train:
#             if gp_type == 'missing':
#                 summary[col] = int(pd.isna(train[col]).sum())
#             elif gp_type =='max':
#                 summary[col] = train[col].max()
#             elif gp_type == 'min':
#                 summary[col] = train[col].min()
#             elif gp_type == 'nunique':
#                 summary[col] = train[col].nunique()
#             else:
#                 raise Exception('Unknown gp_type:%s' % gp_type)
#
#         summary['wtid'] = wtid
#         summary['total'] =  len(train)
#         #logger.debug(summary)
#         df = df.append(summary, ignore_index=True)
#     return df

#
# def get_analysis_enum():
#     col_list = ['wtid','var053','var066','var016','var020','var047',  ]
#
#     train_list = []
#     for wtid in range(1, 34):
#         wtid = str(wtid)
#         train = pd.read_csv(f"./input/{wtid.rjust(3,'0')}/201807.csv", usecols=col_list)
#         train = train.groupby(col_list).agg({'wtid':'count'})
#         train.rename(index=str, columns={"wtid": "count"}, inplace=True)
#         train = train.reset_index()
#         #print(train.shape)
#         train_list.append(train)
#
#     all = pd.concat(train_list)
#     return all


@lru_cache()
@file_cache()
def get_sub_template():
    template = pd.read_csv('./input/template_submit_result.csv')
    template.ts = pd.to_datetime(template.ts)
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



@lru_cache(maxsize=cache_size)
@timed()
def get_train_ex(wtid):
    wtid = str(wtid)
    train = pd.read_csv(f"./input/{wtid.rjust(3,'0')}/201807.csv", parse_dates=['ts'])
    old_shape = train.shape

    train.set_index('ts', inplace=True)

    template = get_sub_template()
    template = template[template.wtid == int(wtid)]
    template.set_index('ts', inplace=True)
    template = template.drop(columns='sn', errors='ignore')

    #logger.debug(f'template={template.shape}')

    train = train.combine_first(template)

    logger.debug(f'Convert train#{wtid} from {old_shape} to {train.shape}')

    train.reset_index(inplace=True)

    train.sort_values('ts', inplace=True)

    train.reset_index(inplace=True, drop=True)

    train['time_sn'] = (train.ts - pd.to_datetime('2018-07-01')).astype(int)   / 1000000000

    train['time_slot_3'] = (train.ts - pd.to_datetime('2018-07-01')).astype(int) // 3000000000
    train['time_slot_7'] = (train.ts - pd.to_datetime('2018-07-01')).astype(int) // 7000000000

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
def get_data_block_all():
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
    train = get_data_block_all()

    missing = get_missing_block_all()

    train['kind'] = 'train'
    missing['kind'] = 'missing'

    all = pd.concat([train, missing])

    all['length'] = all.end - all.begin +1
    all.sort_values(['wtid','col','begin'], inplace=True)

    all['data_type'] = all.col.apply(lambda val: date_type[val].__name__)
    return all.reset_index(drop=True)


def get_break_block():
    for wtid in range(1, 34):
        train = get_train_ex(wtid)
        train = train[date_type.keys()]

def get_missing_block_for_col(wtid, col):
    train = get_train_ex(wtid)
    missing_list = train[pd.isna(train[col])].index

    block_count = 0
    last_missing = 0
    block_list = []
    for missing in missing_list:
        if missing <= last_missing:
            continue

        block_count += 1
        begin, end = get_missing_block_single(wtid, col, missing)
        block_list.append((begin, end))

        last_missing = end


        msg =f'wtid={wtid:2},{col}#{block_count:2},length={1+end-begin:4},' \
                      f'begin={begin},' \
                      f'end={end},' \
                      f'missing={missing},'
        logger.debug(msg)
    logger.debug(f'get {block_count:2} blocks for wtid:{wtid:2}#{col}, type:{date_type[col]}')

    return block_list



def get_missing_block_single(wtid, col, cur_missing):
    train = get_train_ex(wtid)
    begin = train[col].loc[:cur_missing].dropna(how='any').index.max() + 1
    end   = train[col].loc[cur_missing:].dropna(how='any').index.min() - 1
    return begin, end



def get_train_sample_list(wtid, col, args):
    feature_list = []

    block = get_blocks()

    train_block = block.loc[(block.wtid == wtid) & (block.col == col) & (block.kind == 'train')]

    missing_block = block.loc[(block.wtid == wtid) & (block.col == col) & (block.kind == 'missing')]

    for missing_length in missing_block['length'].sort_values().values:

        # if args.file_num==1:
        #     cur_windows = round(missing_length * 0.7)
        # else:
        #     cur_windows = missing_length * 2
        at_least_len_for_block = int(10 * missing_length)
        logger.debug(
            f'get_train_feature:file_num={args.file_num}, at_least_len_for_block={at_least_len_for_block},  '
            f'missing_len={missing_length}')
        for index, cur_block in (train_block[train_block['length'] >= at_least_len_for_block]).iterrows():


            train = get_train_feature_multi_file(wtid, col, args.file_num)


            begin, end = cur_block.begin, cur_block.end
            # Get the data without missing
            block = train.loc[begin:end]

            val_feature = block.iloc[-missing_length*4: -missing_length*3]

            train_feature = get_train_df_by_val(train, val_feature, args)

            logger.debug(f'blockid:{index} , train_shape:{train_feature.shape} '
                         f'train_t_sn:{train_feature.time_sn.min()}, {train_feature.time_sn.min()},'
                         f' val_time_sn:{val_feature.time_sn.min()}:{val_feature.time_sn.max()}')



            feature_list.append((train_feature, val_feature, index))
            # logger.debug(f'Train:{train_feature.shape}, Val:{val_feature.shape}')

    return feature_list


def get_train_df_by_val(train,val_feature, args):
    try:
        window_ratio = args.window
        missing_length = len(val_feature)

        cur_windows = max(3, missing_length * window_ratio)
        cur_windows = int(cur_windows)

        val_begin = val_feature.index.min()
        val_end = val_feature.index.max()

        begin = max(val_begin - cur_windows, 0)
        end = val_end + cur_windows
        logger.debug(f'part#1:{begin},{val_begin}')
        logger.debug(f'part#2:{val_end+1},{end+1}')
        part1 = train.iloc[begin : val_begin]
        part2 = train.iloc[val_end+1 : end+1]

        train_feature = pd.concat([part1, part2])

        train_feature = train_feature.dropna(how='any')

        time_gap = max(30, val_feature.time_sn.max() - val_feature.time_sn.min())
        time_begin = val_feature.time_sn.min() - 5 * time_gap
        time_end = val_feature.time_sn.max() + 5 * time_gap
        # Make the train closed to validate
        train_feature = train_feature[(train_feature.time_sn >= time_begin) & (train_feature.time_sn <= time_end)]
        logger.debug(f'train_t_sn:{train_feature.time_sn.min()}, {train_feature.time_sn.min()},'
                     f' val_time_sn:{val_feature.time_sn.min()}:{val_feature.time_sn.max()}')
        logger.debug(f'Range: Train_val: '
                     f'[{part1.index.min()}, {part1.index.max()} ]({len(part1)}) '
                     f'[{val_feature.index.min()}, {val_feature.index.max()}]({len(val_feature)}), '
                     f'[{part2.index.min()}, {part2.index.max()} ]({len(part2)}), cur_windows:{cur_windows}' )

    except Exception  as e:
        logger.exception(f'Can not get train for val block:{val_begin}:{val_end}, {args}')
        logger.exception(e)
        raise e
    if len(train_feature) == 0:
        logger.exception(f'Train feature length is none, for val block:{val_begin}:{val_end}')
        raise Exception(f'Train feature length is none, for val block:{val_begin}:{val_end}')
    return train_feature


#@timed()
def get_submit_feature_by_block_id(blockid, para ):
    cur_block = get_blocks().loc[blockid]
    logger.debug(f'cur_block:\n{cur_block}')

    kind = cur_block['kind']
    col_name = cur_block['col']
    wtid = cur_block['wtid']
    missing_length = cur_block['length']
    begin, end = cur_block.begin, cur_block.end


    submit = get_train_feature_multi_file(wtid, col_name, para.file_num)


    val_feature = submit.loc[begin:end]

    logger.debug(f'wtid:{wtid}, col:{col_name}, file_num:{para.file_num},   blockid:{blockid}')
    logger.debug(f'Train columns:{submit.columns}')

    train_feature = get_train_df_by_val(submit, val_feature, para)

    logger.debug(f'original: {train_feature.shape}, {val_feature.shape}')
    #
    # time_gap = max(30, val_feature.time_sn.max() - val_feature.time_sn.min())
    # time_begin = val_feature.time_sn.min() - 5 * time_gap
    # time_end = val_feature.time_sn.max() + 5 * time_gap
    # # Make the train closed to validate
    # train_feature = train_feature[(train_feature.time_sn >= time_begin) & (train_feature.time_sn <= time_end)]
    #
    # logger.debug(f'new(filter by time): {train_feature.shape}, {val_feature.shape}')
    #
    # logger.debug(f'{kind}:train_feature:{train_feature.columns}')
    # logger.debug(f'{kind}:val_feature:{val_feature.columns}')
    # if len(train_feature) == 0 or len(val_feature) == 0:
    #     logger.exception(f'train_feature#{kind}:{train_feature.shape}, val_feature:{val_feature.shape} for blockid:{blockid}, cur_file_num:{cur_file_num}')
    #     raise Exception('Error when get feature')
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


def convert_enum(df):
    for col in df:
        if col in date_type and 'int' in date_type[col].__name__:
            df[col] = df[col].astype(int)
    return df

#
# @timed()
# def group_columns(wtid=1):
#     col_list = get_blocks().col.drop_duplicates()
#     existing = []
#     gp_list = []
#     for col in col_list:
#         if col in existing:
#             continue
#         gp = get_closed_columns(col, wtid)
#         gp = list(gp.values)
#         existing.extend(gp)
#         gp_list.append(gp)
#     return sorted(gp_list, key=lambda val: len(val), reverse=True )


# @file_cache()
def get_closed_columns(col_name, wtid=1, threshold=0.9):
    sub = get_train_ex(wtid)

    sub = sub.dropna(how='any')

    cor = np.corrcoef(sub.drop(axis='column', columns=['ts', 'wtid']).T)

    col_list = sub.columns[2:]

    #print(cor.shape, sub.shape)

    cor = pd.DataFrame(index=col_list, columns=col_list, data=cor)[col_name]

    return cor.loc[cor >= threshold].sort_values(ascending=False).index


@timed()
@file_cache()
def get_pure_block_list(kind='data'):
    df = pd.DataFrame()
    for wtid in range(1, 34):
        train = get_train_ex(wtid)
        #print(train.shape)
        if kind == 'data':
            train = train.dropna(how='any')
        else:
            col_list = list(date_type.keys())
            col_list.remove('wtid')
            train = train[col_list]
            train = train[train.sum(axis=1) == 0]
        train['old_index'] = train.index
        train = train[['old_index']]
        train['shift_index'] = train.old_index.shift(1)

        train['jump'] = train.apply(lambda row: row.old_index - 1 == row.shift_index, axis=1)

        block_begin = train[train.jump == False]
        for begin, end_ex in zip(block_begin.old_index, block_begin.old_index.shift(-1)):
            end = train.loc[:end_ex - 1].index.max()
            df = df.append({
                'wtid':int(wtid),
                'begin':begin,
                'end':end,
                'length':end-begin+1,
                 }, ignore_index=True)
    df.wtid = df.wtid.astype(int)

    return df

#
# @lru_cache()
# @file_cache()
# def adjust_block(ratio=0.8):
#     block = get_blocks()
#     block['begin_ex'] = block.begin
#     block['end_ex'] = block.end
#
#     data_block = get_pure_block_list(kind='data')
#
#     wtid_list = range(1, 5)
#     for wtid in wtid_list:
#         for index, row in data_block.loc[data_block.wtid == wtid].iterrows():
#             end = row.end
#             length = row.length
#             # logger.info(f"block.loc[(block.begin >= end) & (block.wtid==wtid) , 'begin_ex']={block.loc[(block.begin >= end) & (block.wtid==wtid) , 'begin_ex'].shape}")
#             block.loc[(block.begin >= end) & (block.wtid == wtid), 'begin_ex'] = block.loc[(block.begin >= end) & (
#             block.wtid == wtid), 'begin_ex'] - ratio * length
#             block.loc[(block.begin >= end) & (block.wtid == wtid), 'end_ex'] = block.loc[(block.begin >= end) & (
#             block.wtid == wtid), 'end_ex'] - ratio * length
#     return block.loc[block.wtid.isin(wtid_list)]


def rename_col_for_merge_across_wtid(wtid, col_name, key=None):
    if key is None:
        key = wtid
    train = get_train_ex(wtid)[[col_name, 'time_sn', 'time_slot_7']]
    train.columns = [f'{col}_{key}' if 'var' in col else col for col in train.columns]
    return train


@lru_cache()
@file_cache()
def get_corr_wtid(col_name):
    train = rename_col_for_merge_across_wtid(1, col_name)

    for wtid in range(2, 34):
        train_tmp = rename_col_for_merge_across_wtid(wtid, col_name)
        train = train.merge(train_tmp, on=['time_slot_7'])
        train = train.drop_duplicates('time_slot_7')
        logger.debug(f'col#{col_name}, the shpae after wtid:{wtid} is:{train.shape}')
    train = train.set_index('time_slot_7')

    train = train.dropna(how='any')

    cor = train[[col for col in train.columns if 'var' in col]]
    col_list = cor.columns

    logger.debug(col_list)
    cor = np.corrcoef(cor.T)

    # print(train.shape)
    #
    # print(train.shape, train.index.min(), train.index.max())
    #
    # print(col_list)
    cor = pd.DataFrame(index=col_list, columns=col_list, data=cor)
    logger.debug(cor.where(cor < 0.99).max().to_frame().T)

    logger.debug(cor.where(cor < 0.99).idxmax().to_frame().T)

    return cor


@lru_cache(maxsize=64)
@timed()
def get_train_feature_multi_file(wtid, col, file_num):
    file_num = int(file_num)
    if file_num <1:
        raise Exception(f'file_num should be large then 1, cur file_num is {file_num}')

    cor = get_corr_wtid(col)
    related_wtid_list = cor[f'{col}_{wtid}'].sort_values(ascending=False)[1:file_num]
    logger.info(f'The top#{file_num} files for wtid:{wtid}, col:{col} is '
                f'{dict(zip(related_wtid_list.index,np.round(related_wtid_list.values,3)))}')
    related_wtid_list = [int(col.split('_')[1]) for col in related_wtid_list.index]

    train = rename_col_for_merge_across_wtid(wtid, col)
    input_len = len(train)
    train = train.rename(columns={f'{col}_{wtid}':col})
    train['id']=train.index
    for related_wtid in related_wtid_list:
        train_tmp = rename_col_for_merge_across_wtid(related_wtid, col)
        train_tmp = train_tmp.drop(axis='column', columns=['time_sn'])
        train = train.merge(train_tmp, how='left', on=['time_slot_7'])
        train = train.drop_duplicates(['id'])
    train = train.set_index('id')
    col_list = [col for col in train.columns if 'var' in col]
    col_list.append('time_sn')
    train = train[col_list]
    train.iloc[:, 1:] = train.iloc[:,1:].fillna(method='ffill')

    if len(train) != input_len:
        logger.exception(f"There are some row are missing for wtid:{wtid}, col:{col}, file_num:{file_num}")
        raise Exception(f"There are some row are missing for wtid:{wtid}, col:{col}, file_num:{file_num}")

    return train.sort_index()



if __name__ == '__main__':

    # columns = list(date_type.keys())
    # columns.remove('wtid')
    # columns = sorted(columns)
    # for wtid in sorted(range(1, 34), reverse=True):
    #     for col in columns:
    #         get_result(wtid, col, 100)


    tmp = get_std_all()
    tmp = tmp.groupby(['col', 'data_type']).agg({'mean': 'mean'}).sort_values(('mean'), ascending=False)
    tmp = tmp.reset_index()

    for col_name in tmp.col:
        get_corr_wtid(col_name)




