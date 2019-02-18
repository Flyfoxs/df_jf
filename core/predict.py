from redlock import RedLockError

from core.feature import *
#from core.check import check_options
import fire


def get_predict_fun(train, args):

    col_name = args['col_name']

    is_enum = True if 'int' in date_type[col_name].__name__ else False

    if is_enum:
        fn = lambda val: predict_stable_col(train, val, 0)
    else:
        fn = lambda val : get_cut_predict(train, val, args)

    return fn


def predict_stable_col(train, val, threshold=0.5):
    cur_ratio = train.iloc[:, 0].value_counts().iloc[:2].sum()/len(train)

    if cur_ratio >=  threshold:
        half = len(train)//2

        #TOP value in the begin
        val_1 = train.iloc[:half, 0].value_counts().index[0]
        res_1 = np.ones(len(val)//2)*val_1

        #TOP value in the end
        val_2 = train.iloc[half:, 0].value_counts().index[0]
        res_2 = np.ones(len(val) - (len(val)//2)) * val_2

        return np.hstack((res_1, res_2))
    else:
        logger.exception(f'Cur ration is {cur_ratio}, threshold: {threshold}, {train.columns}')
        raise Exception(f'Train is not stable:{train.columns}')

def get_momenta_value(arr_begin, arr_end):

    avg_begin = arr_begin.mean()
    avg_end   = arr_end.mean()
    if avg_begin not in arr_begin:
        if avg_begin< avg_end:
            arr_begin = sorted(arr_begin)
        else:
            arr_begin = sorted(arr_begin, reverse=True)
        for val in arr_begin:
            if val > avg_begin:
                avg_begin = val
                break

    if avg_end not in arr_end:
        if avg_begin < avg_end:
            arr_end = sorted(arr_end, reverse=True)
        else:
            arr_end = sorted(arr_end)
        for val in arr_end:
            if val < avg_end:
                avg_end = val
                break

    return avg_begin, avg_end


@timed()
def get_cut_predict(train, val, args):
    from sklearn.linear_model import Ridge, LinearRegression

    if len(val) <=5 :
        logger.debug(f'The input val is len:{len(val)}')
        return predict_stable_col(train, val, 0 )

    momenta_col_length = int(args.momenta_col_length)
    momenta_impact_length = max(1, int(args.momenta_impact_ratio*len(val)))

    clf = LinearRegression()
    np.random.seed(0)

    try:
        clf.fit(train.iloc[:, 1:], train.iloc[:, 0])
    except Exception as e:
        logger.error(f'train:{train.shape}, val:{val.shape}:[{val.index.min()}, {val.index.max()}] '
                     f'{train.columns} ({args.time_sn})')
        logger.error(args)
        logger.error(f'{train.shape}, \n{train.head()}')
        raise e

    cut_len = max(min(momenta_impact_length, len(val)//2-1),1)

    block_begin = val.index.min()
    block_end = val.index.max()
    logger.debug(f'val block:[{block_begin}, {block_end}], {val.columns}')
    if train.shape[1] != val.shape[1]+1:
        logger.error(f'train:{train.shape}, val{val.shape}')
        raise Exception(f'Train shape not same with val:{train.shape}, {val.shape}')

    logger.info(f'train:{train.shape}, val{val.shape}')
    begin_val_arr=train.iloc[:, 0].loc[:(block_begin - 1)].tail(momenta_col_length).values
    end_val_arr = train.iloc[:, 0].loc[(block_end + 1):].head(momenta_col_length).values

    begin_val, end_val = get_momenta_value(begin_val_arr, end_val_arr )

    logger.debug(f'====Begin_val:{begin_val}:{begin_val_arr}, end_val:{begin_val}:{end_val_arr},'
                f' predict range:{cut_len}:{len(val)-cut_len}, cut_len:{cut_len} ')


    res = np.hstack((np.ones(cut_len) * begin_val,
                      clf.predict(val.iloc[cut_len:len(val)-cut_len]),
                      np.ones(cut_len) * end_val
                      ))
    col_name = args.col_name
    is_enum = True if 'int' in date_type[col_name].__name__ else False
    if is_enum:
        res = res.astype(int)
    else:
        res = np.round(res,2)
    return res


def _predict_data_block(train_df, val_df, args):
    score_df = pd.DataFrame()
    col_name = str(train_df.columns[0])
    check_fn = get_predict_fun(train_df, args)
    val_res = check_fn(val_df.iloc[:, 1:])

    if pd.notna(val_df.loc[:, col_name]).all():
        is_enum = True if 'int' in date_type[col_name].__name__ else False
        # print(val_df[col_name].shape, val_res.shape)
        cur_count, cur_loss = score(val_df[col_name], val_res, is_enum)
        #print('=====', type(cur_loss), type(cur_count))

        if args.blk_id is not None:
            args.score = round(cur_loss / cur_count, 4)
            args.score_total = cur_loss
            args.score_count = cur_count
            score_df = score_df.append(args, ignore_index=True)
    else:
        raise Exception(f'{col_name} has none in val_df')

    return val_res, score_df



def predict_section(miss_block_id, wtid, col_name, begin, end, args):
    """
    output:resut, score
    """
    train = get_train_feature_multi_file(wtid, col_name, max(10, args.file_num), args.related_col_count)

    val_df = train.loc[begin:end]

    train_df, val_df = get_train_df_by_val(miss_block_id, train, val_df, args.window,
                                   args.drop_threshold, args.time_sn > 0, args.file_num)

    return _predict_data_block(train_df, val_df, args)


def predict_block_id(miss_block_id, arg):
    """
    base on up, down, left block to estimate the score
    output:resut, score
    """
    #print(arg)
    score_df = pd.DataFrame()
    for direct in ['up', 'down', 'left']:
        train_df, val_df, data_blk_id = \
            get_train_val(miss_block_id, arg.file_num, round(arg.window,2),
                          arg.related_col_count, arg.drop_threshold,
                          arg.time_sn, 0, direct)

        arg.blk_id = miss_block_id
        arg.direct = direct
        res, score_df_tmp = _predict_data_block(train_df, val_df, arg)
        score_df = pd.concat([score_df, score_df_tmp])
    logger.info(f'blk:{miss_block_id},  avg:{round(score_df_tmp.score.mean(),4)}, std:{round(score_df_tmp.score.std(),4)}')

    return res, score_df


def estimate_arg(miss_block_id, arg_df):
    """
    Get the best arg for specific blockid
    :param miss_block_id:
    :param arg_df:
    :return:
    """
    score_df = pd.DataFrame()
    for sn, arg in arg_df.iterrows():
        _, score_df_tmp = predict_block_id(miss_block_id, arg)
        score_df = pd.concat([score_df, score_df_tmp])
    score_df = score_df.groupby([model_paras]).agg({'score':['mean','std']})
    score_df.columns = ['_'.join(item) for item in score_df.columns]
    score_df = score_df.sort_values('score_mean', ascending=False).reset_index()
    return score_df


@timed()
def gen_blk_result(miss_block_id, arg_list=None):
    from core.check import get_args_dynamic
    cur_block = get_blocks().loc[miss_block_id]
    if arg_list is None:
        arg_list = get_args_dynamic(cur_block.col, True)
        if len(arg_list) == 0:
            logger.error(f'No dynamc arg is found for blk:{miss_block_id}')
            return 0
        # print('====', arg_list.shape, arg_list)

    #Get Best args
    score_sn = 0
    #logger.info(arg_list)
    logger.info(f'There are {len(arg_list)} args for blk:{miss_block_id}')
    score_list = estimate_arg(miss_block_id, arg_list)
    print(len(score_list), len(arg_list))
    select_arg = score_list[score_sn][-1]
    logger.info(f'The best score for blkid:{miss_block_id}, avg:{score_list[score_sn][0]}, std:{score_list[score_sn][1]}')

    col_name = cur_block['col']
    wtid = cur_block['wtid']
    #missing_length = cur_block['length']
    begin, end = cur_block.begin, cur_block.end

    adjust_file_num = int(max(5, select_arg.file_num))
    train = get_train_feature_multi_file(wtid, col_name, adjust_file_num, int(select_arg.related_col_count))

    sub = train.loc[begin:end]
    train, sub = get_train_df_by_val(miss_block_id, train, sub,
                                select_arg.window,
                                select_arg.drop_threshold,
                                select_arg.time_sn, select_arg.file_num)

    predict_fn = get_predict_fun(train, select_arg)
    predict_res = predict_fn(sub.iloc[:, 1:])
    predict_res = np.round(predict_res, 2)
    predict_res = pd.Series(predict_res, index=sub.index)
    logger.debug(f'sub={sub.shape}, predict_res={predict_res.shape}, type={type(predict_res)}')

    file_csv = f'./output/blocks/{col_name}_{miss_block_id:06}_{score_list[score_sn][0]:.4f}.csv'
    logger.info(f'Result will save to:{file_csv}')
    predict_res.to_csv(file_csv)
    return len(arg_list)


def process_wtid(wtid):
    file = f'./score/blks/{wtid:02}.h5'
    with factory.create_lock(file, ttl=100000*360): #1hour
        try:
            his_df = pd.read_hdf(file, 'his')

        except FileNotFoundError as e:
            his_df = pd.DataFrame()
        latest = his_df.sort_values('ct', ascending=False).iloc[0]

        from datetime import timedelta
        gap = (pd.to_datetime('now') - latest.ct) / timedelta(minutes=1)
        if gap <= 30:  # 30mins
            logger.warning(
                f'Ignore this time for {col_name}, since the server:{latest.server} already save in {round(gap)} mins ago, {latest.ct}')
            return None

        try:
            score_df = pd.read_hdf(file, 'score')
        except Exception as e:
            score_df = pd.DataFrame()

        from core.check import heart_beart
        heart_beart(file, f'Begin with existing:{len(score_df)}')


        blk = get_blocks()
        for top_n in range(0, 24):
            logger.info(f'Status:top_n:{top_n}, wtid:{wtid}')
            for col in get_predict_col():
                tmp = blk.loc[(blk.col == col) & (blk.wtid == wtid) & (blk.kind == 'missing')].sort_values('length',
                                                                                                           ascending=False)
                todo = [int(tmp.index.values[top_n])]
                for blk_id in todo:
                    try:
                        gen_blk_result(blk_id)
                    except Exception as e:
                        logger.exception(e)
                        logger.error(f'Error when process blkid:{blk_id}')
                heart_beart(file, f'Current:{len(score_df)}')

        his_df = heart_beart(file,f'Done, top_n#{top_n}, wtid:{wtid}')
        score_df.to_hdf(file, 'score', mode='w')
        his_df.to_hdf(file, 'his')







def main():

    for wtid in range(1, 34):
        process_wtid(wtid)
#
# class Score_his:
#
#     def __init__(self, wtid, col_name):
#         self.wtid = wtid
#         self.col_name = col_name
#         self.
#         self.df = self.get_lock()
#
#     def get_lock(self):
#         try:
#             with factory.create_lock(self.file):
#                 import socket
#                 host_name = socket.gethostname()
#                 try:
#                     his_df = pd.read_hdf(self.file, 'his')
#
#                 except FileNotFoundError as e:
#                     his_df = pd.DataFrame()
#
#
#                 latest = his_df.sort_values('ct', ascending=False).iloc[0]
#
#                 from datetime import timedelta
#                 gap = (pd.to_datetime('now') - latest.ct) / timedelta(minutes=1)
#                 if gap <= 30: #30mins
#                     logger.warning(
#                         f'Ignore this time for {col_name}, since the server:{latest.server} already save in {round(gap)} mins ago, {latest.ct}')
#                     return None
#                 his_df = his_df.append({'ct': pd.to_datetime('now'), 'server': host_name, 'msg':'dummy'}, ignore_index=True)
#                 his_df.to_hdf(self.file, 'his')
#
#                 try:
#                     df = pd.read_hdf(self.file, 'score')
#                 except FileNotFoundError as e:
#                     df = pd.DataFrame()
#                 return df
#
#         except RedLockError as e:
#             logger.warning(f'Other Process is already lock:{bin_col}')
#             logger.warning(e)
#             return None
#
#
#
#
#     def put(self, paras):
#         self.df = self.df.append(paras, index_ignore=True)
#         if len(self.df)//10 == 0:
#             self.save()
#
#
#     def save(self):
#
#         self.df.to_hdf(self.file, 'score', mode='w')
#         his_df.to_hdf(self.file, 'his')
#


if __name__ == '__main__':
    """
    blkid, direct, paras, score, score_total, score_count
    """
    main()
    #gen_blk_result(106245)
    # get_train_val_range_left(106245, 2.9)


