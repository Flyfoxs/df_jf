from redlock import RedLockError


from core.feature import *
#from core.check import check_options
import fire
from core.db import *

def get_predict_fun(train, args):

    message = f'feature for blk#{args.blk_id},'\
                f'is{train.shape}:{list(train.columns)}'\
                f'args:{DefaultMunch(None, args)}'
    message = message.replace("'",'')
    message = message.replace('DefaultMunch(None,', '')
    message = message.replace(': ', ':')
    logger.info(message)

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
        if half == 0:
            res_1 =[]
        else:
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

    if len(val) <=5 :
        logger.debug(f'The input val is len:{len(val)}')
        return predict_stable_col(train, val, 0 )

    momenta_col_length = int(args.momenta_col_length)
    momenta_impact_length = max(1, int(args.momenta_impact * len(val)))
    np.random.seed(0)
    clf = get_clf(args)

    try:
        clf.fit(train.iloc[:, 1:], train.iloc[:, 0])
    except Exception as e:
        logger.error(f'train:{train.shape}, val:{val.shape}:[{val.index.min()}, {val.index.max()}] '
                     f'{train.columns} ({args.time_sn})')
        logger.error(args)
        logger.error(f'{train.shape}, \n{train.head()}')
        raise e

    cut_len = int(max(min(momenta_impact_length, len(val)//2-1),1))

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

    logger.info(f'====Begin_val:{begin_val}:{begin_val_arr}, end_val:{begin_val}:{end_val_arr},'
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


def get_clf (args):
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    if args.class_name =='lr':
        return LinearRegression()
    elif args.class_name == 'rf':
        return RandomForestRegressor(n_estimators=int(args.n_estimators), \
                                     max_depth= int(args.max_depth), \
                                     random_state=0)




def _predict_data_block(train_df, val_df, arg):
    col_name = str(train_df.columns[0])
    check_fn = get_predict_fun(train_df, arg)
    val_res = check_fn(val_df.iloc[:, 1:])

    if pd.notna(val_df.loc[:, col_name]).all():
        is_enum = True if 'int' in date_type[col_name].__name__ else False
        # print(val_df[col_name].shape, val_res.shape)
        cur_count, cur_loss = score(val_df[col_name], val_res, is_enum)
        #print('=====', type(cur_loss), type(cur_count))
        logger.info(f'{is_enum},{cur_loss}, {cur_count}, {arg}, ')
        if arg.blk_id is not None:
            arg['score'] = round(cur_loss / cur_count, 4)
            arg['score_total'] = cur_loss
            arg['score_count'] = cur_count
            insert(arg)
            #score_df = score_df.append(args, ignore_index=True)
    else:
        logger.error(f'blk_id:{arg.blk_id},{col_name}....\n{val_df.loc[:, col_name]}')
        raise Exception(f'{col_name} has none in val_df for blk_id:{arg.blk_id}, args{replace_useless_mark(arg)}')

    return val_res, arg



# def predict_section(miss_block_id, wtid, col_name, begin, end, args):
#     """
#     output:resut, score
#     """
#     train = get_train_feature_multi_file(wtid, col_name, max(10, args.file_num), args.related_col_count)
#
#     val_df = train.loc[begin:end]
#
#     train_df, val_df = get_train_df_by_val(miss_block_id, train, val_df, args.window,
#                                    args.drop_threshold, args.time_sn > 0, args.file_num)
#
#     args['blk_id']=miss_block_id
#     return _predict_data_block(train_df, val_df, args) #predict_section


@timed()
def predict_block_id(miss_block_id, arg):
    """
    base on up, down, left block to estimate the score
    output:resut, score
    """

    train_df, val_df, data_blk_id = \
        get_train_val(miss_block_id, arg.file_num, round(arg.window,2),
                      arg.related_col_count, arg.drop_threshold,
                      arg.time_sn, 0, arg.direct, model=0)
    if data_blk_id<0:
        logger.warning(f'Can not find closed block for :{replace_useless_mark(arg)}')
        return None
    arg['blk_id'] = miss_block_id
    res, args_score = _predict_data_block(train_df, val_df, arg) #predict_block_id


    #logger.info(f'blk:{miss_block_id},  avg:{round(score_df_tmp.score.mean(),4)}, std:{round(score_df_tmp.score.std(),4)}')

    return args_score


def estimate_arg(miss_block_id, arg_df):
    """
    Get the best arg for specific blockid
    :param miss_block_id:
    :param arg_df:
    :return:
    """
    score_list = pd.DataFrame()
    for sn, arg in arg_df.iterrows():
        score_arg = predict_block_id(miss_block_id, arg)
        score_list = score_list.append(score_arg, ignore_index=True)

    return score_list


@timed()
def gen_best_sub(best_arg):
    miss_block_id=best_arg.blk_id
    cur_block = get_blocks().loc[best_arg.blk_id]

    score_avg = best_arg["score_mean"]
    score_std = best_arg["score_std"]

    logger.info(f'The select score for blkid:{miss_block_id}, '
                f'avg:{score_avg}, '
                f'std:{score_std},')

    col_name = cur_block['col']
    wtid = cur_block['wtid']
    begin, end = cur_block.begin, cur_block.end

    adjust_file_num = int(max(10, best_arg.file_num))
    train = get_train_feature_multi_file(wtid, col_name, adjust_file_num, int(best_arg.related_col_count))

    sub = train.loc[begin:end]
    train, sub = get_train_df_by_val(miss_block_id, train, sub,
                                     best_arg.window,
                                     best_arg.drop_threshold,
                                     best_arg.time_sn, best_arg.file_num, model=3)


    predict_fn = get_predict_fun(train, best_arg)
    predict_res = predict_fn(sub.iloc[:, 1:])
    predict_res = np.round(predict_res, 2)
    predict_res = pd.Series(predict_res, index=sub.index)
    logger.debug(f'sub={sub.shape}, predict_res={predict_res.shape}, type={type(predict_res)}')

    file_csv = f'./output/blocks/{col_name}_{miss_block_id:06}_{score_avg:.4f}_{score_std:.4f}.csv'
    logger.info(f'Result will save to:{file_csv}')
    predict_res.to_csv(file_csv)
    return score_avg


@timed()
def process_blk_id(bin_col):
    bin_id, col_name = bin_col
    from core.check import check_options, get_miss_blocks_ex

    #wtid = cur_block.wtid
    class_name = check_options().class_name
    reuse_existing = False
    lock_mins = 10
    try:
        with factory.create_lock(bin_col, ttl=1000*60 *lock_mins):
                try:

                    is_continue = check_last_time_by_binid(bin_id,col_name,lock_mins)
                    if not is_continue:
                        logger.warning(f'The binid#{bin_col} is still in processed in {lock_mins} mins')
                        return 'In processing'
                    from core.check import get_args_all, get_args_extend

                    todo = get_args_all(col_name)
                    best = get_best_arg_by_blk(bin_id, col_name, class_name)
                    if best is not None and len(best) > 0 : # and cur_block.length > 10:
                        extend_args = get_args_extend(best)
                        todo = pd.concat([todo, extend_args])

                    arg_list = get_args_missing_by_blk(todo, bin_id, col_name)
                    if len(arg_list) == 0:
                        logger.warning(f'No missing arg is found from todo:{len(todo)} for blk:{blk_id}')
                        return 0

                    score_list_binid = []
                    #Estimate by blk_list
                    miss = get_miss_blocks_ex()
                    miss = miss.loc[(miss.col==col_name ) & (miss.bin_id==bin_id)]
                    for sn, blk_id in enumerate(list(miss.index)):

                        blk = get_blocks()
                        cur_block = blk.iloc[blk_id]

                        arg_list['bin_id'] = bin_id
                        arg_list['blk_id'] = blk_id
                        arg_list['wtid'] = cur_block.wtid
                        arg_list['direct'] = 'down'

                        # logger.info(arg_list)
                        logger.info(f'There are {len(arg_list):02} args for bin:{bin_col}, blk:{blk_id:06},{sn:03}/{len(miss):03}')
                        score_list = estimate_arg(blk_id, arg_list)
                        score_list_binid.append(score_list)
                    return pd.concat(score_list_binid)
                except Exception as e:
                    logger.exception(e)
                    logger.error(f'Error when process blkid:{blk_id}')
                #his_df = heart_beart(file,f'Done, {len(score_df)}, top_n#{top_n}, wtid:{wtid}')
                #score_df.to_hdf(file, 'score', mode='w')
                #his_df.to_hdf(file, 'his')
    except RedLockError as e:
        logger.info(f'Not get the lock for :{bin_col}')
        return 'No Lock'


@timed()
def main():
    from core.check import get_miss_blocks_ex

    from multiprocessing import Pool as ThreadPool  # 进程


    imp_list =  ['var042', 'var046', 'var004', 'var027', 'var034', 'var043', 'var068', 'var003', 'var052', 'var040', 'var056', 'var024']

    blk_list = get_miss_blocks_ex()
    blk_list = blk_list.loc[#(blk_list.kind=='missing') &
                            #(blk_list.wtid==1) &
                            (blk_list.col.isin(imp_list))] #'var004',

    para_list = set([])
    for sn, row in blk_list.iterrows():
        para_list.add((row.bin_id, row.col))

    try:
        pool = ThreadPool(10)
        logger.info(f'There are {len(para_list)} para need to process')
        pool.map(process_blk_id, para_list, chunksize=np.random.randint(1,64))

    except Exception as e:
        logger.exception(e)
        os._exit(9)




if __name__ == '__main__':
    """
    blkid, direct, paras, score, score_total, score_count
    """
    main()
    #gen_blk_result(106245)
    # get_train_val_range_left(106245, 2.9)

    """
    nohup python ./core/predict.py > predict_2.log 2>&1 &
    """


