import sys
sys.path.insert(99, './df_jf')
sys.path.insert(99, '../df_jf')
print(sys.path)

from core.feature import *
import fire


from core.predict import *

@lru_cache()
@timed()
def get_miss_blocks_ex(bin_num=None, gp_name=None, kind ='cut' ):

    if bin_num is None and gp_name is not None:
        file = f'./score/{gp_name}/*'
        bin_num = len(glob(file))
    elif bin_num is None and gp_name is None:
        logger.exception(f'Need input in_num or gp_name')
        raise Exception(f'Need input in_num or gp_name')

    blk_list = []
    blk = get_blocks()
    blk = blk.loc[blk.kind=='missing']
    gp = blk.groupby('col')
    for col_name in gp.groups:
        gp_tmp = gp.get_group(col_name)

        if kind == 'qcut':
            gp_tmp.loc[:,'bin_des'] = pd.qcut(gp_tmp.length, bin_num)
        else:
            gp_tmp.loc[:,'bin_des'] = pd.cut(gp_tmp.length, bin_num)
        gp_tmp.loc[:,'bin_id']  = gp_tmp.bin_des.cat.codes
        blk_list.append(gp_tmp)
    res = pd.concat(blk_list)
    #res.loc[:,'bin_id'] = res.apply(lambda row: f'{row.wtid:02}_{row.bin_id}', axis=1)
    return res

def get_wtid_list_by_bin_id(bin_id, bin_count):
    df = get_miss_blocks_ex(bin_count) #get_wtid_list_by_bin_id
    return df.loc[df.bin_id==bin_id].wtid.drop_duplicates().values

#TODO
@lru_cache(maxsize=32)
@timed()
def get_train_sample_list(bin_id, col, file_num, window, set_list='0', reverse=True):
    arg_loc = locals()

    set_list = set_list.split(',')
    set_list = [int(item) for item in set_list]

    # args = DefaultMunch(None, json.loads(args_json))
    feature_list = []

    block_all = get_blocks()

    bin_num = check_options().bin_count
    block_missing = get_miss_blocks_ex(bin_num) #get_train_sample_list


    for wtid in get_wtid_list_by_bin_id(bin_id,bin_num):
        train_block = block_all.loc[(block_all.wtid == wtid) & (block_all.col == col) & (block_all.kind == 'train')]

        missing_block = block_missing.loc[(block_missing.wtid == wtid) & (block_missing.bin_id == bin_id)
                                          & (block_missing.col == col) & (block_missing.kind == 'missing')]

        train = get_train_feature_multi_file(wtid, col, file_num)

        for missing_length in missing_block['length'].sort_values().values:

            # if args.file_num==1:
            #     cur_windows = round(missing_length * 0.7)
            # else:
            #     cur_windows = missing_length * 2
            at_least_len_for_block = int(10 * missing_length)
            logger.debug(
                f'get_train_feature:file_num={file_num}, at_least_len_for_block={at_least_len_for_block},  '
                f'missing_len/Need_Validate={missing_length}')

            qualified_block = train_block[train_block['length'] >= at_least_len_for_block]

            qualified_block = qualified_block.sort_values('length')

            qualified_block = qualified_block.iloc[set_list,:] #Max is 4

            #
            # if len(qualified_block) <=0 :
            #     logger.exception(f'There is no qualified block was found for missing_length:{missing_length},bin_id:{bin_id}, {col} ')
            #
            # if reverse == 1: #Validate model
            #     sample_count = min(check_options().check_cnt, len(qualified_block))
            #     logger.debug(f'It is in validate model, only pick {sample_count:02} from {len(qualified_block):02} data block, '
            #                 f'for missing:{missing_length:04}, bin_id:{bin_id}, {col}' )
            #     qualified_block = qualified_block.sample(sample_count) #, random_state=1
            # else:
            #     logger.debug(f'Check model, Will generate {len(qualified_block):02} sample for missing:{missing_length:04}, '
            #                 f'bin_id:{bin_id}, {col}')

            for index, cur_block in qualified_block.iterrows():
                begin, end = cur_block.begin, cur_block.end
                # Get the data without missing
                block = train.loc[begin:end]

                #missing_length = max(missing_length,2)

                #logger.info(f'missing_length:{missing_length}')
                if reverse:
                    val_feature = block.iloc[missing_length * -4 : missing_length * -3]
                else:
                    val_feature = block.iloc[missing_length * 3: missing_length * 4]

                #logger.debug(block.head(10))

                #logger.debug(val_feature)


                logger.debug(f'Begin:{begin}, end:{end},{missing_length * 3}/{missing_length * 4}({len(block)})/ block len:{len(block)}, '
                             f'reverse:{reverse}:missing_length/val:{missing_length}')

                train_feature = get_train_df_by_val(train, val_feature, window) #Train

                logger.debug(f'blockid:{index} , train_shape:{train_feature.shape} '
                            f'[{train.index.min()}, [{val_feature.index.min()}, {val_feature.index.max()}], {train.index.max()}]'
                             f'train_t_sn:{train_feature.time_sn.min()}, {train_feature.time_sn.min()},'
                             f' val_time_sn:{val_feature.time_sn.min()}:{val_feature.time_sn.max()}')

                feature_list.append((train_feature, val_feature, index))
                # logger.debug(f'Train:{train_feature.shape}, Val:{val_feature.shape}')
    if len(feature_list) == 0:
        logger.warning(f'Can not find row for:{arg_loc}')

    return feature_list



@timed()
def check_score(args, set_list):
    """

    :param wtid:
    :param col:
    :param args:
        window:[0.5-n]
        momenta_col_length:[1-100]
        momenta_impact_ratio:[100-400]
        input_file_num:[1-n]
        related_col_count:[0-n]
        time_sn:True/False
        class:lr, deploy, gasion
    :param pic:
    :return:
    """
    import matplotlib.pyplot as plt

    bin_id = args['bin_id']
    col = args['col_name']

    train_list = get_train_sample_list(bin_id, col, args.file_num, args.window, set_list)

    count, loss = 0, 0

    for train, val, blockid in train_list :

        is_enum = True if 'int' in date_type[col].__name__ else False
        logger.debug(f'Blockid#{blockid}, train:{train.shape}, val:{val.shape}, file_num:{args.file_num}')
        check_fn = get_predict_fun(train, args)

        # if pic_num:
        #     plt.figure(figsize=(20, 5))
        #     for color, data in zip(['#ff7f0e', '#2ca02c'], [train, val]):
        #         plt.scatter(data.time_sn, data[col], c=color)
        #
        #     x = np.linspace(train.time_sn.min(), train.time_sn.max(), 10000)
        #     plt.plot(x, check_fn(x))
        #     plt.show()

        val_res = check_fn(val.iloc[:, 1:])
        #logger.debug(f'shape of predict output {val_res.shape}, with paras:{local_args}')
        cur_count, cur_loss = score(val[col], val_res, is_enum)

        loss += cur_loss
        count += cur_count
        logger.debug(f'blockid:{blockid}, {train.shape}, {val.shape}, score={round(cur_loss/cur_count,3)}')
    # avg_loss = round(loss/count, 4)

    return loss, count
#
# @lru_cache()
# def get_closed_wtid_list(wtid):
#
#     wtid_count =2
#     col_num = 4
#     s = pd.Series()
#
#     std = get_std_all()
#     col_list = std.loc[std.wtid == wtid].sort_values('mean', ascending=False).col.head(col_num).values
#     for col_name in col_list:
#         cor = get_corr_wtid(col_name)
#         #print(cor.columns)
#         related_wtid_list = cor[f'{col_name}_{wtid}'].sort_values(ascending=False)[:wtid_count]
#         related_wtid_list = [ int(item.split('_')[1]) for item in related_wtid_list.index.values]
#         #print('===', sorted(related_wtid_list))
#         s = s.append(pd.Series(related_wtid_list), ignore_index=True)
#     s = s.value_counts()
#     closed_list = list(s[s>=col_num].index)
#     return sorted(closed_list)

def summary_all_best_score(wtid_list=[-1], top_n=0, **kwargs):
    df = pd.DataFrame()
    for col in get_predict_col():
        gp_name = check_options().gp_name
        df = df.append(get_best_para(gp_name, col, wtid_list, top_n, **kwargs), ignore_index=True) # summary_all_best_score

    df['data_type'] = df.col_name.apply(lambda val: date_type[val].__name__)

    return df.sort_values('score').reset_index(drop=True)


@timed()
def check_score_all():

    #from multiprocessing.dummy import Pool as ThreadPool #线程
    from multiprocessing import Pool as ThreadPool  # 进程

    logger.info(f"Start a poll with size:{check_options().thread}")
    pool = ThreadPool(check_options().thread)

    #summary = summary_all_best_score()

    bin_count = check_options().bin_count
    import itertools
    bin_col_list = itertools.product(range(0, bin_count), get_predict_col(), )

    try:
        pool.map(check_score_column, bin_col_list, chunksize=1)
    except Exception as e:
        logger.exception(e)
        os._exit(9)


def get_args_mini(col_name, para_name, top_n=3):
    tmp = merge_score_col(col_name, [1,2,3]) #get_mini_args

    tmp = tmp.groupby(para_name).agg({'score':['max', 'mean']})
    tmp.columns = ['_'.join(items) for items in tmp.columns]
    tmp = tmp.sort_values('score_mean', ascending=False)

    # wtid_list = [
    #     '1,2,3',
    #     '1',
    #     '2',
    #     '3',
    #     '4',
    #     '1,2,3,4',
    #     '30',
    #     '31',
    #     '32',
    #     '33'
    # ]
    args = set(tmp.index.values[:top_n])
    # for wtid_tmp in wtid_list:
    #     best = get_best_para(col_name, wtid_tmp, 0)[para_name]
    #     if best not in args:
    #         logger.warning(f'The best arg:{best} is not found from {args}, which base on {col_name}, {para_name}, {top_n}')
    #         args.add(best)
    return args

@lru_cache()
def get_window(col_name):
    if check_options().mini > 0:
        return get_args_mini(col_name, 'window', check_options().mini)
    return np.arange(0.1, 3, 0.4)

@lru_cache()
def get_momenta_col_length(col_name):
    if check_options().mini > 0:
        return get_args_mini(col_name, 'momenta_col_length', 2)

    is_enum = True if 'int' in date_type[col_name].__name__ else False
    if is_enum:
        return [1]
    else:
        return [1,2,3,4]

@lru_cache()
def get_momenta_impact_ratio(col_name):
    # if check_options().mini > 0:
    #     return get_args_mini(col_name, 'momenta_impact_length', check_options().mini)

    is_enum = True if 'int' in date_type[col_name].__name__ else False
    if is_enum:
        return [1]
    else:
        return [0.1,0.2,0.3,0.4,0.5]

def get_time_sn(col_name):
    is_enum = True if 'int' in date_type[col_name].__name__ else False
    if is_enum:
        return [False]
    else:
        return [True, False]

@lru_cache()
def get_file_num(col_name):
    if check_options().mini > 0:
        return get_args_mini(col_name, 'file_num', check_options().mini)
    is_enum = True if 'int' in date_type[col_name].__name__ else False
    if is_enum:
        return [1]
    else:
        return range(1,10)

def check_exising_his(score_file):
    try:
        with pd.HDFStore(score_file) as store:
            key_list = store.keys()
            if '/his' in key_list and  len(store['his']) > 0:
                return True
            else:
                return False
    except Exception as e:
        path = os.path.dirname(score_file)
        os.makedirs(path, exist_ok=True)
        return False


def heart_beart(score_file, msg):
    path = os.path.dirname (score_file)
    os.makedirs(path, exist_ok=True)

    # Heart beat
    import socket
    host_name = socket.gethostname()
    if check_exising_his(score_file):
        his_df = pd.read_hdf(score_file, '/his')
    else:
        his_df = pd.DataFrame()

    his_df = his_df.append({'ct': pd.to_datetime('now'), 'server': host_name, 'msg':msg}, ignore_index=True)
    his_df.to_hdf(score_file, 'his')

    return his_df

def check_existing(df, args):
    logger.debug(f'check_existing, df:{df.columns}')
    if len(df)==0:
        return False
    tmp= df.loc[
           (df.wtid == args.wtid) &
           (df.file_num == args.file_num) &
           (df.window == args.window) &
           (df.momenta_col_length == args.momenta_col_length) &
           (df.momenta_impact_ratio == args.momenta_impact_ratio) &
           (df.related_col_count == args.related_col_count) &
           (df.drop_threshold == args.drop_threshold) &
           (df.time_sn == args.time_sn) &
           (df.drop_threshold == args.drop_threshold) &
           (df.class_name == args.class_name)
            ]
    if len(tmp)==0:
        return False
    elif len(tmp)==1:
        logger.info(f'Already existing score for:{args}')
        return True
    else:
        logger.exception(f'Exception status for len:{len(tmp)} {args}')
        raise Exception(f'Exception status for len:{len(tmp)} {args}')





@timed()
def check_score_column(bin_col):
    bin_id , col_name = bin_col

    gp_name = check_options().gp_name
    score_file = f'./score/{gp_name}/{bin_id:02}/{col_name}.h5'
    if check_exising_his(score_file):
        his_df = pd.read_hdf(score_file,'/his')
        latest = his_df.sort_values('ct', ascending=False).iloc[0]

        from datetime import timedelta
        gap = (pd.to_datetime('now') - latest.ct) / timedelta(minutes=1)
        if gap <= check_options().check_gap:
            logger.warning(f'Ignore this time for {col_name}, since the server:{latest.server} already save in {round(gap)} mins ago, {latest.ct}')
            return None
        else:
            logger.info(f'Last time is @{col_name} at {latest.ct}')
    heart_beart(score_file, f'dummpy for:{col_name}, bin_id:{bin_id}')

    # model = check_options().model
    try:
        score_df = pd.read_hdf(score_file,'score')
    except Exception as e:
        logger.info(f'No existing score is found for :{col_name} bin_id:{bin_id}')
        score_df = pd.DataFrame()

    processed_count = 0

    arg_list = get_args_missing(col_name, bin_id)

    logger.info(f'Mini:{check_options().mini}, Todo:{len(arg_list)} Current sample:{score_df.shape}, {col_name},bin_id:{bin_id}' )

    heart_beart(score_file, f'Existing:{len(score_df)}, todo:{len(arg_list)}, type:{date_type[col_name].__name__}')

    for sn, args in arg_list.iterrows():
        try:
            score, count = check_score(args, set_list = '0')
        except Exception as e:
            logger.exception(e)
            os._exit(2)
        args['score'] = round(score/count, 4) if count else 0
        args['score_total'] = score
        args['score_count'] = count
        args['ct'] = pd.to_datetime('now')

        score_df = score_df.append(args, ignore_index=True)
        logger.info(f'Current df:{score_df.shape}, last score is {score:.4f} wtih:\n{args}')

        processed_count += 1

        if processed_count % 100 ==0:
            score_df.to_hdf(score_file, 'score')
            heart_beart(score_file, f'processed:{processed_count}/current:{len(score_df)}, type:{date_type[col_name].__name__}')

    his_df = heart_beart(score_file, f'Done:{processed_count}/current:{len(score_df)}, type:{date_type[col_name].__name__}')

    score_df.ct = score_df.ct.astype('str')
    score_df = score_df.sort_values('ct',ascending=False)
    len_with_dup = len(score_df)
    score_df = score_df.drop_duplicates(model_paras)
    remove_len = len_with_dup - len(score_df)
    if remove_len > 0:
        logger.warning(f'There are {remove_len} records are removed, current len is:{len(score_df)}, old:{len_with_dup}')
    score_df.to_hdf(score_file, 'score', mode='w')
    his_df.to_hdf(score_file, 'his')

    logger.info(f'There are {processed_count} process for {col_name}, Current total:{len(score_df)}')


def get_args_all(col_name):
    df = pd.DataFrame()
    arg_list = []
    window = 0.7
    momenta_col_length = 1
    momenta_impact_ratio = 0.1
    time_sn = True
    related_col_count = 0
    drop_threshold = 1
    class_name = 'lr'
    for window in get_window(col_name):
        window = round(window, 1)
        for momenta_col_length in get_momenta_col_length(col_name):
            for momenta_impact_ratio in get_momenta_impact_ratio(col_name):
                for time_sn in get_time_sn(col_name):
                    for file_num in get_file_num(col_name):
                        args = {
                                'col_name': col_name,
                                'file_num': file_num,
                                'window': window,
                                'momenta_col_length': momenta_col_length,
                                'momenta_impact_ratio': momenta_impact_ratio,
                                'related_col_count': related_col_count,
                                'drop_threshold': drop_threshold,
                                'time_sn': time_sn,
                                'class_name': class_name,
                                }
                        # args = DefaultMunch(None, args)
                        # arg_list.append(args)
                        df = df.append(args,ignore_index=True)

    return df #fill_ext_arg(df, col_name)
#
# def fill_ext_arg(df, col_name):
#     if check_options().mini < 1:
#         return df
#     col_list = df.columns
#     old_len = len(df)
#
#     wtid_list = [
#         '1,2,3',
#         '1,2,3,4',
#         '1,2,3,4,5',
#         '30,31, 32, 33',
#         '1,2,3,4,  30,31, 32, 33',
#         '1,2,3,4,5,30,31, 32, 33',
#         '1',
#         '2',
#         '3',
#         '4',
#         '30',
#         '31',
#         '32',
#         '33'
#     ]
#     for wtid_tmp in wtid_list:
#         gp_name = check_options().gp_name
#         best = get_best_para(gp_name, col_name, wtid_tmp, 0)
#         df = df.append(best)
#
#     df = df.drop_duplicates(col_list)
#
#     logger.info(f'There are {len(df)-old_len} args add to list({old_len}) for col_name:{col_name}, original_df:{col_list}')
#     return df[col_list]


@lru_cache()
@timed()
def get_args_missing(col_name, bin_id):
    todo = get_args_all(col_name)

    gp_name = check_options().gp_name
    score_file = f'./score/{gp_name}/{bin_id:02}/{col_name}.h5'
    try:
        base = pd.read_hdf(score_file, 'score')
        original_len = len(base)
    except Exception as e:
        logger.info(f'It is a new task for {col_name}, todo_bin_id:{bin_id}')
        original_len = 0


    if original_len == 0 :
        logger.info(f'No data is found from file:{score_file}, todo:{todo.shape}')
        todo['bin_id'] = bin_id
        return todo


    # print(col_list)

    #base = base.rename(columns={'ct': 'ct_old', 'wtid': 'wtid_old'})

    todo = todo.merge(base, how='left', on=model_paras)
    todo = todo.loc[pd.isna(todo.ct) & pd.isna(todo.wtid)][model_paras].reset_index(drop=True)
    todo['bin_id'] = int(bin_id)
    logger.info(f'Have {len(todo)} todo, original:{original_len}, {col_name}, bin_id:{bin_id}')
    return todo


@lru_cache()
def check_options():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--bin_count", type=int, default=10, help="How many bins will split for each column")
    #parser.add_argument("--bin_id", type=int, default=10)
    parser.add_argument("--check_gap", type=int, default=10, help="Mins to lock the score file")
    parser.add_argument("--gp_name", type=str, default='lr_bin', help="The folder name to save score")
    parser.add_argument("--set_list", type=str, default='0', required=True, help="The folder name to save score")

    parser.add_argument("-D", '--debug', action='store_true', default=False)
    parser.add_argument("-W", '--warning', action='store_true', default=False)
    parser.add_argument("-L", '--log', action='store_true', default=False)

    if local:
        thread_num = 1
    else:
        thread_num = 8
    parser.add_argument("--thread", type=int, default=thread_num)
    parser.add_argument('--mini', type=int, default=3, help='enable the Mini model' )
    #parser.add_argument("--check_cnt", type=int, default=3, help='How many sample need to generate for each missing block')

    # parser.add_argument('--model', type=str, default='missing', help='missing, new')


    # parser.add_argument("--version", help="check version", type=str, default='lg')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.warning:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    if args.log:
        import socket
        host_name = socket.gethostname()[-1:]
        file = f'score_{host_name}.log'

        handler = logging.FileHandler(file, 'a')
        handler.setFormatter(format)
        logger.addHandler(handler)
        logger.info(f'Save the log to file:{file}')

    return args


#@lru_cache()
def merge_score_col(col_name, wtid_list):
    """
    Merge the score together for one col_name, and multiply wtid

    :param col_name:
    :param wtid_list:
    :return:
    """
    import os
    df_list = []
    from glob import glob
    # ./score/lr/wtid/column.h5


    for wtid in wtid_list:
        wtid = f'{wtid:02}' if wtid is not None and wtid > 0 else '*'
        match = f"./score/lr/{wtid}/*.h5"
        logger.info(f'Match:{match}')
        for file_name in sorted(glob(match)):
            #logger.debug(f'get file_name:{file_name} with {match}')
            if col_name in file_name:
                tmp_df = pd.read_hdf(file_name , 'score')
                #TODO
                #tmp_df = tmp_df.loc[tmp_df.file_num % 2 == 1]
                df_list.append(tmp_df)
        # for file_name in sorted(glob("./score/*.h5")):
        #     if col_name in file_name:
        #         tmp_df = pd.read_hdf(file_name)
        #         df_list.append(tmp_df)
    all = pd.concat(df_list)
    all = all.drop_duplicates(model_paras)
    logger.debug(f'There are {len(df_list)} score files for {col_name}, wtid:{wtid_list}')
    return all



@lru_cache()
@timed()
def get_best_para(gp_name, col_name, bin_id, top_n=0, **kwargs):
    score_file = f'./score/{gp_name}/{bin_id:02}/{col_name}.h5'
    tmp = pd.read_hdf(score_file, 'score')  # get_best_score
    for k, v in kwargs.items():
        tmp_check = tmp.loc[tmp[k]<=v]
        if len(tmp_check) == 0 :
            logger.exception(f'Cannot find value for {k}={v}, column={col_name}, it might be enum type ')
        else:
            tmp = tmp_check


    # tmp = tmp.groupby(model_paras).agg({'score':['mean', 'count','min', 'max']}).reset_index()
    # tmp.columns = ['_'.join(item) if item[1] else item[0] for item in tmp.columns]
    tmp = tmp.sort_values(['score', 'window','momenta_impact_ratio'], ascending=False)
    tmp = tmp.rename(columns={'score_mean':'score'})

    return tmp.iloc[int(top_n)]


def score(val1, val2, enum=False):
    loss = 0
    for real, predict in zip(val1, val2):
        if enum:
            loss += 1 if real == predict else 0
        else:
            loss += np.exp(-100 * abs(real - np.round(predict,2)) / max(abs(real), 1e-15))
    return len(val1), round(loss, 4)


if __name__ == '__main__':
    args = check_options()
    logger.info(f'Program with:{args} ')

    check_score_all()

    """
    python ./core/check.py -L  --bin_count 8 --gp_name lr_bin_8 > debug.log 2>&1 &
    """