import sys
sys.path.insert(99, './df_jf')
sys.path.insert(99, '../df_jf')
print(sys.path)

from core.feature import *
import fire


from core.predict import *

@timed()
def check_score(args, reverse):
    """

    :param wtid:
    :param col:
    :param args:
        window:[0.5-n]
        momenta_col_length:[1-100]
        momenta_impact_length:[100-400]
        input_file_num:[1-n]
        related_col_count:[0-n]
        time_sn:True/False
        class:lr, deploy, gasion
    :param pic:
    :return:
    """
    import matplotlib.pyplot as plt

    wtid = args['wtid']
    col = args['col_name']

    train_list = get_train_sample_list(wtid, col, args.file_num, args.window, reverse)

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
    avg_loss = round(loss/count, 4)

    return avg_loss


def summary_all_best_score(wtid_list=[-1], top_n=0, **kwargs):
    df = pd.DataFrame()
    for col in get_predict_col():
        df = df.append(get_best_para(col, wtid_list, top_n, **kwargs), ignore_index=True)

    df['data_type'] = df.col_name.apply(lambda val: date_type[val].__name__)

    return df.sort_values('score').reset_index(drop=True)


@timed()
def check_score_all():

    #from multiprocessing.dummy import Pool as ThreadPool #线程
    from multiprocessing import Pool as ThreadPool  # 进程

    logger.info(f"Start a poll with size:{check_options().thread}")
    pool = ThreadPool(check_options().thread)

    #summary = summary_all_best_score()
    col_list = get_predict_col()

    pool.map(check_score_column, col_list)

    logger.debug(f'It is done for {check_options().wtid}')

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
def get_momenta_impact_length(col_name):
    if check_options().mini > 0:
        return get_args_mini(col_name, 'momenta_impact_length', check_options().mini)

    is_enum = True if 'int' in date_type[col_name].__name__ else False
    if is_enum:
        return [0]
    else:
        return [100, 200, 300,400, 500, 600]

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
           (df.momenta_impact_length == args.momenta_impact_length) &
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
def check_score_column(col_name):
    wtid = check_options().wtid
    class_name = 'lr'
    score_file = f'./score/{class_name}/{wtid:02}/{col_name}.h5'
    if check_exising_his(score_file):
        his_df = pd.read_hdf(score_file,'/his')
        latest = his_df.sort_values('ct', ascending=False).iloc[0]

        from datetime import timedelta
        gap = (pd.to_datetime('now') - latest.ct) / timedelta(minutes=1)
        if gap <= check_options().check_gap:
            logger.warning(f'For {col_name}, The server:{latest.server} already save in {round(gap)} mins ago, {latest.ct}')
            return None
        else:
            logger.info(f'Last time is @{col_name} at {latest.ct}')

    model = check_options().model
    try:
        score_df = pd.read_hdf(score_file,'score')
    except Exception as e:
        score_df = pd.DataFrame()

    processed_count = 0

    arg_list = get_args_missing(col_name, todo_wtid=wtid)

    logger.info(f'Model:{model}, mini:{check_options().mini}, Current sample:{score_df.shape}, {col_name},wtid:{wtid}' )

    heart_beart(score_file, f'begin with model:{model}, existing:{len(score_df)}, todo:{len(arg_list)}, type:{date_type[col_name].__name__}')

    for sn, args in arg_list.iterrows():

        score = check_score(args, reverse = -1)
        logger.debug(f'Current score is{score:.4f} wtih:{args}')
        args['score'] = score
        args['ct'] = pd.to_datetime('now')

        score_df = score_df.append(args, ignore_index=True)
        processed_count += 1

        if processed_count % 100 ==0:
            score_df.to_hdf(score_file, 'score')
            his_df = heart_beart(score_file, f'processed:{processed_count}/current:{len(score_df)}, type:{date_type[col_name].__name__}')

    his_df = heart_beart(score_file, f'Done:{processed_count}/current:{len(score_df)}, type:{date_type[col_name].__name__}')

    score_df.to_hdf(score_file, 'score', mode='w')
    his_df.to_hdf(score_file, 'his')

    logger.info(f'There are {processed_count} process for {col_name}, total:{len(score_df)}')


def get_args_all(col_name):
    df = pd.DataFrame()
    arg_list = []
    wtid = check_options().wtid
    window = 0.7
    momenta_col_length = 1
    momenta_impact_length = 300
    time_sn = True
    related_col_count = 0
    drop_threshold = 1
    class_name = 'lr'
    for window in get_window(col_name):
        window = round(window, 1)
        for momenta_col_length in get_momenta_col_length(col_name):
            for momenta_impact_length in get_momenta_impact_length(col_name):
                for time_sn in get_time_sn(col_name):
                    for file_num in get_file_num(col_name):
                        args = {#'wtid': wtid,
                                'col_name': col_name,
                                'file_num': file_num,
                                'window': window,
                                'momenta_col_length': momenta_col_length,
                                'momenta_impact_length': momenta_impact_length,
                                'related_col_count': related_col_count,
                                'drop_threshold': drop_threshold,
                                'time_sn': time_sn,
                                'class_name': class_name,
                                }
                        # args = DefaultMunch(None, args)
                        # arg_list.append(args)
                        df = df.append(args,ignore_index=True)

    return fill_ext_arg(df, col_name)

def fill_ext_arg(df, col_name):
    if check_options().mini < 1:
        return df
    col_list = df.columns
    old_len = len(df)

    wtid_list = [
        '1,2,3',
        '1,2,3,4',
        '1,2,3,4,5',
        '30,31, 32, 33',
        '1,2,3,4,  30,31, 32, 33',
        '1,2,3,4,5,30,31, 32, 33',
        '1',
        '2',
        '3',
        '4',
        '30',
        '31',
        '32',
        '33'
    ]
    for wtid_tmp in wtid_list:
        best = get_best_para(col_name, wtid_tmp, 0)
        df = df.append(best)

    df = df.drop_duplicates(col_list)

    logger.info(f'There are {len(df)-old_len} args add to list({old_len}) for col_name:{col_name}')
    return df[col_list]


def get_args_missing(col_name, todo_wtid):
    base = get_args_all(col_name)

    original_len = len(base)

    #base = pd.read_hdf(f'./score/lr/{base_wtid:02}/{col_name}.h5', 'score')
    try:
        todo = pd.read_hdf(f'./score/lr/{todo_wtid:02}/{col_name}.h5', 'score')
    except Exception as e:
        logger.debug(f'It is a new task for {col_name}, wtid:{todo_wtid}')
        base['wtid'] = todo_wtid
        return base

    col_list = list(base.columns.values)

    # print(col_list)

    #base = base.rename(columns={'ct': 'ct_old', 'wtid': 'wtid_old'})

    base = base.merge(todo, how='left', on=col_list)
    todo = base.loc[pd.isna(base.ct)][col_list].reset_index(drop=True)
    todo['wtid'] = int(todo_wtid)
    logger.info(f'Have {len(todo)} missing rows need todo, total:{original_len} for {col_name}, wtid:{todo_wtid}')
    return todo


@lru_cache()
def check_options():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--wtid", type=int, default=1)
    parser.add_argument("--check_gap", type=int, default=15)

    parser.add_argument("-D", '--debug', action='store_true', default=False)
    parser.add_argument("-W", '--warning', action='store_true', default=False)
    parser.add_argument("-L", '--log', action='store_true', default=False)
    parser.add_argument("--thread", type=int, default=8)
    parser.add_argument('--mini', type=int, default=3, help='enable the Mini model' )

    parser.add_argument('--model', type=str, default='missing', help='missing, new')


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
        file = f'score_{args.wtid:02}_{host_name}.log'

        handler = logging.FileHandler(file, 'a')
        handler.setFormatter(format)
        logger.addHandler(handler)
        logger.info(f'Save the log to file:{file}')

    return args


#@lru_cache()
def merge_score_col(col_name, wtid_list):
    import os
    df_list = []
    from glob import glob
    # ./score/lr/wtid/column.h5


    for wtid in wtid_list:
        wtid = f'{wtid:02}' if wtid is not None and wtid > 0 else '*'
        match = f"./score/*/{wtid}/*.h5"
        #logger.debug(f'Match:{match}')
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
    logger.debug(f'There are {len(df_list)} score files for {col_name}, wtid:{wtid_list}')
    return all


@timed()
@lru_cache()
def get_best_para(col_name, wtid_list=[-1], top_n=0, **kwargs):
    if isinstance(wtid_list, str):
        logger.info(f'wtid_list:{wtid_list}')
        wtid_list = wtid_list.split(',')
        logger.info(f'wtid_list:{wtid_list}')
        wtid_list = [int(item) for item in wtid_list if item ]

    tmp = merge_score_col(col_name, wtid_list) # get_best_score
    for k, v in kwargs.items():
        tmp_check = tmp.loc[tmp[k]<=v]
        if len(tmp_check) == 0 :
            logger.exception(f'Cannot find value for {k}={v}, column={col_name}, it might be enum type ')
        else:
            tmp = tmp_check

    col_list = tmp.columns
    col_list = col_list.drop('score')
    col_list = col_list.drop('wtid')
    col_list = col_list.drop('ct')
    #col_list = col_list.drop('col_name')
    #print(col_list)
    tmp = tmp.groupby(list(col_list)).agg({'score':['mean', 'count','min', 'max'], 'wtid':'nunique'}).reset_index()
    tmp.columns = ['_'.join(item) if item[1] else item[0] for item in tmp.columns]
    tmp = tmp.sort_values(['score_mean', 'score_count'], ascending=False)
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