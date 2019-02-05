import sys
sys.path.insert(99, './df_jf')
sys.path.insert(99, '../df_jf')
print(sys.path)

from core.feature import *
import fire


from core.predict import *

@timed()
def check_score(args, pic_num=0):
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

    train_list = get_train_sample_list(wtid, col, args.file_num, args.window)

    count, loss = 0, 0

    if pic_num ==True:
        train_list = train_list[:pic_num]
    for train, val, blockid in train_list :

        is_enum = True if 'int' in date_type[col].__name__ else False
        logger.debug(f'Blockid#{blockid}, train:{train.shape}, val:{val.shape}, file_num:{args.file_num}')
        check_fn = get_predict_fun(blockid, train, args)

        if pic_num:
            plt.figure(figsize=(20, 5))
            for color, data in zip(['#ff7f0e', '#2ca02c'], [train, val]):
                plt.scatter(data.time_sn, data[col], c=color)

            x = np.linspace(train.time_sn.min(), train.time_sn.max(), 10000)
            plt.plot(x, check_fn(x))
            plt.show()

        val_res = check_fn(val.iloc[:, 1:])
        #logger.debug(f'shape of predict output {val_res.shape}, with paras:{local_args}')
        cur_count, cur_loss = score(val[col], val_res, is_enum)

        loss += cur_loss
        count += cur_count
        logger.debug(f'blockid:{blockid}, {train.shape}, {val.shape}, score={round(cur_loss/cur_count,3)}')
    avg_loss = round(loss/count, 4)

    return avg_loss


def summary_all_score():
    df = pd.DataFrame()
    for col in get_predict_col():
        df = df.append(get_best_para(col, None, 0), ignore_index=True)

    df['data_type'] = df.col_name.apply(lambda val: date_type[val].__name__)

    return df.sort_values('score').reset_index(drop=True)


@timed()
def check_score_all():

    from multiprocessing.dummy import Pool as ThreadPool #线程
    #from multiprocessing import Pool as ThreadPool  # 进程

    logger.info(f"Start a poll with size:{check_options().thread}")
    pool = ThreadPool(check_options().thread)

    summary = summary_all_score()
    col_list = summary.loc[summary.score<=1.9].col_name

    pool.map(check_score_column, col_list)

    logger.debug(f'It is done for {check_options().wtid}')

def get_window(col_name):
    return np.arange(0.5, 1.5, 0.2)

def get_momenta_col_length(col_name):
    is_enum = True if 'int' in date_type[col_name].__name__ else False
    if is_enum:
        return [0]
    else:
        return range(1, 10, 2)

def get_momenta_impact_length(col_name):
    is_enum = True if 'int' in date_type[col_name].__name__ else False
    if is_enum:
        return [0]
    else:
        return [100, 200, 300]

def get_time_sn(col_name):
    is_enum = True if 'int' in date_type[col_name].__name__ else False
    if is_enum:
        return [False]
    else:
        return [True, False]

def get_file_num(col_name):
    is_enum = True if 'int' in date_type[col_name].__name__ else False
    if is_enum:
        return [1]
    else:
        return range(1,6)

@timed()
def check_score_column(col_name):


    wtid = check_options().wtid
    window = 0.7
    momenta_col_length = 1
    momenta_impact_length = 300
    time_sn = True
    related_col_count = 0
    drop_threshold = 1
    class_name = 'lr'

    #col_name = f"var{str(col_name_sn).rjust( 3, '0',)}"
    score_file = f'./score/{wtid:02}/{col_name}.h5'
    if os.path.exists(score_file):
        score_df = pd.read_hdf(score_file)
        logger.warning(f'Already existing file:{score_df.shape}, {score_file}')
        return None
    else:
        score_df = pd.DataFrame()
        #lock the partition
        os.makedirs(f'./score/{wtid:02}', exist_ok=True)
        score_df.to_hdf(score_file, 'score')

    from tqdm import tqdm
    for window in tqdm(get_window(col_name)):
        window = round(window, 1)
        for momenta_col_length in get_momenta_col_length(col_name) :
            for momenta_impact_length in get_momenta_impact_length(col_name):
                for time_sn in get_time_sn(col_name):
                    for file_num in get_file_num(col_name):
                        args = {'wtid': wtid,
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
                        args = DefaultMunch(None, args)

                        score = check_score(args)
                        logger.debug(f'Current score is{score:.4f} wtih:{args}')


                        args['score'] = score
                        args['ct'] = pd.to_datetime('now')

                        score_df = score_df.append(args, ignore_index=True)


    score_df.to_hdf(score_file, 'score')
    logger.info(f'Save {score_df.shape} to file:{score_file}')

@lru_cache()
def check_options():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--wtid", type=int, default=1)

    parser.add_argument("-D", '--debug', action='store_true', default=False)
    parser.add_argument("-W", '--warning', action='store_true', default=False)
    parser.add_argument("-L", '--log', action='store_true', default=False)
    parser.add_argument("--thread", type=int, default=3)


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


@lru_cache()
def merge_score_col(col_name):
    import os
    df_list = []
    from glob import glob
    # for file_name in sorted(glob("./score/*/*.h5")):
    #     if col_name in file_name:
    #         tmp_df = pd.read_hdf(file_name)
    #         df_list.append(tmp_df)
    for file_name in sorted(glob("./score/*.h5")):
        if col_name in file_name:
            tmp_df = pd.read_hdf(file_name)
            df_list.append(tmp_df)
    all = pd.concat(df_list)
    logger.info(f'There are {len(df_list)} score files for {col_name}')
    return all


@lru_cache()
@timed()
def get_best_para(col_name, wtid=None, top_n=0):

    if wtid is None or wtid <1 :
        tmp = merge_score_col(col_name)

        col_list = tmp.columns
        col_list = col_list.drop('score')
        col_list = col_list.drop('wtid')
        col_list = col_list.drop('ct')
        #col_list = col_list.drop('col_name')
        print(col_list)
        tmp = tmp.groupby(list(col_list)).agg({'score':['mean', 'count','nunique','min', 'max'], 'wtid':'nunique'}).reset_index()
        tmp.columns = ['_'.join(item) if item[1] else item[0] for item in tmp.columns]
        tmp = tmp.sort_values(['score_mean', 'score_count'], ascending=False)
        tmp = tmp.rename(columns={'score_mean':'score'})
    else:
        wtid =  int(wtid)
        tmp = pd.read_hdf(f'./score/{wtid:02}/{col_name}.h5')
        tmp = tmp.loc[tmp.wtid==wtid]
        tmp = tmp.sort_va(['score', 'ct'], ascending=[False, True]).reset_index(drop=True)
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