import sys
sys.path.insert(99, './df_jf')
sys.path.insert(99, '../df_jf')
print(sys.path)

from core.feature import *
import fire


from core.predict import *

@timed()
def check_score(args, pic=False):
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

    train_list = get_train_feature(wtid, col, args)

    train_list = sorted(train_list, key=lambda val: len(val[1]), reverse=True)

    count, loss = 0, 0

    if pic ==True:
        train_list = train_list[:10]
    for train, val, blockid in train_list :

        is_enum = True if 'int' in date_type[col].__name__ else False
        logger.debug(f'Blockid#{blockid}, train:{train.shape}, val:{val.shape}, file_num:{args.file_num}')
        check_fn = get_predict_fun(blockid, train, args)

        if pic:
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


@timed()
def check_score_all():
    wtid = 1
    window = 0.7
    momenta_col_length = 1
    momenta_impact_length = 300
    time_sn = True
    related_col_count = 0
    drop_threshold = 1

    class_name = 'lr'

    args = check_options()
    for wtid in range(2, 5):
        for col_name_sn in range(args.col_begin, args.col_end+1):
            col_name = f"var{str(col_name_sn).rjust( 3, '0',)}"
            for window in np.arange(0.5, 1.5, 0.2):
                window = round(window,1)
                for momenta_col_length in range(1, 20, 4):
                    for momenta_impact_length in [100, 200, 300]:
                        for time_sn in [True, False]:
                                for file_num in range(1, 6):
                                    args = { 'wtid': wtid,
                                             'col_name': col_name,
                                             'file_num':file_num,
                                             'window': window,
                                             'momenta_col_length':momenta_col_length,
                                             'momenta_impact_length': momenta_impact_length,
                                             'related_col_count': related_col_count,
                                             'drop_threshold': drop_threshold,
                                             'time_sn': time_sn ,
                                             'class_name': class_name,
                                             'ct': pd.to_datetime('now')
                                            }
                                    args = DefaultMunch(None, args)

                                    score = check_score(args)
                                    logger.debug(f'Current score is{score:.4f} wtih:{args}')

                                    score_file = f'./score/{col_name}.h5'
                                    if os.path.exists(score_file):
                                        score_df = pd.read_hdf(score_file)
                                    else:
                                        score_df = pd.DataFrame()
                                    args['score'] = score

                                    score_df = score_df.append(args, ignore_index=True)
                                    logger.info(f'Save {score_df.shape} to file:{score_file}')
                                    score_df.to_hdf(score_file,'score')

def check_options():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--col_begin",  type=int, default=1 )

    # Base on the latest data, not the avg
    parser.add_argument("--col_end",   type=int, default=5 )

    parser.add_argument("-D", '--debug', action='store_true', default=False)
    parser.add_argument("-W", '--warning', action='store_true', default=False)
    parser.add_argument("-L", '--log', action='store_true', default=False)


    # parser.add_argument("--version", help="check version", type=str, default='lg')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.warning:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    if args.log:
        file = f'score_{args.col_begin:2}_{args.col_end:2}.log'
        handler = logging.FileHandler(file, 'a')
        handler.setFormatter(format)
        logger.addHandler(handler)

    return args


@lru_cache()
def get_best_file_num(col_name):
    score_df = check_score_all(pic=False)  # .reset_index()

    ser = score_df.iloc[:, -5:].idxmax(axis=1)
    #print(ser.loc[col_name])
    logger.debug(ser)
    return int(ser.loc[col_name].split('_')[1])


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