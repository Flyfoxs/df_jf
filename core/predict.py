from core.feature import *
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

    if cur_ratio >  threshold:
        half = len(train)//2

        val_1 = train.iloc[:half, 0].value_counts().index[0]
        res_1 = np.ones(len(val)//2)*val_1

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



def get_cut_predict(train, val, args):
    from sklearn.linear_model import Ridge, LinearRegression

    momenta_col_length = int(args.momenta_col_length)
    momenta_impact_length = int(args.momenta_impact_length)
    enable_time = args.time_sn

    clf = LinearRegression()
    np.random.seed(0)

    if not enable_time and int(args.file_num) > 1:
        val   =   val.drop(axis='column', columns=['time_sn'])
        train = train.drop(axis='column', columns=['time_sn'])

    logger.debug(f'train:{train.shape}, val:{val.shape}:[{val.index.min()}, {val.index.max()}] '
                 f'{train.columns} ({args.time_sn})')
    clf.fit(train.iloc[:, 1:], train.iloc[:, 0])

    cut_len = min(momenta_impact_length, len(val)//3)

    block_begin = val.index.min()
    block_end = val.index.max()
    logger.debug(f'val block:[{block_begin}, {block_end}], {val.columns}')

    logger.debug(f'train:{train.shape}, val{val.shape}')

    begin_val_arr=train.iloc[:, 0].loc[:(block_begin - 1)].tail(momenta_col_length).values
    end_val_arr = train.iloc[:, 0].loc[(block_end + 1):].head(momenta_col_length).values

    begin_val, end_val = get_momenta_value(begin_val_arr, end_val_arr )

    logger.debug(f'====Begin_val:{begin_val}:{begin_val_arr}, end_val:{begin_val}:{end_val_arr}, ')

    return np.hstack((np.ones(cut_len) * begin_val,
                      clf.predict(val.iloc[cut_len:len(val)-cut_len]),
                      np.ones(cut_len) * end_val
                      ))


