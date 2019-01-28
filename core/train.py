from core.feature import *
import fire


def score(val1, val2, enum=False):
    loss = 0
    for real, predict in zip(val1, val2):
        if enum:
            loss += 1 if real == predict else 0
        else:
            loss += np.exp(-100 * abs(real - np.round(predict,2)) / max(abs(real), 1e-15))
    return len(val1), round(loss, 4)


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
        logger.error(f'Cur ration is {cur_ratio}, threshold: {threshold}')
        return None


def get_cut_predict(train, val, cut_len):
    from sklearn.linear_model import Ridge, LinearRegression
    clf = LinearRegression()
    np.random.seed(0)
    clf.fit(train.iloc[:, 1:], train.iloc[:, 0])

    if isinstance(val, pd.DataFrame):

        cut_len = min(cut_len, len(val)//3)


        block_begin = val.index.min()
        block_end = val.index.max()
        logger.debug(f'val block:[{block_begin}, {block_end}], {val.columns}')

        logger.debug(f'train:{train.shape}, val{val.shape}')

        begin_val=train.iloc[:, 0].loc[:(block_begin - 1)].tail(1).iat[0]
        end_val = train.iloc[:, 0].loc[(block_end + 1):].iat[0]

        return np.hstack((np.ones(cut_len) * begin_val,
                          clf.predict(val.iloc[cut_len:len(val)-cut_len]),
                          np.ones(cut_len) * end_val
                          ))
    else:
        return clf.predict(val)


def get_predict_fun(blockid, train, args):
    block = get_blocks().iloc[blockid]

    col_name = block['col']

    is_enum = True if 'int' in date_type[col_name].__name__ else False

    if is_enum:
        fn = lambda val: predict_stable_col(train, val, 0)
    else:
        fn = lambda val : get_cut_predict(train, val, args.cut_len)

    return fn

def get_best_file_num(args, col_name):
    score_df = check_score_all(args, pic=False)  # .reset_index()
    score_df.columns = ['_'.join(col) for col in score_df.columns]
    ser = score_df.iloc[:, -5:].idxmax(axis=1)
    #print(ser.loc[col_name])
    return int(ser.loc[col_name].split('_')[1])




@file_cache()
def predict_wtid(version, wtid, args):
    block_list = get_blocks()

    train_ex = get_train_ex(wtid)
    for blockid, missing_block in block_list.loc[
                (block_list.wtid == wtid) & (block_list.kind == 'missing')].iterrows():
        col_name = missing_block.col

        if args.file_num > 0:
            cur_file_num = args.file_num
        else:
            cur_file_num = get_best_file_num(args,col_name)

        logger.info(f'===Predict {wtid:2},{col_name},{blockid:6}, file_num:{cur_file_num}, type:{missing_block.data_type}')
        train, sub = get_submit_feature_by_block_id(blockid, args)

        predict_fn = get_predict_fun(blockid, train, args)
        predict_res = predict_fn(sub.iloc[:, 1:])
        logger.debug(f'sub={sub.shape}, predict_res={predict_res.shape}, type={type(predict_res)}')
        sub[col_name] = predict_res

        begin, end = missing_block.begin, missing_block.end

        logger.debug(
            f'train.loc[begin:end,col_name] = {train_ex.loc[begin:end,col_name].shape}, predict_res:{predict_res.shape}, {begin}, {end}, {wtid}, {col_name}')
        train_ex.loc[begin:end, col_name] = predict_res
        logger.debug(f'wtid:{wtid},col:{missing_block.col}, blockid:{blockid},train_ex:{train_ex.shape}, train:{train.shape}')

    submit = get_sub_template()
    submit = submit.loc[submit.wtid==wtid]
    submit.ts = pd.to_datetime(submit.ts)
    train_ex = train_ex[ train_ex.ts.isin(submit.ts) ]
    train_ex.wtid = train_ex.wtid.astype(int)
    return convert_enum(train_ex)

@file_cache()
def predict_all(version,args, score_avg):

    train_list = []
    from tqdm import tqdm
    for wtid in tqdm(range(1, 34)):
        train_ex =  predict_wtid(version, wtid, args)
        #train_ex = train_ex.set_index(['ts', 'wtid'])
        train_list.append(train_ex)
    train_all = pd.concat(train_list)#.set_index(['ts', 'wtid'])
    train_all.wtid = train_all.wtid.astype(int)


    submit = get_sub_template()
    submit.ts = pd.to_datetime(submit.ts)


    submit = submit[['ts', 'wtid']].merge(train_all, how='left', on=['ts', 'wtid'])

    submit.drop(axis=['column'], columns=['time_sn'], inplace=True)

    submit = convert_enum(submit)

    submit = round(submit, 2)

    file = f'./output/submit_dynamic_file_{version}_{file_count}_{score_avg}.csv'
    submit = submit.iloc[:, :70]
    submit.to_csv(file,index=None)

    logger.debug(f'Sub({submit.shape}) file save to {file}')

    return submit



@timed()

@file_cache()
def check_score_all(args, pic=False):
    std = get_std_all()
    std = std.groupby(['col', 'data_type']).agg({'mean': ['mean', 'min', 'max', 'std']})
    std = std.reset_index().set_index('col')

    std['coef_wtid_avg'] = np.nan
    std['coef_wtid_max'] = np.nan
    std['coef_wtid_id'] = np.nan

    std['score_1_file'] = np.nan
    std['score_2_file'] = np.nan
    std['score_3_file'] = np.nan
    std['score_4_file'] = np.nan
    std['score_5_file'] = np.nan


    for col in get_predict_col():
        corr = get_corr_wtid(col)
        std.loc[col, 'coef_wtid_avg'] = corr.where(corr < 0.999).max().mean()
        std.loc[col, 'coef_wtid_max'] = corr.where(corr < 0.999).max().max()
        std.loc[col, 'coef_wtid_id'] = corr.where(corr < 0.999).max().idxmax()

        for file_num in range(1, 6):
            loss = check_score(col, pic, args)
            std.loc[col, f'score_{file_num}_file'] = loss

    return std.sort_values('score_1_file')


def check_score(col, pic, args):
    local_args = locals()
    logging.getLogger().setLevel(logging.INFO)
    import matplotlib.pyplot as plt

    train_list = get_train_feature(3, col, args)

    train_list = sorted(train_list, key=lambda val: len(val[1]), reverse=True)

    count, loss = 0, 0

    if pic ==True:
        train_list = train_list[:10]
    for train, val, blockid in train_list :

        is_enum = True if 'int' in date_type[col].__name__ else False

        check_fn = get_predict_fun(blockid, train, args)

        if pic:
            plt.figure(figsize=(20, 5))
            for color, data in zip(['#ff7f0e', '#2ca02c'], [train, val]):
                plt.scatter(data.time_sn, data[col], c=color)

            x = np.linspace(train.time_sn.min(), train.time_sn.max(), 10000)
            plt.plot(x, check_fn(x))
            plt.show()

        val_res = check_fn(val.iloc[:, 1:])
        logger.debug(f'shape of predict output {val_res.shape}, with paras:{local_args}')
        cur_count, cur_loss = score(val[col], val_res, is_enum)

        loss += cur_loss
        count += cur_count
        avg_loss = round(loss/count, 4)
        logger.debug(f'blockid:{blockid}, {train.shape}, {val.shape}, score={round(cur_loss/cur_count,3)}')
    logger.info(f'Total loss:{avg_loss:.4f}, is_enum:{is_enum}, col:{col}: ')

    return avg_loss





if __name__ == '__main__':
    """
    python core/train.py predict_wtid 1

    """
    #logging.getLogger().setLevel(logging.INFO)
    #fire.Fire()

    # score_df = check_score_all(version='0126')

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_num", help="How many files need to merge to train set", type=int, default=1)
    parser.add_argument("--cut_len", help="Cut the begin, end to ffill", type=int, default=100)
    parser.add_argument("--top_threshold", help="If the top#2 arrive ?%, then just use ffile", type=float, default=0.6)
    #parser.add_argument("--version", help="check version", type=str, default='lg')
    args = parser.parse_args()

    logger.info(f'Program input:{args}')

    score_df = check_score_all(args, pic=False)

    score_df = pd.read_hdf('./cache/check_score_all==version=lg.h5')

    score_avg = round(score_df.iloc[:, -5].mean(), 4), round(score_df.iloc[:, -5:].max(axis=1).mean(), 4)

    submit = predict_all('cut_lg', args, '_'.join(score_avg))
    #submit = predict_all('0128_lg', args)


