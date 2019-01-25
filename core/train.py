from core.feature import *
import fire


def score(val1, val2, enum=False):
    loss = 0
    for real, predict in zip(val1, val2):
        if enum:
            loss += 1 if real == predict else 0
        else:
            loss += np.exp(-100 * abs(real - predict) / max(abs(real), 1e-15))
    return len(val1), round(loss, 4)



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


@file_cache()
def predict_wtid(wtid):
    block_list = get_blocks()

    train_ex = get_train_ex(wtid)
    for blockid, missing_block in block_list.loc[
                (block_list.wtid == wtid) & (block_list.kind == 'missing')].iterrows():
        col_name = missing_block.col
        train, sub = get_submit_feature_by_block_id(blockid)

        predict_fn = get_predict_fun(blockid, train)
        predict_res = predict_fn(sub.time_sn)
        sub[col_name] = predict_res
        logger.debug(f'predict_res:{predict_res.shape}, {type(predict_res)}')
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
def predict_all():

    train_list = []
    from tqdm import tqdm
    for wtid in tqdm(range(1, 34)):
        train_ex =  predict_wtid(wtid)
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
    file = './output/submit.csv'
    submit.to_csv(file,index=None)

    logger.debug(f'Sub file save to {file}')

    return submit




if __name__ == '__main__':
    """
    python core/train.py predict_wtid 1

    """
    #logging.getLogger().setLevel(logging.INFO)
    #fire.Fire()
    predict_all()



