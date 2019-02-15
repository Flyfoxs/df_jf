from core.feature import *
from core.check import *
from core.predict import *
import fire



def predict_wtid(wtid):
    app_args = options()
    gp_name = app_args.gp_name
    block_list = get_miss_blocks_ex(gp_name=gp_name)

    train_ex = get_train_ex(wtid)
    bin_list = get_bin_id_list(gp_name)
    for bin_id in bin_list:
        for blockid, missing_block in block_list.loc[
                    (block_list.wtid == wtid) &
                    (block_list.kind == 'missing') &
                    (block_list.bin_id == bin_id)
                    #( block_list.col == 'var001')
                        ].iterrows():
            col_name = missing_block.col

            para = get_best_para(gp_name, col_name, bin_id, top_n=app_args.top_n) #predict_wtid

            logger.debug(f'===Predict wtid:{wtid:2}, bin_id:{bin_id} ,{col_name},blockid:{blockid:6}, best_file_num:{para.file_num}, type:{missing_block.data_type}')
            train, sub = get_submit_feature_by_block_id(blockid, para)

            predict_fn = get_predict_fun(train, para)
            predict_res = predict_fn(sub.iloc[:, 1:])
            logger.debug(f'sub={sub.shape}, predict_res={predict_res.shape}, type={type(predict_res)}')
            sub[col_name] = predict_res

            begin, end = missing_block.begin, missing_block.end

            logger.debug(
                f'train.loc[begin:end,col_name] = {train_ex.loc[begin:end,col_name].shape}, predict_res:{predict_res.shape}, {begin}, {end}, {wtid}, {col_name}')
            train_ex.loc[begin:end, col_name] = predict_res

            if pd.isna(train_ex.loc[begin:end, col_name]).any():
                logger.exception(
                    f'wtid:{wtid},col:{missing_block.col}, blockid:{blockid},train_ex:{train_ex.shape}, train:{train.shape}')
                raise Exception('There is Nan in predictin result')

            logger.debug(f'wtid:{wtid},col:{missing_block.col}, blockid:{blockid},train_ex:{train_ex.shape}, train:{train.shape}')

    submit = get_sub_template()
    submit = submit.loc[submit.wtid==wtid]
    submit.ts = pd.to_datetime(submit.ts)
    train_ex = train_ex[ train_ex.ts.isin(submit.ts) ]
    train_ex.wtid = train_ex.wtid.astype(int)
    train_ex = train_ex.drop(axis=['column'], columns=['time_sn'])
    return convert_enum(train_ex)


@file_cache(overwrite=True)
def predict_all(version):
    args = options()

    score_df = estimate_score(args.top_n, args.gp_name)
    score_avg = round(score_df.score_total.sum() / score_df.score_count.sum(), 6)
    logger.info(f'The validate score is {score_avg:.6f} for args:{args}')

    # train_list = []

    # from tqdm import tqdm
    # for wtid in tqdm(range(1, 34)):
    #     train_ex =  predict_wtid(wtid)
    #     #train_ex = train_ex.set_index(['ts', 'wtid'])
    #     train_list.append(train_ex)

    from multiprocessing import Pool as ThreadPool  # 进程

    pool = ThreadPool(16)
    train_list  = pool.map(predict_wtid, range(1, 34))

    train_all = pd.concat(train_list)#.set_index(['ts', 'wtid'])


    submit = get_sub_template()
    submit.ts = pd.to_datetime(submit.ts)
    submit = submit[['ts', 'wtid']].merge(train_all, how='left', on=['ts', 'wtid'])
    submit = round(submit, 2)

    file = f"./output/submit_{score_avg:.6f}_{args}.csv"
    submit = submit.iloc[:, :70]
    file = replace_invalid_filename_char(file)
    submit.to_csv(file,index=None)

    logger.info(f'Sub({submit.shape}) file save to {file}')

    return submit



def options():
    import argparse
    parser = argparse.ArgumentParser()

   # parser.add_argument("--file_num", help="How many files need to merge to train set", type=int, default=1)

    #Base on the latest data, not the avg
    #parser.add_argument("--cut_len", help="fill begin, end of the val with ffill/bfill directly", type=int, default=100)
    parser.add_argument("--top_threshold", help="If the top#2 arrive ?%, then just use ffile", type=float, default=0.6)
    parser.add_argument("-D", '--debug', action='store_true', default=False)
    parser.add_argument("-W", '--warning', action='store_true', default=False)
    parser.add_argument('--version', type=str, default='0211')
    #parser.add_argument('--wtid_list', nargs='+', default=list(range(1,34)))
    #parser.add_argument('--wtid_list', nargs='+', default=[1, 2, 3, 4, 30, 31, 32, 33])
    parser.add_argument('--top_n', type=int, default=0)
    #parser.add_argument('--window', type=float, default=0.7, help='It control how many sample will be choose: window*len(test)')
    parser.add_argument("-L", '--log', action='store_true', default=False)
    parser.add_argument("--gp_name", type=str, help="The folder name to save score")

    # parser.add_argument("--version", help="check version", type=str, default='lg')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.warning:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)

    if args.log:
        file = f'train_{args.top_n}.log'
        handler = logging.FileHandler(file, 'a')
        handler.setFormatter(format)
        logger.addHandler(handler)


    return args


#@timed()
def get_submit_feature_by_block_id(blockid, para ):
    cur_block = get_blocks().loc[blockid]
    logger.debug(f'cur_block:\n{cur_block}')

    kind = cur_block['kind']
    col_name = cur_block['col']
    wtid = cur_block['wtid']
    missing_length = cur_block['length']
    begin, end = cur_block.begin, cur_block.end


    submit = get_train_feature_multi_file(wtid, col_name, int(para.file_num), int(para.related_col_count))


    val_feature = submit.loc[begin:end]

    logger.debug(f'wtid:{wtid}, col:{col_name}, file_num:{para.file_num},   blockid:{blockid}')
    logger.debug(f'Train columns:{submit.columns}')

    train_feature = get_train_df_by_val(submit, val_feature, para.window, para.drop_threshold) #submit

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


if __name__ == '__main__':

    #fire.Fire()

    # score_df = check_score_all(version='0126')



    logger.info(f'Program input:{options()}')

    # sub = predict_wtid(2)
    # logger.info(sub.shape)

    submit = predict_all(options().version)


    """
    python core/submit.py -L --gp_name lr_bin_9 --version 0214_v3 > sub.log 2>&1 &

    """

