from core.check import *


@timed()
def validate():

    from multiprocessing import Pool as ThreadPool  # 进程

    bin_list = get_bin_id_list(check_options().gp_name)
    pool = ThreadPool(16)
    import itertools

    bin_col_list = itertools.product(bin_list, get_predict_col(), )

    score_list = pool.map(validate_bin_id, bin_col_list, chunksize=1)

    score_df = pd.concat(score_list)

    avg = round(score_df.score_total.sum()/score_df.score_count.sum(),7)
    score_file = f'./score/val/validate_{avg :.6f}_({score_df.score.mean():.6f})' \
                 f'_shift_{check_options().shift}_{check_options().gp_name}.h5'

    path = os.path.dirname(score_file)
    os.makedirs(path, exist_ok=True)

    score_df.to_hdf(score_file, 'score')

    logger.info(f'Val save to file:{score_file}')


@timed()
def validate_bin_id(bin_col):
    bin_id, col_name = bin_col

    score_df = pd.DataFrame()


    args = get_best_para(check_options().gp_name, col_name, bin_id, top_n=0) #validate
    args['bin_id'] = bin_id
    logger.debug(args)

    score, count, min_len, max_len, block_count = check_score(args, check_options().shift)

    logger.info(f'bin_id:{bin_id:02},{col_name},Current score is{score:.4f} wtih:{args}')
    #args['score'] = score
    args['score'] = round(score / count, 4)
    args['score_total'] = score
    args['score_count'] = count
    args['ct'] = pd.to_datetime('now')
    score_df = score_df.append(args, ignore_index=True)

    logger.info(f'Validate score {len(score_df)},{col_name},bin_id:{bin_id} is :{score_df.score.mean()}')

    return score_df


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    validate()

    """
  
    python ./core/validate.py --shift 3   --gp_name lr_bin_9 > val3.log 2>&1 &
    python ./core/validate.py --shift 2   --gp_name lr_bin_9 > val2.log 2>&1 &
    python ./core/validate.py --shift 1   --gp_name lr_bin_9 > val1.log 2>&1 &
    """