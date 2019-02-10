from core.check import *


@timed()
def validate():

    from multiprocessing import Pool as ThreadPool  # 进程

    pool = ThreadPool(8)
    score_list = pool.map(validate_wtid, range(1, 34), chunksize=1)

    score_df = pd.concat(score_list)

    avg = round(score_df.score_total.sum()/score_df.score_count.sum(),7)
    score_file = f'./score/val/validate_{avg :.6f}({score_df.score.mean():.6f})_cnt_{check_options().check_cnt}.h5'

    path = os.path.dirname(score_file)
    os.makedirs(path, exist_ok=True)

    score_df.to_hdf(score_file, 'score')

    logger.info(f'Val save to file:{score_file}')


def validate_wtid(wtid):

    score_df = pd.DataFrame()

    for col_name in get_predict_col():

        args = get_best_para(col_name, str(wtid), top_n=0) #validate
        args['wtid'] = wtid
        logger.debug(args)
        score, count = check_score(args, reverse = 1)

        logger.info(f'wtid:{wtid:02},{col_name},Current score is{score:.4f} wtih:{args}')
        #args['score'] = score
        args['score'] = round(score / count, 4)
        args['score_total'] = score
        args['score_count'] = count
        args['ct'] = pd.to_datetime('now')
        score_df = score_df.append(args, ignore_index=True)

    logger.info(f'Validate score {len(score_df)},{col_name},wtid:{wtid} is :{score_df.score.mean()}')

    return score_df


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    validate()