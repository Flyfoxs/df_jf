from core.check import *


@timed()
def validate():
    wtid_list = [1,2,3]

    score_df = pd.DataFrame()
    for wtid in range(1, 34):
        for col_name in get_predict_col():

            score_file = f'./score/val/{wtid:02}/{col_name}.h5'

            args = get_best_para(col_name, ','.join([str(wtid) for wtid in wtid_list]), top_n=0) #validate
            args['wtid'] = wtid
            logger.debug(args)
            score = check_score(args, reverse = 1)

            logger.info(f'wtid:{wtid:02},{col_name},Current score is{score:.4f} wtih:{args}')
            args['score'] = score
            args['ct'] = pd.to_datetime('now')

            score_df = score_df.append(args, ignore_index=True)

    score_df.to_hdf(score_file, 'val')


    logger.debug(f'Validate score {len(score_df)} is :{score_df.score.mean()}')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    validate()