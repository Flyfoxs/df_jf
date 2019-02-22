from core.feature import *
from core.predict import *



@timed()
def merge_file(base_file = './output/0.67917234.csv'):
    bk_list = get_blocks()
    file_list = glob('./output/blocks/*.csv')
    file_list = sorted(file_list)
    base_df = pd.read_csv(base_file)
    base_df.index = base_df.index_ex
    for file_path in file_list:
        import ntpath
        file_name = ntpath.basename(file_path)
        blk_id = int(file_name.split('_')[1])
        cur_bk = bk_list.iloc[blk_id]
        col_name = cur_bk.col
        wtid = cur_bk.wtid
        value = pd.read_csv(file_path, header=None)

        base_df.loc[(base_df.wtid==wtid) & (base_df.index_ex.isin(value.iloc[:,0])), col_name]  = value.iloc[:,1].values

    base_df = convert_enum(base_df)
    file = './output/merge_2.csv'
    logger.info(f'Merge file save to:{file}')
    base_df.iloc[:, :70].to_csv(file, index=None)
    return base_df.iloc[:, :70]

@timed()
def sub_best():
    from multiprocessing import Pool as ThreadPool

    pool = ThreadPool(16)

    blk_list = get_existing_blk()
    arg_list = []

    for blk_id in blk_list:
        args = get_args_existing_by_blk(blk_id)
        if len(args) > 50:
            args['blk_id'] = blk_id
            arg_list.append(args.iloc[0])
    logger.info(f'There are {len(arg_list)} blk_id need to do')
    pool.map(gen_best_sub, arg_list, chunksize=1)


if __name__ == '__main__':
    sub_best()
    merge_file()


