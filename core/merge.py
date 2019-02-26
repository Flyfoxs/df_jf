from core.feature import *
from core.predict import *



@timed()
def merge_file(direct, base_file = './output/0.67917234.csv'):
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
    file = f'./output/merge_{direct}.csv'
    logger.info(f'Merge file save to:{file}')
    base_df.iloc[:, :70].to_csv(file, index=None)
    return base_df.iloc[:, :70]

@timed()
def sub_best(direct=None):
    from multiprocessing import Pool as ThreadPool

    pool = ThreadPool(16)

    blk_list = get_blocks()
    blk_list = blk_list.loc[(blk_list.kind=='missing') &
                            (blk_list.wtid==1) &
                            (blk_list.col=='var004')]

    blk_list = blk_list.index
    arg_list = []

    for blk_id in blk_list:
        args = get_best_arg_by_blk(blk_id, 'lr',direct)
        if args is not None and len(args) > 0:
            args['blk_id'] = blk_id
            arg_list.append(args)
    logger.info(f'There are {len(arg_list)} blk_id need to do')
    pool.map(gen_best_sub, arg_list, chunksize=1)


if __name__ == '__main__':
    direct = None
    sub_best(direct)
    merge_file(direct)


