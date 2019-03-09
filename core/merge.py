
from core.feature import *
from core.predict import *
from core.merge_multiple_file import *






@timed()
def merge_file(base_file = './output/0.67917234.csv', top_n=5):
    base_df = pd.read_csv(base_file)
    base_df.index = base_df.index_ex
    bk_list = get_blocks()
    existing_file = get_existing_blks()
    for blk_id, cnt in existing_file.items():
        cur_blk = bk_list.iloc[blk_id]
        wtid = cur_blk.wtid
        col_name = cur_blk.col

        from core.merge_multiple_file import select_col
        #select_col = select_col[:5]
        if col_name not in select_col:
            logger.warning(f'{col_name} is not in the select list:{select_col}')
            continue


        file_list = glob(f'./output/blocks/{col_name}/*_{blk_id:06}_*.csv')
        file_list = sorted(file_list)

        score_all = np.zeros((cur_blk.length,cnt))

        for sn, file_path in enumerate(file_list):
            value = pd.read_csv(file_path, header=None)
            score_all[:,sn]=value.iloc[:,1].values

        base_df.loc[(base_df.wtid==wtid) &
                    (base_df.index_ex.isin(value.iloc[:,0])), col_name]  \
            = score_all.mean(axis=1)
        logger.info(f'Blk:{blk_id} is done, {file_list}, {score_all.shape}')
        #print(score_all[:3])

    base_df = convert_enum(base_df)
    from core.merge_multiple_file import select_col
    file = f"./output/merge_m2_{len(existing_file)}_{len(select_col)}_{'_'.join(select_col[-2:])}.csv"
    base_df.iloc[:, :70].to_csv(file, index=None)
    logger.info(f'Merge file save to:{file}')
    return base_df.iloc[:, :70]

@timed()
def gen_best():
    from multiprocessing import Pool as ThreadPool

    #imp_list =  ['var042', 'var046', 'var004', 'var027', 'var034', 'var043', 'var068', 'var003', 'var052', 'var040', 'var056', 'var024']
    from core.merge_multiple_file import select_col
    imp_list = select_col
    logger.info(f'There are {len(imp_list)} col need to gen_result:{imp_list}')

    from core.check import get_miss_blocks_ex

    arg_list = []
    for col_name in imp_list:
        for bin_id in range(9):
            best = get_best_arg_by_blk(bin_id, col_name, 'lr', 'down', shift=0) #gen_best
            blk_list = get_miss_blocks_ex()
            blk_list = blk_list.loc[
                (blk_list.kind=='missing') &
                (blk_list.bin_id==bin_id) &
                (blk_list.col==col_name)]
            if best is None:
                logger.warning(f'Can not get arg for:bin_id:{bin_id}, {col_name}, and the block length is:{len(blk_list)}')
                continue
            for blk_id in blk_list.index:
                best = best.copy()
                best['blk_id']  =blk_id
                arg_list.append(best)

    logger.info(f'There are {len(arg_list)} blockid need to process')
    pool = ThreadPool(10)
    pool.map(gen_best_sub, arg_list, chunksize=np.random.randint(1,64))

def get_existing_blks():
    file_list = glob('./output/blocks/*/*.csv')
    file_list = sorted(file_list)
    file_map = DefaultMunch(0)
    for file_path in file_list:
        import ntpath
        file_name = ntpath.basename(file_path)
        blk_id = int(file_name.split('_')[1])
        file_map[blk_id] = file_map[blk_id] +1
        print(file_name.split('_')[1])
    return dict(file_map)

if __name__ == '__main__':
    """
    python ./core/merge.py > merge_8.log 2>&1
    """
    gen_best()
    merge_file()
    #merge_diff_col()


