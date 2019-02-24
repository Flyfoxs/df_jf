import mysql.connector
from core.feature import *





import contextlib

@contextlib.contextmanager
def named_lock(db_session, name, timeout):
    """Get a named mysql lock on a DB session
    """
    lock = db_session.execute("SELECT GET_LOCK(:lock_name, :timeout)",
                              {"lock_name": name, "timeout": timeout}).scalar()
    if lock:
        try:
            yield db_session
        finally:
            db_session.execute("SELECT RELEASE_LOCK(:name)", {"name": name})
    else:
        e = "Could not obtain named lock {} within {} seconds".format(
            name, timeout)
        raise RuntimeError(e)

def get_connect():
    db = mysql.connector.connect(user='ai_lab', password='Had00p!!',
                                 host='vm-ai-2',
                                 database='ai')
    return db

@timed()
def check_last_time_by_blk(blk_id, threshold):
    db = get_connect()

    sql = f""" select IFNULL(max(ct),date'2011-01-01')  from score_list where blk_id = {int(blk_id)}"""
    cur = db.cursor()
    res = cur.execute(sql)

    latest =  cur.fetchone()[0]

    gap = (pd.to_datetime('now') - latest) / timedelta(minutes=1)

    return gap > threshold


@timed()
def check_last_time_by_wtid(key):
    db = get_connect()
    sql = f""" select IFNULL(max(ct),date'2011-01-01')  from score_list where wtid = {int(key)}"""
    # logger.info(sql)
    cur = db.cursor()
    res = cur.execute(sql)
    return cur.fetchone()[0]



def insert(score_ind):
    score_ind = score_ind.fillna(0)
    db = get_connect()

    cur_blk = get_blocks().iloc[score_ind.blk_id]

    score_ind['length'] = cur_blk.length
    import socket
    host_name = socket.gethostname()
    score_ind['server'] = host_name
    score_ind = dict(score_ind )
    print(score_ind)
    print('abc{blk_id}'.format(**score_ind))
    sql = """insert into score_list(
            blk_id  ,
            wtid,
            class_name	 ,
            col_name	 ,
            direct	 ,
            file_num	 ,
            momenta_col_length	 ,
            momenta_impact_ratio	 ,
            drop_threshold	 ,
            related_col_count	 ,
            score	 ,
            score_count	 ,
            score_total	 ,
            time_sn	 ,
            window  ,
            n_estimators,
            max_depth,
            length ,
            server)
                values
                (
            {blk_id}  ,
            {wtid},
            '{class_name}'	 ,
            '{col_name}'	 ,
            '{direct}',
            {file_num}	 ,
            {momenta_col_length}	 ,
            round({momenta_impact_ratio},2)	 ,
            round({drop_threshold},2)		 ,
            {related_col_count}	 ,
            {score}	 ,
            {score_count}	 ,
            {score_total}	 ,
            {time_sn}	 ,
            round({window},2)	  ,
            {n_estimators},
            {max_depth},
            {length},
            '{server}'
               )
                """.format(**score_ind)
    cur = db.cursor()
    #logger.info(sql)
    cur.execute(sql )
    db.commit()

@lru_cache(maxsize=16)
def get_args_existing_by_blk(blk_id, class_name=None):
    db = get_connect()
    class_name = 'null' if class_name is None else f"'{class_name}'"
    sql = f""" select * from score_list where blk_id={blk_id} 
                and class_name=ifnull({class_name}, class_name)"""
    exist_df = pd.read_sql(sql, db)
    if len(exist_df) == 0 :
        return exist_df
    exist_df = exist_df.groupby(model_paras).agg({'score':['mean', 'std']})
    exist_df.columns = [ '_'.join(item) for item in exist_df.columns]
    #logger.info(exist_df.columns)
    exist_df = exist_df.reset_index().sort_values('score_mean', ascending=False)
    return exist_df


def get_best_arg_by_blk(blk_id,class_name=None):
    args = get_args_existing_by_blk(blk_id, class_name)
    if args is not None and len(args)>1:
        return args.iloc[0]
    else:
        return None

@timed()
def get_args_missing_by_blk(original: pd.DataFrame, blk_id):
    exist_df = get_args_existing_by_blk(blk_id)
    threshold = 0.99
    if exist_df is not None and len(exist_df) > 0 and exist_df.score_mean.max() >= threshold:
        max_score = exist_df.score_mean.max()
        logger.info(f'blkid:{blk_id}, col:{exist_df.at[1, "col_name"]}, already the socre:{round(max_score,4)}')
        return exist_df.loc[pd.isna(exist_df.index)]

    original = original.copy().drop(axis='column' , columns=['score_mean', 'score_std'],errors='ignore' )

    if len(exist_df) == 0 :
        return original
    todo = pd.merge(original, exist_df, how='left', on=model_paras)
    # logger.info(f'{todo.shape}, {todo.columns}')
    # logger.info(f'{original.shape}, {original.columns}')
    # logger.info(f'{exist_df.shape}, {exist_df.columns}')

    todo = todo.loc[pd.isna(todo.score_mean)]
    logger.info(f'todo:{len(todo)},miss:{len(original)}, existing:{len(exist_df)}')
    return todo[original.columns]

def get_existing_blk():
    db = get_connect()
    sql = f""" select distinct blk_id from score_list order by blk_id"""
    return pd.read_sql(sql, db).iloc[:,0]