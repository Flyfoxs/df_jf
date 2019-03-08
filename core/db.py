import mysql.connector
import mysqlx
from core.feature import *





import contextlib


version = 3
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

def get_session():
    session = mysqlx.get_session({
        'host': 'vm-ai-2.cisco.com',
        #'port': 3306,
        'user': 'ai_lab',
        'password': 'Had00p!!',
        'schema': 'ai'
    })

    return session

@timed()
def check_last_time_by_binid(bin_id,col_name, threshold):
    db = get_connect()

    sql = f""" select IFNULL(max(ct),date'2011-01-01')  from score_list 
    where version={version}
    and bin_id = {int(bin_id)}
    and col_name='{col_name}'
    """
    cur = db.cursor()
    cur.execute(sql)

    latest =  cur.fetchone()[0]

    gap = (pd.to_datetime('now') - latest) / timedelta(minutes=1)

    return gap > threshold


@timed()
def check_last_time_by_wtid(key):
    db = get_connect()
    sql = f""" select IFNULL(max(ct),date'2011-01-01')  from score_list where 
    version={version} and  wtid = {int(key)}"""
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
    score_ind['time_begin'] = cur_blk.time_begin
    score_ind['time_end'] = cur_blk.time_end
    score_ind = dict(score_ind )
    ##print(score_ind)
    #print('abc{blk_id}'.format(**score_ind))
    sql = """insert into score_list(
            blk_id  ,
            bin_id,
            wtid,
            class_name	 ,
            col_name	 ,
            direct	 ,
            file_num	 ,
            momenta_col_length	 ,
            momenta_impact	 ,
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
            shift,
            time_begin,
            time_end,
            server,
            version)
                values
                (
            {blk_id}  ,
            {bin_id},
            {wtid},
            '{class_name}'	 ,
            '{col_name}'	 ,
            '{direct}',
            {file_num}	 ,
            {momenta_col_length}	 ,
            {momenta_impact}	 ,
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
            {shift},
            {time_begin},
            {time_end},
            '{server}',
            {version}
               )
                """.format(**score_ind, version=version)
    cur = db.cursor()
    logger.info(sql)
    cur.execute(sql )
    db.commit()


def get_args_existing_by_blk(bin_id, col_name, class_name=None, direct=None, shift=0):
    db = get_connect()
    class_name = 'null' if class_name is None else f"'{class_name}'"
    direct = 'null' if direct is None else f"'{direct}'"
    sql = f""" select class_name, 
                        col_name,
                        drop_threshold,
                        file_num,
                        momenta_col_length,
                        momenta_impact,
                        related_col_count,
                        time_sn,
                        window,
                        n_estimators,
                        max_depth,
                        bin_id,
                        sum(score_total)/sum(score_count) score_mean,
                        std(score) score_std,
                        count(*) count_rec,
                        count(distinct blk_id) count_blk
                    from score_list where bin_id={bin_id} 
                                and col_name='{col_name}'
                                and class_name=ifnull({class_name}, class_name)
                                and direct=ifnull({direct}, direct)  
                                and version={version}
                                and shift={shift}
                        group by
                        class_name, 
                        col_name,
                        drop_threshold,
                        file_num,
                        momenta_col_length,
                        momenta_impact,
                        related_col_count,
                        time_sn,
                        window,
                        n_estimators,
                        max_depth,
                        bin_id   
                """
    logger.info(f'get_args_existing_by_blk:{sql}')
    exist_df = pd.read_sql(sql, db)
    if len(exist_df) == 0 :
        return exist_df
    exist_df = exist_df.sort_values('score_mean', ascending=False)
    return exist_df


def get_best_arg_by_blk(bin_id,col_name, class_name=None,direct=None, top=0, shift=0):
    args = get_args_existing_by_blk(bin_id, col_name, class_name,direct, shift)
    if args is not None and len(args)>1:
        args = args.reset_index().sort_values(['score_mean'], ascending=[False])#.head(10)
        #args = args.sort_values('score_std')
        args['bin_id']=bin_id
        return args.iloc[top]
    else:
        return None

@timed()
def get_args_missing_by_blk(original: pd.DataFrame, bin_id, col_name, shift):
    original['file_num'] = original['file_num'].astype(int)
    original['momenta_col_length'] = original['momenta_col_length'].astype(int)
    original['related_col_count'] = original['related_col_count'].astype(int)
    original['time_sn'] = original['time_sn'].astype(int)

    original['n_estimators'] = original['n_estimators'].astype(int)
    original['max_depth'] = original['max_depth'].astype(int)

    original['momenta_impact'] = np.round(original['momenta_impact'], 3)
    original['drop_threshold'] = np.round(original['drop_threshold'], 3)
    original['window'] = np.round(original['window'], 3)


    exist_df = get_args_existing_by_blk(bin_id,col_name, shift=shift)

    threshold = 0.99
    if exist_df is not None and len(exist_df) > 0 and exist_df.score_mean.max() >= threshold:
        max_score = exist_df.score_mean.max()
        logger.info(f'bin_id:{bin_id}, col:{exist_df.at[1, "col_name"]}, already the socre:{round(max_score,4)}')
        return exist_df.loc[pd.isna(exist_df.index)]

    original = original.copy().drop(axis='column' ,
                                    columns=['score_mean', 'score_std',
                                             'bin_id', 'count_blk', 'count_rec',
                                             'length_max', 'score_count'],errors='ignore' )

    #Can not remove time_sn, if only 1 file
    original.loc[(original.file_num == 1) & (original.time_sn == 0), 'time_sn'] = 1
    original = original.drop_duplicates(model_paras)

    if len(exist_df) == 0 :
        return original
    todo = pd.merge(original, exist_df, how='left', on=model_paras)
    # logger.info(f'{todo.shape}, {todo.columns}')
    # logger.info(f'{original.shape}, {original.columns}')
    # logger.info(f'{exist_df.shape}, {exist_df.columns}')

    todo = todo.loc[pd.isna(todo.score_mean)]
    logger.info(f'todo:{len(todo)},miss:{len(original)}, existing:{len(exist_df)}')
    return todo[original.columns].drop_duplicates(model_paras)

def get_existing_blk():
    db = get_connect()
    sql = f""" select distinct blk_id from score_list where version = {version} order by blk_id"""
    return pd.read_sql(sql, db).iloc[:,0]