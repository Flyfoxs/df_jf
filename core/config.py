
local = False
vector_size=200
from file_cache.utils.util_log import *
import numpy as np

cache_size = 64

model_paras = ['class_name', 'col_name',
               'drop_threshold',
               #'add_features',
               'file_num',
               'momenta_col_length', 'momenta_impact_ratio',
               'related_col_count',
               'time_sn',
               'window',]

date_type={
'wtid'	    :np.int16,
'var053'	:np.int16,
'var066'	:np.int16,
'var016'	:np.int16,
'var020'	:np.int16,
'var047'	:np.int16,
'var001'	:np.float64,
'var002'	:np.float64,
'var003'	:np.float64,
'var004'	:np.float64,
'var005'	:np.float64,
'var006'	:np.float64,
'var007'	:np.float64,
'var008'	:np.float64,
'var009'	:np.float64,
'var010'	:np.float64,
'var011'	:np.float64,
'var012'	:np.float64,
'var013'	:np.float64,
'var014'	:np.float64,
'var015'	:np.float64,
'var017'	:np.float64,
'var018'	:np.float64,
'var019'	:np.float64,
'var021'	:np.float64,
'var022'	:np.float64,
'var023'	:np.float64,
'var024'	:np.float64,
'var025'	:np.float64,
'var026'	:np.float64,
'var027'	:np.float64,
'var028'	:np.float64,
'var029'	:np.float64,
'var030'	:np.float64,
'var031'	:np.float64,
'var032'	:np.float64,
'var033'	:np.float64,
'var034'	:np.float64,
'var035'	:np.float64,
'var036'	:np.float64,
'var037'	:np.float64,
'var038'	:np.float64,
'var039'	:np.float64,
'var040'	:np.float64,
'var041'	:np.float64,
'var042'	:np.float64,
'var043'	:np.float64,
'var044'	:np.float64,
'var045'	:np.float64,
'var046'	:np.float64,
'var048'	:np.float64,
'var049'	:np.float64,
'var050'	:np.float64,
'var051'	:np.float64,
'var052'	:np.float64,
'var054'	:np.float64,
'var055'	:np.float64,
'var056'	:np.float64,
'var057'	:np.float64,
'var058'	:np.float64,
'var059'	:np.float64,
'var060'	:np.float64,
'var061'	:np.float64,
'var062'	:np.float64,
'var063'	:np.float64,
'var064'	:np.float64,
'var065'	:np.float64,
'var067'	:np.float64,
'var068'	:np.float64,

}


try:
    from core.config_local import *
except Exception as e:
    logger.exception(e)
    logger.debug("There is no local config")