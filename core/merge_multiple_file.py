import pandas as pd
import numpy as np
from core.feature import *

base_file = 'good_luck.csv'

new_file = 'merge_m2_3152_4_var067_var037_var062_var012_var007_var014_var001_var022_var029_var038_var005_var028_var006_var011_var057_var055_var051_var024_var035_var045_var052_var003_var040_var033_var002_var030_var021_var056_var060_var036_var019_var065.csv'

new_file = '0.70180553000.csv_04_var037_var062_var012.csv'

new_file = 'merge_m0_3152_4_var067_var037_var062_var012.csv'
new_file = '0.67950475000.csv'
new_file = 'good_luck.csv_remote_29152_100_37_var036_var019_var065_2648632_v3.9_train_val.h5.csv'
config={
    # 'var067':[('merge_4_var062_var012.csv',1)],
    # 'var037':[('merge_4_var062_var012.csv',1)],
    # 'var062':[('merge_4_var062_var012.csv',1)],

    'var017': [(new_file, 1)],  # ✔️
    'var009': [(new_file, 1)],
    'var063': [(new_file, 1)],
    'var044': [(new_file, 1)],
    'var064': [(new_file, 1)],
    'var048': [(new_file, 1)],  # ✔️



    'var067': [(new_file,1)],
    'var037': [(new_file,1)],
    'var062': [(new_file,1)],
    'var012': [(new_file,1)],

    'var007': [(new_file,1)],
    'var014': [(new_file, 1)],
    'var001': [(new_file, 1)],
    'var022': [(new_file, 1)],
    'var029': [(new_file, 1)],
    'var038': [(new_file, 1)],
    'var005': [(new_file, 1)],
    'var028': [(new_file, 1)],

    'var006': [(new_file, 1)],
    'var011': [(new_file, 1)],
    'var057': [(new_file, 1)],
    'var055': [(new_file, 1)],
    'var051': [(new_file, 1)],
    'var024': [(new_file, 1)],
    'var035': [(new_file, 1)],
    'var045': [(new_file, 1)],
    'var052': [(new_file, 1)],
    'var003': [(new_file, 1)],
    'var040': [(new_file, 1)],
    'var033': [(new_file, 1)],
    'var002': [(new_file, 1)],
    'var030': [(new_file, 1)],
    'var021': [(new_file, 1)],

    'var060': [(new_file, 1)],
    'var036': [(new_file, 1)],
    'var019': [(new_file, 1)],
    'var065': [(new_file, 1)], #31



    # 'var056': [(new_file, 1)],
    # 'var059': [('0.67950475000.csv', 1)],#Drop 34
    # 'var053' : [('0.67950475000.csv', 1)], Drop
    # 'var016': [('0.67950475000.csv', 1)],#Drop 31, int
    # 'var066': [('0.67950475000.csv', 1)],#Drop 32, int

}

#'var017','var009','var063','var044',  'var064','var048'

# config={
#

#     'var013': [(new_file, 1)],
#     'var008': [(new_file, 1)],
#     'var050': [(new_file, 1)],
#     'var026': [(new_file, 1)],
#     'var025': [(new_file, 1)],
#     'var032': [(new_file, 1)],
#     'var049': [(new_file, 1)],
#     'var039': [(new_file, 1)],
#     'var058': [(new_file, 1)],
# }
#
# config = {
#     'var016': [(new_file, 1)],#Drop 31, int
#     'var066': [(new_file, 1)],#Drop 32, int
#
#     'var053': [(new_file, 1)], #Drop, int
#
#     'var047': [(new_file, 1)], #36 , int
#     'var020': [(new_file,1)], #int
# }

select_col = list(config.keys())

from functools import lru_cache
@lru_cache()
def read_file(base_file):
    base = pd.read_csv(base_file)
    return base

def merge_diff_col(base_file=f'./output/{base_file}', fillzero=False ):

    base = read_file(base_file)
    if fillzero:
        other_col = ['var053','var066','var016','var020','var047']
        print(f'Set some col({len(other_col)}) to Null:{other_col}')
        base.loc[:, other_col] = -1

    col_list = []
    for col in config.keys():
        col_list.append(col)
        merge_res = merge_col(col)
        if merge_res is not None:
            base[col]=merge_res
        if len(col_list) >= 1:
            file_name = f"{base_file}_{fillzero}_{len(col_list):02}_{'_'.join(col_list[-4:])}.csv"
            ##file_name = file_name.replace('var0','')
            base.to_csv(file_name, index=None)


def merge_2_file(col_list,
                 base_file=f'./output/{base_file}',
                 replace_file =f'./output/{new_file}',
                                        #312_0.7082478000000001.csv_m0_29152_100_37_var036_var019_var065_2573414_v3.8.h5.csv',
                 fillzero=True):
    base = pd.read_csv(base_file)
    if fillzero:
        other_col = ['var053', 'var066', 'var016', 'var020', 'var047']
        print(f'Set some col({len(other_col)}) to Null:{other_col}')
        base.loc[:, other_col] = -1
    logger.info('new file:{replace_file}')
    new = pd.read_csv(replace_file)
    print(new.loc[201262, 'var048'])
    print(base.loc[201262, 'var048'])
    base.loc[:,col_list] = new.loc[:,col_list]#.values
    print(base.loc[201262, 'var048'])

    import ntpath
    file_name = ntpath.basename(base_file)

    logger.info(f'Merge col_list {len(col_list)}:{col_list}')
    file_name = f"./output/{file_name}_zero_{fillzero}_14_v2_{len(col_list):02}_{'_'.join(col_list[-2:])}.csv"
    base = convert_enum(base)
    base.to_csv(file_name, index=None)
    logger.info(f'File save to {file_name}')

def merge_col(col):

    if col in config and len(config[col])>0:
        print(f'Try to merge {col} with conf:{config[col]}')
        original_df = pd.DataFrame()
        config_col = config[col]
        config_col = sorted(config_col, key=lambda val: val[1], reverse=True)
        weight = np.zeros(len(config_col))
        for sn, individual_file in enumerate(config_col):
            weight[sn]=individual_file[1]
            individual_file = f'./output/{individual_file[0]}'
            df = read_file(individual_file)
            original_df[str(sn)]=df[[col]].iloc[:,0]

        total = np.sum(weight)
        original_df['avg'] = original_df.apply(lambda row: np.dot(weight,row.values)/total ,axis=1)

        gap = original_df.copy()

        for col in gap  :
            if col != 'avg':
                gap[col] = np.abs(gap[col] - gap['avg'])
            else:
                #print('Remove avg from df')
                del gap['avg']

        original_df['best_index']=gap.idxmin(axis=1)

        original_df['best_value'] = original_df.apply(lambda row: row[row['best_index']], axis=1)

        return original_df['best_value']

    else:
        return None


if __name__ == '__main__':
    #merge_diff_col(fillzero=True)

    merge_2_file(select_col[:6],   fillzero=True)
    #merge_2_file(select_col[-31:], fillzero=True)
    # merge_2_file(select_col,       fillzero=True)



