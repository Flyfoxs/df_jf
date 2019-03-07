import pandas as pd
import numpy as np
config={
    'var067':[('0.67950475000.csv',1)],
    'var037':[('0.67950475000.csv',1)],
    'var062':[('0.67950475000.csv',1)],
    'var012':[('0.67950475000.csv',1)],
    'var007':[('0.67950475000.csv',1)],

    'var014': [('0.67950475000.csv', 1)],
    'var001': [('0.67950475000.csv', 1)],
    'var022': [('0.67950475000.csv', 1)],
    'var029': [('0.67950475000.csv', 1)],
    'var038': [('0.67950475000.csv', 1)],
    'var005': [('0.67950475000.csv', 1)],
    'var028': [('0.67950475000.csv', 1)],

    'var006': [('0.67950475000.csv', 1)],
    'var011': [('0.67950475000.csv', 1)],
    'var057': [('0.67950475000.csv', 1)],
    'var055': [('0.67950475000.csv', 1)],
    'var051': [('0.67950475000.csv', 1)],
    'var024': [('0.67950475000.csv', 1)],
    'var035': [('0.67950475000.csv', 1)],
    'var045': [('0.67950475000.csv', 1)],
    'var052': [('0.67950475000.csv', 1)],
    'var003': [('0.67950475000.csv', 1)],
    'var040': [('0.67950475000.csv', 1)],
    'var033': [('0.67950475000.csv', 1)],
    'var002': [('0.67950475000.csv', 1)],
    'var030': [('0.67950475000.csv', 1)],
    'var021': [('0.67950475000.csv', 1)],
    'var056': [('0.67950475000.csv', 1)],
    'var060': [('0.67950475000.csv', 1)],
    'var036': [('0.67950475000.csv', 1)],
    #'var053' : [('0.67950475000.csv', 1)], Drop




    ##Test
    #'var018':[('0.67950475000.csv',1), ('0.67950475000.csv',1), ('0.67950475000.csv',1)],
    #'var037':[('0.67950475000.csv',1)],

}

select_col = list(config.keys())

def merge_diff_col(base_file='./output/0.70180553000.csv', ):
    base = pd.read_csv(base_file)

    col_list = []
    for col in config.keys():
        col_list.append(col)
        merge_res = merge_col(col)
        if merge_res is not None:
            base[col]=merge_res
        file_name = f"{base_file}_v2_{'_'.join(col_list)}.csv"
        file_name = file_name.replace('var0','')
        base.to_csv(file_name, index=None)


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
            #print('individual_file=', individual_file)
            original_df[str(sn)]=pd.read_csv(individual_file, usecols= [col]).iloc[:,0]

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
    merge_diff_col()

