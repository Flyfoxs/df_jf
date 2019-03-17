- Value_counts over half of the data
- gassuin procedure
- polynomial(2, 3, 4)
- Neural network
- local file & related file
- time_sn add in
- find another way to determind if use multiply file
- Separator begin, middle, end
- Analysis the value_count for begin and end
- Avg the value
- Get best para by wtid and col
- Estimate by itself(Group wtid, begin and end the data block)
- more model, especial for 1 file, such as RF jobs
- more feature:feature from same file, or more file
- more fioe
- group wtid
- 

parameter:
wtid,
window:[0.5-n]
momenta_col_length:[1-100]
momenta_impact_length:[100-400]
input_file_num:[1-n]
related_col_count:[0-n]
time_sn:True/False
class:lr, deploy, gasion
ct:sysdate

split the bin by missing gap and time gap

special logic for long blocking




0.66927826000: Avg 33 wtid to get the top arg for global scope
0.67049026000  Avg 3 wtid,  1-3 to get the top arg for global scope
0.67080426000  Avg 8 wtids, 1-4, 30-33 to get the top arg for global scope
0.67126685000: Pick the top arg for each individual wtid, 1-4, 30-33 is full, others are mini args
0.67797744   : Split the validate by missing gap
               


lesson learn:
The train/validate data closed to test data
Training data estimate(size, wtid, date, environment)


# reuse predict file
# lgb
# col in same file
# Enhance it, if the block count is small
# Remove the useless col

动态特征构造
    Train数据的生成, Validate数据的生成
    1)分布一致
            单个缺失模块
            整体模块的一致性
            上下左右,滑动窗口
    2)数据块大小
    3)时间一致

多模型的支持:
    
    

Train的方法/加速:
    Grid Search
    SMO
    退火算法
    分布式
    断点续传

迁移学习:
    因为超参只有一个, 就是分组方法.
    所以类似的分组方法之间的参数是可以借鉴的.
    
何时停止训练,最优参数的选择:
    如果片面在训练集上面,追求最优结果
    


过拟合:
    参数共享
    滑动窗口
    交叉检验
    







