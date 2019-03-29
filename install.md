## 环境的准备
- mysql: 存储training的数据,相比hdf5文件,可以支持分布式traning

密码及环境配置的修改在config.py中  
    
- Redis: 提供分布式锁,这样可以多台主机并行训练

密码及环境配置的修改在config.py中 

## Training(可以跳过,直接使用training好的模型:./imp/best_arg.h5)
耗时比较久,并且依赖mysql和Redis.
如果需要重新训练模型,可以删除此文件即可.(但是在比较好的服务器上面, 可能也需要训练1天以上)

nohup ./bin/main.sh &

## Prediction 
因为模型已经预训练过了, 可以不用Training,直接predict

nohup ./bin/predict.sh &

预测文件最终输出到./output中


## 思路介绍
- 对所有的wtid及列逐一分析, 生成数据块. 指示出哪些是缺失块,哪些是数据块
- 因为大部分数据都是整列丢失,但是经过分析发现相同列在不同文件之间相关性很高,所以train数据的准备是通过当前文件关联其他文件来获取Feature. 关联哪些文件以及多少个文件是由模型动态学习.
- Training数据的生成, 一个缺失数据块可以从其上下分别生成2个Training样本, 一个用于train,一个用于validate. 同时,通过滑动可以获取足够的样本来模拟test数据
- 针对缺失数据块,逐块预测,分别保存. Training数据的生成也是针对之前预训练的参数来生成.
- 将之前保存的分块结果, 合并到template文件中.



## 模型介绍
### 内部参数

- class_name: 线性回归,随机森林
- col_name: 模型针对的列, var001, var002.... var068
- drop_threshold: 当特征的值大于 (1-drop_threshold) 丢失该特征
- file_num: 从几个文件提取出相关特征
- momenta_impact: 惯性影响的比例(前后都受影响). 如果设置0.1, 就是整个缺失数据的80%用class_name指定的模型来预测.
- momenta_col_length: 惯性特征的统计长度(前后都受影响). 如果是1,就是直接使用ffill, bfill
- related_col_count: 如果是0,就是var001,只从别的文件关联var001作为特征. 如果是1,就是增加相关性最高的另外一个列来作为特征
- col_per: 选取最终特征的部分feature来参与学习,预测. 如果col_per = 0.9, 初步筛选特征有30个, 则只保留相关性最高的27个特征参与训练.
- time_sn: 是否引入时间变量作为一个维度
- window: 根据缺失数据块的大小,选择多少长度的数据参与训练. 如果window=1.5, 缺失数据块长度为100, 则从缺失数据块前后各取150行参与训练.
- n_estimators: 使用树模型时的参数
- max_depth: 使用树模型时的参数


### 超参

只有一个超参数,就是如何对测试数据分组,目前只尝试了按照缺失块的大小分为9组.
具体的分组方法,可参考:check.get_miss_blocks_ex
还可以根据时间来分组, 或者同时根据缺少数据块大小和时间分组, 由于时间有限, 没有一一尝试.


## 代码结构
    
    ├── bin
    │   ├── main.sh 
    │   ├── predict.sh
    ├── cache 存放缓存数据
    ├── core
    │   ├── check.py 参数的生成及动态扩展
    │   ├── config.py 配置文件,密码,服务器地址
    │   ├── config_local.py 本地配置文件,不上传服务器,比如线程大小,缓存大小
    │   ├── db.py 读写DB的方法
    │   ├── feature.py 特征的准备
    │   ├── merge.py 逐块预测缺失数据, 然后合并到template文件
    │   ├── merge_multiple_file.py 用于和队友合并文件时使用
    │   ├── predict.py 训练模型相关
    ├── ddl
    │   ├── ddl 数据库的建表语句
    ├── imp  存放模型参数快照
    │   ├── lr_bin_9.h5
    │   ├── v3.h5
    ├── input 存放训练原始数据
    │   ├── 033
    │   │   └── 201807.csv
    │   ├── submit_example.csv
    │   └── template_submit_result.csv
    ├── notebook 存放jupyter 文件
    │   ├── hyx_code.ipynb
    │   ├── hyx_version_306.ipynb
    │   ├── merge_final_file.ipynb
    ├── output 存放模型的输出结果
    ├── requirements.txt

 
## 其他
- 本模型只测试了68列中的37列
根据和队友共同分析,最终发现本模型对其中37个列比较友好.
具体37个列的list保存在merge_multiple_file.config变量中

- 以上所有代码, 对内存有较高要求, 在700G的服务器上训练通过, 但是得到模型后的预测一般的主机即可.

- 如果性能好的机器, 可以适量加大线程数量. (./core/config_local.py#thred_num)



    