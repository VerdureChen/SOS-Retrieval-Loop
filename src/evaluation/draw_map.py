import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from collections import defaultdict
from itertools import cycle

# 读取TSV文件
file_path = 'tsvs/nq_tsv.tsv'  # 请将'your_file_path.tsv'替换为你的TSV文件的实际路径
#string
df = pd.read_csv(file_path, sep='\t', dtype={'0->1': str, '1->0': str})
def process_id_list(id_list):
    if pd.isnull(id_list) or id_list == '':
        return []
    return list(map(int, id_list.split(',')))

# 应用转换函数到'0->1'和'1->0'列
df['0->1'] = df['0->1'].apply(process_id_list)
df['1->0'] = df['1->0'].apply(process_id_list)

# 绘制图形
plt.figure(figsize=(10, 6))

# 得到所有ID的唯一值列表
all_ids = set()
for ids in df['0->1']:
    all_ids.update(ids)
for ids in df['1->0']:
    all_ids.update(ids)

# 对于每个ID，在不同loop之间绘制连接线
for id in sorted(all_ids):
    # 初始化两个列表，用于存储当前ID在不同loop的位置
    x_vals, y_vals = [], []

    # 检查当前ID在每个loop中是否存在，并记录位置
    for index, row in df.iterrows():
        # if id in row['0->1']:
        #     x_vals.append(row['Loop Num'])
        #     y_vals.append(id)
        # 如果需要连接1->0的变化，可以取消以下注释
        if id in row['1->0']:
           x_vals.append(row['Loop Num'])
           y_vals.append(id)

    # 绘制当前ID在不同loop之间的连接线
    if x_vals:  # 确保列表不为空
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', label=f'ID {id}')

# 设置X轴与Y轴标题及图形标题
plt.xlabel('Loop Number')
plt.ylabel('ID')
plt.title('Connection of the Same IDs Across Different Loops 1->0')

# 由于ID可能很多，图例可能会非常拥挤，所以这里选择不显示图例
# 如果确定要显示图例，可以取消以下注释
# plt.legend(loc='best')

plt.savefig('tsvs/nq_tsv_10.png')