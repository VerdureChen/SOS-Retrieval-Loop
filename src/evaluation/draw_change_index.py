import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle as Rect
# 读取文件
file_path = 'sum_tsvs/summary_change_index.tsv'  # 替换为你的文件路径
with open(file_path, 'r') as file:
    lines = file.readlines()

# 初始化空的DataFrame
data = pd.DataFrame()

# 用于临时存储每个块的数据
block_data = None
method_name = ''
dataset_name = ''

# 遍历文件的每一行
for line in lines:
    # 去除前后空白符如换行符
    line = line.strip()

    # 检查是否是新的数据块的开始
    if len(line.split()) == 2:
        # 如果block_data不为空，则说明前一个块的数据已经读取完毕，可以添加到主数据集中
        if block_data is not None:
            block_data['Method'] = method_name
            block_data['Dataset'] = dataset_name
            data = pd.concat([data, block_data], ignore_index=True)

        # 重置block_data
        block_data = pd.DataFrame()
        method_name, dataset_name = line.split()

    elif 'Loop' in line:
        # 这是列名行，忽略它，因为我们会在添加数据时直接指定列名
        pass

    else:
        # 这是数据行，我们将其添加到block_data中
        loop, avg_0_1, avg_1_0 = line.split()
        temp_df = pd.DataFrame(
            {'Iteration': [loop], 'Average 0->1': [float(avg_0_1)], 'Average 1->0': [float(avg_1_0)]})
        block_data = pd.concat([block_data, temp_df], ignore_index=True)

# 添加最后一个数据块
if block_data is not None:
    block_data['Method'] = method_name
    block_data['Dataset'] = dataset_name
    data = pd.concat([data, block_data], ignore_index=True)

# 将iteration列中的字符串转换为有序分类类型，以便正确地排序x轴
data['Iteration'] = pd.Categorical(data['Iteration'], categories=[f'{i}->{i + 1}' for i in range(1,10)], ordered=True)

# 设置Pandas显示选项以打印整个DataFrame
pd.set_option('display.max_rows', None)  # 设置为None以显示所有行
pd.set_option('display.max_columns', None)  # 设置为None以显示所有列
pd.set_option('display.width', None)  # 自动调整显示宽度

print(data)
# 对不同的数据集取均值
data['Dataset'] = data['Dataset'].astype(str)

# 接下来，对于每个Method，我们计算Average 0->1和Average 1->0的均值
mean_data = data.groupby(['Iteration', 'Method']).agg({'Average 0->1': 'mean', 'Average 1->0': 'mean'}).reset_index()

print(mean_data)

# # 定义方法显示顺序
# method_order = [
#     'BM25', 'Contriever', 'LLM-E', 'BGE-B', 'BM25+U', 'BM25+M', 'BM25+B',
#     'BGE+U', 'BGE+M', 'BGE+B'
# ]
#
# # 将Method列转换为有序分类类型以确保图例正确排序
# mean_data['Method'] = pd.Categorical(mean_data['Method'], categories=method_order, ordered=True)
#
# # 定义子图布局
# fig, ax = plt.subplots(figsize=(15, 7.5))
#
# # 设置seaborn主题和风格
# sns.set_theme(style="ticks", context='talk', font_scale=2.4)
#
# # 绘制所有数据，两个指标使用不同的样式
# sns.lineplot(
#     x='Iteration',
#     y='Average 0->1',
#     hue='Method',
#     style='Method',
#     data=mean_data,
#     palette="#7F8FDD",
#     markers=True,
#     dashes=False,
#     ax=ax
# )
#
# sns.lineplot(
#     x='Iteration',
#     y='Average 1->0',
#     hue='Method',
#     style='Method',
#     data=mean_data,
#     palette="viridis",
#     markers=True,
#     dashes=True,
#     ax=ax
# )
#
# # 设置标题和标签
# ax.set_title('Average Rank Changes by Iteration and Method')
# ax.set_xlabel('Iteration', fontsize=25, weight='bold')
# ax.set_ylabel('Num', fontsize=25, weight='bold')
#
# # 修改坐标轴和标签的大小并加粗
# ax.tick_params(axis='both', which='major', labelsize=20)
# for tick in ax.get_xticklabels() + ax.get_yticklabels():
#     tick.set_fontsize(20)
#     tick.set_fontweight('bold')
#
# # 移除自动生成的图例
# ax.get_legend().remove()
#
# # 创建手动图例项
# legend_elements = [mlines.Line2D([], [], color='m', marker='o', linestyle='-', label='Average 0->1'),
#                    mlines.Line2D([], [], color='c', marker='o', linestyle='--', label='Average 1->0')]
#
# # 创建自定义图例
# ax.legend(handles=legend_elements, loc='upper right', fontsize=20, title_fontsize=20, title='Metrics')
#
# # 调整整体布局，为图例预留足够的空间
# plt.tight_layout(rect=[0, 0.1, 1, 1])
# # 保存整个图形
# plt.savefig('Combined_Averages_change_index.png')
# plt.show()


# 计算所有方法的整体平均值和标准偏差
overall_mean = data.groupby('Iteration').agg({'Average 0->1': ['mean', 'std'], 'Average 1->0': ['mean', 'std']})
overall_mean.columns = ['Average 0->1 Mean', 'Average 0->1 Std', 'Average 1->0 Mean', 'Average 1->0 Std']  # 重命名列
overall_mean = overall_mean.reset_index()
# Set the same seaborn theme and style as in the first script
sns.set_theme(style="ticks", rc={'axes.formatter.limits': (-4, 5)}, font_scale=1.5)
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
# 定义子图布局，此处只需要一个子图
fig, ax = plt.subplots(figsize=(18,16.5))

# Apply the blue color palette used in the first script for consistency
# blue_palette = sns.color_palette("#6A3D9A", as_cmap=True)

# ... [Continue with the plotting code] ...

# Plot the mean and standard deviation for 'Average 0->1'
sns.lineplot(x='Iteration', y='Average 0->1 Mean', data=overall_mean, color="#6A3D9A", label='Average 0->1', ax=ax)
ax.fill_between(overall_mean['Iteration'], overall_mean['Average 0->1 Mean'] - overall_mean['Average 0->1 Std'], overall_mean['Average 0->1 Mean'] + overall_mean['Average 0->1 Std'], color="#6A3D9A", alpha=0.3)

# Plot the mean and standard deviation for 'Average 1->0'
sns.lineplot(x='Iteration', y='Average 1->0 Mean', data=overall_mean, color="#7F8FDD", label='Average 1->0', ax=ax)
ax.fill_between(overall_mean['Iteration'], overall_mean['Average 1->0 Mean'] - overall_mean['Average 1->0 Std'], overall_mean['Average 1->0 Mean'] + overall_mean['Average 1->0 Std'], color="#7F8FDD", alpha=0.3)

# Set the title and label fonts and sizes according to the first script
# ax.set_title('Average Query State Transition Num', fontsize=22, weight='bold')
ax.set_xlabel('Iteration', fontsize=47, weight='bold')
ax.set_ylabel('Avg. Transition Num', fontsize=47, weight='bold')

# Modify the axis and labels to match the first script's bold and size
ax.tick_params(axis='both', which='major', labelsize=37)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontsize(38)
    tick.set_fontweight('bold')

# Adjust the legend to match the position and style in the first script
handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=15, title_fontsize=20)
ax.legend(handles=handles, labels=labels, title='Metric', loc='lower center', bbox_to_anchor=(0.45, -0.51), ncol=1, fontsize=44, title_fontsize=44, frameon=False)

# Adjust overall layout to leave space for the legend, as in the first script
plt.subplots_adjust(bottom=0.38, top=0.98, left=0.11, right=0.98)
# Save the entire figure with the legend
plt.savefig('Overall_Average_Rank_Changes.png')
plt.show()