import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def process_data(file_path):
    # 读取文件
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
        if line.startswith('BM25') or line.startswith('Contriever') or line.startswith('BGE') or line.startswith('LLM-E'):
            # 如果block_data不为空，则说明前一个块的数据已经读取完毕，可以添加到主数据集中
            if block_data is not None:
                block_data['Method'] = method_name
                block_data['Dataset'] = dataset_name
                data = pd.concat([data, block_data], ignore_index=True)

            # 重置block_data
            block_data = pd.DataFrame()
            method_name, dataset_name, _, label = line.split()

        elif 'Loop' in line:
            # 这是列名行，忽略它，因为我们会在添加数据时直接指定列名
            pass

        else:
            # 这是数据行，我们将其添加到block_data中
            loop, avg_answer_rank, avg_human_answer_rank = line.split()
            temp_df = pd.DataFrame(
                {'Iteration': [loop], 'First Right From ALL Sources': [float(avg_answer_rank)], 'First Right From Human': [float(avg_human_answer_rank)]})
            block_data = pd.concat([block_data, temp_df], ignore_index=True)

    # 添加最后一个数据块
    if block_data is not None:
        block_data['Method'] = method_name
        block_data['Dataset'] = dataset_name
        data = pd.concat([data, block_data], ignore_index=True)

    return data

def prepare_data_for_plotting(data):
    # 将iteration列中的字符串转换为有序分类类型，以便正确地排序x轴
    data['Iteration'] = pd.Categorical(data['Iteration'], categories=[f'{i}' for i in range(1, 11)], ordered=True)

    # 对不同的数据集取均值
    data['Dataset'] = data['Dataset'].astype(str)

    # 接下来，对于每个Method，我们计算Average 0->1和Average 1->0的均值
    mean_data = data.groupby(['Iteration', 'Method']).agg({'First Right From ALL Sources': 'mean', 'First Right From Human': 'mean'}).reset_index()

    # 定义方法显示顺序
    method_order = [
        'BM25', 'Contriever', 'LLM-E', 'BGE', 'BM25+U', 'BM25+M', 'BM25+BR',
        'BGE+U', 'BGE+M', 'BGE+BR'
    ]
    # replace Contriver with Contri
    mean_data['Method'] = mean_data['Method'].replace('Contriever', 'Contri')
    # 将Method列转换为有序分类类型以确保图例正确排序
    mean_data['Method'] = pd.Categorical(mean_data['Method'], categories=method_order, ordered=True)

    # 将数据从宽格式转换为长格式
    long_data = pd.melt(mean_data, id_vars=['Iteration', 'Method'],
                        value_vars=['First Right From ALL Sources', 'First Right From Human'],
                        var_name='Metric', value_name='Value')

    return long_data

# 读取并处理两组数据
file_path0 = f'sum_tsvs/summary_rank_label_0.tsv'  # 替换为你的文件路径
file_path1 = f'sum_tsvs/summary_rank_label_1.tsv'  # 替换为你的文件路径

data0 = process_data(file_path0)
data1 = process_data(file_path1)

long_data0 = prepare_data_for_plotting(data0)
long_data1 = prepare_data_for_plotting(data1)

# 绘制子图
# ... [绘图代码，请参见上一个回答中提供的绘图代码] ...
# 绘图代码继续

# 定义子图布局
fig, axes = plt.subplots(1, 2, figsize=(20, 7.5), sharey=True)

# 设置seaborn主题和风格
sns.set_theme(style="ticks", rc={'axes.formatter.limits': (-4, 5)}, font_scale=2.5)
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# 绘制第一个子图（label0）
sns.lineplot(
    ax=axes[0],
    x='Iteration',
    y='Value',
    hue='Method',
    style='Metric',
    palette="mako",
    markers=True,
    dashes=False,
    data=long_data0,  # 这应该是label0的长格式数据
    markersize=25
)
axes[0].set_title('EM = 0')

# 绘制第二个子图（label1）
sns.lineplot(
    ax=axes[1],
    x='Iteration',
    y='Value',
    hue='Method',
    style='Metric',
    palette="mako",
    markers=True,
    dashes=False,
    data=long_data1,  # 这应该是label1的长格式数据
    markersize=25
)
axes[1].set_title('EM = 1')

# 设置坐标轴标签
for ax in axes:
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Rank')

# 移除子图的图例
for ax in axes:
    ax.legend([],[], frameon=False)

# 获取方法和指标的唯一值
methods = long_data0['Method'].cat.categories.tolist()
metrics = long_data0['Metric'].unique().tolist()

# 创建方法图例：从现有的图例中获取颜色和标签
# 去掉legends边框

method_lines = [mlines.Line2D([], [], color=sns.color_palette("mako", len(methods))[i], marker='o', linestyle='',
                              markersize=10) for i in range(len(methods))]
method_legend = fig.legend(handles=method_lines, labels=methods, title='Method', loc='lower center', bbox_to_anchor=(0.45, -0.3), ncol=5, fontsize=25, title_fontsize=25, frameon=False)

# 创建指标图例
metric_markers = ['o', 'X']  # 确保与指标的数量匹配
metric_lines = [mlines.Line2D([], [], color='black', marker=metric_markers[i], linestyle='', markersize=25)
                for i in range(len(metrics))]
metric_legend = fig.legend(handles=metric_lines, labels=metrics, title='Metric', loc='lower center', bbox_to_anchor=(0.45, -0.07), ncol=2, fontsize=25, title_fontsize=25, frameon=False)

# 修改坐标轴和标签的大小并加粗
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=20)
    # bold axis numbers
    for tick in ax.get_xticklabels():
        tick.set_fontsize(20)
        tick.set_fontweight('bold')
    ax.set_xlabel(ax.get_xlabel(), fontsize=25, weight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=25, weight='bold')




# 调整整体布局，为外部图例预留空间
plt.subplots_adjust(right=0.8, bottom=0.2)

# 保存整个图形
plt.savefig('Combined_Average_Ranks.png', bbox_extra_artists=(method_legend, metric_legend), bbox_inches='tight')

plt.show()
