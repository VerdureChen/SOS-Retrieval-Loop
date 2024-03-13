import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 指定字体路径
my_font = FontProperties(fname='/home/xiaoyang2020/chenxiaoyang_11/Rob_LLM/externals/popular-fonts/微软雅黑.ttf', size=20)
# 您提供的数据
right_nums = ['right_0', 'right_1', 'right_2', 'right_3', 'right_4', 'right_5']
loops = list(range(1, 11))

# 创建所有的right和loop组合
right_loop_combinations = [(right, loop) for right in right_nums for loop in loops]
right_values = [int(combo[0].split('_')[1]) for combo in right_loop_combinations]
loop_values = [combo[1] for combo in right_loop_combinations]

# avg_query_counts数据
avg_query_counts = [
    # EM 0 data
    58.88, 74.36, 77.88, 80.9, 83.26, 85, 86.08, 87.54, 88.06, 88.88,
    23.58, 10.12, 7.5, 5.72, 4.74, 3.44, 3.38, 2.44, 2.34, 1.98,
    9.38, 4.58, 3.72, 2.78, 2.14, 1.92, 1.26, 1.44, 0.94, 0.72,
    4.4, 2.98, 2.18, 1.64, 1.4, 1.1, 0.94, 0.64, 0.6, 0.74,
    1.54, 2.74, 2.06, 1.5, 1.08, 1.26, 1.12, 1.14, 1.12, 1.52,
    0.58, 1.36, 2.12, 2.88, 3.56, 3.1, 3.52, 3.24, 3.44, 3.34,
    # EM 1 data
    3.22, 2.44, 2.62, 2.7, 2.94, 2.7, 2.92, 2.86, 2.94, 3.02,
    14.32, 4.08, 2.5, 1.78, 0.96, 1.16, 1.02, 0.86, 0.96, 0.62,
    16.82, 6.12, 3.88, 3.02, 2.26, 1.78, 1.44, 1.66, 1.06, 1.28,
    16.4, 11.92, 8.02, 5.06, 4.2, 3.5, 3.16, 2.26, 3, 1.96,
    25.46, 22.76, 16.84, 14.1, 12.12, 11.64, 9.98, 9.86, 8.78, 8.78,
    25.42, 56.54, 70.68, 77.92, 81.34, 83.4, 85.18, 86.06, 86.76, 87.16
]
# 为了区分EM 0和EM 1的数据，我们创建一个EM列
em_values = ['EM=0'] * (len(right_nums) * len(loops)) + ['EM=1'] * (len(right_nums) * len(loops))

# 创建DataFrame
df = pd.DataFrame({
    'Right_Num': right_values * 2,
    'Loop': loop_values * 2,
    'Avg_Query_Count': avg_query_counts,
    'EM': em_values
})
# 自定义颜色字典
palette = {"EM=1": "#6A3D9A", "EM=0": "#7F8FDD"}
# 筛选出我们想要的loops
df = df[df['Loop'].isin([1, 2, 5, 10])]

# 设置Seaborn的风格
sns.set(style="whitegrid", font_scale=1.5)

# 创建一个Figure对象和6个Axes对象（subplot），所有子图在一行
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 5), sharey=True)

# 将循环 10 的子图放在最后一个位置
desired_loops = [1, 2, 5, 10]
# retrieval acc@5*100 的值
retrieval_acc = [
    0.6895, 0.616, 0.5975, 0.582, 0.569, 0.5615, 0.555, 0.548, 0.545, 0.5405
]

# 创建自定义的图例标签和线
lines_for_legend = []
# 遍历每个desired_loop的子图并绘制条形图
for i, loop in enumerate(desired_loops):
    ax = axes[i]
    loop_df = df[df['Loop'] == loop]

    # 计算EM=0和EM=1的总数
    em0_total = loop_df[loop_df['EM'] == 'EM=0']['Avg_Query_Count'].sum()
    em1_total = loop_df[loop_df['EM'] == 'EM=1']['Avg_Query_Count'].sum()

    # 绘制条形图
    sns.barplot(x='Right_Num', y='Avg_Query_Count', hue='EM', data=loop_df, ax=ax, palette=palette)

    # 在图上添加代表EM=0总数和EM=1总数的直线
    em0_line = ax.axhline(em0_total, color="#7F8FDD", linestyle='--', lw=2)
    em1_line = ax.axhline(em1_total, color="#6A3D9A", linestyle='--', lw=2)

    # 添加直线的标签
    ax.text(0.9, em0_total - 8, f'{em0_total:.2f}', color="#7F8FDD", ha='right', transform=ax.get_yaxis_transform())
    ax.text(0.93, em1_total + 4, f'{em1_total:.2f}', color="#6A3D9A", ha='right', transform=ax.get_yaxis_transform())

    # 添加代表retrieval acc@5*100的直线
    if loop == 10:  # 调整索引以匹配列表长度
        acc_value = retrieval_acc[-1]
    else:
        acc_value = retrieval_acc[loop - 1]
    # 添加代表retrieval acc@5*100的直线
    acc_line = ax.axhline(acc_value * 100, color='#39375B', linestyle=':', lw=2)
    ax.text(0.9, acc_value * 100 + 4, f'{acc_value * 100:.2f}', color='#39375B', ha='right',
            transform=ax.get_yaxis_transform())

    # ax.set_title(f'Iteration {loop}', pad=24)  # 增加标题的垂直间距
    # ax.set_ylabel('Average Query Num' if i == 0 else '')  # 增加 y 轴标签的垂直间距
    # ax.set_xlabel('Context Right Num', labelpad=10)  # 增加 x 轴标签的垂直间距
    ax.set_title(f'迭代 {loop}', fontproperties=my_font, pad=24)  # 增加标题的垂直间距
    ax.set_ylabel('平均查询数量' if i == 0 else '', fontproperties=my_font)  # 增加 y 轴标签的垂直间距
    ax.set_xlabel('正确上下文数量', labelpad=10, fontproperties=my_font)  # 增加 x 轴标签的垂直间距
    if i < len(axes):
        ax.legend().set_visible(False)
    # 如果是最后一个subplot，我们将直线添加到图例
    if i == len(desired_loops) - 1:
        lines_for_legend.append(em0_line)
        lines_for_legend.append(em1_line)
        lines_for_legend.append(acc_line)
# 在整个图形的底部中心位置添加一个图例，现在包括我们的自定义线
handles, labels = axes[-1].get_legend_handles_labels()
custom_labels = ["EM=0 Total", "EM=1 Total", "Acc@5"]
handles.extend(lines_for_legend)  # 将自定义的线添加进图例句柄
labels.extend(custom_labels)  # 添加自定义标签

fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.13, 0.6), ncol=1)

# 调整子图之间的间距和边缘
plt.tight_layout()

# 调整子图的位置以适应图例
fig.subplots_adjust(bottom=0.15)

# 显示图表
plt.show()
plt.savefig('png_tsvs/draw_context.png', dpi=300, bbox_inches='tight')