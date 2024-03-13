import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 指定字体路径
my_font = FontProperties(fname='/home/xiaoyang2020/chenxiaoyang_11/Rob_LLM/externals/popular-fonts/微软雅黑.ttf')
# 设置seaborn的主题
sns.set_theme(style="ticks")
# 读取数据，假设TSV文件的分隔符是制表符（默认）
df = pd.read_csv('/home/xiaoyang2020/chenxiaoyang_11/Rob_LLM/src/evaluation/png_tsvs/percentage/plot_webq_tsv.tsv', delimiter='\t')
#replace gpt-3.5-turbo to GPT-3.5-Turbo, baichuan2-13b-chat to Baichuan2-13B-Chat, qwen-14b-chat to Qwen-14B-Chat, chatglm3-6b to ChatGPT3-6B, llama2-13b-chat to Llama2-13B-Chat, human to Human
df['generate_model_name'] = df['generate_model_name'].replace(['gpt-3.5-turbo', 'baichuan2-13b-chat', 'qwen-14b-chat', 'chatglm3-6b', 'llama2-13b-chat', 'human'], ['ChatGPT', 'Baichuan2', 'Qwen', 'ChatGLM3', 'Llama2', 'Human'])
df = df.melt(id_vars='generate_model_name', var_name='Iteration', value_name='Percentage')
#all value*100 and keep 1 decimal places
df['Percentage'] = df['Percentage'].apply(lambda x: round(x*100, 1))
# 初始化一个空的列表用来存储每个条形的底部位置
bottoms = [0] * len(df['Iteration'].unique())


# 颜色字典，使用十六进制颜色代码
# gpt-3.5-turbo
# baichuan2-13b-chat
# qwen-14b-chat
# chatglm3-6b
# llama2-13b-chat
# human
# colors = {
#     'gpt-3.5-turbo': '#CAB2D5',
#     'baichuan2-13b-chat': '#FEBE6F',
#     'qwen-14b-chat': '#FC9A98',
#     'chatglm3-6b': '#B2DF8A',
#     'llama2-13b-chat': '#6D759F',
#     'human': '#A6CEE3'
# }
# colors = {
#     'gpt-3.5-turbo': '#6A3D9A',
#     'baichuan2-13b-chat': '#9977B7',
#     'qwen-14b-chat': '#9D7BB9',
#     'chatglm3-6b': '#B599C8',
#     'llama2-13b-chat': '#CAB2D5',
#     'human': '#7F8FDD'
# }
# colors = {
#     'GPT-3.5-Turbo': '#6A3D9A',
#     'Baichuan2-13B-Chat': '#9977B7',
#     'Qwen-14B-Chat': '#9D7BB9',
#     'ChatGPT3-6B': '#B599C8',
#     'Llama2-13B-Chat': '#CAB2D5',
#     'Human': '#7F8FDD'
# }

colors ={
    'ChatGPT': '#6A3D9A',
    'Baichuan2': '#9977B7',
    'Qwen': '#9D7BB9',
    'ChatGLM3': '#B599C8',
    'Llama2': '#CAB2D5',
    'Human': '#7F8FDD'
}

# 设置图表大小
plt.figure(figsize=(7.5, 7.5))

# 为每个模型绘制一个条形
for i, model_name in enumerate(df['generate_model_name'].unique()):
    # 从DataFrame中抽取当前模型的数据
    model_data = df[df['generate_model_name'] == model_name]

    # 绘制条形
    plt.bar(
        model_data['Iteration'],  # X轴坐标
        model_data['Percentage'],  # 条形高度
        bottom=bottoms,  # 设置条形的起始位置
        color=colors.get(model_name, '#808080'),
        label=model_name,
    )

    # 更新下一组条形的起始位置
    bottoms = [bottoms[j] + model_data['Percentage'].values[j] for j in range(len(bottoms))]

# 在最右侧的条形的右侧中间添加百分比文本
last_iteration = df['Iteration'].unique()[-1]  # 获取最后一次迭代
last_iteration_data = df[df['Iteration'] == last_iteration]  # 获取最后一次迭代的数据

# 需要确定最后一个条形图的x轴位置
last_bar_index = len(df['Iteration'].unique()) - 1  # 获取最后一个条形图的索引


# 在最右侧的条形的右侧中间添加百分比文本
for i, (model_name, percentage) in enumerate(zip(last_iteration_data['generate_model_name'], last_iteration_data['Percentage'])):
    # 如果bottoms是累积的，需要先找到这个模型的开始高度
    if i == 0:
        # 第一个模型的底部是0
        bottom_height = 0
    else:
        # 其他模型的底部是前一个模型的底部加上前一个模型的百分比
        bottom_height = bottom_height + last_iteration_data['Percentage'].values[i - 1]
    # 计算中点位置
    middle_height = bottom_height + (percentage / 2.0)
    print(f"{model_name}的底部位置是{bottom_height}")
    print(f"{model_name}的高度是{percentage}")
    print(f"{model_name}的中点位置是{middle_height}")

    # # 在条形的右侧中间添加文本
    # plt.text(
    #     x=last_bar_index + 0.4,  # X轴位置稍微偏右，用索引加上一个小的偏移量
    #     y=middle_height,  # Y轴位置为条形的中间位置
    #     s=f"{percentage}%",  # 显示的文本
    #     va='center',  # 垂直居中
    #     ha='left',  # 水平向左对齐
    # )

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
# plt.title('Percentage of Text', fontsize=20)  # 标题字体大小
plt.xlabel('迭代', fontsize=30,  # X轴标签字体大小加粗
              fontweight='bold', fontproperties=my_font)
plt.ylabel('百分比', fontsize=30,  fontweight='bold', fontproperties=my_font)
           # Y轴标签字体大小
# plt.legend(title='bleu', fontsize=14, # 图例字体大小
#            title_fontsize=14)   # 图例标题字体大小，加粗
# plt.legend(loc='upper right', bbox_to_anchor=(1.62, 0.7), fontsize=25)

plt.xticks(fontsize=30, fontweight='bold')  # X轴刻度字体大小
plt.yticks(fontsize=30, fontweight='bold')  # Y轴刻度字体大小
# 图表标题
# plt.title('Percentage of Text ')

# X轴和Y轴标签
# plt.xlabel('Iteration')
# plt.ylabel('Percentage')

# 显示图例
# plt.legend(title='bleu')

# 显示图表
plt.show()

#save
plt.savefig('/home/xiaoyang2020/chenxiaoyang_11/Rob_LLM/src/evaluation/png_tsvs/percentage/chinese_plot_webq_tsv.png', dpi=300, bbox_inches='tight')