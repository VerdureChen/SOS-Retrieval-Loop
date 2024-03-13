import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置主题
sns.set_theme(style="ticks", font_scale=2.3)
figsize = (7.5, 7.5)

# “right answer”的数据，之前已提供的数据
data_right = {
    'loop': ['Ori.', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'BM25': [0.383, 0.27, 0.302, 0.303, 0.306, 0.303, 0.306, 0.309, 0.309, 0.31, 0.31],
    'BGE-Base': [0.511, 0.227, 0.224, 0.241, 0.24, 0.235, 0.239, 0.244, 0.239, 0.235, 0.241],
    'Contriever': [0.476, 0.198, 0.208, 0.224, 0.229, 0.238, 0.231, 0.233, 0.233, 0.223, 0.231],
    'LLM-Embedder': [0.514, 0.257, 0.305, 0.301, 0.316, 0.309, 0.316, 0.311, 0.311, 0.311, 0.324],
}

# “wrong answer”的新数据
data_wrong = {
    'loop': ['Ori.', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'BM25': [0.013, 0.281, 0.28, 0.274, 0.271, 0.26, 0.254, 0.259, 0.253, 0.253, 0.244],
    'BGE-Base': [0.016, 0.475, 0.469, 0.44, 0.435, 0.441, 0.431, 0.421, 0.417, 0.411, 0.42],
    'Contriever': [0.008, 0.497, 0.503, 0.478, 0.474, 0.476, 0.464, 0.467, 0.448, 0.455, 0.463],
    'LLM-Embedder': [0.015, 0.422, 0.412, 0.391, 0.377, 0.375, 0.36, 0.365, 0.362, 0.364, 0.371],
}

# 创建数据框
df_right = pd.DataFrame(data_right)
df_wrong = pd.DataFrame(data_wrong)

# 更新列名和数值格式
df_right.columns = [col.replace('loop1', '+LLM$_z$') for col in df_right.columns]
df_right.columns = [col.replace('origin', 'Ori.') for col in df_right.columns]
df_right.columns = [col.replace('LLM-Embedder', 'LLM-E') for col in df_right.columns]
df_right.columns = [col.replace('BGE-Base', 'BGE') for col in df_right.columns]
df_right.columns = [col.replace('Contriever', 'Contri') for col in df_right.columns]
df_wrong.columns = [col.replace('loop1', '+LLM$_z$') for col in df_wrong.columns]
df_wrong.columns = [col.replace('origin', 'Ori.') for col in df_wrong.columns]
df_wrong.columns = [col.replace('LLM-Embedder', 'LLM-E') for col in df_wrong.columns]
df_wrong.columns = [col.replace('BGE-Base', 'BGE') for col in df_wrong.columns]
df_wrong.columns = [col.replace('Contriever', 'Contri') for col in df_wrong.columns]

# 将acc@5的百分比数值转换为百分比形式并标记为“right”或“wrong”
# df_right[df_right.columns[1:]] = df_right[df_right.columns[1:]].applymap(lambda x: round(x * 100, 1))
# df_wrong[df_wrong.columns[1:]] = df_wrong[df_wrong.columns[1:]].applymap(lambda x: round(x * 100, 1))
df_right[df_right.columns[1:]] = df_right[df_right.columns[1:]].applymap(lambda x: round(x * 100, 1))
df_wrong[df_wrong.columns[1:]] = df_wrong[df_wrong.columns[1:]].applymap(lambda x: round(x * 100, 1))
df_right['Answer Type'] = 'Correct'
df_wrong['Answer Type'] = 'Incorrect'

# 合并两个数据框
df_combined = pd.concat([df_right, df_wrong])

# # 对合并后的数据框进行'melt'操作，准备绘图数据
# df_plot_combined = df_combined.melt(id_vars=['loop', 'Answer Type'], var_name='Model', value_name='acc@5')
#
#
# # 设置字体加粗
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titleweight'] = 'bold'
#
# # 绘制折线图
# plt.figure(figsize=figsize)
# plt.ylim(0, 75)
# plt.xticks(rotation=0)
# # 使用不同的色调区分两组数据
# palette_right = sns.color_palette("mako", n_colors=len(df_right.columns) - 2)
# palette_wrong = sns.color_palette("mako", n_colors=len(df_wrong.columns) - 2)
#
# # 创建一个图和一个轴对象
# fig, ax = plt.subplots()
#
# # 这里假设df_plot_combined是您的DataFrame，并且已经正确设置了
# # 绘制“right answer”的数据
# right_lines = sns.lineplot(ax=ax, data=df_plot_combined[df_plot_combined['Answer Type'] == 'Correct'], x='loop', y='acc@5', hue='Model',
#                            style='Model', markers='o', dashes=False, palette=palette_right, markersize=20)
#
# # 获取“right answer”的线条句柄和标签
# handles_right, labels_right = ax.get_legend_handles_labels()
#
#
#
# # 绘制“wrong answer”的数据
# wrong_lines = sns.lineplot(ax=ax, data=df_plot_combined[df_plot_combined['Answer Type'] == 'Incorrect'], x='loop', y='acc@5', hue='Model',
#                            style='Model', markers='X', dashes=False, palette=palette_wrong, markersize=20)
#
# # 获取所有的线条句柄和标签
# handles, labels = ax.get_legend_handles_labels()
#
#
#
#
#
# # 移除自动生成的图例
# ax.legend([],[], frameon=False)
#
# # 获取模型和答案类型的唯一值
# models = df_plot_combined['Model'].unique().tolist()
# answer_types = ['Correct', 'Incorrect']
#
# # 创建模型的图例
# palette_combined = palette_right + palette_wrong
# model_handles = [plt.Line2D([], [], color=palette_combined[i], linestyle='-', markersize=20) for i in range(len(models))]
# model_legend = ax.legend(handles=model_handles, labels=models, title='Model', loc='lower center', bbox_to_anchor=(0.45, -0.77), ncol=4, fontsize=20, title_fontsize=20, frameon=False)
#
# # 创建答案类型的图例
# type_markers = ['o', 'X']  # 和答案类型数量匹配
# type_handles = [plt.Line2D([], [], color='black', marker=type_markers[i], linestyle='', markersize=20) for i in range(len(answer_types))]
# type_legend = ax.legend(handles=type_handles, labels=answer_types, title='Answer Type', loc='lower center', bbox_to_anchor=(0.45, -0.54), ncol=2, fontsize=20, title_fontsize=20, frameon=False)
#
# # 将模型的图例再次添加到图表中，因为创建答案类型图例时会被覆盖
# ax.add_artist(model_legend)
# # 添加标题和标签
# # plt.title("Acc@5 vs Loop")
# plt.xlabel("Iteration")
#
# plt.ylabel('acc@5 (%)')
#
# # 调整整体布局，为外部图例预留空间
# # plt.subplots_adjust(bottom=0.4)
#
# # 保存整个图形
# plt.savefig('draw_qa.png', bbox_extra_artists=(model_legend, type_legend), bbox_inches='tight', dpi=300)


# WebQ数据集的“right answer”数据
data_right_webq = {
    'loop': ['Ori.', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'BM25': [0.389, 0.325, 0.337, 0.333, 0.327, 0.345, 0.341, 0.327, 0.339, 0.324, 0.325],
    'BGE-Base': [0.441, 0.299, 0.286, 0.293, 0.281, 0.272, 0.275, 0.279, 0.28, 0.267, 0.271],
    'Contriever': [0.458, 0.317, 0.323, 0.331, 0.334, 0.329, 0.332, 0.326, 0.334, 0.332, 0.335],
    'LLM-Embedder': [0.454, 0.338, 0.361, 0.364, 0.382, 0.384, 0.377, 0.38, 0.378, 0.39, 0.381],
}

# WebQ数据集的“wrong answer”数据
data_wrong_webq = {
    'loop': ['Ori.', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'BM25': [0.041, 0.215, 0.206, 0.206, 0.205, 0.21, 0.198, 0.199, 0.197, 0.196, 0.176],
    'BGE-Base': [0.058, 0.326, 0.295, 0.293, 0.291, 0.293, 0.28, 0.291, 0.288, 0.281, 0.28],
    'Contriever': [0.051, 0.371, 0.357, 0.355, 0.339, 0.346, 0.332, 0.346, 0.346, 0.339, 0.34],
    'LLM-Embedder': [0.053, 0.297, 0.274, 0.274, 0.264, 0.272, 0.272, 0.275, 0.261, 0.266, 0.257],
}

# 对WebQ的数据执行相同的处理步骤
df_right_webq = pd.DataFrame(data_right_webq)
df_wrong_webq = pd.DataFrame(data_wrong_webq)

# 更新列名和数值格式
df_right_webq.columns = [col.replace('LLM-Embedder', 'LLM-E') for col in df_right_webq.columns]
df_right_webq.columns = [col.replace('BGE-Base', 'BGE') for col in df_right_webq.columns]
df_right_webq.columns = [col.replace('Contriever', 'Contri') for col in df_right_webq.columns]
df_wrong_webq.columns = [col.replace('LLM-Embedder', 'LLM-E') for col in df_wrong_webq.columns]
df_wrong_webq.columns = [col.replace('BGE-Base', 'BGE') for col in df_wrong_webq.columns]
df_wrong_webq.columns = [col.replace('Contriever', 'Contri') for col in df_wrong_webq.columns]

# 将acc@5的数值转换为百分比形式并标记为“right”或“wrong”
df_right_webq[df_right_webq.columns[1:]] = df_right_webq[df_right_webq.columns[1:]].applymap(lambda x: round(x * 100, 1))
df_wrong_webq[df_wrong_webq.columns[1:]] = df_wrong_webq[df_wrong_webq.columns[1:]].applymap(lambda x: round(x * 100, 1))
df_right_webq['Answer Type'] = 'Correct'
df_wrong_webq['Answer Type'] = 'Incorrect'

# 合并两个数据框
df_combined_webq = pd.concat([df_right_webq, df_wrong_webq])


# TriviaQA数据集的“right answer”数据
data_right_triviaqa = {
    'loop': ['Ori.', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'BM25': [0.612, 0.393, 0.437, 0.438, 0.447, 0.453, 0.446, 0.442, 0.447, 0.445, 0.443],
    'BGE-Base': [0.631, 0.343, 0.358, 0.37, 0.366, 0.367, 0.363, 0.357, 0.369, 0.367, 0.365],
    'Contriever': [0.644, 0.371, 0.387, 0.399, 0.405, 0.397, 0.398, 0.392, 0.401, 0.402, 0.4],
    'LLM-Embedder': [0.603, 0.357, 0.389, 0.393, 0.395, 0.399, 0.41, 0.417, 0.413, 0.41, 0.423],
}

# TriviaQA数据集的“wrong answer”数据
data_wrong_triviaqa = {
    'loop': ['Ori.', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'BM25': [0.017, 0.393, 0.378, 0.361, 0.369, 0.362, 0.362, 0.349, 0.349, 0.358, 0.349],
    'BGE-Base': [0.021, 0.406, 0.412, 0.388, 0.389, 0.396, 0.388, 0.381, 0.382, 0.39, 0.38],
    'Contriever': [0.019, 0.434, 0.433, 0.442, 0.438, 0.428, 0.43, 0.414, 0.415, 0.41, 0.418],
    'LLM-Embedder': [0.022, 0.392, 0.399, 0.403, 0.397, 0.392, 0.385, 0.391, 0.377, 0.367, 0.373],
}

# 对TriviaQA的数据执行相同的处理步骤
df_right_triviaqa = pd.DataFrame(data_right_triviaqa)
df_wrong_triviaqa = pd.DataFrame(data_wrong_triviaqa)

# 更新列名和数值格式
df_right_triviaqa.columns = [col.replace('LLM-Embedder', 'LLM-E') for col in df_right_triviaqa.columns]
df_right_triviaqa.columns = [col.replace('BGE-Base', 'BGE') for col in df_right_triviaqa.columns]
df_right_triviaqa.columns = [col.replace('Contriever', 'Contri') for col in df_right_triviaqa.columns]
df_wrong_triviaqa.columns = [col.replace('LLM-Embedder', 'LLM-E') for col in df_wrong_triviaqa.columns]
df_wrong_triviaqa.columns = [col.replace('BGE-Base', 'BGE') for col in df_wrong_triviaqa.columns]
df_wrong_triviaqa.columns = [col.replace('Contriever', 'Contri') for col in df_wrong_triviaqa.columns]

# 将acc@5的数值转换为百分比形式并标记为“right”或“wrong”
df_right_triviaqa[df_right_triviaqa.columns[1:]] = df_right_triviaqa[df_right_triviaqa.columns[1:]].applymap(lambda x: round(x * 100, 1))
df_wrong_triviaqa[df_wrong_triviaqa.columns[1:]] = df_wrong_triviaqa[df_wrong_triviaqa.columns[1:]].applymap(lambda x: round(x * 100, 1))
df_right_triviaqa['Answer Type'] = 'Correct'
df_wrong_triviaqa['Answer Type'] = 'Incorrect'

# 合并两个数据框
df_combined_triviaqa = pd.concat([df_right_triviaqa, df_wrong_triviaqa])


# PopQA数据集的“right answer”数据
data_right_popqa = {
    'loop': ['Ori.', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'BM25': [0.3, 0.195, 0.206, 0.209, 0.203, 0.211, 0.203, 0.208, 0.204, 0.2, 0.197],
    'BGE-Base': [0.487, 0.182, 0.182, 0.174, 0.176, 0.178, 0.19, 0.177, 0.178, 0.18, 0.181],
    'Contriever': [0.36, 0.194, 0.192, 0.215, 0.218, 0.22, 0.217, 0.214, 0.213, 0.217, 0.216],
    'LLM-Embedder': [0.486, 0.208, 0.235, 0.233, 0.233, 0.226, 0.227, 0.226, 0.222, 0.222, 0.213],
}

# PopQA数据集的“wrong answer”数据
data_wrong_popqa = {
    'loop': ['Ori.', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'BM25': [0.033, 0.29, 0.299, 0.28, 0.296, 0.279, 0.293, 0.288, 0.283, 0.289, 0.29],
    'BGE-Base': [0.039, 0.534, 0.531, 0.525, 0.531, 0.528, 0.516, 0.522, 0.511, 0.511, 0.512],
    'Contriever': [0.039, 0.579, 0.588, 0.597, 0.588, 0.576, 0.586, 0.58, 0.581, 0.582, 0.577],
    'LLM-Embedder': [0.041, 0.533, 0.541, 0.542, 0.548, 0.535, 0.537, 0.546, 0.553, 0.554, 0.541],
}

# 对PopQA的数据执行相同的处理步骤
df_right_popqa = pd.DataFrame(data_right_popqa)
df_wrong_popqa = pd.DataFrame(data_wrong_popqa)

# 更新列名和数值格式
df_right_popqa.columns = [col.replace('LLM-Embedder', 'LLM-E') for col in df_right_popqa.columns]
df_right_popqa.columns = [col.replace('BGE-Base', 'BGE') for col in df_right_popqa.columns]
df_right_popqa.columns = [col.replace('Contriever', 'Contri') for col in df_right_popqa.columns]
df_wrong_popqa.columns = [col.replace('LLM-Embedder', 'LLM-E') for col in df_wrong_popqa.columns]
df_wrong_popqa.columns = [col.replace('BGE-Base', 'BGE') for col in df_wrong_popqa.columns]
df_wrong_popqa.columns = [col.replace('Contriever', 'Contri') for col in df_wrong_popqa.columns]

# 将acc@5的数值转换为百分比形式并标记为“right”或“wrong”
df_right_popqa[df_right_popqa.columns[1:]] = df_right_popqa[df_right_popqa.columns[1:]].applymap(lambda x: round(x * 100, 1))
df_wrong_popqa[df_wrong_popqa.columns[1:]] = df_wrong_popqa[df_wrong_popqa.columns[1:]].applymap(lambda x: round(x * 100, 1))
df_right_popqa['Answer Type'] = 'Correct'
df_wrong_popqa['Answer Type'] = 'Incorrect'

# 合并两个数据框
df_combined_popqa = pd.concat([df_right_popqa, df_wrong_popqa])

# 设置主题
sns.set_theme(style="ticks", font_scale=2.2)

# 设置图的大小
figsize = (27, 7.5)

# 设置字体加粗
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# 创建一个图和四个轴对象
fig, axs = plt.subplots(1, 4, figsize=figsize, sharey=True)

# 定义统一的颜色组合
palette = sns.color_palette("mako", n_colors=4)

# 循环绘制每个数据集的折线图
datasets = {
    'NQ': df_combined,
    'WebQ': df_combined_webq,
    'TriviaQA': df_combined_triviaqa,
    'PopQA': df_combined_popqa
}

for i, (dataset_name, df_combined) in enumerate(datasets.items()):
    df_plot_combined = df_combined.melt(id_vars=['loop', 'Answer Type'], var_name='Model', value_name='acc@5')
    palette_right = sns.color_palette("mako", n_colors=len(df_combined.columns) - 2)
    palette_wrong = sns.color_palette("mako", n_colors=len(df_combined.columns) - 2)
    palette_combined = palette_right + palette_wrong

    # 绘制“right answer”的数据
    right_lines = sns.lineplot(ax=axs[i], data=df_plot_combined[df_plot_combined['Answer Type'] == 'Correct'], x='loop',
                               y='acc@5', hue='Model',
                               style='Model', markers='o', dashes=False, palette=palette_right, markersize=22, linewidth=3)
    # 绘制“wrong answer”的数据
    wrong_lines = sns.lineplot(ax=axs[i], data=df_plot_combined[df_plot_combined['Answer Type'] == 'Incorrect'], x='loop',
                               y='acc@5', hue='Model',
                               style='Model', markers='X', dashes=False, palette=palette_wrong, markersize=22, linewidth=3)
    axs[i].set_title(dataset_name)
    # axs[i].set_ylim(0, 75)
    axs[i].set_xlabel('Iteration')
    if i == 0:
        axs[i].set_ylabel('EMllm')
    axs[i].legend([], [], frameon=False)

# 移除自动生成的图例
plt.legend([], [], frameon=False)

# 获取模型和答案类型的唯一值
models = df_plot_combined['Model'].unique().tolist()
answer_types = ['Correct', 'Incorrect']

# 创建模型的图例
model_handles = [plt.Line2D([], [], color=palette_combined[i], linestyle='-', markersize=20) for i in range(len(models))]
# # 注意这里我们没有关联ax，而是使用了plt.figlegend来创建整个画布的图例
# model_legend = plt.figlegend(handles=model_handles, labels=models, title='Model', loc='upper right', bbox_to_anchor=(1.07, 0.7), ncol=1, fontsize=22, title_fontsize=22, frameon=False)
#
# 创建答案类型的图例
type_markers = ['o', 'X']  # 和答案类型数量匹配
type_handles = [plt.Line2D([], [], color='black', marker=type_markers[i], linestyle='', markersize=20) for i in range(len(answer_types))]
# type_legend = plt.figlegend(handles=type_handles, labels=answer_types, title='Answer', loc='upper right', bbox_to_anchor=(1.07, 0.9), ncol=1, fontsize=22, title_fontsize=22, frameon=False)
# # 将模型的图例再次添加到图表中，因为创建答案类型图例时会被覆盖
# plt.gca().add_artist(model_legend)

# 合并所有图例
plt.figlegend(handles=type_handles+model_handles, labels=  answer_types+models, loc='upper right', bbox_to_anchor=(1.10, 0.75), ncol=1, fontsize=22, title_fontsize=22, frameon=True)


# 调整整体布局
plt.tight_layout(rect=[0, 0.1, 1, 1])

# 保存整个图形
plt.savefig('combined_qa_datasets.png', dpi=300, bbox_inches='tight')