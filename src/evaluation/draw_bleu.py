import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 指定字体路径
my_font = FontProperties(fname='/home/xiaoyang2020/chenxiaoyang_11/Rob_LLM/externals/popular-fonts/微软雅黑.ttf')
datasets = ['nq', 'webq', 'pop', 'tqa']

for dataset in datasets:
    # 读取TSV文件
    file_path = f'png_tsvs/bleu/{dataset}_bleu_3.tsv'  # 替换成你的TSV文件路径
    df = pd.read_csv(file_path, sep='\t')
    #converting columns names to string type
    df.columns = df.columns.astype(str)
    df = df.rename(columns={'0': 'Ori.'})

    # 重命名方法
    #bm25
    # contriever
    # bge-base
    # llm-embedder
    # bm25+upr
    # bm25+monot5
    # bm25+bge
    # bge-base+upr
    # bge-base+monot5
    # bge-base+bge
    df['Method'] = df['Method'].replace('bm25', 'BM25')
    df['Method'] = df['Method'].replace('contriever', 'Contri')
    df['Method'] = df['Method'].replace('bge-base', 'BGE-B')
    df['Method'] = df['Method'].replace('llm-embedder', 'LLM-E')
    df['Method'] = df['Method'].replace('bm25+upr', 'BM25+U')
    df['Method'] = df['Method'].replace('bm25+monot5', 'BM25+M')
    df['Method'] = df['Method'].replace('bm25+bge', 'BM25+BR')
    df['Method'] = df['Method'].replace('bge-base+upr', 'BGE-B+U')
    df['Method'] = df['Method'].replace('bge-base+monot5', 'BGE-B+M')
    df['Method'] = df['Method'].replace('bge-base+bge', 'BGE-B+BR')

    # 转换DataFrame为长格式
    df_long = df.melt(id_vars=['Method'], var_name='loop', value_name='Self-BLEU')

    # 将loop列的数据类型转换为整数，以便在图表中正确排序
    df_long['loop'] = df_long['loop'].astype(str)
    sns.set_theme(style="ticks", rc={'axes.formatter.limits': (-4, 5)}, font_scale=2.6)
    # 画图
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.figure(figsize=(7.5, 7.5))  # 可以调整图片大小
    if dataset != 'pop':# and dataset != 'tqa':
        sns.lineplot(data=df_long, x='loop', y='Self-BLEU', hue='Method', palette="mako", legend=False)
    else:
        sns.lineplot(data=df_long, x='loop', y='Self-BLEU', hue='Method', palette="mako")
    # plt.title('Self-BLEU Values per Method Over Iterations')  # 可以自定义标题
    sns.despine()
    plt.xlabel('迭代', fontproperties=my_font, fontsize=30, fontweight='bold',)  # X轴标签
    plt.ylabel('Self-BLEU')  # Y轴标签
    if dataset == 'pop':# or dataset == 'tqa':
        plt.legend(loc='upper right', bbox_to_anchor=(1.77, 0.93), fontsize=25)  # 图例

    # plt.tight_layout()  # 调整布局
    plt.show()  # 显示图表
    plt.savefig(f'png_tsvs/bleu/{dataset}_bleu.png', bbox_inches='tight')  # 保存图表到文件
