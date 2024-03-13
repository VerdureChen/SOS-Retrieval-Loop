import pandas as pd
import matplotlib.pyplot as plt



# 由于表头是多级的，我们首先定义列名
columns = [
    'nq', 'no_filter_BM25', 'no_filter_BGE-Base', 'no_filter_Contriever', 'no_filter_LLM-Embedder',
    'filter_bleu_BM25', 'filter_bleu_BGE-Base', 'filter_bleu_Contriever', 'filter_bleu_LLM-Embedder',
    'filter_source_BM25', 'filter_source_BGE-Base', 'filter_source_Contriever', 'filter_source_LLM-Embedder'
]

# 读取数据，跳过前两行
# 假设TSV文件是tab分隔，并且位于当前文件夹中
df = pd.read_csv('filter_ret.tsv', sep='\t', skiprows=2, names=columns)

# 设置绘图
plt.figure(figsize=(14, 8))
plt.title('Accuracy over Loops')
plt.xlabel('Loops')
plt.ylabel('Accuracy @5')
print(df)
# 定义每种方法的颜色，确保相同方法使用相同颜色
colors = {
    'BM25': 'red',
    'BGE-Base': 'blue',
    'Contriever': 'green',
    'LLM-Embedder': 'purple'
}

# 绘制线条，并确保图例中每个方法只出现一次
for method, color in colors.items():
    plt.plot(x, df[f'no_filter_{method}'], color=color, label=method)
    plt.plot(x, df[f'filter_bleu_{method}'], color=color, linestyle='--')
    plt.plot(x, df[f'filter_source_{method}'], color=color, linestyle='-.')

# 为了在图例中只显示方法名称（而不是每条线），我们使用以下技巧：
# 绘制一个不可见的数据点，但带有标签，用于图例
for method, color in colors.items():
    plt.plot([], [], color=color, label=method)

# 显示图例
plt.legend()

# 显示网格线
plt.grid(True)




# 存储图像
plt.savefig('png_tsvs/filter_retrieval/filter_ret.png')
