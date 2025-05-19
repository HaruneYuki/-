import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据并解析
df = pd.read_csv('frequent_itemsets.csv', header=None, names=['support', 'itemset'])
df['itemset'] = df['itemset'].apply(ast.literal_eval)

# 生成可读的标签（处理多类别组合）
def format_label(itemset):
    methods = [x.replace('method_', '') for x in itemset if x.startswith('method_')]
    categories = [x.replace('category_', '') for x in itemset if x.startswith('category_')]
    return f"{' & '.join(methods)} → {', '.join(categories)}"  # 组合标签格式

df['label'] = df['itemset'].apply(format_label)


# 按支持度降序排序
df_sorted = df.sort_values('support', ascending=False)

# 筛选包含支付方式且支持度排名前15的项集
top_n = 15
df_top = df_sorted[
    df_sorted['itemset'].apply(lambda s: any(x.startswith('method_') for x in s))
].head(top_n)


plt.figure(figsize=(12, 8))
sns.barplot(
    data=df_top,
    y='label', x='support',
    palette='viridis',  # 根据支持度渐变颜色
    edgecolor='black'
)

# 添加数值标签
for i, (support, label) in enumerate(zip(df_top['support'], df_top['label'])):
    plt.text(
        support + 0.003,  # 横向偏移量
        i,                # 纵向位置
        f"{support:.4f}", # 格式化数值
        va='center',
        fontsize=10
    )

# 优化图表样式
plt.title(f"Top {top_n} 支付方式与商品组合的支持度排名", fontsize=14, pad=20)
plt.xlabel("支持度", fontsize=12)
plt.ylabel("")
plt.xlim(0, df_top['support'].max() * 1.2)  # 扩展X轴范围
plt.grid(axis='x', linestyle='--', alpha=0.7)
sns.despine(left=True)  # 隐藏左轴
plt.tight_layout()
plt.savefig('top_support_combinations.png', dpi=300, bbox_inches='tight')
