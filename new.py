import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import gc
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import os
from itertools import islice
# from efficient_apriori import apriori
import pyarrow.parquet as pq  
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth

# 初始化商品目录映射
with open('product_catalog.json', 'r', encoding='utf-8') as f:
    catalog = json.load(f)
id_to_category = {p['id']: p['category'] for p in catalog['products']}
id_to_price = {p['id']: p['price'] for p in catalog['products']}



plt.rcParams["font.sans-serif"] = ["SimHei"]  # 或 ["Microsoft YaHei", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False    # 解决负号显示问题

CATEGORY_MAPPING = {
    # 电子产品
    '智能手机': '电子产品', 
    '笔记本电脑': '电子产品',
    '平板电脑': '电子产品',
    '智能手表': '电子产品',
    '耳机': '电子产品',
    '音响': '电子产品',
    '相机': '电子产品',
    '摄像机': '电子产品',
    '游戏机': '电子产品',
    
    # 服装
    '上衣': '服装',
    '裤子': '服装',
    '裙子': '服装',
    '内衣': '服装',
    '鞋子': '服装',
    '帽子': '服装',
    '手套': '服装',
    '围巾': '服装',
    '外套': '服装',
    
    # 食品
    '零食': '食品',
    '饮料': '食品',
    '调味品': '食品',
    '米面': '食品',
    '水产': '食品',
    '肉类': '食品',
    '蛋奶': '食品',
    '水果': '食品',
    '蔬菜': '食品',
    
    # 家居
    '家具': '家居',
    '床上用品': '家居',
    '厨具': '家居',
    '卫浴用品': '家居',
    
    # 办公
    '文具': '办公',
    '办公用品': '办公',
    
    # 运动户外
    '健身器材': '运动户外',
    '户外装备': '运动户外',
    
    # 玩具
    '玩具': '玩具',
    '模型': '玩具',
    '益智玩具': '玩具',
    
    # 母婴
    '婴儿用品': '母婴',
    '儿童课外读物': '母婴',
    
    # 汽车用品
    '车载电子': '汽车用品',
    '汽车装饰': '汽车用品'
}

# 创建临时文件夹
os.makedirs('tmp', exist_ok=True)
os.makedirs('test', exist_ok=True)

def process_task1(files):
    valid_ids = set(id_to_category.keys())
    transaction_path = 'test/transactions.csv'

    # Step 1. 流式生成事务文件
    with open(transaction_path, 'w') as f: pass
    
    for file in tqdm(files, desc='生成事务数据'):
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(batch_size=10000, columns=['purchase_history']):
            df_batch = batch.to_pandas()
            df_batch['purchase_history'] = df_batch['purchase_history'].apply(json.loads)
            
            with open(transaction_path, 'a') as f:
                for ph in df_batch['purchase_history']:
                    categories = set()
                    for item in ph['items']:
                        if item['id'] not in valid_ids:
                            continue
                        
                        # 原始分类名称
                        raw_category = id_to_category[item['id']]
                        
                        # 执行分类映射（带默认值处理）
                        mapped_category = CATEGORY_MAPPING.get(
                            raw_category.strip(),  # 清除前后空格
                            raw_category  # 不在映射表中的保持原样
                        )
                        categories.add(mapped_category)
                    if categories:
                        f.write(','.join(sorted(categories)) + '\n')
            del df_batch
            gc.collect()

    # 收集所有实际出现的商品类别
    all_categories = set()
    with open(transaction_path, 'r') as f:
        for line in tqdm(f, desc='收集商品类别'):
            all_categories.update(line.strip().split(','))

    
            
    # ========== 关联规则挖掘阶段 ==========
    # 稀疏矩阵编码（内存优化）
    # 分批次处理事务
    # 初始化TransactionEncoder并手动设置列
    te = TransactionEncoder()
    te.fit([all_categories]) 

    from scipy.sparse import vstack
    batch_size = 10000  # 可根据内存调整
    combined_te_ary = None

    with open(transaction_path, 'r') as f:
        while True:
            lines = list(islice(f, batch_size))
            if not lines:
                break
            transactions_batch = [line.strip().split(',') for line in lines]
            # 转换当前批次
            te_ary_batch = te.transform(transactions_batch, sparse=True)
            # 合并稀疏矩阵
            if combined_te_ary is None:
                combined_te_ary = te_ary_batch
            else:
                combined_te_ary = vstack([combined_te_ary, te_ary_batch])

     # 创建编码后的DataFrame
    encoded_df = pd.DataFrame.sparse.from_spmatrix(combined_te_ary, columns=te.columns_)


    # 挖掘频繁项集（修正后）
    frequent_itemsets = fpgrowth(encoded_df, min_support=0.02, use_colnames=True)
    
    # 生成关联规则（正确实现）
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
        # 转换itemset为可读字符串
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
        # 筛选电子产品相关规则
        electronics_rules = rules[rules['antecedents'].str.contains('电子产品') | 
                                 rules['consequents'].str.contains('电子产品')]
    else:
        electronics_rules = pd.DataFrame()

    # 保存结果
    frequent_itemsets.to_csv('task1_frequent_itemsets.csv', index=False)
    rules.to_csv('task1_rules.csv', index=False)
    electronics_rules.to_csv('task1_electronics_rules.csv', index=False)
    
    # os.remove(transaction_path)  # 可选清理
    return electronics_rules



# 任务2: 支付方式分析
def process_task2(files):
    transaction_path = 'payment_transactions.csv'
    
    # 使用生成器逐批处理（避免内存爆炸）
    with open(transaction_path, 'w') as f:
        for file in tqdm(files, desc='生成事务数据'):
            # 分批次读取（内存优化）
            parquet_file = pq.ParquetFile(file)
            for batch in parquet_file.iter_batches(batch_size=10000):
                # 列裁剪（提升性能）
                batch_df = batch.select(['purchase_history']).to_pandas()
                batch_df['purchase_history'] = batch_df['purchase_history'].apply(json.loads)
                
                # 事务构建逻辑
                for ph in batch_df['purchase_history']:
                    # 支付方式标准化
                    payment_method = f"method_{ph.get('payment_method', 'unknown')}"

                    # 商品类别提取
                    categories = set()
                    for item in ph.get('items', []):
                        raw_category = id_to_category[item['id']]
                        clean_category = CATEGORY_MAPPING.get(raw_category, raw_category)
                        categories.add(f"category_{clean_category}")
                    
                    # 写入组合事务（至少包含支付方式和1个类别）
                    if categories and payment_method.startswith('method_'):
                        transaction = [payment_method] + sorted(categories)
                        f.write(','.join(transaction) + '\n')
                
                del batch_df
                gc.collect()

    # ========== 关联规则挖掘阶段 ==========
    # 稀疏矩阵编码（内存优化）
   
    unique_columns = set()
    with open(transaction_path, 'r') as f:
        for line in tqdm(f, desc='收集唯一项'):
            items = line.strip().split(',')
            unique_columns.update(items)




    te = TransactionEncoder()
    
    te.fit([list(unique_columns)])

    from scipy.sparse import vstack
    batch_size = 10000  # 可根据内存调整
    combined_te_ary = None

    with open(transaction_path, 'r') as f:
        while True:
            lines = list(islice(f, batch_size))
            if not lines:
                break
            transactions_batch = [line.strip().split(',') for line in lines]
            # 转换当前批次
            te_ary_batch = te.transform(transactions_batch, sparse=True)
            # 合并稀疏矩阵
            if combined_te_ary is None:
                combined_te_ary = te_ary_batch
            else:
                combined_te_ary = vstack([combined_te_ary, te_ary_batch])

     # 创建编码后的DataFrame
    encoded_df = pd.DataFrame.sparse.from_spmatrix(combined_te_ary, columns=te.columns_)



    # 挖掘频繁项集（参数调优）
    frequent_itemsets = fpgrowth(
        encoded_df,
        min_support=0.01,  # 适当降低支持度
        use_colnames=True,
        # low_memory=True     # 启用内存优化模式
    )

    # 关联规则生成与筛选
    payment_rules = association_rules(
        frequent_itemsets,
        metric='confidence',
        min_threshold=0.6  
    ).pipe(lambda df: df[
        (df['antecedents'].apply(lambda x: any(s.startswith('method_') for s in x)) |
         df['consequents'].apply(lambda x: any(s.startswith('method_') for s in x)))  
    ])

    # 添加规则类型标签
    payment_rules['rule_type'] = np.where(
        payment_rules['antecedents'].apply(lambda x: any('method_' in s for s in x)),
        '支付方式→商品',
        '商品→支付方式'
    )

    frequent_itemsets.to_csv('task2_frequent_itemsets.csv', index=False)
    payment_rules.to_csv('task2_payment_rules.csv', index=False)


    return payment_rules.sort_values('lift', ascending=False)


def analyze_high_value_payments(files):
    """
    分析高价值商品（价格>5000）的支付方式分布
    :param files: Parquet文件路径列表
    :param id_to_price: 商品ID到价格的映射字典 {id_str: price_float}
    :return: 按频率降序排列的支付方式列表 [(payment_method, count)]
    """
    # ========== 预处理阶段 ==========
    # 构建高价值商品ID集合（O(1)查找优化）
    high_value_ids = {
        str(id) for id, price in id_to_price.items() 
        if isinstance(price, (int, float)) and price > 5000
    }

    # ========== 数据扫描阶段 ==========
    payment_counter = defaultdict(int)
    
    for file in files:
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(batch_size=10000):
            batch_df = batch.select(['purchase_history']).to_pandas()
            batch_df['purchase_history'] = batch_df['purchase_history'].apply(json.loads)
            
            for ph in batch_df['purchase_history']:
                # 检查是否包含高价值商品
                has_high_value = any(
                    str(item.get('id', '')).strip() in high_value_ids 
                    for item in ph.get('items', [])
                )
                
                if has_high_value:
                    # 支付方式标准化
                    raw_payment = ph.get('payment_method', 'unknown').strip().lower()
                    payment_counter[raw_payment] += 1

            del batch_df       # 释放DataFrame内存
            gc.collect()       # 立即触发垃圾回收

    # ========== 结果生成阶段 ==========
    if not payment_counter:
        return []
    
    sorted_payments = sorted(
        payment_counter.items(), 
        key=lambda x: (-x[1], x[0])  # 按频次降序、名称升序排列
    )
    
    return sorted_payments


# 任务3: 时间序列分析（使用结构化存储）
def process_task3(files):
    """分析季节性购物模式（按月统计）"""
    # 初始化按月统计数据
    monthly_total = defaultdict(int)  # 每月总商品数
    monthly_category = defaultdict(lambda: defaultdict(int))  # 每月各品类商品数

    # 流式处理数据
    for file in tqdm(files, desc='处理时间序列数据'):
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(batch_size=10000, columns=['purchase_history']):
            df_batch = batch.to_pandas()
            df_batch['purchase_history'] = df_batch['purchase_history'].apply(json.loads)
            
            for ph in df_batch['purchase_history']:
                try:
                    # 解析时间和商品
                    dt = pd.to_datetime(ph['purchase_date'])
                    month = dt.strftime('%Y-%m')  # 格式化为"年-月"
                    items = ph.get('items', [])
                except:
                    continue
                
                # 统计当月总商品数（使用item长度）
                item_count = len(items)
                monthly_total[month] += item_count
                
                # 统计各品类数量
                category_counter = defaultdict(int)
                for item in items:
                    if item['id'] not in id_to_category:
                        continue
                    raw_cat = id_to_category[item['id']]
                    mapped_cat = CATEGORY_MAPPING.get(raw_cat.strip(), raw_cat)
                    category_counter[mapped_cat] += 1
                
                # 累加到月度统计
                for cat, count in category_counter.items():
                    monthly_category[month][cat] += count

            del df_batch       # 释放DataFrame内存
            gc.collect()       # 立即触发垃圾回收

    # 转换为DataFrame
    # 总购买量
    total_df = pd.DataFrame(
        sorted(monthly_total.items(), key=lambda x: x[0]),
        columns=['month', 'total_items']
    )
    
    # 各品类购买量
    category_data = []
    for month, cats in monthly_category.items():
        for cat, count in cats.items():
            category_data.append({'month': month, 'category': cat, 'count': count})
    category_df = pd.DataFrame(category_data)
    
    # 数据透视（按月+品类）
    pivot_df = category_df.pivot_table(
        index='month', 
        columns='category', 
        values='count', 
        fill_value=0
    )

    # 可视化分析
    plt.figure(figsize=(15, 6))
    
    # 总购买量趋势
    plt.subplot(1, 2, 1)
    plt.plot(total_df['month'], total_df['total_items'], marker='o')
    plt.title('Monthly Total Items Purchased')
    plt.xlabel('Month')
    plt.ylabel('Total Items')
    plt.xticks(rotation=45)
    
    # Top5品类趋势
    plt.subplot(1, 2, 2)
    top_cats = pivot_df.sum().nlargest(5).index
    for cat in top_cats:
        plt.plot(pivot_df.index, pivot_df[cat], marker='o', label=cat)
    plt.title('Top5 Categories Trend')
    plt.xlabel('Month')
    plt.ylabel('Items Purchased')
    plt.xticks(rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('task3_monthly_analysis.png', dpi=300)
    plt.close()

    # 保存结果
    total_df.to_csv('task3_monthly_total.csv', index=False)
    pivot_df.to_csv('task3_category_monthly.csv', index=True)
    
    return {
        'total_trend': total_df,
        'category_trend': pivot_df
    }


# 任务4: 退款分析
def process_task4(files):
    """分析退款相关的商品组合模式"""
    valid_ids = set(id_to_category.keys())
    refund_keywords = ['已退款', '部分退款']
    transaction_path = 'task4_refund_transactions.csv'

    # Step 1. 生成包含退款标记的事务数据
    with open(transaction_path, 'w') as f: pass

    for file in tqdm(files, desc='生成退款事务数据'):
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(batch_size=10000, columns=['purchase_history']):
            df_batch = batch.to_pandas()
            df_batch['purchase_history'] = df_batch['purchase_history'].apply(json.loads)
            
            with open(transaction_path, 'a') as f:
                for ph in df_batch['purchase_history']:
                    # 检查支付状态
                    payment_status = ph.get('payment_status', '')
                    if not any(kw in payment_status for kw in refund_keywords):
                        continue  # 仅处理退款订单
                    
                    # 提取商品类别
                    categories = set()
                    for item in ph.get('items', []):
                        if item['id'] not in valid_ids:
                            continue
                        raw_category = id_to_category[item['id']]
                        mapped_category = CATEGORY_MAPPING.get(raw_category.strip(), raw_category)
                        categories.add(mapped_category)
                    
                    if categories:
                        # 添加退款状态标记
                        refund_tag = f"REFUND_{payment_status.strip()}"
                        transaction = sorted(categories) + [refund_tag]
                        f.write(','.join(transaction) + '\n')
            del df_batch
            gc.collect()

     # ========== 优化编码逻辑 ==========
    def collect_unique_columns_task4(path):
        unique = set()
        with open(path, 'r') as f:
            for line in tqdm(f, desc='收集唯一项（任务4）'):
                unique.update(line.strip().split(','))
        return sorted(unique)

    all_columns = collect_unique_columns_task4(transaction_path)
    
    te = TransactionEncoder()
    te.fit([list(all_columns)])

    # 分批次编码
    from scipy.sparse import vstack
    combined_te_ary = None
    batch_size = 10000

    with open(transaction_path, 'r') as f:
        while True:
            lines = list(islice(f, batch_size))
            if not lines:
                break
            batch_transactions = [line.strip().split(',') for line in lines]
            te_ary_batch = te.transform(batch_transactions, sparse=True)
            if combined_te_ary is None:
                combined_te_ary = te_ary_batch
            else:
                combined_te_ary = vstack([combined_te_ary, te_ary_batch])

    encoded_df = pd.DataFrame.sparse.from_spmatrix(combined_te_ary, columns=te.columns_)
    

    # Step 3. 挖掘频繁项集
    frequent_itemsets = fpgrowth(
        encoded_df,
        min_support=0.005,
        use_colnames=True,
        max_len=4,
        # low_memory=True
    )

    # Step 4. 生成关联规则
    refund_rules = association_rules(
        frequent_itemsets,
        metric='confidence',
        min_threshold=0.4
    )

    # 筛选包含退款标记的规则
    refund_rules = refund_rules[
        refund_rules['consequents'].apply(lambda x: any(s.startswith('REFUND_') for s in x)) |
        refund_rules['antecedents'].apply(lambda x: any(s.startswith('REFUND_') for s in x))
    ]

    # 格式化规则
    def format_refund_rule(row):
        antecedents = [s for s in row['antecedents'] if not s.startswith('REFUND_')]
        consequents = [s for s in row['consequents'] if s.startswith('REFUND_')]
        if not consequents:
            consequents = [s for s in row['consequents']]
            antecedents = [s for s in row['antecedents'] if s.startswith('REFUND_')]
        return pd.Series({
            'antecedents': ', '.join(sorted(antecedents)),
            'consequents': ', '.join(sorted(consequents)),
            **row[['support', 'confidence', 'lift']]
        })

    formatted_rules = refund_rules.apply(format_refund_rule, axis=1)

    # 保存结果
    frequent_itemsets.to_csv('task4_frequent_itemsets.csv', index=False)
    formatted_rules.to_csv('task4_refund_rules.csv', index=False)

    # 可视化top规则
    plt.figure(figsize=(12,6))
    top_rules = formatted_rules.nlargest(10, 'lift')
    sns.barplot(x='lift', y='antecedents', hue='consequents', data=top_rules)
    plt.title('Top 10 Refund Association Rules by Lift')
    plt.xlabel('Lift Score')
    plt.ylabel('Antecedents')
    plt.tight_layout()
    plt.savefig('task4_top_rules.png', dpi=300)
    plt.close()

    return formatted_rules.sort_values('lift', ascending=False)


# 主流程
if __name__ == '__main__':
    parquet_files = glob('30G_data_new/*.parquet')
    
    #任务1: 商品关联规则
    task1_result = process_task1(parquet_files)
    print("任务1完成，关联规则数量:", len(task1_result))
    
    # 任务2: 支付方式分析
    task2_result = process_task2(parquet_files)
    print("任务2完成，支付规则数量:", len(task2_result))

    result = analyze_high_value_payments(
    files=parquet_files,

)
    # 输出结果
    print("高价值商品支付方式分布:")
    for method, count in result:
        print(f"- {method}: {count}次")
    del result  # 主动释放内存
    gc.collect()
    
    # 任务3: 时间序列分析
    result = process_task3(parquet_files)
    del result  # 主动释放内存
    gc.collect()

    
    # 任务4: 退款分析
    task4_result = process_task4(parquet_files)
    print("任务4完成，退款规则数量:", len(task4_result))