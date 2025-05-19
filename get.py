import os
import pyarrow.parquet as pq
import json

# 替换为你的文件夹路径
folder_path = "10G_data_new/"

# 获取所有Parquet文件
parquet_files = [
    os.path.join(folder_path, f) 
    for f in os.listdir(folder_path) 
    if f.endswith('.parquet')
]

def inspect_purchase_history_structure(file_path, num_samples=3):
    """解析单个文件中的JSON结构"""
    try:
        # 读取文件，仅加载目标列
        table = pq.read_table(file_path, columns=['purchase_history'])
        df = table.to_pandas()
        
        # 提取非空样本
        samples = df['purchase_history'].dropna().head(num_samples).tolist()
        
        print(f"检查文件: {file_path}")
        for i, json_str in enumerate(samples):
            print(f"\n样本 {i+1}:")
            try:
                parsed = json.loads(json_str)
                print(json.dumps(parsed, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print("无效的JSON格式")
        print("-" * 50)
    except Exception as e:
        print(f"处理文件{file_path}时出错: {e}")

# 逐个检查文件的前几个样本
for file in parquet_files:
    inspect_purchase_history_structure(file)

# 可选：仅检查第一个文件的第一个行组
# pf = pq.ParquetFile(parquet_files[0])
# rg = pf.read_row_group(0, columns=['purchase_history'])
# df_sample = rg.to_pandas()
