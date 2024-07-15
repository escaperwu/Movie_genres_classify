import pandas as pd

# 加载原始数据集
file_path = 'movies.csv'  # 请将此路径替换为实际的文件路径
movies_df = pd.read_csv(file_path)

# 提取需要的列
selected_columns = ['title', 'overview', 'genres']
filtered_df = movies_df[selected_columns]

# 移除缺失值
filtered_df = filtered_df.dropna(subset=['overview', 'genres'])

# 只选择前5000行数据
filtered_df = filtered_df.head(10000)

# 保存到新的CSV文件
output_file_path = 'filtered_movies.csv'  # 新的文件路径
filtered_df.to_csv(output_file_path, index=False)

print(f"已将筛选后的数据保存到 {output_file_path}")
