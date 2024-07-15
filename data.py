import pandas as pd

# 加载筛选后的数据集
input_file_path = 'filtered_movies.csv'  # 替换为实际的文件路径
movies_df = pd.read_csv(input_file_path)

# 显示基本信息
print("原始数据集基本信息：")
print(movies_df.info())

# 去除缺失值的列
movies_df = movies_df.dropna()

# 独热编码 genres 列
genres_df = movies_df['genres'].str.get_dummies(sep=', ')
movies_df = pd.concat([movies_df, genres_df], axis=1)

# 移除原始 genres 列
movies_df = movies_df.drop('genres', axis=1)

# 保存预处理后的数据集
output_file_path = 'preprocessed_movies.csv'  # 新的文件路径
movies_df.to_csv(output_file_path, index=False)

print(f"预处理后的数据已保存到 {output_file_path}")

# 显示预处理后的数据集基本信息
print("\n预处理后的数据集基本信息：")
print(movies_df.info())

# 显示前几行数据
print("\n预处理后的数据集前五行：")
print(movies_df.head())
