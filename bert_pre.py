import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
import pandas as pd

# 确保 TensorFlow 使用 GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and memory growth is enabled.")
    except RuntimeError as e:
        print(e)
else:
    print("GPU is not available, using CPU.")

# 加载保存的模型
model = tf.keras.models.load_model('saved_model/bert_movie_genre_classifier', custom_objects={'TFBertModel': TFBertModel})

# 加载 Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_genre(description, top_n=3):
    # 将输入的描述性语句转换为序列
    encoding = tokenizer(description, padding='max_length', truncation=True, max_length=128, return_tensors='tf')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # 模型预测
    predictions = model({'input_ids': input_ids, 'attention_mask': attention_mask}, training=False)[0]

    # 获取 top N 的类别及其概率
    movies_df = pd.read_csv('filtered_movies.csv')
    genre_columns = movies_df.drop(columns=['title', 'overview']).columns  # 确保列索引从 genre 开始

    # 调试信息
    print(f"Predictions shape: {predictions.shape}")
    print(f"Number of genre columns: {len(genre_columns)}")

    top_indices = tf.argsort(predictions, direction='DESCENDING')[:top_n]
    top_genres = [(genre_columns[i], predictions[i].numpy()) for i in top_indices.numpy() if i < len(genre_columns)]

    return top_genres

# 示例描述性语句
description = "A group of friends embark on a journey to find a hidden treasure."

# 预测电影类型及其概率
top_genres = predict_genre(description)
print("预测的电影类型及其概率：")
for genre, prob in top_genres:
    print(f"{genre}: {prob:.4f}")
