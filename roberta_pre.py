import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaModel
import numpy as np
import pandas as pd

# 加载保存的模型
model = tf.keras.models.load_model('saved_model/roberta_movie_genre_classifier', custom_objects={'TFRobertaModel': TFRobertaModel})

# 加载 Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# 预测函数
def predict_genre(description, top_n=3):
    # 将输入的描述性语句转换为序列
    encoding = tokenizer(description, padding='max_length', truncation=True, max_length=128, return_tensors='tf')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # 模型预测
    predictions = model.predict({'input_ids': input_ids, 'attention_mask': attention_mask})[0]

    # 获取 top N 的类别及其概率
    genre_columns = pd.read_csv('preprocessed_movies.csv').columns[2:]  # 确保列索引从 genre 开始
    top_indices = predictions.argsort()[-top_n:][::-1]
    top_genres = [(genre_columns[i], predictions[i]) for i in top_indices]

    return top_genres

if __name__ == "__main__":
    # 提示用户输入描述性语句
    description = input("请输入电影的描述性语句：")

    # 预测电影类型及其概率
    top_genres = predict_genre(description)
    print("预测的电影类型及其概率：")
    for genre, prob in top_genres:
        print(f"{genre}: {prob:.4f}")
