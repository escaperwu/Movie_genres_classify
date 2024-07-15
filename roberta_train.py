import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, TFRobertaModel, RobertaConfig
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime
import os
import matplotlib.pyplot as plt
import json

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

# 加载预处理后的数据集
file_path = 'preprocessed_movies.csv'  # 替换为实际的文件路径
movies_df = pd.read_csv(file_path)

# 分离特征和标签
X = movies_df['overview'].values
y = movies_df.drop(columns=['title', 'overview']).values

# 分割数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenization and padding
def tokenize_and_pad(texts, max_length=128):  # 减少序列长度
    return tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )

X_train_encodings = tokenize_and_pad(X_train)
X_val_encodings = tokenize_and_pad(X_val)

# 使用 tf.data.Dataset 创建数据管道
batch_size = 8  # 减少批量大小

train_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': X_train_encodings['input_ids'], 'attention_mask': X_train_encodings['attention_mask']},
    y_train
)).shuffle(buffer_size=len(X_train)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': X_val_encodings['input_ids'], 'attention_mask': X_val_encodings['attention_mask']},
    y_val
)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 配置RoBERTa模型
num_classes = y_train.shape[1]  # Number of genres
config = RobertaConfig.from_pretrained('roberta-base', num_labels=num_classes)
roberta_model = TFRobertaModel.from_pretrained('roberta-base', config=config)

input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')  # 序列长度调整为128
attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')  # 序列长度调整为128

embedding = roberta_model(input_ids, attention_mask=attention_mask)[0]
pooled_output = tf.keras.layers.GlobalAveragePooling1D()(embedding)
dropout = tf.keras.layers.Dropout(0.1)(pooled_output)
output = tf.keras.layers.Dense(num_classes, activation='sigmoid')(dropout)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# 设置 TensorBoard 日志记录
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

history = model.fit(
    train_dataset,
    epochs=5,
    validation_data=val_dataset,
    callbacks=[early_stopping, reduce_lr, tensorboard_callback]
)

# 保存模型
model.save('saved_model/roberta_movie_genre_classifier')
print("模型已保存到 saved_model/roberta_movie_genre_classifier")

# 绘制训练过程中的损失和准确率
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history_roberta.png')
    plt.show()

plot_history(history)

# 计算评估指标
loss, accuracy = model.evaluate(val_dataset)

# 保存评估指标
metrics = {
    'loss': loss,
    'accuracy': accuracy
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print("评估指标已保存到 metrics.json")
