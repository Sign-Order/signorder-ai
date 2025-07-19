import os
from dotenv import load_dotenv
from pymongo import MongoClient
import numpy as np
import json
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.layers import Masking, LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import Add, LayerNormalization

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.regularizers import l2


from sklearn.metrics import f1_score, classification_report

import random

from keras.models import Model
from keras.layers import Input, Masking, LSTM, Dense, Dropout, BatchNormalization, Layer, Conv1D
import tensorflow as tf


from tensorflow.keras.layers import Input, Masking, Bidirectional, GRU, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D


"""
Transformer 계열로 수정했을 때 라이브러리들
"""
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Add
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
client = MongoClient(mongo_db_url)
db = client["dev"]

# sign_language_collection = db["sign_language"]
# name_url_dict = {
#     doc["name"]: doc["url"]
#     for doc in sign_language_collection.find({}, {"_id": 0, "name": 1, "url": 1})
# }


# DATA_PATH = "custom_dataset_2"
# data_list = []
# gesture = {}
# label_index = 0

# for action in name_url_dict.keys():
#     file_path = os.path.join(DATA_PATH, f"{action}.npy")
#     if not os.path.exists(file_path):
#         print(f"⚠️ 데이터 누락: {action} - {file_path} 없음")
#         continue

#     seq_data = np.load(file_path)
#     if seq_data.ndim != 3 or seq_data.shape[0] == 0:
#         print(f"⚠️ 유효하지 않은 시퀀스: {action} - shape: {seq_data.shape}")
#         continue

#     seq_data[:, :, -1] = label_index
#     gesture[label_index] = action  
#     label_index += 1

#     data_list.append(seq_data)

DATA_PATH = "angles"
data_list = []
gesture = {}
label_index = 0

npy_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.npy')]
# if not npy_files:
#     raise ValueError(f"❌ {DATA_PATH} 디렉터리에 .npy 파일이 없습니다.")

# for filename in sorted(npy_files): 
#     action = os.path.splitext(filename)[0]
#     file_path = os.path.join(DATA_PATH, filename)

#     seq_data = np.load(file_path)
#     if seq_data.ndim != 3 or seq_data.shape[0] == 0:
#         print(f"⚠️ 유효하지 않은 시퀀스: {action} - shape: {seq_data.shape}")
#         continue

#     seq_data[:, :, -1] = label_index
#     gesture[label_index] = action  
#     label_index += 1

#     data_list.append(seq_data)
gesture_list = sorted([os.path.splitext(f)[0] for f in npy_files])  # 알파벳순 고정

gesture = {i: name for i, name in enumerate(gesture_list)}  # 라벨 고정
reverse_gesture = {v: k for k, v in gesture.items()}

for action in gesture_list:
    file_path = os.path.join(DATA_PATH, f"{action}.npy")
    seq_data = np.load(file_path)
    seq_data[:, :, -1] = reverse_gesture[action]  # 항상 일치
    data_list.append(seq_data)


if not data_list:
    raise ValueError("❌ 불러온 데이터가 없습니다. 모델 학습 불가")

data = np.concatenate(data_list, axis=0)
print(f"✅ 총 데이터 샘플 수: {data.shape[0]}, 시퀀스 길이: {data.shape[1]}, 피처 수: {data.shape[2]}")

x_data = data[:, :, :-1]
labels = data[:, 0, -1].astype(int)

if not data_list:
    raise ValueError("불러온 데이터가 없습니다. 모델 학습 불가")

data = np.concatenate(data_list, axis=0)
x_data = data[:, :, :-1]
labels = data[:, 0, -1].astype(int)
y_data = to_categorical(labels, num_classes=len(gesture))

x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_data, test_size=0.1, random_state=2021
)

def augment_sequence(seq, jitter_prob=0.3, noise_std=0.01, angle_perturb_range=2.0,
                     stretch_prob=0.3, brightness_std=0.05):
    augmented = seq.copy()

    # 1. ⏱️ Temporal jitter: 순서를 약간 섞음
    if random.random() < jitter_prob:
        idx = np.arange(len(augmented))
        jitter = np.clip(np.random.normal(0, 1, size=len(idx)), -2, 2).astype(int)
        jittered_idx = np.clip(idx + jitter, 0, len(idx) - 1)
        augmented = augmented[jittered_idx]

    # 2. 🌫️ Joint 좌표에 noise 추가
    joint_dim = 21 * 3  # 63
    augmented[:, :joint_dim] += np.random.normal(0, noise_std, size=(augmented.shape[0], joint_dim))

    # 3. 🔄 각도 값에 ±1~2도 perturbation
    angle_dim = 15
    angle_start = joint_dim
    angle_end = joint_dim + angle_dim
    perturb = np.random.uniform(-angle_perturb_range, angle_perturb_range, size=(augmented.shape[0], angle_dim))
    augmented[:, angle_start:angle_end] += perturb

    # 4. 📏 Stretching: 시퀀스 길이를 늘리거나 줄임 (선형 보간)
    if random.random() < stretch_prob:
        factor = random.uniform(0.8, 1.2)  # 20% 늘리거나 줄이기
        new_len = int(len(augmented) * factor)
        x_old = np.linspace(0, 1, len(augmented))
        x_new = np.linspace(0, 1, new_len)

        # 각 열에 대해 보간 수행
        augmented = np.array([np.interp(x_new, x_old, augmented[:, i]) for i in range(augmented.shape[1])]).T

        # 다시 원래 길이로 pad or crop
        if new_len > len(seq):
            augmented = augmented[:len(seq)]
        else:
            pad = np.zeros((len(seq) - new_len, augmented.shape[1]), dtype=np.float32)
            augmented = np.vstack([augmented, pad])

    # 5. 💡 Brightness-like noise: 전체 값에 일정 offset
    brightness_offset = np.random.normal(0, brightness_std)
    augmented[:, :angle_end] += brightness_offset

    return augmented


# 데이터 증강 적용
augmented_train = []
augmented_labels = []

for i in range(len(x_train)):
    augmented_train.append(x_train[i])
    augmented_labels.append(y_train[i])

    # 증강 샘플 추가 (예: 1개씩 증강 → 2배 데이터)
    augmented_train.append(augment_sequence(x_train[i]))
    augmented_labels.append(y_train[i])

x_train = np.array(augmented_train)
y_train = np.array(augmented_labels)

print(f"📦 증강된 학습 데이터: {x_train.shape}, 라벨: {y_train.shape}")


class Attention(Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="attention_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)


# inputs = Input(shape=x_train.shape[1:])
# x = Masking(mask_value=0.0)(inputs)
# x = LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001))(x)
# x = LayerNormalization()(x) 
# x = LSTM(64, return_sequences=True)(x)
# x = LayerNormalization()(x)
# x = Attention()(x)
# x = Dense(64, activation='relu')(x)
# x = Dropout(0.5)(x)
# outputs = Dense(len(gesture), activation='softmax')(x)

# model = Model(inputs, outputs)

# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['acc']
# )

class PositionalEncoding(Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_encoding = self.get_positional_encoding(max_len, d_model)

    def get_positional_encoding(self, max_len, d_model):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def transformer_encoder(inputs, num_heads=4, ff_dim=128, dropout_rate=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1]//num_heads)(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = Add()([inputs, attn_output])
    out1 = LayerNormalization()(out1)

    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(inputs.shape[-1])(ffn)
    out2 = Add()([out1, ffn])
    out2 = LayerNormalization()(out2)
    return out2


def build_sign_transformer(input_shape, num_classes, depth=4, num_heads=4, ff_dim=128):
    inputs = Input(shape=input_shape)  # (seq_len, num_features)
    x = Masking()(inputs)
    x = Dense(128)(x)  # 임베딩 차원 고정

    x = PositionalEncoding(max_len=input_shape[0], d_model=128)(x)

    for _ in range(depth):
        x = transformer_encoder(x, num_heads=num_heads, ff_dim=ff_dim)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


model = build_sign_transformer(
    input_shape=x_train.shape[1:],  # (60, num_features)
    num_classes=len(gesture),
    depth=4,
    num_heads=4,
    ff_dim=128
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # if labels are int
    metrics=['accuracy']
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(weights))

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint('60_v27_masked_angles.keras', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ],
    class_weight=class_weights
)

y_pred = model.predict(x_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

f1 = f1_score(y_true, y_pred_classes, average='weighted')
print(f"F1 Score (Weighted): {f1:.4f}")

print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

fig, loss_ax = plt.subplots(figsize=(16, 10))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['acc'], 'b', label='train acc')
acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper right')

plt.title('Model Training History')
plt.grid(True)
plt.show()

model.save('60_v27_masked_angles.keras')
print("✅ 모델 저장 완료: 60_v27_masked_angles.keras")

with open('60_v27_pad_gesture_dict.json', 'w', encoding='utf-8') as f:
    json.dump(gesture, f, ensure_ascii=False, indent=2)
print("✅ 제스처 라벨 딕셔너리 저장 완료: 60_v27_pad_gesture_dict.json")