from keras.datasets import mnist
import matplotlib.pyplot as plt 
import numpy as np

# === 1. Подгрузка датасета ===
(data_train, target_train), (data_test, target_test) = mnist.load_data()

# формат данных
print(f'Data tain shape: {data_train}\n'
      f'Target train shape: {target_train.shape}\n'
      f'Dtype: {data_train.dtype}\n'
      f'Value range: {data_train.min()} - {data_train.max()}\n')

print(f'Data test shape: {data_test.shape}\n'
      f'Target test shape: {target_test.shape}\n'
      f'Dtype: {target_train.dtype}\n'
      f'Value range: {target_train.min()} - {target_train.max()}')

# Подсчёт уникальных классов (целевых данных)
unique, count = np.unique(target_train, return_counts=True)
for digit, count in zip(unique,count):
    print(f'Digit: {digit}, count: {count} - {count/len(target_train)*100:.1f}%')
    