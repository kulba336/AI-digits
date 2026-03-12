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

# График - примеры цифр
fig, ax = plt.subplots(4,10,figsize = (15,6))

for i in range(40):
    axes = ax[i//10, i%10] # разбиаение диапозона range(40) на двумерный массив [row, col]
    axes.imshow(data_train[i], cmap = 'gray') # отрисовка картинки с цифрой
    axes.set_title(f'{target_train[i]}', fontsize = 12)
    axes.axis('off') # отключение отображения оси координат для графика

plt.suptitle('Digits example', fontsize = 16, fontweight = 'bold')
plt.tight_layout()
plt.savefig('1_digits_example.png', dpi = 150, bbox_inches = 'tight')

# Просмотр одной цифры
sample_digit_index = 0
print(f'\nIndex of digit: {sample_digit_index}\n'
      f'Target digit: {target_train[sample_digit_index]}\n'
      f'Digit shape: {data_train[sample_digit_index].shape}\n')
print(data_train[sample_digit_index])