from keras.datasets import mnist
import matplotlib.pyplot as plt 
import numpy as np

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

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

# === 2. Препроцессинг ===
# Нормализация входных данных (0-255 -> 0-1)
data_train_norm = data_train.astype('float32')/255.0
data_test_norm = data_test.astype('float32')/255.0

print(f'Before norm: [{data_train.min()} - {data_train.max()}]\n'
      f'After norm: [{data_train_norm.min():.2f} - {data_train_norm.max():.2f}]\n')

# reshape формы для слоя Dense
data_train_flat = data_train_norm.reshape(-1,28*28)
data_test_flat = data_test_norm.reshape(-1,28*28)

print(f'Before reshape:\n'
      f'data_train_norm: {data_test_norm.shape}\n'
      f'single digit: {data_test_norm[0].shape}\n')

print(f'After reshape:\n'
      f'data_train_flat: {data_train_flat.shape}\n'
      f'single digit: {data_train_flat[0].shape}\n')

print(f'Check: {28} * {28} = {28*28}')

# One-hot encoding
target_train_cat = to_categorical(target_train, 10)
target_test_cat = to_categorical(target_test, 10)

print(f'\nBefore one-hot: {target_train.shape}\n'
      f'After one-hot: {target_train_cat.shape}\n'
      f'Example target[0]: {target_train[0]} -> {target_train_cat[0]}\n')

# Завершение препроцессинга
print(f'Finished preprocessing\n'
      f'data_train_flat: {data_train_flat.shape}\n'
      f'data_test_flat: {data_test_flat.shape}\n'
      f'target_train_cat: {target_train_cat.shape}\n'
      f'target_test_cat: {target_test_cat.shape}')

# === 3. Создание модели ===
model = Sequential([
    Dense(512,activation = 'relu', input_shape = (784,)),
    Dropout(0.2), # отключение 20% нейронов ждя избежания переобучения (подать на выход нейрона значение 0)
    Dense(256, activation = 'relu'),
    Dropout(0.2),
    Dense(10, activation = 'softmax')
])

model.summary()

# Компиляция
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

# Обучение
# Ранний стоп, для предотвращения ухудшения обучения
# срабатывает в конце каждой эпохи и каждого обновления весов (batch_size)
early_stop = EarlyStopping(
    monitor = 'val_loss', # что отслеживаем
    patience = 5, # сколько шагов ухудшения можно допустить перед стопом
    restore_best_weights = True, # во время стопа откатиться к лучшим значениям весов у нейронов
    verbose = 1 # уведомление о срабатывании (0 - нет, 1 - есть)
)

learn = model.fit(
    data_train_flat, target_train_cat,
    epochs = 1,
    batch_size = 128,
    validation_split = 0.1,
    callbacks = [early_stop], # подключение раннего стопа
    verbose = 1
)

# Тестирование
test_loss, test_acc = model.evaluate(data_test_flat, target_test_cat, verbose = 0)

print(f'Test loss: {test_loss:.4f}\n'
      f'Test accuracy: {test_acc:.4f} ({test_acc * 100:.1f}%)')

# Графики
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,5))

# Accuracy
ax1.plot(learn.history['accuracy'], label = 'Train', linewidth = 2)
ax1.plot(learn.history['val_accuracy'], label = 'Validation', linewidth = 2)
ax1.axhline(y = test_acc, color = 'red', linestyle = '--', label = f'Test ({test_acc:.4f})',linewidth = 2)
ax1.set_title('Model accuracy', fontsize = 14, fontweight = 'bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha = 0.3)

# Loss
ax2.plot(learn.history['loss'],label = 'Train',linewidth = 2)
ax2.plot(learn.history['val_loss'], label = 'Validation', linewidth = 2)
ax2.set_title('Model loss', fontsize = 14, fontweight = 'bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha = 0.3)

plt.tight_layout()
plt.savefig('2_model_fit.png', dpi = 150, bbox_inches = 'tight')

# === 4. Анализ предсказаний ===
# Виды ошибок
      # 1. True Positive (TP) - истинно положительный
      # Истина: 1, предсказано: 1 (предсказано: 5, реально тоже 5)

      # 2. True Negative (TN) - истинно отричательный
      # Истина: 0, предсказано: 0 (предсказано 5, но реально не 5)

      # 3. False Positive (FP) - ложно положительный
      # Истина: 0, предсказано: 1 (реально 5, но предсказано не 5)

      # 4. False Negative (FN) - ложно отрицательный
      # Истина: 1, предсказано: 0 (реально не 5 и предсказано не 5)

# 1. Classification Report - детальная статистика по всем классам
# Критерии отчета
      # 1. Precision (точность) - TP / (TP + FP)
      # из всех предсказанных чисел '5', сколько реально было '5'?

      # 2. Recall (полнота) - TP / (TP + FN)
      # из всех настоящих чисел '5', сколько нашла модель

      # 3. F1-Score (гармоническое среднее) - 2 * (Precision * Recall) / (Precision + Recall)
      # Среднее значение между Precision и Recall

      # 4. Support - сколько примеров класса было в тестах

# 2. Confusion Matrix - таблица ошибок
# Матрица предсказанных классов
# Идеальный вариант - ненулевая диагональ, остальные поля = 0

# Prediction (предсказания)
predictions = model.predict(data_test_flat)
predicted_classes = np.argmax(predictions, axis = 1)
true_target_classes = target_test

# Classification Report
print(f'\nClassification_report:')
print(classification_report(
    true_target_classes, # истинные выходные классы
    predicted_classes, # предсказанные классы
    target_names = [str(i) for i in range(10)] # наименования чисел статистики (0-9)
))

# Confusion Matrix
conf_matrix = confusion_matrix(true_target_classes, predicted_classes)
print(f'\nConfusion matrix:\n {conf_matrix}')

# Подсчёт ошибочных пар
print(f'\nЧастые ошибки:')
errors = []

for i in range(10):
    for j in range(10):
        if i != j and conf_matrix[i,j] > 0:
            errors.append((i,j,conf_matrix[i,j]))

errors.sort(key = lambda x: x[2], reverse = True)

for true, pred, errors in errors[:10]:
    print(f'{true} -> {pred}: {errors} раз')
