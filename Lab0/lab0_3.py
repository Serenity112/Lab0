import numpy as np
import pandas as pd
from func import func

# Создаём массивы с равномерным и нормыльным распределениями
uniform_array = np.random.rand(17, 19)
normal_array = np.random.randn(17, 19)

print("Равномерно распределенные")
print(uniform_array)
print("Нормально распределенные случайные числа")
print(normal_array)

print("Атрибуты массива uniform_array:")
func(uniform_array)
print("Атрибуты массива normal_array:")
func(normal_array)

# Создаём таблицу с классами 0/1
table = pd.read_csv("csv0.csv")
table['class'] = table['class'].map({'Positive': 1,'Negative': 0})
table.to_csv("csv1.csv", index=False)
