import numpy as np
import pandas as pd
from func import func

# ������ ������� � ����������� � ���������� ���������������
uniform_array = np.random.rand(17, 19)
normal_array = np.random.randn(17, 19)

print("���������� ��������������")
print(uniform_array)
print("��������� �������������� ��������� �����")
print(normal_array)

print("�������� ������� uniform_array:")
func(uniform_array)
print("�������� ������� normal_array:")
func(normal_array)

# ������ ������� � �������� 0/1
table = pd.read_csv("csv0.csv")
table['class'] = table['class'].map({'Positive': 1,'Negative': 0})
table.to_csv("csv1.csv", index=False)
