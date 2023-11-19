# Practice-12
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/XE/Downloads/datasetage_kz.csv')
# Преобразование категориальных переменных в бинарные
df = pd.get_dummies(df, columns=['Gender', 'Married'], prefix=['Gender', 'Married'])
# Определение признаков и целевой переменной
X = df.drop(columns=['Age'])
y = df['Income']

# Разделение набора данных на обучающий и тестовый
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Линейная регрессия
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# График линейной функции
plt.figure(figsize=(8, 6))
plt.scatter(X_test['Income'], y_test, alpha=0.5, label='Фактический доход')
plt.scatter(X_test['Income'], y_pred_linear, alpha=0.5, label='Предсказанный доход (Linear Regression)')

# График линейной функции
x_values = np.linspace(X_test['Income'].min(), X_test['Income'].max(), 100)
y_values_linear = linear_model.coef_[0] * x_values + linear_model.intercept_
plt.plot(x_values, y_values_linear, color='red', linestyle='--', label='Линейная функция')

plt.xlabel('Фактический доход')
plt.ylabel('Предсказанный доход')
plt.title('Фактический vs Предсказанный доход (только линейная регрессия)')
plt.legend()
plt.show()




import sys
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

cities_data = pd.read_csv("data.csv", header=0, sep=",")
x = cities_data["Population"]
y = cities_data["Median_Income"]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.xlabel("Population")
plt.ylabel("Median_Income")
plt.title("Population vs Median Income")
plt.show()
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()




import numpy as np

def calculate_slope(x_values, y_values):
    coefficients = np.polyfit(x_values, y_values, 1)
    return coefficients[0]

#  данные о времени учебы (в часах) и оценках студента
study_hours = [2, 4, 6, 8, 10]
grades = [60, 70, 80, 90, 100]

result_slope = calculate_slope(study_hours, grades)
print(f"Учебный наклон: {result_slope}")


