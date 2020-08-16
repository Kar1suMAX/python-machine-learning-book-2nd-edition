import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


df = pd.read_csv('/Users/ken/Documents/python-machine-learning-book-2nd-edition/code/ch02/iris.data', header=None)

# 1-100行目の目的変数の抽出
y = df.iloc[0:100, 4].values
# Iris-setosaを-1
# Iris-virginiaを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)

# 1-100行目の1,3 列目の抽出
X = df.iloc[0:100, [0, 2]].values

# setosaのプロット(red "o")
plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")
# virginiaのプロット(blue "x")
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="virginia")

plt.xlabel("sepal length [cm]")
plt.ylabel("petal length [cm]")
plt.legend(loc="upper left")
plt.show()


