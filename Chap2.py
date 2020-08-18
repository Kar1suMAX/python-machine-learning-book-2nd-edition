import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#Perceptron classifier.
class Perceptron(object):
	def __init__(self, eta=0.01, n_iter=50, random_state=1):
		self.eta          = eta
		self.n_iter       = n_iter
		self.random_state = random_state

	def fit(self, X, y):
		rgen    = np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) # initialise the array by 0, but it has 0.01 std
		self.errors_ = []

		for _ in range(self.n_iter):
				errors = 0
				for xi, target in zip(X, y):
					update = self.eta * (target - self.predict(xi))
					self.w_[1:] += update * xi
					self.w_[0]  += update
					errors += int(update != 0.0)
				self.errors_.append(errors)
		return self

	def net_input(self, X):
		"""Calculate net input"""
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def predict(self, X):
		"""Return class label after unit step"""
		return np.where(self.net_input(X) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier, resolution=0.02):

		# setup marker generator and color map
		markers = ('s', 'x', 'o', '^', 'v')
		colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
		cmap = ListedColormap(colors[:len(np.unique(y))])
		"""
		まず、colorsとmarkersを複数定義した後、ListedColormapを使って色のリストからカラーマップを作成する。
		"""	
		print(colors[:len(np.unique(y))])
		print(cmap)
		# plot the decision surface
		x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
													 np.arange(x2_min, x2_max, resolution))
		"""
		次に、二つの特徴量の最小値と最大値を求め、それらの特徴ベクトルを使ってグリッド配列をxx1とxx2のペアを作成する。
		これには、numpyのmeshgrid関数を使用する。
		二次元の特徴量でパーセプトロン分類器をトレーニングしたため、グリッド配列を一次元にし、Irisトレーニングサブセットと同じ個数の列を持つ行列を作成する必要がある。今回は2列。
		そのために、predictメソッドを使って対応するグリッドポイントのクラスラベルZを予測する。
		"""
		Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
		Z = Z.reshape(xx1.shape)
		plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
		plt.xlim(xx1.min(), xx1.max())
		plt.ylim(xx2.min(), xx2.max())
		"""
		クラスラベルZを予測した後は、xx1およびxx2と同じ次元を持つグリッドに作り替えた上で、等高線図を描画できる。
		contouf関数は、グリッド配列内の予測されたクラスごとに、決定領域をそれぞれ異なる色にマッピングする。
		"""
		# plot class samples
		for idx, cl in enumerate(np.unique(y)):
			plt.scatter(X[y == cl, 0], # y==clがTrueとなる行番号でXの1列目をx
									X[y == cl, 1], # y==clがTrueとなる行番号でXの2列目をy
									alpha =0.8, 
									c     =colors[idx],
									marker=markers[idx], 
									label =cl, 
									edgecolor='black')
			"""
			plt.scatter(x=X[y == cl, 0], 
									y=X[y == cl, 1],
									alpha =0.8, 
									c     =colors[idx],
									marker=markers[idx], 
									label =cl, 
									edgecolor='black')
			"""







#-------------------------------------------------------------------------------------------
df = pd.read_csv('/Users/ken/Documents/python-machine-learning-book-2nd-edition/code/ch02/iris.data', header=None)

# 1-100行目の目的変数の抽出
y = df.iloc[0:100, 4].values
# Iris-setosaを-1
# Iris-virginiaを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)

# 1-100行目の1,3 列目の抽出
X = df.iloc[0:100, [0, 2]].values
#-------------------------------------------------------------------------------------------
# パーセプトロンのオブジェクトの生成(インスタンス化)
ppn = Perceptron(eta=0.1, n_iter=10)
# トレーニングデータへのモデルの適合
ppn.fit(X, y)
# エポックと後分類誤差の関係の折れ線グラフ
plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_, marker="o")
plt.xlabel("epochs")
plt.ylabel("number of updates")
plt.close()
#-------------------------------------------------------------------------------------------
# 決定領域のプロット
plot_decision_regions(X, y, classifier = ppn)
plt.xlabel("sepal length [cm]")
plt.ylabel("petal length [cm]")
plt.legend(loc="upper left")
plt.show()

"""
# setosaのプロット(red "o")
plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")
# virginiaのプロット(blue "x")
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="virginia")

plt.xlabel("sepal length [cm]")
plt.ylabel("petal length [cm]")
plt.legend(loc="upper left")
plt.savefig("setosa_and_virginia.png")
"""




















