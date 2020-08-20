from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import Perceptron
from sklearn.metrics         import accuracy_score
from matplotlib.colors       import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.linear_model    import SGDClassifier
from sklearn.tree            import DecisionTreeClassifier
from pydotplus               import graph_from_dot_data
from sklearn.tree            import export_graphviz
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

		# setup marker generator and color map
		markers = ('s', 'x', 'o', '^', 'v')
		colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
		cmap = ListedColormap(colors[:len(np.unique(y))])

		# plot the decision surface
		x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
													 np.arange(x2_min, x2_max, resolution))
		Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
		Z = Z.reshape(xx1.shape)
		plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
		plt.xlim(xx1.min(), xx1.max())
		plt.ylim(xx2.min(), xx2.max())

		# plot the scatter of the actual datas
		for idx, cl in enumerate(np.unique(y)):
				plt.scatter(x=X[y == cl, 0], 
										y=X[y == cl, 1],
										alpha=0.8, 
										c=colors[idx],
										marker=markers[idx], 
										label=cl, 
										edgecolor='black')

		# highlight test samples by circle
		if test_idx:
				# plot all samples
				X_test, y_test = X[test_idx, :], y[test_idx]

				plt.scatter(X_test[:, 0],
										X_test[:, 1],
										c='',
										edgecolor='black',
										alpha=1.0,
										linewidth=1,
										marker='o',
										s=100, 
										label='test set')



def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))



# Logistic Regression Classifier using gradient descent.
class LogisticRegressionGD(object):
		def __init__(self, eta=0.05, n_iter=100, random_state=1):
				self.eta          = eta
				self.n_iter       = n_iter
				self.random_state = random_state

		def fit(self, X, y):
				rgen       = np.random.RandomState(self.random_state)
				self.w_    = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
				self.cost_ = []

				for i in range(self.n_iter):
						net_input = self.net_input(X)
						output    = self.activation(net_input)
						errors    = (y - output)
						self.w_[1:] += self.eta * X.T.dot(errors)
						self.w_[0]  += self.eta * errors.sum()
						
						# note that we compute the logistic `cost` now
						# instead of the sum of squared errors cost
						cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
						self.cost_.append(cost)
				return self
		
		def net_input(self, X):
				"""Calculate net input"""
				return np.dot(X, self.w_[1:]) + self.w_[0]

		def activation(self, z):
				"""Compute logistic sigmoid activation"""
				return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

		def predict(self, X):
				"""Return class label after unit step"""
				return np.where(self.net_input(X) >= 0.0, 1, 0)
				# equivalent to:
				# return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)




iris = datasets.load_iris()
X    = iris.data[:, [2,3]]
y    = iris.target
print(np.unique(y))
#-------------------------------------------------------------------------------------------
# split training data and test data
# 全体の30%をテストデータにする。
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.3, random_state=1, stratify=y)

print(np.bincount(y))
print(np.bincount(y_train))
print(np.bincount(y_test))
#-------------------------------------------------------------------------------------------
# 特徴量のスケーリング
sc = StandardScaler()
sc.fit(X_train) # 平均値と標準偏差の計算
X_train_std = sc.transform(X_train)
X_test_std  = sc.transform(X_test)
#-------------------------------------------------------------------------------------------
# パーセプトロンで学習
ppn = Perceptron(max_iter= 40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
#-------------------------------------------------------------------------------------------
# 予測
y_pred = ppn.predict(X_test_std)
print( (y_test!=y_pred).sum() )
#-------------------------------------------------------------------------------------------
# 性能指標の計算
print(accuracy_score(y_test, y_pred))
print(ppn.score(X_test_std, y_test))
#-------------------------------------------------------------------------------------------
# 決定領域のプロット
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined     = np.hstack((y_train, y_test))
plot_decision_regions(X_combined_std, y_combined, classifier=ppn, test_idx = range(105,150))
plt.tight_layout()
plt.show()























































