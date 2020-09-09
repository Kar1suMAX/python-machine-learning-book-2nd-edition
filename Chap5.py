import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition         import PCA
from matplotlib.colors             import ListedColormap
from sklearn.linear_model          import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial.distance        import pdist, squareform
from scipy        import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA




df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
											'machine-learning-databases/wine/wine.data',
											header=None)


df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
									 'Alcalinity of ash', 'Magnesium', 'Total phenols',
									 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
									 'Color intensity', 'Hue',
									 'OD280/OD315 of diluted wines', 'Proline']


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

		# setup marker generator and color map
		markers = ('s', 'x', 'o', '^', 'v')
		colors  = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
		cmap    = ListedColormap(colors[:len(np.unique(y))])

		# plot the decision surface
		x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		xx1, xx2       = np.meshgrid(np.arange(x1_min, x1_max, resolution),
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



"""
def rbf_kernel_pca(X, gamma, n_components):
		# Calculate pairwise squared Euclidean distances
		# in the MxN dimensional dataset.
		sq_dists = pdist(X, 'sqeuclidean')

		# Convert pairwise distances into a square matrix.
		mat_sq_dists = squareform(sq_dists)

		# Compute the symmetric kernel matrix.
		K = exp(-gamma * mat_sq_dists)

		# Center the kernel matrix.
		N = K.shape[0]
		one_n = np.ones((N, N)) / N
		K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

		# Obtaining eigenpairs from the centered kernel matrix
		# scipy.linalg.eigh returns them in ascending order
		eigvals, eigvecs = eigh(K)
		eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

		# Collect the top k eigenvectors (projected samples)
		X_pc = np.column_stack((eigvecs[:, i]
														for i in range(n_components)))

		return X_pc
"""





def rbf_kernel_pca(X, gamma, n_components):
		"""
		RBF kernel PCA implementation.

		Parameters
		------------
		X: {NumPy ndarray}, shape = [n_samples, n_features]
				
		gamma: float
			Tuning parameter of the RBF kernel
				
		n_components: int
			Number of principal components to return

		Returns
		------------
		 alphas: {NumPy ndarray}, shape = [n_samples, k_features]
			 Projected dataset 
		 
		 lambdas: list
			 Eigenvalues

		"""
		# Calculate pairwise squared Euclidean distances
		# in the MxN dimensional dataset.
		sq_dists = pdist(X, 'sqeuclidean')

		# Convert pairwise distances into a square matrix.
		mat_sq_dists = squareform(sq_dists)

		# Compute the symmetric kernel matrix.
		K = exp(-gamma * mat_sq_dists)

		# Center the kernel matrix.
		N = K.shape[0]
		one_n = np.ones((N, N)) / N
		K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

		# Obtaining eigenpairs from the centered kernel matrix
		# scipy.linalg.eigh returns them in ascending order
		eigvals, eigvecs = eigh(K)
		eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

		# Collect the top k eigenvectors (projected samples)
		alphas = np.column_stack((eigvecs[:, i]
															for i in range(n_components)))

		# Collect the corresponding eigenvalues
		lambdas = [eigvals[i] for i in range(n_components)]

		return alphas, lambdas


#-------------------------------------------------------------------------------------------
# 使用するデータの読み込み
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=0.3, 
                     stratify=y,
                     random_state=0)
#-------------------------------------------------------------------------------------------
# データを標準化
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std  = sc.transform(X_test)

#-------------------------------------------------------------------------------------------
# 変動行列 SW
np.set_printoptions(precision=4)

mean_vecs = []
for label in range(1, 4):
		mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
		print('MV %s: %s\n' % (label, mean_vecs[label - 1]))


d = 13 # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
		class_scatter = np.zeros((d, d))  # scatter matrix for each class
		for row in X_train_std[y_train == label]:
				row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # make column vectors
				class_scatter += (row - mv).dot((row - mv).T)
		S_W += class_scatter                          # sum class scatter matrices

print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))


# Better: covariance matrix since classes are not equally distributed:



print('Class label distribution: %s' 
			% np.bincount(y_train)[1:])



#-------------------------------------------------------------------------------------------
# スケーリングされたクラス内変動行列 SW（共分散行列）
d = 13  # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
		class_scatter = np.cov(X_train_std[y_train == label].T)
		S_W += class_scatter
print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0],
																										 S_W.shape[1]))
#-------------------------------------------------------------------------------------------
# クラス間変動行列 SB
mean_overall = np.mean(X_train_std, axis=0)
d = 13  # number of features
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
		n = X_train[y_train == i + 1, :].shape[0]
		mean_vec     = mean_vec.reshape(d, 1)  # make column vector
		mean_overall = mean_overall.reshape(d, 1)  # make column vector
		S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))
#-------------------------------------------------------------------------------------------
# 固有値を求める
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
print("--------------------------------------")
# print(eigen_vals, eigen_vecs)
#-------------------------------------------------------------------------------------------
# ## scikit-learnによる線形判別分析
lda         = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)


#plot_decision_regions(X_train_lda, y_train, classifier=lr)
#plt.xlabel('LD 1')
#plt.ylabel('LD 2')
#plt.legend(loc='lower left')
#plt.tight_layout()
#plt.show()







































