import pandas as pd
from io import StringIO
import sys
#from sklearn.preprocessing import Imputer

from sklearn.impute import SimpleImputer

import numpy as np
from sklearn.preprocessing   import LabelEncoder
from sklearn.preprocessing   import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.base            import clone
from itertools               import combinations
from sklearn.metrics         import accuracy_score
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


class SBS():
		def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
				self.scoring    = scoring
				self.estimator  = clone(estimator)
				self.k_features = k_features
				self.test_size  = test_size
				self.random_state = random_state

		def fit(self, X, y):
				
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
																													 random_state=self.random_state)

				dim = X_train.shape[1]
				self.indices_ = tuple(range(dim))
				self.subsets_ = [self.indices_]
				score = self._calc_score(X_train, y_train, 
																 X_test, y_test, self.indices_)
				self.scores_ = [score]

				while dim > self.k_features:
						scores  = []
						subsets = []
						count   = 0

						for p in combinations(self.indices_, r=dim - 1):
								score = self._calc_score(X_train, y_train, 
																				 X_test, y_test, p)
								scores.append(score)
								subsets.append(p)
#								print(dim, p, count, score)
								count += 1

						best = np.argmax(scores)
						self.indices_ = subsets[best] # self.indices_に格納される特徴量を削除していく。残すのではなく、削除するために格納していくことに注意。
						self.subsets_.append(self.indices_)
						dim -= 1

						self.scores_.append(scores[best])
				self.k_score_ = self.scores_[-1]

				return self

		def transform(self, X):
				return X[:, self.indices_]

		def _calc_score(self, X_train, y_train, X_test, y_test, indices):
#				print("------------------------------------------------------------------------")
#				print(indices)
#				print(type(indices))
				self.estimator.fit(X_train[:, indices], y_train)
				y_pred = self.estimator.predict(X_test[:, indices])
				score  = self.scoring(y_test, y_pred) # 性能評価のために、accuracy_scoreを使用。
				return score




df_wine = pd.read_csv('https://archive.ics.uci.edu/'
											'ml/machine-learning-databases/wine/wine.data',
											header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
									 'Alcalinity of ash', 'Magnesium', 'Total phenols',
									 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
									 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
									 'Proline']



# 特徴量とクラスラベルを別々に抽出
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# トレーニングデータセットとテストデータセットに分割
X_train, X_test, y_train, y_test =    train_test_split(X, y, 
										 test_size=0.3, 
										 random_state=0, 
										 stratify=y)
#-------------------------------------------------------------------------------------------
# データを標準化
# 標準化のためのインスタンスを作成
stdsc       = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std  = stdsc.transform(X_test)
#-------------------------------------------------------------------------------------------
# KNNのインスタンス化
knn = KNeighborsClassifier(n_neighbors=5)
#-------------------------------------------------------------------------------------------
# selecting features
print("KNN accuracy before sbs")
knn.fit(X_train_std, y_train)
print('Test accuracy:', knn.score(X_test_std, y_test))

k3 = list((0, 1, 11))
knn.fit(X_train_std[:, k3], y_train)
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))
#-------------------------------------------------------------------------------------------
# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)
#-------------------------------------------------------------------------------------------
# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]
print(sbs.subsets_)
print(k_feat)
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()
plt.close()

print("KNN accuracy after sbs")
knn.fit(X_train_std, y_train)
print('Test accuracy:', knn.score(X_test_std, y_test))
knn.fit(X_train_std[:, k3], y_train)
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))
#-------------------------------------------------------------------------------------------
# データセットの特徴量の名称
feat_labels = df_wine.columns[1:]
# 決定木の個数500個
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)

# 特徴量の重要度を抽出
importances = forest.feature_importances_
# 重要度の降順で特徴量のインデックスを抽出
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
#plt.savefig('images/04_09.png', dpi=300)
plt.show()
plt.close()
#-------------------------------------------------------------------------------------------
# 重要な特徴量だけ出力
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of features that meet this threshold criterion:', 
      X_selected.shape[1])


# Now, let's print the 3 features that met the threshold criterion for feature selection that we set earlier (note that this code snippet does not appear in the actual book but was added to this notebook later for illustrative purposes):



for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

































































