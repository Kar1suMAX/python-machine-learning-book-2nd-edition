import sys
import gzip
import shutil
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from neuralnet import NeuralNetMLP
import joblib


path = "/Users/ken/Documents/python-machine-learning-book-2nd-edition/MNIST/"

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, 
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, 
                               '%s-images-idx3-ubyte' % kind)
    # ファイルを読み込む
    # 引数にファイル、モードを指定    
    with open(labels_path, 'rb') as lbpath:
        # バイナリを文字列に変換:unpack関数の引数にフォーマット、8バイト分のバイナリデータを指定して、
        # マジックナンバー、アイテムの個数を読み込む
        magic, n = struct.unpack('>II', 
                                 lbpath.read(8))
        # ファイルからラベルを読み込み配列を構築、：fromfile関数の引数にファイル、
        # 配列のデータ形式を指定
        labels = np.fromfile(lbpath, 
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        # 画像ピクセル情報の配列のサイズを変更
        # 行数:ラベルのサイズ、列数:特徴量の個数
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
 
    return images, labels


X_train, y_train = load_mnist(path, kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist(path, kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
#---------------------------------------------------------------------------------------
# MNISTの可視化
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
#plt.show()
plt.close()
#---------------------------------------------------------------------------------------
# 筆跡の違いの確認
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
#plt.show()
plt.close()
#---------------------------------------------------------------------------------------
# 圧縮保存
np.savez_compressed('mnist_scaled.npz', 
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test)
#---------------------------------------------------------------------------------------
# データ読み込み
mnist = np.load('mnist_scaled.npz')
print(mnist.files)

X_train, y_train, X_test, y_test = [mnist[f] for f in ['X_train', 'y_train', 
                                    'X_test', 'y_test']]

#---------------------------------------------------------------------------------------
# ニューラルネットワークの準備と学習
filename = "neural_net"
if not os.path.isfile(path + filename):
#---------------------------------------------------------------------------------------
# 初期化
  n_epochs = 200
  nn = NeuralNetMLP(n_hidden=100, 
                    l2=0.01, 
                    epochs=n_epochs, 
                    eta=0.0005,
                    minibatch_size=100, 
                    shuffle=True,
                    seed=1)
#---------------------------------------------------------------------------------------
# 学習
  nn.fit(X_train=X_train[:55000], 
         y_train=y_train[:55000],
         X_valid=X_train[55000:],
         y_valid=y_train[55000:])
  joblib.dump(nn, path + filename, compress=3)
  print("HERE_ compressing the NN module is successfully done!!")

nn = joblib.load(path + filename)

plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()

plt.plot(range(nn.epochs), nn.eval_['train_acc'], 
         label='training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'], 
         label='validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred)
       .astype(np.float) / X_test.shape[0])

print('Test accuracy: %.2f%%' % (acc * 100))





















