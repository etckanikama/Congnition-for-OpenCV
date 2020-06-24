# モデルの作成
import numpy as np
from chainer import Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from sklearn.datasets import fetch_openml

# 1,学習モデル（多層ニューラルネットワークの作成）

# MyMLPクラスはchainer.Chainクラスを継承しており、2つの特殊メソッドを持つ
class MyMLP(Chain):
    # 特殊メソッド(MYMLPクラスがインスタンス化されたときの)
    # 入力層=784（28×28ピクセルのため）,中間層=100,出力層=10
    def __init__(self,n_in=784,n_units=100,n_out=10):
        # 継承？
        super(MyMLP,self).__init__(
        # L.Linear(chainer.links.Linear)関数を利用して学習モデルを作成
        l1 = L.Linear(n_in, n_units),
        l2 = L.Linear(n_units,n_units),
        l3 = L.Linear(n_units,n_out),
        )



    # 特殊メソッド（）
    def __call__(self,x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

# プログラムが開始したことを示す(MNISTのダウンロードに時間を要するため)
print('Start')

# 2, データセットの準備
"""
MINSTを使うために,sklearn.datasets.fetch_openml関数でMNISTのデータセットをダウンロードして、mnist_x,mnist_yに格納する
ダウンロードに時間がかかるが、data_home引数に指定したMNISTのデータセットが存在していればダウンロードはない。data_homeで指定したフォルダにopenmlフォルダが
が作成されその中に保存される

"""

mnist_X, mnist_y = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)

# 3,データセットの変換
x_all = mnist_X.astype(np.float32) / 255
y_all = mnist_y.astype(np.int32) 

# optimizerの作成 
model = MyMLP()
optimizer = optimizers.SGD()
optimizer.setup(model)

# optimizerの最適化
BATCHSIZE = 100
DATASIZE = 70000

for epoch in range(20):
    print('epoch %d' % epoch)
    indexes = np.random.permutation(DATASIZE)
    for i in range(0, DATASIZE, BATCHSIZE):
        x = Variable(x_all[indexes[i:i+BATCHSIZE]])
        t = Variable(y_all[indexes[i:i+BATCHSIZE]])

        model.zerograds()

        y = model(x)

        loss = F.softmax_cross_entropy(y,t)

        loss.backward()

        optimizer.update()

serializers.save_npz("mymodel.npz",model)

# プログラム終了
print("Finish")



