# 多类逻辑回归 --- 使用Gluon

现在让我们使用gluon来更快速地实现一个多类逻辑回归。

## 获取和读取数据

我们仍然使用FashionMNIST。我们将代码保存在[../utils.py](../utils.py)这样这里不用复制一遍。

```{.python .input  n=12}
import sys
sys.path.append('..')
import utils

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
```

## 定义和初始化模型

我们先使用Flatten层将输入数据转成 `batch_size` x `?` 的矩阵，然后输入到10个输出节点的全连接层。照例我们不需要制定每层输入的大小，gluon会做自动推导。

```{.python .input  n=9}
from mxnet import gluon

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(10))
net.initialize()
```

## Softmax和交叉熵损失函数

如果你做了上一章的练习，那么你可能意识到了分开定义Softmax和交叉熵会有数值不稳定性。因此gluon提供一个将这两个函数合起来的数值更稳定的版本

```{.python .input  n=10}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 优化

```{.python .input  n=11}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

## 训练

```{.python .input  n=13}
from mxnet import ndarray as nd
from mxnet import autograd

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.795451, Train acc 0.742238, Test acc 0.764223\nEpoch 1. Loss: 0.575475, Train acc 0.809445, Test acc 0.823918\nEpoch 2. Loss: 0.530982, Train acc 0.823501, Test acc 0.827224\nEpoch 3. Loss: 0.507065, Train acc 0.829627, Test acc 0.838542\nEpoch 4. Loss: 0.490312, Train acc 0.833801, Test acc 0.830529\n"
 }
]
```

## 结论

Gluon提供的函数有时候比手工写的数值更稳定。

## 练习

- 再尝试调大下学习率看看？
- 为什么参数都差不多，但gluon版本比从0开始的版本精度更高？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/740)
