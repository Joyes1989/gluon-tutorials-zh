# 线性回归 --- 从0开始

虽然强大的深度学习框架可以减少很多重复性工作，但如果你过于依赖它提供的便利抽象，那么你可能不会很容易地理解到底深度学习是如何工作的。所以我们的第一个教程是如何只利用ndarray和autograd来实现一个线性回归的训练。

## 线性回归

给定一个数据点集合`X`和对应的目标值`y`，线性模型的目标是找一根线，其由向量`w`和位移`b`组成，来最好地近似每个样本`X[i]`和`y[i]`。用数学符号来表示就是我们将学`w`和`b`来预测，

$$\boldsymbol{\hat{y}} = X \boldsymbol{w} + b$$

并最小化所有数据点上的平方误差

$$\sum_{i=1}^n (\hat{y}_i-y_i)^2.$$

你可能会对我们把古老的线性回归作为深度学习的一个样例表示很奇怪。实际上线性模型是最简单但也可能是最有用的神经网络。一个神经网络就是一个由节点（神经元）和有向边组成的集合。我们一般把一些节点组成层，每一层使用下一层的节点作为输入，并输出给上面层使用。为了计算一个节点值，我们将输入节点值做加权和，然后再加上一个激活函数。对于线性回归而言，它是一个两层神经网络，其中第一层是（下图橙色点）输入，每个节点对应输入数据点的一个维度，第二层是单输出节点（下图绿色点），它使用身份函数（$f(x)=x$）作为激活函数。

![](../img/simple-net-linear.png)

## 创建数据集

这里我们使用一个人工数据集来把事情弄简单些，因为这样我们将知道真实的模型是什么样的。具体来说我们使用如下方法来生成数据

`y[i] = 2 * X[i][0] - 3.4 * X[i][1] + 4.2 + noise`

这里噪音服从均值0和标准差为0.01的正态分布。

```{.python .input  n=13}
from mxnet import ndarray as nd
from mxnet import autograd

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)
```

注意到`X`的每一行是一个长度为2的向量，而`y`的每一行是一个长度为1的向量（标量）。

```{.python .input  n=2}
print(X[0], y[0])
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 1.16307855  2.21220613]\n<NDArray 2 @cpu(0)> \n[-1.0015316]\n<NDArray 1 @cpu(0)>\n"
 }
]
```

## 数据读取

当我们开始训练神经网络的时候，我们需要不断读取数据块。这里我们定义一个函数它每次返回`batch_size`个随机的样本和对应的目标。我们通过python的`yield`来构造一个迭代器。

```{.python .input  n=3}
import random
batch_size = 10
def data_iter():
    # 产生一个随机索引
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i+batch_size,num_examples)])
        yield nd.take(X, j), nd.take(y, j)
```

下面代码读取第一个随机数据块

```{.python .input  n=4}
for data, label in data_iter():
    print(data, label)
    break
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-0.60153437  1.18371439]\n [ 1.68393028  0.77373821]\n [ 0.27119216  0.67585617]\n [-0.70443887  0.11922956]\n [ 0.27752855  0.68164563]\n [ 0.11222199 -0.09502531]\n [-0.19508816  0.6325348 ]\n [-1.27279079  1.47948802]\n [-0.98571479  0.56402695]\n [-0.46957269  1.73034632]]\n<NDArray 10x2 @cpu(0)> \n[-1.02981758  4.93909216  2.4455936   2.38639784  2.4427011   4.74484682\n  1.66952765 -3.38364601  0.31658202 -2.62291455]\n<NDArray 10 @cpu(0)>\n"
 }
]
```

## 初始化模型参数

下面我们随机初始化模型参数

```{.python .input  n=5}
w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros((1,))
params = [w, b]
```

之后训练时我们需要对这些参数求导来更新它们的值，所以我们需要创建它们的梯度。

```{.python .input  n=6}
for param in params:
    param.attach_grad()
```

## 定义模型

线性模型就是将输入和模型做乘法再加上偏移：

```{.python .input  n=7}
# 用于进行预测的模型
def net(X):
    return nd.dot(X, w) + b
```

## 损失函数

我们使用常见的平方误差来衡量预测目标和真实目标之间的差距。

```{.python .input  n=8}
def square_loss(yhat, y):
    # 注意这里我们把y变形成yhat的形状来避免自动广播
    return (yhat - y.reshape(yhat.shape)) ** 2
```

## 优化

虽然线性回归有显试解，但绝大部分模型并没有。所以我们这里通过随机梯度下降来求解。每一步，我们将模型参数沿着梯度的反方向走特定距离，这个距离一般叫学习率。（我们会之后一直使用这个函数，我们将其保存在[utils.py](../utils.py)。）

```{.python .input  n=9}
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
```

## 训练

现在我们可以开始训练了。训练通常需要迭代数据数次，一次迭代里，我们每次随机读取固定数个数据点，计算梯度并更新模型参数。

```{.python .input  n=10}
epochs = 5
learning_rate = .001
# 总共迭代epochs次
for e in range(epochs):
    total_loss = 0
    # 需要对每个样本执行一次网络运算
    for data, label in data_iter():
        # 使用with来帮助做“准备”和“清理”工作
        # 使用autograd.record()来显示要求MXNet来记录求导顺序
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        # 每执行一个样本就更新一次梯度，更新后参数在下次执行net时起作用
        # 由于w的变化是和单个样本x无关的，所以可以对每个样本进行迭代并进行w的更新
        # --- 即当w偏大时，不管应用到哪个样本xi上，Loss对应w的导数都是整数（使得w下次更新变小）
        SGD(params, learning_rate)

        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0, average loss: 9.434924\nEpoch 1, average loss: 0.211757\nEpoch 2, average loss: 0.004862\nEpoch 3, average loss: 0.000208\nEpoch 4, average loss: 0.000105\n"
 }
]
```

训练完成后我们可以比较学到的参数和真实参数

```{.python .input  n=11}
true_w, w
```

```{.json .output n=11}
[
 {
  "data": {
   "text/plain": "([2, -3.4], \n [[ 1.9994446 ]\n  [-3.39988399]]\n <NDArray 2x1 @cpu(0)>)"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=12}
true_b, b
```

```{.json .output n=12}
[
 {
  "data": {
   "text/plain": "(4.2, \n [ 4.19944382]\n <NDArray 1 @cpu(0)>)"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 结论

我们现在看到仅仅使用NDArray和autograd我们可以很容易地实现一个模型。

## 练习

尝试用不同的学习率查看误差下降速度（收敛率）

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/743)
