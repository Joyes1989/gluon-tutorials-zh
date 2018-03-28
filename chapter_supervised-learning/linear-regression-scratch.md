# 线性回归 --- 从0开始

尽管强大的深度学习框架可以减少大量重复性工作，但若过于依赖它提供的便利，你就会很难深入理解深度学习是如何工作的。因此，我们的第一个教程是如何只利用ndarray和autograd来实现一个线性回归的训练。

## 线性回归

给定一个数据点集合`X`和对应的目标值`y`，线性模型的目标就是找到一条使用向量`w`和位移`b`描述的线，来尽可能地近似每个样本`X[i]`和`y[i]`。用数学符号来表示就是：

$$\boldsymbol{\hat{y}} = X \boldsymbol{w} + b$$

并最小化所有数据点上的平方误差

$$\sum_{i=1}^n (\hat{y}_i-y_i)^2.$$

你可能会对我们把古老的线性回归作为深度学习的一个样例表示奇怪。实际上线性模型是最简单、但也是最有用的神经网络。一个神经网络就是一个由节点（神经元）和有向边组成的集合。我们一般把一些节点组成层，每一层先从下面一层的节点获取输入，然后输出给上面的层使用。要计算一个节点值，我们需要将输入节点值做加权和（权数值即 `w`），然后再加上一个**激活函数（activation function）**。对于线性回归而言，它是一个两层神经网络，其中第一层是（下图橙色点）输入，每个节点对应输入数据点的一个维度，第二层是单输出节点（下图绿色点），它使用身份函数（$f(x)=x$）作为激活函数。

![](../img/onelayer.png)

## 创建数据集

这里我们使用一个数据集来尽量简单地解释清楚，真实的模型是什么样的。具体来说，我们使用如下方法来生成数据；随机数值 `X[i]`，其相应的标注为 `y[i]`：

`y[i] = 2 * X[i][0] - 3.4 * X[i][1] + 4.2 + noise`

使用数学符号表示：

$$y = X \cdot w + b + \eta, \quad \text{for } \eta \sim \mathcal{N}(0,\sigma^2)$$

这里噪音服从均值0和标准差为0.01的正态分布。

```{.python .input  n=2}
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

```{.python .input  n=10}
print(X[0][:], X[0:10], y[0])
print(X.shape, y.shape)
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 0.13263007  0.72455114]\n<NDArray 2 @cpu(0)> \n[[ 0.13263007  0.72455114]\n [ 0.75390851 -1.08878267]\n [-0.40836138 -0.6448437 ]\n [-0.95328027 -0.30781594]\n [ 0.46872386  1.5440191 ]\n [ 0.76911944 -0.08660943]\n [ 0.4522135   2.02507949]\n [-0.52178961  0.26948217]\n [ 1.19464409  0.18458311]\n [-0.79071796  2.08164406]]\n<NDArray 10x2 @cpu(0)> \n[ 1.99334157]\n<NDArray 1 @cpu(0)>\n(1000, 2) (1000,)\n"
 }
]
```

如果有兴趣，可以使用安装包中已包括的 Python 绘图包 `matplotlib`，生成第二个特征值 (`X[:, 1]`) 和目标值 `Y` 的散点图，更直观地观察两者间的关系。

```{.python .input  n=12}
import matplotlib.pyplot as plt
plt.scatter(X[:, 1].asnumpy(),y.asnumpy())
plt.show()
```

```{.json .output n=12}
[
 {
  "data": {
   "image/png": "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\nSUVORK5CYII=\n",
   "text/plain": "<matplotlib.figure.Figure at 0x2491e0edeb8>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

## 数据读取

当我们开始训练神经网络的时候，我们需要不断读取数据块。这里我们定义一个函数它每次返回`batch_size`个随机的样本和对应的目标。我们通过python的`yield`来构造一个迭代器。

```{.python .input  n=13}
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

```{.python .input  n=14}
for data, label in data_iter():
    print(data, label)
    break
```

```{.json .output n=14}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.85018671 -0.24604096]\n [ 1.11653626 -0.51288116]\n [-0.46708775 -1.45037067]\n [ 0.3376039   1.42488885]\n [-0.11001937 -0.51679802]\n [-1.79932845 -0.87342685]\n [-0.42830783  0.48033428]\n [-0.06388734 -2.479038  ]\n [-0.48970276 -0.42705107]\n [ 0.38369063 -0.09416867]]\n<NDArray 10x2 @cpu(0)> \n[  6.71870279   8.15876293   8.18831921   0.0352109    5.74462986\n   3.57594538   1.71133506  12.50903416   4.68470144   5.29160929]\n<NDArray 10 @cpu(0)>\n"
 }
]
```

## 初始化模型参数

下面我们随机初始化模型参数

```{.python .input  n=15}
w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros((1,))
params = [w, b]
print(w, b)
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.61778706]\n [-1.1097393 ]]\n<NDArray 2x1 @cpu(0)> \n[ 0.]\n<NDArray 1 @cpu(0)>\n"
 }
]
```

之后训练时我们需要对这些参数求导来更新它们的值，使损失尽量减小；因此我们需要创建它们的梯度。

```{.python .input  n=16}
for param in params:
    param.attach_grad()
```

## 定义模型

线性模型就是将输入和模型的权重（`w`）相乘，再加上偏移（`b`）：

```{.python .input  n=17}
def net(X):
    return nd.dot(X, w) + b
```

```{.python .input  n=20}
print(data, net(data))
```

```{.json .output n=20}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.85018671 -0.24604096]\n [ 1.11653626 -0.51288116]\n [-0.46708775 -1.45037067]\n [ 0.3376039   1.42488885]\n [-0.11001937 -0.51679802]\n [-1.79932845 -0.87342685]\n [-0.42830783  0.48033428]\n [-0.06388734 -2.479038  ]\n [-0.48970276 -0.42705107]\n [ 0.38369063 -0.09416867]]\n<NDArray 10x2 @cpu(0)> \n[[ 0.79827565]\n [ 1.25894606]\n [ 1.32097256]\n [-1.37268782]\n [ 0.50554252]\n [-0.14232571]\n [-0.79764891]\n [ 2.71161723]\n [ 0.17138334]\n [ 0.34154177]]\n<NDArray 10x1 @cpu(0)>\n"
 }
]
```

## 损失函数

我们使用常见的平方误差来衡量预测目标和真实目标之间的差距。

```{.python .input  n=18}
def square_loss(yhat, y):
    # 注意这里我们把y变形成yhat的形状来避免矩阵形状的自动转换
    return (yhat - y.reshape(yhat.shape)) ** 2
```

## 优化

虽然线性回归有显式解，但绝大部分模型并没有。所以我们这里通过随机梯度下降来求解。每一步，我们将模型参数沿着梯度的反方向走特定距离，这个距离一般叫**学习率（learning rate）** `lr`。（我们会之后一直使用这个函数，我们将其保存在[utils.py](../utils.py)。）

```{.python .input  n=10}
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
```

## 训练

现在我们可以开始训练了。训练通常需要迭代数据数次，在这里使用`epochs`表示迭代总次数；一次迭代中，我们每次随机读取固定数个数据点，计算梯度并更新模型参数。

```{.python .input}
# 模型函数
def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2
# 绘制损失随训练次数降低的折线图，以及预测值和真实值的散点图
def plot(losses, X, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             net(X[:sample_size, :]).asnumpy(), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             real_fn(X[:sample_size, :]).asnumpy(), '*g', label='Real')
    fg2.legend()
    plt.show()
```

```{.python .input  n=11}
epochs = 5
learning_rate = .001
niter = 0
losses = []
moving_loss = 0
smoothing_constant = .01

# 训练
for e in range(epochs):    
    total_loss = 0

    for data, label in data_iter():
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)
        total_loss += nd.sum(loss).asscalar()

        # 记录每读取一个数据点后，损失的移动平均值的变化；
        niter +=1
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss

        # correct the bias from the moving averages
        est_loss = moving_loss/(1-(1-smoothing_constant)**niter)

        if (niter + 1) % 100 == 0:
            losses.append(est_loss)
            print("Epoch %s, batch %s. Moving avg of loss: %s. Average loss: %f" % (e, niter, est_loss, total_loss/num_examples))
            plot(losses, X)
```

训练完成后，我们可以比较学得的参数和真实参数

```{.python .input  n=12}
true_w, w
```

```{.python .input  n=13}
true_b, b
```

## 结论

我们现在看到，仅仅是使用NDArray和autograd就可以很容易实现的一个模型。在接下来的教程里，我们会在此基础上，介绍更多现代神经网络的知识，以及怎样使用少量的MXNet代码实现各种复杂的模型。

## 练习

尝试用不同的学习率查看误差下降速度（收敛率）

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/743)
