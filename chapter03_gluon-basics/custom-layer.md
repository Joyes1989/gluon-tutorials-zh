# 设计自定义层

神经网络的一个魅力是它有大量的层，例如全连接、卷积、循环、激活，和各式花样的连接方式。我们之前学到了如何使用Gluon提供的层来构建新的层(`nn.Block`)继而得到神经网络。虽然Gluon提供了大量的[层的定义](https://mxnet.incubator.apache.org/versions/master/api/python/gluon/gluon.html#neural-network-layers)，但我们仍然会遇到现有层不够用的情况。

这时候的一个自然的想法是，我们不是学习了如何只使用基础数值运算包`NDArray`来实现各种的模型吗？它提供了大量的[底层计算函数](https://mxnet.incubator.apache.org/versions/master/api/python/ndarray/ndarray.html)足以实现即使不是100%那也是95%的神经网络吧。

但每次都从头写容易写到怀疑人生。实际上，即使在纯研究的领域里，我们也很少发现纯新的东西，大部分时候是在现有模型的基础上做一些改进。所以很可能大部分是可以沿用前面的而只有一部分是需要自己来实现。

这个教程我们将介绍如何使用底层的`NDArray`接口来实现一个`Gluon`的层，从而可以以后被重复调用。

## 定义一个简单的层

我们先来看如何定义一个简单层，它不需要维护模型参数。事实上这个跟前面介绍的如何使用nn.Block没什么区别。下面代码定义一个层将输入减掉均值。

```{.python .input  n=22}
from mxnet import nd
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
        
    def forward(self, x):
        return x - x.mean()
```

我们可以马上实例化这个层用起来。

```{.python .input  n=23}
layer = CenteredLayer()
layer(nd.array([1,2,3,4,5]))
```

```{.json .output n=23}
[
 {
  "data": {
   "text/plain": "\n[-2. -1.  0.  1.  2.]\n<NDArray 5 @cpu(0)>"
  },
  "execution_count": 23,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们也可以用它来构造更复杂的神经网络：

```{.python .input  n=24}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128))
    net.add(nn.Dense(10))
    net.add(CenteredLayer())
```

确认下输出的均值确实是0：

```{.python .input  n=25}
net.initialize()
# net(x) 函数会直接调用网络对应的forward函数
y = net(nd.random.uniform(shape=(4, 8)))
print(y)
y.mean()
```

```{.json .output n=25}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-0.01488781  0.00914514 -0.02300199  0.01776232 -0.03973699  0.02424165\n   0.01620074  0.02470697 -0.0047338  -0.02825859]\n [-0.06341734  0.00343726 -0.00178964  0.02930544 -0.04777553  0.01109379\n   0.02414597  0.03039037  0.06638297 -0.01333844]\n [-0.01133764  0.00540919  0.01569453  0.0175448  -0.0720921   0.00366512\n   0.01911537  0.021686    0.02784117 -0.03699309]\n [-0.04692436  0.01590124 -0.02870459  0.02790766 -0.02764266  0.010023\n   0.01058409  0.01847621  0.02016774 -0.01019422]]\n<NDArray 4x10 @cpu(0)>\n"
 },
 {
  "data": {
   "text/plain": "\n[ -1.35041778e-09]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 25,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

当然大部分情况你可以看不到一个实实在在的0，而是一个很小的数。例如`5.82076609e-11`。这是因为MXNet默认使用32位float，会带来一定的浮点精度误差。

## 带模型参数的自定义层

虽然`CenteredLayer`可能会告诉实现自定义层大概是什么样子，但它缺少了重要的一块，就是它没有可以学习的模型参数。

记得我们之前访问`Dense`的权重的时候是通过`dense.weight.data()`，这里`weight`是一个`Parameter`的类型。我们可以显示的构建这样的一个参数。

```{.python .input  n=26}
from mxnet import gluon
# 生成指定名字的参数
my_param = gluon.Parameter("exciting_parameter_yay", shape=(3,3))
```

这里我们创建一个$3\times3$大小的参数并取名为"exciting_parameter_yay"。然后用默认方法初始化打印结果。

```{.python .input  n=27}
my_param.initialize()
(my_param.data(), my_param.grad())
```

```{.json .output n=27}
[
 {
  "data": {
   "text/plain": "(\n [[-0.01709467  0.01273976 -0.0414004 ]\n  [ 0.0298536   0.00970284 -0.02629956]\n  [ 0.03533129 -0.04221213  0.02729324]]\n <NDArray 3x3 @cpu(0)>, \n [[ 0.  0.  0.]\n  [ 0.  0.  0.]\n  [ 0.  0.  0.]]\n <NDArray 3x3 @cpu(0)>)"
  },
  "execution_count": 27,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

通常自定义层的时候我们不会直接创建Parameter，而是用过Block自带的一个ParamterDict类型的成员变量`params`，顾名思义，这是一个由字符串名字映射到Parameter的字典。

```{.python .input  n=28}
pd = gluon.ParameterDict(prefix="block1_")
pd.get("exciting_parameter_yay", shape=(3,3))
pd
```

```{.json .output n=28}
[
 {
  "data": {
   "text/plain": "block1_ (\n  Parameter block1_exciting_parameter_yay (shape=(3, 3), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 28,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

现在我们看下如果如果实现一个跟`Dense`一样功能的层，它概念跟前面的`CenteredLayer`的主要区别是我们在初始函数里通过`params`创建了参数：

```{.python .input  n=29}
class MyDense(nn.Block):
    # units --- 输出大小
    # in_units --- 输入大小
    # kwargs --- 给nn.Block使用的参数
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))        

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)
```

我们创建实例化一个对象来看下它的参数，这里我们特意加了前缀`prefix`，这是`nn.Block`初始化函数自带的参数。

```{.python .input  n=30}
dense = MyDense(5, in_units=10, prefix='o_my_dense_')
dense.params
```

```{.json .output n=30}
[
 {
  "data": {
   "text/plain": "o_my_dense_ (\n  Parameter o_my_dense_weight (shape=(10, 5), dtype=<class 'numpy.float32'>)\n  Parameter o_my_dense_bias (shape=(5,), dtype=<class 'numpy.float32'>)\n)"
  },
  "execution_count": 30,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

它的使用跟前面没有什么不一致：

```{.python .input  n=31}
dense.initialize()
dense(nd.random.uniform(shape=(2,10)))
```

```{.json .output n=31}
[
 {
  "data": {
   "text/plain": "\n[[ 0.          0.          0.          0.          0.        ]\n [ 0.03771835  0.          0.          0.          0.        ]]\n<NDArray 2x5 @cpu(0)>"
  },
  "execution_count": 31,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们构造的层跟Gluon提供的层用起来没太多区别：

```{.python .input  n=32}
net = nn.Sequential()
with net.name_scope():
    net.add(MyDense(32, in_units=64))
    net.add(MyDense(2, in_units=32))
net.initialize()
net(nd.random.uniform(shape=(2,64)))
```

```{.json .output n=32}
[
 {
  "data": {
   "text/plain": "\n[[ 0.  0.]\n [ 0.  0.]]\n<NDArray 2x2 @cpu(0)>"
  },
  "execution_count": 32,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=33}
nn.Dense??

```

**仔细的你可能还是注意到了，我们这里指定了输入的大小，而Gluon自带的`Dense`则无需如此。我们已经在前面节介绍过了这个延迟初始化如何使用。但如果实现一个这样的层我们将留到后面介绍了hybridize后。**

## 总结

现在我们知道了如何把前面手写过的层全部包装了Gluon能用的Block，之后再用到的时候就可以飞起来了！

## 练习

1. 怎么修改自定义层里参数的默认初始化函数。
1. (这个比较难），在一个代码Cell里面输入`nn.Dense??`，看看它是怎么实现的。为什么它就可以支持延迟初始化了。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1256)
