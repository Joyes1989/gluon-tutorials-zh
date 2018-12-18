# 初始化模型参数

我们仍然用MLP这个例子来详细解释如何初始化模型参数。

```{.python .input  n=10}
from mxnet.gluon import nn
from mxnet import nd

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        # 两层的网络：一层输出为4的网络 + 一层输出为2的网络
        net.add(nn.Dense(4, activation="relu"))
        net.add(nn.Dense(2))
    return net

x = nd.random.uniform(shape=(3,5))
```

我们知道如果不`initialize()`直接跑forward，那么系统会抱怨说参数没有初始化。

```{.python .input  n=9}
import sys
try:
    net = get_net()
    print(net)
    # 会失败：net未进行权重Weight的初始化
    net(x)
except RuntimeError as err:
    sys.stderr.write(str(err))
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): Dense(None -> 4, Activation(relu))\n  (1): Dense(None -> 2, linear)\n)\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Parameter 'sequential2_dense0_weight' has not been initialized. Note that you should initialize parameters and create Trainer with Block.collect_params() instead of Block.params because the later does not include Parameters of nested child Blocks"
 }
]
```

正确的打开方式是这样

```{.python .input  n=11}
net.initialize()
print(net)
net(x)
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential(\n  (0): Dense(None -> 4, Activation(relu))\n  (1): Dense(None -> 2, linear)\n)\n"
 },
 {
  "data": {
   "text/plain": "\n[[ 0.00546073  0.00333483]\n [ 0.00617615  0.00437204]\n [ 0.00130519  0.00104359]]\n<NDArray 3x2 @cpu(0)>"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 访问模型参数

之前我们提到过可以通过`weight`和`bias`访问`Dense`的参数，他们是`Parameter`这个类：
### 通过下标来访问多层网络的不同层，层下标从0开始：
    1. 这里的net[0]可以访问到网络的第一层（输出为0）
#### 对于每层的网络，可以通过weight、bias成员获取网络的参数：
    1. net[0]是一个dense网络，具有weight/bias参数
    2. 通过weight.data()、bias.data()来获取具体参数值
    3. 通过weight.grad()、bias.grad()来获取参数对应的梯度

```{.python .input  n=12}
w = net[0].weight
b = net[0].bias
print('name: ', net[0].name, '\nweight: ', w, '\nbias: ', b)
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "name:  sequential2_dense0 \nweight:  Parameter sequential2_dense0_weight (shape=(4, 5), dtype=float32) \nbias:  Parameter sequential2_dense0_bias (shape=(4,), dtype=float32)\n"
 }
]
```

然后我们可以通过`data`来访问参数，`grad`来访问对应的梯度

```{.python .input  n=13}
print('weight:', w.data())
print('weight gradient', w.grad())
print('bias:', b.data())
print('bias gradient', b.grad())
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "weight: \n[[-0.03578042  0.01729142 -0.04774426 -0.02267893 -0.05454748]\n [ 0.02446532  0.02188614 -0.02559176 -0.05065439  0.03896836]\n [-0.04247847  0.06293995 -0.01837847  0.02275376  0.04493906]\n [-0.06809997 -0.05640582  0.01719845  0.04731229  0.02431235]]\n<NDArray 4x5 @cpu(0)>\nweight gradient \n[[ 0.  0.  0.  0.  0.]\n [ 0.  0.  0.  0.  0.]\n [ 0.  0.  0.  0.  0.]\n [ 0.  0.  0.  0.  0.]]\n<NDArray 4x5 @cpu(0)>\nbias: \n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\nbias gradient \n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

### 我们也可以通过`collect_params`来访问Block里面所有的参数
    0. collect_params返回结果为dict类型，key为各变量名（格式为block.md中介绍的名字）
    1. （这个会包括所有的子Block）。
    2. 它会返回一个名字到对应Parameter的dict。
    3. 既可以用正常`[]`来访问参数，也可以用`get()`，它不需要填写名字的前缀。

```{.python .input  n=15}
params = net.collect_params()
print(params)
# sequential8_dense0_bias 这个名字是系统默认生成的，可参考block.md
#print(params['sequential8_dense0_bias'].data())
print(params.get('dense0_weight').data())
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "sequential2_ (\n  Parameter sequential2_dense0_weight (shape=(4, 5), dtype=float32)\n  Parameter sequential2_dense0_bias (shape=(4,), dtype=float32)\n  Parameter sequential2_dense1_weight (shape=(2, 4), dtype=float32)\n  Parameter sequential2_dense1_bias (shape=(2,), dtype=float32)\n)\n\n[[-0.03578042  0.01729142 -0.04774426 -0.02267893 -0.05454748]\n [ 0.02446532  0.02188614 -0.02559176 -0.05065439  0.03896836]\n [-0.04247847  0.06293995 -0.01837847  0.02275376  0.04493906]\n [-0.06809997 -0.05640582  0.01719845  0.04731229  0.02431235]]\n<NDArray 4x5 @cpu(0)>\n"
 }
]
```

## 使用不同的初始函数来初始化


    1. 我们一直在使用默认的`initialize`来初始化权重（除了指定GPU `ctx`外）。它会把所有权重初始化成在`[-0.07, 0.07]`之间均匀分布的随机数。我们可以使用别的初始化方法。例如使用均值为0，方差为0.02的正态分布
    2. 通过制定init参数的值，来设置不同的初始化方式


```{.python .input  n=7}
from mxnet import init
# sigma -- σ（西格玛）指标准差
# 重复初始化需要制定force_reinit=True
params.initialize(init=init.Normal(sigma=0.02), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.02804598  0.00220872  0.00701151  0.02721515  0.00500832]\n [ 0.00112992  0.03227538 -0.01813176 -0.00385197 -0.01286032]\n [ 0.03360647 -0.02855298 -0.03083278 -0.02110904 -0.02623655]\n [-0.00293494  0.01282986 -0.01476416  0.04062728  0.01186533]]\n<NDArray 4x5 @cpu(0)> \n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

看得更加清楚点：

```{.python .input  n=53}
params.initialize(init=init.One(), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())
```

```{.json .output n=53}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 1.  1.  1.  1.  1.]\n [ 1.  1.  1.  1.  1.]\n [ 1.  1.  1.  1.  1.]\n [ 1.  1.  1.  1.  1.]]\n<NDArray 4x5 @cpu(0)> \n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

更多的方法参见[init的API](https://mxnet.incubator.apache.org/api/python/optimization.html#the-mxnet-initializer-package). 下面我们自定义一个初始化方法。

```{.python .input  n=54}
class MyInit(init.Initializer):
    def __init__(self):
        super(MyInit, self).__init__()
        # 控制输出细节信息
        self._verbose = True
    def _init_weight(self, _, arr):
        # 初始化权重，使用out=arr后我们不需指定形状
        print('init weight', arr.shape)
        nd.random.uniform(low=5, high=10, out=arr)
    def _init_bias(self, _, arr):
        print('init bias', arr.shape)
        # 初始化偏移
        arr[:] = 2

# FIXME: init_bias doesn't work
params.initialize(init=MyInit(), force_reinit=True)
print(net[0].weight.data(), net[0].bias.data())
```

```{.json .output n=54}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "init weight (4, 5)\ninit weight (2, 4)\n\n[[ 8.37423706  7.58689547  6.38446903  5.66034031  5.87454414]\n [ 8.58429909  8.52237129  6.98029852  7.31575108  7.82710648]\n [ 9.20214272  5.916399    6.02432871  5.72423887  5.82479429]\n [ 7.44028139  5.62416553  6.77806377  8.61040306  9.70215988]]\n<NDArray 4x5 @cpu(0)> \n[ 0.  0.  0.  0.]\n<NDArray 4 @cpu(0)>\n"
 }
]
```

## 延后的初始化 **

我们之前提到过Gluon的一个便利的地方是模型定义的时候不需要指定输入的大小，在之后做forward的时候会自动推测参数的大小。我们具体来看这是怎么工作的。

新创建一个网络，然后打印参数。你会发现两个全连接层的权重的形状里都有0。 这是因为在不知道输入数据的情况下，我们无法判断它们的形状。

```{.python .input  n=59}
net = get_net()
print(net.collect_params())
# 这里可以看到多个weight的大小中存在0，
# 因为我们只在创建dense时指定了输出为4，并没给出数据的大小，因此这里确切大小是未知的
# 这里注意：只要没有输入样本数据，weight/bias都是未知的
print(net.params())
```

```{.json .output n=59}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "sequential13_ (\n  Parameter sequential13_dense0_weight (shape=(4, 0), dtype=<class 'numpy.float32'>)\n  Parameter sequential13_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential13_dense1_weight (shape=(2, 0), dtype=<class 'numpy.float32'>)\n  Parameter sequential13_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)\n"
 },
 {
  "ename": "TypeError",
  "evalue": "'ParameterDict' object is not callable",
  "output_type": "error",
  "traceback": [
   "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
   "\u001b[1;32m<ipython-input-59-d9749a14e68b>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[1;31m# \u8fd9\u91cc\u53ef\u4ee5\u770b\u5230\u591a\u4e2aweight\u7684\u5927\u5c0f\u4e2d\u5b58\u57280\uff0c\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;31m# \u56e0\u4e3a\u6211\u4eec\u53ea\u5728\u521b\u5efadense\u65f6\u6307\u5b9a\u4e86\u8f93\u51fa\u4e3a4\uff0c\u5e76\u6ca1\u7ed9\u51fa\u6570\u636e\u7684\u5927\u5c0f\uff0c\u56e0\u6b64\u8fd9\u91cc\u786e\u5207\u5927\u5c0f\u662f\u672a\u77e5\u7684\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 5\u001b[1;33m \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mnet\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mparams\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[1;31mTypeError\u001b[0m: 'ParameterDict' object is not callable"
  ]
 }
]
```

然后我们初始化

```{.python .input  n=60}
net.initialize(init=MyInit())
# 这里并没有真正执行调用MyInit函数
```

**你会看到我们并没有看到MyInit打印的东西，这是因为我们仍然不知道形状。真正的初始化发生在我们看到数据时。**

```{.python .input  n=61}
net(x)
```

```{.json .output n=61}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "init weight (4, 5)\ninit weight (2, 4)\n"
 },
 {
  "data": {
   "text/plain": "\n[[ 586.34375     467.12890625]\n [ 404.9107666   323.89505005]\n [ 643.9407959   514.78717041]]\n<NDArray 3x2 @cpu(0)>"
  },
  "execution_count": 61,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

这时候我们看到shape里面的0被填上正确的值了。

```{.python .input  n=62}
print(net.collect_params())
```

```{.json .output n=62}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "sequential13_ (\n  Parameter sequential13_dense0_weight (shape=(4, 5), dtype=<class 'numpy.float32'>)\n  Parameter sequential13_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)\n  Parameter sequential13_dense1_weight (shape=(2, 4), dtype=<class 'numpy.float32'>)\n  Parameter sequential13_dense1_bias (shape=(2,), dtype=<class 'numpy.float32'>)\n)\n"
 }
]
```

## 避免延后初始化 * * * * *

有时候我们不想要延后初始化，这时候可以在创建网络的时候指定输入大小。

```{.python .input  n=63}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, in_units=5, activation="relu"))
    net.add(nn.Dense(2, in_units=4))

net.initialize(MyInit())
```

```{.json .output n=63}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "init weight (4, 5)\ninit weight (2, 4)\n"
 }
]
```

## 共享模型参数

有时候我们想在层之间共享同一份参数，我们可以通过Block的`params`输出参数来手动指定参数，而不是让系统自动生成。

```{.python .input  n=64}
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, in_units=4, activation="relu"))
    net.add(nn.Dense(4, in_units=4, activation="relu", params=net[-1].params))
    net.add(nn.Dense(2, in_units=4))


```

初始化然后打印

```{.python .input  n=65}
net.initialize(MyInit())
print(net[0].weight.data())
print(net[1].weight.data())
```

```{.json .output n=65}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "init weight (4, 4)\ninit weight (2, 4)\n\n[[ 9.87129307  6.73616743  9.31154823  5.74070454]\n [ 8.90213299  9.90914726  9.92516136  7.39185143]\n [ 8.76784801  7.4869566   5.02024031  8.1973629 ]\n [ 6.34739685  6.84292316  7.05246067  5.68450165]]\n<NDArray 4x4 @cpu(0)>\n\n[[ 9.87129307  6.73616743  9.31154823  5.74070454]\n [ 8.90213299  9.90914726  9.92516136  7.39185143]\n [ 8.76784801  7.4869566   5.02024031  8.1973629 ]\n [ 6.34739685  6.84292316  7.05246067  5.68450165]]\n<NDArray 4x4 @cpu(0)>\n"
 }
]
```

## 总结

我们可以很灵活地访问和修改模型参数。

## 练习

1. 研究下`net.collect_params()`返回的是什么？`net.params`呢？
1. 如何对每个层使用不同的初始化函数
1. 如果两个层共用一个参数，那么求梯度的时候会发生什么？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/987)
