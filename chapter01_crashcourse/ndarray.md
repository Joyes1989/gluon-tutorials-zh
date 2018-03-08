# 使用NDArray来处理数据

对于机器学习来说，处理数据往往是万事之开头。它包含两个部分：数据读取和当数据已经在内存里时如何处理。本章将关注后者。我们首先介绍`NDArray`，这是MXNet储存和变换数据的主要工具。如果你之前用过`NumPy`，你会发现`NDArray`和`NumPy`的多维数组非常类似。当然，`NDArray`提供更多的功能，首先是CPU和GPU的异步计算，其次是自动求导。这两点使得`NDArray`能更好地支持机器学习。

## 让我们开始

我们先介绍最基本的功能。如果你不懂我们用到的数学操作也不用担心，例如按元素加法，或者正态分布，我们会在之后的章节分别详细介绍。

我们首先从`mxnet`导入`ndarray`这个包

```{.python .input  n=23}
from mxnet import ndarray as nd
```

然后我们创建一个有3行和4列的2D数组（通常也叫矩阵），并且把每个元素初始化成0

```{.python .input  n=25}
nd.zeros((3, 4))
```

```{.json .output n=25}
[
 {
  "data": {
   "text/plain": "\n[[ 0.  0.  0.  0.]\n [ 0.  0.  0.  0.]\n [ 0.  0.  0.  0.]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 25,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

类似的，我们可以创建数组每个元素被初始化成1。

```{.python .input  n=27}
x = nd.ones((3, 4))
print(x, x)
```

```{.json .output n=27}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]]\n<NDArray 3x4 @cpu(0)> \n[[ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]]\n<NDArray 3x4 @cpu(0)>\n"
 }
]
```

或者从python的数组直接构造

```{.python .input  n=28}
nd.array([[1, 2], [3, 5]])
```

```{.json .output n=28}
[
 {
  "data": {
   "text/plain": "\n[[ 1.  2.]\n [ 3.  5.]]\n<NDArray 2x2 @cpu(0)>"
  },
  "execution_count": 28,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们经常需要创建随机数组，就是说每个元素的值都是随机采样而来，这个经常被用来初始化模型参数。下面创建数组，它的元素服从均值0方差1的正态分布。

```{.python .input  n=60}
y = nd.random_normal(0, 1, (3, 3))
y
```

```{.json .output n=60}
[
 {
  "data": {
   "text/plain": "\n[[ 0.05383794 -0.97110999 -2.50748062]\n [-0.77569664 -0.59164989 -0.78821766]\n [ 0.85860497  0.74177283 -0.22794183]]\n<NDArray 3x3 @cpu(0)>"
  },
  "execution_count": 60,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

跟`NumPy`一样，每个数组的形状可以通过`.shape`来获取

```{.python .input  n=61}
print(x.shape, y.shape)
```

```{.json .output n=61}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(3, 4) (3, 3)\n"
 }
]
```

它的大小，就是总元素个数，是形状的累乘。

```{.python .input  n=62}
print(x.size, y.size)
```

```{.json .output n=62}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "12 9\n"
 }
]
```

## 操作符

NDArray支持大量的数学操作符，例如按元素加法：

```{.python .input  n=68}
print("x.shape: {0}, x: {1}".format(x.shape, x))
print("y.shape: {0}, y: {1}".format(y.shape, y))
y = nd.random_normal(0, 10, shape = (3, 4))
print("y.shape: {0}, y: {1}".format(y.shape, y))
z = x + y
print("z = x + y, z: {0}, z.shape: {1}".format(z, z.shape))
```

```{.json .output n=68}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "x.shape: (3, 4), x: \n[[ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]]\n<NDArray 3x4 @cpu(0)>\ny.shape: (3, 4), y: \n[[  2.01314759 -10.73092651   3.50054717 -10.42482758]\n [  5.36052132 -13.27884865  15.1944437  -14.74966049]\n [ 19.0408783   -5.24141979 -15.73443222  12.66255569]]\n<NDArray 3x4 @cpu(0)>\ny.shape: (3, 4), y: \n[[ -1.40078664   8.95064259   2.9670074   -6.01594448]\n [ 13.11195183  12.04055882   5.03590393  -9.71219349]\n [-11.89445019  -5.82562208  -5.50213766   3.71707773]]\n<NDArray 3x4 @cpu(0)>\nz = x + y, z: \n[[ -0.40078664   9.95064259   3.9670074   -5.01594448]\n [ 14.11195183  13.04055882   6.03590393  -8.71219349]\n [-10.89445019  -4.82562208  -4.50213766   4.71707773]]\n<NDArray 3x4 @cpu(0)>, z.shape: (3, 4)\n"
 }
]
```

乘法：

```{.python .input  n=70}
print(x.shape, y.shape)
# 矩阵中的对应元素相乘
x * y
```

```{.json .output n=70}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(3, 4) (3, 4)\n"
 },
 {
  "data": {
   "text/plain": "\n[[ -1.40078664   8.95064259   2.9670074   -6.01594448]\n [ 13.11195183  12.04055882   5.03590393  -9.71219349]\n [-11.89445019  -5.82562208  -5.50213766   3.71707773]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 70,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

指数运算：

```{.python .input  n=47}
nd.exp(x)
```

```{.json .output n=47}
[
 {
  "data": {
   "text/plain": "\n[[ 2.71828175  2.71828175  2.71828175  2.71828175]\n [ 2.71828175  2.71828175  2.71828175  2.71828175]\n [ 2.71828175  2.71828175  2.71828175  2.71828175]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 47,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

也可以转置一个矩阵然后计算矩阵乘法：

```{.python .input  n=49}
nd.dot(x, y.T)
```

```{.json .output n=49}
[
 {
  "data": {
   "text/plain": "\n[[ -1.45954323  15.03398895  18.77302742]\n [ -1.45954323  15.03398895  18.77302742]\n [ -1.45954323  15.03398895  18.77302742]]\n<NDArray 3x3 @cpu(0)>"
  },
  "execution_count": 49,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 广播

当二元操作符左右两边ndarray形状不一样时，系统会尝试将其复制到一个共同的形状。例如`a`的第0维是3, `b`的第0维是1，那么`a+b`时会将`b`沿着第0维复制3遍：

```{.python .input  n=71}
a = nd.arange(3).reshape((3, 1))
b = nd.arange(2).reshape((1, 2))
print("a: {0}, b:{1}".format(a, b))
a + b

print("x: {0}, y: {1}".format(x, y))
y.reshape((3, 4))


```

```{.json .output n=71}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "a: \n[[ 0.]\n [ 1.]\n [ 2.]]\n<NDArray 3x1 @cpu(0)>, b:\n[[ 0.  1.]]\n<NDArray 1x2 @cpu(0)>\nx: \n[[ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]]\n<NDArray 3x4 @cpu(0)>, y: \n[[ -1.40078664   8.95064259   2.9670074   -6.01594448]\n [ 13.11195183  12.04055882   5.03590393  -9.71219349]\n [-11.89445019  -5.82562208  -5.50213766   3.71707773]]\n<NDArray 3x4 @cpu(0)>\n"
 },
 {
  "data": {
   "text/plain": "\n[[ -1.40078664   8.95064259   2.9670074   -6.01594448]\n [ 13.11195183  12.04055882   5.03590393  -9.71219349]\n [-11.89445019  -5.82562208  -5.50213766   3.71707773]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 71,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 跟NumPy的转换

ndarray可以很方便同numpy进行转换

```{.python .input  n=79}
import numpy as np
a = np.array((3, 4))
x = np.ones((2, 3))
y = nd.array(x)
z = y.asnumpy()
print(a.shape, x, x.shape, y, y.shape, z, z.shape)
```

```{.json .output n=79}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(2,) [[ 1.  1.  1.]\n [ 1.  1.  1.]] (2, 3) \n[[ 1.  1.  1.]\n [ 1.  1.  1.]]\n<NDArray 2x3 @cpu(0)> (2, 3) [[ 1.  1.  1.]\n [ 1.  1.  1.]] (2, 3)\n"
 }
]
```

## 替换操作
## 下面展示了“替换”和“原地操作”的区别

注意下列函数的功能和用法：

--- nd.random_normal(u, v, shape(m, n))   # 产生均值为u, 方差为v, 形状为mxn大小的随机矩阵

--- nd.zeros_like(x)  #  产生和x大小相同的zeros全零矩阵

在前面的样例中，我们为每个操作新开内存来存储它的结果。例如，如果我们写`y = x + y`, 我们会把`y`从现在指向的实例转到新建的实例上去。我们可以用Python的`id()`函数来看这个是怎么执行的：

```{.python .input  n=84}
x = nd.random_normal(0, 10, shape = (3, 5))
y = nd.ones((3, 5))
z = 0
print(id(z))
z = x + y
print(id(z))
print("x: {0}, shape: {1}".format(x, x.shape))
print("y: {0}, shape: {1}".format(y, y.shape))
```

```{.json .output n=84}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "1536682416\n2477605486720\nx: \n[[ -8.94258118   3.57444024   4.93839407   7.79328394  -9.04342651]\n [-10.10307312 -12.1407938   -3.91573071  21.56406403  13.16618729]\n [ 10.93822289  -4.32926273  18.27143288   7.15359879 -10.4467001 ]]\n<NDArray 3x5 @cpu(0)>, shape: (3, 5)\ny: \n[[ 1.  1.  1.  1.  1.]\n [ 1.  1.  1.  1.  1.]\n [ 1.  1.  1.  1.  1.]]\n<NDArray 3x5 @cpu(0)>, shape: (3, 5)\n"
 }
]
```

我们可以把结果通过`[:]`写到一个之前开好的数组里：

```{.python .input  n=92}
print(x)
z = nd.zeros_like(x)
print(id(z))
print(z)
# 此赋值操作为原地赋值，不新生成对象
z[:] = z + x + y
print(z)
id(z)
```

```{.json .output n=92}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ -8.94258118   3.57444024   4.93839407   7.79328394  -9.04342651]\n [-10.10307312 -12.1407938   -3.91573071  21.56406403  13.16618729]\n [ 10.93822289  -4.32926273  18.27143288   7.15359879 -10.4467001 ]]\n<NDArray 3x5 @cpu(0)>\n2477607638408\n\n[[ 0.  0.  0.  0.  0.]\n [ 0.  0.  0.  0.  0.]\n [ 0.  0.  0.  0.  0.]]\n<NDArray 3x5 @cpu(0)>\n\n[[ -7.94258118   4.57444      5.93839407   8.79328346  -8.04342651]\n [ -9.10307312 -11.1407938   -2.91573071  22.56406403  14.16618729]\n [ 11.93822289  -3.32926273  19.27143288   8.15359879  -9.4467001 ]]\n<NDArray 3x5 @cpu(0)>\n"
 },
 {
  "data": {
   "text/plain": "2477607638408"
  },
  "execution_count": 92,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

但是这里我们还是为`x+y`创建了临时空间，然后再复制到`z`。需要避免这个开销，我们可以使用操作符的全名版本中的`out`参数：

```{.python .input  n=95}
# 通过对比这里的ID（z）的结果和上面id(z)的结果可知，未生成新的变量
nd.elemwise_add(x, y, out=z)
id(z)
```

```{.json .output n=95}
[
 {
  "data": {
   "text/plain": "2477607638408"
  },
  "execution_count": 95,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

如果可以现有的数组之后不会再用，我们也可以用复制操作符达到这个目的：

```{.python .input  n=96}
before = id(x)
x += y
id(x) == before
```

```{.json .output n=96}
[
 {
  "data": {
   "text/plain": "True"
  },
  "execution_count": 96,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 总结

ndarray模块提供一系列多维数组操作函数。所有函数列表可以参见[NDArray API文档](https://mxnet.incubator.apache.org/api/python/ndarray.html)。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/745)
