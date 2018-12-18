# 使用NDArray来处理数据

对于机器学习来说，处理数据往往是万事之开头。它包含两个部分：数据读取和当数据已经在内存里时如何处理。本章将关注后者。我们首先介绍`NDArray`，这是MXNet储存和变换数据的主要工具。如果你之前用过`NumPy`，你会发现`NDArray`和`NumPy`的多维数组非常类似。当然，`NDArray`提供更多的功能，首先是CPU和GPU的异步计算，其次是自动求导。这两点使得`NDArray`能更好地支持机器学习。

## 让我们开始

我们先介绍最基本的功能。如果你不懂我们用到的数学操作也不用担心，例如按元素加法，或者正态分布，我们会在之后的章节分别详细介绍。

我们首先从`mxnet`导入`ndarray`这个包

```{.python .input  n=1}
from mxnet import ndarray as nd
```

然后我们创建一个有3行和4列的2D数组（通常也叫矩阵），并且把每个元素初始化成0

```{.python .input  n=2}
nd.zeros((3, 4))
```

类似的，我们可以创建数组每个元素被初始化成1。

```{.python .input  n=3}
x = nd.ones((3, 4))
print(x, x)
```

或者从python的数组直接构造

```{.python .input  n=4}
nd.array([[1, 2], [3, 5]])
```

我们经常需要创建随机数组，就是说每个元素的值都是随机采样而来，这个经常被用来初始化模型参数。下面创建数组，它的元素服从均值0方差1的正态分布。

```{.python .input  n=5}
y = nd.random_normal(0, 1, (3, 3))
y
```

跟`NumPy`一样，每个数组的形状可以通过`.shape`来获取

```{.python .input  n=6}
print(x.shape, y.shape)
```

它的大小，就是总元素个数，是形状的累乘。

```{.python .input  n=7}
print(x.size, y.size)
```

## 操作符

NDArray支持大量的数学操作符，例如按元素加法：

```{.python .input  n=8}
print("x.shape: {0}, x: {1}".format(x.shape, x))
print("y.shape: {0}, y: {1}".format(y.shape, y))
y = nd.random_normal(0, 10, shape = (3, 4))
print("y.shape: {0}, y: {1}".format(y.shape, y))
z = x + y
print("z = x + y, z: {0}, z.shape: {1}".format(z, z.shape))
```

乘法：

```{.python .input  n=9}
print(x.shape, y.shape)
# 矩阵中的对应元素相乘
x * y
```

指数运算：

```{.python .input  n=10}
nd.exp(x)
```

也可以转置一个矩阵然后计算矩阵乘法：

```{.python .input  n=11}
nd.dot(x, y.T)
```

## 广播

当二元操作符左右两边ndarray形状不一样时，系统会尝试将其复制到一个共同的形状。例如`a`的第0维是3, `b`的第0维是1，那么`a+b`时会将`b`沿着第0维复制3遍：

```{.python .input  n=13}
a = nd.arange(3).reshape((3, 1))
b = nd.arange(2).reshape((1, 2))
print("a: {0}, b:{1}".format(a, b))
c = a + b
print("c.shape: {0}".format(c.shape))

print("x: {0}, y: {1}".format(x, y))
y.reshape((3, 4))


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

但是这里我们还是为`x+y`创建了临时空间，然后再复制到`z`。需要避免这个开销，我们可以使用操作符的全名版本中的`out`参数：

```{.python .input  n=95}
# 通过对比这里的ID（z）的结果和上面id(z)的结果可知，未生成新的变量
nd.elemwise_add(x, y, out=z)
id(z)
```

如果可以现有的数组之后不会再用，我们也可以用复制操作符达到这个目的：

```{.python .input  n=96}
before = id(x)
x += y
id(x) == before
```

## 总结

ndarray模块提供一系列多维数组操作函数。所有函数列表可以参见[NDArray API文档](https://mxnet.incubator.apache.org/api/python/ndarray.html)。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/745)
