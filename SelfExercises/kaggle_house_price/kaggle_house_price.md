## 1/2. 导入相关库 + 数据

```{.python .input  n=2}
# 需要使用pandas导入数据，numpy执行处理
import pandas as pd
import numpy as np

# 加载训练集和测试集
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# csv数据集中真正有效的特征是MSSubClass: MSSubClass之间内的特征
all_X = pd.concat((train.loc[:, "MSSubClass":"SaleCondition"], 
                  test.loc[:, "MSSubClass": "SaleCondition"]))
```

查看各个数据的格式以及大小信息：

```{.python .input}
train.head()
```

```{.python .input}
all_X.head()
```

```{.python .input}
print(train.shape, test.shape, all_X.shape)
```

## 2/3. 数据预处理
使用pandas对数值特征进行归一化（标准化）处理：

$$ x_i = \frac{x_i - \mathbb{E}x_i} {\text{std}(x_i)} $$

```{.python .input}

print("all_X.dtypes 数据类型: ", type(all_X.dtypes), "\n ", all_X.dtypes, all_X.dtypes != "object")

```

```{.python .input}
# pandas.dtypes 用于查看pandas数据对应的类型信息
# 获取all_X.dtypes中类型为非object类型的feture对应的index列表
numeric_feats = all_X.dtypes[all_X.dtypes != "object"].index
print(all_X.head())
# 针对all_X中的数值元素进行归一化操作，对象类型元素保持不变
all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: (x - x.mean()) / (x.std()) )
print(numeric_feats, all_X.head())
```

#### 把离散数据点转换成数值标签。（对象元素转成one-hot编码）

```{.python .input}
# 对all_X中的对象元素进行One_Hot编码
all_X = pd.get_dummies(all_X, dummy_na=True)
all_X.head()
```

#### 把缺失数据用本特征的平均值估计。

```{.python .input}
all_X = all_X.fillna(all_X.mean())
all_X.head()
```

```{.python .input}
num_train = train.shape[0]
print(train.head(), train.MSSubClass.head())

X_train = all_X[: num_train].as_matrix()
X_test = all_X[num_train: ].as_matrix()

# train 是pandas.core.frame.DataFrame类型，
# train中的每个字段都可以作为train的一个成员直接访问，如train.SalePrice，train.MSSubClass
y_train = train.SalePrice.as_matrix()
print(type(train), train.SalePrice.head())
```

# 通过NDArray方式导入数据-For Gluon
#### Gluon可支持NDArray格式的数据
#### 直接将上面加载好的pandas格式的数据转换成ndarray格式

```{.python .input}
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

X_train = nd.array(X_train)
y_train = nd.array(y_train)
y_train.reshape((num_train, 1))

X_test = nd.array(X_test)
print(type(X_train), type(y_train), type(X_test))
```

#### 定义平方误差损失函数

```{.python .input}
square_loss = gluon.loss.L2Loss()
```

##### 定义算法结果的效果衡量算法 
###### --- 算法最终的衡量指标为RMSE（root-mean-square-error均方根误差）

```{.python .input}
def get_rmse_log(net, X_train, y_train):
    num_train = y_train.shape[0]
    # Return an array whose values are limited to [min, max]. One of max or min must be given.
    clipped_preds = nd.clip(net(X_train), 1, float('inf'))
    ret = np.sqrt(2 * nd.sum(square_loss(nd.log(clipped_preds), nd.log(y_train))).asscalar() / num_train)
    return ret
    
```

### 定义算法的模型
将模型的定义放在一个函数里,供后续多次调用. 这里使用基本的线性回归模型:

```{.python .input}
def get_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        # Dense 层就是全连接层
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net
```

定义一个用于执行训练过程的函数，整合整个训练过程，可用于跑不同的实验：

```{.python .input}
%matplotlib inline 
# 设置matplot的画图结果显示在本页面内，而不是显示在弹出窗口中

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt

def train(net, X_train, y_train, X_test, y_test, epochs, verbose_epoch, learning_rate, weight_decay):
    train_loss = [] # 保存训练的每个阶段对应的Loss损失值函数值
    if X_test is not None:
        test_loss = []
    batch_size = 100
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    # 利用gluon中的数据加载器，每次从dataset_train中获取batch_size大小的数据用于训练
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
    ## 构造训练器，adam 为激活函数的一种，从下面链接可查看gluon支持的所有激活函数
    # https://mxnet.incubator.apache.org/api/python/optimization/optimization.html#the-mxnet-optimizer-package
    trainer = gluon.Trainer(net.collect_params(), 
                            'adam', 
                            {'learning_rate': learning_rate, 'wd': weight_decay})
    # 重新初始化参数和梯度信息，避免多次调用train函数之间产生影响
    # 具体参数用法： https://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=force_reinit
    net.collect_params().initialize(force_reinit=True)
    for epoch in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            # 更新参数列表， 传入的batch_size会被用来进行归一化
            # https://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=step#mxnet.gluon.Trainer.step
            trainer.step(batch_size)
            
            cur_train_loss = get_rmse_log(net, X_train, y_train)
        if epoch > verbose_epoch:
            print(type(epoch), type(cur_train_loss))
            print("Epoch %d, train loss: %f" % (epoch, cur_train_loss))
        train_loss.append(cur_train_loss)
        if X_test is not None:
            cur_test_loss = get_rmse_log(net, X_test, y_test)
            test_loss.append(cur_test_loss)
    plt.plot(train_loss)
    plt.legend(['train'])
    if X_test is not None:
        plt.plot(test_loss)
        plt.legend(['train', 'test'])
    plt.show()
    if X_test is not None:
        return cur_train_loss, cur_test_loss
    else:
        return cur_train_loss
    
```

## 实现K折交叉验证：
为了避免[过拟合](underfit-overfit.md)，不过度依赖训练集的误差来推断测试集的误差，我们使用K折交叉验证的方式：
> 在K折交叉验证中，我们将初始采样数据集分割成$K$个子样本，一个单独的子样本被留作为**验证集**，其他$K-1$个样本集为**测试集**。

我们使用K次交叉验证的测试结果的平均值来作为模型的误差，降低模型的过拟合情况：

```{.python .input}
def k_fold_cross_valid(k, epochs, verbose_epoch, X_train, y_train, learning_rate, weight_decay):
    # 断言
    assert k > 1
    # // 取整除操作符，返回商结果中的整数部分
    fold_size = X_train.shape[0] // k  
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    # k 折数据是事先分好的，在整个训练过程中是不变的，改变的是每次选哪些用来训练和测试
    for test_i in range(k):
        # 获取本次用来测试的数据:（test_i * fold_size : (test_i + 1) * fold_size）之间的fold_size大小的数据
        X_val_test = X_train[test_i * fold_size: (test_i + 1) * fold_size, :]
        y_val_test = y_train[test_i * fold_size: (test_i + 1) * fold_size]
        
        val_train_defined = False
        # 除去上面用来测试的数据后，其余的数据都是训练数据,拼接在一起
        for i in range(k):
            if i != test_i:
                X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    # 将剩余的询量数据拼接起来，构成总的训练集数据
                    print(X_val_train.shape, type(X_val_train), X_cur_fold.shape, type(X_cur_fold))
                    print(X_train.shape, type(X_train))
                    X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
        net = get_net()
        train_loss, test_loss = train(net, X_val_train, y_val_train, X_val_test, y_val_test, 
                                     epochs, verbose_epoch, learning_rate, weight_decay)
        train_loss_sum += train_loss
        print ("Test Loss: %f" % test_loss)
        test_loss_sum += test_loss
    return train_loss_sum / k, test_loss_sum / k

```

训练模型并进行交叉验证，并不断调参优化模型效果

```{.python .input}
k = 7
epochs = 100
verbose_epoch = 95
learning_rate = 25
weight_decay = 0.0
```

基于上述既定参数，进行训练兵验证模型效果：

```{.python .input}
train_loss, test_loss = k_fold_cross_valid(k, epochs, verbose_epoch, 
                                           X_train, y_train, learning_rate, weight_decay)
print("%d-fold validation: Avg train loss: %f, Avg test loss: %f" %(k, train_loss, test_loss))
```

#### 调参注意事项：
（1） 即使训练误差较小时，也可能在K折交叉验证上误差较高
（2） 当训练误差较小时，观察K折交叉验证的误差是否同时降低并小心过拟合
（3） 应当以K折交叉验证最终的验证误差来反馈调节参数

### 用训练好的模型进行预测，并提交结果到Kaggle
首先定义预测函数

```{.python .input}
def learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate, weight_decay):
    net = get_net()
    # 训练模型
    train(net, X_train, y_train, None, None, epochs, verbose_epoch, learning_rate, weight_decay)
    # 利用模型训练好的参数执行预测
    preds = net(X_test).asnumpy()
    # pd - pandas
    test["SalePrice"] = pd.Series(preds.reshape(1, -1)[0])
    
    # 按格式生成结果文件，并保存
    submission = pd.concat([test['Id'], test['SalePrice']], axis = 1)
    submission.to_csv('submission.csv', index = False)
```

执行预测

```{.python .input}
learn(epochs, verbose_epoch, X_train, y_train, test, learning_rate, weight_decay)
```

将上述步骤生成的submission.csv文件提交的kaggle网站即可

### 不断优化调参，提交到Kaggle，争取更好的排名：
（1） 可以在k折交叉验证中我们发现TestError高于trainError，因此可以增加weight_decay的值避免过拟合 (后来发现在Adam算法中不适用)

（2） 考虑更换其他的梯度下降函数来代替上面的Adam函数（上述结果中，效果总是先好，随着迭代变差）

### 如何挑选合适的激活函数以及梯度下降优化方法：
[Ref-1] https://blog.csdn.net/u014381600/article/details/72867109

[Ref-2] https://blog.csdn.net/u014595019/article/details/52562159

（1） **激活函数的作用**：
激活函数的主要作用是提供网络的非线性建模能力。如果没有激活函数，那么该网络仅能够表达线性映射，此时即便有再多的隐藏层，其整个网络跟单层神经网络也是等价的。因此也可以认为，只有加入了激活函数之后，深度神经网络才具备了分层的非线性映射学习能力。 那么激活函数应该具有什么样的性质呢？

（2） **梯度下降优化方法**：使用何种优化方法来执行梯度下降过程：

**1. SGD梯度下降法:**
SGD全名 stochastic gradient descent， 即随机梯度下降。不过这里的SGD其实跟MBGD(minibatch gradient descent)是一个意思,即随机抽取一批样本,以此为根据来更新参数.

**2. Adam优化方法:**
Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。

**3. 优化方法效果对比**

1：用相同数量的超参数来调参，SGD和SGD +momentum 方法性能在测试集上的额误差好于所有的自适应优化算法，尽管有时自适应优化算法在训练集上的loss更小，但是他们在测试集上的loss却依然比SGD方法高，

2：自适应优化算法 在训练前期阶段在训练集上收敛的更快，但是在测试集上这种有点遇到了瓶颈。

3：所有方法需要的迭代次数相同，这就和约定俗成的默认自适应优化算法 需要更少的迭代次数的结论相悖！


### 下面实现基于SGD梯度下降函数的询量模型：

#### 1/2: 搭建基于SGD优化方法的模型训练器 以及 搭建更复杂的网络结构

```{.python .input}
def get_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(256, activation="relu"))
        # Dense 层就是全连接层
        net.add(gluon.nn.Dense(1))
    net.initialize()
    return net
```

```{.python .input}
%matplotlib inline 
# 设置matplot的画图结果显示在本页面内，而不是显示在弹出窗口中

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt

def train_sgd(net, X_train, y_train, X_test, y_test, epochs, verbose_epoch, learning_rate, weight_decay, momentum):
    train_loss = [] # 保存训练的每个阶段对应的Loss损失值函数值
    if X_test is not None:
        test_loss = []
    batch_size = 100
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    # 利用gluon中的数据加载器，每次从dataset_train中获取batch_size大小的数据用于训练
    data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
    ## 构造训练器，adam 为激活函数的一种，从下面链接可查看gluon支持的所有激活函数
    # https://mxnet.incubator.apache.org/api/python/optimization/optimization.html#the-mxnet-optimizer-package
    trainer = gluon.Trainer(net.collect_params(), 
                            'sgd', 
                            {'learning_rate': learning_rate, 'wd': weight_decay, 'momentum': momentum})
    # 重新初始化参数和梯度信息，避免多次调用train函数之间产生影响
    # 具体参数用法： https://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=force_reinit
    net.collect_params().initialize(force_reinit=True)
    for epoch in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            # 更新参数列表， 传入的batch_size会被用来进行归一化
            # https://mxnet.incubator.apache.org/api/python/gluon/gluon.html?highlight=step#mxnet.gluon.Trainer.step
            trainer.step(batch_size)
            
            cur_train_loss = get_rmse_log(net, X_train, y_train)
        if epoch > verbose_epoch:
#             print(type(epoch), type(cur_train_loss))
            print("Epoch %d, train loss: %f" % (epoch, cur_train_loss))
        train_loss.append(cur_train_loss)
        if X_test is not None:
            cur_test_loss = get_rmse_log(net, X_test, y_test)
            test_loss.append(cur_test_loss)
    plt.plot(train_loss)
    plt.legend(['train'])
    if X_test is not None:
        plt.plot(test_loss)
        plt.legend(['train', 'test'])
    plt.show()
    if X_test is not None:
        return cur_train_loss, cur_test_loss
    else:
        return cur_train_loss
    
```

#### 2/3: 基于SGD优化方法的模型训练器实现K折训练过程:

```{.python .input}
def k_fold_cross_valid_sgd(k, epochs, verbose_epoch, X_train, y_train, learning_rate, weight_decay, momentum):
    # 断言
    assert k > 1
    # // 取整除操作符，返回商结果中的整数部分
    fold_size = X_train.shape[0] // k  
    train_loss_sum = 0.0
    test_loss_sum = 0.0
    # k 折数据是事先分好的，在整个训练过程中是不变的，改变的是每次选哪些用来训练和测试
    for test_i in range(k):
        # 获取本次用来测试的数据:（test_i * fold_size : (test_i + 1) * fold_size）之间的fold_size大小的数据
        X_val_test = X_train[test_i * fold_size: (test_i + 1) * fold_size, :]
        y_val_test = y_train[test_i * fold_size: (test_i + 1) * fold_size]
        
        val_train_defined = False
        # 除去上面用来测试的数据后，其余的数据都是训练数据,拼接在一起
        for i in range(k):
            if i != test_i:
                X_cur_fold = X_train[i * fold_size: (i + 1) * fold_size, :]
                y_cur_fold = y_train[i * fold_size: (i + 1) * fold_size]
                if not val_train_defined:
                    X_val_train = X_cur_fold
                    y_val_train = y_cur_fold
                    val_train_defined = True
                else:
                    # 将剩余的询量数据拼接起来，构成总的训练集数据
#                     print(X_val_train.shape, type(X_val_train), X_cur_fold.shape, type(X_cur_fold))
#                     print(X_train.shape, type(X_train))
                    X_val_train = nd.concat(X_val_train, X_cur_fold, dim=0)
                    y_val_train = nd.concat(y_val_train, y_cur_fold, dim=0)
        net = get_net()
        train_loss, test_loss = train_sgd(net, X_val_train, y_val_train, X_val_test, y_val_test, 
                                     epochs, verbose_epoch, learning_rate, weight_decay, momentum)
        train_loss_sum += train_loss
        print ("Test Loss: %f" % test_loss)
        test_loss_sum += test_loss
    return train_loss_sum / k, test_loss_sum / k

```

#### 3/4: 初始化网络的各类参数

```{.python .input}
k = 3
epochs = 60
verbose_epoch = 95
learning_rate = 0.0000008
weight_decay = 1000
momentum = 0.1

train_loss, test_loss = k_fold_cross_valid_sgd(k, epochs, verbose_epoch, 
                                           X_train, y_train, learning_rate, weight_decay, momentum)
print("%d-fold validation: Avg train loss: %f, Avg test loss: %f" %(k, train_loss, test_loss))
```

#### 4/5: 开始执行训练过程:

```{.python .input}
def learn_sgd(epochs, verbose_epoch, X_train, y_train, test, learning_rate, weight_decay, momentum):
    net = get_net()
    # 训练模型
    train_sgd(net, X_train, y_train, None, None, epochs, verbose_epoch, learning_rate, weight_decay, momentum)
    # 利用模型训练好的参数执行预测
    preds = net(X_test).asnumpy()
    # pd - pandas
    test["SalePrice"] = pd.Series(preds.reshape(1, -1)[0])
    
    # 按格式生成结果文件，并保存
    submission = pd.concat([test['Id'], test['SalePrice']], axis = 1)
    submission.to_csv('submission.csv', index = False)

learn_sgd(epochs, verbose_epoch, X_train, y_train, test, learning_rate, weight_decay, momentum)
```
