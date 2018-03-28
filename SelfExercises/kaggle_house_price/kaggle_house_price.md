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
k = 5
epochs = 100
verbose_epoch = 95
learning_rate = 5
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
