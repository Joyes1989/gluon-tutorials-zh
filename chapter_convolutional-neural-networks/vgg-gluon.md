# VGG：使用重复元素的非常深的网络

我们从Alexnet看到网络的层数的激增。这个意味着即使是用Gluon手动写代码一层一层的堆每一层也很麻烦，更不用说从0开始了。幸运的是编程语言提供了很好的方法来解决这个问题：函数和循环。如果网络结构里面有大量重复结构，那么我们可以很紧凑来构造这些网络。第一个使用这种结构的深度网络是VGG。

## VGG架构

VGG的一个关键是使用很多有着相对小的kernel（$3\times 3$）的卷积层然后接上一个池化层，之后再将这个模块重复多次。下面我们先定义一个这样的块：

```{.python .input  n=1}
from mxnet.gluon import nn

def vgg_block(num_convs, channels):
    out = nn.Sequential()
    for _ in range(num_convs):
        out.add(
            nn.Conv2D(channels=channels, kernel_size=3,
                      padding=1, activation='relu')
        )
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out
```

```{.json .output n=1}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\h5py\\tests\\old\\test_attrs_data.py:251: DeprecationWarning: invalid escape sequence \\H\n  s = b\"Hello\\x00\\Hello\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\__init__.py:22: DeprecationWarning: invalid escape sequence \\.\n  module='^{0}\\.'.format(re.escape(__name__)))\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\scipy\\_lib\\_numpy_compat.py:287: DeprecationWarning: invalid escape sequence \\p\n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\externals\\joblib\\func_inspect.py:53: DeprecationWarning: invalid escape sequence \\<\n  '\\<doctest (.*\\.rst)\\[(.*)\\]\\>', source_file).groups()\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\externals\\joblib\\_memory_helpers.py:10: DeprecationWarning: invalid escape sequence \\s\n  cookie_re = re.compile(\"coding[:=]\\s*([-\\w.]+)\")\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\asn1crypto\\core.py:104: DeprecationWarning: invalid escape sequence \\d\n  _OID_RE = re.compile('^\\d+(\\.\\d+)*$')\n"
 }
]
```

我们实例化一个这样的块，里面有两个卷积层，每个卷积层输出通道是128：

```{.python .input  n=3}
from mxnet import nd

blk = vgg_block(2, 128)
blk.initialize()
x = nd.random.uniform(shape=(2,3,16,16))
y = blk(x)
y.shape
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "(2, 128, 8, 8)"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

可以看到经过一个这样的块后，长宽会减半，通道也会改变。

然后我们定义如何将这些块堆起来：

```{.python .input  n=4}
def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out
```

这里我们定义一个最简单的一个VGG结构，它有8个卷积层，和跟Alexnet一样的3个全连接层。这个网络又称VGG 11.

```{.python .input  n=7}
num_outputs = 10
architecture = ((1,64), (1,128), (2,256), (2,512), (2,512))
net = nn.Sequential()
# add name_scope on the outermost Sequential
with net.name_scope():
    net.add(vgg_stack(architecture))
    net.add(nn.Flatten())
    net.add(nn.Dense(4096, activation="relu"))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(4096, activation="relu"))
    net.add(nn.Dropout(.5))
    net.add(nn.Dense(num_outputs))

```

## 模型训练

这里跟Alexnet的训练代码一样除了我们只将图片扩大到$96\times 96$来节省些计算，和默认使用稍微大点的学习率。
**(alex中是默认使用了更小的学习率：0.01)**

```{.python .input}
import sys
sys.path.append('..')
import utils
from mxnet import gluon
from mxnet import init

train_data, test_data = utils.load_data_fashion_mnist(
    batch_size=64, resize=96)

ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 
                        'sgd', {'learning_rate': 0.05})
utils.train(train_data, test_data, net, loss,
            trainer, ctx, num_epochs=1)
```

```{.json .output n=None}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\cbook.py:550: DeprecationWarning: invalid escape sequence \\*\n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\cbook.py:1126: DeprecationWarning: invalid escape sequence \\S\n  _find_dedent_regex = re.compile(\"(?:(?:\\n\\r?)|^)( *)\\S\")\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\cbook.py:1961: DeprecationWarning: invalid escape sequence \\m\n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\pyparsing.py:131: DeprecationWarning: invalid escape sequence \\d\n  xmlcharref = Regex('&#\\d+;')\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\pyparsing.py:2127: DeprecationWarning: invalid escape sequence \\g\n  ret = re.sub(self.escCharReplacePattern,\"\\g<1>\",ret)\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\colors.py:265: DeprecationWarning: invalid escape sequence \\A\n  hexColorPattern = re.compile(\"\\A#[a-fA-F0-9]{6}\\Z\")\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\collections.py:442: DeprecationWarning: invalid escape sequence \\ \n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mlab.py:2210: DeprecationWarning: invalid escape sequence \\ \n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mlab.py:2892: DeprecationWarning: invalid escape sequence \\|\n  delete = set(\"\"\"~!@#$%^&*()-=+~\\|]}[{';: /?.>,<\"\"\")\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\font_manager.py:190: DeprecationWarning: invalid escape sequence \\S\n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\patches.py:515: DeprecationWarning: invalid escape sequence \\ \n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:78: DeprecationWarning: invalid escape sequence \\p\n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:442: DeprecationWarning: invalid escape sequence \\s\n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:474: DeprecationWarning: invalid escape sequence \\s\n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:756: DeprecationWarning: invalid escape sequence \\l\n  for alias, target in [('\\leftparen', '('),\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:758: DeprecationWarning: invalid escape sequence \\l\n  ('\\leftbrace', '{'),\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:760: DeprecationWarning: invalid escape sequence \\l\n  ('\\leftbracket', '['),\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:1041: DeprecationWarning: invalid escape sequence \\{\n  fixes = {'\\{': '{', '\\}': '}', '\\[': '[', '\\]': ']'}\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:1041: DeprecationWarning: invalid escape sequence \\}\n  fixes = {'\\{': '{', '\\}': '}', '\\[': '[', '\\]': ']'}\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:1041: DeprecationWarning: invalid escape sequence \\[\n  fixes = {'\\{': '{', '\\}': '}', '\\[': '[', '\\]': ']'}\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:1041: DeprecationWarning: invalid escape sequence \\]\n  fixes = {'\\{': '{', '\\}': '}', '\\[': '[', '\\]': ']'}\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:2432: DeprecationWarning: invalid escape sequence \\s\n  | Error(\"Expected \\sqrt{value}\"))\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:2437: DeprecationWarning: invalid escape sequence \\o\n  - (p.required_group | Error(\"Expected \\overline{value}\"))\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:2445: DeprecationWarning: invalid escape sequence \\o\n  | Error(\"Expected \\operatorname{value}\"))\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:2700: DeprecationWarning: invalid escape sequence \\c\n  r'AA' : (  ('it', 'A', 1.0), (None, '\\circ', 0.5), 0.0),\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:2922: DeprecationWarning: invalid escape sequence \\p\n  super.children.extend(self.symbol(s, loc, ['\\prime']))\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:3310: DeprecationWarning: invalid escape sequence \\s\n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:3339: DeprecationWarning: invalid escape sequence \\s\n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:3374: DeprecationWarning: invalid escape sequence \\s\n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\mathtext.py:3392: DeprecationWarning: invalid escape sequence \\s\n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\dviread.py:502: DeprecationWarning: invalid escape sequence \\*\n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\dviread.py:796: DeprecationWarning: invalid escape sequence \\*\n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\texmanager.py:35: DeprecationWarning: invalid escape sequence \\*\n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\style\\core.py:35: DeprecationWarning: invalid escape sequence \\S\n  STYLE_FILE_PATTERN = re.compile('([\\S]+).%s$' % STYLE_EXTENSION)\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\axes\\_axes.py:6722: DeprecationWarning: invalid escape sequence \\l\n  \"\"\"\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\quiver.py:175: DeprecationWarning: invalid escape sequence \\s\n  \"\"\" % docstring.interpd.params\nC:\\ProgramData\\Anaconda3\\lib\\site-packages\\matplotlib\\quiver.py:885: DeprecationWarning: invalid escape sequence \\ \n  \"\"\" % docstring.interpd.params\n"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Start training on  gpu(0)\n"
 }
]
```

## 总结

通过使用重复的元素，我们可以通过循环和函数来定义模型。使用不同的配置(`architecture`)可以得到一系列不同的模型。


## 练习

- 尝试多跑几轮，看看跟LeNet/Alexnet比怎么样？
- 尝试下构造VGG其他常用模型，例如VGG16， VGG19. （提示：可以参考[VGG论文](https://arxiv.org/abs/1409.1556)里的表1。）
- 把图片从默认的$224\times 224$降到$96\times 96$有什么影响？


**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/1277)
