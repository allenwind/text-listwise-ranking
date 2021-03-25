import random
import itertools
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split

from attentionpooling import MaskGlobalMaxPooling1D
from tensorflow.keras.preprocessing import sequence
from dataset import SimpleTokenizer, find_best_maxlen
from dataset import load_THUCNews_title_label

# 把分类问题当做listwise标签匹配的问题
# 0.9232

# 来自Transformer的激活函数，效果略有提升

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

def pad(x, maxlen):
    x = sequence.pad_sequences(
        x, 
        maxlen=maxlen,
        dtype="int32",
        padding="post",
        truncating="post",
        value=0
    )
    return x

def batch_pad(x):
    maxlen = max([len(i) for i in x])
    return pad(x, maxlen)

class PositionEmbedding(tf.keras.layers.Layer):
    """可学习的位置Embedding"""

    def __init__(self, maxlen, output_dim, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.output_dim = output_dim
        self.embedding = tf.keras.layers.Embedding(
            input_dim=maxlen,
            output_dim=output_dim
        )

    def call(self, inputs):
        # maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = tf.expand_dims(positions, axis=0)
        return self.embedding(positions)

    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)

def convert_to_listwise(X, y, classes, eps=0.2):
    id2label = {j:i for i,j in classes.items()}
    classes = list(classes)
    for x, label in zip(X, y):
        ls = classes[:]
        random.shuffle(ls)
        labelid = ls.index(id2label[label])

        yield (x, *ls), labelid

def compute_label(size, idx, eps=0.2):
    eps = np.random.uniform(low=0.0, high=eps)
    label = np.random.uniform(size=size)
    label[idx] = 0
    label = label * eps / np.sum(label)
    label[idx] = 1 - eps
    return label

class ListwiseGenerator(tf.keras.utils.Sequence):

    def __init__(self, X, y, classes, tokenizer, batch_size, eps=0.05):
        self.X = X
        self.y = y
        self.id2label = {j:i for i,j in classes.items()}
        self.size = len(classes)
        self.classes = list(classes)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.eps = eps


    def __getitem__(self, idx):
        i = idx * self.batch_size
        j = (idx + 1) * self.batch_size
        X = self.X[i:j]
        labels = self.y[i:j]
        batch_inputs = [[] for _ in range(self.size+1)]
        batch_labels = []
        for x, y in zip(X, labels):
            inputs, label = self.generate_sample(x, y)

            for i, j in enumerate(inputs):
                batch_inputs[i].append(j)

            batch_labels.append(label)

        batch_inputs = [batch_pad(i) for i in batch_inputs]
        batch_labels = np.array(batch_labels)
        return batch_inputs, batch_labels

    def __len__(self):
        return len(self.X) // self.batch_size

    def on_epoch_end(self):
        np.random.RandomState(773).shuffle(self.X)
        np.random.RandomState(773).shuffle(self.y)

    def compute_label(self, idx):
        if self.eps == 0:
            label = np.ones(self.size)
            label[idx] = 1
            return label
        eps = np.random.uniform(low=0.0, high=self.eps)
        label = np.random.uniform(size=self.size)
        label[idx] = 0
        label = label * eps / np.sum(label)
        label[idx] = 1 - eps
        return label

    def generate_sample(self, x, y):
        X = []
        label = self.id2label[y]
        ls = self.classes[:]
        random.shuffle(ls)
        idx = ls.index(label)
        label = self.compute_label(idx)
        X.append(self.tokenizer.encode(x))
        for l in ls:
            X.append(self.tokenizer.encode(l))
        return X, label


X, y, classes = load_THUCNews_title_label()

i = int(len(X) * 0.8)

X1 = X[:i]
y1 = y[:i]

X2 = X[i:]
y2 = y[i:]

tokenizer = SimpleTokenizer()
tokenizer.fit(X + list(classes))

gen = ListwiseGenerator(X1, y1, classes, tokenizer, batch_size=32)
gen_val = ListwiseGenerator(X2, y2, classes, tokenizer, batch_size=32)

maxlen = 48
emaxlen = 2 # 匹配标签的长度
hdims = 128
embedding_dims = 128
num_words = len(tokenizer)
num_classes = len(classes)

# 文本输入 + 标签输入
inputs = [Input(shape=(None,))]
for i in range(num_classes):
    inputs.append(Input(shape=(None,)))

# 计算各输入mask
masks = [Lambda(lambda x: tf.not_equal(x, 0))(x) for x in inputs]

embedding = Embedding(
    num_words,
    embedding_dims,
    embeddings_initializer="glorot_normal",
)
posembedding = PositionEmbedding(maxlen, embedding_dims)
layernom = LayerNormalization()
dropout1 = Dropout(0.1)
conv1 = Conv1D(filters=hdims, kernel_size=2, padding="same", activation=gelu)
conv2 = Conv1D(filters=hdims, kernel_size=2, padding="same", activation=gelu)
conv3 = Conv1D(filters=hdims, kernel_size=2, padding="same", activation=gelu)
pool = MaskGlobalMaxPooling1D()

def encode(x, mask):
    x = embedding(x)
    x = layernom(x)
    # x = dropout1(x)
    x = conv1(x) + x
    x = conv2(x) + x
    x = conv3(x) + x
    x = pool(x, mask=mask)
    return x

class Listwise(tf.keras.layers.Layer):

    def __init__(self, sim_func, **kwargs):
        super(Listwise, self).__init__(**kwargs)
        self.sim_func = sim_func

    def call(self, inputs):
        x, *es = inputs
        sims = []
        for e in es:
            sims.append(self.sim_func(x, e))
        sims = tf.concat(sims, axis=-1)
        sims = tf.math.softmax(sims, axis=-1)
        return sims


dropout2 = Dropout(0.2)
d1 = Dense(4 * hdims)
d2 = Dense(1)
def sim(x1, x2):
    # x*y
    x3 = Multiply()([x1, x2])
    # |x-y|
    x4 = Lambda(lambda x: tf.abs(x[0] - x[1]))([x1, x2])
    x = Concatenate()([x1, x2, x3, x4])
    x = d1(x)
    x = d2(x)
    return x

xs = []
for x, mask in zip(inputs, masks):
    x = encode(x, mask)
    xs.append(x)

outputs = Listwise(sim)(xs)

model = Model(inputs, outputs)
model.summary()
model.compile(
    loss="kl_divergence", 
    optimizer="adam", 
    metrics=["accuracy"]
)

model.fit(
    gen,
    epochs=10,
    validation_data=gen_val,
    validation_batch_size=32,
)
