import tensorflow as tf
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from itertools import chain
import gensim
import re
import nltk
import tempfile


# Define all the helper functions
def iter_files(doc_tree):
    for doc in doc_tree.iterfind('./file'):
        raw_keywords = doc.find('head').find('keywords').text
        keywords = raw_keywords.split(", ")
        file_dict = {}
        file_dict['label'] = keywords_to_label(keywords)
        file_dict['text'] = text_to_vec(doc.find('content').text)
        yield file_dict


def get_all_text(tree):
    elements = tree.iterfind('./file/content')
    for element in elements:
        yield element.text


def text_tokenize(contents):
    tokens = clean_text(contents)
    print (tokens)
    return tokens


def get_vocab(doc_tree):
    input_text = '\n'.join(get_all_text(doc_tree))
    modelVocab = gensim.models.Word2Vec()
    tokens = nltk.sent_tokenize(input_text)
    sentences = []
    for token in tokens:
        sentences.extend(token.split())
    print(sentences[:10])
    modelVocab.build_vocab(sentences)
    model = gensim.models.Word2Vec(sentences, size=50)
    model.save("w2vmodel")
    return modelVocab


# Converts text to a vector
def text_to_vec(text):
    words = nltk.word_tokenize(text)
    return embed_text(words)


# Converts keywords to a label and then to a one-hot encoding
def keywords_to_label(keywords):
    string = ['o', 'o', 'o']
    if "technology" in keywords:
        string[0] = "T"
    if "entertainment" in keywords:
        string[1] = "E"
    if "design" in keywords:
        string[2] = "D"
    label = ''.join(string)
    return encode_label(label)


def encode_label(label):
    vector = [0, 0, 0, 0, 0, 0, 0, 0]
    vector[labels_dict[label]] = 1
    return np.array(vector)

labels_dict = {
        "ooo": 0,
        "Too": 1,
        "oEo": 2,
        "ooD": 3,
        "TEo": 4,
        "ToD": 5,
        "oED": 6,
        "TED": 7
        }


def input_fn(df):
    label = df["label"].values
    text_values = df["text"].values
    return text_values, label


# converts text into a list of words
# removes parenthesised expressions, names before a colon
def clean_text(text):
    text_noparens = re.sub(r'\([^)]*\)', '', text)
    sentences = []
    for line in text_noparens.split('\n'):
        m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
        sentences.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
    words = chain(*sentences)
    return words


def embed_text(text):
    vectors_iterator = [w2v[w.lower()] if w in w2v else w2v["UNK"] for w in text]
    mean = np.mean(vectors_iterator, axis=0)
    return mean

# Define classifier model

steps = 100000      # Number of times to run training step
batch_size = 50
dims = 50           # Dimensionality of vocabulary
classes = 8         # Dimensionality of classes
hidden = 100        # Hidden features
x = tf.placeholder(tf.float32, shape=[None, dims])
y = tf.placeholder(tf.float32, shape=[None, classes])

W = tf.Variable(tf.random_normal([dims, hidden]))
b = tf.Variable(tf.random_normal([hidden]))

h = tf.nn.tanh(tf.matmul(x, W) + b)

V = tf.Variable(tf.random_normal([hidden, classes]))
c = tf.Variable(tf.random_normal([classes]))

u = tf.matmul(h, V) + c
p = tf.nn.softmax(u)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=u))

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Define alternate model
def ffnn_model_fn(text, labels, mode):
    features = text
    input_layer = tf.reshape(features, [-1, 50, 1, 1])
    dense_layer = tf.layers.dense(inputs=input_layer, units=hidden, activation=tf.nn.tanh)
    logits = tf.layers.dense(inputs=dense_layer, units=classes, activation=tf.nn.tanh)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        learning_rate=1e-4,
        optimizer="Adam",
        global_step=tf.contrib.get_global_step()
    )

    predictions = {
        "classes": tf.argmax(
            input=logits,
            axis=1
        ),
        "probabilities": tf.nn.softmax(
            logits,
            name="softmax_tensor"
        )
    }

    return tf.contrib.learn.python.learn.estimators.model_fn.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op
    )

ted_classifier = tf.contrib.learn.Estimator(
    model_fn=ffnn_model_fn, model_dir="./tmp"
)

# Import
print ("Importing vocabulary")
lines = open("glove.6B.50d.txt", "r")
w2v = {line.split()[0]: [float(x) for x in line.split()[1:]] for line in lines}
w2v["UNK"] = np.zeros(dims)

print("Importing and converting text")
tree = ET.parse('./ted_en-20160408.xml')
doc_df = pd.DataFrame(list(iter_files(tree)))

doc_train = doc_df.ix[:1584, :]
doc_validation = doc_df.ix[1585:1835, :]
doc_test = doc_df.ix[-250:, :]

# Runs tensorflow
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print("Begin tensorflow session")


def train_input_fn():
    return input_fn(doc_train)


def test_input_fn():
    return input_fn(doc_test)


def original_session():
    for i in range(steps):
        batch = doc_train.sample(n=batch_size, replace=True)
        #    batch = doc_train[(i*batch_size):((i+1)*batch_size)]
        text, labels = input_fn(batch)
        # batch_text = tf.train.batch(batch["text"].values, batch_size)
        # print (batch_text.shape)
        # batch_label = tf.train.batch(batch["label"].values, batch_size)
        # print(batch_label.shape)
        # assert all(x.shape == (None, dims) for x in batch["text"])
        # assert all(y.shape == (None, classes) for y in batch["label"])
        if i % 100 == 0:
            print("Run training step number", i)
        sess.run(train_step, feed_dict={x: text, y: labels})
    test_text, test_labels = input_fn(doc_test)
    train_text, train_labels = input_fn(doc_train)
    print ("Training accuracy: ", sess.run(accuracy, feed_dict={x: train_text, y: train_labels}))
    print ("Testing accuracy:  ", sess.run(accuracy, feed_dict={x: test_text, y: test_labels}))


def ffnn():
    doc_train_text, doc_train_labels = input_fn(doc_train)
    eval_text, eval_labels = input_fn(doc_test)
    tensor_train_text = tf.constant(list(doc_train_text))
    ted_classifier.fit(
        x=tensor_train_text,
        y=doc_train_labels,
        batch_size=50,
        steps=100
    )
    metrics = {
        "accuracy":
            tf.learn.metric_spec.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes"
            )
    }
    eval_results = ted_classifier.evaluate(
        x=eval_text, y=eval_labels, metrics=metrics
    )
    print (eval_results)

ffnn()
#new_session()
#originalSession()