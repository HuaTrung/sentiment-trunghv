import pandas as pd
import numpy as np
import re
import string
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def load_data_and_labels(data_path):
    data=np.array(pd.read_csv(data_path))
    # Split by pos and neg for train
    data_neg=[]
    data_pos=[]
    for row in data[0:1000]:
        if row[-1]==1:
            data_pos.append(clean_str(row[1]))
        else:
            data_neg.append(clean_str(row[1]))
    x_train = data_pos + data_neg
    positive_labels = [1 for _ in data_pos]
    negative_labels = [0 for _ in data_neg]
    y_train = np.concatenate([positive_labels, negative_labels], 0)

    # Split by pos and neg for test
    data_neg_test=[]
    data_pos_test=[]
    for row in data[2000:2500]:
        if row[-1]==1:
            data_pos_test.append(clean_str(row[1]))
        else:
            data_neg_test.append(clean_str(row[1]))
    x_test = data_pos_test + data_neg_test
    positive_labels_test = [1 for _ in data_pos_test]
    negative_labels_test = [0 for _ in data_neg_test]
    y_test = np.concatenate([positive_labels_test, negative_labels_test], 0)
    return x_train,y_train,x_test,y_test

def clean_str(target):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    target = re.sub(r"\'s", " \'s", target)
    target = re.sub(r"\'ve", " \'ve", target)
    target = re.sub(r"n\'t", " n\'t", target)
    target = re.sub(r"\'re", " \'re", target)
    target = re.sub(r"\'d", " \'d", target)
    target = re.sub(r"\'ll", " \'ll", target)
    target = re.sub(r",", " , ", target)
    target = re.sub(r"!", " ! ", target)
    target = re.sub(r"\(", " \( ", target)
    target = re.sub(r"\)", " \) ", target)
    target = re.sub(r"\?", " \? ", target)
    target = re.sub(r"\s{2,}", " ", target)
    target = re.sub(r"\s{2,}", " ", target)
    return target.translate(str.maketrans('', '', string.punctuation)).strip().lower()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            
def makeFeatureVec(review, model, num_features):
    
    featureVec = np.zeros((100,300), dtype = "float32")

    # Unique word set
    word_index = set(model.index2word)

    # For division we need to count the number of words
    nword = 0

    # Iterate words in a review and if the word is in the unique wordset, add the vector values for each word.
    for word in review[0]:
        if word in word_index:
            featureVec[nword]=model[word]
            nword += 1    
    # Divide the sum of vector values by total number of word in a review.
    #     featureVec = np.divide(featureVec, nword)        

    return featureVec

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes[0], yticklabels=classes[1],
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax