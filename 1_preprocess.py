
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
get_ipython().magic(u'matplotlib inline')


url = 'http://ufldl.stanford.edu/housenumbers/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  return filename

train_filename = maybe_download('train.tar.gz')
test_filename = maybe_download('test.tar.gz')


train_matfile = maybe_download('train_32x32.mat')
test_matfile = maybe_download('test_32x32.mat')
extra_matfile = maybe_download('extra_32x32.mat')


import scipy.io
train_data = scipy.io.loadmat('train_32x32.mat', variable_names='X').get('X')
train_labels = scipy.io.loadmat('train_32x32.mat', variable_names='y').get('y')
test_data = scipy.io.loadmat('test_32x32.mat', variable_names='X').get('X')
test_labels = scipy.io.loadmat('test_32x32.mat', variable_names='y').get('y')
extra_data = scipy.io.loadmat('extra_32x32.mat', variable_names='X').get('X')
extra_labels = scipy.io.loadmat('extra_32x32.mat', variable_names='y').get('y')

print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)
print(extra_data.shape, extra_labels.shape)


# Build Validation Dataset and Labels Based on the Methods in This Paper:
# [https://arxiv.org/pdf/1204.3968.pdf]

import random

random.seed()

n_labels = 10
valid_index = []
valid_index2 = []
train_index = []
for i in np.arange(n_labels):
    valid_index.extend(np.where(train_labels[:,0] == (i+1))[0][:400].tolist())
    train_index.extend(np.where(train_labels[:,0] == (i+1))[0][400:].tolist())
    valid_index2.extend(np.where(extra_labels[:,0] == (i+1))[0][:200].tolist())

random.shuffle(valid_index)
random.shuffle(train_index)
random.shuffle(valid_index2)

valid_data = np.concatenate((train_data[:,:,:,valid_index], extra_data[:,:,:,valid_index2]), axis=3)
valid_labels = np.concatenate((train_labels[valid_index,:], extra_labels[valid_index2,:]), axis=0)
train_data_t = train_data[:,:,:,train_index]
train_labels_t = train_labels[train_index,:]

print(train_data_t.shape, train_labels_t.shape)
print(test_data.shape, test_labels.shape)
print(valid_data.shape, valid_labels.shape)


image_size = 32  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def image_convert(image):
    '''Normalize images and convert RGB to Grayscale'''
    image_data_normal = (image.astype(float) - pixel_depth / 2) / pixel_depth
    image_data = np.ndarray((image.shape[3], image_size, image_size), dtype=np.float32)
    for i in range(image.shape[3]):
        # Use the Conversion Method in This Paper:
        # [http://www.eyemaginary.com/Rendering/TurnColorsGray.pdf]
        image_data[i,:,:] = 0.2989 * image_data_normal[:,:,0,i] +        0.5870 * image_data_normal[:,:,1,i] + 0.1140 * image_data_normal[:,:,2,i]
    return image_data

train_data_c = image_convert(train_data_t)
test_data_c = image_convert(test_data)
valid_data_c = image_convert(valid_data)
train_labels_c = train_labels_t[:,0]
test_labels_c = test_labels[:,0]
valid_labels_c = valid_labels[:,0]

print(train_data_c.shape, train_labels_c.shape)
print(test_data_c.shape, test_labels_c.shape)
print(valid_data_c.shape, valid_labels_c.shape)


pickle_file = 'SVHN.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_data_c,
    'train_labels': train_labels_c,
    'valid_dataset': valid_data_c,
    'valid_labels': valid_labels_c,
    'test_dataset': test_data_c,
    'test_labels': test_labels_c,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
    
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


plt.rcParams['figure.figsize'] = (15.0, 15.0)
f, ax = plt.subplots(nrows=1, ncols=10)

for i, j in enumerate(np.random.randint(0, train_labels_c.shape[0], size=10)):
    ax[i].axis('off')
    ax[i].set_title(train_labels_c[j], loc='center')
    ax[i].imshow(train_data_c[j,:,:])


# Create a Logistic Regression Classifier
clf = LogisticRegression(penalty='l2', tol=0.0001, C=1.0, random_state=None, solver='sag', max_iter=100,multi_class='ovr', verbose=0, n_jobs=4)

clf.fit(train_data_c.reshape(train_data_c.shape[0],-1), train_labels_c)
train_prediction = clf.predict(train_data_c.reshape(train_data_c.shape[0],-1))
test_prediction = clf.predict(test_data_c.reshape(test_data_c.shape[0],-1))

print('Classification report of training data:\n', classification_report(train_labels_c, train_prediction))
print('Confusion Matrix of training data:\n', confusion_matrix(train_labels_c, train_prediction))

print('Classification report of training data:\n', classification_report(test_labels_c, test_prediction))
print('Confusion Matrix of training data:\n', confusion_matrix(test_labels_c, test_prediction))

