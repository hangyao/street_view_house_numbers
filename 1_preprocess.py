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
from sklearn.cross_validation import StratifiedShuffleSplit
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
get_ipython().magic(u'matplotlib inline')


# Download Data Files

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

# Download mat Data Files

train_matfile = maybe_download('train_32x32.mat')
test_matfile = maybe_download('test_32x32.mat')
extra_matfile = maybe_download('extra_32x32.mat')


# Load mat Data Files

import scipy.io
train_data = scipy.io.loadmat('train_32x32.mat', variable_names='X').get('X')
train_labels = scipy.io.loadmat('train_32x32.mat', variable_names='y').get('y')
test_data = scipy.io.loadmat('test_32x32.mat', variable_names='X').get('X')
test_labels = scipy.io.loadmat('test_32x32.mat', variable_names='y').get('y')
extra_data = scipy.io.loadmat('extra_32x32.mat', variable_names='X').get('X')
extra_labels = scipy.io.loadmat('extra_32x32.mat', variable_names='y').get('y')

#print(train_data.shape, train_labels.shape)
#print(test_data.shape, test_labels.shape)
#print(extra_data.shape, extra_labels.shape)

# Stratified Shuffle Validation Data Set

random_state = 43
y = extra_labels[:,0]
index = StratifiedShuffleSplit(y, 10, test_size=25000, random_state=random_state)
for _, valid_index in index:
    valid_data, valid_labels = extra_data[:,:,:,valid_index], extra_labels[valid_index]

#print(valid_data.shape)
#print(valid_labels.shape)

# Check Validation Images

#num_image = 24567
#plt.imshow(valid_data[:,:,:,num_image])
#print(valid_labels[num_image])

# Check Validation Dataset Balance

#num_labels = [None] * 10
#for i in range(10):
#    num_labels[i] = sum(valid_labels == i+1)
#
#print(sum(num_labels))
#print(num_labels)

# Normalize Image Data and Convert to Grayscale
# Reference the publication below for RGB to Grayscale Conversion:
# [http://www.eyemaginary.com/Rendering/TurnColorsGray.pdf]

image_size = 32  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def image_convert(image):
    
    image_data_normal = (image.astype(float) - pixel_depth / 2) / pixel_depth
    image_data = np.ndarray((image.shape[3], image_size, image_size), dtype=np.float32)
    for i in range(image.shape[3]):
        image_data[i,:,:] = 0.2989 * image_data_normal[:,:,0,i] + \
          0.5870 * image_data_normal[:,:,1,i] + 0.1140 * image_data_normal[:,:,2,i]
    return image_data

train_data_convert = image_convert(train_data)
test_data_convert = image_convert(test_data)
valid_data_convert = image_convert(valid_data)
train_labels_convert = train_labels[:,0]
test_labels_convert = test_labels[:,0]
valid_labels_convert = valid_labels[:,0]

# Check Converted Image

#plt.imshow(valid_data_convert[num_image,:,:])
#print(valid_labels_convert[num_image])


# Save Converted Data Set in a Pickle File

pickle_file = 'SVHN.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_data_convert,
    'train_labels': train_labels_convert,
    'valid_dataset': valid_data_convert,
    'valid_labels': valid_labels_convert,
    'test_dataset': test_data_convert,
    'test_labels': test_labels_convert,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
    
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

# Run a Logistic Regression on Training and Test Datasets

# Create a Logistic Regression Classifier
clf = LogisticRegression(penalty='l2', tol=0.0001, C=1.0, random_state=random_state, solver='sag', max_iter=100,multi_class='ovr', verbose=0, n_jobs=4)

clf.fit(train_data_convert.reshape(train_data_convert.shape[0],-1), train_labels_convert)


predicted = clf.predict(test_data_convert.reshape(test_data_convert.shape[0],-1))

# Print Results
print(classification_report(test_labels_convert, predicted))
print(confusion_matrix(test_labels_convert, predicted))

