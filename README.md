# Street View House Numbers
- 1_preprocess_single.ipynb: preprocess cropped single-digit SVHN data.
- 2_CNN_single.ipynb: build a convolutional neural network for cropped single-digit SVHN images.
- 3_preprocess_multi.ipynb: preprocess original multi-digit SVHN data.
- 4_CNN_multi.ipynb: build a convolutional neural network for original multi-digit SVHN images.

## Install

This project requires **Python 2.7** and the following Python libraries installed:

- [TensorFlow](http://www.tensorflow.org/)
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [SciPy library](http://www.scipy.org/scipylib/index.html)
- [Six](http://pypi.python.org/pypi/six/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

## Code

Template code is provided in the following notebook files:
- `1_preprocess_single.ipynb`
- `2_CNN_single.ipynb`
- `3_preprocess_multi.ipynb`
- `4_CNN_multi.ipynb`

## Run

In a terminal or command window, navigate to the top-level project directory `street_view_house_numbers/` (that contains this README) and run one of the following commands:

```python 1_preprocess.ipynb```  
```jupyter notebook 1_preprocess.ipynb```

This will open the iPython Notebook software and project file in your browser.

## Data

This project uses the [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/).

SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images. 
