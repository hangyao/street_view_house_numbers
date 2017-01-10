# Street View House Numbers

- `1_preprocess_single.ipynb`: preprocess cropped single-digit SVHN data.
- `2_CNN_single.ipynb`: build a convolutional neural network for cropped single-digit SVHN images.
- `3_preprocess_multi.ipynb`: preprocess original multi-digit SVHN data.
- `4_CNN_multi.ipynb`: build a convolutional neural network for original multi-digit SVHN images.

### References

1. Ian J. Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, and Vinay Shet (2013). Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks. [arXiv:1312.6082](https://arxiv.org/abs/1312.6082) [cs.CV]

2. Pierre Sermanet, Soumith Chintala, and Yann LeCun (2012). Convolutional Neural Networks Applied to House Numbers Digit Classification. [arXiv:1204.3968](https://arxiv.org/abs/1204.3968) [cs.CV]

3. Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng (2011). Reading Digits in Natural Images with Unsupervised Feature Learning. *NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011*. ([Page](http://ufldl.stanford.edu/housenumbers/)|[PDF](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf))

4. Mark Grundland, and Neil A. Dodgson (2007). Decolorize: Fast, contrast enhancing, color to grayscale conversion. *Pattern Recognition*, **40** (11). [Page](http://dx.doi.org/10.1016/j.patcog.2006.11.003)

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

Project codes are provided in the following notebook files:
- `1_preprocess_single.ipynb`
- `2_CNN_single.ipynb`
- `3_preprocess_multi.ipynb`
- `4_CNN_multi.ipynb`

## Run

In a terminal or command window, navigate to the top-level project directory `street_view_house_numbers/` (that contains this README) and run one of the following commands:

```
python 1_preprocess_single.ipynb
jupyter notebook 1_preprocess_single.ipynb
python 2_CNN_single.ipynb
jupyter notebook 2_CNN_single.ipynb
```

This will open the iPython Notebook software and project file in your browser.

## Data

This project uses the [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/).

SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images. 

## License

The contents of this repository are covered under the [MIT License](LICENSE).
