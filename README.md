# Simple Neural Network for MNIST Dataset

* Simple 3 Layer Neural Network in Python by using only basic library (Numpy).
* Sample multiclass using MNIST data provided by http://yann.lecun.com/exdb/mnist/

## Getting Started
```
git clone https://github.com/ghuw/simple-neural-net
```

### Prerequisites

* Python3 
* 1. Linux
```
sudo apt-get update && sudo apt-get install python3.6
```
* 2. Windows https://www.python.org/downloads/

* Numpy
```
pip3 install numpy
```

* Dill
```
pip3 install dill
```

* Matplotlib
```
pip3 install matplotlib
```

## Usage

### Download and extract dataset from yann.lecun.com/exdb/mnist/ and move to working directory.
* [train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz):  training set images (9912422 bytes) 
* [train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz):  training set labels (28881 bytes) 
* [t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz):   test set images (1648877 bytes) 
* [t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz):   test set labels (4542 bytes) 

## Running the classifier
```
python3 main.py
```

## Author

* [ghuw](https://github.com/ghuw)

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/ghuw/simple-neural-net/blob/master/LICENSE.md) file for details

## Acknowledgments

* This project is inspired from [miloharper](https://github.com/miloharper/simple-neural-network) work.
