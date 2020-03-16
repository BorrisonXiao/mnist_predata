# mnist_predata
A pre-processing script of the MNIST handwritten digits dataset.<br>
The processed data is stored in a HDF5 file named "mnist.h5" with labels
1. training_labels;
2. training_images;
3. test_labels;
4. test_images.

The recovered images (from bytes) were also stored in a directory "/data".<br>

Usage (virtual environment is recommended, python3 required):
1. pip install -r requirements.txt
2. python prep_data.py
