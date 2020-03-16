import numpy as np
from PIL import Image
from os import path


def parse_labels(fname, mn, num):
    """
    Parse labels from the original binary file.
    :return: An array of labels.
    """
    labels_file = open(fname, 'r')
    # > represents big-endian, i4 represents 4-byte signed integer
    numbers = np.fromfile(labels_file, count=2, offset=0, dtype='>i4')
    magic_num = numbers[0]
    assert magic_num == mn, "Magic number does not match"
    N = numbers[1]
    assert N == num, "Number of elements does not match"

    # Note that fromfile DOES NOT reset the file offset
    labels = np.fromfile(labels_file, dtype='>u1')
    labels_file.close()

    return labels


def parse_images(fname, type, mn, num, nr=28, nc=28):
    """
    Parse images from the original binary file.
    :param fname: The original binary file.
    :param type: The type of data, either test or training.
    :param mn: The magic number for validation.
    :param num: The number of elements for validation.
    :param nr: The number of rows for validation, 28 by default.
    :param nc: The number of columns for validation, 28 by default.
    :return: An array of images.
    """
    images_file = open(fname, 'r')
    numbers = np.fromfile(images_file, count=4, offset=0, dtype='>i4')
    magic_num = numbers[0]
    assert magic_num == mn, "Magic number does not match"
    N = numbers[1]
    assert N == num, "Number of elements does not match"
    Nr = numbers[2]
    assert Nr == nr, "Number of rows does not match"
    Nc = numbers[3]
    assert Nc == nc, "Number of columns does not match"

    images_raw = np.fromfile(images_file, dtype='>u1')
    images = np.reshape(images_raw, (N, Nr, Nc))

    # Save test/training images to disk
    for i in range(N):
        save_path = "img/" + type + "/im_" + str(i) + ".jpeg"
        if not path.exists(save_path):
            im = Image.fromarray(images[i])
            im.save(save_path)

    images_file.close()

    return images


def main():
    """
    Pre-process the test and training data from the original binary files and store them into a HDF5 file.
    :return: None
    """
    training_labels = parse_labels('data/train-labels.idx1-ubyte', mn=2049, num=60000)
    print(training_labels)
    print(len(training_labels))

    test_labels = parse_labels('data/t10k-labels.idx1-ubyte', mn=2049, num=10000)
    print(test_labels)
    print(len(test_labels))

    training_images = parse_images('data/train-images.idx3-ubyte', type="training", mn=2051, num=60000)
    print(np.shape(training_images))

    test_images = parse_images('data/t10k-images.idx3-ubyte', type="test", mn=2051, num=10000)
    print(np.shape(test_images))


if __name__ == '__main__':
    main()
