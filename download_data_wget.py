import wget
import os
import tensorflow as tf

# The URLs where the MNIST data can be downloaded.
_DATA_URL = 'http://yann.lecun.com/exdb/mnist/'
_TRAIN_DATA_FILENAME = 'train-images-idx3-ubyte.gz'
_TRAIN_LABELS_FILENAME = 'train-labels-idx1-ubyte.gz'
_TEST_DATA_FILENAME = 't10k-images-idx3-ubyte.gz'
_TEST_LABELS_FILENAME = 't10k-labels-idx1-ubyte.gz'
dataset_dir = '~/datasets/mnist'

#save_dir = os.getcwd()
#os.chdir(dataset_dir)

def _download_dataset_mnist(dataset_dir):
    """Downloads MNIST locally.

    Args:
    dataset_dir: The directory where the temporary files are stored.
    """
    dataset_dir = os.path.expandvars(dataset_dir)
    dataset_dir = os.path.expanduser(dataset_dir)

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    for filename in [_TRAIN_DATA_FILENAME,
                   _TRAIN_LABELS_FILENAME,
                   _TEST_DATA_FILENAME,
                   _TEST_LABELS_FILENAME]:
        filepath = os.path.join(dataset_dir, filename)

        if not os.path.exists(filepath):
            print('Downloading file %s...' % filename)
            wget.download(_DATA_URL + filename, out=dataset_dir)
            print()
            with tf.gfile.GFile(filepath) as f:
                size = f.size()
                print('Successfully downloaded', filename, size, 'bytes.')
        else:
            print('%s file is already downloaded' %url)



def main():

    _download_dataset_mnist(dataset_dir)

#restore the dir
#os.chdir(save_dir)
if __name__ == '__main__':
  main()