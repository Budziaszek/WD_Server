import numpy as np
from skimage import transform, io


def loadData(image, mask, im_shape):
    """This function loads data preprocessed with ... """
    X, y = [], []

    img = transform.resize(image, im_shape, mode='constant')
    img = np.expand_dims(img, -1)
    mask = transform.resize(mask, im_shape, mode='constant')
    mask = np.expand_dims(mask, -1)
    X.append(img)
    y.append(mask)
    X = np.array(X)
    y = np.array(y)
    X -= X.mean()
    X /= X.std()

    return X, y


def loadDataOrig(df, path, im_shape, disp_names=False):
    """This function loads data preprocessed with ... """
    X, y = [], []
    for i, item in df.iterrows():
        if disp_names == True:
            print('i={} {} {}\n'.format(i, path + item[0], path + item[1]))
        img = io.imread(path + item[0])
        img = transform.resize(img, im_shape, mode='constant')
        img = np.expand_dims(img, -1)
        mask = io.imread(path + item[1])
        mask = transform.resize(mask, im_shape, mode='constant')
        mask = np.expand_dims(mask, -1)
        X.append(img)
        y.append(mask)
        break
    X = np.array(X)
    y = np.array(y)

    if disp_names == True:
        print('X/shape={} y.shape={}\n'.format(X.shape, y.shape))

    X -= X.mean()
    X /= X.std()

    if disp_names == True:
        print('### Data loaded')
        print('\t{}'.format(path))
        print('\t{}\t{}'.format(X.shape, y.shape))
        # print( '\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()) )
        print('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
        print('\n')

    return X, y
