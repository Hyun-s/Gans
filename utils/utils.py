import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import scipy

class fid():
    def __init__(self, shape=(75,75,3),sample = None):
        super(fid, self).__init__()
        self.input_shape = shape
        self.resize = layers.Resizing(*self.input_shape[:2])
        self.inception = tf.keras.applications.inception_v3.InceptionV3(
                            include_top=False,
                            weights='imagenet',
                            input_shape=self.input_shape,
                            pooling=None
                        )
        self.sample = None
        if sample != None:
            self.sample = self.run(sample)
    def run(self,x):
        x = self.resize(x)
        x = self.inception(x)
        x = x.numpy()
        # x = tf.reshape(x, [x.shape[0],x.shape[-1]])
        x = x.reshape((x.shape[0],x.shape[-1]))
        return x
    def calc(self, real, fake):
        if self.sample != None:
            real = self.sample
        real = self.run(real)
        fake = self.run(fake)
        mu_r, cov_r = np.mean(real,axis=0), np.cov(real,rowvar=False)
        mu_f, cov_f = np.mean(fake,axis=0), np.cov(fake,rowvar=False)
        mu = np.linalg.norm(mu_r-mu_f)
        cov = np.trace(cov_r+cov_f -2*  scipy.linalg.sqrtm(cov_r@cov_f))
        fid = mu + cov
        if np.iscomplex(fid):
            fid = fid.real
        return fid