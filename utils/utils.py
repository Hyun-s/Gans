import tensorflow as tf
from tensorflow.keras import layers

import torch
from torch import nn

import wandb
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


# torch version
class gradCam(nn.Module):
    '''
    #######
    #usage#
    #######
    test = Discriminator_gradCam(ngpu)
    test.main = netD.main[:11]
    test.classifier = netD.main[11:]

    test.eval()

    pred = test(tensor)
    pred.backward()

    # pull the gradients out of the model
    gradients = test.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = test.get_activations(tensor).detach()

    # weight the channels by corresponding gradients
    for i in range(512):
        try:
            activations[:, i, :, :] *= pooled_gradients[i]
        except:
            pass
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap.cpu(), 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # draw the heatmap
    ax = fig.add_subplot(row,col,idx-1)
    ax.matshow(heatmap.squeeze())
    # plt.show()
    ax = fig.add_subplot(row,col,idx)
    '''
    def __init__(self,model, ngpu, clf_index=None):
        super(gradCam, self).__init__()
        self.ngpu = ngpu
        self.model = model
        self.main = model[:clf_index]
        self.classifier = model[clf_index:]

    def activations_hook(self, grad):
        self.gradients = grad
    def forward(self, input):
        feature = self.main(input)
        h = feature.register_hook(self.activations_hook)
        out = self.classifier(feature)
        return out
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.main(x)

class wandb_logger():
    def __init__(self, nz, device, n=16):
        self.noise = torch.randn(n, nz, 1, 1, device=device)
        # self.G = copy.deepcopy(G)
        # self.D = copy.deepcopy(D)
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')
    def get_featuremap(self,model,layers = ['Conv2d', 'ConvTranspose2d'],g=None, z=None):
        '''
        model = generator or discriminator
        layer_name = Conv2d or ConvTranspose2d
        output conv, convtranspose featuremap
        '''
        feature_dict = {}
        features = {}
        model.eval()
        with torch.no_grad():
            if model.__class__.__name__ == 'Generator':
                x = model.main[0](self.noise)
            else:
                assert g != None, 'Generator required'
                g.eval()
                image = g(self.noise)
                x = model.main[0](image)
                image = image.permute(0,2,3,1).detach().cpu().numpy()
                image = ((image+1)* 127.5).astype(int)
                feature_dict['Fixed_z_image'] = [wandb.Image(img) for img in image]
            for idx, layer in enumerate(model.main[1:]):
                x = layer(x)
                name = model.__class__.__name__ + '_' + \
                    str(idx+1) + '_' +  \
                    layer.__class__.__name__
                if name.split('_')[-1] in layers:
                    var = torch.var(x,axis=1)
                    var = torch.nan_to_num(var).detach().cpu().numpy()
                    feature_dict[name+'_var'] = [wandb.Image(img) for img in var]

                    means = torch.mean(x,axis=1)[:,np.newaxis,:,:]
                    means = torch.nan_to_num(means)
                    feature_dict[name+'_mean'] = [wandb.Image(img) for img in means]
        
        return feature_dict