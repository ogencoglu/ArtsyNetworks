'''
Contact        : oguzhan.gencoglu@tut.fi
'''

import sys
import numpy as np
import pickle
import skimage.transform
import scipy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import lasagne
from lasagne.utils import floatX
from lasagne.layers import InputLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer


def initialize_network(width):
    # initialize network - VGG19 style
    
    net = {}
    net['input'] = InputLayer((1, 3, width, width))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2, mode='average_exc_pad')
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_4'], 2, mode='average_exc_pad')

    return net
    
    
def prepare_image(img, width, means):
    
    # if not RGB, force 3 channels
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.repeat(img, 3, axis=2)
    h, w, _ = img.shape
    if h < w:
        img = skimage.transform.resize(img, (width, w*width/h), preserve_range=True)
    else:
        img = skimage.transform.resize(img, (h*width/w, width), preserve_range=True)

    # crop the center
    h, w, _ = img.shape
    img = img[h//2 - width//2:h//2 + width//2, w//2 - width//2:w//2 + width//2]
    
    rawim = np.copy(img).astype('uint8')
    
    # shuffle axes to c01
    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
    
    # convert RGB to BGR
    img = img[::-1, :, :]
    
    # zero mean scaling
    img = img - means
    
    return rawim, floatX(img[np.newaxis])
    
    
def precompute_activations(layers, base, style):
    # layer activations are precomputed here
    
    input_im_theano = T.tensor4()
    outputs = lasagne.layers.get_output(layers.values(), input_im_theano)
    
    photo_feat = {k: theano.shared(output.eval({input_im_theano: base}))
                    for k, output in zip(layers.keys(), outputs)}
    art_feat = {k: theano.shared(output.eval({input_im_theano: style}))
                    for k, output in zip(layers.keys(), outputs)}
                    
    return photo_feat, art_feat
    

def gram_mat(vecs):
    # theano gram matrix
    
    vecs = vecs.flatten(ndim = 3)
    gram = T.tensordot(vecs, vecs, axes=([2], [2]))
    
    return gram


def content_loss(X, Y, layer):
    # loss function for content
    
    loss = 1./2 * ((Y[layer] - X[layer])**2).sum()
    
    return loss


def style_loss(X, Y, layer):
    # loss function for style
    
    x = X[layer]
    y = Y[layer]
    
    N = x.shape[1]
    M = x.shape[2] * x.shape[3]
    
    loss = 1./(4 * N**2 * M**2) * ((gram_mat(y) - gram_mat(x))**2).sum()
    
    return loss
    
    
def tv_loss(x, k):
    # total variation loss for hf noise reduction
    
    rev = x[:,:,:-1,:-1]
    
    return (((rev - x[:,:,1:,:-1])**2 + (rev - x[:,:,:-1,1:])**2)**k).sum()
     
    
def eval_loss(x0, width):
    # Helper function to interface with scipy.optimize
    
    x0 = floatX(x0.reshape((1, 3, width, width)))
    generated.set_value(x0)
    
    return f_loss().astype('float64')
    
    
def define_global_loss(photo_features, gen_features, art_features, cl_scalar, sl_scalar, tv_scalar, tv_pow):
    # define total loss
    
    losses = []
    losses.append(cl_scalar * content_loss(photo_features, gen_features, 'conv4_2'))
    losses.append(sl_scalar * style_loss(art_features, gen_features, 'conv1_1'))
    losses.append(sl_scalar * style_loss(art_features, gen_features, 'conv2_1'))
    losses.append(sl_scalar * style_loss(art_features, gen_features, 'conv3_1'))
    losses.append(sl_scalar * style_loss(art_features, gen_features, 'conv4_1'))
    losses.append(sl_scalar * style_loss(art_features, gen_features, 'conv5_1'))
    losses.append(tv_scalar * tv_loss(generated, tv_pow))
    total_loss = sum(losses)
    grad = T.grad(total_loss, generated)
    f_loss = theano.function([], total_loss)
    f_grad = theano.function([], grad)
    
    return f_loss, f_grad
    
    
def eval_grad(x0, width):
    # Helper function to interface with scipy.optimize
    
    x0 = floatX(x0.reshape((1, 3, width, width)))
    generated.set_value(x0)
    
    return np.array(f_grad()).flatten().astype('float64')
    
    
def roll_back(x, means):
    # roll back the changes
    
    x = np.copy(x[0])
    x = x + means

    x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)
    
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x
    

if __name__ == '__main__':
    
    photo_path = sys.argv[1]
    art_path = sys.argv[2]

    IMAGE_W = 500
    
    # build VGG net and load weights (unpickle form VGGnet)
    net = initialize_network(IMAGE_W)
    values = pickle.load(open('vgg19_normalized.pkl', 'rb'))['param values']
    lasagne.layers.set_all_param_values(net['pool5'], values)
    
    photo = plt.imread(photo_path)
    art = plt.imread(art_path)
    if(len(photo.shape)==2):
        photo = np.dstack((photo, photo, photo))
    if(art.shape[2]==4):
        art = art[:,:,0:3]
    means = np.mean(np.mean(photo, axis=1), axis=0).reshape((3,1,1))
    rawim, photo = prepare_image(photo, IMAGE_W, means)
    rawim, art = prepare_image(art, IMAGE_W, means) 
    
    # precompute layer activations
    layers = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    layers = {k: net[k] for k in layers}
    photo_features, art_features = precompute_activations(layers, photo, art)
                    
    # Get expressions for layer activations for generated image
    lim = 128
    generated = theano.shared(floatX(np.random.uniform(-1*lim, lim, (1, 3, IMAGE_W, IMAGE_W))))
    
    gen_features = lasagne.layers.get_output(layers.values(), generated)
    gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}
    
    # define loss functions
    f_loss, f_grad = define_global_loss(photo_features, gen_features, art_features, cl_scalar = 0.001, sl_scalar = 2e5, tv_scalar = 1e-8, tv_pow = 1.25)
        
    # start from random noise
    generated.set_value(floatX(np.random.uniform(-1*lim, lim, (1, 3, IMAGE_W, IMAGE_W))))
    x0 = generated.get_value().astype('float64')
    xs = []
    xs.append(x0)
    
    # optimize, saving the result periodically
    iters = 8
    for i in range(iters):
        print(i)
        scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, args=(IMAGE_W,), maxfun=40)
        x0 = generated.get_value().astype('float64')
        xs.append(x0)
        
    plt.figure(figsize=(12,12))
    for i in range(iters):
        plt.subplot(3, 4, i+1)
        plt.gca().xaxis.set_visible(False)    
        plt.gca().yaxis.set_visible(False)    
        plt.imshow(roll_back(xs[i], means))
    plt.tight_layout()
    plt.savefig('progress.png')
    
    plt.figure(figsize=(8,8))
    plt.imshow(roll_back(xs[-1], means), interpolation='nearest')
    plt.savefig('neural_painting.png')