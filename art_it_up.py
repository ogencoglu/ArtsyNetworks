import lasagne
import numpy as np
import pickle
import skimage.transform
import scipy
import theano
import theano.tensor as T
from lasagne.utils import floatX
import matplotlib.pyplot as plt
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

    # central crop
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
    

def gram_mat(vecs):
    # theano gram matrix
    
    vecs = vecs.flatten(ndim = 3)
    gram = T.tensordot(vecs, vecs, axes=([2], [2]))
    
    return gram


def content_loss(X, Y, layer):
    
    loss = 1./2 * ((Y[layer] - X[layer])**2).sum()
    
    return loss


def style_loss(X, Y, layer):
    
    x = X[layer]
    y = Y[layer]
    
    N = x.shape[1]
    M = x.shape[2] * x.shape[3]
    
    loss = 1./(4 * N**2 * M**2) * ((gram_mat(y) - gram_mat(x))**2).sum()
    
    return loss
    
    
def total_variation_loss(x, k):
    
    rev = x[:,:,:-1,:-1]
    
    return (((rev - x[:,:,1:,:-1])**2 + (rev - x[:,:,:-1,1:])**2)**k).sum()
     
    
def eval_loss(x0, width):
    # Helper function to interface with scipy.optimize
    
    x0 = floatX(x0.reshape((1, 3, width, width)))
    generated_image.set_value(x0)
    
    return f_loss().astype('float64')
    
def eval_grad(x0, width):
    # Helper function to interface with scipy.optimize
    
    x0 = floatX(x0.reshape((1, 3, width, width)))
    generated_image.set_value(x0)
    
    return np.array(f_grad()).flatten().astype('float64')
    
    
def deprocess(x, means):
    
    x = np.copy(x[0])
    x = x + means

    x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)
    
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x
    

if __name__ == '__main__':

    IMAGE_W = 600
    
    # build VGG net and load weights (unpickle form VGGnet)
    net = initialize_network(IMAGE_W)
    values = pickle.load(open('vgg19_normalized.pkl', 'rb'))['param values']
    lasagne.layers.set_all_param_values(net['pool5'], values)
    
    photo = plt.imread('tietotalo.jpg')
    means = np.mean(np.mean(photo, axis=1), axis=0).reshape((3,1,1))
    rawim, photo = prepare_image(photo, IMAGE_W, means)
    plt.imshow(rawim)
    

    art = plt.imread('the_starry_night.jpg')
    rawim, art = prepare_image(art, IMAGE_W, means)
    plt.imshow(rawim)
    
    
    layers = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    layers = {k: net[k] for k in layers}
    
    
    # Precompute layer activations for photo and artwork
    input_im_theano = T.tensor4()
    outputs = lasagne.layers.get_output(layers.values(), input_im_theano)
    
    photo_features = {k: theano.shared(output.eval({input_im_theano: photo}))
                    for k, output in zip(layers.keys(), outputs)}
    art_features = {k: theano.shared(output.eval({input_im_theano: art}))
                    for k, output in zip(layers.keys(), outputs)}
                    
    # Get expressions for layer activations for generated image
    generated_image = theano.shared(floatX(np.random.uniform(-128, 128, (1, 3, IMAGE_W, IMAGE_W))))
    
    gen_features = lasagne.layers.get_output(layers.values(), generated_image)
    gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}
    
    # Define loss function
    losses = []
    
    # content loss
    cl_scalar = 0.001
    losses.append(cl_scalar * content_loss(photo_features, gen_features, 'conv4_2'))
    
    # style loss
    sl_scalar = 2e5
    losses.append(sl_scalar * style_loss(art_features, gen_features, 'conv1_1'))
    losses.append(sl_scalar * style_loss(art_features, gen_features, 'conv2_1'))
    losses.append(sl_scalar * style_loss(art_features, gen_features, 'conv3_1'))
    losses.append(sl_scalar * style_loss(art_features, gen_features, 'conv4_1'))
    losses.append(sl_scalar * style_loss(art_features, gen_features, 'conv5_1'))
    
    # total variation penalty
    tv_scalar = 1e-8
    tv_k = 1.25
    losses.append(tv_scalar * total_variation_loss(generated_image, tv_k))
    
    total_loss = sum(losses)
    
    grad = T.grad(total_loss, generated_image)
    
    # Theano functions to evaluate loss and gradient
    f_loss = theano.function([], total_loss)
    f_grad = theano.function([], grad)
        
    # Initialize with a noise image
    lim = 128
    generated_image.set_value(floatX(np.random.uniform(-1*lim, lim, (1, 3, IMAGE_W, IMAGE_W))))
    
    x0 = generated_image.get_value().astype('float64')
    xs = []
    xs.append(x0)
    
    # Optimize, saving the result periodically
    for i in range(8):
        print(i)
        scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, args=(IMAGE_W,), maxfun=50)
        x0 = generated_image.get_value().astype('float64')
        xs.append(x0)
        
    plt.figure(figsize=(12,12))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.gca().xaxis.set_visible(False)    
        plt.gca().yaxis.set_visible(False)    
        plt.imshow(deprocess(xs[i], means))
    plt.tight_layout()
    plt.savefig('progress.png')
    
    plt.figure(figsize=(8,8))
    plt.imshow(deprocess(xs[-1]), interpolation='nearest')
    plt.savefig('neural_painting.png')