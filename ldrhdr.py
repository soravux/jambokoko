
# coding: utf-8

# In[1]:

import os, sys, urllib, gzip, glob, time
import pickle # cPickle as pickle
sys.setrecursionlimit(10000)


# In[2]:

import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')
import numpy as np
from scipy.misc import imread, imsave
from IPython.display import Image as IPImage
from PIL import Image
#from hdrio import imsave


# In[3]:
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import get_output, InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer, get_all_params
from lasagne.nonlinearities import rectify, leaky_rectify, tanh
from lasagne.updates import nesterov_momentum, sgd
from lasagne.objectives import categorical_crossentropy, squared_error
#from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo


# In[4]:

from lasagne.layers import Conv2DLayer as Conv2DLayerSlow
from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerSlow
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast
    print('Using cuda_convnet (faster)')
except ImportError:
    from lasagne.layers import Conv2DLayer as Conv2DLayerFast
    from lasagne.layers import MaxPool2DLayer as MaxPool2DLayerFast
    print('Using lasagne.layers (slower)')


# In[5]:

# Load data
data_path = '/home/local/yahog/neuralnets/ldrhdr/data/'
#hdr = glob.glob(os.path.join(data_path, '*.pkl'))
ldr = glob.glob(os.path.join(data_path, '*.jpg'))

#data_file = 'data/data.pkl'

#if os.path.isfile(data_file):
#    with open(data_file, 'rb') as fhdl:
#        data_ldr, data_hdr = pickle.load(fhdl)
#else:
data_ldr = np.zeros((len(ldr), 1, 256, 256), dtype='float32')
data_hdr = np.zeros((len(ldr), 1, 256, 256), dtype='float32')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], np.array([0.299, 0.587, 0.144])/1.03)


for idx, image in enumerate(ldr):
    data_ldr[idx, 0, :, :] = rgb2gray(imread(image))
#    with open(image[:-8] + "_hdr.pkl", 'rb') as fhdl:
#        data_hdr[idx, 0, :, :] = rgb2gray(pickle.load(fhdl))


# Since it's LatLong, all the bottom half is black. Remove it
data_ldr = data_ldr[:,:,data_ldr.shape[2]/2,:]
#data_hdr = data_hdr[:,:,data_ldr.shape[2]/2,:]

#with open(data_file, 'wb') as fhdl:
#    pickle.dump([data_ldr, data_hdr], fhdl)


# In[6]:

# reshape from (50000, 784) to 4D tensor (50000, 1, 28, 28)
#X = np.reshape(X, (-1, 1, 28, 28))
data_ldr = data_ldr / data_ldr.max()
data_ldr_target = data_ldr.reshape((data_ldr.shape[0], data_ldr.shape[2]*data_ldr.shape[3]))
print('data_ldr type and shape:', data_ldr.dtype, data_ldr.shape)
print('data_ldr.min():', data_ldr.min())
print('data_ldr.max():', data_ldr.max())


# In[7]:

# we need our target to be 1 dimensional
data_hdr = data_hdr.reshape((data_ldr.shape[0], -1))
data_hdr = data_hdr / data_hdr.max()
data_hdr[data_hdr < 0] = 0
print('data_hdr:', data_hdr.dtype, data_hdr.shape)
print('data_hdr.min():', data_hdr.min())
print('data_hdr.max():', data_hdr.max())
#plt.imshow(np.reshape(np.log(data_hdr[0,:]), [256, 256])); plt.colorbar(); plt.show()
#imsave('debug1.exr', np.repeat(np.reshape(data_hdr[0,:], [256, 256])[:,:,np.newaxis], 3, axis=2))
#imsave('debug1_2.exr', np.repeat(np.reshape(data_ldr[0,:], [256, 256])[:,:,np.newaxis], 3, axis=2))


# In[8]:

conv_num_filters = 16
filter_size = 3
pool_size = 2
encode_size = 128
dense_mid_size = 256
pad_in = 'valid'
pad_out = 'full'
"""layers = [
    (InputLayer, {'shape': (None, data_ldr.shape[1], data_ldr.shape[2], data_ldr.shape[3])}), 
    (Conv2DLayerFast, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_in}),
    (Conv2DLayerFast, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_in}),
    (MaxPool2DLayerFast, {'pool_size': pool_size}),
    (Conv2DLayerFast, {'num_filters': 2*conv_num_filters, 'filter_size': filter_size, 'pad': pad_in}),
    (MaxPool2DLayerFast, {'pool_size': pool_size}),
    (ReshapeLayer, {'shape': (([0], -1))}),
    (DenseLayer, {'num_units': dense_mid_size}),
    (DenseLayer, {'name': 'encode', 'num_units': encode_size}),
    (DenseLayer, {'num_units': dense_mid_size}),
    (DenseLayer, {'num_units': 32*30**2}),
    (ReshapeLayer, {'shape': (([0], 2*conv_num_filters, 30, 30))}),
    (Upscale2DLayer, {'scale_factor': pool_size}),
    (Conv2DLayerFast, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_out}),
    (Upscale2DLayer, {'scale_factor': pool_size}),
    (Conv2DLayerFast, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_out}),
    (Upscale2DLayer, {'scale_factor': pool_size}),
    (Conv2DLayerSlow, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_out}),
    (Conv2DLayerSlow, {'num_filters': 1, 'filter_size': filter_size, 'pad': pad_out}),
    (ReshapeLayer, {'shape': (([0], -1))}),
]"""
conv_num_filters = 16
filter_size = 5
pool_size = 4
encode_size = 225
pad_in = 'valid'
pad_out = 'full'
layers = [
    (InputLayer, {'shape': (None, data_ldr.shape[1], data_ldr.shape[2], data_ldr.shape[3])}), 
    (Conv2DLayerFast, {'num_filters': conv_num_filters, 'filter_size': filter_size, 'pad': pad_in}),
    (MaxPool2DLayerFast, {'pool_size': pool_size}),
    (Conv2DLayerFast, {'num_filters': conv_num_filters, 'filter_size': filter_size-1, 'pad': pad_in}),
    (MaxPool2DLayerFast, {'pool_size': pool_size}),
    (ReshapeLayer, {'shape': (([0], -1))}),
    (DenseLayer, {'name': 'encode', 'num_units': encode_size}),
    (DenseLayer, {'num_units': 16*15**2}),
    (ReshapeLayer, {'shape': (([0], conv_num_filters, 15, 15))}),
    (Upscale2DLayer, {'scale_factor': pool_size}),
    (Conv2DLayerFast, {'num_filters': conv_num_filters, 'filter_size': filter_size-1, 'pad': pad_out}),
    (Upscale2DLayer, {'scale_factor': pool_size}),
    (Conv2DLayerFast, {'num_filters': 1, 'filter_size': filter_size, 'pad': pad_out}),
    (ReshapeLayer, {'shape': (([0], -1))}),
]


# In[ ]:

input_var = T.tensor4('inputs')
output_var = T.matrix('outputs')

network = layers[0][0](input_var=input_var, **layers[0][1])
for layer in layers[1:]:
    network = layer[0](network, **layer[1])

prediction = get_output(network)
loss = squared_error(prediction, output_var)
loss = loss.mean()

params = get_all_params(network, trainable=True)
#updates = nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
updates = sgd(loss, params, learning_rate=0.01)


test_prediction = get_output(network, deterministic=True)
test_loss = squared_error(test_prediction, output_var)
test_loss = test_loss.mean()
# As a bonus, also create an expression for the classification accuracy:
#test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), output_var),
#                  dtype=theano.config.floatX)

train_fn = theano.function([input_var, output_var], loss, updates=updates)# , mode=theano.compile.MonitorMode(post_func=theano.compile.monitormode.detect_nan))
#val_fn = theano.function([input_var, output_var], [test_loss, test_acc])
val_fn = theano.function([input_var, output_var], test_loss)



#ae = NeuralNet(
#    layers=layers,
#    max_epochs=20,
#    
#    update=nesterov_momentum,
#    update_learning_rate=0.01,
#    update_momentum=0.975,
#    
#    regression=True,
#    verbose=1
#)
#ae.initialize()
#PrintLayerInfo()(ae)
#pickle.dump(ae, open('conv_ae_init.pkl', 'wb'))



# In[ ]:

#ae.fit(data_ldr, data_ldr_target)

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


X_train = data_ldr
y_train = data_ldr_target
X_val = data_ldr
y_val = data_ldr_target
X_test = data_ldr
y_test = data_ldr_target


if not os.path.isfile('model.npz'):
    print('Training NN')
    num_epochs = 50
    bsize = 200
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, bsize, shuffle=True):
            inputs, targets = batch
            print(inputs.shape, targets.shape)
            #import pdb; pdb.set_trace()
            train_err += train_fn(inputs, targets)
            #import pdb; pdb.set_trace()
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, bsize, shuffle=False):
            inputs, targets = batch
            print(inputs.shape, targets.shape)
            err = val_fn(inputs, targets)
            acc = 0
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

# After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err = val_fn(inputs, targets)
        acc = 0
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

# Optionally, you could now dump the network weights to a file like this:
    np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
#
# And load them again later on like this:
else:
    with np.load('model.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)


prediction = get_output(network)
forward_fn = theano.function([input_var], prediction)

#import pdb; pdb.set_trace()


# In[ ]:

#from nolearn.lasagne.visualize import plot_loss
#plot_loss(ae)


# In[ ]:

# ae.save_params_to('mnist/conv_ae.np')

#pickle.dump(ae, open('conv_ae.pkl','wb'))
#ae = pickle.load(open('conv_ae.pkl','rb'))
#ae.layers


# In[ ]:

X_pred = forward_fn(data_ldr).reshape(-1, 256, 256)
#X_pred = ae.predict(data_ldr).reshape(-1, 256, 256)
X_pred = np.rint(256. * X_pred).astype(int)
X_pred = np.clip(X_pred, a_min = 0, a_max = 255)
X_pred = X_pred.astype('uint8')
print(X_pred.shape , data_ldr.shape)

aaa = np.rint(256. * data_ldr).astype(int)
aaa = np.clip(aaa, a_min = 0, a_max = 255)
aaa = aaa.astype('uint8')
data_ldr = aaa


#import pdb; pdb.set_trace()
for i in range(X_pred.shape[0]):
    imsave('out/{}.png'.format(i), np.concatenate((data_ldr[i,0,:,:], X_pred[i,:,:]), axis=1))




# In[ ]:

#imsave('debug2.exr', np.repeat(X_pred[0,:][:,:,np.newaxis], 3, axis=2))
#X_pred[0,:,:].max()
#from lasagne.layers import get_all_param_values
#print(ae.layers[2][0].__dict__.keys())
#print(get_all_param_values(ae.layers[2][0]))
#print(ae.layers_[2].__dict__.keys())
#print(ae.layers_[20].W.get_value())
#print(data_ldr.shape)
#indata = data_ldr[0,:,:,:][np.newaxis,:,:,:]
#for i in range(len(ae.layers_)):
    #thestack = [ae.layers_[x] for x in range(i+1)]
    #print(thestack)
    #outdata = get_output(thestack, indata)
#from pprint import pprint
#stack = [ae.layers_[i] for i in range(20)]
#outdata = get_output(stack, indata)
#o = [x.eval() for x in outdata]
#monPrint = PrintLayerInfo(); monPrint(ae)
#import pdb; pdb.set_trace()
#a = 123

def poulet():
# In[ ]:

#get_ipython().system(u'mkdir -p data')
#get_ipython().system(u'mkdir -p montage')


# In[ ]:

###  show random inputs / outputs side by side

    def get_picture_array(X, rescale=4):
        array = X.reshape(256,256)
        array = np.clip(array, a_min = 0, a_max = 255)
        return  array.repeat(rescale, axis = 0).repeat(rescale, axis = 1).astype(np.uint8())

    def compare_images(index):
        original_image = Image.fromarray(get_picture_array(255 * data_ldr[index]))
        new_size = (original_image.size[0] * 2, original_image.size[1])
        new_im = Image.new('L', new_size)
        new_im.paste(original_image, (0,0))
        rec_image = Image.fromarray(get_picture_array(X_pred[index]))
        new_im.paste(rec_image, (original_image.size[0],0))
        new_im.save('data/test.png', format="PNG")
        return IPImage('data/test.png')

    compare_images(2)
# compare_images(np.random.randint(50000))


# In[ ]:

## we find the encode layer from our ae, and use it to define an encoding function

    def get_layer_by_name(net, name):
        for i, layer in enumerate(net.get_all_layers()):
            if layer.name == name:
                return layer, i
        return None, None
    encode_layer, encode_layer_index = get_layer_by_name(ae, 'encode')

    def encode_input(encode_layer, X):
        return get_output(encode_layer, inputs=X).eval()

    X_encoded = encode_input(encode_layer, data_ldr)


# In[ ]:

    next_layer = ae.get_all_layers()[encode_layer_index + 1]
    final_layer = ae.get_all_layers()[-1]
    new_layer = InputLayer(shape=(None, encode_layer.num_units))

# N.B after we do this, we won't be able to use the original autoencoder , as the layers are broken up
    next_layer.input_layer = new_layer


# In[ ]:

#def get_output_from_nn(last_layer, X):
#    indices = np.arange(128, X.shape[0], 128)
#    sys.stdout.flush()
#
#    # not splitting into batches can cause a memory error
#    X_batches = np.split(X, indices)
#    out = []
#    for count, X_batch in enumerate(X_batches):
#        out.append(get_output(last_layer, X_batch).eval())
#        sys.stdout.flush()
#    return np.vstack(out)

    def get_output_from_nn(last_layer, X):
        return get_output(last_layer,X).eval()

    def decode_encoded_input(X):
        return get_output_from_nn(final_layer, X)

    X_decoded = 256 * decode_encoded_input(X_encoded[2]) * 2

    X_decoded = np.rint(X_decoded).astype(int)
    X_decoded = np.clip(X_decoded, a_min = 0, a_max = 255)
    X_decoded  = X_decoded.astype('uint8')
    print(X_decoded.shape)

    pic_array = get_picture_array(X_decoded)
    image = Image.fromarray(pic_array)
    image.save('data/test.png', format="PNG")  
    IPImage('data/test.png')


# In[ ]:

    enc_std = X_encoded.std(axis=0)
    enc_mean = X_encoded.mean(axis=0)
    enc_min = X_encoded.min(axis=0)
    enc_max = X_encoded.max(axis=0)
    m = X_encoded.shape[1]


# In[ ]:

    n = 256
    generated = np.random.normal(0, 1, (n, m)) * enc_std + enc_mean
    generated = generated.astype(np.float32).clip(enc_min, enc_max)
    X_decoded = decode_encoded_input(generated) * 256.
    X_decoded = np.rint(X_decoded ).astype(int)
    X_decoded = np.clip(X_decoded, a_min = 0, a_max = 255)
    X_decoded  = X_decoded.astype('uint8')
    get_ipython().system(u'mkdir -p montage')
    for i in range(n):
        pic_array = get_picture_array(X_decoded[i], rescale=1)
        image = Image.fromarray(pic_array)
        image.save('montage/{0:03d}.png'.format(i), format='png')


# In[ ]:

    get_ipython().system(u'montage -mode concatenate -tile 16x montage/*.png montage.png')
    IPImage('montage.png')


# In[ ]:



