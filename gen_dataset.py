# coding: utf-8

import fnmatch
import os
import random
import pickle

import numpy as np
from hdrio import imread, imsave
from envmap import EnvironmentMap
from scipy.ndimage.interpolation import zoom
from matplotlib import pyplot as plt


STACK_FILENAME = 'stack.exr'
TARGET_SIZE = (256, 256)


def getAvailableData(root_path, filename=STACK_FILENAME):
    """Get all the stacks images available."""
    matches = []
    for root, dirnames, filenames in os.walk(root_path):
        for filename in fnmatch.filter(filenames, filename):
            matches.append(os.path.join(root, filename))
    return matches


def generateLDRfromHDR(im_path, out_prefix):
    """Convert an HDR image into a clipped 0-255 value ("simulating" a camera)"""
    print('Processing: ', im_path)
    im = imread(im_path)
    
    h, w, c = im.shape
    im = im[:, w/2 - h/2:w/2 + h/2]

    envmap = EnvironmentMap(im, 'SkyAngular').convertTo('LatLong', TARGET_SIZE[0])
    im = envmap.data
    
    valid = (im > 0) & (~np.isnan(im))
    im_median = np.median(im[valid])
    im_low = np.percentile(im[valid], 3)
    im_high = np.percentile(im[valid], 95)
    
    #scales = (TARGET_SIZE[0]/im.shape[0], TARGET_SIZE[1]/im.shape[1])
    #im = zoom(im, [scales[0], scales[1], 1])
    
    with open(out_prefix + "_hdr.pkl", 'wb') as fhdl:
        pickle.dump(im, fhdl, pickle.HIGHEST_PROTOCOL)
    imsave(out_prefix + '_hdr.exr', im)
    
    # 20th percentile -> value 5
    # 80th percentile -> value 250
    #print("Ratio:", (im_high - im_low))
    ratio = im_high - im_low
    if ratio < 0.1:
        ratio = 0.1
    im_ldr = (im - im_low) * 250. / ratio + 5
    im_ldr = np.clip(im_ldr, 0, 255).astype('uint8')
    
    imsave(out_prefix + '_ldr.jpg', im_ldr)
    
    plt.figure()
    plt.subplot(1,2,1); plt.hist(im.ravel()[im.ravel()<im_high], 50)
    plt.subplot(1,2,2); plt.hist(im_ldr.ravel()[im_ldr.ravel()>0], 50)
    plt.savefig(out_prefix + 'debug.png')
    plt.close()

    
def main():
    im_paths = getAvailableData('/gel/rachmaninoff/data/pictures/master/skycam/')
    im_paths = random.sample(im_paths, 1000)
    for im_path in im_paths:
        root_path = os.path.dirname(im_path)
        out_path = os.path.join('data', "_".join(root_path.split(os.sep)[-2:]))
        try:
            generateLDRfromHDR(im_path, out_path)
        except Exception as e:
            print("Error happened:", e)


if __name__ == '__main__':
    main()
    

