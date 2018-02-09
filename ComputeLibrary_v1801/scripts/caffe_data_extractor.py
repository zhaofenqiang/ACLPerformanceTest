#!/usr/bin/env python
"""Extracts trainable parameters from Caffe models and stores them in numpy arrays.
Usage
    python caffe_data_extractor -m path_to_caffe_model_file -n path_to_caffe_netlist

Saves each variable to a {variable_name}.npy binary file.

Tested with Caffe 1.0 on Python 2.7
"""
import argparse
import caffe
import os
import numpy as np


if __name__ == "__main__":
    # Parse arguments
#    parser = argparse.ArgumentParser('Extract Caffe net parameters')
#    parser.add_argument('-m', dest='modelFile', type=str, required=True, help='Path to Caffe model file')
#    parser.add_argument('-n', dest='netFile', type=str, required=True, help='Path to Caffe netlist')
#    args = parser.parse_args()

    # Create Caffe Net
#    net = caffe.Net(args.netFile, 1, weights=args.modelFile)
   
#    net = caffe.Net("/home/zfq/ACL1801/ComputeLibrary/scripts/inception-v4/deploy_inception-v4.prototxt", 1, weights="/home/zfq/ACL1801/ComputeLibrary/scripts/inception-v4/inception-v4.caffemodel")
    net = caffe.Net("/home/zfq/ACL1801/ComputeLibrary/scripts/renet50/ResNet-50-deploy.prototxt", 1, weights="/home/zfq/ACL1801/ComputeLibrary/scripts/renet50/ResNet-50-model.caffemodel")    
    # Read and dump blobs
#    net_iter = net.params.iteritems()
#    name, blobs = net_iter.next()
#    a = blobs[0].data
#  
    for name, blobs in net.params.iteritems():
        print('Name: {0}, Blobs: {1}'.format(name, len(blobs)))
        for i in range(len(blobs)):
            # Weights
            if i == 0:
                outname = name + "_w"
            # Bias
            elif i == 1:
                outname = name + "_b"
            else:
                outname = name + "_e"

            varname = outname
            if os.path.sep in varname:
                varname = varname.replace(os.path.sep, '_')
            print("Saving variable {0} with shape {1} ...".format(varname, blobs[i].data.shape))
            # Dump as binary
            np.save(varname, blobs[i].data)
