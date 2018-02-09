#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.python import pywrap_tensorflow

#reader = pywrap_tensorflow.NewCheckpointReader("/home/zfq/ACL1801/ComputeLibrary/scripts/inception-v3/inception_v3.ckpt")
#reader = pywrap_tensorflow.NewCheckpointReader("/home/zfq/ACL1801/ComputeLibrary/scripts/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader("/home/zfq/ACL1801/ComputeLibrary/scripts/inception-v4/inception_v4.ckpt")
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:   
    
    if (('RMSProp' in key) or ('Exponential' in key)):
        continue

    if key[:11] == "InceptionV4":
        varname = key[12:]
        if os.path.sep in varname:
            varname = varname.replace(os.path.sep, '_')
#            if (('depthwise_depthwise_weights' in varname) and (var_to_shape_map[key][3] == 1)):
#                a = reader.get_tensor(key)
#                b = a[:,:,:,0]
#            else:
            b = reader.get_tensor(key) 
#            b.flags['C_CONTIGUOUS']
            b = np.asfortranarray(b)
            print("Saving variable {0} with shape {1} ...".format(varname, b.shape))  
            np.save(varname, b)        
            
            
            
#%%
moving_variance = reader.get_tensor(key) 
