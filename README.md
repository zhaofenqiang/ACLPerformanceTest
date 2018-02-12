# [Arm ComputeLibrary v18.01](https://github.com/ARM-software/ComputeLibrary/tree/v18.01) Performance Report  
v17.12 performance report is [here](https://github.com/zhaofenqiang/Test_ComputeLibrary).  
This report is tested on [RK3399](http://wiki.t-firefly.com/index.php/Firefly-RK3399) platform and the Arm ComputeLibrary's version is [**18.01**](https://github.com/ARM-software/ComputeLibrary/tree/v18.01).

## Models
Performance data were collected on [AlexNet](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/master/ComputeLibrary_v1801/examples/graph_alexnet.cpp), [GoogLeNet](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/master/ComputeLibrary_v1801/examples/graph_googlenet.cpp), [SqueezeNet_v1.0](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/master/ComputeLibrary_v1801/examples/graph_squeezenet.cpp), [MobileNet_v1_1_224](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/master/ComputeLibrary_v1801/examples/graph_mobilenet.cpp), [VGG16](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/master/ComputeLibrary_v1801/examples/graph_vgg16.cpp), [VGG19](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/master/ComputeLibrary_v1801/examples/graph_vgg19.cpp), [Inception-v3](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/master/ComputeLibrary_v1801/examples/graph_inception_v3.cpp) and [Inception-v4](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/master/ComputeLibrary_v1801/examples/graph_inception_v4.cpp).
And the pre-trained weights were provided by [caffe model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), and [tensorflow slim model](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).

## Methods    
Unlike v17.12 performance test, only Graph API was used to test the ComputeLibrary performance on v18.01.  
+ ACL Graph API   
The examples is located at [examples/graph_*.cpp](https://github.com/zhaofenqiang/ACLPerformanceTest/tree/master/ComputeLibrary_v1801/examples).  The result is an average time of more than 50 times inferences, but the standard deviation is still considerably very large, about 40%. The performacnce data were collected at a optimal developing board's working state. Here are the images used in the [log](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/master/ComputeLibrary_v1801/log/v1801_log), their correct label is dog, cat, pig respectively.     
![dog](https://raw.githubusercontent.com/zhaofenqiang/ComputeLibrary/master/data/images/dog.jpg)  ![cat](https://raw.githubusercontent.com/zhaofenqiang/ComputeLibrary/master/data/images/cat_227.jpg)  ![pig](https://raw.githubusercontent.com/zhaofenqiang/ACLPerformanceTest/master/ComputeLibrary_v1801/data/images/pig.jpg)  

    For simplicity, vgg16, vgg19, inception-v3, and inception-v4's convolution layer were executed by graph NEON DIRECT and OPENCL DIRECT convolution following the [official](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/fc99318d5062fe93455bedfec7e01e308aa02aff/ComputeLibrary_v1801/examples/graph_inception_v3.cpp#L56) default configuration:   
` 
TargetHint            target_hint      = set_target_hint(argc > 1 ? std::strtol(argv[1], nullptr, 10) : 0);    
  ConvolutionMethodHint convolution_hint = ConvolutionMethodHint::DIRECT;
  `

    And AlexNet, GoogLeNet, SqueezeNet and MobileNet’s convolution layer were executed by GEMM on NEON and DIRECT on OPENCL by [default](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/fc99318d5062fe93455bedfec7e01e308aa02aff/ComputeLibrary_v1801/examples/graph_alexnet.cpp#L57):  
` 
TargetHint            target_hint      = set_target_hint(argc > 1 ? std::strtol(argv[1], nullptr, 10) : 0);    
  ConvolutionMethodHint convolution_hint = target_hint == TargetHint::NEON ? ConvolutionMethodHint::GEMM : ConvolutionMethodHint::DIRECT;
  `

+  [CaffeOnACL](https://github.com/OAID/caffeOnACL)    
The CaffeOnACL performances are directly copied from [CaffeOnACL performance report](https://github.com/OAID/CaffeOnACL/blob/master/acl_openailab/performance_report.pdf)

##  Testing process    
1. Converted pre-trained weights by the scripts [here](https://github.com/zhaofenqiang/ACLPerformanceTest/tree/master/ComputeLibrary_v1801/scripts), although the code is a little ugly and probably can not work correctly. It's just for reference because the [raw scripts](https://github.com/ARM-software/ComputeLibrary/tree/master/scripts) has some problems as mentioned in [#324](https://github.com/ARM-software/ComputeLibrary/issues/324) and caffe_data_extractor.py could failed when a bolb has 3 weights. 
2. Build the ACL by  
`scons Werror=1 -j4 debug=0 asserts=1 neon=1 opencl=1 embed_kernels=1 os=linux arch=arm64-v8a benchmark_tests=1`   
3. Execute the program by   
`LD_LIBRARY_PATH=build ./build/examples/graph_*`
4. Collect the performace data printed on the screen.

# Performance  
Due to these issues [#354](https://github.com/ARM-software/ComputeLibrary/issues/354), [#356](https://github.com/ARM-software/ComputeLibrary/issues/356), [#357](https://github.com/ARM-software/ComputeLibrary/issues/357), there are still some problems with running cnn by graph API on ACL.

SO for now, the Inception-v3, Inception-v4 and MobileNet_v1_1_224's pre-trained weights can be extracted and imported successfully by the [caffe_data_extractor.py](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/master/ComputeLibrary_v1801/scripts/caffe_data_extractor.py) on caffe-model [here](https://github.com/soeaver/caffe-model) or [tensorflow_data_extractor_ckpt.py](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/master/ComputeLibrary_v1801/scripts/tensorflow_data_extractor_ckpt.py) on [tensorflow slim model](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models), but still can not infer correctly. So the inference time were instead collected by random filled weights or incorrect inference. However with this bug [#311](https://github.com/ARM-software/ComputeLibrary/issues/311) fixed in Arm ComputeLibrary v18.01, the inference time can be obtained by random filled weights accurately to a certain degree of confidence, although it's still a little smaller than the real inference time.   

And when testing Inception-v4 on CPU using GEMM libeary, the program would regularly be killed due to some unkown reason, probably because of insufficient memory, so the performace data was not collected.

###  Terminology   
- TPI: The total time for per inference, and the unit of all the data below is millisecond.  
- Graph-NE means using ACL Graph API's NEON library to execute program on CPU.
- Graph-CL means using ACL Graph API's OPENCL library to execute program on GPU.
- Caffe is a short for CaffeOnACL based method.

|   |AlexNet|	SqueezeNet	|GoogLeNet|	MobileNet|	vgg16	|vgg19	|Inception-v3|	Inception-v4
| - | :-: | :-: | :-: | :-: | :-: | :-: |  :-: |  :-: | 
|Graph-NE	|389.2|	189.7	|**358.2**	|615.8	|**3059.8**|	**3702.2**	|**1022**	|Memory N/A
|Graph-CL	|**251.2**	|305.3	|554.6	|**367.4**	|3909.4	|5083.2|	1421|	3276.6
|Caffe-NEON	|569.8	|229.5	|696.6	|485.6				
|Caffe-OpenBLAS	|857.2	|132.9	|1256.6	|281.5				
|Caffe-Mixed Model	|494.9	|**120.9**	|430.3	|264.9				
|Caffe-Opencl	|422.6	|535.5	|1236	|764.8			
