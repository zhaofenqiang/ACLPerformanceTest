[Arm ComputeLibrary v17.12](https://github.com/ARM-software/ComputeLibrary/tree/v17.12) Performance Report    
=====    
A new performance report based on v18.01 can be found [here](https://github.com/zhaofenqiang/ACLPerformanceTest).   
This report is tested on [RK3399](http://wiki.t-firefly.com/index.php/Firefly-RK3399) platform and the Arm Compute Library is version 17.12. Performance data were collected on AlexNet, GoogLeNet, SqueezeNet, MobileNet, VGG16, and VGG19.     

## Methods     
There are several methods to evaluate the performance.
+  [ACL Benchmark test](https://arm-software.github.io/ComputeLibrary/v17.12/tests.xhtml)   
The benchmark test is released at ACL v17.09 as a custom-made testing and validation suite, and for now(17.12) the implemented model only has LeNet, AlexNet and MobileNet. I run 10 iterations for every test, and last result is the average value. Due to its inflexibility and inconvenience, I only test it for AlexNet. It’s located at [ComputeLibrary/tests/benchmark](https://github.com/zhaofenqiang/ACLPerformanceTest/tree/master/ComputeLibrary_v1712/tests/benchmark)   
 
+ ACL Graph API   
It’s located at [ComputeLibrary/arm_compute/graph](https://github.com/zhaofenqiang/ACLPerformanceTest/tree/master/ComputeLibrary_v1712/arm_compute/graph), and the examples is located at [examples/graph_*.cpp](https://github.com/zhaofenqiang/ACLPerformanceTest/tree/master/ComputeLibrary_v1712/examples). The Graph API is a convenient tool for building networks, although the extendibility and convenience is still worse than the popular DL framework. The result is an average time of more than 50 times guaranteed correct inferences, and the standard deviation is considerably very large, about 40%. Here are the images used in the [log](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/master/ComputeLibrary_v1712/log/final_log), their correct label is dog, cat, bee respectively.     
![dog](https://raw.githubusercontent.com/zhaofenqiang/ACLPerformanceTest/master/ComputeLibrary_v1712/data/images/dog.jpg)  ![cat](https://raw.githubusercontent.com/zhaofenqiang/ACLPerformanceTest/master/ComputeLibrary_v1712/data/images/cat_227.jpg)  ![https://raw.githubusercontent.com/zhaofenqiang/ComputeLibrary/master/data/images/bee.jpg](https://raw.githubusercontent.com/zhaofenqiang/ACLPerformanceTest/master/ComputeLibrary_v1712/data/images/bee.jpg)     

+  [CaffeOnACL](https://github.com/OAID/caffeOnACL)    
The CaffeOnACL performances are directly copy from [CaffeOnACL performance report](https://github.com/OAID/CaffeOnACL/blob/master/acl_openailab/performance_report.pdf)

## Models
The test was performed on AlexNet, GoogLeNet, SqueezeNet, MobileNet, VGG16, and VGG19.    
The pre-trained weights were provided by [caffe model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), except MobileNet was download from [tensorflow slim model](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)

##  Testing process    
1. Converted pre-trained weights by [caffe_data_extractor.py](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/master/ComputeLibrary_v1712/scripts/caffe_data_extractor.py) or [tensorflow_data_extractor.py](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/master/ComputeLibrary_v1712/scripts/tensorflow_data_extractor.py). **Note**: The scripts has some problems mentioned in [#324](https://github.com/ARM-software/ComputeLibrary/issues/324) and caffe_data_extractor.py could failed when a bolb has 3 weights. You can find the revised script [here](https://github.com/zhaofenqiang/ACLPerformanceTest/tree/master/ComputeLibrary_v1801/scripts).   
2. Build the ACL by `scons Werror=1 -j4 debug=0 asserts=1 neon=1 opencl=1 embed_kernels=1 os=linux arch=arm64-v8a benchmark_tests=1`   
3. Execute the program by `LD_LIBRARY_PATH=build ./build/examples/graph_*`
4. Collect the performace data printed on the screen.


# Performance  
Dut to some Arm ComputeLibrary v17.12 bugs: [#331](https://github.com/ARM-software/ComputeLibrary/issues/331), [#323](https://github.com/ARM-software/ComputeLibrary/issues/323), [#311](https://github.com/ARM-software/ComputeLibrary/issues/311), the performacne data is not very stable and accurate, just for reference in ACL V17.12.
A new performance report based on v18.01 can be found [here](https://github.com/zhaofenqiang/ACLPerformanceTest).   
###  Terminology  
- TPI: The total time for per inference, and the unit of all the data below is millisecond.  
- CLDirectConv means using CLDirectConvolutionLayer for 1×1,3×3,5×5 convolution layers, and CLConvolutionLayer for other convolution layers.  
- NEDirectConv means using NEDirectConvolutionLayer for 1×1,3×3,5×5 convolution layers, and NEConvolutionLayer for other convolution layers.
- CLConv meas means using CLConvolutionLayer for all convolution by GEMM.  
- NEConv meas means using NEConvolutionLayer for all convolution by GEMM.  
- BM is a short for ACL benchmark method.  
- Graph is a short for ACL graph method.  
- Caffe is a short for CaffeOnACL based method.

|   | AlexNet | GoogLeNet | SqueezeNet	| MobileNet	 | VGG16	| VGG19
| - | :-: | -: | :-: | -: | :-: | -: | 
| Graph-CLDirectConv | 248.4	 | 434.8	 | 309.6	 | 371.8 | 	3997.8 | 	5081.7
| Graph-CLConv |231.3	 | 348.6	 | 244.6	 | 365.8
| Graph-NEDirectConv | 726.9	 | 957.2	 | 764.7 | 	764.7 | 	3463.7 | 	4344.0
| Graph-NEConv |667.8	 | 1278.3	 | 545.9	 | 1325.3
|Caffe-NEON	 | 569.8	 | 696.6	 | 229.5	 | 485.6
|Caffe-OpenBLAS | 	857.2	 | 1256.6 | 	132.9	 | 281.5
|Caffe-Mixed Model	 | 494.9	 | 430.3	 | 120.9	 | 264.9
|Caffe-Opencl	 | 422.6	 | 1236.0	 | 535.5 | 	764.8


