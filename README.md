# [Arm ComputeLibrary v18.01](https://github.com/ARM-software/ComputeLibrary/tree/v18.01) Performance Report  
A older performance report based on v17.12 can be found [here](https://github.com/zhaofenqiang/Test_ComputeLibrary).  
This report is tested on [RK3399](http://wiki.t-firefly.com/index.php/Firefly-RK3399) platform and the Arm Compute Library is version **18.01**. Performance data were collected on AlexNet, GoogLeNet, SqueezeNet_v1.0, MobileNet_v1_1_224, VGG16, VGG19, Inception-v3 and Inception-v4.

## Methods    
Unlike v17.12 performance test, only Graph API was used to represent the ComputeLibrary performance on v18.01.  
+ ACL Graph API   
The examples is located at [examples/graph_*.cpp](https://github.com/zhaofenqiang/ACLPerformanceTest/tree/master/ComputeLibrary_v1801/examples).  The result is an average time of more than 50 times inferences, but the standard deviation is still considerably very large, about 40%. The performacnce data were collected at a optimal develop board's working state. Here are the images used in the [log](), their correct label is dog, cat, bee respectively.     
![dog](https://raw.githubusercontent.com/zhaofenqiang/ComputeLibrary/master/data/images/dog.jpg)  ![cat](https://raw.githubusercontent.com/zhaofenqiang/ComputeLibrary/master/data/images/cat_227.jpg)  ![https://raw.githubusercontent.com/zhaofenqiang/ComputeLibrary/master/data/images/bee.jpg](https://raw.githubusercontent.com/zhaofenqiang/ComputeLibrary/master/data/images/bee.jpg)     

+  [CaffeOnACL](https://github.com/OAID/caffeOnACL)    
The CaffeOnACL performances are directly copy from [CaffeOnACL performance report](https://github.com/OAID/CaffeOnACL/blob/master/acl_openailab/performance_report.pdf)
For simplicity, vgg16, vgg19, inception-v3, and inception-v4's convolution layer were executed by graph NEON DIRECT and OPENCL DIRECT convolution following the [official](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/fc99318d5062fe93455bedfec7e01e308aa02aff/ComputeLibrary_v1801/examples/graph_inception_v3.cpp#L56) default configuration:   
` TargetHint            target_hint      = set_target_hint(argc > 1 ? std::strtol(argv[1], nullptr, 10) : 0);
        ConvolutionMethodHint convolution_hint = ConvolutionMethodHint::DIRECT;`

 And AlexNet, GoogLeNet, SqueezeNet and MobileNet’s convolution layer were executed by GEMM on NEON and DIRECT on OPENCL by [default](https://github.com/zhaofenqiang/ACLPerformanceTest/blob/fc99318d5062fe93455bedfec7e01e308aa02aff/ComputeLibrary_v1801/examples/graph_alexnet.cpp#L57):  
` TargetHint            target_hint      = set_target_hint(argc > 1 ? std::strtol(argv[1], nullptr, 10) : 0);
        ConvolutionMethodHint convolution_hint = target_hint == TargetHint::NEON ? ConvolutionMethodHint::GEMM : ConvolutionMethodHint::DIRECT;`

