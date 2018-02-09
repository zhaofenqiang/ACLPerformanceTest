/*
 * Copyright (c) 2017-2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Nodes.h"
#include "arm_compute/graph/SubGraph.h"
#include "support/ToolchainSupport.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include <cstdlib>
#include <tuple>

using namespace arm_compute::utils;
using namespace arm_compute::graph;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement InceptionV4's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to the weights folder, [optional] image, [optional] labels )
 */
class InceptionV4Example : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        std::string data_path; /* Path to the trainable data */
        std::string image;     /* Image data */
        std::string label;     /* Label data */

        constexpr float mean = 128.f;   /* Mean value to subtract from the channels */
        constexpr float std  = 255.f; /* Standard deviation value to divide from the channels */

        // Set target. 0 (NEON), 1 (OpenCL). By default it is NEON
        TargetHint            target_hint      = set_target_hint(argc > 1 ? std::strtol(argv[1], nullptr, 10) : 0);
        ConvolutionMethodHint convolution_hint = ConvolutionMethodHint::DIRECT;

        // Parse arguments
        if(argc < 2)
        {
            // Print help
            std::cout << "Usage: " << argv[0] << " [target] [path_to_data] [image] [labels]\n\n";
            std::cout << "No data folder provided: using random values\n\n";
        }
        else if(argc == 2)
        {
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " [path_to_data] [image] [labels]\n\n";
            std::cout << "No data folder provided: using random values\n\n";
        }
        else if(argc == 3)
        {
            data_path = argv[2];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " [image] [labels]\n\n";
            std::cout << "No image provided: using random values\n\n";
        }
        else if(argc == 4)
        {
            data_path = argv[2];
            image     = argv[3];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " [labels]\n\n";
            std::cout << "No text file with labels provided: skipping output accessor\n\n";
        }
        else
        {
            data_path = argv[2];
            image     = argv[3];
            label     = argv[4];
        }

        graph << target_hint << convolution_hint << Tensor(TensorInfo(TensorShape(299U, 299U, 3U, 1U), 1, DataType::F32),
                                                           get_input_accessor(image,
                                                                              mean, mean, mean,
                                                                              std, std, std, false /* Do not convert to BGR */))

              << ConvolutionLayer(3U, 3U, 32U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/conv1_3x3_s2_w.npy"),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(2, 2, 0, 0))
              << BatchNormalizationLayer(get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv4_model/conv1_3x3_s2_bn_w.npy"),
                                         get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv4_model/conv1_3x3_s2_bn_b.npy"),
                                         get_random_accessor(1.f, 1.f), get_weights_accessor(data_path,
                                                                                             "/cnn_data/inceptionv4_model/conv1_3x3_s2_scale_b.npy"),
                                         0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))

              << ConvolutionLayer(3U, 3U, 32U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/conv2_3x3_s1_w.npy"),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 0, 0))
              << BatchNormalizationLayer(get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv4_model/conv2_3x3_s1_bn_w.npy"),
                                         get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv4_model/conv2_3x3_s1_bn_b.npy"),
                                         get_random_accessor(1.f, 1.f), get_weights_accessor(data_path,
                                                                                             "/cnn_data/inceptionv4_model/conv2_3x3_s1_scale_b.npy"),
                                         0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))

              << ConvolutionLayer(3U, 3U, 64U,
                                  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/conv3_3x3_s1_w.npy"),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), PadStrideInfo(1, 1, 1, 1))
              << BatchNormalizationLayer(get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv4_model/conv3_3x3_s1_bn_w.npy"),
                                         get_weights_accessor(data_path,
                                                              "/cnn_data/inceptionv4_model/conv3_3x3_s1_bn_b.npy"),
                                         get_random_accessor(1.f, 1.f), get_weights_accessor(data_path,
                                                                                             "/cnn_data/inceptionv4_model/conv3_3x3_s1_scale_b.npy"),
                                         0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))

			<< get_stem_node_A(data_path, "inception_stem1")

			<< get_stem_node_B(data_path, "inception_stem2")

			<< get_stem_node_C(data_path, "inception_stem3")

			<< get_inception_node_A(data_path, "inception_a1")

			<< get_inception_node_A(data_path, "inception_a2")

			<< get_inception_node_A(data_path, "inception_a3")

			<< get_inception_node_A(data_path, "inception_a4")

			<< get_reduction_node_A(data_path, "reduction_a")

			<< get_inception_node_B(data_path, "inception_b1")

			<< get_inception_node_B(data_path, "inception_b2")

			<< get_inception_node_B(data_path, "inception_b3")

			<< get_inception_node_B(data_path, "inception_b4")

			<< get_inception_node_B(data_path, "inception_b5")

			<< get_inception_node_B(data_path, "inception_b6")

			<< get_inception_node_B(data_path, "inception_b7")

			<< get_reduction_node_B(data_path, "reduction_b")

			<< get_inception_node_C(data_path, "inception_c1")

			<< get_inception_node_C(data_path, "inception_c2")

			<< get_inception_node_C(data_path, "inception_c3")

			<< PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 8, PadStrideInfo(1, 1, 0, 0, DimensionRoundingType::CEIL)))

            << FullyConnectedLayer(
			  1000U,
			  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/classifier_w.npy"),
			  get_weights_accessor(data_path, "/cnn_data/inceptionv4_model/classifier_b.npy"))
		    << SoftmaxLayer()
		    << Tensor(get_output_accessor(label, 5));
    }

    void do_run() override
    {
        graph.run();
    }

private:
    Graph graph{};

private:
    BranchLayer get_stem_node_A(const std::string &data_path, std::string &&param_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";
        std::cout << total_path << std::endl;

        SubGraph i_c;
        i_c << ConvolutionLayer(
                    3U, 3U, 96U,
                    get_weights_accessor(data_path, total_path + "3x3_s2_w.npy"),
                    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                    PadStrideInfo(2, 2, 0, 0))
                << BatchNormalizationLayer(
                    get_weights_accessor(data_path, total_path + "3x3_s2_bn_w.npy"),
                    get_weights_accessor(data_path, total_path + "3x3_s2_bn_b.npy"),
                    get_random_accessor(1.f, 1.f),
                    get_weights_accessor(data_path, total_path + "3x3_s2_scale_b.npy"),
                    0.001f)
                << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

            SubGraph i_d;
            i_d << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));

            return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_c), std::move(i_d));
        }

    BranchLayer get_stem_node_B(const std::string &data_path, std::string &&param_path)
       {
           std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";
           std::cout << total_path << std::endl;

           SubGraph i_a;
           i_a << ConvolutionLayer(
        		   1U, 1U, 64U,
			   get_weights_accessor(data_path, total_path + "3x3_reduce_w.npy"),
				  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
				  PadStrideInfo(1, 1, 0, 0))
			  << BatchNormalizationLayer(
				  get_weights_accessor(data_path, total_path + "3x3_reduce_bn_w.npy"),
				  get_weights_accessor(data_path, total_path + "3x3_reduce_bn_b.npy"),
				  get_random_accessor(1.f, 1.f),
				  get_weights_accessor(data_path, total_path + "3x3_reduce_scale_b.npy"),
				  0.001f)
			  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
			  << ConvolutionLayer(
				  3U, 3U, 96U,
				  get_weights_accessor(data_path, total_path + "3x3_w.npy"),
				  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
				  PadStrideInfo(1, 1, 0, 0))
			  << BatchNormalizationLayer(
				  get_weights_accessor(data_path, total_path + "3x3_bn_w.npy"),
				  get_weights_accessor(data_path, total_path + "3x3_bn_b.npy"),
				  get_random_accessor(1.f, 1.f),
				  get_weights_accessor(data_path, total_path + "3x3_scale_b.npy"),
				  0.001f)
			  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));


           SubGraph i_b;
           i_b << ConvolutionLayer(
                   1U, 1U, 64,
                   get_weights_accessor(data_path, total_path + "1x7_reduce_w.npy"),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                   PadStrideInfo(1, 1, 0, 0))
               << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "1x7_reduce_bn_w.npy"),
                   get_weights_accessor(data_path, total_path + "1x7_reduce_bn_b.npy"),
                   get_random_accessor(1.f, 1.f),
                   get_weights_accessor(data_path, total_path + "1x7_reduce_scale_b.npy"),
                   0.001f)
               << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
               << ConvolutionLayer(
                   7U, 1U, 64,
                   get_weights_accessor(data_path, total_path + "1x7_w.npy"),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                   PadStrideInfo(1, 1, 3, 0))
               << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "1x7_bn_w.npy"),
                   get_weights_accessor(data_path, total_path + "1x7_bn_b.npy"),
                   get_random_accessor(1.f, 1.f),
                   get_weights_accessor(data_path, total_path + "1x7_scale_b.npy"),
                   0.001f)
               << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
               << ConvolutionLayer(
                   1U, 7U, 64,
                   get_weights_accessor(data_path, total_path + "7x1_w.npy"),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                   PadStrideInfo(1, 1, 0, 3))
               << BatchNormalizationLayer(
                   get_weights_accessor(data_path, total_path + "7x1_bn_w.npy"),
                   get_weights_accessor(data_path, total_path + "7x1_bn_b.npy"),
                   get_random_accessor(1.f, 1.f),
                   get_weights_accessor(data_path, total_path + "7x1_scale_b.npy"),
                   0.001f)
			  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
			  << ConvolutionLayer(
			      3U, 3U, 96U,
				  get_weights_accessor(data_path, total_path + "3x3_2_w.npy"),
				  std::unique_ptr < arm_compute::graph::ITensorAccessor> (nullptr),
				  PadStrideInfo(1, 1, 0, 0))
	            << BatchNormalizationLayer(
	                   get_weights_accessor(data_path, total_path + "3x3_2_bn_w.npy"),
	                   get_weights_accessor(data_path, total_path + "3x3_2_bn_b.npy"),
	                   get_random_accessor(1.f, 1.f),
	                   get_weights_accessor(data_path, total_path + "3x3_2_scale_b.npy"),
	                   0.001f)
               << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

               return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b));
           }

    BranchLayer get_stem_node_C(const std::string &data_path, std::string &&param_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";
        std::cout << total_path << std::endl;

        SubGraph i_a;
        i_a << ConvolutionLayer(
                    3U, 3U, 192U,
                    get_weights_accessor(data_path, total_path + "3x3_s2_w.npy"),
                    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                    PadStrideInfo(2, 2, 0, 0))
                << BatchNormalizationLayer(
                    get_weights_accessor(data_path, total_path + "3x3_s2_bn_w.npy"),
                    get_weights_accessor(data_path, total_path + "3x3_s2_bn_b.npy"),
                    get_random_accessor(1.f, 1.f),
                    get_weights_accessor(data_path, total_path + "3x3_s2_scale_b.npy"),
                    0.001f)
                << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

            SubGraph i_b;
            i_b << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)));

            return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b));
        }

    BranchLayer get_inception_node_A(const std::string &data_path, std::string &&param_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";
        std::cout << total_path << std::endl;

        SubGraph i_a;
        i_a << ConvolutionLayer(
                1U, 1U, 96U,
                get_weights_accessor(data_path, total_path + "1x1_2_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "1x1_2_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "1x1_2_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "1x1_2_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubGraph i_b;
        i_b << ConvolutionLayer(
                1U, 1U, 64U,
                get_weights_accessor(data_path, total_path + "3x3_reduce_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "3x3_reduce_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "3x3_reduce_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "3x3_reduce_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                3U, 3U, 96U,
                get_weights_accessor(data_path, total_path + "3x3_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 1, 1))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "3x3_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "3x3_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "3x3_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubGraph i_c;
        i_c << ConvolutionLayer(
                1U, 1U, 64U,
                get_weights_accessor(data_path, total_path + "3x3_2_reduce_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "3x3_2_reduce_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "3x3_2_reduce_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "3x3_2_reduce_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                3U, 3U, 96U,
                get_weights_accessor(data_path, total_path + "3x3_2_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 1, 1))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "3x3_2_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "3x3_2_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "3x3_2_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                3U, 3U, 96U,
                get_weights_accessor(data_path, total_path + "3x3_3_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 1, 1))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "3x3_3_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "3x3_3_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "3x3_3_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        //TODO PoolingLayerInfo true or false
        SubGraph i_d;
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true))
            << ConvolutionLayer(
                1U, 1U, 96U,
                get_weights_accessor(data_path, total_path + "1x1_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "1x1_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "1x1_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "1x1_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }


    BranchLayer get_reduction_node_A(const std::string &data_path, std::string &&param_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";
        std::cout << total_path << std::endl;

        SubGraph    i_a;
        i_a << ConvolutionLayer(
                3U, 3U, 384U,
                get_weights_accessor(data_path, total_path + "3x3_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(2, 2, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "3x3_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "3x3_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "3x3_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubGraph i_b;
        i_b << ConvolutionLayer(
                1U, 1U, 192U,
                get_weights_accessor(data_path, total_path + "3x3_2_reduce_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "3x3_2_reduce_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "3x3_2_reduce_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "3x3_2_reduce_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                3U, 3U, 224U,
                get_weights_accessor(data_path, total_path + "3x3_2_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 1, 1))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "3x3_2_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "3x3_2_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "3x3_2_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                3U, 3U, 256U,
                get_weights_accessor(data_path, total_path + "3x3_3_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(2, 2, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "3x3_3_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "3x3_3_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "3x3_3_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubGraph i_c;
        i_c << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
			  // TODO (geopin01) : Remove once we understand why a single node graph does not run in CL
			  << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, 1.f, 0.f));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c));
    }

    BranchLayer get_inception_node_B(const std::string &data_path, std::string &&param_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";
        std::cout << total_path << std::endl;

        SubGraph    i_a;
        i_a << ConvolutionLayer(
                1U, 1U, 384U,
                get_weights_accessor(data_path, total_path + "1x1_2_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "1x1_2_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "1x1_2_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "1x1_2_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubGraph i_b;
        i_b << ConvolutionLayer(
                1U, 1U, 192U,
                get_weights_accessor(data_path, total_path + "1x7_reduce_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "1x7_reduce_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "1x7_reduce_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "1x7_reduce_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                7U, 1U, 224U,
                get_weights_accessor(data_path, total_path + "1x7_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 3, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "1x7_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "1x7_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "1x7_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                1U, 7U, 256U,
                get_weights_accessor(data_path, total_path + "7x1_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 3))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "7x1_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "7x1_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "7x1_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubGraph i_c;
        i_c << ConvolutionLayer(
                1U, 1U, 192U,
                get_weights_accessor(data_path, total_path + "7x1_2_reduce_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "7x1_2_reduce_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "7x1_2_reduce_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "7x1_2_reduce_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                1U, 7U, 192,
                get_weights_accessor(data_path, total_path + "7x1_2_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 3))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "7x1_2_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "7x1_2_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "7x1_2_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                7U, 1U, 224U,
                get_weights_accessor(data_path, total_path + "1x7_2_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 3, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "1x7_2_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "1x7_2_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "1x7_2_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                1U, 7U, 224U,
                get_weights_accessor(data_path, total_path + "7x1_3_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 3))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "7x1_3_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "7x1_3_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "7x1_3_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                7U, 1U, 256U,
                get_weights_accessor(data_path, total_path + "1x7_3_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 3, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "1x7_3_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "1x7_3_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "1x7_3_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubGraph i_d;
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true))
            << ConvolutionLayer(
                1U, 1U, 128U,
                get_weights_accessor(data_path, total_path + "1x1_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "1x1_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "1x1_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "1x1_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }

    BranchLayer get_reduction_node_B(const std::string &data_path, std::string &&param_path)
      {
          std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";
          std::cout << total_path << std::endl;

          SubGraph    i_a;
          i_a << ConvolutionLayer(
                  1U, 1U, 192U,
                  get_weights_accessor(data_path, total_path + "3x3_reduce_w.npy"),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(1, 1, 0, 0))
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, total_path + "3x3_reduce_bn_w.npy"),
                  get_weights_accessor(data_path, total_path + "3x3_reduce_bn_b.npy"),
                  get_random_accessor(1.f, 1.f),
                  get_weights_accessor(data_path, total_path + "3x3_reduce_scale_b.npy"),
                  0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  3U, 3U, 192U,
                  get_weights_accessor(data_path, total_path + "3x3_w.npy"),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(2, 2, 0, 0))
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, total_path + "3x3_bn_w.npy"),
                  get_weights_accessor(data_path, total_path + "3x3_bn_b.npy"),
                  get_random_accessor(1.f, 1.f),
                  get_weights_accessor(data_path, total_path + "3x3_scale_b.npy"),
                  0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

          SubGraph i_b;
          i_b << ConvolutionLayer(
                  1U, 1U, 256U,
                  get_weights_accessor(data_path, total_path + "1x7_reduce_w.npy"),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(1, 1, 0, 0))
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, total_path + "1x7_reduce_bn_w.npy"),
                  get_weights_accessor(data_path, total_path + "1x7_reduce_bn_b.npy"),
                  get_random_accessor(1.f, 1.f),
                  get_weights_accessor(data_path, total_path + "1x7_reduce_scale_b.npy"),
                  0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  7U, 1U, 256U,
                  get_weights_accessor(data_path, total_path + "1x7_w.npy"),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(1, 1, 3, 0))
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, total_path + "1x7_bn_w.npy"),
                  get_weights_accessor(data_path, total_path + "1x7_bn_b.npy"),
                  get_random_accessor(1.f, 1.f),
                  get_weights_accessor(data_path, total_path + "1x7_scale_b.npy"),
                  0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  1U, 7U, 320U,
                  get_weights_accessor(data_path, total_path + "7x1_w.npy"),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(1, 1, 0, 3))
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, total_path + "7x1_bn_w.npy"),
                  get_weights_accessor(data_path, total_path + "7x1_bn_b.npy"),
                  get_random_accessor(1.f, 1.f),
                  get_weights_accessor(data_path, total_path + "7x1_scale_b.npy"),
                  0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << ConvolutionLayer(
                  3U, 3U, 320U,
                  get_weights_accessor(data_path, total_path + "3x3_2_w.npy"),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(2, 2, 0, 0))
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, total_path + "3x3_2_bn_w.npy"),
                  get_weights_accessor(data_path, total_path + "3x3_2_bn_b.npy"),
                  get_random_accessor(1.f, 1.f),
                  get_weights_accessor(data_path, total_path + "3x3_2_scale_b.npy"),
                  0.001f)
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

          SubGraph i_c;
          i_c << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
              // TODO (geopin01) : Remove once we understand why a single node graph does not run in CL
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, 1.f, 0.f));

          return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c));
      }

    BranchLayer get_inception_node_C(const std::string &data_path, std::string &&param_path)
    {
        std::string total_path = "/cnn_data/inceptionv4_model/" + param_path + "_";
        std::cout << total_path << std::endl;

        SubGraph    i_a;
        i_a << ConvolutionLayer(
                1U, 1U, 256U,
                get_weights_accessor(data_path, total_path + "1x1_2_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "1x1_2_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "1x1_2_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "1x1_2_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubGraph i_b1;
        i_b1 << ConvolutionLayer(
                 3U, 1U, 256U,
                 get_weights_accessor(data_path, total_path + "1x3_w.npy"),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 1, 0))
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "1x3_bn_w.npy"),
                 get_weights_accessor(data_path, total_path + "1x3_bn_b.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "1x3_scale_b.npy"),
                 0.001f)
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubGraph i_b2;
        i_b2 << ConvolutionLayer(
                 1U, 3U, 256U,
                 get_weights_accessor(data_path, total_path + "3x1_w.npy"),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 0, 1))
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "3x1_bn_w.npy"),
                 get_weights_accessor(data_path, total_path + "3x1_bn_b.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "3x1_scale_b.npy"),
                 0.001f)
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubGraph i_b;
        i_b << ConvolutionLayer(
                1U, 1U, 384U,
                get_weights_accessor(data_path, total_path + "1x1_3_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "1x1_3_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "1x1_3_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "1x1_3_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_b1), std::move(i_b2));

        SubGraph i_c1;
        i_c1 << ConvolutionLayer(
                 3U, 1U, 256U,
                 get_weights_accessor(data_path, total_path + "1x3_3_w.npy"),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 1, 0))
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "1x3_3_bn_w.npy"),
                 get_weights_accessor(data_path, total_path + "1x3_3_bn_b.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "1x3_3_scale_b.npy"),
                 0.001f)
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubGraph i_c2;
        i_c2 << ConvolutionLayer(
                 1U, 3U, 256U,
                 get_weights_accessor(data_path, total_path + "3x1_3_w.npy"),
                 std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                 PadStrideInfo(1, 1, 0, 1))
             << BatchNormalizationLayer(
                 get_weights_accessor(data_path, total_path + "3x1_3_bn_w.npy"),
                 get_weights_accessor(data_path, total_path + "3x1_3_bn_b.npy"),
                 get_random_accessor(1.f, 1.f),
                 get_weights_accessor(data_path, total_path + "3x1_3_scale_b.npy"),
                 0.001f)
             << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubGraph i_c;
        i_c << ConvolutionLayer(
                1U, 1U, 384U,
                get_weights_accessor(data_path, total_path + "1x1_4_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "1x1_4_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "1x1_4_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "1x1_4_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << ConvolutionLayer(
                1U, 3U, 448U,
                get_weights_accessor(data_path, total_path + "3x1_2_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 1))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "3x1_2_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "3x1_2_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "3x1_2_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
			<< ConvolutionLayer(
				3U, 1U, 512U,
				get_weights_accessor(data_path, total_path + "1x3_2_w.npy"),
				std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
				PadStrideInfo(1, 1, 1, 0))
			<< BatchNormalizationLayer(
				get_weights_accessor(data_path, total_path + "1x3_2_bn_w.npy"),
				get_weights_accessor(data_path, total_path + "1x3_2_bn_b.npy"),
				get_random_accessor(1.f, 1.f),
				get_weights_accessor(data_path, total_path + "1x3_2_scale_b.npy"),
				0.001f)
			<< ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
            << BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_c1), std::move(i_c2));

        SubGraph i_d;
        i_d << PoolingLayer(PoolingLayerInfo(PoolingType::AVG, 3, PadStrideInfo(1, 1, 1, 1, DimensionRoundingType::CEIL), true))
            << ConvolutionLayer(
                1U, 1U, 256U,
                get_weights_accessor(data_path, total_path + "1x1_w.npy"),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "1x1_bn_w.npy"),
                get_weights_accessor(data_path, total_path + "1x1_bn_b.npy"),
                get_random_accessor(1.f, 1.f),
                get_weights_accessor(data_path, total_path + "1x1_scale_b.npy"),
                0.001f)
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b), std::move(i_c), std::move(i_d));
    }
};

/** Main program for Inception V4
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL), [optional] Path to the weights folder, [optional] image, [optional] labels )
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<InceptionV4Example>(argc, argv);
}
