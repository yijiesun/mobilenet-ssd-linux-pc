/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */

#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"
#include "cpu_device.h"
#include "common.hpp"
#include "config.h"
#define DEF_PROTO "models/MobileNetSSD_deploy.prototxt"
#define DEF_MODEL "models/MobileNetSSD_deploy.caffemodel"
#define THREAD_CNT 1
using namespace cv;
using namespace std;

cv::Mat frame;
cv::Mat frame_[THREAD_CNT];
tensor_t input_tensor[THREAD_CNT];
int img_h;
int img_w;
int img_size;
float* input_data[THREAD_CNT];
tensor_t out_tensor[THREAD_CNT];
int thread_num = 0;
std::string image_file;
graph_t graph[THREAD_CNT];
const char* class_names[] = {"background", "aeroplane", "bicycle",   "bird",   "boat",        "bottle",
                                "bus",        "car",       "cat",       "chair",  "cow",         "diningtable",
                                "dog",        "horse",     "motorbike", "person", "pottedplant", "sheep",
                                "sofa",       "train",     "tvmonitor"};
void get_input_data_ssd(Mat& img, float* input_data, int img_h, int img_w)
{
    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float* img_data = ( float* )img.data;
    int hw = img_h * img_w;

    float mean[3] = {127.5, 127.5, 127.5};
    for(int h = 0; h < img_h; h++)
    {
        for(int w = 0; w < img_w; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = 0.007843 * (*img_data - mean[c]);
                img_data++;
            }
        }
    }
}

void post_process_ssd(Mat& img, float threshold, float* outdata, int num)
{
    int raw_h = frame.size().height;
    int raw_w = frame.size().width;
    std::vector<Box> boxes;
    int line_width = raw_w * 0.005;
    printf("detect result num: %d \n", num);
    for(int i = 0; i < num; i++)
    {
        if(outdata[1] >= threshold)
        {
            Box box;
            box.class_idx = outdata[0];
            box.score = outdata[1];
            box.x0 = outdata[2] * raw_w;
            box.y0 = outdata[3] * raw_h;
            box.x1 = outdata[4] * raw_w;
            box.y1 = outdata[5] * raw_h;
            boxes.push_back(box);
            //printf("%s\t:%.0f%%\n", class_names[box.class_idx], box.score * 100);
            //printf("BOX:( %g , %g ),( %g , %g )\n", box.x0, box.y0, box.x1, box.y1);
        }
        outdata += 6;
    }
    for(int i = 0; i < ( int )boxes.size(); i++)
    {
        Box box = boxes[i];
        cv::rectangle(frame, cv::Rect(box.x0, box.y0, (box.x1 - box.x0), (box.y1 - box.y0)), cv::Scalar(255, 255, 0),
                      line_width);
        std::ostringstream score_str;
        score_str << box.score;
        std::string label = std::string(class_names[box.class_idx]) + ": " + score_str.str();
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(frame,
                      cv::Rect(cv::Point(box.x0, box.y0 - label_size.height),
                               cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 0), CV_FILLED);
        cv::putText(frame, label, cv::Point(box.x0, box.y0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}

void mssd_core(Mat &img,graph_t &graph, int thread_num, float* input_data,tensor_t &input_tensor, tensor_t &out_tensor )
{
    struct timeval t0, t1;
    float total_time = 0.f;
    gettimeofday(&t0, NULL);
    get_input_data_ssd(img, input_data, img_h, img_w);
    set_tensor_buffer(input_tensor, input_data, img_size * 4);
    run_graph(graph, 1);
    
    out_tensor = get_graph_output_tensor(graph, 0, 0); 
    int out_dim[4];
    get_tensor_shape(out_tensor, out_dim, 4);
    float* outdata = ( float* )get_tensor_buffer(out_tensor);
    int num = out_dim[1];
    float show_threshold = 0.5;

    post_process_ssd(img, show_threshold, outdata, num);

    gettimeofday(&t1, NULL);
    float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;

    std::cout <<"thread " << thread_num << " times  " << mytime << "ms\n";
}

void cpu_thread(int cpu_num)
{
     mssd_core(frame_[cpu_num],graph[cpu_num], cpu_num,input_data[cpu_num],input_tensor[cpu_num],out_tensor[cpu_num]);
}

int main(int argc, char* argv[])
{
    std::string in_video_file;
    std::string out_video_file;
    get_param_mssd_video_knn(in_video_file,out_video_file);
    std::cout<<"input video: "<<in_video_file<<"\noutput video: "<<out_video_file<<std::endl;

    cv::VideoCapture capture;
    capture.open(in_video_file.c_str());
    capture.set(CV_CAP_PROP_FOURCC, cv::VideoWriter::fourcc ('M', 'J', 'P', 'G'));
    const std::string root_path = get_root_path();
    std::string proto_file;
    std::string model_file;
    const char* pproto_file;
    const char* pmodel_file;
    proto_file = root_path + DEF_PROTO;
    model_file = root_path + DEF_MODEL;

    /* do not let GPU run concat */
    setenv("GPU_CONCAT", "0", 1);
    /* using GPU fp16 */
    setenv("ACL_FP16", "1", 1);
    /* default CPU device using 0,1,2,3 */
    setenv("TENGINE_CPU_LIST", "2", 1);
    /* using fp32 or int8 */
    setenv("KERNEL_MODE", "2", 1);

    // init tengine
    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return -1;

    img_h = 300;
    img_w = 300;
    img_size = img_h * img_w * 3;
    pproto_file = proto_file.c_str();
    pmodel_file = model_file.c_str();
    int node_idx = 0;
    int tensor_idx = 0;
    int dims[] = {1, 3, img_h, img_w};
    // thread 0 for cpu 2A72
    const struct cpu_info* p_info = get_predefined_cpu("rk3399");
    int a72_list[] = {4};
    set_online_cpu(( struct cpu_info* )p_info, a72_list, sizeof(a72_list) / sizeof(int));
    create_cpu_device("a72", p_info);
    int a72_list01[] = {5};
    set_online_cpu(( struct cpu_info* )p_info, a72_list01, sizeof(a72_list01) / sizeof(int));
    create_cpu_device("a7201", p_info);
    // thread 3 for cpu 4A53
    int a53_list[] = {0, 1};
    set_online_cpu(( struct cpu_info* )p_info, a53_list, sizeof(a53_list) / sizeof(int));
    create_cpu_device("a53", p_info);
    int a53_list1[] = {2, 3};
    set_online_cpu(( struct cpu_info* )p_info, a53_list1, sizeof(a53_list1) / sizeof(int));
    create_cpu_device("a5301", p_info);

    for(int i=0;i<THREAD_CNT;i++)
    {
        graph[i] = create_graph(NULL, "caffe", pproto_file, pmodel_file);
        input_data[i] = ( float* )malloc(sizeof(float) * img_size);
        input_tensor[i] = get_graph_input_tensor(graph[i], node_idx, tensor_idx);
        if(input_tensor[i] == nullptr)
            printf("Get input node failed : node_idx: %d, tensor_idx: %d\n", node_idx, tensor_idx);   
        set_tensor_shape(input_tensor[i], dims, 4);
    }

#if (THREAD_CNT>=1)
    set_graph_device(graph[0], "acl_opencl");
#endif
#if (THREAD_CNT>=2)
    if(set_graph_device(graph[1], "a72") < 0)
        std::cerr << "set device a72 failed\n";
#endif
#if (THREAD_CNT>=3)
    if(set_graph_device(graph[2], "a53") < 0)
        std::cerr << "set device a53 failed\n";
#endif
#if (THREAD_CNT>=4)
    if(set_graph_device(graph[3], "a7201") < 0)
         std::cerr << "set device a7201 failed\n";
#endif
#if (THREAD_CNT>=5)
    if(set_graph_device(graph[4], "a5301") < 0)
            std::cerr << "set device a5301 failed\n";
#endif

    for(int i=0;i<THREAD_CNT;i++)
    {
        int ret_prerun = prerun_graph(graph[i]);
        if(ret_prerun < 0)
            std::printf("prerun failed\n"); 
    }

    while(1){

          if (!capture.read(frame))
		{
			cout<<"cannot open video or end of video"<<endl;
            break;
		}
 
        struct timeval t0_, t1_;
        float total_time = 0.f;
        std::thread* t[THREAD_CNT];
        gettimeofday(&t0_, NULL);
       for(int i=0;i<THREAD_CNT;i++)
       {
            frame_[i] = frame.clone();
            t[i] = new std::thread(cpu_thread, i);
       }
        for(int i=0;i<THREAD_CNT;i++)
       {
            t[i]->join();
            delete t[i];
       }

        gettimeofday(&t1_, NULL);
        float mytime = ( float )((t1_.tv_sec * 1000000 + t1_.tv_usec) - (t0_.tv_sec * 1000000 + t0_.tv_usec)) / 1000;
        std::cout <<"thread_done"  << " times  " << mytime << "ms\n";
        std::cout <<" --------------------------------------------------------------------------\n";
        cv::imshow("MSSD", frame);
        cv::waitKey(10) ;

    }
    for(int i=0;i<THREAD_CNT;i++)
    {
        release_graph_tensor(out_tensor[i]);
        release_graph_tensor(input_tensor[i]);
        postrun_graph(graph[i]);
        destroy_graph(graph[i]);
        free(input_data[i]);
    }
    release_tengine();

    return 0;
}
