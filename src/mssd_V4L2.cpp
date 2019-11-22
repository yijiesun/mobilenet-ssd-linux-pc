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

#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"
#include <sys/time.h>
#include <stdio.h>
#include "common.hpp"
#include "config.h"
#include "v4l2/v4l2.h"  

#define DEF_PROTO "models/MobileNetSSD_deploy.prototxt"
#define DEF_MODEL "models/MobileNetSSD_deploy.caffemodel"
#define DEF_IMAGE "tests/images/ssd_dog.jpg"

// struct Box
// {
//     float x0;
//     float y0;
//     float x1;
//     float y1;
//     int class_idx;
//     float score;
// };

V4L2 v4l2_;
  /************MASK-ROI************/
Mat mask;
  /************MASK-ROI************/
cv::Mat rgb;
bool quit;
 pthread_mutex_t mutex_;
void *v4l2_thread(void *threadarg);
void my_handler(int s);
// void get_input_data_ssd(std::string& image_file, float* input_data, int img_h,  int img_w)
void get_input_data_ssd(cv::Mat img, float* input_data, int img_h,  int img_w)
{
    // cv::Mat img = cv::imread(image_file);

    if (img.empty())
    {
        // std::cerr << "Failed to read image file " << image_file << ".\n";
        std::cerr << "Failed to read image from camera.\n";
        return;
    }

    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float *img_data = (float *)img.data;
    int hw = img_h * img_w;

    float mean[3]={127.5,127.5,127.5};
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = 0.007843* (*img_data - mean[c]);
                img_data++;
            }
        }
    }
}

// void post_process_ssd(std::string& image_file,float threshold,float* outdata,int num,std::string& save_name)
void post_process_ssd(cv::Mat &img,cv::Mat &show, float threshold,float* outdata,int num)
{
    const char* class_names[] = {"background",
                            "aeroplane", "bicycle", "bird", "boat",
                            "bottle", "bus", "car", "cat", "chair",
                            "cow", "diningtable", "dog", "horse",
                            "motorbike", "person", "pottedplant",
                            "sheep", "sofa", "train", "tvmonitor"};
    // cv::Mat img = cv::imread(image_file);
    int raw_h = img.size().height;
    int raw_w = img.size().width;
    std::vector<Box> boxes;
    int line_width=raw_w*0.002;
    printf("detect ruesult num: %d \n",num);
    for (int i=0;i<num;i++)
    {
        if(outdata[1]>=threshold)
        {
            Box box;
            box.class_idx=outdata[0];
            box.score=outdata[1];
            box.x0=outdata[2]*raw_w;
            box.y0=outdata[3]*raw_h;
            box.x1=outdata[4]*raw_w;
            box.y1=outdata[5]*raw_h;
            boxes.push_back(box);
            printf("%s\t:%.0f%%\n", class_names[box.class_idx], box.score * 100);
            printf("BOX:( %g , %g ),( %g , %g )\n",box.x0,box.y0,box.x1,box.y1);
        }
        outdata+=6;
    }
    for(int i=0;i<(int)boxes.size();i++)
    {
        Box box=boxes[i];
        cv::rectangle(show, cv::Rect(box.x0, box.y0,(box.x1-box.x0),(box.y1-box.y0)),cv::Scalar(255, 255, 0),line_width);
        std::ostringstream score_str;
        score_str<<box.score;
        std::string label = std::string(class_names[box.class_idx]) + ": " + score_str.str();
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(show, cv::Rect(cv::Point(box.x0,box.y0- label_size.height),
                                  cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 0), CV_FILLED);
        cv::putText(show, label, cv::Point(box.x0, box.y0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    // cv::imwrite(save_name,img);
    // std::cout<<"======================================\n";
    // std::cout<<"[DETECTED IMAGE SAVED]:\t"<< save_name<<"\n";
    // std::cout<<"======================================\n";
}

int main(int argc, char *argv[])
{
   quit = false;
    pthread_mutex_init(&mutex_, NULL);
    const std::string root_path = get_root_path();
    std::string proto_file;
    std::string model_file;
    std::string image_file;
    std::string save_name="save.jpg";

    struct sigaction sigIntHandler;
 
   sigIntHandler.sa_handler = my_handler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;
 
   sigaction(SIGINT, &sigIntHandler, NULL);

    int res;
    while( ( res=getopt(argc,argv,"p:m:i:h"))!= -1)
    {
        switch(res)
        {
            case 'p':
                proto_file=optarg;
                break;
            case 'm':
                model_file=optarg;
                break;
            case 'i':
                image_file=optarg;
                break;
            case 'h':
                std::cout << "[Usage]: " << argv[0] << " [-h]\n"
                          << "   [-p proto_file] [-m model_file] [-i image_file]\n";
                return 0;
            default:
                break;
        }
    }



    const char *model_name = "mssd_300";
    if(proto_file.empty())
    {
        proto_file = root_path + DEF_PROTO;
        std::cout<< "proto file not specified,using "<<proto_file<< " by default\n";

    }
    if(model_file.empty())
    {
        model_file = root_path + DEF_MODEL;
        std::cout<< "model file not specified,using "<<model_file<< " by default\n";
    }
    if(image_file.empty())
    {
        image_file = root_path + DEF_IMAGE;
        std::cout<< "image file not specified,using "<<image_file<< " by default\n";
    }

    std::string dev_num;
    get_param_mms_V4L2(dev_num);
    std::cout<<"open "<<dev_num<<std::endl;

    /************MASK-ROI************/
    bool is_roi_limit;
    get_roi_limit(is_roi_limit);
    std::cout<<"is_roi_limit: "<<is_roi_limit<<std::endl;
    if(is_roi_limit)
        mask=imread("mask.jpg");
    Mat process_frame,show_img;
    process_frame.create(480,640,CV_8UC3);
    show_img.create(480,640,CV_8UC3);
    /************MASK-ROI************/

    v4l2_.init(dev_num.c_str(),640,480);
    v4l2_.open_device();
	v4l2_.init_device();
	v4l2_.start_capturing();

    
    rgb.create(480,640,CV_8UC3);
	pthread_t threads_v4l2;
	int rc = pthread_create(&threads_v4l2, NULL, v4l2_thread, NULL);

    // init tengine
    init_tengine_library();
    if (request_tengine_version("0.1") < 0)
        return 1;
    if (load_model(model_name, "caffe", proto_file.c_str(), model_file.c_str()) < 0)
        return 1;
    std::cout << "load model done!\n";

    // create graph
    graph_t graph = create_runtime_graph("graph", model_name, NULL);
    if (!check_graph_valid(graph))
    {
        std::cout << "create graph0 failed\n";
        return 1;
    }

    // input
    int img_h = 300;
    int img_w = 300;
    int img_size = img_h * img_w * 3;
    float *input_data = (float *)malloc(sizeof(float) * img_size);

    cv::Mat frame,msk_color;
    frame.create(480,640,CV_8UC3);
    msk_color.create(480,640,CV_8UC3);
    

    int node_idx=0;
    int tensor_idx=0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if(!check_tensor_valid(input_tensor))
    {
        printf("Get input node failed : node_idx: %d, tensor_idx: %d\n",node_idx,tensor_idx);
        return 1;
    }

    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    prerun_graph(graph);

    int repeat_count = 1;
    const char *repeat = std::getenv("REPEAT_COUNT");

    if (repeat)
        repeat_count = std::strtoul(repeat, NULL, 10);

    float *outdata;
    int out_dim[4];
    while(1){
        struct timeval t0, t1;
        float total_time = 0.f;

        //pthread_mutex_lock(&mutex_);
         frame = rgb.clone();
        //pthread_mutex_unlock(&mutex_);

        /************MASK-ROI************/
        if(is_roi_limit)
        {
            bitwise_and(frame,mask,process_frame);
            addWeighted(frame,0.8,mask,0.3,-1,show_img);
        }
        else
        {
            process_frame = frame;
            show_img = frame.clone();
        }
        /************MASK-ROI************/
            
        
        for (int i = 0; i < repeat_count; i++)
        {
            get_input_data_ssd(process_frame, input_data, img_h,  img_w);

            gettimeofday(&t0, NULL);
            set_tensor_buffer(input_tensor, input_data, img_size * 4);
            run_graph(graph, 1);

            gettimeofday(&t1, NULL);
            float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
            total_time += mytime;

        }
        std::cout << "--------------------------------------\n";
        std::cout << "repeat " << repeat_count << " times, avg time per run is " << total_time / repeat_count << " ms\n";
        tensor_t out_tensor = get_graph_output_tensor(graph, 0,0);//"detection_out");
        get_tensor_shape( out_tensor, out_dim, 4);
        outdata = (float *)get_tensor_buffer(out_tensor);

        int num=out_dim[1];
        float show_threshold=0.5;
        post_process_ssd(process_frame, show_img, show_threshold, outdata, num);
        cv::imshow("MSSD", show_img);
        cv::waitKey(10) ;
        if (quit)
             break;
    }
    pthread_join(threads_v4l2,NULL);
	v4l2_.stop_capturing();
	v4l2_.uninit_device();
	v4l2_.close_device();
    postrun_graph(graph);
    free(input_data);
    destroy_runtime_graph(graph);
    remove_model(model_name);

    return 0;
}


void *v4l2_thread(void *threadarg)
{
	while (1)
	{
        //pthread_mutex_lock(&mutex_);
        v4l2_.read_frame(rgb);
        //pthread_mutex_unlock(&mutex_);
        cv::waitKey(10) ;
        if (quit)
            pthread_exit(NULL);
    }
}


void my_handler(int s)
{
            quit = true;
            cout<<"Caught signal "<<s<<" quit="<<quit<<endl;
}
 