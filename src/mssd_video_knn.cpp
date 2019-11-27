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
#include "knn/knn.h"
#include "config.h"

#define DEF_PROTO "models/MobileNetSSD_deploy.prototxt"
#define DEF_MODEL "models/MobileNetSSD_deploy.caffemodel"
#define DEF_IMAGE "tests/images/ssd_dog.jpg"
#define DEF_VIDEO_IN "tests/test0.avi"
#define DEF_VIDEO_OUT "tests/result_test0.avi"

#define CLIP(a,b,c) (  (a) = (a)>(c)?(c):((a)<(b)?(b):(a))  )

using namespace cv;
using namespace std;


// struct Box
// {
//     float x0;
//     float y0;
//     float x1;
//     float y1;
//     int class_idx;
//     float score;
// };
const char* class_names[] = {"background",
                        "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair",
                        "cow", "diningtable", "dog", "horse",
                        "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"};
vector<BoxInROI>	boxes_in_roi; 
vector<Box>	boxes_ruff_all; 
vector<Box>	boxes_ruff; 
vector<Box>	boxes; 
vector<Box>	boxes_all; 
int IMG_WID,IMG_HGT;
  /************MASK-ROI************/
Mat mask;
  /************MASK-ROI************/
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
void post_process_ssd(cv::Mat img, float threshold,float* outdata,int num)
{

    // cv::Mat img = cv::imread(image_file);
    int raw_h = img.size().height;
    int raw_w = img.size().width;
    boxes.clear();
    boxes_ruff.clear();
    int line_width=raw_w*0.002;
    printf("detect ruesult num: %d \n",num);
    for (int i=0;i<num;i++)
    {
        if(outdata[0]==15)
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
            if(outdata[1]>=threshold/2)
            {
                Box box;
                box.class_idx=outdata[0];
                box.score=outdata[1];
                box.x0=outdata[2]*raw_w;
                box.y0=outdata[3]*raw_h;
                box.x1=outdata[4]*raw_w;
                box.y1=outdata[5]*raw_h;
                boxes_ruff.push_back(box);
            }
            outdata+=6;
        }

    }
 #if 0
    for(int i=0;i<(int)boxes.size();i++)
    {
        Box box=boxes[i];
        cv::rectangle(img, cv::Rect(box.x0, box.y0,(box.x1-box.x0),(box.y1-box.y0)),cv::Scalar(255, 255, 0),line_width);
        std::ostringstream score_str;
        score_str<<box.score;
        std::string label = std::string(class_names[box.class_idx]) + ": " + score_str.str();
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(img, cv::Rect(cv::Point(box.x0,box.y0- label_size.height),
                                  cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 0), CV_FILLED);
        cv::putText(img, label, cv::Point(box.x0, box.y0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
#endif
    // cv::imwrite(save_name,img);
    // std::cout<<"======================================\n";
    // std::cout<<"[DETECTED IMAGE SAVED]:\t"<< save_name<<"\n";
    // std::cout<<"======================================\n";
}
void init_video_knn(KNN_BGS &knn_bgs,int *knn_conf,int cnt)
{

        knn_bgs.IMG_WID = IMG_WID;
		knn_bgs.IMG_HGT = IMG_HGT;
        knn_bgs.set(knn_conf);
        knn_bgs.pos = 0;
        //v_capture.read(knn_bgs.bk);
        knn_bgs.knn_box_exist_cnt = cnt;
        knn_bgs.useTopRect = knn_conf[8];
		knn_bgs.knn_over_percent = 0.001f;
		knn_bgs.tooSmalltoDrop = knn_conf[9];
		knn_bgs.dilateRatio =  knn_bgs.IMG_WID  / 320 * 5;
        knn_bgs.init();

}
void togetherAllBox(double zoom_value,int x0,int y0,vector<Box> &b0,vector<Box> &b_all )
{
	for (int i = 0; i<b0.size(); i++) {
		float		bx0 = b0[i].x0, by0 = b0[i].y0, bx1= b0[i].x1, by1 = b0[i].y1;
			b0[i].x0= bx0 / zoom_value + x0;
			b0[i].y0 = by0 / zoom_value + y0;
			b0[i].x1 = bx1 / zoom_value + x0;
			b0[i].y1 = by1/ zoom_value + y0;
            CLIP(b0[i].x0,0,IMG_WID-1);
            CLIP(b0[i].y0,0,IMG_HGT-1);
            CLIP(b0[i].x1,0,IMG_WID-1);
            CLIP(b0[i].y1,0,IMG_HGT-1);
		   b_all.push_back(b0[i]);
	}
}

void draw_img(Mat &img)
{
    int line_width=300*0.002;
    for(int i=0;i<(int)boxes_all.size();i++)
    {
        Box box=boxes_all[i];
        cv::rectangle(img, cv::Rect(box.x0, box.y0,(box.x1-box.x0),(box.y1-box.y0)),cv::Scalar(255, 255, 0),line_width);
        std::ostringstream score_str;
        score_str<<box.score;
        std::string label = std::string(class_names[box.class_idx]) + ": " + score_str.str();
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(img, cv::Rect(cv::Point(box.x0,box.y0- label_size.height),
                                  cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 0), CV_FILLED);
        cv::putText(img, label, cv::Point(box.x0, box.y0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

}

int main(int argc, char *argv[])
{
    Mat background,background_mask,frame_mask;
    //background = imread("bk.png");
    const std::string root_path = get_root_path();
    std::string proto_file;
    std::string model_file;
    std::string image_file;
    std::string save_name="save.jpg";
    KNN_BGS knn_bgs;
    float show_threshold=0.5;
    //solid_frame    history_num   knnv  padSize   minContorSize  ( ? )  insideDilate_win_size   insideDilate_scale  useTopRect  tooSmalltoDrop
    // 0, 2, 1, 5, 0, 2, 4, 1, 10, 0
    int knn_conf[10] = { 0, 2, 1, 5, 0, 2, 4, 1, 5, 0};

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
    int x0,y0,w0,h0;
    int x0_,y0_,w0_,h0_;
    int img_size = img_h * img_w * 3;
    float *input_data = (float *)malloc(sizeof(float) * img_size);

    /************MASK-ROI************/
    bool is_roi_limit = 0;
    get_roi_limit(is_roi_limit);
    std::cout<<"is_roi_limit: "<<is_roi_limit<<std::endl;
    if(is_roi_limit)
        mask=imread("mask.jpg");

    /************MASK-ROI************/

    std::string in_video_file =  root_path + DEF_VIDEO_IN;
    std::string out_video_file =  root_path + DEF_VIDEO_OUT;
    get_param_mssd_video_knn(in_video_file,out_video_file);
    std::cout<<"input video: "<<in_video_file<<"\noutput video: "<<out_video_file<<std::endl;

    int knn_box_exist_cnt;
    get_knn_box_exist_cnt(knn_box_exist_cnt);
    std::cout<<"knn_box_exist_cnt: "<<knn_box_exist_cnt<<std::endl;

    double knn_thresh;
    get_knn_thresh(knn_thresh);
    std::cout<<"knn_thresh: "<<knn_thresh<<std::endl;

    cv::Mat frame,diff_mat;
    cv::VideoCapture capture;
    VideoWriter outputVideo;
    capture.open(in_video_file.c_str());
    capture.set(CV_CAP_PROP_FOURCC, cv::VideoWriter::fourcc ('M', 'J', 'P', 'G'));
    //capture.set(CV_CAP_PROP_POS_FRAMES, 10);

    IMG_WID = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    IMG_HGT = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    background=Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);
    //Size sWH = Size( IMG_WID*2, IMG_HGT*2);
     Size sWH = Size( IMG_WID, IMG_HGT);
    bool ret = outputVideo.open( out_video_file.c_str(), cv::VideoWriter::fourcc ('M', 'P', '4', '2'), 25, sWH);

    
    init_video_knn(knn_bgs,knn_conf,knn_box_exist_cnt);
    knn_bgs.knn_thresh = knn_thresh;
   
    diff_mat.create(IMG_HGT,IMG_WID,CV_8UC1);

    Mat process_frame,show_img;
    process_frame.create(IMG_HGT,IMG_WID,CV_8UC3);
    show_img.create(IMG_HGT,IMG_WID,CV_8UC3);
    show_img = Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);
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
    bool first =true;
    int aa =0;
    while(1){
        
        struct timeval t0, t1;
        float total_time = 0.f;

      if (!capture.read(frame))
		{
			cout<<"cannot open video or end of video"<<endl;
            break;
		}
        if(first)
        {
            aa++;
            if(aa >= 10)
            {
                background = frame.clone();
                knn_bgs.bk = frame.clone();
                first = false;
            }
            else
            {
                outputVideo.write(show_img);
                continue;
            }
               
        }
        /************MASK-ROI************/
        if(is_roi_limit)
        {
            bitwise_and(background,mask,background_mask);
            bitwise_and(frame,mask,frame_mask);
            bitwise_and(frame,mask,process_frame);
            addWeighted(frame,0.8,mask,0.3,-1,show_img);
        }
        else
        {
            process_frame = frame;
            show_img = frame.clone();
        }
        /************MASK-ROI************/
      
        knn_bgs.frame = process_frame.clone();
        knn_bgs.pos ++;
        knn_bgs.boundRect.clear();
        knn_bgs.diff2(process_frame, knn_bgs.bk);
        knn_bgs.knn_core();
        knn_bgs.processRects(boxes_all);
       
        boxes_ruff_all.clear();
        boxes_in_roi.clear();
        boxes_all.clear();
        boxes.clear();
        for (int i = 0; i< knn_bgs.boundRect.size(); i++)
		{
                double zoom_value;
            	x0 = knn_bgs.boundRect[i].x;
				y0 = knn_bgs.boundRect[i].y;
				w0 = knn_bgs.boundRect[i].width; 
				h0 = knn_bgs.boundRect[i].height; 
                x0_ =x0-2*knn_bgs.padSize;
                y0_ =y0-2*knn_bgs.padSize;
                w0_ =w0+4*knn_bgs.padSize;
                h0_ =h0+4*knn_bgs.padSize;

                CLIP(x0_,0,(knn_bgs.IMG_WID-1));
                CLIP(y0_,0,(knn_bgs.IMG_HGT-1));
                CLIP(w0_,1,(knn_bgs.IMG_WID-1-x0_));
                CLIP(h0_,1,(knn_bgs.IMG_HGT-1-y0_));
                CLIP(x0,0,(knn_bgs.IMG_WID-1));
                CLIP(y0,0,(knn_bgs.IMG_HGT-1));
                CLIP(w0,1,(knn_bgs.IMG_WID-1-x0));
                CLIP(h0,1,(knn_bgs.IMG_HGT-1-y0));

                if (w0 <= knn_bgs.tooSmalltoDrop + 2 * knn_bgs.padSize || h0 <= knn_bgs.tooSmalltoDrop + 2 * knn_bgs.padSize)
                    continue;
                Mat	img_roi = knn_bgs.frame(cv::Rect(x0, y0, w0, h0));
                Mat	img_show= show_img(cv::Rect(x0, y0, w0, h0));
                Mat img_roi_big;
                img_show.convertTo(img_show, img_show.type(), 1, 50);
                
		        
                if (img_roi.cols <= 100)
				{
					zoom_value = 200.0 / (double)(img_roi.cols);
					int wid = cvRound(zoom_value * img_roi.cols);
					int hgt = cvRound(zoom_value * img_roi.rows);
					resize(img_roi, img_roi_big, Size(wid, hgt));
				}
				else if (img_roi.cols <= 200)
				{
					img_roi_big = img_roi;
                    zoom_value = 1;
				}
				else
				{
					zoom_value = 200.0 / (double)(img_roi.cols);
					int wid = cvRound(zoom_value * img_roi.cols);
					int hgt = cvRound(zoom_value * img_roi.rows);
					resize(img_roi, img_roi_big, Size(wid, hgt));
				}

                for (int i = 0; i < repeat_count; i++)
                {
                    get_input_data_ssd(img_roi_big, input_data, img_h,  img_w);

                    gettimeofday(&t0, NULL);
                    set_tensor_buffer(input_tensor, input_data, img_size * 4);
                    run_graph(graph, 1);

                    gettimeofday(&t1, NULL);
                    float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
                    total_time += mytime;

                }

                tensor_t out_tensor = get_graph_output_tensor(graph, 0,0);//"detection_out");
                get_tensor_shape( out_tensor, out_dim, 4);
                outdata = (float *)get_tensor_buffer(out_tensor);

                int num=out_dim[1];
                
                post_process_ssd(img_roi_big, show_threshold, outdata, num);
                if(boxes.empty())
                {
                    Mat	tmp_hotmap= knn_bgs.hot_map(cv::Rect(x0,y0,w0,h0));
                    tmp_hotmap.convertTo(tmp_hotmap, tmp_hotmap.type(), 1, -10);
                }
                else
                {
                    Mat	tmp_hotmap= knn_bgs.hot_map(cv::Rect(x0_,y0_,w0_,h0_));
                    tmp_hotmap.convertTo(tmp_hotmap, tmp_hotmap.type(), 1, 10);
                }
                   
                togetherAllBox(zoom_value,x0,y0,boxes,boxes_all);
                togetherAllBox(zoom_value,x0,y0,boxes_ruff,boxes_ruff_all);
         }
        
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

        tensor_t out_tensor = get_graph_output_tensor(graph, 0,0);//"detection_out");
        get_tensor_shape( out_tensor, out_dim, 4);
        outdata = (float *)get_tensor_buffer(out_tensor);

        int num=out_dim[1];
        
        post_process_ssd(process_frame, show_threshold, outdata, num);
      
        togetherAllBox(1,0,0,boxes,boxes_all);
        togetherAllBox(1,0,0,boxes_ruff,boxes_ruff_all);
   
        draw_img(show_img);

        std::cout << "--------------------------------------\n";
        std::cout << "repeat " << repeat_count << " times, avg time per run is " << total_time / repeat_count << " ms\n";

        Mat out,hot_map_color,hot_map_color2,hot_map_thresh_color,hot_map;
        hot_map = knn_bgs.hot_map;

        //cv::cvtColor(knn_bgs.bit_and_hotmap_with_diff, hot_map_color, CV_GRAY2BGR);
        cv::cvtColor(knn_bgs.FGMask_origin, hot_map_color2, CV_GRAY2BGR);  
        cv::cvtColor(knn_bgs.hot_map, hot_map_thresh_color, CV_GRAY2BGR);  
        hconcat(show_img,knn_bgs.bk,out);
        hconcat(hot_map_color2,hot_map_thresh_color,hot_map_color2);
        vconcat(out,hot_map_color2,out);
 
        cv::imshow("MSSD", out);
        //outputVideo.write(out);
        outputVideo.write(show_img);
 

        if( cv::waitKey(10) == 'q' )
            break;
    }

    postrun_graph(graph);
    free(input_data);
    destroy_runtime_graph(graph);
    remove_model(model_name);
    outputVideo.release();
    capture.release();
    return 0;
}


