#include <unistd.h>
#include <iostream>
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include "config.h"
using namespace cv;
using namespace std;
int main(int argc, char *argv[])
{
    Mat frame1;
    Mat frame2;
    Mat out;

    cv::VideoCapture capture1;
    cv::VideoCapture capture2;
    VideoWriter outputVideo;
    std::string in_video_file1;
    std::string out_video_file;
    get_param_mssd_video(out_video_file,in_video_file1);
    std::cout<<"input video: "<<in_video_file1<<"\noutput video: "<<out_video_file<<std::endl;

    std::string in_video_file2;
    std::string tmp;
    get_param_mssd_video_knn(tmp,in_video_file2);
    std::cout<<"input video: "<<in_video_file2<<std::endl;


    capture1.open(in_video_file1);
    capture1.set(CV_CAP_PROP_FOURCC, cv::VideoWriter::fourcc ('M', 'J', 'P', 'G'));
    capture2.open(in_video_file2);
    capture2.set(CV_CAP_PROP_FOURCC, cv::VideoWriter::fourcc ('M', 'J', 'P', 'G'));


    Size sWH = Size( capture1.get(CV_CAP_PROP_FRAME_WIDTH)+ capture2.get(CV_CAP_PROP_FRAME_WIDTH), max(capture1.get(CV_CAP_PROP_FRAME_HEIGHT),capture2.get(CV_CAP_PROP_FRAME_HEIGHT)));
    out.create(sWH,CV_8UC3);
    bool ret = outputVideo.open(out_video_file, cv::VideoWriter::fourcc ('M', 'P', '4', '2'), 25, sWH);

    while(1)
    {
        if (!capture1.read(frame1))
		{
			cout<<in_video_file1<<"  cannot open video or end of video"<<endl;
            break;
		}
        if (!capture2.read(frame2))
		{
			cout<<in_video_file1<<"  cannot open video or end of video"<<endl;
            break;
		}
        hconcat(frame1,frame2,out);
        outputVideo.write(out);
        imshow("img",out);
        if( cv::waitKey(10) == 'q' )
            break;
    }
    capture2.release();
    capture1.release();
    outputVideo.release();
}