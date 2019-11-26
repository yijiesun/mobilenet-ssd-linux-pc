//***************************************************************************
//***************************************************************************
//鼠标操作在一张图片上面画线
//***************************************************************************
//***************************************************************************
#include <signal.h>
#include <stdio.h>
#include <termios.h>
#include<iostream>
#include<time.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include "config.h"
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>//图像处理的库,circle()要用到
#include<opencv2/highgui/highgui.hpp>

#define CLIP(a,b) (  (a) = (a)>(b)?(b):((a)<0?0:(a))  )
using namespace std;
using namespace cv;

int IMG_WID,IMG_HGT;
int mode;// 0-camera 1-img 2-video
int key;
int draw_mask_flag; //0-null 1-new draw line 2-drawing 3-save 4-show result
bool quit;
bool save;
Mat origin,mask_line_add_origin_hole,mask_line_color,mask_line_add_origin_solid,mask_line_gray,mask_solid,bitMat;
Point p1, p2,pb;
int isDrawing;
void *keyboard_thread(void *threadarg);
void my_handler(int s);
static void draw(int event, int x, int y, int flags, void *);
void generate_and_save_mask(char * filename);
 int scanKeyboard();


int main()
{
    mode =0;
    int dev_num = 1;
    get_param_mms_cvCaptrue(dev_num);
    std::cout<<"open /dev/video"<<dev_num<<std::endl;
    std::string img_in,tmp,video_in;
    get_param_mssd_img(img_in,tmp);
    std::cout<<"img_in "<<img_in<<std::endl;
    get_param_mssd_video(video_in,tmp);
    std::cout<<"video_in "<<video_in<<std::endl;

    cv::VideoCapture capture(dev_num);
    if(!capture.isOpened())
    {
        mode = 1;
        cout<<"fail to open camera! try to open  img"<<endl;
        origin = imread(img_in.c_str());
       
        if(origin.empty())
        {
            cout<<"fail to open camera! try to open  video"<<endl;
             mode = 2;
        }
        else
             resize(origin,origin,Size(IMG_WID,IMG_HGT));
        
        
        
    }
	if(mode == 0)
    {
        capture.set(CV_CAP_PROP_FRAME_WIDTH, IMG_WID);
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, IMG_HGT);
        IMG_WID = capture.get(CV_CAP_PROP_FRAME_WIDTH);
        IMG_HGT = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
        capture >> origin;
    }
    else if(mode ==2)
    {
        capture.open(video_in.c_str());
		capture.set(CV_CAP_PROP_FOURCC, cv::VideoWriter::fourcc ('M', 'J', 'P', 'G'));
        IMG_WID = capture.get(CV_CAP_PROP_FRAME_WIDTH);
        IMG_HGT = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
        cout<<"IMG_WID "<<IMG_WID<<"  IMG_HGT "<<IMG_HGT<<endl;
    }

    pb.x=-1;
    key = 0;
    draw_mask_flag = 1;
    quit = false;
    struct sigaction sigIntHandler;
   sigIntHandler.sa_handler = my_handler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;
   sigaction(SIGINT, &sigIntHandler, NULL);

	//img = imread("/home/syj/cap_data/img/2019-11-15-11-39-55-465.jpg");
    
    mask_line_gray.create(IMG_HGT,IMG_WID,CV_8UC1);
    mask_solid.create(IMG_HGT,IMG_WID,CV_8UC1);
    mask_line_color.create(IMG_HGT,IMG_WID,CV_8UC3);
    mask_line_add_origin_hole.create(IMG_HGT,IMG_WID,CV_8UC3);
    mask_line_add_origin_solid.create(IMG_HGT,IMG_WID,CV_8UC3);
    mask_solid =Mat::ones(IMG_HGT,IMG_WID,CV_8UC1)*255;
    mask_line_gray =Mat::zeros(IMG_HGT,IMG_WID,CV_8UC1);
    mask_line_color =Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);
    //memset(mask_solid.data,255,IMG_WID*IMG_HGT*sizeof(uchar));
    //memset(mask_line_color.data,0,3*IMG_WID*IMG_HGT*sizeof(uchar));
	namedWindow("image");
	setMouseCallback("image", draw);

	pthread_t keyboard;
	int rc = pthread_create(&keyboard, NULL, keyboard_thread, NULL);

        Mat tmp_color,tmp_gray;
        tmp_gray.create(IMG_HGT,IMG_WID,CV_8UC1);
        tmp_color.create(IMG_HGT,IMG_WID,CV_8UC3);
    bool first = true;
    bool cap =true;
    while(1)
    {
        if(cap && (mode == 0 || mode==2))
        {
            capture >> origin;
            if(origin.empty())
                break;
            //resize(origin,origin,Size(IMG_WID,IMG_HGT));
        }
        
        cvtColor(mask_line_color, tmp_gray, COLOR_RGB2GRAY);
        tmp_gray*=255;
        cvtColor(tmp_gray, tmp_color, COLOR_GRAY2RGB);
        bitwise_not(tmp_color,tmp_color);
        bitwise_and(origin,tmp_color,tmp_color);
        bitwise_or(tmp_color,mask_line_color,mask_line_add_origin_hole);
        mask_line_add_origin_solid = mask_line_add_origin_hole.clone();
        //addWeighted(mask_line_color,0.8,img,1,-1,mask_line_add_origin_solid);
        if(draw_mask_flag == 3)
        {
            imshow("image", mask_line_gray);
             waitKey(1800);
            draw_mask_flag = 4;
            generate_and_save_mask("mask.jpg");
             imshow("image", mask_solid);
             waitKey(1000);
             Mat mask_solid_color;
             cvtColor(mask_solid, mask_solid_color, COLOR_GRAY2RGB);
             addWeighted(mask_line_add_origin_solid,0.8,mask_solid_color,0.3,-1,mask_line_add_origin_solid);
             imwrite("mask_img.jpg",mask_line_add_origin_solid);
             imwrite("bk.jpg",origin);
             cap = true;
        }
        else if(draw_mask_flag == 4)
        {
             Mat mask_solid_color;
             cvtColor(mask_solid, mask_solid_color, COLOR_GRAY2RGB);
             addWeighted(mask_line_add_origin_solid,0.8,mask_solid_color,0.3,-1,mask_line_add_origin_solid);
        }
        imshow("image", mask_line_add_origin_solid);
        waitKey(30);
        if(first)
        {
            first = false;
            cap = false;
        }
        
        if (quit)
         break;
    }

	return 0;
};

void *keyboard_thread(void *threadarg)
{
	while (1)
	{

    
    if(key==0)
        key = scanKeyboard();

        printf("  key:%d\n",key);
        //s-115  d-100 c-99
        if(key == 100)
        {   key = 0;
            mask_line_color =Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);
            mask_solid =Mat::ones(IMG_HGT,IMG_WID,CV_8UC1)*255;
            mask_line_gray =Mat::zeros(IMG_HGT,IMG_WID,CV_8UC1);

            //memset(mask_line_color.data,0,3*IMG_WID*IMG_HGT*sizeof(uchar));
            //memset(mask_solid.data,255,IMG_WID*IMG_HGT*sizeof(uchar));
            //memset(mask_line_gray.data,0,IMG_WID*IMG_HGT*sizeof(uchar));
            draw_mask_flag = 1;
            save=false;
            pb.x=-1;
            cout<<"begin draw roi"<<endl;
        }
        else if(key == 115)
        {
            key = 0;
            draw_mask_flag = 3;
            cout<<"save roi img"<<endl;
        }
        else if(key!=0)
            key = 0;

        if (quit)
            pthread_exit(NULL);
    }
};

void my_handler(int s)
{
            quit = true;
            cout<<"Caught signal "<<s<<" quit="<<quit<<endl;
};

 int scanKeyboard()
{
int in;
struct termios new_settings;
struct termios stored_settings;
tcgetattr(0,&stored_settings);
new_settings = stored_settings;
new_settings.c_lflag &= (~ICANON);
new_settings.c_cc[VTIME] = 0;
tcgetattr(0,&stored_settings);
new_settings.c_cc[VMIN] = 1;
tcsetattr(0,TCSANOW,&new_settings);
 
in = getchar();
 
tcsetattr(0,TCSANOW,&stored_settings);
return in;
};

static void draw(int event, int x, int y, int flags, void *)
{
    if(isDrawing!=2)
        isDrawing = 0;
	if ((event == CV_EVENT_MOUSEMOVE))
	{
        if(flags&CV_EVENT_FLAG_LBUTTON)
        {
            if(draw_mask_flag == 1)
            {
                cout<<"new draw<<endl"<<endl;
                
                p1 = Point(CLIP(x,IMG_WID-1), CLIP(y,IMG_HGT-1));
                draw_mask_flag = 2;
            }
            else if(draw_mask_flag == 2)
            {
                p2 = Point(CLIP(x,IMG_WID-1), CLIP(y,IMG_HGT-1));
                if(isDrawing==2)
                    p1 = p2;
                 isDrawing = 1;
                cout << "drawing  "<<x<<","<<y<< endl;
                line(mask_line_color, p1, p2, Scalar(0, 0, 255),3);
                line(mask_line_gray, p1, p2, Scalar(255, 255, 255),1);
                if(pb.x==-1)
                    pb = p2;
                p1 = p2;
            }
        }
	}
    else if((draw_mask_flag == 2||draw_mask_flag == 1)&&(flags&CV_EVENT_LBUTTONDOWN))
    {
        if(draw_mask_flag == 1)
        {
            p1 = Point(CLIP(x,IMG_WID-1), CLIP(y,IMG_HGT-1));
            draw_mask_flag = 2;
        }
            
        if(isDrawing==2)
        {
            p2 = Point(CLIP(x,IMG_WID-1), CLIP(y,IMG_HGT-1));
            line(mask_line_color, p1, p2, Scalar(0, 0, 255),3);
            line(mask_line_gray, p1, p2, Scalar(255, 255, 255));
             if(pb.x==-1)
                pb = p2;
            p1 = p2;
            cout << "drawing point to point "<< endl;
        }
    }
    else if(draw_mask_flag == 2&&(flags&CV_EVENT_RBUTTONDOWN))
    {
        draw_mask_flag = 3;
        cout << "save roi img"<< endl;
    }
    else if(draw_mask_flag == 2&&(flags&CV_EVENT_MBUTTONUP))
    {
        line(mask_line_color, p1, pb, Scalar(0, 0, 255),3);
        line(mask_line_gray, p1, pb, Scalar(255, 255, 255));
        cout << "line begin point "<< endl;
    }
        if(isDrawing == 0)
        isDrawing =2;
}
void generate_and_save_mask(char * filename)
{
    int white_cnt = 0;

    for(int w=0;w<IMG_WID;w++)
    {
        for(int h=0;h<IMG_HGT;h++)
        {
            if(*(mask_line_gray.data+h*IMG_WID+w)!=0)
                break;
            else
                *(mask_solid.data+h*IMG_WID+w) = 0;
            
        }
    }

    for(int w=0;w<IMG_WID;w++)
    {
        for(int h=IMG_HGT-1;h>=0;h--)
        {
            if(*(mask_line_gray.data+h*IMG_WID+w)!=0)
                break;
            else
                *(mask_solid.data+h*IMG_WID+w) = 0;
            
        }
    }

    imwrite(filename,mask_solid);

};