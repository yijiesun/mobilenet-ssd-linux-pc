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

int key;
int draw_mask_flag; //0-null 1-new draw line 2-drawing 3-save 4-show result
bool quit;
bool save;
Mat img,show_img,show_img_out,Mask,Mask_out;
unsigned char *mask;
Point p1, p2,pb;
int isDrawing;
void *keyboard_thread(void *threadarg);
void my_handler(int s);
static void draw(int event, int x, int y, int flags, void *);
void generate_and_save_mask(char * filename);
 int scanKeyboard();


int main()
{
    int dev_num = 1;
    get_param_mms_cvCaptrue(dev_num);
    std::cout<<"open /dev/video"<<dev_num<<std::endl;
    cv::VideoCapture capture(dev_num);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    pb.x=-1;
    key = 0;
    draw_mask_flag = 1;
    quit = false;
    struct sigaction sigIntHandler;
   sigIntHandler.sa_handler = my_handler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;
   sigaction(SIGINT, &sigIntHandler, NULL);

    mask = (unsigned char *)malloc(640*480/8);

    capture >> img;
	//img = imread("/home/syj/cap_data/img/2019-11-15-11-39-55-465.jpg");
    
    Mask.create(img.rows,img.cols,CV_8UC1);
    Mask_out.create(img.rows,img.cols,CV_8UC1);
    show_img.create(img.rows,img.cols,CV_8UC3);
    show_img_out.create(img.rows,img.cols,CV_8UC3);
    memset(Mask_out.data,255,640*480*sizeof(uchar));
    memset(show_img.data,0,3*640*480*sizeof(uchar));
	namedWindow("image");
	setMouseCallback("image", draw);

	pthread_t keyboard;
	int rc = pthread_create(&keyboard, NULL, keyboard_thread, NULL);
    while(1)
    {
        capture >> img;
        addWeighted(show_img,1,img,1,-1,show_img_out);
        if(draw_mask_flag == 3)
        {
            imshow("image", Mask);
             waitKey(800);
            draw_mask_flag = 4;
            generate_and_save_mask("mask.jpg");
             imshow("image", Mask_out);
             waitKey(1000);
             Mat mask_color;
             cvtColor(Mask_out, mask_color, COLOR_GRAY2RGB);
             addWeighted(show_img_out,0.8,mask_color,0.3,-1,show_img_out);
             imwrite("mask_img.jpg",show_img_out);
        }
        else if(draw_mask_flag == 4)
        {
             Mat mask_color;
             cvtColor(Mask_out, mask_color, COLOR_GRAY2RGB);
             addWeighted(show_img_out,0.8,mask_color,0.3,-1,show_img_out);
        }
        imshow("image", show_img_out);
        waitKey(30);
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
            memset(show_img.data,0,3*640*480*sizeof(uchar));
            memset(Mask_out.data,255,640*480*sizeof(uchar));
            memset(Mask.data,0,640*480*sizeof(uchar));
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
                
                p1 = Point(CLIP(x,639), CLIP(y,479));
                draw_mask_flag = 2;
            }
            else if(draw_mask_flag == 2)
            {
                p2 = Point(CLIP(x,639), CLIP(y,479));
                if(isDrawing==2)
                    p1 = p2;
                 isDrawing = 1;
                cout << "drawing  "<<x<<","<<y<< endl;
                line(show_img, p1, p2, Scalar(0, 0, 255));
                line(Mask, p1, p2, Scalar(255, 255, 255));
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
            p1 = Point(CLIP(x,639), CLIP(y,479));
            draw_mask_flag = 2;
        }
            
        if(isDrawing==2)
        {
            p2 = Point(CLIP(x,639), CLIP(y,479));
            line(show_img, p1, p2, Scalar(0, 0, 255));
            line(Mask, p1, p2, Scalar(255, 255, 255));
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
        line(show_img, p1, pb, Scalar(0, 0, 255));
        line(Mask, p1, pb, Scalar(255, 255, 255));
        cout << "line begin point "<< endl;
    }
        if(isDrawing == 0)
        isDrawing =2;
}
void generate_and_save_mask(char * filename)
{
    int white_cnt = 0;

    for(int w=0;w<640;w++)
    {
        for(int h=0;h<480;h++)
        {
            if(*(Mask.data+h*640+w)!=0)
                break;
            else
                *(Mask_out.data+h*640+w) = 0;
            
        }
    }

    for(int w=0;w<640;w++)
    {
        for(int h=479;h>=0;h--)
        {
            if(*(Mask.data+h*640+w)!=0)
                break;
            else
                *(Mask_out.data+h*640+w) = 0;
            
        }
    }

    imwrite(filename,Mask_out);

};