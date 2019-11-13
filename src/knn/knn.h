#ifndef KNN_H
#define KNN_H
#include <iostream>  
#include <sstream>
#include "opencv2/opencv.hpp"  
  
using namespace cv;  
using namespace std;  

#define FIND_ROI 1
const float defaultDist2Threshold = 20.0f;// 灰度聚类阈值 20

struct PixelHistory
{
	unsigned char *gray;// 历史灰度值
	unsigned char *IsBG;// 对应灰度值的前景/背景判断，1代表判断为背景，0代表判断为前景
};

class KNN_BGS  
{  
public:  
	KNN_BGS();
    ~KNN_BGS(void);
	float knn_over_percent;
	int dilateRatio;
	int solid_frame;
	int history_num;
	int useTopRect;
	int knnv;
	int padSize;
	int IMG_WID;
	int IMG_HGT;
	int pos;
	int minContorSize;
	int insideDilate_win_size;
	int insideDilate_scale;
	double tooSmalltoDrop;
	string saveAdress;
	vector<Rect> boundRect;
	void init(VideoCapture &v_capture);
	void knn_core();
	void saveROI();
	Mat bk;
	cv::Mat frame, fgray, FGMask, showImg;
	void postTreatment();
	void processRects();
	void getTopRects(vector<Rect> &rects0, vector<Rect> &rects);
	//bool sortFun(const cv::Rect &p1, const cv::Rect &p2);
	int buildAndClearSmallContors(vector<vector<Point>> &contours, vector<Rect> &rects, int size);
	void mergeRecs(vector<Rect> &rects, float percent);
	void drawRecs(Mat & img, vector<Rect> &rects, const Scalar& color);
	float DecideOverlap(const Rect &r1, const Rect &r2, Rect &r3);
	void set(int *conf);
	void paddingRecs(vector<Rect> &rects, int size);
	void insideDilate(Mat & bimg, Mat & bout, int win_size, int scale);
private:  
	PixelHistory* framePixelHistory;// 记录一帧图像中每个像素点的历史信息
	
	int frameCnt;
	
	
};  
#endif