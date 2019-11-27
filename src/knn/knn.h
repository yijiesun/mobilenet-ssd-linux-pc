#ifndef KNN_H
#define KNN_H
#include <iostream>  
#include <sstream>
#include "../config.h"
#include "opencv2/opencv.hpp"  
  
using namespace cv;  
using namespace std;  

#define FIND_ROI 1
//const float defaultDist2Threshold = 10.0f;// �ҶȾ�����ֵ 20

struct PixelHistory
{
	unsigned char *gray;// ��ʷ�Ҷ�ֵ
	unsigned char *IsBG;// ��Ӧ�Ҷ�ֵ��ǰ��/�����жϣ�1�����ж�Ϊ������0�����ж�Ϊǰ��
};

class KNN_BGS  
{  
public:  
	KNN_BGS();
    ~KNN_BGS(void);
	double knn_thresh;
	float knn_over_percent;
	int dilateRatio;
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
	int knn_box_exist_cnt;
	double tooSmalltoDrop;
	string saveAdress;
	vector<Box> knn_use_box;
	vector<Rect> boundRect;
	void init();
	void knn_core();
	void saveROI();
	Mat bg_fix_mask;
	Mat last_frame;
	Mat senser_roi_down100,senser_roi_down100_not;
	Mat bk_cnt,bk_cnt_cnt; //used to record bg cnt
	Mat bk;
	Mat hot_map,hot_map_noraml,hot_map_thresh;
	Mat bit_and_hotmap_with_diff;
	cv::Mat frame, fgray, FGMask, showImg,DiffMask,FGMask_origin,senser_roi;
	void postTreatment(Mat &mat);
	void processRects(vector<Box> &box);
	void addBoxToRecs();
	void getTopRects(vector<Rect> &rects0, vector<Rect> &rects);
	//bool sortFun(const cv::Rect &p1, const cv::Rect &p2);
	int buildAndClearSmallContors(vector<vector<Point>> &contours, vector<Rect> &rects, int size);
	void mergeRecs(vector<Rect> &rects, float percent);
	void drawRecs(Mat & img, vector<Rect> &rects, const Scalar& color);
	float DecideOverlap(const Rect &r1, const Rect &r2, Rect &r3);
	void set(int *conf);
	void paddingRecs(vector<Rect> &rects, int size);
	void insideDilate(Mat & bimg, Mat & bout, int win_size, int scale);
	void diff2(Mat &cur,Mat &las);
	void add_diff_in_box_to_mask(vector<Box> &box);
	void update_bg();
private:  
	PixelHistory* framePixelHistory;// ��¼һ֡ͼ����ÿ�����ص����ʷ��Ϣ
	
	int frameCnt;
	
	
};  
#endif