#include "knn.h"

#include<algorithm>
using namespace cv;
using namespace std;

#define CLIP(a,b,c) (  (a) = (a)>(c)?(c):((a)<(b)?(b):(a))  )

KNN_BGS::KNN_BGS()
{
	framePixelHistory = NULL;
	frameCnt = 0;
};

KNN_BGS::~KNN_BGS(void)
{
	
}

void KNN_BGS::init()
{
	int gray = 0;
	FGMask.create(IMG_HGT, IMG_WID, CV_8UC1);
	FGMask_origin.create(IMG_HGT, IMG_WID, CV_8UC1);
	senser_roi.create(IMG_HGT, IMG_WID, CV_8UC1);
	bk_cnt.create(IMG_HGT, IMG_WID, CV_8UC1);
	bk_cnt_cnt.create(IMG_HGT, IMG_WID, CV_8UC1);
	bg_fix_mask.create(IMG_HGT, IMG_WID, CV_8UC1);
	bg_fix_mask=Mat::zeros(IMG_HGT,IMG_WID,CV_8UC1);
	bk_cnt_cnt=Mat::zeros(IMG_HGT,IMG_WID,CV_8UC1);
	bk_cnt=Mat::zeros(IMG_HGT,IMG_WID,CV_8UC1);
	senser_roi=Mat::zeros(IMG_HGT,IMG_WID,CV_8UC1);
	bit_and_hotmap_with_diff.create(IMG_HGT, IMG_WID, CV_8UC1);
	DiffMask.create(IMG_HGT, IMG_WID, CV_8UC1);
										// framePixelHistory����ռ�
	framePixelHistory = (PixelHistory*)malloc(IMG_WID*IMG_HGT * sizeof(PixelHistory));

	for (int i = 0; i < IMG_WID*IMG_HGT; i++)
	{
		framePixelHistory[i].gray = (unsigned char*)malloc((history_num) * sizeof(unsigned char));
		framePixelHistory[i].IsBG = (unsigned char*)malloc((history_num) * sizeof(unsigned char));
		memset(framePixelHistory[i].gray, 0, (history_num ) * sizeof(unsigned char));
		memset(framePixelHistory[i].IsBG, 0, (history_num ) * sizeof(unsigned char));
	}
	
	hot_map.create(IMG_HGT,IMG_WID,CV_8UC1);
	hot_map=Mat::ones(IMG_HGT,IMG_WID,CV_8UC1)+100;

}

void KNN_BGS::knn_core()
{
	//normalize(hot_map, hot_map_noraml, 100, 5, NORM_MINMAX);
	//normalize(hot_map, hot_map_noraml, 12, 1, NORM_MINMAX);
	cv::cvtColor(frame, fgray, CV_BGR2GRAY);
	FGMask_origin.setTo(Scalar(255));
	//hot_map_thresh = hot_map;

	//normalize(hot_map_thresh, hot_map_thresh, 50, 5, NORM_MINMAX);
	int gray = 0;
	bg_fix_mask=Mat::zeros(IMG_HGT,IMG_WID,CV_8UC1);
	for (int i = 0; i < IMG_HGT; i++)
	{
		for (int j = 0; j < IMG_WID; j++)
		{
			bool update_BG = false;
			gray = fgray.at<unsigned char>(i, j);
			int fit = 0;
			int fit_bg = 0;

			for (int n = 0; n < history_num; n++)
			{
				if (fabs((float)gray - framePixelHistory[i*IMG_WID + j].gray[n]) < 20 /* hot_map_thresh.data[i*IMG_WID + j] */  ) //  100-hot_map_noraml.data[i*IMG_WID + j]
				{//foreground
						fit++;
					if (framePixelHistory[i*IMG_WID + j].IsBG[n])
					{
						fit_bg++;
					}
				}
			}
			if (fit_bg >= knnv)
			{
				FGMask_origin.at<unsigned char>(i, j) = 0;
			}
			if(hot_map.data[i*IMG_WID + j] == 0 && senser_roi.data[i*IMG_WID + j]==0)
			{
				FGMask_origin.at<unsigned char>(i, j) = 0;
				bk_cnt.at<unsigned char>(i, j)++;
			}

			int index = frameCnt % history_num;
			framePixelHistory[i*IMG_WID + j].gray[index] = gray;
			framePixelHistory[i*IMG_WID + j].IsBG[index] = fit >= knnv ? 1 : 0;// ��ǰ����Ϊ�����������ʷ��Ϣ



			if(bk_cnt.data[i*IMG_WID + j]>=5)
			{
				if(hot_map.data[i*IMG_WID + j] >=100)
				{
					bk_cnt_cnt.data[i*IMG_WID + j]++;
					hot_map.data[i*IMG_WID + j] = 100;
				}
				if(fit>=2)
				{
					bk_cnt.data[i*IMG_WID + j] = 0;
					bk.data[i*IMG_WID*3 + j*3]= frame.data[i*IMG_WID*3 + j*3];
					bk.data[i*IMG_WID*3 + j*3+1]= frame.data[i*IMG_WID*3 + j*3+1];
					bk.data[i*IMG_WID*3 + j*3+2]= frame.data[i*IMG_WID*3 + j*3+2];
				}
			}
			if(bk_cnt_cnt.data[i*IMG_WID + j]>=10)
			{
				hot_map.data[i*IMG_WID + j] = 0;
				bk_cnt.data[i*IMG_WID + j] = 0;
				senser_roi.data[i*IMG_WID + j]=0;
			}

		}
	}
	frameCnt++;
}

void KNN_BGS::postTreatment(Mat &mat)
{
	#if 1
		cv::medianBlur(mat, mat, 3);
		threshold(mat, mat, 2, 255, CV_THRESH_BINARY);
		Mat element = getStructuringElement(MORPH_RECT, Size(dilateRatio, dilateRatio));
		//erode(mat, mat, element);
		dilate(mat, mat, element);
	#else
		cv::medianBlur(mat, mat, 3);
		threshold(mat, mat, 30, 255, CV_THRESH_BINARY);
		//Mat element = getStructuringElement(MORPH_RECT, Size(dilateRatio, dilateRatio));
		//erode(mat, mat, element);
		//dilate(mat, mat, element);
	#endif

	//dilate(mat, mat, element);
	//if (insideDilate_win_size != 0)
	//{
	//	insideDilate(mat, mat, insideDilate_win_size, insideDilate_scale);
		//insideDilate(mat, mat, insideDilate_win_size, insideDilate_scale);
	//}
}

bool sortFun(const cv::Rect &p1, const cv::Rect &p2)
{
	return p1.width * p1.height > p2.width * p2.height;//��������  
}

void KNN_BGS::getTopRects(vector<Rect> &rects0, vector<Rect> &rects)
{
	for (int i = 0; i<rects0.size(); i++)
	{
		if (i >= useTopRect)
			break;
		rects.push_back(rects0[i]);
	}
}

void KNN_BGS::addBoxToRecs()
{
	vector<Box>::iterator it;
	for(it=knn_use_box.begin();it!=knn_use_box.end();)
	{
		it->show_cnt++;
		Rect rect_tmp;
		rect_tmp.x =it->x0;
		rect_tmp.y = it->y0;
		rect_tmp.width = it->x1 - it->x0;
		rect_tmp.height = it->y1 - it->y0;
		CLIP(rect_tmp.x,0,IMG_WID-1);
		CLIP(rect_tmp.y,0,IMG_HGT-1);
		CLIP(rect_tmp.width,1,IMG_WID-1);
		CLIP(rect_tmp.height,1,IMG_HGT-1);
		boundRect.push_back(rect_tmp);
		if(it->show_cnt>=knn_box_exist_cnt)
			it=knn_use_box.erase(it);
		else
			it++;
	}

}

void KNN_BGS::add_diff_in_box_to_mask(vector<Box> &box)
{

	for (int i = 0; i<box.size(); i++) {
		box[i].show_cnt =0;
		knn_use_box.push_back(box[i]);
		int x0 = box[i].x0-2*padSize;
		int y0 = box[i].y0-2*padSize;
		int w0 = box[i].x1-box[i].x0+4*padSize;
		int h0 = box[i].y1-box[i].y0+4*padSize;
		CLIP(x0,0,IMG_WID-1);
		CLIP(y0,0,IMG_HGT-1);
		CLIP(w0,1,IMG_WID-x0 -1);
		CLIP(h0,1,IMG_HGT-y0 -1);
		int x0_ = box[i].x0-padSize;
		int y0_ = box[i].y0-padSize;
		int w0_ = box[i].x1-box[i].x0+2*padSize;
		int h0_ = box[i].y1-box[i].y0+2*padSize;
		CLIP(x0_,0,IMG_WID-1);
		CLIP(y0_,0,IMG_HGT-1);
		CLIP(w0_,1,IMG_WID-x0_ -1);
		CLIP(h0_,1,IMG_HGT-y0_-1);
		//make hot map
		Mat	tmp= hot_map(cv::Rect(x0,y0,w0,h0));
		tmp.convertTo(tmp, tmp.type(), 1, 255);	
		tmp= senser_roi(cv::Rect(x0,y0,w0,h0));
		tmp.convertTo(tmp, tmp.type(), 1, 255);	
		tmp= bk_cnt(cv::Rect(x0,y0,w0,h0));
		tmp.convertTo(tmp, tmp.type(), 1, -255);	
		tmp= bk_cnt_cnt(cv::Rect(x0,y0,w0,h0));
		tmp.convertTo(tmp, tmp.type(), 1, -255);	
	}
	hot_map_noraml = hot_map -100;
	hot_map_noraml+=254;
	hot_map_noraml-=254;
	hot_map_noraml*=255;
	//normalize(hot_map_noraml, hot_map_noraml, 255, 0, NORM_MINMAX);
     
	bitwise_and(DiffMask,hot_map_noraml,bit_and_hotmap_with_diff);
	bitwise_or(FGMask_origin,bit_and_hotmap_with_diff,FGMask_origin);

	
	bitwise_and(senser_roi,hot_map,senser_roi_down100);

	senser_roi_down100 = 100 -senser_roi_down100;
	bitwise_and(senser_roi_down100,senser_roi,senser_roi_down100);
	senser_roi_down100*=255;

	bitwise_not(senser_roi_down100,senser_roi_down100_not);
	bitwise_and(hot_map,senser_roi_down100_not,senser_roi_down100_not);
	senser_roi_down100-=100;
	hot_map = senser_roi_down100_not+senser_roi_down100;

	senser_roi_down100-=154;
	bk_cnt+=senser_roi_down100;




}
void KNN_BGS::processRects(vector<Box> &box)
{
	add_diff_in_box_to_mask(box);
	FGMask = FGMask_origin.clone();
	 postTreatment(FGMask);

#if FIND_ROI
	std::vector<cv::Rect> boundRectTmp;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarcy;
	//findContours(DiffMask.clone(), contours, hierarcy, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE);
	findContours(FGMask.clone(), contours, hierarcy, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE);
	int rec_nums = buildAndClearSmallContors(contours, boundRectTmp, minContorSize);
	std::sort(boundRectTmp.begin(), boundRectTmp.end(), sortFun);
	getTopRects(boundRectTmp, boundRect);
	addBoxToRecs();

	//mergeRecs(boundRect, knn_over_percent);
	
	paddingRecs(boundRect, padSize);
	mergeRecs(boundRect, knn_over_percent);
#if 1
		vector<Rect>::iterator it;
		int th = 5;
	for(it=boundRect.begin();it!=boundRect.end();it++)
	{
		//up
		while(1)
		{
			int x0,y0,w0,h0;
			x0 = it->x;
			y0 = it->y-1;
			w0 = it->width;
			h0 = it->height;
			if(y0<=0) break;
			Mat tmp=DiffMask(Rect(x0,y0,w0,1))/255;
			int zero_cnt = countNonZero(tmp);
			//cout<<"uzero_cnt "<<zero_cnt<<" w0/3 "<<w0/3<<endl;
			if(zero_cnt==0) break;
			it->y -=1;

		}
		//down
		while(1)
		{
			int x0,y0,w0,h0;
			x0 = it->x;
			y0 = it->y;
			w0 = it->width;
			h0 = it->height+1;
			if(y0+h0>=IMG_HGT-2) break;
			Mat tmp=DiffMask(Rect(x0,y0+h0,w0,1))/255;
			int zero_cnt = countNonZero(tmp);
			//cout<<"dzero_cnt "<<zero_cnt<<" w0/3 "<<w0/3<<endl;
			if(zero_cnt==0) break;
			it->height +=1;
		}
		//left
		while(1)
		{
			int x0,y0,w0,h0;
			x0 = it->x-1;
			y0 = it->y;
			w0 = it->width;
			h0 = it->height;
			if(x0<=0) break;
			Mat tmp=DiffMask(Rect(x0,y0,1,h0))/255;
			int zero_cnt = countNonZero(tmp);
			//cout<<"lzero_cnt "<<zero_cnt<<" w0/3 "<<w0/3<<endl;
			if(zero_cnt==0) break;
			it->x -=1;
		}
		//right
		while(1)
		{
			int x0,y0,w0,h0;
			x0 = it->x;
			y0 = it->y;
			w0 = it->width+1;
			h0 = it->height;
			if(x0+w0>=IMG_WID-2) break;
			Mat tmp=DiffMask(Rect(x0+w0,y0,1,h0))/255;
			int zero_cnt = countNonZero(tmp);
			//cout<<"rzero_cnt "<<zero_cnt<<" w0/3 "<<w0/3<<endl;
			if(zero_cnt==0) break;
			it->width +=1;
		}
		
	}
#endif

	//drawRecs(showImg, boundRect, Scalar(0, 255, 0, 0));


	boundRectTmp.shrink_to_fit();
	contours.shrink_to_fit();
	hierarcy.shrink_to_fit();
#else
	Rect rect_tmp(0,0, IMG_WID, IMG_HGT);
	boundRect.push_back(rect_tmp);
#endif
}

int KNN_BGS::buildAndClearSmallContors(vector<vector<Point>> &contours, vector<Rect> &rects, int size)
{
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	int  cnt = 0;
	for (int i = 0; i<contours.size(); i++)
	{
		Rect rect_tmp = boundingRect((Mat)contours[i]);
		x0 = rect_tmp.x;
		y0 = rect_tmp.y;
		w0 = rect_tmp.width;
		h0 = rect_tmp.height;
		CLIP(x0,0,IMG_WID-1);
		CLIP(y0,0,IMG_HGT-1);
		CLIP(w0,1,IMG_WID-1);
		CLIP(h0,1,IMG_HGT-1);
		if (w0 * h0 >= size)
		{
			rects.push_back(rect_tmp);
			cnt++;
		}
	}
	return cnt;
}

void KNN_BGS::mergeRecs(vector<Rect> &rects, float percent)
{
	int len = rects.size();
	int new_len = 0;
	int ptr = 0;
	Rect tmp;

	for (;;)
	{
		if (ptr >= len)
			break;

		for (int i = 0; i < len; i++)
		{
			if (ptr < 0 || ptr >= rects.size() || i < 0 || i >= rects.size())
				break;
			if (i == ptr)
				continue;
			if (DecideOverlap(rects[ptr], rects[i], tmp) >= percent)
			{
				rects[ptr] = tmp;
				if (rects.begin() + i <= rects.end())
				{
					rects.erase(rects.begin() + i);
					i--;
				}
				else
					break;
			}

		}
		ptr++;
		len = rects.size();
	}

}

void KNN_BGS::drawRecs(Mat & img, vector<Rect> &rects, const Scalar& color)
{
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	for (int i = 0; i< rects.size(); i++)
	{
		x0 = rects[i].x;  //��õ�i����Ӿ��ε����Ͻǵ�x����
		y0 = rects[i].y; //��õ�i����Ӿ��ε����Ͻǵ�y����
		w0 = rects[i].width; //��õ�i����Ӿ��εĿ���
		h0 = rects[i].height; //��õ�i����Ӿ��εĸ߶�

		if (w0 <= tooSmalltoDrop || h0 <= tooSmalltoDrop)
			continue;
		rectangle(img, Point(x0, y0), Point(x0 + w0, y0 + h0), color, 2, 8); //���Ƶ�i����Ӿ���
	}
}

float KNN_BGS::DecideOverlap(const Rect &r1, const Rect &r2, Rect &r3)
{
	//r3 = r1;
	int x1 = r1.x;
	int y1 = r1.y;
	int width1 = r1.width;
	int height1 = r1.height;

	int x2 = r2.x;
	int y2 = r2.y;
	int width2 = r2.width;
	int height2 = r2.height;

	int endx = max(x1 + width1, x2 + width2);
	int startx = min(x1, x2);
	int width = width1 + width2 - (endx - startx);

	int endy = max(y1 + height1, y2 + height2);
	int starty = min(y1, y2);
	int height = height1 + height2 - (endy - starty);

	float ratio = 0.0f;
	float Area, Area1, Area2;

	if (width <= 0 || height <= 0)
		return 0.0f;
	else
	{
		Area = width*height;
		Area1 = width1*height1;
		Area2 = width2*height2;
		ratio = max(Area / (float)Area1, Area / (float)Area2);
		r3.x = startx;
		r3.y = starty;
		r3.width = endx - startx;
		r3.height = endy - starty;
	}

	return ratio;
}

void KNN_BGS::paddingRecs(vector<Rect> &rects, int size)
{
	for (int i = 0; i<rects.size(); i++)
	{
		rects[i].x = min(max(rects[i].x - size, 0), IMG_WID-1);
		rects[i].y = min(max(rects[i].y - size, 0), IMG_HGT-1);
		rects[i].width = rects[i].x + rects[i].width + 2 * size > IMG_WID ? IMG_WID - rects[i].x : rects[i].width + 2 * size;
		rects[i].height = rects[i].y + rects[i].height + 2 * size > IMG_HGT ? IMG_HGT - rects[i].y : rects[i].height + 2 * size;
	}
}

/*insideDilate:�����ͺ���
*���ص����»�����������ͬʱ�а����ص�б�ʱ��������
*bimg:��ֵͼ��
*win_size:��������
*scale:�б���Ч��������
*/
void KNN_BGS::insideDilate(Mat & bimg, Mat & bout, int win_size, int scale)
{
	for (int w = 0 + win_size; w < IMG_WID - win_size; w++)
	{
		for (int h = 0 + win_size; h < IMG_HGT - win_size; h++)
		{
			Point curr(w, h);
			Point refer;
			Mat tmp;
			int l = 0, r = 0, u = 0, d = 0;
			Mat roi_u(bimg, Rect(w, h - win_size, 1, win_size));
			Mat roi_d(bimg, Rect(w, h, 1, win_size));
			Mat roi_l(bimg, Rect(w - win_size, h, win_size, 1));
			Mat roi_r(bimg, Rect(w, h, win_size, 1));
			u = countNonZero(roi_u);
			d = countNonZero(roi_d);
			l = countNonZero(roi_l);
			r = countNonZero(roi_r);
			uchar* data = bout.ptr<uchar>(h);

			if ((u >= scale && d >= scale) || (l >= scale && r >= scale))
				data[w] = 255;
			else
			{
				if(u+d+r+l <= win_size)
					data[w] = 0;
			}
				
		}
	}
}

void KNN_BGS::set(int *conf)
{
	history_num = conf[1];
	knnv = conf[2];
	padSize = conf[3];
	minContorSize = conf[4];
	insideDilate_win_size = conf[6];
	insideDilate_scale = conf[7];
}

void KNN_BGS::saveROI()
{
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	for (int i = 0; i< boundRect.size(); i++)
	{
		x0 = boundRect[i].x;  //��õ�i����Ӿ��ε����Ͻǵ�x����
		y0 = boundRect[i].y; //��õ�i����Ӿ��ε����Ͻǵ�y����
		w0 = boundRect[i].width; //��õ�i����Ӿ��εĿ���
		h0 = boundRect[i].height; //��õ�i����Ӿ��εĸ߶�

		if (w0 <= tooSmalltoDrop || h0 <= tooSmalltoDrop)
			continue;
		cv::Mat src_roi = frame(cv::Rect(x0, y0, w0, h0));
		std::stringstream ss;
		
		ss << saveAdress << pos << "_"<< i << ".bmp" ;
		std::string s = ss.str();
		imwrite(s, src_roi);
	}
}
void KNN_BGS::diff2(Mat &cur,Mat &las)
{
	Mat cur_gray,las_gray;
	cv::cvtColor(cur,cur_gray, CV_BGR2GRAY);
	cv::cvtColor(las,las_gray, CV_BGR2GRAY);
	absdiff(cur_gray,las_gray,DiffMask);
	threshold( DiffMask, DiffMask, 30, 255 , 0 );
	medianBlur(DiffMask,DiffMask,3);    
	Mat element = getStructuringElement(MORPH_RECT, Size(dilateRatio, dilateRatio));
		//erode(mat, mat, element);
	dilate(DiffMask, DiffMask, element);
	//normalize(out, out, 255, 0, NORM_MINMAX);
}