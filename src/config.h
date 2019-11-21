#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <fstream>
#include <string.h>
#include <string>
#include <stdio.h>
#include <sys/time.h>
#include <sys/timeb.h>
void getTimesSecf(char *param);
void getTimesSec(char *param);
bool _str_cmp(char* a, char *b);
void get_param_mssd_img(std::string &in,std::string &out);
void get_param_mssd_video_knn(std::string &in,std::string &out);
void get_param_mssd_video(std::string &in,std::string &out);
void get_param_mms_cvCaptrue(int & dev);
void get_param_mms_V4L2(std::string &dev);
void get_camera_size(int &wid,int &hgt);
void get_show_img(bool &show);
void get_captrue_save_data_floder(std::string &imag,std::string &video);
void get_captrue_data_save_video_mode(int &mode);
void get_captrue_data_save_img_mode(int &mode);
void get_move_percent(double & move);
void get_move_buff_cnt(int & cnt);
void get_roi_limit(bool &roi);
#endif