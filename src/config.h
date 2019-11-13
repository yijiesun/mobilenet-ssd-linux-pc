#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <fstream>
#include <string.h>
#include <string>
#include <stdio.h>

bool _str_cmp(char* a, char *b);
void get_param_mssd_img(std::string &in,std::string &out);
void get_param_mssd_video_knn(std::string &in,std::string &out);
void get_param_mssd_video(std::string &in,std::string &out);
void get_param_mms_cvCaptrue(int & dev);
#endif