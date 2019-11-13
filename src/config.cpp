#include "config.h"
using namespace std;

bool _str_cmp(char* a, char *b)
{	
	int sum = 0;
	for (int i = 0; b[i] != '\0'; i++)
		sum++;
	char tmp[200] = {""};
	strncpy(tmp, a + 0, sum);
	for (int i = 0; a[i] != '\0'; i++)
	{
		if (a[i] == '\n')
			a[i] = (char)NULL;
	}
	return !strcmp(tmp,b);
}

void get_param_mssd_img(std::string &in,std::string &out)
{
    char img_in[200];
    char img_out[200];
    FILE *read_setup;
    string config_file = "../src/config.txt";
	if ((read_setup = fopen(config_file.c_str(), "r")) == NULL) {
		puts("Fail to open config.txt!");
		exit(0);
	}
	char str[200];
	while (fgets(str, 100, read_setup) != NULL) {
        if (_str_cmp(str, (char *)"mssd_img_in"))
			strncpy(img_in, str + 12, 200);
		else if (_str_cmp(str, (char *)"mssd_img_out"))
			strncpy(img_out, str + 13, 200);
		
    }

    in = img_in;
    out = img_out;
}
void get_param_mssd_video_knn(std::string &in,std::string &out)
{
    char img_in[200];
    char img_out[200];
    FILE *read_setup;
    string config_file = "../src/config.txt";
	if ((read_setup = fopen(config_file.c_str(), "r")) == NULL) {
		puts("Fail to open config.txt!");
		exit(0);
	}
	char str[200];
	while (fgets(str, 100, read_setup) != NULL) {
        if (_str_cmp(str, (char *)"mssd_video_knn_in"))
			strncpy(img_in, str + 18, 200);
		else if (_str_cmp(str, (char *)"mssd_video_knn_out"))
			strncpy(img_out, str + 19, 200);
		
    }

    in = img_in;
    out = img_out;

}
void get_param_mssd_video(std::string &in,std::string &out)
{
    char img_in[200];
    char img_out[200];
    FILE *read_setup;
    string config_file = "../src/config.txt";
	if ((read_setup = fopen(config_file.c_str(), "r")) == NULL) {
		puts("Fail to open config.txt!");
		exit(0);
	}
	char str[200];
	while (fgets(str, 100, read_setup) != NULL) {
        if (_str_cmp(str, (char *)"mssd_video_in"))
			strncpy(img_in, str + 14, 200);
		else if (_str_cmp(str, (char *)"mssd_video_out"))
			strncpy(img_out, str + 15, 200);
		
    }

    in = img_in;
    out = img_out;

}

void get_param_mms_cvCaptrue(int & dev)
{
    FILE *read_setup;
    string config_file = "../src/config.txt";
	if ((read_setup = fopen(config_file.c_str(), "r")) == NULL) {
		puts("Fail to open config.txt!");
		exit(0);
	}
	char str[200];
	while (fgets(str, 100, read_setup) != NULL) {
       if (_str_cmp(str, (char *)"mssd_cvCaptrue_dev"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%d", &dev);
		}
		
    }

}