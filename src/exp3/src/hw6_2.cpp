#include <stdlib.h>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float32.h"
#include <geometry_msgs/Twist.h>
#include "sensor_msgs/Image.h"
#include <math.h>
#include <cv_bridge/cv_bridge.h>

using namespace std;
using namespace cv;

#define GRADIENT_THRESHOLD 120

void EdgeDetector(const Mat &input, Mat &output);

int main(int argc, char *argv[])
{
        Mat raw_lena = imread("/home/haojiechen/file_mess/DIP/dip_ws/src/exp3/data/lena.webp");
        if (!raw_lena.data)
        {
                cout << "Error: Cannot load image!" << endl;
                return -1;
        }

        imshow("raw_lena", raw_lena);
        Mat gray_lena = raw_lena.clone();
        cvtColor(raw_lena, gray_lena, COLOR_BGR2GRAY);

        /****************调用边缘检测函数****************/
        Mat gray_lena_e = Mat::zeros(gray_lena.size(), CV_8UC1);
        EdgeDetector(gray_lena, gray_lena_e);
        imshow("EdgeDetector_lena", gray_lena_e);
        
        waitKey(0);
        return 0;
}

void EdgeDetector(const Mat &input, Mat &output){

    int T_size = 5;
    Mat LoG = Mat::zeros(T_size, T_size, CV_64F); // 初始化模板矩阵LoG
    LoG.at<double>(0, 2) = -1;
    LoG.at<double>(1, 1) = -1; LoG.at<double>(1, 2) = -2; LoG.at<double>(1, 3) = -1;
    LoG.at<double>(2, 0) = -1; LoG.at<double>(2, 1) = -2; LoG.at<double>(2, 2) = 16; LoG.at<double>(2, 3) = -2; LoG.at<double>(2, 4) = -1;
    LoG.at<double>(3, 1) = -1; LoG.at<double>(3, 2) = -2; LoG.at<double>(3, 3) = -1;
    LoG.at<double>(4, 2) = -1;

    int edge_size = (T_size - 1) / 2;
    Mat temp_output = Mat::zeros(input.size(), CV_64F);
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            if (j < edge_size || j >= input.cols - edge_size || i < edge_size || i >= input.rows - edge_size){
                temp_output.at<double>(i, j) = (double)input.at<uchar>(i, j);
                continue;
            }
            double total = 0.0;
            for (int m = -edge_size; m <= edge_size; m++)
            {
                for (int n = -edge_size; n <= edge_size; n++)
                {
                    total += LoG.at<double>(m + edge_size, n + edge_size) * (double)input.at<uchar>(i + m, j + n);
                }
            }
            temp_output.at<double>(i, j) = total;
        }    
    }
    temp_output.convertTo(output, CV_8UC1);
    for (int i = 0; i < output.rows; i++)
    {
        for (int j = 0; j < output.cols; j++)
        {
            if(output.at<uchar>(i,j) > GRADIENT_THRESHOLD)
                output.at<uchar>(i,j) = 255;
            else
                output.at<uchar>(i,j) = 0;
        }
    }
}