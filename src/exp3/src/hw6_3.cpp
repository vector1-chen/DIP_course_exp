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

#define GRADIENT_THRESHOLD 200
#define HOUGH_THRESHOLD 90
#define CIRCLE_VOTE_THRESHOLD 100

/***************函数声明，相关参数自行修改***************/
void EdgeDetector(Mat input, Mat output);
void HoughLines(Mat input_gray, Mat output);
void HoughCircles(Mat input_gray, Mat output);
void SobelOperator(Mat &input_gray, Mat &magnitude, Mat &angle);
void gaussianFilter(Mat &input, Mat &output, double sigma);

//Mat raw;

int main(int argc, char *argv[])
{
        //Mat raw_line = imread("/home/eaibot/dip_ws/src/exp3/data/lines.png");
        Mat raw_circle = imread("/home/haojiechen/file_mess/DIP/dip_ws/src/exp3/data/circles.png");
        
        if (!raw_circle.data)
        {
                cout << "Error: Cannot load circle image!" << endl;
                return -1;
        }

        imshow("raw_circle", raw_circle);
        Mat gray_circle = raw_circle.clone();
        cvtColor(raw_circle, gray_circle, COLOR_BGR2GRAY);

        /***************调用霍夫圆变换***************/
        Mat gray_circle_c = Mat::zeros(gray_circle.size(), CV_8UC1);
        Mat raw_circle_edit = raw_circle.clone();
        HoughCircles(gray_circle, raw_circle_edit);
        imshow("HoughCircles", raw_circle_edit);
        
        waitKey(0);
        return 0;
}
/***************下面实现EdgeDetector()函数***************/
void EdgeDetector(Mat input, Mat output){

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
    imshow("edge",output);
}

/***************下面实现HoughLines()函数***************/
void HoughLines(Mat input_gray, Mat output){
    Mat input = input_gray.clone();
    EdgeDetector(input_gray, input);
    //output = input.clone();
    // 参数定义
    int width = input.cols;
    int height = input.rows;
    double diag_len = sqrt(width * width + height * height);
    int r_max = (int)diag_len;
    int r_min = -r_max;
    int r_range = r_max - r_min;
    int theta_range = 180;
    // 创建累加器
    Mat accumulator = Mat::zeros(r_range, theta_range, CV_32SC1);

    // 遍历原图
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (input.at<uchar>(y, x) == 255)
            {
                for (int theta = 0; theta < theta_range; theta++)
                {
                    double theta_rad = theta * CV_PI / 180.0;
                    int rho = (int)(x * cos(theta_rad) + y * sin(theta_rad));
                    rho += r_max;
                    accumulator.at<int>(rho, theta)++;
                }
            }
        }
    }

    for (int r = 0; r < accumulator.rows; r++)
    {
        for (int t = 0; t < accumulator.cols; t++)
        {
            if (accumulator.at<int>(r, t) > HOUGH_THRESHOLD)
            {
                double rho = r - r_max;
                double theta = t * CV_PI / 180.0;
                Point pt1, pt2;
                double a = cos(theta), b = sin(theta);
                double x0 = a * rho, y0 = b * rho;
                pt1.x = cvRound(x0 + 1000 * (-b));
                pt1.y = cvRound(y0 + 1000 * (a));
                pt2.x = cvRound(x0 - 1000 * (-b));
                pt2.y = cvRound(y0 - 1000 * (a));
                line(output, pt1, pt2, Scalar(0,0,255), 1, LINE_AA);
            }
        }
    }
}

/***************下面实现HoughCircles()函数***************/
void HoughCircles(Mat input_gray, Mat output){
    Mat input_filter = input_gray.clone();
    gaussianFilter(input_gray, input_filter, 1);
    Mat input = input_gray.clone();
    EdgeDetector(input_gray, input);
    
    // 参数定义
    int width = input.cols;
    int height = input.rows;
    int min_radius = 10;
    int max_radius = min(width, height)/2;
    int radius_range = max_radius - min_radius;


    // Sobel处理
    Mat magnitude = Mat::zeros(input.size(), CV_64F);
    Mat angle = Mat::zeros(input.size(), CV_64F);
    SobelOperator(input_filter, magnitude, angle);

    // 估计圆心
    Mat center_votes = Mat::zeros(height, width, CV_32SC1);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (input.at<uchar>(y, x) == 255 && magnitude.at<double>(y, x) > 0)
            {
                double theta = angle.at<double>(y, x);
                for (int r = -max_radius; r < max_radius; r += 2)
                {
                    int a = cvRound(x - r * cos(theta));
                    int b = cvRound(y - r * sin(theta));
                    if (a >= 0 && a < width && b >= 0 && b < height)
                    {
                        center_votes.at<int>(b, a)++;
                    }
                }
            }
        }
    }
    for (int y = 0; y < center_votes.rows; y++)
    {
        for (int x = 0; x < center_votes.cols; x++)
        {
            if (center_votes.at<int>(y, x) > CIRCLE_VOTE_THRESHOLD)
            {
                // 估计半径
                vector<double> theta_votes(max_radius - min_radius, 0);
                for (int yy = 0; yy < height; yy++)
                {
                    for (int xx = 0; xx < width; xx++)
                    {
                        if (input.at<uchar>(yy, xx) == 255)
                        {
                            double r = sqrt((xx - x) * (xx - x) + (yy - y) * (yy - y));
                            if (r >= min_radius && r <= max_radius)
                            {
                                theta_votes[r - min_radius]+= 1 / (2 * CV_PI * r);
                            }
                        }
                    }
                }
                //弧度制转为角度制
                for (int r = min_radius; r < max_radius; r++)
                {
                    if (theta_votes[r - min_radius] > 0.65) circle(output, Point(x, y), r, Scalar(0,0,255), 4);
                }
            }
        }
    }
}


/***************SobelOperator()函数***************/
void SobelOperator(Mat &input_gray, Mat &magnitude, Mat &angle) {
    Mat grad_x, grad_y;
    
    // 使用OpenCV内置的Sobel函数
    Sobel(input_gray, grad_x, CV_64F, 1, 0, 3);  // x方向
    Sobel(input_gray, grad_y, CV_64F, 0, 1, 3);  // y方向
    
    // 计算幅值和角度
    cartToPolar(grad_x, grad_y, magnitude, angle);
}



// 空域高斯滤波器函数
void gaussianFilter(Mat &input, Mat &output, double sigma)
{

    //利用高斯函数生成模板
    int T_size = 9;                                    // 模板大小
    Mat Template = Mat::zeros(T_size, T_size, CV_64F); // 初始化模板矩阵
    int center = T_size / 2;                    // 模板中心位置
    double sum = 0.0;
    
    for (int i = 0; i < T_size; i++)
    {
        for (int j = 0; j < T_size; j++)
        {

            /*** 第三步：在此处填充高斯滤波模板元素计算代码 ***/
            Template.at<double>(i, j) = exp(-((i - center) * (i - center) + (j - center) * (j - center)) / (2 * sigma * sigma)) / (2 * CV_PI * sigma * sigma);

            sum += Template.at<double>(i, j); //用于归一化模板元素
        }
    }


    for (int i = 0; i < T_size; i++)
    {
        for (int j = 0; j < T_size; j++)
        {

            /*** 第四步：在此处填充模板归一化代码 ***/
            Template.at<double>(i, j) /= sum;
        }
    }
    // 卷积
    Mat temp_output = Mat::zeros(input.size(), CV_64F);

    /*** 第五步：同第二步，填充模板与输入图像的卷积代码 ***/ 
    int edge_size = (T_size - 1) / 2;
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            if ((j < edge_size || j >= input.cols - edge_size || i < edge_size || i >= input.rows - edge_size)){
                temp_output.at<double>(i, j) = (double)input.at<uchar>(i, j);
                continue;
            }
            double total = 0.0;
            for (int m = -edge_size; m <= edge_size; m++)
            {
                for (int n = -edge_size; n <= edge_size; n++)
                {
                    total += Template.at<double>(m + edge_size, n + edge_size) * (double)input.at<uchar>(i + m, j + n);
                }
            }
            temp_output.at<double>(i, j) = total;
        }
    }


    temp_output.convertTo(output, CV_8UC1);
    imshow("spatial_filtered_image", output);
}