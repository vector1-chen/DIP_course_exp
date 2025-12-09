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

enum CameraState
{
    COMPUTER = 0,
    ZED,
    REALSENSE
};
CameraState state = COMPUTER;
#define pi 3.1415926
using namespace cv;

//空域均值滤波函数
void meanFilter(Mat &input)
{

    //生成模板
    int T_size = 9; 
    //int T_size = 3;                                   // 模板大小
    Mat Template = Mat::zeros(T_size, T_size, CV_64F); // 初始化模板矩阵
    /*** 第一步：在此处填充均值滤波模板 ***/


    // 卷积
    Mat output = Mat::zeros(input.size(), CV_64F);

    /*** 第二步：填充模板与输入图像的卷积代码 ***/    



    output.convertTo(output, CV_8UC1);
    imshow("mean_filtered_image", output);
}
// 空域高斯滤波器函数
void gaussianFilter(Mat &input, double sigma)
{

    //利用高斯函数生成模板
    int T_size = 9;                                    // 模板大小
    Mat Template = Mat::zeros(T_size, T_size, CV_64F); // 初始化模板矩阵
    int center = round(T_size / 2);                    // 模板中心位置
    double sum = 0.0;
    
    for (int i = 0; i < T_size; i++)
    {
        for (int j = 0; j < T_size; j++)
        {

            /*** 第三步：在此处填充高斯滤波模板元素计算代码 ***/

            
            
            sum += Template.at<double>(i, j); //用于归一化模板元素
        }
    }


    for (int i = 0; i < T_size; i++)
    {
        for (int j = 0; j < T_size; j++)
        {

            /*** 第四步：在此处填充模板归一化代码 ***/

        }
    }
    // 卷积
    Mat output = Mat::zeros(input.size(), CV_64F);

    /*** 第五步：同第二步，填充模板与输入图像的卷积代码 ***/ 





    output.convertTo(output, CV_8UC1);
    imshow("spatial_filtered_image", output);
}
// 锐化空域滤波
void sharpenFilter(Mat &input)
{

    //生成模板
    int T_size = 3;                                    // 模板大小
    Mat Template = Mat::zeros(T_size, T_size, CV_64F); // 初始化模板矩阵
    /*** 第六步：填充锐化滤波模板 ***/   

    // 卷积
    Mat output = Mat::zeros(input.size(), CV_64F);

    /*** 第七步：同第二步，填充模板与输入图像的卷积代码 ***/    




    output.convertTo(output, CV_8UC1);
    imshow("sharpen_filtered_image", output);
}
// 膨胀函数
void Dilate(Mat &Src)
{
    Mat Dst = Src.clone();
    Dst.convertTo(Dst, CV_64F);

    /*** 第八步：填充膨胀代码 ***/    



    Dst.convertTo(Dst, CV_8UC1);
    imshow("dilate", Dst);
}
// 腐蚀函数
void Erode(Mat &Src)
{
    Mat Dst = Src.clone();
    Dst.convertTo(Dst, CV_64F);

    /*** 第九步：填充腐蚀代码 ***/    



    Dst.convertTo(Dst, CV_8UC1);
    imshow("erode", Dst);
}

Mat frame_msg;
void rcvCameraCallBack(const sensor_msgs::Image::ConstPtr& img)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    cv_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
    frame_msg = cv_ptr->image;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "exp2_node"); // 初始化 ROS 节点
    ros::NodeHandle n;
    ros::Subscriber camera_sub;
    VideoCapture capture;
    capture.open(0);     

    if(state == COMPUTER)
    {
        capture.open(0);     
        if (!capture.isOpened())
        {
            printf("电脑摄像头没有正常打开\n");
            return 0;
        }
        waitKey(1000);
    }
    else if(state == ZED)
    {
        capture.open(4);     
        if (!capture.isOpened())
        {
            printf("ZED摄像头没有正常打开\n");
            return 0;
        }
        waitKey(1000);
    }
    else if(state == REALSENSE)
    {
        camera_sub = n.subscribe("/camera/color/image_raw",1,rcvCameraCallBack);
    }

    Mat frIn;                                        // 当前帧图片
    while (ros::ok())
    {

        if(state == COMPUTER)
        {
            capture.read(frIn);
            if (frIn.empty())
            {
                printf("没有获取到电脑图像\n");
                continue;
            }
        }
        else if(state == ZED)
        {
            capture.read(frIn);
            if (frIn.empty())
            {
                printf("没有获取到ZED图像\n");
                continue;
            }
            frIn = frIn(cv::Rect(0,0,frIn.cols/2,frIn.rows));//截取zed的左目图片
        }
        else if(state == REALSENSE)
        {
            if(frame_msg.cols == 0)
            {
                printf("没有获取到realsense图像\n");
                ros::spinOnce();
                continue;
            }
            frIn = frame_msg;
        }


        cvtColor(frIn, frIn, COLOR_BGR2GRAY);
        imshow("original_image", frIn);
        //空域均值滤波
	    meanFilter(frIn);
	
        // 空域高斯滤波
        double sigma = 2.5;
        gaussianFilter(frIn, sigma);

        //空域锐化滤波
        sharpenFilter(frIn);

        // 膨胀函数
        Dilate(frIn);

        // 腐蚀函数
        Erode(frIn);

        ros::spinOnce();
        waitKey(5);
    }
    return 0;
}
