#include <stdlib.h>

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "ros/ros.h"
#include "iostream"
#include <cv_bridge/cv_bridge.h>
#include "geometry_msgs/Twist.h"


enum CameraState
{
    COMPUTER = 0,
    ZED,
    REALSENSE
};
CameraState state = REALSENSE;

#define N 255 //灰度level
using namespace std;
using namespace cv;

//计算不同灰度值像素分布
void getGrayLevel(const Mat &img, int* Grayscale){
    for(int i=0;i<=N;i++)
        Grayscale[i]=0;
    for(int i=0;i<img.rows;i++){
        for(int j=0;j<img.cols;j++){
            int gray = (int)img.at<uchar>(i,j);
            Grayscale[gray]++;
        }
    }
}

//getHistImage()--画图像直方图
Mat getHistImage( Mat hist)
{
    Scalar color(172, 172, 100);//划线颜色
    Scalar Background(255,255,255);//背景颜色
    int thickness = 2;	//划线宽度
    int histss[256] = {0};

    /*** 第一步：下面计算不同灰度值的像素分布 ***/
    getGrayLevel(hist, histss);

    int histSize = 500;
    Mat histImage(histSize, histSize, CV_8UC3, Background );//绘制背景

    for (int h = 0; h < 256; h++) {

    /*** 第二步：画出像素的直方图分布 ***/
        int intensity = histss[h]; //直方图的高度
        line(histImage, Point(h*2, histSize - 1), Point(h*2, histSize - intensity / 2), color, thickness);
    }

    return histImage;
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
	ros::init(argc, argv, "exp1_node"); // 初始化 ROS 节点
    ros::NodeHandle n;
    ros::Subscriber camera_sub;
    ros::Publisher vel_pub;
    vel_pub = n.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
	VideoCapture capture;

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
		waitKey(1000);
	}

	Mat frame;//当前帧图片

	int Grayscale[N + 1];//灰度级
	int Grayscale2[N + 1];//均衡化以后的灰度级
	float Gray_f[N + 1];//频率
	int Gray_c[N + 1];//累计密度
    ros::Rate loop_rate(10); // 设置循环频率为10Hz
	while (ros::ok())
	{
        if(state == COMPUTER)
        {
            capture.read(frame);
            if (frame.empty())
            {
                printf("没有获取到电脑图像\n");
                continue;
            }
        }
        else if(state == ZED)
        {
            capture.read(frame);
            if (frame.empty())
            {
                printf("没有获取到ZED图像\n");
                continue;
            }
            frame = frame(cv::Rect(0,0,frame.cols/2,frame.rows));//截取zed的左目图片
        }
        else if(state == REALSENSE)
        {
            if(frame_msg.cols == 0)
            {
                printf("没有获取到realsense图像\n");
                ros::spinOnce();
                continue;
            }
			frame = frame_msg;
        }


		Mat frIn = frame; 
		Mat New;
		cvtColor(frIn,frIn,COLOR_RGB2GRAY,0);

    	/*** 第三步：直方图均衡化处理 ***/
        getGrayLevel(frIn, Grayscale);        
        int total = frIn.rows * frIn.cols;
        for(int i=0;i<=N;i++)
            Gray_f[i] = (float)Grayscale[i]/total; //频率
        Gray_c[0] = Gray_f[0];
        for(int i=1;i<=N;i++)
            Gray_c[i] = Gray_c[i-1] + Gray_f[i];
        for (int i=0;i<New.rows;i++)
        {
            for(int j=0;j<New.cols;j++)
            {
                int gray = (int)frIn.at<uchar>(i,j);
                int newgray = (int)(N*Gray_c[gray]+0.5); //均衡化公式
                New.at<uchar>(i,j) = saturate_cast<uchar>(newgray);
            }
        }
        getGrayLevel(New, Grayscale2);
	
		Mat last = getHistImage(New);
		Mat origi= getHistImage(frIn);
		imshow("his",last);//均衡化后直方图
		imshow("origi",origi);//原直方图
		imshow("Histed",New);//均衡化后图像
		imshow("Origin",frIn);//原图像


    	/*** 第四步：参考demo程序，添加让小车原地旋转代码 ***/
        geometry_msgs::Twist vel_msg;
        vel_msg.linear.x = 0.05;
        vel_msg.linear.y = 0.;
        vel_msg.angular.z = 0.5;
        vel_pub.publish(vel_msg);



        ros::spinOnce(); // 处理回调函数
        waitKey(5);
        loop_rate.sleep(); // 控制循环速率
	
	}
	return 0;
}