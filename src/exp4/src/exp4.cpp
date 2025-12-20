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
CameraState state = REALSENSE;  // 修改为使用电脑摄像头


using namespace cv;
using namespace std;
void Gaussian(const Mat &input, Mat &output, double sigma)
{
    if (output.rows != input.rows || output.cols != input.cols || output.channels() != input.channels())
        return;
    int kernel_size = 9;
    double gaussian_kernel[kernel_size][kernel_size];

    /*** 第一步：结合实验二，在此处填充高斯滤波代码 ***/
    int half_size = kernel_size / 2;
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            Vec3d sum(0.0, 0.0, 0.0);
            double weight = 0.0;
            for (int m = -half_size; m <= half_size; m++)
            {
                for (int n = -half_size; n <= half_size; n++)
                {
                    int x = j + n;
                    int y = i + m;
                    if (x >= 0 && x < input.cols && y >= 0 && y < input.rows)
                    {
                        double gaussian_value = exp(-(m * m + n * n) / (2 * sigma * sigma));
                        Vec3b pixel = input.at<Vec3b>(y, x);  // 获取像素
                        sum[0] += pixel[0] * gaussian_value;  // 分别计算每个通道
                        sum[1] += pixel[1] * gaussian_value;
                        sum[2] += pixel[2] * gaussian_value;
                        weight += gaussian_value;
                    }
                }
            }
            if (weight > 0)
            {
                output.at<Vec3b>(i, j) = Vec3b(
                    (uchar)(sum[0] / weight),
                    (uchar)(sum[1] / weight),
                    (uchar)(sum[2] / weight)
                );
            }
        }
    }
}

void BGR2HSV(const Mat &input, Mat &output)
{
    if (input.rows != output.rows ||
        input.cols != output.cols ||
        input.channels() != 3 ||
        output.channels() != 3)
        return;

	for(int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{

        /*** 第二步：在此处填充RGB转HSV代码 ***/
            // 获取RGB像素值
            Vec3b bgr = input.at<Vec3b>(i, j);
            double b = bgr[0] / 255.0;
            double g = bgr[1] / 255.0;
            double r = bgr[2] / 255.0;

            // 计算最大值和最小值
            double max_val = max({r, g, b});
            double min_val = min({r, g, b});
            
            double h, s, v;
            
            // 计算V (Value/亮度)
            v = (b + g + r) / 3.0;
            
            // 计算S (Saturation/饱和度)
            s = 1- 3 * min_val / (r + g + b + 1e-6);
            
            // 计算H (Hue/色相)
            double theta = acos(0.5 * ((r - g) + (r - b)) / sqrt((r - g) * (r - g) + (r - b) * (g - b)) + 1e-6);
            if (b <= g) {
                h = theta * 180 / CV_PI;
            } else {
                h = 360 - theta * 180 / CV_PI;
            }
            // 确保H在[0, 360)范围内
            if (h < 0) {
                h += 360;
            }
            
            // 将HSV值转换为0-255范围并存储
            output.at<Vec3b>(i, j)[0] = (uchar)(h / 2);        // H: 0-179
            output.at<Vec3b>(i, j)[1] = (uchar)(s * 255);      // S: 0-255
            output.at<Vec3b>(i, j)[2] = (uchar)(v * 255);      // V: 0-255
        

        }
    }
}


void ColorSplitManual(const Mat &hsv_input, Mat &grey_output, const string window)
{
	static int hmin = 0;
	static int hmax = 255;
	static int smin = 0;
	static int smax = 255;
	static int vmin = 0;
	static int vmax = 255;
	createTrackbar("Hmin", window, &hmin, 255);
	createTrackbar("Hmax", window, &hmax, 255);
	createTrackbar("Smin", window, &smin, 255);
	createTrackbar("Smax", window, &smax, 255);
	createTrackbar("Vmin", window, &vmin, 255);
	createTrackbar("Vmax", window, &vmax, 255);

    /*** 第三步：在此处填充阈值分割代码代码 ***/
    // 遍历HSV
    for (int i = 0; i < hsv_input.rows; i++)
    {
        for (int j = 0; j < hsv_input.cols; j++)
        {
            // 获取HSV像素值
            Vec3b hsv_pixel = hsv_input.at<Vec3b>(i, j);
            int h = hsv_pixel[0];  // H(0-179)
            int s = hsv_pixel[1];  // S(0-255)
            int v = hsv_pixel[2];  // V(0-255)

            // 检查像素是否在设定的HSV阈值范围内
            if (h >= hmin && h <= hmax && 
                s >= smin && s <= smax && 
                v >= vmin && v <= vmax)
            {
                // 在阈值范围内，设为白色(255)
                grey_output.at<uchar>(i, j) = 255;
            }
            else
            {
                // 不在阈值范围内，设为黑色(0)
                grey_output.at<uchar>(i, j) = 0;
            }
        }
    }
}

void ColorSplitAuto(const Mat &hsv_input, Mat &bgr_output, vector<vector<Point>> &contours, int hmin, int hmax, int smin, int smax, int vmin, int vmax)
{
    int rw = hsv_input.rows;
	int cl = hsv_input.cols;
    Mat color_region(rw, cl, CV_8UC1);

    /*** 第五步：利用已知的阈值获取颜色区域二值图 ***/
    for (int i = 0; i < rw; i++)
    {
        for (int j = 0; j < cl; j++)
        {
            Vec3b hsv_pixel = hsv_input.at<Vec3b>(i, j);
            int h = hsv_pixel[0];  // H(0-179)
            int s = hsv_pixel[1];  // S(0-255)
            int v = hsv_pixel[2];  // V(0-255)
            
            if (h >= hmin && h <= hmax && 
                s >= smin && s <= smax && 
                v >= vmin && v <= vmax)
            {
                color_region.at<uchar>(i, j) = 255;
            }
            else
            {
                color_region.at<uchar>(i, j) = 0;
            }
        }
    }
    imshow("autosplit:color_region", color_region);

    /* 获取多边形轮廓 */
    vector<Vec4i> hierarchy;
	findContours(color_region, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	vector<vector<Point>> lines(contours.size());
    /* 利用多项式近似平滑轮廓 */
	for(int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(contours[i], lines[i],9,true);
	}
	drawContours(bgr_output, lines, -1,Scalar(0, 0, 255), 2, 8);
}


void GetROI(const Mat &input, Mat &output, const vector<vector<Point>> &contour)
{
    /* 第六步：补充获取颜色区域代码，可使用drawContours函数 */
    // 首先将输出图像初始化为黑色
    output = Mat::zeros(input.size(), input.type());
    
    // 如果没有轮廓，直接返回
    if (contour.empty())
        return;
    
    // 创建掩码图像
    Mat mask = Mat::zeros(input.rows, input.cols, CV_8UC1);
    
    // 使用drawContours填充轮廓区域，生成掩码
    for (int i = 0; i < contour.size(); i++)
    {
        drawContours(mask, contour, i, Scalar(255), FILLED);
    }
    
    // 将原图像中轮廓区域的像素复制到输出图像
    input.copyTo(output, mask);
}

int CountROIPixel(const Mat &input)
{
	int cnt = 0;

    /* 第七步：补充获取颜色区域像素个数的代码 */
    // 遍历图像的每个像素
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            // 检查像素是否为非零（非黑色）
            Vec3b pixel = input.at<Vec3b>(i, j);
            if (pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0)
            {
                cnt++;
            }
        }
    }

    return cnt;
}


/*** 第四步：在第三步基础上修改各颜色阈值 ***/
//{hmin, hmax, smin, smax, vmin, vmax}
int red_thresh[6] = {0};
int green_thresh[6] = {30, 90, 30, 255, 30, 255};
int blue_thresh[6] = {0};
int yellow_thresh[6] = {0};

Mat frame_msg;
void rcvCameraCallBack(const sensor_msgs::Image::ConstPtr& img)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    cv_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
    frame_msg = cv_ptr->image;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "exp4_node"); // 初始化 ROS 节点
    ros::NodeHandle n;
	ros::Publisher vel_pub = n.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    ros::Subscriber camera_sub;
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
    }
    

    Mat frIn;
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


        // 空域高斯滤波
        Mat filter(frIn.size(), CV_8UC3);
        Gaussian(frIn, filter, 3);
        imshow("filter",filter);

        // RGB转HSV
        Mat hsv(frIn.size(), CV_8UC3);
        BGR2HSV(filter, hsv);
        imshow("hsv",hsv);

        // 手动颜色分割
        Mat grey(frIn.rows, frIn.cols, CV_8UC1);
        ColorSplitManual(hsv, grey, "split");
        imshow("split", grey);
        
        int colors = 0;
        int maxs_color_num = 0;
        /* 目标颜色检测 */

	    Mat tmp_line = frIn.clone();
	    Mat tmp_roi = Mat::zeros(frIn.size(), CV_8UC3);
        vector<vector<Point>> contours_r;
        	ColorSplitAuto(hsv, tmp_line, contours_r, green_thresh[0], green_thresh[1], green_thresh[2],
				   green_thresh[3], green_thresh[4], green_thresh[5]);
	    GetROI(frIn, tmp_roi, contours_r);
	    int green_color_num = CountROIPixel(tmp_roi);
        
        // 创建并显示绿色ROI窗口
        namedWindow("green_roi", WINDOW_AUTOSIZE);
        imshow("green_roi", tmp_line);
        
        /* 第八步：结合给出的检测红颜色的代码框架，给出控制小车运动的代码 */






        geometry_msgs::Twist vel;
        vel.linear.x = 0;
        vel.linear.y = 0;
        vel.linear.z = 0;
        vel.angular.x = 0;
        vel.angular.y = 0;
        vel.angular.z = 0;
        if(maxs_color_num)
        {
            switch(colors)
            {
                case 0:
                    vel.linear.x = 0.5;
                    break;
                case 1:
                    vel.linear.x = -0.5;
                    break;
                case 2:
                    vel.angular.z = 0.4;
                    break;
                case 3:
                    vel.angular.z = -0.4;
                    break;
            }
        }
        vel_pub.publish(vel);




        ros::spinOnce();
        waitKey(5);
    }
    return 0;
}