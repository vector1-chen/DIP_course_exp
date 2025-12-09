#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

ros::Publisher vel_pub;

using namespace cv;

enum CameraState
{
    COMPUTER = 0,
    ZED,
    REALSENSE
};
CameraState state = REALSENSE;


Mat frame_msg;
void rcvCameraCallBack(const sensor_msgs::Image::ConstPtr& img)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    cv_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
    frame_msg = cv_ptr->image;
}

int main(int argc, char ** argv)
{
    ros::init(argc, argv, "car_demo");
    ros::NodeHandle n;
    ros::Subscriber camera_sub;
	VideoCapture capture;

    vel_pub = n.advertise<geometry_msgs::Twist>("/cmd_vel", 10);

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
    ros::Rate loop_rate(10); // 设置循环频率为10Hz
    geometry_msgs::Twist vel_msg;

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

        // 发布消息，让小车绕一小半径旋转
        vel_msg.linear.x = 0.05;
        vel_msg.linear.y = 0.;
        vel_msg.angular.z = 0.5;
        vel_pub.publish(vel_msg);


        imshow("frame",frame);

        ros::spinOnce(); // 处理回调函数
        waitKey(5);
        loop_rate.sleep(); // 控制循环速率
    }

}