#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Twist.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

enum CameraState
{
    COMPUTER = 0,
    ZED,
    REALSENSE
};
CameraState state = COMPUTER;  // 修改为使用电脑摄像头

// 定义机器人状态
enum State {
    STATE_LANE_FOLLOW,
    STATE_OBSTACLE_1_AVOID,  // 绿色路线：向左绕行第一个障碍
    STATE_OBSTACLE_2_AVOID,  // 绿色路线：向右绕行第二个障碍
    STATE_SEARCH_TARGET,
    STATE_TRACK_TARGET
};

class RobotVisionController {
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Publisher vel_pub_;
    cv::VideoCapture capture;
    
    State current_state_;
    int lost_cones_counter_;        // 锥桶丢失计数，用于判断驶出赛道
    int avoid_timer_;               // 避障计时器
    int obstacle_count_;            // 避障计数器 (1 or 2)
    
    // 橙色锥桶 HSV 范围 (默认值，需要校准)
    const Scalar cone_lower = Scalar(0, 100, 100); 
    const Scalar cone_upper = Scalar(20, 255, 255);
    
    // 红色目标数字 HSV 范围 (默认值，需要校准)
    const Scalar target_lower = Scalar(160, 100, 100);
    const Scalar target_upper = Scalar(180, 255, 255);

    // 控制参数
    const double KP_LANE = 0.006;   // 循迹比例控制系数
    const double KI_LANE = 0.001;   // 循迹积分控制系数
    double integral_lane = 0.0; // 循迹积分误差
    const double KP_TRACK = 0.008;  // 跟踪比例控制系数
    const double KI_TRACK = 0.002;  // 跟踪积分控制系数
    double integral_track = 0.0; // 跟踪积分误差
    const double LINEAR_SPEED = 0.2; // 默认线速度 (m/s)
    const int IMG_CENTER_X = 320;    // 假设图像宽度 640/2

public:
    RobotVisionController() : it_(nh_), current_state_(STATE_LANE_FOLLOW), lost_cones_counter_(0), avoid_timer_(0), obstacle_count_(0) {
        // 订阅摄像头话题 - 统一使用 ROS 话题订阅
        std::string camera_topic;
        
        if(state == COMPUTER)
        {
            camera_topic = "/usb_cam/image_raw";  // 电脑摄像头 ROS 话题
            ROS_INFO("Using COMPUTER camera mode");
        }
        else if(state == ZED)
        {
            camera_topic = "/zed/zed_node/rgb/image_rect_color";  // ZED 相机话题
            ROS_INFO("Using ZED camera mode");
        }
        else if(state == REALSENSE)
        {
            camera_topic = "/camera/color/image_raw";  // RealSense 话题
            ROS_INFO("Using REALSENSE camera mode");
        }
        
        // 统一订阅话题，使用 imageCb 回调
        image_sub_ = it_.subscribe(camera_topic, 1, &RobotVisionController::imageCb, this);
        ROS_INFO("Subscribed to camera topic: %s", camera_topic.c_str());
        
        waitKey(1000);

        // 发布速度话题 cmd_vel
        vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
        
        ROS_INFO("Robot Vision Controller Initialized in LANE_FOLLOW mode.");
        cv::namedWindow("Debug Mask");
    }

    ~RobotVisionController() {
        cv::destroyWindow("Debug Mask");
    }

    void imageCb(const sensor_msgs::ImageConstPtr& msg) {
        cv_bridge::CvImageConstPtr cv_ptr;
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
        processImage(cv_ptr->image);
    }

    void processImage(Mat img) {
        geometry_msgs::Twist cmd;
        Mat hsv;
        
        // 预处理：高斯模糊去噪，转换HSV
        GaussianBlur(img, img, Size(5, 5), 0);
        cvtColor(img, hsv, COLOR_BGR2HSV);

        // 状态
        switch (current_state_) {
            case STATE_LANE_FOLLOW:
                handleLaneFollow(hsv, cmd);
                break;
            
            case STATE_OBSTACLE_1_AVOID:
            case STATE_OBSTACLE_2_AVOID:
                handleObstacleAvoidance(hsv, cmd);
                break;
            
            case STATE_SEARCH_TARGET:
                handleTargetSearch(hsv, cmd);
                break;

            case STATE_TRACK_TARGET:
                handleTargetTracking(hsv, cmd);
                break;
        }

        vel_pub_.publish(cmd);
        waitKey(3);
    }

    /**
     * 循迹模式：基于锥桶重心进行路径控制
     */
    void handleLaneFollow(Mat hsv, geometry_msgs::Twist &cmd) {
        Mat mask;
        inRange(hsv, cone_lower, cone_upper, mask);
        
        // 形态学操作：开运算去噪
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);

        // 寻找轮廓
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (contours.empty()) {
            lost_cones_counter_++;
            if (lost_cones_counter_ > 50) { // 连续丢失超过50帧，切换到寻找模式
                current_state_ = STATE_SEARCH_TARGET;
                ROS_INFO("Cones lost, switching to Target Search Mode.");
            }
            cmd.linear.x = 0.0;
            return;
        } else {
            lost_cones_counter_ = 0;
        }

        double sum_x = 0;
        int contour_count = 0;
        bool obstacle_detected = false;

        for (const auto& contour : contours) {
            Moments M = moments(contour);
            if (M.m00 > 0) {
                double cx = M.m10 / M.m00;
                double area = contourArea(contour);

                // **障碍物检测启发式：** 面积巨大且靠近图像中心
                if (area > 30000 && cx > 200 && cx < 440) { // 阈值需调试
                    obstacle_detected = true;
                }
                
                sum_x += cx;
                contour_count++;
            }
        }
        
        // 绿色路线避障触发逻辑
        if (obstacle_detected) {
            obstacle_count_++;
            avoid_timer_ = 0; // 重置计时器
            if (obstacle_count_ == 1) {
                current_state_ = STATE_OBSTACLE_1_AVOID;
                ROS_INFO("Obstacle 1 detected! Avoiding Left (Green Route).");
                return;
            } else if (obstacle_count_ == 2) {
                current_state_ = STATE_OBSTACLE_2_AVOID;
                ROS_INFO("Obstacle 2 detected! Avoiding Right (Green Route).");
                return;
            }
            // 超过两个障碍，忽略
        }

        // 普通循迹控制 (基于所有锥桶的平均重心)
        double avg_x = sum_x / contour_count;
        double error = IMG_CENTER_X - avg_x; // 错误：中心偏差

        // PI 控制器
        integral_error_ += error;
        if (integral_error_ > 1000) integral_error_ = 1000;
        if (integral_error_ < -1000) integral_error_ = -1000;
        cmd.angular.z = KP_LANE * error + KI_LANE * integral_error_;
        cmd.linear.x = LINEAR_SPEED;
        
        cv::imshow("Debug Mask", mask);
    }
    
    /**
     * 障碍物规避模式 (简化为计时控制)
     */
    void handleObstacleAvoidance(Mat hsv, geometry_msgs::Twist &cmd) {
        avoid_timer_++;
        const int AVOID_DURATION = 100; // 绕行持续时间 (帧数)

        if (current_state_ == STATE_OBSTACLE_1_AVOID) {
            // 第一个障碍：向左绕行
            cmd.linear.x = 0.2;
            cmd.angular.z = 0.5; // 左转
        } else if (current_state_ == STATE_OBSTACLE_2_AVOID) {
            // 第二个障碍：向右绕行
            cmd.linear.x = 0.2;
            cmd.angular.z = -0.5; // 右转
        }

        if (avoid_timer_ > AVOID_DURATION) {
            // 绕行完成，回到循迹模式
            current_state_ = STATE_LANE_FOLLOW;
            avoid_timer_ = 0;
            ROS_INFO("Avoidance complete, switching to Lane Follow.");
        }
    }

    /**
     * 寻找目标模式：原地旋转寻找红色数字纸张
     */
    void handleTargetSearch(Mat hsv, geometry_msgs::Twist &cmd) {
        Mat mask_target;
        inRange(hsv, target_lower, target_upper, mask_target);
        
        if (countNonZero(mask_target) > 1000) { // 检测到红色色块
            current_state_ = STATE_TRACK_TARGET;
            ROS_INFO("Target found, switching to Tracking Mode.");
            return;
        }

        // 未找到：原地旋转
        cmd.linear.x = 0.0;
        cmd.angular.z = 0.4;
        
        cv::imshow("Debug Mask", mask_target);
    }
    
    /**
     * 目标跟踪模式：跟踪红色数字纸张的重心
     */
    void handleTargetTracking(Mat hsv, geometry_msgs::Twist &cmd) {
        Mat mask;
        inRange(hsv, target_lower, target_upper, mask);
        
        // 形态学操作：闭运算连接数字区域
        Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        double max_area = 0;
        int max_idx = -1;
        
        // 找到最大的红色目标轮廓
        for(int i=0; i<contours.size(); i++){
            double area = contourArea(contours[i]);
            if(area > max_area){
                max_area = area;
                max_idx = i;
            }
        }

        if(max_idx != -1 && max_area > 1000) {
            Moments M = moments(contours[max_idx]);
            double cx = M.m10 / M.m00;
            
            // 跟踪控制：调整角度以使目标居中
            double error = IMG_CENTER_X - cx;
            cmd.angular.z = KP_TRACK * error;
            
            // 距离控制：根据面积调整线速度 (防止碰撞)
            if (max_area > 80000) { // 目标太近
                cmd.linear.x = 0.0;
            } else if (max_area < 10000) { // 目标太远
                cmd.linear.x = 0.25;
            } else { // 保持距离
                cmd.linear.x = 0.15;
            }
        } else {
            // 丢失目标，回到搜索模式
            current_state_ = STATE_SEARCH_TARGET;
        }
        
        cv::imshow("Debug Mask", mask); 
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "course_design_node");
    RobotVisionController controller;
    ros::spin();
    return 0;
}