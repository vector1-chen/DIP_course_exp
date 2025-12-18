#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Twist.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <cmath>
#include <limits>

using namespace cv;
using namespace std;

enum CameraState
{
    COMPUTER = 0,
    ZED,
    REALSENSE
};
CameraState state = REALSENSE;  // 修改为使用电脑摄像头

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
    int area_count;
    Mat mask;
    vector<vector<Point>> contours;
    Mat hsv;

    int frame_width = 640;  // 假设图像宽度
    int frame_height = 480; // 假设图像高度
    int IMG_CENTER_X = 320;    // 假设图像宽度 640/2
    
    // 橙色锥桶 HSV 范围 (默认值，需要校准)
    const Scalar cone_lower = Scalar(153, 45, 47); 
    const Scalar cone_upper = Scalar(180, 255, 255);
    
    // 红色目标数字 HSV 范围 (默认值，需要校准)
    const Scalar target_lower = Scalar(180, 255, 255);
    const Scalar target_upper = Scalar(180, 255, 255);

    // 控制参数
    const double KP_LANE = 0.003;   // 循迹比例控制系数
    const double KI_LANE = 0.000;   // 循迹积分控制系数
    double integral_lane = 0.0; // 循迹积分误差
    const double KP_TRACK = 0.008;  // 跟踪比例控制系数
    const double KI_TRACK = 0.000;  // 跟踪积分控制系数
    double integral_track = 0.0; // 跟踪积分误差
    const double LINEAR_SPEED = 0.4; // 默认线速度 (m/s)


public:
    RobotVisionController() : it_(nh_), current_state_(STATE_LANE_FOLLOW), lost_cones_counter_(0), avoid_timer_(0), obstacle_count_(0), avoid_phase_(RIGHT1), phase_timer_(0) {
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

    void updateMask() {
        inRange(hsv, cone_lower, cone_upper, mask);
        
        // 形态学操作：开运算去噪
        Mat kernel = getStructuringElement(MORPH_RECT, Size(95, 95));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        // 形态学操作：闭运算连接锥桶区域
        // kernel = getStructuringElement(MORPH_RECT, Size(60, 60));
        // morphologyEx(mask, mask, MORPH_CLOSE, kernel);
        // 形态学操作：长方形膨胀增强锥桶区域
        kernel = getStructuringElement(MORPH_RECT, Size(7, 200));
        morphologyEx(mask, mask, MORPH_DILATE, kernel);
    }

    void processImage(Mat img) {
        geometry_msgs::Twist cmd;
        
        // 预处理：高斯模糊去噪，转换HSV
        GaussianBlur(img, img, Size(5, 5), 0);
        cvtColor(img, hsv, COLOR_BGR2HSV);
        updateMask();
        frame_width = img.cols;
        frame_height = img.rows;
        IMG_CENTER_X = frame_width / 2;
        ROS_INFO_ONCE("Image size: %dx%d", frame_width, frame_height);
        ROS_INFO_ONCE("Image center X: %d", IMG_CENTER_X);
        cv::imshow("Debug Mask", mask);


        // 状态
        switch (current_state_) {
            case STATE_LANE_FOLLOW:
                handleLaneFollow(cmd);
                break;
            
            case STATE_OBSTACLE_1_AVOID:
            case STATE_OBSTACLE_2_AVOID:
                handleObstacleAvoidance(cmd);
                break;
            
            case STATE_SEARCH_TARGET:
                handleTargetSearch(cmd);
                break;

            case STATE_TRACK_TARGET:
                handleTargetTracking(cmd);
                break;
        }

        vel_pub_.publish(cmd);
        waitKey(3);
    }

    bool controlOne(geometry_msgs::Twist &cmd, double speed, bool &obstacle_detected){
        double cx_center = INFINITY;
        double error = 0.0;  // 在函数开始处定义error
        
        for (const auto& contour : contours) {
            Moments M = moments(contour);
            if (M.m00 > 0) {
                double cx = M.m10 / M.m00;
                if (cx < frame_width / 5 || cx > frame_width * 4 / 5) continue; // 忽略边缘轮廓{
                if (std::abs(cx - IMG_CENTER_X) < std::abs(cx_center - IMG_CENTER_X)){
                    cx_center = cx;
                }
            
                double area = contourArea(contour);

                if (area < 3500) continue; // 忽略过小轮廓噪声

                // **障碍物检测启发式：** 面积巨大且靠近图像中心
                if (area > 85000 && cx > IMG_CENTER_X - IMG_CENTER_X /5 && cx < IMG_CENTER_X + IMG_CENTER_X /5) { // 阈值需调试
                    area_count++;
                    if (area_count > 5)
                    {
                        obstacle_detected = true;
                        area_count = 0;
                    }
                }
                else if (cx > IMG_CENTER_X - IMG_CENTER_X /5 && cx < IMG_CENTER_X + IMG_CENTER_X /5)
                {
                    area_count = 0;
                }
            }
        }
        if (std::isinf(cx_center)){
            cmd.linear.x = speed;
            cmd.angular.z = 0.0;
        }
        else{
            error = IMG_CENTER_X - cx_center; // 错误：中心偏差

            // PI 控制器
            integral_lane += error;
            if (integral_lane > 1000) integral_lane = 1000;
            if (integral_lane < -1000) integral_lane = -1000;
            cmd.angular.z = KP_LANE * error + KI_LANE * integral_lane;
            cmd.linear.x = speed;
        }
        return std::abs(error) < 50;
    }
    bool controlAll(geometry_msgs::Twist &cmd, double speed, bool &obstacle_detected){
        double sum_x_ = 0.0;
        int contour_count = 0;

        for (const auto& contour : contours) {
            Moments M = moments(contour);
            if (M.m00 > 0) {
                double cx = M.m10 / M.m00;
                double area = contourArea(contour);

                if (area < 3500) continue; // 忽略过小轮廓噪声

                // **障碍物检测启发式：** 面积巨大且靠近图像中心
                if (area > 85000 && cx > IMG_CENTER_X - IMG_CENTER_X /5 && cx < IMG_CENTER_X + IMG_CENTER_X /5) { // 阈值需调试
                    area_count++;
                    if (area_count > 5)
                    {
                        area_count = 0;
                        obstacle_detected = true;
                    }
                }
                else if (cx > IMG_CENTER_X - IMG_CENTER_X /5 && cx < IMG_CENTER_X + IMG_CENTER_X /5)
                {
                    area_count = 0;
                }
                
                sum_x_ += cx;
                contour_count++;
            }
        }
        
        // 普通循迹控制 (基于所有锥桶的平均重心)
        double avg_x = contour_count > 0 ? 
                        sum_x_ / contour_count : 
                        IMG_CENTER_X;
        double error = IMG_CENTER_X - avg_x; // 错误：中心偏差

        // PI 控制器
        integral_lane += error;
        if (integral_lane > 1000) integral_lane = 1000;
        if (integral_lane < -1000) integral_lane = -1000;
        cmd.angular.z = KP_LANE * error + KI_LANE * integral_lane;
        cmd.linear.x = speed;
        return abs(error) < 50;
    }

    /**
     * 循迹模式：基于锥桶重心进行路径控制
     */
    void handleLaneFollow(geometry_msgs::Twist &cmd) {

        // 寻找轮廓
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // if (contours.empty()) {
        //     lost_cones_counter_++;
        //     if (lost_cones_counter_ > 50) { // 连续丢失超过50帧，切换到寻找模式
        //         current_state_ = STATE_SEARCH_TARGET;
        //         ROS_INFO("Cones lost, switching to Target Search Mode.");
        //     }
        //     cmd.linear.x = 0.0;
        //     return;
        // } else {
        //     lost_cones_counter_ = 0;
        // }
        bool obstacle_detected = false;
        if (obstacle_count_ >= 2){
            controlAll(cmd, LINEAR_SPEED, obstacle_detected);
        }
        else 
        {
            controlOne(cmd, LINEAR_SPEED, obstacle_detected);
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
        
    }
    
    /**
     * 障碍物规避模式 (简化为计时控制)
     */
    enum AvoidPhase {
        RIGHT1,
        FORWARD11,
        LEFT1,
        FORWARD12,
        RIGHTCORRECT1,
        LEFT2,
        FORWARD21,
        RIGHT2,
        RIGHTCORRECT2
    };

    AvoidPhase avoid_phase_;
    int phase_timer_;  // 每个阶段的计时器

    void handleObstacleAvoidance(geometry_msgs::Twist &cmd) {
        phase_timer_++;
        
        switch(avoid_phase_) {
            case RIGHT1:
                cmd.linear.x = 0.0;
                cmd.angular.z = -0.25;
                if(phase_timer_ > 70) {
                    avoid_phase_ = FORWARD11;
                    phase_timer_ = 0;
                    ROS_INFO("RIGHT1 Phase");
                }
                break;
            case FORWARD11:
                // 阶段1：右转90度（约30帧 @ 30fps = 1秒）
                cmd.linear.x = 0.2;
                cmd.angular.z = 0.0;  // 右转
                if(phase_timer_ > 100) {
                    avoid_phase_ = LEFT1;
                    phase_timer_ = 0;
                    ROS_INFO("FORWARD1 Phase");
                }
                break;
            case LEFT1:
                cmd.linear.x = 0.0;
                cmd.angular.z = 0.25;
                if(phase_timer_ > 120) {
                    avoid_phase_ = FORWARD12;
                    phase_timer_ = 0;
                    ROS_INFO("LEFT1 Phase");
                }
                break;
            case FORWARD12:
                // 阶段3：左转90度回到原方向（约30帧 = 1秒）
                cmd.linear.x = 0.2;
                cmd.angular.z = 0.0;  // 左转
                if(phase_timer_ > 140) {
                    avoid_phase_ = RIGHTCORRECT1;
                    phase_timer_ = 0;
                    ROS_INFO("FORWARD2 Phase");
                }
                break;
            case RIGHTCORRECT1:
                cmd.linear.x = 0.0;
                cmd.angular.z = -0.25;
                if(phase_timer_ > 60) {
                    current_state_ = STATE_LANE_FOLLOW;
                    avoid_phase_ = LEFT2;
                    phase_timer_ = 0;
                    ROS_INFO("RIGHTCORRECT1 Phase");
                }
                break;



            case LEFT2:
                // 阶段4：前进一段距离确保完全绕过（约40帧）
                cmd.linear.x = 0.0;
                cmd.angular.z = 0.25;
                if(phase_timer_ > 70) {
                    // 绕行完成，回到原状态
                    avoid_phase_ = FORWARD21;
                    phase_timer_ = 0;
                    ROS_INFO("LEFT2 Phase");
                }
                break;
            case FORWARD21:
                // 阶段4：前进一段距离确保完全绕过（约40帧）
                cmd.linear.x = 0.2;
                cmd.angular.z = 0.0;
                if(phase_timer_ > 100) {
                    // 绕行完成，回到原状态
                    avoid_phase_ = RIGHT2;
                    ROS_INFO("FORWARD21 Phase");
                    phase_timer_ = 0;
                }
                break;
            case RIGHT2:
                // 阶段4：前进一段距离确保完全绕过（约40帧）
                cmd.linear.x = 0.0;
                cmd.angular.z = -0.25;
                if(phase_timer_ > 100) {
                    // 绕行完成，回到原状态
                    avoid_phase_ = RIGHTCORRECT2;
                    ROS_INFO("RIGHT2 Phase");
                    phase_timer_ = 0;
                }
                break;
            case RIGHTCORRECT2:
                // 阶段4：前进一段距离确保完全绕过（约40帧）
                cmd.linear.x = 0.15;
                cmd.angular.z = -0.18;
                if(phase_timer_ > 210) {
                    // 绕行完成，回到原状态
                    current_state_ = STATE_LANE_FOLLOW;
                    avoid_phase_ = RIGHT1;
                    ROS_INFO("RIGHTCORRECT2 Phase - Obstacle Avoidance Complete.");
                    phase_timer_ = 0;
                }
                break;
        }
        cmd.linear.x = 0.0; // 默认停止前进
        cmd.angular.z = 0.0; // 默认不转向
        if (phase_timer_ > 10) { // 超时保护，避免卡死
            phase_timer_ = 0;
        }
        if (obstacle_count_ ==1)
        {
            bool null = false;
            ROS_INFO_ONCE("Completing avoiding Obstacle 1 .");
            while(!controlOne(cmd, 0, null));
        }
        else if (obstacle_count_ ==2)
        {
            bool null = false;
            ROS_INFO_ONCE("Completing avoiding Obstacle 2 ");
            while(!controlAll(cmd, 0, null));
        }
    }

    /**
     * 寻找目标模式：原地旋转寻找红色数字纸张
     */
    void handleTargetSearch(geometry_msgs::Twist &cmd) {
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
    void handleTargetTracking(geometry_msgs::Twist &cmd) {
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