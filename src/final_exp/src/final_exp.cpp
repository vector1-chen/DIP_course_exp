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
#include <chrono>

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
    double area_threshold = 85000.0; // 障碍物检测面积阈值

    
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
    double TARGET_AREA = 60000.0;      // 期望的目标面积大小 (用于距离保持)

    // 模板库
    map<int, Mat> templates_;
    bool templates_loaded_;

    // 匹配参数
    const Size TEMPLATE_SIZE = Size(80, 120); // 归一化尺寸 (宽, 高)
    const double MATCH_THRESHOLD = 0.4;     // 匹配阈值 (0-1), 越接近1越严格

    // PID 控制参数
    const double KP_ANGULAR = 3.5;
    const double KI_ANGULAR = 0.01;
    double integral_angular = 0.0;
    const double KP_LINEAR = 0.000020;
    const double KI_LINEAR = 0.000000005;
    const double KD_LINEAR = 0.00000002;
    double previous_error_area = 0.0;
    double integral_linear = 0.0;



public:
    RobotVisionController() : it_(nh_), current_state_(STATE_LANE_FOLLOW), lost_cones_counter_(0), avoid_timer_(0), obstacle_count_(0), avoid_phase_(RIGHT1),  templates_loaded_(false) {
        phase_start_time_ = std::chrono::steady_clock::now();
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

        loadTemplate(0, "/home/haojiechen/file_mess/DIP/dip_ws/src/templates/template0.jpg");
        loadTemplate(1, "/home/haojiechen/file_mess/DIP/dip_ws/src/templates/template1.jpg");
        loadTemplate(2, "/home/haojiechen/file_mess/DIP/dip_ws/src/templates/template2.jpg");

        cv::namedWindow("View");
        cv::namedWindow("Threshold");

        
        ROS_INFO("Robot Vision Controller Initialized in LANE_FOLLOW mode.");
        cv::namedWindow("Debug Mask");
    }

    ~RobotVisionController() {
        cv::destroyWindow("Debug Mask");

        cv::destroyWindow("View");
        cv::destroyWindow("Threshold");
    }


    // 加载模板函数
    void loadTemplate(int id, string path) {  
        Mat img = imread(path, IMREAD_GRAYSCALE);
        if (img.empty()) {
            ROS_ERROR("Failed to load template: %s", path.c_str());
            return;
        }
        // 图像滤波去噪
        GaussianBlur(img, img, Size(5, 5), 0);
        // 二值化处理，保证模板是黑底白字（或者白底黑字，需统一）
        // 这里假设输入图片是白纸黑字，我们统一转为黑底白字进行匹配
        threshold(img, img, 100, 255, THRESH_BINARY_INV);
        // 对1右移
        if (id == 1){
            Mat M = (Mat_<double>(2,3) << 1, 0, 100, 0, 1, 0);
            warpAffine(img, img, M, img.size());
        }
        resize(img, img, TEMPLATE_SIZE);
        templates_[id] = img;
        templates_loaded_ = true;
    }


    void imageCb(const sensor_msgs::ImageConstPtr& msg) {
        if (!templates_loaded_) return;


        cv_bridge::CvImageConstPtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        processImage(cv_ptr->image);

        ROS_INFO_ONCE("Received image %dx%d", cv_ptr->image.cols, cv_ptr->image.rows);

    }

    void updateMask() {
        inRange(hsv, cone_lower, cone_upper, mask);
        
        // 形态学操作：开运算去噪
        Mat kernel = getStructuringElement(MORPH_RECT, Size(80, 80));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        // 形态学操作：闭运算连接锥桶区域
        // kernel = getStructuringElement(MORPH_RECT, Size(60, 60));
        // morphologyEx(mask, mask, MORPH_CLOSE, kernel);
        // 形态学操作：长方形膨胀增强锥桶区域
        kernel = getStructuringElement(MORPH_RECT, Size(7, 250));
        morphologyEx(mask, mask, MORPH_DILATE, kernel);
    }

    void processImage(Mat img) {
        geometry_msgs::Twist cmd;
        
        // 预处理：高斯模糊去噪，转换HSV
        GaussianBlur(img, img, Size(5, 5), 0);
        if (current_state_ == STATE_TRACK_TARGET){
            GaussianBlur(img, img, Size(7, 7), 0);
        }
        cvtColor(img, hsv, COLOR_BGR2HSV);
        frame_width = img.cols;
        frame_height = img.rows;
        IMG_CENTER_X = frame_width / 2;
        area_threshold = (frame_width * frame_height) / 10.6; 
        TARGET_AREA = (frame_width * frame_height) / 11.0;
        ROS_INFO_ONCE("Image size: %dx%d", frame_width, frame_height);
        ROS_INFO_ONCE("Image center X: %d", IMG_CENTER_X);


        // 状态
        switch (current_state_) {
            case STATE_LANE_FOLLOW:
                handleLaneFollow(cmd);
                break;
            
            case STATE_OBSTACLE_1_AVOID:
            case STATE_OBSTACLE_2_AVOID:
                handleObstacleAvoidance(cmd);
                break;
            
            case STATE_TRACK_TARGET:
                processAndTrack(img, cmd);
                break;
        }
        vel_pub_.publish(cmd);
        waitKey(3);
    }

    bool controlOne(geometry_msgs::Twist &cmd, double speed, bool &obstacle_detected){
        double cx_center = INFINITY;
        double error = 0.0;  // 在函数开始处定义error
        double temp_threshold = area_threshold;
        if (obstacle_count_ == 1){
            temp_threshold = area_threshold * 1.35;
        }
        
        for (const auto& contour : contours) {
            Moments M = moments(contour);
            if (M.m00 > 0) {
                double cx = M.m10 / M.m00;
                if (cx < frame_width / 4 || cx > frame_width * 3 / 4) continue; // 忽略边缘轮廓{
                if (std::abs(cx - IMG_CENTER_X) < std::abs(cx_center - IMG_CENTER_X)){
                    cx_center = cx;
                }
            
                double area = contourArea(contour);

                if (area < 3500) continue; // 忽略过小轮廓噪声

                // **障碍物检测启发式：** 面积巨大且靠近图像中心
                if (area > temp_threshold && cx > IMG_CENTER_X - IMG_CENTER_X /5 && cx < IMG_CENTER_X + IMG_CENTER_X /5) { // 阈值需调试
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
        double temp_KP_LANE = KP_LANE;
        if (obstacle_count_ == 2){
            temp_KP_LANE = KP_LANE * 1.2;
        }

        for (const auto& contour : contours) {
            Moments M = moments(contour);
            if (M.m00 > 0) {
                double cx = M.m10 / M.m00;
                double cy = M.m01 / M.m00;
                double area = contourArea(contour);
                if (area < 100) continue; // 忽略过小轮廓

                if (area < 3500) continue; // 忽略过小轮廓噪声

                // **障碍物检测启发式：** 面积巨大且靠近图像中心
                if (area > area_threshold && cx > IMG_CENTER_X - IMG_CENTER_X /5 && cx < IMG_CENTER_X + IMG_CENTER_X /5) { // 阈值需调试
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
        if (obstacle_count_ == 2 && speed == 0){
            cmd.angular.z = (KP_LANE * error + KI_LANE * integral_lane)/3;
            cmd.linear.x = speed;
        }
        else{
            cmd.angular.z = temp_KP_LANE * error + KI_LANE * integral_lane;
            cmd.linear.x = speed;
        }
        return abs(error) < 50;
    }

    /**
     * 循迹模式：基于锥桶重心进行路径控制
     */
    void handleLaneFollow(geometry_msgs::Twist &cmd) {

        updateMask();
        // 寻找轮廓
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (contours.empty()) {
            lost_cones_counter_++;
            if (lost_cones_counter_ > 150 && obstacle_count_ >= 2) { // 连续丢失超过50帧，切换到寻找模式
                current_state_ = STATE_TRACK_TARGET;
                ROS_INFO("Cones lost, switching to Target Search Mode.");
                cmd.angular.z = 0.0;
                cmd.linear.x = 0.0;
            }
            cmd.linear.x = 0.0;
            return;
        } else {
            lost_cones_counter_ = 0;
        }
        bool obstacle_detected = false;
        if (obstacle_count_ >= 2){
            controlAll(cmd, LINEAR_SPEED*1.3, obstacle_detected);
        }
        else 
        {
            if (obstacle_count_ == 1)
                controlOne(cmd, LINEAR_SPEED/4, obstacle_detected);
            else
                controlOne(cmd, LINEAR_SPEED, obstacle_detected);
        }
            // 绿色路线避障触发逻辑
        if (obstacle_detected) {

            obstacle_count_++;
            avoid_timer_ = 0; // 重置计时器
            phase_start_time_ = std::chrono::steady_clock::now(); // 重置时间计时器
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
        cv::imshow("Debug Mask", mask);
        
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
    std::chrono::steady_clock::time_point phase_start_time_;  // 每个阶段的开始时间

    void handleObstacleAvoidance(geometry_msgs::Twist &cmd) {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - phase_start_time_).count();
        
        switch(avoid_phase_) {
            case RIGHT1:
                cmd.linear.x = 0.0;
                cmd.angular.z = -0.25;
                if(elapsed_ms > 2500) {  // 70帧@30fps ≈ 2333ms
                    avoid_phase_ = FORWARD11;
                    phase_start_time_ = std::chrono::steady_clock::now();
                    ROS_INFO("RIGHT1 Phase");
                }
                break;
            case FORWARD11:
                // 阶段1：右转90度
                cmd.linear.x = 0.2;
                cmd.angular.z = 0.0;
                if(elapsed_ms > 3500) {  // 100帧@30fps ≈ 3333ms
                    avoid_phase_ = LEFT1;
                    phase_start_time_ = std::chrono::steady_clock::now();
                    ROS_INFO("FORWARD1 Phase");
                }
                break;
            case LEFT1:
                cmd.linear.x = 0.0;
                cmd.angular.z = 0.25;
                if(elapsed_ms > 4200) {  // 120帧@30fps ≈ 4000ms
                    avoid_phase_ = FORWARD12;
                    phase_start_time_ = std::chrono::steady_clock::now();
                    ROS_INFO("LEFT1 Phase");
                }
                break;
            case FORWARD12:
                // 阶段3：左转90度回到原方向
                cmd.linear.x = 0.2;
                cmd.angular.z = 0.0;
                if(elapsed_ms > 4100) {  // 140帧@30fps ≈ 4667ms
                    avoid_phase_ = RIGHTCORRECT1;
                    phase_start_time_ = std::chrono::steady_clock::now();
                    ROS_INFO("FORWARD2 Phase");
                }
                break;
            case RIGHTCORRECT1:
                cmd.linear.x = 0.0;
                cmd.angular.z = -0.25;
                if(elapsed_ms > 1600) {  // 60帧@30fps ≈ 2000ms
                    current_state_ = STATE_LANE_FOLLOW;
                    avoid_phase_ = LEFT2;
                    phase_start_time_ = std::chrono::steady_clock::now();
                    ROS_INFO("RIGHTCORRECT1 Phase");
                // if (obstacle_count_ ==1)
                // {
                //     bool null = false;
                //     ROS_INFO_ONCE("Completing avoiding Obstacle 1 .");
                //     while(!controlOne(cmd, 0, null));
                //     ROS_INFO_ONCE("Debug");
                // }
                 }
                break;



            case LEFT2:
                // 阶段4：前进一段距离确保完全绕过
                cmd.linear.x = 0.0;
                cmd.angular.z = 0.25;
                if(elapsed_ms > 2500) {  // 70帧@30fps ≈ 2333ms
                    // 绕行完成，回到原状态
                    avoid_phase_ = FORWARD21;
                    phase_start_time_ = std::chrono::steady_clock::now();
                    ROS_INFO("LEFT2 Phase");
                }
                break;
            case FORWARD21:
                // 阶段4：前进一段距离确保完全绕过
                cmd.linear.x = 0.2;
                cmd.angular.z = 0.0;
                if(elapsed_ms > 3333) {  // 100帧@30fps ≈ 3333ms
                    // 绕行完成，回到原状态
                    avoid_phase_ = RIGHT2;
                    ROS_INFO("FORWARD21 Phase");
                    phase_start_time_ = std::chrono::steady_clock::now();
                }
                break;
            case RIGHT2:
                // 阶段4：前进一段距离确保完全绕过
                cmd.linear.x = 0.0;
                cmd.angular.z = -0.25;
                if(elapsed_ms > 3333) {  // 100帧@30fps ≈ 3333ms
                    // 绕行完成，回到原状态
                    avoid_phase_ = RIGHTCORRECT2;
                    ROS_INFO("RIGHT2 Phase");
                    phase_start_time_ = std::chrono::steady_clock::now();
                }
                break;
            case RIGHTCORRECT2:
                // 阶段4：前进一段距离确保完全绕过
                cmd.linear.x = 0.15;
                cmd.angular.z = -0.22;
                if(elapsed_ms > 7000) {  // 210帧@30fps ≈ 7000ms
                    // 绕行完成，回到原状态
                    current_state_ = STATE_LANE_FOLLOW;
                    avoid_phase_ = RIGHT1;
                    ROS_INFO("RIGHTCORRECT2 Phase - Obstacle Avoidance Complete.");
                    phase_start_time_ = std::chrono::steady_clock::now();
                // if (obstacle_count_ ==2)
                // {
                //     bool null = false;
                //     ROS_INFO_ONCE("Completing avoiding Obstacle 2 ");
                //     while(!controlAll(cmd, 0, null));
                //     ROS_INFO_ONCE("Debug");
                // }
                }
                break;
            }
        }


    

    void processAndTrack(Mat frame, geometry_msgs::Twist &cmd) {
        Mat gray, binary;

        // 1. 预处理
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // 自适应阈值处理，应对光照变化
        // blockSize: 11, C: 2. 结果：黑色物体变白，白色背景变黑 (THRESH_BINARY_INV)
        adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 25, 10);

        // 形态学操作：开运算去除噪点，闭运算连接数字
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(binary, binary, MORPH_OPEN, kernel);
        kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(binary, binary, MORPH_CLOSE, kernel);


        // 2. 查找轮廓
        vector<vector<Point>> contours;
        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        int best_digit = -1;
        double max_score = -1.0;
        Rect best_rect;

        for (const auto& cnt : contours) {
            Rect roi_rect = boundingRect(cnt);
            double area = roi_rect.width * roi_rect.height;

            // 3. 筛选候选区域 (根据面积和长宽比)
            // 假设：数字是垂直长方形，面积适中
            double aspect_ratio = (double)roi_rect.width / roi_rect.height;

            
            if (area > (frame.rows * frame.cols) / 100  && area < (frame.rows * frame.cols) && aspect_ratio < 1.5 && aspect_ratio > 0.15) {
                // 先提取基本ROI
                Mat roi = binary(roi_rect).clone();
                
                // 闭运算连接数字笔画
                Mat kernel_roi = getStructuringElement(MORPH_RECT, Size(10, 10));
                morphologyEx(roi, roi, MORPH_CLOSE, kernel_roi);
                
                // 填充轮廓使数字变为实心白色
                vector<vector<Point>> roi_contours;
                findContours(roi.clone(), roi_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                for (const auto& c : roi_contours) {
                    drawContours(roi, vector<vector<Point>>{c}, -1, Scalar(255), FILLED);
                }
                
                // 将处理后的roi放回binary
                roi.copyTo(binary(roi_rect));
                
                // 扩展 ROI 范围，避免裁剪数字边缘
                int padding_w = roi_rect.width * 0.15;
                int padding_h = roi_rect.height * 0.15;
                
                // 计算扩展后的边界并限制在图像范围内
                int x1 = max(roi_rect.x - padding_w, 0);
                int y1 = max(roi_rect.y - padding_h, 0);
                int x2 = min(roi_rect.x + roi_rect.width + padding_w, binary.cols);
                int y2 = min(roi_rect.y + roi_rect.height + padding_h, binary.rows);
                Rect expanded_rect(x1, y1, x2 - x1, y2 - y1);
                
                // 提取扩展后的ROI
                roi = binary(expanded_rect).clone();
                resize(roi, roi, TEMPLATE_SIZE);

                // 4. 模板匹配
                for (auto const& pair : templates_) {
                    int id = pair.first;
                    const Mat& temp_img = pair.second;
                    Mat result;
                    matchTemplate(roi, temp_img, result, TM_CCOEFF_NORMED);
                    
                    double minVal, maxVal;
                    minMaxLoc(result, &minVal, &maxVal);

                    // 如果得分高于阈值且是目前最高的
                    if (maxVal > MATCH_THRESHOLD && maxVal > max_score) {
                        max_score = maxVal;
                        best_digit = id;
                        best_rect = roi_rect;
                    }
                }
            }
        }

        // 5. 跟踪控制逻辑
        if (best_digit != -1) {
            // 画框显示
            rectangle(frame, best_rect, Scalar(0, 255, 0), 2);
            putText(frame, to_string(best_digit), Point(best_rect.x, best_rect.y - 10), 
                    FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
            
            // 计算中心偏差
            double center_x = best_rect.x + best_rect.width / 2.0;
            double error_x = IMG_CENTER_X - center_x; // 假设图像宽720
            integral_angular += error_x;
            if (integral_angular > 1000) integral_angular = 1000;
            if (integral_angular < -1000) integral_angular = -1000;
            
            // 计算面积偏差 (用于前后运动)
            double area = best_rect.width * best_rect.height;
            double error_area = TARGET_AREA - area;
            integral_linear += error_area;
            if (integral_linear > 1000000) integral_linear = 1000000;
            if (integral_linear < -1000000) integral_linear = -1000000;

            // PID 控制
            cmd.angular.z = KP_ANGULAR * error_x + KI_ANGULAR * integral_angular;
            cmd.linear.x = KP_LINEAR * error_area + KI_LINEAR * integral_linear + KD_LINEAR * (error_area - previous_error_area);
            previous_error_area = error_area;
            
            ROS_INFO("Detect: %d | Score: %.2f | Area: %.0f", best_digit, max_score, area);

        }
        imshow("View", frame);
    }
};



int main(int argc, char** argv) {
    ros::init(argc, argv, "course_design_node");
    RobotVisionController controller;
    ros::spin();
    return 0;
}