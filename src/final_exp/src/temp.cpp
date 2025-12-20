/**
 * 基于模板匹配的数字跟踪节点
 * 功能：识别 A4 纸上的黑色数字 0, 1, 2 并控制机器人跟踪
 */

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Twist.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>
#include <map>

using namespace cv;
using namespace std;

class DigitTracker {
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Publisher vel_pub_;

    // 模板库
    map<int, Mat> templates_;
    bool templates_loaded_;

    // 匹配参数
    const Size TEMPLATE_SIZE = Size(80, 120); // 归一化尺寸 (宽, 高)
    const double MATCH_THRESHOLD = 0.4;     // 匹配阈值 (0-1), 越接近1越严格

    // PID 控制参数
    const double KP_ANGULAR = 0.005;
    const double TARGET_AREA = 20000.0;      // 期望的目标面积大小 (用于距离保持)
    const double KP_LINEAR = 0.00001;

public:
    DigitTracker() : it_(nh_), templates_loaded_(false) {
        image_sub_ = it_.subscribe("/camera/color/image_raw", 1, &DigitTracker::imageCb, this);
        vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
        
        // !!! 请修改这里的路径为你电脑上的实际路径 !!!
        loadTemplate(0, "/home/haojiechen/file_mess/DIP/dip_ws/src/templates/template0.jpg");
        loadTemplate(1, "/home/haojiechen/file_mess/DIP/dip_ws/src/templates/template1.jpg");
        loadTemplate(2, "/home/haojiechen/file_mess/DIP/dip_ws/src/templates/template2.jpg");

        cv::namedWindow("View");
        cv::namedWindow("Threshold");
    }

    ~DigitTracker() {
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
        // 二值化处理，保证模板是黑底白字（或者白底黑字，需统一）
        // 这里假设输入图片是白纸黑字，我们统一转为黑底白字进行匹配
        threshold(img, img, 100, 255, THRESH_BINARY_INV);
        resize(img, img, TEMPLATE_SIZE);
        templates_[id] = img;
        templates_loaded_ = true;
        ROS_INFO("Loaded Template %d", id);
        imshow("Template " + to_string(id), img);
    }

    void imageCb(const sensor_msgs::ImageConstPtr& msg) {
        if (!templates_loaded_) return;

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        processAndTrack(cv_ptr->image);
    }

    void processAndTrack(Mat frame) {
        geometry_msgs::Twist cmd;
        Mat gray, binary;

        // 1. 预处理
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // 自适应阈值处理，应对光照变化
        // blockSize: 11, C: 2. 结果：黑色物体变白，白色背景变黑 (THRESH_BINARY_INV)
        // adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 25, 10);
        // imshow("Threshold Raw", binary);

        // 直方图均衡化增强对比度
        equalizeHist(gray, gray);
        // 全局阈值二值化
        threshold(gray, binary, 127, 255, THRESH_BINARY);
        // 黑白反转
        bitwise_not(binary, binary);

        // 形态学操作：开运算去除噪点，闭运算连接数字
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(binary, binary, MORPH_OPEN, kernel);
        kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(binary, binary, MORPH_CLOSE, kernel);

        imshow("Threshold", binary); // 调试显示二值化图

        // 2. 查找轮廓
        vector<vector<Point>> contours;
        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        int best_digit = -1;
        double max_score = -1.0;
        Rect best_rect;

        for (const auto& cnt : contours) {
            //ROS_INFO("Found contour with area: %.2f", contourArea(cnt));
            Rect roi_rect = boundingRect(cnt);
            double area = contourArea(cnt);

            // 3. 筛选候选区域 (根据面积和长宽比)
            // 假设：数字是垂直长方形，面积适中
            double aspect_ratio = (double)roi_rect.width / roi_rect.height;
            
            if (area > 1000 && area < 100000 && aspect_ratio < 0.9 && aspect_ratio > 0.2) {
                ROS_INFO("Candidate ROI - Area: %.2f, Aspect Ratio: %.2f", area, aspect_ratio);
                // 提取 ROI 并预处理（与模板一致）
                Mat roi = binary(roi_rect).clone();
                // 闭运算连接数字笔画
                Mat kernel_roi = getStructuringElement(MORPH_RECT, Size(50, 50));
                morphologyEx(roi, roi, MORPH_CLOSE, kernel_roi);
                imshow("ROI Before Resize", roi);
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
            double error_x = 320 - center_x; // 假设图像宽640
            
            // 计算面积偏差 (用于前后运动)
            double area = best_rect.width * best_rect.height;
            double error_area = TARGET_AREA - area;

            // PID 控制
            cmd.angular.z = KP_ANGULAR * error_x;

            if (abs(error_x) < 50) { // 只有当大致对准时才前后移动
                if (area > TARGET_AREA * 1.2) cmd.linear.x = -0.1; // 太近，后退
                else if (area < TARGET_AREA * 0.8) cmd.linear.x = 0.15; // 太远，前进
                else cmd.linear.x = 0.0; // 距离合适
            }
            
            ROS_INFO("Detect: %d | Score: %.2f | Area: %.0f", best_digit, max_score, area);

        } else {
            // 没找到目标，停止或原地搜索
            cmd.linear.x = 0.0;
            cmd.angular.z = 0.0; 
        }

        vel_pub_.publish(cmd);
        imshow("View", frame);
        waitKey(3);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "digit_tracker_node");
    DigitTracker tracker;
    ros::spin();
    return 0;
}