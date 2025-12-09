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
#define BADTHRESHOLD 50
using namespace std;
using namespace cv;

void BadPointDetect(Mat input, Mat output){
    template<int, 3> kernel = { { -1, -1, -1 },
                                 { -1,  8, -1 },
                                 { -1, -1, -1 } };
    int edge_size = 1;
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            if (j < edge_size || j >= input.cols - edge_size || i < edge_size || i >= input.rows - edge_size){
                output.at<uchar>(i, j) = input.at<uchar>(i, j);
                continue;
            }
            double total = 0.0;
            for (int m = -edge_size; m <= edge_size; m++)
            {
                for (int n = -edge_size; n <= edge_size; n++)
                {
                    total += kernel[m + edge_size][n + edge_size] * (double)input.at<uchar>(i + m, j + n);
                }
            }
            output.at<uchar>(i, j) = total;
        }
    }
    for (int i = 0; i < output.rows; i++)
    {
        for (int j = 0; j < output.cols; j++)
        {
            if(output.at<uchar>(i,j) > BADTHRESHOLD)
                output.at<uchar>(i,j) = 255;
            else
                output.at<uchar>(i,j) = 0;
        }
    }
}