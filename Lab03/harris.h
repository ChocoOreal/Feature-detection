#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
Mat detectHarrist(Mat img);
void showHarrisCorners(Mat grayImg, Mat originalImg, int size, double threshold, Scalar lineColor);