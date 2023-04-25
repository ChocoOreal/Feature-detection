#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;


Mat filter(Mat_<double> src, Mat_<double> kernel);
void gaussianKernel(double* kernel, int size, float sigma);