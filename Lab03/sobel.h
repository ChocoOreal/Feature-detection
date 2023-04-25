#pragma once
#include <opencv2/opencv.hpp>
#include "filter.h"
using namespace cv;
using namespace std;

// detect edges by sobel
// src is the grayscale image, note that src's must be double
// dst is the destination for the result, it will be in CV_64F
// direction specifies the direction of gradient: 
//-1 is on both direction, 1 is vertical, 0 is horizontal
int detectBySobel(Mat src, Mat& dst, int direction);
