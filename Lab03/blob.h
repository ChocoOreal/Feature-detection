#pragma once
#include <opencv2/opencv.hpp>
#include "structure.h"
#include <vector>
using namespace std;
using namespace cv;

Mat detectBlob(Mat img, Mat originalImg, float threshold);
vector<Keypoint> findBlobKeyPoints(Mat img, float threshold);