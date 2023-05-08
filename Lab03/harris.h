#pragma once
#include <opencv2/opencv.hpp>
#include "structure.h"
using namespace cv;
using namespace std;
Mat detectHarrist(Mat grayImg, Mat originalImg, int size, double threshold, Scalar lineColor);
vector<DogKeypoint> getHarrisKeypoint(Mat grayImg, vector<vector<Mat>>& gaussSpace, int size, double threshold);