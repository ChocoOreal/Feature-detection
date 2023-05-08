#pragma once
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

struct Layer {
	Mat convoledImg;
	float sigma;
};
struct Keypoint {
	int x, y;
	float sigma;
	int octave;
};

struct DogKeypoint {
	int x, y, octave;
	int s;
	double sigma;
	double angle;
};