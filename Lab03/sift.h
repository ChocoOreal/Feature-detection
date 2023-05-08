#pragma once
#include <opencv2/opencv.hpp>
#include "dog.h"
#include "blob.h"
#include "harris.h"
using namespace cv;
using namespace std;

struct siftKeypoints {
	int x, y;
	double sigma;
	double angle;
	Mat feature;
};


class MySIFT {
private:
	Mat img;
	vector<vector<Mat>> gaussSpace;
	vector<vector<Mat>> gradX;
	vector<vector<Mat>> gradY;
	vector<DogKeypoint> keypoints;
	vector<siftKeypoints> siftKPs;
public:
	MySIFT(vector<vector<Mat>> gaussSpace_, vector<DogKeypoint> keypoints_, Mat img_) {
		gaussSpace = gaussSpace_;
		keypoints = keypoints_;
		img = img_;
	}

	void computeGradient();
	void computeOrientation(double minDist, double oriScale, int nbins, double thresholdFor2ndRef = 0.8);
	void constructKeypointDescriptor(double minDist, int nhist, int nori, double descriptorScale);
	vector<siftKeypoints> getSiftKPs() {
		return siftKPs;
	}
};

Mat matchBySIFT(Mat img1, Mat img2, int detector, double relativeThreshold, Mat originalImg1, Mat originalImg2);