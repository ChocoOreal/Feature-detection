#pragma once
#include <opencv2/opencv.hpp>
#include "structure.h"
#include <vector>
using namespace std;
using namespace cv;

class Blob {
private:
	Mat img;
	vector<Mat> LoGImg;
	vector<double> sigma;
	int scaleSpaceSize;
	vector<DogKeypoint> KPs;

public:
	Blob(Mat img_, int scaleSpaceSize_, double base_sigma) {
		sigma = vector<double>(scaleSpaceSize_ + 2);
		sigma[0] = base_sigma;
		scaleSpaceSize = scaleSpaceSize_;
		img = img_;
		LoGImg = vector<Mat>(scaleSpaceSize_ + 2);
	}
	void findScaleSpace();
	void findBlobKeyPoints(float threshold);
	void convolveImgWithLoG();
	vector<DogKeypoint> getKeypoints() {
		vector<DogKeypoint> keypointList;
		for (int i = 0; i < KPs.size(); i++) {
			DogKeypoint currentKP = KPs[i];
			currentKP.s -= 1;
			keypointList.push_back(KPs[i]);
		}
		return keypointList;
	}
	vector<vector<Mat>> getGaussSpace();
};

Mat detectBlob(Mat img, Mat originalImg, float threshold, double baseSigma, int scaleSpaceSize);
vector<DogKeypoint> findBlobInterestedPoints(Mat img, vector<vector<Mat>>& gaussSpace, float threshold);