#pragma once
#include <opencv2/opencv.hpp>
#include "structure.h"
#include "filter.h"
#include <vector>
using namespace std;
using namespace cv;

class MyDOG {
private:
	Mat img;
	double minDist;
	int numOctave;
	int numSpace;
	double minSigma;
	double assumedSigma;
	vector<vector<Mat>> dogSpace;
	vector<vector<Mat>> gaussSpace;
	vector<DogKeypoint> keypoints;
public:
	MyDOG(Mat img_, double minDist_, int numOctave_, int numLayer_, double minSigma_, double assumedSigma_) {
		img = img_.clone();
		minDist = minDist_;
		numOctave = numOctave_;
		numSpace = numLayer_;
		minSigma = minSigma_;
		assumedSigma = assumedSigma_;
		gaussSpace = vector<vector<Mat>>(numOctave);
		dogSpace = vector<vector<Mat>>(numOctave);
	}
	void computeGaussSpace();
	void computeDoGSpace();
	void findExtremaOfDogSpace(double threshold);
	void localizeKeypoints(double threshold);
	void discardKeypointsOnEdge(double threshold);
	Mat gradient(int octave, int layer, int x, int y);
	Mat hessian(int octave, int layer, int x, int y);
	void quadraticInterpolate(Mat& offset, double& value, double dogValue, Mat hessian, Mat gradient);
	vector<DogKeypoint> getKeypoints() {
		vector<DogKeypoint> returnedKPs;
		for (int i = 0; i < keypoints.size(); i++) {
			DogKeypoint kp = keypoints[i];
			kp.s -= 1;
			returnedKPs.push_back(kp);
		}
		return returnedKPs;
	}
	vector<vector<Mat>> getGaussSpace() {
		vector<vector<Mat>> returnedGaussSpace(numOctave);
		for (int i = 0; i < gaussSpace.size(); i++) {
			for (int j = 1; j < gaussSpace[0].size() - 2; j++) {
				returnedGaussSpace[i].push_back(gaussSpace[i][j].clone());
			}
		}
		return returnedGaussSpace;
	}
	
};


Mat detectDog(Mat img, Mat oc);



vector<DogKeypoint> findInterestedPoints(Mat img, vector<vector<Mat>>& gaussSpace);
Mat matchBySIFT(Mat img1, Mat img2, double relativeThreshold, Mat originalImg1, Mat originalImg2);