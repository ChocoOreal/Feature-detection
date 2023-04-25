#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include "sobel.h"
#include "harris.h"
using namespace cv;
using namespace std;


Mat readImg(string fname, Mat& original) {
	Mat img = imread(fname);
	original = img;
	Mat imgray;
	cvtColor(img, imgray, COLOR_BGR2GRAY);
	imgray.convertTo(imgray, CV_64F, 1.0 / 255.0); //convert to double
	return imgray;
}

int main() {
	Mat original;
	Mat img = readImg("D:\\CV\\TestImages\\02.jpg", original);

	// harris
	Mat R = detectHarrist(img);
	Mat imgColor;
	img.convertTo(imgColor, CV_8UC1, 255.0);
	cvtColor(imgColor, imgColor, COLOR_GRAY2BGR, 3);
	showHarrisCorners(img, original, 5, 0.0003, Scalar(0, 0, 255));
	waitKey(0);
}