#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include "sobel.h"
#include "harris.h"
#include "blob.h"
#include "dog.h"
#include "sift.h"
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
	srand(time(0));
	Mat original1;
	Mat img1 = readImg("D:\\CV\\02_.jpg", original1);
	Mat original2;
	Mat img2 = readImg("D:\\CV\\TestImages\\02.jpg", original2);


	// harris
	/*Mat R = detectHarrist(img);
	Mat imgColor;
	img.convertTo(imgColor, CV_8UC1, 255.0);
	cvtColor(imgColor, imgColor, COLOR_GRAY2BGR, 3);
	showHarrisCorners(img, original, 5, 0.0003, Scalar(0, 0, 255));*/
	//Mat final = detectBlob(img1, original1,0.1);
	/*Mat final = detectDog(img, original);*/

	Mat res = matchBySIFT(img1, img2, 3, 0.8, original1, original2);

	//vector<vector<Mat>> gaussSpace;
	//vector<DogKeypoint> dogkeypoints = findInterestedPoints(img2, gaussSpace);
	//MySIFT siftmodel(gaussSpace, dogkeypoints);
	//siftmodel.computeGradient();
	//siftmodel.computeOrientation(0.5, 1.5, 36);
	//siftmodel.constructKeypointDescriptor(0.5, 4, 8, 6);
	//vector<siftKeypoints> siftkps = siftmodel.getSiftKPs();
	//for (int i = 0; i < 10; i++)
	//	cout << siftkps[i].feature;
	
	imshow("res", res);
	waitKey(0);
}