#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>
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
	Mat imgray;
	int rows = img.rows;
	int cols = img.cols;
	double scale = max(rows, cols) / 512;
	resize(img, original, Size(), 1.0 / scale, 1.0 / scale, INTER_AREA);
	cvtColor(original, imgray, COLOR_BGR2GRAY);
	imgray.convertTo(imgray, CV_64F, 1.0 / 255.0); //convert to double
	return imgray;
}

int main(int argc, char* argv[]) {
	srand(time(0));
	Mat res;
	const String keys =
		"{help h usage ? || print this message   }"
		"{@path|| path to image}"
		"{@method|1| which action will be taken, 1 is harris, 2 is blob, 3 is dog, 4 is matchBySIFT}"
		"{window|5| window size for non-max suppression, use when method is harris (method=1)}"
		"{threshold|0.05| contrast threshold, can be used when method=1, 2, 3}"
		"{octave|5| number of octaves}"
		"{layer|3| number of layers in each octave}"
		"{min-sigma|0.8| default min sigma in one octave}"
		"{edge|10| threshold for edgeness}"
		"{detector|3| detector for sift algorithm, 1 is harris, 2 is blob, 3 is dog}"
		"{path2|| path to 2nd image}"
		"{relativeThreshold|0.6| threshold for filtering bad matches in KNN}"
		;

	CommandLineParser parser(argc, argv, keys);
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}
	string path = parser.get<string>("@path");
	int method = parser.get<int>("@method");
	if (method == 1) {
		Mat original1;
		Mat img1 = readImg(path, original1);
		int wsize = parser.get<int>("window");
		double threshold = parser.get<double>("threshold");
		cout << "detecting Harris with window size " << wsize << " and threshold " << threshold << "\n";
		res = detectHarrist(img1, original1, wsize, threshold, Scalar(0, 0, 255));

	}
	else if (method == 2) {
		Mat original1;
		Mat img1 = readImg(path, original1);
		double threshold = parser.get<double>("threshold");
		double minSigma = parser.get<double>("min-sigma");
		int numLayer = parser.get<int>("layer");

		res = detectBlob(img1, original1, threshold, minSigma, numLayer);
	}
	else if (method == 3) {
		Mat original1;
		Mat img1 = readImg(path, original1);
		int numOctave = parser.get<int>("octave");
		int numLayer = parser.get<int>("layer");
		double minSigma = parser.get<double>("min-sigma");
		double threshold = parser.get<double>("threshold");
		int edgeness = parser.get<int>("edge");
		res = detectDog(img1, original1, numOctave, numLayer, minSigma, threshold, edgeness);
	}
	else if (method == 4) {
		double relativeThreshold = parser.get<double>("relativeThreshold");
		int detector = parser.get<int>("detector");
		string path2 = parser.get<string>("path2");
		Mat original1;
		Mat img1 = readImg(path, original1);
		Mat original2;
		Mat img2 = readImg(path2, original2);
		res = matchBySIFT(img1, img2, detector, relativeThreshold, original1, original2);

	}
	

	imshow("res", res);
	waitKey(0);
}