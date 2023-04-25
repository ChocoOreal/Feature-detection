#include "sobel.h"
int detectBySobel(Mat src, Mat& dst, int direction) {
	double kerV[9] = { -1.0, 0.0, 1.0,
					-2.0, 0.0, 2.0,
					-1.0, 0.0, 1.0};
	double kerH[9] = { 1.0, 2.0, 1.0,
					  0.0, 0.0, 0.0,
					 -1.0, -2.0, -1.0 };
	Mat img_padded;
	copyMakeBorder(src, img_padded, 1, 1, 1, 1, BORDER_CONSTANT);
	if (img_padded.empty()) return 0;
	
	Mat kernelH, kernelV, edgeH, edgeV, mag;
	
	
	
	/*edgeV.convertTo(edgeV, CV_8UC1, 1 / 256.0);
	edgeH.convertTo(edgeH, CV_8UC1, 1 / 256.0);*/
	if (direction == 0 || direction == -1) {
		kernelH = Mat(3, 3, CV_64F, kerH);
		edgeH = filter(img_padded, kernelH);
		if (edgeH.empty()) return 0;
	}
	if (direction == 1 || direction == -1) {
		kernelV = Mat(3, 3, CV_64F, kerV);
		edgeV = filter(img_padded, kernelV);
		if (edgeV.empty()) return 0;

	}
	if (direction == -1) {
		
		//dst = abs(edgeH) + abs(edgeV);
		magnitude(edgeH, edgeV, dst);
		
	}
	else if (direction == 0) dst = edgeH;
	else if (direction == 1) dst = edgeV;
	else
		cout << "direction must be in [-1, 0, 1]";
	//dst.convertTo(dst, CV_8U, 255.0f);
	return 1;
}