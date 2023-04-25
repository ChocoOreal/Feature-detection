#include "sobel.h"
#include "harris.h"
#include "filter.h"


Mat detectHarrist(Mat img) {
	double gaussKern[9];
	gaussianKernel(gaussKern, 3, 1.4);
	Mat harrisWindow(3, 3, CV_64F, gaussKern);
	Mat Ix, Iy;
	float k = 0.04f;
	// find gradient of both direction
	detectBySobel(img, Ix, 0);
	detectBySobel(img, Iy, 1);

	// find matrix M, we leverage gaussian kernel for this task
	Mat Ix2, Iy2, Ixy;
	Mat Ix2_ = Ix.mul(Ix);
	Mat Iy2_ = Iy.mul(Iy);
	Mat Ixy_ = Ix.mul(Iy);
	copyMakeBorder(Ix2_, Ix2, 1, 1, 1, 1, BORDER_CONSTANT);
	copyMakeBorder(Iy2_, Iy2, 1, 1, 1, 1, BORDER_CONSTANT);
	copyMakeBorder(Ixy_, Ixy, 1, 1, 1, 1, BORDER_CONSTANT);

	Ix2 = filter(Ix2, harrisWindow);
	Iy2 = filter(Iy2, harrisWindow);
	Ixy = filter(Ixy, harrisWindow);

	Mat det = Ix2.mul(Iy2) - Ixy.mul(Ixy);
	Mat trace = Ix2 + Iy2;
	Mat response = det - k * trace.mul(trace);
	return response;
}

void showHarrisCorners(Mat grayImg, Mat originalImg, int size, double threshold, Scalar lineColor) {
	Mat R = detectHarrist(grayImg);
	for (int i = 0; i < R.rows; i += size) {
		for (int j = 0; j < R.cols; j += size) {
			double max = 0;
			int r_ind = i, c_ind = j;
			bool marked = false;

			// find local maxima
			for (int nrow = 0; nrow < size; nrow++) {
				for (int ncol = 0; ncol < size; ncol++) {
					if (i + nrow < R.rows && j + ncol < R.cols && R.at<double>(i + nrow, j + ncol) > threshold) {
						marked = true;
						if (R.at<double>(i + nrow, j + ncol) > max) {
							max = R.at<double>(i + nrow, j + ncol);
							r_ind = i + nrow;
							c_ind = j + ncol;
						}
					}
				}
			}
			if (marked)
				circle(originalImg, Point(c_ind, r_ind), 1, lineColor);

		}
	}
	imshow("after", originalImg);
}