#include "blob.h"
#include "filter.h"
Layer* scaleSpace(Mat img, int scaleSpaceSize, double base_sigma) {
	// Approximate LoG by DoG
	// DoG ~ sigma2 * LoG, so no need to multiply by sigma2 for scale normalization
	Mat loGaussian;
	Layer* scaleSpaceArray = new Layer[scaleSpaceSize];
	double factor = pow(2.0, 1.0 / scaleSpaceSize);
	for (int i = 1; i <= scaleSpaceSize; i++) {
		double sigma = base_sigma * pow(factor, i);
		int kernelSize = floor(sigma) * 6 + 1; // ensure the kernel size wouldnt be too small for a certain sigma
		double* kernel = new double[kernelSize * kernelSize];
		NormLoG(kernel, kernelSize, sigma);
		Mat loGaussian = Mat(kernelSize, kernelSize, CV_64F, kernel);

		
		filter2D(img, scaleSpaceArray[i].convoledImg, -1, loGaussian, Point(-1, -1), 0.0, BORDER_CONSTANT);
		cout << scaleSpaceArray[i].convoledImg.rows << " " << scaleSpaceArray[i].convoledImg.cols << "\n";
		scaleSpaceArray[i].sigma = sigma;
		sigma *= factor;
		delete[] kernel;
	}

	return scaleSpaceArray;
}


vector<Keypoint> findBlobKeyPoints(Mat img, float threshold) {
	int x_ind[3] = { 0, -1, 1 };
	int y_ind[3] = { 0, -1, 1 };
	int scaleSpaceSize = 10;
	Layer* space = scaleSpace(img, scaleSpaceSize, 1.6);
	vector<Keypoint> KPs;
	for (int z = 1; z < scaleSpaceSize - 1; z++) {
		Mat current = space[z].convoledImg;
		Mat prev = space[z - 1].convoledImg;
		Mat next = space[z + 1].convoledImg;
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				bool marked = true;
				if (current.at<double>(i, j) > threshold) {

					for (int x1 = 0; x1 < 3; x1++) {
						for (int y1 = 0; y1 < 3; y1++) {

							if (0 <= x_ind[x1] + i && x_ind[x1] + i < current.rows &&
								0 <= y_ind[y1] + j && y_ind[y1] + j < current.cols &&
								(prev.at<double>(x_ind[x1] + i, y_ind[y1] + j) > current.at<double>(i, j) ||
									next.at<double>(x_ind[x1] + i, y_ind[y1] + j) > current.at<double>(i, j)))
								marked = false;
							if (x1 != 0 && y1 != 0 &&
								0 <= x_ind[x1] + i && x_ind[x1] + i < current.rows &&
								0 <= y_ind[y1] + j && y_ind[y1] + j < current.cols &&
								current.at<double>(x_ind[x1] + i, y_ind[y1] + j) > current.at<double>(i, j))
								marked = false;

						}

					}
					if (marked == true) {
						Keypoint kp;
						kp.x = i;
						kp.y = j;
						kp.sigma = space[z].sigma;
						KPs.push_back(kp);

					}
				}
			}
		}
	}
	//cout << space[0].convoledImg;
	delete[] space;
	return KPs;
}

Mat detectBlob(Mat img, Mat originalImg, float threshold) {
	vector<Keypoint> KPs = findBlobKeyPoints(img, threshold);
	for (int i = 0; i < KPs.size(); i++) {
		circle(originalImg, Point(KPs[i].y, KPs[i].x), KPs[i].sigma * sqrt(2), Scalar(0, 0, 255), 2);
	}
	return originalImg;
}

