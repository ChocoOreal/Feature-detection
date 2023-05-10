#include "sift.h"
void MySIFT::computeGradient() {
	int numOctave = gaussSpace.size();
	int numLayer = gaussSpace[0].size();
	gradX = vector<vector<Mat>>(numOctave);
	gradY = vector<vector<Mat>>(numOctave);
	for (int i = 0; i < numOctave; i++) {
		for (int j = 0; j < numLayer; j++) {
			gradX[i].push_back(Mat::zeros(gaussSpace[i][j].size(), CV_64F));
			gradY[i].push_back(Mat::zeros(gaussSpace[i][j].size(), CV_64F));
		}
	}

	for (int o = 0; o < numOctave; o++) {
		for (int s = 0; s < numLayer; s++) {
			for (int i = 1; i < gaussSpace[o][s].rows - 1; i++) {
				for (int j = 1; j < gaussSpace[o][s].cols - 1; j++) {
					double currentX = (gaussSpace[o][s].at<double>(i + 1, j) - gaussSpace[o][s].at<double>(i - 1, j)) * 0.5;
					double currentY = (gaussSpace[o][s].at<double>(i, j + 1) - gaussSpace[o][s].at<double>(i, j - 1)) * 0.5;
					gradX[o][s].at<double>(i, j) = currentX;
					gradY[o][s].at<double>(i, j) = currentY;

				}
			}
		}
	}
}

void MySIFT::computeOrientation(double minDist, double oriScale, int nbins, double thresholdFor2ndRef) {
	double c = 2 * 3.14;
	vector<double> histogram(nbins, 0);
	vector<vector<Mat>> gradient;
	int numOctave = gaussSpace.size();
	int numLayer = gaussSpace[0].size();
	vector<DogKeypoint> newKPs;
	int countLoop = 0;

	for (auto it = keypoints.begin(); it != keypoints.end(); it++) {
		//cout << "countloop " << countLoop << "\n";
		double x = it->x;
		double y = it->y;
		double sigma = it->sigma;
		int layer = it->s;
		
		double dist = minDist * pow(2.0, it->octave);
		//countLoop++;
		// size of gaussian windows
		double currentScale = oriScale * sigma;

		// check if the keypoint is far enough from the image's borders
		if (3 * currentScale <= x && x <= img.rows - 3 * currentScale &&
			3 * currentScale <= y && y <= img.cols - 3 * currentScale) {
			for (int i = floor((x - 3 * currentScale) / dist); i < floor((x + 3 * currentScale) / dist); i++) {
				for (int j = floor((y - 3 * currentScale) / dist); j < floor((y + 3 * currentScale) / dist); j++) {
					// compute gradient wrt x
					

					double dx = gradX[it->octave][layer].at<double>(i, j);


					// compute gradient wrt y
					
					double dy = gradY[it->octave][layer].at<double>(i, j);


					// compute the magnitude of gradient
					double normGrad = sqrt(dx * dx + dy * dy);

					// Find the distance from current point to keypoint
					double distanceToKeySquared = (i * dist - x) * (i * dist - x) + (j * dist - y) * (j * dist - y);

					// the contribution of current point
					double weight = exp(-distanceToKeySquared / (2 * currentScale * currentScale)) * normGrad;
					double angle = atan2(dy, dx);
					if (angle < 0) angle += c;

					// find the index of current point in the histogram

					int binIndex = floor(nbins / (c) * (angle));
					//cout << "binInd " << binIndex << "\n";
					histogram[binIndex] += weight;
				}
			}
		}

		// smooth histogram found
		vector<double> smoothedHist(nbins);
		smoothedHist[0] = (histogram[nbins - 1] + histogram[0] + histogram[1 % nbins]) / 3;
		for (int i = 1; i < nbins; i++) {
			smoothedHist[i] = (histogram[(i - 1) % nbins] + histogram[i] + histogram[(i + 1) % nbins]) / 3;
		}

		// find all local maxima 
		vector<int> peaks;
		double max = 0;
		int peakIndex = 0;
		for (int i = 0; i < nbins; i++) {
			if (smoothedHist[i] > max) {
				max = smoothedHist[i];
				peakIndex = i;
			}
		}
		int countAngles = 0;
		for (int i = 0; i < nbins; i++) {
			if (smoothedHist[i] > thresholdFor2ndRef * max) {
				int k;
				if (i == 0) k = nbins - 1;
				else k = i - 1;
				if (smoothedHist[i] > smoothedHist[k] && smoothedHist[i] > smoothedHist[(i + 1) % nbins])
				{
					double left = smoothedHist[k];
					double right = smoothedHist[(i + 1) % nbins];

					// reference https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
					double interpolatedPeak = (peakIndex + 0.5 * (left - right) / (left - 2 * max + right)) * c / nbins;
					if (countAngles == 0)
						it->angle = interpolatedPeak;
					else {
						DogKeypoint newKP;
						newKP.octave = it->octave;
						newKP.s = it->s;
						newKP.sigma = it->sigma;
						newKP.x = it->x;
						newKP.y = it->y;
						newKP.angle = interpolatedPeak;
						newKPs.push_back(newKP);
					}
					countAngles++;
				}
			}
		}
	}
	keypoints.insert(keypoints.end(), newKPs.begin(), newKPs.end());
}

void MySIFT::constructKeypointDescriptor(double minDist, int nhist, int nori, double descriptorScale)
{
	double c = 2 * 3.14;
	double* h = new double[nhist * nhist * nori] {0};
	uchar* f8bit = new uchar[nhist * nhist * nori]{ 0 };

	for (int i = 0; i < keypoints.size(); i++) {
		double sigma = keypoints[i].sigma;
		double x = keypoints[i].x;
		double y = keypoints[i].y;
		int octave = keypoints[i].octave;
		double windowSize = descriptorScale * sigma;
		int layer = keypoints[i].s;
		if (sqrt(2.0) * windowSize <= x && x <= img.rows - sqrt(2.0) * windowSize &&
			sqrt(2.0) * windowSize <= y && y <= img.cols - sqrt(2.0) * windowSize
			) {
			double dist = minDist * pow(2.0, octave);
			double temp = (nhist + 1) / nhist;
			double startm = (x - sqrt(2.0) * windowSize) / dist;
			double endm = (x + sqrt(2.0) * windowSize) / dist;
			double startn = (y - sqrt(2.0) * windowSize) / dist;
			double endn = (y + sqrt(2.0) * windowSize) / dist;


			for (int l = 0; l < nhist * nhist * nori; l++) h[l] = 0;

			for (int m = startm; m < endm; m++) {
				for (int n = startn; n < endn; n++) {
					double rotatedX = ((m * dist - x) * cos(keypoints[i].angle) + (n * dist - y) * sin(keypoints[i].angle)) / sigma;
					double rotatedY = (-(m * dist - x) * sin(keypoints[i].angle) + (n * dist - y) * cos(keypoints[i].angle)) / sigma;
					double myMax = abs(rotatedX) > abs(rotatedY) ? abs(rotatedX) : abs(rotatedY);


					if (myMax < descriptorScale * temp) {

						// compute the normalized angle of current pixel
						double dx = gradX[octave][layer].at<double>(m, n);

						double dy = gradY[octave][layer].at<double>(m, n);
						double angle = atan2(dy, dx);
						if (angle < 0) angle += c;
						double normedAngle = angle - keypoints[i].angle;
						while (normedAngle < 0) {
							normedAngle += c;
						}
						while (normedAngle > c) {
							normedAngle -= c;
						}


						// compute contribution of current pixel
						double diffX = m * dist - x;
						double diffY = n * dist - y;

						double weight = exp(-(diffX * diffX + diffY * diffY) / (2 * windowSize * windowSize)) * sqrt(dx * dx + dy * dy);
						double distX, distY;


						for (int i = 0; i < nhist; i++) {
							distX = abs((i + 1 - (1 + nhist) / 2) * 2 * descriptorScale / nhist - rotatedX);
							if (distX <= 2 * descriptorScale / nhist) {
								for (int j = 0; j < nhist; j++) {
									distY = abs((j + 1 - (1 + nhist) / 2) * 2 * descriptorScale / nhist - rotatedY);
									if (distY <= 2 * descriptorScale / nhist) {
										for (int k = 0; k < nori; k++) {

											double distAngle = c * k / nori - normedAngle;
											while (distAngle < 0) {
												distAngle += c;
											}
											while (distAngle > c) {
												distAngle -= c;
											}
											if (distAngle < c / nori) {

												double tempCoeff = nhist / (2 * descriptorScale);
												h[i * nhist * nori + j * nori + k] += (1 - tempCoeff * distX)
													* (1 - tempCoeff * distY) * (1 - nori / c * distAngle) * weight;
											}
										}
									}
								}
							}

						}


					}
				}
			}

			double normF = 0;

			for (int l = 0; l < nhist * nhist * nori; l++) {
				normF += h[l] * h[l];
			}

			normF = sqrt(normF);
			siftKeypoints newSiftKP;
			newSiftKP.angle = keypoints[i].angle;
			newSiftKP.x = x;
			newSiftKP.y = y;
			newSiftKP.sigma = sigma;
			newSiftKP.feature = Mat(nhist * nhist * nori, 1, CV_8U);
			for (int l = 0; l < nhist * nhist * nori; l++) {
				h[l] = min(h[l], 0.2 * normF);
				newSiftKP.feature.at<uchar>(l, 0) = round(512 * h[l] / normF) > 255 ? 255 : (uchar)round(512 * h[l] / normF);
			}

			siftKPs.push_back(newSiftKP);
		}

	}
	delete[] h;
}

Mat matchBySIFT(Mat img1, Mat img2, int detector,double relativeThreshold, Mat originalImg1, Mat originalImg2) {
	vector<vector<Mat>> gaussSpace1;
	vector<vector<Mat>> gaussSpace2;
	vector<DogKeypoint> dogkeypoints1;
	vector<DogKeypoint> dogkeypoints2;
	double minDist = 1.0;
	if (detector == 1) {
		dogkeypoints1 = findInterestedPoints(img1, gaussSpace1);
		cout << "dogkp1 " << dogkeypoints1.size() << "\n";
		dogkeypoints2 = findInterestedPoints(img2, gaussSpace2);
		minDist = 0.5;
	}
	else if (detector == 2) {
		dogkeypoints1 = findBlobInterestedPoints(img1, gaussSpace1, 0.1);
		dogkeypoints2 = findBlobInterestedPoints(img2, gaussSpace2, 0.1);
		minDist = 1;
	}
	else if (detector == 3) {
		dogkeypoints1 = getHarrisKeypoint(img1, gaussSpace1, 5, 0.08);
		dogkeypoints2 = getHarrisKeypoint(img2, gaussSpace2, 5, 0.08);
		minDist = 1;
	}
	MySIFT* siftmodel1 = new MySIFT(gaussSpace1, dogkeypoints1, img1);
	MySIFT* siftmodel2 = new MySIFT(gaussSpace2, dogkeypoints2, img2);
	siftmodel1->computeGradient();
	siftmodel2->computeGradient();
	siftmodel1->computeOrientation(minDist, 1.5, 36);
	siftmodel2->computeOrientation(minDist, 1.5, 36);

	siftmodel1->constructKeypointDescriptor(minDist, 4, 8, 6);
	siftmodel2->constructKeypointDescriptor(minDist, 4, 8, 6);
	vector<siftKeypoints> siftkps1 = siftmodel1->getSiftKPs();
	vector<siftKeypoints> siftkps2 = siftmodel2->getSiftKPs();
	vector<int> keypointmatch;
	cout << "knn \n";
	cout << siftkps1.size() << " " << siftkps2.size() << "\n";

	for (int i = 0; i < siftkps1.size(); i++) {

		int minInd = -1;
		int secondMinInd = -1;
		double minDist = 1e5;
		double secondMinDist = 1e5;

		for (int j = 0; j < siftkps2.size(); j++) {
			double dist = norm(siftkps1[i].feature, siftkps2[j].feature, NORM_L2, noArray());
			
			if (dist < minDist) {
				secondMinDist = minDist;
				minDist = dist;
				secondMinInd = minInd;
				minInd = j;

			}
			else if (dist < secondMinDist) {
				secondMinDist = dist;
				secondMinInd = j;
			}


		}
		if (minDist < relativeThreshold * secondMinDist) {
			keypointmatch.push_back(i);
			keypointmatch.push_back(minInd);
		}

	}

	int newrows = max(img1.rows, img2.rows);
	int newcols = img1.cols + img2.cols;

	Mat res(newrows, newcols, originalImg1.type());
	originalImg1.copyTo(res(Rect(0, 0, img1.cols, img1.rows)));
	originalImg2.copyTo(res(Rect(img1.cols, 0, img2.cols, img2.rows)));
	int N = keypointmatch.size() - 1;
	for (int i = 0; i < N; i += 2) {
		int R = rand() % (255);
		int G = rand() % (255);
		int B = rand() % (255);
		int ind1 = keypointmatch[i];
		int ind2 = keypointmatch[i + 1];
		line(res, Point(siftkps1[ind1].y, siftkps1[ind1].x), Point(siftkps2[ind2].y + img1.cols, siftkps2[ind2].x), Scalar(B, G, R), 1);
	}

	return res;
}