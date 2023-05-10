#include "dog.h"

// generate gaussian kernel for each octaves
void MyDOG::generateKernel() {

	// If an image already has a blur scale of sigma1, after scaling it with sigma2, the final image
	// will be having blur scale of sigma that satisfies sigma^2 = sigma1^2 + sigma2^2
	// We will scale sigma based on the size of image. Say, img2 has size twice the size of img1,
	// its blur scale sigma2 should be twice sigma1 of img1.
	// 
	// In this case, we want our target img to have blur of minSigma and since the original is assumed
	// to have blur of assumedSigma, we have blur the original img with 
	// sigma = sqrt(minSigma^2 - assumedSigma^2)
	// Furthermore, the target img will have size 1/dist of the original one, we have to take 
	// 1/dist * sigma for blur scale.
	double currentSigma = 1 / minDist * sqrt(minSigma * minSigma - assumedSigma * assumedSigma);
	int kernelSize = floor(currentSigma) * 6 + 1; // We want to assure that the size of kernel wont be too small
	double* kernelArray = new double[kernelSize * kernelSize];
	gaussianKernel(kernelArray, kernelSize, currentSigma);
	Mat kernelMat(kernelSize, kernelSize, CV_64F, kernelArray);
	kernel.push_back(kernelMat);

	// The factor for multiplying sigma after each iteration is 2^(1 / numSpace)
	// Our numSpace should have +3 since after taking DoG, the numSpace will be decreased by 1
	// 2 other will be for the first layer (the base img), and the layer after the last layer that
	// can contribute to DoG
	for (int s = 1; s <= numSpace + 2; s++) {
		double sig1 = pow(2.0, 2.0 * s / numSpace);
		double sig2 = pow(2.0, 2.0 * (s - 1) / numSpace);
		currentSigma = minSigma / minDist * sqrt(sig1 - sig2);
		kernelSize = floor(currentSigma) * 6 + 1; // We want to assure that the size of kernel wont be too small
		kernelArray = new double[kernelSize * kernelSize];
		gaussianKernel(kernelArray, kernelSize, currentSigma);

		kernelMat = Mat(kernelSize, kernelSize, CV_64F, kernelArray);
		kernel.push_back(kernelMat);

	}
}


// find gaussian space to approximate DoG.
// Our pyramid will have numOctave octaves, each octaves include images of different size
// numSpace is our desirable number of DoG layers in an octave
// minSigma is the smallest sigma to start with in one octave
// assumedSigma is the sigma already corporated with the input image
void MyDOG::computeGaussSpace() {
	Mat baseImg;
	Mat imgInOctave;

	resize(img, baseImg, Size(), 1/ minDist, 1 / minDist, INTER_LINEAR);
		
	// Now we blur the baseImg with gaussian kernel found
	filter2D(baseImg, imgInOctave, -1, kernel[0], Point(-1, -1), 0.0, BORDER_CONSTANT);
	gaussSpace[0].push_back(imgInOctave.clone());

	
	for (int s = 1; s <= numSpace + 2; s++) {
		filter2D(gaussSpace[0][s - 1], imgInOctave, -1, kernel[s], Point(-1, -1), 0.0, BORDER_CONSTANT);
		gaussSpace[0].push_back(imgInOctave.clone());
	}
	double dist = minDist;
	// Compute subsequent octaves
	for (int o = 1; o < numOctave; o++) {

		// Half the size of the img in the previous octave
		dist *= 2;
		Mat newImg;
		// We take the image in numSpace(th) layer of the previous octave for the first img in this
		// current octave. (An octave ends when sigma of current layer is twice that of the
		// beginning layer of the octave)

		resize(gaussSpace[o - 1][numSpace], imgInOctave, Size(), 0.5, 0.5, INTER_AREA);
		gaussSpace[o].push_back(imgInOctave.clone());
		for (int s = 1; s <= numSpace + 2; s++) {
			filter2D(gaussSpace[o][s - 1], imgInOctave, -1, kernel[s], Point(-1, -1), 0.0, BORDER_CONSTANT);
			gaussSpace[o].push_back(imgInOctave.clone());
		}
	}
}

void MyDOG::computeDoGSpace() {
	Mat dogImg;
	int numLayer = gaussSpace[0].size();

	for (int o = 0; o < numOctave; o++) {
		for (int s = 0; s < numLayer - 1; s++) {
			dogImg = gaussSpace[o][s + 1] - gaussSpace[o][s];
			dogSpace[o].push_back(dogImg);
		}
	}
	cout << dogSpace[0][0].size() << "\n";
}

void MyDOG::findExtremaOfDogSpace(double threshold) {
	int numOctave = dogSpace.size();
	int numLayer = dogSpace[0].size();
	int x_ind[9] = { 0, -1, 1, -1, 1, 1, 0, -1, 0};
	int y_ind[9] = { 0, -1, 1, 1, -1, 0, 1, 0, -1};

	// Say, we want 3 layers in our dog space, 
	// in gaussian space, we must have:
	// @ * * * @ @, @ is extra image, * is considered image
	// |/|/|/|/|/
	// @ * * * @, two @ in the beginning and the end of octave will assure that when we find
	// the extrema, we wont be missing any img
	for (int o = 0; o < numOctave; o++) {
		for (int s = 1; s < numLayer - 1; s++) {
			for (int i = 1; i < dogSpace[o][s].rows - 1; i++) {
				for (int j = 1; j < dogSpace[o][s].cols - 1; j++) {
					double sample = dogSpace[o][s].at<double>(i, j);
					bool isExtrema = true;
					for (int k = 0; k < 9; k++) {
						if (sample < dogSpace[o][s - 1].at<double>(x_ind[k] + i, y_ind[k] + j) ||
							sample < dogSpace[o][s + 1].at<double>(x_ind[k] + i, y_ind[k] + j) ||
							(k != 0 && sample < dogSpace[o][s].at<double>(x_ind[k] + i, y_ind[k] + j))) {
							isExtrema = false;
							break;
						}
					}
					if (isExtrema && sample >= 0.8 * threshold) {
						DogKeypoint kp;
						kp.x = i;
						kp.y = j;
						kp.s = s;
						kp.octave = o;
						keypoints.push_back(kp);
					}
				}
			}
		}
	}
}


// interpolate keypoints and threshold low contrast
void MyDOG::localizeKeypoints(double threshold) {
	int numLayer = dogSpace[0].size();
	vector<DogKeypoint> afterLocalized;
	for (int kp = 0; kp < keypoints.size(); kp++) {
		Mat offset;
		double value;
		int octave = keypoints[kp].octave;
		int layer = keypoints[kp].s;
		int x = keypoints[kp].x;
		int y = keypoints[kp].y;
		int trialCount = 0;
		do {
			if (trialCount == 6) {

				break;
			}
			Mat grad = gradient(octave, layer, x, y);
			Mat hess = hessian(octave, layer, x, y);
			quadraticInterpolate(offset, value, dogSpace[octave][layer].at<double>(x, y), hess, grad);
			layer = round(layer + offset.at<double>(0, 0));
			x = round(x + offset.at<double>(1, 0));
			y = round(y + offset.at<double>(2, 0));
			trialCount++;
		} while (abs(offset.at<double>(0, 0)) > 0.6 ||
			abs(offset.at<double>(1, 0)) > 0.6 ||
			abs(offset.at<double>(2, 0)) > 0.6
			);

		// If we can find and offset  that satisfies max(abs(offset)) < 0.6 
		if (trialCount < 6 && abs(value) >= threshold) { // threshold low contrast
			double currentDist = minDist * pow(2, octave);
			DogKeypoint newKP;
			newKP.sigma = minSigma * pow(2.0, layer / (numLayer - 2) +  octave);
			newKP.x = x;
			newKP.y = y;
			newKP.s = layer;
			newKP.octave = octave;
			afterLocalized.push_back(newKP);
		}
		
	}
	keypoints = afterLocalized;
}

void MyDOG::discardKeypointsOnEdge(double threshold) {
	vector<DogKeypoint> afterDiscard;
	for (int i = 0; i < keypoints.size(); i++) {
		int octave = keypoints[i].octave;
		int layer = keypoints[i].s;
		int x = keypoints[i].x;
		int y = keypoints[i].y;
		Mat h = hessian(octave, layer, x, y);
		double trace = h.at<double>(1, 1) + h.at<double>(2, 2);
		double det = h.at<double>(1, 1) * h.at<double>(2, 2) - h.at<double>(1, 2) * h.at<double>(2, 1);
		double isEdge = trace * trace / det;
		if (isEdge <=  (threshold + 1) * (threshold + 1) / threshold) {
			keypoints[i].x = x * pow(2, keypoints[i].octave) * minDist;
			keypoints[i].y = y * pow(2, keypoints[i].octave) * minDist;
			
			afterDiscard.push_back(keypoints[i]);
		}
	}
	keypoints = afterDiscard;
}

Mat MyDOG::gradient(int octave, int layer, int x, int y) {
	Mat result(3, 1, CV_64F);
	result.at<double>(0, 0) = 0.5 * (dogSpace[octave][layer + 1].at<double>(x, y) - dogSpace[octave][layer - 1].at<double>(x, y));
	result.at<double>(1, 0) = 0.5 * (dogSpace[octave][layer].at<double>(x + 1, y) - dogSpace[octave][layer].at<double>(x - 1, y));
	result.at<double>(2, 0) = 0.5 * (dogSpace[octave][layer].at<double>(x, y + 1) - dogSpace[octave][layer].at<double>(x, y - 1));
	return result;
}

Mat MyDOG::hessian(int octave, int layer, int x, int y) {
	Mat result(3, 3, CV_64F);
	result.at<double>(0, 0) = dogSpace[octave][layer + 1].at<double>(x, y) + 
								dogSpace[octave][layer - 1].at<double>(x, y) -
								2 * dogSpace[octave][layer].at<double>(x, y);
	result.at<double>(1, 1) = dogSpace[octave][layer].at<double>(x + 1, y) +
								dogSpace[octave][layer].at<double>(x - 1, y) -
								2 * dogSpace[octave][layer].at<double>(x, y);
	result.at<double>(2, 2) = dogSpace[octave][layer].at<double>(x, y + 1) +
								dogSpace[octave][layer].at<double>(x, y - 1) -
								2 * dogSpace[octave][layer].at<double>(x, y);
	result.at<double>(0, 1) = result.at<double>(1, 0) = (dogSpace[octave][layer + 1].at<double>(x + 1, y) -
														dogSpace[octave][layer + 1].at<double>(x - 1, y) -
														dogSpace[octave][layer - 1].at<double>(x + 1, y) +
														dogSpace[octave][layer - 1].at<double>(x - 1, y)) / 4;
	result.at<double>(0, 2) = result.at<double>(2, 0) = (dogSpace[octave][layer + 1].at<double>(x, y + 1) -
														dogSpace[octave][layer + 1].at<double>(x, y - 1) -
														dogSpace[octave][layer - 1].at<double>(x, y + 1) +
														dogSpace[octave][layer - 1].at<double>(x, y - 1)) / 4;
	result.at<double>(1, 2) = result.at<double>(2, 1) = (dogSpace[octave][layer].at<double>(x + 1, y + 1) -
														dogSpace[octave][layer].at<double>(x + 1, y - 1) -
														dogSpace[octave][layer].at<double>(x - 1, y + 1) +
														dogSpace[octave][layer].at<double>(x - 1, y - 1)) / 4;
	return result;
}

void MyDOG::quadraticInterpolate(Mat& offset, double& value, double dogValue, Mat hessian, Mat gradient) {
	offset = - hessian.inv() * gradient;
	Mat temp =  0.5 * gradient.t() * hessian.inv() * gradient;
	value = dogValue - temp.at<double>(0, 0);
}


Mat detectDog(Mat img, Mat oc, int numOctave, int numLayer, double minSigma, double contrastThreshold, double edgeThreshold) {
	double minDist = 0.5;
	cout << "detecting DoG with " << numOctave << " octaves, " << numLayer << "layers per octave, " << "min sigma " << minSigma <<
		", contrast threshold " << contrastThreshold << ", edge threshold " << edgeThreshold <<"\n";
	MyDOG dog(img, minDist, numOctave, numLayer, minSigma, 0);
	dog.generateKernel();
	dog.computeGaussSpace();
	dog.computeDoGSpace();
	dog.findExtremaOfDogSpace(contrastThreshold);
	dog.localizeKeypoints(contrastThreshold);
	dog.discardKeypointsOnEdge(edgeThreshold);
	vector<DogKeypoint> kps = dog.getKeypoints();
	cout << "Total keypoints found" << kps.size() << "\n";
	for (int i = 0; i < kps.size(); i++) {
		circle(oc, Point(kps[i].y, kps[i].x), kps[i].sigma * sqrt(2), Scalar(0, 0, 255), 2);
	}
	return oc;
}


vector<DogKeypoint> findInterestedPoints(Mat img, vector<vector<Mat>>& gaussSpace) {
	vector<DogKeypoint> keypoints;
	double minDist = 0.5;
	int numOctave = 5;
	int numSpace = 3;
	double minSigma = 0.8;
	double contrastThreshold = 0.015;
	cout << "detecting DoG keypoints with " << numOctave << " octaves, " << numSpace << "layers per octave, "
		<< "min sigma is " << minSigma << ", contrast threshold is " << contrastThreshold << "\n";

	MyDOG dog(img, minDist, numOctave, numSpace, minSigma, 0);
	dog.generateKernel();
	dog.computeGaussSpace();
	dog.computeDoGSpace();
	dog.findExtremaOfDogSpace(contrastThreshold);
	dog.localizeKeypoints(contrastThreshold);
	dog.discardKeypointsOnEdge(10);
	vector<DogKeypoint> kps = dog.getKeypoints();
	cout << "Total keypoints found " << kps.size() << "\n";
	gaussSpace = dog.getGaussSpace();
	
	return kps;
}




