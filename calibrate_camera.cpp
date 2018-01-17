#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

using namespace cv;  
using namespace std;

static void cameraCalibration(bool displayCorners, Size boardSize, 
							  float squareSize, bool useExist);
static double computeReprojectionError(const vector< vector<Point3f> >& objectPnts, 
									 const vector< vector<Point2f> >& imagePnts, 
									 const vector<Mat>& rvecs,
									 const vector<Mat>& tvecs,
									 const Mat& cameraMatrix,
									 const Mat& distCoeff);

int main() {
	Size boardSize;
	boardSize.width = 9;
	boardSize.height = 6;
	bool displayCorners = true;
	bool useExist = true;
	float squareSize = 23.0;
	cameraCalibration(displayCorners,boardSize,squareSize,useExist);
	return 0; 
}

static void cameraCalibration(bool displayCorners, Size boardSize, float squareSize, bool useExist) {
	
	vector< vector<Point2f> > imagePnts;
	vector< vector<Point3f> > objectPnts;
	vector<string> goodImageList;
	int nImages = 25, maxScale = 2;
	int i = 0, j = 0, k = 0;
	Size imageSize;
	stringstream ss;
	
	imagePnts.resize(nImages);
	
	VideoCapture cam(1);
	if (!cam.isOpened()) return;
	
	namedWindow("camera_feed",CV_WINDOW_AUTOSIZE);
	Mat cameraFeed;
	
	if (useExist) {
		for (int idx = 0; idx<nImages; idx++) {
			ss << "cameraCalib_" << idx << ".png";
			goodImageList.push_back(ss.str());
			ss.str(string());
		}
	}
	
	// take calibration images
	while (i<nImages) {
		if (!useExist) cam >> cameraFeed;
		else cameraFeed = imread(goodImageList[i]);
		
		imshow("camera_feed",cameraFeed);
		char key = (char)waitKey(25);
		if (key==27) return;
		if (key=='c'||useExist) {
			cvtColor(cameraFeed,cameraFeed,COLOR_BGR2GRAY);
			if (cameraFeed.empty()) continue;
			
			if (imageSize == Size()) imageSize = cameraFeed.size();
			else if (imageSize != cameraFeed.size()) {
				cout << "CameraCalib: inconsistent image sizes, skipping the frame." << endl;
				continue;
			}
			
			bool chessboardFound = false;
			
			vector<Point2f> &corners = imagePnts[i];
			
			for (int scale = 1; scale <= maxScale; scale++) {
				Mat timg;
				if (scale == 1) timg = cameraFeed;
				else resize(cameraFeed,timg,Size(),scale,scale);
				chessboardFound  = findChessboardCorners(timg,boardSize,corners,
								CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
				if (chessboardFound) {
					if (scale>1) {
						Mat cornersMat(corners);
						cornersMat *= 1./scale;
					}
					break;
				}
			}
			
			if (displayCorners && chessboardFound) {
				cout << "CameraCalib: displaying corners for image: " << i+1 << "/" << nImages << endl;
				Mat cimg, cimg1;
				cvtColor(cameraFeed,cimg,COLOR_GRAY2BGR);
				drawChessboardCorners(cimg, boardSize, corners, chessboardFound);
				double sf = 600./MAX(cameraFeed.rows, cameraFeed.cols);
				resize(cimg,cimg1,Size(),sf,sf);
				imshow("Corners",cimg1);
				char c = (char)waitKey(500);
				if (c == 27) exit(-1);
			}
			
			if (!chessboardFound) {
				cout << "CameraCalib: chessboard pattern not found!" << endl;
				continue;
			}
			
			cornerSubPix(cameraFeed,corners,Size(11,11),Size(-1,-1),
						 TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,30,0.01));
			
			if (!useExist) {
				ss << "cameraCalib_" << i << ".png";
				imwrite(ss.str(),cameraFeed);
				goodImageList.push_back(ss.str());
				ss.str(string());
			}
			i++;
		}
		
	}
	
	cam.release();
	
	objectPnts.resize(nImages);
	
	// construct 3D coordinates for world points
	for (i=0;i<nImages;i++) {
		for (j=0;j<boardSize.height;j++)
			for (k=0;k<boardSize.width;k++)
				objectPnts[i].push_back(Point3f(k*squareSize,j*squareSize,0));
	}
	
	cout << "CameraCalib: running calibration ..." << endl;
	
	Mat cameraMatrix = initCameraMatrix2D(objectPnts,imagePnts,imageSize,0);
	Mat distCoeff;
	vector<Mat> rvecs, tvecs;
	
	double rms = calibrateCamera(objectPnts,imagePnts, imageSize, cameraMatrix,
								 distCoeff, rvecs, tvecs, 
								 cv::CALIB_USE_INTRINSIC_GUESS +
								 cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_RATIONAL_MODEL +
								 cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5); 	
	
	double err_reproject = computeReprojectionError(objectPnts,imagePnts,rvecs,tvecs,cameraMatrix,distCoeff);
	cout << "CameraCalib: reprojection error = " << err_reproject << endl;
	
	// distortion correction
	cout << "CameraCalib: applying distortion correction" << endl;
	Mat map1,map2, view, rview, canvas;
	double sf;
	int w,h;
	
	sf = 600./MAX(imageSize.width,imageSize.height);
	w = cvRound(imageSize.width*sf);
	h = cvRound(imageSize.height*sf);
	canvas.create(h,w*2,CV_8UC1);
	
	initUndistortRectifyMap(cameraMatrix,distCoeff,Mat(),
							getOptimalNewCameraMatrix(cameraMatrix,distCoeff,imageSize,1,imageSize,0),
							imageSize,CV_16SC2,map1,map2);
	for  (i = 0; i<goodImageList.size();i++) {
		view = imread(goodImageList[i],cv::IMREAD_GRAYSCALE);
		if (view.empty()) continue;
		cout << "CameraCalib: applying distortion correction on image: " << i+1 << "/" << nImages << endl;
		remap(view,rview,map1,map2,cv::INTER_LINEAR);
		Mat canvasLeft = canvas(Rect(0,0,w,h));
		Mat canvasRight = canvas(Rect(w,0,w,h));
		resize(view,canvasLeft,canvasLeft.size(),0,0,CV_INTER_AREA);
		resize(rview,canvasRight,canvasRight.size(),0,0,CV_INTER_AREA);
		imshow("Undistorted Images", canvas);
		char key = waitKey();
		if (key == 27) break;
		else continue;
	}
	
	// save calibration parameters
	FileStorage fs("calib_param.yml",FileStorage::WRITE);
	if (fs.isOpened()) {
		fs << "M" << cameraMatrix << "D" << distCoeff << "R" << rvecs << "T" << tvecs;
		fs.release();
		cout << "CameraCalib: camera calibration params saved." << endl;
	} else cout << "CameraCalib: failed to save calibration params !!" << endl;
}

static double computeReprojectionError(const vector< vector<Point3f> >& objectPnts, 
									   const vector< vector<Point2f> >& imagePnts, 
									   const vector<Mat>& rvecs,
									   const vector<Mat>& tvecs,
									   const Mat& cameraMatrix,
									   const Mat& distCoeff) {
	vector<Point2f> imagePnts2;
	size_t totalPnts = 0;
	double totalErr = 0;
	for (size_t i = 0; i<objectPnts.size(); i++) {
		projectPoints(objectPnts[i], rvecs[i], tvecs[i], cameraMatrix, distCoeff, imagePnts2);
		totalErr += norm(imagePnts[i],imagePnts2,NORM_L2);
		totalPnts += objectPnts[i].size();
	}
	
	return sqrt(totalErr/totalPnts);
}