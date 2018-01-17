#include <iostream>
#include <aruco/aruco.h>
#include <aruco/cvdrawingutils.h>
#include <opencv2/highgui.hpp>

using namespace std;

int main(int argc, char **argv) {
	
	cv::VideoCapture cam;
	cam.open(0);
	
	if (!cam.isOpened()) {
		cout << "Aruco_marker: camera failed to open, exit." << endl;
		exit(-1);
	}
	
	cv::FileStorage fs;
	fs.open("calib_param.yml",cv::FileStorage::READ);
	if (!fs.isOpened()) {
		cout << "Aruco_marker: file failed to open, exit." << endl;
		exit(-1);
	}
	
	char key = ' ';
	cv::Mat cameraFeed, cameraFeedCopy;
	cv::Mat cameraMatrix, distCoeff;
	cv::Size imageSize;
	aruco::MarkerDetector markerDetector;
	aruco::CameraParameters cameraParams;
	std::map<uint32_t,aruco::MarkerPoseTracker> markerPoseTrackers;
	
	markerDetector.setDictionary(aruco::Dictionary::ARUCO_MIP_36h12);
	
	aruco::MarkerMap markerMap;
	
	fs["M"] >> cameraMatrix;
	fs["D"] >> distCoeff;
	fs.release();
	
	cout << "Aruco_marker: start" << endl;
	
	while (key != 27) {
		vector<aruco::Marker> markers;
		
		cam >> cameraFeed;
		cameraFeed.copyTo(cameraFeedCopy);
		
		if (imageSize==cv::Size()) {
			imageSize = cameraFeed.size();
			cameraParams.setParams(cameraMatrix,distCoeff,imageSize);
			cout << "Aruco_marker: image size = " << imageSize << endl;
		}
		
		if (!cameraParams.isValid()) {
			std::cerr << "Aruco_marker: invalid camera parameters, exit." << endl;
			exit(-1);
		}
		
		markers = markerDetector.detect(cameraFeed);
		
		for (int i = 0; i<markers.size(); i++) {
			cout << "Aruco_marker: " << markers[i] << endl;
			aruco::Marker marker = markers[i];
			bool estimated = markerPoseTrackers[marker.id].estimatePose(marker,cameraParams,0.078);
			marker.draw(cameraFeedCopy,cv::Scalar(0,0,255),2);
			if (estimated) {
				cout << "Aruco_marker: rvec =" << marker.Rvec << "\n tvec =" << marker.Tvec << endl;
				aruco::CvDrawingUtils::draw3dAxis(cameraFeedCopy,marker,cameraParams);
			}
		}
		
		cv::imshow("Tracking",cameraFeedCopy);
		key = cv::waitKey(30);
	}
	
    return 0;
}
