#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <aruco/aruco.h>
#include <aruco/cvdrawingutils.h>

#include <cpprest/http_client.h>
#include <cpprest/json.h>

using namespace std;
using namespace web;
using namespace web::http;

client::http_client mClient(U("http://localhost:56789"));

void sendPoseToServer(array<float,9> pose, int objId);

int main() {
	char key = ' ';
	float markerSize = 0.041;
	
	cv::Mat cameraFeed, cameraFeedCopy;
	cv::Mat cameraMatrix, distCoeff;
	cv::Size imageSize;
	cv::VideoCapture cam;
	cv::FileStorage fs;
	
	aruco::MarkerDetector markerDetector;
	aruco::CameraParameters cameraParams;
	aruco::MarkerMap markerMapConfig;
	aruco::MarkerMapPoseTracker markerMapPoseTracker;
	vector< aruco::Marker> markers;
	
	cam.open(1);
	if (!cam.isOpened()) {
		cerr << "Aruco: camera failed to open, exit." << endl;
		exit(-1);
	}
//  cam.set(CV_CAP_PROP_EXPOSURE,0.5);
// 	cam.set(CV_CAP_PROP_CONTRAST,0.125); // default 0.12549

	cout << "exposure: " << cam.get(CV_CAP_PROP_EXPOSURE) << " contrast: " << cam.get(CV_CAP_PROP_CONTRAST) << endl;
	cv::waitKey();
// 	return 0;

	fs.open("calib_param.yml",cv::FileStorage::READ);
	if (!fs.isOpened()) {
		cerr << "Aruco: calibration file failed to open, exit." << endl;
		exit(-1);
	}
	fs["M"] >> cameraMatrix;
	fs["D"] >> distCoeff;
	fs.release();
	
	markerMapConfig.readFromFile("aruco_map_config_1.yml");
	markerMapConfig = markerMapConfig.convertToMeters(markerSize);
	
	cam >> cameraFeed;
	imageSize = cameraFeed.size();
	cameraParams.setParams(cameraMatrix,distCoeff,imageSize);
	if (!cameraParams.isValid()) {
		cerr << "Aruco: invalid camera parameters, exit." << endl;
		exit(-1);
	}
	
	markerMapPoseTracker.setParams(cameraParams,markerMapConfig);
	
	markerDetector.setDictionary(aruco::Dictionary::ARUCO_MIP_36h12);
	
	while (key!=27) {
		cam >> cameraFeed;
		cameraFeed.copyTo(cameraFeedCopy);
		
		markers = markerDetector.detect(cameraFeed);
		for (int i=0; i<markers.size(); i++) {
			markers[i].draw(cameraFeedCopy,cv::Scalar(0,0,255),2);
		}
		
		if (markerMapPoseTracker.estimatePose(markers)) {
			cv::Mat rvec,tvec,rotationMatrix;
			markerMapPoseTracker.getRvec().copyTo(rvec);
			markerMapPoseTracker.getTvec().copyTo(tvec);
			cv::Rodrigues(rvec,rotationMatrix);
			
			array<float,9> pose;
			float fx,fy,fz,ux,uy,uz,tx,ty,tz;
			
			tx = tvec.at<float>(0);
			ty = - tvec.at<float>(1);
			tz = tvec.at<float>(2);
			
			fx = rotationMatrix.at<float>(0,1);
			fy = - rotationMatrix.at<float>(1,1);
			fz = rotationMatrix.at<float>(2,1);
			
			ux = rotationMatrix.at<float>(0,2);
			uy = - rotationMatrix.at<float>(1,2);
			uz = rotationMatrix.at<float>(2,2);
			
			pose[0] = tx; pose[1] = ty; pose[2] = tz;
			pose[3] = fx; pose[4] = fy; pose[5] = fz;
			pose[6] = ux; pose[7] = uy; pose[8] = uz;
			
			sendPoseToServer(pose,0);
			
			cout << "Aruco: forward = (" << fx << ", " << fy << ", " << fz << ")" << endl; 
			cout << "Aruco: upward = (" << ux << ", " << uy << ", " << uz << ")" << endl;
			cout << "Aruco: position = (" << tx << ". " << ty << ", " << tz << ")" << endl << endl;
			
			aruco::CvDrawingUtils::draw3dAxis(cameraFeedCopy,
											  cameraParams,
											  rvec,
											  tvec,
									 		  markerMapConfig[0].getMarkerSize()*5);
		}
		
		cv::imshow("marker map tracking", cameraFeedCopy);
		key = cv::waitKey(33);
	}
	
	cam.release();
	
	return 0;
}

void sendPoseToServer(array<float,9> pose, int objId) {
	if (pose.size()!=9) return;
	
	json::value poseObj;
	poseObj[U("id")] = json::value::number(objId);
	poseObj[U("tx")] = json::value::number(pose[0]);
	poseObj[U("ty")] = json::value::number(pose[1]);
	poseObj[U("tz")] = json::value::number(pose[2]);
	poseObj[U("fx")] = json::value::number(pose[3]);
	poseObj[U("fy")] = json::value::number(pose[4]);
	poseObj[U("fz")] = json::value::number(pose[5]);
	poseObj[U("ux")] = json::value::number(pose[6]);
	poseObj[U("uy")] = json::value::number(pose[7]);
	poseObj[U("uz")] = json::value::number(pose[8]);
	
	http::http_request request;
	request.set_method(http::methods::PUT);
	request.headers().set_content_type(U("application/json"));
	request.set_request_uri(U("/"));
	request.set_body(poseObj);
	
	try {
	pplx::task<http::http_response> requestTask = mClient
		.request(request)
		.then([=](pplx::task<http::http_response> task) {
			http::http_response response;
			try {
				response = task.get();
				printf("Aruco: received response status code:%u\n", response.status_code());
			} catch (exception& e) {
				printf("Aruco: response error exception:%s\n", e.what());
			}
			return task;
		});
	} catch (exception& e) {
		printf("Aruco: request error exception:%s\n", e.what());
	}
}