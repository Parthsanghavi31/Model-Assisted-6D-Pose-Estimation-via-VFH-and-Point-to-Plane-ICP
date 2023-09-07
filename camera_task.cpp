/**
 * Copyright (C) Ghost Robotics - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Avik De <avik@ghostrobotics.io>
 */
#include "camera_task.hpp"
#include <opencv2/imgproc.hpp>

bool CameraTask::init(const std::string &serial) {
	pts_C_ = std::make_shared<gr::PointCloud_t>();
	
	// Start camera
	rs2::context ctx;
	cam_ = std::make_unique<ctrl::RSCamera>(ctx, serial, 0, 0, Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero(), 0, 60, "");
	return cam_->enabled();
}

bool CameraTask::sampleNewCamData() {
	bool got_frame = cam_->getFrame(true);

	if (!got_frame) {
		return false; // stop since one of the frames will be null
	}

	// // Publish front and rear depth images for the autonomy team, and gripper cam depth image
	// image_list_mutex_.lock();
	// image_list_.clear();
	// if (cams_.size()>=4) {
	// 	image_list_.push_back(cams_[2].depthData());
	// 	image_list_.push_back(cams_[3].depthData());
	// }
	// if (cams_.size()>=5) {
	// 	cams_[4].updateExtrinsics(latest_data_.pgc_W, latest_data_.Rgc);
	// 	image_list_.push_back(cams_[4].depthData());
	// }
	// // Publish RGB IPC images metadata at a decimated rate
	// if (cams_.size()>=1) {
	// 	rgb_ipc_dec_ctr_ = (rgb_ipc_dec_ctr_ + 1) % 30;
	// 	if (rgb_ipc_dec_ctr_ == 0) {
	// 		for (size_t i=0; i<cams_.size(); ++i) {
	// 			if (cams_[i].video_rate_ > 0) {
	// 				image_list_.push_back(cams_[i].rgbData());
	// 			}
	// 		}
	// 	}
	// }
	// image_list_mutex_.unlock();

	const double depth_time = cam_->processCloud(pts_C_);
	double dt_ms = depth_time - depth_time_prev_; // in milliseconds
	if (dt_ms > 1e-3) {
		avg_rate_ += 0.1 * (1e3/dt_ms - avg_rate_);
	}
	depth_time_prev_ = depth_time;

	const double rgb_time = cam_->colorTimestamp();
	dt_ms = rgb_time - rgb_time_prev_; // in milliseconds
	if (dt_ms > 1e-3) {
		avg_rate_rgb_ += 0.1 * (1e3/dt_ms - avg_rate_rgb_);
	}
	rgb_time_prev_ = rgb_time;

	return true;
}

void CameraTask::cleanup() {
	if (cam_)
		cam_->stop();
}

void CameraTask::update() {
	const ImageData &rgb_data = cam_->rgbData();
	input_img_ = cv::Mat(rgb_data.height, rgb_data.width, CV_8UC3, (uint8_t *)cam_->colorBytes());

	output_img_ = input_img_;
	
	/// NOTE:
	// Run any algorithms here
	// pts_C_ contains the point cloud
	// The intrinsics are in rgbData() and depthData()

	/// FIXME: just testing adding PCL
	cloud_.push_back(PointT(1.0f, 2.0f, 3.0f));
	cloud_.clear();

	if (user_x_ > 0 && user_y_ > 0) {
		// Draw something where the user clicked
		cv::circle(output_img_, cv::Point2f((float)user_x_, (float)user_y_), 10, cv::Scalar(255, 0, 0), -1);
	}
}

void CameraTask::userInput(int x, int y) {
	user_x_ = x;
	user_y_ = y;
}
