/**
 * Copyright (C) Ghost Robotics - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Avik De <avik@ghostrobotics.io>
 */
#pragma once

#include "rs_camera.hpp"
#include <memory>
#include <opencv2/core.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

class CameraTask {
public:
	typedef pcl::PointXYZ PointT;

	bool init(const std::string &serial);

	bool sampleNewCamData();

	void cleanup();

	std::shared_ptr<gr::PointCloud_t> pts_C_;

	double avgRate() const { return avg_rate_; }
	double avgRateRGB() const { return avg_rate_rgb_; }

	// Return wrapper around RGB image
	cv::Mat outputImage() const { return output_img_; }
	cv::Mat inputImage() const { return input_img_; }

	void update();

	// Call this with user input in the window
	void userInput(int x, int y);

private:
	std::unique_ptr<ctrl::RSCamera> cam_;
	double depth_time_prev_ = 0, rgb_time_prev_ = 0, avg_rate_ = 0, avg_rate_rgb_ = 0;

	cv::Mat input_img_, output_img_;

	int user_x_ = -1, user_y_ = -1;

	/// FIXME: just testing adding PCL
	pcl::PointCloud<PointT> cloud_;
};
