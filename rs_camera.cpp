/**
 * Copyright (C) Ghost Robotics - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Avik De <avik@ghostrobotics.io>
 */
#include "rs_camera.hpp"
#define LOGURU_WITH_STREAMS 1
#include <loguru.hpp>
#include <chrono>
using namespace std::chrono;

namespace ctrl {

RSCamera::RSCamera(const rs2::context &ctx, const std::string &serial, float depth_exposure, int i, const Eigen::Matrix3f &RotT, const Eigen::Vector3f &translation, int video_flip_method, int video_rate, const std::string &video_ipc_path_root) noexcept : 
	i_(i), serial_(serial), video_rate_(video_rate),
	// use the provided context
	pipeline_(ctx)
{
	memset(&depth_data_, 0, sizeof(depth_data_));
	memset(&rgb_data_, 0, sizeof(rgb_data_));

	// depth_data_.image_type = ImageType_BYTES;
	depth_data_.width = 640;
	depth_data_.height = 360;
	depth_data_.cam_idx = i;
	const char *depth_encoding = "16UC1"; // Z16 in realsense
	strncpy(depth_data_.encoding, depth_encoding, sizeof(depth_data_.encoding));
	// Extrinsics
	Eigen::Map<Eigen::Matrix3f>(depth_data_.RotT) = RotT;
	Eigen::Map<Eigen::Vector3f>(depth_data_.trans) = translation;

	// RGB data
	// rgb_data_.image_type = ImageType_GST_IPC;
	rgb_data_.width = 960;
	rgb_data_.height = 540;
	rgb_data_.cam_idx = 10 + i;
	// Extrinsics
	Eigen::Map<Eigen::Matrix3f>(rgb_data_.RotT) = RotT;
	Eigen::Map<Eigen::Vector3f>(rgb_data_.trans) = translation;

	if (!start())
		return; // camera is not working; leave enabled as false
	
	// Check that the depth profile was selected
	auto profile = pipeline_.get_active_profile();
	auto depth_profile = rs2::video_stream_profile(profile.get_stream(RS2_STREAM_DEPTH));
	
	LOG_S(INFO) << profile.get_device().get_info(RS2_CAMERA_INFO_NAME) << 
	" sn=" << serial << ", depth=" << 
	depth_profile.width() << "x" << depth_profile.height() << 
	"@" << depth_profile.fps() << "(" << (uint32_t)depth_profile.format() << ")";
	storeIntrinsics(&depth_data_, depth_profile);

	if (video_rate_ > 0) {
		// Print color settings
		auto rgb_profile = rs2::video_stream_profile(profile.get_stream(RS2_STREAM_COLOR));
		LOG_S(INFO) << "\trgb=" << 
		rgb_profile.width() << "x" << rgb_profile.height() << 
		"@" << rgb_profile.fps() << "(" << (uint32_t)rgb_profile.format() << ")";
		storeIntrinsics(&rgb_data_, rgb_profile);

		// auto ir_profile = rs2::video_stream_profile(profile.get_stream(RS2_STREAM_INFRARED));
	}

	// Setting options
	auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();

	if (depth_exposure > 0.5f) {
		try {
			depth_sensor.set_option(RS2_OPTION_EXPOSURE, depth_exposure);
			LOG_S(INFO) << "\texposure=" << depth_exposure;
		} catch (const rs2::error& e) {
			LOG_S(ERROR) << "\tFailed to set option. (" << e.what() << ")";
		}
	} else {
		LOG_S(INFO) << "\texposure=auto";
	}

	// Set depth units in case we need to hardcode https://gitlab.com/ghostrobotics/controls/-/merge_requests/812
	try {
		depth_sensor.set_option(RS2_OPTION_DEPTH_UNITS, depth_units_);
	} catch (const rs2::error& e) {
		LOG_S(ERROR) << "Failed to set option. (" << e.what() << ")";
	}

	// Set filter block parameters
	dec_filter_.set_option(RS2_OPTION_FILTER_MAGNITUDE, 4);
	temporal_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.4f);
	temporal_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20);
	// spatial_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.6f);
	// spatial_filter_.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 8);
	
	if (video_rate_ > 0) {
		// initPushIPC(video_ipc_path_root + std::to_string(i), rgb_data_.width, rgb_data_.height, video_rate_, "YUY2");
		// strncpy(rgb_data_.encoding, rx_pipeline_prefix_.c_str(), sizeof(rgb_data_.encoding));
		// rgb_data_.flip_method = video_flip_method;
	}
	
	b_enabled_ = true;
}

void RSCamera::updateExtrinsics(const Eigen::Vector3f &trans, const Eigen::Matrix3f &Rot) {
	Eigen::Map<Eigen::Matrix3f>(rgb_data_.RotT) = Eigen::Map<Eigen::Matrix3f>(depth_data_.RotT) = Rot.transpose();
	Eigen::Map<Eigen::Vector3f>(rgb_data_.trans) = Eigen::Map<Eigen::Vector3f>(depth_data_.trans) = trans;
}

void RSCamera::storeIntrinsics(ImageData *dest, const rs2::video_stream_profile &profile) {
	try {
		//If the stream is indeed a video stream, we can now simply call get_intrinsics()
		rs2_intrinsics intrinsics = profile.get_intrinsics();

		dest->principal_point[0] = intrinsics.ppx;
		dest->principal_point[1] = intrinsics.ppy;
		dest->focal_length[0] = intrinsics.fx;
		dest->focal_length[1] = intrinsics.fy;
		strncpy(dest->distortion_model, rs2_distortion_to_string(intrinsics.model), sizeof(dest->distortion_model));
		memcpy(dest->distortion_params, intrinsics.coeffs, sizeof(dest->distortion_params));
	}
	catch (const std::exception& e) {
		LOG_S(WARNING) << "Failed to get intrinsics for the given stream: " << e.what();
	}
}

bool RSCamera::start() noexcept {
	rs2::config cfg;
	try {
		if (serial_.length() > 0)
			cfg.enable_device(serial_);

		if (video_rate_ > 0) {
			// Color for video streaming
			cfg.enable_stream(RS2_STREAM_COLOR, rgb_data_.width, rgb_data_.height, RS2_FORMAT_BGR8, video_rate_);
			// cfg.enable_stream(RS2_STREAM_COLOR, 960, 540, RS2_FORMAT_RGB8, video_rate_); // tablet decoder testing
			// cfg.enable_stream(RS2_STREAM_INFRARED, 640, 360, RS2_FORMAT_Y8, depth_rate_);
		}

		// Start streaming depth
		cfg.enable_stream(RS2_STREAM_DEPTH, depth_data_.width, depth_data_.height, RS2_FORMAT_Z16, depth_rate_);

		pipeline_.start(cfg); // can throw an exception
	}
	catch (const rs2::error & e) {
		LOG_S(ERROR) << "Error starting camera " << i_ << " with serial " << serial_ << " " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what();
		return false;
	}
	return true;
}

bool RSCamera::stop() noexcept {
	if (!b_enabled_)
		return false;
	try {
		pipeline_.stop();
	} catch (const rs2::error& e) {
		LOG_S(ERROR) << "Failed to stop pipeline for camera " << i_ << " (" << e.what() << ")";
		return false;
	}
	return true;
}

bool RSCamera::getFrame(bool b_wait) noexcept {
	if (!b_enabled_)
		return false;
	static bool ret;
	try {
		if (b_wait)
			ret = pipeline_.try_wait_for_frames(&frameset_, 500); // timeout in ms
		else
			ret = pipeline_.poll_for_frames(&frameset_);
	} catch (const rs2::error& e) {
		LOG_S(ERROR) << "Failed to get depth data for camera " << i_ << " (" << e.what() << ")";
		ret = false;
	}

	if (video_rate_ > 0) {
		color_frame_ = frameset_.get_color_frame();
		uint64_t color_frame_num = color_frame_ ? color_frame_.get_frame_number() : 0;
		if (color_frame_num != prev_color_frame_num_) {
			// Process the RGB frame
			color_data_size_ = color_frame_.get_data_size();
			color_data_ = (const char *)color_frame_.get_data();
		} else {
			color_data_size_ = 0; // no new frame
		}
		prev_color_frame_num_ = color_frame_num;
	}

	return ret;
}

double RSCamera::processCloud(std::shared_ptr<gr::PointCloud_t> pts_C) noexcept {
	// RS2 filters: example here https://dev.intelrealsense.com/docs/post-processing-filters (copy?)
	depth_frame_ = frameset_.get_depth_frame();

	// Copy depth and store data pointer
	raw_depth_.resize(depth_frame_.get_data_size());
	memcpy(&raw_depth_[0], depth_frame_.get_data(), depth_frame_.get_data_size());
	depth_data_.data = &raw_depth_[0];
	depth_data_.data_size = depth_frame_.get_data_size();

	double timestamp = depth_frame_.get_timestamp();

	// Convert to point cloud, and cam to body frame
	// if (i_ < 4) {
	depth_frame_ = dec_filter_.process(depth_frame_);
	depth_frame_ = temporal_filter_.process(depth_frame_);
	auto t1 = steady_clock::now();
	*pts_C = d2c_(depth_frame_, i_<2);
	auto t2 = steady_clock::now();
	pc_calc_dt_ = (int)duration_cast<microseconds>(t2-t1).count();
		// pts_B->resizeLike(pts_C_filt);
		// bTc(pts_B, pts_C_filt);
	// } else if (i_ == 4) {
	// 	// No point cloud conversion for gripper cam at the moment
	// }
	return timestamp;
}

} // namespace ctrl
