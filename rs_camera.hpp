
#pragma once

#include "depth_to_cloud.hpp"

// From pb
typedef struct _ImageData {
  const char *data; 
	uint32_t width; 
	uint32_t height; 
	/* If BYTES type, will try to match ROS
	If GST_IPC type, will contain the RX pipeline prefix */
	char encoding[500]; 
	uint32_t cam_idx; /* 0--n depth, 10--10+n color */
	/* Extrinsics: p_B = RotT^T * p_C + trans */
	float RotT[9]; /* transpose of the rotation matrix */
	float trans[3]; /* translation */
	/* Intrinsics */
	float principal_point[2]; 
	float focal_length[2]; 
	/* Distortion */
	char distortion_model[30]; 
	float distortion_params[5]; 
	uint32_t data_size; /* store the size before proto encoding */
	/* If image */
	// ImageType image_type; 
	uint32_t flip_method; /* For GST_IPC only, gstreamer flip-method */
} ImageData;

namespace ctrl {

// Class for managing a single camera
class RSCamera {
public:
	// Called from the main thread
	// exposure=0 => auto
	// video true => enable RGB in pipeline
	RSCamera(const rs2::context &ctx, const std::string &serial, float depth_exposure, int i, const Eigen::Matrix3f &RotT, const Eigen::Vector3f &translation, int video_flip_method, int video_rate, const std::string &video_ipc_path_root) noexcept;

	// Stop streaming; return false if there is an error
	bool stop() noexcept;

	// Start streaming; return false if there is an error
	bool start() noexcept;

	bool enabled() const { return b_enabled_; }

	// Process the depth frame, populate a body-frame point cloud to pts_B, and return the timestamp of the cloud
	double processCloud(std::shared_ptr<gr::PointCloud_t> pts_C) noexcept;

	// Check if there is a new frame, if wait is true it will wait for a short time, otherwise check and return immediately
	bool getFrame(bool b_wait) noexcept;

	const int i_; // correspond to realsense index, or -1 for test src
	const std::string serial_;
	const int video_rate_; // set 0 to disable video streaming
	const static int depth_rate_ = 60;

	// Info about the latest color frame
	int colorBytesSize() const { return color_data_size_; }
	const char * colorBytes() const { return color_data_; }
	double colorTimestamp() const { return color_frame_.get_timestamp(); }

	// Info about the latest depth frame
	const ImageData &depthData() const { return depth_data_; }

	// Info about the RGB data (note - this does not really change after init)
	const ImageData &rgbData() const { return rgb_data_; }

	// Update extrinsics in case they changed after the constructor
	void updateExtrinsics(const Eigen::Vector3f &trans, const Eigen::Matrix3f &Rot);

private:
	rs2::frameset frameset_;
	rs2::pipeline pipeline_;
	int pc_calc_dt_;
	uint64_t prev_color_frame_num_ = 0, prev_ir_frame_num_ = 0;
	rs2::frame color_frame_, depth_frame_;
	// Potentially could change these in the future
	bool b_enabled_ = false; // set true in constructor after starting
	std::vector<char> raw_depth_;

	/// NOTE: removed the ability to have # processing threads < # cams
	// If that is reinstated, these should be part of the depth processing object
	rs2::decimation_filter dec_filter_;  // Decimation - reduces depth frame density
	// rs2::spatial_filter spatial_filter_;
	rs2::temporal_filter temporal_filter_;
	DepthToCloudLibRS d2c_; //can choose different implementations
	
	// Transform points from camera to body frame
	void bTc(std::shared_ptr<gr::PointCloud_t> pts_B, Eigen::Ref<const gr::PointCloud_t> pts_C) const {
		*pts_B = pts_C * Eigen::Map<const Eigen::Matrix3f>(depth_data_.RotT); // right-multiply by RotT which is the transpose
		pts_B->rowwise() += Eigen::Map<const Eigen::Vector3f>(depth_data_.trans).transpose(); 
	}

	int color_data_size_ = 0; // latest color frame data size
	const char * color_data_ = nullptr; // latest color frame data pointer

	ImageData depth_data_;
	ImageData rgb_data_;
	float depth_units_ = 1e-3f;

	void storeIntrinsics(ImageData *dest, const rs2::video_stream_profile &profile);
};

} // namespace ctrl
