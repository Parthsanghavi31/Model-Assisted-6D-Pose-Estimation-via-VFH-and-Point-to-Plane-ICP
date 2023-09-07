
#pragma once

#include <Eigen/Core>

#ifdef USE_CAMERAS_ON
#include <librealsense2/rs.hpp>     // Include RealSense Cross Platform API
#endif
#include <vector>
#ifdef _MSC_VER
// #pragma warning(disable : 4244)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#ifdef _MSC_VER
// #pragma warning(default : 4244)
#else
#pragma GCC diagnostic pop
#endif

namespace gr
{

typedef Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> PointCloud_t;

} // namespace gr

namespace ctrl {

inline gr::PointCloud_t filterPoints(const Eigen::ArrayXi &is_selected, const gr::PointCloud_t &pts) {
	// Copy the C frame filtered points
	gr::PointCloud_t pts_filt(is_selected.sum(), 3);

	int rownew = 0;
	for (int j = 0; j < pts.rows(); ++j) {
		if (is_selected[j]) {       
			pts_filt.row(rownew) = pts.row(j);
			rownew++;
		}
	}
	return pts_filt;
}

#ifdef USE_CAMERAS_ON
// Abstract base class for this calculation
class DepthToCloud {
public:
	// Returns pts_C_filt
	virtual gr::PointCloud_t operator()(const rs2::frame &processed_depth, bool b_downward_facing) = 0;

protected:
	// For filtering points in camera frame
	float zmin_ = 0.14f, zmax_ = 2.0f, ymax_ = 0.25f, x_over_z_max_ = 0.9f;

	// Filtering the points using eigen
	gr::PointCloud_t filterEigen(const gr::PointCloud_t &pts_C, bool b_downward_facing);
};

class DepthToCloudLibRS : public DepthToCloud {
public:
	gr::PointCloud_t operator()(const rs2::frame &processed_depth, bool b_downward_facing);
protected:	
	rs2::pointcloud pc_;
	rs2::points pts_;
};
#endif

} // namespace ctrl
