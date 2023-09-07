/**
 * Copyright (C) Ghost Robotics - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Avik De <avik@ghostrobotics.io>
 */
#include "depth_to_cloud.hpp"

namespace ctrl {

gr::PointCloud_t DepthToCloud::filterEigen(const gr::PointCloud_t &pts_C, bool b_downward_facing) {

	// Filter the points
	Eigen::ArrayXi is_selected;
	if (b_downward_facing) { //downward cameras
		is_selected = (pts_C.col(2).array() > zmin_).cast<int>() * 
		(pts_C.col(2).array() < zmax_).cast<int>() * 
		(pts_C.col(1).array().abs() < ymax_).cast<int>();
	} else { //front and back horizontal camera
		//No filter in y or x to create obstacles
		is_selected = (pts_C.col(2).array() > zmin_).cast<int>() * 
		(pts_C.col(2).array() < zmax_).cast<int>();
	}

	return filterPoints(is_selected, pts_C);
}

gr::PointCloud_t DepthToCloudLibRS::operator()(const rs2::frame &processed_depth, bool b_downward_facing) {
	// Convert to point cloud after decimation; get vertices
	pts_ = pc_.calculate(processed_depth);
	const rs2::vertex *v = pts_.get_vertices();
	size_t Npts = pts_.size();

	auto pts_C = Eigen::Map<const gr::PointCloud_t>((float *)v, Npts, 3);
	return filterEigen(pts_C, b_downward_facing);
}

} // namespace ctrl
