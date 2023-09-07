#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4996)

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <omp.h>

// RealSense headers
#include <librealsense2/rs.hpp>

// OpenCV headers
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

// PCL headers
#include <vtkCamera.h>
#include <pcl/surface/mls.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_box.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/features/vfh.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>

// Eigen headers (might be already included with PCL, but just in case)
#define EIGEN_DONT_VECTORIZE 
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#include <Eigen/Dense>

//Boost Headers
#include <boost/thread/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

// Custom Header file
#include "FEC.h"
#include "iterative_closest_point.hpp"



const int NUM_THREADS = std::thread::hardware_concurrency();

struct CameraIntrinsics {
    double fx, fy, cx, cy;
};

float depth_scale = 0.001;

cv::Point clicked_point;
bool is_point_selected = false;

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        clicked_point = cv::Point(x, y);
        is_point_selected = true;
    }
}
boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer(
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud1, 
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud2, 
    float z_offset = 0.05)  // Default offset set to 0.5 units, adjust as needed
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(1, 1, 1);

    // Transform cloud2 by applying an offset in the z-direction
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_transformed(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << 0.0, 0.0, z_offset;
    pcl::transformPointCloud(*cloud2, *cloud2_transformed, transform);

    // Add the first cloud with a red color
    if (cloud1->points.size() > 0) {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(cloud1, 255, 0, 0);
        viewer->addPointCloud<pcl::PointXYZ>(cloud1, red, "cloud1");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud1");
    }

    // Add the transformed second cloud with a blue color
    if (cloud2_transformed->points.size() > 0) {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue(cloud2_transformed, 0, 0, 255);
        viewer->addPointCloud<pcl::PointXYZ>(cloud2_transformed, blue, "cloud2");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud2");
    }

    viewer->initCameraParameters();

    return viewer;
}



// RANSAC-related structures and functions

struct PlaneFitResult {
    Eigen::Vector4f equation;
    std::vector<int> inliers;
};
struct Result {
    Eigen::Vector4f eq;
    std::vector<int> inliers;
};
pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

std::mutex mtx;

void ransac_thread(int iterations, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float thresh, Result& bestResult) {
    std::srand((unsigned)time(0));  // Seed the random number generator
    std::vector<Eigen::Vector3f> pt_samples(3);
    Eigen::Vector3f vecA, vecB, vecC;
    Eigen::MatrixXf points = cloud->getMatrixXfMap(3, 4, 0);  // Moved out of the loop

    for (int it = 0; it < iterations; ++it) {
        for (int s = 0; s < 3; ++s) {
            int idx = std::rand() % cloud->points.size();
            pcl::PointXYZ sampled_point = cloud->points[idx];
            pt_samples[s] = Eigen::Vector3f(sampled_point.x, sampled_point.y, sampled_point.z);  // Direct assignment
        }

        vecA = pt_samples[1] - pt_samples[0];
        vecB = pt_samples[2] - pt_samples[0];
        vecC = vecA.cross(vecB);  // No need to normalize here

        float k = -vecC.dot(pt_samples[1]);
        Eigen::Vector4f plane_eq(vecC[0], vecC[1], vecC[2], k);

        Eigen::Vector3f normal(plane_eq[0], plane_eq[1], plane_eq[2]);
        float d = plane_eq[3];

        Eigen::ArrayXf dists = (points.transpose() * normal).array() + d;
        dists = dists.abs() / normal.norm();

        std::vector<int> inliers;
        inliers.reserve(points.cols());  // Reserve memory

        for (int i = 0; i < dists.size(); ++i) {
            if (dists[i] <= thresh) {
                inliers.push_back(i);
            }
        }

        if (inliers.size() > bestResult.inliers.size()) {
            mtx.lock();
            bestResult.eq = plane_eq;
            bestResult.inliers = inliers;
            mtx.unlock();
        }
    }
}



PlaneFitResult fit(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float thresh = 1e-2, int maxIteration = 150) {
    std::vector<std::thread> threads;
    std::vector<Result> results(NUM_THREADS, {Eigen::Vector4f::Zero(), {}});

    int iterationsPerThread = maxIteration / NUM_THREADS;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(ransac_thread, iterationsPerThread, cloud, thresh, std::ref(results[i]));
    }

    for (auto& t : threads) {
        t.join();
    }

    // Retrieve the best result
    Result best = results[0];
    for (int i = 1; i < NUM_THREADS; ++i) {
        if (results[i].inliers.size() > best.inliers.size()) {
            best = results[i];
        }
    }

    PlaneFitResult result;
    result.equation = best.eq;
    result.inliers = best.inliers;

    return result;
}

// generate cluster color randomly
int* rand_rgb() {
    int* rgb = new int[3];
    rgb[0] = rand() % 255;
    rgb[1] = rand() % 255;
    rgb[2] = rand() % 255;
    return rgb;
}


void loadPLYFile(const char* file_name, pcl::PointCloud<pcl::PointXYZ> &cloud)
{
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(file_name, cloud) == -1)
    {
        PCL_ERROR("Failed to load PLY file.");
        return;
    }

    std::vector<int> index;
    pcl::removeNaNFromPointCloud(cloud, cloud, index);
}

void loadFile(const char* file_name, pcl::PointCloud<pcl::PointXYZ> &cloud)
{
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_name, cloud) == -1)
    {
        PCL_ERROR("Failed to load point cloud.");
        return;
    }

    std::vector<int> index;
    pcl::removeNaNFromPointCloud(cloud, cloud, index);
}


Eigen::Vector4f computeCentroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    return centroid;
}

void moveToOrigin(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Eigen::Vector4f& centroid) {
    for (pcl::PointXYZ &point : cloud->points) {
        point.x -= centroid[0];
        point.y -= centroid[1];
        point.z -= centroid[2];
    }
}
Eigen::Vector4f centroid_source; 
Eigen::Vector4f centroid_target; 



int main() 
{
    // Check if a RealSense device is connected
    rs2::context ctx;
    if (ctx.query_devices().size() == 0) {
        std::cerr << "No RealSense devices were found!" << std::endl;
        return -1;
    }

    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

    rs2::pipeline_profile profile = pipe.start(cfg);

    // Retrieve the intrinsic parameters for the depth stream
    rs2::video_stream_profile depth_stream_profile = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    const rs2_intrinsics& intrinsics_struct = depth_stream_profile.get_intrinsics();

    CameraIntrinsics intrinsics;
    intrinsics.fx = intrinsics_struct.fx;
    intrinsics.fy = intrinsics_struct.fy;
    intrinsics.cx = intrinsics_struct.ppx;
    intrinsics.cy = intrinsics_struct.ppy;

    rs2::align align_to(RS2_STREAM_COLOR);

    rs2::spatial_filter spat_filter;
    rs2::temporal_filter temp_filter;
    rs2::disparity_transform depth_to_disparity(true);
    rs2::disparity_transform disparity_to_depth(false);
    rs2::hole_filling_filter hole_filling;
    rs2::decimation_filter dec_filter;
    float decimation_magnitude = 1.0f; // You can adjust this value as needed
    dec_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, decimation_magnitude);


    cv::namedWindow("Color Image", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Color Image", onMouse, NULL);

    cv::Mat depth_img;  // Declare depth_img here
    cv::Mat color_img;  // Declare color_img here

    // Declare the pointcloud object and points object
    rs2::pointcloud pc;
    rs2::points points;

   while (!is_point_selected) {
    rs2::frameset data = pipe.wait_for_frames();
    rs2::frameset aligned_data = align_to.process(data);
     // Start the timer before applying the decimation filter.
    auto start_time = std::chrono::high_resolution_clock::now();

    // Apply the decimation filter
    rs2::frame depth_decimated = dec_filter.process(aligned_data.get_depth_frame());

    // Stop the timer after the decimation filter has been applied.
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "Time taken by decimation filter: " << elapsed_time << " microseconds" << std::endl;
    
    rs2::frame depth_filtered = depth_decimated;
    depth_filtered = depth_to_disparity.process(depth_filtered);
    // Note: Removing the duplicate depth frame assignment here
    depth_filtered = spat_filter.process(depth_filtered);
    depth_filtered = temp_filter.process(depth_filtered);
    depth_filtered = disparity_to_depth.process(depth_filtered);
    depth_filtered = hole_filling.process(depth_filtered);

    points = pc.calculate(depth_filtered);

    rs2::video_frame color = aligned_data.get_color_frame();
    color_img = cv::Mat(cv::Size(color.get_width(), color.get_height()), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
    color_img.convertTo(color_img, -1, 1, 0); 
    cv::imshow("Color Image", color_img);        
    depth_img = cv::Mat(cv::Size(depth_filtered.as<rs2::depth_frame>().get_width(), depth_filtered.as<rs2::depth_frame>().get_height()), CV_16UC1, (void*)depth_filtered.get_data(), cv::Mat::AUTO_STEP);

    if (cv::waitKey(1) == 27) {
        break;
    }
    }

    if (is_point_selected) {
    auto start1 = std::chrono::high_resolution_clock::now();

    std::cout << "Selected point on color image: (" << clicked_point.x << ", " << clicked_point.y << ")" << std::endl;
    
    // Convert the RealSense SDK point cloud to PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    auto vertices = points.get_vertices();
    for (int i = 0; i < points.size(); ++i) {
        if (vertices[i].z) {  // Check if the depth is valid
            cloud->push_back(pcl::PointXYZ(vertices[i].x, vertices[i].y, vertices[i].z));
        }
    }
    std::cout << "Number of points in the point cloud: " << cloud->size() << std::endl;

    // Convert the clicked point to a 3D point
    float clicked_z = depth_img.at<ushort>(clicked_point.y, clicked_point.x) * depth_scale;
    float clicked_x = (clicked_point.x - intrinsics.cx) * clicked_z / intrinsics.fx;
    float clicked_y = (clicked_point.y - intrinsics.cy) * clicked_z / intrinsics.fy;
    pcl::PointXYZ clicked_3d_point(clicked_x, clicked_y, clicked_z);

    // Extract points inside the cubic ROI
    double half_side_length = 0.25;  // 25 cm for half the side, adjust as needed
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_roi(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setInputCloud(cloud);
    crop.setMin(Eigen::Vector4f(clicked_3d_point.x - half_side_length, clicked_3d_point.y - half_side_length, clicked_3d_point.z - half_side_length, 1.0));
    crop.setMax(Eigen::Vector4f(clicked_3d_point.x + half_side_length, clicked_3d_point.y + half_side_length, clicked_3d_point.z + half_side_length, 1.0));
    crop.filter(*cloud_roi);

    // Remove ROI from the original point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_minus_roi(new pcl::PointCloud<pcl::PointXYZ>);
    crop.setNegative(true);  // Extract points outside the box
    crop.filter(*cloud_minus_roi);

    // Apply Voxel Grid filter on the point cloud outside the ROI
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud_minus_roi);
    voxel_grid.setLeafSize(0.01f, 0.01f, 0.01f);  // Adjust this value as needed
    voxel_grid.filter(*cloud_downsampled);

    // Combine the downsampled point cloud with the untouched ROI
    *cloud_downsampled += *cloud_roi;

    std::cout << "Number of points in the point cloud after Voxel Grid filter: " << cloud_downsampled->size() << std::endl;
    
    auto start_time2 = std::chrono::high_resolution_clock::now();

    // Create a PassThrough filter instance
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud_downsampled);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(clicked_z - 0.30, clicked_z + 0.30); // PassThrough filter between the clicked_z (converted to meters) and clicked_z + 0.10 m.
        pass.filter(*cloud_downsampled);

    auto end_time2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2);
    std::cout << "[Pass Through Filter]: " << duration2.count() << "ms" << std::endl;
    
    std::cout << "Number of points in the point cloud after Pass Through filter: " << cloud_downsampled->size() << std::endl;

    auto start_time3 = std::chrono::high_resolution_clock::now();

    PlaneFitResult result = fit(cloud_downsampled, 0.006, 800);
    Eigen::Vector4f bestPlane = result.equation;

    auto end_time3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time3 - start_time3);
    std::cout << "[Ransac Plane Segmentation]: " << duration3.count() << "ms" << std::endl;

    // Create a pcl::PointIndices from the inliers
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices());
    inliers_plane->indices = result.inliers;

    // Use pcl::ExtractIndices to remove the inliers from the cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_downsampled);
    extract.setIndices(inliers_plane);
    extract.setNegative(true);  // true means we want to get the points that are NOT in the indices list
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_without_inliers(new pcl::PointCloud<pcl::PointXYZ>);
    extract.filter(*cloud_without_inliers);

    // Save the point cloud with Pass through and Voxel grid filter
    pcl::io::savePCDFileASCII("cloud_downsampled.pcd", *cloud_downsampled);
    std::cout << "Saved " << cloud_downsampled->points.size() << " data points to cloud_downsampled.pcd." << std::endl;
    std::cout <<"hello"<<endl;

    // Save the point cloud without the inliers
    pcl::io::savePCDFileASCII("cloud_without_inliers.pcd", *cloud_without_inliers);
    std::cout << "Saved " << cloud_without_inliers->points.size() << " data points to cloud_without_inliers.pcd." << std::endl;

    // Perform Fast Euclidean Clustering on filtered_cloud
    std::vector<pcl::PointIndices> cluster_indices = FEC(cloud_without_inliers, 20, 0.015, 50);

    // Prepare to color the clusters
    std::vector<unsigned char> color;
    for (int i_segment = 0; i_segment < cluster_indices.size(); i_segment++) {
        color.push_back(static_cast<unsigned char>(rand() % 256));
        color.push_back(static_cast<unsigned char>(rand() % 256));
        color.push_back(static_cast<unsigned char>(rand() % 256));
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_point(new pcl::PointCloud<pcl::PointXYZRGB>());

    #pragma omp parallel for
    for (size_t i = 0; i < cluster_indices.size(); i++) {
        std::vector<pcl::PointXYZRGB> local_points;
        for (size_t j = 0; j < cluster_indices[i].indices.size(); j++) {
            pcl::PointXYZRGB point;
            point.x = cloud_without_inliers->points[cluster_indices[i].indices[j]].x;
            point.y = cloud_without_inliers->points[cluster_indices[i].indices[j]].y;
            point.z = cloud_without_inliers->points[cluster_indices[i].indices[j]].z;
            point.r = color[int(3) * i];
            point.g = color[int(3) * i + 1];
            point.b = color[int(3) * i + 2];
            local_points.push_back(point);
        }
        #pragma omp critical
        {
            color_point->insert(color_point->end(), local_points.begin(), local_points.end());
        }
    }

    // Save the colored clusters to a file
    pcl::io::savePCDFileASCII("D:\\CPP_Grasp_Synthesis\\color_clusters.pcd", *color_point);
    std::cout << "Saved " << color_point->points.size() << " data points to color_clusters.pcd." << std::endl;

    // Compute the centroids of all clusters and find the one closest to the clicked point
    std::vector<Eigen::Vector4f> centroids(cluster_indices.size());
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters(cluster_indices.size());

    #pragma omp parallel for
    for (size_t i = 0; i < cluster_indices.size(); i++) {
        clusters[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        for (size_t j = 0; j < cluster_indices[i].indices.size(); j++) {
            clusters[i]->points.push_back(cloud_without_inliers->points[cluster_indices[i].indices[j]]);
        }
        pcl::compute3DCentroid(*clusters[i], centroids[i]);
    }

    // Find the closest centroid to the clicked point
    float min_dist = FLT_MAX;
    int closest_cluster_idx = -1;
    for (size_t i = 0; i < centroids.size(); i++) {
        pcl::PointXYZ p1 = clicked_3d_point;
        pcl::PointXYZ p2 = pcl::PointXYZ(centroids[i][0], centroids[i][1], centroids[i][2]);
        float dist = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));

        if (dist < min_dist) {
            min_dist = dist;
            closest_cluster_idx = i;
        }
    }

    std::cout << "The clicked point belongs to cluster " << closest_cluster_idx << std::endl;

    if (closest_cluster_idx == -1) {
        std::cout << "No cluster is found for the selected point." << std::endl;
        return -1;  // Exit if no closest cluster is found
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr closest_cluster = clusters[closest_cluster_idx];
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_with_clicked_point(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr sor_filtered_cluster(new pcl::PointCloud<pcl::PointXYZ>());

    try {
    // Adjust the dimensions 
    closest_cluster->width = closest_cluster->points.size();
    closest_cluster->height = 1;  // Assuming unorganized point cloud

    std::cout << "Size of selected cluster before Region growing: " << closest_cluster->points.size() << std::endl;

    // Save the transformed point cloud
    pcl::io::savePCDFile("D:\\CPP_Grasp_Synthesis\\transformed_a_new1.pcd", *closest_cluster);
    std::cout << "Saved the transformed point cloud to 'transformed_a_new1.pcd'" << std::endl;

    // Applying the Statistical Outlier Removal filter
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(closest_cluster);
    sor.setMeanK(50); // Number of neighbors to analyze for each point
    sor.setStddevMulThresh(1.0); // Standard deviation multiplier threshold
    sor.filter(*sor_filtered_cluster);

    std::cout << "Size of cluster after SOR: " << sor_filtered_cluster->points.size() << std::endl;

    // Optionally, save the filtered point cloud
    pcl::io::savePCDFile("D:\\CPP_Grasp_Synthesis\\sor_filtered.pcd", *sor_filtered_cluster);
    std::cout << "Saved the filtered point cloud to 'sor_filtered.pcd'" << std::endl;

    } catch (const pcl::PCLException& e) {
        std::cerr << "PCL exception: " << e.what() << std::endl;
    }

    // Apply region growing segmentation on the closest cluster
    pcl::search::Search<pcl::PointXYZ>::Ptr tree1 = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree1);
    normal_estimator.setInputCloud(sor_filtered_cluster);
    normal_estimator.setKSearch(50);
    normal_estimator.compute(*normals);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(50);
    reg.setMaxClusterSize(50000);
    reg.setSearchMethod(tree1);
    reg.setNumberOfNeighbours(30);
    reg.setInputCloud(sor_filtered_cluster);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(1.0);

    std::vector<pcl::PointIndices> clusters_rg;
    reg.extract(clusters_rg);
    int cluster_idx_with_clicked_point = -1;
    float min_distance_to_clicked_point = FLT_MAX;

    for (size_t i = 0; i < clusters_rg.size(); i++) {
        for (int idx : clusters_rg[i].indices) {
            pcl::PointXYZ point = sor_filtered_cluster->points[idx];
            float dist = sqrt(pow(clicked_3d_point.x - point.x, 2) + 
                            pow(clicked_3d_point.y - point.y, 2) + 
                            pow(clicked_3d_point.z - point.z, 2));

            if (dist < min_distance_to_clicked_point) {
                min_distance_to_clicked_point = dist;
                cluster_idx_with_clicked_point = i;
            }
        }
    }

    if (cluster_idx_with_clicked_point != -1) {
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(sor_filtered_cluster);

    pcl::IndicesPtr indices_of_interest(new std::vector<int>(clusters_rg[cluster_idx_with_clicked_point].indices.begin(), clusters_rg[cluster_idx_with_clicked_point].indices.end()));
    extract.setIndices(indices_of_interest);

    extract.filter(*cluster_with_clicked_point);
    
    std::cout << "The cluster containing (or closest to) the clicked point has " << cluster_with_clicked_point->points.size() << " points." << std::endl;

    }
    if (cluster_idx_with_clicked_point != -1) {
        std::cout << "The closest region-growing cluster has " << cluster_with_clicked_point->points.size() << " points." << std::endl;
        pcl::io::savePCDFile("D:\\CPP_Grasp_Synthesis\\region_growing_closest.pcd", *cluster_with_clicked_point);
        std::cout << "Saved the closest region-growing cluster to 'region_growing_closest.pcd'" << std::endl;
    } else {
        std::cout << "No closest region-growing cluster is found." << std::endl;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr source (new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target (new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    Eigen::Matrix4f T;  

    {
    std::string original = "D:\\CPP_Grasp_Synthesis\\build\\Release\\cloud_downsampled.pcd";
    std::string target_path = "D:\\CPP_Grasp_Synthesis\\region_growing_closest.pcd";
    std::string source_path = "D:\\CPP_Grasp_Synthesis\\models\\obj_000005.ply";

    loadPLYFile(source_path.c_str(), *source);
    loadFile(target_path.c_str(), *cloud_target);
    loadFile(original.c_str(), *raw_cloud);


    // Convert D435i's point cloud (source) from mm to meters 
    float conversion_factor = 0.001;  // Conversion factor from mm to m

    for (pcl::PointXYZ &point : source->points) {
        point.x *= conversion_factor;
        point.y *= conversion_factor;
        point.z *= conversion_factor;
    }

    // Compute centroids
    centroid_source = computeCentroid(source);
    centroid_target = computeCentroid(cloud_target);

    // Move point clouds to origin
    moveToOrigin(source, centroid_source);
    moveToOrigin(cloud_target, centroid_target);

    // Compute normals for both point clouds
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr source_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr target_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.03);

    ne.setInputCloud(source);
    ne.compute(*source_normals);

    ne.setInputCloud(cloud_target);
    ne.compute(*target_normals);
    std::cout<< "Calculating VFH for Source and Target point clouds for Rough Alignment"<<endl;
    
    // Compute VFH descriptors for both point clouds
    pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
    pcl::PointCloud<pcl::VFHSignature308>::Ptr source_vfh (new pcl::PointCloud<pcl::VFHSignature308>);
    pcl::PointCloud<pcl::VFHSignature308>::Ptr target_vfh (new pcl::PointCloud<pcl::VFHSignature308>);
    vfh.setInputCloud(source);
    vfh.setInputNormals(source_normals);
    vfh.setSearchMethod(tree);
    vfh.compute(*source_vfh);

    vfh.setInputCloud(cloud_target);
    vfh.setInputNormals(target_normals);
    vfh.compute(*target_vfh);

    // Match the VFH descriptors
    pcl::KdTreeFLANN<pcl::VFHSignature308> match_search;
    match_search.setInputCloud(target_vfh);
    std::vector<int> match_idx(1);
    std::vector<float> match_dist(1);
    match_search.nearestKSearch(*source_vfh, 0, 1, match_idx, match_dist);

    // Use ICP to refine the pose estimation
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source);
    icp.setInputTarget(cloud_target);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);

    Eigen::Matrix4f transformation = icp.getFinalTransformation();
    Eigen::Matrix4f inverse_transformation = transformation.inverse();


    // Create a copy of the source point cloud
    pcl::transformPointCloud(*source, *cloud_source, inverse_transformation);


}

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans (new pcl::PointCloud<pcl::PointXYZ>());

{
    Eigen::MatrixXf source_matrix = cloud_source->getMatrixXfMap(3,4,0).transpose();
    Eigen::MatrixXf target_matrix = cloud_target->getMatrixXfMap(3,4,0).transpose();

    int max_iteration = 500;
    float tolenrance = 0.000001;

    // call icp
    ICP_OUT icp_result = icp(source_matrix.cast<double>(), target_matrix.cast<double>(), max_iteration, tolenrance);

    int iter = icp_result.iter;
    T = icp_result.trans.cast<float>();
    std::vector<float> distances = icp_result.distances;

    Eigen::MatrixXf source_trans_matrix = source_matrix;

    int row = source_matrix.rows();

    // Validate matrix dimensions and indices
    std::cout << "source_matrix size: " << source_matrix.rows() << "x" << source_matrix.cols() << std::endl;

    Eigen::MatrixXf source_trans4d = Eigen::MatrixXf::Ones(4, row);

    for (int i = 0; i < row; i++)
    {
        // std::cout << "Current i: " << i << std::endl; // Debug current index
        source_trans4d.block<3,1>(0,i) = source_matrix.block<1,3>(i,0).transpose();
    }

    source_trans4d = T * source_trans4d;

    for (int i = 0; i < row; i++)
    {
        source_trans_matrix.block<1,3>(i,0) = source_trans4d.block<3,1>(0,i).transpose();
    }

    pcl::PointCloud<pcl::PointXYZ> temp_cloud;
    temp_cloud.width = row;
    temp_cloud.height = 1;
    temp_cloud.points.resize(row);

    for (size_t n = 0; n < row; n++) 
    {
        temp_cloud[n].x = source_trans_matrix(n,0);
        temp_cloud[n].y = source_trans_matrix(n,1);
        temp_cloud[n].z = source_trans_matrix(n,2);	
    }

    cloud_source_trans = temp_cloud.makeShared();

        // Adjust the transformed source cloud by adding the centroid of the target cloud
    for (pcl::PointXYZ &point : cloud_source_trans->points) {
    point.x += centroid_target[0];
    point.y += centroid_target[1];
    point.z += centroid_target[2];
    }

    for (pcl::PointXYZ &point : cloud_target->points) {
    point.x += centroid_target[0];
    point.y += centroid_target[1];
    point.z += centroid_target[2];
    }
    
}

{ // visualization
    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);

    // // red
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(source, 0, 255, 0);
    // viewer->addPointCloud<pcl::PointXYZ>(source, source_color, "source");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source");

    // green
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(raw_cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(raw_cloud, source_color, "original");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original");


    // blue
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(cloud_target, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_target, target_color, "target");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target");

    // red
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_trans_color(cloud_source_trans, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_source_trans, source_trans_color, "source trans");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source trans");

    // Add a coordinate system at the centroid of source point cloud
    Eigen::Vector4f centroid_source;
    pcl::compute3DCentroid(*source, centroid_source);
    Eigen::Affine3f transform_source = Eigen::Affine3f::Identity();
    transform_source.translation() << centroid_source[0], centroid_source[1], centroid_source[2];
    // viewer->addCoordinateSystem(0.1, transform_source, "coordinate system source", 0);
    
    // Add a coordinate system at the centroid of target point cloud
    Eigen::Vector4f centroid_target;
    pcl::compute3DCentroid(*cloud_target, centroid_target);
    Eigen::Affine3f transform_target = Eigen::Affine3f::Identity();
    transform_target.translation() << centroid_target[0], centroid_target[1], centroid_target[2];
    viewer->addCoordinateSystem(0.1, transform_target, "coordinate system target", 0);
    
    // Add a coordinate system at the centroid of source to visualize the orientation change
    Eigen::Affine3f transform_orientation_change = Eigen::Affine3f(T); // Convert T to an Affine transformation
    transform_orientation_change.translation() << centroid_source[0], centroid_source[1], centroid_source[2]; // Use the original centroid of the source cloud
    // viewer->addCoordinateSystem(0.1, transform_orientation_change, "coordinate system orientation change", 0);

    viewer->getRenderWindow()->GetRenderers()->GetFirstRenderer()->GetActiveCamera()->SetParallelProjection(1);
    viewer->resetCamera();
    viewer->spin();
}      
   
}
cv::destroyAllWindows();
pipe.stop();
return 0;
}