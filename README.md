# Model-Assisted-6D-Pose-Estimation-via-VFH-and-Point-to-Point-ICP

## Overview

- This Project is a comprehensive C++ application designed to capture, process, and analyze 3D point cloud data. 
- Leveraging the power of the RealSense camera, OpenCV for image processing, and the Point Cloud Library (PCL) for 3D data manipulation, this project offers real-time interaction, object identification, and tracking in a 3D space.
- Designed for moderately cluttered environments, this approach extends the principles of Method-1. It incorporates the Viewpoint Feature Histogram (VFH) and Point-to-Point Iterative Closest Point (ICP) algorithms for detailed pose estimation. 
- Object segmentation from the environment is achieved using RANSAC-based plane detection, fast Euclidean clustering, and Region Growing methods. This ensures accurate pose estimation even when objects are spaced as close as 3 cm apart.


## Features

- **RealSense Camera Integration**: Captures depth and color data in real-time.
- **Point Cloud Visualization**: Visualizes 3D point cloud data.
- **Advanced Filtering**: Uses spatial, temporal, disparity, hole-filling, and decimation filters to enhance data quality.
- **User Interaction**: Allows users to select points on the color image to focus on specific regions.
- **Segmentation**: Employs voxel grid filtering, pass-through filtering, and RANSAC plane fitting for point cloud segmentation.
- **Data Processing**: Implements statistical outlier removal and region-growing segmentation to refine point cloud data.
- **Alignment with ICP**: Aligns captured point cloud with a reference dataset using the Iterative Closest Point (ICP) algorithm.

## Prerequisites

- CMake (version 3.10 or higher)
- C++17 compiler
- RealSense SDK 2.0
- OpenCV (with core, imgproc, and highgui components)
- Point Cloud Library (PCL)
- Eigen3 (version 3.3 or higher)
- RYML
- Loguru

## Dependencies
- **C++17**: The project uses C++17 features.
- **Eigen3**: Required for matrix and linear algebra operations.
- **ryml**: YAML parser library.
- **PCL (Point Cloud Library)**: Used for point cloud processing.
- **Intel RealSense SDK**: For interfacing with Intel RealSense cameras.
- **OpenCV**: Used for image processing tasks.
- **loguru**: Logging library (ensure the library is in the specified path).
- **Threads**: For multithreading capabilities.

## Installation
```bash
# Clone the repository
git clone https://github.com/Parthsanghavi31/Model-Assisted-6D-Pose-Estimation-via-VFH-and-Point-to-Point-ICP-.git

# Navigate to the project directory
cd Model-Assisted-6D-Pose-Estimation-via-VFH-and-Point-to-Point-ICP-

# Create a build directory and navigate to it
mkdir build && cd build

# Compile the project
cmake ..
make
```
## Usage

```bash
# Run the main executable
./grasp_synthesis
```

- Use the GUI to select a point of interest in the real-time depth image. The program will process the depth data, detect and segment objects, and display the results in a 3D visualization window.

## Results
- All the Results can be Found in the Report on Grasp Synthesis Project

## Contributing
- Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
- This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
- Ghost Robotics for providing the opportunity to work on this innovative project.
- Intel for the RealSense technology.
- The open-source community for the libraries and tools.