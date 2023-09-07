
#define LOGURU_WITH_STREAMS 1
#include <loguru.hpp>

#include <opencv2/highgui.hpp>

#include "camera_task.hpp"
#include "config_file.hpp"
#include <sstream>
#include <iomanip>

using namespace ctrl;
using namespace std::chrono_literals;

void onMouse(int event, int x, int y, int, void* user_ptr) {
	CameraTask *cam_task = (CameraTask *)user_ptr;
	if (event == cv::EVENT_LBUTTONUP) {
		// LOG_F(INFO, "event=%d, x=%d, y=%d", event, x, y);
		cam_task->userInput(x, y);
	}
}

int main(int argc, char **argv) {
	const cv::String prog_name = "grasp_synthesis";
	loguru::set_thread_name("main");

	if (argc <= 1) {
		LOG_F(ERROR, "Pass path to config.yaml");
		return -1;
	}

	Config_t cfg;
	cfg.load(argv[1]);
	CameraTask cam_task;

	if (!cam_task.init(cfg.serial)) {
		return -1; // quit
	}
	// cam.start();

	cv::namedWindow(prog_name, cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback(prog_name, onMouse, &cam_task);

	int print_ctr = 0;
	bool keep_running = true;

	while (keep_running) {
		cam_task.sampleNewCamData();

		// Run any algorithms here
		cam_task.update();

		cv::imshow(prog_name, cam_task.outputImage());

		// std::this_thread::sleep_for(100ms);
		print_ctr++;
		if (print_ctr % 10 == 0)
			LOG_S(INFO) << std::fixed << std::setprecision(2) <<
				"running... rate=" << cam_task.avgRate() << 
				" rgb_rate=" << cam_task.avgRateRGB() << 
				" cloud_size=" << cam_task.pts_C_->size();

		// Process events
		cv::waitKey(1);
		if (cv::getWindowProperty(prog_name, cv::WND_PROP_VISIBLE) < 0.5)
			keep_running = false;
	}

	cam_task.cleanup();

	// Destroy all windows
	cv::destroyAllWindows();
	LOG_F(INFO, "Exiting");
	return 0;
}
