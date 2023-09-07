/**
 * Copyright (C) Ghost Robotics - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Avik De <avik@ghostrobotics.io>
 */
#pragma once

#include <ryml_std.hpp>
#include <ryml.hpp>
#include <string>

class Config_t {
public:
	// Config data
	std::string serial; // realsense serial

	bool load(const std::string& yaml_filename);

private:
	ryml::Tree tree_;
};
