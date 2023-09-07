#include "config_file.hpp"
#include <fstream>
#include <sstream>
#include <loguru.hpp>

bool Config_t::load(const std::string& yaml_filename) {
	// Open the config file
	std::ifstream yaml_cfg(yaml_filename);
	
	if (!yaml_cfg.good()) {
		LOG_F(ERROR, "Pass path to config.yaml");
		return false;
	}
	std::ostringstream contents;
	contents << yaml_cfg.rdbuf();
	tree_ = ryml::parse_in_arena(ryml::to_csubstr(contents.str()));
	c4::yml::ConstNodeRef root = tree_.rootref();

	if (root.has_child("serial")) {
		root["serial"] >> serial;
	}

	return true;
}
