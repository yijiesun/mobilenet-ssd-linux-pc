# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/syj/work/Tengine/examples/mobilenet_ssd

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/syj/work/Tengine/examples/mobilenet_ssd/build

# Include any dependencies generated for this target.
include CMakeFiles/CAPTRUE.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CAPTRUE.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CAPTRUE.dir/flags.make

CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.o: CMakeFiles/CAPTRUE.dir/flags.make
CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.o: ../src/mssd_cvCaptrue.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/syj/work/Tengine/examples/mobilenet_ssd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.o -c /home/syj/work/Tengine/examples/mobilenet_ssd/src/mssd_cvCaptrue.cpp

CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/syj/work/Tengine/examples/mobilenet_ssd/src/mssd_cvCaptrue.cpp > CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.i

CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/syj/work/Tengine/examples/mobilenet_ssd/src/mssd_cvCaptrue.cpp -o CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.s

CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.o.requires:

.PHONY : CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.o.requires

CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.o.provides: CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.o.requires
	$(MAKE) -f CMakeFiles/CAPTRUE.dir/build.make CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.o.provides.build
.PHONY : CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.o.provides

CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.o.provides.build: CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.o


CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.o: CMakeFiles/CAPTRUE.dir/flags.make
CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.o: /home/syj/work/Tengine/examples/common/common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/syj/work/Tengine/examples/mobilenet_ssd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.o -c /home/syj/work/Tengine/examples/common/common.cpp

CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/syj/work/Tengine/examples/common/common.cpp > CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.i

CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/syj/work/Tengine/examples/common/common.cpp -o CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.s

CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.o.requires:

.PHONY : CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.o.requires

CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.o.provides: CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.o.requires
	$(MAKE) -f CMakeFiles/CAPTRUE.dir/build.make CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.o.provides.build
.PHONY : CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.o.provides

CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.o.provides.build: CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.o


# Object files for target CAPTRUE
CAPTRUE_OBJECTS = \
"CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.o" \
"CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.o"

# External object files for target CAPTRUE
CAPTRUE_EXTERNAL_OBJECTS =

CAPTRUE: CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.o
CAPTRUE: CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.o
CAPTRUE: CMakeFiles/CAPTRUE.dir/build.make
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
CAPTRUE: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
CAPTRUE: CMakeFiles/CAPTRUE.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/syj/work/Tengine/examples/mobilenet_ssd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable CAPTRUE"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CAPTRUE.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CAPTRUE.dir/build: CAPTRUE

.PHONY : CMakeFiles/CAPTRUE.dir/build

CMakeFiles/CAPTRUE.dir/requires: CMakeFiles/CAPTRUE.dir/src/mssd_cvCaptrue.cpp.o.requires
CMakeFiles/CAPTRUE.dir/requires: CMakeFiles/CAPTRUE.dir/home/syj/work/Tengine/examples/common/common.cpp.o.requires

.PHONY : CMakeFiles/CAPTRUE.dir/requires

CMakeFiles/CAPTRUE.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CAPTRUE.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CAPTRUE.dir/clean

CMakeFiles/CAPTRUE.dir/depend:
	cd /home/syj/work/Tengine/examples/mobilenet_ssd/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/syj/work/Tengine/examples/mobilenet_ssd /home/syj/work/Tengine/examples/mobilenet_ssd /home/syj/work/Tengine/examples/mobilenet_ssd/build /home/syj/work/Tengine/examples/mobilenet_ssd/build /home/syj/work/Tengine/examples/mobilenet_ssd/build/CMakeFiles/CAPTRUE.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CAPTRUE.dir/depend

