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
include CMakeFiles/VIDEO_KNN.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/VIDEO_KNN.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/VIDEO_KNN.dir/flags.make

CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.o: CMakeFiles/VIDEO_KNN.dir/flags.make
CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.o: ../src/mssd_video_knn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/syj/work/Tengine/examples/mobilenet_ssd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.o -c /home/syj/work/Tengine/examples/mobilenet_ssd/src/mssd_video_knn.cpp

CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/syj/work/Tengine/examples/mobilenet_ssd/src/mssd_video_knn.cpp > CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.i

CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/syj/work/Tengine/examples/mobilenet_ssd/src/mssd_video_knn.cpp -o CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.s

CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.o.requires:

.PHONY : CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.o.requires

CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.o.provides: CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.o.requires
	$(MAKE) -f CMakeFiles/VIDEO_KNN.dir/build.make CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.o.provides.build
.PHONY : CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.o.provides

CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.o.provides.build: CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.o


CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.o: CMakeFiles/VIDEO_KNN.dir/flags.make
CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.o: ../src/knn/knn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/syj/work/Tengine/examples/mobilenet_ssd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.o -c /home/syj/work/Tengine/examples/mobilenet_ssd/src/knn/knn.cpp

CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/syj/work/Tengine/examples/mobilenet_ssd/src/knn/knn.cpp > CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.i

CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/syj/work/Tengine/examples/mobilenet_ssd/src/knn/knn.cpp -o CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.s

CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.o.requires:

.PHONY : CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.o.requires

CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.o.provides: CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.o.requires
	$(MAKE) -f CMakeFiles/VIDEO_KNN.dir/build.make CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.o.provides.build
.PHONY : CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.o.provides

CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.o.provides.build: CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.o


CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.o: CMakeFiles/VIDEO_KNN.dir/flags.make
CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.o: /home/syj/work/Tengine/examples/common/common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/syj/work/Tengine/examples/mobilenet_ssd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.o -c /home/syj/work/Tengine/examples/common/common.cpp

CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/syj/work/Tengine/examples/common/common.cpp > CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.i

CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/syj/work/Tengine/examples/common/common.cpp -o CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.s

CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.o.requires:

.PHONY : CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.o.requires

CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.o.provides: CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.o.requires
	$(MAKE) -f CMakeFiles/VIDEO_KNN.dir/build.make CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.o.provides.build
.PHONY : CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.o.provides

CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.o.provides.build: CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.o


# Object files for target VIDEO_KNN
VIDEO_KNN_OBJECTS = \
"CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.o" \
"CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.o" \
"CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.o"

# External object files for target VIDEO_KNN
VIDEO_KNN_EXTERNAL_OBJECTS =

VIDEO_KNN: CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.o
VIDEO_KNN: CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.o
VIDEO_KNN: CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.o
VIDEO_KNN: CMakeFiles/VIDEO_KNN.dir/build.make
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
VIDEO_KNN: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
VIDEO_KNN: CMakeFiles/VIDEO_KNN.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/syj/work/Tengine/examples/mobilenet_ssd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable VIDEO_KNN"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/VIDEO_KNN.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/VIDEO_KNN.dir/build: VIDEO_KNN

.PHONY : CMakeFiles/VIDEO_KNN.dir/build

CMakeFiles/VIDEO_KNN.dir/requires: CMakeFiles/VIDEO_KNN.dir/src/mssd_video_knn.cpp.o.requires
CMakeFiles/VIDEO_KNN.dir/requires: CMakeFiles/VIDEO_KNN.dir/src/knn/knn.cpp.o.requires
CMakeFiles/VIDEO_KNN.dir/requires: CMakeFiles/VIDEO_KNN.dir/home/syj/work/Tengine/examples/common/common.cpp.o.requires

.PHONY : CMakeFiles/VIDEO_KNN.dir/requires

CMakeFiles/VIDEO_KNN.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/VIDEO_KNN.dir/cmake_clean.cmake
.PHONY : CMakeFiles/VIDEO_KNN.dir/clean

CMakeFiles/VIDEO_KNN.dir/depend:
	cd /home/syj/work/Tengine/examples/mobilenet_ssd/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/syj/work/Tengine/examples/mobilenet_ssd /home/syj/work/Tengine/examples/mobilenet_ssd /home/syj/work/Tengine/examples/mobilenet_ssd/build /home/syj/work/Tengine/examples/mobilenet_ssd/build /home/syj/work/Tengine/examples/mobilenet_ssd/build/CMakeFiles/VIDEO_KNN.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/VIDEO_KNN.dir/depend
