
cmake_minimum_required (VERSION 2.8)
project(MSSD)
add_definitions(-std=c++11)
set( TENGINE_DIR /home/syj/work/Tengine) 
set( INSTALL_DIR ${TENGINE_DIR}/install/)
set( TENGINE_LIBS tengine hclcpu)

if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
    set( PROTOBUF_LIB ${PROTOBUF_DIR}/arm32_lib/libprotobuf.so)
endif()
if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
    set( PROTOBUF_LIB ${PROTOBUF_DIR}/arm64_lib/libprotobuf.so)
endif()

if( BLAS_DIR)
    if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
         set( BLAS_LIB ${BLAS_DIR}/arm32/lib/libopenblas.so)
    endif()
    if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
         set( BLAS_LIB ${BLAS_DIR}/arm64/lib/libopenblas.so)
    endif()
endif()

if ( NOT (ARM OR ANDROID))
set (BLAS_LIB openblas)
endif()

set( CAPTRUE_SRCS src/mssd_cvCaptrue.cpp src/config.cpp ../common/common.cpp)
set( IMG_SRCS src/mssd_img.cpp src/config.cpp ../common/common.cpp)
set( VIDEO_KNN_SRCS src/mssd_video_knn.cpp src/config.cpp src/knn/knn.cpp ../common/common.cpp)
set( VIDEO_SRCS src/mssd_video.cpp src/knn/knn.cpp src/config.cpp ../common/common.cpp)
#opencv
find_package(OpenCV REQUIRED)

#include
include_directories(${INSTALL_DIR}/include 
                    ${TENGINE_DIR}/examples/common )

#lib
link_directories( ${INSTALL_DIR}/lib)

#exe
add_executable(CAPTRUE ${CAPTRUE_SRCS})
add_executable(IMG ${IMG_SRCS})
add_executable(VIDEO_KNN ${VIDEO_KNN_SRCS})
add_executable(VIDEO ${VIDEO_SRCS})
if( ANDROID)
   target_link_libraries(MSSD ${TENGINE_LIBS} ${PROTOBUF_LIB} ${OpenCV_LIBS} ${BLAS_LIB})
else()
   target_link_libraries(CAPTRUE ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
   target_link_libraries(IMG ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
   target_link_libraries(VIDEO_KNN ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
   target_link_libraries(VIDEO ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
endif()


