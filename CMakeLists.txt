cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_CXX_COMPILER "g++")
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread" )

project(StereoCalibration)
add_executable(StereoCalibration main.cpp)

include_directories(include)
include_directories("E:/jsoncpp-1.8.4/jsoncpp-install/include")
target_link_libraries(StereoCalibration E:/jsoncpp-1.8.4/jsoncpp-install/lib/jsoncpp.lib)

include_directories("E:/boost_1_65_1")
target_link_libraries(StereoCalibration E:/boost_1_65_1/install_x64/boost_filesystem-vc140-mt-1_65_1.lib 
                            E:/boost_1_65_1/install_x64/libboost_filesystem-vc140-mt-1_65_1.lib
                            E:/boost_1_65_1/install_x64/libboost_system-vc140-mt-1_65_1.lib
                            E:/boost_1_65_1/install_x64/boost_system-vc140-mt-1_65_1.lib)

include_directories("E:/ceres/install/eigen/eigen3")

# //opencv
set(OpenCV_INCLUDE_DIR "E:/opencv-3.4.1/install/include")
set(OpenCV_DIR "E:/opencv-3.4.1/build")
include_directories(${OpenCV_INCLUDE_DIR})
find_package(OpenCV REQUIRED)
target_link_libraries(StereoCalibration ${OpenCV_LIBS})

# //open3d
include_directories("E:/Open3D-0.8_vs2015-Release-X64-master/build/3rdparty_install/include" 
                    "E:/Open3D-0.8_vs2015-Release-X64-master/src" 
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty" 
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/dirent"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/dirent/dirent"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/Eigen"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/flann"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/flann/algorithms"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/flann/flann"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/flann/flann/algorithms"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/flann/flann/nn"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/flann/flann/util"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/flann/nn"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/flann/util"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/glew/include"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/GLFW/include"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/jsoncpp/include"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/liblzf"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/liblzf/liblzf"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/tomasakeninemoeller/include"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/tomasakeninemoeller/include/tomasakeninemoeller"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/libpng"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/rply"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/rply/etc"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/rply/manual"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/rply/rply"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/rply/rply/etc"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/rply/rply/manual"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/tinyfiledialogs/include"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/tinygltf"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/tinyobjloader"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/qhull/src"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/fmt/include"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/PoissonRecon"
                    "E:/Open3D-0.8_vs2015-Release-X64-master/3rdparty/librealsense/include")
target_link_libraries(StereoCalibration E:/Open3D-0.8_vs2015-Release-X64-master/build/lib/Release/Open3D.lib
                                        E:/Open3D-0.8_vs2015-Release-X64-master/build/lib/Release/glew.lib
                                        E:/Open3D-0.8_vs2015-Release-X64-master/build/lib/Release/glfw3.lib
                                        E:/Open3D-0.8_vs2015-Release-X64-master/build/lib/Release/jsoncpp.lib
                                        E:/Open3D-0.8_vs2015-Release-X64-master/build/lib/Release/png.lib
                                        E:/Open3D-0.8_vs2015-Release-X64-master/build/lib/Release/zlib.lib
                                        E:/Open3D-0.8_vs2015-Release-X64-master/build/lib/Release/tinyfiledialogs.lib
                                        E:/Open3D-0.8_vs2015-Release-X64-master/build/lib/Release/tinyobjloader.lib
                                        E:/Open3D-0.8_vs2015-Release-X64-master/build/lib/Release/qhullcpp.lib
                                        E:/Open3D-0.8_vs2015-Release-X64-master/build/lib/Release/qhullstatic_r.lib
                                        E:/Open3D-0.8_vs2015-Release-X64-master/build/3rdparty_install/lib/jpeg-static.lib
                                        E:/Open3D-0.8_vs2015-Release-X64-master/build/3rdparty_install/lib/turbojpeg-static.lib
                                        opengl32.lib
                                        glu32.lib)

# //ceres
include_directories("E:/ceres/install/ceres/include" 
                    "E:/ceres/install/eigen/eigen3"
                    "E:/ceres/install/eigen" 
                    "E:/ceres/install/glog/include" 
                    "E:/ceres/install/gflags/include")
target_link_libraries(StereoCalibration E:/ceres/install/gflags/lib/gflags.lib 
                                    E:/ceres/install/glog/lib/glog.lib 
                                    E:/ceres/install/ceres/lib/ceres.lib)

find_package(OpenMP REQUIRED)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")


