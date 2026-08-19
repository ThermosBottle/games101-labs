cmake_minimum_required(VERSION 3.10)
project(Rasterizer)

set(OpenCV_DIR "D:/opencv/build/mingw64-build")
find_package(OpenCV REQUIRED)

include_directories("D:\\Program Files\\Eigen3\\include")
set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include ./include)

add_executable(Rasterizer main.cpp rasterizer.hpp rasterizer.cpp global.hpp Triangle.hpp Triangle.cpp Texture.hpp Texture.cpp Shader.hpp OBJ_Loader.h)
target_link_libraries(Rasterizer ${OpenCV_LIBRARIES})
#target_compile_options(Rasterizer PUBLIC -Wall -Wextra -pedantic)