cmake_minimum_required(VERSION 3.10)
project(RayTracing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(Threads REQUIRED)

add_executable(RayTracing main.cpp Object.hpp Vector.cpp Vector.hpp Sphere.hpp global.hpp Triangle.hpp Scene.cpp
        Scene.hpp Light.hpp AreaLight.hpp BVH.cpp BVH.hpp Bounds3.hpp Ray.hpp Material.hpp Intersection.hpp
        Renderer.cpp Renderer.hpp
        CudaTypes.cuh CudaMath.cuh CudaKernels.cuh CudaKernels.cu
        CudaScene.hpp CudaScene.cpp CudaScene.cu)
set_target_properties(RayTracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(RayTracing PRIVATE Threads::Threads)

add_executable(CudaValidation CudaValidation.cu CudaKernels.cu CudaTypes.cuh CudaMath.cuh CudaKernels.cuh)
set_target_properties(CudaValidation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(CudaPathTracing CudaPathTracing.cu CudaScene.cpp CudaScene.cu
        CudaKernels.cu Scene.cpp BVH.cpp Vector.cpp
        CudaTypes.cuh CudaMath.cuh CudaKernels.cuh CudaScene.hpp)
set_target_properties(CudaPathTracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
