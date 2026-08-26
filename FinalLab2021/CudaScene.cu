#include "CudaScene.hpp"

#include <stdexcept>

#include <cuda_runtime.h>

namespace
{
void checkCuda(cudaError_t error, const char *operation)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

template <typename T>
void uploadVector(const std::vector<T> &source, T **destination)
{
    if (source.empty())
        return;
    checkCuda(cudaMalloc(reinterpret_cast<void **>(destination), source.size() * sizeof(T)), "cudaMalloc");
    checkCuda(cudaMemcpy(*destination, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
}

}

CudaScene::~CudaScene()
{
    release();
}

void CudaScene::release()
{
    cudaFree(dMaterials);
    cudaFree(dTriangles);
    cudaFree(dSpheres);
    cudaFree(dPrimitives);
    cudaFree(dBvh);
    dMaterials = nullptr;
    dTriangles = nullptr;
    dSpheres = nullptr;
    dPrimitives = nullptr;
    dBvh = nullptr;
}

void CudaScene::uploadDeviceBuffers()
{
    uploadVector(hostMaterials, &dMaterials);
    uploadVector(hostTriangles, &dTriangles);
    uploadVector(hostSpheres, &dSpheres);
    uploadVector(hostPrimitives, &dPrimitives);
    uploadVector(hostBvh, &dBvh);
}