#pragma once

#include "CudaTypes.cuh"

__host__ __device__ inline CudaVec3 operator+(CudaVec3 a, CudaVec3 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ inline CudaVec3 operator-(CudaVec3 a, CudaVec3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ inline CudaVec3 operator*(CudaVec3 a, float b)
{
    return {a.x * b, a.y * b, a.z * b};
}

__host__ __device__ inline CudaVec3 operator*(float b, CudaVec3 a) { return a * b; }
__host__ __device__ inline CudaVec3 operator*(CudaVec3 a, CudaVec3 b)
{ return {a.x * b.x, a.y * b.y, a.z * b.z}; }
__host__ __device__ inline CudaVec3 operator/(CudaVec3 a, float b) { return a * (1.0f / b); }
__host__ __device__ inline float cudaDot(CudaVec3 a, CudaVec3 b);
__host__ __device__ inline CudaVec3 cudaCross(CudaVec3 a, CudaVec3 b)
{ return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
__host__ __device__ inline float cudaLength2(CudaVec3 a) { return cudaDot(a, a); }
__host__ __device__ inline CudaVec3 cudaNormalize(CudaVec3 a)
{ const float n = sqrtf(cudaLength2(a)); return n > 0.0f ? a / n : CudaVec3{0,0,0}; }

__host__ __device__ inline float cudaDot(CudaVec3 a, CudaVec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}