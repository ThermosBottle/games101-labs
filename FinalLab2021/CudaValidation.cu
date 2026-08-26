#include "CudaKernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

static bool check(cudaError_t e,const char *what){if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",what,cudaGetErrorString(e));return false;}return true;}
int main(){
    CudaTriangle tri{{-1, -1, 0},{1,-1,0},{0,1,0},{0,0,1},0};
    CudaSphere sph{{3,0,0},1,1,1};
    CudaPrimitive p[2]={{CudaPrimitiveType::Triangle,0,0,{-1,-1,-.01f},{1,1,.01f}}, {CudaPrimitiveType::Sphere,0,1,{2,-1,-1},{4,1,1}}};
    CudaBvhNode b[3]{}; b[0]={{-1,-1,-1},{4,1,1},1,2,0,0}; b[1]=p[0].boundsMin.x==p[0].boundsMin.x?CudaBvhNode{p[0].boundsMin,p[0].boundsMax,0,0,0,1}:CudaBvhNode{}; b[2]={p[1].boundsMin,p[1].boundsMax,0,0,1,1};
    CudaRay rays[3]={{ {0,0,2},{0,0,-1},{0,0,-1},0.001f,100},{{0,0,2},{3,0,-2},{0,0,0},0.001f,100},{{0,3,2},{0,0,-1},{0,0,-1},0.001f,100}};
    CudaTriangle *dt;CudaSphere *ds;CudaPrimitive *dp;CudaBvhNode *db;CudaRay *dr;cudaMalloc(&dt,sizeof(tri));cudaMalloc(&ds,sizeof(sph));cudaMalloc(&dp,sizeof(p));cudaMalloc(&db,sizeof(b));cudaMalloc(&dr,sizeof(rays));cudaMemcpy(dt,&tri,sizeof(tri),cudaMemcpyHostToDevice);cudaMemcpy(ds,&sph,sizeof(sph),cudaMemcpyHostToDevice);cudaMemcpy(dp,p,sizeof(p),cudaMemcpyHostToDevice);cudaMemcpy(db,b,sizeof(b),cudaMemcpyHostToDevice);cudaMemcpy(dr,rays,sizeof(rays),cudaMemcpyHostToDevice);
    CudaValidationStats stats{};validateCudaIntersections(dr,3,dt,1,ds,1,dp,2,db,3,stats);
    std::printf("triangle_hits=%u sphere_hits=%u bvh_hits=%u mismatches=%u max_distance_error=%g max_normal_error=%g\n",stats.triangleHits,stats.sphereHits,stats.bvhHits,stats.mismatches,stats.maxDistanceError,stats.maxNormalError);
    bool ok=stats.triangleHits==1&&stats.sphereHits==1&&stats.bvhHits==2&&stats.mismatches==0&&stats.maxDistanceError<1e-4f&&stats.maxNormalError<1e-4f;
    cudaFree(dt);cudaFree(ds);cudaFree(dp);cudaFree(db);cudaFree(dr);return ok?0:1;
}