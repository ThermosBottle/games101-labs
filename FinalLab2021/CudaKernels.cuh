#pragma once

#include <cstdint>
#include "CudaTypes.cuh"
#include "CudaPhotonMap.cuh"

struct CudaValidationStats
{
    uint32_t triangleHits;
    uint32_t sphereHits;
    uint32_t bvhHits;
    uint32_t mismatches;
    float maxDistanceError;
    float maxNormalError;
};

void launchCudaIntersectionKernel(const CudaRay *, uint32_t,
                                  const CudaTriangle *, uint32_t,
                                  const CudaSphere *, uint32_t,
                                  const CudaPrimitive *, uint32_t,
                                  const CudaBvhNode *, uint32_t,
                                  CudaHit *, CudaHit *);

void validateCudaIntersections(const CudaRay *rays, uint32_t rayCount,
                               const CudaTriangle *triangles, uint32_t triangleCount,
                               const CudaSphere *spheres, uint32_t sphereCount,
                               const CudaPrimitive *primitives, uint32_t primitiveCount,
                               const CudaBvhNode *bvh, uint32_t bvhCount,
                               CudaValidationStats &stats);

void launchCudaDiffuseBounce(const CudaRay *cameraRays, uint32_t width, uint32_t height,
                             const CudaTriangle *triangles, uint32_t triangleCount,
                             const CudaSphere *spheres, uint32_t sphereCount,
                             const CudaPrimitive *primitives, uint32_t primitiveCount,
                             const CudaBvhNode *bvh, uint32_t bvhCount,
                             const CudaMaterial *materials,
                             CudaVec3 lightPosition, CudaVec3 lightEmission,
                             CudaVec3 background, uint32_t sampleIndex,
                             CudaVec3 *framebuffer);

// Render one sample per pixel.  The primary rays are generated on the GPU and
// the path state is then iterated entirely on the GPU (no per-bounce host
// synchronization or device-pointer downloads are required).
void launchCudaPathTracing(uint32_t width, uint32_t height, float fov,
                           CudaVec3 eye, uint32_t sampleIndex, uint32_t maxDepth,
                           float russianRoulette,
                           const CudaTriangle *triangles, uint32_t triangleCount,
                           const CudaSphere *spheres, uint32_t sphereCount,
                           const CudaPrimitive *primitives, uint32_t primitiveCount,
                            const CudaBvhNode *bvh, uint32_t bvhCount,
                            const CudaMaterial *materials, float totalEmitterArea,
                            CudaVec3 background,
                           CudaVec3 *framebuffer);

void launchCudaSceneSmokeKernel(const CudaPrimitive *devicePrimitives,
                                uint32_t primitiveCount,
                                uint32_t *deviceCounter);

void launchCudaSppmCamera(uint32_t width, uint32_t height, float fov,
                          CudaVec3 eye, uint32_t iteration, uint32_t maxDepth,
                          float roulette, CudaSceneView scene,
                          SPPMPixel *visiblePoints);
void launchCudaSppmPhotons(uint32_t photonCount, uint32_t iteration,
                           uint32_t maxDepth, float roulette,
                           CudaSceneView scene, std::vector<Photon> &photons);
void launchCudaSppmGather(SPPMPixel *visiblePoints, uint32_t count,
                          CudaPhotonKdTreeView photons, CudaSceneView scene);
void launchCudaSppmUpdate(SPPMPixel *visiblePoints, uint32_t count,
                          float alpha);
void launchCudaSppmResolve(const SPPMPixel *visiblePoints, uint32_t count,
                           uint32_t emittedPhotonCount, CudaVec3 *framebuffer);