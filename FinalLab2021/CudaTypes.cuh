#pragma once

#include <cstdint>

// These structures contain only trivially copyable fields so they can be
// uploaded to device memory without pointer fix-up or virtual dispatch.
struct CudaVec3
{
    float x;
    float y;
    float z;
};

struct CudaRay
{
    CudaVec3 origin;
    CudaVec3 direction;
    CudaVec3 inverseDirection;
    float tMin;
    float tMax;
};

enum class CudaPrimitiveType : uint32_t
{
    Triangle = 0,
    Sphere = 1
};

// Keep these values in sync with MaterialType.  The explicit event type is
// important: a zero diffuse colour is not a reliable indication of whether a
// surface can be used by the density estimator.
enum CudaMaterialType : uint32_t
{
    CudaDiffuse = 0,
    CudaMicrofacet = 1,
    CudaMirror = 2,
    CudaGlass = 3
};

struct CudaMaterial
{
    uint32_t type;
    CudaVec3 emission;
    CudaVec3 diffuse;
    CudaVec3 specular;
    float roughness;
    float ior;
};

struct CudaBsdfSample
{
    CudaVec3 direction;
    CudaVec3 weight;       // multiplier for path throughput
    float pdf;             // solid-angle PDF; zero for delta events
    float eta;             // eta_i / eta_t for transmission, otherwise 1
    uint32_t isDelta;
    uint32_t isTransmission;
    uint32_t valid;
};

struct CudaTriangle
{
    CudaVec3 v0;
    CudaVec3 v1;
    CudaVec3 v2;
    CudaVec3 normal;
    uint32_t materialIndex;
};

struct CudaSphere
{
    CudaVec3 center;
    float radius;
    float radiusSquared;
    uint32_t materialIndex;
};

struct CudaPrimitive
{
    CudaPrimitiveType type;
    uint32_t geometryIndex;
    uint32_t materialIndex;
    CudaVec3 boundsMin;
    CudaVec3 boundsMax;
};

// A linear BVH uses integer indices instead of host pointers. A leaf stores
// one index into CudaPrimitive; interior nodes store two child indices.
struct CudaBvhNode
{
    CudaVec3 boundsMin;
    CudaVec3 boundsMax;
    uint32_t leftChild;
    uint32_t rightChild;
    uint32_t primitiveIndex;
    uint32_t primitiveCount;
};

struct CudaHit
{
    float distance;
    uint32_t primitiveIndex;
    uint32_t materialIndex;
    CudaVec3 position;
    CudaVec3 normal;
    float barycentricU;
    float barycentricV;
    uint32_t hit;
};

// A compact, non-owning view of the scene. It is passed by value to kernels
// so the launch interface does not have to repeat every device pointer.
struct CudaSceneView
{
    const CudaTriangle *triangles;
    uint32_t triangleCount;
    const CudaSphere *spheres;
    uint32_t sphereCount;
    const CudaPrimitive *primitives;
    uint32_t primitiveCount;
    const CudaBvhNode *bvh;
    uint32_t bvhCount;
    const CudaMaterial *materials;
    float totalEmitterArea;
};