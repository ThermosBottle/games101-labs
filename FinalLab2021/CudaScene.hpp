#pragma once

#include <cstdint>
#include <vector>

#include "CudaTypes.cuh"

class Scene;

// CudaScene owns the flat host staging arrays and their corresponding device
// allocations. CPU scene objects remain owned by Scene and are never exposed
// to CUDA kernels through raw host pointers.
class CudaScene
{
public:
    CudaScene() = default;
    ~CudaScene();

    CudaScene(const CudaScene &) = delete;
    CudaScene &operator=(const CudaScene &) = delete;

    void upload(const Scene &scene);
    void release();

    const std::vector<CudaPrimitive> &primitives() const { return hostPrimitives; }
    const std::vector<CudaTriangle> &triangles() const { return hostTriangles; }
    const std::vector<CudaSphere> &spheres() const { return hostSpheres; }
    const std::vector<CudaMaterial> &materials() const { return hostMaterials; }
    const std::vector<CudaBvhNode> &bvh() const { return hostBvh; }
    uint32_t primitiveCount() const { return static_cast<uint32_t>(hostPrimitives.size()); }
    uint32_t triangleCount() const { return static_cast<uint32_t>(hostTriangles.size()); }
    uint32_t sphereCount() const { return static_cast<uint32_t>(hostSpheres.size()); }
    float totalEmitterArea() const;
    CudaSceneView deviceView() const
    {
        return {dTriangles, triangleCount(), dSpheres, sphereCount(),
                dPrimitives, primitiveCount(), dBvh,
                static_cast<uint32_t>(hostBvh.size()), dMaterials,
                totalEmitterArea()};
    }
    const CudaPrimitive *devicePrimitives() const { return dPrimitives; }
    const CudaTriangle *deviceTriangles() const { return dTriangles; }
    const CudaSphere *deviceSpheres() const { return dSpheres; }
    const CudaMaterial *deviceMaterials() const { return dMaterials; }
    const CudaBvhNode *deviceBvh() const { return dBvh; }

private:
    std::vector<CudaMaterial> hostMaterials;
    std::vector<CudaTriangle> hostTriangles;
    std::vector<CudaSphere> hostSpheres;
    std::vector<CudaPrimitive> hostPrimitives;
    std::vector<CudaBvhNode> hostBvh;

    CudaMaterial *dMaterials = nullptr;
    CudaTriangle *dTriangles = nullptr;
    CudaSphere *dSpheres = nullptr;
    CudaPrimitive *dPrimitives = nullptr;
    CudaBvhNode *dBvh = nullptr;

    void uploadDeviceBuffers();
};