#include "CudaScene.hpp"
#include "Scene.hpp"
#include "Sphere.hpp"
#include "Triangle.hpp"
#include "Material.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace
{
CudaVec3 toCuda(const Vector3f &v) { return {v.x, v.y, v.z}; }

struct BvhItem { uint32_t index; };

uint32_t buildBvh(const std::vector<CudaPrimitive> &primitives,
                  std::vector<BvhItem> items,
                  std::vector<CudaBvhNode> &nodes)
{
    CudaBvhNode node{};
    node.boundsMin = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
    node.boundsMax = {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
    for (const BvhItem &item : items)
    {
        const CudaPrimitive &p = primitives[item.index];
        node.boundsMin.x = std::min(node.boundsMin.x, p.boundsMin.x);
        node.boundsMin.y = std::min(node.boundsMin.y, p.boundsMin.y);
        node.boundsMin.z = std::min(node.boundsMin.z, p.boundsMin.z);
        node.boundsMax.x = std::max(node.boundsMax.x, p.boundsMax.x);
        node.boundsMax.y = std::max(node.boundsMax.y, p.boundsMax.y);
        node.boundsMax.z = std::max(node.boundsMax.z, p.boundsMax.z);
    }

    const uint32_t nodeIndex = static_cast<uint32_t>(nodes.size());
    nodes.push_back(node);
    if (items.size() == 1)
    {
        nodes[nodeIndex].primitiveIndex = items[0].index;
        nodes[nodeIndex].primitiveCount = 1;
        return nodeIndex;
    }

    const float ex = node.boundsMax.x - node.boundsMin.x;
    const float ey = node.boundsMax.y - node.boundsMin.y;
    const float ez = node.boundsMax.z - node.boundsMin.z;
    const int axis = ey > ex && ey >= ez ? 1 : (ez > ex && ez > ey ? 2 : 0);
    const auto center = [&](const BvhItem &item) {
        const CudaPrimitive &p = primitives[item.index];
        const float values[3] = {(p.boundsMin.x + p.boundsMax.x) * 0.5f,
                                 (p.boundsMin.y + p.boundsMax.y) * 0.5f,
                                 (p.boundsMin.z + p.boundsMax.z) * 0.5f};
        return values[axis];
    };
    std::sort(items.begin(), items.end(), [&](const BvhItem &a, const BvhItem &b) {
        return center(a) < center(b);
    });
    const auto middle = items.begin() + items.size() / 2;
    nodes[nodeIndex].leftChild = buildBvh(primitives, {items.begin(), middle}, nodes);
    nodes[nodeIndex].rightChild = buildBvh(primitives, {middle, items.end()}, nodes);
    return nodeIndex;
}
}

void CudaScene::upload(const Scene &scene)
{
    release();
    hostMaterials.clear();
    hostTriangles.clear();
    hostSpheres.clear();
    hostPrimitives.clear();
    hostBvh.clear();

    std::unordered_map<const Material *, uint32_t> materials;
    const auto materialIndex = [&](const Material *m) {
        const auto found = materials.find(m);
        if (found != materials.end())
            return found->second;
        const uint32_t index = static_cast<uint32_t>(hostMaterials.size());
        hostMaterials.push_back({static_cast<uint32_t>(m->m_type), toCuda(m->m_emission),
                                 toCuda(m->Kd), toCuda(m->Ks), m->roughness, m->ior});
        materials.emplace(m, index);
        return index;
    };

    for (Object *object : scene.get_objects())
    {
        if (const auto *sphere = dynamic_cast<const Sphere *>(object))
        {
            const uint32_t material = materialIndex(sphere->m);
            const uint32_t geometry = static_cast<uint32_t>(hostSpheres.size());
            const Vector3f radius(sphere->radius);
            hostSpheres.push_back({toCuda(sphere->center), sphere->radius, sphere->radius2, material});
            hostPrimitives.push_back({CudaPrimitiveType::Sphere, geometry, material,
                                      toCuda(sphere->center - radius), toCuda(sphere->center + radius)});
        }
        else if (const auto *mesh = dynamic_cast<const MeshTriangle *>(object))
        {
            const uint32_t material = materialIndex(mesh->m);
            for (const Triangle &triangle : mesh->triangles)
            {
                const uint32_t geometry = static_cast<uint32_t>(hostTriangles.size());
                hostTriangles.push_back({toCuda(triangle.v0), toCuda(triangle.v1),
                                          toCuda(triangle.v2), toCuda(triangle.normal), material});
                const Vector3f lo = Vector3f::Min(triangle.v0, Vector3f::Min(triangle.v1, triangle.v2));
                const Vector3f hi = Vector3f::Max(triangle.v0, Vector3f::Max(triangle.v1, triangle.v2));
                hostPrimitives.push_back({CudaPrimitiveType::Triangle, geometry, material,
                                          toCuda(lo), toCuda(hi)});
            }
        }
    }

    std::vector<BvhItem> items;
    for (uint32_t i = 0; i < hostPrimitives.size(); ++i)
        items.push_back({i});
    if (!items.empty())
        buildBvh(hostPrimitives, std::move(items), hostBvh);
    uploadDeviceBuffers();
}

float CudaScene::totalEmitterArea() const
{
    const auto lengthSquared = [](const CudaVec3 &v) {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    };
    float result = 0.0f;
    for (const CudaPrimitive &primitive : hostPrimitives)
    {
        if (lengthSquared(hostMaterials[primitive.materialIndex].emission) <= 1e-12f)
            continue;
        if (primitive.type == CudaPrimitiveType::Triangle)
        {
            const CudaTriangle &t = hostTriangles[primitive.geometryIndex];
            const CudaVec3 e1{t.v1.x - t.v0.x, t.v1.y - t.v0.y, t.v1.z - t.v0.z};
            const CudaVec3 e2{t.v2.x - t.v0.x, t.v2.y - t.v0.y, t.v2.z - t.v0.z};
            const CudaVec3 cross{e1.y * e2.z - e1.z * e2.y,
                                 e1.z * e2.x - e1.x * e2.z,
                                 e1.x * e2.y - e1.y * e2.x};
            result += 0.5f * std::sqrt(lengthSquared(cross));
        }
        else
        {
            const CudaSphere &s = hostSpheres[primitive.geometryIndex];
            result += 12.566370614359172f * s.radiusSquared;
        }
    }
    return result;
}