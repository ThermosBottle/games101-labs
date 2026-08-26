#include "CudaPhotonMap.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cuda_runtime.h>

namespace
{
float coordinate(const CudaVec3 &p, uint32_t axis)
{
    return axis == 0 ? p.x : axis == 1 ? p.y : p.z;
}

uint32_t buildTree(const std::vector<Photon> &photons,
                   std::vector<uint32_t> indices,
                   std::vector<CudaPhotonKdNode> &nodes)
{
    if (indices.empty()) return UINT32_MAX;
    CudaVec3 lo = photons[indices.front()].position;
    CudaVec3 hi = lo;
    for (uint32_t index : indices)
    {
        const CudaVec3 p = photons[index].position;
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    const CudaVec3 extent{hi.x - lo.x, hi.y - lo.y, hi.z - lo.z};
    const uint32_t axis = extent.y > extent.x && extent.y >= extent.z ? 1u
                        : extent.z > extent.x && extent.z > extent.y ? 2u : 0u;
    const auto middle = indices.begin() + indices.size() / 2;
    std::nth_element(indices.begin(), middle, indices.end(),
        [&](uint32_t a, uint32_t b) { return coordinate(photons[a].position, axis) <
                                             coordinate(photons[b].position, axis); });
    const uint32_t node = static_cast<uint32_t>(nodes.size());
    nodes.push_back({*middle, UINT32_MAX, UINT32_MAX, axis});
    nodes[node].leftChild = buildTree(photons, {indices.begin(), middle}, nodes);
    nodes[node].rightChild = buildTree(photons, {middle + 1, indices.end()}, nodes);
    return node;
}
}

CudaPhotonMap::~CudaPhotonMap() { release(); }

void CudaPhotonMap::release()
{
    cudaFree(devicePhotons);
    cudaFree(deviceNodes);
    devicePhotons = nullptr;
    deviceNodes = nullptr;
    hostPhotons.clear();
    hostNodes.clear();
}

void CudaPhotonMap::build(const std::vector<Photon> &photons)
{
    release();
    hostPhotons = photons;
    if (hostPhotons.empty()) return;
    std::vector<uint32_t> indices(hostPhotons.size());
    for (uint32_t i = 0; i < indices.size(); ++i) indices[i] = i;
    buildTree(hostPhotons, std::move(indices), hostNodes);
    if (cudaMalloc(reinterpret_cast<void **>(&devicePhotons),
                   hostPhotons.size() * sizeof(Photon)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&deviceNodes),
                   hostNodes.size() * sizeof(CudaPhotonKdNode)) != cudaSuccess)
        throw std::runtime_error("failed to allocate the photon KD tree");
    cudaMemcpy(devicePhotons, hostPhotons.data(), hostPhotons.size() * sizeof(Photon),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNodes, hostNodes.data(), hostNodes.size() * sizeof(CudaPhotonKdNode),
               cudaMemcpyHostToDevice);
}

CudaPhotonKdTreeView CudaPhotonMap::deviceView() const
{
    return {devicePhotons, deviceNodes, photonCount(),
            static_cast<uint32_t>(hostNodes.size())};
}