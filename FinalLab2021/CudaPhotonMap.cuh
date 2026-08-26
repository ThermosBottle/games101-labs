#pragma once

#include <cstdint>
#include <vector>
#include "CudaTypes.cuh"

struct Photon
{
    CudaVec3 position;
    CudaVec3 direction;
    CudaVec3 power;
    uint32_t valid;
};

struct CudaPhotonKdNode
{
    uint32_t photonIndex;
    uint32_t leftChild;
    uint32_t rightChild;
    uint32_t axis;
};

struct CudaPhotonKdTreeView
{
    const Photon *photons;
    const CudaPhotonKdNode *nodes;
    uint32_t photonCount;
    uint32_t nodeCount;
};

// Persistent per-pixel state. tau is accumulated local reflected energy;
// it is not final radiance and must be divided by the current disk area and
// the total number of emitted photons only during resolve.
struct SPPMPixel
{
    CudaVec3 position;
    CudaVec3 normal;
    CudaVec3 viewDirection;
    CudaVec3 beta;
    CudaVec3 emitted;
    uint32_t cameraSampleCount;
    CudaVec3 tau;
    CudaVec3 newFlux;
    float radiusSquared;
    uint32_t photonCount;
    uint32_t newPhotonCount;
    uint32_t materialIndex;
    uint32_t valid;
};

class CudaPhotonMap
{
public:
    ~CudaPhotonMap();
    CudaPhotonMap(const CudaPhotonMap &) = delete;
    CudaPhotonMap &operator=(const CudaPhotonMap &) = delete;
    CudaPhotonMap() = default;

    void build(const std::vector<Photon> &photons);
    void release();
    CudaPhotonKdTreeView deviceView() const;
    uint32_t photonCount() const { return static_cast<uint32_t>(hostPhotons.size()); }

private:
    std::vector<Photon> hostPhotons;
    std::vector<CudaPhotonKdNode> hostNodes;
    Photon *devicePhotons = nullptr;
    CudaPhotonKdNode *deviceNodes = nullptr;
};