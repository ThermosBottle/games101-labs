#include "CudaKernels.cuh"
#include "CudaMath.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

namespace {
__device__ bool hitTri(const CudaRay&r,const CudaTriangle&t,CudaHit&h){CudaVec3 e1=t.v1-t.v0,e2=t.v2-t.v0,p=cudaCross(r.direction,e2);float d=cudaDot(e1,p);if(fabsf(d)<1e-7f)return false;float q=1/d;CudaVec3 s=r.origin-t.v0;float u=cudaDot(s,p)*q;if(u<0||u>1)return false;CudaVec3 z=cudaCross(s,e1);float v=cudaDot(r.direction,z)*q;float x=cudaDot(e2,z)*q;if(v<0||u+v>1||x<r.tMin||x>r.tMax||x>=h.distance)return false;h={x,0,t.materialIndex,r.origin+r.direction*x,t.normal,u,v,1};return true;}
__device__ bool hitSphere(const CudaRay&r,const CudaSphere&s,CudaHit&h){CudaVec3 q=r.origin-s.center;float b=cudaDot(q,r.direction),c=cudaDot(q,q)-s.radiusSquared,D=b*b-c;if(D<0)return false;float x=-b-sqrtf(D);if(x<r.tMin)x=-b+sqrtf(D);if(x<r.tMin||x>r.tMax||x>=h.distance)return false;CudaVec3 p=r.origin+r.direction*x;h={x,0,s.materialIndex,p,cudaNormalize(p-s.center),0,0,1};return true;}
__device__ CudaHit intersectAll(const CudaRay&r,const CudaTriangle*t,const CudaSphere*s,const CudaPrimitive*p,uint32_t np){CudaHit h{};h.distance=r.tMax;for(uint32_t i=0;i<np;++i){bool ok=p[i].type==CudaPrimitiveType::Triangle?hitTri(r,t[p[i].geometryIndex],h):hitSphere(r,s[p[i].geometryIndex],h);if(ok)h.primitiveIndex=i;}return h;}
__device__ CudaHit intersectBvh(const CudaRay&r,const CudaTriangle*t,const CudaSphere*s,const CudaPrimitive*p,const CudaBvhNode*n,uint32_t nn){CudaHit h{};h.distance=r.tMax;if(!nn)return h;uint32_t stack[64],top=0;stack[top++]=0;while(top){const uint32_t ni=stack[--top];const CudaBvhNode& b=n[ni];bool inside=true;for(int a=0;a<3;++a){float o=a==0?r.origin.x:(a==1?r.origin.y:r.origin.z),d=a==0?r.direction.x:(a==1?r.direction.y:r.direction.z),lo=a==0?b.boundsMin.x:(a==1?b.boundsMin.y:b.boundsMin.z),hi=a==0?b.boundsMax.x:(a==1?b.boundsMax.y:b.boundsMax.z);float inv=1.0f/d,x=(lo-o)*inv,y=(hi-o)*inv;if(x>y){float q=x;x=y;y=q;}if(fmaxf(r.tMin,x)>fminf(h.distance,y)){inside=false;break;}}if(!inside)continue;if(b.primitiveCount){const uint32_t pi=b.primitiveIndex;bool ok=p[pi].type==CudaPrimitiveType::Triangle?hitTri(r,t[p[pi].geometryIndex],h):hitSphere(r,s[p[pi].geometryIndex],h);if(ok)h.primitiveIndex=pi;}else if(top+2<64){stack[top++]=b.leftChild;stack[top++]=b.rightChild;}}return h;}
__global__ void kernel(const CudaRay*r,uint32_t nr,const CudaTriangle*t,const CudaSphere*s,const CudaPrimitive*p,uint32_t np,const CudaBvhNode*n,uint32_t nn,CudaHit*a,CudaHit*b){uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<nr){a[i]=intersectAll(r[i],t,s,p,np);b[i]=intersectBvh(r[i],t,s,p,n,nn);}}
}
void launchCudaIntersectionKernel(const CudaRay*r,uint32_t nr,const CudaTriangle*t,uint32_t nt,const CudaSphere*s,uint32_t ns,const CudaPrimitive*p,uint32_t np,const CudaBvhNode*n,uint32_t nn,CudaHit*a,CudaHit*b){(void)nt;(void)ns;if(nr)kernel<<<(nr+127)/128,128>>>(r,nr,t,s,p,np,n,nn,a,b);cudaDeviceSynchronize();}
__global__ void smokeKernel(const CudaPrimitive *p,uint32_t n,uint32_t *counter){uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&p[i].boundsMin.x<=p[i].boundsMax.x)atomicAdd(counter,1u);}
void launchCudaSceneSmokeKernel(const CudaPrimitive*p,uint32_t n,uint32_t*c){if(n)smokeKernel<<<(n+127)/128,128>>>(p,n,c);cudaDeviceSynchronize();}

void validateCudaIntersections(const CudaRay *rays, uint32_t count,
                               const CudaTriangle *triangles, uint32_t triangleCount,
                               const CudaSphere *spheres, uint32_t sphereCount,
                               const CudaPrimitive *primitives, uint32_t primitiveCount,
                               const CudaBvhNode *bvh, uint32_t bvhCount,
                               CudaValidationStats &stats)
{
    stats = {};
    if (!count) return;
    CudaRay *dr=nullptr; CudaHit *da=nullptr,*db=nullptr;
    cudaMalloc(&dr,count*sizeof(CudaRay)); cudaMalloc(&da,count*sizeof(CudaHit)); cudaMalloc(&db,count*sizeof(CudaHit));
    cudaMemcpy(dr,rays,count*sizeof(CudaRay),cudaMemcpyHostToDevice);
    launchCudaIntersectionKernel(dr,count,triangles,triangleCount,spheres,sphereCount,primitives,primitiveCount,bvh,bvhCount,da,db);
    std::vector<CudaHit> a(count),b(count); std::vector<CudaPrimitive> hp(primitiveCount); cudaMemcpy(a.data(),da,count*sizeof(CudaHit),cudaMemcpyDeviceToHost); cudaMemcpy(b.data(),db,count*sizeof(CudaHit),cudaMemcpyDeviceToHost); if(primitiveCount)cudaMemcpy(hp.data(),primitives,primitiveCount*sizeof(CudaPrimitive),cudaMemcpyDeviceToHost);
    for(uint32_t i=0;i<count;++i){if(a[i].hit && a[i].primitiveIndex < primitiveCount && hp[a[i].primitiveIndex].type == CudaPrimitiveType::Triangle)++stats.triangleHits; if(a[i].hit && a[i].primitiveIndex < primitiveCount && hp[a[i].primitiveIndex].type == CudaPrimitiveType::Sphere)++stats.sphereHits; if(b[i].hit)++stats.bvhHits; if(a[i].hit!=b[i].hit||(a[i].hit&&a[i].primitiveIndex!=b[i].primitiveIndex)){++stats.mismatches;continue;}if(a[i].hit){stats.maxDistanceError=fmaxf(stats.maxDistanceError,fabsf(a[i].distance-b[i].distance));CudaVec3 dn=a[i].normal-b[i].normal;stats.maxNormalError=fmaxf(stats.maxNormalError,sqrtf(cudaLength2(dn)));}}
    cudaFree(dr);cudaFree(da);cudaFree(db);
}

__device__ CudaVec3 cosineSample(CudaVec3 n, unsigned int seed)
{
    float u = (seed * 1664525u + 1013904223u) * 2.3283064365e-10f;
    float v = (seed * 22695477u + 1u) * 2.3283064365e-10f;
    float r = sqrtf(u), phi = 6.283185307f * v;
    CudaVec3 t = fabsf(n.x) > fabsf(n.z) ? cudaNormalize(CudaVec3{-n.y, n.x, 0}) : cudaNormalize(CudaVec3{0, -n.z, n.y});
    CudaVec3 b = cudaCross(n, t);
    return cudaNormalize(t * (r*cosf(phi)) + b * (r*sinf(phi)) + n * sqrtf(fmaxf(0.0f, 1.0f-u)));
}

// A small per-pixel generator is preferable to global mutable RNG state: it
// makes samples independent between CUDA blocks and gives reproducible images
// when the same sample count is requested.
__device__ unsigned int pathHash(unsigned int x)
{
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
    x *= 0x846ca68bu; return x ^ (x >> 16);
}

__device__ float pathRandom(unsigned int &state)
{
    state = pathHash(state + 0x9e3779b9u);
    return (state + 1.0f) * 2.3283064365386963e-10f;
}

__device__ CudaVec3 pathCosineSample(CudaVec3 n, unsigned int &state)
{
    const float u = pathRandom(state);
    const float v = pathRandom(state);
    const float radius = sqrtf(u);
    const float phi = 6.28318530718f * v;
    const CudaVec3 tangent = fabsf(n.x) > fabsf(n.z)
        ? cudaNormalize(CudaVec3{-n.y, n.x, 0.0f})
        : cudaNormalize(CudaVec3{0.0f, -n.z, n.y});
    const CudaVec3 bitangent = cudaCross(n, tangent);
    return cudaNormalize(tangent * (radius * cosf(phi)) +
                         bitangent * (radius * sinf(phi)) +
                         n * sqrtf(fmaxf(0.0f, 1.0f - u)));
}

// The following helpers mirror Material::eval/pdf/sample in Material.hpp.
// Keeping them as device functions avoids copying CPU polymorphic Material
// objects to the GPU while preserving the same GGX Cook-Torrance model.
__device__ float ggxDistribution(CudaVec3 n, CudaVec3 h, float roughness)
{
    const float ndoth = fmaxf(0.0f, cudaDot(n, h));
    const float alpha = fmaxf(0.001f, roughness * roughness);
    const float alpha2 = alpha * alpha;
    const float denominator = ndoth * ndoth * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (3.14159265f * denominator * denominator);
}

__device__ float ggxGeometry(float ndotv, float roughness)
{
    const float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
    return ndotv / (ndotv * (1.0f - k) + k);
}

__device__ CudaVec3 schlickFresnel(CudaVec3 ks, float cosTheta)
{
    const float f = powf(1.0f - fminf(1.0f, fmaxf(0.0f, cosTheta)), 5.0f);
    return ks + (CudaVec3{1.0f, 1.0f, 1.0f} - ks) * f;
}

__device__ CudaVec3 evaluateBsdf(const CudaMaterial &m, CudaVec3 incoming,
                                CudaVec3 outgoing, CudaVec3 n)
{
    const CudaVec3 v = cudaNormalize(incoming * -1.0f);
    const CudaVec3 l = cudaNormalize(outgoing);
    const float ndotv = fmaxf(0.0f, cudaDot(n, v));
    const float ndotl = fmaxf(0.0f, cudaDot(n, l));
    if (ndotv <= 0.0f || ndotl <= 0.0f) return {0.0f, 0.0f, 0.0f};
    if (m.type == 0u) return m.diffuse * (1.0f / 3.14159265f);

    const CudaVec3 h = cudaNormalize(v + l);
    const float vdoth = fmaxf(0.0f, cudaDot(v, h));
    const float d = ggxDistribution(n, h, m.roughness);
    const float g = ggxGeometry(ndotv, m.roughness) * ggxGeometry(ndotl, m.roughness);
    const CudaVec3 f = schlickFresnel(m.specular, vdoth);
    const CudaVec3 specular = f * (d * g / fmaxf(1e-7f, 4.0f * ndotv * ndotl));
    return (CudaVec3{1.0f, 1.0f, 1.0f} - f) * (m.diffuse * (1.0f / 3.14159265f)) + specular;
}

__device__ float microfacetSpecularProbability(const CudaMaterial &m)
{
    // The sampling probabilities are also used by evaluateBsdfPdf(). Keep
    // both paths tied to the same luminance-based mixture probability.
    const float luminance = 0.2126f * m.specular.x +
                            0.7152f * m.specular.y +
                            0.0722f * m.specular.z;
    return fminf(0.95f, fmaxf(0.05f, luminance));
}

__device__ float ggxPdf(const CudaMaterial &m, CudaVec3 incoming,
                        CudaVec3 outgoing, CudaVec3 n)
{
    const CudaVec3 v = cudaNormalize(incoming * -1.0f);
    const CudaVec3 l = cudaNormalize(outgoing);
    if (cudaDot(n, l) <= 0.0f) return 0.0f;

    const CudaVec3 h = cudaNormalize(v + l);
    const float ndoth = cudaDot(n, h);
    const float vdoth = fabsf(cudaDot(v, h));
    return (ndoth > 0.0f && vdoth > 1e-7f)
        ? ggxDistribution(n, h, m.roughness) * ndoth / (4.0f * vdoth)
        : 0.0f;
}

__device__ float evaluateBsdfPdf(const CudaMaterial &m, CudaVec3 incoming,
                                  CudaVec3 outgoing, CudaVec3 n)
{
    const CudaVec3 v = cudaNormalize(incoming * -1.0f);
    const CudaVec3 l = cudaNormalize(outgoing);
    const float ndotl = cudaDot(n, l);
    if (ndotl <= 0.0f) return 0.0f;
    if (m.type == 0u) return ndotl / 3.14159265f;

    const float pSpecular = microfacetSpecularProbability(m);
    const float diffusePdf = ndotl / 3.14159265f;
    const float specularPdf = ggxPdf(m, incoming, outgoing, n);
    return (1.0f - pSpecular) * diffusePdf + pSpecular * specularPdf;
}

__device__ CudaVec3 sampleBsdf(const CudaMaterial &m, CudaVec3 incoming,
                               CudaVec3 n, unsigned int &state, float &pdf)
{
    if (m.type == 0u)
    {
        CudaVec3 direction = pathCosineSample(n, state);
        pdf = fmaxf(0.0f, cudaDot(n, direction)) / 3.14159265f;
        return direction;
    }

    const CudaVec3 v = cudaNormalize(incoming * -1.0f);
    if (cudaDot(n, v) <= 0.0f) { pdf = 0.0f; return {0.0f, 0.0f, 0.0f}; }

    const float pSpecular = microfacetSpecularProbability(m);
    CudaVec3 direction;
    if (pathRandom(state) < pSpecular)
    {
        const float alpha = fmaxf(0.001f, m.roughness * m.roughness);
        const float u = pathRandom(state);
        const float phi = 6.28318530718f * pathRandom(state);
        const float tanTheta2 = alpha * alpha * u / fmaxf(1.0f - u, 1e-7f);
        const float cosTheta = 1.0f / sqrtf(1.0f + tanTheta2);
        const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
        const CudaVec3 tangent = fabsf(n.x) > fabsf(n.z)
            ? cudaNormalize(CudaVec3{-n.y, n.x, 0.0f})
            : cudaNormalize(CudaVec3{0.0f, -n.z, n.y});
        const CudaVec3 bitangent = cudaCross(n, tangent);
        const CudaVec3 h = cudaNormalize(tangent * (sinTheta * cosf(phi)) +
                                          bitangent * (sinTheta * sinf(phi)) + n * cosTheta);
        direction = cudaNormalize(v * -1.0f - h * (2.0f * cudaDot(v * -1.0f, h)));
    }
    else
    {
        direction = pathCosineSample(n, state);
    }

    // Both branches use this same mixture PDF, including the probability of
    // selecting the branch that generated direction.
    pdf = evaluateBsdfPdf(m, incoming, direction, n);
    return direction;
}

__device__ float primitiveArea(const CudaPrimitive &p, const CudaTriangle *triangles,
                               const CudaSphere *spheres)
{
    if (p.type == CudaPrimitiveType::Triangle)
    {
        const CudaTriangle &t = triangles[p.geometryIndex];
        return 0.5f * sqrtf(cudaLength2(cudaCross(t.v1 - t.v0, t.v2 - t.v0)));
    }
    return 12.566370614359172f * spheres[p.geometryIndex].radiusSquared;
}

__device__ bool sampleEmitter(const CudaTriangle *triangles, const CudaSphere *spheres,
                              const CudaPrimitive *primitives, uint32_t primitiveCount,
                              const CudaMaterial *materials, float totalArea,
                              unsigned int &rng, CudaVec3 &point, CudaVec3 &normal,
                              CudaVec3 &emission, float &pdfArea)
{
    if (totalArea <= 0.0f) return false;
    float target = pathRandom(rng) * totalArea;
    for (uint32_t i = 0; i < primitiveCount; ++i)
    {
        const CudaPrimitive &p = primitives[i];
        const CudaMaterial &m = materials[p.materialIndex];
        if (cudaLength2(m.emission) <= 1e-12f) continue;
        const float area = primitiveArea(p, triangles, spheres);
        if (target > area) { target -= area; continue; }
        if (p.type == CudaPrimitiveType::Triangle)
        {
            const CudaTriangle &t = triangles[p.geometryIndex];
            const float u = sqrtf(pathRandom(rng));
            const float v = pathRandom(rng);
            point = t.v0 * (1.0f - u) + t.v1 * (u * (1.0f - v)) + t.v2 * (u * v);
            normal = t.normal;
        }
        else
        {
            const CudaSphere &s = spheres[p.geometryIndex];
            const float z = 1.0f - 2.0f * pathRandom(rng);
            const float phi = 6.28318530718f * pathRandom(rng);
            const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
            normal = CudaVec3{r * cosf(phi), r * sinf(phi), z};
            point = s.center + normal * s.radius;
        }
        emission = m.emission;
        pdfArea = 1.0f / totalArea;
        return true;
    }
    return false;
}

__global__ void pathTracingKernel(uint32_t width, uint32_t height, float fov,
                                  CudaVec3 eye, uint32_t sampleIndex,
                                  uint32_t maxDepth, float roulette,
                                  const CudaTriangle *triangles,
                                  const CudaSphere *spheres,
                                  const CudaPrimitive *primitives,
                                  uint32_t primitiveCount,
                                  const CudaBvhNode *bvh, uint32_t bvhCount,
                                  const CudaMaterial *materials,
                                  float totalEmitterArea,
                                  CudaVec3 background, CudaVec3 *framebuffer)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const uint32_t pixel = y * width + x;
    unsigned int rng = pathHash(pixel ^ (sampleIndex * 0x51ed270bu));

    // Match the CPU camera: eye=(278,273,-800), forward=+Z, vertical FOV.
    const float scale = tanf(fov * 0.5f * 0.0174532925199433f);
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float jx = pathRandom(rng) - 0.5f;
    const float jy = pathRandom(rng) - 0.5f;
    const float sx = (2.0f * (static_cast<float>(x) + 0.5f + jx) / width - 1.0f) * aspect * scale;
    const float sy = (1.0f - 2.0f * (static_cast<float>(y) + 0.5f + jy) / height) * scale;
    CudaRay ray{eye, cudaNormalize(CudaVec3{-sx, sy, 1.0f}), {}, 1e-3f, 1e30f};
    CudaVec3 throughput{1.0f, 1.0f, 1.0f};
    CudaVec3 radiance{0.0f, 0.0f, 0.0f};
    float previousBsdfPdf = 0.0f;
    bool previousWasBsdfSample = false;
    CudaVec3 previousPoint{0.0f, 0.0f, 0.0f};

    // Iterative path state replaces CPU recursion and avoids device recursion
    // overhead. Continuation rays use the material-specific BSDF sampler.
    for (uint32_t depth = 0; depth < maxDepth; ++depth)
    {
        const CudaHit hit = intersectBvh(ray, triangles, spheres, primitives, bvh, bvhCount);
        if (!hit.hit) { radiance = radiance + throughput * background; break; }
        const CudaMaterial material = materials[hit.materialIndex];
        if (cudaLength2(material.emission) > 1e-12f)
        {
            float weight = 1.0f;
            if (previousWasBsdfSample && totalEmitterArea > 0.0f)
            {
                const CudaVec3 toLight = hit.position - previousPoint;
                const float distance2 = cudaLength2(toLight);
                const CudaVec3 lightDir = cudaNormalize(toLight);
                const float cosLight = cudaDot(lightDir * -1.0f, hit.normal);
                const float lightPdf = distance2 / fmaxf(1e-7f, cosLight * totalEmitterArea);
                const float a = previousBsdfPdf * previousBsdfPdf;
                const float b = lightPdf * lightPdf;
                weight = (cosLight > 0.0f && isfinite(lightPdf)) ? a / fmaxf(1e-7f, a + b) : 0.0f;
            }
            radiance = radiance + throughput * material.emission * weight;
        }

        CudaVec3 normal = hit.normal;
        if (cudaDot(normal, ray.direction) > 0.0f) normal = normal * -1.0f;

        // OLD ALGORITHM (disabled): fixed point-light approximation.
        // const CudaVec3 toLight = lightPosition - hit.position;
        // const float lightDistance2 = cudaLength2(toLight);
        // ... evaluate BSDF * lightEmission * cosSurface / lightDistance2 ...

        // New algorithm: sample an emissive surface by area, convert its area
        // PDF to solid-angle PDF, then combine it with the BSDF PDF using MIS.
        CudaVec3 lightPoint, lightNormal, lightEmission;
        float lightPdfArea = 0.0f;
        if (sampleEmitter(triangles, spheres, primitives, primitiveCount, materials,
                          totalEmitterArea, rng, lightPoint, lightNormal,
                          lightEmission, lightPdfArea))
        {
            const CudaVec3 toLight = lightPoint - hit.position;
            const float lightDistance2 = cudaLength2(toLight);
            const float lightDistance = sqrtf(lightDistance2);
            const CudaVec3 wi = toLight / fmaxf(1e-7f, lightDistance);
            const float cosSurface = fmaxf(0.0f, cudaDot(normal, wi));
            const float cosLight = fmaxf(0.0f, cudaDot(lightNormal, wi * -1.0f));
            const float lightPdf = lightPdfArea * lightDistance2 / fmaxf(1e-7f, cosLight);
            const CudaVec3 directBsdf = evaluateBsdf(material, ray.direction, wi, normal);
            const float actualBsdfPdf = evaluateBsdfPdf(material, ray.direction, wi, normal);
            const float a = lightPdf * lightPdf;
            const float b = actualBsdfPdf * actualBsdfPdf;
            const float mis = (lightPdf > 0.0f && cosSurface > 0.0f && cosLight > 0.0f)
                ? a / fmaxf(1e-7f, a + b) : 0.0f;
            if (cosSurface > 0.0f && cosLight > 0.0f && lightPdf > 0.0f)
            {
                const CudaRay shadow{hit.position + normal * 1e-3f, wi, {},
                                     1e-3f, lightDistance - 2e-3f};
                if (!intersectBvh(shadow, triangles, spheres, primitives, bvh, bvhCount).hit)
                    radiance = radiance + throughput * directBsdf * lightEmission *
                               (cosSurface * mis / lightPdf);
            }
        }

        float continuation = 1.0f;
        if (depth >= 3)
        {
            // Match Scene::castRay(): CPU uses the scene-wide 0.8 roulette
            // probability, then divides surviving paths by that probability.
            continuation = fminf(0.95f, fmaxf(0.05f, roulette));
            if (pathRandom(rng) > continuation) break;
        }
        float pdf = 0.0f;
        const CudaVec3 wi = sampleBsdf(material, ray.direction, normal, rng, pdf);
        const float cosine = fmaxf(0.0f, cudaDot(normal, wi));
        if (cosine <= 0.0f || pdf <= 1e-7f) break;
        const CudaVec3 bsdf = evaluateBsdf(material, ray.direction, wi, normal);
        throughput = throughput * bsdf * (cosine / (pdf * continuation));
        previousPoint = hit.position;
        previousBsdfPdf = pdf;
        previousWasBsdfSample = true;
        ray = {hit.position + normal * 1e-3f, wi, {}, 1e-3f, 1e30f};
    }
    framebuffer[pixel] = framebuffer[pixel] + radiance;
}

void launchCudaPathTracing(uint32_t width, uint32_t height, float fov,
                           CudaVec3 eye, uint32_t sampleIndex, uint32_t maxDepth,
                           float russianRoulette,
                           const CudaTriangle *triangles, uint32_t triangleCount,
                           const CudaSphere *spheres, uint32_t sphereCount,
                           const CudaPrimitive *primitives, uint32_t primitiveCount,
                           const CudaBvhNode *bvh, uint32_t bvhCount,
                           const CudaMaterial *materials, float totalEmitterArea,
                           CudaVec3 background,
                           CudaVec3 *framebuffer)
{
    (void)triangleCount; (void)sphereCount;
    const dim3 block(8, 8);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    if (width && height)
        pathTracingKernel<<<grid, block>>>(width, height, fov, eye, sampleIndex,
            maxDepth, russianRoulette, triangles, spheres, primitives, primitiveCount,
            bvh, bvhCount, materials, totalEmitterArea, background, framebuffer);
    cudaDeviceSynchronize();
}

__global__ void diffuseKernel(const CudaRay *rays, uint32_t count,
                              const CudaTriangle *triangles, const CudaSphere *spheres,
                              const CudaPrimitive *primitives, uint32_t primitiveCount,
                              const CudaBvhNode *bvh, uint32_t bvhCount,
                              const CudaMaterial *materials, CudaVec3 lightPosition,
                              CudaVec3 lightEmission, CudaVec3 background,
                              uint32_t sampleIndex, CudaVec3 *framebuffer)
{
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count)return;
    CudaHit h=intersectBvh(rays[i],triangles,spheres,primitives,bvh,bvhCount); CudaVec3 result=background;
    if(h.hit){CudaVec3 toLight=lightPosition-h.position;float d2=cudaLength2(toLight),d=sqrtf(d2);if(d>0){CudaVec3 wi=toLight/d;float cosine=fmaxf(0.0f,cudaDot(h.normal,wi));CudaRay shadow{h.position+h.normal*1e-3f,wi,wi,1e-3f,d-2e-3f};CudaHit blocker=intersectBvh(shadow,triangles,spheres,primitives,bvh,bvhCount);if(!blocker.hit){CudaVec3 kd=materials[h.materialIndex].diffuse;result=lightEmission*kd*(cosine/(3.14159265f*d2));}}CudaVec3 bounce=cosineSample(h.normal,i+sampleIndex*9781u);(void)bounce;}
    framebuffer[i]=framebuffer[i]+result;
}

void launchCudaDiffuseBounce(const CudaRay *rays, uint32_t width, uint32_t height,
                             const CudaTriangle *triangles, uint32_t triangleCount,
                             const CudaSphere *spheres, uint32_t sphereCount,
                             const CudaPrimitive *primitives, uint32_t primitiveCount,
                             const CudaBvhNode *bvh, uint32_t bvhCount,
                             const CudaMaterial *materials, CudaVec3 lightPosition,
                             CudaVec3 lightEmission, CudaVec3 background,
                             uint32_t sampleIndex, CudaVec3 *framebuffer)
{
    (void)triangleCount; (void)sphereCount;
    uint32_t count=width*height; if(count) diffuseKernel<<<(count+127)/128,128>>>(rays,count,triangles,spheres,primitives,primitiveCount,bvh,bvhCount,materials,lightPosition,lightEmission,background,sampleIndex,framebuffer);
}