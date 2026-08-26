#include "CudaKernels.cuh"
#include "CudaScene.hpp"
#include "Scene.hpp"
#include "Triangle.hpp"
#include "Sphere.hpp"
#include "Material.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// global.hpp declares this symbol for the CPU geometry helpers included by
// the scene-loading path.  The CUDA renderer does not link Renderer.cpp, so it
// provides the same scene-wide epsilon here.
const float EPSILON = 1e-3f;

namespace
{
void checkCuda(cudaError_t error, const char *operation)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

float byteColor(float value)
{
    value = fminf(1.0f, fmaxf(0.0f, value));
    return powf(value, 0.6f) * 255.0f;
}
}

int main(int argc, char **argv)
{
    // Usage: CudaPathTracing [spp] [width] [height] [maxDepth] [diffuse|microfacet] [output.ppm]
    const int spp = argc > 1 ? std::max(1, std::atoi(argv[1])) : 16;
    const int width = argc > 2 ? std::max(1, std::atoi(argv[2])) : 784;
    const int height = argc > 3 ? std::max(1, std::atoi(argv[3])) : 784;
    const int maxDepth = argc > 4 ? std::max(1, std::atoi(argv[4])) : 8;
    const std::string mode = argc > 5 ? argv[5] : "diffuse";
    const std::string output = argc > 6 ? argv[6] : "render-cuda-" + mode + ".ppm";
    if (mode != "diffuse" && mode != "microfacet")
    {
        std::cerr << "Invalid mode. Use 'diffuse' or 'microfacet'.\n";
        return 1;
    }
    const MaterialType materialType = mode == "microfacet" ? MICROFACET : DIFFUSE;

    try
    {
        Scene scene(width, height);
        Material *red = new Material(materialType, Vector3f(0.0f));
        red->Kd = Vector3f(0.63f, 0.065f, 0.05f);
        Material *green = new Material(materialType, Vector3f(0.0f));
        green->Kd = Vector3f(0.14f, 0.45f, 0.091f);
        Material *white = new Material(materialType, Vector3f(0.0f));
        white->Kd = Vector3f(0.725f, 0.71f, 0.68f);
        const Vector3f emission = 8.0f * Vector3f(0.747f + 0.058f,
            0.747f + 0.258f, 0.747f) + 15.6f * Vector3f(0.740f + 0.287f,
            0.740f + 0.160f, 0.740f) + 18.4f * Vector3f(0.737f + 0.642f,
            0.737f + 0.159f, 0.737f);
        Material *light = new Material(materialType, emission);
        light->Kd = Vector3f(0.65f);
        Material *sphereMaterial = new Material(materialType, Vector3f(0.0f));
        sphereMaterial->Kd = Vector3f(0.5f);
        sphereMaterial->Ks = Vector3f(0.04f);
        sphereMaterial->roughness = 0.25f;

        // Paths are relative to the project root, which is also where the
        // supplied models directory lives.  Run this executable from there.
        MeshTriangle floor("models/cornellbox/floor.obj", white);
        MeshTriangle shortbox("models/cornellbox/shortbox.obj", white);
        MeshTriangle tallbox("models/cornellbox/tallbox.obj", white);
        MeshTriangle left("models/cornellbox/left.obj", red);
        MeshTriangle right("models/cornellbox/right.obj", green);
        MeshTriangle lightMesh("models/cornellbox/light.obj", light);
        Sphere sphere(Vector3f(380.0f, 100.0f, 200.0f), 80.0f, sphereMaterial);
        scene.Add(&floor); scene.Add(&shortbox); scene.Add(&tallbox);
        scene.Add(&left); scene.Add(&right); scene.Add(&lightMesh); scene.Add(&sphere);
        scene.buildBVH();

        CudaScene cudaScene;
        cudaScene.upload(scene);
        const size_t pixelCount = static_cast<size_t>(width) * height;
        CudaVec3 *deviceFramebuffer = nullptr;
        checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceFramebuffer),
                             pixelCount * sizeof(CudaVec3)), "cudaMalloc framebuffer");
        checkCuda(cudaMemset(deviceFramebuffer, 0, pixelCount * sizeof(CudaVec3)),
                  "cudaMemset framebuffer");

        const CudaVec3 eye{278.0f, 273.0f, -800.0f};
        const CudaVec3 background{0.0f, 0.0f, 0.0f};
        for (int sample = 0; sample < spp; ++sample)
        {
            launchCudaPathTracing(width, height, static_cast<float>(scene.fov), eye,
                static_cast<uint32_t>(sample), static_cast<uint32_t>(maxDepth),
                scene.RussianRoulette, cudaScene.deviceTriangles(), cudaScene.triangleCount(),
                cudaScene.deviceSpheres(), cudaScene.sphereCount(), cudaScene.devicePrimitives(),
                static_cast<uint32_t>(cudaScene.primitiveCount()), cudaScene.deviceBvh(),
                static_cast<uint32_t>(cudaScene.bvh().size()), cudaScene.deviceMaterials(),
                cudaScene.totalEmitterArea(), background,
                deviceFramebuffer);
            if ((sample + 1) % std::max(1, spp / 10) == 0 || sample + 1 == spp)
                std::cout << "CUDA samples: " << sample + 1 << "/" << spp << "\n";
        }

        std::vector<CudaVec3> framebuffer(pixelCount);
        checkCuda(cudaMemcpy(framebuffer.data(), deviceFramebuffer,
                             pixelCount * sizeof(CudaVec3), cudaMemcpyDeviceToHost),
                  "cudaMemcpy framebuffer");
        cudaFree(deviceFramebuffer);

        FILE *file = std::fopen(output.c_str(), "wb");
        if (!file) throw std::runtime_error("cannot open output file: " + output);
        std::fprintf(file, "P6\n%d %d\n255\n", width, height);
        for (const CudaVec3 &sum : framebuffer)
        {
            const float scale = 1.0f / static_cast<float>(spp);
            const unsigned char pixel[3] = {
                static_cast<unsigned char>(byteColor(sum.x * scale)),
                static_cast<unsigned char>(byteColor(sum.y * scale)),
                static_cast<unsigned char>(byteColor(sum.z * scale))};
            std::fwrite(pixel, 1, 3, file);
        }
        std::fclose(file);
        std::cout << "CUDA render complete: " << output << "\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << "CUDA render failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}