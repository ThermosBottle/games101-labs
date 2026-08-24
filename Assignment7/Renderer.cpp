#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include "Scene.hpp"
#include "Renderer.hpp"

inline float deg2rad(const float& deg) { return deg * M_PI / 180.0; }
const float EPSILON = 0.00001;

void Renderer::Render(const Scene& scene)
{
    std::vector<Vector3f> framebuffer(scene.width * scene.height);
    const float scale = tan(deg2rad(scene.fov * 0.5));
    const float imageAspectRatio = scene.width / static_cast<float>(scene.height);
    const Vector3f eye_pos(278, 273, -800);
    const int spp = 16;
    std::cout << "SPP: " << spp << "\n";

    std::atomic<uint32_t> nextRow{0};
    std::atomic<uint32_t> completedRows{0};
    const unsigned int hardwareThreads = std::thread::hardware_concurrency();
    const unsigned int workerCount = std::max(1u, std::min(
        hardwareThreads == 0 ? 1u : hardwareThreads,
        static_cast<unsigned int>(scene.height)));
    std::vector<std::thread> workers;
    workers.reserve(workerCount);

    auto renderRows = [&]() {
        uint32_t j;
        while ((j = nextRow.fetch_add(1, std::memory_order_relaxed)) <
               static_cast<uint32_t>(scene.height)) {
            for (uint32_t i = 0; i < static_cast<uint32_t>(scene.width); ++i) {
                const float x = (2 * (i + 0.5f) / scene.width - 1) * imageAspectRatio * scale;
                const float y = (1 - 2 * (j + 0.5f) / scene.height) * scale;
                const Vector3f dir = normalize(Vector3f(-x, y, 1));
                Vector3f color;
                for (int k = 0; k < spp; ++k)
                    color += scene.castRay(Ray(eye_pos, dir), 0);
                framebuffer[j * scene.width + i] = color / static_cast<float>(spp);
            }
            completedRows.fetch_add(1, std::memory_order_relaxed);
        }
    };

    for (unsigned int i = 0; i < workerCount; ++i)
        workers.emplace_back(renderRows);

    while (completedRows.load(std::memory_order_relaxed) < static_cast<uint32_t>(scene.height)) {
        UpdateProgress(completedRows.load(std::memory_order_relaxed) /
                       static_cast<float>(scene.height));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    for (auto& worker : workers)
        worker.join();
    UpdateProgress(1.f);

    FILE* fp = fopen("binary.ppm", "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        unsigned char color[3];
        color[0] = static_cast<unsigned char>(255 * std::pow(clamp(0, 1, framebuffer[i].x), 0.6f));
        color[1] = static_cast<unsigned char>(255 * std::pow(clamp(0, 1, framebuffer[i].y), 0.6f));
        color[2] = static_cast<unsigned char>(255 * std::pow(clamp(0, 1, framebuffer[i].z), 0.6f));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);
}