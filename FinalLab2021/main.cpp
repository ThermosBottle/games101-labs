#include "Renderer.hpp"
#include "Scene.hpp"
#include "Triangle.hpp"
#include "Sphere.hpp"
#include "Vector.hpp"
#include "global.hpp"
#include <chrono>
#include <string.h>
#include <string>

// In the main function of the program, we create the scene (create objects and
// lights) as well as set the options for the render (image width and height,
// maximum recursion depth, field-of-view, etc.). We then call the render
// function().
int main(int argc, char **argv)
{

    // Change the definition here to change resolution
    // Usage: RayTracing [spp] [width] [height] [maxDepth]
    //                     [diffuse|microfacet] [output.ppm]
    const int spp = argc > 1 ? std::max(1, atoi(argv[1])) : 16;
    const int width = argc > 2 ? std::max(1, atoi(argv[2])) : 784;
    const int height = argc > 3 ? std::max(1, atoi(argv[3])) : width;
    const int maxDepth = argc > 4 ? std::max(1, atoi(argv[4])) : 8;
    Scene scene(width, height);
    scene.maxDepth = maxDepth;
    MaterialType type = DIFFUSE;
    std::string method = "diffuse";
    if (argc > 5)
    {
        if (strcmp(argv[5], "diffuse") == 0)
            type = DIFFUSE;
        else if (strcmp(argv[5], "microfacet") == 0)
        {
            type = MICROFACET;
            method = "microfacet";
        }
        else
        {
            std::cerr << "Invalid material type. Use 'diffuse' or 'microfacet'.\n";
            return 1;
        }
    }

    const std::string output = argc > 6 ? argv[6] : std::string();
    const auto totalStart = std::chrono::steady_clock::now();
    Material *red = new Material(type, Vector3f(0.0f));
    red->Kd = Vector3f(0.63f, 0.065f, 0.05f);
    Material *green = new Material(type, Vector3f(0.0f));
    green->Kd = Vector3f(0.14f, 0.45f, 0.091f);
    Material *white = new Material(type, Vector3f(0.0f));
    white->Kd = Vector3f(0.725f, 0.71f, 0.68f);
    Material *light = new Material(type, (8.0f * Vector3f(0.747f + 0.058f, 0.747f + 0.258f, 0.747f) + 15.6f * Vector3f(0.740f + 0.287f, 0.740f + 0.160f, 0.740f) + 18.4f * Vector3f(0.737f + 0.642f, 0.737f + 0.159f, 0.737f)));
    light->Kd = Vector3f(0.65f);

    // Microfacet sphere used to validate the GGX Cook-Torrance BRDF. The
    // sphere uses a low diffuse albedo and a moderate roughness so that the
    // specular highlight is visible without being excessively sharp.
    Material *sphereMaterial = new Material(type, Vector3f(0.0f));
    // The diffuse base makes the sphere visibly illuminated across its
    // surface; the Microfacet specular term is then added on top of it.
    sphereMaterial->Kd = Vector3f(0.5f);
    sphereMaterial->Ks = Vector3f(0.04f); // dielectric normal-incidence F0
    sphereMaterial->roughness = 0.25f;

    MeshTriangle floor("../models/cornellbox/floor.obj", white);
    MeshTriangle shortbox("../models/cornellbox/shortbox.obj", white);
    MeshTriangle tallbox("../models/cornellbox/tallbox.obj", white);
    MeshTriangle left("../models/cornellbox/left.obj", red);
    MeshTriangle right("../models/cornellbox/right.obj", green);
    MeshTriangle light_("../models/cornellbox/light.obj", light);
    Sphere sphere(Vector3f(380.0f, 100.0f, 200.0f), 80.0f, sphereMaterial);

    scene.Add(&floor);
    scene.Add(&shortbox);
    scene.Add(&tallbox);
    scene.Add(&left);
    scene.Add(&right);
    scene.Add(&light_);
    scene.Add(&sphere);

    scene.buildBVH();
    const auto setupEnd = std::chrono::steady_clock::now();

    Renderer r;

    const RenderStats stats = r.Render(scene, spp, method, output);
    const auto totalEnd = std::chrono::steady_clock::now();
    const double setupMs = std::chrono::duration<double, std::milli>(setupEnd - totalStart).count();
    const double renderMs = stats.renderMs;
    const double totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    std::cout << "Render complete: \n";
    std::cout << "BENCHMARK cpu scene_setup_ms=" << setupMs
              << " render_ms=" << renderMs << " output_ms=" << stats.outputMs
              << " total_ms=" << totalMs << "\n";

    return 0;
}