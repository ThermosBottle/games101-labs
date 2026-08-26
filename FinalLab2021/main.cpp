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
    Scene scene(784, 784);
    int spp = 16;
    if (argc == 2)

    {
        spp = atoi(argv[1]);
    }
    MaterialType type = DIFFUSE;
    std::string method = "diffuse";
    if (argc == 3)
    {
        spp = atoi(argv[1]);
        if (strcmp(argv[2], "diffuse") == 0)
            type = DIFFUSE;
        else if (strcmp(argv[2], "microfacet") == 0)
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

    Renderer r;

    auto start = std::chrono::system_clock::now();
    r.Render(scene, spp, method);
    auto stop = std::chrono::system_clock::now();

    std::cout << "Render complete: \n";
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::hours>(stop - start).count() << " hours\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::minutes>(stop - start).count() << " minutes\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count() << " seconds\n";

    return 0;
}