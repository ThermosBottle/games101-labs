//
// Created by Göksu Güvendiren on 2019-05-14.
//

#include "Scene.hpp"

void Scene::buildBVH()
{
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);

    emitters.clear();
    emitterAreaCdf.clear();
    totalEmitterArea = 0.0f;
    for (Object* object : objects)
    {
        if (object->hasEmit())
        {
            totalEmitterArea += object->getArea();
            emitters.push_back(object);
            emitterAreaCdf.push_back(totalEmitterArea);
        }
    }
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray);
}

void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    if (emitters.empty() || totalEmitterArea <= 0.0f)
    {
        pdf = 0.0f;
        return;
    }
    const float p = get_random_float() * totalEmitterArea;
    const auto it = std::lower_bound(emitterAreaCdf.begin(), emitterAreaCdf.end(), p);
    const size_t index = static_cast<size_t>(it - emitterAreaCdf.begin());
    emitters[std::min(index, emitters.size() - 1)]->Sample(pos, pdf);
    // The emitter is selected with probability area / totalEmitterArea and
    // the point on that emitter is sampled with density 1 / area. Their
    // product is therefore 1 / totalEmitterArea. Keeping this PDF explicit
    // makes the estimator correct when more than one light is present.
    pdf = 1.0f / totalEmitterArea;
}

bool Scene::trace(
    const Ray &ray,
    const std::vector<Object *> &objects,
    float &tNear, uint32_t &index, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k)
    {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear)
        {
            *hitObject = objects[k];
            tNear = tNearK;
            index = indexK;
        }
    }

    return (*hitObject != nullptr);
}

// Implementation of Path Tracing
Vector3f Scene::castRay(const Ray &ray, int depth) const
{
    // TODO Implement Path Tracing Algorithm here
    Vector3f L_dir(0, 0, 0);
    Intersection intersection = intersect(ray);
    if (!intersection.happened)
    {
        return L_dir;
    }
    if (intersection.m->hasEmission())
        return intersection.m->getEmission();
    // Stop only after checking emission, so a light hit at the final allowed
    // bounce is still visible. This prevents unbounded recursion and noise.
    if (depth >= maxDepth)
        return L_dir;

    Intersection pos;
    float pdf_light;
    sampleLight(pos, pdf_light);
    Vector3f hitPoint = intersection.coords;
    Vector3f N = intersection.normal;
    // An invalid direct-light sample only invalidates direct lighting. Do not
    // return here: the indirect bounce below is still a valid contribution.
    const bool validLightSample =
        std::isfinite(pdf_light) && pdf_light > 1e-7f &&
        std::isfinite(pos.coords.x) && std::isfinite(pos.coords.y) &&
        std::isfinite(pos.coords.z);
    if (validLightSample)
    {
        const Vector3f toLight = pos.coords - intersection.coords;
        const float lightDistance = std::sqrt(dotProduct(toLight, toLight));
        if (std::isfinite(lightDistance) && lightDistance > EPSILON)
        {
            const Vector3f lightDir = toLight / lightDistance;
            const float cosSurface = dotProduct(lightDir, N);
            const float cosLight = dotProduct(-lightDir, pos.normal);
            if (std::isfinite(cosSurface) && std::isfinite(cosLight) &&
                cosSurface > 0.0f && cosLight > 0.0f)
            {
                const Vector3f shadowOrig = hitPoint + N * EPSILON;
                Ray shadowRay(shadowOrig, lightDir);
                // Test only the finite segment from the surface to the
                // sampled light. The light itself is therefore not treated
                // as an occluder, and objects behind it cannot shadow it.
                shadowRay.t_min = EPSILON;
                shadowRay.t_max = std::max(
                    static_cast<double>(EPSILON),
                    static_cast<double>(lightDistance) - 4.0 * EPSILON);
                const Intersection shadowIntersection = intersect(shadowRay);
                if (!shadowIntersection.happened)
                {
                    const Vector3f direct =
                        pos.emit * intersection.m->eval(ray.direction, lightDir, N) *
                        cosSurface * cosLight /
                        (lightDistance * lightDistance) / pdf_light;
                    if (std::isfinite(direct.x) && std::isfinite(direct.y) &&
                        std::isfinite(direct.z))
                        L_dir += direct;
                }
            }
        }
    }

    // Do not roulette the first few bounces: those paths carry most of the
    // useful image energy. Russian roulette is unbiased after reweighting,
    // but applying it too early increases visible variance.
    const float continuationProbability = depth >= 3 ? RussianRoulette : 1.0f;
    if (get_random_float() > continuationProbability)
        return L_dir;
    Vector3f L_indir(0, 0, 0);
    Vector3f wi = intersection.m->sample(ray.direction, N);
    Vector3f newOrig = dotProduct(wi, N) < 0 ? hitPoint - N * EPSILON : hitPoint + N * EPSILON;

    Ray indirRay(newOrig, wi);
    float pdf = intersection.m->pdf(ray.direction, wi, N);
    const float cosIndirect = dotProduct(wi, N);
    if (std::isfinite(pdf) && pdf > 1e-7f && std::isfinite(cosIndirect) &&
        cosIndirect > 0.0f && continuationProbability > 1e-7f)
    {
        const Vector3f indirect = castRay(Ray(newOrig, wi), depth + 1) *
                                   intersection.m->eval(ray.direction, wi, N) *
                                   cosIndirect / pdf / continuationProbability;
        if (std::isfinite(indirect.x) && std::isfinite(indirect.y) &&
            std::isfinite(indirect.z))
            L_indir += indirect;
    }
    const Vector3f result = L_dir + L_indir;
    return (std::isfinite(result.x) && std::isfinite(result.y) &&
            std::isfinite(result.z)) ? result : Vector3f();
}