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

    Intersection pos;
    float pdf_light;
    sampleLight(pos, pdf_light);
    if (pdf_light <= EPSILON)
        return L_dir;
    Vector3f lightDir = normalize(pos.coords - intersection.coords);
    float lightDistance = (pos.coords - intersection.coords).norm();
    Vector3f hitPoint = intersection.coords;
    Vector3f N = intersection.normal;
    Vector3f shadowOrig = dotProduct(lightDir, N) < 0 ? hitPoint - N * EPSILON : hitPoint + N * EPSILON;

    Ray shadowRay(shadowOrig, lightDir);
    Intersection shadowIntersection = intersect(shadowRay);

    if (shadowIntersection.happened && std::fabs(shadowIntersection.distance - lightDistance) < EPSILON)
    {
        // visible
        L_dir += pos.emit * intersection.m->eval(ray.direction, lightDir, N) * dotProduct(lightDir, N) * dotProduct(-lightDir, pos.normal) / powf(lightDistance, 2) / pdf_light;
    }

    if (get_random_float() > RussianRoulette)
        return L_dir;
    Vector3f L_indir(0, 0, 0);
    Vector3f wi = intersection.m->sample(ray.direction, N);
    Vector3f newOrig = dotProduct(wi, N) < 0 ? hitPoint - N * EPSILON : hitPoint + N * EPSILON;

    Ray indirRay(newOrig, wi);
    float pdf = intersection.m->pdf(ray.direction, wi, N);
    // to aviod division by zero
    if (pdf > EPSILON)
    {
        L_indir += castRay(Ray(newOrig, wi), depth + 1) * intersection.m->eval(ray.direction, wi, N) * dotProduct(wi, N) / pdf / RussianRoulette;
    }
    return L_dir + L_indir;
}