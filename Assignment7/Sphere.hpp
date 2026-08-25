//
// Created by LEI XU on 5/13/19.
//

#ifndef RAYTRACING_SPHERE_H
#define RAYTRACING_SPHERE_H

#include "Object.hpp"
#include "Vector.hpp"
#include "Bounds3.hpp"
#include "Material.hpp"

class Sphere : public Object{
public:
    // The Cornell Box scene is measured in world-space units where a very
    // small positive sphere hit is a numerical self-intersection. Ignore
    // such hits instead of allowing the ray to continue to the far root of
    // the same sphere. This prevents black self-shadowing points on the
    // sphere surface. The value is intentionally larger than EPSILON because
    // the current float intersection calculation can leave a visible gap.
    static constexpr float MIN_VALID_SPHERE_HIT = 0.5f;

    Vector3f center;
    float radius, radius2;
    Material *m;
    float area;
    Sphere(const Vector3f &c, const float &r, Material* mt = new Material()) : center(c), radius(r), radius2(r * r), m(mt), area(4 * M_PI *r *r) {}
    bool intersect(const Ray& ray) {
        // analytic solution
        Vector3f L = ray.origin - center;
        float a = dotProduct(ray.direction, ray.direction);
        float b = 2 * dotProduct(ray.direction, L);
        float c = dotProduct(L, L) - radius2;
        float t0, t1;
        if (!solveQuadratic(a, b, c, t0, t1)) return false;
        const bool originInside = dotProduct(L, L) < radius2;

        // For a ray starting just outside the sphere, t0 can be an
        // extremely close self-intersection. Do not switch to t1 in that
        // case: t1 is the far side of the same sphere and would incorrectly
        // shadow a visible light. A ray starting inside the sphere must use
        // t1, because t1 is its valid exit point.
        if (!originInside && t0 <= MIN_VALID_SPHERE_HIT)
            return false;
        const float t = originInside ? t1 : t0;
        return std::isfinite(t) && t > MIN_VALID_SPHERE_HIT &&
               t >= ray.t_min && t <= ray.t_max;
    }
    bool intersect(const Ray& ray, float &tnear, uint32_t &index) const
    {
        // analytic solution
        Vector3f L = ray.origin - center;
        float a = dotProduct(ray.direction, ray.direction);
        float b = 2 * dotProduct(ray.direction, L);
        float c = dotProduct(L, L) - radius2;
        float t0, t1;
        if (!
        solveQuadratic(a, b, c, t0, t1)) return false;
        const bool originInside = dotProduct(L, L) < radius2;

        // Ignore a near self-hit from the sphere surface instead of falling
        // through to the far root. This keeps BVH/visibility queries from
        // treating the back side of the same sphere as an occluder.
        if (!originInside && t0 <= MIN_VALID_SPHERE_HIT)
            return false;
        const float t = originInside ? t1 : t0;
        if (!std::isfinite(t) || t <= MIN_VALID_SPHERE_HIT ||
            t < ray.t_min || t > ray.t_max)
            return false;
        tnear = t;

        return true;
    }
    Intersection getIntersection(Ray ray){
        Intersection result;
        result.happened = false;
        Vector3f L = ray.origin - center;
        float a = dotProduct(ray.direction, ray.direction);
        float b = 2 * dotProduct(ray.direction, L);
        float c = dotProduct(L, L) - radius2;
        float t0, t1;
        if (!solveQuadratic(a, b, c, t0, t1)) return result;
        const bool originInside = dotProduct(L, L) < radius2;

        // A shadow ray is launched from hitPoint + normal * EPSILON. If it
        // immediately meets this sphere again, that root is numerical self
        // intersection, not a real blocker. Returning no hit is important:
        // selecting t1 here would incorrectly report the far side of the
        // sphere and make large parts of the sphere black.
        if (!originInside && t0 <= MIN_VALID_SPHERE_HIT)
            return result;
        const float t = originInside ? t1 : t0;
        if (!std::isfinite(t) || t <= MIN_VALID_SPHERE_HIT ||
            t < ray.t_min || t > ray.t_max)
            return result;
        result.happened=true;

        result.coords = Vector3f(ray.origin + ray.direction * t);
        result.normal = normalize(Vector3f(result.coords - center));
        result.m = this->m;
        result.obj = this;
        result.distance = t;
        return result;

    }
    void getSurfaceProperties(const Vector3f &P, const Vector3f &I, const uint32_t &index, const Vector2f &uv, Vector3f &N, Vector2f &st) const
    { N = normalize(P - center); }

    Vector3f evalDiffuseColor(const Vector2f &) const
    {
        // Assignment7 stores the diffuse albedo in Material::Kd.  The
        // current path tracer evaluates the BRDF directly, but Object still
        // requires this legacy color-query method to be well-defined.
        return m ? m->Kd : Vector3f(0.5f);
    }
    Bounds3 getBounds(){
        return Bounds3(Vector3f(center.x-radius, center.y-radius, center.z-radius),
                       Vector3f(center.x+radius, center.y+radius, center.z+radius));
    }
    void Sample(Intersection &pos, float &pdf){
        float theta = 2.0 * M_PI * get_random_float(), phi = M_PI * get_random_float();
        Vector3f dir(std::cos(phi), std::sin(phi)*std::cos(theta), std::sin(phi)*std::sin(theta));
        pos.coords = center + radius * dir;
        pos.normal = dir;
        pos.emit = m->getEmission();
        pdf = 1.0f / area;
    }
    float getArea(){
        return area;
    }
    bool hasEmit(){
        return m->hasEmission();
    }
};




#endif //RAYTRACING_SPHERE_H
