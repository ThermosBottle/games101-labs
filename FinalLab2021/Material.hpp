//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_MATERIAL_H
#define RAYTRACING_MATERIAL_H

#include "Vector.hpp"

enum MaterialType
{
    DIFFUSE,
    MICROFACET,
    MIRROR,
    GLASS
};

class Material
{
private:
    // Compute reflection direction
    Vector3f reflect(const Vector3f &I, const Vector3f &N) const
    {
        return I - 2 * dotProduct(I, N) * N;
    }

    // Compute refraction direction using Snell's law
    //
    // We need to handle with care the two possible situations:
    //
    //    - When the ray is inside the object
    //
    //    - When the ray is outside.
    //
    // If the ray is outside, you need to make cosi positive cosi = -N.I
    //
    // If the ray is inside, you need to invert the refractive indices and negate the normal N
    Vector3f refract(const Vector3f &I, const Vector3f &N, const float &ior) const
    {
        float cosi = clamp(-1, 1, dotProduct(I, N));
        float etai = 1, etat = ior;
        Vector3f n = N;
        if (cosi < 0)
        {
            cosi = -cosi;
        }
        else
        {
            std::swap(etai, etat);
            n = -N;
        }
        float eta = etai / etat;
        float k = 1 - eta * eta * (1 - cosi * cosi);
        return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
    }

    // Compute Fresnel equation
    //
    // \param I is the incident view direction
    //
    // \param N is the normal at the intersection point
    //
    // \param ior is the material refractive index
    //
    // \param[out] kr is the amount of light reflected
    void fresnel(const Vector3f &I, const Vector3f &N, const float &ior, float &kr) const
    {
        float cosi = clamp(-1, 1, dotProduct(I, N));
        float etai = 1, etat = ior;
        if (cosi > 0)
        {
            std::swap(etai, etat);
        }
        // Compute sini using Snell's law
        float sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi));
        // Total internal reflection
        if (sint >= 1)
        {
            kr = 1;
        }
        else
        {
            float cost = sqrtf(std::max(0.f, 1 - sint * sint));
            cosi = fabsf(cosi);
            float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
            float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
            kr = (Rs * Rs + Rp * Rp) / 2;
        }
        // As a consequence of the conservation of energy, transmittance is given by:
        // kt = 1 - kr;
    }

    Vector3f toWorld(const Vector3f &a, const Vector3f &N)
    {
        Vector3f B, C;
        if (std::fabs(N.x) > std::fabs(N.y))
        {
            float invLen = 1.0f / std::sqrt(N.x * N.x + N.z * N.z);
            C = Vector3f(N.z * invLen, 0.0f, -N.x * invLen);
        }
        else
        {
            float invLen = 1.0f / std::sqrt(N.y * N.y + N.z * N.z);
            C = Vector3f(0.0f, N.z * invLen, -N.y * invLen);
        }
        B = crossProduct(C, N);
        return a.x * B + a.y * C + a.z * N;
    }

    // GGX/Trowbridge-Reitz normal distribution function. The half vector H
    // describes the orientation of the microscopic mirror facets.
    float distributionGGX(const Vector3f &N, const Vector3f &H) const
    {
        const float NdotH = std::max(0.0f, dotProduct(N, H));
        const float alpha = std::max(0.001f, roughness * roughness);
        const float alpha2 = alpha * alpha;
        const float denominator = NdotH * NdotH * (alpha2 - 1.0f) + 1.0f;
        return alpha2 / (M_PI * denominator * denominator);
    }

    // Schlick-GGX approximation of the Smith masking-shadowing term for one
    // direction. It models the fact that a facet can be hidden from either
    // the viewer or the light.
    float geometrySchlickGGX(float NdotV) const
    {
        const float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
        return NdotV / (NdotV * (1.0f - k) + k);
    }

    float geometrySmith(const Vector3f &N,
                        const Vector3f &V,
                        const Vector3f &L) const
    {
        return geometrySchlickGGX(std::max(0.0f, dotProduct(N, V))) *
               geometrySchlickGGX(std::max(0.0f, dotProduct(N, L)));
    }

    // Schlick's approximation gives the angle-dependent Fresnel reflectance.
    Vector3f fresnelSchlick(float cosTheta) const
    {
        const float factor = std::pow(1.0f - clamp(0.0f, 1.0f, cosTheta), 5.0f);
        return Ks + (Vector3f(1.0f) - Ks) * factor;
    }

public:
    MaterialType m_type;
    // Vector3f m_color;
    Vector3f m_emission;
    float ior;
    Vector3f Kd, Ks;
    float specularExponent;
    // GGX roughness: low values produce a sharp highlight, high values a
    // broad highlight. This is independent of the legacy exponent field.
    float roughness;
    // Texture tex;

    inline Material(MaterialType t = DIFFUSE, Vector3f e = Vector3f(0, 0, 0));
    inline MaterialType getType();
    // inline Vector3f getColor();
    inline Vector3f getColorAt(double u, double v);
    inline Vector3f getEmission();
    inline bool hasEmission();

    // sample a ray by Material properties
    inline Vector3f sample(const Vector3f &wi, const Vector3f &N);
    // given a ray, calculate the PdF of this ray
    inline float pdf(const Vector3f &wi, const Vector3f &wo, const Vector3f &N);
    // given a ray, calculate the contribution of this ray
    inline Vector3f eval(const Vector3f &wi, const Vector3f &wo, const Vector3f &N);
};

Material::Material(MaterialType t, Vector3f e)
{
    m_type = t;
    // m_color = c;
    m_emission = e;
    ior = 1.5f;
    Kd = Vector3f(0.5f);
    Ks = Vector3f(0.04f);
    specularExponent = 32.0f;
    roughness = 0.5f;
}

MaterialType Material::getType() { return m_type; }
/// Vector3f Material::getColor(){return m_color;}
Vector3f Material::getEmission() { return m_emission; }
bool Material::hasEmission()
{
    if (m_emission.norm() > EPSILON)
        return true;
    else
        return false;
}

Vector3f Material::getColorAt(double u, double v)
{
    return Vector3f();
}

Vector3f Material::sample(const Vector3f &wi, const Vector3f &N)
{
    switch (m_type)
    {
    case DIFFUSE:
    {
        // Diffuse materials continue to use cosine-weighted hemisphere
        // sampling. Its PDF is kept in the DIFFUSE branch below.
        const float x_1 = get_random_float();
        const float x_2 = get_random_float();
        const float z = std::sqrt(1.0f - x_1);
        const float r = std::sqrt(x_1);
        const float phi = 2.0f * M_PI * x_2;
        Vector3f localRay(r * std::cos(phi), r * std::sin(phi), z);
        return toWorld(localRay, N);

        break;
    }
    case MICROFACET:
    {
        // GGX importance sampling: sample the microfacet half-vector H
        // instead of sampling the outgoing direction uniformly/cosine-wise.
        // wi points from the previous ray origin toward the surface, so the
        // conventional surface-to-view direction is V = -wi.
        const Vector3f V = normalize(-wi);
        const float NdotV = dotProduct(N, V);
        if (NdotV <= 0.0f)
            return Vector3f();

        const float alpha = std::max(0.001f, roughness * roughness);
        const float x_1 = get_random_float();
        const float x_2 = get_random_float();
        const float phi = 2.0f * M_PI * x_2;

        // Invert the GGX NDF CDF to sample H in the local tangent frame.
        const float tanTheta2 = alpha * alpha * x_1 /
                                std::max(1.0f - x_1, 1e-7f);
        const float cosTheta = 1.0f / std::sqrt(1.0f + tanTheta2);
        const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
        const Vector3f localH(
            sinTheta * std::cos(phi),
            sinTheta * std::sin(phi),
            cosTheta);
        const Vector3f H = normalize(toWorld(localH, N));

        // Reflect the incident direction -V about H to obtain the outgoing
        // direction. If H is sampled outside the visible reflection domain,
        // the path has zero contribution and will be rejected by the caller.
        return normalize(reflect(-V, H));
    }
    }
    // Keep unsupported future material types well-defined instead of
    // returning an uninitialized value.
    return Vector3f();
}

float Material::pdf(const Vector3f &wi, const Vector3f &wo, const Vector3f &N)
{
    switch (m_type)
    {
    case DIFFUSE:
    {
        // PDF for the cosine-weighted diffuse sampler above.
        const float cosTheta = dotProduct(wo, N);
        return cosTheta > 0.0f ? cosTheta / M_PI : 0.0f;
        break;
    }
    case MICROFACET:
    {
        // The half-vector sampler has density D(H) * (N dot H). Mapping
        // from H to wo introduces the Jacobian 1 / (4 * |V dot H|).
        // This PDF must match sample() for the estimator to remain unbiased.
        const Vector3f V = normalize(-wi);
        const Vector3f L = normalize(wo);
        const Vector3f H = normalize(V + L);
        const float NdotH = dotProduct(N, H);
        const float VdotH = std::fabs(dotProduct(V, H));
        const float NdotL = dotProduct(N, L);
        if (NdotH <= 0.0f || VdotH <= 1e-7f || NdotL <= 0.0f)
            return 0.0f;

        return distributionGGX(N, H) * NdotH / (4.0f * VdotH);
    }
    }
    return 0.0f;
}

Vector3f Material::eval(const Vector3f &wi, const Vector3f &wo, const Vector3f &N)
{
    switch (m_type)
    {
    case DIFFUSE:
    {
        // calculate the contribution of diffuse   model
        float cosalpha = dotProduct(N, wo);
        if (cosalpha > 0.0f)
        {
            Vector3f diffuse = Kd / M_PI;
            return diffuse;
        }
        else
            return Vector3f(0.0f);
        break;
    }
    case MICROFACET:
    {
        // The path tracer passes wi as the direction of the incoming ray
        // (from the camera/path origin toward the surface). Convert it to
        // the conventional surface-to-view direction before evaluating
        // the Cook-Torrance BRDF.
        const Vector3f V = normalize(-wi);
        const Vector3f L = normalize(wo);
        const float NdotV = std::max(0.0f, dotProduct(N, V));
        const float NdotL = std::max(0.0f, dotProduct(N, L));
        if (NdotV <= 0.0f || NdotL <= 0.0f)
            return Vector3f();

        const Vector3f H = normalize(V + L);
        const float VdotH = std::max(0.0f, dotProduct(V, H));
        const float D = distributionGGX(N, H);
        const float G = geometrySmith(N, V, L);
        const Vector3f F = fresnelSchlick(VdotH);
        const float denominator = 4.0f * NdotV * NdotL;

        // Cook-Torrance microfacet BRDF: D is the facet distribution,
        // G is masking/shadowing, and F is Fresnel reflectance.
        const Vector3f specular = F * (D * G / denominator);

        // Add the diffuse base while avoiding double-counting reflected
        // energy: energy that is reflected by the specular lobe is removed
        // from the diffuse lobe using (1 - F).
        const Vector3f diffuse = (Vector3f(1.0f) - F) * (Kd / M_PI);
        const Vector3f result = diffuse + specular;
        return (std::isfinite(result.x) && std::isfinite(result.y) &&
                std::isfinite(result.z))
                   ? result
                   : Vector3f();
    }
    }
    return Vector3f();
}

#endif // RAYTRACING_MATERIAL_H
