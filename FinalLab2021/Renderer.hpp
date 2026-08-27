//
// Created by goksu on 2/25/20.
//
#include "Scene.hpp"
#include <string>

#pragma once
struct RenderStats
{
    double renderMs = 0.0;
    double outputMs = 0.0;
};

struct hit_payload
{
    float tNear;
    uint32_t index;
    Vector2f uv;
    Object *hit_obj;
};

class Renderer
{
public:
    RenderStats Render(const Scene &scene, const int spp, const std::string &method,
                       const std::string &outputName = std::string());

private:
};
