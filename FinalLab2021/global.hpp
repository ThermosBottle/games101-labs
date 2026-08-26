#pragma once
#include <iostream>
#include <cmath>
#include <random>
#include <thread>
#include <chrono>

#undef M_PI
#define M_PI 3.141592653589793f

extern const float  EPSILON;
const float kInfinity = std::numeric_limits<float>::max();

inline float clamp(const float &lo, const float &hi, const float &v)
{ return std::max(lo, std::min(hi, v)); }

inline  bool solveQuadratic(const float &a, const float &b, const float &c, float &x0, float &x1)
{
    float discr = b * b - 4 * a * c;
    if (discr < 0) return false;
    else if (discr == 0) x0 = x1 = - 0.5 * b / a;
    else {
        float q = (b > 0) ?
                  -0.5 * (b + sqrt(discr)) :
                  -0.5 * (b - sqrt(discr));
        x0 = q / a;
        x1 = c / q;
    }
    if (x0 > x1) std::swap(x0, x1);
    return true;
}

inline float get_random_float()
{
    // One generator per worker avoids constructing random_device and mt19937
    // for every sample and keeps random state out of shared memory.
    thread_local std::mt19937 rng([] {
        std::random_device dev;
        const auto clockSeed = static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        const auto threadSeed = static_cast<unsigned int>(
            std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::seed_seq seed{dev(), clockSeed, threadSeed};
        return std::mt19937(seed);
    }());
    thread_local std::uniform_real_distribution<float> dist(0.f, 1.f);
    return dist(rng);
}

inline void UpdateProgress(float progress)
{
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
};
