// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

const float rst::rasterizer::SSAA_OFFSET[4][2] = {
    { 0.375, 0.125 },
    { 0.875, 0.375 },
    { 0.125, 0.625 },
    { 0.625, 0.875 }
};


static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

static bool insideTriangle(float x, float y, const Vector3f* _v)
{
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    std::tuple<float, float, float> barycentric = computeBarycentric2D(x, y, _v);
    return std::get<0>(barycentric) >= 0 && std::get<1>(barycentric) >= 0 && std::get<2>(barycentric) >= 0;
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            // vert.x() = 0.5*width*(vert.x()+1.0);
            // Flip Y here so screen Y goes downwards (OpenCV expects top-left origin)
            vert.x() = 0.5*width*(1.0 - vert.x());
            vert.y() = 0.5*height*(1.0 - vert.y());
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
        }


        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    int N = 4; // SSAA 4x

    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle
    for (int x = std::floor(std::min({v[0].x(), v[1].x(), v[2].x()})); x <= std::ceil(std::max({v[0].x(), v[1].x(), v[2].x()})); ++x) {
        for (int y = std::floor(std::min({v[0].y(), v[1].y(), v[2].y()})); y <= std::ceil(std::max({v[0].y(), v[1].y(), v[2].y()})); ++y) {
            for (int sample_i = 0; sample_i < N; ++sample_i) {
                float x_sample = x + SSAA_OFFSET[sample_i][0];
                float y_sample = y + SSAA_OFFSET[sample_i][1];
                int sample_index = get_sample_index(x, y, sample_i);

                if (insideTriangle(x_sample, y_sample, t.v)) {
                    auto [alpha, beta, gamma] = computeBarycentric2D(x_sample, y_sample, t.v);
                    float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;

                    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                    if (z_interpolated < sample_depth_buf[sample_index]) {
                        sample_depth_buf[sample_index] = z_interpolated;
                        sample_color_buf[sample_index] = t.getColor();
                    }
                }
            }

            Eigen::Vector3f resolved_color = Eigen::Vector3f::Zero();
            for (int sample_i = 0; sample_i < N; ++sample_i) {
                resolved_color += sample_color_buf[get_sample_index(x, y, sample_i)];
            }
            frame_buf[get_index(x, y)] = resolved_color / static_cast<float>(N);
        }
    }

}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(sample_color_buf.begin(), sample_color_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(sample_depth_buf.begin(), sample_depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    sample_color_buf.resize(w * h * 4, Eigen::Vector3f{0, 0, 0});
    sample_depth_buf.resize(w * h * 4); // SSAA 4x
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

// 4x SSAA
// Screen: bottom-left origin
// Sample: top-left origin
int rst::rasterizer::get_sample_index(int sample_x, int sample_y, int sample_i)
{
    return ((height-1-sample_y)*width + sample_x) * 4 + sample_i;
}


void rst::rasterizer::set_pixel(const int x, const int y, const Eigen::Vector3f& color, const float coverage_ratio)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-y)*width + x;
    frame_buf[ind] = color*coverage_ratio + frame_buf[ind]*(1-coverage_ratio);

}

// 下标:  0          1          2          3        ← y=2 (屏幕顶行)
//       (R,G,B)    (R,G,B)    (R,G,B)    (R,G,B)

// 下标:  4          5          6          7        ← y=1
//       (R,G,B)    (R,G,B)    (R,G,B)    (R,G,B)

// 下标:  8          9          10         11       ← y=0 (屏幕底行)
//       (R,G,B)    (R,G,B)    (R,G,B)    (R,G,B)

// clang-format on