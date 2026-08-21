//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>
class Texture
{
private:
    cv::Mat image_data;

public:
    Texture(const std::string &name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    Eigen::Vector3f getColorBilinear(float u, float v)
    {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;

        int x = floor(u_img);
        int y = floor(v_img);

        float u_ratio = u_img - x;
        float v_ratio = v_img - y;
        float u_opposite = 1 - u_ratio;
        float v_opposite = 1 - v_ratio;

        Eigen::Vector3f result(0, 0, 0);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
                int x_idx = std::min(std::max(x + i, 0), width - 1);
                int y_idx = std::min(std::max(y + j, 0), height - 1);
                auto color = image_data.at<cv::Vec3b>(y_idx, x_idx);
                Eigen::Vector3f color_vec(color[0], color[1], color[2]);
                result += color_vec * ((i == 0) ? u_opposite : u_ratio) * ((j == 0) ? v_opposite : v_ratio);
            }
        return result;
    }
};
#endif // RASTERIZER_TEXTURE_H
