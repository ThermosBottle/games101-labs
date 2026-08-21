#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> control_points;
void rasterize_line(const cv::Point2f &p, cv::Mat &window);

void mouse_handler(int event, int x, int y, int flags, void *userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN && control_points.size() < 4)
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", "
                  << y << ")" << '\n';
        control_points.emplace_back(x, y);
    }
}

void naive_bezier(const std::vector<cv::Point2f> &points, cv::Mat &window)
{
    auto &p_0 = points[0];
    auto &p_1 = points[1];
    auto &p_2 = points[2];
    auto &p_3 = points[3];

    for (double t = 0.0; t <= 1.0; t += 0.001)
    {
        auto point = std::pow(1 - t, 3) * p_0 + 3 * t * std::pow(1 - t, 2) * p_1 +
                     3 * std::pow(t, 2) * (1 - t) * p_2 + std::pow(t, 3) * p_3;

        window.at<cv::Vec3b>(point.y, point.x)[2] = 255;
    }
}

cv::Point2f recursive_bezier(const std::vector<cv::Point2f> &control_points, float t)
{
    // TODO: Implement de Casteljau's algorithm
    if (control_points.size() == 1)
    {
        return control_points[0];
    }
    else
    {
        std::vector<cv::Point2f> next_points;
        for (size_t i = 0; i < control_points.size() - 1; ++i)
        {
            cv::Point2f p = (1 - t) * control_points[i] + t * control_points[i + 1];
            next_points.push_back(p);
        }
        return recursive_bezier(next_points, t);
    }
}

void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window)
{
    // TODO: Iterate through all t = 0 to t = 1 with small steps, and call de Casteljau's
    // recursive Bezier algorithm.
    for (double t = 0.0; t <= 1.0; t += 0.001)
    {
        cv::Point2f point = recursive_bezier(control_points, t);
        rasterize_line(point, window);
        // window.at<cv::Vec3b>(point.y, point.x)[1] = 255;
    }
}

// Antialiased line rasterization function
void rasterize_line(const cv::Point2f &p, cv::Mat &window)
{
    cv::Point2f p0 = cv::Point2f(std::floor(p.x), std::floor(p.y)) + cv::Point2f(0.5, 0.5);
    cv::Point2f p1 = p0 + cv::Point2f(0, 1);
    cv::Point2f p2 = p0 + cv::Point2f(1, 0);
    cv::Point2f p3 = p0 + cv::Point2f(1, 1);

    float max_dist = 1.0f * std::sqrt(2.0f);

    std::function<float(const cv::Point2f &, const cv::Point2f &)> distance = [](const cv::Point2f &a, const cv::Point2f &b)
    {
        return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
    };

    float d0 = distance(p0, p);
    float d1 = distance(p1, p);
    float d2 = distance(p2, p);
    float d3 = distance(p3, p);

    window.at<cv::Vec3b>(p0.y, p0.x)[1] = std::min(255.0f, window.at<cv::Vec3b>(p0.y, p0.x)[1] + (255.0f * std::max(0.0f, 1.0f - d0 / max_dist)));
    window.at<cv::Vec3b>(p1.y, p1.x)[1] = std::min(255.0f, window.at<cv::Vec3b>(p1.y, p1.x)[1] + (255.0f * std::max(0.0f, 1.0f - d1 / max_dist)));
    window.at<cv::Vec3b>(p2.y, p2.x)[1] = std::min(255.0f, window.at<cv::Vec3b>(p2.y, p2.x)[1] + (255.0f * std::max(0.0f, 1.0f - d2 / max_dist)));
    window.at<cv::Vec3b>(p3.y, p3.x)[1] = std::min(255.0f, window.at<cv::Vec3b>(p3.y, p3.x)[1] + (255.0f * std::max(0.0f, 1.0f - d3 / max_dist)));
}

int main()
{
    cv::Mat window = cv::Mat(700, 700, CV_8UC3, cv::Scalar(0));
    cv::cvtColor(window, window, cv::COLOR_BGR2RGB);
    cv::namedWindow("Bezier Curve", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Bezier Curve", mouse_handler, nullptr);

    int key = -1;
    while (key != 27)
    {
        for (auto &point : control_points)
        {
            cv::circle(window, point, 3, {255, 255, 255}, 3);
        }

        if (control_points.size() == 4)
        {
            // naive_bezier(control_points, window);
            bezier(control_points, window);

            cv::imshow("Bezier Curve", window);
            cv::imwrite("my_bezier_curve.png", window);
            key = cv::waitKey(0);

            return 0;
        }

        cv::imshow("Bezier Curve", window);
        key = cv::waitKey(20);
    }

    return 0;
}
