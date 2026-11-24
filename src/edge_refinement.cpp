#include "segmentation.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr int inward = 9;
constexpr int before_profile = 19;
constexpr int after_profile = 15;

struct EdgeRay {
    cv::Point2f point;
    cv::Point2f outward;
    bool locked = false;
};

struct EdgeShift {
    int pixels = 0;
    int max_outward = 0;
    bool locked = false;
};

int cyclic(int i, int count) {
    return (i + count) % count;
}

std::vector<EdgeRay> makeRays(const std::vector<cv::Point>& points, const cv::Size& size,
                              const cv::Mat& uncertain) {
    const int count = static_cast<int>(points.size());
    std::vector<cv::Point2f> centers(count);
    const std::array<int, 7> weights{1, 2, 3, 4, 3, 2, 1};
    for (int i = 0; i < count; ++i) {
        for (int j = -3; j <= 3; ++j) {
            centers[i] += cv::Point2f(points[cyclic(i + j, count)]) * static_cast<float>(weights[j + 3]);
        }
        centers[i] *= 1.0f / 16.0f;
    }
    const float direction = cv::contourArea(points, true) < 0 ? 1.0f : -1.0f;
    std::vector<EdgeRay> rays(count);
    for (int i = 0; i < count; ++i) {
        const cv::Point& p = points[i];
        const auto tangent = centers[cyclic(i + 8, count)] - centers[cyclic(i - 8, count)];
        const float length = std::hypot(tangent.x, tangent.y);
        const bool at_border = p.x < 3 || p.y < 3 || p.x >= size.width - 3 || p.y >= size.height - 3;
        auto& ray = rays[i];
        const bool near_subtitle = !uncertain.empty() && uncertain.at<uchar>(p) != 0;
        ray.locked = at_border || near_subtitle || length < 1.0f;
        ray.point = ray.locked ? cv::Point2f(p) : centers[i];
        if (!ray.locked) {
            ray.outward = cv::Point2f(-tangent.y, tangent.x) * (direction / length);
        }
    }
    return rays;
}

float colorDistance(const cv::Vec3f& a, const cv::Vec3f& b) {
    const cv::Vec3f d = a - b;
    return std::sqrt(0.65f * d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}

float gray(const cv::Vec3f& bgr) {
    return bgr[0] * 0.114f + bgr[1] * 0.587f + bgr[2] * 0.299f;
}

EdgeShift chooseEdge(const EdgeRay& ray, const cv::Vec3f* profile, const uchar* occupancy,
                     const cv::Size& size, int radius, int scene_value, bool sole_zone, bool upper_zone) {
    if (ray.locked) {
        return {0, 0, true};
    }
    // 从轮廓往外走，遇到另一段前景就停止。狭窄的手臂间隙不能被两侧同时填上。
    int free_space = radius + 12;
    for (int offset = 4; offset < free_space; ++offset) {
        if (occupancy[offset] != 0) {
            free_space = offset;
            break;
        }
    }
    if (free_space < 6) {
        return {0, 0, true};
    }
    const bool narrow = free_space < radius + 12;
    int max_outward = narrow ? std::min(radius, (free_space - 2) / 2) : radius;
    const int outside = narrow ? free_space - 2 : radius + 7;
    const cv::Vec3f background = (profile[outside] + profile[outside - 1] + profile[outside - 2]) / 3.0f;
    const bool downward = sole_zone && ray.outward.y > 0.54f;
    // 瞳孔与虹膜之间也有强边缘。上部原边缘仍有身体颜色时，只允许轻微内收。
    const cv::Vec3f present = profile[0];
    const bool blue_at_edge = present[0] - present[2] > 0.14f * scene_value &&
                              present[0] - present[1] > 0.045f * scene_value;
    const int min_offset = upper_zone && !blue_at_edge ? -1 : -inward;
    if (sole_zone && ray.outward.y > 0.32f && !downward) {
        max_outward = std::min(max_outward, 13);
    }
    const float distance_cost = 0.25f + 0.0021f * scene_value;
    const float min_score = std::max(3.0f, 0.029f * scene_value);
    float best_edge = -std::numeric_limits<float>::infinity();
    float base_edge = 0.0f;
    float best_sole = -std::numeric_limits<float>::infinity();
    float base_sole = -std::numeric_limits<float>::infinity();
    int edge_shift = 0;
    int sole_shift = 0;
    for (int d = min_offset; d <= max_outward; ++d) {
        const cv::Point2f q = ray.point + ray.outward * static_cast<float>(d);
        if (q.x < 0 || q.y < 0 || q.x > size.width - 1 || q.y > size.height - 1) {
            continue;
        }
        if (!downward || (d >= -5 && d <= 5)) {
            const int end = narrow ? free_space - 1 : radius + 10;
            const cv::Vec3f inner = (profile[d - 5] + profile[d - 4] + profile[d - 3]) / 3.0f;
            const cv::Vec3f outer = (profile[std::min(d + 3, end)] + profile[std::min(d + 4, end)] +
                                     profile[std::min(d + 5, end)]) / 3.0f;
            const cv::Vec3f farther = (profile[std::min(d + 6, end)] + profile[std::min(d + 8, end)] +
                                       profile[std::min(d + 10, end)]) / 3.0f;
            // 真正外缘有色彩或亮度变化，外侧短距离内应趋近背景。这样可以跨过暗指尖。
            const float score = colorDistance(inner, outer) - 0.55f * colorDistance(outer, farther) -
                                0.58f * colorDistance(outer, background) - distance_cost * std::abs(d);
            if (score > best_edge) {
                best_edge = score;
                edge_shift = d;
            }
            if (d == 0) {
                base_edge = score;
            }
        }
        if (downward && d >= -4 && d <= 11 && (!narrow || d + 6 < free_space)) {
            const float dark = gray(profile[d]);
            const float drop = (gray(profile[d - 6]) + gray(profile[d - 5])) / 2.0f - dark;
            const float rise = (gray(profile[d + 5]) + gray(profile[d + 6])) / 2.0f - dark;
            if (drop >= std::max(3.0f, 0.023f * scene_value) &&
                rise >= std::max(2.0f, 0.014f * scene_value)) {
                // 鞋底方向优先寻找亮度谷底，避免把后面更强的倒影边缘当作轮廓。
                const float score = 1.4f * std::min(drop, rise) + 0.35f * std::max(drop, rise) -
                                    0.52f * std::abs(d);
                if (score > best_sole) {
                    best_sole = score;
                    sole_shift = d;
                }
                if (d == 0) {
                    base_sole = score;
                }
            }
        }
    }
    if (downward && best_sole >= min_score && best_sole - base_sole >= 1.5f) {
        return {sole_shift, std::min(max_outward, 11), false};
    }
    int shift = best_edge >= min_score && best_edge - base_edge >= 1.5f ? edge_shift : 0;
    if ((!sole_zone || ray.outward.y <= 0.32f) && best_edge >= min_score) {
        // 深色指尖边缘有一小段与蓝墙的混色。只在外侧仍明显暗于背景时保留它。
        const float background_gray = gray(background);
        const float contrast = background_gray - gray(profile[edge_shift - 3]);
        if (contrast > 0.20f * scene_value) {
            const float dark_limit = background_gray - std::max(0.14f * scene_value, 0.38f * contrast);
            const int limit = std::min(max_outward, shift + (sole_zone ? 2 : 4));
            while (shift < limit && gray(profile[shift + 1]) < dark_limit) {
                const auto q = ray.point + ray.outward * static_cast<float>(shift + 1);
                if (q.x < 0 || q.y < 0 || q.x > size.width - 1 || q.y > size.height - 1) {
                    break;
                }
                ++shift;
            }
        }
    }
    return {shift, downward ? std::min(max_outward, 5) : max_outward, false};
}

std::vector<cv::Point> alignedContour(const cv::Mat& smoothed, const cv::Mat& mask,
                                      const std::vector<cv::Point>& outline, const cv::Rect& box,
                                      int radius, int scene_value, bool feet_visible, const cv::Mat& uncertain) {
    const auto rays = makeRays(outline, mask.size(), uncertain);
    const int count = static_cast<int>(rays.size());
    const int samples = before_profile + radius + after_profile + 1;
    cv::Mat map_x(count, samples, CV_32F), map_y(count, samples, CV_32F);
    for (int i = 0; i < count; ++i) {
        float* cx = map_x.ptr<float>(i);
        float* cy = map_y.ptr<float>(i);
        for (int j = 0; j < samples; ++j) {
            const auto q = rays[i].point + rays[i].outward * static_cast<float>(j - before_profile);
            cx[j] = q.x;
            cy[j] = q.y;
        }
    }
    cv::Mat colors, occupied;
    cv::remap(smoothed, colors, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::remap(mask, occupied, map_x, map_y, cv::INTER_NEAREST, cv::BORDER_REPLICATE);

    const float sole_top = box.y + 0.87f * box.height;
    const float upper_bottom = box.y + 0.38f * box.height;
    std::vector<EdgeShift> shifts;
    shifts.reserve(count);
    for (int i = 0; i < count; ++i) {
        const bool sole_zone = feet_visible && rays[i].point.y >= sole_top;
        shifts.push_back(chooseEdge(rays[i], colors.ptr<cv::Vec3f>(i) + before_profile,
                                    occupied.ptr<uchar>(i) + before_profile, mask.size(),
                                    radius, scene_value, sole_zone, rays[i].point.y < upper_bottom));
    }
    // 一条孤立的剖面容易追到阴影和压缩块，用相邻剖面的中值压掉孤立跳变。
    std::vector<int> median(count);
    for (int i = 0; i < count; ++i) {
        std::array<int, 11> neighbors{};
        for (int j = -5; j <= 5; ++j) {
            neighbors[j + 5] = shifts[cyclic(i + j, count)].pixels;
        }
        std::nth_element(neighbors.begin(), neighbors.begin() + 5, neighbors.end());
        median[i] = neighbors[5];
    }
    std::vector<cv::Point> result(count);
    for (int i = 0; i < count; ++i) {
        if (shifts[i].locked) {
            result[i] = outline[i];
            continue;
        }
        float smooth_shift = 0.0f;
        for (int j = -4; j <= 4; ++j) {
            smooth_shift += (5 - std::abs(j)) * median[cyclic(i + j, count)];
        }
        const float offset = std::min(smooth_shift / 25.0f, static_cast<float>(shifts[i].max_outward));
        const auto q = rays[i].point + rays[i].outward * offset;
        result[i] = cv::Point(std::clamp(cvRound(q.x), 0, mask.cols - 1),
                             std::clamp(cvRound(q.y), 0, mask.rows - 1));
    }
    return result;
}

void repairDarkNotches(const cv::Mat& smoothed, cv::Mat& mask, const cv::Rect& box,
                       int radius, int scene_value, const cv::Mat& uncertain) {
    if (radius == 0) {
        return;
    }
    const int top = std::max(0, box.y - radius);
    const int bottom = std::min(mask.rows, box.y + box.height * 38 / 100 + radius);
    cv::Mat upper = mask.rowRange(top, bottom);
    const int kernel_size = 2 * radius + 1;
    cv::Mat closed, addition;
    cv::morphologyEx(upper, closed, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size)));
    // 纵向闭运算也能跨过眼边的短凹口，连接其上下两段原本连贯的边缘。
    cv::Mat vertical;
    cv::morphologyEx(upper, vertical, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, kernel_size)));
    cv::bitwise_or(closed, vertical, closed);
    cv::subtract(closed, upper, addition);
    const int cutoff = box.y + box.height * 38 / 100 - top;
    addition.rowRange(cutoff, addition.rows).setTo(0);
    if (!uncertain.empty()) {
        addition.setTo(0, uncertain.rowRange(top, bottom));
    }

    cv::Mat labels, stats, centroids;
    const int count = cv::connectedComponentsWithStats(addition, labels, stats, centroids, 8, CV_32S);
    for (int i = 1; i < count; ++i) {
        const int* row = stats.ptr<int>(i);
        const cv::Rect region(row[0], row[1], row[2], row[3]);
        const int area = row[cv::CC_STAT_AREA];
        if (area < 8 || area > 2 * radius * radius || region.width > 5 * radius / 2 ||
            region.height > 5 * radius / 2) {
            continue;
        }
        int supported = 0;
        int blue = 0;
        for (int y = region.y; y < region.br().y; ++y) {
            const auto* id = labels.ptr<int>(y);
            const auto* color = smoothed.ptr<cv::Vec3f>(y + top);
            for (int x = region.x; x < region.br().x; ++x) {
                if (id[x] != i) {
                    continue;
                }
                const cv::Vec3f p = color[x];
                const bool is_blue = p[0] - p[2] > 0.125f * scene_value && p[0] - p[1] > 0.035f * scene_value;
                blue += is_blue;
                const float value = std::max({p[0], p[1], p[2]});
                const bool near_green = p[1] >= p[0] - 0.025f * scene_value &&
                                        p[0] - p[2] < 0.08f * scene_value && value < 0.83f * scene_value;
                supported += !is_blue && (value < 0.62f * scene_value || near_green);
            }
        }
        // 眼边的窄凹口含有一小片暗像素。手臂间的蓝色缺口则保留。
        if (100 * supported >= 25 * area && 100 * blue < 55 * area) {
            cv::Mat selected;
            cv::compare(labels(region), i, selected, cv::CMP_EQ);
            upper(region).setTo(255, selected);
        }
    }
}

} // namespace

void refineSilhouetteEdges(const cv::Mat& bgr, cv::Mat& solid, int scene_value,
                           bool feet_visible, const cv::Mat& repaired_pixels, const Parameters& p) {
    if (scene_value < 27) {
        return;
    }
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(solid, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.size() != 1 || contours[0].size() < 24 || contours[0].size() >= 32767) {
        return;
    }
    cv::Mat smoothed;
    cv::GaussianBlur(bgr, smoothed, cv::Size(5, 5), 0.9);
    smoothed.convertTo(smoothed, CV_32FC3);
    const cv::Rect box = cv::boundingRect(contours[0]);
    cv::Mat uncertain;
    if (!repaired_pixels.empty()) {
        // 插值区域没有实拍边缘证据，沿用已经去字幕的粗轮廓。
        const int size = 2 * p.edge_refine_radius + 9;
        cv::dilate(repaired_pixels, uncertain,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size, size)));
    }
    const auto adjusted = alignedContour(smoothed, solid, contours[0], box, p.edge_refine_radius,
                                          scene_value, feet_visible, uncertain);
    cv::Mat refined = cv::Mat::zeros(solid.size(), CV_8UC1);
    cv::fillPoly(refined, std::vector<std::vector<cv::Point>>{adjusted}, cv::Scalar(255));
    const cv::Rect active = cv::boundingRect(refined);
    for (int y = active.y; y < active.br().y; ++y) {
        const auto* color = smoothed.ptr<cv::Vec3f>(y);
        const auto* was = solid.ptr<uchar>(y);
        auto* mask = refined.ptr<uchar>(y);
        for (int x = active.x; x < active.br().x; ++x) {
            const cv::Vec3f p = color[x];
            if (was[x] == 0 && p[0] > 0.48f * scene_value &&
                p[0] - p[2] > std::max(10.0f, 0.195f * scene_value) &&
                p[0] - p[1] > 0.082f * scene_value) {
                mask[x] = 0;
            }
        }
    }
    repairDarkNotches(smoothed, refined, box, p.eye_notch_radius, scene_value, uncertain);
    if (scene_value < 160) {
        // 淡入淡出时压缩色块会让剖面锯齿化，用小尺度平滑消除突跳。
        cv::GaussianBlur(refined, refined, cv::Size(9, 9), 2.1);
        cv::threshold(refined, refined, 127, 255, cv::THRESH_BINARY);
    } else {
        cv::medianBlur(refined, refined, 5);
    }
    solid = refined;
}
