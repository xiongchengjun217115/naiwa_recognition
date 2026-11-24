#include "segmentation.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace {

constexpr int fixed_shift = 4;
constexpr int fixed_scale = 1 << fixed_shift;

std::vector<cv::Point2f> equalArcPoints(const std::vector<cv::Point>& boundary, float& step) {
    std::vector<float> distances(boundary.size() + 1, 0.0f);
    for (std::size_t i = 0; i < boundary.size(); ++i) {
        const auto d = boundary[(i + 1) % boundary.size()] - boundary[i];
        distances[i + 1] = distances[i] + std::hypot(static_cast<float>(d.x), static_cast<float>(d.y));
    }
    const float length = distances.back();
    const int count = cvRound(length / 1.45f);
    if (count < 12) {
        return {};
    }
    step = length / count;
    std::vector<cv::Point2f> points;
    points.reserve(count);
    std::size_t segment = 0;
    for (int i = 0; i < count; ++i) {
        const float pos = i * step;
        while (segment + 1 < boundary.size() && distances[segment + 1] <= pos) {
            ++segment;
        }
        const auto from = cv::Point2f(boundary[segment]);
        const auto to = cv::Point2f(boundary[(segment + 1) % boundary.size()]);
        const float fraction = (pos - distances[segment]) /
                               std::max(0.001f, distances[segment + 1] - distances[segment]);
        points.push_back(from + (to - from) * fraction);
    }
    return points;
}

std::vector<float> lowShrinkKernel(float sigma, int radius) {
    std::vector<float> near(2 * radius + 1), far(2 * radius + 1);
    float near_sum = 0.0f;
    float far_sum = 0.0f;
    for (int j = -radius; j <= radius; ++j) {
        const float t = j / sigma;
        near[j + radius] = std::exp(-0.5f * t * t);
        far[j + radius] = std::exp(-t * t / 6.0f);
        near_sum += near[j + radius];
        far_sum += far[j + radius];
    }
    // 小尺度去锯齿，大尺度修正平滑向内收缩的趋势。手指和脚尖较少被磨小。
    for (int j = 0; j < static_cast<int>(near.size()); ++j) {
        near[j] = 1.5f * near[j] / near_sum - 0.5f * far[j] / far_sum;
    }
    return near;
}

std::vector<cv::Point> smoothCurve(const std::vector<cv::Point>& outline, const cv::Mat& bgr,
                                  float sigma, int scene_value) {
    float step = 1.0f;
    const auto points = equalArcPoints(outline, step);
    if (points.empty()) {
        return {};
    }
    const int count = static_cast<int>(points.size());
    const int radius = std::min(count / 2 - 1, cvCeil(6.0f * sigma / step));
    const auto weights = lowShrinkKernel(sigma / step, radius);
    const auto tip_weights = lowShrinkKernel(0.62f * sigma / step, radius);
    const auto size = bgr.size();
    std::vector<cv::Point> fixed;
    fixed.reserve(count);
    for (int i = 0; i < count; ++i) {
        const auto pos = points[i];
        auto edited = pos;
        // 裁到画面边界时锁住可见断面，以免轮廓被拉出黑色细缝。
        if (pos.x > 0.01f && pos.y > 0.01f && pos.x < size.width - 1.01f && pos.y < size.height - 1.01f) {
            const auto a = pos - points[(i + count - 5) % count];
            const auto b = points[(i + 5) % count] - pos;
            const cv::Vec3b color = bgr.at<cv::Vec3b>(cvRound(pos.y), cvRound(pos.x));
            const bool dark_tip = scene_value >= 160 && std::max({color[0], color[1], color[2]}) < 0.55f * scene_value &&
                                  a.dot(b) < 0.78f * std::hypot(a.x, a.y) * std::hypot(b.x, b.y);
            const auto& local_kernel = dark_tip ? tip_weights : weights;
            cv::Point2f averaged;
            for (int j = -radius; j <= radius; ++j) {
                averaged += points[(i + j + count) % count] * local_kernel[j + radius];
            }
            const auto delta = averaged - pos;
            const float distance = std::hypot(delta.x, delta.y);
            edited += delta * std::min(1.0f, 1.15f * sigma / std::max(0.001f, distance));
        }
        fixed.emplace_back(cvRound(fixed_scale * std::clamp(edited.x, 0.0f, size.width - 1.0f)),
                           cvRound(fixed_scale * std::clamp(edited.y, 0.0f, size.height - 1.0f)));
    }
    return fixed;
}

void findNarrowGaps(const cv::Mat& bgr, const cv::Mat& solid, int scene_value, cv::Mat& anchor) {
    cv::Mat stripe = cv::Mat::zeros(solid.size(), CV_8UC1);
    const int blue_over_green = std::max(4, scene_value / 48);
    const int side_drop = std::max(8, scene_value / 9);
    const int probe = 6;
    for (int y = 0; y < bgr.rows; ++y) {
        const auto* color = bgr.ptr<cv::Vec3b>(y);
        const auto* inside = solid.ptr<uchar>(y);
        auto* stripe_row = stripe.ptr<uchar>(y);
        for (int x = probe; x < bgr.cols - probe; ++x) {
            const auto c = color[x];
            if (!inside[x] || c[0] < scene_value / 2 || c[0] - c[1] < blue_over_green ||
                c[2] - c[0] > scene_value / 25) {
                continue;
            }
            // 手臂贴住身体时，蓝墙只剩细亮条。两侧均要有较暗的暖色边缘。
            const auto left = color[x - probe];
            const auto right = color[x + probe];
            if (c[0] - std::max(left[0], right[0]) >= side_drop &&
                left[2] > left[0] && right[2] > right[0]) {
                stripe_row[x] = 255;
            }
        }
    }
    const auto vertical = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 9));
    cv::morphologyEx(stripe, stripe, cv::MORPH_CLOSE, vertical);
    cv::morphologyEx(stripe, stripe, cv::MORPH_OPEN, vertical);
    cv::bitwise_or(anchor, stripe, anchor);
}

cv::Mat expandBackgroundGaps(const cv::Mat& bgr, const cv::Mat& anchor, int scene_value) {
    if (cv::countNonZero(anchor) == 0) {
        return {};
    }
    cv::Mat reach;
    cv::dilate(anchor, reach, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25)));
    cv::Mat candidate = anchor.clone();
    for (int y = 0; y < candidate.rows; ++y) {
        const auto* rgb = bgr.ptr<cv::Vec3b>(y);
        const auto* close = reach.ptr<uchar>(y);
        auto* out = candidate.ptr<uchar>(y);
        for (int x = 0; x < candidate.cols; ++x) {
            const cv::Vec3b c = rgb[x];
            if (close[x] && c[0] - c[2] >= std::max(2, scene_value / 60) &&
                c[0] - c[1] >= std::max(2, scene_value / 70)) {
                out[x] = 255;
            }
        }
    }
    // 蓝地面中的一道灰暗压缩块可能把同一条腿缝割成两段，沿竖向接回背景。
    cv::morphologyEx(candidate, candidate, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 21)));
    cv::bitwise_and(candidate, reach, candidate);
    cv::bitwise_or(candidate, anchor, candidate);
    cv::Mat labels;
    const int count = cv::connectedComponents(candidate, labels, 8, CV_32S);
    std::vector<int> hits(count, 0);
    for (int y = 0; y < labels.rows; ++y) {
        const int* id = labels.ptr<int>(y);
        const auto* seed = anchor.ptr<uchar>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (seed[x]) {
                ++hits[id[x]];
            }
        }
    }
    // 仅处理贴着原蓝色空隙的连通区域，独立的暗色零件不受影响。
    for (int y = 0; y < labels.rows; ++y) {
        const int* id = labels.ptr<int>(y);
        auto* out = candidate.ptr<uchar>(y);
        for (int x = 0; x < labels.cols; ++x) {
            out[x] = id[x] != 0 && hits[id[x]] >= 10 ? 255 : 0;
        }
    }
    cv::medianBlur(candidate, candidate, 3);
    cv::bitwise_or(candidate, anchor, candidate);
    return candidate;
}

void removeBlueFringe(const cv::Mat& source, const cv::Mat& mask, cv::Mat& output, int scene_value) {
    cv::Mat interior;
    cv::erode(mask, interior, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    const std::array<cv::Point, 8> directions{
        cv::Point(1, 0), cv::Point(-1, 0), cv::Point(0, 1), cv::Point(0, -1),
        cv::Point(1, 1), cv::Point(-1, 1), cv::Point(1, -1), cv::Point(-1, -1)};
    const cv::Rect limits(cv::Point(), mask.size());
    for (int y = 0; y < mask.rows; ++y) {
        const auto* keep = mask.ptr<uchar>(y);
        const auto* core = interior.ptr<uchar>(y);
        const auto* original = source.ptr<cv::Vec3b>(y);
        auto* dst = output.ptr<cv::Vec3b>(y);
        for (int x = 0; x < mask.cols; ++x) {
            if (!keep[x] || core[x] || original[x][0] - original[x][1] < 3) {
                continue;
            }
            const cv::Point here(x, y);
            const cv::Vec3f observed(original[x]);
            float best_error = std::max(26.0f, 0.0060f * scene_value * scene_value);
            for (const auto& d : directions) {
                const auto outside = here + 3 * d;
                const auto inside = here - 3 * d;
                if (!limits.contains(outside) || !limits.contains(inside) ||
                    mask.at<uchar>(outside) || !interior.at<uchar>(inside)) {
                    continue;
                }
                const cv::Vec3f bg(source.at<cv::Vec3b>(outside));
                const cv::Vec3f fg(source.at<cv::Vec3b>(inside));
                if (bg[0] - bg[2] < 0.12f * scene_value || bg[0] - bg[1] < 0.04f * scene_value ||
                    observed[0] - observed[1] < fg[0] - fg[1] + 4.0f) {
                    continue;
                }
                const auto difference = fg - bg;
                const float spread = difference.dot(difference);
                if (spread < 100.0f) {
                    continue;
                }
                // C ≈ a F + (1-a) B。已经知道两侧颜色时，估算颜色混合，再把 B 换成黑色。
                const float fraction = (observed - bg).dot(difference) / spread;
                if (fraction < 0.22f || fraction > 0.95f) {
                    continue;
                }
                const auto error = observed - (bg + fraction * difference);
                const float fit = error.dot(error);
                if (fit < best_error) {
                    best_error = fit;
                    dst[x] = cv::Vec3b(fg * fraction);
                }
            }
        }
    }
}

} // namespace

void finishSilhouette(const cv::Mat& bgr, Detection& detection, const Parameters& p) {
    if (detection.outer.size() < 3) {
        return;
    }
    const float sigma = p.contour_smooth_px * (detection.scene_value < 160 ? 1.42f : 1.0f);
    if (p.contour_smooth_px > 0) {
        detection.subpixel_outer = smoothCurve(detection.outer, bgr, sigma, detection.scene_value);
    }
    const int shift = detection.subpixel_outer.empty() ? 0 : fixed_shift;
    const auto& outline = shift ? detection.subpixel_outer : detection.outer;
    cv::Mat mask = cv::Mat::zeros(bgr.size(), CV_8UC1);
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{outline}, cv::Scalar(255), cv::LINE_8, shift);

    const cv::Rect region = detection.bounds;
    cv::Mat solid = cv::Mat::zeros(region.size(), CV_8UC1);
    cv::drawContours(solid, std::vector<std::vector<cv::Point>>{detection.outer}, 0,
                     cv::Scalar(255), cv::FILLED, cv::LINE_8, cv::noArray(), 0, -region.tl());
    cv::Mat anchor;
    cv::subtract(solid, detection.mask(region), anchor);
    findNarrowGaps(bgr(region), solid, detection.scene_value, anchor);
    const auto gaps = expandBackgroundGaps(bgr(region), anchor, detection.scene_value);
    if (!gaps.empty()) {
        mask(region).setTo(0, gaps);
    }
    detection.mask = mask;
    detection.bounds = cv::boundingRect(mask);
    detection.area = cv::countNonZero(mask(detection.bounds));
}

cv::Mat makeCutout(const cv::Mat& bgr, const Detection& detection, const SubtitleMask& subtitle) {
    cv::Mat result = cv::Mat::zeros(bgr.size(), CV_8UC3);
    if (detection.bounds.empty()) {
        return result;
    }
    // 加 4 像素背景供边缘混色判断，实际写入仍严格由掩膜限制。
    const auto box = detection.bounds;
    const auto region = cv::Rect(box.x - 4, box.y - 4, box.width + 8, box.height + 8) &
                        cv::Rect(cv::Point(), bgr.size());
    cv::Mat repaired;
    const cv::Mat source = repairSubtitle(bgr, region, subtitle, detection.scene_value, repaired);
    const cv::Mat mask = detection.mask(region);
    cv::Mat output = result(region);
    source.copyTo(output, mask);
    removeBlueFringe(source, mask, output, detection.scene_value);
    return result;
}
