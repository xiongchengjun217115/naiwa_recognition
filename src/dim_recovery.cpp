#include "segmentation.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <utility>

namespace {

constexpr int scale = 4;
constexpr int shift_limit = 4; // 在小图上最多平移 4 个像素。

cv::Mat smallGray(const cv::Mat& frame) {
    cv::Mat small, gray;
    cv::resize(frame, small, cv::Size((frame.cols + scale - 1) / scale,
                                     (frame.rows + scale - 1) / scale),
               0, 0, cv::INTER_AREA);
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

struct Match {
    cv::Point shift;
    double score = 0.0;
};

Match matchReference(const cv::Mat& gray, const Detection& current,
                     const DimReference& reference, double min_score) {
    if (reference.gray_patch.empty()) {
        return {};
    }
    const cv::Rect box = reference.small_bounds;
    const cv::Rect search = cv::Rect(box.x - shift_limit, box.y - shift_limit,
                                     box.width + 2 * shift_limit, box.height + 2 * shift_limit) &
                            cv::Rect(cv::Point(), gray.size());
    cv::Mat correlations;
    cv::matchTemplate(gray(search), reference.gray_patch, correlations, cv::TM_CCOEFF_NORMED);
    double score = 0.0;
    cv::Point location;
    cv::minMaxLoc(correlations, nullptr, &score, nullptr, &location);
    if (score < min_score) {
        return {};
    }
    const cv::Point shift = (search.tl() + location - box.tl()) * scale;
    const cv::Rect overlap = (reference.bounds + shift) & current.bounds;
    if (overlap.empty()) {
        return {};
    }
    cv::Mat shared;
    cv::bitwise_and(reference.mask(overlap - shift), current.mask(overlap), shared);
    const int intersection = cv::countNonZero(shared);
    // 保留当前帧可见证据。大部分当前前景应与邻帧重合，防止跨镜头借轮廓。
    if (100LL * intersection < 87LL * current.area || 100LL * intersection < 40LL * reference.area) {
        return {};
    }
    return {shift, score};
}

} // namespace

DimReference makeDimReference(const cv::Mat& bgr, const Detection& reliable) {
    if (reliable.area == 0) {
        return {};
    }
    const cv::Mat gray = smallGray(bgr);
    const cv::Rect box = reliable.bounds;
    // 模板用头、肩和上身附近的灰度结构，减少字幕与地板倒影的影响。
    const int context = box.width / 15;
    const int left = std::max(0, (box.x - context) / scale);
    const int top = std::max(0, (box.y - 10) / scale);
    const int right = std::min(gray.cols, (box.br().x + context) / scale);
    const int bottom = std::min(gray.rows, (box.y + box.height * 74 / 100) / scale);
    const cv::Rect part(left, top, right - left, bottom - top);
    if (part.width < 4 || part.height < 4) {
        return {};
    }
    return {gray(part).clone(), part, reliable.mask, reliable.bounds, reliable.area};
}

void recoverDimSilhouette(const cv::Mat& bgr, Detection& current,
                          const DimReference& before, const DimReference& after,
                          const Parameters& p) {
    if (current.area == 0 || current.scene_value >= p.fade_below_v) {
        return;
    }
    const cv::Mat gray = smallGray(bgr);
    const Match left = matchReference(gray, current, before, p.fade_match_pct / 100.0);
    const Match right = matchReference(gray, current, after, p.fade_match_pct / 100.0);
    const bool use_before = left.score >= right.score;
    const Match match = use_before ? left : right;
    if (match.score <= 0.0) {
        return;
    }
    const DimReference& prior = use_before ? before : after;
    cv::Mat shifted;
    const cv::Matx23d move(1, 0, match.shift.x, 0, 1, match.shift.y);
    cv::warpAffine(prior.mask, shifted, move, current.mask.size(), cv::INTER_NEAREST,
                   cv::BORDER_CONSTANT, cv::Scalar(0));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(shifted, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return;
    }
    // 蓝色边缘清理可能留下很小的孤立像素。用主轮廓约束平移后的二值掩膜。
    const auto main = std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
        return cv::contourArea(a) < cv::contourArea(b);
    });
    if (contours.size() > 1) {
        cv::Mat body = cv::Mat::zeros(shifted.size(), CV_8UC1);
        cv::drawContours(body, contours, static_cast<int>(main - contours.begin()), cv::Scalar(255), cv::FILLED);
        cv::bitwise_and(shifted, body, shifted);
    }
    current.mask = shifted;
    current.outer = std::move(*main);
    current.bounds = cv::boundingRect(current.outer);
    current.area = cv::countNonZero(current.mask(current.bounds));
    current.fade_match = match.score;
}
