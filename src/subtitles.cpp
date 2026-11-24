#include "segmentation.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>

namespace {

cv::Mat disk(int radius) {
    return cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * radius + 1, 2 * radius + 1));
}

void strokeEvidence(const cv::Mat& bgr, int scene_value, int radius,
                    cv::Mat& contrast, cv::Mat& eligible) {
    cv::Mat gray, local_dark, b, g, r, chroma, warm;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    // 字幕有深描边。亮笔画像素比邻域最暗点更亮。
    cv::erode(gray, local_dark, disk(radius));
    cv::subtract(gray, local_dark, contrast);
    cv::compare(gray, scene_value / 2, eligible, cv::CMP_GE);

    cv::extractChannel(bgr, b, 0);
    cv::extractChannel(bgr, g, 1);
    cv::extractChannel(bgr, r, 2);
    cv::max(g, r, warm);
    cv::subtract(warm, b, chroma);
    cv::compare(chroma, std::max(1, cvRound(0.024 * scene_value)), warm, cv::CMP_GE);
    cv::bitwise_and(eligible, warm, eligible);
}

cv::Mat brightStrokes(const cv::Mat& contrast, const cv::Mat& eligible,
                      int scene_value, int min_contrast_pct) {
    cv::Mat bright;
    cv::compare(contrast, std::max(4, scene_value * min_contrast_pct / 100), bright, cv::CMP_GE);
    cv::bitwise_and(bright, eligible, bright);
    return bright;
}

bool matchesFaintSubtitle(const cv::Mat& contrast, const cv::Mat& eligible,
                          const SubtitleMask& subtitle, int scene_value, int stroke_count) {
    const int surround_count = cv::countNonZero(subtitle.surround);
    if (scene_value < 85 || surround_count == 0) {
        return false;
    }
    // 半透明字幕的单个像素反差低，整组笔画的命中比例仍明显高于笔画周围。
    const int thresholds[] = {8, 4};
    const double min_rates[] = {0.13, 0.32};
    const double min_gaps[] = {0.07, 0.075};
    for (int i = 0; i < 2; ++i) {
        const cv::Mat weak = brightStrokes(contrast, eligible, scene_value, thresholds[i]);
        cv::Mat hits;
        cv::bitwise_and(weak, subtitle.strokes, hits);
        const double stroke_rate = cv::countNonZero(hits) / static_cast<double>(stroke_count);
        if (stroke_rate < min_rates[i]) {
            continue;
        }
        cv::bitwise_and(weak, subtitle.surround, hits);
        const double surround_rate = cv::countNonZero(hits) / static_cast<double>(surround_count);
        if (stroke_rate - surround_rate >= min_gaps[i]) {
            return true;
        }
    }
    return false;
}

} // namespace

SubtitleMask calibrateSubtitles(const std::string& input_path, int total, const Parameters& p) {
    if (p.subtitle_samples == 0 || total < 6) {
        return {};
    }
    cv::VideoCapture input(input_path, cv::CAP_FFMPEG);
    if (!input.isOpened()) {
        return {};
    }
    const int stride = std::max(1, total / p.subtitle_samples);
    cv::Mat votes, visible, frame, half;
    cv::Size full_size;
    int samples = 0;
    for (int n = 0; input.grab(); ++n) {
        if (n % stride != 0 || !input.retrieve(frame) || frame.empty()) {
            continue;
        }
        const BodySeed seed = findBodySeed(frame, p);
        if (seed.scene_value < 85 || seed.mask.empty()) {
            continue;
        }
        if (votes.empty()) {
            full_size = frame.size();
            votes = cv::Mat::zeros(seed.mask.size(), CV_32SC1);
            visible = cv::Mat::zeros(seed.mask.size(), CV_32SC1);
        }
        if (frame.size() != full_size) {
            return {};
        }
        cv::resize(frame, half, seed.mask.size(), 0, 0, cv::INTER_AREA);
        cv::Mat contrast, eligible;
        strokeEvidence(half, seed.scene_value, 4, contrast, eligible);
        cv::Mat strokes = brightStrokes(contrast, eligible, seed.scene_value, 18);
        // 主体可能遮住字幕。只在主体横向范围之外学习当地的固定笔画。
        const int left = std::max(0, seed.bounds.x - 5);
        const int right = std::min(half.cols, seed.bounds.br().x + 5);
        strokes.colRange(left, right).setTo(0);
        if (left > 0) {
            visible.colRange(0, left) += 1;
        }
        if (right < half.cols) {
            visible.colRange(right, half.cols) += 1;
        }
        cv::Mat increments;
        strokes.convertTo(increments, CV_32SC1, 1.0 / 255.0);
        cv::add(votes, increments, votes);
        ++samples;
    }
    if (samples < 6) {
        return {};
    }

    cv::Mat repeat = cv::Mat::zeros(votes.size(), CV_8UC1);
    const int min_votes = std::max(6, (12 * samples + 99) / 100);
    for (int y = 0; y < repeat.rows; ++y) {
        const int* hits = votes.ptr<int>(y);
        const int* observations = visible.ptr<int>(y);
        auto* out = repeat.ptr<uchar>(y);
        for (int x = 0; x < repeat.cols; ++x) {
            if (hits[x] >= min_votes && 5 * hits[x] >= observations[x]) {
                out[x] = 255;
            }
        }
    }
    // 排除孤立的重复噪点和面积过大的静物。
    cv::Mat labels, stats, centers;
    const int count = cv::connectedComponentsWithStats(repeat, labels, stats, centers, 8, CV_32S);
    for (int label = 1; label < count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < 8 || area > repeat.rows * repeat.cols / 30 ||
            stats.at<int>(label, cv::CC_STAT_HEIGHT) > repeat.rows / 9) {
            cv::Mat rejected;
            cv::compare(labels, label, rejected, cv::CMP_EQ);
            repeat.setTo(0, rejected);
        }
    }
    if (cv::countNonZero(repeat) == 0) {
        return {};
    }

    // 从反复出现的亮笔画向外扩几像素，把黑描边也标为需要重建的像素。
    cv::Mat cover;
    cv::morphologyEx(repeat, cover, cv::MORPH_CLOSE, disk(6));
    cv::dilate(cover, cover, disk(5));
    std::vector<std::vector<cv::Point>> holes;
    cv::findContours(cover, holes, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(cover, holes, -1, cv::Scalar(255), cv::FILLED);
    cv::Mat full_strokes, full_cover;
    cv::resize(repeat, full_strokes, full_size, 0, 0, cv::INTER_NEAREST);
    cv::resize(cover, full_cover, full_size, 0, 0, cv::INTER_NEAREST);
    const cv::Rect box = cv::boundingRect(full_cover);
    const cv::Rect bounds = cv::Rect(box.x - 12, box.y - 12, box.width + 24, box.height + 24) &
                            cv::Rect(cv::Point(), full_size);
    const cv::Mat strokes = full_strokes(bounds).clone();
    cv::Mat far, near, surround;
    cv::dilate(strokes, far, disk(10));
    cv::dilate(strokes, near, disk(3));
    cv::subtract(far, near, surround);
    return {bounds, strokes, full_cover(bounds).clone(), surround};
}

cv::Mat repairSubtitle(const cv::Mat& bgr, const cv::Rect& search, const SubtitleMask& subtitle,
                       int scene_value, cv::Mat& repaired_pixels) {
    const cv::Rect overlap = subtitle.bounds & search;
    if (overlap.empty()) {
        return bgr(search);
    }
    const cv::Mat region = bgr(subtitle.bounds);
    cv::Mat contrast, eligible;
    strokeEvidence(region, scene_value, 8, contrast, eligible);
    cv::Mat evidence = brightStrokes(contrast, eligible, scene_value, 18);
    cv::bitwise_and(evidence, subtitle.strokes, evidence);
    // 同一位置并非总有字幕。只有本帧也观察到一定比例的亮笔画才处理。
    const int stroke_count = cv::countNonZero(subtitle.strokes);
    if (7 * cv::countNonZero(evidence) < stroke_count &&
        !matchesFaintSubtitle(contrast, eligible, subtitle, scene_value, stroke_count)) {
        return bgr(search);
    }
    cv::Mat fixed = region.clone();
    for (int x = 0; x < fixed.cols; ++x) {
        int y = 0;
        while (y < fixed.rows) {
            if (subtitle.cover.ptr<uchar>(y)[x] == 0) {
                ++y;
                continue;
            }
            const int first = y;
            while (y < fixed.rows && subtitle.cover.ptr<uchar>(y)[x] != 0) {
                ++y;
            }
            if (first == 0 || y == fixed.rows) {
                continue;
            }
            const cv::Vec3b above = region.ptr<cv::Vec3b>(first - 1)[x];
            const cv::Vec3b below = region.ptr<cv::Vec3b>(y)[x];
            for (int j = first; j < y; ++j) {
                const double weight = static_cast<double>(j - first + 1) / (y - first + 1);
                cv::Vec3b& pixel = fixed.ptr<cv::Vec3b>(j)[x];
                for (int channel = 0; channel < 3; ++channel) {
                    pixel[channel] = cv::saturate_cast<uchar>((1.0 - weight) * above[channel] +
                                                              weight * below[channel]);
                }
            }
        }
    }
    // 列间轻微平滑，避免重建边缘出现细小锯齿；原始视频仍用于最后绘线。
    cv::Mat smooth;
    cv::GaussianBlur(fixed, smooth, cv::Size(9, 1), 2.0);
    smooth.copyTo(fixed, subtitle.cover);
    cv::Mat patch = bgr(search).clone();
    fixed(overlap - subtitle.bounds.tl()).copyTo(patch(overlap - search.tl()));
    repaired_pixels = cv::Mat::zeros(search.size(), CV_8UC1);
    subtitle.cover(overlap - subtitle.bounds.tl()).copyTo(repaired_pixels(overlap - search.tl()));
    return patch;
}
