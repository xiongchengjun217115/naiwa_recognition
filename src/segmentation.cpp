#include "segmentation.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace {

struct ParameterField {
    const char* name;
    int& value;
    int min;
    int max;
};

// 配置文件、保存和合法性检查共用这张字段表。
std::array<ParameterField, 22> parameterFields(Parameters& p) {
    return {{
        {"hue_min", p.hue_min, 0, 179},
        {"hue_max", p.hue_max, 0, 179},
        {"edge_hue_max", p.edge_hue_max, 0, 179},
        {"seed_saturation", p.seed_saturation, 0, 255},
        {"edge_saturation", p.edge_saturation, 0, 255},
        {"seed_chroma_pct", p.seed_chroma_pct, 0, 100},
        {"edge_chroma_pct", p.edge_chroma_pct, 0, 100},
        {"visible_v", p.visible_v, 0, 255},
        {"seed_open_radius", p.seed_open_radius, 0, 60},
        {"recover_radius", p.recover_radius, 0, 80},
        {"loose_open_radius", p.loose_open_radius, 0, 20},
        {"close_radius", p.close_radius, 0, 20},
        {"edge_refine_radius", p.edge_refine_radius, 0, 32},
        {"eye_notch_radius", p.eye_notch_radius, 0, 32},
        {"contour_smooth_px", p.contour_smooth_px, 0, 14},
        {"floor_chroma_pct", p.floor_chroma_pct, 0, 50},
        {"floor_contrast_pct", p.floor_contrast_pct, 0, 30},
        {"subtitle_samples", p.subtitle_samples, 0, 1000},
        {"fade_below_v", p.fade_below_v, 0, 180},
        {"fade_match_pct", p.fade_match_pct, 50, 100},
        {"min_area_per_10000", p.min_area_per_10000, 1, 10000},
        {"line_width", p.line_width, 1, 12}
    }};
}

cv::Mat ellipseKernel(int radius) {
    const int size = 2 * radius + 1;
    return cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
}

int sceneValue(const cv::Mat& hsv) {
    // 对小图均匀取样。第 85 百分位对少量亮水印、个别高光不敏感。
    std::array<int, 256> histogram{};
    int samples = 0;
    for (int y = 0; y < hsv.rows; y += 4) {
        const cv::Vec3b* row = hsv.ptr<cv::Vec3b>(y);
        for (int x = 0; x < hsv.cols; x += 4) {
            ++histogram[row[x][2]];
            ++samples;
        }
    }
    const int rank = (85 * samples + 99) / 100;
    int cumulative = 0;
    for (int v = 0; v < 256; ++v) {
        cumulative += histogram[v];
        if (cumulative >= rank) {
            return v;
        }
    }
    return 255;
}

cv::Mat chromaMap(const cv::Mat& bgr, bool include_green) {
    cv::Mat warm, blue, difference;
    cv::extractChannel(bgr, warm, 2);
    if (include_green) {
        cv::Mat green;
        cv::extractChannel(bgr, green, 1);
        cv::max(warm, green, warm);
    }
    cv::extractChannel(bgr, blue, 0);
    // 8 位无符号减法把负值截为 0，排除 B 通道占优的蓝色像素。
    cv::subtract(warm, blue, difference);
    return difference;
}

cv::Mat warmMask(const cv::Mat& hsv, const cv::Mat& chroma, int hue_min, int hue_max,
                 int saturation, int min_chroma) {
    cv::Mat mask, warm;
    cv::inRange(hsv, cv::Scalar(hue_min, saturation, 1), cv::Scalar(hue_max, 255, 255), mask);
    cv::compare(chroma, min_chroma, warm, cv::CMP_GE);
    cv::bitwise_and(mask, warm, mask);
    return mask;
}

cv::Rect expandRect(const cv::Rect& rect, int padding, const cv::Size& size) {
    return cv::Rect(rect.x - padding, rect.y - padding,
                    rect.width + 2 * padding, rect.height + 2 * padding) &
           cv::Rect(cv::Point(), size);
}

int trimReflection(cv::Mat& candidate, const cv::Mat& bgr, const cv::Mat& chroma,
                   int seed_bottom, int seed_height, int scene_v, int contrast_pct) {
    if (contrast_pct == 0 || scene_v < 38) {
        return 0;
    }
    const int above = 8;
    const int below = 7;
    const int first = std::max(above, seed_bottom - seed_height * 13 / 100);
    const int last = candidate.rows - below;
    if (first >= last) {
        return 0;
    }
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.0);
    cv::Mat peak_below;
    cv::dilate(chroma, peak_below, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 24)),
               cv::Point(0, 0));
    const int step = std::max(2, cvRound(scene_v * contrast_pct / 100.0));
    const int warm_above = std::max(2, scene_v * 9 / 100);
    std::vector<int> seam(candidate.cols, -1);
    std::vector<int> score(candidate.cols, 0);

    // 脚掌和倒影之间常有窄暗带。只在主体下部寻找先变暗、再变亮的位置。
    for (int y = first; y < last; ++y) {
        const auto* upper = gray.ptr<uchar>(y - above);
        const auto* middle = gray.ptr<uchar>(y);
        const auto* lower = gray.ptr<uchar>(y + below);
        const auto* upper_chroma = chroma.ptr<uchar>(y - above);
        const auto* lower_peak = peak_below.ptr<uchar>(y + 3);
        const auto* above_mask = candidate.ptr<uchar>(y - above);
        for (int x = 0; x < candidate.cols; ++x) {
            const int down = upper[x] - middle[x];
            const int up = lower[x] - middle[x];
            if (above_mask[x] == 0 || down < step + step / 2 || up < step ||
                middle[x] > scene_v / 2 || upper_chroma[x] < warm_above ||
                lower_peak[x] > upper_chroma[x] + step) {
                continue;
            }
            const int strength = 5 * std::min(down, up) + down + up;
            if (strength >= score[x]) {
                score[x] = strength;
                seam[x] = y;
            }
        }
    }

    // 多列给出接近的高度才采信。用局部中值跨过脚趾处少量缺失的列。
    constexpr int neighborhood = 10;
    std::vector<int> confirmed(candidate.cols, -1);
    for (int x = 0; x < candidate.cols; ++x) {
        std::array<int, 2 * neighborhood + 1> nearby{};
        int count = 0;
        for (int j = std::max(0, x - neighborhood); j <= std::min(candidate.cols - 1, x + neighborhood); ++j) {
            if (seam[j] >= 0) {
                nearby[count++] = seam[j];
            }
        }
        if (count < 6) {
            continue;
        }
        std::sort(nearby.begin(), nearby.begin() + count);
        const int median = nearby[count / 2];
        const int agreeing = static_cast<int>(std::count_if(nearby.begin(), nearby.begin() + count,
                                                            [median](int y) { return std::abs(y - median) <= 7; }));
        if (agreeing >= 6) {
            confirmed[x] = seam[x] >= 0 && std::abs(seam[x] - median) <= 7 ? seam[x] : median;
        }
    }

    // 暗带在脚掌边缘可能有缺口。只把可靠的切线向旁边延伸一小段。
    int clipped_columns = 0;
    for (int x = 0; x < candidate.cols; ++x) {
        int limit = confirmed[x];
        for (int d = 1; limit < 0 && d <= 18; ++d) {
            if (x - d >= 0 && confirmed[x - d] >= 0) {
                limit = confirmed[x - d];
            } else if (x + d < candidate.cols && confirmed[x + d] >= 0) {
                limit = confirmed[x + d];
            }
        }
        if (limit < 0) {
            continue;
        }
        if (confirmed[x] < 0 && peak_below.ptr<uchar>(limit + 3)[x] >
                                std::max(warm_above, static_cast<int>(chroma.ptr<uchar>(limit - above)[x])) + step) {
            continue;
        }
        bool clipped = false;
        for (int y = limit + 2; y < candidate.rows; ++y) {
            uchar& value = candidate.ptr<uchar>(y)[x];
            clipped |= value != 0;
            value = 0;
        }
        clipped_columns += clipped;
    }
    if (clipped_columns == 0) {
        return 0;
    }

    // 足底暗带在许多列一致时，补去从漏切列延伸的细长倒影。
    // 保留仍成宽片的区域，避免把另一只落在不同高度的脚截掉。
    std::vector<int> heights;
    for (int y : confirmed) {
        if (y >= 0) {
            heights.push_back(y);
        }
    }
    if (heights.size() >= 45) {
        std::sort(heights.begin(), heights.end());
        const int median = heights[heights.size() / 2];
        const auto start = std::lower_bound(heights.begin(), heights.end(), median - 8);
        const auto end = std::upper_bound(start, heights.end(), median + 8);
        if (4 * (end - start) >= 3 * static_cast<long>(heights.size())) {
            const int cap = std::min(candidate.rows, *(end - 1) + 6);
            const int thin_width = std::max(10, seed_height / 16);
            const auto rows = candidate.rowRange(cap, candidate.rows);
            bool thin = true;
            for (int row = 0; row < rows.rows; ++row) {
                if (cv::countNonZero(rows.row(row)) > thin_width) {
                    thin = false;
                    break;
                }
            }
            if (thin && !rows.empty()) {
                candidate.rowRange(cap, candidate.rows).setTo(0);
            }
        }
    }
    return clipped_columns;
}

std::vector<cv::Point> largestOuterContour(const cv::Mat& mask, const cv::Point& offset) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, offset);
    int best = -1;
    double best_area = 0.0;
    for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area > best_area) {
            best_area = area;
            best = i;
        }
    }
    return best >= 0 ? std::move(contours[best]) : std::vector<cv::Point>{};
}

} // namespace

void loadParameters(const std::string& path, Parameters& p) {
    cv::FileStorage file(path, cv::FileStorage::READ);
    if (!file.isOpened()) {
        throw std::runtime_error("Cannot open configuration: " + path);
    }
    for (const ParameterField& field : parameterFields(p)) {
        const cv::FileNode node = file[field.name];
        if (!node.empty()) {
            if (!node.isInt()) {
                throw std::runtime_error(std::string("Expected an integer for ") + field.name);
            }
            node >> field.value;
        }
    }
    validateParameters(p);
}

void saveParameters(const std::string& path, Parameters p) {
    cv::FileStorage file(path, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    if (!file.isOpened()) {
        throw std::runtime_error("Cannot write configuration: " + path);
    }
    for (const ParameterField& field : parameterFields(p)) {
        file << field.name << field.value;
    }
}

void validateParameters(Parameters p) {
    for (const ParameterField& field : parameterFields(p)) {
        const int value = field.value;
        if (value < field.min || value > field.max) {
            throw std::runtime_error(std::string("Out of range: ") + field.name);
        }
    }
    if (p.hue_min > p.hue_max || p.hue_max > p.edge_hue_max || p.edge_saturation > p.seed_saturation ||
        p.edge_chroma_pct > p.seed_chroma_pct) {
        throw std::runtime_error("Expected hue_min <= hue_max <= edge_hue_max and edge S/chroma <= seed S/chroma");
    }
}

BodySeed findBodySeed(const cv::Mat& bgr, const Parameters& p) {
    BodySeed body;
    // 半尺寸图只用来选主体，最终边缘由原像素决定。
    cv::Mat small, small_hsv;
    cv::resize(bgr, small, cv::Size((bgr.cols + 1) / 2, (bgr.rows + 1) / 2),
               0, 0, cv::INTER_AREA);
    cv::cvtColor(small, small_hsv, cv::COLOR_BGR2HSV);
    body.scene_value = sceneValue(small_hsv);
    if (body.scene_value < p.visible_v) {
        return body;
    }

    const int min_area = std::max(1, cvRound(small.total() * p.min_area_per_10000 / 10000.0));
    const int seed_chroma = std::max(2, cvRound(body.scene_value * p.seed_chroma_pct / 100.0));
    cv::Mat seed = warmMask(small_hsv, chromaMap(small, false), p.hue_min, p.hue_max,
                            p.seed_saturation, seed_chroma);
    // 细字笔画易被开运算消除，较宽的人物区域能留下一个稳定的主体。
    if (p.seed_open_radius > 0) {
        cv::morphologyEx(seed, seed, cv::MORPH_OPEN, ellipseKernel((p.seed_open_radius + 1) / 2));
    }

    cv::Mat labels, stats, centroids;
    const int count = cv::connectedComponentsWithStats(seed, labels, stats, centroids, 8, CV_32S);
    int best_label = 0;
    int best_area = min_area - 1;
    for (int label = 1; label < count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area > best_area) {
            best_label = label;
            best_area = area;
        }
    }
    if (best_label == 0) {
        return body;
    }
    body.bounds = cv::Rect(stats.at<int>(best_label, cv::CC_STAT_LEFT),
                           stats.at<int>(best_label, cv::CC_STAT_TOP),
                           stats.at<int>(best_label, cv::CC_STAT_WIDTH),
                           stats.at<int>(best_label, cv::CC_STAT_HEIGHT));
    cv::compare(labels, best_label, body.mask, cv::CMP_EQ);
    return body;
}

Detection detectSilhouette(const cv::Mat& bgr, const Parameters& p, const SubtitleMask& subtitle) {
    BodySeed seed = findBodySeed(bgr, p);
    Detection result;
    result.mask = cv::Mat::zeros(bgr.size(), CV_8UC1);
    result.scene_value = seed.scene_value;
    if (seed.mask.empty()) {
        return result;
    }
    const cv::Rect seed_rect = seed.bounds;
    const int radius = (p.recover_radius + 1) / 2;
    if (radius > 0) {
        cv::dilate(seed.mask, seed.mask, ellipseKernel(radius));
    }
    cv::Mat support;
    cv::resize(seed.mask, support, bgr.size(), 0, 0, cv::INTER_NEAREST);

    // 在随主体移动的工作区域内，使用较宽松的颜色条件找回暗手脚和亚像素混色边缘。
    const cv::Rect original_seed(2 * seed_rect.x, 2 * seed_rect.y,
                                 2 * seed_rect.width, 2 * seed_rect.height);
    const int padding = std::max({2 * radius + p.close_radius + 3,
                                  original_seed.height / 8, p.edge_refine_radius + 15});
    const cv::Rect search = expandRect(original_seed, padding, bgr.size());
    cv::Mat repaired_pixels;
    const cv::Mat patch = repairSubtitle(bgr, search, subtitle, seed.scene_value, repaired_pixels);
    cv::Mat hsv;
    cv::cvtColor(patch, hsv, cv::COLOR_BGR2HSV);
    const cv::Mat chroma = chromaMap(patch, true);
    const int edge_chroma = std::max(1, cvRound(result.scene_value * p.edge_chroma_pct / 100.0));
    cv::Mat loose = warmMask(hsv, chroma, p.hue_min, p.edge_hue_max, p.edge_saturation, edge_chroma);
    const bool feet_visible = seed_rect.br().y < seed.mask.rows;
    const int seed_bottom = original_seed.br().y - search.y;
    if (feet_visible && p.floor_chroma_pct > p.edge_chroma_pct) {
        // 地板倒影通常偏灰。只在相对主体的脚部带提高色差下限。
        const int start = std::max(0, seed_bottom - original_seed.height * 13 / 100);
        const cv::Range rows(start, loose.rows);
        cv::Mat warm_floor;
        cv::compare(chroma.rowRange(rows), cvRound(seed.scene_value * p.floor_chroma_pct / 100.0),
                    warm_floor, cv::CMP_GE);
        cv::Mat lower = loose.rowRange(rows);
        cv::bitwise_and(lower, warm_floor, lower);
    }
    cv::Mat candidate;
    cv::bitwise_and(loose, support(search), candidate);
    // 粗种子有时在细脚踝处断开。允许较厚的宽松色块接回手脚，同时淘汰细字笔画。
    cv::Mat thick;
    cv::morphologyEx(loose, thick, cv::MORPH_OPEN, ellipseKernel(p.loose_open_radius));
    cv::bitwise_or(candidate, thick, candidate);
    if (p.close_radius > 0) {
        cv::morphologyEx(candidate, candidate, cv::MORPH_CLOSE, ellipseKernel(p.close_radius));
    }
    cv::medianBlur(candidate, candidate, 5);
    auto outer = largestOuterContour(candidate, search.tl());
    // 强倒影会把整个连通轮廓向画面底部拖长。仅出现这个征兆时才沿暗带切分，
    // 以免在侧身或屈膝时把足面上的自身阴影当作足底。
    const int near_bottom = bgr.rows - std::max(20, original_seed.height / 14);
    if (feet_visible && !outer.empty() && cv::boundingRect(outer).br().y >= near_bottom) {
        result.floor_columns = trimReflection(candidate, patch, chroma, seed_bottom,
                                              original_seed.height, result.scene_value, p.floor_contrast_pct);
        if (result.floor_columns > 0) {
            cv::medianBlur(candidate, candidate, 3);
            outer = largestOuterContour(candidate, search.tl());
        }
    }
    if (outer.empty()) {
        return result;
    }
    if (p.edge_refine_radius > 0) {
        cv::Mat solid = cv::Mat::zeros(patch.size(), CV_8UC1);
        const std::vector<std::vector<cv::Point>> contours{outer};
        cv::drawContours(solid, contours, 0, cv::Scalar(255), cv::FILLED,
                         cv::LINE_8, cv::noArray(), 0, -search.tl());
        refineSilhouetteEdges(patch, solid, result.scene_value, feet_visible, repaired_pixels, p);
        auto refined = largestOuterContour(solid, search.tl());
        if (!refined.empty()) {
            outer = std::move(refined);
        }
    }
    result.outer = std::move(outer);
    // 用外轮廓填实眼、嘴等封闭区域，包括画面裁切时的闭合边。
    cv::drawContours(result.mask, std::vector<std::vector<cv::Point>>{result.outer}, 0,
                     cv::Scalar(255), cv::FILLED);
    // 只清除轮廓里面成片的蓝色缝隙。深色指尖的混色边不能仅因 H 偏蓝便挖掉。
    cv::Mat blue, channel_b, channel_r, blue_strength, interior;
    cv::inRange(hsv, cv::Scalar(93, 32, std::max(4, seed.scene_value * 24 / 100)),
                cv::Scalar(138, 255, 255), blue);
    cv::extractChannel(patch, channel_b, 0);
    cv::extractChannel(patch, channel_r, 2);
    cv::subtract(channel_b, channel_r, blue_strength);
    cv::compare(blue_strength, std::max(5, seed.scene_value * 11 / 100), blue_strength, cv::CMP_GE);
    cv::bitwise_and(blue, blue_strength, blue);
    cv::Mat mask_patch = result.mask(search);
    cv::erode(mask_patch, interior, ellipseKernel(2));
    cv::bitwise_and(blue, interior, blue);
    cv::morphologyEx(blue, blue, cv::MORPH_OPEN, ellipseKernel(2));
    mask_patch.setTo(0, blue);
    result.bounds = cv::boundingRect(result.outer);
    result.area = cv::countNonZero(result.mask(result.bounds));
    return result;
}

cv::Mat drawSilhouette(const cv::Mat& bgr, const Detection& detection, int line_width) {
    cv::Mat output = bgr.clone();
    if (detection.outer.empty()) {
        return output;
    }
    const int shift = detection.subpixel_outer.empty() ? 0 : 4;
    const auto& points = shift == 0 ? detection.outer : detection.subpixel_outer;
    const int right = (bgr.cols - 1) << shift;
    const int bottom = (bgr.rows - 1) << shift;
    const cv::Scalar white(255, 255, 255);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const cv::Point a = points[i];
        const cv::Point b = points[(i + 1) % points.size()];
        // 屏幕外的边缘不可见，跳过沿画框延伸的人工闭合线。
        if ((a.x == b.x && (a.x == 0 || a.x == right)) ||
            (a.y == b.y && (a.y == 0 || a.y == bottom))) {
            continue;
        }
        cv::line(output, a, b, white, line_width, cv::LINE_AA, shift);
    }
    return output;
}
