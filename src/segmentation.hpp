#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct Parameters {
    int hue_min = 0;
    int hue_max = 52;
    int edge_hue_max = 91;
    int seed_saturation = 46;
    int edge_saturation = 12;
    int seed_chroma_pct = 15;
    int edge_chroma_pct = 3;
    int visible_v = 5;
    int seed_open_radius = 12;
    int recover_radius = 18;
    int loose_open_radius = 5;
    int close_radius = 2;
    int edge_refine_radius = 20;
    int eye_notch_radius = 17;
    int contour_smooth_px = 7;
    int floor_chroma_pct = 9;
    int floor_contrast_pct = 3;
    int subtitle_samples = 150;
    int fade_below_v = 50;
    int fade_match_pct = 88;
    int min_area_per_10000 = 18;
    int line_width = 2;
};

void loadParameters(const std::string& path, Parameters& p);
void saveParameters(const std::string& path, Parameters p);
void validateParameters(Parameters p);

struct Detection {
    cv::Mat mask;
    std::vector<cv::Point> outer;
    std::vector<cv::Point> subpixel_outer;  // 固定点坐标，每像素 16 格。
    cv::Rect bounds;
    int scene_value = 0;
    int area = 0;
    int floor_columns = 0;
    double fade_match = 0.0;
};

struct BodySeed {
    cv::Mat mask;
    cv::Rect bounds;
    int scene_value = 0;
};

struct SubtitleMask {
    cv::Rect bounds;
    cv::Mat strokes;
    cv::Mat cover;
    cv::Mat surround;
};

struct DimReference {
    cv::Mat gray_patch;
    cv::Rect small_bounds;
    cv::Mat mask;
    cv::Rect bounds;
    int area = 0;
};

BodySeed findBodySeed(const cv::Mat& bgr, const Parameters& p);
SubtitleMask calibrateSubtitles(const std::string& input_path, int total, const Parameters& p);
cv::Mat repairSubtitle(const cv::Mat& bgr, const cv::Rect& search, const SubtitleMask& subtitle,
                       int scene_value, cv::Mat& repaired_pixels);
Detection detectSilhouette(const cv::Mat& bgr, const Parameters& p, const SubtitleMask& subtitle);
void refineSilhouetteEdges(const cv::Mat& bgr, cv::Mat& solid, int scene_value,
                           bool feet_visible, const cv::Mat& repaired_pixels, const Parameters& p);
DimReference makeDimReference(const cv::Mat& bgr, const Detection& reliable);
void recoverDimSilhouette(const cv::Mat& bgr, Detection& current,
                          const DimReference& before, const DimReference& after,
                          const Parameters& p);
cv::Mat drawSilhouette(const cv::Mat& bgr, const Detection& detection, int line_width);
void finishSilhouette(const cv::Mat& bgr, Detection& detection, const Parameters& p);
cv::Mat makeCutout(const cv::Mat& bgr, const Detection& detection, const SubtitleMask& subtitle);
