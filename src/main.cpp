#include "segmentation.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr char window_name[] = "Nailong contour - adjust parameters";
constexpr int compare_header = 32;

cv::Mat comparisonImage(const cv::Mat& original, const cv::Mat& mask,
                        const cv::Mat& result, int frame_number, bool cutout) {
    const cv::Size size((original.cols + 1) / 2, (original.rows + 1) / 2);
    cv::Mat panel(size.height + compare_header, 3 * size.width, CV_8UC3,
                  cv::Scalar(24, 24, 24));
    cv::Mat source_area = panel(cv::Rect(0, compare_header, size.width, size.height));
    cv::Mat mask_area = panel(cv::Rect(size.width, compare_header, size.width, size.height));
    cv::Mat outline_area = panel(cv::Rect(2 * size.width, compare_header, size.width, size.height));
    cv::resize(original, source_area, size, 0, 0, cv::INTER_AREA);
    cv::resize(result, outline_area, size, 0, 0, cv::INTER_AREA);
    cv::Mat small_mask;
    cv::resize(mask, small_mask, size, 0, 0, cv::INTER_NEAREST);
    cv::cvtColor(small_mask, mask_area, cv::COLOR_GRAY2BGR);
    const std::string titles[] = {"Original  |  " + std::to_string(frame_number),
                                  "Binary mask", cutout ? "Black background" : "Outer contour"};
    for (int i = 0; i < 3; ++i) {
        cv::putText(panel, titles[i], cv::Point(i * size.width + 9, 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.56, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
    }
    return panel;
}

void markChanged(int, void* state) {
    *static_cast<bool*>(state) = true;
}

void addSlider(const char* name, int initial, int maximum, bool& changed) {
    cv::createTrackbar(name, window_name, nullptr, maximum, markChanged, &changed);
    cv::setTrackbarPos(name, window_name, initial);
}

void tune(cv::VideoCapture& input, cv::Mat frame, int total, double fps,
          const Parameters& initial, const SubtitleMask& subtitles, const std::string& save_path,
          bool cutout) {
    if (total < 1) {
        throw std::runtime_error("The tuning window needs a file with a known frame count");
    }
    bool changed = true;
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1240, 740);
    if (total > 1) {
        addSlider("Frame", 0, total - 1, changed);
    }
    addSlider("H min", initial.hue_min, 179, changed);
    addSlider("Seed H max", initial.hue_max, 179, changed);
    addSlider("Edge H max", initial.edge_hue_max, 179, changed);
    addSlider("Seed S min", initial.seed_saturation, 255, changed);
    addSlider("Edge S min", initial.edge_saturation, 255, changed);
    addSlider("Seed R-B %", initial.seed_chroma_pct, 100, changed);
    addSlider("Edge max(R,G)-B %", initial.edge_chroma_pct, 100, changed);
    addSlider("Open radius", initial.seed_open_radius, 60, changed);
    addSlider("Recover radius", initial.recover_radius, 80, changed);
    addSlider("Loose open radius", initial.loose_open_radius, 20, changed);
    addSlider("Close radius", initial.close_radius, 20, changed);
    addSlider("Edge search px", initial.edge_refine_radius, 32, changed);
    addSlider("Contour smooth px", initial.contour_smooth_px, 14, changed);
    addSlider("Min area / 10000", initial.min_area_per_10000,
              std::max(500, initial.min_area_per_10000), changed);
    addSlider("Floor contrast %", initial.floor_contrast_pct, 30, changed);
    addSlider("Floor max(R,G)-B %", initial.floor_chroma_pct, 50, changed);

    Parameters p = initial;
    bool playing = false;
    int shown = 0;
    std::cout << "Space: play/pause | A/D: previous/next | S: save settings | Q/Esc: close\n";
    while (cv::getWindowProperty(window_name, cv::WND_PROP_VISIBLE) >= 1) {
        const int requested = total > 1 ? cv::getTrackbarPos("Frame", window_name) : 0;
        if (requested != shown) {
            // 可变帧时间戳下，直接按帧号跳转可能错一帧。往回拖动时从头数帧。
            if (requested < shown) {
                if (!input.set(cv::CAP_PROP_POS_FRAMES, 0)) {
                    throw std::runtime_error("Cannot rewind the input video");
                }
                shown = -1;
            }
            while (shown + 1 < requested) {
                if (!input.grab()) {
                    throw std::runtime_error("Cannot advance to the requested frame");
                }
                ++shown;
            }
            if (!input.read(frame) || frame.empty()) {
                throw std::runtime_error("Cannot decode the requested frame");
            }
            shown = requested;
            changed = true;
        }
        if (changed) {
            p.edge_hue_max = cv::getTrackbarPos("Edge H max", window_name);
            p.hue_max = std::min(cv::getTrackbarPos("Seed H max", window_name), p.edge_hue_max);
            p.hue_min = std::min(cv::getTrackbarPos("H min", window_name), p.hue_max);
            p.seed_saturation = cv::getTrackbarPos("Seed S min", window_name);
            p.edge_saturation = std::min(cv::getTrackbarPos("Edge S min", window_name), p.seed_saturation);
            p.seed_chroma_pct = cv::getTrackbarPos("Seed R-B %", window_name);
            p.edge_chroma_pct = std::min(cv::getTrackbarPos("Edge max(R,G)-B %", window_name), p.seed_chroma_pct);
            p.seed_open_radius = cv::getTrackbarPos("Open radius", window_name);
            p.recover_radius = cv::getTrackbarPos("Recover radius", window_name);
            p.loose_open_radius = cv::getTrackbarPos("Loose open radius", window_name);
            p.close_radius = cv::getTrackbarPos("Close radius", window_name);
            p.edge_refine_radius = cv::getTrackbarPos("Edge search px", window_name);
            p.contour_smooth_px = cv::getTrackbarPos("Contour smooth px", window_name);
            p.min_area_per_10000 = std::max(1, cv::getTrackbarPos("Min area / 10000", window_name));
            p.floor_contrast_pct = cv::getTrackbarPos("Floor contrast %", window_name);
            p.floor_chroma_pct = cv::getTrackbarPos("Floor max(R,G)-B %", window_name);
            Detection detection = detectSilhouette(frame, p, subtitles);
            finishSilhouette(frame, detection, p);
            const cv::Mat result = cutout ? makeCutout(frame, detection, subtitles)
                                          : drawSilhouette(frame, detection, p.line_width);
            cv::imshow(window_name, comparisonImage(frame, detection.mask, result, shown, cutout));
            changed = false;
        }
        const int key = cv::waitKey(playing ? std::max(1, cvRound(1000.0 / fps)) : 25);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
        if (key == 's' || key == 'S') {
            saveParameters(save_path, p);
            std::cout << "Saved " << save_path << '\n';
        }
        if (key == ' ') {
            playing = !playing;
        }
        if (cv::getWindowProperty(window_name, cv::WND_PROP_VISIBLE) < 1) {
            break;
        }
        int next = shown;
        if (key == 'a' || key == 'A') {
            playing = false;
            next = std::max(0, shown - 1);
        } else if (key == 'd' || key == 'D' || playing) {
            next = std::min(total - 1, shown + 1);
            if (next == total - 1) {
                playing = false;
            }
        }
        if (next != shown && cv::getTrackbarPos("Frame", window_name) == shown) {
            cv::setTrackbarPos("Frame", window_name, next);
        }
    }
    cv::destroyAllWindows();
}

cv::VideoWriter openWriter(const std::string& path, const std::string& codec,
                           double fps, const cv::Size& size, bool is_color = true) {
    if (codec.size() != 4) {
        throw std::runtime_error("The codec must have exactly four characters");
    }
    const int fourcc = cv::VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]);
    cv::VideoWriter output(path, cv::CAP_FFMPEG, fourcc, fps, size, is_color);
    if (!output.isOpened()) {
        throw std::runtime_error("Cannot create " + path + " with codec " + codec);
    }
    return output;
}

// FFmpeg 的常用视频编码要求偶数宽高。给奇数尺寸补一列或一行黑色。
cv::Size evenSize(const cv::Size& size) {
    return cv::Size(size.width + size.width % 2, size.height + size.height % 2);
}

cv::Mat padForVideo(const cv::Mat& image) {
    if (image.cols % 2 == 0 && image.rows % 2 == 0) {
        return image;
    }
    cv::Mat padded;
    cv::copyMakeBorder(image, padded, 0, image.rows % 2, 0, image.cols % 2,
                       cv::BORDER_CONSTANT, cv::Scalar());
    return padded;
}

double maskIoU(const Detection& a, const Detection& b) {
    const cv::Rect overlap = a.bounds & b.bounds;
    if (overlap.empty()) {
        return 0.0;
    }
    cv::Mat shared;
    cv::bitwise_and(a.mask(overlap), b.mask(overlap), shared);
    const int intersection = cv::countNonZero(shared);
    return static_cast<double>(intersection) / (a.area + b.area - intersection);
}

void writeMetrics(std::ostream& out, int number, double pts, const cv::Size& size,
                  const Detection& detection, const Detection& previous) {
    const cv::Rect box = detection.bounds;
    const bool found = detection.area > 0;
    out << number << ',' << pts << ',' << detection.scene_value << ',' << detection.area
        << ',' << box.x << ',' << box.y << ',' << box.width << ',' << box.height
        << ',' << (found && box.x == 0) << ',' << (found && box.y == 0)
        << ',' << (found && box.br().x == size.width) << ',' << (found && box.br().y == size.height)
        << ',' << detection.floor_columns << ',' << detection.fade_match << ',';
    if (found && previous.area > 0) {
        out << maskIoU(detection, previous);
    }
    out << '\n';
}

void checkPaths(const std::vector<std::string>& paths) {
    for (std::size_t i = 0; i < paths.size(); ++i) {
        if (paths[i].empty()) {
            continue;
        }
        const auto a = std::filesystem::weakly_canonical(paths[i]);
        for (std::size_t j = 0; j < i; ++j) {
            if (paths[j].empty()) {
                continue;
            }
            const auto b = std::filesystem::weakly_canonical(paths[j]);
            if (a == b || (std::filesystem::exists(a) && std::filesystem::exists(b) &&
                           std::filesystem::equivalent(a, b))) {
                throw std::runtime_error("Please use different paths for each file: " + paths[i]);
            }
        }
    }
}

struct VideoFrame {
    cv::Mat image;
    Detection detection;
    int number = 0;
    double pts_ms = 0.0;
};

void process(cv::VideoCapture& input, cv::Mat frame, int total, double fps,
             const Parameters& p, const SubtitleMask& subtitles,
             const std::string& output_path, const std::string& compare_path,
             const std::string& mask_path, const std::string& metrics_path,
             const std::string& codec, bool cutout) {
    const cv::Size size = frame.size();
    cv::VideoWriter output, mask_output;
    if (!output_path.empty()) {
        output = openWriter(output_path, codec, fps, evenSize(size));
    }
    if (!mask_path.empty()) {
        // 掩膜单独用无损灰度 FFV1，保证解码仍只有 0 和 255 两种值。
        mask_output = openWriter(mask_path, "FFV1", fps, evenSize(size), false);
    }
    cv::VideoWriter comparison;
    if (!compare_path.empty()) {
        const cv::Size compare_size(3 * ((size.width + 1) / 2), (size.height + 1) / 2 + compare_header);
        comparison = openWriter(compare_path, codec, fps, evenSize(compare_size));
    }
    std::ofstream metrics;
    if (!metrics_path.empty()) {
        metrics.open(metrics_path);
        if (!metrics) {
            throw std::runtime_error("Cannot create metrics file: " + metrics_path);
        }
        metrics << "frame,pts_ms,scene_v,mask_px,x,y,width,height,touch_left,touch_top,"
                   "touch_right,touch_bottom,floor_columns,fade_match,prev_iou\n"
                << std::fixed << std::setprecision(6);
    }

    int number = 0;
    int found = 0;
    int recovered = 0;
    Detection previous;
    VideoFrame last_bright;
    std::deque<VideoFrame> pending;
    const int lookahead = std::clamp(cvRound(fps * 0.45), 1, 60);
    double detection_seconds = 0.0;
    const auto start = std::chrono::steady_clock::now();
    auto write_frame = [&](const VideoFrame& item) {
        Detection detection = item.detection;
        const auto before_finish = std::chrono::steady_clock::now();
        finishSilhouette(item.image, detection, p);
        detection_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - before_finish).count();
        if (output.isOpened() || comparison.isOpened()) {
            const cv::Mat rendered = cutout ? makeCutout(item.image, detection, subtitles)
                                             : drawSilhouette(item.image, detection, p.line_width);
            if (output.isOpened()) {
                output.write(padForVideo(rendered));
            }
            if (comparison.isOpened()) {
                comparison.write(padForVideo(comparisonImage(item.image, detection.mask, rendered,
                                                              item.number, cutout)));
            }
        }
        if (mask_output.isOpened()) {
            mask_output.write(padForVideo(detection.mask));
        }
        if (metrics.is_open()) {
            writeMetrics(metrics, item.number, item.pts_ms, size, detection, previous);
        }
        found += detection.area > 0;
        recovered += detection.fade_match > 0.0;
        previous = detection;
        if ((item.number + 1) % 300 == 0) {
            std::cout << "Processed " << item.number + 1 << " frames\n" << std::flush;
        }
    };

    auto write_dim_frames = [&](const VideoFrame* next_bright, std::size_t count) {
        const DimReference before = makeDimReference(last_bright.image, last_bright.detection);
        const DimReference after = next_bright ? makeDimReference(next_bright->image, next_bright->detection)
                                              : DimReference{};
        for (std::size_t i = 0; i < count; ++i) {
            VideoFrame& dim = pending.front();
            const auto before_match = std::chrono::steady_clock::now();
            // 长时间低照度时，超过时间窗口的旧轮廓不可用于补全。
            const bool recent = last_bright.detection.area > 0 && dim.number - last_bright.number <= lookahead;
            recoverDimSilhouette(dim.image, dim.detection, recent ? before : DimReference{}, after, p);
            detection_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - before_match).count();
            write_frame(dim);
            pending.pop_front();
        }
    };

    do {
        if (frame.size() != size) {
            throw std::runtime_error("Frame size changed during decoding");
        }
        const auto before_detection = std::chrono::steady_clock::now();
        Detection detection = detectSilhouette(frame, p, subtitles);
        detection_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - before_detection).count();
        VideoFrame item{std::move(frame), std::move(detection), number, input.get(cv::CAP_PROP_POS_MSEC)};
        ++number;
        if (item.detection.scene_value < p.fade_below_v) {
            pending.push_back(std::move(item));
            if (pending.size() > static_cast<std::size_t>(lookahead)) {
                write_dim_frames(nullptr, 1);
            }
        } else {
            if (!pending.empty()) {
                write_dim_frames(&item, pending.size());
            }
            write_frame(item);
            last_bright = std::move(item);
        }
    } while (input.read(frame) && !frame.empty());
    if (!pending.empty()) {
        write_dim_frames(nullptr, pending.size());
    }
    output.release();
    mask_output.release();
    comparison.release();
    if (metrics.is_open()) {
        metrics.close();
        if (!metrics) {
            throw std::runtime_error("Error writing the metrics file");
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << std::fixed << std::setprecision(3) << size.width << 'x' << size.height
              << " at " << fps << " fps, decoded " << number << ", detected " << found
              << ", empty " << number - found << ", dark-frame repairs " << recovered << '\n'
              << "Segmentation " << 1000.0 * detection_seconds / number << " ms/frame | total "
              << seconds << " s, " << number / seconds << " frames/s\n";
    if (total > 0 && number != total) {
        throw std::runtime_error("The number of decoded frames differs from the video metadata");
    }
}

} // namespace

int main(int argc, char* argv[]) {
    const cv::String keys =
        "{help h usage ? |       | Show help }"
        "{input i        |       | Input video file }"
        "{output o       |       | Output video file }"
        "{mode           | outline | Visual output: outline or black }"
        "{mask           |       | Optional full-resolution binary mask in FFV1 MKV }"
        "{compare        |       | Optional three-panel comparison video }"
        "{metrics        |       | Optional per-frame CSV }"
        "{config c       |       | Optional OpenCV YAML parameters file }"
        "{tune           | false | Open the interactive tuning window }"
        "{save-config    | tuned.yml | File written by S in tuning mode }"
        "{codec          | avc1  | Output FourCC (e.g. avc1, mp4v, FFV1) }"
        "{threads        | 2     | OpenCV processing threads }";
    cv::CommandLineParser args(argc, argv, keys);
    args.about("Nailong outer contour - C++17 / OpenCV 4");
    if (argc == 1 || args.has("help")) {
        args.printMessage();
        return 0;
    }
    try {
        const std::string input_path = args.get<std::string>("input");
        const std::string output_path = args.get<std::string>("output");
        const std::string mode = args.get<std::string>("mode");
        const std::string mask_path = args.get<std::string>("mask");
        const std::string compare_path = args.get<std::string>("compare");
        const std::string metrics_path = args.get<std::string>("metrics");
        const std::string config_path = args.get<std::string>("config");
        const std::string save_path = args.get<std::string>("save-config");
        const std::string codec = args.get<std::string>("codec");
        const bool tuning = args.get<bool>("tune");
        const int threads = args.get<int>("threads");
        if (!args.check()) {
            args.printErrors();
            return 1;
        }
        if (input_path.empty() || (!tuning && output_path.empty() && mask_path.empty() && compare_path.empty()) ||
            threads < 1) {
            throw std::runtime_error("Use --input=FILE and --output, --mask, --compare or --tune. Threads must be positive.");
        }
        if (mode != "outline" && mode != "black") {
            throw std::runtime_error("Use --mode=outline or --mode=black.");
        }
        const bool cutout = mode == "black";
        if (tuning) {
            checkPaths({input_path, save_path});
        } else {
            checkPaths({input_path, config_path, output_path, compare_path, mask_path, metrics_path});
        }
        cv::setNumThreads(threads);
        Parameters p;
        if (!config_path.empty()) {
            loadParameters(config_path, p);
        }
        cv::VideoCapture input(input_path, cv::CAP_FFMPEG);
        cv::Mat first;
        if (!input.isOpened() || !input.read(first) || first.empty()) {
            throw std::runtime_error("Cannot decode input video: " + input_path);
        }
        const double fps = input.get(cv::CAP_PROP_FPS);
        const int total = cvRound(input.get(cv::CAP_PROP_FRAME_COUNT));
        if (!std::isfinite(fps) || fps <= 0.0) {
            throw std::runtime_error("Cannot determine the input frame rate");
        }
        const auto calibration_start = std::chrono::steady_clock::now();
        const SubtitleMask subtitles = calibrateSubtitles(input_path, total, p);
        const double calibration_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - calibration_start).count();
        std::cout << "Subtitle calibration " << calibration_time << " s, "
                  << (subtitles.cover.empty() ? 0 : cv::countNonZero(subtitles.cover))
                  << " candidate pixels\n";
        if (tuning) {
            tune(input, first, total, fps, p, subtitles, save_path, cutout);
        } else {
            process(input, first, total, fps, p, subtitles, output_path, compare_path, mask_path,
                    metrics_path, codec, cutout);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
