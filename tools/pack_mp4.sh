#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo 'Usage: pack_mp4.sh ORIGINAL.mp4 OPENCV_VIDEO.mkv OUTPUT.mp4' >&2
    exit 2
fi

source_video=$1
processed_video=$2
output_video=$3
temp_dir=$(mktemp -d)
trap 'rm -rf -- "$temp_dir"' EXIT

video_timebase=$(ffprobe -v error -select_streams v:0 -show_entries stream=time_base \
    -of default=noprint_wrappers=1:nokey=1 "$source_video")
if [[ ! $video_timebase =~ ^1/([1-9][0-9]*)$ ]]; then
    echo 'Expected a source video time base of the form 1/N.' >&2
    exit 1
fi
timescale=${BASH_REMATCH[1]}

# 读取源视频的逐帧呈现时间。任意短间隔都保留，画面顺序由 C++ 程序提供。
ffprobe -v error -select_streams v:0 -show_entries frame=best_effort_timestamp \
    -of csv=p=0 "$source_video" | awk -F, '$1 ~ /^-?[0-9]+$/ {print $1}' > "$temp_dir/pts.txt"
source_count=$(wc -l < "$temp_dir/pts.txt")
processed_count=$(ffprobe -v error -count_frames -select_streams v:0 \
    -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 "$processed_video")
if [[ ! $processed_count =~ ^[1-9][0-9]*$ ]] || [[ $source_count -ne $processed_count ]]; then
    echo 'Source and processed video must have the same nonzero number of frames.' >&2
    exit 1
fi
start_pts=$(ffprobe -v error -select_streams v:0 -show_entries stream=start_pts \
    -of default=noprint_wrappers=1:nokey=1 "$source_video")
duration_ts=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration_ts \
    -of default=noprint_wrappers=1:nokey=1 "$source_video")
last_pts=$(tail -n 1 "$temp_dir/pts.txt")
if [[ ! $start_pts =~ ^-?[0-9]+$ || ! $duration_ts =~ ^[0-9]+$ ]]; then
    echo 'Expected a source stream with a known start and duration.' >&2
    exit 1
fi
last_duration=$((start_pts + duration_ts - last_pts))
if ((last_duration <= 0)); then
    echo 'Could not determine the final frame duration.' >&2
    exit 1
fi

# 用二分查找式把帧号 N 映射为源时间戳。这个文件只在封装时使用。
awk -v tb="$video_timebase" '
    { pts[NR - 1] = $1 }
    function tree(l, r, m) {
        if (r - l == 1) { printf "%s", pts[l]; return }
        m = int((l + r) / 2)
        printf "if(lt(N,%d),", m
        tree(l, m)
        printf ","
        tree(m, r)
        printf ")"
    }
    END {
        printf "settb=expr=%s,setpts=\047", tb
        tree(0, NR)
        print "\047"
    }
' "$temp_dir/pts.txt" > "$temp_dir/timing.filter"

ffmpeg -hide_banner -loglevel error -n -copyts -i "$processed_video" -i "$source_video" \
    -map 0:v:0 -map '1:a?' -filter_script:v "$temp_dir/timing.filter" \
    -fps_mode passthrough -enc_time_base "$video_timebase" -video_track_timescale "$timescale" \
    -c:v libx264 -preset veryfast -crf 17 -bf 0 -pix_fmt yuv420p -c:a copy \
    -bsf:v "setts=pts=PTS:dts=DTS:duration=$last_duration" \
    -movflags +faststart "$output_video"
echo "Packaged $source_count frames into $output_video"
