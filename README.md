# 奶龙外轮廓识别与黑底抠图

本项目用 C++17 和 OpenCV 4.x 对 `homework.mp4` 逐帧做传统图像分割。黑底模式把彩色主体复制到黑色画布，并可同时导出全分辨率二值掩膜。白色外轮廓模式可用于观察边界位置。两种模式都使用同一份分割结果。

## 桌面交付文件夹

- 打开 `处理结果/homework_black.mp4` 播放黑底成片。需要逐像素使用纯黑画面时，打开同目录的 `homework_black_lossless.mkv`。
- `处理结果/homework_mask.mkv` 是全片灰度二值掩膜。`homework_comparison_black.mp4` 是原片、掩膜和黑底成片的三栏对照。
- 同目录的 PNG 文件供查看三栏对照和单帧效果，`frame_metrics_black.csv` 记录逐帧测试数据。
- 输入视频已放在 `原片/homework.mp4`。在本文件夹执行下方命令即可编译和复现结果。

## Ubuntu 24.04 编译和运行

在本项目根目录执行。`ffmpeg` 用于封装原始音轨及精确的逐帧播放时间，可以单独运行 C++ 程序输出画面轨。

```bash
sudo apt update
sudo apt install build-essential cmake libopencv-dev ffmpeg
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

以下命令读取随项目打包的原片，输出黑底无损画面轨、独立灰度掩膜和逐帧统计。`--compare=comparison.mkv` 可额外输出三栏视频，从左到右是原画面、二值掩膜、黑底成片。只需要掩膜时可以省略 `--output`。

```bash
input="原片/homework.mp4"
./build/nailong_contour --input="$input" --config=parameters.yml --mode=black \
  --codec=FFV1 --output=cutout_master.mkv --mask=homework_mask.mkv \
  --metrics=frame_metrics_black.csv
```

脚本将原视频的逐帧呈现时间和音轨写入播放版 MP4。第二条命令将音轨封装进无损 MKV，并使无损画面与掩膜仍以同一个起点播放。无损视频适合读取像素和再次合成。

```bash
bash tools/pack_mp4.sh "$input" cutout_master.mkv homework_black.mp4
ffmpeg -n -copyts -i cutout_master.mkv -i "$input" \
  -map 0:v:0 -map '1:a:0?' -c copy -avoid_negative_ts disabled homework_black_lossless.mkv
```

用 `--mode=outline` 可输出白线。OpenCV 本身以输入的平均帧率写出静音画面轨。`./build/nailong_contour --help` 可查看全部参数。`--threads=2` 是默认值。掩膜输出固定使用 FFV1 与 MKV 容器，解码后 255 表示身体，0 表示背景。

Ubuntu 24.04 的 `libopencv-dev` 软件包提供 OpenCV 4.6。 [Ubuntu 软件包](https://packages.ubuntu.com/noble/libopencv-dev)

## 运行窗口和滑动条

```bash
./build/nailong_contour --input="$input" --config=parameters.yml --mode=black \
  --tune --save-config=tuned.yml
```

- 打开时停在首帧。用 `Frame` 定位，`A` / `D` 单帧移动，空格播放或暂停，`Q` / `Esc` 退出。为匹配可变时间戳视频的准确帧号，往回拖动时从头顺序定位。
- 拖动 H、S、色差、形态学半径、面积、倒影、`Edge search px` 或 `Contour smooth px`，三栏会马上重算当前帧。`S` 键保存当前参数，之后在批处理命令中使用 `--config=tuned.yml`。上界冲突时，程序以较宽上界为限调整较窄上界。宽松的 S 和色差下限也受严格下限约束。
- 窗口显示当前帧的空间分割，适合观察阈值改动。淡出时的邻帧补全结果请看批处理的 `--compare`。修改 `subtitle_samples` 后重新打开窗口会重学字幕。

## 项目结构

```text
奶龙轮廓识别与黑底抠图/
├── CMakeLists.txt            C++17 编译目标和 OpenCV 组件
├── parameters.yml            识别、去倒影和绘制参数
├── README.md                 使用、算法和实测说明
├── 原片/                    原始输入视频
├── 处理结果/                黑底成片、掩膜、对照和逐帧统计
├── src/
│   ├── main.cpp              顺序解码、视频与掩膜输出、滑动条、CSV
│   ├── segmentation.hpp      参数与各步骤的数据结构
│   ├── segmentation.cpp      颜色掩膜、连通域、脚底处理、绘线
│   ├── edge_refinement.cpp   彩色剖面定位边缘、脚底亮度谷、眼缘凹口修补
│   ├── silhouette_finish.cpp  闭合曲线平滑、背景缝隙清理、黑底合成
│   ├── subtitles.cpp         学习反复出现的字幕笔画、重建遮挡位置
│   └── dim_recovery.cpp      暗帧与清楚邻帧的模板匹配
├── tools/
│   └── pack_mp4.sh           把原始时间戳与音轨封装回处理后 MP4
└── validation/
    ├── visual_checks.csv    半尺寸画面检查点与各版结果
    └── edge_points.csv      原分辨率边缘检查点与各版结果
```

## 原视频观察

原片解码得到 1,981 帧，画面为 916 × 888，时长约 33.6 秒。蓝色墙面与暖黄色身体有明显色差，但腹部偏浅，手掌、脚底以及转身后的阴面偏棕黑。前两段的黄色字幕与身体接近，后段浅色字幕会贴到腿或覆盖身体边缘。地板倒影也有暖色，尤其容易跟着脚底连成一个区域。近景裁到了手臂或身体，三处调暗片段还带有明显的压缩色块。

## 分割步骤

| 步骤 | 实现和目的 | 省略后的常见现象 |
| --- | --- | --- |
| 1. 估计亮度 | 半尺寸图转 HSV，均匀采样 V，取第 85 百分位 `V85`。以它缩放色差阈值，小面积亮水印对统计的影响有限。`V85` 过低时输出空掩膜。 | 固定的色差下限在淡出时容易吃掉暗部，过低又会接受正常画面里的杂色。 |
| 2. 找主体种子 | 严格的 H、S 和 `R−B` 联合选暖色。椭圆开运算先削掉窄笔画，再选过面积门槛的最大 8 连通域。主体框仅用于计算范围。 | 字幕可能与暖色区域竞争。用整帧宽松阈值会留下大量杂点。 |
| 3. 处理字幕遮挡 | 在部分较亮的帧上，从主体横向范围之外学习重复出现、带深描边的笔画。正式处理中，本帧的笔画需达到明显的对比度，或在低对比度下也比笔画周围有更多一致响应，才重建重叠区域。重建用笔画上下的原图像素逐列插值，再稍微平滑列间差异。 | 贴到身体的文字可能被接进轮廓，淡入的半透明笔画也可能留在黑底画面中。直接清空笔画像素会在身体上挖出黑洞。 |
| 4. 找回四肢和浅色边缘 | 在原分辨率的动态区域用较宽的 H、S 和 `max(R,G)−B`，补入眼缘和暗手掌。允许两类宽松像素进入最终候选，分别是靠近种子的像素和经过开运算仍足够厚的候选。后者能接回严格分割中断开的脚踝和脚掌。 | 只用粗种子会漏掉低饱和度或细窄的身体部位。 |
| 5. 压制地面倒影 | 主体粗掩膜未触及底边时，在主体底部带提高色差门槛。候选轮廓被拖到画面下方时，再从灰度图里寻找脚掌下的暗谷。只有多列找到相近的暗谷才据此截断暖色倒影；在暗谷高度一致且下面只剩细条时，再去掉少量残留。 | 偏灰的阴影和带暖色的倒影可能连进脚底。把暗谷切分用于全部姿态则容易切掉弯曲或抬起的脚。 |
| 6. 沿原图复查边缘 | 对初始轮廓各点沿法线采样 BGR。真实边缘通常同时满足近内外侧有对比和外侧颜色接近稳定背景。若外侧一小段仍明显比蓝墙暗，允许在此延长搜索，补回偏暗的指尖。脚掌外侧也可在有暖色或边缘反差时补回。朝下的脚底优先找亮度暗谷，狭窄间隙限制搜索距离，字幕重建区沿用已去字幕的粗轮廓。 | 暗手脚会按暖色区域提前截断。墙和地板交界处的深色混色像素也可能在大腿边缘形成小凸起。 |
| 7. 填实主体并判断眼缘 | 利用暗色或偏绿的像素证据，对小眼缘凹口做局部修补。填充最大外轮廓，再找出其内部有蓝墙证据的空隙。眼睛和嘴的颜色仍随主体保留。 | 眼缘低色差造成的开口会留在头部轮廓上，手臂与身体或双腿之间的背景也可能被填入。 |
| 8. 极暗时利用邻帧 | 批处理给暗帧最多约 0.45 秒的前后参考。灰度模板匹配估计少量平移，匹配分数和前景重合度都达到门槛才采用邻帧掩膜。切镜头或几乎全黑时，当前帧证据不够便维持当帧的结果。 | 量化严重的暗帧可能只剩上半身。无条件套用邻帧会让切镜头或快速动作留下旧轮廓。 |
| 9. 平滑并重建二值掩膜 | 沿闭合曲线以约 1.45 像素的间距重采样。用窄高斯窗口平均，再减去一部分宽窗口的平均，压低锯齿并抵消普通平均造成的边缘内缩。暗场略加大窗口，暗色的尖转弯用较小窗口，贴画边的点保持原位。用亚像素曲线填出掩膜，从已确认的蓝色区域找回真实空隙。狭长亮缝需要两侧的较暗暖色边缘和连续的竖向证据。 | 暗场肩膀和眼侧会有毛刺。过量平滑会把手指、下颌和脚尖磨圆。接触在一起的双脚也可能把腿缝里的蓝地板包在掩膜里。 |
| 10. 合成黑底 | 将整张画布置 0，用最终掩膜把彩色主体复制进去。若边缘两像素内有可确认的蓝墙混色，用内外两侧的颜色估计该像素的前景成分。已检测到的覆盖字幕使用步骤 3 的重建色。 | 掩膜本身只显示形状，没有原画色彩。直接复制含混色的边缘像素，在黑底上会留细蓝边。 |

字幕校准的采样只用来学习反复出现的笔画。正式识别按解码顺序处理全部帧，读取到结束才关闭视频输出。

## 读代码时常用的 OpenCV 函数

`cv::Mat` 保存图像像素或中间掩膜。`frame(rect)` 是指向原有数据的小窗口。需要独立修改像素时用 `.clone()`。单通道掩膜 `CV_8UC1` 每个像素有一个 8 位值，本项目用 `255` 表示前景、`0` 表示背景。

| 调用 | 参数和输出 | 在本项目中的用途 |
| --- | --- | --- |
| `VideoCapture(path, CAP_FFMPEG)`、`read(frame)` | `path` 为视频路径，`read` 成功时返回 `true` 并写入下一帧的 BGR 图像。`get` 读取 FPS、帧数及毫秒时间戳。校准用 `grab` 顺序推进，到采样点才用 `retrieve` 取图。 | 按顺序覆盖原视频，统计真实解码帧数。 [VideoCapture](https://docs.opencv.org/4.6.0/d8/dfe/classcv_1_1VideoCapture.html) |
| `resize(src, dst, size, 0, 0, interpolation)` | `size` 是目标宽高。`INTER_AREA` 适合缩小彩色视频帧。`INTER_NEAREST` 复制最近像素，使上采样和预览后二值值仍为 0 或 255。 | 粗定位、模板匹配和三栏缩略预览。 [几何变换](https://docs.opencv.org/4.6.0/da/d54/group__imgproc__transform.html) |
| `cvtColor(src, dst, code)`、`extractChannel(src, dst, channel)` | 转换枚举指定 BGR→HSV 或 BGR→灰度等。8 位 HSV 的 H 为 0–179，S 和 V 为 0–255。`extractChannel` 的 0、1、2 对应 B、G、R。 | 用 HSV 表示颜色范围，用原色通道算蓝背景与暖色的差异。 [颜色转换](https://docs.opencv.org/4.6.0/de/d25/imgproc_color_conversions.html) |
| `inRange(src, low, high, dst)`、`compare`、`subtract`、`bitwise_and/or` | `inRange` 用含端点的区间逐通道检查，符合条件得到 255。`compare` 对图像与标量执行比较，`subtract` 在 8 位无符号图像上将负值截到 0，按位与/或合并掩膜。 | 暖色、亮度、相邻像素和色差等条件可分别查看，再组合。 [矩阵运算](https://docs.opencv.org/4.6.0/d2/de8/group__core__array.html) |
| `getStructuringElement(MORPH_ELLIPSE, size)`、`erode`、`dilate`、`morphologyEx` | 核描述检查哪些邻居。腐蚀看局部最小值，膨胀看局部最大值。`MORPH_OPEN` 先腐蚀后膨胀，`MORPH_CLOSE` 顺序相反。 | 削掉窄字幕、放宽主体邻域、连上小断缝。对灰度图做腐蚀还可取得字幕附近的局部暗值。 [图像滤波](https://docs.opencv.org/4.6.0/d4/d86/group__imgproc__filter.html) |
| `GaussianBlur(src, dst, size, sigmaX)`、`medianBlur(src, dst, ksize)` | 高斯滤波用权重做局部平均。中值滤波用邻域中值替换中心像素，核边长取奇数。 | 前者平滑脚底亮度及字幕插值，后者去除二值边缘的单像素尖刺。 [图像滤波](https://docs.opencv.org/4.6.0/d4/d86/group__imgproc__filter.html) |
| `connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S)` | 返回标签数，0 号为背景。`labels` 保存每像素标签，`stats` 给出各区域面积与外接矩形，`centroids` 是质心。`8` 使斜邻居也相连。 | 用面积在候选色块里选出人物种子。 [形状分析](https://docs.opencv.org/4.6.0/d3/dc0/group__imgproc__shape.html) |
| `findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, offset)` | 输出若干条边界点序列。`RETR_EXTERNAL` 取最外层，`CHAIN_APPROX_SIMPLE` 压缩共线点，`offset` 把局部坐标加回原图。`contourArea` 计算面积供筛选，`boundingRect` 计算点集外接矩形用于定位和统计。 | 得到沿身体形状走的点，去掉孤立候选。 [形状分析](https://docs.opencv.org/4.6.0/d3/dc0/group__imgproc__shape.html) |
| `drawContours(mask, ..., FILLED)`、`fillPoly(mask, polygons, Scalar(255), LINE_8, shift)`、`line(image, a, b, color, width, LINE_AA, shift)` | 两种填充函数将闭合内部置为 255。`shift=4` 表示输入整数坐标带 4 位小数，例如 x=1617 表示 101.0625 像素。`LINE_8` 保持输出掩膜只有 0 和 255。`LINE_AA` 在彩色图上平滑白线。 | 填充闭合区域，或绘制白色外边线并跳过画框上的人工闭合线段。 [绘图函数](https://docs.opencv.org/4.6.0/d6/d6e/group__imgproc__draw.html) |
| `remap(src, samples, map_x, map_y, interpolation)` | 两张浮点坐标图存放原图采样点的位置。`INTER_LINEAR` 插值颜色，`INTER_NEAREST` 采样 0/255 掩膜。 | 同时比较很多条轮廓法线上的颜色变化。 [几何变换](https://docs.opencv.org/4.6.0/da/d54/group__imgproc__transform.html) |
| `cv::Mat::zeros(size, CV_8UC3)`、`source.copyTo(output, mask)` | 前者构建全零的三通道彩色画布。后者把同大小源图的像素复制到输出，只在 `mask` 非零的位置写入。 | 在纯黑背景上复制奶龙可见的彩色像素。 [cv::Mat](https://docs.opencv.org/4.6.0/d3/d63/classcv_1_1Mat.html) |
| `matchTemplate(search, templ, scores, TM_CCOEFF_NORMED)`、`minMaxLoc`、`warpAffine` | 滑动小模板生成归一化相关分数图。最大值的位置给出平移量。`warpAffine` 用 2×3 矩阵、最近邻采样和黑色边界平移二值掩膜。 | 在调暗遮盖颜色细节时，确认同一姿态后借用清晰邻帧的形状。 [模板匹配](https://docs.opencv.org/4.6.0/df/dfb/group__imgproc__object.html) |
| `VideoWriter`、`write`、`release` | 指定文件名、FourCC、帧率、尺寸和彩色标记，`write` 追加帧，`release` 关闭编码器并完成文件。 | 用同一帧率写彩色结果、对照画面和灰度 FFV1 掩膜。 [VideoWriter](https://docs.opencv.org/4.6.0/dd/d9e/classcv_1_1VideoWriter.html) |
| `createTrackbar`、`getTrackbarPos`、`imshow`、`waitKey` | 创建整数滑块，读当前值，显示图像，并处理窗口输入事件。 | 暂停在同一帧，拖动滑块观察掩膜变化。 [OpenCV Trackbar 教程](https://docs.opencv.org/4.6.0/da/d6a/tutorial_trackbar.html) |

## 关键参数怎么调

先停在有问题的一帧，看中间的掩膜和右边的成片。一次改一组参数，同步复查站立、抬手和转身三类画面。像素被切掉时优先调边缘搜索或颜色条件，位置正确但有锯齿时调平滑尺度。

| 参数及默认值 | 调节方法 |
| --- | --- |
| `hue_min=0`、`hue_max=52`、`edge_hue_max=91` | 前两个限定严格种子的 H，第三个放宽边缘的 H。绿色眼缘出现缺口时，先小幅提高宽松上界。过宽会吃进混色背景。 |
| `seed_saturation=46`、`edge_saturation=12` | 调低宽松下限可以补回浅色或暗色边缘。边缘出现灰白噪点时反向微调，观察手脚是否一起缩水。 |
| `seed_chroma_pct=15`、`edge_chroma_pct=3`、`visible_v=5` | 色差阈值按 `V85 × 百分比 / 100` 取整，并保留很小的绝对下限。暗手遗漏时小幅降低宽松色差值。`visible_v` 控制几乎全黑的丢弃门槛。 |
| `seed_open_radius=12`、`recover_radius=18`、`loose_open_radius=5`、`close_radius=2` | 单位是原图像素半径，半尺寸步骤会折算。字幕粘连先试稍增开运算半径。细脚踝或手指掉落时，试减小 `loose_open_radius` 或增加 `recover_radius`。要补狭缝可将 `close_radius` 提高 1 像素，再确认手臂和身体之间的真实空隙仍存在。 |
| `edge_refine_radius=20`、`eye_notch_radius=17` | 决定外向边缘搜索和眼缘小凹口修补的最大尺度，单位为原图像素。趾尖或手指略内缩时先试把搜索半径提高数像素。动作模糊时过大会追到邻近阴影。眼缘仍有小凹口时小幅提高第二项，并在抬手处确认蓝色间隙。 |
| `contour_smooth_px=7` | 等弧长滤波的尺度，取 0 可直接观察原始轮廓。肩线起锯齿时可每次加 1 像素，然后复查手脚。若脚趾和指尖的转折变钝则往回调。 |
| `min_area_per_10000=18` | 粗种子至少占全画面的 18/10000，约 0.18%。人物很小被跳过时减小，孤立小色块出线时增大。 |
| `floor_chroma_pct=9`、`floor_contrast_pct=3` | 倒影偏灰时优先小幅提高脚部色差门槛。脚尖暗部缩水则往回调。第二项越小越容易接受灰度暗谷，取 0 可以单独关闭暗谷切分。 |
| `subtitle_samples=150` | 设为 0 跳过字幕学习，便于直接对比。字幕样式更多时可增加采样量，再检查学习到的区域有没有吞掉身体。 |
| `fade_below_v=50`、`fade_match_pct=88` | 前者限定可借邻帧的暗度范围，取 0 关闭补全。后者越高越谨慎。转场轮廓错位时提高匹配门槛，再复查极暗帧还能否补齐。 |
| `line_width=2` | 仅用于白线输出模式，单位为原图像素。 |

## 黑底版实测

在 Ubuntu 24.04.5 LTS x86-64、GCC 13.3.0、CMake 3.28.3、OpenCV 4.6.0 上完成 Release 编译。处理覆盖全部 1,981 帧，分辨率为 916 × 888，输入平均帧率约 58.981。1,979 帧有前景，编号 1229、1230 的两帧近黑转场输出空掩膜，38 帧使用邻帧补全。分割平均用时 14.56 ms/帧，视频解码并同时写出黑底、掩膜、对照三路 FFV1 的整次运行耗时 81.39 秒，随后单独封装 MP4。

- 回读了全部源帧、灰度掩膜和无损黑底帧。每帧掩膜只有 0 和 255，非零像素数与 CSV 一致。无损视频中所有掩膜外像素均为 `RGB (0, 0, 0)`。用半径 2 像素的核腐蚀掩膜后，剩余的主体内侧像素有 99.12% 与源图的颜色逐通道相等。需要逐像素二值数据或纯黑背景时使用 FFV1。
- 复核了全片的掩膜面积、触边信息、相邻帧交并比及每隔 16 帧抽出的 124 张画面，另检查 32 个针对性帧，覆盖暗场、抬手、踢步、转身、淡入字幕和出界。
- `validation/edge_points.csv` 有 28 个靠近轮廓的身体点和 12 个邻近背景点。白线版命中 26 个身体点，这版 28 个均命中，12 个背景点均排除。编号 245 的 `(789, 582)` 和 `(777, 561)` 是新补回的深色右手点。同一帧 y=826 的右脚掩膜最外侧 x 由 689 扩展到 695。独立的 `validation/visual_checks.csv` 里，51 个身体点和 27 个背景点也均通过。后者用 458 × 444 的半尺寸坐标。两组都是定点回归检查。
- 在编号 590 的暗场，从 y=430 到 839 对两侧 x 坐标计算 4 像素步长的二阶差分，平均绝对值从白线版的 1.08 像素降到 0.55 像素，用于量化这一帧肩至上臂边缘的局部起伏。编号 595 与反馈截图最接近，对照图及原尺寸 PNG 使用了这一帧。编号 729 验证抬手、脚掌和腿缝。编号 16、1280 检查了淡入字幕造成的重叠。
- 在 Xvfb 中启动了黑底调参界面，将 `Contour smooth px` 从 7 调到 8 并用 `S` 保存，交付配置中的默认值为 7。

完整统计位于 `处理结果/frame_metrics_black.csv`。`touch_*` 表示前景是否触边，`floor_columns` 是以灰度谷去倒影时实际删过像素的列数，`fade_match` 是采用的邻帧模板相关分数，`prev_iou` 是相邻非空掩膜的交并比。帧号从 0 开始。`pts_ms` 记录 OpenCV 解码器返回的毫秒值，后端缺失时可能返回 0，精确的播放时间以封装后的 MP4 中的 PTS 为准。

`homework_black.mp4` 和 `homework_comparison_black.mp4` 保留原片全部 1,981 个视频帧的呈现时间、视频总时长和原 AAC 数据。无损黑底 MKV 与掩膜 MKV 使用同一组平均帧率时间戳，可按相同帧号对齐。

### 仍有误差的位置

- 快速甩手和脚贴地时，手指、脚尖与背景混色，脚底暗边也会与倒影相接。个别模糊边缘仍可能偏移少量像素，极细的夹缝也可能残留局部混色。
- 压在身体上的字幕区域使用邻近像素估色，纹理会变平滑。字幕刚淡入时的极低对比度笔画仍可能留下淡痕。
