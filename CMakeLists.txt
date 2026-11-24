cmake_minimum_required(VERSION 3.16)
project(nailong_contour LANGUAGES CXX)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui videoio)

add_executable(nailong_contour
    src/main.cpp
    src/segmentation.cpp
    src/edge_refinement.cpp
    src/silhouette_finish.cpp
    src/subtitles.cpp
    src/dim_recovery.cpp
)
target_compile_features(nailong_contour PRIVATE cxx_std_17)
set_target_properties(nailong_contour PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(nailong_contour PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(nailong_contour PRIVATE ${OpenCV_LIBS})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(nailong_contour PRIVATE -Wall -Wextra -Wpedantic)
endif()
