# CMakeList.txt : CMake project for Image Upscaler, include source and define
# project specific logic here.
#

include(FetchContent)

# Download pre-built ONNX Runtime (no compilation needed)
set(ONNXRUNTIME_VERSION "1.22.0")
FetchContent_Declare(
    onnxruntime
    URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime)

set(ONNXRUNTIME_ROOT "${onnxruntime_SOURCE_DIR}")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

# Add source to this project's executable.
add_executable (CMakeTarget "Image Upscaler.cpp" "core/Image.h" "core/Image.cpp" "includes/stb_image.h" "includes/stb_image_write.h" "core/Scaler.h" "core/Scaler.cpp" "interpolation/IInterpolator.h" "interpolation/Bilinear.h" "interpolation/Bilinear.cpp" "metrics/Metrics.h" "metrics/Metrics.cpp" "interpolation/Bicubic.h" "interpolation/Bicubic.cpp" "srcnn/SRCNNUpscaler.h")

target_include_directories(CMakeTarget PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_directories(CMakeTarget PRIVATE "${ONNXRUNTIME_LIB_DIR}")
target_link_libraries(CMakeTarget PRIVATE onnxruntime)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CMakeTarget PROPERTY CXX_STANDARD 20)
endif()

# Copy onnxruntime.dll next to the executable so it can be found at runtime
add_custom_command(TARGET CMakeTarget POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll"
        $<TARGET_FILE_DIR:CMakeTarget>
    COMMENT "Copying onnxruntime.dll to output directory"
)

# TODO: Add tests and install targets if needed.
