# CMakeList.txt : CMake project for Image Upscaler, include source and define
# project specific logic here.
#

include(FetchContent)

set(OPENCV_VERSION "4.12.0")
FetchContent_Declare(
    opencv_prebuilt
    URL "https://github.com/opencv/opencv/releases/download/${OPENCV_VERSION}/opencv-${OPENCV_VERSION}-windows.exe"
    DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_MakeAvailable(opencv_prebuilt)

# The .exe is a self-extracting archive — extract it
set(OPENCV_EXTRACT_DIR "C:/opencv_temp")
if(NOT EXISTS "${OPENCV_EXTRACT_DIR}/opencv/build")
    file(MAKE_DIRECTORY "${OPENCV_EXTRACT_DIR}")
    file(COPY "${opencv_prebuilt_SOURCE_DIR}/opencv-${OPENCV_VERSION}-windows.exe"
         DESTINATION "${OPENCV_EXTRACT_DIR}")
    execute_process(
        COMMAND "${OPENCV_EXTRACT_DIR}/opencv-${OPENCV_VERSION}-windows.exe"
            -o${OPENCV_EXTRACT_DIR} -y
        WORKING_DIRECTORY "${OPENCV_EXTRACT_DIR}"
        RESULT_VARIABLE extract_result
    )
    file(REMOVE "${OPENCV_EXTRACT_DIR}/opencv-${OPENCV_VERSION}-windows.exe")
    if(NOT extract_result EQUAL 0)
        message(FATAL_ERROR "Failed to extract OpenCV. Result: ${extract_result}")
    endif()
endif()

set(OPENCV_BUILD_DIR "${OPENCV_EXTRACT_DIR}/opencv/build")
set(OPENCV_INCLUDE_DIR "${OPENCV_BUILD_DIR}/include")
set(OPENCV_LIB_DIR "${OPENCV_BUILD_DIR}/x64/vc16/lib")
set(OPENCV_BIN_DIR "${OPENCV_BUILD_DIR}/x64/vc16/bin")

# Add source to this project's executable.
add_executable (CMakeTarget
    "Image Upscaler.cpp"
    "core/Image.h" "core/Image.cpp"
    "includes/stb_image.h" "includes/stb_image_write.h"
    "core/Scaler.h" "core/Scaler.cpp"
    "interpolation/IInterpolator.h"
    "interpolation/Bilinear.h" "interpolation/Bilinear.cpp"
    "interpolation/Bicubic.h" "interpolation/Bicubic.cpp"
    "metrics/Metrics.h" "metrics/Metrics.cpp"
    "srcnn/SRCNNUpscaler.h" "srcnn/SRCNNUpscaler.cpp"
)

target_include_directories(CMakeTarget PRIVATE "${OPENCV_INCLUDE_DIR}")
target_link_directories(CMakeTarget PRIVATE "${OPENCV_LIB_DIR}")

# Link debug lib (d suffix) for Debug builds, release lib otherwise
target_link_libraries(CMakeTarget PRIVATE
    $<IF:$<CONFIG:Debug>,opencv_world4120d,opencv_world4120>
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CMakeTarget PROPERTY CXX_STANDARD 20)
endif()

# Copy OpenCV DLLs next to the executable
file(GLOB OPENCV_DLLS "${OPENCV_BIN_DIR}/*.dll")
foreach(dll ${OPENCV_DLLS})
    add_custom_command(TARGET CMakeTarget POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" $<TARGET_FILE_DIR:CMakeTarget>
    )
endforeach()