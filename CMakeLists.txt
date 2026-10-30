# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
# cmake -S . -B build -G Ninja
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("Image Upscaler")

find_package (Python3 REQUIRED)
message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")

# cmake --build . --target prepare_data
add_custom_target(prepare_data
    COMMAND ${CMAKE_COMMAND} -E env PYTHONUNBUFFERED=1
        ${Python3_EXECUTABLE} -I -X utf8 CNN/prepare_data.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Downloading and preparing training and benchmark datasets"
    USES_TERMINAL
)

# cmake --build . --target train_srcnn / train_vdsr   (one target per architecture in CNN/models.py)
foreach(arch srcnn vdsr)
    add_custom_target(train_${arch}
        COMMAND ${CMAKE_COMMAND} -E env PYTHONUNBUFFERED=1
            ${Python3_EXECUTABLE} -X utf8 CNN/train_cnn.py --arch ${arch}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Training ${arch} model and exporting to ONNX"
        USES_TERMINAL
    )
endforeach()

# cmake --build . --target export_onnx     (exports every models/pth/best_*.pth)
add_custom_target(export_onnx
    COMMAND ${CMAKE_COMMAND} -E env PYTHONUNBUFFERED=1
        ${Python3_EXECUTABLE} -X utf8 CNN/export_onnx.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Exporting trained checkpoints to ONNX"
    USES_TERMINAL
)

# Include sub-projects.
add_subdirectory ("Image Upscaler")
