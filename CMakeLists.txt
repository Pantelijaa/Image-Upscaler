# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
# cmake -S . -B build -G Ninja
cmake_minimum_required (VERSION 3.25)

# Enable Hot Reload for MSVC compilers.
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")

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

# cmake --build . --target train_<arch>_x<scale> / resume_<arch>_x<scale>   (per architecture in CNN/models.py and scale factor)
foreach(arch srcnn vdsr)
    foreach(scale 2 4)
        add_custom_target(train_${arch}_x${scale}
            COMMAND ${CMAKE_COMMAND} -E env PYTHONUNBUFFERED=1
                ${Python3_EXECUTABLE} -X utf8 CNN/train_cnn.py --arch ${arch} --scale ${scale}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Training ${arch} x${scale} model and exporting to ONNX"
            USES_TERMINAL
        )
        add_custom_target(resume_${arch}_x${scale}
            COMMAND ${CMAKE_COMMAND} -E env PYTHONUNBUFFERED=1
                ${Python3_EXECUTABLE} -X utf8 CNN/train_cnn.py --arch ${arch} --scale ${scale} --resume
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Resuming ${arch} x${scale} training from its last checkpoint"
            USES_TERMINAL
        )
    endforeach()
endforeach()

# cmake --build . --target export_onnx     (exports every models/pth/best_*.pth)
add_custom_target(export_onnx
    COMMAND ${CMAKE_COMMAND} -E env PYTHONUNBUFFERED=1
        ${Python3_EXECUTABLE} -X utf8 CNN/export_onnx.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Exporting trained checkpoints to ONNX"
    USES_TERMINAL
)

# cmake --build . --target quantize_onnx   (int8 copy of every float model in models/onnx)
add_custom_target(quantize_onnx
    COMMAND ${CMAKE_COMMAND} -E env PYTHONUNBUFFERED=1
        ${Python3_EXECUTABLE} -X utf8 CNN/export_onnx.py --int8
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Quantizing ONNX models to int8"
    USES_TERMINAL
)

# Include sub-projects.
add_subdirectory ("Image Upscaler")
