# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("Image Upscaler")

find_package (Python3 REQUIRED)
message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")

# cmake --build . --target train_srcnn
add_custom_target(train_srcnn
    COMMAND ${Python3_EXECUTABLE} CNN/CNN.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Training SRCNN model"
)

# Include sub-projects.
add_subdirectory ("Image Upscaler")
