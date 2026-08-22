cmake_minimum_required(VERSION 3.20)

project(kami_hair_solver VERSION 0.7.2 LANGUAGES C CXX CUDA)

if(WIN32 AND NOT MSVC)
    message(FATAL_ERROR "CUDA版はMSVC/Visual Studio 2022で構築してください")
endif()

find_package(CUDAToolkit 12.9 REQUIRED)

option(KAMI_BUILD_BLENDER_EXTENSION "Blender Extension を構築する" ON)
set(KAMI_BLENDER_EXECUTABLE "" CACHE FILEPATH "Blender 5.2 実行ファイル")
set(KAMI_EIGEN3_INCLUDE_DIR "" CACHE PATH "Eigen3 の include/eigen3 ディレクトリ")

if(KAMI_EIGEN3_INCLUDE_DIR)
    if(NOT EXISTS "${KAMI_EIGEN3_INCLUDE_DIR}/Eigen/Core")
        message(FATAL_ERROR "KAMI_EIGEN3_INCLUDE_DIR に Eigen/Core がありません")
    endif()
    add_library(kami_eigen INTERFACE)
    target_include_directories(kami_eigen INTERFACE "${KAMI_EIGEN3_INCLUDE_DIR}")
else()
    find_package(Eigen3 3.3 QUIET NO_MODULE)
    if(NOT Eigen3_FOUND)
        include(FetchContent)
        FetchContent_Declare(eigen
            URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
            URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72)
        set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(eigen)
    endif()
    add_library(kami_eigen INTERFACE)
    target_link_libraries(kami_eigen INTERFACE Eigen3::Eigen)
endif()

add_library(kami_hair_solver SHARED
    src/api.cpp
    src/solver.cpp
    src/cuda_backend.cu)
target_compile_features(kami_hair_solver PUBLIC cxx_std_17)
target_compile_features(kami_hair_solver PRIVATE cuda_std_17)
target_compile_definitions(kami_hair_solver PRIVATE KHS_BUILD_DLL)
target_include_directories(kami_hair_solver PRIVATE include src)
target_link_libraries(kami_hair_solver PRIVATE kami_eigen CUDA::cudart_static CUDA::cublas)
set_target_properties(kami_hair_solver PROPERTIES
    CUDA_ARCHITECTURES 120
    CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    WINDOWS_EXPORT_ALL_SYMBOLS NO)
if(WIN32)
    set_target_properties(kami_hair_solver PROPERTIES PREFIX "")
endif()
if(MSVC)
    target_compile_options(kami_hair_solver PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:/W4>
        $<$<COMPILE_LANGUAGE:C,CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:C,CXX>:/EHsc>
        $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
else()
    target_compile_options(kami_hair_solver PRIVATE -Wall -Wextra -Wpedantic)
    if(MINGW)
        target_link_options(kami_hair_solver PRIVATE -static-libgcc -static-libstdc++ -static)
    endif()
endif()

enable_testing()
add_executable(kami_hair_solver_tests tests/test_solver.cpp)
target_compile_features(kami_hair_solver_tests PRIVATE cxx_std_17)
target_include_directories(kami_hair_solver_tests PRIVATE include)
target_link_libraries(kami_hair_solver_tests PRIVATE kami_hair_solver)
add_test(NAME kami_hair_solver_core COMMAND kami_hair_solver_tests)

add_executable(kami_hair_solver_c_api_tests tests/test_c_api.c)
target_compile_features(kami_hair_solver_c_api_tests PRIVATE c_std_11)
target_include_directories(kami_hair_solver_c_api_tests PRIVATE include)
target_link_libraries(kami_hair_solver_c_api_tests PRIVATE kami_hair_solver)
if(MSVC)
    target_compile_options(kami_hair_solver_tests PRIVATE /utf-8)
    target_compile_options(kami_hair_solver_c_api_tests PRIVATE /utf-8)
endif()
add_test(NAME kami_hair_solver_c_api COMMAND kami_hair_solver_c_api_tests)

if(KAMI_BUILD_BLENDER_EXTENSION AND WIN32)
    set(KAMI_EXTENSION_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/blender_extension/kami_hair_solver")
    set(KAMI_EXTENSION_STAGE "${CMAKE_CURRENT_BINARY_DIR}/blender-extension/kami_hair_solver")
    set(KAMI_PACKAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/packages")
    set(KAMI_PACKAGE "${KAMI_PACKAGE_DIR}/kami_hair_solver_3-${PROJECT_VERSION}-windows-x64.zip")
    file(GLOB KAMI_EXTENSION_FILES CONFIGURE_DEPENDS
        "${KAMI_EXTENSION_SOURCE}/*.py"
        "${KAMI_EXTENSION_SOURCE}/*.toml"
        "${KAMI_EXTENSION_SOURCE}/*.md"
        "${KAMI_EXTENSION_SOURCE}/LICENSE")
    add_custom_command(
        OUTPUT "${KAMI_EXTENSION_STAGE}/.stage-complete"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${KAMI_EXTENSION_STAGE}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${KAMI_EXTENSION_STAGE}/bin"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${KAMI_EXTENSION_SOURCE}" "${KAMI_EXTENSION_STAGE}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "$<TARGET_FILE:kami_hair_solver>" "${KAMI_EXTENSION_STAGE}/bin/"
        COMMAND "${CMAKE_COMMAND}" -E touch "${KAMI_EXTENSION_STAGE}/.stage-complete"
        DEPENDS kami_hair_solver ${KAMI_EXTENSION_FILES}
        VERBATIM)
    add_custom_target(stage-blender-extension
        DEPENDS "${KAMI_EXTENSION_STAGE}/.stage-complete")

    if(KAMI_BLENDER_EXECUTABLE)
        if(NOT EXISTS "${KAMI_BLENDER_EXECUTABLE}")
            message(FATAL_ERROR "KAMI_BLENDER_EXECUTABLE がありません: ${KAMI_BLENDER_EXECUTABLE}")
        endif()
        add_custom_target(blender-extension-test
            COMMAND "${KAMI_BLENDER_EXECUTABLE}" --background --factory-startup
                --python-exit-code 1
                --python "${CMAKE_CURRENT_SOURCE_DIR}/tests/blender_extension_smoke.py"
                -- --extension-stage "${KAMI_EXTENSION_STAGE}"
            DEPENDS stage-blender-extension
            VERBATIM)
        add_custom_target(blender-extension
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${KAMI_PACKAGE_DIR}"
            COMMAND "${KAMI_BLENDER_EXECUTABLE}" --command extension build
                --source-dir "${KAMI_EXTENSION_STAGE}"
                --output-filepath "${KAMI_PACKAGE}"
            DEPENDS stage-blender-extension
            BYPRODUCTS "${KAMI_PACKAGE}"
            VERBATIM)
    endif()
endif()
