cmake_minimum_required(VERSION 3.20)
project(VulkanRayTracer LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags:
# - /arch:AVX2 is compatible with both Debug and Release.
# - Optimization flags (/O2, /Oi, etc.) are restricted to Release/RelWithDebInfo so they do not conflict with /RTC1 in Debug.
if(MSVC)
    add_compile_options(/arch:AVX2)
    add_compile_options(
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/Oi>
        $<$<CONFIG:Release>:/Ot>
        $<$<CONFIG:Release>:/fp:fast>
        $<$<CONFIG:RelWithDebInfo>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/Oi>
        $<$<CONFIG:RelWithDebInfo>:/Ot>
        $<$<CONFIG:RelWithDebInfo>:/fp:fast>
    )
else()
    add_compile_options(-march=native)
    add_compile_options(
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:RelWithDebInfo>:-O3>
        $<$<CONFIG:RelWithDebInfo>:-ffast-math>
    )
endif()

# Find the official Vulkan SDK
find_package(Vulkan REQUIRED)

# Automatically download cgltf via FetchContent
include(FetchContent)
FetchContent_Declare(
    cgltf
    GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
    GIT_TAG v1.14
)
FetchContent_MakeAvailable(cgltf)

set(VERT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shader.vert")
set(FRAG_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shader.frag")
set(VERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/shader.vert.spv")
set(FRAG_SPV "${CMAKE_CURRENT_BINARY_DIR}/shader.frag.spv")

# Compile GLSL shaders to SPIR-V using glslc with -O (optimizes SPIR-V instructions, inlines, and eliminates dead code)
add_custom_command(
    OUTPUT ${VERT_SPV}
    COMMAND Vulkan::glslc -O ${VERT_SRC} -o ${VERT_SPV}
    DEPENDS ${VERT_SRC}
    COMMENT "Compiling shader.vert to SPIR-V (optimized)"
)

add_custom_command(
    OUTPUT ${FRAG_SPV}
    COMMAND Vulkan::glslc -O ${FRAG_SRC} -o ${FRAG_SPV}
    DEPENDS ${FRAG_SRC}
    COMMENT "Compiling shader.frag to SPIR-V (optimized)"
)

add_executable(VulkanRayTracer WIN32 main.cpp terrain.cpp ${VERT_SPV} ${FRAG_SPV})

# Include cgltf header and link Vulkan
target_include_directories(VulkanRayTracer PRIVATE ${cgltf_SOURCE_DIR})
target_link_libraries(VulkanRayTracer PRIVATE Vulkan::Vulkan)

# Automatically copy .spv files right next to the executable
add_custom_command(TARGET VulkanRayTracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${VERT_SPV} $<TARGET_FILE_DIR:VulkanRayTracer>/shader.vert.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FRAG_SPV} $<TARGET_FILE_DIR:VulkanRayTracer>/shader.frag.spv
)

# Automatically copy dragon.glb if present in source folder to binary folder
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dragon.glb")
    add_custom_command(TARGET VulkanRayTracer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/dragon.glb" $<TARGET_FILE_DIR:VulkanRayTracer>/dragon.glb
    )
endif()

# Automatically copy JAXA AW3D30 GeoTIFF files if present in source folder to binary folder
file(GLOB TIFF_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.tif")
foreach(tif ${TIFF_FILES})
    add_custom_command(TARGET VulkanRayTracer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${tif}" $<TARGET_FILE_DIR:VulkanRayTracer>/
    )
endforeach()