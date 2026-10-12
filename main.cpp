#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
#include "terrain.h"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <thread>
#include <unordered_map>

const float PI = 3.1415926535897932385f;

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }
    Vec3 operator/(const Vec3& v) const { return Vec3(x / v.x, y / v.y, z / v.z); }
    Vec3 operator-() const { return Vec3(-x, -y, -z); }
};

inline Vec3 operator*(float s, const Vec3& v) {
    return Vec3(v.x * s, v.y * s, v.z * s);
}

struct Vec4 {
    float x, y, z, w;
    Vec4() : x(0), y(0), z(0), w(0) {}
    Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    Vec4 operator+(const Vec4& v) const { return Vec4(x + v.x, y + v.y, z + v.z, w + v.w); }
    Vec4 operator-(const Vec4& v) const { return Vec4(x - v.x, y - v.y, z - v.z, w - v.w); }
    Vec4 operator*(float s) const { return Vec4(x * s, y * s, z * s, w * s); }
    Vec4 operator/(float s) const { return Vec4(x / s, y / s, z / s, w / s); }
};

inline Vec4 operator*(float s, const Vec4& v) {
    return Vec4(v.x * s, v.y * s, v.z * s, v.w * s);
}

inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
inline Vec3 normalize(const Vec3& v) {
    float len = std::sqrt(dot(v, v));
    return (len > 0.0f) ? Vec3(v.x / len, v.y / len, v.z / len) : Vec3(0, 0, 0);
}
inline Vec3 min_v(const Vec3& a, const Vec3& b) {
    return Vec3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}
inline Vec3 max_v(const Vec3& a, const Vec3& b) {
    return Vec3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

inline Vec3 refract_c(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    float cos_theta = std::min(dot(uv * -1.0f, n), 1.0f);
    Vec3 r_out_perp = (uv + n * cos_theta) * etai_over_etat;
    float k = 1.0f - dot(r_out_perp, r_out_perp);
    if (k < 0.0f) return Vec3(0, 0, 0);
    Vec3 r_out_parallel = n * (-std::sqrt(std::abs(k)));
    return r_out_perp + r_out_parallel;
}

const Vec3 SUN_DIR = normalize(Vec3(0.58f, 0.32f, -0.75f));

struct PushConstants {
    float origin[4];
    float horizontal[4];
    float vertical[4];
    float lower_left_corner[4];
    float cam_dir[4];
    float resolution[2];
    float time;
    float exposure;
    float aperture;
    float focal_dist;
    int32_t night_scene;
    int32_t dof_enabled;
    int32_t tlas_count;
    int32_t tlas_offset;
};

struct BVHNodeGPU {
    float aabb_min[4]; // xyz = aabb_min, w = left_child or first_tri
    float aabb_max[4]; // xyz = aabb_max, w = 0 (interior) or tri_count (leaf)
};

struct TLASNodeGPU {
    float aabb_min[4]; // xyz = aabb_min, w = left_child or blas_root
    float aabb_max[4]; // xyz = aabb_max, w = 0 (interior) or 1 (leaf)
};

struct GPUTriangle {
    float v0[4];
    float v1[4];
    float v2[4];
    float n0[4];
    float n1[4];
    float n2[4];
};

struct CPU_Triangle {
    Vec3 v0, v1, v2;
    Vec3 n0, n1, n2;
    Vec3 centroid;
    Vec3 bmin, bmax;
    float mat_id = 3.0f;  // 3.0 = dragon (glass), 4.0 = terrain
    float feature = 0.0f; // 1.0 = water
};

struct SubChunkBLASInfo {
    int chunk_x, chunk_z;
    int sub_x, sub_z;
    Vec3 aabb_min;
    Vec3 aabb_max;
    Vec3 centroid;
    int blas_root_lod[3];
    int current_lod;
};

struct InstanceInfo {
    Vec3 aabb_min;
    Vec3 aabb_max;
    Vec3 centroid;
    int blas_root;
};

// Window & Camera Settings
HWND g_hwnd = nullptr;
bool g_running = true;
bool g_paused = false;
bool g_night_scene = false;
bool g_dof_enabled = false;
float g_aperture = 0.015f;
float g_focal_dist = 8.5f;
float g_adapted_luminance = 1.0f;

int g_width = 1280, g_height = 720;
bool g_mouse_down = false;
POINT g_last_mouse{};

// First-Person Camera State
Vec3 g_cam_pos(0.0f, 2.4f, 6.5f);
float g_cam_yaw = 0.0f;
float g_cam_pitch = -0.08f;
float g_cam_speed = 7.0f;

// Vulkan globals
VkInstance g_instance = VK_NULL_HANDLE;
VkSurfaceKHR g_surface = VK_NULL_HANDLE;
VkPhysicalDevice g_pdevice = VK_NULL_HANDLE;
VkDevice g_device = VK_NULL_HANDLE;
VkQueue g_queue = VK_NULL_HANDLE;
uint32_t g_queue_family = 0;
VkSwapchainKHR g_swapchain = VK_NULL_HANDLE;
VkFormat g_swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;
VkExtent2D g_swapchain_extent{};
std::vector<VkImage> g_swapchain_images;
std::vector<VkImageView> g_swapchain_views;
std::vector<VkFramebuffer> g_framebuffers;
VkRenderPass g_render_pass = VK_NULL_HANDLE;
VkPipelineLayout g_pipeline_layout = VK_NULL_HANDLE;
VkPipeline g_pipeline = VK_NULL_HANDLE;
VkCommandPool g_cmd_pool = VK_NULL_HANDLE;
std::vector<VkCommandBuffer> g_cmd_buffers;

VkDescriptorSetLayout g_desc_set_layout = VK_NULL_HANDLE;
VkDescriptorPool g_desc_pool = VK_NULL_HANDLE;
VkDescriptorSet g_desc_set = VK_NULL_HANDLE;

VkBuffer g_bvh_buffer = VK_NULL_HANDLE;
VkDeviceMemory g_bvh_memory = VK_NULL_HANDLE;
VkBuffer g_tri_buffer = VK_NULL_HANDLE;
VkDeviceMemory g_tri_memory = VK_NULL_HANDLE;
VkBuffer g_caustic_buffer = VK_NULL_HANDLE;
VkDeviceMemory g_caustic_memory = VK_NULL_HANDLE;

// Dynamic Host-Coherent TLAS Buffer (2048 nodes capacity)
VkBuffer g_tlas_buffer = VK_NULL_HANDLE;
VkDeviceMemory g_tlas_memory = VK_NULL_HANDLE;
void* g_tlas_mapped_ptr = nullptr;
const size_t MAX_TLAS_NODES_PER_FRAME = 2048;
const size_t TLAS_FRAME_SLICE_BYTES = MAX_TLAS_NODES_PER_FRAME * sizeof(TLASNodeGPU);

const int MAX_FRAMES_IN_FLIGHT = 2;
VkSemaphore g_image_avail_semaphores[MAX_FRAMES_IN_FLIGHT];
VkSemaphore g_render_finished_semaphores[MAX_FRAMES_IN_FLIGHT];
VkFence g_in_flight_fences[MAX_FRAMES_IN_FLIGHT];
uint32_t g_current_frame = 0;

InstanceInfo g_dragon_instance{};
std::vector<SubChunkBLASInfo> g_subchunk_infos;
int g_active_subchunks_count = 0;
int g_culled_subchunks_count = 0;
int g_lod_counts[3] = { 0, 0, 0 };

std::string resolve_filepath(const std::string& filename) {
    std::ifstream f1(filename, std::ios::binary);
    if (f1.is_open()) return filename;

    char exe_path[MAX_PATH];
    if (GetModuleFileNameA(NULL, exe_path, MAX_PATH) != 0) {
        std::string dir(exe_path);
        size_t last_slash = dir.find_last_of("\\/");
        if (last_slash != std::string::npos) {
            std::string p = dir.substr(0, last_slash + 1) + filename;
            std::ifstream f2(p, std::ios::binary);
            if (f2.is_open()) return p;
        }
    }
    return filename;
}

std::vector<char> read_file(const std::string& filename) {
    std::string path = resolve_filepath(filename);
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) return {};
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    return buffer;
}

uint32_t find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(g_pdevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return 0;
}

void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(g_device, &bufferInfo, nullptr, &buffer);

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(g_device, buffer, &memReqs);

    VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = find_memory_type(memReqs.memoryTypeBits, properties);
    vkAllocateMemory(g_device, &allocInfo, nullptr, &bufferMemory);

    vkBindBufferMemory(g_device, buffer, bufferMemory, 0);
}

void copy_buffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = g_cmd_pool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(g_device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(cmd, srcBuffer, dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(g_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(g_queue);

    vkFreeCommandBuffers(g_device, g_cmd_pool, 1, &cmd);
}

bool load_dragon_glb(const std::string& filename, std::vector<CPU_Triangle>& out_triangles) {
    std::string path = resolve_filepath(filename);

    cgltf_options options{};
    cgltf_data* data = nullptr;
    cgltf_result result = cgltf_parse_file(&options, path.c_str(), &data);
    if (result != cgltf_result_success) return false;

    result = cgltf_load_buffers(&options, data, path.c_str());
    if (result != cgltf_result_success) {
        cgltf_free(data);
        return false;
    }

    std::vector<Vec3> positions;
    std::vector<Vec3> normals;

    auto process_primitive = [&](cgltf_primitive* prim, const float* transform) {
        if (prim->type != cgltf_primitive_type_triangles) return;

        cgltf_accessor* pos_acc = nullptr;
        cgltf_accessor* norm_acc = nullptr;

        for (cgltf_size a = 0; a < prim->attributes_count; ++a) {
            if (prim->attributes[a].type == cgltf_attribute_type_position) {
                pos_acc = prim->attributes[a].data;
            }
            else if (prim->attributes[a].type == cgltf_attribute_type_normal) {
                norm_acc = prim->attributes[a].data;
            }
        }
        if (!pos_acc) return;

        cgltf_size vert_count = pos_acc->count;
        std::vector<Vec3> p_buf(vert_count);
        std::vector<Vec3> n_buf(vert_count, Vec3(0, 0, 0));

        for (cgltf_size i = 0; i < vert_count; ++i) {
            float p[3];
            cgltf_accessor_read_float(pos_acc, i, p, 3);
            if (transform) {
                float x = p[0] * transform[0] + p[1] * transform[4] + p[2] * transform[8] + transform[12];
                float y = p[0] * transform[1] + p[1] * transform[5] + p[2] * transform[9] + transform[13];
                float z = p[0] * transform[2] + p[1] * transform[6] + p[2] * transform[10] + transform[14];
                p_buf[i] = Vec3(x, y, z);
            }
            else {
                p_buf[i] = Vec3(p[0], p[1], p[2]);
            }

            if (norm_acc) {
                float n[3];
                cgltf_accessor_read_float(norm_acc, i, n, 3);
                if (transform) {
                    float nx = n[0] * transform[0] + n[1] * transform[4] + n[2] * transform[8];
                    float ny = n[0] * transform[1] + n[1] * transform[5] + n[2] * transform[9];
                    float nz = n[0] * transform[2] + n[1] * transform[6] + n[2] * transform[10];
                    n_buf[i] = normalize(Vec3(nx, ny, nz));
                }
                else {
                    n_buf[i] = normalize(Vec3(n[0], n[1], n[2]));
                }
            }
        }

        if (prim->indices) {
            cgltf_size idx_count = prim->indices->count;
            for (cgltf_size k = 0; k + 2 < idx_count; k += 3) {
                uint32_t i0 = (uint32_t)cgltf_accessor_read_index(prim->indices, k);
                uint32_t i1 = (uint32_t)cgltf_accessor_read_index(prim->indices, k + 1);
                uint32_t i2 = (uint32_t)cgltf_accessor_read_index(prim->indices, k + 2);

                if (i0 < vert_count && i1 < vert_count && i2 < vert_count) {
                    positions.push_back(p_buf[i0]);
                    positions.push_back(p_buf[i1]);
                    positions.push_back(p_buf[i2]);
                    normals.push_back(n_buf[i0]);
                    normals.push_back(n_buf[i1]);
                    normals.push_back(n_buf[i2]);
                }
            }
        }
        else {
            for (cgltf_size k = 0; k + 2 < vert_count; k += 3) {
                positions.push_back(p_buf[k]);
                positions.push_back(p_buf[k + 1]);
                positions.push_back(p_buf[k + 2]);
                normals.push_back(n_buf[k]);
                normals.push_back(n_buf[k + 1]);
                normals.push_back(n_buf[k + 2]);
            }
        }
        };

    if (data->nodes_count > 0) {
        for (cgltf_size n = 0; n < data->nodes_count; ++n) {
            cgltf_node* node = &data->nodes[n];
            if (!node->mesh) continue;
            float world_matrix[16];
            cgltf_node_transform_world(node, world_matrix);
            for (cgltf_size p = 0; p < node->mesh->primitives_count; ++p) {
                process_primitive(&node->mesh->primitives[p], world_matrix);
            }
        }
    }
    else {
        for (cgltf_size m = 0; m < data->meshes_count; ++m) {
            for (cgltf_size p = 0; p < data->meshes[m].primitives_count; ++p) {
                process_primitive(&data->meshes[m].primitives[p], nullptr);
            }
        }
    }

    cgltf_free(data);
    if (positions.empty()) return false;

    Vec3 bmin(1e30f, 1e30f, 1e30f);
    Vec3 bmax(-1e30f, -1e30f, -1e30f);
    for (const auto& p : positions) {
        bmin = min_v(bmin, p);
        bmax = max_v(bmax, p);
    }

    Vec3 center = (bmin + bmax) * 0.5f;
    Vec3 extent = bmax - bmin;
    float max_dim = std::max(extent.x, std::max(extent.y, extent.z));
    float scale = 2.05f / (max_dim > 1e-4f ? max_dim : 1.0f);

    for (size_t i = 0; i < positions.size(); ++i) {
        Vec3 p = positions[i];
        p.x = (p.x - center.x) * scale;
        p.y = (p.y - bmin.y) * scale + 0.002f;
        p.z = (p.z - center.z) * scale - 3.5f;
        positions[i] = p;
    }

    for (size_t i = 0; i < positions.size(); i += 3) {
        CPU_Triangle tri{};
        tri.v0 = positions[i];
        tri.v1 = positions[i + 1];
        tri.v2 = positions[i + 2];
        tri.n0 = normals[i];
        tri.n1 = normals[i + 1];
        tri.n2 = normals[i + 2];
        tri.centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
        tri.bmin = min_v(tri.v0, min_v(tri.v1, tri.v2));
        tri.bmax = max_v(tri.v0, max_v(tri.v1, tri.v2));
        tri.mat_id = 3.0f;
        tri.feature = 0.0f;
        out_triangles.push_back(tri);
    }

    return true;
}

void generate_fallback_dragon(std::vector<CPU_Triangle>& out_triangles) {
    const int segments = 48;
    const int rings = 12;
    std::vector<Vec3> positions;

    for (int i = 0; i <= segments; ++i) {
        float t = float(i) / float(segments);
        float s = t * 4.0f * PI;
        Vec3 center(std::sin(s * 0.7f) * 0.35f, 0.05f + t * 1.9f + std::sin(s) * 0.12f, -3.5f - std::cos(s * 0.5f) * 0.45f);
        float radius = (0.28f + 0.15f * std::sin(t * PI)) * (1.0f - 0.5f * t);

        for (int j = 0; j <= rings; ++j) {
            float theta = float(j) / float(rings) * 2.0f * PI;
            Vec3 norm = normalize(Vec3(std::cos(theta), std::sin(theta) * 0.5f, -std::cos(theta) * 0.3f));
            positions.push_back(center + norm * radius);
        }
    }

    for (int i = 0; i < segments; ++i) {
        for (int j = 0; j < rings; ++j) {
            uint32_t i0 = i * (rings + 1) + j;
            uint32_t i1 = i0 + 1;
            uint32_t i2 = (i + 1) * (rings + 1) + j;
            uint32_t i3 = i2 + 1;

            auto add_tri = [&](uint32_t a, uint32_t b, uint32_t c) {
                CPU_Triangle tri{};
                tri.v0 = positions[a]; tri.v1 = positions[b]; tri.v2 = positions[c];
                tri.centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
                tri.bmin = min_v(tri.v0, min_v(tri.v1, tri.v2));
                tri.bmax = max_v(tri.v0, max_v(tri.v1, tri.v2));
                tri.mat_id = 3.0f;
                tri.feature = 0.0f;
                out_triangles.push_back(tri);
                };
            add_tri(i0, i2, i1);
            add_tri(i1, i2, i3);
        }
    }
}

void generate_smooth_normals(std::vector<CPU_Triangle>& triangles) {
    if (triangles.empty()) return;

    struct GridKey {
        int32_t x, y, z;
        bool operator==(const GridKey& o) const {
            return x == o.x && y == o.y && z == o.z;
        }
    };

    struct GridKeyHash {
        size_t operator()(const GridKey& k) const {
            size_t h = 2166136261u;
            h = (h ^ size_t(k.x)) * 16777619u;
            h = (h ^ size_t(k.y)) * 16777619u;
            h = (h ^ size_t(k.z)) * 16777619u;
            return h;
        }
    };

    auto make_key = [](const Vec3& p) -> GridKey {
        return GridKey{
            (int32_t)std::floor(p.x * 10000.0f + 0.5f),
            (int32_t)std::floor(p.y * 10000.0f + 0.5f),
            (int32_t)std::floor(p.z * 10000.0f + 0.5f)
        };
        };

    std::unordered_map<GridKey, Vec3, GridKeyHash> normal_acc;
    normal_acc.reserve(triangles.size() * 2);

    for (const auto& tri : triangles) {
        Vec3 e01 = tri.v1 - tri.v0;
        Vec3 e02 = tri.v2 - tri.v0;
        Vec3 e12 = tri.v2 - tri.v1;

        float l01 = std::sqrt(dot(e01, e01));
        float l02 = std::sqrt(dot(e02, e02));
        float l12 = std::sqrt(dot(e12, e12));

        if (l01 < 1e-7f || l02 < 1e-7f || l12 < 1e-7f) continue;

        Vec3 fn = cross(e01, e02);
        float fn_len = std::sqrt(dot(fn, fn));
        if (fn_len < 1e-7f) continue;
        Vec3 un = fn / fn_len;

        float cos0 = std::clamp(dot(e01, e02) / (l01 * l02), -1.0f, 1.0f);
        float a0 = std::acos(cos0);

        float cos1 = std::clamp(dot(e01 * -1.0f, e12) / (l01 * l12), -1.0f, 1.0f);
        float a1 = std::acos(cos1);

        float cos2 = std::clamp(dot(e02 * -1.0f, e12 * -1.0f) / (l02 * l12), -1.0f, 1.0f);
        float a2 = std::acos(cos2);

        normal_acc[make_key(tri.v0)] = normal_acc[make_key(tri.v0)] + un * a0;
        normal_acc[make_key(tri.v1)] = normal_acc[make_key(tri.v1)] + un * a1;
        normal_acc[make_key(tri.v2)] = normal_acc[make_key(tri.v2)] + un * a2;
    }

    for (auto& tri : triangles) {
        Vec3 fn = cross(tri.v1 - tri.v0, tri.v2 - tri.v0);
        Vec3 un = normalize(fn);

        auto get_norm = [&](const Vec3& p) -> Vec3 {
            auto it = normal_acc.find(make_key(p));
            if (it != normal_acc.end()) {
                Vec3 n = normalize(it->second);
                if (dot(n, n) > 0.5f) return n;
            }
            return un;
            };

        tri.n0 = get_norm(tri.v0);
        tri.n1 = get_norm(tri.v1);
        tri.n2 = get_norm(tri.v2);
    }
}

// High-performance Binned Surface Area Heuristic (SAH) BVH Builder
void build_bvh_recursive(std::vector<CPU_Triangle>& tris, int start, int end, int node_idx,
    std::vector<BVHNodeGPU>& nodes, std::vector<GPUTriangle>& out_tris) {
    Vec3 bmin(1e30f, 1e30f, 1e30f);
    Vec3 bmax(-1e30f, -1e30f, -1e30f);
    Vec3 cmin(1e30f, 1e30f, 1e30f);
    Vec3 cmax(-1e30f, -1e30f, -1e30f);

    for (int i = start; i < end; ++i) {
        bmin = min_v(bmin, tris[i].bmin);
        bmax = max_v(bmax, tris[i].bmax);
        cmin = min_v(cmin, tris[i].centroid);
        cmax = max_v(cmax, tris[i].centroid);
    }

    nodes[node_idx].aabb_min[0] = bmin.x; nodes[node_idx].aabb_min[1] = bmin.y; nodes[node_idx].aabb_min[2] = bmin.z;
    nodes[node_idx].aabb_max[0] = bmax.x; nodes[node_idx].aabb_max[1] = bmax.y; nodes[node_idx].aabb_max[2] = bmax.z;

    int count = end - start;
    if (count <= 4) {
        int first_tri = (int)out_tris.size();
        for (int i = start; i < end; ++i) {
            GPUTriangle gt{};
            gt.v0[0] = tris[i].v0.x; gt.v0[1] = tris[i].v0.y; gt.v0[2] = tris[i].v0.z; gt.v0[3] = tris[i].mat_id;
            gt.v1[0] = tris[i].v1.x; gt.v1[1] = tris[i].v1.y; gt.v1[2] = tris[i].v1.z; gt.v1[3] = 0.0f;
            gt.v2[0] = tris[i].v2.x; gt.v2[1] = tris[i].v2.y; gt.v2[2] = tris[i].v2.z; gt.v2[3] = 0.0f;
            gt.n0[0] = tris[i].n0.x; gt.n0[1] = tris[i].n0.y; gt.n0[2] = tris[i].n0.z; gt.n0[3] = tris[i].feature;
            gt.n1[0] = tris[i].n1.x; gt.n1[1] = tris[i].n1.y; gt.n1[2] = tris[i].n1.z; gt.n1[3] = 0.0f;
            gt.n2[0] = tris[i].n2.x; gt.n2[1] = tris[i].n2.y; gt.n2[2] = tris[i].n2.z; gt.n2[3] = 0.0f;
            out_tris.push_back(gt);
        }
        nodes[node_idx].aabb_min[3] = (float)first_tri;
        nodes[node_idx].aabb_max[3] = (float)count;
        return;
    }

    Vec3 cspan = cmax - cmin;
    const int NUM_BINS = 12;
    int best_axis = -1;
    int best_split_idx = -1;
    float best_cost = 1e30f;

    auto aabb_area = [](const Vec3& mn, const Vec3& mx) -> float {
        Vec3 e = mx - mn;
        return 2.0f * (std::max(0.0f, e.x) * std::max(0.0f, e.y) +
            std::max(0.0f, e.y) * std::max(0.0f, e.z) +
            std::max(0.0f, e.z) * std::max(0.0f, e.x));
        };

    struct Bin {
        Vec3 bmin = Vec3(1e30f, 1e30f, 1e30f);
        Vec3 bmax = Vec3(-1e30f, -1e30f, -1e30f);
        int tri_count = 0;
    };

    for (int axis = 0; axis < 3; ++axis) {
        float span = (axis == 0 ? cspan.x : (axis == 1 ? cspan.y : cspan.z));
        if (span < 1e-4f) continue;

        float axis_min = (axis == 0 ? cmin.x : (axis == 1 ? cmin.y : cmin.z));
        Bin bins[NUM_BINS];

        for (int i = start; i < end; ++i) {
            float c = (axis == 0 ? tris[i].centroid.x : (axis == 1 ? tris[i].centroid.y : tris[i].centroid.z));
            int b = std::clamp((int)(((c - axis_min) / span) * NUM_BINS), 0, NUM_BINS - 1);
            bins[b].tri_count++;
            bins[b].bmin = min_v(bins[b].bmin, tris[i].bmin);
            bins[b].bmax = max_v(bins[b].bmax, tris[i].bmax);
        }

        Vec3 left_min[NUM_BINS - 1], left_max[NUM_BINS - 1];
        int left_count[NUM_BINS - 1];
        Vec3 cur_min(1e30f, 1e30f, 1e30f), cur_max(-1e30f, -1e30f, -1e30f);
        int cur_cnt = 0;
        for (int i = 0; i < NUM_BINS - 1; ++i) {
            cur_cnt += bins[i].tri_count;
            cur_min = min_v(cur_min, bins[i].bmin);
            cur_max = max_v(cur_max, bins[i].bmax);
            left_min[i] = cur_min;
            left_max[i] = cur_max;
            left_count[i] = cur_cnt;
        }

        Vec3 right_min[NUM_BINS - 1], right_max[NUM_BINS - 1];
        int right_count[NUM_BINS - 1];
        cur_min = Vec3(1e30f, 1e30f, 1e30f); cur_max = Vec3(-1e30f, -1e30f, -1e30f);
        cur_cnt = 0;
        for (int i = NUM_BINS - 1; i > 0; --i) {
            cur_cnt += bins[i].tri_count;
            cur_min = min_v(cur_min, bins[i].bmin);
            cur_max = max_v(cur_max, bins[i].bmax);
            right_min[i - 1] = cur_min;
            right_max[i - 1] = cur_max;
            right_count[i - 1] = cur_cnt;
        }

        for (int i = 0; i < NUM_BINS - 1; ++i) {
            if (left_count[i] == 0 || right_count[i] == 0) continue;
            float cost = left_count[i] * aabb_area(left_min[i], left_max[i]) +
                right_count[i] * aabb_area(right_min[i], right_max[i]);
            if (cost < best_cost) {
                best_cost = cost;
                best_axis = axis;
                best_split_idx = i;
            }
        }
    }

    int mid = start + count / 2;
    if (best_axis != -1) {
        float span = (best_axis == 0 ? cspan.x : (best_axis == 1 ? cspan.y : cspan.z));
        float axis_min = (best_axis == 0 ? cmin.x : (best_axis == 1 ? cmin.y : cmin.z));
        float split_pos = axis_min + (float(best_split_idx + 1) / float(NUM_BINS)) * span;

        auto it = std::partition(tris.begin() + start, tris.begin() + end,
            [best_axis, split_pos](const CPU_Triangle& t) {
                float c = (best_axis == 0 ? t.centroid.x : (best_axis == 1 ? t.centroid.y : t.centroid.z));
                return c < split_pos;
            });
        mid = (int)(it - tris.begin());
        if (mid == start || mid == end) {
            mid = start + count / 2;
            std::nth_element(tris.begin() + start, tris.begin() + mid, tris.begin() + end,
                [best_axis](const CPU_Triangle& a, const CPU_Triangle& b) {
                    if (best_axis == 0) return a.centroid.x < b.centroid.x;
                    if (best_axis == 1) return a.centroid.y < b.centroid.y;
                    return a.centroid.z < b.centroid.z;
                });
        }
    }
    else {
        int axis = 0;
        if (cspan.y > cspan.x) axis = 1;
        if (cspan.z > (axis == 1 ? cspan.y : cspan.x)) axis = 2;
        std::nth_element(tris.begin() + start, tris.begin() + mid, tris.begin() + end,
            [axis](const CPU_Triangle& a, const CPU_Triangle& b) {
                if (axis == 0) return a.centroid.x < b.centroid.x;
                if (axis == 1) return a.centroid.y < b.centroid.y;
                return a.centroid.z < b.centroid.z;
            });
    }

    int left_child = (int)nodes.size();
    nodes.push_back({});
    nodes.push_back({});

    nodes[node_idx].aabb_min[3] = (float)left_child;
    nodes[node_idx].aabb_max[3] = 0.0f;

    build_bvh_recursive(tris, start, mid, left_child, nodes, out_tris);
    build_bvh_recursive(tris, mid, end, left_child + 1, nodes, out_tris);
}

void build_tlas_recursive(std::vector<InstanceInfo>& instances, int start, int end, int node_idx,
    std::vector<TLASNodeGPU>& out_nodes) {
    Vec3 bmin(1e30f, 1e30f, 1e30f);
    Vec3 bmax(-1e30f, -1e30f, -1e30f);
    Vec3 cmin(1e30f, 1e30f, 1e30f);
    Vec3 cmax(-1e30f, -1e30f, -1e30f);

    for (int i = start; i < end; ++i) {
        bmin = min_v(bmin, instances[i].aabb_min);
        bmax = max_v(bmax, instances[i].aabb_max);
        cmin = min_v(cmin, instances[i].centroid);
        cmax = max_v(cmax, instances[i].centroid);
    }

    bmin.y -= 0.5f;
    bmax.y += 0.5f;

    out_nodes[node_idx].aabb_min[0] = bmin.x; out_nodes[node_idx].aabb_min[1] = bmin.y; out_nodes[node_idx].aabb_min[2] = bmin.z;
    out_nodes[node_idx].aabb_max[0] = bmax.x; out_nodes[node_idx].aabb_max[1] = bmax.y; out_nodes[node_idx].aabb_max[2] = bmax.z;

    int count = end - start;
    if (count == 1) {
        out_nodes[node_idx].aabb_min[3] = (float)instances[start].blas_root;
        out_nodes[node_idx].aabb_max[3] = 1.0f;
        return;
    }

    Vec3 cspan = cmax - cmin;
    int axis = 0;
    if (cspan.y > cspan.x) axis = 1;
    if (cspan.z > (axis == 1 ? cspan.y : cspan.x)) axis = 2;

    int mid = start + count / 2;
    std::nth_element(instances.begin() + start, instances.begin() + mid, instances.begin() + end,
        [axis](const InstanceInfo& a, const InstanceInfo& b) {
            if (axis == 0) return a.centroid.x < b.centroid.x;
            if (axis == 1) return a.centroid.y < b.centroid.y;
            return a.centroid.z < b.centroid.z;
        });

    int left_child = (int)out_nodes.size();
    out_nodes.push_back({});
    out_nodes.push_back({});

    out_nodes[node_idx].aabb_min[3] = (float)left_child;
    out_nodes[node_idx].aabb_max[3] = 0.0f;

    build_tlas_recursive(instances, start, mid, left_child, out_nodes);
    build_tlas_recursive(instances, mid, end, left_child + 1, out_nodes);
}

inline bool cpu_hit_aabb(const Vec3& orig, const Vec3& inv_d, const Vec3& bmin, const Vec3& bmax, float t_max) {
    Vec3 t0 = (bmin - orig) * inv_d;
    Vec3 t1 = (bmax - orig) * inv_d;
    Vec3 tmin = min_v(t0, t1);
    Vec3 tmax = max_v(t0, t1);
    float tenter = std::max(std::max(tmin.x, tmin.y), tmin.z);
    float texit = std::min(std::min(tmax.x, tmax.y), tmax.z);
    return tenter <= texit && texit > 0.0f && tenter < t_max;
}

inline bool cpu_hit_triangle(const Vec3& orig, const Vec3& dir, float t_min, float t_max,
    const GPUTriangle& tri, float& t_hit, Vec3& norm, Vec3& geo_norm) {
    Vec3 v0(tri.v0[0], tri.v0[1], tri.v0[2]);
    Vec3 v1(tri.v1[0], tri.v1[1], tri.v1[2]);
    Vec3 v2(tri.v2[0], tri.v2[1], tri.v2[2]);

    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = cross(dir, e2);
    float det = dot(e1, pvec);
    if (std::abs(det) < 1e-7f) return false;
    float inv_det = 1.0f / det;

    Vec3 tvec = orig - v0;
    float u = dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;

    Vec3 qvec = cross(tvec, e1);
    float v = dot(dir, qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = dot(e2, qvec) * inv_det;
    if (t < t_min || t > t_max) return false;

    t_hit = t;
    geo_norm = normalize(cross(e1, e2));
    float w = 1.0f - u - v;
    Vec3 n0(tri.n0[0], tri.n0[1], tri.n0[2]);
    Vec3 n1(tri.n1[0], tri.n1[1], tri.n1[2]);
    Vec3 n2(tri.n2[0], tri.n2[1], tri.n2[2]);
    norm = normalize(n0 * w + n1 * u + n2 * v);
    if (dot(norm, geo_norm) < 0.0f) norm = norm * -1.0f;
    return true;
}

bool cpu_raycast(const Vec3& orig, const Vec3& dir, float t_min, float t_max,
    const std::vector<BVHNodeGPU>& nodes, const std::vector<GPUTriangle>& tris,
    float& hit_t, Vec3& hit_p, Vec3& hit_n, Vec3& hit_gn, bool& front_face) {
    Vec3 inv_d(1.0f / (std::abs(dir.x) > 1e-6f ? dir.x : (dir.x >= 0 ? 1e-6f : -1e-6f)),
        1.0f / (std::abs(dir.y) > 1e-6f ? dir.y : (dir.y >= 0 ? 1e-6f : -1e-6f)),
        1.0f / (std::abs(dir.z) > 1e-6f ? dir.z : (dir.z >= 0 ? 1e-6f : -1e-6f)));

    float closest = t_max;
    bool hit = false;
    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const auto& node = nodes[node_idx];

        Vec3 bmin(node.aabb_min[0], node.aabb_min[1], node.aabb_min[2]);
        Vec3 bmax(node.aabb_max[0], node.aabb_max[1], node.aabb_max[2]);
        if (!cpu_hit_aabb(orig, inv_d, bmin, bmax, closest)) continue;

        int tri_count = (int)node.aabb_max[3];
        if (tri_count > 0) {
            int first_tri = (int)node.aabb_min[3];
            for (int i = 0; i < tri_count; ++i) {
                float t = 0.0f;
                Vec3 n, gn;
                if (cpu_hit_triangle(orig, dir, t_min, closest, tris[first_tri + i], t, n, gn)) {
                    closest = t;
                    hit = true;
                    hit_t = t;
                    hit_p = orig + dir * t;
                    front_face = (dot(dir, gn) < 0.0f);
                    hit_gn = front_face ? gn : (gn * -1.0f);
                    hit_n = front_face ? n : (n * -1.0f);
                }
            }
        }
        else {
            int left_child = (int)node.aabb_min[3];
            stack[stack_ptr++] = left_child + 1;
            stack[stack_ptr++] = left_child;
        }
    }
    return hit;
}

void compute_caustic_map(const std::vector<BVHNodeGPU>& nodes, const std::vector<GPUTriangle>& tris,
    std::vector<Vec4>& out_caustic_map) {
    const int MAP_RES = 512;
    out_caustic_map.assign(MAP_RES * MAP_RES, Vec4(0, 0, 0, 1));

    const Vec3 light_forward = SUN_DIR * -1.0f;
    Vec3 light_up(0, 1, 0);
    Vec3 light_right = normalize(cross(light_up, light_forward));
    light_up = cross(light_forward, light_right);

    const Vec3 teapot_center(0.0f, 1.0f, -3.5f);
    const Vec3 plane_center = teapot_center - light_forward * 3.5f;

    const float min_x = -4.0f, max_x = 2.0f;
    const float min_z = -4.5f, max_z = 1.5f;
    const float world_span = max_x - min_x;

    const int RAYS_1D = 600;
    const float beam_span = 3.2f;
    const float step = beam_span / float(RAYS_1D);

    const float pixel_size = world_span / float(MAP_RES);
    const float FLUX_SCALE = (step * step) / (pixel_size * pixel_size);

    const float ior_g = 1.520f;
    const Vec3 sigma_a(0.20f, 0.05f, 0.01f);

    struct WavefrontVertex {
        Vec3 hit_r, hit_g, hit_b;
        Vec3 flux;
        float dist;
        bool valid;
    };

    std::vector<WavefrontVertex> wf_grid(RAYS_1D * RAYS_1D);
    unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> workers;

    auto trace_worker = [&](int y_start, int y_end) {
        for (int y = y_start; y < y_end; ++y) {
            float v = (float(y) + 0.5f) * step - (beam_span * 0.5f);
            for (int x = 0; x < RAYS_1D; ++x) {
                float u = (float(x) + 0.5f) * step - (beam_span * 0.5f);
                int idx = y * RAYS_1D + x;
                auto& vert = wf_grid[idx];
                vert.valid = false;

                Vec3 curr_orig = plane_center + light_right * u + light_up * v;
                Vec3 curr_dir = light_forward;
                float current_ior = 1.0f;
                Vec3 exit_p(0, 0, 0);
                Vec3 exit_dir_g(0, 0, 0);
                float total_glass_dist = 0.0f;
                float total_T = 1.0f;
                bool escaped = false;

                for (int bounce = 0; bounce < 6; ++bounce) {
                    float hit_t = 0.0f;
                    Vec3 hit_p, hit_n, hit_gn;
                    bool front_face = true;

                    if (!cpu_raycast(curr_orig, curr_dir, 0.001f, 15.0f, nodes, tris, hit_t, hit_p, hit_n, hit_gn, front_face)) {
                        escaped = (bounce > 0 && current_ior == 1.0f);
                        break;
                    }

                    bool entering = (current_ior == 1.0f);
                    float next_ior = entering ? ior_g : 1.0f;
                    float eta_rel = current_ior / next_ior;

                    Vec3 normal = hit_n;
                    if (dot(curr_dir, normal) > 0.0f) normal = normal * -1.0f;

                    float cos_i = std::min(dot(curr_dir * -1.0f, normal), 1.0f);
                    float r0 = (1.0f - ior_g) / (1.0f + ior_g);
                    r0 *= r0;
                    float F = r0 + (1.0f - r0) * std::pow(1.0f - cos_i, 5.0f);
                    total_T *= (1.0f - F);

                    if (!entering) {
                        total_glass_dist += hit_t;
                    }

                    Vec3 refr = refract_c(curr_dir, normal, eta_rel);
                    if (dot(refr, refr) < 0.001f) {
                        curr_dir = curr_dir - normal * (2.0f * dot(curr_dir, normal));
                        curr_orig = hit_p + normal * 0.002f;
                    }
                    else {
                        curr_dir = normalize(refr);
                        curr_orig = hit_p - normal * 0.002f;
                        current_ior = next_ior;
                    }

                    exit_p = hit_p;
                    exit_dir_g = curr_dir;

                    if (current_ior == 1.0f && curr_dir.y < -0.02f) {
                        float test_t = 0.0f; Vec3 tp, tn, tgn; bool tfront;
                        if (!cpu_raycast(curr_orig, curr_dir, 0.002f, 15.0f, nodes, tris, test_t, tp, tn, tgn, tfront)) {
                            escaped = true;
                            break;
                        }
                    }
                }

                if (!escaped || exit_dir_g.y >= -0.02f) continue;

                float tg = -exit_p.y / exit_dir_g.y;
                if (tg <= 0.0f) continue;

                Vec3 dir_r = normalize(exit_dir_g + Vec3(0.008f, 0.004f, 0.0f));
                Vec3 dir_b = normalize(exit_dir_g - Vec3(0.008f, 0.004f, 0.0f));

                float tr = -exit_p.y / (dir_r.y < -0.02f ? dir_r.y : exit_dir_g.y);
                float tb = -exit_p.y / (dir_b.y < -0.02f ? dir_b.y : exit_dir_g.y);

                vert.hit_g = exit_p + exit_dir_g * tg;
                vert.hit_r = exit_p + dir_r * tr;
                vert.hit_b = exit_p + dir_b * tb;
                vert.dist = tg;

                float cos_ground = -exit_dir_g.y;
                Vec3 flux_rgb(
                    total_T * std::exp(-sigma_a.x * total_glass_dist),
                    total_T * std::exp(-sigma_a.y * total_glass_dist),
                    total_T * std::exp(-sigma_a.z * total_glass_dist)
                );

                vert.flux = flux_rgb * (cos_ground * FLUX_SCALE);
                vert.valid = true;
            }
        }
        };

    int rows_per_thread = RAYS_1D / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int y_start = t * rows_per_thread;
        int y_end = (t == num_threads - 1) ? RAYS_1D : y_start + rows_per_thread;
        workers.emplace_back(trace_worker, y_start, y_end);
    }
    for (auto& w : workers) w.join();
    workers.clear();

    std::vector<std::vector<Vec4>> thread_grids(num_threads, std::vector<Vec4>(MAP_RES * MAP_RES, Vec4(0, 0, 0, 0)));

    auto raster_worker = [&](unsigned int thread_id, int y_start, int y_end) {
        auto& local_grid = thread_grids[thread_id];

        auto to_pixel = [&](float wx, float wz, float& px, float& pz) -> bool {
            float u = (wx - min_x) / world_span;
            float v = (wz - min_z) / world_span;
            if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) return false;
            px = u * float(MAP_RES - 1);
            pz = v * float(MAP_RES - 1);
            return true;
            };

        for (int y = y_start; y < y_end; ++y) {
            for (int x = 0; x < RAYS_1D - 1; ++x) {
                const auto& v00 = wf_grid[y * RAYS_1D + x];
                const auto& v10 = wf_grid[y * RAYS_1D + (x + 1)];
                const auto& v01 = wf_grid[(y + 1) * RAYS_1D + x];
                const auto& v11 = wf_grid[(y + 1) * RAYS_1D + (x + 1)];

                auto raster_tri = [&](const WavefrontVertex& a, const WavefrontVertex& b, const WavefrontVertex& c) {
                    if (!a.valid || !b.valid || !c.valid) return;

                    float p0x, p0z, p1x, p1z, p2x, p2z;
                    if (!to_pixel(a.hit_g.x, a.hit_g.z, p0x, p0z)) return;
                    if (!to_pixel(b.hit_g.x, b.hit_g.z, p1x, p1z)) return;
                    if (!to_pixel(c.hit_g.x, c.hit_g.z, p2x, p2z)) return;

                    if (std::max({ std::abs(p1x - p0x), std::abs(p2x - p0x), std::abs(p1z - p0z), std::abs(p2z - p0z) }) > 35.0f) {
                        return;
                    }

                    float cross_area = (p1x - p0x) * (p2z - p0z) - (p2x - p0x) * (p1z - p0z);
                    float area_pix = std::abs(cross_area) * 0.5f;
                    Vec3 tri_flux = (a.flux + b.flux + c.flux) * 0.3333f * 0.5f * 2.2f;

                    if (area_pix < 1.5f) {
                        float cx = (p0x + p1x + p2x) * 0.3333f;
                        float cz = (p0z + p1z + p2z) * 0.3333f;

                        int ix = int(std::floor(cx));
                        int iz = int(std::floor(cz));
                        if (ix >= 0 && ix < MAP_RES - 1 && iz >= 0 && iz < MAP_RES - 1) {
                            float fx = cx - float(ix);
                            float fz = cz - float(iz);
                            int p00 = iz * MAP_RES + ix;
                            local_grid[p00].x += tri_flux.x * (1.0f - fx) * (1.0f - fz);
                            local_grid[p00].y += tri_flux.y * (1.0f - fx) * (1.0f - fz);
                            local_grid[p00].z += tri_flux.z * (1.0f - fx) * (1.0f - fz);

                            local_grid[p00 + 1].x += tri_flux.x * fx * (1.0f - fz);
                            local_grid[p00 + 1].y += tri_flux.y * fx * (1.0f - fz);
                            local_grid[p00 + 1].z += tri_flux.z * fx * (1.0f - fz);

                            local_grid[p00 + MAP_RES].x += tri_flux.x * (1.0f - fx) * fz;
                            local_grid[p00 + MAP_RES].y += tri_flux.y * (1.0f - fx) * fz;
                            local_grid[p00 + MAP_RES].z += tri_flux.z * (1.0f - fx) * fz;

                            local_grid[p00 + MAP_RES + 1].x += tri_flux.x * fx * fz;
                            local_grid[p00 + MAP_RES + 1].y += tri_flux.y * fx * fz;
                            local_grid[p00 + MAP_RES + 1].z += tri_flux.z * fx * fz;
                        }
                        return;
                    }

                    int min_ix = std::clamp((int)std::floor(std::min({ p0x, p1x, p2x })), 0, MAP_RES - 1);
                    int max_ix = std::clamp((int)std::ceil(std::max({ p0x, p1x, p2x })), 0, MAP_RES - 1);
                    int min_iz = std::clamp((int)std::floor(std::min({ p0z, p1z, p2z })), 0, MAP_RES - 1);
                    int max_iz = std::clamp((int)std::ceil(std::max({ p0z, p1z, p2z })), 0, MAP_RES - 1);

                    float flux_per_pixel = 1.0f / area_pix;
                    float e0_x = p1x - p0x, e0_z = p1z - p0z;
                    float e1_x = p2x - p1x, e1_z = p2z - p1z;
                    float e2_x = p0x - p2x, e2_z = p0z - p2z;
                    bool ccw = (cross_area > 0.0f);

                    for (int py = min_iz; py <= max_iz; ++py) {
                        float sample_z = float(py) + 0.5f;
                        for (int px = min_ix; px <= max_ix; ++px) {
                            float sample_x = float(px) + 0.5f;
                            float f0 = (sample_x - p0x) * e0_z - (sample_z - p0z) * e0_x;
                            float f1 = (sample_x - p1x) * e1_z - (sample_z - p1z) * e1_x;
                            float f2 = (sample_x - p2x) * e2_z - (sample_z - p2z) * e2_x;

                            bool inside = ccw ? (f0 >= 0.0f && f1 >= 0.0f && f2 >= 0.0f) : (f0 <= 0.0f && f1 <= 0.0f && f2 <= 0.0f);
                            if (inside) {
                                int p_idx = py * MAP_RES + px;
                                local_grid[p_idx].x += tri_flux.x * flux_per_pixel;
                                local_grid[p_idx].y += tri_flux.y * flux_per_pixel;
                                local_grid[p_idx].z += tri_flux.z * flux_per_pixel;
                            }
                        }
                    }
                    };

                raster_tri(v00, v10, v01);
                raster_tri(v10, v11, v01);
            }
        }
        };

    int quad_rows = (RAYS_1D - 1) / num_threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int y_start = t * quad_rows;
        int y_end = (t == num_threads - 1) ? (RAYS_1D - 1) : y_start + quad_rows;
        workers.emplace_back(raster_worker, t, y_start, y_end);
    }
    for (auto& w : workers) w.join();

    for (size_t i = 0; i < out_caustic_map.size(); ++i) {
        Vec4 sum(0, 0, 0, 1);
        for (unsigned int t = 0; t < num_threads; ++t) {
            sum.x += thread_grids[t][i].x;
            sum.y += thread_grids[t][i].y;
            sum.z += thread_grids[t][i].z;
        }
        out_caustic_map[i] = sum;
    }

    std::vector<Vec4> aa_temp = out_caustic_map;
    for (int y = 1; y < MAP_RES - 1; ++y) {
        for (int x = 1; x < MAP_RES - 1; ++x) {
            const Vec4& c = aa_temp[y * MAP_RES + x];
            const Vec4& u = aa_temp[(y - 1) * MAP_RES + x];
            const Vec4& d = aa_temp[(y + 1) * MAP_RES + x];
            const Vec4& l = aa_temp[y * MAP_RES + (x - 1)];
            const Vec4& r = aa_temp[y * MAP_RES + (x + 1)];

            out_caustic_map[y * MAP_RES + x] = Vec4(
                c.x * 0.76f + (u.x + d.x + l.x + r.x) * 0.06f,
                c.y * 0.76f + (u.y + d.y + l.y + r.y) * 0.06f,
                c.z * 0.76f + (u.z + d.z + l.z + r.z) * 0.06f,
                1.0f
            );
        }
    }
}

void init_dragon_bvh() {
    std::vector<CPU_Triangle> dragon_tris;
    if (!load_dragon_glb("dragon.glb", dragon_tris)) {
        generate_fallback_dragon(dragon_tris);
    }
    for (auto& t : dragon_tris) t.mat_id = 3.0f;
    generate_smooth_normals(dragon_tris);

    std::vector<BVHNodeGPU> bvh_nodes;
    std::vector<GPUTriangle> gpu_triangles;

    // 1. Dragon BLAS
    bvh_nodes.push_back({});
    build_bvh_recursive(dragon_tris, 0, (int)dragon_tris.size(), 0, bvh_nodes, gpu_triangles);

    Vec3 d_bmin(1e30f, 1e30f, 1e30f), d_bmax(-1e30f, -1e30f, -1e30f);
    for (const auto& t : dragon_tris) {
        d_bmin = min_v(d_bmin, t.bmin);
        d_bmax = max_v(d_bmax, t.bmax);
    }
    g_dragon_instance.aabb_min = d_bmin;
    g_dragon_instance.aabb_max = d_bmax;
    g_dragon_instance.centroid = (d_bmin + d_bmax) * 0.5f;
    g_dragon_instance.blas_root = 0;

    std::vector<Vec4> caustic_map;
    compute_caustic_map(bvh_nodes, gpu_triangles, caustic_map);

    // 2. Terrain Sub-chunks & LoD BLASes
    std::vector<TerrainChunk> terrain_chunks;
    TerrainSettings t_settings{};
    load_aw3d30_terrain_chunks(".", t_settings, terrain_chunks);

    g_subchunk_infos.clear();

    for (const auto& tc : terrain_chunks) {
        for (const auto& sc : tc.sub_chunks) {
            SubChunkBLASInfo info{};
            info.chunk_x = tc.chunk_x;
            info.chunk_z = tc.chunk_z;
            info.sub_x = sc.sub_x;
            info.sub_z = sc.sub_z;
            info.aabb_min = Vec3(sc.aabb_min[0], sc.aabb_min[1], sc.aabb_min[2]);
            info.aabb_max = Vec3(sc.aabb_max[0], sc.aabb_max[1], sc.aabb_max[2]);
            info.centroid = Vec3(sc.centroid[0], sc.centroid[1], sc.centroid[2]);
            info.current_lod = 0;

            for (int lod = 0; lod < 3; ++lod) {
                const auto& l_tris = sc.lods[lod].triangles;
                if (l_tris.empty()) continue;

                std::vector<CPU_Triangle> c_tris;
                c_tris.reserve(l_tris.size());
                for (const auto& tt : l_tris) {
                    CPU_Triangle tri{};
                    tri.v0 = Vec3(tt.v0[0], tt.v0[1], tt.v0[2]);
                    tri.v1 = Vec3(tt.v1[0], tt.v1[1], tt.v1[2]);
                    tri.v2 = Vec3(tt.v2[0], tt.v2[1], tt.v2[2]);
                    tri.n0 = Vec3(tt.n0[0], tt.n0[1], tt.n0[2]);
                    tri.n1 = Vec3(tt.n1[0], tt.n1[1], tt.n1[2]);
                    tri.n2 = Vec3(tt.n2[0], tt.n2[1], tt.n2[2]);
                    tri.mat_id = tt.v0[3];
                    tri.feature = tt.n0[3];
                    tri.centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
                    tri.bmin = min_v(tri.v0, min_v(tri.v1, tri.v2));
                    tri.bmax = max_v(tri.v0, max_v(tri.v1, tri.v2));
                    c_tris.push_back(tri);
                }

                int blas_root = (int)bvh_nodes.size();
                bvh_nodes.push_back({});
                build_bvh_recursive(c_tris, 0, (int)c_tris.size(), blas_root, bvh_nodes, gpu_triangles);
                info.blas_root_lod[lod] = blas_root;
            }
            g_subchunk_infos.push_back(info);
        }
    }

    VkDeviceSize bvh_size = bvh_nodes.size() * sizeof(BVHNodeGPU);
    VkDeviceSize tri_size = gpu_triangles.size() * sizeof(GPUTriangle);
    VkDeviceSize caustic_size = caustic_map.size() * sizeof(Vec4);
    VkDeviceSize tlas_buffer_size = MAX_FRAMES_IN_FLIGHT * TLAS_FRAME_SLICE_BYTES;

    VkBuffer staging_bvh, staging_tri, staging_caustic;
    VkDeviceMemory staging_bvh_mem, staging_tri_mem, staging_caustic_mem;

    create_buffer(bvh_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_bvh, staging_bvh_mem);
    void* data;
    vkMapMemory(g_device, staging_bvh_mem, 0, bvh_size, 0, &data);
    memcpy(data, bvh_nodes.data(), (size_t)bvh_size);
    vkUnmapMemory(g_device, staging_bvh_mem);

    create_buffer(tri_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_tri, staging_tri_mem);
    vkMapMemory(g_device, staging_tri_mem, 0, tri_size, 0, &data);
    memcpy(data, gpu_triangles.data(), (size_t)tri_size);
    vkUnmapMemory(g_device, staging_tri_mem);

    create_buffer(caustic_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_caustic, staging_caustic_mem);
    vkMapMemory(g_device, staging_caustic_mem, 0, caustic_size, 0, &data);
    memcpy(data, caustic_map.data(), (size_t)caustic_size);
    vkUnmapMemory(g_device, staging_caustic_mem);

    create_buffer(bvh_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, g_bvh_buffer, g_bvh_memory);
    create_buffer(tri_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, g_tri_buffer, g_tri_memory);
    create_buffer(caustic_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, g_caustic_buffer, g_caustic_memory);

    create_buffer(tlas_buffer_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, g_tlas_buffer, g_tlas_memory);
    vkMapMemory(g_device, g_tlas_memory, 0, tlas_buffer_size, 0, &g_tlas_mapped_ptr);

    copy_buffer(staging_bvh, g_bvh_buffer, bvh_size);
    copy_buffer(staging_tri, g_tri_buffer, tri_size);
    copy_buffer(staging_caustic, g_caustic_buffer, caustic_size);

    vkDestroyBuffer(g_device, staging_bvh, nullptr);
    vkFreeMemory(g_device, staging_bvh_mem, nullptr);
    vkDestroyBuffer(g_device, staging_tri, nullptr);
    vkFreeMemory(g_device, staging_tri_mem, nullptr);
    vkDestroyBuffer(g_device, staging_caustic, nullptr);
    vkFreeMemory(g_device, staging_caustic_mem, nullptr);

    VkDescriptorBufferInfo bvh_info{ g_bvh_buffer, 0, bvh_size };
    VkDescriptorBufferInfo tri_info{ g_tri_buffer, 0, tri_size };
    VkDescriptorBufferInfo caustic_info{ g_caustic_buffer, 0, caustic_size };
    VkDescriptorBufferInfo tlas_info{ g_tlas_buffer, 0, tlas_buffer_size };

    VkWriteDescriptorSet descriptorWrites[4]{};
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = g_desc_set;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &bvh_info;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = g_desc_set;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &tri_info;

    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = g_desc_set;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &caustic_info;

    descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[3].dstSet = g_desc_set;
    descriptorWrites[3].dstBinding = 3;
    descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[3].descriptorCount = 1;
    descriptorWrites[3].pBufferInfo = &tlas_info;

    vkUpdateDescriptorSets(g_device, 4, descriptorWrites, 0, nullptr);
}

VkShaderModule create_shader_module(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule module;
    vkCreateShaderModule(g_device, &createInfo, nullptr, &module);
    return module;
}

void cleanup_swapchain() {
    for (auto fb : g_framebuffers) vkDestroyFramebuffer(g_device, fb, nullptr);
    for (auto iv : g_swapchain_views) vkDestroyImageView(g_device, iv, nullptr);
    if (g_swapchain) vkDestroySwapchainKHR(g_device, g_swapchain, nullptr);
    g_framebuffers.clear();
    g_swapchain_views.clear();
}

void create_swapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_pdevice, g_surface, &caps);

    if (caps.currentExtent.width == 0 || caps.currentExtent.height == 0) return;

    g_swapchain_extent = caps.currentExtent;
    if (g_swapchain_extent.width == UINT32_MAX) {
        g_swapchain_extent = { (uint32_t)g_width, (uint32_t)g_height };
    }

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
        imageCount = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    createInfo.surface = g_surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = g_swapchain_format;
    createInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    createInfo.imageExtent = g_swapchain_extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = caps.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    createInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(g_device, &createInfo, nullptr, &g_swapchain) != VK_SUCCESS) {
        createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        vkCreateSwapchainKHR(g_device, &createInfo, nullptr, &g_swapchain);
    }

    uint32_t actualCount = 0;
    vkGetSwapchainImagesKHR(g_device, g_swapchain, &actualCount, nullptr);
    g_swapchain_images.resize(actualCount);
    vkGetSwapchainImagesKHR(g_device, g_swapchain, &actualCount, g_swapchain_images.data());

    g_swapchain_views.resize(actualCount);
    for (size_t i = 0; i < actualCount; i++) {
        VkImageViewCreateInfo ivInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        ivInfo.image = g_swapchain_images[i];
        ivInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivInfo.format = g_swapchain_format;
        ivInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ivInfo.subresourceRange.levelCount = 1;
        ivInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(g_device, &ivInfo, nullptr, &g_swapchain_views[i]);
    }

    g_framebuffers.resize(actualCount);
    for (size_t i = 0; i < actualCount; i++) {
        VkImageView attachments[] = { g_swapchain_views[i] };
        VkFramebufferCreateInfo fbInfo{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        fbInfo.renderPass = g_render_pass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = attachments;
        fbInfo.width = g_swapchain_extent.width;
        fbInfo.height = g_swapchain_extent.height;
        fbInfo.layers = 1;
        vkCreateFramebuffer(g_device, &fbInfo, nullptr, &g_framebuffers[i]);
    }
}

void init_vulkan() {
    const char* exts[] = { "VK_KHR_surface", "VK_KHR_win32_surface" };
    VkInstanceCreateInfo instInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instInfo.enabledExtensionCount = 2;
    instInfo.ppEnabledExtensionNames = exts;
    vkCreateInstance(&instInfo, nullptr, &g_instance);

    VkWin32SurfaceCreateInfoKHR surfInfo{ VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
    surfInfo.hwnd = g_hwnd;
    surfInfo.hinstance = GetModuleHandle(nullptr);
    vkCreateWin32SurfaceKHR(g_instance, &surfInfo, nullptr, &g_surface);

    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(g_instance, &devCount, nullptr);
    std::vector<VkPhysicalDevice> devs(devCount);
    vkEnumeratePhysicalDevices(g_instance, &devCount, devs.data());
    g_pdevice = devs[0];
    for (auto d : devs) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            g_pdevice = d;
            break;
        }
    }

    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_pdevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(g_pdevice, &qfCount, qfs.data());
    for (uint32_t i = 0; i < qfCount; i++) {
        VkBool32 present = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(g_pdevice, i, g_surface, &present);
        if ((qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
            g_queue_family = i;
            break;
        }
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qInfo.queueFamilyIndex = g_queue_family;
    qInfo.queueCount = 1;
    qInfo.pQueuePriorities = &prio;

    const char* devExts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo devInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &qInfo;
    devInfo.enabledExtensionCount = 1;
    devInfo.ppEnabledExtensionNames = devExts;
    vkCreateDevice(g_pdevice, &devInfo, nullptr, &g_device);
    vkGetDeviceQueue(g_device, g_queue_family, 0, &g_queue);

    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = g_swapchain_format;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkRenderPassCreateInfo rpInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &colorAttachment;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    vkCreateRenderPass(g_device, &rpInfo, nullptr, &g_render_pass);

    VkCommandPoolCreateInfo cpInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpInfo.queueFamilyIndex = g_queue_family;
    cpInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(g_device, &cpInfo, nullptr, &g_cmd_pool);

    VkDescriptorSetLayoutBinding bindings[4]{};
    for (uint32_t b = 0; b < 4; ++b) {
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = 4;
    layoutInfo.pBindings = bindings;
    vkCreateDescriptorSetLayout(g_device, &layoutInfo, nullptr, &g_desc_set_layout);

    VkDescriptorPoolSize poolSizes[1]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 4;

    VkDescriptorPoolCreateInfo descPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    descPoolInfo.poolSizeCount = 1;
    descPoolInfo.pPoolSizes = poolSizes;
    descPoolInfo.maxSets = 1;
    vkCreateDescriptorPool(g_device, &descPoolInfo, nullptr, &g_desc_pool);

    VkDescriptorSetAllocateInfo allocDescInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocDescInfo.descriptorPool = g_desc_pool;
    allocDescInfo.descriptorSetCount = 1;
    allocDescInfo.pSetLayouts = &g_desc_set_layout;
    vkAllocateDescriptorSets(g_device, &allocDescInfo, &g_desc_set);

    create_swapchain();

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &g_desc_set_layout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;
    vkCreatePipelineLayout(g_device, &plInfo, nullptr, &g_pipeline_layout);

    auto vertCode = read_file("shader.vert.spv");
    auto fragCode = read_file("shader.frag.spv");
    VkShaderModule vertModule = create_shader_module(vertCode);
    VkShaderModule fragModule = create_shader_module(fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo iAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    iAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vpState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vpState.viewportCount = 1;
    vpState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 1;
    cb.pAttachments = &cbAtt;

    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dyn{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynStates;

    VkGraphicsPipelineCreateInfo pipeInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vInput;
    pipeInfo.pInputAssemblyState = &iAssembly;
    pipeInfo.pViewportState = &vpState;
    pipeInfo.pRasterizationState = &raster;
    pipeInfo.pMultisampleState = &ms;
    pipeInfo.pColorBlendState = &cb;
    pipeInfo.pDynamicState = &dyn;
    pipeInfo.layout = g_pipeline_layout;
    pipeInfo.renderPass = g_render_pass;
    vkCreateGraphicsPipelines(g_device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &g_pipeline);

    vkDestroyShaderModule(g_device, vertModule, nullptr);
    vkDestroyShaderModule(g_device, fragModule, nullptr);

    g_cmd_buffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo cbAlloc{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbAlloc.commandPool = g_cmd_pool;
    cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAlloc.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    vkAllocateCommandBuffers(g_device, &cbAlloc, g_cmd_buffers.data());

    VkSemaphoreCreateInfo semInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkCreateSemaphore(g_device, &semInfo, nullptr, &g_image_avail_semaphores[i]);
        vkCreateSemaphore(g_device, &semInfo, nullptr, &g_render_finished_semaphores[i]);
        vkCreateFence(g_device, &fInfo, nullptr, &g_in_flight_fences[i]);
    }

    init_dragon_bvh();
}

void cleanup_vulkan() {
    vkDeviceWaitIdle(g_device);

    if (g_tlas_buffer) {
        if (g_tlas_mapped_ptr) {
            vkUnmapMemory(g_device, g_tlas_memory);
            g_tlas_mapped_ptr = nullptr;
        }
        vkDestroyBuffer(g_device, g_tlas_buffer, nullptr);
        g_tlas_buffer = VK_NULL_HANDLE;
    }
    if (g_tlas_memory) {
        vkFreeMemory(g_device, g_tlas_memory, nullptr);
        g_tlas_memory = VK_NULL_HANDLE;
    }

    if (g_bvh_buffer) vkDestroyBuffer(g_device, g_bvh_buffer, nullptr);
    if (g_bvh_memory) vkFreeMemory(g_device, g_bvh_memory, nullptr);
    if (g_tri_buffer) vkDestroyBuffer(g_device, g_tri_buffer, nullptr);
    if (g_tri_memory) vkFreeMemory(g_device, g_tri_memory, nullptr);
    if (g_caustic_buffer) vkDestroyBuffer(g_device, g_caustic_buffer, nullptr);
    if (g_caustic_memory) vkFreeMemory(g_device, g_caustic_memory, nullptr);

    if (g_desc_pool) vkDestroyDescriptorPool(g_device, g_desc_pool, nullptr);
    if (g_desc_set_layout) vkDestroyDescriptorSetLayout(g_device, g_desc_set_layout, nullptr);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(g_device, g_image_avail_semaphores[i], nullptr);
        vkDestroySemaphore(g_device, g_render_finished_semaphores[i], nullptr);
        vkDestroyFence(g_device, g_in_flight_fences[i], nullptr);
    }
    vkDestroyCommandPool(g_device, g_cmd_pool, nullptr);
    vkDestroyPipeline(g_device, g_pipeline, nullptr);
    vkDestroyPipelineLayout(g_device, g_pipeline_layout, nullptr);
    cleanup_swapchain();
    vkDestroyRenderPass(g_device, g_render_pass, nullptr);
    vkDestroyDevice(g_device, nullptr);
    vkDestroySurfaceKHR(g_instance, g_surface, nullptr);
    vkDestroyInstance(g_instance, nullptr);
}

void update_first_person_camera(float dt) {
    if (dt <= 0.0f || dt > 0.1f) dt = 0.016f;

    Vec3 fwd(std::sin(g_cam_yaw), 0.0f, -std::cos(g_cam_yaw));
    Vec3 right(std::cos(g_cam_yaw), 0.0f, std::sin(g_cam_yaw));

    float speed = g_cam_speed;
    if (GetAsyncKeyState(VK_SHIFT) & 0x8000) speed *= 3.0f;
    if (GetAsyncKeyState(VK_CONTROL) & 0x8000) speed *= 0.35f;

    Vec3 move(0, 0, 0);
    if ((GetAsyncKeyState('W') & 0x8000) || (GetAsyncKeyState(VK_UP) & 0x8000))    move = move + fwd;
    if ((GetAsyncKeyState('S') & 0x8000) || (GetAsyncKeyState(VK_DOWN) & 0x8000))  move = move - fwd;
    if ((GetAsyncKeyState('D') & 0x8000) || (GetAsyncKeyState(VK_RIGHT) & 0x8000)) move = move + right;
    if ((GetAsyncKeyState('A') & 0x8000) || (GetAsyncKeyState(VK_LEFT) & 0x8000))  move = move - right;
    if (GetAsyncKeyState(VK_SPACE) & 0x8000) move = move + Vec3(0.0f, 1.0f, 0.0f);
    if ((GetAsyncKeyState('C') & 0x8000) || (GetAsyncKeyState('E') & 0x8000)) move = move - Vec3(0.0f, 1.0f, 0.0f);

    float len = std::sqrt(dot(move, move));
    if (len > 1e-4f) {
        g_cam_pos = g_cam_pos + (move / len) * (speed * dt);
    }
}

struct FrustumPlane {
    Vec3 n;
    float d;
};

void render_frame(float time, float dt) {
    update_first_person_camera(dt);

    vkWaitForFences(g_device, 1, &g_in_flight_fences[g_current_frame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult res = vkAcquireNextImageKHR(g_device, g_swapchain, UINT64_MAX,
        g_image_avail_semaphores[g_current_frame],
        VK_NULL_HANDLE, &imageIndex);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
        vkDeviceWaitIdle(g_device);
        cleanup_swapchain();
        create_swapchain();
        return;
    }

    Vec3 forward = normalize(Vec3(
        std::sin(g_cam_yaw) * std::cos(g_cam_pitch),
        std::sin(g_cam_pitch),
        -std::cos(g_cam_yaw) * std::cos(g_cam_pitch)
    ));
    Vec3 right = normalize(cross(forward, Vec3(0.0f, 1.0f, 0.0f)));
    Vec3 up = cross(right, forward);

    float vfov_rad = 55.0f * PI / 180.0f;
    float aspect_ratio = float(g_swapchain_extent.width) / float(g_swapchain_extent.height);
    float h = std::tan(vfov_rad * 0.5f);
    float w = h * aspect_ratio;

    Vec3 vec_TL = normalize(forward - right * w + up * h);
    Vec3 vec_TR = normalize(forward + right * w + up * h);
    Vec3 vec_BL = normalize(forward - right * w - up * h);
    Vec3 vec_BR = normalize(forward + right * w - up * h);

    FrustumPlane planes[6];
    planes[0] = { forward,  -dot(forward,  g_cam_pos + forward * 0.1f) };     // Near
    planes[1] = { -forward, -dot(-forward, g_cam_pos + forward * 25000.0f) }; // Far
    planes[2] = { normalize(cross(vec_BL, up)),    -dot(normalize(cross(vec_BL, up)),    g_cam_pos) }; // Left
    planes[3] = { normalize(cross(up, vec_BR)),    -dot(normalize(cross(up, vec_BR)),    g_cam_pos) }; // Right
    planes[4] = { normalize(cross(vec_TL, right)), -dot(normalize(cross(vec_TL, right)), g_cam_pos) }; // Top
    planes[5] = { normalize(cross(right, vec_BL)), -dot(normalize(cross(right, vec_BL)), g_cam_pos) }; // Bottom

    auto aabb_in_frustum = [&](const Vec3& bmin, const Vec3& bmax) -> bool {
        for (int p = 0; p < 6; ++p) {
            const auto& pl = planes[p];
            Vec3 p_vert(
                pl.n.x > 0.0f ? bmax.x : bmin.x,
                pl.n.y > 0.0f ? bmax.y : bmin.y,
                pl.n.z > 0.0f ? bmax.z : bmin.z
            );
            if (dot(p_vert, pl.n) + pl.d < -1.0f) {
                return false;
            }
        }
        return true;
        };

    const float MAX_SHADOW_DIST = 25000.0f;
    const Vec3 shadow_extrude = SUN_DIR * -MAX_SHADOW_DIST;
    const float MAX_TERRAIN_VIEW_DIST = 22000.0f;

    std::vector<InstanceInfo> active_instances;
    active_instances.reserve(g_subchunk_infos.size() + 1);

    active_instances.push_back(g_dragon_instance);

    int active_subs = 0;
    int culled_subs = 0;
    int lod_counts[3] = { 0, 0, 0 };

    // Expanded LoD distance boundaries: keep high and medium mesh density far into the horizon
    for (auto& sc : g_subchunk_infos) {
        Vec3 closest_pt = max_v(sc.aabb_min, min_v(g_cam_pos, sc.aabb_max));
        Vec3 diff = g_cam_pos - closest_pt;
        float dist = std::sqrt(dot(diff, diff));

        if (dist > MAX_TERRAIN_VIEW_DIST) {
            culled_subs++;
            continue;
        }

        bool near_cam = (dist < 50.0f);
        bool in_frustum = aabb_in_frustum(sc.aabb_min, sc.aabb_max);

        Vec3 swept_min = min_v(sc.aabb_min, sc.aabb_min + shadow_extrude);
        Vec3 swept_max = max_v(sc.aabb_max, sc.aabb_max + shadow_extrude);
        bool in_shadow = aabb_in_frustum(swept_min, swept_max);

        if (near_cam || in_frustum || in_shadow) {
            int target_lod = sc.current_lod;
            if (sc.current_lod == 0) {
                if (dist > 1400.0f) target_lod = 1;
            }
            else if (sc.current_lod == 1) {
                if (dist < 1000.0f) target_lod = 0;
                else if (dist > 7500.0f) target_lod = 2;
            }
            else { // Current is 2
                if (dist < 6000.0f) target_lod = 1;
            }
            sc.current_lod = target_lod;

            active_instances.push_back({
                sc.aabb_min,
                sc.aabb_max,
                sc.centroid,
                sc.blas_root_lod[target_lod]
                });

            active_subs++;
            lod_counts[target_lod]++;
        }
        else {
            culled_subs++;
        }
    }

    g_active_subchunks_count = active_subs;
    g_culled_subchunks_count = culled_subs;
    g_lod_counts[0] = lod_counts[0];
    g_lod_counts[1] = lod_counts[1];
    g_lod_counts[2] = lod_counts[2];

    std::vector<TLASNodeGPU> tlas_nodes;
    tlas_nodes.push_back({});
    build_tlas_recursive(active_instances, 0, (int)active_instances.size(), 0, tlas_nodes);

    int tlas_offset = (int)(g_current_frame * MAX_TLAS_NODES_PER_FRAME);
    uint8_t* dst_ptr = ((uint8_t*)g_tlas_mapped_ptr) + (g_current_frame * TLAS_FRAME_SLICE_BYTES);
    memcpy(dst_ptr, tlas_nodes.data(), tlas_nodes.size() * sizeof(TLASNodeGPU));

    vkResetFences(g_device, 1, &g_in_flight_fences[g_current_frame]);
    VkCommandBuffer cmd = g_cmd_buffers[g_current_frame];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkRenderPassBeginInfo rpBegin{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    rpBegin.renderPass = g_render_pass;
    rpBegin.framebuffer = g_framebuffers[imageIndex];
    rpBegin.renderArea.extent = g_swapchain_extent;
    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{ 0.0f, 0.0f, (float)g_swapchain_extent.width, (float)g_swapchain_extent.height, 0.0f, 1.0f };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{ {0,0}, g_swapchain_extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipeline_layout, 0, 1, &g_desc_set, 0, nullptr);

    float viewport_height = 2.0f * h;
    float viewport_width = aspect_ratio * viewport_height;

    Vec3 w_cam = forward * -1.0f;
    Vec3 u_cam = right;
    Vec3 v_cam = up;

    Vec3 origin = g_cam_pos;
    Vec3 horizontal = u_cam * viewport_width;
    Vec3 vertical = v_cam * viewport_height;
    Vec3 lower_left_corner = origin - (horizontal / 2.0f) - (vertical / 2.0f) - w_cam;

    float sun_alignment = std::max(0.0f, dot(forward, SUN_DIR));
    float target_lum = g_night_scene ? 0.35f : (1.0f + std::pow(sun_alignment, 4.0f) * 1.5f);

    g_adapted_luminance += (target_lum - g_adapted_luminance) * (1.0f - std::exp(-dt * 3.0f));
    float eye_exposure = 1.0f / (g_adapted_luminance);

    PushConstants pc{};
    pc.origin[0] = origin.x; pc.origin[1] = origin.y; pc.origin[2] = origin.z;
    pc.horizontal[0] = horizontal.x; pc.horizontal[1] = horizontal.y; pc.horizontal[2] = horizontal.z;
    pc.vertical[0] = vertical.x; pc.vertical[1] = vertical.y; pc.vertical[2] = vertical.z;
    pc.lower_left_corner[0] = lower_left_corner.x; pc.lower_left_corner[1] = lower_left_corner.y; pc.lower_left_corner[2] = lower_left_corner.z;
    pc.cam_dir[0] = forward.x; pc.cam_dir[1] = forward.y; pc.cam_dir[2] = forward.z;
    pc.resolution[0] = (float)g_swapchain_extent.width;
    pc.resolution[1] = (float)g_swapchain_extent.height;
    pc.time = time;
    pc.exposure = eye_exposure;
    pc.aperture = g_aperture;
    pc.focal_dist = g_focal_dist;
    pc.night_scene = g_night_scene ? 1 : 0;
    pc.dof_enabled = g_dof_enabled ? 1 : 0;
    pc.tlas_count = (int32_t)tlas_nodes.size();
    pc.tlas_offset = tlas_offset;

    vkCmdPushConstants(cmd, g_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    VkSemaphore waitSems[] = { g_image_avail_semaphores[g_current_frame] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSems;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkSemaphore sigSems[] = { g_render_finished_semaphores[g_current_frame] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = sigSems;
    vkQueueSubmit(g_queue, 1, &submitInfo, g_in_flight_fences[g_current_frame]);

    VkPresentInfoKHR presInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    presInfo.waitSemaphoreCount = 1;
    presInfo.pWaitSemaphores = sigSems;
    presInfo.swapchainCount = 1;
    presInfo.pSwapchains = &g_swapchain;
    presInfo.pImageIndices = &imageIndex;
    VkResult presRes = vkQueuePresentKHR(g_queue, &presInfo);
    if (presRes == VK_ERROR_OUT_OF_DATE_KHR || presRes == VK_SUBOPTIMAL_KHR) {
        vkDeviceWaitIdle(g_device);
        cleanup_swapchain();
        create_swapchain();
    }

    g_current_frame = (g_current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CLOSE:
        g_running = false;
        DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_SIZE:
        if (wp != SIZE_MINIMIZED && g_device && LOWORD(lp) > 0 && HIWORD(lp) > 0) {
            g_width = LOWORD(lp);
            g_height = HIWORD(lp);
            vkDeviceWaitIdle(g_device);
            cleanup_swapchain();
            create_swapchain();
        }
        return 0;
    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
        g_mouse_down = true;
        g_last_mouse.x = LOWORD(lp);
        g_last_mouse.y = HIWORD(lp);
        SetCapture(hwnd);
        return 0;
    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
        g_mouse_down = false;
        ReleaseCapture();
        return 0;
    case WM_MOUSEMOVE:
        if (g_mouse_down) {
            int dx = LOWORD(lp) - g_last_mouse.x;
            int dy = HIWORD(lp) - g_last_mouse.y;
            g_last_mouse.x = LOWORD(lp);
            g_last_mouse.y = HIWORD(lp);
            g_cam_yaw += dx * 0.003f;
            g_cam_pitch -= dy * 0.003f;
            g_cam_pitch = std::clamp(g_cam_pitch, -1.52f, 1.52f);
        }
        return 0;
    case WM_MOUSEWHEEL: {
        float delta = GET_WHEEL_DELTA_WPARAM(wp) / 120.0f;
        g_cam_speed = std::clamp(g_cam_speed + delta * 2.0f, 1.0f, 120.0f);
        return 0;
    }
    case WM_KEYDOWN:
        if (wp == 'P') g_paused = !g_paused;
        if (wp == 'N') g_night_scene = !g_night_scene;
        if (wp == 'F') g_dof_enabled = !g_dof_enabled;
        if (wp == 'R') {
            g_cam_pos = Vec3(0.0f, 2.4f, 6.5f);
            g_cam_yaw = 0.0f;
            g_cam_pitch = -0.08f;
        }
        if (wp == VK_OEM_4) g_focal_dist = std::max(2.0f, g_focal_dist - 0.5f);
        if (wp == VK_OEM_6) g_focal_dist += 0.5f;
        if (wp == VK_OEM_PLUS) g_aperture = std::min(0.12f, g_aperture + 0.005f);
        if (wp == VK_OEM_MINUS) g_aperture = std::max(0.0f, g_aperture - 0.005f);
        if (wp == VK_ESCAPE) g_running = false;
        return 0;
    }
    return DefWindowProc(hwnd, msg, wp, lp);
}

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int nCmdShow) {
    WNDCLASSEX wc{ sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, hInst, nullptr, nullptr, nullptr, nullptr, "VulkanRT", nullptr };
    RegisterClassEx(&wc);
    g_hwnd = CreateWindow(wc.lpszClassName, "First-Person Photoreal Vulkan Ray Tracer", WS_OVERLAPPEDWINDOW, 100, 100, g_width, g_height, nullptr, nullptr, hInst, nullptr);

    ShowWindow(g_hwnd, nCmdShow);
    UpdateWindow(g_hwnd);

    init_vulkan();

    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastTitleUpdate = startTime;
    float accumulated_time = 0.0f;
    uint32_t fps_counter = 0;

    MSG msg{};
    while (g_running) {
        while (PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) g_running = false;
        }
        if (!g_running) break;

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> delta = now - startTime;
        startTime = now;
        float dt = delta.count();

        if (!g_paused) accumulated_time += dt;

        render_frame(accumulated_time, dt);
        fps_counter++;

        std::chrono::duration<float> titleElapsed = now - lastTitleUpdate;
        if (titleElapsed.count() >= 0.5f) {
            float fps = fps_counter / titleElapsed.count();
            float ms = (1000.0f / (fps > 0 ? fps : 1.0f));
            float cull_pct = (g_active_subchunks_count + g_culled_subchunks_count > 0)
                ? (100.0f * (float)g_culled_subchunks_count / (float)(g_active_subchunks_count + g_culled_subchunks_count))
                : 0.0f;
            char title[256];
            sprintf_s(title, "First-Person RT | FPS: %.1f (%.2f ms) | Sub-chunks: %d active, %d culled (%.0f%%) | LoD: [H:%d, M:%d, L:%d] | Pos: (%.1f, %.1f, %.1f) | [WASD, Shift, F: DoF, N: Night]",
                fps, ms, g_active_subchunks_count, g_culled_subchunks_count, cull_pct,
                g_lod_counts[0], g_lod_counts[1], g_lod_counts[2],
                g_cam_pos.x, g_cam_pos.y, g_cam_pos.z);
            SetWindowText(g_hwnd, title);
            fps_counter = 0;
            lastTitleUpdate = now;
        }
    }

    cleanup_vulkan();
    UnregisterClass(wc.lpszClassName, hInst);
    return 0;
}