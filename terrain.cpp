#include "terrain.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace {

    struct Vec3f {
        float x, y, z;
        Vec3f(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
        Vec3f operator-(const Vec3f& b) const { return Vec3f(x - b.x, y - b.y, z - b.z); }
        Vec3f operator+(const Vec3f& b) const { return Vec3f(x + b.x, y + b.y, z + b.z); }
        Vec3f operator*(float s) const { return Vec3f(x * s, y * s, z * s); }
    };

    inline Vec3f normalize_v(const Vec3f& v) {
        float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        return len > 1e-6f ? Vec3f(v.x / len, v.y / len, v.z / len) : Vec3f(0, 1, 0);
    }

    std::string find_file(const std::string& pattern) {
        std::ifstream direct(pattern, std::ios::binary);
        if (direct.is_open()) return pattern;

#if defined(_WIN32)
        char exe_buf[MAX_PATH];
        if (GetModuleFileNameA(NULL, exe_buf, MAX_PATH) != 0) {
            std::string exe_str(exe_buf);
            size_t last_slash = exe_str.find_last_of("\\/");
            if (last_slash != std::string::npos) {
                std::string p = exe_str.substr(0, last_slash + 1) + pattern;
                std::ifstream f2(p, std::ios::binary);
                if (f2.is_open()) return p;
            }
        }
#endif
        return pattern;
    }

    bool read_geotiff(const std::string& filepath, uint32_t& width, uint32_t& height,
        uint16_t& bits_per_sample, std::vector<uint8_t>& raw_bytes) {
        std::string path = find_file(filepath);
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return false;

        char header[8];
        file.read(header, 8);
        if (file.gcount() < 8) return false;

        bool le = (header[0] == 'I' && header[1] == 'I');
        if (!le && !(header[0] == 'M' && header[1] == 'M')) return false;

        auto r_u16 = [le](const uint8_t* p) -> uint16_t {
            return le ? (p[0] | (p[1] << 8)) : ((p[0] << 8) | p[1]);
            };
        auto r_u32 = [le](const uint8_t* p) -> uint32_t {
            return le ? (p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24))
                : ((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]);
            };

        uint16_t version = r_u16((uint8_t*)&header[2]);
        if (version != 42) return false;

        uint32_t ifd_offset = r_u32((uint8_t*)&header[4]);
        file.seekg(ifd_offset);

        uint8_t n_buf[2];
        file.read((char*)n_buf, 2);
        uint16_t num_entries = r_u16(n_buf);

        width = 0; height = 0; bits_per_sample = 16;
        uint32_t rows_per_strip = 0;
        std::vector<uint32_t> strip_offsets;
        std::vector<uint32_t> strip_byte_counts;

        for (uint16_t i = 0; i < num_entries; ++i) {
            uint8_t entry[12];
            file.read((char*)entry, 12);
            uint16_t tag = r_u16(&entry[0]);
            uint16_t type = r_u16(&entry[2]);
            uint32_t count = r_u32(&entry[4]);

            auto read_val = [&](uint32_t& out_val) {
                if (type == 3) out_val = le ? (entry[8] | (entry[9] << 8)) : ((entry[8] << 8) | entry[9]);
                else out_val = r_u32(&entry[8]);
                };

            if (tag == 256) read_val(width);
            else if (tag == 257) read_val(height);
            else if (tag == 258) { uint32_t b = 16; read_val(b); bits_per_sample = (uint16_t)b; }
            else if (tag == 278) read_val(rows_per_strip);
            else if (tag == 273) {
                if (count == 1) {
                    uint32_t off = 0; read_val(off); strip_offsets.push_back(off);
                }
                else {
                    uint32_t off_pos = r_u32(&entry[8]);
                    std::streampos cur = file.tellg();
                    file.seekg(off_pos);
                    strip_offsets.resize(count);
                    for (uint32_t k = 0; k < count; ++k) {
                        uint8_t ob[4]; file.read((char*)ob, type == 3 ? 2 : 4);
                        strip_offsets[k] = (type == 3) ? r_u16(ob) : r_u32(ob);
                    }
                    file.seekg(cur);
                }
            }
            else if (tag == 279) {
                if (count == 1) {
                    uint32_t bc = 0; read_val(bc); strip_byte_counts.push_back(bc);
                }
                else {
                    uint32_t bc_pos = r_u32(&entry[8]);
                    std::streampos cur = file.tellg();
                    file.seekg(bc_pos);
                    strip_byte_counts.resize(count);
                    for (uint32_t k = 0; k < count; ++k) {
                        uint8_t bb[4]; file.read((char*)bb, type == 3 ? 2 : 4);
                        strip_byte_counts[k] = (type == 3) ? r_u16(bb) : r_u32(bb);
                    }
                    file.seekg(cur);
                }
            }
        }

        if (width == 0 || height == 0 || strip_offsets.empty()) return false;
        if (rows_per_strip == 0) rows_per_strip = height;

        size_t bytes_per_pixel = bits_per_sample / 8;
        raw_bytes.resize(width * height * bytes_per_pixel);

        uint32_t rows_read = 0;
        for (size_t s = 0; s < strip_offsets.size(); ++s) {
            file.seekg(strip_offsets[s]);
            uint32_t cur_rows = (s + 1 == strip_offsets.size()) ? (height - rows_read) : rows_per_strip;
            uint32_t bytes_to_read = cur_rows * width * (uint32_t)bytes_per_pixel;
            if ((rows_read * width + cur_rows * width) * bytes_per_pixel > raw_bytes.size()) break;
            file.read((char*)(raw_bytes.data() + rows_read * width * bytes_per_pixel), bytes_to_read);
            rows_read += cur_rows;
        }
        return true;
    }

    // High-performance procedural noise & ridged multifractal terrain synthesis
    inline float hash2d(float x, float z) {
        float fx = std::sin(x * 127.1f + z * 311.7f) * 43758.5453f;
        return fx - std::floor(fx);
    }

    inline float grad_noise(float x, float z) {
        float ix = std::floor(x);
        float iz = std::floor(z);
        float fx = x - ix;
        float fz = z - iz;

        float ux = fx * fx * (3.0f - 2.0f * fx);
        float uz = fz * fz * (3.0f - 2.0f * fz);

        float a = hash2d(ix, iz);
        float b = hash2d(ix + 1.0f, iz);
        float c = hash2d(ix, iz + 1.0f);
        float d = hash2d(ix + 1.0f, iz + 1.0f);

        return a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz;
    }

    float sample_procedural_elevation(float wx, float wz) {
        float d = std::sqrt(wx * wx + wz * wz);

        // 1. Domain warping: simulates tectonic folding and water flow twisting
        float qx = grad_noise(wx * 0.00028f + 1.2f, wz * 0.00028f + 4.3f);
        float qz = grad_noise(wx * 0.00028f + 8.1f, wz * 0.00028f + 2.7f);

        float rx = grad_noise((wx + qx * 900.0f) * 0.00045f, (wz + qz * 900.0f) * 0.00045f);
        float rz = grad_noise((wx + qx * 900.0f) * 0.00045f + 5.2f, (wz + qz * 900.0f) * 0.00045f + 1.3f);

        float warped_x = wx + rx * 650.0f;
        float warped_z = wz + rz * 650.0f;

        // 2. Continental massif scale
        float base_massif = (std::sin(warped_x * 0.00018f) * std::cos(warped_z * 0.00016f)) * 750.0f;

        // 3. Sharp alpine arêtes and ridged glacial peaks (1.0 - abs(noise))
        float n1 = grad_noise(warped_x * 0.00042f, warped_z * 0.00042f);
        float ridge1 = 1.0f - std::abs(n1 * 2.0f - 1.0f);
        ridge1 = ridge1 * ridge1 * 880.0f;

        float n2 = grad_noise(warped_x * 0.00095f + 3.1f, warped_z * 0.00095f + 7.4f);
        float ridge2 = 1.0f - std::abs(n2 * 2.0f - 1.0f);
        ridge2 = std::pow(ridge2, 2.5f) * 440.0f;

        // 4. Dendritic erosion gullies (sharp V-shaped incision)
        float gully_noise = grad_noise(warped_x * 0.0018f + 9.2f, warped_z * 0.0018f + 1.6f);
        float gully = (1.0f - std::pow(std::abs(gully_noise * 2.0f - 1.0f), 0.7f)) * -140.0f;

        // 5. Rolling foothills & plateau modulation
        float foothills = (std::sin(wx * 0.0042f + 0.5f) * std::cos(wz * 0.0038f + 1.1f)) * 45.0f;
        float alpine_total = base_massif + ridge1 + ridge2 + gully - 380.0f;

        float w_alpine = std::clamp((d - 120.0f) / 600.0f, 0.0f, 1.0f);
        w_alpine = w_alpine * w_alpine * (3.0f - 2.0f * w_alpine);

        return foothills + w_alpine * alpine_total;
    }

} // namespace

bool load_aw3d30_terrain_chunks(const std::string& directory,
    const TerrainSettings& settings,
    std::vector<TerrainChunk>& out_chunks) {
    (void)directory;

    uint32_t dsm_w = 0, dsm_h = 0;
    uint16_t dsm_bps = 0;
    std::vector<uint8_t> dsm_bytes;
    bool dsm_loaded = read_geotiff("ALPSMLC30_N051E095_DSM.tif", dsm_w, dsm_h, dsm_bps, dsm_bytes);

    uint32_t msk_w = 0, msk_h = 0;
    uint16_t msk_bps = 0;
    std::vector<uint8_t> msk_bytes;
    bool msk_loaded = read_geotiff("ALPSMLC30_N051E095_MSK.tif", msk_w, msk_h, msk_bps, msk_bytes);

    const int16_t* dsm_data = dsm_loaded ? reinterpret_cast<const int16_t*>(dsm_bytes.data()) : nullptr;
    const uint8_t* msk_data = msk_loaded ? msk_bytes.data() : nullptr;

    int peak_px = 1800, peak_py = 1800;
    float peak_elev = 2600.0f;
    float dx_m = settings.tile_width_m / 3599.0f;
    float dz_m = settings.tile_height_m / 3599.0f;

    if (dsm_loaded && dsm_bytes.size() >= dsm_w * dsm_h * 2) {
        int16_t max_h = -32768;
        for (uint32_t y = 200; y < dsm_h - 200; y += 8) {
            for (uint32_t x = 200; x < dsm_w - 200; x += 8) {
                int16_t val = dsm_data[y * dsm_w + x];
                if (val > max_h && val < 8848) {
                    max_h = val;
                    peak_px = x;
                    peak_py = y;
                }
            }
        }
        peak_elev = float(max_h);
    }

    auto sample_dsm_bilinear = [&](float px, float py) -> float {
        px = std::clamp(px, 0.0f, float(dsm_w - 1));
        py = std::clamp(py, 0.0f, float(dsm_h - 1));
        int x0 = (int)std::floor(px);
        int y0 = (int)std::floor(py);
        int x1 = std::min(x0 + 1, (int)dsm_w - 1);
        int y1 = std::min(y0 + 1, (int)dsm_h - 1);
        float fx = px - float(x0);
        float fy = py - float(y0);

        auto read_elev = [&](int x, int y) -> float {
            int16_t v = dsm_data[y * dsm_w + x];
            return (v > -500 && v < 8848) ? float(v) : peak_elev - 1200.0f;
            };

        float v00 = read_elev(x0, y0);
        float v10 = read_elev(x1, y0);
        float v01 = read_elev(x0, y1);
        float v11 = read_elev(x1, y1);

        return (1.0f - fx) * ((1.0f - fy) * v00 + fy * v01) + fx * ((1.0f - fy) * v10 + fy * v11);
        };

    const float center_x = 0.0f;
    const float center_z = -3.5f;
    float base_center_elev = sample_procedural_elevation(center_x, center_z);

    auto evaluate_world_point = [&](float wx, float wz, float& out_elev, float& out_water) {
        float raw_elev = 0.0f;
        if (dsm_data) {
            float px = float(peak_px) + wx / dx_m;
            float py = float(peak_py) + wz / dz_m;
            raw_elev = sample_dsm_bilinear(px, py) - peak_elev;
            int ix = std::clamp((int)std::round(px), 0, (int)msk_w - 1);
            int iy = std::clamp((int)std::round(py), 0, (int)msk_h - 1);
            out_water = (msk_data && msk_data[iy * msk_w + ix] == 2) ? 1.0f : 0.0f;
        }
        else {
            raw_elev = sample_procedural_elevation(wx, wz) - base_center_elev;
            out_water = 0.0f;
        }

        float d_terrace = std::sqrt((wx - center_x) * (wx - center_x) + (wz - center_z) * (wz - center_z));
        const float terrace_r = settings.terrace_radius;
        const float blend_r = settings.blend_radius;

        if (d_terrace <= terrace_r) {
            out_elev = 0.0f;
        }
        else if (d_terrace < blend_r) {
            float t = (d_terrace - terrace_r) / (blend_r - terrace_r);
            float smooth_t = t * t * (3.0f - 2.0f * t);
            out_elev = raw_elev * smooth_t;
        }
        else {
            out_elev = raw_elev;
        }
        };

    const int CHUNKS_X = settings.chunks_x;
    const int CHUNKS_Z = settings.chunks_z;
    const int SUB_PER_CHUNK = settings.subchunks_per_chunk;

    const float R_NEAR = 32.0f;
    const float R_FAR = 22000.0f;

    auto uv_to_world = [&](float u, float v, float& wx, float& wz) {
        float au = std::abs(u);
        float av = std::abs(v);
        wx = center_x + std::copysign(au * R_NEAR + std::pow(au, 2.2f) * R_FAR, u);
        wz = center_z + std::copysign(av * R_NEAR + std::pow(av, 2.2f) * R_FAR, v);
        };

    const int SUB_VERTS = 33;

    out_chunks.clear();
    out_chunks.resize(CHUNKS_X * CHUNKS_Z);

    size_t total_subchunks = 0;
    size_t total_tris_lod0 = 0;

    for (int cz = 0; cz < CHUNKS_Z; ++cz) {
        float chunk_v0 = -1.0f + 2.0f * (float(cz) / float(CHUNKS_Z));
        float chunk_v1 = -1.0f + 2.0f * (float(cz + 1) / float(CHUNKS_Z));

        for (int cx = 0; cx < CHUNKS_X; ++cx) {
            float chunk_u0 = -1.0f + 2.0f * (float(cx) / float(CHUNKS_X));
            float chunk_u1 = -1.0f + 2.0f * (float(cx + 1) / float(CHUNKS_X));

            int chunk_idx = cz * CHUNKS_X + cx;
            auto& chunk = out_chunks[chunk_idx];
            chunk.chunk_x = cx;
            chunk.chunk_z = cz;
            chunk.sub_chunks.resize(SUB_PER_CHUNK * SUB_PER_CHUNK);

            for (int sz = 0; sz < SUB_PER_CHUNK; ++sz) {
                float v0 = chunk_v0 + (chunk_v1 - chunk_v0) * (float(sz) / float(SUB_PER_CHUNK));
                float v1 = chunk_v0 + (chunk_v1 - chunk_v0) * (float(sz + 1) / float(SUB_PER_CHUNK));

                for (int sx = 0; sx < SUB_PER_CHUNK; ++sx) {
                    float u0 = chunk_u0 + (chunk_u1 - chunk_u0) * (float(sx) / float(SUB_PER_CHUNK));
                    float u1 = chunk_u0 + (chunk_u1 - chunk_u0) * (float(sx + 1) / float(SUB_PER_CHUNK));

                    int sub_idx = sz * SUB_PER_CHUNK + sx;
                    auto& sub = chunk.sub_chunks[sub_idx];
                    sub.parent_chunk_x = cx;
                    sub.parent_chunk_z = cz;
                    sub.sub_x = sx;
                    sub.sub_z = sz;

                    std::vector<Vec3f> s_pos(SUB_VERTS * SUB_VERTS);
                    std::vector<Vec3f> s_nrm(SUB_VERTS * SUB_VERTS);
                    std::vector<float> s_wtr(SUB_VERTS * SUB_VERTS);

                    float du = (u1 - u0) / float(SUB_VERTS - 1);
                    float dv = (v1 - v0) / float(SUB_VERTS - 1);

                    for (int j = 0; j < SUB_VERTS; ++j) {
                        float v = v0 + (v1 - v0) * (float(j) / float(SUB_VERTS - 1));
                        for (int i = 0; i < SUB_VERTS; ++i) {
                            float u = u0 + (u1 - u0) * (float(i) / float(SUB_VERTS - 1));

                            float wx, wz, wy, wtr;
                            uv_to_world(u, v, wx, wz);
                            evaluate_world_point(wx, wz, wy, wtr);

                            int v_idx = j * SUB_VERTS + i;
                            s_pos[v_idx] = Vec3f(wx, wy, wz);
                            s_wtr[v_idx] = wtr;

                            float wx_p, wz_p, wx_m, wz_m;
                            uv_to_world(u + du, v, wx_p, wz_p);
                            uv_to_world(u - du, v, wx_m, wz_m);
                            float eps_x = std::max(6.0f, std::abs(wx_p - wx_m) * 0.5f);
                            uv_to_world(u, v + dv, wx_p, wz_p);
                            uv_to_world(u, v - dv, wx_m, wz_m);
                            float eps_z = std::max(6.0f, std::abs(wz_p - wz_m) * 0.5f);

                            float h_px, h_nx, h_pz, h_nz, dum;
                            evaluate_world_point(wx + eps_x, wz, h_px, dum);
                            evaluate_world_point(wx - eps_x, wz, h_nx, dum);
                            evaluate_world_point(wx, wz + eps_z, h_pz, dum);
                            evaluate_world_point(wx, wz - eps_z, h_nz, dum);

                            Vec3f n = normalize_v(Vec3f((h_nx - h_px) / (2.0f * eps_x), 1.0f, (h_nz - h_pz) / (2.0f * eps_z)));
                            if (n.y < 0.0f) n = n * -1.0f;
                            s_nrm[v_idx] = n;

                            sub.aabb_min[0] = std::min(sub.aabb_min[0], wx);
                            sub.aabb_min[1] = std::min(sub.aabb_min[1], wy);
                            sub.aabb_min[2] = std::min(sub.aabb_min[2], wz);
                            sub.aabb_max[0] = std::max(sub.aabb_max[0], wx);
                            sub.aabb_max[1] = std::max(sub.aabb_max[1], wy);
                            sub.aabb_max[2] = std::max(sub.aabb_max[2], wz);
                        }
                    }

                    sub.centroid[0] = (sub.aabb_min[0] + sub.aabb_max[0]) * 0.5f;
                    sub.centroid[1] = (sub.aabb_min[1] + sub.aabb_max[1]) * 0.5f;
                    sub.centroid[2] = (sub.aabb_min[2] + sub.aabb_max[2]) * 0.5f;

                    const int lod_steps[3] = { 1, 2, 4 };
                    const float SKIRT_DEPTH = 45.0f;

                    for (int lod = 0; lod < 3; ++lod) {
                        int step = lod_steps[lod];
                        int quads = (SUB_VERTS - 1) / step;
                        sub.lods[lod].triangles.reserve(quads * quads * 2 + quads * 8);

                        for (int j = 0; j < SUB_VERTS - 1; j += step) {
                            for (int i = 0; i < SUB_VERTS - 1; i += step) {
                                uint32_t i00 = j * SUB_VERTS + i;
                                uint32_t i10 = j * SUB_VERTS + (i + step);
                                uint32_t i01 = (j + step) * SUB_VERTS + i;
                                uint32_t i11 = (j + step) * SUB_VERTS + (i + step);

                                auto emit_tri = [&](uint32_t a, uint32_t b, uint32_t c) {
                                    TerrainTriangle tri{};
                                    tri.v0[0] = s_pos[a].x; tri.v0[1] = s_pos[a].y; tri.v0[2] = s_pos[a].z; tri.v0[3] = 4.0f;
                                    tri.v1[0] = s_pos[b].x; tri.v1[1] = s_pos[b].y; tri.v1[2] = s_pos[b].z; tri.v1[3] = 0.0f;
                                    tri.v2[0] = s_pos[c].x; tri.v2[1] = s_pos[c].y; tri.v2[2] = s_pos[c].z; tri.v2[3] = 0.0f;

                                    tri.n0[0] = s_nrm[a].x; tri.n0[1] = s_nrm[a].y; tri.n0[2] = s_nrm[a].z; tri.n0[3] = s_wtr[a];
                                    tri.n1[0] = s_nrm[b].x; tri.n1[1] = s_nrm[b].y; tri.n1[2] = s_nrm[b].z; tri.n1[3] = 0.0f;
                                    tri.n2[0] = s_nrm[c].x; tri.n2[1] = s_nrm[c].y; tri.n2[2] = s_nrm[c].z; tri.n2[3] = 0.0f;
                                    sub.lods[lod].triangles.push_back(tri);
                                    };

                                emit_tri(i00, i01, i10);
                                emit_tri(i10, i01, i11);
                            }
                        }

                        auto emit_skirt_segment = [&](uint32_t a, uint32_t b) {
                            Vec3f p0 = s_pos[a];
                            Vec3f p1 = s_pos[b];
                            Vec3f p0_down = Vec3f(p0.x, p0.y - SKIRT_DEPTH, p0.z);
                            Vec3f p1_down = Vec3f(p1.x, p1.y - SKIRT_DEPTH, p1.z);
                            Vec3f n0 = s_nrm[a];
                            Vec3f n1 = s_nrm[b];

                            TerrainTriangle t1{};
                            t1.v0[0] = p0.x; t1.v0[1] = p0.y; t1.v0[2] = p0.z; t1.v0[3] = 4.0f;
                            t1.v1[0] = p0_down.x; t1.v1[1] = p0_down.y; t1.v1[2] = p0_down.z; t1.v1[3] = 0.0f;
                            t1.v2[0] = p1.x; t1.v2[1] = p1.y; t1.v2[2] = p1.z; t1.v2[3] = 0.0f;
                            t1.n0[0] = n0.x; t1.n0[1] = n0.y; t1.n0[2] = n0.z; t1.n0[3] = s_wtr[a];
                            t1.n1[0] = n0.x; t1.n1[1] = n0.y; t1.n1[2] = n0.z; t1.n1[3] = 0.0f;
                            t1.n2[0] = n1.x; t1.n2[1] = n1.y; t1.n2[2] = n1.z; t1.n2[3] = 0.0f;
                            sub.lods[lod].triangles.push_back(t1);

                            TerrainTriangle t2{};
                            t2.v0[0] = p1.x; t2.v0[1] = p1.y; t2.v0[2] = p1.z; t2.v0[3] = 4.0f;
                            t2.v1[0] = p0_down.x; t2.v1[1] = p0_down.y; t2.v1[2] = p0_down.z; t2.v1[3] = 0.0f;
                            t2.v2[0] = p1_down.x; t2.v2[1] = p1_down.y; t2.v2[2] = p1_down.z; t2.v2[3] = 0.0f;
                            t2.n0[0] = n1.x; t2.n0[1] = n1.y; t2.n0[2] = n1.z; t2.n0[3] = s_wtr[b];
                            t2.n1[0] = n0.x; t2.n1[1] = n0.y; t2.n1[2] = n0.z; t2.n1[3] = 0.0f;
                            t2.n2[0] = n1.x; t2.n2[1] = n1.y; t2.n2[2] = n1.z; t2.n2[3] = 0.0f;
                            sub.lods[lod].triangles.push_back(t2);
                            };

                        for (int i = 0; i < SUB_VERTS - 1; i += step) {
                            emit_skirt_segment(0 * SUB_VERTS + i + step, 0 * SUB_VERTS + i);
                            emit_skirt_segment((SUB_VERTS - 1) * SUB_VERTS + i, (SUB_VERTS - 1) * SUB_VERTS + i + step);
                        }
                        for (int j = 0; j < SUB_VERTS - 1; j += step) {
                            emit_skirt_segment((j)*SUB_VERTS + 0, (j + step) * SUB_VERTS + 0);
                            emit_skirt_segment((j + step) * SUB_VERTS + (SUB_VERTS - 1), (j)*SUB_VERTS + (SUB_VERTS - 1));
                        }
                    }

                    sub.aabb_min[1] -= SKIRT_DEPTH;

                    chunk.aabb_min[0] = std::min(chunk.aabb_min[0], sub.aabb_min[0]);
                    chunk.aabb_min[1] = std::min(chunk.aabb_min[1], sub.aabb_min[1]);
                    chunk.aabb_min[2] = std::min(chunk.aabb_min[2], sub.aabb_min[2]);
                    chunk.aabb_max[0] = std::max(chunk.aabb_max[0], sub.aabb_max[0]);
                    chunk.aabb_max[1] = std::max(chunk.aabb_max[1], sub.aabb_max[1]);
                    chunk.aabb_max[2] = std::max(chunk.aabb_max[2], sub.aabb_max[2]);

                    total_subchunks++;
                    total_tris_lod0 += sub.lods[0].triangles.size();
                }
            }

            chunk.centroid[0] = (chunk.aabb_min[0] + chunk.aabb_max[0]) * 0.5f;
            chunk.centroid[1] = (chunk.aabb_min[1] + chunk.aabb_max[1]) * 0.5f;
            chunk.centroid[2] = (chunk.aabb_min[2] + chunk.aabb_max[2]) * 0.5f;
        }
    }

    std::cout << "[AW3D30] Initialized " << total_subchunks << " sub-chunks with perimeter skirts ("
        << total_tris_lod0 << " LoD 0 triangles).\n";
    return true;
}

bool load_aw3d30_terrain(const std::string& directory,
    const TerrainSettings& settings,
    std::vector<TerrainTriangle>& out_triangles) {
    std::vector<TerrainChunk> chunks;
    if (!load_aw3d30_terrain_chunks(directory, settings, chunks)) return false;
    out_triangles.clear();
    for (const auto& c : chunks) {
        for (const auto& sc : c.sub_chunks) {
            out_triangles.insert(out_triangles.end(), sc.lods[0].triangles.begin(), sc.lods[0].triangles.end());
        }
    }
    return true;
}