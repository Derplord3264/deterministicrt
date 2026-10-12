#pragma once

#include <vector>
#include <string>
#include <cstdint>

struct TerrainTriangle {
    float v0[4]; // x, y, z, mat_id (4.0 = terrain)
    float v1[4];
    float v2[4];
    float n0[4]; // nx, ny, nz, feature_flag (1.0 = water)
    float n1[4];
    float n2[4];
};

struct SubChunkLOD {
    std::vector<TerrainTriangle> triangles;
};

struct TerrainSubChunk {
    int parent_chunk_x = 0;
    int parent_chunk_z = 0;
    int sub_x = 0;
    int sub_z = 0;
    float aabb_min[3]{ 1e30f, 1e30f, 1e30f };
    float aabb_max[3]{ -1e30f, -1e30f, -1e30f };
    float centroid[3]{ 0.0f, 0.0f, 0.0f };
    SubChunkLOD lods[3]; // 0: High (2304 tris), 1: Med (640 tris), 2: Low (192 tris)
};

struct TerrainChunk {
    int chunk_x = 0;
    int chunk_z = 0;
    float aabb_min[3]{ 1e30f, 1e30f, 1e30f };
    float aabb_max[3]{ -1e30f, -1e30f, -1e30f };
    float centroid[3]{ 0.0f, 0.0f, 0.0f };
    std::vector<TerrainSubChunk> sub_chunks;
};

struct TerrainSettings {
    float tile_width_m = 69240.0f;     // ~69.2 km (1 degree longitude)
    float tile_height_m = 111195.0f;   // ~111.2 km (1 degree latitude)

    int chunks_x = 8;
    int chunks_z = 8;
    int subchunks_per_chunk = 2;       // 2x2 = 4 subchunks per chunk (256 subchunks total)

    float terrace_radius = 5.5f;       // Terrace around origin
    float blend_radius = 20.0f;        // Smooth transition to mountains
};

bool load_aw3d30_terrain_chunks(const std::string& directory,
    const TerrainSettings& settings,
    std::vector<TerrainChunk>& out_chunks);

bool load_aw3d30_terrain(const std::string& directory,
    const TerrainSettings& settings,
    std::vector<TerrainTriangle>& out_triangles);