#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    vec4 origin;
    vec4 horizontal;
    vec4 vertical;
    vec4 lower_left_corner;
    vec4 cam_dir;
    vec2 resolution;
    float time;
    float exposure;
    float aperture;
    float focal_dist;
    int night_scene;
    int dof_enabled;
    int tlas_count;
    int tlas_offset;
} push;

const float PI = 3.1415926535897932385;
const float TWO_PI = 6.2831853071795864769;
const float GOLDEN_ANGLE = 2.399963229728653;
const float INF = 1e30;

// SYNCHRONIZED EXACTLY WITH main.cpp
const vec3 SUN_DIR = normalize(vec3(0.42, 0.58, -0.70));
const vec3 SUN_COLOR = vec3(3.4, 3.2, 2.9);
const float SUN_ANGULAR_RADIUS = 0.032;
// Calibrated soft penumbra spread matching reference render
const float FOREGROUND_SHADOW_SPREAD = 0.095;

#define SHAPE_SPHERE 0
#define SHAPE_BOX    1
#define SHAPE_MESH   2

#define MAT_CHECKER  0
#define MAT_DIFFUSE  1
#define MAT_METAL    2
#define MAT_GLASS    3
#define MAT_TERRAIN  4

struct SceneObject {
    int shape;
    int mat_type;
    vec3 pos;
    vec3 extents;
    vec3 albedo;
    vec3 color2;
    float ior;
    float dispersion;
};

#define NUM_OBJECTS 5
SceneObject g_scene[NUM_OBJECTS];

struct BVHNode {
    vec4 aabb_min;
    vec4 aabb_max;
};

struct TLASNode {
    vec4 aabb_min;
    vec4 aabb_max;
};

struct GPU_Triangle {
    vec4 v0;
    vec4 v1;
    vec4 v2;
    vec4 n0;
    vec4 n1;
    vec4 n2;
};

layout(std430, set = 0, binding = 0) readonly buffer BVHBuffer {
    BVHNode bvh_nodes[];
};

layout(std430, set = 0, binding = 1) readonly buffer TriangleBuffer {
    GPU_Triangle triangles[];
};

layout(std430, set = 0, binding = 2) readonly buffer CausticBuffer {
    vec4 caustic_map[];
};

layout(std430, set = 0, binding = 3) readonly buffer TLASBuffer {
    TLASNode tlas_nodes[];
};

// -----------------------------------------------------------------------------------------
// Smooth Gradient Noise & Voronoi Cellular Library
// -----------------------------------------------------------------------------------------
float remap(float v, float inMin, float inMax, float outMin, float outMax) {
    return outMin + (((v - inMin) / max(inMax - inMin, 0.00001)) * (outMax - outMin));
}

float remapClamped(float v, float inMin, float inMax, float outMin, float outMax) {
    return clamp(remap(v, inMin, inMax, outMin, outMax), min(outMin, outMax), max(outMin, outMax));
}

float IGN(vec2 p) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(p, magic.xy)));
}

vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return -1.0 + 2.0 * fract((p3.xx + p3.yz) * p3.zy);
}

float gradNoise2D(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    vec2 ga = hash22(i + vec2(0.0, 0.0));
    vec2 gb = hash22(i + vec2(1.0, 0.0));
    vec2 gc = hash22(i + vec2(0.0, 1.0));
    vec2 gd = hash22(i + vec2(1.0, 1.0));

    float va = dot(ga, f - vec2(0.0, 0.0));
    float vb = dot(gb, f - vec2(1.0, 0.0));
    float vc = dot(gc, f - vec2(0.0, 1.0));
    float vd = dot(gd, f - vec2(1.0, 1.0));

    return 0.5 + 0.5 * mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y);
}

float voronoi2D(vec2 p, out float id) {
    vec2 n = floor(p);
    vec2 f = fract(p);
    float md = 8.0;
    float cell_id = 0.0;

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = hash22(n + g) * 0.45 + 0.5;
            vec2 r = g + o - f;
            float d = dot(r, r);
            if (d < md) {
                md = d;
                cell_id = fract(sin(dot(n + g, vec2(12.9898, 78.233))) * 43758.5453);
            }
        }
    }
    id = cell_id;
    return sqrt(md);
}

const mat2 NOISE_ROT = mat2(0.80, 0.60, -0.60, 0.80);

float fbm2D(vec2 p) {
    float f = 0.500 * gradNoise2D(p); p = NOISE_ROT * p * 2.02;
    f += 0.250 * gradNoise2D(p); p = NOISE_ROT * p * 2.03;
    f += 0.125 * gradNoise2D(p);
    return f;
}

float fbm2D_lod(vec2 p, float dist, float f0, int max_octaves) {
    float theta_pixel = 1.041 / push.resolution.y; 
    float pixel_footprint = max(dist * theta_pixel, 0.0001);
    float max_k = log2(1.0 / max(2.0 * f0 * pixel_footprint, 0.00001));

    float sum = 0.0;
    float amp = 0.5;
    vec2 cur_p = p;

    for (int i = 0; i < 3; ++i) {
        if (i >= max_octaves) break;
        float fade = (i == 0) ? 1.0 : clamp(max_k - float(i) + 1.0, 0.0, 1.0);

        if (fade <= 0.0) {
            sum += amp * 0.5;
        } else {
            float n = gradNoise2D(cur_p);
            sum += amp * mix(0.5, n, fade);
        }
        cur_p = NOISE_ROT * cur_p * 2.025;
        amp *= 0.5;
    }
    return sum;
}

float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float noise3D(vec3 x) {
    vec3 p = floor(x);
    vec3 u = x - p;
    u = u * u * (3.0 - 2.0 * u);

    float n000 = hash31(p + vec3(0.0, 0.0, 0.0));
    float n100 = hash31(p + vec3(1.0, 0.0, 0.0));
    float n010 = hash31(p + vec3(0.0, 1.0, 0.0));
    float n110 = hash31(p + vec3(1.0, 1.0, 0.0));
    float n001 = hash31(p + vec3(0.0, 0.0, 1.0));
    float n101 = hash31(p + vec3(1.0, 0.0, 1.0));
    float n011 = hash31(p + vec3(0.0, 1.0, 1.0));
    float n111 = hash31(p + vec3(1.0, 1.0, 1.0));

    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
               mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

float billow3D(vec3 p) {
    return 1.0 - abs(noise3D(p) * 2.0 - 1.0);
}

float getBaseCloudNoise(vec3 p) {
    float w0 = billow3D(p);
    float w1 = billow3D(p * 2.02);
    float w2 = billow3D(p * 4.05);
    float worley = w0 * 0.625 + w1 * 0.250 + w2 * 0.125;

    float p0 = noise3D(p);
    float p1 = noise3D(p * 2.02);
    float p2 = noise3D(p * 4.05);
    float perlin = p0 * 0.625 + p1 * 0.250 + p2 * 0.125;

    return remapClamped(perlin, 0.0, 1.0, worley * 0.88, 1.0);
}

float getDetailNoise(vec3 p) {
    float d0 = billow3D(p * 2.0);
    float d1 = billow3D(p * 4.0);
    return d0 * 0.65 + d1 * 0.35;
}

// -----------------------------------------------------------------------------------------
// Atmospheric Sky & Volumetric Clouds
// -----------------------------------------------------------------------------------------
float HGPhase(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 0.0001), 1.5));
}

float dualHGPhase(float cosTheta) {
    float forward  = HGPhase(cosTheta, 0.82);
    float backward = HGPhase(cosTheta, -0.25);
    return mix(forward, backward, 0.20);
}

const float EARTH_RADIUS      = 350000.0;
const float CLOUD_ALT_BOTTOM  = 1200.0;
const float CLOUD_ALT_TOP     = 5200.0;

bool intersect_atmosphere(vec3 ro, vec3 rd, float r_bottom, float r_top, out float t_enter, out float t_exit) {
    vec3 center = vec3(0.0, -EARTH_RADIUS, 0.0);
    vec3 p = ro - center;

    float b = dot(p, rd);
    float c = dot(p, p);

    float r_bot = EARTH_RADIUS + r_bottom;
    float r_tp  = EARTH_RADIUS + r_top;

    float disc_top = b * b - (c - r_tp * r_tp);
    if (disc_top < 0.0) return false;
    float t_top = -b + sqrt(disc_top);
    if (t_top <= 0.0) return false;

    float disc_bot = b * b - (c - r_bot * r_bot);
    float t_bot = 0.0;
    if (disc_bot >= 0.0) {
        float tb1 = -b - sqrt(disc_bot);
        float tb2 = -b + sqrt(disc_bot);
        if (tb1 > 0.0) {
            t_bot = tb1;
        } else if (tb2 > 0.0) {
            t_bot = tb2;
        }
    }

    t_enter = max(t_bot, 0.0);
    t_exit  = t_top;
    return t_exit > t_enter;
}

struct Weather {
    float coverage;
    float cloudType;
    float heightScale;
};

Weather sampleWeather(vec2 p) {
    vec2 wp = p * 0.000032;
    float n1 = gradNoise2D(wp * 0.75 + vec2(2.1, 4.3));
    float n2 = gradNoise2D(wp * 1.60 + vec2(7.8, 1.2));
    float covRaw = n1 * 0.70 + n2 * 0.30;
    float cov = remapClamped(covRaw, 0.36, 0.76, 0.0, 1.0);
    cov = pow(cov, 1.35);

    float typeNoise = fbm2D(wp * 1.3 + vec2(5.3, 8.7));
    float cType = smoothstep(0.25, 0.75, typeNoise);

    float hNoise = fbm2D(wp * 2.1 + vec2(9.1, 3.4));
    float hScale = mix(0.35, 1.0, smoothstep(0.18, 0.82, hNoise));

    return Weather(cov, cType, hScale);
}

float getDensityHeightGradient(float h, float cloudType) {
    float bottomGrad = smoothstep(0.00, 0.08, h);
    float cumulusTop = 1.0 - smoothstep(0.30, 0.85, h);
    float congestTop = 1.0 - smoothstep(0.60, 0.98, h);
    float topGrad    = mix(cumulusTop, congestTop, cloudType);
    return bottomGrad * topGrad;
}

float getCloudShadow(vec3 p, vec3 L, vec3 wind) {
    if (L.y <= 0.01) return 1.0;
    float t = (CLOUD_ALT_BOTTOM + 200.0 - p.y) / L.y;
    if (t <= 0.0) return 1.0;
    vec3 cp = p + L * t;
    vec2 xz = cp.xz + wind.xz;
    Weather w = sampleWeather(xz);
    if (w.coverage <= 0.02) return 1.0;

    vec3 cPos = cp * 0.00035 + wind;
    float baseNoise = billow3D(cPos);
    float cloud = remapClamped(baseNoise * 0.85, 1.0 - w.coverage, 1.0, 0.0, 1.0);
    return clamp(1.0 - cloud * 0.85, 0.28, 1.0);
}

vec3 render_volumetric_sky(vec3 ro, vec3 rd, vec3 sun_dir, float time, vec2 pixel_coord, vec3 wind) {
    float zenith = clamp(rd.y, 0.0, 1.0);

    vec3 horizonCol = (push.night_scene != 0) ? vec3(0.04, 0.06, 0.12) : vec3(0.66, 0.78, 0.94);
    vec3 zenithCol  = (push.night_scene != 0) ? vec3(0.008, 0.015, 0.04) : vec3(0.10, 0.35, 0.86);
    vec3 sky = mix(horizonCol, zenithCol, pow(zenith, 0.60));

    float cosTheta = clamp(dot(rd, sun_dir), 0.0, 1.0);
    float sunTheta = acos(cosTheta);
    float sunRadius = 0.020;
    
    float sunDisk = smoothstep(sunRadius, sunRadius - 0.003, sunTheta);
    vec3 realSunColor = (push.night_scene != 0) ? vec3(4.0, 4.5, 6.0) : vec3(60.0, 54.0, 44.0);
    vec3 light_col = (push.night_scene != 0) ? vec3(0.35, 0.45, 0.65) * 1.2 : SUN_COLOR;

    vec3 sunGlow = light_col * (pow(cosTheta, 2048.0) * 8.0 + pow(cosTheta, 64.0) * 1.8 + pow(cosTheta, 12.0) * 0.6);
    sky += sunGlow;
    sky = mix(sky, realSunColor, sunDisk);

    if (rd.y > 0.01) {
        float tCirrus = (7500.0 - ro.y) / max(rd.y, 0.001);
        vec3 cp = ro + rd * tCirrus;
        vec2 cCoord = cp.xz * 0.00006 + vec2(push.time * 0.0005, push.time * 0.0002);
        float warp = gradNoise2D(cCoord * 3.0 + vec2(push.time * 0.0001, 0.0));
        vec2 shearedCoord = vec2(cCoord.x * 2.2 + cCoord.y * 1.1 + warp * 0.35, cCoord.y * 0.8 - cCoord.x * 0.4);
        float cirrusNoise = fbm2D(shearedCoord * 2.5);
        float cirrusCov = smoothstep(0.42, 0.78, cirrusNoise);
        if (cirrusCov > 0.01) {
            float cPhase = mix(HGPhase(cosTheta, 0.75), HGPhase(cosTheta, -0.2), 0.15);
            vec3 cirrusSun = light_col * (cPhase * 3.0 + 0.9);
            vec3 cirrusCol = mix(vec3(0.72, 0.84, 0.98), cirrusSun, cirrusCov * 0.8);
            float cirrusAlpha = cirrusCov * 0.38 * smoothstep(0.01, 0.10, rd.y);
            sky = mix(sky, cirrusCol, cirrusAlpha);
        }
    }

    if (rd.y > 0.055) {
        float t_enter, t_exit;
        if (intersect_atmosphere(ro, rd, CLOUD_ALT_BOTTOM, CLOUD_ALT_TOP, t_enter, t_exit)) {
            t_exit = min(t_exit, t_enter + 22000.0);
            
            const int STEPS = 24;
            float dt = (t_exit - t_enter) / float(STEPS);
            float pixelDither = IGN(pixel_coord);
            float t = t_enter + dt * pixelDither;

            float transmittance = 1.0;
            vec3 cloudColor = vec3(0.0);
            float phase = dualHGPhase(cosTheta);
            vec3 earthCenter = vec3(0.0, -EARTH_RADIUS, 0.0);

            for (int i = 0; i < STEPS; ++i) {
                vec3 P = ro + rd * t;
                float alt = length(P - earthCenter) - EARTH_RADIUS;

                Weather w = sampleWeather(P.xz + wind.xz);
                if (w.coverage > 0.01) {
                    float localBase = CLOUD_ALT_BOTTOM + (1.0 - w.coverage) * 250.0;
                    float localTop  = mix(CLOUD_ALT_BOTTOM + 800.0, CLOUD_ALT_TOP, pow(w.coverage, 0.70) * w.heightScale);

                    if (alt >= localBase && alt <= localTop) {
                        float h = (alt - localBase) / max(localTop - localBase, 50.0);
                        float hGrad = getDensityHeightGradient(h, w.cloudType);

                        if (hGrad > 0.005) {
                            vec3 cloudPos = P * 0.00035 + wind;
                            cloudPos.xz *= mix(1.10, 0.90, h);

                            float baseShape = getBaseCloudNoise(cloudPos) * hGrad;
                            float baseCloud = remapClamped(baseShape, 1.0 - w.coverage, 1.0, 0.0, 1.0);

                            if (baseCloud > 0.001) {
                                float detail = getDetailNoise(cloudPos * 3.4 - wind * 0.4);
                                float erosionMod = mix(0.38, 0.12, h);
                                float cloudDensity = remapClamped(baseCloud, detail * erosionMod, 1.0, 0.0, 1.0);
                                float d = cloudDensity * 3.8;

                                if (d > 0.001) {
                                    float lightOD = 0.0;
                                    float lStep = 80.0;
                                    vec3 lp = P;
                                    
                                    for (int j = 0; j < 3; ++j) {
                                        lp += sun_dir * lStep;
                                        float lAlt = length(lp - earthCenter) - EARTH_RADIUS;
                                        Weather lw = sampleWeather(lp.xz + wind.xz);
                                        float lBase = CLOUD_ALT_BOTTOM + (1.0 - lw.coverage) * 250.0;
                                        float lTop  = mix(CLOUD_ALT_BOTTOM + 800.0, CLOUD_ALT_TOP, pow(lw.coverage, 0.70) * lw.heightScale);
                                        if (lAlt >= lBase && lAlt <= lTop && lw.coverage > 0.02) {
                                            float lh = (lAlt - lBase) / max(lTop - lBase, 50.0);
                                            float lGrad = getDensityHeightGradient(lh, lw.cloudType);
                                            if (lGrad > 0.01) {
                                                vec3 lPos = lp * 0.00035 + wind;
                                                float lBaseNoise = getBaseCloudNoise(lPos) * lGrad;
                                                float lDens = remapClamped(lBaseNoise, 1.0 - lw.coverage, 1.0, 0.0, 1.0);
                                                lightOD += lDens * lStep;
                                            }
                                        }
                                        lStep *= 2.0;
                                    }

                                    float tau = lightOD * 0.0065;
                                    float directBeer    = exp(-tau);
                                    float forwardOct1   = exp(-tau * 0.25) * 0.50;
                                    float forwardOct2   = exp(-tau * 0.08) * 0.30;
                                    float sunVisibility = directBeer + forwardOct1 + forwardOct2;
                                    float powder        = 1.0 - exp(-d * 2.2);

                                    vec3 ambientTop    = (push.night_scene != 0) ? vec3(0.04, 0.06, 0.12) : vec3(0.42, 0.62, 0.95);
                                    vec3 ambientBottom = (push.night_scene != 0) ? vec3(0.01, 0.015, 0.03) : vec3(0.68, 0.62, 0.55) * 0.6;
                                    vec3 ambient = mix(ambientBottom, ambientTop, h);
                                    ambient *= clamp(1.0 - d * 0.18, 0.25, 1.0);

                                    vec3 sunIllum = light_col * (sunVisibility * powder * phase * 3.6);
                                    vec3 S = sunIllum + ambient * 1.35;

                                    float extinction = exp(-d * dt * 0.0035);
                                    cloudColor += transmittance * (1.0 - extinction) * S;
                                    transmittance *= extinction;

                                    if (transmittance < 0.01) break;
                                }
                            }
                        }
                    }
                }
                t += dt;
            }

            float haze = 1.0 - exp(-t_enter * 0.000028);
            cloudColor = mix(cloudColor, horizonCol, haze * 0.85);

            float horizonFade = smoothstep(0.055, 0.14, rd.y);
            sky = mix(sky, cloudColor, (1.0 - transmittance) * horizonFade);
        }
    }
    return sky;
}

vec3 sample_reflection_sky(vec3 rd, vec3 sun_dir, vec3 wind) {
    float zenith = clamp(rd.y, 0.0, 1.0);
    vec3 horizonCol = (push.night_scene != 0) ? vec3(0.04, 0.06, 0.12) : vec3(0.66, 0.78, 0.94);
    vec3 zenithCol  = (push.night_scene != 0) ? vec3(0.008, 0.015, 0.04) : vec3(0.10, 0.35, 0.86);
    vec3 sky = mix(horizonCol, zenithCol, pow(zenith, 0.60));

    float cosTheta = clamp(dot(rd, sun_dir), 0.0, 1.0);
    vec3 light_col = (push.night_scene != 0) ? vec3(0.35, 0.45, 0.65) * 1.2 : SUN_COLOR;
    sky += light_col * (pow(cosTheta, 1024.0) * 7.0 + pow(cosTheta, 45.0) * 2.0);
    return sky;
}

// -----------------------------------------------------------------------------------------
// Scene Setup
// -----------------------------------------------------------------------------------------
void init_scene() {
    g_scene[0] = SceneObject(
        SHAPE_SPHERE, MAT_CHECKER,
        vec3(0.0, -1e9, 0.0), vec3(0.0),
        vec3(0.0), vec3(0.0),
        1.0, 0.0
    );

    g_scene[1] = SceneObject(
        SHAPE_BOX, MAT_DIFFUSE,
        vec3(2.4, 0.6, -3.9), vec3(0.6, 0.6, 0.6),
        vec3(0.86, 0.86, 0.84), vec3(0.0),
        1.0, 0.0
    );

    g_scene[2] = SceneObject(
        SHAPE_SPHERE, MAT_METAL,
        vec3(-1.8, 0.5, -2.0), vec3(0.5, 0.0, 0.0),
        vec3(1.00, 0.82, 0.38), vec3(0.0),
        1.0, 0.0
    );

    g_scene[3] = SceneObject(
        SHAPE_MESH, MAT_GLASS,
        vec3(0.0, 1.0, -3.5), vec3(1.15, 1.15, 1.15),
        vec3(1.00, 1.00, 1.00), vec3(0.0),
        1.52, 0.04
    );

    g_scene[4] = SceneObject(
        SHAPE_MESH, MAT_TERRAIN,
        vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0),
        vec3(0.80, 0.80, 0.80), vec3(0.0),
        1.0, 0.0
    );
}

struct Ray {
    vec3 orig;
    vec3 dir;
};

vec3 ray_at(Ray r, float t) {
    return r.orig + t * r.dir;
}

struct HitRecord {
    vec3 p;
    vec3 normal;
    vec3 geo_normal;
    int obj_id;
    float t;
    bool front_face;
    float feature;
};

bool hit_sphere(SceneObject s, int id, Ray r, float t_min, float t_max, out HitRecord rec) {
    vec3 oc = r.orig - s.pos;
    float radius = s.extents.x;
    float half_b = dot(oc, r.dir);
    float c = dot(oc, oc) - radius * radius;
    float disc = half_b * half_b - c;
    if (disc < 0.0) return false;
    float sqrtd = sqrt(disc);
    float root = -half_b - sqrtd;
    bool front = true;
    if (root < t_min || t_max < root) {
        root = -half_b + sqrtd;
        front = false;
        if (root < t_min || t_max < root) return false;
    }
    rec.t = root;
    rec.front_face = front;
    rec.p = ray_at(r, root);
    vec3 outward_normal = (rec.p - s.pos) / radius;
    rec.geo_normal = front ? outward_normal : -outward_normal;
    rec.normal = rec.geo_normal;
    rec.obj_id = id;
    rec.feature = 0.0;
    return true;
}

bool hit_box(SceneObject b, int id, Ray r, float t_min, float t_max, out HitRecord rec) {
    vec3 pmin = b.pos - b.extents;
    vec3 pmax = b.pos + b.extents;
    vec3 invD = 1.0 / r.dir;
    float t0_col = t_min, t1_col = t_max;
    int hit_axis_0 = -1, hit_axis_1 = -1;
    float sign_0 = 1.0, sign_1 = 1.0;

    for (int a = 0; a < 3; a++) {
        float d = invD[a];
        float t0 = (pmin[a] - r.orig[a]) * d;
        float t1 = (pmax[a] - r.orig[a]) * d;
        float s0 = -1.0, s1 = 1.0;
        if (d < 0.0) {
            float tmp = t0; t0 = t1; t1 = tmp;
            s0 = 1.0; s1 = -1.0;
        }
        if (t0 > t0_col) { t0_col = t0; hit_axis_0 = a; sign_0 = s0; }
        if (t1 < t1_col) { t1_col = t1; hit_axis_1 = a; sign_1 = s1; }
        if (t1_col <= t0_col) return false;
    }

    if (t0_col > t_min) {
        rec.t = t0_col;
        vec3 out_norm = vec3(0.0);
        out_norm[hit_axis_0] = sign_0;
        rec.front_face = dot(r.dir, out_norm) < 0.0;
        rec.geo_normal = rec.front_face ? out_norm : -out_norm;
        rec.normal = rec.geo_normal;
        rec.p = ray_at(r, rec.t);
        rec.obj_id = id;
        rec.feature = 0.0;
        return true;
    } else if (t1_col < t_max) {
        rec.t = t1_col;
        vec3 out_norm = vec3(0.0);
        out_norm[hit_axis_1] = sign_1;
        rec.front_face = dot(r.dir, out_norm) < 0.0;
        rec.geo_normal = rec.front_face ? out_norm : -out_norm;
        rec.normal = rec.geo_normal;
        rec.p = ray_at(r, rec.t);
        rec.obj_id = id;
        rec.feature = 0.0;
        return true;
    }
    return false;
}

bool hit_mesh(SceneObject obj, int id, Ray r, float t_min, float t_max, out HitRecord rec) {
    vec3 inv_d = 1.0 / r.dir;
    float closest_t = t_max;
    bool hit_anything = false;

    int tlas_stack[8];
    int tlas_ptr = 0;
    if (push.tlas_count > 0) tlas_stack[tlas_ptr++] = 0;

    while (tlas_ptr > 0) {
        int t_idx = tlas_stack[--tlas_ptr];
        TLASNode tnode = tlas_nodes[push.tlas_offset + t_idx];

        vec3 t0 = (tnode.aabb_min.xyz - r.orig) * inv_d;
        vec3 t1 = (tnode.aabb_max.xyz - r.orig) * inv_d;
        vec3 tmin = min(t0, t1);
        vec3 tmax = max(t0, t1);
        float tenter = max(max(tmin.x, tmin.y), tmin.z);
        float texit  = min(min(tmax.x, tmax.y), tmax.z);

        if (tenter > texit || texit < t_min || tenter > closest_t) continue;

        if (tnode.aabb_max.w > 0.5) {
            int blas_root = int(tnode.aabb_min.w);
            int blas_stack[20];
            int blas_ptr = 0;
            blas_stack[blas_ptr++] = blas_root;

            while (blas_ptr > 0) {
                int node_idx = blas_stack[--blas_ptr];
                BVHNode node = bvh_nodes[node_idx];

                vec3 bt0 = (node.aabb_min.xyz - r.orig) * inv_d;
                vec3 bt1 = (node.aabb_max.xyz - r.orig) * inv_d;
                vec3 btmin = min(bt0, bt1);
                vec3 btmax = max(bt0, bt1);
                float btenter = max(max(btmin.x, btmin.y), btmin.z);
                float btexit  = min(min(btmax.x, btmax.y), btmax.z);

                if (btenter > btexit || btexit < t_min || btenter > closest_t) continue;

                int tri_count = int(node.aabb_max.w);
                if (tri_count > 0) {
                    int first_tri = int(node.aabb_min.w);
                    for (int i = 0; i < tri_count; ++i) {
                        GPU_Triangle tri = triangles[first_tri + i];

                        vec3 e1 = tri.v1.xyz - tri.v0.xyz;
                        vec3 e2 = tri.v2.xyz - tri.v0.xyz;
                        vec3 pvec = cross(r.dir, e2);
                        float det = dot(e1, pvec);
                        if (abs(det) < 1e-7) continue;
                        float inv_det = 1.0 / det;

                        vec3 tvec = r.orig - tri.v0.xyz;
                        float u = dot(tvec, pvec) * inv_det;
                        if (u < 0.0 || u > 1.0) continue;

                        vec3 qvec = cross(tvec, e1);
                        float v = dot(r.dir, qvec) * inv_det;
                        if (v < 0.0 || u + v > 1.0) continue;

                        float t = dot(e2, qvec) * inv_det;
                        if (t >= t_min && t < closest_t) {
                            closest_t = t;
                            hit_anything = true;

                            rec.t = t;
                            rec.p = ray_at(r, t);
                            rec.obj_id = (tri.v0.w >= 3.5) ? 4 : id;
                            rec.feature = tri.n0.w;

                            vec3 geo_norm = normalize(cross(e1, e2));
                            rec.front_face = dot(r.dir, geo_norm) < 0.0;
                            rec.geo_normal = rec.front_face ? geo_norm : -geo_norm;

                            float w = 1.0 - u - v;
                            vec3 interp_norm = w * tri.n0.xyz + u * tri.n1.xyz + v * tri.n2.xyz;
                            vec3 smooth_norm = length(interp_norm) > 1e-5 ? normalize(interp_norm) : geo_norm;
                            if (dot(smooth_norm, geo_norm) < 0.0) smooth_norm = -smooth_norm;
                            rec.normal = rec.front_face ? smooth_norm : -smooth_norm;
                        }
                    }
                } else {
                    int left_child = int(node.aabb_min.w);
                    blas_stack[blas_ptr++] = left_child + 1;
                    blas_stack[blas_ptr++] = left_child;
                }
            }
        } else {
            int left_child = int(tnode.aabb_min.w);
            tlas_stack[tlas_ptr++] = left_child + 1;
            tlas_stack[tlas_ptr++] = left_child;
        }
    }

    return hit_anything;
}

bool hit_mesh_any(Ray r, float t_min, float t_max) {
    vec3 inv_d = 1.0 / r.dir;
    int tlas_stack[8];
    int tlas_ptr = 0;
    if (push.tlas_count > 0) tlas_stack[tlas_ptr++] = 0;

    while (tlas_ptr > 0) {
        int t_idx = tlas_stack[--tlas_ptr];
        TLASNode tnode = tlas_nodes[push.tlas_offset + t_idx];

        vec3 t0 = (tnode.aabb_min.xyz - r.orig) * inv_d;
        vec3 t1 = (tnode.aabb_max.xyz - r.orig) * inv_d;
        vec3 tmin = min(t0, t1);
        vec3 tmax = max(t0, t1);
        float tenter = max(max(tmin.x, tmin.y), tmin.z);
        float texit  = min(min(tmax.x, tmax.y), tmax.z);

        if (tenter > texit || texit < t_min || tenter > t_max) continue;

        if (tnode.aabb_max.w > 0.5) {
            int blas_root = int(tnode.aabb_min.w);
            int blas_stack[16];
            int blas_ptr = 0;
            blas_stack[blas_ptr++] = blas_root;

            while (blas_ptr > 0) {
                int node_idx = blas_stack[--blas_ptr];
                BVHNode node = bvh_nodes[node_idx];

                vec3 bt0 = (node.aabb_min.xyz - r.orig) * inv_d;
                vec3 bt1 = (node.aabb_max.xyz - r.orig) * inv_d;
                vec3 btmin = min(bt0, bt1);
                vec3 btmax = max(bt0, bt1);
                float btenter = max(max(btmin.x, btmin.y), btmin.z);
                float btexit  = min(min(btmax.x, btmax.y), btmax.z);

                if (btenter > btexit || btexit < t_min || btenter > t_max) continue;

                int tri_count = int(node.aabb_max.w);
                if (tri_count > 0) {
                    int first_tri = int(node.aabb_min.w);
                    for (int i = 0; i < tri_count; ++i) {
                        GPU_Triangle tri = triangles[first_tri + i];
                        if (tri.v0.w < 3.5) continue; // Terrain only

                        vec3 e1 = tri.v1.xyz - tri.v0.xyz;
                        vec3 e2 = tri.v2.xyz - tri.v0.xyz;
                        vec3 pvec = cross(r.dir, e2);
                        float det = dot(e1, pvec);
                        if (abs(det) < 1e-7) continue;
                        float inv_det = 1.0 / det;

                        vec3 tvec = r.orig - tri.v0.xyz;
                        float u = dot(tvec, pvec) * inv_det;
                        if (u < 0.0 || u > 1.0) continue;

                        vec3 qvec = cross(tvec, e1);
                        float v = dot(r.dir, qvec) * inv_det;
                        if (v < 0.0 || u + v > 1.0) continue;

                        float t = dot(e2, qvec) * inv_det;
                        if (t >= t_min && t < t_max) return true;
                    }
                } else {
                    int left_child = int(node.aabb_min.w);
                    blas_stack[blas_ptr++] = left_child + 1;
                    blas_stack[blas_ptr++] = left_child;
                }
            }
        } else {
            int left_child = int(tnode.aabb_min.w);
            tlas_stack[tlas_ptr++] = left_child + 1;
            tlas_stack[tlas_ptr++] = left_child;
        }
    }
    return false;
}

bool hit_mesh_blocker_closest(Ray r, float t_min, float t_max, out float hit_t) {
    vec3 inv_d = 1.0 / r.dir;
    int tlas_stack[8];
    int tlas_ptr = 0;
    if (push.tlas_count > 0) tlas_stack[tlas_ptr++] = 0;

    float closest_t = t_max;
    bool hit_any = false;

    while (tlas_ptr > 0) {
        int t_idx = tlas_stack[--tlas_ptr];
        TLASNode tnode = tlas_nodes[push.tlas_offset + t_idx];

        vec3 t0 = (tnode.aabb_min.xyz - r.orig) * inv_d;
        vec3 t1 = (tnode.aabb_max.xyz - r.orig) * inv_d;
        vec3 tmin = min(t0, t1);
        vec3 tmax = max(t0, t1);
        float tenter = max(max(tmin.x, tmin.y), tmin.z);
        float texit  = min(min(tmax.x, tmax.y), tmax.z);

        if (tenter > texit || texit < t_min || tenter > closest_t) continue;

        if (tnode.aabb_max.w > 0.5) {
            int blas_root = int(tnode.aabb_min.w);
            int blas_stack[20];
            int blas_ptr = 0;
            blas_stack[blas_ptr++] = blas_root;

            while (blas_ptr > 0) {
                int node_idx = blas_stack[--blas_ptr];
                BVHNode node = bvh_nodes[node_idx];

                vec3 bt0 = (node.aabb_min.xyz - r.orig) * inv_d;
                vec3 bt1 = (node.aabb_max.xyz - r.orig) * inv_d;
                vec3 btmin = min(bt0, bt1);
                vec3 btmax = max(bt0, bt1);
                float btenter = max(max(btmin.x, btmin.y), btmin.z);
                float btexit  = min(min(btmax.x, btmax.y), btmax.z);

                if (btenter > btexit || btexit < t_min || btenter > closest_t) continue;

                int tri_count = int(node.aabb_max.w);
                if (tri_count > 0) {
                    int first_tri = int(node.aabb_min.w);
                    for (int i = 0; i < tri_count; ++i) {
                        GPU_Triangle tri = triangles[first_tri + i];
                        if (tri.v0.w < 3.5) continue;

                        vec3 e1 = tri.v1.xyz - tri.v0.xyz;
                        vec3 e2 = tri.v2.xyz - tri.v0.xyz;
                        vec3 pvec = cross(r.dir, e2);
                        float det = dot(e1, pvec);
                        if (abs(det) < 1e-7) continue;
                        float inv_det = 1.0 / det;

                        vec3 tvec = r.orig - tri.v0.xyz;
                        float u = dot(tvec, pvec) * inv_det;
                        if (u < 0.0 || u > 1.0) continue;

                        vec3 qvec = cross(tvec, e1);
                        float v = dot(r.dir, qvec) * inv_det;
                        if (v < 0.0 || u + v > 1.0) continue;

                        float t = dot(e2, qvec) * inv_det;
                        if (t >= t_min && t < closest_t) {
                            closest_t = t;
                            hit_any = true;
                        }
                    }
                } else {
                    int left_child = int(node.aabb_min.w);
                    blas_stack[blas_ptr++] = left_child + 1;
                    blas_stack[blas_ptr++] = left_child;
                }
            }
        } else {
            int left_child = int(tnode.aabb_min.w);
            tlas_stack[tlas_ptr++] = left_child + 1;
            tlas_stack[tlas_ptr++] = left_child;
        }
    }

    hit_t = closest_t;
    return hit_any;
}

bool world_hit(Ray r, float t_min, float t_max, out HitRecord rec) {
    float closest = t_max;
    bool hit = false;
    HitRecord temp;

    for (int i = 1; i < NUM_OBJECTS; ++i) {
        if (i == 4) continue;

        bool obj_hit = false;
        if (g_scene[i].shape == SHAPE_SPHERE) {
            obj_hit = hit_sphere(g_scene[i], i, r, t_min, closest, temp);
        } else if (g_scene[i].shape == SHAPE_BOX) {
            obj_hit = hit_box(g_scene[i], i, r, t_min, closest, temp);
        } else if (g_scene[i].shape == SHAPE_MESH) {
            obj_hit = hit_mesh(g_scene[i], i, r, t_min, closest, temp);
        }

        if (obj_hit) {
            closest = temp.t;
            rec = temp;
            hit = true;
        }
    }
    return hit;
}

// -----------------------------------------------------------------------------------------
// Soft Shadows (Analytical Foreground Cone-Tracing)
// -----------------------------------------------------------------------------------------
float sd_box(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sd_scene_opaque(vec3 p) {
    float d = INF;
    for (int i = 1; i < NUM_OBJECTS; ++i) {
        if (g_scene[i].mat_type == MAT_GLASS || g_scene[i].mat_type == MAT_TERRAIN) continue;

        float od = INF;
        if (g_scene[i].shape == SHAPE_SPHERE) {
            od = length(p - g_scene[i].pos) - g_scene[i].extents.x;
        } else if (g_scene[i].shape == SHAPE_BOX) {
            od = sd_box(p - g_scene[i].pos, g_scene[i].extents);
        }
        d = min(d, od);
    }
    return d;
}

// Smooth, noise-free analytical soft shadow for foreground sphere and box
float eval_soft_shadow(vec3 ro, vec3 rd, float mint, float maxt) {
    float res = 1.0;
    float t = mint;
    float ph = 1e10;

    for (int i = 0; i < 32; ++i) {
        vec3 p = ro + rd * t;
        float h = sd_scene_opaque(p);
        if (h < 0.0004) return 0.0;

        float y = h * h / (2.0 * ph);
        float d = sqrt(max(0.0, h * h - y * y));
        float cone_r = max(0.001, t - y) * FOREGROUND_SHADOW_SPREAD;

        res = min(res, clamp(d / cone_r, 0.0, 1.0));
        ph = h;

        t += clamp(h * 0.55, 0.012, 0.28);
        if (t > maxt) break;
    }
    return smoothstep(0.0, 1.0, res);
}

// Crisp, high-contrast glass shadow for the teapot
float eval_teapot_glass_transmittance(vec3 p, vec3 normal, vec3 sun_dir) {
    vec3 orig = p + normal * 0.003;
    HitRecord dummy;
    bool center_hit = hit_mesh(g_scene[3], 3, Ray(orig, sun_dir), 0.01, 20.0, dummy);
    if (!center_hit) {
        return 1.0;
    }

    // Soft silhouette edge
    vec3 sun_up = abs(sun_dir.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 sun_r = normalize(cross(sun_up, sun_dir));
    vec3 sun_t = cross(sun_dir, sun_r);
    float hits = 1.0;
    const float PENUMBRA = 0.038;
    for (int i = 0; i < 3; ++i) {
        float phi = float(i) * (TWO_PI / 3.0);
        vec3 dir = normalize(sun_dir + (sun_r * cos(phi) + sun_t * sin(phi)) * PENUMBRA);
        if (hit_mesh(g_scene[3], 3, Ray(orig, dir), 0.01, 20.0, dummy)) {
            hits += 1.0;
        }
    }
    float occ = hits / 4.0;
    return mix(1.0, 0.08, occ);
}

float calc_ambient_occlusion(vec3 p, vec3 n) {
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 4; ++i) {
        float h = 0.04 + 0.12 * float(i);
        float d = sd_scene_opaque(p + h * n);
        occ += (h - d) * sca;
        sca *= 0.72;
    }
    return clamp(1.0 - 1.8 * occ, 0.0, 1.0);
}

// -----------------------------------------------------------------------------------------
// Large Mountain Soft Shadows (Scale-Adaptive Blocker Cone for Distant Peaks)
// -----------------------------------------------------------------------------------------
float eval_large_scale_mountain_shadow(vec3 p, vec3 normal, vec3 sun_dir, float view_dist) {
    float normal_bias = max(0.25, view_dist * 0.0006);
    vec3 shadow_orig = p + normal * normal_bias + sun_dir * (normal_bias * 0.5);
    float t_min = normal_bias * 1.5;
    const float MAX_SHADOW_RANGE = 55000.0;

    float center_blocker_t = 0.0;
    bool center_hit = hit_mesh_blocker_closest(Ray(shadow_orig, sun_dir), t_min, MAX_SHADOW_RANGE, center_blocker_t);

    float dither = IGN(gl_FragCoord.xy);
    vec3 sun_up = abs(sun_dir.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 sun_right = normalize(cross(sun_up, sun_dir));
    vec3 sun_top = cross(sun_dir, sun_right);

    if (!center_hit) {
        float test_phi = dither * TWO_PI;
        vec3 test_dir = normalize(sun_dir + (sun_right * cos(test_phi) + sun_top * sin(test_phi)) * (SUN_ANGULAR_RADIUS * 1.25));
        float outer_blocker_t = 0.0;
        if (!hit_mesh_blocker_closest(Ray(shadow_orig, test_dir), t_min, MAX_SHADOW_RANGE, outer_blocker_t)) {
            return 1.0;
        }
        center_blocker_t = outer_blocker_t;
    }

    float blocker_scale = clamp(sqrt(center_blocker_t / 450.0), 0.15, 2.5);
    float penumbra_radius = SUN_ANGULAR_RADIUS * blocker_scale;

    const int SAMPLES = 6;
    float shadow_accum = 0.0;
    float total_weight = 0.0;
    float rot = dither * TWO_PI;

    for (int i = 0; i < SAMPLES; ++i) {
        float r = sqrt((float(i) + 0.5) / float(SAMPLES));
        float phi = float(i) * GOLDEN_ANGLE + rot;

        vec2 disk_offset = vec2(cos(phi), sin(phi)) * (r * penumbra_radius);
        vec3 sample_dir = normalize(sun_dir + sun_right * disk_offset.x + sun_top * disk_offset.y);

        bool s_hit = hit_mesh_any(Ray(shadow_orig, sample_dir), t_min, MAX_SHADOW_RANGE);

        float weight = 1.0 - 0.20 * r;
        if (!s_hit) shadow_accum += weight;
        total_weight += weight;
    }

    return smoothstep(0.0, 1.0, shadow_accum / total_weight);
}

// -----------------------------------------------------------------------------------------
// Metal Sphere Catacaustic: On-The-Fly Coddington Curvature & Energy Bounded
// -----------------------------------------------------------------------------------------
vec3 eval_metal_sphere_caustic(vec3 p, vec3 rec_norm, vec3 sun_dir) {
    vec3 sphere_pos = g_scene[2].pos;
    float radius = g_scene[2].extents.x;
    vec3 albedo = g_scene[2].albedo;

    vec3 to_sphere = sphere_pos - p;
    float dist = length(to_sphere);
    if (dist < radius * 1.005 || dist > 4.2) return vec3(0.0);

    vec3 L = sun_dir;
    vec3 v_est = -to_sphere / dist;
    vec3 N = normalize(L + v_est);

    for (int iter = 0; iter < 4; ++iter) {
        vec3 S = sphere_pos + radius * N;
        vec3 to_pt = p - S;
        float d = length(to_pt);
        if (d < 1e-4) break;
        N = normalize(L + to_pt / d);
    }

    vec3 S = sphere_pos + radius * N;
    vec3 ray_dir = p - S;
    float d = length(ray_dir);
    if (d < 1e-4) return vec3(0.0);
    vec3 D = ray_dir / d;

    float cos_theta_i = dot(N, L);
    if (cos_theta_i <= 0.02) return vec3(0.0);

    vec3 R_dir = reflect(-L, N);
    float align = dot(R_dir, D);
    if (align < 0.985) return vec3(0.0);

    float rec_factor = max(dot(rec_norm, -D), 0.0);
    if (rec_factor <= 0.0) return vec3(0.0);

    HitRecord dummy;
    if (hit_box(g_scene[1], 1, Ray(S + N * 0.01, L), 0.02, 30.0, dummy)) return vec3(0.0);

    float rho_s = 0.5 * radius * cos_theta_i;
    float rho_t = (0.5 * radius) / max(cos_theta_i, 1e-4);

    float delta_s = 1.0 - d / rho_s;
    float delta_t = 1.0 - d / rho_t;

    float sun_waist = d * tan(0.032);
    float J_s = sqrt(delta_s * delta_s + (sun_waist / rho_s) * (sun_waist / rho_s));
    float J_t = sqrt(delta_t * delta_t + (sun_waist / rho_t) * (sun_waist / rho_t));

    float det_J = max(J_s * J_t, 0.035);
    float flux_gain = min(1.0 / det_J, 35.0);

    vec3 F = albedo + (vec3(1.0) - albedo) * pow(1.0 - cos_theta_i, 5.0);
    float envelope = smoothstep(0.985, 0.9998, align);
    float geom = (radius * radius * cos_theta_i) / (4.0 * max(d * d, 0.05)) * rec_factor;

    return albedo * F * SUN_COLOR * (flux_gain * geom * envelope);
}

// -----------------------------------------------------------------------------------------
// Teapot Diacaustics: High-Precision Transposed Bilinear Reconstruction
// -----------------------------------------------------------------------------------------
// Maps coordinates correctly using px * 512 + py (column-major in CausticBuffer)
// and performs smooth, non-negative bilinear interpolation to eliminate Gibbs ringing.
vec3 sample_continuous_caustic(vec3 p) {
    const vec2 map_min = vec2(-4.0, -4.5);
    const vec2 map_max = vec2( 2.0,  1.5);
    vec2 uv = (p.xz - map_min) / (map_max - map_min);

    if (uv.x < 0.001 || uv.x > 0.999 || uv.y < 0.001 || uv.y > 0.999) {
        return vec3(0.0);
    }

    vec2 coord = clamp(uv * 511.0, 0.0, 511.0);
    vec2 i0 = floor(coord);
    vec2 f = coord - i0;

    int x0 = int(i0.x);
    int y0 = int(i0.y);
    int x1 = min(x0 + 1, 511);
    int y1 = min(y0 + 1, 511);

    // Column-major buffer index px * 512 + py (resolves coordinate transposition)
    vec3 c00 = caustic_map[x0 * 512 + y0].rgb;
    vec3 c10 = caustic_map[x1 * 512 + y0].rgb;
    vec3 c01 = caustic_map[x0 * 512 + y1].rgb;
    vec3 c11 = caustic_map[x1 * 512 + y1].rgb;

    vec3 caustic = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
    return max(vec3(0.0), caustic);
}

vec3 refract_vec(vec3 uv, vec3 n, float etai_over_etat) {
    float cos_theta = min(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

float fresnel_schlick(float cosine, float ref_idx) {
    float r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

// -----------------------------------------------------------------------------------------
// Shading Pipelines
// -----------------------------------------------------------------------------------------
vec3 shade_surface(vec3 albedo, HitRecord rec, vec3 sun_dir) {
    float NdotL = max(dot(rec.normal, sun_dir), 0.0);

    float opaque_shadow = eval_soft_shadow(rec.p + rec.geo_normal * 0.005, sun_dir, 0.02, 14.0);
    if (hit_mesh_any(Ray(rec.p + rec.geo_normal * 0.01, sun_dir), 0.05, 25000.0)) {
        opaque_shadow = 0.0;
    }

    float glass_transmittance = eval_teapot_glass_transmittance(rec.p, rec.geo_normal, sun_dir);
    vec3 direct = albedo * SUN_COLOR * (NdotL * opaque_shadow * glass_transmittance);

    vec3 diacaustic = sample_continuous_caustic(rec.p);
    vec3 caustic_light = albedo * SUN_COLOR * diacaustic * opaque_shadow;

    vec3 catacaustic = eval_metal_sphere_caustic(rec.p, rec.normal, sun_dir) * opaque_shadow;

    float ao = calc_ambient_occlusion(rec.p, rec.normal);
    vec3 ambient_sky = mix(vec3(0.55, 0.58, 0.62), vec3(0.25, 0.45, 0.85), max(rec.normal.y, 0.0));
    if (push.night_scene != 0) ambient_sky *= 0.15;
    vec3 ambient = ambient_sky * 0.30 * albedo * ao;

    return direct + caustic_light + catacaustic + ambient;
}

// -----------------------------------------------------------------------------------------
// Terrain Texturing & Geological Shading Pipeline
// -----------------------------------------------------------------------------------------
vec3 shade_terrain(HitRecord rec, vec3 view_dir, vec3 sun_dir, vec3 wind, float dragon_cShadow) {
    float dist = length(rec.p - push.origin.xyz);
    float slope = 1.0 - clamp(rec.normal.y, 0.0, 1.0);
    float alt = rec.p.y;

    vec3 eff_normal = rec.normal;
    float micro_ao = 1.0;

    // 1. Surface Tangent Frame
    vec3 up_axis = abs(rec.normal.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up_axis, rec.normal));
    vec3 bitangent = cross(rec.normal, tangent);

    // 2. Foreground Pebble / Gravel Details (< 40m)
    if (dist < 40.0) {
        float fade = 1.0 - dist / 40.0;
        float stone_id;
        float v_dist = voronoi2D(rec.p.xz * 14.0, stone_id);
        float stone_facet = smoothstep(0.04, 0.28, v_dist);
        vec2 stone_tilt = hash22(vec2(stone_id * 100.0, stone_id * 50.0)) * 0.35;
        float crevice_ao = smoothstep(0.03, 0.22, v_dist);
        micro_ao = mix(1.0, crevice_ao * 0.75 + 0.25, fade);

        vec3 pebble_grad = (tangent * stone_tilt.x + bitangent * stone_tilt.y) * (stone_facet * 0.35 * fade);
        eff_normal = normalize(eff_normal - pebble_grad);
    }
    
    // 3. Mountain Rock Strata & Ridge Fractures
    float tectonic_strata = 0.0;
    if (dist > 10.0 && dist < 35000.0) {
        vec3 fault_plane = normalize(vec3(0.58, 0.72, -0.38));
        float p_bedding = dot(rec.p, fault_plane) * 0.032 + gradNoise2D(rec.p.xz * 0.0035) * 0.85;
        float fault_fracture = gradNoise2D(rec.p.xz * 0.007 + vec2(42.1, 19.3));
        
        tectonic_strata = sin(p_bedding * 6.2831) * 0.5 + sin(p_bedding * 18.849 + fault_fracture * 3.0) * 0.25;
        tectonic_strata = smoothstep(-0.35, 0.35, tectonic_strata);

        float strata_fade = smoothstep(0.12, 0.45, slope) * (1.0 - smoothstep(18000.0, 32000.0, dist));
        vec3 strata_offset = (tangent * fault_plane.x + bitangent * fault_plane.z) * (tectonic_strata * 0.40 * strata_fade);
        eff_normal = normalize(eff_normal - strata_offset);
    }

    // 4. Large-scale Cliff Crags
    if (dist > 25.0 && dist < 35000.0) {
        float fade = smoothstep(25.0, 120.0, dist) * (1.0 - smoothstep(22000.0, 34000.0, dist));
        float eps = max(4.0, dist * 0.0035);
        vec2 p_crag = rec.p.xz * 0.005;
        float h0 = fbm2D_lod(p_crag, dist, 0.005, 3);
        float hx = fbm2D_lod(p_crag + vec2(eps * 0.005, 0.0), dist, 0.005, 3);
        float hz = fbm2D_lod(p_crag + vec2(0.0, eps * 0.005), dist, 0.005, 3);

        vec3 crag_deriv = (tangent * (hx - h0) + bitangent * (hz - h0)) / (eps * 0.005);
        eff_normal = normalize(eff_normal - crag_deriv * (0.35 * fade));
    }

    // 5. Alpine Biome Mosaic & Altitude-Layered Materials
    float macro_geology = gradNoise2D(rec.p.xz * 0.00025 + vec2(15.2, 73.1));
    float biome_patch   = gradNoise2D(rec.p.xz * 0.00140 + vec2(33.8, 89.4));
    float soil_noise    = fbm2D_lod(rec.p.xz * 0.025 + vec2(3.1, 5.7), dist, 0.025, 3);

    vec3 grass_weathered = vec3(0.20, 0.24, 0.14);
    vec3 grass_dry_straw = vec3(0.34, 0.29, 0.18);
    vec3 peat_dark_soil  = vec3(0.14, 0.10, 0.08);
    vec3 alpine_heather  = vec3(0.26, 0.20, 0.17);
    
    vec3 meadow_col = mix(grass_weathered, grass_dry_straw, biome_patch);
    meadow_col = mix(meadow_col, alpine_heather, smoothstep(0.40, 0.70, macro_geology) * 0.40);
    meadow_col = mix(meadow_col, peat_dark_soil, smoothstep(0.55, 0.80, soil_noise) * 0.55);

    // Terrace around spawn platform
    float d_dragon = length(rec.p.xz - vec2(0.0, -3.5));
    float terrace_mask = 1.0 - smoothstep(4.0, 10.5, d_dragon);
    if (terrace_mask > 0.001) {
        vec3 terrace_stone = mix(vec3(0.42, 0.40, 0.38), vec3(0.50, 0.48, 0.45), gradNoise2D(rec.p.xz * 1.0));
        meadow_col = mix(meadow_col, terrace_stone, terrace_mask * 0.90);
    }

    // Rock, Scree & Granitic Cliffs
    vec3 scree_granite  = vec3(0.36, 0.35, 0.34);
    vec3 scree_oxidized = vec3(0.44, 0.37, 0.28);
    vec3 scree_col = mix(scree_granite, scree_oxidized, biome_patch);
    float talus_slope = smoothstep(0.10, 0.28, slope) * (1.0 - smoothstep(0.35, 0.55, slope));

    vec3 rock_granite  = vec3(0.34, 0.35, 0.37);
    vec3 rock_dolomite = vec3(0.46, 0.44, 0.40);
    vec3 rock_cliff    = mix(rock_granite, rock_dolomite, macro_geology);
    rock_cliff *= (0.78 + 0.38 * tectonic_strata);

    float rock_factor = smoothstep(0.18, 0.42, slope);
    vec3 ground_base = mix(meadow_col, scree_col, talus_slope * 0.85);
    vec3 albedo = mix(ground_base, rock_cliff, rock_factor);

    // High Altitude Snow on Flatter Ridges
    float snow_altitude = smoothstep(1400.0, 2200.0, alt);
    float snow_flatness = 1.0 - smoothstep(0.20, 0.48, slope);
    float snow_amount = clamp(snow_altitude * snow_flatness * (0.8 + 0.4 * macro_geology), 0.0, 1.0);
    vec3 snow_col = vec3(0.92, 0.95, 0.98);
    albedo = mix(albedo, snow_col, snow_amount);

    vec3 light_col = (push.night_scene != 0) ? vec3(0.35, 0.45, 0.65) * 1.2 : SUN_COLOR;

    // 6. Direct Illumination & Scale-Adaptive Shadows
    float NdotL = max(dot(eff_normal, sun_dir), 0.0);
    float terminator = smoothstep(0.0, 0.10, NdotL);
    float effective_NdotL = NdotL * terminator;

    float opaque_shadow = 1.0;
    if (effective_NdotL > 0.0001) {
        if (dist > 35.0) {
            opaque_shadow = eval_large_scale_mountain_shadow(rec.p, eff_normal, sun_dir, dist);
        } else {
            // Near camera: Smooth analytical shadow to keep the floor clean and dither-free
            opaque_shadow = eval_soft_shadow(rec.p + eff_normal * 0.008, sun_dir, 0.02, 16.0);
        }
    } else {
        opaque_shadow = 0.0;
    }

    float cloud_shadow = getCloudShadow(rec.p, sun_dir, wind);
    opaque_shadow *= cloud_shadow;

    // Glass occlusion near origin
    float glass_transmittance = 1.0;
    if (dist < 15.0) {
        glass_transmittance = eval_teapot_glass_transmittance(rec.p, eff_normal, sun_dir);
    }

    vec3 direct = albedo * (light_col * effective_NdotL) * (opaque_shadow * glass_transmittance);

    // 7. Ambient Skylight & Valley Ground Bounce
    float valley_depth = clamp((alt + 100.0) / 450.0, 0.45, 1.0);
    float macro_ao = valley_depth * micro_ao;
    vec3 sky_ambient = mix(vec3(0.60, 0.72, 0.88), vec3(0.16, 0.38, 0.92), eff_normal.y * 0.5 + 0.5) * (1.20 * macro_ao);
    vec3 ground_bounce = SUN_COLOR * 0.12 * max(-eff_normal.y * 0.5 + 0.5, 0.0) * max(SUN_DIR.y, 0.0) * macro_ao;
    vec3 ambient = albedo * (sky_ambient + ground_bounce);

    // 8. Foreground Caustics
    vec3 caustic_total = vec3(0.0);
    if (dist < 15.0) {
        vec3 diacaustic = sample_continuous_caustic(rec.p);
        caustic_total += albedo * SUN_COLOR * diacaustic * opaque_shadow;

        vec3 catacaustic = eval_metal_sphere_caustic(rec.p, eff_normal, sun_dir) * opaque_shadow;
        caustic_total += catacaustic;
    }

    return direct + ambient + caustic_total;
}

// -----------------------------------------------------------------------------------------
// Recursive Path Tracing Loop
// -----------------------------------------------------------------------------------------
vec3 trace_path(Ray r, vec3 sun_dir, vec3 wind, float dragon_cShadow, out HitRecord primary_rec, out bool hit_primary) {
    vec3 throughput = vec3(1.0);
    vec3 accumulated = vec3(0.0);
    hit_primary = false;

    for (int depth = 5; depth > 0; --depth) {
        HitRecord rec;
        if (!world_hit(r, 0.001, INF, rec)) {
            if (depth == 5) {
                accumulated += throughput * render_volumetric_sky(push.origin.xyz, r.dir, SUN_DIR, push.time, gl_FragCoord.xy, wind);
            } else {
                accumulated += throughput * sample_reflection_sky(r.dir, sun_dir, wind);
            }
            break;
        }

        if (depth == 5) {
            primary_rec = rec;
            hit_primary = true;
        }

        SceneObject obj = g_scene[rec.obj_id];

        if (obj.mat_type == MAT_DIFFUSE) {
            accumulated += throughput * shade_surface(obj.albedo, rec, sun_dir);
            break;
        }
        else if (obj.mat_type == MAT_METAL) {
            vec3 reflected = reflect(r.dir, rec.normal);
            float cos_theta = max(dot(-r.dir, rec.normal), 0.0);
            vec3 F = obj.albedo + (vec3(1.0) - obj.albedo) * pow(1.0 - cos_theta, 5.0);

            throughput *= F;
            r = Ray(rec.p + rec.geo_normal * 0.0025, reflected);
        }
        else if (obj.mat_type == MAT_TERRAIN) {
            accumulated += throughput * shade_terrain(rec, r.dir, sun_dir, wind, dragon_cShadow);
            break;
        }
        else if (obj.mat_type == MAT_GLASS) {
            float ref_idx = obj.ior;
            float etai_over_etat = rec.front_face ? (1.0 / ref_idx) : ref_idx;
            vec3 unit_dir = r.dir;

            vec3 n = rec.normal;
            float cos_theta_raw = dot(-unit_dir, n);
            if (cos_theta_raw < 0.001) {
                n = normalize(n + (-unit_dir) * (0.02 - cos_theta_raw));
            }

            float cos_theta = min(dot(-unit_dir, n), 1.0);
            float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
            float refl = fresnel_schlick(cos_theta, etai_over_etat);

            bool cannot_refract = (etai_over_etat * sin_theta > 1.0);
            vec3 refl_dir = reflect(unit_dir, n);

            if (cannot_refract) {
                r = Ray(rec.p + rec.geo_normal * 0.0025, refl_dir);
            } else if (rec.front_face) {
                vec3 sky_refl = sample_reflection_sky(refl_dir, sun_dir, wind);
                accumulated += throughput * refl * sky_refl;

                vec3 refr_dir = refract_vec(unit_dir, n, etai_over_etat);
                r = Ray(rec.p - rec.geo_normal * 0.0025, refr_dir);
                throughput *= (1.0 - refl) * vec3(0.96, 0.99, 0.97);
            } else {
                if (refl > 0.85) {
                    r = Ray(rec.p + rec.geo_normal * 0.0025, refl_dir);
                    throughput *= refl;
                } else {
                    vec3 refr_dir = refract_vec(unit_dir, n, etai_over_etat);
                    r = Ray(rec.p - rec.geo_normal * 0.0025, refr_dir);
                    throughput *= (1.0 - refl);
                }
            }
        }
    }
    return accumulated;
}

// -----------------------------------------------------------------------------------------
// ACES Tonemapping
// -----------------------------------------------------------------------------------------
vec3 aces_fitted(vec3 color) {
    const mat3 ACESInputMat = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );

    const mat3 ACESOutputMat = mat3(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );

    vec3 v = ACESInputMat * color;
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return clamp(ACESOutputMat * (a / b), 0.0, 1.0);
}

vec3 linear_to_srgb(vec3 c) {
    vec3 low = c * 12.92;
    vec3 high = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(low, high, step(vec3(0.0031308), c));
}

vec3 tonemap_aces(vec3 hdr) {
    float exp_factor = (push.exposure > 0.001) ? push.exposure : 0.88;
    return linear_to_srgb(aces_fitted(hdr * exp_factor));
}

vec3 evaluate_sample(Ray r, vec3 wind, float dragon_cShadow, float cam_cloud_shadow,
                     vec3 horizonCol, vec3 zenithCol, vec3 light_col, out HitRecord primary_rec, out bool hit_primary) {
    vec3 sample_col = trace_path(r, SUN_DIR, wind, dragon_cShadow, primary_rec, hit_primary);

    if (hit_primary) {
        float hit_dist = primary_rec.t;
        float sampleY = (r.orig.y + primary_rec.p.y) * 0.5;
        float heightFactor = exp(-max(sampleY, 0.0) * 0.00045);

        vec3 extinction_coeff = vec3(0.000030, 0.000055, 0.000105);
        vec3 optical_depth = extinction_coeff * (hit_dist * heightFactor);
        vec3 transmittance = exp(-optical_depth);

        float cull_fade = 1.0 - smoothstep(22000.0, 36000.0, hit_dist);
        transmittance = mix(vec3(0.0), transmittance, cull_fade);

        float cos_v_sun = max(dot(r.dir, SUN_DIR), 0.0);
        float mie_sun = pow(cos_v_sun, 32.0);
        vec3 rayleigh_inscatter = mix(horizonCol, zenithCol, clamp(r.dir.y * 0.6 + 0.4, 0.0, 1.0));
        vec3 inscatter_col = mix(rayleigh_inscatter, light_col * 0.45, mie_sun * (0.65 * cam_cloud_shadow));

        sample_col = sample_col * transmittance + inscatter_col * (vec3(1.0) - transmittance);
    }
    return sample_col;
}

// -----------------------------------------------------------------------------------------
// Main Pipeline
// -----------------------------------------------------------------------------------------
void main() {
    init_scene();

    const vec3 seedOffset = vec3(2661.1, -1277.6, -1194.2);
    vec3 wind = vec3(push.time * 0.008, push.time * 0.002, push.time * 0.006) + seedOffset;

    float inv_w = 1.0 / (push.resolution.x - 1.0);
    float inv_h = 1.0 / (push.resolution.y - 1.0);

    float i_coord = gl_FragCoord.x - 0.5;
    float j_coord = (push.resolution.y - gl_FragCoord.y) - 0.5;

    float u_base = i_coord * inv_w;
    float v_base = j_coord * inv_h;
    vec3 pixel_base_dir = push.lower_left_corner.xyz + u_base * push.horizontal.xyz + v_base * push.vertical.xyz - push.origin.xyz;

    vec3 cam_forward = normalize(push.cam_dir.xyz);
    vec3 cam_right = normalize(cross(cam_forward, vec3(0.0, 1.0, 0.0)));
    vec3 cam_up = cross(cam_right, cam_forward);

    float pixelDither = IGN(gl_FragCoord.xy);

    vec3 light_col = (push.night_scene != 0) ? vec3(0.35, 0.45, 0.65) * 1.2 : SUN_COLOR;
    vec3 horizonCol = (push.night_scene != 0) ? vec3(0.04, 0.06, 0.12) : vec3(0.66, 0.78, 0.94);
    vec3 zenithCol  = (push.night_scene != 0) ? vec3(0.008, 0.015, 0.04) : vec3(0.10, 0.35, 0.86);

    float cam_cloud_shadow = getCloudShadow(push.origin.xyz, SUN_DIR, wind);
    float dragon_cShadow = getCloudShadow(vec3(0.0, 1.0, -3.5), SUN_DIR, wind);

    Ray r0 = Ray(push.origin.xyz, normalize(pixel_base_dir));
    HitRecord rec0;
    bool hit0 = false;
    vec3 col0 = evaluate_sample(r0, wind, dragon_cShadow, cam_cloud_shadow, horizonCol, zenithCol, light_col, rec0, hit0);

    bool is_glass = (hit0 && g_scene[rec0.obj_id].mat_type == MAT_GLASS);
    float coc_pixels = 0.0;
    if (push.dof_enabled != 0 && push.aperture > 0.0) {
        float d = hit0 ? rec0.t : 10000.0;
        coc_pixels = abs(1.0 - push.focal_dist / max(d, 0.001)) * push.aperture * (push.resolution.y / 1.041);
    }

    bool needs_extra_samples = is_glass || (coc_pixels > 1.8);

    vec3 final_hdr;
    if (!needs_extra_samples) {
        final_hdr = col0;
    } else {
        vec3 accumulated = col0;
        const vec2 sub_offsets[3] = vec2[](
            vec2(-0.35,  0.35),
            vec2( 0.35, -0.35),
            vec2( 0.35,  0.35)
        );

        for (int s = 0; s < 3; ++s) {
            vec3 pinhole_dir = normalize(pixel_base_dir + (sub_offsets[s].x * inv_w) * push.horizontal.xyz + (sub_offsets[s].y * inv_h) * push.vertical.xyz);
            Ray r;
            if (push.dof_enabled != 0 && push.aperture > 0.0) {
                float phi = float(s) * (TWO_PI / 3.0) + pixelDither * TWO_PI;
                vec3 pupil_offset = (cos(phi) * cam_right + sin(phi) * cam_up) * (push.aperture * 0.7);

                float ft = push.focal_dist / max(dot(pinhole_dir, cam_forward), 0.001);
                vec3 focus_point = push.origin.xyz + pinhole_dir * ft;
                vec3 eye_orig = push.origin.xyz + pupil_offset;
                r = Ray(eye_orig, normalize(focus_point - eye_orig));
            } else {
                r = Ray(push.origin.xyz, pinhole_dir);
            }

            HitRecord rec_s;
            bool hit_s = false;
            accumulated += evaluate_sample(r, wind, dragon_cShadow, cam_cloud_shadow, horizonCol, zenithCol, light_col, rec_s, hit_s);
        }
        final_hdr = accumulated * 0.25;
    }

    vec3 final_color = tonemap_aces(final_hdr);
    final_color += (pixelDither - 0.5) / 128.0;
    outColor = vec4(clamp(final_color, 0.0, 1.0), 1.0);
}