#version 450
layout(local_size_x = 256) in;
layout(std430, binding = 0) writeonly buffer Noise { float values[]; } noise;
layout(push_constant) uniform Parameters { uint count; uint seed_low; uint seed_high; } p;
uint hash32(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16); }
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.count) return;
    uint pair = i >> 1;
    float u1 = (float(hash32(pair ^ p.seed_low)) + 1.0) / 4294967297.0;
    float u2 = (float(hash32(pair ^ p.seed_high ^ 0x9e3779b9u)) + 0.5) / 4294967296.0;
    float radius = sqrt(-2.0 * log(u1));
    float angle = 6.283185307179586 * u2;
    noise.values[i] = radius * ((i & 1u) == 0u ? cos(angle) : sin(angle));
}
