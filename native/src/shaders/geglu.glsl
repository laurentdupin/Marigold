#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float data[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input { float data[]; } input_buffer;
layout(push_constant) uniform Parameters { uint rows; uint dimensions; } p;
float erf_approx(float value) {
    float sign_value = value < 0.0 ? -1.0 : 1.0;
    float x = abs(value);
    float t = 1.0 / (1.0 + 0.3275911 * x);
    float polynomial =
        (((((1.061405429 * t - 1.453152027) * t) +
        1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return sign_value * (1.0 - polynomial * exp(-x * x));
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.rows * p.dimensions) return;
    uint row = i / p.dimensions;
    uint d = i % p.dimensions;
    uint base = row * p.dimensions * 2;
    float gate = input_buffer.data[base + p.dimensions + d];
    float gelu = 0.5 * gate *
        (1.0 + erf_approx(gate * 0.7071067811865476));
    output_buffer.data[i] = input_buffer.data[base + d] * gelu;
}
