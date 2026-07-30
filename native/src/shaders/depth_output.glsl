#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float data[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Decoded { float data[]; } decoded;
layout(push_constant) uniform Parameters {
    uint source_width; uint source_height; uint target_width; uint target_height;
} p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint count = p.target_width * p.target_height;
    if (i >= count) return;
    uint x = i % p.target_width;
    uint y = i / p.target_width;
    uint sx = min(p.source_width - 1, x * p.source_width / p.target_width);
    uint sy = min(p.source_height - 1, y * p.source_height / p.target_height);
    uint source = sy * p.source_width + sx;
    uint spatial = p.source_width * p.source_height;
    float sum = 0.0;
    for (uint c = 0; c < 3; ++c)
        sum += clamp(decoded.data[c * spatial + source] * 0.5 + 0.5, 0.0, 1.0);
    output_buffer.data[i] = sum / 3.0;
}
