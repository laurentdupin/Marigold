#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float data[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input { float data[]; } input_buffer;
layout(push_constant) uniform Parameters { uint width; uint height; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint spatial = p.width * p.height;
    if (i >= 3 * spatial) return;
    uint channel = i / spatial;
    uint position = i % spatial;
    output_buffer.data[i] =
        input_buffer.data[position * 3 + channel] * 2.0 - 1.0;
}
