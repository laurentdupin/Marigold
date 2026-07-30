#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) buffer Values { float data[]; } values;
layout(set = 0, binding = 1, std430) readonly buffer Channel { float data[]; } channel_buffer;
layout(push_constant) uniform Parameters { uint channels; uint spatial; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < p.channels * p.spatial)
        values.data[i] += channel_buffer.data[i / p.spatial];
}
