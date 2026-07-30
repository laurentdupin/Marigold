#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float data[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input { float data[]; } input_buffer;
layout(push_constant) uniform Parameters { uint tokens; uint channels; uint reverse; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.tokens * p.channels) return;
    uint token = i / p.channels;
    uint channel = i % p.channels;
    uint nchw = channel * p.tokens + token;
    if (p.reverse == 0) output_buffer.data[i] = input_buffer.data[nchw];
    else output_buffer.data[nchw] = input_buffer.data[i];
}
