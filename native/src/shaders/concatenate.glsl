#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float data[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Left { float data[]; } left_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Right { float data[]; } right_buffer;
layout(push_constant) uniform Parameters { uint left_count; uint right_count; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < p.left_count) output_buffer.data[i] = left_buffer.data[i];
    else if (i < p.left_count + p.right_count)
        output_buffer.data[i] = right_buffer.data[i - p.left_count];
}
