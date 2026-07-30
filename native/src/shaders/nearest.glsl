#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float data[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input { float data[]; } input_buffer;
layout(push_constant) uniform Parameters {
    uint input_width; uint input_height; uint output_width; uint output_height; uint channels;
} p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint output_spatial = p.output_width * p.output_height;
    if (i >= p.channels * output_spatial) return;
    uint channel = i / output_spatial;
    uint position = i % output_spatial;
    uint x = position % p.output_width;
    uint y = position / p.output_width;
    uint sx = min(p.input_width - 1, x * p.input_width / p.output_width);
    uint sy = min(p.input_height - 1, y * p.input_height / p.output_height);
    output_buffer.data[i] =
        input_buffer.data[(channel * p.input_height + sy) * p.input_width + sx];
}
