#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float data[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Posterior { float data[]; } posterior;
layout(set = 0, binding = 2, std430) readonly buffer Noise { float data[]; } noise;
layout(push_constant) uniform Parameters { uint count; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.count) return;
    float log_variance = clamp(posterior.data[p.count + i], -30.0, 20.0);
    output_buffer.data[i] =
        (posterior.data[i] + exp(0.5 * log_variance) * noise.data[i]) *
        0.18215;
}
