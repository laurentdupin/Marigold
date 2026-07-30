#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Prediction {
    float data[];
} prediction;
layout(set = 0, binding = 2, std430) readonly buffer Sample {
    float data[];
} sample_buffer;
layout(push_constant) uniform Parameters {
    uint count;
    float alpha;
    float previous_alpha;
} parameters;
void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index >= parameters.count) {
        return;
    }
    const float sqrt_alpha = sqrt(parameters.alpha);
    const float sqrt_beta = sqrt(1.0 - parameters.alpha);
    const float original =
        sqrt_alpha * sample_buffer.data[index] -
        sqrt_beta * prediction.data[index];
    const float epsilon =
        sqrt_alpha * prediction.data[index] +
        sqrt_beta * sample_buffer.data[index];
    output_buffer.data[index] =
        sqrt(parameters.previous_alpha) * original +
        sqrt(1.0 - parameters.previous_alpha) * epsilon;
}
