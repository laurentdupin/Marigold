#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) buffer Prediction { float data[]; } prediction;
layout(set = 0, binding = 1, std430) readonly buffer Noise { float data[]; } noise;
layout(push_constant) uniform Parameters { uint count; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.count) return;
    const float alpha = 0.00466009508818388;
    const float c_skip = 2.505007534736592e-9;
    float original =
        sqrt(alpha) * noise.data[i] -
        sqrt(1.0 - alpha) * prediction.data[i];
    prediction.data[i] =
        (original + c_skip * noise.data[i]) / 0.18215;
}
