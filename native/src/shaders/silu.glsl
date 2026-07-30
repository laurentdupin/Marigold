#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) buffer Values { float data[]; } values;
layout(push_constant) uniform Parameters { uint count; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < p.count) {
        float v = values.data[i];
        values.data[i] = v / (1.0 + exp(-v));
    }
}
