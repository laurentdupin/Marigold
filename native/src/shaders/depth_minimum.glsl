#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) readonly buffer Input { float values[]; } input_data;
layout(std430, binding = 1) writeonly buffer Range { float minimum; } range_data;
layout(push_constant) uniform Parameters { uint count; } p;
void main() { float v = input_data.values[0]; for (uint i = 1; i < p.count; ++i) v = min(v, input_data.values[i]); range_data.minimum = v; }
