#version 450
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Depth { float values[]; } depth;
layout(std430, binding = 1) readonly buffer Range { float minimum; } range_data;
layout(push_constant) uniform Parameters { uint count; } p;
void main() { uint i=gl_GlobalInvocationID.x; if(i>=p.count)return; float d=1.0-range_data.minimum; depth.values[i]=d>0.0?(depth.values[i]-range_data.minimum)/d:0.0; }
