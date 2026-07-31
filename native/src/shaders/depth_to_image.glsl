#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer Depth { float values[]; } depth;
layout(r32f, binding = 1) uniform writeonly image2D output_image;
layout(push_constant) uniform Parameters { uint width; uint height; } p;
void main(){uvec2 xy=gl_GlobalInvocationID.xy;if(xy.x>=p.width||xy.y>=p.height)return;imageStore(output_image,ivec2(xy),vec4(depth.values[xy.y*p.width+xy.x],0,0,1));}
