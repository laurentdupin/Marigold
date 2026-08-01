#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(r32f, set = 0, binding = 0) uniform writeonly image2D output_image;
layout(set = 0, binding = 1, std430) readonly buffer Depth {
    float values[];
} depth;
layout(push_constant) uniform Parameters { uint width; uint height; } p;
void main(){uvec2 xy=gl_GlobalInvocationID.xy;if(xy.x>=p.width||xy.y>=p.height)return;imageStore(output_image,ivec2(xy),vec4(depth.values[xy.y*p.width+xy.x],0,0,1));}
