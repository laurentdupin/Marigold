#version 450 core
layout(local_size_x=8,local_size_y=8,local_size_z=1)in;
layout(set=0,binding=0,std430)writeonly buffer O{float d[];}o;
layout(set=0,binding=1,std430)readonly buffer I{vec4 d[];}i;
layout(set=0,binding=2,std430)readonly buffer W{vec4 d[];}w;
layout(set=0,binding=3,std430)readonly buffer B{float d[];}b;
layout(push_constant)uniform P{uint rows;uint inputs;uint outputs;}p;
shared vec4 a[256];shared vec4 q[256];
void main(){uint cb=gl_WorkGroupID.x*32+gl_LocalInvocationID.x*4,rb=gl_WorkGroupID.y*32+gl_LocalInvocationID.y*4,l=gl_LocalInvocationID.y*8+gl_LocalInvocationID.x,iv=p.inputs/4;float s[4][4];for(uint r=0;r<4;r++)for(uint c=0;c<4;c++)s[r][c]=0;for(uint z=0;z<iv;z+=8){for(uint n=l;n<256;n+=64){uint t=n/8,k=z+n%8,r=gl_WorkGroupID.y*32+t;a[n]=r<p.rows&&k<iv?i.d[r*iv+k]:vec4(0);}for(uint n=l;n<256;n+=64){uint t=n/8,k=z+n%8,c=gl_WorkGroupID.x*32+t;q[n]=c<p.outputs&&k<iv?w.d[c*iv+k]:vec4(0);}barrier();for(uint k=0;k<min(8,iv-z);k++){vec4 x[4],y[4];for(uint r=0;r<4;r++)x[r]=a[(gl_LocalInvocationID.y*4+r)*8+k];for(uint c=0;c<4;c++)y[c]=q[(gl_LocalInvocationID.x*4+c)*8+k];for(uint r=0;r<4;r++)for(uint c=0;c<4;c++)s[r][c]+=dot(x[r],y[c]);}barrier();}for(uint r=0;r<4;r++){uint rr=rb+r;if(rr>=p.rows)continue;for(uint c=0;c<4;c++){uint cc=cb+c;if(cc<p.outputs)o.d[rr*p.outputs+cc]=s[r][c]+b.d[cc];}}}
