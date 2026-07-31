#version 450 core
layout(local_size_x=16,local_size_y=8,local_size_z=1) in;
layout(set=0,binding=0,std430) writeonly buffer O{float d[];}o;
layout(set=0,binding=1,std430) readonly buffer I{vec4 d[];}i;
layout(set=0,binding=2,std430) readonly buffer W{uint d[];}w;
layout(set=0,binding=3,std430) readonly buffer B{float d[];}b;
layout(push_constant) uniform P{uint rows;uint inputs;uint outputs;}p;
#define K 8
#define KS 9
shared vec4 it[40*KS];
shared vec4 wt[64*KS];
vec4 rw4(uint n){return vec4(
 unpackHalf2x16(w.d[n>>1]),unpackHalf2x16(w.d[(n>>1)+1]));}
void main(){
 uint cb=gl_WorkGroupID.x*64+gl_LocalInvocationID.x;
 uint rb=gl_WorkGroupID.y*40+gl_LocalInvocationID.y*5;
 float s[5][4];
 for(uint r=0;r<5;++r)for(uint c=0;c<4;++c)s[r][c]=0.0;
 uint lane=gl_LocalInvocationID.y*16+gl_LocalInvocationID.x;
 uint iv=p.inputs/4;
 for(uint base=0;base<iv;base+=K){
  for(uint n=lane;n<40*K;n+=128){
   uint tr=n/K,inner=base+n%K,row=gl_WorkGroupID.y*40+tr;
   it[tr*KS+n%K]=row<p.rows&&inner<iv?i.d[row*iv+inner]:vec4(0.0);
  }
  for(uint n=lane;n<64*K;n+=128){
   uint tc=n/K,inner=base+n%K,col=gl_WorkGroupID.x*64+tc;
   wt[tc*KS+n%K]=col<p.outputs&&inner<iv
    ?rw4((inner*p.outputs+col)*4):vec4(0.0);
  }
  barrier();uint count=min(K,iv-base);
  for(uint inner=0;inner<count;++inner){vec4 a[5],q[4];
   for(uint r=0;r<5;++r)a[r]=it[(gl_LocalInvocationID.y*5+r)*KS+inner];
   for(uint c=0;c<4;++c)q[c]=wt[(gl_LocalInvocationID.x+c*16)*KS+inner];
   for(uint r=0;r<5;++r)for(uint c=0;c<4;++c)s[r][c]+=dot(a[r],q[c]);}
  barrier();}
 for(uint r=0;r<5;++r){uint row=rb+r;if(row>=p.rows)continue;
  for(uint c=0;c<4;++c){uint col=cb+c*16;if(col<p.outputs)
   o.d[row*p.outputs+col]=s[r][c]+b.d[col];}}
}
