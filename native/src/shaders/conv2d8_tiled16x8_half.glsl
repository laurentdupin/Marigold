#version 450 core
layout(local_size_x=16,local_size_y=8,local_size_z=1) in;
layout(set=0,binding=0,std430) writeonly buffer O{float d[];}o;
layout(set=0,binding=1,std430) readonly buffer I{float d[];}i;
layout(set=0,binding=2,std430) readonly buffer W{uint d[];}w;
layout(set=0,binding=3,std430) readonly buffer B{float d[];}b;
layout(push_constant) uniform P{uint iw;uint ih;uint ic;uint ow;uint oh;uint oc;
 uint kernel;uint stride;int padding;uint has_bias;uint batches;uint blocks;}p;
shared float st[1440];shared vec4 kt[144];
vec4 rw4(uint n){vec2 a=unpackHalf2x16(w.d[n>>1]);vec2 z=unpackHalf2x16(w.d[(n>>1)+1]);return vec4(a,z);}
float rw(uint n){vec2 v=unpackHalf2x16(w.d[n>>1]);return (n&1)==0?v.x:v.y;}
void main(){
 uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y;
 uint block=gl_GlobalInvocationID.z%p.blocks,batch=gl_GlobalInvocationID.z/p.blocks,cb=block*8;
 bool valid=x<p.ow&&y<p.oh&&cb<p.oc&&batch<p.batches;
 vec4 s0=vec4(0),s1=vec4(0);uint lane=gl_LocalInvocationID.y*16+gl_LocalInvocationID.x;
 int ox=int(gl_WorkGroupID.x*16)-1,oy=int(gl_WorkGroupID.y*8)-1;
 uint ib=batch*p.ic*p.ih*p.iw,ob=batch*p.oc*p.oh*p.ow;
 for(uint cb8=0;cb8<p.ic;cb8+=8){
  for(uint n=lane;n<1440;n+=128){uint co=n/180,ti=n%180,c=cb8+co;
   int px=ox+int(ti%18),py=oy+int(ti/18);
   st[n]=c<p.ic&&px>=0&&px<int(p.iw)&&py>=0&&py<int(p.ih)
    ?i.d[ib+(c*p.ih+uint(py))*p.iw+uint(px)]:0;}
  for(uint n=lane;n<144;n+=128){uint co=n/18,q=n%18,pair=q/9,k=q%9,c=cb8+co;
   uint ch=cb+pair*4,wi=(c*9+k)*p.oc+ch;
   kt[n]=c>=p.ic?vec4(0):(ch+3<p.oc?rw4(wi):vec4(
    ch<p.oc?rw(wi):0,ch+1<p.oc?rw(wi+1):0,
    ch+2<p.oc?rw(wi+2):0,ch+3<p.oc?rw(wi+3):0));}
  barrier();if(valid)for(uint co=0;co<8&&cb8+co<p.ic;++co)
   for(uint ky=0;ky<3;++ky)for(uint kx=0;kx<3;++kx){
    float v=st[co*180+(gl_LocalInvocationID.y+ky)*18+gl_LocalInvocationID.x+kx];
    uint k=ky*3+kx;s0+=v*kt[co*18+k];s1+=v*kt[co*18+9+k];}
  barrier();}
 if(!valid)return;
 for(uint off=0;off<4;++off){uint ch=cb+off;if(ch<p.oc)o.d[ob+(ch*p.oh+y)*p.ow+x]=s0[off]+(p.has_bias!=0?b.d[ch]:0);}
 for(uint off=0;off<4;++off){uint ch=cb+4+off;if(ch<p.oc)o.d[ob+(ch*p.oh+y)*p.ow+x]=s1[off]+(p.has_bias!=0?b.d[ch]:0);}
}
