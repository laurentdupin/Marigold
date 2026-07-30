#version 450 core
layout(local_size_x=8,local_size_y=8,local_size_z=1)in;
layout(set=0,binding=0,std430)writeonly buffer O{float d[];}o;
layout(set=0,binding=1,std430)readonly buffer I{float d[];}i;
layout(set=0,binding=2,std430)readonly buffer W{float d[];}w;
layout(set=0,binding=3,std430)readonly buffer B{float d[];}b;
layout(push_constant)uniform P{uint iw;uint ih;uint ic;uint ow;uint oh;uint oc;uint kernel;uint stride;int padding;uint has_bias;uint batches;uint blocks;}p;
shared float st[289];shared float kt[72];
void main(){
 uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y;
 uint block=gl_GlobalInvocationID.z%p.blocks,batch=gl_GlobalInvocationID.z/p.blocks,cb=block*8;
 bool valid=x<p.ow&&y<p.oh&&cb<p.oc&&batch<p.batches;
 float s[8]=float[8](0,0,0,0,0,0,0,0);uint lane=gl_LocalInvocationID.y*8+gl_LocalInvocationID.x;
 int ox=int(gl_WorkGroupID.x*16)-p.padding,oy=int(gl_WorkGroupID.y*16)-p.padding;
 uint ib=batch*p.ic*p.ih*p.iw,ob=batch*p.oc*p.oh*p.ow;
 for(uint c=0;c<p.ic;++c){
  for(uint n=lane;n<289;n+=64){int px=ox+int(n%17),py=oy+int(n/17);
   st[n]=px>=0&&px<int(p.iw)&&py>=0&&py<int(p.ih)?i.d[ib+(c*p.ih+uint(py))*p.iw+uint(px)]:0;}
  for(uint n=lane;n<72;n+=64){uint off=n/9,ch=cb+off;
   kt[n]=ch<p.oc?w.d[(ch*p.ic+c)*9+n%9]:0;}
  barrier();if(valid)for(uint ky=0;ky<3;++ky)for(uint kx=0;kx<3;++kx){
   float v=st[(gl_LocalInvocationID.y*2+ky)*17+gl_LocalInvocationID.x*2+kx];uint k=ky*3+kx;
   for(uint off=0;off<8;++off)s[off]+=v*kt[off*9+k];}barrier();
 }
 if(!valid)return;for(uint off=0;off<8;++off){uint ch=cb+off;if(ch<p.oc)
  o.d[ob+(ch*p.oh+y)*p.ow+x]=s[off]+(p.has_bias!=0?b.d[ch]:0);}
}
