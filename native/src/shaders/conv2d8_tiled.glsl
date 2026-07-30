#version 450 core
layout(local_size_x=8,local_size_y=8,local_size_z=1)in;
layout(set=0,binding=0,std430)writeonly buffer O{float d[];}o;
layout(set=0,binding=1,std430)readonly buffer I{float d[];}i;
layout(set=0,binding=2,std430)readonly buffer W{float d[];}w;
layout(set=0,binding=3,std430)readonly buffer B{float d[];}b;
layout(push_constant)uniform P{uint iw;uint ih;uint ic;uint ow;uint oh;uint oc;uint kernel;uint stride;int padding;uint has_bias;uint batches;uint blocks;}p;
shared float st[100];shared float kt[72];
void main(){uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y,bl=gl_GlobalInvocationID.z%p.blocks,ba=gl_GlobalInvocationID.z/p.blocks,cb=bl*8,l=gl_LocalInvocationID.y*8+gl_LocalInvocationID.x;bool v=x<p.ow&&y<p.oh&&cb<p.oc&&ba<p.batches;float s[8]=float[8](0,0,0,0,0,0,0,0);int ox=int(gl_WorkGroupID.x*8)-1,oy=int(gl_WorkGroupID.y*8)-1;uint ib=ba*p.ic*p.ih*p.iw,ob=ba*p.oc*p.oh*p.ow;for(uint c=0;c<p.ic;c++){for(uint n=l;n<100;n+=64){int px=ox+int(n%10),py=oy+int(n/10);st[n]=px>=0&&px<int(p.iw)&&py>=0&&py<int(p.ih)?i.d[ib+(c*p.ih+uint(py))*p.iw+uint(px)]:0;}for(uint n=l;n<72;n+=64){uint f=n/9,ch=cb+f;kt[n]=ch<p.oc?w.d[(ch*p.ic+c)*9+n%9]:0;}barrier();if(v)for(uint ky=0;ky<3;ky++)for(uint kx=0;kx<3;kx++){float z=st[(gl_LocalInvocationID.y+ky)*10+gl_LocalInvocationID.x+kx];uint k=ky*3+kx;for(uint f=0;f<8;f++)s[f]+=z*kt[f*9+k];}barrier();}if(!v)return;for(uint f=0;f<8;f++){uint ch=cb+f;if(ch<p.oc)o.d[ob+(ch*p.oh+y)*p.ow+x]=s[f]+(p.has_bias!=0?b.d[ch]:0);}}
