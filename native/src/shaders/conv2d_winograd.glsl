#version 450 core
layout(local_size_x=8,local_size_y=8,local_size_z=1) in;
layout(set=0,binding=0,std430) writeonly buffer O{float d[];}o;
layout(set=0,binding=1,std430) readonly buffer I{float d[];}i;
layout(set=0,binding=2,std430) readonly buffer W{float d[];}w;
layout(set=0,binding=3,std430) readonly buffer B{float d[];}b;
layout(push_constant) uniform P{uint iw;uint ih;uint ic;uint ow;uint oh;uint oc;
 uint kernel;uint stride;int padding;uint has_bias;uint batches;uint blocks;}p;
void main(){
 uint tx=gl_GlobalInvocationID.x,ty=gl_GlobalInvocationID.y;
 uint block=gl_GlobalInvocationID.z%p.blocks,batch=gl_GlobalInvocationID.z/p.blocks,cb=block*4;
 uint ox=tx*2,oy=ty*2;if(ox>=p.ow||oy>=p.oh||cb>=p.oc||batch>=p.batches)return;
 float m[4][16];for(uint q=0;q<4;++q)for(uint n=0;n<16;++n)m[q][n]=0;
 uint ib=batch*p.ic*p.ih*p.iw,ob=batch*p.oc*p.oh*p.ow;
 for(uint c=0;c<p.ic;++c){
  float d[4][4];for(uint y=0;y<4;++y)for(uint x=0;x<4;++x){
   int px=int(ox+x)-p.padding,py=int(oy+y)-p.padding;
   d[y][x]=px>=0&&px<int(p.iw)&&py>=0&&py<int(p.ih)?i.d[ib+(c*p.ih+uint(py))*p.iw+uint(px)]:0;}
  float t[4][4],v[4][4];for(uint r=0;r<4;++r){
   t[r][0]=d[r][0]-d[r][2];t[r][1]=d[r][1]+d[r][2];
   t[r][2]=-d[r][1]+d[r][2];t[r][3]=d[r][1]-d[r][3];}
  for(uint x=0;x<4;++x){v[0][x]=t[0][x]-t[2][x];v[1][x]=t[1][x]+t[2][x];
   v[2][x]=-t[1][x]+t[2][x];v[3][x]=t[1][x]-t[3][x];}
  for(uint q=0;q<4;++q){uint ch=cb+q;if(ch>=p.oc)continue;uint wb=(ch*p.ic+c)*16;
   for(uint y=0;y<4;++y)for(uint x=0;x<4;++x)m[q][y*4+x]+=v[y][x]*w.d[wb+y*4+x];}
 }
 for(uint q=0;q<4;++q){uint ch=cb+q;if(ch>=p.oc)continue;float a[2][4];
  for(uint x=0;x<4;++x){a[0][x]=m[q][x]+m[q][4+x]+m[q][8+x];
   a[1][x]=m[q][4+x]-m[q][8+x]-m[q][12+x];}
  float bias=p.has_bias!=0?b.d[ch]:0;
  float y00=a[0][0]+a[0][1]+a[0][2]+bias;float y01=a[0][1]-a[0][2]-a[0][3]+bias;
  float y10=a[1][0]+a[1][1]+a[1][2]+bias;float y11=a[1][1]-a[1][2]-a[1][3]+bias;
  o.d[ob+(ch*p.oh+oy)*p.ow+ox]=y00;
  if(ox+1<p.ow)o.d[ob+(ch*p.oh+oy)*p.ow+ox+1]=y01;
  if(oy+1<p.oh){o.d[ob+(ch*p.oh+oy+1)*p.ow+ox]=y10;
   if(ox+1<p.ow)o.d[ob+(ch*p.oh+oy+1)*p.ow+ox+1]=y11;}}
}
