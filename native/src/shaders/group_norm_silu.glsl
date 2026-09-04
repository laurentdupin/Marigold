#version 450 core
layout(local_size_x = 256) in;
layout(set=0,binding=0,std430) writeonly buffer Output{float data[];}output_buffer;
layout(set=0,binding=1,std430) readonly buffer Input{float data[];}input_buffer;
layout(set=0,binding=2,std430) readonly buffer Scale{float data[];}scale;
layout(set=0,binding=3,std430) readonly buffer Bias{float data[];}bias;
layout(push_constant) uniform Parameters{uint channels;uint spatial;uint groups;float epsilon;}p;
shared float sums[256];shared float squares[256];
void main(){
 uint lane=gl_LocalInvocationID.x,group=gl_WorkGroupID.x;
 uint channels_per_group=p.channels/p.groups,count=channels_per_group*p.spatial,base=group*count;
 float sum=0,square=0;for(uint i=lane;i<count;i+=256){float v=input_buffer.data[base+i];sum+=v;square+=v*v;}
 sums[lane]=sum;squares[lane]=square;barrier();
 for(uint stride=128;stride>0;stride>>=1){if(lane<stride){sums[lane]+=sums[lane+stride];squares[lane]+=squares[lane+stride];}barrier();}
 float mean=sums[0]/float(count);
 float inverse=inversesqrt(squares[0]/float(count)-mean*mean+p.epsilon);
 for(uint i=lane;i<count;i+=256){uint channel=group*channels_per_group+i/p.spatial;
  float v=(input_buffer.data[base+i]-mean)*inverse*scale.data[channel]+bias.data[channel];
  output_buffer.data[base+i]=v/(1.0+exp(-v));}
}
