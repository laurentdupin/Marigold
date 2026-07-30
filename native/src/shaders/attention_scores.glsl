#version 450 core
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std430) writeonly buffer Scores { float data[]; } scores;
layout(set = 0, binding = 1, std430) readonly buffer Query { float data[]; } query_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Key { float data[]; } key_buffer;
layout(push_constant) uniform Parameters {
    uint queries; uint keys; uint heads; uint head_dimensions;
} p;
void main() {
    uint key = gl_GlobalInvocationID.x;
    uint row = gl_WorkGroupID.y;
    uint query = row / p.heads;
    uint head = row % p.heads;
    if (query >= p.queries || key >= p.keys) return;
    uint dimensions = p.heads * p.head_dimensions;
    float sum = 0.0;
    for (uint d = 0; d < p.head_dimensions; ++d) {
        sum += query_buffer.data[
                query * dimensions + head * p.head_dimensions + d] *
            key_buffer.data[
                key * dimensions + head * p.head_dimensions + d];
    }
    scores.data[(head * p.queries + query) * p.keys + key] =
        sum * inversesqrt(float(p.head_dimensions));
}
