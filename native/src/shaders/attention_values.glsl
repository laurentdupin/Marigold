#version 450 core
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float data[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Scores { float data[]; } scores;
layout(set = 0, binding = 2, std430) readonly buffer Value { float data[]; } value_buffer;
layout(push_constant) uniform Parameters {
    uint queries; uint keys; uint heads; uint head_dimensions;
} p;
void main() {
    uint channel = gl_GlobalInvocationID.x;
    uint row = gl_WorkGroupID.y;
    uint query = row / p.heads;
    uint head = row % p.heads;
    if (query >= p.queries || channel >= p.head_dimensions) return;
    uint dimensions = p.heads * p.head_dimensions;
    float sum = 0.0;
    for (uint key = 0; key < p.keys; ++key) {
        sum += scores.data[(head * p.queries + query) * p.keys + key] *
            value_buffer.data[
                key * dimensions + head * p.head_dimensions + channel];
    }
    output_buffer.data[
        query * dimensions + head * p.head_dimensions + channel] = sum;
}
