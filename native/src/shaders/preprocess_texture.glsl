#version 450

layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0) uniform sampler2D source_image;
layout(std430, binding = 1) writeonly buffer Output { float values[]; } output_data;
layout(push_constant) uniform Parameters {
    uint source_width;
    uint source_height;
    uint target_width;
    uint target_height;
} parameters;

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= parameters.target_width || y >= parameters.target_height) return;
    float sx = float(parameters.source_width) / float(parameters.target_width);
    float sy = float(parameters.source_height) / float(parameters.target_height);
    float support_x = max(1.0, sx);
    float support_y = max(1.0, sy);
    float inverse_x = sx >= 1.0 ? 1.0 / sx : 1.0;
    float inverse_y = sy >= 1.0 ? 1.0 / sy : 1.0;
    float center_x = sx * (float(x) + 0.5);
    float center_y = sy * (float(y) + 0.5);
    int begin_x = max(int(center_x - support_x + 0.5), 0);
    int end_x = min(int(center_x + support_x + 0.5), int(parameters.source_width));
    int begin_y = max(int(center_y - support_y + 0.5), 0);
    int end_y = min(int(center_y + support_y + 0.5), int(parameters.source_height));
    vec3 sum = vec3(0.0);
    float total = 0.0;
    for (int iy = begin_y; iy < end_y; ++iy) {
        float wy = max(0.0, 1.0 - abs((float(iy) - center_y + 0.5) * inverse_y));
        for (int ix = begin_x; ix < end_x; ++ix) {
            float wx = max(0.0, 1.0 - abs((float(ix) - center_x + 0.5) * inverse_x));
            float weight = wx * wy;
            sum += texelFetch(source_image, ivec2(ix, iy), 0).bgr * weight;
            total += weight;
        }
    }
    vec3 rgb = total > 0.0 ? sum / total : vec3(0.0);
    uint spatial = parameters.target_width * parameters.target_height;
    uint index = y * parameters.target_width + x;
    output_data.values[index] = rgb.r * 2.0 - 1.0;
    output_data.values[spatial + index] = rgb.g * 2.0 - 1.0;
    output_data.values[2 * spatial + index] = rgb.b * 2.0 - 1.0;
}
