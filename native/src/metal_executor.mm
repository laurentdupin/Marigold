#include "metal_executor.h"
#include "inferbridge/native_harness_diffusion_shape.h"
#include "inferbridge/native_harness_metal_texture.h"
#include "inferbridge/native_harness_precision.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace marigold_native {
namespace {

MPSShape* shape(std::initializer_list<NSInteger> values) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:values.size()];
    for (NSInteger value : values) [result addObject:@(value)];
    return result;
}

MPSShape* shape(const TensorView& tensor) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:tensor.rank];
    for (std::uint32_t i = 0; i < tensor.rank; ++i)
        [result addObject:@(tensor.dimensions[i])];
    return result;
}

NSString* ns(const std::string& value) {
    return [NSString stringWithUTF8String:value.c_str()];
}

int dimension(MPSGraphTensor* value, int index) {
    return [value.shape[index] intValue];
}

class GraphBuilder {
public:
    GraphBuilder(
        const ModelBundle& model,
        const TokenTensor& prompt,
        bool fp16,
        bool full_v1,
        int width,
        int height)
        : unet_(model.unet()), vae_(model.vae()), prompt_(prompt),
          fp16_(fp16), full_v1_(full_v1), width_(width), height_(height),
          graph_([MPSGraph new]) {}

    void build() {
        const int latent_width = width_ / 8;
        const int latent_height = height_ / 8;
        rgb_ = [graph_ placeholderWithShape:shape({1, 3, height_, width_})
                                      dataType:MPSDataTypeFloat32 name:@"rgb"];
        target_noise_ = [graph_ placeholderWithShape:
            shape({1, 4, latent_height, latent_width})
            dataType:MPSDataTypeFloat32 name:@"target_noise"];

        MPSGraphTensor* posterior = vae_encode(internal(rgb_));
        MPSGraphTensor* latent = [graph_ sliceTensor:posterior dimension:1
            start:0 length:4 name:nil];
        latent = multiply(latent, scalar(0.18215f));
        MPSGraphTensor* target = internal(target_noise_);
        if (!full_v1_) {
            MPSGraphTensor* sample = [graph_ concatTensors:@[latent, target]
                dimension:1 name:nil];
            MPSGraphTensor* prediction = unet(sample, 999);
            constexpr float alpha = 0.00466009508818388f;
            constexpr float c_skip = 2.505007534736592e-9f;
            MPSGraphTensor* original = add(
                multiply(target, scalar(std::sqrt(alpha))),
                multiply(prediction, scalar(-std::sqrt(1.0f - alpha))));
            target = add(multiply(original, scalar(1.0f)),
                multiply(target, scalar(c_skip)));
        } else {
            constexpr std::array<int, 10> timesteps = {
                901, 801, 701, 601, 501, 401, 301, 201, 101, 1};
            constexpr std::array<float, 11> alpha_products = {
                0.014004888944327831f, 0.0365464948117733f,
                0.08191665261983871f, 0.15981632471084595f,
                0.27499884366989136f, 0.4228813052177429f,
                0.5888184309005737f, 0.7521430850028992f,
                0.8929802179336548f, 0.9982960224151611f,
                0.9991499781608582f};
            for (std::size_t step = 0; step < timesteps.size(); ++step) {
                MPSGraphTensor* sample = [graph_ concatTensors:@[latent, target]
                    dimension:1 name:nil];
                MPSGraphTensor* prediction = unet(sample, timesteps[step]);
                const float alpha = alpha_products[step];
                const float previous = alpha_products[step + 1];
                MPSGraphTensor* original = add(
                    multiply(target, scalar(std::sqrt(alpha))),
                    multiply(prediction, scalar(-std::sqrt(1.0f - alpha))));
                MPSGraphTensor* epsilon = add(
                    multiply(prediction, scalar(std::sqrt(alpha))),
                    multiply(target, scalar(std::sqrt(1.0f - alpha))));
                target = add(
                    multiply(original, scalar(std::sqrt(previous))),
                    multiply(epsilon, scalar(std::sqrt(1.0f - previous))));
            }
        }
        target = multiply(target, scalar(1.0f / 0.18215f));
        decoded_ = external(vae_decode(target));
    }

    MPSGraph* graph() const { return graph_; }
    MPSGraphTensor* rgb() const { return rgb_; }
    MPSGraphTensor* target_noise() const { return target_noise_; }
    MPSGraphTensor* decoded() const { return decoded_; }

private:
    MPSGraphTensor* internal(MPSGraphTensor* value) {
        return fp16_ && value.dataType != MPSDataTypeFloat16
            ? [graph_ castTensor:value toType:MPSDataTypeFloat16 name:nil]
            : value;
    }

    MPSGraphTensor* external(MPSGraphTensor* value) {
        return value.dataType == MPSDataTypeFloat32 ? value
            : [graph_ castTensor:value toType:MPSDataTypeFloat32 name:nil];
    }

    MPSGraphTensor* constant(
        const SafeTensors& model, const std::string& name) {
        const std::string key = (&model == &unet_ ? "unet:" : "vae:") + name;
        const auto cached = constant_tensors_.find(key);
        if (cached != constant_tensors_.end()) return cached->second;
        const TensorView& tensor = model.tensor(name);
        MPSGraphTensor* result = nil;
        if (fp16_) {
            const void* bytes = tensor.data.bytes;
            std::size_t byte_count =
                static_cast<std::size_t>(tensor.elements) * sizeof(std::uint16_t);
            if (tensor.data.dtype == TensorDType::f32) {
                auto& cache = &model == &unet_ ? unet_half_ : vae_half_;
                auto found = cache.find(name);
                if (found == cache.end()) {
                    std::vector<std::uint16_t> values(
                        static_cast<std::size_t>(tensor.elements));
                    for (std::uint64_t i = 0; i < tensor.elements; ++i)
                        values[static_cast<std::size_t>(i)] =
                            inferbridge::native::float_to_half(tensor.data[i]);
                    found = cache.emplace(name, std::move(values)).first;
                }
                bytes = found->second.data();
                byte_count = found->second.size() * sizeof(std::uint16_t);
            }
            NSData* half_data = [NSData dataWithBytesNoCopy:
                const_cast<void*>(bytes) length:byte_count
                freeWhenDone:NO];
            result = [graph_ constantWithData:half_data
                shape:shape(tensor) dataType:MPSDataTypeFloat16];
        } else {
            const void* bytes = tensor.data.bytes;
            std::size_t byte_count =
                static_cast<std::size_t>(tensor.elements) * sizeof(float);
            if (tensor.data.dtype == TensorDType::f16) {
                auto& cache = &model == &unet_ ? unet_float_ : vae_float_;
                auto found = cache.find(name);
                if (found == cache.end()) {
                    std::vector<float> values(
                        static_cast<std::size_t>(tensor.elements));
                    for (std::uint64_t i = 0; i < tensor.elements; ++i)
                        values[static_cast<std::size_t>(i)] = tensor.data[i];
                    found = cache.emplace(name, std::move(values)).first;
                }
                bytes = found->second.data();
                byte_count = found->second.size() * sizeof(float);
            }
            NSData* data = [NSData dataWithBytesNoCopy:
                const_cast<void*>(bytes) length:byte_count freeWhenDone:NO];
            result = [graph_ constantWithData:data
                shape:shape(tensor) dataType:MPSDataTypeFloat32];
        }
        constant_tensors_.emplace(key, result);
        return result;
    }

    MPSGraphTensor* prompt_constant() {
        const auto cached = constant_tensors_.find("prompt");
        if (cached != constant_tensors_.end()) return cached->second;
        MPSGraphTensor* result = nil;
        if (fp16_) {
            if (prompt_half_.empty())
                prompt_half_ = inferbridge::native::pack_fp16(
                    prompt_.values.data(), prompt_.values.size());
            NSData* data = [NSData dataWithBytesNoCopy:prompt_half_.data()
                length:prompt_half_.size() * sizeof(std::uint16_t)
                freeWhenDone:NO];
            result = [graph_ constantWithData:data
                shape:shape({1, static_cast<NSInteger>(prompt_.tokens),
                             static_cast<NSInteger>(prompt_.dimensions)})
                dataType:MPSDataTypeFloat16];
        } else {
            NSData* data = [NSData dataWithBytesNoCopy:
                const_cast<float*>(prompt_.values.data())
                length:prompt_.values.size() * sizeof(float) freeWhenDone:NO];
            result = [graph_ constantWithData:data
                shape:shape({1, static_cast<NSInteger>(prompt_.tokens),
                             static_cast<NSInteger>(prompt_.dimensions)})
                dataType:MPSDataTypeFloat32];
        }
        constant_tensors_.emplace("prompt", result);
        return result;
    }

    MPSGraphTensor* scalar(float value) {
        return internal([graph_ constantWithScalar:value
            dataType:MPSDataTypeFloat32]);
    }

    MPSGraphTensor* add(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }

    MPSGraphTensor* multiply(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ multiplicationWithPrimaryTensor:a secondaryTensor:b
            name:nil];
    }

    MPSGraphConvolution2DOpDescriptor* descriptor(
        int stride, int before, int after) {
        return [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride strideInY:stride
            dilationRateInX:1 dilationRateInY:1 groups:1
            paddingLeft:before paddingRight:after
            paddingTop:before paddingBottom:after
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    }

    MPSGraphTensor* conv(
        const SafeTensors& model,
        MPSGraphTensor* value,
        const std::string& prefix,
        int stride = 1,
        int before = 1,
        int after = 1) {
        MPSGraphTensor* result = [graph_ convolution2DWithSourceTensor:value
            weightsTensor:constant(model, prefix + ".weight")
            descriptor:descriptor(stride, before, after) name:ns(prefix)];
        if (model.contains(prefix + ".bias")) {
            const int channels = static_cast<int>(
                model.tensor(prefix + ".bias").elements);
            result = add(result, [graph_ reshapeTensor:
                constant(model, prefix + ".bias")
                withShape:shape({1, channels, 1, 1}) name:nil]);
        }
        return result;
    }

    MPSGraphTensor* linear(
        const SafeTensors& model,
        MPSGraphTensor* value,
        const std::string& prefix) {
        MPSGraphTensor* weight = [graph_ transposeTensor:
            constant(model, prefix + ".weight")
            dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* result = [graph_
            matrixMultiplicationWithPrimaryTensor:value
            secondaryTensor:weight name:ns(prefix)];
        if (model.contains(prefix + ".bias"))
            result = add(result, constant(model, prefix + ".bias"));
        return result;
    }

    MPSGraphTensor* silu(MPSGraphTensor* value) {
        return multiply(value, [graph_ sigmoidWithTensor:value name:nil]);
    }

    MPSGraphTensor* gelu(MPSGraphTensor* value) {
        MPSGraphTensor* error = [graph_ erfWithTensor:
            multiply(value, scalar(0.7071067811865475f)) name:nil];
        return multiply(multiply(value, scalar(0.5f)),
                        add(error, scalar(1.0f)));
    }

    MPSGraphTensor* group_norm(
        const SafeTensors& model,
        MPSGraphTensor* value,
        const std::string& prefix,
        float epsilon) {
        const int channels = dimension(value, 1);
        const int height = dimension(value, 2);
        const int width = dimension(value, 3);
        MPSGraphTensor* grouped = [graph_ reshapeTensor:value
            withShape:shape({1, 32, channels / 32, height, width}) name:nil];
        NSArray<NSNumber*>* axes = @[@2, @3, @4];
        MPSGraphTensor* mean = [graph_ meanOfTensor:grouped axes:axes name:nil];
        MPSGraphTensor* variance = [graph_ varianceOfTensor:grouped
            meanTensor:mean axes:axes name:nil];
        MPSGraphTensor* normalized = [graph_ normalizationWithTensor:grouped
            meanTensor:mean varianceTensor:variance gammaTensor:nil
            betaTensor:nil epsilon:epsilon name:ns(prefix)];
        normalized = [graph_ reshapeTensor:normalized
            withShape:shape({1, channels, height, width}) name:nil];
        MPSGraphTensor* gamma = [graph_ reshapeTensor:
            constant(model, prefix + ".weight")
            withShape:shape({1, channels, 1, 1}) name:nil];
        MPSGraphTensor* beta = [graph_ reshapeTensor:
            constant(model, prefix + ".bias")
            withShape:shape({1, channels, 1, 1}) name:nil];
        return add(multiply(normalized, gamma), beta);
    }

    MPSGraphTensor* layer_norm(
        const SafeTensors& model,
        MPSGraphTensor* value,
        const std::string& prefix) {
        NSArray<NSNumber*>* axes = @[@(-1)];
        MPSGraphTensor* mean = [graph_ meanOfTensor:value axes:axes name:nil];
        MPSGraphTensor* variance = [graph_ varianceOfTensor:value
            meanTensor:mean axes:axes name:nil];
        return [graph_ normalizationWithTensor:value meanTensor:mean
            varianceTensor:variance gammaTensor:constant(model, prefix + ".weight")
            betaTensor:constant(model, prefix + ".bias") epsilon:1.0e-5f
            name:ns(prefix)];
    }

    MPSGraphTensor* nearest(
        MPSGraphTensor* value, int target_height, int target_width) {
        const int source_height = dimension(value, 2);
        const int source_width = dimension(value, 3);
        std::vector<std::int32_t> y(static_cast<std::size_t>(target_height));
        std::vector<std::int32_t> x(static_cast<std::size_t>(target_width));
        for (int i = 0; i < target_height; ++i)
            y[static_cast<std::size_t>(i)] =
                static_cast<std::int32_t>(
                    std::int64_t(i) * source_height / target_height);
        for (int i = 0; i < target_width; ++i)
            x[static_cast<std::size_t>(i)] =
                static_cast<std::int32_t>(
                    std::int64_t(i) * source_width / target_width);
        NSData* y_data = [NSData dataWithBytes:y.data()
            length:y.size() * sizeof(std::int32_t)];
        NSData* x_data = [NSData dataWithBytes:x.data()
            length:x.size() * sizeof(std::int32_t)];
        MPSGraphTensor* y_indices = [graph_ constantWithData:y_data
            shape:shape({target_height}) dataType:MPSDataTypeInt32];
        MPSGraphTensor* x_indices = [graph_ constantWithData:x_data
            shape:shape({target_width}) dataType:MPSDataTypeInt32];
        value = [graph_ gatherWithUpdatesTensor:value indicesTensor:y_indices
            axis:2 batchDimensions:0 name:nil];
        return [graph_ gatherWithUpdatesTensor:value indicesTensor:x_indices
            axis:3 batchDimensions:0 name:nil];
    }

    MPSGraphTensor* resnet(
        const SafeTensors& model,
        MPSGraphTensor* input,
        const std::string& prefix) {
        MPSGraphTensor* hidden = silu(group_norm(
            model, input, prefix + ".norm1", 1.0e-6f));
        hidden = conv(model, hidden, prefix + ".conv1");
        hidden = silu(group_norm(
            model, hidden, prefix + ".norm2", 1.0e-6f));
        hidden = conv(model, hidden, prefix + ".conv2");
        MPSGraphTensor* residual = input;
        if (model.contains(prefix + ".conv_shortcut.weight"))
            residual = conv(model, input, prefix + ".conv_shortcut", 1, 0, 0);
        return add(hidden, residual);
    }

    MPSGraphTensor* spatial_attention(
        MPSGraphTensor* input, const std::string& prefix) {
        const int channels = dimension(input, 1);
        const int height = dimension(input, 2);
        const int width = dimension(input, 3);
        const int tokens = height * width;
        MPSGraphTensor* normalized = group_norm(
            vae_, input, prefix + ".group_norm", 1.0e-6f);
        normalized = [graph_ reshapeTensor:normalized
            withShape:shape({1, channels, tokens}) name:nil];
        normalized = [graph_ transposeTensor:normalized dimension:1
            withDimension:2 name:nil];
        MPSGraphTensor* q = linear(vae_, normalized, prefix + ".to_q");
        MPSGraphTensor* k = linear(vae_, normalized, prefix + ".to_k");
        MPSGraphTensor* v = linear(vae_, normalized, prefix + ".to_v");
        k = [graph_ transposeTensor:k dimension:1 withDimension:2 name:nil];
        MPSGraphTensor* scores = [graph_
            matrixMultiplicationWithPrimaryTensor:q secondaryTensor:k name:nil];
        scores = multiply(scores, scalar(1.0f / std::sqrt(float(channels))));
        scores = [graph_ softMaxWithTensor:scores axis:-1 name:nil];
        MPSGraphTensor* attended = [graph_
            matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v
            name:nil];
        attended = linear(vae_, attended, prefix + ".to_out.0");
        attended = [graph_ transposeTensor:attended dimension:1
            withDimension:2 name:nil];
        attended = [graph_ reshapeTensor:attended
            withShape:shape({1, channels, height, width}) name:nil];
        return add(input, attended);
    }

    MPSGraphTensor* vae_mid(MPSGraphTensor* hidden, const std::string& prefix) {
        hidden = resnet(vae_, hidden, prefix + ".resnets.0");
        hidden = spatial_attention(hidden, prefix + ".attentions.0");
        return resnet(vae_, hidden, prefix + ".resnets.1");
    }

    MPSGraphTensor* vae_encode(MPSGraphTensor* image) {
        MPSGraphTensor* hidden = conv(vae_, image, "encoder.conv_in");
        for (int block = 0; block < 4; ++block) {
            const std::string root = "encoder.down_blocks." +
                std::to_string(block);
            for (int layer = 0; layer < 2; ++layer)
                hidden = resnet(vae_, hidden, root + ".resnets." +
                    std::to_string(layer));
            if (block != 3)
                hidden = conv(vae_, hidden, root + ".downsamplers.0.conv",
                    2, 0, 1);
        }
        hidden = vae_mid(hidden, "encoder.mid_block");
        hidden = silu(group_norm(vae_, hidden,
            "encoder.conv_norm_out", 1.0e-6f));
        hidden = conv(vae_, hidden, "encoder.conv_out");
        return conv(vae_, hidden, "quant_conv", 1, 0, 0);
    }

    MPSGraphTensor* vae_decode(MPSGraphTensor* latent) {
        MPSGraphTensor* hidden = conv(
            vae_, latent, "post_quant_conv", 1, 0, 0);
        hidden = conv(vae_, hidden, "decoder.conv_in");
        hidden = vae_mid(hidden, "decoder.mid_block");
        for (int block = 0; block < 4; ++block) {
            const std::string root = "decoder.up_blocks." +
                std::to_string(block);
            for (int layer = 0; layer < 3; ++layer)
                hidden = resnet(vae_, hidden, root + ".resnets." +
                    std::to_string(layer));
            if (block != 3) {
                hidden = nearest(hidden, dimension(hidden, 2) * 2,
                    dimension(hidden, 3) * 2);
                hidden = conv(vae_, hidden, root + ".upsamplers.0.conv");
            }
        }
        hidden = silu(group_norm(vae_, hidden,
            "decoder.conv_norm_out", 1.0e-6f));
        return conv(vae_, hidden, "decoder.conv_out");
    }

    MPSGraphTensor* attention(
        MPSGraphTensor* query_input,
        MPSGraphTensor* key_value_input,
        const std::string& prefix,
        int heads) {
        const int query_tokens = dimension(query_input, 1);
        const int key_tokens = dimension(key_value_input, 1);
        MPSGraphTensor* q = linear(unet_, query_input, prefix + ".to_q");
        MPSGraphTensor* k = linear(unet_, key_value_input, prefix + ".to_k");
        MPSGraphTensor* v = linear(unet_, key_value_input, prefix + ".to_v");
        const int channels = dimension(q, 2);
        const int head_dimensions = channels / heads;
        q = [graph_ reshapeTensor:q
            withShape:shape({1, query_tokens, heads, head_dimensions}) name:nil];
        k = [graph_ reshapeTensor:k
            withShape:shape({1, key_tokens, heads, head_dimensions}) name:nil];
        v = [graph_ reshapeTensor:v
            withShape:shape({1, key_tokens, heads, head_dimensions}) name:nil];
        q = [graph_ transposeTensor:q permutation:@[@0, @2, @1, @3] name:nil];
        k = [graph_ transposeTensor:k permutation:@[@0, @2, @3, @1] name:nil];
        v = [graph_ transposeTensor:v permutation:@[@0, @2, @1, @3] name:nil];
        MPSGraphTensor* scores = [graph_
            matrixMultiplicationWithPrimaryTensor:q secondaryTensor:k name:nil];
        scores = multiply(scores,
            scalar(1.0f / std::sqrt(float(head_dimensions))));
        scores = [graph_ softMaxWithTensor:scores axis:-1 name:nil];
        MPSGraphTensor* attended = [graph_
            matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v
            name:nil];
        attended = [graph_ transposeTensor:attended
            permutation:@[@0, @2, @1, @3] name:nil];
        attended = [graph_ reshapeTensor:attended
            withShape:shape({1, query_tokens, channels}) name:nil];
        return linear(unet_, attended, prefix + ".to_out.0");
    }

    MPSGraphTensor* feed_forward(
        MPSGraphTensor* input, const std::string& prefix) {
        MPSGraphTensor* projected = linear(
            unet_, input, prefix + ".net.0.proj");
        const int dimensions = dimension(projected, 2) / 2;
        MPSGraphTensor* value = [graph_ sliceTensor:projected dimension:2
            start:0 length:dimensions name:nil];
        MPSGraphTensor* gate = [graph_ sliceTensor:projected dimension:2
            start:dimensions length:dimensions name:nil];
        return linear(unet_, multiply(value, gelu(gate)), prefix + ".net.2");
    }

    MPSGraphTensor* transformer(
        MPSGraphTensor* input,
        const std::string& prefix,
        int heads) {
        const int channels = dimension(input, 1);
        const int height = dimension(input, 2);
        const int width = dimension(input, 3);
        const int tokens = height * width;
        MPSGraphTensor* normalized = group_norm(
            unet_, input, prefix + ".norm", 1.0e-6f);
        normalized = [graph_ reshapeTensor:normalized
            withShape:shape({1, channels, tokens}) name:nil];
        normalized = [graph_ transposeTensor:normalized
            dimension:1 withDimension:2 name:nil];
        MPSGraphTensor* current = linear(unet_, normalized, prefix + ".proj_in");
        const std::string block = prefix + ".transformer_blocks.0";
        MPSGraphTensor* normalized_tokens =
            layer_norm(unet_, current, block + ".norm1");
        current = add(current, attention(normalized_tokens, normalized_tokens,
            block + ".attn1", heads));
        current = add(current, attention(
            layer_norm(unet_, current, block + ".norm2"),
            prompt_constant(), block + ".attn2", heads));
        current = add(current, feed_forward(
            layer_norm(unet_, current, block + ".norm3"), block + ".ff"));
        current = linear(unet_, current, prefix + ".proj_out");
        current = [graph_ transposeTensor:current
            dimension:1 withDimension:2 name:nil];
        current = [graph_ reshapeTensor:current
            withShape:shape({1, channels, height, width}) name:nil];
        return add(input, current);
    }

    MPSGraphTensor* embedding_mlp(
        MPSGraphTensor* input, const std::string& prefix) {
        return linear(unet_, silu(linear(
            unet_, input, prefix + ".linear_1")), prefix + ".linear_2");
    }

    MPSGraphTensor* time_embedding(int timestep_value) {
        std::vector<float> values(320);
        for (int i = 0; i < 160; ++i) {
            const float frequency = std::exp(
                -std::log(10000.0f) * static_cast<float>(i) / 160.0f);
            const float argument = static_cast<float>(timestep_value) * frequency;
            values[i] = std::cos(argument);
            values[160 + i] = std::sin(argument);
        }
        NSData* data = [NSData dataWithBytes:values.data()
            length:values.size() * sizeof(float)];
        MPSGraphTensor* timestep = internal([graph_ constantWithData:data
            shape:shape({1, 320}) dataType:MPSDataTypeFloat32]);
        return embedding_mlp(timestep, "time_embedding");
    }

    MPSGraphTensor* unet_resnet(
        MPSGraphTensor* input,
        MPSGraphTensor* time,
        const std::string& prefix) {
        MPSGraphTensor* hidden = silu(group_norm(
            unet_, input, prefix + ".norm1", 1.0e-5f));
        hidden = conv(unet_, hidden, prefix + ".conv1");
        MPSGraphTensor* projected = linear(
            unet_, silu(time), prefix + ".time_emb_proj");
        projected = [graph_ reshapeTensor:projected
            withShape:shape({1, dimension(hidden, 1), 1, 1}) name:nil];
        hidden = add(hidden, projected);
        hidden = silu(group_norm(
            unet_, hidden, prefix + ".norm2", 1.0e-5f));
        hidden = conv(unet_, hidden, prefix + ".conv2");
        MPSGraphTensor* residual = input;
        if (unet_.contains(prefix + ".conv_shortcut.weight"))
            residual = conv(unet_, input, prefix + ".conv_shortcut", 1, 0, 0);
        return add(hidden, residual);
    }

    MPSGraphTensor* unet(MPSGraphTensor* sample, int timestep) {
        MPSGraphTensor* time = time_embedding(timestep);
        MPSGraphTensor* hidden = conv(unet_, sample, "conv_in");
        std::vector<MPSGraphTensor*> skips;
        skips.push_back(hidden);
        constexpr int down_heads[3] = {5, 10, 20};
        for (int block = 0; block < 4; ++block) {
            const std::string root = "down_blocks." + std::to_string(block);
            for (int layer = 0; layer < 2; ++layer) {
                hidden = unet_resnet(hidden, time,
                    root + ".resnets." + std::to_string(layer));
                if (block < 3)
                    hidden = transformer(hidden,
                        root + ".attentions." + std::to_string(layer),
                        down_heads[block]);
                skips.push_back(hidden);
            }
            if (block != 3) {
                hidden = conv(unet_, hidden, root + ".downsamplers.0.conv",
                    2, 1, 1);
                skips.push_back(hidden);
            }
        }
        hidden = unet_resnet(hidden, time, "mid_block.resnets.0");
        hidden = transformer(hidden, "mid_block.attentions.0", 20);
        hidden = unet_resnet(hidden, time, "mid_block.resnets.1");

        constexpr int up_heads[4] = {0, 20, 10, 5};
        for (int block = 0; block < 4; ++block) {
            const std::string root = "up_blocks." + std::to_string(block);
            for (int layer = 0; layer < 3; ++layer) {
                MPSGraphTensor* skip = skips.back();
                skips.pop_back();
                hidden = [graph_ concatTensors:@[hidden, skip]
                                     dimension:1 name:nil];
                hidden = unet_resnet(hidden, time,
                    root + ".resnets." + std::to_string(layer));
                if (block != 0)
                    hidden = transformer(hidden,
                        root + ".attentions." + std::to_string(layer),
                        up_heads[block]);
            }
            if (block != 3) {
                hidden = nearest(hidden, dimension(skips.back(), 2),
                    dimension(skips.back(), 3));
                hidden = conv(unet_, hidden, root + ".upsamplers.0.conv");
            }
        }
        hidden = silu(group_norm(unet_, hidden, "conv_norm_out", 1.0e-5f));
        return conv(unet_, hidden, "conv_out");
    }

    const SafeTensors& unet_;
    const SafeTensors& vae_;
    const TokenTensor& prompt_;
    bool fp16_;
    bool full_v1_;
    int width_;
    int height_;
    MPSGraph* graph_;
    MPSGraphTensor* rgb_ = nil;
    MPSGraphTensor* target_noise_ = nil;
    MPSGraphTensor* decoded_ = nil;
    std::unordered_map<std::string, std::vector<std::uint16_t>> unet_half_;
    std::unordered_map<std::string, std::vector<std::uint16_t>> vae_half_;
    std::unordered_map<std::string, std::vector<float>> unet_float_;
    std::unordered_map<std::string, std::vector<float>> vae_float_;
    std::unordered_map<std::string, MPSGraphTensor*> constant_tensors_;
    std::vector<std::uint16_t> prompt_half_;
};

struct PlanKey {
    int width;
    int height;
    bool operator==(const PlanKey& other) const {
        return width == other.width && height == other.height;
    }
};

struct PlanHash {
    std::size_t operator()(const PlanKey& key) const {
        return static_cast<std::size_t>(key.width) * 65537u + key.height;
    }
};

struct Plan {
    MPSGraph* graph = nil;
    MPSGraphExecutable* executable = nil;
};

class MetalExternalJob final : public ExternalJob {
public:
    explicit MetalExternalJob(
        std::shared_ptr<inferbridge::native_harness::metal::Submission> value)
        : submission_(std::move(value)) {}
    ExternalJobState state() const override {
        if (submission_->cancelled()) return ExternalJobState::cancelled;
        return submission_->complete() ? ExternalJobState::complete :
            ExternalJobState::running;
    }
    void cancel() override { submission_->cancel(); }
private:
    std::shared_ptr<inferbridge::native_harness::metal::Submission> submission_;
};

}  // namespace

class MetalExecutor::Impl {
public:
    Impl(const ModelBundle& model, const TokenTensor& prompt, bool full_v1)
        : model_(model), prompt_(prompt), full_v1_(full_v1) {
        const auto precision = inferbridge::native::requested_precision();
        if (precision == inferbridge::native::Precision::int8)
            throw std::invalid_argument("Marigold Metal does not support INT8 yet");
        fp16_ = precision == inferbridge::native::Precision::fp16 ||
            precision == inferbridge::native::Precision::automatic;
        device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil) throw std::runtime_error("Metal is unavailable");
        queue_ = [device_ newCommandQueue];
        graph_device_ = [MPSGraphDevice deviceWithMTLDevice:device_];
        if (queue_ == nil || graph_device_ == nil)
            throw std::runtime_error("could not initialize Marigold Metal");
        inferbridge::native_harness::metal::label_queue(queue_, "Marigold");
        texture_pipeline_ = std::make_unique<
            inferbridge::native_harness::metal::TexturePipeline>(
                device_, "Marigold");
        auxiliary_pool_ = std::make_shared<
            inferbridge::native_harness::metal::AuxiliaryTensorPool>(
                device_, "Marigold");
    }

    void set_cache_path(const std::string& cache_path) {
        std::lock_guard<std::mutex> guard(mutex_);
        cache_path_ = cache_path;
    }

    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) {
        std::uint32_t width = 0u, height = 0u;
        inferbridge::native_harness::fit_diffusion_shape(
            request.width, request.height, width, height);
        const std::size_t latent_count = static_cast<std::size_t>(4u) *
            (width / 8u) * (height / 8u);
        std::mt19937_64 generator(request.seed);
        std::normal_distribution<float> normal;
        std::vector<float> noise(latent_count);
        for (float& value : noise) value = normal(generator);
        const float mean[3] = {0.5f, 0.5f, 0.5f};
        const float deviation[3] = {0.5f, 0.5f, 0.5f};
        inferbridge::native_harness::metal::Request texture_request;
        texture_request.input_texture = request.shared_texture_handle;
        texture_request.input_width = request.width;
        texture_request.input_height = request.height;
        texture_request.input_format = request.rgba ?
            inferbridge::native_harness::metal::PixelFormat::rgba8 :
            inferbridge::native_harness::metal::PixelFormat::bgra8;
        texture_request.wait_event = request.wait_fence_handle;
        texture_request.wait_value = request.wait_fence_value;
        texture_request.output_texture = request.output_texture_handle;
        texture_request.output_width = request.output_width;
        texture_request.output_height = request.output_height;
        texture_request.signal_event = request.signal_fence_handle;
        texture_request.signal_value = request.signal_fence_value;
        std::lock_guard<std::mutex> guard(mutex_);
        @autoreleasepool {
            auto prepared = texture_pipeline_->prepare(
                texture_request, width, height, mean, deviation);
            const Plan& plan = get_presentation_plan(width, height);
            auto auxiliary = auxiliary_pool_->acquire({
                {{1, 4, height / 8, width / 8}, MPSDataTypeFloat32,
                    sizeof(float), MTLResourceStorageModeShared,
                    "Target Noise"}});
            std::memcpy(auxiliary->buffer(0).contents, noise.data(),
                noise.size() * sizeof(float));
            prepared.retained_resources.push_back(auxiliary);
            NSArray<MPSGraphTensorData*>* inputs = @[prepared.input_data,
                auxiliary->data(0)];
            MPSGraphExecutableExecutionDescriptor* execution =
                [MPSGraphExecutableExecutionDescriptor new];
            execution.waitUntilCompleted = NO;
            NSArray<MPSGraphTensorData*>* results = [plan.executable
                runAsyncWithMTLCommandQueue:texture_pipeline_->queue()
                inputsArray:inputs resultsArray:@[prepared.output_data]
                executionDescriptor:execution];
            if (results.count != 1u)
                throw std::runtime_error("Marigold Metal output binding failed");
            return std::make_shared<MetalExternalJob>(
                texture_pipeline_->finish(prepared, width, height));
        }
    }

    ImageTensor infer(
        const float* rgb,
        std::uint32_t width,
        std::uint32_t height,
        const float* target_noise) {
        std::vector<float> image(
            static_cast<std::size_t>(3) * width * height);
        for (std::uint32_t y = 0; y < height; ++y)
            for (std::uint32_t x = 0; x < width; ++x)
                for (std::uint32_t c = 0; c < 3; ++c)
                    image[(static_cast<std::size_t>(c) * height + y) * width + x] =
                        rgb[(static_cast<std::size_t>(y) * width + x) * 3 + c] *
                            2.0f - 1.0f;
        const std::size_t latent_elements =
            static_cast<std::size_t>(4) * (width / 8) * (height / 8);

        std::lock_guard<std::mutex> guard(mutex_);
        @autoreleasepool {
            const Plan& plan = get_plan(width, height);
            id<MTLBuffer> rgb_buffer = [device_ newBufferWithBytes:image.data()
                length:image.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> target_buffer = [device_ newBufferWithBytes:target_noise
                length:latent_elements * sizeof(float)
                options:MTLResourceStorageModeShared];
            if (rgb_buffer == nil || target_buffer == nil)
                throw std::bad_alloc();
            NSArray<MPSGraphTensorData*>* inputs = @[
                [[MPSGraphTensorData alloc] initWithMTLBuffer:rgb_buffer
                    shape:shape({1, 3, static_cast<NSInteger>(height),
                                 static_cast<NSInteger>(width)})
                    dataType:MPSDataTypeFloat32],
                [[MPSGraphTensorData alloc] initWithMTLBuffer:target_buffer
                    shape:shape({1, 4, static_cast<NSInteger>(height / 8),
                                 static_cast<NSInteger>(width / 8)})
                    dataType:MPSDataTypeFloat32]
            ];
            MPSGraphExecutableExecutionDescriptor* execution =
                [MPSGraphExecutableExecutionDescriptor new];
            execution.waitUntilCompleted = YES;
            NSArray<MPSGraphTensorData*>* results = [plan.executable
                runWithMTLCommandQueue:queue_ inputsArray:inputs
                resultsArray:nil executionDescriptor:execution];
            if (results.count != 1)
                throw std::runtime_error("Marigold Metal returned incomplete output");
            const std::uint32_t decoded_width = width / 8 * 8;
            const std::uint32_t decoded_height = height / 8 * 8;
            ImageTensor output{3, decoded_height, decoded_width,
                std::vector<float>(static_cast<std::size_t>(3) *
                    decoded_width * decoded_height)};
            [results[0].mpsndarray readBytes:output.values.data() strideBytes:nil];
            return output;
        }
    }

private:
    const Plan& get_plan(std::uint32_t width, std::uint32_t height) {
        const PlanKey key{static_cast<int>(width), static_cast<int>(height)};
        auto found = plans_.find(key);
        if (found != plans_.end()) return found->second;
        MPSGraphCompilationDescriptor* descriptor =
            [MPSGraphCompilationDescriptor new];
        descriptor.optimizationLevel = MPSGraphOptimizationLevel0;
        descriptor.waitForCompilationCompletion = YES;
        NSURL* package = cache_url(key);
        MPSGraphExecutable* executable = nil;
        if (@available(macOS 14.0, *)) {
            if (package != nil && [[NSFileManager defaultManager]
                    fileExistsAtPath:package.path]) {
                @try {
                    executable = [[MPSGraphExecutable alloc]
                        initWithMPSGraphPackageAtURL:package
                        compilationDescriptor:descriptor];
                } @catch (NSException*) {
                    [[NSFileManager defaultManager]
                        removeItemAtURL:package error:nil];
                    executable = nil;
                }
            }
        }
        MPSGraph* graph = nil;
        if (executable == nil) {
            GraphBuilder builder(
                model_, prompt_, fp16_, full_v1_, key.width, key.height);
            builder.build();
            graph = builder.graph();
            NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feeds =
                [NSMutableDictionary dictionary];
            feeds[builder.rgb()] = [[MPSGraphShapedType alloc]
                initWithShape:shape({1, 3, key.height, key.width})
                dataType:MPSDataTypeFloat32];
            feeds[builder.target_noise()] = [[MPSGraphShapedType alloc]
                initWithShape:shape({1, 4, key.height / 8, key.width / 8})
                dataType:MPSDataTypeFloat32];
            executable = [builder.graph()
                compileWithDevice:graph_device_ feeds:feeds
                targetTensors:@[builder.decoded()] targetOperations:nil
                compilationDescriptor:descriptor];
            if (@available(macOS 14.0, *)) {
                if (executable != nil && package != nil) {
                    @try {
                        [executable serializeToMPSGraphPackageAtURL:package
                                                         descriptor:nil];
                    } @catch (NSException*) {
                        [[NSFileManager defaultManager]
                            removeItemAtURL:package error:nil];
                    }
                }
            }
        }
        if (executable == nil)
            throw std::runtime_error("failed to compile Marigold Metal graph");
        executable.options = MPSGraphOptionsSynchronizeResults;
        return plans_.emplace(key, Plan{graph, executable})
            .first->second;
    }

    const Plan& get_presentation_plan(
        std::uint32_t width, std::uint32_t height) {
        const PlanKey key{-static_cast<int>(width), static_cast<int>(height)};
        auto found = plans_.find(key);
        if (found != plans_.end()) return found->second;
        GraphBuilder builder(model_, prompt_, fp16_, full_v1_, width, height);
        builder.build();
        MPSGraph* graph = builder.graph();
        MPSGraphTensor* low = [graph constantWithScalar:-1.0
            dataType:MPSDataTypeFloat32];
        MPSGraphTensor* one = [graph constantWithScalar:1.0
            dataType:MPSDataTypeFloat32];
        MPSGraphTensor* decoded = [graph clampWithTensor:builder.decoded()
            minValueTensor:low maxValueTensor:one name:nil];
        MPSGraphTensor* depth = [graph meanOfTensor:decoded axes:@[@1] name:nil];
        depth = [graph multiplicationWithPrimaryTensor:
            [graph additionWithPrimaryTensor:depth secondaryTensor:one name:nil]
            secondaryTensor:[graph constantWithScalar:0.5
                dataType:MPSDataTypeFloat32] name:nil];
        MPSGraphTensor* minimum = [graph reductionMinimumWithTensor:depth
            axes:@[@0, @1, @2, @3] name:nil];
        MPSGraphTensor* denominator = [graph maximumWithPrimaryTensor:
            [graph subtractionWithPrimaryTensor:one secondaryTensor:minimum name:nil]
            secondaryTensor:[graph constantWithScalar:1.0e-12
                dataType:MPSDataTypeFloat32] name:nil];
        depth = [graph divisionWithPrimaryTensor:
            [graph subtractionWithPrimaryTensor:depth secondaryTensor:minimum name:nil]
            secondaryTensor:denominator name:@"normalized_depth"];
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feeds =
            [NSMutableDictionary dictionary];
        feeds[builder.rgb()] = [[MPSGraphShapedType alloc]
            initWithShape:shape({1, 3, height, width}) dataType:MPSDataTypeFloat32];
        feeds[builder.target_noise()] = [[MPSGraphShapedType alloc]
            initWithShape:shape({1, 4, height / 8, width / 8})
            dataType:MPSDataTypeFloat32];
        MPSGraphCompilationDescriptor* descriptor =
            [MPSGraphCompilationDescriptor new];
        descriptor.optimizationLevel = MPSGraphOptimizationLevel0;
        descriptor.waitForCompilationCompletion = YES;
        MPSGraphExecutable* executable = [graph compileWithDevice:graph_device_
            feeds:feeds targetTensors:@[depth] targetOperations:nil
            compilationDescriptor:descriptor];
        if (executable == nil)
            throw std::runtime_error("failed to compile Marigold Metal presentation graph");
        executable.options = MPSGraphOptionsSynchronizeResults;
        return plans_.emplace(key, Plan{graph, executable}).first->second;
    }

    NSURL* cache_url(const PlanKey& key) const {
        if (@available(macOS 14.0, *)) {
            if (cache_path_.empty()) return nil;
            NSString* directory = [[NSString
                stringWithUTF8String:cache_path_.c_str()]
                stringByAppendingPathComponent:@"MarigoldMetalGraphCache-v1"];
            if (![[NSFileManager defaultManager]
                    createDirectoryAtPath:directory
                    withIntermediateDirectories:YES attributes:nil error:nil])
                return nil;
            const NSOperatingSystemVersion os =
                NSProcessInfo.processInfo.operatingSystemVersion;
            const char* model_identity = full_v1_
                ? "da9c13e214461c2cf82e4a0f125d914976522e53806a54d508e30ea5b8cd67f2"
                : "953f1ea06169fc6c358b09ecc96a6ee32515e81540442d16239f82348ea62614";
            const std::string name =
                std::string(model_identity) + "-" +
                std::to_string(key.width) + "x" +
                std::to_string(key.height) + "-" +
                (fp16_ ? "fp16" : "fp32") + "-" +
                std::to_string(device_.registryID) + "-macos" +
                std::to_string(os.majorVersion) + "." +
                std::to_string(os.minorVersion) + ".mpsgraphpackage";
            return [NSURL fileURLWithPath:[directory
                stringByAppendingPathComponent:ns(name)]];
        }
        return nil;
    }

    const ModelBundle& model_;
    const TokenTensor& prompt_;
    bool full_v1_ = false;
    bool fp16_ = false;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    MPSGraphDevice* graph_device_ = nil;
    std::unordered_map<PlanKey, Plan, PlanHash> plans_;
    std::mutex mutex_;
    std::unique_ptr<inferbridge::native_harness::metal::TexturePipeline>
        texture_pipeline_;
    std::shared_ptr<inferbridge::native_harness::metal::AuxiliaryTensorPool>
        auxiliary_pool_;
    std::string cache_path_;
};

MetalExecutor::MetalExecutor(
    const ModelBundle& model, const TokenTensor& prompt, bool full_v1)
    : impl_(std::make_unique<Impl>(model, prompt, full_v1)) {}

MetalExecutor::~MetalExecutor() = default;

void MetalExecutor::set_cache_path(const std::string& cache_path) {
    impl_->set_cache_path(cache_path);
}

ImageTensor MetalExecutor::infer(
    const float* rgb,
    std::uint32_t width,
    std::uint32_t height,
    const float* target_noise) {
    return impl_->infer(rgb, width, height, target_noise);
}

std::shared_ptr<ExternalJob> MetalExecutor::submit_texture(
    const ExternalTextureRequest& request) {
    return impl_->submit_texture(request);
}

}  // namespace marigold_native
