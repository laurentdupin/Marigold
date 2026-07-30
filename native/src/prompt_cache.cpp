#include "prompt_cache.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace marigold_native {
namespace {

constexpr std::array<char, 8> magic = {
    'M', 'A', 'R', 'P', 'R', 'M', '0', '1'};
constexpr std::uint32_t header_bytes = 512;
constexpr char text_hash[] =
    "bc1827c465450322616f06dea41596eac7d493f4e95904dcb51f0fc745c4e13f";
struct ModelIdentity {
    const char* revision;
    const char* unet_hash;
    const char* vae_hash;
};
constexpr ModelIdentity identities[] = {
    {
        "04a73502f7fd8fc5e59947b9df3b2266d71d6849",
        "953f1ea06169fc6c358b09ecc96a6ee32515e81540442d16239f82348ea62614",
        "a4302e1efa25f3a47ceb7536bc335715ad9d1f203e90c2d25507600d74006e89",
    },
    {
        "f4fc453d7d217cbe30ddcad3eb311d1ad9a11c4c",
        "da9c13e214461c2cf82e4a0f125d914976522e53806a54d508e30ea5b8cd67f2",
        "3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a",
    },
};

std::uint32_t u32(const char* data) {
    return std::uint32_t(static_cast<unsigned char>(data[0])) |
        (std::uint32_t(static_cast<unsigned char>(data[1])) << 8) |
        (std::uint32_t(static_cast<unsigned char>(data[2])) << 16) |
        (std::uint32_t(static_cast<unsigned char>(data[3])) << 24);
}

void expect(
    const char* actual,
    std::size_t bytes,
    const char* expected,
    const char* field) {
    if (std::strlen(expected) + 1 > bytes ||
        std::strncmp(actual, expected, bytes) != 0) {
        throw std::runtime_error(
            std::string("Marigold prompt cache ") + field + " mismatch");
    }
}

}  // namespace

TokenTensor load_empty_prompt_cache(
    const std::string& path,
    bool full_v1) {
    std::ifstream stream(
        std::filesystem::u8path(path), std::ios::binary);
    std::array<char, header_bytes> header{};
    stream.read(header.data(), header.size());
    if (!stream ||
        !std::equal(magic.begin(), magic.end(), header.begin()) ||
        u32(header.data() + 8) != 1 ||
        u32(header.data() + 12) != header_bytes ||
        u32(header.data() + 16) != 2 ||
        u32(header.data() + 20) != 1024) {
        throw std::runtime_error("invalid Marigold prompt cache header");
    }
    const ModelIdentity& identity = identities[full_v1 ? 1 : 0];
    expect(header.data() + 24, 41, identity.revision, "revision");
    expect(header.data() + 65, 65, identity.unet_hash, "UNet hash");
    expect(header.data() + 130, 65, identity.vae_hash, "VAE hash");
    expect(header.data() + 195, 65, text_hash, "text hash");
    TokenTensor prompt{
        2, 1024, std::vector<float>(2 * 1024)};
    stream.read(
        reinterpret_cast<char*>(prompt.values.data()),
        static_cast<std::streamsize>(
            prompt.values.size() * sizeof(float)));
    if (!stream || stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("invalid Marigold prompt cache payload");
    }
    return prompt;
}

}  // namespace marigold_native
