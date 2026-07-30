#include "marigold_native.h"

int main(void) {
    marigold_context* context = 0;
    if (marigold_native_abi_version() != MARIGOLD_NATIVE_ABI_VERSION) {
        return 1;
    }
    if (marigold_create(0, 0, 0, &context) != MARIGOLD_INVALID_ARGUMENT) {
        return 2;
    }
    marigold_destroy(context);
    return 0;
}
