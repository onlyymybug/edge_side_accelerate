#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
toolchain/android-ndk-r27b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android30-clang++ \
  -std=c++17 -O2 -shared -fPIC -static-libstdc++ native/decode_diag.cpp -ldl -o native/libaw_decode_diag.so
