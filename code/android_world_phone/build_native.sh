#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd -P)"
ndk_dir="${ANDROID_NDK_HOME:-$project_dir/toolchain/android-ndk-r27b}"
bundle_dir="${PHONE_BUNDLE_DIR:-$project_dir/../../phone_bundle}"
"$ndk_dir/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android30-clang++" \
  -std=c++17 -O2 -static-libstdc++ "$project_dir/native/vlm_server.cpp" \
  -L "$bundle_dir/lib" -lgeniex -o "$project_dir/native/vlm_server"
