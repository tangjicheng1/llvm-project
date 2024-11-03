#!/bin/bash
cmake -S llvm -B build -DCMAKE_BUILD_TYPE=Debug -G Ninja -DLLVM_TARGETS_TO_BUILD=X86
