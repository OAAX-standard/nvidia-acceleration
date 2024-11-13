# nvidia-acceleration

This folder contains the source code of the shared library and the Docker image that can be used by AI application developers to benefit from the acceleration offered by Nvidia GPUs on x86_64 and aarch64 machines.

## Artifacts

- The OAAX runtime is available as a shared library that can be used by developers to load and run optimized models on an Nvidia GPU.

## Usage

### Using the runtime library

To use the runtime library, you need to have the Nvidia driver and CUDA toolkit installed on the machine.

The CUDA OAAX runtime can be used just like the other OAAX runtimes. You can find various and diverse usage examples in the [examples](https://github.com/oaax-standard/examples) repository.

> Note: Some library files in the `deps/` folder are empty due to the limitations of GitHub on file sizes. Please replace the empty files with the actual files from the Nvidia CUDA Toolkit, then rebuild onnxruntime locally.  
Run the following command to find the empty libraries:  
`
find runtime-library/deps/ -iname "*.so*"  -type f -empty
`