# Notes and Research Folder

- Description of any other tools, technologies and APIs needed.
- Links to reference guides, or examples.
- Description of development environment or link to development environment.

## The following links are some important Github repository links:
1. [ONNX-MLIR Github Repository](https://github.com/onnx/onnx-mlir/)
2. [AI on IBM Z Github launch point](https://github.com/IBM/ai-on-z-101/)
3. [ONNX Model Zoo](https://github.com/onnx/models/)

## The following links are some similar sdks for references:
1. [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/tensorflow/openpose_demo/openpose.html/)
2. [AutoTVM](https://tvm.apache.org/docs/tutorial/autotvm_relay_x86.html/)

## How to set up the development environment of onnx-mlir:
An easy way to set up the development environment is to use a prebuilt Docker image. These images are created as a result of a successful merge build on the trunk. This means that the latest image represents the tip of the trunk. Currently there are both Release and Debug mode images for amd64, ppc64le and s390x saved in Docker Hub as, respectively, [onnxmlirczar/onnx-mlir](https://hub.docker.com/r/onnxmlirczar/onnx-mlir) and [onnxmlirczar/onnx-mlir-dev](https://hub.docker.com/r/onnxmlirczar/onnx-mlir-dev). You can find more details in the [ONNX-MLIR Building and Developping ONNX-MLIR using Docker](https://github.com/Jiaqing-ASU/onnx-mlir/blob/main/docs/Docker.md). For development purposes, you need to use [onnxmlirczar/onnx-mlir-dev](https://hub.docker.com/r/onnxmlirczar/onnx-mlir-dev).

After downloading the Docker image, you need to download the latest code of Python interface from [Jiaqing-ASU/onnx-mlir](https://github.com/Jiaqing-ASU/onnx-mlir). The code is saved in the branch [python-interface](https://github.com/Jiaqing-ASU/onnx-mlir/tree/python-interface). Next step is to create a Docker container using the above Docker image. Here, you need to use [bind mount](https://docs.docker.com/storage/bind-mounts/) to use the files from [python-interface](https://github.com/Jiaqing-ASU/onnx-mlir/tree/python-interface) in the [onnxmlirczar/onnx-mlir-dev](https://hub.docker.com/r/onnxmlirczar/onnx-mlir-dev). The command to start such a Docker container should be something like:
```
docker run -it -v [local path]:[container path] [container ID] /bin/bash
```

One thing that needs to be noticed here is that it is recommend to use the following command to obtain the code on GitHub:
```
git clone --recursive https://github.com/Jiaqing-ASU/onnx-mlir.git
```

If you download a compressed file, it will have the following compilation errors due to the lack of relevant files in the third_party folder.
```
CMake Error at CMakeLists.txt:169 (add_subdirectory):
  The source directory

    /workdir/onnx-mlir/third_party/benchmark

  does not contain a CMakeLists.txt file.
```

If you do meet the above errors, what you can do is that keep the original third_party folder which comes from Docker images and do NOT replace it with the third_party folder downloaded from Github.

After you successfully start a Docker container, you can build the project based on the [Installation of ONNX-MLIR on Linux / OSX](https://github.com/Jiaqing-ASU/onnx-mlir/blob/main/docs/BuildOnLinuxOSX.md). One thing that needs special attention. Docker has default CPU, memory and swap settings according to different hardware environments. If you encounter the following errors in the process of building, you need to adjust the settings in docker accordingly. You can check the [Docker Runtime options with Memory, CPUs, and GPUs](https://docs.docker.com/config/containers/resource_constraints/) for more details.
```
c++: fatal error: Killed signal terminated program cc1plus
compilation terminated.
```

## How to run the mnist example:
All the mnist examples are under the [mnist_example](https://github.com/Jiaqing-ASU/onnx-mlir/tree/python-interface/docs/mnist_example). There are 4 examples in total:
1. [mnist_exe.py](https://github.com/Jiaqing-ASU/onnx-mlir/blob/python-interface/docs/mnist_example/mnist_exe.py): Given an compiled ONNX model file, run the Execution Session only.
2. [mnist_compile.py](https://github.com/Jiaqing-ASU/onnx-mlir/blob/python-interface/docs/mnist_example/mnist_compile.py): Given an ONNX model file, run the Compilation Session only.
2. [mnist_compile_exe.py](https://github.com/Jiaqing-ASU/onnx-mlir/blob/python-interface/docs/mnist_example/mnist_compile_exe.py): Given an ONNX model file, run both Execution Session and Compilation Session and gives the final result using PyOnnxMlirCompiler and PyRuntime.
3. [mnist_exe_plus.py](https://github.com/Jiaqing-ASU/onnx-mlir/blob/python-interface/docs/mnist_example/mnist_exe_plus.py): Given an ONNX model file, run both Execution Session and Compilation Session and gives the final result using PyRuntimePlus.

The PyRuntimePlus interface is one Python package including operations from both PyOnnxMlirCompiler and PyRuntime.

For more details of PyOnnxMlirCompiler, PyRuntime and PyRuntimePlus:
1. [PyOnnxMlirCompiler](https://github.com/Jiaqing-ASU/onnx-mlir/tree/python-interface/src/Compiler)
2. [PyRuntime](https://github.com/Jiaqing-ASU/onnx-mlir/tree/python-interface/src/Runtime)
3. [PyRuntimePlus](https://github.com/Jiaqing-ASU/onnx-mlir/tree/python-interface/src/RuntimePlus)

Using pybind, a C/C++ binary can be directly imported by the Python interpreter. For PyRuntime, such binary is generated by PyExecutionSession (src/Runtime/PyExecutionSession.hpp) and built as a shared library to build/Debug/lib/PyRuntime.cpython-<target>.so. Before running all the examples, the module should be imported normally by the Python interpreter as long as it is in your PYTHONPATH. Another alternative is to create a symbolic link to it in your working directory.
```
cd <working directory>
ln -s ../../build/Debug/lib/PyOnnxMlirCompiler.cpython-38-x86_64-linux-gnu.so
ln -s ../../build/Debug/lib/PyRuntime.cpython-38-x86_64-linux-gnu.so
python3
```

To run the mnist_compile.py and mnist_exe_plus.py, since the compiler needs to know where to find its runtime, you should set ONNX_MLIR_RUNTIME_DIR to proper path as shown in the following before running the python script.
```
export ONNX_MLIR_RUNTIME_DIR=../../build/Debug/lib
```

Without the above settings, you will meet the following error:
```
/usr/bin/ld: cannot find -lcruntime
collect2: error: ld returned 1 exit status
c++ ./mnist-O3.o -o ./mnist-O3.so -shared -fPIC -L/usr/lib -lcruntime
Error message: 
Program path: /usr/bin/c++
Command execution failed.
Failed to compile with error code 12
```