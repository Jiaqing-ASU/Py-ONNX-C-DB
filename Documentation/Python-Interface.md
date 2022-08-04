# IBM Telum AI & ONNX-MLIR Python Toolkit Python-Interface Documentation

## Story 1: Compiling an ONNX from Python using ONNX-MLIR (Done)
We implemented this function in onnx-mlir. We provide a Python interface to compile models named PyOnnxMlirCompiler.

### Running the PyOnnxMlirCompiler interface

An ONNX model can be compiled directly using the `onnx-mlir -O3 --EmitLib` command.
The resulting library can then be executed using Python as shown in [onnx-mlir README](https://github.com/onnx/onnx-mlir/blob/main/docs/UsingPyRuntime.md). At times, it might be convenient to also compile a model directly in Python. We explores the Python methods to do so.

```python
from PyOnnxMlirCompiler import OnnxMlirCompiler, OnnxMlirTarget, OnnxMlirOption

# Load onnx model and create Onnx Mlir Compiler object.
# Compiler needs to know where to find its runtime. Set ONNX_MLIR_RUNTIME_DIR
# to proper path, e.g. export ONNX_MLIR_RUNTIME_DIR=../../build/Debug/lib.
file = './mnist.onnx'
compiler = OnnxMlirCompiler(file)
# Set optimization level to -O3 and enable verbose mode.
compiler.set_option(OnnxMlirOption.opt_level, "3")
compiler.set_option(OnnxMlirOption.verbose, "")
print("Compile", file, "with opt level", compiler.get_option(OnnxMlirOption.opt_level), "for arch", compiler.get_option(OnnxMlirOption.target_arch), "and CPU", compiler.get_option(OnnxMlirOption.target_cpu))
# Generate the library file. Success when rc == 0.
rc = compiler.compile('./mnist-O3', OnnxMlirTarget.emit_lib)
model = compiler.get_output_file_name()
if rc:
    print("Failed to compile with error code", rc)
    exit(1)
print("Compiled onnx file", file, "to", model, "with rc", rc)
```

The `PyOnnxMlirCompiler` module exports the `OnnxMlirCompiler` class to drive the
compilation of a ONNX model into an executable model.
Typically, a compiler object is created for a given model by giving it the file name of the ONNX model.
Then, compiler options can be set/reset to steer the compilation to generate the desired executable.
Finally, the compilation itself is performed by calling the `compile()` command where the user passes the output file name (without extension) as well as the desired compilation target.
Typical targets are `emit_lib` or `emit_jni`.

The modules also export two enums that defines which compiler options can be set (`OnnxMlirOption`) and what is the target output of the compilation (`OnnxMlirTarget`).

The `compile` commands returns a return code reflecting the status of the compilation.
A zero value indicates success, and nonzero values reflect the error code.
Because different Operating Systems may have different suffixes for libraries,
the output file name can be retrieved using the `get_output_file_name()` method.

### PyOnnxMlirCompiler model API

The complete interface to OnnxMlirCompiler can be seen in the sources mentioned previously.
However, using the constructor and the methods below are enough to compile models.

```python
def __init__(self, file_name: str):
    """
    Constructor for an ONNX model contained in a file.

    Args:
        file_name: relative or absolute path to your ONNX model.
    """

def __init__(self, input_buffer: void *, buffer_size: int):
    """
    Constructor for an ONNX model contained in an input buffer.

    Args:
        input_buffer: buffer containing the protobuf representation of the model.
        buffer_size: byte size of the input buffer.
    """

def set_option_from_env(self, env_var_ame: str):
    """
    Method to provide the compiler its options via an environment variable.

    Args:
        env_var_ame: name of the environment variable containing the onnx-mlir options.
        All options listed by `onnx-mlir --help` are supported.

    Returns:
        Zero in case of success, error code in case of failure.
    """

def set_option(self, kind: OnnxMlirOption, value: str):
    """
    Method to provide one compiler option.

    Args:
        kind: name of the option being set. Typical values are
        OnnxMlirOption.opt_level (e.g. "0".."5"), target_triple, target_arch,
        target_cpu, target_accel, or verbose.
        value: value of the option being set.

    Returns:
        Zero in case of success, error code in case of failure.
    """

def clear_option(self, kind: OnnxMlirOption):
    """
    Method to clear one compiler option.

    Args:
        kind: name of the option being cleared.
    """

def get_option(self, kind: OnnxMlirOption):
    """
    Method to get one compiler option.

    Args:
        kind: name of the option being retrieved.

    Returns:
        String corresponding to the compiler option's current value.
    """

def compile(self, output_base_name: str, target: OnnxMlirTarget):
    """
    Method to compile a model.

    Args:
        output_base_name: base name (relative or absolute, without suffix)
        where the compiled model should be written into.
        target: target for the compiler's output. Typical values are
        OnnxMlirTarget.emit_lib or emit_jni.

    Returns:
        Zero on success, error code on failure.
    """

def get_output_file_name(self):
    """
    Method to provide the full (absolute or relative) output file name, including
    its suffix.

    Returns:
        String containing the fle name after successful compilation; empty string on failure.
    """

def get_error_message(self):
    """
    Method to provide the compilation error message.

    Returns:
        String containing the error message; empty string on success.
    """
```

This interfaces uses two Python enums to describe the compiler options and targets. The compiler options are `OnnxMlirOption.target_triple`, `target_arch`, `target_cpu`, `target_accel`, `opt_level`, `opt_flag`, `llc_flag`, `llvm_flag`, and `verbose`.
* Target options uses defined LLVM strings to indicate the compilation target.
* The `opt_level` indicates the optimization level, typically indicated as `-O0` to `-O5`. Values passed in the `set_option` should simply be the numeral value, i.e. `0` to `5`.
* Flags options enable the user to pass specific option strings to the optimizer (`opt`) and the llvm `llc` compiler.
* The verbose option enable a verbose mode of the compiler. The `set_option` simply set this boolean flag on, regardless of the passed string value.

The target compiler options are `OnnxMlirTarget.emit_onnx_basic`, `emit_onnxir`, `emit_mlir`, `emit_llvmir`, `emit_obj`, `emit_lib`, and `emit_jni`. Their meaning is
defined by executing `onnx-mlir --help`. Notable values are:
*  `emit_onnxir` to list the onnx operations in the MLIR textual representation,
*  `emit_lib` to generate an executable library of the model, and
*  `emit_jni` to generate a Java jar file of the model.

## Story 2: Provide one interface which contains two kinds of functions: Compilation and Run (Done)
The design of onnx MLIR separates the compilation and operation of the model. We understand that for some professional users, this design has many benefits. However, for other users, such a design has a certain threshold or will bring some confusion to users. In order to simplify the difficulty of users, we designed an interface to package the compilation and operation of the model. Currently, onnx MLIR can provide users with three types of interfaces. The following are examples of three types of interfaces.
1. [PyOnnxMlirCompiler](https://github.com/Jiaqing-ASU/onnx-mlir/blob/python-interface/docs/mnist_example/mnist_compile.py)
2. [PyRuntime](https://github.com/Jiaqing-ASU/onnx-mlir/blob/python-interface/docs/mnist_example/mnist.py)
3. [PyRuntimePlus](https://github.com/Jiaqing-ASU/onnx-mlir/blob/python-interface/docs/mnist_example/mnist_exe_plus.py)

## Story 3: Solve the thread safe issue for all interfaces (Done)
We have met an issue of onnx-mlir while working on the Python interface. The issue is that the later compilation will be still working under the first or previous set of optimizations. Here is an example:
```python
copt1= OMCompilerOptions()
copt1.opt_level = 3
copt1.maccel = NNPA
sess1.compile(copt1)

# Sometime later with a sess2

copt2= OMCompilerOptions()
sess2.compile(copt2)
```
The second compile is still working under the first set of optimizations, as the optimization flags in onnx-mlir are global.

One approach is to reset all options when starting the compile by creating a new OMCompilerOptions, then set the ones that were set, and then compile. And then the issue is that these are not thread safe. To solve this, we can simply do a lock around the omCompilerFromFile or in pyOnnxMlirCompiler.  But doing it there only solve the issue of multithreading in python, whereas a better solution would solve the issue for all interfaces.

Actually if we launch a process when compiling using the `onnx-mlir` exec as a target, it will work for multi-threading and multi-processing (that one always worked to begin with, the problem is only when sharing an address space as some parameters are shared among all threads, namely not thread private).

The only think for multi-threading, is that right now onnx-mlir rely on `ONNX_MLIR_FLAG` which is a fixed name. So the user cannot set this value in one thread to the env variable of the current process, and not hope that another thread may do the same. The approach is to create a thread unique name, and add as a parameter to `onnx-mlir` a option to set a custom name.

Our approach is to have a custom env var name when we have multiple processes. We could add an option to read the env var name from the environment. This name (ONNX_MLIR_FLAG) is currently a [`extern const std::string OnnxMlirEnvOptionName`](https://github.com/onnx/onnx-mlir/blob/62de6adc89ea5c0cf8bc1f58857dfe74d589618c/src/Compiler/CompilerUtils.cpp). We could transform this as an option. We export and define the new option:
```cpp
extern llvm::cl::opt<std::string> menvVarName;
llvm::cl::opt<std::string> menvVarName("menvVarName",
    llvm::cl::desc("Override default option env var OnnxMlirEnvOptionName: ONNX_MLIR_FLAGS"),
    llvm::cl::value_desc("option env var"), llvm::cl::init("ONNX_MLIR_FLAGS"),
    llvm::cl::cat(OnnxMlirOptions));
```

This new option can be supported as follows:
```cpp
void setTargetEnvVar(const std::string &envVarName) {
  assert(envVarName != "" && "Expecting valid target envVarName description");
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set envVarName\"" << envVarName << "\"\n");
  menvVarName = envVarName;
}

void clearTargetEnvVar() { menvVarName.clear(); }

std::string getTargetEnvVarOption() {
  return (menvVarName != "") ? "--menvVarName=" + menvVarName : "";
}
```

We define this new option env var with default value “ONNX_MLIR_FLAG". The options are formally taken care of in [onnx-mlir.cpp](https://github.com/onnx/onnx-mlir/blob/main/src/onnx-mlir.cpp#L61). We change the `OnnxMlirEnvOptionName` to `menvVarName` as follows.
```cpp
// Parse options from argc/argv and env var.
llvm::cl::ParseCommandLineOptions(argc, argv,
    "ONNX-MLIR modular optimizer driver\n", nullptr,
    menvVarName.c_str());
```
Here, we need to give the ONNX_MLIR_FLAG name to the option processing. So, we have to process the options manually (argv) to see if we have an "—option-env-var” and pass it there. Once that is done, we can define a thread unique name for the env var and launch a process to compile using “onnx-mlir”.
The process is launched as follows defined in [CompilerUtils.cpp](https://github.com/onnx/onnx-mlir/blob/62de6adc89ea5c0cf8bc1f58857dfe74d589618c/src/Compiler/CompilerUtils.cpp#L419) and we can do the same in [OnnxMlirCompiler.cpp](https://github.com/Jiaqing-ASU/onnx-mlir/blob/python-interface/src/Compiler/OnnxMlirCompiler.cpp#L52).

## Story 4: Provide pip installable packages (Ongoing)
To make this Python package much easier for users to use, we are going to provide pip installable packages. We need to first create a simple setup.py and related functions to install a simple skeleton package. Verify it shows up on pip package list. And then we need to investigate how to detect and help users to install the pre-requested non-python library, mainly the onnx-mlir. Our Package pre-request onnx-mlir and if we cannot help users to install this, we should give appropriate error messages.

## Story 5: Display onnx-mlir model details and signatures in Python (Ongoing)
At present, we can output instructions and related intermediate files(.bc files or .ll files) while running in onnx-mlir for users in Python. However, we need to further investigate Tensorflow and Pytoch to see what information those tools which users might be more familiar to can provide in the process of compiling models. We try to provide some similar or even more valuable information in the Python interfaces.