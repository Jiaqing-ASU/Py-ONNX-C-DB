# IBM Telum AI & ONNX-MLIR Python Toolkit Plan

## Overview

Create a toolkit that enables users to leverage ONNX-MLIR from python. 
In phase 1, with this toolkit: 
1. ONNX models can be compiled in python, specifying needed args.
2. Compiled ONNX models can be immediately tested in python (as a python object)
3. A user can display graph details, including whether an operation will use available accelerators (IBM Telum via NNPA). 
4. A user can access graph attributes via python API (e.g., input tensor dims).
Additional functions can be discussed as time allows:
1. A TensorFlow or Pytorch model can be compiled directly via transparent conversion to onnx.
2. Users can use pip to directly install the Python package and inport it and use it.

## Stories / Tasks
1. Name project (all members)
2. Establish temporary github repository including creating github project board and stories/tasks.
3. Story: I can install our python toolkit using PIP
•Initial tasks:
•(Investigation spike) How-to create a pip installable package
•Create a simple setup.py and related functions to install a simple skeleton package. Verify it shows up on pip package 
list.
•(Investigation spike) How to detect and prereq non-python library (onnx-mlir)
•Our Package prereq ONNX-MLIR and fail if not found with appropriate error message.
4. Story: Compiling an ONNX from Python using ONNX-MLIR
•Initial tasks:
•(Investigation spike) Invoking ONNX-MLIR compiler using C API
•Create a dummy python program and demonstrate calling ONNX-MLIR compiler with ONNX model.
•(Design spike) Design project compilation APIs and hold design review (virtual)
•(Dev spike – must be broken down to smaller work) Implement python compiler interface. (Includes creating tests and 
all related tasks).
5. Story: Display ONNX-MLIR model details and signature (in python)
•(Design/investigate spike) Investigate different compile output and IR, explore suitable output to be displayed. 
6. Story: Use model program from python
•(Design/investigate spike) Carry over from compilation story; for discussion after 4 is complete