# C Neural Network Library

Welcome to the C Neural Network Library—a comprehensive toolkit thoughtfully engineered for creating, training, and evaluating neural networks using the C programming language. Designed with dedication, this library offers a host of features that redefine the landscape of neural network development.

## Features

- **Efficient N-Dimensional Matrix Operations**: Seamlessly perform a diverse range of operations on N-dimensional matrices. Our matrix library is designed to support a variety of operations, enhancing your efficiency and versatility in neural network computations.

- **Optimized Matrix Operations**: Enhance your neural networks with optimized matrix operations. Our library provides three optimization modes: classical operations, Single Instruction, Multiple Data (SIMD) efficiency, and the power of the Intel Math Kernel Library (MKL).

- **TensorFlow and PyTorch-Inspired Neural Network Architecture**: Embark on neural network development using a syntax reminiscent of well-known frameworks like TensorFlow and PyTorch. The architecture of our neural network module draws inspiration from these widely used libraries, facilitating familiarity and streamlining your workflow.

- **Simplicity and Precision**: Dive into a C-based neural network implementation that artfully balances simplicity and precision. Each component is thoughtfully crafted to deliver an intuitive experience, making intricate neural network tasks accessible and manageable.

- **Dynamic Optimization Techniques**: Our library offers adaptive optimization techniques. Harness the capabilities of Stochastic Gradient Descent (SGD) and Adam optimizers to refine your models with precision and efficiency.

- **Customizable Network Architecture**: Tailor your neural network's architecture to your specific requirements. Integrate various layers, including dense and convolutional, and experiment with activation functions like ReLU and sigmoid to shape your network's performance.

- **Optimized Matrix Operations**: Enhance your neural networks with optimized matrix operations. Our library provides three optimization modes: classical operations, Single Instruction, Multiple Data (SIMD) efficiency, and the power of the Intel Math Kernel Library (MKL).

- **Illustrative Code Examples**: Explore our library through illustrative code examples. These examples illuminate a clear path for understanding and applying neural network concepts in diverse scenarios.

# Simple interface

![alt text](https://github.com/dipezed/Neural-Network-Librairy-C/blob/main/img-readme.png?raw=true "Logo of Safe_Link")

# Optimizers

This library supports various optimization algorithms, including:

    Stochastic Gradient Descent (SGD)
    Adam

# Optimization Modes in Makefile

The Makefile provides three optimization modes for matrix operations:

    -DUSE_SIMD=0 : Classic matrix operations.
    -DUSE_SIMD=1 : Matrix operations using Single Instruction, Multiple Data (SIMD) instructions (explained below).
    -DUSE_SIMD=2 : Matrix operations using the Intel Math Kernel Library (MKL).

# About SIMD

SIMD (Single Instruction, Multiple Data) is a type of computer architecture that enables parallel processing of data elements. When USE_SIMD is enabled, the library uses SIMD instructions to speed up matrix operations, resulting in improved performance.
# Contribution

Contributions are welcome! If you have ideas for improvement, bug fixes, or new features to add, feel free to open a pull request.