# Use Ubuntu 22.04 which has newer GLIBC and GLIBCXX versions
FROM ubuntu:22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    libeigen3-dev \
    gdb \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory for your project
WORKDIR /app

# Copy all source files and makefiles
COPY . /app/

# Build the project
RUN make

# Set environment variable for Eigen include path
ENV EIGEN3_INCLUDE_DIR=/usr/include/eigen3

# Set the working directory to the Bin directory
WORKDIR /app/Bin

# Command to provide a bash shell for interactive use
CMD ["/bin/bash"]