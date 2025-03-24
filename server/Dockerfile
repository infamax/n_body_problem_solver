FROM ubuntu:22.04 AS build

# install development deps
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        gcc \
        g++ \
        clang \
        cmake \
        curl \
        make \
        protobuf-compiler \
        git \
        pkg-config \
        libssl-dev \
        autoconf libtool \
        nano \ 
        vim \
    && rm -rf /var/apt/lists/*

# install cuda toolkit
RUN apt-get update \
    && apt-get install -y \
        nvidia-cuda-toolkit \
    && rm -rf /var/apt/lists/*

# install grpc
WORKDIR /deps 

RUN git clone --recurse-submodules -b v1.62.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc \
    && mkdir -p /deps/grpc/build \ 
    && cd /deps/grpc/build \
    && cmake -DgRPC_INSTALL=ON \
        -DgRPC_BUILD_TESTS=OFF \
        .. && \
        make -j8 install 

WORKDIR /usr/src/app

EXPOSE 9999

COPY . /usr/src/app

RUN mkdir build \
    && cd build \
    && cmake .. \
    && make 

CMD ["/usr/src/app/build/n_body_problem_solver_server"]

