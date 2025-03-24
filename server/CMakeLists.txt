cmake_minimum_required(VERSION 3.05)

project(n_body_problem_solver_server)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

add_library(
    protolib
    proto/n_body_problem.proto
)

target_link_libraries(protolib gRPC::grpc++)
target_include_directories(protolib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
protobuf_generate(TARGET protolib LANGUAGE cpp)
protobuf_generate(
    TARGET protolib 
    LANGUAGE grpc 
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}"
)

cuda_add_library(
    n_body_problem_gpu_solver
    src/gpu/n_body_problem_solver.cu
)


target_include_directories(
    n_body_problem_gpu_solver
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(${PROJECT_NAME} src/server.cpp)
target_link_libraries(${PROJECT_NAME} protolib n_body_problem_gpu_solver)
