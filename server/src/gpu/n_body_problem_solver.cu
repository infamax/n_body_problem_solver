#include "server/gpu/n_body_problem_solver.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace n_body_problem_solver {

inline constexpr float SOFTENING_FACTOR = 0.001;
inline constexpr int NUM_THREADS = 256;

__global__ void one_step_euler_n_body_problem(float3* positions, float3* velocities, float* weights, float dt, float G, int n) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        float acceleration_x = 0.0; 
        float acceleration_y = 0.0; 
        float acceleration_z = 0.0;

        for (int tile = 0; tile < gridDim.x; ++tile) {
            __shared__ float3 shared_positions[NUM_THREADS];
            __shared__ float shared_weights[NUM_THREADS];
            shared_positions[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];
            shared_weights[threadIdx.x] = weights[tile * blockDim.x + threadIdx.x];
            __syncthreads();

            for (int j = 0; j < NUM_THREADS; ++j) {
            float dx = shared_positions[j].x - positions[index].x;
            float dy = shared_positions[j].y - positions[index].y;
            float dz = shared_positions[j].z - positions[index].z;
            float dist_square = dx*dx + dy*dy + dz*dz + SOFTENING_FACTOR * SOFTENING_FACTOR;
            float inverse_dist = rsqrtf(dist_square);
            float inverse_dist_power_3 = inverse_dist * inverse_dist * inverse_dist;

            acceleration_x += dx * inverse_dist_power_3 * G * shared_weights[j]; 
            acceleration_y += dy * inverse_dist_power_3 * G * shared_weights[j]; 
            acceleration_z += dz * inverse_dist_power_3 * G * shared_weights[j];
            }
            __syncthreads();
        }

        velocities[index].x += dt * acceleration_x; 
        velocities[index].y += dt * acceleration_y; 
        velocities[index].z += dt * acceleration_z;

        positions[index].x += dt * velocities[index].x;
        positions[index].y += dt * velocities[index].y;
        positions[index].z += dt * velocities[index].z;
    }
}


SolutionNBodyProblem gpu_solve_n_body_problem(const NBodyProblemData& data) {
    float3* positions = new float3[data.num_bodies];
    float3* velocities = new float3[data.num_bodies];
    float* weigths = new float[data.num_bodies];

    for (int i = 0; i < data.num_bodies; ++i) {
        positions[i].x = data.bodies[i].position[0];
        positions[i].y = data.bodies[i].position[1];
        positions[i].z = data.bodies[i].position[2];

        velocities[i].x = data.bodies[i].vecocities[0];
        velocities[i].y = data.bodies[i].vecocities[1];
        velocities[i].z = data.bodies[i].vecocities[2];

        weigths[i] = data.bodies[i].weight;
    }

    SolutionNBodyProblem solution;
    solution.iterations.reserve(data.num_iterations);

    const int num_blocks = (data.num_bodies + NUM_THREADS - 1) / NUM_THREADS;

    float3* gpu_positions;
    float3* gpu_velocities;
    float* gpu_weights;

    cudaMalloc((void**)&gpu_positions, sizeof(float3) * data.num_bodies);
    cudaMalloc((void**)&gpu_velocities, sizeof(float3) * data.num_bodies);
    cudaMalloc((void**)&gpu_weights, sizeof(float) * data.num_bodies);
    cudaMemcpy(gpu_weights, weigths, sizeof(float) * data.num_bodies, cudaMemcpyHostToDevice);

    for (int num_iteration = 0; num_iteration < data.num_iterations; ++num_iteration) {
        cudaMemcpy(gpu_positions, positions, sizeof(float3) * data.num_bodies, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_velocities, velocities, sizeof(float3) * data.num_bodies, cudaMemcpyHostToDevice);

        one_step_euler_n_body_problem<<<num_blocks, NUM_THREADS>>>(gpu_positions, gpu_velocities, gpu_weights, data.dt, data.G, data.num_bodies);

        cudaMemcpy(positions, gpu_positions, sizeof(float3) * data.num_bodies, cudaMemcpyDeviceToHost);
        cudaMemcpy(velocities, gpu_velocities, sizeof(float3) * data.num_bodies, cudaMemcpyDeviceToHost);

        std::vector<Body> bodies;
        bodies.reserve(data.num_bodies);

        for (int body_index = 0; body_index < data.num_bodies; ++body_index) {
            bodies.push_back(
                Body{
                    .name = data.bodies[body_index].name,
                    .position = {positions[body_index].x, positions[body_index].y, positions[body_index].z},
                    .vecocities = {velocities[body_index].x, velocities[body_index].y, velocities[body_index].z},
                    .weight = weigths[body_index]
                }
            );
        }

        solution.iterations.push_back(
            OneIterSolutionNBodyProblem{
                .num_iteration=num_iteration,
                .bodies=bodies
            }
        );
    }

    cudaFree(gpu_positions);
    cudaFree(gpu_velocities);
    cudaFree(gpu_weights);

    delete[] positions;
    delete[] velocities;
    delete[] weigths;
    return solution;
}


} // namespace n_body_problem_solver
