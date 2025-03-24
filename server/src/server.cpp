#include <iostream>
#include <memory>

#include <grpcpp/grpcpp.h>

#include "proto/n_body_problem.pb.h"
#include "proto/n_body_problem.grpc.pb.h"
#include "server/gpu/n_body_problem_solver.h"


class NBodyProblemSolverService : public NBodyProblemSolver::Service {
    grpc::Status solve(grpc::ServerContext* context, const NBodyProblemData* request, NBodyProblemSolution* response) override {
        const auto calculation_mode = request->calculation_mode();
        switch (calculation_mode) {
            case CalculationMode::CPU: {
                throw std::runtime_error("Not implemented");
            }

            case CalculationMode::GPU: {
                n_body_problem_solver::NBodyProblemData data;
                data.num_bodies = request->bodies().size();
                data.dt = request->dt();
                data.G = request->g();
                data.num_iterations = request->num_iterations();
                data.bodies.reserve(data.num_bodies);

                for (const auto& body : request->bodies()) {
                    data.bodies.push_back(
                        n_body_problem_solver::Body{
                            .name = body.name(),
                            .position = {body.position().x(), body.position().y(), body.position().z()},
                            .vecocities = {body.velocity().x(), body.velocity().y(), body.velocity().z()},
                            .weight = body.weight()
                        }
                    );
                }

                const auto solution = n_body_problem_solver::gpu_solve_n_body_problem(data);
                for (const auto& iter : solution.iterations) {
                    auto proto_iter = response->add_iterations();
                    proto_iter->set_num_iteration(iter.num_iteration);
                    for (const auto& body : iter.bodies) {
                        auto proto_body = proto_iter->add_bodies();
                        proto_body->set_name(body.name);
                        proto_body->set_weight(body.weight);
                        Velocity vel;
                        vel.set_x(body.vecocities[0]);
                        vel.set_y(body.vecocities[1]);
                        vel.set_z(body.vecocities[2]);
                        *proto_body->mutable_velocity() = vel;
                        Position pos;
                        pos.set_x(body.position[0]);
                        pos.set_y(body.position[1]);
                        pos.set_z(body.position[2]);
                        *proto_body->mutable_position() = pos;
                    }
                }
                return grpc::Status::OK;
            }
        }

        return grpc::Status::OK;
    }
};


int main() {
    NBodyProblemSolverService service;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("[::]:9999", grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    server->Wait();
    return 0;
}
