#include <vector>
#include <string>

namespace n_body_problem_solver {

struct Body {
    std::string name = "";
    float position[3];
    float vecocities[3];
    float weight = 0.0;
};

struct NBodyProblemData {
    int num_bodies = 0;
    int num_iterations = 0;
    float dt = 0.1;
    std::vector<Body> bodies;
    float G = 1.0;
};

struct OneIterSolutionNBodyProblem {
    int num_iteration = 0;
    std::vector<Body> bodies;
};

struct SolutionNBodyProblem {
    std::vector<OneIterSolutionNBodyProblem> iterations;
};

SolutionNBodyProblem gpu_solve_n_body_problem(const NBodyProblemData& data);

} // namespace n_body_problem_solver
