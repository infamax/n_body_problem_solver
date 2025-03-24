import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import grpc 

from dataclasses import dataclass

from n_body_problem_pb2 import NBodyProblemData, Body, Position, Velocity, CalculationMode, NBodyProblemSolution
from n_body_problem_pb2_grpc import NBodyProblemSolverStub

@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class TrajectoryAnimation:
    def __init__(self, trajectories: np.ndarray):
        self._trajectories = trajectories
        self._count_trajectories = trajectories.shape[0]


    def step(self, number_step: int) -> tuple[np.ndarray, np.ndarray]:
        x_points = []
        y_points = []
        one_trajectory = self._trajectories[number_step]
        for body_trajectory in one_trajectory:
            x_points.append(body_trajectory[0])
            y_points.append(body_trajectory[1])
        return (x_points, y_points)


    def get_bounding_box(self) -> BoundingBox:
        x_min = self._trajectories[:, :, 0].min()
        x_max = self._trajectories[:, :, 0].max()
        y_min = self._trajectories[:, :, 1].min()
        y_max = self._trajectories[:, :, 1].max()
        return BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )
    
    
    @property
    def count_trajectories(self) -> int:
        return self._count_trajectories


def generate_data(num_bodies: int, dt: float, num_iterations: int) -> NBodyProblemData:
    res = NBodyProblemData()
    res.dt = dt
    res.G = 1.0
    res.num_iterations = num_iterations
    res.calculation_mode = CalculationMode.GPU

    for idx in range(num_bodies):
        position = np.random.uniform(low=0, high=200.0, size=3)
        velocity = np.random.uniform(low=0, high=1000.0, size=3)
        res.bodies.append(
            Body(
                name=f"body_{idx}",
                position=Position(
                    x=position[0],
                    y=position[1],
                    z=position[2],
                ),
                velocity=Velocity(
                    x=velocity[0],
                    y=velocity[1],
                    z=velocity[2],
                ),
                weight=np.random.uniform(low=0, high=10000, size=1)[0]
            )
        )

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-bodies", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num-iterations", type=int, default=100)

    args = parser.parse_args()
    channel = grpc.insecure_channel("localhost:9999")
    client = NBodyProblemSolverStub(channel)

    n_body_problem_data = generate_data(
        num_bodies=args.num_bodies,
        dt=args.dt,
        num_iterations=args.num_iterations,
    )

    future = client.solve.future(n_body_problem_data)
    res: NBodyProblemSolution = future.result()

    trajectories = []

    for iteration in res.iterations:
        trajectory = []
        for body in iteration.bodies:
            trajectory.append([body.position.x, body.position.y])
        trajectories.append(trajectory)
    
    trajectories = np.array(trajectories)
    trajectory_animation = TrajectoryAnimation(trajectories)
    bounding_box = trajectory_animation.get_bounding_box()

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                        xlim=(bounding_box.x_min, bounding_box.x_max), ylim=(bounding_box.y_min, bounding_box.y_max))
    particles, = ax.plot([], [], 'bo', ms=6)
    particles = particles

    # rect is the box edge
    rect = plt.Rectangle((bounding_box.x_min, bounding_box.y_min),
                        bounding_box.x_max - bounding_box.x_min,
                        bounding_box.y_max - bounding_box.y_min,
                        ec='none', lw=2, fc='none')
    ax.add_patch(rect)

    def init():
        particles.set_data([], [])
        rect.set_edgecolor('none')
        return particles, rect

    def animate(number_step: int):
        trajectory = trajectory_animation.step(number_step)

        rect.set_edgecolor('k')
        particles.set_data(trajectory[0], trajectory[1])
        return particles, rect

    ani = animation.FuncAnimation(fig, animate, frames=trajectory_animation.count_trajectories,
                                interval=100, blit=True, init_func=init)
    writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist="Me"),
                                bitrate=1800)
    ani.save("n_body_problem_animation.gif", writer=writer)

        
    # print(res)



main()
    


