import numpy as np
import matplotlib.pyplot as plt


def de_casteljau(P0, P1, P2, P3, t):
    Q0 = (1 - t) * P0 + t * P1
    Q1 = (1 - t) * P1 + t * P2
    Q2 = (1 - t) * P2 + t * P3
    R0 = (1 - t) * Q0 + t * Q1
    R1 = (1 - t) * Q1 + t * Q2
    R = (1 - t) * R0 + t * R1
    return R


def bezier_curve(P0, P1, P2, P3, num_points=100):
    curve_points = []
    for t in np.linspace(0, 1, num_points):
        point = de_casteljau(P0, P1, P2, P3, t)
        curve_points.append(point)
    return np.array(curve_points)


def generate_control_points(radius, num_segments=25):
    control_points = []
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments
        control_points.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(control_points)


def draw_bezier_circle(radius, num_segments=25):
    control_points = generate_control_points(radius, num_segments)
    bezier_points = []

    for i in range(num_segments):
        P0 = control_points[i]
        P1 = (P0 + control_points[(i + 1) % num_segments]) / 2
        P2 = (P1 + control_points[(i + 2) % num_segments]) / 2
        P3 = control_points[(i + 3) % num_segments]

        bezier_points.append(bezier_curve(P0, P1, P2, P3))

    bezier_points = np.vstack(bezier_points)

    plt.plot(bezier_points[:, 0], bezier_points[:, 1], label='Bezier Curve', color='blue')
    plt.scatter(control_points[:, 0], control_points[:, 1], color='red', label='Control Points', zorder=5)
    for i in range(num_segments):
        plt.plot([control_points[i, 0], control_points[(i + 1) % num_segments, 0]],
                 [control_points[i, 1], control_points[(i + 1) % num_segments, 1]], 'r--')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Bezier Approximation of a Circle with {num_segments} Segments')
    plt.legend()
    plt.grid(True)
    plt.show()


# 调用函数
radius = 1
num_segments = 25
draw_bezier_circle(radius, num_segments)

