import numpy as np
import copy
import math
from PIL import Image
from sympy import interpolate
from scipy.spatial.transform import Rotation as R

def LookAt(eye, center, up):
    f = (np.array(center) - np.array(eye)) / np.linalg.norm(np.array(center) - np.array(eye))
    u = np.array(up) / np.linalg.norm(np.array(up))
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    view = np.identity(4)
    view[0, :3] = s
    view[1, :3] = u
    view[2, :3] = -f
    view[:3, 3] = -np.dot(np.array(eye), view[:3, :3])

    return view

def Ortho(left, right, bottom, top, near, far):
    ortho = np.identity(4)
    ortho[0, 0] = 2 / (right - left)
    ortho[1, 1] = 2 / (top - bottom)
    ortho[2, 2] = -2 / (far - near)
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)

    return ortho

def Perspective(fov, aspect=1.0, near=0.1, far=10.0):
    f = 1.0 / np.tan(np.radians(fov) / 2)
    depth = far - near

    proj = np.zeros((4, 4))
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / depth
    proj[2, 3] = 2 * far * near / depth
    proj[3, 2] = -1

    return proj

def rotation_around_axis(axis, theta):
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(np.radians(theta))
    sin_theta = np.sin(np.radians(theta))
    ux, uy, uz = axis

    rotation_matrix = np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ])

    return rotation_matrix

def quaternion_slerp(q1, q2, t):
    """在两个四元数之间进行球面线性插值"""
    q1 = np.array(q1)
    q2 = np.array(q2)
    # 计算四元数的点积
    dot_product = np.dot(q1, q2)
    # 如果点积为负，反转一个四元数以获得最短路径
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product
    # 如果四元数非常接近，使用线性插值
    if dot_product > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    # 计算插值
    theta_0 = np.arccos(dot_product)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    s1 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    return (s1 * q1) + (s2 * q2)

class Triangle(object):
    def __init__(self):
        self.vertices = np.zeros((3, 3))
        self.colors = np.zeros((3, 3))

    def setVertex(self, ind, x, y, z):
        self.vertices[ind] = [x, y, z]

    def setColor(self, ind, r, g, b):
        self.colors[ind] = [r, g, b]

    def rotate_norm(self, thea):
        A, B, C = self.vertices
        center = (A + B + C) / 3.0
        norm = np.cross(B - A, C - A)
        norm = norm / np.linalg.norm(norm)
        rotation_matrix = rotation_around_axis(norm, thea)
        self.vertices = np.dot(self.vertices - center, rotation_matrix.T) + center

    def inside(self, x, y, z):
        A, B, C = self.vertices
        v0 = C - A
        v1 = B - A
        v2 = np.array([x, y, z]) - A

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        w = 1.0 - u - v

        return u >= 0 and v >= 0 and w >= 0

    def to_homogeneous_coordinates(self):
        return np.hstack((self.vertices, np.ones((3, 1))))

    def rotate(self, rotation_matrix):#e
        """应用旋转矩阵到三角形的顶点"""
        self.vertices = (rotation_matrix @ self.vertices.T).T

    def rotate_with_quaternion(self, q):
        """应用四元数旋转到三角形的顶点"""
        rot = R.from_quat(q)
        self.vertices = rot.apply(self.vertices)
    def interpolate(self, other_triangle, t):
        """在两个三角形之间进行插值"""
        # 插值颜色
        new_colors = (1 - t) * self.colors + t * other_triangle.colors
        # 初始和最终旋转矩阵转换为四元数
        identity_quaternion = [1, 0, 0, 0]  # 初始没有旋转
        x_rotation = rotation_around_axis([1, 0, 0], 30)
        y_rotation = rotation_around_axis([0, 1, 0], 60)
        z_rotation = rotation_around_axis([0, 0, 1], 30)
        final_rotation_matrix = z_rotation @ y_rotation @ x_rotation
        final_quaternion = R.from_matrix(final_rotation_matrix).as_quat()
        # 插值四元数
        interpolated_quaternion = quaternion_slerp(identity_quaternion, final_quaternion, t)
        # 创建新的三角形
        new_triangle = Triangle()
        new_triangle.vertices = np.copy(self.vertices)  # 复制初始顶点位置
        new_triangle.colors = new_colors
        # 应用插值后的四元数旋转
        new_triangle.rotate_with_quaternion(interpolated_quaternion)
        return new_triangle


class Rasterization(object):
    def __init__(self, width, height):
        self.color_buf = np.zeros((height, width, 3))
        self.depth_buf = np.full((height, width), np.inf)
        self.view_m = np.eye(4)
        self.proj_m = np.eye(4)

    def setViewM(self, mat):
        self.view_m = mat

    def setProjM(self, mat):
        self.proj_m = mat

    def rasterize_triangle(self, t):
        H, W, _ = self.color_buf.shape

        v4 = t.to_homogeneous_coordinates()
        v4 = (self.proj_m @ self.view_m @ v4.T).T
        v4 = v4 / np.repeat(v4[:, 3:], 4, axis=1)

        v4 = v4[:, :3] * 0.5 + 0.5

        raster_t = Triangle()
        raster_t.setVertex(0, v4[0][0] * W, v4[0][1] * H, v4[0][2])
        raster_t.setVertex(1, v4[1][0] * W, v4[1][1] * H, v4[1][2])
        raster_t.setVertex(2, v4[2][0] * W, v4[2][1] * H, v4[2][2])

        for x in range(W):
            for y in range(H):
                if raster_t.inside(x, H - 1 - y, 0):
                    A, B, C = raster_t.vertices
                    v0, v1, v2 = B - A, C - A, np.array([x, H - 1 - y, 0]) - A
                    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
                    d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
                    denom = d00 * d11 - d01 * d01
                    v = (d11 * d20 - d01 * d21) / denom
                    w = (d00 * d21 - d01 * d20) / denom
                    u = 1.0 - v - w
                    z = u * A[2] + v * B[2] + w * C[2]
                    if z < self.depth_buf[y][x]:
                        self.depth_buf[y][x] = z
                        self.color_buf[y][x] = (u * t.colors[0] + v * t.colors[1] + w * t.colors[2])

    def render(self, t_list):
        for t in t_list:
            self.rasterize_triangle(t)

Ra = Rasterization(256, 256)
Ra.setViewM(LookAt(np.array([0, 0, 3]), np.array([0, 0, 0]), np.array([0, 1, 0])))
Ra.setProjM(Perspective(60))#Ortho

# 创建初始和终止状态的三角形
T_init = Triangle()
T_init.setVertex(0,0.0,0.0,0.0)
T_init.setVertex(1,0.0,1.0,0.0)
T_init.setVertex(2,1.0,0.0,0.0)
T_init.setColor(0,1.0,0.0,0.0)
T_init.setColor(1,1.0,0.0,0.0)
T_init.setColor(2,1.0,0.0,0.0)

T_final = Triangle()
T_final.setVertex(0,0.0,0.0,0.0)
T_final.setVertex(1,0.0,1.0,0.0)
T_final.setVertex(2,1.0,0.0,0.0)
T_final.setColor(0,0.0,1.0,0.0)
T_final.setColor(1,0.0,1.0,0.0)
T_final.setColor(2,0.0,1.0,0.0)

# 渲染t=0, 0.25, 0.5, 0.75, 1时刻的三角形
for t in [0, 0.25, 0.5, 0.75, 1]:
    interpolated_triangle = T_init.interpolate(T_final, t)
    Ra.render([interpolated_triangle])
    Image.fromarray((Ra.color_buf * 255).astype("uint8")).show()

#a(perspective/ortho)
'''
t = Triangle()
t.setVertex(0, 0, 1.0, 0.0)
t.setVertex(1, -1.0, 0.0, 1.0)
t.setVertex(2, 1.0, 0.0, 1.0)

t.setColor(0, 2.0, 2.5, 3.0)
t.setColor(1, 2.5, 3.0, 2.0)
t.setColor(2, 3.0, 2.0, 2.5)

R.render([t])
'''

#b
'''
t = Triangle()
t.setVertex(0, 0, 1.0, 0.0)
t.setVertex(1, -1.0, 0.0, 1.0)
t.setVertex(2, 1.0, 0.0, 1.0)

t.setColor(0, 2.0, 2.5, 3.0)
t.setColor(1, 2.5, 3.0, 2.0)
t.setColor(2, 3.0, 2.0, 2.5)

t=t.rotate(90)

R.render([t])
'''

#c
'''
t = Triangle()
t.setVertex(0, 0, 1.0, 0.0)
t.setVertex(1, -1.0, 0.0, 1.0)
t.setVertex(2, 1.0, 0.0, 1.0)

t.setColor(0, 2.0, 2.5, 3.0)
t.setColor(1, 2.5, 3.0, 2.0)
t.setColor(2, 3.0, 2.0, 2.5)

R.render([t])
'''

#d
'''
t1 = Triangle()
t1.setVertex(0, 1.0, 0.0, 0.0)
t1.setVertex(1, 0.0, 1.0, 0.0)
t1.setVertex(2, 0.0, 0.0, 1.0)
t1.setColor(0, 1.0, 0.0, 0.0)
t1.setColor(1, 1.0, 0.0, 0.0)
t1.setColor(2, 1.0, 0.0, 0.0)

t2 = Triangle()
t2.setVertex(0, 1.5, 0.0, 0.0)
t2.setVertex(1, 0.0, 1.5, 0.0)
t2.setVertex(2, 0.0,0.0, 1.5)
t2.setColor(0, 0.0, 1.0, 0.0)
t2.setColor(1, 0.0, 1.0, 0.0)
t2.setColor(2, 0.0, 1.0, 0.0)
R.render([t1, t2])
'''