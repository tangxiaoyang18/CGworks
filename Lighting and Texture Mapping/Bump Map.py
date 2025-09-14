import numpy as np
from PIL import Image  # pip install pillow
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileShader, compileProgram
import glm  # pip install PyGLM
import time

# 球体生成函数
def draw_sphere(radius, latitude, longitude):
    # 球体顶点数据和纹理坐标
    vertice = []
    texcoord = []
    indices = []
    tangent = []
    for i in range(latitude + 1):
        lat = np.pi * i / latitude  # 纬度角度
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)

        for j in range(longitude + 1):
            lon = 2 * np.pi * j / longitude  # 经度角度
            sin_lon = np.sin(lon)
            cos_lon = np.cos(lon)

            # 计算顶点位置 (x, y, z)
            x = radius * cos_lon * sin_lat
            y = radius * cos_lat
            z = radius * sin_lon * sin_lat
            vertice.append([x, y, z, 1.0])

            # 计算纹理坐标 (u, v)
            u = 1 - j / longitude
            v = i / latitude
            texcoord.append([u, v])

            # 计算切线(沿纹理坐标u方向)
            tangent.append([-y, x, 0.0])

    # 生成索引，分割成四边形，再将每个四边形分成两个三角形
    for i in range(latitude):
        for j in range(longitude):
            # 计算四边形的4个顶点
            a = i * (longitude + 1) + j
            b = a + 1
            c = (i + 1) * (longitude + 1) + j
            d = c + 1

            # 第一三角形 (a, b, b)
            indices.extend([a, b, c])
            # 第二三角形 (b, d, c)
            indices.extend([b, d, c])

    points = []
    texcoords = []
    norms = []
    tangents = []
    for i in range(len(indices)):
        points.append(vertice[indices[i]])
        texcoords.append(texcoord[indices[i]])
        norms.append(vertice[indices[i]][:3])
        tangents.append(tangent[indices[i]])

    # 将数据转换为 numpy 数组
    points = np.array(points, np.float32)
    texcoords = np.array(texcoords, np.float32)
    norms = np.array(norms, np.float32)
    tangents = np.array(tangents, np.float32)

    return points, texcoords, norms, tangents

# 顶点着色器代码
VERTEX_SHADER = """
#version 430
layout(location = 0) in vec4 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec3 norm;
layout(location = 3) in vec3 T; // Tangent vector

uniform mat4 MVP;
uniform mat4 M;

out vec2 vt_texcoord;
out vec3 fragPos; // Fragment position in world space
out vec3 vt_norm; // Normal vector
out vec3 vt_T;    // Tangent vector

void main() {
    gl_Position = MVP * position;
    fragPos = (M * position).xyz;
    vt_norm = normalize(norm);
    vt_T = normalize(T);
    vt_texcoord = texcoord;
}
"""

# 片段着色器代码
FRAGMENT_SHADER = """
#version 430
uniform sampler2D tex0; // Diffuse texture
uniform sampler2D tex1; // Bump map
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 eyePos;

in vec3 fragPos;
in vec3 vt_norm;
in vec3 vt_T;
in vec2 vt_texcoord;

void main() {
    vec3 vt_B = cross(vt_norm, vt_T); // Bitangent vector
    float bumpScale = 0.01;
    float delta = 0.001;

    // Sample bump map and calculate height differences
    float bump = texture(tex1, vt_texcoord).r;
    float left = texture(tex1, vt_texcoord + vec2(-delta, 0.0)).r * bumpScale;
    float right = texture(tex1, vt_texcoord + vec2(delta, 0.0)).r * bumpScale;
    float up = texture(tex1, vt_texcoord + vec2(0.0, delta)).r * bumpScale;
    float down = texture(tex1, vt_texcoord + vec2(0.0, -delta)).r * bumpScale;

    float du = (left - right) / (2.0 * delta);
    float dv = (down - up) / (2.0 * delta);

    // Adjust the normal vector with bump map influence
    float bumpFactor = 0.5;
    vec3 adjustedNormal = vt_norm + bumpFactor * (du * vt_B - dv * vt_T);
    adjustedNormal = normalize(adjustedNormal);

    // Lighting calculations
    vec3 ambient = 0.1 * lightColor;

    vec3 lightDir = normalize(lightPos - fragPos);
    vec3 viewDir = normalize(eyePos - fragPos);
    vec3 reflectDir = reflect(-lightDir, adjustedNormal);

    float diff = max(dot(adjustedNormal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = 0.5 * spec * lightColor;

    vec3 textureColor = texture(tex0, vt_texcoord).rgb;
    vec3 result = (ambient + diffuse + specular) * textureColor;

    gl_FragColor = vec4(result, 1.0);
}
"""

# 全局变量
shaderProgram = None
VAO = None
vertexCount = 0
timeStep = 0
last_save_time = 0  # 上次保存图像的时间
frame_count = 0  # 用于命名图片的帧计数

# 初始化 OpenGL 环境
def initialize():
    global shaderProgram, VAO, vertexCount

    # 编译着色器程序
    vertex_shader = compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragment_shader = compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    shaderProgram = compileProgram(vertex_shader, fragment_shader)

    # 生成球体数据
    vertices, texcoords, normals, tangents = draw_sphere(1.0, 64, 64)
    vertexCount = vertices.shape[0]

    # 创建 VAO 和 VBO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    buffer_size = vertices.nbytes + texcoords.nbytes + normals.nbytes + tangents.nbytes
    glBufferData(GL_ARRAY_BUFFER, buffer_size, None, GL_STATIC_DRAW)

    # 分块填充顶点数据
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
    glBufferSubData(GL_ARRAY_BUFFER, vertices.nbytes, texcoords.nbytes, texcoords)
    glBufferSubData(GL_ARRAY_BUFFER, vertices.nbytes + texcoords.nbytes, normals.nbytes, normals)
    glBufferSubData(GL_ARRAY_BUFFER, vertices.nbytes + texcoords.nbytes + normals.nbytes, tangents.nbytes, tangents)

    # 设置顶点属性指针
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(vertices.nbytes))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(vertices.nbytes + texcoords.nbytes))
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(vertices.nbytes + texcoords.nbytes + normals.nbytes))
    glEnableVertexAttribArray(3)

    # 加载纹理
    load_texture('earthmap.jpg', GL_TEXTURE0, GL_RGB)
    load_texture('Bump.jpg', GL_TEXTURE1, GL_LUMINANCE)

# 加载纹理函数
def load_texture(filename, texture_unit, format):
    texture = glGenTextures(1)
    glActiveTexture(texture_unit)
    glBindTexture(GL_TEXTURE_2D, texture)

    image = np.array(Image.open(filename))
    glTexImage2D(GL_TEXTURE_2D, 0, format, image.shape[1], image.shape[0], 0, format, GL_UNSIGNED_BYTE, image)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

def save_frame_as_image(filename, width, height):
    # 创建一个适当大小的数组来存储像素数据
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

    # 将数据转换为PIL图像对象
    image = Image.frombytes("RGB", (width, height), pixels)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)  # OpenGL的坐标系和图像的坐标系不同，翻转图像

    # 保存图像
    image.save(filename)


# 渲染函数
def render():
    global shaderProgram, VAO, vertexCount, timeStep,last_save_time,frame_count

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shaderProgram)

    # 计算 MVP 矩阵
    proj = glm.perspective(glm.radians(45.0), 640/480, 0.1, 10.0)
    view = glm.lookAt(glm.vec3(2, 2, 2), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    model = glm.rotate(glm.mat4(1.0), glm.radians(timeStep), glm.vec3(0, 1, 0))
    mvp = proj * view * model

    # 设置 Uniform
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "MVP"), 1, GL_FALSE, glm.value_ptr(mvp))
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "M"), 1, GL_FALSE, glm.value_ptr(model))
    glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), 2.0, 2.0, 2.0)
    glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), 1.1, 1.1, 1.1)
    glUniform3f(glGetUniformLocation(shaderProgram, "eyePos"), 0.0, 0.0, 5.0)

    # 绘制
    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLES, 0,vertexCount)

    # 增加时间步长用于模型旋转
    timeStep += 0.01

    # 保存图像的逻辑
    current_time = time.time()
    if current_time - last_save_time >= 1.0:  # 每隔 1 秒保存一次
        filename = f"frame_{frame_count:04d}.png"
        save_frame_as_image(filename, 800, 600)
        print(f"Saved frame: {filename}")
        last_save_time = current_time
        frame_count += 1

    # 交换缓冲区以显示当前帧
    glutSwapBuffers()

    # 主函数

def main():
        # 初始化 GLUT 窗口
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(800, 600)
        glutCreateWindow(b"Bump Mapping")

        # 启用深度测试
        glEnable(GL_DEPTH_TEST)

        # 初始化 OpenGL
        initialize()

        # 注册渲染函数
        glutDisplayFunc(render)
        glutIdleFunc(render)

        # 开始主循环
        glutMainLoop()

    # 程序入口
if __name__ == "__main__":
    main()

