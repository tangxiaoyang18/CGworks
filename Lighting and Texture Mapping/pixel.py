import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from pyrr import Matrix44
from PIL import Image
# 创建球体网格
def create_sphere(radius, segments, rings):
    vertices = []
    indices = []
    for i in range(rings + 1):
        theta = i * np.pi / rings
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(segments + 1):
            phi = j * 2 * np.pi / segments
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            x = cos_phi * sin_theta
            y = cos_theta
            z = sin_phi * sin_theta
            vertices.extend([radius * x, radius * y, radius * z, x, y, z])

            if i < rings and j < segments:
                a = i * (segments + 1) + j
                b = a + segments + 1
                indices.extend([a, b, a + 1, b, b + 1, a + 1])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

# 顶点着色器
VERTEX_SHADER = """
#version 430

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 VertexNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // 世界空间坐标
    FragPos = vec3(model * vec4(aPos, 1.0));
    VertexNormal = normalize(mat3(transpose(inverse(model))) * aNormal);

    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

# 片段着色器
FRAGMENT_SHADER = """
#version 430

in vec3 FragPos;
in vec3 VertexNormal;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 viewPos;
uniform vec3 objectColor;
uniform float samplingFactor;

void main()
{
    // 模拟像素采样位置调整
    vec2 pixelPosition = gl_FragCoord.xy * samplingFactor;

    // 法线和光照计算
    vec3 norm = normalize(VertexNormal);
    vec3 lightDir = normalize(lightPos - FragPos);

    // 环境光
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // 漫反射
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // 镜面反射
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;

    FragColor = vec4(result, 1.0);
}
"""

def save_image(width, height):
    # 读取当前OpenGL帧缓冲内容
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

    # 将数据转换为PIL图像格式（OpenGL的坐标系是左下角为原点，图像坐标系是左上角为原点，需翻转）
    img_data = np.frombuffer(data, dtype=np.uint8)
    img_data = img_data.reshape((height, width, 3))
    img_data = np.flipud(img_data)  # 翻转图片数据，以适应PIL格式

    # 使用PIL保存图像
    img = Image.fromarray(img_data)
    img.save("pixel.png")
    print("Image saved as output_image.png")

def main():
    # 初始化GLFW
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "Pixel Sampling", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # 初始化OpenGL
    glEnable(GL_DEPTH_TEST)

    # 编译着色器
    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

    # 创建球体网格
    vertices, indices = create_sphere(1.0, 64, 32)

    # 创建VAO、VBO、EBO
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # 设置顶点属性
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)

    glBindVertexArray(0)

    # 光照和对象颜色
    lightPos = np.array([1.2, 1.0, 2.0], dtype=np.float32)
    lightColor = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    objectColor = np.array([1.0, 0.5, 0.31], dtype=np.float32)
    viewPos = np.array([0.0, 0.0, 3.0], dtype=np.float32)

    # 投影矩阵
    projection = Matrix44.perspective_projection(45.0, 800 / 600, 0.1, 100.0)
    view = Matrix44.look_at(
        eye=viewPos,
        target=[0.0, 0.0, 0.0],
        up=[0.0, 1.0, 0.0]
    )
    model = Matrix44.identity()



    # 主循环
    sampling_factor = 1.0  # 初始采样因子

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # 动态调整采样因子（可根据需求修改）
        sampling_factor = max(0.5, min(1.0, 1.0 / np.linalg.norm(viewPos)))

        # 清屏
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 使用着色器
        glUseProgram(shader)

        # 设置uniform变量
        glUniform1f(glGetUniformLocation(shader, "samplingFactor"), sampling_factor)
        glUniform3fv(glGetUniformLocation(shader, "lightPos"), 1, lightPos)
        glUniform3fv(glGetUniformLocation(shader, "viewPos"), 1, viewPos)
        glUniform3fv(glGetUniformLocation(shader, "lightColor"), 1, lightColor)
        glUniform3fv(glGetUniformLocation(shader, "objectColor"), 1, objectColor)

        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model)
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, projection)

        # 绘制球体
        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glfw.swap_buffers(window)

    save_image(800, 640)
    # 清理资源
    glfw.terminate()

if __name__ == "__main__":
    main()
