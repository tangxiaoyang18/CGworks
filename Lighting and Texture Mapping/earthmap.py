import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from pyrr import Matrix44
from PIL import Image
import time

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

            x = -cos_phi * sin_theta
            y = -cos_theta
            z = sin_phi * sin_theta

            # 计算纹理坐标
            u = j / segments
            v = i / rings

            vertices.extend([radius * x, radius * y, radius * z, x, y, z, u, v])

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
layout(location = 2) in vec2 aTexCoords;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // 世界空间坐标
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoords = aTexCoords;

    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

# 片段着色器
FRAGMENT_SHADER = """
#version 430

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

out vec4 FragColor;

uniform sampler2D textureMap;

uniform vec3 lightPos;
uniform vec3 viewPos;

void main()
{
    vec3 normal = normalize(Normal);

    // 计算光照方向
    vec3 lightDir = normalize(lightPos - FragPos);

    // 环境光
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * vec3(1.0);

    // 漫反射
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0);

    // 镜面反射
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = vec3(0.5) * spec;

    // 从纹理中获取颜色
    vec3 objectColor = texture(textureMap, TexCoords).rgb;

    // 合并光照结果
    vec3 result = (ambient + diffuse + specular) * objectColor;

    FragColor = vec4(result, 1.0);
}
"""

def load_texture(filename):
    image = Image.open(filename)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image = image.convert("RGB")
    img_data = image.tobytes()

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)

    # 设置纹理参数
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return texture


def set_uniform_mat4(shader, name, matrix):
    location = glGetUniformLocation(shader, name)
    glUniformMatrix4fv(location, 1, GL_FALSE, matrix)

def setuniformmat4(shader, name, matrix):
    location = glGetUniformLocation(shader, name)
    glUniformMatrix4fv(location, 1, GL_FALSE, matrix)

def save_frame_as_image(filename, width, height):
    # 创建一个适当大小的数组来存储像素数据
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

    # 将数据转换为PIL图像对象
    image = Image.frombytes("RGB", (width, height), pixels)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)  # OpenGL的坐标系和图像的坐标系不同，翻转图像

    # 保存图像
    image.save(filename)

def main():
    # 初始化GLFW
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "Bump Mapping", None, None)
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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))  # 位置
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(12))  # 法线
    glEnableVertexAttribArray(1)

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(24))  # 纹理坐标
    glEnableVertexAttribArray(2)

    glBindVertexArray(0)

    # 加载纹理
    textureMap = load_texture("earthmap.jpg")  # 颜色纹理

    # 投影矩阵
    projection = Matrix44.perspective_projection(45.0, 800 / 600, 0.1, 100.0)
    view = Matrix44.look_at(
        eye=[0.5, 0.5, 2.5],
        target=[0.0, 0.0, 0.0],
        up=[0.0, 1.0, 0.0]
    )
    model = Matrix44.identity()

    rotation_speed = 0.0001

    # 主渲染循环
    last_save_time = time.time()  # 记录最后一次保存时间
    save_interval_seconds = 1  # 每1秒保存一次图像

    # 主循环
    while not glfw.window_should_close(window):
        glfw.poll_events()

        # 清屏
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        model=Matrix44.from_eulers([0.0,0.0,rotation_speed])*model

        # 使用着色器
        glUseProgram(shader)

        # 绑定纹理
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, textureMap)
        glUniform1i(glGetUniformLocation(shader, "textureMap"), 0)

        # 设置uniform变量
        glUniform3fv(glGetUniformLocation(shader, "lightPos"), 1, [100, 500, 100])
        glUniform3fv(glGetUniformLocation(shader, "viewPos"), 1, [0.0, 0.0, 0.0])

        # 设置投影和视图矩阵
        setuniformmat4(shader, "projection", projection)
        setuniformmat4(shader, "view", view)
        setuniformmat4(shader, "model", model)

        # 绘制球体
        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        #glBindVertexArray(0)

        current_time = time.time()
        if current_time - last_save_time >= save_interval_seconds:
            save_frame_as_image("frame_{}.png".format(int(current_time)), 800, 600)
            last_save_time = current_time  # 更新保存时间

        glfw.swap_buffers(window)

    # 清理资源
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteBuffers(1, [EBO])
    glDeleteProgram(shader)
    glfw.terminate()

if __name__ == "__main__":
    main()
