import numpy as np
from PIL import Image  # pip install pillow
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
import glm  # pip install PyGLM
import OpenGL.GL as gl
import ctypes
width=800
length=600

VERTEX_SHADER = """
// 顶点着色器
#version 430

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 texcoord;
layout(location = 2) in vec3 norm;

uniform mat4 MVP;
uniform mat4 M; // 法线变换矩阵
uniform mat4 lightSpaceMatrix;

out vec2 vt_texcoord;
out vec3 fragPosition;
out vec3 fragNormal;
out vec4 ShadowCoords;

void main() {
    gl_Position = MVP * position;
    fragPosition = (M * position).xyz;
    fragNormal = normalize(mat3(M) * norm);
    vt_texcoord = texcoord;
    ShadowCoords = lightSpaceMatrix * position;
}
"""

FRAGMENT_SHADER = """
#version 430

uniform vec3 lightPos; // 光源在世界坐标下的位置
uniform vec3 lightColor; // 光源的颜色
uniform sampler2D shadowMap; // 深度贴图
uniform bool isShadowPass; // 控制是否是阴影传递
uniform vec3 materialSpecular;
uniform vec3 lightSpecular;

in vec3 fragPosition; // 片段位置
in vec3 fragNormal; // 片段法线
in vec2 vt_texcoord; // 片段纹理坐标
in vec4 ShadowCoords; // 片段在光源空间中的坐标

out vec4 FragColor;

float ShadowCalculation(vec4 fragPosLightSpace)
{
    // 执行透视除法
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;

    // 将投影坐标从 [-1,1] 映射到 [0,1]
    projCoords = projCoords * 0.5 + 0.5;

    // 获取最近片段的深度（使用xy分量）
    float closestDepth = texture(shadowMap, projCoords.xy).r;

    // 获取当前片段的深度
    float currentDepth = projCoords.z;

    // 检查当前片段是否在阴影中
    float bias = 0.005; // 解决自我遮挡的问题
    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    //shadow =  closestDepth==1 ? 1.0 : 0.0;

    return shadow;
}

void main() {

    if (isShadowPass) {
        // 不需要输出颜色，因为我们在渲染深度值
        discard;
    } else {
        vec3 norm = normalize(fragNormal);
        vec3 lightDir = normalize(lightPos - fragPosition);
        float distance = length(lightPos - fragPosition);
        float attenuation = 1.0 / (distance * distance);

        // 环境光
        float ambientStrength = 0.01;
        vec3 ambient = ambientStrength * lightColor;

        // 漫反射
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = lightColor * attenuation * diff;

        vec3 kd = vec3(0.5, 0.5, 0.5); // 设置一个简单的颜色

        // Specular
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(max(dot(lightDir, reflectDir), 0.0), 32);
        vec3 specular = lightSpecular * (spec * materialSpecular); 


        vec3 color = ambient + kd * diffuse + specular;

        // 计算阴影因子
        float shadow = ShadowCalculation(ShadowCoords);

        // 应用阴影
        color *= (1.0 - shadow); // 如果片段在阴影中，则降低亮度
        FragColor = vec4(color, 1.0);
    }
}
"""

FRAGMENT_SHADER1 = """
#version 430

uniform vec3 lightPos; // 光源在世界坐标下的位置
uniform vec3 lightColor; // 光源的颜色
uniform bool isShadowPass; // 控制是否是阴影传递
uniform vec3 materialSpecular;
uniform vec3 lightSpecular;

in vec3 fragPosition; // 片段位置
in vec3 fragNormal; // 片段法线
in vec2 vt_texcoord; // 片段纹理坐标
in vec4 ShadowCoords; // 片段在光源空间中的坐标

out vec4 FragColor;



void main() {

    if (isShadowPass) {
        // 不需要输出颜色，因为我们在渲染深度值
        discard;
    } else {
        vec3 norm = normalize(fragNormal);
        vec3 lightDir = normalize(lightPos - fragPosition);
        float distance = length(lightPos - fragPosition);
        float attenuation = 1.0 / (distance * distance);

        // 环境光
        float ambientStrength = 0.01;
        vec3 ambient = ambientStrength * lightColor;

        // 漫反射
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = lightColor * attenuation * diff;

        vec3 kd = vec3(1, 1, 1); // 设置一个简单的颜色

        // Specular
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(max(dot(lightDir, reflectDir), 0.0), 32);
        vec3 specular = lightSpecular * (spec * materialSpecular); 


        vec3 color = ambient + kd * diffuse + specular;

        // 计算阴影因子
        //float shadow = ShadowCalculation(ShadowCoords);

        // 应用阴影
        //color *= (1.0 - shadow); // 如果片段在阴影中，则降低亮度
        FragColor = vec4(color, 1.0);
    }
}
"""

shaderProgram = None
VAO = None
plane_VAO = None
Num_T = 0
Num_Planes = 0

def compile_shader(source, shader_type):
    shader_id = glCreateShader(shader_type)
    glShaderSource(shader_id, source)
    glCompileShader(shader_id)
    if not glGetShaderiv(shader_id, GL_COMPILE_STATUS):
        error_message = glGetShaderInfoLog(shader_id)
        raise RuntimeError(f"Shader compilation failed: {error_message.decode('utf-8')}")
    return shader_id

def sphere(radius=1.0, slices=64, stacks=32):
    points, texcoords, norms = [], [], []

    for i in range(stacks + 1):
        phi = i / stacks * np.pi
        for j in range(slices + 1):
            theta = j / slices * 2 * np.pi
            
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            
            u = j / slices
            v = i / stacks
            
            points.extend([x, y, z, 1.0])
            texcoords.extend([u, v])
            norms.extend([x, y, z])  # 法线方向与顶点位置相同
    
    indices = []
    for i in range(stacks):
        for j in range(slices):
            p0 = i * (slices + 1) + j
            p1 = p0 + 1
            p2 = (i + 1) * (slices + 1) + j
            p3 = p2 + 1
            indices.extend([p0, p2, p1])
            indices.extend([p1, p2, p3])
    
    return np.array(points, np.float32), np.array(texcoords, np.float32), np.array(norms, np.float32), np.array(indices, np.uint32)

def plane(width=10.0, height=10.0):
    points = [
        -width/2, -1, -height/2,
         width/2, -1, -height/2,
         width/2, -1,  height/2,
        -width/2, -1,  height/2
    ]
    texcoords = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    norms = [0.0, 1.0, 0.0] * 4
    indices = [0, 1, 2, 0, 2, 3]

    return np.array(points, np.float32), np.array(texcoords, np.float32), np.array(norms, np.float32), np.array(indices, np.uint32)

def calc_mvp(width, height):
    proj = glm.perspective(glm.radians(60.0), float(width) / float(height), 0.1, 40.0)
    view = glm.lookAt(glm.vec3(10, 2, 5), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    model = glm.mat4(1.0)
    mvp = proj * view * model
    return model, mvp


def calc_lmvp(width, height):
    proj = glm.perspective(glm.radians(60.0), float(width) / float(height), 0.1, 40.0)
    view =  glm.lookAt(glm.vec3(3, 4, 0), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
    model = glm.mat4(1.0)
    mvp = proj * view * model
    return model, mvp

def initliaze():
    global VERTEX_SHADER
    global FRAGMENT_SHADER
    global shaderProgram
    global shaderProgram1
    global VAO
    global plane_VAO
    global Num_T
    global Num_Planes

    vertexshader = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragmentshader = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    fragmentshader1 = compile_shader(FRAGMENT_SHADER1, GL_FRAGMENT_SHADER)

    shaderProgram = shaders.compileProgram(vertexshader, fragmentshader)
    shaderProgram1 = shaders.compileProgram(vertexshader, fragmentshader1)

    # 球体的数据
    points, texcoords, norms, indices = sphere(radius=1, slices=60, stacks=60)
    Num_T = indices.size
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, points.nbytes + texcoords.nbytes + norms.nbytes, None, GL_STATIC_DRAW)
    glBufferSubData(GL_ARRAY_BUFFER, 0, points.nbytes, points)
    glBufferSubData(GL_ARRAY_BUFFER, points.nbytes, texcoords.nbytes, texcoords)
    glBufferSubData(GL_ARRAY_BUFFER, points.nbytes + texcoords.nbytes, norms.nbytes, norms)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(points.nbytes))
    glEnableVertexAttribArray(1)

    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(points.nbytes + texcoords.nbytes))
    glEnableVertexAttribArray(2)

    # 地面的数据
    plane_points, plane_texcoords, plane_norms, plane_indices = plane()
    Num_Planes = plane_indices.size

    plane_VAO = glGenVertexArrays(1)
    glBindVertexArray(plane_VAO)

    plane_VBO = glGenBuffers(1)
    plane_EBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, plane_VBO)
    glBufferData(GL_ARRAY_BUFFER, plane_points.nbytes + plane_texcoords.nbytes + plane_norms.nbytes, None, GL_STATIC_DRAW)
    glBufferSubData(GL_ARRAY_BUFFER, 0, plane_points.nbytes, plane_points)
    glBufferSubData(GL_ARRAY_BUFFER, plane_points.nbytes, plane_texcoords.nbytes, plane_texcoords)
    glBufferSubData(GL_ARRAY_BUFFER, plane_points.nbytes + plane_texcoords.nbytes, plane_norms.nbytes, plane_norms)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, plane_EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, plane_indices.nbytes, plane_indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(plane_points.nbytes))
    glEnableVertexAttribArray(1)

    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(plane_points.nbytes + plane_texcoords.nbytes))
    glEnableVertexAttribArray(2)

    # 创建深度贴图
    global depthMap, depthMapFBO

    # 创建深度贴图纹理
    depthMap = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depthMap)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, length, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)

    # 设置纹理参数
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    # 创建帧缓冲对象
    depthMapFBO = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0)

    # 禁用颜色缓冲区
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)


    # 解绑帧缓冲对象
    glBindFramebuffer(GL_FRAMEBUFFER, 0)


def render():
    glClearColor(0, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glUseProgram(shaderProgram1)
    # 绑定深度贴图帧缓冲区
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO)
    glViewport(0, 0, width, length)
    glClear(GL_DEPTH_BUFFER_BIT)
    # 设置光源视角的投影和视图矩阵
    lm_mat, lmvp_mat=calc_lmvp(width, length)
    light_space_matrix =lmvp_mat
    # 从光源视角渲染场景
    mvp_loc = glGetUniformLocation(shaderProgram1, "MVP")
    m_loc = glGetUniformLocation(shaderProgram1, "M")
    glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, glm.value_ptr(lmvp_mat))
    glUniformMatrix4fv(m_loc, 1, GL_FALSE, glm.value_ptr(lm_mat))
    light_space_loc = glGetUniformLocation(shaderProgram1, "lightSpaceMatrix")
    glUniformMatrix4fv(light_space_loc, 1, GL_FALSE, glm.value_ptr(light_space_matrix))
    glUniform1i(glGetUniformLocation(shaderProgram1, "isShadowPass"), 0)  # 设置为阴影传递
    # 传递光源位置和颜色
    lightPos = np.array([3.0, 4.0, 0.0], np.float32)
    lightColor = np.array([17.0, 17.0, 17.0], np.float32)
    glUniform3f(glGetUniformLocation(shaderProgram1, "lightPos"), lightPos[0], lightPos[1], lightPos[2])
    glUniform3f(glGetUniformLocation(shaderProgram1, "lightColor"), lightColor[0], lightColor[1], lightColor[2])
    # 绘制球体
    model = glm.mat4(1.0)
    model_loc = glGetUniformLocation(shaderProgram1, "model")
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, Num_T, GL_UNSIGNED_INT, None)
    # 绘制平面
    model = glm.translate(glm.mat4(1.0), glm.vec3(0, -1.0, 0))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
    glBindVertexArray(plane_VAO)
    glDrawElements(GL_TRIANGLES, Num_Planes, GL_UNSIGNED_INT, None)
    glUseProgram(0)
    # 绑定默认帧缓冲区
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glViewport(0, 0, width, length)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # 从摄像机视角渲染场景
    glUseProgram(shaderProgram)
    mvp_loc = glGetUniformLocation(shaderProgram, "MVP")
    m_loc = glGetUniformLocation(shaderProgram, "M")
    m_mat, mvp_mat = calc_mvp(800, 600)
    glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, glm.value_ptr(mvp_mat))
    glUniformMatrix4fv(m_loc, 1, GL_FALSE, glm.value_ptr(m_mat))
    light_space_loc = glGetUniformLocation(shaderProgram, "lightSpaceMatrix")
    glUniformMatrix4fv(light_space_loc, 1, GL_FALSE, glm.value_ptr(light_space_matrix))
    glUniform1i(glGetUniformLocation(shaderProgram, "isShadowPass"), 0)
    # 传递光源位置和颜色
    lightPos = np.array([3.0, 4.0, 0.0], np.float32)
    lightColor = np.array([17.0, 17.0, 17.0], np.float32)
    glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), lightPos[0], lightPos[1], lightPos[2])
    glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), lightColor[0], lightColor[1], lightColor[2])
    # 传递深度贴图
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, depthMap)
    glUniform1i(glGetUniformLocation(shaderProgram, "shadowMap"), 0)
    # 设置为非阴影传递
    # 绘制球体
    model = glm.mat4(1.0)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, Num_T, GL_UNSIGNED_INT, None)
    # 绘制平面
    model = glm.translate(glm.mat4(1.0), glm.vec3(0, -1.0, 0))
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
    glBindVertexArray(plane_VAO)
    glDrawElements(GL_TRIANGLES, Num_Planes, GL_UNSIGNED_INT, None)
    glUseProgram(0)
    save_image()
    glutSwapBuffers()

def save_image():
    from PIL import Image
    x, y, width, height = 0, 0, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    data = gl.glReadPixels(x, y, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("output.png")

def main():
    glutInit([])
    glutSetOption(GLUT_MULTISAMPLE, 8)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)
    glutInitWindowSize(width, length)
    glutCreateWindow(b"pyopengl with glut")
    initliaze()
    glutDisplayFunc(render)
    glutMainLoop()

if __name__ == '__main__':
    main()