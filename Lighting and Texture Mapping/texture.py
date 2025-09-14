import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from OpenGL.arrays import vbo
from OpenGL.GL import shaders

import glm # pip install PyGLM

VERTEX_SHADER = """
#version 430
 
    layout(location = 0) in vec4 position;
    layout(location = 1) in vec2 texcoord;
    uniform mat4 MVP;

    out vec2 vt_texcoord;
    void main() {
        gl_Position = MVP * position;
        vt_texcoord = texcoord;
        
    } 
"""

FRAGMENT_SHADER = """
#version 430
    in vec2 vt_texcoord;
    uniform sampler2D tex0;
    
    void main() {
    
        gl_FragColor = texture(tex0, vt_texcoord);
 
    }
"""

shaderProgram = None
VAO = None
tex = None

def quad():

    positions, texcoords = [], []

    positions.append([-100.0, 100.0, 0.0, 1.0])
    texcoords.append([0, 0.0])
    positions.append([100.0, 100.0, 0.0, 1.0])
    texcoords.append([10.0, 0.0])
    positions.append([100.0, -100.0, 0.0, 1.0])
    texcoords.append([10.0, 10.0])
    positions.append([-100.0, -100.0, 0.0, 1.0])
    texcoords.append([0, 10.0])

    positions = np.array(positions, np.float32)
    texcoords = np.array(texcoords, np.float32)

    return positions, texcoords



def initliaze():
    global VERTEXT_SHADER
    global FRAGMEN_SHADER
    global shaderProgram
    global VAO
    global tex

    vertexshader = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragmentshader = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    shaderProgram = shaders.compileProgram(vertexshader, fragmentshader)

    points, texcoords = quad()
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, points.nbytes+texcoords.nbytes, None, GL_STATIC_DRAW)
    glBufferSubData(GL_ARRAY_BUFFER, 0, points.nbytes, points)
    glBufferSubData(GL_ARRAY_BUFFER, points.nbytes, texcoords.nbytes, texcoords)


    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, None)
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(points.nbytes))
    glEnableVertexAttribArray(1)

    tex = glGenTextures(1)
    img = np.array(Image.open('chessboard.jpg'))

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    #tag使用mipmap
    glGenerateMipmap(GL_TEXTURE_2D)

    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT )
    #tag使用线性过滤
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    #tag启用各向异性过滤
    max_anisotropy = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, max_anisotropy)


def calc_mvp(width, height):
    proj = glm.perspective(glm.radians(60.0),float(width)/float(height), 1,500.0)
    view = glm.lookAt(glm.vec3(0.0,10.0,100.0), glm.vec3(0,0,0),glm.vec3(0,1,0))

    model =  glm.mat4(1.0)
    model = glm.rotate(model, glm.radians(90), glm.vec3(1, 0, 0))

    mvp = proj * view * model

    return mvp


def save_image(width, height):
    # 读取当前OpenGL帧缓冲内容
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

    # 将数据转换为PIL图像格式（OpenGL的坐标系是左下角为原点，图像坐标系是左上角为原点，需翻转）
    img_data = np.frombuffer(data, dtype=np.uint8)
    img_data = img_data.reshape((height, width, 3))
    img_data = np.flipud(img_data)  # 翻转图片数据，以适应PIL格式

    # 使用PIL保存图像
    img = Image.fromarray(img_data)
    img.save("output_image.png")
    print("Image saved as output_image.png")

def render():
    global shaderProgram
    global VAO
    global tex

    glClearColor(0, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    #glEnable(GL_CULL_FACE)
    glEnable(GL_MULTISAMPLE)

    #tag启用抗锯齿功能
    glEnable(GL_POLYGON_SMOOTH)
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

    #tag启用纹理抗锯齿
    glEnable(GL_TEXTURE_2D)


    glUseProgram(shaderProgram)

    mvp_loc = glGetUniformLocation(shaderProgram,"MVP")
    mvp_mat = calc_mvp(640, 480)
    glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, glm.value_ptr(mvp_mat))

    glActiveTexture(GL_TEXTURE0)
    glUniform1i(glGetUniformLocation(shaderProgram, "tex0"), 0)

    glBindVertexArray(VAO)
    glDrawArrays(GL_QUADS, 0, 4)

    glUseProgram(0)

    save_image(640,480)

    glutSwapBuffers()


def main():

    glutInit([])

    #tag启用多重采样
    glutSetOption(GLUT_MULTISAMPLE, 8)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE )

    glutInitWindowSize(640, 480)
    glutCreateWindow(b"pyopengl with glut")
    initliaze()
    glutDisplayFunc(render)

    glutMainLoop()


if __name__ == '__main__':
    main()
