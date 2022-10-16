"""
Adapted from: https://github.com/BryceQing/OPENCV_AR
"""
from pathlib import Path
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
from android.android_camera import AndroidCamera
import cv2.aruco as aruco
from PIL import Image
import numpy as np

from opengl.objloader import * #Load obj and corresponding material and textures.
from opengl.matrixTrans import *

class Filter:
    def __init__(self):
        self.pre_trans_x = None
        self.pre_trans_y = None
        self.pre_trans_z = None
        
    def update(self, tvecs) -> bool:
        trans_x, trans_y, trans_z = tvecs[0][0][0], tvecs[0][0][1], tvecs[0][0][2]
        is_mark_move = False
        if self.pre_trans_x is not None:
            if abs(self.pre_trans_x - trans_x) > 0.001 or abs(self.pre_trans_y - trans_y) > 0.002 or abs(self.pre_trans_z - trans_z) > 0.015:
                dis_x = abs(self.pre_trans_x - trans_x)
                dis_y = abs(self.pre_trans_y - trans_y)
                dis_z = abs(self.pre_trans_z - trans_z)
                
                is_mark_move = True
        self.pre_trans_x, self.pre_trans_y, self.pre_trans_z = trans_x, trans_y, trans_z
        return is_mark_move

class AR_Render:
    
    def __init__(self, camera : AndroidCamera, obj_path : str, obj_scale=0.02) -> None:

        self.camera_matrix = np.load("camera_matrix.numpy", allow_pickle=True)
        self.dist_coeffs = np.load("camera_dist_coeffs.numpy", allow_pickle=True)
            
        self.device = camera.device
        self.cam_w, self.cam_h = map(int, (camera.device.get(3), camera.device.get(4)))  
        self.initOpengl(self.cam_w, self.cam_h)  
        self.model_scale = obj_scale
        
        self.projectMatrix = intrinsic2Project(self.camera_matrix, self.cam_w, self.cam_h, 0.01, 100.0)      
        self.obj_file = Path(obj_path)        
        self.loadModel(obj_path)
    
        # Model translate that you can adjust by key board 'w', 's', 'a', 'd'
        self.translate_x, self.translate_y, self.translate_z = 0, 0, 0
        self.pre_extrinsicMatrix = None
        
        self.filter = Filter()
        self.image = None
        

    def loadModel(self, object_path):
        
        """[loadModel from object_path]
        
        Arguments:
            object_path {[string]} -- [path of model]
        """
        
        print(object_path)
        self.model = OBJ(object_path, swapyz = True)

  
    def initOpengl(self, width, height, pos_x = 500, pos_y = 500, window_name = b'Aruco Demo'):
        
        """[Init opengl configuration]
        
        Arguments:
            width {[int]} -- [width of opengl viewport]
            height {[int]} -- [height of opengl viewport]
        
        Keyword Arguments:
            pos_x {int} -- [X cordinate of viewport] (default: {500})
            pos_y {int} -- [Y cordinate of viewport] (default: {500})
            window_name {bytes} -- [Window name] (default: {b'Aruco Demo'})
        """
        
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(pos_x, pos_y)
     
        self.window_id = glutCreateWindow(window_name)
        glutDisplayFunc(self.draw_scene)
        glutIdleFunc(self.draw_scene)
        
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glShadeModel(GL_SMOOTH)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        
        # # Assign texture
        glEnable(GL_TEXTURE_2D)
        
        # setup lighting
        glLight(GL_LIGHT0, GL_POSITION,  (2, 0, 2, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0, 0, 0, 0.1))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0.2, 0.2, 0.2, 0.5))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.2,0.2,0.2,0.5)) 
        
    
 
    def draw_scene(self):
        """[Opengl render loop]
        """
        if self.image is not None:
            self.draw_background(self.image)  # draw background
            # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.draw_objects(self.image, mark_size = 0.06) # draw the 3D objects.
            glutSwapBuffers()
    
 
    def draw_background(self, image):
        """[Draw the background and tranform to opengl format]
        
        Arguments:
            image {[np.array]} -- [frame from your camera]
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Setting background image project_matrix and model_matrix.
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(33.7, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
     
        # Convert image to OpenGL texture format
        bg_image = cv2.flip(image, 0)
        bg_image = Image.fromarray(bg_image)     
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)
  
        # Create background texture
        texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)
                
        glTranslatef(0.0,0.0,-10.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 4.0,  3.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-4.0,  3.0, 0.0)
        glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)
 
 
 
    def draw_objects(self, image, mark_size = 0.05):
        """[draw models with opengl]
        
        Arguments:
            image {[np.array]} -- [frame from your camera]
        
        Keyword Arguments:
            mark_size {float} -- [aruco mark size: unit is meter] (default: {0.07})
        """
        # aruco data
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)      
        parameters =  aruco.DetectorParameters_create()

        height, width, _channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
        
        rvecs, tvecs, model_matrix = None, None, None
        
        if ids is not None and corners is not None:
            rvecs, tvecs, _= aruco.estimatePoseSingleMarkers(corners, mark_size , self.camera_matrix, self.dist_coeffs)
            new_rvecs = rvecs[0,:,:]
            new_tvecs = tvecs[0,:,:]
            
            projectMatrix = intrinsic2Project(self.camera_matrix, width, height, 0.01, 100.0)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glMultMatrixf(projectMatrix)
            
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
        
            
            if tvecs is not None:
                if self.filter.update(tvecs): # the mark is moving
                    model_matrix = extrinsic2ModelView(rvecs, tvecs)
                else:
                    model_matrix = self.pre_extrinsicMatrix
            else:
                model_matrix =  self.pre_extrinsicMatrix
            
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE )
                
            if model_matrix is not None:     
                self.pre_extrinsicMatrix = model_matrix
                glLoadMatrixf(model_matrix)
                glScaled(self.model_scale, self.model_scale, self.model_scale)
                glTranslatef(self.translate_x, self.translate_y, self.translate_y)
                glCallList(self.model.gl_list)

            glDisable(GL_LIGHT0)
            glDisable(GL_LIGHTING)
            glDisable(GL_COLOR_MATERIAL)
                
            cv2.waitKey(20)
    
    def set_frame(self, frame):
        self.image = frame

        glutPostRedisplay()  
        glutMainLoopEvent()
        # glutIdleFunc()
                
    def run(self):
        glutMainLoop()