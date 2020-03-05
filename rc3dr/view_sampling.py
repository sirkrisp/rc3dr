import numpy as np

from pyquaternion import Quaternion

from .parameters import ViewSamplerParameters

class ViewSampler:

    def __init__(self, params : ViewSamplerParameters):
        super().__init__()
        self.params = params
    
    def sample_views(self):
        # views expressed in end-effector coordinates => e_T_c

        # main transformation
        # NOTE We treat the camera as it would look in positive z direction
        e_trans_e_c = Quaternion(axis=[0,1,0], angle=np.pi).transformation_matrix
        e_trans_e_c[2,3] = self.params.radius
        angle_y = np.random.rand(self.params.n_views) * self.params.cone_alpha - self.params.cone_alpha / 2
        angle_z = np.random.rand(self.params.n_views) * np.pi - np.pi/2

        # angle disturbances (bias to fusion center)
        # angle_x_dist = np.random.randn(num_views) * np.pi/8
        # angle_y_dist = np.random.randn(num_views) * np.pi/8
        # angle_z_dist = np.random.randn(num_views) * np.pi/8
        angle_x_dist = np.zeros(self.params.n_views)
        angle_y_dist = np.zeros(self.params.n_views)
        angle_z_dist = np.zeros(self.params.n_views)

        views = []

        for i in range(self.params.n_views):

            # main transformation
            rot_y = Quaternion(axis=[0, 1, 0], angle=angle_y[i])
            rot_z = Quaternion(axis=[0, 0, 1], angle=angle_z[i])
            e_rot_e_c = rot_z * rot_y
            e_T_c = e_rot_e_c.transformation_matrix @ e_trans_e_c

            # camera points in opposite z direction
            # rot_z = Quaternion(axis=[0, 0, 1], angle=np.pi)

            # disturb orientation
            rot_x_dist = Quaternion(axis=[1, 0, 0], angle=angle_x_dist[i])
            rot_y_dist = Quaternion(axis=[0, 1, 0], angle=angle_y_dist[i])
            rot_z_dist = Quaternion(axis=[0, 0, 1], angle=angle_z_dist[i])

            rot_dist = rot_x_dist * rot_y_dist * rot_z_dist

            e_T_c = e_T_c @ rot_dist.transformation_matrix

            view = np.zeros(7)
            view[:3] = e_T_c[:3,3]
            view[2] += self.params.offset_z
            quat = Quaternion(matrix = e_T_c) # translation gets ignored
            view[3:6] = quat.axis 
            view[6] = quat.angle

            views.append(view.tolist())
        
        return views
