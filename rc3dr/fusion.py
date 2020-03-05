from dataclasses import dataclass
from typing import Tuple

from .parameters import CameraParamters, TSDFVolumeParameters

import numpy as np
from numba import cuda
import math

class TSDFFusion:
    def __init__(self, cam_params : CameraParamters, tsdf_vol_params : TSDFVolumeParameters):
        # external parameters
        self.cam_params = cam_params
        self.tsdf_vol_params = tsdf_vol_params

        # Uncertainty in depth
        self.mu = max(cam_params.unc, tsdf_vol_params.voxel_size * np.sqrt(3))

    def update(self, F, W, w_T_f, R, W_R = 1):
        """
        Args: 
            F           : Fusion volume (TSDF values). Negative values -> possibly inside. Positive values -> outside.
                        NOTE centered around end-effector
            W           : Weight volume
            w_T_f       : Projection from fusion volume to world
            R           : Raw depth values (array 2D)
            W_R         : Weight of raw depth values
        """
        c_T_f = np.linalg.inv(self.cam_params.w_T_c) @ w_T_f
        tsdf_fusion[30, 128](F, W, R, W_R, c_T_f, self.cam_params.K, self.tsdf_vol_params.voxel_size, self.mu, self.cam_params.near)

@cuda.jit
def tsdf_fusion(F, W, R, W_R, c_T_f, K, voxel_size, mu, camera_near):
    """
    Args: 
        F           : Fusion volume (TSDF values). Negative values -> possibly inside. Positive values -> outside.
                      NOTE centered around end-effector
        W           : Weight volume
        c_T_f       : Projection from fusion volume to camera view
        R           : Raw depth values (array 2D)
        W_R         : Weight of raw depth values
        K           : Camera intrinsics
        voxel _size : Size of fusion voxel
        mu          : Maximum distance to nearest surface. Points in the visible region that 
                      are greater than mu away from the nearest surface get truncated.
    """
    start_x, start_y, start_z = cuda.grid(3)
    stride_x, stride_y, stride_z = cuda.gridsize(3)
    
    for x in range(start_x, F.shape[0], stride_x):
        for y in range(start_y, F.shape[1], stride_y):
            for z in range(start_z, F.shape[2], stride_z):

                # Voxel grid coordinates (x, y, z) in end-effector coordinates
                x_e = (x - F.shape[0] // 2) * voxel_size + voxel_size * 0.5
                y_e = (y - F.shape[1] // 2) * voxel_size + voxel_size * 0.5
                z_e = (z - F.shape[2] // 2) * voxel_size + voxel_size * 0.5

                # Fusion volume 
                # z_e *= -1
                
                # Voxel grid coordinates (x, y, z) in camera (view) coordinates
                # p_c = c_T_e * p_e
                x_c = c_T_f[0,0] * x_e + c_T_f[0,1] * y_e + c_T_f[0,2] * z_e + c_T_f[0,3]
                y_c = c_T_f[1,0] * x_e + c_T_f[1,1] * y_e + c_T_f[1,2] * z_e + c_T_f[1,3]
                z_c = c_T_f[2,0] * x_e + c_T_f[2,1] * y_e + c_T_f[2,2] * z_e + c_T_f[2,3]

                # Skip invalid volume depth
                # TODO Do we need camera_near?
                # NOTE We treat the camera as it would look in positive z direction
                if(z_c <= camera_near):
                    continue

                # Correct for inverted axis:   
                # The actual OpenGL camera looks in negative z-direction (the z axis is flipped).
                # However, in the code we treat the camera as it would look in positive z direction.
                # This translates to an inverted x-axis.
                x_c *= -1
                # Flip y because of image coordinate system.
                y_c *= -1
                
                # Camera coordinates to image pixels
                pixel_x = int(math.floor(K[0,0] * (x_c / z_c) + K[0,2]))
                pixel_y = int(math.floor(K[1,1] * (y_c / z_c) + K[1,2]))
                
                # Skip if outside view frustum
                im_h = R.shape[0]
                im_w = R.shape[1]
                if (pixel_x < 0 or pixel_x >= im_w or pixel_y < 0 or pixel_y >= im_h):
                    continue
                
                # Skip invalid raw depth values
                # NOTE order of coordinates
                z_raw = R[pixel_y, pixel_x]
                # NOTE Normally z_raw == 0 means invalid depth input. However, as in this case most
                # pixels are zero, it means there is no object.
                if z_raw <= camera_near:
                    # F[x, y, z] = 1
                    continue
                
                # NOTE I measure depth in positive z direction
                # TODO Change to measure depth in negative z direction
                z_c *= -1

                # F_{R_k}(p)
                z_diff = z_raw - (-z_c)
                # NOTE In the paper the raw depth value is measured in negative z axis.
                # Here we measure the depth in the positive z direction.
                if z_diff < -mu: # z_raw < z_c + mu
                    # Ignore non-visible points farther than mu from the nearest surface
                    continue
                # Visible points farther than mu from the nearest surface get truncated
                F_R = min(1.0, z_diff / mu)

                # Integrate TSDF
                W_last = W[x, y, z]
                W[x, y, z] = W[x, y, z] + W_R
                F[x, y, z] = (W_last * F[x, y, z] + W_R * F_R) / W[x, y, z] 




