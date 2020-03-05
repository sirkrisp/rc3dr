from dataclasses import dataclass
from typing import Tuple
import numpy as np
from pyquaternion import Quaternion

@dataclass
class ViewTraverserParameters:
    n_steps     : np.int32      = 10

@dataclass
class ObjectExplorerParameters:
    cone_alpha  : np.float32    = 180 / 360.0 * 2 * np.pi
    n_steps     : np.int32      = 50


@dataclass
class ViewSamplerParameters:
    cone_alpha  : np.float32    = 180.0 / 360.0 * 2 * np.pi
    n_views     : np.int32      = 100
    # TODO init that sets radius
    radius      : np.float32    = 1 + 0.75*np.sqrt(2)                       # radius >= (tsdf volume width*sqrt(2))/2 + (camera near)
    offset_z    : np.float32    = 0.5                                       # osset_z because here we regard the end-effector as the joint 
                                                                            # connecting the last two links. TODO end-effector without offset.
                                                                            # NOTE Link always points in z-direction.

@dataclass
class ViewSelectorParameters:
    max_m       : np.int32      = 10000
    n_sample_pts: np.int32      = 1000
    num_pts_new : np.float32    = 0.2                                       # percentage of points new adding information
    w_smooth    : np.float32    = 1e8                                       # NOTE depends on mesh size!


@dataclass
class InverseKinematicsSolverParameters:
    """ Inverse kinematics parameters
    """
    # -- Inverse kinematics parameters --
    eps         : np.float32    = 1e-4
    max_iter    : np.int32      = 2000
    dt          : np.float32    = 1e-1
    damp        : np.float32    = 1e-6                                      # Damping factor for pseudo inverse
    max_resample: np.int32      = 5
    

@dataclass
class CameraParamters:
    """ Camera Paramters for a 35mm full frame camera with resolution 625 x 625 pixels and FOV 60 degrees.
    """
    sensor_size : np.float32    = 0.035
    resolution  : np.int32      = 625
    pixel_size  : np.float32    = sensor_size / resolution
    angle       : np.float32    = 60 / 360 * 2 * np.pi                      # vertical field of view (FOV) [rad]
    focal       : np.float32    = sensor_size / 2 / np.tan(angle / 2)
    alpha       : np.float32    = focal / pixel_size
    px          : np.int32      = resolution // 2                           # principal offset [pixel]
    py          : np.int32      = resolution // 2
    near        : np.float32    = 1.0                                       # view frustum
    far         : np.float32    = 5.0
    K           : np.ndarray    = np.array([[alpha,  0,      px],
                                            [0,      alpha,  py],
                                            [0,      0,      1]])
    # NOTE The OpenGL camera has a flipped z-axis. However, throughout the code I treat
    # the camera as it would look in positive z-direction. This allows us to define 
    # the camera pose as a homogenous transformation from camera (view) to world space.
    w_R_w_c     : np.ndarray    = np.eye(3)
    w_t_w_c     : np.ndarray    = np.array([[0,3,-1]], dtype=np.float32).T
    w_T_c       : np.ndarray    = np.vstack([np.hstack([w_R_w_c, w_t_w_c]),
                                             np.array([[0,0,0,1]])])
    unc         : np.float32    = 0.02                                      # uncertainty of depth measurement
    
    def c_T_e(self, w_T_e : np.ndarray):
        return np.linalg.inv(self.w_T_c) @ w_T_e


@dataclass
class TSDFVolumeParameters:
    """ Volume parameters for the truncated signed distance function (TSDF).
    """
    size        : np.float32    = 1.5
    resolution  : np.int32      = 200
    voxel_size  : np.float32    = size / resolution
    shape       : Tuple[int]    = (resolution, resolution, resolution)
    e_R_e_f     : np.ndarray    = np.eye(3)
    e_t_e_f     : np.ndarray    = np.array([[0,0,0.5]], dtype=np.float32).T # TODO init z-coord from robot config
    e_T_f       : np.ndarray    = np.vstack([np.hstack([e_R_e_f, e_t_e_f]), # Transformation from fusion space to end-effector
                                             np.array([[0,0,0,1]])])


