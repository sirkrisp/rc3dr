import numpy as np
import scipy as sp
from pyquaternion import Quaternion
import igl

def q_to_json(robot, q):
    json = {}
    for i in range(0, len(robot.model.names)-1): # omit universe joint 
        json[robot.model.names[i+1]] = q[i]
    return json


def rgb_to_depth(R, cam_params):
    """
    Args:
        R   : 3d image array with values 0..255
    """
    R = R[:,:,0].astype(np.float32) # only look at one channel (gray scale)
    mask = R != 0

    # to normalized device coordinates
    R[mask] = 1 - 2 * R[mask] / 255    

    # to eye space
    n = cam_params.near
    f = cam_params.far
    A = (f + n) / (f - n)
    B = -2*f*n / (f - n)
    R[mask] = B / (R[mask] - A)

    R = np.ascontiguousarray(R, dtype=np.float32)

    return R


def views_e_to_f(views_e, f_T_e):
    """ Project from end-effector space to fusion space.
    """
    # views expressed in fusion space
    views_f = []

    # Project views from end-effector space to fusion space
    for i in range(len(views_e)):
        view = views_e[i].copy()
        e_T_c = Quaternion(axis=view[3:6], angle=view[6]).transformation_matrix
        e_T_c[:3,3] = view[:3]

        f_T_c = f_T_e @ e_T_c
        view[:3] = f_T_c[:3,3]
        quat = Quaternion(matrix = f_T_c)
        view[3:6] = quat.axis
        view[6] = quat.angle
        views_f.append(view)
    
    return views_f

def views_e_to_generalized_coords(views_e, w_T_c, robot, ee_id, inv_kin_solver):
    num_views = len(views_e)

    # For 6DOF robot
    # TODO read from robot configuration
    views_q = np.ndarray((num_views, 6))

    failed_views = []

    for i in range(num_views):
        
        # e_T_c
        view = views_e[i]
        e_T_c = Quaternion(axis=view[3:6], angle=view[6]).transformation_matrix
        e_T_c[:3,3] = np.array(view[:3])
        
        # w_T_e
        w_T_e = w_T_c @ np.linalg.inv(e_T_c)
        
        # compute inverse kinematics
        success, q = inv_kin_solver.solve(w_T_e, robot, ee_id)
        
        if not success:
            print(q)
            failed_views.append(views_e[i])
            print("failed to compute inverse kinematics for view", i)
        
        views_q[i,:] = q

    return views_q

def views_to_cam_poses(views):
    num_views = len(views)
    cam_poses = np.ndarray((num_views, 6))
    for i in range(num_views):

        cam_pos = np.array(views[i][:3])
        cam_dir = np.array([0,0,1]) # looking in positive z direction

        # rotate cam_dir
        q = Quaternion(axis=views[i][3:6], angle=views[i][6])
        cam_dir = q.rotate(cam_dir)

        cam_poses[i,:3] = cam_pos
        cam_poses[i,3:] = cam_dir
    return cam_poses

# adopted from here:
# https://libigl.github.io/libigl-python-bindings/tutorials/
def smooth_data(v, f, w):
    # Smoothing = Minimize curvature
    #
    # min_{p*}{E(p*)}
    #
    # E(p*) = sum_{p in points on mesh}{A_p * ((Lp*)^2 + w(p* - p)^2)},
    # where A_p is the voronoi region around point p
    #
    # Solve dE(p*)/dp* = 0
    # => (L'ML + wM)p* = (wM)p ~= Ax = b
    #
    # L = M^(-1)L_w, w = cotangent[c]/uniform[u]/...
    #
    # L_c defined in slides 5 page 66
    # M and L defined in slides 6 page 23  
    
    # Constructing the squared Laplacian and squared Hessian energy
    # NOTE L_c is sparse
    # L_c = igl.cotmatrix(v, f) # discrete laplacian
    
    # L_c.data[np.isnan(L_c.data)] = 0
    # L_c.data[L_c.data > 1e7] = 1e7
    # L_c.data[L_c.data < -1e7] = -1e7
    
    # Uniform laplacian 
    # adopted from here: 
    # https://libigl.github.io/libigl-python-bindings/igl_docs/#cotmatrix
    adj = igl.adjacency_matrix(f)
    # Sum each row = number of neighbours
    adj_sum = np.squeeze(np.asarray(np.sum(adj, axis=1)))
    # Convert row sums into diagonal of sparse matrix
    adj_diag = sp.sparse.diags(adj_sum)
    # Build uniform laplacian
    L_u = adj - adj_diag
    
    # NOTE M is sparse
    M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    # Give corrupted triangles only very little weight
    mask = np.logical_or(np.isnan(M.data), M.data == 0)
    M.data[mask] = 1e-7
    
    M_inv = sp.sparse.diags(1 / M.diagonal())
    
    # NOTE L_c can become very weird when dealing with corrupted meshes
    # L = M_inv @ L_c
    L = M_inv @ L_u
    
    A = L.T @ M @ L + w * M
    # TODO lambda * ID ?
    A += sp.sparse.identity(A.shape[0])
    b = w * M @ v
    
    v_star = sp.sparse.linalg.spsolve(A, b)
    
    return v_star


