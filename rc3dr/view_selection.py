import numpy as np
from numba import cuda
import math
from pyquaternion import Quaternion

# processing
import igl
from skimage import measure

from .parameters import ViewSelectorParameters, CameraParamters, TSDFVolumeParameters

from .utils import views_e_to_f, smooth_data

class ViewSelector:

    def __init__(self, params : ViewSelectorParameters, cam_params : CameraParamters, tsdf_vol_params : TSDFVolumeParameters):
        super().__init__()
        self.params = params
        self.cam_params = cam_params
        self.tsdf_vol_params = tsdf_vol_params

    def tsdf_to_mesh(self, F):
        """ Generate mesh from F (Truncated Signed Distance Function - TSDF).
        """
        # NOTE we choose a slightly negative level to avoid the creation of surfaces between zero space and positive space
        # since zero space can also mean "no information".
        verts, faces, normals, values = measure.marching_cubes_lewiner(F, level=0, spacing=[self.tsdf_vol_params.voxel_size]*3)
        # center points
        verts = verts - self.tsdf_vol_params.size / 2

        return verts, faces

    # def simplify_mesh(self, verts, faces):
    #     """ Make mesh smaller.
    #     """
    #     # TODO This step can fail
    #     # TODO Extract largest component of mesh
    #     # TODO Smooth mesh
    #     success, U, G, J, I = igl.decimate(verts, faces, max_m=self.params.max_m)
    #     return success, U, G

    def sample_verts_from_mesh(self, verts, faces):
        """ Sample verts from mesh. Vertices with higher curvature get higher weight.
        """
        curvature = igl.gaussian_curvature(verts, faces)

        # clip values
        q95 = np.quantile(curvature, 0.95)
        curvature[np.isnan(curvature)] = q95
        mask = np.abs(curvature) > q95
        curvature[mask] = np.sign(curvature[mask]) * q95

        # Compute weights/ probabilities
        p = np.abs(curvature)
        p /= np.sum(p)

        verts_choice = np.random.choice(np.arange(verts.shape[0]), self.params.n_sample_pts, replace=False, p=p)

        return verts_choice, p

    def compute_geometry_values(self, view, points, normals):
        """ Compute geometry values based on the angle between surface normal and ray direction.
        Args:
            view    : list of 7 values. view[:3] = translation. view[3:6] = rotation axis. view[6] = rotation angle.
        """
        # Camera position
        cam_pos = np.array(view[:3])
        
        # Geometry scores
        # ray_dir point to camera
        ray_dir = (cam_pos - points)
        ray_dir /= np.linalg.norm(ray_dir, axis=1, keepdims=True)
        
        geom_scores = np.maximum(0, np.sum(normals * ray_dir, axis=1))
        
        # Check if inside view frustum and update geometry scores accordingly
        
        # Camera direction
        q_axis = view[3:6]
        q_angle = view[6]
        q = Quaternion(axis=q_axis, angle=q_angle)
        
        # NOTE We treat the camera as it would look in positive z direction
        cam_dir = np.array([[0, 0, 1]]).T
        cam_dir = q.rotate(cam_dir)
        
        # ray_dir camera to point
        ray_dir = -ray_dir
        
        # TODO np.abs(angles) <= fov/2 is a FALSE CRITERIUM
        # => FOV is not a cone but a pyramid
        # NOTE angle around cam direction => +/- fov/2
        angles = np.arccos(np.sum(ray_dir * cam_dir, axis = 1))
        in_fov = np.abs(angles) <= self.cam_params.angle/2
        
        geom_scores *= in_fov
        
        return geom_scores

    def compute_visibility_values(self, view, points, F):
        """ Compute visibility by ray marching through F (tsdf).
        """
        visibility = np.zeros(points.shape[0])
        # NOTE It can still happen to skip a voxel
        mu = self.tsdf_vol_params.voxel_size*np.sqrt(3)
        check_visibility[30,128]( visibility, np.array(view[:3]), points, F, self.cam_params.K, self.tsdf_vol_params.voxel_size, mu)
        return visibility

    def greedy_best_subset(self, scores):
        """ Greedy approach: Incrementally take views that improve the objective the most.
        Args:
            scores  (np.ndarray): 2D array of scores (num_views, num_points)
        """
        num_views = scores.shape[0]
        # Only have to track max scores

        # Set of best views
        # Set of views left

        best_views = set()
        views_left = set(np.arange(num_views))

        max_scores = np.zeros(self.params.n_sample_pts)
        # each view should at least add 2% point scores
        gamma = 0.02 * self.params.n_sample_pts


        while True:
            
            total_score = np.sum(max_scores)
            
            best_improvement = 0
            best_v = None
            
            for v in views_left:
                
                # scores with new view v
                scores_v = max_scores.copy()
                mask = scores[v,:] > max_scores
                scores_v[mask] = scores[v,mask]
                
                improvement = np.sum(scores_v) - total_score - gamma
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_v = v
            
            if best_v != None:
                best_views.add(best_v)
                views_left.remove(best_v)
                
                # update max scores
                mask = scores[best_v,:] > max_scores
                max_scores[mask] = scores[best_v,mask]
            else:
                break
        
        return list(best_views), total_score

    def select_views(self, views_e, F):
        """ Select a subset views that contain the most information about F.

        The pipeline is as follows:
        TSDF --> Mesh --> Simplify Mesh --> Normals --> Sample points weighted by curvature
          --> Compute scores for each view --> Greedily select best subset of views

        Args:
            F       : 3D Fusion volume
            views_e : list of views expressed in end-effector space.
        """
        # Project views from end-effector space

        f_T_e = np.linalg.inv(self.tsdf_vol_params.e_T_f)
        views_f = views_e_to_f(views_e, f_T_e)

        print("TSDF to mesh...")
        verts, faces = self.tsdf_to_mesh(F)

        print("Smooth data...")
        # success, verts, faces = self.simplify_mesh(verts, faces)
        v_star = smooth_data(verts, faces, self.params.w_smooth)


        print("Normals...")
        normals = igl.per_vertex_normals(v_star, faces)

        print("Sample vertices ...")
        verts_choice, _ = self.sample_verts_from_mesh(v_star, faces)

        # NOTE sample from original vertices
        verts = verts[verts_choice,:]
        normals = normals[verts_choice,:]

        num_views = len(views_e)
        scores = np.ndarray((num_views, self.params.n_sample_pts))

        print("Compute scores ...")
        for i in range(num_views):
            scores[i,:] = self.compute_geometry_values(views_f[i], verts, normals)
            visibility = self.compute_visibility_values(views_f[i], verts, F)
            scores[i,:] *= visibility
        
        print("Greedy best subset ...")
        best_views, _ = self.greedy_best_subset(scores)

        return best_views


@cuda.jit
def check_visibility(visibility, view_pos, points, tsdf_volume, K, voxel_size, mu):
    """
    Args:
        view: e_T_c (projection from camera view to end-effector)
        points: points expressed in end-effector system. (n_points, 3)
    """
    start_x = cuda.grid(1)
    stride_x = cuda.gridsize(1)
    
    for i in range(start_x, points.shape[0], stride_x):
        # ray marching from point to camera
        # check if visible
        
        ray_point_x = points[i,0]
        ray_point_y = points[i,1]
        ray_point_z = points[i,2]
        
        ray_direction_x = view_pos[0] - points[i,0]
        ray_direction_y = view_pos[1] - points[i,1]
        ray_direction_z = view_pos[2] - points[i,2]
        
        # normalize ray direction
        ray_norm = math.sqrt(ray_direction_x**2 + ray_direction_y**2 + ray_direction_z**2)
        ray_direction_x /= ray_norm
        ray_direction_y /= ray_norm
        ray_direction_z /= ray_norm
        
        # NOTE Assumes quadratic volume
        tsdf_dim = voxel_size * tsdf_volume.shape[0]
        
        while True:
            # go in ray direction
            ray_point_x += mu * ray_direction_x
            ray_point_y += mu * ray_direction_y
            ray_point_z += mu * ray_direction_z
            
            # get volume coordinates
            tmp = ((ray_point_x + tsdf_dim / 2) / tsdf_dim) * tsdf_volume.shape[0]
            v_x = int(math.floor(tmp))
            tmp = ((ray_point_y + tsdf_dim / 2) / tsdf_dim) * tsdf_volume.shape[0]
            v_y = int(math.floor(tmp))
            tmp = ((ray_point_z + tsdf_dim / 2) / tsdf_dim) * tsdf_volume.shape[0]
            v_z = int(math.floor(tmp))
            
            if (v_x < 0 or v_x >= tsdf_volume.shape[0] 
                or v_y < 0 or v_y >= tsdf_volume.shape[1] 
                or v_z < 0 or v_z >= tsdf_volume.shape[2]):
                visibility[i] = 1
                break
                
            tsdf_value = tsdf_volume[v_x, v_y, v_z]
            
            if tsdf_value <= 0:
                visibility[i] = 0
                break
        