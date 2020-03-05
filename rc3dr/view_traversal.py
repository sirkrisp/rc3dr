import numpy as np
from pyquaternion import Quaternion
from .parameters import ViewTraverserParameters, CameraParamters, TSDFVolumeParameters
from .utils import views_e_to_f, views_e_to_generalized_coords, q_to_json

class ViewTraverser:


    def __init__(self, params : ViewTraverserParameters, cam_params : CameraParamters, tsdf_params : TSDFVolumeParameters):
        super().__init__()
        self.params = params
        self.cam_params = CameraParamters
        self.tsdf_params = tsdf_params

    def greedy_shortest_path(self, views_q):
        num_nodes = views_q.shape[0]
        node = 0
        nodes_left = set(np.arange(num_nodes).tolist())

        greedy_traversal = []

        while len(nodes_left) > 0:
            
            q = views_q[node,:]
            
            min_cost = np.finfo(np.float32).max
            
            for n in nodes_left:
                cost = np.max(np.abs(views_q[n,:] - q))
                if cost < min_cost:
                    node = n
                    min_cost = cost
            
            nodes_left.remove(node)
            greedy_traversal.append(node)

        return greedy_traversal

    # TODO select step size on relative distance between T1 and T2
    def generate_path_sphere(self, T1, T2):
        """ T1 and T2 are assumed to be centered around a zero.
        """
        
        t1 = np.array(T1[:3])
        t2 = np.array(T2[:3])
        q1 = Quaternion(axis = T1[3:6], angle=T1[6])
        q2 = Quaternion(axis = T2[3:6], angle=T2[6])
        f_T_c_1 = q1.transformation_matrix
        f_T_c_1[:3,3] = t1
        f_T_c_2 = q2.transformation_matrix
        f_T_c_2[:3,3] = t2
        
        # NOTE np.norm(t1) == np.norm(t2)
        norm = np.linalg.norm(t1)
        assert np.abs(norm - np.linalg.norm(t2)) < 1e-4 , "T1 and T2 have to be centered around a sphere"
        
        # rotate from 1 -> 2 on sphere
        axis = np.cross(t1 / norm, t2 / norm)
        angle = np.arccos(np.sum(t1 * t2 / (norm**2)))
        
        quat_main = Quaternion(axis = axis, angle = angle)
        T_main = quat_main.transformation_matrix

        # T_diff = f_T_c_2 * (T_main * f_T_c_1)^-1
        # => f_T_c_2 = T_diff * T_main * f_T_c_1
        T_diff = f_T_c_2 @ np.linalg.inv(T_main @ f_T_c_1)
        quat_diff = Quaternion(matrix = T_diff)

        assert np.abs(np.sum(T_diff[:3,3])) < 1e-4

        for i in range(self.params.n_steps):
            
            quat_main_step = Quaternion(axis = quat_main.axis, angle = quat_main.angle * i / self.params.n_steps)
            quat_diff_step = Quaternion(axis = quat_diff.axis, angle = quat_diff.angle * i / self.params.n_steps)

            # NOTE
            # f_T_c_2 = T_diff * T_main * f_T_c_1
            # => f_T_c[:3,:3] =  T_diff[:3,:3] * T_main[:3,:3] * f_T_c_1[:3,:3]
            # => f_T_c[:3,3] = T_diff[:3,:3] * T_main[:3,:3] * f_T_c_1[:3,3] since T_diff and T_main are pure rotations

            trans = quat_diff_step.rotate(quat_main_step.rotate(t1))
            quat = quat_diff_step * quat_main_step * q1
            
            f_T_c = quat.transformation_matrix
            f_T_c[:3,3] = trans
            
            yield f_T_c


    def get_steps(self, views, robot, ee_id, inv_kin_solver):
        """
        Args:
            views   : views expressed in end-effector space
        """
        print("Views to generalized coords...")
        views_q = views_e_to_generalized_coords(views, self.cam_params.w_T_c, robot, ee_id, inv_kin_solver)

        print("Compute greedy shortest path...")
        # TODO Instead of greedy use Christofides algorithm
        greedy_traversal = self.greedy_shortest_path(views_q)

        print("Compute steps...")
        # TODO Computing steps takes too much time..
        f_T_e = np.linalg.inv(self.tsdf_params.e_T_f)
        w_T_c = self.cam_params.w_T_c
        views_f = views_e_to_f(views, f_T_e)

        q_steps = []
        view_steps_e = []
        view_step_e = np.ndarray(7)

        for t in range(len(greedy_traversal) - 1):
            
            T1 = views_f[greedy_traversal[t]]
            T2 = views_f[greedy_traversal[t+1]]
            # NOTE Expects that views lie on a sphere around the fusion volume.
            steps12 = self.generate_path_sphere(T1, T2)
            
            q = None
            for f_T_c in steps12:
                # compute end-effector pose from f_T_c
                e_T_c = self.tsdf_params.e_T_f @ f_T_c
                c_T_e = np.linalg.inv(e_T_c)
                w_T_e = self.cam_params.w_T_c @ c_T_e

                success, q = inv_kin_solver.solve(w_T_e, robot, ee_id, q)
                q_steps.append(q_to_json(robot, q))

                # save view step e for visualization purpose
                view_step_e[:3] = e_T_c[:3,3]
                quat = Quaternion(matrix =e_T_c )
                view_step_e[3:6] = quat.axis
                view_step_e[6] = quat.angle
                view_steps_e.append(view_step_e.tolist())
        
        return q_steps, greedy_traversal, view_steps_e