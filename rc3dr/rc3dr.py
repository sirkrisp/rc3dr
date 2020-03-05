import socketio
from pyrsistent import m # https://stackoverflow.com/questions/5844672/delete-an-element-from-a-dictionary/50341031#50341031
import numpy as np
from numba import cuda
import cv2 # image decode
from pinocchio.robot_wrapper import RobotWrapper # read robot
import os # filename of robot arm
from pyquaternion import Quaternion

# Parameter data classes
from .parameters import ViewTraverserParameters, ObjectExplorerParameters, ViewSelectorParameters, ViewSamplerParameters, InverseKinematicsSolverParameters, CameraParamters, TSDFVolumeParameters

# Processing
from .kinematics import InverseKinematicsSolver
from .fusion import TSDFFusion

# RC3DR
from .object_exploration import ObjectExplorer
from .view_sampling import ViewSampler
from .view_selection import ViewSelector
from .view_traversal import ViewTraverser

# utils
from .utils import q_to_json, rgb_to_depth

# Instantiate data classes
inv_kin_solver_params = InverseKinematicsSolverParameters()
cam_params = CameraParamters()
tsdf_vol_params = TSDFVolumeParameters()

obj_expl_params = ObjectExplorerParameters()
view_sampler_params = ViewSamplerParameters()
view_selector_params = ViewSelectorParameters()
view_traverser_params = ViewTraverserParameters()


class RC3DR(socketio.AsyncNamespace):
    
    def __init__(self, namespace):
        super().__init__(namespace)

        self.filename    : str           = os.path.join(os.getcwd(), "./media/models/urdf/6dof_macro.urdf")
        self.ee_name     : str           = "u5ee"

        # User specific data
        self.user_sids = set() # sid = socket id
        self.robot_map = m()
        self.q_map = m() # used to start from last configuration when computing inverse kinematics
        self.tsdf_map = m()
        self.views_map = m()

        # Synchronization
        self.sync_vars_map = m()

        # Data that can be easily accessed from console (for debugging)
        self.depth_img_map = m()
        self.w_T_e_des_map = m()

        # Processing
        self.inv_kin_solver = InverseKinematicsSolver(inv_kin_solver_params)
        self.tsdf_fuser = TSDFFusion(cam_params, tsdf_vol_params)

        # RC3DR
        # explore -> sample views -> select best views -> traverse shortest path
        self.object_explorer = ObjectExplorer(obj_expl_params)
        self.view_sampler = ViewSampler(view_sampler_params)
        self.view_selector = ViewSelector(view_selector_params, cam_params, tsdf_vol_params)
        self.view_traverser = ViewTraverser(view_traverser_params, cam_params, tsdf_vol_params)

    def on_connect(self, sid, environ):
        self.user_sids.add(sid)
        robot = RobotWrapper.BuildFromURDF(self.filename)
        self.robot_map = self.robot_map.set(sid, robot)
        self.q_map = self.q_map.set(sid, np.zeros(6))
        F = cuda.device_array_like(np.zeros(tsdf_vol_params.shape))
        W = cuda.device_array_like(np.zeros(tsdf_vol_params.shape))
        self.tsdf_map = self.tsdf_map.set(sid, (F, W))
        self.sync_vars_map = self.sync_vars_map.set(sid, dict())

    def on_disconnect(self, sid):
        self.user_sids.remove(sid)
        self.robot_map = self.robot_map.remove(sid)
        self.q_map = self.q_map.remove(sid)
        self.tsdf_map = self.tsdf_map.remove(sid)
        self.sync_vars_map = self.sync_vars_map.remove(sid)

        if sid in self.views_map: self.views_map = self.views_map.remove(sid)
        if sid in self.depth_img_map: self.depth_img_map = self.depth_img_map.remove(sid)
        if sid in self.w_T_e_des_map: self.w_T_e_des_map = self.w_T_e_des_map.remove(sid)



    # ============================================ #
    #                    RC3DR                     #
    # ============================================ #

    def on_explore(self, sid = None, data = None):
        print("explore")
        if(sid == None):
            sid = list(self.user_sids)[-1]

        w_T_e_des = np.array(data['w_T_e_des'])
        w_T_e_des = np.reshape(w_T_e_des, newshape=(4,4)).T

        robot = self.robot_map[sid]
        q = self.q_map[sid]
        joint_id = robot.index(self.ee_name)

        # compute exploration steps
        exploration_steps = self.object_explorer.get_steps()

        # solve inverse kinematics for exploration steps
        q_steps = []
        for e_T_explore in exploration_steps:
            w_T_explore = w_T_e_des @ e_T_explore
            success, q = self.inv_kin_solver.solve(w_T_explore, robot, joint_id, q)
            q_steps.append(q_to_json(robot, q))

        # sync q with local map
        self.q_map = self.q_map.set(sid, q)
        return {"q_steps" : q_steps}

    def on_sample_views(self, sid = None, data = None):
        print("sample views")
        if(sid == None):
            sid = list(self.user_sids)[-1]
        # NOTE views are sampled with respect to end effector.
        views = self.view_sampler.sample_views() # (num_sample_views, 7)
        self.views_map = self.views_map.set(sid, views)
        return {"views" : views}

    def on_select_views(self, sid = None, data = None):
        print("select views")
        if(sid == None):
            sid = list(self.user_sids)[-1]
        F, W = self.tsdf_map[sid]
        Fhost = F.copy_to_host()
        views = self.views_map[sid]
        view_ids = self.view_selector.select_views(views, Fhost)
        print("views to list")
        views = np.array(views)[view_ids, :].tolist()
        self.views_map = self.views_map.set(sid, views)
        print("return")
        # return {"view_ids" : list(view_ids)}
        return {"views" : views}

    def on_traverse_views(self, sid = None, data = None):
        print("traverse views")
        if(sid == None):
            sid = list(self.user_sids)[-1]
        robot = self.robot_map[sid]
        views = self.views_map[sid]
        ee_id = robot.index(self.ee_name)
        q_steps, traversal, view_steps_e = self.view_traverser.get_steps(views, robot, ee_id, self.inv_kin_solver)
        return {"q_steps" : q_steps, "traversal" : traversal, "views" : view_steps_e}


    # ============================================ #
    #                  Processing                  #
    # ============================================ #

    def on_inverse(self, sid = None, data = None):
        # print("inverse")
        if(sid == None):
            sid = list(self.user_sids)[-1]
        robot = self.robot_map[sid]
        # construct homogeneous transformation matrix from data
        w_T_e_des = np.array(data['w_T_e_des'])
        w_T_e_des = np.reshape(w_T_e_des, newshape=(4,4)).T
        # sync with local w_T_e_des map
        self.w_T_e_des_map = self.w_T_e_des_map.set(sid, w_T_e_des)
        # compute inverse
        # TODO ee_name should be in some param class
        joint_id = robot.index(self.ee_name)
        q = self.q_map[sid]
        success, q = self.inv_kin_solver.solve(w_T_e_des, robot, joint_id, q)
        # sync with local q map
        self.q_map = self.q_map.set(sid, q)
        return q_to_json(robot, q)

    def on_tsdf_fusion(self, sid = None, data = None):
        if(sid == None):
            sid = list(self.user_sids)[-1]
        print("fuse")
        # get tsdf of user
        F, W = self.tsdf_map[sid]      
        
        # process depth buffer
        depth_buffer = data["depth_buffer"]
        R = cv2.imdecode(np.frombuffer(depth_buffer, np.uint8), -1) # raw depth data
        R = rgb_to_depth(R, cam_params)
        self.depth_img_map = self.depth_img_map.set(sid, R)

        # projection from end effector to depth camera
        w_T_f = np.array(data["w_T_f"], dtype=np.float32).reshape(4,4).T

        # update tsdf
        self.tsdf_fuser.update(F, W, w_T_f, R)

    def on_depth_buffer(self, sid = None, data = None):
        if(sid == None):
            sid = list(self.user_sids)[-1]
        depth_buffer = data["depth_buffer"]
        R = cv2.imdecode(np.frombuffer(depth_buffer, np.uint8), -1) # raw depth data
        R = rgb_to_depth(R, cam_params)
        self.depth_img_map = self.depth_img_map.set(sid, R)


    # ============================================ #
    #               Synchronization                #
    # ============================================ #

    def on_update_var(self, sid=None, data=None):
        if(sid == None):
            sid = list(self.user_sids)[-1]
        sync_vars = self.sync_vars_map[sid]
        var_name = data["var_name"]
        value = data["value"]
        sync_vars[var_name] = value
        self.sync_vars_map = self.sync_vars_map.set(sid, sync_vars)

    # called from client
    def on_sync_var(self, sid, data):
        var_name = data["var_name"]
        sync_vars = self.sync_vars_map[sid]
        return {"value" : sync_vars[var_name]}


    def on_update_e_T_f(self, sid = None, data = None):
        quat_el = data["quat"]
        quat = Quaternion(quat_el[0], quat_el[1], quat_el[2], quat_el[3])
        tsdf_vol_params.e_T_f = tsdf_vol_params.e_T_f @ quat.transformation_matrix



    # ============================================ #
    # Getters/Setters/Utility debugging functions  #
    # ============================================ #
    
    def get_tsdf_volume(self, sid = None):
        if(sid == None):
            sid = list(self.user_sids)[-1]
        F, W = self.tsdf_map[sid]  
        return F.copy_to_host()
    
    
    # async def get_e_pose(self, callback, sid = None):
    #     if(sid == None):
    #         print(self.users)
    #         sid = list(self.users)[-1]
    #     await self.emit("get_e_pose", room=sid, callback=callback)

    # async def scatter(self, data, sid=None):
    #     if(sid == None):
    #         print(self.users)
    #         sid = list(self.users)[-1]
    #     await self.emit("scatter", data, room=sid)