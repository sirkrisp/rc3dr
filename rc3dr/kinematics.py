import numpy as np
from numpy.linalg import norm, solve

import pinocchio
pinocchio.switchToNumpyMatrix()
from pinocchio.robot_wrapper import RobotWrapper

from .parameters import InverseKinematicsSolverParameters # for type support

class InverseKinematicsSolver:
    def __init__(self, inv_kin_sol_params : InverseKinematicsSolverParameters):
        super().__init__()
        self.inv_kin_sol_params = inv_kin_sol_params
        
    def solve(self, M_des, robot, joint_id, q=None):
        """ Inverse kinematics for specified joint (joint_id) and desired pose M_des (array 4x4)
        NOTE The code below is adopted from here (01.03.2020): 
        https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_i-inverse-kinematics.html
        """

        oMdes = pinocchio.SE3(M_des[:3,:3], M_des[:3,3])

        if np.all(q == None): q = pinocchio.neutral(robot.model)
        
        i=0
        iter_resample = 0
        while True:
            # forward
            pinocchio.forwardKinematics(robot.model, robot.data, q)
            
            # error between desired pose and the current one
            dMi = oMdes.actInv(robot.data.oMi[joint_id])
            err = pinocchio.log(dMi).vector
            
            # Termination criteria
            if norm(err) < self.inv_kin_sol_params.eps:
                success = True
                break
            if i >= self.inv_kin_sol_params.max_iter:
                if iter_resample <= self.inv_kin_sol_params.max_resample:
                    q = pinocchio.randomConfiguration(robot.model)
                    iter_resample += 1
                    continue
                else:
                    success = False
                    break
                
            # Jacobian
            J = pinocchio.computeJointJacobian(robot.model, robot.data, q, joint_id)
            
            # P controller (?)
            # v* =~ A * e
            v = - J.T.dot(solve(J.dot(J.T) + self.inv_kin_sol_params.damp * np.eye(6), err))
            
            # integrate (in this case only sum)
            q = pinocchio.integrate(robot.model, q, v*self.inv_kin_sol_params.dt)
            
            i += 1

        if not success: q = pinocchio.neutral(robot.model)
        q_arr = np.squeeze(np.array(q))
        q_arr = np.mod(np.abs(q_arr), 2*np.pi) * np.sign(q_arr)
        mask = np.abs(q_arr) > np.pi
        # If angle abs(q) is larger than pi, represent angle as the shorter angle inv_sign(q) * (2*pi - abs(q))
        # This is to avoid conflicting with angle limits. 
        q_arr[mask] = (-1) * np.sign(q_arr[mask]) * (2*np.pi - np.abs(q_arr[mask]))
        return success, q_arr

    
