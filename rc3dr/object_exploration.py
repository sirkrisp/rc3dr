
import numpy as np
from pyquaternion import Quaternion

from .parameters import ObjectExplorerParameters

#    Y
#    |
#    |
#    .-----X
#  ／
# Z

class ObjectExplorer:
    def __init__(self, params : ObjectExplorerParameters):
        super().__init__()
        self.params = params
        self.num_first_move = int(0.5 * self.params.n_steps)
        self.num_second_move = int(0.5 * self.params.n_steps)
        # self.num_third_move = int(0.1 * num_orientations)
    
    def get_steps(self):
        num = 0
        angle_x = 0
        angle_y = 0
        for i in range(self.params.n_steps+1):
            orientation = np.array([0,0,1], dtype = np.float32)
            # if i < self.num_first_move:
            #     angle_x += self.cone_alpha / 2 / self.num_first_move
            # elif i < (self.num_first_move + self.num_second_move):
            #     angle_x = self.cone_alpha / 2 * np.cos((i - self.num_first_move) / self.num_second_move * 2 * np.pi)
            #     angle_y = self.cone_alpha / 2 * np.sin((i - self.num_first_move) / self.num_second_move * 2 * np.pi)
            # else:
            #     # go back to original orientation
            #     angle_x -= self.cone_alpha / 2 / self.num_third_move

            if i <= self.num_first_move:
                angle_x = self.params.cone_alpha / 2 * np.sin(i / self.num_second_move * 2 * np.pi)
            else:
                angle_y = self.params.cone_alpha / 2 * np.sin((i - self.num_first_move) / self.num_second_move * 2 * np.pi)

            rot_x = Quaternion(axis=[1, 0, 0], angle=angle_x)
            rot_y = Quaternion(axis=[0, 1, 0], angle=angle_y)
            rot_xy = rot_x * rot_y

            e_T_explore = rot_xy.transformation_matrix

            yield e_T_explore


