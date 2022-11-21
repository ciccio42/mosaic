from robosuite.wrappers.wrapper import Wrapper
import numpy as np
from pyquaternion import Quaternion
import robosuite.utils.transform_utils as T
import copy

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
pick_place_logger = logging.getLogger(name="PickPlaceLogger")

def get_rel_action(action, base_pos, base_quat):
    if action.shape[0] == 7:
        cmd_quat = T.axisangle2quat(action[3:6])
        quat = T.quat_multiply(T.quat_inverse(base_quat), cmd_quat)
        aa = T.quat2axisangle(quat)
        return np.concatenate((action[:3] - base_pos, aa, action[6:]))
    else:
        cmd_quat = Quaternion(angle=action[3] * np.pi, axis=action[4:7])
        cmd_quat = np.array([cmd_quat.x, cmd_quat.y, cmd_quat.z, cmd_quat.w])
        quat = T.quat_multiply(T.quat_inverse(base_quat), cmd_quat)
        aa = T.quat2axisangle(quat)
        return np.concatenate((action[:3] - base_pos, aa, action[7:]))

def project_point(point, sim, camera='agentview', frame_width=320, frame_height=320):
    model_matrix = np.zeros((3, 4))
    model_matrix[:3, :3] = sim.data.get_camera_xmat(camera).T

    fovy = sim.model.cam_fovy[sim.model.camera_name2id(camera)]
    f = 0.5 * frame_height / np.tan(fovy * np.pi / 360)
    camera_matrix = np.array(((f, 0, frame_width / 2), (0, f, frame_height / 2), (0, 0, 1)))

    MVP_matrix = camera_matrix.dot(model_matrix)
    cam_coord = np.ones((4, 1))
    cam_coord[:3, 0] = point - sim.data.get_camera_xpos(camera)

    clip = MVP_matrix.dot(cam_coord)
    row, col = clip[:2].reshape(-1) / clip[2]
    row, col = row, frame_height - col
    return int(max(col, 0)), int(max(row, 0))

def post_proc_obs(obs, env):
    new_obs = {}
    from PIL import Image
    robot_name = env.robots[0].robot_model.naming_prefix
    for k in obs.keys():
        if k.startswith(robot_name):
            name = k[len(robot_name):]
            if isinstance(obs[k], np.ndarray):
                new_obs[name] = obs[k].copy()
            else:
                new_obs[name] = obs[k]
        else:
            if isinstance(obs[k], np.ndarray):
                new_obs[k] = obs[k].copy()
            else:
                new_obs[k] = obs[k]

    frame_height, frame_width = 320, 320
    if 'image' in obs:
        new_obs['image'] = obs['image'].copy()[::-1]
        frame_height, frame_width = new_obs['image'].shape[0], new_obs['image'].shape[1]
    if 'depth' in obs:
        new_obs['depth'] = obs['depth'].copy()[::-1]
    if 'hand_image' in obs:
        new_obs['hand_image'] = obs['hand_image'].copy()[::-1]

    aa = T.quat2axisangle(obs[robot_name+'eef_quat'])
    flip_points = np.array(project_point(obs[robot_name+'eef_pos'], env.sim, \
        frame_width=frame_width, frame_height=frame_height))
    flip_points[0] = frame_height - flip_points[0]
    flip_points[1] = frame_width - flip_points[1]
    new_obs['eef_point'] = flip_points
    new_obs['ee_aa'] = np.concatenate((obs[robot_name+'eef_pos'], aa)).astype(np.float32)
    return new_obs

class CustomOSCPoseWrapper(Wrapper):
    def __init__(self, env, ranges):
        super().__init__(env)
        self.action_repeat=5
        self.ranges = ranges

    def convert_rotation_gripper_to_world(self, action, base_pos, base_quat):
        """
            Take the current delta defined with respect to the gripper frame and convert it with respect to the world frame
            
            Args:
                action (): .
                base_pos (): .  
                base_quat (): . 
        """
        if action.shape[0] == 7:
            # Create rotation matrix R^{world}_{new_ee}
            R_w_new_ee = T.quat2mat(T.axisangle2quat(action[3:6]))
            # Create rotation matrix R^{ee}_{world}
            R_ee_world = T.matrix_inverse(T.quat2mat(base_quat))
            # Compute the delta with rispect to the base frame
            delta_world = R_w_new_ee @ R_ee_world
            # pick_place_logger.debug(f"Delta world {T.mat2euler(delta_world)}")
            euler = -T.mat2euler(delta_world)
            aa = T.quat2axisangle(T.mat2quat(T.euler2mat(euler)))
            return np.concatenate((action[:3] - base_pos, aa, action[6:]))
        else:
            # retrieve command quaternion
            cmd_quat = Quaternion(angle=action[3], axis=action[4:7])
            cmd_quat = np.array([cmd_quat.x, cmd_quat.y, cmd_quat.z, cmd_quat.w])
            # Create rotation matrix R^{world}_{new_ee}
            R_w_new_ee = T.quat2mat(cmd_quat)
            # Create rotation matrix R^{ee}_{world}
            R_ee_world = T.matrix_inverse(T.quat2mat(base_quat))
            # Compute the delta with rispect to the base frame
            delta_world = R_w_new_ee @ R_ee_world
            # pick_place_logger.debug(f"Delta world {T.mat2euler(delta_world)}")
            euler = -T.mat2euler(delta_world)
            aa = T.quat2axisangle(T.mat2quat(T.euler2mat(euler)))
            return np.concatenate((action[:3] - base_pos, aa, action[7:]))

    def step(self, action):
        reward = -100.0
        pick_place_logger.debug("-------------------------------------------")
        for _ in range(self.action_repeat):
            # take the current position and gripper orientation with respect to world
            pick_place_logger.debug(f"Target position {action[:3]}")
            base_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
            base_quat = T.mat2quat(np.reshape(self.env.sim.data.site_xmat[self.env.robots[0].eef_site_id], (3,3)))
            global_action = self.convert_rotation_gripper_to_world(action, base_pos, base_quat)
            pick_place_logger.debug(f"Global delta position {global_action[:3]}")
            obs, reward_t, done, info = self.env.step(global_action)
            reward = max(reward, reward_t)
        pick_place_logger.debug("----------------------------------------------\n\n")
        return post_proc_obs(obs, self.env), reward, done, info

    def reset(self):
        obs = super().reset()
        return post_proc_obs(obs, self.env)

    def _get_observation(self):
        return post_proc_obs(self.env._get_observation(), self.env)
