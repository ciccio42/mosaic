import robosuite_env.utils.utils as utils

def get_env(env_name, ranges, **kwargs):
    if env_name == 'Sawyer_PickPlaceDistractor':
        from robosuite_env.tasks.new_pp import SawyerPickPlace
        env = SawyerPickPlace
    elif env_name == 'Panda_PickPlaceDistractor':
        from robosuite_env.tasks.new_pp import PandaPickPlace
        env = PandaPickPlace
    elif env_name == 'UR5e_PickPlaceDistractor':
        from robosuite_env.tasks.new_pp import UR5ePickPlace
        env = UR5ePickPlace
    elif env_name == 'Panda_NutAssemblyDistractor':
        from robosuite_env.tasks.nut_assembly import PandaNutAssemblyDistractor
        env = PandaNutAssemblyDistractor
    elif env_name == 'Sawyer_NutAssemblyDistractor':
        from robosuite_env.tasks.nut_assembly import SawyerNutAssemblyDistractor
        env = SawyerNutAssemblyDistractor
    elif env_name == 'UR5e_NutAssemblyDistractor':
        from robosuite_env.tasks.nut_assembly import UR5eNutAssemblyDistractor
        env = UR5eNutAssemblyDistractor
    elif env_name == 'Panda_BlockStacking':
        from robosuite_env.tasks.stack import PandaBlockStacking
        env = PandaBlockStacking
    elif env_name == 'Sawyer_BlockStacking':
        from robosuite_env.tasks.stack import SawyerBlockStacking
        env = SawyerBlockStacking
    elif env_name == 'Panda_Basketball':
        from robosuite_env.tasks.basketball import PandaBasketball
        env = PandaBasketball
    elif env_name == 'Sawyer_Basketball':
        from robosuite_env.tasks.basketball import SawyerBasketball
        env = SawyerBasketball
    elif env_name == 'Panda_Insert':
        from robosuite_env.tasks.insert import PandaInsert
        env = PandaInsert
    elif env_name == 'Sawyer_Insert':
        from robosuite_env.tasks.insert import SawyerInsert
        env = SawyerInsert
    elif env_name == 'Panda_Drawer':
        from robosuite_env.tasks.drawer import PandaDrawer
        env = PandaDrawer
    elif env_name == 'Sawyer_Drawer':
        from robosuite_env.tasks.drawer import SawyerDrawer
        env = SawyerDrawer
    elif env_name == 'Panda_Button':
        from robosuite_env.tasks.press_button import PandaButton
        env = PandaButton
    elif env_name == 'Sawyer_Button':
        from robosuite_env.tasks.press_button import SawyerButton
        env = SawyerButton
    elif env_name == 'Panda_Door':
        from robosuite_env.tasks.door import PandaDoor
        env = PandaDoor
    elif env_name == 'Sawyer_Door':
        from robosuite_env.tasks.door import SawyerDoor
        env = SawyerDoor
    else:
        raise NotImplementedError
    
    # get env configuration file
    task_name = env_name.split('_')[1]
    env_conf = utils.read_conf_file(task_name=task_name)

    env = env( 
              mount_types=env_conf['mount_types'],
              gripper_types=env_conf['gripper_types'],
              table_full_size=env_conf['table_full_size'],
              table_offset=env_conf['table_offset'],
              robot_offset=env_conf['robot_offset'],
              horizon=env_conf['horizon'],
              camera_names=env_conf['camera_names'],
              camera_heights=env_conf['camera_heights'],
              camera_widths=env_conf['camera_widths'],
              camera_depths=env_conf['camera_depths'],
              camera_poses=env_conf['camera_poses'],
              camera_attribs=env_conf['camera_attribs'],
              camera_gripper=env_conf['camera_gripper'],
              y_ranges=env_conf['y_ranges'],
              env_conf=env_conf,
              **kwargs
              )

    if kwargs['controller_configs']['type'] == "IK_POSE":
        from robosuite_env.custom_ik_wrapper import CustomIKWrapper
        return CustomIKWrapper(env, ranges=ranges)
    else:
        from robosuite_env.custom_osc_pose_wrapper import CustomOSCPoseWrapper
        return CustomOSCPoseWrapper(env, ranges=ranges)
