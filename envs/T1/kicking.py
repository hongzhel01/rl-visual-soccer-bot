import os
import csv

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    torch_rand_float,
    get_euler_xyz,
    quat_rotate,
)

assert gymtorch

import torch

import numpy as np
from envs.base_task import BaseTask

from utils.utils import apply_randomization

from .visual_feature_extractor import VisualFeatureExtractor
import torch.nn.functional as F  # 添加这一行来定义 'F'
from ultralytics import YOLO

class Kicking(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)

        # 初始化视觉特征提取器

        self._create_envs()
        self._init_visual_extractor()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()
        self._init_csv_logging()
        self.env_resets = 0
        self.env_successes = 0
        self.env_falling = 0
        self.ball_velocities = []
        self.saveImages = True

        # ============ 新增：初始化课程学习和步态相位控制 ============
        # 延迟导入，避免在 _init_buffers 之前导入导致 device 未定义
        from .curriculum_manager import CurriculumManager
        from .gait_phase_controller import GaitPhaseController

        self.curriculum_manager = CurriculumManager(self.cfg, self.num_envs, self.device)
        self.gait_phase_controller = GaitPhaseController(self.device, self.num_envs)

        print("✅ 课程学习管理器已初始化")
        print("✅ 步态相位控制器已初始化")
        # ============================================================

    def _init_visual_extractor(self):
        # 确保设备定义正确
        if not hasattr(self, 'device'):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 每隔多少步处理一次图片 (1 表示每一帧都处理)
        self.visual_process_interval = 1 
        # 记录当前步数
        self.visual_step_counter = 0
        # print("device:", self.device)

        # 设置总特征维度为 52
        self.visual_feature_dim = 52
        
        # 将 feature_dim 设为 52 以匹配 visual_feature_dim
        self.visual_extractor = VisualFeatureExtractor(
            feature_dim=self.visual_feature_dim,
            device=self.device
        )

        # 加载 YOLOv8 官方权重（建议使用 nano 版本以保证仿真速度）
        self.yolo_model = YOLO('yolov8n.pt').to(self.device) 
        
        # 初始化摄像头角度缓冲区
        self.camera_pitch_states = torch.full((self.num_envs,), 1.4, device=self.device)
        self.camera_yaw_states = torch.zeros(self.num_envs, device=self.device)
        
        # print(f"✅ 视觉特征提取器初始化完成 (维度: {self.visual_feature_dim})")


    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        # Load robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = asset_cfg["default_dof_drive_mode"]
        asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = asset_cfg["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = asset_cfg["fix_base_link"]
        asset_options.density = asset_cfg["density"]
        asset_options.angular_damping = asset_cfg["angular_damping"]
        asset_options.linear_damping = asset_cfg["linear_damping"]
        asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
        asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
        asset_options.armature = asset_cfg["armature"]
        asset_options.thickness = asset_cfg["thickness"]
        asset_options.disable_gravity = asset_cfg["disable_gravity"]

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.robot_body_names = self.gym.get_asset_rigid_body_names(robot_asset)

        # Load ball asset
        ball_cfg = self.cfg["ball"]
        ball_root = os.path.dirname(ball_cfg["file"])
        ball_file = os.path.basename(ball_cfg["file"])
        
        ball_asset_options = gymapi.AssetOptions()
        ball_asset_options.density = ball_cfg["density"]
        ball_asset_options.angular_damping = 0.0
        ball_asset_options.linear_damping = 0.0
        ball_asset_options.max_angular_velocity = 1000.0
        ball_asset_options.max_linear_velocity = 1000.0
        ball_asset_options.disable_gravity = False
        ball_asset_options.replace_cylinder_with_capsule = False
        ball_asset_options.thickness = ball_cfg["thickness"]
        
        ball_asset = self.gym.load_asset(self.sim, ball_root, ball_file, ball_asset_options)

        # Store ball properties
        self.ball_radius = 0.05  # From URDF
        self.ball_init_pos = to_torch(ball_cfg["init_pos"], device=self.device)
        self.ball_init_rot = to_torch(ball_cfg["init_rot"], device=self.device)
        self.ball_init_lin_vel = to_torch(ball_cfg["init_lin_vel"], device=self.device)
        self.ball_init_ang_vel = to_torch(ball_cfg["init_ang_vel"], device=self.device)

        # Continue with robot setup...
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
            self.dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
            self.torque_limits[i] = dof_props_asset["effort"][i].item()

        self.dof_stiffness = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_friction = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in self.dof_names[i]:
                    self.dof_stiffness[:, i] = self.cfg["control"]["stiffness"][name]
                    self.dof_damping[:, i] = self.cfg["control"]["damping"][name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")
        self.dof_stiffness = apply_randomization(self.dof_stiffness, self.cfg["randomization"].get("dof_stiffness"))
        self.dof_damping = apply_randomization(self.dof_damping, self.cfg["randomization"].get("dof_damping"))
        self.dof_friction = apply_randomization(self.dof_friction, self.cfg["randomization"].get("dof_friction"))

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        penalized_contact_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg["rewards"]["terminate_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["base_name"])

        # prepare penalized and termination contact indices
        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])

        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.feet_indices = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["foot_names"][i])
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(-5, 0.0, -5)
        env_upper = gymapi.Vec3(5, 5, 5)
        self.envs = []
        self.actor_handles = []
        self.ball_handles = []  # Store ball handles
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            #pos = self.env_origins[i].clone()
            #start_pose.p = gymapi.Vec3(*pos)

            # Create robot actor
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, asset_cfg["name"], i)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)

            # Create ball actor
            ball_pose = gymapi.Transform()
            ball_pose.p = gymapi.Vec3(*self.env_origins[i])
            ball_pose.p += gymapi.Vec3(self.ball_init_pos[0], self.ball_init_pos[1], self.ball_init_pos[2])
            ball_pose.r = gymapi.Quat(self.ball_init_rot[0], self.ball_init_rot[1], self.ball_init_rot[2], self.ball_init_rot[3])

            ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_pose, ball_cfg["name"], i)

            ball_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
            
            # Set ball properties
            ball_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, ball_handle)
            ball_shape_props[0].restitution = ball_cfg["restitution"]
            ball_shape_props[0].friction = ball_cfg["friction"]
            ball_shape_props[0].rolling_friction = ball_cfg["rolling_friction"]
            #ball_shape_props[0].contact_offset = ball_cfg["contact_offset"]
            self.gym.set_actor_rigid_shape_properties(env_handle, ball_handle, ball_shape_props)

            # Store handles
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.ball_handles.append(ball_handle)

            # 3. ⭐ 草皮（添加在这里）
            terrain = self.terrain
            # print("terrain:", terrain.grass_asset)
            if terrain.grass_asset is not None:
                grass_pose = gymapi.Transform()
                move_x = 0.0   
                move_y = 0.0
                grass_pose.p = gymapi.Vec3(self.env_origins[i][0] + move_x,
                                        self.env_origins[i][1] + move_y,
                                        terrain.grass_z)
                grass_actor = self.gym.create_actor(
                    env_handle,
                    terrain.grass_asset,
                    grass_pose,
                    f"grass_{i}",
                    i,
                    0
                )
                self.gym.set_rigid_body_color(
                    env_handle, grass_actor, 0,
                    gymapi.MESH_VISUAL,
                    terrain.grass_color
                )

        # 创建球门资产
        self._create_goal_assets()
        
        # 为每个环境创建球门
        self.goal_handles = []
        for i in range(self.num_envs):
            env_handle = self.envs[i]
            
            # 创建左侧球门
            left_goal_handles = self._create_goal_in_env(env_handle, i, is_left_goal=True)
            # 创建右侧球门
            right_goal_handles = self._create_goal_in_env(env_handle, i, is_left_goal=False)
            
            self.goal_handles.append((left_goal_handles, right_goal_handles))



        # Initialize ball state tensors
        self.ball_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_rot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.ball_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.ball_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)


        # ===== 在这里添加摄像头系统 =====
        # 改进的摄像头配置
        self.camera_handles = []
        self.global_camera_handles = []

        # 摄像头属性配置
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 640
        self.camera_props.height = 480
        self.camera_props.enable_tensors = True
        self.camera_props.horizontal_fov = 75.0  # 增加视野

        # 全局摄像头属性
        self.global_camera_props = gymapi.CameraProperties()
        self.global_camera_props.width = 640
        self.global_camera_props.height = 480
        self.global_camera_props.enable_tensors = True

        # 为每个环境创建摄像头
        for i in range(self.num_envs):
            env_handle = self.envs[i]
            env_origin = self.env_origins[i]
            
            # 机器人摄像头
            agent_camera_handles = []
            
            # 假设只有一个机器人，如果需要多个机器人可以循环
            camera_handle = self.gym.create_camera_sensor(env_handle, self.camera_props)
            
            # 获取机器人身体句柄（通常是基座）
            robot_handle = self.actor_handles[i]
            body_handle = self.gym.get_actor_rigid_body_handle(env_handle, robot_handle, 0)
            
            # 设置摄像头相对于机器人的位置和旋转
            local_transform = gymapi.Transform()
            # 位置：向前20cm，向上22cm（在机器人前方稍高的位置）
            local_transform.p = gymapi.Vec3(0.4, 0, 0.22)
            # 旋转：向下倾斜60度（1.047弧度），朝向地面
            local_transform.r = gymapi.Quat.from_euler_zyx(0, 0.8, 0)  # -1.047 rad ≈ -60°
            
            # 将摄像头附加到机器人身体
            self.gym.attach_camera_to_body(
                camera_handle, env_handle, body_handle, 
                local_transform, gymapi.FOLLOW_POSITION
            )
            
            agent_camera_handles.append(camera_handle)
            self.camera_handles.append(agent_camera_handles)
            
            # 全局摄像头 - 改进的位置
            global_camera_handle = self.gym.create_camera_sensor(env_handle, self.global_camera_props)
            
            # 根据环境位置设置全局摄像头
            camera_height = 3.0  # 更高的视角
            camera_distance = 4.0  # 更远的距离
            
            cam_pos = gymapi.Vec3(
                env_origin[0] + camera_distance,
                env_origin[1],
                env_origin[2] + camera_height
            )
            target_pos = gymapi.Vec3(
                env_origin[0],
                env_origin[1], 
                env_origin[2]
            )
            
            self.gym.set_camera_location(global_camera_handle, env_handle, cam_pos, target_pos)
            self.global_camera_handles.append(global_camera_handle)

        # 初始化图像缓冲区
        self._init_camera_buffers()

    def _init_camera_buffers(self):
        """初始化摄像头图像缓冲区"""
        
        # print(f"📷 初始化摄像头系统")
        # print(f"  - 机器人摄像头: 640x480")
        # print(f"  - 全局摄像头: 640x480")
        
        # 机器人摄像头缓冲区
        self.camera_images = torch.zeros(
            self.num_envs, 1,  # 每个环境一个机器人摄像头
            480, 640, 3,  # 高度, 宽度, 通道
            dtype=torch.uint8, device=self.device
        )
        
        # 全局摄像头缓冲区
        self.global_camera_images = torch.zeros(
            self.num_envs,
            480, 640, 3,
            dtype=torch.uint8, device=self.device
        )
        
        # 存储摄像头尺寸
        self.camera_width = 640
        self.camera_height = 480
        
        # 创建保存目录
        self.camera_save_dir = "robot_camera_images"
        self.global_camera_save_dir = "global_camera_images"
        
        for dir_path in [self.camera_save_dir, self.global_camera_save_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        self.video_writer = None
        cam_cfg = self.cfg.get("camera", {})
        self.video_enabled = cam_cfg.get("enable_video", False)
        self.video_fps = cam_cfg.get("video_fps", 30)
        self.video_file = cam_cfg.get("video_file", "videos/robot_cam.mp4")

        # print(f"✅ 摄像头系统初始化完成")


    def _process_camera_images(self):
        """处理摄像头图像"""
        try:
            # 渲染所有摄像头
            self.gym.render_all_camera_sensors(self.sim)
            
            # 处理机器人摄像头
            for env_id in range(self.num_envs):
                # 机器人摄像头
                camera_handle = self.camera_handles[env_id][0]
                
                try:
                    # 获取摄像头图像
                    camera_image = self.gym.get_camera_image(
                        self.sim, self.envs[env_id], camera_handle, gymapi.IMAGE_COLOR
                    )
                    
                    if camera_image is not None and camera_image.size > 0:
                        # 处理图像格式
                        processed_image = self._process_image_format(camera_image)
                        
                        if processed_image is not None:
                            # 转换为张量
                            image_tensor = torch.tensor(processed_image, device=self.device)
                            
                            # 调整尺寸并存储
                            if image_tensor.dim() == 3:  # H x W x C
                                h, w, c = image_tensor.shape
                                
                                # 如果尺寸不匹配，调整图像尺寸
                                if h != self.camera_height or w != self.camera_width:
                                    resized_image = self._resize_image_tensor(image_tensor, self.camera_height, self.camera_width)
                                    self.camera_images[env_id, 0] = resized_image
                                else:
                                    self.camera_images[env_id, 0] = image_tensor

                                # ★ 在这里追加写视频（每步一帧）
                                if self.video_enabled and env_id == 0:
                                    import cv2
                                    frame = self.camera_images[env_id, 0].cpu().numpy()
                                    if frame is not None and frame.ndim == 3:
                                        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                        if self.video_writer is None:
                                            # 确保目录存在
                                            video_dir = os.path.dirname(self.video_file)
                                            if video_dir and not os.path.exists(video_dir):
                                                os.makedirs(video_dir, exist_ok=True)
                                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                            h2, w2, _ = bgr.shape
                                            self.video_writer = cv2.VideoWriter(self.video_file, fourcc, self.video_fps, (w2, h2))
                                            if not self.video_writer.isOpened():
                                                print(f"❌ VideoWriter 打开失败: {self.video_file}")
                                                self.video_writer = None
                                            #else:
                                                #print(f"🎥 启动机器人摄像头连续录像: {self.video_file}")
                                        if self.video_writer is not None and self.video_writer.isOpened():
                                            vel_x = self.base_lin_vel[env_id, 0].item()
                                            text = f"RobotLinVelX: {vel_x:.3f} m/s"
                                            cv2.putText(
                                                bgr,  # 要画字的图像（这里是BGR）
                                                text,  # 文本内容
                                                (10, 30),  # 左下角坐标
                                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                                1.0,  # 字体缩放
                                                (255, 255, 255),  # 颜色：白色
                                                2,  # 线宽
                                                cv2.LINE_AA  # 抗锯齿
                                            )
                                            self.video_writer.write(bgr)

                except Exception as e:
                    print(f"❌ 处理机器人摄像头失败 - 环境{env_id}: {e}")
            
            # 处理全局摄像头
            for env_id in range(self.num_envs):
                camera_handle = self.global_camera_handles[env_id]
                
                try:
                    global_image = self.gym.get_camera_image(
                        self.sim, self.envs[env_id], camera_handle, gymapi.IMAGE_COLOR
                    )
                    
                    if global_image is not None and global_image.size > 0:
                        processed_image = self._process_image_format(global_image)
                        
                        if processed_image is not None:
                            image_tensor = torch.tensor(processed_image, device=self.device)
                            
                            if image_tensor.dim() == 3:
                                h, w, c = image_tensor.shape
                                
                                if h != self.camera_height or w != self.camera_width:
                                    resized_image = self._resize_image_tensor(image_tensor, self.camera_height, self.camera_width)
                                    self.global_camera_images[env_id] = resized_image
                                else:
                                    self.global_camera_images[env_id] = image_tensor
                        
                except Exception as e:
                    print(f"❌ 处理全局摄像头失败 - 环境{env_id}: {e}")
                    
        except Exception as e:
            print(f"❌ 摄像头处理错误: {e}")


    def _process_image_format(self, raw_image):
        """处理图像格式转换"""
        try:
            if raw_image is None:
                return None
                
            # 处理不同的图像格式
            if len(raw_image.shape) == 2:
                # 灰度图像 - 转换为RGB
                h, w = raw_image.shape
                rgb_image = np.stack([raw_image] * 3, axis=-1)
                return rgb_image
                
            elif len(raw_image.shape) == 3:
                h, w, c = raw_image.shape
                
                if c == 4:  # RGBA格式
                    # 移除alpha通道，BGR转RGB
                    import cv2
                    bgr_image = raw_image[:, :, :3]
                    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                    return rgb_image
                    
                elif c == 3:  # RGB或BGR格式
                    # 假设是BGR，转换为RGB
                    import cv2
                    rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
                    return rgb_image
                else:
                    print(f"⚠️ 未知的通道数: {c}")
                    return raw_image[:, :, :3]  # 取前3个通道
            else:
                print(f"⚠️ 未知的图像维度: {len(raw_image.shape)}")
                return None
                
        except Exception as e:
            print(f"❌ 图像格式处理错误: {e}")
            return None

    def _resize_image_tensor(self, image_tensor, target_height, target_width):
        """调整图像张量尺寸"""
        try:
            # 将张量转换为numpy数组进行处理
            if image_tensor.is_cuda:
                image_np = image_tensor.cpu().numpy()
            else:
                image_np = image_tensor.numpy()
            
            # 使用OpenCV调整尺寸
            import cv2
            resized_np = cv2.resize(image_np, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            # 转换回张量
            resized_tensor = torch.tensor(resized_np, device=self.device)
            return resized_tensor
            
        except Exception as e:
            print(f"❌ 调整图像尺寸失败: {e}")
            return image_tensor


    def _create_goal_assets(self):
        """创建球门资产（使用长方体）"""
        goal_cfg = self.cfg["goal"]
        
        # 球门尺寸
        width = goal_cfg["width"]
        height = goal_cfg["height"] 
        depth = goal_cfg["depth"]
        post_radius = goal_cfg["post_radius"]
        
        # 创建球门资产选项
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.density = 1000.0  # 高密度使其稳定
        
        # 使用长方体创建简单的球门框架
        # 门柱使用细长的长方体代替圆柱体
        post_asset = self.gym.create_box(
            self.sim, depth, post_radius * 2, height, asset_options
        )
        
        # 横梁资产
        crossbar_asset = self.gym.create_box(
            self.sim, depth, width, post_radius * 2, asset_options
        )
        
        self.post_asset = post_asset
        self.crossbar_asset = crossbar_asset

    def _create_goal_in_env(self, env_handle, env_id, is_left_goal=False):
        """在指定环境中创建球门"""
        pos = self.env_origins[env_id].clone()

        # ================= 🔧 手动调整区域 =================
        # 这里直接控制球门距离场地中心 (0,0) 有多远
        # 你的场地长 14米，半场就是 7米。
        # 如果觉得球门太远，把这个数字改小（比如 6.0）
        # 如果觉得球门太近，把这个数字改大（比如 7.5）
        distance_from_center = 6.0

        # 这里的 0.05 是向下埋的深度，防止悬空
        embed_depth = 0.05
        # ==================================================

        # 1. 计算 X 轴坐标
        if is_left_goal:
            # 左球门在负轴
            base_x = pos[0] - distance_from_center
        else:
            # 右球门在正轴
            base_x = pos[0] + distance_from_center

        base_y = pos[1]

        # 2. 计算 Z 轴高度 (确保落地)
        goal_height = self.cfg["goal"]["height"]
        base_z = pos[2] + goal_height / 2.0 - embed_depth

        # 打印调试信息（只打印第一个环境的，防止刷屏）
        if env_id == 0:
            side = "LEFT" if is_left_goal else "RIGHT"
            print(f"🥅 创建{side}球门: X={base_x:.2f}, Y={base_y:.2f}, Z={base_z:.2f}")

        goal_handles = []

        # --- 创建左门柱 ---
        left_post_pose = gymapi.Transform()
        left_post_pose.p = gymapi.Vec3(
            base_x,
            base_y - self.cfg["goal"]["width"] / 2 + self.cfg["goal"]["post_radius"],
            base_z
        )
        # 确保门柱垂直，重置旋转
        left_post_pose.r = gymapi.Quat(0, 0, 0, 1)

        left_post_handle = self.gym.create_actor(
            env_handle, self.post_asset, left_post_pose, "goal_post", env_id, 0, 0
        )
        goal_handles.append(left_post_handle)

        # --- 创建右门柱 ---
        right_post_pose = gymapi.Transform()
        right_post_pose.p = gymapi.Vec3(
            base_x,
            base_y + self.cfg["goal"]["width"] / 2 - self.cfg["goal"]["post_radius"],
            base_z
        )
        right_post_pose.r = gymapi.Quat(0, 0, 0, 1)

        right_post_handle = self.gym.create_actor(
            env_handle, self.post_asset, right_post_pose, "goal_post", env_id, 0, 0
        )
        goal_handles.append(right_post_handle)

        # --- 创建横梁 ---
        crossbar_pose = gymapi.Transform()
        crossbar_pose.p = gymapi.Vec3(
            base_x,
            base_y,
            # 横梁高度 = 地面 + 球门全高 - 半径 - 埋地深度
            pos[2] + goal_height - self.cfg["goal"]["post_radius"] - embed_depth
        )
        crossbar_pose.r = gymapi.Quat(0, 0, 0, 1)

        crossbar_handle = self.gym.create_actor(
            env_handle, self.crossbar_asset, crossbar_pose, "crossbar", env_id, 0, 0
        )
        goal_handles.append(crossbar_handle)

        # 设置为白色
        for handle in goal_handles:
            self.gym.set_rigid_body_color(
                env_handle, handle, 0, gymapi.MESH_VISUAL,
                gymapi.Vec3(1.0, 1.0, 1.0)
            )

        return goal_handles


    def _process_rigid_body_props(self, props, i):
        for j in range(self.num_bodies):
            if j == self.base_indice:
                props[j].com.x, self.base_mass_scaled[i, 0] = apply_randomization(
                    props[j].com.x, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.y, self.base_mass_scaled[i, 1] = apply_randomization(
                    props[j].com.y, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.z, self.base_mass_scaled[i, 2] = apply_randomization(
                    props[j].com.z, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].mass, self.base_mass_scaled[i, 3] = apply_randomization(
                    props[j].mass, self.cfg["randomization"].get("base_mass"), return_noise=True
                )
            else:
                props[j].com.x = apply_randomization(props[j].com.x, self.cfg["randomization"].get("other_com"))
                props[j].com.y = apply_randomization(props[j].com.y, self.cfg["randomization"].get("other_com"))
                props[j].com.z = apply_randomization(props[j].com.z, self.cfg["randomization"].get("other_com"))
                props[j].mass = apply_randomization(props[j].mass, self.cfg["randomization"].get("other_mass"))
            props[j].invMass = 1.0 / props[j].mass
        return props

    def _process_rigid_shape_props(self, props):
        for i in self.foot_shape_indices:
            props[i].friction = apply_randomization(0.0, self.cfg["randomization"].get("friction"))
            props[i].compliance = apply_randomization(0.0, self.cfg["randomization"].get("compliance"))
            props[i].restitution = apply_randomization(0.0, self.cfg["randomization"].get("restitution"))
        return props

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        

    def _init_buffers(self):
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]
        # print(f"📊 观测维度: {self.num_obs} (包含{self.visual_feature_dim}维视觉特征)")
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)

        # 存储视觉特征的缓冲区
        self.visual_features_buf = torch.zeros(self.num_envs, self.visual_feature_dim, 
                                              dtype=torch.float, device=self.device)

        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_ball_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # Buffer for ball-only resets
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.min_ball_vel_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.time_since_ball_is_still_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.time_since_ball_is_moving_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.extras["rew_terms"] = {}
        self.extras["ball_detected"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # ====== 初始化YOLO相关可视化状态 ======
        self.ball_visible = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)# 存储球是否在视野中的缓冲区
        self.ball_center_offset = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)# 存储球中心偏移的缓冲区
        self.steps_since_seen = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)# 存储自上次看到球以来的步数缓冲区

        self.extras["ball_detected"] = self.ball_visible  # 之后会在 _extract_visual_features 里更新

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        root_states = gymtorch.wrap_tensor(actor_root_state)

        # === 修改这里 ===
        # 计算实际每个环境的 actor 数量
        # 每个环境包含：机器人 + 球 + 草皮(可选) + 球门(左右各3个刚体)
        total_actors = root_states.shape[0]
        # 保存为成员变量，reset 时按此步长计算全局 actor 索引（不能写死 2 或 3）
        self.actors_per_env = total_actors // self.num_envs

        # print(f"调试信息: total_actors={total_actors}, num_envs={self.num_envs}, actors_per_env={self.actors_per_env}")

        # Reshape root states to separate all actors
        self.root_states = root_states.view(self.num_envs, self.actors_per_env, 13)
        
        # 定义各个 actor 的索引
        self.robot_actor_index = 0  # 机器人
        self.ball_actor_index = 1   # 球
        self.grass_actor_index = 2  # 草皮
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        
        # 修改 body_states 的 reshape
        # 每个环境的 body 数量 = 机器人body数 + 球body数 + 草皮body数
        total_bodies = body_state.shape[0] // self.num_envs
        self.body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, total_bodies, 13)
        
        # 获取机器人状态
        self.base_pos = self.root_states[:, self.robot_actor_index, 0:3]  # Robot position
        self.base_quat = self.root_states[:, self.robot_actor_index, 3:7]  # Robot quaternion
        
        # 获取球状态
        self.ball_pos = self.root_states[:, self.ball_actor_index, 0:3]  # Ball position
        self.ball_rot = self.root_states[:, self.ball_actor_index, 3:7]  # Ball quaternion
        
        # 修改 ball_lin_vel 和 ball_ang_vel 的获取方式
        # 球在 body_states 中的索引：球是第 2 个 actor，紧跟在机器人所有刚体之后，
        # 所以索引就是机器人的刚体数量。不能用 -1/-2 倒数，因为末尾是草皮和球门的刚体。
        self.ball_body_index = self.num_bodies

        self.ball_lin_vel = self.body_states[:, self.ball_body_index, 7:10]  # Ball linear velocity
        self.ball_ang_vel = self.body_states[:, self.ball_body_index, 10:13]  # Ball angular velocity
        
        # 更新 feet indices（确保它们在正确的范围内）
        if hasattr(self, 'feet_indices'):
            self.feet_pos = self.body_states[:, self.feet_indices, 0:3]
            self.feet_quat = self.body_states[:, self.feet_indices, 3:7]

        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, self.robot_actor_index, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        #self.commands = torch.zeros(self.num_envs, self.cfg["commands"]["num_commands"], dtype=torch.float, device=self.device)
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, self.robot_actor_index, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, self.robot_actor_index, 10:13])
        # Only apply gravity to robot's state
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()
        self.curriculum_prob = torch.zeros(
            1 + 2 * self.cfg["commands"]["lin_vel_levels"],
            1 + 2 * self.cfg["commands"]["ang_vel_levels"],
            dtype=torch.float,
            device=self.device,
        )
        self.curriculum_prob[self.cfg["commands"]["lin_vel_levels"], self.cfg["commands"]["ang_vel_levels"]] = 1.0
        self.env_curriculum_level = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self.mean_lin_vel_level = 0.0
        self.mean_ang_vel_level = 0.0
        self.max_lin_vel_level = 0.0
        self.max_ang_vel_level = 0.0


        # 获取 body_state 的实际大小
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # 计算每个环境的刚体数
        total_bodies_all_envs = body_state.shape[0]  # 所有环境的总刚体数
        self.total_bodies_per_env = total_bodies_all_envs // self.num_envs
        
        # print(f"Debug: total_bodies_all_envs = {total_bodies_all_envs}")
        # print(f"Debug: num_envs = {self.num_envs}")
        # print(f"Debug: total_bodies_per_env = {self.total_bodies_per_env}")
        
        # 使用正确的刚体数量初始化 pushing_forces 和 pushing_torques
        total_bodies = self.total_bodies_per_env  # 使用实际计算出的值
        
        self.pushing_forces = torch.zeros(self.num_envs, total_bodies, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, total_bodies, 3, dtype=torch.float, device=self.device)
        
        if hasattr(self, 'feet_indices'):
            self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
            self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
            self.last_feet_pos = torch.zeros_like(self.feet_pos)
            self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        
        self.dof_pos_ref = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_pos = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["init_state"]["default_joint_angles"].keys():
                if name in self.dof_names[i]:
                    self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"]["default"]

        # 按名字查找膝关节索引，供摆动相抬腿奖励使用（写死下标容易指到髋关节）
        self.knee_dof_indices = torch.tensor(
            [i for i in range(self.num_dofs) if "Knee" in self.dof_names[i]],
            dtype=torch.long,
            device=self.device,
        )

        self.last_ball_lin_vel_world = torch.zeros_like(self.body_states[:, self.ball_body_index, 7:10]) # World frame

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        self.reward_scales_ball_rolling = self.cfg["rewards"]["ball_rolling_scale"].copy()

        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        for key in list(self.reward_scales_ball_rolling.keys()):
            scale = self.reward_scales_ball_rolling[key]
            self.reward_scales_ball_rolling[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

    def _init_csv_logging(self):
        """Initialize CSV logging for reward values"""
        # Check if CSV logging is enabled in config
        self.csv_logging_enabled = self.cfg.get("basic", {}).get("enable_csv_logging", True)
        
        if not self.csv_logging_enabled:
            print("CSV logging disabled in configuration")
            return
            
        # Only log for environment 0 (single environment setup)
        self.log_env_id = 0
        
        # Create logs directory if it doesn't exist
        self.log_dir = "logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create CSV file with timestamp
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(self.log_dir, f"debug_csv/reward_log_{timestamp}.csv")
        
        # Prepare CSV headers
        self.csv_headers = [
            "episode_step", "total_reward",
            "ball_pos_x", "ball_pos_y", "ball_pos_z",
            "ball_vel_x", "ball_vel_y", "ball_vel_z",
            "robot_pos_x", "robot_pos_y", "robot_pos_z",
            "robot_lin_vel_x", "robot_lin_vel_y", "robot_lin_vel_z",
            "ball_speed", "ball_distance_to_robot"
        ]
        reward_names = ["reward_" + name for name in self.reward_names]
        self.csv_headers.extend(reward_names)  # Add all individual reward terms
        
        # Initialize CSV file with headers
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.csv_headers)
        
        print(f"CSV logging initialized: {self.csv_filename}")

    def _log_rewards_to_csv(self):
        """Log current step rewards to CSV file"""
        # Check if CSV logging is enabled
        if not getattr(self, 'csv_logging_enabled', False):
            return
            
        # Only log for the specified environment (0)
        if hasattr(self, 'episode_length_buf'):
            episode_step = self.episode_length_buf[self.log_env_id].item()
            total_reward = self.rew_buf[self.log_env_id].item()
            
            # Get ball and robot state information
            ball_pos = self.ball_pos[self.log_env_id].cpu().numpy()
            ball_vel_world = self.root_states[self.log_env_id, 1, 7:10].cpu().numpy()
            robot_pos = self.base_pos[self.log_env_id].cpu().numpy()
            robot_lin_vel = self.base_lin_vel[self.log_env_id].cpu().numpy()
            
            # Calculate derived metrics
            ball_speed = torch.norm(self.root_states[self.log_env_id, 1, 7:10]).item()
            ball_distance_to_robot = torch.norm(self.ball_pos[self.log_env_id] - self.base_pos[self.log_env_id]).item()
            
            # Prepare row data
            row_data = [
                episode_step, total_reward,
                ball_pos[0], ball_pos[1], ball_pos[2],
                ball_vel_world[0], ball_vel_world[1], ball_vel_world[2],
                robot_pos[0], robot_pos[1], robot_pos[2],
                robot_lin_vel[0], robot_lin_vel[1], robot_lin_vel[2],
                ball_speed, ball_distance_to_robot
            ]
            
            # Add individual reward terms
            for reward_name in self.reward_names:
                if reward_name in self.extras["rew_terms"]:
                    reward_value = self.extras["rew_terms"][reward_name][self.log_env_id].item()
                    row_data.append(reward_value)
                else:
                    row_data.append(0.0)  # Default if reward term not found
            
            # Write to CSV
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_data)

    def reset(self):
        """Reset all robots"""
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        #self._resample_commands()
        self._compute_observations()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        
        self.env_resets += env_ids.shape[0]

        velocity_before_reset = torch.norm(self.root_states[env_ids, 1, 7:10], dim=1)
        
        # only get velocities that are greater than 0.1
        velocity_before_reset = velocity_before_reset[velocity_before_reset > 0.1]
        if len(velocity_before_reset) > 0:
            # append velocities sepeate
            for velocity in velocity_before_reset:
                self.ball_velocities.append(velocity.item())

        # Reset robot
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # Continue with existing reset code...
        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 0, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.min_ball_vel_buf[env_ids] = 0.0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.time_since_ball_is_still_buf[env_ids] = 0.0
        self.time_since_ball_is_moving_buf[env_ids] = 0.0
        self.cmd_resample_time[env_ids] = 0

        # ============ 新增：重置步态相位和课程阶段 ============
        self.gait_phase_controller.reset(env_ids)
        self.curriculum_manager.update_stage(env_ids)
        # ====================================================

        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.extras["time_outs"] = self.time_out_buf
        self.last_ball_lin_vel_world[env_ids] = 0.0 # Reset for selected envs

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = apply_randomization(self.default_dof_pos, self.cfg["randomization"].get("init_dof_pos"))
        self.dof_vel[env_ids] = 0.0
        # 机器人在全局 actor 列表中的索引 = 环境号 * 每个环境的 actor 数 + 机器人偏移
        # 这样才能只更新机器人的 DOF（之前写死乘 2，环境数大于 1 时会索引错对象）
        env_ids_int32 = (self.actors_per_env * env_ids + self.robot_actor_index).to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        # Initialize robot states (index 0)
        self.root_states[env_ids, 0, :] = self.base_init_state
        self.root_states[env_ids, 0, :2] += self.env_origins[env_ids, :2]
        #self.root_states[env_ids, 0, :2] = apply_randomization(self.root_states[env_ids, 0, :2], self.cfg["randomization"].get("init_base_pos_xy"))
        #self.root_states[env_ids, 0, 2] += self.terrain.terrain_heights(self.root_states[env_ids, 0, :2])
        self.root_states[env_ids, 0, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            apply_randomization(
                torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
                self.cfg["randomization"].get("init_base_ang")
                ),
        )
        self.root_states[env_ids, 0, 7:9] = apply_randomization(
            torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("init_base_lin_vel_xy"),
        )

        # Reset ball in front of the (newly reset) robot
        self._reset_ball_at_robot_front(env_ids)

        # Update the simulation with new state tensor for both robot and ball
        # The self.root_states tensor has been updated for both.
        # 更新 actor indices（步长取实际的每环境 actor 数，不能写死 3）
        robot_actor_indices = (self.actors_per_env * env_ids + self.robot_actor_index).to(dtype=torch.int32)
        ball_actor_indices = (self.actors_per_env * env_ids + self.ball_actor_index).to(dtype=torch.int32)
        
        actor_indices_to_update = torch.stack((robot_actor_indices, ball_actor_indices), dim=-1).view(-1).to(dtype=torch.int32)
        num_indices = actor_indices_to_update.shape[0]

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(actor_indices_to_update),
            num_indices
        )

    def _reset_ball_at_robot_front(self, env_ids_to_reset_ball):
        """Resets the ball in front of the robot for the specified environment IDs."""
        if len(env_ids_to_reset_ball) == 0:
            return

        robot_pos = self.root_states[env_ids_to_reset_ball, self.robot_actor_index, 0:3]
        robot_quat = self.root_states[env_ids_to_reset_ball, self.robot_actor_index, 3:7]
        


        # 球相对机器人的局部偏移量 (x 向前, y 向左)
        ball_init_pos = torch.zeros(len(env_ids_to_reset_ball), 3, device=self.device)
        ball_init_pos[:, 0] = apply_randomization(ball_init_pos[:, 0], self.cfg["randomization"].get("ball_init_pos_x"))
        ball_init_pos[:, 1] = apply_randomization(ball_init_pos[:, 1], self.cfg["randomization"].get("ball_init_pos_y"))

        # 把局部偏移整体旋转到世界坐标系
        # 原来的写法是先旋转 [1,1,0] 再逐元素相乘，那不是旋转，机器人有偏航时位置会偏
        offset_world = quat_rotate(robot_quat, ball_init_pos)

        # Calculate ball's target XY position
        ball_target_xy = robot_pos[:, 0:2] + offset_world[:, 0:2]
        
        # Calculate ball's target Z position (on the ground + ball radius)
        if hasattr(self, 'terrain'):
            ball_target_z = self.terrain.terrain_heights(ball_target_xy) + self.ball_radius
        else: # Fallback if no terrain, assume ground is at z=0 relative to env_origin
             # This assumes env_origins are at z=0 for the ground level.
            ball_target_z = torch.full_like(ball_target_xy[:, 0], self.ball_radius)



        # 设置球位置使用新的索引
        self.root_states[env_ids_to_reset_ball, self.ball_actor_index, 0] = ball_target_xy[:, 0]
        self.root_states[env_ids_to_reset_ball, self.ball_actor_index, 1] = ball_target_xy[:, 1]
        self.root_states[env_ids_to_reset_ball, self.ball_actor_index, 2] = ball_target_z

        # Set ball orientation to default (identity quaternion)
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(len(env_ids_to_reset_ball), 1)
        self.root_states[env_ids_to_reset_ball, 1, 3:7] = identity_quat
        
        # Set ball linear and angular velocities to zero
        self.root_states[env_ids_to_reset_ball, 1, 7:13] = 0.0

        # Update only the ball actors in the simulation
        ball_actor_indices = (self.actors_per_env * env_ids_to_reset_ball + self.ball_actor_index).to(dtype=torch.int32)
        if len(ball_actor_indices) > 0:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states), # Send full buffer, but only indices for balls are used effectively for this call.
                gymtorch.unwrap_tensor(ball_actor_indices),
                len(ball_actor_indices)
            )

    def _teleport_robot(self):
        if self.terrain.type == "plane":
            return
        out_x_min = self.root_states[:, 0, 0] < -0.75 * self.terrain.border_size
        out_x_max = self.root_states[:, 0, 0] > self.terrain.env_width + 0.75 * self.terrain.border_size
        out_y_min = self.root_states[:, 0, 1] < -0.75 * self.terrain.border_size
        out_y_max = self.root_states[:, 0, 1] > self.terrain.env_length + 0.75 * self.terrain.border_size

        self.root_states[out_x_min, 0, 0] += self.terrain.env_width + self.terrain.border_size
        self.root_states[out_x_max, 0, 0] -= self.terrain.env_width + self.terrain.border_size
        self.root_states[out_y_min, 0, 1] += self.terrain.env_length + self.terrain.border_size
        self.root_states[out_y_max, 0, 1] -= self.terrain.env_length + self.terrain.border_size
        self.body_states[out_x_min, :, 0] += self.terrain.env_width + self.terrain.border_size
        self.body_states[out_x_max, :, 0] -= self.terrain.env_width + self.terrain.border_size
        self.body_states[out_y_min, :, 1] += self.terrain.env_length + self.terrain.border_size
        self.body_states[out_y_max, :, 1] -= self.terrain.env_length + self.terrain.border_size

        if out_x_min.any() or out_x_max.any() or out_y_min.any() or out_y_max.any():
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self._refresh_feet_state()

    def _resample_commands(self):
        env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        if self.cfg["commands"]["curriculum"]:
            self._resample_curriculum_commands(env_ids)
        else:
            self.commands[env_ids, 0] = torch_rand_float(
                self.cfg["commands"]["lin_vel_x"][0], self.cfg["commands"]["lin_vel_x"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(
                self.cfg["commands"]["lin_vel_y"][0], self.cfg["commands"]["lin_vel_y"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
            self.commands[env_ids, 2] = torch_rand_float(
                self.cfg["commands"]["ang_vel_yaw"][0], self.cfg["commands"]["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
        self.gait_frequency[env_ids] = torch_rand_float(
            self.cfg["commands"]["gait_frequency"][0], self.cfg["commands"]["gait_frequency"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        still_envs = env_ids[torch.randperm(len(env_ids))[: int(self.cfg["commands"]["still_proportion"] * len(env_ids))]]
        self.commands[still_envs, :] = 0.0
        self.gait_frequency[still_envs] = 0.0
        self.cmd_resample_time[env_ids] += torch.randint(
            int(self.cfg["commands"]["resampling_time_s"][0] / self.dt),
            int(self.cfg["commands"]["resampling_time_s"][1] / self.dt),
            (len(env_ids),),
            device=self.device,
        )

    def _update_curriculum(self, env_ids):
        if not self.cfg["commands"]["curriculum"]:
            return
        success = self.episode_length_buf[env_ids] > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt) * (
            1 - self.cfg["commands"]["episode_length_toler"]
        )
        success &= torch.abs(self.filtered_lin_vel[env_ids, 0] - self.commands[env_ids, 0]) < self.cfg["commands"]["lin_vel_x_toler"]
        success &= torch.abs(self.filtered_lin_vel[env_ids, 1] - self.commands[env_ids, 1]) < self.cfg["commands"]["lin_vel_y_toler"]
        success &= torch.abs(self.filtered_ang_vel[env_ids, 2] - self.commands[env_ids, 2]) < self.cfg["commands"]["ang_vel_yaw_toler"]
        for i in range(len(env_ids)):
            if success[i]:
                x = self.env_curriculum_level[env_ids[i], 0] + self.cfg["commands"]["lin_vel_levels"]
                y = self.env_curriculum_level[env_ids[i], 1] + self.cfg["commands"]["ang_vel_levels"]
                self.curriculum_prob[x, y] += self.cfg["commands"]["update_rate"]
                if x > 0:
                    self.curriculum_prob[x - 1, y] += self.cfg["commands"]["update_rate"]
                if x < self.curriculum_prob.shape[0] - 1:
                    self.curriculum_prob[x + 1, y] += self.cfg["commands"]["update_rate"]
                if y > 0:
                    self.curriculum_prob[x, y - 1] += self.cfg["commands"]["update_rate"]
                if y < self.curriculum_prob.shape[1] - 1:
                    self.curriculum_prob[x, y + 1] += self.cfg["commands"]["update_rate"]
        self.curriculum_prob.clamp_(max=1.0)

    def _resample_curriculum_commands(self, env_ids):
        grid_idx = torch.multinomial(self.curriculum_prob.flatten(), len(env_ids), replacement=True)
        lin_vel_level = grid_idx % self.curriculum_prob.shape[1] - self.cfg["commands"]["lin_vel_levels"]
        ang_vel_level = grid_idx // self.curriculum_prob.shape[1] - self.cfg["commands"]["ang_vel_levels"]
        self.env_curriculum_level[env_ids, 0] = lin_vel_level
        self.env_curriculum_level[env_ids, 1] = ang_vel_level
        self.mean_lin_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 0]).float())
        self.mean_ang_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 1]).float())
        self.max_lin_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 0]))
        self.max_ang_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 1]))
        self.commands[env_ids, 0] = (
            lin_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.cfg["commands"]["lin_vel_x_resolution"]
        self.commands[env_ids, 1] = (
            torch.abs(lin_vel_level)
            * torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
            * self.cfg["commands"]["lin_vel_y_resolution"]
        )
        self.commands[env_ids, 2] = (
            ang_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.cfg["commands"]["ang_vel_resolution"]

    def step(self, actions):
        # pre physics step
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        dof_targets = self.default_dof_pos + self.cfg["control"]["action_scale"] * self.actions

        # perform physics step
        self.torques.zero_()
        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(dof_torques - friction, min=-self.torque_limits, max=self.torque_limits)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
        self.torques /= self.cfg["control"]["decimation"]
        self.render()

        self.gym.step_graphics(self.sim)

        # ===== 摄像头处理 =====
        # 注意：_process_camera_images 内部已经调用了 render_all_camera_sensors，
        # 所以这里只在需要提取特征的步数做一次，避免同一步重复渲染、重复写视频帧
        if self.common_step_counter % self.visual_process_interval == 0:
            self._process_camera_images()

            if self.camera_images is not None:
                # 这里的提取过程耗时约 0.0023 秒
                self._extract_visual_features()


        # save_interval = self.cfg.get("camera", {}).get("save_interval", 10)  # 默认每10帧保存一次
        # if (self.common_step_counter % save_interval == 0) & self.saveImages:
        #     self._save_camera_images()

        # Store previous ball velocity in world frame *before* refreshing root states for current step
        prev_ball_lin_vel_world = self.root_states[:, 1, 7:10].clone()

        # post physics step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.ball_pos[:] = self.root_states[:, 1, 0:3]
        self.ball_lin_vel[:] = self.body_states[:, self.ball_body_index, 7:10]
        self.ball_ang_vel[:] = self.body_states[:, self.ball_body_index, 10:13]

        self.base_pos[:] = self.root_states[:, 0, 0:3]
        self.base_quat[:] = self.root_states[:, 0, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 0, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel[:] = self.base_lin_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_lin_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self.filtered_ang_vel[:] = self.base_ang_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_ang_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self._refresh_feet_state()

        self.episode_length_buf += 1
        self.min_ball_vel_buf = torch.where(
            self.ball_lin_vel[:, 0] > 0.1,
            self.min_ball_vel_buf + 1.0,
            torch.zeros_like(self.min_ball_vel_buf)
        )
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)

        # ============ 新增：更新步态相位 ============
        self.gait_phase_controller.update(self.dt)

        # 检查课程学习成功
        for env_id in range(self.num_envs):
            if self.curriculum_manager.check_stage_success(
                env_id, self.ball_pos, self.base_pos,
                self.ball_visible, self.ball_center_offset,
                self.projected_gravity, self.ball_lin_vel
            ):
                self.curriculum_manager.increment_success(env_id)
        # ==========================================

        # Update time_since_ball_is_still_buf
        ball_speed_threshold = self.cfg["rewards"].get("ball_stationary_speed_threshold", 0.1) # Configurable threshold
        ball_is_active = torch.norm(self.root_states[:, 1, 7:10], dim=-1) > ball_speed_threshold
        self.time_since_ball_is_still_buf = torch.where(ball_is_active, 
                                                        torch.zeros_like(self.time_since_ball_is_still_buf), 
                                                        self.time_since_ball_is_still_buf + self.dt)
        self.time_since_ball_is_moving_buf = torch.where(~ball_is_active, 
                                                        torch.zeros_like(self.time_since_ball_is_moving_buf), 
                                                        self.time_since_ball_is_moving_buf + self.dt)

        self._kick_robots()
        self._push_robots()
        self._check_termination() # Sets self.reset_buf and potentially self.reset_ball_buf

        # Handle ball-only resets (ball too far, but robot is not resetting)
        ball_only_reset_env_ids = (self.reset_ball_buf & ~self.reset_buf).nonzero(as_tuple=False).flatten()
        if len(ball_only_reset_env_ids) > 0:
            self._reset_ball_at_robot_front(ball_only_reset_env_ids)
            # Update convenience tensors for the reset balls as _compute_observations will use them
            self.ball_pos[ball_only_reset_env_ids] = self.root_states[ball_only_reset_env_ids, 1, 0:3]
            ball_quat_reset = self.root_states[ball_only_reset_env_ids, 1, 3:7]
            self.ball_rot[ball_only_reset_env_ids] = ball_quat_reset
            # Velocities in root_states are world, convert to local for convenience tensors
            world_lin_vel_reset = self.root_states[ball_only_reset_env_ids, 1, 7:10]
            world_ang_vel_reset = self.root_states[ball_only_reset_env_ids, 1, 10:13]
            self.ball_lin_vel[ball_only_reset_env_ids] = quat_rotate_inverse(ball_quat_reset, world_lin_vel_reset)
            self.ball_ang_vel[ball_only_reset_env_ids] = quat_rotate_inverse(ball_quat_reset, world_ang_vel_reset)
            
            self.reset_ball_buf[ball_only_reset_env_ids] = False

        self._compute_reward()

        # Log rewards to CSV file
        self._log_rewards_to_csv()

        # Update last_ball_lin_vel_world *before* potential full reset for next step's calculation
        # For envs that were not reset (neither full nor ball-only), this is their current velocity.
        # For envs that *were* reset (either full or ball-only), their velocity was set to 0.0 during reset,
        # so this correctly reflects their "last" velocity as 0 before the next step.
        self.last_ball_lin_vel_world[:] = self.body_states[:, self.ball_body_index, 7:10]

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._reset_idx(env_ids) # This will call _reset_root_states, which handles ball reset too
            self.reset_ball_buf[env_ids] = False # Ball reset is handled by full reset
            # For fully reset environments, ensure their last_ball_lin_vel_world is also 0 for next step
            self.last_ball_lin_vel_world[env_ids] = 0.0

        #self._teleport_robot()

        self._compute_observations()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 0, 7:13]
        self.last_feet_pos[:] = self.feet_pos

        # print(f"env_resets: {self.env_resets}, env_successes: {self.env_successes}, env_falling: {self.env_falling}")
        # if len(self.ball_velocities) > 0:
        #     print(f"ball_velocities average: {np.mean(self.ball_velocities)}, std: {np.std(self.ball_velocities)}, max: {np.max(self.ball_velocities)}")

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _extract_visual_features(self):

        try:
            self.visual_extractor.model.eval() 
            for env_id in range(self.num_envs):
                # 1. 获取当前环境的 RGB 图像
                raw_rgb = self.camera_images[env_id, 0].cpu().numpy()
                
                # 2. YOLO 推理
                results = self.yolo_model.predict(raw_rgb, conf=0.01, verbose=False)
                
                ball_center = None
                for result in results:
                    # 寻找类别为 'sports ball' (ID: 32) 的检测结果
                    for box in result.boxes:
                        if int(box.cls) == 32:
                            # print(f"⚽ 环境 {env_id} 检测到足球，置信度: {box.conf.item():.2f}")
                            # 获取归一化后的 xywh (x_center, y_center, width, height)
                            xywhn = box.xywhn.cpu().numpy()[0]
                            ball_center = [xywhn[0], xywhn[1]] # 获取中心点 [0~1]
                            break
                
                # 3. 如果识别到足球，调用调整角度的函数
                if ball_center is not None:
                    #print(f"🎯 环境 {env_id} 识别到足球，中心位置: {ball_center}")
                    self.ball_visible[env_id] = 1.0

                    #更新足球中心偏移量和步数计数器
                    err_x = ball_center[0] - 0.5
                    err_y = ball_center[1] - 0.5
                    offset = (err_x ** 2 + err_y ** 2) ** 0.5  # L2 距离，0 表示刚好在中心

                    self.ball_center_offset[env_id] = offset
                    self.steps_since_seen[env_id] = 0.0

                    self._adjust_camera_by_ball(env_id, ball_center)
                else:
                    #print(f"❌ 环境 {env_id} 未识别到足球，保持当前摄像头角度")

                    # 本帧没看到球
                    self.ball_visible[env_id] = 0.0
                    self.ball_center_offset[env_id] = 1.0  # 最大偏差（可以理解为“完全看不到”）
                    self.steps_since_seen[env_id] += 1.0

                    # 如果超过一定步数没看到球，简单的环视逻辑，主动寻找足球
                    max_steps_without_seeing = self.cfg.get("camera",{}).get("max_steps_without_seeing", 50)
                    if self.steps_since_seen[env_id] >= max_steps_without_seeing:
                        #调试
                        #print(f"🔄 环境 {env_id} 超过 {max_steps_without_seeing} 步未见球，执行环视动作")
                        # 环视逻辑：让摄像头缓慢左右摆动
                        sweep_amplitude = 1.5  # 最大偏航角幅度（弧度）
                        sweep_speed = 0.05     # 每步的偏航角增量
                        
                        self.camera_yaw_states[env_id] += sweep_speed
                        if self.camera_yaw_states[env_id] > sweep_amplitude:
                            self.camera_yaw_states[env_id] = -sweep_amplitude  # 回到左侧开始新一轮环视
                        
                        # 构造新的变换
                        local_transform = gymapi.Transform()
                        local_transform.p = gymapi.Vec3(0.4, 0, 0.22) # 保持原安装高度
                        local_transform.r = gymapi.Quat.from_euler_zyx(0, self.camera_pitch_states[env_id].item(), self.camera_yaw_states[env_id].item())

                        # 获取机器人基座句柄并重新挂载摄像头
                        body_handle = self.gym.get_actor_rigid_body_handle(self.envs[env_id], self.actor_handles[env_id], 0)
                        self.gym.attach_camera_to_body(
                            self.camera_handles[env_id][0], self.envs[env_id], body_handle, 
                            local_transform, gymapi.FOLLOW_POSITION
                        )
            
            # 预先定义灰度权重并移至 GPU
            rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=self.device)
            
            for env_id in range(self.num_envs):
                # 1. 立即将原始图像移至 GPU [cite: 251]
                # self.camera_images 通常在 CPU 上，先搬运再计算
                raw_rgb = self.camera_images[env_id, 0].to(self.device).float()
                
                # 2. 在 GPU 上进行灰度转换 (H, W, 3) -> (1, 1, H, W) [cite: 111, 204]
                # 这样产生的 gray 张量天然就在 GPU 上
                gray = (raw_rgb * rgb_weights).sum(dim=-1).unsqueeze(0).unsqueeze(0)
                
                # 3. 在 GPU 上进行缩放至论文指定的 160x120 [cite: 125, 205]
                # 归一化必须在 GPU 张量上进行 [cite: 186]
                input_tensor = F.interpolate(gray, size=(120, 160), mode='bilinear') / 255.0
                
                # 4. 执行推理提取特征 [cite: 204]
                with torch.no_grad():
                    features = self.visual_extractor.model(input_tensor)
                
                # 5. 存入特征缓冲区 (52维)
                self.visual_features_buf[env_id] = features.squeeze(0)
                
        except Exception as e:
            print(f"📷 提取失败: {e}")

    def _adjust_camera_by_ball(self, env_id, ball_center):
        """
        ball_center: [x, y] 范围 0~1。0.5 代表中心。
        """
        # 计算图像中心偏差
        err_x = ball_center[0] - 0.5
        err_y = ball_center[1] - 0.5

        # 增益参数：决定摄像头转动的灵敏度
        # 注意：这里的正负号需根据实际效果微调
        kp_yaw = -0.6   
        kp_pitch = 0.5  

        # 增量更新当前角度
        self.camera_yaw_states[env_id] += err_x * kp_yaw
        self.camera_pitch_states[env_id] += err_y * kp_pitch

        # 角度限制：防止摄像头转动过度（弧度制）
        self.camera_pitch_states[env_id] = torch.clamp(self.camera_pitch_states[env_id], 0.1, 1.4)
        self.camera_yaw_states[env_id] = torch.clamp(self.camera_yaw_states[env_id], -1.5, 1.5)

        # 构造新的变换
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(0.4, 0, 0.22) # 保持原安装高度
        # 应用更新后的 Pitch 和 Yaw
        local_transform.r = gymapi.Quat.from_euler_zyx(0, self.camera_pitch_states[env_id].item(), self.camera_yaw_states[env_id].item())

        # 获取机器人基座句柄并重新挂载摄像头
        body_handle = self.gym.get_actor_rigid_body_handle(self.envs[env_id], self.actor_handles[env_id], 0)
        self.gym.attach_camera_to_body(
            self.camera_handles[env_id][0], self.envs[env_id], body_handle, 
            local_transform, gymapi.FOLLOW_POSITION
        )


    def _save_camera_images(self):
        """保存摄像头图片"""
        try:
            import cv2
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 获取保存间隔配置
            save_interval = self.cfg.get("camera", {}).get("save_interval", 10)
            
            # 保存所有环境的图片
            for env_id in range(self.num_envs):
                # 保存机器人摄像头图片
                if self.camera_images is not None and env_id < len(self.camera_handles):
                    robot_image = self.camera_images[env_id, 0].cpu().numpy()
                    
                    if robot_image is not None and len(robot_image.shape) == 3:
                        # RGB转BGR用于OpenCV保存
                        bgr_image = cv2.cvtColor(robot_image, cv2.COLOR_RGB2BGR)
                        filename = f"{self.camera_save_dir}/robot_env{env_id}_{timestamp}_step{self.common_step_counter}.png"
                        
                        # 添加文本信息
                        height, width = bgr_image.shape[:2]
                        cv2.putText(bgr_image, f"Env{env_id}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(bgr_image, f"Step{self.common_step_counter}", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(bgr_image, f"Interval{save_interval}", (10, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        #是否检测到足球
                        # cv2.putText(bgr_image, f"BallVisible:{int(self.ball_visible[env_id].item())}", (10, 150), 
                        #         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        #偏移量
                        # cv2.putText(bgr_image, f"BallOffset:{self.ball_center_offset[env_id].item():.3f}", (10, 190), 
                        #         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        #偏航角
                        # cv2.putText(bgr_image, f"CamYaw:{self.camera_yaw_states[env_id].item():.3f}", (10, 230), 
                        #         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        cv2.imwrite(filename, bgr_image)
                
                # 保存全局摄像头图片
                if self.global_camera_images is not None:
                    global_image = self.global_camera_images[env_id].cpu().numpy()
                    if global_image is not None and len(global_image.shape) == 3:
                        bgr_image = cv2.cvtColor(global_image, cv2.COLOR_RGB2BGR)
                        filename = f"{self.global_camera_save_dir}/global_env{env_id}_{timestamp}_step{self.common_step_counter}.png"
                        
                        # 添加文本信息
                        height, width = bgr_image.shape[:2]
                        cv2.putText(bgr_image, f"Env{env_id}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(bgr_image, f"Step{self.common_step_counter}", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(bgr_image, f"Interval{save_interval}", (10, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        cv2.imwrite(filename, bgr_image)
            
            # 打印保存信息（避免打印太多信息）
            if self.common_step_counter % (save_interval * 10) == 0:  # 每保存10次打印一次
                print(f"📸 保存摄像头图片 - 步骤 {self.common_step_counter}, 间隔 {save_interval} 帧")
                        
        except Exception as e:
            print(f"❌ 保存摄像头图片失败: {e}")

    def _kick_robots(self):
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            self.root_states[:, 0, 7:10] = apply_randomization(self.root_states[:, 0, 7:10], self.cfg["randomization"].get("kick_lin_vel"))
            self.root_states[:, 0, 10:13] = apply_randomization(self.root_states[:, 0, 10:13], self.cfg["randomization"].get("kick_ang_vel"))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        """Random push the robots. Emulates an impulse by setting a randomized force."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == 0:
            # 调试信息
            # print(f"Debug: pushing_forces shape = {self.pushing_forces.shape}")
            # print(f"Debug: total_bodies_per_env = {self.total_bodies_per_env}")
            # print(f"Debug: Expected shape = [{self.num_envs}, {self.total_bodies_per_env}, 3]")
            
            # 使用正确的刚体索引
            # 假设机器人的基座是第一个刚体（索引0）
            robot_base_body_index = 0
            
            # 只对机器人基座施加力
            self.pushing_forces[:, robot_base_body_index, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, robot_base_body_index, :]),
                self.cfg["randomization"].get("push_force"),
            )
            self.pushing_torques[:, robot_base_body_index, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, robot_base_body_index, :]),
                self.cfg["randomization"].get("push_torque"),
            )
        elif self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == np.ceil(
            self.cfg["randomization"]["push_duration_s"] / self.dt
        ):
            # 重置力
            self.pushing_forces.zero_()
            self.pushing_torques.zero_()
        
        # 重塑张量形状
        # 形状应该是 [num_envs * total_bodies_per_env, 3]
        forces_flat = self.pushing_forces.view(-1, 3)
        torques_flat = self.pushing_torques.view(-1, 3)
        
        # print(f"Debug: forces_flat shape = {forces_flat.shape}")
        # print(f"Debug: Should be [{self.num_envs * self.total_bodies_per_env}, 3] = [{self.num_envs * self.total_bodies_per_env}, 3]")
        
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(forces_flat),
            gymtorch.unwrap_tensor(torques_flat),
            gymapi.LOCAL_SPACE,
        )

    def _refresh_feet_state(self):
        self.feet_pos[:] = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.body_states[:, self.feet_indices, 3:7]
        roll, _, yaw = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_roll[:] = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_yaw[:] = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        feet_edge_relative_pos = (
            to_torch(self.cfg["asset"]["feet_edge_pos"], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self.num_envs, len(self.feet_indices), -1, -1)
        )
        expanded_feet_pos = self.feet_pos.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 3)
        expanded_feet_quat = self.feet_quat.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 4)
        feet_edge_pos = expanded_feet_pos + quat_rotate(expanded_feet_quat, feet_edge_relative_pos.reshape(-1, 3))
        self.feet_contact[:] = torch.any(
            (feet_edge_pos[:, 2] - self.terrain.terrain_heights(feet_edge_pos) < 0.01).reshape(
                self.num_envs, len(self.feet_indices), feet_edge_relative_pos.shape[2]
            ),
            dim=2,
        )

    def _check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.reset_buf |= self.root_states[:, 0, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        self.reset_buf |= self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"]
        self.time_out_buf = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.min_ball_vel_buf > np.ceil(self.cfg["rewards"]["min_ball_vel_s"] / self.dt)
        self.time_out_buf |= self.episode_length_buf == self.cmd_resample_time

        # Add termination if ball is still for too long
        max_ball_still_time = self.cfg["rewards"].get("max_ball_still_time_s", 4.0) # Configurable duration
        self.reset_buf |= self.time_since_ball_is_still_buf > max_ball_still_time

        # Add termination if ball is moving for too long
        max_ball_moving_time = self.cfg["rewards"].get("max_ball_moving_time_s", 4.0) # Configurable duration
        self.reset_buf |= self.time_since_ball_is_moving_buf > max_ball_moving_time

        # count a success if ball is moving for too long
        self.env_successes += torch.sum(self.min_ball_vel_buf > np.ceil(self.cfg["rewards"]["min_ball_vel_s"] / self.dt))
        self.env_falling += torch.sum(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"])
        self.env_falling += torch.sum(self.root_states[:, 0, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"])



    def _compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """

        ball_is_moving = torch.norm(self.ball_lin_vel, dim=-1) >= 0.1  # True if ball is "still"

        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            raw_reward_values = self.reward_functions[i]() # Shape: (num_envs)

            normal_scale = self.reward_scales[name] # Scalar, default scale for this reward

            # Initialize effective_scales with normal_scale. This applies if:
            # 1. No specific ball_rolling_scale is defined for this reward.
            # 2. A specific ball_rolling_scale is defined, but the ball is NOT moving.
            effective_scales_for_envs = torch.full_like(raw_reward_values, normal_scale)

            # Check if a specific scale for "ball moving" scenarios exists for this reward name
            ball_moving_specific_scale = self.reward_scales_ball_rolling.get(name) # Scalar or None

            if ball_moving_specific_scale is not None:
                # A specific scale for a moving ball exists for this reward.
                # We apply this scale IF the ball is moving.
                # If the ball is not moving, effective_scales_for_envs (which is normal_scale) is used.
                effective_scales_for_envs = torch.where(ball_is_moving,
                                                        torch.full_like(raw_reward_values, ball_moving_specific_scale),
                                                        effective_scales_for_envs)

            rew = raw_reward_values * effective_scales_for_envs
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew # Store the final scaled reward
        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)


    def _compute_observations_visual(self):
        """计算观测值（包含视觉特征）"""
        
        # 1. 计算基础观测
        ball_pos_world_frame = self.ball_pos - self.base_pos
        relative_ball_pos = quat_rotate_inverse(self.base_quat, ball_pos_world_frame)
        
        # 基础观测向量
        base_obs_parts = [
            apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
            apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
            apply_randomization(relative_ball_pos[:, 0:2], self.cfg["noise"].get("ball_pos")) * self.cfg["normalization"]["ball_pos"],  
            apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],
            apply_randomization(self.dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
            self.actions,
        ]
        
        base_obs = torch.cat(base_obs_parts, dim=-1)
        
        # 2. 获取视觉特征
        visual_obs = self.visual_features_buf.clone()
        
        # 3. 对视觉特征进行处理
        try:
            # 尝试添加噪声
            visual_noise_params = self.cfg.get("noise", {}).get("visual_noise_params")
            if visual_noise_params is not None:
                visual_obs = apply_randomization(visual_obs, visual_noise_params)
            else:
                # 如果没有配置参数，使用简化的噪声
                visual_noise_scale = self.cfg.get("noise", {}).get("visual_noise", 0.0)
                if visual_noise_scale > 0:
                    # 简单的手动添加噪声
                    noise = torch.randn_like(visual_obs) * visual_noise_scale
                    visual_obs = visual_obs + noise
        except Exception as e:
            print(f"⚠️ 视觉特征随机化失败: {e}")
            # 继续执行，不添加噪声
        
        # 4. 归一化视觉特征
        visual_scale = self.cfg.get("normalization", {}).get("visual_scale", 1.0)
        visual_obs = visual_obs * visual_scale
        
        # 5. 拼接基础观测和视觉观测
        self.obs_buf = torch.cat([base_obs, visual_obs], dim=-1)
        
        # 6. 计算特权观测
        if hasattr(self, 'privileged_obs_buf'):
            # 基础特权观测
            privileged_base_parts = [
                self.base_mass_scaled,
                apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                apply_randomization(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),
                self.ball_lin_vel[:, 0:2],
                self.feet_pos[:, 0, 0:2],
                self.feet_pos[:, 1, 0:2],
                self.pushing_forces[:, 0, :] * self.cfg["normalization"]["push_force"],
                self.pushing_torques[:, 0, :] * self.cfg["normalization"]["push_torque"],
            ]
            
            privileged_base_obs = torch.cat(privileged_base_parts, dim=-1)
            
            # 添加视觉特征到特权观测
            privileged_visual_obs = self.visual_features_buf.clone() * visual_scale
            self.privileged_obs_buf = torch.cat([privileged_base_obs, privileged_visual_obs], dim=-1)
            self.extras["privileged_obs"] = self.privileged_obs_buf
        
        # 7. 调试输出
        if self.common_step_counter % 500 == 0:
            print(f"📊 观测维度: 基础={base_obs.shape[1]}, 视觉={visual_obs.shape[1]}, 总计={self.obs_buf.shape[1]}")
            print(f"  视觉特征范围: [{visual_obs.min():.3f}, {visual_obs.max():.3f}], 均值={visual_obs.mean():.3f}")

        if self.common_step_counter == 0:
            print(f"DEBUG: base_obs shape: {base_obs.shape}")
            print(f"DEBUG: visual_obs shape: {visual_obs.shape}")
            print(f"DEBUG: Total obs_buf shape: {self.obs_buf.shape}") # 确认这里是不是 96

    def _compute_observations(self):
        """计算观测值（优化版：应用 Leaky Top-K 掩码，防止信号闪烁）"""
        
        # 1. 计算基础观测 (保持不变)
        ball_pos_world_frame = self.ball_pos - self.base_pos
        relative_ball_pos = quat_rotate_inverse(self.base_quat, ball_pos_world_frame)
        
        base_obs_parts = [
            apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
            apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
            # 注意：如果这里的噪声太大，机器人就必须依赖视觉。如果视觉不稳定，就会踢空。
            apply_randomization(relative_ball_pos[:, 0:2], self.cfg["noise"].get("ball_pos")) * self.cfg["normalization"]["ball_pos"],  
            apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],
            apply_randomization(self.dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
            self.actions,
        ]
        base_obs = torch.cat(base_obs_parts, dim=-1)
        
        # 2. 获取并初步处理视觉特征
        visual_obs = self.visual_features_buf.clone()
        
        # 3. 噪声处理 (保持不变)
        try:
            visual_noise_params = self.cfg.get("noise", {}).get("visual_noise_params")
            if visual_noise_params is not None:
                visual_obs = apply_randomization(visual_obs, visual_noise_params)
            else:
                visual_noise_scale = self.cfg.get("noise", {}).get("visual_noise", 0.0)
                if visual_noise_scale > 0:
                    visual_obs = visual_obs + torch.randn_like(visual_obs) * visual_noise_scale
        except Exception as e:
            print(f"⚠️ 视觉特征随机化失败: {e}")
            
        # 4. 归一化与特征增强
        visual_scale = self.cfg.get("normalization", {}).get("visual_scale", 10.0)
        # 建议：稍微放大视觉特征的权重，确保网络重视它
        visual_obs = visual_obs * visual_scale

        # --- [优化修改点]: Leaky Top-K 掩码 ---
        k = self.cfg.get("visual", {}).get("top_k", 8)
        # 初始值设为 0.1（而不是 0.0），这样非关键特征依然能提供微弱的上下文信息，
        # 保持梯度连续。放在 if 外面定义，避免 k >= 特征维度时下面打印日志报 NameError
        leak_factor = 0.1

        if k < visual_obs.shape[-1]:
            # 1. 找到 Top-K
            _, indices = torch.topk(visual_obs.abs(), k, dim=-1) # 建议用绝对值 abs() 找能量最大的

            # 2. 创建"漏水"掩码 (Leaky Mask)
            mask = torch.ones_like(visual_obs) * leak_factor

            # 3. 将 Top-K 的位置设为 1.0
            mask.scatter_(-1, indices, 1.0)

            # 4. 应用掩码
            visual_obs = visual_obs * mask
        # ----------------------------------------
        
        # 5. 拼接
        # print(f"📊 基础观测形状: {base_obs.shape}")
        # print(f"📊 基础观测: {base_obs}")
        # print(f"📊 视觉特征形状: {visual_obs.shape}")
        # print(f"📊 视觉特征: {visual_obs}")
        self.obs_buf = torch.cat([base_obs, visual_obs], dim=-1)
        
        # 6. 特权观测 (保持不变)
        if hasattr(self, 'privileged_obs_buf'):
            privileged_base_parts = [
                self.base_mass_scaled,
                apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                apply_randomization(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),
                self.ball_lin_vel[:, 0:2],
                self.feet_pos[:, 0, 0:2],
                self.feet_pos[:, 1, 0:2],
                self.pushing_forces[:, 0, :] * self.cfg["normalization"]["push_force"],
                self.pushing_torques[:, 0, :] * self.cfg["normalization"]["push_torque"],
            ]
            privileged_base_obs = torch.cat(privileged_base_parts, dim=-1)
            # 特权观测通常不需要掩码，给 Critic 看完整信息有助于它评估状态价值
            # 如果显存允许，建议这里用 self.visual_features_buf.clone() 而不是处理后的 visual_obs
            self.privileged_obs_buf = torch.cat([privileged_base_obs, visual_obs], dim=-1)
            self.extras["privileged_obs"] = self.privileged_obs_buf
        
        # 7. 调试输出
        if self.common_step_counter % 1000 == 0:
             print(f"📊 Top-K Leaky Mask: 保留前 {k} 个特征 (1.0), 其余特征缩放为 {leak_factor}")


    # ------------ reward functions----------------
    def _reward_survival(self):
        # Reward survival
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axes)
       return torch.exp(-torch.square(0 - self.filtered_lin_vel[:, 0]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axes)
        return torch.exp(-torch.square(0 - self.filtered_lin_vel[:, 1]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        return torch.exp(-torch.square(0 - self.filtered_ang_vel[:, 2]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_base_height(self):
        # Tracking of base height
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        return torch.square(base_height - self.cfg["rewards"]["base_height_target"])

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 1.0, dim=-1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.filtered_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

    def _reward_root_acc(self):
        # Penalize root accelerations
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 0, 7:13]) / self.dt), dim=-1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=-1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg["rewards"]["soft_dof_vel_limit"]).clip(min=0.0, max=1.0),
            dim=-1,
        )

    def _reward_torque_limits(self):
        # Penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg["rewards"]["soft_torque_limit"]).clip(min=0.0),
            dim=-1,
        )

    def _reward_torque_tiredness(self):
        # Penalize torque tiredness
        return torch.sum(torch.square(self.torques / self.torque_limits).clip(max=1.0), dim=-1)

    def _reward_power(self):
        # Penalize power
        return torch.sum((self.torques * self.dof_vel).clip(min=0.0), dim=-1)

    def _reward_feet_slip(self):
        # Penalize feet velocities when contact
        return (
            torch.sum(
                torch.square((self.last_feet_pos - self.feet_pos) / self.dt).sum(dim=-1) * self.feet_contact.float(),
                dim=-1,
            )
            * (self.episode_length_buf > 1).float()
        )

    def _reward_feet_vel_z(self):
        return torch.sum(torch.square((self.last_feet_pos - self.feet_pos) / self.dt)[:, :, 2], dim=-1)

    def _reward_feet_roll(self):
        return torch.sum(torch.square(self.feet_roll), dim=-1)

    def _reward_feet_yaw_diff(self):
        return torch.square((self.feet_yaw[:, 1] - self.feet_yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_yaw_mean(self):
        feet_yaw_mean = self.feet_yaw.mean(dim=-1) + torch.pi * (torch.abs(self.feet_yaw[:, 1] - self.feet_yaw[:, 0]) > torch.pi)
        return torch.square((get_euler_xyz(self.base_quat)[2] - feet_yaw_mean + torch.pi) % (2 * torch.pi) - torch.pi)

    
    def _reward_left_feet_x(self):
        # Calculate the distance between feet on the x-axis
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_x_distance =  torch.abs(
            torch.cos(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0])
            - torch.sin(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1])
        )
        
        # Get the reference distance from config
        target_distance = self.cfg["rewards"]["feet_distance_ref_x"]
        
        # Normalize the difference between actual and target distance
        normalized_diff = -torch.abs(feet_x_distance - target_distance)
        
        reward = torch.exp(2.0 * (normalized_diff - 1.0) + 2)  # Exponential increase

        #print(f"X-axis: feet_x_distance={feet_x_distance.mean().item():.4f}, target={target_distance:.4f}, normalized_diff={normalized_diff.mean().item():.4f}, reward={reward.mean().item():.4f}")
        
        return reward
    
    def _reward_left_feet_y(self):
        # Calculate the distance between feet on the y-axis
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_y_distance = torch.abs(
            torch.sin(base_yaw) * (self.feet_pos[:, 1, 1] - self.feet_pos[:, 0, 1])
            + torch.cos(base_yaw) * (self.feet_pos[:, 1, 0] - self.feet_pos[:, 0, 0])
        )
        
        # Get the reference distance from config
        target_distance = self.cfg["rewards"]["feet_distance_ref_y"]
        
        # Normalize the difference between actual and target distance
        normalized_diff = -torch.abs(feet_y_distance - target_distance)
        
        reward = torch.exp(2.0 * (normalized_diff - 1.0) + 2)  # Exponential increase
        
        #print(f"Y-axis: feet_y_distance={feet_y_distance.mean().item():.4f}, target={target_distance:.4f}, normalized_diff={normalized_diff.mean().item():.4f}, reward={reward.mean().item():.4f}")
        
        return reward
    
    def _reward_ball_velocity_target_direction(self):
        """Rewards kicking the ball towards a target position in the world frame."""
        # cfg["rewards"]["ball_target_position"] - e.g. [5.0, 0.0, 0.0] (target position in world space)
        # cfg["rewards"]["max_ball_vel_target_reward"] - max reward for this component
        # cfg["rewards"]["ball_vel_target_direction_sigma"] - for scaling the reward
        # cfg["rewards"]["ball_velocity_decay_time"] - time constant for exponential decay when ball is moving
        
        # Get the target position from config
        target_position = to_torch([5.0, 0.0, 0.05], device=self.device).unsqueeze(0)
        
        # Get current ball position and velocity (in world frame)
        ball_pos_world = self.body_states[:, self.ball_body_index, 0:3]
        ball_vel_world = self.body_states[:, self.ball_body_index, 7:10]
        
        # Calculate the direction vector from ball to target
        ball_to_target = target_position - ball_pos_world
        distance_to_target = torch.norm(ball_to_target, dim=-1, keepdim=True)
        
        # Normalize the direction vector (avoid division by zero)
        ball_to_target_normalized = ball_to_target / (distance_to_target + 1e-6)
        
        # Project ball velocity onto the target direction
        velocity_towards_target = torch.sum(ball_vel_world * ball_to_target_normalized, dim=-1)
        
        # Reward only positive velocity towards the target
        # Using an exponential function for a smoother reward landscape
        sigma = self.cfg["rewards"].get("ball_vel_target_direction_sigma", 1.0)
        base_reward = velocity_towards_target
        
        # Add decay factor based on how long the ball has been moving
        decay_time_constant = self.cfg["rewards"].get("ball_velocity_decay_time", 2.0)  # Time constant in seconds
        decay_factor = torch.exp(-self.time_since_ball_is_moving_buf / decay_time_constant)
        
        # Apply decay to the reward
        reward = base_reward * decay_factor
        
        # Clamp the reward to avoid excessively large values
        max_reward = self.cfg["rewards"].get("max_ball_vel_target_reward", 5.0)
        
        return torch.clamp(reward, min=0.0, max=max_reward)

    def _reward_kicking_foot_approach_ball_stationary(self):
        """Rewards moving the kicking foot towards the ball, only if the ball is stationary."""
        # cfg["rewards"]["ball_stationary_speed_threshold"] - Max speed for ball to be "stationary"
        # cfg["rewards"]["approach_proximity_sigma"] - For exp decay of distance reward
        # cfg["rewards"]["max_approach_reward"]
        # cfg["rewards"]["foot_velocity_towards_ball_scale"] - Scale for velocity reward
        # cfg["rewards"]["foot_velocity_weight"] - Weight for velocity vs proximity reward (0-1)

        current_ball_pos_world = self.body_states[:, self.ball_body_index, 0:3]

        foot_ball_dist_left = torch.norm(self.feet_pos[:, 0, :] - current_ball_pos_world, dim=-1)
        foot_ball_dist_right = torch.norm(self.feet_pos[:, 1, :] - current_ball_pos_world, dim=-1)

        foot_ball_dist = torch.min(foot_ball_dist_left, foot_ball_dist_right)

        # Proximity reward (existing)
        proximity_sigma = self.cfg["rewards"].get("approach_proximity_sigma", 0.1)
        proximity_value = torch.exp(-foot_ball_dist / proximity_sigma) 
        
        # Only give reward if the ball is stationary
        reward = proximity_value

        max_reward = self.cfg["rewards"].get("max_approach_reward", 2.0)
        return torch.clamp(reward, min=0.0, max=max_reward)

    def _reward_body_alignment_for_kick(self):
        """Rewards aligning the robot's body towards the ball and a target."""
        # cfg["rewards"]["kick_target_pos_world"] - e.g., [5.0, 0.0, 0.0] (a point in world space)
        # cfg["rewards"]["alignment_to_ball_sigma"]
        # cfg["rewards"]["alignment_to_target_sigma"]
        # cfg["rewards"]["max_alignment_reward"]

        robot_pos_world = self.base_pos
        robot_quat_world = self.base_quat
        ball_pos_world = self.root_states[:, 1, 0:3]
        
        # Robot's forward vector in world frame
        # Assuming robot's local forward is X-axis: [1,0,0]
        robot_forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        robot_forward_world = quat_rotate(robot_quat_world, robot_forward_local)
        
        # Vector from robot to ball
        robot_to_ball_world = ball_pos_world - robot_pos_world
        robot_to_ball_world_normalized = robot_to_ball_world / (torch.norm(robot_to_ball_world, dim=-1, keepdim=True) + 1e-6)

        # Alignment with a fixed target position
        kick_target_pos_world = to_torch(self.cfg["rewards"].get("kick_target_pos_world", [5.0, 0.0, self.ball_radius]), device=self.device).unsqueeze(0)
        robot_to_target_world = kick_target_pos_world - robot_pos_world
        robot_to_target_world_normalized = robot_to_target_world / (torch.norm(robot_to_target_world, dim=-1, keepdim=True) + 1e-6)
        
        alignment_to_target = torch.sum(robot_forward_world * robot_to_target_world_normalized, dim=-1)
        sigma_target = self.cfg["rewards"].get("alignment_to_target_sigma", 0.5)
        reward_align_target = torch.exp((alignment_to_target - 1.0) / sigma_target)


        max_reward = self.cfg["rewards"].get("max_alignment_reward", 1.0)
        return torch.clamp(reward_align_target, min=0.0, max=max_reward)
    
    def _reward_body_angle(self):
        base_pitch, base_roll, base_yaw = get_euler_xyz(self.base_quat)
        pitch_normalized = (base_pitch + torch.pi) % (2 * torch.pi) - torch.pi
        roll_normalized = (base_roll + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Calculate absolute distance from 0
        pitch_distance = torch.abs(pitch_normalized)
        roll_distance = torch.abs(roll_normalized)
        
        # Calculate penalty (current implementation)
        penalty = torch.square(pitch_distance) + torch.square(roll_distance)
        
        # Convert to positive reward (decreasing from 1 to 0)
        reward = 1.0 / (0.1 + penalty**2) - 1.0
        
        return reward

    def _reward_ball_acceleration(self):
        """Rewards ball acceleration towards the target direction, encouraging effective kicks."""
        # Get current and previous ball velocities in world frame
        current_ball_vel_world = self.body_states[:, self.ball_body_index, 7:9]
        prev_ball_vel_world = self.last_ball_lin_vel_world[:,:2]
        
        # Calculate ball acceleration (change in velocity / time)
        ball_acceleration = (current_ball_vel_world - prev_ball_vel_world) / self.dt

        ball_effective_acceleration = ball_acceleration[:, 0] - torch.abs(ball_acceleration[:, 1])
        
        # Get parameters from config with defaults
        acceleration_scale = self.cfg["rewards"].get("ball_acceleration_scale", 10.0)
        max_acceleration_reward = self.cfg["rewards"].get("max_ball_acceleration_reward", 1.0)
        
        # Only reward positive acceleration towards target
        # Using tanh for smooth, bounded rewardball_effective_acceleration
        reward = torch.tanh(torch.clamp(ball_effective_acceleration, min=0.0) / acceleration_scale) * max_acceleration_reward
        
        return reward

    def _reward_waiting(self):
        """Reward that increases quadratically with time elapsed in the episode."""
        # Get current progress through episode (0 to 1)
        progress = self.episode_length_buf / np.ceil(self.cfg["rewards"]["max_ball_still_time_s"] / self.dt)
        
        # Quadratic reward from 0 to 1 based on progress
        reward = progress * progress
        
        return reward
    
    def _reward_ball_visible_align(self):
        """
        球在视野中时，鼓励它出现在画面中央。
        ball_center_offset 越小，奖励越大；看不到球时奖励为 0。
        """

        # 0 表示刚好在中心，1 左右表示偏得比较远
        # 在_extract_visual_features中存储了self.ball_center_offset = (err_x ** 2 + err_y ** 2) ** 0.5
        offset = self.ball_center_offset  # 0 ~ 1 左右
        visible = self.ball_visible       # 0 or 1

        # 控制衰减速度的参数，数值越大，对偏差越敏感
        alpha = self.cfg["rewards"].get("ball_align_sigma", 5.0)

        # 形状：offset=0 时 ~1，偏离变大时指数衰减
        align_reward = torch.exp(-alpha * offset)

        # 只在看到球的时候生效，没看到球的时候visible==0
        return align_reward * visible
    
    def _reward_blind_speed_penalty(self):
        """
        长时间看不到球时，对机器人在地面平面上的速度给惩罚。
        返回的是速度大小 v_xy,真正的惩罚由 YAML 里的负权重决定。
        """
        # 连续多少步没看到球才开始惩罚
        steps_threshold = self.cfg["rewards"].get("steps_since_seen_threshold", 10)

        # mask: 只有当 steps_since_seen > 阈值 且当前看不到球时才生效
        cond = (self.steps_since_seen > steps_threshold) * (self.ball_visible < 0.5)
        mask = cond.float()

        # 机器人本体在 x-y 平面的速度（已经做了滤波）
        v_xy = torch.norm(self.filtered_lin_vel[:, :2], dim=-1)  # >= 0

        # 返回正数，scale 设为负即为惩罚
        return v_xy * mask

    # ============ 新增：步态相位相关奖励函数 ============

    def _reward_gait_phase_alignment(self):
        """
        奖励：鼓励机器人在正确的相位执行踢球动作
        当相位在摆动相（0.5-1.0）且接近球时给予奖励
        """
        phase = self.gait_phase_controller.phase
        # 只比较水平面距离：两边都取前两维，否则 [N,3]-[N,2] 无法广播会直接报错
        dist_to_ball = torch.norm(self.ball_pos[:, :2] - self.base_pos[:, :2], dim=1)

        # 在摆动相（0.5-1.0）且接近球（<0.3m）时给奖励
        in_swing_phase = (phase > 0.5).float()
        close_to_ball = (dist_to_ball < 0.3).float()

        reward = in_swing_phase * close_to_ball
        return reward

    def _reward_stance_stability(self):
        """
        奖励：在支撑相（0.0-0.5）保持稳定，减少晃动
        """
        phase = self.gait_phase_controller.phase
        ang_vel = torch.norm(self.base_ang_vel[:, :2], dim=1)

        # 在支撑相时奖励低角速度
        in_stance_phase = (phase < 0.5).float()
        stability_reward = torch.exp(-3.0 * ang_vel)

        reward = in_stance_phase * stability_reward
        return reward

    def _reward_smooth_phase_transition(self):
        """
        奖励：平滑的相位过渡，避免突然的动作变化
        """
        dof_vel = self.dof_vel
        dof_acc = (dof_vel - self.last_dof_vel) / self.dt

        # 惩罚过大的关节加速度
        acc_penalty = torch.sum(torch.square(dof_acc), dim=1)
        reward = torch.exp(-0.01 * acc_penalty)

        return reward

    def _reward_swing_leg_clearance(self):
        """
        奖励：在摆动相抬起腿，避免拖地
        """
        phase = self.gait_phase_controller.phase

        # 膝关节下标在 _init_buffers 里按名字查好（T1 的膝关节是 3 和 9，不是 2 和 8）
        knee_bend = torch.abs(self.dof_pos[:, self.knee_dof_indices]).sum(dim=-1)

        # 在摆动相时奖励膝关节弯曲（抬腿）
        in_swing_phase = (phase > 0.5).float()

        reward = in_swing_phase * knee_bend * 0.1
        return reward

    def _reward_contact_timing(self):
        """
        奖励：在正确的相位时脚应该接触地面
        支撑相（0.0-0.5）应该有脚接触，摆动相（0.5-1.0）不应该有
        """
        phase = self.gait_phase_controller.phase

        in_stance = (phase < 0.5).float()
        in_swing = (phase >= 0.5).float()

        # 直接用 _refresh_feet_state 算好的触地标志，而不是返回一个常数
        num_contacts = self.feet_contact.float().sum(dim=-1)
        both_contact = (num_contacts >= 2).float()   # 双脚触地
        any_contact = (num_contacts >= 1).float()    # 至少一只脚触地

        # 支撑相：双脚触地得分；摆动相：只有一只脚触地（另一只抬起）得分
        reward = in_stance * both_contact + in_swing * any_contact * (1.0 - both_contact)

        return reward

    def _reward_kick_impact_timing(self):
        """
        奖励：在相位接近1.0时（摆动相末期）与球接触
        这是最佳的踢球时机
        """
        phase = self.gait_phase_controller.phase
        # 同样只比较水平面距离，避免 [N,3]-[N,2] 的广播错误
        dist_to_ball = torch.norm(self.ball_pos[:, :2] - self.base_pos[:, :2], dim=1)
        ball_vel = torch.norm(self.ball_lin_vel, dim=1)

        # 最佳踢球相位：0.85-1.0
        optimal_phase = (phase > 0.85).float()

        # 球开始移动表示踢中了
        ball_kicked = (ball_vel > 0.5).float()

        # 距离很近
        very_close = (dist_to_ball < 0.15).float()

        reward = optimal_phase * ball_kicked * very_close * 10.0
        return reward

    # ====================================================
