import os
import numpy as np
import torch

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

import isaaclab.utils.math as math_utils

class SimulationDataSaver:
    def __init__(self, save_path_root, data_type, sub_dir_name, pose_mode='world'):
        """
        初始化仿真数据保存类
        
        Args:
            save_path_root (str): 保存数据的根目录
            data_type (str): 数据类型，'pcd' 或 'npz'
            sub_dir_name (str): 子目录名称
            pose_mode (str): 位姿存储模式，'world' 表示世界坐标系，'relative' 表示相对坐标系
        """
        if data_type not in ['pcd', 'npz']:
            raise ValueError("data_type must be 'pcd' or 'npz'")
        
        if pose_mode not in ['world', 'relative']:
            raise ValueError("pose_mode must be 'world' or 'relative'")
        
        self.save_path_root = save_path_root
        self.data_type = data_type
        self.sub_dir_name = sub_dir_name
        self.pose_mode = pose_mode
        
        # 创建根目录
        os.makedirs(self.save_path_root, exist_ok=True)
        
        # 创建子目录
        self.sub_dir_path = os.path.join(self.save_path_root, self.sub_dir_name)
        os.makedirs(self.sub_dir_path, exist_ok=True)
        
        # 初始化计数器
        self.num_in = 0
        self.T = 0  # 输入周期
        self.pose_data_dict = {}  # 用于存储位姿数据，key为环境编号，value为累积的位姿数据
        self.expected_num_envs = None  # 记录期望的环境数量
        
        # 用于相对坐标系的上一帧位姿存储 (使用torch tensor)
        if self.pose_mode == 'relative':
            self.prev_poses = {}  # 存储每个环境的上一帧位姿

    def _create_subdirectories(self, start_idx, end_idx):
        """
        创建二级子目录
        
        Args:
            start_idx (int): 起始索引
            end_idx (int): 结束索引
        """
        for i in range(start_idx, end_idx):
            dir_name = f"{i:04d}"
            dir_path = os.path.join(self.sub_dir_path, dir_name)
            os.makedirs(dir_path, exist_ok=True)

    def save_data(self, *args):
        """
        保存数据接口
        根据数据类型决定输入参数：
        - pcd: 需要两个参数 (point_cloud_tensor, mask_tensor)
        - npz: 需要两个参数 (pos_tensor, quat_tensor)
        """
        if self.T < 1:
            return
        if self.data_type == 'pcd':
            if len(args) != 2:
                raise ValueError("For pcd type, save_data expects 2 arguments: (point_cloud_tensor, mask_tensor)")
            self._save_point_cloud_with_mask(args[0], args[1])
        elif self.data_type == 'npz':
            if len(args) != 2:
                raise ValueError("For npz type, save_data expects 2 arguments: (pos_tensor, quat_tensor)")
            self._save_pose_separate(args[0], args[1])
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")

    def _save_point_cloud_with_mask(self, point_cloud_tensor: torch.Tensor, mask_tensor: torch.Tensor):
        """
        保存带掩码的点云数据
        
        Args:
            point_cloud_tensor (torch.Tensor): 形状为 NxBx3 的点云数据
            mask_tensor (torch.Tensor): 形状为 NxB 的掩码数据
        """
        if len(point_cloud_tensor.shape) != 3 or point_cloud_tensor.shape[2] != 3:
            raise ValueError(f"Point cloud tensor shape should be NxBx3, got {point_cloud_tensor.shape}")
        
        if len(mask_tensor.shape) != 2:
            raise ValueError(f"Mask tensor shape should be NxB, got {mask_tensor.shape}")
        
        if point_cloud_tensor.shape[0] != mask_tensor.shape[0] or point_cloud_tensor.shape[1] != mask_tensor.shape[1]:
            raise ValueError(f"Mismatched shapes: point_cloud_tensor {point_cloud_tensor.shape}, mask_tensor {mask_tensor.shape}")
        
        N = point_cloud_tensor.shape[0]  # num_envs
        B = point_cloud_tensor.shape[1]
        
        # 验证环境数量一致性
        if self.expected_num_envs is None:
            self.expected_num_envs = N
        else:
            assert N == self.expected_num_envs, f"Number of environments mismatch! Expected {self.expected_num_envs}, got {N}"
        
        # 更新计数器
        self.num_in += 1
        
        # 创建二级子目录
        start_idx = (self.T - 1) * N + 1
        end_idx = (self.T - 1) * N + N + 1
        self._create_subdirectories(start_idx, end_idx)
        
        # 拆分并保存每个环境的数据
        for i in range(N):
            env_point_cloud = point_cloud_tensor[i].cpu().numpy()  # Bx3
            env_mask = mask_tensor[i].cpu().numpy()  # B
            
            # 应用掩码，获取有效点云
            valid_indices = env_mask.astype(bool)
            valid_points = env_point_cloud[valid_indices]  # Mx3 where M <= B
            
            # 确定保存路径
            subdir_idx = (self.T - 1) * N + i + 1
            subdir_name = f"{subdir_idx:04d}"
            subdir_path = os.path.join(self.sub_dir_path, subdir_name)
            
            # 保存文件名
            file_name = f"{self.num_in:03d}.pcd"
            file_path = os.path.join(subdir_path, file_name)
            
            # 保存点云数据
            self._save_point_cloud_file(valid_points, file_path)

    def _save_point_cloud_file(self, points, filename):
        """
        保存点云文件，如果安装了open3d则使用open3d，否则保存为numpy格式
        """
        if OPEN3D_AVAILABLE:
            # 使用open3d保存
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(filename, pcd)
        else:
            # 如果没有open3d，保存为numpy格式
            npy_filename = filename.replace('.pcd', '.npy')
            np.save(npy_filename, points)

    def _calculate_frame_difference(self, current_pos, current_quat, prev_pos, prev_quat):
        """
        计算相对于上一帧的位姿差分
        
        Args:
            current_pos: 当前位置 [x, y, z] (torch tensor)
            current_quat: 当前姿态 [w, x, y, z] (标量在前) (torch tensor)
            prev_pos: 上一帧位置 [x, y, z] (torch tensor)
            prev_quat: 上一帧姿态 [w, x, y, z] (标量在前) (torch tensor)
        
        Returns:
            diff_pos: 位置差分 (numpy array)
            diff_quat: 姿态差分 (归一化) (numpy array)
        """
        # 计算姿态差分
        # 姿态差分 = prev_quat^(-1) * current_quat
        prev_quat_inv = math_utils.quat_inv(prev_quat)  # 假设quat_inv也支持(w, x, y, z)
        diff_quat = math_utils.quat_mul(prev_quat_inv, current_quat)  # 假设quat_mul也支持(w, x, y, z)
        
        # 计算位置差分
        # 位置差分 = R_prev^(-1) * (current_pos - prev_pos)
        pos_diff = current_pos - prev_pos
        
        # 将位置差转换为四元数形式 [0, dx, dy, dz]
        pos_diff_quat = torch.cat([torch.zeros_like(pos_diff[..., :1]), pos_diff], dim=-1)
        
        # 旋转位置差: q_inv * pos_diff_quat * q
        temp = math_utils.quat_mul(prev_quat_inv, pos_diff_quat)
        rotated_pos_diff = math_utils.quat_mul(temp, prev_quat)
        
        # 提取旋转后的位置差 (虚部)
        diff_pos = rotated_pos_diff[..., 1:]
        
        # 归一化四元数
        diff_quat_norm = torch.norm(diff_quat, dim=-1, keepdim=True)
        diff_quat = diff_quat / diff_quat_norm
        
        # 最后才转换为numpy
        return diff_pos.cpu().numpy(), diff_quat.cpu().numpy()

    def _save_pose_separate(self, pos_tensor: torch.Tensor, quat_tensor: torch.Tensor):
        """
        分别保存位置和姿态数据
        
        Args:
            pos_tensor (torch.Tensor): 形状为 Nx3 的位置数据 (x, y, z)
            quat_tensor (torch.Tensor): 形状为 Nx4 的姿态数据 (w, x, y, z)
        """
        if len(pos_tensor.shape) != 2 or pos_tensor.shape[1] != 3:
            raise ValueError(f"Position tensor shape should be Nx3, got {pos_tensor.shape}")
        
        if len(quat_tensor.shape) != 2 or quat_tensor.shape[1] != 4:
            raise ValueError(f"Quaternion tensor shape should be Nx4, got {quat_tensor.shape}")
        
        if pos_tensor.shape[0] != quat_tensor.shape[0]:
            raise ValueError(f"Mismatched shapes: pos_tensor {pos_tensor.shape}, quat_tensor {quat_tensor.shape}")
        
        N = pos_tensor.shape[0]  # num_envs
        
        # 验证环境数量一致性
        if self.expected_num_envs is None:
            self.expected_num_envs = N
        else:
            assert N == self.expected_num_envs, f"Number of environments mismatch! Expected {self.expected_num_envs}, got {N}"
        
        # 更新计数器
        self.num_in += 1
        
        # 拆分并处理每个环境的数据
        for i in range(N):
            env_pos = pos_tensor[i:i+1, :]  # 1x3 (保持torch tensor)
            env_quat = quat_tensor[i:i+1, :]  # 1x4 (保持torch tensor)
            
            env_key = (self.T - 1) * N + i + 1
            
            # 根据模式处理位姿数据
            if self.pose_mode == 'world':
                # 直接保存绝对世界位姿（不使用差分）
                # 这样保存的位姿与点云坐标系一致，便于恢复
                processed_pos = env_pos.cpu().numpy()  # 1x3
                processed_quat = env_quat.cpu().numpy()  # 1x4 (w, x, y, z)
            else:  # self.pose_mode == 'relative'
                if env_key not in self.prev_poses:
                    # 首次记录，直接保存为初始位姿 (实际上应为单位姿态)
                    # 第一帧存储为零位姿
                    processed_pos = np.array([[0.0, 0.0, 0.0]])  # 1x3
                    processed_quat = np.array([[1.0, 0.0, 0.0, 0.0]])  # 1x4 (w, x, y, z)
                    # 保存当前位姿作为下一帧的参考 (格式为 w, x, y, z)
                    self.prev_poses[env_key] = {
                        'pos': env_pos.clone(),
                        'quat': env_quat.clone()
                    }
                else:
                    # 计算相对于上一帧的位姿差分
                    prev_pos = self.prev_poses[env_key]['pos']
                    prev_quat = self.prev_poses[env_key]['quat']
                    
                    # 计算位姿差分 (保持在torch中处理)
                    diff_pos, diff_quat = self._calculate_frame_difference(
                        env_pos[0], env_quat[0], prev_pos[0], prev_quat[0]
                    )
                    
                    processed_pos = diff_pos.reshape(1, -1)  # 1x3
                    processed_quat = diff_quat.reshape(1, -1)  # 1x4
                    
                    # 更新上一帧位姿为当前位姿 (格式为 w, x, y, z)
                    self.prev_poses[env_key]['pos'] = env_pos.clone()
                    self.prev_poses[env_key]['quat'] = env_quat.clone()

            # 获取或初始化该环境的数据
            env_key = (self.T - 1) * N + i + 1
            if env_key not in self.pose_data_dict:
                # 初始化时存储pos和quat分离的数据
                self.pose_data_dict[env_key] = {
                    'pos': processed_pos,
                    'quat': processed_quat
                }
            else:
                # 追加到现有数据
                self.pose_data_dict[env_key]['pos'] = np.vstack([
                    self.pose_data_dict[env_key]['pos'], 
                    processed_pos
                ])
                self.pose_data_dict[env_key]['quat'] = np.vstack([
                    self.pose_data_dict[env_key]['quat'], 
                    processed_quat
                ])
            
            # 保存文件 - 存储pos和quat两个数组
            file_path = os.path.join(self.sub_dir_path, f"{env_key:04d}.npz")
            np.savez(file_path, 
                     pos=self.pose_data_dict[env_key]['pos'],
                     quat=self.pose_data_dict[env_key]['quat'])

    def update_period(self):
        """
        更新输入周期 T，并重置输入数量 num_in
        """
        self.T += 1
        self.num_in = 0

        if self.pose_mode == 'relative':
            self.prev_poses = {}

    def get_current_stats(self):
        """
        获取当前统计信息
        """
        return {
            'num_in': self.num_in,
            'T': self.T,
            'expected_num_envs': self.expected_num_envs,
            'pose_mode': self.pose_mode
        }

    def reset_poses_baseline(self):
        """
        重置所有环境的上一帧位姿，主要用于相对坐标系模式
        """
        if self.pose_mode == 'relative':
            self.prev_poses = {}

