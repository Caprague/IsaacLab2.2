import os
import numpy as np
import torch

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


class SimulationDataSaver:
    def __init__(self, save_path_root, data_type, sub_dir_name, max_sequence=100, T_max=10):
        """初始化仿真数据保存类"""
        if data_type not in ['pcd', 'npz']:
            raise ValueError("data_type must be 'pcd' or 'npz'")
        self.save_path_root = save_path_root
        self.data_type = data_type
        self.sub_dir_name = sub_dir_name
        self.max_sequence = max_sequence
        self.T_max = T_max
        # 标记是否已经输入设定了环境数量，设定后才能保存数据
        self.initialed_flag = False
        self.num_envs = -1

        # 创建根目录
        os.makedirs(self.save_path_root, exist_ok=True)
        # 创建子目录
        self.sub_dir_path = os.path.join(self.save_path_root, self.sub_dir_name)
        os.makedirs(self.sub_dir_path, exist_ok=True)

        # 用于存储位姿数据的字典
        self.pose_data_dict = {}

    def save_data(self, env_ids, *args):
        """保存数据接口"""
        if not self.initialed_flag:
            print(f"Warning: num_envs has not been initialized. Please call set_num_envs() before saving data.")
            return
        if len(env_ids) == 0:
            return

        valid_env_ids = []
        valid_indices = []
        for idx, env_id in enumerate(env_ids):
            if int(env_id) >= self.num_envs or int(env_id) < 0:
                print(f"Warning: env_id {int(env_id)} out of range. Skipping.")
                continue
            if self.T_counters[int(env_id)] > self.T_max:
                if not self.done[int(env_id)]:
                    print(f"Env {int(env_id)} has reached the maximum collection period (Subdir={self.sub_dir_name}, Type={self.data_type}, T_max={self.T_max}). Stopping data collection.")
                    self.done[int(env_id)] = True
                continue
            valid_env_ids.append(int(env_id))
            valid_indices.append(idx)

        if len(valid_env_ids) == 0:
            return

        # 根据有效索引裁剪输入数据
        valid_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                valid_arg = torch.index_select(arg, 0, torch.tensor(valid_indices, device=arg.device))
                valid_args.append(valid_arg)
            else:
                raise TypeError(f"Unsupported argument type in save_data: {type(arg)}. Only torch.Tensor is supported !!!")

        # 使用有效的 env_ids 和 数据进行保存
        if self.data_type == 'pcd':
            if len(valid_args) != 2:
                raise ValueError("For pcd type, save_data expects 2 arguments")
            self._save_point_cloud_with_mask(valid_env_ids, valid_args[0], valid_args[1])
        elif self.data_type == 'npz':
            if len(valid_args) != 2:
                raise ValueError("For npz type, save_data expects 2 arguments")
            self._save_pose_separate(valid_env_ids, valid_args[0], valid_args[1])

    def _save_point_cloud_with_mask(self, env_ids, point_cloud_tensor: torch.Tensor, mask_tensor: torch.Tensor):
        """保存带掩码的点云数据 (保持原逻辑不变)"""
        if len(point_cloud_tensor.shape) != 3 or point_cloud_tensor.shape[2] != 3:
            raise ValueError(f"Point cloud tensor shape should be NxBx3, got {point_cloud_tensor.shape}")
        if len(mask_tensor.shape) != 2:
            raise ValueError(f"Mask tensor shape should be NxB, got {mask_tensor.shape}")
        if len(env_ids) != point_cloud_tensor.shape[0]:
            raise ValueError(f"Mismatched shapes: point_cloud_tensor first dim {point_cloud_tensor.shape[0]} does not match env_ids length {len(env_ids)}")

        # 处理每个环境的数据
        for idx, env_id in enumerate(env_ids):
            current_T = self.T_counters[int(env_id)]
            current_num_in = self.num_in_counters[int(env_id)]

            # 检查是否需要更新周期（达到最大序列长度）
            if current_num_in >= self.max_sequence:
                self.update_period([int(env_id)])
                # 更新后的计数器值
                current_T = self.T_counters[int(env_id)]
                current_num_in = self.num_in_counters[int(env_id)]

            if current_T > self.T_max:
                return

            # 获取数据
            env_point_cloud = point_cloud_tensor[idx].cpu().numpy()  # Bx3
            env_mask = mask_tensor[idx].cpu().numpy()  # B

            # 应用掩码
            valid_indices = env_mask.astype(bool)
            valid_points = env_point_cloud[valid_indices]  # Mx3

            # 确定保存路径
            subdir_name = f"%04d" % ((current_T - 1) * self.num_envs + int(env_id))
            subdir_path = os.path.join(self.sub_dir_path, subdir_name)
            os.makedirs(subdir_path, exist_ok=True)
            file_name = f"{current_num_in:03d}.pcd"
            file_path = os.path.join(subdir_path, file_name)

            # 保存点云数据
            self._save_point_cloud_file(valid_points, file_path)

            # 更新该环境的计数器
            self.num_in_counters[int(env_id)] += 1

    def _save_point_cloud_file(self, points, filename):
        """保存点云文件"""
        if OPEN3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(filename, pcd)
        else:
            npy_filename = filename.replace('.pcd', '.npy')
            np.save(npy_filename, points)

    def _save_pose_separate(self, env_ids, pos_tensor: torch.Tensor, quat_tensor: torch.Tensor):
        """
        保存位姿数据：内部缓存 -> 追加新数据 -> 立即写入文件（覆盖模式）
        """
        if len(pos_tensor.shape) != 2 or pos_tensor.shape[1] != 3:
            raise ValueError(f"Position tensor shape should be Nx3, got {pos_tensor.shape}")
        if len(quat_tensor.shape) != 2 or quat_tensor.shape[1] != 4:
            raise ValueError(f"Quaternion tensor shape should be Nx4, got {quat_tensor.shape}")
        if len(env_ids) != pos_tensor.shape[0]:
            raise ValueError(f"Mismatched shapes: pos_tensor first dim {pos_tensor.shape[0]} does not match env_ids length {len(env_ids)}")

        # 处理每个环境的数据
        for idx, env_id in enumerate(env_ids):
            current_T = self.T_counters[int(env_id)]
            current_num_in = self.num_in_counters[int(env_id)]

            # 检查是否需要更新周期（达到最大序列长度）
            if current_num_in >= self.max_sequence:
                self.update_period([int(env_id)])
                current_T = self.T_counters[int(env_id)]
                current_num_in = self.num_in_counters[int(env_id)] # 重置后的计数器

            if current_T > self.T_max:
                return

            # 获取当前帧数据
            env_pos = pos_tensor[idx:idx+1, :].cpu().numpy()  # 1x3
            env_quat = quat_tensor[idx:idx+1, :].cpu().numpy()  # 1x4

            # --- 内部存储与实时写入逻辑 ---
            dict_key = int(env_id)
            
            # 1. 初始化或追加数据到内存字典
            if dict_key not in self.pose_data_dict:
                # 首次数据
                self.pose_data_dict[dict_key] = {
                    'pos': env_pos,
                    'quat': env_quat,
                    'current_T': current_T
                }
            else:
                # 检查是否跨周期了（理论上 update_period 已处理，这里双重保险）
                if self.pose_data_dict[dict_key]['current_T'] != current_T:
                    self.pose_data_dict[dict_key] = {
                        'pos': env_pos,
                        'quat': env_quat,
                        'current_T': current_T
                    }
                else:
                    # 追加数据
                    self.pose_data_dict[dict_key]['pos'] = np.vstack([
                        self.pose_data_dict[dict_key]['pos'], env_pos
                    ])
                    self.pose_data_dict[dict_key]['quat'] = np.vstack([
                        self.pose_data_dict[dict_key]['quat'], env_quat
                    ])

            # 计算当前文件索引
            file_idx = ((current_T - 1) * self.num_envs) + int(env_id)
            file_path = os.path.join(self.sub_dir_path, f"{file_idx:04d}.npz")
            
            # 获取当前内存中的最新数据并写入（覆盖写入）
            current_data = self.pose_data_dict[dict_key]
            np.savez(file_path, 
                     pos=current_data['pos'], 
                     quat=current_data['quat'])
            
            # 更新计数器
            self.num_in_counters[int(env_id)] += 1

    def set_num_envs(self, num_envs: int = 1):
        """设置仿真环境的总数量，并标记初始化完成。"""
        self.num_envs = num_envs
        self.T_counters = [1] * num_envs
        self.num_in_counters = [0] * num_envs
        self.initialed_flag = True
        self.done = [False] * num_envs

    def update_period(self, env_ids=None):
        """更新输入周期 T，并重置输入数量 num_in"""
        if env_ids is None:
            env_ids = range(self.num_envs)
        for env_id in env_ids:
            if int(env_id) < self.num_envs:
                if self.T_counters[int(env_id)] <= self.T_max:
                    self.T_counters[int(env_id)] += 1
                self.num_in_counters[int(env_id)] = 0

    def reset_input_counter(self, env_ids=None):
        """
        重置指定环境的输入计数器。
        1. 清除内部缓存 (pose_data_dict)。
        2. 清空外部文件内容（删除文件）。
        3. 重置计数器。
        """
        if env_ids is None:
            env_ids = range(self.num_envs)

        for env_id in env_ids:
            if int(env_id) < self.num_envs:
                if int(env_id) in self.pose_data_dict:
                    del self.pose_data_dict[int(env_id)]
                self.num_in_counters[env_id] = 0

    def get_current_stats(self):
        """获取当前统计信息"""
        return {
            'num_in_counters': self.num_in_counters,
            'max_sequence': self.max_sequence,
            'T_counters': self.T_counters,
            'T_max': self.T_max
        }

