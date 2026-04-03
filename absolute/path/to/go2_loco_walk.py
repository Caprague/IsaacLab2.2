    def _compute_propriception_obs(self, command: np.ndarray, history_length: int = 1, noise_enable: bool = False):
        """
        根据观测配置文件中的proprioception观测组，按次序进行观测项计算、历史长度堆叠、clip和scale
        
        Args:
            command: 控制命令
            history_length: 观测历史长度
            noise_enable: 是否启用噪声（暂时不考虑）
            
        Returns:
            处理后的观测向量
        """
        # 确保观测历史字典中存在proprioception组
        if 'proprioception' not in self._obs_history:
            self._obs_history['proprioception'] = {}
            
        # 获取proprioception观测组的配置
        proprio_config = self._obs_config.get('proprioception', {})
        obs_items = proprio_config.get('items', {})
        
        # 如果配置中指定了history_length，使用配置中的值
        config_history_length = proprio_config.get('history_length', history_length)
        
        # 存储当前帧的所有观测项
        current_obs = {}
        
        # 1. 计算各个观测项的值
        # 控制命令
        if 'velocity_commands' in obs_items:
            current_obs['velocity_commands'] = command.copy()
            
        # 线速度（机体坐标系）
        if 'lin_vel' in obs_items:
            lin_vel_I = self.robot.get_linear_velocity()
            pos_IB, q_IB = self.robot.get_world_pose()
            R_IB = quat_to_rot_matrix(q_IB)
            R_BI = R_IB.transpose()
            lin_vel_b = np.matmul(R_BI, lin_vel_I)
            current_obs['lin_vel'] = lin_vel_b
            
        # 角速度（机体坐标系）
        if 'ang_vel' in obs_items:
            ang_vel_I = self.robot.get_angular_velocity()
            pos_IB, q_IB = self.robot.get_world_pose()
            R_IB = quat_to_rot_matrix(q_IB)
            R_BI = R_IB.transpose()
            ang_vel_b = np.matmul(R_BI, ang_vel_I)
            current_obs['ang_vel'] = ang_vel_b
            
        # 重力向量（机体坐标系）
        if 'projected_gravity' in obs_items:
            pos_IB, q_IB = self.robot.get_world_pose()
            R_IB = quat_to_rot_matrix(q_IB)
            R_BI = R_IB.transpose()
            gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))
            current_obs['projected_gravity'] = gravity_b
            
        # 关节位置偏差
        if 'joint_pos' in obs_items:
            current_obs['joint_pos'] = self.robot.get_joint_positions() - self.default_pos
            
        # 关节速度
        if 'joint_vel' in obs_items:
            current_obs['joint_vel'] = self.robot.get_joint_velocities()
            
        # 上一动作
        if 'prev_action' in obs_items:
            current_obs['prev_action'] = self._previous_action.copy()
            
        # 地形高度扫描
        if 'height_scan' in obs_items:
            current_obs['height_scan'] = self._get_height_scan()
        
        # 2. 更新观测历史
        for item_name, obs_value in current_obs.items():
            if item_name not in self._obs_history['proprioception']:
                self._obs_history['proprioception'][item_name] = []
            
            # 添加当前观测到历史记录
            self._obs_history['proprioception'][item_name].append(obs_value)
            
            # 保持历史记录长度不超过指定的历史长度
            while len(self._obs_history['proprioception'][item_name]) > config_history_length:
                self._obs_history['proprioception'][item_name].pop(0)
        
        # 3. 按照配置中的顺序处理观测项（应用clip和scale，并堆叠历史）
        processed_obs = []
        
        for item_name, item_config in obs_items.items():
            if item_name in self._obs_history['proprioception']:
                # 获取该项的历史观测
                history_data = self._obs_history['proprioception'][item_name]
                
                # 对每个历史时间步的数据应用clip和scale
                for data in history_data:
                    # 复制数据以避免修改原始数据
                    processed_data = data.copy()
                    
                    # 应用clip
                    if 'clip' in item_config:
                        clip_min, clip_max = item_config['clip']
                        processed_data = np.clip(processed_data, clip_min, clip_max)
                    
                    # 应用scale
                    if 'scale' in item_config:
                        processed_data = processed_data * item_config['scale']
                    
                    # 添加到处理后的观测向量
                    processed_obs.extend(processed_data)
        
        # 转换为numpy数组
        return np.array(processed_obs)

    def _compute_observation(self, command):
        """
        计算完整的观测向量
        """
        # 仅使用proprioception观测组
        proprio_obs = self._compute_propriception_obs(command)
        
        # 可以在这里添加其他观测组的处理
        
        return proprio_obs