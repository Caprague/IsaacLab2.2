import os
import sys
import numpy as np
from pathlib import Path
import argparse
import time

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("错误: 未找到Open3D库，请运行: pip install open3d")
    sys.exit(1)


class DualPointCloudVisualizer:
    def __init__(self, root_path_1, root_path_2=None, play_delya=0.1):
        self.root_path_1 = Path(root_path_1)
        self.root_path_2 = Path(root_path_2) if root_path_2 else None
        
        self.last_next_time = 0
        self.playback_delay = play_delya  # 播放延迟（秒）
        
        self.pcd_files_1 = []
        self.pcd_files_2 = []
        
        self._find_pcd_files(self.root_path_1, self.pcd_files_1, "窗口1")
        
        if self.root_path_2:
            self._find_pcd_files(self.root_path_2, self.pcd_files_2, "窗口2")
        else:
            pass

    def _find_pcd_files(self, root_path, file_list, window_name):
        if not root_path.exists():
            print(f"错误: 路径不存在 - {root_path}")
            return

        direct_pcds = list(root_path.glob("*.pcd"))
        
        if direct_pcds:
            file_list.extend(direct_pcds)
            print(f"[{window_name}] 检测到模式1: 直接加载 {len(direct_pcds)} 个文件")
        else:
            for pcd_file in root_path.rglob("*.pcd"):
                file_list.append(pcd_file)
            print(f"[{window_name}] 检测到模式2: 递归加载 {len(file_list)} 个文件")

        file_list.sort(key=lambda x: (x.parent.name, x.stem))

    def normalize_point_cloud(self, pcd):
        if not pcd.points:
            return pcd
            
        points = np.asarray(pcd.points)
        max_coord = np.max(np.abs(points))
        
        if max_coord > 10.0:
            min_bound = points.min(axis=0)
            max_bound = points.max(axis=0)
            center = (min_bound + max_bound) / 2.0
            scale = max_bound - min_bound
            max_scale = np.max(scale)
            if max_scale == 0: max_scale = 1.0
            points = (points - center) / max_scale
            pcd.points = o3d.utility.Vector3dVector(points)
        else:
            center = pcd.get_center()
            pcd.translate(-center)
            
        return pcd

    def check_normalization(self, pcd, filename):
        """检查点云是否在-0.5到0.5的归一化范围内"""
        if not pcd.points:
            return True
            
        points = np.asarray(pcd.points)
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        
        if np.any(min_vals < -0.51) or np.any(max_vals > 0.51):
            print(f"警告: 文件 {filename} 的点云未在[-0.5, 0.5]范围内 - X:[{min_vals[0]:.3f},{max_vals[0]:.3f}], Y:[{min_vals[1]:.3f},{max_vals[1]:.3f}], Z:[{min_vals[2]:.3f},{max_vals[2]:.3f}]")
            return False
        return True

    def visualize(self, start_idx=0, end_idx=None):
        max_len_1 = len(self.pcd_files_1)
        max_len_2 = len(self.pcd_files_2)
        
        if max_len_1 == 0:
            print("错误: 未找到任何数据")
            return

        if end_idx is None:
            end_idx = max(max_len_1, max_len_2)

        # --- 创建窗口 1 ---
        vis1 = o3d.visualization.VisualizerWithKeyCallback()
        vis1.create_window(window_name="视图 1 (主)", width=960, height=540, left=0, top=0)
        pcd1 = o3d.geometry.PointCloud()
        vis1.add_geometry(pcd1)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0, 0, 0])
        vis1.add_geometry(coordinate_frame)
        
        # --- 创建窗口 2 ---
        vis2 = None
        pcd2 = None
        if self.root_path_2 and max_len_2 > 0:
            vis2 = o3d.visualization.VisualizerWithKeyCallback()
            vis2.create_window(window_name="视图 2 (对比)", width=960, height=540, left=960, top=0)
            pcd2 = o3d.geometry.PointCloud()
            vis2.add_geometry(pcd2)
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0, 0, 0])
            vis2.add_geometry(coordinate_frame)

        current_idx = start_idx
        is_first_frame = True

        def update_single_window(vis, pcd_obj, file_list, idx, color=[0.2, 0.8, 0.2]):
            if idx >= len(file_list):
                return False
            
            file_path = file_list[idx]
            if idx == current_idx: 
                print(f"[{vis.get_window_name()}] 加载: {file_path.name}")
            
            try:
                loaded_pcd = o3d.io.read_point_cloud(str(file_path))
                if len(loaded_pcd.points) == 0:
                    return False

                if not self.check_normalization(loaded_pcd, file_path.name):
                    loaded_pcd = self.normalize_point_cloud(loaded_pcd)
                loaded_pcd.paint_uniform_color(color)
                loaded_pcd.estimate_normals()

                pcd_obj.points = loaded_pcd.points
                pcd_obj.colors = loaded_pcd.colors
                pcd_obj.normals = loaded_pcd.normals
                
                vis.update_geometry(pcd_obj)
                
                if is_first_frame:
                    vis.reset_view_point(True)
                
                vis.poll_events()
                vis.update_renderer()
                return True
            except Exception as e:
                print(f"加载失败: {e}")
                return False

        def next_callback(vis):
            nonlocal current_idx, is_first_frame
            
            current_time = time.time()
            if current_time - self.last_next_time < self.playback_delay:
                return  # 如果时间间隔小于延迟，则不执行操作
            
            self.last_next_time = current_time  # 更新最后执行时间
            
            update_single_window(vis1, pcd1, self.pcd_files_1, current_idx, [0.2, 0.8, 0.2])
            
            if vis2 is not None:
                if current_idx < len(self.pcd_files_2):
                    update_single_window(vis2, pcd2, self.pcd_files_2, current_idx, [0.8, 0.2, 0.2])
            
            current_idx += 1
            is_first_frame = False

        def prev_callback(vis):
            nonlocal current_idx
            current_idx -= 1
            if current_idx < 0: current_idx = 0
            next_callback(vis)

        def restart_callback(vis):
            nonlocal current_idx, is_first_frame
            current_idx = 0
            is_first_frame = True
            next_callback(vis)

        for vis in [vis1, vis2] if vis2 else [vis1]:
            vis.register_key_callback(ord("N"), next_callback)
            vis.register_key_callback(ord("P"), prev_callback)
            vis.register_key_callback(ord("R"), restart_callback)
            vis.register_key_callback(ord("Q"), lambda vis: vis.close())

        print("\n" + "="*40)
        print("双路可视化已启动")
        print("按 'N' 下一帧 (同步) | 按 'P' 上一帧 | 按 'R' 重置")
        print("="*40)
        
        # 初始加载
        next_callback(vis1)
        
        if vis2:
            try:
                while True:
                    vis1.poll_events()
                    vis1.update_renderer()
                    vis2.poll_events()
                    vis2.update_renderer()
            except:
                pass
        else:
            vis1.run()
            
        # 确保销毁窗口
        vis1.destroy_window()
        if vis2:
            vis2.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="点云双路可视化工具")
    parser.add_argument("--root_path_1", type=str, required=True, help="第一个数据根目录")
    parser.add_argument("--root_path_2", type=str, default=None, help="第二个数据根目录 (可选)")
    parser.add_argument("--start_idx", type=int, default=0, help="起始索引")
    args = parser.parse_args()
    
    try:
        visualizer = DualPointCloudVisualizer(args.root_path_1, args.root_path_2)
        visualizer.visualize(args.start_idx)
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  pcd 点云序列可视化工具 (支持单/双路对比)")
    print("="*60)
    
    if len(sys.argv) == 1:
        print("用法示例:")
        print("  1. 单路模式:")
        print("     python3 data_visualizer.py --root_path_1 /path/to/your/data")
        print("\n  2. 双路对比模式:")
        print("     python3 data_visualizer.py --root_path_1 /path/to/data_A --root_path_2 /path/to/data_B")
        print("\n  3. 指定起始帧:")
        print("     python3 data_visualizer.py --root_path_1 /path/to/data --start_idx 100")
        print("\n快捷键说明:")
        print("  [N] 下一帧 (双路同步) | [P] 上一帧 | [R] 重置 | [Q] 退出")
        print("="*60 + "\n")
    else:
        parser = argparse.ArgumentParser(description="点云双路可视化工具")
        parser.add_argument("--root_path_1", type=str, required=True, help="第一个数据根目录")
        parser.add_argument("--root_path_2", type=str, default=None, help="第二个数据根目录 (可选)")
        parser.add_argument("--start_idx", type=int, default=0, help="起始索引")
        args = parser.parse_args()
        
        try:
            visualizer = DualPointCloudVisualizer(args.root_path_1, args.root_path_2, 0.05)
            visualizer.visualize(args.start_idx)
        except Exception as e:
            print(f"运行时错误: {e}")

