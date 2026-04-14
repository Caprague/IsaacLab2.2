import os
import numpy as np
import torch
from pathlib import Path

from data_collector import SimulationDataSaver

# 测试代码
if __name__ == "__main__":
    import shutil

    def test_pcd_saver():
        print("=== 测试PCD数据保存（带掩码）===")
        
        # 创建测试根目录
        root_dir = "./test_pcd_data"
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        
        # 创建PCD保存器
        saver = SimulationDataSaver(save_path_root=root_dir, data_type='pcd', sub_dir_name='pointclouds')
        
        # 生成测试点云数据和掩码
        point_cloud_1 = torch.randn(5, 100, 3)
        mask_1 = torch.rand(5, 100) > 0.3  # 随机掩码，约70%的点有效
        print(f"保存第一批次点云数据 (shape: {point_cloud_1.shape}), 掩码(shape: {mask_1.shape})")
        saver.save_data(point_cloud_1, mask_1)
        
        # 再保存一批数据
        point_cloud_2 = torch.randn(5, 100, 3)
        mask_2 = torch.rand(5, 100) > 0.2  # 随机掩码，约80%的点有效
        print(f"保存第二批次点云数据 (shape: {point_cloud_2.shape}), 掩码(shape: {mask_2.shape})")
        saver.save_data(point_cloud_2, mask_2)
        
        # 再保存一批数据
        point_cloud_3 = torch.randn(5, 100, 3)
        mask_3 = torch.rand(5, 100) > 0.4  # 随机掩码，约60%的点有效
        print(f"保存第三批次点云数据 (shape: {point_cloud_3.shape}), 掩码(shape: {mask_3.shape})")
        saver.save_data(point_cloud_3, mask_3)
        
        # 更新周期
        print("更新周期 T -> T+1")
        saver.update_period()
        
        # 再保存一批数据
        point_cloud_4 = torch.randn(5, 100, 3)
        mask_4 = torch.rand(5, 100) > 0.5  # 随机掩码，约50%的点有效
        print(f"保存第四批次点云数据 (shape: {point_cloud_4.shape}), 掩码(shape: {mask_4.shape}) after period update")
        saver.save_data(point_cloud_4, mask_4)
        
        # 再保存一批数据
        point_cloud_5 = torch.randn(5, 100, 3)
        mask_5 = torch.rand(5, 100) > 0.5  # 随机掩码，约50%的点有效
        print(f"保存第五批次点云数据 (shape: {point_cloud_5.shape}), 掩码(shape: {mask_5.shape}) after period update")
        saver.save_data(point_cloud_5, mask_5)
        
        # 检查生成的目录结构
        print("\n生成的目录结构:")
        for dirpath, dirnames, filenames in os.walk(root_dir):
            level = dirpath.replace(root_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(dirpath)}/")
            subindent = ' ' * 2 * (level + 1)
            for filename in filenames:
                print(f"{subindent}{filename}")
        
        print()

    def test_npz_saver():
        print("=== 测试NPZ数据保存（分离存储pos和quat）===")
        
        # 创建测试根目录
        root_dir = "./test_npz_data"
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        
        # 创建NPZ保存器
        saver = SimulationDataSaver(save_path_root=root_dir, data_type='npz', sub_dir_name='poses')
        
        # 生成测试位置和姿态数据
        pos_1 = torch.randn(5, 3)  # 位置 (x, y, z)
        quat_1 = torch.randn(5, 4)  # 四元数 (qx, qy, qz, qw)
        # 标准化四元数
        quat_1 = torch.nn.functional.normalize(quat_1, p=2, dim=1)
        print(f"保存第一批次位姿数据 pos(shape: {pos_1.shape}), quat(shape: {quat_1.shape})")
        saver.save_data(pos_1, quat_1)
        
        # 再保存一批数据 (模拟时间序列)
        pos_2 = torch.randn(5, 3)
        quat_2 = torch.randn(5, 4)
        quat_2 = torch.nn.functional.normalize(quat_2, p=2, dim=1)
        print(f"保存第二批次位姿数据 pos(shape: {pos_2.shape}), quat(shape: {quat_2.shape})")
        saver.save_data(pos_2, quat_2)
        
        # 再保存一批数据 (模拟时间序列)
        pos_3 = torch.randn(5, 3)
        quat_3 = torch.randn(5, 4)
        quat_3 = torch.nn.functional.normalize(quat_3, p=2, dim=1)
        print(f"保存第三批次位姿数据 pos(shape: {pos_3.shape}), quat(shape: {quat_3.shape})")
        saver.save_data(pos_3, quat_3)
        
        # 更新周期
        print("更新周期 T -> T+1")
        saver.update_period()
        
        # 再保存一批数据
        pos_4 = torch.randn(5, 3)
        quat_4 = torch.randn(5, 4)
        quat_4 = torch.nn.functional.normalize(quat_4, p=2, dim=1)
        print(f"保存第四批次位姿数据 pos(shape: {pos_4.shape}), quat(shape: {quat_4.shape}) after period update")
        saver.save_data(pos_4, quat_4)
        
        # 再保存一批数据
        pos_5 = torch.randn(5, 3)
        quat_5 = torch.randn(5, 4)
        quat_5 = torch.nn.functional.normalize(quat_5, p=2, dim=1)
        print(f"保存第五批次位姿数据 pos(shape: {pos_5.shape}), quat(shape: {quat_5.shape}) after period update")
        saver.save_data(pos_5, quat_5)
        
        # 检查生成的目录结构
        print("\n生成的目录结构:")
        for dirpath, dirnames, filenames in os.walk(root_dir):
            level = dirpath.replace(root_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(dirpath)}/")
            subindent = ' ' * 2 * (level + 1)
            for filename in filenames:
                print(f"{subindent}{filename}")
        
        # 验证NPZ文件内容
        print("\n验证NPZ文件内容:")
        for filename in os.listdir(os.path.join(root_dir, 'poses')):
            if filename.endswith('.npz'):
                filepath = os.path.join(root_dir, 'poses', filename)
                data = np.load(filepath)
                print(f"File {filename}: pos shape = {data['pos'].shape}, quat shape = {data['quat'].shape}")
        
        print()

    def test_edge_cases():
        print("=== 测试边界情况 ===")
        
        # 测试错误的数据类型
        try:
            invalid_saver = SimulationDataSaver("./invalid_test", "invalid_type", "subdir")
        except ValueError as e:
            print(f"捕获到预期错误: {e}")
        
        # 测试错误的点云形状
        root_dir = "./edge_case_test"
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        
        saver = SimulationDataSaver(root_dir, "pcd", "test_pcd")
        try:
            invalid_pc = torch.randn(10, 3)  # 错误形状，应该是NxMx3
            invalid_mask = torch.rand(10, 3)  # 错误形状，应该是NxM
            saver.save_data(invalid_pc, invalid_mask)
        except ValueError as e:
            print(f"捕获到预期错误: {e}")
        
        # 测试错误的掩码形状
        try:
            invalid_pc = torch.randn(5, 100, 3)  # 正确形状
            invalid_mask = torch.rand(3, 50)  # 错误形状，与点云不匹配
            saver.save_data(invalid_pc, invalid_mask)
        except ValueError as e:
            print(f"捕获到预期错误: {e}")
        
        # 测试错误的位姿形状
        try:
            invalid_pos = torch.randn(10, 5)  # 错误形状，应该是Nx3
            invalid_quat = torch.randn(10, 4)  # 正确形状
            saver.save_data(invalid_pos, invalid_quat)
        except ValueError as e:
            print(f"捕获到预期错误: {e}")
        
        # 测试错误的位姿形状
        try:
            invalid_pos = torch.randn(10, 3)  # 正确形状
            invalid_quat = torch.randn(8, 4)  # 错误形状，与位置不匹配
            saver.save_data(invalid_pos, invalid_quat)
        except ValueError as e:
            print(f"捕获到预期错误: {e}")
        
        print()

    print("开始测试SimulationDataSaver类\n")
    
    test_pcd_saver()
    test_npz_saver()
    test_edge_cases()
    
    print("所有测试完成！")
    
    # 清理测试文件
    test_dirs = ["./test_pcd_data", "./test_npz_data", "./edge_case_test"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"已清理测试目录: {test_dir}")
