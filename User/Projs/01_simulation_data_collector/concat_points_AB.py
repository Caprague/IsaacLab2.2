import open3d as o3d
import numpy as np
import torch
import os


@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def transform_points(
    points: torch.Tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Transform input points in a given frame to a target frame.

    This function transform points from a source frame to a target frame. The transformation is defined by the
    position :math:`t` and orientation :math:`R` of the target frame in the source frame.

    .. math::
        p_{target} = R_{target} \times p_{source} + t_{target}

    If the input `points` is a batch of points, the inputs `pos` and `quat` must be either a batch of
    positions and quaternions or a single position and quaternion. If the inputs `pos` and `quat` are
    a single position and quaternion, the same transformation is applied to all points in the batch.

    If either the inputs :attr:`pos` and :attr:`quat` are None, the corresponding transformation is not applied.

    Args:
        points: Points to transform. Shape is (N, P, 3) or (P, 3).
        pos: Position of the target frame. Shape is (N, 3) or (3,).
            Defaults to None, in which case the position is assumed to be zero.
        quat: Quaternion orientation of the target frame in (w, x, y, z). Shape is (N, 4) or (4,).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        Transformed points in the target frame. Shape is (N, P, 3) or (P, 3).

    Raises:
        ValueError: If the inputs `points` is not of shape (N, P, 3) or (P, 3).
        ValueError: If the inputs `pos` is not of shape (N, 3) or (3,).
        ValueError: If the inputs `quat` is not of shape (N, 4) or (4,).
    """
    points_batch = points.clone()
    # check if inputs are batched
    is_batched = points_batch.dim() == 3
    # -- check inputs
    if points_batch.dim() == 2:
        points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
    if points_batch.dim() != 3:
        raise ValueError(f"Expected points to have dim = 2 or dim = 3: got shape {points.shape}")
    if not (pos is None or pos.dim() == 1 or pos.dim() == 2):
        raise ValueError(f"Expected pos to have dim = 1 or dim = 2: got shape {pos.shape}")
    if not (quat is None or quat.dim() == 1 or quat.dim() == 2):
        raise ValueError(f"Expected quat to have dim = 1 or dim = 2: got shape {quat.shape}")
    # -- rotation
    if quat is not None:
        # convert to batched rotation matrix
        rot_mat = matrix_from_quat(quat)
        if rot_mat.dim() == 2:
            rot_mat = rot_mat[None]  # (3, 3) -> (1, 3, 3)
        # convert points to matching batch size (N, P, 3) -> (N, 3, P)
        # and apply rotation
        points_batch = torch.matmul(rot_mat, points_batch.transpose_(1, 2))
        # (N, 3, P) -> (N, P, 3)
        points_batch = points_batch.transpose_(1, 2)
    # -- translation
    if pos is not None:
        # convert to batched translation vector
        if pos.dim() == 1:
            pos = pos[None, None, :]  # (3,) -> (1, 1, 3)
        else:
            pos = pos[:, None, :]  # (N, 3) -> (N, 1, 3)
        # apply translation
        points_batch += pos
    # -- return points in same shape as input
    if not is_batched:
        points_batch = points_batch.squeeze(0)  # (1, P, 3) -> (P, 3)

    return points_batch


def read_pcd(file_path):
    """读取pcd文件并打印点云信息"""
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return None
    
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        print(f"\nPCD文件: {file_path}")
        print(f"点数量: {len(pcd.points)}")
        if len(pcd.colors) > 0:
            print(f"颜色数量: {len(pcd.colors)}")
        if len(pcd.normals) > 0:
            print(f"法线数量: {len(pcd.normals)}")
        
        return pcd
    except Exception as e:
        print(f"读取PCD文件时出错: {e}")
        return None


def read_and_print_npz(file_path):
    """读取npz文件并打印内容"""
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return None
    
    try:
        data = np.load(file_path)
        print(f"\nNPZ文件: {file_path}")
        print("包含的数组:")
        for key in data.files:
            array = data[key]
            print(f"  - {key}: shape={array.shape}, dtype={array.dtype}")
            print(f"    内容: {array}")
        
        return data
    except Exception as e:
        print(f"读取NPZ文件时出错: {e}")
        return None


def create_coordinate_frame(size=1.0, origin=[0, 0, 0]):
    """创建一个坐标系框架可视化对象"""
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size,
        origin=origin
    )
    return coordinate_frame


class PointCloudProcessor:
    def __init__(self, scale=3.2, bound=1.0):
        """
        初始化处理器
        :param scale: 归一化缩放因子 (对应之前的 3.2)
        :param bound: 内部点判定边界 (对应之前的 [-0.5, 0.5])
        """
        self.scale = scale
        self.bound = bound
        self.half_bound = bound / 2 # 如果你的逻辑是基于 0.5 的范围

    def pcd_to_tensor(self, pcd):
        """将Open3D点云转换为PyTorch张量"""
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if len(pcd.colors) > 0 else None
        normals = np.asarray(pcd.normals) if len(pcd.normals) > 0 else None
        
        points_tensor = torch.from_numpy(points).float() * self.scale
        colors_tensor = torch.from_numpy(colors).float() if colors is not None else None
        normals_tensor = torch.from_numpy(normals).float() if normals is not None else None
        
        return points_tensor, colors_tensor, normals_tensor

    def tensor_to_pcd(self, points_tensor, colors_tensor=None, normals_tensor=None):
        """将PyTorch张量转换回Open3D点云"""
        points = points_tensor.numpy() # 转回原始尺度
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors_tensor is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors_tensor.numpy())
        if normals_tensor is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals_tensor.numpy())
            
        return pcd

    def process_frame(self, pcd_t1, pcd_t2, pos_t1, quat_t1, pos_t2, quat_t2):
        """
        核心处理流水线
        :param pcd_t1: Open3D PointCloud, 当前帧点云
        :param pcd_t2: Open3D PointCloud, 参考帧点云
        :param pos_t1: Tensor (3,), t1时刻的位置
        :param quat_t1: Tensor (4,), t1时刻的四元数(wxyz)
        :param pos_t2: Tensor (3,), t2时刻的位置
        :param quat_t2: Tensor (4,), t2时刻的四元数(wxyz)
        :return: Dict containing 'inside_pcd', 'outside_pcd', 'ref_pcd'
        """
        # 1. 转换为 Tensor
        tensor_t1, colors_t1, normals_t1 = self.pcd_to_tensor(pcd_t1)
        tensor_t2, colors_t2, normals_t2 = self.pcd_to_tensor(pcd_t2)

        # 2. 计算 t2 的旋转矩阵 (用于逆变换)
        rot_matrix_2 = matrix_from_quat(quat_t2)

        # 3. 变换到世界坐标系
        points_world_t1 = transform_points(tensor_t1, pos_t1, quat_t1)
        points_world_t2 = transform_points(tensor_t2, pos_t2, quat_t2)

        # 4. 变换到 t2 坐标系 (P_t2 = R_t2^T * (P_world - t_t2))
        points_world_t1_centered = points_world_t1 - pos_t2
        points_world_t2_centered = points_world_t2 - pos_t2
        
        tensor_t1_view = torch.matmul(points_world_t1_centered, rot_matrix_2)
        tensor_t2_view = torch.matmul(points_world_t2_centered, rot_matrix_2)

        # 5. 归一化
        tensor_t1_view_norm = tensor_t1_view / self.scale
        tensor_t2_view_norm = tensor_t2_view / self.scale

        # 6. 内外点分割 (Mask)
        mask_inside = (tensor_t1_view_norm.abs() < self.half_bound).all(dim=1)

        # 7. 转回 Open3D PointCloud
        pcd_inside = self.tensor_to_pcd(tensor_t1_view_norm[mask_inside], colors_t1[mask_inside] if colors_t1 is not None else None)
        pcd_outside = self.tensor_to_pcd(tensor_t1_view_norm[~mask_inside], colors_t1[~mask_inside] if colors_t1 is not None else None)
        pcd_ref = self.tensor_to_pcd(tensor_t2_view_norm, colors_t2, normals_t2)

        # 8. 上色 (可选，也可以在可视化端处理)
        pcd_inside.paint_uniform_color([1, 0, 0])       # 红色: 内部点
        pcd_outside.paint_uniform_color([0, 0, 1])    # 蓝色: 外部点
        pcd_ref.paint_uniform_color([0, 1, 0])          # 绿色: 参考点云
        pcd_inside.estimate_normals()
        pcd_outside.estimate_normals()
        pcd_outside.estimate_normals()

        return {
            "inside_pcd": pcd_inside,
            "outside_pcd": pcd_outside,
            "ref_pcd": pcd_ref
        }


if __name__ == "__main__":
    # 设置文件路径（请根据实际情况修改）
    TIME_IDX = 50
    SUBDIR_IDX = 35
    
    t1 = TIME_IDX
    t2 = TIME_IDX + 20
    pcd_file_t1 = f"/home/gms/Isaac/IsaacLab2.2/DataCollection/Meta/complete/%04d/%03d.pcd" % (SUBDIR_IDX, t1)
    pcd_file_t2 = f"/home/gms/Isaac/IsaacLab2.2/DataCollection/Meta/complete/%04d/%03d.pcd" % (SUBDIR_IDX, t2)
    npz_file = "/home/gms/Isaac/IsaacLab2.2/DataCollection/Meta/transform/%04d.npz" % SUBDIR_IDX
    # pcd_file_t1 = f"/home/gms/Isaac/IsaacLab2.2/DataCollection/Meta/partial/%04d/%03d.pcd" % (SUBDIR_IDX, t1)
    # pcd_file_t2 = f"/home/gms/Isaac/IsaacLab2.2/DataCollection/Meta/partial/%04d/%03d.pcd" % (SUBDIR_IDX, t2)
    # npz_file = "/home/gms/Isaac/IsaacLab2.2/DataCollection/Meta/transform/%04d.npz" % SUBDIR_IDX
    
    print("=" * 50)
    print("开始读取PCD和NPZ文件")
    print("=" * 50)
    
    # --- 1. 读取文件 ---
    pcd_t1 = read_pcd(pcd_file_t1) 
    pcd_t2 = read_pcd(pcd_file_t2)
    pose_data = read_and_print_npz(npz_file)
    
    if pcd_t1 is None or pcd_t2 is None or pose_data is None:
        print("文件读取失败，请检查路径。")
        exit()

    # --- 2. 准备位姿数据 (Tensor) ---
    tensor_pos = torch.from_numpy(pose_data['pos']).float()
    tensor_quat = torch.from_numpy(pose_data['quat']).float()
    
    # 索引对齐 (注意：这里假设 npz 中的索引是 0-based，如果数据是 1-based 需要调整)
    pos_t1 = tensor_pos[t1]
    quat_t1 = tensor_quat[t1]
    pos_t2 = tensor_pos[t2]
    quat_t2 = tensor_quat[t2]

    # --- 3. 核心功能调用 (任务1的封装) ---
    print("\n" + "=" * 50)
    print("执行核心点云处理流水线...")
    print("=" * 50)
    
    # 初始化处理器 (参数与之前保持一致)
    processor = PointCloudProcessor(scale=3.2, bound=1.0)
    
    # 调用封装好的接口 (自动完成：PCD转Tensor -> 拼接处理 -> 分割内外点 -> 输出PCD)
    result = processor.process_frame(pcd_t1, pcd_t2, pos_t1, quat_t1, pos_t2, quat_t2)

    # --- 4. 可视化结果 ---
    # result 中已经包含了上好颜色的点云
    print(f"处理完成！")
    print(f"内部点数量: {len(result['inside_pcd'].points)}")
    print(f"外部点数量: {len(result['outside_pcd'].points)}")
    
    world_coord_frame = create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [
            world_coord_frame, 
            result['ref_pcd'],      # 绿色: 参考点云 (t2)
            result['inside_pcd'],   # 红色: 内部点 (t1)
            result['outside_pcd']   # 橙色: 外部点 (t1)
        ], 
        window_name="封装版: t2坐标系下的点云处理"
    )
