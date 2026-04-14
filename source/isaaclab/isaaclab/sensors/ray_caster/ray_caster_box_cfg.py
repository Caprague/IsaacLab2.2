# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast box sensor."""


from dataclasses import MISSING
from typing import Literal

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_GREEN_MARKER_CFG, RAY_CASTER_BLUE_MARKER_CFG
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .patterns.patterns_cfg import PatternBaseCfg
from .ray_caster_box import RayCasterBox


@configclass
class RayCasterBoxCfg(SensorBaseCfg):
    """支持多层穿透的射线盒传感器配置.

    此传感器配置扩展了标准射线投射传感器以支持多层穿透检测，使能够在边界框内进行全面的3D地形采样。
    与仅返回第一个碰撞点的标准射线投射不同，此传感器通过障碍物递归投射射线以捕捉垂直结构、遮挡区域
    和复杂的地形特征。

    传感器使用盒网格模式，射线从三个正交面（例如左、上、前）开始并投向其对面。
    每条射线穿透多层障碍物直到到达对面或达到最大迭代限制。

    注意:
        传感器仅支持偏航对齐模式，这对于传感器跟随机器人位置和航向但保持垂直
        方向的地形扫描非常理想。

    示例:
        .. code-block:: python

            from isaaclab.sensors import RayCasterBoxCfg
            from isaaclab.sensors.patterns import BoxGridPatternCfg

            gt_scanner = RayCasterBoxCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base",
                offset=RayCasterBoxCfg.OffsetCfg(pos=(0.28945, 0.0, -0.04682)),
                pattern_cfg=patterns.BoxGridPatternCfg(
                    resolution=0.05,
                    size=(3.2, 3.2, 3.2),
                ),
                max_distance=3.2,
                max_iterations=5,
                epsilon=1e-2,
                box_vis=False,
                debug_vis=True,
                mesh_prim_paths=["/World/ground"],
            )
    """

    @configclass
    class OffsetCfg:
        """传感器框架相对于父框架的偏移姿态."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """相对于父框架的平移. 默认为 (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """相对于父框架的四元数旋转 (w, x, y, z). 默认为 (1.0, 0.0, 0.0, 0.0)."""

    class_type: type = RayCasterBox
    """指定传感器的类类型."""

    mesh_prim_paths: list[str] = MISSING
    """用于射线投射的网格基元路径列表.

    注意:
        目前仅支持单个静态网格. 我们正在努力支持多个静态网格和动态网格.
    """

    offset: OffsetCfg = OffsetCfg()
    """传感器框架相对于父框架的偏移姿态. 默认为单位矩阵."""

    pattern_cfg: PatternBaseCfg = MISSING
    """定义本地射线起始位置和方向的模式.

    对于3D盒采样，使用 :class:`BoxGridPatternCfg` 定义边界框尺寸和采样方向.
    """

    max_distance: float = 1e2
    """传感器投射射线的最大距离（米）. 默认为 1e6."""

    max_iterations: int = 5
    """每条射线的最大穿透迭代次数. 默认为 5.

    每次迭代从先前的碰撞点（带有小的epsilon偏移）投射射线以检测下一层障碍物.
    更高的值允许检测更多层但增加计算时间.

    注意:
        这可防止在有许多重叠障碍物的复杂场景中出现无限循环.
    """

    epsilon: float = 1e-3
    """碰撞后添加的小偏移量（米），以避免自相交. 默认为 1e-3.

    每次碰撞后，下次射线从碰撞点加上epsilon乘以射线方向投射，
    以避免立即与同表面再次碰撞.
    """

    drift_range: tuple[float, float] = (0.0, 0.0)
    """在世界坐标系中添加到射线起始位置的漂移范围（米）xyz. 默认为 (0.0, 0.0).

    对于浮动基机器人，这对于模拟机器人姿态估计中的漂移很有用.
    """

    ray_cast_drift_range: dict[str, tuple[float, float]] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    """在本地投影框架中添加到投影射线点的漂移范围（米）. 默认为
    包含每个x、y和z轴零漂移的字典.

    对于浮动基机器人，这对于模拟机器人姿态估计中的漂移很有用.
    """
    
    box_vis: bool = False
    """是否启用盒子可视化."""

    box_visualizer_cfg: VisualizationMarkersCfg = RAY_CASTER_BLUE_MARKER_CFG.replace(prim_path="/Visuals/RayCasterBox")
    """盒子可视化器配置."""

    hits_visualizer_cfg: VisualizationMarkersCfg = RAY_CASTER_GREEN_MARKER_CFG.replace(prim_path="/Visuals/RayCasterBoxHits")
    """射线击中点可视化器配置."""
    
    data_normalization: bool = False
    """仅当时 data_box_clip 有效时可用，控制是否对包围盒裁剪后的数据进行归一化处理，范围 (-0.5, +0.5)"""
    
    data_collection: bool = False
    """是否启用数据收集."""

    data_save_path: str | None = None
    """数据保存路径."""