"""配置射线投射传感器。"""

from dataclasses import MISSING
from typing import Literal

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_LIDAR_MARKER_CFG
from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .patterns.patterns_cfg import PatternBaseCfg
from .ray_caster_lidar import RayCasterLidar


@configclass
class RayCasterLidarCfg(SensorBaseCfg):
    """射线投射传感器的配置类。"""

    @configclass
    class OffsetCfg:
        """传感器坐标系相对于其父坐标系的偏移位姿。"""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """相对于父坐标系的平移量。默认为 (0.0, 0.0, 0.0)。"""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """相对于父坐标系的四元数旋转 (w, x, y, z)。默认为 (1.0, 0.0, 0.0, 0.0)。"""

    class_type: type = RayCasterLidar
    """对应的传感器类类型。"""

    mesh_prim_paths: list[str] = MISSING
    """用于进行射线投射的网格图元路径列表。

    注意：
        目前仅支持单个静态网格。我们正在开发对多个静态网格和动态网格的支持。
    """

    offset: OffsetCfg = OffsetCfg()
    """传感器坐标系相对于其父坐标系的偏移位姿。默认为单位矩阵（无偏移）。"""

    attach_yaw_only: bool | None = None
    """射线的起始位置和方向是否仅跟随偏航（yaw）角旋转。
    默认为 None，此时不会触发弃用警告。

    这在进行高度图（height map）的射线投射时非常有用，因为通常只需要偏航角信息。

    .. deprecated:: 2.1.1

        此属性已被弃用，未来将被移除。请改用 :attr:`ray_alignment`。

        若想获得与此参数设为 ``True`` 或 ``False`` 相同的行为，请将 :attr:`ray_alignment`
        分别设为 ``"yaw"`` 或 ``"base"``。
    """

    ray_alignment: Literal["base", "yaw", "world"] = "base"
    """指定射线投影到地面时所参考的坐标系。默认为 "base"。

    可选模式如下：

    * ``base``：射线的起始位置和方向跟随根物体的完整位置和姿态。
    * ``yaw``：射线的起始位置和方向跟随根物体的位置以及姿态中的偏航分量。
      这在进行高度图射线投射时非常有用。
    * ``world``：射线的起始位置和方向始终保持固定。这在与机器人上的建图包结合使用，
      或在全局坐标系中查询射线投射结果时非常有用。
    """

    pattern_cfg: PatternBaseCfg = MISSING
    """定义局部射线起始位置和方向的模式配置对象。"""

    max_distance: float = 1e6
    """从传感器出发进行射线投射的最大距离（米）。默认为 1e6。"""

    drift_range: tuple[float, float] = (0.0, 0.0)
    """在世界坐标系下，添加到射线起始位置（xyz）的漂移范围（米）。默认为 (0.0, 0.0)。

    对于浮动基座机器人，这可用于模拟机器人位姿估计中的漂移。
    """

    ray_cast_drift_range: dict[str, tuple[float, float]] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    """在局部投影坐标系下，添加到投影射线点的漂移范围（米）。默认为
    一个在 x, y, z 轴上漂移均为零的字典。

    对于浮动基座机器人，这可用于模拟机器人位姿估计中的漂移。
    """

    visualizer_cfg: VisualizationMarkersCfg = RAY_CASTER_LIDAR_MARKER_CFG.replace(prim_path="/Visuals/RayCaster")
    """可视化标记的配置对象。默认为 RAY_CASTER_MARKER_CFG。

    注意：
        仅在启用调试可视化时使用此属性。
    """

    dynamic_pattern: bool = False
    """是否在每次传感器更新时动态更新射线模式。

    启用后，模式生成函数会在每次更新时被调用以生成新的扫描模式。
    这对于需要通过随时间循环不同扫描模式来模拟连续扫描的传感器（如 Livox Mid-360）非常有用。

    默认为 False，这意味着射线模式仅在初始化时计算一次，并在整个仿真过程中保持静态。
    这是大多数射线投射传感器的默认行为。
    """

    data_collection: bool = False
    """是否启用数据收集。"""

    data_save_path: str | None = None
    """数据保存路径。"""
