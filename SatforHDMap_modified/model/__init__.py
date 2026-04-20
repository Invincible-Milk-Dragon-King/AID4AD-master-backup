from project_paths import ensure_repo_root_on_sys_path

ensure_repo_root_on_sys_path()

from .hdmapnet import HDMapNet
from .camera_baseline import HDMapNetCameraBaseline
from .ipm_net import IPMNet
from .lift_splat import LiftSplat
from .pointpillar import PointPillar

def get_model(method, data_conf, args, instance_seg=True, embedded_dim=16, direction_pred=True, angle_class=36):
    if method == 'lift_splat':
        model = LiftSplat(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_cam':
        if hasattr(args, 'branch_mode') and args.branch_mode == 'camera_only':
            model = HDMapNetCameraBaseline(data_conf, args, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class)
        else:
            model = HDMapNet(data_conf, args, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False)
    elif method == 'HDMapNet_lidar':
        model = PointPillar(data_conf, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_fusion':
        model = HDMapNet(data_conf, args, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True)
    else:
        raise NotImplementedError

    return model
