_base_ = ['./nusc_newsplit_480_60x30_24e_AID4AD.py']

model = dict(
    use_AID4AD=True,
    AID4AD_only=False,
    branch_mode='fusion',
    return_branch_features=True,
    distill_cfg=dict(
        teacher_ckpt='checkpoints/camera_only/best.pth',
        teacher_branch_mode='camera_only',
        feature_name='camera_branch_feature',
        loss_name='mse',
        loss_weight=1.0,
    ),
)
