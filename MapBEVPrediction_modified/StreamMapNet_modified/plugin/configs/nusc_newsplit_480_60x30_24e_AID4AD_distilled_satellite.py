_base_ = [
    './nusc_newsplit_480_60x30_24e_AID4AD.py',
    './_base_/branch_experiment_runtime.py',
]

model = dict(
    use_AID4AD=True,
    AID4AD_only=False,
    branch_mode='fusion',
    return_branch_features=True,
    distill_cfg=dict(
        _delete_=True,
        teacher_ckpt='./branch_runs/sat_only_newsplit/model_last.pth',
        teacher_branch_mode='satellite_only',
        feature_name='satellite_branch_feature',
        loss_name='mse',
        loss_weight=1.0,
        use_mask=True,
        mask_thickness=3,
    ),
)

custom_hooks = [
    dict(
        type='BranchPostTrainHook',
        priority='LOW',
        eval_after_train=True,
        experiment_name='fusion_distilled_satellite_newsplit',
        branch_mode='fusion',
    ),
]
