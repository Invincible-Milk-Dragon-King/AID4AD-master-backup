_base_ = ['./nusc_newsplit_480_60x30_24e_AID4AD.py']

model = dict(
    use_AID4AD=True,
    AID4AD_only=False,
    branch_mode='camera_only',
    return_branch_features=True,
    distill_cfg=None,
)
