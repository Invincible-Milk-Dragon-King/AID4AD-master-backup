BRANCH_MODE_CAMERA_ONLY = 'camera_only'
BRANCH_MODE_FUSION = 'fusion'
BRANCH_MODE_DROP_SATELLITE = 'drop_satellite'
BRANCH_MODE_SATELLITE_ONLY = 'satellite_only'


def resolve_branch_mode(use_AID4AD, AID4AD_only, branch_mode=None):
    if branch_mode is not None:
        return branch_mode
    if AID4AD_only:
        return BRANCH_MODE_SATELLITE_ONLY
    if use_AID4AD:
        return BRANCH_MODE_FUSION
    return BRANCH_MODE_CAMERA_ONLY


def uses_aerial_encoder(branch_mode):
    return branch_mode in (BRANCH_MODE_FUSION, BRANCH_MODE_SATELLITE_ONLY)


def uses_aerial_fuser(branch_mode):
    return branch_mode == BRANCH_MODE_FUSION
