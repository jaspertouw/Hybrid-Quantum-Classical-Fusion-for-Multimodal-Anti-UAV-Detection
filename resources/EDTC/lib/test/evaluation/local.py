from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # Required by uavtrack_eh.py — not defined in base EnvSettings class
    settings.prj_dir          = '/home/ssuttorp/mm-uav-threat-detection/resources/EDTC'
    settings.save_dir         = '/home/ssuttorp/mm-uav-threat-detection/resources/EDTC'

    # Where tracker output (.txt result files) are saved
    settings.results_path     = '/home/ssuttorp/mm-uav-threat-detection/resources/EDTC/test/tracking_results'

    # Where pretrained model weights are stored
    settings.network_path     = '/home/ssuttorp/pretrained_models'

    # AntiUAV dataset — test split (must be preprocessed first with scripts/preprocess_antiuav.py)
    settings.antiuav_path     = '/home/ssuttorp/Anti-UAV-RGBT/test'

    # Disable visualization for headless HPC evaluation
    settings.show_result      = False

    # Training datasets — not needed for evaluation
    settings.lasot_path       = ''
    settings.got10k_path      = ''
    settings.trackingnet_path = ''
    settings.uav_path         = ''
    settings.otb_path         = ''
    settings.nfs_path         = ''

    return settings
