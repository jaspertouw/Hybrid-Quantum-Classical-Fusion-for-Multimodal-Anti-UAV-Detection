from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.uavtrack_dual.config import cfg, update_config_from_file


def parameters(yaml_name: str, model=None, search_area_scale=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    # Load dual-stream experiment config
    yaml_file = os.path.join(prj_dir, 'experiments/uavtrack_dual/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg

    # Template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    if search_area_scale is not None:
        params.search_factor = search_area_scale
    else:
        params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path (the same IR-trained transformer backbone)
    if model is None:
        raise NotImplementedError("Please set --params__model to the UAVTrackEH checkpoint path.")
    else:
        params.checkpoint = model

    params.save_all_boxes = False

    return params
