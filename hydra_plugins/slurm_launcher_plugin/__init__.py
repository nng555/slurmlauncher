from omegaconf import OmegaConf

OmegaConf.register_new_resolver('join', lambda x: '_'.join([str(v) for v in x]) if x is not None else 'local')
OmegaConf.register_new_resolver('eval', eval)
