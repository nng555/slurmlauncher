from omegaconf import OmegaConf
import sys

OmegaConf.register_new_resolver('join', lambda x : ','.join([str(v) for v in x]) if x is not None else 'local')
OmegaConf.register_new_resolver('eval', eval)

def ternary(x):
    cond, tout, fout = x.split('?')
    if eval(cond):
        return tout
    else:
        return fout

OmegaConf.register_new_resolver('cond', ternary)
