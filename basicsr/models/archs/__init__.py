import importlib
from os import path as osp

from basicsr.utils import scandir

# automatically scan and import arch modules
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith("_arch.py")]
# import all the arch modules
_arch_modules = [importlib.import_module(f"basicsr.models.archs.{file_name}") for file_name in arch_filenames]


def dynamic_instantiation(modules, cls_type, opt):
    """Dynamically instantiate class.

    Args:
        modules (list[importlib modules]): List of modules from importlib
            files.
        cls_type (str): Class type.
        opt (dict): Class initialization kwargs.

    Returns:
        class: Instantiated class.
    """

    for module in modules:  # 在当前的 module 对象（也就是当前循环到的那个架构模块）里面，查找一个名字叫做 cls_type (比如 'Restormer') 的东西
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f"{cls_type} is not found.")
    return cls_(**opt)


def define_network(opt):  # 使用opt中的参数，动态实例化cla_type类型的network（即restormeer）
    network_type = opt.pop("type")
    net = dynamic_instantiation(_arch_modules, network_type, opt)
    return net
