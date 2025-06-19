import importlib, sys, os, glob
from ament_index_python.packages import get_packages_with_prefixes

_cpp = importlib.import_module(__name__ + ".bindings")
sys.modules["bindings"] = _cpp                     # top-level alias (편의)

def _refresh():
    _cpp.export_new_factories()

    for mod in (_cpp, sys.modules.get("bindings")):
        if mod is None:
            continue
        for attr in dir(mod):
            if attr.startswith("create_"):
                globals()[attr[7:]] = getattr(mod, attr)


def _auto_scan():
    for _, prefix in get_packages_with_prefixes().items():
        plugdir = os.path.join(prefix, "lib", "mujoco_ros_sim_plugins")
        if not os.path.isdir(plugdir):
            continue
        for so in glob.glob(os.path.join(plugdir, "*.so")):
            _cpp.load_plugin_library(so)
    _refresh()

_auto_scan()                                      # import 시 한 번 수행

def load_plugin_library(path: str):
    _cpp.load_plugin_library(path)
    _refresh()

# re-export
from .controller_interface import ControllerInterface
from .mujoco_ros_sim       import MujocoSimNode
