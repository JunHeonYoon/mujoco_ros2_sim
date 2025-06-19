#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#define BOOST_MPL_LIMIT_LIST_SIZE 40

#include <boost/python.hpp>
#include <boost/mpl/vector.hpp>
#include <eigenpy/eigenpy.hpp>

#include <dlfcn.h>
#include <memory>
#include <stdexcept>

#include "mujoco_ros_sim/ControllerInterface.hpp"
#include "mujoco_ros_sim/ControllerRegistry.hpp"

namespace bp = boost::python;
using JointDict         = mujoco_ros_sim::JointDict;
using ControllerSP      = std::shared_ptr<ControllerInterface>;
using ControllerFactory = std::function<std::unique_ptr<ControllerInterface>(double, JointDict)>;
using SigT              = boost::mpl::vector<ControllerSP, double, JointDict>;


struct JointDict_from_python
{
  static void* convertible(PyObject* obj)
  { return PyMapping_Check(obj) ? obj : nullptr; }

  static void construct(PyObject* obj, bp::converter::rvalue_from_python_stage1_data* data)
  {
    void* storage =
        reinterpret_cast<bp::converter::rvalue_from_python_storage<JointDict>*>(data)
        ->storage.bytes;
    new (storage) JointDict;
    JointDict& dst = *static_cast<JointDict*>(storage);

    bp::dict py(bp::handle<>(bp::borrowed(obj)));

    auto list2vec = [](const bp::object& lst, auto& vec)
    {
      using T = typename std::decay_t<decltype(vec)>::value_type;
      bp::stl_input_iterator<T> it(lst), end;
      vec.assign(it, end);
    };
    list2vec(py["joint_names"],    dst.joint_names);
    list2vec(py["actuator_names"], dst.actuator_names);

    auto dict2umap = [](const bp::dict& d, auto& map)
    {
      bp::list keys = d.keys();
      for (long i = 0; i < bp::len(keys); ++i)
      {
        std::string k = bp::extract<std::string>(keys[i]);
        int         v = bp::extract<int>(d[keys[i]]);
        map[k] = v;
      }
    };
    dict2umap(bp::extract<bp::dict>(py["jname_to_jid"]), dst.jname_to_jid);
    dict2umap(bp::extract<bp::dict>(py["aname_to_aid"]), dst.aname_to_aid);

    data->convertible = storage;
  }
};

static VecMap pyDict_to_VecMap(const bp::dict &py)
{
  VecMap out;
  bp::list keys = py.keys();
  for (Py_ssize_t i = 0; i < bp::len(keys); ++i)
  {
    std::string key = bp::extract<std::string>( keys[i] );
    Eigen::VectorXd vec = bp::extract<Eigen::VectorXd>( py[keys[i]] );
    out.emplace(std::move(key), std::move(vec));
  }
  return out;
}

void updateState_wrapper(ControllerInterface &self,
                         const bp::dict &pos,
                         const bp::dict &vel,
                         const bp::dict &tau,
                         const bp::dict &sens,
                         double t)
{
  self.updateState( pyDict_to_VecMap(pos),
                    pyDict_to_VecMap(vel),
                    pyDict_to_VecMap(tau),
                    pyDict_to_VecMap(sens),
                    t );
}

static bp::dict map_to_pydict(const CtrlInputMap &m)
{
  bp::dict out;
  for (const auto &kv : m)
    out[kv.first] = kv.second;
  return out;
}

static bp::dict getCtrlInput_wrapper(const ControllerInterface &self)
{
  return map_to_pydict( self.getCtrlInput() );
}

static void load_plugin_library(const std::string& path)
{
  if (!dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL))
      throw std::runtime_error(dlerror());
}


static void export_new_factories()
{
  bp::object mod = bp::scope();
  auto& registry = ControllerRegistry::instance().map();

  for (auto& kv : registry)
  {
    const std::string pyname = kv.first + "_cpp";
    if (PyObject_HasAttrString(mod.ptr(), pyname.c_str()))
      continue;

    ControllerFactory fac = kv.second;
    auto wrapper = [fac](double dt, JointDict jd)->ControllerSP
    { return ControllerSP(fac(dt, std::move(jd)).release()); };

    bp::object pyfunc = bp::make_function(wrapper,
                                      bp::default_call_policies(),
                                      SigT());

    bp::object top = bp::import("bindings");
    PyObject_SetAttrString(top.ptr(),  pyname.c_str(), pyfunc.ptr());

    bp::object pkg = bp::import("mujoco_ros_sim.bindings");
    PyObject_SetAttrString(pkg.ptr(),  pyname.c_str(), pyfunc.ptr());
  }
}


BOOST_PYTHON_MODULE(bindings)
{
  eigenpy::enableEigenPy();

  bp::register_ptr_to_python< std::shared_ptr<ControllerInterface> >();
  bp::class_<ControllerInterface, std::shared_ptr<ControllerInterface>, boost::noncopyable >("ControllerInterface", bp::no_init)
    .def("starting",      &ControllerInterface::starting)
    .def("updateState",   &updateState_wrapper)
    .def("compute",       &ControllerInterface::compute)
    .def("getCtrlInput",  &getCtrlInput_wrapper);

  bp::converter::registry::push_back(&JointDict_from_python::convertible, &JointDict_from_python::construct, bp::type_id<JointDict>());

  export_new_factories();

  bp::def("load_plugin_library",   &load_plugin_library,  bp::args("path"));
  bp::def("export_new_factories",  &export_new_factories);
}
