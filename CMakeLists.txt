cmake_minimum_required(VERSION 3.17)
project(mujoco_ros_sim)
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Boost   REQUIRED COMPONENTS python)
find_package(eigenpy REQUIRED)
find_package(rclcpp  REQUIRED)
find_package(rclpy  REQUIRED)

add_library(bindings SHARED 
            src/bindings.cpp
            src/ControllerRegistry.cpp)

set_target_properties(bindings PROPERTIES OUTPUT_NAME bindings)

target_link_options(bindings PUBLIC "-Wl,-export-dynamic")

target_include_directories(bindings PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>
                           ${Boost_INCLUDE_DIRS} 
                           ${eigenpy_INCLUDE_DIRS})

target_link_libraries(bindings PUBLIC
                      ${Boost_LIBRARIES} 
                      eigenpy::eigenpy 
                      rclcpp::rclcpp)

add_library(mujoco_ros_sim::bindings ALIAS bindings)

install(DIRECTORY   include/ 
        DESTINATION include)

ament_python_install_package(mujoco_ros_sim)

install(PROGRAMS    ${CMAKE_CURRENT_SOURCE_DIR}/mujoco_ros_sim/mujoco_ros_sim.py
        DESTINATION lib/${PROJECT_NAME}
        RENAME      mujoco_ros_sim)

install(TARGETS bindings
        EXPORT  mujoco_ros_simTargets
        LIBRARY DESTINATION lib)

add_custom_command(TARGET  bindings POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                           $<TARGET_FILE:bindings>
                           ${CMAKE_CURRENT_BINARY_DIR}/bindings.so
                   COMMENT "Staging unprefixed Python binding")

install(FILES       ${CMAKE_CURRENT_BINARY_DIR}/bindings.so
        DESTINATION ${PYTHON_INSTALL_DIR}/mujoco_ros_sim)

install(DIRECTORY   launch           
        DESTINATION share/${PROJECT_NAME})

install(DIRECTORY   mujoco_menagerie 
        DESTINATION share/${PROJECT_NAME})

install(FILES       resource/${PROJECT_NAME}
        DESTINATION share/ament_index/resource_index/packages)

install(FILES       package.xml          
        DESTINATION share/${PROJECT_NAME})

install(FILES cmake/MRSController.cmake
        DESTINATION share/${PROJECT_NAME}/cmake)

install(EXPORT      mujoco_ros_simTargets
        FILE        mujoco_ros_simTargets.cmake
        NAMESPACE   mujoco_ros_sim::
        DESTINATION share/${PROJECT_NAME}/cmake)

ament_export_include_directories(include)
ament_export_libraries(bindings)
ament_export_targets(mujoco_ros_simTargets)
ament_package()