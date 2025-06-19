function(mrs_add_controller TARGET)
  if(ARGN STREQUAL "")
    message(FATAL_ERROR "mrs_add_controller(${TARGET}) needs at least one source file")
  endif()

  add_library(${TARGET} SHARED ${ARGN})
  set_target_properties(${TARGET} PROPERTIES PREFIX "" OUTPUT_NAME ${TARGET})

  find_package(Eigen3 REQUIRED)
  find_package(rclcpp  REQUIRED)
  find_package(eigenpy REQUIRED)

  target_link_libraries(${TARGET} PUBLIC
    mujoco_ros_sim::bindings  # registry + bridge
    rclcpp::rclcpp
    Eigen3::Eigen
    eigenpy::eigenpy)

  target_include_directories(${TARGET}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>)

  install(TARGETS ${TARGET}
    LIBRARY DESTINATION lib/${PROJECT_NAME}_plugins)
endfunction()
