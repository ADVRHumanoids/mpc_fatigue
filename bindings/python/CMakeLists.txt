cmake_minimum_required(VERSION 3.5)

set(PYBIND11_PYTHON_VERSION 2.7 CACHE STRING "")
find_package(pybind11 QUIET)

if(${pybind11_FOUND})

    message(STATUS "Compiling python bindings")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})

    find_package(urdfdom REQUIRED)
    find_package(Boost REQUIRED COMPONENTS system)
    find_package(casadi REQUIRED)

    message(STATUS " ---- ${CASADI_LIBRARIES}")

    pybind11_add_module(pynocchio_casadi pynocchio_casadi.cpp)
    target_link_libraries(pynocchio_casadi PUBLIC
        ${CASADI_LIBRARIES}
        ${urdfdom_LIBRARIES}
        ${Boost_LIBRARIES})

    install(TARGETS pynocchio_casadi
            DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION})

    file(WRITE ${CMAKE_BINARY_DIR}/__init__.py "")
    install(FILES ${CMAKE_BINARY_DIR}/__init__.py DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION})
            
endif()
