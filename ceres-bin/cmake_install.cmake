# Install script for directory: /home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ceres" TYPE FILE FILES
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/autodiff_cost_function.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/autodiff_local_parameterization.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/c_api.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/ceres.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/conditioned_cost_function.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/cost_function.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/cost_function_to_functor.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/covariance.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/crs_matrix.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/cubic_interpolation.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/dynamic_autodiff_cost_function.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/dynamic_cost_function_to_functor.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/dynamic_numeric_diff_cost_function.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/fpclassify.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/gradient_checker.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/gradient_problem.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/gradient_problem_solver.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/iteration_callback.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/jet.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/local_parameterization.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/loss_function.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/normal_prior.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/numeric_diff_cost_function.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/numeric_diff_options.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/ordered_groups.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/problem.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/rotation.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/sized_cost_function.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/solver.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/types.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/version.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ceres/internal" TYPE FILE FILES
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/internal/autodiff.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/internal/disable_warnings.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/internal/eigen.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/internal/fixed_array.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/internal/macros.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/internal/manual_constructor.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/internal/numeric_diff.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/internal/port.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/internal/reenable_warnings.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/internal/scoped_ptr.h"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/include/ceres/internal/variadic_evaluate.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ceres/internal" TYPE FILE FILES "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-bin/config/ceres/internal/config.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Ceres/CeresTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Ceres/CeresTargets.cmake"
         "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-bin/CMakeFiles/Export/share/Ceres/CeresTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Ceres/CeresTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Ceres/CeresTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Ceres" TYPE FILE FILES "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-bin/CMakeFiles/Export/share/Ceres/CeresTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Ceres" TYPE FILE FILES "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-bin/CMakeFiles/Export/share/Ceres/CeresTargets-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Ceres" TYPE FILE RENAME "CeresConfig.cmake" FILES "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-bin/CeresConfig-install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Ceres" TYPE FILE FILES
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-bin/CeresConfigVersion.cmake"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/cmake/FindEigen.cmake"
    "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-solver-1.11.0/cmake/FindGlog.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-bin/internal/ceres/cmake_install.cmake")
  include("/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-bin/examples/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/tusharc31/Desktop/CV/SENIOR/3d-reconstruction-from-accidental-motion/ceres-bin/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
