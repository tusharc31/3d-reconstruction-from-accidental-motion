# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2015 Google Inc. All rights reserved.
# http://ceres-solver.org/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Google Inc. nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Authors: keir@google.com (Keir Mierle)
#          alexs.mac@gmail.com (Alex Stewart)

cmake_minimum_required(VERSION 2.8.0)
cmake_policy(VERSION 2.8)

cmake_policy(SET CMP0003 NEW)
if (POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
endif()

project(Ceres C CXX)

# Make CMake aware of the cmake folder for local FindXXX scripts,
# append rather than set in case the user has passed their own
# additional paths via -D.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(UpdateCacheVariable)

# Set up the git hook to make Gerrit Change-Id: lines in commit messages.
include(AddGerritCommitHook)
add_gerrit_commit_hook()

# On OS X, add the Homebrew prefix to the set of prefixes searched by
# CMake in find_path & find_library.  This should ensure that we can
# still build Ceres even if Homebrew is installed in a non-standard
# location (not /usr/local).
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  find_program(HOMEBREW_EXECUTABLE brew)
  mark_as_advanced(FORCE HOMEBREW_EXECUTABLE)
  if (HOMEBREW_EXECUTABLE)
    # Detected a Homebrew install, query for its install prefix.
    execute_process(COMMAND ${HOMEBREW_EXECUTABLE} --prefix
      OUTPUT_VARIABLE HOMEBREW_INSTALL_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Detected Homebrew with install prefix: "
      "${HOMEBREW_INSTALL_PREFIX}, adding to CMake search paths.")
    list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_INSTALL_PREFIX}")
  endif()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# Set postfixes for generated libraries based on buildtype.
set(CMAKE_RELEASE_POSTFIX "")
set(CMAKE_DEBUG_POSTFIX "-debug")

# Important: Always bump the second number (e.g. 1.3.x to 1.4.0) for any
# release that changes the ABI. The ABI changes for almost any modification to
# include/ceres (e.g. the public API). If you are unsure about whether
# something is an ABI change, please ask on the list.
set(CERES_VERSION_MAJOR 1)
set(CERES_VERSION_MINOR 11)
set(CERES_VERSION_PATCH 0)
set(CERES_VERSION
    ${CERES_VERSION_MAJOR}.${CERES_VERSION_MINOR}.${CERES_VERSION_PATCH})

enable_testing()

option(MINIGLOG "Use a stripped down version of glog." OFF)
option(GFLAGS "Enable Google Flags." ON)
option(SUITESPARSE "Enable SuiteSparse." ON)
option(CXSPARSE "Enable CXSparse." ON)
option(LAPACK "Enable use of LAPACK." ON)
# Template specializations for the Schur complement based solvers. If
# compile time, binary size or compiler performance is an issue, you
# may consider disabling this.
option(SCHUR_SPECIALIZATIONS "Enable fixed-size schur specializations." ON)
option(CUSTOM_BLAS
       "Use handcoded BLAS routines (usually faster) instead of Eigen."
       ON)
# Multithreading using OpenMP
option(OPENMP "Enable threaded solving in Ceres (requires OpenMP)" ON)
# Enable the use of Eigen as a sparse linear algebra library for
# solving the nonlinear least squares problems. Enabling this
# option will result in an LGPL licensed version of Ceres Solver
# as the Simplicial Cholesky factorization in Eigen is licensed under the LGPL.
# This does not affect the covariance estimation algorithm, as it
# depends on the sparse QR factorization algorithm, which is licensed
# under the MPL.
OPTION(EIGENSPARSE
  "Enable Eigen as a sparse linear algebra library, WARNING: results in an LGPL licensed Ceres." OFF)
if (NOT WIN32)
  # Ceres does not use C++11 internally, however it does use shared_ptr
  # (required) and unordered_map (if available), both of which were present in
  # previous iterations of what became C++11.  GCC & Clang can have both TR1 &
  # C++11 versions of both shared_ptr & unordered_map and by default on Linux,
  # we will detect the TR1 versions if they exist, as they do NOT require
  # -std=c++11 to be passed when compiling Ceres, and any client code that uses
  # Ceres.  This will result in conflicts if the client code uses C++11.
  # Enabling this option forces the use of the C++11 versions (& -std=c++11) if
  # available.
  #
  # This option is not available on Windows, as there, any new (C++11 etc)
  # features available are on by default and there is no analogue to -std=c++11.
  option(CXX11 "Enable use of C++11 headers if available (requires client code use C++11)." OFF)
endif(NOT WIN32)
option(EXPORT_BUILD_DIR
  "Export build directory using CMake (enables external use without install)." OFF)
option(BUILD_TESTING "Enable tests" OFF)
option(BUILD_DOCUMENTATION "Build User's Guide (html)" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_SHARED_LIBS "Build Ceres as a shared library." OFF)
if (MSVC)
  option(MSVC_USE_STATIC_CRT
    "MS Visual Studio: Use static C-Run Time Library in place of shared." OFF)

  if (BUILD_TESTING AND BUILD_SHARED_LIBS)
    message(
      "-- Disabling tests. The flags BUILD_TESTING and BUILD_SHARED_LIBS"
      " are incompatible with MSVC.")
    update_cache_variable(BUILD_TESTING OFF)
  endif (BUILD_TESTING AND BUILD_SHARED_LIBS)
endif (MSVC)

# IOS is defined iff using the iOS.cmake CMake toolchain to build a static
# library for iOS.
if (IOS)
  message(STATUS "Building Ceres for iOS platform: ${IOS_PLATFORM}")

  # Ceres requires at least iOS 7.0+.
  if (IOS_SDK_VERSION VERSION_LESS 7.0)
    message(FATAL_ERROR "Unsupported iOS SDK version: ${IOS_SDK_VERSION}, Ceres "
      "required at least iOS version 7.0")
  endif()

  update_cache_variable(MINIGLOG ON)
  message(STATUS "Building for iOS: Forcing use of miniglog instead of glog.")

  update_cache_variable(SUITESPARSE OFF)
  update_cache_variable(CXSPARSE OFF)
  update_cache_variable(GFLAGS OFF)
  update_cache_variable(OPENMP OFF)
  # Apple claims that the BLAS call dsyrk_ is a private API, and will not allow
  # you to submit to the Apple Store if the symbol is present.
  update_cache_variable(LAPACK OFF)
  message(STATUS "Building for iOS: SuiteSparse, CXSparse, LAPACK, gflags, "
    "and OpenMP are not available.")

  update_cache_variable(BUILD_EXAMPLES OFF)
  message(STATUS "Building for iOS: Will not build examples.")
endif (IOS)

unset(CERES_COMPILE_OPTIONS)

# Eigen.
find_package(Eigen REQUIRED)
if (EIGEN_FOUND)
  if (EIGEN_VERSION VERSION_LESS 3.1.0)
    message(FATAL_ERROR "-- Ceres requires Eigen version >= 3.1.0 in order "
      "that Eigen/SparseCore be available, detected version of Eigen is: "
      "${EIGEN_VERSION}")
  endif (EIGEN_VERSION VERSION_LESS 3.1.0)

  message("-- Found Eigen version ${EIGEN_VERSION}: ${EIGEN_INCLUDE_DIRS}")
  # Ensure that only MPL2 licensed code is part of the default build.
  message("")
  message("   ===============================================================")
  if (EIGENSPARSE)
    list(APPEND CERES_COMPILE_OPTIONS CERES_USE_EIGEN_SPARSE)
    message("   Enabling the use of Eigen as a sparse linear algebra library ")
    message("   for solving the nonlinear least squares problems. Enabling ")
    message("   this option results in an LGPL licensed version of ")
    message("   Ceres Solver as the Simplicial Cholesky factorization in Eigen")
    message("   is licensed under the LGPL. ")

    if (EIGEN_VERSION VERSION_LESS 3.2.2)
      message("   WARNING:")
      message("")
      message("   Your version of Eigen is older than version 3.2.2.")
      message("   The performance of SPARSE_NORMAL_CHOLESKY and SPARSE_SCHUR")
      message("   linear solvers will suffer. ")
    endif (EIGEN_VERSION VERSION_LESS 3.2.2)

  else (EIGENSPARSE)
    message("   Disabling the use of Eigen as a sparse linear algebra library.")
    message("   This does not affect the covariance estimation algorithm ")
    message("   which can still use the EIGEN_SPARSE_QR algorithm.")
    add_definitions(-DEIGEN_MPL2_ONLY)
  endif (EIGENSPARSE)
    message("   ===============================================================")
    message("")
endif (EIGEN_FOUND)

# LAPACK (& BLAS).
if (LAPACK)
  find_package(LAPACK QUIET)
  if (LAPACK_FOUND)
    message("-- Found LAPACK library: ${LAPACK_LIBRARIES}")
  else (LAPACK_FOUND)
    message("-- Did not find LAPACK library, disabling LAPACK support.")
  endif (LAPACK_FOUND)

  find_package(BLAS QUIET)
  if (BLAS_FOUND)
    message("-- Found BLAS library: ${BLAS_LIBRARIES}")
  else (BLAS_FOUND)
    message("-- Did not find BLAS library, disabling LAPACK support.")
  endif (BLAS_FOUND)

  if (NOT (LAPACK_FOUND AND BLAS_FOUND))
    update_cache_variable(LAPACK OFF)
    list(APPEND CERES_COMPILE_OPTIONS CERES_NO_LAPACK)
  endif (NOT (LAPACK_FOUND AND BLAS_FOUND))
else (LAPACK)
  message("-- Building without LAPACK.")
  list(APPEND CERES_COMPILE_OPTIONS CERES_NO_LAPACK)
endif (LAPACK)

# SuiteSparse.
if (SUITESPARSE AND NOT LAPACK)
  # If user has disabled LAPACK, but left SUITESPARSE ON, turn it OFF,
  # LAPACK controls whether Ceres will be linked, directly or indirectly
  # via SuiteSparse to LAPACK.
  message("-- Disabling SuiteSparse as use of LAPACK has been disabled, "
    "turn ON LAPACK to enable (optional) building with SuiteSparse.")
  update_cache_variable(SUITESPARSE OFF)
endif (SUITESPARSE AND NOT LAPACK)
if (SUITESPARSE)
  # By default, if SuiteSparse and all dependencies are found, Ceres is
  # built with SuiteSparse support.

  # Check for SuiteSparse and dependencies.
  find_package(SuiteSparse)
  if (SUITESPARSE_FOUND)
    # On Ubuntu the system install of SuiteSparse (v3.4.0) up to at least
    # Ubuntu 13.10 cannot be used to link shared libraries.
    if (BUILD_SHARED_LIBS AND
        SUITESPARSE_IS_BROKEN_SHARED_LINKING_UBUNTU_SYSTEM_VERSION)
      message(FATAL_ERROR "You are attempting to build Ceres as a shared "
        "library on Ubuntu using a system package install of SuiteSparse "
        "3.4.0. This package is broken and does not support the "
        "construction of shared libraries (you can still build Ceres as "
        "a static library).  If you wish to build a shared version of Ceres "
        "you should uninstall the system install of SuiteSparse "
        "(libsuitesparse-dev) and perform a source install of SuiteSparse "
        "(we recommend that you use the latest version), "
        "see http://ceres-solver.org/building.html for more information.")
    endif (BUILD_SHARED_LIBS AND
      SUITESPARSE_IS_BROKEN_SHARED_LINKING_UBUNTU_SYSTEM_VERSION)

    # By default, if all of SuiteSparse's dependencies are found, Ceres is
    # built with SuiteSparse support.
    message("-- Found SuiteSparse ${SUITESPARSE_VERSION}, "
            "building with SuiteSparse.")
  else (SUITESPARSE_FOUND)
    # Disable use of SuiteSparse if it cannot be found and continue.
    message("-- Did not find all SuiteSparse dependencies, disabling "
      "SuiteSparse support.")
    update_cache_variable(SUITESPARSE OFF)
    list(APPEND CERES_COMPILE_OPTIONS CERES_NO_SUITESPARSE)
  endif (SUITESPARSE_FOUND)
else (SUITESPARSE)
  message("-- Building without SuiteSparse.")
  list(APPEND CERES_COMPILE_OPTIONS CERES_NO_SUITESPARSE)
endif (SUITESPARSE)

# CXSparse.
if (CXSPARSE)
  # Don't search with REQUIRED as we can continue without CXSparse.
  find_package(CXSparse)
  if (CXSPARSE_FOUND)
    # By default, if CXSparse and all dependencies are found, Ceres is
    # built with CXSparse support.
    message("-- Found CXSparse version: ${CXSPARSE_VERSION}, "
      "building with CXSparse.")
  else (CXSPARSE_FOUND)
    # Disable use of CXSparse if it cannot be found and continue.
    message("-- Did not find CXSparse, Building without CXSparse.")
    update_cache_variable(CXSPARSE OFF)
    list(APPEND CERES_COMPILE_OPTIONS CERES_NO_CXSPARSE)
  endif (CXSPARSE_FOUND)
else (CXSPARSE)
  message("-- Building without CXSparse.")
  list(APPEND CERES_COMPILE_OPTIONS CERES_NO_CXSPARSE)
  # Mark as advanced (remove from default GUI view) the CXSparse search
  # variables in case user enabled CXSPARSE, FindCXSparse did not find it, so
  # made search variables visible in GUI for user to set, but then user disables
  # CXSPARSE instead of setting them.
  mark_as_advanced(FORCE CXSPARSE_INCLUDE_DIR
                         CXSPARSE_LIBRARY)
endif (CXSPARSE)

# Ensure that the user understands they have disabled all sparse libraries.
if (NOT SUITESPARSE AND NOT CXSPARSE AND NOT EIGENSPARSE)
  message("   ===============================================================")
  message("   Compiling without any sparse library: SuiteSparse, CXSparse ")
  message("   & Eigen (Sparse) are all disabled or unavailable.  No sparse ")
  message("   linear solvers (SPARSE_NORMAL_CHOLESKY & SPARSE_SCHUR)")
  message("   will be available when Ceres is used.")
  message("   ===============================================================")
endif(NOT SUITESPARSE AND NOT CXSPARSE AND NOT EIGENSPARSE)

# GFlags.
if (GFLAGS)
  # Don't search with REQUIRED as we can continue without gflags.
  find_package(Gflags)
  if (GFLAGS_FOUND)
    message("-- Found Google Flags header in: ${GFLAGS_INCLUDE_DIRS}, "
      "in namespace: ${GFLAGS_NAMESPACE}")
  else (GFLAGS_FOUND)
    message("-- Did not find Google Flags (gflags), Building without gflags "
      "- no tests or tools will be built!")
    update_cache_variable(GFLAGS OFF)
  endif (GFLAGS_FOUND)
else (GFLAGS)
  message("-- Google Flags disabled; no tests or tools will be built!")
  # Mark as advanced (remove from default GUI view) the gflags search
  # variables in case user enabled GFLAGS, FindGflags did not find it, so
  # made search variables visible in GUI for user to set, but then user disables
  # GFLAGS instead of setting them.
  mark_as_advanced(FORCE GFLAGS_INCLUDE_DIR
                         GFLAGS_LIBRARY
                         GFLAGS_NAMESPACE)
endif (GFLAGS)

# MiniGLog.
if (MINIGLOG)
  message("-- Compiling minimal glog substitute into Ceres.")
  set(GLOG_INCLUDE_DIRS internal/ceres/miniglog)
  message("-- Using minimal glog substitute (include): ${GLOG_INCLUDE_DIRS}")

  # Mark as advanced (remove from default GUI view) the glog search
  # variables in case user disables MINIGLOG, FindGlog did not find it, so
  # made search variables visible in GUI for user to set, but then user enables
  # MINIGLOG instead of setting them.
  mark_as_advanced(FORCE GLOG_INCLUDE_DIR
                         GLOG_LIBRARY)
else (MINIGLOG)
  # Don't search with REQUIRED so that configuration continues if not found and
  # we can output an error messages explaining MINIGLOG option.
  find_package(Glog)
  if (GLOG_FOUND)
    message("-- Found Google Log header in: ${GLOG_INCLUDE_DIRS}")
  else (GLOG_FOUND)
    message(FATAL_ERROR "Can't find Google Log. Please set GLOG_INCLUDE_DIR & "
      "GLOG_LIBRARY or enable MINIGLOG option to use minimal glog "
      "implementation.")
  endif (GLOG_FOUND)
endif (MINIGLOG)

if (NOT SCHUR_SPECIALIZATIONS)
  list(APPEND CERES_COMPILE_OPTIONS CERES_RESTRICT_SCHUR_SPECIALIZATION)
  message("-- Disabling Schur specializations (faster compiles)")
endif (NOT SCHUR_SPECIALIZATIONS)

if (NOT CUSTOM_BLAS)
  list(APPEND CERES_COMPILE_OPTIONS CERES_NO_CUSTOM_BLAS)
  message("-- Disabling custom blas")
endif (NOT CUSTOM_BLAS)

if (OPENMP)
  # Clang does not (yet) support OpenMP.
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    update_cache_variable(OPENMP OFF)
    message("-- Compiler is Clang, disabling OpenMP.")
    list(APPEND CERES_COMPILE_OPTIONS CERES_NO_THREADS)
  else (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Find quietly s/t as we can continue without OpenMP if it is not found.
    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
      message("-- Building with OpenMP.")
      list(APPEND CERES_COMPILE_OPTIONS CERES_USE_OPENMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      if (UNIX)
        # At least on Linux, we need pthreads to be enabled for mutex to
        # compile.  This may not work on Windows or Android.
        find_package(Threads REQUIRED)
        list(APPEND CERES_COMPILE_OPTIONS CERES_HAVE_PTHREAD)
        list(APPEND CERES_COMPILE_OPTIONS CERES_HAVE_RWLOCK)
      endif (UNIX)
    else (OPENMP_FOUND)
      message("-- Failed to find OpenMP, disabling.")
      update_cache_variable(OPENMP OFF)
      list(APPEND CERES_COMPILE_OPTIONS CERES_NO_THREADS)
    endif (OPENMP_FOUND)
  endif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
else (OPENMP)
  message("-- Building without OpenMP (disabling multithreading).")
  list(APPEND CERES_COMPILE_OPTIONS CERES_NO_THREADS)
endif (OPENMP)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_HAS_CXX11_FLAG)
if (CXX11 AND COMPILER_HAS_CXX11_FLAG)
  # Update CMAKE_REQUIRED_FLAGS used by CheckCXXSourceCompiles to include
  # -std=c++11 s/t we will detect the C++11 versions of unordered_map &
  # shared_ptr if they exist.
  set(CMAKE_REQUIRED_FLAGS -std=c++11)
endif (CXX11 AND COMPILER_HAS_CXX11_FLAG)

# Set the Ceres compile definitions for the unordered_map configuration.
include(FindUnorderedMap)
find_unordered_map()
if (UNORDERED_MAP_FOUND)
  if (HAVE_UNORDERED_MAP_IN_STD_NAMESPACE)
    list(APPEND CERES_COMPILE_OPTIONS CERES_STD_UNORDERED_MAP)
  endif(HAVE_UNORDERED_MAP_IN_STD_NAMESPACE)
  if (HAVE_UNORDERED_MAP_IN_TR1_NAMESPACE)
    list(APPEND CERES_COMPILE_OPTIONS CERES_STD_UNORDERED_MAP_IN_TR1_NAMESPACE)
  endif(HAVE_UNORDERED_MAP_IN_TR1_NAMESPACE)
  if (HAVE_TR1_UNORDERED_MAP_IN_TR1_NAMESPACE)
    list(APPEND CERES_COMPILE_OPTIONS CERES_TR1_UNORDERED_MAP)
  endif(HAVE_TR1_UNORDERED_MAP_IN_TR1_NAMESPACE)
else (UNORDERED_MAP_FOUND)
  message("-- Replacing unordered_map/set with map/set (warning: slower!), "
    "try enabling CXX11 option if you expect C++11 to be available.")
  list(APPEND CERES_COMPILE_OPTIONS CERES_NO_UNORDERED_MAP)
endif()

# Set the Ceres compile definitions for the shared_ptr configuration.
include(FindSharedPtr)
find_shared_ptr()
if (SHARED_PTR_FOUND)
  if (SHARED_PTR_TR1_MEMORY_HEADER)
    list(APPEND CERES_COMPILE_OPTIONS CERES_TR1_MEMORY_HEADER)
  endif (SHARED_PTR_TR1_MEMORY_HEADER)
  if (SHARED_PTR_TR1_NAMESPACE)
    list(APPEND CERES_COMPILE_OPTIONS CERES_TR1_SHARED_PTR)
  endif (SHARED_PTR_TR1_NAMESPACE)
else (SHARED_PTR_FOUND)
  message(FATAL_ERROR "Unable to find shared_ptr, try enabling CXX11 option "
    "if you expect C++11 to be available.")
endif (SHARED_PTR_FOUND)

# To ensure that CXX11 accurately reflects whether we are using C++11,
# check if it is required given where the potentially C++11 features Ceres
# uses were found, and disable it if C++11 is not being used.
if (CXX11)
  if (NOT HAVE_SHARED_PTR_IN_STD_NAMESPACE AND
      NOT HAVE_UNORDERED_MAP_IN_STD_NAMESPACE)
    message("-- Failed to find C++11 components in C++11 locations & "
      "namespaces, disabling CXX11.")
    update_cache_variable(CXX11 OFF)
  else()
    message("   ==============================================================")
    message("   Compiling Ceres using C++11.  This will result in a version ")
    message("   of Ceres that will require the use of C++11 in client code.")
    message("   ==============================================================")
    list(APPEND CERES_COMPILE_OPTIONS CERES_USE_CXX11)
    if (COMPILER_HAS_CXX11_FLAG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif()
  endif()
endif(CXX11)

include_directories(
  include
  internal
  internal/ceres
  ${GLOG_INCLUDE_DIRS})
# Eigen SparseQR generates various compiler warnings related to unused and
# uninitialised local variables, which prevents Ceres compilation as we use
# -Werror.  To avoid having to individually suppress these warnings around
# the #include statments for Eigen headers across all GCC/Clang versions, we
# tell CMake to treat Eigen headers as system headers.  This results in all
# compiler warnings from them being suppressed.
#
# Note that this is *not* propagated to clients, ie CERES_INCLUDE_DIRS
# used by clients after find_package(Ceres) does not identify Eigen as
# as system headers.
include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})

if (SUITESPARSE)
  include_directories(${SUITESPARSE_INCLUDE_DIRS})
endif (SUITESPARSE)

if (CXSPARSE)
  include_directories(${CXSPARSE_INCLUDE_DIRS})
endif (CXSPARSE)

if (GFLAGS)
  include_directories(${GFLAGS_INCLUDE_DIRS})
endif (GFLAGS)

if (BUILD_SHARED_LIBS)
  message("-- Building Ceres as a shared library.")
  # The CERES_BUILDING_SHARED_LIBRARY compile definition is NOT stored in
  # CERES_COMPILE_OPTIONS as it must only be defined when Ceres is compiled
  # not when it is used as it controls the CERES_EXPORT macro which provides
  # dllimport/export support in MSVC.
  add_definitions(-DCERES_BUILDING_SHARED_LIBRARY)
  list(APPEND CERES_COMPILE_OPTIONS CERES_USING_SHARED_LIBRARY)
else (BUILD_SHARED_LIBS)
  message("-- Building Ceres as a static library.")
endif (BUILD_SHARED_LIBS)

# Change the default build type from Debug to Release, while still
# supporting overriding the build type.
#
# The CACHE STRING logic here and elsewhere is needed to force CMake
# to pay attention to the value of these variables.
if (NOT CMAKE_BUILD_TYPE)
  message("-- No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
else (NOT CMAKE_BUILD_TYPE)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("\n=================================================================================")
    message("\n-- Build type: Debug. Performance will be terrible!")
    message("-- Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
    message("\n=================================================================================")
  endif (CMAKE_BUILD_TYPE STREQUAL "Debug")
endif (NOT CMAKE_BUILD_TYPE)

# Set the default Ceres flags to an empty string.
set (CERES_CXX_FLAGS)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  if (CMAKE_COMPILER_IS_GNUCXX)
    # Linux
    if (CMAKE_SYSTEM_NAME MATCHES "Linux")
      if (NOT GCC_VERSION VERSION_LESS 4.2)
        set (CERES_CXX_FLAGS "${CERES_CXX_FLAGS} -march=native -mtune=native")
      endif (NOT GCC_VERSION VERSION_LESS 4.2)
    endif (CMAKE_SYSTEM_NAME MATCHES "Linux")
    # Mac OS X
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
      set (CERES_CXX_FLAGS "${CERES_CXX_FLAGS} -msse3")
      # Use of -fast only applicable for Apple's GCC
      # Assume this is being used if GCC version < 4.3 on OSX
      execute_process(COMMAND ${CMAKE_C_COMPILER}
        ARGS ${CMAKE_CXX_COMPILER_ARG1} -dumpversion
        OUTPUT_VARIABLE GCC_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      if (GCC_VERSION VERSION_LESS 4.3)
        set (CERES_CXX_FLAGS "${CERES_CXX_FLAGS} -fast")
      endif (GCC_VERSION VERSION_LESS 4.3)
    endif (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  endif (CMAKE_COMPILER_IS_GNUCXX)
endif (CMAKE_BUILD_TYPE STREQUAL "Release")

set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CERES_CXX_FLAGS}")

if (MINGW)
  # MinGW produces code that segfaults when performing matrix multiplications
  # in Eigen when compiled with -O3 (see [1]), as such force the use of -O2
  # which works.
  #
  # [1] http://eigen.tuxfamily.org/bz/show_bug.cgi?id=556
  message("-- MinGW detected, forcing -O2 instead of -O3 in Release for Eigen due "
          "to a MinGW bug: http://eigen.tuxfamily.org/bz/show_bug.cgi?id=556")
  string(REPLACE "-O3" "-O2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  update_cache_variable(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif (MINGW)

# After the tweaks for the compile settings, disable some warnings on MSVC.
if (MSVC)
  # Disable signed/unsigned int conversion warnings.
  add_definitions("/wd4018")
  # Disable warning about using struct/class for the same symobl.
  add_definitions("/wd4099")
  # Disable warning about the insecurity of using "std::copy".
  add_definitions("/wd4996")
  # Disable performance warning about int-to-bool conversion.
  add_definitions("/wd4800")
  # Disable performance warning about fopen insecurity.
  add_definitions("/wd4996")
  # Disable warning about int64 to int32 conversion. Disabling
  # this warning may not be correct; needs investigation.
  # TODO(keir): Investigate these warnings in more detail.
  add_definitions("/wd4244")
  # It's not possible to use STL types in DLL interfaces in a portable and
  # reliable way. However, that's what happens with Google Log and Google Flags
  # on Windows. MSVC gets upset about this and throws warnings that we can't do
  # much about. The real solution is to link static versions of Google Log and
  # Google Test, but that seems tricky on Windows. So, disable the warning.
  add_definitions("/wd4251")

  # Google Flags doesn't have their DLL import/export stuff set up correctly,
  # which results in linker warnings. This is irrelevant for Ceres, so ignore
  # the warnings.
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4049")

  # Update the C/CXX flags for MSVC to use either the static or shared
  # C-Run Time (CRT) library based on the user option: MSVC_USE_STATIC_CRT.
  list(APPEND C_CXX_FLAGS
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL
    CMAKE_CXX_FLAGS_RELWITHDEBINFO)

  foreach(FLAG_VAR ${C_CXX_FLAGS})
    if (MSVC_USE_STATIC_CRT)
      # Use static CRT.
      if (${FLAG_VAR} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${FLAG_VAR} "${${FLAG_VAR}}")
      endif (${FLAG_VAR} MATCHES "/MD")
    else (MSVC_USE_STATIC_CRT)
      # Use shared, not static, CRT.
      if (${FLAG_VAR} MATCHES "/MT")
        string(REGEX REPLACE "/MT" "/MD" ${FLAG_VAR} "${${FLAG_VAR}}")
      endif (${FLAG_VAR} MATCHES "/MT")
    endif (MSVC_USE_STATIC_CRT)
  endforeach()

  # Tuple sizes of 10 are used by Gtest.
  add_definitions("-D_VARIADIC_MAX=10")
endif (MSVC)



# Use a larger inlining threshold for Clang, since it hobbles Eigen,
# resulting in an unreasonably slow version of the blas routines. The
# -Qunused-arguments is needed because CMake passes the inline
# threshold to the linker and clang complains about it and dies.
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Qunused-arguments -mllvm -inline-threshold=600")

  # Older versions of Clang (<= 2.9) do not support the 'return-type-c-linkage'
  # option, so check for its presence before adding it to the default flags set.
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-Wno-return-type-c-linkage"
                          HAVE_RETURN_TYPE_C_LINKAGE)
  if (HAVE_RETURN_TYPE_C_LINKAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-return-type-c-linkage")
  endif(HAVE_RETURN_TYPE_C_LINKAGE)
endif ()

# Xcode 4.5.x used Clang 4.1 (Apple version), this has a bug that prevents
# compilation of Ceres.
if (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER}
    ARGS ${CMAKE_CXX_COMPILER_ARG1} -dumpversion
    OUTPUT_VARIABLE CLANG_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  # Use version > 4.0 & < 4.2 to catch all 4.1(.x) versions.
  if (CLANG_VERSION VERSION_GREATER 4.0 AND
      CLANG_VERSION VERSION_LESS 4.2)
    message(FATAL_ERROR "You are attempting to build Ceres on OS X using Xcode "
      "4.5.x (Clang version: ${CLANG_VERSION}). This version of Clang has a "
      "bug that prevents compilation of Ceres, please update to "
      "Xcode >= 4.6.3.")
  endif (CLANG_VERSION VERSION_GREATER 4.0 AND
    CLANG_VERSION VERSION_LESS 4.2)
endif (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

# Configure the Ceres config.h compile options header using the current
# compile options and put the configured header into the Ceres build
# directory.  Note that the ceres/internal subdir in <build>/config where
# the configured config.h is placed is important, because Ceres will be
# built against this configured header, it needs to have the same relative
# include path as it would if it were in the source tree (or installed).
list(REMOVE_DUPLICATES CERES_COMPILE_OPTIONS)
include(CreateCeresConfig)
create_ceres_config("${CERES_COMPILE_OPTIONS}"
  ${CMAKE_BINARY_DIR}/config/ceres/internal)
# Force the location containing the configured config.h to the front of the
# include_directories list (by default it is appended to the back) to ensure
# that if the user has an installed version of Ceres in the same location as one
# of the dependencies (e.g. /usr/local) that we find the config.h we just
# configured, not the (older) installed config.h.
include_directories(BEFORE ${CMAKE_BINARY_DIR}/config)

add_subdirectory(internal/ceres)

if (BUILD_DOCUMENTATION)
  find_package(Sphinx QUIET)
  if (NOT SPHINX_FOUND)
    message("-- Failed to find Sphinx, disabling build of documentation.")
    update_cache_variable(BUILD_DOCUMENTATION OFF)
  else()
    # Generate the User's Guide (html).
    # The corresponding target is ceres_docs, but is included in ALL.
    message("-- Build the HTML documentation.")
    add_subdirectory(docs)
  endif()
endif (BUILD_DOCUMENTATION)

if (BUILD_EXAMPLES)
  message("-- Build the examples.")
  add_subdirectory(examples)
else (BUILD_EXAMPLES)
  message("-- Do not build any example.")
endif (BUILD_EXAMPLES)

# Setup installation of Ceres public headers.
file(GLOB CERES_HDRS ${CMAKE_SOURCE_DIR}/include/ceres/*.h)
install(FILES ${CERES_HDRS} DESTINATION include/ceres)

file(GLOB CERES_PUBLIC_INTERNAL_HDRS ${CMAKE_SOURCE_DIR}/include/ceres/internal/*.h)
install(FILES ${CERES_PUBLIC_INTERNAL_HDRS} DESTINATION include/ceres/internal)

# Also setup installation of Ceres config.h configured with the current
# build options into the installed headers directory.
install(FILES ${CMAKE_BINARY_DIR}/config/ceres/internal/config.h
        DESTINATION include/ceres/internal)

if (MINIGLOG)
  # Install miniglog header if being used as logging #includes appear in
  # installed public Ceres headers.
  install(FILES ${CMAKE_SOURCE_DIR}/internal/ceres/miniglog/glog/logging.h
          DESTINATION include/ceres/internal/miniglog/glog)
endif (MINIGLOG)

# Ceres supports two mechanisms by which it can be detected & imported into
# client code which uses CMake via find_package(Ceres):
#
#   1) Installation (e.g. to /usr/local), using CMake's install() function.
#
#   2) (Optional) Export of the current build directory into the local CMake
#      package registry, using CMake's export() function.  This allows use of
#      Ceres from other projects without requiring installation.
#
# In both cases, we need to generate a configured CeresConfig.cmake which
# includes additional autogenerated files which in concert create an imported
# target for Ceres in a client project when find_package(Ceres) is invoked.
# The key distinctions are where this file is located, and whether client code
# references installed copies of the compiled Ceres headers/libraries,
# (option #1: installation), or the originals in the source/build directories
# (option #2: export of build directory).
#
# NOTE: If Ceres is both exported and installed, provided that the installation
#       path is present in CMAKE_MODULE_PATH when find_package(Ceres) is called,
#       the installed version is preferred.

# Create a CeresConfigVersion.cmake file containing the version information,
# used by both export() & install().
configure_file("${CMAKE_SOURCE_DIR}/cmake/CeresConfigVersion.cmake.in"
               "${CMAKE_BINARY_DIR}/CeresConfigVersion.cmake" @ONLY)

# Install method #1: Put Ceres in CMAKE_INSTALL_PREFIX: /usr/local or equivalent.

# Set the install path for the installed CeresConfig.cmake configuration file
# relative to CMAKE_INSTALL_PREFIX.
if (WIN32)
  set(RELATIVE_CMAKECONFIG_INSTALL_DIR CMake)
else ()
  set(RELATIVE_CMAKECONFIG_INSTALL_DIR share/Ceres)
endif ()

# This "exports" for installation all targets which have been put into the
# export set "CeresExport". This generates a CeresTargets.cmake file which,
# when read in by a client project as part of find_package(Ceres) creates
# imported library targets for Ceres (with dependency relations) which can be
# used in target_link_libraries() calls in the client project to use Ceres.
install(EXPORT CeresExport
        DESTINATION ${RELATIVE_CMAKECONFIG_INSTALL_DIR} FILE CeresTargets.cmake)

# Save the relative path from the installed CeresConfig.cmake file to the
# install prefix.  We do not save an absolute path in case the installed package
# is subsequently relocated after installation (on Windows).
file(RELATIVE_PATH INSTALL_ROOT_REL_CONFIG_INSTALL_DIR
     ${CMAKE_INSTALL_PREFIX}/${RELATIVE_CMAKECONFIG_INSTALL_DIR}
     ${CMAKE_INSTALL_PREFIX})

# Configure a CeresConfig.cmake file for an installed version of Ceres from the
# template, reflecting the current build options.
#
# NOTE: The -install suffix is necessary to distinguish the install version from
#       the exported version, which must be named CeresConfig.cmake in
#       CMAKE_BINARY_DIR to be detected.  The suffix is removed when
#       it is installed.
set(SETUP_CERES_CONFIG_FOR_INSTALLATION TRUE)
configure_file("${CMAKE_SOURCE_DIR}/cmake/CeresConfig.cmake.in"
               "${CMAKE_BINARY_DIR}/CeresConfig-install.cmake" @ONLY)

# Install the configuration files into the same directory as the autogenerated
# CeresTargets.cmake file.  We include the find_package() scripts for libraries
# whose headers are included in the public API of Ceres and should thus be
# present in CERES_INCLUDE_DIRS.
install(FILES "${CMAKE_BINARY_DIR}/CeresConfig-install.cmake"
        RENAME CeresConfig.cmake
        DESTINATION ${RELATIVE_CMAKECONFIG_INSTALL_DIR})
install(FILES "${CMAKE_BINARY_DIR}/CeresConfigVersion.cmake"
              "${CMAKE_SOURCE_DIR}/cmake/FindEigen.cmake"
              "${CMAKE_SOURCE_DIR}/cmake/FindGlog.cmake"
        DESTINATION ${RELATIVE_CMAKECONFIG_INSTALL_DIR})

# Create an uninstall target to remove all installed files.
configure_file("${CMAKE_SOURCE_DIR}/cmake/uninstall.cmake.in"
               "${CMAKE_BINARY_DIR}/cmake/uninstall.cmake"
               @ONLY)
add_custom_target(uninstall
                  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake/uninstall.cmake)

# Install method #2: Put Ceres build into local CMake registry.
#
# Optionally export the Ceres build directory into the local CMake package
# registry (~/.cmake/packages on *nix & OS X).  This allows the detection &
# use of Ceres without requiring that it be installed.
if (EXPORT_BUILD_DIR)
  message("-- Export Ceres build directory to local CMake package registry.")

  # Save the relative path from the build directory to the source directory.
  file(RELATIVE_PATH INSTALL_ROOT_REL_CONFIG_INSTALL_DIR
    ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR})

  # Analogously to install(EXPORT ...), export the Ceres target from the build
  # directory as a package called Ceres into the local CMake package registry.
  export(TARGETS ceres FILE ${CMAKE_BINARY_DIR}/CeresTargets.cmake)
  export(PACKAGE ${CMAKE_PROJECT_NAME})

  # Configure a CeresConfig.cmake file for the export of the Ceres build
  # directory from the template, reflecting the current build options.
  set(SETUP_CERES_CONFIG_FOR_INSTALLATION FALSE)
  configure_file("${CMAKE_SOURCE_DIR}/cmake/CeresConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/CeresConfig.cmake" @ONLY)

endif (EXPORT_BUILD_DIR)
