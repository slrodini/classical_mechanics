cmake_minimum_required(VERSION 3.20)

project(MD CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED)


add_executable(main main.cc)
target_link_libraries(main PUBLIC /usr/local/lib/libraylib.a)


add_executable(rkn_main rkn_main.cc)
target_link_libraries(rkn_main PUBLIC /usr/local/lib/libraylib.a)