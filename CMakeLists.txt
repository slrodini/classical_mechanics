cmake_minimum_required(VERSION 3.20)

project(MD CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-O3 -funroll-loops -Wall -Wextra -mtune=native -march=native")

add_executable(md_rk md_rk.cc)
target_link_libraries(md_rk PUBLIC /usr/local/lib/libraylib.a)


add_executable(md_rkn md_rkn.cc)
target_link_libraries(md_rkn PUBLIC /usr/local/lib/libraylib.a)



add_executable(ex_3_18 ex_3_18.cc)
target_link_libraries(ex_3_18 PUBLIC /usr/local/lib/libraylib.a)
