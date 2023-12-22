cmake_minimum_required(VERSION 3.5)
project(eot_cpp)

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include(FetchContent)
FetchContent_Declare(matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
        GIT_TAG origin/master)
FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
    FetchContent_Populate(matplotplusplus)
    add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

find_package(PCL 1.8 REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(project_SOURCES ./main.cpp
                    ./EOT.cpp
                    ./utilities.cpp)

set(project_HEADERS ./EOT.h
                    ./utilities.h
                    ./dbscan.h
                    ./globaldef.h)
 
add_executable(${PROJECT_NAME} ${project_SOURCES} ${project_HEADERS})

target_link_libraries (${PROJECT_NAME} ${PCL_LIBRARIES} ${Boost_LIBRARIES} matplot)
