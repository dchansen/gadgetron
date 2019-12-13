

add_library(gadgetron_toolbox_pmri INTERFACE
        )
target_include_directories(gadgetron_toolbox_pmri
        INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(gadgetron_toolbox_pmri INTERFACE gadgetron_toolbox_operator)
if(CUDA_FOUND)


  add_subdirectory(gpu)
endif()
add_subdirectory(cpu)