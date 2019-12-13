



add_library(gadgetron_toolbox_cpuparallelmri
        cpu_sense_utilities.h
        cpu_sense_utilities.cpp
        hoNonCartesianSenseOperator.cpp
        )

set_target_properties(gadgetron_toolbox_cpuparallelmri PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

target_link_libraries(gadgetron_toolbox_cpuparallelmri
        gadgetron_toolbox_log
        gadgetron_toolbox_cpucore
        gadgetron_toolbox_cpucore_math
        gadgetron_toolbox_pmri
        gadgetron_toolbox_cpunfft
        )


target_include_directories(gadgetron_toolbox_cpuparallelmri
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS gadgetron_toolbox_cpuparallelmri
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        COMPONENT main)
