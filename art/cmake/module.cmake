function(nart_case_module_add)
    set(options)
    set(oneValueArgs CASE_MODULE)
    set(multiValueArgs SRC LINK FLAGS INCLUDE LINK_STATIC LINK_SHARED)
    cmake_parse_arguments(PP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    string(TOUPPER ${PP_CASE_MODULE} UPPER_MODULE_NAME)

    add_library(nart_case_module_${PP_CASE_MODULE}_static STATIC ${PP_SRC})
    add_library(nart_case_module_${PP_CASE_MODULE}_shared SHARED ${PP_SRC})

    target_link_libraries(nart_case_module_${PP_CASE_MODULE}_static PUBLIC nart_case_include nart_case_static)
    target_link_libraries(nart_case_module_${PP_CASE_MODULE}_shared PUBLIC nart_case_include nart_case_shared)

    set_target_properties(
        nart_case_module_${PP_CASE_MODULE}_static nart_case_module_${PP_CASE_MODULE}_shared
        PROPERTIES
        OUTPUT_NAME art_module_${PP_CASE_MODULE}
    )
    target_link_libraries(nart_case_module_${PP_CASE_MODULE}_static PUBLIC ${PP_LINK} ${PP_LINK_STATIC})
    target_link_libraries(nart_case_module_${PP_CASE_MODULE}_shared PUBLIC ${PP_LINK} ${PP_LINK_SHARED})

    target_compile_options(nart_case_module_${PP_CASE_MODULE}_static PUBLIC ${PP_FLAGS})
    target_compile_options(nart_case_module_${PP_CASE_MODULE}_shared PUBLIC ${PP_FLAGS})

    target_compile_definitions(nart_case_module_${PP_CASE_MODULE}_static PUBLIC BUILD_WITH_${UPPER_MODULE_NAME})
    target_compile_definitions(nart_case_module_${PP_CASE_MODULE}_shared PUBLIC BUILD_WITH_${UPPER_MODULE_NAME})

    target_include_directories(nart_case_module_${PP_CASE_MODULE}_static PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PP_INCLUDE})
    target_include_directories(nart_case_module_${PP_CASE_MODULE}_shared PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PP_INCLUDE})

    # install
    install(
        TARGETS nart_case_module_${PP_CASE_MODULE}_static nart_case_module_${PP_CASE_MODULE}_shared
        EXPORT nart_case_targets
        LIBRARY DESTINATION lib COMPONENT Runtime
        ARCHIVE DESTINATION lib COMPONENT Development
    )
endfunction()
