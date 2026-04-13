cmake_minimum_required(VERSION 3.14)
project(MyGGMLProject)

# 1. 出力先とC++標準の設定
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

# =====================================================================
# 【重要】GGMLの読み込み前に、すべてのフラグと設定を済ませる！
# =====================================================================

# GGMLのハードウェア最適化フラグ
set(GGML_NATIVE OFF CACHE BOOL "Enable GGML NATIVE" FORCE)
set(GGML_AVX2 ON CACHE BOOL "Enable AVX2" FORCE)
set(GGML_F16C ON CACHE BOOL "Enable F16C" FORCE)
set(GGML_FMA  ON CACHE BOOL "Enable FMA" FORCE)

# ※GGMLのバージョンによって変数名が異なる対策として、両方ONにしておくのが最も安全です
set(GGML_VNNI ON CACHE BOOL "Enable AVX-VNNI" FORCE)
set(GGML_AVX_VNNI ON CACHE BOOL "Enable AVX-VNNI" FORCE)

# clang-clの仕様に合わせて、C/C++全体のコンパイラフラグ変数に直接ねじ込む
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /clang:-mavxvnni -Wno-unused-function -Wno-error")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-mavxvnni -Wno-unused-function -Wno-error")
endif()

# =====================================================================
# 2. 設定がすべて終わった状態で、GGMLを読み込む
# =====================================================================
add_subdirectory(ggml)

# 念のためのダメ押し（ターゲットが作られた後に直接フラグを叩き込む）
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if (TARGET ggml-cpu)
        target_compile_options(ggml-cpu PRIVATE /clang:-mavxvnni -Wno-unused-function -Wno-error)
    endif()
endif()

# =====================================================================
# 3. 自分のアプリの設定
# =====================================================================
add_executable(my_app src/main.cpp)
target_include_directories(my_app PRIVATE include)
target_link_libraries(my_app PRIVATE ggml)