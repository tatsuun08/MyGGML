#include "ggml.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <random>

// 時間計測用のヘルパー関数
double get_time_ms(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct simple_model {
    struct ggml_tensor* a; // 入力データ (F32)
    struct ggml_tensor* b; // 重み (F16 または Q8_0)
    struct ggml_context* ctx;
};

// 指定された型（F16やQ8_0）でモデルをロードし、メモリを確保する
void load_model(simple_model& model, const std::vector<float>& a, const void* b_data, size_t b_size, int rows_A, int cols_A, int rows_B, int cols_B, ggml_type b_type) {
    size_t ctx_size = 0;

    // 1. 入力テンソル（行列A）のサイズ (F32)
    ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32);

    // 2. 重みテンソル（行列B）のサイズ (F16 or Q8_0)
    ctx_size += b_size;

    // 3. 計算結果テンソルのサイズ (F32)
    // ggml_mul_mat(b, a) の結果は 行列Bの行数(rows_B) × 行列Aの行数(rows_A) になります
    ctx_size += rows_A * rows_B * ggml_type_size(GGML_TYPE_F32);

    // 4. オーバーヘッドとパディング
    ctx_size += 3 * ggml_tensor_overhead(); // テンソルはa, b, resultの3つ
    ctx_size += ggml_graph_overhead();
    ctx_size += 100 * 1024 * 1024; // アライメントや中間計算用の余裕分（8MB）

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    model.ctx = ggml_init(params);

    // テンソルの作成
    // ggml_new_tensor_2d(ctx, type, cols(ne0), rows(ne1)) の順序になることに注意
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_A, rows_A);
    model.b = ggml_new_tensor_2d(model.ctx, b_type, cols_B, rows_B);

    // データのコピー
    memcpy(model.a->data, a.data(), ggml_nbytes(model.a));
    memcpy(model.b->data, b_data, ggml_nbytes(model.b));
}

// 計算グラフの構築
struct ggml_cgraph* build_graph(const simple_model& model) {
    struct ggml_cgraph* gf = ggml_new_graph(model.ctx);

    // GGMLの制約: ggml_mul_mat(重み, 入力) の順序にする
    // 第1引数は量子化/F16等の型、第2引数はF32である必要があります
    struct ggml_tensor* result = ggml_mul_mat(model.ctx, model.b, model.a);

    ggml_build_forward_expand(gf, result);
    return gf;
}

int main(void) {
    ggml_time_init();

    // 速度差を明確にするための行列サイズ
    // A: M x K (入力データ), B: N x K (重み)
    const int M = 1; // rows_A
    const int K = 16384; // cols_A = cols_B
    const int N = 16384; // rows_B

    // 乱数でF32のデータを生成
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> matrix_A(M * K);
    std::vector<float> matrix_B(N * K);

    for (int i = 0; i < M * K; ++i) matrix_A[i] = dist(rng);
    for (int i = 0; i < N * K; ++i) matrix_B[i] = dist(rng);

    // ----------------------------------------------------
    // 1. FP16 用のデータ変換
    // ----------------------------------------------------
    std::vector<ggml_fp16_t> matrix_B_f16(N * K);
    for (int i = 0; i < N * K; ++i) {
        matrix_B_f16[i] = ggml_fp32_to_fp16(matrix_B[i]);
    }

    // ----------------------------------------------------
    // 2. INT8 (Q8_0) 用のデータ量子化 (最新API対応版)
    // ----------------------------------------------------
    // GGMLのQ8_0はブロックサイズが32なので、Kは32の倍数である必要があります
    assert(K % 32 == 0);

    // ggml_row_sizeを使って1行あたりの必要バイト数を取得し、全体のバッファサイズを計算
    size_t q8_0_size = ggml_row_size(GGML_TYPE_Q8_0, K) * N;
    std::vector<char> matrix_B_q8(q8_0_size);

    // F32配列から指定した型へ量子化
    ggml_quantize_chunk(
        GGML_TYPE_Q8_0,       // target quantization type
        matrix_B.data(),      // source F32 data
        matrix_B_q8.data(),   // destination buffer
        0,                    // start row
        N,                    // nrows
        K,                    // elements per row
        nullptr               // imatrix (Q8_0では不要)
    );

    // ----------------------------------------------------
    // 3. 実行とパフォーマンス計測
    // ----------------------------------------------------
    const int n_threads = 16; // ベンチマーク用のスレッド数
    const int num_runs = 20; // ループ回数

    // --- FP16 の測定 ---
    simple_model model_f16;
    load_model(model_f16, matrix_A, matrix_B_f16.data(), matrix_B_f16.size() * sizeof(ggml_fp16_t), M, K, N, K, GGML_TYPE_F16);

    // グラフの構築は1回だけ行う
    struct ggml_cgraph* gf_f16 = build_graph(model_f16);
    ggml_graph_compute_with_ctx(model_f16.ctx, gf_f16, n_threads); // ウォームアップ

    auto start_f16 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        ggml_graph_compute_with_ctx(model_f16.ctx, gf_f16, n_threads);
    }
    auto end_f16 = std::chrono::high_resolution_clock::now();
    double time_f16 = get_time_ms(start_f16, end_f16) / num_runs;

    // --- Q8_0 の測定 ---
    simple_model model_q8_0;
    load_model(model_q8_0, matrix_A, matrix_B_q8.data(), q8_0_size, M, K, N, K, GGML_TYPE_Q8_0);

    // グラフの構築は1回だけ行う
    struct ggml_cgraph* gf_q8 = build_graph(model_q8_0);
    ggml_graph_compute_with_ctx(model_q8_0.ctx, gf_q8, n_threads); // ウォームアップ

    auto start_q8 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        ggml_graph_compute_with_ctx(model_q8_0.ctx, gf_q8, n_threads);
    }
    auto end_q8 = std::chrono::high_resolution_clock::now();
    double time_q8 = get_time_ms(start_q8, end_q8) / num_runs;

    // --- 結果出力 ---
    printf("Matrix size: A(%d x %d), B(%d x %d)\n", M, K, N, K);
    printf("Threads    : %d\n", n_threads);
    printf("--------------------------------------\n");
    printf("FP16 Avg Compute Time: %8.2f ms\n", time_f16);
    printf("Q8_0 Avg Compute Time: %8.2f ms\n", time_q8);
    printf("--------------------------------------\n");

    // メモリ解放
    ggml_free(model_f16.ctx);
    ggml_free(model_q8_0.ctx);

    return 0;
}