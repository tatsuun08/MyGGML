# MyGGML
## xeon 6を用いた行列積の性能比較<br>
行列A[M x K] B[K x N]の行列積の理論値計算回数

### 行列サイズの設定

$M = 512, K = 16384, N = 16384$

### 実行環境
| 項目 | スペック詳細 |
| :--- | :--- |
| **CPU** | Intel Xeon 6 6740P(48 core) |
| **Memory** | DDR5-6400 256GB  |
| **Compiler** | GCC|
| **OS** | Ubuntu 20.04 LTS |

### 演算回数（理論値）

$計算式: 2 \times M \times K \times N$


$総演算回数: 2 \times 512 \times 16384 \times 16384 \simeq 275 \times 10^9 = 275[Gops]$

||1サイクルあたりの演算|理論性能[TFLOPS]|実行時間[ms]|
|----|:----:|:----:|:----:|
|①スカラ計算(1 core)|1|0.0021|約130800|
|②AVX-512 (1 core) <br> FMA x 2 ※1※2|$32\times2\times2=128$|0.4224 ※3|約650|
|③AVX-512 (48 core)|6144|20.275|約13.5|
|④AMX (48 core)|49152|162.2|約1.7|

※1 FMA(Fused Multily Add) 乗算と加算を1命令で実行できる<br>
※2 Xeon 6はFMA演算器を2つ搭載している <br>
※3 SIMD実行時はターボクロックで動作することを想定している




