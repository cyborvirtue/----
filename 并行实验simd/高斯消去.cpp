#include <iostream>
#include <chrono>
#include<immintrin.h>

using namespace std;
using namespace chrono;

const int N = 1400;
float a[N][N];

void init()
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            a[i][j] = float(rand()) / 10;
}
void Serial()//串行算法
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
            a[k][j] /= a[k][k]; // 将当前行的每个元素除以对角元素，使对角元素变为1
            a[k][k] = 1.0; 

        // 对当前行下面的所有行进行消元操作，目的是将这些行的当前列元素变为0
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++)
                a[i][j] -= a[i][k] * a[k][j]; 
            a[i][k] = 0; 
        }
    }
}
/*void NEONGaussianElimination() {
    for (int k = 0; k < N; k++) {
        float32x4_t vk = vdupq_n_f32(a[k][k]);
        for (int j = k + 1; j + 3 < N; j += 4) {
            float32x4_t va = vld1q_f32(&a[k][j]);
            va = vdivq_f32(va, vk);
            vst1q_f32(&a[k][j], va);
        }
        a[k][k] = 1.0;

        for (int i = k + 1; i < N; i++) {
            float32x4_t vi = vdupq_n_f32(a[i][k]);
            for (int j = k + 1; j + 3 < N; j += 4) {
                float32x4_t vak = vld1q_f32(&a[k][j]);
                float32x4_t vai = vld1q_f32(&a[i][j]);
                float32x4_t vx = vmulq_f32(vi, vak);
                vai = vsubq_f32(vai, vx);
                vst1q_f32(&a[i][j], vai);
            }
            a[i][k] = 0.0;
        }
    }
}*/
void LU_avx() 
{
    float new_mat[N][N];  // 临时矩阵存储转换后的数据

    // 复制原矩阵到新矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            new_mat[i][j] = a[i][j];
        }
    }

    for (int i = 0; i < N; i++) {
        if (new_mat[i][i] == 0) continue;  // 如果对角元素为0，跳过当前行

        float div = 1.0f / new_mat[i][i];  // 计算并取倒数以减少除法操作的成本
        __m256 div8 = _mm256_set1_ps(div);

        for (int j = i + 1; j < N; j++) {
            int k = i;
            for (; k <= N - 8; k += 8) { // 保证处理的元素总数不超出列的界限
                __m256 mat_i = _mm256_loadu_ps(&new_mat[i][k]);
                __m256 mat_j = _mm256_loadu_ps(&new_mat[j][k]);
                __m256 res = _mm256_fnmadd_ps(mat_i, div8, mat_j);
                _mm256_storeu_ps(&new_mat[j][k], res);
            }
            for (; k < N; k++) {  // 处理尾部不足8个的部分
                new_mat[j][k] -= new_mat[i][k] * div;
            }
        }

        new_mat[i][i] = 1.0f;
        for (int j = i + 1; j < N; j++) {
            new_mat[j][i] = 0.0;
        }
    }

    // 将结果复制回原始矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = new_mat[i][j];
        }
    }
}

void SSE()//SSE指令集
{
    __m128 t0, t1, t2, t3;
    for (int k = 0; k < N; k++)
    {
        float temp1[4] = { a[k][k],a[k][k],a[k][k],a[k][k] };
        t0 = _mm_loadu_ps(temp1);
        int j;
        for (j = k + 1; j + 3 < N; j += 4)//j+4的值为下一次load操作的起始位置
        {
            t1 = _mm_loadu_ps(a[k] + j);
            t2 = _mm_div_ps(t1, t0);
            _mm_storeu_ps(a[k] + j, t2);
        }
        for (; j < N; j++)
            a[k][j] /= a[k][k];

        a[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float temp2[4] = { a[i][k],a[i][k],a[i][k],a[i][k] };
            t0 = _mm_loadu_ps(temp2);
            int j;
            for (j = k + 1; j + 3 < N; j += 4)
            {
                t1 = _mm_loadu_ps(a[k] + j);
                t2 = _mm_loadu_ps(a[i] + j);
                t3 = _mm_mul_ps(t0, t1);
                t2 = _mm_sub_ps(t2, t3);
                _mm_storeu_ps(a[i] + j, t2);
            }
            for (; j < N; j++)
                a[i][j] -= a[i][k] * a[k][j];
            a[i][k] = 0.0;
        }
    }

}

void AlignedParallelAlgorithm()
{
    __m128 t0, t1, t2, t3;
    for (int k = 0; k < N; k++)
    {
        float temp1[4] = { a[k][k],a[k][k],a[k][k],a[k][k] };
        t0 = _mm_loadu_ps(temp1);
        int j;
        for (j = k + 1; j < N; j++) {
            if (((size_t)(a[k] + j)) % 16 == 0)
                break;
            a[k][j] /= a[k][k];
        }
        for (; j + 3 < N; j += 4)//j+4的值为下一次load操作的起始位置
        {
            t1 = _mm_load_ps(a[k] + j);
            t2 = _mm_div_ps(t1, t0);
            _mm_store_ps(a[k] + j, t2);
        }
        for (; j < N; j++)
            a[k][j] /= a[k][k];

        a[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float temp2[4] = { a[i][k],a[i][k],a[i][k],a[i][k] };
            t0 = _mm_loadu_ps(temp2);
            int j;
            for (j = k + 1; j < N; j++) {
                if (((size_t)(a[i] + j)) % 16 == 0)
                    break;
                a[i][j] -= a[i][k] * a[k][j];
            }
            for (; j + 3 < N; j += 4)
            {
                t1 = _mm_loadu_ps(a[k] + j);
                t2 = _mm_load_ps(a[i] + j);
                t3 = _mm_mul_ps(t0, t1);
                t2 = _mm_sub_ps(t2, t3);
                _mm_store_ps(a[i] + j, t2);
            }
            for (; j < N; j++)
                a[i][j] -= a[i][k] * a[k][j];
            a[i][k] = 0.0;
        }
    }

}
void measureExecutionTime(void (*algorithm)()) {
    auto start = high_resolution_clock::now();
    algorithm();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Execution time: " << duration.count() / 1000.0 << " milliseconds" << endl;
}
void show()
{

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            cout << a[i][j] << ' ';
        cout << endl;
    }
}


int main()
{

    init();  // 初始化矩阵

    cout << "Serial:" << endl;
    measureExecutionTime(Serial);

    init();  // 重新初始化矩阵以清除之前的计算结果
    cout << "Parallel Algorithm:" << endl;
    measureExecutionTime(SSE);

    init();
    cout << "AVX" << endl;
    measureExecutionTime(LU_avx);

    init();  // 再次初始化矩阵
    cout << "Aligned Parallel Algorithm:" << endl;
    measureExecutionTime(AlignedParallelAlgorithm);


    
    return 0;
}
