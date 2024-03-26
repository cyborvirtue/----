#include <iostream>
#include <windows.h>
#include <omp.h> // 引入OpenMP支持
#include <immintrin.h> // 引入SIMD指令集支持
using namespace std;
const int N = 9000;
int a[N];
int b[N][N];
int sum[N];
int LOOP = 10;

void init()
{
    for (int i = 0; i < N; i++)
        a[i] = i;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            b[i][j] = i + j;
}

void ordinary()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&begin);
    for (int l = 0; l < LOOP; l++)
    {
        for (int i = 0; i < N; i++)
        {
            sum[i] = 0;
            for (int j = 0; j < N; j++)
                sum[i] += a[j] * b[j][i];
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end);
    cout << "ordinary:" << (end - begin) * 1000.0 / freq / LOOP << "ms" << endl;
}

void optimize()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&begin);
    for (int l = 0; l < LOOP; l++)
    {
        for (int i = 0; i < N; i++)
            sum[i] = 0;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                sum[i] += a[j] * b[j][i];
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end);
    cout << "optimize:" << (end - begin) * 1000.0 / freq / LOOP << "ms" << endl;
}

void unroll()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&begin);
    for (int l = 0; l < LOOP; l++)
    {
        for (int i = 0; i < N; i++)
            sum[i] = 0;
        for (int j = 0; j < N; j += 10)
        {
            int tmp0 = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0, tmp5 = 0, tmp6 = 0, tmp7 = 0, tmp8 = 0, tmp9 = 0;
            for (int i = 0; i < N; i++)
            {
                tmp0 += a[j + 0] * b[j + 0][i];
                tmp1 += a[j + 1] * b[j + 1][i];
                tmp2 += a[j + 2] * b[j + 2][i];
                tmp3 += a[j + 3] * b[j + 3][i];
                tmp4 += a[j + 4] * b[j + 4][i];
                tmp5 += a[j + 5] * b[j + 5][i];
                tmp6 += a[j + 6] * b[j + 6][i];
                tmp6 += a[j + 6] * b[j + 6][i];
                tmp7 += a[j + 7] * b[j + 7][i];
                tmp8 += a[j + 8] * b[j + 8][i];
                tmp9 += a[j + 9] * b[j + 9][i];
            }
            sum[j + 0] = tmp0;
            sum[j + 1] = tmp1;
            sum[j + 2] = tmp2;
            sum[j + 3] = tmp3;
            sum[j + 4] = tmp4;
            sum[j + 5] = tmp5;
            sum[j + 6] = tmp6;
            sum[j + 7] = tmp7;
            sum[j + 8] = tmp8;
            sum[j + 9] = tmp9;
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end);
    cout << "unroll:" << (end - begin) * 1000.0 / freq / LOOP << "ms" << endl;
}

void simd_and_parallel_optimized() {
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&begin);
    for (int l = 0; l < LOOP; l++) {
        for (int i = 0; i < N; i++)
            sum[i] = 0;

#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            __m256i vsum = _mm256_setzero_si256(); // 使用SIMD指令集初始化累加器
            for (int j = 0; j < N; j += 8) { // 假设N是8的倍数
                __m256i va = _mm256_loadu_si256((__m256i*) & a[j]); // 加载a[j]到j+7的元素
                __m256i vb = _mm256_loadu_si256((__m256i*) & b[j][i]); // 加载b的对应行
                __m256i vprod = _mm256_mullo_epi32(va, vb); // 对应元素相乘
                vsum = _mm256_add_epi32(vsum, vprod); // 将乘积累加到vsum
            }
            // 由于vsum中包含了多个并行计算的结果，需要将它们合并为一个标量值
            int buffer[8];
            _mm256_storeu_si256((__m256i*)buffer, vsum); // 将vsum存储到临时缓冲区
            for (int k = 0; k < 8; ++k) { // 遍历缓冲区，完成累加
                sum[i] += buffer[k];
            }
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end);
    cout << "simd_and_parallel_optimized: " << (end - begin) * 1000.0 / freq / LOOP << " ms" << endl;
}

int main()
{
    init();
    ordinary();
    optimize();
    unroll();
    simd_and_parallel_optimized();
}