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

    

    init();  // 再次初始化矩阵
    cout << "Aligned Parallel Algorithm:" << endl;
    measureExecutionTime(AlignedParallelAlgorithm);



    return 0;
}