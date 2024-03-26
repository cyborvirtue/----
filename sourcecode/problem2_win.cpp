#include <iostream>
#include <windows.h>
using namespace std;

#define ull unsigned long long int

const ull N = 512; // 定义数组大小
ull a[N]; // 定义全局数组a
ull b[N]; // 定义辅助数组b，用于递归计算
int LOOP = 10; // 定义循环次数，用于可能的性能测试

void init()
{
    for (ull i = 0; i < N; i++)
        a[i] = i;
}

void ordinary()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq); // 获取性能计数器的频率
    QueryPerformanceCounter((LARGE_INTEGER*)&begin); // 获取开始时间
    for (int l = 0; l < LOOP; l++)
    {
        ull sum = 0; // 定义求和变量
        for (int i = 0; i < N; i++)
            sum += a[i]; // 计算数组a的所有元素之和
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end); // 获取结束时间
    cout << "ordinary:" << (end - begin) * 1000.0 / freq / LOOP << "ms" << endl; // 输出执行时间
}

void optimize()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq); // 获取性能计数器的频率
    QueryPerformanceCounter((LARGE_INTEGER*)&begin); // 获取开始时间
    for (int l = 0; l < LOOP; l++)
    {
        ull sum1 = 0, sum2 = 0; // 定义两个求和变量
        for (int i = 0; i < N - 1; i += 2)
            sum1 += a[i], sum2 += a[i + 1]; // 两个元素一组进行求和，以减少循环的迭代次数
        ull sum = sum1 + sum2; // 将两个部分的和相加
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end); // 获取结束时间
    cout << "optimize:" << (end - begin) * 1000.0 / freq / LOOP << "ms" << endl; // 输出执行时间
}

void optimizeFourWay() {
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq); // 获取性能计数器的频率
    QueryPerformanceCounter((LARGE_INTEGER*)&begin); // 获取开始时间

    for (int l = 0; l < LOOP; l++) {
        ull sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0; // 定义四个求和变量
        for (int i = 0; i < N; i += 4) {
            sum1 += a[i];
            if (i + 1 < N) sum2 += a[i + 1];
            if (i + 2 < N) sum3 += a[i + 2];
            if (i + 3 < N) sum4 += a[i + 3];
        }
        ull sum = sum1 + sum2 + sum3 + sum4; // 将四个部分的和相加
    }

    QueryPerformanceCounter((LARGE_INTEGER*)&end); // 获取结束时间
    cout << "optimizeFourWay: " << (end - begin) * 1000.0 / freq / LOOP << "ms" << endl; // 输出执行时间
}
int main()
{
    init(); // 初始化数组a
    ordinary(); // 执行普通求和方法
    optimize(); // 执行优化后的求和方法
    optimizeFourWay();
}
