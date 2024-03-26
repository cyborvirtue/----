#include <iostream>
#include <windows.h>
using namespace std;

#define ull unsigned long long int

const ull N = 512; // ���������С
ull a[N]; // ����ȫ������a
ull b[N]; // ���帨������b�����ڵݹ����
int LOOP = 10; // ����ѭ�����������ڿ��ܵ����ܲ���

void init()
{
    for (ull i = 0; i < N; i++)
        a[i] = i;
}

void ordinary()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq); // ��ȡ���ܼ�������Ƶ��
    QueryPerformanceCounter((LARGE_INTEGER*)&begin); // ��ȡ��ʼʱ��
    for (int l = 0; l < LOOP; l++)
    {
        ull sum = 0; // ������ͱ���
        for (int i = 0; i < N; i++)
            sum += a[i]; // ��������a������Ԫ��֮��
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end); // ��ȡ����ʱ��
    cout << "ordinary:" << (end - begin) * 1000.0 / freq / LOOP << "ms" << endl; // ���ִ��ʱ��
}

void optimize()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq); // ��ȡ���ܼ�������Ƶ��
    QueryPerformanceCounter((LARGE_INTEGER*)&begin); // ��ȡ��ʼʱ��
    for (int l = 0; l < LOOP; l++)
    {
        ull sum1 = 0, sum2 = 0; // ����������ͱ���
        for (int i = 0; i < N - 1; i += 2)
            sum1 += a[i], sum2 += a[i + 1]; // ����Ԫ��һ�������ͣ��Լ���ѭ���ĵ�������
        ull sum = sum1 + sum2; // ���������ֵĺ����
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end); // ��ȡ����ʱ��
    cout << "optimize:" << (end - begin) * 1000.0 / freq / LOOP << "ms" << endl; // ���ִ��ʱ��
}

void optimizeFourWay() {
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq); // ��ȡ���ܼ�������Ƶ��
    QueryPerformanceCounter((LARGE_INTEGER*)&begin); // ��ȡ��ʼʱ��

    for (int l = 0; l < LOOP; l++) {
        ull sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0; // �����ĸ���ͱ���
        for (int i = 0; i < N; i += 4) {
            sum1 += a[i];
            if (i + 1 < N) sum2 += a[i + 1];
            if (i + 2 < N) sum3 += a[i + 2];
            if (i + 3 < N) sum4 += a[i + 3];
        }
        ull sum = sum1 + sum2 + sum3 + sum4; // ���ĸ����ֵĺ����
    }

    QueryPerformanceCounter((LARGE_INTEGER*)&end); // ��ȡ����ʱ��
    cout << "optimizeFourWay: " << (end - begin) * 1000.0 / freq / LOOP << "ms" << endl; // ���ִ��ʱ��
}
int main()
{
    init(); // ��ʼ������a
    ordinary(); // ִ����ͨ��ͷ���
    optimize(); // ִ���Ż������ͷ���
    optimizeFourWay();
}
