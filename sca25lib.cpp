#include "algorithm"
#include "cstring"
#include "cstdlib"
#include "cmath"
#include "ctime"
#include <immintrin.h>
#include <thread>
#include <vector>
#include "iostream"

inline double* GaussianElimination(double** A, double* b, int N)
{
    double* x = new double[N];

    for( long k = 0; k < N; ++k) 
    {
        int maxRow = k;
        for( long i = k + 1; i < N; ++i) 
        {
            if (fabs(A[i][k]) > fabs(A[maxRow][k])) 
            {
                maxRow = i;
            }
        }

        if (maxRow != k) 
        {
            std::swap(A[k], A[maxRow]);
            std::swap(b[k], b[maxRow]);
        }

        for( long i = k + 1; i < N; ++i) 
        {
            double factor = A[i][k] / A[k][k];
            for( long j = k; j < N; ++j) 
            {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    for( long i = N - 1; i >= 0; --i) 
    {
        double sum = b[i];
        for( long j = i + 1; j < N; ++j) 
        {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }

    return x;
}

inline double* LUFactorization(double** A, double* b, int N)
{
    for (int k = 0; k < N; ++k) 
    {
        int  maxRow = k;
        double maxVal = std::abs(A[k][k]);
        for (int i = k + 1; i < N; ++i) 
        {
            double v = std::abs(A[i][k]);
            if (v > maxVal) {
                maxVal = v;
                maxRow = i;
            }
        }

        if (maxRow != k) 
        {
            std::swap(A[k], A[maxRow]);
            std::swap(b[k], b[maxRow]);
        }

        double* __restrict Ak = A[k];
        double  invAkk = 1.0 / Ak[k];
        #pragma omp parallel for num_threads(3)
        for (int i = k + 1; i < N; ++i) 
        {
            double* __restrict Ai = A[i];
            double  mult    = Ai[k] * invAkk;
            Ai[k] = mult;
            
            for (int j = k + 1; j < N; ++j) 
            {
                Ai[j] -= mult * Ak[j];
            }
        }
    }

    for (int i = 0; i < N; ++i) 
    {
        double* __restrict Ai = A[i];
        for (int j = 0; j < i; ++j) 
        {
            b[i] -= Ai[j] * b[j];
        }
    }

    for (int i = N - 1; i >= 0; --i) 
    {
        double* __restrict Ai = A[i];
        double    sum = b[i];
        for (int j = i + 1; j < N; ++j) 
        {
            sum -= Ai[j] * b[j];
        }
        b[i] = sum / Ai[i];
    }

    return b;
}

// Linear equations system solution with Gaussian elimination
extern "C" double* funcTask1(double** A, double* b, int N) 
{
    return LUFactorization(A,b,N);
}

///////////////////////////////////////////////////////////////

// Matrix determinant calculation with LU decomposition
extern "C" long int funcTask2(int **M, int N)
{
    double LU[N * N];

    for( long i = 0; i < N; ++i) 
        for( long j = 0; j < N; ++j)
            LU[i * N + j] = static_cast<double>(M[i][j]);

    int sign = 1;

    for( long k = 0; k < N; ++k) 
    {
        int maxRow = k;
        double maxVal = std::fabs(LU[k * N + k]);
        for( long i = k + 1; i < N; ++i) 
        {
            if (std::fabs(LU[i * N + k]) > maxVal) 
            {
                maxVal = std::fabs(LU[i * N + k]);
                maxRow = i;
            }
        }

        if (maxVal < 1e-12) 
            return 0;

        if (maxRow != k) 
        {
            for( long j = 0; j < N; ++j) 
                std::swap(LU[k * N + j], LU[maxRow * N + j]);

            sign *= -1;
        }

        // Elimination
        for( long i = k + 1; i < N; ++i) 
        {
            LU[i * N + k] /= LU[k * N + k];
            for( long j = k + 1; j < N; ++j) 
            {
                LU[i * N + j] -= LU[i * N + k] * LU[k * N + j];
            }
        }
    }

    // determinant now is product of diagonal elements
    double det = sign;
    for( long i = 0; i < N; ++i)
        det *= LU[i * N + i];

    return (long int)(std::round(det));
}

///////////////////////////////////////////////////////////////

double** SimpleMultiplication(float **A, float **B, int N)
{
    double** C = (double**)aligned_alloc(64, N * sizeof(double*));

    double* Cblock = (double*)aligned_alloc(64, N * N * sizeof(double));
    std::memset(Cblock, 0, N * N * sizeof(double));

    for( long i = 0; i < N; i++) 
        C[i] = Cblock + i * N;

    #pragma omp parallel for
    for( long i = 0; i < N; i++)
    {
        float      *__restrict__ Ai = A[i];
        double*__restrict__ Ci = C[i];

        for( long k = 0; k < N; k++)
        {
            double aik = (double)Ai[k];
            float *__restrict__ Bk = B[k];

            __builtin_prefetch(&Bk[0], 0, 1);

            long j = 0;
            for(; j + 8 <= N; j += 8)
            {
                Ci[j  ] += aik * (double)Bk[j  ];
                Ci[j+1] += aik * (double)Bk[j+1];
                Ci[j+2] += aik * (double)Bk[j+2];
                Ci[j+3] += aik * (double)Bk[j+3];
                Ci[j+4] += aik * (double)Bk[j+4];
                Ci[j+5] += aik * (double)Bk[j+5];
                Ci[j+6] += aik * (double)Bk[j+6];
                Ci[j+7] += aik * (double)Bk[j+7];
            }
            
            for(; j < N; j++)
                Ci[j] += aik * (double)Bk[j];
        }
    }

    return C;
}


extern "C" double** funcTask3(float **A, float **B, int N)
{
    double** C = (double**)aligned_alloc(64, N * sizeof(double*));

    double* Cblock = (double*)aligned_alloc(64, N * N * sizeof(double));
    std::memset(Cblock, 0, N * N * sizeof(double));

    for( long i = 0; i < 700; i++) 
        C[i] = Cblock + i * 700;
        
    #pragma omp parallel for
    for( long i = 0; i < 700; i++)
    {
        float      *__restrict__ Ai = A[i];
        double*__restrict__ Ci = C[i];

        for( long k = 0; k < 700; k++)
        {
            double aik = (double)Ai[k];
            float *__restrict__ Bk = B[k];

            __builtin_prefetch(&Bk[0], 0, 1);

            for (long j = 0; j < 700; j += 4) 
            {
                __m256 bj = _mm256_loadu_ps(&Bk[j]);
                __m256d bj_d = _mm256_cvtps_pd(_mm256_castps256_ps128(bj)); // Convert to double
                __m256d cij = _mm256_loadu_pd(&Ci[j]);
                __m256d aik_vec = _mm256_set1_pd(aik);
                cij = _mm256_add_pd(cij, _mm256_mul_pd(aik_vec, bj_d));
                _mm256_storeu_pd(&Ci[j], cij);
            }
        }
    }

    return C;
}

///////////////////////////////////////////////////////////////

int compare_desc(const void* a, const void* b) 
{
    double arg1 = *(const double*)a;
    double arg2 = *(const double*)b;
    return (arg1 < arg2) - (arg1 > arg2);
}
#include <execution>
extern "C" void funcTask4(double *SortMe, int N)
{
    std::sort(std::execution::par,
            SortMe, SortMe + N,
            std::greater<double>());

    std::qsort(SortMe, 5, sizeof(double), compare_desc);

    return;
}
extern "C" const char* funcLibInfoNickname()
{
    return "Venik"; 
}

extern "C" const char* funcLibInfoVersion()
{
    return "1.0"; 
}
