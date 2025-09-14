#include "algorithm"
#include "cstring"
#include "cmath"
#include "omp.h"

inline double* GaussianElimination(double** A, double* b, int N)
{
    double* x = new double[N];

    for (int k = 0; k < N; ++k) 
    {
        int maxRow = k;
        for (int i = k + 1; i < N; ++i) 
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

        for (int i = k + 1; i < N; ++i) 
        {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < N; ++j) 
            {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    for (int i = N - 1; i >= 0; --i) 
    {
        double sum = b[i];
        for (int j = i + 1; j < N; ++j) 
        {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }

    return x;
}

inline double* LUFactorization(double** A, double* b, int N)
{
    double* __restrict__ A_flat = new double[N * N];

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A_flat[i * N + j] = A[i][j];

    for (int k = 0; k < N; ++k) 
    {
        int maxIdx = k; 
        double maxVal = std::abs(A_flat[k * N + k]); 
        for (int i = k + 1; i < N; ++i) 
        { 
            if (std::abs(A_flat[i * N + k]) > maxVal) 
            { 
                maxVal = std::abs(A_flat[i * N + k]); 
                maxIdx = i; 
            } 
        } 

        if (maxIdx != k) 
        {
            for (int j = 0; j < N; ++j) 
                std::swap(A_flat[k * N + j], A_flat[maxIdx * N + j]);

            std::swap(b[k], b[maxIdx]); 
        }

        // LU factorization 
        double inv = 1.0 / A_flat[k * N + k];
        #pragma omp parallel for
        for (int i = k + 1; i < N; ++i) 
        {
            __builtin_prefetch(&A_flat[i * N + k], 0, 1);
            A_flat[i * N + k] *= inv; 
            for (int j = k + 1; j < N; ++j) 
                A_flat[i * N + j] -= A_flat[i * N + k] * A_flat[k * N + j]; 
        } 
    } 

    // Forward substitution 
    for (int i = 0; i < N; ++i) 
    { 
        for (int j = 0; j < i; ++j) 
            b[i] -= A_flat[i * N + j] * b[j]; 
    } 

    double* x = new double[N];
    // Back substitution
    for (int i = N - 1; i >= 0; --i) 
    { 
        double sum = b[i]; 
        for (int j = i + 1; j < N; ++j) 
            sum -= A_flat[i * N + j] * x[j]; 
        x[i] = sum / A_flat[i * N + i]; 
    }

    delete[] A_flat;

    return x;
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
    double **LU = new double*[N];
    for (int i = 0; i < N; ++i) 
    {
        LU[i] = new double[N];
        for (int j = 0; j < N; ++j)
        {
            LU[i][j] = static_cast<double>(M[i][j]);
        }
    }

    int sign = 1;

    for (int k = 0; k < N; ++k) 
    {
        int maxRow = k;
        double maxVal = std::fabs(LU[k][k]);
        for (int i = k + 1; i < N; ++i) 
        {
            if (std::fabs(LU[i][k]) > maxVal) 
            {
                maxVal = std::fabs(LU[i][k]);
                maxRow = i;
            }
        }

        if (maxVal == 0.0)
        {
            // Singular matrix
            sign = 0;
            break;
        }

        if (maxRow != k) 
        {
            std::swap(LU[k], LU[maxRow]);
            sign *= -1;
        }

        // Elimination
        for (int i = k + 1; i < N; ++i) 
        {
            LU[i][k] /= LU[k][k];
            for (int j = k + 1; j < N; ++j) 
            {
                LU[i][j] -= LU[i][k] * LU[k][j];
            }
        }
    }

    // determinant now is product of diagonal elements
    double det = sign;
    for (int i = 0; i < N; ++i)
        det *= LU[i][i];

    for (int i = 0; i < N; ++i)
        delete[] LU[i];
    delete[] LU;

    return static_cast<long int>(std::round(det));
}

///////////////////////////////////////////////////////////////

long double** SimpleMultiplication(float **A, float **B, int N)
{
    long double a_ik;
    long double** C = new long double*[N];

    #pragma omp parallel for num_threads(3) 
    for(int i = 0; i < N; i++) 
    { 
        C[i] = new long double[N]{0}; 
        for(int k = 0; k < N; k++) 
        {
            a_ik = (long double)A[i][k]; 
            for(int j = 0; j < N; j++) 
            {
                C[i][j] += a_ik * static_cast<long double>(B[k][j]); 
            } 
        } 
    }

    return C;
}

// Non naive matixes multiplication
extern "C" long double** funcTask3(float **A, float **B, int N)
{
    long double** C = (long double**)aligned_alloc(64, N * sizeof(long double*));

    long double* Cblock = (long double*)aligned_alloc(64, N * N * sizeof(long double));
    std::memset(Cblock, 0, N * N * sizeof(long double));

    for(int i = 0; i < N; i++) 
        C[i] = Cblock + i * N;

    #pragma omp parallel for num_threads(3) schedule(static)
    for(int i = 0; i < N; i++)
    {
        float      *__restrict__ Ai = A[i];
        long double*__restrict__ Ci = C[i];

        for(int k = 0; k < N; k++)
        {
            long double aik = (long double)Ai[k];
            float *__restrict__ Bk = B[k];

            __builtin_prefetch(&Bk[0], 0, 1);

            int j = 0;
            for(; j + 4 <= N; j += 4)
            {
                Ci[j  ] += aik * (long double)Bk[j  ];
                Ci[j+1] += aik * (long double)Bk[j+1];
                Ci[j+2] += aik * (long double)Bk[j+2];
                Ci[j+3] += aik * (long double)Bk[j+3];
            }
            
            for(; j < N; j++)
                Ci[j] += aik * (long double)Bk[j];
        }
    }

    return C;
}

extern "C" const char* funcLibInfoNickname()
{
    return "Venik"; 
}

extern "C" const char* funcLibInfoVersion()
{
    return "1.0"; 
}
