/* Include benchmark-specific header. */
#include "gemver.h"
#define max(x, y) ( (x) > (y) ? (x) : (y) )
#define min(x, y) ( (x) < (y) ? (x) : (y) )

double bench_t_start, bench_t_end;

static
double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
      printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start()
{
    bench_t_start = rtclock ();
}

void bench_timer_stop()
{
    bench_t_end = rtclock ();
}

void bench_timer_print()
{
    printf ("%0.6lf\n", bench_t_end - bench_t_start);
}


static
void init_array (int n,
    double *alpha,
    double *beta,
    double A[ n][n],
    double u1[ n],
    double v1[ n],
    double u2[ n],
    double v2[ n],
    double w[ n],
    double x[ n],
    double y[ n],
    double z[ n])
{

    *alpha = 1.5;
    *beta = 1.2;

    double fn = (double)n;
    #pragma omp  parallel for
    for (int i = 0; i < n; i++) {
        u1[i] = i;
        u2[i] = ((i+1)/fn)/2.0;
        v1[i] = ((i+1)/fn)/4.0;
        v2[i] = ((i+1)/fn)/6.0;
        y[i] = ((i+1)/fn)/8.0;
        z[i] = ((i+1)/fn)/9.0;
        x[i] = 0.0;
        w[i] = 0.0;
        for (int j = 0; j < n; j++) {
            A[i][j] = (double) (i*j % n) / n;
        }
    }
}

static
void print_array(int n,
    double w[ n])
{
    int i;
    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stderr, "begin dump: %s", "w");
    for (i = 0; i < n; i++) {
        if (i % 20 == 0) fprintf (stderr, "\n");
        fprintf (stderr, "%0.2lf ", w[i]);
    }
    fprintf(stderr, "\nend   dump: %s\n", "w");
    fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static
void kernel_gemver(int n,
    double alpha,
    double beta,
    double A[ n][n],
    double u1[ n],
    double v1[ n],
    double u2[ n],
    double v2[ n],
    double w[ n],
    double x[ n],
    double y[ n],
    double z[ n])
{
    omp_set_num_threads(1);
    if (n < 400) {
        omp_set_num_threads(1);
    }
    #pragma omp parallel
    {
        int BLOCK_SIZE = n / omp_get_num_threads();
        #pragma omp for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
            }
        }

        #pragma omp for
        for (int j1 = 0; j1 < n; j1 += BLOCK_SIZE) {
            for (int i = 0; i < n; ++i) {
                for (int j2 = 0; j2 < min(BLOCK_SIZE, n - j1); ++j2) {
                    x[j1 + j2] = x[j1 + j2] + beta * A[i][j1 + j2] * y[i];
                }
            }
        }

        #pragma omp for
        for (int i = 0; i < n; ++i) {
            x[i] = x[i] + z[i];
        }

        #pragma omp for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                w[i] = w[i] + alpha * A[i][j] * x[j];
            }
        }
    }
}

int main(int argc, char** argv)
{


    int n = atoi(argv[1]);
    double alpha;
    double beta;
    double (*A)[n][n]; A = (double(*)[n][n])malloc ((n) * (n) * sizeof(double));
    double (*u1)[n]; u1 = (double(*)[n])malloc ((n) * sizeof(double));
    double (*v1)[n]; v1 = (double(*)[n])malloc ((n) * sizeof(double));
    double (*u2)[n]; u2 = (double(*)[n])malloc ((n) * sizeof(double));
    double (*v2)[n]; v2 = (double(*)[n])malloc ((n) * sizeof(double));
    double (*w)[n]; w = (double(*)[n])malloc ((n) * sizeof(double));
    double (*x)[n]; x = (double(*)[n])malloc ((n) * sizeof(double));
    double (*y)[n]; y = (double(*)[n])malloc ((n) * sizeof(double));
    double (*z)[n]; z = (double(*)[n])malloc ((n) * sizeof(double));

    init_array (n, &alpha, &beta,
       *A,
       *u1,
       *v1,
       *u2,
       *v2,
       *w,
       *x,
       *y,
       *z);

    bench_timer_start();

    kernel_gemver (n, alpha, beta,
     *A,
     *u1,
     *v1,
     *u2,
     *v2,
     *w,
     *x,
     *y,
     *z);

    bench_timer_stop();

    bench_timer_print();

    // print_array(n, *w);

    free((void*)A);
    free((void*)u1);
    free((void*)v1);
    free((void*)u2);
    free((void*)v2);
    free((void*)w);
    free((void*)x);
    free((void*)y);
    free((void*)z);

    return 0;
}
