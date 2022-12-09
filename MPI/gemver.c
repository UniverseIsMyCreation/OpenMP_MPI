#include "gemver.h"
#define max(x, y) ( (x) > (y) ? (x) : (y) )
#define min(x, y) ( (x) < (y) ? (x) : (y) )

double bench_t_start, bench_t_end;
int use_proc;

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
    int size,
    int start,
    int end,
    double *alpha,
    double *beta,
    double A[ size][n],
    double u1[ size],
    double v1[ n],
    double u2[ size],
    double v2[ n],
    double w[ size],
    double x[ n],
    double y[ size],
    double z[ n])
{
    *alpha = 1.5;
    *beta = 1.2;

    double fn = (double)n;
    for (int i = 0; i < n; i++) {
        v1[i] = ((i+1)/fn)/4.0;
        v2[i] = ((i+1)/fn)/6.0;
        z[i] = ((i+1)/fn)/9.0;
        x[i] = 0.0;
    }

    for (int i = start; i < end; i++) {
        u1[i - start] = i;
        u2[i - start] = ((i+1)/fn)/2.0;
        y[i - start] = ((i+1)/fn)/8.0;
        w[i- start] = 0.0;
        for (int j = 0; j < n; j++) {
            A[i - start][j] = (double) (i*j % n) / n;
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
void kernel_gemver(int n, int size,
    double alpha,
    double beta,
    double A[ size][n],
    double u1[ size],
    double v1[ n],
    double u2[ size],
    double v2[ n],
    double w[ size],
    double x[ n],
    double y[ size],
    double z[ n])
{

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
        }
    }

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < n; ++j) {
            x[j] = x[j] + beta * A[i][j] * y[i];
        }
    }

    /* ---------------------------- MPI ---------------------------- */

    MPI_Status status[1];
    double (*cur)[n]; cur = (double(*)[n])malloc ((n) * sizeof(double));

    int myrank, ranksize;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranksize);

    if (!myrank) {
        for (int j = 0; j < n; j++) {
            x[j] = x[j] + z[j];
        }
        for (int i = 1; i < use_proc; i++) {
            MPI_Recv((*cur), n, MPI_DOUBLE, i, 13, MPI_COMM_WORLD, &status[0]);
            for (int j = 0; j < n; j++) {
                x[j] = x[j] + (*cur)[j];
            }
        }
        //for (int i = 1; i < use_proc; i++) {                           //for use * comment it
        //    MPI_Send(x, n, MPI_DOUBLE, i, 13, MPI_COMM_WORLD);         //for use * comment it
        //}                                                              //for use * comment it
    } else {
        MPI_Send(x, n, MPI_DOUBLE, 0, 13, MPI_COMM_WORLD);
        //MPI_Recv(x, n, MPI_DOUBLE, 0, 13, MPI_COMM_WORLD, &status[0]); //for use * comment it
    }
    // MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD); //*

    /* ---------------------------- MPI ---------------------------- */

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < n; j++) {
            w[i] = w[i] + alpha * A[i][j] * x[j];
        }
    }
}

int main(int argc, char** argv)
{

    /* ---------------------------- MPI ---------------------------- */

    int n = N;
    int myrank, ranksize;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranksize);

    use_proc = ranksize;
    int k = n / use_proc;
    int m = n % use_proc;
    if (k == 0) {
        k = 1;
        m = 0;
        use_proc = n;
    }

    for (int i = 0; i < use_proc; i++) {
        if (myrank == i) {
            int start = min(i, m) + i * k;
            int size = k + (i < m);
            int end = start + size;
            MPI_Request req[1];
            MPI_Status status[1];

            double alpha;
            double beta;
            double (*A)[size][n]; A = (double(*)[size][n])malloc ((size) * (n) * sizeof(double));
            double (*u1)[size]; u1 = (double(*)[size])malloc ((size) * sizeof(double));
            double (*v1)[n]; v1 = (double(*)[n])malloc ((n) * sizeof(double));
            double (*u2)[size]; u2 = (double(*)[size])malloc ((size) * sizeof(double));
            double (*v2)[n]; v2 = (double(*)[n])malloc ((n) * sizeof(double));
            double (*w)[size]; w = (double(*)[size])malloc ((size) * sizeof(double));
            double (*x)[n]; x = (double(*)[n])malloc ((n) * sizeof(double));
            double (*y)[size]; y = (double(*)[size])malloc ((size) * sizeof(double));
            double (*z)[n]; z = (double(*)[n])malloc ((n) * sizeof(double));

            init_array (n, size, start, end, &alpha, &beta,
                       *A,
                       *u1,
                       *v1,
                       *u2,
                       *v2,
                       *w,
                       *x,
                       *y,
                       *z);

            double time_1;
            if (myrank == 0) {
                time_1 = MPI_Wtime();
            }

            kernel_gemver (n, size, alpha, beta,
                           *A,
                           *u1,
                           *v1,
                           *u2,
                           *v2,
                           *w,
                           *x,
                           *y,
                           *z);

            if (myrank == 0) {
                size = k + (0 < m);
                for (int k = 0; k < size; k++) {
                    printf("%lf\n", (*w)[k]);
                }
                double (*cur)[size]; cur = (double(*)[size])malloc ((size) * sizeof(double));
                for (int j = 1; j < use_proc; j++) {
                    size = k + (j < m);
                    MPI_Recv(*cur, size, MPI_DOUBLE, j, 13, MPI_COMM_WORLD, &status[0]);
                    for (int k = 0; k < size; k++) {
                        printf("%lf\n", (*cur)[k]);
                    }
                }
                printf("MPI --- %lf\n", MPI_Wtime() - time_1);
            } else {
                MPI_Send(w, size, MPI_DOUBLE, 0, 13, MPI_COMM_WORLD);
            }

            free((void*)A);
            free((void*)u1);
            free((void*)v1);
            free((void*)u2);
            free((void*)v2);
            free((void*)w);
            free((void*)x);
            free((void*)y);
            free((void*)z);
            break;
        }
    }
    MPI_Finalize();

    /* ---------------------------- MPI ---------------------------- */
    return 0;
}
