import subprocess
import os

opt = ["-O0", "-O2", "-O3", "-O4"]
th = [1, 2, 4, 8, 16, 32, 64, 128, 160]
ds = [4000, 2000, 400, 120, 40]
for o in opt:
    os.system("xlc_r "+o+" -qsmp=omp gemver.c -o gemver")
    for d_s in ds:
        for num_th in th:
            dev = []
            for i in range(20):
                proc = subprocess.Popen(["OMP_NUM_THREADS="+str(num_th)+" ./gemver "+str(d_s)], stdout=subprocess.PIPE, shell=True)
                (out, err) = proc.communicate()
                dev.append(float(out))
            print(sum(dev) / len(dev), end=', ')
        print()
