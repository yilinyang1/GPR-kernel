#include <math.h>
#include <stdio.h>
#include <numeric>

using namespace std;

extern "C" double ee_kernel(double* x, double* xp, double lens, int cols){
    double res, inner;
    double* diff = new double[cols];
    for (int i=0; i < cols; i++){
        diff[i] = pow((x[i] - xp[i]) / lens, 2.0);
    }
    inner = 0.5 * accumulate(diff, diff + cols, 0.0);
    res = exp(-inner);
    delete diff;
    return res;
}

extern "C" double ef_kernel(double* x, double* xp, double lens, int d, int cols){
    double pre, inner, res;
    double* diff = new double[cols];

    pre = 1.0 * (x[d] - xp[d]) / pow(lens, 2.0);
    for (int i=0; i < cols; i++){
        diff[i] = pow((x[i] - xp[i]) / lens, 2.0);
    }
    inner = 0.5 * accumulate(diff, diff + cols, 0.0);
    res = pre * exp(-inner);
    delete diff;
    return res;
}

extern "C" double fe_kernel(double* x, double* xp, double lens, int d, int cols){
    double pre, inner, res;
    double* diff = new double[cols];

    pre = -1.0 * (x[d] - xp[d]) / pow(lens, 2.0);
    for (int i=0; i < cols; i++){
        diff[i] = pow((x[i] - xp[i]) / lens, 2.0);
    }
    inner = 0.5 * accumulate(diff, diff + cols, 0.0);
    res = pre * exp(-inner);
    delete diff;
    return res;
}

extern "C" double ff_kernel(double* x, double* xp, double lens, int d, int dp, int cols){
    double pre, inner, res, delta;
    int length = *(&x + 1) - x;
    double* diff = new double[cols];
    
    if (d == dp){
        delta = 1.0;
    }else{
        delta = 0.0;
    }

    pre = 1.0 / pow(lens, 2.0) * (delta - (x[d] - xp[d]) * (x[dp] - xp[dp]) / pow(lens, 2.0));
    for (int i=0; i < cols; i++){
        diff[i] = pow((x[i] - xp[i]) / lens, 2.0);
    }
    inner = 0.5 * accumulate(diff, diff + cols, 0.0);
    res = pre * exp(-inner);
    delete diff;
    return res;
}


extern "C" int kernel_train(double** X_train, double lens, int n, int dim, double** res){
    // since X_train is a pointer, we don't know its rows and cols, thus we receive n and dim
    // as the rows and cols. Also we fill the results in "res" directly after calculation.

    int config_id, d, id1, id2, d1, d2;
    
    for (int i=0; i < n; i++){
        for (int j=0; j < n; j++){
            if (i <= j){
                res[i][j] = ee_kernel(X_train[i], X_train[j], lens, dim);
            }else{
                res[i][j] = res[j][i];
            }
        }
    }
    
    for (int i=0; i < n; i++){
        for (int j=n; j < n * (1 + dim); j++){
            config_id = j % n;
            d = j / n - 1;
            res[i][j] = ef_kernel(X_train[i], X_train[config_id], lens, d, dim);
        }
    }

    for (int i=n; i < n * (1 + dim); i++){
        for (int j=0; j < n; j++){
            // config_id = i % n;
            // d = i / n - 1;
            // res[i][j] = fe_kernel(X_train[config_id], X_train[j], lens, d, dim);
            res[i][j] = res[j][i];
        }
    }

    for (int i=n; i < n * (1 + dim); i++){
        for (int j=n; j < n * (1 + dim); j++){
            if (i <= j){
                id1 = i % n;
                id2 = j % n;
                d1 = i / n - 1;
                d2 = j / n - 1;
                res[i][j] = ff_kernel(X_train[id1], X_train[id2], lens, d1, d2, dim);
            }else{
                res[i][j] = res[j][i];
            }
        }
    }
    
    return 0;
}
