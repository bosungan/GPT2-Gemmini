#ifndef RISCV_MATH_H
#define RISCV_MATH_H

#define FLOAT_EPSILON    1e-7f
#define EXP_LOWER_BOUND  -16.2f
#define EXP_UPPER_BOUND  30.0f

double abs_riscv(double x){
    return (x > 0) ? x : (-1) * x;
}
double exp_riscv(double x) {
    double sum = 1.0f;  // Initialize sum of series
    double term = 1.0f; // First term of the series
    int n = 1;

    if(x < EXP_LOWER_BOUND){
        return 0.0f;
    }

    if(x > EXP_UPPER_BOUND){
        return 3.4e38; //JUST FOR TIME REDUCE
    }

    while (abs_riscv(term) > FLOAT_EPSILON) {
        term *= x / n;
        sum += term;
        n++;

        //printf("exp sum %d: %.16f\n", n, sum);
    }

    return sum;
}

// Function to calculate tanh(x) using Taylor series
double tanh_riscv(double x) {
    double apprx = 1.0f;
    double x_2   = 2 * x;

    apprx = apprx - ((double)2 / (exp_riscv(x_2) + (double)1));
    return apprx;
}


#endif // RISCV_MATH_H
