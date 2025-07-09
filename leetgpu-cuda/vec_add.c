#include <stdio.h>

#define ARRAY_SIZE 8

void vec_add_host(float* a_h, float* b_h, float* c_h, int n) {
    for (int i = 0; i < n; i++) {
        c_h[i] = a_h[i] + b_h[i];
    }
}

void vec_print_host(float* a_h, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f\n", a_h[i]);
    }
}

int main(int argc, char** argv) {

    float a[ARRAY_SIZE] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float b[ARRAY_SIZE] = {0.0f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f};
    float c[ARRAY_SIZE];

    vec_add_host(a, b, c, ARRAY_SIZE);

    vec_print_host(c, ARRAY_SIZE);
    
    return 0;
}


