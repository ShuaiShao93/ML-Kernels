#pragma once

#include <iostream>
#include <cmath>
#include <sys/time.h>

void randomize_matrix(float *mat, int N) {
  struct timeval time;
  gettimeofday(&time, NULL);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

bool verify_matrix(float *mat1, float *mat2, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; mat1 + i && mat2 + i && i < N; i++) {
    diff = fabs((double)mat1[i] - (double)mat2[i]);
    if (diff > 1e-2) {
      printf("error. %5.2f,%5.2f,%d\n", mat1[i], mat2[i], i);
      return false;
    }
  }
  return true;
}