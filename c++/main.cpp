#include <stdio.h>
#include <string>

using namespace std;

int main(int argc, char** argv) {
  int product = 1;
  for (int i = 1; i < argc; i++) {
    product *= (int)atoi(argv[i]);
  }
  printf("product: %d\n", product);
}