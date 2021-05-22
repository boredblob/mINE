#include <stdio.h>

using namespace std;

const int width = 10;
const int height = 10;

void fillTriangleArr(int w, int h, int arr[][width]) {
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      // arr[y][x] = ((y+x < 10) ? y+x : 18-y-x) % 3;
      arr[y][x] = (y*width+x) % 3;
    }
  }
  return;
}

int main(int argc, char** argv) {
  int triangle[height][width] = {0};
  fillTriangleArr(width, height, triangle);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      printf("%d ", triangle[y][x]);
    }
    printf("\n");
  }
  printf("ello\n");
  return 0;
}