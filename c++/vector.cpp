#include <stdio.h>

// template<class T, int N> class vector {
  
// }

class point {
  double x;
  double y;
  public:
    point(double xx = 0, double yy = 0): x(xx), y(yy) {
      printf("%d, %d\n", xx, yy);
    }
    point(const point& p): x(p.x), y(p.y) {
    }
    double X() {
      return x;
    }
    double Y() {
      return y;
    }
    const point& zero() {
      x = y = 1;
      return *this;
    }
    const point& operator=(const point& p) {
      if (this != &p) {
        x = p.x;
        y = p.y;
      }
      return *this;
    }
    const point operator+(point& p) {
      return point(x+p.X(), y+p.Y());
    }
    ~point(){
    }
};

int main(int argc, char** argv) {
  point A(1, 3);
  point B(3, 7);
  point S = A + B;
  printf("A: %d, %d\n", A.X(), A.Y());
  printf("B: %d, %d\n", B.X(), B.Y());
  printf("S: %d, %d\n", S.X(), S.Y());
  return 0;
}