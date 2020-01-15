#include <iostream>
#include <string>
#include <iomanip>

void linebreak(int lines = 1);

int main(int argc, char* argv[]) {
    // linebreak(2);
    for (int i = 0; i < argc; i++) {
        std::cout << std::hex << std::stoi(argv[i]);
        linebreak();
    }
    // linebreak(2);
    return 0;
}

void linebreak(int lines = 1) {
    for (int i = 0; i < lines; i++) {
        std::cout << "\n";
    }
}