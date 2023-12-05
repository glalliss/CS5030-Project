#include <iostream>
#include <fstream>
#include <string>

int main() {
    // Open the serial CSV file
    std::ifstream file1("output_serial.csv");
    if (!file1.is_open()) {
        std::cerr << "Error opening output_serial.csv" << std::endl;
        return 1;
    }
    // Open the shared cpu CSV file
    std::ifstream file2("output_shared_cpu.csv");
    if (!file2.is_open()) {
        std::cerr << "Error opening output_shared_cpu.csv" << std::endl;
        return 1;
    }
    // Open the shared gpu CSV file
    std::ifstream file3("output_shared_gpu.csv");
    if (!file3.is_open()) {
        std::cerr << "Error opening output_shared_gpu.csv" << std::endl;
        return 1;
    }
    // Open the distributed cpu CSV file
    std::ifstream file4("output_distributed_cpu.csv");
    if (!file4.is_open()) {
        std::cerr << "Error opening output_distributed_cpu.csv" << std::endl;
        return 1;
    }
    // Open the distributed gpu CSV file
    std::ifstream file5("output_distributed_gpu.csv");
    if (!file5.is_open()) {
        std::cerr << "Error opening output_distributed_gpu.csv" << std::endl;
        return 1;
    }
    // Read and process data from all files
    std::string line1, line2, line3, line4, line5;
    try
    {
        while (std::getline(file1, line1) && std::getline(file2, line2) && std::getline(file3, line3) && std::getline(file4, line4) && std::getline(file5, line5)) {
            if (line1 != line2 || line1 != line3 || line1 != line4 || line1 != line5)
            {
                std::cout << "Output files are not the same" << std::endl;
                return 1;
            }
            if (line2 != line3 || line2 != line4 || line2 != line5)
            {
                std::cout << "Output files are not the same" << std::endl;
                return 1;
            }
            if (line3 != line4 || line3 != line5)
            {
                std::cout << "Output files are not the same" << std::endl;
                return 1;
            }
            if (line4 != line5)
            {
                std::cout << "Output files are not the same" << std::endl;
                return 1;
            }
        }
    }
    catch(const std::exception& e)
    {
        // std::cerr << e.what() << '\n';
        std::cerr << "Output files are not the same" << std::endl;
        return 1;
    }
    std::cout << "Output files are the same!" << std::endl;
    // Close the files
    file1.close();
    file2.close();
    file3.close();
    file4.close();
    file5.close();
    // Finish successfully
    return 0;
}
