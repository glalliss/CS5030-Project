#include <iostream>
#include <fstream>
#include <string>
#include <vector>

bool areFilesSame(const std::vector<std::string>& filenames) {
    // Open all the files
    std::vector<std::ifstream> fileStreams;
    for (const auto& filename : filenames) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return false;
        }
        fileStreams.push_back(std::move(file));
    }
    // Read and compare lines
    std::vector<std::string> lines(filenames.size());
    while (true) {
        for (size_t i = 0; i < fileStreams.size(); ++i) {
            if (!std::getline(fileStreams[i], lines[i])) {
                // End of file reached for one of the files
                if (i > 0) {
                    std::cerr << "Files have different number of lines." << std::endl;
                    return false;
                }
                // All files have been read completely
                return true;
            }
        }
        // Compare lines from all files
        for (size_t i = 1; i < lines.size(); ++i) {
            if (lines[i] != lines[0]) {
                std::cerr << "Files are different at line " << i + 1 << std::endl;
                return false;
            }
        }
    }
}

int main() {
    std::vector<std::string> filenames = {"file1.csv", "file2.csv", "file3.csv", "file4.csv", "file5.csv"};
    if (areFilesSame(filenames)) {
        std::cout << "All files are the same line by line." << std::endl;
    } else {
        std::cout << "Files are not the same line by line." << std::endl;
    }
    return 0;
}
