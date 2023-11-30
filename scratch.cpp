#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <mpi.h>

struct Point {
    double x, y, z;
    int cluster;
    double minDist;
};

MPI_Datatype createPointType() {
    MPI_Datatype mpiPointType;
    MPI_Datatype types[5] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_DOUBLE};
    int blocklengths[5] = {1, 1, 1, 1, 1};
    MPI_Aint offsets[5];

    offsets[0] = offsetof(Point, x);
    offsets[1] = offsetof(Point, y);
    offsets[2] = offsetof(Point, z);
    offsets[3] = offsetof(Point, cluster);
    offsets[4] = offsetof(Point, minDist);

    MPI_Type_create_struct(5, blocklengths, offsets, types, &mpiPointType);
    MPI_Type_commit(&mpiPointType);

    return mpiPointType;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size < 2) {
        std::cerr << "This program requires at least 2 processes." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_File file;
    MPI_Status status;

    int numPoints;

    // Process 0 opens the file and reads the number of points
    if (world_rank == 0) {
        std::ifstream input("your_csv_file.csv");
        if (!input.is_open()) {
            std::cerr << "Error opening the CSV file." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Count the number of lines in the file
        numPoints = std::count(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>(), '\n');

        input.close();
    }

    // Broadcast the number of points to all processes
    MPI_Bcast(&numPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the number of points per process
    int pointsPerProcess = numPoints / world_size;
    int remainder = numPoints % world_size;

    // Open the file collectively
    MPI_File_open(MPI_COMM_WORLD, "your_csv_file.csv", MPI_MODE_RDONLY, MPI_INFO_NULL, &file);

    // Set the file view for each process
    MPI_File_set_view(file, world_rank * pointsPerProcess * sizeof(Point), createPointType(), MPI_BYTE, "native", MPI_INFO_NULL);

    // Allocate memory for local points
    std::vector<Point> localPoints(pointsPerProcess + (world_rank == world_size - 1 ? remainder : 0));

    // Read data from the file
    MPI_File_read(file, localPoints.data(), pointsPerProcess, createPointType(), &status);

    // Close the file
    MPI_File_close(&file);

    // Scatter the data to all processes
    MPI_Scatter(localPoints.data(), pointsPerProcess, createPointType(),
                localPoints.data(), pointsPerProcess, createPointType(),
                0, MPI_COMM_WORLD);

    // Print the data read by each process
    for (const auto& point : localPoints) {
        std::cout << "Rank " << world_rank << ": x=" << point.x
                  << ", y=" << point.y << ", z=" << point.z
                  << ", cluster=" << point.cluster << ", minDist=" << point.minDist << std::endl;
    }

    MPI_Finalize();
    return 0;
}
