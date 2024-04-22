#include <iostream>
#include <vector>

struct Voxel {
    int x;
    int y;
    int z;
};

std::vector<Voxel> getInterceptedVoxels(const Voxel& start, const Voxel& end) {
    std::vector<Voxel> interceptedVoxels;

    // Calculate the direction vector
    int dx = end.x - start.x;
    int dy = end.y - start.y;
    int dz = end.z - start.z;

    // Determine the number of steps
    int numSteps = std::max(std::max(std::abs(dx), std::abs(dy)), std::abs(dz));

    std::cout << "numSteps: " << numSteps << std::endl;

    // Calculate the step sizes
    float stepX = static_cast<float>(dx) / numSteps;
    float stepY = static_cast<float>(dy) / numSteps;
    float stepZ = static_cast<float>(dz) / numSteps;

    std::cout << "stepX: " << stepX << std::endl;
    std::cout << "stepY: " << stepY << std::endl;
    std::cout << "stepZ: " << stepZ << std::endl;

    // Iterate through the line
    float currentX = start.x;
    float currentY = start.y;
    float currentZ = start.z;

    for (int i = 0; i <= numSteps; ++i) {
        Voxel voxel;
        voxel.x = static_cast<int>(currentX);
        voxel.y = static_cast<int>(currentY);
        voxel.z = static_cast<int>(currentZ);

        interceptedVoxels.push_back(voxel);

        // Update the current position
        currentX += stepX;
        currentY += stepY;
        currentZ += stepZ;
    }

    return interceptedVoxels;
}

int main() {
    // Example usage
    Voxel start = {0.5f, 0.3f, 0.4f};
    Voxel end = {5, 2, 3};

    std::vector<Voxel> intercepted = getInterceptedVoxels(start, end);

    // Print the intercepted voxels
    for (const auto& voxel : intercepted) {
        std::cout << "Voxel: (" << voxel.x << ", " << voxel.y << ", " << voxel.z << ")\n";
    }

    return 0;
}
