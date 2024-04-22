#include <iostream>
#include <vector>

// struct Voxel {
//     float x;
//     float y;
//     float z;
// };

// std::vector<Voxel> getInterceptedVoxels(const Voxel& start, const Voxel& end) {
//     std::vector<Voxel> interceptedVoxels;

//     // Calculate the direction vector
//     float dx = end.x - start.x;
//     float dy = end.y - start.y;
//     float dz = end.z - start.z;

//     // Determine the number of steps
//     int numSteps = std::max(std::max(std::abs(dx), std::abs(dy)), std::abs(dz));

//     // Calculate the step sizes
//     float stepX = dx / numSteps;
//     float stepY = dy / numSteps;
//     float stepZ = dz / numSteps;

//     // Iterate through the line
//     float currentX = start.x;
//     float currentY = start.y;
//     float currentZ = start.z;

//     for (int i = 0; i <= numSteps; ++i) {
//         Voxel voxel;
//         voxel.x = currentX;
//         voxel.y = currentY;
//         voxel.z = currentZ;

//         interceptedVoxels.push_back(voxel);

//         // Update the current position
//         currentX += stepX;
//         currentY += stepY;
//         currentZ += stepZ;
//     }

//     return interceptedVoxels;
// }

int main() {
    // // Example usage
    // Voxel start = {0.5f, 0.2f, 0.3f};
    // Voxel end = {5.8f, 2.4f, 3.9f};

    // std::vector<Voxel> intercepted = getInterceptedVoxels(start, end);

    // // Print the intercepted voxels
    // for (const auto& voxel : intercepted) {
    //     std::cout << "Voxel: (" << voxel.x << ", " << voxel.y << ", " << voxel.z << ")\n";
    // }

    // return 0;
    int N = 10;
    int iter = 0;

    for (int i = 0; i < N; i++)
    {
        iter++;
        std::cout << i << std::endl;

        if(iter == 20) {break;}

        if (i == 5)
        {
            i--;
            continue;
        }
    }
    return 0;
    
}
