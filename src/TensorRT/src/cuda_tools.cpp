
/*
 *  系统关于CUDA的功能函数
 */


#include "cuda_tools.hpp"

namespace CUDATools{
    bool check_driver(CUresult e, const char* call, int line, const char *file) {
        if (e != CUDA_SUCCESS) {

            const char* message = nullptr;
            const char* name = nullptr;
            cuGetErrorString(e, &message);
            cuGetErrorName(e, &name);
            INFOE("CUDA Driver error %s # %s, code = %s [ %d ] in file %s:%d", call, message, name, e, file, line);
            return false;
        }
        return true;
    }

    bool check_runtime(cudaError_t e, const char* call, int line, const char *file){
        if (e != cudaSuccess) {
            INFOE("CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
            return false;
        }
        return true;
    }

    bool check_device_id(int device_id){
        int device_count = -1;
        checkCudaRuntime(cudaGetDeviceCount(&device_count));
        if(device_id < 0 || device_id >= device_count){
            INFOE("Invalid device id: %d, count = %d", device_id, device_count);
            return false;
        }
        return true;
    }

    int current_device_id(){
        int device_id = 0;
        checkCudaRuntime(cudaGetDevice(&device_id));
        return device_id;
    }

    // dim3这种数据结构用于表示某种三维数据的维度信息，例如表示CUDA中的grid或block概念的维度信息
    dim3 grid_dims(int numJobs) {
        int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
        return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
    }

    // GPU_BLOCK_THREADS的大小是512，即一个虚拟的block里面最多有512个thread，一个实际最小执行单元SM包含32个thread，则一个block中最后可以有16个SM
    // grid和block的维度最多都可以有三个维度，因此用dim3这种struct来表示
    dim3 block_dims(int numJobs) {
        return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    }

    std::string device_capability(int device_id){
        cudaDeviceProp prop;
        checkCudaRuntime(cudaGetDeviceProperties(&prop, device_id));
        return iLogger::format("%d.%d", prop.major, prop.minor);
    }

    std::string device_name(int device_id){
        cudaDeviceProp prop;
        checkCudaRuntime(cudaGetDeviceProperties(&prop, device_id));
        return prop.name;
    }

    std::string device_description(){

        cudaDeviceProp prop;
        size_t free_mem, total_mem;
        int device_id = 0;

        checkCudaRuntime(cudaGetDevice(&device_id));
        checkCudaRuntime(cudaGetDeviceProperties(&prop, device_id));
        checkCudaRuntime(cudaMemGetInfo(&free_mem, &total_mem));

        return iLogger::format(
            "[ID %d]<%s>[arch %d.%d][GMEM %.2f GB/%.2f GB]",
            device_id, prop.name, prop.major, prop.minor, 
            free_mem / 1024.0f / 1024.0f / 1024.0f,
            total_mem / 1024.0f / 1024.0f / 1024.0f
        );
    }

    // 此类用于切换当前要使用的device
    AutoDevice::AutoDevice(int device_id){
        // 先将当前正在使用的device记录下来
        cudaGetDevice(&old_);
        // 然后再尝试切换使用指定的device
        checkCudaRuntime(cudaSetDevice(device_id));
    }

    // 在此类的析构函数中切换回原来使用的device
    AutoDevice::~AutoDevice(){
        checkCudaRuntime(cudaSetDevice(old_));
    }
}