/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TENSORRT_BUFFERS_H
#define TENSORRT_BUFFERS_H

#include <NvInfer.h>
#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>
#include "common.h"
#include "half.h"

namespace tensorrt_buffer {

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
    template<typename AllocFunc, typename FreeFunc>
    class GenericBuffer {
    public:
        //!
        //! \brief Construct an empty buffer.
        //!
        GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
                : mSize(0), mCapacity(0), mType(type), mBuffer(nullptr) {
        }

        //!
        //! \brief Construct a buffer with the specified allocation size in bytes.
        //!
        GenericBuffer(size_t size, nvinfer1::DataType type)
                : mSize(size), mCapacity(size), mType(type) {
            if (!allocFn(&mBuffer, this->nbBytes())) {
                throw std::bad_alloc();
            }
            // assert(mBuffer != nullptr);
			// memset(mBuffer, 0, this->nbBytes());
        }

        GenericBuffer(GenericBuffer &&buf)
                : mSize(buf.mSize), mCapacity(buf.mCapacity), mType(buf.mType), mBuffer(buf.mBuffer) {
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mType = nvinfer1::DataType::kFLOAT;
            buf.mBuffer = nullptr;
        }

        GenericBuffer &operator=(GenericBuffer &&buf) {
            if (this != &buf) {
                freeFn(mBuffer);
                mSize = buf.mSize;
                mCapacity = buf.mCapacity;
                mType = buf.mType;
                mBuffer = buf.mBuffer;
                // Reset buf.
                buf.mSize = 0;
                buf.mCapacity = 0;
                buf.mBuffer = nullptr;
            }
            return *this;
        }

        //!
        //! \brief Returns pointer to underlying array.
        //!
        void *data() {
            return mBuffer;
        }

        //!
        //! \brief Returns pointer to underlying array.
        //!
        const void *data() const {
            return mBuffer;
        }

        //!
        //! \brief Returns the size (in number of elements) of the buffer.
        //!
        size_t size() const {
            return mSize;
        }

        //!
        //! \brief Returns the size (in bytes) of the buffer.
        //!
        size_t nbBytes() const {
            return this->size() * tensorrt_common::getElementSize(mType);
        }

        //!
        //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
        //!
        void resize(size_t newSize) {
            mSize = newSize;
            if (mCapacity < newSize) {
                freeFn(mBuffer);
                if (!allocFn(&mBuffer, this->nbBytes())) {
                    throw std::bad_alloc{};
                }
                mCapacity = newSize;
            }
        }

        //!
        //! \brief Overload of resize that accepts Dims
        //!
        void resize(const nvinfer1::Dims &dims) {
            return this->resize(tensorrt_common::volume(dims));
        }

        ~GenericBuffer() {
            freeFn(mBuffer);
        }

    private:
        size_t mSize{0}, mCapacity{0};
        nvinfer1::DataType mType;
        void *mBuffer;
        AllocFunc allocFn;
        FreeFunc freeFn;
    };

    class DeviceAllocator {
    public:
        bool operator()(void **ptr, size_t size) const {
            return cudaMalloc(ptr, size) == cudaSuccess;
        }
    };

    class DeviceFree {
    public:
        void operator()(void *ptr) const {
            cudaFree(ptr);
        }
    };

    class HostAllocator {
    public:
        bool operator()(void **ptr, size_t size) const 
        {
            *ptr = std::malloc(size);
            return *ptr != nullptr;
            //if(*ptr == NULL) std::cout << "malloc memory for raw data in host failed!" << std::endl;

            // 使用不可分页内存？
            //return cudaMallocHost(ptr, size) == cudaSuccess;
            //checkCudaRuntime(cudaMallocHost(ptr, size));
        }
    };

    class HostFree {
    public:
        void operator()(void *ptr) const {
            std::free(ptr);
            //checkCudaRuntime(cudaFreeHost(ptr));
        }
    };

    using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
    using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

//!
//! \brief  The ManagedBuffer class groups together a pair of corresponding device and host buffers.
//!
    // 一个ManagedBuffer对象中包含一段分配好的host内存和gpu内存，就类似于在host和gou上各分配一个Tensor对象的内存
    // 一个ManagedBuffer对象表示一个binding（不一定是网络的直接输入或输出变量，只要是网络推理所需要的变量即可，如预处理的仿射矩阵）
    class ManagedBuffer {
    public:
        DeviceBuffer deviceBuffer;
        HostBuffer hostBuffer;
    };

//!
//! \brief  The BufferManager class handles host and device buffer allocation and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The BufferManager class is meant to be
//!          used to simplify buffer management and any interactions between buffers and the engine.
//!
    // 一个BufferManager对象中包含多个ManagedBuffer对象
    class BufferManager {
    public:
        static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

        //!
        //! \brief Create a BufferManager for handling buffer interactions with engine.
        //!
        BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int batchSize = 0,
                      const nvinfer1::IExecutionContext *context = nullptr, 
                      const int num_raw_device = 0, const int num_post_device = 0,
                      const int* Size_raw_post_cpy = nullptr)
                : mEngine(engine), mBatchSize(batchSize), 
                  mnum_raw_device(num_raw_device),
                  mnum_post_device(num_post_device){

            
            // for debug
            bool succ = true;
            // Full Dims implies no batch size.
            assert(engine->hasImplicitBatchDimension() || mBatchSize == 0);
            assert(mnum_raw_device>=0 && 'number of raw image on GPU can not be < 0');
            assert(mnum_post_device>=0 && 'number of postprocessed variable on GPU can not be < 0');

            mnum_raw_post_device = mnum_raw_device + mnum_post_device;

            // Create host and device buffers
            // 如果是在gpu上进行预处理，则只需要开辟用于存储预处理结果的gpu内存和用于存储直接推理结果的gpu内存；
            // ICudaEngine和ExecutionContex中的大部分以index为参数的函数在Tensorrt8.5.0开始都被弃用了，改为以TensorName为参数！！
            for (int i = 0; i < mEngine->getNbIOTensors(); i++) 
            {
                auto const binding_name = mEngine->getIOTensorName(i);
                mNames[binding_name] = i;
                auto dims = context ? context->getTensorShape(binding_name) : mEngine->getTensorShape(binding_name);
                size_t vol = context || !mBatchSize ? 1 : static_cast<size_t>(mBatchSize);
                // 默认网络的输入输出变量都是FP32类型的
                nvinfer1::DataType type = mEngine->getTensorDataType(binding_name);

                // for debug, and type is kFLOAT!
                // bool type_float = (type == nvinfer1::DataType::kFLOAT);
                //printf("datatype is kfloat? %d\n", type_float);

                int vecDim = mEngine->getTensorVectorizedDim(binding_name);
                if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
                {
                    int scalarsPerVec = mEngine->getTensorComponentsPerElement(binding_name);
                    dims.d[vecDim] = tensorrt_common::divUp(dims.d[vecDim], scalarsPerVec);
                    vol *= scalarsPerVec;
                }
                vol *= tensorrt_common::volume(dims);

                std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};

                manBuf->deviceBuffer = DeviceBuffer(vol, type);

                if (!mnum_raw_post_device){
                    manBuf->hostBuffer = HostBuffer(vol, type);
                }
                mDeviceBindings.emplace_back(manBuf->deviceBuffer.data());
                mManagedBuffers.emplace_back(std::move(manBuf));

                if (mEngine->getTensorIOMode(binding_name) == nvinfer1::TensorIOMode::kINPUT){
                    mnum_input_binding++;
                }
                
            }
            //printf("allocate for bindings device succeed? %d\n", succ);
            // int num_output = mEngine->getNbBindings() - mnum_input_binding;
            // added by CRS 2024.02
            if (mnum_raw_post_device != 0)
            {
                assert(Size_raw_post_cpy != nullptr && "Please provide the array of bytesize of variables");
                for(int i = 0; i < mnum_raw_post_device; i++)
                {
                    std::unique_ptr<ManagedBuffer> mBuffer_raw_post_(new ManagedBuffer());
                    
                    // kINT32 - signed 32-bit int format。  其实给仿射矩阵的内存分配时可以使用kFLOAT,这只是用来计算bytes的而已
                    // tensorrt中不支持UINT8,即unsigned int8数据类型，这里只是借用kINT8来计算所需的内存字节数
                    // kUINT8是unsigned int8。这里分配内存以kINT8的基本单位，是因为原图像的每个通道就是UINT8类型。
                    // 后续强制转换void*到float*、int32_t*（针对yolov8seg的缩放填充情况）和uint8_t*（原图像数据）即可！
                    mBuffer_raw_post_->deviceBuffer = DeviceBuffer(Size_raw_post_cpy[i], nvinfer1::DataType::kUINT8);
                    std::cout << "size of raw post " << i << ": " << Size_raw_post_cpy[i] << std::endl;
                    mBuffer_raw_post_->hostBuffer   = HostBuffer(Size_raw_post_cpy[i], nvinfer1::DataType::kUINT8);
                    std::cout << "size of raw post " << i << ": " << Size_raw_post_cpy[i] << std::endl;
                    // 当vector中存储的是unique_ptr时，需要转移所有权的，因为该内存最多可以被一个ptr所占有
                    mBuffer_raw_post_vec.push_back(std::move(mBuffer_raw_post_));
                    // assert(mBuffer_raw_post_vec[i]->hostBuffer.data() != nullptr && "Something woring with host buffer");
                }
            }
            //printf("allocate for vars device succeed? %d\n", succ);
        }
        
        //!
        //! \brief Returns a vector of device buffers that you can use directly as
        //!        bindings for the execute and enqueue methods of IExecutionContext.
        //!
        std::vector<void *> &getDeviceBindings() {
            return mDeviceBindings;
        }

        //!
        //! \brief Returns a vector of device buffers.
        //!
        const std::vector<void *> &getDeviceBindings() const {
            return mDeviceBindings;
        }

        //!
        //! \brief Returns the device buffer corresponding to tensorName.
        //!        Returns nullptr if no such tensor can be found.
        //!
        void *getBindingDeviceBuffer(const std::string &tensorName) const {
            return getBuffer(false, tensorName, -1);
        }

        //!
        //! \brief Returns the host buffer corresponding to tensorName.
        //!        Returns nullptr if no such tensor can be found.
        //!
        void *getBindingHostBuffer(const std::string &tensorName) const {
            return getBuffer(true, tensorName, -1);
        }

        // In C++, "" is empty string, '' is empty char!
        void *getVarDeviceBuffer(const int index) const{
            return getBuffer(false, "", index);
        }

        void *getVarHostBuffer(const int index) const{
            return getBuffer(true, "", index);
        }

        //!
        //! \brief Returns the size of the host and device buffers that correspond to tensorName.
        //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
        //!
        size_t size(const std::string &tensorName) const {
            auto record = mNames.find(tensorName);
            if (record == mNames.end())
                return kINVALID_SIZE_VALUE;
            return mManagedBuffers[record->second]->hostBuffer.nbBytes();
        }

        //!
        //! \brief Dump host buffer with specified tensorName to ostream.
        //!        Prints error message to std::ostream if no such tensor can be found.
        //!
        void dumpBuffer(std::ostream &os, const std::string &tensorName) {
            int index = mEngine->getBindingIndex(tensorName.c_str());
            if (index == -1) {
                os << "Invalid tensor name" << std::endl;
                return;
            }
            void *buf = mManagedBuffers[index]->hostBuffer.data();
            size_t bufSize = mManagedBuffers[index]->hostBuffer.nbBytes();
            nvinfer1::Dims bufDims = mEngine->getBindingDimensions(index);
            size_t rowCount = static_cast<size_t>(bufDims.nbDims > 0 ? bufDims.d[bufDims.nbDims - 1] : mBatchSize);
            int leadDim = mBatchSize;
            int *trailDims = bufDims.d;
            int nbDims = bufDims.nbDims;

            // Fix explicit Dimension networks
            if (!leadDim && nbDims > 0) {
                leadDim = bufDims.d[0];
                ++trailDims;
                --nbDims;
            }

            os << "[" << leadDim;
            for (int i = 0; i < nbDims; i++)
                os << ", " << trailDims[i];
            os << "]" << std::endl;
            switch (mEngine->getBindingDataType(index)) {
                case nvinfer1::DataType::kINT32:
                    print<int32_t>(os, buf, bufSize, rowCount);
                    break;
                case nvinfer1::DataType::kFLOAT:
                    print<float>(os, buf, bufSize, rowCount);
                    break;
                case nvinfer1::DataType::kHALF:
                    print<half_float::half>(os, buf, bufSize, rowCount);
                    break;
                case nvinfer1::DataType::kINT8:
                    assert(0 && "Int8 network-level input and output is not supported");
                    // break;
                case nvinfer1::DataType::kBOOL:
                    assert(0 && "Bool network-level input and output are not supported");
                    // break;
            }
        }

        //!
        //! \brief Templated print function that dumps buffers of arbitrary type to std::ostream.
        //!        rowCount parameter controls how many elements are on each line.
        //!        A rowCount of 1 means that there is only 1 element on each line.
        //!
        template<typename T>
        void print(std::ostream &os, void *buf, size_t bufSize, size_t rowCount) {
            assert(rowCount != 0);
            assert(bufSize % sizeof(T) == 0);
            T *typedBuf = static_cast<T *>(buf);
            size_t numItems = bufSize / sizeof(T);
            for (int i = 0; i < static_cast<int>(numItems); i++) {
                // Handle rowCount == 1 case
                if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
                    os << typedBuf[i] << std::endl;
                else if (rowCount == 1)
                    os << typedBuf[i];
                    // Handle rowCount > 1 case
                else if (i % rowCount == 0)
                    os << typedBuf[i];
                else if (i % rowCount == rowCount - 1)
                    os << " " << typedBuf[i] << std::endl;
                else
                    os << " " << typedBuf[i];
            }
        }
        
        //!! edited by CRS 2024.02. 
        //!! add an arg "byteSize_" to define the number of bytes (src image + affine matrix for input) that need to be copied between cpu and gpu
        
        //!
        //! \brief Copy the contents of input host buffers to input device buffers synchronously.
        //!
        void copyInputToDevice(const bool to_var, size_t byteSize_ = 0, 
                                const std::vector<std::string> &bindings_cpy = std::vector<std::string>()) {
            if(to_var) 
                assert(byteSize_ && "Please give a specific byteSize to copy a var");
            else 
            assert(!bindings_cpy.empty() && "Please specify which input bindings need to be copied");
            memcpyBuffers(to_var, true, false, byteSize_, bindings_cpy, false);
        }
        
        //!
        //! \brief Copy the contents of output device buffers to output host buffers synchronously.
        //!
        void copyOutputToHost(const bool from_var, size_t byteSize_ = 0, 
                            const std::vector<std::string> &bindings_cpy = std::vector<std::string>()) {
            if(from_var)
                assert(byteSize_ && "Please give a specific byteSize to copy a var");
            else
                assert(!bindings_cpy.empty() && "Please specify which ouput bindings need to be copied");
            memcpyBuffers(from_var, false, true, byteSize_, bindings_cpy, false);
        }

        //!
        //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
        //!
        void copyInputToDeviceAsync(const bool to_var, size_t byteSize_ = 0, 
                                    const std::vector<std::string> &bindings_cpy = std::vector<std::string>(), 
                                    const cudaStream_t &stream = 0) 
        {
            if (to_var)
                assert(byteSize_ && "Please give a specific byteSize to copy a var");
            else
                assert(!bindings_cpy.empty() && "Please specify which input bindings need to be copied");
            memcpyBuffers(to_var, true, false, byteSize_, bindings_cpy, true, stream);
        }

        //!
        //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
        //!
        void copyOutputToHostAsync(const bool from_var, size_t byteSize_ = 0, 
                                    const std::vector<std::string> &bindings_cpy = std::vector<std::string>(), 
                                    const cudaStream_t &stream = 0) {
            if(from_var) 
                assert(byteSize_ && "Please give a specific byteSize to copy a var");
            else 
                assert(!bindings_cpy.empty() && "Please specify which ouput bindings need to be copied");
            memcpyBuffers(from_var, false, true, byteSize_, bindings_cpy, true, stream);
        }

        ~BufferManager() = default;

    private:
        // edited by CRS 2024.02. 用于host和device之间数据传输的内存对象是mBuffer_preproc_vec(输入）和 mBuffer_postproc_vec（输出）
        void* getBuffer(const bool isHost, const std::string &tensorName, const int ind) const 
        {
            auto record = mNames.find(tensorName);
            if (record == mNames.end() || tensorName.empty())
            {
                if (!mnum_raw_post_device){
                    printf("Please provide a correct name of engine binding! line 479 in buffers.h\n");
                    return nullptr;
                }
                if (ind < 0 || ind > mnum_raw_post_device){
                    printf("Incorrect index for raw images or post-processed variables! line 483 in buffers.h\n");
                    return nullptr;
                }
                printf("Hello I'm here!\n");
                assert(mBuffer_raw_post_vec[ind]->hostBuffer.data() != NULL && "Weired!");
                if(isHost)
                    return mBuffer_raw_post_vec[ind]->hostBuffer.data();
                else
                    return mBuffer_raw_post_vec[ind]->deviceBuffer.data();
                // return (isHost ? mBuffer_raw_post_vec[ind]->hostBuffer.data() : mBuffer_raw_post_vec[ind]->deviceBuffer.data());
            }
            else
            {
                if (mnum_raw_post_device && isHost){
                    printf("For raw images or post-processed variables in host, please provide an index! line 492 in buffers.h");
                    return nullptr;
                }
                return (isHost ? mManagedBuffers[record->second]->hostBuffer.data() : mManagedBuffers[record->second]->deviceBuffer.data());
            }
        }

        bool tenosrIsInput(const std::string& tensorName) const
        {
            return mEngine->getTensorIOMode(tensorName.c_str()) == nvinfer1::TensorIOMode::kINPUT;
        }

        // edited by CRS 2024.02. 
        // add arg "from_to_var" to define which ones to copy: var or bindings
        // add arg "byteSize_" to define the specific byteSize of var that need to be copy between cpu and gpu
        void
        memcpyBuffers(const bool from_to_var, const bool copyInput, const bool deviceToHost, const size_t byteSize_ = 0, 
                        const std::vector<std::string> &bindings_cpy = std::vector<std::string>(),
                        const bool async = false, const cudaStream_t &stream = 0) {
            assert(copyInput != deviceToHost && "Value of copyInput and deviceToHost can not be same");
            const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

            if (from_to_var){
                assert(mnum_raw_post_device == 0 && "There is not input or output variable except engine bindings");
                assert(byteSize_ == 0 && "To copy var from or to GPU, a specific byteSize should be given");
                for (int i = 0; i < mnum_raw_post_device; i++) {
                    if ((copyInput && i < mnum_raw_device) || (!copyInput && i >= mnum_raw_device)) {
                        void *dstPtr
                            = deviceToHost ? mBuffer_raw_post_vec[i]->hostBuffer.data()
                                        : mBuffer_raw_post_vec[i]->deviceBuffer.data();
                        const void *srcPtr
                            = deviceToHost ? mBuffer_raw_post_vec[i]->deviceBuffer.data()
                                        : mBuffer_raw_post_vec[i]->hostBuffer.data();
                        // host-device之间的数据传输操作可以是异步的（即数据传输操作不会阻塞host线程上的其他操作）
                        // 是否异步与stream是显式流还是默认流无关？
                        if (async)
                            CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize_, memcpyType, stream));
                        else
                            CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize_, memcpyType));
                    }
                }
            }
            else{
                std::vector<std::string> v = bindings_cpy;
                for (auto const& n : v) {
                    if (((copyInput && tenosrIsInput(n)) || (!copyInput && !tenosrIsInput(n)))
                        && (mNames.find(n) != mNames.end())){
                        void *dstPtr
                            = deviceToHost ? mManagedBuffers[mNames[n]]->hostBuffer.data()
                                        : mManagedBuffers[mNames[n]]->deviceBuffer.data();
                        const void *srcPtr
                            = deviceToHost ? mManagedBuffers[mNames[n]]->deviceBuffer.data()
                                        : mManagedBuffers[mNames[n]]->hostBuffer.data();
                        const size_t byteSize = mManagedBuffers[mNames[n]]->hostBuffer.nbBytes();
                        if (async)
                            CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
                        else
                            CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
                    }
                }
                
            }
            
        }

        std::shared_ptr<nvinfer1::ICudaEngine> mEngine;              //!< The pointer to the engine
        int mBatchSize;                                              //!< The batch size for legacy networks, 0 otherwise.
        std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers; //!< The vector of pointers to managed buffers
        std::vector<void *> mDeviceBindings;                          //!< The vector of device buffers needed for engine execution
        std::unordered_map<std::string, int32_t> mNames;              //!< The map of tensor name and index pairs. Added for TRT Version>= 8.5
        
        // added by CRS 2024.02. 
        int mnum_input_binding = 0;
        int mnum_raw_post_device; //!< The number of raw image and postprocessed variable that is copied between GPU and CPU
        int mnum_raw_device;
        int mnum_post_device;
        std::vector<std::unique_ptr<ManagedBuffer>> mBuffer_raw_post_vec;
        
    };

} // namespace tensorrt_buffer

#endif // TENSORRT_BUFFERS_H
