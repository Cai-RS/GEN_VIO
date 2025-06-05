
#include <cuda_tools.hpp>

#include "cuda_runtime_api.h"
// #include <CMakeCUDACompilerId.cpp1.ii>


#define MAX_DEPTH_1 16
#define INSERTION_SORT 8
const double EPS = 0.0000001;

#define GPU_CHECK(ans)                                                         \
    { GPUAssert((ans), __FILE__, __LINE__); }
inline void GPUAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
};

__device__ __host__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

// CUDA语言编写的函数前面需要写 __host__，__device__, __global__等标识符
// 其中，__host__表明此函数只能由CPU调用和执行，这与普通函数无异；
// __global__标注的函数为核函数，需要由CPU调用（其实也可以从device调用，如动态并行时），在GPU上执行，它会通过某种方式把函数的参数共享到GPU上，然后交由GPU上不同的块并行地执行函数所定义的任务。
// 同时__global__函数的执行是异步的（相对于当前的host线程）。因为其任务实际是由GPU执行，一旦开始后就会返回控制权。
// __device__表明此函数只能由GPU调用和执行，因此该函数一般就是在核函数内被调用（因为核函数就是运行在GPU上，只能通过此方式才能在GPU上调用__device__函数）。具体地，__device__函数是由device上的每个线程所执行。
// 之所以要有__device__类型的函数，是因为__global__函数中可能有些功能需要单独取出作为一个子函数，由于__global__函数是在GPU上执行，因此该子函数必须得是被GPU调用的类型，因此就需要__device__类型函数（因为如果该子函数是被CPU调用，则意味着调用__global__函数时要阻塞host线程！）
// https://whatghost.github.io/2020/03/16/cuda-func-excu-space-specifiers/index.html 
namespace YoloV8
{

    const int NUM_BOX_ELEMENT = 8;      // left, top, right, bottom, confidence, class, 
                                        // keepflag, row_index(output)

    static __global__ void decode_kernel_v8_Seg(float *predict, int num_bboxes, int num_classes, float confidence_threshold, float* invert_affine_matrix, float* parray, int MAX_BOXES_DECODED){
        
        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= num_bboxes) return;

        float* pitem            = predict + (4 + num_classes + 32) * position;
        float* class_confidence = pitem + 4;
        float confidence        = *class_confidence++;
        int label               = 0;
        for(int i = 1; i < num_classes; ++i, ++class_confidence){
            if(*class_confidence > confidence){
                confidence = *class_confidence;
                label      = i;
            }
        }

        if(confidence < confidence_threshold)
            return;
        
        int index = (int)*parray;
        // 这句其实不需要有，跟上面position的判断重复了
        if(index >= MAX_BOXES_DECODED)
            return;
        // CUDA中的原子操作，具有线程保护的功能，atomicAdd(float* old, int new)，计算(*old + new)并将结果保存在原处，而返回的是*old的值
        index = atomicAdd(parray, 1);

        float cx         = *pitem++;
        float cy         = *pitem++;
        float width      = *pitem++;
        float height     = *pitem++;
        float left   = cx - width  * 0.5f;
        float top    = cy - height * 0.5f;
        float right  = cx + width  * 0.5f;
        float bottom = cy + height * 0.5f;
        affine_project(invert_affine_matrix, left,  top,    &left,  &top);
        affine_project(invert_affine_matrix, right, bottom, &right, &bottom);
        // 前两个数用来保存此处初步解码得到的数量 和 之后经过NMS后得到的数量
        float *pout_item = parray + 2 + index * NUM_BOX_ELEMENT;
        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = confidence;
        *pout_item++ = label;
        *pout_item++ = 1;  // 1 = keep, 0 = ignore
        *pout_item++ = position;  // row_index  表示该解码出的bbox位于网络预测结果的第几个（一行表示一个）
    }

    // 选择排序，倒序
    __device__ void selection_sort(float *data, int left, int right) 
    {
        for (int i = left; i <= right; ++i) {
            float max_val = data[i*8+4];
            int max_idx = i;

            // Find the largest value in the range [left, right].
            for (int j = i + 1; j <= right; ++j) 
            {
                float val_j = data[j*8+4];

                // if (val_j-max_val>EPS) 
                if (val_j > max_val) 
                {
                    max_idx = j;
                    max_val = val_j;
                }
            }

            // Swap the values.
            float temp_value = 0;
            if (i != max_idx) 
            {
                for (int k=0; k<8; k++)
                {
                    temp_value = data[max_idx*8+k];
                    data[max_idx*8+k] = data[i*8+k];
                    data[i*8+k] = temp_value;
                }
            }
        }
    }

    // 使用CUDA的动态并行来实现快速排序，倒序
    static __global__ void cdp_simple_quicksort(float *data, int left, float* bbox_num_ptr, int set_right, int depth) 
    {
        // 当递归的深度大于设定的MAX_DEPTH或者待排序的数组长度小于设定的阈值，直接调用简单选择排序
        int right = set_right;
        if (depth == 0)
        {
            // 实际解码得到的bbox可能并没有那么多
            if ((int(*bbox_num_ptr)) < (set_right + 1)){
                right = int(*bbox_num_ptr) -1;
            }
        }
        if (depth >= MAX_DEPTH_1 || right - left <= INSERTION_SORT) {
            // 凡是在kernel函数中调用的自定义函数，都是__device__函数，即必须由device调用与执行
            selection_sort(data, left, right);
            return;
        }
        // 找到当前区间的左端和右端bbox中的conf元素所在内存
        float *left_ptr  = data + left * 8 + 4;
        float *right_ptr = data + right * 8 + 4;
        float pivot = data[(left + right) / 2 * 8 + 4];
        // partition
        while (left_ptr <= right_ptr) {
            float left_val  = *left_ptr;
            float right_val = *right_ptr;

            // 浮点数的比较貌似不用这么麻烦吧？
            // while (left_val - pivot > EPS) //找到左侧第一个小于(pivot+EPS)的
            while (left_val > pivot) { // 找到左侧第一个小于pivot的
                left_ptr+=8;
                left_val = *left_ptr;
            }

            // while (pivot - right_val > EPS) // 找到右侧第一个大于（pivot-EPS）的
            while (pivot > right_val) { // 找到右侧第一个大于pivot的
                right_ptr-=8;
                right_val = *right_ptr;
            }

            // do swap
            if (left_ptr <= right_ptr) 
            {
                float left_value = 0.0;
                float* left_value_ptr = left_ptr-4;
                for (int i=0; i<8; i++){
                    left_value       = *left_value_ptr++;
                    *(left_ptr-4+i)  = *(right_ptr-4+i);
                    *(right_ptr-4+i) = left_value;
                }
                left_ptr+=8;
                right_ptr-=8;
            }
        }

        // recursive
        int n_right = (right_ptr - data -4) / 8;
        int n_left  = (left_ptr - data -4) / 8;
        // Launch a new block to sort the the left part.
        if (left < (right_ptr - data -4)/8) {
            cudaStream_t l_stream;
            // 设置非阻塞流
            cudaStreamCreateWithFlags(&l_stream, cudaStreamNonBlocking);
            // 这里说明了__global__函数不仅可以被CPU调用，也可以从device中调用！
            cdp_simple_quicksort<<<1, 1, 0, l_stream>>>(data, left, bbox_num_ptr, n_right, depth + 1);
            cudaStreamDestroy(l_stream);
        }

        // Launch a new block to sort the the right part.
        if ((left_ptr - data -4)/8 < right) {
            cudaStream_t r_stream;
            // 设置非阻塞流
            cudaStreamCreateWithFlags(&r_stream, cudaStreamNonBlocking);
            cdp_simple_quicksort<<<1, 1, 0, r_stream>>>(data, n_left, bbox_num_ptr, right, depth + 1);
            cudaStreamDestroy(r_stream);
        }
    }

    static __device__ float box_iou(
        float aleft, float atop, float aright, float abottom, 
        float bleft, float btop, float bright, float bbottom)
    {

        float cleft 	= max(aleft, bleft);
        float ctop 		= max(atop, btop);
        float cright 	= min(aright, bright);
        float cbottom 	= min(abottom, bbottom);
        
        float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
        if(c_area == 0.0f)
            return 0.0f;
        
        float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
        float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
        return c_area / (a_area + b_area - c_area);
    }

    static __global__ void nms_kernel(float* bboxes, int max_objects, float threshold)
    {
        int position = (blockDim.x * blockIdx.x + threadIdx.x);
        //int count = min((int)*bboxes, max_objects);
        // NMS得到的bbox数量已经足够了，不再进行NMS后续比较。否则继续执行NMS直到所有bbox都与其之前的bbox比较一遍
        // if ((int)*(bboxes+1) >= count) return;

        if (position >= max_objects) return;
        
        // left, top, right, bottom, confidence, class, keepflag，row_index
        float* pcurrent = bboxes + 2 + position * NUM_BOX_ELEMENT;
        // 因为bbox是已经根据置信度从大到小排序了，因此这里只需要考虑在当前bbox之前的bbox即可
        for(int i = 0; i < position; ++i)
        {
            // 要注意，不能改变别的bbox的数据，只能访问，否则会对别的解码线程造成干扰！！这里其实线程不安全！
            float* pitem = bboxes + 2 + i * NUM_BOX_ELEMENT;
            if(pcurrent[5] != pitem[5]) continue;
            // 其实这个if一定会成立
            if(pitem[4] >= pcurrent[4])
            {
                //if(pitem[4] == pcurrent[4])
                    //continue;

                float iou = box_iou(
                    pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                    pitem[0],    pitem[1],    pitem[2],    pitem[3]
                );

                if(iou > threshold){
                    pcurrent[6] = 0;  // 1=keep, 0=ignore
                    return;
                }
            }
        }

        atomicAdd(bboxes+1, 1);
    } 
    
    static __global__ void decode_single_mask_kernel(int left, int top, float *mask_weights, float *mask_predict, int seg_map_width, int seg_map_height,
                                                    unsigned char *seg_map_device, int mask_dim, int out_width, int out_height) {

        // mask_predict to mask_out
        // mask_weights @ mask_predict
        int dx = blockDim.x * blockIdx.x + threadIdx.x;
        int dy = blockDim.y * blockIdx.y + threadIdx.y;
        if (dx >= out_width || dy >= out_height) return;

        int sx = left + dx;
        int sy = top + dy;
        // 仿射投影到mask图之后超出了图像范围
        if (sx < 0 || sx >= seg_map_width || sy < 0 || sy >= seg_map_height) {
            //seg_map_device[sy * seg_map_width + sx] = 0;
            return;
        }

        float cumprod = 0;
        for (int ic = 0; ic < mask_dim; ++ic) {
            float cval = mask_predict[(ic * seg_map_height + sy) * seg_map_width+ sx];
            float wval = mask_weights[ic];
            cumprod += cval * wval;
        }

        float alpha = 1.0f / (1.0f + exp(-cumprod));  // sigmoid
        // mask解码后的存储是行优先
        unsigned char prob = (unsigned char)(alpha * 255);
        seg_map_device[sy * seg_map_width + sx] = prob;

    }
    
    static __global__ void build_full_seg_mask_kernel(int left, int top, unsigned char *full_seg_map_device, 
                            int full_seg_map_width, int full_seg_map_height, unsigned char *small_seg_map_device,
                            int small_seg_map_width, int small_seg_map_height, int full_map_bbox_width, 
                            int full_map_bbox_height, unsigned char cls_label, unsigned char obj_id, unsigned char is_valid) 
    {
        int dx = blockDim.x * blockIdx.x + threadIdx.x;
        int dy = blockDim.y * blockIdx.y + threadIdx.y;
        if (dx >= full_map_bbox_width || dy >= full_map_bbox_height) 
        {
            return;
        }
        int sx = left + dx;
        int sy = top + dy;
        if (sx < 0 || sx >= full_seg_map_width || sy < 0 || sy >= full_seg_map_height) 
        {
            return;
        }

        float prob = 0.0;
        float sx_small   = sx * (small_seg_map_width / (float)full_seg_map_width);
        float sy_small   = sy * (small_seg_map_height / (float)full_seg_map_height);
        
        int sx_small_low = int(sx_small);
        int sy_small_low = int(sy_small);
        int sx_small_up  = sx_small_low + 1;
        int sy_small_up  = sy_small_low + 1;

        float d_x_low = sx_small - sx_small_low;
        float d_y_low = sy_small - sy_small_low;

        if (sx_small_up < small_seg_map_width && sy_small_up < small_seg_map_height){
            prob += small_seg_map_device[sy_small_low * small_seg_map_width + sx_small_low] * (1-d_x_low) * (1-d_y_low);
            prob += small_seg_map_device[sy_small_low * small_seg_map_width + sx_small_up] * d_x_low * (1-d_y_low);
            prob += small_seg_map_device[sy_small_up * small_seg_map_width + sx_small_low] * (1-d_x_low) * d_y_low;
            prob += small_seg_map_device[sy_small_up * small_seg_map_width + sx_small_up] * d_x_low * d_y_low;
            // prob += small_seg_map_device[sy_small_low * small_seg_map_width + sx_small_low] * 1.0;
        }
        else if(sy_small_up < small_seg_map_height){
            // prob += small_seg_map_device[sy_small_low * small_seg_map_width + sx_small_low] * 1.0;

            prob += small_seg_map_device[sy_small_low * small_seg_map_width + sx_small_low] * (1-d_y_low);
            prob += small_seg_map_device[sy_small_up * small_seg_map_width + sx_small_low] * d_y_low;
        }
        else if(sx_small_up < small_seg_map_width){
            // prob += small_seg_map_device[sy_small_low * small_seg_map_width + sx_small_low] * 1.0;
            prob += small_seg_map_device[sy_small_low * small_seg_map_width + sx_small_low] * (1-d_x_low);
            prob += small_seg_map_device[sy_small_low * small_seg_map_width + sx_small_up] * d_x_low;
        }
        else{
            prob += small_seg_map_device[sy_small_low * small_seg_map_width + sx_small_low] * 1.0;
        }

        int drift_size_prob = full_seg_map_height * full_seg_map_width * 2;
        int index_prob      = drift_size_prob + sy * full_seg_map_width + sx;
        int index_mask_obj   = full_seg_map_height * full_seg_map_width * 3 + sy * full_seg_map_width + sx;
        int index_mask_bg  = full_seg_map_height * full_seg_map_width * 4 + sy * full_seg_map_width + sx;
        // 不知道是不是由于TensorRT模型推理使用了float16精度的原因，推理出来的物体像素的prob值比较低....这里最后取大约0.40
        // 概率大于0.5，且比之前的值要大（可能在放大的过程中有重叠的点）
        // if (prob > 128 && prob > float(full_seg_map_device[index_prob]))
        if (prob > 76)
        {
            // 行的每个元素sx应该也要*2才对，因为每个元素的cls 和 obj id是紧紧挨在一起的
            // full_seg_map_device[sy*full_seg_map_width*2+sx]   = cls_label;
            // full_seg_map_device[sy*full_seg_map_width*2+sx+1]  = obj_id;
            full_seg_map_device[sy*full_seg_map_width*2+2*sx]   = cls_label;
            full_seg_map_device[sy*full_seg_map_width*2+2*sx+1] = obj_id;
            full_seg_map_device[index_prob] = (unsigned char)prob;
            if(is_valid == 1) full_seg_map_device[index_mask_obj] = 255;
            full_seg_map_device[index_mask_bg] = 0;
        }
        // mask用于检测新的FAST点，把概率定的大一些，避免检测到错误地区的点。然后valid物体的像素点prob最低不能低于0.25。
        // 理论上概率小于0.5的点要被视为背景点。为了排除一些检测置信度很低的动态物体的影响，把放入mask_for_obj的背景点的置信度再提高些，则把物体检测置信度限制为大约0.15。反正背景的区域足够大！
        else if(prob > 56)
        {
            // if(is_valid == 1) full_seg_map_device[index_mask_obj] = 255;
            full_seg_map_device[index_mask_bg] = 0;
        }
    }
    
    // 对每个物体的边界框范围内采样像素点及其深度值
    static __global__ void sample_pixel_objs(int num_objs, int num_sampled_pixel, float* bbox_info_device, 
                                            float mMaxDepthObj, float mMinDepthObj, float mbf, unsigned char* seg_map, 
                                            int seg_map_width, int seg_map_height, float* depth_map, float* buffer_device_pixel,
                                            float fx, float fy, float cx, float cy, float left_shift, float top_shift, 
                                            int W_dep_map, int H_dep_map, float left_shift_dep, float top_shift_dep)
    {
        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= num_objs) return;

        float* buffer_cur_obj = buffer_device_pixel + position * (num_sampled_pixel * 3 + 2 + 7);

        buffer_cur_obj[0] = bbox_info_device[5*position];
        int id_   = (int)bbox_info_device[5*position];
        unsigned char id_obj = (unsigned char)id_;
        float left   = bbox_info_device[5*position+1];
        float top    = bbox_info_device[5*position+2];
        float width  = bbox_info_device[5*position+3];
        float height = bbox_info_device[5*position+4];
        
        float right  = left + width;
        float bottom = top + height;
        // assert(right < seg_map_width && "The right border of bbox should be less than the width of full image!");
        // assert(bottom < seg_map_height && "The bottom border of bbox should be less than the height of full image!");
        int interval_ = sqrt((width * height) / num_sampled_pixel) + 0.5;
        if (interval_ == 0) interval_ = 1;

        // 这里不+1，即不向上取整，虽然bbox右边界和下边界附近一些点不会被遍历到，但这也避免了所取的点的坐标超出整个图像的范围
        int num_interval_u = width / interval_;
        int num_interval_v = height / interval_;
        
        unsigned char id_pixel;
        float u, v;
        int index_seg_map_obj, index_depth_map;
        float depth;
        int shift = 0;
        int num_pixel_find = 0;
        float ave_x = 0.0, ave_y = 0.0, ave_z = 0.0;

        if(num_interval_u < 2 || num_interval_v < 2)
        {
            printf("weird! No. 403 line in yolo_seg_decode.cu\n");
            // abort();
        }

        while(num_pixel_find < num_sampled_pixel && shift < interval_)
        {
            // 比较该物体bbox的宽和高，优先沿着较长的方向采样
            if (width <= height)
            {
                
                // 做到尽量均匀采样。且bbox边界内部附近的点尽量不要
                for(int i = 1; i < (num_interval_u-1); ++i)
                {
                    if (num_pixel_find >= num_sampled_pixel) break;
                    u = left + interval_ * i + shift;
                    if (u <= left_shift) continue;
                    if (u > right || u >= seg_map_width) break;
                    for(int j = 1; j < (num_interval_v-1); ++j)
                    {
                        if (num_pixel_find >= num_sampled_pixel) break;
                        v = top + interval_ * j + shift;
                        if (v <= top_shift) continue;
                        if (v > bottom || v >= seg_map_height) break;
                        index_seg_map_obj = ((int)v) * seg_map_width * 2 + ((int)u) * 2 + 1;
                        // dep_map和seg_map在padded后的大小是不同的，其pad的方式也是不同的！
                        index_depth_map = ((int)(v-top_shift+top_shift_dep)) * W_dep_map + ((int)(u-left_shift+left_shift_dep));
                        
                        if (seg_map[index_seg_map_obj] == id_obj)
                        {
                            depth = depth_map[index_depth_map];
                            if (depth <= 0) continue;
                            depth = mbf/depth;
                            // 超过指定深度范围的物体就放弃采样像素点
                            if (depth > mMaxDepthObj || depth < mMinDepthObj) continue;
                            ave_z = ave_z + depth;
                            // 真实图像的像素位置需要减去left_shift和top_shift,然后再lift到归一化平面坐标
                            ave_x = ave_x + (u-left_shift-cx) / fx * depth;
                            ave_y = ave_y + (v-top_shift-cy) / fy * depth;
                            buffer_cur_obj[2+num_pixel_find*3] = u-left_shift;
                            buffer_cur_obj[2+num_pixel_find*3+1] = v-top_shift;
                            buffer_cur_obj[2+num_pixel_find*3+2] = depth;
                            ++num_pixel_find;
                        }
                    }
                }
            }
            else
            {
                for(int j = 1; j < (num_interval_v-1); ++j)
                {
                    if (num_pixel_find >= num_sampled_pixel) break;
                    v = top + interval_ * j + shift;
                    if (v <= top_shift) continue;
                    if (v > bottom || v >= seg_map_height) break;
                    for(int i = 1; i < (num_interval_u-1); ++i)
                    {
                        if (num_pixel_find >= num_sampled_pixel) break;
                        u = left + interval_ * i + shift;
                        if (u <= left_shift) continue;
                        if (u > right || u >= seg_map_width) break;
                        index_seg_map_obj = ((int)v) * seg_map_width * 2 + ((int)u) * 2 + 1;
                        index_depth_map = ((int)(v-top_shift+top_shift_dep)) * W_dep_map + ((int)(u-left_shift+left_shift_dep));

                        if (seg_map[index_seg_map_obj] == id_obj)
                        {
                            depth = depth_map[index_depth_map];
                            if (depth <= 0) continue;
                            depth = mbf/depth;
                            if (depth > mMaxDepthObj || depth < mMinDepthObj) continue;
                            ave_z = ave_z + depth;
                            // 真实图像的像素位置需要减去left_shift和top_shift
                            ave_x = ave_x + (u-left_shift-cx) / fx * depth;
                            ave_y = ave_y + (v-top_shift-cy) / fy * depth;
                            buffer_cur_obj[2+num_pixel_find*3] = u-left_shift;
                            buffer_cur_obj[2+num_pixel_find*3+1] = v-top_shift;
                            buffer_cur_obj[2+num_pixel_find*3+2] = depth;
                            ++num_pixel_find;
                        }
                    }
                }
            }
            shift = shift + 1;
        } 
        if(num_pixel_find > 0)
        {
            ave_x = ave_x / num_pixel_find;
            ave_y = ave_y / num_pixel_find;
            ave_z = ave_z / num_pixel_find;
        }
        
        buffer_cur_obj[1] = num_pixel_find;
        int num_size = 3 * num_sampled_pixel + 2;
        // u_min, u_max, v_min, v_max
        buffer_cur_obj[num_size] = left-left_shift;
        buffer_cur_obj[num_size+1] = right-left_shift;
        buffer_cur_obj[num_size+2] = top-top_shift;
        buffer_cur_obj[num_size+3] = bottom-top_shift;
        buffer_cur_obj[num_size+4] = ave_x;
        buffer_cur_obj[num_size+5] = ave_y;
        buffer_cur_obj[num_size+6] = ave_z;
    }


    void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
                                int seg_map_width, int seg_map_height, unsigned char *small_seg_map_device,
                                int mask_dim, int map_bbox_width, int map_bbox_height, cudaStream_t stream) {
        // mask_weights is mask_dim(32 element) gpu pointer
        dim3 grid((map_bbox_width + 31) / 32, (map_bbox_height + 31) / 32);
        dim3 block(32, 32);

        checkCudaKernel(decode_single_mask_kernel<<<grid, block, 0, stream>>>(
            left, top, mask_weights, mask_predict, seg_map_width, seg_map_height, 
            small_seg_map_device, mask_dim, map_bbox_width, map_bbox_height));
    }

    // 
    void build_full_seg_mask(float left, float top, unsigned char *full_seg_map_device, int full_seg_map_width, 
                            int full_seg_map_height, unsigned char *small_seg_map_device,
                            int small_seg_map_width, int small_seg_map_height, int full_map_bbox_width, 
                            int full_map_bbox_height, unsigned char cls_label, unsigned char obj_id,
                            unsigned char is_valid_solid_obj, cudaStream_t stream) 
    {
        // mask_weights is mask_dim(32 element) gpu pointer
        dim3 grid((full_map_bbox_width + 31) / 32, (full_map_bbox_height + 31) / 32);
        dim3 block(32, 32);

        checkCudaKernel(build_full_seg_mask_kernel<<<grid, block, 0, stream>>>(
            left, top, full_seg_map_device, full_seg_map_width, full_seg_map_height,
            small_seg_map_device, small_seg_map_width, small_seg_map_height, 
            full_map_bbox_width, full_map_bbox_height, cls_label, obj_id, is_valid_solid_obj));
    }

    void decode_kernel_invoker(float* predict, int num_bboxes, int num_classes, float confidence_threshold, float* invert_affine_matrix, float* dstparray, int max_objects, cudaStream_t stream){
        
        auto grid = CUDATools::grid_dims(num_bboxes);
        auto block = CUDATools::block_dims(num_bboxes);
        checkCudaKernel(decode_kernel_v8_Seg<<<grid, block, 0, stream>>>(predict, num_bboxes, num_classes, confidence_threshold, invert_affine_matrix, dstparray, max_objects));            
    }

    void sorted_bbox_for_NMS(float* bbox_ptr, float* bbox_num_ptr, int nitems, cudaStream_t stream){
        // Prepare CDP for the max depth 'MAX_DEPTH_1'.
        GPU_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH_1));
        
        int left = 0;
        int set_right = nitems - 1;
        cdp_simple_quicksort<<<1, 1, 0, stream>>>(bbox_ptr, left, bbox_num_ptr, set_right, 0);
        // 不希望gpu的任何操作都会阻塞host
        // GPU_CHECK(cudaDeviceSynchronize());
    }

    void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, cudaStream_t stream){
        
        auto grid = CUDATools::grid_dims(max_objects);
        auto block = CUDATools::block_dims(max_objects);
        checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold));
    }

    void sample_pixel_for_objs(int num_objs, int num_sampled_pixel, float* bbox_info_device, 
                                float mMaxDepthObj, float mMinDepthObj, float mbf, 
                                unsigned char* seg_map, int seg_map_width, int seg_map_height, 
                                float* depth_map, float* buffer_device_pixel,  float fx, float fy,
                                float cx, float cy, float left_shift, float top_shift, int W_dep_map,
                                int H_dep_map, float left_shift_dep, float top_shift_dep, cudaStream_t stream)
    {
        auto grid = CUDATools::grid_dims(num_objs);
        auto block = CUDATools::block_dims(num_objs);
        sample_pixel_objs<<<grid, block, 0, stream>>>(num_objs, num_sampled_pixel, bbox_info_device, mMaxDepthObj, 
                                                        mMinDepthObj, mbf, seg_map, seg_map_width, seg_map_height, 
                                                        depth_map, buffer_device_pixel, fx, fy, cx, cy, left_shift, 
                                                        top_shift, W_dep_map, H_dep_map, left_shift_dep, top_shift_dep);
    }
};