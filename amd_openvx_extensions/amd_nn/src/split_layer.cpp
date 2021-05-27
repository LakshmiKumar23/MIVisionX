/*
Copyright (c) 2017 - 2020 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "kernels.h"

static vx_status VX_CALLBACK validateSplitLayer(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check input and output tensor dimensions
    vx_size num_dims;
    vx_int32 axis;
    vx_enum type, out_type, scalar_type;
    vx_size input_dims[4], output1_dims[4], output2_dims[4], output3_dims[4], output4_dims[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    if(num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #5 num_dims=%ld (must be 4)\n", num_dims);
    if((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: split: #5 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if(num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #1 num_dims=%ld (must be 4)\n", num_dims);
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: split: #1 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, output1_dims, sizeof(output1_dims)));
    if(type != out_type) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #1 output type(%d) does not match input type(%d)\n", out_type, type);
    // set output tensor configuration
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[0], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[0], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[0], VX_TENSOR_DIMS, output1_dims, sizeof(output1_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if(num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #2 num_dims=%ld (must be 4)\n", num_dims);
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: split: #2 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output2_dims, sizeof(output2_dims)));
    if(type != out_type) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #2 output type(%d) does not match input type(%d)\n", out_type, type);
    // set output tensor configuration
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, output2_dims, sizeof(output2_dims)));

    if(parameters[2])
    {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
        if(num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #3 num_dims=%ld (must be 4)\n", num_dims);
        if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: split: #3 type=%d (must be float)\n", type);
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output3_dims, sizeof(output3_dims)));
        if(type != out_type) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #3 output type(%d) does not match input type(%d)\n", out_type, type);
        // set output tensor configuration
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, output3_dims, sizeof(output3_dims)));
    }

    if(parameters[3])
    {
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
        if(num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #4 num_dims=%ld (must be 4)\n", num_dims);
        if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: split: #4 type=%d (must be float)\n", type);
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, output4_dims, sizeof(output4_dims)));
        if(type != out_type) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #4 output type(%d) does not match input type(%d)\n", out_type, type);
        // set output tensor configuration
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DIMS, output4_dims, sizeof(output4_dims)));
    }

    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32 && scalar_type != VX_TYPE_INT64) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[5], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(axis < 0 || axis > 3) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: split: #6 scalar type=%d (must be greater than -5 and lesser than 3)\n", axis); 

    // check if the input and sum of all output are of the same size in memory
    if (((output1_dims[0]*output1_dims[1]*output1_dims[2]*output1_dims[3]) 
            + (output2_dims[0]*output2_dims[1]*output2_dims[2]*output2_dims[3]) 
            + (output3_dims[0]*output3_dims[1]*output3_dims[2]*output3_dims[3])  
            + (output4_dims[0]*output4_dims[1]*output4_dims[2]*output4_dims[3]) != (input_dims[0]*input_dims[1]*input_dims[2]*input_dims[3]))) \
         return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split:dimension mismatch; output1_dims[%ldx%ldx%ldx%ld] output2_dims[%ldx%ldx%ldx%ld] \
                        output3_dims[%ldx%ldx%ldx%ld] output4_dims[%ldx%ldx%ldx%ld] input_dims[%ldx%ldx%ldx%ld]\n", 
                        output1_dims[3], output1_dims[2], output1_dims[1], output1_dims[0], 
                        output2_dims[3], output2_dims[2], output2_dims[1], output2_dims[0],
                        output3_dims[3], output3_dims[2], output3_dims[1], output3_dims[0],
                        output4_dims[3], output4_dims[2], output4_dims[1], output4_dims[0],
                        input_dims[3], input_dims[2], input_dims[1], input_dims[0]);

    if(parameters[6]) 
    {
        //ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        //ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        //if(num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #5 num_dims=%ld (must be 4)\n", num_dims);
        if(type != VX_TYPE_INT64) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: split: #7 type=%d (must be int64)\n", type);
        //ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    }


    return VX_SUCCESS;
}


/*static vx_status VX_CALLBACK processSplitLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    SplitLayerLocalData * data= NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->output_mem[0], sizeof(data->output_mem[0])));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->output_mem[1], sizeof(data->output_mem[1])));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_OPENCL, &data->output_mem[2], sizeof(data->output_mem[2])));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_OPENCL, &data->output_mem[3], sizeof(data->output_mem[3])));

    vx_int32 axis;
    vx_size input_stride[4];
    vx_int32 num_outputs = 2;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[5], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_STRIDE_OPENCL, input_stride, sizeof(input_stride)));

    if(parameters[2])
    {
        if(parameters[3]) 
            num_outputs = 4;
        else
            num_outputs = 3;
    }

    if(!parameters[6])          //splits tensor equally along given axis
    {
        if(axis == 0)
        {
            ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[0], 0, 0, data->memsizeInBytes[0], 0, NULL, NULL));
            ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[1], 0, (input_stride[1])/num_outputs, data->memsizeInBytes[1], 0, NULL, NULL));
            if(parameters[2])
            {
                ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[2], 0, (2*input_stride[1])/num_outputs, data->memsizeInBytes[2], 0, NULL, NULL));
                if(parameters[3])
                {
                    ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[3], 0, (3*input_stride[1])/num_outputs, data->memsizeInBytes[3], 0, NULL, NULL));
                }
            }
        }
        else if(axis == 1)
        {
            ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[0], 0, 0, data->memsizeInBytes[0], 0, NULL, NULL));
            ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[1], 0, (input_stride[2])/num_outputs, data->memsizeInBytes[1], 0, NULL, NULL));
            if(parameters[2])
            {
                ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[2], 0, (2*input_stride[2])/num_outputs, data->memsizeInBytes[2], 0, NULL, NULL));
                if(parameters[3])
                {
                    ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[3], 0, (3*input_stride[2])/num_outputs, data->memsizeInBytes[3], 0, NULL, NULL));
                }
            }
        }
    }
    else
    {   
        add split details -- cl_mem 
        cl_mem split_mem;
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[6], VX_TENSOR_BUFFER_OPENCL, split_mem, sizeof(data->input_mem)));

        if(axis == 0)
        {
            ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[0], 0, 0, data->memsizeInBytes[0], 0, NULL, NULL));
            ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[1], 0, (*split_mem*input_stride[0]*input_stride[1]), data->memsizeInBytes[1], 0, NULL, NULL));
            if(parameters[2])
            {
                ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[2], 0, (*split_mem*input_stride[0]*input_stride[1]), data->memsizeInBytes[2], 0, NULL, NULL));
                if(parameters[3])
                {
                    ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[3], 0, (*split_mem*input_stride[0]*input_stride[1]), data->memsizeInBytes[3], 0, NULL, NULL));
                }
            }
        }
        else if(axis == 1)
        {
            ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[0], 0, 0, data->memsizeInBytes[0], 0, NULL, NULL));
            ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[1], 0, (*split_mem*input_stride[0]*input_stride[2]), data->memsizeInBytes[1], 0, NULL, NULL));
            if(parameters[2])
            {
                ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[2], 0, (*split_mem*input_stride[0]*input_stride[2]), data->memsizeInBytes[2], 0, NULL, NULL));
                if(parameters[3])
                {
                    ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem[3], 0, (*split_mem*input_stride[0]*input_stride[2]), data->memsizeInBytes[3], 0, NULL, NULL));
                }
            }
        }
    }

    return VX_SUCCESS;
}*/

//! \brief The kernel target support callback.
static vx_status VX_CALLBACK query_target_support(vx_graph graph, vx_node node,
    vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
    vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
    )
{
    supported_target_affinity = AGO_TARGET_AFFINITY_GPU;
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK opencl_codegen(
    vx_node node,                                  // [input] node
    const vx_reference parameters[],               // [input] parameters
    vx_uint32 num,                                 // [input] number of parameters
    bool opencl_load_function,                     // [input]  false: normal OpenCL kernel; true: reserved
    char opencl_kernel_function_name[64],          // [output] kernel_name for clCreateKernel()
    std::string& opencl_kernel_code,               // [output] string for clCreateProgramWithSource()
    std::string& opencl_build_options,             // [output] options for clBuildProgram()
    vx_uint32& opencl_work_dim,                    // [output] work_dim for clEnqueueNDRangeKernel()
    vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
    vx_size opencl_local_work[],                   // [output] local_work[] for clEnqueueNDRangeKernel()
    vx_uint32& opencl_local_buffer_usage_mask,     // [output] reserved: must be ZERO
    vx_uint32& opencl_local_buffer_size_in_bytes   // [output] reserved: must be ZERO
)
{
    //get tensor dimensions
    vx_size input_dims[4], output_dims[4];
    vx_size num_of_dims;
    vx_enum type;
    vx_size input_stride[4], output_stride[4];

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, input_dims, sizeof(input_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_STRIDE_OPENCL, input_stride, sizeof(input_stride)));

    /*ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output_dims, sizeof(output_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_STRIDE_OPENCL, output_stride, sizeof(output_stride)));
    */
    vx_int32 num_outputs = 2;
    if(parameters[2])
    {
        if(parameters[3]) 
            num_outputs = 4;
        else
            num_outputs = 3;
    }
    strcpy(opencl_kernel_function_name, "split_layer");

    opencl_work_dim = 3;
    opencl_global_work[0] = input_dims[0]*input_dims[1];
    opencl_global_work[1] = input_dims[2];
    opencl_global_work[2] = input_dims[3];

    // Setting variables required by the interface
    opencl_local_buffer_usage_mask = 0;
    opencl_local_buffer_size_in_bytes = 0;

    if (num_of_dims == 4) {
        char item[8192];
        sprintf(item,
            "#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
            "__kernel void %s( ");
        opencl_kernel_code = item;

        for(int i = 0; i < num_outputs; i++){
                sprintf(item,
                "__global float * out%d, uint out%d_offset, uint4 out%d_stride"  // i, i, i
                , i, i, i);
            opencl_kernel_code += item;
        }
        sprintf(item,
            ",__global uchar * in, uint in_offset, uint4 in_stride, uint axis");
        opencl_kernel_code = item;
        if(parameters[6]) {
            sprintf(item,
            "__global uchar * split, uint split_offset, uint4 split_stride ");
            opencl_kernel_code += item;
        }
        sprintf(item,
            "                 )\n"
            "{   \n"
            "   uint c = get_global_id(0); \n "
            "   uint x = get_global_id(1); \n "
            "   uint y = get_global_id(2); \n "
            "   int num_outputs = %d; \n"
            "   if(axis == 0) { \n"
            "       "
            "   } \n"
            "   else if(axis == 1) { \n"
            "   } \n"
            /*"   int i = y*out_stride.s2 + x*out_stride.s1 + c*out_stride.s0; \n"
            "   int old_idx = 0; \n"
            "   int idx = i; \n"
            "   for(int k = num_axis-1, j = 0; k >= 0; k--, j++) {  \n"
            "       int order = 3- ((__global int *)(order_buf+order_offset))[j]; \n"
            "       old_idx += (idx/out_stride[k]) * (in_stride[order]);  \n"
            "       idx %%= (out_stride[k]);  \n"
            "   } \n"
            "   out += out_offset + i; \n"
            "   in += in_offset + old_idx; \n"
            "   *(__global float *)&out[0] = *(__global float *)&in[0];  \n"*/
            "}\n", opencl_kernel_function_name, (int)num_outputs);
        opencl_kernel_code = item;
    }
    return VX_SUCCESS;
}

/*static vx_status VX_CALLBACK initializeSplitLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_size dims[4];
    vx_enum type;
    SplitLayerLocalData * data = new SplitLayerLocalData;
    memset(data, 0, sizeof(*data));
    ERROR_CHECK_STATUS(createGraphHandle(node, &data->handle));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, dims, sizeof(dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    // allocate memory for 1st output
    data->memsizeInBytes[0] = dims[0]*dims[1]*dims[2]*dims[3]*sizeof(type);

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, dims, sizeof(dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    // allocate memory for 2nd output
    data->memsizeInBytes[1] = dims[0]*dims[1]*dims[2]*dims[3]*sizeof(type);

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, dims, sizeof(dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    // allocate memory for 3rd output
    data->memsizeInBytes[2] = dims[0]*dims[1]*dims[2]*dims[3]*sizeof(type);

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, dims, sizeof(dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
    // allocate memory for 4th output
    data->memsizeInBytes[3] = dims[0]*dims[1]*dims[2]*dims[3]*sizeof(type);

    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    return VX_SUCCESS;
}

static vx_status VX_CALLBACK uninitializeSplitLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    SplitLayerLocalData * data = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));
    if (data) {
        ERROR_CHECK_STATUS(releaseGraphHandle(node, data->handle));
        delete data;
    }
    return VX_SUCCESS;
}*/

//! \brief The kernel execution.
static vx_status VX_CALLBACK host_kernel(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    return VX_ERROR_NOT_IMPLEMENTED;
}

//! \brief The kernel publisher.
vx_status publishSplitLayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.split_layer", VX_KERNEL_SPLIT_LAYER_AMD, host_kernel, 6, validateSplitLayer, nullptr, nullptr);
    ERROR_CHECK_OBJECT(kernel);

    amd_kernel_query_target_support_f query_target_support_f = query_target_support;
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = opencl_codegen;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT, &query_target_support_f, sizeof(query_target_support_f)));
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK, &opencl_codegen_callback_f, sizeof(opencl_codegen_callback_f)));

    // set kernel parameters.
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 6, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));

    // finalize and release kernel object.
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    return VX_SUCCESS;
}

VX_API_ENTRY vx_node VX_API_CALL vxSplitLayer(vx_graph graph, vx_tensor output1, vx_tensor output2, vx_tensor output3, vx_tensor output4, vx_tensor input, vx_int32 axis, vx_tensor split)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
    vx_scalar s_axis = vxCreateScalarWithSize(context, VX_TYPE_INT32, &axis, sizeof(axis));
        vx_reference params[] = {
            (vx_reference)output1,
            (vx_reference)output2,
            (vx_reference)output3,
            (vx_reference)output4,
            (vx_reference)input,
            (vx_reference)split,
        };
        node = createNode(graph, VX_KERNEL_SPLIT_LAYER_AMD, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}