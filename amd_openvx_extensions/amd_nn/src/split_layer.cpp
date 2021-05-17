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

#include <stdio.h>
#include <sys/stat.h>
struct SplitLayerLocalData {
    NeuralNetworkCommonHandle * handle;
    cl_mem input_mem;
    cl_mem output_mem[4];
    vx_size memsizeInBytes[4];
};


static vx_status VX_CALLBACK validateSplitLayer(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // check input and output tensor dimensions
    vx_size num_dims;
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
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[0], VX_TENSOR_DIMS, output1_dims, sizeof(output_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if(num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #2 num_dims=%ld (must be 4)\n", num_dims);
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: split: #2 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, output2_dims, sizeof(output2_dims)));
    if(type != out_type) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #2 output type(%d) does not match input type(%d)\n", out_type, type);
    // set output tensor configuration
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, output2_dims, sizeof(output_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if(num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #3 num_dims=%ld (must be 4)\n", num_dims);
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: split: #3 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_DIMS, output3_dims, sizeof(output3_dims)));
    if(type != out_type) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #3 output type(%d) does not match input type(%d)\n", out_type, type);
    // set output tensor configuration
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_TENSOR_DIMS, output3_dims, sizeof(output_dims)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    if(num_dims != 4) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #4 num_dims=%ld (must be 4)\n", num_dims);
    if ((out_type != VX_TYPE_FLOAT32) && (out_type != VX_TYPE_FLOAT16)) return ERRMSG(VX_ERROR_INVALID_TYPE, "validate: split: #4 type=%d (must be float)\n", type);
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_DIMS, output4_dims, sizeof(output4_dims)));
    if(type != out_type) return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split: #4 output type(%d) does not match input type(%d)\n", out_type, type);
    // set output tensor configuration
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DATA_TYPE, &out_type, sizeof(out_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[3], VX_TENSOR_DIMS, output4_dims, sizeof(output_dims)));

    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &scalar_type, sizeof(scalar_type)));
    if(scalar_type != VX_TYPE_INT32) return VX_ERROR_INVALID_TYPE;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[5], &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(axis < -4 || axis > 3) return ERRMSG(VX_ERROR_INVALID_VALUE, "validate: split: #6 scalar type=%d (must be greater than -5 and lesser than 3)\n", axis); 

    // check if the input and sum of all output are of the same size in memory
    if (((output1_dims[0]*output1_dims[1]*output1_dims[2]*output1_dims[3]) 
            + (output2_dims[0]*output2_dims[1]*output2_dims[2]*output2_dims[3]) 
            + (output3_dims[0]*output3_dims[1]*output3_dims[2]*output3_dims[3])  
            + (output4_dims[0]*output4_dims[1]*output4_dims[2]*output4_dims[3]) != (input_dims[0]*input_dims[1]*input_dims[2]*input_dims[3]))) 
         return ERRMSG(VX_ERROR_INVALID_DIMENSION, "validate: split:dimension mismatch; output1_dims[%ldx%ldx%ldx%ld] output2_dims[%ldx%ldx%ldx%ld] 
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


static vx_status VX_CALLBACK processSplitLayer(vx_node node, const vx_reference * parameters, vx_uint32 num)
{
    SplitLayerLocalData * data= NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)));

    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[4], VX_TENSOR_BUFFER_OPENCL, &data->input_mem, sizeof(data->input_mem)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_BUFFER_OPENCL, &data->output_mem[0], sizeof(data->output_mem[0])));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_BUFFER_OPENCL, &data->output_mem[1], sizeof(data->output_mem[1])));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[2], VX_TENSOR_BUFFER_OPENCL, &data->output_mem[2], sizeof(data->output_mem[2])));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[3], VX_TENSOR_BUFFER_OPENCL, &data->output_mem[3], sizeof(data->output_mem[3])));


    /* make changes to copy correct data...need to calculate src and dst offset
    ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem, 0, 0, data->memsizeInBytes[0], 0, NULL, NULL));
    ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem, 0, 0, data->memsizeInBytes[1], 0, NULL, NULL));
    ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem, 0, 0, data->memsizeInBytes[2], 0, NULL, NULL));
    ERROR_CHECK_STATUS(clEnqueueCopyBuffer(data->handle->cmdq, data->input_mem, data->output_mem, 0, 0, data->memsizeInBytes[3], 0, NULL, NULL));
    */

    return VX_SUCCESS;
}


static vx_status VX_CALLBACK initializeSplitLayer(vx_node node, const vx_reference *parameters, vx_uint32 num)
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
}

//! \brief The kernel publisher.
vx_status publishSplitLayer(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context, "com.amd.nn_extension.split_layer", VX_KERNEL_SPLIT_LAYER_AMD, processSplitLayer, 6, validateReshapeLayer, initializeSplitLayer, uninitializeSplitLayer);
    ERROR_CHECK_OBJECT(kernel);

    // enable OpenCL buffer access since the kernel_f callback uses OpenCL buffers instead of host accessible buffers
    vx_bool enableBufferAccess = vx_true_e;
    ERROR_CHECK_STATUS(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_BUFFER_ACCESS_ENABLE, &enableBufferAccess, sizeof(enableBufferAccess)));
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