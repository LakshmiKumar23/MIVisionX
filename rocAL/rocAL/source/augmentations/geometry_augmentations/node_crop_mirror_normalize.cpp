/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_crop_mirror_normalize.h"
#include "exception.h"


CropMirrorNormalizeNode::CropMirrorNormalizeNode(const std::vector<rocalTensor *> &inputs,
                                                 const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs),
        _mirror(MIRROR_RANGE[0], MIRROR_RANGE[1]) {
        _crop_param = std::make_shared<RocalCropParam>(_batch_size);
}

void CropMirrorNormalizeNode::create_node() {
    if(_node)
        return;

    if(_crop_param->crop_h == 0 || _crop_param->crop_w == 0)
        THROW("Uninitialized destination dimension - Invalid Crop Sizes")

    _crop_param->create_array(_graph);

    std::vector<float> mean_vx, std_dev_vx;
    int mean_stddev_array_size = _batch_size * _inputs[0]->info().get_channels();
    if(!_std_dev[0])
        THROW("Standard deviation value cannot be 0");
    mean_vx.resize(mean_stddev_array_size, -(_mean[0] / _std_dev[0]));
    std_dev_vx.resize(mean_stddev_array_size, (1 / _std_dev[0]));
    
    if(_inputs[0]->info().get_channels() == 3) {
        if(!(_std_dev[0] && _std_dev[1] && _std_dev[2]))
            THROW("Standard deviation value cannot be 0");
        std_dev_vx[0] = 1 / _std_dev[0];
        std_dev_vx[1] = 1 / _std_dev[1];
        std_dev_vx[2] = 1 / _std_dev[2];
        mean_vx[0] = -(_mean[0] * std_dev_vx[0]);
        mean_vx[1] = -(_mean[1] * std_dev_vx[1]);
        mean_vx[2] = -(_mean[2] * std_dev_vx[2]);
        for (uint i = 1, j = 3; i < _batch_size; i++ , j += 3) {
        mean_vx[j] = mean_vx[0];
        mean_vx[j + 1] = mean_vx[1];
        mean_vx[j + 2] = mean_vx[2];
        std_dev_vx[j] = std_dev_vx[0];
        std_dev_vx[j + 1] = std_dev_vx[1];
        std_dev_vx[j + 2] = std_dev_vx[2];
        }
    }

    _mean_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, mean_stddev_array_size);
    _std_dev_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, mean_stddev_array_size);
    vx_status status = VX_SUCCESS;
    status |= vxAddArrayItems(_mean_array, mean_stddev_array_size, mean_vx.data(), sizeof(vx_float32));
    status |= vxAddArrayItems(_std_dev_array, mean_stddev_array_size, std_dev_vx.data(), sizeof(vx_float32));
    _mirror.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    if(status != 0)
        THROW(" vxAddArrayItems failed in the crop_mirror_normalize node (vxExtrppNode_CropMirrorNormalize)  node: "+ TOSTR(status) + "  "+ TOSTR(status))

    int input_layout = (int)_inputs[0]->info().layout();
    int output_layout = (int)_outputs[0]->info().layout();
    int roi_type = (int)_inputs[0]->info().roi_type();
    vx_scalar in_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &input_layout);
    vx_scalar out_layout_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &output_layout);
    vx_scalar roi_type_vx = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_INT32, &roi_type);
    _node = vxExtrppNode_CropMirrorNormalize(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(),
                                             _mean_array, _std_dev_array, _mirror.default_array(), in_layout_vx, out_layout_vx, roi_type_vx, _batch_size);
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop mirror normalize (vxExtrppNode_CropMirrorNormalize) failed: " + TOSTR(status))
}

void CropMirrorNormalizeNode::update_node() {
    _crop_param->set_image_dimensions(_inputs[0]->info().get_roi());
    _crop_param->update_array();
    std::vector<uint32_t> crop_h_dims, crop_w_dims;
    _crop_param->get_crop_dimensions(crop_w_dims, crop_h_dims);
    _outputs[0]->update_tensor_roi(crop_w_dims, crop_h_dims);
    _mirror.update_array();
    
    // Obtain the crop coordinates and update the roi
    auto x1 = _crop_param->get_x1_arr_val();
    auto y1 = _crop_param->get_y1_arr_val();
    std::vector<uint32_t> src_roi(_batch_size * 4, 0);
    for(unsigned i = 0, j = 0; i < _batch_size; i++, j+= 4) {
        src_roi[j] = x1[i];
        src_roi[j + 1] = y1[i];
        src_roi[j + 2] = crop_w_dims[i];
        src_roi[j + 3] = crop_h_dims[i];
    }
    vx_status status;
    status = vxCopyArrayRange((vx_array)_src_tensor_roi, 0, _batch_size * 4, sizeof(vx_uint32), src_roi.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != 0)
        WRN("ERROR: vxCopyArrayRange _src_tensor_roi failed " + TOSTR(status));
}

void CropMirrorNormalizeNode::init(int crop_h, int crop_w, float start_x, float start_y, std::vector<float>& mean, std::vector<float>& std_dev, IntParam *mirror) {
    _crop_param->x1 = start_x;
    _crop_param->y1 = start_y;
    _crop_param->crop_h = crop_h;
    _crop_param->crop_w = crop_w;
    _mean   = mean;
    _std_dev = std_dev;
    _mirror.set_param(core(mirror));
}
