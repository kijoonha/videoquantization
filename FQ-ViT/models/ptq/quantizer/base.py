# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn


class BaseQuantizer(nn.Module):

    def __init__(self, bit_type, observer, module_type):
        super(BaseQuantizer, self).__init__()
        self.bit_type = bit_type
        self.observer = observer
        self.module_type = module_type

    def get_reshape_range(self, inputs):
        range_shape = None
        # print("inputs.shape: ", inputs.shape)
        #예를 들면, inputs shape:  torch.Size([2, 16, 56, 56, 128])
        if self.module_type == 'conv_weight':
            # print("conv, weight : -1, 1, 1, 1")
            if len(inputs.shape) == 5: ##3dConv를 위해 추가한 부분
                range_shape=(-1,1,1,1,1)
            else:
                range_shape = (-1, 1, 1, 1)
        elif self.module_type == 'linear_weight':
            # print("linear weight: -1, 1")
            range_shape = (-1, 1)
        elif self.module_type == 'activation':
            if len(inputs.shape) == 2:
                range_shape = (1, -1)
                # print("activation len = 2, (1, -1)")
            elif len(inputs.shape) == 3:
                range_shape = (1, 1, -1)
                # print("activation len = 3, (1, 1, -1)")
            elif len(inputs.shape) == 4:
                range_shape = (1, -1, 1, 1)
                # print("activation len = 4, (1,-1, 1, 1)")
            elif len(inputs.shape) == 5:  #임의의 메쏘드
                range_shape = (1, 1, 1, 1, -1)
                # print("activation len = 5, (1, 1, 1,1,-1)")
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return range_shape

    def update_quantization_params(self, *args, **kwargs):
        pass

    def quant(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    def dequantize(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    def forward(self, inputs):
        # print("before quantization: ", inputs)
        # print("input shape: ", inputs.shape)
        outputs = self.quant(inputs)
        # print("after quantization: ", outputs)
        outputs = self.dequantize(outputs)
        # print("after dequantization: ", outputs)
        return outputs
