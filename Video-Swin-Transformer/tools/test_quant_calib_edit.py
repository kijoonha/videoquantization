import argparse
import os
import os.path as osp
import warnings
import math
import time
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import random
import sys
import numpy as np

import mmcv
import torch
from mmcv import Config, Config2, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model


from mmaction.datasets import build_dataloader, build_dataset

from mmaction.models import build_model_quant
from mmaction.utils import register_module_hooks

# TODO import test functions from mmcv and delete them from mmaction2
try: 
    from mmcv.engine import multi_gpu_test, single_gpu_test
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'DeprecationWarning: single_gpu_test, multi_gpu_test, '
        'collect_results_cpu, collect_results_gpu from mmaction2 will be '
        'deprecated. Please install mmcv through master branch.')
    from mmaction.apis import multi_gpu_test, single_gpu_test


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    #인자 추가
    parser.add_argument('model',
                        choices=[
                            'deit_tiny', 'deit_small', 'deit_base', 'vit_base',
                            'vit_large', 'swin_tiny', 'swin_small', 'swin_base'
                        ],
                        help='model')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--quant', default=False, action='store_true')
    parser.add_argument('--ptf', default=False, action='store_true')
    parser.add_argument('--lis', default=False, action='store_true')
    parser.add_argument('--quant-method',
                        default='minmax',
                        choices=['minmax', 'ema', 'omse', 'percentile'])
    parser.add_argument('--calib_batchsize',
                        default=100,
                        type=int,
                        help='batchsize of calibration set')
    parser.add_argument('--calib-iter', default=3, type=int)
    parser.add_argument('--val-batchsize',
                        default=100,
                        type=int,
                        help='batchsize of validation set')
    parser.add_argument('--num-workers',
                        default=16,
                        type=int,
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--print-freq',
                        default=100,
                        type=int,
                        help='print frequency')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    #인자 추가 끝

    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--onnx',
        action='store_true',
        help='Whether to test with onnx model or not')
    parser.add_argument(
        '--tensorrt',
        action='store_true',
        help='Whether to test with TensorRT engine or not')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args




def seed(seed=0):

    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def inference_pytorch(args, cfg_quant, cfg, distributed, data_loader):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    # build the model and load checkpoint
    model = build_model_quant(
        cfg.model, cfg_quant=cfg_quant, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    #obj_type recognizer?
    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[1])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    return outputs

def inference_pytorch_quant (args, cfg_quant, cfg, distributed, data_loader, video_list, quant):
    """Get predictions by pytorch models."""
    
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    # build the model and load checkpoint
    model = build_model_quant(
        cfg.model, cfg_quant=cfg_quant, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    # model:  <class 'mmaction.models.recognizers.recognizer3d.Recognizer3D'>
    # model backbone: <class 'mmaction.models.backbones.swin_transformer.SwinTransformer3D_quant'>
    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    # state_dict = model.state_dict()
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    current_device = torch.cuda.current_device()  # 현재 GPU 디바이스 인덱스를 가져옴
    device_num = torch.device(f'cuda:{current_device}')  # 현재 GPU를 torch.device 객체로 생성
    if quant:
        print('Calibrating...')
        if cfg_quant.print_model:
            model.backbone.print_model()

        ##calibration 을 여기서 수행?원래코드의 calibrating이후
        model.backbone.model_open_calibrate()
        with torch.no_grad():
            for i, imgs in enumerate(video_list):
                # print("여기 실행되나?")
                if i == len(video_list) - 1:
                    # This is used for OMSE method to
                    # calculate minimum quantization error
                    model.backbone.model_open_last_calibrate()
                imgs=imgs.squeeze(0)
                model.to(device_num)
                imgs = imgs.to(device_num)
                output = model.backbone(imgs)
        model.backbone.model_close_calibrate()
        model.backbone.model_quant() 
        print(f"cfg_quant.act_dequant: {cfg_quant.act_dequant}")
        print(f"cfg_act_quant_layer: {cfg_quant.act_quant_layer}")
        print(f"input_quant: {cfg_quant.input_quant}")
        if cfg_quant.act_dequant:
            model.backbone.act_dequant()
        print('Validating...')
        
        

    if not distributed:
        model = MMDataParallel(model, device_ids=[current_device]) #calibration device랑 같은거로해야함)
        #model class:  <class 'mmcv.parallel.data_parallel.MMDataParallel'>
        # print("module: ", type(model.module))->module:  <class 'mmaction.models.recognizers.recognizer3d.Recognizer3D'>
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=True)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
        print(f"cfg_quant.act_dequant: {cfg_quant.act_dequant}")
        print(f"cfg_act_quant_layer: {cfg_quant.act_quant_layer}")
        print(f"input_quant: {cfg_quant.input_quant}")

    return outputs

def inference_tensorrt(ckpt_path, distributed, data_loader, batch_size):
    """Get predictions by TensorRT engine.

    For now, multi-gpu mode and dynamic tensor shape are not supported.
    """
    assert not distributed, \
        'TensorRT engine inference only supports single gpu mode.'
    import tensorrt as trt
    from mmcv.tensorrt.tensorrt_utils import (torch_dtype_from_trt,
                                              torch_device_from_trt)

    # load engine
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(ckpt_path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    # For now, only support fixed input tensor
    cur_batch_size = engine.get_binding_shape(0)[0]
    assert batch_size == cur_batch_size, \
        ('Dataset and TensorRT model should share the same batch size, '
         f'but get {batch_size} and {cur_batch_size}')

    context = engine.create_execution_context()

    # get output tensor
    dtype = torch_dtype_from_trt(engine.get_binding_dtype(1))
    shape = tuple(context.get_binding_shape(1))
    device = torch_device_from_trt(engine.get_location(1))
    output = torch.empty(
        size=shape, dtype=dtype, device=device, requires_grad=False)

    # get predictions
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        bindings = [
            data['imgs'].contiguous().data_ptr(),
            output.contiguous().data_ptr()
        ]
        context.execute_async_v2(bindings,
                                 torch.cuda.current_stream().cuda_stream)
        results.extend(output.cpu().numpy())
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return results


def inference_onnx(ckpt_path, distributed, data_loader, batch_size):
    """Get predictions by ONNX.

    For now, multi-gpu mode and dynamic tensor shape are not supported.
    """
    assert not distributed, 'ONNX inference only supports single gpu mode.'

    import onnx
    import onnxruntime as rt

    # get input tensor name
    onnx_model = onnx.load(ckpt_path)
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1

    # For now, only support fixed tensor shape
    input_tensor = None
    for tensor in onnx_model.graph.input:
        if tensor.name == net_feed_input[0]:
            input_tensor = tensor
            break
    cur_batch_size = input_tensor.type.tensor_type.shape.dim[0].dim_value
    assert batch_size == cur_batch_size, \
        ('Dataset and ONNX model should share the same batch size, '
         f'but get {batch_size} and {cur_batch_size}')

    # get predictions
    sess = rt.InferenceSession(ckpt_path)
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        imgs = data['imgs'].cpu().numpy()
        onnx_result = sess.run(None, {net_feed_input[0]: imgs})[0]
        results.extend(onnx_result)
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return results


def main():
    args = parse_args()
    seed(args.seed)

    if args.tensorrt and args.onnx:
        raise ValueError(
            'Cannot set onnx mode and tensorrt mode at the same time.')

    cfg = Config.fromfile(args.config)
    cfg_quant = Config2((args.ptf, args.lis, args.quant_method))
    # cfg.update(cfg2.__dict__)
    
    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    if args.out:
        # Overwrite output_config from args.out
        output_config = Config._merge_a_into_b(
            dict(out=args.out), output_config)

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    if args.eval:
        # Overwrite eval_config from args.eval
        eval_config = Config._merge_a_into_b(
            dict(metrics=args.eval), eval_config)
    if args.eval_options:
        # Add options from args.eval_options
        eval_config = Config._merge_a_into_b(args.eval_options, eval_config)

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    dataset_type = cfg.data.test.type
    if output_config.get('out', None):
        if 'output_format' in output_config:
            # ugly workround to make recognition and localization the same
            warnings.warn(
                'Skip checking `output_format` in localization task.')
        else:
            out = output_config['out']
            # make sure the dirname of the output path exists
            mmcv.mkdir_or_exist(osp.dirname(out))
            _, suffix = osp.splitext(out)
            if dataset_type == 'AVADataset':
                assert suffix[1:] == 'csv', ('For AVADataset, the format of '
                                             'the output file should be csv')
            else:
                assert suffix[1:] in file_handlers, (
                    'The format of the output '
                    'file should be json, pickle or yaml')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])
    
    current_device = torch.cuda.current_device() 
    device0 = torch.device(f'cuda:{current_device}')
    
    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)
    video_list=[]
    # dataloader:  <class 'torch.utils.data.dataloader.DataLoader'>
    if args.quant:
        #calibration dataset 생성
        calib_dataset = build_dataset(cfg.data.calib)
        calib_loader_setting = dict(
            # videos_per_gpu=args.calib_batchsize,
            videos_per_gpu=1,
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            dist=distributed,
            shuffle=True)
        calib_loader = build_dataloader(calib_dataset, **calib_loader_setting)
        # "loader shape: ", calib_loader.dataset[0]['imgs'].shape) : [1,3,1,224,224]
        # Get calibration set.
        # video_list = []
        for i, data in enumerate(calib_loader):
            imgs = data.get('imgs')
            label = data.get('label')
            if i == args.calib_iter:            
                break
            imgs.to(device0)
            label.to(device0)
            video_list.append(imgs)
        print('Calibrating...')
        
    if args.tensorrt:
        outputs = inference_tensorrt(args.checkpoint, distributed, data_loader,
                                     dataloader_setting['videos_per_gpu'])
    elif args.onnx:
        outputs = inference_onnx(args.checkpoint, distributed, data_loader,
                                 dataloader_setting['videos_per_gpu'])
    else:
        outputs = inference_pytorch_quant(args, cfg_quant, cfg, distributed, data_loader, video_list, args.quant)


    rank, _ = get_dist_info()
    if rank == 0:
        if output_config.get('out', None):
            out = output_config['out']
            print(f'\nwriting results to {out}')
            dataset.dump_results(outputs, **output_config)
        if eval_config:
            eval_res = dataset.evaluate(outputs, **eval_config)
            for name, val in eval_res.items():
                print(f'{name}: {val:.04f}')
    print(f"quant  method: {args.quant_method}")
    print(f"bit type of attention: {cfg_quant.BIT_TYPE_S}")
    print("lis?: ", {cfg_quant.lis})
if __name__ == '__main__':   
    print(f"quant  method: minmax")
    
    # print("no lis")
    main()