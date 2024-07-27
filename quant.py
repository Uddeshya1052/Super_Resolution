from pytorch_nndct.apis import torch_quantizer
import torch
from srgan_model1 import Generator
from argparse import ArgumentParser
import torch.nn as nn
from ops import *
import torch.nn.functional as F
def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model", default="/workspace/SRGAN_upload/SRGAN_genetwice_200.pt", help="the pt model")
    parser.add_argument('--quant_mode', default='test', choices=['calib', 'test'], help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
    parser.add_argument('--output_dir', default='./compiled_xmodel_new', help='save xmodel')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    return parser


class GeneratorDPU(nn.Module):
    def __init__(self, img_feat=3, n_feats=64, kernel_size=3, num_block=16, act=nn.LeakyReLU(0.01)):
        super(GeneratorDPU, self).__init__()

        self.conv01 = conv(in_channel=img_feat, out_channel=n_feats, kernel_size=9, BN=False, act=act)

        resblocks = [ResBlock(channels=n_feats, kernel_size=3, act=act) for _ in range(num_block)]
        self.body = nn.Sequential(*resblocks)
        self.conv02 = conv(in_channel=n_feats, out_channel=n_feats, kernel_size=3, BN=True, act=None)
        
    def forward(self, x):
        x = self.conv01(x)
        _skip_connection = x
        x = self.body(x)
        x = self.conv02(x)
        feat = x + _skip_connection
        return feat

class GeneratorCPU(nn.Module):
    def __init__(self, n_feats=64, scale=4, act=nn.LeakyReLU(0.01)):
        super(GeneratorCPU, self).__init__()

        if scale == 4:
            upsample_blocks = [Upsampler(channel=n_feats, kernel_size=3, scale=2, act=act) for _ in range(2)]
        else:
            upsample_blocks = [Upsampler(channel=n_feats, kernel_size=3, scale=scale, act=act)]

        self.tail = nn.Sequential(*upsample_blocks)
        self.last_conv = conv(in_channel=n_feats, out_channel=3, kernel_size=3, BN=False, act=None)

    def forward(self, x):
        x = self.tail(x)
        x = self.last_conv(x)
        return x


def quant(args):
    print(torch.__version__)
    device = torch.device(args.device)

    # Instantiate the generator model
    model = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=16)
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    # Define the DPU part of the model up to self.body
    state_dict=model.state_dict()
    #print (state_dict.keys())
    dpu_model = GeneratorDPU()
    #cpu_model = GeneratorCPU()
    #dpu_model.load_state_dict(model.state_dict()) 
    # Generate random input
    dpu_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('conv01.') or key.startswith('body.')or key.startswith('conv02.'):
            dpu_state_dict[key] = value

    dpu_model.load_state_dict(dpu_state_dict)

    #cpu_state_dict = {}
    # for key, value in state_dict.items():
    #     if key.startswith('conv02.') or key.startswith('tail.'):
    #         cpu_state_dict[key] = value


    # for key, value in state_dict.items():
    #     if key.startswith('tail.') or key.startswith('last_conv.'):
    #         #cpu_state_dict[key.split('.', 1)[1]] = value
    #         cpu_state_dict[key] = value
    #print(cpu_state_dict.keys)
    #cpu_model.load_state_dict(cpu_state_dict)

    rand_in = torch.randn(1, 3, 64, 64).to(device)

    # Perform quantization on the DPU part only
    quantizer = torch_quantizer(args.quant_mode, model, rand_in)
    quantized_model = quantizer.quant_model
    quantized_model = quantized_model.to(device)

    quantized_model.eval()
    results = quantized_model(rand_in)

    if args.quant_mode == 'calib':
        # Export quantization configuration
        quantizer.export_quant_config()
        #print('ok')

    elif args.quant_mode == 'test':
        # Export xmodel
        quantizer.export_xmodel(output_dir=args.output_dir, deploy_check=True)
        

def quant_comb(args):
    print(torch.__version__)
    device = torch.device(args.device)

    # Instantiate the generator model
    model = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=16)
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    # Define the DPU part of the model up to self.body
    state_dict = model.state_dict()
    dpu_model = GeneratorDPU()
    cpu_model = GeneratorCPU()

    dpu_state_dict = {}
    cpu_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('conv01.') or key.startswith('body.'):
            dpu_state_dict[key] = value
        elif key.startswith('conv02.') or key.startswith('tail.') or key.startswith('last_conv.'):
            cpu_state_dict[key] = value

    dpu_model.load_state_dict(dpu_state_dict)
    cpu_model.load_state_dict(cpu_state_dict)

    rand_in_dpu = torch.randn(1, 3, 640, 256).to(device)
    rand_in_cpu = dpu_model(rand_in_dpu)

    # Perform quantization on the DPU part only
    quantizer_dpu = torch_quantizer(args.quant_mode, dpu_model, rand_in_dpu)
    quantized_model_dpu = quantizer_dpu.quant_model
    quantized_model_dpu = quantized_model_dpu.to(device)

    # Perform quantization on the CPU part only
    quantizer_cpu = torch_quantizer(args.quant_mode, cpu_model, rand_in_cpu)
    quantized_model_cpu = quantizer_cpu.quant_model
    quantized_model_cpu = quantized_model_cpu.to(device)

    # Export xmodel for both DPU and CPU parts
    if args.quant_mode == 'calib':
        # Export quantization configuration
        quantizer_dpu.export_quant_config()
        quantizer_cpu.export_quant_config()

    elif args.quant_mode == 'test':
        # Export xmodel
        quantizer_dpu.export_xmodel(output_dir=args.output_dir + '/dpu', deploy_check=True)
        quantizer_cpu.export_xmodel(output_dir=args.output_dir + '/cpu', deploy_check=True)


if __name__ == "__main__":
    args = get_parser().parse_args()
    quant(args)