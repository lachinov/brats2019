import argparse
import torch.onnx
import os
from collections import OrderedDict
from torch.onnx.symbolic_registry import register_op
from torch.onnx.symbolic_helper import parse_args
from model import UNet

parser = argparse.ArgumentParser(description="PyTorch SegTHOR")
parser.add_argument("--name", default="test", type=str, help="experiment name")
parser.add_argument("--models_path", default="./models", type=str, help="path to models folder")
parser.add_argument("--input_size", default=(128, 128, 128), help="Input image size", nargs="+", type=int)

@parse_args('v', 'i', 'v', 'v', 'f', 'i')
def group_norm_symbolic(g, input, num_groups, weight, bias, eps, cudnn_enabled):
    return g.op("ExperimentalDetectronGroupNorm", input, weight, bias, num_groups_i=num_groups, eps_f=eps)

def main():
    opt = parser.parse_args()
    print(torch.__version__)
    print(opt)

    enc_layers = [1,2,2,4]
    dec_layers = [1,1,1,1]
    number_of_channels=[int(8*2**i) for i in range(1,1+len(enc_layers))]#[16,32,64,128]
    model = UNet(depth=len(enc_layers), encoder_layers=enc_layers, decoder_layers=dec_layers, number_of_channels=number_of_channels, number_of_outputs=3)
    s = torch.load(os.path.join(opt.models_path, opt.name, opt.name+'best_model.pth'), map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in s['model'].state_dict().items():
        name = k[7:] # remove 'module' word in the beginning of keys.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    x = torch.randn(1, 4, opt.input_size[0], opt.input_size[1], opt.input_size[2], requires_grad=True)

    register_op("group_norm", group_norm_symbolic, "", 10)

    torch_out = torch.onnx.export(model,  # model being run
                                  [x,],  # model input (or a tuple for multiple inputs)
                                  os.path.join(opt.models_path, opt.name, opt.name+".onnx"),  # where to save the model (can be a file or file-like object)
                                  export_params=True,
                                  verbose=True,  # store the trained parameter weights inside the model file
                                  opset_version=10
                                  )

if __name__ == "__main__":
    main()


