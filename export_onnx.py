import argparse
import torch.onnx
import train
import os



parser = argparse.ArgumentParser(description="PyTorch BraTS2019")
#parser.add_argument("--test_path", default="", type=str, help="path to train data")
parser.add_argument("--name", default="test", type=str, help="experiment name")
parser.add_argument("--models_path", default="/models", type=str, help="path to models folder")
parser.add_argument("--input_size", default=(128, 128, 128), help="Input image size", nargs="+", type=int)


def main():
    opt = parser.parse_args()
    print(torch.__version__)
    print(opt)

    trainer = train.Trainer(name=opt.name, models_root=opt.models_path, rewrite=False, connect_tb=False)
    trainer.load_best()
    trainer.model = trainer.model.module.cpu()
    trainer.model = trainer.model.train(False)
    trainer.state.cuda = False

    x = torch.randn(1, 1, opt.input_size[0], opt.input_size[1], opt.input_size[2], requires_grad=True)


    torch_out = torch.onnx.export(trainer.model,  # model being run
                                  [x,],  # model input (or a tuple for multiple inputs)
                                  os.path.join(opt.models_path, opt.name, opt.name+"_export.onnx"),  # where to save the model (can be a file or file-like object)
                                  export_params=True,
                                  verbose=True)  # store the trained parameter weights inside the model file

if __name__ == "__main__":
    main()


