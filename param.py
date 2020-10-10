import argparse
parser = argparse.ArgumentParser()
# 输入输出参数
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--input_nc", type=int, default=1, help="number of image channels")
parser.add_argument("--output_nc", type=int, default=128, help="number of output channels")

# 训练参数
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")

# 网络参数设置
parser.add_argument("--num_downs", type=int, default=5, help="number of down_sample")
parser.add_argument("--ngf", type=int, default=256, help="the number of filters in the last conv layer")


# 记录参数
parser.add_argument("--savepoint", type=int, default=50, help="save network")


opt=parser.parse_args()