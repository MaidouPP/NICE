#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import nice
import numpy as np
import torch
import torchvision as tv
import utils

from torch.autograd import Variable

kMNISTInputDim = 784
kMNISTInputSize = 28
kMNISTNumExamples = 100


def ShowImagesInGrid(data, rows, cols, save_path="original_images.png"):
    plt.axis('off')
    fig = plt.figure(figsize=(rows, cols))
    for i in range(1, cols * rows + 1):
        img = data[i-1].reshape((kMNISTInputSize, kMNISTInputSize))
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)
    # plt.show(block=True)
    if save_path is not None:
        plt.savefig(save_path)

def PrepareMNISTData(dataset):
    # dataset = mnist_dataset(root='./data')
    x = np.zeros((kMNISTNumExamples, kMNISTInputDim))
    for i in range(10):
        idx = dataset.train_labels==i
        # Loop on the digits
        for j in range(10):
            # Loop on the different shadowing methods
            x[j * 10 + i] = dataset.train_data[idx][j].reshape((-1,))

    m_val = np.ones((kMNISTNumExamples, kMNISTInputDim))
    # Get masks
    m_val[:10, :392] = 0
    m_val[10:20, 392:] = 0
    m_val[20:30, ::2] = 0
    m_val[30:40, 1::2] = 0
    for i in range(28):
        m_val[40:50, (i*28):((2*i+1)*14)] = 0
    for i in range(28):
        m_val[50:60, ((2*i+1)*14):((i+1)*28)] = 0
    m_val[60:70, 196:588] = 0
    for i in range(28):
        m_val[70:80, ((4*i+1)*7):((4*i+3)*7)] = 0
    m_val[80:90] = np.random.binomial(n=1, p=.25, size=(10,784))
    m_val[90:] = np.random.binomial(n=1, p=.1, size=(10,784))

    ShowImagesInGrid(x, 10, 10)
    after = np.multiply(m_val, x)
    ShowImagesInGrid(after, 10, 10, save_path="after.png")
    return m_val, x

def Reconstruct(mask, mask_val, x, flow, iters=300, lr=0.001, save_path=None):
    device = torch.device("cuda:0")
    x_mixed = np.where(mask==1, x, mask_val)
    i = 0
    mean = torch.load('./statistics/mnist_mean.pt')
    x_mixed = np.reshape(x_mixed, (kMNISTNumExamples, 1, kMNISTInputSize, kMNISTInputSize))
    x_mixed_var = Variable(torch.Tensor(x_mixed), requires_grad=True)
    x_mixed_tensor = utils.prepare_data(
        x_mixed_var, 'mnist', zca=None, mean=mean).to(device)
    inputs = Variable(x_mixed_tensor, requires_grad=True)
    # inputs = Variable(torch.Tensor(x_mixed).cuda(), requires_grad=True)
    lr_ = np.float64(lr)
    while i < iters:
        # print("iter: ", i)
        loss = flow(inputs).mean()
        loss.backward()
        inc = lr_ * inputs.grad
        # print(inputs.grad.data)
        inputs[mask!=1].data += inc[mask!=1]
        i += 1
    if save_path is not None:
        result = inputs.detach().reshape((-1, 1, kMNISTInputSize, kMNISTInputSize))
        tv.utils.save_image(tv.utils.make_grid(result.cpu()), save_path+str(iters)+".png")
    return inputs.detach().cpu().numpy()

def main(args):
    transform  = tv.transforms.Compose([tv.transforms.Grayscale(num_output_channels=1),
                                        tv.transforms.ToTensor()])
    trainset = tv.datasets.MNIST(root='~/torch/data/MNIST',
                                 train=True, download=True, transform=transform)
    prior = utils.StandardLogistic()
    device = torch.device("cuda:0")
    flow = nice.NICE(prior=utils.StandardLogistic(),
                coupling=4,
                in_out_dim=kMNISTInputDim,
                mid_dim=1000,
                hidden=5,
                mask_config=1).to(device)

    mask, x = PrepareMNISTData(trainset)
    ShowImagesInGrid(x, 10, 10, save_path="original.png")
    flow.load_state_dict(torch.load(args.model_path)['model_state_dict'])

    mask_val = np.random.uniform(size=(kMNISTNumExamples, kMNISTInputDim))
    mask_val = np.multiply(mask_val, x)
    iters = args.iters
    result = Reconstruct(mask, mask_val, x, flow, iters=iters, save_path="./inpainting/mnist_")
    ShowImagesInGrid(result, 10, 10, save_path="reconstructed_"+ str(iters) + ".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('MNIST NICE PyTorch inpainting experiment.')
    parser.add_argument('--model_path',
                        help='Saved model path.',
                        type=str,
                        default='./models/mnist/mnist_bs200_logistic_cp4_md1000_hd5_iter25000.tar')
    parser.add_argument('--iters',
                        help='Number of iterations.',
                        type=int,
                        default=300)
    args = parser.parse_args()
    main(args)
