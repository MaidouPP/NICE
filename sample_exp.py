#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


def VisualizeWeights(wts):
    plt.plot(wts)
    plt.title("Sorted rescaling layer weights")
    plt.xlabel("Order of magnitude of weight")
    plt.ylabel("Value of weight")
    plt.savefig("weights_scaling.png")

def GetScaleMask(scales, ktop=100, reverse=False):
    print(ktop, reverse)
    if not reverse:
        # Pick the largest ktop Sdd (least important latent vars)
        idxs = torch.argsort(scales)[::1]
        weights = scales
        weights, _ = torch.sort(weights)
        weights = weights.cpu().numpy()[0]
        print(weights)
        VisualizeWeights(weights)
    else:
        # Pick the smallest ktop Sdd (most important latent vars)
        idxs = torch.argsort(scales)
    idxs = idxs[:, ktop]
    mask = torch.ones((kMNISTInputDim,))
    mask[idxs] = 0.0
    return mask

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

    # mask, x = PrepareMNISTData(trainset)
    # ShowImagesInGrid(x, 10, 10, save_path="original.png")
    scaling_weights = torch.load(args.model_path)['model_state_dict']['scaling.scale']
    flow.load_state_dict(torch.load(args.model_path)['model_state_dict'])

    # Sort the scales
    mask = GetScaleMask(scaling_weights, ktop=args.ktop, reverse=True)
    samples = flow.sample(args.sample_size, mask=mask.cuda()).cpu()
    mean = torch.load('./statistics/mnist_mean.pt')
    result = utils.prepare_data(
        samples, 'mnist', zca=None, mean=mean, reverse=True)
    tv.utils.save_image(tv.utils.make_grid(result),
                        './samples_masked/result_true_'+str(args.ktop)+'.png')


    mask = GetScaleMask(scaling_weights, ktop=args.ktop, reverse=False)
    samples = flow.sample(args.sample_size).cpu()
    mean = torch.load('./statistics/mnist_mean.pt')
    result = utils.prepare_data(
        samples, 'mnist', zca=None, mean=mean, reverse=True)
    tv.utils.save_image(tv.utils.make_grid(result),
                        './samples_masked/result_false_'+str(args.ktop)+'.png')

    samples = flow.sample(args.sample_size).cpu()
    mean = torch.load('./statistics/mnist_mean.pt')
    result = utils.prepare_data(
        samples, 'mnist', zca=None, mean=mean, reverse=True)
    tv.utils.save_image(tv.utils.make_grid(result),
                        './samples_masked/result_original_'+str(args.ktop)+'.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('MNIST NICE PyTorch inpainting experiment.')
    parser.add_argument('--model_path',
                        help='Saved model path.',
                        type=str,
                        default='./models/mnist/mnist_bs200_logistic_cp4_md1000_hd5_iter25000.tar')
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--ktop',
                        help='Least top n latent variables',
                        type=int,
                        default=100)
    # parser.add_argument('--reverse',
    #                     help='Whether reverse the sort order',
    #                     type=lambda x: (str(x).lower() == 'true'),
    #                     default=False)
    args = parser.parse_args()
    main(args)
