import torch.optim as optim
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms

from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt

import scipy.misc
import math
import time


def image_loader(image_name):
    image = Image.open(image_name)
    img_shape = (image.size[1], image.size[0])
    print('image size (HxW): ', img_shape)
    image = Variable(loader(image))
    image = image.unsqueeze(0)  # Transform to 4 Dim Tensor From 3 Dim
    return image

def save_image(input, path):
    image = input.data.clone().cpu()
    image = image.view(3, imsize, image.size()[3])
    image = unloader(image)
    scipy.misc.imsave(path, image)

imsize = 224

loader = transforms.Compose([
    transforms.Resize((imsize,imsize)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

unloader = transforms.ToPILImage()

# CUDA configurations
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# Content and shape
shape = image_loader("imgs/h.jpg").type(dtype)   # shape reference image
content = image_loader("imgs/u.jpg").type(dtype) # content reference image
pastiche = image_loader("imgs/u.jpg").type(dtype)# optimized image
pastiche.data = torch.randn(pastiche.data.size()).type(dtype)

pastiche = nn.Parameter(pastiche.data)
content_layers = ['conv_4']
shape_layers = ['pool_1']
loss = nn.MSELoss()
loss_network = models.vgg19_bn(pretrained=True)
optimizer = optim.Adam([pastiche], lr=0.08)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

use_cuda = torch.cuda.is_available()
if use_cuda:
    loss_network.cuda()

# Loss
shape_weight = 100
content_weight = 1

patch_div = 4
angle_div = 12
distance_div = 5

def get_dist(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def get_angle(p1, p2):
    return math.degrees(math.atan2(p2[0] - p1[0], p2[1] - p1[1])) + 180.0

def embedding(tensor, sample_idx):
    tensor = tensor.squeeze(0)
    tensor = tensor.mean(dim=0)

    tensor_mean = tensor.mean()
    tensor = torch.pow(tensor - tensor_mean.detach(), 2)

    descriptor = Variable(torch.zeros(patch_div * patch_div * angle_div * distance_div),
                          requires_grad=True).type(dtype)
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            for k, sample in enumerate(sample_idx):
                distance = get_dist([i, j], sample)
                if distance >= 32 or distance < 1:
                    continue
                d = int(math.log2(distance))
                angle = get_angle([i, j], sample)
                a = int((angle - 0.001) / 30)
                descriptor[k * angle_div * distance_div + d * angle_div + a] += tensor[i, j]

    plt.figure(figsize=(25, 25))
    plt.imshow(tensor.cpu().detach().numpy(), aspect='equal', cmap='viridis')
    plt.colorbar()
    plt.savefig("generated/pool1-%d.png" % (time.time()))
    plt.close()
    return descriptor

# Shape reference
i = 0
not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer

sample_idx = torch.LongTensor(patch_div * patch_div, 2)
shape_descriptor = Variable(torch.zeros(patch_div * patch_div * angle_div * distance_div)).type(dtype)

for layer in list(loss_network.features):
    layer = not_inplace(layer)
    if use_cuda:
        layer.cuda()
    shape = layer.forward(shape)
    if isinstance(layer, nn.MaxPool2d):
        i += 1
        name = "pool_" + str(i)
        if name in shape_layers:
            tensor = shape.squeeze(0)
            tensor = tensor.mean(dim=0)

            patch_size = int(tensor.shape[0] / patch_div)
            indices = [i * patch_size for i in range(patch_div)]
            partition = torch.Tensor(patch_div * patch_div, patch_size, patch_size)
            for i in range(patch_div):
                for j in range(patch_div):
                    partition[patch_div * i + j] = tensor[indices[i]:indices[i] + patch_size,
                                                   indices[j]:indices[j] + patch_size]
            partition = partition.unsqueeze(0)
            partition_mean = partition.mean()
            partition = torch.abs(partition - partition_mean)
            partition = F.avg_pool2d(partition, kernel_size=2, stride=2)
            partition = partition.view(patch_div * patch_div, -1)
            _, index = torch.max(partition, dim=1)
            for i, idx in enumerate(index):
                sample_idx[i] = torch.LongTensor([int(i / patch_div) * patch_size + int(idx / (patch_size / 2)) * 2,
                                                  (i % patch_div) * patch_size + idx % (patch_size / 2) * 2])
            tensor_mean = tensor.mean()
            tensor = torch.pow(tensor - tensor_mean.detach(), 2)

            for i in range(tensor.shape[0]):
                for j in range(tensor.shape[1]):
                    for k, sample in enumerate(sample_idx):
                        distance = get_dist([i, j], sample)
                        if distance >= 32 or distance < 1:
                            continue
                        d = int(math.log2(distance))
                        angle = get_angle([i, j], sample)
                        a = int((angle - 0.001) / 30)
                        shape_descriptor[k * angle_div * distance_div + d * angle_div + a] += tensor[i, j]
            # visualize 'pool_1' average feature map
            plt.figure(figsize=(25, 25))
            plt.imshow(tensor.cpu().detach().numpy(), aspect='equal', cmap='viridis')
            plt.colorbar()
            plt.savefig("generated/pool1-%d.png" % (time.time()))
            plt.close()
            break

    if isinstance(layer, nn.ReLU):
        i += 0

# Content reference
i = 0
for layer in list(loss_network.features):
    layer = not_inplace(layer)
    if use_cuda:
        layer.cuda()
    content = layer.forward(content)
    if isinstance(layer, nn.Conv2d):
        name = "conv_" + str(i)
        if name in content_layers:
            break
    if isinstance(layer, nn.ReLU):
        i += 1

# Trainer
num_epochs = 50

for i in range(1, num_epochs+1):
    path = "generated/pastiche-%d.png" % (i - 1)
    pastiche.data.clamp_(0, 1)
    save_image(pastiche, path)

    time_start = time.time()

    optimizer.zero_grad()

    j = 0
    k = 0
    shape_loss = 0
    content_loss = 0

    pastiche_features = pastiche
    for layer in list(loss_network.features):
        layer = not_inplace(layer)
        if use_cuda:
            layer.cuda()

        pastiche_features = layer.forward(pastiche_features)

        if isinstance(layer, nn.MaxPool2d):
            j += 1
            name = "pool_" + str(j)
            if name in shape_layers:
                pastiche_descriptor = embedding(pastiche_features, sample_idx)
                shape_loss += loss(pastiche_descriptor, shape_descriptor.detach())
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(k)
            if name in content_layers:
                content_loss += loss(pastiche_features, content.detach())
        if isinstance(layer, nn.ReLU):
            k += 1

    total_loss = shape_weight * shape_loss + content_weight * content_loss
    total_loss.backward()
    print(total_loss.data.cpu().numpy())
    time_end = time.time()
    print('Time', time_end - time_start)

    optimizer.step()

    scheduler.step()

    print("Iteration: %d" % (i))

    if i == num_epochs:
        path = "generated/pastiche-%d.png" % (i)
        pastiche.data.clamp_(0, 1)
        save_image(pastiche, path)

