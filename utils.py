import cv2
import os
from tqdm import tqdm
from torchvision import datasets
import torch
from torch.autograd import Variable
import random

# Grayscale preprocessing
def color2gray(root,save):
    n = 1
    file_list = os.listdir(root)
    if not os.path.isdir(save):
        os.makedirs(save)
    for f in tqdm(file_list):
        rgb_img = cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(save, str(n) + '.png'), rgb_img)
        n+=1

# color2gray(os.path.join('./data/tgt_data/..'), os.path.join('./data/src_data/..'))
# print('color2gray already done')

def data_load(path, subfolder, transform, batch_size, shuffle=False, drop_last=True):
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]

    n = 0
    for i in range(dset.__len__()):
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1
        n += 1

    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# def gradient_penalty(disc, real_data, generated_data):
#
#     batch_size = real_data.size(0)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     epsilon = torch.rand(batch_size, 1, device=device, requires_grad=True)
#     epsilon_exp = epsilon.expand(batch_size, real_data.nelement() / batch_size).contiguous().view(real_data.size())
#
#     interpolates = epsilon_exp * real_data + (1 - epsilon_exp) * generated_data
#
#     disc_interpolates = disc(interpolates)
#
#     gradients = autograd.grad(outputs=disc_interpolates.mean(),
#                               inputs=interpolates,
#                               create_graph=True, retain_graph=True
#                               )[0]
#
#     gradients = gradients.view(gradients.size(0), -1)
#
#     return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


def Gray(tensor, size, batch_size):
    R = tensor[:, 0, :, :]
    G = tensor[:, 1, :, :]
    B = tensor[:, 2, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B

    return tensor.view(batch_size, 1, size, size)
