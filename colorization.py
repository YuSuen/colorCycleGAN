import argparse
import os
import time
import matplotlib.pyplot as plt
import pickle
from torchvision import transforms, models
import torch
import itertools
import torch.nn as nn
import utils
import network as net

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='color',  help='project name')
parser.add_argument('--src_data', required=False, default='src_data/',  help='src data path')
parser.add_argument('--tgt_data', required=False, default='tgt_data/',  help='tgt data path')
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--in_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--out_ndc', type=int, default=1, help='output channel for discriminator')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--train_epoch', type=int, default=500)
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
args = parser.parse_args()

print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# results save path
if not os.path.isdir(os.path.join(args.name + '_results', 'Colorization')):
    os.makedirs(os.path.join(args.name + '_results', 'Colorization'))

transform = transforms.Compose([
         transforms.Resize((args.input_size, args.input_size)),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
 ])

train_loader_src = utils.data_load(os.path.join('data', args.src_data), 'train', transform, args.batch_size, shuffle=True, drop_last=True)
train_loader_tgt = utils.data_load(os.path.join('data', args.tgt_data), 'train', transform, args.batch_size, shuffle=True, drop_last=True)
test_loader_src = utils.data_load(os.path.join('data', args.src_data), 'test', transform, 1, shuffle=True, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

A2BG = net.generator(args.in_ngc, args.out_ngc, args.ngf)
B2AG = net.generator(args.in_ngc, args.out_ngc, args.ngf)

AD = net.discriminator(args.in_ndc, args.out_ndc, args.ndf)
BD = net.discriminator(args.in_ndc, args.out_ndc, args.ndf)

print('---------- Networks initialized -------------')
utils.print_network(A2BG)
utils.print_network(AD)
print('-----------------------------------------------')

vgg16 = models.vgg16(pretrained=True)
vgg16 = net.VGG(vgg16.features[:23]).to(device)

A2BG.to(device)
B2AG.to(device)
AD.to(device)
BD.to(device)

A2BG.train()
B2AG.train()
AD.train()
BD.train()

# loss
criterion_GAN = nn.MSELoss().to(device)
criterion_cycle = nn.L1Loss().to(device)
criterion_identity = nn.L1Loss().to(device)
criterion_color = nn.L1Loss().to(device)

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad,A2BG.parameters()), filter(lambda p: p.requires_grad,B2AG.parameters())),lr=args.lrG, betas=(args.beta1, args.beta2))
optimizer_D_A = torch.optim.Adam(filter(lambda p: p.requires_grad,AD.parameters()), lr=args.lrD, betas=(args.beta1, args.beta2))
optimizer_D_B = torch.optim.Adam(filter(lambda p: p.requires_grad,BD.parameters()), lr=args.lrD, betas=(args.beta1, args.beta2))

lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_G, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
lr_scheduler_D_A = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_D_A, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
lr_scheduler_D_B = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_D_B, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor

target_real = torch.ones(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
target_fake = torch.zeros(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)

fake_A_buffer = utils.ReplayBuffer()
fake_B_buffer = utils.ReplayBuffer()

torch.backends.cudnn.benchmark = True
train_hist = {}
train_hist['G_loss'] = []
train_hist['G_identity_loss'] = []
train_hist['G_GAN_loss'] = []
train_hist['G_cycle_loss'] = []
train_hist['G_Color_loss'] = []
train_hist['D_loss'] = []
train_hist['per_epoch_time'] = []
train_hist['total_time'] = []

print('training start!')
start_time = time.time()

for epoch in range(args.train_epoch):
    epoch_start_time = time.time()
    G_losses = []
    G_identity_losses = []
    G_GAN_losses = []
    G_cycle_losses = []
    G_Color_losses = []
    D_losses = []

    for (x, _), (y, _)in zip(train_loader_src, train_loader_tgt):
        x, y= x.to(device), y.to(device)

        # Set model input
        real_A = x
        real_B = y

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()
        # Identity loss

        # G_A2B(B) should equal B if real B is fed
        same_B = A2BG(real_B)
        real_B_feature=vgg16(real_B)
        same_B_feature=vgg16(same_B)
        loss_identity_B = criterion_identity(same_B_feature[2], real_B_feature[2]) * 5.0
        # loss_identity_B = criterion_identity(same_B, real_B) * 5.0


        # G_B2A(A) should equal A if real A is fed
        same_A = B2AG(real_A)
        real_A_feature=vgg16(real_A)
        same_A_feature=vgg16(same_A)
        loss_identity_A = criterion_identity(same_A_feature[2], real_A_feature[2]) * 5.0
        # loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        loss_G_identity = loss_identity_A + loss_identity_B
        G_identity_losses.append(loss_G_identity.item())
        train_hist['G_identity_loss'].append(loss_G_identity.item())

        ###################################
        # GAN loss
        #A2B
        fake_B = A2BG(real_A)
        pred_fake = BD(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        # color loss
        # RA_Gray=utils.Gray(real_A,args.input_size,args.batch_size)
        RA_Gray = real_A[:, 0, :, :]
        FB_Gray = utils.Gray(fake_B,args.input_size,args.batch_size)
        color_loss_A2B = criterion_color(FB_Gray,RA_Gray) * 10.0

        #B2A
        fake_A = B2AG(real_B)
        pred_fake = AD(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # color loss
        RB_Gray = utils.Gray(real_B, args.input_size, args.batch_size)
        # FA_Gray = utils.Gray(fake_A, args.input_size, args.batch_size)
        FA_Gray = fake_A[:, 0, :, :]
        color_loss_B2A = criterion_color(FA_Gray, RB_Gray) * 10.0

        loss_G_GAN = loss_GAN_A2B + loss_GAN_B2A
        G_GAN_losses.append(loss_G_GAN.item())
        train_hist['G_GAN_loss'].append(loss_G_GAN.item())

        loss_Color = color_loss_B2A + color_loss_A2B
        G_Color_losses.append(loss_Color.item())
        train_hist['G_Color_loss'].append(loss_Color.item())

        ###################################

        # Cycle loss
        recovered_A = B2AG(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = A2BG(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        loss_G_cycle = loss_cycle_ABA + loss_cycle_BAB
        G_cycle_losses.append(loss_G_cycle.item())
        train_hist['G_cycle_loss'].append(loss_G_cycle.item())

        ###################################
        # Total loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + color_loss_B2A + color_loss_A2B + loss_identity_A + loss_identity_B
        loss_G.backward()

        G_losses.append(loss_G.item())
        train_hist['G_loss'].append(loss_G.item())
        optimizer_G.step()

        ###################################

        # train D

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = AD(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = AD(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward(retain_graph=True)

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######

        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = BD(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = BD(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward(retain_graph=True)

        loss_D = loss_D_A + loss_D_B

        D_losses.append(loss_D.item())
        train_hist['D_loss'].append(loss_D.item())


        optimizer_D_B.step()

    ###################################

    per_epoch_time = time.time() - epoch_start_time
    train_hist['per_epoch_time'].append(per_epoch_time)

    print(
        '[%d/%d] - time: %.2f, G_loss: %.3f, G_identity_loss: %.3f, G_GAN_loss: %.3f, G_cycle_loss: %.3f, D_loss: %.3f' % (
        (epoch + 1), args.train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(G_losses)), torch.mean(torch.FloatTensor(G_identity_losses)),
        torch.mean(torch.FloatTensor(G_GAN_losses)), torch.mean(torch.FloatTensor(G_cycle_losses)), torch.mean(torch.FloatTensor(D_losses))))

    with torch.no_grad():
        A2BG.eval()
        for n, (x, _) in enumerate(test_loader_src):
            x = x.to(device)
            G_recon = A2BG(x)
            result = torch.cat((x[0], G_recon[0]), 2)
            path = os.path.join(args.name + '_results', 'Colorization', str(epoch+1) + '_epoch_' + args.name + '_test_' + str(n) + '.png')
            plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)

    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(A2BG.state_dict(), os.path.join(args.name + '_results', 'A2BG_generator_latest.pkl'))
    torch.save(AD.state_dict(), os.path.join(args.name + '_results', 'AD_discriminator_latest.pkl'))
    torch.save(B2AG.state_dict(), os.path.join(args.name + '_results', 'B2AG_generator_latest.pkl'))
    torch.save(BD.state_dict(), os.path.join(args.name + '_results', 'BD_discriminator_latest.pkl'))

total_time = time.time() - start_time
train_hist['total_time'].append(total_time)

print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_time'])), args.train_epoch, total_time))
print("Training finish!... save training results")

torch.save(A2BG.state_dict(), os.path.join(args.name + '_results',  'A2BG_generator_param.pkl'))
torch.save(AD.state_dict(), os.path.join(args.name + '_results',  'AD_discriminator_param.pkl'))
torch.save(B2AG.state_dict(), os.path.join(args.name + '_results',  'B2AG_generator_param.pkl'))
torch.save(BD.state_dict(), os.path.join(args.name + '_results',  'BD_discriminator_param.pkl'))
with open(os.path.join(args.name + '_results',  'train_hist.pkl'), 'wb') as f:
    pickle.dump(train_hist, f)