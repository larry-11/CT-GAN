from tqdm import trange
from torch.nn import functional as F
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils import *
import numpy as np
import random
import types
from ops import compute_grad_gp_wgan, compute_grad_gp
import torchvision.utils as vutils


def trainSinGAN(seg, data_loader, networks, opts, stage, args, additional):
    torch.set_default_tensor_type(torch.DoubleTensor)
    # avg meter
    d_losses = AverageMeter()
    g_losses = AverageMeter()

    # set nets
    D = networks[0]
    G = networks[1]
    # set opts
    d_opt = opts['d_opt']
    g_opt = opts['g_opt']
    # switch to train mode
    D.train()
    G.train()
    # summary writer
    train_it = iter(data_loader)
    total_iter = 1000 * (stage + 1)
    decay_lr = 800 * (stage + 1)

    d_iter = 3
    g_iter = 3

    t_train = trange(0, total_iter, initial=0, total=total_iter)

    z_rec = additional['z_rec']

    for z_idx in range(len(z_rec)):
        z_rec[z_idx] = z_rec[z_idx].cuda(args.gpu, non_blocking=True)

    x_in = next(train_it).unsqueeze(1)
    # trilinear
    x_in = x_in.cuda(args.gpu, non_blocking=True)
    x_org = x_in
    x_in = F.interpolate(x_in, (int(args.size_list[stage]/6), args.size_list[stage], args.size_list[stage]), mode='trilinear', align_corners=True)

    x_in_list = [x_in]
    for xidx in range(1, stage + 1):
        x_tmp = F.interpolate(x_org, (int(args.size_list[xidx]/6), args.size_list[xidx], args.size_list[xidx]), mode='trilinear', align_corners=True)
        x_in_list.append(x_tmp)
    # target cube list

    for i in t_train:
        rec = open(os.path.join(args.log_dir, "loss_{}.txt".format(str(stage))), "a")

        if i == decay_lr:
            for param_group in d_opt.param_groups:
                    param_group['lr'] *= 0.1
            for param_group in g_opt.param_groups:
                    param_group['lr'] *= 0.1

        for _ in range(g_iter):
            g_opt.zero_grad()
            x_rec_list = G(z_rec)
            # reconstruction loss 
            g_rec = F.mse_loss(x_rec_list[-1], x_in)
            # projection loss
            l_ax = F.mse_loss(x_rec_list[-1][0][0][0,:,:], x_in[0][0][0,:,:])
            l_co = F.mse_loss(x_rec_list[-1][0][0][:,0,:], x_in[0][0][:,0,:])
            l_sa = F.mse_loss(x_rec_list[-1][0][0][:,:,0], x_in[0][0][:,:,0])
            g_pro = (l_ax + l_co + l_sa) / 3
            # seg loss
            start_x = int((random.random()/2)*args.size_list[stage])
            start_y = int((random.random()/2)*args.size_list[stage])
            start_z = int((random.random()/12)*args.size_list[stage])
            end_x = start_x + int(args.size_list[stage]/2)
            end_y = start_y + int(args.size_list[stage]/2)
            end_z = start_z + int(args.size_list[stage]/12)
            x_rec_slice = x_rec_list[-1][:,:,start_z:end_z,start_x:end_x,start_y:end_y]
            x_in_slice  = x_in[:,:,start_z:end_z,start_x:end_x,start_y:end_y]
            x_rec_slice = F.interpolate(x_rec_slice, (16, 96, 96), mode='trilinear', align_corners=True).detach().cpu().numpy()
            x_in_slice = F.interpolate(x_in_slice, (16, 96, 96), mode='trilinear', align_corners=True).detach().cpu().numpy()
            x_rec_seg = seg.prediction(x_rec_slice[0][0]*255)
            x_in_seg = seg.prediction(x_in_slice[0][0]*255)
            x_rec_seg = np.multiply(x_rec_seg, 1.0 / 255.0)
            x_in_seg  = np.multiply(x_in_seg, 1.0 / 255.0)
            x_rec_seg = torch.from_numpy(x_rec_seg).cuda(args.gpu)
            x_in_seg = torch.from_numpy(x_in_seg).cuda(args.gpu)
            g_seg = F.mse_loss(x_rec_seg, x_in_seg)
            # calculate rmse for each scale
            rmse_list = [1.0]
            for rmseidx in range(1, stage + 1):
                rmse = torch.sqrt(F.mse_loss(x_rec_list[rmseidx], x_in_list[rmseidx]))
                rmse_list.append(rmse)
            if (i+1)%250==0:
                np.save(file="{}/gen_stage{}_iter{}.npy".format(args.log_dir, str(stage), str(i)), arr=x_rec_list[-1].cpu().detach().numpy())
                np.save(file="{}/inp_stage{}_iter{}.npy".format(args.log_dir, str(stage), str(i)), arr=x_in.cpu().detach().numpy())

            z_list = [rmse_list[z_idx] * torch.randn(args.batch_size, 1, int(args.size_list[z_idx]/6), args.size_list[z_idx],
                                               args.size_list[z_idx]).cuda(args.gpu, non_blocking=True) for z_idx in range(stage + 1)]

            x_fake_list = G(z_list)
            g_fake_logit = D(x_fake_list[-1])
            ones = torch.ones((args.batch_size, 1)).cuda(args.gpu)

            if args.gantype == 'wgangp':
                # wgan gp
                g_fake = -torch.mean(g_fake_logit, (2, 3))
                g_loss = g_fake + 10.0 * g_rec + 10.0 * g_pro + 10 * g_seg
                
            elif args.gantype == 'zerogp':
                # zero centered GP
                g_fake = F.binary_cross_entropy_with_logits(g_fake_logit, ones, reduction='none').mean()
                g_loss = g_fake + 100.0 * g_rec + 100.0 * g_pro + 10 * g_seg

            elif args.gantype == 'lsgan':
                # lsgan
                g_fake = F.mse_loss(torch.mean(g_fake_logit, (2, 3, 4)), 0.9 * ones)
                g_loss = g_fake + 100.0 * g_rec + 100.0 * g_pro + 10 * g_seg


            g_loss.backward()
            g_opt.step()

            g_losses.update(g_loss.item(), x_in.size(0))

        # Update discriminator
        for _ in range(d_iter):
            x_in.requires_grad = True

            d_opt.zero_grad()
            x_fake_list = G(z_list)

            d_fake_logit = D(x_fake_list[-1].detach())
            d_real_logit = D(x_in)

            ones = torch.ones((args.batch_size, 1)).cuda(args.gpu)
            zeros = torch.zeros((args.batch_size, 1)).cuda(args.gpu)

            if args.gantype == 'wgangp':
                # wgan gp
                d_fake = torch.mean(d_fake_logit, (2, 3))
                d_real = -torch.mean(d_real_logit, (2, 3))
                d_gp = compute_grad_gp_wgan(D, x_in, x_fake_list[-1], args.gpu)
                d_loss = d_real + d_fake + 0.1 * d_gp
                
            elif args.gantype == 'zerogp':
                # zero centered GP
                d_fake = F.binary_cross_entropy_with_logits(d_fake_logit, zeros, reduction='none').mean()
                d_real = F.binary_cross_entropy_with_logits(d_real_logit, ones, reduction='none').mean()
                d_gp = compute_grad_gp(torch.mean(d_real_logit, (2, 3)), x_in)
                d_loss = d_real + d_fake + 10.0 * d_gp

            elif args.gantype == 'lsgan':
                # lsgan
                d_fake = F.mse_loss(torch.mean(d_fake_logit, (2, 3, 4)), zeros)
                d_real = F.mse_loss(torch.mean(d_real_logit, (2, 3, 4)), ones)
                d_loss = d_real + d_fake

            d_loss.backward()
            d_opt.step()

            d_losses.update(d_loss.item(), x_in.size(0))

        rec.write('Iter: {} Loss: D {d_losses.avg:.5f} G {g_losses.avg:.5f} G_re {g_losses1:.5f} G_pro {g_losses2:.5f} G_seg {g_losses3:.5f}\n'
                                .format(str(i), d_losses=d_losses, g_losses=g_losses, g_losses1=g_rec, g_losses2=g_pro, g_losses3=g_seg))
        rec.close()
        t_train.set_description('Stage: [{}/{}] Avg Loss: D[{d_losses.avg:.3f}] G[{g_losses.avg:.3f}] RMSE[{rmse:.3f}]'
                                .format(stage, args.num_scale, d_losses=d_losses, g_losses=g_losses, rmse=rmse_list[-1]))
