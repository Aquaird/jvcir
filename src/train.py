import torch
from torch.autograd import Variable

import numpy as np
from utils.utils import AverageMeter
from utils.eval import Accuracy, getPreds, MPJPE, pointMPJPE
from utils.debugger import Debugger
import cv2
import ref
from progress.bar import Bar

def step(split, epoch, opt, dataLoader, model, criterion, optimizer = None):
    if split == 'train':
        model.train()
    else:
        model.eval()
    Loss, Acc, Mpjpe, Loss3D = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  
    nIters = len(dataLoader)
    bar = Bar('==>', max=nIters)
  
    for i, (input, target2D, target3D, meta) in enumerate(dataLoader):
        input_var = torch.autograd.Variable(input).float().cuda()
        target2D_var = torch.autograd.Variable(target2D).float().cuda()
        target3D_var = torch.autograd.Variable(target3D).float().cuda()
    
        output = model(input_var)
        reg = output[opt.nStack]
        if opt.DEBUG >= 2:
            gt = getPreds(target2D.cpu().numpy()) * 4
            pred = getPreds((output[opt.nStack - 1].data).cpu().numpy()) * 4
            debugger = Debugger()
            debugger.addImg((input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8))
            debugger.addPoint2D(pred[0], (255, 0, 0))
            debugger.addPoint2D(gt[0], (0, 0, 255))
            debugger.showImg()
            debugger.saveImg('debug/{}.png'.format(i))

        loss = opt.regWeight / 16 * criterion(reg, target3D_var[:,:,2])
        Loss3D.update(loss.data.item(), input.size(0))
        for k in range(opt.nStack):
            loss += criterion(output[k], target2D_var)

        Loss.update(loss.data.item(), input.size(0))
        Acc.update(Accuracy((output[opt.nStack - 1].data).cpu().numpy(), (target2D_var.data).cpu().numpy()))
        mpjpe, num3D = MPJPE((output[opt.nStack - 1].data).cpu().numpy(), (reg.data).cpu().numpy(), meta)
        if num3D > 0:
            Mpjpe.update(mpjpe, num3D)
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
        Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Loss3D {loss3d.avg:.6f} | Acc {Acc.avg:.6f} | Mpjpe {Mpjpe.avg:.6f} ({Mpjpe.val:.6f})'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = split, Mpjpe=Mpjpe, loss3d = Loss3D)
        bar.next()

    bar.finish()
    return Loss.avg, Acc.avg, Mpjpe.avg, Loss3D.avg
  
def step_gan(split, epoch, opt, realDataLoader, fakeDataLoader, generator, discriminator, criterion_2d, optimizer_G=None, optimizer_D=None):
    if split == 'train':
        generator.train()
        discriminator.train()
    else:
        generator.eval()
        discriminator.eval()

    LossG, LossD, Loss2D, Acc, Mpjpe  = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
  
    nIters = len(dataLoader)
    bar = Bar('==>', max=nIters)
  
    for i, (real, meta) in enumerate(realdataLoader):
        real_3d = torch.autograd.Variable(real).float().cuda()
        # generate a batch of fake 3d pose
        (z, z_target2D, z_target3D, meta) = next(iter((fakeDataLoader)))
        z_var = Variable(z).float().cuda()
        z_target2d_var = Variable(z_target2D).float().cuda()
        z_target3d_var = Variable(z_target3D).float().cuda()

        fake_3d = generator(z_var)[opt.nStack+3].detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_3d)) + torch.mean(discriminator(fake_3d))
        LossD.update(loss_D)
        if split == 'train':
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

        if i % opt.n_critic == 0:
            # train generator
            output = generator(z_var)
            gen_3d = output[opt.nStack+3]
            loss_G = -torch.mean(discriminator(gen_3d))
            LossG.update(loss_G)
            loss_2d = 0
            for k in range(opt.nStack):
                loss_2d += criterion_2d(output[k], z_target2D_var)
            Loss2D.update(loss_2d.data.item(), z.size(0))
            Acc.update(Accuracy((output[opt.nStack - 1].data).cpu().numpy(), (z_target2D_var.data).cpu().numpy()))
            mpjpe, num3D = pointMPJPE(gen_3d.data.cpu().numpy(), meta)
            if num3D > 0:
                Mpjpe.update(mpjpe, num3D)

            if split == 'train':
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

        Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | LossG {lossG.avg:.6f} | LossD {lossD.avg:.6f} | Loss2D {loss2D.avg:.6f} | Acc {Acc.avg:.6f} | Mpjpe {Mpjpe.avg:.6f} ({Mpjpe.val:.6f})'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, lossG=LossG, lossD=LossD, loss2D=Loss2D
                , Acc=Acc, split = split, Mpjpe=Mpjpe)
        bar.next()

    bar.finish()
    return LossG.avg, LossD.avg, Loss2D.avg,  Acc.avg, Mpjpe.avg
 

def train(epoch, opt, train_loader, model, criterion, optimizer):
    return step('train', epoch, opt, train_loader, model, criterion, optimizer)
  
def val(epoch, opt, val_loader, model, criterion):
    return step('val', epoch, opt, val_loader, model, criterion)

def train_gan(epoch, opt, real_loader, fake_loader,  generator, discriminator, criterion_2d, optimizer_G, optimizer_D):
    return step_gan('train', epoch, opt, real_loader, fake_loader,  generator, discriminator, criterion_2d, optimizer_G, optimizer_D)

def val_gan(epoch, opt, real_loader, fake_loader, generator, discriminator, criterion_2d):
    return step_gan('train', epoch, opt, real_loader, fake_loader,  generator, discriminator, criterion_2d)
