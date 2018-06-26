import os
import time
import datetime

import torch
import torch.utils.data
from opts import opts
import ref
from models.hg_3d import HourglassNet3D
from models.dis_lstm import import Discriminator
from datasets.fusion import Fusion
from datasets.h36m import TRUE
from utils.logger import Logger
from train import train_gan, val_gan
import pickle
from functools import partial
pickle.load = partial(pickle.load, encoding='latin1')
pickle.Unpickler = partial(pickle.Unpickler, encoding='latin1')

def main():
    opt = opts().parse()
    now = datetime.datetime.now()
    logger = Logger(opt.saveDir + '/logs_{}'.format(now.isoformat()))

    if opt.loadGModel != 'none':
        generator = torch.load(opt.loadGModel, map_location=lambda storage, loc:storage, pickle_module=pickle).cuda()
    else:
        generator = HourglassNet3D(opt.nStack, opt.nModules, opt.nFeats, opt.nRegModules, opt.nCamModules).cuda()

    if opt.loadDModel != 'none':
        discriminator = torch.load(opt.loadDModel, map_location=lambda storage, loc:storage, pickle_module=pickle).cuda()
    else:
        discriminator = Discriminator(3, opt.sizeLSTM).cuda()

    criterion = torch.nn.MSELoss().cuda()
    optimizer_G = torch.optim.adam(generator.parameters(), opt.ganLR, eps=ref.epsilon, weight_decay=ref.ganWeightDecay)
    optimizer_D = torch.optim.RMSprop(dis.parameters(), opt.ganLR)

    val_real_loader = torch.utils.data.DataLoader(
            TRUE(opt, 'val'), 
            batch_size = 1, 
            shuffle = False,
            num_workers = int(ref.nThreads)
            )

    val_fake_loader = torch.utils.data.DataLoader(
            Fusion(opt, 'val'), 
            batch_size = 1, 
            shuffle = False,
            num_workers = int(ref.nThreads)
            )

    if opt.test:
        val_gan(0, opt, val_real_loader, val_fake_loadre, generator, discriminator, criterion)
        return

    train_real_loader = torch.utils.data.DataLoader(
            TRUE(opt, 'train'), 
            batch_size = opt.trainBatch, 
            shuffle = True if opt.DEBUG == 0 else False,
            num_workers = int(ref.nThreads)
            )

    train_fake_loader = torch.utils.data.DataLoader(
            Fusion(opt, 'train'), 
            batch_size = opt.trainBatch, 
            shuffle = True if opt.DEBUG == 0 else False,
            num_workers = int(ref.nThreads)
            )

    for epoch in range(1, opt.nEpochs + 1):
        lossg_train, lossd_train, loss2d_train, acc_train, mpjpe_train = train_gan(epoch, opt, train_real_loader, train_fake_loader, generator, discriminator, criterion, optimizer_G, optimizer_D)
        logger.scalar_summary('lossg_train', lossg_train, epoch)
        logger.scalar_summary('lossd_train', lossd_train, epoch)
        logger.scalar_summary('loss2d_train', loss2d_train, epoch)
        logger.scalar_summary('acc_train', acc_train, epoch)
        logger.scalar_summary('mpjpe_train', mpjpe_train, epoch)
        if epoch % opt.valIntervals == 0:
            lossg_val, lossd_val, loss2d_val, acc_val, mpjpe_val = val_gan(epoch, opt, val_real_loader, val_fake_loader, generator, discriminator, criterion)
            logger.scalar_summary('lossg_val', lossg_val, epoch)
            logger.scalar_summary('lossd_val', lossd_val, epoch)
            logger.scalar_summary('loss2d_val', loss2d_val, epoch)
            logger.scalar_summary('acc_val', acc_val, epoch)
            logger.scalar_summary('mpjpe_val', mpjpe_val, epoch)
            torch.save(model, os.path.join(opt.saveDir, 'model_{}.pth'.format(epoch)))
            logger.write('{:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f}\n'.format(lossg_train, lossd_train, loss2d_train, acc_train, mpjpe_train, lossg_val, lossd_val, loss2d_val, acc_val, mpjpe_val))
        else:
            logger.write('{:8f} {:8f} {:8f} {:8f} {:8f}\n'.format(lossg_train, lossd_train, loss2d_train, acc_train, mpjpe_train))

    logger.close()

if __name__ == '__main__':
    main()
