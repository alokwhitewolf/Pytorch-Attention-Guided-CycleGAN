import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import ImageDataset
from models import Gen, Dis, Attn
from losses import realTargetLoss, fakeTargetLoss, cycleLoss

from torchutils import toZeroThreshold, weights_init, Plotter, save_checkpoint
import itertools
from PIL import Image

cudaAvailable = False
if torch.cuda.is_available():
    cudaAvailable = True
Tensor = torch.cuda.FloatTensor if cudaAvailable else torch.Tensor

genA2B = Gen() 
genB2A = Gen()
disA = Dis()
disB = Dis()
AttnA = Attn()
AttnB = Attn()

genA2B.apply(weights_init)
genB2A.apply(weights_init)
disA.apply(weights_init)
disB.apply(weights_init)
AttnA.apply(weights_init)
AttnB.apply(weights_init)

if cudaAvailable:
    genA2B.cuda()
    genB2A.cuda()

    disA.cuda()
    disB.cuda()

    AttnA.cuda()
    AttnB.cuda()
optG = torch.optim.Adam(itertools.chain(genA2B.parameters(), genB2A.parameters()),lr=0.0001, betas=(0.9, 0.999))
optD = torch.optim.Adam(itertools.chain(disA.parameters(), disB.parameters()),lr=0.0001, betas=(0.9, 0.999))
optAttn = torch.optim.Adam(itertools.chain(AttnA.parameters(), AttnB.parameters()),lr=0.0001, betas=(0.9, 0.999))

lrScheduler = torch.optim.lr_scheduler.MultiStepLR(optAttn, milestones=[30], gamma=0.01)
# attributes to plot and its freq
attributes =[('AdvLossA', 1),
            ('AdvLossB', 1),
            ('LossCycleA', 1),
            ('LossCycleB', 1),
            ('DisLossA', 1),
            ('DisLossB', 1)
            ]
plotter = Plotter(attributes)
dataroot = 'datasets/apple2orange/'
batchSize = 1
n_cpu = 4
size = 256

transforms_ = [ transforms.Resize(int(size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

dataloader = DataLoader(ImageDataset(dataroot, transforms_=transforms_, unaligned=True), 
    batch_size=batchSize, shuffle=True, num_workers=n_cpu)
startEpoch = 0
nofEpoch = 100

plotEvery = 1
saveEvery = 2

resume = False
if resume:
    pass

passDisWhole = True
for epoch in range(startEpoch, nofEpoch):
    # reset counters for logging & plotting 
    if epoch >=30:
     passDisWhole = False
    print('epoch -- >', epoch)
    countAdvLossA = 0.0
    countAdvLossB = 0.0
    countLossCycleA = 0.0
    countLossCycleB = 0.0
    countDisLossA = 0.0
    countDisLossB = 0.0
    for i, batch in enumerate(dataloader):
        if i % 100 == 0:
            print(i)
        realA, realB = batch['A'].type(Tensor), batch['B'].type(Tensor)

        # optgen zero
        # optattn zero 
        optG.zero_grad()
        optAttn.zero_grad()
        
        # A --> A'' 
        attnMapA = toZeroThreshold(AttnA(realA))
        fgA = attnMapA * realA
        bgA = (1 - attnMapA) * realA
        genB = genA2B(fgA) 
        fakeB = (attnMapA * genB) + bgA
        fakeBcopy = fakeB.clone()
        attnMapfakeB = toZeroThreshold(AttnB(fakeB))
        fgfakeB = attnMapfakeB * fakeB
        bgfakeB = (1 - attnMapfakeB) * fakeB
        genA_ = genB2A(fgfakeB)
        A_ = (attnMapfakeB * genA_) + bgfakeB

        # B --> B''
        attnMapB = toZeroThreshold(AttnB(realB))
        fgB = attnMapB * realB
        bgB = (1 - attnMapB) * realB
        genA = genB2A(fgB) 
        fakeA = (attnMapB * genA) + bgB
        fakeAcopy = fakeA.clone()
        attnMapfakeA = toZeroThreshold(AttnA(fakeA))
        fgfakeA = attnMapfakeA * fakeA
        bgfakeA = (1 - attnMapfakeA) * fakeA
        genB_ = genA2B(fgfakeA)
        B_ = (attnMapfakeA * genB_) + bgfakeA

        # Gen , Attn and cyclic loss
        if passDisWhole:
            AdvLossA = realTargetLoss(disA(fakeA)) + realTargetLoss(disA(A_))
            AdvLossB = realTargetLoss(disB(fakeB)) + realTargetLoss(disB(B_))
        else:
            AdvLossA = realTargetLoss(disA(genA)) + realTargetLoss(disA(genA_))
            AdvLossB = realTargetLoss(disB(genB)) + realTargetLoss(disB(genB_))
        
        LossCycleA = cycleLoss(realA, A_) 
        LossCycleB = cycleLoss(realB, B_) 
        totalloss = AdvLossA + AdvLossB + LossCycleA + LossCycleB


        totalloss.backward(retain_graph=True)
        optG.step()
        optAttn.step()

        # Dis Loss and update
        optD.zero_grad()
        if passDisWhole:
            DisLossA = fakeTargetLoss(disA(fakeA)) + fakeTargetLoss(disA(A_)) + 2*realTargetLoss(disA(realA))
            DisLossB = fakeTargetLoss(disB(fakeB)) + fakeTargetLoss(disB(B_)) + 2*realTargetLoss(disA(realB))
        else:
            DisLossA = fakeTargetLoss(disA(genA)) + fakeTargetLoss(disA(genA_)) + 2*realTargetLoss(disA(realA))
            DisLossB = fakeTargetLoss(disB(genB)) + fakeTargetLoss(disB(genB_)) + 2*realTargetLoss(disA(realB))

        totalloss = DisLossA + DisLossB


        totalloss.backward()
        optD.step()
        
        # update counter
        countAdvLossA += AdvLossA.item()
        countAdvLossB += AdvLossB.item()
        countLossCycleA += LossCycleA.item()
        countLossCycleB += LossCycleB.item()
        countDisLossA += DisLossA.item()
        countDisLossB += DisLossB.item()

    plotter.log('AdvLossA', countAdvLossA / (i + 1))
    plotter.log('AdvLossB', countAdvLossB / (i + 1))
    plotter.log('LossCycleA', countLossCycleA / (i + 1))
    plotter.log('LossCycleB', countLossCycleB / (i + 1))
    plotter.log('DisLossA', countDisLossA / (i + 1))
    plotter.log('DisLossB', countDisLossB / (i + 1))
    
    if (epoch + 1) % plotEvery == 0:
        plotter.plot('AdvLosses', ['AdvLossA', 'AdvLossB'], filename='AdvLosses.png')
        plotter.plot('CycleLosses', ['LossCycleA', 'LossCycleB'], filename='CycleLosses.png', ymax=1.0)
        plotter.plot('DisLosses', ['DisLossA', 'DisLossB'], filename='DisLosses.png')

    if (epoch + 1) % saveEvery == 0:
        save_checkpoint({
            'epoch' : epoch + 1,
            'optG' : optG.state_dict(),
            'optD' : optD.state_dict(),
            'optAttn' : optAttn.state_dict(),
            'plotter' : plotter,
            'genA2B' : genA2B.state_dict(),
            'genB2A' : genB2A.state_dict(),
            'disA' : disA.state_dict(),
            'disB' : disB.state_dict(),
            'AttnA' : AttnA.state_dict(),
            'AttnB' : AttnB.state_dict()
            }, 
            filename='models/checkpoint_'+str(epoch)+'.pth.tar'
            )

    lrScheduler.step()