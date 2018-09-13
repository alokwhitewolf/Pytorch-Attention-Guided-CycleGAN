import torch

cudaAvailable = False
if torch.cuda.is_available():
    cudaAvailable = True
Tensor = torch.cuda.FloatTensor if cudaAvailable else torch.Tensor

BCELoss = torch.nn.BCELoss()
L1Loss = torch.nn.L1Loss()

def realTargetLoss(x):
	target = Tensor(x.shape[0], 1).fill_(1.0)
	return BCELoss(x, target)

def fakeTargetLoss(x):
	target = Tensor(x.shape[0], 1).fill_(0.0)
	return BCELoss(x, target)

def cycleLoss(a, a_):
	loss = L1Loss(a, a_)
	return loss
