from metrics import AverageNonzeroTripletsMetric, AccumulatedAccuracyMetric
# Strategies for selecting triplets within a minibatch
from utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector
from losses import OnlineTripletLoss
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from datasets import BalancedBatchSampler
from torch.optim import lr_scheduler
import torch.optim as optim
from trainer import fit
from torch.utils import data
import os

dim_embedding=10


# --------------------------------------------load CIFAR10-----------------------------------------
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
my_batch = 10

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=my_batch,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=my_batch,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --------------------------------------------if GPU-----------------------------------------
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# --------------------------------------------define Network-----------------------------------------
cfg = {
    'shallow': [32, 'M', 32, 'M', 'D', 32, 32, 'M', 'D'],
    'shallow2': [32, 32, 'M', 'D', 64, 64, 'M', 'D'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(4096, dim_embedding)
        # 512->64

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                # 2
            elif x == 'D':
                layers += [nn.Dropout(p=0.2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


# --------------------------------------------Embedding extraction-----------------------------------------


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), dim_embedding))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for i, (images, target) in enumerate(dataloader):
            embeddings[k:k+len(images)] = model(images.to(device)
                                                ).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


# --------------------------------------------Training-----------------------------------------
# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(torch.tensor(
    np.asarray(trainset.targets)), n_classes=10, n_samples=25)
test_batch_sampler = BalancedBatchSampler(torch.tensor(
    np.asarray(testset.targets)), n_classes=10, n_samples=25)

cuda = True
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(
    trainset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(
    testset, batch_sampler=test_batch_sampler, **kwargs)

cuda=device

# Set up the network and training parameters
'''
net = VGG('shallow')

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)

net.to(device)
'''


margin = .6
embedding_net = VGG('shallow2')
model = embedding_net
# if cuda:
#    model.cuda()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)

loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 150

fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler,
    n_epochs, cuda, log_interval, metrics=[AverageNonzeroTripletsMetric()])

# save our model
#PATH = './saved_models/shallow2_ran_1.pth'
#torch.save(model.state_dict(), PATH)

train_embeddings_ocl, train_labels_ocl = extract_embeddings(trainloader, model)
test_embeddings_ocl, test_labels_ocl = extract_embeddings(testloader, model)

my_dataset = data.TensorDataset(torch.Tensor(
    train_embeddings_ocl), torch.Tensor(train_labels_ocl))
trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=256,
                                          shuffle=True, num_workers=0)


class Final(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(dim_embedding, 10)

    def forward(self, x):
        out = self.classifier(x)
        return out


finLayer = Final().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(finLayer.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = finLayer(inputs.to(device))
        # batch_hard_triplet_loss(labels=labels,embeddings=outputs,margin=0.3)
        loss = criterion(outputs, labels.to(device, dtype=torch.long))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print('%d, loss: %.3f' %
          (epoch + 1, running_loss / len(trainloader)))

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = finLayer(model(images.to(device)))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
