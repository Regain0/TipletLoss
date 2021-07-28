from metrics import AverageNonzeroTripletsMetric, AccumulatedAccuracyMetric
# Strategies for selecting triplets within a minibatch
from utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector
from losses import OnlineTripletLoss
from networks import EmbeddingNet
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
import time
import matplotlib.pyplot as plt

dim_embedding=10

# --------------------------------------------load CIFAR10-----------------------------------------
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
my_batch = 256

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
    'shallow': [32,'M', 32, 'M', 'D', 32, 32, 'M', 'D'],
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
        self.classifier = nn.Linear(512, dim_embedding)
        #512->64

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
                #2
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


# --------------------------------------------Calculating-----------------------------------------
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
embedding_net = VGG('VGG11')
model = embedding_net
#if cuda:
#    model.cuda()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
model.to(device)

PATH = './saved_models/vgg11_ran_20.pth'
SD=torch.load(PATH)
model.load_state_dict(SD)

train_embeddings_ocl, train_labels_ocl = extract_embeddings(trainloader, model)

test_embeddings_ocl, test_labels_ocl = extract_embeddings(testloader, model)

MaxSize=50000

def cal_avg_dist(embeddings, labels, cnt_type):
    assert embeddings.shape[0]==labels.shape[0]
    cnt_sample=embeddings.shape[0]
    embedding_dimension=embeddings.shape[1]
    #print(embeddings.shape)
    #print(labels.shape)
    for i in range(cnt_type):
        a = np.zeros(shape=(MaxSize,embedding_dimension))
        cnt=0
        for j in range(cnt_sample):
            if(labels[j]==i):
                a[cnt]=embeddings[j]
                cnt+=1
        print("Class ",i)
        #print(a[0:cnt])
        total_dist=0
        for m in range(cnt):
            for n in range(m+1,cnt):
                total_dist+=np.linalg.norm(a[m]-a[n], ord=2, keepdims=False)
        
        if cnt<=1:
            avg_dist=0
        else:
            avg_dist=total_dist/(cnt*(cnt-1)/2)

        print("Avg dist:",avg_dist)

print(type(test_embeddings_ocl))
print(test_embeddings_ocl)
print(type(test_labels_ocl))
print(test_labels_ocl)

print("----------train set----------")
#cal_avg_dist(train_embeddings_ocl,train_labels_ocl,10)
print("----------test set----------")
#cal_avg_dist(test_embeddings_ocl,test_labels_ocl,10)

def dist_from_single(anchor,anchor_class, train_embeddings, train_labels,test_embeddings,test_labels, cnt_type):
    assert train_embeddings.shape[0] == train_labels.shape[0]
    assert test_embeddings.shape[0] == test_labels.shape[0]
    train_cnt_sample = train_embeddings.shape[0]
    train_embedding_dimension = train_embeddings.shape[1]
    test_cnt_sample = test_embeddings.shape[0]
    test_embedding_dimension = test_embeddings.shape[1]

    train_classified_sample = np.zeros(
        shape=(cnt_type, MaxSize, train_embedding_dimension))
    train_classified_cnt = [0 for i in range(10)]

    test_classified_sample = np.zeros(
        shape=(cnt_type, MaxSize, test_embedding_dimension))
    test_classified_cnt = [0 for i in range(10)]

    for i in range(train_cnt_sample):
        train_classified_sample[int(train_labels[i])][train_classified_cnt[int(train_labels[i])]] = train_embeddings[i]
        train_classified_cnt[int(train_labels[i])] += 1

    for i in range(test_cnt_sample):
        test_classified_sample[int(test_labels[i])][test_classified_cnt[int(test_labels[i])]] = test_embeddings[i]
        test_classified_cnt[int(test_labels[i])] += 1


    print("our selected anchor from class",anchor_class)

    print("-------------------from train set---------------------")

    for i in range(cnt_type):
        total_dist = 0
        min_dist = np.linalg.norm(
            train_classified_sample[i][0]-anchor, ord=2, keepdims=False)
        max_dist = np.linalg.norm(
            train_classified_sample[i][0]-anchor, ord=2, keepdims=False)

        for j in range(train_classified_cnt[i]):
            current_dist = np.linalg.norm(
                train_classified_sample[i][j]-anchor, ord=2, keepdims=False)
            total_dist += current_dist

            if(current_dist>max_dist):
                max_dist=current_dist
            if(current_dist<min_dist):
                min_dist=current_dist
                
        avg_dist = total_dist/train_classified_cnt[i]

        print("class",i)
        print("avg_dist from anchor",avg_dist)
        print("max_dist from anchor",max_dist)
        print("min_dist from anchor",min_dist)

    print("-------------------from test set---------------------")

    for i in range(cnt_type):
        total_dist = 0
        min_dist = np.linalg.norm(
            test_classified_sample[i][0]-anchor, ord=2, keepdims=False)
        max_dist = np.linalg.norm(
            test_classified_sample[i][0]-anchor, ord=2, keepdims=False)

        for j in range(test_classified_cnt[i]):
            current_dist = np.linalg.norm(
                test_classified_sample[i][j]-anchor, ord=2, keepdims=False)
            total_dist += current_dist

            if(current_dist>max_dist):
                max_dist=current_dist
            if(current_dist<min_dist):
                min_dist=current_dist
                
        avg_dist = total_dist/test_classified_cnt[i]

        print("class",i)
        print("avg_dist from anchor",avg_dist)
        print("max_dist from anchor",max_dist)
        print("min_dist from anchor",min_dist)

'''
print("----------one point picked from train set----------")

anchor=train_embeddings_ocl[0]
anchor_class=train_labels_ocl[0]
dist_from_single(anchor,anchor_class,train_embeddings_ocl,train_labels_ocl,test_embeddings_ocl,test_labels_ocl,10)


print("----------one point picked from test set----------")
time_start=time.time()
anchor=test_embeddings_ocl[0]
anchor_class=test_labels_ocl[0]
dist_from_single(anchor,anchor_class,train_embeddings_ocl,train_labels_ocl,test_embeddings_ocl,test_labels_ocl,10)
time_end=time.time()
print('totally cost',time_end-time_start)

'''
def max_dist_distribution(chosen_class,sample_embeddings, sample_labels,target_embeddings,target_labels, cnt_type,image_name):
    assert sample_embeddings.shape[0] == sample_labels.shape[0]
    assert target_embeddings.shape[0] == target_labels.shape[0]
    sample_cnt_sample = sample_embeddings.shape[0]
    sample_embedding_dimension = sample_embeddings.shape[1]
    target_cnt_sample = target_embeddings.shape[0]
    target_embedding_dimension = target_embeddings.shape[1]


    sample_classified_sample = np.zeros(
        shape=(cnt_type, MaxSize, sample_embedding_dimension))
    sample_classified_cnt = [0 for i in range(10)]

    target_classified_sample = np.zeros(
        shape=(cnt_type, MaxSize, target_embedding_dimension))
    target_classified_cnt = [0 for i in range(10)]

    for i in range(sample_cnt_sample):
        sample_classified_sample[int(sample_labels[i])][sample_classified_cnt[int(sample_labels[i])]] = sample_embeddings[i]
        sample_classified_cnt[int(sample_labels[i])] += 1

    for i in range(target_cnt_sample):
        target_classified_sample[int(target_labels[i])][target_classified_cnt[int(target_labels[i])]] = target_embeddings[i]
        target_classified_cnt[int(target_labels[i])] += 1


    print("our selected anchor from class",chosen_class)

    list_max_dist=[]
    for i in range(sample_classified_cnt[chosen_class]):
        anchor=sample_classified_sample[chosen_class][i]
        max_dist = np.linalg.norm(
            target_classified_sample[chosen_class][0]-anchor, ord=2, keepdims=False)
        for j in range(target_classified_cnt[chosen_class]):
            current_dist = np.linalg.norm(
                target_classified_sample[chosen_class][j]-anchor, ord=2, keepdims=False)
            if(current_dist>max_dist):
                max_dist=current_dist
        list_max_dist.append(max_dist)
    #print(list_max_dist)
    np_max_dist = np.array(list_max_dist)

    my_bin=[]
    for i in range(31):
        my_bin.append(0.1*i)

    plt.cla
    plt.hist(np_max_dist, bins =  my_bin) 
    plt.title("histogram") 
    plt.savefig("./saved_pics/"+image_name)
    plt.show()
    return list_max_dist

def avg_dist_distribution(chosen_class,sample_embeddings, sample_labels,target_embeddings,target_labels, cnt_type,image_name):
    assert sample_embeddings.shape[0] == sample_labels.shape[0]
    assert target_embeddings.shape[0] == target_labels.shape[0]
    sample_cnt_sample = sample_embeddings.shape[0]
    sample_embedding_dimension = sample_embeddings.shape[1]
    target_cnt_sample = target_embeddings.shape[0]
    target_embedding_dimension = target_embeddings.shape[1]


    sample_classified_sample = np.zeros(
        shape=(cnt_type, MaxSize, sample_embedding_dimension))
    sample_classified_cnt = [0 for i in range(10)]

    target_classified_sample = np.zeros(
        shape=(cnt_type, MaxSize, target_embedding_dimension))
    target_classified_cnt = [0 for i in range(10)]

    for i in range(sample_cnt_sample):
        sample_classified_sample[int(sample_labels[i])][sample_classified_cnt[int(sample_labels[i])]] = sample_embeddings[i]
        sample_classified_cnt[int(sample_labels[i])] += 1

    for i in range(target_cnt_sample):
        target_classified_sample[int(target_labels[i])][target_classified_cnt[int(target_labels[i])]] = target_embeddings[i]
        target_classified_cnt[int(target_labels[i])] += 1


    print("our selected anchor from class",chosen_class)

    list_avg_dist=[]
    for i in range(sample_classified_cnt[chosen_class]):
        anchor=sample_classified_sample[chosen_class][i]
        total_dist=0
        for j in range(target_classified_cnt[chosen_class]):
            current_dist = np.linalg.norm(
                target_classified_sample[chosen_class][j]-anchor, ord=2, keepdims=False)
            total_dist+=current_dist
        avg_dist=total_dist/target_classified_cnt[chosen_class]
        list_avg_dist.append(avg_dist)
    #print(list_max_dist)
    np_avg_dist = np.array(list_avg_dist)

    my_bin=[]
    for i in range(31):
        my_bin.append(0.1*i)

    plt.hist(np_avg_dist, bins =  my_bin) 
    plt.title("histogram") 
    plt.savefig("./saved_pics/"+image_name)
    plt.show()

#max_dist_distribution(0,train_embeddings_ocl,train_labels_ocl,train_embeddings_ocl,train_labels_ocl,10,"class_0_sample_train_to_train.jpg")
#max_dist_distribution(0,test_embeddings_ocl,test_labels_ocl,train_embeddings_ocl,train_labels_ocl,10,"class_0_sample_test_to_train.jpg")

#max_dist_distribution(0,train_embeddings_ocl,train_labels_ocl,test_embeddings_ocl,test_labels_ocl,10,"class_0_sample_train_to_test.jpg")
#max_dist_distribution(0,test_embeddings_ocl,test_labels_ocl,test_embeddings_ocl,test_labels_ocl,10,"class_0_sample_test_to_test.jpg")


avg_dist_distribution(0,train_embeddings_ocl,train_labels_ocl,test_embeddings_ocl,test_labels_ocl,10,"class_0_sample_train_to_test.jpg")
avg_dist_distribution(0,train_embeddings_ocl,train_labels_ocl,train_embeddings_ocl,train_labels_ocl,10,"class_0_sample_train_to_train.jpg")
#avg_dist_distribution(0,test_embeddings_ocl,test_labels_ocl,test_embeddings_ocl,test_labels_ocl,10,"class_0_sample_test_to_test.jpg")
#avg_dist_distribution(0,test_embeddings_ocl,test_labels_ocl,train_embeddings_ocl,train_labels_ocl,10,"class_0_sample_test_to_train.jpg")



