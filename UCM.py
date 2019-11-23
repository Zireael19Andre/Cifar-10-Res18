import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from ResNET import ResNet18
from torch.optim.lr_scheduler import _LRScheduler
from collections import Counter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from DatasetALLtype import ImageFolder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



parser = argparse.ArgumentParser(description='UC_Merced Training')
parser.add_argument('--outf', default='/home/marulab/PycharmProjects/dl', help='folder to output images and model checkpoints')
parser.add_argument('--net', default='/home/marulab/PycharmProjects/dl/Resnet18.pth', help="path to net (to continue training)")
args = parser.parse_args()


EPOCH =135
pre_epoch = 0
BATCH_SIZE =6
train_transform = transforms.Compose([
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

Train_Dataset=ImageFolder(root='/home/marulab/PycharmProjects/Dataset/UC_Merced/Train',transform=train_transform)
trainloader=DataLoader(Train_Dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

Test_Dataset=ImageFolder(root='/home/marulab/PycharmProjects/Dataset/UC_Merced/Test',transform=test_transform)
testloader=DataLoader(Test_Dataset, batch_size=4, shuffle=False, num_workers=2)



net = ResNet18().to(device)


Loss_list= []
Accuarcy_list= []
'''
class MultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]
'''

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0)
'''
milestones=[70,135,180]
scheduler=MultiStepLR(optimizer,milestones,gamma=0.1, last_epoch=-1)
'''


if __name__ == "__main__":
    best_acc = 0
    print("Start Training, Resnet-18!")
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch,EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):

                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()



                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()



                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()


                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('Test Accuracy：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total



                    Loss_list.append(sum_loss / (len(testloader)))
                    Accuarcy_list.append(100 * acc / (len(testloader)))

                    x1 = range(pre_epoch, EPOCH)
                    x2 = range(pre_epoch, EPOCH)
                    y1 = Accuarcy_list
                    y2 = Loss_list

                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth'%  (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write("EPOCH=%03d,Loss= %.03f" % (epoch + 1, sum_loss))
                    f.write('\n')
                    f.flush()

                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

plt.subplot(2,1,1)
plt.plot(x1,y1,'o-')
plt.title('Test Accuracy vs. Epoches')
plt.ylabel('Test Accuracy')
plt.subplot(2,1,2)
plt.plot(x2,y2,'.-')
plt.xlabel('Test Loss vs. Epoches')
plt.ylabel('Test Loss')
plt.show()
plt.savefig("Accuracy_Loss.jpg")