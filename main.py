from torchvision import utils
from basic_fcn import *
from dataloader import *
from transfer import Transfer
from utils import *
import torchvision
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import math
from tqdm import tqdm
import gc

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.xavier_uniform_(m.bias.data.view(m.bias.data.shape[0],1))
        #a = math.sqrt(3) * math.sqrt(2/m.bias.data.shape[0])
        #torch.nn.init._no_grad_uniform_(m.bias.data, -a, a)
        
        
def print_log(s):
    log = open('unet.log', 'a')
    log.write(s+'\n')
    log.close()
    print(s)

    
def train(model, criterion, epochs, train_loader, val_loader, test_loader, use_gpu, name, debug=False):
    if debug:
        initMem = {}
        print("Initialized with")
        initMem = checkM(initMem)
        
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    if use_gpu:
        device = torch.device("cuda:0")
        model = torch.nn.DataParallel(model)
        model.to(device)
    
    if debug:
        print("Use GPU:  ")
        initMem = checkM(initMem)
        
    
    val_loss_set = []
    val_acc_set = []
    val_iou_set = []
    
    start_epoch = 14
    
    # Early Stop criteria
    minLoss = 1e6
    minLossIdx = 0
    earliestStopEpoch = 20
    earlyStopDelta = 7
    for epoch in range(start_epoch, epochs):
        ts = time.time()
             
                  
        if debug:
            singleM = {}
            
        for iter, (inputs, tar, labels) in enumerate(train_loader):
            
            #print(inputs.shape)
            optimizer.zero_grad()
            #print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            #startMem = checkM(startMem)
            #singleM = checkM(singleM)
            
            del tar
            
            if use_gpu:
                inputs = inputs.to(device)# Move your inputs onto the gpu
                labels = labels.to(device) # Move your labels onto the gpu
            
                
#             print('inputs shape: ', inputs.shape)
            outputs = model(inputs)
            del inputs
            loss = criterion(outputs, Variable(labels.long()))
            del labels
            del outputs
            
            if debug:
                print("\n**********************************************\nPost Loss")
                #backMem = checkM(backMem, True)
                singleM = checkM(singleM)
                #print("start vs back diff")
                #memDiff(startMem, backMem)
            loss.backward()
            loss = loss.item()
            
            if debug:
                print("\n**********************************************\nPost Backward")
                #postLossMem = checkM(postLossMem, True)
                #print("Post loss vs back diff")
                singleM = checkM(singleM)
                #memDiff(backMem, postLossMem)
                
            optimizer.step()
            
            if debug:
                print("\n**********************************************\nPost Step")
                singleM = checkM(singleM)
                #finalMem = checkM(finalMem, True)
                #memDiff(postLossMem, finalMem)
            

            if iter % 10 == 0:
                print_log("epoch{}, iter{}, loss: {}".format(epoch, iter, loss))
            
            if debug:    
                print("\n**********************************************\nFinal")
                singleM = checkM(singleM)
            
                print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
        
        
        # calculate val loss each epoch
        val_loss, val_acc, val_iou = val(model, val_loader, criterion, use_gpu)
        val_loss_set.append(val_loss)
        val_acc_set.append(val_acc)
        val_iou_set.append(val_iou)
        
        print_log("epoch {}, time {}, train loss {}, val loss {}, val acc {}, val iou {}".format(epoch, time.time() - ts,
                                                                                                loss, val_loss,
                                                                                                val_acc,
                                                                                                val_iou))
        
        torch.save(model, 'weights_transfer/epoch-{}'.format(epoch))
        
        # Early stopping
        if val_loss < minLoss:
            # Store new best
            torch.save(model, name)
            minLoss = val_loss
            minLossIdx = epoch
            
        # If passed min threshold, and no new min has been reached for delta epochs
        elif epoch > earliestStopEpoch and (epoch - minLossIdx) > earlyStopDelta:
            print("Stopping early at {}".format(minLossIdx))
            break
        # TODO what is this for?
        #model.train()
        
    return val_loss_set, val_acc_set, val_iou_set


def val(model, val_loader, criterion, use_gpu):
    
    # set to evaluation mode 
    model.eval()

    softmax = nn.Softmax(dim = 1)
    
    loss = []
    pred = []
    acc = []
    
    IOU_init = False
    if use_gpu:
        device = torch.device("cuda:0")
        
        #model.to(device)
        
    for iter, (inputs, tar, labels) in tqdm(enumerate(val_loader)):
        
        if not IOU_init:
            IOU_init = True
            IOU = np.zeros((1,19))
        del tar
        
        if use_gpu:
            inputs = inputs.to(device)
            labels = labels.to(device)
            

            
        with torch.no_grad():   
            outputs = model(inputs)  
            del inputs
            loss.append(criterion(outputs, labels.long()).item())
            prediction = softmax(outputs) 
            del outputs
            acc.append(pixel_acc(prediction, labels))
            IOU = IOU + np.array(iou(prediction, labels))
            del prediction
            del labels
        
    
    acc = sum(acc)/len(acc)
    avg_loss = sum(loss)/len(loss) 
    IOU = IOU/iter  
    
    return avg_loss, acc, IOU      
       
    
    
    
def test(model, use_gpu):
    
    softmax = nn.Softmax(dim = 1)
    
    pred = []
    acc = []
    if use_gpu:
        device = torch.device("cuda:0")
        
        model.to(device)
    
    IOU_init = False
    for iter, (X, tar, Y) in enumerate(test_loader):
        
        if not IOU_init:
            IOU_init = True
            IOU = np.zeros((1,tar.shape[1]))
        
        if use_gpu:
            inputs = X.to(device)
            labels = Y.to(device)
        else:
            inputs, labels = X, Y
                    
        
        outputs = model(inputs)  
        
        prediction = softmax(outputs)
        acc.append(pixel_acc(prediction, labels))
        IOU = IOU + np.array(iou(prediction, Y))
        
    acc = sum(acc)/len(acc)        
    IOU = IOU/iter

    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    
    return acc, IOU
    
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
        
    def forward(self, x):
        news = self.shape.copy()
        news[0] =  x.shape[0]
        
#         print('xshape: ', x.shape)
        
        return x.view(*tuple(news))

def getTransferModel(n_class, batch_size):
    
    decoder = nn.Sequential(
        Reshape([batch_size,512,16,32]),
        
        nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        ),
        
        nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ),
            
        nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ),
            
        nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ),
        
            
        nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ),

        nn.Conv2d(32, n_class, kernel_size=1)
        )
    decoder.apply(init_weights)
        

    
    model = models.resnet34(pretrained=True)
        
    for param in model.parameters():
        # False implies no retraining
        param.requires_grad=False
        
    del param

    model.avgpool = nn.Identity() 
        
    model.fc = decoder
    #print(model)
    return model

def checkM(prev, q=False):
    out = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda and not q:
                    name = str(obj.size())
                    if name in out:
                        out[name] += 1
                    else:
                        out[name] = 1
                    
        except:
            pass
        
    for key in out:
        if key not in prev:
            print("new: " + key + " : " + str(out[key]))
        elif prev[key] != out[key]:
            #print("diff (new - old): " + key + " : " + str(out[key] - prev[key]))
            print("diff (new - old): " + key + " : " + str(out[key])+ " - " +str(prev[key]))
            
    for key in prev:
        if key not in out:
            print("dropped: " + key + " : " + str(prev[key]))
    return out

def memDiff(prev, out):
    for key in out:
        if key not in prev:
            print("new: " + key + " : " + str(out[key]))
        elif prev[key] != out[key]:
            print("diff (new - old): " + key + " : " + str(out[key] - prev[key]))
            
    for key in prev:
        if key not in out:
            print("dropped: " + key + " : " + str(prev[key]))

if __name__ == "__main__":
    batch_size = 3
    train_dataset = CityScapesDataset(csv_file='train.csv', resizing=True)
    val_dataset = CityScapesDataset(csv_file='val.csv', resizing=True)
    test_dataset = CityScapesDataset(csv_file='test.csv')
    train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          num_workers=8,
                          shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                          batch_size=batch_size,
                          num_workers=8,
                          shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          num_workers=4,
                          shuffle=True)
    
    
    epochs     = 100
    criterion = torch.nn.CrossEntropyLoss()
    # Fix magic number
#     model = getTransferModel(34, batch_size)
    model = torch.load('weights_transfer/epoch-13')
#     model = Transfer(34)
    
    
    epochs     = 100
    use_gpu = torch.cuda.is_available()
    train(model, criterion, epochs, train_loader, val_loader, test_loader, use_gpu, "Transfer")
    
    
    model.load_state_dict(torch.load('./save_param'))