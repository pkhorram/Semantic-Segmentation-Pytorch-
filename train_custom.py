from torchvision import utils
from custom import *
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import math
from tqdm import tqdm
import gc
import os

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.xavier_uniform_(m.bias.data.view(m.bias.data.shape[0],1))
        #a = math.sqrt(3) * math.sqrt(2/m.bias.data.shape[0])
        #torch.nn.init._no_grad_uniform_(m.bias.data, -a, a)
        
        

    
def train(model, criterion, epochs, train_loader, val_loader, test_loader, use_gpu, name):
    
    #Create non-existing logfiles
    logname = 'logfile.txt'
    i = 1
    if os.path.exists('logfile.txt') == True:
        
        logname = 'logfile' + str(i) + '.txt'
        while os.path.exists('logfile' + str(i) + '.txt'):
            i+=1
            logname = 'logfile' + str(i) + '.txt'

    print('Loading results to logfile: ' + logname)
    with open(logname, "a") as file:
        file.write("Lofile DATA: Validation Loss and Accuracy\n") 
    
    logname_summary = 'logfile' + str(i) + '_summary.txt'    
    print('Loading Summary to : ' + logname_summary) 
    
    
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    if use_gpu:
        device = torch.device("cuda:0")
        model = torch.nn.DataParallel(model)
        model.to(device)
        
        
    
    val_loss_set = []
    val_acc_set = []
    val_iou_set = []
    
    
    training_loss = []
    
    # Early Stop criteria
    minLoss = 1e6
    minLossIdx = 0
    earliestStopEpoch = 10
    earlyStopDelta = 5
    for epoch in range(epochs):
        ts = time.time()

                
                  
        for iter, (inputs, tar, labels) in tqdm(enumerate(train_loader)):

            optimizer.zero_grad()
            del tar
            
            if use_gpu:
                inputs = inputs.to(device)# Move your inputs onto the gpu
                labels = labels.to(device) # Move your labels onto the gpu
            
                
            outputs = model(inputs)
            del inputs
            loss = criterion(outputs, Variable(labels.long()))
            del labels
            del outputs

            loss.backward()
            loss = loss#.item()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        
        # calculate val loss each epoch
        val_loss, val_acc, val_iou = val(model, val_loader, criterion, use_gpu)
        val_loss_set.append(val_loss)
        val_acc_set.append(val_acc)
        val_iou_set.append(val_iou)
        
        print("epoch {}, time {}, train loss {}, val loss {}, val acc {}, val iou {}".format(epoch, time.time() - ts,
                                                                                                loss.item(), val_loss,
                                                                                                val_acc,
                                                                                                val_iou))        
        training_loss.append(loss.item())
        
        torch.save(model, 'weights_custom/epoch-{}'.format(epoch))
        
        with open(logname, "a") as file:
            file.write("writing!\n")
            file.write("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
            file.write("\n training Loss:   " + str(loss.item()))
            file.write("\n Validation Loss: " + str(val_loss_set[-1]))
            file.write("\n Validation acc:  " + str(val_acc_set[-1]))
            file.write("\n Validation iou:  " + str(val_iou_set[-1]) + "\n ")                                             
                                                                                                
                                                                                                
        
        # Early stopping
        if val_loss < minLoss:
            # Store new best
            torch.save(model, name)
            minLoss = val_loss#.item()
            minLossIdx = epoch
            
        # If passed min threshold, and no new min has been reached for delta epochs
        elif epoch > earliestStopEpoch and (epoch - minLossIdx) > earlyStopDelta:
            print("Stopping early at {}".format(minLossIdx))
            break
        # TODO what is this for?
        #model.train()

        
        
    with open(logname_summary, "a") as file:
            file.write("Summary!\n")
            file.write("Stopped early at {}".format(minLossIdx))
            file.write("\n training Loss:   " + str(training_loss))        
            file.write("\n Validation Loss: " + str(val_loss_set))
            file.write("\n Validation acc:  " + str(val_acc_set))
            file.write("\n Validation iou:  " + str(val_iou_set) + "\n ")
            
        
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
        
    for iter, (X, tar, Y) in tqdm(enumerate(val_loader)):
        
        if not IOU_init:
            IOU_init = True
            IOU = np.zeros((1,19))
            
        if use_gpu:
            inputs = X.to(device)
            labels = Y.to(device)
            
        else:
            inputs, labels = X, Y

            
        with torch.no_grad():   
            outputs = model(inputs)    
            loss.append(criterion(outputs, labels.long()).item())
            prediction = softmax(outputs) 
            acc.append(pixel_acc(prediction, labels))
            IOU = IOU + np.array(iou(prediction, labels))
        
    
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
    
def checkM():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

if __name__ == "__main__":
    train_dataset = CityScapesDataset(csv_file='train.csv',resizing=True)
    val_dataset = CityScapesDataset(csv_file='val.csv',resizing=True)
    test_dataset = CityScapesDataset(csv_file='test.csv')
    train_loader = DataLoader(dataset=train_dataset,
                          batch_size=2,
                          num_workers=8,
                          shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                          batch_size=2,
                          num_workers=8,
                          shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                          batch_size=2,
                          num_workers=8,
                          shuffle=True)
    
    
    epochs     = 100
    criterion = torch.nn.CrossEntropyLoss()
    # Fix magic number
    model = Custom(n_class=34)
    model.apply(init_weights)
    
    
    epochs     = 100
    use_gpu = torch.cuda.is_available()

    train(model, criterion, epochs, train_loader, val_loader, test_loader, use_gpu, "Custom")
    
    
    model.load_state_dict(torch.load('./save_param'))