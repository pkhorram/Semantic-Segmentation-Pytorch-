from torchvision import utils
from basic_fcn import *
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import math
from tqdm import tqdm

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.xavier_uniform_(m.bias.data.view(m.bias.data.shape[0],1))
        #a = math.sqrt(3) * math.sqrt(2/m.bias.data.shape[0])
        #torch.nn.init._no_grad_uniform_(m.bias.data, -a, a)
        
        
def print_log(s):
    log = open('baseline.log', 'a')
    log.write(s+'\n')
    log.close()
    print(s)
    
def train(model, criterion, epochs, train_loader, val_loader, test_loader, use_gpu, name):

    optimizer = optim.Adam(fcn_model.parameters(), lr=5e-3)
    if use_gpu:
        device = torch.device("cuda:0")
#         model = torch.nn.DataParallel(model)
        model.to(device)
        
        
    
    val_loss_set = []
    val_acc_set = []
    val_iou_set = []
    
    # Early Stop criteria
    minLoss = 1e6
    minLossIdx = 0
    earliestStopEpoch = 10
    earlyStopDelta = 5
    for epoch in range(epochs):
        ts = time.time()
        
        #print(np.array(val_loss).shape)
        # early-stopping 
#         if epoch > 11:
#             if val_loss[-1] < val_loss[-10]:
#                 open('save_param', 'w').close()
#                 torch.save(fcn_model.state_dict(), 'save_param')
                
                  
        for iter, (X, tar, Y) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            
            
            if use_gpu:
                inputs = X.to(device)# Move your inputs onto the gpu
                labels = Y.to(device) # Move your labels onto the gpu
            else:
                inputs, labels = X, Y # Unpack variables into inputs and labels
                
            outputs = model(inputs)
#             loss = criterion(outputs, Variable(labels.long()))
            loss = criterion(outputs, labels.long())
            
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print_log("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        # calculate val loss each epoch
        val_loss, val_acc, val_iou = val(model, val_loader, criterion, use_gpu)
        val_loss_set.append(val_loss)
        val_acc_set.append(val_acc)
        val_iou_set.append(val_iou)
        
        print_log("epoch {}, time {}, train loss {}, val loss {}, val acc {}, val iou {}".format(epoch, time.time() - ts,
                                                                                                loss.item(), val_loss,
                                                                                                val_acc,
                                                                                                val_iou))
        
        torch.save(model, 'weights_baseline/epoch-{}'.format(epoch))
        
        # Early stopping
        if val_loss < minLoss:
            # Store new best
            torch.save(model, name)
            minLoss = val_loss
            minLossIdx = epoch
            
        # If passed min threshold, and no new min has been reached for delta epochs
        elif epoch > earliestStopEpoch and (epoch - minLossIdx) > earlyStopDelta:
            print_log("Stopping early at {}".format(minLossIdx))
            break
        # TODO what is this for?
        #model.train()
        
    return val_loss_set, val_acc_set, val_iou_set, predictions


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
            IOU = np.zeros((1,tar.shape[1]))
            
        if use_gpu:
            inputs = X.to(device)
            labels = Y.to(device)
            
        else:
            inputs, labels = X, Y

            
        with torch.no_grad():   
            outputs = model(inputs)    
            loss.append(criterion(outputs, labels.long()).item())
            prediction = softmax(outputs) 
            acc.append(pixel_acc(prediction, labels).item())
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
                    
        
        outputs = fcn_model(inputs)  
        
        prediction = softmax(outputs)
        acc.append(pixel_acc(prediction, labels))
        IOU = IOU + np.array(iou(prediction, Y))
        
    acc = sum(acc)/len(acc)        
    IOU = IOU/iter

    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    
    return acc, IOU
    
if __name__ == "__main__":
    train_dataset = CityScapesDataset(csv_file='train.csv')
    val_dataset = CityScapesDataset(csv_file='val.csv')
    test_dataset = CityScapesDataset(csv_file='test.csv')
    train_loader = DataLoader(dataset=train_dataset,
                          batch_size=2,
                          num_workers=4,
                          shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                          batch_size=2,
                          num_workers=4,
                          shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                          batch_size=3,
                          num_workers=4,
                          shuffle=True)
    
    
    epochs     = 100
    criterion = torch.nn.CrossEntropyLoss()
    # Fix magic number
    fcn_model = FCN(n_class=34)
    fcn_model.apply(init_weights)
    #fcn_model = torch.load('FCN')
    
    fcn_model = FCN(n_class=34)
    fcn_model.apply(init_weights)
    
    epochs     = 100
    use_gpu = torch.cuda.is_available()
#     if use_gpu:
#         device = torch.device("cuda:0")
#         fcn_model = torch.nn.DataParallel(fcn_model)
#         fcn_model.to(device)
#     val(fcn_model, val_loader, criterion, use_gpu)
    train(fcn_model, criterion, epochs, train_loader, val_loader, test_loader, use_gpu, "FCN")
    
    
    fcn_model.load_state_dict(torch.load('./save_param'))