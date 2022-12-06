import os
from PIL import Image
import utils
import torch
import numpy as np
from model import Model, ClassifierModel
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

def data_prepare(train_path, test_path):
    """
        DATA PREPARATION (TrainSet and TestSet)
    """
    train_img = os.listdir(train_path)
    train_img.sort()
    trainset = []
    train_label = [0 for _ in range(len(train_img))]
    for num in train_img:
        img_path = os.path.join(train_path, num)
        image = Image.open(img_path)
        img = np.asarray(image)
        trainset.append(img)

    dir_name = os.listdir(test_path)
    testset = []
    test_label = []
    for class_id in dir_name:
        Catgs = os.path.join(test_path, class_id)
        ImageList = os.listdir(Catgs)
        for name in ImageList:
            img_path = os.path.join(Catgs, name)
            image = Image.open(img_path)
            img = np.asarray(image)
            testset.append(img)
            test_label.append(int(class_id))

    return trainset, train_label, testset, test_label

def train(model, data_loader, train_optimizer,lr_scheduler, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    model.train()
   
    total_train_loss = 0

    for trns_1, trns_2, _ in data_loader:
        trns_1, trns_2 = trns_1.to(device), trns_2.to(device)
        
        # get augmentation embedding
        _, out_1 = model(trns_1)
        _, out_2 = model(trns_2)
       
        loss = utils.xt_xent(out_1, out_2)

        total_train_loss += loss.item()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
    lr_scheduler.step(loss)
    
    avg_train_loss = total_train_loss / len(data_loader)
    print('Epoch:%3d' % epoch, '|Train Loss:%8.4f ' % avg_train_loss, end="")

    return model, avg_train_loss, train_optimizer.param_groups[0]['lr']

def test(model, data_loader,max_acc):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    with torch.no_grad():
        model.eval()
        all_emb = []
        all_cls = []
        
        for trns_1, _, label in data_loader:
            trns_1, label =  trns_1.to(device), label.to(device)

            output, _ = model(trns_1)
            all_emb.append(output)
            all_cls.append(label)

        embedding = torch.cat(all_emb, dim = 0)
        classes = torch.cat(all_cls, dim = 0)
        
        acc = utils.KNN(embedding, classes, batch_size=16)

        print("|Test Acc: %.5f" % acc)
        
        if acc > max_acc:
            max_acc = acc
            print("-------------saving model--------------")
            torch.save(model, "model.pth")
            
        return acc, max_acc

def downstram( train_data_loader, val_data_loader, test_data_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("model.pth")
    
    model = ClassifierModel(model)
    model = model.to(device)
    
    all_emb = []
    all_cls = []
    model.eval()
    for data,_, target in test_data_loader:
        data, target = data.to(device), target.to(device)
        output,_= model(data)
        all_emb.append(output)
        all_cls.append(target)

    embedding = torch.cat(all_emb, dim = 0)
    classes = torch.cat(all_cls, dim = 0)
    
    acc = utils.KNN(embedding, classes, batch_size=16)
    print('At the beginning: %3.4f' % acc)
    model.train()
    
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-5)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.0001, min_lr=0)
    loss_fn = nn.CrossEntropyLoss()
    
    epochs = 200

    min_val_loss = float("inf")
    print("Begin downstram task training...") 
    max_acc = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        total_val_loss = 0
        train_hit = 0
        val_hit = 0
    
        for data, _, target in train_data_loader:
            data, target = data.to(device), target.to(device)
            
            _,output = model(data)
            
            # loss function
            loss = loss_fn(output, target)

            total_train_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
           
            train_hit += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            # do back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        lr_scheduler.step(loss)

        with torch.no_grad():
            model.eval()
            all_emb = []
            all_cls = []
            for data,_, target in val_data_loader:
                data, target = data.to(device), target.to(device)
                output,_= model(data)
                all_emb.append(output)
                all_cls.append(target)

        embedding = torch.cat(all_emb, dim = 0)
        classes = torch.cat(all_cls, dim = 0)
        
        acc = utils.KNN(embedding, classes, batch_size=16)
                

        avg_train_loss = total_train_loss / len(train_data_loader)

        print('Epoch:%3d' % epoch
              , '|Train Loss:%8.4f' % (avg_train_loss)
              , '|Val Acc:%3.4f' % (acc))

        if acc > max_acc:
            max_acc = acc
            print("-------------saving model--------------")
            torch.save(model, "model.pth")
    print('Downstream task Done.')

def main():
    # data prepare
    train_path = 'unlabeled'
    test_path = 'test'
    
    
    print("Data Preparing...")
    trainset, train_label, testset,test_label = data_prepare(train_path=train_path, test_path=test_path)
    
    # dataloader
    print("Begin Data Loader...")
    train_dataset = utils.TransDataset(trainset, train_label, utils.train_aug)
    train_data_loader = DataLoader(train_dataset, batch_size = 512, shuffle=True,  num_workers=2)
    test_dataset = utils.TransDataset(testset, test_label, utils.test_aug)
    test_data_loader = DataLoader(test_dataset, batch_size = 16, shuffle=True, num_workers=2)

    # model setup and optimizer config
    feature_dim = 512
    learning_rate = 1e-3
    epochs = 200

    model = Model(feature_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-5)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.0001, min_lr=0)
    acc = 0
    # train
    print("Begin Training...")
    for epoch in range(1, epochs + 1):
        model, train_loss, lr = train(model, train_data_loader, optimizer, lr_scheduler, epoch)
        save_acc, acc = test(model, test_data_loader, acc)

        
    print('Pretest task Done.')
    
    # downstream task
    print('Start to implement downstream task.')
    TOTAL_SIZE = len(test_dataset)
    ratio = 0.8
    finetune_len = round(TOTAL_SIZE * ratio)
    val_len = round(TOTAL_SIZE * (1 - ratio))

    finetune_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [finetune_len, val_len])
    finetune_dl = DataLoader(finetune_dataset, batch_size = 16, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_dataset, batch_size = 8, shuffle=True, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    downstram(finetune_dl, val_dl,test_data_loader)
        
        
    # Embedding
    print("Start to Save the Embedding...")
    torch.cuda.empty_cache()
    
    train_dataset = utils.TransDataset(trainset, train_label, utils.test_aug)
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)   
    final_model = torch.load("model.pth").to(device)
    with torch.no_grad():
        final_model.eval()
        emb = []
        for trns_1, _, _ in train_data_loader:
            trns_1 = trns_1.to(device)
  
            output,_ = final_model(trns_1)
            emb.append(output)

        final_result = torch.cat(emb, dim=0)
        
    np.save('emb_result', final_result.cpu().numpy())
    
    print("Done.")

if __name__ == "__main__":
    main()




