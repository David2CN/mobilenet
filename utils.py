import torch
import numpy as np
from tqdm import tqdm
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

def epoch_loop(model: Module, data_loaders: dict,
                cuda: bool, optimizer: Optimizer, criterion: Module):

    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]

    train_loss = 0
    val_loss = 0
    train_acc = 0
    val_acc = 0
    
    model.train()
    for data, target in tqdm(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * data.size(0)
        
        _, pred = torch.max(output, 1)
        train_acc += torch.sum(np.squeeze(pred.eq(target.data.view_as(pred)))).item()

    
    model.eval()
    for data, target in tqdm(val_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        
        output = model(data)
        loss = criterion(output, target)
        val_loss += loss.item() * data.size(0)

        _, pred = torch.max(output, 1)
        val_acc += torch.sum(np.squeeze(pred.eq(target.data.view_as(pred)))).item()    

    return train_loss, val_loss, train_acc, val_acc


def train(model: Module, data_loaders: dict, data_sizes: dict, optimizer: Optimizer, 
        criterion: Module, epochs: int=10, writer: SummaryWriter=None, model_path: str="./",
                initial_epochs: int=0):
    
    cuda = torch.cuda.is_available()
    
    val_loss_min = np.Inf
    for epoch in range(initial_epochs+1, epochs+1):
        
        train_loss, val_loss, train_acc, val_acc = epoch_loop(model, data_loaders, cuda, optimizer, criterion)
                
        train_loss = train_loss / data_sizes["train"]
        val_loss = val_loss / data_sizes["val"]

        train_acc = train_acc / data_sizes["train"]
        val_acc = val_acc / data_sizes["val"]

        # Log the running loss
        if writer:
            writer.add_scalars('loss',
                            { 'train' : train_loss, 'validation' : val_loss },
                            epoch)
            writer.add_scalars('accuracy',
                            { 'train' : train_acc, 'validation' : val_acc },
                            epoch)
            writer.flush()

        
        print(f"Epoch {epoch}/{epochs}: loss- {train_loss:.3f}, acc- {train_acc:.3f}, val_loss- {val_loss:.3f}, val_acc- {val_acc:.3f}") 
        
        if val_loss <= val_loss_min:
            print(f"val_loss decreased from {val_loss_min:.4f} to {val_loss:.4f}. saving model ...")
            torch.save(model.state_dict(), f"{model_path}model.{epoch:02d}-{val_loss:.4f}.pt")
            val_loss_min = val_loss


def test(model, data_loaders, writer=None):

    cuda = torch.cuda.is_available()

    classes = ["airplane", "automobile", "bird", "cat",
          "deer", "dog", "frog", "horse", "ship",
          "truck"]
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    test_loader = data_loaders["test"]
    model.eval()
    for data, target in tqdm(test_loader):

        if cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not cuda else np.squeeze(correct_tensor.cpu().numpy())
        
        # calculate test accuracy for each object class
        for i in range(data.size(0)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    result = ""
    for i in range(10):
        if class_total[i] > 0:
            result += f"Test Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%\n"
        else:
            result += f"Test Accuracy of {classes}: N/A (no training examples)\n"

    acc = 100. * np.sum(class_correct) / np.sum(class_total)
    result += f"Test Accuracy (Overall): {acc:.2f}%"

    print(result)
    # Log the running loss
    if writer:
        writer.add_text("test_results", result, 0)
        writer.flush()


from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue

        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
