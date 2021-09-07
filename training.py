import torch.optim as optim
#import torch.optim.lr_scheduler
from model_f import Model
import torch
import torch.nn as nn
from functions import EarlyStopping
from custom_CosineAnnealingWarmupRestart import *
from dataloader import *
from Args import *
import wandb


wandb.init(project='Light_face_recognition', entity='dmswl0707', name = Args["name"])

# train setting
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Model.to(device)
print(device)

# 하이퍼 파라미터 변경
optimizer = optim.Adam(model.parameters(), lr=Args["lr"], betas=(0.9, 0.999), eps=1e-08)
#optimizer = optim.SGD(model_ft.parameters(), momentum=0.9, lr=0.001, weight_decay=5e-3)
scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=1, eta_max=0.001, T_up=10, gamma=0.5)
#exp_lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=2, eta_max=3e-6, T_up=25, gamma=0.5)
Epoch = Args["Epoch"]
patience = Args["patience"]

wandb.watch(model)


def training():

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(Epoch):
        print("===================================================")
        print("epoch: ", epoch + 1)

        train_loss, val_loss = [], []
        avg_train_loss, avg_val_loss =  [], []

        total = 0
        v_total = 0

        train_loss = 0.0
        train_correct = 0.0
        val_loss = 0.0
        val_correct = 0.0

        model.train()

        for inputs, labels in loaders["train"]:
            optimizer.zero_grad()
            X = inputs.to(device)
            y = labels.to(device)

            y_pred = model(X)
            loss = criterion(y_pred, y)

            loss.backward()
            # step 마다 돌아가는 부분
            optimizer.step()

            # acc를 구하는 부분
            _, preds = torch.max(y_pred, 1)

            total += y.size(0)
            train_loss += loss.item()
            #train_loss.append(loss.item())
            train_correct += (preds == y).sum().item()

            np_train_loss = np.average(train_loss)
            np_train_acc = np.average(100. * float(train_correct) / total)

        epoch_loss = np_train_loss / len(loaders["train"])
        epoch_acc = np_train_acc

        print("train loss: {:.4f}, acc: {:4f}".format(epoch_loss, epoch_acc))

        wandb.log({
            "custom_epoch": epoch,
            "Train Loss": epoch_loss,
            "Train Accuracy": epoch_acc,
            "Train error": 100 - epoch_acc,
            "lr" : optimizer.param_groups[0]['lr'] # 학습률 로깅
        })

        with torch.no_grad():

            model.eval()
            for val_inputs, val_labels in loaders["val"]:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                y_val_pred = model(val_inputs)
                v_loss = criterion(y_val_pred, val_labels)

                #validation - acc를 구하는 부분
                _, v_preds = torch.max(y_val_pred, 1)

                v_total += val_labels.size(0)
                val_loss += v_loss.item()
                #val_loss.append(v_loss.item())
                val_correct += (v_preds == val_labels).sum().item()

                np_val_loss = np.average(val_loss)
                np_val_acc = np.average(100. * float(val_correct) / v_total)

            val_epoch_loss = np_val_loss / len(loaders["val"])
            val_epoch_acc = np_val_acc

            print("val loss: {:.4f}, acc: {:4f}".format(val_epoch_loss, val_epoch_acc))

            wandb.log({
                "Val Loss": val_epoch_loss,
                "Val Accuracy": val_epoch_acc,
                "Val error": 100 - val_epoch_acc,
            })

            scheduler.step()

            early_stopping(val_epoch_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
