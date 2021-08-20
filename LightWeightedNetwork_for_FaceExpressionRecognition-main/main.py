from train import *

# excute train.py


if __name__ == "__main__":

    model_ft = model_ft.to(device)
    print(summary(model_ft, (3, 50, 50)))

    optimizer = optim.SGD(model_ft.parameters(), momentum=0.9, lr=0.001, weight_decay=5e-3)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.001)

    model, train_loss, val_loss = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=200)
