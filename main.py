from training import *


if __name__ == "__main__":
    training()

    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.001)
    #model, train_loss, val_loss = training()
