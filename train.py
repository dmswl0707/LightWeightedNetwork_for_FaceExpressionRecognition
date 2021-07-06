import time
from dataloader import *
from functions import *
from model import *
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

writer = SummaryWriter()
patience = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device) # GPU인지 CPU인지

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    train_losses, val_losses = [], []
    avg_train_losses, avg_val_losses = [], []

    since = time.time()

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, num_epochs + 1):

        for phase in ['train', 'val']:
            if phase == 'train':

                model.train()

                for batch, (inputs, labels) in enumerate(loaders['train'], 1):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                    train_losses.append(loss.item())

            if phase == 'val':
                model.eval()

                with torch.no_grad():

                    for batch, (inputs, labels) in enumerate(loaders['val']):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs)
                        val_loss = criterion(outputs, labels)
                        val_losses.append(val_loss.item())

                    train_loss = np.average(train_losses)
                    val_loss = np.average(val_losses)
                    avg_train_losses.append(train_loss)
                    avg_val_losses.append(val_loss)

                    epoch_len = len(str(num_epochs))

                    print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                                 f'train_loss: {train_loss:.4f} ' +
                                 f'valid_loss: {val_loss:.4f}')

                    print(print_msg)

                    train_losses = []
                    val_losses = []

                    early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

    model.load_state_dict(torch.load('checkpoint.pt'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    graph_loss(avg_train_losses, avg_val_losses)

    return model_ft, avg_train_losses, avg_val_losses

model_ft = Model(num_classes=7)
criterion = nn.CrossEntropyLoss()

