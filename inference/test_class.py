from main import *


PATH = '/Users/ChoiEunJi.DESKTOP-BO1GKPC/Desktop/LightWeightedNetwork_for_FaceExpressionRecognition-main/checkpoint/checkpoint.pt'
model=Model(num_classes=7)
model.load_state_dict(torch.load(PATH))

class_correct = list(0. for i in range(7))
class_total = list(0. for i in range(7))

test_loss = 0.0
correct = 0
total = 0

model.eval()
with torch.no_grad():
    for data in loaders['test']:
        images, labels = data
        #images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)
        c = (pred == labels).squeeze()  # 예측과 실제 라벨 비교

        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)

        for i in range(labels.shape[0]):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

            total += labels.size(0)
            correct += (pred == labels).sum().item()

test_loss = test_loss / len(test_idx)
print('Test Loss: {:.6f}\n'.format(test_loss))

print('Accuracy of the network on test images: %4f %%' % (
        100 * correct / total))

for i in range(7):
    print('Accuracy of %5s : %4f %%' % (
        categories[i], 100 * class_correct[i] / class_total[i]))




