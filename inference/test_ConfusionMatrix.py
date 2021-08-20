from main import *
import pandas as pd
import seaborn as sns


PATH = '/Users/ChoiEunJi.DESKTOP-BO1GKPC/Desktop/LightWeightedNetwork_for_FaceExpressionRecognition-main/checkpoint/checkpoint.pt'

nb_classes = 7
confusion_matrix = np.zeros((nb_classes, nb_classes))
model = Model(num_classes=7)
model.cuda()

model.load_state_dict(torch.load(PATH))

model.eval()
with torch.no_grad():
    for i, (inputs, classes) in enumerate(loaders['test']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

plt.figure(figsize=(15,10))

print(confusion_matrix)
#print(confusion_matrix.diag()/confusion_matrix.sum(1))

class_names = list(categories)
df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()