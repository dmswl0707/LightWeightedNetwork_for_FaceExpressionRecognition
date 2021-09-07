
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import numpy as np

transform_train = transforms.Compose([transforms.Resize((50,50)),
                                    transforms.RandomRotation(degrees=30),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5),(0.5))
                                    ])

transform_test = transforms.Compose([transforms.Resize((50,50)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5),(0.5))])

train_dataset = ImageFolder(root='/home/eunji/project_dir/LightWeightedNetwork_for_FaceExpressionRecognition-main/CK+48/train', transform=transform_train)
test_dataset = ImageFolder(root='/home/eunji/project_dir/LightWeightedNetwork_for_FaceExpressionRecognition-main/CK+48/test', transform=transform_test)


categories = list(train_dataset.class_to_idx.keys())
print(categories)
num_classes = len(categories)

print({'train':len(train_dataset)})
print({'test':len(test_dataset)})

print({'num_classes': num_classes})


image_h = 50
image_w = 50

val_size = 0.25 

train_num_data = len(train_dataset)
test_num_data = len(test_dataset)

indices1 = list(range(train_num_data))
np.random.shuffle(indices1)

indices2 = list(range(test_num_data))
np.random.shuffle(indices2)

val_split = int(np.floor(val_size * train_num_data))
val_idx, train_idx = indices1[:val_split], indices1[val_split:]


test_split = int(np.floor(test_num_data))
test_idx = indices2[:test_split]

print(len(test_idx), len(val_idx), len(train_idx))

