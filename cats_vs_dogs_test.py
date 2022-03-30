import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {
    'test': datasets.ImageFolder('dataset/test', data_transforms['test'])
}

dataloaders = {
    'test': torch.utils.data.DataLoader(
        image_datasets['test'],
        batch_size=64,
        shuffle=False)
}

model = models.resnet50(pretrained=False).to(device)

model.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 1)).to(device)

model.load_state_dict(torch.load('cats-vs-dogs-model.h5'))

with torch.no_grad(): # Gradient를 업데이트하지 않는다
    test_corrects = 0
    
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.unsqueeze(dim=1).float().to(device)

        outputs = model(inputs)
        #print(outputs)
        decisions = torch.sigmoid(outputs).cpu().data.numpy()
        preds = decisions > 0.5
        #print(preds)
        test_corrects += torch.sum(preds == labels.data)
        #print(test_corrects)

    test_acc = test_corrects.double() / len(image_datasets['test'])

    print('Test result, acc: {:.4f}'.format(test_acc))