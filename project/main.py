# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from io import BytesIO

from torchvision import transforms
from data import myDataSet
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torch import nn, sigmoid, optim
from torchvision.models import resnet18
import zipfile
from myUtils import *
from tqdm import tqdm
import csv


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        # 输出维度是1000，是在ImageNet上训练得到的
        self.resnet = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        x = sigmoid(x)
        return x


def main():
    # 定义图像的预处理方式
    transform = transforms.Compose([
        transforms.Resize(256),  # 图像大小调整为256*256
        transforms.CenterCrop(224),  # 从中心裁剪出224*224大小的图片
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        # 在各个通道上，用这些均值和方差进行归一化，这些值是在ImageNet上得到的
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    my_dataset = myDataSet('../dogs-vs-cats-redux-kernels-edition/train.zip', transform)

    train_size = int(0.8 * len(my_dataset))
    val_size = len(my_dataset) - train_size
    train_dataset, val_dataset = random_split(my_dataset, [train_size, val_size])

    train_iter = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_iter = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)

    num_epochs, lr, weight_decay, batch_size = 10, 0.0005, 0, 16
    model = ResNet18()
    # 进行训练
    loss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 将model放到GPU上
    model.to(device)

    for epoch in range(num_epochs):
        # 训练
        model.train()
        pbar = tqdm(total=len(train_iter))
        for i, (image, label) in enumerate(train_iter):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            # view和reshape都可以改变形状，view占用原内存空间，reshape需要开一段新的空间
            label = label.view(16, 1).to(torch.float32)
            _loss = loss(model(image), label)
            _loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_description(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_iter)}], Loss: {_loss.item():.4f}')
        # 进行正确率判断
        pbar.close()
        train_acc = evaluate_accuracy(model, train_iter)
        val_acc = evaluate_accuracy(model, val_iter)

        print(f"在epoch{epoch}轮训练中,测试集正确率为:{train_acc},验证集正确率为:{val_acc}")

    # 进行预测
    results = {'id': [], 'label': []}
    image_dir = '../dogs-vs-cats-redux-kernels-edition/test.zip'

    model.eval()
    with zipfile.ZipFile(image_dir, 'r') as zip_file:
        # 遍历zip_file
        for file_name in zip_file.namelist():
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                with zip_file.open(file_name) as image_file:
                    image = transform(Image.open(BytesIO(image_file.read()))).to(device)
                    # 文件格式为 num.jpg
                    img_id = file_name.split('.')[0]
                    img_pred = model(image.unsqueeze(0)).item()
                    print(file_name, "cat" if img_pred<0.5 else "dog")
                    results['id'].append(img_id.split("/")[1])
                    results['label'].append(img_pred)

    # for file_name in os.listdir(image_dir):
    #     if file_name.endswith('.jpg') or file_name.endswith('.png'):
    #         image_path = os.path.join(image_dir, file_name)
    #         image = transform(Image.open(image_path).convert('RGB'))
    #         image = image.to(device)
    #
    #         # 文件格式为 num.jpg
    #         img_id = file_name.split('.')[0]
    #         img_pred = model(image).item()
    #         results['id'].append(img_id)
    #         results['label'].append(img_pred)

    rows = zip(results['id'], results['label'])

    with open('submission.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['name', 'age', 'gender'])
        for row in rows:
            writer.writerow(row)


def test():
    transform = transforms.Compose([
        transforms.Resize(256),  # 图像大小调整为256*256
        transforms.CenterCrop(224),  # 从中心裁剪出224*224大小的图片
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        # 在各个通道上，用这些均值和方差进行归一化，这些值是在ImageNet上得到的
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    with zipfile.ZipFile("D:\\MyDocuments\\kaggle\\dogs-vs-cats\\dogs-vs-cats-redux-kernels-edition\\train.zip",
                         'r') as zip_file:
        # 遍历zip_file
        for file_name in zip_file.namelist():
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                label = file_name.split('.')[0]
                print(file_name)
                path = file_name
                print(0 if label == 'cat' else 1)
                break
    with zipfile.ZipFile("D:\\MyDocuments\\kaggle\\dogs-vs-cats\\dogs-vs-cats-redux-kernels-edition\\train.zip",
                         'r') as zip_file:
        with zip_file.open(path) as image_file:
            image = BytesIO(image_file.read())
            print(transform(Image.open(image)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test()
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
