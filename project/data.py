from io import BytesIO

from torch.utils.data import Dataset
from PIL import Image
import zipfile


# 制作猫狗的数据集
class myDataSet(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.image_list = []
        self.label_list = []

        with zipfile.ZipFile(path, 'r') as zip_file:
            # 遍历zip_file
            for file_name in zip_file.namelist():
                if file_name.endswith('.jpg') or file_name.endswith('.png'):
                    label = file_name.split('.')[0]
                    self.image_list.append(file_name)
                    self.label_list.append(0.0 if label == 'train/cat' else 1.0)

        # 以下是从文件中读取的方法
        # for file_name in os.listdir(path):
        #     if file_name.endswith('.jpg') or file_name.endswith('.png'):
        #         image_path = os.path.join(path, file_name)
        #         # 文件格式为 cat.num.jpg
        #         label = file_name.split('.')[0]
        #         self.image_list.append(image_path)
        #         self.label_list.append(0 if label == 'cat' else 1)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        with zipfile.ZipFile(self.path, 'r') as zip_file:
            with zip_file.open(image_path) as image_file:
                image = Image.open(BytesIO(image_file.read()))

        if self.transform:
            image = self.transform(image)

        label = self.label_list[idx]
        return image, label
