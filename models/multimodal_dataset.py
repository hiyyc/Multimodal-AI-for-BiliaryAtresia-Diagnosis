import csv
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils.tools import combined_all_list_without_empty, get_phase_list
from torch.utils.data.distributed import DistributedSampler

class MultimodalDataset(Dataset):
    """
    patient_img_csv_file: [[patient_id, label], ]
    patient_record_csv_file: [[patient_id, feature_value1, feature_value2, ..., label], ]
    return: {'record': [], 'image': Img, 'label': label}
    """
    def __init__(self, patient_img_csv_file, patient_record_csv_file,
                 root_dir, logger, phase, missing_modal=1, img_types=[0, 1, 2, 3], transform=None):
        self.data_list = []
        self.root_dir = root_dir
        self.logger = logger
        self.phase = phase
        self.missing_modal = missing_modal
        self.img_types = img_types
        self.transform = transform
        with open(patient_record_csv_file, 'r', encoding='utf-8', newline='') as record_rf:
            self.records_dict = {i[0]: i[1:] for i in csv.reader(record_rf)}
        with open(patient_img_csv_file, 'r', encoding='utf-8', newline='') as img_rf:
            img_reader = csv.reader(img_rf)
            if self.phase == 'train':
                for i in img_reader:
                    # debug
                    # if img_reader.line_num > 10:
                    #     break
                    if not self.img_reader_filter(i):
                        continue
                    the_imgs_dir = os.path.join(self.root_dir, str(i[0]))
                    tmp_data = {}
                    for img_type in self.img_types:
                        tmp_data[f'image_path{img_type}'] = []
                        imgs_path = os.path.join(the_imgs_dir, str(img_type))
                        if (not os.path.exists(imgs_path)) or (not os.listdir(imgs_path)):
                            # logger.warning(f'{i[0]} does not have type {img_type} pic')
                            tmp_data[f'image_path{img_type}'].append('')
                            continue
                        for j in os.listdir(imgs_path):
                            tmp_data[f'image_path{img_type}'].append(f'{img_type}/{j}')
                    item_list = tmp_data['image_path0']
                    for j in self.img_types[1:]:
                        item_list = combined_all_list_without_empty(item_list, tmp_data[f'image_path{j}'])
                    for k in item_list:
                        self.data_list.append({
                            'id': i[0],
                            'image_path0': k[0],
                            'image_path1': k[1],
                            'image_path2': k[2],
                            'image_path3': k[3],
                            'label': 1 if str(self.records_dict[i[0]][-1]) == '1' else 0
                        })
            elif self.phase == 'train_small_no_missing':
                for i in img_reader:
                    if not self.img_reader_filter(i):
                        continue
                    the_imgs_dir = os.path.join(self.root_dir, str(i[0]))
                    data_flag = True
                    img_list_dict = {}
                    for img_type in self.img_types:
                        img_list_dict[img_type] = []
                        imgs_path = os.path.join(the_imgs_dir, str(img_type))
                        if (not os.path.exists(imgs_path)) or (not os.listdir(imgs_path)):
                            # logger.warning(f'{i[0]} does not have type {img_type} pic')
                            data_flag = False
                            break
                        for j in os.listdir(imgs_path):
                            img_list_dict[img_type].append(f'{img_type}/{j}')
                    if data_flag:
                        # logger.debug(f'ready to gene {i[0]} dataset')
                        self.item_list = img_list_dict[0]
                        len_list = [len(img_list_dict[iti]) for iti in img_types]
                        max_len_num = max(len_list)
                        for j in range(max_len_num):
                            tmp_data = {
                                'id': i[0],
                                'label': 1 if str(self.records_dict[i[0]][-1]) == '1' else 0
                            }
                            for itj in img_types:
                                len_img = len(img_list_dict[itj])
                                if len_img < max_len_num:
                                    tmp_j = random.randint(0, len_img - 1)
                                    tmp_data[f'image_path{itj}'] = img_list_dict[itj][tmp_j]
                                else:
                                    tmp_data[f'image_path{itj}'] = img_list_dict[itj][j]
                            self.data_list.append(tmp_data)
            elif self.phase == 'train_small':
                for i in img_reader:
                    if not self.img_reader_filter(i):
                        continue
                    the_imgs_dir = os.path.join(self.root_dir, str(i[0]))
                    filepath_list = [[] for i in self.img_types]
                    for img_type in self.img_types:
                        imgs_path = os.path.join(the_imgs_dir, str(img_type))
                        if (not os.path.exists(imgs_path)) or (not os.listdir(imgs_path)):
                            # logger.warning(f'{i[0]} does not have type {img_type} pic')
                            filepath_list[img_type].append('')
                            continue
                        for j in os.listdir(imgs_path):
                            filepath_list[img_type].append(f'{img_type}/{j}')
                    len_list = [len(filepath_list[iti]) for iti in img_types]
                    max_len_num = max(len_list)
                    for j in range(max_len_num):
                        tmp_data = {
                            'id': i[0],
                            'label': 1 if str(self.records_dict[i[0]][-1]) == '1' else 0
                        }
                        for itj in img_types:
                            len_img = len(filepath_list[itj])
                            if len_img < max_len_num:
                                tmp_j = random.randint(0, len_img - 1)
                                tmp_data[f'image_path{itj}'] = filepath_list[itj][tmp_j]
                            else:
                                tmp_data[f'image_path{itj}'] = filepath_list[itj][j]
                        self.data_list.append(tmp_data)
            elif self.phase == 'test_patient_bs':
                for i in img_reader:
                    if not self.img_reader_filter(i):
                        continue
                    the_imgs_dir = os.path.join(self.root_dir, str(i[0]))
                    tmp_data = {}
                    for img_type in self.img_types:
                        tmp_data[f'image_path{img_type}'] = []
                        imgs_path = os.path.join(the_imgs_dir, str(img_type))
                        if (not os.path.exists(imgs_path)) or (not os.listdir(imgs_path)):
                            # logger.warning(f'{i[0]} does not have pic type {img_type}')
                            tmp_data[f'image_path{img_type}'].append('')
                            continue
                        for j in os.listdir(imgs_path):
                            tmp_data[f'image_path{img_type}'].append(f'{img_type}/{j}')
                    item_list = tmp_data['image_path0']
                    for j in self.img_types[1:]:
                        item_list = combined_all_list_without_empty(item_list, tmp_data[f'image_path{j}'])
                    for k in item_list:
                        tmp_data = {
                            'id': i[0],
                            'label': 1 if str(self.records_dict[i[0]][-1]) == '1' else 0,
                            'image_path0': k[0],
                            'image_path1': k[1],
                            'image_path2': k[2],
                            'image_path3': k[3],
                        }
                        self.data_list.append(tmp_data)
            elif self.phase == 'test_patient':
                for i in img_reader:
                    if not self.img_reader_filter(i):
                        continue
                    the_imgs_dir = os.path.join(self.root_dir, str(i[0]))
                    tmp_data = {}
                    for img_type in self.img_types:
                        tmp_data[f'image_path{img_type}'] = []
                        imgs_path = os.path.join(the_imgs_dir, str(img_type))
                        if (not os.path.exists(imgs_path)) or (not os.listdir(imgs_path)):
                            # logger.warning(f'{i[0]} does not have pic type {img_type}')
                            tmp_data[f'image_path{img_type}'].append('')
                            continue
                        for j in os.listdir(imgs_path):
                            tmp_data[f'image_path{img_type}'].append(f'{img_type}/{j}')
                    item_list = tmp_data['image_path0']
                    for j in self.img_types[1:]:
                        item_list = combined_all_list_without_empty(item_list, tmp_data[f'image_path{j}'])
                    image_combination = []
                    for k in item_list:
                        image_combination.append({
                            'image_path0': k[0],
                            'image_path1': k[1],
                            'image_path2': k[2],
                            'image_path3': k[3],
                        })
                    self.data_list.append({
                        'id': i[0],
                        'image_combination': image_combination,
                        'label': 1 if str(self.records_dict[i[0]][-1]) == '1' else 0
                    })
        self.train_phase = get_phase_list('train')
        self.test_phase = get_phase_list('test')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        the_id = self.data_list[idx]['id']
        record = self.records_dict[the_id][:-1]
        record = [float(x) for x in record]
        data = {
            'label': int(self.data_list[idx]['label']),
            'record': torch.tensor(record)
        }
        id_dir = os.path.join(self.root_dir, the_id)

        if self.phase == 'test_patient_bs':
            for i in self.img_types:
                # data[f'img_org_{i}'] is for heatmap
                if self.missing_modal == 0:
                    data[f'image{i}'] = \
                        self.img_item_filter_all_zero(self.data_list[idx], i, id_dir)
                else:
                    data[f'image{i}'] = \
                        self.img_item_filter(self.data_list[idx], i, id_dir)
            data['id'] = the_id
            return data
        elif self.phase in self.train_phase:
            for i in self.img_types:
                # data[f'img_org_{i}'] is for heatmap
                if self.missing_modal == 0:
                    data[f'image{i}'] = \
                        self.img_item_filter_all_zero(self.data_list[idx], i, id_dir)
                else:
                    data[f'image{i}'] = \
                        self.img_item_filter(self.data_list[idx], i, id_dir)
            return data
        elif self.phase in self.test_phase:
            image_combination = self.data_list[idx]['image_combination']
            data['image_combination'] = []
            for img_c in image_combination:
                img_dict = {}
                for i in self.img_types:
                    # img_dict[f'img_org_{i}'] is for heatmap
                    if self.missing_modal == 0:
                        img_dict[f'image{i}'] = \
                            self.img_item_filter_all_zero(img_c, i, id_dir)
                    else:
                        img_dict[f'image{i}'] = \
                            self.img_item_filter(img_c, i, id_dir)
                data['image_combination'].append(img_dict)
            return data

    def img_item_filter(self, img_c, img_type, id_dir):
        if img_c[f'image_path{img_type}'] == '' and img_type == 0:
            img_o = Image.new('RGB', (200, 200), (255, 255, 255))
        elif img_c[f'image_path{img_type}'] == '':
            if img_c[f'image_path0'] == '':
                if img_c[f'image_path1'] == '':
                    img_o = Image.new('RGB', (200, 200), (255, 255, 255))
                else:
                    img_o = Image.open(os.path.join(id_dir, img_c[f'image_path1']))
                # img_o = Image.new('RGB', (200, 200), (255, 255, 255))
            else:
                img_o = Image.open(os.path.join(id_dir, img_c[f'image_path0']))
        else:
            img_o = Image.open(os.path.join(id_dir, img_c[f'image_path{img_type}']))
        if self.transform is not None:
            if img_o.mode == 'L':
                img_o = img_o.convert('RGB')
            img_t = self.transform(img_o)
        return img_t

    def img_item_filter_all_zero(self, img_c, img_type, id_dir):
        if img_c[f'image_path{img_type}'] == '':
            img_o = Image.new('RGB', (200, 200), (255, 255, 255))
        else:
            img_o = Image.open(os.path.join(id_dir, img_c[f'image_path{img_type}']))
        if self.transform is not None:
            if img_o.mode == 'L':
                img_o = img_o.convert('RGB')
            img_t = self.transform(img_o)
        return img_t

    def img_reader_filter(self, i):
        if not i[0] in self.records_dict:
            # self.logger.warning(f'{i[0]} does not have records')
            return False
        if str(self.records_dict[i[0]][-1]) != str(i[-1]):
            self.logger.error(f'{i[0]}\'s record label is not the same with the pic label')
        return True


class MultimodalDataLoader:
    def __init__(self, args, phase):
        self.num_workers = args['num_workers'] if 'num_workers' in args else 4
        self.batch_size = args['batch_size'] if 'batch_size' in args else 16
        self.test_batch_size = args['test_batch_size'] if 'test_batch_size' in args else 16
        self.img_size = args['img_size'] if 'img_size' in args else 224
        self.mean = args['mean'] if 'mean' in args else [0.485, 0.456, 0.406]
        self.std = args['std'] if 'std' in args else [0.229, 0.224, 0.225]

        self.train_img_csv = args['train_img_csv']
        self.val_img_csv = args['val_img_csv']
        self.test_img_csv = args['test_img_csv']
        self.record_csv = args['record_csv']
        self.img_dir = args['img_dir']
        self.logger = args['logger']
        self.img_types = args['img_types']
        self.phase = phase
        self.missing_modal = args['missing_modal']

    @property
    def train(self):
        tf = transforms.Compose([
            # transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1)),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        dataset_train = MultimodalDataset(self.train_img_csv,
                                          self.record_csv,
                                          self.img_dir,
                                          self.logger,
                                          self.phase,
                                          missing_modal=self.missing_modal,
                                          img_types=self.img_types,
                                          transform=tf)
        train_sampler = DistributedSampler(dataset_train)
        return DataLoader(dataset_train, batch_size=self.batch_size,
                          sampler=train_sampler, num_workers=self.num_workers)

    @property
    def eval(self):
        dataset_val = MultimodalDataset(self.val_img_csv,
                                         self.record_csv,
                                         self.img_dir,
                                         self.logger,
                                         self.phase,
                                         missing_modal=self.missing_modal,
                                         img_types=self.img_types,
                                         transform=self.get_test_tf())
        val_sampler = DistributedSampler(dataset_val)
        return DataLoader(dataset_val, batch_size=self.batch_size,
                          sampler=val_sampler, num_workers=self.num_workers)

    @property
    def test(self):
        dataset_test = MultimodalDataset(self.test_img_csv,
                                         self.record_csv,
                                         self.img_dir,
                                         self.logger,
                                         self.phase,
                                         missing_modal=self.missing_modal,
                                         img_types=self.img_types,
                                         transform=self.get_test_tf())
        return DataLoader(dataset_test, batch_size=self.test_batch_size,
                                  shuffle=False, num_workers=self.num_workers,
                                  pin_memory=True)

    def get_test_tf(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
