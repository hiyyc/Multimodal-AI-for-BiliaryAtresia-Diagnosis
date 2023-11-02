import os
import torch


def cal_norm(dataset):
    print('Compute mean and variance for training data.')
    print(len(dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return list(mean.numpy()), list(std.numpy())


def combined_all_list(list1, list2):
    if len(list1) == 0:
        return list2
    elif len(list2) == 0:
        return list1
    elif isinstance(list1[0], list) and isinstance(list2[0], list):
        return [x.extend(y) for x in list1 for y in list2]
    elif isinstance(list1[0], list):
        return [x + [y] for x in list1 for y in list2]
    elif isinstance(list2[0], list):
        return [y + [x] for x in list1 for y in list2]
    else:
        return [[x, y] for x in list1 for y in list2]


def combined_all_list_without_empty(list1, list2):
    assert len(list1) != 0 and len(list2) != 0
    if isinstance(list1[0], list) and isinstance(list2[0], list):
        return [x.extend(y) for x in list1 for y in list2]
    elif isinstance(list1[0], list):
        return [x + [y] for x in list1 for y in list2]
    elif isinstance(list2[0], list):
        return [[x] + y for x in list1 for y in list2]
    else:
        return [[x, y] for x in list1 for y in list2]


def get_project_root():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = cur_path[:cur_path.find('Multimodal-AI-for-BiliaryAtresia-Diagnosis')+len('Multimodal-AI-for-BiliaryAtresia-Diagnosis')]
    return root_path

def get_phase_list(phase):
    if 'train' in phase:
        return ['train', 'train_small', 'train_small_no_missing']
    elif 'test' in phase:
        return ['test_patient']
