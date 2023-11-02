import os
import numpy as np
from rich.progress import track
from utils.tools import get_project_root
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import timm.scheduler
from multimodal_model import MultimodalCAFusion

from multimodal_dataset import MultimodalDataLoader
from utils.evaluation import cal_evaluation

from config import config, dataloader_args, set_seed


def train(model, dataloaders, criterion, optimizer, scheduler,
          config):
    model_save_path = config['model_save_path']
    img_types = config['img_types']
    device = config['device']
    logger = config['logger']
    num_epochs = config['num_epochs']
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    local_rank = torch.distributed.get_rank()

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            score_list = []
            label_list = []
            # Iterate over data.
            dataloaders[phase].sampler.set_epoch(epoch)
            # for inputs in tqdm(dataloaders[phase]):
            for inputs in track(dataloaders[phase], description=f'{phase}', total=len(dataloaders[phase]), update_period=1):
                inputs['record'] = inputs['record'].to(device)
                inputs['label'] = inputs['label'].to(device)
                labels = inputs['label']
                # print(labels)

                for i in img_types:
                    inputs[f'image{i}'] = inputs[f'image{i}'].to(device)
                # for inputs, label in dataloaders[phase]:
                #     inputs = inputs.to(device)
                #     labels = label.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # outputs = model(inputs)

                    # for dataparallel
                    # outputs, outputs_tabular, outputs_img0, outputs_img1 = model(inputs)
                    outputs, outputs_tabular, outputs_img0, outputs_img1, outputs_img2, outputs_img3 = model(inputs)

                    # outputs= model(inputs)
                    # loss = criterion(outputs, labels)

                    loss_all = criterion(outputs, labels)
                    loss_tabular = criterion(outputs_tabular, labels)
                    loss_img0 = criterion(outputs_img0, labels)
                    loss_img1 = criterion(outputs_img1, labels)

                    # four loss
                    loss_img2 = criterion(outputs_img2, labels)
                    loss_img3 = criterion(outputs_img3, labels)

                    # loss = loss_all + loss_tabular + loss_img0 + loss_img1
                    loss = loss_all + loss_tabular + loss_img0 + loss_img1 + loss_img2 + loss_img3

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                # statistics
                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels.data)

                score_list.extend(outputs.detach().cpu().numpy())
                label_list.extend(labels.cpu().numpy())

            if phase == 'train':
                scheduler.step()
                # scheduler.step(epoch)

            dataset_size = len(label_list)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train' and local_rank == 0:
                logger.info(f'{phase} evaluation is following')
                cal_evaluation(score_list, label_list, logger)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc and local_rank == 0:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                logger.info('val evaluation is following')
                cal_evaluation(score_list, label_list, logger)
        # if epoch % 5 == 0:
        #     torch.save(best_model_wts, f'{model_save_path}_epoch-{epoch}.pth.tar')
        if local_rank == 0:
            logger.info(f'save checkpoint {epoch}')
            torch.save(model.state_dict(), f'{model_save_path}_checkpoint.pth.tar')
            torch.save(best_model_wts, f'{model_save_path}_best_checkpoint.pth.tar')

    # load best model weights
    if local_rank == 0:
        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        logger.info('Best val Acc: {:4f}'.format(best_acc))
        torch.save(best_model_wts, model_save_path)
        model.load_state_dict(best_model_wts)
    return model


def test(model, dataloader, config):
    model.eval()
    img_types = config['img_types']
    device = config['device']
    logger = config['logger']
    score_list = []
    label_list = []
    for inputs in track(dataloader, description='Test', total=len(dataloader)):
        inputs['record'] = inputs['record'].to(device)
        inputs['label'] = inputs['label'].to(device)
        labels = inputs['label']
        for i in img_types:
            inputs[f'image{i}'] = inputs[f'image{i}'].to(device)
        with torch.no_grad():
            outputs, outputs_tabular, outputs_img0, outputs_img1, outputs_img2, outputs_img3 = model(inputs)
            # _, preds = torch.max(outputs, 1)
            score_list.extend(outputs.detach().cpu().numpy())
            label_list.extend(labels.cpu().numpy())
    cal_evaluation(score_list, label_list, logger, 'test')


def test_patient(model, patient_dataloader, config):
    model.eval()
    img_types = config['img_types']
    device = config['device']
    logger = config['logger']
    score_list = []
    label_list = []

    # for inputs in tqdm(patient_dataloader):
    for inputs in track(patient_dataloader, description="Testing...", total=len(patient_dataloader), update_period=1):
        inputs["record"] = inputs["record"].to(device)
        inputs["label"] = inputs["label"].to(device)
        labels = inputs["label"].to(device)
        for img_c in inputs['image_combination']:
            for i in img_types:
                img_c[f'image{i}'] = img_c[f'image{i}'].to(device)
        with torch.no_grad():
            outputs = model(inputs)
            score_list.append(outputs.detach().cpu().numpy())
            label_list.extend(labels.detach().cpu().numpy())
    cal_evaluation(score_list, label_list, logger, 'test')


def test_patient_bs(model, patient_dataloader, config):
    model.eval()
    img_types = config['img_types']
    device = config['device']
    logger = config['logger']
    score_list = []
    label_list = []
    score_dict = {}
    label_dict = {}

    # for inputs in tqdm(patient_dataloader):
    for inputs in track(patient_dataloader, description="Testing..."):
        inputs["record"] = inputs["record"].to(device)
        for i in img_types:
            inputs[f'image{i}'] = inputs[f'image{i}'].to(device)
        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs.detach().cpu().numpy()

            for i in range(len(inputs['id'])):
                if inputs['id'][i] not in score_dict:
                    score_dict[inputs['id'][i]] = []
                score_dict[inputs['id'][i]].append(outputs[i])
                if inputs['id'][i] not in label_dict:
                    label_dict[inputs['id'][i]] = inputs["label"][i]
                else:
                    assert label_dict[inputs['id'][i]] == inputs["label"][i]
    for k, v in score_dict.items():
        rlt = np.mean(v, axis=0)
        score_list.append(rlt)
        label_list.append(label_dict[k])
    cal_evaluation(score_list, label_list, logger, 'test')


def train_main(config, dataloader_args):
    logger = config['logger']
    phase = 'train'
    model = MultimodalCAFusion(config, phase)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(config['device'])
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    model.to(config['device'])

    criterion = nn.CrossEntropyLoss()
    # optimizer_ft = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer_ft = optim.AdamW(model.parameters(), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.2)
    # optimizer_ft = torch.optim.AdamW(model.parameters(),lr=1e-3,eps=1e-8)
    # exp_lr_scheduler = timm.scheduler.CosineLRScheduler(optimizer_ft, t_initial = 20, lr_min = 1e-4, warmup_t = 20, warmup_lr_init = 1e-4)

    logger.info(f'model_save_path: {config["model_save_path"]}')
    dataloader_train = MultimodalDataLoader(dataloader_args, phase).train
    dataloader_val = MultimodalDataLoader(dataloader_args, phase).eval
    dataloaders = {
        'train': dataloader_train,
        'val': dataloader_val
    }
    model = train(model, dataloaders, criterion, optimizer_ft,
                  exp_lr_scheduler, config)
    logger.debug('training finished')
    if torch.distributed.get_rank() == 0:
        test(model, MultimodalDataLoader(dataloader_args, phase).test, config)


def test_main(config, dataloader_args):
    logger = config['logger']
    logger.debug('test begin ...')
    phase = 'test_patient'
    model = MultimodalCAFusion(config, phase)
    model.to(config['device'])
    # model.load_state_dict(torch.load(config['model_save_path']))
    state_dict = torch.load(config['model_save_path'], map_location=config['device'])
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of DataParallel
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    if phase == 'test_patient_bs':
        test_patient_bs(model, MultimodalDataLoader(dataloader_args, phase).test, config)
    elif phase == 'test_patient':
        test_patient(model, MultimodalDataLoader(dataloader_args, phase).test, config)
    else:
        test(model, MultimodalDataLoader(dataloader_args, phase).test, config)


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_default_dtype(torch.float32)
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    set_seed(666 + local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # set_seed(666)
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    config['logger'].info(f'dataloader args: {dataloader_args}')
    config['logger'].info(f'config: {config}')
    train_main(config, dataloader_args)
    if local_rank == 0:
        test_main(config, dataloader_args)


if __name__ == '__main__':
    main()
    # CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=2905 train_multimodal_model.py