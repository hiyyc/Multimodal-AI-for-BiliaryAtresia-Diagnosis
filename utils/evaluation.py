from datetime import datetime
import os
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
import numpy as np

from utils.tools import get_project_root


def draw_pic(score_array, label_onehot, logger, pic_save_path, pic_save_name, num_class=2):
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    precision_dict = dict()
    recall_dict = dict()
    average_precision_dict = dict()

    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(label_onehot[:, i], score_array[:, i])
        average_precision_dict[i] = average_precision_score(label_onehot[:, i], score_array[:, i])
        # print(precision_dict[i].shape, recall_dict[i].shape, average_precision_dict[i])

    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(label_onehot.ravel(), score_array.ravel())
    average_precision_dict["micro"] = average_precision_score(label_onehot, score_array, average="micro")

    pic_save_name_prefix = os.path.join(pic_save_path, pic_save_name)
    plt.figure()
    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='auc = {0:0.4f}'.format(roc_auc_dict["micro"]),
             color='cornflowerblue')
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    label_str = ['nonBA', 'BA']
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    # plt.xlim([0.0, 1.0])
    plt.xlim([-0.02, 1.02])
    # plt.ylim([0.0, 1.05])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(pic_save_name_prefix + '-roc.pdf', bbox_inches='tight')
    plt.show()

    # # PR
    # plt.figure()
    # plt.step(recall_dict['micro'], precision_dict['micro'], where='post')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # # plt.ylim([0.0, 1.0])
    # # plt.xlim([0.0, 1.0])
    # plt.xlim([-0.02, 1.02])
    # plt.ylim([-0.02, 1.02])
    # # plt.title(
    # #     'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    # #         .format(average_precision_dict["micro"]))
    # # plt.savefig(pic_save_name_prefix + '-pr.jpg', dpi=300)
    # plt.savefig(pic_save_name_prefix + '-pr.pdf', bbox_inches='tight')
    # plt.show()


def cal_evaluation(score_list, label_list, logger, phase='train', num_class=2, save_name=None):
    score_array = np.array(score_list)
    label_array = np.array(label_list).reshape(-1, 1)
    label_onehot = np.zeros((label_array.shape[0], 2))
    label_onehot[np.arange(label_array.shape[0]), label_array.flatten()] = 1
    FN = np.sum((np.argmax(score_array, axis=1) == 0) & (np.argmax(label_onehot, axis=1) == 1))
    FP = np.sum((np.argmax(score_array, axis=1) == 1) & (np.argmax(label_onehot, axis=1) == 0))
    TN = np.sum((np.argmax(score_array, axis=1) == 0) & (np.argmax(label_onehot, axis=1) == 0))
    TP = np.sum((np.argmax(score_array, axis=1) == 1) & (np.argmax(label_onehot, axis=1) == 1))

    # micro
    fpr_micro, tpr_micro, _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    sensitivity = TPR = TP / (TP + FN)
    specificity = TN / (TN + FP)
    FPR = FP / (FP + TN)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)

    logger.info('FN, FP, TP, TN are {}, {}, {}, {}'.format(FN, FP, TP, TN))
    logger.info(f'roc-auc, sensitivity, specificity, accuracy, PPV, NPV, FPR are\n{roc_auc_micro}\t{sensitivity}\t{specificity}\t{accuracy}\t{PPV}\t{NPV}\t{FPR}')
    logger.info(f'roc-auc, sensitivity, specificity, accuracy, PPV, NPV, FPR are\n{roc_auc_micro:.4f} & {sensitivity:.4f} & {specificity:.4f} & {accuracy:.4f} & {PPV:.4f} & {NPV:.4f} & {FPR:.4f}')
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    draw_pic(score_array, label_onehot, None, os.path.join(get_project_root(), "rlt"), f'{now_time}-test')
    return roc_auc_micro, sensitivity, specificity, accuracy, PPV, NPV, FPR


def test_model_per_pic(model, device, dataloader, pic_save_path, pic_save_name, num_class=2):
    model.eval()  # Set model to evaluate mode
    score_list = []
    label_list = []

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)  # (batchsize, nclass)
        score_list.extend(outputs.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())
    cal_evaluation(score_list, label_list, pic_save_path, pic_save_name, num_class)


def test_model_per_patient_by_pic(model, device, patient_dataloader, pic_save_path, pic_save_name, img_type=0, num_class=2):
    model.eval()  # Set model to evaluate mode
    score_list = []
    label_list = []

    # Iterate over data.
    for idx, inputs in enumerate(patient_dataloader):
        images_list, record, label = inputs['images_dict'][img_type], inputs['label']
        images_list = images_list.to(device)
        label = label.to(device)

        rlt = []
        for img in images_list:
            output = model([img])
            rlt.append(output.detach().cpu().numpy())

        rlt_array = np.array(rlt)
        score_list.append(np.sum(rlt_array, axis=0))
        label_list.append(label.cpu().numpy())
    cal_evaluation(score_list, label_list, pic_save_path, pic_save_name, num_class)


def test_model_per_patient(model, device, patient_dataloader, pic_save_path, pic_save_name, num_class=2):
    model.eval()  # Set model to evaluate mode
    score_list = []
    label_list = []

    # Iterate over data.
    for idx, inputs in enumerate(patient_dataloader):
        images_dict, record, label = inputs['images_dict'], inputs['record'], inputs['label']
        images_dict = images_dict.to(device)
        record = record.to(device)
        label = label.to(device)

        outputs = model(images_dict, record)  # TODO: to unified the input of the net

        score_list.extend(outputs.detach().cpu().numpy())
        label_list.extend(label.cpu().numpy())
    cal_evaluation(score_list, label_list, pic_save_path, pic_save_name, num_class)

