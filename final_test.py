import os
import yaml
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.utils import count_params, AverageMeter, intersectionAndUnion, color_map

from image_utils import save_img, make_test_detailed_img

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Measurement:
    def __init__(self, num_classes:int, ignore_idx=None) :
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
    
    def _make_confusion_matrix(self, pred:np.ndarray, target:np.ndarray):
        """make confusion matrix

        Args:
            pred (numpy.ndarray): segmentation model's prediction score matrix
            target (numpy.ndarray): label
            num_classes (int): the number of classes
        """
        assert pred.shape[0] == target.shape[0], "pred and target ndarray's batchsize must have same value"
        N = pred.shape[0]
        # prediction score to label
        pred_label = pred.argmax(axis=1) # (N, H, W)
        
        pred_1d = np.reshape(pred_label, (N, -1)) # (N, HxW)
        target_1d = np.reshape(target, (N, -1)) # (N, HxW)
        # num_classes * gt + pred = category
        cats = self.num_classes * target_1d + pred_1d # (N, HxW)
        conf_mat = np.apply_along_axis(lambda x: np.bincount(x, minlength=self.num_classes**2), axis=-1, arr=cats) # (N, 9)
        conf_mat = np.reshape(conf_mat, (N, self.num_classes, self.num_classes)) # (N, 3, 3)
        return conf_mat
    
    def accuracy(self, pred, target):
        '''
        Args:
            pred: (N, C, H, W), ndarray
            target : (N, H, W), ndarray
        Returns:
            the accuracy per pixel : acc(int)
        '''
        N = pred.shape[0]
        pred = pred.argmax(axis=1) # (N, H, W)
        pred = np.reshape(pred, (pred.shape[0], pred.shape[1]*pred.shape[2])) # (N, HxW)
        target = np.reshape(target, (target.shape[0], target.shape[1]*target.shape[2])) # (N, HxW)
        
        if self.ignore_idx != None:
             not_ignore_idxs = np.where(target != self.ignore_idx) # where target is not equal to ignore_idx
             pred = pred[not_ignore_idxs]
             target = target[not_ignore_idxs]
             
        return np.mean(np.sum(pred==target, axis=-1)/pred.shape[-1])
    
    def miou(self, conf_mat:np.ndarray):
        iou_list = []
        sum_col = np.sum(conf_mat, -2) # (N, 3)
        sum_row = np.sum(conf_mat, -1) # (N, 3)
        for i in range(self.num_classes):
            batch_mean_iou = np.mean(conf_mat[:, i, i] / (sum_col[:, i]+sum_row[:, i]-conf_mat[:, i, i]+1e-8))
            iou_list.append(batch_mean_iou)
        iou_ndarray = np.array(iou_list)
        miou = np.mean(iou_ndarray)
        return miou, iou_list
    
    def precision(self, conf_mat:np.ndarray):
        # confmat shape (N, self.num_classes, self.num_classes)
        sum_col = np.sum(conf_mat, -2)# (N, num_classes) -> 0으로 예측, 1로 예측 2로 예측 각각 sum
        precision_per_class = np.mean(np.array([conf_mat[:, i, i]/ (sum_col[:, i]+1e-7) for i in range(self.num_classes)]), axis=-1) # list안에 (N, )가 클래스개수만큼.-> (3, N) -> 평균->(3,)
        # multi class에 대해 recall / precision을 구할 때에는 클래스 모두 합쳐 평균을 낸다.
        mprecision = np.mean(precision_per_class)
        return mprecision, precision_per_class

    def recall(self, conf_mat:np.ndarray):
        # confmat shape (N, self.num_classes, self.num_classes)
        sum_row = np.sum(conf_mat, -1)# (N, 3) -> 0으로 예측, 1로 예측 2로 예측 각각 sum
        recall_per_class = np.mean(np.array([conf_mat[:, i, i]/ sum_row[:, i] for i in range(self.num_classes)]), axis=-1) # list안에 (N, )가 클래스개수만큼.-> (3, N) -> 평균->(3,)
        mrecall = np.mean(recall_per_class)
        return mrecall, recall_per_class
    
    def f1score(self, recall, precision):
        return 2*recall*precision/(recall + precision)
    
    def measure(self, pred:np.ndarray, target:np.ndarray):
        conf_mat = self._make_confusion_matrix(pred, target)
        acc = self.accuracy(pred, target)
        miou, iou_list = self.miou(conf_mat)
        precision, _ = self.precision(conf_mat)
        recall, _ = self.recall(conf_mat)
        f1score = self.f1score(recall, precision)
        return acc, miou, iou_list, precision, recall, f1score
        
    __call__ = measure
    
def listmean(l:list):
    ret = 0
    for i in range(len(l)):
        ret += l[i]
    ret /= len(l)
    return ret

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from saved weights in multi-GPU training"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[len('module.'):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    return new_state_dict

def evaluate(model, loader, mode, cfg, save_path, ddp=False):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    palette = color_map(cfg['dataset'])
    img_ret = []
    measurement = Measurement(cfg["nclass"])
    if cfg['save_map']:
        save_path = save_path.split('.pth')[0]    # checkpoints/pascal_92_73.7.pth -> checkpoints/pascal_92_73.7
        os.makedirs(os.path.join(save_path, 'mask'), exist_ok=True)           # for mask
        os.makedirs(os.path.join(save_path, 'color_mask'), exist_ok=True)   # for colorized mask
    test_acc, test_miou = 0, 0
    test_precision, test_recall, test_f1score = 0, 0, 0
    iou_per_class = np.array([0]*cfg["nclass"], dtype=np.float64)
    with torch.no_grad():
        for img, mask, ids, img_ori in loader:

            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred_score = model(img)
                # pred = model(img).argmax(dim=1)
                pred = pred_score.argmax(dim=1)

            # Save prediction mask
            # if cfg['save_map']:
            #     img_path, mask_path = id[0].split(' ')
            #     mask_name = os.path.basename(mask_path)
                
            #     pred_map = pred[0].cpu().numpy().astype(np.uint8)
            #     pred_map = Image.fromarray(pred_map)
            #     pred_colormap = pred_map.convert('P')
            #     pred_colormap.putpalette(palette)
                
            #     pred_map.save(os.path.join(save_path, 'mask', mask_name))
            #     pred_colormap.save(os.path.join(save_path, 'color_mask', mask_name))
            acc_pixel, batch_miou, iou_ndarray, precision, recall, f1score = measurement(pred_score.detach().cpu().numpy(), mask.detach().cpu().numpy())
            test_acc += acc_pixel
            test_miou += batch_miou
            iou_per_class += iou_ndarray
            
            test_precision += precision
            test_recall += recall
            test_f1score += f1score
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            if ddp:
                reduced_intersection = torch.from_numpy(intersection).cuda()
                reduced_union = torch.from_numpy(union).cuda()
                reduced_target = torch.from_numpy(target).cuda()

                dist.all_reduce(reduced_intersection)
                dist.all_reduce(reduced_union)
                dist.all_reduce(reduced_target)

                intersection_meter.update(reduced_intersection.cpu().numpy())
                union_meter.update(reduced_union.cpu().numpy())
            else:
                intersection_meter.update(intersection)
                union_meter.update(union)
            viz = make_test_detailed_img(img_ori.detach().cpu().numpy(), pred_score.detach().cpu().numpy(), \
            mask.detach().cpu().numpy())
            img_ret.append([viz, ids[0]])

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)
    test_acc = test_acc / len(loader)
    # test_miou = test_miou / len(loader)
    test_ious = np.round((iou_per_class / len(loader)), 5).tolist()
    test_miou = listmean(test_ious)
    test_precision /= len(loader)
    test_recall /= len(loader)
    # test_f1score /= len(loader)
    test_f1score = (2 * test_precision * test_recall) / (test_precision + test_recall)
    
    result_txt = "load model(.pt) : %s \n Testaccuracy: %.4f, Test miou: %.4f" % (save_path,  test_acc, test_miou)       
    result_txt += f"\niou per class {list(map(lambda x: round(x, 4), test_ious))}"
    result_txt += f"\nprecision : {test_precision:.4f}, recall : {test_recall:.4f}, f1score : {test_f1score:.4f} " 
    print(result_txt)
    result_save_path = os.path.join(".", "test_save_files", cfg["dataset"]+"_test")
    os.makedirs(result_save_path, exist_ok=True)
    with open(os.path.join(result_save_path,"result.txt"), "w") as f:
        f.write(result_txt)
        
    img_save_path = os.path.join(result_save_path, "imgs")
    os.makedirs(img_save_path, exist_ok=True)
    for viz in img_ret:
        save_img(img_save_path, viz[1], viz[0])
    return mIOU, iou_class, img_ret


def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, default="./configs/CWFID_percent30.yaml")
    parser.add_argument('--ckpt-path', type=str, default="D:/PrevMatch/save_files/CWFID_percent305/best.pth")
    parser.add_argument('--save-map', type=str, default=False)

    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg['save_map'] = args.save_map
    
    model = DeepLabV3Plus(cfg)
    ckpt = torch.load(args.ckpt_path)['model']
    ckpt = remove_module_prefix(ckpt) if cfg['dataset'] != 'pascal' else ckpt
    model.load_state_dict(ckpt)
    model.cuda()
    print('Total params: {:.1f}M\n'.format(count_params(model)))

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'test')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    
    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    mIoU, iou_class, img_ret = evaluate(model, valloader, eval_mode, cfg, save_path=args.ckpt_path)

    for (cls_idx, iou) in enumerate(iou_class):
        print('***** Evaluation ***** >>>> Class [{:} {:}] '
                    'IoU: {:.2f}'.format(cls_idx, CLASSES["default"][cls_idx], iou))
    print('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
    
    
if __name__ == '__main__':
    main()
