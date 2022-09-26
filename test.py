import os
import argparse
import h5py
import math
import numpy as np
from tqdm import tqdm
import nibabel as nib
from medpy import metric
import torch
import torch.nn.functional as F

from model import Net

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--save', type=str,  default='0', help='GPU to use')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_all_case(noise_net, net, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, test_save_path=None):
    total_metric = 0.0
    for image_path in tqdm(image_list):
        if '\\' in image_path:
            id = image_path.split('\\')[-2]
        else:
            id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map, noise_map, uncertainty_map = test_single_case(noise_net, net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        total_metric += np.asarray(single_metric)

        if args.save == '1':
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(score_map.astype(np.float32), np.eye(4)), test_save_path + id + "_score.nii.gz")
            nib.save(nib.Nifti1Image(uncertainty_map.astype(np.float32), np.eye(4)), test_save_path + id + "_uncertainty.nii.gz")
            nib.save(nib.Nifti1Image(noise_map.astype(np.float32), np.eye(4)), test_save_path + id + "_noise.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    return avg_metric

def test_single_case(noise_net, net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    uncertainty_map = np.zeros(image.shape).astype(np.float32)
    noise_map = np.zeros(image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).to(device)
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                noise1 = noise_net.gen_noise(test_patch)
                noise = noise1.cpu().data.numpy()
                noise = noise[0, 0, :, :, :]

                T = 8
                stride = 1
                preds = torch.zeros([stride * T, 2, 112, 112, 80]).to(device)
                for i in range(T//2):
                    noise_for_monto_carlo = noise_net.gen_noise(test_patch) + torch.clamp(torch.randn_like(test_patch) * 0.1, -0.2, 0.2)
                    inputs = test_patch + noise_for_monto_carlo
                    with torch.no_grad():
                        preds[2 * stride * i:2 *stride * (i + 1)] = net(inputs)
                preds = F.softmax(preds, dim=1)
                preds = preds.reshape(T, stride, 2, 112, 112, 80)
                preds = torch.mean(preds, dim=0)
                uncertainty = -1.0 * torch.sum(preds*torch.log(preds + 1e-6), dim=0)
                uncertainty = uncertainty[0].cpu().data.numpy()

                test_patch = test_patch.cpu().data.numpy()
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y + np.expand_dims(test_patch, axis=0)
                noise_map[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = noise_map[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + noise
                uncertainty_map[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = uncertainty_map[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + uncertainty
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    noise_map = noise_map / cnt
    uncertainty_map = uncertainty_map/cnt
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        noise_map = noise_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        uncertainty_map = uncertainty_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map[0], noise_map, uncertainty_map

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, jc, hd, asd

def test_calculate_metric():
    net = Net(1, 2, 16, has_dropout=False).to(device)
    net.load_model('./model.pth')
    net.eval()

    noise_net = Net(1, 2, 16, has_dropout=False).to(device)
    noise_net.load_model('./noise.pth')
    noise_net.eval()

    with open(os.path.join('dataset', 'test.list'), 'r') as f:
        image_list = f.readlines()
    image_list = [os.path.join('dataset', os.path.join(item.replace('\n', ''), 'mri_norm2.h5')) for item in image_list]

    avg_metric = test_all_case(noise_net, net, image_list, num_classes=2, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, test_save_path='./prediction/')
    return avg_metric

if __name__ == '__main__':
    test_calculate_metric()
