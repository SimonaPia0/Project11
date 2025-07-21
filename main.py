from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import json

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes_mod, lostandfound
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class BoundaryAwareCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255, boundary_weight=10.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        loss = self.ce(input, target)
        target_np = target.detach().cpu().numpy()
        boundary_mask = np.zeros_like(target_np, dtype=np.uint8)

        for i in range(target_np.shape[0]):
            label = target_np[i]
            edge = cv2.Canny((label * 255).astype(np.uint8), 50, 150)
            boundary_mask[i] = (edge > 0).astype(np.uint8)

        boundary_mask = torch.tensor(boundary_mask).to(target.device).bool()
        loss_boundary = F.cross_entropy(input, target, weight=self.ce.weight, 
                                      ignore_index=self.ignore_index, reduction='none')
        if loss_boundary[boundary_mask].numel() > 0:
            loss += self.boundary_weight * loss_boundary[boundary_mask].mean()
        return loss

def calibra_soglia_confidenza(model, loader, device, percentile=10):
    """Stima una soglia di confidenza MSP con conformal prediction (percentile su pixel corretti)."""
    print("[INFO] Calibrazione soglia conformal prediction...")
    model.eval()
    conf_list = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Calibrazione CP"):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            softmax = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(softmax, dim=1)  # [B, H, W]

            correct = preds.eq(labels).cpu().numpy()
            conf_np = confidence.cpu().numpy()

            conf_list.extend(conf_np[correct])  # solo confidenza sui pixel corretti

    if len(conf_list) == 0:
        print("[WARNING] Nessuna confidenza raccolta (conf_list è vuoto). Restituisco soglia di default = 0.5")
        return 0.5

    threshold = np.percentile(conf_list, percentile)
    print(f"[INFO] Soglia CP al {100 - percentile}% di affidabilità: {threshold:.3f}")
    return threshold

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--img_zip_path", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--ann_zip_path", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'my', 'lostandfound'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss', 'boundary'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)

    """if opts.dataset == 'my':
        train_transform = et.ExtCompose([
            et.ExtResize((96, 256)),
            et.ExtRandomCrop(size=(96, 256), pad_if_needed=True),
            et.ExtColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            et.ExtRandomHorizontalFlip(p=0.5),
            et.ExtRandomVerticalFlip(p=0.2),
            #et.ExtGaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(96, 256)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = cityscapes_mod.Cityscapes_Mod(root=opts.data_root, split='train', transform=train_transform)
        val_dst = cityscapes_mod.Cityscapes_Mod(root=opts.data_root, split='val', transform=val_transform)"""

    if opts.dataset == 'lostandfound':
        # Trasformazioni per LostAndFound, puoi modificarle secondo il dataset
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = lostandfound.LostAndFoundDatasetFromMasks(root=opts.data_root, split='train', transform=train_transform)  # se non usi training su LostAndFound
        val_dst = lostandfound.LostAndFoundDatasetFromMasks(root=opts.data_root, split='test', transform=val_transform)

    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples (senza salvataggio immagini)"""
    metrics.reset()
    ret_samples = []

    threshold = getattr(opts, 'calibrated_threshold', 0.2)
    unknown_label = opts.num_classes - 1

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            softmax = torch.nn.functional.softmax(outputs, dim=1)
            confidence, _ = torch.max(softmax, dim=1)
            preds = outputs.detach().max(dim=1)[1]

            # Mark pixels as unknown if below threshold
            unknown_mask = confidence < threshold
            preds[unknown_mask] = unknown_label

            num_unknown = unknown_mask.sum().item()
            total_pixels = unknown_mask.numel()
            perc_unknown = 100 * num_unknown / total_pixels
            #print(f"[DEBUG] Unknown pixels: {num_unknown} ({perc_unknown:.2f}%)")

            preds_np = preds.cpu().numpy()
            targets = labels.cpu().numpy()

            valid_mask = targets != unknown_label
            targets_filtered = targets[valid_mask]
            preds_filtered = preds_np[valid_mask]

            metrics.update(targets_filtered, preds_filtered)

            if ret_samples_ids is not None and i in ret_samples_ids:
                ret_samples.append((images[0].cpu().numpy(), targets[0], preds_np[0]))

    score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'lostandfound':
        opts.num_classes = 19 + 1

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    if train_dst is not None:
        train_loader = data.DataLoader(
            train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
            drop_last=True)  # drop_last=True to ignore single-image batches.
    else:
        train_loader = None

    val_loader = data.DataLoader(
            val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
        
    if len(val_loader) == 0:
        print("[!] Val loader is empty!")
        return

    print("Dataset: %s, Train set: %s, Val set: %d" % (
    opts.dataset,
    len(train_dst) if train_dst is not None else 'N/A',
    len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    if train_dst is not None:
        # === Calcolo automatico dei pesi di classe ===
        print("Calcolo pesi per CrossEntropyLoss...")

        class_counts = torch.zeros(opts.num_classes)
        total_pixels = 0

        tmp_loader = data.DataLoader(train_dst, batch_size=1, shuffle=False)

        for _, label in tqdm(tmp_loader, desc="Scanning dataset"):
            label = label.squeeze()  # [H, W]
            for cls in range(opts.num_classes):
                class_counts[cls] += torch.sum(label == cls).item()
            total_pixels += label.numel()

        # Frequenza inversa normalizzata
        class_freq = class_counts / total_pixels
        weights = 1.0 / (class_freq + 1e-6)  # evita divisione per zero
        weights = weights / weights.sum() * opts.num_classes  # normalizzazione (opzionale)

        print("Class weights:", weights)
        weights = weights.to(device)
    else:
        weights = None

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        if weights is not None:
            criterion = nn.CrossEntropyLoss(ignore_index=255, weight=weights, reduction='mean')
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'boundary':
        if weights is not None:
            criterion = BoundaryAwareCrossEntropyLoss(weight=weights, ignore_index=255, boundary_weight=5.0)
        else:
            criterion = BoundaryAwareCrossEntropyLoss(ignore_index=255, boundary_weight=5.0)

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        

        """if opts.dataset.lower() == 'lostandfound':
            checkpoint = torch.load(opts.ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state'], strict=True)
            model = nn.DataParallel(model)
            model.to(device)
        else:"""
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
        # Carica solo i pesi del backbone e del resto tranne la testa
        state_dict = checkpoint['model_state']
        filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.classifier.3')}
        model.load_state_dict(filtered_dict, strict=False)

        # Metti la testa con il numero di classi corretto (già fatta al momento della creazione del modello)
        model = nn.DataParallel(model)
        model.to(device)


        optimizer = torch.optim.SGD(params=[
            {'params': model.module.backbone.parameters(), 'lr': 0.001},
            {'params': model.module.classifier.parameters(), 'lr': 0.01},
        ], lr=0.01, momentum=0.9, weight_decay=opts.weight_decay)

        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        print(f"Cur itrs: {cur_itrs}, Total itrs: {opts.total_itrs}")
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        # Calibrazione soglia CP
        threshold_cp = calibra_soglia_confidenza(model, val_loader, device, percentile=10)
        opts.calibrated_threshold = threshold_cp

        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device,
            metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))

        all_pred_scores = []
        all_gt_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)  # [B, C, H, W]
                probs = torch.softmax(outputs, dim=1)  # [B, C, H, W]
                confidence, preds = torch.max(probs, dim=1)  # [B, H, W]

                # Anomaly mask: confidenza sotto la soglia
                anomaly_scores = (1.0 - confidence).view(-1).cpu()  # inverso: più alto = più anomalo
                all_pred_scores.append(anomaly_scores)

                # GT anomaly: label == 20 (o qualsiasi classe "anomala")
                gt_anomaly_mask = (labels == 20).view(-1).cpu().int()
                all_gt_labels.append(gt_anomaly_mask)

        all_pred_scores = torch.cat(all_pred_scores).numpy()
        all_gt_labels = torch.cat(all_gt_labels).numpy()

        roc_auc = roc_auc_score(all_gt_labels, all_pred_scores)
        precision, recall, _ = precision_recall_curve(all_gt_labels, all_pred_scores)
        pr_auc = auc(recall, precision)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        f1_max = np.max(f1_scores)

        print(f"Anomaly Detection Metrics:\nROC-AUC: {roc_auc:.4f}\nPR-AUC: {pr_auc:.4f}\nMax F1: {f1_max:.4f}")

        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()