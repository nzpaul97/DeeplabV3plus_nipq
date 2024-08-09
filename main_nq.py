from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import time

from my_lib.train_test import (
    resume_checkpoint,
    create_checkpoint,
    test,
    train_aux,
    # train_aux_target_avgbit,
    train_aux_target_bops,
    CosineWithWarmup,
    bops_cal,
    bit_loss,
    accuracy
)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/cephfs/seunghun/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
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
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step', 'cosine_warmup'],
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
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
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
    
    # NIPQ part
    parser.add_argument("--mode", default='avgbit', choices=['avgbit', 'bops'], help='average bit mode')

    parser.add_argument("--a_scale", default=0, type=float)
    parser.add_argument("--w_scale", default=0, type=float)
    parser.add_argument("--bops_scale", default=0, type=float, help='using teacher')

    parser.add_argument("--target", default=4, type=float, help='target bitops or avgbit')

    parser.add_argument("--ckpt_path", help="checkpoint directory", default='./checkpoint')

    parser.add_argument("--warmuplen", default=3, type=int, help='scheduler warm up epoch')
    parser.add_argument("--ft_epoch", default=3, type=int, help='tuning epoch')

    parser.add_argument("--ts", action='store_true', help='using teacher')

    parser.add_argument("--ts_model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='teacher model name')

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
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():

    def train_aux_target_avgbit(train_loader, model, model_ema, model_t, criterion, optimizer, epoch, 
                        ema_rate=0.9999, bit_scale_a = 0, bit_scale_w = 0, target_ours=None, scale=0):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        model.train()

        if model_t is not None:
            model_t.eval()
            model_t.cuda()

        end = time.time()

        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if not isinstance(model, torch.nn.DataParallel):
                input = input.cuda()

            target = target.to(device="cuda", dtype=torch.long)
            target = target.cuda(non_blocking=True)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input).cuda()
                target_var = torch.autograd.Variable(target)

                if model_t is not None:
                    output_t = model_t(input_var)      
            output = model(input_var)

            loss = None        
            loss_class = criterion(output, target_var)
            
            loss_bit = bit_loss(model, epoch, bit_scale_a, bit_scale_w, target_ours, False)
            loss_class = loss_class + loss_bit
            
            if model_t is not None:
                loss_kd = -1 * torch.mean(
                    torch.sum(torch.nn.functional.softmax(output_t, dim=1) 
                            * torch.nn.functional.log_softmax(output, dim=1), dim=1))
                loss = loss_class + loss_kd 
            else:
                loss = loss_class
            losses.update(loss_class.data.item(), input.size(0))

            # measure accuracy and record loss
            if isinstance(output, tuple):
                prec1, prec5 = accuracy(output[0].data, target, topk=(1, 5))
            else:
                # print(output.data.shape, target.shape)
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if model_ema is not None:
                for module, module_ema in zip(model.module.modules(), model_ema.module.modules()):
                    target = []

                    if hasattr(module, "c"):    # QIL
                        target.append("c")
                        target.append("d")
                    
                    if hasattr(module, "p"):    # PACT
                        target.append("p")

                    if hasattr(module, "s"):    # lsq
                        target.append("s")

                    if hasattr(module, "e"):    # proposed
                        target.append("e")    
                        target.append("f")  


                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                        target.extend(["weight", "bias"])
                        
                        if hasattr(module, "scale"):
                            target.extend(["scale", "shift"])

                    if isinstance(module, (torch.nn.BatchNorm2d)):
                        target.extend(["weight", "bias", "running_mean", "running_var"])

                        if module.num_batches_tracked is not None:
                            module_ema.num_batches_tracked.data = module.num_batches_tracked.data

                    for t in target:
                        base = getattr(module, t, None)    
                        ema = getattr(module_ema, t, None)    

                        if base is not None and hasattr(base, "data"):                        
                            ema.data += (1 - ema_rate) * (base.data - ema.data)   

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _print_freq = 10
            if ((i+1) % _print_freq) == 0:
                numel_a = 0
                numel_w = 0
                loss_bit_a = 0
                loss_bit_au = 0
                loss_bit_w = 0
                loss_bit_wu = 0
                
                for name, module in model.named_modules():
                    if hasattr(module, "bit") and hasattr(module, "weight") and module.quant:
                        bit = 2 + torch.sigmoid(module.bit)*12
                        loss_bit_w += bit * module.weight.numel()
                        loss_bit_wu += torch.round(bit) * module.weight.numel()
                        numel_w += module.weight.numel()
                        
                    if hasattr(module, "bit") and hasattr(module, "out_shape") and module.quant:
                        bit = 2 + torch.sigmoid(module.bit)*12
                        loss_bit_a += bit * np.prod(module.out_shape)
                        loss_bit_au += torch.round(bit) * np.prod(module.out_shape)
                        numel_a += np.prod(module.out_shape)
                    
                if numel_a > 0:
                    a_bit = (loss_bit_a / numel_a).item()
                    au_bit = (loss_bit_au / numel_a).item()
                else:
                    a_bit = -1
                    au_bit = -1
                
                if numel_w > 0:
                    w_bit = (loss_bit_w / numel_w).item()
                    wu_bit = (loss_bit_wu / numel_w).item()
                else:
                    w_bit = -1
                    wu_bit = -1
                    
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                    'a bit {a_bit:.2f}[{au_bit:.2f}]\t'
                    'w bit {w_bit:.2f}[{wu_bit:.2f}]\t'.format(
                        epoch, i+1, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5, 
                        a_bit = a_bit, au_bit = au_bit, 
                        w_bit = w_bit, wu_bit = wu_bit))          
        return top1.avg, losses.avg, None

    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

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
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))
    test_loader = val_loader

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        if 'nq' in opts.model:
            network.convert_to_separable_conv_nq(model.classifier)
        else:
            network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    if opts.ckpt:
        ckpt = torch.load(opts.ckpt, map_location=torch.device('cpu')) #NOTE
        if opts.test_only:
            model.load_state_dict(ckpt, False) #NOTE    
        else:
            model.load_state_dict(ckpt['model_state'], False) #NOTE

    ### NIPQ PART
    if opts.mode == 'avgbit':
        print(f'** act_q : {opts.a_scale} / weight_q : {opts.w_scale}')
    elif opts.mode == 'bops':
        print(f'** bops scale : {opts.bops_scale}')
    else :
        raise NotImplementedError()

    from my_lib.nipq import QuantOps as Q
    Q.initialize(model, act=opts.a_scale > 0, weight=opts.w_scale > 0)
    
    img = torch.Tensor(1, 3, 224, 224) if opts.dataset =='imagenet' else torch.Tensor(1, 3, 513, 513)  

    def forward_hook(module, inputs, outputs):
        module.out_shape = outputs.shape
        
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (Q.ReLU, Q.Sym, Q.HSwish)):
            hooks.append(module.register_forward_hook(forward_hook))
        
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(forward_hook))

    model.eval()
    model.cuda()
    model(img.cuda())

    for hook in hooks:
        hook.remove()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

        # if model_ema is not None:
        #     model_ema.cuda()
        #     model_ema = torch.nn.DataParallel(model_ema, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    ###

    def bit_cal(model):
        numel_a = 0
        numel_w = 0
        loss_bit_a = 0
        loss_bit_au = 0
        loss_bit_w = 0
        loss_bit_wu = 0

        w_bit=-1
        a_bit=-1
        au_bit=-1
        wu_bit=-1
        for name, module in model.named_modules():
            if hasattr(module, "bit") and hasattr(module, "weight") and module.quant:
                bit = 2 + torch.sigmoid(module.bit)*12
                loss_bit_w += bit * module.weight.numel()
                loss_bit_wu += torch.round(bit) * module.weight.numel()
                numel_w += module.weight.numel()
                
            if hasattr(module, "bit") and hasattr(module, "out_shape") and module.quant:
                bit = 2 + torch.sigmoid(module.bit)*12
                loss_bit_a += bit * np.prod(module.out_shape)
                loss_bit_au += torch.round(bit) * np.prod(module.out_shape)
                numel_a += np.prod(module.out_shape)
            
        if numel_a > 0:
            a_bit = (loss_bit_a / numel_a).item()
            au_bit = (loss_bit_au / numel_a).item()

        if numel_w > 0:
            w_bit = (loss_bit_w / numel_w).item()
            wu_bit = (loss_bit_wu / numel_w).item()
        
        return a_bit, au_bit, w_bit, wu_bit

    def categorize_param(model):    
        weight = []
        bnbias = []   

        for name, param in model.named_parameters():         
            if not param.requires_grad:
                continue  
            elif len(param.shape) == 1 or name.endswith(".bias"): # bnbias and quant parameters
                bnbias.append(param)
            else:
                weight.append(param)
        return (weight, bnbias)

    # Set up optimizer
    ### TODO
    # not sure about the optimizer part

    backbone_weight, backbone_bnbias = categorize_param(model.module.backbone)
    classifier_weight, classifier_bnbias = categorize_param(model.module.classifier)

    optimizer = torch.optim.SGD(params=[
        {'params': backbone_bnbias, 'weight_decay': 0., 'lr': opts.lr},
        {'params': backbone_weight, 'weight_decay': opts.weight_decay, 'lr': opts.lr},
        {'params': classifier_bnbias, 'weight_decay': 0., 'lr': opts.lr},
        {'params': classifier_weight, 'weight_decay': opts.weight_decay, 'lr': opts.lr},
        # {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        # {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, nesterov=True)

    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs+opts.ft_epoch, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    elif opts.lr_policy == 'cosine_warmup':
        scheduler = CosineWithWarmup(optimizer, warmup_len=opts.warmuplen, warmup_start_multiplier=0.1,
                max_epochs=opts.total_itrs+opts.ft_epoch, eta_min=1e-3, last_epoch=-1)
    
    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


    best_acc = 0
    best_acc_ema = 0 

    if opts.mode == 'bops' :
        print(f'mode : {opts.mode}')
        train_aux_target_avgbit = train_aux_target_bops
    model_ema = None
    model_t = None
    
    if opts.ts :
        print('==> Using teacher model')

        model_t = network.modeling.__dict__[opts.ts_model](num_classes=opts.num_classes, output_stride=opts.output_stride)

        ckpt_t = './checkpoints/baseline_deeplabv3plus_mobilenet_voc_os16.pth'

        ckpt_t = torch.load(ckpt_t) #NOTE
        model_t.load_state_dict(ckpt_t, False) #NOTE

        for params in model_t.parameters():
            params.requires_grad = False

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
        print(metrics.to_str(val_score))
        return

    # print(type(opts.total_itrs), opts.total_itrs)
    for epoch in range(int(opts.total_itrs)):
        train_acc, train_loss, _ = train_aux_target_avgbit(train_loader, model, model_ema, model_t, criterion, optimizer, epoch, 
                                                            ema_rate=0.99, bit_scale_a = opts.a_scale, bit_scale_w = opts.w_scale,
                                                            target_ours = opts.target, scale=opts.bops_scale)
        # train_regular(train_loader, model, model_ema, None, 0, 0, criterion, optimizer, epoch, [1.,], ema_rate=0.9997)
        acc_base, test_loss = test(test_loader, model, criterion, epoch, False)

        acc_ema = 0        
        if model_ema is not None:
            acc_ema, loss_ema = test(test_loader, model_ema, criterion, epoch)

        a_bit, au_bit, w_bit, wu_bit = bit_cal(model)
        bops_total = bops_cal(model)
        print(f'Epoch : [{epoch}] / a_bit : {au_bit}bit / w_bit : {wu_bit}bit / bops : {bops_total.item()}GBops')

        metrics = StreamSegMetrics(opts.num_classes)
        model.eval()
        val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)  
        print(metrics.to_str(val_score))   

        is_best = acc_base > best_acc
        best_acc = max(acc_base, best_acc)
        
        is_best_ema = acc_ema > best_acc_ema
        best_acc_ema = max(acc_ema, best_acc_ema)    
        
        print('==> Save the model')  
        prefix = "%s_%s" % (opts.model, opts.dataset)
        if opts.ckpt_path is not None:
            create_checkpoint(
                optimizer, epoch, opts.ckpt_path,
                model, is_best, best_acc, 
                model_ema, is_best_ema, best_acc_ema, prefix=prefix)
        scheduler.step()   

    metrics = StreamSegMetrics(opts.num_classes)
    model.eval()
    val_score, ret_samples = validate(
                opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)  
    print(metrics.to_str(val_score))                  


    # BN tuning phase
    Q.initialize(model, act=opts.a_scale > 0, weight=opts.w_scale > 0, noise=False)

    for name, module in model.named_modules():
        if isinstance(module, (Q.ReLU, Q.Sym, Q.HSwish, Q.Conv2d, Q.Linear)):
            module.bit.requires_grad = False

    best_acc = 0
    best_acc_ema = 0 


    for epoch in range(opts.total_itrs, opts.total_itrs+opts.ft_epoch):
        train_acc, train_loss, _  = train_aux(train_loader, model, model_ema, model_t, criterion, optimizer, epoch, 
                                            ema_rate=0.99, bit_scale_a = 0, bit_scale_w = 0)
        acc_base, test_loss = test(test_loader, model, criterion, epoch, False)

        acc_ema = 0        
        if model_ema is not None:
            acc_ema, _ = test(test_loader, model_ema, criterion, epoch)
            
        metrics = StreamSegMetrics(opts.num_classes)
        model.eval()
        val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)  
        print(metrics.to_str(val_score))   

        is_best = acc_base > best_acc
        best_acc = max(acc_base, best_acc)
        
        is_best_ema = acc_ema > best_acc_ema
        best_acc_ema = max(acc_ema, best_acc_ema)    

        if opts.ckpt_path is not None:
            create_checkpoint(
                optimizer, epoch, opts.ckpt_path,
                model, is_best, best_acc, 
                model_ema, is_best_ema, best_acc_ema, prefix=prefix)
        scheduler.step()                 

    metrics = StreamSegMetrics(opts.num_classes)
    model.eval()
    val_score, ret_samples = validate(
                opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)  
    print(metrics.to_str(val_score))

    # def save_ckpt(path):
    #     """ save current model
    #     """
    #     torch.save({
    #         "cur_itrs": cur_itrs,
    #         "model_state": model.module.state_dict(),
    #         "optimizer_state": optimizer.state_dict(),
    #         "scheduler_state": scheduler.state_dict(),
    #         "best_score": best_score,
    #     }, path)
    #     print("Model saved as %s" % path)

    # utils.mkdir('checkpoints')
    # # Restore
    # best_score = 0.0
    # cur_itrs = 0
    # cur_epochs = 0
    # if opts.ckpt is not None and os.path.isfile(opts.ckpt):
    #     # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
    #     checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    #     model.load_state_dict(checkpoint["model_state"])
    #     model = nn.DataParallel(model)
    #     model.to(device)
    #     if opts.continue_training:
    #         optimizer.load_state_dict(checkpoint["optimizer_state"])
    #         scheduler.load_state_dict(checkpoint["scheduler_state"])
    #         cur_itrs = checkpoint["cur_itrs"]
    #         best_score = checkpoint['best_score']
    #         print("Training state restored from %s" % opts.ckpt)
    #     print("Model restored from %s" % opts.ckpt)
    #     del checkpoint  # free memory
    # else:
    #     print("[!] Retrain")
    #     model = nn.DataParallel(model)
    #     model.to(device)

    # # ==========   Train Loop   ==========#
    # vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
    #                                   np.int32) if opts.enable_vis else None  # sample idxs for visualization
    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    # if opts.test_only:
    #     model.eval()
    #     val_score, ret_samples = validate(
    #         opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
    #     print(metrics.to_str(val_score))
    #     return

    # interval_loss = 0
    # while True:  # cur_itrs < opts.total_itrs:
    #     # =====  Train  =====
    #     model.train()
    #     cur_epochs += 1
    #     for (images, labels) in train_loader:
    #         cur_itrs += 1

    #         images = images.to(device, dtype=torch.float32)
    #         labels = labels.to(device, dtype=torch.long)

    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         np_loss = loss.detach().cpu().numpy()
    #         interval_loss += np_loss
    #         if vis is not None:
    #             vis.vis_scalar('Loss', cur_itrs, np_loss)

    #         if (cur_itrs) % 10 == 0:
    #             interval_loss = interval_loss / 10
    #             print("Epoch %d, Itrs %d/%d, Loss=%f" %
    #                   (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
    #             interval_loss = 0.0

    #         if (cur_itrs) % opts.val_interval == 0:
    #             save_ckpt('checkpoints/0620/latest_%s_%s_os%d.pth' %
    #                       (opts.model, opts.dataset, opts.output_stride))
    #             print("validation...")
    #             model.eval()
    #             val_score, ret_samples = validate(
    #                 opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
    #                 ret_samples_ids=vis_sample_id)
    #             print(metrics.to_str(val_score))
    #             if val_score['Mean IoU'] > best_score:  # save best model
    #                 best_score = val_score['Mean IoU']
    #                 save_ckpt('checkpoints/0620/best_%s_%s_os%d.pth' %
    #                           (opts.model, opts.dataset, opts.output_stride))

    #             if vis is not None:  # visualize validation score and samples
    #                 vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
    #                 vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
    #                 vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

    #                 for k, (img, target, lbl) in enumerate(ret_samples):
    #                     img = (denorm(img) * 255).astype(np.uint8)
    #                     target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
    #                     lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
    #                     concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
    #                     vis.vis_image('Sample %d' % k, concat_img)
    #             model.train()
    #         scheduler.step()

    #         if cur_itrs >= opts.total_itrs:
    #             return


if __name__ == '__main__':
    main()
