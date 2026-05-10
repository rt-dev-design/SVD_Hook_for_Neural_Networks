import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from sklearn.metrics import confusion_matrix

import os
import time
import argparse

from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter

from util.utils import *
from util.logger import build_logger
from util.experiment_setting_hash import combined_hash
from util.hook_handler import build_hook_handler
from util.effective_step_size_statistics import *


parser = argparse.ArgumentParser(description='Hello World Deep Learning: ResNet18 MNIST Classification')

# dataset and data loading
parser.add_argument('--dataset', default='mnist', type=str, help='mnist')
parser.add_argument('--data_path', default='./data/', type=str, help='path to the directory containing the MNIST dataset')
parser.add_argument('--image_width', default=224, type=int, help='image width to resize to')
parser.add_argument('--image_height', default=224, type=int, help='image height to resize to')
parser.add_argument('--num_classes', default=10, type=int, help='number of digit classes')
parser.add_argument('--num_workers', default=6, type=int, help='number of workers for the data loader')

# model building parameters
parser.add_argument('--backbone', default='resnet18', type=str, help='currently only resnet18 is used in this hello world example')
parser.add_argument('--use_pretrained_cnn', default='False', type=str2bool, help='use pretrained cnn weights to initialize or not')
parser.add_argument('--dropout', default=0.0, type=float, help='dropout before classifier')

# initialization or checkpoint loading
parser.add_argument('--load_checkpoint', default='False', type=str2bool, help='whether to load a checkpoint for training')
parser.add_argument('--checkpoint_path', default='', type=str, help='checkpoint path')
parser.add_argument('--std_for_init', default=0.04, type=float, help='std for initializing classifier weights')

# training parameters
parser.add_argument('--random_seed', default=1, type=int, help='random seed set in all sorts of library code for reproduction')
parser.add_argument('--max_epochs', default=5, type=int, help='max number of epochs to train for')
parser.add_argument('--train_batch', default=128, type=int, help='train batch size')

parser.add_argument('--max_lr', default=1e-3, type=float, help='target learning rate for the one cycle scheduler')
parser.add_argument('--max_lr_backbone', default=1e-3, type=float, help='target learning rate for finetuning the backbone')
parser.add_argument('--pct_start', default=0.1, type=float, help='pct_start for the one cycle scheduler')

parser.add_argument('--beta_1', default=0.90, type=float, help='beta_1 for Adam')
parser.add_argument('--beta_2', default=0.99, type=float, help='beta_2 for Adam')
parser.add_argument('--eps', default=1e-8, type=float, help='eps for Adam')

parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
parser.add_argument('--gradient_clipping', default='False', type=str2bool, help='whether to use gradient clipping')
parser.add_argument('--max_norm', default=1.0, type=float, help='max norm used in gradient clipping')

# testing parameters
parser.add_argument('--test_freq', default=1, type=int, help='how often in terms of epochs to test during training')
parser.add_argument('--test_batch', default=256, type=int, help='test batch size')
parser.add_argument('--test_before_train', default='False', type=str2bool, help='whether to test first right before any training')

# GPU and hardware related parameters
parser.add_argument('--device', default='0', type=str, help='CUDA_VISIBLE_DEVICES, as well as the actual GPU indices the program will see')
parser.add_argument('--developing_using_very_little_gpu', default='False', type=str2bool, help='whether to launch a dummy development experiment using only a little GPU')

# monitoring and bookkeeping
parser.add_argument('--why_what_how_of_this_experiment', default="", type=str, help='add description of this experiment')
parser.add_argument('--enable_hook_feature', default='True', type=str2bool, help='whether to enable some Pytorch hook monitoring of the model during training')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

IGNORED_ARGS_FOR_HASHING = [
        'device', 'num_workers', 'random_seed',
        'data_path',
        'test_freq', 'test_before_train',
        'developing_using_very_little_gpu',
        'why_what_how_of_this_experiment', 'enable_hook_feature',
        'experiment_name', 'save_path', 'test_batch'
    ]

best_mca = 0.0
best_mpca = 0.0
best_mca_epoch = 0
best_mpca_epoch = 0
time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
exp_name = '%s_ResNet18_%s' % (args.dataset, time_str)
save_path = './result/%s' % exp_name
args.experiment_name = exp_name
args.save_path = save_path

logger = build_logger("logger", save_path, "log.txt", use_this_logger_for_global_exceptions=True)
batch_logger = build_logger("batch_logger", save_path, "batch_log.txt")
effective_step_size_logger = build_logger("effective_step_size_logger", save_path, "effective_size_log.txt")
writer = SummaryWriter(save_path + "/tensorboard")
append_text_to_file(save_path, "description.txt", args.why_what_how_of_this_experiment)


class ResNet18ForMNIST(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.backbone != 'resnet18':
            raise NotImplementedError('This hello world file currently supports only ResNet18.')

        if args.use_pretrained_cnn:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            backbone = models.resnet18(weights=None)

        # MNIST is grayscale. Change the first convolution from 3 input channels to 1.
        backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.dropout = nn.Dropout(args.dropout) if args.dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(in_features, args.num_classes)

    def forward(self, images):
        features = self.backbone(images)
        features = self.dropout(features)
        score = self.classifier(features)
        return score


def build_mnist_dataset(args):
    transform = transforms.Compose([
        transforms.Resize((args.image_height, args.image_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(
        root=args.data_path,
        train=True,
        download=True,
        transform=transform
    )

    test_set = datasets.MNIST(
        root=args.data_path,
        train=False,
        download=True,
        transform=transform
    )

    return train_set, test_set


def main():
    set_random_seeds(args.random_seed)

    train_set, test_set = build_mnist_dataset(args)
    if args.developing_using_very_little_gpu:
        steal_only_a_little_gpu(train_set)
        steal_only_a_little_gpu(test_set)

    train_loader = data.DataLoader(
        train_set,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = data.DataLoader(
        test_set,
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = torch.nn.DataParallel(ResNet18ForMNIST(args)).cuda()

    if args.enable_hook_feature:
        hook_handler = build_hook_handler(
            args, logger, writer, model,
            [
                'module.backbone.conv1',
                'module.backbone.layer1',
                'module.backbone.layer2',
                'module.backbone.layer3',
                'module.backbone.layer4',
                'module.classifier'
            ],
            100,
        )
        hook_handler.hook()

    criterion = nn.CrossEntropyLoss().cuda()

    backbone_parameters = [p for name, p in model.named_parameters() if 'module.backbone' in name]
    classifier_parameters = [p for name, p in model.named_parameters() if 'module.backbone' not in name]

    optimizer = torch.optim.AdamW([
        {'params': backbone_parameters, 'lr': args.max_lr_backbone},
        {'params': classifier_parameters, 'lr': args.max_lr},
    ], args.max_lr, betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=args.weight_decay)

    div_factor = 100
    final_div_factor = 10000
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=[args.max_lr_backbone, args.max_lr],
        epochs=args.max_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=args.pct_start,
        cycle_momentum=False,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )

    logger.info("arguments:")
    logger.info(args)
    logger.info("model:")
    logger.info('Number of parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    logger.info(model)

    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info("Model loaded from %s at epoch %d" % (args.checkpoint_path, start_epoch))
    else:
        start_epoch = 1
        initialize_model_parameters(model, args)
        logger.info('Initialized model parameters using the scheme defined in the method initialize_model_parameters and the hyperparameters for pretrained weights')

    logger.info("experiment setting hash: " + combined_hash(args, model, ignore_in_ns=IGNORED_ARGS_FOR_HASHING))

    if args.test_before_train:
        logger.info("Testing the just-initialized model before any training")
        if args.enable_hook_feature:
            hook_handler.set_enabled(False)
        test_log = validate(test_loader, model, criterion, 0)
        if args.enable_hook_feature:
            hook_handler.set_enabled(True)
        logger.info('accuracy: %.2f%%, mean-acc: %.2f%%, loss: %.4f, time: %.1fs' % (test_log['group_acc'], test_log['mean_acc'], test_log['loss'], test_log['time']))
        writer.add_scalars("loss", {
            'test': test_log['loss'],
        }, 0)
        writer.add_scalars("accuracy", {
            'test': test_log['group_acc'],
        }, 0)

    for epoch in range(start_epoch, args.max_epochs + 1):
        logger.info('Training at epoch %d/%d' % (epoch, args.max_epochs))
        train_log = train(train_loader, model, criterion, optimizer, scheduler, hook_handler if args.enable_hook_feature else None, epoch)
        logger.info('accuracy: %.2f%%, loss: %.4f, time: %.1fs, current lr: %s' % (train_log['group_acc'], train_log['loss'], train_log['time'], scheduler.get_last_lr()))

        if epoch % args.test_freq == 0:
            logger.info('Testing at epoch %d' % (epoch))
            if args.enable_hook_feature:
                hook_handler.set_enabled(False)
            test_log = validate(test_loader, model, criterion, epoch)
            if args.enable_hook_feature:
                hook_handler.set_enabled(True)
            logger.info('accuracy: %.2f%%, mean-acc: %.2f%%, loss: %.4f, time: %.1fs' % (test_log['group_acc'], test_log['mean_acc'], test_log['loss'], test_log['time']))
            logger.info('So far, best MCA %.2f%% occurred at epoch %d.' % (test_log['best_mca'], test_log['best_mca_epoch']))
            logger.info('So far, best MPCA %.2f%% occurred at epoch %d.' % (test_log['best_mpca'], test_log['best_mpca_epoch']))

            writer.add_scalars("loss", {
                'train': train_log['loss'],
                'test': test_log['loss'],
            }, epoch)
            writer.add_scalars("accuracy", {
                'train': train_log['group_acc'],
                'test': test_log['group_acc'],
            }, epoch)

            if epoch == test_log['best_mca_epoch'] or epoch == test_log['best_mpca_epoch']:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                result_path = save_path + '/epoch%d_%.2f%%.pth' % (epoch, test_log['group_acc'])
                torch.save(state, result_path)
                logger.info("Saved checkpoint to %s at epoch %d." % (result_path, epoch))

    writer.close()
    if args.enable_hook_feature:
        hook_handler.unhook()


def train(train_loader, model, criterion, optimizer, scheduler, hook_handler, epoch):
    """Train for one epoch on the training set"""
    epoch_timer = Timer()
    batch_timer = Timer()
    losses = AverageMeter()
    accuracies = AverageMeter()
    num_batches = len(train_loader)

    model.train()
    for i, (images, activity) in enumerate(train_loader):
        batch_size = images.shape[0]
        images = images.cuda()
        activity = activity.cuda()

        score = model(images)
        loss = criterion(score, activity)
        group_acc = accuracy(score, activity)

        losses.update(loss, batch_size)
        accuracies.update(group_acc, batch_size)

        optimizer.zero_grad()
        loss.backward()
        if args.gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        scheduler.step()

        batch_logger.info("batch %d/%d in epoch %d, loss: %.8f, accuracy: %.8f, lr: %s, time: %.2fs" % (i + 1, num_batches, epoch, loss, group_acc, scheduler.get_last_lr(), batch_timer.timeit()))

        if hook_handler is not None and hook_handler.should_monitor():
            g_norm = calculate_grad_norm(model)
            efs = effective_step_stats(optimizer)
            effective_step_size_logger.info("batch %d/%d in epoch %d, gradient norm: %s" % (i + 1, num_batches, epoch, g_norm))
            effective_step_size_logger.info("batch %d/%d in epoch %d, effective step size: %s" % (i + 1, num_batches, epoch, efs))
            writer.add_scalar('grad_norm', g_norm, (epoch - 1) * num_batches + i)
            for group_index, step_stats in efs.items():
                writer.add_scalar('effective_step/group%d/mean' % (group_index), step_stats['mean'], (epoch - 1) * num_batches + i)
                writer.add_scalar('effective_step/group%d/max' % (group_index), step_stats['max'], (epoch - 1) * num_batches + i)
                writer.add_scalar('effective_step/group%d/median' % (group_index), step_stats['median'], (epoch - 1) * num_batches + i)

    train_log = {
        'epoch': epoch,
        'time': epoch_timer.timeit(),
        'loss': losses.avg,
        'group_acc': accuracies.avg * 100.0
    }
    return train_log


@torch.no_grad()
def validate(test_loader, model, criterion, epoch):
    global best_mca, best_mpca, best_mca_epoch, best_mpca_epoch
    epoch_timer = Timer()
    batch_timer = Timer()
    num_batches = len(test_loader)
    losses = AverageMeter()
    accuracies = AverageMeter()
    true = []
    pred = []

    model.eval()
    for i, (images, activity) in enumerate(test_loader):
        batch_size = images.shape[0]
        images = images.cuda()
        activity = activity.cuda()

        score = model(images)
        true = true + activity.tolist()
        pred = pred + torch.argmax(score, dim=1).tolist()

        loss = criterion(score, activity)
        group_acc = accuracy(score, activity)

        losses.update(loss, batch_size)
        accuracies.update(group_acc, batch_size)

        batch_logger.info("batch %d/%d in epoch %d, loss: %.8f, accuracy: %.8f, time: %.2fs" % (i + 1, num_batches, epoch, loss, group_acc, batch_timer.timeit()))

    acc = accuracies.avg * 100.0
    confusion = confusion_matrix(true, pred, labels=list(range(args.num_classes)))
    mean_acc = np.mean([
        confusion[i, i] / confusion[i, :].sum()
        for i in range(confusion.shape[0])
        if confusion[i, :].sum() > 0
    ]) * 100.0

    if acc > best_mca:
        best_mca = acc
        best_mca_epoch = epoch
    if mean_acc > best_mpca:
        best_mpca = mean_acc
        best_mpca_epoch = epoch

    test_log = {
        'time': epoch_timer.timeit(),
        'loss': losses.avg,
        'group_acc': acc,
        'mean_acc': mean_acc,
        'best_mca': best_mca,
        'best_mpca': best_mpca,
        'best_mca_epoch': best_mca_epoch,
        'best_mpca_epoch': best_mpca_epoch,
    }

    return test_log


def initialize_model_parameters(model, args):
    initialization_scheme_save_path = args.save_path + '/model_initialization'

    module_dict = {}
    for module_name, module in model.named_modules():
        module_dict[module_name] = module

    for pname, param in model.named_parameters():
        append_text_to_file(initialization_scheme_save_path, "all_parameters.txt", pname + ": " + str(param.shape))

    general_initialization_method = lambda w: nn.init.trunc_normal_(w, mean=0.0, std=args.std_for_init)

    # ResNet18 is initialized by torchvision itself. If pretrained weights are not used,
    # torchvision's default initialization is retained for the backbone.
    # The modified grayscale conv1 and the classifier are initialized explicitly below.

    for module_name, module in module_dict.items():
        if module_name == 'module.backbone.conv1':
            general_initialization_method(module.weight)
            append_text_to_file(initialization_scheme_save_path, "backbone.txt", module_name + ".weight: " + general_initialization_method.__name__)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                append_text_to_file(initialization_scheme_save_path, "backbone.txt", module_name + ".bias: " + nn.init.zeros_.__name__)

    classifier_initialization_method = general_initialization_method
    for module_name, module in module_dict.items():
        if 'module.classifier' in module_name:
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                classifier_initialization_method(module.weight)
                append_text_to_file(initialization_scheme_save_path, "classifier.txt", module_name + ".weight: " + classifier_initialization_method.__name__)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    append_text_to_file(initialization_scheme_save_path, "classifier.txt", module_name + ".bias: " + nn.init.zeros_.__name__)


def accuracy(output, target):
    output = torch.argmax(output, dim=1)
    correct = torch.sum(torch.eq(target.int(), output.int())).float()
    return correct.item() / output.shape[0]


if __name__ == '__main__':
    main()