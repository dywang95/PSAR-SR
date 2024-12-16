import argparse
import torch
import os
import copy
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from dataset.dataloader import get_dataloader, MyTestDataSet
from common import AverageMeter, calc_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='')
    parser.add_argument('--eval_file', type=str, default='')
    parser.add_argument('--evallabel_file', type=str, default='')
    parser.add_argument('--outputs_dir', type=str, default='./model/')
    parser.add_argument('--net', type=str, default='FSRCNN')
    parser.add_argument('--Channel', type=int, default=3)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--per_epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    if args.net == 'FSRCNN':
        from psar_fsrcnn import PSAR
    elif args.net == 'CARN':
        from psar_carn import PSAR
    elif args.net == 'SRResNet':
        from psar_srresnet import PSAR
    elif args.net == 'RCAN':
        from psar_rcan import PSAR

    model = PSAR().to(device)

    no_grad = [
        'tl',
        'tr'
    ]

    dataloader, data_size = get_dataloader(args)
    criterion1 = nn.L1Loss()
    criterion = nn.MSELoss()
    num_epochs = args.epoch
    per_epochs = args.per_epochs
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    train_dataloader = dataloader['train']
    def prepare(sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(device)
        return sample_batched

    eval_dataset = MyTestDataSet(args.eval_file, args.evallabel_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    best_weights1 = copy.deepcopy(model.state_dict())
    best_epoch1 = 0
    best_psnr1 = 0.0

    # D训练
    for epoch in range(num_epochs):
        pn = 0.0
        pn1 = 0.0
        scheduler.step()
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(data_size - data_size % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, num_epochs - 1))
            if epoch < per_epochs:
                mode = 'train1'
                for name, param in model.named_parameters():
                    if name in no_grad:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

                for i_batch, sample_batched in enumerate(train_dataloader):
                    sample_batched = prepare(sample_batched)
                    inputs = sample_batched['LR']
                    labels = sample_batched['HR']

                    preds, _, outbpt, outbp = model(inputs, mode)
                    loss = criterion(preds/255, labels/255) + criterion1(outbp/255, labels * outbpt/255)

                    epoch_losses.update(loss.item(), len(inputs))

                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

                    t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                    t.update(len(inputs))
            else:
                mode = 'train2'
                for name, param in model.named_parameters():
                    if name in no_grad:
                        param.requires_grad = True
                    else:
                        param.requires_grad = True
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

                for i_batch, sample_batched in enumerate(train_dataloader):
                    sample_batched = prepare(sample_batched)
                    inputs = sample_batched['LR']
                    labels = sample_batched['HR']

                    preds, time, outbpt, outbp = model(inputs, mode)

                    loss1 = criterion(preds/255, labels/255) + criterion1(outbp/255, labels * outbpt/255) + torch.mean(time[:, 0])

                    epoch_losses.update(loss1.item(), len(inputs))

                    optimizer.zero_grad()
                    loss1.backward()

                    optimizer.step()

                    t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                    t.update(len(inputs))
        torch.save(model, os.path.join(args.outputs_dir, 'model_epoch_{}.pth'.format(epoch)))
        model.eval()
        epoch_psnr = AverageMeter()
        epoch_psnr1 = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.permute(0, 3, 1, 2)
            labels = labels.permute(0, 3, 1, 2)

            inputs = inputs.float()
            labels = labels.float()

            inputs = inputs.to(device)
            labels = labels.to(device)

            if epoch < per_epochs:
                with torch.no_grad():
                    preds, _, _, _ = model(inputs, 'train1')
                    pn += calc_psnr(labels, preds)
            else:
                with torch.no_grad():
                    preds, _, _, _ = model(inputs, 'test')
                    pn1 += calc_psnr(labels, preds)

        if epoch < per_epochs:
            epoch_psnr.update(pn / 1.0, 1)
            print('lr:{:.7f}, per eval psnr: {:.4f}'.format(scheduler.get_lr()[0], epoch_psnr.avg))
        else:
            epoch_psnr1.update(pn1 / 1.0, 1)
            print('lr:{:.7f}, per eval psnr: {:.4f}'.format(scheduler.get_lr()[0], best_psnr), 'eval psnr: {:.4f}'.format(epoch_psnr1.avg))
