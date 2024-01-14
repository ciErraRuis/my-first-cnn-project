import torch
from torchsummary import summary
import torchvision
import os
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
from timm.loss import SoftTargetCrossEntropy
from timm.data.mixup import Mixup
import wandb
import matplotlib.pyplot as plt
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", DEVICE)
import sys
sys.path.append('../model')
from convneXt import ConvneXt

# Need to set DATA_DIR to your data's path
DATA_DIR = 'data/11-785-s23-hw2p2-classification'
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "dev")

# Configs
config = {
    'batch_size': 128,
    'lr': 2e-3,
    'epochs': 100,
    'drop_rate': [0, 0.1, 0.2, 0.4],
    'stage_depth': [3, 3, 27, 3],
    'stage_dim': [96, 192, 384, 768],
    'eps': 1e-3,
    'label_smoothing': 0.1,
    'optimizer': 'adamw',
    'scheduler': 'CosineAnnealing',
    'scheduler_warmup': 0,
    'scheduler_place': 'epoch',
    'min_lr': 1e-6,
    'weight_decay': 1e-2,
    'augment': 'randAug',
    'mixup_alpha': 0.8,
    'cutmix': 1,
    'mixup_prob': 1,
    'mixup_switch_prob': 0.5,
    'mixup_mode': 'batch'
}

def train(model, dataloader, optimizer, criterion, scaler, mixup_fn):
    model.train()
    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                     leave=False, position=0, desc='Train', ncols=5)
    
    num_correct = 0
    total_loss = 0
    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # Zero gradients
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # print("before mixing: image: ", images.shape, "labels: ", labels.shape)
        mixed_images, mixed_labels = mixup_fn(images, labels)
        # print("image shape: ", images.shape, "labels shape: ", labels.shape)
        with torch.cuda.amp.autocast():  # This implements mixed precision. Thats it!
            outputs = model(mixed_images)
            loss = criterion(outputs, mixed_labels)
        # print("outputs shape: ", outputs.shape)
        # Update no. of correct predictions & loss as we iterate
        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct /
                                  (config['batch_size']*(i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.07f}".format(float(optimizer.param_groups[0]['lr']))
        )

        # This is a replacement for loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)  # This is a replacement for optimizer.step()
        scaler.update()

        batch_bar.update()  # Update tqdm bar

    batch_bar.close()  # You need this to close the tqdm bar

    acc = 100 * num_correct / (config['batch_size'] * len(dataloader))
    total_loss = float(total_loss / len(dataloader))
    return acc, total_loss

def validate(model, dataloader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                     position=0, leave=False, desc='Val', ncols=5)
    num_correct = 0.0
    total_loss = 0.0

    for i, (images, labels) in enumerate(dataloader):
        # Move images to device
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs, labels)

        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct /
                                  (config['batch_size']*(i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct)

        batch_bar.update()

    batch_bar.close()
    acc = 100 * num_correct / (config['batch_size'] * len(dataloader))
    total_loss = float(total_loss / len(dataloader))
    return acc, total_loss


def main():
    # mean: tensor([0.5116, 0.4026, 0.3519]), std: tensor([0.3073, 0.2697, 0.2587])
    # Transforms using torchvision - Refer https://pytorch.org/vision/stable/transforms.html
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandAugment(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.5116, 0.4026, 0.3519), std=(0.3073, 0.2697, 0.2587))
    ])

    # Most torchvision transforms are done on PIL images. So you convert it into a tensor at the end with ToTensor()
    # But there are some transforms which are performed after ToTensor() : e.g - Normalization
    # Normalization Tip - Do not blindly use normalization that is not suitable for this dataset
    valid_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.5116, 0.4026, 0.3519], std=[0.3073, 0.2697, 0.2587])
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        TRAIN_DIR, transform=train_transforms)
    valid_dataset = torchvision.datasets.ImageFolder(
        VAL_DIR, transform=valid_transforms)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    print("Number of classes    : ", len(train_dataset.classes))
    print("No. of train images  : ", train_dataset.__len__())
    print("Shape of image       : ", train_dataset[0][0].shape)
    print("Batch size           : ", config['batch_size'])
    print("Train batches        : ", train_loader.__len__())
    print("Val batches          : ", valid_loader.__len__())

    model = ConvneXt().to(DEVICE)
    summary(model, (3, 224, 224))
    criterion = SoftTargetCrossEntropy()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config['patience'], min_lr=config['min_lr'], verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    scaler = torch.cuda.amp.GradScaler()

    gc.collect()  # These commands help you when you face CUDA OOM error
    torch.cuda.empty_cache()

    """# Wandb"""
    # API Key is in your wandb account, under settings (wandb.ai/settings)
    wandb.login(key="37061bfbfadfedb56c0e835f1ebf019bfe55febd")

    # Create your wandb run
    run = wandb.init(
        # Wandb creates random run names if you skip this field
        name="convnextS_lr2e-3_cosinelr_weightdecay1e-2_randaug_mixupofficial",
        reinit=True,  # Allows reinitalizing runs when you re-run this cell
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project="hw2p2",  # Project should be created in your wandb account
        config=config  # Wandb Config for your run
    )

    """# Experiments"""
    best_valacc = 0.0
    mixup_fn = Mixup(
        mixup_alpha=config['mixup_alpha'], cutmix_alpha=config['cutmix'], cutmix_minmax=None,
        prob=config['mixup_prob'], switch_prob=config['mixup_switch_prob'], mode='batch',
        label_smoothing=config['label_smoothing'], num_classes=7000)

    for epoch in range(config['epochs']):
        curr_lr = float(optimizer.param_groups[0]['lr'])
        train_acc, train_loss = train(
            model, train_loader, optimizer, criterion, scaler, mixup_fn)

        print("\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.07f}".format(
            epoch + 1,
            config['epochs'],
            train_acc,
            train_loss,
            curr_lr))

        val_acc, val_loss = validate(model, valid_loader)
        print("Val Acc {:.04f}%\t Val Loss {:.04f}".format(val_acc, val_loss))

        scheduler.step()

        wandb.log({"train_loss": train_loss, 'train_Acc': train_acc, 'validation_Acc': val_acc,
                   'validation_loss': val_loss, "learning_Rate": curr_lr})

        # #Save model in drive location if val_acc is better than best recorded val_acc
        if val_acc >= best_valacc:
            print("Saving model")
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_acc': val_acc,
                        'epoch': epoch}, './convneXt_lr=2e-3_cosinelr_weightdecay=1e-2_randaug_mixupofficial.pth')
            best_valacc = val_acc

    run.finish()
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch}, './convneXt_lr=2e-3_cosinelr_weightdecay=1e-2_randaug_mixupofficial.pth')

if __name__ == '__main__':
    main()
