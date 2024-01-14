import numpy as np
import torch
import torchvision
from PIL import Image
import glob
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import gc
import os
import wandb
import sys
sys.path.append('../model')
from convneXt import ConvneXt
from convneXt_with_Arcface import ConvneXt_with_Arcface

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "convneXt_lr=2e-3_cosinelr_weightdecay=1e-2_randaug_mixupofficial.pth"
DATA_DIR = "data/11-785-s23-hw2p2-classification"
VAL_DIR = "data/11-785-s23-hw2p2-verification"
TRAIN_DIR = os.path.join(DATA_DIR, "train")

config = {
    'batch_size': 128,
    'Arcface_margin': 0.4,
    'Arcface_scale': 64,
    'epochs': 20,
    'lr': 1e-3,
    'weight_decay': 1e-2,
    'min_lr': 1e-5,
    'margin': 0.45
}

def train_verification(model, dataloader, optimizer, criterion, scaler):
    model.train()

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                     leave=False, position=0, desc='Train', ncols=5)

    num_correct = 0
    total_loss = 0

    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # Zero gradients
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # print("image shape: ", images.shape, "labels shape: ", labels.shape)
        with torch.cuda.amp.autocast():  # This implements mixed precision. Thats it!
            outputs = model(images, labels)
            loss = criterion(outputs, labels)

        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct /
                                  (config['batch_size']*(i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            numcorrect=num_correct,
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

def eval_verification(unknown_images, known_images, known_paths, model, similarity, batch_size=config['batch_size'], mode='val', threshold=0.33):
    model.eval()
    unknown_feats, known_feats = [], []
    batch_bar = tqdm(total=len(unknown_images)//batch_size,
                     dynamic_ncols=True, position=0, leave=False, desc=mode)

    # We load the images as batches for memory optimization and avoiding CUDA OOM errors
    for i in range(0, unknown_images.shape[0], batch_size):
        # Slice a given portion upto batch_size
        unknown_batch = unknown_images[i:i+batch_size]
        with torch.no_grad():
            unknown_feat = model(unknown_batch.float().to(
                DEVICE))  # Get features from model
        unknown_feats.append(unknown_feat)
        batch_bar.update()

    batch_bar.close()

    batch_bar = tqdm(total=len(known_images)//batch_size,
                     dynamic_ncols=True, position=0, leave=False, desc=mode)
    for i in range(0, known_images.shape[0], batch_size):
        known_batch = known_images[i:i+batch_size]
        with torch.no_grad():
            known_feat = model(known_batch.float().to(DEVICE))
        known_feats.append(known_feat)
        batch_bar.update()

    batch_bar.close()

    # Concatenate all the batches
    unknown_feats = torch.cat(unknown_feats, dim=0)
    known_feats = torch.cat(known_feats, dim=0)
    similarity_values = torch.stack(
        [similarity(unknown_feats, known_feature) for known_feature in known_feats])

    max_similarity_values, predictions = similarity_values.max(0)
    max_similarity_values, predictions = max_similarity_values.cpu(
    ).numpy(), predictions.cpu().numpy()
    
    predictions = predictions.reshape(-1)
    # print(predictions)

    # Note that in unknown identities, there are identities without correspondence in known identities.
    # Therefore, these identities should be not similar to all the known identities, i.e. max similarity will be below a certain
    # threshold compared with those identities with correspondence.

    # In early submission, you can ignore identities without correspondence, simply taking identity with max similarity value
    # pred_id_strings = [known_paths[i] for i in predictions] # Map argmax indices to identity strings

    # After early submission, remove the previous line and uncomment the following code

    NO_CORRESPONDENCE_LABEL = 'n000000'
    pred_id_strings = []
    for idx, prediction in enumerate(predictions):
        # why < ? Thank about what is your similarity metric
        if max_similarity_values[idx] < threshold:
            pred_id_strings.append(NO_CORRESPONDENCE_LABEL)
        else:
            pred_id_strings.append(known_paths[prediction])

    if mode == 'val':
        CSV_PATH = os.path.join(VAL_DIR, "verification_dev.csv")
        true_ids = pd.read_csv(CSV_PATH)['label'].tolist()
        accuracy = accuracy_score(pred_id_strings, true_ids)
        return accuracy * 100

    return pred_id_strings

 #------------------------------------------------------------------------------
def main():
    # get the evaluation data
    known_regex = os.path.join(VAL_DIR, "known/*/*")
    known_paths = [i.split("/")[-2] for i in sorted(glob.glob(known_regex))]
    known_images = [Image.open(p) for p in tqdm(sorted(glob.glob(known_regex)))]

    unknown_dev_regex = os.path.join(VAL_DIR, "unknown_dev/*")
    unknown_test_regex = os.path.join(VAL_DIR, "unknown_test/*")
    unknown_dev_images = [Image.open(p) for p in tqdm(sorted(glob.glob(unknown_dev_regex)))]
    unknown_test_images = [Image.open(p) for p in tqdm(sorted(glob.glob(unknown_test_regex)))]

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5116, 0.4026, 0.3519], std=[0.3073, 0.2697, 0.2587])])
    unknown_dev_images = torch.stack([transforms(x) for x in unknown_dev_images])
    unknown_test_images = torch.stack([transforms(x) for x in unknown_test_images])
    known_images  = torch.stack([transforms(y) for y in known_images])

    similarity_metric = torch.nn.CosineSimilarity(dim= 1, eps= 1e-6)

    # model
    cls_model = ConvneXt().to(DEVICE)
    saved = torch.load(MODEL_PATH)
    cls_model.load_state_dict(saved['model_state_dict'])
    model = ConvneXt_with_Arcface(cls_model, cls_model.embedding, cls_model.class_num,
                                  margin=config['margin']).to(DEVICE)
    # criterion
    criterion = torch.nn.CrossEntropyLoss()
    # scaler
    scaler = torch.cuda.amp.GradScaler()
    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config['epochs'],
                                                           eta_min=config['min_lr'])
    # train dataset and dataloader
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandAugment(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5116, 0.4026, 0.3519], std=[0.3073, 0.2697, 0.2587])])
    
    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config['batch_size'],
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=4)
    
    gc.collect()  # These commands help you when you face CUDA OOM error
    torch.cuda.empty_cache()

    """# Wandb"""
    # API Key is in your wandb account, under settings (wandb.ai/settings)
    wandb.login(key="37061bfbfadfedb56c0e835f1ebf019bfe55febd")

    # Create your wandb run
    run = wandb.init(
        # Wandb creates random run names if you skip this field
        name="ConvnextWithArcface_verification_lr=1e-3_weightdecay=1e-2_margin=0.4",
        reinit=True,  # Allows reinitalizing runs when you re-run this cell
        # id =, ### Insert specific run id here if you want to resume a previous run
        # resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
        project="hw2p2",  # Project should be created in your wandb account
        config=config  # Wandb Config for your run
    )

    """experiment"""
    best_acc = 0

    for epoch in range(config['epochs']):

        curr_lr = float(optimizer.param_groups[0]['lr'])

        train_acc, train_loss = train_verification(model, train_dataloader,optimizer,
                                                    criterion, scaler)
        print("\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.08f}".format(
            epoch + 1,
            config['epochs'],
            train_acc,
            train_loss,
            curr_lr))

        val_acc = eval_verification(unknown_dev_images, known_images, known_paths, cls_model,
                                    similarity_metric, config['batch_size'], mode='val')
        print("Val Acc: ", val_acc)

        scheduler.step()

        wandb.log({"train_loss": train_loss, 'train_Acc': train_acc, 'validation_Acc': val_acc,
                   "learning_Rate": curr_lr})

        print("Saving model")

        if val_acc >= best_acc:
            torch.save({'model_state_dict': model.state_dict(),
                    'classification_model_state_dict': cls_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'epoch': epoch}, './ArcFace_verification_margin=0.45.pth')
            print("better model saved!")
            best_acc = val_acc

    run.finish()
  #-----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
