import torch
from torchsummary import summary
import torchvision
import os
import gc
from tqdm import tqdm
from PIL import Image
import sys
sys.path.append("../model")
from convneXt import ConvneXt
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE:", DEVICE)


DATA_DIR = '../../hw2p2/data/11-785-s23-hw2p2-classification'
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "dev")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "../../hw2p2/convnextS_lr2e-3_cosinelr_weightdecay1e-2_randaug_mixupofficial.pth"

config = {
    'batch_size': 128,
}

# input a trained model and the dataloader to be tested
# output the predicted result
def test(model, dataloader):
    # turn to evaluation mode
    model.eval()

    # load the batch bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                    position=0, leave=False, desc='Test')
    test_results = []

    for i, (images) in enumerate(dataloader):

        images = images.to(DEVICE)

        with torch.inference_mode():
            outputs = model(images)

        outputs = torch.argmax(outputs, axis=1).detach().cpu().numpy().tolist()
        test_results.extend(outputs)

        batch_bar.update()

    batch_bar.close()

    return test_results


class ClassificationTestDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, transforms=torchvision.transforms.ToTensor()):

        self.data_dir = data_dir
        self.transforms = transforms

        self.img_paths = list(map(lambda fname: os.path.join(self.data_dir, fname),
                              sorted(os.listdir(self.data_dir))))
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, ind):
        return self.transforms(Image.open(self.img_paths[ind]))
    

def main():

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.5116, 0.4026, 0.3519], std=[0.3073, 0.2697, 0.2587])
        ])
    
    # create test dataloader
    test_dataset = ClassificationTestDataset(TEST_DIR, transforms=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size = config['batch_size'],
                                              num_workers=2)

    model = ConvneXt().to(DEVICE)
    summary(model, (3, 224, 224))

    gc.collect()  # These commands help you when you face CUDA OOM error
    torch.cuda.empty_cache()

    trained_model = torch.load(MODEL_PATH)    
    model.load_state_dict(trained_model['model_state_dict'])
    test_results = test(model, test_loader)

    with open("classification_submission.csv", "w+") as f:
        f.write("id,label\n")
        for i in range(len(test_dataset)):
            f.write("{},{}\n".format(
                str(i).zfill(5) + ".jpg", test_results[i]))

if __name__ == '__main__':
    main()