import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Physio
from dataset import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device ')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)

parser.add_argument("--modelfolder", type=str, default="")

parser.add_argument("--ratio",type=float,default=0.7)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--dataset",type=str,default="SMD")
args = parser.parse_args()

AUC_ROC = 0
BEST_Fc1 = 0
BEST_AUC_F1_K = 0


train_data_path_list = []
test_data_path_list = []
label_data_path_list = []

data_set_number = [args.dataset]
for data_set_id in data_set_number:
    file = f"{data_set_id}_train.pkl"
    train_data_path_list.append("/content/drive/MyDrive/data/dataset/" + args.dataset + '/' + file)
    test_data_path_list.append("/content/drive/MyDrive/data/dataset/" + args.dataset + '/' + file.replace("_train.pkl", "_test.pkl"))
    label_data_path_list.append("/content/drive/MyDrive/data/dataset/" + args.dataset + '/' + file.replace("_train.pkl", "_test_label.pkl"))

diffusion_step_list = [50]

conditional_list = [True]

split_list = [10]



try:
    os.mkdir("train_result")
except:
    pass


for training_epoch in range(0,6):
    print(f"begin to train for training_epoch {training_epoch} ...")
    try:
        os.mkdir(f"train_result/save{training_epoch}")
    except:
        pass
    for diffusion_step in diffusion_step_list:
        for conditional in conditional_list:
            for split in split_list:

                for i, train_data_path in enumerate(train_data_path_list):
                    path = "config/" + args.config
                    with open(path, "r") as f:
                        config = yaml.safe_load(f)

                    config["model"]["is_conditional"] = conditional

                    config["diffusion"]["num_steps"] = diffusion_step
                    # print(json.dumps(config, indent=4))

                    foldername = f"./train_result/save{training_epoch}/" + f"{train_data_path.replace('_train.pkl', '').replace('/content/drive/MyDrive/data/dataset/PSM/', '')}" + "_conditional:" + str(
                        conditional) + "_split:" + str(
                        split) + "_diffusion_step:" + str(diffusion_step) + "/"
                    print('model folder:', foldername)
                    try:
                        os.makedirs(foldername)
                    except:
                        pass
                    with open(foldername + "config.json", "w") as f:
                        json.dump(config, f, indent=4)

                    test_data_path = test_data_path_list[i]
                    label_data_path = label_data_path_list[i]

                    train_loader, valid_loader, test_loader1, test_loader2 = get_dataloader(
                        train_data_path,
                        test_data_path,
                        label_data_path,
                        batch_size=12,
                        split=split
                    )

                    if args.dataset == 'PSM':
                      target_dim = 25
                    elif args.dataset == 'SMD2':
                      target_dim = 38
                    elif args.dataset == 'SWAT':
                      target_dim = 51
                    elif args.dataset == 'SMAP2':
                      target_dim = 25
                    elif args.dataset == 'MSL':
                      target_dim = 55
                    else:
                      target_dim = 30
                    

                    model = CSDI_Physio(config, args.device,target_dim,ratio = args.ratio).to(args.device)

                    AUC_ROC, BEST_Fc1, BEST_AUC_F1_K = train(
                        model,
                        config["train"],
                        train_loader,
                        valid_loader=valid_loader,
                        foldername=foldername,
                        test_loader1=test_loader1,
                        test_loader2=test_loader2
                    )
                    print("AUC_ROC : {:0.4f}, BEST_Fc1 : {:0.4f}, BEST_AUC_F1_K : {:0.4f}".format(AUC_ROC, BEST_Fc1, BEST_AUC_F1_K))

