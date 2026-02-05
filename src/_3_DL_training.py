


from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
import torch

from src.dataloader_blind import Blind_Single_Dataset, ToTensor_trace_blind
from src.trainer import trainer_singletask_blind


def train_pipeline_singletask_dl(config,epochs, X, Y, classes,device, model_type= "cnn", loss_type = "CCE", dropout = False ):
    num_epochs = epochs
    #further split to train and validation
    X_train, X_val, Y_train, Y_val= train_test_split(X, Y, test_size=0.1,random_state=0)
    print("Spliting data to train and validation")
    print("Y_train.shape:",Y_train.shape)
    print("Y_val.shape:",Y_val.shape)
    print("Load into Blind_Dataset:")
    dataloadertrain = Blind_Single_Dataset(X_train, Y_train, transform=transforms.Compose([ToTensor_trace_blind()]))
    dataloaderval = Blind_Single_Dataset(X_val, Y_val, transform=transforms.Compose([ToTensor_trace_blind()]))


    batch_size = config["batch_size"]
    dataloaders = {"train": torch.utils.data.DataLoader(dataloadertrain, batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=0),
                   "train_peer_loss": torch.utils.data.DataLoader(dataloadertrain, batch_size=batch_size,
                                                                  shuffle=True,
                                                                  num_workers=0),
                   "val": torch.utils.data.DataLoader(dataloaderval, batch_size=batch_size,
                                                      shuffle=True, num_workers=0)
                   }
    dataset_sizes = {"train": len(dataloadertrain), "val": len(dataloaderval)}

    num_sample_pts = X.shape[-1]
    model = trainer_singletask_blind(config, num_epochs, num_sample_pts, dataloaders, dataset_sizes, model_type, classes, device,loss_type=loss_type, dropout = dropout)
    return model

