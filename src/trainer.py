import numpy as np
import torch
import time
from torch import nn

from src.loss_functions import MeanAbsoluteError, GeneralizedCrossEntropy, NormalizedCrossEntropy, NormalizedFocalLoss, \
    FocalLoss
from src.net import MLP, CNN, weight_init

def f_alpha(epoch, r = 0.1):
    if r == 0.1 or r == 0.2:
        # Sparse setting
        alpha1 = np.linspace(0.0, 0.0, num=20)
        alpha2 = np.linspace(0.0, 1, num=20)
        alpha3 = np.linspace(1, 2, num=50)
        alpha4 = np.linspace(2, 5, num=50)
        alpha5 = np.linspace(5, 10, num=100)
        alpha6 = np.linspace(10, 20, num=100)
    else:
        # Uniform/Random noise setting
        alpha1 = np.linspace(0.0, 0.0, num=20)
        alpha2 = np.linspace(0.0, 0.1, num=20)
        alpha3 = np.linspace(1, 2, num=50)
        alpha4 = np.linspace(2, 2.5, num=50)
        alpha5 = np.linspace(2.5, 3.3, num=100)
        alpha6 = np.linspace(3.3, 5, num=100)

    alpha = np.concatenate((alpha1, alpha2, alpha3, alpha4, alpha5, alpha6), axis=0)
    return alpha[epoch]

def trainer_singletask_blind(config,num_epochs,num_sample_pts, dataloaders, dataset_sizes,model_type, classes, device, loss_type = "CCE", dropout = False):
    # Build the model
    if model_type == "mlp":
        model = MLP(config,loss_type, num_sample_pts, classes,dropout).to(device)
    elif model_type == "cnn":
        model = CNN(config,loss_type, num_sample_pts, classes,dropout).to(device)
    weight_init(model, config['kernel_initializer'])
    print(model)
    # Creates the optimizer
    lr = config["lr"]
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    # This is the trainning Loop


    if "PEER_LOSS" in loss_type:
        print("USING PEER_LOSS")
        peer_loader = dataloaders["train_peer_loss"]
    # if "MSE" in loss_type:
    #     criterion = nn.MSELoss()
    if "CCE" in loss_type:
        criterion = nn.CrossEntropyLoss()
    if "MAE" == loss_type:
        criterion = MeanAbsoluteError(num_classes= classes)
    elif "GCE" == loss_type:
        criterion = GeneralizedCrossEntropy(num_classes= classes, q=0.5)
    elif "NCE" == loss_type:
        criterion = NormalizedCrossEntropy(num_classes= classes)
    elif "FL" == loss_type:
        criterion = FocalLoss(num_classes=classes,gamma=0.5)
    elif "NFL" == loss_type:
        criterion = NormalizedFocalLoss(num_classes=classes,gamma=0.5)

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=5*lr,
                                                  step_size_up=10, cycle_momentum=False)
    # start = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch +1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  # ,
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            tk0 = dataloaders[phase]  # tqdm(dataloader[phase])

            for (traces, labels) in tk0: #(traces, labels)
                inputs = traces.to(device)
                labels = labels.to(device)
                labels = labels.long()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if "CCE" in loss_type :
                        _, preds = torch.max(outputs, dim=1)

                    if "PEER_LOSS" in loss_type:
                        peer_iter = iter(peer_loader)
                        input1 = next(peer_iter)[0]
                        peer_output1 = model(input1.to(device))
                        peer_target2 = next(peer_iter)[1]
                        #convert peer_target2
                        peer_target2 = torch.Tensor(peer_target2.float())
                        peer_target2 = torch.autograd.Variable(peer_target2.to(device))
                        # Peer Loss with Cross-Entropy loss: L(f(x), y) - L(f(x1), y2)
                        loss = criterion(outputs, labels.long()) - f_alpha(epoch) * (criterion(peer_output1, peer_target2.long()))
                        # loss = criterion(outputs, labels.long()) - criterion(peer_output1, peer_target2.long())
                    else:
                        loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # print("preds inside: ", preds.shape)
                # print("labels inside: ", labels.data.shape)
                # print("preds == labels.data: ", (preds == labels.data).all())
                            # if phase == 'train':
            scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            inputs.detach()
            labels.detach()
            # Here we calculate the GE, NTGE and the accuracy over the X_attack traces.
            print('{} Epoch Loss: {:.4f}'.format(phase, epoch_loss))
            if epoch_loss < 1 and phase == 'val': #early stopping.
                model.eval()
                return model
        model.eval()
        # model.to("cpu")
        # if (epoch + 1) % 10 == 0 and epoch != 0:

    print("Finished Training Model")
    return model