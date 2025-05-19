import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

# import internal libs
from data import prepare_dataset
from model import prepare_model
from utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src
from utils.avgmeter import MetricTracker
from utils.tools import evaluation

class FlipLabelDataset(Dataset):
    """Flip the label of the dataset with noise rate"""

    def __init__(self, 
                 dataset: Dataset, 
                 noise_rate: float,
                 pos_class: int = 0,
                 neg_class: int = 1,
                 seed: int = 0):
        """init the dataset

        Args:
            dataset: the dataset
            noise_rate: the noise rate
            pos_class: the positive class
            neg_class: the negative class
            seed: the seed
        """
        self.dataset = dataset
        noise_size = int(noise_rate * len(self.dataset))
        self.noise_idxes = random.Random(seed).sample(range(len(self.dataset)), noise_size)
        self.pos_class = pos_class
        self.neg_class = neg_class
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if idx in self.noise_idxes:
            y = self.pos_class if y == self.neg_class else self.neg_class
            isFlip = True
        else:
            isFlip = False
        return x, y, isFlip


def train(save_path: str,
          device: torch.device,
          model: nn.Module,
          trainset: Dataset,
          testset: Dataset,
          optim: str,
          epochs: int,
          lr: float,
          batch_size: int,
          weight_decay: float,
          momentum: float,
          step_size: list,
          steps_saving: int,
          seed: int) -> None:
    """train the model

    Args:
        save_path: the path to save results
        device: GPU or CPU
        model: the model to train
        trainset: the train dataset
        testset: the test dataset
        optim: the optimizer
        epochs: the epochs
        lr: the learning rate
        batch_size: the batch size
        weight_decay: the weight decay
        momentum: the momentum
        step_size: the StepLR's step size
        steps_saving: the steps to save the model
        seed: the seed
    """
    logger = get_logger(__name__)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ## set up the basic component for training
    # put the model to GPU or CPU
    model = model.to(device)
    # set the optimizer
    if optim == "SGD":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                                    lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optim == "Adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                     lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"optimizer should be SGD or Adam but got {optim}")
    
    # set the scheduler
    if len(step_size) == 1:
        logger.info(f"StepLR is used with step_size {step_size[0]}")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size[0], gamma=0.1, last_epoch=-1)
    else:
        logger.info(f"MultiStepLR is used with step_size {step_size}")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    
    # set the loss function
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    
    ## set up the data part
    # set the testset loader
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # create the seeds for the first phase
    seeds = random.Random(seed).sample(range(10000000), k=epochs)

    # initialize the tracker
    tracker = MetricTracker()

    for epoch in range(1, epochs+1):
        logger.info(f"######Epoch - {epoch}")
        set_seed(seeds[epoch-1])   
        # create the batches for train
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        # train the model
        model.train()
        for batch_idx, (inputs, labels, flips) in enumerate(tqdm(trainloader)):
            # set the inputs to device
            inputs, labels = inputs.to(device), labels.to(device)
            # set the outputs
            outputs = model(inputs)
            # set the loss
            losses = loss_fn(outputs, labels)
            loss = torch.mean(losses)
            # set zero grad
            optimizer.zero_grad()
            # set the loss
            loss.backward()
            # set the optimizer
            optimizer.step()
            # set the predictions
            preds = outputs.max(1)[1]
            # set the loss and accuracy
            tracker.update({
                "train_loss": loss.item(),
                "train_acc": (preds == labels).float().mean().item()
            }, n = inputs.size(0))
            if sum(flips) > 0:
                tracker.update({
                    "train_acc_flip": (preds[flips] == labels[flips]).float().mean().item()
                }, n = sum(flips).item())
            tracker.update({
                "train_acc_clean": (preds[~flips] == labels[~flips]).float().mean().item()
            }, n = sum(~flips).item())

        # print the train loss and accuracy
        logger.info(tracker)

        # eval on the testset
        test_loss, test_acc, _ = evaluation(device, model, testloader)
        # print the test loss and accuracy
        logger.info(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")

        # update the tracker
        tracker.track({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch": epoch,
        })

        # update the scheduler
        scheduler.step()

        # save the results
        if epoch % 20 == 1 or epoch == epochs:
            torch.save(model.state_dict(), 
                        os.path.join(save_path, f"model_epoch{epoch}.pt"))
            tracker.save_to_csv(os.path.join(save_path, f"train.csv"))
            

def filter_and_balance_dataset(dataset: Dataset,
                               pos_class: int, 
                               neg_class: int,) -> Dataset:
    """filter and balance the dataset with pos_class and neg_class
    
    Args:
        dataset: the dataset
        pos_class: the positive class
        neg_class: the negative class
    
    Returns:
        the filtered and balanced dataset
    """
    logger = get_logger(f"{__name__}.filter_and_balance_dataset")
    
    # filter and balance the dataset
    pos_idx = [i for i, (_, label) in enumerate(dataset) if label == pos_class]
    neg_idx = [i for i, (_, label) in enumerate(dataset) if label == neg_class]
    min_len = min(len(pos_idx), len(neg_idx))
    logger.info(f"pos_class: {len(pos_idx)}, neg_class: {len(neg_idx)}, min_len: {min_len}")
    
    idx = pos_idx[:min_len] + neg_idx[:min_len]
    return Subset(dataset, idx)


def add_args() -> argparse.Namespace:
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="simple verification")
    ## the basic setting of exp
    parser.add_argument('--device', default=0, type=int,
                        help="set the device.")
    parser.add_argument("--seed", default=0, type=int,
                        help="set the seed.")
    parser.add_argument("--save_root", default="../outs/tmp/", type=str,
                        help='the path of saving results.')
    parser.add_argument("--resume_path", default=None, type=str,
                        help='the path of pretrained model.')
    parser.add_argument("--dataset", default="mnist", type=str,
                        help='the dataset name.')
    parser.add_argument("--pos_class", default=0, type=int,
                        help='the positive class index.')
    parser.add_argument("--neg_class", default=1, type=int,
                        help='the negative class index.')
    parser.add_argument("--noise_rate", default=0.2, type=float,
                        help='the noise rate.')
    parser.add_argument("--model", default="LeNet", type=str,
                        help='the model name.')
    parser.add_argument("--optimizer", default="Adam", type=str,
                        help='the optimizer name.')
    parser.add_argument('--epochs', default=60, type=int,
                        help="set iteration number")
    parser.add_argument("--lr", default=12e-4, type=float,
                        help="set the learning rate.")
    parser.add_argument("--bs", default=60, type=int,
                        help="set the batch size")
    parser.add_argument("--wd", default=0.0, type=float,
                        help="set the weight decay")
    parser.add_argument("--momentum", default=0.0, type=float,
                        help="set the momentum rate")
    parser.add_argument("--step_size", default=[60], type=int, nargs="+",
                        help="set the StepLR stepsize")
    parser.add_argument("--steps_saving", default=60, type=int,
                        help="set the steps to save the model")
    # set if using debug mod
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    exp_name = "-".join([get_datetime(),
                         f"seed{args.seed}",
                         f"{args.dataset}",
                         f"{args.pos_class}_{args.neg_class}",
                         f"{args.model}",
                         f"{args.optimizer}",
                         f"epochs{args.epochs}",
                         f"lr{args.lr}",
                         f"bs{args.bs}",])
    args.save_path = os.path.join(args.save_root, exp_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args


def main():
    # get the args.
    args = add_args()
    # set the logger
    set_logger(args.save_path)
    # get the logger
    logger = get_logger(__name__, args.verbose)
    # set the seed
    set_seed(args.seed)
    # set the device
    args.device = set_device(args.device)
    # save the current src
    save_current_src(save_path = args.save_path)

    # show the args.
    logger.info("#########parameters settings....")
    log_settings(args)

    # prepare the dataset
    logger.info("#########preparing dataset....")
    trainset, testset = prepare_dataset(args.dataset, randomize=False, normalize=True)
    
    # filter the dataset with pos_class and neg_class
    trainset_binary = filter_and_balance_dataset(trainset, args.pos_class, args.neg_class)
    testset_binary = filter_and_balance_dataset(testset, args.pos_class, args.neg_class)
    
    # flip the label of the dataset
    trainset_binary_flip = FlipLabelDataset(
        trainset_binary, args.noise_rate, args.pos_class, args.neg_class
    )

    # prepare the model
    logger.info("#########preparing model....")
    model = prepare_model(args.model, args.dataset, args.seed)
    if args.resume_path:
        logger.info(f"load the pretrained model from {args.resume_path}")
        model.load_state_dict(torch.load(args.resume_path))
    logger.info(model)

    # train the model
    logger.info("#########training model....")
    train(save_path = os.path.join(args.save_path, "exp"),
          device = args.device,
          model = model,
          trainset = trainset_binary_flip,
          testset = testset_binary,
          optim = args.optimizer,
          epochs = args.epochs,
          lr = args.lr,
          batch_size = args.bs,
          weight_decay = args.wd,
          momentum = args.momentum,
          step_size = args.step_size,
          steps_saving = args.steps_saving,
          seed = args.seed)


if __name__ == "__main__":
    main()
