import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import random
import shap
import pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

# import internal libs
from data import prepare_dataset
from model import prepare_model
from utils import get_datetime, set_logger, get_logger, set_seed, set_device, \
    log_settings, save_current_src

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
        self.noisy_idxes = random.Random(seed).sample(range(len(self.dataset)), noise_size)
        self.pos_class = pos_class
        self.neg_class = neg_class
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if idx in self.noisy_idxes:
            x, y = self.dataset[idx]
            if y == self.pos_class:
                y = self.neg_class
            elif y == self.neg_class:
                y = self.pos_class
            else:
                raise ValueError("Invalid label")
            return x, y
        else:
            return self.dataset[idx]


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


def unnormalize(images):
    # images = images / 2 + 0.5
    for t, m, s in zip(images, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)):
        t.mul_(s).add_(m)
    return images


def cal_shapley_val(save_path: str,
                    device: torch.device,
                    model: nn.Module,
                    dataset: FlipLabelDataset,
                    is_noisy: bool,
                    test_size: int = 10,
                    background_size: int = 100,
                    seed: int = 0):
    """calculate the shapley value

    Args:
        device (torch.device): put the model on the device.
        model (nn.Module): the model.
        dataset (Dataset): the dataset used to choose the samples.
        is_noisy (bool): if test on the noisy samples.
        test_size (int): the test size.
        background_size (int): the background size.
        seed (int): the seed.

    Return:
        None
    """
    logger = get_logger(f"{__name__}.cal_shapley_val")
    os.makedirs(os.path.join(save_path, "pkls"), exist_ok=True)

    # prepare the background, the background is 100 clean samples.
    background_idxes = random.Random(seed).sample(
        [i for i in range(len(dataset)) if i not in dataset.noisy_idxes], background_size
    )
    background = torch.stack([dataset[i][0] for i in background_idxes]).to(device)

    # prepare the explainer
    explainer = shap.DeepExplainer(model, background)

    # prepare the test images
    if is_noisy:
        test_idxes = random.Random(seed).sample(dataset.noisy_idxes, test_size)
    else:
        test_idxes = random.Random(seed).sample(
            [i for i in range(len(dataset)) if i not in dataset.noisy_idxes and i not in background_idxes], test_size
        )
    for idx in tqdm(test_idxes):
        x, _ = dataset[idx]
        x = x.unsqueeze(0).to(device)

        # calculate the shapley value
        shap_values, indexes = explainer.shap_values(x, ranked_outputs=2)
        logger.debug(f"idx: {idx}, shap_values: {shap_values[0].shape}")

        image_numpy = unnormalize(x[0].cpu()).permute(1, 2, 0).unsqueeze(0).numpy()
        shap_numpy = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
        indexes = [i.cpu().numpy() for i in indexes]

        # save the shapley value
        result_dict = {
            "shap_values": shap_numpy,
            "image": image_numpy,
            "indexes": indexes,
        }
        with open(os.path.join(save_path, "pkls", f"{idx}.pkl"), "wb") as f:
            pickle.dump(result_dict, f)

        shap.image_plot(shap_numpy, image_numpy, indexes, show=False)
        plt.savefig(os.path.join(save_path, f"{idx}.png"))
        plt.close()


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
    parser.add_argument("--dataset", default="cifar10", type=str,
                        help='the dataset name.')
    parser.add_argument("--pos_class", default=0, type=int,
                        help='the positive class index.')
    parser.add_argument("--neg_class", default=1, type=int,
                        help='the negative class index.')
    parser.add_argument("--noise_rate", default=0.2, type=float,
                        help='the noise rate.')
    parser.add_argument("--model", default="vgg11", type=str,
                        help='the model name.')
    parser.add_argument("--background_size", default=100, type=int,
                        help='the sample size for the background.')
    parser.add_argument("--test_size", default=100, type=int,
                        help='the sample size for the test images.')
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
                         f"noise_rate{args.noise_rate}",
                         f"{args.model}",])
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
    trainset, _ = prepare_dataset(args.dataset, randomize=False, normalize=True)
    
    # filter the dataset with pos_class and neg_class
    trainset_binary = filter_and_balance_dataset(trainset, args.pos_class, args.neg_class)
    
    # flip the label of the dataset
    trainset_binary_flip = FlipLabelDataset(
        trainset_binary, args.noise_rate, args.pos_class, args.neg_class
    )

    # prepare the model
    logger.info("#########preparing model....")
    model = prepare_model(args.model, args.dataset, args.seed)    
    model.load_state_dict(torch.load(args.resume_path))
    logger.info(f"load the pretrained model from {args.resume_path}")
    
    model.to(args.device)
    model.eval()

    # calculate the shapley value for the model on selected samples.
    logger.info("#########calculating the shapley on noisy samples....")
    cal_shapley_val(save_path=os.path.join(args.save_path, "shapley/noisy_samples"), 
                    device=args.device, 
                    model=model, 
                    dataset=trainset_binary_flip,
                    is_noisy=True,
                    test_size=args.test_size,
                    background_size=args.background_size,)
    
    logger.info("#########calculating the shapley on clean samples....")
    cal_shapley_val(save_path=os.path.join(args.save_path, "shapley/clean_samples"), 
                    device=args.device, 
                    model=model, 
                    dataset=trainset_binary_flip,
                    is_noisy=False,
                    test_size=args.test_size,
                    background_size=args.background_size,)


if __name__ == "__main__":
    main()