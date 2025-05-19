import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import pandas as pd

# import from internal libs
from utils.plot import plot_multiple_curves

def plot_train(data: pd.DataFrame,
               save_path: str):
    """plot the train metrics.

    Args:
        data (pd.DataFrame): the data.
        save_path (str): the save path.
    """
    print("plot the train metrics.")\
    
    try:
        x = data["iter"].values
        xlabel = "iterations"
    except:
        x = data["epoch"].values
        xlabel = "epochs"

    # get the data
    losses = {
        "train_loss": (x, data["train_loss"].values),
        "test_loss": (x, data["test_loss"].values)
    }
    accs = {
        "train_acc": (x, data["train_acc"].values),
        "train_acc_clean": (x, data["train_acc_clean"].values),
        "train_acc_flip": (x, data["train_acc_flip"].values),
        "test_acc": (x, data["test_acc"].values)
    }

    # plot the loss
    plot_multiple_curves(losses, save_path, f"loss.png", 
        xlabel=xlabel)
    
    # plot the acc
    plot_multiple_curves(accs, save_path, f"acc.png",
        xlabel=xlabel, ylim=[0, 1])
    
    print("plot the train metrics done.")


def main():
    parser = argparse.ArgumentParser(
        description="plot the metrics of the model.")
    parser.add_argument("--data_path", "-d", default=None, type=str,
                        help="the path of the data.")
    
    args = parser.parse_args()

    # load the data
    assert args.data_path.endswith(".csv"), \
        "the data must be csv file."
    data = pd.read_csv(args.data_path)
    
    # get the filename
    filename = os.path.basename(args.data_path)
    # get the save path
    save_path = os.path.dirname(args.data_path)

    if "train" in filename:
        plot_train(data, save_path)
    else:
        raise ValueError("unknown filename.")


if __name__ == "__main__":
    main()