"""
Author: Matěj Hejda [scripts forked from Jiaqi Gu (jqgu@utexas.edu)]
"""

#import datetime
import os
import sys
import pickle
from pathlib import Path
import numpy as np
#from datetime import datetime

import torch

##### Import local packages
root_folder = str(Path(os.getcwd()).parents[0])
sys.path.append(root_folder)

def save_model(
    model,
    directory,
    uid,
    train_acc_meter,
    train_loss_meter,
    validation_acc_meter,
    validation_loss_meter,
    test_acc_meter,
    logger,
    results_dict_logger,
    save_full_model=False,  # If True, saves the model object alongside the state_dict. If False, saves only the state_dict.
    suffix="",
):  # For distinguishing repeated runs, for example
    if not isinstance(suffix, str):
        suffix = str(suffix)

    if suffix != "":
        presuffix = "-"
    else:
        presuffix = ""

    """Save PyTorch model in path"""
    if len(uid) != 0:
        directory = os.path.join(directory, uid)
        if not os.path.exists(directory):
            os.mkdir(directory)

    path_cp = os.path.join(directory, "checkpoints")

    if not os.path.exists(path_cp):
        os.mkdir(path_cp)

    fname = '' # Previously: datetime.today().strftime("%y%m%d-%H%M")

    ######
    new_path = os.path.join(
        path_cp,
        fname
        #+ f"__{uid}{suffix}_{test_acc_meter.name}={test_acc_meter.avg*100:.2f}"
        + f"{suffix}_{test_acc_meter.name}={test_acc_meter.avg*100:.2f}"
        + ".pt",
    )

    
    if save_full_model:
        saved_dict = {}
        saved_dict.update({"model": model, "state_dict": model.state_dict()})
        torch.save(saved_dict, new_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Saved torch model + its state dictionary into {path_cp}")
    else:
        torch.save(model.state_dict(), new_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Saved torch model's state dictionary into {path_cp}")

    # JSON-save some stats: training
    results_dict_logger.save(
        uid + presuffix + suffix,
        train_acc_meter.name + "_epochs",
        list(train_acc_meter.get_history_epochs() * 100),
    )
    results_dict_logger.save(
        uid + presuffix + suffix,
        train_loss_meter.name + "_epochs",
        list(train_loss_meter.get_history_epochs()),
    )
    # JSON-save some stats: validation
    results_dict_logger.save(
        uid + presuffix + suffix,
        validation_acc_meter.name + "_epochs",
        list(validation_acc_meter.get_history_epochs() * 100),
    )
    results_dict_logger.save(
        uid + presuffix + suffix,
        validation_loss_meter.name + "_epochs",
        list(validation_loss_meter.get_history_epochs()),
    )
    # JSON-save some stats: testing
    results_dict_logger.save(
        uid + presuffix + suffix, test_acc_meter.name, test_acc_meter.avg * 100
    )
    try:
        ## Attempt to save the Confusion Matrix
        cm = test_acc_meter.get_confusion_matrix()
        cm_np = cm.cpu().numpy().astype(dtype=np.int16)
        cm_np_list = cm_np.tolist()
        results_dict_logger.save(
            uid + presuffix + suffix, test_acc_meter.name + "_ConfMatr", cm_np_list
        )
    except:
        logger.debug(
            f"Saving of {test_acc_meter.name} confusion matrix raised an error. Skipping..."
        )
        pass


########
