import sys
import os
import time
import torch
import torch.nn as nn
import snntorch.functional as SF
import numpy as np
from snntorch import spikegen

# Import SEPhIA
## Add directory above the parent to path, to allow imports from there
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import (
    RemapTargets,
    quick_dump_to_json,
    quick_dump_to_npy,
    save_checkpoint,
)

import picsim
picsim.enable_logging_into_file("training.log")
picsim.set_logger_verbose(True)
from picsim import LOGGER
from funcs_training import *

#########################################
LOGGER.info("=== RUN STARTED ===")
#########################################
# Experiment parameters
p = {}

p["train_from_checkpoint"] = False
if p["train_from_checkpoint"]:
    _label = p["label"] # Prevent overwrite from previous
    checkpoint_path = "./path/to/cp.pth"
    cp = torch.load(checkpoint_path, weights_only=True)
    p = cp["params"]

    p["train_from_checkpoint"] = True
    p["checkpoint_path"] = checkpoint_path
    p["label"] = _label

else:
    p["num_tsteps"] = 35
    p["batch_size"] = 128
    
    p["DS_name"] = "FMNIST"
    p["DS_classes"] = [0, 1, 4, 5]
    p["DS_N_samples_train"] = 21760
    p["DS_N_samples_val"] = 2048  # 4096
    p["DS_N_samples_test"] = 3900  # 8192
    p["use_ratecoded_MNIST"] = True

    p["SNN_model"] = "tiled"  # "regular" or "tiled"
    p["SNN_layer_widths"] = ((32, 16), (16, len(p["DS_classes"])))
    p["input_size"] = p["SNN_layer_widths"][0][0] 
    p["SNN_tile_counts"] = (2, 1)
    p["SNN_lif_thresholds"] = (0.5, 0.25)
    p["SNN_lif_betas"] = (0.99, 0.99)
    p["SNN_lif_reset_mechanism"] = "zero"
    p["ref_period_enabled"] = False
    p["ref_period_timesteps"] = 1

    p["comb_channel_spacing_hz"] = 63e9
    p["comb_peak_channel_power_dbm"] = -8
    p["eoconv_unity_opt_P_per_WDM_channel_dbm"] = -8
    p["eoconv_spatial_dim_broadcast_divide_power"] = False
    p["sephia_post_divide_power"] = False
    p["sephia_specs"] = {
        "mrr_Q": 20000,
        "mrr_ER_db": 20.0,
        "mrr_IL_db": 0.0,
        "mrr_reso_shift_range_pm": -210,
    }
    p["use_bpds"] = True

    p["optimizer_lr"] = 4e-2
    p["optimizer_scheduler_enabled"] = True
    p["optimizer_scheduler_type"] = "CosineAnnealingLR"
    p["optimizer_scheduler_min_lr"] = 0
    p["optimizer_weight_decay"] = 0
    p["init_range_nonsigma"] = (-1, 2)
    p["init_range_sigma"] = (0.9, 1.1)
    p["dropout_enabled"] = True
    p["dropout_p"] = 0.15

    p["evaluate_untrained_model"] = True
    p["accuracy_from_membrane_potential"] = False
    
    p["configs_override"] = {
        "specs": {
            "mrr_Q": 20000,
            "mrr_ER_db": 10.0,
            "mrr_IL_db": 0.0,
            "mrr_reso_shift_range_pm": -250,
            "pds_noise_thermal_enabled": True,
            "pds_noise_shot_enabled": True,
        }
    }

p["cuda_id"] = 6
p["repeats"] = 2
p["num_epochs"] = 20
enable_saving = True
p["save_detailed_training_accuracies"] = True
p["train_iteratively"] = False
p["iterative_warmup_epochs"] = 0


def RunTraining(p, label="RunX"):
    p["label"] = Generate_Label(p, label)
    print(f"Experiment label: {p['label']}")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{p['cuda_id']}")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    LOGGER.info(f"CUDA: Using device: {device}")

    ds = Load_Dataset(p, LOGGER)

    if p["SNN_model"] == "regular":
        net = Get_Model_Regular(p, device)
    elif p["SNN_model"] == "tiled":
        net = Get_Model_Tiled(p, device)
    else:
        raise ValueError(f"Unknown SNN model type: {p['SNN_model']}")
    
    criterion = SF.loss.ce_max_membrane_loss(weight=ds.train_class_weights.to(device))

    LOGGER.info(f"Experiment: {p['label']}, repeats * epochs: {p['repeats']} * {p['num_epochs']}")
    
    for i_r in range(p["repeats"]):
        start_time_rep = time.time()
        LOGGER.info(f"- Repeat {i_r}...")

        run_states = {}
        run_states["parameters"] = p

        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=p["optimizer_lr"],
            weight_decay=p["optimizer_weight_decay"],
            fused=None,
        )  # Adds weight decay to prevent overfitting to easy classes

        if p["optimizer_scheduler_enabled"]:
            scheduler = Get_Scheduler(optimizer, p)

        train_loss_log = []
        train_acc_log = []
        train_acc_epochs_log = []
        val_loss_log = []
        val_acc_log = []
        val_acc_per_class_log = []

        if p["train_from_checkpoint"] is True:
            checkpoint = torch.load(
                p["checkpoint_path"], map_location=device, weights_only=True
            )
            try:
                net.load_state_dict(checkpoint["model_state_dict"], strict=True)
                LOGGER.info(f"train_from_checkpoint = True; using {p['checkpoint_path']}")
            except Exception as e:
                LOGGER.error(f"Failed to load checkpoint: {str(e)}")
        else:
            reset_params_uniform_and_diag(
                net,
                range_nonsigma=p["init_range_nonsigma"],
                range_sigma=p["init_range_sigma"],
                suppress_inhibitory_connections=True,
                suppress_shift=-1,
                LOGGER=LOGGER,
            )

        # Evaluate (validate) model once, as it is initiated (and untrained) -> this should serve as a baseline
        if p["evaluate_untrained_model"]:
            net.eval()
            with torch.no_grad():
                val_losses = []
                val_cm = torch.zeros((len(ds.classes), len(ds.classes)), device=device)

                # Iterate over samples
                for X_val, y_val in ds.val_loader:
                    y_val = RemapTargets(y_val, ds.classes).to(device)
                    if p["use_ratecoded_MNIST"]:
                        X_val = spikegen.rate(X_val, num_steps=p["num_tsteps"]).to(
                            device
                        )
                    else:
                        X_val = X_val.expand(p["num_tsteps"], -1, -1).to(device)

                    spk_rec, mem_rec = net(X_val)

                    # Since the updated version returns full memory, we have to select it
                    spk_rec = spk_rec[net.num_layers - 1]
                    mem_rec = mem_rec[net.num_layers - 1]

                    val_loss = criterion(mem_rec, y_val)
                    val_losses.append(val_loss.item())

                    if p["accuracy_from_membrane_potential"]:
                        results = torch.argmax(torch.sum(mem_rec, dim=0), dim=1)
                    else:
                        results = torch.argmax(torch.sum(spk_rec, dim=0), dim=1)
                    for i_v, v in enumerate(results):
                        val_cm[y_val[i_v], v] += 1
                
                avg_val_loss = sum(val_losses) / len(val_losses)
                val_loss_log.append(avg_val_loss)

                val_accuracy, _ = process_confusion_matrix_and_get_accuracy(
                    val_cm,
                    ds,
                    title="Untrained_CM",
                    repeat=None,
                    epoch=None,
                    LOGGER=LOGGER,
                    device=device,
                )

                val_acc_log.append(val_accuracy)
                printout = f"Untrained model: [avg_val_loss: {avg_val_loss:.4f}], [val_accuracy: {val_accuracy:.2f}%], "
                LOGGER.info(printout)

        best_val_accuracy = 0.0
        best_val_state_dict = None

        for epoch in range(p["num_epochs"]):
            net, val_accuracy = Run_Single_Epoch_TrainAndVal(
                net,
                p,
                i_r,
                epoch,
                ds,
                criterion,
                optimizer,
                scheduler,
                train_loss_log,
                train_acc_log,
                train_acc_epochs_log,
                val_loss_log,
                val_acc_log,
                val_acc_per_class_log,
                LOGGER,
                device,
            )

            if not p["train_iteratively"] or (
                (best_val_accuracy <= val_accuracy) or (epoch < p["iterative_warmup_epochs"])
            ):
                best_val_accuracy = val_accuracy
                best_val_state_dict = net.state_dict()

                if enable_saving:
                    save_checkpoint(
                        net,
                        p,
                        optimizer=optimizer,
                        repeat=i_r,
                        epoch=epoch,
                        acc=val_accuracy,
                    )
                    LOGGER.info("Accepted latest model, checkpoint saved.")
                else:
                    LOGGER.info("Accepted latest model, checkpoint not saved.")
            else:
                LOGGER.info("Rejected latest model, reloading state dict...")
                net.load_state_dict(best_val_state_dict)

        test_accuracy, test_cm = Get_Test_Accuracy(
            net=net, ds=ds, p=p, i_r=i_r, LOGGER=LOGGER, device=device
        )
        rep_printout = f"- test acc (global): {test_accuracy:.4f}, "

        run_states["confmat_test"] = test_cm.int().cpu().numpy().tolist()
        run_states["train_loss"] = train_loss_log
        run_states["val_loss"] = val_loss_log
        run_states["val_accuracy"] = val_acc_log
        run_states["val_accuracy_per_class"] = val_acc_per_class_log

        if enable_saving:
            quick_dump_to_json(
                run_states,
                label=p["label"],
                repeat=i_r,
            )
        if p["save_detailed_training_accuracies"]:
            quick_dump_to_npy(
                label=p["label"],
                data=np.array(train_acc_log),
                data_name="train_acc_log",
                repeat=i_r,
            )

        elapsed_time = time.time() - start_time_rep
        LOGGER.info(rep_printout + f"repeat time: [{elapsed_time // 60:.0f}m{elapsed_time % 60:.0f}s]")

    del net


def Repopulate_MRR_Params(p, specs):
    """Repopulate the MRR parameters in the experiment parameters dictionary."""
    p["sephia_specs"]["mrr_Q"] = specs["mrr_Q"]
    p["sephia_specs"]["mrr_ER_db"] = specs["mrr_ER_db"]
    p["sephia_specs"]["mrr_IL_db"] = 0

    p["configs_override"]["specs"]["mrr_Q"] = specs["mrr_Q"]
    p["configs_override"]["specs"]["mrr_ER_db"] = specs["mrr_ER_db"]
    p["configs_override"]["specs"]["mrr_IL_db"] = specs["mrr_IL_db"]

    LOGGER.info(f"Populated MRR specs: {p['sephia_specs']}")


def Repopulate_Comb_Params(p, spacing_ghz):
    """Repopulate the comb parameters in the experiment parameters dictionary."""
    p["comb_channel_spacing_hz"] = spacing_ghz * 1e9
    p["sephia_specs"]["mrr_reso_shift_range_pm"] = -210 * spacing_ghz / 63
    p["configs_override"]["specs"]["mrr_reso_shift_range_pm"] = -250 * spacing_ghz / 63
    LOGGER.info(f"Populated comb specs: {p['comb_channel_spacing_hz']} Hz")
    return p


def Repopulate_Optical_Power(p, power_dbm):
    """Repopulate the optical power parameters in the experiment parameters dictionary."""
    p["comb_peak_channel_power_dbm"] = power_dbm
    p["eoconv_unity_opt_P_per_WDM_channel_dbm"] = power_dbm
    LOGGER.info(f"Populated optical power: {p['comb_peak_channel_power_dbm']} dBm")
    return p


if __name__ == "__main__":

    MRR_specs1 = {
        "mrr_Q": 20000,
        "mrr_ER_db": 15.0,
        "mrr_IL_db": 0.0,
    }

    MRR_specs2 = {
        "mrr_Q": 20000,
        "mrr_ER_db": 6.0,
        "mrr_IL_db": 0.0,
    }

    MRR_specs3 = {
        "mrr_Q": 10000,
        "mrr_ER_db": 6.0,
        "mrr_IL_db": 0.0,
    }

    Repopulate_MRR_Params(p, MRR_specs3)
    Repopulate_Comb_Params(p, 50.0)
    Repopulate_Optical_Power(p, -12.0)
    RunTraining(p, label="RunD(NoDecay,LR=4e-2)")