import sys
import time
import torch
import torch.nn as nn
import snntorch.functional as SF
from snntorch import spikegen
from matrepr import to_str

sys.path.append("/home/hejda/work/sephia/")

from utils import RemapTargets, quick_dump_to_json, save_checkpoint
from SEPhIA_OE2 import OESNN_SEPhIA_MultiTiled2
import picsim
from picsim.components.lasers import CombLaser_Ideal as IdealComb
from picsim.datasets.mnist_reduced import ReducedFMNIST
from picsim import DEFAULT_DEVICE


def reset_params_uniform_and_diag(net, 
                                  range_nonsigma=(-2, 2), 
                                  range_sigma=(0.9, 1.1),
                                  suppress_inhibitory_connections=False,
                                  suppress_shift=-2,
                                  LOGGER=None):
    if True:
        LOGGER.info(f"  - model_parameters (non-sigma) initiated: uniform_(a={range_nonsigma[0]}, b={range_nonsigma[1]})")
        if suppress_inhibitory_connections:
            LOGGER.info(f"  - model_parameters (non-sigma): inhibitory MRR connections suppressed by {suppress_shift}.")

        for name, param in net.named_parameters():
            if "Sigma" not in name:
                torch.nn.init.uniform_(param, a=range_nonsigma[0], b=range_nonsigma[1])
                
                if suppress_inhibitory_connections:
                    if param.data.ndim == 2:
                        param.data[:,1::2] += suppress_shift

    if True:
        LOGGER.info(f"  - model_parameters (sigma) initiated:     uniform_(a={range_sigma[0]}, b={range_sigma[1]})")
        for name, param in net.named_parameters():
            if "Sigma" in name:
                torch.nn.init.uniform_(param, a=range_sigma[0], b=range_sigma[1])


def process_confusion_matrix_and_get_accuracy(
    val_cm, ds, title="Val_CM", repeat=None, epoch=0, LOGGER=None, device=DEFAULT_DEVICE
):
    val_class_counts = torch.clamp(torch.sum(val_cm, dim=1, keepdim=True), min=1e-8)
    val_cm_norm = torch.round(val_cm / val_class_counts * 100, decimals=2)
    val_cm_norm_TPs = (
        torch.sum(val_cm_norm * torch.eye(len(ds.classes), device=device), dim=0)
        .cpu()
        .numpy()
    )
    val_cm_TPs = torch.sum(val_cm * torch.eye(len(ds.classes), device=device), dim=0)
    val_accuracy = (torch.sum(val_cm_TPs) / torch.sum(val_cm)).item() * 100

    cm_title = f"{title}: "
    if repeat is not None:
        cm_title += f"[R:{repeat}], "
    if epoch is not None:
        cm_title += f"[E:{epoch}], "
    cm_title += f"[acc_per_class: {val_cm_norm_TPs}]"

    val_cm_str = to_str(
        val_cm.cpu().numpy(),
        title=cm_title,
        row_labels=[f"Target({i})" for i in ds.classes],
        col_labels=[f"Pred({i})" for i in ds.classes],
    )
    LOGGER.info(val_cm_str)
    return val_accuracy, val_cm_norm_TPs


def Load_Dataset(p, LOGGER):
    if p["DS_name"] == "FMNIST":
        ds = ReducedFMNIST(
            specific_classes=p["DS_classes"],
            n_features=p["SNN_layer_widths"][0][0],
            batch_size=p["batch_size"],
            N_samples_train=p["DS_N_samples_train"],
            N_samples_val=p["DS_N_samples_val"],
            N_samples_test=p["DS_N_samples_test"],
        )
    else:
        raise ValueError(f"Unknown dataset name: {p['DS_name']}")

    LOGGER.info(f"Class training sample counts: {ds.train_class_counts}")
    LOGGER.info(f"Class validation sample counts: {ds.val_class_counts}")
    LOGGER.info(f"Class test sample counts: {ds.test_class_counts}")
    return ds


def Get_Model_Regular(p, device):
    raise NotImplementedError("Regular (non-tiled) model is not implemented in this codebase.")


def Get_Model_Tiled(p, device):
    comb = IdealComb(
        num_lines=p["input_size"],
        peak_channel_power_dbm=p["comb_peak_channel_power_dbm"],
        spacing_hz=p["comb_channel_spacing_hz"],
        device=device,
    )
    picsim.set_global_wavelengths(comb.peak_locations)

    net = OESNN_SEPhIA_MultiTiled2(
        comb_class=comb,
        num_tsteps=p["num_tsteps"],
        layer_widths=p["SNN_layer_widths"],
        tile_counts=p["SNN_tile_counts"],
        eoconv_unity_opt_P_per_WDM_channel_dbm=p["eoconv_unity_opt_P_per_WDM_channel_dbm"],
        eoconv_spatial_dim_broadcast_divide_power=p["eoconv_spatial_dim_broadcast_divide_power"],
        sephia_specs=p["sephia_specs"],
        lif_thresholds=p["SNN_lif_thresholds"],
        lif_betas=p["SNN_lif_betas"],
        reset_mechanism=p["SNN_lif_reset_mechanism"],
        ref_period_enabled=p["ref_period_enabled"],
        ref_period_timesteps=p["ref_period_timesteps"],
        use_bpds=p["use_bpds"],
        sephia_post_divide_power=p["sephia_post_divide_power"],
        disable_sephia_at_output=True,
        dropout=(p["dropout_enabled"], p["dropout_p"]),
        batch_size=p["batch_size"],
        configs_override=p["configs_override"],
        device=device,
    ).to(device)
    return net


def Get_Scheduler(optimizer, p):
    if p["optimizer_scheduler_type"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=p["num_epochs"], eta_min=p["optimizer_scheduler_min_lr"]
        )
    elif p["optimizer_scheduler_type"] == "Linear":
        total_epochs, lr_min, lr_max = (
            p["num_epochs"],
            p["optimizer_scheduler_min_lr"],
            p["optimizer_lr"],
        )

        def linear_decay_lambda(epoch):
            if epoch >= total_epochs:
                return lr_min / lr_max
            return 1.0 - (epoch / total_epochs) * (1.0 - (lr_min / lr_max))

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=p["optimizer_scheduler_min_lr"] / p["optimizer_lr"],
            total_iters=p["num_epochs"],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=linear_decay_lambda
        )
    else:
        raise ValueError(f"Unknown optimizer scheduler type: {p['optimizer_scheduler_type']}")
    return scheduler


def Run_Single_Epoch_TrainAndVal(
    net, p, i_r, epoch, ds, criterion, optimizer, scheduler,
    train_loss_log, train_acc_log, train_acc_epochs_log,
    val_loss_log, val_acc_log, val_acc_per_class_log,
    LOGGER, device,
):
    net.train()
    start_time_epoch = time.time()
    printout = f"-- Epoch [{epoch + 1}/{p['num_epochs']}]: "

    train_losses = []
    train_TPs = 0
    
    for i_d, (X, y) in enumerate(ds.train_loader):
        if p["use_ratecoded_MNIST"]:
            spk_rec, mem_rec = net(
                spikegen.rate(X.to(device), num_steps=p["num_tsteps"])
            )
        else:
            spk_rec, mem_rec = net(X.expand(p["num_tsteps"], -1, -1))

        # Since the updated version returns full memory, we have to select it
        spk_rec = spk_rec[net.num_layers - 1]
        mem_rec = mem_rec[net.num_layers - 1]

        y_test = RemapTargets(y, ds.classes).to(device)
        loss = criterion(mem_rec, y_test)
        train_losses.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if p["accuracy_from_membrane_potential"]:
                results = torch.argmax(torch.sum(mem_rec, dim=0), dim=1)
            else:
                results = torch.argmax(torch.sum(spk_rec, dim=0), dim=1)

            TPs = torch.sum(torch.eq(results, y_test).float()).item()
            train_TPs += TPs
            if p["save_detailed_training_accuracies"]:
                train_acc_log.append(TPs / y_test.numel() * 100)
    train_acc_epochs_log.append(train_TPs / p["DS_N_samples_train"] * 100)
    printout += f"[mean_train_acc: {train_acc_epochs_log[-1]:.2f}%], "

    avg_train_loss = sum(train_losses) / len(train_losses)
    train_loss_log.append(avg_train_loss)
    printout += f"[avg_train_loss: {avg_train_loss:.4f}], "

    if p["optimizer_scheduler_enabled"]:
        scheduler.step()
        printout += f"[lr: {float(scheduler.get_last_lr()[0]):.6f}], "

    net.eval()
    with torch.no_grad():
        val_losses = []
        val_correct = 0
        val_total = 0

        val_cm = torch.zeros((len(ds.classes), len(ds.classes)), device=device)
        
        for X_val, y_val in ds.val_loader:
            y_val = RemapTargets(y_val, ds.classes).to(device)
            if p["use_ratecoded_MNIST"]:
                X_val = spikegen.rate(X_val, num_steps=p["num_tsteps"]).to(device)
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
        printout += f"[avg_val_loss: {avg_val_loss:.4f}], "

        val_accuracy, val_cm_norm_TPs = process_confusion_matrix_and_get_accuracy(
            val_cm,
            ds,
            title="Val_CM",
            repeat=i_r,
            epoch=epoch,
            LOGGER=LOGGER,
            device=device,
        )

        val_acc_per_class_log.append(val_cm_norm_TPs.tolist())

        val_acc_log.append(val_accuracy)
        printout += f"[val_acc: {val_accuracy:.2f}%], "

    elapsed_time = time.time() - start_time_epoch
    printout += f"[epoch_time: {elapsed_time // 60:.0f}m{elapsed_time % 60:.0f}s], "
    LOGGER.info(printout)

    return net, val_accuracy


def Generate_Label(p, label):
    ds_stats = f'({p["DS_name"][0:3]}{p["SNN_layer_widths"][0][0]}={str(len(p["DS_classes"]))})'
    comb_stats = f'({p["comb_channel_spacing_hz"]*1e-9:.1f}GHz,{p["comb_peak_channel_power_dbm"]}dBm)'
    mrr_stats = f'({int(p["configs_override"]["specs"]["mrr_Q"]/1000)}K,{p["configs_override"]["specs"]["mrr_IL_db"]:.2f}dB,{p["configs_override"]["specs"]["mrr_ER_db"]:.0f}dB)'
    
    if p["ref_period_enabled"]:
        rp_len = p["ref_period_timesteps"]
    else:
        rp_len = 0
    rp_stats = f'(RP={rp_len})'

    if p["SNN_model"] == "tiled":
        tile_stats = f'(Tiled={p["SNN_tile_counts"][0]})'
    elif p["SNN_model"] == "regular":
        tile_stats = f"(NonTiled)"

    return label + "_" + ds_stats + comb_stats + mrr_stats + rp_stats + tile_stats


def Get_Test_Accuracy(net, ds, p, i_r, LOGGER, device):
    net.eval()
    with torch.no_grad():
        test_cm = torch.zeros((len(ds.classes), len(ds.classes)), device=device)

        for (X_test, y_test) in ds.test_loader:
            y_test = RemapTargets(y_test, ds.classes).to(device)
            if p["use_ratecoded_MNIST"]:
                X_test = spikegen.rate(X_test, num_steps=p["num_tsteps"]).to(device)
            else:
                X_test = X_test.expand(p["num_tsteps"], -1, -1).to(device)

            spk_rec, mem_rec = net(X_test)

            # Since the updated version returns full memory, we have to select it
            spk_rec = spk_rec[net.num_layers - 1]
            mem_rec = mem_rec[net.num_layers - 1]

            if p["accuracy_from_membrane_potential"]:
                results = torch.argmax(torch.sum(mem_rec, dim=0), dim=1)
            else:
                results = torch.argmax(torch.sum(spk_rec, dim=0), dim=1)
            for i_v, v in enumerate(results):
                test_cm[y_test[i_v], v] += 1
        
        test_accuracy, _ = process_confusion_matrix_and_get_accuracy(
            test_cm,
            ds,
            title="Test_CM",
            repeat=i_r,
            epoch=None,
            LOGGER=LOGGER,
            device=device,
        )        

        return test_accuracy, test_cm


def Make_Block_Diagonal(matrix, F):
    """Creates a block diagonal matrix by zeroing out off-diagonal blocks."""
    result = matrix.clone()
    rows, cols = matrix.shape
    block_height = rows // F
    block_width = cols // F
    
    for i in range(F):
        for j in range(F):
            if i != j:
                row_start = i * block_height
                row_end = (i + 1) * block_height if i < F - 1 else rows
                col_start = j * block_width
                col_end = (j + 1) * block_width if j < F - 1 else cols
                result[row_start:row_end, col_start:col_end] = 0
    
    return result