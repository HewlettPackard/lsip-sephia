import numpy as np
import time
import torch
import torch.nn as nn
import json
import os
import snntorch as snn


def RemapTargets(targets, classes, src_class_count=10, device="cpu"):
    remap = torch.zeros(src_class_count, dtype=torch.long, device=device)
    for i_v, v in enumerate(classes):
        remap[v] = i_v
    return remap[targets]


def quick_dump_to_json(run_states, label, repeat=None):
    filepath = create_results_dir(label)

    if repeat is not None:
        repeat_str = f"R{str(repeat).zfill(2)}_"
    else:
        repeat_str = ""

    time_hm = time.strftime("%Hh%Mm", time.localtime())
    full_filepath = f'{filepath}/{repeat_str}run_states_({time_hm}).json'
    with open(full_filepath, 'w') as json_file:
        json.dump(run_states, json_file, indent=4)

    print(f"run_states dict saved to: {full_filepath}")


def quick_dump_to_npy(label, data, data_name, repeat=None):
    filepath = create_results_dir(label)

    if repeat is not None:
        repeat_str = f"R{str(repeat).zfill(2)}_"
    else:
        repeat_str = ""

    time_hm = time.strftime("%Hh%Mm", time.localtime())
    full_filepath = f'{filepath}/{repeat_str}run_{data_name}_({time_hm}).npy'
    np.save(full_filepath, data)

    print(f"{data_name} saved to: {full_filepath}")


def save_checkpoint(model, p, optimizer=None, repeat=0, epoch=0, acc=0):
    filepath = create_results_dir(p["label"])
    filepath = filepath + f"/cps_R{str(repeat).zfill(2)}"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    try:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'params': p,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        time_hm = time.strftime("%Hh%Mm", time.localtime())
        torch.save(checkpoint, f'{filepath}/R{str(repeat).zfill(2)}_E{str(epoch).zfill(2)}_Acc{acc:.2f}_({time_hm}).pth')
        return True
        
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")
        return False


def create_results_dir(label):
    if not os.path.exists("results"):
        os.makedirs("results")

    filepath = f"results/{time.strftime('%Y-%m-%d', time.localtime())}__{label}"
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    return filepath