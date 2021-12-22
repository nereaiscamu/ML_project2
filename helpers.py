from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import sys
sys.path.append('./Data/')



# Andrew's
def get_loss_vl(outputs, targets):
  # Transpose because torch expects dim 1 to contain the classes
  # Add ignore_index
  return F.cross_entropy(outputs.transpose(1, 2), targets, ignore_index=-1)

# Andrew's
def get_accuracy_vl(outputs, targets):
  flat_outputs = outputs.argmax(dim=2).flatten()
  flat_targets = targets.flatten()

  # Mask the outputs and targets
  mask = flat_targets != -1

  return 100 * (flat_outputs[mask] == flat_targets[mask]).sum() / sum(mask)

def get_val_loss(model, device, val_loader=None, val_set=None):
    """
    Computes and returns the validation loss for a given set or loader

    Args:
        model: Model for which the validation loss gets computed
        device: Device on which the computation should be run (cuda or cpu)
        val_loader: DataLoader of the validation set
        val_set: Validation set

    Returns:
        Validation loss
    """
    model.eval()
    if val_set != None:
        val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False) 

    assert val_loader != None

    for _, batch in enumerate(val_loader):
        inputs = batch["input"].float().to(device)
        lengths = batch["length"]
        targets = batch["target"][:, :max(lengths)].to(device)

        outputs = model(inputs, lengths)
        loss = get_loss_vl(outputs, targets)
    
    return loss.item()


# DM
def evaluate_model(model, device, loader=None, dataset=None):
    """
    Evaluates the model and returns the accuracy

    Args:
        model: Model for which the validation loss gets computed
        device: Device on which the computation should be run (cuda or cpu)
        val_loader: DataLoader of the validation set
        val_set: Validation set

    Returns:
        Validation loss
    """
    model.eval()
    if dataset != None:
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False) 

    assert loader != None

    correct = 0
    total = 0
    for _, batch in enumerate(loader):
        inputs = batch["input"].float().to(device)
        lengths = batch["length"]
        targets = batch["target"][:, :max(lengths)].to(device)

        preds = model(inputs, lengths)
        preds = preds.argmax(dim=2).flatten()
        targets = targets.flatten()

        # Mask the outputs and targets
        mask = targets != -1
        
        correct += (preds[mask] == targets[mask]).sum()
        total += sum(mask)
    
    total = total.cpu().numpy()
    acc = 100 * correct.item()/total    
    return acc


def split_dataset(dataset):
    """
    Splits the dataset in 80% train, 10% validation, 10% test

    Args:
        dataset: Dataset which gets splitted

    Returns:
        Splitted dataset in train, validation and test set
    """
    # Split Train/Val/Test
    len_dataset = len(dataset)

    len_tr = int(0.8*len_dataset)
    len_rem = len_dataset - len_tr
    train_set, rem_set = torch.utils.data.random_split(dataset, [len_tr, len_rem])

    len_rem_set = len(rem_set)
    len_val = int(0.5*len_rem_set)
    len_test = len_rem_set - len_val
    val_set, test_set = torch.utils.data.random_split(rem_set, [len_val, len_test])

    return train_set, val_set, test_set
