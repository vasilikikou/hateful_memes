import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from transformers import AdamW
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


parser = argparse.ArgumentParser(description="Takes as arguments the paths to the data files to train a BERT for "
                                             "sequence classification and the path to the folder in which "
                                             "to save the model checkpoint")
parser.add_argument("--task", help="Which task to perform. Choose between \"gold_hate\", \"gold_pc\" and \"gold_attack\"")
parser.add_argument("--train_path", help="Path to json file with training data")
parser.add_argument("--val_path", help="Path to json file with validation data")
parser.add_argument("--label_path", help="Path to csv file that contains the labels")
parser.add_argument("--results_path", help="Path to folder in which to save the model checkpoint")

# Training method
def training(model, train_dataloader, criterion, optimizer, device):
    # Set to train mode
    model.train()
    total_loss, total_accuracy = 0, 0
    # Iterate through the training batches
    for batch in tqdm(train_dataloader, desc="Iteration"):
        # Push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        # Clear gradients
        model.zero_grad()
        # Get model outputs
        outputs = model(sent_id, attention_mask=mask)
        # Get loss
        loss = criterion(outputs.logits, labels)
        # Add to the total loss
        total_loss = total_loss + loss
        # Backward pass to calculate the gradients
        loss.backward()
        # Update parameters
        optimizer.step()
    # Compute the training loss of the epoch
    epoch_loss = total_loss / len(train_dataloader)

    return model, epoch_loss

# Evaluation method
def evaluate(model, val_dataloader, criterion, device):
    print("\nEvaluating...")
    # Set to eval mode
    model.eval()
    total_loss, total_accuracy = 0, 0
    # Iterate through the validation batches
    for batch in val_dataloader:
        # Push the batch to gpu
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        # Deactivate autograd
        with torch.no_grad():
            # Get model outputs
            outputs = model(sent_id, attention_mask=mask)
            # Get loss
            loss = criterion(outputs.logits, labels)
            total_loss = total_loss + loss

    # Compute the validation loss of the epoch
    epoch_loss = total_loss / len(val_dataloader)

    return model, epoch_loss

def encode_targets(gold, tags):
    """
    Encoding method for multilabel tasks
    :param gold: List with the gold labels of each instance separated with ";"
    :param tags: List of existing labels to use for encoding
    :return: List with the encoded binary gold vectors for each instance
    """
    encoded_targets = []
    for g in gold:
        # Create the binary vector with all zeros
        y = np.zeros(len(tags), dtype=float)
        # Get the labels of the instance
        target_tags = g.split(";")

        for i in range(0, len(tags)):
            # If the label is in the list assign 1 to its position
            if tags[i] in target_tags:
                y[i] = 1
        encoded_targets.append(y)
    return encoded_targets

def encode_binary(gold):
    encoded_targets = []
    for g in gold:
        # If the label for the hateful on not task is hateful then assign 1
        if g == "hateful":
            y = np.array([1])
        else:
            y = np.array([0])
        encoded_targets.append(y)
    return encoded_targets

def main(task_mode, train_file_path, val_file_path, labels_file_path, results_path):
    # Define the GPU ids to be used
    d_ids = [0]
    device = f"cuda:{d_ids[0]}"

    # Load labels
    labels_df = pd.read_csv(labels_file_path, header=None)
    labels_list = labels_df[0].to_list()

    # Load data
    with open(train_file_path, encoding="utf-8") as j_file:
        train_data = pd.read_json(j_file, lines=True)
    train_data["text"] = train_data["text"].apply(lambda x: x.lower())
    train_images_texts = dict(zip(train_data.id, train_data.text))
    train_images_labels = dict(zip(train_data.id, train_data[task_mode]))

    with open(val_file_path, encoding="utf-8") as j_file:
        val_data = pd.read_json(j_file, lines=True)
    val_data["text"] = val_data["text"].apply(lambda x: x.lower())
    val_images_texts = dict(zip(val_data.id, val_data.text))
    val_images_labels = dict(zip(val_data.id, val_data[task_mode]))

    # Define BERT model to be used
    model_name = "bert-base-uncased"
    # Construct a BERT tokenizer based on WordPiece
    bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # Set max_len as the length of the largest training sentence
    max_len = max([len(bert_tokenizer.encode(s)) for s in list(train_images_texts.values())])
    print("The maximum sentence length in training based on BERT BPEs is", max_len)

    if max_len > 512:
        print("Reducing max length to 512")
        max_len = 512

    # Tokenize and encode sentences in each set
    x_train = bert_tokenizer.__call__(
        list(train_images_texts.values()),
        max_length = max_len,
        padding=True,
        truncation=True,
        return_attention_mask = True,
        return_tensors="pt"
    )
    x_val = bert_tokenizer.__call__(
        list(val_images_texts.values()),
        max_length = max_len,
        padding=True,
        truncation=True,
        return_attention_mask = True,
        return_tensors="pt"
    )

    # Encode gold labels
    if task_mode != "gold_hate":
        train_targets = encode_targets(list(train_images_labels.values()), labels_list)
        val_targets = encode_targets(list(val_images_labels.values()), labels_list)
    else:
        train_targets = encode_binary(list(train_images_labels.values()))
        val_targets = encode_binary(list(val_images_labels.values()))

    # Convert lists to tensors in order to feed them to our PyTorch model
    train_y = torch.FloatTensor(train_targets)
    val_y = torch.FloatTensor(val_targets)

    # Define batch size
    batch_size = 32

    # Create a dataloader for each set
    print("Creating dataloaders...")
    # Train
    train_dataset = TensorDataset(x_train['input_ids'], x_train['attention_mask'], train_y)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    # Val
    val_dataset = TensorDataset(x_val['input_ids'], x_val['attention_mask'], val_y)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


    print("Creating model...")
    if task_mode != "gold_hate":
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(labels_list),
                                                              output_attentions=True)
    else:
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1,
                                                          output_attentions=True)

    # Push model to gpu
    model = model.to(device)
    # Define the optimizer and the learning rate
    optimizer = AdamW(model.parameters(), lr = 2e-5)
    # Define the loss
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_epoch = -1
    train_losses=[]
    val_losses=[]
    epochs = 100
    # Define the number of epochs to wait for early stopping
    patience = 3

    print("Starting training...")
    # Train the model
    for epoch in range(epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        model, train_loss = training(model, train_dataloader, criterion, optimizer, device)
        model, val_loss = evaluate(model, val_dataloader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print("\nTraining Loss:", train_loss)
        print("Validation Loss:", val_loss)

        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            # Save the model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
            output_model_file = os.path.join(results_path, "txtbert_" + task_mode + ".bin")
            torch.save(model_to_save.state_dict(), output_model_file)

        # Early stopping
        if ((epoch - best_epoch) >= patience):
            print("No improvement in", patience, "epochs. Stopped training.")
            break

if __name__ == '__main__':
    args = parser.parse_args()
    task = args.task
    train_file_path = args.train_path
    val_file_path = args.val_path
    label_file_path = args.label_path
    results_path = args.results_path

    main(task, train_file_path, val_file_path, label_file_path, results_path)