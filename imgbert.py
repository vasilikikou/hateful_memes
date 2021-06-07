import os
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from transformers import AdamW
from build_imgbert import ImgBERT
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import TensorDataset, DataLoader


parser = argparse.ArgumentParser(description="Takes as arguments the paths to the data files and image embeddings file "
                                             "in order to train a BERT for image and sequence classification and "
                                             "the path to the folder in which "
                                             "to save the model checkpoint")
parser.add_argument("--task", help="Which task to perform. Choose between \"gold_hate\", \"gold_pc\" and \"gold_attack\"")
parser.add_argument("--train_path", help="Path to json file with training data")
parser.add_argument("--val_path", help="Path to json file with validation data")
parser.add_argument("--label_path", help="Path to csv file that contains the labels")
parser.add_argument("--results_path", help="Path to folder in which to save the model checkpoint")
parser.add_argument("--image_embeddings_path", help="Path to the pickle file that contains the image embeddings")

# Training method
def training(model, train_dataloader, criterion, optimizer, device):
    # Set to train mode
    model.train()
    total_loss, total_accuracy = 0, 0
    # Iterate through the training batches
    for batch in tqdm(train_dataloader, desc="Iteration"):
        # Push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels, img_embs = batch
        # Clear gradients
        model.zero_grad()
        # Get model outputs
        logits, _ = model(sent_id, mask=mask, img_emb=img_embs)
        # Get loss
        loss = criterion(logits, labels)
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
        sent_id, mask, labels, img_embs = batch
        # Deactivate autograd
        with torch.no_grad():
            # Get model outputs
            logits, _ = model(sent_id, mask=mask, img_emb=img_embs)
            # Get loss
            loss = criterion(logits, labels)
            total_loss = total_loss + loss

    # Compute the validation loss of the epoch
    epoch_loss = total_loss / len(val_dataloader)

    return model, epoch_loss

def encode_targets(gold, tags):
    encoded_targets = []
    for g in gold:
        y = np.zeros(len(tags), dtype=float)

        target_tags = g.split(";")

        for i in range(0, len(tags)):
            # can leave zero if zero, else make one
            if tags[i] in target_tags:
                y[i] = 1
        encoded_targets.append(y)
    return encoded_targets

def encode_binary(gold):
    encoded_targets = []
    for g in gold:
        if g == "hateful":
            y = np.array([1])
        else:
            y = np.array([0])
        encoded_targets.append(y)
    return encoded_targets


def main(task_mode, train_file_path, val_file_path, labels_file_path, img_emb_path, results_path):
    # Define the GPU ids to be used
    d_ids = [0]
    device = f"cuda:{d_ids[0]}"

    # Load labels
    labels_df = pd.read_csv(labels_file_path, header=None)
    labels_list = labels_df[0].to_list()

    # Load image embeddings
    with open(img_emb_path, "rb") as i:
        all_images_embs = pickle.load(i)

    # Load data
    with open(train_file_path, encoding="utf-8") as j_file:
        train_data = pd.read_json(j_file, lines=True)
    train_data["text"] = train_data["text"].apply(lambda x: x.lower())
    train_images_texts = dict(zip(train_data.id, train_data.text))
    train_images_labels = dict(zip(train_data.id, train_data[task_mode]))
    train_images_embs = {k: all_images_embs[k] for k in list(train_images_texts.keys())}

    with open(val_file_path, encoding="utf-8") as j_file:
        val_data = pd.read_json(j_file, lines=True)
    val_data["text"] = val_data["text"].apply(lambda x: x.lower())
    val_images_texts = dict(zip(val_data.id, val_data.text))
    val_images_labels = dict(zip(val_data.id, val_data[task_mode]))
    val_images_embs = {k: all_images_embs[k] for k in list(val_images_texts.keys())}

    # Define BERT model
    model_name = "bert-base-uncased"
    # Construct a BERT tokenizer based on WordPiece
    bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # Set max_len to the maximum length of the training data
    max_len = max([len(bert_tokenizer.encode(s)) for s in list(train_images_texts.values())])
    print("The maximum sentence length in training based on BERT BPEs is", max_len)

    if max_len > 512:
        print("Reducing max length to 512")
        max_len = 512

    # Tokenize and encode sentences in each set
    x_train = bert_tokenizer.__call__(
        list(train_images_texts.values()),
        max_length=max_len,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    x_val = bert_tokenizer.__call__(
        list(val_images_texts.values()),
        max_length=max_len,
        padding=True,
        truncation=True,
        return_attention_mask=True,
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

    train_images = torch.tensor(np.array(list(train_images_embs.values())))
    val_images = torch.tensor(np.array(list(val_images_embs.values())))

    batch_size = 32

    # Create a dataloader for each set
    print("Creating dataloaders...")
    # Train
    train_dataset = TensorDataset(x_train['input_ids'], x_train['attention_mask'], train_y, train_images)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    # Val
    val_dataset = TensorDataset(x_val['input_ids'], x_val['attention_mask'], val_y, val_images)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    print("Creating model...")
    if task_mode != "gold_hate":
        num_labels = len(labels_list)
    else:
        num_labels = 1

    bert = BertModel.from_pretrained(model_name, output_attentions=True)
    model = ImgBERT(bert, num_labels=num_labels)

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
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(results_path, "imgbert_" + task_mode + ".bin")
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
    img_emb_path = args.image_embeddings_path

    main(task, train_file_path, val_file_path, label_file_path, img_emb_path, results_path)