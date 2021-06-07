import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from scipy.special import expit
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


parser = argparse.ArgumentParser(description="Loads a trained checkpoint and predicts the probabilities for each of "
                                             "each task")
parser.add_argument("--train_path", help="Path to json file with training data")
parser.add_argument("--test_path", help="Path to json file with test data for prediction")
parser.add_argument("--pc_labels_path", help="Path to csv file that contains the protected category labels")
parser.add_argument("--attack_labels_path", help="Path to csv file that contains the attack type labels")
parser.add_argument("--results_path", help="Path to folder where the model checkpoints are saved")


def main(train_file_path, test_file_path, pc_labels_path, attack_labels_path, results_path):
    print("Reading and encoding data...")
    # Read tags
    tags_pc_df = pd.read_csv(pc_labels_path, header=None)
    tags_pc_list = tags_pc_df[0].to_list()

    tags_attack_df = pd.read_csv(attack_labels_path, header=None)
    tags_attack_list = tags_attack_df[0].to_list()

    d_ids = [0]
    device = f"cuda:{d_ids[0]}"

    # Read train
    with open(train_file_path, encoding="utf-8") as j_file:
        train_data = pd.read_json(j_file, lines=True)
    train_data["text"] = train_data["text"].apply(lambda x: x.lower())
    train_cases_texts = dict(zip(train_data["id"], train_data["text"]))

    # Read test
    with open(test_file_path, encoding="utf-8") as j_file:
        test_data = pd.read_json(j_file, lines=True)
    test_data["text"] = test_data["text"].apply(lambda x: x.lower())
    test_cases_texts = dict(zip(test_data["id"], test_data["text"]))

    # Tokenizer
    # Construct a BERT tokenizer based on WordPiece
    bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Set max_len to the maximum length of the training data
    max_len = max([len(bert_tokenizer.encode(s)) for s in list(train_cases_texts.values())])
    print("The maximum sentence length in training based on BERT BPEs is", max_len)

    if max_len > 512:
        print("Reducing max length to 512")
        max_len = 512

    # Encode data
    x_test = bert_tokenizer.__call__(
        list(test_cases_texts.values()),
        max_length = max_len,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    test_data = TensorDataset(x_test['input_ids'], x_test['attention_mask'])
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=1)

    print("Loading models...")
    # Load model for hate
    model_hate_state_dict = torch.load(os.path.join(results_path, "txtbert_gold_hate.bin"),
                                  map_location="cpu")
    model_hate = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1,
                                                            state_dict=model_hate_state_dict,
                                                            output_attentions=True)
    model_hate.to(device)

    # Load model for protected category
    model_pc_state_dict = torch.load(os.path.join(results_path, "txtbert_gold_pc.bin"),
                                  map_location="cpu")
    model_pc = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(tags_pc_list),
                                                            state_dict=model_pc_state_dict,
                                                            output_attentions=True)
    model_pc.to(device)


    # Load model for attack type
    model_attack_state_dict = torch.load(os.path.join(results_path, "txtbert_gold_attack.bin"),
                                  map_location="cpu")
    model_attack = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(tags_attack_list),
                                                            state_dict=model_attack_state_dict,
                                                            output_attentions=True)
    model_attack.to(device)

    print("Inference")
    # Inference
    model_pc.eval()
    model_attack.eval()
    model_hate.eval()
    test_results = []

    for case_num, batch in tqdm(enumerate(test_dataloader)):
        batch = [t.to(device) for t in batch]
        sent_id, mask = batch
        with torch.no_grad():
            # Get output probs from both models
            hate_outputs = model_hate(sent_id, attention_mask=mask)
            pc_outputs = model_pc(sent_id, attention_mask=mask)
            attack_outputs = model_attack(sent_id, attention_mask=mask)
            hate_probs = expit(hate_outputs.logits.detach().cpu().numpy()[0])
            pc_probs = expit(pc_outputs.logits.detach().cpu().numpy()[0])
            attack_probs = expit(attack_outputs.logits.detach().cpu().numpy()[0])
            # Save probs for each label
            pred_hate={}
            pred_pc = {}
            pred_attack = {}
            pred_hate["hateful"] = hate_probs[0]
            pred_hate["not_hateful"] = 1 - hate_probs[0]
            for i, pc in enumerate(tags_pc_list):
                pred_pc[pc] = pc_probs[i]
            for i, attack in enumerate(tags_attack_list):
                pred_attack[attack] = attack_probs[i]
            max_pc = max(list(pred_pc.values()))
            max_attack = max(list(pred_attack.values()))
            pred_pc["pc_empty"] = 1 - max_pc
            pred_attack["attack_empty"] = 1 - max_attack
            test_results.append({"id": list(test_cases_texts.keys())[case_num], "set_name": "test",
                                 "pred_hate": pred_hate,
                                 "pred_pc": pred_pc, "pred_attack": pred_attack})

    res_df = pd.DataFrame(test_results, columns=["id", "set_name", "pred_hate", "pred_pc", "pred_attack"])
    res_df.to_json(os.path.join(results_path, "txtbert_results.json"), orient="records", lines=True)
    print("Saved results")

if __name__ == '__main__':
    args = parser.parse_args()
    train_file_path = args.train_path
    test_file_path = args.test_path
    pc_labels_path = args.pc_labels_path
    attack_labels_path = args.attack_labels_path
    results_path = args.results_path

    main(train_file_path, test_file_path, pc_labels_path, attack_labels_path, results_path)