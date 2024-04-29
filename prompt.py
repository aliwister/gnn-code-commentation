import csv
import numpy as np
import torch
from argparse import ArgumentParser
import pandas as pd
from torch_geometric.loader import DataLoader

from sklearn.cluster import KMeans
from math import ceil
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from transformers import GPTJForCausalLM, AutoTokenizer
from model.gnn_encoder import GNNModel
import evaluate
from util.graph import create_graph
from util.prompt import create_incontext_prompt, get_answer
from train import encode
import pdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def process_string_to_array(string):
    stripped_string = string.strip('[]')
    array = np.array([float(num) for num in stripped_string.split()])
    return array

def load_training_set_with_logits(dataset_train_raw, dataset_train_logits):
    df_train_raw = pd.read_csv(dataset_train_raw)
    dtype_dict = {'idx': int, 'rep': str}
    df_train_processed = pd.read_csv(dataset_train_logits, dtype=dtype_dict)
    df_train_processed['data'] = df_train_processed['rep'].apply(process_string_to_array)
    name_to_question_text = df_train_raw['question_text'].to_dict()
    name_to_program_text = df_train_raw['decomposition'].to_dict()
    df_train_processed['question_text'] = df_train_processed['idx'].map(name_to_question_text)
    df_train_processed['decomposition'] = df_train_processed['idx'].map(name_to_program_text)
    return df_train_processed

def cluster_data(df, n_clusters):
    kmeans = KMeans(n_clusters=45)
    labels_train = kmeans.fit_predict(list(df['data'].values))
    return kmeans, labels_train

def get_samples(df, cluster, num):
    fdf = df
    if cluster > -1:
        fdf = df[df['label'] == cluster]
    r = fdf.sample(n=5)
    #pdb.set_trace()
    #if cluster > -1:
    #print(r)
    if (num ==2):
        return r.iloc[0]['decomposition'], r.iloc[0]['question_text'], r.iloc[1]['decomposition'], r.iloc[1]['question_text']
    if (num==1):
        return r.iloc[0]['decomposition'], r.iloc[0]['question_text']

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_train_raw', type=str, default='/home/ali.lawati/gnn-incontext/data/Break-dataset/logical-forms/dev.csv')
    parser.add_argument('--dataset_train_logits', type=str, default='/home/ali.lawati/gnn-incontext/logits10.csv')
    parser.add_argument('--dataset_test', type=str, default='/home/ali.lawati/gnn-incontext/data/Break-dataset/logical-forms/simple-test.csv')
    parser.add_argument('--saved_gnn_model', type=str, default="gnn10.pt") 
    parser.add_argument('--lang_model', type=str, default="EleutherAI/gpt-j-6B") 
    args = parser.parse_args()


    model = GNNModel()
    model.load_state_dict(torch.load(args.saved_gnn_model))
    model.to(device)
    lang_model = GPTJForCausalLM.from_pretrained(args.lang_model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map = 'auto').to(torch.float16).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.lang_model)
    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.padding_side = 'left'

    df_train = load_training_set_with_logits(args.dataset_train_raw, args.dataset_train_logits)

    df_test = pd.read_csv(args.dataset_test)
    data = DataLoader(df_test[:].apply(lambda x: create_graph(x['program'], tokenizer, x.name), axis=1).tolist(), batch_size=8, shuffle=False)
    test_idx, test_reps = encode(model, data, lang_model, device)

    kmeans, df_train['label'] = cluster_data(df_train, 45)
    #pdb.set_trace()

    test_reps = np.array(test_reps).astype(np.float64)
    labels_test = kmeans.predict(test_reps)

    predictions1, predictions2, predictions3, references = [], [], [], []
    prompts1, prompts2 = [], []

    batch_size = 16
    for i in range(0, len(test_idx)):
        prog = df_test.iloc[i]['decomposition']
        print(prog, df_test.iloc[i]['question_text'])
        p_args1 = get_samples(df_train, -1, 2) + (prog,)
        p_args2 = get_samples(df_train, labels_test[i], 2) + (prog,)
        prompt1 = create_incontext_prompt(*p_args1)
        prompt2 = create_incontext_prompt(*p_args2)
        prompts1.append(prompt1)
        prompts2.append(prompt2)
        references.append(df_test.iloc[i]['question_text'])

        #prompts1.extend(batch_prompt1)
        #prompts2.extend(batch_prompt2)
        #prompt2 = create_incontext_prompt(get_samples(df_train, labels_test[i]))
        #response1 = get_answer(batch_prompt1, lang_model, tokenizer, device)
        #response2 = get_answer(batch_prompt2, lang_model, tokenizer, device)
        
        #predictions1.extend(response1)
        #predictions2.extend(response2)
        #predictions3.append(response3)

    data_dict = {
        'prompt1': np.squeeze(prompts1),      # First array as the 'ID' column
        'prompt2': np.squeeze(prompts2),    # Second array as the 'Name' column
        'ref': references      # Third array as the 'Age' column
    }
    df = pd.DataFrame(data_dict)
    df.to_csv('./test/prompts', index=False)

    """with open("prompts-answer10.csv", 'w') as f:
        writer = csv.writer(f)
            # Write header (optional)
        writer.writerow(['ref', 'pred1', 'pred2'])
        
        # Write each row with one element from each list
        for item1, item2, item3 in zip(references, np.squeeze(predictions1), np.squeeze(predictions2)):
            writer.writerow([item1, item2, item3])    
    #pdb.set_trace()
    bleu_metric = evaluate.load("bleu")
    result = bleu_metric.compute(predictions=np.squeeze(predictions1), references=references)
    print("BLEU score:", result["bleu"])

    result = bleu_metric.compute(predictions=np.squeeze(predictions2), references=references)
    print("BLEU score:", result["bleu"])
    """
    #result = bleu_metric.compute(predictions=np.squeeze(predictions3), references=references)
    #print("BLEU score:", result["bleu"])

    """cluster = labels_test[i]
    rep = test_reps[i]
    df_train_filter = df_train[df_train['label']==1]
    # Calculate cosine similarity between the given value and each array in the 'data' column
    similarities = cosine_similarity(df_train_filter['data'].tolist(), rep.reshape(1, -1))

    # Convert similarities to a 1D array for easier processing
    similarities = similarities.ravel()

    # Find the top three highest cosine similarities and their indices
    top_3_indices = np.argsort(similarities)[-3:]  # Get indices of the highest 3 values
    top_3_similarities = similarities[top_3_indices]"""




