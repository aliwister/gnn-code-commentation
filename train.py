from argparse import ArgumentParser
import pandas as pd

from torch_geometric.loader import DataLoader

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import torch

from math import ceil
import csv

from tqdm import tqdm
from transformers import GPTJForCausalLM, AutoTokenizer
from model.gnn_encoder import GNNModel
from util.graph import create_graph

#os.environ["DEBUSSY"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

# Training loop with K-means loss
def train(model, data_loader, optimizer): #, loss_fn):
    model.train()
    total_loss = 0
    for _,batch in tqdm(enumerate(data_loader, 0), unit="batch", total=len(data_loader)):
        batch.to(device)
        out, loss = model.train_step(batch, lang_model)
        total_loss = total_loss + loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 
    return total_loss


def encode(model, data_loader, lang_model, device):
    # Extract pooled graph representations
    pooled_representations = []
    pooled_indexes = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
            batch.to(device)
            out, loss = model.train_step(batch, lang_model)
            pooled_representations.append(out)
            pooled_indexes.append(batch.idx)

    #pdb.set_trace()
    pooled_representations = torch.cat(pooled_representations, dim=0).cpu().numpy()
    pooled_indexes = torch.cat(pooled_indexes, dim=0).cpu().numpy()
    return pooled_indexes, pooled_representations

# Validate
def eval(model, data_loader, lang_model, device, output_file, fig_file):
    # Extract pooled graph representations
    model.eval()
    pooled_representations = []
    pooled_indexes = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
            batch.to(device)
            out, loss = model.train_step(batch, lang_model)
            pooled_representations.append(out)
            pooled_indexes.append(batch.idx)

    #pdb.set_trace()
    pooled_representations = torch.cat(pooled_representations, dim=0).cpu().numpy()
    pooled_indexes = torch.cat(pooled_indexes, dim=0).cpu().numpy()
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
            # Write header (optional)
        writer.writerow(['idx', str('rep')])
        
        # Write each row with one element from each list
        for item1, item2 in zip(pooled_indexes, pooled_representations):
            writer.writerow([item1, item2])

    silhouette_scores = []
    nrange = range(5,100)
    # Cluster the pooled graph representations using KMeans
    for i in nrange:
        kmeans = KMeans(n_clusters=i)
        labels = kmeans.fit_predict(pooled_representations)
        silhouette = silhouette_score(pooled_representations, labels)
        silhouette_scores.append(silhouette)

        print(f"Silhouette Score: {silhouette:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(nrange, silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig(f"{fig_file}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/home/ali.lawati/gnn-incontext/data/Break-dataset/logical-forms/dev.csv')
    parser.add_argument('--model', type=str, default="EleutherAI/gpt-j-6B") 
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--evaluate_only', type=bool, default=1)
    parser.add_argument('--lr', type=float, default=.001) # learning-rate
    parser.add_argument('--gnn_model_file', type=str, default="gnn-fri.pt") 
    parser.add_argument('--dataset_train_logits', type=str, default='logits-fri.csv')
    args = parser.parse_args()

    gnn_model_file = f"{args.epochs}_{args.gnn_model_file}"
    dataset_train_logits = f"{args.epochs}_{args.dataset_train_logits}"

    lang_model = GPTJForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(torch.float16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Define a list of graphs
    df = pd.read_csv(args.dataset)
    transformed_list = df[:].apply(lambda x: create_graph(x['program'], tokenizer, x.name), axis=1).tolist()
    data = DataLoader(transformed_list, batch_size=8, shuffle=False)

    #if not args.evaluate_only:
        # Initialize model and optimizer
    model = GNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    for param in lang_model.parameters():
        param.requires_grad = False

    for epoch in range(args.epochs):
        print('Training...')
        loss = train(model, data, optimizer) #, kmeans_loss_fn)
        print(f'Epoch: {epoch:02d}, '
            f'Loss: {loss:.4f}, ')
    torch.save(model.state_dict(), gnn_model_file)

    model = GNNModel()
    model.load_state_dict(torch.load(gnn_model_file))
    model.to(device).eval()
    eval(model, data, lang_model, device, dataset_train_logits, f"{args.epochs}_sil_fri.png")