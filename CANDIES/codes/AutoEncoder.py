import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.distributions import Poisson
from ZINB_encoder import *

class RNAEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super(RNAEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class RNADecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(RNADecoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ProteinEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super(ProteinEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class ProteinDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ProteinDecoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ATACEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super(ATACEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class ATACDecoder(nn.Module):
    def __init__(self, d_in, d_hid, d_out, zero_inflaten=True, basic_module="Linear"):
        super().__init__()
        self.zero_inflaten = zero_inflaten
        self.basic_module = basic_module

        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_out)

        self._dec_mean = nn.Sequential(
            nn.Linear(d_out, d_out),
            MeanAct()
        )
        self._dec_disp = nn.Sequential(
            nn.Linear(d_out, d_out),
            DispAct()
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        recon = self.fc2(x)

        _mean = self._dec_mean(recon)
        _disp = self._dec_disp(recon)

        return recon, _mean, _disp

class ZIPDecoder(nn.Module):
    def __init__(self, d_in, d_hid, d_out, zero_inflaten=True, is_recons=False, basic_module="Linear"):
        super().__init__()
        self.zero_inflaten = zero_inflaten
        self.is_recons = is_recons
        self.basic_module = basic_module

        self.conv1 = GCNConv(d_in, d_hid)
        self.conv2 = GCNConv(d_hid, d_out)

        self.pi_output = nn.Linear(d_hid[-1], d_out) if zero_inflaten else None
        self.rho_output = nn.Linear(d_hid[-1], d_out)
        self.expr_output = nn.Linear(d_hid[-1], d_out) if is_recons else None
        self.peak_bias = nn.Parameter(torch.randn(1, d_out))

    def forward(self, x, edge_index=None):
        x = F.relu(self.conv1(x,edge_index))
        x = self.conv2(x,edge_index)

        pi = F.sigmoid(self.pi_output(x)) if self.zero_inflaten else None
        rho = self.rho_output(x)
        omega = F.softmax(rho + self.peak_bias, dim=-1)
        recons_expr = self.expr_output(x) if self.is_recons else None

        return pi, None, omega, recons_expr


def ZIPLoss(x, pi, omega, scale_factor=1.0, ridge_lambda=0.0):
    eps = 1e-10
    lamb = omega * scale_factor
    if pi is not None:
        po_case = -Poisson(lamb).log_prob(x) - torch.log(1.0 - pi + eps)
        zero_po = torch.exp(-lamb)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_po) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, po_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        return torch.mean(result)

    else:
        result = -Poisson(lamb).log_prob(x)
        return torch.mean(result)

class NBLoss(nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, x, mean, disp, scale_factor=1.0, ridge_lambda=0.0, device=None):
        eps = 1e-10
        scale_factor = torch.Tensor([1.0]).to(device)
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        result = torch.mean(nb_final)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(mean)
            result += ridge
        # result = torch.mean(result)
        return result


def train_model(adata_omics1, adata_omics2, adj_spatial_omics1, adj_spatial_omics2, epochs=1000, rna_latent_dim=64, protein_latent_dim=64):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    rna_in_channels = adata_omics1.shape[1]
    rna_hidden_channels = 128
    rna_output_dim = rna_in_channels

    protein_in_channels = adata_omics2.shape[1]
    protein_hidden_channels = 128
    protein_output_dim = protein_in_channels

    rna_encoder = RNAEncoder(rna_in_channels, rna_hidden_channels, rna_latent_dim).to(device)
    rna_decoder = RNADecoder(rna_latent_dim, rna_hidden_channels, rna_output_dim).to(device)

    protein_encoder = ProteinEncoder(protein_in_channels, protein_hidden_channels, protein_latent_dim).to(device)
    protein_decoder = ProteinDecoder(protein_latent_dim, protein_hidden_channels, protein_output_dim).to(device)

    rna_features = torch.tensor(adata_omics1.X, dtype=torch.float).to(device)
    edge_index1 = adj_spatial_omics1

    protein_features = torch.tensor(adata_omics2.X, dtype=torch.float).to(device)
    edge_index2 = adj_spatial_omics2

    optimizer_rna = optim.Adam(list(rna_encoder.parameters()) + list(rna_decoder.parameters()), lr=0.001)
    optimizer_protein = optim.Adam(list(protein_encoder.parameters()) + list(protein_decoder.parameters()), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        rna_encoder.train()
        rna_decoder.train()

        rna_latent = rna_encoder(rna_features, edge_index1)
        rna_reconstructed = rna_decoder(rna_latent)

        loss_rna = criterion(rna_reconstructed, rna_features)

        optimizer_rna.zero_grad()
        loss_rna.backward()
        optimizer_rna.step()

        protein_encoder.train()
        protein_decoder.train()

        protein_latent = protein_encoder(protein_features, edge_index2)
        protein_reconstructed = protein_decoder(protein_latent)

        loss_protein = criterion(protein_reconstructed, protein_features)

        optimizer_protein.zero_grad()
        loss_protein.backward()
        optimizer_protein.step()

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Loss RNA: {loss_rna.item():.4f}, Loss Protein: {loss_protein.item():.4f}")

    torch.save(rna_encoder.state_dict(), 'rna_encoder_model.pth')
    torch.save(rna_decoder.state_dict(), 'rna_decoder_model.pth')
    torch.save(protein_encoder.state_dict(), 'protein_encoder_model.pth')
    torch.save(protein_decoder.state_dict(), 'protein_decoder_model.pth')
    print("Training complete!")

    rna_encoder.eval()
    protein_encoder.eval()

    rna_latent = rna_encoder(rna_features, edge_index1)
    protein_latent = protein_encoder(protein_features, edge_index2)

    adata_omics1.obsm['emb_latent_omics1'] = rna_latent.cpu().detach().numpy()
    adata_omics2.obsm['emb_latent_omics2'] = protein_latent.cpu().detach().numpy()

    print("Latent representations have been successfully added to adata_omics.obsm!")

def train_atac(adata_omics2, adj_spatial_omics2, hidden_channels = 128,epochs=1000, atac_latent_dim=64):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    atac_in_channels = adata_omics2.obsm['X_lsi'].shape[1]
    atac_output_dim = atac_in_channels

    atac_encoder = ATACEncoder(atac_in_channels, hidden_channels, atac_latent_dim).to(device)
    atac_decoder = ATACDecoder(atac_latent_dim, hidden_channels, atac_output_dim).to(device)

    atac_features = torch.tensor(adata_omics2.obsm['X_lsi'], dtype=torch.float).to(device)
    edge_index2 = adj_spatial_omics2

    optimizer_atac = optim.Adam(list(atac_encoder.parameters()) + list(atac_decoder.parameters()), lr=0.001)
    criterion = nn.MSELoss()
    loss_nb = NBLoss()

    for epoch in range(epochs):
        atac_encoder.train()
        atac_decoder.train()

        atac_latent = atac_encoder(atac_features, edge_index2)
        recons_expr, _mean, _disp = atac_decoder(atac_latent)

        # Loss calculations with checks
        loss_recon = criterion(recons_expr, atac_features)
        loss_2 = loss_nb(atac_features, _mean, _disp, device=device)

        # Log NaN or Inf values for debugging
        if torch.isnan(loss_2).any() or torch.isinf(loss_2).any():
            print(f"NaN detected in loss_2 at epoch {epoch}")

        loss_atac = loss_recon + loss_2

        optimizer_atac.zero_grad()
        loss_atac.backward()
        optimizer_atac.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss atac: {loss_atac.item():.4f}")

    print("Training complete!")

    # Save embeddings
    atac_encoder.eval()
    atac_latent = atac_encoder(atac_features, edge_index2)
    adata_omics2.obsm['emb_latent_omics2'] = atac_latent.cpu().detach().numpy()
    print("Latent representations have been successfully added to adata_omics.obsm!")
