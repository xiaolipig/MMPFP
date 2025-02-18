import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from Bio import PDB
from backbones.RepVGG import RepVGGBlock,RepVGG

class PDBProcessor:
    def __init__(self):
        self.aa_dict = {'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4, 'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8,
                        'LEU': 9, 'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14, 'SER': 15, 'THR': 16,
                        'VAL': 17, 'TRP': 18, 'TYR': 19}
        self.esm_embed_dim = 128

    def parse_pdb(self, pdb_file):
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        seq = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if PDB.is_aa(residue):
                        seq.append(residue.get_resname())
        return seq

    def one_hot_encode(self, seq):
        encoding = np.zeros((len(seq), len(self.aa_dict)), dtype=np.float32)
        for i, aa in enumerate(seq):
            if aa in self.aa_dict:
                encoding[i, self.aa_dict[aa]] = 1.0
        return encoding

    def esm_encode(self, seq):
        np.random.seed(42)
        return np.random.rand(len(seq), self.esm_embed_dim).astype(np.float32)

    def preprocess(self, pdb_file):
        seq = self.parse_pdb(pdb_file)
        one_hot_features = self.one_hot_encode(seq)
        esm_features = self.esm_encode(seq)
        fused_features = np.concatenate((one_hot_features, esm_features), axis=1)
        return ms.Tensor(fused_features, dtype=ms.float32)



class PDBProcessor:
    def __init__(self, distance_threshold=8.0):
        self.distance_threshold = distance_threshold

    def parse_pdb(self, pdb_file):
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        coordinates = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    if PDB.is_aa(residue):
                        ca = residue['CA']
                        coordinates.append(ca.get_coord())

        return np.array(coordinates)

    def pairwise_distance(self, coords):
        n = coords.shape[0]
        distance_matrix = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix

    def compute_contact_map(self, pdb_file):
        coords = self.parse_pdb(pdb_file)
        if len(coords) == 0:
            raise ValueError("error")

        dist_matrix = self.pairwise_distance(coords)
        contact_map = (dist_matrix < self.distance_threshold).astype(np.float32)  #  Contact Map
        return ms.Tensor(contact_map, dtype=ms.float32)



class GCNLayer(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.weight = ms.Parameter(ms.Tensor(np.random.randn(in_channels, out_channels).astype(np.float32)))
        self.bias = ms.Parameter(ms.Tensor(np.zeros((1, out_channels)).astype(np.float32)))

    def construct(self, x, adj):
        # 计算 D^(-1/2) * A * D^(-1/2)
        degree = ops.ReduceSum()(adj, axis=1) + 1e-6
        d_inv_sqrt = ops.Pow()(degree, -0.5)
        d_inv_sqrt = ops.ExpandDims()(d_inv_sqrt, 1)
        normalized_adj = adj * d_inv_sqrt * ops.Transpose()(d_inv_sqrt, (1, 0))

        support = ops.MatMul()(x, self.weight)
        output = ops.MatMul()(normalized_adj, support)
        return output + self.bias



class GCN(nn.Cell):
    def __init__(self, in_channels, hidden_dim, out_dim, num_layers=2):
        super(GCN, self).__init__()
        self.layers = nn.CellList([GCNLayer(in_channels if i == 0 else hidden_dim,
                                             hidden_dim if i < num_layers - 1 else out_dim) for i in range(num_layers)])
        self.relu = nn.ReLU()

    def construct(self, x, adj):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x, adj))
        x = self.layers[-1](x, adj)
        return x.mean(axis=1)


# part B fusion
class FeatureFusion(nn.Cell):
    def __init__(self, esm_dim, one_hot_dim, gcn_in_dim, gcn_hidden_dim, gcn_out_dim, num_classes):
        super(FeatureFusion, self).__init__()

        # RepVGG ESM + One-hot
        repvgg_input_dim = esm_dim + one_hot_dim
        self.repvgg = RepVGG(in_channels=repvgg_input_dim, num_classes=num_classes)


        self.gcn = GCN(in_channels=gcn_in_dim, hidden_dim=gcn_hidden_dim, out_dim=gcn_out_dim)

        self.fc = nn.Dense(num_classes + gcn_out_dim, num_classes)

    def construct(self, esm_feature, one_hot_feature, contact_map):


        fusion_feature = ops.Concat(1)([esm_feature, one_hot_feature])
        A = self.repvgg(fusion_feature)


        B = self.gcn(contact_map, contact_map)


        combined_feature = ops.Concat(1)([A, B])
        output = self.fc(combined_feature)
        return output



