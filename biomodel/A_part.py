import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from Bio import PDB
from encoder.transformerEncoder import SelfAttention,TransformerEncoder
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



class ProteinTransformer(nn.Cell):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(ProteinTransformer, self).__init__()
        self.encoders = nn.SequentialCell([TransformerEncoder(embed_dim, num_heads) for _ in range(num_layers)])

    def construct(self, x):
        return self.encoders(x)
