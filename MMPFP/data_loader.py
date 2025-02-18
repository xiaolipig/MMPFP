import os
import numpy as np
from mindspore.dataset import Dataset
from mindspore import Tensor
from biopython import PDB


class ProteinDataset(Dataset):
    def __init__(self, txt_file, N_PDB_FILES):

        self.file_paths = self._read_txt(txt_file)
        self.N_PDB_FILES = N_PDB_FILES

    def _read_txt(self, txt_file):

        with open(txt_file, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    def _parse_pdb(self, pdb_file):

        parser = PDB.PPBuilder()
        structure = parser.get_structure('protein', pdb_file)

        sequence = ''
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() != "HOH":  # exclude h2o
                        sequence += residue.get_resname()

        gcn_feature = np.random.rand(256)
        repvgg_feature = np.random.rand(256)

        label = np.random.randint(0, 3)

        return sequence, gcn_feature, repvgg_feature, label

    def __getitem__(self, index):

        pdb_file = self.file_paths[index % len(self.file_paths)]
        protein, gcn, repvgg, label = self._parse_pdb(pdb_file)

        return Tensor(protein), Tensor(gcn, dtype=np.float32), Tensor(repvgg, dtype=np.float32), Tensor(label,
                                                                                                        dtype=np.int32)

    def __len__(self):

        return len(self.file_paths)

