from Bio import PDB
import torch
import torch.nn.functional as F
import esm

def extract_sequence_from_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    seq = []
    ppb = PDB.PPBuilder()
    for pp in ppb.build_peptides(structure):
        seq.append(str(pp.get_sequence()))
    return "".join(seq)

def one_hot_encode(sequence):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
    indices = [aa_to_idx[aa] for aa in sequence if aa in aa_to_idx]
    indices = torch.tensor(indices, dtype=torch.long)
    return F.one_hot(indices, num_classes=len(amino_acids)).float()

def esm1b_encode(sequence):
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm1b_t33_650M_UR50S")
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    data = [("protein1", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    return token_representations[0, 1:len(sequence) + 1].mean(0)

if __name__ == "__main__":
    sequence = extract_sequence_from_pdb('1mbn.pdb')
    one_hot = one_hot_encode(sequence)
    print("One-Hot :", one_hot.shape)
    esm_embedding = esm1b_encode(sequence)
    print("ESM-1b :", esm_embedding.shape)
