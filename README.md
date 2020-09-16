
# Neural Graph Fingerprints in tf.keras (tf >= 2.0)

This package contains an implementation of two tf.keras layers (in tf.keras from tensorflow >= 2.0) which correspond to the operators necessary for computing neural fingerprints for molecular graphs.
The method is based on the work of Duvenaud et. al. A technical description of the algorithm can be found in the original paper:

Title: Convolutional Networks on Graphs for Learning Molecular Fingerprints (by Duvenaud et. al.)
Link: https://arxiv.org/abs/1509.09292

The implementation in this package is essentially an upgrade to tf.keras with tensorflow >= 2.0 of the keiser-lab implementation of the Duvenaud algorithm. The original implementation can be found at:

		https://github.com/keiserlab/keras-neural-graph-fingerprint.

The goal was to create a simple, transparent and accessible version of Duvenauds algorithm which runs smoothly on tf.keras with tensorflow >= 2.0. Large parts of both neural fingerprint layer implementations were rewritten by the author using different, more explicit methods which can be readily modified. 

The script tf_keras_layers_neural_graph_convolutions offers the following two tf.keras layer classes (child classes of tf.keras.layers.layer):

- NeuralFingerprintHidden: Takes the place of the operation of the hidden graph convolution in Duvenauds algorithm (see matrices H in paper).

- NeuralFingerprintOutput: Takes the place of the operation of the readout convolution in Duvenauds algorithm (see matrices W in paper).

The package contains an example with a simple water solubility prediction task to illustrate how to use the layers to construct a convolutional neural graph fingerprint network.

The scripts auxiliary_functions_atom_bond_features, auxiliary_functions_graph_tensorion and auxiliary_functions_neural_graph_convolutions contain auxiliary functions which were taken from the keiser-lab implementation of the Duvenaud algorithm (https://github.com/keiserlab/keras-neural-graph-fingerprint). This implementation operates within the graph tensorisation framework which was offered in the keiser-lab implementation. We thus copied parts of the readme of the keiser-lab implementation which still apply to this new implementation below.


# Molecule Representation

## Atom, bond and edge tensors

This codebase uses tensor matrices to represent molecules. Each molecule is described by a combination of the following four tensors:

- atom matrix, size: (max_atoms, num_atom_features) This matrix defines the atom features. Each column in the atom matrix represents the feature vector for the atom at the index of that column.

- edge matrix, size: (max_atoms, max_degree) This matrix defines the connectivity between atoms. Each column in the edge matrix represent the neighbours of an atom. The neighbours are encoded by an integer representing the index of their feature vector in the atom matrix. As atoms can have a variable number of neighbours, not all rows will have a neighbour index defined. These entries are filled with the masking value of -1 (this explicit edge matrix masking value is important for the layers to work).

- bond tensor size: (max_atoms, max_degree, num_bond_features) This matrix defines the atom features. The first two dimensions of this tensor represent the bonds defined in the edge tensor. The column in the bond tensor at the position of the bond index in the edge tensor defines the features of that bond. Bonds that are unused are masked with 0 vectors.
    
- atoms existence, size (max_atoms,). This binary 1d array indicates the number of atoms of a molecule. If a molecule has (say) 2 atoms, the array is (1,1,0,...,0).

## Batch representations

This codes deals with molecules in batches. An extra dimension is added to all of the four tensors at the first index. Their respective sizes become:

- atom matrix, size: (num_molecules, max_atoms, num_atom_features)
- edge matrix, size: (num_molecules, max_atoms, max_degree)
- bond tensor size: (num_molecules, max_atoms, max_degree, num_bond_features)
- atoms existence size: (num_molecules, max_atoms)

As molecules have different numbers of atoms, max_atoms needs to be defined for the entire dataset. Unused atom columns are masked by 0 vectors.

# NeuralFingerprint layers

The relevant tf.keras layers are defined in tf_keras_layers_neural_graph_convolutions.

- NeuralFingerprintHidden takes a set of molecules (represented by [atoms, bonds, edges, atoms_existence]), and returns the convolved feature vectors of the higher layers by applying a  neural network with 1 layers. Only the feature vectors change at each iteration, so for higher layers only the atom tensor needs to be replaced by the convolved output of the previous NeuralFingerprintHidden.

- NeuralFingerprintOutput takes a set of molecules (represented by [atoms, bonds, edges, atoms_existence]), and returns the fingerprint output for that layer by applying a 1-layer neural network with softmax output. According to the original paper, the fingerprints of all layers need to be summed. But these are neural nets, so feel free to play around with the architectures!

# Why the atoms_existence tensor?

The additional input tensor "atoms_existence" was added to the framework to account for a subtle theoretical gap in the keiser-lab implementations: 
atoms associated with a zero feature vector (which can theoretically happen after at least one convolution) AND with degree 0 can still exist and can thus not be ignored. As an example imagine a single carbon atom as input molecule whose atom feature vector gets mapped to zero in the first convolution. The previous implementations would from this moment on treat the carbon atom as nonexistent and thus the molecule as empty.

# Dependencies

- tensorflow >= 2.0
- rdkit 2020.03.3.0 
- numpy 1.19.1 
- pandas 1.1.1.

# Acknowledgements

The implementation of both neural fingerprint layers is inspired by (but different from) the two tf.keras graph convolutional layer implementations which can be found in the keiser-lab implementation:

    - https://github.com/keiserlab/keras-neural-graph-fingerprint/blob/master/NGF/layers.py.

The scripts auxiliary_functions_atom_bond_features, auxiliary_functions_graph_tensorion and auxiliary_functions_neural_graph_convolutions contain useful auxiliary functions which were also taken from the keiser-lab implementation with only minor changes.
