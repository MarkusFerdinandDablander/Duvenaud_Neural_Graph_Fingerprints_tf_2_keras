''' 
PART_4

This part (i.e. PART_4 of this script) contains an implementation of two new tf.keras layers (in tf.keras from tensorflow >= 2.0) which correspond to the operators necessary for computing neural fingerprints for molecular graphs.
The method is based on the work of Duvenaud et. al. A technical description of the algorithm can be found in the original paper:

Title: Convolutional Networks on Graphs for Learning Molecular Fingerprints (by Duvenaud et. al.)
Link: https://arxiv.org/abs/1509.09292

This script offers the following two tf.keras layer classes (child classes of tf.keras.layers.layer):

- NeuralFingerprintHidden: Corresponds to the operation of the hidden graph convolution in Duvenauds algorithm (see matrices H in paper)
- NeuralFingerprintOutput: Corresponds to the operation of the readout convolution in Duvenauds algorithm (see matrices W in paper)

The code was written to work with tf.keras on the basis of tensorflow 2.2.0, rdkit 2020.03.3.0, numpy 1.19.1 and pandas 1.1.1.

Both tf.keras layers were implemented by Markus Ferdinand Dablander, DPhil (= PhD) student at Mathematical Institute, Oxford University, August 2020.

The implementation of both layers below is inspired by the two tf.keras graph convolutional layer implementations which can be found in the keiser-lab implementation:

	- https://github.com/keiserlab/keras-neural-graph-fingerprint/blob/master/NGF/layers.py

However, large parts of the implementation were completely rewritten by the author using different, more explicit methods which can be readily understood and modified. 
The goal was to create a simple, transparent and accessible version of Duvenauds algorithm which runs smoothly on tf.keras with tensorflow >= 2.0. 

An addition input tensor "atoms_existence" was added by the author to the framework to account for a subtle gap in previous implementations: 
atoms associated with a zero feature vector (which can theoretically happen after at least one convolution) AND with degree 0 can still exist and can thus not be ignored. 
As an example imagine a single carbon atom as input molecule whose atom feature vector gets mapped to zero in the first convolution. Previous implementations would from
this moment on treat the carbon atom as nonexistent and thus the molecule as empty.
'''


# import packages
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, InputSpec, Dropout
from tensorflow.keras import initializers, regularizers, constraints
from . import auxiliary_functions_graph_tensorisation as afgt
from . import auxiliary_functions_neural_graph_convolutions as afngf



class NeuralFingerprintHidden(tf.keras.layers.Layer):
	"""
	Hidden layer in a neural graph convolution (as in Duvenaud et. al.,
	2015). This layer takes a graph as an input. The graph is represented as by
	four tensors.
	
	- The atoms tensor represents the features of the nodes.
	- The bonds tensor represents the features of the edges.
	- The edges tensor represents the connectivity (which atoms are connected to which)
	- the atoms_existence tensor represents how many atoms each molecule has (its atom count) in form of a binary 1d array.
	
	It returns a tensor containing the updated atom feature vectors for each molecule.
	
	Input: (atoms, bonds, edges, atoms_existence)
	
	- atoms: shape = (num_molecules, max_atoms, num_atom_features))
	- bonds: shape = (num_molecules, max_atoms, max_degree, num_bond_features))
	- edges: shape = (num_molecules, max_atoms, max_degree)
	- atoms_existence: shape = (num_molecules, max_atoms)
	
	Output: atoms_updated
	
	- atoms_updated: updated (i.e. convolved) atom features with shape = (num_molecules, max_atoms, conv_width))
	"""
	

	def __init__(self, 
				 conv_width,
				 activation = tf.keras.activations.relu, 
				 use_bias = True, 
				 kernel_initializer = tf.keras.initializers.GlorotUniform,
				 bias_initializer = tf.keras.initializers.Zeros,
				 dropout_rate_input = 0,
				 **kwargs):
		
		super(NeuralFingerprintHidden, self).__init__(**kwargs)
		
		self.conv_width = conv_width
		self.activation = activation
		self.use_bias = use_bias
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer
		self.dropout_rate_input = dropout_rate_input
		self.max_degree = None # value of max_degree is not known just yet, but note that it further down defines the number of weight matrices W_d


	def build(self, inputs_shape):

		# import dimensions
		(max_atoms, max_degree, num_atom_features, num_bond_features, num_molecules) = afngf.mol_shapes_to_dims(mol_shapes = inputs_shape[0:3])
		num_atom_bond_features = num_atom_features + num_bond_features
		
		# set value for attribute self.max_degree
		self.max_degree = max_degree

		# generate dropout latyer for input
		self.Drop_input = Dropout(self.dropout_rate_input) 
		
		# generate trainable layers D_0, ..., D_{max_degree} (for each degree d = 0,...,max_degree we convolve with a different dense layer D_i)
		
		self.D_list = []
		
		for degree in range(0, self.max_degree + 1):
			
			# initialize dense layers D_1,...,D_{max_degree}
			exec("self.D_" + str(degree) + " = Dense(units = self.conv_width, activation = self.activation, use_bias = self.use_bias, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)")
			exec("self.D_list.append(self.D_" + str(degree) + ")")

			# initialize dense layers D_1,...,D_{max_degree}
			exec("self.D_" + str(degree) + " = Dense(units = self.conv_width, activation = self.activation, use_bias = self.use_bias, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer)")
			exec("self.D_list.append(self.D_" + str(degree) + ")")
		


	def call(self, inputs, mask=None):
		
		atoms = inputs[0] # atoms.shape = (num_molecules, max_atoms, num_atom_features )
		bonds = inputs[1] # bonds.shape = (num_molecules, max_atoms, max_degree, num_edge_features)
		edges = inputs[2] # edges.shape = (num_molecules, max_atoms, max_degree)
		atoms_existence = inputs[3] # atoms_existence.shape (num_molecules, max_atoms)
		atoms_existence = tf.reshape(atoms_existence, shape = (tf.shape(atoms_existence)[0], 1, tf.shape(atoms_existence)[1])) # reshape atoms_existence to dimension (num_molecules, 1, max_atoms)

		# import dimensions
		max_atoms = atoms.shape[1]
		num_atom_features = atoms.shape[-1]
		num_bond_features = bonds.shape[-1]
		
		# sum the edge features for each atom
		summed_bond_features = tf.math.reduce_sum(bonds, axis = -2) # summed_bond_features.shape = (num_molecules, max_atoms, num_edge_features)
		
		# for each atom, look up the features of it's neighbour
		neighbour_atom_features = afngf.neighbour_lookup(atoms, edges, include_self = True)

		# sum along degree axis to get summed neighbour features
		summed_atom_features = tf.reduce_sum(neighbour_atom_features, axis = -2) # summed_atom_features.shape = (num_molecules, max_atoms, num_atom_features)
		
		# concatenate the summed atom and bond features
		summed_atom_bond_features = tf.concat([summed_atom_features, summed_bond_features], axis = -1) # summed_atom_bond_features.shape = (num_molecules, max_atoms, num_atom_bond_features)

		# create a matrix that stores for each atom, the degree it is
		atom_degrees = tf.math.reduce_sum(tf.cast(tf.math.not_equal(edges, -1), dtype = np.float32), axis = -1, keepdims = True) # atom_degrees.shape = (num_molecules, max_atoms, 1) 

		# for each degree we now apply a different trainable layer
		new_features_by_degree = []
		
		for degree in range(0, self.max_degree + 1):

			# choose the right degree-dependent layer
			D_degree = self.D_list[int(degree)]

			# apply dropout
			new_unmasked_features = self.Drop_input(summed_atom_bond_features)
			
			# apply dense layer
			new_unmasked_features = D_degree(new_unmasked_features)
			
			# create mask for this degree and perform degree-dependent masking via multiplication with binary 0-1 tensor
			atom_masks_this_degree = tf.cast(tf.math.equal(atom_degrees, degree), dtype = np.float32) # atom_masks_this_degree.shape = (num_molecules, max_atoms, 1)
			new_masked_features = new_unmasked_features * atom_masks_this_degree

			new_features_by_degree.append(new_masked_features)

		# sum the masked feature tensors for all degree types
		atoms_updated = tf.keras.layers.Add()(new_features_by_degree) # atoms_updated.shape = (num_molecules, max_atoms, self.conv_width)
		
		# finally set feature rows of atoms_updated which correspond to non-existing atoms to 0 via 0-1 binary multiplicative masking (this step is where we need atoms_existence)
		atoms_updated = tf.linalg.matrix_transpose(tf.linalg.matrix_transpose(atoms_updated) * atoms_existence)

		return atoms_updated


	def get_config(self):
		
		base_config = super(NeuralFingerprintHidden, self).get_config()
		
		config = {'Number of Output Units (Convolutional Width)': self.conv_width,
				'Activation Function': self.activation,
				'Usage of Bias Vector': self.use_bias,
				'Kernel Initalizer': self.kernel_initializer,
				'Bias Initializer': self.bias_initializer,
				'Input Layer Dropout Rate': self.dropout_rate_input}

		return dict(list(config.items()) + list(base_config.items()))

 
class NeuralFingerprintOutput(tf.keras.layers.Layer):
	"""
	Output layer in a neural graph convolution (as in Duvenaud et. al.,
	2015). This layer takes a graph as an input. The graph is represented as by
	four tensors.
	
	- The atoms tensor represents the features of the nodes.
	- The bonds tensor represents the features of the edges.
	- The edges tensor represents the connectivity (which atoms are connected to which)
	- the atoms_existence tensor represents how many atoms each molecule has (its atom count) in form of a binary 1d array.
	
	It returns the layer-based neural graph fingeprint for the layer features specified by the input. The neural fingerprints of all layers need to be summed up
	to obtain the fingerprint of the whole molecule according to Duvenaud.
	
	Input: (atoms, bonds, edges, atoms_existence)
	
	- atoms: shape = (num_molecules, max_atoms, num_atom_features))
	- bonds: shape = (num_molecules, max_atoms, max_degree, num_bond_features))
	- edges: shape = (num_molecules, max_atoms, max_degree)
	- atoms_existence: shape = (num_molecules, max_atoms)
	
	Output: fp_layerwise
	
	- fp_layerwise: Neural fingerprint for graph layer specified by input, with shape = (num_molecules, fp_length)
	"""
	
	
	def __init__(self, 
				 fp_length, 
				 activation = tf.keras.activations.softmax, 
				 use_bias = True, 
				 kernel_initializer = tf.keras.initializers.GlorotUniform,
				 bias_initializer = tf.keras.initializers.Zeros,
				 dropout_rate_input = 0,
				 **kwargs):
		
		super(NeuralFingerprintOutput, self).__init__(**kwargs)
		
		self.fp_length = fp_length
		self.activation = activation
		self.use_bias = use_bias
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer
		self.dropout_rate_input = dropout_rate_input

		
	def build(self, inputs_shape):

		# import dimensions
		(max_atoms, max_degree, num_atom_features, num_bond_features, num_molecules) = afngf.mol_shapes_to_dims(mol_shapes = inputs_shape[0:3])
		num_atom_bond_features = num_atom_features + num_bond_features

		# initialize dropout layer
		self.Drop_input = Dropout(self.dropout_rate_input)

		# initialize trainable layer
		self.D = Dense(units = self.fp_length, activation = self.activation, use_bias = self.use_bias, kernel_initializer = self.kernel_initializer, bias_initializer = self.bias_initializer) 

	   
	def call(self, inputs, mask = None):
		
		atoms = inputs[0] # atoms.shape = (num_molecules, max_atoms, num_atom_features )
		bonds = inputs[1] # bonds.shape = (num_molecules, max_atoms, max_degree, num_edge_features)
		edges = inputs[2] # edges.shape = (num_molecules, max_atoms, max_degree)
		atoms_existence = inputs[3] # atoms_existence.shape (num_molecules, max_atoms)
		atoms_existence = tf.reshape(atoms_existence, shape = (tf.shape(atoms_existence)[0], 1, tf.shape(atoms_existence)[1])) # reshape atoms_existence to dimension (num_molecules, 1, max_atoms)

		# import dimensions
		max_atoms = atoms.shape[1]
		num_atom_features = atoms.shape[-1]
		num_bond_features = bonds.shape[-1]

		# sum the edge features for each atom
		summed_bond_features = tf.math.reduce_sum(bonds, axis=-2) # summed_bond_features.shape = (num_molecules, max_atoms, num_edge_features)

		# concatenate the atom features and summed bond features
		summed_atom_bond_features = tf.concat([atoms, summed_bond_features], axis=-1) # summed_atom_bond_features.shape = (num_molecules, max_atoms, num_atom_bond_features)

		# apply dropout
		neural_fp_atomwise = self.Drop_input(summed_atom_bond_features)

		# apply trainable layer to compute fingerprint
		neural_fp_atomwise = self.D(neural_fp_atomwise)

		# set feature rows of neural_fp_atomwise which correspond to non-existing atoms to 0 via 0-1 binary multiplicative masking (this step is where we need atoms_existence)
		neural_fp_atomwise_masked = tf.linalg.matrix_transpose(tf.linalg.matrix_transpose(neural_fp_atomwise) * atoms_existence)
		
		# add up all atomwise neural fingerprints of all existing atoms in each molecule to obtain the layerwise neural fingerprint
		neural_fp_layerwise = tf.math.reduce_sum(neural_fp_atomwise_masked, axis=-2) # neural_fp_layerwise.shape = (num_molecules, self.fp_length)

		return neural_fp_layerwise
	
	
	def get_config(self):
		
		base_config = super(NeuralFingerprintOutput, self).get_config()
		
		config = {'Number of Output Units (Neural Fingerprint Length)': self.fp_length,
				'Activation Function': self.activation,
				'Usage of Bias Vector': self.use_bias,
				'Kernel Initalizer': self.kernel_initializer,
				'Bias Initializer': self.bias_initializer,
				'Input Layer Dropout Rate': self.dropout_rate_input}

		return dict(list(config.items()) + list(base_config.items()))
	

