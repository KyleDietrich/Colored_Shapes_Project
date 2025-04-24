# Colored Shapes Project

This repository reproduces and extends experiments from 

Foundations of Computer Vision (Torralba, Isola, Freeman, 2024). 

We explore:

-- Autoencoder Training on a synthetic dataset of colored shapes.

-- Contrastive Learning (alignment + uniformity) to learn 2D embeddings that capture either color or 
   shape invariances based on different data augmentations.

# Project Structure

colored_shapes_project/

  ├── dataset.py

  ├── examples.py
  
  ├── loss_functions.py
  
  ├── training.py

  ├── models.py
  
  ├── autoencoder_experiment.py
  
  ├── contrastive_experiment.py

  ├── analyze_autoencoder_embeddings.py

  ├── visualize_embeddings.py
  
  └── README.md

probably explain what each does here

# Installation
-- clone 

-- install dependencies

	pip install torch torchvision torchaudio
 
	pip install numpy pillow matplotlib
 
	pip install scikit-learn

# Usage
__All commands assume you are at the project root and have all depedencies installed__


**Preview synthetic dataset:**

	python examples.py



**Train the 128-dim convolutional auto-encoder:**

	python autoencoder_experiment.py



**Train the contrastive models (color-sensitive & shape-sensitive):**

	python constrative_experiment.py



**Analyze & visualize auto-encoder embeddings:**

	python analyze_autoencoder_embeddings.py



**Visualize 2-D contrastive models (Be sure to change the model under CONFIG in the file):**

	python visualize_embeddings.py
	


**Re-running with CPU only (Set the env-var before any script):**

 	set CUDA_VISIBLE_DEVICES=
 

# References
-- Foundations of Computer Vision (Torralba, Isola, Freeman, 2024).

-- Wang & Isola (ICML 2020), Understanding Contrastive Representation 
		 Learning Through Alignment and Uniformity on the Hypersphere.

# Authors
-- Kyle Dietrich, Dietrich.191@osu.edu

-- Preston Hines, Hines.470@osu.edu
