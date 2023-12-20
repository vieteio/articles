# Viete for deep learning

## 1. Abstract

Category theory offers a visual mathematical language for describing arbitrary structures, providing better capabilities for describing project structure and code than industrial languages ​​and development environments. Viete addresses 	the problem of adapting category theory and, through a series of examples, reveals the capabilities of language in application to various tasks. This article proposes a categorical DL Ops solution for implementing optimizations in decoder-only LLM models.

## 2. Introduction

### 2.1. Problem statement

Working with and training LLM requires significant computational resources and memory. Various optimizations are applied to reduce running time and/or memory requirements. Optimizations are integrated into the model by adding code for them to the model’s code or by writing model graph handlers that modify parts of the layers or work with certain tensors - a more universal approach. However, each approach has its limitations. When a new model is released, it necessitates repeated efforts to add optimizations to it. If multiple optimizations that are mathematically compatible need to be applied, it requires studying their code and making efforts to achieve technical compatibility.

Examples:
1. In the Hugging Face library, the gradient checkpointing optimization call fragment is contained in the Transformer block code of each LLM model. When a new model is released, the fragment is copied into its Transformer block.
2. Optimizations applied to computational graphs of models usually identify necessary layers or tensors with simple patterns, for example, by the names of fields in models. For instance, QLoRA checks for the presence of words like “lm_head”, “norm”, or “embed_tokens” in field names. Such requirements for model fields are not described in the repository's README. If a Data Science specialist wants to apply QLoRA to a model with a new architecture, then he would need to study the code to understand how to properly name the fields of his architecture so that the optimization affects all the necessary fields and only them.
3. Adding parallel pipeline optimization in PyTorch requires converting the model to an nn.Sequential module. A typical implementation of a Transformer block uses nn.ModuleList. Although mathematically compatible, technically they are different.

### 2.2 Proposed solution

Since optimization techniques are mathematical in nature, they can be described by functors in the language of category theory. Such a description of optimizations is universal and functional:
1. The description of optimization is separated from the description of the model.
2. The functor can be applied to any model that satisfies the requirements of the original category, resulting in an optimized version.
3. Transformations of the computational graph are tied to computational logic, and not through naming variables.
4. Several functors can be applied to use multiple optimizations.

Thus, any mathematically compatible model optimizations can be easily combined. Technical limitations and integration difficulties caused by implementation specifics disappear as a phenomenon.

In this article, we propose a visual notation for the categorical description of LLM models and optimization functors, as an extension of the Viete editor syntax. We will demonstrate with examples of gradient checkpointing and parallel pipeline optimizations that the descriptions are compact. We will also describe the possibility of simultaneous use of several optimization functors.

The implementation of full notation support in the editor will allow for a compact description of neural network execution optimizations and configure the launch of neural networks with optimizations without the need for development efforts.

### 2.3 Related work

In the paper “Backprop as a Functor” https://arxiv.org/abs/1711.10455 a categorical representation of neural networks is described, demonstrating the possibilities for parallelism in computation and training of neural networks. In our work, a categorical description of neural networks is given in the context of applying optimizations to LLM models and a categorical description of specific optimizations widely used in practice is provided.

## 3. LLM model optimizations

In this section, we briefly describe the gradient checkpointing and parallel pipeline optimizations in order to provide a categorical description of them later.

### 3.1. Gradient checkpointing

Gradient checkpointing is a memory optimization during model training. During forward calculation, not all tensors at the output of layers are saved, but only every $\sqrt{L}$ out of $L$ layers. The saved tensors are called checkpoints. During backpropagation, a tensor that is not saved is computed from the nearest checkpoint. This involves computing all intermediate tensors from the checkpoint to the required tensor. These are stored in memory, as they will be used immediately in the next backward steps after the current one with the same checkpoint. As soon as the checkpoint changes, the intermediate tensors to the previous checkpoint are deleted. There are no more intermediate tensors than $\sqrt{L}$, as well as checkpoints, i.e. no more than $2*\sqrt{L}$ tensors are stored in memory. Each tensor except checkpoints is recomputed, i.e. the time of forward computations approximately doubles.

A detailed description with a good visualization of gradient checkpointing can be found here: https://github.com/cybertronai/gradient-checkpointing.

### 3.2 Parallel pipeline

Parallel pipeline is an efficient distribution of computations for a batch across multiple GPUs on a single machine. This optimization is applied during model training. The batch is divided into n parts according to the number of GPUs. Similarly, LLM layers are divided into n consecutive parts according to the number of GPUs. Parts of batches are sequentially fed to the first GPU for computation on the first part of the layers. When the first part of the batches is processed on the first GPU, simultaneously with the new part of the batches beginning to be processed on the first GPU, parallel computations on the second GPU on the second part of the layers start for the first part of the batches. This mechanism allows the computation of each part of the batches on all GPUs in a chain, performing parallel calculations for other parts on other GPUs.

A general description of parallel pipeline can be found here https://pytorch.org/docs/stable/pipeline.html, and an example of its application to LLM models is here: https://pytorch.org/tutorials/intermediate/pipeline_tutorial.html.

## 4. Categorical description of LLM model optimizations

To describe models and functors in Viete, the decomposable mode is used, which allows for the description of Cartesian Closed Categories.

The categorical schemes presented in this and subsequent sections can be viewed in the Viete editor via this link: https://editor.viete.io/912c8bb6-1109-4c03-8025-8e9ac61b0f87.

### 4.1 Neural network model

Let’s categorically describe a decoder-only LLM to build a functor into an optimized model. We consider an LLM as a neural network with sequentially arranged layers - this is sufficient for a high-level description of LLM models, as they consist of a sequence of Transformer (or Decoder) blocks. LLM contains parallel computing lines in Residual connections inside the Transformer block, but since we apply optimizations to the LLM blocks as black boxes, this does not affect our logic.

We'll break down the categorical description of the LLM model using an example, and then give a description in the general case

#### 4.1.1 Example

In the scheme below, the LLM model consists of 6 Transformer blocks. The neural network here is a function parameterized by weights, which takes an input tensor $x$ and returns an output tensor $y$. Categorically, we represent this function by the morphism $forward$ from the object x to the object y. This is the arrow labeled $forward$ on the left of the scheme. The architecture of LLM is the internal implementation of a function. We represent it as a decomposition of the morphism $forward$ into a chain of morphisms, corresponding to Transformer blocks - on the right of the scheme.

![Example](https://github.com/vieteio/articles/assets/800129/4c16e675-dc8c-448e-b114-ff6aaf82136b)

Number Transformer blocks in LLM with numbers from $1$ to $L$. A block with number $l$ is a parameterized by weights $w_l$ function $f$ (short for forward) from the input tensor to the output tensor. The function for a specific block with weights $w_l$ can be denoted as $f_{w_l}=f(w_l)$.
Objects in the chain are a pair (cartesian product) of the tensor t at the output of the block and the number of this block $layer$. For $x$ the input tensor of the first block, we also add $layer=0$.

Then the morphism in the chain should compute a new tensor and a block number $layer$. Therefore, we represent the morphism as a cartesian product of two morphisms: $+1$ and $f$. Here the morphism $f$ for each block $l$ corresponds to the function $f_{w_l}$. For brevity we omit the index $w_l$. We get different morphisms between different pairs of objects, but with the same label $f$. Similarly, for example, in the category Preorder, all morphisms are denoted with the same label $\leq$.

#### 4.1.2 General scheme

Now let's describe the categorical scheme $𝓜$ for a model with an arbitrary number of layers. To do this, it is sufficient to describe the morphism of one block. We will do this and, for clarity, we will additionally show how the block arrows add up to the composition. The scheme is presented below. The Composition of several morphisms $+1$ from the tensor $x$ to $t$ - the output of the $l$ block - gives a morphism $+l$. For a tensor $y$, the morphism to the block number will be $+L$.
![General scheme](https://github.com/vieteio/articles/assets/800129/1c462574-e60c-43f2-9dc5-f69ff7d78467)
