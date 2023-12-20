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

### 4.1. Neural network model

Let’s categorically describe a decoder-only LLM to build a functor into an optimized model. We consider an LLM as a neural network with sequentially arranged layers - this is sufficient for a high-level description of LLM models, as they consist of a sequence of Transformer (or Decoder) blocks. LLM contains parallel computing lines in Residual connections inside the Transformer block, but since we apply optimizations to the LLM blocks as black boxes, this does not affect our logic.

We'll break down the categorical description of the LLM model using an example, and then give a description in the general case

#### 4.1.1. Example

In the scheme below, the LLM model consists of 6 Transformer blocks. The neural network here is a function parameterized by weights, which takes an input tensor $x$ and returns an output tensor $y$. Categorically, we represent this function by the morphism $forward$ from the object x to the object y. This is the arrow labeled $forward$ on the left of the scheme. The architecture of LLM is the internal implementation of a function. We represent it as a decomposition of the morphism $forward$ into a chain of morphisms, corresponding to Transformer blocks - on the right of the scheme.

![Example](https://github.com/vieteio/articles/assets/800129/4c16e675-dc8c-448e-b114-ff6aaf82136b)

Number Transformer blocks in LLM with numbers from $1$ to $L$. A block with number $l$ is a parameterized by weights $w_l$ function $f$ (short for forward) from the input tensor to the output tensor. The function for a specific block with weights $w_l$ can be denoted as $f_{w_l}=f(w_l)$.
Objects in the chain are a pair (cartesian product) of the tensor t at the output of the block and the number of this block $layer$. For $x$ the input tensor of the first block, we also add $layer=0$.

Then the morphism in the chain should compute a new tensor and a block number $layer$. Therefore, we represent the morphism as a cartesian product of two morphisms: $+1$ and $f$. Here the morphism $f$ for each block $l$ corresponds to the function $f_{w_l}$. For brevity we omit the index $w_l$. We get different morphisms between different pairs of objects, but with the same label $f$. Similarly, for example, in the category Preorder, all morphisms are denoted with the same label $\leq$.

#### 4.1.2. General scheme

Now let's describe the categorical scheme $𝓜$ for a model with an arbitrary number of layers. To do this, it is sufficient to describe the morphism of one block. We will do this and, for clarity, we will additionally show how the block arrows add up to the composition. The scheme is presented below. The Composition of several morphisms $+1$ from the tensor $x$ to $t$ - the output of the $l$ block - gives a morphism $+l$. For a tensor $y$, the morphism to the block number will be $+L$.
![General scheme](https://github.com/vieteio/articles/assets/800129/1c462574-e60c-43f2-9dc5-f69ff7d78467)

### 4.2. Backward computation

Gradient checkpointing and parallel pipeline are optimizations of the neural network training process. They affect both forward and backward learning steps. For the neural networks described above, only the forward step was detailed. In the paper “Backprop as Functor” https://arxiv.org/abs/1711.10455 the construction of the backprop step is derived from the forward step with the functor $ℒ: Para \rightarrow Learn$. That is, we can obtain a backward computation scheme from a forward computation scheme by applying the functor.

We can apply the optimization functor to our LLM model scheme $𝓜$ to obtain the scheme of forward optimized computations $𝓜_{opt}$. Applying the functor $ℒ$ to this scheme will allow us to get a complete scheme of neural network training with optimization. For each optimization, the functor for backward computation $ℒ$ must be extended to support the peculiarities of the computations. The corresponding extension is provided for each optimization where needed.

### 4.3. Memory state during calculations

Gradient checkpointing optimizes neural network execution in terms of memory. To demonstrate the optimization operation, we need to show how memory consumption changes during neural network execution.

We describe categorically the computations during the operation of the neural network. The object in this category is the memory state before or after the computation of block $l$, with the morphism being the transition from one state to another during the computation of a block.

Below is an example of a categorical scheme for a LLM with 6 blocks and a general scheme $𝓜_{mem}$ for a model with an arbitrary number of layers. The latter describes a morphism for a block with number $l$ and examples of the composition of a chain of block morphisms.

<img width="355" alt="Example" src="https://github.com/vieteio/articles/assets/800129/7956071f-eddf-472c-9ac9-871808335ed0">
<img width="463" alt="General scheme" src="https://github.com/vieteio/articles/assets/800129/51f86e9f-6173-4456-8787-0d90a2372b6a">

Before the start of computations, only the input tensor $x$ is contained in memory. After the computation of the first block, the tensor $t$ at the output of this block is added to the state. After each next block one more tensor $t$ is added. The morphism $f_l$ with index $l$ means composition of $l$ block morphisms $f$ from the input tensor $x$ to the output of block number $l$.

**Note.** The tensors at the output of the block are stored in memory for later use in backpropagation. Therefore, it is more accurate to say here that the tensor $t$ is not only the value computed by the block, but also all intermediate values computed inside the block, for example in the attention and MLP layers. In a typical implementation of the attention layer, three intermediate attention matrices are computed. The label $t$ here means not the same object as in the section above.

General state at the block output $l$ is the Cartesian product $x \times t^l = x \times t^{l-1} \times t$. The morphism for block $l$ transitions from $x \times t^{l-2} \times t = x \times t^{l-1}$ to $x \times t^{l-1} \times t$. This morphism is the Cartesian product of the identity  morphism for $x$, combinations of morphisms that group $t^{l-2} \times t$ into $t^{l-1}$ and morphism $f$, which computes a new tensor $t$ in the product $t^{l-2} \times t$.

The complete scheme of this Cartesian product is as follows:

<img width="372" alt="Scheme with group morphisms" src="https://github.com/vieteio/articles/assets/800129/957847a4-8d2f-419b-a0da-aa7c5bbad9a0">

The tensor $t$ at the output of the previous block is copied by the morphism $∆$, then the projection $pr_1$ is saved for later use (in backpropagation), and the projection $pr_2$ passed $t$ for computation in block $l$.
In the schemes for the neural network, the identity morphism and grouping morphisms are omitted for brevity and clarity, leaving only the computationally meaningful morphism $f$.

#### 4.3.2.1. Functor $𝒰: 𝓜 \rightarrow 𝓜_{mem}$

The categorical scheme $𝓜_{mem}$ can be constructed from $𝓜$ using a functor $𝒰: 𝓜 \rightarrow 𝓜_{mem}$ with the following transformations of objects and morphisms:

$𝒰_1(l \times t)=x \times t^{l-1} \times t$

$𝒰_2(+1 \times f)$ translates morphism from $𝓜$ into a diagram with groupings morphism as described above. Formally, this can be written as:

$𝒰_2(+1 \times f)=1_x \times (1_{t^{l-2}} \times (pr_1 \circ ∆)) \times f \circ pr_2 \circ ∆$. Recall that here $f$ is an abbreviation for $f_{w_l}$.

The inverse functor $𝒰^{-1}$, operating from $𝓜$ to $𝓜_{mem}$, can also be considered.

#### 4.3.2. Backpropagation

During the backpropagation stage, gradients are computed using the tensors from the forward step. The gradient is computed as $g=backward(g_{previous}, t)$. Here $t$, from a categorical point of view, is a saved copy of the tensor that has been computed by the morphism $pr_1 \circ ∆ \circ f_l$. The functor $ℒ$, constructing the scheme of backward computations based on forward computations, should translate that computed tensor $t$ into the saved copy. Thus, the functor should translate the morphism $pr_2 \circ ∆ \circ f_l$ into morphism $pr_1 \circ ∆ \circ f_l$, $ℒ(pr_2 \circ ∆ \circ f_l)=pr_1 \circ ∆ \circ f_l$.

### 4.4 Batch computations

During model training and often during its execution, data is fed into the neural network in batches, adding a batch dimension to the input tensor. The batch size is not important for gradient checkpoint optimization, but it is important for the parallel pipeline - the batch size should be greater than 1, preferably at least equal to the number of GPUs in the computer.

We describe the categorical scheme of a neural network that takes batches as input. Science batch size affects the computation time, we replace the numbering of the blocks with a computation timestamp.

<img width="516" alt="Batched computations" src="https://github.com/vieteio/articles/assets/800129/3d554a12-8f5e-4061-b3c2-cffbb15bae8d">

The objects in the scheme are a Cartesian product of time - a timestamp from the start of the computation, and a batch of tensors. The block morphism is a Cartesian product of two morphisms: $+time$, which increases the timestamp by the duration of the computation, and $f$, which computes the output batch of tensors of size $bs$.

Accurately calculating the actual computation time is difficult due to many factors - parallelism in computations, different types of memory with varying access speeds in the GPU, delays in transferring data from one memory to another. Therefore, the number of computational operations per GPU can be chosen as a time metric.

**Note.** The unit of measurement for the number of computational operations can be FLOP. The number of operations can be converted to execution time (with certain accuracy) through FLOPS, known for the GPU.

With the accepted metric (the number of computational operations), let’s denote the computation time of a Transformer block on a single GPU on the input tensor $t$ as $T_c$. Then computing a batch of size $bs$ requires $bs$ times more computations: $T_{c \textunderscore bs}=T_c * bs$. Computing $l$ consecutive Transformer blocks on the input $x$ requires $l * T_c * bs$ operations. This is reflected in the categorical scheme.

### 4.5. Gradient checkpointing
#### 4.5.1. Decomposition of the computation of $l$ layers into computation up to the checkpoint and after

In gradient checkpointing, the computation of layer $l$ occurs as the computation of a checkpoint and several layers behind the checkpoint.

This can be shown by representing a morphism from $x$ to the tensor $t$ at the output of block $l$ through the following decomposition in the optimized computation diagram $𝓜_{gch}$:

<img width="460" alt="Gradient checkpointing model" src="https://github.com/vieteio/articles/assets/800129/24de0743-f7c8-4976-8656-493cbe69d090">

**Note.** Checkpoints are computed every $\sqrt{L}$ blocks. Here and below, for simplicity, we assume that $L$ is such that $\sqrt{L}$ is an integer.

In general, the tensor at the output of the block number $l$, if this is not a checkpoint, is computed as several steps from the nearest previous checkpoint. Represent $l$ as a sum $l=k*L+i$, where $k \in ℕ \cup \{0\}$, $0 \leq i < \sqrt{L}$.

The morphism $f_l$ is presented as a decomposition $f_l=f_i \circ pr_2 \circ ∆ \circ f_{k*\sqrt{L}}$. Here, the morphism $f_{k*\sqrt{L}}$ goes from the input tensor $x$ to the tensor $ch$, which will be saved as a checkpoint, and the morphism $f_i$ goes from the checkpoint to the tensor $t$. Morphism $∆$ copies the checkpoint for saving, and the projection $pr_2$ directs the checkpoint for computation in the next block. Similarly, the morphism for the layer label $+l$ is represented as a decomposition $+l=(+i) \circ (+k*\sqrt{L})$

#### 4.5.2. Memory state during computations with checkpoints

Saving checkpoints can be depicted in the categorical diagram $𝓜_{gch \textunderscore mem}$ for executing an LLM with gradient checkpointing.

<img width="401" alt="Example" src="https://github.com/vieteio/articles/assets/800129/c520845b-e9e0-4c4f-9205-184480e9444a">

<img width="473" alt="General schema" src="https://github.com/vieteio/articles/assets/800129/98345195-467f-405e-9b7d-8fb501df1f78">

The general diagram $𝓜_{gch \textunderscore mem}$ shows the memory state after execution block $l$. Only checkpoints and the last computed tensor $t$ are saved, intermediate tensors are deleted, freeing up memory.

The functor $𝒰_{gch}: 𝓜_{gch} \rightarrow 𝓜_{gch \textunderscore mem}$ and the inverse functor $𝒰_{gch}^{-1}$ can be described similarly to functors $𝒰$ and $𝒰^{-1}$.

#### 4.5.3. Backpropagation

Consider that in gradient checkpointing, the morphism $f_l$ to the tensor at the output of block $l$ decomposes: $f_l=f_i \circ pr_2 \circ ∆ \circ f_{k * \sqrt{L}}$. During the backward step, the saved checkpoint will be used, i.e. projection $pr_2$ needs to be replaced with $pr_1$. The functor for backpropagation $ℒ$ in gradient checkpointing should translate the morphism $f_i \circ pr_2 \circ∆ \circ f_{k * \sqrt{L}}$ for the tensor $t$ into morphism $f_i \circ pr_1 \circ ∆ \circ f_{k * \sqrt{L}}$.

Since during backpropagation, computations along the blocks go in the reverse order, if a tensort at the output of block $l=i + k * \sqrt{L}$ is computed, then all previous $i-1$ computed tensors from the last checkpoint should be saved, as they will be needed in the next backward steps. Thus, the functor $ℒ$ for backpropagation in gradient checkpointing should translate the object $x \times ch^k \times t$ at the output of block $l=i + k*\sqrt{L}$ to object $x \times ch^k \times t^{i-1} \times t$.

### 4.6. Parallel pipeline

In parallel pipeline, the batch is divided into $GPUn$ parts where $GPUn$ is the number of GPUs in the computer. The blocks of the LLM model are also divided into $GPUn$ sequential groups, each of $L/GPUn$ blocks. The weights of each group of blocks are loaded onto their respective GPUs card. Each batch part is then sequentially processed through the layers on different GPUs. While one part of the batch is being processed, other parts are executed on other GPUs or wait idle until a GPU becomes available for computation.

The morphism of executing a whole batch in parallel pipeline is a Cartesian product of morphisms $f$ - executing batch parts on GPUs, or $id$ - the identity idle batch morphism. While each batch part is idle or is being processed on its GPU, passing through blocks on that GPU, the configuration of batch distribution across GPUs remains unchanged. Morphisms of execution (or idleness) of batches in one configuration can be combined into a composition. We obtain a composition of several morphisms $f$ into the morphism $f_{L/GPUn}$ and several $id$ morphisms into $id$ morphism.

Below is an example of a parallel pipeline operation scheme.

<img width="661" alt="Parallel pipeline" src="https://github.com/vieteio/articles/assets/800129/6a0aa383-c6a4-40ea-bae0-99eea26d91bf">

The initial state - input $x^{bs}$ is divided into parts $x^{bs/GPUn}$. The example separately highlights the first two parts, while the remaining {GPUn-2} parts are grouped into {x^{bs/GPUn*(GPUn - 2)}$.
The computation time of one batch part on a GPU is calculated using the metric of the number of computations on one GPU: $T_{c \textunderscore bs/GPUn}=T_c*bs/GPUn$. The computation time for $L/GPUn$ layers $T_{c \textunderscore bs/GPUn \textunderscore L/GPUn}=(L/GPUn)*T_c*(bs/GPUn)$. During this time, batch parts are computed in parallel on all GPUs or idle. Thus, in the first run through the first $L/GPUn$ layers, computation in Transformer blocks is performed only for the first batch part. For the other parts, the identity $id$ transformation is performed.

<img width="594" alt="Batch part computation" src="https://github.com/vieteio/articles/assets/800129/fb13e30d-dd7d-4e24-a00a-b22bd2e8d4c5">

After computing $L$ layers on GPUn groups with $L/GPUn$ layers per group for the first batch part, the computation ends with obtaining the final tensors $y$. For the other batch parts, only intermediate tensors $t$ are yet computed.

<img width="860" alt="First batch part full model computation" src="https://github.com/vieteio/articles/assets/800129/7516fd2a-7cd7-47d1-8d70-d584eacaef47">

Computations continue for the time of $GPUn-1$ groups, i.e. for the time $(GPUn-1) * T_{c \textunderscore bs \textunderscore L/GPUn}=(L-1) * \frac{(GPUn-1)}{GPU} * T_c*(bs/GPUn)$. The total computation time will be $T_{forward}=(L + (L-1) * \frac{(GPUn-1)}{GPU}) * T_c*(bs/GPUn)$.

#### 4.6.1. Backpropagation

The functor $ℒ$ translates forward computation into backward, i.e. morphism $f_L$ is transferred into $backward_L$. Morphisms of waiting are translated into morphisms of waiting for the backward step:

$ℒ(id_{GPUn-b} \circ f_L \circ id_{b-1})=id_{b-1} \circ backward_L \circ id_{GPUn-b}.$

$ℒ(T_c * bs/GPUn)=ℒ(T_b*bs/GPUn)$, where $T_b$ is the execution time of the backward step on the tensor $t$.



