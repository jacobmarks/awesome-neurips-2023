## Poster Session 4: Wednesday, Dec 13, 15:00 CT

### $\textbf{A}^2\textbf{CiD}^2$: Accelerating Asynchronous Communication in Decentralized Deep Learning

**Authors:** Adel Nabli, Eugene Belilovsky, Edouard Oyallon

**<details><summary>Abstract**</summary>

Distributed training of Deep Learning models has been critical to many recent successes in the field. Current standard methods primarily rely on synchronous centralized algorithms which induce major communication bottlenecks and synchronization locks at scale. Decentralized asynchronous algorithms are emerging as a potential alternative but their practical applicability still lags. In order to mitigate the increase in communication cost that naturally comes with scaling the number of workers, we introduce a principled asynchronous, randomized, gossip-based optimization algorithm which works thanks to a continuous local momentum named $\textbf{A}^2\textbf{CiD}^2$. Our method allows each worker to continuously process mini-batches without stopping, and run a peer-to-peer averaging routine in parallel, reducing idle time. In addition to inducing a significant communication acceleration at no cost other than adding a local momentum variable, minimal adaptation is required to incorporate $\textbf{A}^2\textbf{CiD}^2$ to standard asynchronous approaches. Our theoretical analysis proves accelerated rates compared to previous asynchronous decentralized baselines and we empirically show that using our $\textbf{A}^2\textbf{CiD}^2$ momentum significantly decrease communication costs in poorly connected networks. In particular, we show consistent improvement on the ImageNet dataset using up to 64 asynchronous workers (A100 GPUs) and various communication network topologies.</details>

### $k$-Means Clustering with Distance-Based Privacy

**Authors:** Alessandro Epasto, Vahab Mirrokni, Shyam Narayanan, Peilin Zhong

**<details><summary>Abstract**</summary>

In this paper, we initiate the study of Euclidean clustering with Distance-based privacy. Distance-based privacy is motivated by the fact that it is often only needed to protect the privacy of exact, rather than approximate, locations. We provide constant-approximate algorithms for $k$-means and $k$-median clustering, with additive error depending only on the attacker's precision bound $\rho$, rather than the radius $\Lambda$ of the space. In addition, we empirically demonstrate that our algorithm performs significantly better than previous differentially private clustering algorithms, as well as naive distance-based private clustering baselines.</details>

### $p$-Poisson surface reconstruction in curl-free flow from point clouds

**Authors:** Yesom Park, Taekyung Lee, Jooyoung Hahn, Myungjoo Kang

**<details><summary>Abstract**</summary>

The aim of this paper is the reconstruction of a smooth surface from an unorganized point cloud sampled by a closed surface, with the preservation of geometric shapes, without any further information other than the point cloud. Implicit neural representations (INRs) have recently emerged as a promising approach to surface reconstruction. However, the reconstruction quality of existing methods relies on ground truth implicit function values or surface normal vectors. In this paper, we show that proper supervision of partial differential equations and fundamental properties of differential vector fields are sufficient to robustly reconstruct high-quality surfaces. We cast the $p$-Poisson equation to learn a signed distance function (SDF) and the reconstructed surface is implicitly represented by the zero-level set of the SDF. For efficient training, we develop a variable splitting structure by introducing a gradient of the SDF as an auxiliary variable and impose the $p$-Poisson equation directly on the auxiliary variable as a hard constraint. Based on the curl-free property of the gradient field, we impose a curl-free constraint on the auxiliary variable, which leads to a more faithful reconstruction. Experiments on standard benchmark datasets show that the proposed INR provides a superior and robust reconstruction. The code is available at https://github.com/Yebbi/PINC.</details>

### 3D Copy-Paste: Physically Plausible Object Insertion for Monocular 3D Detection

**Authors:** Yunhao Ge, Hong-Xing Yu, Cheng Zhao, Yuliang Guo, Xinyu Huang, Liu Ren, Laurent Itti, Jiajun Wu

**<details><summary>Abstract**</summary>

A major challenge in monocular 3D object detection is the limited diversity and quantity of objects in real datasets. While augmenting real scenes with virtual objects holds promise to improve both the diversity and quantity of the objects, it remains elusive due to the lack of an effective 3D object insertion method in complex real captured scenes. In this work, we study augmenting complex real indoor scenes with virtual objects for monocular 3D object detection. The main challenge is to automatically identify plausible physical properties for virtual assets (e.g., locations, appearances, sizes, etc.) in cluttered real scenes. To address this challenge, we propose a physically plausible indoor 3D object insertion approach to automatically copy virtual objects and paste them into real scenes. The resulting objects in scenes have 3D bounding boxes with plausible physical locations and appearances. In particular, our method first identifies physically feasible locations and poses for the inserted objects to prevent collisions with the existing room layout. Subsequently, it estimates spatially-varying illumination for the insertion location, enabling the immersive blending of the virtual objects into the original scene with plausible appearances and cast shadows. We show that our augmentation method significantly improves existing monocular 3D object models and achieves state-of-the-art performance. For the first time, we demonstrate that a physically plausible 3D object insertion, serving as a generative data augmentation technique, can lead to significant improvements for discriminative downstream tasks such as monocular 3D object detection. Code: https://github.com/gyhandy/3D-Copy-Paste.</details>

### 3D molecule generation by denoising voxel grids

**Authors:** Pedro O. Pinheiro, Joshua Rackers, Joseph Kleinhenz, Michael Maser, Omar Mahmood, Andrew Watkins, Stephen Ra, Vishnu Sresht, Saeed Saremi

**<details><summary>Abstract**</summary>

We propose a new score-based approach to generate 3D molecules represented as atomic densities on regular grids.First, we train a denoising neural network that learns to map from a smooth distribution of noisy molecules to the distribution of real molecules.Then, we follow the _neural empirical Bayes_ framework [Saremi and Hyvarinen, 2019] and generate molecules in two steps: (i) sample noisy density grids from a smooth distribution via underdamped Langevin Markov chain Monte Carlo, and (ii) recover the "clean" molecule by denoising the noisy grid with a single step.Our method, _VoxMol_, generates molecules in a fundamentally different way than the current state of the art (ie, diffusion models applied to atom point clouds). It differs in terms of the data representation, the noise model, the network architecture and the generative modeling algorithm.Our experiments show that VoxMol captures the distribution of drug-like molecules better than state of the art, while being faster to generate samples.</details>

### [Spotlight] 3D-LLM: Injecting the 3D World into Large Language Models

**Authors:** Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, Chuang Gan

**<details><summary>Abstract**</summary>

Large language models (LLMs) and Vision-Language Models (VLMs) have been proved to excel at multiple tasks, such as commonsense reasoning. Powerful as these models can be, they are not grounded in the 3D physical world, which involves richer concepts such as spatial relationships, affordances, physics, layout, and so on. In this work, we propose to inject the 3D world into large language models, and introduce a whole new family of 3D-LLMs. Specifically, 3D-LLMs can take 3D point clouds and their features as input and perform a diverse set of 3D-related tasks, including captioning, dense captioning, 3D question answering, task decomposition, 3Dgrounding, 3D-assisted dialog, navigation, and so on. Using three types of prompting mechanisms that we design, we are able to collect over 300k 3D-language data covering these tasks. To efficiently train 3D-LLMs, we first utilize a 3D feature extractor that obtains 3D features from rendered multi-view images. Then, we use 2D VLMs as our backbones to train our 3D-LLMs. By introducing a 3D localization mechanism, 3D-LLMs could better capture 3D spatial information.  Experiments on ScanQA  show that our model outperforms state-of-the-art baselines by a large margin (\textit{e.g.}, the BLEU-1 score surpasses state-of-the-art score by 9\%). Furthermore, experiments on our held-in datasets for 3D captioning, task composition, and 3D-assisted dialogue show that our model outperforms 2D VLMs. Qualitative examples also show that our model could perform more tasks beyond the scope of existing LLMs and VLMs. Our model and data will be publicly available.</details>

### [Spotlight] 4M: Massively Multimodal Masked Modeling

**Authors:** David Mizrahi, Roman Bachmann, Oguzhan Kar, Teresa Yeo, Mingfei Gao, Afshin Dehghan, Amir Zamir

**<details><summary>Abstract**</summary>

Current machine learning models for vision are often highly specialized and limited to a single modality and task. In contrast, recent large language models exhibit a wide range of capabilities, hinting at a possibility for similarly versatile models in computer vision.In this paper, we take a step in this direction and propose a multimodal training scheme called 4M. It consists of training a single unified Transformer encoder-decoder using a masked modeling objective across a wide range of input/output modalities  – including text, images, geometric, and semantic modalities, as well as neural network feature maps. 4M achieves scalability by unifying the representation space of all modalities through mapping them into discrete tokens and performing multimodal masked modeling on a small randomized subset of tokens.4M leads to models that exhibit several key capabilities: (1) they can perform a diverse set of vision tasks out of the box, (2) they excel when fine-tuned for unseen downstream tasks or new input modalities, and (3) they can function as a generative model that can be conditioned on arbitrary modalities, enabling a wide variety of expressive multimodal editing capabilities with remarkable flexibility.Through experimental analyses, we demonstrate the potential of 4M for training versatile and scalable foundation models for vision tasks, setting the stage for further exploration in multimodal learning for vision and other domains.</details>

### A Bayesian Approach To Analysing Training Data Attribution In Deep Learning

**Authors:** Elisa Nguyen, Minjoon Seo, Seong Joon Oh

**<details><summary>Abstract**</summary>

Training data attribution (TDA) techniques find influential training data for the model's prediction on the test data of interest. They approximate the impact of down- or up-weighting a particular training sample. While conceptually useful, they are hardly applicable to deep models in practice, particularly because of their sensitivity to different model initialisation. In this paper, we introduce a Bayesian perspective on the TDA task, where the learned model is treated as a Bayesian posterior and the TDA estimates as random variables. From this novel viewpoint, we observe that the influence of an individual training sample is often overshadowed by the noise stemming from model initialisation and SGD batch composition. Based on this observation, we argue that TDA can only be reliably used for explaining deep model predictions that are consistently influenced by certain training data, independent of other noise factors. Our experiments demonstrate the rarity of such noise-independent training-test data pairs but confirm their existence. We recommend that future researchers and practitioners trust TDA estimates only in such cases. Further, we find a disagreement between ground truth and estimated TDA distributions and encourage future work to study this gap. Code is provided at https://github.com/ElisaNguyen/bayesian-tda.</details>

### A Bayesian Take on Gaussian Process Networks

**Authors:** Enrico Giudice, Jack Kuipers, Giusi Moffa

**<details><summary>Abstract**</summary>

Gaussian Process Networks (GPNs) are a class of directed graphical models which employ Gaussian processes as priors for the conditional expectation of each variable given its parents in the network. The model allows the description of continuous joint distributions in a compact but flexible manner with minimal parametric assumptions on the dependencies between variables. Bayesian structure learning of GPNs requires computing the posterior over graphs of the network and is computationally infeasible even in low dimensions. This work implements Monte Carlo and Markov Chain Monte Carlo methods to sample from the posterior distribution of network structures. As such, the approach follows the Bayesian paradigm, comparing models via their marginal likelihood and computing the posterior probability of the GPN features. Simulation studies show that our method outperforms state-of-the-art algorithms in recovering the graphical structure of the network and provides an accurate approximation of its posterior distribution.</details>

### A Definition of Continual Reinforcement Learning

**Authors:** David Abel, Andre Barreto, Benjamin Van Roy, Doina Precup, Hado van Hasselt, Satinder Singh

**<details><summary>Abstract**</summary>

In a standard view of the reinforcement learning problem, an agent’s goal is to efficiently identify a policy that maximizes long-term reward. However, this perspective is based on a restricted view of learning as finding a solution, rather than treating learning as endless adaptation. In contrast, continual reinforcement learning refers to the setting in which the best agents never stop learning. Despite the importance of continual reinforcement learning, the community lacks a simple definition of the problem that highlights its commitments and makes its primary concepts precise and clear. To this end, this paper is dedicated to carefully defining the continual reinforcement learning problem. We formalize the notion of agents that “never stop learning” through a new mathematical language for analyzing and cataloging agents. Using this new language, we define a continual learning agent as one that can be understood as carrying out an implicit search process indefinitely, and continual reinforcement learning as the setting in which the best agents are all continual learning agents. We provide two motivating examples, illustrating that traditional views of multi-task reinforcement learning and continual supervised learning are special cases of our definition. Collectively, these definitions and perspectives formalize many intuitive concepts at the heart of learning, and open new research pathways surrounding continual learning agents.</details>

### [Spotlight] A Dynamical System View of Langevin-Based Non-Convex Sampling

**Authors:** Mohammad Reza Karimi Jaghargh, Ya-Ping Hsieh, Andreas Krause

**<details><summary>Abstract**</summary>

Non-convex sampling is a key challenge in machine learning, central to non-convex optimization in deep learning as well as to approximate probabilistic inference. Despite its significance, theoretically there remain some important challenges: Existing guarantees suffer from the drawback of lacking guarantees for the last-iterates, and little is known beyond the elementary schemes of stochastic gradient Langevin dynamics. To address these issues, we develop a novel framework that lifts the above issues by harnessing several tools from the theory of dynamical systems. Our key result is that, for a large class of state-of-the-art sampling schemes, their last-iterate convergence in Wasserstein distances can be reduced to the study of their continuous-time counterparts, which is much better understood. Coupled with standard assumptions of MCMC sampling, our theory immediately yields the last-iterate Wasserstein convergence of many advanced sampling schemes such as mirror Langevin, proximal, randomized mid-point, and Runge-Kutta methods.</details>

### [Spotlight] A Graph-Theoretic Framework for Understanding Open-World Semi-Supervised Learning

**Authors:** Yiyou Sun, Zhenmei Shi, Yixuan Li

**<details><summary>Abstract**</summary>

Open-world semi-supervised learning aims at inferring both known and novel classes in unlabeled data, by harnessing prior knowledge from a labeled set with known classes. Despite its importance, there is a lack of theoretical foundations for this problem. This paper bridges the gap by formalizing a graph-theoretic framework tailored for the open-world setting, where the clustering can be theoretically characterized by graph factorization. Our graph-theoretic framework illuminates practical algorithms and provides guarantees. In particular, based on our graph formulation, we apply the algorithm called Spectral Open-world Representation Learning (SORL), and show that minimizing our loss is equivalent to performing spectral decomposition on the graph. Such equivalence allows us to derive a provable error bound on the clustering performance for both known and novel classes, and analyze rigorously when labeled data helps. Empirically, SORL can match or outperform several strong baselines on common benchmark datasets, which is appealing for practical usage while enjoying theoretical guarantees.</details>

### A Heavy-Tailed Algebra for Probabilistic Programming

**Authors:** Feynman Liang, Liam Hodgkinson, Michael Mahoney

**<details><summary>Abstract**</summary>

Despite the successes of probabilistic models based on passing noise through neural networks, recent work has identified that such methods often fail to capture tail behavior accurately---unless the tails of the base distribution are appropriately calibrated.  To overcome this deficiency, we propose a systematic approach for analyzing the tails of random variables, and we illustrate how this approach can be used during the static analysis (before drawing samples) pass of a probabilistic programming language (PPL) compiler.  To characterize how the tails change under various operations, we develop an algebra which acts on a three-parameter family of tail asymptotics and which is based on the generalized Gamma distribution.  Our algebraic operations are closed under addition and multiplication; they are capable of distinguishing sub-Gaussians with differing scales; and they handle ratios sufficiently well to reproduce the tails of most important statistical distributions directly from their definitions.  Our empirical results confirm that inference algorithms that leverage our heavy-tailed algebra attain superior performance across a number of density modeling and variational inference (VI) tasks.</details>

### A Hierarchical Spatial Transformer for Massive Point Samples  in Continuous Space

**Authors:** Wenchong He, Zhe Jiang, Tingsong Xiao, Zelin Xu, Shigang Chen, Ronald Fick, MILES MEDINA, Christine Angelini

**<details><summary>Abstract**</summary>

Transformers are widely used deep learning architectures. Existing transformers are mostly designed for sequences (texts or time series), images or videos, and graphs. This paper proposes a novel transformer model for massive (up to a million) point samples in continuous space. Such data are ubiquitous in environment sciences (e.g., sensor observations), numerical simulations (e.g., particle-laden flow, astrophysics), and location-based services (e.g., POIs and trajectories). However, designing a transformer for massive spatial points is non-trivial due to several challenges, including implicit long-range and multi-scale dependency on irregular points in continuous space, a non-uniform point distribution, the potential high computational costs of calculating all-pair attention across massive points, and the risks of over-confident predictions due to varying point density. To address these challenges, we propose a new hierarchical spatial transformer model, which includes multi-resolution representation learning within a quad-tree hierarchy and efficient spatial attention via coarse approximation. We also design an uncertainty quantification branch to estimate prediction confidence related to input feature noise and point sparsity. We provide a theoretical analysis of computational time complexity and memory costs. Extensive experiments on both real-world and synthetic datasets show that our method outperforms multiple baselines in prediction accuracy and our model can scale up to one million points on one NVIDIA A100 GPU. The code is available at https://github.com/spatialdatasciencegroup/HST</details>

### A Partially-Supervised Reinforcement Learning Framework for Visual Active Search

**Authors:** Anindya Sarkar, Nathan Jacobs, Yevgeniy Vorobeychik

**<details><summary>Abstract**</summary>

Visual active search (VAS) has been proposed as a  modeling framework in which visual cues are used to guide exploration, with the goal of identifying regions of interest in a large geospatial area. Its potential applications include identifying hot spots of rare wildlife poaching activity, search-and-rescue scenarios, identifying illegal trafficking of weapons, drugs, or people, and many others. State of the art approaches to VAS include applications of deep reinforcement learning (DRL), which yield end-to-end search policies, and traditional active search, which combines predictions with custom algorithmic approaches. While the DRL framework has been shown to greatly outperform traditional active search in such domains, its end-to-end nature does not make full use of supervised information attained either during training, or during actual search, a significant limitation if search tasks differ significantly from those in the training distribution. We propose an approach that combines the strength of both DRL and conventional active search approaches by decomposing the search policy into a prediction module, which produces a geospatial distribution of regions of interest based on task embedding and search history, and a search module, which takes the predictions and search history as input and outputs the search distribution. In addition, we develop a novel meta-learning approach for jointly learning the resulting combined policy that can make effective use of supervised information obtained both at training and decision time. Our extensive experiments demonstrate that the proposed representation and meta-learning frameworks significantly outperform state of the art in visual active search on several problem domains.</details>

### A Randomized Approach to Tight Privacy Accounting

**Authors:** Jiachen (Tianhao) Wang, Saeed Mahloujifar, Tong Wu, Ruoxi Jia, Prateek Mittal

**<details><summary>Abstract**</summary>

Bounding privacy leakage over compositions, i.e., privacy accounting, is a key challenge in differential privacy (DP). However, the privacy parameter ($\varepsilon$ or $\delta$) is often easy to estimate but hard to bound. In this paper, we propose a new differential privacy paradigm called estimate-verify-release (EVR), which tackles the challenges of providing a strict upper bound for the privacy parameter in DP compositions by converting an *estimate* of privacy parameter into a formal guarantee. The EVR paradigm first verifies whether the mechanism meets the *estimated* privacy guarantee, and then releases the query output based on the verification result. The core component of the EVR is privacy verification. We develop a randomized privacy verifier using Monte Carlo (MC) technique. Furthermore, we propose an MC-based DP accountant that outperforms existing DP accounting techniques in terms of accuracy and efficiency. MC-based DP verifier and accountant is applicable to an important and commonly used class of DP algorithms, including the famous DP-SGD. An empirical evaluation shows the proposed EVR paradigm improves the utility-privacy tradeoff for privacy-preserving machine learning.</details>

### A Robust Exact Algorithm for the Euclidean Bipartite Matching Problem

**Authors:** Akshaykumar Gattani, Sharath Raghvendra, Pouyan Shirzadian

**<details><summary>Abstract**</summary>

Algorithms for the minimum-cost bipartite matching can be used to estimate Wasserstein distance between two distributions.Given two sets $A$ and $B$ of $n$ points in a $2$-dimensional Euclidean space, one can use a fast implementation of the Hungarian method to compute a minimum-cost bipartite matching of $A$ and $B$ in $\tilde{O}(n^2)$ time. Let $\Delta$ be the spread, i.e., the ratio of the distance of the farthest to the closest pair of points in $A\cup B$. In this paper, we present a new algorithm to compute a minimum-cost bipartite matching of $A$ and $B$ with a similar worst-case execution time of $\tilde{O}(n^2 \log \Delta)$. However, when $A$ and $B$ are drawn independently and identically from a fixed distribution that is not known to the algorithm, the execution time of our algorithm is, in expectation, $\tilde{O}(n^{7/4}\log \Delta)$.To the best of our knowledge, our algorithm is the first one to achieve a sub-quadratic execution time even for stochastic point sets with real-valued coordinates.Our algorithm extends to any dimension $d$, where it runs in $\tilde{O}(n^{2-\frac{1}{2d}}\Phi(n))$ time for stochastic point sets $A$ and $B$; here $\Phi(n)$ is the query/update time of a dynamic weighted nearest neighbor data structure. Our algorithm can be seen as a careful adaptation of the Hungarian method in the geometric divide-and-conquer framework.</details>

### [Spotlight] A Scalable Neural Network for DSIC Affine Maximizer Auction Design

**Authors:** Zhijian Duan, Haoran Sun, Yurong Chen, Xiaotie Deng

**<details><summary>Abstract**</summary>

Automated auction design aims to find empirically high-revenue mechanisms through machine learning. Existing works on multi item auction scenarios can be roughly divided into RegretNet-like and affine maximizer auctions (AMAs) approaches. However, the former cannot strictly ensure dominant strategy incentive compatibility (DSIC), while the latter faces scalability issue due to the large number of allocation candidates. To address these limitations, we propose AMenuNet, a scalable neural network that constructs the AMA parameters (even including the allocation menu) from bidder and item representations. AMenuNet is always DSIC and individually rational (IR) due to the properties of AMAs, and it enhances scalability by generating candidate allocations through a neural network. Additionally, AMenuNet is permutation equivariant, and its number of parameters is independent of auction scale. We conduct extensive experiments to demonstrate that AMenuNet outperforms strong baselines in both contextual and non-contextual multi-item auctions, scales well to larger auctions, generalizes well to different settings, and identifies useful deterministic allocations. Overall, our proposed approach offers an effective solution to automated DSIC auction design, with improved scalability and strong revenue performance in various settings.</details>

### [Oral] A Single-Loop Accelerated Extra-Gradient Difference Algorithm with Improved Complexity Bounds for Constrained Minimax Optimization

**Authors:** Yuanyuan Liu, Fanhua Shang, Weixin An, Junhao Liu, Hongying Liu, Zhouchen Lin

**Oral Presentation:** We, Dec 13, 13:30 -- Oral 4A

**<details><summary>Abstract**</summary>

In this paper, we propose a novel extra-gradient difference acceleration algorithm for solving constrained nonconvex-nonconcave (NC-NC) minimax problems. In particular, we design a new extra-gradient difference step to obtain an important quasi-cocoercivity property, which plays a key role to significantly improve the convergence rate in the constrained NC-NC setting without additional structural assumption. Then momentum acceleration is also introduced into our dual accelerating update step. Moreover, we prove that, to find an $\epsilon$-stationary point of the function $f$, our algorithm attains the complexity $\mathcal{O}(\epsilon^{-2})$ in the constrained NC-NC setting, while the best-known complexity bound is $\widetilde{\mathcal{O}}(\epsilon^{-4})$, where $\widetilde{\mathcal{O}}(\cdot)$ hides logarithmic factors compared to $\mathcal{O}(\cdot)$. As the special cases of the constrained NC-NC setting, our algorithm can also obtain the same complexity $\mathcal{O}(\epsilon^{-2})$ for both the nonconvex-concave (NC-C) and convex-nonconcave (C-NC) cases, while the best-known complexity bounds are $\widetilde{\mathcal{O}}(\epsilon^{-2.5})$ for the NC-C case and $\widetilde{\mathcal{O}}(\epsilon^{-4})$ for the C-NC case. For fair comparison with existing algorithms, we also analyze the complexity bound to find $\epsilon$-stationary point of the primal function $\phi$ for the constrained NC-C problem, which shows that our algorithm can improve the complexity bound from $\widetilde{\mathcal{O}}(\epsilon^{-3})$ to $\mathcal{O}(\epsilon^{-2})$. To the best of our knowledge, this is the first time that the proposed algorithm improves the best-known complexity bounds from $\mathcal{O}(\epsilon^{-4})$ and $\widetilde{\mathcal{O}}(\epsilon^{-3})$ to $\mathcal{O}(\epsilon^{-2})$ in both the NC-NC and NC-C settings.</details>

### A Unified Algorithm Framework for Unsupervised Discovery of Skills based on Determinantal Point Process

**Authors:** Jiayu Chen, Vaneet Aggarwal, Tian Lan

**<details><summary>Abstract**</summary>

Learning rich skills under the option framework without supervision of external rewards is at the frontier of reinforcement learning research. Existing works mainly fall into two distinctive categories: variational option discovery that maximizes the diversity of the options through a mutual information loss (while ignoring coverage) and Laplacian-based methods that focus on improving the coverage of options by increasing connectivity of the state space (while ignoring diversity). In this paper, we show that diversity and coverage in unsupervised option discovery can indeed be unified under the same mathematical framework. To be specific, we explicitly quantify the diversity and coverage of the learned options through a novel use of Determinantal Point Process (DPP) and optimize these objectives to discover options with both superior diversity and coverage. Our proposed algorithm, ODPP, has undergone extensive evaluation on challenging tasks created with Mujoco and Atari. The results demonstrate that our algorithm outperforms state-of-the-art baselines in both diversity- and coverage-driven categories.</details>

### A Unified Conditional Framework for Diffusion-based Image Restoration

**Authors:** Yi Zhang, Xiaoyu Shi, Dasong Li, Xiaogang Wang, Jian Wang, Hongsheng Li

**<details><summary>Abstract**</summary>

Diffusion Probabilistic Models (DPMs) have recently shown remarkable performance in image generation tasks, which are capable of generating highly realistic images. When adopting DPMs for image restoration tasks, the crucial aspect lies in how to integrate the conditional information to guide the DPMs to generate accurate and natural output, which has been largely overlooked in existing works. In this paper, we present a unified conditional framework based on diffusion models for image restoration. We leverage a lightweight UNet to predict initial guidance and the diffusion model to learn the residual of the guidance. By carefully designing the basic module and integration module for the diffusion model block, we integrate the guidance and other auxiliary conditional information into every block of the diffusion model to achieve spatially-adaptive generation conditioning. To handle high-resolution images, we propose a simple yet effective inter-step patch-splitting strategy to produce arbitrary-resolution images without grid artifacts. We evaluate our conditional framework on three challenging tasks: extreme low-light denoising, deblurring, and JPEG restoration, demonstrating its significant improvements in perceptual quality and the generalization to restoration tasks. The code will be released at https://zhangyi-3.github.io/project/UCDIR/.</details>

### A Unified Framework for Rank-based Loss Minimization

**Authors:** Rufeng Xiao, Yuze Ge, Rujun Jiang, Yifan Yan

**<details><summary>Abstract**</summary>

The empirical loss, commonly referred to as the average loss, is extensively utilized for training machine learning models. However, in order to address the diverse performance requirements of machine learning models, the use of the rank-based loss is prevalent, replacing the empirical loss in many cases. The rank-based loss comprises a weighted sum of sorted individual losses, encompassing both convex losses like the spectral risk, which includes the empirical risk and conditional value-at-risk, and nonconvex losses such as the human-aligned risk and the sum of the ranked range loss. In this paper, we introduce a unified framework for the optimization of the rank-based loss through the utilization of a proximal alternating direction method of multipliers. We demonstrate the convergence and convergence rate of the proposed algorithm under mild conditions. Experiments conducted on synthetic and real datasets illustrate the effectiveness and efficiency of the proposed algorithm.</details>

### [Spotlight] A Unified Generalization Analysis of Re-Weighting and Logit-Adjustment for Imbalanced Learning

**Authors:** Zitai Wang, Qianqian Xu, Zhiyong Yang, Yuan He, Xiaochun Cao, Qingming Huang

**<details><summary>Abstract**</summary>

Real-world datasets are typically imbalanced in the sense that only a few classes have numerous samples, while many classes are associated with only a few samples. As a result, a naive ERM learning process will be biased towards the majority classes, making it difficult to generalize to the minority classes. To address this issue, one simple but effective approach is to modify the loss function to emphasize the learning on minority classes, such as re-weighting the losses or adjusting the logits via class-dependent terms. However, existing generalization analysis of such losses is still coarse-grained and fragmented, failing to explain some empirical results. To bridge this gap between theory and practice, we propose a novel technique named data-dependent contraction to capture how these modified losses handle different classes. On top of this technique, a fine-grained generalization bound is established for imbalanced learning, which helps reveal the mystery of re-weighting and logit-adjustment in a unified manner. Furthermore, a principled learning algorithm is developed based on the theoretical insights. Finally, the empirical results on benchmark datasets not only validate the theoretical results but also demonstrate the effectiveness of the proposed method.</details>

### A Variational Perspective on High-Resolution ODEs

**Authors:** Hoomaan Maskan, Konstantinos Zygalakis, Alp Yurtsever

**<details><summary>Abstract**</summary>

We consider unconstrained minimization of smooth convex functions. We propose a novel variational perspective using forced Euler-Lagrange equation that allows for studying high-resolution ODEs. Through this, we obtain a faster convergence rate for gradient norm minimization using Nesterov's accelerated gradient method. Additionally, we show that Nesterov's method can be interpreted as a rate-matching discretization of an appropriately chosen high-resolution ODE. Finally, using the results from the new variational perspective, we propose a stochastic method for noisy gradients. Several numerical experiments compare and illustrate our stochastic algorithm with state of the art methods.</details>

### A case for reframing automated medical image classification as segmentation

**Authors:** Sarah Hooper, Mayee Chen, Khaled Saab, Kush Bhatia, Curtis Langlotz, Christopher Ré

**<details><summary>Abstract**</summary>

Image classification and segmentation are common applications of deep learning to radiology. While many tasks can be framed using either classification or segmentation, classification has historically been cheaper to label and more widely used. However, recent work has drastically reduced the cost of training segmentation networks. In light of this recent work, we reexamine the choice of training classification vs. segmentation models. First, we use an information theoretic approach to analyze why segmentation vs. classification models may achieve different performance on the same dataset and overarching task. We then implement multiple methods for using segmentation models to classify medical images, which we call *segmentation-for-classification*, and compare these methods against traditional classification on three retrospective datasets. We use our analysis and experiments to summarize the benefits of switching from segmentation to classification, including: improved sample efficiency, enabling improved performance with fewer labeled images (up to an order of magnitude lower), on low-prevalence classes, and on certain rare subgroups (up to 161.1\% improved recall); improved robustness to spurious correlations (up to 44.8\% improved robust AUROC); and improved model interpretability, evaluation, and error analysis.</details>

### A*Net: A Scalable Path-based Reasoning Approach for Knowledge Graphs

**Authors:** Zhaocheng Zhu, Xinyu Yuan, Michael Galkin, Louis-Pascal Xhonneux, Ming Zhang, Maxime Gazeau, Jian Tang

**<details><summary>Abstract**</summary>

Reasoning on large-scale knowledge graphs has been long dominated by embedding methods. While path-based methods possess the inductive capacity that embeddings lack, their scalability is limited by the exponential number of paths. Here we present A\*Net, a scalable path-based method for knowledge graph reasoning. Inspired by the A\* algorithm for shortest path problems, our A\*Net learns a priority function to select important nodes and edges at each iteration, to reduce time and memory footprint for both training and inference. The ratio of selected nodes and edges can be specified to trade off between performance and efficiency. Experiments on both transductive and inductive knowledge graph reasoning benchmarks show that A\*Net achieves competitive performance with existing state-of-the-art path-based methods, while merely visiting 10% nodes and 10% edges at each iteration. On a million-scale dataset ogbl-wikikg2, A\*Net not only achieves a new state-of-the-art result, but also converges faster than embedding methods. A\*Net is the first path-based method for knowledge graph reasoning at such scale.</details>

### A-NeSI: A Scalable Approximate Method for Probabilistic Neurosymbolic Inference

**Authors:** Emile van Krieken, Thiviyan Thanapalasingam, Jakub Tomczak, Frank van Harmelen, Annette Ten Teije

**<details><summary>Abstract**</summary>

We study the problem of combining neural networks with symbolic reasoning. Recently introduced frameworks for Probabilistic Neurosymbolic Learning (PNL), such as DeepProbLog, perform exponential-time exact inference, limiting the scalability of PNL solutions. We introduce Approximate Neurosymbolic Inference (A-NeSI): a new framework for PNL that uses neural networks for scalable approximate inference. A-NeSI 1) performs approximate inference in polynomial time without changing the semantics of probabilistic logics; 2) is trained using data generated by the background knowledge; 3) can generate symbolic explanations of predictions; and 4) can guarantee the satisfaction of logical constraints at test time, which is vital in safety-critical applications. Our experiments show that A-NeSI is the first end-to-end method to solve three neurosymbolic tasks with exponential combinatorial scaling. Finally, our experiments show that A-NeSI achieves explainability and safety without a penalty in performance.</details>

### A3FL: Adversarially Adaptive Backdoor Attacks to Federated Learning

**Authors:** Hangfan Zhang, Jinyuan Jia, Jinghui Chen, Lu Lin, Dinghao Wu

**<details><summary>Abstract**</summary>

Federated Learning (FL) is a distributed machine learning paradigm that allows multiple clients to train a global model collaboratively without sharing their local training data. Due to its distributed nature, many studies have shown that it is vulnerable to backdoor attacks. However, existing studies usually used a predetermined, fixed backdoor trigger or optimized it based solely on the local data and model without considering the global training dynamics. This leads to sub-optimal and less durable attack effectiveness, i.e., their attack success rate is low when the attack budget is limited and decreases quickly if the attacker can no longer perform attacks anymore. To address these limitations, we propose A3FL, a new backdoor attack which adversarially adapts the backdoor trigger to make it less likely to be removed by the global training dynamics. Our key intuition is that the difference between the global model and the local model in FL makes the local-optimized trigger much less effective when transferred to the global model. We solve this by optimizing the trigger to even survive the worst-case scenario where the global model was trained to directly unlearn the trigger. Extensive experiments on benchmark datasets are conducted for twelve existing defenses to comprehensively evaluate the effectiveness of our A3FL. Our code is available at https://github.com/hfzhang31/A3FL.</details>

### ANPL: Towards Natural Programming with Interactive Decomposition

**Authors:** Di Huang, Ziyuan Nan, Xing Hu, Pengwei Jin, Shaohui Peng, Yuanbo Wen, Rui Zhang, Zidong Du, Qi Guo, Yewen Pu, Yunji Chen

**<details><summary>Abstract**</summary>

Though LLMs are capable of generating plausible programs, it’s challenging to interact with the LLMs further to revise the program, especially if the user’s specific requirements are different from the initial proposal. In this paper, we introduce ANPL, an interactive programming system that ensures users can always refine the generated code towards their specific programmatic intents via structureddecompositions. Borrowing the paradigm of sketching from program synthesis, an ANPL program consists of a set of input-outputs that it must satisfy, a “sketch” — control/data flow expressed in precise code (e.g. Python), and “holes” — sub-modules to be implemented by the LLM specified with natural language. The user revises an ANPL program by either modifying the sketch, changing the language used to describe the holes, or providing additional input-outputs to a particular hole, turning it into a sub-ANPL program that can be solved recursively. This workflow allows the users to offload programming burdens to the LLM as much as possible while retaining the ability to pinpoint and resolve bugs locally, without exposing the rest of the program to the LLM. We deploy ANPL on the Abstraction and Reasoning Corpus (ARC), a set of unique tasks that are challenging for state-of-the-art AI systems, showing it outperforms baseline programming systems that (a) without the ability to decompose tasks interactively and (b) without the guarantee that the modules can be correctly composed together. Additional evaluations on APPS, HumanEval, and real-world programming tasks have validated that the ANPL framework is applicable to multiple programming domains. We release the ANPL solutions to the ARC tasks as a dataset, providing insights into how humans decompose novel tasks programmatically.</details>

### [Spotlight] ARTree: A Deep Autoregressive Model for Phylogenetic Inference

**Authors:** Tianyu Xie, Cheng Zhang

**<details><summary>Abstract**</summary>

Designing flexible probabilistic models over tree topologies is important for developing efficient phylogenetic inference methods. To do that, previous works often leverage the similarity of tree topologies via hand-engineered heuristic features which would require domain expertise and may suffer from limited approximation capability. In this paper, we propose a deep autoregressive model for phylogenetic inference based on graph neural networks (GNNs), called ARTree. By decomposing a tree topology into a sequence of leaf node addition operations and modeling the involved conditional distributions based on learnable topological features via GNNs, ARTree can provide a rich family of distributions over tree topologies that have simple sampling algorithms, without using heuristic features. We demonstrate the effectiveness and efficiency of our method on a benchmark of challenging real data tree topology density estimation and variational Bayesian phylogenetic inference problems.</details>

### ASIF: Coupled Data Turns Unimodal Models to Multimodal without Training

**Authors:** Antonio Norelli, Marco Fumero, Valentino Maiorca, Luca Moschella, Emanuele Rodolà, Francesco Locatello

**<details><summary>Abstract**</summary>

CLIP proved that aligning visual and language spaces is key to solving many vision tasks without explicit training, but required to train image and text encoders from scratch on a huge dataset. LiT improved this by only training the text encoder and using a pre-trained vision network. In this paper, we show that a common space can be created without any training at all, using single-domain encoders (trained with or without supervision) and a much smaller amount of image-text pairs. Furthermore, our model has unique properties. Most notably, deploying a new version with updated training samples can be done in a matter of seconds. Additionally, the representations in the common space are easily interpretable as every dimension corresponds to the similarity of the input to a unique entry in the multimodal dataset. Experiments on standard zero-shot visual benchmarks demonstrate the typical transfer ability of image-text models. Overall, our method represents a simple yet surprisingly strong baseline for foundation multi-modal models, raising important questions on their data efficiency and on the role of retrieval in machine learning.</details>

### ATTA: Anomaly-aware Test-Time Adaptation for Out-of-Distribution Detection in Segmentation

**Authors:** Zhitong Gao, Shipeng Yan, Xuming He

**<details><summary>Abstract**</summary>

Recent advancements in dense out-of-distribution (OOD) detection have primarily focused on scenarios where the training and testing datasets share a similar domain, with the assumption that no domain shift exists between them. However, in real-world situations, domain shift often exits and significantly affects the accuracy of existing out-of-distribution (OOD) detection models. In this work, we propose a dual-level OOD detection framework to handle domain shift and semantic shift jointly. The first level distinguishes whether domain shift exists in the image by leveraging global low-level features, while the second level identifies pixels with semantic shift by utilizing dense high-level feature maps. In this way, we can selectively adapt the model to unseen domains as well as enhance model's capacity in detecting novel classes. We validate the efficacy of our proposed method on several OOD segmentation benchmarks, including those with significant domain shifts and those without, observing consistent performance improvements across various baseline models. Code is available at https://github.com/gaozhitong/ATTA.</details>

### AVIDa-hIL6: A Large-Scale VHH Dataset Produced from an Immunized Alpaca for Predicting Antigen-Antibody Interactions

**Authors:** Hirofumi Tsuruta, Hiroyuki Yamazaki, Ryota Maeda, Ryotaro Tamura, Jennifer Wei, Zelda Mariet, Poomarin Phloyphisut, Hidetoshi Shimokawa, Joseph R. Ledsam, Lucy Colwell, Akihiro Imura

**<details><summary>Abstract**</summary>

Antibodies have become an important class of therapeutic agents to treat human diseases.To accelerate therapeutic antibody discovery, computational methods, especially machine learning, have attracted considerable interest for predicting specific interactions between antibody candidates and target antigens such as viruses and bacteria.However, the publicly available datasets in existing works have notable limitations, such as small sizes and the lack of non-binding samples and exact amino acid sequences.To overcome these limitations, we have developed AVIDa-hIL6, a large-scale dataset for predicting antigen-antibody interactions in the variable domain of heavy chain of heavy chain antibodies (VHHs), produced from an alpaca immunized with the human interleukin-6 (IL-6) protein, as antigens.By leveraging the simple structure of VHHs, which facilitates identification of full-length amino acid sequences by DNA sequencing technology, AVIDa-hIL6 contains 573,891 antigen-VHH pairs with amino acid sequences.All the antigen-VHH pairs have reliable labels for binding or non-binding, as generated by a novel labeling method.Furthermore, via introduction of artificial mutations, AVIDa-hIL6 contains 30 different mutants in addition to wild-type IL-6 protein.This characteristic provides opportunities to develop machine learning models for predicting changes in antibody binding by antigen mutations.We report experimental benchmark results on AVIDa-hIL6 by using machine learning models.The results indicate that the existing models have potential, but further research is needed to generalize them to predict effective antibodies against unknown mutants.The dataset is available at https://avida-hil6.cognanous.com.</details>

### Accelerating Value Iteration with Anchoring

**Authors:** Jongmin Lee, Ernest Ryu

**<details><summary>Abstract**</summary>

Value Iteration (VI) is foundational to the theory and practice of modern reinforcement learning, and it is known to converge at a $\mathcal{O}(\gamma^k)$-rate. Surprisingly, however, the optimal rate for the VI setup was not known, and finding a general acceleration mechanism has been an open problem. In this paper, we present the first accelerated VI for both the Bellman consistency and optimality operators. Our method, called Anc-VI, is based on an \emph{anchoring} mechanism (distinct from Nesterov's acceleration), and it reduces the Bellman error faster than standard VI. In particular, Anc-VI exhibits a $\mathcal{O}(1/k)$-rate for $\gamma\approx 1$ or even $\gamma=1$, while standard VI has rate $\mathcal{O}(1)$ for $\gamma\ge 1-1/k$, where $k$ is the iteration count. We also provide a complexity lower bound matching the upper bound up to a constant factor of $4$, thereby establishing optimality of the accelerated rate of Anc-VI. Finally, we show that the anchoring mechanism provides the same benefit in the approximate VI and Gauss--Seidel VI setups as well.</details>

### Accessing Higher Dimensions for Unsupervised Word Translation

**Authors:** Sida Wang

**<details><summary>Abstract**</summary>

The striking ability of unsupervised word translation has been demonstrated recently with the help of low-dimensional word vectors / pretraining, which is used by all successful methods and assumed to be necessary. We test and challenge this assumption by developing a method that can also make use of high dimensional signal. Freed from the limits of low dimensions, we show that relying on low-dimensional vectors and their incidental properties miss out on better denoising methods and signals in high dimensions, thus stunting the potential of the data. Our results show that unsupervised translation can be achieved more easily and robustly than previously thought -- less than 80MB and minutes of CPU time is required to achieve over 50\% accuracy for English to Finnish, Hungarian, and Chinese translations when trained in the same domain; even under domain mismatch, the method still works fully unsupervised on English NewsCrawl to Chinese Wikipedia and English Europarl to Spanish Wikipedia, among others. These results challenge prevailing assumptions on the necessity and superiority of low-dimensional vectors and show that the higher dimension signal can be used rather than thrown away.</details>

### Action Inference by Maximising Evidence: Zero-Shot Imitation from Observation with World Models

**Authors:** Xingyuan Zhang, Philip Becker-Ehmck, Patrick van der Smagt, Maximilian Karl

**<details><summary>Abstract**</summary>

Unlike most reinforcement learning agents which require an unrealistic amount of environment interactions to learn a new behaviour, humans excel at learning quickly by merely observing and imitating others. This ability highly depends on the fact that humans have a model of their own embodiment that allows them to infer the most likely actions that led to the observed behaviour. In this paper, we propose Action Inference by Maximising Evidence (AIME) to replicate this behaviour using world models. AIME consists of two distinct phases. In the first phase, the agent learns a world model from its past experience to understand its own body by maximising the ELBO. While in the second phase, the agent is given some observation-only demonstrations of an expert performing a novel task and tries to imitate the expert's behaviour. AIME achieves this by defining a policy as an inference model and maximising the evidence of the demonstration under the policy and world model. Our method is "zero-shot" in the sense that it does not require further training for the world model or online interactions with the environment after given the demonstration. We empirically validate the zero-shot imitation performance of our method on the Walker and Cheetah embodiment of the DeepMind Control Suite and find it outperforms the state-of-the-art baselines. Code is available at: https://github.com/argmax-ai/aime.</details>

### Active Reasoning in an Open-World Environment

**Authors:** Manjie Xu, Guangyuan Jiang, Wei Liang, Chi Zhang, Yixin Zhu

**<details><summary>Abstract**</summary>

Recent advances in vision-language learning have achieved notable success on *complete-information* question-answering datasets through the integration of extensive world knowledge. Yet, most models operate *passively*, responding to questions based on pre-stored knowledge. In stark contrast, humans possess the ability to *actively* explore, accumulate, and reason using both newfound and existing information to tackle *incomplete-information* questions. In response to this gap, we introduce **Conan**, an interactive open-world environment devised for the assessment of *active reasoning*. **Conan** facilitates active exploration and promotes multi-round abductive inference, reminiscent of rich, open-world settings like Minecraft. Diverging from previous works that lean primarily on single-round deduction via instruction following, **Conan** compels agents to actively interact with their surroundings, amalgamating new evidence with prior knowledge to elucidate events from incomplete observations. Our analysis on \bench underscores the shortcomings of contemporary state-of-the-art models in active exploration and understanding complex scenarios. Additionally, we explore *Abduction from Deduction*, where agents harness Bayesian rules to recast the challenge of abduction as a deductive process. Through **Conan**, we aim to galvanize advancements in active reasoning and set the stage for the next generation of artificial intelligence agents adept at dynamically engaging in environments.</details>

### Active representation learning for general task space with applications in robotics

**Authors:** Yifang Chen, Yingbing Huang, Simon Du, Kevin Jamieson, Guanya Shi

**<details><summary>Abstract**</summary>

Representation learning based on multi-task pretraining has become a powerful approach in many domains. In particular, task-aware representation learning aims to learn an optimal representation for a specific target task by sampling data from a set of source tasks, while task-agnostic representation learning seeks to learn a universal representation for a class of tasks.  In this paper, we propose a general and versatile algorithmic and theoretic framework for \emph{active representation learning}, where the learner optimally chooses which source tasks to sample from. This framework, along with a tractable meta algorithm, allows most arbitrary target and source task spaces (from discrete to continuous), covers both task-aware and task-agnostic settings, and is compatible with deep representation learning practices. We provide several instantiations under this framework, from bilinear and feature-based nonlinear to general nonlinear cases. In the bilinear case, by leveraging the non-uniform spectrum of the task representation and the calibrated source-target relevance, we prove that the sample complexity to achieve $\varepsilon$-excess risk on target scales with $(k^*)^2 ||v^*||_2^2 \varepsilon^{-2}$ where $k^*$ is the effective dimension of the target and $||v^*||_2^2 \in (0,1]$ represents the connection between source and target space. Compared to the passive one, this can save up to $\frac{1}{d_W}$ of sample complexity, where $d_W$ is the task space dimension. Finally, we demonstrate different instantiations of our meta algorithm in synthetic datasets and robotics problems, from pendulum simulations to real-world drone flight datasets. On average, our algorithms outperform baselines by 20%-70%.</details>

### Actively Testing Your Model While It Learns: Realizing Label-Efficient Learning in Practice

**Authors:** Dayou Yu, Weishi Shi, Qi Yu

**<details><summary>Abstract**</summary>

In active learning (AL), we focus on reducing the data annotation cost from the model training perspective. However, "testing'', which often refers to the model evaluation process of using empirical risk to estimate the intractable true generalization risk, also requires data annotations. The annotation cost for "testing'' (model evaluation) is under-explored. Even in works that study active model evaluation or active testing (AT), the learning and testing ends are disconnected. In this paper, we propose a novel active testing while learning (ATL) framework that integrates active learning with active testing. ATL provides an unbiased sample-efficient estimation of the model risk during active learning. It leverages test samples annotated from different periods of a dynamic active learning process to achieve fair model evaluations based on a theoretically guaranteed optimal integration of different test samples. Periodic testing also enables effective early-stopping to further save the total annotation cost. ATL further integrates an "active feedback'' mechanism, which is inspired by human learning, where the teacher (active tester) provides immediate guidance given by the prior performance of the student (active learner). Our theoretical result reveals that active feedback maintains the label complexity of the integrated learning-testing objective, while improving the model's generalization capability. We study the realistic setting where we maximize the performance gain from choosing "testing'' samples for feedback without sacrificing the risk estimation accuracy. An agnostic-style analysis and empirical evaluations on real-world datasets demonstrate that the ATL framework can effectively improve the annotation efficiency of both active learning and evaluation tasks.</details>

### AdANNS: A Framework for Adaptive Semantic Search

**Authors:** Aniket Rege, Aditya Kusupati, Sharan Ranjit S, Alan Fan, Qingqing Cao, Sham Kakade, Prateek Jain, Ali Farhadi

**<details><summary>Abstract**</summary>

Web-scale search systems learn an encoder to embed a given query which is then hooked into an approximate nearest neighbor search (ANNS) pipeline to retrieve similar data points. To accurately capture tail queries and data points, learned representations typically are _rigid, high-dimensional_ vectors that are generally used as-is in the entire ANNS pipeline and can lead to computationally expensive retrieval. In this paper, we argue that instead of rigid representations, different stages of ANNS can leverage _adaptive representations_ of varying capacities to achieve significantly better accuracy-compute trade-offs, i.e., stages of ANNS that can get away with more approximate computation should use a lower-capacity representation of the same data point. To this end, we introduce AdANNS, a novel ANNS design framework that explicitly leverages the flexibility of Matryoshka Representations. We demonstrate state-of-the-art accuracy-compute trade-offs using novel AdANNS-based key ANNS building blocks like search data structures (AdANNS-IVF) and quantization (AdANNS-OPQ). For example on ImageNet retrieval, AdANNS-IVF is up to $\mathbf{1.5}$% more accurate than the rigid representations-based IVF at the same compute budget; and matches accuracy while being up to $\mathbf{90}	imes$ faster in _wall-clock time_. For Natural Questions, $32$-byte AdANNS-OPQ matches the accuracy of the $64$-byte OPQ baseline constructed using rigid representations -- _same accuracy at half the cost!_ We further show that the gains from AdANNS translate to modern-day composite ANNS indices that combine search structures and quantization. Finally, we demonstrate that AdANNS can enable inference-time adaptivity for compute-aware search on ANNS indices built non-adaptively on matryoshka representations. Code is open-sourced at https://github.com/RAIVNLab/AdANNS.</details>

### Adapting to Continuous Covariate Shift via Online Density Ratio Estimation

**Authors:** Yu-Jie Zhang, Zhen-Yu Zhang, Peng Zhao, Masashi Sugiyama

**<details><summary>Abstract**</summary>

Dealing with distribution shifts is one of the central challenges for modern machine learning. One fundamental situation is the covariate shift, where the input distributions of data change from the training to testing stages while the input-conditional output distribution remains unchanged. In this paper, we initiate the study of a more challenging scenario --- continuous covariate shift --- in which the test data appear sequentially, and their distributions can shift continuously. Our goal is to adaptively train the predictor such that its prediction risk accumulated over time can be minimized. Starting with the importance-weighted learning, we theoretically show the method works effectively if the time-varying density ratios of test and train inputs can be accurately estimated. However, existing density ratio estimation methods would fail due to data scarcity at each time step. To this end, we propose an online density ratio estimation method that can appropriately reuse historical information. Our method is proven to perform well by enjoying a dynamic regret bound, which finally leads to an excess risk guarantee for the predictor. Empirical results also validate the effectiveness.</details>

### Adaptive SGD with Polyak stepsize and Line-search: Robust Convergence and Variance Reduction

**Authors:** Xiaowen Jiang, Sebastian Stich

**<details><summary>Abstract**</summary>

The recently proposed stochastic Polyak stepsize (SPS) and stochastic line-search (SLS) for SGD have shown remarkable effectiveness when training over-parameterized models. However, two issues remain unsolved in this line of work. First, in non-interpolation settings, both algorithms only guarantee convergence to a neighborhood of a solution which may result in a worse output than the initial guess. While artificially decreasing the adaptive stepsize has been proposed to address this issue (Orvieto et al.), this approach results in slower convergence rates under interpolation. Second, intuitive line-search methods equipped with variance-reduction (VR) fail to converge (Dubois-Taine et al.). So far, no VR methods successfully accelerate these two stepsizes with a convergence guarantee.In this work, we make two contributions:Firstly, we propose two new robust variants of SPS and SLS, called AdaSPS and AdaSLS, which achieve optimal asymptotic rates in both strongly-convex or convex and interpolation or non-interpolation settings, except for the case when we have both strong convexity and non-interpolation. AdaSLS requires no knowledge of problem-dependent parameters, and AdaSPS requires only a lower bound of the optimal function value as input. Secondly, we propose a novel VR method that can use Polyak stepsizes or line-search to achieve acceleration. When it is equipped with AdaSPS or AdaSLS, the resulting algorithms obtain the optimal ratefor optimizing convex smooth functions. Finally, numerical experiments on synthetic and real datasets validate our theory and demonstrate the effectiveness and robustness of our algorithms.</details>

### Adversarial Robustness through Random Weight Sampling

**Authors:** Yanxiang Ma, Minjing Dong, Chang Xu

**<details><summary>Abstract**</summary>

Deep neural networks have been found to be vulnerable in a variety of tasks. Adversarial attacks can manipulate network outputs, resulting in incorrect predictions. Adversarial defense methods aim to improve the adversarial robustness of networks by countering potential attacks. In addition to traditional defense approaches, randomized defense mechanisms have recently received increasing attention from researchers. These methods introduce different types of perturbations during the inference phase to destabilize adversarial attacks.Although promising empirical results have been demonstrated by these approaches, the defense performance is quite sensitive to the randomness parameters, which are always manually tuned without further analysis. On the contrary, we propose incorporating random weights into the optimization to fully exploit the potential of randomized defense. To perform better optimization of randomness parameters, we conduct a theoretical analysis of the connections between randomness parameters and gradient similarity as well as natural performance. From these two aspects, we suggest imposing theoretically-guided constraints on random weights during optimizations, as these weights play a critical role in balancing natural performance and adversarial robustness. We derive both the upper and lower bounds of random weight parameters by considering prediction bias and gradient similarity. In this study, we introduce the Constrained Trainable Random Weight (CTRW), which adds random weight parameters to the optimization and includes a constraint guided by the upper and lower bounds to achieve better trade-offs between natural and robust accuracy. We evaluate the effectiveness of CTRW on several datasets and benchmark convolutional neural networks. Our results indicate that our model achieves a robust accuracy approximately 16% to 17% higher than the baseline model under PGD-20 and 22% to 25% higher on Auto Attack.</details>

### Agnostic Multi-Group Active Learning

**Authors:** Nicholas Rittler, Kamalika Chaudhuri

**<details><summary>Abstract**</summary>

Inspired by the problem of improving classification accuracy on rare or hard subsets of a population, there has been recent interest in models of learning where the goal is to generalize to a collection of distributions, each representing a ``group''. We consider a variant of this problem from the perspective of active learning, where the learner is endowed with the power to decide which examples are labeled from each distribution in the collection, and the goal is to minimize the number of label queries while maintaining PAC-learning guarantees. Our main challenge is that standard active learning techniques such as disagreement-based active learning do not directly apply to the multi-group learning objective. We modify existing algorithms to provide a consistent active learning algorithm for an agnostic formulation of multi-group learning, which given a collection of $G$ distributions and a hypothesis class $\mathcal{H}$ with VC-dimension $d$, outputs an $\epsilon$-optimal hypothesis using $\tilde{O}\left( (\nu^2/\epsilon^2) G d  \theta_{\mathcal{G}}^2 \log^2(1/\epsilon) + G\log(1/\epsilon)/\epsilon^2 \right)$ label queries, where $\theta_{\mathcal{G}}$ is the worst-case disagreement coefficient over the collection. Roughly speaking, this guarantee improves upon the label complexity of standard multi-group learning in regimes where disagreement-based active learning algorithms may be expected to succeed, and the number of groups is not too large. We also consider the special case where each distribution in the collection is individually realizable with respect to $\mathcal{H}$, and demonstrate $\tilde{O}\left( G d \theta_{\mathcal{G}} \log(1/\epsilon) \right)$ label queries are sufficient for learning in this case. We further give an approximation result for the full agnostic case inspired by the group realizable strategy.</details>

### Aiming towards the minimizers: fast convergence of SGD for overparametrized problems

**Authors:** Chaoyue Liu, Dmitriy Drusvyatskiy, Misha Belkin, Damek Davis, Yian Ma

**<details><summary>Abstract**</summary>

Modern machine learning paradigms, such as deep learning, occur in or close to the interpolation regime, wherein the number of model parameters is much larger than the number of data samples. In this work, we propose a regularity condition within the interpolation regime which endows the stochastic gradient method with the same worst-case iteration complexity as the deterministic gradient method, while using only a single sampled gradient (or a minibatch) in each iteration. In contrast, all existing guarantees require the stochastic gradient method to take small steps, thereby resulting in a much slower linear rate of convergence. Finally, we demonstrate that our condition holds when training sufficiently wide feedforward neural networks with a linear output layer.</details>

### [Spotlight] Aleatoric and Epistemic Discrimination: Fundamental Limits of Fairness Interventions

**Authors:** Hao Wang, Luxi He, Rui Gao, Flavio Calmon

**<details><summary>Abstract**</summary>

Machine learning (ML) models can underperform on certain population groups due to choices made during model development and bias inherent in the data. We categorize sources of discrimination in the ML pipeline into two classes: aleatoric discrimination, which is inherent in the data distribution, and epistemic discrimination, which is due to decisions made during model development. We quantify aleatoric discrimination by determining the performance limits of a model under fairness constraints, assuming perfect knowledge of the data distribution. We demonstrate how to characterize aleatoric discrimination by applying Blackwell's results on comparing statistical experiments. We then quantify epistemic discrimination as the gap between a model's accuracy when fairness constraints are applied and the limit posed by aleatoric discrimination. We apply this approach to benchmark existing fairness interventions and investigate fairness risks in data with missing values. Our results indicate that state-of-the-art fairness interventions are effective at removing epistemic discrimination on standard (overused) tabular datasets. However, when data has missing values, there is still significant room for improvement in handling aleatoric discrimination.</details>

### An $\varepsilon$-Best-Arm Identification Algorithm for Fixed-Confidence and Beyond

**Authors:** Marc Jourdan, Rémy Degenne, Emilie Kaufmann

**<details><summary>Abstract**</summary>

We propose EB-TC$\varepsilon$, a novel sampling rule for $\varepsilon$-best arm identification in stochastic bandits.It is the first instance of Top Two algorithm analyzed for approximate best arm identification. EB-TC$\varepsilon$ is an *anytime* sampling rule that can therefore be employed without modification for fixed confidence or fixed budget identification (without prior knowledge of the budget).We provide three types of theoretical guarantees for EB-TC$\varepsilon$.First, we prove bounds on its expected sample complexity in the fixed confidence setting, notably showing its asymptotic optimality in combination with an adaptive tuning of its exploration parameter.We complement these findings with upper bounds on its probability of error at any time and for any slack parameter, which further yield upper bounds on its simple regret at any time.Finally, we show through numerical simulations that EB-TC$\varepsilon$ performs favorably compared to existing algorithms for different approximate best arm identification tasks.</details>

### An Efficient Doubly-Robust Test for the Kernel Treatment Effect

**Authors:** Diego Martinez Taboada, Aaditya Ramdas, Edward Kennedy

**<details><summary>Abstract**</summary>

The average treatment effect, which is the difference in expectation of the counterfactuals, is probably the most popular target effect in causal inference with binary treatments. However, treatments may have effects beyond the mean, for instance decreasing or increasing the variance. We propose a new kernel-based test for distributional effects of the treatment. It is, to the best of our knowledge, the first kernel-based, doubly-robust test with provably valid type-I error. Furthermore, our proposed algorithm is computationally efficient, avoiding the use of permutations.</details>

### An Empirical Study Towards Prompt-Tuning for Graph Contrastive Pre-Training in Recommendations

**Authors:** Haoran Yang, Xiangyu Zhao, Yicong Li, Hongxu Chen, Guandong Xu

**<details><summary>Abstract**</summary>

Graph contrastive learning (GCL) has emerged as a potent technology for numerous graph learning tasks. It has been successfully applied to real-world recommender systems, where the contrastive loss and the downstream recommendation objectives are always combined to form the overall objective function. Such a strategy is inconsistent with the original GCL paradigm, where graph embeddings are pre-trained without involving downstream training objectives. In this paper, we innovatively propose a prompt-enhanced framework for GCL-based recommender systems, namely CPTPP, which can fully leverage the advantages of the original GCL protocol through prompt tuning. Specifically, we first summarise user profiles in graph recommender systems to automatically generate personalized user prompts. These prompts will then be combined with pre-trained user embeddings to conduct prompt-tuning in downstream tasks, thereby narrowing the distinct targets between pre-training and downstream tasks. Extensive experiments on three benchmark datasets validate the effectiveness of CPTPP against state-of-the-art baselines. A further visualization experiment demonstrates that user embeddings generated by CPTPP have a more uniform distribution, indicating a better capacity to model the diversity of user preferences.The implementation code is available online to ease reproducibility: https://anonymous.4open.science/r/CPTPP-F8F4</details>

### An Inductive Bias for Tabular Deep Learning

**Authors:** Ege Beyazit, Jonathan Kozaczuk, Bo Li, Vanessa Wallace, Bilal Fadlallah

**<details><summary>Abstract**</summary>

Deep learning methods have achieved state-of-the-art performance in most modeling tasks involving images, text and audio, however, they typically underperform tree-based methods on tabular data. In this paper, we hypothesize that a significant contributor to this performance gap is the interaction between irregular target functions resulting from the heterogeneous nature of tabular feature spaces, and the well-known tendency of neural networks to learn smooth functions. Utilizing tools from spectral analysis, we show that functions described by tabular datasets often have high irregularity, and that they can be smoothed by transformations such as scaling and ranking in order to improve performance. However, because these transformations tend to lose information or negatively impact the loss landscape during optimization, they need to be rigorously fine-tuned for each feature to achieve performance gains. To address these problems, we propose introducing frequency reduction as an inductive bias. We realize this bias as a neural network layer that promotes learning low-frequency representations of the input features, allowing the network to operate in a space where the target function is more regular. Our proposed method introduces less computational complexity than a fully connected layer, while significantly improving neural network performance, and speeding up its convergence on 14 tabular datasets.</details>

### An Optimal Structured Zeroth-order Algorithm for Non-smooth Optimization

**Authors:** Marco Rando, Cesare Molinari, Lorenzo Rosasco, Silvia Villa

**<details><summary>Abstract**</summary>

Finite-difference methods are a class of algorithms designed to solve black-box optimization problems by approximating a gradient of the target function on a set of directions. In black-box optimization, the non-smooth setting is particularly relevant since, in practice, differentiability and smoothness assumptions cannot be verified. To cope with nonsmoothness, several authors use a smooth approximation of the target function and show that finite difference methods approximate its gradient. Recently, it has been proved that imposing a structure in the directions allows improving performance. However, only the smooth setting was considered. To close this gap, we introduce and analyze O-ZD, the first structured finite-difference algorithm for non-smooth black-box optimization. Our method exploits a smooth approximation of the target function and we prove that it approximates its gradient on a subset of random {\em orthogonal} directions. We analyze the convergence of O-ZD under different assumptions.  For non-smooth convex functions, we obtain the optimal complexity. In the non-smooth non-convex setting, we characterize the number of iterations needed to bound the expected norm of the smoothed gradient. For smooth functions, our analysis recovers existing results for structured zeroth-order methods for the convex case and extends them to the non-convex setting. We conclude with numerical simulations where assumptions are satisfied, observing that our algorithm has very good practical performances.</details>

### Anchor Data Augmentation

**Authors:** Nora Schneider, Shirin Goshtasbpour, Fernando Perez-Cruz

**<details><summary>Abstract**</summary>

We propose a novel algorithm for data augmentation in nonlinear over-parametrized regression. Our data augmentation algorithm borrows from the literature on causality. Contrary to the current state-of-the-art solutions that rely on modifications of Mixup algorithm, we extend the recently proposed distributionally robust Anchor regression (AR) method for data augmentation. Our Anchor Data Augmentation (ADA) uses several replicas of the modified samples in AR to provide more training examples, leading to more robust regression predictions. We apply ADA to linear and nonlinear regression problems using neural networks. ADA is competitive with state-of-the-art C-Mixup solutions.</details>

### [Spotlight] Anonymous and Copy-Robust Delegations for Liquid Democracy

**Authors:** Markus Utke, Ulrike Schmidt-Kraepelin

**<details><summary>Abstract**</summary>

Liquid democracy with ranked delegations is a novel voting scheme that unites the practicability of representative democracy with the idealistic appeal of direct democracy: Every voter decides between casting their vote on a question at hand or delegating their voting weight to some other, trusted agent. Delegations are transitive, and since voters may end up in a delegation cycle, they are encouraged to indicate not only a single delegate, but a set of potential delegates and a ranking among them. Based on the delegation preferences of all voters, a delegation rule selects one representative per voter. Previous work has revealed a trade-off between two properties of delegation rules called anonymity and copy-robustness. To overcome this issue we study two fractional delegation rules: Mixed Borda branching, which generalizes a rule satisfying copy-robustness, and the random walk rule, which satisfies anonymity. Using the Markov chain tree theorem, we show that the two rules are in fact equivalent, and simultaneously satisfy generalized versions of the two properties. Combining the same theorem with Fulkerson's algorithm, we develop  a polynomial-time algorithm for computing the outcome of the studied delegation rule. This algorithm is of independent interest, having applications in semi-supervised learning and graph theory.</details>

### Anytime Model Selection in Linear Bandits

**Authors:** Parnian Kassraie, Nicolas Emmenegger, Andreas Krause, Aldo Pacchiano

**<details><summary>Abstract**</summary>

Model selection in the context of bandit optimization is a challenging problem, as it requires balancing exploration and exploitation not only for action selection, but also for model selection. One natural approach is to rely on online learning algorithms that treat different models as experts. Existing methods, however, scale poorly ($\mathrm{poly}M$) with the number of models $M$ in terms of their regret.Our key insight is that, for model selection in linear bandits, we can emulate full-information feedback to the online learner with a favorable bias-variance trade-off. This allows us to develop ALEXP, which has an exponentially improved ($\log M$) dependence on $M$ for its regret.ALEXP has anytime guarantees on its regret, and neither requires knowledge of the horizon $n$, nor relies on an initial purely exploratory stage.Our approach utilizes a  novel time-uniform analysis of the Lasso, establishing a new connection between online learning and high-dimensional statistics.</details>

### [Spotlight] Approximate Heavy Tails in Offline (Multi-Pass) Stochastic Gradient Descent

**Authors:** Kruno Lehman, Alain Durmus, Umut Simsekli

**<details><summary>Abstract**</summary>

A recent line of empirical studies has demonstrated that SGD might exhibit a heavy-tailed behavior in practical settings, and the heaviness of the tails might correlate with the overall performance. In this paper, we investigate the emergence of such heavy tails. Previous works on this problem only considered, up to our knowledge, online (also called single-pass) SGD, in which the emergence of heavy tails in theoretical findings is contingent upon access to an infinite amount of data. Hence, the underlying mechanism generating the reported heavy-tailed behavior in practical settings, where the amount of training data is finite, is still not well-understood. Our contribution aims to fill this gap. In particular, we show that the stationary distribution of offline (also called multi-pass) SGD exhibits ‘approximate’ power-law tails and the approximation error is controlled by how fast the empirical distribution of the training data converges to the true underlying data distribution in the Wasserstein metric. Our main takeaway is that, as the number of data points increases, offline SGD will behave increasingly ‘power-law-like’. To achieve this result, we first prove nonasymptotic Wasserstein convergence bounds for offline SGD to online SGD as the number of data points increases, which can be interesting on their own. Finally, we illustrate our theory on various experiments conducted on synthetic data and neural networks.</details>

### Asynchrony-Robust Collaborative Perception via Bird's Eye View Flow

**Authors:** Sizhe Wei, Yuxi Wei, Yue Hu, Yifan Lu, Yiqi Zhong, Siheng Chen, Ya Zhang

**<details><summary>Abstract**</summary>

Collaborative perception can substantially boost each agent's perception ability by facilitating communication among multiple agents. However, temporal asynchrony among agents is inevitable in the real world due to communication delays, interruptions, and clock misalignments. This issue causes information mismatch during multi-agent fusion, seriously shaking the foundation of collaboration. To address this issue, we propose CoBEVFlow, an asynchrony-robust collaborative perception system based on bird's eye view (BEV) flow. The key intuition of CoBEVFlow is to compensate motions to align asynchronous collaboration messages sent by multiple agents. To model the motion in a scene, we propose BEV flow, which is a collection of the motion vector corresponding to each spatial location. Based on BEV flow, asynchronous perceptual features can be reassigned to appropriate positions, mitigating the impact of asynchrony. CoBEVFlow has two advantages: (i) CoBEVFlow can handle asynchronous collaboration messages sent at irregular, continuous time stamps without discretization; and (ii) with BEV flow, CoBEVFlow only transports the original perceptual features, instead of generating new perceptual features, avoiding additional noises. To validate CoBEVFlow's efficacy, we create IRregular V2V(IRV2V), the first synthetic collaborative perception dataset with various temporal asynchronies that simulate different real-world scenarios. Extensive experiments conducted on both IRV2V and the real-world dataset DAIR-V2X show that CoBEVFlow consistently outperforms other baselines and is robust in extremely asynchronous settings. The code is available at https://github.com/MediaBrain-SJTU/CoBEVFlow.</details>

### Attacks on Online Learners: a Teacher-Student Analysis

**Authors:** Riccardo Giuseppe Margiotta, Sebastian Goldt, Guido Sanguinetti

**<details><summary>Abstract**</summary>

Machine learning models are famously vulnerable to adversarial attacks: small ad-hoc perturbations of the data that can catastrophically alter the model predictions. While a large literature has studied the case of test-time attacks on pre-trained models, the important case of attacks in an online learning setting has received little attention so far. In this work, we use a control-theoretical perspective to study the scenario where an attacker may perturb data labels to manipulate the learning dynamics of an online learner. We perform a theoretical analysis of the problem in a teacher-student setup, considering different attack strategies, and obtaining analytical results for the steady state of simple linear learners. These results enable us to prove that a discontinuous transition in the learner's accuracy occurs when the attack strength exceeds a critical threshold. We then study empirically attacks on learners with complex architectures using real data, confirming the insights of our theoretical analysis. Our findings show that greedy attacks can be extremely efficient, especially when data stream in small batches.</details>

### [Spotlight] Auditing for Human Expertise

**Authors:** Rohan Alur, Loren Laine, Darrick Li, Manish Raghavan, Devavrat Shah, Dennis Shung

**<details><summary>Abstract**</summary>

High-stakes prediction tasks (e.g., patient diagnosis) are often handled by trained human experts. A common source of concern about automation in these settings is that experts may exercise intuition that is difficult to model and/or have access to information (e.g., conversations with a patient) that is simply unavailable to a would-be algorithm. This raises a natural question whether human experts add value which could not be captured by an algorithmic predictor.We develop a statistical framework under which we can pose this question as a natural hypothesis test. Indeed, as our framework highlights, detecting human expertise is more subtle than simply comparing the accuracy of expert predictions to those made by a particular learning algorithm. Instead, we propose a simple procedure which tests whether expert predictions are statistically independent from the outcomes of interest after conditioning on the available inputs (‘features’). A rejection of our test thus suggests that human experts may add value to any algorithm trained on the available data, and has direct implications for whether human-AI ‘complementarity’ is achievable in a given prediction task.We highlight the utility of our procedure using admissions data collected from the emergency department of a large academic hospital system, where we show that physicians’ admit/discharge decisions for patients with acute gastrointestinal bleeding (AGIB) appear to be incorporating information that is not available to a standard algorithmic screening tool. This is despite the fact that the screening tool is arguably more accurate than physicians’ discretionary decisions, highlighting that – even absent normative concerns about accountability or interpretability – accuracy is insufficient to justify algorithmic automation.</details>

### Autodecoding Latent 3D Diffusion Models

**Authors:** Evangelos Ntavelis, Aliaksandr Siarohin, Kyle Olszewski, Chaoyang Wang, Luc V Gool, Sergey Tulyakov

**<details><summary>Abstract**</summary>

Diffusion-based methods have shown impressive visual results in the text-to-image domain. They first learn a latent space using an autoencoder, then run a denoising process on the bottleneck to generate new samples. However, learning an autoencoder requires substantial data in the target domain. Such data is scarce for 3D generation, prohibiting the learning of large-scale diffusion models for 3D synthesis. We present a novel approach to the generation of static and articulated 3D assets that has a 3D autodecoder at its core. The 3D autodecoder framework embeds properties learned from the target dataset in the latent space, which can then be decoded into a volumetric representation for rendering view-consistent appearance and geometry. We then identify the appropriate intermediate volumetric latent space, and introduce robust normalization and de-normalization operations to learn a 3D diffusion from 2D images or monocular videos of rigid or articulated objects. Our approach is flexible enough to use either existing camera supervision or no camera information at all -- instead efficiently learning it during training. Our evaluations demonstrate that our generation results outperform state-of-the-art alternatives on various benchmark datasets and metrics, including multi-view image datasets of synthetic objects, real in-the-wild videos of moving people, and a large-scale, real video dataset of static objects.</details>

### Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger

**Authors:** Zhiqi Bu, Yu-Xiang Wang, Sheng Zha, George Karypis

**<details><summary>Abstract**</summary>

Per-example gradient clipping is a key algorithmic step that enables practical differential private (DP) training for deep learning models. The choice of clipping threshold $R$, however, is vital for achieving high accuracy under DP. We propose an easy-to-use replacement, called automatic clipping, that eliminates the need to tune $R$ for any DP optimizers, including DP-SGD, DP-Adam, DP-LAMB and many others.The automatic variants are as private and computationally efficient as existing DP optimizers, but require no DP-specific hyperparameters and thus make DP training as amenable as the standard non-private training. We give a rigorous convergence analysis of automatic DP-SGD in the non-convex setting, showing that it can enjoy an asymptotic convergence rate that matches the standard SGD, under a symmetric gradient noise assumption of the per-sample gradients (commonly used in the non-DP literature). We demonstrate on various language and vision tasks that automatic clipping outperforms or matches the state-of-the-art, and can be easily employed with minimal changes to existing codebases.</details>

### Autonomous Capability Assessment of Sequential Decision-Making Systems in Stochastic Settings

**Authors:** Pulkit Verma, Rushang Karia, Siddharth Srivastava

**<details><summary>Abstract**</summary>

It is essential for users to understand what their AI systems can and can't do in order to use them safely. However, the problem of enabling users to assess AI systems with sequential decision-making (SDM) capabilities is relatively understudied. This paper presents a new approach for modeling the capabilities of black-box AI systems that can plan and act, along with the possible effects and requirements for executing those capabilities in stochastic settings. We present an active-learning approach that can effectively interact with a black-box SDM system and learn an interpretable probabilistic model describing its capabilities. Theoretical analysis of the approach identifies the conditions under which the learning process is guaranteed to converge to the correct model of the agent; empirical evaluations on different agents and simulated scenarios show that this approach is few-shot generalizable and can effectively describe the capabilities of arbitrary black-box SDM agents in a sample-efficient manner.</details>

### [Oral] BEDD: The MineRL BASALT Evaluation and Demonstrations Dataset for Training and Benchmarking Agents that Solve Fuzzy Tasks

**Authors:** Stephanie Milani, Anssi Kanervisto, Karolis Ramanauskas, Sander Schulhoff, Brandon Houghton, Rohin Shah

**Oral Presentation:** We, Dec 13, 14:15 -- Oral 4B

**<details><summary>Abstract**</summary>

The MineRL BASALT competition has served to catalyze advances in learning from human feedback through four hard-to-specify tasks in Minecraft, such as create and photograph a waterfall. Given the completion of two years of BASALT competitions, we offer to the community a formalized benchmark through the BASALT Evaluation and Demonstrations Dataset (BEDD), which serves as a resource for algorithm development and performance assessment. BEDD consists of a collection of 26 million image-action pairs from nearly 14,000 videos of human players completing the BASALT tasks in Minecraft. It also includes over 3,000 dense pairwise human evaluations of human and algorithmic agents. These comparisons serve as a fixed, preliminary leaderboard for evaluating newly-developed algorithms. To enable this comparison, we present a streamlined codebase for benchmarking new algorithms against the leaderboard. In addition to presenting these datasets, we conduct a detailed analysis of the data from both datasets to guide algorithm development and evaluation. The released code and data are available at https://github.com/minerllabs/basalt-benchmark.</details>

### BERT Lost Patience Won't Be Robust to Adversarial Slowdown

**Authors:** Zachary Coalson, Gabriel Ritter, Rakesh Bobba, Sanghyun Hong

**<details><summary>Abstract**</summary>

In this paper, we systematically evaluate the robustness of multi-exit language models against adversarial slowdown. To audit their robustness, we design a slowdown attack that generates natural adversarial text bypassing early-exit points. We use the resulting WAFFLE attack as a vehicle to conduct a comprehensive evaluation of three multi-exit mechanisms with the GLUE benchmark against adversarial slowdown. We then show our attack significantly reduces the computational savings provided by the three methods in both white-box and black-box settings. The more complex a mechanism is, the more vulnerable it is to adversarial slowdown. We also perform a linguistic analysis of the perturbed text inputs, identifying common perturbation patterns that our attack generates, and comparing them with standard adversarial text attacks. Moreover, we show that adversarial training is ineffective in defeating our slowdown attack, but input sanitization with a conversational model, e.g., ChatGPT, can remove perturbations effectively. This result suggests that future work is needed for developing efficient yet robust multi-exit models. Our code is available at: https://github.com/ztcoalson/WAFFLE</details>

### BIOT: Biosignal Transformer for Cross-data Learning in the Wild

**Authors:** Chaoqi Yang, M Westover, Jimeng Sun

**<details><summary>Abstract**</summary>

Biological signals, such as electroencephalograms (EEG), play a crucial role in numerous clinical applications, exhibiting diverse data formats and quality profiles. Current deep learning models for biosignals (based on CNN, RNN, and Transformers) are typically specialized for specific datasets and clinical settings, limiting their broader applicability. This paper explores the development of a flexible biosignal encoder architecture that can enable pre-training on multiple datasets and fine-tuned on downstream biosignal tasks with different formats.To overcome the unique challenges associated with biosignals of various formats, such as mismatched channels, variable sample lengths, and prevalent missing val- ues, we propose Biosignal Transformer (BIOT). The proposed BIOT model can enable cross-data learning with mismatched channels, variable lengths, and missing values by tokenizing different biosignals into unified "sentences" structure. Specifically, we tokenize each channel separately into fixed-length segments containing local signal features and then rearrange the segments to form a long "sentence". Channel embeddings and relative position embeddings are added to each segment (viewed as "token") to preserve spatio-temporal features.The BIOT model is versatile and applicable to various biosignal learning settings across different datasets, including joint pre-training for larger models. Comprehensive evaluations on EEG, electrocardiogram (ECG), and human activity sensory signals demonstrate that BIOT outperforms robust baselines in common settings and facilitates learning across multiple datasets with different formats. Using CHB-MIT seizure detection task as an example, our vanilla BIOT model shows 3% improvement over baselines in balanced accuracy, and the pre-trained BIOT models (optimized from other data sources) can further bring up to 4% improvements. Our repository is public at https://github.com/ycq091044/BIOT.</details>

### Back-Modality: Leveraging Modal Transformation for Data Augmentation

**Authors:** Zhi Li, Yifan Liu, Yin Zhang

**<details><summary>Abstract**</summary>

We introduce Back-Modality, a novel data augmentation schema predicated on modal transformation. Data from an initial modality undergoes transformation to an intermediate modality, followed by a reverse transformation. This framework serves dual roles. On one hand, it operates as a general data augmentation strategy. On the other hand, it allows for other augmentation techniques, suitable for the intermediate modality, to enhance the initial modality. For instance, data augmentation methods applicable to pure text can be employed to augment images, thereby facilitating the cross-modality of data augmentation techniques. To validate the viability and efficacy of our framework, we proffer three instantiations of Back-Modality: back-captioning, back-imagination, and back-speech. Comprehensive evaluations across tasks such as image classification, sentiment classification, and textual entailment demonstrate that our methods consistently enhance performance under data-scarce circumstances.</details>

### BadTrack: A Poison-Only Backdoor Attack on Visual Object Tracking

**Authors:** Bin Huang, Jiaqian Yu, Yiwei Chen, Siyang Pan, Qiang Wang, Zhi Wang

**<details><summary>Abstract**</summary>

Visual object tracking (VOT) is one of the most fundamental tasks in computer vision community. State-of-the-art VOT trackers extract positive and negative examples that are used to guide the tracker to distinguish the object from the background. In this paper, we show that this characteristic can be exploited to introduce new threats and hence propose a simple yet effective poison-only backdoor attack. To be specific, we poison a small part of the training data by attaching a predefined trigger pattern to the background region of each video frame, so that the trigger appears almost exclusively in the extracted negative examples. To the best of our knowledge, this is the first work that reveals the threat of poison-only backdoor attack on VOT trackers. We experimentally show that our backdoor attack can significantly degrade the performance of both two-stream Siamese and one-stream Transformer trackers on the poisoned data while gaining comparable performance with the benign trackers on the clean data.</details>

### Bandit Social Learning under Myopic Behavior

**Authors:** Kiarash Banihashem, MohammadTaghi Hajiaghayi, Suho Shin, Aleksandrs Slivkins

**<details><summary>Abstract**</summary>

We study social learning dynamics motivated by reviews on online platforms. Theagents collectively follow a simple multi-armed bandit protocol, but each agentacts myopically, without regards to exploration. We allow a wide range of myopicbehaviors that are consistent with (parameterized) confidence intervals for the arms’expected rewards. We derive stark exploration failures for any such behavior, andprovide matching positive results. As a special case, we obtain the first generalresults on failure of the greedy algorithm in bandits, thus providing a theoreticalfoundation for why bandit algorithms should explore.</details>

### BanditPAM++: Faster $k$-medoids Clustering

**Authors:** Mo Tiwari, Ryan Kang, Donghyun Lee, Sebastian Thrun, Ilan Shomorony, Martin Zhang

**<details><summary>Abstract**</summary>

Clustering is a fundamental task in data science with wide-ranging applications. In $k$-medoids clustering, cluster centers must be actual datapoints and arbitrary distance metrics may be used; these features allow for greater interpretability of the cluster centers and the clustering of exotic objects in $k$-medoids clustering, respectively. $k$-medoids clustering has recently grown in popularity due to the discovery of more efficient $k$-medoids algorithms. In particular, recent research has proposed BanditPAM, a randomized $k$-medoids algorithm with state-of-the-art complexity and clustering accuracy. In this paper, we present BanditPAM++, which accelerates BanditPAM via two algorithmic improvements, and is $O(k)$ faster than BanditPAM in complexity and substantially faster than BanditPAM in wall-clock runtime. First, we demonstrate that BanditPAM has a special structure that allows the reuse of clustering information $	extit{within}$ each iteration. Second, we demonstrate that BanditPAM has additional structure that permits the reuse of information $	extit{across}$ different iterations. These observations inspire our proposed algorithm, BanditPAM++, which returns the same clustering solutions as BanditPAM but often several times faster. For example, on the CIFAR10 dataset, BanditPAM++ returns the same results as BanditPAM but runs over 10$	imes$ faster. Finally, we provide a high-performance C++ implementation of BanditPAM++, callable from Python and R, that may be of interest to practitioners at https://github.com/motiwari/BanditPAM. Auxiliary code to reproduce all of our experiments via a one-line script is available at https://github.com/ThrunGroup/BanditPAM_plusplus_experiments.</details>

### Batch Bayesian Optimization For Replicable Experimental Design

**Authors:** Zhongxiang Dai, Quoc Phong Nguyen, Sebastian Tay, Daisuke Urano, Richalynn Leong, Bryan Kian Hsiang Low, Patrick Jaillet

**<details><summary>Abstract**</summary>

Many real-world experimental design problems (a) evaluate multiple experimental conditions in parallel and (b) replicate each condition multiple times due to large and heteroscedastic observation noise. Given a fixed total budget, this naturally induces a trade-off between evaluating more unique conditions while replicating each of them fewer times vs. evaluating fewer unique conditions and replicating each more times. Moreover, in these problems, practitioners may be risk-averse and hence prefer an input with both good average performance and small variability. To tackle both challenges, we propose the Batch Thompson Sampling for Replicable Experimental Design (BTS-RED) framework, which encompasses three algorithms. Our BTS-RED-Known and BTS-RED-Unknown algorithms, for, respectively, known and unknown noise variance, choose the number of replications adaptively rather than deterministically such that an input with a larger noise variance is replicated more times. As a result, despite the noise heteroscedasticity, both algorithms enjoy a theoretical guarantee and are asymptotically no-regret. Our Mean-Var-BTS-RED algorithm aims at risk-averse optimization and is also asymptotically no-regret. We also show the effectiveness of our algorithms in two practical real-world applications: precision agriculture and AutoML.</details>

### Batchnorm Allows Unsupervised Radial Attacks

**Authors:** Amur Ghose, Apurv Gupta, Yaoliang Yu, Pascal Poupart

**<details><summary>Abstract**</summary>

The construction of adversarial examples usually requires the existence of soft or hard labels for each instance, with respect to which a loss gradient provides the signal for construction of the example. We show that for batch normalized deep image recognition architectures, intermediate latents that are produced after a batch normalization step by themselves suffice to produce adversarial examples using an intermediate loss solely utilizing angular deviations, without relying on any label. We motivate our loss through the geometry of batch normed representations and their concentration of norm on a hypersphere and distributional proximity to Gaussians. Our losses expand intermediate latent based attacks that usually require labels. The success of our method implies that leakage of intermediate representations may create a security breach for deployed models, which persists even when the model is transferred to downstream usage. Removal of batch norm weakens our attack, indicating it contributes to this vulnerability. Our attacks also succeed against LayerNorm empirically, thus being relevant for transformer architectures, most notably vision transformers which we analyze.</details>

### BayesTune: Bayesian Sparse Deep Model Fine-tuning

**Authors:** Minyoung Kim, Timothy Hospedales

**<details><summary>Abstract**</summary>

Deep learning practice is increasingly driven by powerful foundation models (FM), pre-trained at scale and then fine-tuned for specific tasks of interest. A key property of this workflow is the efficacy of performing sparse or parameter-efficient fine-tuning, meaning that by updating only a tiny fraction of the whole FM parameters on a downstream task can lead to surprisingly good performance, often even superior to a full model update. However, it is not clear what is the optimal and principled way to select which parameters to update. Although a growing number of sparse fine-tuning ideas have been proposed, they are mostly not  satisfactory, relying on hand-crafted heuristics or heavy approximation. In this paper we propose a novel Bayesian sparse fine-tuning algorithm: we place a (sparse) Laplace prior for each parameter of the FM, with the mean equal to the initial value and the scale parameter having a hyper-prior that encourages small scale. Roughly speaking, the posterior means of the scale parameters indicate how important it is to update the corresponding parameter away from its initial value when solving the downstream task. Given the sparse prior, most scale parameters are small a posteriori, and the few large-valued scale parameters identify those FM parameters that crucially need to be updated away from their initial values. Based on this, we can threshold the scale parameters to decide which parameters to update or freeze, leading to a principled sparse fine-tuning strategy. To efficiently infer the posterior distribution of the scale parameters, we adopt the Langevin MCMC sampler, requiring only two times the complexity of the vanilla SGD. Tested on popular NLP benchmarks as well as the VTAB vision tasks, our approach shows significant improvement over the state-of-the-arts (e.g., 1% point higher than the best SOTA when fine-tuning RoBERTa for GLUE and SuperGLUE benchmarks).</details>

### [Spotlight] Bayesian Extensive-Rank Matrix Factorization with Rotational Invariant Priors

**Authors:** Farzad Pourkamali, Nicolas Macris

**<details><summary>Abstract**</summary>

We consider a statistical model for matrix factorization in a regime where the rank of the two hidden matrix factors grows linearly with their dimension and their product is corrupted by additive noise. Despite various approaches, statistical and algorithmic limits of such problems have remained elusive. We study a Bayesian setting with the assumptions that (a) one of the matrix factors is symmetric, (b) both factors as well as the additive noise have rotational invariant priors, (c) the priors are known to the statistician. We derive analytical formulas for Rotation Invariant Estimators to reconstruct the two matrix factors, and conjecture that these are optimal in the large-dimension limit, in the sense that they minimize the average mean-square-error. We provide numerical checks which confirm the optimality conjecture when confronted to Oracle Estimators which are optimal by definition, but involve the ground-truth. Our derivation relies on a combination of tools, namely random matrix theory transforms, spherical integral formulas, and the replica method from statistical mechanics.</details>

### Bayesian Optimisation of Functions on Graphs

**Authors:** Xingchen Wan, Pierre Osselin, Henry Kenlay, Binxin Ru, Michael A Osborne, Xiaowen Dong

**<details><summary>Abstract**</summary>

The increasing availability of graph-structured data motivates the task of optimising over functions defined on the node set of graphs. Traditional graph search algorithms can be applied in this case, but they may be sample-inefficient and do not make use of information about the function values; on the other hand, Bayesian optimisation is a class of promising black-box solvers with superior sample efficiency, but it has scarcely been applied to such novel setups. To fill this gap, we propose a novel Bayesian optimisation framework that optimises over functions defined on generic, large-scale and potentially unknown graphs. Through the learning of suitable kernels on graphs, our framework has the advantage of adapting to the behaviour of the target function. The local modelling approach further guarantees the efficiency of our method. Extensive experiments on both synthetic and real-world graphs demonstrate the effectiveness of the proposed optimisation framework.</details>

### Bayesian Risk-Averse Q-Learning with Streaming Observations

**Authors:** Yuhao Wang, Enlu Zhou

**<details><summary>Abstract**</summary>

We consider a robust reinforcement learning problem, where a learning agent learns from a simulated training environment. To account for the model mis-specification between this training environment and the true environment due to lack of data, we adopt a formulation of Bayesian risk MDP (BRMDP) with infinite horizon, which uses Bayesian posterior to estimate the transition model and impose a risk functional to account for the model uncertainty. Observations from the real environment that is out of the agent's control arrive periodically and are utilized by the agent to update the Bayesian posterior to reduce model uncertainty. We theoretically demonstrate that BRMDP balances the trade-off between robustness and conservativeness, and we further develop a multi-stage Bayesian risk-averse Q-learning algorithm to solve BRMDP with streaming observations from real environment. The proposed algorithm learns a risk-averse yet optimal policy that depends on the availability of real-world observations. We provide a theoretical guarantee of strong convergence for the proposed algorithm.</details>

### Beyond Black-Box Advice: Learning-Augmented Algorithms for MDPs with Q-Value Predictions

**Authors:** Tongxin Li, Yiheng Lin, Shaolei Ren, Adam Wierman

**<details><summary>Abstract**</summary>

We study the tradeoff between consistency and robustness in the context of a single-trajectory time-varying Markov Decision Process (MDP) with untrusted machine-learned advice. Our work departs from the typical approach of treating advice as coming from black-box sources by instead considering a setting where additional information about  how the advice is generated is available. We prove a first-of-its-kind consistency and robustness tradeoff given Q-value advice under a general MDP model that includes both continuous and discrete state/action spaces. Our results highlight that utilizing Q-value advice enables dynamic pursuit of the better of machine-learned advice and a robust baseline, thus result in near-optimal performance guarantees, which provably improves what can be obtained solely with black-box advice.</details>

### Beyond Confidence: Reliable Models Should Also Consider Atypicality

**Authors:** Mert Yuksekgonul, Linjun Zhang, James Zou, Carlos Guestrin

**<details><summary>Abstract**</summary>

While most machine learning models can provide confidence in their predictions, confidence is insufficient to understand a prediction's reliability. For instance, the model may have a low confidence prediction if the input is not well-represented in the training dataset or if the input is inherently ambiguous. In this work, we investigate the relationship between how atypical~(rare) a sample or a class is and the reliability of a model's predictions. We first demonstrate that atypicality is strongly related to miscalibration and accuracy. In particular, we empirically show that predictions for atypical inputs or atypical classes are more overconfident and have lower accuracy. Using these insights, we show incorporating atypicality improves uncertainty quantification and model performance for discriminative neural networks and large language models. In a case study, we show that using atypicality improves the performance of a skin lesion classifier across different skin tone groups without having access to the group attributes. Overall, we propose that models should use not only confidence but also atypicality to improve uncertainty quantification and performance. Our results demonstrate that simple post-hoc atypicality estimators can provide significant value.</details>

### [Spotlight] Beyond Myopia: Learning from Positive and Unlabeled Data through Holistic Predictive Trends

**Authors:** Wang Xinrui, Wenhai Wan, Chuanxing Geng, Shao-Yuan Li, Songcan Chen

**<details><summary>Abstract**</summary>

Learning binary classifiers from positive and unlabeled data (PUL) is vital in many real-world applications, especially when verifying negative examples is difficult. Despite the impressive empirical performance of recent PUL methods, challenges like accumulated errors and increased estimation bias persist due to the absence of negative labels. In this paper, we unveil an intriguing yet long-overlooked observation in PUL: \textit{resampling the positive data in each training iteration to ensure a balanced distribution between positive and unlabeled examples results in strong early-stage performance. Furthermore, predictive trends for positive and negative classes display distinctly different patterns.} Specifically, the scores (output probability) of unlabeled negative examples consistently decrease, while those of unlabeled positive examples show largely chaotic trends. Instead of focusing on classification within individual time frames, we innovatively adopt a holistic approach, interpreting the scores of each example as a temporal point process (TPP). This reformulates the core problem of PUL as recognizing trends in these scores. We then propose a novel TPP-inspired measure for trend detection and prove its asymptotic unbiasedness in predicting changes. Notably, our method accomplishes PUL without requiring additional parameter tuning or prior assumptions, offering an alternative perspective for tackling this problem. Extensive experiments verify the superiority of our method, particularly in a highly imbalanced real-world setting, where it achieves improvements of up to $11.3\%$ in key metrics.</details>

### Beyond NTK with Vanilla Gradient Descent: A Mean-Field Analysis of Neural Networks with Polynomial Width, Samples, and Time

**Authors:** Arvind Mahankali, Jeff Z. HaoChen, Kefan Dong, Margalit Glasgow, Tengyu Ma

**<details><summary>Abstract**</summary>

Despite recent theoretical progress on the non-convex optimization of two-layer neural networks, it is still an open question whether gradient descent on neural networks without unnatural modifications can achieve better sample complexity than kernel methods. This paper provides a clean mean-field analysis of projected gradient flow on polynomial-width two-layer neural networks. Different from prior works, our analysis does not require unnatural modifications of the optimization algorithm. We prove that with sample size $n = O(d^{3.1})$ where $d$ is the dimension of the inputs, the network trained with projected gradient flow converges in polynomial time to a non-trivial error that is not achievable by kernel methods using $n \ll  d^4$ samples, hence demonstrating a clear separation between unmodified gradient descent and NTK. As a corollary, we show that projected gradient descent with a positive learning rate and a polynomial number of iterations converges to low error with the same sample complexity.</details>

### Beyond Uniform Sampling: Offline Reinforcement Learning with Imbalanced Datasets

**Authors:** Zhang-Wei Hong, Aviral Kumar, Sathwik Karnik, Abhishek Bhandwaldar, Akash Srivastava, Joni Pajarinen, Romain Laroche, Abhishek Gupta, Pulkit Agrawal

**<details><summary>Abstract**</summary>

Offline reinforcement learning (RL) enables learning a decision-making policy without interaction with the environment. This makes it particularly beneficial in situations where such interactions are costly. However, a known challenge for offline RL algorithms is the distributional mismatch between the state-action distributions of the learned policy and the dataset, which can significantly impact performance. State-of-the-art algorithms address it by constraining the policy to align with the state-action pairs in the dataset. However, this strategy struggles on datasets that predominantly consist of trajectories collected by low-performing policies and only a few trajectories from high-performing ones. Indeed, the constraint to align with the data leads the policy to imitate low-performing behaviors predominating the dataset. Our key insight to address this issue is to constrain the policy to the policy that collected the good parts of the dataset rather than all data. To this end, we optimize the importance sampling weights to emulate sampling data from a data distribution generated by a nearly optimal policy. Our method exhibits considerable performance gains (up to five times better) over the existing approaches in state-of-the-art offline RL algorithms over 72 imbalanced datasets with varying types of imbalance.</details>

### BiMatting: Efficient Video Matting via Binarization

**Authors:** Haotong Qin, Lei Ke, Xudong Ma, Martin Danelljan, Yu-Wing Tai, Chi-Keung Tang, Xianglong Liu, Fisher Yu

**<details><summary>Abstract**</summary>

Real-time video matting on edge devices faces significant computational resource constraints, limiting the widespread use of video matting in applications such as online conferences and short-form video production. Binarization is a powerful compression approach that greatly reduces computation and memory consumption by using 1-bit parameters and bitwise operations. However, binarization of the video matting model is not a straightforward process, and our empirical analysis has revealed two primary bottlenecks: severe representation degradation of the encoder and massive redundant computations of the decoder. To address these issues, we propose BiMatting, an accurate and efficient video matting model using binarization. Specifically, we construct shrinkable and dense topologies of the binarized encoder block to enhance the extracted representation. We sparsify the binarized units to reduce the low-information decoding computation. Through extensive experiments, we demonstrate that BiMatting outperforms other binarized video matting models, including state-of-the-art (SOTA) binarization methods, by a significant margin. Our approach even performs comparably to the full-precision counterpart in visual quality. Furthermore, BiMatting achieves remarkable savings of 12.4$	imes$ and 21.6$	imes$ in computation and storage, respectively, showcasing its potential and advantages in real-world resource-constrained scenarios. Our code and models are released at https://github.com/htqin/BiMatting .</details>

### Binary Radiance Fields

**Authors:** Seungjoo Shin, Jaesik Park

**<details><summary>Abstract**</summary>

In this paper, we propose binary radiance fields (BiRF), a storage-efficient radiance field representation employing binary feature encoding that encodes local features using binary encoding parameters in a format of either $+1$ or $-1$. This binarization strategy lets us represent the feature grid with highly compact feature encoding and a dramatic reduction in storage size. Furthermore, our 2D-3D hybrid feature grid design enhances the compactness of feature encoding as the 3D grid includes main components while 2D grids capture details. In our experiments, binary radiance field representation successfully outperforms the reconstruction performance of state-of-the-art (SOTA) efficient radiance field models with lower storage allocation. In particular, our model achieves impressive results in static scene reconstruction, with a PSNR of 32.03 dB for Synthetic-NeRF scenes, 34.48 dB for Synthetic-NSVF scenes, 28.20 dB for Tanks and Temples scenes while only utilizing 0.5 MB of storage space, respectively. We hope the proposed binary radiance field representation will make radiance fields more accessible without a storage bottleneck.</details>

### Block-State Transformers

**Authors:** Jonathan Pilault, Mahan Fathi, Orhan Firat, Chris Pal, Pierre-Luc Bacon, Ross Goroshin

**<details><summary>Abstract**</summary>

State space models (SSMs) have shown impressive results on tasks that require modeling long-range dependencies and efficiently scale to long sequences owing to their subquadratic runtime complexity.Originally designed for continuous signals, SSMs have shown superior performance on a plethora of tasks, in vision and audio; however, SSMs still lag Transformer performance in Language Modeling tasks.In this work, we propose a hybrid layer named Block-State Transformer (*BST*), that internally combines an SSM sublayer for long-range contextualization, and a Block Transformer sublayer for short-term representation of sequences.We study three different, and completely *parallelizable*, variants that integrate SSMs and block-wise attention.We show that our model outperforms similar Transformer-based architectures on language modeling perplexity and generalizes to longer sequences. In addition, the Block-State Transformer demonstrates a more than *tenfold* increase in speed at the layer level compared to the Block-Recurrent Transformer when model parallelization is employed.</details>

### Boosting Learning for LDPC Codes to Improve the Error-Floor Performance

**Authors:** Hee-Youl Kwak, Dae-Young Yun, Yongjune Kim, Sang-Hyo Kim, Jong-Seon No

**<details><summary>Abstract**</summary>

Low-density parity-check (LDPC) codes have been successfully commercialized in communication systems due to their strong error correction capabilities and simple decoding process. However, the error-floor phenomenon of LDPC codes, in which the error rate stops decreasing rapidly at a certain level, presents challenges for achieving extremely low error rates and deploying LDPC codes in scenarios demanding ultra-high reliability. In this work, we propose training methods for neural min-sum (NMS) decoders to eliminate the error-floor effect. First, by leveraging the boosting learning technique of ensemble networks, we divide the decoding network into two neural decoders and train the post decoder to be specialized for uncorrected words that the first decoder fails to correct. Secondly, to address the vanishing gradient issue in training, we introduce a block-wise training schedule that locally trains a block of weights while retraining the preceding block. Lastly, we show that assigning different weights to unsatisfied check nodes effectively lowers the error-floor with a minimal number of weights. By applying these training methods to standard LDPC codes, we achieve the best error-floor performance compared to other decoding methods. The proposed NMS decoder, optimized solely through novel training methods without additional modules, can be integrated into existing LDPC decoders without incurring extra hardware costs. The source code is available at https://github.com/ghy1228/LDPC_Error_Floor.</details>

### Bounce: Reliable High-Dimensional Bayesian Optimization for Combinatorial and Mixed Spaces

**Authors:** Leonard Papenmeier, Luigi Nardi, Matthias Poloczek

**<details><summary>Abstract**</summary>

Impactful applications such as materials discovery, hardware design, neural architecture search, or portfolio optimization require optimizing high-dimensional black-box functions with mixed and combinatorial input spaces.While Bayesian optimization has recently made significant progress in solving such problems, an in-depth analysis reveals that the current state-of-the-art methods are not reliable. Their performances degrade substantially when the unknown optima of the function do not have a certain structure. To fill the need for a reliable algorithm for combinatorial and mixed spaces, this paper proposes Bounce that relies on a novel map of various variable types into nested embeddings of increasing dimensionality.Comprehensive experiments show that Bounce reliably achieves and often even improves upon state-of-the-art performance on a variety of high-dimensional problems.</details>

### Brain encoding models based on multimodal transformers can transfer across language and vision

**Authors:** Jerry Tang, Meng Du, Vy Vo, VASUDEV LAL, Alexander Huth

**<details><summary>Abstract**</summary>

Encoding models have been used to assess how the human brain represents concepts in language and vision. While language and vision rely on similar concept representations, current encoding models are typically trained and tested on brain responses to each modality in isolation. Recent advances in multimodal pretraining have produced transformers that can extract aligned representations of concepts in language and vision. In this work, we used representations from multimodal transformers to train encoding models that can transfer across fMRI responses to stories and movies. We found that encoding models trained on brain responses to one modality can successfully predict brain responses to the other modality, particularly in cortical regions that represent conceptual meaning. Further analysis of these encoding models revealed shared semantic dimensions that underlie concept representations in language and vision. Comparing encoding models trained using representations from multimodal and unimodal transformers, we found that multimodal transformers learn more aligned representations of concepts in language and vision. Our results demonstrate how multimodal transformers can provide insights into the brain’s capacity for multimodal processing.</details>

### Bringing regularized optimal transport to lightspeed: a splitting method adapted for GPUs

**Authors:** Jacob Lindbäck, Zesen Wang, Mikael Johansson

**<details><summary>Abstract**</summary>

We present an efficient algorithm for regularized optimal transport. In contrast toprevious methods, we use the Douglas-Rachford splitting technique to developan efficient solver that can handle a broad class of regularizers. The algorithmhas strong global convergence guarantees, low per-iteration cost, and can exploitGPU parallelization, making it considerably faster than the state-of-the-art formany problems. We illustrate its competitiveness in several applications, includingdomain adaptation and learning of generative models.</details>

### Bypass Exponential Time Preprocessing: Fast Neural Network Training via Weight-Data Correlation Preprocessing

**Authors:** Josh Alman, 杰昊 梁, Zhao Song, Ruizhe Zhang, Danyang Zhuo

**<details><summary>Abstract**</summary>

Over the last decade, deep neural networks have transformed our society, and they are already widely applied in various machine learning applications. State-of-the-art deep neural networks are becoming larger in size every year to deliver increasing model accuracy, and as a result, model training consumes substantial computing resources and will only consume more in the future.Using current training methods, in each iteration, to process a data point $x \in \mathbb{R}^d$ in a layer, we need to spend $\Theta(md)$ time to evaluate all the $m$ neurons in the layer. This means processing the entire layer takes $\Theta(nmd)$ time for $n$ data points. Recent work [Song, Yang and Zhang, NeurIPS 2021] reduces this time per iteration to $o(nmd)$, but requires exponential time to preprocess either the data or the neural network weights, making it unlikely to have practical usage. In this work, we present a new preprocessing method that simply stores the weight-data correlation in a tree data structure in order to quickly and dynamically detect which neurons fire at each iteration. Our method requires only $O(nmd)$ time in preprocessing and still achieves $o(nmd)$ time per iteration. We complement our new algorithm with a lower bound, proving that assuming a popular conjecture from complexity theory, one could not substantially speed up our algorithm for dynamic detection of firing neurons.</details>

### CELLE-2: Translating Proteins to Pictures and Back with a Bidirectional Text-to-Image Transformer

**Authors:** Emaad Khwaja, Yun Song, Aaron Agarunov, Bo Huang

**<details><summary>Abstract**</summary>

We present CELL-E 2, a novel bidirectional transformer that can generate images depicting protein subcellular localization from the amino acid sequences (and vice versa). Protein localization is a challenging problem that requires integrating sequence and image information, which most existing methods ignore. CELL-E 2 extends the work of CELL-E, not only capturing the spatial complexity of protein localization and produce probability estimates of localization atop a nucleus image, but also being able to generate sequences from images, enabling de novo protein design. We train and finetune CELL-E 2 on two large-scale datasets of human proteins. We also demonstrate how to use CELL-E 2 to create hundreds of novel nuclear localization signals (NLS). Results and interactive demos are featured at https://bohuanglab.github.io/CELL-E_2/.</details>

### [Spotlight] CS4ML: A general framework for active learning with arbitrary data based on Christoffel functions

**Authors:** Juan M. Cardenas, Ben Adcock, Nick Dexter

**<details><summary>Abstract**</summary>

We introduce a general framework for active learning in regression problems. Our framework extends the standard setup by allowing for general types of data, rather than merely pointwise samples of the target function. This generalization covers many cases of practical interest, such as data acquired in transform domains (e.g., Fourier data), vector-valued data (e.g., gradient-augmented data), data acquired along continuous curves, and, multimodal data (i.e., combinations of different types of measurements).  Our framework considers random sampling according to a finite number of sampling measures and arbitrary nonlinear approximation spaces (model classes). We introduce the concept of \textit{generalized Christoffel functions} and show how these can be used to optimize the sampling measures. We prove that this leads to near-optimal sample complexity in various important cases. This paper focuses on applications in scientific computing, where active learning is often desirable, since it is usually expensive to generate data. We demonstrate the efficacy of our framework for gradient-augmented learning with polynomials, Magnetic Resonance Imaging (MRI) using generative models and adaptive sampling for solving PDEs using Physics-Informed Neural Networks (PINNs).</details>

### [Spotlight] Can Language Models Solve Graph Problems in Natural Language?

**Authors:** Heng Wang, Shangbin Feng, Tianxing He, Zhaoxuan Tan, Xiaochuang Han, Yulia Tsvetkov

**<details><summary>Abstract**</summary>

Large language models (LLMs) are increasingly adopted for a variety of tasks with implicit graphical structures, such as planning in robotics, multi-hop question answering or knowledge probing, structured commonsense reasoning, and more. While LLMs have advanced the state-of-the-art on these tasks with structure implications, whether LLMs could explicitly process textual descriptions of graphs and structures, map them to grounded conceptual spaces, and perform structured operations remains underexplored. To this end, we propose NLGraph (Natural Language Graph), a comprehensive benchmark of graph-based problem solving designed in natural language. NLGraph contains 29,370 problems, covering eight graph reasoning tasks with varying complexity from simple tasks such as connectivity and shortest path up to complex problems such as maximum flow and simulating graph neural networks. We evaluate LLMs (GPT-3/4) with various prompting approaches on the NLGraph benchmark and find that 1) language models do demonstrate preliminary graph reasoning abilities, 2) the benefit of advanced prompting and in-context learning diminishes on more complex graph problems, while 3) LLMs are also (un)surprisingly brittle in the face of spurious correlations in graph and problem settings. We then propose Build-a-Graph Prompting and Algorithmic Prompting, two instruction-based approaches to enhance LLMs in solving natural language graph problems. Build-a-Graph and Algorithmic prompting improve the performance of LLMs on NLGraph by 3.07% to 16.85% across multiple tasks and settings, while how to solve the most complicated graph reasoning tasks in our setup with language models remains an open research question.</details>

### Cappy: Outperforming and Boosting Large Multi-Task LMs with a Small Scorer

**Authors:** Bowen Tan, Yun Zhu, Lijuan Liu, Eric Xing, Zhiting Hu, Jindong Chen

**<details><summary>Abstract**</summary>

Large language models (LLMs) such as T0, FLAN, and OPT-IML excel in multi-tasking under a unified instruction-following paradigm, where they also exhibit remarkable generalization abilities to unseen tasks. Despite their impressive performance, these LLMs, with sizes ranging from several billion to hundreds of billions of parameters, demand substantial computational resources, making their training and inference expensive and inefficient. Furthermore, adapting these models to downstream applications, particularly complex tasks, is often unfeasible due to the extensive hardware requirements for finetuning, even when utilizing parameter-efficient approaches such as prompt tuning. Additionally, the most powerful multi-task LLMs, such as OPT-IML-175B and FLAN-PaLM-540B, are not publicly accessible, severely limiting their customization potential. To address these challenges, we introduce a pretrained small scorer, \textit{Cappy}, designed to enhance the performance and efficiency of multi-task LLMs. With merely 360 million parameters, Cappy functions either independently on classification tasks or serve as an auxiliary component for LLMs, boosting their performance. Moreover, Cappy enables efficiently integrating downstream supervision without requiring LLM finetuning nor the access to their parameters. Our experiments demonstrate that, when working independently on 11 language understanding tasks from PromptSource, Cappy outperforms LLMs that are several orders of magnitude larger. Besides, on 45 complex tasks from BIG-Bench, Cappy boosts the performance of the advanced multi-task LLM, FLAN-T5, by a large margin. Furthermore, Cappy is flexible to cooperate with other LLM adaptations, including finetuning and in-context learning, offering additional performance enhancement.</details>

### Category-Extensible Out-of-Distribution Detection via Hierarchical Context Descriptions

**Authors:** Kai Liu, Zhihang Fu, Chao Chen, Sheng Jin, Ze Chen, Mingyuan Tao, Rongxin Jiang, Jieping Ye

**<details><summary>Abstract**</summary>

The key to OOD detection has two aspects: generalized feature representation and precise category description. Recently, vision-language models such as CLIP provide significant advances in both two issues, but constructing precise category descriptions is still in its infancy due to the absence of unseen categories. This work introduces two hierarchical contexts, namely perceptual context and spurious context, to carefully describe the precise category boundary through automatic prompt tuning. Specifically, perceptual contexts perceive the inter-category difference (e.g., cats vs apples) for current classification tasks, while spurious contexts further identify spurious (similar but exactly not) OOD samples for every single category (e.g., cats vs panthers, apples vs peaches). The two contexts hierarchically construct the precise description for a certain category, which is, first roughly classifying a sample to the predicted category and then delicately identifying whether it is truly an ID sample or actually OOD. Moreover, the precise descriptions for those categories within the vision-language framework present a novel application: CATegory-EXtensible OOD detection (CATEX). One can efficiently extend the set of recognizable categories by simply merging the hierarchical contexts learned under different sub-task settings. And extensive experiments are conducted to demonstrate CATEX’s effectiveness, robustness, and category-extensibility. For instance, CATEX consistently surpasses the rivals by a large margin with several protocols on the challenging ImageNet-1K dataset. In addition, we offer new insights on how to efficiently scale up the prompt engineering in vision-language models to recognize thousands of object categories, as well as how to incorporate large language models (like GPT-3) to boost zero-shot applications.</details>

### Causal Discovery in Semi-Stationary Time Series

**Authors:** Shanyun Gao, Raghavendra Addanki, Tong Yu, Ryan Rossi, Murat Kocaoglu

**<details><summary>Abstract**</summary>

Discovering causal relations from observational time series without making the stationary assumption is a significant challenge. In practice, this challenge is common in many areas, such as retail sales, transportation systems, and medical science. Here, we consider this problem for a class of non-stationary time series problems. The structural causal model (SCM) of this type of time series, called the semi-stationary time series, exhibits that a finite number of different causal mechanisms occur sequentially and periodically across time. This model holds considerable practical utility because it can represent periodicity, including common occurrences such as seasonality and diurnal variation. We propose a constraint-based, non-parametric algorithm for discovering causal relations in this setting. The resulting algorithm, PCMCI$_{\Omega}$, can capture the alternating and recurring changes in the causal mechanisms and then identify the underlying causal graph with conditional independence (CI) tests. We show that this algorithm is sound in identifying causal relations on discrete time series. We validate the algorithm with extensive experiments on continuous and discrete simulated data. We also apply our algorithm to a real-world climate dataset.</details>

### Causal Effect Regularization: Automated Detection and Removal of Spurious Correlations

**Authors:** Abhinav Kumar, Amit Deshpande, Amit Sharma

**<details><summary>Abstract**</summary>

In many classification datasets, the task labels are spuriously correlated with some input attributes. Classifiers trained on such datasets often rely on these attributes for prediction, especially when the spurious correlation is high,  and thus fail to generalize whenever there is a shift in the attributes' correlation at deployment. If we assume that the spurious attributes are known a priori, several methods have been proposed to learn a classifier that is invariant to the specified attributes.  However, in real-world data, information about spurious attributes is typically unavailable. Therefore, we propose a method to automatically identify spurious attributes by estimating their causal effect on the label and then use a regularization objective to mitigate the classifier's reliance on them. Compared to a recent method for identifying spurious attributes, we find that our method is more accurate in removing the attribute from the learned model, especially when spurious correlation is high. Specifically, across synthetic, semi-synthetic, and real-world datasets, our method shows significant improvement in a metric used to quantify the dependence of a classifier on spurious attributes ($\Delta$Prob), while obtaining better or similar accuracy. In addition, our method mitigates the reliance on spurious attributes even under noisy estimation of causal effects. To explain the empirical robustness of our method, we create a simple linear classification task with two sets of attributes: causal and spurious. We prove that our method only requires that the ranking of estimated causal effects is correct across attributes to select the correct classifier.</details>

### Causal Imitability Under Context-Specific Independence Relations

**Authors:** Fateme Jamshidi, Sina Akbari, Negar Kiyavash

**<details><summary>Abstract**</summary>

Drawbacks of ignoring the causal mechanisms when performing imitation learning have recently been acknowledged. Several approaches both to assess the feasibility of imitation and to circumvent causal confounding and causal misspecifications have been proposed in the literature.However, the potential benefits of the incorporation of additional information about the underlying causal structure are left unexplored.An example of such overlooked information is context-specific independence (CSI), i.e., independence that holds only in certain contexts.We consider the problem of causal imitation learning when CSI relations are known.We prove that the decision problem pertaining to the feasibility of imitation in this setting is NP-hard.Further, we provide a necessary graphical criterion for imitation learning under CSI and show that under a structural assumption, this criterion is also sufficient.Finally, we propose a sound algorithmic approach for causal imitation learning which takes both CSI relations and data into account.</details>

### Causal-structure Driven Augmentations for Text OOD Generalization

**Authors:** Amir Feder, Yoav Wald, Claudia Shi, Suchi Saria, David Blei

**<details><summary>Abstract**</summary>

The reliance of text classifiers on spurious correlations can lead to poor generalization at deployment, raising concerns about their use in safety-critical domains such as healthcare. In this work, we propose to use counterfactual data augmentation, guided by knowledge of the causal structure of the data, to simulate interventions on spurious features and to learn more robust text classifiers. We show that this strategy is appropriate in prediction problems where the label is spuriously correlated with an attribute. Under the assumptions of such problems, we discuss the favorable sample complexity of counterfactual data augmentation, compared to importance re-weighting. Pragmatically, we match examples using auxiliary data, based on diff-in-diff methodology, and use a large language model (LLM) to represent a conditional probability of text. Through extensive experimentation on learning caregiver-invariant predictors of clinical diagnoses from medical narratives and on semi-synthetic data, we demonstrate that our method for simulating interventions improves out-of-distribution (OOD) accuracy compared to baseline invariant learning algorithms.</details>

### Chanakya: Learning Runtime Decisions for Adaptive Real-Time Perception

**Authors:** Anurag Ghosh, Vaibhav Balloli, Akshay Nambi, Aditya Singh, Tanuja Ganu

**<details><summary>Abstract**</summary>

Real-time perception requires planned resource utilization. Computational planning in real-time perception is governed by two considerations -- accuracy and latency. There exist run-time decisions (e.g. choice of input resolution) that induce tradeoffs affecting performance on a given hardware, arising from intrinsic (content, e.g. scene clutter) and extrinsic (system, e.g. resource contention) characteristics. Earlier runtime execution frameworks employed rule-based decision algorithms and operated with a fixed algorithm latency budget to balance these concerns, which is sub-optimal and inflexible. We propose Chanakya, a learned approximate execution framework that naturally derives from the streaming perception paradigm, to automatically learn decisions induced by these tradeoffs instead. Chanakya is trained via novel rewards balancing accuracy and latency implicitly, without approximating either objectives. Chanakya simultaneously considers intrinsic and extrinsic context, and predicts decisions in a flexible manner. Chanakya, designed with low overhead in mind, outperforms state-of-the-art static and dynamic execution policies on public datasets on both server GPUs and edge devices.</details>

### Characterization of Overfitting in Robust Multiclass Classification

**Authors:** Jingyuan Xu, Weiwei Liu

**<details><summary>Abstract**</summary>

This paper considers the following question: Given the number of classes m, the number of robust accuracy queries k, and the number of test examples in the dataset n, how much can adaptive algorithms robustly overfit the test dataset? We solve this problem by equivalently giving near-matching upper and lower bounds of the robust overfitting bias in multiclass classification problems.</details>

### Characterizing Out-of-Distribution Error via Optimal Transport

**Authors:** Yuzhe Lu, Yilong Qin, Runtian Zhai, Andrew Shen, Ketong Chen, Zhenlin Wang, Soheil Kolouri, Simon Stepputtis, Joseph Campbell, Katia Sycara

**<details><summary>Abstract**</summary>

Out-of-distribution (OOD) data poses serious challenges in deployed machine learning models,so methods of predicting a model's performance on OOD data without labels are important for machine learning safety.While a number of methods have been proposed by prior work,they often underestimate the actual error, sometimes by a large margin, which greatly impacts their applicability to real tasks.In this work, we identify *pseudo-label shift*, or the difference between the predicted and true OOD label distributions, as a key indicator to this underestimation. Based on this observation, we introduce a novel method for estimating model performance by leveraging optimal transport theory, Confidence Optimal Transport (COT), and show that it provably provides more robust error estimates in the presence of pseudo-label shift. Additionally, we introduce an empirically-motivated variant of COT, Confidence Optimal Transport with Thresholding (COTT), which applies thresholding to the individual transport costs and further improves the accuracy of COT's error estimates. We evaluate COT and COTT on a variety of standard benchmarks that induce various types of distribution shift -- synthetic, novel subpopulation, and natural -- and show that our approaches significantly outperform existing state-of-the-art methods with up to 3x lower prediction errorS.</details>

### Characterizing the Impacts of Semi-supervised Learning for Weak Supervision

**Authors:** Jeffrey Li, Jieyu Zhang, Ludwig Schmidt, Alexander Ratner

**<details><summary>Abstract**</summary>

Labeling training data is a critical and expensive step in producing high accuracy ML models, whether training from scratch or fine-tuning. To make labeling more efficient, two major approaches are programmatic weak supervision (WS) and semi-supervised learning (SSL). More recent works have either explicitly or implicitly used techniques at their intersection, but in various complex and ad hoc ways. In this work, we define a simple, modular design space to study the use of SSL techniques for WS more systematically. Surprisingly, we find that fairly simple methods from our design space match the performance of more complex state-of-the-art methods, averaging a 3 p.p. increase in accuracy/F1-score across 8 standard WS benchmarks. Further, we provide practical guidance on when different components are worth their added complexity and training costs. Contrary to current understanding, we find using SSL is not necessary to obtain the best performance on most WS benchmarks but is more effective when: (1) end models are smaller, and (2) WS provides labels for only a small portion of training examples.</details>

### Chatting Makes Perfect: Chat-based Image Retrieval

**Authors:** Matan Levy, Rami Ben-Ari, Nir Darshan, Dani Lischinski

**<details><summary>Abstract**</summary>

Chats emerge as an effective user-friendly approach for information retrieval, and are successfully employed in many domains, such as customer service, healthcare, and finance. However, existing image retrieval approaches typically address the case of a single query-to-image round, and the use of chats for image retrieval has been mostly overlooked. In this work, we introduce ChatIR: a chat-based image retrieval system that engages in a conversation with the user to elicit information, in addition to an initial query, in order to clarify the user's search intent. Motivated by the capabilities of today's foundation models, we leverage Large Language Models to generate follow-up questions to an initial image description. These questions form a dialog with the user in order to retrieve the desired image from a large corpus. In this study, we explore the capabilities of such a system tested on a large dataset and reveal that engaging in a dialog yields significant gains in image retrieval. We start by building an evaluation pipeline from an existing manually generated dataset and explore different modules and training strategies for ChatIR. Our comparison includes strong baselines derived from related applications trained with Reinforcement Learning. Our system is capable of retrieving the target image from a pool of 50K images with over 78% success rate after 5 dialogue rounds, compared to 75% when questions are asked by humans, and 64% for a single shot text-to-image retrieval. Extensive evaluations reveal the strong capabilities and examine the limitations of CharIR under different settings. Project repository is available at https://github.com/levymsn/ChatIR.</details>

### Class-Distribution-Aware Pseudo-Labeling for Semi-Supervised Multi-Label Learning

**Authors:** Ming-Kun Xie, Jiahao Xiao, Hao-Zhe Liu, Gang Niu, Masashi Sugiyama, Sheng-Jun Huang

**<details><summary>Abstract**</summary>

Pseudo-labeling has emerged as a popular and effective approach for utilizing unlabeled data. However, in the context of semi-supervised multi-label learning (SSMLL), conventional pseudo-labeling methods encounter difficulties when dealing with instances associated with multiple labels and an unknown label count. These limitations often result in the introduction of false positive labels or the neglect of true positive ones. To overcome these challenges, this paper proposes a novel solution called Class-Aware Pseudo-Labeling (CAP) that performs pseudo-labeling in a class-aware manner. The proposed approach introduces a regularized learning framework incorporating class-aware thresholds, which effectively control the assignment of positive and negative pseudo-labels for each class. Notably, even with a small proportion of labeled examples, our observations demonstrate that the estimated class distribution serves as a reliable approximation. Motivated by this finding, we develop a class-distribution-aware thresholding strategy to ensure the alignment of pseudo-label distribution with the true distribution. The correctness of the estimated class distribution is theoretically verified, and a generalization error bound is provided for our proposed method. Extensive experiments on multiple benchmark datasets confirm the efficacy of CAP in addressing the challenges of SSMLL problems.</details>

### [Oral] ClimSim: A large multi-scale dataset for hybrid physics-ML climate emulation

**Authors:** Sungduk Yu, Walter Hannah, Liran Peng, Jerry Lin, Mohamed Aziz Bhouri, Ritwik Gupta, Björn Lütjens, Justus C. Will, Gunnar Behrens, Julius Busecke, Nora Loose, Charles Stern, Tom Beucler, Bryce Harrop, Benjamin Hillman, Andrea Jenney, Savannah L. Ferretti, Nana Liu, Animashree Anandkumar, Noah Brenowitz, Veronika Eyring, Nicholas Geneva, Pierre Gentine, Stephan Mandt, Jaideep Pathak, Akshay Subramaniam, Carl Vondrick, Rose Yu, Laure Zanna, Tian Zheng, Ryan Abernathey, Fiaz Ahmed, David Bader, Pierre Baldi, Elizabeth Barnes, Christopher Bretherton, Peter Caldwell, Wayne Chuang, Yilun Han, YU HUANG, Fernando Iglesias-Suarez, Sanket Jantre, Karthik Kashinath, Marat Khairoutdinov, Thorsten Kurth, Nicholas Lutsko, Po-Lun Ma, Griffin Mooers, J. David Neelin, David Randall, Sara Shamekh, Mark Taylor, Nathan Urban, Janni Yuval, Guang Zhang, Mike Pritchard

**Oral Presentation:** We, Dec 13, 13:45 -- Oral 4B

**<details><summary>Abstract**</summary>

Modern climate projections lack adequate spatial and temporal resolution due to computational constraints. A consequence is inaccurate and imprecise predictions of critical processes such as storms. Hybrid methods that combine physics with machine learning (ML) have introduced a new generation of higher fidelity climate simulators that can sidestep Moore's Law by outsourcing compute-hungry, short, high-resolution simulations to ML emulators. However, this hybrid ML-physics simulation approach requires domain-specific treatment and has been inaccessible to ML experts because of lack of training data and relevant, easy-to-use workflows. We present ClimSim, the largest-ever dataset designed for hybrid ML-physics research. It comprises multi-scale climate simulations, developed by a consortium of climate scientists and ML researchers. It consists of 5.7 billion pairs of multivariate input and output vectors that isolate the influence of locally-nested, high-resolution, high-fidelity physics on a host climate simulator's macro-scale physical state.The dataset is global in coverage, spans multiple years at high sampling frequency, and is designed such that resulting emulators are compatible with downstream coupling into operational climate simulators. We implement a range of deterministic and stochastic regression baselines to highlight the ML challenges and their scoring. The data (https://huggingface.co/datasets/LEAP/ClimSim_high-res) and code (https://leap-stc.github.io/ClimSim) are released openly to support the development of hybrid ML-physics and high-fidelity climate simulations for the benefit of science and society.</details>

### Cluster-aware Semi-supervised Learning: Relational Knowledge Distillation Provably Learns Clustering

**Authors:** Yijun Dong, Kevin Miller, Qi Lei, Rachel Ward

**<details><summary>Abstract**</summary>

Despite the empirical success and practical significance of (relational) knowledge distillation that matches (the relations of) features between teacher and student models, the corresponding theoretical interpretations remain limited for various knowledge distillation paradigms. In this work, we take an initial step toward a theoretical understanding of relational knowledge distillation (RKD), with a focus on semi-supervised classification problems. We start by casting RKD as spectral clustering on a population-induced graph unveiled by a teacher model. Via a notion of clustering error that quantifies the discrepancy between the predicted and ground truth clusterings, we illustrate that RKD over the population provably leads to low clustering error. Moreover, we provide a sample complexity bound for RKD with limited unlabeled samples. For semi-supervised learning, we further demonstrate the label efficiency of RKD through a general framework of cluster-aware semi-supervised learning that assumes low clustering errors. Finally, by unifying data augmentation consistency regularization into this cluster-aware framework, we show that despite the common effect of learning accurate clusterings, RKD facilitates a "global" perspective through spectral clustering, whereas consistency regularization focuses on a "local" perspective via expansion.</details>

### ClusterFomer: Clustering As A Universal Visual Learner

**Authors:** James Liang, Yiming Cui, Qifan Wang, Tong Geng, Wenguan Wang, Dongfang Liu

**<details><summary>Abstract**</summary>

This paper presents ClusterFormer, a universal vision model that is based on the Clustering paradigm with TransFormer. It comprises two novel designs: 1) recurrent cross-attention clustering, which reformulates the cross-attention mechanism in Transformer and enables recursive updates of cluster centers to facilitate strong representation learning; and 2) feature dispatching, which uses the updated cluster centers to redistribute image features through similarity-based metrics, resulting in a transparent pipeline. This elegant design streamlines an explainable and transferable workflow, capable of tackling heterogeneous vision tasks (i.e., image classification, object detection, and image segmentation) with varying levels of clustering granularity (i.e., image-, box-, and pixel-level). Empirical results demonstrate that ClusterFormer outperforms various well-known specialized architectures, achieving 83.41% top-1 acc. over ImageNet-1K for image classification, 54.2% and 47.0% mAP over MS COCO for object detection and instance segmentation, 52.4% mIoU over ADE20K for semantic segmentation, and 55.8% PQ over COCO Panoptic for panoptic segmentation. This work aims to initiate a paradigm shift in universal visual understanding and to benefit the broader field.</details>

### CoDet: Co-occurrence Guided Region-Word Alignment for Open-Vocabulary Object Detection

**Authors:** Chuofan Ma, Yi Jiang, Xin Wen, Zehuan Yuan, Xiaojuan Qi

**<details><summary>Abstract**</summary>

Deriving reliable region-word alignment from image-text pairs is critical to learnobject-level vision-language representations for open-vocabulary object detection.Existing methods typically rely on pre-trained or self-trained vision-languagemodels for alignment, which are prone to limitations in localization accuracy orgeneralization capabilities. In this paper, we propose CoDet, a novel approachthat overcomes the reliance on pre-aligned vision-language space by reformulatingregion-word alignment as a co-occurring object discovery problem. Intuitively, bygrouping images that mention a shared concept in their captions, objects corresponding to the shared concept shall exhibit high co-occurrence among the group.CoDet then leverages visual similarities to discover the co-occurring objects andalign them with the shared concept. Extensive experiments demonstrate that CoDethas superior performances and compelling scalability in open-vocabulary detection,e.g., by scaling up the visual backbone, CoDet achieves 37.0 $AP^m_{novel}$ and 44.7 $AP^m_{all}$ on OV-LVIS, surpassing the previous SoTA by 4.2 $AP^m_{novel}$ and 9.8 $AP^m_{all}$. Code is available at https://github.com/CVMI-Lab/CoDet.</details>

### CoLA: Exploiting Compositional Structure for Automatic and Efficient Numerical Linear Algebra

**Authors:** Andres Potapczynski, Marc Finzi, Geoff Pleiss, Andrew Wilson

**<details><summary>Abstract**</summary>

Many areas of machine learning and science involve large linear algebra problems, such as eigendecompositions, solving linear systems, computing matrix exponentials, and trace estimation. The matrices involved often have Kronecker, convolutional, block diagonal, sum, or product structure. In this paper, we propose a simple but general framework for large-scale linear algebra problems in machine learning, named CoLA (Compositional Linear Algebra). By combining a linear operator abstraction with compositional dispatch rules, CoLA automatically constructs memory and runtime efficient numerical algorithms. Moreover, CoLA provides memory efficient automatic differentiation, low precision computation, and GPU acceleration in both JAX and PyTorch, while also accommodating new objects, operations, and rules in downstream packages via multiple dispatch. CoLA can accelerate many algebraic operations, while making it easy to prototype matrix structures and algorithms, providing an appealing drop-in tool for virtually any computational effort that requires linear algebra. We showcase its efficacy across a broad range of applications, including partial differential equations, Gaussian processes, equivariant model construction, and unsupervised learning.</details>

### Cognitive Model Discovery via Disentangled RNNs

**Authors:** Kevin Miller, Maria Eckstein, Matt Botvinick, Zeb Kurth-Nelson

**<details><summary>Abstract**</summary>

Computational cognitive models are a fundamental tool in behavioral neuroscience. They embody in software precise hypotheses about the cognitive mechanisms underlying a particular behavior. Constructing these models is typically a difficult iterative process that requires both inspiration from the literature and the creativity of an individual researcher. Here, we adopt an alternative approach to learn parsimonious cognitive models directly from data. We fit behavior data using a recurrent neural network that is penalized for carrying excess information between timesteps, leading to sparse, interpretable representations and dynamics. When fitting synthetic behavioral data from known cognitive models, our method recovers the underlying form of those models. When fit to choice data from rats performing a bandit task, our method recovers simple and interpretable models that make testable predictions about neural mechanisms.</details>

### Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise

**Authors:** Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, Tom Goldstein

**<details><summary>Abstract**</summary>

Standard diffusion models involve an image transform  -- adding Gaussian noise -- and an image restoration operator that inverts this degradation.  We observe that the generative behavior of diffusion models is not strongly dependent on the choice of image degradation, and in fact, an entire family of generative models can be constructed by varying this choice. Even when using completely deterministic degradations (e.g., blur, masking, and more), the training and test-time update rules that underlie diffusion models can be easily generalized to create generative models. The success of these fully deterministic models calls into question the community's understanding of diffusion models, which relies on noise in either gradient Langevin dynamics or variational inference and paves the way for generalized diffusion models that invert arbitrary processes.</details>

### Collaborative Learning via Prediction Consensus

**Authors:** Dongyang Fan, Celestine Mendler-Dünner, Martin Jaggi

**<details><summary>Abstract**</summary>

We consider a collaborative learning setting where the goal of each agent is to improve their own model by leveraging the expertise of collaborators, in addition to their own training data. To facilitate the exchange of expertise among agents, we propose a distillation-based method leveraging shared unlabeled auxiliary data, which is pseudo-labeled by the collective. Central to our method is a trust weighting scheme that serves to adaptively weigh the influence of each collaborator on the pseudo-labels until a consensus on how to label the auxiliary data is reached. We demonstrate empirically that our collaboration scheme is able to significantly boost individual models’ performance in the target domain from which the auxiliary data is sampled. At the same time, it can provably mitigate the negative impact of bad models on the collective. By design, our method adeptly accommodates heterogeneity in model architectures and substantially reduces communication overhead compared to typical collaborative learning methods.</details>

### Compositional Foundation Models for Hierarchical Planning

**Authors:** Anurag Ajay, Seungwook Han, Yilun Du, Shuang Li, Abhi Gupta, Tommi Jaakkola, Josh Tenenbaum, Leslie Kaelbling, Akash Srivastava, Pulkit Agrawal

**<details><summary>Abstract**</summary>

To make effective decisions in novel environments with long-horizon goals, it is crucial to engage in hierarchical reasoning across spatial and temporal scales. This entails planning abstract subgoal sequences, visually reasoning about the underlying plans, and executing actions in accordance with the devised plan through visual-motor control. We propose Compositional Foundation Models for Hierarchical Planning (HiP), a foundation model which leverages multiple expert foundation model trained on language, vision and action data individually jointly together to solve long-horizon tasks. We use a large language model to construct symbolic plans that are grounded in the environment through a large video diffusion model. Generated video plans are then grounded to visual-motor control, through an inverse dynamics model that infers actions from generated videos. To enable effective reasoning within this hierarchy, we enforce consistency between the models via iterative refinement. We illustrate the efficacy and adaptability of our approach in three different long-horizon table-top manipulation tasks.</details>

### Compositional Policy Learning in Stochastic Control Systems with Formal Guarantees

**Authors:** Đorđe Žikelić, Mathias Lechner, Abhinav Verma, Krishnendu Chatterjee, Thomas Henzinger

**<details><summary>Abstract**</summary>

Reinforcement learning has shown promising results in learning neural network policies for complicated control tasks. However, the lack of formal guarantees about the behavior of such policies remains an impediment to their deployment. We propose a novel method for learning a composition of neural network policies in stochastic environments, along with a formal certificate which guarantees that a specification over the policy's behavior is satisfied with the desired probability. Unlike prior work on verifiable RL, our approach leverages the compositional nature of logical specifications provided in SpectRL, to learn over graphs of probabilistic reach-avoid specifications. The formal guarantees are provided by learning neural network policies together with reach-avoid supermartingales (RASM) for the graph’s sub-tasks and then composing them into a global policy. We also derive a tighter lower bound compared to previous work on the probability of reach-avoidance implied by a RASM, which is required to find a compositional policy with an acceptable probabilistic threshold for complex tasks with multiple edge policies. We implement a prototype of our approach and evaluate it on a Stochastic Nine Rooms environment.</details>

### Concept Distillation: Leveraging Human-Centered Explanations for Model Improvement

**Authors:** Avani Gupta, Saurabh Saini, P J Narayanan

**<details><summary>Abstract**</summary>

Humans use abstract *concepts* for understanding instead of hard features. Recent interpretability research has focused on human-centered concept explanations of neural networks. Concept Activation Vectors (CAVs) estimate a model's sensitivity and possible biases to a given concept. We extend CAVs from post-hoc analysis to ante-hoc training to reduce model bias through fine-tuning using an additional *Concept Loss*. Concepts are defined on the final layer of the network in the past. We generalize it to intermediate layers, including the last convolution layer. We also introduce *Concept Distillation*, a method to define rich and effective concepts using a pre-trained knowledgeable model as the teacher. Our method can sensitize or desensitize a model towards concepts. We show applications of concept-sensitive training to debias several classification problems. We also show a way to induce prior knowledge into a reconstruction problem. We show that concept-sensitive training can improve model interpretability, reduce biases, and induce prior knowledge.</details>

### Conditional Score Guidance for Text-Driven Image-to-Image Translation

**Authors:** Hyunsoo Lee, Minsoo Kang, Bohyung Han

**<details><summary>Abstract**</summary>

We present a novel algorithm for text-driven image-to-image translation based on a pretrained text-to-image diffusion model. Our method aims to generate a target image by selectively editing the regions of interest in a source image, defined by a modifying text, while preserving the remaining parts. In contrast to existing techniques that solely rely on a target prompt, we introduce a new score function that additionally considers both the source image and the source text prompt, tailored to address specific translation tasks. To this end, we derive the conditional score function in a principled manner, decomposing it into the standard score and a guiding term for target image generation. For the gradient computation of the guiding term, we assume a Gaussian distribution of the posterior distribution and estimate its mean and variance to adjust the gradient without additional training. In addition, to improve the quality of the conditional score guidance, we incorporate a simple yet effective mixup technique, which combines two cross-attention maps derived from the source and target latents. This strategy is effective for promoting a desirable fusion of the invariant parts in the source image and the edited regions aligned with the target prompt, leading to high-fidelity target image generation. Through comprehensive experiments, we demonstrate that our approach achieves outstanding image-to-image translation performance on various tasks.</details>

### Conformal Prediction Sets for Ordinal Classification

**Authors:** Prasenjit Dey, Srujana Merugu, Sivaramakrishnan R Kaveri

**<details><summary>Abstract**</summary>

Ordinal classification (OC), i.e., labeling instances along classes with a natural ordering, is common in multiple  applications such as size or budget based recommendations and disease severity labeling.  Often in practical scenarios, it is desirable to obtain a small set of likely classes with a guaranteed high chance of including the true class. Recent works on conformal prediction (CP) address this problem for the classification setting with non-ordered labels but the resulting prediction sets (PS) are often non-contiguous and unsuitable for ordinal classification. In this work, we propose a framework to adapt existing CP methods to generate contiguous sets with guaranteed coverage and minimal cardinality. Our framework employs a novel non-parametric approach for modeling unimodal distributions. Empirical results on both synthetic and real-world datasets demonstrate our method outperforms SOTA baselines by 4% on Accuracy@K and 8% on PS size.</details>

### Conformal Prediction for Uncertainty-Aware Planning with Diffusion Dynamics Model

**Authors:** Jiankai Sun, Yiqi Jiang, Jianing Qiu, Parth Nobel, Mykel J Kochenderfer, Mac Schwager

**<details><summary>Abstract**</summary>

Robotic applications often involve working in environments that are uncertain, dynamic, and partially observable. Recently, diffusion models have been proposed for learning trajectory prediction models trained from expert demonstrations, which can be used for planning in robot tasks. Such models have demonstrated a strong ability to overcome challenges such as multi-modal action distributions, high-dimensional output spaces, and training instability. It is crucial to quantify the uncertainty of these dynamics models when using them for planning. In this paper, we quantify the uncertainty of diffusion dynamics models using Conformal Prediction (CP). Given a finite number of exchangeable expert trajectory examples (called the “calibration set”), we use CP to obtain a set in the trajectory space (called the “coverage region”) that is guaranteed to contain the output of the diffusion model with a user-defined probability (called the “coverage level”). In PlanCP, inspired by concepts from conformal prediction, we modify the loss function for training the diffusion model to include a quantile term to encourage more robust performance across the variety of training examples. At test time, we then calibrate PlanCP with a conformal prediction process to obtain coverage sets for the trajectory prediction with guaranteed coverage level. We evaluate our algorithm on various planning tasks and model-based offline reinforcement learning tasks and show that it reduces the uncertainty of the learned trajectory prediction model. As a by-product, our algorithm PlanCP outperforms prior algorithms on existing offline RL benchmarks and challenging continuous planning tasks. Our method can be combined with most model-based planning approaches to produce uncertainty estimates of the closed-loop system.</details>

### Connecting Certified and Adversarial Training

**Authors:** Yuhao Mao, Mark Müller, Marc Fischer, Martin Vechev

**<details><summary>Abstract**</summary>

Training certifiably robust neural networks remains a notoriously hard problem.While adversarial training optimizes under-approximations of the worst-case loss, which leads to insufficient regularization for certification, sound certified training methods, optimize loose over-approximations, leading to over-regularization and poor (standard) accuracy.In this work, we propose TAPS, an (unsound) certified training method that combines IBP and PGD training to optimize more precise, although not necessarily sound, worst-case loss approximations, reducing over-regularization and increasing certified and standard accuracies.Empirically, TAPS achieves a new state-of-the-art in many settings, e.g., reaching a certified accuracy of $22$% on TinyImageNet for $\ell_\infty$-perturbations with radius $\epsilon=1/255$. We make our implementation and networks public at https://github.com/eth-sri/taps.</details>

### Connecting Pre-trained Language Model and Downstream Task via Properties of Representation

**Authors:** Chenwei Wu, Holden Lee, Rong Ge

**<details><summary>Abstract**</summary>

Recently, researchers have found that representations learned by large-scale pre-trained language models are useful in various downstream tasks. However, there is little theoretical understanding of how pre-training performance is related to downstream task performance. In this paper, we analyze how this performance transfer depends on the properties of the downstream task and the structure of the representations. We consider a log-linear model where a word can be predicted from its context through a network having softmax as its last layer. We show that even if the downstream task is highly structured and depends on a simple function of the hidden representation, there are still cases when a low pre-training loss cannot guarantee good performance on the downstream task. On the other hand, we propose and empirically validate the existence of an ``anchor vector'' in the representation space, and show that this assumption, together with properties of the downstream task,  guarantees performance transfer.</details>

### Constrained Policy Optimization with Explicit Behavior Density For Offline Reinforcement Learning

**Authors:** Jing Zhang, Chi Zhang, Wenjia Wang, Bingyi Jing

**<details><summary>Abstract**</summary>

Due to the inability to interact with the environment, offline reinforcement learning (RL) methods face the challenge of estimating the Out-of-Distribution (OOD) points. Existing methods for addressing this issue either control policy to exclude the OOD action or make the $Q$ function pessimistic. However, these methods can be overly conservative or fail to identify OOD areas accurately. To overcome this problem, we propose a Constrained Policy optimization with Explicit Behavior density (CPED) method that utilizes a flow-GAN model to explicitly estimate the density of behavior policy. By estimating the explicit density, CPED can accurately identify the safe region and enable exploration within the region, resulting in less conservative learning policies.  We further provide theoretical results for both the flow-GAN estimator and performance guarantee for CPED by showing that CPED can find the optimal $Q$-function value. Empirically, CPED outperforms existing alternatives on various standard offline reinforcement learning tasks, yielding higher expected returns.</details>

### Context Shift Reduction for Offline Meta-Reinforcement Learning

**Authors:** Yunkai Gao, Rui Zhang, Jiaming Guo, Fan Wu, Qi Yi, Shaohui Peng, Siming Lan, Ruizhi Chen, Zidong Du, Xing Hu, Qi Guo, Ling Li, Yunji Chen

**<details><summary>Abstract**</summary>

Offline meta-reinforcement learning (OMRL) utilizes pre-collected offline datasets to enhance the agent's generalization ability on unseen tasks. However, the context shift problem arises due to the distribution discrepancy between the contexts used for training (from the behavior policy) and testing (from the exploration policy). The context shift problem leads to incorrect task inference and further deteriorates the generalization ability of the meta-policy. Existing OMRL methods either overlook this problem or attempt to mitigate it with additional information. In this paper, we propose a novel approach called Context Shift Reduction for OMRL (CSRO) to address the context shift problem with only offline datasets. The key insight of CSRO is to minimize the influence of policy in context during both the meta-training and meta-test phases.  During meta-training, we design a max-min mutual information representation learning mechanism to diminish the impact of the behavior policy on task representation. In the meta-test phase, we introduce the non-prior context collection strategy to reduce the effect of the exploration policy. Experimental results demonstrate that CSRO significantly reduces the context shift and improves the generalization ability, surpassing previous methods across various challenging domains.</details>

### [Spotlight] Context-PIPs: Persistent Independent Particles Demands Context Features

**Authors:** Weikang Bian, Zhaoyang Huang, Xiaoyu Shi, Yitong Dong, Yijin Li, Hongsheng Li

**<details><summary>Abstract**</summary>

We tackle the problem of Persistent Independent Particles (PIPs), also called Tracking Any Point (TAP), in videos, which specifically aims at estimating persistent long-term trajectories of query points in videos. Previous methods attempted to estimate these trajectories independently to incorporate longer image sequences, therefore, ignoring the potential benefits of incorporating spatial context features. We argue that independent video point tracking also demands spatial context features.To this end, we propose a novel framework Context-PIPs, which effectively improves point trajectory accuracy by aggregating spatial context features in videos. Context-PIPs contains two main modules: 1) a SOurse Feature Enhancement (SOFE) module, and 2) a TArget Feature Aggregation (TAFA) module. Context-PIPs significantly improves PIPs all-sided, reducing 11.4\% Average Trajectory Error of Occluded Points (ATE-Occ) on CroHD and increasing 11.8\% Average Percentage of Correct Keypoint (A-PCK) on TAP-Vid-Kinectics. Demos are available at \url{https://wkbian.github.io/Projects/Context-PIPs/}.</details>

### Context-guided Embedding Adaptation for Effective Topic Modeling in Low-Resource Regimes

**Authors:** Yishi Xu, Jianqiao Sun, Yudi Su, Xinyang Liu, Zhibin Duan, Bo Chen, Mingyuan Zhou

**<details><summary>Abstract**</summary>

Embedding-based neural topic models have turned out to be a superior option for low-resourced topic modeling. However, current approaches consider static word embeddings learnt from source tasks as general knowledge that can be transferred directly to the target task, discounting the dynamically changing nature of word meanings in different contexts, thus typically leading to sub-optimal results when adapting to new tasks with unfamiliar contexts. To settle this issue, we provide an effective method that centers on adaptively generating semantically tailored word embeddings for each task by fully exploiting contextual information. Specifically, we first condense the contextual syntactic dependencies of words into a semantic graph for each task, which is then modeled by a Variational Graph Auto-Encoder to produce task-specific word representations. On this basis, we further impose a learnable Gaussian mixture prior on the latent space of words to efficiently learn topic representations from a clustering perspective, which contributes to diverse topic discovery and fast adaptation to novel tasks. We have conducted a wealth of quantitative and qualitative experiments, and the results show that our approach comprehensively outperforms established topic models.</details>

### Contextual Stochastic Bilevel Optimization

**Authors:** Yifan Hu, Jie Wang, Yao Xie, Andreas Krause, Daniel Kuhn

**<details><summary>Abstract**</summary>

We introduce contextual stochastic bilevel optimization (CSBO) -- a stochastic bilevel optimization framework with the lower-level problem minimizing an expectation conditioned on some contextual information and the upper-level decision variable. This framework extends classical stochastic bilevel optimization when the lower-level decision maker responds optimally not only to the decision of the upper-level decision maker but also to some side information and when there are multiple or even infinite many followers. It captures important applications such as meta-learning, personalized federated learning, end-to-end learning, and Wasserstein distributionally robust optimization with side information (WDRO-SI). Due to the presence of contextual information, existing single-loop methods for classical stochastic bilevel optimization are unable to converge. To overcome this challenge, we introduce an efficient double-loop gradient method based on the Multilevel Monte-Carlo (MLMC) technique and establish its sample and computational complexities. When specialized to stochastic nonconvex optimization, our method matches  existing lower bounds. For meta-learning, the complexity of our method does not depend on the number of tasks. Numerical experiments further validate our theoretical results.</details>

### Continuous Parametric Optical Flow

**Authors:** Jianqin Luo, Zhexiong Wan, yuxin mao, Bo Li, Yuchao Dai

**<details><summary>Abstract**</summary>

In this paper, we present continuous parametric optical flow, a parametric representation of dense and continuous motion over arbitrary time interval. In contrast to existing discrete-time representations (i.e., flow in between consecutive frames), this new representation transforms the frame-to-frame pixel correspondences to dense continuous flow. In particular, we present a temporal-parametric model that employs B-splines to fit point trajectories using a limited number of frames. To further improve the stability and robustness of the trajectories, we also add an encoder with a neural ordinary differential equation (ODE) to represent features associated with specific times. We also contribute a synthetic dataset and introduce two evaluation perspectives to measure the accuracy and robustness of continuous flow estimation. Benefiting from the combination of explicit parametric modeling and implicit feature optimization, our model focuses on motion continuity and outperforms than the flow-based and point-tracking approaches for fitting long-term and variable sequences.</details>

### Contrast, Attend and Diffuse to Decode High-Resolution Images from Brain Activities

**Authors:** Jingyuan Sun, Mingxiao Li, Zijiao Chen, Yunhao Zhang, Shaonan Wang, Marie-Francine Moens

**<details><summary>Abstract**</summary>

Decoding visual stimuli from neural responses recorded by functional Magnetic Resonance Imaging (fMRI) presents an intriguing intersection between cognitive neuroscience and machine learning, promising advancements in understanding human visual perception. However, the task is challenging due to the noisy nature of fMRI signals and the intricate pattern of brain visual representations. To mitigate these challenges, we introduce a two-phase fMRI representation learning framework. The first phase pre-trains an fMRI feature learner with a proposed Double-contrastive Mask Auto-encoder to learn denoised representations. The second phase tunes the feature learner to attend to neural activation patterns most informative for visual reconstruction with guidance from an image auto-encoder. The optimized fMRI feature learner then conditions a latent diffusion model to reconstruct image stimuli from brain activities. Experimental results demonstrate our model's superiority in generating high-resolution and semantically accurate images, substantially exceeding previous state-of-the-art methods by 39.34% in the 50-way-top-1 semantic classification accuracy. The code implementations will be available at https://github.com/soinx0629/vis_dec_neurips/.</details>

### [Spotlight] Contrastive Lift: 3D Object Instance Segmentation by Slow-Fast Contrastive Fusion

**Authors:** Yash Bhalgat, Iro Laina, João Henriques, Andrea Vedaldi, Andrew Zisserman

**<details><summary>Abstract**</summary>

Instance segmentation in 3D is a challenging task due to the lack of large-scale annotated datasets. In this paper, we show that this task can be addressed effectively by leveraging instead 2D pre-trained models for instance segmentation. We propose a novel approach to lift 2D segments to 3D and fuse them by means of a neural field representation, which encourages multi-view consistency across frames. The core of our approach is a slow-fast clustering objective function, which is scalable and well-suited for scenes with a large number of objects. Unlike previous approaches, our method does not require an upper bound on the number of objects or object tracking across frames. To demonstrate the scalability of the slow-fast clustering, we create a new semi-realistic dataset called the Messy Rooms dataset, which features scenes with up to 500 objects per scene. Our approach outperforms the state-of-the-art on challenging scenes from the ScanNet, Hypersim, and Replica datasets, as well as on our newly created Messy Rooms dataset, demonstrating the effectiveness and scalability of our slow-fast clustering method.</details>

### Contrastive Sampling Chains in Diffusion Models

**Authors:** Junyu Zhang, Daochang Liu, Shichao Zhang, Chang Xu

**<details><summary>Abstract**</summary>

The past few years have witnessed great success in the use of diffusion models (DMs) to generate high-fidelity images with the help of stochastic differential equations (SDEs). However, discretization error is an inevitable limitation when utilizing numerical solvers to solve SDEs. To address this limitation, we provide a theoretical analysis demonstrating that an appropriate combination of the contrastive loss and score matching serves as an upper bound of the KL divergence between the true data distribution and the model distribution. To obtain this bound, we utilize a contrastive loss to construct a contrastive sampling chain to fine-tuning the pre-trained DM. In this manner, our method reduces the discretization error and thus yields a smaller gap between the true data distribution and our model distribution. Moreover, the presented method can be applied to fine-tuning various pre-trained DMs, both with or without fast sampling algorithms, contributing to better sample quality or slightly faster sampling speeds. To validate the efficacy of our method, we conduct comprehensive experiments. For example, on CIFAR10, when applied to a pre-trained EDM, our method improves the FID from 2.04 to 1.88 with 35 neural function evaluations (NFEs), and reduces NFEs from 35 to 25 to achieve the same 2.04 FID.</details>

### Convergence analysis of ODE models for accelerated first-order methods via positive semidefinite kernels

**Authors:** Jungbin Kim, Insoon Yang

**<details><summary>Abstract**</summary>

We propose a novel methodology that systematically analyzes ordinary differential equation (ODE) models for first-order optimization methods by converting the task of proving convergence rates into verifying the positive semidefiniteness of specific Hilbert-Schmidt integral operators. Our approach is based on the performance estimation problems (PEP) introduced by Drori and Teboulle. Unlike previous works on PEP, which rely on finite-dimensional linear algebra, we use tools from functional analysis. Using the proposed method, we establish convergence rates of various accelerated gradient flow models, some of which are new. As an immediate consequence of our framework, we show a correspondence between minimizing function values and minimizing gradient norms.</details>

### [Spotlight] Convergence of Alternating Gradient Descent for Matrix Factorization

**Authors:** Rachel Ward, Tamara Kolda

**<details><summary>Abstract**</summary>

We consider alternating gradient descent (AGD) with fixed step size applied to the asymmetric matrix factorization objective.  We show that, for a rank-$r$ matrix $A \in \mathbb{R}^{m \times n}$,  $T = C ( \frac{\sigma_1(A)}{\sigma_r(A)} )^2 \log(1/\epsilon)$  iterations of alternating gradient descent suffice to reach an $\epsilon$-optimal factorization   $\| A - X_{T} Y_{T}' \|^2 \leq \epsilon \| A \|^2$   with high probability  starting from an atypical random initialization. The  factors have rank $d \geq r$ so that $X_{T}\in \mathbb{R}^{m \times d}$ and $Y_{T} \in\mathbb{R}^{n \times d}$, and mild overparameterization suffices for the constant  $C$ in the iteration complexity $T$ to be an absolute constant.   Experiments suggest that our proposed initialization is not merely of theoretical benefit, but rather significantly improves the convergence rate of gradient descent in practice. Our proof is conceptually simple: a uniform Polyak-Lojasiewicz (PL) inequality and uniform Lipschitz smoothness constant are guaranteed for a sufficient number of iterations, starting from our random initialization.  Our proof method should be useful for extending and simplifying convergence analyses for a broader class of nonconvex low-rank factorization problems.</details>

### Convergent Bregman Plug-and-Play Image Restoration for Poisson Inverse Problems

**Authors:** Samuel Hurault, Ulugbek Kamilov, Arthur Leclaire, Nicolas Papadakis

**<details><summary>Abstract**</summary>

Plug-and-Play (PnP) methods are efficient iterative algorithms for solving ill-posed image inverse problems. PnP methods are obtained by using deep Gaussian denoisers instead of the proximal operator or the gradient-descent step within proximal algorithms. Current PnP schemes rely on data-fidelity terms that have either Lipschitz gradients or closed-form proximal operators, which is not applicable to Poisson inverse problems. Based on the observation that the Gaussian noise is not the adequate noise model in this setting, we propose to generalize PnP using the Bregman Proximal Gradient (BPG) method. BPG replaces the Euclidean distance with a Bregman divergence that can better capture the smoothness properties of the problem. We introduce the Bregman Score Denoiser specifically parametrized and trained for the new Bregman geometry and prove that it corresponds to the proximal operator of a nonconvex potential. We propose two PnP algorithms based on the Bregman Score Denoiser for solving Poisson inverse problems. Extending the convergence results of BPG in the nonconvex settings, we show that the proposed methods converge, targeting stationary points of an explicit global functional. Experimental evaluations conducted on various Poisson inverse problems validate the convergence results and showcase effective restoration performance.</details>

### Cookie Consent Has Disparate Impact on Estimation Accuracy

**Authors:** Erik Miehling, Rahul Nair, Elizabeth Daly, Karthikeyan Natesan Ramamurthy, Robert Redmond

**<details><summary>Abstract**</summary>

Cookies are designed to enable more accurate identification and tracking of user behavior, in turn allowing for more personalized ads and better performing ad campaigns. Given the additional information that is recorded, questions related to privacy and fairness naturally arise. How does a user's consent decision influence how much the system can learn about their demographic and tastes? Is the impact of a user's consent decision on the recommender system's ability to learn about their latent attributes uniform across demographics? We investigate these questions in the context of an engagement-driven recommender system using simulation. We empirically demonstrate that when consent rates exhibit demographic-dependence, user consent has a disparate impact on the recommender agent's ability to estimate users' latent attributes. In particular, we find that when consent rates are demographic-dependent, a user disagreeing to share their cookie may counter-intuitively cause the recommender agent to know more about the user than if the user agreed to share their cookie. Furthermore, the gap in base consent rates across demographics serves as an amplifier: users from the lower consent rate demographic who agree to cookie sharing generally experience higher estimation errors than the same users from the higher consent rate demographic, and conversely for users who choose to disagree to cookie sharing, with these differences increasing in consent rate gap. We discuss the need for new notions of fairness that encourage consistency between a user's privacy decisions and the system's ability to estimate their latent attributes.</details>

### Correlation Aware Sparsified Mean Estimation Using Random Projection

**Authors:** Shuli Jiang, PRANAY SHARMA, Gauri Joshi

**<details><summary>Abstract**</summary>

We study the problem of communication-efficient distributed vector mean estimation, which is a commonly used subroutine in distributed optimization and Federated Learning (FL). Rand-$k$ sparsification is a commonly used technique to reduce communication cost, where each client sends $k < d$ of its coordinates to the server. However, Rand-$k$ is agnostic to any correlations, that might exist between clients in practical scenarios. The recently proposed Rand-$k$-Spatial estimator leverages the cross-client correlation information at the server to improve Rand-$k$'s performance. Yet, the performance of Rand-$k$-Spatial is suboptimal, and improving mean estimation is key to a faster convergence in distributed optimization. We propose the Rand-Proj-Spatial estimator with a more flexible encoding-decoding procedure, which generalizes the encoding of Rand-$k$ by projecting the client vectors to a random $k$-dimensional subspace. We utilize Subsampled Randomized Hadamard Transform (SRHT) as the projection matrix, and show that Rand-Proj-Spatial with SRHT outperforms Rand-$k$-Spatial, using the correlation information more efficiently. Furthermore, we propose an approach to incorporate varying degrees of correlation, and suggest a practical variant of Rand-Proj-Spatial when the correlation information is not available to the server. Finally, experiments on real-world distributed optimization tasks showcase the superior performance of Rand-Proj-Spatial compared to Rand-$k$-Spatial and other more sophisticated sparsification techniques.</details>

### Correlative Information Maximization: A Biologically Plausible Approach to Supervised Deep Neural Networks without Weight Symmetry

**Authors:** Bariscan Bozkurt, Cengiz Pehlevan, Alper Erdogan

**<details><summary>Abstract**</summary>

The backpropagation algorithm has experienced remarkable success in training large-scale artificial neural networks; however, its biological plausibility has been strongly criticized, and it remains an open question whether the brain employs supervised learning mechanisms akin to it. Here, we propose correlative information maximization between layer activations as an alternative normative approach to describe the signal propagation in biological neural networks in both forward and backward directions. This new framework addresses many concerns about the biological-plausibility of conventional artificial neural networks and the backpropagation algorithm. The coordinate descent-based optimization of the corresponding objective, combined with the mean square error loss function for fitting labeled supervision data, gives rise to a neural network structure that emulates a more biologically realistic network of multi-compartment pyramidal neurons with dendritic processing and lateral inhibitory neurons. Furthermore, our approach provides a natural resolution to the weight symmetry problem between forward and backward signal propagation paths, a significant critique against the plausibility of the conventional backpropagation algorithm. This is achieved by leveraging two alternative, yet equivalent forms of the correlative mutual information objective. These alternatives intrinsically lead to forward and backward prediction networks without weight symmetry issues, providing a compelling solution to this long-standing challenge.</details>

### CosNet: A Generalized Spectral Kernel Network

**Authors:** Yanfang Xue, Pengfei Fang, Jinyue Tian, Shipeng Zhu, hui xue

**<details><summary>Abstract**</summary>

Complex-valued representation exists inherently in the time-sequential data that can be derived from the integration of harmonic waves. The non-stationary spectral kernel, realizing a complex-valued feature mapping, has shown its potential to analyze the time-varying statistical characteristics of the time-sequential data, as a result of the modeling frequency parameters. However, most existing spectral kernel-based methods eliminate the imaginary part, thereby limiting the representation power of the spectral kernel. To tackle this issue, we propose a generalized spectral kernel network, namely, \underline{Co}mplex-valued \underline{s}pectral kernel \underline{Net}work (CosNet), which includes spectral kernel mapping generalization (SKMG) module and complex-valued spectral kernel embedding (CSKE) module. Concretely, the SKMG module is devised to generalize the spectral kernel mapping in the real number domain to the complex number domain, recovering the inherent complex-valued representation for the real-valued data. Then a following CSKE module is further developed to combine the complex-valued spectral kernels and neural networks to effectively capture long-range or periodic relations of the data. Along with the CosNet, we study the effect of the complex-valued spectral kernel mapping via theoretically analyzing the bound of covering number and generalization error. Extensive experiments demonstrate that CosNet performs better than the mainstream kernel methods and complex-valued neural networks.</details>

### Counterfactual Generation with Identifiability Guarantees

**Authors:** Hanqi Yan, Lingjing Kong, Lin Gui, Yuejie Chi, Eric Xing, Yulan He, Kun Zhang

**<details><summary>Abstract**</summary>

Counterfactual generation lies at the core of various machine learning tasks, including image translation and controllable text generation. This generation process usually requires the identification of the disentangled latent representations, such as content and style, that underlie the observed data. However, it becomes more  challenging when faced with a scarcity of paired data and labelling information. Existing disentangled methods crucially rely on oversimplified assumptions, such as assuming independent content and style variables, to identify the latent variables, even though such assumptions may not hold for complex data distributions. For instance, food reviews tend to involve words like “tasty”, whereas movie reviews commonly contain words such as “thrilling” for the same positive sentiment. This problem is exacerbated when data are sampled from multiple domains since the dependence between content and style may vary significantly over domains. In this work, we tackle the domain-varying dependence between the content and the style variables inherent in the counterfactual generation task. We provide identification guarantees for such latent-variable models by leveraging the relative sparsity of the influences from different latent variables. Our theoretical insights enable the development of a doMain AdapTive counTerfactual gEneration model, called (MATTE). Our theoretically grounded framework achieves state-of-the-art performance in unsupervised style transfer tasks, where neither paired data nor style labels are utilized, across four large-scale datasets.</details>

### Counterfactual-Augmented Importance Sampling for Semi-Offline Policy Evaluation

**Authors:** Shengpu Tang, Jenna Wiens

**<details><summary>Abstract**</summary>

In applying reinforcement learning (RL) to high-stakes domains, quantitative and qualitative evaluation using observational data can help practitioners understand the generalization performance of new policies. However, this type of off-policy evaluation (OPE) is inherently limited since offline data may not reflect the distribution shifts resulting from the application of new policies. On the other hand, online evaluation by collecting rollouts according to the new policy is often infeasible, as deploying new policies in these domains can be unsafe. In this work, we propose a semi-offline evaluation framework as an intermediate step between offline and online evaluation, where human users provide annotations of unobserved counterfactual trajectories. While tempting to simply augment existing data with such annotations, we show that this naive approach can lead to biased results. Instead, we design a new family of OPE estimators based on importance sampling (IS) and a novel weighting scheme that incorporate counterfactual annotations without introducing additional bias. We analyze the theoretical properties of our approach, showing its potential to reduce both bias and variance compared to standard IS estimators. Our analyses reveal important practical considerations for handling biased, noisy, or missing annotations. In a series of proof-of-concept experiments involving bandits and a healthcare-inspired simulator, we demonstrate that our approach outperforms purely offline IS estimators and is robust to imperfect annotations. Our framework, combined with principled human-centered design of annotation solicitation, can enable the application of RL in high-stakes domains.</details>

### Cross-Episodic Curriculum for Transformer Agents

**Authors:** Lucy Xiaoyang Shi, Yunfan Jiang, Jake Grigsby, Linxi Fan, Yuke Zhu

**<details><summary>Abstract**</summary>

We present a new algorithm, Cross-Episodic Curriculum (CEC), to boost the learning efficiency and generalization of Transformer agents. Central to CEC is the placement of cross-episodic experiences into a Transformer’s context, which forms the basis of a curriculum. By sequentially structuring online learning trials and mixed-quality demonstrations, CEC constructs curricula that encapsulate learning progression and proficiency increase across episodes. Such synergy combined with the potent pattern recognition capabilities of Transformer models delivers a powerful cross-episodic attention mechanism. The effectiveness of CEC is demonstrated under two representative scenarios: one involving multi-task reinforcement learning with discrete control, such as in DeepMind Lab, where the curriculum captures the learning progression in both individual and progressively complex settings; and the other involving imitation learning with mixed-quality data for continuous control, as seen in RoboMimic, where the curriculum captures the improvement in demonstrators' expertise. In all instances, policies resulting from CEC exhibit superior performance and strong generalization. Code is open-sourced on the project website https://cec-agent.github.io/ to facilitate research on Transformer agent learning.</details>

### DAMEX: Dataset-aware Mixture-of-Experts for visual understanding of mixture-of-datasets

**Authors:** Yash Jain, Harkirat Behl, Zsolt Kira, Vibhav Vineet

**<details><summary>Abstract**</summary>

Construction of a universal detector poses a crucial question: How can we most effectively train a model on a large mixture of datasets? The answer lies in learning dataset-specific features and ensembling their knowledge but do all this in a single model. Previous methods achieve this by having separate detection heads on a common backbone but that results in a significant increase in parameters. In this work, we present Mixture-of-Experts as a solution, highlighting that MoE are much more than a scalability tool. We propose Dataset-Aware Mixture-of-Experts, DAMEX where we train the experts to become an `expert' of a dataset by learning to route each dataset tokens to its mapped expert. Experiments on Universal Object-Detection Benchmark show that we outperform the existing state-of-the-art by average +10.2 AP score and improve over our non-MoE baseline by average +2.0 AP score. We also observe consistent gains while mixing datasets with (1) limited availability, (2) disparate domains and (3) divergent label sets. Further, we qualitatively show that DAMEX is robust against expert representation collapse. Code is available at https://github.com/jinga-lala/DAMEX</details>

### DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting

**Authors:** Salva Rühling Cachay, Bo Zhao, Hailey Joren, Rose Yu

**<details><summary>Abstract**</summary>

While diffusion models can successfully generate data and make predictions, they are predominantly designed for static images. We propose an approach for training diffusion models for dynamics forecasting that leverages the temporal dynamics encoded in the data, directly coupling it with the diffusion steps in the network. We train a stochastic, time-conditioned interpolator and a backbone forecaster networkthat mimic the forward and reverse processes of conventional diffusion models, respectively. This design choice naturally encodes multi-step and long-range forecasting capabilities, allowing for highly flexible, continuous-time sampling trajectories and the ability to trade-off performance with accelerated sampling at inference time. In addition, the dynamics-informed diffusion process imposes a strong inductive bias, allowing for improved computational efficiency compared to traditional Gaussian noise-based diffusion models. Our approach performs competitively on probabilistic skill score metrics in complex dynamics forecasting of sea surface temperatures, Navier-Stokes flows, and spring mesh systems.</details>

### Data Minimization at Inference Time

**Authors:** Cuong Tran, Nando Fioretto

**<details><summary>Abstract**</summary>

In high-stakes domains such as legal, banking, hiring, and healthcare, learning models frequently rely on sensitive user information for inference, necessitating the complete set of features. This not only poses significant privacy risks for individuals but also demands substantial human effort from organizations to verify information accuracy. This study asks whether it is necessary to use all input features for accurate predictions at inference time. The paper demonstrates that, in a personalized setting, individuals may only need to disclose a small subset of features without compromising decision-making accuracy. The paper also provides an efficient sequential algorithm to determine the appropriate attributes for each individual to provide. Evaluations across various learning tasks show that individuals can potentially report as little as 10\% of their information while maintaining the same accuracy level as a model that employs the full set of user information.</details>

### Dataset Diffusion: Diffusion-based Synthetic Data Generation for Pixel-Level Semantic Segmentation

**Authors:** Quang Nguyen, Truong Vu, Anh Tran, Khoi Nguyen

**<details><summary>Abstract**</summary>

Preparing training data for deep vision models is a labor-intensive task. To address this, generative models have emerged as an effective solution for generating synthetic data. While current generative models produce image-level category labels, we propose a novel method for generating pixel-level semantic segmentation labels using the text-to-image generative model Stable Diffusion (SD). By utilizing the text prompts, cross-attention, and self-attention of SD, we introduce three new techniques: class-prompt appending, class-prompt cross-attention, and self-attention exponentiation. These techniques enable us to generate segmentation maps corresponding to synthetic images. These maps serve as pseudo-labels for training semantic segmenters, eliminating the need for labor-intensive pixel-wise annotation. To account for the imperfections in our pseudo-labels, we incorporate uncertainty regions into the segmentation, allowing us to disregard loss from those regions. We conduct evaluations on two datasets, PASCAL VOC and MSCOCO, and our approach significantly outperforms concurrent work. Our benchmarks and code will be released at https://github.com/VinAIResearch/Dataset-Diffusion.</details>

### [Spotlight] DeWave: Discrete Encoding of EEG Waves for EEG to Text Translation

**Authors:** Yiqun Duan, Charles Chau, Zhen Wang, Yu-Kai Wang, Chin-teng Lin

**<details><summary>Abstract**</summary>

The translation of brain dynamics into natural language is pivotal for brain-computer interfaces (BCIs), a field that has seen substantial growth in recent years. With the swift advancement of large language models, such as ChatGPT, the need to bridge the gap between the brain and languages becomes increasingly pressing. Current methods, however, require eye-tracking fixations or event markers to segment brain dynamics into word-level features, which can restrict the practical application of these systems. These event markers may not be readily available or could be challenging to acquire during real-time inference, and the sequence of eye fixations may not align with the order of spoken words. To tackle these issues, we introduce a novel framework, DeWave, that integrates discrete encoding sequences into open-vocabulary EEG-to-text translation tasks. DeWave uses a quantized variational encoder to derive discrete codex encoding and align it with pre-trained language models. This discrete codex representation brings forth two advantages: 1) it alleviates the order mismatch between eye fixations and spoken words by introducing text-EEG contrastive alignment training, and 2) it minimizes the interference caused by individual differences in EEG waves through an invariant discrete codex. Our model surpasses the previous baseline (40.1 and 31.7) by 3.06% and 6.34\%, respectively, achieving 41.35 BLEU-1 and 33.71 Rouge-F on the ZuCo Dataset. Furthermore, this work is the first to facilitate the translation of entire EEG signal periods without the need for word-level order markers (e.g., eye fixations), scoring 20.5 BLEU-1 and 29.5 Rouge-1 on the ZuCo Dataset, respectively.</details>

### Decompose Novel into Known: Part Concept Learning For 3D Novel Class Discovery

**Authors:** Tingyu Weng, Jun Xiao, Haiyong Jiang

**<details><summary>Abstract**</summary>

In this work, we address 3D novel class discovery (NCD) that discovers novel classes from an unlabeled dataset by leveraging the knowledge of disjoint known classes. The key challenge of 3D NCD is that learned features by known class recognition are heavily biased and hinder generalization to novel classes. Since geometric parts are more generalizable across different classes, we propose to decompose novel into known parts, coined DNIK, to mitigate the above problems. DNIK learns a part concept bank encoding rich part geometric patterns from known classes so that novel 3D shapes can be represented as part concept compositions to facilitate cross-category generalization. Moreover, we formulate three constraints on part concepts to ensure diverse part concepts without collapsing. A part relation encoding module (PRE) is also developed to leverage part-wise spatial relations for better recognition. We construct three 3D NCD tasks for evaluation and extensive experiments show that our method achieves significantly superior results than SOTA baselines (+11.7%, +14.1%, and +16.3% improvements on average for three tasks, respectively). Code and data will be released.</details>

### Decompose a Task into Generalizable Subtasks in Multi-Agent Reinforcement Learning

**Authors:** Zikang Tian, Ruizhi Chen, Xing Hu, Ling Li, Rui Zhang, Fan Wu, Shaohui Peng, Jiaming Guo, Zidong Du, Qi Guo, Yunji Chen

**<details><summary>Abstract**</summary>

In recent years, Multi-Agent Reinforcement Learning (MARL) techniques have made significant strides in achieving high asymptotic performance in single task. However, there has been limited exploration of model transferability across tasks. Training a model from scratch for each task can be time-consuming and expensive, especially for large-scale Multi-Agent Systems. Therefore, it is crucial to develop methods for generalizing the model across tasks. Considering that there exist task-independent subtasks across MARL tasks, a model that can decompose such subtasks from the source task could generalize to target tasks. However, ensuring true task-independence of subtasks poses a challenge. In this paper, we propose to \textbf{d}ecompose a \textbf{t}ask in\textbf{to} a series of \textbf{g}eneralizable \textbf{s}ubtasks (DT2GS), a novel framework that addresses this challenge by utilizing a scalable subtask encoder and an adaptive subtask semantic module. We show that these components endow subtasks with two properties critical for task-independence: avoiding overfitting to the source task and maintaining consistent yet scalable semantics across tasks. Empirical results demonstrate that DT2GS possesses sound zero-shot generalization capability across tasks, exhibits sufficient transferability, and outperforms existing methods in both multi-task and single-task problems.</details>

### [Spotlight] Deep Fractional Fourier Transform

**Authors:** Hu Yu, Jie Huang, Lingzhi LI, man zhou, Feng Zhao

**<details><summary>Abstract**</summary>

Existing deep learning-based computer vision methods usually operate in the spatial and frequency domains, which are two orthogonal \textbf{individual} perspectives for image processing.In this paper, we introduce a new spatial-frequency analysis tool, Fractional Fourier Transform (FRFT), to provide comprehensive \textbf{unified} spatial-frequency perspectives.The FRFT is a unified continuous spatial-frequency transform that simultaneously reflects an image's spatial and frequency representations, making it optimal for processing non-stationary image signals.We explore the properties of the FRFT for image processing and present a fast implementation of the 2D FRFT, which facilitates its widespread use.Based on these explorations, we introduce a simple yet effective operator, Multi-order FRactional Fourier Convolution (MFRFC), which exhibits the remarkable merits of processing images from more perspectives in the spatial-frequency plane. Our proposed MFRFC is a general and basic operator that can be easily integrated into various tasks for performance improvement.We experimentally evaluate the MFRFC on various computer vision tasks, including object detection, image classification, guided super-resolution, denoising, dehazing, deraining, and low-light enhancement. Our proposed MFRFC consistently outperforms baseline methods by significant margins across all tasks.</details>

### Deep Gaussian Markov Random Fields for Graph-Structured Dynamical Systems

**Authors:** Fiona Lippert, Bart Kranstauber, Emiel van Loon, Patrick Forré

**<details><summary>Abstract**</summary>

Probabilistic inference in high-dimensional state-space models is computationally challenging. For many spatiotemporal systems, however, prior knowledge about the dependency structure of state variables is available. We leverage this structure to develop a computationally efficient approach to state estimation and learning in graph-structured state-space models with (partially) unknown dynamics and limited historical data. Building on recent methods that combine ideas from deep learning with principled inference in Gaussian Markov random fields (GMRF), we reformulate graph-structured state-space models as Deep GMRFs defined by simple spatial and temporal graph layers. This results in a flexible spatiotemporal prior that can be learned efficiently from a single time sequence via variational inference. Under linear Gaussian assumptions, we retain a closed-form posterior, which can be sampled efficiently using the conjugate gradient method, scaling favourably compared to classical Kalman filter based approaches.</details>

### Deep Insights into Noisy Pseudo Labeling on Graph Data

**Authors:** Botao WANG, Jia Li, Yang Liu, Jiashun Cheng, Yu Rong, Wenjia Wang, Fugee Tsung

**<details><summary>Abstract**</summary>

Pseudo labeling (PL) is a wide-applied strategy to enlarge the labeled dataset by self-annotating the potential samples during the training process. Several works have shown that it can improve the graph learning model performance in general. However, we notice that the incorrect labels can be fatal to the graph training process. Inappropriate PL may result in the performance degrading, especially on graph data where the noise can propagate. Surprisingly, the corresponding error is seldom theoretically analyzed in the literature. In this paper, we aim to give deep insights of PL on graph learning models. We first present the error analysis of PL strategy by showing that the error is bounded by the confidence of PL threshold and consistency of multi-view prediction. Then, we theoretically illustrate the effect of PL on convergence property. Based on the analysis, we propose a cautious pseudo labeling methodology in which we pseudo label the samples with highest confidence and multi-view consistency. Finally, extensive experiments demonstrate that the proposed strategy improves graph learning process and outperforms other PL strategies on link prediction and node classification tasks.</details>

### Deep Patch Visual Odometry

**Authors:** Zachary Teed, Lahav Lipson, Jia Deng

**<details><summary>Abstract**</summary>

We propose Deep Patch Visual Odometry (DPVO), a new deep learning system for monocular Visual Odometry (VO). DPVO uses a novel recurrent network architecture designed for tracking image patches across time. Recent approaches to VO have significantly improved the state-of-the-art accuracy by using deep networks to predict dense flow between video frames. However, using dense flow incurs a large computational cost, making these previous methods impractical for many use cases. Despite this, it has been assumed that dense flow is important as it provides additional redundancy against incorrect matches. DPVO disproves this assumption, showing that it is possible to get the best accuracy and efficiency by exploiting the advantages of sparse patch-based matching over dense flow. DPVO introduces a novel recurrent update operator for patch based correspondence coupled with differentiable bundle adjustment. On Standard benchmarks, DPVO outperforms all prior work, including the learning-based state-of-the-art VO-system (DROID) using a third of the memory while running 3x faster on average. Code is available at https://github.com/princeton-vl/DPVO</details>

### Deep Recurrent Optimal Stopping

**Authors:** Niranjan Damera Venkata, Chiranjib Bhattacharyya

**<details><summary>Abstract**</summary>

Deep neural networks (DNNs) have recently emerged as a powerful paradigm for solving Markovian optimal stopping problems. However, a ready extension of DNN-based methods to non-Markovian settings requires significant state and parameter space expansion, manifesting the curse of dimensionality. Further, efficient state-space transformations permitting Markovian approximations, such as those afforded by recurrent neural networks (RNNs), are either structurally infeasible or are confounded by the curse of non-Markovianity. Considering these issues, we introduce, for the first time, an optimal stopping policy gradient algorithm (OSPG) that can leverage RNNs effectively in non-Markovian settings by implicitly optimizing value functions without recursion, mitigating the curse of non-Markovianity. The OSPG algorithm is derived from an inference procedure on a novel Bayesian network representation of discrete-time non-Markovian optimal stopping trajectories and, as a consequence, yields an offline policy gradient algorithm that eliminates expensive Monte Carlo policy rollouts.</details>

### DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization

**Authors:** Haoran Ye, Jiarui Wang, Zhiguang Cao, Helan Liang, Yong Li

**<details><summary>Abstract**</summary>

Ant Colony Optimization (ACO) is a meta-heuristic algorithm that has been successfully applied to various Combinatorial Optimization Problems (COPs). Traditionally, customizing ACO for a specific problem requires the expert design of knowledge-driven heuristics. In this paper, we propose DeepACO, a generic framework that leverages deep reinforcement learning to automate heuristic designs. DeepACO serves to strengthen the heuristic measures of existing ACO algorithms and dispense with laborious manual design in future ACO applications. As a neural-enhanced meta-heuristic, DeepACO consistently outperforms its ACO counterparts on eight COPs using a single neural model and a single set of hyperparameters. As a Neural Combinatorial Optimization method, DeepACO performs better than or on par with problem-specific methods on canonical routing problems. Our code is publicly available at https://github.com/henry-yeh/DeepACO.</details>

### DeepPCR: Parallelizing Sequential Operations in Neural Networks

**Authors:** Federico Danieli, Miguel Sarabia, Xavier Suau Cuadros, Pau Rodriguez, Luca Zappella

**<details><summary>Abstract**</summary>

Parallelization techniques have become ubiquitous for accelerating inference and training of deep neural networks. Despite this, several operations are still performed in a sequential manner. For instance, the forward and backward passes are executed layer-by-layer, and the output of diffusion models is produced by applying a sequence of denoising steps. This sequential approach results in a computational cost proportional to the number of steps involved, presenting a potential bottleneck as the number of steps increases. In this work, we introduce DeepPCR, a novel algorithm which parallelizes typically sequential operations in order to speed up inference and training of neural networks. DeepPCR is based on interpreting a sequence of $L$ steps as the solution of a specific system of equations, which we recover using the Parallel Cyclic Reduction algorithm. This reduces the complexity of computing the sequential operations from $\mathcal{O}(L)$ to $\mathcal{O}(\log_2L)$, thus yielding a speedup for large $L$. To verify the theoretical lower complexity of the algorithm, and to identify regimes for speedup, we test the effectiveness of DeepPCR in parallelizing the forward and backward pass in multi-layer perceptrons, and reach speedups of up to $30\times$ for the forward and $200\times$ for the backward pass. We additionally showcase the flexibility of DeepPCR by parallelizing training of ResNets with as many as 1024 layers, and generation in diffusion models, enabling up to $7\times$ faster training and $11\times$ faster generation, respectively, when compared to the sequential approach.</details>

### DeepSimHO: Stable Pose Estimation for Hand-Object Interaction via Physics Simulation

**Authors:** Rong Wang, Wei Mao, Hongdong Li

**<details><summary>Abstract**</summary>

This paper addresses the task of 3D pose estimation for a hand interacting with an object from a single image observation. When modeling hand-object interaction, previous works mainly exploit proximity cues, while overlooking the dynamical nature that the hand must stably grasp the object to counteract gravity and thus preventing the object from slipping or falling. These works fail to leverage dynamical constraints in the estimation and consequently often produce unstable results. Meanwhile, refining unstable configurations with physics-based reasoning remains challenging, both by the complexity of contact dynamics and by the lack of effective and efficient physics inference in the data-driven learning framework. To address both issues, we present DeepSimHO: a novel deep-learning pipeline that combines forward physics simulation and backward gradient approximation with a neural network. Specifically, for an initial hand-object pose estimated by a base network, we forward it to a physics simulator to evaluate its stability. However, due to non-smooth contact geometry and penetration, existing differentiable simulators can not provide reliable state gradient. To remedy this, we further introduce a deep network to learn the stability evaluation process from the simulator, while smoothly approximating its gradient and thus enabling effective back-propagation. Extensive experiments show that our method noticeably improves the stability of the estimation and achieves superior efficiency over test-time optimization. The code is available at https://github.com/rongakowang/DeepSimHO.</details>

### [Spotlight] Delegated Classification

**Authors:** Eden Saig, Inbal Talgam-Cohen, Nir Rosenfeld

**<details><summary>Abstract**</summary>

When machine learning is outsourced to a rational agent, conflicts of interest might arise and severely impact predictive performance. In this work, we propose a theoretical framework for incentive-aware delegation of machine learning tasks. We model delegation as a principal-agent game, in which accurate learning can be incentivized by the principal using performance-based contracts. Adapting the economic theory of contract design to this setting, we define budget-optimal contracts and prove they take a simple threshold form under reasonable assumptions. In the binary-action case, the optimality of such contracts is shown to be equivalent to the classic Neyman-Pearson lemma, establishing a formal connection between contract design and statistical hypothesis testing. Empirically, we demonstrate that budget-optimal contracts can be constructed using small-scale data, leveraging recent advances in the study of learning curves and scaling laws. Performance and economic outcomes are evaluated using synthetic and real-world classification tasks.</details>

### [Spotlight] Demystifying Softmax Gating Function in Gaussian Mixture of Experts

**Authors:** Huy Nguyen, TrungTin Nguyen, Nhat Ho

**<details><summary>Abstract**</summary>

Understanding the parameter estimation of softmax gating Gaussian mixture of experts has remained a long-standing open problem in the literature. It is mainly due to three fundamental theoretical challenges associated with the softmax gating function: (i) the identifiability only up to the translation of parameters; (ii) the intrinsic interaction via partial differential equations between the softmax gating and the expert functions in the Gaussian density; (iii) the complex dependence between the numerator and denominator of the conditional density of softmax gating Gaussian mixture of experts. We resolve these challenges by proposing novel Voronoi loss functions among parameters and establishing the convergence rates of maximum likelihood estimator (MLE) for solving parameter estimation in these models. When the true number of experts is unknown and over-specified, our findings show a connection between the convergence rate of the MLE and a solvability problem of a system of polynomial equations.</details>

### Dense-Exponential Random Features: Sharp Positive Estimators of the Gaussian Kernel

**Authors:** Valerii Likhosherstov, Krzysztof M Choromanski, Kumar Avinava Dubey, Frederick Liu, Tamas Sarlos, Adrian Weller

**<details><summary>Abstract**</summary>

The problem of efficient approximation of a linear operator induced by the Gaussian or softmax kernel is often addressed using random features (RFs) which yield an unbiased approximation of the operator's result. Such operators emerge in important applications ranging from kernel methods to efficient Transformers. We propose parameterized, positive, non-trigonometric RFs which approximate Gaussian and softmax-kernels. In contrast to traditional RF approximations, parameters of these new methods can be optimized to reduce the variance of the approximation, and the optimum can be expressed in closed form. We show that our methods lead to variance reduction in practice (e^{10}-times smaller variance and beyond) and outperform previous methods in a kernel regression task. Using our proposed mechanism, we also present FAVOR#, a method for self-attention approximation in Transformers. We show that FAVOR# outperforms other random feature methods in speech modelling and natural language processing.</details>

### Depth-discriminative Metric Learning for Monocular 3D Object Detection

**Authors:** Wonhyeok Choi, Mingyu Shin, Sunghoon Im

**<details><summary>Abstract**</summary>

Monocular 3D object detection poses a significant challenge due to the lack of depth information in RGB images. Many existing methods strive to enhance the object depth estimation performance by allocating additional parameters for object depth estimation, utilizing extra modules or data. In contrast, we introduce a novel metric learning scheme that encourages the model to extract depth-discriminative features regardless of the visual attributes without increasing inference time and model size. Our method employs the distance-preserving function to organize the feature space manifold in relation to ground-truth object depth. The proposed $(K,B,\epsilon)$-quasi-isometric loss leverages predetermined pairwise distance restriction as guidance for adjusting the distance among object descriptors without disrupting the non-linearity of the natural feature manifold. Moreover, we introduce an auxiliary head for object-wise depth estimation, which enhances depth quality while maintaining the inference time. The broad applicability of our method is demonstrated through experiments that show improvements in overall performance when integrated into various baselines. The results show that our method consistently improves the performance of various baselines by 23.51\% and 5.78\% on average across KITTI and Waymo, respectively.</details>

### Detection Based Part-level Articulated Object Reconstruction from Single RGBD Image

**Authors:** Yuki Kawana, Tatsuya Harada

**<details><summary>Abstract**</summary>

We propose an end-to-end trainable, cross-category method for reconstructing multiple man-made articulated objects from a single RGBD image, focusing on part-level shape reconstruction and pose and kinematics estimation. We depart from previous works that rely on learning instance-level latent space, focusing on man-made articulated objects with predefined part counts. Instead, we propose a novel alternative approach that employs part-level representation, representing instances as combinations of detected parts. While our detect-then-group approach effectively handles instances with diverse part structures and various part counts, it faces issues of false positives, varying part sizes and scales, and an increasing model size due to end-to-end training. To address these challenges, we propose 1) test-time kinematics-aware part fusion to improve detection performance while suppressing false positives, 2) anisotropic scale normalization for part shape learning to accommodate various part sizes and scales, and 3) a balancing strategy for cross-refinement between feature space and output space to improve part detection while maintaining model size. Evaluation on both synthetic and real data demonstrates that our method successfully reconstructs variously structured multiple instances that previous works cannot handle, and outperforms prior works in shape reconstruction and kinematics estimation.</details>

### DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation

**Authors:** Shentong Mo, Enze Xie, Ruihang Chu, Lanqing Hong, Matthias Niessner, Zhenguo Li

**<details><summary>Abstract**</summary>

Recent Diffusion Transformers (i.e., DiT) have demonstrated their powerful effectiveness in generating high-quality 2D images. However, it is unclear how the Transformer architecture performs equally well in 3D shape generation, as previous 3D diffusion methods mostly adopted the U-Net architecture. To bridge this gap, we propose a novel Diffusion Transformer for 3D shape generation, named DiT-3D, which can directly operate the denoising process on voxelized point clouds using plain Transformers. Compared to existing U-Net approaches, our DiT-3D is more scalable in model size and produces much higher quality generations.Specifically, the DiT-3D adopts the design philosophy of DiT but modifies it by incorporating 3D positional and patch embeddings to aggregate input from voxelized point clouds.To reduce the computational cost of self-attention in 3D shape generation, we incorporate 3D window attention into Transformer blocks, as the increased 3D token length resulting from the additional dimension of voxels can lead to high computation.Finally, linear and devoxelization layers are used to predict the denoised point clouds. In addition, we empirically observe that the pre-trained DiT-2D checkpoint on ImageNet can significantly improve DiT-3D on ShapeNet.Experimental results on the ShapeNet dataset demonstrate that the proposed DiT-3D achieves state-of-the-art performance in high-fidelity and diverse 3D point cloud generation.</details>

### DiViNeT: 3D Reconstruction from Disparate Views using Neural Template Regularization

**Authors:** Aditya Vora, Akshay Gadi Patil, Hao Zhang

**<details><summary>Abstract**</summary>

We present a volume rendering-based neural surface reconstruction method that takes as few as three disparate RGB images as input. Our key idea is to regularize the reconstruction, which is severely ill-posed and leaving significant gaps between the sparse views, by learning a set of neural templates that act as surface priors. Our method, coined DiViNet, operates in two stages. The first stage learns the templates, in the form of 3D Gaussian functions, across different scenes, without 3D supervision. In the reconstruction stage, our predicted templates serve as anchors to help “stitch” the surfaces over sparse regions. We demonstrate that our approach is not only able to complete the surface geometry but also reconstructs surface details to a reasonable extent from few disparate input views. On the DTU and BlendedMVS datasets, our approach achieves the best reconstruction quality among existing methods in the presence of such sparse views and performs on par, if not better, with competing methods when dense views are employed as inputs.</details>

### DiffKendall: A Novel Approach for Few-Shot Learning with Differentiable Kendall's Rank Correlation

**Authors:** Kaipeng Zheng, Huishuai Zhang, Weiran Huang

**<details><summary>Abstract**</summary>

Few-shot learning aims to adapt models trained on the base dataset to novel tasks where the categories were not seen by the model before. This often leads to a relatively uniform distribution of feature values across channels on novel classes, posing challenges in determining channel importance for novel tasks. Standard few-shot learning methods employ geometric similarity metrics such as cosine similarity and negative Euclidean distance to gauge the semantic relatedness between two features. However, features with high geometric similarities may carry distinct semantics, especially in the context of few-shot learning. In this paper, we demonstrate that the importance ranking of feature channels is a more reliable indicator for few-shot learning than geometric similarity metrics. We observe that replacing the geometric similarity metric with Kendall’s rank correlation only during inference is able to improve the performance of few-shot learning across a wide range of methods and datasets with different domains. Furthermore, we propose a carefully designed differentiable loss for meta-training to address the non-differentiability issue of Kendall’s rank correlation. By replacing geometric similarity with differentiable Kendall’s rank correlation, our method can integrate with numerous existing few-shot approaches and is ready for integrating with future state-of-the-art methods that rely on geometric similarity metrics. Extensive experiments validate the efficacy of the rank-correlation-based approach, showcasing a significant improvement in few-shot learning.</details>

### DiffUTE: Universal Text Editing Diffusion Model

**Authors:** Haoxing Chen, Zhuoer Xu, Zhangxuan Gu, jun lan, 行 郑, Yaohui Li, Changhua Meng, Huijia Zhu, Weiqiang Wang

**<details><summary>Abstract**</summary>

Diffusion model based language-guided image editing has achieved great success recently. However, existing state-of-the-art diffusion models struggle with rendering correct text and text style during generation. To tackle this problem, we propose a universal self-supervised text editing diffusion model (DiffUTE), which aims to replace or modify words in the source image with another one while maintaining its realistic appearance. Specifically, we build our model on a diffusion model and carefully modify the network structure to enable the model for drawing multilingual characters with the help of glyph and position information. Moreover, we design a self-supervised learning framework to leverage large amounts of web data to improve the representation ability of the model. Experimental results show that our method achieves an impressive performance and enables controllable editing on in-the-wild images with high fidelity. Our code will be avaliable in \url{https://github.com/chenhaoxing/DiffUTE}.</details>

### [Spotlight] Differentiable Registration of Images and LiDAR Point Clouds with VoxelPoint-to-Pixel Matching

**Authors:** Junsheng Zhou, Baorui Ma, Wenyuan Zhang, Yi Fang, Yu-Shen Liu, Zhizhong Han

**<details><summary>Abstract**</summary>

Cross-modality registration between 2D images captured by cameras and 3D point clouds from LiDARs is a crucial task in computer vision and robotic. Previous methods estimate 2D-3D correspondences by matching point and pixel patterns learned by neural networks, and use Perspective-n-Points (PnP) to estimate rigid transformation during post-processing. However, these methods struggle to map points and pixels to a shared latent space robustly since points and pixels have very different characteristics with patterns learned in different manners (MLP and CNN), and they also fail to construct supervision directly on the transformation since the PnP is non-differentiable, which leads to unstable registration results. To address these problems, we propose to learn a structured cross-modality latent space to represent pixel features and 3D features via a differentiable probabilistic PnP solver. Specifically, we design a triplet network to learn VoxelPoint-to-Pixel matching, where we represent 3D elements using both voxels and points to learn the cross-modality latent space with pixels. We design both the voxel and pixel branch based on CNNs to operate convolutions on voxels/pixels represented in grids, and integrate an additional point branch to regain the information lost during voxelization. We train our framework end-to-end by imposing supervisions directly on the predicted pose distribution with a probabilistic PnP solver. To explore distinctive patterns of cross-modality features, we design a novel loss with adaptive-weighted optimization for cross-modality feature description. The experimental results on KITTI and nuScenes datasets show significant improvements over the state-of-the-art methods.</details>

### Differentially Private Decoupled Graph Convolutions for Multigranular Topology Protection

**Authors:** Eli Chien, Wei-Ning Chen, Chao Pan, Pan Li, Ayfer Ozgur, Olgica Milenkovic

**<details><summary>Abstract**</summary>

Graph Neural Networks (GNNs) have proven to be highly effective in solving real-world learning problems that involve graph-structured data. However, GNNs can also inadvertently expose sensitive user information and interactions through their model predictions. To address these privacy concerns, Differential Privacy (DP) protocols are employed to control the trade-off between provable privacy protection and model utility. Applying standard DP approaches to GNNs directly is not advisable due to two main reasons. First, the prediction of node labels, which relies on neighboring node attributes through graph convolutions, can lead to privacy leakage. Second, in practical applications, the privacy requirements for node attributes and graph topology may differ. In the latter setting, existing DP-GNN models fail to provide multigranular trade-offs between graph topology privacy, node attribute privacy, and GNN utility. To address both limitations, we propose a new framework termed Graph Differential Privacy (GDP), specifically tailored to graph learning. GDP ensures both provably private model parameters as well as private predictions. Additionally, we describe a novel unified notion of graph dataset adjacency to analyze the properties of GDP for different levels of graph topology privacy. Our findings reveal that DP-GNNs, which rely on graph convolutions, not only fail to meet the requirements for multigranular graph topology privacy but also necessitate the injection of DP noise that scales at least linearly with the maximum node degree. In contrast, our proposed Differentially Private Decoupled Graph Convolutions (DPDGCs) represent a more flexible and efficient alternative to graph convolutions that still provides the necessary guarantees of GDP. To validate our approach, we conducted extensive experiments on seven node classification benchmarking and illustrative synthetic datasets. The results demonstrate that DPDGCs significantly outperform existing DP-GNNs in terms of privacy-utility trade-offs.</details>

### Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence

**Authors:** Grace Luo, Lisa Dunlap, Dong Huk Park, Aleksander Holynski, Trevor Darrell

**<details><summary>Abstract**</summary>

Diffusion models have been shown to be capable of generating high-quality images, suggesting that they could contain meaningful internal representations. Unfortunately, the feature maps that encode a diffusion model's internal information are spread not only over layers of the network, but also over diffusion timesteps, making it challenging to extract useful descriptors. We propose Diffusion Hyperfeatures, a framework for consolidating multi-scale and multi-timestep feature maps into per-pixel feature descriptors that can be used for downstream tasks. These descriptors can be extracted for both synthetic and real images using the generation and inversion processes. We evaluate the utility of our Diffusion Hyperfeatures on the task of semantic keypoint correspondence: our method achieves superior performance on the SPair-71k real image benchmark. We also demonstrate that our method is flexible and transferable: our feature aggregation network trained on the inversion features of real image pairs can be used on the generation features of synthetic image pairs with unseen objects and compositions. Our code is available at https://diffusion-hyperfeatures.github.io.</details>

### Diffusion Self-Guidance for Controllable Image Generation

**Authors:** Dave Epstein, Allan Jabri, Ben Poole, Alexei Efros, Aleksander Holynski

**<details><summary>Abstract**</summary>

Large-scale generative models are capable of producing high-quality images from detailed prompts. However, many aspects of an image are difficult or impossible to convey through text. We introduce self-guidance, a method that provides precise control over properties of the generated image by guiding the internal representations of diffusion models. We demonstrate that the size, location, and appearance of objects can be extracted from these representations, and show how to use them to steer the sampling process. Self-guidance operates similarly to standard classifier guidance, but uses signals present in the pretrained model itself, requiring no additional models or training. We demonstrate the flexibility and effectiveness of self-guided generation through a wide range of challenging image manipulations, such as modifying the position or size of a single object (keeping the rest of the image unchanged), merging the appearance of objects in one image with the layout of another, composing objects from multiple images into one, and more. We also propose a new method for reconstruction using self-guidance, which allows extending our approach to editing real images.</details>

### Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning

**Authors:** Seungyong Moon, Junyoung Yeom, Bumsoo Park, Hyun Oh Song

**<details><summary>Abstract**</summary>

Discovering achievements with a hierarchical structure in procedurally generated environments presents a significant challenge.This requires an agent to possess a broad range of abilities, including generalization and long-term reasoning. Many prior methods have been built upon model-based or hierarchical approaches, with the belief that an explicit module for long-term planning would be advantageous for learning hierarchical dependencies. However, these methods demand an excessive number of environment interactions or large model sizes, limiting their practicality. In this work, we demonstrate that proximal policy optimization (PPO), a simple yet versatile model-free algorithm, outperforms previous methods when optimized with recent implementation practices. Moreover, we find that the PPO agent can predict the next achievement to be unlocked to some extent, albeit with limited confidence. Based on this observation, we introduce a novel contrastive learning method, called achievement distillation, which strengthens the agent's ability to predict the next achievement. Our method exhibits a strong capacity for discovering hierarchical achievements and shows state-of-the-art performance on the challenging Crafter environment in a sample-efficient manner while utilizing fewer model parameters.</details>

### Discovering Intrinsic Spatial-Temporal Logic Rules to Explain Human Actions

**Authors:** Chengzhi Cao, Chao Yang, Ruimao Zhang, Shuang Li

**<details><summary>Abstract**</summary>

We propose an interpretable model to uncover the behavioral patterns of human movements by analyzing their trajectories. Our approach is based on the belief that human actions are driven by intentions and are influenced by environmental factors such as spatial relationships with surrounding objects. To model this, we use a set of spatial-temporal logic rules that include intention variables as principles. These rules are automatically discovered and used to capture the dynamics of human actions. To learn the model parameters and rule content, we design an EM learning algorithm that treats the unknown rule content as a latent variable. In the E-step, we evaluate the posterior over the latent rule content, and in the M-step, we optimize the rule generator and model parameters by maximizing the expected log-likelihood. Our model has wide-ranging applications in areas such as sports analytics, robotics, and autonomous cars. We demonstrate the model's superior interpretability and prediction performance on both pedestrian and NBA basketball player datasets, achieving promising results.</details>

### Distributed Inference and Fine-tuning of Large Language Models Over The Internet

**Authors:** Alexander Borzunov, Max Ryabinin, Artem Chumachenko, Dmitry Baranchuk, Tim Dettmers, Younes Belkada, Pavel Samygin, Colin Raffel

**<details><summary>Abstract**</summary>

Large language models (LLMs) are useful in many NLP tasks and become more capable with size, with the best open-source models having over 50 billion parameters. However, using these 50B+ models requires high-end hardware, making them inaccessible to most researchers. In this work, we investigate methods for cost-efficient inference and fine-tuning of LLMs, comparing local and distributed strategies. We observe that a large enough model (50B+) can run efficiently even on geodistributed devices in a consumer-grade network. This could allow running LLM efficiently by pooling together idle compute resources of multiple research groups and volunteers. We address two open problems: (1) how to perform inference and fine-tuning reliably if any device can disconnect abruptly and (2) how to partition LLMs between devices with uneven hardware, joining and leaving at will. In order to do that, we develop special fault-tolerant inference algorithms and load-balancing protocols that automatically assign devices to maximize the total system throughput. We showcase these algorithms in Petals — a decentralized system that runs Llama 2 (70B) and BLOOM (176B) over the Internet up to $10\times$ faster than offloading for interactive generation. We evaluate the performance of our system in simulated conditions and a real-world setup spanning two continents.</details>

### Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning

**Authors:** Tyler Kastner, Murat Erdogdu, Amir-massoud Farahmand

**<details><summary>Abstract**</summary>

We consider the problem of learning models for risk-sensitive reinforcement learning. We theoretically demonstrate that proper value equivalence, a method of learning models which can be used to plan optimally in the risk-neutral setting, is not sufficient to plan optimally in the risk-sensitive setting. We leverage distributional reinforcement learning to introduce two new notions of model equivalence, one which is general and can be used to plan for any risk measure, but is intractable; and a practical variation which allows one to choose which risk measures they may plan optimally for. We demonstrate how our models can be used to augment any model-free risk-sensitive algorithm, and provide both tabular and large-scale experiments to demonstrate our method’s ability.</details>

### Distributionally Robust Bayesian Optimization with $\varphi$-divergences

**Authors:** Hisham Husain, Vu Nguyen, Anton van den Hengel

**<details><summary>Abstract**</summary>

The study of robustness has received much attention due to its inevitability in data-driven settings where many systems face uncertainty. One such example of concern is Bayesian Optimization (BO), where uncertainty is multi-faceted, yet there only exists a limited number of works dedicated to this direction. In particular, there is the work of Kirschner et al., which bridges the existing literature of Distributionally Robust Optimization (DRO) by casting the BO problem from the lens of DRO. While this work is pioneering, it admittedly suffers from various practical shortcomings such as finite contexts assumptions, leaving behind the main question \textit{Can one devise a computationally tractable algorithm for solving this DRO-BO problem}? In this work, we tackle this question to a large degree of generality by considering robustness against data-shift in $\varphi$-divergences, which subsumes many popular choices, such as the $\chi^2$-divergence, Total Variation, and the extant Kullback-Leibler (KL) divergence. We show that the DRO-BO problem in this setting is equivalent to a finite-dimensional optimization problem which, even in the continuous context setting, can be easily implemented with provable sublinear regret bounds. We then show experimentally that our method surpasses existing methods, attesting to the theoretical results.</details>

### Distributionally Robust Ensemble of Lottery Tickets Towards Calibrated  Sparse Network Training

**Authors:** Hitesh Sapkota, Dingrong Wang, Zhiqiang Tao, Qi Yu

**<details><summary>Abstract**</summary>

The recently developed sparse network training methods, such as Lottery Ticket Hypothesis (LTH) and its variants, have shown impressive learning capacity by finding sparse sub-networks from a dense one. While these methods could largely sparsify deep networks, they generally focus more on realizing comparable accuracy to dense counterparts yet neglect network calibration. However, how to achieve calibrated network predictions lies at the core of improving model reliability, especially when it comes to addressing the overconfident issue and out-of-distribution cases. In this study, we propose a novel Distributionally Robust Optimization (DRO) framework to achieve an ensemble of lottery tickets towards calibrated network sparsification. Specifically, the proposed DRO ensemble aims to learn multiple diverse and complementary sparse sub-networks (tickets) with the guidance of uncertainty sets, which encourage tickets to gradually capture different data distributions from easy to hard and naturally complement each other. We theoretically justify the strong calibration performance by showing how the proposed robust training process guarantees to lower the confidence of incorrect predictions. Extensive experimental results on several benchmarks show that our proposed lottery ticket ensemble leads to a clear calibration improvement without sacrificing accuracy and burdening inference costs. Furthermore, experiments on OOD datasets demonstrate the robustness of our approach in the open-set environment.</details>

### Diversified Outlier Exposure for Out-of-Distribution Detection via Informative Extrapolation

**Authors:** Jianing Zhu, Yu Geng, Jiangchao Yao, Tongliang Liu, Gang Niu, Masashi Sugiyama, Bo Han

**<details><summary>Abstract**</summary>

Out-of-distribution (OOD) detection is important for deploying reliable machine learning models on real-world applications. Recent advances in outlier exposure have shown promising results on OOD detection via fine-tuning model with informatively sampled auxiliary outliers. However, previous methods assume that the collected outliers can be sufficiently large and representative to cover the boundary between ID and OOD data, which might be impractical and challenging. In this work, we propose a novel framework, namely, Diversified Outlier Exposure (DivOE), for effective OOD detection via informative extrapolation based on the given auxiliary outliers. Specifically, DivOE introduces a new learning objective, which diversifies the auxiliary distribution by explicitly synthesizing more informative outliers for extrapolation during training. It leverages a multi-step optimization method to generate novel outliers beyond the original ones, which is compatible with many variants of outlier exposure. Extensive experiments and analyses have been conducted to characterize and demonstrate the effectiveness of the proposed DivOE. The code is publicly available at: https://github.com/tmlr-group/DivOE.</details>

### Diversifying Spatial-Temporal Perception for Video Domain Generalization

**Authors:** Kun-Yu Lin, Jia-Run Du, Yipeng Gao, Jiaming Zhou, Wei-Shi Zheng

**<details><summary>Abstract**</summary>

Video domain generalization aims to learn generalizable video classification models for unseen target domains by training in a source domain.A critical challenge of video domain generalization is to defend against the heavy reliance on domain-specific cues extracted from the source domain when recognizing target videos. To this end, we propose to perceive diverse spatial-temporal cues in videos, aiming to discover potential domain-invariant cues in addition to domain-specific cues. We contribute a novel model named Spatial-Temporal Diversification Network (STDN), which improves the diversity from both space and time dimensions of video data. First, our STDN proposes to discover various types of spatial cues within individual frames by spatial grouping. Then, our STDN proposes to explicitly model spatial-temporal dependencies between video contents at multiple space-time scales by spatial-temporal relation modeling. Extensive experiments on three benchmarks of different types demonstrate the effectiveness and versatility of our approach.</details>

### [Spotlight] DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining

**Authors:** Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V Le, Tengyu Ma, Adams Wei Yu

**<details><summary>Abstract**</summary>

The mixture proportions of pretraining data domains (e.g., Wikipedia, books, web text) greatly affect language model (LM) performance. In this paper, we propose Domain Reweighting with Minimax Optimization (DoReMi), which first trains a small proxy model using group distributionally robust optimization (Group DRO) over domains to produce domain weights (mixture proportions) without knowledge of downstream tasks. We then resample a dataset with these domain weights and train a larger, full-sized model. In our experiments, we use DoReMi on a 280M-parameter proxy model to find domain weights for training an 8B-parameter model (30x larger) more efficiently. On The Pile, DoReMi improves perplexity across all domains, even when it downweights a domain. DoReMi improves average few-shot downstream accuracy by 6.5% points over a baseline model trained using The Pile's default domain weights and reaches the baseline accuracy with 2.6x fewer training steps. On the GLaM dataset, DoReMi, which has no knowledge of downstream tasks, even matches the performance of using domain weights tuned on downstream tasks.</details>

### Does Graph Distillation See Like Vision Dataset Counterpart?

**Authors:** Beining Yang, Kai Wang, Qingyun Sun, Cheng Ji, Xingcheng Fu, Hao Tang, Yang You, Jianxin Li

**<details><summary>Abstract**</summary>

Training on large-scale graphs has achieved remarkable results in graph representation learning, but its cost and storage have attracted increasing concerns. Existing graph condensation methods primarily focus on optimizing the feature matrices of condensed graphs while overlooking the impact of the structure information from the original graphs. To investigate the impact of the structure information, we conduct analysis from the spectral domain and empirically identify substantial Laplacian Energy Distribution (LED) shifts in previous works. Such shifts lead to poor performance in cross-architecture generalization and specific tasks, including anomaly detection and link prediction. In this paper, we propose a novel Structure-broadcasting Graph Dataset Distillation (\textbf{SGDD}) scheme for broadcasting the original structure information to the generation of the synthetic one, which explicitly prevents overlooking the original structure information. Theoretically, the synthetic graphs by SGDD are expected to have smaller LED shifts than previous works, leading to superior performance in both cross-architecture settings and specific tasks.We validate the proposed SGDD~across 9 datasets and achieve state-of-the-art results on all of them: for example, on YelpChi dataset, our approach maintains 98.6\% test accuracy of training on the original graph dataset with 1,000 times saving on the scale of the graph. Moreover, we empirically evaluate there exist 17.6\% $\sim$ 31.4\% reductions in LED shift crossing 9 datasets. Extensive experiments and analysis verify the effectiveness and necessity of the proposed designs. The code will be made public.</details>

### Does a sparse ReLU network training problem always admit an optimum ?

**Authors:** QUOC-TUNG LE, Remi Gribonval, Elisa Riccietti

**<details><summary>Abstract**</summary>

Given a training set, a loss function, and a neural network architecture, it is often taken for granted that optimal network parameters exist, and a common practice is to apply available optimization algorithms to search for them. In this work, we show that the existence of an optimal solution is not always guaranteed, especially in the context of sparse ReLU neural networks.In particular, we first show that optimization problems involving deep networks with certain sparsity patterns do not always have optimal parameters, and that optimization algorithms may then diverge. Via a new topological relation between sparse ReLU neural networks and their linear counterparts, we derive --using existing tools from real algebraic geometry-- an algorithm to verify that a given sparsity pattern suffers from this issue. Then, the existence of a global optimum is proved for every concrete optimization problem involving a shallow sparse ReLU neural network of output dimension one. Overall, the analysis is based on the investigation of two topological properties of the space of functions implementable as sparse ReLU neural networks: a best approximation property, and a closedness property, both in the uniform norm. This is studied both for (finite) domains corresponding to practical training on finite training sets, and for more general domains such as the unit cube. This allows us to provide conditions for the guaranteed existence of an optimum given a sparsity pattern. The results apply not only to several sparsity patterns proposed in recent works on network pruning/sparsification, but also to classical dense neural networks, including architectures not covered by existing results.</details>

### [Spotlight] Double Gumbel Q-Learning

**Authors:** David Yu-Tung Hui, Aaron Courville, Pierre-Luc Bacon

**<details><summary>Abstract**</summary>

We show that Deep Neural Networks introduce two heteroscedastic Gumbel noise sources into Q-Learning.  To account for these noise sources, we propose Double Gumbel Q-Learning, a Deep Q-Learning algorithm applicable for both discrete and continuous control.  In discrete control, we derive a closed-form expression for the loss function of our algorithm.  In continuous control, this loss function is intractable and we therefore derive an approximation with a hyperparameter whose value regulates pessimism in Q-Learning.  We present a default value for our pessimism hyperparameter that enables DoubleGum to outperform DDPG, TD3, SAC, XQL, quantile regression, and Mixture-of-Gaussian Critics in aggregate over 33 tasks from DeepMind Control, MuJoCo, MetaWorld, and Box2D and show that tuning this hyperparameter may further improve sample efficiency.</details>

### Double Pessimism is Provably Efficient for Distributionally Robust Offline Reinforcement Learning: Generic Algorithm and Robust Partial Coverage

**Authors:** Jose Blanchet, Miao Lu, Tong Zhang, Han Zhong

**<details><summary>Abstract**</summary>

We study distributionally robust offline reinforcement learning (RL), which seeks to find an optimal robust policy purely from an offline dataset that can perform well in perturbed environments. We propose a generic algorithm framework Doubly Pessimistic Model-based Policy Optimization ($\texttt{P}^2\texttt{MPO}$) for robust offline RL, which features a novel combination of a flexible model estimation subroutine and a doubly pessimistic policy optimization step. Here the double pessimism principle is crucial to overcome the distribution shift incurred by i) the mismatch between behavior policy and the family of target policies; and ii) the perturbation of the nominal model. Under certain accuracy assumptions on the model estimation subroutine, we show that $\texttt{P}^2\texttt{MPO}$ is provably sample-efficient with robust partial coverage data, which means that the offline dataset has good coverage of the distributions induced by the optimal robust policy and perturbed models around the nominal model. By tailoring specific model estimation subroutines for concrete examples including tabular Robust Markov Decision Process (RMDP), factored RMDP, and RMDP with kernel and neural function approximations, we show that $\texttt{P}^2\texttt{MPO}$ enjoys a $\tilde{\mathcal{O}}(n^{-1/2})$ convergence rate, where $n$ is the number of trajectories in the offline dataset. Notably, these models, except for the tabular case, are first identified and proven tractable by this paper. To the best of our knowledge, we first propose a general learning principle --- double pessimism --- for robust offline RL and show that it is provably efficient in the context of general function approximations.</details>

### Double and Single Descent in Causal Inference with an Application to High-Dimensional Synthetic Control

**Authors:** Jann Spiess, guido imbens, Amar Venugopal

**<details><summary>Abstract**</summary>

Motivated by a recent literature on the double-descent phenomenon in machine learning, we consider highly over-parameterized models in causal inference, including synthetic control with many control units. In such models, there may be so many free parameters that the model fits the training data perfectly. We first investigate high-dimensional linear regression for imputing wage data and estimating average treatment effects, where we find that models with many more covariates than sample size can outperform simple ones. We then document the performance of high-dimensional synthetic control estimators with many control units. We find that adding control units can help improve imputation performance even beyond the point where the pre-treatment fit is perfect. We provide a unified theoretical perspective on the performance of these high-dimensional models. Specifically, we show that more complex models can be interpreted as model-averaging estimators over simpler ones, which we link to an improvement in average performance. This perspective yields concrete insights into the use of synthetic control when control units are many relative to the number of pre-treatment periods.</details>

### Drift doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection

**Authors:** Chengsen Wang, Zirui Zhuang, Qi Qi, Jingyu Wang, Xingyu Wang, Haifeng Sun, Jianxin Liao

**<details><summary>Abstract**</summary>

Many unsupervised methods have recently been proposed for multivariate time series anomaly detection. However, existing works mainly focus on stable data yet often omit the drift generated from non-stationary environments, which may lead to numerous false alarms. We propose **D**ynamic **D**ecomposition with **D**iffusion **R**econstruction (D$^3$R), a novel anomaly detection network for real-world unstable data to fill the gap. D$^3$R tackles the drift via decomposition and reconstruction. In the decomposition procedure, we utilize data-time mix-attention to dynamically decompose long-period multivariate time series, overcoming the limitation of the local sliding window. The information bottleneck is critical yet difficult to determine in the reconstruction procedure. To avoid retraining once the bottleneck changes, we control it externally by noise diffusion and directly reconstruct the polluted data. The whole model can be trained end-to-end. Extensive experiments on various real-world datasets demonstrate that D$^3$R significantly outperforms existing methods, with a 11% average relative improvement over the previous SOTA models.</details>

### DropPos: Pre-Training Vision Transformers by Reconstructing Dropped Positions

**Authors:** Haochen Wang, Junsong Fan, Yuxi Wang, Kaiyou Song, Tong Wang, ZHAO-XIANG ZHANG

**<details><summary>Abstract**</summary>

As it is empirically observed that Vision Transformers (ViTs) are quite insensitive to the order of input tokens, the need for an appropriate self-supervised pretext task that enhances the location awareness of ViTs is becoming evident. To address this, we present DropPos, a novel pretext task designed to reconstruct Dropped Positions. The formulation of DropPos is simple: we first drop a large random subset of positional embeddings and then the model classifies the actual position for each non-overlapping patch among all possible positions solely based on their visual appearance. To avoid trivial solutions, we increase the difficulty of this task by keeping only a subset of patches visible. Additionally, considering there may be different patches with similar visual appearances, we propose position smoothing and attentive reconstruction strategies to relax this classification problem, since it is not necessary to reconstruct their exact positions in these cases. Empirical evaluations of DropPos show strong capabilities. DropPos outperforms supervised pre-training and achieves competitive results compared with state-of-the-art self-supervised alternatives on a wide range of downstream benchmarks. This suggests that explicitly encouraging spatial reasoning abilities, as DropPos does, indeed contributes to the improved location awareness of ViTs. The code is publicly available at https://github.com/Haochen-Wang409/DropPos.</details>

### DynGFN: Towards Bayesian Inference of Gene Regulatory Networks with GFlowNets

**Authors:** Lazar Atanackovic, Alexander Tong, Bo Wang, Leo J Lee, Yoshua Bengio, Jason Hartford

**<details><summary>Abstract**</summary>

One of the grand challenges of cell biology is inferring the gene regulatory network (GRN) which describes interactions between genes and their products that control gene expression and cellular function. We can treat this as a causal discovery problem but with two non-standard challenges: (1) regulatory networks are inherently cyclic so we should not model a GRN as a directed acyclic graph (DAG), and (2) observations have significant measurement noise so for typical sample sizes, there will always be a large equivalence class of graphs that are likely given the data, and we want methods that capture this uncertainty. Existing methods either focus on challenge (1), identifying cyclic structure from dynamics, or on challenge (2) learning complex Bayesian posteriors over directed acyclic graphs, but not both. In this paper we leverage the fact that it is possible to estimate the ``velocity'' of the expression of a gene with RNA velocity techniques to develop an approach that addresses both challenges. Because we have access to velocity information, we can treat the Bayesian structure learning problem as a problem of sparse identification of a dynamical system, capturing cyclic feedback loops through time. We leverage Generative Flow Networks (GFlowNets) to estimate the posterior distribution over the combinatorial space of possible sparse dependencies. Our results indicate that our method learns posteriors that better encapsulate the distributions of cyclic structures compared to counterpart state-of-the-art Bayesian structure learning approaches.</details>

### Dynamic Personalized Federated Learning with Adaptive Differential Privacy

**Authors:** Xiyuan Yang, Wenke Huang, Mang Ye

**<details><summary>Abstract**</summary>

Personalized federated learning with differential privacy has been considered a feasible solution to address non-IID distribution of data and privacy leakage risks. However, current personalized federated learning methods suffer from inflexible personalization and convergence difficulties due to two main factors: 1) Firstly, we observe that the prevailing personalization methods mainly achieve this by personalizing a fixed portion of the model, which lacks flexibility. 2) Moreover, we further demonstrate that the default gradient calculation is sensitive to the widely-used clipping operations in differential privacy, resulting in difficulties in convergence. Considering that Fisher information values can serve as an effective measure for estimating the information content of parameters by reflecting the model sensitivity to parameters, we aim to leverage this property to address the aforementioned challenges. In this paper, we propose a novel federated learning method with Dynamic Fisher Personalization and Adaptive Constraint (FedDPA) to handle these challenges. Firstly, by using layer-wise Fisher information to measure the information content of local parameters, we retain local parameters with high Fisher values during the personalization process, which are considered informative, simultaneously prevent these parameters from noise perturbation. Secondly, we introduce an adaptive approach by applying differential constraint strategies to personalized parameters and shared parameters identified in the previous for better convergence.  Our method boosts performance through flexible personalization while mitigating the slow convergence caused by clipping operations. Experimental results on CIFAR-10, FEMNIST and SVHN dataset demonstrate the effectiveness of our approach in achieving better performance and robustness against clipping, under personalized federated learning with differential privacy.</details>

### Dynamo-Depth: Fixing Unsupervised Depth Estimation for Dynamical Scenes

**Authors:** Yihong Sun, Bharath Hariharan

**<details><summary>Abstract**</summary>

Unsupervised monocular depth estimation techniques have demonstrated encouraging results but typically assume that the scene is static. These techniques suffer when trained on dynamical scenes, where apparent object motion can equally be explained by hypothesizing the object's independent motion, or by altering its depth. This ambiguity causes depth estimators to predict erroneous depth for moving objects. To resolve this issue, we introduce Dynamo-Depth, an unifying approach that disambiguates dynamical motion by jointly learning monocular depth, 3D independent flow field, and motion segmentation from unlabeled monocular videos. Specifically, we offer our key insight that a good initial estimation of motion segmentation is sufficient for jointly learning depth and independent motion despite the fundamental underlying ambiguity. Our proposed method achieves state-of-the-art performance on monocular depth estimation on Waymo Open and nuScenes Dataset with significant improvement in the depth of moving objects. Code and additional results are available at https://dynamo-depth.github.io.</details>

### EDGI: Equivariant Diffusion for Planning with Embodied Agents

**Authors:** Johann Brehmer, Joey Bose, Pim de Haan, Taco Cohen

**<details><summary>Abstract**</summary>

Embodied agents operate in a structured world, often solving tasks with spatial, temporal, and permutation symmetries. Most algorithms for planning and model-based reinforcement learning (MBRL) do not take this rich geometric structure into account, leading to sample inefficiency and poor generalization. We introduce the Equivariant Diffuser for Generating Interactions (EDGI), an algorithm for MBRL and planning that is equivariant with respect to the product of the spatial symmetry group SE(3), the discrete-time translation group ℤ, and the object permutation group Sₙ. EDGI follows the Diffuser framework by Janner et al. (2022) in treating both learning a world model and planning in it as a conditional generative modeling problem, training a diffusion model on an offline trajectory dataset. We introduce a new SE(3) × ℤ × Sₙ-equivariant diffusion model that supports multiple representations. We integrate this model in a planning loop, where conditioning and classifier guidance let us softly break the symmetry for specific tasks as needed. On object manipulation and navigation tasks, EDGI is substantially more sample efficient and generalizes better across the symmetry group than non-equivariant models.</details>

### EMMA-X: An EM-like Multilingual Pre-training Algorithm for Cross-lingual Representation Learning

**Authors:** Ping Guo, Xiangpeng Wei, Yue Hu, Baosong Yang, Dayiheng Liu, Fei Huang, jun xie

**<details><summary>Abstract**</summary>

Expressing universal semantics common to all languages is helpful to understand the meanings of complex and culture-specific sentences. The research theme underlying this scenario focuses on learning universal representations across languages with the usage of massive parallel corpora. However, due to the sparsity and scarcity of parallel data, there is still a big challenge in learning authentic ``universals'' for any two languages. In this paper, we propose Emma-X: an EM-like Multilingual pre-training Algorithm, to learn Cross-lingual universals with the aid of excessive multilingual non-parallel data. Emma-X unifies the cross-lingual representation learning task and an extra semantic relation prediction task within an EM framework. Both the extra semantic classifier and the cross-lingual sentence encoder approximate the semantic relation of two sentences, and supervise each other until convergence. To evaluate Emma-X, we conduct experiments on xrete, a newly introduced benchmark containing 12 widely studied cross-lingual tasks that fully depend on sentence-level representations. Results reveal that Emma-X achieves state-of-the-art performance. Further geometric analysis of the built representation space with three requirements demonstrates the superiority of Emma-X over advanced models.</details>

### Effective Bayesian Heteroscedastic Regression with Deep Neural Networks

**Authors:** Alexander Immer, Emanuele Palumbo, Alexander Marx, Julia Vogt

**<details><summary>Abstract**</summary>

Flexibly quantifying both irreducible aleatoric and model-dependent epistemic uncertainties plays an important role for complex regression problems. While deep neural networks in principle can provide this flexibility and learn heteroscedastic aleatoric uncertainties through non-linear functions, recent works highlight that maximizing the log likelihood objective parameterized by mean and variance can lead to compromised mean fits since the gradient are scaled by the predictive variance, and propose adjustments in line with this premise. We instead propose to use the natural parametrization of the Gaussian, which has been shown to be more stable for heteroscedastic regression based on non-linear feature maps and Gaussian processes. Further, we emphasize the significance of principled regularization of the network parameters and prediction. We therefore propose an efficient Laplace approximation for heteroscedastic neural networks that allows automatic regularization through empirical Bayes and provides epistemic uncertainties, both of which improve generalization.We showcase on a range of regression problems—including a new heteroscedastic image regression benchmark—that our methods are scalable, improve over previous approaches for heteroscedastic regression, and provide epistemic uncertainty without requiring hyperparameter tuning.</details>

### [Spotlight] Effective Human-AI Teams via Learned Natural Language Rules and Onboarding

**Authors:** Hussein Mozannar, Jimin Lee, Dennis Wei, Prasanna Sattigeri, Subhro Das, David Sontag

**<details><summary>Abstract**</summary>

People are relying on AI agents to assist them with various tasks. The human must know when to rely on the agent, collaborate with the agent, or ignore its suggestions. In this work, we propose to learn rules grounded in data regions and described in natural language that illustrate how the human should collaborate with the AI. Our novel region discovery algorithm finds local regions in the data as neighborhoods in an embedding space that corrects the human prior. Each region is then described using an iterative and contrastive procedure where a large language model describes the region. We then teach these rules to the human via an onboarding stage. Through user studies on object detection and question-answering tasks, we show that our method can lead to more accurate human-AI teams. We also evaluate our region discovery and description algorithms separately.</details>

### Efficient Activation Function Optimization through Surrogate Modeling

**Authors:** Garrett Bingham, Risto Miikkulainen

**<details><summary>Abstract**</summary>

Carefully designed activation functions can improve the performance of neural networks in many machine learning tasks.  However, it is difficult for humans to construct optimal activation functions, and current activation function search algorithms are prohibitively expensive.  This paper aims to improve the state of the art through three steps: First, the benchmark datasets Act-Bench-CNN, Act-Bench-ResNet, and Act-Bench-ViT were created by training convolutional, residual, and vision transformer architectures from scratch with 2,913 systematically generated activation functions. Second, a characterization of the benchmark space was developed, leading to a new surrogate-based method for optimization. More specifically, the spectrum of the Fisher information matrix associated with the model's predictive distribution at initialization and the activation function's output distribution were found to be highly predictive of performance. Third, the surrogate was used to discover improved activation functions in several real-world tasks, with a surprising finding: a sigmoidal design that outperformed all other activation functions was discovered, challenging the status quo of always using rectifier nonlinearities in deep learning.  Each of these steps is a contribution in its own right; together they serve as a practical and theoretical foundation for further research on activation function optimization.</details>

### Efficient Adaptation of Large Vision Transformer via Adapter Re-Composing

**Authors:** Wei Dong, Dawei Yan, Zhijun Lin, Peng Wang

**<details><summary>Abstract**</summary>

The advent of high-capacity pre-trained models has revolutionized problem-solving in computer vision, shifting the focus from training task-specific models to adapting pre-trained models. Consequently, effectively adapting large pre-trained models to downstream tasks in an efficient manner has become a prominent research area. Existing solutions primarily concentrate on designing lightweight adapters and their interaction with pre-trained models, with the goal of minimizing the number of parameters requiring updates. In this study, we propose a novel Adapter Re-Composing (ARC) strategy that addresses efficient pre-trained model adaptation from a fresh perspective. Our approach considers the reusability of adaptation parameters and introduces a parameter-sharing scheme. Specifically, we leverage symmetric down-/up-projections to construct bottleneck operations, which are shared across layers. By learning low-dimensional re-scaling coefficients, we can effectively re-compose layer-adaptive adapters. This parameter-sharing strategy in adapter design allows us to further reduce the number of new parameters while maintaining satisfactory performance, thereby offering a promising approach to compress the adaptation cost. We conduct experiments on 24 downstream image classification tasks using various Vision Transformer variants to evaluate our method. The results demonstrate that our approach achieves compelling transfer learning performance with a reduced parameter count. Our code is available at https://github.com/DavidYanAnDe/ARC.</details>

### Efficient Algorithms for Generalized Linear Bandits with Heavy-tailed Rewards

**Authors:** Bo Xue, Yimu Wang, Yuanyu Wan, Jinfeng Yi, Lijun Zhang

**<details><summary>Abstract**</summary>

This paper investigates the problem of generalized linear bandits with heavy-tailed rewards, whose $(1+\epsilon)$-th moment is bounded for some $\epsilon\in (0,1]$. Although there exist methods for generalized linear bandits, most of them focus on bounded or sub-Gaussian rewards and are not well-suited for many real-world scenarios, such as financial markets and web-advertising. To address this issue, we propose two novel algorithms based on truncation and mean of medians. These algorithms achieve an almost optimal regret bound of $\widetilde{O}(dT^{\frac{1}{1+\epsilon}})$, where $d$ is the dimension of contextual information and $T$ is the time horizon. Our truncation-based algorithm supports online learning, distinguishing it from existing truncation-based approaches. Additionally, our mean-of-medians-based algorithm requires only $O(\log T)$ rewards and one estimator per epoch, making it more practical. Moreover, our algorithms improve the regret bounds by a logarithmic factor compared to existing algorithms when $\epsilon=1$. Numerical experimental results confirm the merits of our algorithms.</details>

### Efficient Equivariant Transfer Learning from Pretrained Models

**Authors:** Sourya Basu, Pulkit Katdare, Prasanna Sattigeri, Vijil Chenthamarakshan, Katherine Driggs-Campbell, Payel Das, Lav Varshney

**<details><summary>Abstract**</summary>

Efficient transfer learning algorithms are key to the success of foundation models on diverse downstream tasks even with limited data. Recent works of Basu et al. (2023) and Kaba et al. (2022) propose group averaging (equitune) and optimization-based methods, respectively, over features from group-transformed inputs to obtain equivariant outputs from non-equivariant neural networks. While Kaba et al. (2022) are only concerned with training from scratch, we find that equitune performs poorly on equivariant zero-shot tasks despite good finetuning results. We hypothesize that this is because pretrained models provide better quality features for certain transformations than others and simply averaging them is deleterious. Hence, we propose λ-equitune that averages the features using importance weights, λs. These weights are learned directly from the data using a small neural network, leading to excellent zero-shot and finetuned results that outperform equitune. Further, we prove that λ-equitune is equivariant and a universal approximator of equivariant functions. Additionally, we show that the method of Kaba et al. (2022) used with appropriate loss functions, which we call equizero, also gives excellent zero-shot and finetuned performance. Both equitune and equizero are special cases of λ- equitune. To show the simplicity and generality of our method, we validate on a wide range of diverse applications and models such as 1) image classification using CLIP, 2) deep Q-learning, 3) fairness in natural language generation (NLG), 4) compositional generalization in languages, and 5) image classification using pretrained CNNs such as Resnet and Alexnet.</details>

### Efficient Exploration in Continuous-time Model-based Reinforcement Learning

**Authors:** Lenart Treven, Jonas Hübotter, Bhavya, Florian Dorfler, Andreas Krause

**<details><summary>Abstract**</summary>

Reinforcement learning algorithms typically consider discrete-time dynamics, even though the underlying systems are often continuous in time. In this paper, we introduce a model-based reinforcement learning algorithm that represents continuous-time dynamics using nonlinear ordinary differential equations (ODEs). We capture epistemic uncertainty using well-calibrated probabilistic models, and use the optimistic principle for exploration. Our regret bounds surface the importance of the measurement selection strategy (MSS), since in continuous time we not only must decide how to explore, but also when to observe the underlying system. Our analysis demonstrates that the regret is sublinear when modeling ODEs with Gaussian Processes (GP) for common choices of MSS, such as equidistant sampling. Additionally, we propose an adaptive, data-dependent, practical MSS that, when combined with GP dynamics, also achieves sublinear regret with significantly fewer samples. We showcase the benefits of continuous-time modeling over its discrete-time counterpart, as well as our proposed adaptive MSS over standard baselines, on several applications.</details>

### Efficient RL with Impaired Observability: Learning to Act with Delayed and Missing State Observations

**Authors:** Minshuo Chen, Yu Bai, H. Vincent Poor, Mengdi Wang

**<details><summary>Abstract**</summary>

In real-world reinforcement learning (RL) systems, various forms of {\it impaired observability} can complicate matters. These situations arise when an agent is unable to observe the most recent state of the system due to latency or lossy channels, yet the agent must still make real-time decisions. This paper introduces a theoretical investigation into efficient RL in control systems where agents must act with delayed and missing state observations. We establish near-optimal regret bounds, of the form $\tilde{\mathcal{O}}(\sqrt{{\rm poly}(H) SAK})$, for RL in both the delayed and missing observation settings. Despite impaired observability posing significant challenges to the policy class and planning, our results demonstrate that learning remains efficient, with the regret bound optimally depending on the state-action size of the original system. Additionally, we provide a characterization of the performance of the optimal policy under impaired observability, comparing it to the optimal value obtained with full observability.</details>

### Efficiently incorporating quintuple interactions into geometric deep learning force fields

**Authors:** Zun Wang, Guoqing Liu, Yichi Zhou, Tong Wang, Bin Shao

**<details><summary>Abstract**</summary>

Machine learning force fields (MLFFs) have instigated a groundbreaking shift in molecular dynamics (MD) simulations across a wide range of fields, such as physics, chemistry, biology, and materials science. Incorporating higher order many-body interactions can enhance the expressiveness and accuracy of models. Recent models have achieved this by explicitly including up to four-body interactions. However, five-body interactions, which have relevance in various fields, are still challenging to incorporate efficiently into MLFFs. In this work, we propose the quintuple network (QuinNet), an end-to-end graph neural network that efficiently expresses many-body interactions up to five-body interactions with \emph{ab initio} accuracy. By analyzing the topology of diverse many-body interactions, we design the model architecture to efficiently and explicitly represent these interactions. We evaluate QuinNet on public datasets of small molecules, such as MD17 and its revised version, and show that it is compatible with other state-of-the-art models on these benchmarks. Moreover, QuinNet surpasses many leading models on larger and more complex molecular systems, such as MD22 and Chignolin, without increasing the computational complexity. We also use QuinNet as a force field for molecular dynamics (MD) simulations to demonstrate its accuracy and stability, and conduct an ablation study to elucidate the significance of five-body interactions. We open source our implementation at https://github.com/Zun-Wang/QuinNet.</details>

### Elastic Decision Transformer

**Authors:** Yueh-Hua Wu, Xiaolong Wang, Masashi Hamaya

**<details><summary>Abstract**</summary>

This paper introduces Elastic Decision Transformer (EDT), a significant advancement over the existing Decision Transformer (DT) and its variants. Although DT purports to generate an optimal trajectory, empirical evidence suggests it struggles with trajectory stitching, a process involving the generation of an optimal or near-optimal trajectory from the best parts of a set of sub-optimal trajectories. The proposed EDT differentiates itself by facilitating trajectory stitching during action inference at test time, achieved by adjusting the history length maintained in DT. Further, the EDT optimizes the trajectory by retaining a longer history when the previous trajectory is optimal and a shorter one when it is sub-optimal, enabling it to "stitch" with a more optimal trajectory. Extensive experimentation demonstrates EDT's ability to bridge the performance gap between DT-based and Q Learning-based approaches. In particular, the EDT outperforms Q Learning-based methods in a multi-task regime on the D4RL locomotion benchmark and Atari games.</details>

### Empowering Collaborative Filtering with Principled Adversarial Contrastive Loss

**Authors:** An Zhang, Leheng Sheng, Zhibo Cai, Xiang Wang, Tat-Seng Chua

**<details><summary>Abstract**</summary>

Contrastive Learning (CL) has achieved impressive performance in self-supervised learning tasks, showing superior generalization ability. Inspired by the success, adopting CL into collaborative filtering (CF) is prevailing in semi-supervised topK recommendations. The basic idea is to routinely conduct heuristic-based data augmentation and apply contrastive losses (e.g., InfoNCE) on the augmented views. Yet, some CF-tailored challenges make this adoption suboptimal, such as the issue of out-of-distribution, the risk of false negatives, and the nature of top-K evaluation. They necessitate the CL-based CF scheme to focus more on mining hard negatives and distinguishing false negatives from the vast unlabeled user-item interactions, for informative contrast signals. Worse still, there is limited understanding of contrastive loss in CF methods, especially w.r.t. its generalization ability. To bridge the gap, we delve into the reasons underpinning the success of contrastive loss in CF, and propose a principled Adversarial InfoNCE loss (AdvInfoNCE), which is a variant of InfoNCE, specially tailored for CF methods. AdvInfoNCE adaptively explores and assigns hardness to each negative instance in an adversarial fashion and further utilizes a fine-grained hardness-aware ranking criterion to empower the recommender’s generalization ability. Training CF models with AdvInfoNCE, we validate the effectiveness of AdvInfoNCE on both synthetic and real-world benchmark datasets, thus showing its generalization ability to mitigate out-of-distribution problems. Given the theoretical guarantees and empirical superiority of AdvInfoNCE over most contrastive loss functions, we advocate its adoption as a standard loss in recommender systems, particularly for the out-of-distribution tasks. Codes are available at https://github.com/LehengTHU/AdvInfoNCE.</details>

### Empowering Convolutional Neural Nets with MetaSin Activation

**Authors:** Farnood Salehi, Tunç Aydin, André Gaillard, Guglielmo Camporese, Yuxuan Wang

**<details><summary>Abstract**</summary>

ReLU networks have remained the default choice for models in the area of image prediction despite their well-established spectral bias towards learning low frequencies faster, and consequently their difficulty of reproducing high frequency visual details. As an alternative, sin networks showed promising results in learning implicit representations of visual data. However training these networks in practically relevant settings proved to be difficult, requiring careful initialization, dealing with issues due to inconsistent gradients, and a degeneracy in local minima. In this work, we instead propose replacing a baseline network’s existing activations with a novel ensemble function with trainable parameters. The proposed MetaSin activation can be trained reliably without requiring intricate initialization schemes, and results in consistently lower test loss compared to alternatives. We demonstrate our method in the areas of Monte-Carlo denoising and image resampling where we set new state-of-the-art through a knowledge distillation based training procedure. We present ablations on hyper-parameter settings, comparisons with alternative activation function formulations, and discuss the use of our method in other domains, such as image classification.</details>

### Encoding Human Behavior in Information Design through Deep Learning

**Authors:** Guanghui Yu, Wei Tang, Saumik Narayanan, Chien-Ju Ho

**<details><summary>Abstract**</summary>

We initiate the study of $\textit{behavioral information design}$ through deep learning. In information design, a $\textit{sender}$ aims to persuade a $\textit{receiver}$ to take certain actions by strategically revealing information. We address scenarios in which the receiver might exhibit different behavior patterns other than the standard Bayesian rational assumption. We propose HAIDNet, a neural-network-based optimization framework for information design that can adapt to multiple representations of human behavior. Through extensive simulation, we show that HAIDNet can not only recover information policies that are near-optimal compared with known analytical solutions, but also can extend to designing information policies for settings that are computationally challenging (e.g., when there are multiple receivers) or for settings where there are no known solutions in general (e.g., when the receiver behavior does not follow the Bayesian rational assumption). We also conduct real-world human-subject experiments and demonstrate that our framework can capture human behavior from data and lead to more effective information policy for real-world human receivers.</details>

### Enhancing Adaptive History Reserving by Spiking Convolutional Block Attention Module in Recurrent Neural Networks

**Authors:** Qi Xu, Yuyuan Gao, Jiangrong Shen, Yaxin Li, Xuming Ran, Huajin Tang, Gang Pan

**<details><summary>Abstract**</summary>

Spiking neural networks (SNNs) serve as one type of efficient model to process spatio-temporal patterns in time series, such as the Address-Event Representation data collected from Dynamic Vision Sensor (DVS). Although convolutional SNNs have achieved remarkable performance on these AER datasets, benefiting from the predominant spatial feature extraction ability of convolutional structure, they ignore temporal features related to sequential time points. In this paper, we develop a recurrent spiking neural network (RSNN) model embedded with an advanced spiking convolutional block attention module (SCBAM) component to combine both spatial and temporal features of spatio-temporal patterns. It invokes the history information in spatial and temporal channels adaptively through SCBAM, which brings the advantages of efficient memory calling and history redundancy elimination. The performance of our model was evaluated in DVS128-Gesture dataset and other time-series datasets. The experimental results show that the proposed SRNN-SCBAM model makes better use of the history information in spatial and temporal dimensions with less memory space, and achieves higher accuracy compared to other models.</details>

### Enhancing Motion Deblurring in High-Speed Scenes with Spike Streams

**Authors:** Shiyan Chen, Jiyuan Zhang, Yajing Zheng, Tiejun Huang, Zhaofei Yu

**<details><summary>Abstract**</summary>

Traditional cameras produce desirable vision results but struggle with motion blur in high-speed scenes due to long exposure windows. Existing frame-based deblurring algorithms face challenges in extracting useful motion cues from severely blurred images. Recently, an emerging bio-inspired vision sensor known as the spike camera has achieved an extremely high frame rate while preserving rich spatial details, owing to its novel sampling mechanism. However, typical binary spike streams are relatively low-resolution, degraded image signals devoid of color information, making them unfriendly to human vision. In this paper, we propose a novel approach that integrates the two modalities from two branches, leveraging spike streams as auxiliary visual cues for guiding deblurring in high-speed motion scenes. We propose the first spike-based motion deblurring model with bidirectional information complementarity. We introduce a content-aware motion magnitude attention module that utilizes learnable mask to extract relevant information from blurry images effectively, and we incorporate a transposed cross-attention fusion module to efficiently combine features from both spike data and blurry RGB images.Furthermore, we build two extensive synthesized datasets for training and validation purposes, encompassing high-temporal-resolution spikes, blurry images, and corresponding sharp images. The experimental results demonstrate that our method effectively recovers clear RGB images from highly blurry scenes and outperforms state-of-the-art deblurring algorithms in multiple settings.</details>

### Enhancing Sharpness-Aware Optimization Through Variance Suppression

**Authors:** Bingcong Li, Georgios Giannakis

**<details><summary>Abstract**</summary>

Sharpness-aware minimization (SAM) has well documented merits in enhancing generalization of deep neural networks, even without sizable data augmentation. Embracing the geometry of the loss function, where neighborhoods of 'flat minima' heighten generalization ability, SAM seeks 'flat valleys' by minimizing the maximum loss caused by an *adversary* perturbing parameters within the neighborhood.Although critical to account for sharpness of the loss function, such an '*over-friendly* adversary' can curtail the outmost level of generalization. The novel approach of this contribution fosters stabilization of adversaries through *variance suppression* (VaSSO) to avoid such friendliness. VaSSO's *provable* stability safeguards its numerical improvement over SAM in model-agnostic tasks, including image classification and machine translation. In addition, experiments confirm that VaSSO endows SAM with robustness against high levels of label noise.</details>

### [Spotlight] Epistemic Neural Networks

**Authors:** Ian Osband, Zheng Wen, Seyed Mohammad Asghari, Vikranth Dwaracherla, MORTEZA IBRAHIMI, Xiuyuan Lu, Benjamin Van Roy

**<details><summary>Abstract**</summary>

Intelligence relies on an agent's knowledge of what it does not know.This capability can be assessed based on the quality of joint predictions of labels across multiple inputs.In principle, ensemble-based approaches can produce effective joint predictions, but the computational costs of large ensembles become prohibitive.We introduce the epinet: an architecture that can supplement any conventional neural network, including large pretrained models, and can be trained with modest incremental computation to estimate uncertainty.With an epinet, conventional neural networks outperform very large ensembles, consisting of hundreds or more particles, with orders of magnitude less computation.The epinet does not fit the traditional framework of Bayesian neural networks.To accommodate development of approaches beyond BNNs, such as the epinet, we introduce the epistemic neural network (ENN) as a general interface for models that produce joint predictions.</details>

### Equal Opportunity of Coverage in Fair Regression

**Authors:** Fangxin Wang, Lu Cheng, Ruocheng Guo, Kay Liu, Philip S Yu

**<details><summary>Abstract**</summary>

We study fair machine learning (ML) under predictive uncertainty to enable reliable and trustworthy decision-making. The seminal work of 'equalized coverage' proposed an uncertainty-aware fairness notion. However, it does not guarantee equal coverage rates across more fine-grained groups (e.g., low-income females) conditioning on the true label and is biased in the assessment of uncertainty. To tackle these limitations, we propose a new uncertainty-aware fairness -- Equal Opportunity of Coverage (EOC) -- that aims to achieve two properties: (1) coverage rates for different groups with similar outcomes are close, and (2) the coverage rate for the entire population remains at a predetermined level. Further, the prediction intervals should be narrow to be informative. We propose Binned Fair Quantile Regression (BFQR), a distribution-free post-processing method to improve EOC with reasonable width for any trained ML models. It first calibrates a hold-out set to bound deviation from EOC, then leverages conformal prediction to maintain EOC on a test set, meanwhile optimizing prediction interval width. Experimental results demonstrate the effectiveness of our method in improving EOC.</details>

### Equivariant Single View Pose Prediction Via Induced and Restriction Representations

**Authors:** Owen Howell, David Klee, Ondrej Biza, Linfeng Zhao, Robin Walters

**<details><summary>Abstract**</summary>

Learning about the three-dimensional world from two-dimensional images is a fundamental problem in computer vision. An ideal neural network architecture for such tasks would leverage the fact that objects can be rotated and translated in three dimensions to make predictions about novel images. However, imposing $SO(3)$-equivariance on two-dimensional inputs is difficult because the group of three-dimensional rotations does not have a natural action on the two-dimensional plane. Specifically, it is possible that an element of $SO(3)$ will rotate an image out of plane. We show that an algorithm that learns a three-dimensional representation of the world from two dimensional images must satisfy certain consistency properties which we formulate as $SO(2)$-equivariance constraints. We use the induced representation of $SO(2)$ on $SO(3)$ to construct and classify architectures that have two-dimensional inputs and which satisfy these consistency constraints. We prove that any architecture which respects said consistency constraints can be realized as an instance of our construction. We show that three previously proposed neural architectures for 3D pose prediction are special cases of our construction. We propose a new algorithm that is a learnable generalization of previously considered methods. We test our architecture on three pose predictions task and achieve SOTA results on both the PASCAL3D+ and SYMSOL pose estimation tasks.</details>

### Errors-in-variables Fr\'echet Regression with Low-rank Covariate Approximation

**Authors:** Dogyoon Song, Kyunghee Han

**<details><summary>Abstract**</summary>

Fr\'echet regression has emerged as a promising approach for regression analysis involving non-Euclidean response variables. However, its practical applicability has been hindered by its reliance on ideal scenarios with abundant and noiseless covariate data. In this paper, we present a novel estimation method that tackles these limitations by leveraging the low-rank structure inherent in the covariate matrix. Our proposed framework combines the concepts of global Fr\'echet regression and principal component regression, aiming to improve the efficiency and accuracy of the regression estimator. By incorporating the low-rank structure, our method enables more effective modeling and estimation, particularly in high-dimensional and errors-in-variables regression settings. We provide a theoretical analysis of the proposed estimator's large-sample properties, including a comprehensive rate analysis of bias, variance, and additional variations due to measurement errors. Furthermore, our numerical experiments provide empirical evidence that supports the theoretical findings, demonstrating the superior performance of our approach. Overall, this work introduces a promising framework for regression analysis of non-Euclidean variables, effectively addressing the challenges associated with limited and noisy covariate data, with potential applications in diverse fields.</details>

### Estimating Riemannian Metric with Noise-Contaminated Intrinsic Distance

**Authors:** Jiaming Qiu, Xiongtao Dai

**<details><summary>Abstract**</summary>

We extend metric learning by studying the Riemannian manifold structure of the underlying data space induced by similarity measures between data points. The key quantity of interest here is the Riemannian metric, which characterizes the Riemannian geometry and defines straight lines and derivatives on the manifold. Being able to estimate the Riemannian metric allows us to gain insights into the underlying manifold and compute geometric features such as the geodesic curves. We model the observed similarity measures as noisy responses generated from a function of the intrinsic geodesic distance between data points. A new local regression approach is proposed to learn the Riemannian metric tensor and its derivatives based on a Taylor expansion for the squared geodesic distances, accommodating different types of data such as continuous, binary, or comparative responses. We develop theoretical foundation for our method by deriving the rates of convergence for the asymptotic bias and variance of the estimated metric tensor. The proposed method is shown to be versatile in simulation studies and real data applications involving taxi trip time in New York City and MNIST digits.</details>

### Estimating and Controlling for Equalized Odds via Sensitive Attribute Predictors

**Authors:** Beepul Bharti, Paul Yi, Jeremias Sulam

**<details><summary>Abstract**</summary>

As the use of machine learning models in real world high-stakes decision settings continues to grow, it is highly important that we are able to audit and control for any potential fairness violations these models may exhibit towards certain groups. To do so, one naturally requires access to sensitive attributes, such as demographics, biological sex, or other potentially sensitive features that determine group membership. Unfortunately, in many settings, this information is often unavailable. In this work we study the well known equalized odds (EOD) definition of fairness. In a setting without sensitive attributes, we first provide tight and computable upper bounds for the EOD violation of a predictor. These bounds precisely reflect the worst possible EOD violation. Second, we demonstrate how one can provably control the worst-case EOD by a new post-processing correction method. Our results characterize when directly controlling for EOD with respect to the predicted sensitive attributes is -- and when is not -- optimal when it comes to controlling worst-case EOD. Our results hold under assumptions that are milder than previous works, and we illustrate these results with experiments on synthetic and real datasets.</details>

### Estimating the Rate-Distortion Function by Wasserstein Gradient Descent

**Authors:** Yibo Yang, Stephan Eckstein, Marcel Nutz, Stephan Mandt

**<details><summary>Abstract**</summary>

In the theory of lossy compression, the rate-distortion (R-D) function $R(D)$ describes how much a data source can be compressed (in bit-rate) at any given level of fidelity (distortion). Obtaining $R(D)$ for a given data source establishes the fundamental performance limit for all compression algorithms.  We propose a new method to estimate $R(D)$ from the perspective of optimal transport. Unlike the classic Blahut--Arimoto algorithm which fixes the support of the reproduction distribution in advance, our Wasserstein gradient descent algorithm learns the support of the optimal reproduction distribution by moving particles. We prove its local convergence and analyze the sample complexity of our R-D estimator based on a connection to entropic optimal transport.  Experimentally, we obtain comparable or tighter bounds than state-of-the-art neural network methods on low-rate sources while requiring considerably less tuning and computation effort.  We also highlight a connection to maximum-likelihood deconvolution and introduce a new class of sources that can be used as test cases with known solutions to the R-D problem.</details>

### Evaluating Cognitive Maps and Planning in Large Language Models with CogEval

**Authors:** Ida Momennejad, Hosein Hasanbeig, Felipe Vieira Frujeri, Hiteshi Sharma, Nebojsa Jojic, Hamid Palangi, Robert Ness, Jonathan Larson

**<details><summary>Abstract**</summary>

Recently an influx of studies claims emergent cognitive abilities in large language models (LLMs). Yet, most rely on anecdotes, overlook contamination of training sets, or lack systematic Evaluation involving multiple tasks, control conditions, multiple iterations, and statistical robustness tests. Here we make two major contributions. First, we propose CogEval, a cognitive science-inspired protocol for the systematic evaluation of cognitive capacities in LLMs. The CogEval protocol can be followed for the evaluation of various abilities. Second, here we follow CogEval to systematically evaluate cognitive maps and planning ability across eight LLMs (OpenAI GPT-4, GPT-3.5-turbo-175B, davinci-003-175B, Google Bard, Cohere-xlarge-52.4B, Anthropic Claude-1-52B, LLaMA-13B, and Alpaca-7B). We base our task prompts on human experiments, which offer both established construct validity for evaluating planning, and are absent from LLM training sets. We find that, while LLMs show apparent competence in a few planning tasks with simpler structures, systematic evaluation reveals striking failure modes in planning tasks, including hallucinations of invalid trajectories and falling in loops. These findings do not support the idea of emergent out-of-the-box planning ability in LLMs. This could be because LLMs do not understand the latent relational structures underlying planning problems, known as cognitive maps, and fail at unrolling goal-directed trajectories based on the underlying structure. Implications for application and future directions are discussed.</details>

### Evaluating Neuron Interpretation Methods of NLP Models

**Authors:** Yimin Fan, Fahim Dalvi, Nadir Durrani, Hassan Sajjad

**<details><summary>Abstract**</summary>

Neuron interpretation offers valuable insights into how knowledge is structured within a deep neural network model. While a number of neuron interpretation methods have been proposed in the literature, the field lacks a comprehensive comparison among these methods. This gap hampers progress due to the absence of standardized metrics and benchmarks. The commonly used evaluation metric has limitations, and creating ground truth annotations for neurons is impractical. Addressing these challenges, we propose an evaluation framework based on voting theory. Our hypothesis posits that neurons consistently identified by different methods carry more significant information. We rigorously assess our framework across a diverse array of neuron interpretation methods. Notable findings include: i) despite the theoretical differences among the methods, neuron ranking methods share over 60% of their rankings when identifying salient neurons, ii) the neuron interpretation methods are most sensitive to the last layer representations, iii) Probeless neuron ranking emerges as the most consistent method.</details>

### Exact Representation of Sparse Networks with Symmetric Nonnegative Embeddings

**Authors:** Sudhanshu Chanpuriya, Ryan Rossi, Anup Rao, Tung Mai, Nedim Lipka, Zhao Song, Cameron Musco

**<details><summary>Abstract**</summary>

Graph models based on factorization of the adjacency matrix often fail to capture network structures related to links between dissimilar nodes (heterophily). We introduce a novel graph factorization model that leverages two nonnegative vectors per node to interpretably account for links between both similar and dissimilar nodes. We prove that our model can exactly represent any graph with low *arboricity*, a property that many real-world networks satisfy; our proof also applies to related models but has much greater scope than the closest prior bound, which is based on low *max degree*. Our factorization also has compelling properties besides expressiveness: due to its symmetric structure and nonnegativity, fitting the model inherently finds node communities, and the model's link predictions can be interpreted in terms of these communities. In experiments on real-world networks, we demonstrate our factorization's effectiveness on a variety of tasks, including community detection and link prediction.</details>

### Exact recovery and Bregman hard clustering of node-attributed Stochastic Block Model

**Authors:** Maximilien Dreveton, Felipe Fernandes, Daniel Figueiredo

**<details><summary>Abstract**</summary>

Classic network clustering tackles the problem of identifying sets of nodes (communities) that have similar connection patterns. However, in many scenarios nodes also have attributes that are correlated and can also be used to identify node clusters. Thus, network information (edges) and node information (attributes) can be jointly leveraged to design high-performance clustering algorithms. Under a general model for the network and node attributes, this work establishes an information-theoretic criteria for the exact recovery of community labels and characterizes a phase transition determined by the Chernoff-Hellinger divergence of the model. The criteria shows how network and attribute information can be exchanged in order to have exact recovery (e.g., more reliable network information requires less reliable attribute information). This work also presents an iterative clustering algorithm that maximizes the joint likelihood, assuming that the probability distribution of network interactions and node attributes belong to exponential families. This covers a broad range of possible interactions (e.g., edges with weights) and attributes (e.g., non-Gaussian models) while also exploring the connection between exponential families and Bregman divergences. Extensive numerical experiments using synthetic and real data indicate that the proposed algorithm outperforms algorithms that leverage only network or only attribute information as well as recently proposed algorithms that perform clustering using both sources of information. The contributions of this work provide insights into the fundamental limits and practical techniques for inferring community labels on node-attributed networks.</details>

### Experiment Planning with Function Approximation

**Authors:** Aldo Pacchiano, Jonathan Lee, Emma Brunskill

**<details><summary>Abstract**</summary>

We study the problem of experiment planning with function approximation in contextual bandit problems. In settings where there is a significant overhead to deploying adaptive algorithms; for example, when the execution of the data collection policies is required to be distributed, or a human in the loop is needed to implement these policies, producing in advance a set of policies for data collection is paramount. We study the setting where a large dataset of contexts -but not rewards- is available and may be used by the learner to design an effective data collection strategy. Although when rewards are linear this problem has been well studied, results are still missing for more complex reward models. In this work we propose two experiment planning strategies compatible with function approximation, first an eluder planning and sampling procedure that can recover optimality guarantees depending on the eluder dimension of the reward function class, and second we show the uniform sampler achieves competitive optimality rates in the setting where the number of actions is small. We finalize our results introducing a statistical gap fleshing out the fundamental differences between planning and adaptive learning and provide results for planning with model selection.</details>

### Expert load matters: operating networks at high accuracy and low manual effort

**Authors:** Sara Sangalli, Ertunc Erdil, Ender Konukoglu

**<details><summary>Abstract**</summary>

In human-AI collaboration systems for critical applications, in order to ensure minimal error, users should set an operating point based on model confidence to determine when the decision should be delegated to human experts. Samples for which model confidence is lower than the operating point would be manually analysed by experts to avoid mistakes.Such systems can become truly useful only if they consider two aspects: models should be confident only for samples for which they are accurate, and the number of samples delegated to experts should be minimized.The latter aspect is especially crucial for applications where available expert time is limited and expensive, such as healthcare. The trade-off between the model accuracy and the number of samples delegated to experts can be represented by a curve that is similar to an ROC curve, which we refer to as confidence operating characteristic (COC) curve. In this paper, we argue that deep neural networks should be trained by taking into account both accuracy and expert load and, to that end, propose a new complementary loss function for classification that maximizes the area under this COC curve.This promotes simultaneously the increase in network accuracy and the reduction in number of samples delegated to humans.We perform experiments on multiple computer vision and medical image datasets for classification.Our results demonstrate that the proposed loss improves classification accuracy and delegates less number of decisions to experts, achieves better out-of-distribution samples detection and on par calibration performance compared to existing loss functions.</details>

### Explain Any Concept: Segment Anything Meets Concept-Based Explanation

**Authors:** Ao Sun, Pingchuan Ma, Yuanyuan Yuan, Shuai Wang

**<details><summary>Abstract**</summary>

EXplainable AI (XAI) is an essential topic to improve human understanding of deep neural networks (DNNs) given their black-box internals. For computer vision tasks, mainstream pixel-based XAI methods explain DNN decisions by identifying important pixels, and emerging concept-based XAI explore forming explanations with concepts (e.g., a head in an image). However, pixels are generally hard to interpret and sensitive to the imprecision of XAI methods, whereas “concepts” in prior works require human annotation or are limited to pre-defined concept sets. On the other hand, driven by large-scale pre-training, Segment Anything Model (SAM) has been demonstrated as a powerful and promotable framework for performing precise and comprehensive instance segmentation, enabling automatic preparation of concept sets from a given image. This paper for the first time explores using SAM to augment concept-based XAI. We offer an effective and flexible concept-based explanation method, namely Explain Any Concept (EAC), which explains DNN decisions with any concept. While SAM is highly effective and offers an “out-of-the-box” instance segmentation, it is costly when being integrated into defacto XAI pipelines. We thus propose a lightweight per-input equivalent (PIE) scheme, enabling efficient explanation with a surrogate model. Our evaluation  over two popular datasets (ImageNet and COCO) illustrate the highly encouraging performance of EAC over commonly-used XAI methods.</details>

### Explainable Brain Age Prediction using coVariance Neural Networks

**Authors:** Saurabh Sihag, Gonzalo Mateos, Corey McMillan, Alejandro Ribeiro

**<details><summary>Abstract**</summary>

In computational neuroscience, there has been an increased interest in developing machine learning algorithms that leverage brain imaging data to provide estimates of "brain age" for an individual. Importantly, the discordance between brain age and chronological age (referred to as "brain age gap") can capture accelerated aging due to adverse health conditions and therefore, can reflect increased vulnerability towards neurological disease or cognitive impairments. However, widespread adoption of brain age for clinical decision support has been hindered due to lack of transparency and methodological justifications in most existing brain age prediction algorithms. In this paper, we leverage coVariance neural networks (VNN) to propose an explanation-driven and anatomically interpretable framework for brain age prediction using cortical thickness features. Specifically, our brain age prediction framework extends beyond the coarse metric of brain age gap in Alzheimer’s disease (AD) and we make two important observations: (i) VNNs can assign anatomical interpretability to elevated brain age gap in AD by identifying contributing brain regions, (ii) the interpretability offered by VNNs is contingent on their ability to exploit specific eigenvectors of the anatomical covariance matrix. Together, these observations facilitate an explainable and anatomically interpretable perspective to the task of brain age prediction.</details>

### Exploring Diverse In-Context Configurations for Image Captioning

**Authors:** Xu Yang, Yongliang Wu, Mingzhuo Yang, Haokun Chen, Xin Geng

**<details><summary>Abstract**</summary>

After discovering that Language Models (LMs) can be good in-context few-shot learners, numerous strategies have been proposed to optimize in-context sequence configurations. Recently, researchers in Vision-Language (VL) domains also develop their few-shot learners, while they only use the simplest way, \ie, randomly sampling, to configure in-context image-text pairs. In order to explore the effects of varying configurations on VL in-context learning, we devised four strategies for image selection and four for caption assignment to configure in-context image-text pairs for image captioning. Here Image Captioning is used as the case study since it can be seen as the visually-conditioned LM. Our comprehensive experiments yield two counter-intuitive but valuable insights, highlighting the distinct characteristics of VL in-context learning due to multi-modal synergy, as compared to the NLP case. Furthermore, in our exploration of optimal combination strategies, we observed an average performance enhancement of 20.9 of CIDEr scores compared to the baseline. The code is given in https://github.com/yongliang-wu/ExploreCfg.</details>

### [Spotlight] Exposing Attention Glitches with Flip-Flop Language Modeling

**Authors:** Bingbin Liu, Jordan Ash, Surbhi Goel, Akshay Krishnamurthy, Cyril Zhang

**<details><summary>Abstract**</summary>

Why do large language models sometimes output factual inaccuracies and exhibit erroneous reasoning? The brittleness of these models, particularly when executing long chains of reasoning, currently seems to be an inevitable price to pay for their advanced capabilities of coherently synthesizing knowledge, pragmatics, and abstract thought. Towards making sense of this fundamentally unsolved problem, this work identifies and analyzes the phenomenon of _attention glitches_, in which the Transformer architecture's inductive biases intermittently fail to capture robust reasoning. To isolate the issue, we introduce _flip-flop language modeling_ (FFLM), a parametric family of synthetic benchmarks designed to probe the extrapolative behavior of neural language models. This simple generative task requires a model to copy binary symbols over long-range dependencies, ignoring the tokens in between. We find that Transformer FFLMs suffer from a long tail of sporadic reasoning errors, some of which we can eliminate using various regularization techniques. Our preliminary mechanistic analyses show why the remaining errors may be very difficult to diagnose and resolve. We hypothesize that attention glitches account for (some of) the closed-domain hallucinations in natural LLMs.</details>

### [Spotlight] Expressive Sign Equivariant Networks for Spectral Geometric Learning

**Authors:** Derek Lim, Joshua Robinson, Stefanie Jegelka, Haggai Maron

**<details><summary>Abstract**</summary>

Recent work has shown the utility of developing machine learning models that respect the structure and symmetries of eigenvectors. These works promote sign invariance, since for any eigenvector v the negation -v is also an eigenvector. However, we show that sign invariance is theoretically limited for tasks such as building orthogonally equivariant models and learning node positional encodings for link prediction in graphs. In this work, we demonstrate the benefits of sign equivariance for these tasks. To obtain these benefits, we develop novel sign equivariant neural network architectures. Our models are based on a new analytic characterization of sign equivariant polynomials and thus inherit provable expressiveness properties. Controlled synthetic experiments show that our networks can achieve the theoretically predicted benefits of sign equivariant models.</details>

### Expressive probabilistic sampling in recurrent neural networks

**Authors:** Shirui Chen, Linxing Jiang, Rajesh PN Rao, Eric Shea-Brown

**<details><summary>Abstract**</summary>

In sampling-based Bayesian models of brain function, neural activities are assumed to be samples from probability distributions that the brain uses for probabilistic computation. However, a comprehensive understanding of how mechanistic models of neural dynamics can sample from arbitrary distributions is still lacking. We use tools from functional analysis and stochastic differential equations to explore the minimum architectural requirements for $\textit{recurrent}$ neural circuits to sample from complex distributions. We first consider the traditional sampling model consisting of a network of neurons whose outputs directly represent the samples ($\textit{sampler-only}$ network). We argue that synaptic current and firing-rate dynamics in the traditional model have limited capacity to sample from a complex probability distribution. We show that the firing rate dynamics of a recurrent neural circuit with a separate set of output units can sample from an arbitrary probability distribution. We call such circuits $\textit{reservoir-sampler networks}$ (RSNs). We propose an efficient training procedure based on denoising score matching that finds recurrent and output weights such that the RSN implements Langevin sampling. We empirically demonstrate our model's ability to sample from several complex data distributions using the proposed neural dynamics and discuss its applicability to developing the next generation of sampling-based Bayesian brain models.</details>

### Extending the Design Space of Graph Neural Networks by Rethinking Folklore Weisfeiler-Lehman

**Authors:** Jiarui Feng, Lecheng Kong, Hao Liu, Dacheng Tao, Fuhai Li, Muhan Zhang, Yixin Chen

**<details><summary>Abstract**</summary>

Message passing neural networks (MPNNs) have emerged as the most popular framework of graph neural networks (GNNs) in recent years. However, their expressive power is limited by the 1-dimensional Weisfeiler-Lehman (1-WL) test. Some works are inspired by $k$-WL/FWL (Folklore WL) and design the corresponding neural versions. Despite the high expressive power, there are serious limitations in this line of research. In particular, (1) $k$-WL/FWL requires at least $O(n^k)$ space complexity, which is impractical for large graphs even when $k=3$; (2) The design space of $k$-WL/FWL is rigid, with the only adjustable hyper-parameter being $k$. To tackle the first limitation, we propose an extension, $(k, t)$-FWL. We theoretically prove that even if we fix the space complexity to $O(n^k)$ (for any $k \geq 2$) in $(k, t)$-FWL, we can construct an expressiveness hierarchy up to solving the graph isomorphism problem. To tackle the second problem, we propose $k$-FWL+, which considers any equivariant set as neighbors instead of all nodes, thereby greatly expanding the design space of $k$-FWL. Combining these two modifications results in a flexible and powerful framework $(k, t)$-FWL+. We demonstrate $(k, t)$-FWL+ can implement most existing models with matching expressiveness. We then introduce an instance of $(k,t)$-FWL+ called Neighborhood$^2$-FWL (N$^2$-FWL), which is practically and theoretically sound. We prove that N$^2$-FWL is no less powerful than 3-WL, and can encode many substructures while only requiring $O(n^2)$ space. Finally, we design its neural version named **N$^2$-GNN** and evaluate its performance on various tasks. N$^2$-GNN achieves record-breaking results on ZINC-Subset (**0.059**) and ZINC-Full (**0.013**), outperforming previous SOTA results by 10.6% and 40.9\%, respectively. Moreover, N$^2$-GNN achieves new SOTA results on the BREC dataset (**71.8\%**) among all existing high-expressive GNN methods.</details>

### Factorized Contrastive Learning: Going Beyond Multi-view Redundancy

**Authors:** Paul Pu Liang, Zihao Deng, Martin Q. Ma, James Zou, Louis-Philippe Morency, Ruslan Salakhutdinov

**<details><summary>Abstract**</summary>

In a wide range of multimodal tasks, contrastive learning has become a particularly appealing approach since it can successfully learn representations from abundant unlabeled data with only pairing information (e.g., image-caption or video-audio pairs). Underpinning these approaches is the assumption of multi-view redundancy - that shared information between modalities is necessary and sufficient for downstream tasks. However, in many real-world settings, task-relevant information is also contained in modality-unique regions: information that is only present in one modality but still relevant to the task. How can we learn self-supervised multimodal representations to capture both shared and unique information relevant to downstream tasks? This paper proposes FactorCL, a new multimodal representation learning method to go beyond multi-view redundancy. FactorCL is built from three new contributions: (1) factorizing task-relevant information into shared and unique representations, (2) capturing task-relevant information via maximizing MI lower bounds and removing task-irrelevant information via minimizing MI upper bounds, and (3) multimodal data augmentations to approximate task relevance without labels. On large-scale real-world datasets, FactorCL captures both shared and unique information and achieves state-of-the-art results on six benchmarks.</details>

### Fair Canonical Correlation Analysis

**Authors:** Zhuoping Zhou, Davoud Ataee Tarzanagh, Bojian Hou, Boning Tong, Jia Xu, Yanbo Feng, Qi Long, Li Shen

**<details><summary>Abstract**</summary>

This paper investigates fairness and bias in Canonical Correlation Analysis (CCA), a widely used statistical technique for examining the relationship between two sets of variables. We present a framework that alleviates unfairness by minimizing the correlation disparity error associated with protected attributes. Our approach enables the CCA model to learn global projection matrices from all data points while ensuring that these matrices yield comparable correlation levels to group-specific projection matrices. Experimental evaluation on both synthetic and real-world datasets demonstrates the efficacy of our method in reducing unfairness without compromising CCA model accuracy. These findings emphasize the importance of considering fairness in CCA applications to real-world problems.</details>

### Fair, Polylog-Approximate Low-Cost Hierarchical Clustering

**Authors:** Marina Knittel, Max Springer, John Dickerson, MohammadTaghi Hajiaghayi

**<details><summary>Abstract**</summary>

Research in fair machine learning, and particularly clustering, has been crucial in recent years given the many ethical controversies that modern intelligent systems have posed. Ahmadian et al. [2020] established the study of fairness in hierarchical clustering, a stronger, more structured variant of its well-known flat counterpart, though their proposed algorithm that optimizes for Dasgupta's [2016] famous cost function was highly theoretical. Knittel et al. [2023] then proposed the first practical fair approximation for cost, however they were unable to break the polynomial-approximate barrier they posed as a hurdle of interest. We break this barrier, proposing the first truly polylogarithmic-approximate low-cost fair hierarchical clustering, thus greatly bridging the gap between the best fair and vanilla hierarchical clustering approximations.</details>

### Fast Attention Over Long Sequences With Dynamic Sparse Flash Attention

**Authors:** Matteo Pagliardini, Daniele Paliotta, Martin Jaggi, François Fleuret

**<details><summary>Abstract**</summary>

Transformer-based language models have found many diverse applications requiring them to process sequences of increasing length. For these applications, the causal self-attention---which is the only component scaling quadratically w.r.t. the sequence length---becomes a central concern. While many works have proposed schemes to sparsify the attention patterns and reduce the computational overhead of self-attention, those are often limited by implementation concerns and end up imposing a simple and static structure over the attention matrix. Conversely, implementing more dynamic sparse attention often results in runtimes significantly slower than computing the full attention using the Flash implementation from Dao et al. (2022). We extend FlashAttention to accommodate a large class of attention sparsity patterns that, in particular, encompass key/query dropping and hashing-based attention. This leads to implementations with no computational complexity overhead and a multi-fold runtime speedup on top of FlashAttention. Even with relatively low degrees of sparsity, our method improves visibly upon FlashAttention as the sequence length increases. Without sacrificing perplexity, we increase the training speed of a transformer language model by $2.0\times$ and $3.3\times$ for sequences of respectively $8k$ and $16k$ tokens.</details>

### Fast Exact Leverage Score Sampling from Khatri-Rao Products with Applications to Tensor Decomposition

**Authors:** Vivek Bharadwaj, Osman Asif Malik, Riley Murray, Laura Grigori, Aydin Buluc, James Demmel

**<details><summary>Abstract**</summary>

We present a data structure to randomly sample rows from the Khatri-Rao product of several matrices according to the exact distribution of its leverage scores. Our proposed sampler draws each row in time logarithmic in the height of the Khatri-Rao product and quadratic in its column count, with persistent space overhead at most the size of the input matrices. As a result, it tractably draws samples even when the matrices forming the Khatri-Rao product have tens of millions of rows each. When used to sketch the linear least-squares problems arising in Candecomp / PARAFAC decomposition, our method achieves lower asymptotic complexity per solve than recent state-of-the-art methods. Experiments on billion-scale sparse tensors and synthetic data validate our theoretical claims, with our algorithm achieving higher accuracy than competing methods as the decomposition rank grows.</details>

### Fast Trainable Projection for Robust Fine-tuning

**Authors:** Junjiao Tian, Yen-Cheng Liu, James S Smith, Zsolt Kira

**<details><summary>Abstract**</summary>

Robust fine-tuning aims to achieve competitive in-distribution (ID) performance while maintaining the out-of-distribution (OOD) robustness of a pre-trained model when transferring it to a downstream task. Recently, projected gradient descent has been successfully used in robust fine-tuning by constraining the deviation from the initialization of the fine-tuned model explicitly through projection. However, algorithmically, two limitations prevent this method from being adopted more widely, scalability and efficiency. In this paper, we propose a new projection-based fine-tuning algorithm, Fast Trainable Projection (FTP) for computationally efficient learning of per-layer projection constraints, resulting in an average 35% speedup on our benchmarks compared to prior works. FTP can be combined with existing optimizers such as AdamW, and be used in a plug-and-play fashion. Finally, we show that FTP is a special instance of hyper-optimizers that tune the hyper-parameters of optimizers in a learnable manner through nested differentiation. Empirically, we show superior robustness on OOD datasets, including domain shifts and natural corruptions, across four different vision tasks with five different pre-trained models. Additionally, we demonstrate that FTP is broadly applicable and beneficial to other learning scenarios such as low-label and continual learning settings thanks to its easy adaptability. The code will be available at https://github.com/GT-RIPL/FTP.git.</details>

### Fast and Regret Optimal Best Arm Identification: Fundamental Limits and Low-Complexity Algorithms

**Authors:** Qining Zhang, Lei Ying

**<details><summary>Abstract**</summary>

This paper considers a stochastic Multi-Armed Bandit (MAB) problem with dual objectives: (i) quick identification and commitment to the optimal arm, and (ii) reward maximization throughout a sequence of $T$ consecutive rounds. Though each objective has been individually well-studied, i.e., best arm identification for (i) and regret minimization for (ii), the simultaneous realization of both objectives remains an open problem, despite its practical importance. This paper introduces \emph{Regret Optimal Best Arm Identification} (ROBAI) which aims to achieve these dual objectives. To solve ROBAI with both pre-determined stopping time and adaptive stopping time requirements, we present an algorithm called EOCP and its variants respectively, which not only achieve asymptotic optimal regret in both Gaussian and general bandits, but also commit to the optimal arm in $\mathcal{O}(\log T)$ rounds with pre-determined stopping time and $\mathcal{O}(\log^2 T)$ rounds with adaptive stopping time. We further characterize lower bounds on the commitment time (equivalent to the sample complexity) of ROBAI, showing that EOCP and its variants are sample optimal with pre-determined stopping time, and almost sample optimal with adaptive stopping time. Numerical results confirm our theoretical analysis and reveal an interesting ``over-exploration'' phenomenon carried by classic UCB algorithms, such that EOCP has smaller regret even though it stops exploration much earlier than UCB, i.e., $\mathcal{O}(\log T)$ versus $\mathcal{O}(T)$, which suggests over-exploration is unnecessary and potentially harmful to system performance.</details>

### Feature Selection in the Contrastive Analysis Setting

**Authors:** Ethan Weinberger, Ian Covert, Su-In Lee

**<details><summary>Abstract**</summary>

Contrastive analysis (CA) refers to the exploration of variations uniquely enriched in a _target_ dataset as compared to a corresponding _background_ dataset generated from sources of variation that are irrelevant to a given task. For example, a biomedical data analyst may wish to find a small set of genes to use as a proxy for variations in genomic data only present among patients with a given disease (target) as opposed to healthy control subjects (background). However, as of yet the problem of feature selection in the CA setting has received little attention from the machine learning community. In this work we present contrastive feature selection (CFS),a method for performing feature selection in the CA setting. We motivate our approach with a novel information-theoretic analysis of representation learning in the CA setting, and we empirically validate CFS on a semi-synthetic dataset and four real-world biomedical datasets. We find that our method consistently outperforms previously proposed state-of-the-art supervised and fully unsupervised feature selection methods not designed for the CA setting. An open-source implementation of our method is available at https://github.com/suinleelab/CFS.</details>

### Fed-CO$_{2}$: Cooperation of Online and Offline Models for Severe Data Heterogeneity in Federated Learning

**Authors:** Zhongyi Cai, Ye Shi, Wei Huang, Jingya Wang

**<details><summary>Abstract**</summary>

Federated Learning (FL) has emerged as a promising distributed learning paradigm that enables multiple clients to learn a global model collaboratively without sharing their private data. However, the effectiveness of FL is highly dependent on the quality of the data that is being used for training. In particular, data heterogeneity issues, such as label distribution skew and feature skew, can significantly impact the performance of FL. Previous studies in FL have primarily focused on addressing label distribution skew data heterogeneity, while only a few recent works have made initial progress in tackling feature skew issues. Notably, these two forms of data heterogeneity have been studied separately and have not been well explored within a unified FL framework. To address this gap, we propose Fed-CO$_2$, a universal FL framework that handles both label distribution skew and feature skew within a Cooperation mechanism between the Online and Offline models. Specifically, the online model learns general knowledge that is shared among all clients, while the offline model is trained locally to learn the specialized knowledge of each individual client. To further enhance model cooperation in the presence of feature shifts, we design an intra-client knowledge transfer mechanism that reinforces mutual learning between the online and offline models, and an inter-client knowledge transfer mechanism to increase the models’ domain generalization ability. Extensive experiments show that our Fed-CO$_2$ outperforms a wide range of existing personalized federated learning algorithms in terms of handling label distribution skew and feature skew, both individually and collectively. The empirical results are supported by our convergence analyses in a simplified setting.</details>

### Federated Learning via Meta-Variational Dropout

**Authors:** Insu Jeon, Minui Hong, Junhyeog Yun, Gunhee Kim

**<details><summary>Abstract**</summary>

Federated Learning (FL) aims to train a global inference model from remotely distributed clients, gaining popularity due to its benefit of improving data privacy. However, traditional FL often faces challenges in practical applications, including model overfitting and divergent local models due to limited and non-IID data among clients. To address these issues, we introduce a novel Bayesian meta-learning approach called meta-variational dropout (MetaVD). MetaVD learns to predict client-dependent dropout rates via a shared hypernetwork, enabling effective model personalization of FL algorithms in limited non-IID data settings. We also emphasize the posterior adaptation view of meta-learning and the posterior aggregation view of Bayesian FL via the conditional dropout posterior. We conducted extensive experiments on various sparse and non-IID FL datasets. MetaVD demonstrated excellent classification accuracy and uncertainty calibration performance, especially for out-of-distribution (OOD) clients. MetaVD compresses the local model parameters needed for each client, mitigating model overfitting and reducing communication costs. Code is available at https://github.com/insujeon/MetaVD.</details>

### Federated Learning with Client Subsampling, Data Heterogeneity, and Unbounded Smoothness: A New Algorithm and Lower Bounds

**Authors:** Michael Crawshaw, Yajie Bao, Mingrui Liu

**<details><summary>Abstract**</summary>

We study the problem of Federated Learning (FL) under client subsampling and data heterogeneity with an objective function that has potentially unbounded smoothness. This problem is motivated by empirical evidence that the class of relaxed smooth functions, where the Lipschitz constant of the gradient scales linearly with the gradient norm, closely resembles the loss functions of certain neural networks such as recurrent neural networks (RNNs) with possibly exploding gradient. We introduce EPISODE++, the first algorithm to solve this problem. It maintains historical statistics for each client to construct control variates and decide clipping behavior for sampled clients in the current round. We prove that EPISODE++ achieves linear speedup in the number of participating clients, reduced communication rounds, and resilience to data heterogeneity. Our upper bound proof relies on novel techniques of recursively bounding the client updates under unbounded smoothness and client subsampling, together with a refined high probability analysis. In addition, we prove a lower bound showing that the convergence rate of a special case of clipped minibatch SGD (without randomness in the stochastic gradient and with randomness in client subsampling) suffers from an explicit dependence on the maximum gradient norm of the objective in a sublevel set, which may be large. This effectively demonstrates that applying gradient clipping to minibatch SGD in our setting does not eliminate the problem of exploding gradients. Our lower bound is based on new constructions of hard instances tailored to client subsampling and a novel analysis of the trajectory of the algorithm in the presence of clipping. Lastly, we provide an experimental evaluation of EPISODE++ when training RNNs on federated text classification tasks, demonstrating that EPISODE++ outperforms strong baselines in FL. The code is available at https://github.com/MingruiLiu-ML-Lab/episode_plusplus.</details>

### Federated Linear Bandits with Finite Adversarial Actions

**Authors:** Li Fan, Ruida Zhou, Chao Tian, Cong Shen

**<details><summary>Abstract**</summary>

We study a federated linear bandits model, where $M$ clients communicate with a central server to solve a linear contextual bandits problem with finite adversarial action sets that may be different across clients. To address the unique challenges of **adversarial finite** action sets, we propose the FedSupLinUCB algorithm, which extends the principles of SupLinUCB and OFUL algorithms in linear contextual bandits. We prove that FedSupLinUCB achieves a total regret of $\tilde{O}(\sqrt{d T})$, where $T$ is the total number of arm pulls from all clients, and $d$ is the ambient dimension of the linear model. This matches the minimax lower bound and thus is order-optimal (up to polylog terms). We study both asynchronous and synchronous cases and show that the communication cost can be controlled as $O(d M^2 \log(d)\log(T))$ and $O(\sqrt{d^3 M^3} \log(d))$, respectively. The FedSupLinUCB design is further extended to two scenarios: (1) variance-adaptive, where a total regret of $\tilde{O} (\sqrt{d \sum \nolimits_{t=1}^{T} \sigma_t^2})$ can be achieved with $\sigma_t^2$ being the noise variance of round $t$; and (2) adversarial corruption, where a total regret of $\tilde{O}(\sqrt{dT} + d C_p)$ can be achieved with $C_p$ being the total corruption budget. Experiment results corroborate the theoretical analysis and demonstrate the effectiveness of \alg on both synthetic and real-world datasets.</details>

### Federated Spectral Clustering via Secure Similarity Reconstruction

**Authors:** Dong Qiao, Chris Ding, Jicong Fan

**<details><summary>Abstract**</summary>

Federated learning has a significant advantage in protecting information privacy. Many scholars proposed various secure learning methods within the framework of federated learning but the study on secure federated unsupervised learning especially clustering is limited. We in this work propose a secure kernelized factorization method for federated spectral clustering on distributed dataset. The method is non-trivial because the kernel or similarity matrix for spectral clustering is computed by data pairs, which violates the principle of privacy protection. Our method implicitly constructs an approximation for the kernel matrix on distributed data such that we can perform spectral clustering under the constraint of privacy protection. We provide a convergence guarantee of the optimization algorithm, reconstruction error bounds of the Gaussian kernel matrix, and the sufficient condition of correct clustering of our method. We also present some results of differential privacy. Numerical results on synthetic and real datasets demonstrate that the proposed method is efficient and accurate in comparison to the baselines.</details>

### Few-shot Generation via Recalling  Brain-Inspired Episodic-Semantic Memory

**Authors:** Zhibin Duan, Zhiyi Lv, Chaojie Wang, Bo Chen, Bo An, Mingyuan Zhou

**<details><summary>Abstract**</summary>

Aimed at adapting a generative model to a novel generation task with only a few given data samples, the capability of few-shot generation is crucial for many real-world applications with limited data, \emph{e.g.}, artistic domains.Instead of training from scratch, recent works tend to leverage the prior knowledge stored in previous datasets, which is quite similar to the memory mechanism of human intelligence, but few of these works directly imitate the memory-recall mechanism that humans make good use of in accomplishing creative tasks,  \emph{e.g.}, painting and writing.Inspired by the memory mechanism of human brain, in this work, we carefully design a variational structured memory module (VSM), which can simultaneously store both episodic and semantic memories to assist existing generative models efficiently recall these memories during sample generation.Meanwhile, we introduce a bionic memory updating strategy for the conversion between episodic and semantic memories, which can also model the uncertainty during conversion.Then, we combine the developed VSM with various generative models under the Bayesian framework, and evaluate these memory-augmented generative models with few-shot generation tasks, demonstrating the effectiveness of our methods.</details>

### Finding Counterfactually Optimal Action Sequences in Continuous State Spaces

**Authors:** Stratis Tsirtsis, Manuel Rodriguez

**<details><summary>Abstract**</summary>

Whenever a clinician reflects on the efficacy of a sequence of treatment decisions for a patient, they may try to identify critical time steps where, had they made different decisions, the patient's health would have improved. While recent methods at the intersection of causal inference and reinforcement learning promise to aid human experts, as the clinician above, to *retrospectively* analyze sequential decision making processes, they have focused on environments with finitely many discrete states. However, in many practical applications, the state of the environment is inherently continuous in nature. In this paper, we aim to fill this gap. We start by formally characterizing a sequence of discrete actions and continuous states using finite horizon Markov decision processes and a broad class of bijective structural causal models. Building upon this characterization, we formalize the problem of finding counterfactually optimal action sequences and show that, in general, we cannot expect to solve it in polynomial time. Then, we develop a search method based on the A* algorithm that, under a natural form of Lipschitz continuity of the environment’s dynamics, is guaranteed to return the optimal solution to the problem. Experiments on real clinical data show that our method is very efficient in practice, and it has the potential to offer interesting insights for sequential decision making tasks.</details>

### Finding Order in Chaos: A Novel Data Augmentation Method for Time Series in Contrastive Learning

**Authors:** Berken Utku Demirel, Christian Holz

**<details><summary>Abstract**</summary>

The success of contrastive learning is well known to be dependent on data augmentation.Although the degree of data augmentations has been well controlled by utilizing pre-defined techniques in some domains like vision, time-series data augmentation is less explored and remains a challenging problem due to the complexity of the data generation mechanism, such as the intricate mechanism involved in the cardiovascular system.Moreover, there is no widely recognized and general time-series augmentation method that can be applied across different tasks.In this paper, we propose a novel data augmentation method for time-series tasks that aims to connect intra-class samples together, and thereby find order in the latent space.Our method builds upon the well-known data augmentation technique of mixup by incorporating a novel approach that accounts for the non-stationary nature of time-series data.Also, by controlling the degree of chaos created by data augmentation, our method leads to improved feature representations and performance on downstream tasks.We evaluate our proposed method on three time-series tasks, including heart rate estimation, human activity recognition, and cardiovascular disease detection. Extensive experiments against the state-of-the-art methods show that the proposed method outperforms prior works on optimal data generation and known data augmentation techniques in three tasks, reflecting the effectiveness of the presented method. The source code is available at double-blind policy.</details>

### [Oral] Fine-Tuning Language Models with Just Forward Passes

**Authors:** Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason Lee, Danqi Chen, Sanjeev Arora

**Oral Presentation:** We, Dec 13, 14:15 -- Oral 4A

**<details><summary>Abstract**</summary>

Fine-tuning language models (LMs) has yielded success on diverse downstream tasks, but as LMs grow in size, backpropagation requires a prohibitively large amount of memory. Zeroth-order (ZO) methods can in principle estimate gradients using only two forward passes but are theorized to be catastrophically slow for optimizing large models. In this work, we propose a memory-efficient zerothorder optimizer (MeZO), adapting the classical ZO-SGD method to operate in-place, thereby fine-tuning LMs with the same memory footprint as inference. For example, with a single A100 80GB GPU, MeZO can train a 30-billion parameter model, whereas fine-tuning with backpropagation can train only a 2.7B LM with the same budget. We conduct comprehensive experiments across model types (masked and autoregressive LMs), model scales (up to 66B), and downstream tasks (classification, multiple-choice, and generation). Our results demonstrate that (1) MeZO significantly outperforms in-context learning and linear probing; (2) MeZO achieves comparable performance to fine-tuning with backpropagation across multiple tasks, with up to 12× memory reduction and up to 2× GPU-hour reduction in our implementation; (3) MeZO is compatible with both full-parameter and parameter-efficient tuning techniques such as LoRA and prefix tuning; (4) MeZO can effectively optimize non-differentiable objectives (e.g., maximizing accuracy or F1). We support our empirical findings with theoretical insights, highlighting how adequate pre-training and task prompts enable MeZO to fine-tune huge models, despite classical ZO analyses suggesting otherwise.</details>

### Fixing the NTK: From Neural Network Linearizations to Exact Convex Programs

**Authors:** Rajat Vadiraj Dwaraknath, Tolga Ergen, Mert Pilanci

**<details><summary>Abstract**</summary>

Recently, theoretical analyses of deep neural networks have broadly focused on two directions: 1) Providing insight into neural network training by SGD in the limit of infinite hidden-layer width and infinitesimally small learning rate (also known as gradient flow) via the Neural Tangent Kernel (NTK), and 2) Globally optimizing the regularized training objective via cone-constrained convex reformulations of ReLU networks. The latter research direction also yielded an alternative formulation of the ReLU network, called a gated ReLU network, that is globally optimizable via efficient unconstrained convex programs. In this work, we interpret the convex program for this gated ReLU network as a Multiple Kernel Learning (MKL) model with a weighted data masking feature map and establish a connection to the NTK. Specifically, we show that for a particular choice of mask weights that do not depend on the learning targets, this kernel is equivalent to the NTK of the gated ReLU network on the training data. A consequence of this lack of dependence on the targets is that the NTK cannot perform better than the optimal MKL kernel on the training set. By using iterative reweighting, we improve the weights induced by the NTK to obtain the optimal MKL kernel which is equivalent to the solution of the exact convex reformulation of the gated ReLU network. We also provide several numerical simulations corroborating our theory. Additionally, we provide an analysis of the prediction error of the resulting optimal kernel via consistency results for the group lasso.</details>

### FlowCam: Training Generalizable 3D Radiance Fields without Camera Poses via Pixel-Aligned Scene Flow

**Authors:** Cameron Smith, Yilun Du, Ayush Tewari, Vincent Sitzmann

**<details><summary>Abstract**</summary>

Reconstruction of 3D neural fields from posed images has emerged as a promising method for self-supervised representation learning. The key challenge preventing the deployment of these 3D scene learners on large-scale video data is their dependence on precise camera poses from structure-from-motion, which is prohibitively expensive to run at scale. We propose a method that jointly reconstructs camera poses and 3D neural scene representations online and in a single forward pass. We estimate poses by first lifting frame-to-frame optical flow to 3D scene flow via differentiable rendering, preserving locality and shift-equivariance of the image processing backbone. SE(3) camera pose estimation is then performed via a weighted least-squares fit to the scene flow field. This formulation enables us to jointly supervise pose estimation and a generalizable neural scene representation via re-rendering the input video, and thus, train end-to-end and fully self-supervised on real-world video datasets. We demonstrate that our method performs robustly on diverse, real-world video, notably on sequences traditionally challenging to optimization-based pose estimation techniques.</details>

### FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective

**Authors:** Kun Yi, Qi Zhang, Wei Fan, Hui He, Liang Hu, Pengyang Wang, Ning An, Longbing Cao, Zhendong Niu

**<details><summary>Abstract**</summary>

Multivariate time series (MTS) forecasting has shown great importance in numerous industries. Current state-of-the-art graph neural network (GNN)-based forecasting methods usually require both graph networks (e.g., GCN) and temporal networks (e.g., LSTM) to capture inter-series (spatial) dynamics and intra-series (temporal) dependencies, respectively. However, the uncertain compatibility of the two networks puts an extra burden on handcrafted model designs. Moreover, the separate spatial and temporal modeling naturally violates the unified spatiotemporal inter-dependencies in real world, which largely hinders the forecasting performance. To overcome these problems, we explore an interesting direction of directly applying graph networks and rethink MTS forecasting from a pure graph perspective. We first define a novel data structure, hypervariate graph, which regards each series value (regardless of variates or timestamps) as a graph node, and represents sliding windows as space-time fully-connected graphs. This perspective considers spatiotemporal dynamics unitedly and reformulates classic MTS forecasting into the predictions on hypervariate graphs. Then, we propose a novel architecture Fourier Graph Neural Network (FourierGNN) by stacking our proposed Fourier Graph Operator (FGO) to perform matrix multiplications in Fourier space. FourierGNN accommodates adequate expressiveness and achieves much lower complexity, which can effectively and efficiently accomplish {the forecasting}. Besides, our theoretical analysis reveals FGO's equivalence to graph convolutions in the time domain, which further verifies the validity of FourierGNN. Extensive experiments on seven datasets have demonstrated our superior performance with higher efficiency and fewer parameters compared with state-of-the-art methods. Code is available at this repository: https://github.com/aikunyi/FourierGNN.</details>

### FourierHandFlow: Neural 4D Hand Representation Using Fourier Query Flow

**Authors:** Jihyun Lee, Junbong Jang, Donghwan Kim, Minhyuk Sung, Tae-Kyun Kim

**<details><summary>Abstract**</summary>

Recent 4D shape representations model continuous temporal evolution of implicit shapes by (1) learning query flows without leveraging shape and articulation priors or (2) decoding shape occupancies separately for each time value. Thus, they do not effectively capture implicit correspondences between articulated shapes or regularize jittery temporal deformations. In this work, we present FourierHandFlow, which is a spatio-temporally continuous representation for human hands that combines a 3D occupancy field with articulation-aware query flows represented as Fourier series. Given an input RGB sequence, we aim to learn a fixed number of Fourier coefficients for each query flow to guarantee smooth and continuous temporal shape dynamics. To effectively model spatio-temporal deformations of articulated hands, we compose our 4D representation based on two types of Fourier query flow: (1) pose flow that models query dynamics influenced by hand articulation changes via implicit linear blend skinning and (2) shape flow that models query-wise displacement flow. In the experiments, our method achieves state-of-the-art results on video-based 4D reconstruction while being computationally more efficient than the existing 3D/4D implicit shape representations. We additionally show our results on motion inter- and extrapolation and texture transfer using the learned correspondences of implicit shapes. To the best of our knowledge, FourierHandFlow is the first neural 4D continuous hand representation learned from RGB videos. The code will be publicly accessible.</details>

### From Discrete Tokens to High-Fidelity Audio Using Multi-Band Diffusion

**Authors:** Robin San Roman, Yossi Adi, Antoine Deleforge, Romain Serizel, Gabriel Synnaeve, Alexandre Defossez

**<details><summary>Abstract**</summary>

Deep generative models can generate high-fidelity audio conditioned on varioustypes of representations (e.g., mel-spectrograms, Mel-frequency Cepstral Coefficients(MFCC)). Recently, such models have been used to synthesize audiowaveforms conditioned on highly compressed representations. Although suchmethods produce impressive results, they are prone to generate audible artifactswhen the conditioning is flawed or imperfect. An alternative modeling approach isto use diffusion models. However, these have mainly been used as speech vocoders(i.e., conditioned on mel-spectrograms) or generating relatively low samplingrate signals. In this work, we propose a high-fidelity multi-band diffusion-basedframework that generates any type of audio modality (e.g., speech, music, environmentalsounds) from low-bitrate discrete representations. At equal bit rate,the proposed approach outperforms state-of-the-art generative techniques in termsof perceptual quality. Training and evaluation code are available on the facebookresearch/audiocraft github project. Samples are available on the followinglink (https://ai.honu.io/papers/mbd/).</details>

### [Spotlight] Full-Atom Protein Pocket Design via Iterative Refinement

**Authors:** ZAIXI ZHANG, Zepu Lu, Hao Zhongkai, Marinka Zitnik, Qi Liu

**<details><summary>Abstract**</summary>

The design of \emph{de novo} functional proteins that bind with specific ligand molecules is crucial in various domains like therapeutics and bio-engineering. One vital yet challenging step is to design the protein pocket, the cavity region of protein where the ligand binds with. Existing methods suffer from inefficient generation, insufficient context modeling (ligand molecule), and incapability of generating sidechain atoms. To overcome the limitations, we propose a \textbf{F}ull-\textbf{A}tom \textbf{I}terative \textbf{R}efinement framework (\textbf{FAIR}) for protein pocket sequence (i.e., residue types) and 3D structure co-design. Generally, FAIR consists of two steps that follow a coarse-to-fine pipeline (backbone atoms to full atoms including sidechain) for full-atom generation. For efficiency, all residue types and structures are updated together in each round (i.e., full-shot refinement). In the first step, the residue types and backbone coordinates are updated with a hierarchical context encoder and two structure refinement modules capturing inter-residue and pocket-ligand interactions. The second step further models the sidechain atoms of pockets and updates residue types to achieve sequence-structure consistency. The structure of the binding ligand is also updated along with the above refinement iterations accounting for its flexibility. Finally, extensive evaluations showthat FAIR outperforms baselines in efficiently designing high-quality pocket sequences and structures. Specifically, the average improvements on AAR and RMSD are over 10$\%$.</details>

### Fully Dynamic $k$-Clustering in $\tilde O(k)$ Update Time

**Authors:** Sayan Bhattacharya, Martín Costa, Silvio Lattanzi, Nikos Parotsidis

**<details><summary>Abstract**</summary>

We present a $O(1)$-approximate fully dynamic algorithm for the $k$-median and $k$-means problems on metric spaces with amortized update time $\tilde O(k)$ and worst-case query time $\tilde O(k^2)$. We complement our theoretical analysis with the first in-depth experimental study for the dynamic $k$-median problem on general metrics, focusing on comparing our dynamic algorithm to the current state-of-the-art by Henzinger and Kale [ESA'20]. Finally, we also provide a lower bound for dynamic $k$-median which shows that any $O(1)$-approximate algorithm with $\tilde O(\text{poly}(k))$ query time must have $\tilde \Omega(k)$ amortized update time, even in the incremental setting.</details>

### Functional Equivalence and Path Connectivity of Reducible Hyperbolic Tangent Networks

**Authors:** Matthew Farrugia-Roberts

**<details><summary>Abstract**</summary>

Understanding the learning process of artificial neural networks requires clarifying the structure of the parameter space within which learning takes place. A neural network parameter's functional equivalence class is the set of parameters implementing the same input--output function. For many architectures, almost all parameters have a simple and well-documented functional equivalence class. However, there is also a vanishing minority of reducible parameters, with richer functional equivalence classes caused by redundancies among the network's units.In this paper, we give an algorithmic characterisation of unit redundancies and reducible functional equivalence classes for a single-hidden-layer hyperbolic tangent architecture. We show that such functional equivalence classes are piecewise-linear path-connected sets, and that for parameters with a majority of redundant units, the sets have a diameter of at most 7 linear segments.</details>

### GAN You See Me? Enhanced Data Reconstruction Attacks against Split Inference

**Authors:** Ziang Li, Mengda Yang, Yaxin Liu, Juan Wang, Hongxin Hu, Wenzhe Yi, Xiaoyang Xu

**<details><summary>Abstract**</summary>

Split Inference (SI) is an emerging deep learning paradigm that addresses computational constraints on edge devices and preserves data privacy through collaborative edge-cloud approaches. However, SI is vulnerable to Data Reconstruction Attacks (DRA), which aim to reconstruct users' private prediction instances. Existing attack methods suffer from various limitations. Optimization-based DRAs do not leverage public data effectively, while Learning-based DRAs depend heavily on auxiliary data quantity and distribution similarity. Consequently, these approaches yield unsatisfactory attack results and are sensitive to defense mechanisms. To overcome these challenges, we propose a GAN-based LAtent Space Search attack (GLASS) that harnesses abundant prior knowledge from public data using advanced StyleGAN technologies. Additionally, we introduce GLASS++ to enhance reconstruction stability. Our approach represents the first GAN-based DRA against SI, and extensive evaluation across different split points and adversary setups demonstrates its state-of-the-art performance. Moreover, we thoroughly examine seven defense mechanisms, highlighting our method's capability to reveal private information even in the presence of these defenses.</details>

### GEQ: Gaussian Kernel Inspired Equilibrium Models

**Authors:** Mingjie Li, Yisen Wang, Zhouchen Lin

**<details><summary>Abstract**</summary>

Despite the connection established by optimization-induced deep equilibrium models (OptEqs) between their output and the underlying hidden optimization problems, the performance of it along with its related works is still not good enough especially when compared to deep networks. One key factor responsible for this performance limitation is the use of linear kernels to extract features in these models. To address this issue, we propose a novel approach by replacing its linear kernel with a new function that can readily capture nonlinear feature dependencies in the input data. Drawing inspiration from classical machine learning algorithms, we introduce Gaussian kernels as the alternative function and then propose our new equilibrium model, which we refer to as GEQ. By leveraging Gaussian kernels, GEQ can effectively extract the nonlinear information embedded within the input features, surpassing the performance of the original OptEqs. Moreover, GEQ can be perceived as a weight-tied neural network with infinite width and depth. GEQ also enjoys better theoretical properties and improved overall performance. Additionally, our GEQ exhibits enhanced stability when confronted with various samples. We further substantiate the effectiveness and stability of GEQ through a series of comprehensive experiments.</details>

### GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning

**Authors:** Haiteng Zhao, Shengchao Liu, Ma Chang, Hannan Xu, Jie Fu, Zhihong Deng, Lingpeng Kong, Qi Liu

**<details><summary>Abstract**</summary>

Molecule property prediction has gained significant attention in recent years. The main bottleneck is the label insufficiency caused by expensive lab experiments. In order to alleviate this issue and to better leverage textual knowledge for tasks, this study investigates the feasibility of employing natural language instructions to accomplish molecule-related tasks in a zero-shot setting. We discover that existing molecule-text models perform poorly in this setting due to inadequate treatment of instructions and limited capacity for graphs.  To overcome these issues, we propose GIMLET, which unifies language models for both graph and text data. By adopting generalized position embedding, our model is extended to encode both graph structures and instruction text without additional graph encoding modules. GIMLET also decouples encoding of the graph from tasks instructions in the attention mechanism, enhancing the generalization of graph features across novel tasks. We construct a dataset consisting of more than two thousand molecule tasks with corresponding instructions derived from task descriptions. We pretrain GIMLET on the molecule tasks along with instructions, enabling the model to transfer effectively to a broad range of tasks. Experimental results demonstrate that GIMLET significantly outperforms molecule-text baselines in instruction-based zero-shot learning, even achieving closed results to supervised GNN models on tasks such as toxcast and muv.</details>

### GLOBER: Coherent Non-autoregressive Video Generation via GLOBal Guided Video DecodER

**Authors:** Mingzhen Sun, Weining Wang, Zihan Qin, Jiahui Sun, Sihan Chen, Jing Liu

**<details><summary>Abstract**</summary>

Video generation necessitates both global coherence and local realism. This work presents a novel non-autoregressive method GLOBER, which first generates global features to obtain comprehensive global guidance and then synthesizes video frames based on the global features to generate coherent videos. Specifically, we propose a video auto-encoder, where a video encoder encodes videos into global features, and a video decoder, built on a diffusion model, decodes the global features and synthesizes video frames in a non-autoregressive manner. To achieve maximum flexibility, our video decoder perceives temporal information through normalized frame indexes, which enables it to synthesize arbitrary sub video clips with predetermined starting and ending frame indexes. Moreover, a novel adversarial loss is introduced to improve the global coherence and local realism between the synthesized video frames. Finally, we employ a diffusion-based video generator to fit the global features outputted by the video encoder for video generation. Extensive experimental results demonstrate the effectiveness and efficiency of our proposed method, and new state-of-the-art results have been achieved on multiple benchmarks.</details>

### GMSF: Global Matching Scene Flow

**Authors:** Yushan Zhang, Johan Edstedt, Bastian Wandt, Per-Erik Forssen, Maria Magnusson, Michael Felsberg

**<details><summary>Abstract**</summary>

We tackle the task of scene flow estimation from point clouds. Given a source and a target point cloud, the objective is to estimate a translation from each point in the source point cloud to the target, resulting in a 3D motion vector field. Previous dominant scene flow estimation methods require complicated coarse-to-fine or recurrent architectures as a multi-stage refinement. In contrast, we propose a significantly simpler single-scale one-shot global matching to address the problem. Our key finding is that reliable feature similarity between point pairs is essential and sufficient to estimate accurate scene flow. We thus propose to decompose the feature extraction step via a hybrid local-global-cross transformer architecture which is crucial to accurate and robust feature representations. Extensive experiments show that the proposed Global Matching Scene Flow (GMSF) sets a new state-of-the-art on multiple scene flow estimation benchmarks. On FlyingThings3D, with the presence of occlusion points, GMSF reduces the outlier percentage from the previous best performance of 27.4% to 5.6%. On KITTI Scene Flow, without any fine-tuning, our proposed method shows state-of-the-art performance. On the Waymo-Open dataset, the proposed method outperforms previous methods by a large margin. The code is available at https://github.com/ZhangYushan3/GMSF.</details>

### GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction

**Authors:** Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, Ying Shan

**<details><summary>Abstract**</summary>

This paper aims to efficiently enable Large Language Models (LLMs) to use multi-modal tools.The advanced proprietary LLMs, such as ChatGPT and GPT-4, have shown great potential for tool usage through sophisticated prompt engineering.Nevertheless, these models typically rely on prohibitive computational costs and publicly inaccessible data.To address these challenges, we propose the GPT4Tools based on self-instruct to enable open-source LLMs, such as LLaMA and OPT, to use tools.It generates an instruction-following dataset by prompting an advanced teacher with various multi-modal contexts.By using the Low-Rank Adaptation (LoRA) optimization, our approach facilitates the open-source LLMs to solve a range of visual problems, including visual comprehension and image generation.Moreover, we provide a benchmark to evaluate the ability of LLMs to use tools, which is performed in both zero-shot and fine-tuning ways.Extensive experiments demonstrate the effectiveness of our method on various language models, which not only significantly improves the accuracy of invoking seen tools, but also enables the zero-shot capacity for unseen tools.</details>

### Gacs-Korner Common Information Variational Autoencoder

**Authors:** Michael Kleinman, Alessandro Achille, Stefano Soatto, Jonathan Kao

**<details><summary>Abstract**</summary>

We propose a notion of common information that allows one to quantify and separate the information that is shared between two random variables from the information that is unique to each. Our notion of common information is defined by an optimization problem over a family of functions and recovers the G\'acs-K\"orner common information as a special case. Importantly, our notion can be approximated empirically using samples from the underlying data distribution. We then provide a method to partition and quantify the common and unique information using a simple modification of a traditional variational auto-encoder. Empirically, we demonstrate that our formulation allows us to learn semantically meaningful common and unique factors of variation even on high-dimensional data such as images and videos. Moreover, on datasets where ground-truth latent factors are known, we show that we can accurately quantify the common information between the random variables.</details>

### Generalized Bayesian Inference for Scientific Simulators via Amortized Cost Estimation

**Authors:** Richard Gao, Michael Deistler, Jakob H Macke

**<details><summary>Abstract**</summary>

Simulation-based inference (SBI) enables amortized Bayesian inference for simulators with implicit likelihoods. But when we are primarily interested in the quality of predictive simulations, or when the model cannot exactly reproduce the observed data (i.e., is misspecified), targeting the Bayesian posterior may be overly restrictive. Generalized Bayesian Inference (GBI) aims to robustify inference for (misspecified) simulator models, replacing the likelihood-function with a cost function that evaluates the goodness of parameters relative to data. However, GBI methods generally require running multiple simulations to estimate the cost function at each parameter value during inference, making the approach computationally infeasible for even moderately complex simulators. Here, we propose amortized cost estimation (ACE) for GBI to address this challenge: We train a neural network to approximate the cost function, which we define as the expected distance between simulations produced by a parameter and observed data. The trained network can then be used with MCMC to infer GBI posteriors for any observation without running additional simulations. We show that, on several benchmark tasks, ACE accurately predicts cost and provides predictive simulations that are closer to synthetic observations than other SBI methods, especially for misspecified simulators. Finally, we apply ACE to infer parameters of the Hodgkin-Huxley model given real intracellular recordings from the Allen Cell Types Database. ACE identifies better data-matching parameters while being an order of magnitude more simulation-efficient than a standard SBI method. In summary, ACE combines the strengths of SBI methods and GBI to perform robust and simulation-amortized inference for scientific simulators.</details>

### [Oral] Generalizing Nonlinear ICA Beyond Structural Sparsity

**Authors:** Yujia Zheng, Kun Zhang

**Oral Presentation:** We, Dec 13, 14:00 -- Oral 4A

**<details><summary>Abstract**</summary>

Nonlinear independent component analysis (ICA) aims to uncover the true latent sources from their observable nonlinear mixtures. Despite its significance, the identifiability of nonlinear ICA is known to be impossible without additional assumptions. Recent advances have proposed conditions on the connective structure from sources to observed variables, known as Structural Sparsity, to achieve identifiability in an unsupervised manner. However, the sparsity constraint may not hold universally for all sources in practice. Furthermore, the assumptions of bijectivity of the mixing process and independence among all sources, which arise from the setting of ICA, may also be violated in many real-world scenarios. To address these limitations and generalize nonlinear ICA, we propose a set of new identifiability results in the general settings of undercompleteness, partial sparsity and source dependence, and flexible grouping structures. Specifically, we prove identifiability when there are more observed variables than sources (undercomplete), and when certain sparsity and/or source independence assumptions are not met for some changing sources. Moreover, we show that even in cases with flexible grouping structures (e.g., part of the sources can be divided into irreducible independent groups with various sizes), appropriate identifiability results can also be established. Theoretical claims are supported empirically on both synthetic and real-world datasets.</details>

### Graph Convolutional Kernel Machine versus Graph Convolutional Networks

**Authors:** Zhihao Wu, Zhao Zhang, Jicong Fan

**<details><summary>Abstract**</summary>

Graph convolutional networks (GCN) with one or two hidden layers have been widely used in handling graph data that are prevalent in various disciplines. Many studies showed that the gain of making GCNs deeper is tiny or even negative. This implies that the complexity of graph data is often limited and shallow models are often sufficient to extract expressive features for various tasks such as node classification. Therefore, in this work, we present a framework called graph convolutional kernel machine (GCKM) for graph-based machine learning. GCKMs are built upon kernel functions integrated with graph convolution. An example is the graph convolutional kernel support vector machine (GCKSVM) for node classification, for which we analyze the generalization error bound and discuss the impact of the graph structure. Compared to GCNs, GCKMs require much less effort in architecture design, hyperparameter tuning, and optimization. More importantly, GCKMs are guaranteed to obtain globally optimal solutions and have strong generalization ability and high interpretability. GCKMs are composable, can be extended to large-scale data, and are applicable to various tasks (e.g., node or graph classification, clustering, feature extraction, dimensionality reduction). The numerical results on benchmark datasets show that, besides the aforementioned advantages, GCKMs have at least competitive accuracy compared to GCNs.</details>

### Graph of Circuits with GNN for Exploring the Optimal Design Space

**Authors:** Aditya Shahane, Saripilli Swapna Manjiri, Ankesh Jain, Sandeep Kumar

**<details><summary>Abstract**</summary>

The design automation of analog circuits poses significant challenges in terms of the large design space, complex interdependencies between circuit specifications, and resource-intensive simulations. To address these challenges, this paper presents an innovative framework called the Graph of Circuits Explorer (GCX). Leveraging graph structure learning along with graph neural networks, GCX enables the creation of a surrogate model that facilitates efficient exploration of the optimal design space within a semi-supervised learning framework which reduces the need for large labelled datasets. The proposed approach comprises three key stages. First, we learn the geometric representation of circuits and enrich it with technology information to create a comprehensive feature vector. Subsequently, integrating feature-based graph learning with few-shot and zero-shot learning  enhances the generalizability in predictions for unseen circuits. Finally, we introduce two algorithms namely, EASCO and ASTROG which upon integration with GCX optimize the available samples to yield the optimal circuit configuration meeting the designer's criteria. The effectiveness of the proposed approach is demonstrated through simulated performance evaluation of various circuits, using derived parameters in 180nm CMOS technology. Furthermore, the generalizability of the approach is extended to higher-order topologies and different technology nodes such as 65nm and 45nm CMOS process nodes.</details>

### GraphAdapter: Tuning Vision-Language Models With Dual Knowledge Graph

**Authors:** Xin Li, Dongze Lian, Zhihe Lu, Jiawang Bai, Zhibo Chen, Xinchao Wang

**<details><summary>Abstract**</summary>

Adapter-style efficient transfer learning (ETL) has shown excellent performance in the tuning of vision-language models (VLMs) under the low-data regime, where only a few additional parameters are introduced to excavate the task-specific knowledge based on the general and powerful representation of VLMs. However, most adapter-style works face two limitations: (i) modeling task-specific knowledge with a single modality only; and (ii) overlooking the exploitation of the inter-class relationships in downstream tasks, thereby leading to sub-optimal solutions. To mitigate that, we propose an effective adapter-style tuning strategy, dubbed GraphAdapter, which performs the textual adapter by explicitly modeling the dual-modality structure knowledge (i.e., the correlation of different semantics/classes in textual and visual modalities) with a dual knowledge graph. In particular, the dual knowledge graph is established with two sub-graphs, i.e., a textual knowledge sub-graph, and a visual knowledge sub-graph, where the nodes and edges represent the semantics/classes and their correlations in two modalities, respectively. This enables the textual feature of each prompt to leverage the task-specific structure knowledge from both textual and visual modalities, yielding a more effective classifier for downstream tasks. Extensive experimental results on 11 benchmark datasets reveal that our GraphAdapter significantly outperforms the previous adapter-based methods.</details>

### GraphMP: Graph Neural Network-based Motion Planning with Efficient Graph Search

**Authors:** Xiao Zang, Miao Yin, Jinqi Xiao, Saman Zonouz, Bo Yuan

**<details><summary>Abstract**</summary>

Motion planning, which aims to find a high-quality collision-free path in the configuration space, is a fundamental task in robotic systems. Recently, learning-based motion planners, especially the graph neural network-powered, have shown promising planning performance. However, though the state-of-the-art GNN planner can efficiently extract and learn graph information, its inherent mechanism is not well suited for graph search process, hindering its further performance improvement. To address this challenge and fully unleash the potential of GNN in motion planning, this paper proposes GraphMP, a neural motion planner for both low and high-dimensional planning tasks. With the customized model architecture and training mechanism design, GraphMP can simultaneously perform efficient graph pattern extraction and graph search processing, leading to strong planning performance. Experiments on a variety of environments, ranging from 2D Maze to 14D dual KUKA robotic arm, show that our proposed GraphMP achieves significant improvement on path quality and planning speed over the state-of-the-art learning-based and classical planners; while preserving the competitive success rate.</details>

### Greedy Poisson Rejection Sampling

**Authors:** Gergely Flamich

**<details><summary>Abstract**</summary>

One-shot channel simulation is a fundamental data compression problem concerned with encoding a single sample from a target distribution $Q$ using a coding distribution $P$ using as few bits as possible on average. Algorithms that solve this problem find applications in neural data compression and differential privacy and can serve as a more efficient and natural alternative to quantization-based methods. Unfortunately, existing solutions are too slow or have limited applicability, preventing their widespread adaptation. In this paper, we conclusively solve one-shot channel simulation for one-dimensional problems where the target-proposal density ratio is unimodal by describing an algorithm with optimal runtime. We achieve this by constructing a rejection sampling procedure equivalent to greedily searching over the points of a Poisson process. Hence, we call our algorithm greedy Poisson rejection sampling (GPRS) and analyze the correctness and time complexity of several of its variants. Finally, we empirically verify our theorems, demonstrating that GPRS significantly outperforms the current state-of-the-art method, A* coding.</details>

### [Spotlight] Group Fairness in Peer Review

**Authors:** Haris Aziz, Evi Micha, Nisarg Shah

**<details><summary>Abstract**</summary>

Large conferences such as NeurIPS and AAAI serve as crossroads  of various AI fields, since they attract submissions from a vast number of communities. However, in some cases, this has resulted in a poor reviewing experience for some communities, whose submissions get assigned to less qualified reviewers outside of their communities. An often-advocated solution is to break up any such large conference into smaller conferences, but this can lead to isolation of communities and harm interdisciplinary research. We tackle this challenge by introducing a  notion of group fairness, called the core, which requires that every possible community (subset of researchers) to be treated in a way that prevents them from unilaterally benefiting by  withdrawing from a large conference. We study a simple peer review model, prove that it always admits a reviewing assignment in the core, and design an efficient algorithm to find one such assignment. We use real data from CVPR and ICLR conferences to compare our algorithm to existing reviewing assignment algorithms on a number of metrics.</details>

### Guarantees for Self-Play in Multiplayer Games via Polymatrix Decomposability

**Authors:** Revan MacQueen, James Wright

**<details><summary>Abstract**</summary>

Self-play is a technique for machine learning in multi-agent systems where a learning algorithm learns by interacting with copies of itself. Self-play is useful for generating large quantities of data for learning, but has the drawback that the agents the learner will face post-training may have dramatically different behavior than the learner came to expect by interacting with itself. For the special case of two-player constant-sum games, self-play that reaches Nash equilibrium is guaranteed to produce strategies that perform well against any post-training opponent; however, no such guarantee exists for multiplayer games. We show that in games that approximately decompose into a set of two-player constant-sum games (called constant-sum polymatrix games) where global $\epsilon$-Nash equilibria are boundedly far from Nash equilibria in each subgame (called subgame stability), any no-external-regret algorithm that learns by self-play will produce a strategy with bounded vulnerability. For the first time, our results identify a structural property of multiplayer games that enable performance guarantees for the strategies produced by a broad class of self-play algorithms. We demonstrate our findings through experiments on Leduc poker.</details>

### Guiding Large Language Models via Directional Stimulus Prompting

**Authors:** Zekun Li, Baolin Peng, Pengcheng He, Michel Galley, Jianfeng Gao, Xifeng Yan

**<details><summary>Abstract**</summary>

We introduce Directional Stimulus Prompting, a novel framework for guiding black-box large language models (LLMs) towards specific desired outputs. Instead of directly adjusting LLMs, our method employs a small tunable policy model (e.g., T5) to generate an auxiliary directional stimulus prompt for each input instance. These directional stimulus prompts act as nuanced, instance-specific hints and clues to guide LLMs in generating desired outcomes, such as including specific keywords in the generated summary. Our approach sidesteps the challenges of direct LLM tuning by optimizing the policy model to explore directional stimulus prompts that align LLMs with desired behaviors. The policy model can be optimized through 1) supervised fine-tuning using labeled data and 2) reinforcement learning from offline or online rewards based on the LLM's output. We evaluate our method across various tasks, including summarization, dialogue response generation, and chain-of-thought reasoning. Our experiments indicate a consistent improvement in the performance of LLMs such as ChatGPT, Codex, and InstructGPT on these supervised tasks with minimal labeled data. Remarkably, by utilizing merely 80 dialogues from the MultiWOZ dataset, our approach boosts ChatGPT's performance by a relative 41.4%, achieving or exceeding the performance of some fully supervised state-of-the-art models. Moreover, the instance-specific chain-of-thought prompt generated through our method enhances InstructGPT's reasoning accuracy, outperforming both generalized human-crafted prompts and those generated through automatic prompt engineering. The code and data are publicly available at https://github.com/Leezekun/Directional-Stimulus-Prompting.</details>

### [Spotlight] HIQL: Offline Goal-Conditioned RL with Latent States as Actions

**Authors:** Seohong Park, Dibya Ghosh, Benjamin Eysenbach, Sergey Levine

**<details><summary>Abstract**</summary>

Unsupervised pre-training has recently become the bedrock for computer vision and natural language processing. In reinforcement learning (RL), goal-conditioned RL can potentially provide an analogous self-supervised approach for making use of large quantities of unlabeled (reward-free) data. However, building effective algorithms for goal-conditioned RL that can learn directly from diverse offline data is challenging, because it is hard to accurately estimate the exact value function for faraway goals. Nonetheless, goal-reaching problems exhibit structure, such that reaching distant goals entails first passing through closer subgoals. This structure can be very useful, as assessing the quality of actions for nearby goals is typically easier than for more distant goals. Based on this idea, we propose a hierarchical algorithm for goal-conditioned RL from offline data. Using one action-free value function, we learn two policies that allow us to exploit this structure: a high-level policy that treats states as actions and predicts (a latent representation of) a subgoal and a low-level policy that predicts the action for reaching this subgoal. Through analysis and didactic examples, we show how this hierarchical decomposition makes our method robust to noise in the estimated value function. We then apply our method to offline goal-reaching benchmarks, showing that our method can solve long-horizon tasks that stymie prior methods, can scale to high-dimensional image observations, and can readily make use of action-free data. Our code is available at https://seohong.me/projects/hiql/</details>

### Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery

**Authors:** Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, Tom Goldstein

**<details><summary>Abstract**</summary>

The strength of modern generative models lies in their ability to be controlled through prompts. Hard prompts comprise interpretable words and tokens, and are typically hand-crafted by humans.  Soft prompts, on the other hand, consist of continuous feature vectors.  These can be discovered using powerful optimization methods, but they cannot be easily edited, re-used across models, or plugged into a text-based interface. We describe an easy-to-use approach to automatically optimize hard text prompts through efficient gradient-based optimization. Our approach can be readily applied to text-to-image and text-only applications alike. This method allows API users to easily generate, discover, and mix and match image concepts without prior knowledge of how to prompt the model. Furthermore, using our method, we can bypass token-level content filters imposed by Midjourney by optimizing through the open-sourced text encoder.</details>

### Have it your way: Individualized Privacy Assignment for DP-SGD

**Authors:** Franziska Boenisch, Christopher Mühl, Adam Dziedzic, Roy Rinberg, Nicolas Papernot

**<details><summary>Abstract**</summary>

When training a machine learning model with differential privacy, one sets a privacy budget. This uniform budget represents an overall maximal privacy violation that any user is willing to face by contributing their data to the training set. We argue that this approach is limited because different users may have different privacy expectations. Thus, setting a uniform privacy budget across all points may be overly conservative for some users or, conversely, not sufficiently protective for others. In this paper, we capture these preferences through individualized privacy budgets. To demonstrate their practicality, we introduce a variant of Differentially Private Stochastic Gradient Descent (DP-SGD) which supports such individualized budgets. DP-SGD is the canonical approach to training models with differential privacy. We modify its data sampling and gradient noising mechanisms to arrive at our approach, which we call Individualized DP-SGD (IDP-SGD). Because IDP-SGD provides privacy guarantees tailored to the preferences of individual users and their data points, we empirically find it to improve privacy-utility trade-offs.</details>

### [Spotlight] Hierarchical Integration Diffusion Model for Realistic Image Deblurring

**Authors:** Zheng Chen, Yulun Zhang, Ding Liu, bin xia, Jinjin Gu, Linghe Kong, Xin Yuan

**<details><summary>Abstract**</summary>

Diffusion models (DMs) have recently been introduced in image deblurring and exhibited promising performance, particularly in terms of details reconstruction. However, the diffusion model requires a large number of inference iterations to recover the clean image from pure Gaussian noise, which consumes massive computational resources. Moreover, the distribution synthesized by the diffusion model is often misaligned with the target results, leading to restrictions in distortion-based metrics. To address the above issues, we propose the Hierarchical Integration Diffusion Model (HI-Diff), for realistic image deblurring. Specifically, we perform the DM in a highly compacted latent space to generate the prior feature for the deblurring process. The deblurring process is implemented by a regression-based method to obtain better distortion accuracy. Meanwhile, the highly compact latent space ensures the efficiency of the DM. Furthermore, we design the hierarchical integration module to fuse the prior into the regression-based model from multiple scales, enabling better generalization in complex blurry scenarios. Comprehensive experiments on synthetic and real-world blur datasets demonstrate that our HI-Diff outperforms state-of-the-art methods. Code and trained models are available at https://github.com/zhengchen1999/HI-Diff.</details>

### Hierarchical Vector Quantized Transformer for Multi-class Unsupervised Anomaly Detection

**Authors:** Ruiying Lu, YuJie Wu, Long Tian, Dongsheng Wang, Bo Chen, Xiyang Liu, Ruimin Hu

**<details><summary>Abstract**</summary>

Unsupervised image Anomaly Detection (UAD) aims to learn robust and discriminative representations of normal samples. While separate solutions per class endow expensive computation and limited generalizability, this paper focuses on building a unified framework for multiple classes. Under such a challenging setting, popular reconstruction-based networks with continuous latent representation assumption always suffer from the "identical shortcut" issue, where both normal and abnormal samples can be well recovered and difficult to distinguish. To address this pivotal issue, we propose a hierarchical vector quantized prototype-oriented Transformer under a probabilistic framework. First, instead of learning the continuous representations, we preserve the typical normal patterns as discrete iconic prototypes, and confirm the importance of Vector Quantization in preventing the model from falling into the shortcut. The vector quantized iconic prototypes are integrated into the Transformer for reconstruction, such that the abnormal data point is flipped to a normal data point. Second, we investigate an exquisite hierarchical framework to relieve the codebook collapse issue and replenish frail normal patterns. Third, a prototype-oriented optimal transport method is proposed to better regulate the prototypes and hierarchically evaluate the abnormal score. By evaluating on MVTec-AD and VisA datasets, our model surpasses the state-of-the-art alternatives and possesses good interpretability. The code is available at https://github.com/RuiyingLu/HVQ-Trans.</details>

### Horospherical Decision Boundaries for Large Margin Classification in Hyperbolic Space

**Authors:** Xiran Fan, Chun-Hao Yang, Baba Vemuri

**<details><summary>Abstract**</summary>

Hyperbolic spaces have been quite popular in the recent past for representing hierarchically organized data. Further, several classification algorithms for data in these spaces have been proposed in the literature. These algorithms mainly use either hyperplanes or geodesics for decision boundaries in a large margin classifiers setting leading to a non-convex optimization problem. In this paper, we propose a novel large margin classifier based on horospherical decision boundaries that leads to a geodesically convex optimization problem that can be optimized using any Riemannian gradient descent technique guaranteeing a globally optimal solution. We present several experiments depicting the competitive performance of our classifier in comparison to SOTA.</details>

### Hybrid Search for Efficient Planning with Completeness Guarantees

**Authors:** Kalle Kujanpää, Joni Pajarinen, Alexander Ilin

**<details><summary>Abstract**</summary>

Solving complex planning problems has been a long-standing challenge in computer science. Learning-based subgoal search methods have shown promise in tackling these problems, but they often suffer from a lack of completeness guarantees, meaning that they may fail to find a solution even if one exists. In this paper, we propose an efficient approach to augment a subgoal search method to achieve completeness in discrete action spaces. Specifically, we augment the high-level search with low-level actions to execute a multi-level (hybrid) search, which we call complete subgoal search. This solution achieves the best of both worlds: the practical efficiency of high-level search and the completeness of low-level search. We apply the proposed search method to a recently proposed subgoal search algorithm and evaluate the algorithm trained on offline data on complex planning problems. We demonstrate that our complete subgoal search not only guarantees completeness but can even improve performance in terms of search expansions for instances that the high-level could solve without low-level augmentations. Our approach makes it possible to apply subgoal-level planning for systems where completeness is a critical requirement.</details>

### Hyperbolic Graph Neural Networks at Scale: A Meta Learning Approach

**Authors:** Nurendra Choudhary, Nikhil Rao, Chandan Reddy

**<details><summary>Abstract**</summary>

The progress in hyperbolic neural networks (HNNs) research is hindered by their absence of inductive bias mechanisms, which are essential for generalizing to new tasks and facilitating scalable learning over large datasets. In this paper, we aim to alleviate these issues by learning generalizable inductive biases from the nodes’ local subgraph and transfer them for faster learning over new subgraphs with a disjoint set of nodes, edges, and labels in a few-shot setting. We introduce a novel method, Hyperbolic GRAph Meta Learner (H-GRAM), that, for the tasks of node classification and link prediction, learns transferable information from a set of support local subgraphs in the form of hyperbolic meta gradients and label hyperbolic protonets to enable faster learning over a query set of new tasks dealing with disjoint subgraphs. Furthermore, we show that an extension of our meta-learning framework also mitigates the scalability challenges seen in HNNs faced by existing approaches. Our comparative analysis shows that H-GRAM effectively learns and transfers information in multiple challenging few-shot settings compared to other state-of-the-art baselines. Additionally, we demonstrate that, unlike standard HNNs, our approach is able to scale over large graph datasets and improve performance over its Euclidean counterparts.</details>

### Hyperbolic Space with Hierarchical Margin Boosts Fine-Grained Learning from Coarse Labels

**Authors:** Shu-Lin Xu, Yifan Sun, Faen Zhang, Anqi Xu, Xiu-Shen Wei, Yi Yang

**<details><summary>Abstract**</summary>

Learning fine-grained embeddings from coarse labels is a challenging task due to limited label granularity supervision, i.e., lacking the detailed distinctions required for fine-grained tasks. The task becomes even more demanding when attempting few-shot fine-grained recognition, which holds practical significance in various applications. To address these challenges, we propose a novel method that embeds visual embeddings into a hyperbolic space and enhances their discriminative ability with a hierarchical cosine margins manner. Specifically, the hyperbolic space offers distinct advantages, including the ability to capture hierarchical relationships and increased expressive power, which favors modeling fine-grained objects. Based on the hyperbolic space, we further enforce relatively large/small similarity margins between coarse/fine classes, respectively, yielding the so-called hierarchical cosine margins manner. While enforcing similarity margins in the regular Euclidean space has become popular for deep embedding learning, applying it to the hyperbolic space is non-trivial and validating the benefit for coarse-to-fine generalization is valuable. Extensive experiments conducted on five benchmark datasets showcase the effectiveness of our proposed method, yielding state-of-the-art results surpassing competing methods.</details>

### [Spotlight] Hypernetwork-based Meta-Learning for Low-Rank Physics-Informed Neural Networks

**Authors:** Woojin Cho, Kookjin Lee, Donsub Rim, Noseong Park

**<details><summary>Abstract**</summary>

In various engineering and applied science applications, repetitive numerical simulations of partial differential equations (PDEs) for varying input parameters are often required (e.g., aircraft shape optimization over many design parameters) and solvers are required to perform rapid execution. In this study, we suggest a path that potentially opens up a possibility for physics-informed neural networks (PINNs), emerging deep-learning-based solvers, to be considered as one such solver. Although PINNs have pioneered a proper integration of deep-learning and scientific computing, they require repetitive time-consuming training of neural networks, which is not suitable for many-query scenarios. To address this issue, we propose a lightweight low-rank PINNs containing only hundreds of model parameters and an associated hypernetwork-based meta-learning algorithm, which allows efficient approximation of solutions of PDEs for varying ranges of PDE input parameters. Moreover, we show that the proposed method is effective in overcoming a challenging issue, known as "failure modes" of PINNs.</details>

### IBA: Towards Irreversible Backdoor Attacks in Federated Learning

**Authors:** Thuy Dung Nguyen, Tuan Nguyen, Anh Tran, Khoa D Doan, Kok-Seng Wong

**<details><summary>Abstract**</summary>

Federated learning (FL) is a distributed learning approach that enables machine learning models to be trained on decentralized data without compromising end devices' personal, potentially sensitive data. However, the distributed nature and uninvestigated data intuitively introduce new security vulnerabilities, including backdoor attacks. In this scenario, an adversary implants backdoor functionality into the global model during training, which can be activated to cause the desired misbehaviors for any input with a specific adversarial pattern.Despite having remarkable success in triggering and distorting model behavior, prior backdoor attacks in FL often hold impractical assumptions, limited imperceptibility, and durability. Specifically, the adversary needs to control a sufficiently large fraction of clients or know the data distribution of other honest clients. In many cases, the trigger inserted is often visually apparent, and the backdoor effect is quickly diluted if the adversary is removed from the training process. To address these limitations, we propose a novel backdoor attack framework in FL, \dung{the 	extbf{Irreversible Backdoor Attack} (\texttt{IBA})}, that jointly learns the optimal and visually stealthy trigger and then gradually implants the backdoor into a global model. This approach allows the adversary to execute a backdoor attack that can evade both human and machine inspections. Additionally, we enhance the efficiency and durability of the proposed attack by selectively poisoning the model's parameters that are least likely updated by the main task's learning process and constraining the poisoned model update to the vicinity of the global model.Finally, we evaluate the proposed attack framework on several benchmark datasets, including MNIST, CIFAR-10, and Tiny ImageNet, and achieved high success rates while simultaneously bypassing existing backdoor defenses and achieving a more durable backdoor effect compared to other backdoor attacks. Overall, \texttt{IBA}\footnote{Code for this paper is published at \url{https://github.com/sail-research/iba}.} offers a more effective, stealthy, and durable approach to backdoor attacks in FL.</details>

### IMPRESS: Evaluating the Resilience of Imperceptible Perturbations Against Unauthorized Data Usage in Diffusion-Based  Generative AI

**Authors:** Bochuan Cao, Changjiang Li, Ting Wang, Jinyuan Jia, Bo Li, Jinghui Chen

**<details><summary>Abstract**</summary>

Diffusion-based image generation models,  such as Stable Diffusion or DALL·E 2, are able to learn from given images and generate high-quality samples following the guidance from prompts. For instance, they can be used to create artistic images that mimic the style of an artist based on his/her original artworks or to maliciously edit the original images for fake content. However, such ability also brings serious ethical issues without proper authorization from the owner of the original images. In response, several attempts have been made to protect the original images from such unauthorized data usage by adding imperceptible perturbations, which are designed to mislead the diffusion model and make it unable to properly generate new samples. In this work, we introduce a perturbation purification platform, named IMPRESS, to evaluate the effectiveness of imperceptible perturbations as a protective measure.IMPRESS is based on the key observation that imperceptible perturbations could lead to a perceptible inconsistency between the original image and the diffusion-reconstructed image, which can be used to devise a new optimization strategy for purifying the image, which may weaken the protection of the original image from unauthorized data usage (e.g., style mimicking, malicious editing).The proposed IMPRESS platform offers a comprehensive evaluation of several contemporary protection methods, and can be used as an evaluation platform for future protection methods.</details>

### Identification of Nonlinear Latent Hierarchical Models

**Authors:** Lingjing Kong, Biwei Huang, Feng Xie, Eric Xing, Yuejie Chi, Kun Zhang

**<details><summary>Abstract**</summary>

Identifying latent variables and causal structures from observational data is essential to many real-world applications involving biological data, medical data, and unstructured data such as images and languages. However, this task can be highly challenging, especially when observed variables are generated by causally related latent variables and the relationships are nonlinear. In this work, we investigate the identification problem for nonlinear latent hierarchical causal models in which observed variables are generated by a set of causally related latent variables, and some latent variables may not have observed children. We show that the identifiability of causal structures and latent variables (up to invertible transformations) can be achieved under mild assumptions: on causal structures, we allow for multiple paths between any pair of variables in the graph, which relaxes latent tree assumptions in prior work; on structural functions, we permit general nonlinearity and multi-dimensional continuous variables, alleviating existing work's parametric assumptions. Specifically, we first develop an identification criterion in the form of novel identifiability guarantees for an elementary latent variable model. Leveraging this criterion, we show that both causal structures and latent variables of the hierarchical model can be identified asymptotically by explicitly constructing an estimation procedure. To the best of our knowledge, our work is the first to establish identifiability guarantees for both causal structures and latent variables in nonlinear latent hierarchical models.</details>

### Implicit Manifold Gaussian Process Regression

**Authors:** Bernardo Fichera, Slava Borovitskiy, Andreas Krause, Aude G Billard

**<details><summary>Abstract**</summary>

Gaussian process regression is widely used because of its ability to provide well-calibrated uncertainty estimates and handle small or sparse datasets. However, it struggles with high-dimensional data. One possible way to scale this technique to higher dimensions is to leverage the implicit low-dimensional manifold upon which the data actually lies, as postulated by the manifold hypothesis. Prior work ordinarily requires the manifold structure to be explicitly provided though, i.e. given by a mesh or be known to be one of the well-known manifolds like the sphere. In contrast, in this paper we propose a Gaussian process regression technique capable of inferring implicit structure directly from data (labeled and unlabeled) in a fully differentiable way. For the resulting model, we discuss its convergence to the Matérn Gaussian process on the assumed manifold. Our technique scales up to hundreds of thousands of data points, and may improve the predictive performance and calibration of the standard Gaussian process regression in high-dimensional settings.</details>

### Implicit Regularization in Over-Parameterized Support Vector Machine

**Authors:** Yang Sui, Xin HE, Yang Bai

**<details><summary>Abstract**</summary>

In this paper, we design a regularization-free algorithm for high-dimensional support vector machines (SVMs) by integrating over-parameterization with Nesterov's smoothing method, and provide theoretical guarantees for the induced implicit regularization phenomenon. In particular, we construct an over-parameterized hinge loss function and estimate the true parameters by leveraging regularization-free gradient descent on this loss function. The utilization of Nesterov's method enhances the computational efficiency of our algorithm, especially in terms of determining the stopping criterion and reducing computational complexity. With appropriate choices of initialization, step size, and smoothness parameter, we demonstrate that unregularized gradient descent achieves a near-oracle statistical convergence rate. Additionally, we verify our theoretical findings through a variety of numerical experiments and compare the proposed method with explicit regularization. Our results illustrate the advantages of employing implicit regularization via gradient descent in conjunction with over-parameterization in sparse SVMs.</details>

### [Spotlight] Implicit Variational Inference for High-Dimensional Posteriors

**Authors:** Anshuk Uppal, Kristoffer Stensbo-Smidt, Wouter Boomsma, Jes Frellsen

**<details><summary>Abstract**</summary>

In variational inference, the benefits of Bayesian models rely on accurately capturing the true posterior distribution. We propose using neural samplers that specify implicit distributions, which are well-suited for approximating complex multimodal and correlated posteriors in high-dimensional spaces. Our approach introduces novel bounds for approximate inference using implicit distributions by locally linearising the neural sampler. This is distinct from existing methods that rely on additional discriminator networks and unstable adversarial objectives. Furthermore, we present a new sampler architecture that, for the first time, enables implicit distributions over tens of millions of latent variables, addressing computational concerns by using differentiable numerical approximations. We empirically show that our method is capable of recovering correlations across layers in large Bayesian neural networks, a property that is crucial for a network's performance but notoriously challenging to achieve. To the best of our knowledge, no other method has been shown to accomplish this task for such large models. Through experiments in downstream tasks, we demonstrate that our expressive posteriors outperform state-of-the-art uncertainty quantification methods, validating the effectiveness of our training algorithm and the quality of the learned implicit approximation.</details>

### Improving Adversarial Transferability via Intermediate-level Perturbation Decay

**Authors:** Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**<details><summary>Abstract**</summary>

Intermediate-level attacks that attempt to perturb feature representations following an adversarial direction drastically have shown favorable performance in crafting transferable adversarial examples. Existing methods in this category are normally formulated with two separate stages, where a directional guide is required to be determined at first and the scalar projection of the intermediate-level perturbation onto the directional guide is enlarged thereafter. The obtained perturbation deviates from the guide inevitably in the feature space, and it is revealed in this paper that such a deviation may lead to sub-optimal attack. To address this issue, we develop a novel intermediate-level method that crafts adversarial examples within a single stage of optimization. In particular, the proposed method, named intermediate-level perturbation decay (ILPD), encourages the intermediate-level perturbation to be in an effective adversarial direction and to possess a great magnitude simultaneously. In-depth discussion verifies the effectiveness of our method. Experimental results show that it outperforms state-of-the-arts by large margins in attacking various victim models on ImageNet (+10.07% on average) and CIFAR-10 (+3.88% on average). Our code is at https://github.com/qizhangli/ILPD-attack.</details>

### Improving CLIP Training with Language Rewrites

**Authors:** Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, Yonglong Tian

**<details><summary>Abstract**</summary>

Contrastive Language-Image Pre-training (CLIP) stands as one of the most effective and scalable methods for training transferable vision models using paired image and text data. CLIP models are trained using contrastive loss, which typically relies on data augmentations to prevent overfitting and shortcuts. However, in the CLIP training paradigm, data augmentations are exclusively applied to image inputs, while language inputs remain unchanged throughout the entire training process, limiting the exposure of diverse texts to the same image. In this paper, we introduce Language augmented CLIP (LaCLIP), a simple yet highly effective approach to enhance CLIP training through language rewrites. Leveraging the in-context learning capability of large language models, we rewrite the text descriptions associated with each image. These rewritten texts exhibit diversity in sentence structure and vocabulary while preserving the original key concepts and meanings. During training, LaCLIP randomly selects either the original texts or the rewritten versions as text augmentations for each image. Extensive experiments on CC3M, CC12M, RedCaps and LAION-400M datasets show that CLIP pre-training with language rewrites significantly improves the transfer performance without computation or memory overhead during training. Specifically for ImageNet zero-shot accuracy, LaCLIP outperforms CLIP by 8.2% on CC12M and 2.4% on LAION-400M.</details>

### Improving Robustness with Adaptive Weight Decay

**Authors:** Mohammad Amin Ghiasi, Ali Shafahi, Reza Ardekani

**<details><summary>Abstract**</summary>

We propose adaptive weight decay, which automatically tunes the hyper-parameter for weight decay during each training iteration. For classification problems, we propose changing the value of the weight decay hyper-parameter on the fly based on the strength of updates from the classification loss (i.e., gradient of cross-entropy), and the regularization loss (i.e., $\ell_2$-norm of the weights). We show that this simple modification can result in large improvements in adversarial robustness — an area which suffers from robust overfitting — without requiring extra data accros various datasets and architecture choices. For example, our reformulation results in 20\% relative robustness improvement for CIFAR-100, and 10\% relative robustness improvement on CIFAR-10 comparing to the best tuned hyper-parameters of traditional weight decay resulting in models that have comparable performance to SOTA robustness methods. In addition, this method has other desirable properties, such as less sensitivity to learning rate, and smaller weight norms, which the latter contributes to robustness to overfitting to label noise, and pruning.</details>

### Incentives in Private Collaborative Machine Learning

**Authors:** Rachael Sim, Yehong Zhang, Nghia Hoang, Xinyi Xu, Bryan Kian Hsiang Low, Patrick Jaillet

**<details><summary>Abstract**</summary>

Collaborative machine learning involves training models on data from multiple parties but must incentivize their participation. Existing data valuation methods fairly value and reward each party based on  shared data or model parameters but neglect the privacy risks involved. To address this, we introduce _differential privacy_ (DP) as an incentive. Each party can select its required DP guarantee and perturb its _sufficient statistic_ (SS) accordingly. The mediator values the perturbed SS by the Bayesian surprise it elicits about the model parameters. As our valuation function enforces a _privacy-valuation trade-off_, parties are deterred from selecting excessive DP guarantees that reduce the utility of the grand coalition's model. Finally, the mediator rewards each party with different posterior samples of the model parameters. Such rewards still satisfy existing incentives like fairness but additionally preserve DP and a high similarity to the grand coalition's posterior. We empirically demonstrate the effectiveness and practicality of our approach on synthetic and real-world datasets.</details>

### [Spotlight] Inference-Time Intervention: Eliciting Truthful Answers from a Language Model

**Authors:** Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg

**<details><summary>Abstract**</summary>

We introduce Inference-Time Intervention (ITI), a technique designed to enhance the "truthfulness" of large language models (LLMs). ITI operates by shifting model activations during inference, following a learned set of directions across a limited number of attention heads. This intervention significantly improves the performance of LLaMA models on the TruthfulQA benchmark. On an instruction-finetuned LLaMA called Alpaca, ITI improves its truthfulness from $32.5\%$ to $65.1\%$. We identify a tradeoff between truthfulness and helpfulness and demonstrate how to balance it by tuning the intervention strength. ITI is minimally invasive and computationally inexpensive. Moreover, the technique is data efficient: while approaches like RLHF require extensive annotations, ITI locates truthful directions using only few hundred examples. Our findings suggest that LLMs may have an internal representation of the likelihood of something being true, even as they produce falsehoods on the surface.</details>

### Injecting Multimodal Information into Rigid Protein Docking via Bi-level Optimization

**Authors:** Ruijia Wang, YiWu Sun, Yujie Luo, Shaochuan Li, Cheng Yang, Xingyi Cheng, Hui Li, Chuan Shi, Le Song

**<details><summary>Abstract**</summary>

The structure of protein-protein complexes is critical for understanding binding dynamics, biological mechanisms, and intervention strategies. Rigid protein docking, a fundamental problem in this field, aims to predict the 3D structure of complexes from their unbound states without conformational changes. In this scenario, we have access to two types of valuable information: sequence-modal information, such as coevolutionary data obtained from multiple sequence alignments, and structure-modal information, including the 3D conformations of rigid structures. However, existing docking methods typically utilize single-modal information, resulting in suboptimal predictions. In this paper, we propose xTrimoBiDock (or BiDock for short), a novel rigid docking model that effectively integrates sequence- and structure-modal information through bi-level optimization. Specifically, a cross-modal transformer combines multimodal information to predict an inter-protein distance map. To achieve rigid docking, the roto-translation transformation is optimized to align the docked pose with the predicted distance map. In order to tackle this bi-level optimization problem, we unroll the gradient descent of the inner loop and further derive a better initialization for roto-translation transformation based on spectral estimation. Compared to baselines, BiDock achieves a promising result of a maximum 234% relative improvement in challenging antibody-antigen docking problem.</details>

### Inner-Outer Aware Reconstruction Model for Monocular 3D Scene Reconstruction

**Authors:** Yu-Kun Qiu, Guo-Hao Xu, Wei-Shi Zheng

**<details><summary>Abstract**</summary>

Monocular 3D scene reconstruction aims to reconstruct the 3D structure of scenes based on posed images. Recent volumetric-based methods directly predict the truncated signed distance function (TSDF) volume and have achieved promising results. The memory cost of volumetric-based methods will grow cubically as the volume size increases, so a coarse-to-fine strategy is necessary for saving memory. Specifically, the coarse-to-fine strategy distinguishes surface voxels from non-surface voxels, and only potential surface voxels are considered in the succeeding procedure. However, the non-surface voxels have various features, and in particular, the voxels on the inner side of the surface are quite different from those on the outer side since there exists an intrinsic gap between them. Therefore, grouping inner-surface and outer-surface voxels into the same class will force the classifier to spend its capacity to bridge the gap. By contrast, it is relatively easy for the classifier to distinguish inner-surface and outer-surface voxels due to the intrinsic gap. Inspired by this, we propose the inner-outer aware reconstruction (IOAR) model. IOAR explores a new coarse-to-fine strategy to classify outer-surface, inner-surface and surface voxels. In addition, IOAR separates occupancy branches from TSDF branches to avoid mutual interference between them. Since our model can better classify the surface, outer-surface and inner-surface voxels, it can predict more precise meshes than existing methods. Experiment results on ScanNet, ICL-NUIM and TUM-RGBD datasets demonstrate the effectiveness and generalization of our model. The code is available at https://github.com/YorkQiu/InnerOuterAwareReconstruction.</details>

### Integration-free Training for Spatio-temporal Multimodal Covariate Deep Kernel Point Processes

**Authors:** YIXUAN ZHANG, Quyu Kong, Feng Zhou

**<details><summary>Abstract**</summary>

In this study, we propose a novel deep spatio-temporal point process model, Deep Kernel Mixture Point Processes (DKMPP), that incorporates multimodal covariate information. DKMPP is an enhanced version of Deep Mixture Point Processes (DMPP), which uses a more flexible deep kernel to model complex relationships between events and covariate data, improving the model's expressiveness. To address the intractable training procedure of DKMPP due to the non-integrable deep kernel, we utilize an integration-free method based on score matching, and further improve efficiency by adopting a scalable denoising score matching method. Our experiments demonstrate that DKMPP and its corresponding score-based estimators outperform baseline models, showcasing the advantages of incorporating covariate information, utilizing a deep kernel, and employing score-based estimators.</details>

### Interactive Multi-fidelity Learning for Cost-effective Adaptation of Language Model with Sparse Human Supervision

**Authors:** Jiaxin Zhang, Zhuohang Li, Kamalika Das, Sricharan Kumar

**<details><summary>Abstract**</summary>

Large language models (LLMs) have demonstrated remarkable capabilities in various tasks. However, their suitability for domain-specific tasks, is limited due to their immense scale at deployment, susceptibility to misinformation, and more importantly, high data annotation costs. We propose a novel Interactive Multi-Fidelity Learning (IMFL) framework for cost-effective development of small domain-specific LMs under limited annotation budgets. Our approach formulates the domain-specific fine-tuning process as a multi-fidelity learning problem, focusing on identifying the optimal acquisition strategy that balances between low-fidelity automatic LLM annotations and high-fidelity human annotations to maximize model performance. We further propose an exploration-exploitation query strategy that enhances annotation diversity and informativeness, incorporating two innovative designs: 1) prompt retrieval that selects in-context examples from human-annotated samples to improve LLM annotation, and 2) variable batch size that controls the order for choosing each fidelity to facilitate knowledge distillation, ultimately enhancing annotation quality. Extensive experiments on financial and medical tasks demonstrate that IMFL achieves superior performance compared with single fidelity annotations. Given a limited budget of human annotation, IMFL significantly outperforms the $\bf 3\times$ human annotation baselines in all four tasks and achieves very close performance as $\bf 5\times$ human annotation on two of the tasks. These promising results suggest that the high human annotation costs in domain-specific tasks can be significantly reduced by employing IMFL, which utilizes fewer human annotations, supplemented with cheaper and faster LLM (e.g., GPT-3.5) annotations to achieve comparable performance.</details>

### Inverse Reinforcement Learning with the Average Reward Criterion

**Authors:** Feiyang Wu, Jingyang Ke, Anqi Wu

**<details><summary>Abstract**</summary>

We study the problem of Inverse Reinforcement Learning (IRL) with an average-reward criterion. The goal is to recover an unknown policy and a reward function when the agent only has samples of states and actions from an experienced agent. Previous IRL methods assume that the expert is trained in a discounted environment, and the discount factor is known. This work alleviates this assumption by proposing an average-reward framework with efficient learning algorithms. We develop novel stochastic first-order methods to solve the IRL problem under the average-reward setting, which requires solving an Average-reward Markov Decision Process (AMDP) as a subproblem. To solve the subproblem, we develop a Stochastic Policy Mirror Descent (SPMD) method under general state and action spaces that needs $\mathcal{O}(1/\varepsilon)$ steps of gradient computation. Equipped with SPMD, we propose the Inverse Policy Mirror Descent (IPMD) method for solving the IRL problem with a $\mathcal{O}(1/\varepsilon^2)$ complexity. To the best of our knowledge, the aforementioned complexity results are new in IRL with the average reward criterion. Finally, we corroborate our analysis with numerical experiments using the MuJoCo benchmark and additional control tasks.</details>

### Is This Loss Informative? Faster Text-to-Image Customization by Tracking Objective Dynamics

**Authors:** Anton Voronov, Mikhail Khoroshikh, Artem Babenko, Max Ryabinin

**<details><summary>Abstract**</summary>

Text-to-image generation models represent the next step of evolution in image synthesis, offering a natural way to achieve flexible yet fine-grained control over the result.One emerging area of research is the fast adaptation of large text-to-image models to smaller datasets or new visual concepts.However, many efficient methods of adaptation have a long training time, which limits their practical applications, slows down experiments, and spends excessive GPU resources.In this work, we study the training dynamics of popular text-to-image personalization methods (such as Textual Inversion or DreamBooth), aiming to speed them up.We observe that most concepts are learned at early stages and do not improve in quality later, but standard training convergence metrics fail to indicate that.Instead, we propose a simple drop-in early stopping criterion that only requires computing the regular training objective on a fixed set of inputs for all training iterations.Our experiments on Stable Diffusion for 48 different concepts and three personalization methods demonstrate the competitive performance of our approach, which makes adaptation up to 8 times faster with no significant drops in quality.</details>

### Joint Learning of Label and Environment Causal Independence for Graph Out-of-Distribution Generalization

**Authors:** Shurui Gui, Meng Liu, Xiner Li, Youzhi Luo, Shuiwang Ji

**<details><summary>Abstract**</summary>

We tackle the problem of graph out-of-distribution (OOD) generalization. Existing graph OOD algorithms either rely on restricted assumptions or fail to exploit environment information in training data. In this work, we propose to simultaneously incorporate label and environment causal independence (LECI) to fully make use of label and environment information, thereby addressing the challenges faced by prior methods on identifying causal and invariant subgraphs. We further develop an adversarial training strategy to jointly optimize these two properties for casual subgraph discovery with theoretical guarantees. Extensive experiments and analysis show that LECI significantly outperforms prior methods on both synthetic and real-world datasets, establishing LECI as a practical and effective solution for graph OOD generalization.</details>

### Joint Training of Deep Ensembles Fails Due to Learner Collusion

**Authors:** Alan Jeffares, Tennison Liu, Jonathan Crabbé, Mihaela van der Schaar

**<details><summary>Abstract**</summary>

Ensembles of machine learning models have been well established as a powerful method of improving performance over a single model. Traditionally, ensembling algorithms train their base learners independently or sequentially with the goal of optimizing their joint performance. In the case of deep ensembles of neural networks, we are provided with the opportunity to directly optimize the true objective: the joint performance of the ensemble as a whole. Surprisingly, however, directly minimizing the loss of the ensemble appears to rarely be applied in practice. Instead, most previous research trains individual models independently with ensembling performed _post hoc_. In this work, we show that this is for good reason - _joint optimization of ensemble loss results in degenerate behavior_. We approach this problem by decomposing the ensemble objective into the strength of the base learners and the diversity between them. We discover that joint optimization results in a phenomenon in which base learners collude to artificially inflate their apparent diversity. This pseudo-diversity fails to generalize beyond the training data, causing a larger generalization gap. We proceed to comprehensively demonstrate the practical implications of this effect on a range of standard machine learning tasks and architectures by smoothly interpolating between independent training and joint optimization.</details>

### Keypoint-Augmented Self-Supervised Learning for Medical Image Segmentation with Limited Annotation

**Authors:** Zhangsihao Yang, Mengwei Ren, Kaize Ding, Guido Gerig, Yalin Wang

**<details><summary>Abstract**</summary>

Pretraining CNN models (i.e., UNet) through self-supervision has become a powerful approach to facilitate medical image segmentation under low annotation regimes. Recent contrastive learning methods encourage similar global representations when the same image undergoes different transformations, or enforce invariance across different image/patch features that are intrinsically correlated. However, CNN-extracted global and local features are limited in capturing long-range spatial dependencies that are essential in biological anatomy. To this end, we present a keypoint-augmented fusion layer that extracts representations preserving both short- and long-range self-attention. In particular, we augment the CNN feature map at multiple scales by incorporating an additional input that learns long-range spatial self-attention among localized keypoint features. Further, we introduce both global and local self-supervised pretraining for the framework. At the global scale, we obtain global representations from both the bottleneck of the UNet, and by aggregating multiscale keypoint features. These global features are subsequently regularized through image-level contrastive objectives. At the local scale, we define a distance-based criterion to first establish correspondences among keypoints and encourage similarity between their features. Through extensive experiments on both MRI and CT segmentation tasks, we demonstrate the architectural advantages of our proposed method in comparison to both CNN and Transformer-based UNets, when all architectures are trained with randomly initialized weights. With our proposed pretraining strategy, our method further outperforms existing SSL methods by producing more robust self-attention and achieving state-of-the-art segmentation results. The code is available at https://github.com/zshyang/kaf.git.</details>

### [Spotlight] L-CAD: Language-based Colorization with Any-level Descriptions using Diffusion Priors

**Authors:** zheng chang, Shuchen Weng, Peixuan Zhang, Yu Li, Si Li, Boxin Shi

**<details><summary>Abstract**</summary>

Language-based colorization produces plausible and visually pleasing colors under the guidance of user-friendly natural language descriptions. Previous methods implicitly assume that users provide comprehensive color descriptions for most of the objects in the image, which leads to suboptimal performance. In this paper, we propose a unified model to perform language-based colorization with any-level descriptions. We leverage the pretrained cross-modality generative model for its robust language understanding and rich color priors to handle the inherent ambiguity of any-level descriptions. We further design modules to align with input conditions to preserve local spatial structures and prevent the ghosting effect. With the proposed novel sampling strategy, our model achieves instance-aware colorization in diverse and complex scenarios. Extensive experimental results demonstrate our advantages of effectively handling any-level descriptions and outperforming both language-based and automatic colorization methods. The code and pretrained modelsare available at: https://github.com/changzheng123/L-CAD.</details>

### LEACE: Perfect linear concept erasure in closed form

**Authors:** Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, Stella Biderman

**<details><summary>Abstract**</summary>

Concept erasure aims to remove specified features from a representation. It can improve fairness (e.g. preventing a classifier from using gender or race) and interpretability (e.g. removing a concept to observe changes in model behavior). We introduce LEAst-squares Concept Erasure (LEACE), a closed-form method which provably prevents all linear classifiers from detecting a concept while changing the representation as little as possible, as measured by a broad class of norms. We apply LEACE to large language models with a novel procedure called concept scrubbing, which erases target concept information from _every_ layer in the network. We demonstrate our method on two tasks: measuring the reliance of language models on part-of-speech information, and reducing gender bias in BERT embeddings. Our code is available at https://github.com/EleutherAI/concept-erasure.</details>

### LICO: Explainable Models with Language-Image COnsistency

**Authors:** Yiming Lei, Zilong Li, Yangyang Li, Junping Zhang, Hongming Shan

**<details><summary>Abstract**</summary>

Interpreting the decisions of deep learning models has been actively studied since the explosion of deep neural networks. One of the most convincing interpretation approaches is salience-based visual interpretation, such as Grad-CAM, where the generation of attention maps depends merely on categorical labels. Although existing interpretation methods can provide explainable decision clues, they often yield partial correspondence between image and saliency maps due to the limited discriminative information from one-hot labels. This paper develops a Language-Image COnsistency model for explainable image classification, termed LICO,  by correlating learnable linguistic prompts with corresponding visual features in a coarse-to-fine manner. Specifically, we first establish a coarse global manifold structure alignment by minimizing the distance between the distributions of image and language features. We then achieve fine-grained saliency maps by applying optimal transport (OT) theory to assign local feature maps with class-specific prompts. Extensive experimental results on eight benchmark datasets demonstrate that the proposed LICO achieves a significant improvement in generating more explainable attention maps in conjunction with existing interpretation methods such as Grad-CAM. Remarkably, LICO improves the classification performance of existing  models without introducing any computational overhead during inference.</details>

### LMC: Large Model Collaboration with Cross-assessment for Training-Free Open-Set Object Recognition

**Authors:** Haoxuan Qu, Xiaofei Hui, Yujun Cai, Jun Liu

**<details><summary>Abstract**</summary>

Open-set object recognition aims to identify if an object is from a class that has been encountered during training or not. To perform open-set object recognition accurately, a key challenge is how to reduce the reliance on spurious-discriminative features. In this paper, motivated by that different large models pre-trained through different paradigms can possess very rich while distinct implicit knowledge, we propose a novel framework named Large Model Collaboration (LMC) to tackle the above challenge via collaborating different off-the-shelf large models in a training-free manner. Moreover, we also incorporate the proposed framework with several novel designs to effectively extract implicit knowledge from large models. Extensive experiments demonstrate the efficacy of our proposed framework. Code is available \href{https://github.com/Harryqu123/LMC}{here}.</details>

### Language Is Not All You Need: Aligning Perception with Language Models

**Authors:** Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Nils Bjorck, Vishrav Chaudhary, Subhojit Som, XIA SONG, Furu Wei

**<details><summary>Abstract**</summary>

A big convergence of language, multimodal perception, action, and world modeling is a key step toward artificial general intelligence. In this work, we introduce KOSMOS-1, a Multimodal Large Language Model (MLLM) that can perceive general modalities, learn in context (i.e., few-shot), and follow instructions (i.e., zero-shot). Specifically, we train KOSMOS-1 from scratch on web-scale multi-modal corpora, including arbitrarily interleaved text and images, image-caption pairs, and text data. We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning. Experimental results show that KOSMOS-1 achieves impressive performance on (i) language understanding, generation, and even OCR-free NLP (directly fed with document images), (ii) perception-language tasks, including multimodal dialogue, image captioning, visual question answering, and (iii) vision tasks, such as image recognition with descriptions (specifying classification via text instructions). We also show that MLLMs can benefit from cross-modal transfer, i.e., transfer knowledge from language to multimodal, and from multimodal to language. In addition, we introduce a dataset of Raven IQ test, which diagnoses the nonverbal reasoning capability of MLLMs.</details>

### Language Models Can Improve Event Prediction by Few-Shot Abductive Reasoning

**Authors:** Xiaoming Shi, Siqiao Xue, Kangrui Wang, Fan Zhou, James Zhang, Jun Zhou, Chenhao Tan, Hongyuan Mei

**<details><summary>Abstract**</summary>

Large language models have shown astonishing performance on a wide range of reasoning tasks. In this paper, we investigate whether they could reason about real-world events and help improve the prediction performance of event sequence models. We design LAMP, a framework that integrates a large language model in event prediction. Particularly, the language model performs abductive reasoning to assist an event sequence model: the event model proposes predictions on future events given the past; instructed by a few expert-annotated demonstrations, the language model learns to suggest possible causes for each proposal; a search module finds out the previous events that match the causes; a scoring function learns to examine whether the retrieved events could actually cause the proposal. Through extensive experiments on several challenging real-world datasets, we demonstrate that our framework---thanks to the reasoning capabilities of large language models---could significantly outperform the state-of-the-art event sequence models.</details>

### Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting

**Authors:** Miles Turpin, Julian Michael, Ethan Perez, Samuel Bowman

**<details><summary>Abstract**</summary>

Large Language Models (LLMs) can achieve strong performance on many tasks by producing step-by-step reasoning before giving a final output, often referred to as chain-of-thought reasoning (CoT). It is tempting to interpret these CoT explanations as the LLM's process for solving a task. This level of transparency into LLMs' predictions would yield significant safety benefits. However, we find that CoT explanations can systematically misrepresent the true reason for a model's prediction. We demonstrate that CoT explanations can be heavily influenced by adding biasing features to model inputs—e.g., by reordering the multiple-choice options in a few-shot prompt to make the answer always "(A)"—which models systematically fail to mention in their explanations. When we bias models toward incorrect answers, they frequently generate CoT explanations rationalizing those answers. This causes accuracy to drop by as much as 36% on a suite of 13 tasks from BIG-Bench Hard, when testing with GPT-3.5 from OpenAI and Claude 1.0 from Anthropic. On a social-bias task, model explanations justify giving answers in line with stereotypes without mentioning the influence of these social biases. Our findings indicate that CoT explanations can be plausible yet misleading, which risks increasing our trust in LLMs without guaranteeing their safety. Building more transparent and explainable systems will require either improving CoT faithfulness through targeted efforts or abandoning CoT in favor of alternative methods.</details>

### Language Models Meet World Models: Embodied Experiences Enhance Language Models

**Authors:** Jiannan Xiang, Tianhua Tao, Yi Gu, Tianmin Shu, Zirui Wang, Zichao Yang, Zhiting Hu

**<details><summary>Abstract**</summary>

While large language models (LMs) have shown remarkable capabilities across numerous tasks, they often struggle with simple reasoning and planning in physical environments, such as understanding object permanence or planning household activities. The limitation arises from the fact that LMs are trained only on written text and miss essential embodied knowledge and skills. In this paper, we propose a new paradigm of enhancing LMs by finetuning them with world models, to gain diverse embodied knowledge while retaining their general language capabilities. Our approach deploys an embodied agent in a world model, particularly a simulator of the physical world (VirtualHome), and acquires a diverse set of embodied experiences through both goal-oriented planning and random exploration. These experiences are then used to finetune LMs to teach diverse abilities of reasoning and acting in the physical world, e.g., planning and completing goals, object permanence and tracking, etc. Moreover, it is desirable to preserve the generality of LMs during finetuning, which facilitates generalizing the embodied knowledge across tasks rather than being tied to specific simulations. We thus further introduce the classical elastic weight consolidation (EWC) for selective weight updates, combined with low-rank adapters (LoRA) for training efficiency. Extensive experiments show our approach substantially improves base LMs on 18 downstream tasks by 64.28% on average. In particular, the small LMs (1.3B, 6B, and 13B) enhanced by our approach match or even outperform much larger LMs (e.g., ChatGPT).</details>

### Large Language Models Are Semi-Parametric Reinforcement Learning Agents

**Authors:** Danyang Zhang, Lu Chen, Situo Zhang, Hongshen Xu, Zihan Zhao, Kai Yu

**<details><summary>Abstract**</summary>

Inspired by the insights in cognitive science with respect to human memory and reasoning mechanism, a novel evolvable LLM-based (Large Language Model) agent framework is proposed as Rememberer. By equipping the LLM with a long-term experience memory, Rememberer is capable of exploiting the experiences from the past episodes even for different task goals, which excels an LLM-based agent with fixed exemplars or equipped with a transient working memory. We further introduce **R**einforcement **L**earning with **E**xperience **M**emory (**RLEM**) to update the memory. Thus, the whole system can learn from the experiences of both success and failure, and evolve its capability without fine-tuning the parameters of the LLM. In this way, the proposed Rememberer constitutes a semi-parametric RL agent. Extensive experiments are conducted on two RL task sets to evaluate the proposed framework. The average results with different initialization and training sets exceed the prior SOTA by 4% and 2% for the success rate on two task sets and demonstrate the superiority and robustness of Rememberer.</details>

### Large Language Models as Commonsense Knowledge for Large-Scale Task Planning

**Authors:** Zirui Zhao, Wee Sun Lee, David Hsu

**<details><summary>Abstract**</summary>

Large-scale task planning is a major challenge.  Recent work exploits large  language models (LLMs) directly as a policy and shows surprisingly  interesting results.  This paper shows that LLMs provide a  commonsense model of the world in addition to a policy that acts on it.  The world model and the policy can be combined in a search  algorithm, such as Monte Carlo Tree Search (MCTS), to scale up task  planning.  In our new LLM-MCTS algorithm, the LLM-induced world model  provides a commonsense prior belief for MCTS to achieve effective reasoning;  the LLM-induced policy acts as a heuristic to guide the search, vastly  improving search efficiency. Experiments show that LLM-MCTS outperforms  both MCTS alone and policies induced by LLMs (GPT2 and GPT3.5) by a wide  margin, for complex, novel tasks.   Further experiments and analyses on multiple tasks --  multiplication, travel planning, object rearrangement --  suggest minimum description length (MDL)  as a general guiding principle: if the  description length of the world model is substantially smaller than that of  the  policy, using LLM as a world model for model-based planning is likely better  than using LLM solely as a policy.</details>

### Large Language Models of Code Fail at Completing Code with Potential Bugs

**Authors:** Tuan Dinh, Jinman Zhao, Samson Tan, Renato Negrinho, Leonard Lausen, Sheng Zha, George Karypis

**<details><summary>Abstract**</summary>

Large language models of code (Code-LLMs) have recently brought tremendous advances to code completion, a fundamental feature of programming assistance and code intelligence. However, most existing works ignore the possible presence of bugs in the code context for generation, which are inevitable in software development. Therefore, we introduce and study the buggy-code completion problem, inspired by the realistic scenario of real-time code suggestion where the code context contains potential bugs – anti-patterns that can become bugs in the completed program. To systematically study the task, we introduce two datasets: one with synthetic bugs derived from semantics-altering operator changes (buggy-HumanEval) and one with realistic bugs derived from user submissions to coding problems (buggy-FixEval). We find that the presence of potential bugs significantly degrades the generation performance of the high-performing Code-LLMs. For instance, the passing rates of CODEGEN-2B-MONO on test cases of buggy-HumanEval drop more than 50% given a single potential bug in the context. Finally, we investigate several post-hoc methods for mitigating the adverse effect of potential bugs and find that there remains a large gap in post-mitigation performance.</details>

### Large-Scale Distributed Learning via Private On-Device LSH

**Authors:** Tahseen Rabbani, Marco Bornstein, Furong Huang

**<details><summary>Abstract**</summary>

Locality-sensitive hashing (LSH) based frameworks have been used efficiently to select weight vectors in a dense hidden layer with high cosine similarity to an input, enabling dynamic pruning.  While this type of scheme has been shown to improve computational training efficiency, existing algorithms require repeated randomized projection of the full layer weight, which is impractical for computational- and memory-constrained devices.  In a distributed setting, deferring LSH analysis to a centralized host is (i) slow if the device cluster is large and (ii) requires access to input data which is forbidden in a federated context.  Using a new family of hash functions, we develop the first private, personalized, and memory-efficient on-device LSH framework.Our framework enables privacy and personalization by allowing each device to generate hash tables, without the help of a central host, using device-specific hashing hyper-parameters (e.g., number of hash tables or hash length).Hash tables are generated with a compressed set of the full weights, and can be serially generated and discarded if the process is memory-intensive.This allows devices to avoid maintaining (i) the fully-sized model and (ii) large amounts of hash tables in local memory for LSH analysis. We prove several statistical and sensitivity properties of our hash functions, and experimentally demonstrate that our framework is competitive in training large scale recommender networks compared to other LSH frameworks which assume unrestricted on-device capacity.</details>

### Last-Iterate Convergent Policy Gradient Primal-Dual Methods for Constrained MDPs

**Authors:** Dongsheng Ding, Chen-Yu Wei, Kaiqing Zhang, Alejandro Ribeiro

**<details><summary>Abstract**</summary>

We study the problem of computing an optimal policy of an infinite-horizon discounted constrained Markov decision process (constrained MDP). Despite the popularity of Lagrangian-based policy search methods used in practice, the oscillation of policy iterates in these methods has not been fully understood, bringing out issues such as violation of constraints and sensitivity to hyper-parameters. To fill this gap, we employ the Lagrangian method to cast a constrained MDP into a constrained saddle-point problem in which max/min players correspond to primal/dual variables, respectively, and develop two single-time-scale policy-based primal-dual algorithms with non-asymptotic convergence of their policy iterates to an optimal constrained policy. Specifically, we first propose a regularized policy gradient primal-dual (RPG-PD) method that updates the policy using an entropy-regularized policy gradient, and the dual variable via a quadratic-regularized gradient ascent, simultaneously. We prove that the policy primal-dual iterates of RPG-PD converge to a regularized saddle point with a sublinear rate, while the policy iterates converge sublinearly to an optimal constrained policy. We further instantiate RPG-PD in large state or action spaces by including function approximation in policy parametrization, and establish similar sublinear last-iterate policy convergence. Second, we propose an optimistic policy gradient primal-dual (OPG-PD) method that employs the optimistic gradient method to update primal/dual variables, simultaneously. We prove that the policy primal-dual iterates of OPG-PD converge to a saddle point that contains an optimal constrained policy, with a linear rate. To the best of our knowledge, this work appears to be the first non-asymptotic policy last-iterate convergence result for single-time-scale algorithms in constrained MDPs. We further validate the merits and the effectiveness of our methods in computational experiments.</details>

### Latent SDEs on Homogeneous Spaces

**Authors:** Sebastian Zeng, Florian Graf, Roland Kwitt

**<details><summary>Abstract**</summary>

We consider the problem of variational Bayesian inference in a latent variable model where a (possibly complex) observed stochastic process is governed by the unobserved solution of a latent stochastic differential equation (SDE). Motivated by the challenges that arise when trying to learn a latent SDE in $\mathbb{R}^n$ from large-scale data, such as efficient gradient computation, we take a step back and study a specific subclass instead. In our case, the SDE evolves inside a homogeneous latent space and is induced by stochastic dynamics of the corresponding (matrix) Lie group. In the context of learning problems, SDEs on the $n$-dimensional unit sphere are arguably the most relevant incarnation of this setup. For variational inference, the sphere not only facilitates using a uniform prior on the initial state of the SDE, but we also obtain a particularly simple and intuitive expression for the KL divergence between the approximate posterior and prior process in the evidence lower bound. We provide empirical evidence that a latent SDE of the proposed type can be learned efficiently by means of an existing one-step geometric Euler-Maruyama scheme. Despite restricting ourselves to a less diverse class of SDEs, we achieve competitive or even state-of-the-art performance on a collection of time series interpolation and classification benchmarks.</details>

### LayoutGPT: Compositional Visual Planning and Generation with Large Language Models

**Authors:** Weixi Feng, Wanrong Zhu, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Xuehai He, S Basu, Xin Eric Wang, William Yang Wang

**<details><summary>Abstract**</summary>

Attaining a high degree of user controllability in visual generation often requires intricate, fine-grained inputs like layouts. However, such inputs impose a substantial burden on users when compared to simple text inputs. To address the issue, we study how Large Language Models (LLMs) can serve as visual planners by generating layouts from text conditions, and thus collaborate with visual generative models. We propose LayoutGPT, a method to compose in-context visual demonstrations in style sheet language to enhance visual planning skills of LLMs. We show that LayoutGPT can generate plausible layouts in multiple domains, ranging from 2D images to 3D indoor scenes. LayoutGPT also shows superior performance in converting challenging language concepts like numerical and spatial relations to layout arrangements for faithful text-to-image generation. When combined with a downstream image generation model, LayoutGPT outperforms text-to-image models/systems by 20-40\% and achieves comparable performance as human users in designing visual layouts for numerical and spatial correctness. Lastly, LayoutGPT achieves comparable performance to supervised methods in 3D indoor scene synthesis, demonstrating its effectiveness and potential in multiple visual domains.</details>

### Learning Adversarial Low-rank Markov Decision Processes with Unknown Transition and Full-information Feedback

**Authors:** Canzhe Zhao, Ruofeng Yang, Baoxiang Wang, Xuezhou Zhang, Shuai Li

**<details><summary>Abstract**</summary>

In this work, we study the low-rank MDPs with adversarially changed losses in the full-information feedback setting. In particular, the unknown transition probability kernel admits a low-rank matrix decomposition \citep{REPUCB22}, and the loss functions may change adversarially but are revealed to the learner at the end of each episode. We propose a policy optimization-based algorithm POLO, and we prove that it attains the $\widetilde{O}(dA^{\frac{1}{2}}K^{\frac{3}{4}}\ln^{\frac{1}{4}}M/(1-\gamma)^2)$ regret guarantee, where $d$ is rank of the transition kernel (and hence the dimension of the unknown representations), $A$ is the cardinality of the action space, $M$ is the cardinality of the model class, and $\gamma$ is the discounted factor. Notably, our algorithm is oracle-efficient and has a regret guarantee with no dependence on the size of potentially arbitrarily large state space. Furthermore, we also prove an $\Omega(\frac{\gamma^2}{1-\gamma} \sqrt{d A K})$ regret lower bound for this problem, showing that low-rank MDPs are statistically more difficult to learn than linear MDPs in the regret minimization setting. To the best of our knowledge, we present the first algorithm that interleaves representation learning, exploration, and exploitation to achieve the sublinear regret guarantee for RL with nonlinear function approximation and adversarial losses.</details>

### Learning Dynamic Attribute-factored World Models for Efficient Multi-object Reinforcement Learning

**Authors:** Fan Feng, Sara Magliacane

**<details><summary>Abstract**</summary>

In many reinforcement learning tasks, the agent has to learn to interact with many objects of different types and generalize to unseen combinations and numbers of objects. Often a task is a composition of previously learned tasks (e.g. block stacking).These are examples of compositional generalization, in which we compose object-centric representations to solve complex tasks. Recent works have shown the benefits of object-factored representations and hierarchical abstractions for improving sample efficiency in these settings. On the other hand, these methods do not fully exploit the benefits of factorization in terms of object attributes. In this paper, we address this opportunity and introduce the Dynamic Attribute FacTored RL (DAFT-RL) framework. In DAFT-RL, we leverage object-centric representation learning to extract objects from visual inputs. We learn to classify them into classes and infer their latent parameters. For each class of object, we learn a class template graph that describes how the dynamics and reward of an object of this class factorize according to its attributes. We also learn an interaction pattern graph that describes how objects of different classes interact with each other at the attribute level. Through these graphs and a dynamic interaction graph that models the interactions between objects, we can learn a policy that can then be directly applied in a new environment by estimating the interactions and latent parameters.We evaluate DAFT-RL in three benchmark datasets and show our framework outperforms the state-of-the-art in generalizing across unseen objects with varying attributes and latent parameters, as well as in the composition of previously learned tasks.</details>

### Learning Environment-Aware Affordance for 3D Articulated Object Manipulation under Occlusions

**Authors:** Ruihai Wu, Kai Cheng, Yan Zhao, Chuanruo Ning, Guanqi Zhan, Hao Dong

**<details><summary>Abstract**</summary>

Perceiving and manipulating 3D articulated objects in diverse environments is essential for home-assistant robots. Recent studies have shown that point-level affordance provides actionable priors for downstream manipulation tasks. However, existing works primarily focus on single-object scenarios with homogeneous agents, overlooking the realistic constraints imposed by the environment and the agent's morphology, e.g., occlusions and physical limitations. In this paper, we propose an environment-aware affordance framework that incorporates both object-level actionable priors and environment constraints. Unlike object-centric affordance approaches, learning environment-aware affordance faces the challenge of combinatorial explosion due to the complexity of various occlusions, characterized by their quantities, geometries, positions and poses. To address this and enhance data efficiency, we introduce a novel contrastive affordance learning framework capable of training on scenes containing a single occluder and generalizing to scenes with complex occluder combinations. Experiments demonstrate the effectiveness of our proposed approach in learning affordance considering environment constraints.</details>

### [Spotlight] Learning Layer-wise Equivariances Automatically using Gradients

**Authors:** Tycho van der Ouderaa, Alexander Immer, Mark van der Wilk

**<details><summary>Abstract**</summary>

Convolutions encode equivariance symmetries into neural networks leading to better generalisation performance. However, symmetries provide fixed hard constraints on the functions a network can represent, need to be specified in advance, and can not be adapted. Our goal is to allow for flexible symmetry constraints that can automatically be learned from data using gradients. Learning symmetry and associated weight connectivity structures from scratch is difficult for two reasons. First, it requires efficient and flexible parameterisations of layer-wise equivariances. Secondly, symmetries are constraints and are therefore not encouraged by training losses measuring data fit. We overcome these challenges by improving upon parameterisations of soft equivariance and show that we can learn the amount of equivariance in layers by optimising the marginal likelihood, estimated using differentiable Laplace approximations. The marginal likelihood balances data fit and model complexity enabling layer-wise symmetry discovery in deep neural networks. We demonstrate the ability to automatically learn layer-wise equivariances on image classification tasks, achieving equivalent or improved performance over baselines with hard-coded symmetry.</details>

### Learning Nonparametric Latent Causal Graphs with Unknown Interventions

**Authors:** Yibo Jiang, Bryon Aragam

**<details><summary>Abstract**</summary>

We establish conditions under which latent causal graphs are nonparametrically identifiable and can be reconstructed from unknown interventions in the latent space. Our primary focus is the identification of the latent structure in measurement models without parametric assumptions such as linearity or Gaussianity. Moreover, we do not assume the number of hidden variables is known, and we show that at most one unknown intervention per hidden variable is needed. This extends a recent line of work on learning causal representations from observations and interventions. The proofs are constructive and introduce two new graphical concepts---_imaginary subsets_ and _isolated edges_---that may be useful in their own right. As a matter of independent interest, the proofs also involve a novel characterization of the limits of edge orientations within the equivalence class of DAGs induced by _unknown_ interventions. These are the first results to characterize the conditions under which causal representations are identifiable without making any parametric assumptions in a general setting with unknown interventions and without faithfulness.</details>

### Learning Regularized Monotone Graphon Mean-Field Games

**Authors:** Fengzhuo Zhang, Vincent Tan, Zhaoran Wang, Zhuoran Yang

**<details><summary>Abstract**</summary>

This paper studies two fundamental problems in regularized Graphon Mean-Field Games (GMFGs). First, we establish the existence of a Nash Equilibrium (NE)  of any $\lambda$-regularized GMFG   (for  $\lambda\geq 0$). This result relies on weaker conditions than previous works analyzing both unregularized GMFGs ($\lambda=0$) and $\lambda$-regularized MFGs, which are special cases of GMFGs. Second, we propose provably efficient algorithms to learn the NE in  weakly monotone GMFGs, motivated by Lasry and Lions (2007). Previous literature either only analyzed continuous-time algorithms or required extra conditions to analyze discrete-time algorithms. In contrast, we design a discrete-time algorithm and derive its convergence rate  solely under  weakly monotone conditions. Furthermore, we develop and analyze the action-value function estimation procedure during the online learning process, which is absent from algorithms for monotone GMFGs. This serves as a sub-module in our optimization algorithm. The efficiency of the designed algorithm is corroborated by empirical evaluations.</details>

### Learning Robust Statistics for Simulation-based Inference under Model Misspecification

**Authors:** Daolang Huang, Ayush Bharti, Amauri Souza, Luigi Acerbi, Samuel Kaski

**<details><summary>Abstract**</summary>

Simulation-based inference (SBI) methods such as approximate Bayesian computation (ABC),  synthetic likelihood, and neural posterior estimation (NPE) rely on simulating statistics to infer parameters of intractable likelihood models. However, such methods are known to yield untrustworthy and misleading inference outcomes under model misspecification, thus hindering their widespread applicability. In this work, we propose the first general approach to handle model misspecification that works across different classes of SBI methods. Leveraging the fact that the choice of statistics determines the degree of misspecification in SBI, we introduce a regularized loss function that penalizes those statistics that increase the mismatch between the data and the model. Taking NPE and ABC as use cases, we demonstrate the superior performance of our method on high-dimensional time-series models that are artificially misspecified. We also apply our method to real data from the field of radio propagation where the model is known to be misspecified. We show empirically that the method yields robust inference in misspecified scenarios, whilst still being accurate when the model is well-specified.</details>

### Learning Rule-Induced Subgraph Representations for Inductive Relation Prediction

**Authors:** Tianyu Liu, Qitan Lv, Jie Wang, Shuling Yang, Hanzhu Chen

**<details><summary>Abstract**</summary>

Inductive relation prediction (IRP)---where entities can be different during training and inference---has shown great power for completing evolving knowledge graphs. Existing works mainly focus on using graph neural networks (GNNs) to learn the representation of the subgraph induced from the target link, which can be seen as an implicit rule-mining process to measure the plausibility of the target link. However, these methods are not able to differentiate the target link and other links during message passing, hence the final subgraph representation will contain irrelevant rule information to the target link, which reduces the reasoning performance and severely hinders the applications for real-world scenarios. To tackle this problem, we propose a novel $\textit{single-source edge-wise}$ GNN model to learn the $\textbf{R}$ule-induc$\textbf{E}$d $\textbf{S}$ubgraph represen$\textbf{T}$ations $(\textbf{REST}$), which encodes relevant rules and eliminates irrelevant rules within the subgraph. Specifically, we propose a $\textit{single-source}$ initialization approach to initialize edge features only for the target link, which guarantees the relevance of mined rules and target link. Then we propose several RNN-based functions for $\textit{edge-wise}$ message passing to model the sequential property of mined rules. REST is a simple and effective approach with theoretical support to learn the $\textit{rule-induced subgraph representation}$. Moreover, REST does not need node labeling, which significantly accelerates the subgraph preprocessing time by up to $\textbf{11.66}\times$. Experiments on inductive relation prediction benchmarks demonstrate the effectiveness of our REST.</details>

### Learning Trajectories are Generalization Indicators

**Authors:** Jingwen Fu, Zhizheng Zhang, Dacheng Yin, Yan Lu, Nanning Zheng

**<details><summary>Abstract**</summary>

This paper explores the connection between learning trajectories of Deep Neural Networks (DNNs) and their generalization capabilities when optimized using (stochastic) gradient descent algorithms. Instead of concentrating solely on the generalization error of the DNN post-training, we present a novel perspective for analyzing generalization error by investigating the contribution of each update step to the change in generalization error. This perspective enable a more direct comprehension of how the learning trajectory influences generalization error. Building upon this analysis, we propose a new generalization bound that incorporates more extensive trajectory information.Our proposed generalization bound depends on the complexity of learning trajectory and the ratio between the bias and diversity of training set. Experimental observations reveal that our method effectively captures the generalization error throughout the training process. Furthermore, our approach can also track changes in generalization error when adjustments are made to learning rates and label noise levels. These results demonstrate that learning trajectory information is a valuable indicator of a model's generalization capabilities.</details>

### [Spotlight] Learning Universal Policies via Text-Guided Video Generation

**Authors:** Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, Pieter Abbeel

**<details><summary>Abstract**</summary>

A goal of artificial intelligence is to construct an agent that can solve a wide variety of tasks. Recent progress in text-guided image synthesis has yielded models with an impressive ability to generate complex novel images, exhibiting combinatorial generalization across domains. Motivated by this success, we investigate whether such tools can be used to construct more general-purpose agents. Specifically, we cast the sequential decision making problem as a text-conditioned video generation problem, where, given a text-encoded specification of a desired goal, a planner synthesizes a set of future frames depicting its planned actions in the future, after which control actions are extracted from the generated video. By leveraging text as the underlying goal specification, we are able to naturally and combinatorially generalize to novel goals. The proposed policy-as-video formulation can further represent environments with different state and action spaces in a unified space of images, which, for example, enables learning and generalization across a variety of robot manipulation tasks. Finally, by leveraging pretrained language embeddings and widely available videos from the internet, the approach enables knowledge transfer through predicting highly realistic video plans for real robots.</details>

### Learning non-Markovian Decision-Making from State-only Sequences

**Authors:** Aoyang Qin, Feng Gao, Qing Li, Song-Chun Zhu, Sirui Xie

**<details><summary>Abstract**</summary>

Conventional imitation learning assumes access to the actions of demonstrators, but these motor signals are often non-observable in naturalistic settings. Additionally, sequential decision-making behaviors in these settings can deviate from the assumptions of a standard Markov Decision Process (MDP). To address these challenges, we explore deep generative modeling of state-only sequences with non-Markov Decision Process (nMDP), where the policy is an energy-based prior in the latent space of the state transition generator. We develop maximum likelihood estimation to achieve model-based imitation, which involves short-run MCMC sampling from the prior and importance sampling for the posterior. The learned model enables $\textit{decision-making as inference}$: model-free policy execution is equivalent to prior sampling, model-based planning is posterior sampling initialized from the policy. We demonstrate the efficacy of the proposed method in a prototypical path planning task with non-Markovian constraints and show that the learned model exhibits strong performances in challenging domains from the MuJoCo suite.</details>

### Learning to Modulate pre-trained Models in RL

**Authors:** Thomas Schmied, Markus Hofmarcher, Fabian Paischer, Razvan Pascanu, Sepp Hochreiter

**<details><summary>Abstract**</summary>

Reinforcement Learning (RL) has been successful in various domains like robotics, game playing, and simulation. While RL agents have shown impressive capabilities in their specific tasks, they insufficiently adapt to new tasks. In supervised learning, this adaptation problem is addressed by large-scale pre-training followed by fine-tuning to new down-stream tasks. Recently, pre-training on multiple tasks has been gaining traction in RL. However, fine-tuning a pre-trained model often suffers from catastrophic forgetting. That is, the performance on the pre-training tasks deteriorates when fine-tuning on new tasks. To investigate the catastrophic forgetting phenomenon, we first jointly pre-train a model on datasets from two benchmark suites, namely Meta-World and DMControl. Then, we evaluate and compare a variety of fine-tuning methods prevalent in natural language processing, both in terms of performance on new tasks, and how well performance on pre-training tasks is retained. Our study shows that with most fine-tuning approaches, the performance on pre-training tasks deteriorates significantly. Therefore, we propose a novel method, Learning-to-Modulate (L2M), that avoids the degradation of learned skills by modulating the information flow of the frozen pre-trained model via a learnable modulation pool. Our method achieves state-of-the-art performance on the Continual-World benchmark, while retaining performance on the pre-training tasks. Finally, to aid future research in this area, we release a dataset encompassing 50 Meta-World and 16 DMControl tasks.</details>

### Learning to Reason and Memorize with Self-Notes

**Authors:** Jack Lanchantin, Shubham Toshniwal, Jason Weston, arthur szlam, Sainbayar Sukhbaatar

**<details><summary>Abstract**</summary>

Large language models have been shown to struggle with multi-step reasoning, and do not retain previous reasoning steps for future use. We propose a simple method for solving both of these problems by allowing the model to take Self-Notes. Unlike recent chain-of-thought or scratchpad approaches, the model can deviate from the input context at any time to explicitly think and write down its thoughts. This allows the model to perform reasoning on the fly as it reads the context and even integrate previous reasoning steps, thus enhancing its memory with useful information and enabling multi-step reasoning. Experiments across a wide variety of tasks demonstrate that our method can outperform chain-of-thought and scratchpad methods by taking Self-Notes that interleave the input text.</details>

### Learning to Tokenize for Generative Retrieval

**Authors:** Weiwei Sun, Lingyong Yan, Zheng Chen, Shuaiqiang Wang, Haichao Zhu, Pengjie Ren, Zhumin Chen, Dawei Yin, Maarten Rijke, Zhaochun Ren

**<details><summary>Abstract**</summary>

As a new paradigm in information retrieval, generative retrieval directly generates a ranked list of document identifiers (docids) for a given query using generative language models (LMs).How to assign each document a unique docid (denoted as document tokenization) is a critical problem, because it determines whether the generative retrieval model can precisely retrieve any document by simply decoding its docid.Most existing methods adopt rule-based tokenization, which is ad-hoc and does not generalize well.In contrast, in this paper we propose a novel document tokenization learning method, GenRet, which learns to encode the complete document semantics into docids.GenRet learns to tokenize documents into short discrete representations (i.e., docids) via a discrete auto-encoding approach.We develop a progressive training scheme to capture the autoregressive nature of docids and diverse clustering techniques to stabilize the training process.Based on the semantic-embedded docids of any set of documents, the generative retrieval model can learn to generate the most relevant docid only according to the docids' semantic relevance to the queries.We conduct experiments on the NQ320K, MS MARCO, and BEIR datasets.GenRet establishes the new state-of-the-art on the NQ320K dataset.Compared to generative retrieval baselines, GenRet can achieve significant improvements on unseen documents.Moreover, GenRet can also outperform comparable baselines on MS MARCO and BEIR, demonstrating the method's generalizability.</details>

### Learning-to-Rank Meets Language: Boosting Language-Driven Ordering Alignment for Ordinal Classification

**Authors:** Rui Wang, Peipei Li, Huaibo Huang, Chunshui Cao, Ran He, Zhaofeng He

**<details><summary>Abstract**</summary>

We present a novel language-driven ordering alignment method for ordinal classification. The labels in ordinal classification contain additional ordering relations, making them prone to overfitting when relying solely on training data. Recent developments in pre-trained vision-language models inspire us to leverage the rich ordinal priors in human language by converting the original task into a vision-language alignment task. Consequently, we propose L2RCLIP, which fully utilizes the language priors from two perspectives. First, we introduce a complementary prompt tuning technique called RankFormer, designed to enhance the ordering relation of original rank prompts. It employs token-level attention with residual-style prompt blending in the word embedding space. Second, to further incorporate language priors, we revisit the approximate bound optimization of vanilla cross-entropy loss and restructure it within the cross-modal embedding space. Consequently, we propose a cross-modal ordinal pairwise loss to refine the CLIP feature space, where texts and images maintain both semantic alignment and ordering alignment.  Extensive experiments on three ordinal classification tasks, including facial age estimation, historical color image (HCI) classification, and aesthetic assessment demonstrate its promising performance.</details>

### Leveraging Locality and Robustness to Achieve Massively Scalable Gaussian Process Regression

**Authors:** Robert Allison, Anthony Stephenson, Samuel F, Edward O Pyzer-Knapp

**<details><summary>Abstract**</summary>

The accurate predictions and principled uncertainty measures provided by GP regression incur $O(n^3)$ cost which is prohibitive for modern-day large-scale applications. This has motivated extensive work on computationally efficient approximations. We introduce a new perspective by exploring robustness properties and limiting behaviour of GP nearest-neighbour (GPnn) prediction. We demonstrate through theory and simulation that as the data-size $n$ increases, accuracy of estimated parameters and GP model assumptions become increasingly irrelevant to GPnn predictive accuracy. Consequently, it is sufficient to spend small amounts of work on parameter estimation in order to achieve high MSE accuracy, even in the presence of gross misspecification. In contrast, as $n \rightarrow \infty$, uncertainty calibration and NLL are shown to remain sensitive to just one parameter, the additive noise-variance; but we show that this source of inaccuracy can be corrected for, thereby achieving both well-calibrated uncertainty measures and accurate predictions at remarkably low computational cost. We exhibit a very simple GPnn regression algorithm with stand-out performance compared to other state-of-the-art GP approximations as measured on large UCI datasets. It operates at a small fraction of those other methods' training costs, for example on a basic laptop taking about 30 seconds to train on a dataset of size $n = 1.6 \times 10^6$.</details>

### Lie Point Symmetry and Physics-Informed Networks

**Authors:** Tara Akhound-Sadegh, Laurence Perreault-Levasseur, Johannes Brandstetter, Max Welling, Siamak Ravanbakhsh

**<details><summary>Abstract**</summary>

Symmetries have been leveraged to improve the generalization of neural networks through different mechanisms from data augmentation to equivariant architectures. However, despite their potential, their integration into neural solvers for partial differential equations (PDEs) remains largely unexplored. We explore the integration of PDE symmetries, known as Lie point symmetries, in a major family of neural solvers known as physics-informed neural networks (PINNs). We propose a loss function that informs the network about Lie point symmetries in the same way that PINN models try to enforce the underlying PDE through a loss function. Intuitively, our symmetry loss ensures that the infinitesimal generators of the Lie group conserve the PDE solutions.. Effectively, this means that once the network learns a solution, it also learns the neighbouring solutions generated by Lie point symmetries.Empirical evaluations indicate that the inductive bias introduced by the Lie point symmetries of the PDEs greatly boosts the sample efficiency of PINNs.</details>

### LightSpeed: Light and Fast Neural Light Fields on Mobile Devices

**Authors:** Aarush Gupta, Junli Cao, Chaoyang Wang, Ju Hu, Sergey Tulyakov, Jian Ren, László Jeni

**<details><summary>Abstract**</summary>

Real-time novel-view image synthesis on mobile devices is prohibitive due to the limited computational power and storage. Using volumetric rendering methods, such as NeRF and its derivatives, on mobile devices is not suitable due to the high computational cost of volumetric rendering. On the other hand, recent advances in neural light field representations have shown promising real-time view synthesis results on mobile devices. Neural light field methods learn a direct mapping from a ray representation to the pixel color. The current choice of ray representation is either stratified ray sampling or Plücker coordinates, overlooking the classic light slab (two-plane) representation, the preferred representation to interpolate between light field views. In this work, we find that using the light slab representation is an efficient representation for learning a neural light field. More importantly, it is a lower-dimensional ray representation enabling us to learn the 4D ray space using feature grids which are significantly faster to train and render. Although mostly designed for frontal views, we show that the light-slab representation can be further extended to non-frontal scenes using a divide-and-conquer strategy. Our method provides better rendering quality than prior light field methods and a significantly better trade-off between rendering quality and speed than prior light field methods.</details>

### Likelihood-Based Diffusion Language Models

**Authors:** Ishaan Gulrajani, Tatsunori Hashimoto

**<details><summary>Abstract**</summary>

Despite a growing interest in diffusion-based language models, existing work has not shown that these models can attain nontrivial likelihoods on standard language modeling benchmarks. In this work, we take the first steps towards closing the likelihood gap between autoregressive and diffusion-based language models, with the goal of building and releasing a diffusion model which outperforms a small but widely-known autoregressive model. We pursue this goal through algorithmic improvements, scaling laws, and increased compute. On the algorithmic front, we introduce several methodological improvements for the maximum-likelihood training of diffusion language models. We then study scaling laws for our diffusion models and find compute-optimal training regimes which differ substantially from autoregressive models. Using our methods and scaling analysis, we train and release Plaid 1B, a large diffusion language model which outperforms GPT-2 124M in likelihood on benchmark datasets and generates fluent samples in unconditional and zero-shot control settings.</details>

### Local Convergence of Gradient Methods for Min-Max Games: Partial Curvature Generically Suffices

**Authors:** Guillaume Wang, Lénaïc Chizat

**<details><summary>Abstract**</summary>

We study the convergence to local Nash equilibria of gradient methods for two-player zero-sum differentiable games.It is well-known that, in the continuous-time setting, such dynamics converge locally when $S \succ 0$ and may diverge when $S=0$, where $S\succeq 0$ is the symmetric part of the Jacobian at equilibrium that accounts for the "potential" component of the game. We show that these dynamics also converge as soon as $S$ is nonzero (*partial curvature*) and the eigenvectors of the antisymmetric part $A$ are in general position with respect to the kernel of $S$.We then study the convergence rate when $S \ll A$ and prove that it typically depends on the *average* of the eigenvalues of $S$, instead of the minimum as an analogy with minimization problems would suggest.To illustrate our results, we consider the problem of computing mixed Nash equilibria of continuous games. We show that, thanks to partial curvature, conic particle methods -- which optimize over both weights and supports of the mixed strategies -- generically converge faster than fixed-support methods.For min-max games, it is thus beneficial to add degrees of freedom "with curvature": this can be interpreted as yet another benefit of over-parameterization.</details>

### Localized Symbolic Knowledge Distillation for Visual Commonsense Models

**Authors:** Jae Sung Park, Jack Hessel, Khyathi Chandu, Paul Pu Liang, Ximing Lu, Peter West, Youngjae Yu, Qiuyuan Huang, Jianfeng Gao, Ali Farhadi, Yejin Choi

**<details><summary>Abstract**</summary>

Instruction following vision-language (VL) models offer a flexibleinterface that supports a broad range of multimodal tasks in a zero-shot fashion.However, interfaces that operate on full images do not directly enable the user to“point to" and access specific regions within images. This capability is importantnot only to support reference-grounded VL benchmarks, but also, for practicalapplications that require precise within-image reasoning. We build LocalizedVisual Commonsense model which allows users to specify (multiple) regions-as-input. We train our model by sampling localized commonsense knowledgefrom a large language model (LLM): specifically, we prompt a LLM to collectcommonsense knowledge given a global literal image description and a localliteral region description automatically generated by a set of VL models. Thispipeline is scalable and fully automatic, as no aligned or human-authored imageand text pairs are required. With a separately trained critic model that selectshigh quality examples, we find that training on the localized commonsense corpusexpanded solely from images can successfully distill existing VL models to supporta reference-as-input interface. Empirical results and human evaluations in zero-shotsettings demonstrate that our distillation method results in more precise VL modelsof reasoning compared to a baseline of passing a generated referring expression.</details>

### Locally Invariant Explanations: Towards Stable and Unidirectional Explanations through Local Invariant Learning

**Authors:** Amit Dhurandhar, Karthikeyan Natesan Ramamurthy, Kartik Ahuja, Vijay Arya

**<details><summary>Abstract**</summary>

Locally interpretable model agnostic explanations (LIME) method is one of the most popular methods used to explain black-box models at a per example level. Although many variants have been proposed, few provide a simple way to produce high fidelity explanations that are also stable and intuitive. In this work, we provide a novel perspective by proposing a model agnostic local explanation method inspired by the invariant risk minimization (IRM) principle -- originally proposed for (global) out-of-distribution generalization -- to provide such high fidelity explanations that are also stable and unidirectional across nearby examples. Our method is based on a game theoretic formulation where we theoretically show that our approach has a strong tendency to eliminate features where the gradient of the black-box function abruptly changes sign in the locality of the example we want to explain, while in other cases it is more careful and will choose a more conservative (feature) attribution, a behavior which can be highly desirable for recourse. Empirically, we show on tabular, image and text data that the quality of our explanations with neighborhoods formed using random perturbations are much better than LIME and in some cases even comparable to other methods that use realistic neighbors sampled from the data manifold. This is desirable given that learning a manifold to either create realistic neighbors or to project explanations is typically expensive or may even be impossible. Moreover, our algorithm is simple and efficient to train, and can ascertain stable input features for local decisions of a black-box without access to side information such as a (partial) causal graph as has been seen in some recent works.</details>

### Lockdown: Backdoor Defense for Federated Learning  with Isolated Subspace Training

**Authors:** Tiansheng Huang, Sihao Hu, Ka-Ho Chow, Fatih Ilhan, Selim Tekin, Ling Liu

**<details><summary>Abstract**</summary>

Federated learning (FL) is vulnerable to backdoor attacks due to its distributed computing nature. Existing defense solution usually requires larger amount of computation in either the training or testing phase, which limits their practicality in the resource-constrain scenarios. A more practical defense, i.e., neural network (NN) pruning based defense has been proposed in centralized backdoor setting. However, our empirical study shows that traditional pruning-based solution suffers 	extit{poison-coupling} effect in FL, which significantly degrades the defense performance.This paper presents Lockdown, an isolated subspace training method to mitigate the poison-coupling effect. Lockdown follows three key procedures. First, it modifies the training protocol by isolating the training subspaces for different clients. Second, it utilizes randomness in initializing isolated subspacess, and performs subspace pruning and subspace recovery to segregate the subspaces between malicious and benign clients. Third, it introduces quorum consensus to cure the global model by purging malicious/dummy parameters. Empirical results show that Lockdown achieves 	extit{superior} and 	extit{consistent} defense performance compared to existing representative approaches against backdoor attacks. Another value-added property of Lockdown is the communication-efficiency and model complexity reduction, which are both critical for resource-constrain FL scenario. Our code is available at \url{https://github.com/git-disl/Lockdown}.</details>

### Look Ma, No Hands! Agent-Environment Factorization of Egocentric Videos

**Authors:** Matthew Chang, Aditya Prakash, Saurabh Gupta

**<details><summary>Abstract**</summary>

The analysis and use of egocentric videos for robotics tasks is made challenging by occlusion and the visual mismatch between the human hand and a robot end-effector. Past work views the human hand as a nuisance and removes it from the scene. However, the hand also provides a valuable signal for learning. In this work, we propose to extract a factored representation of the scene that separates the agent (human hand) and the environment. This alleviates both occlusion and mismatch while preserving the signal, thereby easing the design of models for downstream robotics tasks. At the heart of this factorization is our proposed Video Inpainting via Diffusion Model (VIDM) that leverages both a prior on real-world images (through a large-scale pre-trained diffusion model) and the appearance of the object in earlier frames of the video (through attention). Our experiments demonstrate the effectiveness of VIDM at improving the in-painting quality in egocentric videos and the power of our factored representation for numerous tasks: object detection, 3D reconstruction of manipulated objects, and learning of reward functions, policies, and affordances from videos.</details>

### Loss Dynamics of Temporal Difference Reinforcement Learning

**Authors:** Blake Bordelon, Paul Masset, Henry Kuo, Cengiz Pehlevan

**<details><summary>Abstract**</summary>

Reinforcement learning has been successful across several applications in which agents have to learn to act in environments with sparse feedback. However, despite this empirical success there is still a lack of theoretical understanding of how the parameters of reinforcement learning models and the features used to represent states interact to control the dynamics of learning. In this work, we use concepts from statistical physics, to study the typical case learning curves for temporal difference learning of a value function with linear function approximators. Our theory is derived under a Gaussian equivalence hypothesis where averages over the random trajectories are replaced with temporally correlated Gaussian feature averages and we validate our assumptions on small scale Markov Decision Processes. We find that the stochastic semi-gradient noise due to subsampling the space of possible episodes leads to significant plateaus in the value error, unlike in traditional gradient descent dynamics. We study how learning dynamics and plateaus depend on feature structure, learning rate, discount factor, and reward function. We then analyze how strategies like learning rate annealing and reward shaping can favorably alter learning dynamics and plateaus. To conclude, our work introduces new tools to open a new direction towards developing a theory of learning dynamics in reinforcement learning.</details>

### Low Tensor Rank Learning of Neural Dynamics

**Authors:** Arthur Pellegrino, N Alex Cayco Gajic, Angus Chadwick

**<details><summary>Abstract**</summary>

Learning relies on coordinated synaptic changes in recurrently connected populations of neurons. Therefore, understanding the collective evolution of synaptic connectivity over learning is a key challenge in neuroscience and machine learning. In particular, recent work has shown that the weight matrices of task-trained RNNs are typically low rank, but how this low rank structure unfolds over learning is unknown. To address this, we investigate the rank of the 3-tensor formed by the weight matrices throughout learning. By fitting RNNs of varying rank to large-scale neural recordings during a motor learning task, we find that the inferred weights are low-tensor-rank and therefore evolve over a fixed low-dimensional subspace throughout the entire course of learning. We next validate the observation of low-tensor-rank learning on an RNN trained to solve the same task. Finally, we present a set of mathematical results bounding the matrix and tensor ranks of gradient descent learning dynamics which show that low-tensor-rank weights emerge naturally in RNNs trained to solve low-dimensional tasks. Taken together, our findings provide insight on the evolution of population connectivity over learning in both biological and artificial neural networks, and enable reverse engineering of learning-induced changes in recurrent dynamics from large-scale neural recordings.</details>

### MIM4DD: Mutual Information Maximization for Dataset Distillation

**Authors:** Yuzhang Shang, Zhihang Yuan, Yan Yan

**<details><summary>Abstract**</summary>

Dataset distillation (DD) aims to synthesize a small dataset whose test performance is comparable to a full dataset using the same model. State-of-the-art (SoTA) methods optimize synthetic datasets primarily by matching heuristic indicators extracted from two networks: one from real data and one from synthetic data (see Fig.1, Left), such as gradients and training trajectories. DD is essentially a compression problem that emphasizes on maximizing the preservation of information contained in the data. We argue that well-defined metrics which measure the amount of shared information between variables in information theory are necessary for success measurement, but are never considered by previous works. Thus, we introduce mutual information (MI) as the metric to quantify the shared information between the synthetic and the real datasets, and devise MIM4DD numerically maximizing the MI via a newly designed optimizable objective within a contrastive learning framework to update the synthetic dataset. Specifically, we designate the samples in different datasets who share the same labels as positive pairs, and vice versa negative pairs. Then we respectively pull and push those samples in positive and negative pairs into contrastive space via minimizing NCE loss. As a result, the targeted MI can be transformed into a lower bound represented by feature maps of samples, which is numerically feasible. Experiment results show that MIM4DD can be implemented as an add-on module to existing SoTA DD methods.</details>

### Making Scalable Meta Learning Practical

**Authors:** Sang Choe, Sanket Vaibhav Mehta, Hwijeen Ahn, Willie Neiswanger, Pengtao Xie, Emma Strubell, Eric Xing

**<details><summary>Abstract**</summary>

Despite its flexibility to learn diverse inductive biases in machine learning programs, meta learning (i.e.,\ learning to learn) has long been recognized to suffer from poor scalability due to its tremendous compute/memory costs, training instability, and a lack of efficient distributed training support. In this work, we focus on making scalable meta learning practical by introducing SAMA, which combines advances in both implicit differentiation algorithms and systems. Specifically, SAMA is designed to flexibly support a broad range of adaptive optimizers in the base level of meta learning programs, while reducing computational burden by avoiding explicit computation of second-order gradient information, and exploiting efficient distributed training techniques implemented for first-order gradients. Evaluated on multiple large-scale meta learning benchmarks, SAMA showcases up to 1.7/4.8x increase in throughput and 2.0/3.8x decrease in memory consumption respectively on single-/multi-GPU setups compared to other baseline meta learning algorithms. Furthermore, we show that SAMA-based data optimization leads to consistent improvements in text classification accuracy with BERT and RoBERTa large language models, and achieves state-of-the-art results in both small- and large-scale data pruning on image classification tasks, demonstrating the practical applicability of scalable meta learning across language and vision domains.</details>

### Marich: A Query-efficient Distributionally Equivalent Model Extraction Attack

**Authors:** Pratik Karmakar, Debabrota Basu

**<details><summary>Abstract**</summary>

We study design of black-box model extraction attacks that can *send minimal number of queries from* a *publicly available dataset* to a target ML model through a predictive API with an aim *to create an informative and distributionally equivalent replica* of the target.First, we define *distributionally equivalent* and *Max-Information model extraction* attacks, and reduce them into a variational optimisation problem. The attacker sequentially solves this optimisation problem to select the most informative queries that simultaneously maximise the entropy and reduce the mismatch between the target and the stolen models. This leads to *an active sampling-based query selection algorithm*, Marich, which is *model-oblivious*. Then, we evaluate Marich on different text and image data sets, and different models, including CNNs and BERT. Marich extracts models that achieve $\sim 60-95\%$ of true model's accuracy and uses $\sim 1,000 - 8,500$ queries from the publicly available datasets, which are different from the private training datasets. Models extracted by Marich yield prediction distributions, which are $\sim2-4\times$ closer to the target's distribution in comparison to the existing active sampling-based attacks. The extracted models also lead to 84-96$\%$ accuracy under membership inference attacks. Experimental results validate that Marich is *query-efficient*, and capable of performing task-accurate, high-fidelity, and informative model extraction.</details>

### MarioGPT: Open-Ended Text2Level Generation through Large Language Models

**Authors:** Shyam Sudhakaran, Miguel González-Duque, Matthias Freiberger, Claire Glanois, Elias Najarro, Sebastian Risi

**<details><summary>Abstract**</summary>

Procedural Content Generation (PCG) is a technique to generate complex and diverse environments in an automated way. However, while generating content with PCG methods is often straightforward, generating meaningful content that reflects specific intentions and constraints remains challenging. Furthermore, many PCG algorithms lack the ability to generate content in an open-ended manner. Recently, Large Language Models (LLMs) have shown to be incredibly effective in many diverse domains. These trained LLMs can be fine-tuned, re-using information and accelerating training for new tasks. Here, we introduce MarioGPT, a fine-tuned GPT2 model trained to generate tile-based game levels, in our case Super Mario Bros levels. MarioGPT can not only generate diverse levels, but can be text-prompted for controllable level generation, addressing one of the key challenges of current PCG techniques. As far as we know, MarioGPT is the first text-to-level model and combined with novelty search it enables the generation of diverse levels with varying play-style dynamics (i.e. player paths) and the open-ended discovery of an increasingly diverse range of content. Code available at https://github.com/shyamsn97/mario-gpt.</details>

### Mass-Producing Failures of Multimodal Systems with Language Models

**Authors:** Shengbang Tong, Erik Jones, Jacob Steinhardt

**<details><summary>Abstract**</summary>

Deployed multimodal models can fail in ways that evaluators did not anticipate. In order to find these failures before deployment, we introduce MultiMon, a system that automatically identifies systematic failures---generalizable, natural-language descriptions that describe categories of individual failures. To uncover systematic failures, MultiMon scrapes for examples of erroneous agreement: inputs that produce the same output, but should not. It then prompts a language model to identify common categories and describe them in natural language. We use MultiMon to find 14 systematic failures (e.g."ignores quantifiers'') of the CLIP text-encoder, each comprising hundreds of distinct inputs (e.g."a shelf with a few/many books''). Because CLIP is the backbone for most state-of-the-art multimodal models, these inputs produce failures in Midjourney 5.1, DALL-E, VideoFusion, and others. MultiMon can also steer towards failures relevant to specific use cases, such as self-driving cars. We see MultiMon as a step towards evaluation that autonomously explores the long-tail of potential system failures.</details>

### MathNAS: If Blocks Have a Role in Mathematical Architecture Design

**Authors:** Qinsi Wang, Jinghan Ke, Zhi Liang, Sihai Zhang

**<details><summary>Abstract**</summary>

Neural Architecture Search (NAS) has emerged as a favoured method for unearthing effective neural architectures. Recent development of large models has intensified the demand for faster search speeds and more accurate search results. However, designing large models by NAS is challenging due to the dramatical increase of search space and the associated huge performance evaluation cost. Consider a typical modular search space widely used in NAS, in which a neural architecture consists of $m$ block nodes and a block node has $n$ alternative blocks. Facing the space containing $n^m$ candidate networks, existing NAS methods attempt to find the best one by searching and evaluating candidate networks directly.Different from the general strategy that takes architecture search as a whole problem, we propose a novel divide-and-conquer strategy by making use of the modular nature of the search space.Here, we introduce MathNAS, a general NAS framework based on mathematical programming.  In MathNAS, the performances of all possible building blocks in the search space are calculated first, and then the performance of a network is directly predicted based on the performances of its building blocks.Although estimating block performances involves network training, just as what happens for network performance evaluation in existing NAS methods, predicting network performance is completely training-free and thus extremely fast. In contrast to the $n^m$ candidate networks to evaluate in existing NAS methods, which requires training and a formidable computational burden, there are only $m*n$ possible blocks to handle in MathNAS.Therefore, our approach effectively reduces the complexity of network performance evaluation. The superiority of MathNAS is validated on multiple large-scale CV and NLP benchmark datasets. Notably on ImageNet-1k, MathNAS achieves 82.5\% top-1 accuracy, 1.2\% and 0.96\% higher than Swin-T and LeViT-256, respectively. In addition, when deployed on mobile device, MathNAS achieves real-time search and dynamic network switching within 1s (0.4s on TX2 GPU), surpassing baseline dynamic networks in on-device performance.</details>

### [Spotlight] Max-Margin Token Selection in Attention Mechanism

**Authors:** Davoud Ataee Tarzanagh, Yingcong Li, Xuechen Zhang, Samet Oymak

**<details><summary>Abstract**</summary>

Attention mechanism is a central component of the transformer architecture which led to the phenomenal success of large language models. However, the theoretical principles underlying the attention mechanism are poorly understood, especially its nonconvex optimization dynamics. In this work, we explore the seminal softmax-attention model $f(X)=\langle Xv, \texttt{softmax}(XWp)\rangle$, where $X$ is the token sequence and $(v,W,p)$ are trainable parameters. We prove that running gradient descent on $p$, or equivalently $W$, converges in direction to a max-margin solution that separates *locally-optimal* tokens from non-optimal ones. This clearly formalizes attention as an optimal token selection mechanism. Remarkably, our results are applicable to general data and precisely characterize *optimality* of tokens in terms of the value embeddings $Xv$ and problem geometry. We also provide a broader regularization path analysis that establishes the margin maximizing nature of attention even for nonlinear prediction heads. When optimizing $v$ and $p$ simultaneously with logistic loss, we identify conditions under which the regularization paths directionally converge to their respective hard-margin SVM solutions where $v$ separates the input features based on their labels. Interestingly, the SVM formulation of $p$ is influenced by the support vector geometry of $v$. Finally, we verify our theoretical findings via numerical experiments and provide insights.</details>

### Max-Sliced Mutual Information

**Authors:** Dor Tsur, Ziv Goldfeld, Kristjan Greenewald

**<details><summary>Abstract**</summary>

Quantifying dependence between high-dimensional random variables is central to statistical learning and inference. Two classical methods are canonical correlation analysis (CCA), which identifies maximally correlated projected versions of the original variables, and Shannon's mutual information, which is a universal dependence measure that also captures high-order dependencies. However, CCA only accounts for linear dependence, which may be insufficient for certain applications, while mutual information is often infeasible to compute/estimate in high dimensions. This work proposes a middle ground in the form of a scalable information-theoretic generalization of CCA, termed max-sliced mutual information (mSMI). mSMI equals the maximal mutual information between low-dimensional projections of the high-dimensional variables, which reduces back to CCA in the Gaussian case. It enjoys the best of both worlds: capturing intricate dependencies in the data while being amenable to fast computation and scalable estimation from samples. We show that mSMI retains favorable structural properties of Shannon's mutual information, like variational forms and identification of independence. We then study statistical estimation of mSMI, propose an efficiently computable neural estimator, and couple it with formal non-asymptotic error bounds. We present experiments that demonstrate the utility of mSMI for several tasks, encompassing independence testing, multi-view representation learning, algorithmic fairness, and generative modeling. We observe that mSMI consistently outperforms competing methods with little-to-no computational overhead.</details>

### Maximum Average Randomly Sampled: A Scale Free and Non-parametric Algorithm for Stochastic Bandits

**Authors:** Masoud Moravej Khorasani, Erik Weyer

**<details><summary>Abstract**</summary>

Upper Confidence Bound (UCB) methods are one of the most effective methods in dealing with the exploration-exploitation trade-off in online decision-making problems. The confidence bounds utilized in UCB methods tend to be constructed based on concentration equalities which are usually dependent on a parameter of scale (e.g. a bound on the payoffs, a variance, or a subgaussian parameter) that must be known in advance. The necessity of knowing a scale parameter a priori and the fact that the confidence bounds only use the tail information can deteriorate the performance of the UCB methods.Here we propose a data-dependent UCB algorithm called MARS (Maximum Average Randomly Sampled) in a non-parametric setup for multi-armed bandits with symmetric rewards. The algorithm does not depend on any scaling, and the data-dependent upper confidence bound is constructed based on the maximum average of randomly sampled rewards inspired by the work of Hartigan in the 1960s and 70s. A regret bound for the multi-armed bandit problem is derived under the same assumptions as for the $\psi$-UCB method without incorporating any correction factors. The method is illustrated and compared with baseline algorithms in numerical experiments.</details>

### [Oral] Mesogeos: A multi-purpose dataset for data-driven wildfire modeling in the Mediterranean

**Authors:** Spyridon Kondylatos, Ioannis Prapas, Gustau Camps-Valls, Ioannis Papoutsis

**Oral Presentation:** We, Dec 13, 13:30 -- Oral 4B

**<details><summary>Abstract**</summary>

We introduce Mesogeos, a large-scale multi-purpose dataset for wildfire modeling in the Mediterranean. Mesogeos integrates variables representing wildfire drivers (meteorology, vegetation, human activity) and historical records of wildfire ignitions and burned areas for 17 years (2006-2022). It is designed as a cloud-friendly spatio-temporal dataset, namely a datacube, harmonizing all variables in a grid of 1km x 1km x 1-day resolution. The datacube structure offers opportunities to assess machine learning (ML) usage in various wildfire modeling tasks. We extract two ML-ready datasets that establish distinct tracks to demonstrate this potential: (1) short-term wildfire danger forecasting and (2) final burned area estimation given the point of ignition. We define appropriate metrics and baselines to evaluate the performance of models in each track. By publishing the datacube, along with the code to create the ML datasets and models, we encourage the community to foster the implementation of additional tracks for mitigating the increasing threat of wildfires in the Mediterranean.</details>

### Meta-AdaM: An Meta-Learned Adaptive Optimizer with Momentum for Few-Shot Learning

**Authors:** Siyuan Sun, Hongyang Gao

**<details><summary>Abstract**</summary>

We introduce Meta-AdaM, a meta-learned adaptive optimizer with momentum, designed for few-shot learning tasks that pose significant challenges to deep learning models due to the limited number of labeled examples. Meta-learning has been successfully employed to address these challenges by transferring meta-learned prior knowledge to new tasks. Most existing works focus on meta-learning an optimal model initialization or an adaptive learning rate learner for rapid convergence. However, these approaches either neglect to consider weight-update history for the adaptive learning rate learner or fail to effectively integrate momentum for fast convergence, as seen in many-shot learning settings. To tackle these limitations, we propose a meta-learned learning rate learner that utilizes weight-update history as input to predict more appropriate learning rates for rapid convergence. Furthermore, for the first time, our approach incorporates momentum into the optimization process of few-shot learning via a double look-ahead mechanism, enabling rapid convergence similar to many-shot settings. Extensive experimental results on benchmark datasets demonstrate the effectiveness of the proposed Meta-AdaM.</details>

### Meta-Adapter: An Online Few-shot Learner for Vision-Language Model

**Authors:** cheng cheng, Lin Song, Ruoyi Xue, Hang Wang, Hongbin Sun, Yixiao Ge, Ying Shan

**<details><summary>Abstract**</summary>

The contrastive vision-language pre-training, known as CLIP, demonstrates remarkable potential in perceiving open-world visual concepts, enabling effective zero-shot image recognition.  Nevertheless, few-shot learning methods based on CLIP typically require offline fine-tuning of the parameters on few-shot samples, resulting in longer inference time and the risk of overfitting in certain domains.  To tackle these challenges, we propose the Meta-Adapter, a lightweight residual-style adapter, to refine the CLIP features guided by the few-shot samples in an online manner.  With a few training samples, our method can enable effective few-shot learning capabilities and generalize to unseen data or tasks without additional fine-tuning, achieving competitive performance and high efficiency.  Without bells and whistles, our approach outperforms the state-of-the-art online few-shot learning method by an average of 3.6\% on eight image classification datasets with higher inference speed.  Furthermore, our model is simple and flexible, serving as a plug-and-play module directly applicable to downstream tasks.  Without further fine-tuning, Meta-Adapter obtains notable performance improvements in open-vocabulary object detection and segmentation tasks.</details>

### Meta-Learning with Neural Bandit Scheduler

**Authors:** Yunzhe Qi, Yikun Ban, Tianxin Wei, Jiaru Zou, Huaxiu Yao, Jingrui He

**<details><summary>Abstract**</summary>

Meta-learning has been proven an effective learning paradigm for training machine learning models with good generalization ability. Apart from the common practice of uniformly sampling the meta-training tasks, existing methods working on task scheduling strategies are mainly based on pre-defined sampling protocols or the assumed task-model correlations, and greedily make scheduling decisions, which can lead to sub-optimal performance bottlenecks of the meta-model. In this paper, we propose a novel task scheduling framework under Contextual Bandits settings, named BASS, which directly optimizes the task scheduling strategy based on the status of the meta-model. By balancing the exploitation and exploration in meta-learning task scheduling, BASS can help tackle the challenge of limited knowledge about the task distribution during the early stage of meta-training, while simultaneously exploring potential benefits for forthcoming meta-training iterations through an adaptive exploration strategy. Theoretical analysis and extensive experiments are presented to show the effectiveness of our proposed framework.</details>

### Minimax Risks and Optimal Procedures for Estimation under Functional Local Differential Privacy

**Authors:** Bonwoo Lee, Jeongyoun Ahn, Cheolwoo Park

**<details><summary>Abstract**</summary>

As concerns about data privacy continue to grow, differential privacy (DP) has emerged as a fundamental concept that aims to guarantee privacy by ensuring individuals' indistinguishability in data analysis. Local differential privacy (LDP) is a rigorous type of DP that requires individual data to be privatized before being sent to the collector, thus removing the need for a trusted third party to collect data. Among the numerous (L)DP-based approaches, functional DP has gained considerable attention in the DP community because it connects DP to statistical decision-making by formulating it as a hypothesis-testing problem and also exhibits Gaussian-related properties. However, the utility of privatized data is generally lower than that of non-private data, prompting research into optimal mechanisms that maximize the statistical utility for given privacy constraints. In this study, we investigate how functional LDP preserves the statistical utility by analyzing minimax risks of univariate mean estimation as well as nonparametric density estimation. We leverage the contraction property of functional LDP mechanisms and classical information-theoretical bounds to derive private minimax lower bounds. Our theoretical study reveals that it is possible to establish an interpretable, continuous balance between the statistical utility and privacy level, which has not been achieved under the $\epsilon$-LDP framework. Furthermore, we suggest minimax optimal mechanisms based on Gaussian LDP (a type of functional LDP) that achieve the minimax upper bounds and show via a numerical study that they are superior to the counterparts derived under $\epsilon$-LDP. The theoretical and empirical findings of this work suggest that Gaussian LDP should be considered a reliable standard for LDP.</details>

### Mirror Diffusion Models for Constrained and Watermarked Generation

**Authors:** Guan-Horng Liu, Tianrong Chen, Evangelos Theodorou, Molei Tao

**<details><summary>Abstract**</summary>

Modern successes of diffusion models in learning complex, high-dimensional data distributions are attributed, in part, to their capability to construct diffusion processes with analytic transition kernels and score functions. The tractability results in a simulation-free framework with stable regression losses, from which reversed, generative processes can be learned at scale. However, when data is confined to a constrained set as opposed to a standard Euclidean space, these desirable characteristics appear to be lost based on prior attempts. In this work, we propose Mirror Diffusion Models (MDM), a new class of diffusion models that generate data on convex constrained sets without losing any tractability. This is achieved by learning diffusion processes in a dual space constructed from a mirror map, which, crucially, is a standard Euclidean space. We derive efficient computation of mirror maps for popular constrained sets, such as simplices and $\ell_2$-balls, showing significantly improved performance of MDM over existing methods. For safety and privacy purposes, we also explore constrained sets as a new mechanism to embed invisible but quantitative information (i.e., watermarks) in generated data, for which MDM serves as a compelling approach. Our work brings new algorithmic opportunities for learning tractable diffusion on complex domains.</details>

### Mixed-Initiative Multiagent Apprenticeship Learning for Human Training of Robot Teams

**Authors:** Esmaeil Seraj, Jerry Xiong, Mariah Schrum, Matthew Gombolay

**<details><summary>Abstract**</summary>

Extending recent advances in Learning from Demonstration (LfD) frameworks to multi-robot settings poses critical challenges such as environment non-stationarity due to partial observability which is detrimental to the applicability of existing methods. Although prior work has shown that enabling communication among agents of a robot team can alleviate such issues, creating inter-agent communication under existing Multi-Agent LfD (MA-LfD) frameworks requires the human expert to provide demonstrations for both environment actions and communication actions, which necessitates an efficient communication strategy on a known message spaces. To address this problem, we propose Mixed-Initiative Multi-Agent Apprenticeship Learning (MixTURE). MixTURE enables robot teams to learn from a human expert-generated data a preferred policy to accomplish a collaborative task, while simultaneously learning emergent inter-agent communication to enhance team coordination. The key ingredient to MixTURE's success is automatically learning a communication policy, enhanced by a mutual-information maximizing reverse model that rationalizes the underlying expert demonstrations without the need for human generated data or an auxiliary reward function. MixTURE outperforms a variety of relevant baselines on diverse data generated by human experts in complex heterogeneous domains. MixTURE is the first MA-LfD framework to enable learning multi-robot collaborative policies directly from real human data, resulting in ~44% less human workload, and ~46% higher usability score.</details>

### Mixture Weight Estimation and Model Prediction in Multi-source Multi-target Domain Adaptation

**Authors:** Yuyang Deng, Ilja Kuzborskij, Mehrdad Mahdavi

**<details><summary>Abstract**</summary>

We consider a problem of learning a model from multiple sources with the goal to performwell on a new target distribution.  Such problem arises inlearning with data collected from multiple sources (e.g. crowdsourcing) orlearning in distributed systems, where the data can be highly heterogeneous. Thegoal of learner is to mix these data sources in a target-distribution aware way andsimultaneously minimize the empirical risk on the mixed source.  The literature has made some tangible advancements in establishingtheory of learning on mixture domain.  However, there are still two unsolved problems. Firstly, how to estimate the optimal mixture of sources, given a target domain; Secondly, when there are numerous target domains, we have to solve empirical risk minimization for each target on possibly unique mixed source data , which is computationally expensive. In this paper we address both problems efficiently and with guarantees.We cast the first problem, mixture weight estimation as convex-nonconcave compositional minimax, and propose an efficient stochasticalgorithm with provable stationarity guarantees.Next, for the second problem, we identify that for certain regime,solving ERM for each target domain individually can be avoided, and instead parameters for a target optimalmodel can be viewed as a non-linear function ona space of the mixture coefficients.To this end, we show that in offline setting, a GD-trained overparameterized neural network can provably learn such function.Finally, we also consider an online setting and propose an label efficient online algorithm, which predicts parameters for new models given arbitrary sequence of mixing coefficients, while enjoying optimal regret.</details>

### MoVie: Visual Model-Based Policy Adaptation for View Generalization

**Authors:** Sizhe Yang, Yanjie Ze, Huazhe Xu

**<details><summary>Abstract**</summary>

Visual Reinforcement Learning (RL) agents trained on limited views face significant challenges in generalizing their learned abilities to unseen views. This inherent difficulty is known as the problem of $	extit{view generalization}$. In this work, we systematically categorize this fundamental problem into four distinct and highly challenging scenarios that closely resemble real-world situations. Subsequently, we propose a straightforward yet effective approach to enable successful adaptation of visual $	extbf{Mo}$del-based policies for $	extbf{Vie}$w generalization ($	extbf{MoVie}$) during test time, without any need for explicit reward signals and any modification during training time. Our method demonstrates substantial advancements across all four scenarios encompassing a total of $	extbf{18}$ tasks sourced from DMControl, xArm, and Adroit, with a relative improvement of $\mathbf{33}$%, $\mathbf{86}$%, and $\mathbf{152}$% respectively. The superior results highlight the immense potential of our approach for real-world robotics applications. Code and videos are available at https://yangsizhe.github.io/MoVie/.</details>

### Modality-Agnostic Self-Supervised Learning with Meta-Learned Masked Auto-Encoder

**Authors:** Huiwon Jang, Jihoon Tack, Daewon Choi, Jongheon Jeong, Jinwoo Shin

**<details><summary>Abstract**</summary>

Despite its practical importance across a wide range of modalities, recent advances in self-supervised learning (SSL) have been primarily focused on a few well-curated domains, e.g., vision and language, often relying on their domain-specific knowledge. For example, Masked Auto-Encoder (MAE) has become one of the popular architectures in these domains, but less has explored its potential in other modalities. In this paper, we develop MAE as a unified, modality-agnostic SSL framework. In turn, we argue meta-learning as a key to interpreting MAE as a modality-agnostic learner, and propose enhancements to MAE from the motivation to jointly improve its SSL across diverse modalities, coined MetaMAE as a result. Our key idea is to view the mask reconstruction of MAE as a meta-learning task: masked tokens are predicted by adapting the Transformer meta-learner through the amortization of unmasked tokens. Based on this novel interpretation, we propose to integrate two advanced meta-learning techniques. First, we adapt the amortized latent of the Transformer encoder using gradient-based meta-learning to enhance the reconstruction. Then, we maximize the alignment between amortized and adapted latents through task contrastive learning which guides the Transformer encoder to better encode the task-specific knowledge. Our experiment demonstrates the superiority of MetaMAE in the modality-agnostic SSL benchmark (called DABS), significantly outperforming prior baselines.</details>

### Model Shapley: Equitable Model Valuation with Black-box Access

**Authors:** Xinyi Xu, Thanh Lam, Chuan Sheng Foo, Bryan Kian Hsiang Low

**<details><summary>Abstract**</summary>

Valuation methods of data and machine learning (ML) models are essential to the establishment of AI marketplaces. Importantly, certain practical considerations (e.g., operational constraints, legal restrictions) favor the use of model valuation over data valuation. Also, existing marketplaces that involve trading of pre-trained ML models call for an equitable model valuation method to price them. In particular, we investigate the black-box access setting which allows querying a model (to observe predictions) without disclosing model-specific information (e.g., architecture and parameters). By exploiting a Dirichlet abstraction of a model’s predictions, we propose a novel and equitable model valuation method called model Shapley. We also leverage a Lipschitz continuity of model Shapley to design a learning approach for predicting the model Shapley values (MSVs) of many vendors’ models (e.g., 150) in a large-scale marketplace. We perform extensive empirical validation on the effectiveness of model Shapley using various real-world datasets and heterogeneous model types.</details>

### Model-Based Reparameterization Policy Gradient Methods: Theory and Practical Algorithms

**Authors:** Shenao Zhang, Boyi Liu, Zhaoran Wang, Tuo Zhao

**<details><summary>Abstract**</summary>

ReParameterization (RP) Policy Gradient Methods (PGMs) have been widely adopted for continuous control tasks in robotics and computer graphics. However, recent studies have revealed that, when applied to long-term reinforcement learning problems, model-based RP PGMs may experience chaotic and non-smooth optimization landscapes with exploding gradient variance, which leads to slow convergence. This is in contrast to the conventional belief that reparameterization methods have low gradient estimation variance in problems such as training deep generative models. To comprehend this phenomenon, we conduct a theoretical examination of model-based RP PGMs and search for solutions to the optimization difficulties. Specifically, we analyze the convergence of the model-based RP PGMs and pinpoint the smoothness of function approximators as a major factor that affects the quality of gradient estimation. Based on our analysis, we propose a spectral normalization method to mitigate the exploding variance issue caused by long model unrolls. Our experimental results demonstrate that proper normalization significantly reduces the gradient variance of model-based RP PGMs. As a result, the performance of the proposed method is comparable or superior to other gradient estimators, such as the Likelihood Ratio (LR) gradient estimator. Our code is available at https://github.com/agentification/RP_PGM.</details>

### Model-enhanced Vector Index

**Authors:** Hailin Zhang, Yujing Wang, Qi Chen, Ruiheng Chang, Ting Zhang, Ziming Miao, Yingyan Hou, Yang Ding, Xupeng Miao, Haonan Wang, Bochen Pang, Yuefeng Zhan, Hao Sun, Weiwei Deng, Qi Zhang, Fan Yang, Xing Xie, Mao Yang, Bin CUI

**<details><summary>Abstract**</summary>

Embedding-based retrieval methods construct vector indices to search for document representations that are most similar to the query representations. They are widely used in document retrieval due to low latency and decent recall performance. Recent research indicates that deep retrieval solutions offer better model quality, but are hindered by unacceptable serving latency and the inability to support document updates. In this paper, we aim to enhance the vector index with end-to-end deep generative models, leveraging the differentiable advantages of deep retrieval models while maintaining desirable serving efficiency. We propose Model-enhanced Vector Index (MEVI), a differentiable model-enhanced index empowered by a twin-tower representation model. MEVI leverages a Residual Quantization (RQ) codebook to bridge the sequence-to-sequence deep retrieval and embedding-based models. To substantially reduce the inference time, instead of decoding the unique document ids in long sequential steps, we first generate some semantic virtual cluster ids of candidate documents in a small number of steps, and then leverage the well-adapted embedding vectors to further perform a fine-grained search for the relevant documents in the candidate virtual clusters. We empirically show that our model achieves better performance on the commonly used academic benchmarks MSMARCO Passage and Natural Questions, with comparable serving latency to dense retrieval solutions.</details>

### MomentDiff: Generative Video Moment Retrieval from Random to Real

**Authors:** Pandeng Li, Chen-Wei Xie, Hongtao Xie, Liming Zhao, Lei Zhang, Yun Zheng, Deli Zhao, Yongdong Zhang

**<details><summary>Abstract**</summary>

Video moment retrieval pursues an efficient and generalized solution to identify the specific temporal segments within an untrimmed video that correspond to a given language description.To achieve this goal, we provide a generative diffusion-based framework called MomentDiff, which simulates a typical human retrieval process from random browsing to gradual localization.Specifically, we first diffuse the real span to random noise, and learn to denoise the random noise to the original span with the guidance of similarity between text and video.This allows the model to learn a mapping from arbitrary random locations to real moments, enabling the ability to locate segments from random initialization.Once trained, MomentDiff could sample random temporal segments as initial guesses and iteratively refine them to generate an accurate temporal boundary.Different from discriminative works (e.g., based on learnable proposals or queries), MomentDiff with random initialized spans could resist the temporal location biases from datasets.To evaluate the influence of the temporal location biases, we propose two ``anti-bias'' datasets with location distribution shifts, named Charades-STA-Len and Charades-STA-Mom.The experimental results demonstrate that our efficient framework consistently outperforms state-of-the-art methods on three public benchmarks, and exhibits better generalization and robustness on the proposed anti-bias datasets. The code, model, and anti-bias evaluation datasets will be released publicly.</details>

### MotionGPT: Human Motion as a Foreign Language

**Authors:** Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, Tao Chen

**<details><summary>Abstract**</summary>

Though the advancement of pre-trained large language models unfolds, the exploration of building a unified model for language and other multimodal data, such as motion, remains challenging and untouched so far. Fortunately, human motion displays a semantic coupling akin to human language, often perceived as a form of body language. By fusing language data with large-scale motion models, motion-language pre-training that can enhance the performance of motion-related tasks becomes feasible. Driven by this insight, we propose MotionGPT, a unified, versatile, and user-friendly motion-language model to handle multiple motion-relevant tasks. Specifically, we employ the discrete vector quantization for human motion and transfer 3D motion into motion tokens, similar to the generation process of word tokens. Building upon this "motion vocabulary", we perform language modeling on both motion and text in a unified manner, treating human motion as a specific language. Moreover, inspired by prompt learning, we pre-train MotionGPT with a mixture of motion-language data and fine-tune it on prompt-based question-and-answer tasks. Extensive experiments demonstrate that MotionGPT achieves state-of-the-art performances on multiple motion tasks including text-driven motion generation, motion captioning, motion prediction, and motion in-between.</details>

### Multi-Agent First Order Constrained Optimization in Policy Space

**Authors:** Youpeng Zhao, Yaodong Yang, Zhenbo Lu, Wengang Zhou, Houqiang Li

**<details><summary>Abstract**</summary>

In the realm of multi-agent reinforcement learning (MARL), achieving high performance is crucial for a successful multi-agent system.Meanwhile, the ability to avoid unsafe actions is becoming an urgent and imperative problem to solve for real-life applications. Whereas, it is still challenging to develop a safety-aware method for multi-agent systems in MARL. In this work, we introduce a novel approach called Multi-Agent First Order Constrained Optimization in Policy Space (MAFOCOPS), which effectively addresses the dual objectives of attaining satisfactory performance and enforcing safety constraints. Using data generated from the current policy, MAFOCOPS first finds the optimal update policy by solving a constrained optimization problem in the nonparameterized policy space. Then, the update policy is projected back into the parametric policy space to achieve a feasible policy. Notably, our method is first-order in nature, ensuring the ease of implementation, and exhibits an approximate upper bound on the worst-case constraint violation. Empirical results show that our approach achieves remarkable performance while satisfying safe constraints on several safe MARL benchmarks.</details>

### Multi-Prompt Alignment for Multi-Source Unsupervised Domain Adaptation

**Authors:** Haoran Chen, Xintong Han, Zuxuan Wu, Yu-Gang Jiang

**<details><summary>Abstract**</summary>

Most existing methods for unsupervised domain adaptation (UDA) rely on a shared network to extract domain-invariant features. However, when facing multiple source domains, optimizing such a network involves updating the parameters of the entire network, making it both computationally expensive and challenging, particularly when coupled with min-max objectives. Inspired by recent advances in prompt learning that adapts high-capacity models for downstream tasks in a computationally economic way, we introduce Multi-Prompt Alignment (MPA), a simple yet efficient framework for multi-source UDA. Given a source and target domain pair, MPA first trains an individual prompt to minimize the domain gap through a contrastive loss. Then, MPA denoises the learned prompts through an auto-encoding process and aligns them by maximizing the agreement of all the reconstructed prompts. Moreover, we show that the resulting subspace acquired from the auto-encoding process can easily generalize to a streamlined set of target domains, making our method more efficient for practical usage. Extensive experiments show that MPA achieves state-of-the-art results on three popular datasets with an impressive average accuracy of 54.1% on DomainNet.</details>

### Multi-task Graph Neural Architecture Search with Task-aware Collaboration and Curriculum

**Authors:** Yijian Qin, Xin Wang, Ziwei Zhang, Hong Chen, Wenwu Zhu

**<details><summary>Abstract**</summary>

Graph neural architecture search (GraphNAS) has shown great potential for automatically designing graph neural architectures for graph related tasks. However, multi-task GraphNAS capable of handling multiple tasks simultaneously has been largely unexplored in literature, posing great challenges to capture the complex relations and influences among different tasks. To tackle this problem, we propose a novel multi-task graph neural architecture search with task-aware collaboration and curriculum (MTGC3), which is able to simultaneously discover optimal architectures for different tasks and learn the collaborative relationships among different tasks in a joint manner. Specifically, we design the layer-wise disentangled supernet capable of managing multiple architectures in a unified framework, which combines with our proposed soft task-collaborative module to learn the transferability relationships between tasks. We further develop the task-wise curriculum training strategy to improve the architecture search procedure via reweighing the influence of different tasks based on task difficulties. Extensive experiments show that our proposed MTGC3 model achieves state-of-the-art performance against several baselines in multi-task scenarios, demonstrating its ability to discover effective architectures and capture the collaborative relationships for multiple tasks.</details>

### Multinomial Logistic Regression: Asymptotic Normality on Null Covariates in High-Dimensions

**Authors:** Kai Tan, Pierre C Bellec

**<details><summary>Abstract**</summary>

This paper investigates the asymptotic distribution of the maximum-likelihood estimate (MLE) in multinomial logistic models in the high-dimensional regime where dimension and sample size are of the same order. While classical large-sample theory provides asymptotic normality of the MLE under certain conditions, such classical results are expected to fail in high-dimensions as documented for the binary logistic case in the seminal work of Sur and Candès [2019]. We address this issue in classification problems with 3 or more classes, by developing asymptotic normality and asymptotic chi-square results for the multinomial logistic MLE (also known as cross-entropy minimizer) on null covariates. Our theory leads to a new methodology to test the significance of a given feature. Extensive simulation studies on synthetic data corroborate these asymptotic results and confirm the validity of proposed p-values for testing the significance of a given feature.</details>

### Multiplication-Free Transformer Training via Piecewise Affine Operations

**Authors:** Atli Kosson, Martin Jaggi

**<details><summary>Abstract**</summary>

Multiplications are responsible for most of the computational cost involved in neural network training and inference. Recent research has thus looked for ways to reduce the cost associated with them. Inspired by Mogami 2020, we replace multiplication with a cheap piecewise affine approximation that is achieved by adding the bit representation of the floating point numbers together as integers. We show that transformers can be trained with the resulting modified matrix multiplications on both vision and language tasks with little to no performance impact, and without changes to the training hyperparameters. We further replace all non-linearities in the networks making them fully and jointly piecewise affine in both inputs and weights. Finally, we show that we can eliminate all multiplications in the entire training process, including operations in the forward pass, backward pass and optimizer update, demonstrating the first successful training of modern neural network architectures in a fully multiplication-free fashion.</details>

### NAP: Neural 3D Articulated Object Prior

**Authors:** Jiahui Lei, Congyue Deng, William B Shen, Leonidas Guibas, Kostas Daniilidis

**<details><summary>Abstract**</summary>

We propose Neural 3D Articulated object Prior (NAP), the first 3D deep generative model to synthesize 3D articulated object models. Despite the extensive research on generating 3D static objects, compositions, or scenes, there are hardly any approaches on capturing the distribution of articulated objects, a common object category for human and robot interaction. To generate articulated objects, we first design a novel articulation tree/graph parameterization and then apply a diffusion-denoising probabilistic model over this representation where articulated objects can be generated via denoising from random complete graphs. In order to capture both the geometry and the motion structure whose distribution will affect each other, we design a graph denoising network for learning the reverse diffusion process. We propose a novel distance that adapts widely used 3D generation metrics to our novel task to evaluate generation quality. Experiments demonstrate our high performance in articulated object generation as well as its applications on conditioned generation, including Part2Motion, PartNet-Imagination, Motion2Part, and GAPart2Object.</details>

### NCDL:  A Framework for Deep Learning on non-Cartesian Lattices

**Authors:** Joshua Horacsek, Usman Alim

**<details><summary>Abstract**</summary>

The use of non-Cartesian grids is a niche but important topic in sub-fields of the numerical sciences such as simulation and scientific visualization. However, non-Cartesian approaches are virtually unexplored in machine learning. This is likely due to the difficulties in the representation of data on non-Cartesian domains and the lack of support for standard machine learning operations on non-Cartesian data. This paper proposes a new data structure called the lattice tensor which generalizes traditional tensor spatio-temporal operations to lattice tensors, enabling the use of standard machine learning algorithms on non-Cartesian data. However, data need not reside on a non-Cartesian structure, we use non-Dyadic downsampling schemes to bring Cartesian data into a non-Cartesian space for further processing.   We introduce a software library that implements the lattice tensor container (with some common machine learning operations), and   demonstrate its effectiveness. Our method provides a general framework for machine learning on non-Cartesian domains, addressing the challenges mentioned above and filling a gap in the current literature.</details>

### NEO-KD: Knowledge-Distillation-Based Adversarial Training for Robust Multi-Exit Neural Networks

**Authors:** Seokil Ham, Jungwuk Park, Dong-Jun Han, Jaekyun Moon

**<details><summary>Abstract**</summary>

While multi-exit neural networks are regarded as a promising solution for making efficient inference via early exits, combating adversarial attacks remains a challenging problem. In multi-exit networks, due to the high dependency among different submodels, an adversarial example targeting a specific exit not only degrades the performance of the target exit but also reduces the performance of all other exits concurrently. This makes multi-exit networks highly vulnerable to simple adversarial attacks. In this paper, we propose NEO-KD, a knowledge-distillation-based adversarial training strategy that tackles this fundamental challenge based on two key contributions. NEO-KD first resorts to neighbor knowledge distillation to guide the output of the adversarial examples to tend to the ensemble outputs of neighbor exits of clean data. NEO-KD also employs exit-wise orthogonal knowledge distillation for reducing adversarial transferability across different submodels. The result is a significantly improved robustness against adversarial attacks. Experimental results on various datasets/models show that our method achieves the best adversarial accuracy with reduced computation budgets, compared to the baselines relying on existing adversarial training or knowledge distillation techniques for multi-exit networks.</details>

### NICE: NoIse-modulated Consistency rEgularization for Data-Efficient GANs

**Authors:** Yao Ni, Piotr Koniusz

**<details><summary>Abstract**</summary>

Generative Adversarial Networks (GANs) are powerful tools for image synthesis. However, they require access to vast amounts of training data, which is often costly and prohibitive. Limited data affects GANs, leading to discriminator overfitting and training instability. In this paper, we present a novel approach called NoIse-modulated Consistency rEgularization (NICE) to overcome these challenges. To this end, we introduce an adaptive multiplicative noise into the discriminator to modulate its latent features. We demonstrate the effectiveness of such a modulation in preventing discriminator overfitting by adaptively reducing the Rademacher complexity of the discriminator. However, this modulation leads to an unintended consequence of increased gradient norm, which can undermine the stability of GAN training. To mitigate this undesirable effect, we impose a constraint on the discriminator, ensuring its consistency for the same inputs under different noise modulations. The constraint effectively penalizes the first and second-order gradients of latent features, enhancing GAN stability. Experimental evidence aligns with our theoretical analysis, demonstrating the reduction of generalization error and gradient penalization of NICE. This substantiates the efficacy of NICE in reducing discriminator overfitting and improving stability of GAN training. NICE achieves state-of-the-art results on CIFAR-10, CIFAR-100, ImageNet and FFHQ datasets when trained with limited data, as well as in low-shot generation tasks.</details>

### Natural Language Instruction-following with Task-related Language Development and Translation

**Authors:** Jing-Cheng Pang, Xin-Yu Yang, Si-Hang Yang, Xiong-Hui Chen, Yang Yu

**<details><summary>Abstract**</summary>

Natural language-conditioned reinforcement learning (RL) enables agents to follow human instructions. Previous approaches generally implemented language-conditioned RL by providing the policy with human instructions in natural language (NL) and training the policy to follow instructions. In this is outside-in approach, the policy must comprehend the NL and manage the task simultaneously. However, the unbounded NL examples often bring much extra complexity for solving concrete RL tasks, which can distract policy learning from completing the task. To ease the learning burden of the policy, we investigate an inside-out scheme for natural language-conditioned RL by developing a task language (TL) that is task-related and easily understood by the policy, thus reducing the policy learning burden. Besides, we employ a translator to translate natural language into the TL, which is used in RL to achieve efficient policy training. We implement this scheme as TALAR (TAsk Language with predicAte Representation) that learns multiple predicates to model object relationships as the TL. Experiments indicate that TALAR not only better comprehends NL instructions but also leads to a better instruction-following policy that significantly improves the success rate over baselines and adapts to unseen expressions of NL instruction. Besides, the TL is also an effective sub-task abstraction compatible with hierarchical RL.</details>

### Near-optimal learning with average Hölder smoothness

**Authors:** Guy Kornowski, Steve Hanneke, Aryeh Kontorovich

**<details><summary>Abstract**</summary>

We generalize the notion of average Lipschitz smoothness proposed by Ashlagi et al. (COLT 2021) by extending it to Hölder smoothness. This measure of the "effective smoothness" of a function is sensitive to the underlying distribution and can be dramatically smaller than its classic "worst-case" Hölder constant.We consider both the realizable and the agnostic (noisy) regression settings, proving upper and lower risk bounds in terms of the average Hölder smoothness; these rates improve upon both previously known rates even in the special case of average Lipschitz smoothness.Moreover, our lower bound is tight in the realizable setting up to log factors, thus we establish the minimax rate.From an algorithmic perspective, since our notion of average smoothness is defined with respect to the unknown underlying distribution, the learner does not have an explicit representation of the function class, hence is unable to execute ERM. Nevertheless, we provide distinct learning algorithms that achieve both (nearly) optimal learning rates.Our results hold in any totally bounded metric space, and are stated in terms of its intrinsic geometry.Overall, our results show that the classic worst-case notion of Hölder smoothness can be essentially replaced by its average, yielding considerably sharper guarantees.</details>

### Nearly Optimal Bounds for Cyclic Forgetting

**Authors:** William Swartworth, Deanna Needell, Rachel Ward, Mark Kong, Halyun Jeong

**<details><summary>Abstract**</summary>

We provide theoretical bounds on the forgetting quantity in the continual learning setting for linear tasks, where each round of learning corresponds to projecting onto a linear subspace. For a cyclic task ordering on $T$ tasks repeated $m$ times each, we prove the best known upper bound of $O(T^2/m)$ on the forgetting. Notably, our bound holds uniformly over all choices of tasks and is independent of the ambient dimension. Our main technical contribution is a characterization of the union of all numerical ranges of products of $T$ (real or complex) projections as a sinusoidal spiral, which may be of independent interest.</details>

### Nearly Optimal VC-Dimension and Pseudo-Dimension Bounds for Deep Neural Network Derivatives

**Authors:** Yahong Yang, Haizhao Yang, Yang Xiang

**<details><summary>Abstract**</summary>

This paper addresses the problem of  nearly optimal Vapnik--Chervonenkis dimension (VC-dimension) and pseudo-dimension estimations of the derivative functions of deep neural networks (DNNs). Two important applications of these estimations include: 1) Establishing a nearly tight approximation result of DNNs in the Sobolev space; 2)  Characterizing the generalization error of machine learning methods with loss functions involving function derivatives. This theoretical investigation fills the gap of learning error estimations for a wide range of physics-informed machine learning models and applications including generative models, solving partial differential equations, operator learning, network compression, distillation, regularization, etc.</details>

### Necessary and Sufficient Conditions for Optimal Decision Trees using Dynamic Programming

**Authors:** Jacobus van der Linden, Mathijs de Weerdt, Emir Demirović

**<details><summary>Abstract**</summary>

Global optimization of decision trees has shown to be promising in terms of accuracy, size, and consequently human comprehensibility. However, many of the methods used rely on general-purpose solvers for which scalability remains an issue.Dynamic programming methods have been shown to scale much better because they exploit the tree structure by solving subtrees as independent subproblems. However, this only works when an objective can be optimized separately for subtrees.We explore this relationship in detail and show necessary and sufficient conditions for such separability and generalize previous dynamic programming approaches into a framework that can optimize any combination of separable objectives and constraints.Experiments on five application domains show the general applicability of this framework, while outperforming the scalability of general-purpose solvers by a large margin.</details>

### Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization

**Authors:** Fu Luo, Xi Lin, Fei Liu, Qingfu Zhang, Zhenkun Wang

**<details><summary>Abstract**</summary>

Neural combinatorial optimization (NCO) is a promising learning-based approach for solving challenging combinatorial optimization problems without specialized algorithm design by experts. However, most constructive NCO methods cannot solve problems with large-scale instance sizes, which significantly diminishes their usefulness for real-world applications. In this work, we propose a novel Light Encoder and Heavy Decoder (LEHD) model with a strong generalization ability to address this critical issue. The LEHD model can learn to dynamically capture the relationships between all available nodes of varying sizes, which is beneficial for model generalization to problems of various scales. Moreover, we develop a data-efficient training scheme and a flexible solution construction mechanism for the proposed LEHD model. By training on small-scale problem instances, the LEHD model can generate nearly optimal solutions for the Travelling Salesman Problem (TSP) and the Capacitated Vehicle Routing Problem (CVRP) with up to 1000 nodes, and also generalizes well to solve real-world TSPLib and CVRPLib problems. These results confirm our proposed LEHD model can significantly improve the state-of-the-art performance for constructive NCO.</details>

### [Spotlight] Neural Injective Functions for Multisets, Measures and Graphs via a Finite Witness Theorem

**Authors:** Tal Amir, Steven Gortler, Ilai Avni, Ravina Ravina, Nadav Dym

**<details><summary>Abstract**</summary>

Injective multiset functions have a key role in the theoretical study of machine learning on multisets and graphs. Yet, there remains a gap between the provably injective multiset functions considered in theory, which typically rely on polynomial moments, and the multiset functions used in practice, which rely on $\textit{neural moments}$ — whose injectivity on multisets has not been studied to date.In this paper, we bridge this gap by showing that moments of neural networks do define injective multiset functions,  provided that an analytic non-polynomial activation is used. The number of moments required by our theory is optimal essentially up to a multiplicative factor of two. To prove this result, we state and prove a $\textit{finite witness theorem}$, which is of independent interest. As a corollary to our main theorem, we derive new approximation results for functions on multisets and measures, and new separation results for graph neural networks. We also provide two negative results: (1) moments of piecewise-linear neural networks cannot be injective multiset functions; and (2) even when moment-based multiset functions are injective, they can never be bi-Lipschitz.</details>

### Neural Lad: A Neural Latent Dynamics Framework for Times Series Modeling

**Authors:** ting li, Jianguo Li, Zhanxing Zhu

**<details><summary>Abstract**</summary>

Neural ordinary differential equation (Neural ODE) is an elegant yet powerful framework to learn the temporal dynamics for time series modeling.However, we observe that existing Neural ODE forecasting models suffer from two disadvantages:i) controlling the latent states only through the linear transformation over the local change of the observed signals may be inadequate;ii) lacking the ability to capture the inherent periodical property in time series forecasting tasks;To overcome the two issues, we introduce a new neural ODE framework called \textbf{Neural Lad}, a \textbf{Neural} \textbf{La}tent \textbf{d}ynamics model in which the latent representations evolve with an ODE enhanced by the change of observed signal and seasonality-trend characterization. We incorporate the local change of input signal into the latent dynamics in an attention-based manner and design a residual architecture over basis expansion to depict the periodicity in the underlying dynamics. To accommodate the multivariate time series forecasting, we extend the  Neural Lad through learning an adaptive relationship between multiple time series.        Experiments demonstrate that our model can achieve better or comparable performance against existing neural ODE families and transformer variants in various datasets. Remarkably, the empirical superiority of Neural Lad is consistent across  short and long-horizon forecasting for both univariate, multivariate and even irregular sampled time series.</details>

### Neural Sampling in Hierarchical Exponential-family Energy-based Models

**Authors:** Xingsi Dong, Si Wu

**<details><summary>Abstract**</summary>

Bayesian brain theory suggests that the brain employs generative models to understand the external world. The sampling-based perspective posits that the brain infers the posterior distribution through samples of stochastic neuronal responses. Additionally, the brain continually updates its generative model to approach the true distribution of the external world. In this study, we introduce the Hierarchical Exponential-family Energy-based (HEE) model, which captures the dynamics of inference and learning. In the HEE model, we decompose the partition function into individual layers and leverage a group of neurons with shorter time constants to sample the gradient of the decomposed normalization term. This allows our model to estimate the partition function and perform inference simultaneously, circumventing the negative phase encountered in conventional energy-based models (EBMs). As a result, the learning process is localized both in time and space, and the model is easy to converge. To match the brain's rapid computation, we demonstrate that neural adaptation can serve as a momentum term, significantly accelerating the inference process. On natural image datasets, our model exhibits representations akin to those observed in the biological visual system. Furthermore, for the machine learning community, our model can generate observations through joint or marginal generation. We show that marginal generation outperforms joint generation and achieves performance on par with other EBMs.</details>

### NeuroGF: A Neural Representation for Fast Geodesic Distance and Path Queries

**Authors:** Qijian Zhang, Junhui Hou, Yohanes Adikusuma, Wenping Wang, Ying He

**<details><summary>Abstract**</summary>

Geodesics play a critical role in many geometry processing applications. Traditional algorithms for computing geodesics on 3D mesh models are often inefficient and slow, which make them impractical for scenarios requiring extensive querying of arbitrary point-to-point geodesics. Recently, deep implicit functions have gained popularity for 3D geometry representation, yet there is still no research on neural implicit representation of geodesics. To bridge this gap, we make the first attempt to represent geodesics using implicit learning frameworks. Specifically, we propose neural geodesic field (NeuroGF), which can be learned to encode all-pairs geodesics of a given 3D mesh model, enabling to efficiently and accurately answer queries of arbitrary point-to-point geodesic distances and paths. Evaluations on common 3D object models and real-captured scene-level meshes demonstrate our exceptional performances in terms of representation accuracy and querying efficiency. Besides, NeuroGF also provides a convenient way of jointly encoding both 3D geometry and geodesics in a unified representation. Moreover, the working mode of per-model overfitting is further extended to generalizable learning frameworks that can work on various input formats such as unstructured point clouds, which also show satisfactory performances for unseen shapes and categories. Our code and data are available at https://github.com/keeganhk/NeuroGF.</details>

### No-regret Algorithms for Fair Resource Allocation

**Authors:** Abhishek Sinha, Ativ Joshi, Rajarshi Bhattacharjee, Cameron Musco, Mohammad Hajiesmaili

**<details><summary>Abstract**</summary>

We consider a fair resource allocation problem in the no-regret setting against an unrestricted adversary. The objective is to allocate resources equitably among several agents in an online fashion so that the difference of the aggregate $\alpha$-fair utilities of the agents achieved by an optimal static clairvoyant allocation and the online policy grows sublinearly with time. The problem inherits its difficulty from the non-separable nature of the global $\alpha$-fairness function. Previously, it was shown that no online policy could achieve a sublinear standard regret in this problem. In this paper, we propose an efficient online resource allocation policy, called Online Fair Allocation ($\texttt{OFA}$), that achieves sublinear $c_\alpha$-approximate regret with approximation factor $c_\alpha=(1-\alpha)^{-(1-\alpha)}\leq 1.445,$ for $0\leq \alpha < 1$. Our upper bound on the $c_\alpha$-regret for this problem exhibits a surprising \emph{phase transition} phenomenon -- transitioning from a power-law to a constant at the critical exponent $\alpha=\frac{1}{2}.$  Our result also resolves an open problem in designing an efficient no-regret policy for the online job scheduling problem in certain parameter regimes. Along the way, we introduce new algorithmic and analytical techniques, including greedy estimation of the future gradients for non-additive global reward functions and bootstrapping second-order regret bounds, which may be of independent interest.</details>

### Non-Rigid Shape Registration via Deep Functional Maps Prior

**Authors:** puhua jiang, Mingze Sun, Ruqi Huang

**<details><summary>Abstract**</summary>

In this paper, we propose a learning-based framework for non-rigid shape registra- tion without correspondence supervision. Traditional shape registration techniques typically rely on correspondences induced by extrinsic proximity, therefore can fail in the presence of large intrinsic deformations. Spectral mapping methods overcome this challenge by embedding shapes into, geometric or learned, high- dimensional spaces, where shapes are easier to align. However, due to the dependency on abstract, non-linear embedding schemes, the latter can be vulnerable with respect to perturbed or alien input. In light of this, our framework takes the best of both worlds. Namely, we deform source mesh towards the target point cloud, guided by correspondences induced by high-dimensional embeddings learned from deep functional maps (DFM). In particular, the correspondences are dynamically updated according to the intermediate registrations and filtered by consistency prior, which prominently robustify the overall pipeline. Moreover, in order to alleviate the requirement of extrinsically aligned input, we train an orientation regressor on a set of aligned synthetic shapes independent of the training shapes for DFM. Empirical results show that, with as few as dozens of training shapes of limited variability, our pipeline achieves state-of-the-art results on several benchmarks of non-rigid point cloud matching, but also delivers high-quality correspondences between unseen challenging shape pairs that undergo both significant extrinsic and intrinsic defor- mations, in which case neither traditional registration methods nor intrinsic methods work. The code is available at https://github.com/rqhuang88/DFR.</details>

### Nonparametric Boundary Geometry in Physics Informed Deep Learning

**Authors:** Scott Cameron, Arnu Pretorius, S Roberts

**<details><summary>Abstract**</summary>

Engineering design problems frequently require solving systems ofpartial differential equations with boundary conditions specified onobject geometries in the form of a triangular mesh. These boundarygeometries are provided by a designer and are problem dependent.The efficiency of the design process greatly benefits from fast turnaroundtimes when repeatedly solving PDEs on various geometries. However,most current work that uses machine learning to speed up the solutionprocess relies heavily on a fixed parameterization of the geometry, whichcannot be changed after training. This severely limits the possibility ofreusing a trained model across a variety of design problems.In this work, we propose a novel neural operator architecture which acceptsboundary geometry, in the form of triangular meshes, as input and produces anapproximate solution to a given PDE as output. Once trained, the model can beused to rapidly estimate the PDE solution over a new geometry, without the need forretraining or representation of the geometry to a pre-specified parameterization.</details>

### Nonparametric Identifiability of Causal Representations from Unknown Interventions

**Authors:** Julius von Kügelgen, Michel Besserve, Liang Wendong, Luigi Gresele, Armin Kekić, Elias Bareinboim, David Blei, Bernhard Schölkopf

**<details><summary>Abstract**</summary>

We study causal representation learning, the task of inferring latent causal variables and their causal relations from high-dimensional functions (“mixtures”) of the variables. Prior work relies on weak supervision, in the form of counterfactual pre- and post-intervention views or temporal structure; places restrictive assumptions, such as linearity, on the mixing function or latent causal model; or requires partial knowledge of the generative process, such as the causal graph or intervention targets. We instead consider the general setting in which both the causal model and the mixing function are nonparametric. The learning signal takes the form of multiple datasets, or environments, arising from unknown interventions in the underlying causal model. Our goal is to identify both the ground truth latents and their causal graph up to a set of ambiguities which we show to be irresolvable from interventional data. We study the fundamental setting of two causal variables and prove that the observational distribution and one perfect intervention per node suffice for identifiability, subject to a genericity condition. This condition rules out spurious solutions that involve fine-tuning of the intervened and observational distributions, mirroring similar conditions for nonlinear cause-effect inference. For an arbitrary number of variables, we show that at least one pair of distinct perfect interventional domains per node guarantees identifiability. Further, we demonstrate that the strengths of causal influences among the latent variables are preserved by all equivalent solutions, rendering the inferred representation appropriate for drawing causal conclusions from new data. Our study provides the first identifiability results for the general nonparametric setting with unknown interventions, and elucidates what is possible and impossible for causal representation learning without more direct supervision.</details>

### Nonparametric Teaching for Multiple Learners

**Authors:** Chen Zhang, Xiaofeng Cao, Weiyang Liu, Ivor Tsang, James Kwok

**<details><summary>Abstract**</summary>

We study the problem of teaching multiple learners simultaneously in the nonparametric iterative teaching setting, where the teacher iteratively provides examples to the learner for accelerating the acquisition of a target concept. This problem is motivated by the gap between current single-learner teaching setting and the real-world scenario of human instruction where a teacher typically imparts knowledge to multiple students. Under the new problem formulation, we introduce a novel framework -- Multi-learner Nonparametric Teaching (MINT). In MINT, the teacher aims to instruct multiple learners, with each learner focusing on learning a scalar-valued target model. To achieve this, we frame the problem as teaching a vector-valued target model and extend the target model space from a scalar-valued reproducing kernel Hilbert space used in single-learner scenarios to a vector-valued space. Furthermore, we demonstrate that MINT offers significant teaching speed-up over repeated single-learner teaching, particularly when the multiple learners can communicate with each other. Lastly, we conduct extensive experiments to validate the practicality and efficiency of MINT.</details>

### [Spotlight] Normalizing flow neural networks by JKO scheme

**Authors:** Chen Xu, Xiuyuan Cheng, Yao Xie

**<details><summary>Abstract**</summary>

Normalizing flow is a class of deep generative models for efficient sampling and likelihood estimation, which achieves attractive performance, particularly in high dimensions. The flow is often implemented using a sequence of invertible residual blocks. Existing works adopt special network architectures and regularization of flow trajectories. In this paper, we develop a neural ODE flow network called JKO-iFlow, inspired by the Jordan-Kinderleherer-Otto (JKO) scheme, which unfolds the discrete-time dynamic of the Wasserstein gradient flow. The proposed method stacks residual blocks one after another, allowing efficient block-wise training of the residual blocks, avoiding sampling SDE trajectories and score matching or variational learning, thus reducing the memory load and difficulty in end-to-end training. We also develop adaptive time reparameterization of the flow network with a progressive refinement of the induced trajectory in probability space to improve the model accuracy further. Experiments with synthetic and real data show that the proposed JKO-iFlow network achieves competitive performance compared with existing flow and diffusion models at a significantly reduced computational and memory cost.</details>

### Offline Reinforcement Learning for Mixture-of-Expert Dialogue Management

**Authors:** Dhawal Gupta, Yinlam Chow, Azamat Tulepbergenov, Mohammad Ghavamzadeh, Craig Boutilier

**<details><summary>Abstract**</summary>

Reinforcement learning (RL) has shown great promise for developing agents for dialogue management (DM) that are non-myopic, conduct rich conversations, and maximize overall user satisfaction. Despite the advancements in RL and language models (LMs), employing RL to drive conversational chatbots still poses significant challenges. A primary issue stems from RL’s dependency on online exploration for effective learning, a process that can be costly. Moreover, engaging in online interactions with humans during the training phase can raise safety concerns, as the LM can potentially generate unwanted outputs. This issue is exacerbated by the combinatorial action spaces facing these algorithms, as most LM agents generate responses at the word level. We develop various RL algorithms, specialized in dialogue planning, that leverage recent Mixture-of-Expert Language Models (MoE-LMs)---models that capture diverse semantics, generate utterances reflecting different intents, and are amenable for multi-turn DM. By exploiting the MoE-LM structure, our methods significantly reduce the size of the action space and improve the efficacy of RL-based DM. We evaluate our methods in open-domain dialogue to demonstrate their effectiveness with respect to the diversity of intent in generated utterances and overall DM performance.</details>

### On Calibrating Diffusion Probabilistic Models

**Authors:** Tianyu Pang, Cheng Lu, Chao Du, Min Lin, Shuicheng Yan, Zhijie Deng

**<details><summary>Abstract**</summary>

Recently, diffusion probabilistic models (DPMs) have achieved promising results in diverse generative tasks. A typical DPM framework includes a forward process that gradually diffuses the data distribution and a reverse process that recovers the data distribution from time-dependent data scores. In this work, we observe that the stochastic reverse process of data scores is a martingale, from which concentration bounds and the optional stopping theorem for data scores can be derived. Then, we discover a simple way for calibrating an arbitrary pretrained DPM, with which the score matching loss can be reduced and the lower bounds of model likelihood can consequently be increased. We provide general calibration guidelines under various model parametrizations. Our calibration method is performed only once and the resulting models can be used repeatedly for sampling. We conduct experiments on multiple datasets to empirically validate our proposal. Our code is available at https://github.com/thudzj/Calibrated-DPMs.</details>

### On Generalization Bounds for Projective Clustering

**Authors:** Maria Sofia Bucarelli, Matilde Larsen, Chris Schwiegelshohn, Mads Toftrup

**<details><summary>Abstract**</summary>

Given a set of points, clustering consists of finding a partition of a point set into $k$ clusters such that the center to which a point is assigned is as close as possible. Most commonly, centers are points themselves, which leads to the famous $k$-median and $k$-means objectives. One may also choose centers to be $j$ dimensional subspaces, which gives rise to subspace clustering. In this paper, we consider learning bounds for these problems. That is, given a set of $n$ samples $P$ drawn independently from some unknown, but fixed distribution $\mathcal{D}$, how quickly does a solution computed on $P$ converge to the optimal clustering of $\mathcal{D}$?We give several near optimal results. In particular, 1. For center-based objectives, we show a convergence rate of $\tilde{O}\left(\sqrt{{k}/{n}}\right)$. This matches the known optimal bounds of [Fefferman, Mitter, and Narayanan, Journal of the Mathematical Society 2016] and [Bartlett, Linder, and Lugosi, IEEE Trans. Inf. Theory 1998] for $k$-means and extends it to other important objectives such as $k$-median. 2. For subspace clustering with $j$-dimensional subspaces, we show a convergence rate of $\tilde{O}\left(\sqrt{{(kj^2)}/{n}}\right)$. These are the first provable bounds for most of these problems. For the specific case of projective clustering, which generalizes $k$-means, we show a converge rate of $\Omega\left(\sqrt{{(kj)}/{n}}\right)$ is necessary, thereby proving that the bounds from [Fefferman, Mitter, and Narayanan, Journal of the Mathematical Society 2016] are essentially optimal.</details>

### On Imitation in Mean-field Games

**Authors:** Giorgia Ramponi, Pavel Kolev, Olivier Pietquin, Niao He, Mathieu Lauriere, Matthieu Geist

**<details><summary>Abstract**</summary>

We explore the problem of imitation learning (IL) in the context of mean-field games (MFGs), where the goal is to imitate the behavior of a population of agents following a Nash equilibrium policy according to some unknown payoff function. IL in MFGs presents new challenges compared to single-agent IL, particularly when both the reward function and the transition kernel depend on the population distribution. In this paper, departing from the existing literature on IL for MFGs, we introduce a new solution concept called the Nash imitation gap. Then we show that when only the reward depends on the population distribution, IL in MFGs can be reduced to single-agent IL with similar guarantees. However, when the dynamics is population-dependent, we provide a novel upper-bound that suggests IL is harder in this setting. To address this issue, we propose a new adversarial formulation where the reinforcement learning problem is replaced by a mean-field control (MFC) problem, suggesting progress in IL within MFGs may have to build upon MFC.</details>

### On Learning Latent Models with Multi-Instance Weak Supervision

**Authors:** Kaifu Wang, Efthymia Tsamoura, Dan Roth

**<details><summary>Abstract**</summary>

We consider a weakly supervised learning scenario where the supervision signal is generated by a transition function $\sigma$ of labels associated with multiple input instances. We formulate this problem as *multi-instance Partial Label Learning (multi-instance PLL)*, which is an extension to the standard PLL problem. Our problem is met in different fields, including latent structural learning and neuro-symbolic integration. Despite the existence of many learning techniques, limited theoretical analysis has been dedicated to this problem. In this paper, we provide the first theoretical study of multi-instance PLL with possibly an unknown transition $\sigma$. Our main contributions are as follows: First, we proposed a necessary and sufficient condition for the learnability of the problem. This condition nontrivially generalizes and relaxes the existing *small ambiguity degree* in PLL literature since we allow the transition to be deterministic. Second, we derived Rademacher-style error bounds based on the top-$k$ surrogate loss that is widely used in the neuro-symbolic literature. Furthermore, we conclude with empirical experiments for learning with an unknown transition. The empirical results align with our theoretical findings; however, they also expose the issue of scalability in the weak supervision literature.</details>

### On Measuring Fairness in Generative Models

**Authors:** Christopher Teo, Milad Abdollahzadeh, Ngai-Man (Man) Cheung

**<details><summary>Abstract**</summary>

Recently, there has been increased interest in fair generative models. In this work,we conduct, for the first time, an in-depth study on fairness measurement, acritical component in gauging progress on fair generative models. We make threecontributions. First, we conduct a study that reveals that the existing fairnessmeasurement framework has considerable measurement errors, even when highlyaccurate sensitive attribute (SA) classifiers are used. These findings cast doubtson previously reported fairness improvements. Second, to address this issue,we propose CLassifier Error-Aware Measurement (CLEAM), a new frameworkwhich uses a statistical model to account for inaccuracies in SA classifiers. Ourproposed CLEAM reduces measurement errors significantly, e.g., 4.98%→0.62%for StyleGAN2 w.r.t. Gender. Additionally, CLEAM achieves this with minimaladditional overhead. Third, we utilize CLEAM to measure fairness in importanttext-to-image generator and GANs, revealing considerable biases in these modelsthat raise concerns about their applications. Code and more resources: https://sutd-visual-computing-group.github.io/CLEAM/.</details>

### On Occlusions in Video Action Detection: Benchmark Datasets And Training Recipes

**Authors:** Rajat Modi, Vibhav Vineet, Yogesh Rawat

**<details><summary>Abstract**</summary>

This paper explores the impact of occlusions in video action detection. We facilitatethis study by introducing five new benchmark datasets namely O-UCF and O-JHMDB consisting of synthetically controlled static/dynamic occlusions, OVIS-UCF and OVIS-JHMDB consisting of occlusions with realistic motions and Real-OUCF for occlusions in realistic-world scenarios. We formally confirm an intuitiveexpectation: existing models suffer a lot as occlusion severity is increased andexhibit different behaviours when occluders are static vs when they are moving.We discover several intriguing phenomenon emerging in neural nets: 1) transformerscan naturally outperform CNN models which might have even used occlusion as aform of data augmentation during training 2) incorporating symbolic-componentslike capsules to such backbones allows them to bind to occluders never even seenduring training and 3) Islands of agreement (similar to the ones hypothesized inHinton et Al’s GLOM) can emerge in realistic images/videos without instance-levelsupervision, distillation or contrastive-based objectives(eg. video-textual training).Such emergent properties allow us to derive simple yet effective training recipeswhich lead to robust occlusion models inductively satisfying the first two stages ofthe binding mechanism (grouping/segregation). Models leveraging these recipesoutperform existing video action-detectors under occlusion by 32.3% on O-UCF,32.7% on O-JHMDB & 2.6% on Real-OUCF in terms of the vMAP metric. The code for this work has been released at https: //github.com/rajatmodi62/OccludedActionBenchmark.</details>

### On Separate Normalization in Self-supervised Transformers

**Authors:** Xiaohui Chen, Yinkai Wang, Yuanqi Du, Soha Hassoun, Liping Liu

**<details><summary>Abstract**</summary>

Self-supervised training methods for transformers have demonstrated remarkable performance across various domains. Previous transformer-based models, such as masked autoencoders (MAE), typically utilize a single normalization layer for both the [CLS] symbol and the tokens. We propose in this paper a simple modification that employs separate normalization layers for the tokens and the [CLS] symbol to better capture their distinct characteristics and enhance downstream task performance. Our method aims to alleviate the potential negative effects of using the same normalization statistics for both token types, which may not be optimally aligned with their individual roles. We empirically show that by utilizing a separate normalization layer, the [CLS] embeddings can better encode the global contextual information and are distributed more uniformly in its anisotropic space. When replacing the conventional normalization layer with the two separate layers, we observe an average  2.7% performance improvement over the image, natural language, and graph domains.</details>

### On kernel-based statistical learning theory in the mean field limit

**Authors:** Christian Fiedler, Michael Herty, Sebastian Trimpe

**<details><summary>Abstract**</summary>

In many applications of machine learning, a large number of variables are considered. Motivated by machine learning of interacting particle systems, we consider the situation when the number of input variables goes to infinity. First, we continue the recent investigation of the mean field limit of kernels and their reproducing kernel Hilbert spaces, completing the existing theory. Next, we provide results relevant for approximation with such kernels in the mean field limit, including a representer theorem. Finally, we use these kernels in the context of statistical learning in the mean field limit, focusing on Support Vector Machines. In particular, we show mean field convergence of empirical and infinite-sample solutions as well as the convergence of the corresponding risks. On the one hand, our results establish rigorous mean field limits in the context of kernel methods, providing new theoretical tools and insights for large-scale problems. On the other hand, our setting corresponds to a new form of limit of learning problems, which seems to have not been investigated yet in the statistical learning theory literature.</details>

### On the Ability of Graph Neural Networks to Model Interactions Between Vertices

**Authors:** Noam Razin, Tom Verbin, Nadav Cohen

**<details><summary>Abstract**</summary>

Graph neural networks (GNNs) are widely used for modeling complex interactions between entities represented as vertices of a graph. Despite recent efforts to theoretically analyze the expressive power of GNNs, a formal characterization of their ability to model interactions is lacking. The current paper aims to address this gap. Formalizing strength of interactions through an established measure known as separation rank, we quantify the ability of certain GNNs to model interaction between a given subset of vertices and its complement, i.e. between the sides of a given partition of input vertices. Our results reveal that the ability to model interaction is primarily determined by the partition's walk index --- a graph-theoretical characteristic defined by the number of walks originating from the boundary of the partition. Experiments with common GNN architectures corroborate this finding. As a practical application of our theory, we design an edge sparsification algorithm named Walk Index Sparsification (WIS), which preserves the ability of a GNN to model interactions when input edges are removed. WIS is simple, computationally efficient, and in our experiments has markedly outperformed alternative methods in terms of induced prediction accuracy. More broadly, it showcases the potential of improving GNNs by theoretically analyzing the interactions they can model.</details>

### On the Convergence of Black-Box Variational Inference

**Authors:** Kyurae Kim, Jisu Oh, Kaiwen Wu, Yian Ma, Jacob Gardner

**<details><summary>Abstract**</summary>

We provide the first convergence guarantee for black-box variational inference (BBVI) with the reparameterization gradient.  While preliminary investigations worked on simplified versions of BBVI (e.g., bounded domain, bounded support, only optimizing for the scale, and such), our setup does not need any such algorithmic modifications.  Our results hold for log-smooth posterior densities with and without strong log-concavity and the location-scale variational family.  Notably, our analysis reveals that certain algorithm design choices commonly employed in practice, such as nonlinear parameterizations of the scale matrix, can result in suboptimal convergence rates.  Fortunately, running BBVI with proximal stochastic gradient descent fixes these limitations and thus achieves the strongest known convergence guarantees.  We evaluate this theoretical insight by comparing proximal SGD against other standard implementations of BBVI on large-scale Bayesian inference problems.</details>

### On the Convergence of CART under Sufficient Impurity Decrease Condition

**Authors:** Rahul Mazumder, Haoyue Wang

**<details><summary>Abstract**</summary>

The decision tree is a flexible machine-learning model that finds its success in numerous applications. It is usually fitted in a recursively greedy manner using CART. In this paper, we study the convergence rate of CART under a regression setting. First, we prove an upper bound on the prediction error of CART under a sufficient impurity decrease (SID) condition \cite{chi2020asymptotic} -- our result is an improvement over the known result by \cite{chi2020asymptotic} under a similar assumption. We show via examples that this error bound cannot be further improved by more than a constant or a log factor. Second, we introduce a few easy-to-check sufficient conditions of the SID condition. In particular, we show that the SID condition can be satisfied by an additive model when the component functions satisfy a ``locally reverse Poincare inequality". We discuss a few familiar function classes in non-parametric estimation to demonstrate the usefulness of this conception.</details>

### [Spotlight] On the Gini-impurity Preservation For Privacy Random Forests

**Authors:** XinRan Xie, Man-Jie Yuan, Xuetong Bai, Wei Gao, Zhi-Hua Zhou

**<details><summary>Abstract**</summary>

Random forests have been one successful ensemble algorithms in machine learning. Various techniques have been utilized to preserve the privacy of random forests from anonymization, differential privacy, homomorphic encryption, etc., whereas it rarely takes into account some crucial ingredients of learning algorithm. This work presents a new encryption to preserve data's Gini impurity, which plays a crucial role during the construction of random forests. Our basic idea is to modify the structure of binary search tree to store several examples in each node,  and encrypt data features by incorporating label and order information. Theoretically, we prove that our scheme preserves the minimum Gini impurity in ciphertexts without decrypting, and present the security guarantee for encryption. For random forests, we encrypt data features based on our Gini-impurity-preserving scheme, and take the homomorphic encryption scheme CKKS to encrypt data labels  due to their importance and privacy. We  conduct extensive experiments to show the effectiveness, efficiency and security of our proposed method.</details>

### [Spotlight] On the Learnability of Multilabel Ranking

**Authors:** Vinod Raman, UNIQUE SUBEDI, Ambuj Tewari

**<details><summary>Abstract**</summary>

Multilabel ranking is a central task in machine learning. However, the most fundamental question of learnability in a multilabel ranking setting with relevance-score feedback remains unanswered. In this work, we characterize the learnability of multilabel ranking problems in both batch and online settings for a large family of ranking losses. Along the way, we give two equivalence classes of ranking losses based on learnability that capture most losses used in practice.</details>

### On the Robustness of Removal-Based Feature Attributions

**Authors:** Chris Lin, Ian Covert, Su-In Lee

**<details><summary>Abstract**</summary>

To explain predictions made by complex machine learning models, many feature attribution methods have been developed that assign importance scores to input features. Some recent work challenges the robustness of these methods by showing that they are sensitive to input and model perturbations, while other work addresses this issue by proposing robust attribution methods. However, previous work on attribution robustness has focused primarily on gradient-based feature attributions, whereas the robustness of removal-based attribution methods is not currently well understood. To bridge this gap, we theoretically characterize the robustness properties of removal-based feature attributions. Specifically, we provide a unified analysis of such methods and derive upper bounds for the difference between intact and perturbed attributions, under settings of both input and model perturbations. Our empirical results on synthetic and real-world data validate our theoretical results and demonstrate their practical implications, including the ability to increase attribution robustness by improving the model’s Lipschitz regularity.</details>

### On the Role of Entanglement and Statistics in Learning

**Authors:** Srinivasan Arunachalam, Vojtech Havlicek, Louis Schatzki

**<details><summary>Abstract**</summary>

In this work we make progress in understanding the relationship between learning models when given access to entangled measurements, separable measurements and statistical measurements in the quantum statistical query ($\mathsf{QSQ}$) model. To this end, we show the following results.$\textbf{Entanglement versus separable measurements.}$ The goal here is to learn an unknown $f$ from the concept class $\mathcal{C} \subseteq \{f:\{0,1\}^n\rightarrow [k]\}$ given copies of  $\frac{1}{\sqrt{2^n}}\sum_x \ket{x,f(x)}$.  We show that, if $T$ copies suffice to learn $f$ using entangled measurements, then $O(nT^2)$ copies suffice to learn $f$ using just separable measurements. Additionally, we exhibit a concept class $\mathcal{C}$ for which, in order to learn some \emph{property} of $f$, the sample complexity of learning using entangled measurements is exponentially smaller than separable measurements.$\textbf{Entangled versus statistical measurements}$ The goal here is to learn a function $f \in \mathcal{C}$ given access to separable measurements and statistical measurements.   We exhibit a concept class $\mathcal{C}$ based on degree-$2$ functions that gives an exponential separation between $\mathsf{QSQ}$ learning and quantum learning with entangled measurements (even in the presence of noise). This proves the "quantum analogue" of the seminal result of (Blum, 2003) that separates classical $\mathsf{SQ}$ learning from classical $\mathsf{PAC}$ learning with classification~noise.$\textbf{$\mathsf{QSQ}$ lower bounds for learning states.}$ The main technical contribution is to introduce a quantum statistical query dimension ($\mathsf{QSDA}$), which we use to give lower bounds on the $\mathsf{QSQ}$ complexity of learning. Using this, we prove exponential $\mathsf{QSQ}$ lower bounds for testing purity of quantum states, learning CCHL states, coset states of Abelian groups, degree-$2$ functions, planted bi-clique states and learning output states of Clifford circuits of depth polylog($n$).$\textbf{Further applications.}$ Using our $\mathsf{QSQ}$ lower bounds give an $\textit{unconditional}$ separation between weak and strong error mitigation and prove lower bounds for learning distributions in the $\mathsf{QSQ}$ model. Prior works by (Quek et al., 2022), (Hinsche et al., 2022), and (Neitner et al., 23) proved the analogous results $\textit{assuming}$ diagonal measurements and our work removes this assumption.</details>

### [Spotlight] On the Role of Randomization in Adversarially Robust Classification

**Authors:** Lucas Gnecco Heredia, Muni Sreenivas Pydi, Laurent Meunier, Benjamin Negrevergne, Yann Chevaleyre

**<details><summary>Abstract**</summary>

Deep neural networks are known to be vulnerable to small adversarial perturbations in test data. To defend against adversarial attacks, probabilistic classifiers have been proposed as an alternative to deterministic ones. However, literature has conflicting findings on the effectiveness of probabilistic classifiers in comparison to deterministic ones. In this paper, we clarify the role of randomization in building adversarially robust classifiers.Given a base hypothesis set of deterministic classifiers, we show the conditions under which a randomized ensemble outperforms the hypothesis set in adversarial risk, extending previous results.Additionally, we show that for any probabilistic binary classifier (including randomized ensembles), there exists a deterministic classifier that outperforms it. Finally, we give an explicit description of the deterministic hypothesis set that contains such a deterministic classifier for many types of commonly used probabilistic classifiers, *i.e.* randomized ensembles and parametric/input noise injection.</details>

### On the Size and Approximation Error of Distilled Datasets

**Authors:** Alaa Maalouf, Murad Tukan, Noel Loo, Ramin Hasani, Mathias Lechner, Daniela Rus

**<details><summary>Abstract**</summary>

Dataset Distillation is the task of synthesizing small datasets from large ones while still retaining comparable predictive accuracy to the original uncompressed dataset. Despite significant empirical progress in recent years, there is little understanding of the theoretical limitations/guarantees of dataset distillation, specifically, what excess risk is achieved by distillation compared to the original dataset, and how large are distilled datasets? In this work, we take a theoretical view on kernel ridge regression (KRR) based methods of dataset distillation such as Kernel Inducing Points. By transforming ridge regression in random Fourier features (RFF) space, we provide the first proof of the existence of small (size) distilled datasets and their corresponding excess risk for shift-invariant kernels. We prove that a small set of instances exists in the original input space such that its solution in the RFF space coincides with the solution of the original data. We further show that a KRR solution can be generated using this distilled set of instances which gives an approximation towards the KRR solution optimized on the full input data. The size of this set is linear in the dimension of the RFF space of the input set or alternatively near linear in the number of effective degrees of freedom, which is a function of the kernel, number of data points, and the regularization parameter $\lambda$. The error bound of this distilled set is also a function of $\lambda$.  We verify our bounds analytically and empirically.</details>

### On-the-Fly Adapting Code Summarization on Trainable Cost-Effective Language Models

**Authors:** Yufan Cai, Yun Lin, Chenyan Liu, Jinglian Wu, Yifan Zhang, Yiming Liu, Yeyun Gong, Jin Song Dong

**<details><summary>Abstract**</summary>

Deep learning models are emerging to summarize source code to comment,  facilitating tasks of code documentation and program comprehension.  Scaled-up large language models trained on large open corpus have achieved good performance in such tasks.  However, in practice, the subject code in one certain project can be specific,  which may not align with the overall training corpus.  Some code samples from other projects may be contradictory and introduce inconsistencies when the models try to fit all the samples.  In this work, we introduce a novel approach, Adacom, to improve the performance of comment generators by on-the-fly model adaptation.  This research is motivated by the observation that deep comment generators  often need to strike a balance as they need to fit all the training samples.  Specifically, for one certain target code $c$,  some training samples $S_p$ could have made more contributions while other samples $S_o$ could have counter effects.  However, the traditional fine-tuned models need to fit both $S_p$ and $S_o$ from a global perspective,   leading to compromised performance for one certain target code $c$.  In this context, we design Adacom to  (1) detect whether the model might have a compromised performance on a target code $c$ and  (2) retrieve a few helpful training samples $S_p$ that have contradictory samples in the training dataset and,  (3) adapt the model on the fly by re-training the $S_p$ to strengthen the helpful samples and unlearn the harmful samples.  Our extensive experiments on 7 comment generators and 4 public datasets show that  (1) can significantly boost the performance of comment generation (BLEU4 score by on average 14.9\%, METEOR by 12.2\%, and ROUGE-L by 7.4\%),  (2) the adaptation on one code sample is cost-effective and acceptable as an on-the-fly solution, and  (3) can adapt well on out-of-distribution code samples.</details>

### One-Step Diffusion Distillation via Deep Equilibrium Models

**Authors:** Zhengyang Geng, Ashwini Pokle, J. Zico Kolter

**<details><summary>Abstract**</summary>

Diffusion models excel at producing high-quality samples but naively require hundreds of iterations, prompting multiple attempts to distill the generation process into a faster network. However, many existing approaches suffer from a variety of challenges: the process for distillation training can be complex, often requiring multiple training stages, and the resulting models perform poorly when utilized in single-step generative applications. In this paper, we introduce a simple yet effective means of distilling diffusion models *directly* from the initial noise to the resulting image. Of particular importance to our approach is to leverage a new Deep Equilibrium (DEQ) model as the distilled architecture: the Generative Equilibrium Transformer (GET). Our method enables fully offline training with just noise/image pairs from the diffusion model while achieving superior performance compared to existing one-step methods on comparable training budgets. We demonstrate that the DEQ architecture is crucial to this capability, as GET matches a $5	imes$ larger ViT in terms of FID scores while striking a critical balance of computational cost and image quality. Code, checkpoints, and datasets are available [here](https://github.com/locuslab/get).</details>

### [Spotlight] Online Control for Meta-optimization

**Authors:** Xinyi Chen, Elad Hazan

**<details><summary>Abstract**</summary>

Choosing the optimal hyperparameters, including learning rate and momentum, for specific optimization instances is a significant yet non-convex challenge. This makes conventional iterative techniques such as hypergradient descent \cite{baydin2017online} insufficient in obtaining global optimality guarantees.We consider the more general task of meta-optimization -- online learning of the best optimization algorithm given problem instances, and introduce a novel approach based on control theory. We show how meta-optimization can be formulated as an optimal control problem, departing from existing literature that use stability-based methods to study optimization. Our approach leverages convex relaxation techniques in the recently-proposed nonstochastic control framework to overcome the challenge of nonconvexity, and obtains regret guarantees vs. the best offline solution. This guarantees that in meta-optimization, we can learn a method that attains convergence comparable to that of the best optimization method in hindsight from a class of methods.</details>

### Online Inventory Problems: Beyond the i.i.d. Setting with Online Convex Optimization

**Authors:** Massil HIHAT, Stéphane Gaïffas, Guillaume Garrigos, Simon Bussy

**<details><summary>Abstract**</summary>

We study multi-product inventory control problems where a manager makes sequential replenishment decisions based on partial historical information in order to minimize its cumulative losses. Our motivation is to consider general demands, losses and dynamics to go beyond standard models which usually rely on newsvendor-type losses, fixed dynamics, and unrealistic i.i.d. demand assumptions. We propose MaxCOSD, an online algorithm that has provable guarantees even for problems with non-i.i.d. demands and stateful dynamics, including for instance perishability. We consider what we call non-degeneracy assumptions on the demand process, and argue that they are necessary to allow learning.</details>

### Optimal Parameter and Neuron Pruning for Out-of-Distribution Detection

**Authors:** Chao Chen, Zhihang Fu, Kai Liu, Ze Chen, Mingyuan Tao, Jieping Ye

**<details><summary>Abstract**</summary>

For a machine learning model deployed in real world scenarios, the ability of detecting out-of-distribution (OOD) samples is indispensable and challenging. Most existing OOD detection methods focused on exploring advanced training skills or training-free tricks to prevent the model from yielding overconfident confidence score for unknown samples. The training-based methods require expensive training cost and rely on OOD samples which are not always available, while most training-free methods can not efficiently utilize the prior information from the training data. In this work, we propose an \textbf{O}ptimal \textbf{P}arameter and \textbf{N}euron \textbf{P}runing (\textbf{OPNP}) approach, which aims to identify and remove those parameters and neurons that lead to over-fitting. The main method is divided into two steps. In the first step, we evaluate the sensitivity of the model parameters and neurons by averaging gradients over all training samples. In the second step, the parameters and neurons with exceptionally large or close to zero sensitivities are removed for prediction. Our proposal is training-free, compatible with other post-hoc methods, and exploring the information from all training data. Extensive experiments are performed on multiple OOD detection tasks and model architectures, showing that our proposed OPNP consistently outperforms the existing methods by a large margin.</details>

### Optimal Unbiased Randomizers for Regression with Label Differential Privacy

**Authors:** Badih Ghazi, Pritish Kamath, Ravi Kumar, Ethan Leeman, Pasin Manurangsi, Avinash V Varadarajan, Chiyuan Zhang

**<details><summary>Abstract**</summary>

We propose a new family of label randomizers for  training _regression_ models under the constraint of label differential privacy (DP). In particular, we leverage the trade-offs between bias and variance to construct better label randomizers depending on a privately estimated prior distribution over the labels. We demonstrate that these randomizers achieve state-of-the-art privacy-utility trade-offs on several datasets, highlighting the importance of reducing bias when training neural networks with label DP. We also provide theoretical results shedding light on the structural properties of the optimal unbiased randomizers.</details>

### Optimal and Fair Encouragement Policy Evaluation and Learning

**Authors:** Angela Zhou

**<details><summary>Abstract**</summary>

In consequential domains, it is often impossible to compel individuals to take treatment, so that optimal policy rules are merely suggestions in the presence of human non-adherence to treatment recommendations. In these same domains, there may be heterogeneity both in who responds in taking-up treatment, and heterogeneity in treatment efficacy. For example, in social services, a persistent puzzle is the gap in take-up of beneficial services among those who may benefit from them the most. When in addition the decision-maker has distributional preferences over both access and average outcomes, the optimal decision rule changes. We study identification, doubly-robust estimation, and robust estimation under potential violations of positivity. We consider fairness constraints such as demographic parity in treatment take-up, and other constraints, via constrained optimization. Our framework can be extended to handle algorithmic recommendations under an often-reasonable covariate-conditional exclusion restriction, using our robustness checks for lack of positivity in the recommendation. We develop a two-stage, online learning-based algorithm for solving over parametrized policy classes under general constraints to obtain variance-sensitive regret bounds. We assess improved recommendation rules in a stylized case study of optimizing recommendation of supervised release in the PSA-DMF pretrial risk-assessment tool while reducing surveillance disparities.</details>

### Optimistic Meta-Gradients

**Authors:** Sebastian Flennerhag, Tom Zahavy, Brendan O'Donoghue, Hado van Hasselt, András György, Satinder Singh

**<details><summary>Abstract**</summary>

We study the connection between gradient-based meta-learning and convex optimisation. We observe that gradient descent with momentum is a special case of meta-gradients, and building on recent results in optimisation, we prove convergence rates for meta learning in the single task setting. While a meta-learned update rule can yield faster convergence up to constant factor, it is not sufficient for acceleration. Instead, some form of optimism is required. We show that optimism in meta-learning can be captured through the recently proposed Bootstrapped Meta-Gradient (Flennerhag et. al., 2022) method, providing deeper insight into its underlying mechanics.</details>

### PAC-Bayes Generalization Certificates for Learned Inductive Conformal Prediction

**Authors:** Apoorva Sharma, Sushant Veer, Asher Hancock, Heng Yang, Marco Pavone, Anirudha Majumdar

**<details><summary>Abstract**</summary>

Inductive Conformal Prediction (ICP) provides a practical and effective approach for equipping deep learning models with uncertainty estimates in the form of set-valued predictions which are guaranteed to contain the ground truth with high probability.Despite the appeal of this coverage guarantee, these sets may not be efficient: the size and contents of the prediction sets are not directly controlled, and instead depend on the underlying model and choice of score function.To remedy this, recent work has proposed learning model and score function parameters using data to directly optimize the efficiency of the ICP prediction sets.While appealing, the generalization theory for such an approach is lacking: direct optimization of empirical efficiency may yield prediction sets that are either no longer efficient on test data, or no longer obtain the required coverage on test data.In this work, we use PAC-Bayes theory to obtain generalization bounds on both the coverage and the efficiency of set-valued predictors which can be directly optimized to maximize efficiency while satisfying a desired test coverage.In contrast to prior work, our framework allows us to utilize the entire calibration dataset to learn the parameters of the model and score function, instead of requiring a separate hold-out set for obtaining test-time coverage guarantees.We leverage these theoretical results to provide a practical algorithm for using calibration data to simultaneously fine-tune the parameters of a model and score function while guaranteeing test-time coverage and efficiency of the resulting prediction sets.We evaluate the approach on regression and classification tasks, and outperform baselines calibrated using a Hoeffding bound-based PAC guarantee on ICP, especially in the low-data regime.</details>

### PLASTIC: Improving Input and Label Plasticity for Sample Efficient Reinforcement Learning

**Authors:** Hojoon Lee, Hanseul Cho, HYUNSEUNG KIM, DAEHOON GWAK, Joonkee Kim, Jaegul Choo, Se-Young Yun, Chulhee Yun

**<details><summary>Abstract**</summary>

In Reinforcement Learning (RL), enhancing sample efficiency is crucial, particularly in scenarios when data acquisition is costly and risky. In principle, off-policy RL algorithms can improve sample efficiency by allowing multiple updates per environment interaction. However, these multiple updates often lead the model to overfit to earlier interactions, which is referred to as the loss of plasticity. Our study investigates the underlying causes of this phenomenon by dividing plasticity into two aspects. Input plasticity, which denotes the model's adaptability to changing input data, and label plasticity, which denotes the model's adaptability to evolving input-output relationships. Synthetic experiments on the CIFAR-10 dataset reveal that finding smoother minima of loss landscape enhances input plasticity, whereas refined gradient propagation improves label plasticity. Leveraging these findings, we introduce the **PLASTIC** algorithm, which harmoniously combines techniques to address both concerns. With minimal architectural modifications, PLASTIC achieves competitive performance on benchmarks including Atari-100k and Deepmind Control Suite. This result emphasizes the importance of preserving the model's plasticity to elevate the sample efficiency in RL. The code is available at https://github.com/dojeon-ai/plastic.</details>

### PROTES: Probabilistic Optimization with Tensor Sampling

**Authors:** Anastasiia Batsheva, Andrei Chertkov, Gleb Ryzhakov, Ivan Oseledets

**<details><summary>Abstract**</summary>

We developed a new method PROTES for black-box optimization, which is based on the probabilistic sampling from a probability density function given in the low-parametric tensor train format. We tested it on complex multidimensional arrays and discretized multivariable functions taken, among others, from real-world applications, including unconstrained binary optimization and optimal control problems, for which the possible number of elements is up to $2^{1000}$. In numerical experiments, both on analytic model functions and on complex problems, PROTES outperforms popular discrete optimization methods (Particle Swarm Optimization, Covariance Matrix Adaptation, Differential Evolution, and others).</details>

### PUCA: Patch-Unshuffle and Channel Attention for Enhanced Self-Supervised Image Denoising

**Authors:** Hyemi Jang, Junsung Park, Dahuin Jung, Jaihyun Lew, Ho Bae, Sungroh Yoon

**<details><summary>Abstract**</summary>

Although supervised image denoising networks have shown remarkable performance on synthesized noisy images, they often fail in practice due to the difference between real and synthesized noise. Since clean-noisy image pairs from the real world are extremely costly to gather, self-supervised learning, which utilizes noisy input itself as a target, has been studied. To prevent a self-supervised denoising model from learning identical mapping, each output pixel should not be influenced by its corresponding input pixel; This requirement is known as J-invariance. Blind-spot networks (BSNs) have been a prevalent choice to ensure J-invariance in self-supervised image denoising. However, constructing variations of BSNs by injecting additional operations such as downsampling can expose blinded information, thereby violating J-invariance. Consequently, convolutions designed specifically for BSNs have been allowed only, limiting architectural flexibility. To overcome this limitation, we propose PUCA, a novel J-invariant U-Net architecture, for self-supervised denoising. PUCA leverages patch-unshuffle/shuffle to dramatically expand receptive fields while maintaining J-invariance and dilated attention blocks (DABs) for global context incorporation. Experimental results demonstrate that PUCA achieves state-of-the-art performance, outperforming existing methods in self-supervised image denoising.</details>

### PanoGRF: Generalizable Spherical Radiance Fields for Wide-baseline Panoramas

**Authors:** Zheng Chen, Yan-Pei Cao, Yuan-Chen Guo, Chen Wang, Ying Shan, Song-Hai Zhang

**<details><summary>Abstract**</summary>

Achieving an immersive experience enabling users to explore virtual environments with six degrees of freedom (6DoF) is essential for various applications such as virtual reality (VR). Wide-baseline panoramas are commonly used in these applications to reduce network bandwidth and storage requirements. However, synthesizing novel views from these panoramas remains a key challenge. Although existing neural radiance field methods can produce photorealistic views under narrow-baseline and dense image captures, they tend to overfit the training views when dealing with wide-baseline panoramas due to the difficulty in learning accurate geometry from sparse $360^{\circ}$ views. To address this problem, we propose PanoGRF, Generalizable Spherical Radiance Fields for Wide-baseline Panoramas, which construct spherical radiance fields incorporating $360^{\circ}$ scene priors. Unlike generalizable radiance fields trained on perspective images, PanoGRF avoids the information loss from panorama-to-perspective conversion and directly aggregates geometry and appearance features of 3D sample points from each panoramic view based on spherical projection. Moreover, as some regions of the panorama are only visible from one view while invisible from others under wide baseline settings, PanoGRF incorporates $360^{\circ}$ monocular depth priors into spherical depth estimation to improve the geometry features. Experimental results on multiple panoramic datasets demonstrate that PanoGRF significantly outperforms state-of-the-art generalizable view synthesis methods for wide-baseline panoramas (e.g., OmniSyn) and perspective images (e.g., IBRNet, NeuRay).</details>

### PanoGen: Text-Conditioned Panoramic Environment Generation for Vision-and-Language Navigation

**Authors:** Jialu Li, Mohit Bansal

**<details><summary>Abstract**</summary>

Vision-and-Language Navigation requires the agent to follow language instructions to navigate through 3D environments. One main challenge in Vision-and-Language Navigation is the limited availability of photorealistic training environments, which makes it hard to generalize to new and unseen environments. To address this problem, we propose PanoGen, a generation method that can potentially create an infinite number of diverse panoramic environments conditioned on text. Specifically, we collect room descriptions by captioning the room images in existing Matterport3D environments, and leverage a state-of-the-art text-to-image diffusion model to generate the new panoramic environments. We use recursive outpainting over the generated images to create consistent 360-degree panorama views. Our new panoramic environments share similar semantic information with the original environments by conditioning on text descriptions, which ensures the co-occurrence of objects in the panorama follows human intuition, and creates enough diversity in room appearance and layout with image outpainting. Lastly, we explore two ways of utilizing PanoGen in VLN pre-training and fine-tuning. We generate instructions for paths in our PanoGen environments with a speaker built on a pre-trained vision-and-language model for VLN pre-training, and augment the visual observation with our panoramic environments during agents' fine-tuning to avoid overfitting to seen environments. Empirically, learning with our PanoGen environments achieves the new state-of-the-art on the Room-to-Room, Room-for-Room, and CVDN datasets. Besides, we find that pre-training with our PanoGen speaker data is especially effective for CVDN, which has under-specified instructions and needs commonsense knowledge to reach the target. Lastly, we show that the agent can benefit from training with more generated panoramic environments, suggesting promising results for scaling up the PanoGen environments to enhance agents' generalization to unseen environments.</details>

### [Spotlight] Parallel Sampling of Diffusion Models

**Authors:** Andy Shih, Suneel Belkhale, Stefano Ermon, Dorsa Sadigh, Nima Anari

**<details><summary>Abstract**</summary>

Diffusion models are powerful generative models but suffer from slow sampling, often taking 1000 sequential denoising steps for one sample. As a result, considerable efforts have been directed toward reducing the number of denoising steps, but these methods hurt sample quality. Instead of reducing the number of denoising steps (trading quality for speed), in this paper we explore an orthogonal approach: can we run the denoising steps in parallel (trading compute for speed)? In spite of the sequential nature of the denoising steps, we show that surprisingly it is possible to parallelize sampling via Picard iterations, by guessing the solution of future denoising steps and iteratively refining until convergence. With this insight, we present ParaDiGMS, a novel method to accelerate the sampling of pretrained diffusion models by denoising multiple steps in parallel. ParaDiGMS is the first diffusion sampling method that enables trading compute for speed and is even compatible with existing fast sampling techniques such as DDIM and DPMSolver. Using ParaDiGMS, we improve sampling speed by 2-4x across a range of robotics and image generation models, giving state-of-the-art sampling speeds of 0.2s on 100-step DiffusionPolicy and 14.6s on 1000-step StableDiffusion-v2 with no measurable degradation of task reward, FID score, or CLIP score.</details>

### Parameter and Computation Efficient Transfer Learning for Vision-Language Pre-trained Models

**Authors:** Qiong Wu, Wei Yu, Yiyi Zhou, Shubin Huang, Xiaoshuai Sun, Rongrong Ji

**<details><summary>Abstract**</summary>

With ever increasing parameters and computation, vision-language pre-trained (VLP) models exhibit prohibitive expenditure in downstream task adaption. Recent endeavors mainly focus on parameter efficient transfer learning (PETL) for VLP models by only updating a small number of parameters. However, excessive computational overhead still plagues the application of VLPs. In this paper, we aim at parameter and computation efficient transfer learning (PCETL) for VLP models. In particular, PCETL not only needs to limit the number of trainable parameters in VLP models, but also to reduce the computational redundancy during inference, thus enabling a more efficient transfer. To approach this target, we propose a novel dynamic architecture skipping (DAS) approach towards effective PCETL. Instead of directly optimizing the intrinsic architectures of VLP models, DAS first observes the significances of their modules to downstream tasks via a reinforcement learning (RL) based process, and then skips the redundant ones with lightweight networks, i.e. adapters, according to the obtained rewards. In this case, the VLP model can well maintain the scale of trainable parameters while speeding up its inference on downstream tasks. To validate DAS, we apply it to two representative VLP models, namely ViLT and METER, and conduct extensive experiments on a bunch of VL tasks. The experimental results not only show the great advantages of DAS in reducing computational complexity, e.g. -11.97%  FLOPs of METER on VQA2.0, but also confirm its competitiveness against existing PETL methods in terms of parameter scale and performance. Our source code is given in our appendix.</details>

### Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense

**Authors:** Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, Mohit Iyyer

**<details><summary>Abstract**</summary>

The rise in malicious usage of large language models, such as fake content creation and academic plagiarism, has motivated the development of approaches that identify AI-generated text, including those based on watermarking or outlier detection. However, the robustness of these detection algorithms to paraphrases of AI-generated text remains unclear. To stress test these detectors, we build a 11B parameter paraphrase generation model (DIPPER) that can paraphrase paragraphs, condition on surrounding context, and control lexical diversity and content reordering. Paraphrasing text generated by three large language models (including GPT3.5-davinci-003) with DIPPER successfully evades several detectors, including watermarking, GPTZero, DetectGPT, and OpenAI's text classifier. For example, DIPPER drops detection accuracy of DetectGPT from 70.3% to 4.6% (at a constant false positive rate of 1%), without appreciably modifying the input semantics.To increase the robustness of AI-generated text detection to paraphrase attacks, we introduce a simple defense that relies on retrieving semantically-similar generations and must be maintained by a language model API provider. Given a candidate text, our algorithm searches a database of sequences previously generated by the API, looking for sequences that match the candidate text within a certain threshold. We empirically verify our defense using a database of 15M generations from a fine-tuned T5-XXL model and find that it can detect 80% to 97% of paraphrased generations across different settings while only classifying 1% of human-written sequences as AI-generated. We open-source our models, code and data.</details>

### [Spotlight] Pareto Frontiers in Deep Feature Learning: Data, Compute, Width, and Luck

**Authors:** Benjamin Edelman, Surbhi Goel, Sham Kakade, Eran Malach, Cyril Zhang

**<details><summary>Abstract**</summary>

In modern deep learning, algorithmic choices (such as width, depth, and learning rate) are known to modulate nuanced resource tradeoffs. This work investigates how these complexities necessarily arise for feature learning in the presence of computational-statistical gaps. We begin by considering offline sparse parity learning, a supervised classification problem which admits a statistical query lower bound for gradient-based training of a multilayer perceptron. This lower bound can be interpreted as a *multi-resource tradeoff frontier*: successful learning can only occur if one is sufficiently rich (large model), knowledgeable (large dataset), patient (many training iterations), or lucky (many random guesses). We show, theoretically and experimentally, that sparse initialization and increasing network width yield significant improvements in sample efficiency in this setting. Here, width plays the role of parallel search: it amplifies the probability of finding "lottery ticket" neurons, which learn sparse features more sample-efficiently. Finally, we show that the synthetic sparse parity task can be useful as a proxy for real problems requiring axis-aligned feature learning. We demonstrate improved sample efficiency on tabular classification benchmarks by using wide, sparsely-initialized MLP models; these networks sometimes outperform tuned random forests.</details>

### Patch n’ Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution

**Authors:** Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim Alabdulmohsin, Avital Oliver, Piotr Padlewski, Alexey Gritsenko, Mario Lucic, Neil Houlsby

**<details><summary>Abstract**</summary>

The ubiquitous and demonstrably suboptimal choice of resizing images to a fixed resolution before processing them with computer vision models has not yet been successfully challenged. However, models such as the Vision Transformer (ViT) offer flexible sequence-based modeling, and hence varying input sequence lengths.  We take advantage of this with NaViT (Native Resolution ViT) which uses sequence packing during training to process inputs of arbitrary resolutions and aspect ratios. Alongside flexible model usage, we demonstrate improved training efficiency for large-scale supervised and contrastive image-text pretraining. NaViT can be efficiently transferred to standard tasks such as object detection, image and video classification, and leads to improved results on robustness and fairness benchmarks. At inference time, the input resolution flexibility can be used to smoothly navigate the test-time cost-performance trade-off. We believe that NaViT marks a departure from the standard, CNN-designed, input and modelling pipeline used by most computer vision models, and represents a promising direction for ViTs.</details>

### Path Regularization: A Convexity and Sparsity Inducing Regularization for Parallel ReLU Networks

**Authors:** Tolga Ergen, Mert Pilanci

**<details><summary>Abstract**</summary>

Understanding the fundamental principles behind the success of deep neural networks is one of the most important open questions in the current literature. To this end, we study the training problem of deep neural networks and introduce an analytic approach to unveil hidden convexity in the optimization landscape. We consider a deep parallel ReLU network architecture, which also includes standard deep networks and ResNets as its special cases. We then show that pathwise regularized training problems can be represented as an exact convex optimization problem. We further prove that the equivalent convex problem is regularized via a group sparsity inducing norm. Thus, a path regularized parallel ReLU network can be viewed as a parsimonious convex model in high dimensions. More importantly, since the original training problem may not be trainable in polynomial-time, we propose an approximate algorithm with a fully polynomial-time complexity in all data dimensions. Then, we prove strong global optimality guarantees for this algorithm. We also provide experiments corroborating our theory.</details>

### [Spotlight] Paxion: Patching Action Knowledge in Video-Language Foundation Models

**Authors:** Zhenhailong Wang, Ansel Blume, Sha Li, Genglin Liu, Jaemin Cho, Zineng Tang, Mohit Bansal, Heng Ji

**<details><summary>Abstract**</summary>

Action knowledge involves the understanding of textual, visual, and temporal aspects of actions. We introduce the **Action Dynamics Benchmark (ActionBench)** containing two carefully designed probing tasks: Action Antonym and Video Reversal, which targets multimodal alignment capabilities and temporal understanding skills of the model, respectively. Despite recent video-language models’ (VidLM) impressive performance on various benchmark tasks, our diagnostic tasks reveal their surprising deficiency (near-random performance) in action knowledge, suggesting that current models rely on object recognition abilities as a shortcut for action understanding. To remedy this, we propose a novel framework, **Paxion**, along with a new **Discriminative Video Dynamics Modeling (DVDM)** objective. The Paxion framework utilizes a **Knowledge Patcher** network to encode new action knowledge and a **Knowledge Fuser** component to integrate the Patcher into frozen VidLMs without compromising their existing capabilities. Due to limitations of the widely-used Video-Text Contrastive (VTC) loss for learning action knowledge, we introduce the DVDM objective to train the Knowledge Patcher. DVDM forces the model to encode the correlation between the action text and the correct ordering of video frames. Our extensive analyses show that Paxion and DVDM together effectively fill the gap in action knowledge understanding (~50% → 80%), while maintaining or improving performance on a wide spectrum of both object- and action-centric downstream tasks.</details>

### Perceptual Kalman Filters: Online State Estimation under a Perfect Perceptual-Quality Constraint

**Authors:** Dror Freirich, Tomer Michaeli, Ron Meir

**<details><summary>Abstract**</summary>

Many practical settings call for the reconstruction of temporal signals from corrupted or missing data. Classic examples include decoding, tracking, signal enhancement and denoising. Since the reconstructed signals are ultimately viewed by humans, it is desirable to achieve reconstructions that are pleasing to human perception.Mathematically, perfect perceptual-quality is achieved when the distribution of restored signals is the same as that of natural signals, a requirement which has been heavily researched in static estimation settings (i.e. when a whole signal is processed at once). Here, we study the problem of optimal causal  filtering under a perfect perceptual-quality constraint, which is a task of fundamentally different nature.  Specifically, we analyze a Gaussian Markov signal observed through a linear noisy transformation. In the absence of perceptual constraints, the Kalman filter is known to be optimal in the MSE sense for this setting. Here, we show that adding the perfect perceptual quality constraint (i.e. the requirement of temporal consistency), introduces a fundamental dilemma whereby the filter may have to ``knowingly'' ignore new information revealed by the observations in order to conform to its past decisions. This often comes at the cost of a significant increase in the MSE (beyond that encountered in static settings). Our analysis goes beyond the classic innovation process of the Kalman filter, and introduces the novel concept of an unutilized information process. Using this tool, we present a recursive formula for perceptual filters, and demonstrate the qualitative effects of perfect perceptual-quality estimation on a video reconstruction problem.</details>

### Performance-optimized deep neural networks are evolving into worse models of inferotemporal visual cortex

**Authors:** Drew Linsley, Ivan F Rodriguez Rodriguez, Thomas FEL, Michael Arcaro, Saloni Sharma, Margaret Livingstone, Thomas Serre

**<details><summary>Abstract**</summary>

One of the most impactful findings in computational neuroscience over the past decade is that the object recognition accuracy of deep neural networks (DNNs) correlates with their ability to predict neural responses to natural images in the inferotemporal (IT) cortex. This discovery supported the long-held theory that object recognition is a core objective of the visual cortex, and suggested that more accurate DNNs would serve as better models of IT neuron responses to images. Since then, deep learning has undergone a revolution of scale: billion parameter-scale DNNs trained on billions of images are rivaling or outperforming humans at visual tasks including object recognition. Have today's DNNs become more accurate at predicting IT neuron responses to images as they have grown more accurate at object recognition?Surprisingly, across three independent experiments, we find that this is not the case. DNNs have become progressively worse models of IT as their accuracy has increased on ImageNet. To understand why DNNs experience this trade-off and evaluate if they are still an appropriate paradigm for modeling the visual system, we turn to recordings of IT that capture spatially resolved maps of neuronal activity elicited by natural images. These neuronal activity maps reveal that DNNs trained on ImageNet learn to rely on different visual features than those encoded by IT and that this problem worsens as their accuracy increases. We successfully resolved this issue with the neural harmonizer, a plug-and-play training routine for DNNs that aligns their learned representations with humans. Our results suggest that harmonized DNNs break the trade-off between ImageNet accuracy and neural prediction accuracy that assails current DNNs and offer a path to more accurate models of biological vision. Our work indicates that the standard approach for modeling IT with task-optimized DNNs needs revision, and other biological constraints, including human psychophysics data, are needed to accurately reverse-engineer the visual cortex.</details>

### PolyDiffuse: Polygonal Shape Reconstruction via Guided Set Diffusion Models

**Authors:** Jiacheng Chen, Ruizhi Deng, Yasutaka Furukawa

**<details><summary>Abstract**</summary>

This paper presents \textit{PolyDiffuse}, a novel structured reconstruction algorithm that transforms visual sensor data into polygonal shapes with Diffusion Models (DM), an emerging machinery amid exploding generative AI, while formulating reconstruction as a generation process conditioned on sensor data. The task of structured reconstruction poses two fundamental challenges to DM: 1) A structured geometry is a ''set'' (e.g., a set of polygons for a floorplan geometry), where a sample of $N$ elements has $N!$ different but equivalent representations, making the denoising highly ambiguous; and 2) A ''reconstruction'' task has a single solution, where an initial noise needs to be chosen carefully, while any initial noise works for a generation task.Our technical contribution is the introduction of a Guided Set Diffusion Model where 1) the forward diffusion process learns \textit{guidance networks} to control noise injection so that one representation of a sample remains distinct from its other permutation variants, thus resolving denoising ambiguity; and 2) the reverse denoising process reconstructs polygonal shapes, initialized and directed by the guidance networks, as a conditional generation process subject to the sensor data.We have evaluated our approach for reconstructing two types of polygonal shapes: floorplan as a set of polygons and HD map for autonomous cars as a set of polylines.Through extensive experiments on standard benchmarks, we demonstrate that PolyDiffuse significantly advances the current state of the art and enables broader practical applications.</details>

### Polynomial-Time Linear-Swap Regret Minimization in Imperfect-Information Sequential Games

**Authors:** Gabriele Farina, Charilaos Pipis

**<details><summary>Abstract**</summary>

No-regret learners seek to minimize the difference between the loss they cumulated through the actions they played, and the loss they would have cumulated in hindsight had they consistently modified their behavior according to some strategy transformation function. The size of the set of transformations considered by the learner determines a natural notion of rationality. As the set of transformations each learner considers grows, the strategies played by the learners recover more complex game-theoretic equilibria, including correlated equilibria in normal-form games and extensive-form correlated equilibria in extensive-form games. At the extreme, a no-swap-regret agent is one that minimizes regret against the set of all functions from the set of strategies to itself. While it is known that the no-swap-regret condition can be attained efficiently in nonsequential (normal-form) games, understanding what is the strongest notion of rationality that can be attained efficiently in the worst case in sequential (extensive-form) games is a longstanding open problem. In this paper we provide a positive result, by showing that it is possible, in any sequential game, to retain polynomial-time (in the game tree size) iterations while achieving sublinear regret with respect to all linear transformations of the mixed strategy space, a notion called no-linear-swap regret. This notion of hindsight rationality is as strong as no-swap-regret in nonsequential games, and stronger than no-trigger-regret in sequential games—thereby proving the existence of a subset of extensive-form correlated equilibria robust to linear deviations, which we call linear-deviation correlated equilibria, that can be approached efficiently.</details>

### [Spotlight] Posterior Contraction Rates for Matérn Gaussian Processes on Riemannian Manifolds

**Authors:** Paul Rosa, Slava Borovitskiy, Alexander Terenin, Judith Rousseau

**<details><summary>Abstract**</summary>

Gaussian processes are used in many machine learning applications that rely on uncertainty quantification. Recently, computational tools for working with these models in geometric settings, such as when inputs lie on a Riemannian manifold, have been developed. This raises the question: can these intrinsic models be shown theoretically to lead to better performance, compared to simply embedding all relevant quantities into $\mathbb{R}^d$ and using the restriction of an ordinary Euclidean Gaussian process? To study this, we prove optimal contraction rates for intrinsic Matérn Gaussian processes defined on compact Riemannian manifolds. We also prove analogous rates for extrinsic processes using trace and extension theorems between manifold and ambient Sobolev spaces: somewhat surprisingly, the rates obtained turn out to coincide with those of the intrinsic processes, provided that their smoothness parameters are matched appropriately. We illustrate these rates empirically on a number of examples, which, mirroring prior work, show that intrinsic processes can achieve better performance in practice. Therefore, our work shows that finer-grained analyses are needed to distinguish between different levels of data-efficiency of geometric Gaussian processes, particularly in settings which involve small data set sizes and non-asymptotic behavior.</details>

### Posterior Sampling for Competitive RL: Function Approximation and Partial Observation

**Authors:** Shuang Qiu, Ziyu Dai, Han Zhong, Zhaoran Wang, Zhuoran Yang, Tong Zhang

**<details><summary>Abstract**</summary>

This paper investigates posterior sampling algorithms for competitive reinforcement learning (RL) in the context of general function approximations. Focusing on zero-sum Markov games (MGs) under two critical settings, namely self-play and adversarial learning, we first propose the self-play and adversarial generalized eluder coefficient (GEC) as complexity measures for function approximation, capturing the exploration-exploitation trade-off in MGs. Based on self-play GEC, we propose a model-based self-play posterior sampling method to control both players to learn Nash equilibrium, which can successfully handle the partial observability of states. Furthermore, we identify a set of partially observable MG models fitting MG learning with the adversarial policies of the opponent. Incorporating the adversarial GEC, we propose a model-based posterior sampling method for learning adversarial MG with potential partial observability. We further provide low regret bounds for proposed algorithms that can scale sublinearly with the proposed GEC and the number of episodes $T$. To the best of our knowledge, we for the first time develop generic model-based posterior sampling algorithms for competitive RL that can be applied to a majority of tractable zero-sum MG classes in both fully observable and partially observable MGs with self-play and adversarial learning.</details>

### Posterior Sampling with Delayed Feedback for Reinforcement Learning with Linear Function Approximation

**Authors:** Nikki Lijing Kuang, Ming Yin, Mengdi Wang, Yu-Xiang Wang, Yian Ma

**<details><summary>Abstract**</summary>

Recent studies in reinforcement learning (RL) have made significant progress by leveraging function approximation to alleviate the sample complexity hurdle for better performance. Despite the success, existing provably efficient algorithms typically rely on the accessibility of immediate feedback upon taking actions. The failure to account for the impact of delay in observations can significantly degrade the performance of real-world systems due to the regret blow-up. In this work, we tackle the challenge of delayed feedback in RL with linear function approximation by employing posterior sampling, which has been shown to empirically outperform the popular UCB algorithms in a wide range of regimes. We first introduce \textit{Delayed-PSVI}, an optimistic value-based algorithm that effectively explores the value function space via noise perturbation with posterior sampling. We provide the first analysis for posterior sampling algorithms with delayed feedback in RL and show our algorithm achieves $\widetilde{O}(\sqrt{d^3H^3 T} + d^2H^2 \mathbb{E}[\tau])$ worst-case regret in the presence of unknown stochastic delays. Here $\mathbb{E}[\tau]$ is the expected delay. To further improve its computational efficiency and to expand its applicability in high-dimensional RL problems, we incorporate a gradient-based approximate sampling scheme via Langevin dynamics for \textit{Delayed-LPSVI}, which maintains the same order-optimal regret guarantee with $\widetilde{O}(dHK)$ computational cost. Empirical evaluations are performed to demonstrate the statistical and computational efficacy of our algorithms.</details>

### Predict-then-Calibrate: A New Perspective of Robust Contextual LP

**Authors:** Chunlin Sun, Linyu Liu, Xiaocheng Li

**<details><summary>Abstract**</summary>

Contextual optimization, also known as predict-then-optimize or prescriptive analytics, considers an optimization problem with the presence of covariates (context or side information). The goal is to learn a prediction model (from the training data) that predicts the objective function from the covariates, and then in the test phase, solve the optimization problem with the covariates but without the observation of the objective function. In this paper, we consider a risk-sensitive version of the problem and propose a generic algorithm design paradigm called predict-then-calibrate. The idea is to first develop a prediction model without concern for the downstream risk profile or robustness guarantee, and then utilize calibration (or recalibration) methods to quantify the uncertainty of the prediction. While the existing methods suffer from either a restricted choice of the prediction model or strong assumptions on the underlying data, we show the disentangling of the prediction model and the calibration/uncertainty quantification has several advantages. First, it imposes no restriction on the prediction model and thus fully unleashes the potential of off-the-shelf machine learning methods. Second, the derivation of the risk and robustness guarantee can be made independent of the choice of the prediction model through a data-splitting idea. Third, our paradigm of predict-then-calibrate applies to both (risk-sensitive) robust and (risk-neutral) distributionally robust optimization (DRO) formulations. Theoretically, it gives new generalization bounds for the contextual LP problem and sheds light on the existing results of DRO for contextual LP. Numerical experiments further reinforce the advantage of the predict-then-calibrate paradigm in that an improvement on either the prediction model or the calibration model will lead to a better final performance.</details>

### Predicting a Protein's Stability under a Million Mutations

**Authors:** Jeffrey Ouyang-Zhang, Daniel Diaz, Adam Klivans, Philipp Kraehenbuehl

**<details><summary>Abstract**</summary>

Stabilizing proteins is a foundational step in protein engineering. However, the evolutionary pressure of all extant proteins makes identifying the scarce number of mutations that will improve thermodynamic stability challenging. Deep learning has recently emerged as a powerful tool for identifying promising mutations.Existing approaches, however, are computationally expensive, as the number of model inferences scales with the number of mutations queried. Our main contribution is a simple, parallel decoding algorithm.Mutate Everything is capable of predicting the effect of all single and double mutations in one forward pass. It is even versatile enough to predict higher-order mutations with minimal computational overhead.We build Mutate Everything on top of ESM2 and AlphaFold, neither of which were trained to predict thermodynamic stability.We trained on the Mega-Scale cDNA proteolysis dataset and achieved state-of-the-art performance on single and higher-order mutations on S669, ProTherm, and ProteinGym datasets.Our code is available at https://github.com/jozhang97/MutateEverything.</details>

### [Spotlight] Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception?

**Authors:** Xiaoxiao Sun, Nidham Gazagnadou, Vivek Sharma, Lingjuan Lyu, Hongdong Li, Liang Zheng

**<details><summary>Abstract**</summary>

Hand-crafted image quality metrics, such as PSNR and SSIM, are commonly used to evaluate model privacy risk under reconstruction attacks. Under these metrics, reconstructed images that are determined to resemble the original one generally indicate more privacy leakage. Images determined as overall dissimilar, on the other hand, indicate higher robustness against attack. However, there is no guarantee that these metrics well reflect human opinions, which offers trustworthy judgement for model privacy leakage. In this paper, we comprehensively study the faithfulness of these hand-crafted metrics to human perception of privacy information from the reconstructed images. On 5 datasets ranging from natural images, faces, to fine-grained classes, we use 4 existing attack methods to reconstruct images from many different classification models and, for each reconstructed image, we ask multiple human annotators to assess whether this image is recognizable. Our studies reveal that the hand-crafted metrics only have a weak correlation with the human evaluation of privacy leakage and that even these metrics themselves often contradict each other. These observations suggest risks of current metrics in the community. To address this potential risk, we propose a learning-based measure called SemSim to evaluate the Semantic Similarity between the original and reconstructed images. SemSim is trained with a standard triplet loss, using an original image as an anchor, one of its recognizable reconstructed images as a positive sample, and an unrecognizable one as a negative. By training on human annotations, SemSim exhibits a greater reflection of privacy leakage on the semantic level. We show that SemSim has a significantly higher correlation with human judgment compared with existing metrics. Moreover, this strong correlation generalizes to unseen datasets, models and attack methods. We envision this work as a milestone for image quality evaluation closer to the human level. The project webpage can be accessed at https://sites.google.com/view/semsim.</details>

### [Spotlight] Private (Stochastic) Non-Convex Optimization Revisited: Second-Order Stationary Points and Excess Risks

**Authors:** Daogao Liu, Arun Ganesh, Sewoong Oh, Abhradeep Guha Thakurta

**<details><summary>Abstract**</summary>

We reconsider the challenge of non-convex optimization under differential privacy constraint. Building upon the previous variance-reduced algorithm SpiderBoost, we propose a novel framework that employs two types of gradient oracles: one that estimates the gradient at a single point and a more cost-effective option that calculates the gradient difference between two points. Our framework can ensure continuous accuracy of gradient estimations and subsequently enhances the rates of identifying second-order stationary points.Additionally, we consider a more challenging task by attempting to locate the global minima of a non-convex objective via the exponential mechanism without almost any assumptions. Our preliminary results suggest that the regularized exponential mechanism can effectively emulate previous empirical and population risk bounds, negating the need for smoothness assumptions for algorithms with polynomial running time. Furthermore, with running time factors excluded, the exponential mechanism demonstrates promising population risk bound performance, and we provide a nearly matching lower bound.</details>

### Probabilistic Invariant Learning with Randomized Linear Classifiers

**Authors:** Leonardo Cotta, Gal Yehuda, Assaf Schuster, Chris Maddison

**<details><summary>Abstract**</summary>

Designing models that are both expressive and preserve known invariances of tasks is an increasingly hard problem. Existing solutions tradeoff invariance for computational or memory resources. In this work, we show how to leverage randomness and design models that are both expressive and invariant but use less resources. Inspired by randomized algorithms, our key insight is that accepting probabilistic notions of universal approximation and invariance can reduce our resource requirements. More specifically, we propose a class of binary classification models called Randomized Linear Classifiers (RLCs). We give parameter and sample size conditions in which RLCs can, with high probability, approximate any (smooth) function while preserving invariance to compact group transformations. Leveraging this result, we design three RLCs that are provably probabilistic invariant for classification tasks over sets, graphs, and spherical data. We show how these models can achieve probabilistic invariance and universality using less resources than (deterministic) neural networks and their invariant counterparts. Finally, we empirically demonstrate the benefits of this new class of models on invariant tasks where deterministic invariant neural networks are known to struggle.</details>

### Projection-Free Methods for Solving Nonconvex-Concave Saddle Point Problems

**Authors:** Morteza Boroun, Erfan Yazdandoost Hamedani, Afrooz Jalilzadeh

**<details><summary>Abstract**</summary>

In this paper, we investigate a class of constrained saddle point (SP) problems where the objective function is nonconvex-concave and smooth. This class of problems has wide applicability in machine learning, including robust multi-class classification and dictionary learning. Several projection-based primal-dual methods have been developed to tackle this problem; however, the availability of methods with projection-free oracles remains limited. To address this gap, we propose efficient single-loop projection-free methods reliant on first-order information. In particular, using regularization and nested approximation techniques, we propose a primal-dual conditional gradient method that solely employs linear minimization oracles to handle constraints. Assuming that the constraint set in the maximization is strongly convex, our method achieves an $\epsilon$-stationary solution within $\mathcal{O}(\epsilon^{-6})$ iterations. When the projection onto the constraint set of maximization is easy to compute, we propose a one-sided projection-free method that achieves an $\epsilon$-stationary solution within $\mathcal{O}(\epsilon^{-4})$ iterations. Moreover, we present improved iteration complexities of our methods under a strong concavity assumption. To the best of our knowledge, our proposed algorithms are among the first projection-free methods with convergence guarantees for solving nonconvex-concave SP problems.</details>

### [Spotlight] Promises and Pitfalls of Threshold-based Auto-labeling

**Authors:** Harit Vishwakarma, Heguang Lin, Frederic Sala, Ramya Korlakai Vinayak

**<details><summary>Abstract**</summary>

Creating large-scale high-quality labeled datasets is a major bottleneck in supervised machine learning workflows. Threshold-based auto-labeling (TBAL), where validation data obtained from humans is used to find a confidence threshold above which the data is machine-labeled, reduces reliance on manual annotation. TBAL is emerging as a widely-used solution in practice. Given the long shelf-life and diverse usage of the resulting datasets, understanding when the data obtained by such auto-labeling systems can be relied on is crucial. This is the first work to analyze TBAL systems and derive sample complexity bounds on the amount of human-labeled validation data required for guaranteeing the quality of machine-labeled data. Our results provide two crucial insights. First, reasonable chunks of unlabeled data can be automatically and accurately labeled by seemingly bad models. Second, a hidden downside of TBAL systems is potentially prohibitive validation data usage. Together, these insights describe the promise and pitfalls of using such systems. We validate our theoretical guarantees with extensive experiments on synthetic and real datasets.</details>

### [Spotlight] Provable Training for Graph Contrastive Learning

**Authors:** Yue Yu, Xiao Wang, Mengmei Zhang, Nian Liu, Chuan Shi

**<details><summary>Abstract**</summary>

Graph Contrastive Learning (GCL) has emerged as a popular training approach for learning node embeddings from augmented graphs without labels. Despite the key principle that maximizing the similarity between positive node pairs while minimizing it between negative node pairs is well established, some fundamental problems are still unclear. Considering the complex graph structure, are some nodes consistently well-trained and following this principle even with different graph augmentations? Or are there some nodes more likely to be untrained across graph augmentations and violate the principle? How to distinguish these nodes and further guide the training of GCL? To answer these questions, we first present experimental evidence showing that the training of GCL is indeed imbalanced across all nodes. To address this problem, we propose the metric "node compactness", which is the lower bound of how a node follows the GCL principle related to the range of augmentations. We further derive the form of node compactness theoretically through bound propagation, which can be integrated into binary cross-entropy as a regularization. To this end, we propose the PrOvable Training (POT) for GCL, which regularizes the training of GCL to encode node embeddings that follows the GCL principle better. Through extensive experiments on various benchmarks, POT consistently improves the existing GCL approaches, serving as a friendly plugin.</details>

### Public Opinion Field Effect Fusion in Representation Learning for Trending Topics Diffusion

**Authors:** Junliang Li, Yang Yajun, Qinghua Hu, Xin Wang, Hong Gao

**<details><summary>Abstract**</summary>

Trending topic diffusion and prediction analysis is an important problem and has been well studied in social networks. Representation learning is an effective way to extract node embeddings, which can help for topic propagation analysis by completing downstream tasks such as link prediction and node classification. In real world, there are often several trending topics or opinion leaders in public opinion space at the same time and they can be regarded as different centers of public opinion. A public opinion field will be formed surrounding every center. These public opinion fields compete for public's attention and it will potentially affect the development of public opinion. However, the existing methods do not consider public opinion field effect for trending topics diffusion. In this paper, we introduce three well-known observations about public opinion field effect in media and communication studies, and propose a novel and effective heterogeneous representation learning framework to incorporate public opinion field effect and social circle influence effect. To the best of our knowledge, our work is the first to consider these effects in representation learning for trending topic diffusion. Extensive experiments on real-world datasets validate the superiority of our model.</details>

### [Spotlight] Puzzlefusion: Unleashing the Power of Diffusion Models for Spatial Puzzle Solving

**Authors:** Sepidehsadat (Sepid) Hossieni, Mohammad Amin Shabani, Saghar Irandoust, Yasutaka Furukawa

**<details><summary>Abstract**</summary>

This paper presents an end-to-end neural architecture based on Diffusion Models for spatial puzzle solving, particularly jigsaw puzzle and room arrangement tasks.In the latter task, for instance, the proposed system ``PuzzleFusion'' takes a set of room layouts as polygonal curves in the top-down view and aligns the room layout pieces by estimating their 2D translations and rotations, akin to solving the jigsaw puzzle of room layouts. A surprising discovery of the paper is that the simple use of a Diffusion Model effectively solves these challenging spatial puzzle tasks as a conditional generation process. To enable learning of an end-to-end neural system, the paper introduces new datasets with ground-truth arrangements: 1) 2D Voronoi Jigsaw Dataset, a synthetic one where pieces are generated by voronoi diagram of 2D pointset; and 2) MagicPlan Dataset, a real one from a production pipeline by MagicPlan, where pieces are room layouts constructed by augmented reality App by real-estate consumers.The qualitative and quantitative evaluations demonstrate that the proposed approach outperforms the competing methods by significant margins in all three spatial puzzle tasks. We have provided code and data in https://sepidsh.github.io/puzzlefusion.</details>

### PyNeRF: Pyramidal Neural Radiance Fields

**Authors:** Haithem Turki, Michael Zollhöfer, Christian Richardt, Deva Ramanan

**<details><summary>Abstract**</summary>

Neural Radiance Fields (NeRFs) can be dramatically accelerated by spatial grid representations. However, they do not explicitly reason about scale and so introduce aliasing artifacts when reconstructing scenes captured at different camera distances. Mip-NeRF and its extensions propose scale-aware renderers that project volumetric frustums rather than point samples. But such approaches rely on positional encodings that are not readily compatible with grid methods. We propose a simple modification to grid-based models by training model heads at different spatial grid resolutions. At render time, we simply use coarser grids to render samples that cover larger volumes. Our method can be easily applied to existing accelerated NeRF methods and significantly improves rendering quality (reducing error rates by 20–90% across synthetic and unbounded real-world scenes) while incurring minimal performance overhead (as each model head is quick to evaluate). Compared to Mip-NeRF, we reduce error rates by 20% while training over 60x faster.</details>

### Quantum Bayesian Optimization

**Authors:** Zhongxiang Dai, Gregory Kang Ruey Lau, Arun Verma, YAO SHU, Bryan Kian Hsiang Low, Patrick Jaillet

**<details><summary>Abstract**</summary>

Kernelized bandits, also known as Bayesian optimization (BO), has been a prevalent method for optimizing complicated black-box reward functions. Various BO algorithms have been theoretically shown to enjoy upper bounds on their cumulative regret which are sub-linear in the number $T$ of iterations, and a regret lower bound of $\Omega(\sqrt{T})$ has been derived which represents the unavoidable regrets for any classical BO algorithm. Recent works on quantum bandits have shown that with the aid of quantum computing, it is possible to achieve tighter regret upper bounds better than their corresponding classical lower bounds. However, these works are restricted to either multi-armed or linear bandits, and are hence not able to solve sophisticated real-world problems with non-linear reward functions. To this end, we introduce the quantum-Gaussian process-upper confidence bound (Q-GP-UCB) algorithm. To the best of our knowledge, our Q-GP-UCB is the first BO algorithm able to achieve a regret upper bound of $\mathcal{O}(\text{poly}\log T)$, which is significantly smaller than its regret lower bound of $\Omega(\sqrt{T})$ in the classical setting. Moreover, thanks to our novel analysis of the confidence ellipsoid, our Q-GP-UCB with the linear kernel achieves a smaller regret than the quantum linear UCB algorithm from the previous work. We use simulations, as well as an experiment using a real quantum computer, to verify that the theoretical quantum speedup achieved by our Q-GP-UCB is also potentially relevant in practice.</details>

### [Oral] Quilt-1M: One Million Image-Text Pairs for Histopathology

**Authors:** Wisdom Ikezogwo, Saygin Seyfioglu, Fatemeh Ghezloo, Dylan Geva, Fatwir Sheikh Mohammed, Pavan Kumar Anand, Ranjay Krishna, Linda Shapiro

**Oral Presentation:** We, Dec 13, 14:00 -- Oral 4B

**<details><summary>Abstract**</summary>

Recent accelerations in multi-modal applications have been made possible with the plethora of image and text data available online. However, the scarcity of analogous data in the medical field, specifically in histopathology, has slowed comparable progress. To enable similar representation learning for histopathology, we turn to YouTube, an untapped resource of videos, offering $1,087$ hours of valuable educational histopathology videos from expert clinicians.From YouTube, we curate QUILT: a large-scale vision-language dataset consisting of $802, 144$ image and text pairs.QUILT was automatically curated using a mixture of models, including large language models, handcrafted algorithms, human knowledge databases, and automatic speech recognition.In comparison, the most comprehensive datasets curated for histopathology amass only around $200$K samples.We combine QUILT with datasets from other sources, including Twitter, research papers, and the internet in general, to create an even larger dataset: QUILT-1M, with $1$M paired image-text samples, marking it as the largest vision-language histopathology dataset to date. We demonstrate the value of QUILT-1M by fine-tuning a pre-trained CLIP model. Our model outperforms state-of-the-art models on both zero-shot and linear probing tasks for classifying new histopathology images across $13$ diverse patch-level datasets of $8$ different sub-pathologies and cross-modal retrieval tasks.</details>

### RDumb: A simple approach that questions our progress in continual test-time adaptation

**Authors:** Ori Press, Steffen Schneider, Matthias Kümmerer, Matthias Bethge

**<details><summary>Abstract**</summary>

Test-Time Adaptation (TTA) allows to update pre-trained models to changing data distributions at deployment time. While early work tested these algorithms for individual fixed distribution shifts, recent work proposed and applied methods for continual adaptation over long timescales. To examine the reported progress in the field, we propose the Continually Changing Corruptions (CCC) benchmark to measure asymptotic performance of TTA techniques. We find that eventually all but one state-of-the-art methods collapse and perform worse than a non-adapting model, including models specifically proposed to be robust to performance collapse. In addition, we introduce a simple baseline, "RDumb", that periodically resets the model to its pretrained state. RDumb performs better or on par with the previously proposed state-of-the-art in all considered benchmarks.Our results show that previous TTA approaches are neither effective at regularizing adaptation to avoid collapse nor able to outperform a simplistic resetting strategy.</details>

### RECKONING: Reasoning through Dynamic Knowledge Encoding

**Authors:** Zeming Chen, Gail Weiss, Eric Mitchell, Asli Celikyilmaz, Antoine Bosselut

**<details><summary>Abstract**</summary>

Recent studies on transformer-based language models show that they can answer questions by reasoning over knowledge provided as part of the context (i.e., in-context reasoning). However, since the available knowledge is often not filtered for a particular question, in-context reasoning can be sensitive to distractor facts, additional content that is irrelevant to a question but that may be relevant for a different question (i.e., not necessarily random noise). In these situations, the model fails todistinguish the necessary knowledge to answer the question, leading to spurious reasoning and degraded performance. This reasoning failure contrasts with the model’s apparent ability to distinguish its contextual knowledge from all the knowledge it has memorized during pre-training. Following this observation, we propose teaching the model to reason more robustly by folding the provided contextual knowledge into the model’s parameters before presenting it with a question. Our method, RECKONING, is a bi-level learning algorithm that teaches language models to reason by updating their parametric knowledge through back-propagation, allowing them to answer questions using the updated parameters. During training, the inner loop rapidly adapts a copy of the model weights to encode contextual knowledge into its parameters. In the outer loop, the model learns to use the updated weights to reproduce and answer reasoning questions about the memorized knowledge. Our experiments on three diverse multi-hop reasoning datasets show that RECKONING’s performance improves over the in-context reasoning baseline (by up to 4.5%). We also find that compared to in-context reasoning, RECKONING generalizes better to longer reasoning chains unseen during training, is more robust to distractors in the context, and is computationally more efficient when multiple questions are asked about the same knowledge.</details>

### REx: Data-Free Residual Quantization Error Expansion

**Authors:** Edouard YVINEC, Arnaud Dapogny, Matthieu Cord, Kevin Bailly

**<details><summary>Abstract**</summary>

Deep neural networks (DNNs) are ubiquitous in computer vision and natural language processing, but suffer from high inference cost. This problem can be addressed by quantization, which consists in converting floating point operations into a lower bit-width format. With the growing concerns on privacy rights, we focus our efforts on data-free methods. However, such techniques suffer from their lack of adaptability to the target devices, as a hardware typically only supports specific bit widths. Thus, to adapt to a variety of devices, a quantization method shall be flexible enough to find good accuracy v.s. speed trade-offs for every bit width and target device. To achieve this, we propose REx, a quantization method that leverages residual error expansion, along with group sparsity. We show experimentally that REx enables better trade-offs (in terms of accuracy given any target bit-width) on both convnets and transformers for computer vision, as well as NLP models. In particular, when applied to large language models, we show that REx elegantly solves the outlier problem that hinders state-of-the-art quantization methods.In addition, REx is backed off by strong theoretical guarantees on the preservation of the predictive function of the original model. Lastly, we show that REx is agnostic to the quantization operator and can be used in combination with previous quantization work.</details>

### RanPAC: Random Projections and Pre-trained Models for Continual Learning

**Authors:** Mark D. McDonnell, Dong Gong, Amin Parvaneh, Ehsan Abbasnejad, Anton van den Hengel

**<details><summary>Abstract**</summary>

Continual learning (CL) aims to incrementally learn different tasks (such as classification) in a non-stationary data stream without forgetting old ones. Most CL works focus on tackling catastrophic forgetting under a learning-from-scratch paradigm. However, with the increasing prominence of foundation models, pre-trained models equipped with informative representations have become available for various downstream requirements. Several CL methods based on pre-trained models have been explored, either utilizing pre-extracted features directly (which makes bridging distribution gaps challenging) or incorporating adaptors (which may be subject to forgetting). In this paper, we propose a concise and effective approach for CL with pre-trained models. Given that forgetting occurs during parameter updating, we contemplate an alternative approach that exploits training-free random projectors and class-prototype accumulation, which thus bypasses the issue. Specifically, we inject a frozen Random Projection layer with nonlinear activation between the pre-trained model's feature representations and output head, which captures interactions between features with expanded dimensionality, providing enhanced linear separability for class-prototype-based CL. We also demonstrate the importance of decorrelating the class-prototypes to reduce the distribution disparity when using pre-trained representations. These techniques prove to be effective and circumvent the problem of forgetting for both class- and domain-incremental continual learning. Compared to previous methods applied to pre-trained ViT-B/16 models, we reduce final error rates by between 10% and 62% on seven class-incremental benchmark datasets, despite not using any rehearsal memory. We conclude that the full potential of pre-trained models for simple, effective, and fast continual learning has not hitherto been fully tapped. Code is available at https://github.com/RanPAC/RanPAC.</details>

### Rank-DETR for High Quality Object Detection

**Authors:** Yifan Pu, Weicong Liang, Yiduo Hao, YUHUI YUAN, Yukang Yang, Chao Zhang, Han Hu, Gao Huang

**<details><summary>Abstract**</summary>

Modern detection transformers (DETRs) use a set of object queries to predict a list of bounding boxes, sort them by their classification confidence scores, and select the top-ranked predictions as the final detection results for the given input image. A highly performant object detector requires accurate ranking for the bounding box predictions. For DETR-based detectors, the top-ranked bounding boxes suffer from less accurate localization quality due to the misalignment between classification scores and localization accuracy, thus impeding the construction of high-quality detectors. In this work, we introduce a simple and highly performant DETR-based object detector by proposing a series of rank-oriented designs, combinedly called Rank-DETR. Our key contributions include: (i) a rank-oriented architecture design that can prompt positive predictions and suppress the negative ones to ensure lower false positive rates, as well as (ii) a rank-oriented loss function and matching cost design that prioritizes predictions of more accurate localization accuracy during ranking to boost the AP under high IoU thresholds. We apply our method to improve the recent SOTA methods (e.g., H-DETR and DINO-DETR) and report strong COCO object detection results when using different backbones such as ResNet-$50$, Swin-T, and Swin-L, demonstrating the effectiveness of our approach. Code is available at \url{https://github.com/LeapLabTHU/Rank-DETR}.</details>

### [Spotlight] Rank-N-Contrast: Learning Continuous Representations for Regression

**Authors:** Kaiwen Zha, Peng Cao, Jeany Son, Yuzhe Yang, Dina Katabi

**<details><summary>Abstract**</summary>

Deep regression models typically learn in an end-to-end fashion without explicitly emphasizing a regression-aware representation. Consequently, the learned representations exhibit fragmentation and fail to capture the continuous nature of sample orders, inducing suboptimal results across a wide range of regression tasks. To fill the gap, we propose Rank-N-Contrast (RNC), a framework that learns continuous representations for regression by contrasting samples against each other based on their rankings in the target space. We demonstrate, theoretically and empirically, that RNC guarantees the desired order of learned representations in accordance with the target orders, enjoying not only better performance but also significantly improved robustness, efficiency, and generalization. Extensive experiments using five real-world regression datasets that span computer vision, human-computer interaction, and healthcare verify that RNC achieves state-of-the-art performance, highlighting its intriguing properties including better data efficiency, robustness to spurious targets and data corruptions, and generalization to distribution shifts.</details>

### RayDF: Neural Ray-surface Distance Fields with Multi-view Consistency

**Authors:** Zhuoman Liu, Bo Yang, Yan Luximon, Ajay Kumar, Jinxi Li

**<details><summary>Abstract**</summary>

In this paper, we study the problem of continuous 3D shape representations. The majority of existing successful methods are coordinate-based implicit neural representations. However, they are inefficient to render novel views or recover explicit surface points. A few works start to formulate 3D shapes as ray-based neural functions, but the learned structures are inferior due to the lack of multi-view geometry consistency. To tackle these challenges, we propose a new framework called RayDF. It consists of three major components: 1) the simple ray-surface distance field, 2) the novel dual-ray visibility classifier, and 3) a multi-view consistency optimization module to drive the learned ray-surface distances to be multi-view geometry consistent. We extensively evaluate our method on three public datasets, demonstrating remarkable performance in 3D surface point reconstruction on both synthetic and challenging real-world 3D scenes, clearly surpassing existing coordinate-based and ray-based baselines. Most notably, our method achieves a 1000x faster speed than coordinate-based methods to render an 800x800 depth image, showing the superiority of our method for 3D shape representation. Our code and data are available at https://github.com/vLAR-group/RayDF</details>

### ReMaX: Relaxing for Better Training on Efficient Panoptic Segmentation

**Authors:** Shuyang Sun, WEIJUN WANG, Andrew Howard, Qihang Yu, Philip Torr, Liang-Chieh Chen

**<details><summary>Abstract**</summary>

This paper presents a new mechanism to facilitate the training of mask transformers for efficient panoptic segmentation, democratizing its deployment. We observe that due to the high complexity in the training objective of panoptic segmentation, it will inevitably lead to much higher penalization on false positive. Such unbalanced loss makes the training process of the end-to-end mask-transformer based architectures difficult, especially for efficient models. In this paper, we present ReMaX that adds relaxation to mask predictions and class predictions during the training phase for panoptic segmentation. We demonstrate that via these simple relaxation techniques during training, our model can be consistently improved by a clear margin without any extra computational cost on inference. By combining our method with efficient backbones like MobileNetV3-Small, our method achieves new state-of-the-art results for efficient panoptic segmentation on COCO, ADE20K and Cityscapes. Code and pre-trained checkpoints will be available at https://github.com/google-research/deeplab2.</details>

### ReSync: Riemannian Subgradient-based Robust Rotation Synchronization

**Authors:** Huikang Liu, Xiao Li, Anthony Man-Cho So

**<details><summary>Abstract**</summary>

This work presents ReSync, a Riemannian subgradient-based algorithm for solving the robust rotation synchronization problem, which arises in various engineering applications. ReSync solves a least-unsquared minimization formulation over the rotation group, which is nonsmooth and nonconvex, and aims at recovering the underlying rotations directly. We provide strong theoretical guarantees for ReSync under the random corruption setting. Specifically, we first show that the initialization procedure of ReSync yields a proper initial point that lies in a local region around the ground-truth rotations. We next establish the weak sharpness property of the aforementioned formulation and then utilize this property to derive the local linear convergence of ReSync to the ground-truth rotations. By combining these guarantees, we conclude that ReSync converges linearly to the ground-truth rotations under appropriate conditions. Experiment results demonstrate the effectiveness of ReSync.</details>

### Real-Time Motion Prediction via Heterogeneous Polyline Transformer with Relative Pose Encoding

**Authors:** Zhejun Zhang, Alexander Liniger, Christos Sakaridis, Fisher Yu, Luc V Gool

**<details><summary>Abstract**</summary>

The real-world deployment of an autonomous driving system requires its components to run on-board and in real-time, including the motion prediction module that predicts the future trajectories of surrounding traffic participants. Existing agent-centric methods have demonstrated outstanding performance on public benchmarks. However, they suffer from high computational overhead and poor scalability as the number of agents to be predicted increases. To address this problem, we introduce the K-nearest neighbor attention with relative pose encoding (KNARPE), a novel attention mechanism allowing the pairwise-relative representation to be used by Transformers. Then, based on KNARPE we present the Heterogeneous Polyline Transformer with Relative pose encoding (HPTR), a hierarchical framework enabling asynchronous token update during the online inference. By sharing contexts among agents and reusing the unchanged contexts, our approach is as efficient as scene-centric methods, while performing on par with state-of-the-art agent-centric methods. Experiments on Waymo and Argoverse-2 datasets show that HPTR achieves superior performance among end-to-end methods that do not apply expensive post-processing or model ensembling. The code is available at https://github.com/zhejz/HPTR.</details>

### Regret Minimization via Saddle Point Optimization

**Authors:** Johannes Kirschner, Alireza Bakhtiari, Kushagra Chandak, Volodymyr Tkachuk, Csaba Szepesvari

**<details><summary>Abstract**</summary>

A long line of works characterizes the sample complexity of regret minimization in sequential decision-making by min-max programs. In the corresponding saddle-point game, the min-player optimizes the sampling distribution against an adversarial max-player that chooses confusing models leading to large regret. The most recent instantiation of this idea is the decision-estimation coefficient (DEC), which was shown to provide nearly tight lower and upper bounds on the worst-case expected regret in structured bandits and reinforcement learning. By re-parametrizing the offset DEC with the confidence radius and solving the corresponding min-max program, we derive an anytime variant of the Estimation-To-Decisions algorithm (Anytime-E2D). Importantly, the algorithm optimizes the exploration-exploitation trade-off online instead of via the analysis. Our formulation leads to a practical algorithm for finite model classes and linear feedback models. We illustrate the results by deriving improved rates for high-dimensional linear bandits. Lastly, we point out connections to the information ratio, decoupling coefficient and PAC-DEC, and numerically evaluate the performance of E2D on simple examples.</details>

### [Spotlight] Regularized Behavior Cloning for Blocking the Leakage of Past Action Information

**Authors:** Seokin Seo, HyeongJoo Hwang, Hongseok Yang, Kee-Eung Kim

**<details><summary>Abstract**</summary>

For partially observable environments, imitation learning with observation histories (ILOH) assumes that control-relevant information is sufficiently captured in the observation histories for imitating the expert actions. In the offline setting wherethe agent is required to learn to imitate without interaction with the environment, behavior cloning (BC) has been shown to be a simple yet effective method for imitation learning. However, when the information about the actions executed in the past timesteps leaks into the observation histories, ILOH via BC often ends up imitating its own past actions. In this paper, we address this catastrophic failure by proposing a principled regularization for BC, which we name Past Action Leakage Regularization (PALR). The main idea behind our approach is to leverage the classical notion of conditional independence to mitigate the leakage. We compare different instances of our framework with natural choices of conditional independence metric and its estimator. The result of our comparison advocates the use of a particular kernel-based estimator for the conditional independence metric. We conduct an extensive set of experiments on benchmark datasets in order to assess the effectiveness of our regularization method. The experimental results show that our method significantly outperforms prior related approaches, highlighting its potential to successfully imitate expert actions when the past action information leaks into the observation histories.</details>

### Regularizing Neural Networks with Meta-Learning Generative Models

**Authors:** Shin'ya Yamaguchi, Daiki Chijiwa, Sekitoshi Kanai, Atsutoshi Kumagai, Hisashi Kashima

**<details><summary>Abstract**</summary>

This paper investigates methods for improving generative data augmentation for deep learning. Generative data augmentation leverages the synthetic samples produced by generative models as an additional dataset for classification with small dataset settings. A key challenge of generative data augmentation is that the synthetic data contain uninformative samples that degrade accuracy. This can be caused by the synthetic samples not perfectly representing class categories in real data and uniform sampling not necessarily providing useful samples for tasks. In this paper, we present a novel strategy for generative data augmentation called *meta generative regularization* (MGR). To avoid the degradation of generative data augmentation, MGR utilizes synthetic samples for regularizing feature extractors instead of training classifiers. These synthetic samples are dynamically determined to minimize the validation losses through meta-learning. We observed that MGR can avoid the performance degradation of naive generative data augmentation and boost the baselines. Experiments on six datasets showed that MGR is effective particularly when datasets are smaller and stably outperforms baselines by up to 7 percentage points on test accuracy.</details>

### Rehearsal Learning for Avoiding Undesired Future

**Authors:** Tian Qin, Tian-Zuo Wang, Zhi-Hua Zhou

**<details><summary>Abstract**</summary>

Machine learning (ML) models have been widely used to make predictions. Instead of a predictive statement about future outcomes, in many situations we want to pursue a decision: what can we do to avoid the undesired future if an ML model predicts so? In this paper, we present a rehearsal learning framework, in which decisions that can persuasively avoid the happening of undesired outcomes can be found and recommended. Based on the influence relation, we characterize the generative process of variables with structural rehearsal models, consisting of a probabilistic graphical model called rehearsal graphs and structural equations, and find actionable decisions that can alter the outcome by reasoning under a Bayesian framework. Moreover, we present a probably approximately correct bound to quantify the associated risk of a decision. Experiments validate the effectiveness of the proposed rehearsal learning framework and the informativeness of the bound.</details>

### [Spotlight] Relax, it doesn’t matter how you get there: A new self-supervised approach for multi-timescale behavior analysis

**Authors:** Mehdi Azabou, Michael Mendelson, Nauman Ahad, Maks Sorokin, Shantanu Thakoor, Carolina Urzay, Eva Dyer

**<details><summary>Abstract**</summary>

Unconstrained and natural  behavior consists of dynamics that are complex and  unpredictable, especially when trying to predict what will happen  multiple steps into the future. While some success has been found in building representations of animal behavior under constrained or simplified task-based conditions, many of these models cannot be applied to free and naturalistic settings where behavior becomes increasingly hard to model. In this work, we develop a multi-task representation learning model for animal behavior that combines two novel components: (i) an action-prediction objective that aims to predict the  distribution of actions over future timesteps, and (ii) a multi-scale architecture that builds separate latent spaces to accommodate short- and long-term dynamics. After demonstrating the ability of the method to build representations of both local and global dynamics in robots in varying environments and terrains, we apply our method to the MABe 2022 Multi-Agent Behavior challenge, where our model ranks first overall on both mice and fly benchmarks. In all of these cases, we show that our model can build representations that capture the many different factors that drive behavior and solve a wide range of downstream tasks.</details>

### Repetition In Repetition Out: Towards Understanding Neural Text Degeneration from the Data Perspective

**Authors:** Huayang Li, Tian Lan, Zihao Fu, Deng Cai, Lemao Liu, Nigel Collier, Taro Watanabe, Yixuan Su

**<details><summary>Abstract**</summary>

There are a number of diverging hypotheses about the neural text degeneration problem, i.e., generating repetitive and dull loops, which makes this problem both interesting and confusing. In this work, we aim to advance our understanding by presenting a straightforward and fundamental explanation from the data perspective. Our preliminary investigation reveals a strong correlation between the degeneration issue and the presence of repetitions in training data. Subsequent experiments also demonstrate that by selectively dropping out the attention to repetitive words in training data, degeneration can be significantly minimized. Furthermore, our empirical analysis illustrates that prior works addressing the degeneration issue from various standpoints, such as the high-inflow words, the likelihood objective, and the self-reinforcement phenomenon, can be interpreted by one simple explanation. That is, penalizing the repetitions in training data is a common and fundamental factor for their effectiveness. Moreover, our experiments reveal that penalizing the repetitions in training data remains critical even when considering larger model sizes and instruction tuning.</details>

### Reproducibility in Multiple Instance Learning: A Case For Algorithmic Unit Tests

**Authors:** Edward Raff, James Holt

**<details><summary>Abstract**</summary>

Multiple Instance Learning (MIL) is a sub-domain of classification problems with positive and negative labels and a "bag" of inputs, where the label is positive if and only if a positive element is contained within the bag, and otherwise is negative. Training in this context requires associating the bag-wide label to instance-level information, and implicitly contains a causal assumption and asymmetry to the task (i.e., you can't swap the labels without changing the semantics). MIL problems occur in healthcare (one malignant cell indicates cancer), cyber security (one malicious executable makes an infected computer), and many other tasks. In this work, we examine five of the most prominent deep-MIL models and find that none of them respects the standard MIL assumption. They are able to learn anti-correlated instances, i.e., defaulting to "positive" labels until seeing a negative counter-example, which should not be possible for a correct MIL model. We suspect that enhancements and other works derived from these models will share the same issue. In any context in which these models are being used, this creates the potential for learning incorrect models, which creates risk of operational failure.  We identify and demonstrate this problem via a proposed ``algorithmic unit test'', where we create synthetic datasets that can be solved by a MIL respecting model, and which clearly reveal learning that violates MIL assumptions. The five evaluated methods each fail one or more of these tests. This provides a model-agnostic way to identify violations of modeling assumptions, which we hope will be useful for future development and evaluation of MIL models.</details>

### Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone

**Authors:** Zeyinzi Jiang, Chaojie Mao, Ziyuan Huang, Ao Ma, Yiliang Lv, Yujun Shen, Deli Zhao, Jingren Zhou

**<details><summary>Abstract**</summary>

Parameter-efficient tuning has become a trend in transferring large-scale foundation models to downstream applications. Existing methods typically embed some light-weight tuners into the backbone, where both the design and the learning of the tuners are highly dependent on the base model. This work offers a new tuning paradigm, dubbed Res-Tuning, which intentionally unbinds tuners from the backbone. With both theoretical and empirical evidence, we show that popular tuning approaches have their equivalent counterparts under our unbinding formulation, and hence can be integrated into our framework effortlessly. Thanks to the structural disentanglement, we manage to free the design of tuners from the network architecture, facilitating flexible combination of various tuning strategies. We further propose a memory-efficient variant of Res-Tuning, where the bypass i.e., formed by a sequence of tuners) is effectively detached from the main branch, such that the gradients are back-propagated only to the tuners but not to the backbone. Such a detachment also allows one-time backbone forward for multi-task inference. Extensive experiments on both discriminative and generative tasks demonstrate the superiority of our method over existing alternatives from the perspectives of efficacy and efficiency. Project page: https://res-tuning.github.io/.</details>

### Reusing Pretrained Models by Multi-linear Operators for Efficient Training

**Authors:** Yu Pan, Ye Yuan, Yichun Yin, Zenglin Xu, Lifeng Shang, Xin Jiang, Qun Liu

**<details><summary>Abstract**</summary>

Training large models from scratch usually costs a substantial amount of resources. Towards this problem, recent studies such as bert2BERT and LiGO have reused small pretrained models to initialize a large model (termed the ``target model''), leading to a considerable acceleration in training. Despite the successes of these previous studies, they  grew pretrained models by mapping partial weights only, ignoring potential correlations across the entire model. As we show in this paper, there are inter- and intra-interactions among the weights of both the pretrained and the target models. As a result, the partial mapping may not capture the complete information and lead to inadequate growth. In this paper, we propose a method that linearly correlates each weight of the target model to all the weights of the pretrained model to further enhance acceleration ability. We utilize multi-linear operators to reduce computational and spacial complexity, enabling acceptable resource requirements. Experiments demonstrate that our method can save 76\% computational costs on DeiT-base transferred from DeiT-small, which outperforms bert2BERT by +12\% and LiGO by +21\%, respectively.</details>

### RevColV2: Exploring Disentangled Representations in Masked Image Modeling

**Authors:** Qi Han, Yuxuan Cai, Xiangyu Zhang

**<details><summary>Abstract**</summary>

Masked image modeling (MIM) has become a prevalent pre-training setup for vision foundation models and attains promising performance. Despite its success, existing MIM methods discard the decoder network during downstream applica- tions, resulting in inconsistent representations between pre-training and fine-tuning and can hamper downstream task performance. In this paper, we propose a new architecture, RevColV2, which tackles this issue by keeping the entire autoen- coder architecture during both pre-training and fine-tuning. The main body of RevColV2 contains bottom-up columns and top-down columns, between which information is reversibly propagated and gradually disentangled. Such design enables our architecture with the nice property: maintaining disentangled low-level and semantic information at the end of the network in MIM pre-training. Our experimental results suggest that a foundation model with decoupled features can achieve competitive performance across multiple downstream vision tasks such as image classification, semantic segmentation and object detection. For exam- ple, after intermediate fine-tuning on ImageNet-22K dataset, RevColV2-L attains 88.4\% top-1 accuracy on ImageNet-1K classification and 58.6 mIoU on ADE20K semantic segmentation. With extra teacher and large scale dataset, RevColv2-L achieves 62.1 APbox on COCO detection and 60.4 mIoU on ADE20K semantic segmentation.</details>

### Revisiting Area Convexity: Faster Box-Simplex Games and Spectrahedral Generalizations

**Authors:** Arun Jambulapati, Kevin Tian

**<details><summary>Abstract**</summary>

We investigate area convexity [Sherman17], a mysterious tool introduced to tackle optimization problems under the challenging $\ell_\infty$ geometry. We develop a deeper understanding of its relationship with conventional analyses of extragradient methods [Nemirovski04, Nesterov07]. We also give improved solvers for the subproblems required by variants of the [Sherman17] algorithm, designed through the lens of relative smoothness [BBT17, LFN18}.Leveraging these new tools, we give a state-of-the-art first-order algorithm for solving box-simplex games (a primal-dual formulation of $\ell_\infty$ regression) in a $d \times n$ matrix with bounded rows, using $O(\log d \cdot \epsilon^{-1})$ matrix-vector queries. As a consequence, we obtain improved complexities for approximate maximum flow, optimal transport, min-mean-cycle, and other basic combinatorial optimization problems. We also develop a near-linear time algorithm for a matrix generalization of box-simplex games, capturing a family of problems closely related to semidefinite programs recently used as subroutines in robust statistics and numerical linear algebra.</details>

### Revisiting Logistic-softmax Likelihood in Bayesian Meta-Learning for Few-Shot Classification

**Authors:** Tianjun Ke, Haoqun Cao, Zenan Ling, Feng Zhou

**<details><summary>Abstract**</summary>

Meta-learning has demonstrated promising results in few-shot classification (FSC) by learning to solve new problems using prior knowledge. Bayesian methods are effective at characterizing uncertainty in FSC, which is crucial in high-risk fields. In this context, the logistic-softmax likelihood is often employed as an alternative to the softmax likelihood in multi-class Gaussian process classification due to its conditional conjugacy property. However, the theoretical property of logistic-softmax is not clear and previous research indicated that the inherent uncertainty of logistic-softmax leads to suboptimal performance. To mitigate these issues, we revisit and redesign the logistic-softmax likelihood, which enables control of the 	extit{a priori} confidence level through a temperature parameter. Furthermore, we theoretically and empirically show that softmax can be viewed as a special case of logistic-softmax and logistic-softmax induces a larger family of data distribution than softmax. Utilizing modified logistic-softmax, we integrate the data augmentation technique into the deep kernel based Gaussian process meta-learning framework, and derive an analytical mean-field approximation for task-specific updates. Our approach yields well-calibrated uncertainty estimates and achieves comparable or superior results on standard benchmark datasets. Code is publicly available at \url{https://github.com/keanson/revisit-logistic-softmax}.</details>

### Riemannian Laplace approximations for Bayesian neural networks

**Authors:** Federico Bergamin, Pablo Moreno-Muñoz, Søren Hauberg, Georgios Arvanitidis

**<details><summary>Abstract**</summary>

Bayesian neural networks often approximate the weight-posterior with a Gaussian distribution. However, practical posteriors are often, even locally, highly non-Gaussian, and empirical performance deteriorates. We propose a simple parametric approximate posterior that adapts to the shape of the true posterior through a Riemannian metric that is determined by the log-posterior gradient. We develop a Riemannian Laplace approximation where samples naturally fall into weight-regions with low negative log-posterior. We show that these samples can be drawn by solving a system of ordinary differential equations, which can be done efficiently by leveraging the structure of the Riemannian metric and automatic differentiation. Empirically, we demonstrate that our approach consistently improves over the conventional Laplace approximation across tasks. We further show that, unlike the conventional Laplace approximation, our method is not overly sensitive to the choice of prior, which alleviates a practical pitfall of current approaches.</details>

### Riemannian Residual Neural Networks

**Authors:** Isay Katsman, Eric Chen, Sidhanth Holalkere, Anna Asch, Aaron Lou, Ser Nam Lim, Christopher De Sa

**<details><summary>Abstract**</summary>

Recent methods in geometric deep learning have introduced various neural networks to operate over data that lie on Riemannian manifolds. Such networks are often necessary to learn well over graphs with a hierarchical structure or to learn over manifold-valued data encountered in the natural sciences. These networks are often inspired by and directly generalize standard Euclidean neural networks. However, extending Euclidean networks is difficult and has only been done for a select few manifolds. In this work, we examine the residual neural network (ResNet) and show how to extend this construction to general Riemannian manifolds in a geometrically principled manner. Originally introduced to help solve the vanishing gradient problem, ResNets have become ubiquitous in machine learning due to their beneficial learning properties, excellent empirical results, and easy-to-incorporate nature when building varied neural networks. We find that our Riemannian ResNets mirror these desirable properties: when compared to existing manifold neural networks designed to learn over hyperbolic space and the manifold of symmetric positive definite matrices, we outperform both kinds of networks in terms of relevant testing metrics and training dynamics.</details>

### Risk-Averse Active Sensing for Timely Outcome Prediction under Cost Pressure

**Authors:** Yuchao Qin, Mihaela van der Schaar, Changhee Lee

**<details><summary>Abstract**</summary>

Timely outcome prediction is essential in healthcare to enable early detection and intervention of adverse events. However, in longitudinal follow-ups to patients' health status, cost-efficient acquisition of patient covariates is usually necessary due to the significant expense involved in screening and lab tests. To balance the timely and accurate outcome predictions with acquisition costs, an effective active sensing strategy is crucial. In this paper, we propose a novel risk-averse active sensing approach RAS that addresses the composite decision problem of when to conduct the acquisition and which measurements to make. Our approach decomposes the policy into two sub-policies: acquisition scheduler and feature selector, respectively. Moreover, we introduce a novel risk-aversion training strategy to focus on the underrepresented subgroup of high-risk patients for whom timely and accurate prediction of disease progression is of greater value. Our method outperforms baseline active sensing approaches in experiments with both synthetic and real-world datasets, and we illustrate the significance of our policy decomposition and the necessity of a risk-averse sensing policy through case studies.</details>

### Robust and Actively Secure Serverless Collaborative Learning

**Authors:** Nicholas Franzese, Adam Dziedzic, Christopher A. Choquette-Choo, Mark R Thomas, Muhammad Ahmad Kaleem, Stephan Rabanser, Congyu Fang, Somesh Jha, Nicolas Papernot, Xiao Wang

**<details><summary>Abstract**</summary>

Collaborative machine learning (ML) is widely used to enable institutions to learn better models from distributed data. While collaborative approaches to learning intuitively protect user data, they remain vulnerable to either the server, the clients, or both, deviating from the protocol. Indeed, because the protocol is asymmetric, a malicious server can abuse its power to reconstruct client data points. Conversely, malicious clients can corrupt learning with malicious updates. Thus, both clients and servers require a guarantee when the other cannot be trusted to fully cooperate. In this work, we propose a peer-to-peer (P2P) learning scheme that is secure against malicious servers and robust to malicious clients. Our core contribution is a generic framework that transforms any (compatible) algorithm for robust aggregation of model updates to the setting where servers and clients can act maliciously. Finally, we demonstrate the computational efficiency of our approach even with 1-million parameter models trained by 100s of peers on standard datasets.</details>

### SE(3) Diffusion Model-based Point Cloud Registration for Robust 6D Object Pose Estimation

**Authors:** Haobo Jiang, Mathieu Salzmann, Zheng Dang, Jin Xie, Jian Yang

**<details><summary>Abstract**</summary>

In this paper, we introduce an SE(3) diffusion model-based point cloud registration framework for 6D object pose estimation in real-world scenarios. Our approach formulates the 3D registration task as a denoising diffusion process, which progressively refines the pose of the source point cloud to obtain a precise alignment with the model point cloud. Training our framework involves two operations: An SE(3) diffusion process and an SE(3) reverse process. The SE(3) diffusion process gradually perturbs the optimal rigid transformation of a pair of point clouds by continuously injecting noise (perturbation transformation). By contrast, the SE(3) reverse process focuses on learning a denoising network that refines the noisy transformation step-by-step, bringing it closer to the optimal transformation for accurate pose estimation. Unlike standard diffusion models used in linear Euclidean spaces, our diffusion model operates on the SE(3) manifold. This requires exploiting the linear Lie algebra $\mathfrak{se}(3)$ associated with SE(3) to constrain the transformation transitions during the diffusion and reverse processes. Additionally, to effectively train our denoising network, we derive a registration-specific variational lower bound as the optimization objective for model learning. Furthermore, we show that our denoising network can be constructed with a surrogate registration model, making our approach applicable to different deep registration networks. Extensive experiments demonstrate that our diffusion registration framework presents outstanding pose estimation performance on the real-world TUD-L, LINEMOD, and Occluded-LINEMOD datasets.</details>

### SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models

**Authors:** Martin Gonzalez, Nelson Fernandez Pinto, Thuy Tran, elies Gherbi, Hatem Hajri, Nader Masmoudi

**<details><summary>Abstract**</summary>

A potent class of generative models known as Diffusion Probabilistic Models(DPMs) has become prominent. A forward diffusion process adds gradually noiseto data, while a model learns to gradually denoise. Sampling from pre-trainedDPMs is obtained by solving differential equations (DE) defined by the learntmodel, a process which has shown to be prohibitively slow. Numerous efforts onspeeding-up this process have consisted on crafting powerful ODE solvers.Despite being quick, such solvers do not usually reach the optimal qualityachieved by available slow SDE solvers. Our goal is to propose SDE solvers thatreach optimal quality without requiring several hundreds or thousands of NFEsto achieve that goal. We propose Stochastic Explicit ExponentialDerivative-free Solvers (SEEDS), improving and generalizing ExponentialIntegrator approaches to the stochastic case on several frameworks. After carefully analyzing the formulation of exactsolutions of diffusion SDEs, we craft SEEDS to analytically compute the linearpart of such solutions. Inspired by the Exponential Time-Differencing method,SEEDS use a novel treatment of the stochastic components of solutions,enabling the analytical computation of their variance, and contains high-orderterms allowing to reach optimal quality sampling $\sim3$-$5\times$ faster than previousSDE methods. We validate our approach on several image generation benchmarks,showing that SEEDS outperform or are competitive with previous SDE solvers.Contrary to the latter, SEEDS are derivative and training free, and we fullyprove strong convergence guarantees for them.</details>

### SQ Lower Bounds for Non-Gaussian Component Analysis with Weaker Assumptions

**Authors:** Ilias Diakonikolas, Daniel Kane, Lisheng Ren, Yuxin Sun

**<details><summary>Abstract**</summary>

We study the complexity of Non-Gaussian Component Analysis (NGCA) in the Statistical Query (SQ) model.Prior work developed a methodology to prove SQ lower bounds for NGCA that have been applicable to a wide range of contexts.In particular, it was known that for any univariate distribution $A$ satisfying certain conditions,distinguishing between a standard multivariate Gaussian and a distribution that behaves like $A$ in a random hidden direction and like a standard Gaussian in the orthogonal complement, is SQ-hard.The required conditions were that (1) $A$ matches many low-order moments with a standard Gaussian,and (2) the chi-squared norm of $A$ with respect to the standard Gaussian is finite.While the moment-matching condition is clearly necessary for hardness, the chi-squared condition was only required for technical reasons.In this work, we establish that the latter condition is indeed not necessary.In particular, we prove near-optimal SQ lower bounds for NGCA under the moment-matching condition only.</details>

### [Spotlight] STEVE-1: A Generative Model for Text-to-Behavior in Minecraft

**Authors:** Shalev Lifshitz, Keiran Paster, Harris Chan, Jimmy Ba, Sheila McIlraith

**<details><summary>Abstract**</summary>

Constructing AI models that respond to text instructions is challenging, especially for sequential decision-making tasks. This work introduces an instruction-tuned Video Pretraining (VPT) model for Minecraft called STEVE-1, demonstrating that the unCLIP approach, utilized in DALL•E 2, is also effective for creating instruction-following sequential decision-making agents. STEVE-1 is trained in two steps: adapting the pretrained VPT model to follow commands in MineCLIP's latent space, then training a prior to predict latent codes from text. This allows us to finetune VPT through self-supervised behavioral cloning and hindsight relabeling, bypassing the need for costly human text annotations. By leveraging pretrained models like VPT and MineCLIP and employing best practices from text-conditioned image generation, STEVE-1 costs just $60 to train and can follow short-horizon open-ended text and visual instructions in Minecraft. STEVE-1 sets a new bar for open-ended instruction following in Minecraft with low-level controls (mouse and keyboard) and raw pixel inputs, far outperforming previous baselines and robustly completing 12 of 13 tasks in our early-game evaluation suite. We provide experimental evidence highlighting key factors for downstream performance, including pretraining, classifier-free guidance, and data scaling. All resources, including our model weights, training scripts, and evaluation tools are made available for further research.</details>

### STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning

**Authors:** Weipu Zhang, Gang Wang, Jian Sun, Yetian Yuan, Gao Huang

**<details><summary>Abstract**</summary>

Recently, model-based reinforcement learning algorithms have demonstrated remarkable efficacy  in visual input environments. These approaches begin by constructing a parameterized simulation world model of the real environment through self-supervised learning. By leveraging the imagination of the world model, the agent's policy is enhanced without the constraints of sampling from the real environment. The performance of these algorithms heavily relies on the sequence modeling and generation capabilities of the world model. However, constructing a perfectly accurate model of a complex unknown environment is nearly impossible. Discrepancies between the model and reality may cause the agent to pursue virtual goals, resulting in subpar performance in the real environment. Introducing random noise into model-based reinforcement learning has been proven beneficial.In this work, we introduce Stochastic Transformer-based wORld Model (STORM), an efficient world model architecture that combines the strong sequence modeling and generation capabilities of Transformers with the stochastic nature of variational autoencoders. STORM achieves a mean human performance of $126.7\%$ on the Atari $100$k benchmark, setting a new record among state-of-the-art methods that do not employ lookahead search techniques. Moreover, training an agent with $1.85$ hours of real-time interaction experience on a single NVIDIA GeForce RTX 3090 graphics card requires only $4.3$ hours, showcasing improved efficiency compared to previous methodologies.</details>

### Safe Exploration in Reinforcement Learning: A Generalized Formulation and Algorithms

**Authors:** Akifumi Wachi, Wataru Hashimoto, Xun Shen, Kazumune Hashimoto

**<details><summary>Abstract**</summary>

Safe exploration is essential for the practical use of reinforcement learning (RL) in many real-world scenarios. In this paper, we present a generalized safe exploration (GSE) problem as a unified formulation of common safe exploration problems. We then propose a solution of the GSE problem in the form of a meta-algorithm for safe exploration, MASE, which combines an unconstrained RL algorithm with an uncertainty quantifier to guarantee safety in the current episode while properly penalizing unsafe explorations before actual safety violation to discourage them in future episodes. The advantage of MASE is that we can optimize a policy while guaranteeing with a high probability that no safety constraint will be violated under proper assumptions. Specifically, we present two variants of MASE with different constructions of the uncertainty quantifier: one based on generalized linear models with theoretical guarantees of safety and near-optimality, and another that combines a Gaussian process to ensure safety with a deep RL algorithm to maximize the reward. Finally, we demonstrate that our proposed algorithm achieves better performance than state-of-the-art algorithms on grid-world and Safety Gym benchmarks without violating any safety constraints, even during training.</details>

### [Spotlight] Safety Verification of Decision-Tree Policies in Continuous Time

**Authors:** Christian Schilling, Anna Lukina, Emir Demirović, Kim Larsen

**<details><summary>Abstract**</summary>

Decision trees have gained popularity as interpretable surrogate models for learning-based control policies. However, providing safety guarantees for systems controlled by decision trees is an open challenge. We show that the problem is undecidable even for systems with the simplest dynamics, and PSPACE-complete for finite-horizon properties. The latter can be verified for discrete-time systems via bounded model checking. However, for continuous-time systems, such an approach requires discretization, thereby weakening the guarantees for the original system. This paper presents the first algorithm to directly verify decision-tree controlled system in continuous time. The key aspect of our method is exploiting the decision-tree structure to propagate a set-based approximation through the decision nodes. We demonstrate the effectiveness of our approach by verifying safety of several decision trees distilled to imitate neural-network policies for nonlinear systems.</details>

### [Spotlight] Sample Complexity of Forecast Aggregation

**Authors:** Tao Lin, Yiling Chen

**<details><summary>Abstract**</summary>

We consider a Bayesian forecast aggregation model where $n$ experts, after observing private signals about an unknown binary event, report their posterior beliefs about the event to a principal, who then aggregates the reports into a single prediction for the event. The signals of the experts and the outcome of the event follow a joint distribution that is unknown to the principal, but the principal has access to i.i.d. "samples" from the distribution, where each sample is a tuple of the experts' reports (not signals) and the realization of the event. Using these samples, the principal aims to find an $\varepsilon$-approximately optimal aggregator, where optimality is measured in terms of the expected squared distance between the aggregated prediction and the realization of the event. We show that the sample complexity of this problem is at least $\tilde \Omega(m^{n-2} / \varepsilon)$ for arbitrary discrete distributions, where $m$ is the size of each expert's signal space.  This sample complexity grows exponentially in the number of experts $n$. But, if the experts' signals are independent conditioned on the realization of the event, then the sample complexity is significantly reduced, to $\tilde O(1 / \varepsilon^2)$, which does not depend on $n$. Our results can be generalized to non-binary events.  The proof of our results uses a reduction from the distribution learning problem and reveals the fact that forecast aggregation is almost as difficult as distribution learning.</details>

### Sample-Efficient and Safe Deep Reinforcement Learning via Reset Deep Ensemble Agents

**Authors:** Woojun Kim, Yongjae Shin, Jongeui Park, Youngchul Sung

**<details><summary>Abstract**</summary>

Deep reinforcement learning (RL) has achieved remarkable success in solving complex tasks through its integration with deep neural networks (DNNs) as function approximators. However, the reliance on DNNs has introduced a new challenge called primacy bias, whereby these function approximators tend to prioritize early experiences, leading to overfitting. To alleviate this bias, a reset method has been proposed, which involves periodic resets of a portion or the entirety of a deep RL agent while preserving the replay buffer. However, the use of this method can result in performance collapses after executing the reset, raising concerns from the perspective of safe RL and regret minimization. In this paper, we propose a novel reset-based method that leverages deep ensemble learning to address the limitations of the vanilla reset method and enhance sample efficiency. The effectiveness of the proposed method is validated through various experiments including those in the domain of safe RL. Numerical results demonstrate its potential for real-world applications requiring high sample efficiency and safety considerations.</details>

### SatLM: Satisfiability-Aided Language Models Using Declarative Prompting

**Authors:** Xi Ye, Qiaochu Chen, Isil Dillig, Greg Durrett

**<details><summary>Abstract**</summary>

Prior work has combined chain-of-thought prompting in large language models (LLMs) with programmatic representations to perform effective and transparent reasoning. While such an approach works well for tasks that only require forward reasoning (e.g., straightforward arithmetic), it is less effective for constraint solving problems that require more sophisticated planning and search. In this paper, we propose a new satisfiability-aided language modeling (SatLM) approach for improving the reasoning capabilities of LLMs. We use an LLM to generate a declarative task specification rather than an imperative program and leverage an off-the-shelf automated theorem prover to derive the final answer. This approach has two key advantages. The declarative specification is closer to the problem description than the reasoning steps are, so the LLM can parse it out of the description more accurately. Furthermore, by offloading the actual reasoning task to an automated theorem prover, our approach can guarantee the correctness of the answer with respect to the parsed specification and avoid planning errors in the solving process. We evaluate SATLM on 8 different datasets and show that it consistently outperforms program-aided LMs in the imperative paradigm. In particular, SATLM outperforms program-aided LMs by 23% on a challenging subset of the GSM arithmetic reasoning dataset; SATLM also achieves a new SoTA on LSAT and BoardgameQA, surpassing previous models that are trained on the respective training sets.</details>

### Scale-Space Hypernetworks for Efficient Biomedical Image Analysis

**Authors:** Jose Javier Gonzalez Ortiz, John Guttag, Adrian Dalca

**<details><summary>Abstract**</summary>

Convolutional Neural Networks (CNNs) are the predominant model used for a variety of medical image analysis tasks. At inference time, these models are computationally intensive, especially with volumetric data.In principle, it is possible to trade accuracy for computational efficiency by manipulating the rescaling factor in the downsample and upsample layers of CNN architectures.However, properly exploring the accuracy-efficiency trade-off is prohibitively expensive with existing models.To address this, we introduce Scale-Space HyperNetworks (SSHN), a method that learns a spectrum of CNNs with varying internal rescaling factors.A single SSHN characterizes an entire Pareto accuracy-efficiency curve of models that match, and occasionally surpass, the outcomes of training many separate networks with fixed rescaling factors.We demonstrate the proposed approach in several medical image analysis applications, comparing SSHN against strategies with both fixed and dynamic rescaling factors.We find that SSHN consistently provides a better accuracy-efficiency trade-off at a fraction of the training cost. Trained SSHNs enable the user to quickly choose a rescaling factor that appropriately balances accuracy and computational efficiency for their particular needs at inference.</details>

### Seeing is not Believing: Robust Reinforcement Learning against Spurious Correlation

**Authors:** Wenhao Ding, Laixi Shi, Yuejie Chi, DING ZHAO

**<details><summary>Abstract**</summary>

Robustness has been extensively studied in reinforcement learning (RL) to handle various forms of uncertainty such as random perturbations, rare events, and malicious attacks. In this work, we consider one critical type of robustness against spurious correlation, where different portions of the state do not have correlations induced by unobserved confounders. These spurious correlations are ubiquitous in real-world tasks, for instance, a self-driving car usually observes heavy traffic in the daytime and light traffic at night due to unobservable human activity. A model that learns such useless or even harmful correlation could catastrophically fail when the confounder in the test case deviates from the training one. Although motivated, enabling robustness against spurious correlation poses significant challenges since the uncertainty set, shaped by the unobserved confounder and causal structure, is difficult to characterize and identify. Existing robust algorithms that assume simple and unstructured uncertainty sets are therefore inadequate to address this challenge. To solve this issue, we propose Robust State-Confounded Markov Decision Processes (RSC-MDPs) and theoretically demonstrate its superiority in avoiding learning spurious correlations compared with other robust RL counterparts. We also design an empirical algorithm to learn the robust optimal policy for RSC-MDPs, which outperforms all baselines in eight realistic self-driving and manipulation tasks.</details>

### Segment Anything in High Quality

**Authors:** Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan liu, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu

**<details><summary>Abstract**</summary>

The recent Segment Anything Model (SAM) represents a big leap in scaling up segmentation models, allowing for powerful zero-shot capabilities and flexible prompting. Despite being trained with 1.1 billion masks, SAM's mask prediction quality falls short in many cases, particularly when dealing with objects that have intricate structures. We propose HQ-SAM, equipping SAM with the ability to accurately segment any object, while maintaining SAM's original promptable design, efficiency, and zero-shot generalizability. Our careful design reuses and preserves the pre-trained model weights of SAM, while only introducing minimal additional parameters and computation. We design a learnable High-Quality Output Token, which is injected into SAM's mask decoder and is responsible for predicting the high-quality mask. Instead of only applying it on mask-decoder features, we first fuse them with early and final ViT features for improved mask details. To train our introduced learnable parameters, we compose a dataset of 44K fine-grained masks from several sources. HQ-SAM is only trained on the introduced detaset of 44k masks, which takes only 4 hours on 8 GPUs. We show the efficacy of HQ-SAM in a suite of 10 diverse segmentation datasets across different downstream tasks, where 8 out of them are evaluated in a zero-shot transfer protocol. Our code and pretrained models are at https://github.com/SysCV/SAM-HQ.</details>

### Segment Everything Everywhere All at Once

**Authors:** Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Wang, Lijuan Wang, Jianfeng Gao, Yong Jae Lee

**<details><summary>Abstract**</summary>

In this work, we present SEEM, a promotable and interactive model for segmenting everything everywhere all at once in an image. In SEEM, we propose a novel and versatile decoding mechanism that enables diverse prompting for all types of segmentation tasks, aiming at a universal interface that behaves like large language models (LLMs). More specifically, SEEM is designed with four desiderata:i) Versatility. We introduce a new visual prompt to unify different spatial queries including points, boxes, scribbles, and masks, which can further generalize to a different referring image; ii) Compositionality. We learn a joint visual-semantic space between text and visual prompts, which facilitates the dynamic composition of two prompt types required for various segmentation tasks, as shown in Fig. 1;iii) Interactivity. We further incorporate learnable memory prompts into the decoder to retain segmentation history through mask-guided cross-attention from the decoder to image features; iv) Semantic awareness. We use a text encoder to encode text queries and mask labels into the same semantic space for open-vocabulary segmentation. We conduct a comprehensive empirical study to validate the effectiveness of SEEM across diverse segmentation tasks. The results demonstrate that SEEM exhibits robust generalizing to unseen user intents as it learns to compose prompts of different types in a unified representation space. Our approach achieves competitive performance on interactive segmentation, generic segmentation, referring segmentation, and video object segmentation on 9 datasets with minimum 1/100 supervision in a single set of weights.</details>

### Self-Chained Image-Language Model for Video Localization and Question Answering

**Authors:** Shoubin Yu, Jaemin Cho, Prateek Yadav, Mohit Bansal

**<details><summary>Abstract**</summary>

Recent studies have shown promising results on utilizing large pre-trained image-language models for video question answering. While these image-language models can efficiently bootstrap the representation learning of video-language models, they typically concatenate uniformly sampled video frames as visual inputs without explicit language-aware, temporal modeling. When only a portion of a video input is relevant to the language query, such uniform frame sampling can often lead to missing important visual cues. Although humans often find a video moment to focus on and rewind the moment to answer questions, training a query-aware video moment localizer often requires expensive annotations and high computational costs. To address this issue, we propose Self-Chained Video Localization-Answering (SeViLA), a novel framework that leverages a single image-language model (BLIP- 2) to tackle both temporal keyframe localization and question answering on videos. SeViLA framework consists of two modules: Localizer and Answerer, where both are parameter-efficiently fine-tuned from BLIP-2. We propose two ways of chaining these modules for cascaded inference and self-refinement. First, in the forward chain, the Localizer finds multiple language-aware keyframes in a video, which the Answerer uses to predict the answer. Second, in the reverse chain, the Answerer generates keyframe pseudo-labels to refine the Localizer, alleviating the need for expensive video moment localization annotations. Our SeViLA framework outperforms several strong baselines/previous works on five challenging video question answering and event prediction benchmarks, and achieves the state-of-the-art in both fine-tuning (NExT-QA and STAR) and zero-shot (NExT-QA, STAR, How2QA, and VLEP) settings. We show a comprehensive analysis of our framework, including the impact of Localizer, comparisons of Localizer with other temporal localization models, pre-training/self-refinement of Localizer, and varying the number of keyframes.</details>

### Self-Refine: Iterative Refinement with Self-Feedback

**Authors:** Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, Peter Clark

**<details><summary>Abstract**</summary>

Like humans, large language models (LLMs) do not always generate the best output on their first try. Motivated by how humans refine their written text, we introduce Self-Refine, an approach for improving initial outputs from LLMs through iterative feedback and refinement. The main idea is to generate an initial output using an LLMs; then, the same LLMs provides *feedback* for its output and uses it to *refine* itself, iteratively. Self-Refine does not require any supervised training data, additional training, or reinforcement learning, and instead uses a single LLM as the generator, refiner and the feedback provider.  We evaluate Self-Refine across 7 diverse tasks, ranging from dialog response generation to mathematical reasoning, using state-of-the-art (GPT-3.5, ChatGPT, and GPT-4) LLMs. Across all evaluated tasks, outputs generated with Self-Refine are preferred by humans and automatic metrics over those generated with the same LLM using conventional one-step generation, improving by $\sim$20\% absolute on average in task performance. Our work demonstrates that even state-of-the-art LLMs like GPT-4 can be further improved at test-time using our simple, standalone approach.</details>

### Self-Supervised Learning with Lie Symmetries for Partial Differential Equations

**Authors:** Grégoire Mialon, Quentin Garrido, Hannah Lawrence, Danyal Rehman, Yann LeCun, Bobak Kiani

**<details><summary>Abstract**</summary>

Machine learning for differential equations paves the way for computationally efficient alternatives to numerical solvers, with potentially broad impacts in science and engineering. Though current algorithms typically require simulated training data tailored to a given setting, one may instead wish to learn useful information from heterogeneous sources, or from real dynamical systems observations that are messy or incomplete. In this work, we learn general-purpose representations of PDEs from heterogeneous data by implementing joint embedding methods for self-supervised learning (SSL), a framework for unsupervised representation learning that has had notable success in computer vision. Our representation outperforms baseline approaches to invariant tasks, such as regressing the coefficients of a PDE, while also improving the time-stepping performance of neural solvers. We hope that our proposed methodology will prove useful in the eventual development of general-purpose foundation models for PDEs.</details>

### Self-Supervised Visual Acoustic Matching

**Authors:** Arjun Somayazulu, Changan Chen, Kristen Grauman

**<details><summary>Abstract**</summary>

Acoustic matching aims to re-synthesize an audio clip to sound as if it were recorded in a target acoustic environment. Existing methods assume access to paired training data, where the audio is observed in both source and target environments, but this limits the diversity of training data or requires the use of simulated data or heuristics to create paired samples. We propose a self-supervised approach to visual acoustic matching where training samples include only the target scene image and audio---without acoustically mismatched source audio for reference. Our approach jointly learns to disentangle room acoustics and re-synthesize audio into the target environment, via a conditional GAN framework and a novel metric that quantifies the level of residual acoustic information in the de-biased audio. Training with either in-the-wild web data or simulated data, we demonstrate it outperforms the state-of-the-art on multiple challenging datasets and a wide variety of real-world audio and environments.</details>

### Self-supervised Graph Neural Networks via Low-Rank Decomposition

**Authors:** Liang Yang, Runjie Shi, Qiuliang Zhang, bingxin niu, Zhen Wang, Xiaochun Cao, Chuan Wang

**<details><summary>Abstract**</summary>

Self-supervised learning is introduced to train graph neural networks (GNNs) by employing propagation-based GNNs designed for semi-supervised learning tasks. Unfortunately, this common choice tends to cause two serious issues. Firstly, global parameters cause the model lack the ability to capture the local property. Secondly, it is difficult to handle networks beyond homophily without label information.This paper tends to break through the common choice of employing propagation-based GNNs, which aggregate representations of nodes belonging to different classes and tend to lose discriminative information. If the propagation in each ego-network is just between the nodes from the same class, the obtained representation matrix should follow the low-rank characteristic. To meet this requirement, this paper proposes the Low-Rank Decomposition-based GNNs (LRD-GNN-Matrix) by employing Low-Rank Decomposition to the attribute matrix. Furthermore, to incorporate long-distance information, Low-Rank Tensor Decomposition-based GNN (LRD-GNN-Tensor) is proposed by constructing the node attribute tensor from selected similar ego-networks and performing Low-Rank Tensor Decomposition. The employed tensor nuclear norm facilitates the capture of the long-distance relationship between original and selected similar ego-networks. Extensive experiments demonstrate the superior performance and the robustness  of  LRD-GNNs.</details>

### [Spotlight] Separable Physics-Informed Neural Networks

**Authors:** Junwoo Cho, Seungtae Nam, Hyunmo Yang, Seok-Bae Yun, Youngjoon Hong, Eunbyung Park

**<details><summary>Abstract**</summary>

Physics-informed neural networks (PINNs) have recently emerged as promising data-driven PDE solvers showing encouraging results on various PDEs. However, there is a fundamental limitation of training PINNs to solve multi-dimensional PDEs and approximate very complex solution functions.The number of training points (collocation points) required on these challenging PDEs grows substantially, and it is severely limited due to the expensive computational costs and heavy memory overhead.To overcome this limit, we propose a network architecture and training algorithm for PINNs.The proposed method, separable PINN (SPINN), operates on a per-axis basis to decrease the number of network propagations in multi-dimensional PDEs instead of point-wise processing in conventional PINNs.We also propose using forward-mode automatic differentiation to reduce the computational cost of computing PDE residuals, enabling a large number of collocation points ($>10^7$) on a single commodity GPU. The experimental results show significantly reduced computational costs ($62	imes$ in wall-clock time, $1,394	imes$ in FLOPs given the same number of collocation points) in multi-dimensional PDEs while achieving better accuracy.Furthermore, we present that SPINN can solve a chaotic (2+1)-d Navier-Stokes equation much faster than the best-performing prior method (9 minutes vs. 10 hours in a single GPU), maintaining accuracy.Finally, we showcase that SPINN can accurately obtain the solution of a highly nonlinear and multi-dimensional PDE, a (3+1)-d Navier-Stokes equation.For visualized results and code, please see https://jwcho5576.github.io/spinn.github.io/.</details>

### Sequential Predictive Two-Sample and Independence Testing

**Authors:** Aleksandr Podkopaev, Aaditya Ramdas

**<details><summary>Abstract**</summary>

We study the problems of sequential nonparametric two-sample and independence testing. Sequential tests process data online and allow using observed data to decide whether to stop and reject the null hypothesis or to collect more data, while maintaining type I error control. We build upon the principle of (nonparametric) testing by betting, where a gambler places bets on future observations and their wealth measures evidence against the null hypothesis. While recently developed kernel-based betting strategies often work well on simple distributions, selecting a suitable kernel for high-dimensional or structured data, such as images, is often nontrivial. To address this drawback, we design prediction-based betting strategies that rely on the following fact: if a sequentially updated predictor starts to consistently determine (a) which distribution an instance is drawn from, or (b) whether an instance is drawn from the joint distribution or the product of the marginal distributions (the latter produced by external randomization), it provides evidence against the two-sample or independence nulls respectively. We empirically demonstrate the superiority of our tests over kernel-based approaches under structured settings. Our tests can be applied beyond the case of independent and identically distributed data, remaining valid and powerful even when the data distribution drifts over time.</details>

### Sequential Subset Matching for Dataset Distillation

**Authors:** JIAWEI DU, Qin Shi, Joey Tianyi Zhou

**<details><summary>Abstract**</summary>

Dataset distillation is a newly emerging task that synthesizes a small-size dataset used in training deep neural networks (DNNs) for reducing data storage and model training costs. The synthetic datasets are expected to capture the essence of the knowledge contained in real-world datasets such that the former yields a similar performance as the latter. Recent advancements in distillation methods have produced notable improvements in generating synthetic datasets. However, current state-of-the-art methods treat the entire synthetic dataset as a unified entity and optimize each synthetic instance equally . This static optimization approach may lead to performance degradation in dataset distillation. Specifically, we argue that static optimization can give rise to a coupling issue within the synthetic data, particularly when a larger amount of synthetic data is being optimized. This coupling issue, in turn, leads to the failure of the distilled dataset to extract the high-level features learned by the deep neural network (DNN) in the latter epochs.In this study, we propose a new dataset distillation strategy called Sequential Subset Matching (SeqMatch), which tackles this problem by adaptively optimizing the synthetic data to encourage sequential acquisition of knowledge during dataset distillation. Our analysis indicates that SeqMatch effectively addresses the coupling issue by sequentially generating the synthetic instances, thereby enhancing its performance significantly. Our proposed SeqMatch outperforms state-of-the-art methods in various datasets, including SVNH, CIFAR-10, CIFAR-100, and Tiny ImageNet.</details>

### Sharp Recovery Thresholds of Tensor PCA Spectral Algorithms

**Authors:** Michael Feldman, David Donoho

**<details><summary>Abstract**</summary>

Many applications seek to recover low-rank approximations of noisy tensor data. We consider several practical and effective matricization strategies which construct specific matrices from such tensors and then apply spectral methods; the strategies include tensor unfolding, partial tracing, power iteration, and recursive unfolding. We settle the behaviors of unfolding and partial tracing, identifying sharp thresholds in signal-to-noise ratio above which the signal is partially recovered. In particular, we extend previous results to a much larger class of tensor shapes where axis lengths may be different. For power iteration and recursive unfolding, we prove that under conditions where previous algorithms partially recovery the signal, these methods achieve (asymptotically) exact recovery. Our analysis deploys random matrix theory to obtain sharp thresholds which elude perturbation and concentration bounds. Specifically, we rely upon recent disproportionate random matrix results, which describe sequences of matrices with diverging aspect ratio.</details>

### [Spotlight] SimFBO: Towards Simple, Flexible and Communication-efficient Federated Bilevel Learning

**Authors:** Yifan Yang, Peiyao Xiao, Kaiyi Ji

**<details><summary>Abstract**</summary>

Federated bilevel optimization (FBO) has shown great potential recently in machine learning and edge computing due to the emerging nested optimization structure in meta-learning, fine-tuning, hyperparameter tuning, etc. However, existing FBO algorithms often involve complicated computations and require multiple sub-loops per iteration, each of which contains a number of communication rounds. In this paper, we propose a simple and flexible FBO framework named SimFBO, which is easy to implement without sub-loops, and includes a generalized server-side aggregation and update for improving communication efficiency. We further propose System-level heterogeneity robust FBO (ShroFBO) as a variant of SimFBO with stronger resilience to heterogeneous local computation. We show that SimFBO and ShroFBO provably achieve a linear convergence speedup with partial client participation and client sampling without replacement, as well as improved sample and communication complexities. Experiments demonstrate the effectiveness of the proposed methods over existing FBO algorithms.</details>

### SimMMDG: A Simple and Effective Framework for Multi-modal Domain Generalization

**Authors:** Hao Dong, Ismail Nejjar, Han Sun, Eleni Chatzi, Olga Fink

**<details><summary>Abstract**</summary>

In real-world scenarios, achieving domain generalization (DG) presents significant challenges as models are required to generalize to unknown target distributions. Generalizing to unseen multi-modal distributions poses even greater difficulties due to the distinct properties exhibited by different modalities. To overcome the challenges of achieving domain generalization in multi-modal scenarios, we propose SimMMDG, a simple yet effective multi-modal DG framework. We argue that mapping features from different modalities into the same embedding space impedes model generalization. To address this, we propose splitting the features within each modality into modality-specific and modality-shared components. We employ supervised contrastive learning on the modality-shared features to ensure they possess joint properties and impose distance constraints on modality-specific features to promote diversity. In addition, we introduce a cross-modal translation module to regularize the learned features, which can also be used for missing-modality generalization. We demonstrate that our framework is theoretically well-supported and achieves strong performance in multi-modal DG on the EPIC-Kitchens dataset and the novel Human-Animal-Cartoon (HAC) dataset introduced in this paper. Our source code and HAC dataset are available at https://github.com/donghao51/SimMMDG.</details>

### Simple and Asymmetric Graph Contrastive Learning without Augmentations

**Authors:** Teng Xiao, Huaisheng Zhu, Zhengyu Chen, Suhang Wang

**<details><summary>Abstract**</summary>

Graph Contrastive Learning (GCL) has shown superior performance in representation learning in graph-structured data. Despite their success, most existing GCL methods rely on prefabricated graph augmentation and homophily assumptions. Thus, they fail to generalize well to heterophilic graphs where connected nodes may have different class labels and dissimilar features. In this paper, we study the problem of conducting contrastive learning on homophilic and heterophilic graphs. We find that we can achieve promising performance simply by considering an asymmetric view of the neighboring nodes. The resulting simple algorithm, Asymmetric Contrastive Learning for Graphs (GraphACL), is easy to implement and does not rely on graph augmentations and homophily assumptions. We provide theoretical and empirical evidence that GraphACL can capture one-hop local neighborhood information and two-hop monophily similarity, which are both important for modeling heterophilic graphs. Experimental results show that the simple GraphACL significantly outperforms state-of-the-art graph contrastive learning and self-supervised learning methods on homophilic and heterophilic graphs. The code of GraphACL is available at https://github.com/tengxiao1/GraphACL.</details>

### Simplifying and Empowering Transformers for Large-Graph Representations

**Authors:** Qitian Wu, Wentao Zhao, Chenxiao Yang, Hengrui Zhang, Fan Nie, Haitian Jiang, Yatao Bian, Junchi Yan

**<details><summary>Abstract**</summary>

Learning representations on large-sized graphs is a long-standing challenge due to the inter-dependence nature involved in massive data points. Transformers, as an emerging class of foundation encoders for graph-structured data, have shown promising performance on small graphs due to its global attention capable of capturing all-pair influence beyond neighboring nodes. Even so, existing approaches tend to inherit the spirit of Transformers in language and vision tasks, and embrace complicated models by stacking deep multi-head attentions. In this paper, we critically demonstrate that even using a one-layer attention can bring up surprisingly competitive performance across node property prediction benchmarks where node numbers range from thousand-level to billion-level. This encourages us to rethink the design philosophy for Transformers on large graphs, where the global attention is a computation overhead hindering the scalability. We frame the proposed scheme as Simplified Graph Transformers (SGFormer), which is empowered by a simple attention model that can efficiently propagate information among arbitrary nodes in one layer. SGFormer requires none of positional encodings, feature/graph pre-processing or augmented loss. Empirically, SGFormer successfully scales to the web-scale graph ogbn-papers100M and yields up to 141x inference acceleration over SOTA Transformers on medium-sized graphs. Beyond current results, we believe the proposed methodology alone enlightens a new technical path of independent interest for building Transformers on large graphs.</details>

### SmooSeg: Smoothness Prior for Unsupervised Semantic Segmentation

**Authors:** Mengcheng Lan, Xinjiang Wang, Yiping Ke, Jiaxing Xu, Litong Feng, Wayne Zhang

**<details><summary>Abstract**</summary>

Unsupervised semantic segmentation is a challenging task that segments images into semantic groups without manual annotation. Prior works have primarily focused on leveraging prior knowledge of semantic consistency or priori concepts from self-supervised learning methods, which often overlook the coherence property of image segments. In this paper, we demonstrate that the smoothness prior, asserting that close features in a metric space share the same semantics, can significantly simplify segmentation by casting unsupervised semantic segmentation as an energy minimization problem. Under this paradigm, we propose a novel approach called SmooSeg that harnesses self-supervised learning methods to model the closeness relationships among observations as smoothness signals. To effectively discover coherent semantic segments, we introduce a novel smoothness loss that promotes piecewise smoothness within segments while preserving discontinuities across different segments. Additionally, to further enhance segmentation quality, we design an asymmetric teacher-student style predictor that generates smoothly updated pseudo labels, facilitating an optimal fit between observations and labeling outputs. Thanks to the rich supervision cues of the smoothness prior, our SmooSeg significantly outperforms STEGO in terms of pixel accuracy on three datasets: COCOStuff (+14.9\%), Cityscapes (+13.0\%), and Potsdam-3 (+5.7\%).</details>

### [Spotlight] Smoothed Online Learning for Prediction in Piecewise Affine Systems

**Authors:** Adam Block, Max Simchowitz, Russ Tedrake

**<details><summary>Abstract**</summary>

The problem of piecewise affine (PWA) regression and planning is of foundational importance to the study of online learning, control, and robotics, where it provides a theoretically and empirically tractable setting to study systems undergoing sharp changes in the dynamics.  Unfortunately, due to the discontinuities that arise when crossing into different ``pieces,'' learning in general sequential settings is impossible and practical algorithms are forced to resort to heuristic approaches.  This paper builds on the recently developed smoothed online learning framework and provides the first algorithms for prediction and simulation in PWA systems whose regret is polynomial in all relevant problem parameters under a weak smoothness assumption; moreover, our algorithms are efficient in the number of calls to an optimization oracle.  We further apply our results to the problems of one-step prediction and multi-step simulation regret in piecewise affine dynamical systems, where the learner is tasked with simulating trajectories and regret is measured in terms of the Wasserstein distance between simulated and true data.  Along the way, we develop several technical tools of more general interest.</details>

### [Oral] Smoothing the Landscape Boosts the Signal for SGD: Optimal Sample Complexity for Learning Single Index Models

**Authors:** Alex Damian, Eshaan Nichani, Rong Ge, Jason Lee

**Oral Presentation:** We, Dec 13, 13:45 -- Oral 4A

**<details><summary>Abstract**</summary>

We focus on the task of learning a single index model $\sigma(w^\star \cdot x)$ with respect to the isotropic Gaussian distribution in $d$ dimensions. Prior work has shown that the sample complexity of learning $w^\star$ is governed by the information exponent $k^\star$ of the link function $\sigma$, which is defined as the index of the first nonzero Hermite coefficient of $\sigma$. Ben Arous et al. (2021) showed that $n \gtrsim d^{k^\star-1}$ samples suffice for learning $w^\star$ and that this is tight for online SGD. However, the CSQ lower bound for gradient based methods only shows that $n \gtrsim d^{k^\star/2}$ samples are necessary. In this work, we close the gap between the upper and lower bounds by showing that online SGD on a smoothed loss learns $w^\star$ with $n \gtrsim d^{k^\star/2}$ samples. We also draw connections to statistical analyses of tensor PCA and to the implicit regularization effects of minibatch SGD on empirical losses.</details>

### Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models

**Authors:** Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alex Dimakis, Sanjay Shakkottai

**<details><summary>Abstract**</summary>

We present the first framework to solve linear inverse problems leveraging pre-trained \textit{latent} diffusion models.  Previously proposed algorithms (such as DPS and DDRM) only apply to \textit{pixel-space} diffusion models.  We theoretically analyze our algorithm showing provable sample recovery in a linear model setting. The algorithmic insight obtained from our analysis extends to more general settings often considered in practice. Experimentally, we outperform previously proposed posterior sampling algorithms in a wide variety of problems including random inpainting, block inpainting, denoising, deblurring, destriping, and super-resolution.</details>

### Solving a Class of Non-Convex  Minimax Optimization in Federated Learning

**Authors:** Xidong Wu, Jianhui Sun, Zhengmian Hu, Aidong Zhang, Heng Huang

**<details><summary>Abstract**</summary>

The minimax problems arise throughout machine learning applications, ranging from adversarial training and policy evaluation in reinforcement learning to AUROC maximization. To address the large-scale distributed data challenges across multiple clients with communication-efficient distributed training, federated learning (FL) is gaining popularity. Many optimization algorithms for minimax problems have been developed in the centralized setting (\emph{i.e.}, single-machine). Nonetheless, the algorithm for minimax problems under FL is still underexplored. In this paper, we study a class of federated nonconvex minimax optimization problems. We propose FL algorithms (FedSGDA+ and FedSGDA-M) and reduce existing complexity results for the most common minimax problems. For nonconvex-concave problems, we propose FedSGDA+ and reduce the communication complexity to $O(\varepsilon^{-6})$. Under nonconvex-strongly-concave and nonconvex-PL minimax settings, we prove that FedSGDA-M has the best-known sample complexity of $O(\kappa^{3} N^{-1}\varepsilon^{-3})$ and the best-known communication complexity of $O(\kappa^{2}\varepsilon^{-2})$. FedSGDA-M is the first algorithm to match the best sample complexity $O(\varepsilon^{-3})$ achieved by the single-machine method under the nonconvex-strongly-concave setting. Extensive experimental results on fair classification and AUROC maximization show the efficiency of our algorithms.</details>

### Sorting with Predictions

**Authors:** Xingjian Bai, Christian Coester

**<details><summary>Abstract**</summary>

We explore the fundamental problem of sorting through the lens of learning-augmented algorithms, where algorithms can leverage possibly erroneous predictions to improve their efficiency. We consider two different settings: In the first setting, each item is provided a prediction of its position in the sorted list. In the second setting, we assume there is a ``quick-and-dirty'' way of comparing items, in addition to slow-and-exact comparisons. For both settings, we design new and simple algorithms using only $O(\sum_i \log \eta_i)$ exact comparisons, where $\eta_i$ is a suitably defined prediction error for the $i$th element. In particular, as the quality of predictions deteriorates, the number of comparisons degrades smoothly from $O(n)$ to $O(n\log n)$. We prove that this comparison complexity is theoretically optimal with respect to the examined error measures. An experimental evaluation against existing adaptive and non-adaptive sorting algorithms demonstrates the potential of applying learning-augmented algorithms in sorting tasks.</details>

### Sparse Modular Activation for Efficient Sequence Modeling

**Authors:** Liliang Ren, Yang Liu, Shuohang Wang, Yichong Xu, Chenguang Zhu, Cheng Xiang Zhai

**<details><summary>Abstract**</summary>

Recent hybrid models combining Linear State Space Models (SSMs) with self-attention mechanisms have demonstrated impressive results across a range of sequence modeling tasks. However, current approaches apply attention modules statically and uniformly to all elements in the input sequences, leading to sub-optimal quality-efficiency trade-offs. To address this limitation, we introduce Sparse Modular Activation (SMA), a general mechanism enabling neural networks to sparsely and dynamically activate sub-modules for sequence elements in a differentiable manner. Through allowing each element to skip non-activated sub-modules, SMA reduces computation and memory consumption of neural networks at both training and inference stages. To validate the effectiveness of SMA on sequence modeling, we design a novel neural architecture, SeqBoat, which employs SMA to sparsely activate a Gated Attention Unit (GAU) based on the state representations learned from an SSM. By constraining the GAU to only conduct local attention on the activated inputs, SeqBoat can achieve linear inference complexity with theoretically infinite attention span, and provide substantially better quality-efficiency trade-off than the chunking-based models. With experiments on a wide range of tasks, including long sequence modeling, speech classification and language modeling, SeqBoat brings new state-of-the-art results among hybrid models with linear complexity, and reveals the amount of attention needed for each task through the learned sparse activation patterns. Our code is publicly available at https://github.com/renll/SeqBoat.</details>

### Sparse Parameterization for Epitomic Dataset Distillation

**Authors:** Xing Wei, Anjia Cao, Funing Yang, Zhiheng Ma

**<details><summary>Abstract**</summary>

The success of deep learning relies heavily on large and diverse datasets, but the storage, preprocessing, and training of such data present significant challenges. To address these challenges, dataset distillation techniques have been proposed to obtain smaller synthetic datasets that capture the essential information of the originals. In this paper, we introduce a Sparse Parameterization for Epitomic datasEt Distillation (SPEED) framework, which leverages the concept of dictionary learning and sparse coding to distill epitomes that represent pivotal information of the dataset. SPEED prioritizes proper parameterization of the synthetic dataset and introduces techniques to capture spatial redundancy within and between synthetic images. We propose Spatial-Agnostic Epitomic Tokens (SAETs) and Sparse Coding Matrices (SCMs) to efficiently represent and select significant features. Additionally, we build a Feature-Recurrent Network (FReeNet) to generate hierarchical features with high compression and storage efficiency. Experimental results demonstrate the superiority of SPEED in handling high-resolution datasets, achieving state-of-the-art performance on multiple benchmarks and downstream applications. Our framework is compatible with a variety of dataset matching approaches, generally enhancing their performance. This work highlights the importance of proper parameterization in epitomic dataset distillation and opens avenues for efficient representation learning. Source code is available at https://github.com/MIV-XJTU/SPEED.</details>

### Spatio-Angular Convolutions for Super-resolution in Diffusion MRI

**Authors:** Matthew Lyon, Paul Armitage, Mauricio A Álvarez

**<details><summary>Abstract**</summary>

Diffusion MRI (dMRI) is a widely used imaging modality, but requires long scanning times to acquire high resolution datasets. By leveraging the unique geometry present within this domain, we present a novel approach to dMRI angular super-resolution that extends upon the parametric continuous convolution (PCConv) framework. We introduce several additions to the operation including a Fourier feature mapping, 'global' co-ordinates, and domain specific context. Using this framework, we build a fully parametric continuous convolution network (PCCNN) and compare against existing models. We demonstrate the PCCNN performs competitively while using significantly fewer parameters. Moreover, we show that this formulation generalises well to clinically relevant downstream analyses such as fixel-based analysis, and neurite orientation dispersion and density imaging.</details>

### SpecTr: Fast Speculative Decoding via Optimal Transport

**Authors:** Ziteng Sun, Ananda Theertha Suresh, Jae Hun Ro, Ahmad Beirami, Himanshu Jain, Felix Yu

**<details><summary>Abstract**</summary>

Autoregressive sampling from large language models has led to state-of-the-art results in several natural language tasks.However, autoregressive sampling generates tokens one at a time making it slow, and even prohibitive in certain tasks. One way to speed up sampling is *speculative decoding*: use a small model to sample a *draft* (block or sequence of tokens), and then score all tokens in the draft by the large language model in parallel. A subset of the tokens in the draft are accepted (and the rest rejected) based on a statistical method to guarantee that the final output follows the distribution of the large model. In this work, we provide a principled understanding of speculative decoding through the lens of optimal transport (OT) with *membership cost*. This framework can be viewed as an extension of the well-known *maximal-coupling* problem. This new formulation enables us to generalize the speculative decoding method to allow for a set of $k$ candidates at the token-level, which leads to an improved optimal membership cost. We show that the optimal draft selection algorithm (transport plan) can be computed via linear programming, whose best-known runtime is exponential in $k$. We then propose a valid draft selection algorithm whose acceptance probability is $(1-1/e)$-optimal multiplicatively. Moreover, it can be computed in time almost linear with size of domain of a single token.Using this new draft selection algorithm, we develop a new autoregressive sampling algorithm called *SpecTr*, which provides speedup in decoding while ensuring that there is no quality degradation in the decoded output.We experimentally demonstrate that for state-of-the-art large language models, the proposed approach achieves a wall clock speedup of 2.13X, a further 1.37X speedup over speculative decoding on standard benchmarks.</details>

### Spectral Entry-wise Matrix Estimation for Low-Rank Reinforcement Learning

**Authors:** Stefan Stojanovic, Yassir Jedra, Alexandre Proutiere

**<details><summary>Abstract**</summary>

We study matrix estimation problems arising in reinforcement learning with low-rank structure. In low-rank bandits, the matrix to be recovered specifies the expected arm rewards, and for low-rank Markov Decision Processes (MDPs), it characterizes the transition kernel of the MDP. In both cases, each entry of the matrix carries important information, and we seek estimation methods with low entry-wise prediction error. Importantly, these methods further need to accommodate for inherent correlations in the available data (e.g. for MDPs, the data consists of system trajectories). We investigate the performance of  simple spectral-based matrix estimation approaches: we show that they efficiently recover the singular subspaces of the matrix and exhibit nearly-minimal entry-wise prediction error. These new results on low-rank matrix estimation make it possible to devise reinforcement learning algorithms that fully exploit the underlying low-rank structure. We provide two examples of such algorithms: a regret minimization algorithm for low-rank bandit problems, and a best policy identification algorithm for low-rank MDPs. Both algorithms yield state-of-the-art performance guarantees.</details>

### Spectral Invariant Learning for Dynamic Graphs under Distribution Shifts

**Authors:** Zeyang Zhang, Xin Wang, Ziwei Zhang, Zhou Qin, Weigao Wen, Hui Xue', Haoyang Li, Wenwu Zhu

**<details><summary>Abstract**</summary>

Dynamic graph neural networks (DyGNNs) currently struggle with handling distribution shifts that are inherent in dynamic graphs.Existing work on DyGNNs with out-of-distribution settings only focuses on the time domain, failing to handle cases involving distribution shifts in the spectral domain. In this paper, we discover that there exist cases with distribution shifts unobservable in the time domain while observable in the spectral domain, and propose to study distribution shifts on dynamic graphs in the spectral domain for the first time.However, this investigation poses two key challenges: i) it is non-trivial to capture different graph patterns that are driven by various frequency components entangled in the spectral domain; and ii) it remains unclear how to handle distribution shifts with the discovered spectral patterns. To address these challenges, we propose Spectral Invariant Learning for Dynamic Graphs under Distribution Shifts (SILD), which can handle distribution shifts on dynamic graphs by capturing and utilizing invariant and variant spectral patterns. Specifically, we first design a DyGNN with Fourier transform to obtain the ego-graph trajectory spectrums, allowing the mixed dynamic graph patterns to be transformed into separate frequency components. We then develop a disentangled spectrum mask to filter graph dynamics from various frequency components and discover the invariant and variant spectral patterns. Finally, we propose invariant spectral filtering, which encourages the model to rely on invariant patterns for generalization under distribution shifts. Experimental results on synthetic and real-world dynamic graph datasets demonstrate the superiority of our method for both node classification and link prediction tasks under distribution shifts.</details>

### Stochastic Approximation Algorithms for Systems of Interacting Particles

**Authors:** Mohammad Reza Karimi Jaghargh, Ya-Ping Hsieh, Andreas Krause

**<details><summary>Abstract**</summary>

Interacting particle systems have proven highly successful in various machinelearning tasks, including approximate Bayesian inference and neural network optimization. However, the analysis of thesesystems often relies on the simplifying assumption of the \emph{mean-field} limit, where particlenumbers approach infinity and infinitesimal step sizes are used. In practice, discrete time steps,finite particle numbers, and complex integration schemes are employed, creating a theoretical gapbetween continuous-time and discrete-time processes. In this paper, we present a novel frameworkthat establishes a precise connection between these discrete-time schemes and their correspondingmean-field limits in terms of convergence properties and asymptotic behavior. By adopting a dynamical system perspective, our framework seamlessly integrates various numerical schemes that are typically analyzed independently. For example, our framework provides a unified treatment of optimizing an infinite-width two-layer neural network and sampling via Stein Variational Gradient descent, which were previously studied in isolation.</details>

### Strategic Apple Tasting

**Authors:** Keegan Harris, Chara Podimata, Steven Wu

**<details><summary>Abstract**</summary>

Algorithmic decision-making in high-stakes domains often involves assigning decisions to agents with incentives to strategically modify their input to the algorithm. In addition to dealing with incentives, in many domains of interest (e.g. lending and hiring) the decision-maker only observes feedback regarding their policy for rounds in which they assign a positive decision to the agent; this type of feedback is often referred to as apple tasting (or one-sided) feedback. We formalize this setting as an online learning problem with apple-tasting feedback where a principal makes decisions about a sequence of $T$ agents, each of which is represented by a context that may be strategically modified. Our goal is to achieve sublinear strategic regret, which compares the performance of the principal to that of the best fixed policy in hindsight, if the agents were truthful when revealing their contexts. Our main result is a learning algorithm which incurs $\tilde{\mathcal{O}}(\sqrt{T})$ strategic regret when the sequence of agents is chosen stochastically. We also give an algorithm capable of handling adversarially-chosen agents, albeit at the cost of $\tilde{\mathcal{O}}(T^{(d+1)/(d+2)})$ strategic regret (where $d$ is the dimension of the context). Our algorithms can be easily adapted to the setting where the principal receives bandit feedback---this setting generalizes both the linear contextual bandit problem (by considering agents with incentives) and the strategic classification problem (by allowing for partial feedback).</details>

### Strategic Behavior in Two-sided Matching Markets with Prediction-enhanced Preference-formation

**Authors:** Stefania Ionescu, Yuhao Du, Kenneth Joseph, Ancsa Hannak

**<details><summary>Abstract**</summary>

Two-sided matching markets have long existed to pair agents in the absence of regulated exchanges.  A common example is school choice, where a matching mechanism uses student and school preferences to assign students to schools. In such settings, forming preferences is both difficult and critical. Prior work has suggested various prediction mechanisms that help agents make decisions about their preferences. Although often deployed together, these matching and prediction mechanisms are almost always analyzed separately. The present work shows that at the intersection of the two lies a previously unexplored type of strategic behavior: agents returning to the market (e.g., schools) can attack future predictions by interacting short-term non-optimally with their matches. Here, we first introduce this type of strategic behavior, which we call an adversarial interaction attack. Next, we construct a formal economic model that captures the feedback loop between prediction mechanisms designed to assist agents and the matching mechanism used to pair them. Finally, in a simplified setting, we prove that returning agents can benefit from using adversarial interaction attacks and gain progressively more as the trust in and accuracy of predictions increases. We also show that this attack increases inequality in the student population.</details>

### Structural Pruning for Diffusion Models

**Authors:** Gongfan Fang, Xinyin Ma, Xinchao Wang

**<details><summary>Abstract**</summary>

Generative modeling has recently undergone remarkable advancements, primarily propelled by the transformative implications of Diffusion Probabilistic Models (DPMs). The impressive capability of these models, however, often entails significant computational overhead during both training and inference. To tackle this challenge, we present Diff-Pruning, an efficient compression method tailored for learning lightweight diffusion models from pre-existing ones, without the need for extensive re-training. The essence of Diff-Pruning is encapsulated in a Taylor expansion over pruned timesteps, a process that disregards non-contributory diffusion steps and ensembles informative gradients to identify important weights. Our empirical assessment, undertaken across several datasets highlights two primary benefits of our proposed method: 1) Efficiency: it enables approximately a 50\% reduction in FLOPs at a mere 10% to 20% of the original training expenditure; 2) Consistency: the pruned diffusion models inherently preserve generative behavior congruent with their pre-trained models.</details>

### [Spotlight] Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data

**Authors:** Xin Zheng, Miao Zhang, Chunyang Chen, Quoc Viet Hung Nguyen, Xingquan Zhu, Shirui Pan

**<details><summary>Abstract**</summary>

Graph condensation, which reduces the size of a large-scale graph by synthesizing a small-scale condensed graph as its substitution, has immediate benefits for various graph learning tasks.However, existing graph condensation methods rely on the joint optimization of nodes and structures in the condensed graph, and overlook critical issues in effectiveness and generalization ability.In this paper, we advocate a new Structure-Free Graph Condensation paradigm, named SFGC, to distill a large-scale graph into a small-scale graph node set without explicit graph structures, i.e., graph-free data.Our idea is to implicitly encode topology structure information into the node attributes in the synthesized graph-free data, whose topology is reduced to an identity matrix.Specifically, SFGC contains two collaborative components: (1) a training trajectory meta-matching scheme for effectively synthesizing small-scale graph-free data;(2) a graph neural feature score metric for dynamically evaluating the quality of the condensed data. Through training trajectory meta-matching, SFGC aligns the long-term GNN learning behaviors between the large-scale graph and the condensed small-scale graph-free data, ensuring comprehensive and compact transfer of informative knowledge to the graph-free data.Afterward, the underlying condensed graph-free data would be dynamically evaluated with the graph neural feature score, which is a closed-form metric for ensuring the excellent expressiveness of the condensed graph-free data.Extensive experiments verify the superiority of SFGC across different condensation ratios.</details>

### Structured Neural Networks for Density Estimation and Causal Inference

**Authors:** Asic Chen, Ruian (Ian) Shi, Xiang Gao, Ricardo Baptista, Rahul Krishnan

**<details><summary>Abstract**</summary>

Injecting structure into neural networks enables learning functions that satisfy invariances with respect to subsets of inputs. For instance, when learning generative models using neural networks, it is advantageous to encode the conditional independence structure of observed variables, often in the form of Bayesian networks. We propose the Structured Neural Network (StrNN), which injects structure through masking pathways in a neural network. The masks are designed via a novel relationship we explore between neural network architectures and binary matrix factorization, to ensure that the desired independencies are respected. We devise and study practical algorithms for this otherwise NP-hard design problem based on novel objectives that control the model architecture. We demonstrate the utility of StrNN in three applications: (1) binary and Gaussian density estimation with StrNN, (2) real-valued density estimation with Structured Autoregressive Flows (StrAFs) and Structured Continuous Normalizing Flows (StrCNF), and (3) interventional and counterfactual analysis with StrAFs for causal inference. Our work opens up new avenues for learning neural networks that enable data-efficient generative modeling and the use of normalizing flows for causal effect estimation.</details>

### Structured Prediction with Stronger Consistency Guarantees

**Authors:** Anqi Mao, Mehryar Mohri, Yutao Zhong

**<details><summary>Abstract**</summary>

We present an extensive study of surrogate losses for structured prediction supported by *$H$-consistency bounds*. These are recently introduced guarantees that are more relevant to learning than Bayes-consistency, since they are not asymptotic and since they take into account the hypothesis set $H$ used. We first show that no non-trivial $H$-consistency bound can be derived for widely used surrogate structured prediction losses. We then define several new families of surrogate losses, including *structured comp-sum losses* and *structured constrained losses*, for which we prove $H$-consistency bounds and thus Bayes-consistency. These loss functions readily lead to new structured prediction algorithms with stronger theoretical guarantees, based on their minimization. We describe efficient algorithms for minimizing several of these surrogate losses, including a new *structured logistic loss*.</details>

### Structured Semidefinite Programming for Recovering Structured Preconditioners

**Authors:** Arun Jambulapati, Jerry Li, Christopher Musco, Kirankumar Shiragur, Aaron Sidford, Kevin Tian

**<details><summary>Abstract**</summary>

We develop a general framework for finding approximately-optimal preconditioners for solving linear systems. Leveraging this framework we obtain improved runtimes for fundamental preconditioning and linear system solving problems including:Diagonal preconditioning. We give an algorithm which, given positive definite $\mathbf{K} \in \mathbb{R}^{d \times d}$ with $\mathrm{nnz}(\mathbf{K})$ nonzero entries, computes an $\epsilon$-optimal diagonal preconditioner in time $\widetilde{O}(\mathrm{nnz}(\mathbf{K}) \cdot \mathrm{poly}(\kappa^\star,\epsilon^{-1}))$, where $\kappa^\star$ is the optimal condition number of the rescaled matrix.Structured linear systems. We give an algorithm which, given $\mathbf{M} \in \mathbb{R}^{d \times d}$ that is either the pseudoinverse of a graph Laplacian matrix or a constant spectral approximation of one, solves linear systems in $\mathbf{M}$ in $\widetilde{O}(d^2)$ time. Our diagonal preconditioning results improve state-of-the-art runtimes of $\Omega(d^{3.5})$ attained by general-purpose semidefinite programming, and our solvers improve state-of-the-art runtimes of $\Omega(d^{\omega})$ where $\omega > 2.3$ is the current matrix multiplication constant. We attain our results via new algorithms for a class of semidefinite programs (SDPs) we call matrix-dictionary approximation SDPs, which we leverage to solve an associated problem we call matrix-dictionary recovery.</details>

### [Spotlight] SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks

**Authors:** Bill Yuchen Lin, Yicheng Fu, Karina Yang, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Prithviraj (Raj) Ammanabrolu, Yejin Choi, Xiang Ren

**<details><summary>Abstract**</summary>

We introduce SwiftSage, a novel agent framework inspired by the dual-process theory of human cognition, designed to excel in action planning for complex interactive reasoning tasks. SwiftSage integrates the strengths of behavior cloning and prompting large language models (LLMs) to enhance task completion performance. The framework comprises two primary modules: the Swift module, representing fast and intuitive thinking, and the Sage module, emulating deliberate thought processes. The Swift module is a small encoder-decoder LM fine-tuned on the oracle agent's action trajectories, while the Sage module employs LLMs such as GPT-4 for subgoal planning and grounding. We develop a heuristic method to harmoniously integrate the two modules, resulting in a more efficient and robust problem-solving process. In 30 tasks from the ScienceWorld benchmark, SwiftSage significantly outperforms other methods such as SayCan, ReAct, and Reflexion, demonstrating its effectiveness in solving complex interactive tasks.</details>

### Symbol-LLM: Leverage Language Models for Symbolic System in Visual Human Activity Reasoning

**Authors:** Xiaoqian Wu, Yong-Lu Li, Jianhua Sun, Cewu Lu

**<details><summary>Abstract**</summary>

Human reasoning can be understood as a cooperation between the intuitive, associative "System-1'' and the deliberative, logical "System-2''. For existing System-1-like methods in visual activity understanding, it is crucial to integrate System-2 processing to improve explainability, generalization, and data efficiency. One possible path of activity reasoning is building a symbolic system composed of symbols and rules, where one rule connects multiple symbols, implying human knowledge and reasoning abilities.Previous methods have made progress, but are defective with limited symbols from handcraft and limited rules from visual-based annotations, failing to cover the complex patterns of activities and lacking compositional generalization. To overcome the defects, we propose a new symbolic system with two ideal important properties: broad-coverage symbols and rational rules. Collecting massive human knowledge via manual annotations is expensive to instantiate this symbolic system. Instead, we leverage the recent advancement of LLMs (Large Language Models) as an approximation of the two ideal properties, i.e., Symbols from Large Language Models (Symbol-LLM). Then, given an image, visual contents from the images are extracted andchecked as symbols and activity semantics are reasoned out based on rules via fuzzy logic calculation.Our method shows superiority in extensive activity understanding tasks. Code and data are available at https://mvig-rhos.com/symbol_llm.</details>

### SyncTREE: Fast Timing Analysis for Integrated Circuit Design through a Physics-informed Tree-based Graph Neural Network

**Authors:** Yuting Hu, Jiajie Li, Florian Klemme, Gi-Joon Nam, Tengfei Ma, Hussam Amrouch, Jinjun Xiong

**<details><summary>Abstract**</summary>

Nowadays integrated circuits (ICs) are underpinning all major information technology innovations including the current trends of artificial intelligence (AI). Modern IC designs often involve analyses of complex phenomena (such as timing, noise, and power etc.) for tens of billions of electronic components, like resistance (R), capacitance (C), transistors and gates, interconnected in various complex structures. Those analyses often need to strike a balance between accuracy and speed as those analyses need to be carried out many times throughout the entire IC design cycles. With the advancement of AI, researchers also start to explore news ways in leveraging AI to improve those analyses. This paper focuses on one of the most important analyses, timing analysis for interconnects. Since IC interconnects can be represented as an RC-tree, a specialized graph as tree, we design a novel tree-based graph neural network, SyncTREE, to speed up the timing analysis by incorporating both the structural and physical properties of electronic circuits. Our major innovations include (1) a two-pass message-passing (bottom-up and top-down) for graph embedding, (2) a tree contrastive loss to guide learning, and (3) a closed formular-based approach to conduct fast timing.  Our experiments show that, compared to conventional GNN models, SyncTREE achieves the best timing prediction in terms of both delays and slews, all in reference to the industry golden numerical analyses results on real IC design data.</details>

### TART: A plug-and-play Transformer module for task-agnostic reasoning

**Authors:** Kush Bhatia, Avanika Narayan, Christopher De Sa, Christopher Ré

**<details><summary>Abstract**</summary>

Large language models (LLMs) exhibit in-context learning abilities which enable the same model to perform several tasks without any task-specific training. In contrast, traditional adaptation approaches, such as fine-tuning, modify the underlying models for each specific task. In-context learning, however, consistently underperforms task-specific tuning approaches even when presented with the same examples. While most existing approaches (e.g., prompt engineering) focus on the LLM's learned representations to patch this performance gap, our experiments actually reveal that LLM representations contain sufficient information to make good predictions. As such, we focus on the LLM's reasoning abilities and demonstrate that this performance gap exists due to their inability to perform simple probabilistic reasoning tasks. This raises an intriguing question: Are LLMs actually capable of learning how to reason in a task-agnostic manner? We answer this in the affirmative and, as a proof of concept, propose TART which generically improves an LLM's reasoning abilities using a synthetically trained reasoning module. TART trains this Transformer-based reasoning module in a task-agnostic manner using only synthetic logistic regression tasks and composes it with an arbitrary real-world pre-trained model without any additional training. With a single inference module, TART improves performance across different model families (GPT-Neo, Pythia, Bloom), model sizes (100M - 6B), tasks (14 NLP classification tasks), and even across different modalities (audio and vision). On the RAFT Benchmark, TART improves GPT-Neo (125M)'s performance such that it outperforms Bloom (176B), and is within $4$% of GPT-3.</details>

### TD Convergence: An Optimization Perspective

**Authors:** Kavosh Asadi, Shoham Sabach, Yao Liu, Omer Gottesman, Rasool Fakoor

**<details><summary>Abstract**</summary>

We study the convergence behavior of the celebrated temporal-difference (TD) learning algorithm. By looking at the algorithm through the lens of optimization, we first argue that TD can be viewed as an iterative optimization algorithm where the function to be minimized changes per iteration. By carefully investigating the divergence displayed by TD on a classical counter example, we identify two forces that determine the convergent or divergent behavior of the algorithm. We next formalize our discovery in the linear TD setting with quadratic loss and prove that convergence of TD hinges on the interplay between these two forces. We extend this optimization perspective to prove convergence of TD in a much broader setting than just linear approximation and squared loss. Our results provide a theoretical explanation for the successful application of TD in reinforcement learning.</details>

### Temporal Conditioning Spiking Latent Variable Models of the Neural Response to Natural Visual Scenes

**Authors:** Gehua Ma, Runhao Jiang, Rui Yan, Huajin Tang

**<details><summary>Abstract**</summary>

Developing computational models of neural response is crucial for understanding sensory processing and neural computations. Current state-of-the-art neural network methods use temporal filters to handle temporal dependencies, resulting in an **unrealistic and inflexible processing paradigm**. Meanwhile, these methods target **trial-averaged firing rates** and fail to capture important features in spike trains. This work presents the temporal conditioning spiking latent variable models (***TeCoS-LVM***) to simulate the neural response to natural visual stimuli. We use spiking neurons to produce spike outputs that directly match the recorded trains. This approach helps to avoid losing information embedded in the original spike trains. We exclude the temporal dimension from the model parameter space and introduce a temporal conditioning operation to allow the model to adaptively explore and exploit temporal dependencies in stimuli sequences in a **natural paradigm**. We show that TeCoS-LVM models can produce more realistic spike activities and accurately fit spike statistics than powerful alternatives. Additionally, learned TeCoS-LVM models can generalize well to longer time scales. Overall, while remaining computationally tractable, our model effectively captures key features of neural coding systems. It thus provides a useful tool for building accurate predictive computational accounts for various sensory perception circuits.</details>

### Temporally Disentangled Representation Learning under Unknown Nonstationarity

**Authors:** Xiangchen Song, Weiran Yao, Yewen Fan, Xinshuai Dong, Guangyi Chen, Juan Carlos Niebles, Eric Xing, Kun Zhang

**<details><summary>Abstract**</summary>

In unsupervised causal representation learning for sequential data with time-delayed latent causal influences, strong identifiability results for the disentanglement of causally-related latent variables have been established in stationary settings by leveraging temporal structure.However, in nonstationary setting, existing work only partially addressed the problem by either utilizing observed auxiliary variables (e.g., class labels and/or domain indexes) as side information or assuming simplified latent causal dynamics. Both constrain the method to a limited range of scenarios.In this study, we further explored the Markov Assumption under time-delayed causally related process in nonstationary setting and showed that under mild conditions, the independent latent components can be recovered from their nonlinear mixture up to a permutation and a component-wise transformation, without the observation of auxiliary variables. We then introduce NCTRL, a principled estimation framework, to reconstruct time-delayed latent causal variables and identify their relations from measured sequential data only.Empirical evaluations demonstrated the reliable identification of time-delayed latent causal influences, with our methodology substantially outperforming existing baselines that fail to exploit the nonstationarity adequately and then, consequently, cannot distinguish distribution shifts.</details>

### Test-Time Distribution Normalization for Contrastively Learned Visual-language Models

**Authors:** Yifei Zhou, Juntao Ren, Fengyu Li, Ramin Zabih, Ser Nam Lim

**<details><summary>Abstract**</summary>

Advances in the field of visual-language contrastive learning have made it possible for many downstream applications to be carried out efficiently and accurately by simply taking the dot product between image and text representations.  One of the most representative approaches proposed recently known as CLIP has quickly garnered widespread adoption due to its effectiveness. CLIP is trained with an InfoNCE loss that takes into account both positive and negative samples to help learn a much more robust representation space. This paper however reveals that the common downstream practice of taking a dot product is only a zeroth-order approximation of the optimization goal, resulting in a loss of information during test-time. Intuitively, since the model has been optimized based on the InfoNCE loss, test-time procedures should ideally also be in alignment. The question lies in how one can retrieve any semblance of negative samples information during inference in a computationally efficient way. We propose Distribution Normalization (DN), where we approximate the mean representation of a batch of test samples and use such a mean to represent what would be analogous to negative samples in the InfoNCE loss. DN requires no retraining or fine-tuning and can be effortlessly applied during inference. Extensive experiments on a wide variety of downstream tasks exhibit a clear advantage of DN over the dot product on top of other existing test-time augmentation methods.</details>

### Text Promptable Surgical Instrument Segmentation with Vision-Language Models

**Authors:** Zijian Zhou, Oluwatosin Alabi, Meng Wei, Tom Vercauteren, Miaojing Shi

**<details><summary>Abstract**</summary>

In this paper, we propose a novel text promptable surgical instrument segmentation approach to overcome challenges associated with diversity and differentiation of surgical instruments in minimally invasive surgeries. We redefine the task as text promptable, thereby enabling a more nuanced comprehension of surgical instruments and adaptability to new instrument types. Inspired by recent advancements in vision-language models, we leverage pretrained image and text encoders as our model backbone and design a text promptable mask decoder consisting of attention- and convolution-based prompting schemes for surgical instrument segmentation prediction. Our model leverages multiple text prompts for each surgical instrument through a new mixture of prompts mechanism, resulting in enhanced segmentation performance. Additionally, we introduce a hard instrument area reinforcement module to improve image feature comprehension and segmentation precision. Extensive experiments on several surgical instrument segmentation datasets demonstrate our model's superior performance and promising generalization capability. To our knowledge, this is the first implementation of a promptable approach to surgical instrument segmentation, offering significant potential for practical application in the field of robotic-assisted surgery.</details>

### Textually Pretrained Speech Language Models

**Authors:** Michael Hassid, Tal Remez, Tu Anh Nguyen, Itai Gat, Alexis CONNEAU, Felix Kreuk, Jade Copet, Alexandre Defossez, Gabriel Synnaeve, Emmanuel Dupoux, Roy Schwartz, Yossi Adi

**<details><summary>Abstract**</summary>

Speech language models (SpeechLMs) process and generate acoustic data only, without textual supervision. In this work, we propose TWIST, a method for training SpeechLMs using a warm-start from a pretrained textual language models. We show using both automatic and human evaluations that TWIST outperforms a cold-start SpeechLM across the board. We empirically analyze the effect of different model design choices such as the speech tokenizer, the pretrained textual model, and the dataset size. We find that model and dataset scale both play an important role in constructing better-performing SpeechLMs. Based on our observations, we present the largest (to the best of our knowledge) SpeechLM both in terms of number of parameters and training data. We additionally introduce two spoken versions of the StoryCloze textual benchmark to further improve model evaluation and advance future research in the field. We make speech samples, code and models publicly available.</details>

### The Curious Price of Distributional Robustness in Reinforcement Learning with a Generative Model

**Authors:** Laixi Shi, Gen Li, Yuting Wei, Yuxin Chen, Matthieu Geist, Yuejie Chi

**<details><summary>Abstract**</summary>

This paper investigates model robustness in reinforcement learning (RL) via the framework of distributionally robust Markov decision processes (RMDPs). Despite recent efforts, the sample complexity of RMDPs is much less understood regardless of the uncertainty set in use; in particular, there exist large gaps between existing upper and lower bounds, and it is unclear if distributional robustness bears any statistical implications when benchmarked against standard RL. In this paper, assuming access to a generative model, we derive the sample complexity of RMDPs---when the uncertainty set is measured via either total variation or $\chi^2$ divergence over the full range of uncertainty levels---using a model-based algorithm called distributionally robust value iteration, and develop  minimax lower bounds to benchmark its tightness. Our results not only strengthen the prior art in both directions of upper and lower bounds, but also deliver surprising messages that learning RMDPs is not necessarily easier or more difficult than standard MDPs. In the case of total variation, we establish the minimax-optimal sample complexity of RMDPs which is always smaller than that of standard MDPs. In the case of $\chi^2$ divergence, we establish the sample complexity of RMDPs that is tight up to polynomial factors of the effective horizon, and grows linearly with respect to the uncertainty level when it approaches infinity.</details>

### The Distortion of Binomial Voting Defies Expectation

**Authors:** Yannai A. Gonczarowski, Gregory Kehne, Ariel Procaccia, Ben Schiffer, Shirley Zhang

**<details><summary>Abstract**</summary>

In computational social choice, the distortion of a voting rule quantifies the degree to which the rule overcomes limited preference information to select a socially desirable outcome. This concept has been investigated extensively, but only through a worst-case lens. Instead, we study the expected distortion of voting rules with respect to an underlying distribution over voter utilities. Our main contribution is the design and analysis of a novel and intuitive rule, binomial voting, which provides strong distribution-independent guarantees for both expected distortion and expected welfare.</details>

### The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter

**Authors:** AJAY JAISWAL, Shiwei Liu, Tianlong Chen, Zhangyang "Atlas" Wang

**<details><summary>Abstract**</summary>

Large pre-trained transformers are $\textit{show-stealer}$ in modern-day deep learning, and it becomes crucial to comprehend the parsimonious patterns that exist within them as they grow in scale. With exploding parameter counts, Lottery Ticket Hypothesis (LTH) and its variants, have lost their pragmatism in sparsifying them due to high computation and memory bottleneck of repetitive $\textit{train-prune-retrain}$ routine of iterative magnitude pruning (IMP) which worsens with increasing model size. In this paper, we comprehensively study $\textit{induced sparse patterns}$ across multiple large pre-trained vision and language transformers. We propose the existence of -- $\textbf{essential sparsity}$ defined with a $\textbf{sharp dropping point}$ beyond which the performance declines much faster w.r.t the rise of sparsity level, when we directly remove weights with the smallest magnitudes in $\textbf{one-shot}$. We also present an intriguing emerging phenomenon of $\textbf{abrupt sparsification}$ during the pre-training of BERT, i.e., BERT suddenly becomes heavily sparse in pre-training after certain iterations. Moreover, our observations also indicate a $\textbf{counter-intuitive}$ finding that BERT trained with a larger amount of pre-training data tends to have a better ability to condense knowledge in comparatively relatively fewer parameters. Lastly, we investigate the effect of the pre-training loss on essential sparsity and discover that self-supervised learning (SSL) objectives trigger stronger emergent sparsification properties than supervised learning (SL). All our codes will be publicly available.</details>

### The Memory-Perturbation Equation: Understanding Model's Sensitivity to Data

**Authors:** Peter Nickl, Lu Xu, Dharmesh Tailor, Thomas Möllenhoff, Mohammad Emtiyaz Khan

**<details><summary>Abstract**</summary>

Understanding model’s sensitivity to its training data is crucial but can also be challenging and costly, especially during training. To simplify such issues, we present the Memory-Perturbation Equation (MPE) which relates model's sensitivity to perturbation in its training data. Derived using Bayesian principles, the MPE unifies existing sensitivity measures, generalizes them to a wide-variety of models and algorithms, and unravels useful properties regarding sensitivities. Our empirical results show that sensitivity estimates obtained during training can be used to faithfully predict generalization on unseen test data. The proposed equation is expected to be useful for future research on robust and adaptive learning.</details>

### The Rank-Reduced Kalman Filter: Approximate Dynamical-Low-Rank Filtering In High Dimensions

**Authors:** Jonathan Schmidt, Philipp Hennig, Jörg Nick, Filip Tronarp

**<details><summary>Abstract**</summary>

Inference and simulation in the context of high-dimensional dynamical systems remain computationally challenging problems.Some form of dimensionality reduction is required to make the problem tractable in general.In this paper, we propose a novel approximate Gaussian filtering and smoothing methodwhich propagates low-rank approximations of the covariance matrices.This is accomplished by projecting the Lyapunov equations associated with the prediction step to a manifold of low-rank matrices,which are then solved by a recently developed, numerically stable, dynamical low-rank integrator.Meanwhile, the update steps are made tractable by noting that the covariance update only transforms the column space of the covariance matrix, which is low-rank by construction.The algorithm differentiates itself from existing ensemble-based approaches in thatthe low-rank approximations of the covariance matrices are deterministic, rather than stochastic.Crucially, this enables the method to reproduce the exact Kalman filter as the low-rank dimension approaches the true dimensionality of the problem.Our method reduces computational complexity from cubic (for the Kalman filter) to quadratic in the state-space size in the worst-case, and can achieve linear complexity if the state-space model satisfies certain criteria.Through a set of experiments in classical data-assimilation and spatio-temporal regression, we show that the proposed method consistently outperforms the ensemble-based methods in terms of error in the mean and covariance with respect to the exact Kalman filter. This comes at no additional cost in terms of asymptotic computational complexity.</details>

### [Spotlight] The Rashomon Importance Distribution: Getting RID of Unstable, Single Model-based Variable Importance

**Authors:** Jon Donnelly, Srikar Katta, Cynthia Rudin, Edward Browne

**<details><summary>Abstract**</summary>

Quantifying variable importance is essential for answering high-stakes questions in fields like genetics, public policy, and medicine. Current methods generally calculate variable importance for a given model trained on a given dataset. However, for a given dataset, there may be many models that explain the target outcome equally well; without accounting for all possible explanations, different researchers may arrive at many conflicting yet equally valid conclusions given the same data. Additionally, even when accounting for all possible explanations for a given dataset, these insights may not generalize because not all good explanations are stable across reasonable data perturbations. We propose a new variable importance framework that quantifies the importance of a variable across the set of all good models and is stable across the data distribution. Our framework is extremely flexible and can be integrated with most existing model classes and global variable importance metrics. We demonstrate through experiments that our framework recovers variable importance rankings for complex simulation setups where other methods fail. Further, we show that our framework accurately estimates the _true importance_ of a variable for the underlying data distribution. We provide theoretical guarantees on the consistency and finite sample error rates for our estimator. Finally, we demonstrate its utility with a real-world case study exploring which genes are important for predicting HIV load in persons with HIV, highlighting an important gene that has not previously been studied in connection with HIV.</details>

### The Utility of “Even if” Semifactual Explanation to Optimise Positive Outcomes

**Authors:** Eoin Kenny, Weipeng Huang

**<details><summary>Abstract**</summary>

When users receive either a positive or negative outcome from an automated system, Explainable AI (XAI) has almost exclusively focused on how to mutate negative outcomes into positive ones by crossing a decision boundary using counterfactuals (e.g., *"If you earn 2k more, we will accept your loan application"*). Here, we instead focus on positive outcomes, and take the novel step of using XAI to optimise them (e.g., *"Even if you wish to half your down-payment, we will still accept your loan application"*). Explanations such as these that employ "even if..." reasoning, and do not cross a decision boundary, are known as semifactuals. To instantiate semifactuals in this context, we introduce the concept of *Gain* (i.e., how much a user stands to benefit from the explanation), and consider the first causal formalisation of semifactuals. Tests on benchmark datasets show our algorithms are better at maximising gain compared to prior work, and that causality is important in the process. Most importantly however, a user study supports our main hypothesis by showing people find semifactual explanations more useful than counterfactuals when they receive the positive outcome of a loan acceptance.</details>

### The emergence of clusters in self-attention dynamics

**Authors:** Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, Philippe Rigollet

**<details><summary>Abstract**</summary>

Viewing Transformers as interacting particle systems, we describe the geometry of learned representations when the weights are not time-dependent. We show that particles, representing tokens, tend to cluster toward particular limiting objects as time tends to infinity. Using techniques from dynamical systems and partial differential equations, we show that type of limiting object that emerges depends on the spectrum of the value matrix. Additionally, in the one-dimensional case we prove that the self-attention matrix converges to a low-rank Boolean matrix. The combination of these results mathematically confirms the empirical observation made by Vaswani et al. [ VSP`17 ] that leaders appear in a sequence of tokens when processed by Transformers.</details>

### The geometry of hidden representations of large transformer models

**Authors:** Lucrezia Valeriani, Diego Doimo, Francesca Cuturello, Alessandro Laio, Alessio Ansuini, Alberto Cazzaniga

**<details><summary>Abstract**</summary>

Large transformers are powerful architectures used for self-supervised data analysis across various data types, including protein sequences, images, and text. In these models, the semantic structure of the dataset emerges from a sequence of transformations between one representation and the next. We characterize the geometric and statistical properties of these representations and how they change as we move through the layers.By analyzing the intrinsic dimension (ID) and neighbor composition, we find that the representations evolve similarly in transformers trained on protein language taskand image reconstruction tasks. In the first layers, the data manifold expands, becoming high-dimensional, and then contracts significantly in the intermediate layers. In the last part of the model, the ID remains approximately constant or forms a second shallow peak. We show that the semantic information of the dataset is better expressed at the end of the first peak, and this phenomenon can be observed across many models trained on diverse datasets.Based on our findings, we point out an explicit strategy to identify, without supervision, the layers that maximize semantic content: representations at intermediate layers corresponding to a relative minimum of the ID profile are more suitable for downstream learning tasks.</details>

### The s-value: evaluating stability with respect to distributional shifts

**Authors:** Suyash Gupta, Dominik Rothenhäusler

**<details><summary>Abstract**</summary>

Common statistical measures of uncertainty such as $p$-values and confidence intervals quantify the uncertainty due to sampling, that is, the uncertainty due to not observing the full population. However, sampling is not the only source of uncertainty. In practice, distributions change between locations and across time. This makes it difficult to gather knowledge that transfers across data sets. We propose a measure of instability that quantifies the distributional instability of a statistical parameter with respect to Kullback-Leibler divergence, that is, the sensitivity of the parameter under general distributional perturbations within a Kullback-Leibler divergence ball. In addition, we quantify the instability of parameters with respect to directional or variable-specific shifts. Measuring instability with respect to directional shifts can be used to detect under which kind of distribution shifts a statistical conclusion might be reversed. We discuss how such knowledge can inform data collection for transfer learning of statistical parameters under shifted distributions. We evaluate the performance of the proposed measure on real data and show that it can elucidate the distributional instability of a parameter with respect to certain shifts and can be used to improve estimation accuracy under shifted distributions.</details>

### This Looks Like Those: Illuminating Prototypical Concepts Using Multiple Visualizations

**Authors:** Chiyu Ma, Brandon Zhao, Chaofan Chen, Cynthia Rudin

**<details><summary>Abstract**</summary>

We present ProtoConcepts, a method for interpretable image classification combining deep learning and case-based reasoning using prototypical parts. Existing work in prototype-based image classification uses a "this looks like that'' reasoning process, which dissects a test image by finding prototypical parts and combining evidence from these prototypes to make a final classification. However, all of the existing prototypical part-based image classifiers provide only one-to-one comparisons, where a single training image patch serves as a prototype to compare with a part of our test image. With these single-image comparisons, it can often be difficult to identify the underlying concept being compared (e.g., "is it comparing the color or the shape?''). Our proposed method modifies the architecture of prototype-based networks to instead learn prototypical concepts which are visualized using multiple image patches. Having multiple visualizations of the same prototype allows us to more easily identify the concept captured by that prototype (e.g., "the test image and the related training patches are all the same shade of blue''), and allows our model to create richer, more interpretable visual explanations. Our experiments show that our ``this looks like those'' reasoning process can be applied as a modification to a wide range of existing prototypical image classification networks while achieving comparable accuracy on benchmark datasets.</details>

### [Spotlight] Thought Cloning: Learning to Think while Acting by Imitating Human Thinking

**Authors:** Shengran Hu, Jeff Clune

**<details><summary>Abstract**</summary>

Language is often considered a key aspect of human thinking, providing us with exceptional abilities to generalize, explore, plan, replan, and adapt to new situations. However, Reinforcement Learning (RL) agents are far from human-level performance in any of these abilities. We hypothesize one reason for such cognitive deficiencies is that they lack the benefits of thinking in language and that we can improve AI agents by training them to $\textit{think like humans do}$. We introduce a novel Imitation Learning framework, Thought Cloning, where the idea is to not just clone the behaviors of human demonstrators, $\textit{but also the thoughts humans have as they perform these behaviors}$. While we expect Thought Cloning to truly shine at scale on internet-sized datasets (e.g. online videos with transcripts), here we conduct experiments in a domain where the thinking and action data are synthetically generated. Results reveal that Thought Cloning learns much faster than Behavioral Cloning and its performance advantage grows the further out of distribution test tasks are, highlighting its ability to better handle novel situations. Thought Cloning also provides important benefits for AI Safety and Interpretability, and makes it easier to debug and improve AI. Because we can observe the agent’s thoughts, we can (1) more easily diagnose why things are going wrong, making it easier to fix the problem, (2) steer the agent by correcting its thinking, or (3) prevent it from doing unsafe things it plans to do. Overall, by training agents $\textit{how to think}$ as well as behave, Thought Cloning creates safer, more powerful agents.</details>

### Tight Bounds for Volumetric Spanners and Applications

**Authors:** Aditya Bhaskara, Sepideh Mahabadi, Ali Vakilian

**<details><summary>Abstract**</summary>

Given a set of points of interest, a volumetric spanner is a subset of the points using which all the points can be expressed using "small" coefficients (measured in an appropriate norm). Formally, given a set of vectors $X = [v_1, v_2, \dots, v_n]$, the goal is to find $T \subseteq [n]$ such that every $v \in X$ can be expressed as $\sum_{i\in T} \alpha_i v_i$, with $\Vert \alpha \Vert$ being small.  This notion, which has also been referred to as a well-conditioned basis, has found several applications, including bandit linear optimization, determinant maximization, and matrix low rank approximation. In this paper, we give almost optimal bounds on the size of volumetric spanners for all $\ell_p$ norms, and show that they can be constructed using a simple local search procedure. We then show the applications of our result to other tasks and in particular the problem of finding coresets for the Minimum Volume Enclosing Ellipsoid (MVEE) problem.</details>

### Time Series Kernels based on Nonlinear Vector AutoRegressive Delay Embeddings

**Authors:** Giovanni De Felice, John Goulermas, Vladimir Gusev

**<details><summary>Abstract**</summary>

Kernel design is a pivotal but challenging aspect of time series analysis, especially in the context of small datasets. In recent years, Reservoir Computing (RC) has emerged as a powerful tool to compare time series based on the underlying dynamics of the generating process rather than the observed data. However, the performance of RC highly depends on the hyperparameter setting, which is hard to interpret and costly to optimize because of the recurrent nature of RC. Here, we present a new kernel for time series based on the recently established equivalence between reservoir dynamics and Nonlinear Vector AutoRegressive (NVAR) processes. The kernel is non-recurrent and depends on a small set of meaningful hyperparameters, for which we suggest an effective heuristic. We demonstrate excellent performance on a wide range of real-world classification tasks, both in terms of accuracy and speed. This further advances the understanding of RC representation learning models and extends the typical use of the NVAR framework to kernel design and representation of real-world time series data.</details>

### Time Series as Images: Vision Transformer for Irregularly Sampled Time Series

**Authors:** Zekun Li, Shiyang Li, Xifeng Yan

**<details><summary>Abstract**</summary>

Irregularly sampled time series are increasingly prevalent, particularly in medical domains. While various specialized methods have been developed to handle these irregularities, effectively modeling their complex dynamics and pronounced sparsity remains a challenge. This paper introduces a novel perspective by converting irregularly sampled time series into line graph images, then utilizing powerful pre-trained vision transformers for time series classification in the same way as image classification. This method not only largely simplifies specialized algorithm designs but also presents the potential to serve as a universal framework for time series modeling. Remarkably, despite its simplicity, our approach outperforms state-of-the-art specialized algorithms on several popular healthcare and human activity datasets. Especially in the rigorous leave-sensors-out setting where a portion of variables is omitted during testing, our method exhibits strong robustness against varying degrees of missing observations, achieving an impressive improvement of 42.8% in absolute F1 score points over leading specialized baselines even with half the variables masked. Code and data are available at https://github.com/Leezekun/ViTST.</details>

### Tools for Verifying Neural Models' Training Data

**Authors:** Dami Choi, Yonadav Shavit, David Duvenaud

**<details><summary>Abstract**</summary>

It is important that consumers and regulators can verify the provenance of large neural models to evaluate their capabilities and risks. We introduce the concept of a "Proof-of-Training-Data": any protocol that allows a model trainer to convince a Verifier of the training data that produced a set of model weights. Such protocols could verify the amount and kind of data and compute used to train the model, including whether it was trained on specific harmful or beneficial data sources. We explore efficient verification strategies for Proof-of-Training-Data that are compatible with most current large-model training procedures. These include a method for the model-trainer to verifiably pre-commit to a random seed used in training, and a method that exploits models' tendency to temporarily overfit to training data in order to detect whether a given data-point was included in training. We show experimentally that our verification procedures can catch a wide variety of attacks, including all known attacks from the Proof-of-Learning literature.</details>

### Top-Ambiguity Samples Matter: Understanding Why Deep Ensemble Works in Selective Classification

**Authors:** Qiang Ding, Yixuan Cao, Ping Luo

**<details><summary>Abstract**</summary>

Selective classification allows a machine learning model to reject some hard inputs and thus improve the reliability of its predictions. In this area, the ensemble method is powerful in practice, but there has been no solid analysis on why the ensemble method works. Inspired by an interesting empirical result that the improvement of the ensemble largely comes from top-ambiguity samples where its member models diverge, we prove that, based on some assumptions, the ensemble has a lower selective risk than the member model for any coverage within a range. The proof is nontrivial since the selective risk is a non-convex function of the model prediction. The assumptions and the theoretical results are supported by systematic experiments on both computer vision and natural language processing tasks.</details>

### [Spotlight] Topological Parallax: A Geometric Specification for Deep Perception Models

**Authors:** Abraham Smith, Michael Catanzaro, Gabrielle Angeloro, Nirav Patel, Paul Bendich

**<details><summary>Abstract**</summary>

For safety and robustness of AI systems, we introduce _topological parallax_ as a theoretical and computational tool that compares a trained model to a reference dataset to determine whether they have similar multiscale geometric structure. Our proofs and examples show that this geometric similarity between dataset and model is essential to trustworthy interpolation and perturbation, and we conjecture that this new concept will add value to the current debate regarding the unclear relationship between "overfitting"' and "generalization'' in applications of deep-learning. In typical deep-learning applications, an explicit geometric description of the model isimpossible, but parallax can estimate topological features (components, cycles, voids, etc.)in the model by examining the effect on the Rips complex of geodesic distortions using the reference dataset.Thus, parallax indicates whether the model shares similar multiscale geometric features with the dataset.Parallax presents theoretically via topological data analysis [TDA] as a bi-filtered persistence module,and the key properties of this module are stable under perturbation of the reference dataset.</details>

### Toward Understanding Generative Data Augmentation

**Authors:** Chenyu Zheng, Guoqiang Wu, Chongxuan LI

**<details><summary>Abstract**</summary>

Generative data augmentation, which scales datasets by obtaining fake labeled examples from a trained conditional generative model, boosts classification performance in various learning tasks including (semi-)supervised learning, few-shot learning, and adversarially robust learning. However, little work has theoretically investigated the effect of generative data augmentation. To fill this gap, we establish a general stability bound in this not independently and identicallydistributed (non-i.i.d.) setting, where the learned distribution is dependent on the original train set and generally not the same as the true distribution. Our theoretical result includes the divergence between the learned distribution and the true distribution. It shows that generative data augmentation can enjoy a faster learning rate when the order of divergence term is $o(\max\left( \log(m)\beta_m, 1 / \sqrt{m})\right)$, where $m$ is the train set size and $\beta_m$ is the corresponding stability constant. We further specify the learning setup to the Gaussian mixture model and generative adversarial nets. We prove that in both cases, though generative data augmentation does not enjoy a faster learning rate, it can improve the learning guarantees at a constant level when the train set is small, which is significant when the awful overfitting occurs. Simulation results on the Gaussian mixture model and empirical results on generative adversarial nets support our theoretical conclusions.</details>

### Towards Consistent Video Editing with Text-to-Image Diffusion Models

**Authors:** Zicheng Zhang, Bonan Li, Xuecheng Nie, Congying Han, Tiande Guo, Luoqi Liu

**<details><summary>Abstract**</summary>

Existing works have advanced Text-to-Image (TTI) diffusion models for video editing in a one-shot learning manner. Despite their low requirements of data and computation, these methods might produce results of unsatisfied consistency with text prompt as well as temporal sequence, limiting their applications in the real world. In this paper, we propose to address the above issues with a novel EI$^2$ model towards Enhancing vIdeo Editing consIstency of TTI-based frameworks. Specifically, we analyze and find that the inconsistent problem is caused by newly added modules into TTI models for learning temporal information. These modules lead to covariate shift in the feature space, which harms the editing capability. Thus, we design EI$^2$ to tackle the above drawbacks with two classical modules: Shift-restricted Temporal Attention Module (STAM) and Fine-coarse Frame Attention Module (FFAM). First, through theoretical analysis, we demonstrate that covariate shift is highly related to Layer Normalization, thus STAM employs a Instance Centering layer replacing it to preserve the distribution of temporal features.  In addition, STAM employs an attention layer with normalized mapping to transform temporal features while constraining the variance shift.  As the second part, we incorporate STAM with a novel FFAM, which efficiently leverages fine-coarse spatial information of overall frames to further enhance temporal consistency. Extensive experiments demonstrate the superiority of the proposed EI$^2$ model.</details>

### Towards Distribution-Agnostic Generalized Category Discovery

**Authors:** Jianhong Bai, Zuozhu Liu, Hualiang Wang, Ruizhe Chen, Lianrui Mu, Xiaomeng Li, Joey Tianyi Zhou, YANG FENG, Jian Wu, Haoji Hu

**<details><summary>Abstract**</summary>

Data imbalance and open-ended distribution are two intrinsic characteristics of the real visual world. Though encouraging progress has been made in tackling each challenge separately, few works dedicated to combining them towards real-world scenarios. While several previous works have focused on classifying close-set samples and detecting open-set samples during testing, it's still essential to be able to classify unknown subjects as human beings. In this paper, we formally define a more realistic task as distribution-agnostic generalized category discovery (DA-GCD): generating fine-grained predictions for both close- and open-set classes in a long-tailed open-world setting. To tackle the challenging problem, we propose a Self-**Ba**lanced **Co**-Advice co**n**trastive framework (BaCon), which consists of a contrastive-learning branch and a pseudo-labeling branch, working collaboratively to provide interactive supervision to resolve the DA-GCD task. In particular, the contrastive-learning branch provides reliable distribution estimation to regularize the predictions of the pseudo-labeling branch, which in turn guides contrastive learning through self-balanced knowledge transfer and a proposed novel contrastive loss. We compare BaCon with state-of-the-art methods from two closely related fields: imbalanced semi-supervised learning and generalized category discovery. The effectiveness of BaCon is demonstrated with superior performance over all baselines and comprehensive analysis across various datasets. Our code is publicly available.</details>

### Towards Efficient and Accurate Winograd Convolution via Full Quantization

**Authors:** Tianqi Chen, Weixiang Xu, Weihan Chen, Peisong Wang, Jian Cheng

**<details><summary>Abstract**</summary>

The Winograd algorithm is an efficient convolution implementation, which performs calculations in the transformed domain. To further improve the computation efficiency, recent works propose to combine it with model quantization. Although Post-Training Quantization has the advantage of low computational cost and has been successfully applied in many other scenarios, a severe accuracy drop exists when utilizing it in Winograd convolution. Besides, despite the Winograd algorithm consisting of four stages, most existing methods only quantize the element-wise multiplication stage, leaving a considerable portion of calculations in full precision.In this paper, observing the inconsistency among different transformation procedures, we present PTQ-Aware Winograd (PAW) to optimize them collaboratively under a unified objective function. Moreover, we explore the full quantization of faster Winograd (tile size $\geq4$) for the first time. We further propose a hardware-friendly method called Factorized Scale Quantization (FSQ), which can effectively balance the significant range differences in the Winograd domain. Experiments demonstrate the effectiveness of our method, e.g., with 8-bit quantization and a tile size of 6, our method outperforms the previous Winograd PTQ method by 8.27\% and 5.38\% in terms of the top-1 accuracy on ResNet-18 and ResNet-34, respectively.</details>

### Towards Evaluating Transfer-based Attacks Systematically, Practically, and Fairly

**Authors:** Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**<details><summary>Abstract**</summary>

The adversarial vulnerability of deep neural networks (DNNs) has drawn great attention due to the security risk of applying these models in real-world applications. Based on transferability of adversarial examples, an increasing number of transfer-based methods have been developed to fool black-box DNN models whose architecture and parameters are inaccessible. Although tremendous effort has been exerted, there still lacks a standardized benchmark that could be taken advantage of to compare these methods systematically, fairly, and practically. Our investigation shows that the evaluation of some methods needs to be more reasonable and more thorough to verify their effectiveness, to avoid, for example, unfair comparison and insufficient consideration of possible substitute/victim models. Therefore, we establish a transfer-based attack benchmark (TA-Bench) which implements 30+ methods. In this paper, we evaluate and compare them comprehensively on 10 popular substitute/victim models on ImageNet. New insights about the effectiveness of these methods are gained and guidelines for future evaluations are provided.</details>

### Towards Higher Ranks via Adversarial Weight Pruning

**Authors:** Yuchuan Tian, Hanting Chen, Tianyu Guo, Chao Xu, Yunhe Wang

**<details><summary>Abstract**</summary>

Convolutional Neural Networks (CNNs) are hard to deploy on edge devices due to its high computation and storage complexities. As a common practice for model compression, network pruning consists of two major categories: unstructured and structured pruning, where unstructured pruning constantly performs better. However, unstructured pruning presents a structured pattern at high pruning rates, which limits its performance. To this end, we propose a Rank-based PruninG (RPG) method to maintain the ranks of sparse weights in an adversarial manner. In each step, we minimize the low-rank approximation error for the weight matrices using singular value decomposition, and maximize their distance by pushing the weight matrices away from its low rank approximation. This rank-based optimization objective guides sparse weights towards a high-rank topology. The proposed method is conducted in a gradual pruning fashion to stabilize the change of rank during training. Experimental results on various datasets and different tasks demonstrate the effectiveness of our algorithm in high sparsity. The proposed RPG outperforms the state-of-the-art performance by 1.13\% top-1 accuracy on ImageNet in ResNet-50 with 98\% sparsity. The codes are available at https://github.com/huawei-noah/Efficient-Computing/tree/master/Pruning/RPG and https://gitee.com/mindspore/models/tree/master/research/cv/RPG.</details>

### [Oral] Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective

**Authors:** Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, Liwei Wang

**Oral Presentation:** We, Dec 13, 14:15 -- Oral 4C

**<details><summary>Abstract**</summary>

Recent studies have discovered that Chain-of-Thought prompting (CoT) can dramatically improve the performance of Large Language Models (LLMs), particularly when dealing with complex tasks involving mathematics or reasoning. Despite the enormous empirical success, the underlying mechanisms behind CoT and how it unlocks the potential of LLMs remain elusive. In this paper, we take a first step towards theoretically answering these questions. Specifically, we examine the \emph{expressivity} of LLMs with CoT in solving fundamental mathematical and decision-making problems. By using circuit complexity theory, we first give impossibility results showing that bounded-depth Transformers are unable to directly produce correct answers for basic arithmetic/equation tasks unless the model size grows \emph{super-polynomially} with respect to the input length. In contrast, we then prove by construction that autoregressive Transformers of \emph{constant size} suffice to solve both tasks by generating CoT derivations using a commonly used math language format. Moreover, we show LLMs with CoT can handle a general class of decision-making problems known as Dynamic Programming, thus justifying its power in tackling complex real-world tasks. Finally, an extensive set of experiments show that, while Transformers always fail to directly predict the answers, they can consistently learn to generate correct solutions step-by-step given sufficient CoT demonstrations.</details>

### Towards Robust and Expressive Whole-body Human Pose and Shape Estimation

**Authors:** Hui En Pang, Zhongang Cai, Lei Yang, Qingyi Tao, Zhonghua Wu, Tianwei Zhang, Ziwei Liu

**<details><summary>Abstract**</summary>

Whole-body pose and shape estimation aims to jointly predict different behaviors (e.g., pose, hand gesture, facial expression) of the entire human body from a monocular image. Existing methods often exhibit suboptimal performance due to the complexity of in-the-wild scenarios. We argue that the prediction accuracy of these models is significantly affected by the quality of the _bounding box_, e.g., scale, alignment. The natural discrepancy between the ideal bounding box annotations and model detection results is particularly detrimental to the performance of whole-body pose and shape estimation.In this paper, we propose a novel framework to enhance the robustness of whole-body pose and shape estimation. Our framework incorporates three new modules to address the above challenges from three perspectives: (1) a **Localization Module** enhances the model's awareness of the subject's location and semantics within the image space; (2) a **Contrastive Feature Extraction Module** encourages the model to be invariant to robust augmentations by incorporating a contrastive loss and positive samples; (3) a **Pixel Alignment Module** ensures the reprojected mesh from the predicted camera and body model parameters are more accurate and pixel-aligned. We perform comprehensive experiments to demonstrate the effectiveness of our proposed framework on body, hands, face and whole-body benchmarks.</details>

### Towards Self-Interpretable Graph-Level Anomaly Detection

**Authors:** Yixin Liu, Kaize Ding, Qinghua Lu, Fuyi Li, Leo Yu Zhang, Shirui Pan

**<details><summary>Abstract**</summary>

Graph-level anomaly detection (GLAD) aims to identify graphs that exhibit notable dissimilarity compared to the majority in a collection. However, current works primarily focus on evaluating graph-level abnormality while failing to provide meaningful explanations for the predictions, which largely limits their reliability and application scope. In this paper, we investigate a new challenging problem, explainable GLAD, where the learning objective is to predict the abnormality of each graph sample with corresponding explanations, i.e., the vital subgraph that leads to the predictions. To address this challenging problem, we propose a Self-Interpretable Graph aNomaly dETection model (SIGNET for short) that detects anomalous graphs as well as generates informative explanations simultaneously. Specifically, we first introduce the multi-view subgraph information bottleneck (MSIB) framework, serving as the design basis of our self-interpretable GLAD approach. This way SIGNET is able to not only measure the abnormality of each graph based on cross-view mutual information but also provide informative graph rationales by extracting bottleneck subgraphs from the input graph and its dual hypergraph in a self-supervised way. Extensive experiments on 16 datasets demonstrate the anomaly detection capability and self-interpretability of SIGNET.</details>

### Towards Semi-Structured Automatic ICD Coding via Tree-based Contrastive Learning

**Authors:** Chang Lu, Chandan Reddy, Ping Wang, Yue Ning

**<details><summary>Abstract**</summary>

Automatic coding of International Classification of Diseases (ICD) is a multi-label text categorization task that involves extracting disease or procedure codes from clinical notes. Despite the application of state-of-the-art natural language processing (NLP) techniques, there are still challenges including limited availability of data due to privacy constraints and the high variability of clinical notes caused by different writing habits of medical professionals and various pathological features of patients. In this work, we investigate the semi-structured nature of clinical notes and propose an automatic algorithm to segment them into sections. To address the variability issues in existing ICD coding models with limited data, we introduce a contrastive pre-training approach on sections using a soft multi-label similarity metric based on tree edit distance. Additionally, we design a masked section training strategy to enable ICD coding models to locate sections related to ICD codes. Extensive experimental results demonstrate that our proposed training strategies effectively enhance the performance of existing ICD coding methods.</details>

### Towards Test-Time Refusals via Concept Negation

**Authors:** Peiran Dong, Song Guo, Junxiao Wang, Bingjie WANG, Jiewei Zhang, Ziming Liu

**<details><summary>Abstract**</summary>

Generative models produce unbounded outputs, necessitating the use of refusal techniques to confine their output space. Employing generative refusals is crucial in upholding the ethical and copyright integrity of synthesized content, particularly when working with widely adopted diffusion models. "Concept negation'' presents a promising paradigm to achieve generative refusals, as it effectively defines and governs the model's output space based on concepts, utilizing natural language interfaces that are readily comprehensible to humans. However, despite the valuable contributions of prior research to the field of concept negation, it still suffers from significant limitations. The existing concept negation methods, which operate based on the composition of score or noise predictions from the diffusion process, are limited to independent concepts (e.g., ``a blonde girl`` without ``glasses``) and fail to consider the interconnected nature of concepts in reality (e.g., ``Mickey mouse eats ice cream`` without ``Disney characters``). Keeping the limitations in mind, we propose a novel framework, called $ProtoRe$, to improve the flexibility of concept negation via test-time negative concept identification along with purification in the feature space. $ProtoRe$ works by incorporating CLIP's language-contrastive knowledge to identify the prototype of negative concepts, extract the negative features from outputs using the prototype as a prompt, and further refine the attention maps by retrieving negative features. Our evaluation on multiple benchmarks shows that $ProtoRe$ outperforms state-of-the-art methods under various settings, in terms of the effectiveness of purification and the fidelity of generative images.</details>

### Tracking Most Significant Shifts in Nonparametric Contextual Bandits

**Authors:** Joe Suk, Samory Kpotufe

**<details><summary>Abstract**</summary>

We study nonparametric contextual bandits where Lipschitz mean reward functions may change over time.We first establish the minimax dynamic regret rate in this less understood setting in terms of number of changes $L$ and total-variation $V$, both capturing all changes in distribution over context space, and argue that state-of-the-art procedures are suboptimal in this setting.Next, we tend to the question of an _adaptivity_ for this setting, i.e. achieving the minimax rate without knowledge of $L$ or $V$. Quite importantly, we posit that the bandit  problem, viewed locally at a given context $X_t$, should not be affected by reward changes in other parts of context space $\cal X$. We therefore propose a notion of _change_, which we term _experienced significant shifts_, that better accounts for locality, and thus counts considerably less changes than $L$ and $V$. Furthermore, similar to recent work on non-stationary MAB (Suk & Kpotufe, 2022), _experienced significant shifts_ only count the most _significant_ changes in mean rewards, e.g., severe best-arm changes relevant to observed contexts.Our main result is to show that this more tolerant notion of change can in fact be adapted to.</details>

### Train Faster, Perform Better: Modular Adaptive Training in Over-Parameterized Models

**Authors:** Yubin Shi, Yixuan Chen, Mingzhi Dong, Xiaochen Yang, Dongsheng Li, Yujiang Wang, Robert Dick, Qin Lv, Yingying Zhao, Fan Yang, Tun Lu, Ning Gu, Li Shang

**<details><summary>Abstract**</summary>

Despite their prevalence in deep-learning communities, over-parameterized models convey high demands of computational costs for proper training. This work studies the fine-grained, modular-level learning dynamics of over-parameterized models to attain a more efficient and fruitful training strategy. Empirical evidence reveals that when scaling down into network modules, such as heads in self-attention models, we can observe varying learning patterns implicitly associated with each module's trainability. To describe such modular-level learning capabilities, we introduce a novel concept dubbed modular neural tangent kernel (mNTK), and we demonstrate that the quality of a module's learning is tightly associated with its mNTK's principal eigenvalue $\lambda_{\max}$. A large $\lambda_{\max}$  indicates that the module learns features with better convergence, while those miniature ones may impact generalization negatively. Inspired by the discovery, we propose a novel training strategy termed Modular Adaptive Training (MAT) to update those modules with their $\lambda_{\max}$ exceeding a dynamic threshold selectively, concentrating the model on learning common features and ignoring those inconsistent ones. Unlike most existing training schemes with a complete BP cycle across all network modules, MAT can significantly save computations by its partially-updating strategy and can further improve performance. Experiments show that MAT nearly halves the computational cost of model training and outperforms the accuracy of baselines.</details>

### Train Hard, Fight Easy: Robust Meta Reinforcement Learning

**Authors:** Ido Greenberg, Shie Mannor, Gal Chechik, Eli Meirom

**<details><summary>Abstract**</summary>

A major challenge of reinforcement learning (RL) in real-world applications is the variation between environments, tasks or clients. Meta-RL (MRL) addresses this issue by learning a meta-policy that adapts to new tasks. Standard MRL methods optimize the average return over tasks, but often suffer from poor results in tasks of high risk or difficulty. This limits system reliability since test tasks are not known in advance. In this work, we define a robust MRL objective with a controlled robustness level. Optimization of analogous robust objectives in RL is known to lead to both **biased gradients** and **data inefficiency**. We prove that the gradient bias disappears in our proposed MRL framework. The data inefficiency is addressed via the novel Robust Meta RL algorithm (RoML). RoML is a meta-algorithm that generates a robust version of any given MRL algorithm, by identifying and over-sampling harder tasks throughout training. We demonstrate that RoML achieves robust returns on multiple navigation and continuous control benchmarks.</details>

### Training Neural Networks is NP-Hard in Fixed Dimension

**Authors:** Vincent Froese, Christoph Hertrich

**<details><summary>Abstract**</summary>

We study the parameterized complexity of training two-layer neural networks with respect to the dimension of the input data and the number of hidden neurons, considering ReLU and linear threshold activation functions. Albeit the computational complexity of these problems has been studied numerous times in recent years, several questions are still open. We answer questions by Arora et al. (ICLR 2018) and Khalife and Basu (IPCO 2022) showing that both problems are NP-hard for two dimensions, which excludes any polynomial-time algorithm for constant dimension. We also answer a question by Froese et al. (JAIR 2022) proving W[1]-hardness for four ReLUs (or two linear threshold neurons) with zero training error. Finally, in the ReLU case, we show fixed-parameter tractability for the combined parameter number of dimensions and number of ReLUs if the network is assumed to compute a convex map. Our results settle the complexity status regarding these parameters almost completely.</details>

### [Oral] Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection

**Authors:** Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, Song Mei

**Oral Presentation:** We, Dec 13, 14:00 -- Oral 4C

**<details><summary>Abstract**</summary>

Neural sequence models based on the transformer architecture have demonstrated remarkable \emph{in-context learning} (ICL) abilities, where they can perform new tasks when prompted with training and test examples, without any parameter update to the model. This work first provides a comprehensive statistical theory for transformers to perform ICL. Concretely, we show that transformers can implement a broad class of standard machine learning algorithms in context, such as least squares, ridge regression, Lasso, learning generalized linear models, and gradient descent on two-layer neural networks, with near-optimal predictive power on various in-context data distributions. Using an efficient implementation of in-context gradient descent as the underlying mechanism, our transformer constructions admit mild size bounds, and can be learned with polynomially many pretraining sequences.    Building on these ``base'' ICL algorithms, intriguingly, we show that transformers can implement more complex ICL procedures involving \emph{in-context algorithm selection}, akin to what a statistician can do in real life---A \emph{single} transformer can adaptively select different base ICL algorithms---or even perform qualitatively different tasks---on different input sequences, without any explicit prompting of the right algorithm or task. We both establish this in theory by explicit constructions, and also observe this phenomenon experimentally. In theory, we construct two general mechanisms for algorithm selection with concrete examples: pre-ICL testing, and post-ICL validation. As an example, we use the post-ICL validation mechanism to construct a transformer that can perform nearly Bayes-optimal ICL on a challenging task---noisy linear models with mixed noise levels. Experimentally, we demonstrate the strong in-context algorithm selection capabilities of standard transformer architectures.</details>

### Transportability for Bandits with Data from Different Environments

**Authors:** Alexis Bellot, Alan Malek, Silvia Chiappa

**<details><summary>Abstract**</summary>

A unifying theme in the design of intelligent agents is to efficiently optimize a policy based on what prior knowledge of the problem is available and what actions can be taken to learn more about it. Bandits are a canonical instance of this task that has been intensely studied in the literature. Most methods, however, typically rely solely on an agent's experimentation in a single environment (or multiple closely related environments). In this paper, we relax this assumption and consider the design of bandit algorithms from a combination of batch data and qualitative assumptions about the relatedness across different environments, represented in the form of causal models. In particular, we show that it is possible to exploit invariances across environments, wherever they may occur in the underlying causal model, to consistently improve learning. The resulting bandit algorithm has a sub-linear regret bound with an explicit dependency on a term that captures how informative related environments are for the task at hand; and may have substantially lower regret than experimentation-only bandit instances.</details>

### [Spotlight] Tree Variational Autoencoders

**Authors:** Laura Manduchi, Moritz Vandenhirtz, Alain Ryser, Julia Vogt

**<details><summary>Abstract**</summary>

We propose Tree Variational Autoencoder (TreeVAE), a new generative hierarchical clustering model  that learns a flexible tree-based posterior distribution over latent variables. TreeVAE hierarchically divides samples according to their intrinsic characteristics, shedding light on hidden structures in the data. It adapts its architecture to discover the optimal tree for encoding dependencies between latent variables. The proposed tree-based generative architecture enables lightweight conditional inference and improves generative performance by utilizing specialized leaf decoders.   We show that TreeVAE uncovers underlying clusters in the data and finds meaningful hierarchical relations between the different groups on a variety of datasets, including real-world imaging data.   We present empirically that TreeVAE provides a more competitive log-likelihood lower bound than the sequential counterparts.   Finally, due to its generative nature, TreeVAE is able to generate new samples from the discovered clusters via conditional sampling.</details>

### [Oral] Tree of Thoughts: Deliberate Problem Solving with Large Language Models

**Authors:** Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, Karthik Narasimhan

**Oral Presentation:** We, Dec 13, 13:45 -- Oral 4C

**<details><summary>Abstract**</summary>

Language models are increasingly being deployed for general problem solving across a wide range of tasks, but are still confined to token-level, left-to-right decision-making processes during inference. This means they can fall short in tasks that require exploration, strategic lookahead, or where initial decisions play a pivotal role. To surmount these challenges, we introduce a new framework for language model inference, Tree of Thoughts (ToT), which generalizes over the popular Chain of Thought approach to prompting language models, and enables exploration over coherent units of text (thoughts) that serve as intermediate steps toward problem solving. ToT allows LMs to perform deliberate decision making by considering multiple different reasoning paths and self-evaluating choices to decide the next course of action, as well as looking ahead or backtracking when necessary to make global choices.Our experiments show that ToT significantly enhances language models' problem-solving abilities on three novel tasks requiring non-trivial planning or search: Game of 24, Creative Writing, and Mini Crosswords. For instance, in Game of 24, while GPT-4 with chain-of-thought prompting only solved 4\% of tasks, our method achieved a success rate of 74\%. Code repo with all prompts: https://github.com/princeton-nlp/tree-of-thought-llm.</details>

### TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models

**Authors:** Jiaqi Xue, Mengxin Zheng, Ting Hua, Yilin Shen, Yepeng Liu, Ladislau Bölöni, Qian Lou

**<details><summary>Abstract**</summary>

Large Language Models (LLMs) are progressively being utilized as machine learning services and interface tools for various applications. However, the security implications of LLMs, particularly in relation to adversarial and Trojan attacks, remain insufficiently examined. In this paper, we propose TrojLLM, an automatic and black-box framework to effectively generate universal and stealthy triggers. When these triggers are incorporated into the input data, the LLMs' outputs can be maliciously manipulated. Moreover, the framework also supports embedding Trojans within discrete prompts, enhancing the overall effectiveness and precision of the triggers' attacks. Specifically, we propose a trigger discovery algorithm for generating universal triggers for various inputs by querying victim LLM-based APIs using few-shot data samples. Furthermore, we introduce a novel progressive Trojan poisoning algorithm designed to generate poisoned prompts that retain efficacy and transferability across a diverse range of models. Our experiments and results demonstrate TrojLLM's capacity to effectively insert Trojans into text prompts in real-world black-box LLM APIs including GPT-3.5 and GPT-4, while maintaining exceptional performance on clean test sets. Our work sheds light on the potential security risks in current models and offers a potential defensive approach. The source code of TrojLLM is available at https://github.com/UCF-ML-Research/TrojLLM.</details>

### Trust Your $\nabla$: Gradient-based Intervention Targeting for Causal Discovery

**Authors:** Mateusz Olko, Michał Zając, Aleksandra Nowak, Nino Scherrer, Yashas Annadani, Stefan Bauer, Łukasz Kuciński, Piotr Miłoś

**<details><summary>Abstract**</summary>

Inferring causal structure from data is a challenging task of fundamental importance in science. Often, observational data alone is not enough to uniquely identify a system’s causal structure. The use of interventional data can address this issue, however, acquiring these samples typically demands a considerable investment of time and physical or financial resources. In this work, we are concerned with the acquisition of interventional data in a targeted manner to minimize the number of required experiments. We propose a novel Gradient-based Intervention Targeting method, abbreviated GIT, that ’trusts’ the gradient estimator of a gradient-based causal discovery framework to provide signals for the intervention targeting function. We provide extensive experiments in simulated and real-world datasets and demonstrate that GIT performs on par with competitive baselines, surpassing them in the low-data regime.</details>

### Two Sides of The Same Coin: Bridging Deep Equilibrium Models and Neural ODEs via Homotopy Continuation

**Authors:** Shutong Ding, Tianyu Cui, Jingya Wang, Ye Shi

**<details><summary>Abstract**</summary>

Deep Equilibrium Models (DEQs) and Neural Ordinary Differential Equations (Neural ODEs) are two branches of implicit models that have achieved remarkable success owing to their superior performance and low memory consumption. While both are implicit models, DEQs and Neural ODEs are derived from different mathematical formulations. Inspired by homotopy continuation, we establish a connection between these two models and illustrate that they are actually two sides of the same coin. Homotopy continuation is a classical method of solving nonlinear equations based on a corresponding ODE. Given this connection, we proposed a new implicit model called HomoODE that inherits the property of high accuracy from DEQs and the property of stability from Neural ODEs. Unlike DEQs, which explicitly solve an equilibrium-point-finding problem via Newton's methods in the forward pass, HomoODE solves the equilibrium-point-finding problem implicitly using a modified Neural ODE via homotopy continuation. Further, we developed an acceleration method for HomoODE with a shared learnable initial point. It is worth noting that our model also provides a better understanding of why Augmented Neural ODEs work as long as the augmented part is regarded as the equilibrium point to find. Comprehensive experiments with several image classification tasks demonstrate that HomoODE surpasses existing implicit models in terms of both accuracy and memory consumption.</details>

### UNSSOR: Unsupervised Neural Speech Separation by Leveraging Over-determined Training Mixtures

**Authors:** Zhong-Qiu Wang, Shinji Watanabe

**<details><summary>Abstract**</summary>

In reverberant conditions with multiple concurrent speakers, each microphone acquires a mixture signal of multiple speakers at a different location. In over-determined conditions where the microphones out-number speakers, we can narrow down the solutions to speaker images and realize unsupervised speech separation by leveraging each mixture signal as a constraint (i.e., the estimated speaker images at a microphone should add up to the mixture). Equipped with this insight, we propose UNSSOR, an algorithm for $\underline{u}$nsupervised $\underline{n}$eural $\underline{s}$peech $\underline{s}$eparation by leveraging $\underline{o}$ver-determined training mixtu$\underline{r}$es. At each training step, we feed an input mixture to a deep neural network (DNN) to produce an intermediate estimate for each speaker, linearly filter the estimates, and optimize a loss so that, at each microphone, the filtered estimates of all the speakers can add up to the mixture to satisfy the above constraint. We show that this loss can promote unsupervised separation of speakers. The linear filters are computed in each sub-band based on the mixture and DNN estimates through the forward convolutive prediction (FCP) algorithm. To address the frequency permutation problem incurred by using sub-band FCP, a loss term based on minimizing intra-source magnitude scattering is proposed. Although UNSSOR requires over-determined training mixtures, we can train DNNs to achieve under-determined separation (e.g., unsupervised monaural speech separation). Evaluation results on two-speaker separation in reverberant conditions show the effectiveness and potential of UNSSOR.</details>

### UP-DP: Unsupervised Prompt Learning for Data Pre-Selection with Vision-Language Models

**Authors:** Xin Li, Sima Behpour, Thang Long Doan, Wenbin He, Liang Gou, Liu Ren

**<details><summary>Abstract**</summary>

In this study, we investigate the task of data pre-selection, which aims to select instances for labeling from an unlabeled dataset through a single pass, thereby optimizing performance for undefined downstream tasks with a limited annotation budget. Previous approaches to data pre-selection relied solely on visual features extracted from foundation models, such as CLIP and BLIP-2, but largely ignored the powerfulness of text features. In this work, we argue that, with proper design, the joint feature space of both vision and text can yield a better representation for data pre-selection. To this end, we introduce UP-DP, a simple yet effective unsupervised prompt learning approach that adapts vision-language models, like BLIP-2, for data pre-selection. Specifically, with the BLIP-2 parameters frozen, we train text prompts to extract the joint features with improved representation, ensuring a diverse cluster structure that covers the entire dataset. We extensively compare our method with the state-of-the-art using seven benchmark datasets in different settings, achieving up to a performance gain of 20\%. Interestingly, the prompts learned from one dataset demonstrate significant generalizability and can be applied directly to enhance the feature extraction of BLIP-2 from other datasets. To the best of our knowledge, UP-DP is the first work to incorporate unsupervised prompt learning in a vision-language model for data pre-selection.</details>

### UP-NeRF: Unconstrained Pose Prior-Free Neural Radiance Field

**Authors:** Injae Kim, Minhyuk Choi, Hyunwoo Kim

**<details><summary>Abstract**</summary>

Neural Radiance Field (NeRF) has enabled novel view synthesis with high fidelity given images and camera poses. Subsequent works even succeeded in eliminating the necessity of pose priors by jointly optimizing NeRF and camera pose. However, these works are limited to relatively simple settings such as photometrically consistent and occluder-free image collections or a sequence of images from a video. So they have difficulty handling unconstrained images with varying illumination and transient occluders. In this paper, we propose **UP-NeRF** (**U**nconstrained **P**ose-prior-free **Ne**ural **R**adiance **F**ields) to optimize NeRF with unconstrained image collections without camera pose prior. We tackle these challenges with surrogate tasks that optimize color-insensitive feature fields and a separate module for transient occluders to block their influence on pose estimation. In addition, we introduce a candidate head to enable more robust pose estimation and transient-aware depth supervision to minimize the effect of incorrect prior. Our experiments verify the superior performance of our method compared to the baselines including BARF and its variants in a challenging internet photo collection, *Phototourism dataset*. The code of UP-NeRF is available at https://github.com/mlvlab/UP-NeRF.</details>

### Unbalanced Low-rank Optimal Transport Solvers

**Authors:** Meyer Scetbon, Michal Klein, Giovanni Palla, Marco Cuturi

**<details><summary>Abstract**</summary>

The relevance of optimal transport methods to machine learning has long been hindered by two salient limitations.First, the $O(n^3)$ computational cost of standard sample-based solvers (when used on batches of $n$ samples) is prohibitive.Second, the mass conservation constraint makes OT solvers too rigid in practice: because they must match \textit{all} points from both measures, their output can be heavily influenced by outliers.A flurry of recent works in OT has addressed these computational and modelling limitations, but has resulted in two separate strains of methods:While the computational outlook was much improved by entropic regularization, more recent $O(n)$ linear-time \textit{low-rank} solvers hold the promise to scale up OT further.On the other hand, modelling rigidities have been eased owing to unbalanced variants of OT, that rely on penalization terms to promote, rather than impose, mass conservation.The goal of this paper is to merge these two strains, to achieve the promise of \textit{both} versatile/scalable unbalanced/low-rank OT solvers. We propose custom algorithms to implement these extensions for the linear OT problem and its Fused-Gromov-Wasserstein generalization, and demonstrate their practical relevance to challenging spatial transcriptomics matching problems.</details>

### [Spotlight] Uncovering the Hidden Dynamics of Video Self-supervised Learning under Distribution Shifts

**Authors:** Pritam Sarkar, Ahmad Beirami, Ali Etemad

**<details><summary>Abstract**</summary>

Video self-supervised learning (VSSL) has made significant progress in recent years. However, the exact behavior and dynamics of these models under different forms of distribution shift are not yet known. In this paper, we comprehensively study the behavior of six popular self-supervised methods (v-SimCLR, v-MoCo, v-BYOL, v-SimSiam, v-DINO, v-MAE) in response to various forms of natural distribution shift, i.e., (i) context shift, (ii) viewpoint shift, (iii) actor shift, (iv) source shift, (v) generalizability to unknown classes (zero-shot), and (vi) open-set recognition. To perform this extensive study, we carefully craft a test bed consisting of 17 in-distribution and out-of-distribution benchmark pairs using available public datasets and a series of evaluation protocols to stress-test the different methods under the intended shifts. Our study uncovers a series of intriguing findings and interesting behaviors of VSSL methods. For instance, we observe that while video models generally struggle with context shifts, v-MAE and supervised learning exhibit more robustness. Moreover, our study shows that v-MAE is a strong temporal learner, whereas contrastive methods, v-SimCLR and v-MoCo, exhibit strong performances against viewpoint shifts. When studying the notion of open-set recognition, we notice a trade-off between closed-set and open-set recognition performance if the pretrained VSSL encoders are used without finetuning. We hope that our work will contribute to the development of robust video representation learning frameworks for various real-world scenarios. The project page and code are available at: https://pritamqu.github.io/OOD-VSSL.</details>

### Understanding Neural Network Binarization with Forward and Backward Proximal Quantizers

**Authors:** Yiwei Lu, Yaoliang Yu, Xinlin Li, Vahid Partovi Nia

**<details><summary>Abstract**</summary>

In neural network binarization, BinaryConnect (BC) and its variants are considered the standard. These methods apply the sign function in their forward pass and their respective gradients are backpropagated to update the weights. However, the derivative of the sign function is zero whenever defined, which consequently freezes training. Therefore, implementations of BC (e.g., BNN) usually replace the derivative of sign in the backward computation with identity or other approximate gradient alternatives. Although such practice works well empirically, it is largely a heuristic or ``training trick.'' We aim at shedding some light on these training tricks from the optimization perspective. Building from existing theory on ProxConnect (PC, a generalization of BC), we (1) equip PC with different forward-backward quantizers and obtain ProxConnect++ (PC++) that includes existing binarization techniques as special cases; (2) derive a principled way to synthesize forward-backward quantizers with automatic theoretical guarantees; (3) illustrate our theory by proposing an enhanced binarization algorithm BNN++; (4) conduct image classification experiments on CNNs and vision transformers, and empirically verify that BNN++ generally achieves competitive results on binarizing these models.</details>

### Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry

**Authors:** Yong-Hyun Park, Mingi Kwon, Jaewoong Choi, Junghyo Jo, Youngjung Uh

**<details><summary>Abstract**</summary>

Despite the success of diffusion models (DMs), we still lack a thorough understanding of their latent space. To understand the latent space $\mathbf{x}_t \in \mathcal{X}$, we analyze them from a geometrical perspective. Our approach involves deriving the local latent basis within $\mathcal{X}$ by leveraging the pullback metric associated with their encoding feature maps. Remarkably, our discovered local latent basis enables image editing capabilities by moving $\mathbf{x}_t$, the latent space of DMs, along the basis vector at specific timesteps. We further analyze how the geometric structure of DMs evolves over diffusion timesteps and differs across different text conditions. This confirms the known phenomenon of coarse-to-fine generation, as well as reveals novel insights such as the discrepancy between $\mathbf{x}_t$ across timesteps, the effect of dataset complexity, and the time-varying influence of text prompts. To the best of our knowledge, this paper is the first to present image editing through $\mathbf{x}$-space traversal, editing only once at specific timestep $t$ without any additional training, and providing thorough analyses of the latent structure of DMs.The code to reproduce our experiments can be found at the [link](https://github.com/enkeejunior1/Diffusion-Pullback).</details>

### Uniform Convergence with Square-Root Lipschitz Loss

**Authors:** Lijia Zhou, Zhen Dai, Frederic Koehler, Nati Srebro

**<details><summary>Abstract**</summary>

We establish generic uniform convergence guarantees for Gaussian data in terms of the Radamacher complexity of the hypothesis class and the Lipschitz constant of the square root of the scalar loss function.  We show how these guarantees substantially generalize previous results based on smoothness (Lipschitz constant of the derivative), and allow us to handle the broader class of square-root-Lipschtz losses, which includes also non-smooth loss functions appropriate for studying phase retrieval and ReLU regression, as well as rederive and better understand “optimistic rate” and interpolation learning guarantees.</details>

### Unlocking Deterministic Robustness Certification on ImageNet

**Authors:** Kai Hu, Andy Zou, Zifan Wang, Klas Leino, Matt Fredrikson

**<details><summary>Abstract**</summary>

Despite the promise of Lipschitz-based methods for provably-robust deep learning with deterministic guarantees, current state-of-the-art results are limited to feed-forward Convolutional Networks (ConvNets) on low-dimensional data, such as CIFAR-10. This paper investigates strategies for expanding certifiably robust training to larger, deeper models.A key challenge in certifying deep networks is efficient calculation of the Lipschitz bound for residual blocks found in ResNet and ViT architectures.We show that fast ways of bounding the Lipschitz constant for conventional ResNets are loose, and show how to address this by designing a new residual block, leading to the *Linear ResNet* (LiResNet) architecture.We then introduce *Efficient Margin MAximization* (EMMA), a loss function that stabilizes robust training by penalizing worst-case adversarial examples from multiple classes simultaneously.Together, these contributions yield new *state-of-the-art* robust accuracy on CIFAR-10/100 and Tiny-ImageNet under $\ell_2$ perturbations.Moreover, for the first time, we are able to scale up fast deterministic robustness guarantees to ImageNet, demonstrating that this approach to robust learning can be applied to real-world applications.</details>

### Unsupervised Anomaly Detection with Rejection

**Authors:** Lorenzo Perini, Jesse Davis

**<details><summary>Abstract**</summary>

Anomaly detection aims at detecting unexpected behaviours in the data. Because anomaly detection is usually an unsupervised task, traditional anomaly detectors learn a decision boundary by employing heuristics based on intuitions, which are hard to verify in practice. This introduces some uncertainty, especially close to the decision boundary, that may reduce the user trust in the detector's predictions. A way to combat this is by allowing the detector to reject predictions with high uncertainty (Learning to Reject). This requires employing a confidence metric that captures the distance to the decision boundary and setting a rejection threshold to reject low-confidence predictions. However, selecting a proper metric and setting the rejection threshold without labels are challenging tasks. In this paper, we solve these challenges by setting a constant rejection threshold on the stability metric computed by ExCeeD. Our insight relies on a theoretical analysis of such a metric. Moreover, setting a constant threshold results in strong guarantees: we estimate the test rejection rate, and derive a theoretical upper bound for both the rejection rate and the expected prediction cost. Experimentally, we show that our method outperforms some metric-based methods.</details>

### Unsupervised Graph Neural Architecture Search with Disentangled Self-Supervision

**Authors:** Zeyang Zhang, Xin Wang, Ziwei Zhang, Guangyao Shen, Shiqi Shen, Wenwu Zhu

**<details><summary>Abstract**</summary>

The existing graph neural architecture search (GNAS) methods heavily rely on supervised labels during the search process, failing to handle ubiquitous scenarios where supervisions are not available. In this paper, we study the problem of unsupervised graph neural architecture search, which remains unexplored in the literature. The key problem is to discover the latent graph factors that drive the formation of graph data as well as the underlying relations between the factors and the optimal neural architectures. Handling this problem is challenging given that the latent graph factors together with architectures are highly entangled due to the nature of the graph and the complexity of the neural architecture search process. To address the challenge, we propose a novel Disentangled Self-supervised Graph Neural Architecture Search (DSGAS) model, which is able to discover the optimal architectures capturing various latent graph factors in a self-supervised fashion based on unlabeled graph data. Specifically, we first design a disentangled graph super-network capable of incorporating multiple architectures with factor-wise disentanglement, which are optimized simultaneously. Then, we estimate the performance of architectures under different factors by our proposed self-supervised training with joint architecture-graph disentanglement. Finally, we propose a contrastive search with architecture augmentations to discover architectures with factor-specific expertise. Extensive experiments on 11 real-world datasets demonstrate that the proposed model is able to achieve state-of-the-art performance against several baseline methods in an unsupervised manner.</details>

### Unsupervised Image Denoising with Score Function

**Authors:** Yutong Xie, Mingze Yuan, Bin Dong, Quanzheng Li

**<details><summary>Abstract**</summary>

Though achieving excellent performance in some cases, current unsupervised learning methods for single image denoising usually have constraints in applications. In this paper, we propose a new approach which is more general and applicable to complicated noise models. Utilizing the property of score function, the gradient of logarithmic probability, we define a solving system for denoising. Once the score function of noisy images has been estimated, the denoised result can be obtained through the solving system. Our approach can be applied to multiple noise models, such as the mixture of multiplicative and additive noise combined with structured correlation. Experimental results show that our method is comparable when the noise model is simple, and has good performance in complicated cases where other methods are not applicable or perform poorly.</details>

### Unsupervised Video Domain Adaptation for Action Recognition: A Disentanglement Perspective

**Authors:** Pengfei Wei, Lingdong Kong, Xinghua Qu, Yi Ren, Zhiqiang Xu, Jing Jiang, Xiang Yin

**<details><summary>Abstract**</summary>

Unsupervised video domain adaptation is a practical yet challenging task. In this work, for the first time, we tackle it from a disentanglement view. Our key idea is to handle the spatial and temporal domain divergence separately through disentanglement. Specifically, we consider the generation of cross-domain videos from two sets of latent factors, one encoding the static information and another encoding the dynamic information. A Transfer Sequential VAE (TranSVAE) framework is then developed to model such generation. To better serve for adaptation, we propose several objectives to constrain the latent factors. With these constraints, the spatial divergence can be readily removed by disentangling the static domain-specific information out, and the temporal divergence is further reduced from both frame- and video-levels through adversarial learning. Extensive experiments on the UCF-HMDB, Jester, and Epic-Kitchens datasets verify the effectiveness and superiority of TranSVAE compared with several state-of-the-art approaches.</details>

### VRA: Variational Rectified Activation for Out-of-distribution Detection

**Authors:** Mingyu Xu, Zheng Lian, Bin Liu, Jianhua Tao

**<details><summary>Abstract**</summary>

Out-of-distribution (OOD) detection is critical to building reliable machine learning systems in the open world. Researchers have proposed various strategies to reduce model overconfidence on OOD data. Among them, ReAct is a typical and effective technique to deal with model overconfidence, which truncates high activations to increase the gap between in-distribution and OOD. Despite its promising results, is this technique the best choice? To answer this question, we leverage the variational method to find the optimal operation and verify the necessity of suppressing abnormally low and high activations and amplifying intermediate activations in OOD detection, rather than focusing only on high activations like ReAct. This motivates us to propose a novel technique called ``Variational Rectified Activation (VRA)'', which simulates these suppression and amplification operations using piecewise functions. Experimental results on multiple benchmark datasets demonstrate that our method outperforms existing post-hoc strategies. Meanwhile, VRA is compatible with different scoring functions and network architectures. Our code is available at https://github.com/zeroQiaoba/VRA.</details>

### Variance-Reduced Gradient Estimation via Noise-Reuse in Online Evolution Strategies

**Authors:** Oscar Li, James Harrison, Jascha Sohl-Dickstein, Virginia Smith, Luke Metz

**<details><summary>Abstract**</summary>

Unrolled computation graphs are prevalent throughout machine learning but present challenges to automatic differentiation (AD) gradient estimation methods when their loss functions exhibit extreme local sensitivtiy, discontinuity, or blackbox characteristics. In such scenarios, online evolution strategies methods are a more capable alternative, while being more parallelizable than vanilla evolution strategies (ES) by interleaving partial unrolls and gradient updates. In this work, we propose a general class of unbiased online evolution strategies methods. We analytically and empirically characterize the variance of this class of gradient estimators and identify the one with the least variance, which we term Noise-Reuse Evolution Strategies (NRES). Experimentally, we show NRES results in faster convergence than existing AD and ES methods in terms of wall-clock time and number of unroll steps across a variety of applications, including learning dynamical systems, meta-training learned optimizers, and reinforcement learning.</details>

### Variational Annealing on Graphs for Combinatorial Optimization

**Authors:** Sebastian Sanokowski, Wilhelm Berghammer, Sepp Hochreiter, Sebastian Lehner

**<details><summary>Abstract**</summary>

Several recent unsupervised learning methods use probabilistic approaches to solve combinatorial optimization (CO) problems based on the assumption of statistically independent solution variables. We demonstrate that this assumption imposes performance limitations in particular on difficult problem instances. Our results corroborate that an autoregressive approach which captures statistical dependencies among solution variables yields superior performance on many popular CO problems. We introduce Subgraph Tokenization in which the configuration of a set of solution variables is represented by a single token. This tokenization technique alleviates the drawback of the long sequential sampling procedure which is inherent to autoregressive methods without sacrificing expressivity. Importantly, we theoretically motivate an annealed entropy regularization and show empirically that it is essential for efficient and stable learning.</details>

### VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks

**Authors:** Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, Jifeng Dai

**<details><summary>Abstract**</summary>

Large language models (LLMs) have notably accelerated progress towards artificial general intelligence (AGI), with their impressive zero-shot capacity for user-tailored tasks, endowing them with immense potential across a range of applications. However, in the field of computer vision, despite the availability of numerous powerful vision foundation models (VFMs), they are still restricted to tasks in a pre-defined form, struggling to match the open-ended task capabilities of LLMs. In this work, we present an LLM-based framework for vision-centric tasks, termed VisionLLM. This framework provides a unified perspective for vision and language tasks by treating images as a foreign language and aligning vision-centric tasks with language tasks that can be flexibly defined and managed using language instructions. An LLM-based decoder can then make appropriate predictions based on these instructions for open-ended tasks. Extensive experiments show that the proposed VisionLLM can achieve different levels of task customization through language instructions, from fine-grained object-level to coarse-grained task-level customization, all with good results. It's noteworthy that, with a generalist LLM-based framework, our model can achieve over 60% mAP on COCO, on par with detection-specific models. We hope this model can set a new baseline for generalist vision and language models. The code shall be released.</details>

### WCLD: Curated Large Dataset of Criminal Cases from Wisconsin Circuit Courts

**Authors:** Elliott Ash, Naman Goel, Nianyun Li, Claudia Marangon, Peiyao Sun

**<details><summary>Abstract**</summary>

Machine learning based decision-support tools in criminal justice systems are subjects of intense discussions and academic research. There are important open questions about the utility and fairness of such tools. Academic researchers often rely on a few small datasets that are not sufficient to empirically study various real-world aspects of these questions. In this paper, we contribute WCLD, a curated large dataset of 1.5 million criminal cases from circuit courts in the U.S. state of Wisconsin. We used reliable public data from 1970 to 2020 to curate attributes like prior criminal counts and recidivism outcomes. The dataset contains large number of samples from five racial groups, in addition to information like sex and age (at judgment and first offense). Other attributes in this dataset include neighborhood characteristics obtained from census data, detailed types of offense, charge severity, case decisions, sentence lengths, year of filing etc. We also provide pseudo-identifiers for judge, county and zipcode. The dataset will not only enable researchers to more rigorously study algorithmic fairness in the context of criminal justice, but also relate algorithmic challenges with various systemic issues. We also discuss in detail the process of constructing the dataset and provide a datasheet. The WCLD dataset is available at https://clezdata.github.io/wcld/.</details>

### Weighted ROC Curve in Cost Space: Extending AUC to Cost-Sensitive Learning

**Authors:** HuiYang Shao, Qianqian Xu, Zhiyong Yang, Peisong Wen, Gao Peifeng, Qingming Huang

**<details><summary>Abstract**</summary>

In this paper, we aim to tackle flexible cost requirements for long-tail datasets, where we need to construct a (a) cost-sensitive and (b) class-distribution robust learning framework. The misclassification cost and the area under the ROC curve (AUC) are popular metrics for (a) and (b), respectively. However, limited by their formulations, models trained with AUC cannot be applied to cost-sensitive decision problems, and models trained with fixed costs are sensitive to the class distribution shift. To address this issue, we present a new setting where costs are treated like a dataset to deal with arbitrarily unknown cost distributions. Moreover, we propose a novel weighted version of AUC where the cost distribution can be integrated into its calculation through decision thresholds.  To formulate this setting, we propose a novel bilevel paradigm to bridge weighted AUC (WAUC) and cost. The inner-level problem approximates the optimal threshold from sampling costs, and the outer-level problem minimizes the WAUC loss over the optimal threshold distribution. To optimize this bilevel paradigm, we employ a stochastic optimization algorithm (SACCL) to optimize it. Finally, experiment results show that our algorithm performs better than existing cost-sensitive learning methods and two-stage AUC decisions approach.</details>

### When can Regression-Adjusted Control Variate Help? Rare Events, Sobolev Embedding and Minimax Optimality

**Authors:** Jose Blanchet, Haoxuan Chen, Yiping Lu, Lexing Ying

**<details><summary>Abstract**</summary>

This paper studies the use of a machine learning-based estimator as a control variate for mitigating the variance of Monte Carlo sampling. Specifically, we seek to uncover the key factors that influence the efficiency of control variates in reducing variance. We examine a prototype estimation problem that involves simulating the moments of a Sobolev function based on observations obtained from (random) quadrature nodes. Firstly, we establish an information-theoretic lower bound for the problem. We then study a specific quadrature rule that employs a nonparametric regression-adjusted control variate to reduce the variance of the Monte Carlo simulation. We demonstrate that this kind of quadrature rule can improve the Monte Carlo rate and achieve the minimax optimal rate under a sufficient smoothness assumption. Due to the Sobolev Embedding Theorem, the sufficient smoothness assumption eliminates the existence of rare and extreme events. Finally, we show that, in the presence of rare and extreme events, a truncated version of the Monte Carlo algorithm can achieve the minimax optimal rate while the control variate cannot improve the convergence rate.</details>

### [Spotlight] Which Models have Perceptually-Aligned Gradients? An Explanation via Off-Manifold Robustness

**Authors:** Suraj Srinivas, Sebastian Bordt, Himabindu Lakkaraju

**<details><summary>Abstract**</summary>

One of the remarkable properties of robust computer vision models is that their input-gradients are often aligned with human perception, referred to in the literature as perceptually-aligned gradients (PAGs). Despite only being trained for classification, PAGs cause robust models to have rudimentary generative capabilities, including image generation, denoising, and in-painting. However, the underlying mechanisms behind these phenomena remain unknown. In this work, we provide a first explanation of PAGs via \emph{off-manifold robustness}, which states that models must be more robust off- the data manifold than they are on-manifold. We first demonstrate theoretically that off-manifold robustness leads input gradients to lie approximately on the data manifold, explaining their perceptual alignment. We then show that Bayes optimal models satisfy off-manifold robustness, and confirm the same empirically for robust models trained via gradient norm regularization, randomized smoothing, and adversarial training with projected gradient descent. Quantifying the perceptual alignment of model gradients via their similarity with the gradients of generative models, we show that off-manifold robustness correlates well with perceptual alignment. Finally, based on the levels of on- and off-manifold robustness, we identify three different regimes of robustness that affect both perceptual alignment and model accuracy: weak robustness, bayes-aligned robustness, and excessive robustness. Code is available at https://github.com/tml-tuebingen/pags.</details>

### White-Box Transformers via Sparse Rate Reduction

**Authors:** Yaodong Yu, Sam Buchanan, Druv Pai, Tianzhe Chu, Ziyang Wu, Shengbang Tong, Benjamin Haeffele, Yi Ma

**<details><summary>Abstract**</summary>

In this paper, we contend that the objective of representation learning is to compress and transform the distribution of the data, say sets of tokens, towards a mixture of low-dimensional Gaussian distributions supported on incoherent subspaces. The quality of the final representation can be measured by a unified objective function called sparse rate reduction. From this perspective, popular deep networks such as transformers can be naturally viewed as realizing iterative schemes to optimize this objective incrementally. Particularly, we show that the standard transformer block can be derived from alternating optimization on complementary parts of this objective: the multi-head self-attention operator can be viewed as a gradient descent step to compress the token sets by minimizing their lossy coding rate, and the subsequent multi-layer perceptron can be viewed as attempting to sparsify the representation of the tokens. This leads to a family of white-box transformer-like deep network architectures which are mathematically fully interpretable. Despite their simplicity, experiments show that these networks indeed learn to optimize the designed objective: they compress and sparsify representations of large-scale real-world vision datasets such as ImageNet, and achieve performance very close to thoroughly engineered transformers such as ViT. Code is at https://github.com/Ma-Lab-Berkeley/CRATE.</details>

### [Oral] Why think step by step? Reasoning emerges from the locality of experience

**Authors:** Ben Prystawski, Michael Li, Noah Goodman

**Oral Presentation:** We, Dec 13, 13:30 -- Oral 4C

**<details><summary>Abstract**</summary>

Humans have a powerful and mysterious capacity to reason. Working through a set of mental steps enables us to make inferences we would not be capable of making directly even though we get no additional data from the world. Similarly, when large language models generate intermediate steps (a chain of thought) before answering a question, they often produce better answers than they would directly. We investigate why and how chain-of-thought reasoning is useful in language models, testing the hypothesis that reasoning is effective when training data consists of overlapping local clusters of variables that influence each other strongly. These training conditions enable the chaining of accurate local inferences to estimate relationships between variables that were not seen together in training. We prove that there will exist a "reasoning gap", where reasoning through intermediate variables reduces bias, for the simple case of an autoregressive density estimator trained on local samples from a chain-structured probabilistic model. We then test our hypothesis experimentally in more complex models, training an autoregressive language model on samples from Bayes nets but only including a subset of variables in each sample. We test language models’ ability to match conditional probabilities with and without intermediate reasoning steps, finding that intermediate steps are only helpful when the training data is locally structured with respect to dependencies between variables. The combination of locally structured observations and reasoning is much more data-efficient than training on all variables. Our results illustrate how the effectiveness of reasoning step by step is rooted in the local statistical structure of the training data.</details>

### Wide Neural Networks as Gaussian Processes: Lessons from Deep Equilibrium Models

**Authors:** Tianxiang Gao, Xiaokai Huo, Hailiang Liu, Hongyang Gao

**<details><summary>Abstract**</summary>

Neural networks with wide layers have attracted significant attention due to their equivalence to Gaussian processes, enabling perfect fitting of training data while maintaining generalization performance, known as benign overfitting. However, existing results mainly focus on shallow or finite-depth networks, necessitating a comprehensive analysis of wide neural networks with infinite-depth layers, such as neural ordinary differential equations (ODEs) and deep equilibrium models (DEQs). In this paper, we specifically investigate the deep equilibrium model (DEQ), an infinite-depth neural network with shared weight matrices across layers. Our analysis reveals that as the width of DEQ layers approaches infinity, it converges to a Gaussian process, establishing what is known as the Neural Network and Gaussian Process (NNGP) correspondence. Remarkably, this convergence holds even when the limits of depth and width are interchanged, which is not observed in typical infinite-depth Multilayer Perceptron (MLP) networks. Furthermore, we demonstrate that the associated Gaussian vector remains non-degenerate for any pairwise distinct input data, ensuring a strictly positive smallest eigenvalue of the corresponding kernel matrix using the NNGP kernel. These findings serve as fundamental elements for studying the training and generalization of DEQs, laying the groundwork for future research in this area.</details>

### Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model

**Authors:** Zirui Liu, Guanchu Wang, Shaochen (Henry) Zhong, Zhaozhuo Xu, Daochen Zha, Ruixiang Tang, Zhimeng Jiang, Kaixiong Zhou, Vipin Chaudhary, Shuai Xu, Xia Hu

**<details><summary>Abstract**</summary>

As the model size grows rapidly, fine-tuning the large pre-trained language model has become increasingly difficult due to its extensive memory usage. Previous works usually focus on reducing the number of trainable parameters in the network. While the model parameters do contribute to memory usage, the primary memory bottleneck during training arises from storing feature maps, also known as activations, as they are crucial for gradient calculation. Notably, machine learning models are typically trained using stochastic gradient descent.We argue that in stochastic optimization, models can handle noisy gradients as long as the gradient estimator is unbiased with reasonable variance.Following this motivation, we propose a new family of unbiased estimators called \sas, for matrix production with reduced variance, which only requires storing the sub-sampled activations for calculating the gradient.Our work provides both theoretical and experimental evidence that, in the context of tuning transformers, our proposed estimators exhibit lower variance compared to existing ones.By replacing the linear operation with our approximated one in transformers, we can achieve up to 2.7X peak memory reduction with almost no accuracy drop and enables up to $6.4\times$ larger batch size.Under the same hardware, \sas enables better down-streaming task performance by applying larger models and/or faster training speed with larger batch sizes.The code is available at https://anonymous.4open.science/r/WTACRS-A5C5/.</details>

### Your representations are in the network: composable and parallel adaptation for large scale models

**Authors:** Yonatan Dukler, Alessandro Achille, Hao Yang, Varsha Vivek, Luca Zancato, Benjamin Bowman, Avinash Ravichandran, Charless Fowlkes, Ashwin Swaminathan, Stefano Soatto

**<details><summary>Abstract**</summary>

We present a framework for transfer learning that efficiently adapts a large base-model by learning lightweight cross-attention modules attached to its intermediate activations.We name our approach InCA (Introspective-Cross-Attention) and show that it can efficiently survey a network’s representations and identify strong performing adapter models for a downstream task.During training, InCA enables training numerous adapters efficiently and in parallel, isolated from the frozen base model. On the ViT-L/16 architecture, our experiments show that a single adapter, 1.3% of the full model, is able to reach full fine-tuning accuracy on average across 11 challenging downstream classification tasks.Compared with other forms of parameter-efficient adaptation, the isolated nature of the InCA adaptation is computationally desirable for large-scale models. For instance, we adapt ViT-G/14 (1.8B+ parameters) quickly with 20+ adapters in parallel on a single V100 GPU (76% GPU memory reduction) and exhaustively identify its most useful representations.We further demonstrate how the adapters learned by InCA can be incrementally modified or combined for flexible learning scenarios and our approach achieves state of the art performance on the ImageNet-to-Sketch multi-task benchmark.</details>

### ZipLM: Inference-Aware Structured Pruning of Language Models

**Authors:** Eldar Kurtić, Elias Frantar, Dan Alistarh

**<details><summary>Abstract**</summary>

The breakthrough performance of large language models (LLMs) comes with major computational footprints and high deployment costs. In this paper, we progress towards resolving this problem by proposing a novel structured compression approach for LLMs, called ZipLM. ZipLM achieves state-of-the-art accuracy-vs-speedup, while matching a set of desired target runtime speedups in any given inference environment. Specifically, given a model, a dataset, an inference environment, as well as a set of speedup targets, ZipLM iteratively identifies and removes components with the worst loss-runtime trade-off. Unlike prior methods that specialize in either the *post-training/one-shot* or the *gradual compression* setting, and only for specific families of models such as BERT (*encoder*) or GPT (*decoder*), ZipLM produces state-of-the-art compressed models across all these settings. Furthermore, ZipLM achieves superior results for a fraction of the computational cost relative to prior distillation and pruning techniques, making it a cost-effective approach for generating an entire family of smaller, faster, and highly accurate models, guaranteed to meet the desired inference specifications. In particular, ZipLM outperforms all prior BERT-base distillation and pruning techniques, such as CoFi, MiniLM, and TinyBERT. Moreover, it matches the performance of the heavily optimized MobileBERT model, obtained via extensive architecture search, by simply pruning the baseline BERT-large model. When compressing GPT2, ZipLM outperforms DistilGPT2 while being 60\% smaller and 30\% faster. Our code is available at: https://github.com/IST-DASLab/ZipLM.</details>

