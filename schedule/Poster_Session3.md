## Poster Session 3: Wednesday, Dec 13, 08:45 CT

### $L_2$-Uniform Stability of Randomized Learning Algorithms: Sharper Generalization Bounds and Confidence Boosting

**Authors:** Xiaotong Yuan, Ping Li

**<details><summary>Abstract**</summary>

Exponential generalization bounds with near-optimal rates have recently been established for uniformly stable algorithms~\citep{feldman2019high,bousquet2020sharper}. We seek to extend these best known high probability bounds from deterministic learning algorithms to the regime of randomized learning. One simple approach for achieving this goal is to define the stability for the expectation over the algorithm's randomness, which may result in sharper parameter but only leads to guarantees regarding the on-average generalization error. Another natural option is to consider the stability conditioned on the algorithm's randomness, which is way more stringent but may lead to generalization with high probability jointly over the randomness of sample and algorithm. The present paper addresses such a tension between these two alternatives and makes progress towards relaxing it inside a classic framework of confidence-boosting. To this end, we first introduce a novel concept of $L_2$-uniform stability that holds uniformly over data but in second-moment over the algorithm's randomness. Then as a core contribution of this work, we prove a strong exponential bound on the first-moment of generalization error under the notion of $L_2$-uniform stability. As an interesting consequence of the bound, we show that a bagging-based meta algorithm leads to near-optimal generalization with high probability jointly over the randomness of data and algorithm. We further substantialize these generic results to stochastic gradient descent (SGD) to derive sharper exponential bounds for convex or non-convex optimization with natural time-decaying learning rates, which have not been possible to prove with the existing stability-based generalization guarantees.</details>

### $\texttt{TACO}$: Temporal Latent Action-Driven Contrastive Loss for Visual Reinforcement Learning

**Authors:** Ruijie Zheng, Xiyao Wang, Yanchao Sun, Shuang Ma, Jieyu Zhao, Huazhe Xu, Hal Daumé III, Furong Huang

**<details><summary>Abstract**</summary>

Despite recent progress in reinforcement learning (RL) from raw pixel data, sample inefficiency continues to present a substantial obstacle. Prior works have attempted to address this challenge by creating self-supervised auxiliary tasks, aiming to enrich the agent's learned representations with control-relevant information for future state prediction.However, these objectives are often insufficient to learn representations that can represent the optimal policy or value function, and they often consider tasks with small, abstract discrete action spaces and thus overlook the importance of action representation learning in continuous control.In this paper, we introduce $\texttt{TACO}$: $\textbf{T}$emporal $\textbf{A}$ction-driven $\textbf{CO}$ntrastive Learning, a simple yet powerful temporal contrastive learning approach that facilitates the concurrent acquisition of latent state and action representations for agents. $\texttt{TACO}$ simultaneously learns a state and an action representation by optimizing the mutual information between representations of current states paired with action sequences and representations of the corresponding future states. Theoretically, $\texttt{TACO}$ can be shown to learn state and action representations that encompass sufficient information for control, thereby improving sample efficiency.For online RL, $\texttt{TACO}$ achieves 40% performance boost after one million environment interaction steps on average across nine challenging visual continuous control tasks from Deepmind Control Suite. In addition, we show that $\texttt{TACO}$ can also serve as a plug-and-play module adding to existing offline visual RL methods to establish the new state-of-the-art performance for offline visual RL across offline datasets with varying quality.</details>

### $\varepsilon$-fractional core stability in Hedonic Games.

**Authors:** Simone Fioravanti, Michele Flammini, Bojana Kodric, Giovanna Varricchio

**<details><summary>Abstract**</summary>

Hedonic Games (HGs) are a classical framework modeling coalition formation of strategic agents guided by their individual preferences. According to these preferences, it is desirable that a coalition structure (i.e. a partition of agents into coalitions) satisfies some form of stability. The most well-known and natural of such notions is arguably core-stability. Informally, a partition is core-stable if no subset of agents would like to deviate by regrouping in a so-called core-blocking coalition. Unfortunately, core-stable partitions seldom exist and even when they do, it is often computationally intractable to find one. To circumvent these problems, we propose the notion of $\varepsilon$-fractional core-stability, where at most an $\varepsilon$-fraction of all possible coalitions is allowed to core-block. It turns out that such a relaxation may guarantee both existence and polynomial-time computation. Specifically, we design efficient algorithms returning an $\varepsilon$-fractional core-stable partition, with $\varepsilon$ exponentially decreasing in the number of agents, for two fundamental classes of HGs: Simple Fractional and Anonymous. From a probabilistic point of view, being the definition of $\varepsilon$-fractional core equivalent to requiring that uniformly sampled coalitions core-block with probability lower than $\varepsilon$, we further extend the definition to handle more complex sampling distributions. Along this line, when valuations have to be learned from samples in a PAC-learning fashion, we give positive and negative results on which distributions allow the efficient computation of outcomes that are $\varepsilon$-fractional core-stable with arbitrarily high confidence.</details>

### 3D-IntPhys: Towards More Generalized 3D-grounded Visual Intuitive Physics under Challenging Scenes

**Authors:** Haotian Xue, Antonio Torralba, Josh Tenenbaum, Dan Yamins, Yunzhu Li, Hsiao-Yu Tung

**<details><summary>Abstract**</summary>

Given a visual scene, humans have strong intuitions about how a scene can evolve over time under given actions. The intuition, often termed visual intuitive physics, is a critical ability that allows us to make effective plans to manipulate the scene to achieve desired outcomes without relying on extensive trial and error. In this paper, we present a framework capable of learning 3D-grounded visual intuitive physics models from videos of complex scenes with fluids. Our method is composed of a conditional Neural Radiance Field (NeRF)-style visual frontend and a 3D point-based dynamics prediction backend, using which we can impose strong relational and structural inductive bias to capture the structure of the underlying environment. Unlike existing intuitive point-based dynamics works that rely on the supervision of dense point trajectory from simulators, we relax the requirements and only assume access to multi-view RGB images and (imperfect) instance masks acquired using color prior. This enables the proposed model to handle scenarios where accurate point estimation and tracking are hard or impossible. We generate datasets including three challenging scenarios involving fluid, granular materials, and rigid objects in the simulation. The datasets do not include any dense particle information so most previous 3D-based intuitive physics pipelines can barely deal with that. We show our model can make long-horizon future predictions by learning from raw images and significantly outperforms models that do not employ an explicit 3D representation space. We also show that once trained, our model can achieve strong generalization in complex scenarios under extrapolate settings.</details>

### [Spotlight] 4D Panoptic Scene Graph Generation

**Authors:** Jingkang Yang, Jun CEN, WENXUAN PENG, Shuai Liu, Fangzhou Hong, Xiangtai Li, Kaiyang Zhou, Qifeng Chen, Ziwei Liu

**<details><summary>Abstract**</summary>

We are living in a three-dimensional space while moving forward through a fourth dimension: time. To allow artificial intelligence to develop a comprehensive understanding of such a 4D environment, we introduce **4D Panoptic Scene Graph (PSG-4D)**, a new representation that bridges the raw visual data perceived in a dynamic 4D world and high-level visual understanding. Specifically, PSG-4D abstracts rich 4D sensory data into nodes, which represent entities with precise location and status information, and edges, which capture the temporal relations. To facilitate research in this new area, we build a richly annotated PSG-4D dataset consisting of 3K RGB-D videos with a total of 1M frames, each of which is labeled with 4D panoptic segmentation masks as well as fine-grained, dynamic scene graphs. To solve PSG-4D, we propose PSG4DFormer, a Transformer-based model that can predict panoptic segmentation masks, track masks along the time axis, and generate the corresponding scene graphs via a relation component. Extensive experiments on the new dataset show that our method can serve as a strong baseline for future research on PSG-4D. In the end, we provide a real-world application example to demonstrate how we can achieve dynamic scene understanding by integrating a large language model into our PSG-4D system.</details>

### A Dual-Stream Neural Network Explains the Functional Segregation of Dorsal and Ventral Visual Pathways in Human Brains

**Authors:** Minkyu Choi, Kuan Han, Xiaokai Wang, Yizhen Zhang, Zhongming Liu

**<details><summary>Abstract**</summary>

The human visual system uses two parallel pathways for spatial processing and object recognition. In contrast, computer vision systems tend to use a single feedforward pathway, rendering them less robust, adaptive, or efficient than human vision. To bridge this gap, we developed a dual-stream vision model inspired by the human eyes and brain. At the input level, the model samples two complementary visual patterns to mimic how the human eyes use magnocellular and parvocellular retinal ganglion cells to separate retinal inputs to the brain. At the backend, the model processes the separate input patterns through two branches of convolutional neural networks (CNN) to mimic how the human brain uses the dorsal and ventral cortical pathways for parallel visual processing. The first branch (WhereCNN) samples a global view to learn spatial attention and control eye movements. The second branch (WhatCNN) samples a local view to represent the object around the fixation. Over time, the two branches interact recurrently to build a scene representation from moving fixations. We compared this model with the human brains processing the same movie and evaluated their functional alignment by linear transformation. The WhereCNN and WhatCNN branches were found to differentially match the dorsal and ventral pathways of the visual cortex, respectively, primarily due to their different learning objectives, rather than their distinctions in retinal sampling or sensitivity to attention-driven eye movements. These model-based results lead us to speculate that the distinct responses and representations of the ventral and dorsal streams are more influenced by their distinct goals in visual attention and object recognition than by their specific bias or selectivity in retinal inputs. This dual-stream model takes a further step in brain-inspired computer vision, enabling parallel neural networks to actively explore and understand the visual surroundings.</details>

### A Heat Diffusion Perspective on Geodesic Preserving Dimensionality Reduction

**Authors:** Guillaume Huguet, Alexander Tong, Edward De Brouwer, Yanlei Zhang, Guy Wolf, Ian Adelstein, Smita Krishnaswamy

**<details><summary>Abstract**</summary>

Diffusion-based manifold learning methods have proven useful in representation learning and dimensionality reduction of modern high dimensional, high throughput, noisy datasets. Such datasets are especially present in fields like biology and physics. While it is thought that these methods preserve underlying manifold structure of data by learning a proxy for geodesic distances, no specific theoretical links have been established. Here, we establish such a link via results in Riemannian geometry explicitly connecting heat diffusion to manifold distances. In this process, we also formulate a more general heat kernel based manifold embedding method that we call heat geodesic embeddings. This novel perspective makes clearer the choices available in manifold learning and denoising. Results show that our method outperforms existing state of the art in preserving ground truth manifold distances,  and preserving cluster structure in toy datasets. We also showcase our method on single cell RNA-sequencing datasets with both continuum and cluster structure, where our method enables interpolation of withheld timepoints of data. Finally, we show that parameters of our more general method can be configured to give results similar to PHATE (a state-of-the-art diffusion based manifold learning method) as well as SNE (an attraction/repulsion neighborhood based method that forms the basis of t-SNE).</details>

### [Spotlight] A Holistic Approach to Unifying Automatic Concept Extraction and Concept Importance Estimation

**Authors:** Thomas FEL, Victor Boutin, Louis Béthune, Remi Cadene, Mazda Moayeri, Léo Andéol, Mathieu Chalvidal, Thomas Serre

**<details><summary>Abstract**</summary>

In recent years, concept-based approaches have emerged as some of the most promising explainability methods to help us interpret the decisions of Artificial Neural Networks (ANNs). These methods seek to discover intelligible visual ``concepts'' buried within the complex patterns of ANN activations in two key steps: (1) concept extraction followed by (2) importance estimation. While these two steps are shared across methods, they all differ in their specific implementations. Here, we introduce a unifying theoretical framework that recast the first step -- concept extraction problem -- as a special case of **dictionary learning**, and we formalize the second step -- concept importance estimation -- as a more general form of **attribution method**.This framework offers several advantages as it allows us: (i) to propose new evaluation metrics for comparing different concept extraction approaches; (ii) to leverage modern attribution methods and evaluation metrics to extend and systematically evaluate state-of-the-art concept-based approaches and importance estimation techniques; (iii)  to derive theoretical guarantees regarding the optimality of such methods. We further leverage our framework to try to tackle a crucial question in explainability: how to *efficiently* identify clusters of data points that are classified based on a similar shared strategy.To illustrate these findings and to highlight the main strategies of a model, we introduce a visual representation called the strategic cluster graph. Finally, we present Lens, a dedicated website that offers a complete compilation of these visualizations for all classes of the ImageNet dataset.</details>

### A Long $N$-step Surrogate Stage Reward for Deep Reinforcement Learning

**Authors:** Junmin Zhong, Ruofan Wu, Jennie Si

**<details><summary>Abstract**</summary>

We introduce a new stage reward estimator  named the long $N$-step surrogate stage (LNSS) reward for deep reinforcement learning (RL). It aims at mitigating the high variance problem, which has shown impeding successful convergence of learning, hurting task performance, and hindering applications of deep RL in continuous control problems. In this paper we show that LNSS, which utilizes a long reward trajectory of  rewards of future steps, provides consistent performance improvement measured by average reward, convergence speed, learning success rate,and variance reduction in $Q$ values and rewards.  Our evaluations are based on a variety of environments in DeepMind Control Suite and OpenAI Gym  by using  LNSS in baseline deep RL algorithms such as DDPG, D4PG, and TD3. We show  that LNSS reward has enabled good results that have been challenging to obtain by deep RL previously. Our analysis also shows that  LNSS exponentially reduces the upper bound on the variances of $Q$ values from respective single-step methods.</details>

### A Metadata-Driven Approach to Understand Graph Neural Networks

**Authors:** Ting Wei Li, Qiaozhu Mei, Jiaqi Ma

**<details><summary>Abstract**</summary>

Graph Neural Networks (GNNs) have achieved remarkable success in various applications, but their performance can be sensitive to specific data properties of the graph datasets they operate on. Current literature on understanding the limitations of GNNs has primarily employed a \emph{model-driven} approach that leverage heuristics and domain knowledge from network science or graph theory to model the GNN behaviors, which is time-consuming and highly subjective. In this work, we propose a \emph{metadata-driven} approach to analyze the sensitivity of GNNs to graph data properties, motivated by the increasing availability of graph learning benchmarks. We perform a multivariate sparse regression analysis on the metadata derived from benchmarking GNN performance across diverse datasets, yielding a set of salient data properties. To validate the effectiveness of our data-driven approach, we focus on one identified data property, the degree distribution, and investigate how this property influences GNN performance through theoretical analysis and controlled experiments. Our theoretical findings reveal that datasets with more balanced degree distribution exhibit better linear separability of node representations, thus leading to better GNN performance. We also conduct controlled experiments using synthetic datasets with varying degree distributions, and the results align well with our theoretical findings. Collectively, both the theoretical analysis and controlled experiments verify that the proposed metadata-driven approach is effective in identifying critical data properties for GNNs.</details>

### [Spotlight] A One-Size-Fits-All Approach to Improving Randomness in Paper Assignment

**Authors:** Yixuan Xu, Steven Jecmen, Zimeng Song, Fei Fang

**<details><summary>Abstract**</summary>

The assignment of papers to reviewers is a crucial part of the peer review processes of large publication venues, where organizers (e.g., conference program chairs) rely on algorithms to perform automated paper assignment. As such, a major challenge for the organizers of these processes is to specify paper assignment algorithms that find appropriate assignments with respect to various desiderata. Although the main objective when choosing a good paper assignment is to maximize the expertise of each reviewer for their assigned papers, several other considerations make introducing randomization into the paper assignment desirable: robustness to malicious behavior, the ability to evaluate alternative paper assignments, reviewer diversity, and reviewer anonymity. However, it is unclear in what way one should randomize the paper assignment in order to best satisfy all of these considerations simultaneously. In this work, we present a practical, one-size-fits-all method for randomized paper assignment intended to perform well across different motivations for randomness. We show theoretically and experimentally that our method outperforms currently-deployed methods for randomized paper assignment on several intuitive randomness metrics, demonstrating that the randomized assignments produced by our method are general-purpose.</details>

### [Spotlight] A Privacy-Friendly Approach to Data Valuation

**Authors:** Jiachen (Tianhao) Wang, Yuqing Zhu, Yu-Xiang Wang, Ruoxi Jia, Prateek Mittal

**<details><summary>Abstract**</summary>

Data valuation, a growing field that aims at quantifying the usefulness of individual data sources for training machine learning (ML) models, faces notable yet often overlooked privacy challenges. This paper studies these challenges with a focus on KNN-Shapley, one of the most practical data valuation methods nowadays. We first emphasize the inherent privacy risks of KNN-Shapley, and demonstrate the significant technical challenges in adapting KNN-Shapley to accommodate differential privacy (DP). To overcome these challenges, we introduce TKNN-Shapley, a refined variant of KNN-Shapley that is privacy-friendly, allowing for straightforward modifications to incorporate DP guarantee (DP-TKNN-Shapley). We show that DP-TKNN-Shapley has several advantages and offers a superior privacy-utility tradeoff compared to naively privatized KNN-Shapley. Moreover, even non-private TKNN-Shapley matches KNN-Shapley's performance in discerning data quality. Overall, our findings suggest that TKNN-Shapley is a promising alternative to KNN-Shapley, particularly for real-world applications involving sensitive data.</details>

### A Recurrent Neural Circuit Mechanism of Temporal-scaling Equivariant Representation

**Authors:** Junfeng Zuo, Xiao Liu, Ying Nian Wu, Si Wu, Wenhao Zhang

**<details><summary>Abstract**</summary>

Time perception is critical in our daily life. An important feature of time perception is temporal scaling (TS): the ability to generate temporal sequences (e.g., motor actions) at different speeds. However, it is largely unknown about the math principle underlying temporal scaling in recurrent circuits in the brain. To shed insight, the present study investigates the temporal scaling from the Lie group point of view. We propose a canonical nonlinear recurrent circuit dynamics, modeled as a continuous attractor network, whose neuronal population responses embed a temporal sequence that is TS equivariant. Furthermore, we found the TS group operators can be explicitly represented by a control input fed into the recurrent circuit, where the input gain determines the temporal scaling factor (group parameter), and the spatial offset between the control input and network state emerges the generator.  The neuronal responses in the recurrent circuit are also consistent with experimental findings. We illustrated that the recurrent circuit can drive a feedforward circuit to generate complex temporal sequences with different time scales, even in the case of negative time scaling (''time reversal'').  Our work for the first time analytically links the abstract temporal scaling group and concrete neural circuit dynamics.</details>

### A Simple Yet Effective Strategy to Robustify the Meta Learning Paradigm

**Authors:** Qi Wang, Yiqin Lv, yanghe feng, Zheng Xie, Jincai Huang

**<details><summary>Abstract**</summary>

Meta learning is a promising paradigm to enable skill transfer across tasks.Most previous methods employ the empirical risk minimization principle in optimization.However, the resulting worst fast adaptation to a subset of tasks can be catastrophic in risk-sensitive scenarios.To robustify fast adaptation, this paper optimizes meta learning pipelines from a distributionally robust perspective and meta trains models with the measure of tail task risk.We take the two-stage strategy as heuristics to solve the robust meta learning problem, controlling the worst fast adaptation cases at a certain probabilistic level. Experimental results show that our simple method can improve the robustness of meta learning to task distributions and reduce the conditional expectation of the worst fast adaptation risk.</details>

### A Smooth Binary Mechanism for Efficient Private Continual Observation

**Authors:** Joel Daniel Andersson, Rasmus Pagh

**<details><summary>Abstract**</summary>

In privacy under continual observation we study how to release differentially private estimates based on a dataset that evolves over time. The problem of releasing private prefix sums of $x_1, x_2, x_3,\dots\in${$0,1$} (where the value of each $x_i$ is to be private) is particularly well-studied, and a generalized form is used in state-of-the-art methods for private stochastic gradient descent (SGD).The seminal binary mechanism privately releases the first $t$ prefix sums with noise of variance polylogarithmic in $t$. Recently, Henzinger et al. and Denisov et al. showed that it is possible to improve on the binary mechanism in two ways: The variance of the noise can be reduced by a (large) constant factor, and also made more even across time steps. However, their algorithms for generating the noise distribution are not as efficient as one would like in terms of computation time and (in particular) space.We address the efficiency problem by presenting a simple alternative to the binary mechanism in which 1) generating the noise takes constant average time per value, 2) the variance is reduced by a factor about 4 compared to the binary mechanism, and 3) the noise distribution at each step is identical. Empirically, a simple Python implementation of our approach outperforms the running time of the approach of Henzinger et al., as well as an attempt to improve their algorithm using high-performance algorithms for multiplication with Toeplitz matrices.</details>

### [Spotlight] A Spectral Algorithm for List-Decodable Covariance Estimation in Relative Frobenius Norm

**Authors:** Ilias Diakonikolas, Daniel Kane, Jasper Lee, Ankit Pensia, Thanasis Pittas

**<details><summary>Abstract**</summary>

We study the problem of list-decodable Gaussian covariance estimation. Given a multiset $T$ of $n$ points in $\mathbb{R}^d$ such that an unknown $\alpha<1/2$ fraction of points in $T$ are i.i.d. samples from an unknown Gaussian $\mathcal{N}(\mu, \Sigma)$, the goal is to output a list of $O(1/\alpha)$ hypotheses at least one of which is close to $\Sigma$ in relative Frobenius norm. Our main result is a $\mathrm{poly}(d,1/\alpha)$ sample and time algorithm for this task that guarantees relative Frobenius norm error of $\mathrm{poly}(1/\alpha)$. Importantly, our algorithm relies purely on spectral techniques. As a corollary, we obtain an efficient spectral algorithm for robust partial clustering of Gaussian mixture models (GMMs) --- a key ingredient in the recent work of [BakDJKKV22] on robustly learning arbitrary GMMs. Combined with the other components of [BakDJKKV22], our new method yields the first Sum-of-Squares-free algorithm for robustly learning GMMs, resolving an open problem proposed by Vempala and Kothari. At the technical level, we develop a novel multi-filtering method for list-decodable covariance estimation that may be useful in other settings.</details>

### A Unified Approach to Count-Based Weakly Supervised Learning

**Authors:** Vinay Shukla, Zhe Zeng, Kareem Ahmed, Guy Van den Broeck

**<details><summary>Abstract**</summary>

High-quality labels are often very scarce, whereas unlabeled data with inferred weak labels occurs more naturally. In many cases, these weak labels dictate the frequency of each respective class over a set of instances. In this paper, we develop a unified approach to learning from such weakly-labeled data, which we call *count-based weakly-supervised learning*. At the heart of our approach is the ability to compute the probability of exactly $k$ out of $n$ outputs being set to true. This computation is differentiable, exact, and efficient. Building upon the previous computation, we derive a *count loss* penalizing the model for deviations in its distribution from an arithmetic constraint defined over label counts.</details>

### A Unified Approach to Domain Incremental Learning with Memory: Theory and Algorithm

**Authors:** Haizhou Shi, Hao Wang

**<details><summary>Abstract**</summary>

Domain incremental learning aims to adapt to a sequence of domains with access to only a small subset of data (i.e., memory) from previous domains. Various methods have been proposed for this problem, but it is still unclear how they are related and when practitioners should choose one method over another. In response, we propose a unified framework, dubbed Unified Domain Incremental Learning (UDIL), for domain incremental learning with memory. Our UDIL **unifies** various existing methods, and our theoretical analysis shows that UDIL always achieves a tighter generalization error bound compared to these methods. The key insight is that different existing methods correspond to our bound with different **fixed** coefficients; based on insights from this unification, our UDIL allows **adaptive** coefficients during training, thereby always achieving the tightest bound. Empirical results show that our UDIL outperforms the state-of-the-art domain incremental learning methods on both synthetic and real-world datasets. Code will be available at https://github.com/Wang-ML-Lab/unified-continual-learning.</details>

### A graphon-signal analysis of graph neural networks

**Authors:** Ron Levie

**<details><summary>Abstract**</summary>

We present an approach for analyzing message passing graph neural networks (MPNNs) based on an extension of graphon analysis to a so called graphon-signal analysis. A MPNN is a function that takes a graph and a signal on the graph (a graph-signal) and returns some value. Since the input space of MPNNs is non-Euclidean, i.e., graphs can be of any size and topology, properties such as generalization are less well understood for MPNNs than for Euclidean neural networks. We claim that one important missing ingredient in past work is a meaningful notion of graph-signal similarity measure, that endows the space of inputs to MPNNs with a regular structure. We present such a similarity measure, called the graphon-signal cut distance, which makes the space of all graph-signals a dense subset of a compact metric space -- the graphon-signal space. Informally, two deterministic graph-signals are close in cut-distance if they ``look like'' they were sampled from the same random graph-signal model. Hence, our cut distance is a natural notion of graph-signal similarity, which allows comparing any pair of graph-signals of any size and topology. We prove that MPNNs are Lipschitz continuous functions over the graphon-signal metric space. We then give two applications of this result: 1) a generalization bound for MPNNs, and, 2) the stability of MPNNs to subsampling of graph-signals. Our results apply to any regular enough MPNN on any distribution of graph-signals, making the analysis rather universal.</details>

### ALIM: Adjusting Label Importance Mechanism for Noisy Partial Label Learning

**Authors:** Mingyu Xu, Zheng Lian, Lei Feng, Bin Liu, Jianhua Tao

**<details><summary>Abstract**</summary>

Noisy partial label learning (noisy PLL) is an important branch of weakly supervised learning. Unlike PLL where the ground-truth label must conceal in the candidate label set, noisy PLL relaxes this constraint and allows the ground-truth label may not be in the candidate label set. To address this challenging problem, most of the existing works attempt to detect noisy samples and estimate the ground-truth label for each noisy sample. However, detection errors are unavoidable. These errors can accumulate during training and continuously affect model optimization. To this end, we propose a novel framework for noisy PLL with theoretical interpretations, called ``Adjusting Label Importance Mechanism (ALIM)''. It aims to reduce the negative impact of detection errors by trading off the initial candidate set and model outputs. ALIM is a plug-in strategy that can be integrated with existing PLL approaches. Experimental results on multiple benchmark datasets demonstrate that our method can achieve state-of-the-art performance on noisy PLL. Our code is available at: https://github.com/zeroQiaoba/ALIM.</details>

### ASPEN: Breaking Operator Barriers for Efficient Parallelization of Deep Neural Networks

**Authors:** Jongseok Park, Kyungmin Bin, Gibum Park, Sangtae Ha, Kyunghan Lee

**<details><summary>Abstract**</summary>

Modern Deep Neural Network (DNN) frameworks use tensor operators as the main building blocks of DNNs. However, we observe that operator-based construction of DNNs incurs significant drawbacks in parallelism in the form of synchronization barriers. Synchronization barriers of operators confine the scope of parallel computation to each operator and obscure the rich parallel computation opportunities that exist across operators. To this end, we present ASPEN, a novel parallel computation solution for DNNs that achieves fine-grained dynamic execution of DNNs, which (1) removes the operator barriers and expresses DNNs in dataflow graphs of fine-grained tiles to expose the parallel computation opportunities across operators, and (2) exploits these opportunities by dynamically locating and scheduling them in runtime. This novel approach of ASPEN enables opportunistic parallelism, a new class of parallelism for DNNs that is unavailable in the existing operator-based approaches. ASPEN also achieves high resource utilization and memory reuse by letting each resource asynchronously traverse depthwise in the DNN graph to its full computing potential. We provide challenges and solutions to our approach and show that our proof-of-concept implementation of ASPEN on CPU shows exceptional performance, outperforming state-of-the-art inference systems of TorchScript and TVM by up to 3.2$\times$ and 4.3$\times$, respectively.</details>

### ATMAN: Understanding Transformer Predictions Through Memory Efficient Attention Manipulation

**Authors:** Björn Deiseroth, Mayukh Deb, Samuel Weinbach, Manuel Brack, Patrick Schramowski, Kristian Kersting

**<details><summary>Abstract**</summary>

Generative transformer models have become increasingly complex, with large numbers of parameters and the ability to process multiple input modalities. Current methods for explaining their predictions are resource-intensive. Most crucially, they require prohibitively large amounts of additional memory, since they rely on backpropagation which allocates almost twice as much GPU memory as the forward pass. This makes it difficult, if not impossible, to use explanations in production. We present AtMan that provides explanations of generative transformer models at almost no extra cost. Specifically, AtMan is a modality-agnostic perturbation method that manipulates the attention mechanisms of transformers to produce relevance maps for the input with respect to the output prediction. Instead of using backpropagation, AtMan applies a parallelizable token-based search method relying on cosine similarity neighborhood in the embedding space. Our exhaustive experiments on text and image-text benchmarks demonstrate that AtMan outperforms current state-of-the-art gradient-based methods on several metrics while being computationally efficient. As such, AtMan is suitable for use in large model inference deployments.</details>

### AUDIT: Audio Editing by Following Instructions with Latent Diffusion Models

**Authors:** Yuancheng Wang, Zeqian Ju, Xu Tan, Lei He, Zhizheng Wu, Jiang Bian, sheng zhao

**<details><summary>Abstract**</summary>

Audio editing is applicable for various purposes, such as adding background sound effects, replacing a musical instrument, and repairing damaged audio. Recently, some diffusion-based methods achieved zero-shot audio editing by using a diffusion and denoising process conditioned on the text description of the output audio. However, these methods still have some problems: 1) they have not been trained on editing tasks and cannot ensure good editing effects; 2) they can erroneously modify audio segments that do not require editing; 3) they need a complete description of the output audio, which is not always available or necessary in practical scenarios. In this work, we propose AUDIT, an instruction-guided audio editing model based on latent diffusion models. Specifically, 	extbf{AUDIT} has three main design features: 1) we construct triplet training data (instruction, input audio, output audio) for different audio editing tasks and train a diffusion model using instruction and input (to be edited) audio as conditions and generating output (edited) audio; 2) it can automatically learn to only modify segments that need to be edited by comparing the difference between the input and output audio; 3) it only needs edit instructions instead of full target audio descriptions as text input. AUDIT achieves state-of-the-art results in both objective and subjective metrics for several audio editing tasks (e.g., adding, dropping, replacement, inpainting, super-resolution). Demo samples are available at https://audit-demopage.github.io/.</details>

### Accelerated Training via Incrementally Growing Neural Networks using Variance Transfer and Learning Rate Adaptation

**Authors:** Xin Yuan, Pedro Savarese, Michael Maire

**<details><summary>Abstract**</summary>

We develop an approach to efficiently grow neural networks, within which parameterization and optimization strategies are designed by considering their effects on the training dynamics.  Unlike existing growing methods, which follow simple replication heuristics or utilize auxiliary gradient-based local optimization, we craft a parameterization scheme which dynamically stabilizes weight, activation, and gradient scaling as the architecture evolves, and maintains the inference functionality of the network.  To address the optimization difficulty resulting from imbalanced training effort distributed to subnetworks fading in at different growth phases, we propose a learning rate adaption mechanism that rebalances the gradient contribution of these separate subcomponents.  Experiments show that our method achieves comparable or better accuracy than training large fixed-size models, while saving a substantial portion of the original training computation budget.  We demonstrate that these gains translate into real wall-clock training speedups.</details>

### Achieving $\mathcal{O}(\epsilon^{-1.5})$ Complexity in Hessian/Jacobian-free Stochastic Bilevel Optimization

**Authors:** Yifan Yang, Peiyao Xiao, Kaiyi Ji

**<details><summary>Abstract**</summary>

In this paper, we revisit the bilevel optimization problem, in which the upper-level objective function is generally nonconvex and the lower-level objective function is strongly convex. Although this type of problem has been studied extensively, it still remains an open question how to achieve an $\mathcal{O}(\epsilon^{-1.5})$ sample complexity of $\mathcal{O}(\epsilon^{-1.5})$ in Hessian/Jacobian-free stochastic bilevel optimization without any second-order derivative computation. To fill this gap, we propose a novel Hessian/Jacobian-free bilevel optimizer named FdeHBO, which features a simple fully single-loop structure, a projection-aided finite-difference Hessian/Jacobian-vector approximation, and momentum-based updates. Theoretically, we show that FdeHBO requires $\mathcal{O}(\epsilon^{-1.5})$ iterations (each using $\mathcal{O}(1)$ samples and only first-order gradient information) to find an $\epsilon$-accurate stationary point. As far as we know, this is the first Hessian/Jacobian-free method with an $\mathcal{O}(\epsilon^{-1.5})$ sample complexity for nonconvex-strongly-convex stochastic bilevel optimization.</details>

### Active Learning-Based Species Range Estimation

**Authors:** Christian Lange, Elijah Cole, Grant Horn, Oisin Mac Aodha

**<details><summary>Abstract**</summary>

We propose a new active learning approach for efficiently estimating the geographic range of a species from a limited number of on the ground observations. We model the range of an unmapped species of interest as the weighted combination of estimated ranges obtained from a set of different species. We show that it is possible to generate this candidate set of ranges by using models that have been trained on large weakly supervised community collected observation data. From this, we develop a new active querying approach that sequentially selects geographic locations to visit that best reduce our uncertainty over an unmapped species’ range. We conduct a detailed evaluation of our approach and compare it to existing active learning methods using an evaluation dataset containing expert-derived ranges for one thousand species. Our results demonstrate that our method outperforms alternative active learning methods and approaches the performance of end-to-end trained models, even when only using a fraction of the data. This highlights the utility of active learning via transfer learned spatial representations for species range estimation. It also emphasizes the value of leveraging emerging large-scale crowdsourced datasets, not only for modeling a species' range, but also for actively discovering them.</details>

### [Spotlight] Adaptive Data Analysis in a Balanced Adversarial Model

**Authors:** Kobbi Nissim, Uri Stemmer, Eliad Tsfadia

**<details><summary>Abstract**</summary>

In adaptive data analysis, a mechanism gets $n$ i.i.d. samples from an unknown distribution $\cal{D}$, andis required to provide accurate estimations to a sequence of adaptively chosen statistical queries with respect to $\cal{D}$.Hardt and Ullman (FOCS 2014) and Steinke and Ullman (COLT 2015) showed that in general, it is computationally hard to answer more than $\Theta(n^2)$ adaptive queries, assuming the existence of one-way functions. However, these negative results strongly rely on an adversarial model that significantly advantages the adversarial analyst over the mechanism, as the analyst, who chooses the adaptive queries, also chooses the underlying distribution $\cal{D}$. This imbalance raises questions with respect to the applicability of the obtained hardness results -- an analyst who has complete knowledge of the underlying distribution $\cal{D}$ would have little need, if at all, to issue statistical queries to a mechanism which only holds a finite number of samples from $\cal{D}$.We consider more restricted adversaries, called \emph{balanced}, where each such adversary consists of two separated algorithms: The \emph{sampler} who is the entity that chooses the distribution and provides the samples to the mechanism, and the \emph{analyst} who chooses the adaptive queries, but has no prior knowledge of the underlying distribution (and hence has no a priori advantage with respect to the mechanism). We improve the quality of previous lower bounds by revisiting them using an efficient \emph{balanced} adversary, under standard public-key cryptography assumptions. We show that these stronger hardness assumptions are unavoidable in the sense that any computationally bounded \emph{balanced} adversary that has the structure of all known attacks, implies the existence of public-key cryptography.</details>

### Adaptive Principal Component Regression with Applications to Panel Data

**Authors:** Anish Agarwal, Keegan Harris, Justin Whitehouse, Steven Wu

**<details><summary>Abstract**</summary>

Principal component regression (PCR) is a popular technique for fixed-design error-in-variables regression, a generalization of the linear regression setting in which the observed covariates are corrupted with random noise. We provide the first time-uniform finite sample guarantees for online (regularized) PCR whenever data is collected adaptively. Since the proof techniques for PCR in the fixed design setting do not readily extend to the online setting, our results rely on adapting tools from modern martingale concentration to the error-in-variables setting. As an application of our bounds, we provide a framework for counterfactual estimation of unit-specific treatment effects in panel data settings when interventions are assigned adaptively. Our framework may be thought of as a generalization of the synthetic interventions framework where data is collected via an adaptive intervention assignment policy.</details>

### Adaptive Privacy Composition for Accuracy-first Mechanisms

**Authors:** Ryan Rogers, Gennady Samorodnitsk, Steven Wu, Aaditya Ramdas

**<details><summary>Abstract**</summary>

Although there has been work to develop ex-post private mechanisms from Ligett et al. '17 and Whitehouse et al '22 that seeks to provide privacy guarantees subject to a target level of accuracy, there was not a way to use them in conjunction with differentially private mechanisms.  Furthermore, there has yet to be work in developing a theory for how these ex-post privacy mechanisms compose, so that we can track the accumulated privacy over several mechanisms.  We develop privacy filters that allow an analyst to adaptively switch between differentially private mechanisms and ex-post private mechanisms subject to an overall privacy loss guarantee.  We show that using  a particular ex-post private mechanism --- noise reduction mechanisms --- can substantially outperform baseline approaches that use existing privacy loss composition bounds.  We use the common task of returning as many counts as possible subject to a relative error guarantee and an overall privacy budget as a motivating example.</details>

### Addressing the speed-accuracy simulation trade-off for adaptive spiking neurons

**Authors:** Luke Taylor, Andrew King, Nicol S Harper

**<details><summary>Abstract**</summary>

The adaptive leaky integrate-and-fire (ALIF) model is fundamental within computational neuroscience and has been instrumental in studying our brains $\textit{in silico}$. Due to the sequential nature of simulating these neural models, a commonly faced issue is the speed-accuracy trade-off: either accurately simulate a neuron using a small discretisation time-step (DT), which is slow, or more quickly simulate a neuron using a larger DT and incur a loss in simulation accuracy. Here we provide a solution to this dilemma, by algorithmically reinterpreting the ALIF model, reducing the sequential simulation complexity and permitting a more efficient parallelisation on GPUs. We computationally validate our implementation to obtain over a $50\times$ training speedup using small DTs on synthetic benchmarks. We also obtained a comparable performance to the standard ALIF implementation on different supervised classification tasks - yet in a fraction of the training time. Lastly, we showcase how our model makes it possible to quickly and accurately fit real electrophysiological recordings of cortical neurons, where very fine sub-millisecond DTs are crucial for capturing exact spike timing.</details>

### Adversarial Examples Might be Avoidable: The Role of Data Concentration in Adversarial Robustness

**Authors:** Ambar Pal, Jeremias Sulam, Rene Vidal

**<details><summary>Abstract**</summary>

The susceptibility of modern machine learning classifiers to adversarial examples has motivated theoretical results suggesting that these might be unavoidable. However, these results can be too general to be applicable to natural data distributions. Indeed, humans are quite robust for tasks involving vision. This apparent conflict motivates a deeper dive into the question: Are adversarial examples truly unavoidable? In this work, we theoretically demonstrate that a key property of the data distribution -- concentration on small-volume subsets of the input space -- determines whether a robust classifier exists. We further demonstrate that, for a data distribution concentrated on a union of low-dimensional linear subspaces, exploiting data structure naturally leads to classifiers that enjoy good robustness guarantees, improving upon methods for provable certification in certain regimes.</details>

### Adversarial Self-Training Improves Robustness and Generalization for Gradual Domain Adaptation

**Authors:** Lianghe Shi, Weiwei Liu

**<details><summary>Abstract**</summary>

Gradual Domain Adaptation (GDA), in which the learner is provided with additional intermediate domains, has been theoretically and empirically studied in many contexts. Despite its vital role in security-critical scenarios, the adversarial robustness of the GDA model remains unexplored. In this paper, we adopt the effective gradual self-training method and replace vanilla self-training with adversarial self-training (AST). AST first predicts labels on the unlabeled data and then adversarially trains the model on the pseudo-labeled distribution. Intriguingly, we find that gradual AST improves not only adversarial accuracy but also clean accuracy on the target domain. We reveal that this is because adversarial training (AT) performs better than standard training when the pseudo-labels contain a portion of incorrect labels. Accordingly, we first present the generalization error bounds for gradual AST in a multiclass classification setting. We then use the optimal value of the Subset Sum Problem to bridge the standard error on a real distribution and the adversarial error on a pseudo-labeled distribution. The result indicates that AT may obtain a tighter bound than standard training on data with incorrect pseudo-labels. We further present an example of a conditional Gaussian distribution to provide more insights into why gradual AST can improve the clean accuracy for GDA.</details>

### Advice Querying under Budget Constraint for Online Algorithms

**Authors:** Ziyad Benomar, Vianney Perchet

**<details><summary>Abstract**</summary>

Several problems have been extensively studied in the learning-augmented setting, where the algorithm has access to some, possibly incorrect, predictions. However, it is assumed in most works that the predictions are provided to the algorithm as input, with no constraint on their size. In this paper, we consider algorithms with access to a limited number of predictions, that they can request at any time during their execution. We study three classical problems in competitive analysis, the ski rental problem, the secretary problem, and the non-clairvoyant job scheduling. We address the question of when to query predictions and how to use them.</details>

### AiluRus: A Scalable ViT Framework for Dense Prediction

**Authors:** Jin Li, Yaoming Wang, XIAOPENG ZHANG, Bowen Shi, Dongsheng Jiang, Chenglin Li, Wenrui Dai, Hongkai Xiong, Qi Tian

**<details><summary>Abstract**</summary>

Vision transformers (ViTs) have emerged as a prevalent architecture for vision tasks owing to their impressive performance. However, their complexity dramatically increases when handling long token sequences, particularly for dense prediction tasks that require high-resolution input. Notably, dense prediction tasks, such as semantic segmentation or object detection, emphasize more on the contours or shapes of objects, while the texture inside objects is less informative. Motivated by this observation, we propose to apply adaptive resolution for different regions in the image according to their importance. Specifically, at the intermediate layer of the ViT, we select anchors from the token sequence using the proposed spatial-aware density-based clustering algorithm. Tokens that are adjacent to anchors are merged to form low-resolution regions, while others are preserved independently as high-resolution. This strategy could significantly reduce the number of tokens, and the following layers only handle the reduced token sequence for acceleration. At the output end, the resolution of the feature map is recovered by unfolding merged tokens for task prediction. Consequently, we can considerably accelerate ViTs for dense prediction tasks. The proposed method is evaluated across three different datasets and demonstrates promising performance. For instance, "Segmenter ViT-L" can be accelerated by 48\% FPS without fine-tuning, while maintaining the performance. Moreover, our method can also be applied to accelerate fine-tuning. Experiments indicate that we can save 52\% training time while accelerating 2.46$\times$ FPS with only a 0.09\% performance drop.</details>

### AlberDICE: Addressing Out-Of-Distribution Joint Actions in Offline Multi-Agent RL via Alternating Stationary Distribution Correction Estimation

**Authors:** Daiki E. Matsunaga, Jongmin Lee, Jaeseok Yoon, Stefanos Leonardos, Pieter Abbeel, Kee-Eung Kim

**<details><summary>Abstract**</summary>

One of the main challenges in offline Reinforcement Learning (RL) is the distribution shift that arises from the learned policy deviating from the data collection policy. This is often addressed by avoiding out-of-distribution (OOD) actions during policy improvement as their presence can lead to substantial performance degradation. This challenge is amplified in the offline Multi-Agent RL (MARL) setting since the joint action space grows exponentially with the number of agents.To avoid this curse of dimensionality, existing MARL methods adopt either value decomposition methods or fully decentralized training of individual agents. However, even when combined with standard conservatism principles, these methods can still result in the selection of OOD joint actions in offline MARL. To this end, we introduce AlberDICE,an offline MARL algorithm that alternatively performs centralized training of individual agents based on stationary distribution optimization. AlberDICE circumvents the exponential complexity of MARL by computing the best response of one agent at a time while effectively avoiding OOD joint action selection. Theoretically, we show that the alternating optimization procedure converges to Nash policies. In the experiments, we demonstrate that AlberDICE significantly outperforms baseline algorithms on a standard suite of MARL benchmarks.</details>

### Aligning Language Models with Human Preferences via a Bayesian Approach

**Authors:** Jiashuo WANG, Haozhao Wang, Shichao Sun, Wenjie Li

**<details><summary>Abstract**</summary>

In the quest to advance human-centric natural language generation (NLG) systems, ensuring alignment between NLG models and human preferences is crucial. For this alignment, current popular methods leverage a reinforcement learning (RL) approach with a reward model trained on feedback from humans. However, inherent disagreements due to the subjective nature of human preferences pose a significant challenge for training the reward model, resulting in a deterioration of the NLG performance. To tackle this issue, previous approaches typically rely on majority voting or averaging to consolidate multiple inconsistent preferences into a merged one. Although straightforward to understand and execute, such methods suffer from an inability to capture the nuanced degrees of disaggregation among humans and may only represent a specialized subset of individuals, thereby lacking the ability to quantitatively disclose the universality of human preferences. To address this challenge, this paper proposes a novel approach, which employs a Bayesian framework to account for the distribution of disagreements among human preferences as training a preference model, and names it as $\textbf{d-PM}$. Besides, considering the RL strategy's inefficient and complex training process over the training efficiency, we further propose utilizing the contrastive learning strategy to train the NLG model with the preference scores derived from the d-PM model. Extensive experiments on two human-centric NLG tasks, i.e., emotional support conversation and integrity ``Rule-of-Thumb'' generation, show that our method consistently exceeds previous SOTA models in both automatic and human evaluations.</details>

### Aligning Optimization Trajectories with Diffusion Models for Constrained Design Generation

**Authors:** Giorgio Giannone, Akash Srivastava, Ole Winther, Faez Ahmed

**<details><summary>Abstract**</summary>

Generative models have significantly influenced both vision and language domains, ushering in innovative multimodal applications. Although these achievements have motivated exploration in scientific and engineering fields, challenges emerge, particularly in constrained settings with limited data where precision is crucial. Traditional engineering optimization methods rooted in physics often surpass generative models in these contexts. To address these challenges, we introduce Diffusion Optimization Models (DOM) and Trajectory Alignment (TA), a learning framework that demonstrates the efficacy of aligning the sampling trajectory of diffusion models with the trajectory derived from physics-based iterative optimization methods. This alignment ensures that the sampling process remains grounded in the underlying physical principles. This alignment eliminates the need for costly preprocessing, external surrogate models, or extra labeled data, generating feasible and high-performance designs efficiently. We apply our framework to structural topology optimization, a fundamental problem in mechanical design, evaluating its performance on in- and out-of-distribution configurations. Our results demonstrate that TA outperforms state-of-the-art deep generative models on in-distribution configurations and halves the inference computational cost. When coupled with a few steps of optimization, it also improves manufacturability for out-of-distribution conditions. DOM's efficiency and performance improvements significantly expedite design processes and steer them toward optimal and manufacturable outcomes, highlighting the potential of generative models in data-driven design.</details>

### [Spotlight] Alignment with human representations supports robust few-shot learning

**Authors:** Ilia Sucholutsky, Tom Griffiths

**<details><summary>Abstract**</summary>

Should we care whether AI systems have representations of the world that are similar to those of humans? We provide an information-theoretic analysis that suggests that there should be a U-shaped relationship between the degree of representational alignment with humans and performance on few-shot learning tasks. We confirm this prediction empirically, finding such a relationship in an analysis of the performance of 491 computer vision models. We also show that highly-aligned models are more robust to both natural adversarial attacks and domain shifts. Our results suggest that human-alignment is often a sufficient, but not necessary, condition for models to make effective use of limited data, be robust, and generalize well.</details>

### [Spotlight] Alternation makes the adversary weaker in two-player games

**Authors:** Volkan Cevher, Ashok Cutkosky, Ali Kavis, Georgios Piliouras, Stratis Skoulakis, Luca Viano

**<details><summary>Abstract**</summary>

Motivated by alternating game-play in two-player games, we study an altenating variant of the \textit{Online Linear Optimization} (OLO). In alternating OLO,  a \textit{learner} at each round $t \in [n]$ selects a vector $x^t$ and then an \textit{adversary} selects a cost-vector $c^t \in [-1,1]^n$. The learner then experiences cost $(c^t + c^{t-1})^\top x^t$ instead of $(c^t)^\top x^t$ as in standard OLO. We establish that under this small twist, the $\Omega(\sqrt{T})$ lower bound on the regret is no longer valid. More precisely, we present two online learning algorithms for alternating OLO that respectively admit $\mathcal{O}((\log n)^{4/3} T^{1/3})$ regret for the $n$-dimensional simplex and $\mathcal{O}(\rho \log T)$ regret for the ball of radius $\rho>0$. Our results imply that in alternating game-play, an agent can always guarantee $\mathcal{\tilde{O}}((\log n)^{4/3} T^{1/3})$ regardless the strategies of the other agent while the regret bound improves to $\mathcal{O}(\log T)$ in case the agent admits only two actions.</details>

### An Efficient Dataset Condensation Plugin and Its Application to Continual Learning

**Authors:** Enneng Yang, Li Shen, Zhenyi Wang, Tongliang Liu, Guibing Guo

**<details><summary>Abstract**</summary>

Dataset condensation (DC) distills a large real-world dataset into a small synthetic dataset, with the goal of training a network from scratch on the latter that performs similarly to the former. State-of-the-art (SOTA) DC methods have achieved satisfactory results through techniques such as accuracy, gradient, training trajectory, or distribution matching. However, these works all perform matching in the high-dimension pixel spaces, ignoring that natural images are usually locally connected and have lower intrinsic dimensions, resulting in low condensation efficiency.  In this work, we propose a simple-yet-efficient dataset condensation plugin that matches the raw and synthetic datasets in a low-dimensional manifold. Specifically, our plugin condenses raw images into two low-rank matrices instead of parameterized image matrices. Our plugin can be easily incorporated into existing DC methods, thereby containing richer raw dataset information at limited storage costs to improve the downstream applications' performance.  We verify on multiple public datasets that when the proposed plugin is combined with SOTA DC methods, the performance of the network trained on synthetic data is significantly improved compared to traditional DC methods. Moreover, when applying the DC methods as a plugin to continual learning tasks, we observed that our approach effectively mitigates catastrophic forgetting of old tasks under limited memory buffer constraints and avoids the problem of raw data privacy leakage.</details>

### An Information Theory Perspective on Variance-Invariance-Covariance Regularization

**Authors:** Ravid Shwartz-Ziv, Randall Balestriero, Kenji Kawaguchi, Tim G. J. Rudner, Yann LeCun

**<details><summary>Abstract**</summary>

Variance-Invariance-Covariance Regularization (VICReg) is a self-supervised learning (SSL) method that has shown promising results on a variety of tasks. However, the fundamental mechanisms underlying VICReg remain unexplored. In this paper, we present an information-theoretic perspective on the VICReg objective. We begin by deriving information-theoretic quantities for deterministic networks as an alternative to unrealistic stochastic network assumptions. We then relate the optimization of the VICReg objective to mutual information optimization, highlighting underlying assumptions and facilitating a constructive comparison with other SSL algorithms and derive a generalization bound for VICReg, revealing its inherent advantages for downstream tasks. Building on these results, we introduce a family of SSL methods derived from information-theoretic principles that outperform existing SSL techniques.</details>

### An Information-Theoretic Evaluation of Generative Models in Learning Multi-modal Distributions

**Authors:** Mohammad Jalali, Cheuk Ting Li, Farzan Farnia

**<details><summary>Abstract**</summary>

The evaluation of generative models has received significant attention in the machine learning community.  When applied to a multi-modal distribution which is common among image datasets, an intuitive evaluation criterion is the number of modes captured by the generative model. While several scores have been proposed to evaluate the quality and diversity of a model's generated data, the correspondence between existing scores and the number of modes in the distribution is unclear. In this work, we propose an information-theoretic diversity evaluation method for multi-modal underlying distributions. We utilize the R\'enyi Kernel Entropy (RKE) as an evaluation score based on quantum information theory to measure the number of modes in generated samples. To interpret the proposed evaluation method, we show that the RKE score can output the number of modes of a mixture of sub-Gaussian components. We also prove estimation error bounds for estimating the RKE score from limited data, suggesting a fast convergence of the empirical RKE score to the score for the underlying data distribution. Utilizing the RKE score, we conduct an extensive evaluation of state-of-the-art generative models over standard image datasets. The numerical results indicate that while the recent algorithms for training generative models manage to improve the mode-based diversity over the earlier architectures, they remain incapable of capturing the full diversity of real data. Our empirical results provide a ranking of widely-used generative models based on the RKE score of their generated samples.</details>

### An Iterative Self-Learning Framework for Medical Domain Generalization

**Authors:** Zhenbang Wu, Huaxiu Yao, David Liebovitz, Jimeng Sun

**<details><summary>Abstract**</summary>

Deep learning models have been widely used to assist doctors with clinical decision-making. However, these models often encounter a significant performance drop when applied to data that differs from the distribution they were trained on. This challenge is known as the domain shift problem. Existing domain generalization algorithms attempt to address this problem by assuming the availability of domain IDs and training a single model to handle all domains. However, in healthcare settings, patients can be classified into numerous latent domains, where the actual domain categorizations are unknown. Furthermore, each patient domain exhibits distinct clinical characteristics, making it sub-optimal to train a single model for all domains. To overcome these limitations, we propose SLGD, a self-learning framework that iteratively discovers decoupled domains and trains personalized classifiers for each decoupled domain. We evaluate the generalizability of SLGD across spatial and temporal data distribution shifts on two real-world public EHR datasets: eICU and MIMIC-IV. Our results show that SLGD achieves up to 11% improvement in the AUPRC score over the best baseline.</details>

### [Spotlight] An Optimal and Scalable Matrix Mechanism for Noisy Marginals under Convex Loss Functions

**Authors:** Yingtai Xiao, Guanlin He, Danfeng Zhang, Daniel Kifer

**<details><summary>Abstract**</summary>

Noisy marginals are a common form of confidentiality-protecting data release and are useful for many downstream tasks such as contingency table analysis, construction of Bayesian networks, and even synthetic data generation. Privacy mechanisms that provide unbiased noisy answers to linear queries (such as marginals) are known as matrix mechanisms.We propose ResidualPlanner, a matrix mechanism for marginals with Gaussian noise that is both optimal and scalable. ResidualPlanner can optimize for many loss functions that can be written as a convex function of marginal variances (prior work was restricted to just one predefined objective function). ResidualPlanner can optimize the accuracy of marginals in large scale settings in seconds, even when the previous state of the art (HDMM) runs out of memory. It even runs on datasets with 100 attributes in a couple of minutes. Furthermore ResidualPlanner can efficiently compute variance/covariance values for each marginal (prior methods quickly run out of memory, even for relatively small datasets).</details>

### Anytime-Competitive Reinforcement Learning with Policy Prior

**Authors:** Jianyi Yang, Pengfei Li, Tongxin Li, Adam Wierman, Shaolei Ren

**<details><summary>Abstract**</summary>

This paper studies the problem of Anytime-Competitive Markov Decision Process (A-CMDP). Existing works on Constrained Markov Decision Processes (CMDPs) aim to optimize the expected reward while constraining the expected cost over random dynamics, but the cost in a specific episode can still be unsatisfactorily high. In contrast, the goal of A-CMDP is to optimize the expected reward while guaranteeing a bounded cost in each round of any episode against a policy prior. We propose a new algorithm, called Anytime-Competitive Reinforcement Learning (ACRL), which provably guarantees the anytime cost constraints. The regret analysis shows the policy asymptotically matches the optimal reward achievable under the anytime competitive constraints. Experiments on the application of carbon-intelligent computing verify the reward performance and cost constraint guarantee of ACRL.</details>

### Approximate Allocation Matching for Structural Causal Bandits with Unobserved Confounders

**Authors:** Lai Wei, Muhammad Qasim Elahi, Mahsa Ghasemi, Murat Kocaoglu

**<details><summary>Abstract**</summary>

Structural causal bandit provides a framework for online decision-making problems when causal information is available. It models the stochastic environment with a structural causal model (SCM) that governs the causal relations between random variables. In each round, an agent applies an intervention (or no intervention) by setting certain variables to some constants and receives a stochastic reward from a non-manipulable variable. Though the causal structure is given, the observational and interventional distributions of these random variables are unknown beforehand, and they can only be learned through interactions with the environment. Therefore, to maximize the expected cumulative reward, it is critical to balance the explore-versus-exploit tradeoff. We assume each random variable takes a finite number of distinct values, and consider a semi-Markovian setting, where random variables are affected by unobserved confounders. Using the canonical SCM formulation to discretize the domains of unobserved variables, we efficiently integrate samples to reduce model uncertainty. This gives the decision maker a natural advantage over those in a classical multi-armed bandit setup. We provide a logarithmic asymptotic regret lower bound for the structural causal bandit problem. Inspired by the lower bound, we design an algorithm that can utilize the causal structure to accelerate the learning process and take informative and rewarding interventions. We establish that our algorithm achieves a logarithmic regret and demonstrate that it outperforms the existing methods via simulations.</details>

### Approximation-Generalization Trade-offs under (Approximate) Group Equivariance

**Authors:** Mircea Petrache, Shubhendu Trivedi

**<details><summary>Abstract**</summary>

The explicit incorporation of task-specific inductive biases through symmetry has emerged as a general design precept in the development of high-performance machine learning models. For example, group equivariant neural networks have demonstrated impressive performance across various domains and applications such as protein and drug design. A prevalent intuition about such models is that the integration of relevant symmetry results in enhanced generalization. Moreover, it is posited that when the data and/or the model exhibits only approximate or partial symmetry, the optimal or best-performing model is one where the model symmetry aligns with the data symmetry. In this paper, we conduct a formal unified investigation of these intuitions. To begin, we present quantitative bounds that demonstrate how models capturing task-specific symmetries lead to improved generalization. Utilizing this quantification, we examine the more general question of dealing with approximate/partial symmetries. We establish, for a given symmetry group, a quantitative comparison between the approximate equivariance of the model and that of the data distribution, precisely connecting model equivariance error and data equivariance error. Our result delineates the conditions under which the model equivariance error is optimal, thereby yielding the best-performing model for the given task and data.</details>

### Architecture Matters: Uncovering Implicit Mechanisms in Graph Contrastive Learning

**Authors:** Xiaojun Guo, Yifei Wang, Zeming Wei, Yisen Wang

**<details><summary>Abstract**</summary>

With the prosperity of contrastive learning for visual representation learning (VCL), it is also adapted to the graph domain and yields promising performance. However, through a systematic study of various graph contrastive learning (GCL) methods, we observe that some common phenomena among existing GCL methods that are quite different from the original VCL methods, including 1) positive samples are not a must for GCL; 2) negative samples are not necessary for graph classification, neither for node classification when adopting specific normalization modules; 3) data augmentations have much less influence on GCL, as simple domain-agnostic augmentations (e.g., Gaussian noise) can also attain fairly good performance. By uncovering how the implicit inductive bias of GNNs works in contrastive learning, we theoretically provide insights into the above intriguing properties of GCL. Rather than directly porting existing VCL methods to GCL, we advocate for more attention toward the unique architecture of graph learning and consider its implicit influence when designing GCL methods. Code is available at https://github.com/PKU-ML/ArchitectureMattersGCL.</details>

### Are aligned neural networks adversarially aligned?

**Authors:** Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Pang Wei Koh, Daphne Ippolito, Florian Tramer, Ludwig Schmidt

**<details><summary>Abstract**</summary>

Large language models are now tuned to align with the goals of their creators, namely to be "helpful and harmless." These models should respond helpfully to user questions, but refuse to answer requests that could cause harm. However, adversarial users can construct inputs which circumvent attempts at alignment. In this work, we study adversarial alignment, and ask to what extent these models remain aligned when interacting with an adversarial user who constructs worst-case inputs (adversarial examples). These inputs are designed to cause the model to emit harmful content that would otherwise be prohibited.We show that existing NLP-based optimization attacks are insufficiently powerful to reliably attack aligned text models: even when current NLP-based attacks fail, we can find adversarial inputs with brute force. As a result, the failure of current attacks should not be seen as proof that aligned text models remain aligned under adversarial inputs. However the recent trend in large-scale ML models is multimodal models that allow users to provide images that influence the text that is generated. We show these models can be easily attacked, i.e., induced to perform arbitrary un-aligned behavior through adversarial perturbation of the input image. We conjecture that improved NLP attacks may demonstrate this same level of adversarial control over text-only models.</details>

### Assessor360: Multi-sequence Network for Blind Omnidirectional Image Quality Assessment

**Authors:** Tianhe Wu, Shuwei Shi, Haoming Cai, Mingdeng Cao, Jing Xiao, Yinqiang Zheng, Yujiu Yang

**<details><summary>Abstract**</summary>

Blind Omnidirectional Image Quality Assessment (BOIQA) aims to objectively assess the human perceptual quality of omnidirectional images (ODIs) without relying on pristine-quality image information. It is becoming more significant with the increasing advancement of virtual reality (VR) technology. However, the quality assessment of ODIs is severely hampered by the fact that the existing BOIQA pipeline lacks the modeling of the observer's browsing process. To tackle this issue, we propose a novel multi-sequence network for BOIQA called Assessor360, which is derived from the realistic multi-assessor ODI quality assessment procedure. Specifically, we propose a generalized Recursive Probability Sampling (RPS) method for the BOIQA task, combining content and details information to generate multiple pseudo viewport sequences from a given starting point. Additionally, we design a Multi-scale Feature Aggregation (MFA) module with a Distortion-aware Block (DAB) to fuse distorted and semantic features of each viewport. We also devise Temporal Modeling Module (TMM) to learn the viewport transition in the temporal domain. Extensive experimental results demonstrate that Assessor360 outperforms state-of-the-art methods on multiple OIQA datasets. The code and models are available at https://github.com/TianheWu/Assessor360.</details>

### Assumption violations in causal discovery and the robustness of score matching

**Authors:** Francesco Montagna, Atalanti Mastakouri, Elias Eulig, Nicoletta Noceti, Lorenzo Rosasco, Dominik Janzing, Bryon Aragam, Francesco Locatello

**<details><summary>Abstract**</summary>

When domain knowledge is limited and experimentation is restricted by ethical, financial, or time constraints, practitioners turn to observational causal discovery methods to recover the causal structure, exploiting the statistical properties of their data. Because causal discovery without further assumptions is an ill-posed problem, each algorithm comes with its own set of usually untestable assumptions, some of which are hard to meet in real datasets. Motivated by these considerations, this paper extensively benchmarks the empirical performance of recent causal discovery methods on observational _iid_ data generated under different background conditions, allowing for violations of the critical assumptions required by each selected approach. Our experimental findings show that score matching-based methods demonstrate surprising performance in the false positive and false negative rate of the inferred graph in these challenging scenarios, and we provide theoretical insights into their performance. This work is also the first effort to benchmark the stability of causal discovery algorithms with respect to the values of their hyperparameters. Finally, we hope this paper will set a new standard for the evaluation of causal discovery methods and can serve as an accessible entry point for practitioners interested in the field, highlighting the empirical implications of different algorithm choices.</details>

### Asymptotically Optimal Quantile Pure Exploration for Infinite-Armed Bandits

**Authors:** Evelyn Xiao-Yue Gong, Mark Sellke

**<details><summary>Abstract**</summary>

We study pure exploration with infinitely many bandit arms generated \iid from an unknown distribution. Our goal is to efficiently select a single high quality arm whose average reward is, with probability $1-\delta$, within $\varepsilon$ of being with the top $\eta$-fraction of arms; this is a natural adaptation of the classical PAC guarantee for infinite action sets. We consider both the fixed confidence and fixed budget settings, aiming respectively for optimal \emph{expected} and \emph{fixed} sample complexity.For fixed confidence, we give an algorithm with expected sample complexity $O\left(\frac{\log (1/\eta)\log (1/\delta)}{\eta\varepsilon^2}\right)$. This is optimal except for the $\log (1/\eta)$ factor, and the $\delta$-dependence closes a quadratic gap in the literature. For fixed budget, we show the asymptotically optimal sample complexity as $\delta\to 0$ is $c^{-1}\log(1/\delta)\big(\log\log(1/\delta)\big)^2$ to leading order; equivalently, the optimal failure probability with exactly $N$ samples decays as $\exp\big(-(1\pm o(1))\frac{cN}{\log^2 N}\big)$.The value of $c$ depends explicitly on the problem parameters (including the unknown arm distribution) through a certain Fisher information distance. Even the strictly super-linear dependence on $\log(1/\delta)$ was not known and resolves a question of Grossman-Moshkovitz (FOCS 2015).</details>

### [Spotlight] Attentive Transfer Entropy to Exploit Transient Emergence of Coupling Effect

**Authors:** Xiaolei Ru, XINYA ZHANG, Zijia Liu, Jack Murdoch Moore, Gang Yan

**<details><summary>Abstract**</summary>

We consider the problem of reconstructing coupled networks (e.g., biological neural networks) connecting large numbers of variables (e.g.,nerve cells), of which state evolution is governed by dissipative dynamics consisting of strong self-drive (dominants the evolution) and weak coupling-drive. The core difficulty is sparseness of coupling effect that emerges (the coupling force is significant) only momentarily and otherwise remains quiescent in time series (e.g., neuronal activity sequence). Here we learn the idea from attention mechanism to guide the classifier to make inference focusing on the critical regions of time series data where coupling effect may manifest. Specifically, attention coefficients are assigned autonomously by artificial neural networks trained to maximise the Attentive Transfer Entropy (ATEn), which is a novel generalization of the iconic transfer entropy metric. Our results show that, without any prior knowledge of dynamics, ATEn explicitly identifies areas where the strength of coupling-drive is distinctly greater than zero. This innovation substantially improves reconstruction performance for both synthetic and real directed coupling networks using data generated by neuronal models widely used in neuroscience.</details>

### [Spotlight] Auditing Fairness by Betting

**Authors:** Ben Chugg, Santiago Cortes-Gomez, Bryan Wilder, Aaditya Ramdas

**<details><summary>Abstract**</summary>

We provide practical, efficient, and nonparametric methods for auditing the fairness of deployed classification and regression models. Whereas previous work relies on a fixed-sample size, our methods are sequential and allow for the continuous monitoring of incoming data, making them highly amenable to tracking the fairness of real-world systems. We also allow the data to be collected by a  probabilistic policy as opposed to sampled uniformly from the population. This enables auditing to be conducted on data gathered for another purpose. Moreover, this policy may change over time and different policies may be used on different subpopulations. Finally, our methods can handle distribution shift resulting from either changes to the model or changes in the underlying population. Our approach is based on recent progress in anytime-valid inference and game-theoretic statistics---the ``testing by betting'' framework in particular. These connections ensure that our methods are interpretable, fast, and easy to implement. We demonstrate the efficacy of our approach on three benchmark fairness datasets.</details>

### Augmentation-Aware Self-Supervision for Data-Efficient GAN Training

**Authors:** Liang Hou, Qi Cao, Yige Yuan, Songtao Zhao, Chongyang Ma, Siyuan Pan, Pengfei Wan, Zhongyuan Wang, Huawei Shen, Xueqi Cheng

**<details><summary>Abstract**</summary>

Training generative adversarial networks (GANs) with limited data is challenging because the discriminator is prone to overfitting. Previously proposed differentiable augmentation demonstrates improved data efficiency of training GANs. However, the augmentation implicitly introduces undesired invariance to augmentation for the discriminator since it ignores the change of semantics in the label space caused by data transformation, which may limit the representation learning ability of the discriminator and ultimately affect the generative modeling performance of the generator. To mitigate the negative impact of invariance while inheriting the benefits of data augmentation, we propose a novel augmentation-aware self-supervised discriminator that predicts the augmentation parameter of the augmented data. Particularly, the prediction targets of real data and generated data are required to be distinguished since they are different during training. We further encourage the generator to adversarially learn from the self-supervised discriminator by generating augmentation-predictable real and not fake data. This formulation connects the learning objective of the generator and the arithmetic $-$ harmonic mean divergence under certain assumptions. We compare our method with state-of-the-art (SOTA) methods using the class-conditional BigGAN and unconditional StyleGAN2 architectures on data-limited CIFAR-10, CIFAR-100, FFHQ, LSUN-Cat, and five low-shot datasets. Experimental results demonstrate significant improvements of our method over SOTA methods in training data-efficient GANs.</details>

### [Spotlight] Balancing memorization and generalization in RNNs for high performance brain-machine Interfaces

**Authors:** Joseph Costello, Hisham Temmar, Luis Cubillos, Matthew Mender, Dylan Wallace, Matt Willsey, Parag Patil, Cynthia Chestek

**<details><summary>Abstract**</summary>

Brain-machine interfaces (BMIs) can restore motor function to people with paralysis but are currently limited by the accuracy of real-time decoding algorithms. Recurrent neural networks (RNNs) using modern training techniques have shown promise in accurately predicting movements from neural signals but have yet to be rigorously evaluated against other decoding algorithms in a closed-loop setting. Here we compared RNNs to other neural network architectures in real-time, continuous decoding of finger movements using intracortical signals from nonhuman primates. Across one and two finger online tasks, LSTMs (a type of RNN) outperformed convolutional and transformer-based neural networks, averaging 18% higher throughput than the convolution network. On simplified tasks with a reduced movement set, RNN decoders were allowed to memorize movement patterns and matched able-bodied control. Performance gradually dropped as the number of distinct movements increased but did not go below fully continuous decoder performance. Finally, in a two-finger task where one degree-of-freedom had poor input signals, we recovered functional control using RNNs trained to act both like a movement classifier and continuous decoder. Our results suggest that RNNs can enable functional real-time BMI control by learning and generating accurate movement patterns.</details>

### Bandit Task Assignment with Unknown Processing Time

**Authors:** Shinji Ito, Daisuke Hatano, Hanna Sumita, Kei Takemura, Takuro Fukunaga, Naonori Kakimura, Ken-Ichi Kawarabayashi

**<details><summary>Abstract**</summary>

This study considers a novel problem setting, referred to as \textit{bandit task assignment}, that incorporates the processing time of each task in the bandit setting. In this problem setting, a player sequentially chooses a set of tasks to start so that the set of processing tasks satisfies a given combinatorial constraint. The reward and processing time for each task follow unknown distributions, values of which are revealed only after the task has been completed. The problem generalizes the stochastic combinatorial semi-bandit problem and the budget-constrained bandit problem. For this problem setting, we propose an algorithm based on upper confidence bounds~(UCB) combined with a phased-update approach. The proposed algorithm admits a gap-dependent regret upper bound of $O(MN(1/\Delta){\log T})$ and a gap-free regret upper bound of $\tilde{O}( \sqrt{MNT} )$, where $N$ is the number of the tasks, $M$ is the maximum number of tasks run at the same time, $T$ is the time horizon, and $\Delta$ is the gap between expected per-round rewards of the optimal and best suboptimal sets of tasks. These regret bounds nearly match lower bounds.</details>

### Bayes beats Cross Validation: Efficient and Accurate Ridge Regression via Expectation Maximization

**Authors:** Shu Yu Tew, Mario Boley, Daniel Schmidt

**<details><summary>Abstract**</summary>

We present a novel method for tuning the regularization hyper-parameter, $\lambda$, of a ridge regression that is faster to compute than leave-one-out cross-validation (LOOCV) while yielding estimates of the regression parameters of equal, or particularly in the setting of sparse covariates, superior quality to those obtained by minimising the LOOCV risk. The LOOCV risk can suffer from multiple and bad local minima for finite $n$ and thus requires the specification of a set of candidate $\lambda$, which can fail to provide good solutions. In contrast, we show that the proposed method is guaranteed to find a unique optimal solution for large enough $n$, under relatively mild conditions, without requiring the specification of any difficult to determine hyper-parameters. This is based on a Bayesian formulation of ridge regression that we prove to have a unimodal posterior for large enough $n$, allowing for  both the optimal $\lambda$ and the regression coefficients to be jointly learned within an iterative expectation maximization (EM) procedure. Importantly, we show that by utilizing an appropriate preprocessing step, a single iteration of the main EM loop can be implemented in $O(\min(n, p))$ operations, for input data with $n$ rows and $p$ columns. In contrast, evaluating a single value of $\lambda$ using fast LOOCV costs $O(n \min(n, p))$ operations when using the same preprocessing. This advantage amounts to an asymptotic improvement of a factor of $l$ for $l$ candidate values for $\lambda$ (in the regime $q, p \in O(\sqrt{n})$ where $q$ is the number of regression targets).</details>

### Better Correlation and Robustness: A Distribution-Balanced Self-Supervised Learning Framework for Automatic Dialogue Evaluation

**Authors:** Peiwen Yuan, Xinglin Wang, Jiayi Shi, Bin Sun, Yiwei Li, Prof. Kan

**<details><summary>Abstract**</summary>

Turn-level dialogue evaluation models (TDEMs), using self-supervised learning (SSL) framework, have achieved state-of-the-art performance in open-domain dialogue evaluation. However, these models inevitably face two potential problems. First, they have low correlations with humans on medium coherence samples as the SSL framework often brings training data with unbalanced coherence distribution. Second, the SSL framework leads TDEM to nonuniform score distribution. There is a danger that the nonuniform score distribution will weaken the robustness of TDEM through our theoretical analysis. To tackle these problems, we propose Better Correlation and Robustness (BCR), a distribution-balanced self-supervised learning framework for TDEM. Given a dialogue dataset, BCR offers an effective training set reconstructing method to provide coherence-balanced training signals and further facilitate balanced evaluating abilities of TDEM. To get a uniform score distribution, a novel loss function is proposed, which can adjust adaptively according to the uniformity of score distribution estimated by kernel density estimation. Comprehensive experiments on 17 benchmark datasets show that vanilla BERT-base using BCR outperforms SOTA methods significantly by 11.3% on average. BCR also demonstrates strong generalization ability as it can lead multiple SOTA methods to attain better correlation and robustness.</details>

### Beyond Exponential Graph: Communication-Efficient Topologies for Decentralized Learning via Finite-time Convergence

**Authors:** Yuki Takezawa, Ryoma Sato, Han Bao, Kenta Niwa, Makoto Yamada

**<details><summary>Abstract**</summary>

Decentralized learning has recently been attracting increasing attention for its applications in parallel computation and privacy preservation. Many recent studies stated that the underlying network topology with a faster consensus rate (a.k.a. spectral gap) leads to a better convergence rate and accuracy for decentralized learning. However, a topology with a fast consensus rate, e.g., the exponential graph, generally has a large maximum degree, which incurs significant communication costs. Thus, seeking topologies with both a fast consensus rate and small maximum degree is important. In this study, we propose a novel topology combining both a fast consensus rate and small maximum degree called the Base-$\left(k+1\right)$ Graph. Unlike the existing topologies, the Base-$\left(k+1\right)$ Graph enables all nodes to reach the exact consensus after a finite number of iterations for any number of nodes and maximum degree $k$. Thanks to this favorable property, the Base-$\left(k+1\right)$ Graph endows Decentralized SGD (DSGD) with both a faster convergence rate and more communication efficiency than the exponential graph. We conducted experiments with various topologies, demonstrating that the Base-$\left(k+1\right)$ Graph enables various decentralized learning methods to achieve higher accuracy with better communication efficiency than the existing topologies. Our code is available at https://github.com/yukiTakezawa/BaseGraph.</details>

### Bi-Level Offline Policy Optimization with Limited Exploration

**Authors:** Wenzhuo Zhou

**<details><summary>Abstract**</summary>

We study offline reinforcement learning (RL) which seeks to learn a good policy based on a fixed, pre-collected dataset. A fundamental challenge behind this task is the distributional shift due to the dataset lacking sufficient exploration, especially under function approximation. To tackle this issue, we propose a bi-level structured policy optimization algorithm that models a hierarchical interaction between the policy (upper-level) and the value function (lower-level). The lower level focuses on constructing a confidence set of value estimates that maintain sufficiently small weighted average Bellman errors, while controlling uncertainty arising from distribution mismatch. Subsequently, at the upper level, the policy aims to maximize a conservative value estimate from the confidence set formed at the lower level. This novel formulation preserves the maximum flexibility of the implicitly induced exploratory data distribution, enabling the power of model extrapolation. In practice, it can be solved through a computationally efficient, penalized adversarial estimation procedure. Our theoretical regret guarantees do not rely on any data-coverage and completeness-type assumptions, only requiring realizability. These guarantees also demonstrate that the learned policy represents the ``best effort'' among all policies, as no other policies can outperform it. We evaluate our model using a blend of synthetic, benchmark, and real-world datasets for offline RL, showing that it performs competitively with state-of-the-art methods.</details>

### [Spotlight] Bifurcations and loss jumps in RNN training

**Authors:** Lukas Eisenmann, Zahra Monfared, Niclas Göring, Daniel Durstewitz

**<details><summary>Abstract**</summary>

Recurrent neural networks (RNNs) are popular machine learning tools for modeling and forecasting sequential data and for inferring dynamical systems (DS) from observed time series. Concepts from DS theory (DST) have variously been used to further our understanding of both, how trained RNNs solve complex tasks, and the training process itself. Bifurcations are particularly important phenomena in DS, including RNNs, that refer to topological (qualitative) changes in a system's dynamical behavior as one or more of its parameters are varied. Knowing the bifurcation structure of an RNN will thus allow to deduce many of its computational and dynamical properties, like its sensitivity to parameter variations or its behavior during training. In particular, bifurcations may account for sudden loss jumps observed in RNN training that could severely impede the training process. Here we first mathematically prove for a particular class of ReLU-based RNNs that certain bifurcations are indeed associated with loss gradients tending toward infinity or zero. We then introduce a novel heuristic algorithm for detecting all fixed points and $k$-cycles in ReLU-based RNNs and their existence and stability regions, hence bifurcation manifolds in parameter space. In contrast to previous numerical algorithms for finding fixed points and common continuation methods, our algorithm provides $\textit{exact}$ results and returns fixed points and cycles up to high orders with surprisingly good scaling behavior. We exemplify the algorithm on the analysis of the training process of RNNs, and find that the recently introduced technique of generalized teacher forcing completely avoids certain types of bifurcations in training. Thus, besides facilitating the DST analysis of trained RNNs, our algorithm provides a powerful instrument for analyzing the training process itself.</details>

### Birder: Communication-Efficient 1-bit Adaptive Optimizer for Practical Distributed DNN Training

**Authors:** Hanyang Peng, Shuang Qin, Yue Yu, Jin Wang, Hui Wang, Ge Li

**<details><summary>Abstract**</summary>

Various gradient compression algorithms have been proposed to alleviate the communication bottleneck in distributed learning, and they have demonstrated effectiveness in terms of high compression ratios and theoretical low communication complexity. However, when it comes to practically training modern deep neural networks (DNNs), these algorithms have yet to match the inference performance of uncompressed SGD-momentum (SGDM) and adaptive optimizers (e.g.,Adam). More importantly, recent studies suggest that these algorithms actually offer no speed advantages over SGDM/Adam when used with common distributed DNN training frameworks ( e.g., DistributedDataParallel (DDP)) in the typical settings, due to heavy compression/decompression computation or incompatibility with the efficient All-Reduce or the requirement of uncompressed warmup at the early stage. For these reasons, we propose a novel 1-bit adaptive optimizer, dubbed *Bi*nary *r*andomization a*d*aptive optimiz*er* (**Birder**). The quantization of Birder can be easily and lightly computed, and it does not require warmup with its uncompressed version in the beginning. Also, we devise Hierarchical-1-bit-All-Reduce to further lower the communication volume. We theoretically prove that it promises the same convergence rate as the Adam. Extensive experiments, conducted on 8 to 64 GPUs (1 to 8 nodes) using DDP, demonstrate that Birder achieves comparable inference performance to uncompressed SGDM/Adam, with up to ${2.5 	imes}$ speedup for training ResNet-50 and ${6.3	imes}$ speedup for training BERT-Base. Code is publicly available at https://openi.pcl.ac.cn/c2net_optim/Birder.</details>

### Block Coordinate Plug-and-Play Methods for Blind Inverse Problems

**Authors:** Weijie Gan, shirin shoushtari, Yuyang Hu, Jiaming Liu, Hongyu An, Ulugbek Kamilov

**<details><summary>Abstract**</summary>

Plug-and-play (PnP) prior is a well-known class of methods for solving imaging inverse problems by computing fixed-points of operators combining physical measurement models and learned image denoisers. While PnP methods have been extensively used for image recovery with known measurement operators, there is little work on PnP for solving blind inverse problems. We address this gap by presenting a new block-coordinate PnP (BC-PnP) method that efficiently solves this joint estimation problem by introducing learned denoisers as priors on both the unknown image and the unknown measurement operator. We present a new convergence theory for BC-PnP compatible with blind inverse problems by considering nonconvex data-fidelity terms and expansive denoisers. Our theory analyzes the convergence of BC-PnP to a stationary point of an implicit function associated with an approximate minimum mean-squared error (MMSE) denoiser. We numerically validate our method on two blind inverse problems: automatic coil sensitivity estimation in magnetic resonance imaging (MRI) and blind image deblurring. Our results show that BC-PnP provides an efficient and principled framework for using denoisers as PnP priors for jointly estimating measurement operators and images.</details>

### Block Low-Rank Preconditioner with Shared Basis for Stochastic Optimization

**Authors:** Jui-Nan Yen, Sai Surya Duvvuri, Inderjit Dhillon, Cho-Jui Hsieh

**<details><summary>Abstract**</summary>

Adaptive methods with non-diagonal preconditioning have shown state-of-the-art results on various tasks. However, their computational complexity and memory requirement makes it challenging to scale these methods to modern neural network architectures. To address this challenge, some previous works have adopted block-diagonal preconditioners. However, the memory cost of storing the block-diagonal matrix remains substantial, leading to the use of smaller block sizes and ultimately resulting in suboptimal performance. To reduce the time and memory complexity without sacrificing performance, we propose approximating each diagonal block of the second moment matrix by low-rank matrices and enforcing the same basis for the blocks within each layer.  We provide theoretical justification for such sharing and design an algorithm to efficiently maintain this shared-basis block low-rank approximation during training. Our results on a deep autoencoder and a transformer benchmark demonstrate that the proposed method outperforms first-order methods with slightly more time and memory usage, while also achieving competitive or superior performance compared to other second-order methods with less time and memory usage.</details>

### Blocked Collaborative Bandits: Online Collaborative Filtering with Per-Item Budget Constraints

**Authors:** Soumyabrata Pal, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**<details><summary>Abstract**</summary>

We consider the problem of \emph{blocked} collaborative bandits where there are multiple users, each with an associated multi-armed bandit problem. These users are grouped into \emph{latent} clusters such that the mean reward vectors of users within the same cluster are identical. Our goal is to design algorithms that maximize the cumulative reward accrued by all the users over time, under the \emph{constraint} that no arm of a user is pulled more than $\mathsf{B}$ times. This problem has been originally considered by \cite{Bresler:2014}, and designing regret-optimal algorithms for it has since remained an open problem.In this work, we propose an algorithm called B-LATTICE (Blocked Latent bAndiTs via maTrIx ComplEtion) that collaborates across users, while simultaneously satisfying the budget constraints, to maximize their cumulative rewards. Theoretically, under certain reasonable assumptions on the latent structure, with $\mathsf{M}$ users, $\mathsf{N}$ arms, $\mathsf{T}$ rounds per user, and $\mathsf{C}=O(1)$ latent clusters, B-LATTICE achieves a per-user regret of  $\widetilde{O}(\sqrt{\mathsf{T}(1 + \mathsf{N}\mathsf{M}^{-1})})$ under a budget constraint of $\mathsf{B}=\Theta(\log \mathsf{T})$. These are the first sub-linear regret bounds for this problem, and match the minimax regret bounds when $\mathsf{B}=\mathsf{T}$. Empirically, we demonstrate that our algorithm has superior performance over baselines even when $\mathsf{B}=1$. B-LATTICE is a phased algorithm where in each phase it clusters users into groups and collaborates across users within a group to quickly learn their reward models.</details>

### [Spotlight] Blockwise Parallel Transformers for Large Context Models

**Authors:** Hao Liu, Pieter Abbeel

**<details><summary>Abstract**</summary>

Transformers have emerged as the cornerstone of state-of-the-art natural language processing models, showcasing exceptional performance across a wide range of AI applications. However, the memory demands posed by the self-attention mechanism and the large feedforward network in Transformers limit their ability to handle long sequences, thereby creating challenges for tasks involving multiple long sequences or long-term dependencies. We present a distinct approach, Blockwise Parallel Transformer (BPT), that leverages blockwise computation of self-attention and feedforward network fusion to minimize memory costs. By processing longer input sequences while maintaining memory efficiency, BPT enables training sequences 32 times longer than vanilla Transformers and up to 4 times longer than previous memory-efficient methods. Extensive experiments on language modeling and reinforcement learning tasks demonstrate the effectiveness of BPT in reducing memory requirements and improving performance.</details>

### Boundary Guided Learning-Free Semantic Control with Diffusion Models

**Authors:** Ye Zhu, Yu Wu, Zhiwei Deng, Olga Russakovsky, Yan Yan

**<details><summary>Abstract**</summary>

Applying pre-trained generative denoising diffusion models (DDMs) for downstream tasks such as image semantic editing usually requires either fine-tuning DDMs or learning auxiliary editing networks in the existing literature. In this work, we present our BoundaryDiffusion method for efficient, effective and light-weight semantic control with frozen pre-trained DDMs, without learning any extra networks. As one of the first learning-free diffusion editing works, we start by seeking a more comprehensive understanding of the intermediate high-dimensional latent spaces by theoretically and empirically analyzing their probabilistic and geometric behaviors in the Markov chain. We then propose to further explore the critical step in the denoising trajectory that characterizes the convergence of a pre-trained DDM and introduce an automatic search method. Last but not least, in contrast to the conventional understanding that DDMs have relatively poor semantic behaviors (in generic latent spaces), we prove that the critical latent space we found already forms semantic subspace boundaries at the generic level in unconditional DDMs, which allows us to do controllable manipulation by guiding the denoising trajectory towards the targeted boundary via a single-step operation. We conduct extensive experiments on multiple DPMs architectures (DDPM, iDDPM) and datasets (CelebA, CelebA-HQ, LSUN-church, LSUN-bedroom, AFHQ-dog) with different resolutions (64, 256), achieving superior or state-of-the-art performance in various task scenarios (image semantic editing, text-based editing, unconditional semantic control) to demonstrate the effectiveness.</details>

### [Oral] Brain Diffusion for Visual Exploration: Cortical Discovery using Large Scale Generative Models

**Authors:** Andrew Luo, Maggie Henderson, Leila Wehbe, Michael Tarr

**Oral Presentation:** We, Dec 13, 08:00 -- Oral 3A

**<details><summary>Abstract**</summary>

A long standing goal in neuroscience has been to elucidate the functional organization of the brain. Within higher visual cortex, functional accounts have remained relatively coarse, focusing on regions of interest (ROIs) and taking the form of selectivity for broad categories such as faces, places, bodies, food, or words. Because the identification of such ROIs has typically relied on manually assembled stimulus sets consisting of isolated objects in non-ecological contexts, exploring functional organization without robust a priori hypotheses has been challenging. To overcome these limitations, we introduce a data-driven approach in which we synthesize images predicted to activate a given brain region using paired natural images and fMRI recordings, bypassing the need for category-specific stimuli. Our approach -- Brain Diffusion for Visual Exploration ("BrainDiVE") -- builds on recent generative methods by combining large-scale diffusion models with brain-guided image synthesis. Validating our method, we demonstrate the ability to synthesize preferred images with appropriate semantic specificity for well-characterized category-selective ROIs. We then show that BrainDiVE can characterize differences between ROIs selective for the same high-level category. Finally we identify novel functional subdivisions within these ROIs, validated with behavioral data. These  results advance our understanding of the fine-grained functional organization of human visual cortex, and provide well-specified constraints for further examination of cortical organization using hypothesis-driven methods.</details>

### Brain-like Flexible Visual Inference by Harnessing Feedback Feedforward Alignment

**Authors:** Tahereh Toosi, Elias Issa

**<details><summary>Abstract**</summary>

In natural vision, feedback connections support versatile visual inference capabilities such as making sense of the occluded or noisy bottom-up sensory information or mediating pure top-down processes such as imagination. However, the mechanisms by which the feedback pathway learns to give rise to these capabilities flexibly are not clear. We propose that top-down effects emerge through alignment between feedforward and feedback pathways, each optimizing its own objectives. To achieve this co-optimization, we introduce Feedback-Feedforward Alignment (FFA), a learning algorithm that leverages feedback and feedforward pathways as mutual credit assignment computational graphs, enabling alignment. In our study, we demonstrate the effectiveness of FFA in co-optimizing classification and reconstruction tasks on widely used MNIST and CIFAR10 datasets. Notably, the alignment mechanism in FFA endows feedback connections with emergent visual inference functions, including denoising, resolving occlusions, hallucination, and imagination. Moreover, FFA offers bio-plausibility compared to traditional backpropagation (BP) methods in implementation. By repurposing the computational graph of credit assignment into a goal-driven feedback pathway, FFA alleviates weight transport problems encountered in BP, enhancing the bio-plausibility of the learning algorithm. Our study presents FFA as a promising proof-of-concept for the mechanisms underlying how feedback connections in the visual cortex support flexible visual functions. This work also contributes to the broader field of visual inference underlying perceptual phenomena and has implications for developing more biologically inspired learning algorithms.</details>

### Brant: Foundation Model for Intracranial Neural Signal

**Authors:** Daoze Zhang, Zhizhang Yuan, YANG YANG, Junru Chen, Jingjing Wang, Yafeng Li

**<details><summary>Abstract**</summary>

We propose a foundation model named Brant for modeling intracranial recordings, which learns powerful representations of intracranial neural signals by pre-training, providing a large-scale, off-the-shelf model for medicine. Brant is the largest model in the field of brain signals and is pre-trained on a large corpus of intracranial data collected by us. The design of Brant is to capture long-term temporal dependency and spatial correlation from neural signals, combining the information in both time and frequency domains. As a foundation model, Brant achieves SOTA performance on various downstream tasks (i.e. neural signal forecasting, frequency-phase forecasting, imputation and seizure detection), showing the generalization ability to a broad range of tasks. The low-resource label analysis and representation visualization further illustrate the effectiveness of our pre-training strategy. In addition, we explore the effect of model size to show that a larger model with a higher capacity can lead to performance improvements on our dataset. The source code and pre-trained weights are available at: https://zju-brainnet.github.io/Brant.github.io/.</details>

### Bridging the Domain Gap: Self-Supervised 3D Scene Understanding with Foundation Models

**Authors:** Zhimin Chen, Longlong Jing, Yingwei Li, Bing Li

**<details><summary>Abstract**</summary>

Foundation models have achieved remarkable results in 2D and language tasks like image segmentation, object detection, and visual-language understanding. However, their potential to enrich 3D scene representation learning is largely untapped due to the existence of the domain gap. In this work, we propose an innovative methodology called Bridge3D to address this gap by pre-training 3D models using features, semantic masks, and captions sourced from foundation models. Specifically, our method employs semantic masks from foundation models to guide the masking and reconstruction process for the masked autoencoder, enabling more focused attention on foreground representations. Moreover, we bridge the 3D-text gap at the scene level using image captioning foundation models, thereby facilitating scene-level knowledge distillation. We further extend this bridging effort by introducing an innovative object-level knowledge distillation method that harnesses highly accurate object-level masks and semantic text data from foundation models. Our methodology significantly surpasses the performance of existing state-of-the-art methods in 3D object detection and semantic segmentation tasks. For instance, on the ScanNet dataset, Bridge3D improves the baseline by a notable margin of 6.3%. Code will be available at: https://github.com/Zhimin-C/Bridge3D</details>

### Bucks for Buckets (B4B): Active Defenses Against Stealing Encoders

**Authors:** Jan Dubiński, Stanisław Pawlak, Franziska Boenisch, Tomasz Trzcinski, Adam Dziedzic

**<details><summary>Abstract**</summary>

Machine Learning as a Service (MLaaS) APIs provide ready-to-use and high-utility encoders that generate vector representations for given inputs. Since these encoders are very costly to train, they become lucrative targets for model stealing attacks during which an adversary leverages query access to the API to replicate the encoder locally at a fraction of the original training costs. We propose *Bucks for Buckets (B4B)*, the first *active defense* that prevents stealing while the attack is happening without degrading representation quality for legitimate API users. Our defense relies on the observation that the representations returned to adversaries who try to steal the encoder's functionality cover a significantly larger fraction of the embedding space than representations of legitimate users who utilize the encoder to solve a particular downstream task. B4B leverages this to adaptively adjust the utility of the returned representations according to a user's coverage of the embedding space. To prevent adaptive adversaries from eluding our defense by simply creating multiple user accounts (sybils), B4B also individually transforms each user's representations. This prevents the adversary from directly aggregating representations over multiple accounts to create their stolen encoder copy. Our active defense opens a new path towards securely sharing and democratizing encoders over public APIs.</details>

### Byzantine-Tolerant Methods for Distributed Variational Inequalities

**Authors:** Nazarii Tupitsa, Abdulla Jasem Almansoori, Yanlin Wu, Martin Takac, Karthik Nandakumar, Samuel Horváth, Eduard Gorbunov

**<details><summary>Abstract**</summary>

Robustness to Byzantine attacks is a necessity for various distributed training scenarios. When the training reduces to the process of solving a minimization problem, Byzantine robustness is relatively well-understood. However, other problem formulations, such as min-max problems or, more generally, variational inequalities, arise in many modern machine learning and, in particular, distributed learning tasks. These problems significantly differ from the standard minimization ones and, therefore, require separate consideration. Nevertheless, only one work [Abidi et al., 2022] addresses this important question in the context of Byzantine robustness. Our work makes a further step in this direction by providing several (provably) Byzantine-robust methods for distributed variational inequality, thoroughly studying their theoretical convergence, removing the limitations of the previous work, and providing numerical comparisons supporting the theoretical findings.</details>

### C-Disentanglement: Discovering Causally-Independent Generative Factors under  an Inductive Bias of Confounder

**Authors:** Xiaoyu Liu, Jiaxin Yuan, Bang An, Yuancheng Xu, Yifan Yang, Furong Huang

**<details><summary>Abstract**</summary>

Representation learning assumes that real-world data is generated by a few semantically meaningful generative factors (i.e., sources of variation) and aims to discover them in the latent space. These factors are expected to be causally disentangled, meaning that distinct factors are encoded into separate latent variables, and changes in one factor will not affect the values of the others. Compared to statistical independence, causal disentanglement allows more controllable data generation, improved robustness, and better generalization. However, most existing work assumes unconfoundedness in the discovery process, that there are no common causes to the generative factors and thus obtain only statistical independence. In this paper, we recognize the importance of modeling confounders in discovering causal generative factors. Unfortunately, such factors are not identifiable without proper inductive bias. We fill the gap by introducing a framework entitled Confounded-Disentanglement (C-Disentanglement), the first framework that explicitly introduces the inductive bias of confounder via labels from domain expertise. In addition, we accordingly propose an approach to sufficiently identify the causally-disentangled factors under any inductive bias of the confounder.  We conduct extensive experiments on both synthetic and real-world datasets. Our method demonstrates competitive results compared to various SOTA baselines in obtaining causally disentangled features and downstream tasks under domain shifts.</details>

### CADet: Fully Self-Supervised Out-Of-Distribution Detection With Contrastive Learning

**Authors:** Charles Guille-Escuret, Pau Rodriguez, David Vazquez, Ioannis Mitliagkas, Joao Monteiro

**<details><summary>Abstract**</summary>

Handling out-of-distribution (OOD) samples has become a major stake in the real-world deployment of machine learning systems. This work explores the use of self-supervised contrastive learning to the simultaneous detection of two types of OOD samples: unseen classes and adversarial perturbations. First, we pair self-supervised contrastive learning with the maximum mean discrepancy (MMD) two-sample test. This approach enables us to robustly test whether two independent sets of samples originate from the same distribution, and we demonstrate its effectiveness by discriminating between CIFAR-10 and CIFAR-10.1 with higher confidence than previous work. Motivated by this success, we introduce CADet (Contrastive Anomaly Detection), a novel method for OOD detection of single samples. CADet draws inspiration from MMD, but leverages the similarity between contrastive transformations of a same sample. CADet outperforms existing adversarial detection methods in identifying adversarially perturbed samples on ImageNet and achieves comparable performance to unseen label detection methods on two challenging benchmarks: ImageNet-O and iNaturalist. Significantly, CADet is fully self-supervised and requires neither labels for in-distribution samples nor access to OOD examples.</details>

### CBD: A Certified Backdoor Detector Based on Local Dominant Probability

**Authors:** Zhen Xiang, Zidi Xiong, Bo Li

**<details><summary>Abstract**</summary>

Backdoor attack is a common threat to deep neural networks. During testing, samples embedded with a backdoor trigger will be misclassified as an adversarial target by a backdoored model, while samples without the backdoor trigger will be correctly classified. In this paper, we present the first certified backdoor detector (CBD), which is based on a novel, adjustable conformal prediction scheme based on our proposed statistic local dominant probability. For any classifier under inspection, CBD provides 1) a detection inference, 2) the condition under which the attacks are guaranteed to be detectable for the same classification domain, and 3) a probabilistic upper bound for the false positive rate. Our theoretical results show that attacks with triggers that are more resilient to test-time noise and have smaller perturbation magnitudes are more likely to be detected with guarantees. Moreover, we conduct extensive experiments on four benchmark datasets considering various backdoor types, such as BadNet, CB, and Blend. CBD achieves comparable or even higher detection accuracy than state-of-the-art detectors, and it in addition provides detection certification. Notably, for backdoor attacks with random perturbation triggers bounded by $\ell_2\leq0.75$ which achieves more than 90\% attack success rate, CBD achieves 100\% (98\%), 100\% (84\%), 98\% (98\%), and 72\% (40\%) empirical (certified) detection true positive rates on the four benchmark datasets GTSRB, SVHN, CIFAR-10, and TinyImageNet, respectively, with low false positive rates.</details>

### CD-GraB: Coordinating Distributed Example Orders for Provably Accelerated Training

**Authors:** A. Feder Cooper, Wentao Guo, Duc Khiem Pham, Tiancheng Yuan, Charlie Ruan, Yucheng Lu, Christopher De Sa

**<details><summary>Abstract**</summary>

Recent research on online Gradient Balancing (GraB) has revealed that there exist permutation-based example orderings that are guaranteed to outperform random reshuffling (RR). Whereas RR arbitrarily permutes training examples, GraB leverages stale gradients from prior epochs to order examples -- achieving a provably faster convergence rate than RR. However, GraB is limited by design: While it demonstrates an impressive ability to scale-up training on centralized data, it does not naturally extend to modern distributed ML workloads. We therefore propose Coordinated Distributed GraB (CD-GraB), which uses insights from prior work on kernel thinning to translate the benefits of provably faster permutation-based example ordering to distributed settings. With negligible overhead, CD-GraB exhibits a linear speedup in convergence rate over centralized GraB and outperforms baselines empirically, including distributed RR, on a variety of benchmark tasks.</details>

### [Spotlight] CODA: Generalizing to Open and Unseen Domains with Compaction and Disambiguation

**Authors:** Chaoqi Chen, Luyao Tang, Yue Huang, Xiaoguang Han, Yizhou Yu

**<details><summary>Abstract**</summary>

The generalization capability of machine learning systems degenerates notably when the test distribution drifts from the training distribution. Recently, Domain Generalization (DG) has been gaining momentum in enabling machine learning models to generalize to unseen domains. However, most DG methods assume that training and test data share an identical label space, ignoring the potential unseen categories in many real-world applications. In this paper, we delve into a more general but difficult problem termed Open Test-Time DG (OTDG), where both domain shift and open class may occur on the unseen test data. We propose Compaction and Disambiguation (CODA), a novel two-stage framework for learning compact representations and adapting to open classes in the wild. To meaningfully regularize the model's decision boundary, CODA introduces virtual unknown classes and optimizes a new training objective to insert unknowns into the latent space by compacting the embedding space of source known classes. To adapt target samples to the source model, we then disambiguate the decision boundaries between known and unknown classes with a test-time training objective, mitigating the adaptivity gap and catastrophic forgetting challenges. Experiments reveal that CODA can significantly outperform the previous best method on standard DG datasets and harmonize the classification accuracy between known and unknown classes.</details>

### CORL: Research-oriented Deep Offline Reinforcement Learning Library

**Authors:** Denis Tarasov, Alexander Nikulin, Dmitry Akimov, Vladislav Kurenkov, Sergey Kolesnikov

**<details><summary>Abstract**</summary>

CORL is an open-source library that provides thoroughly benchmarked single-file implementations of both deep offline and offline-to-online reinforcement learning algorithms. It emphasizes a simple developing experience with a straightforward codebase and a modern analysis tracking tool. In CORL, we isolate methods implementation into separate single files, making performance-relevant details easier to recognize. Additionally, an experiment tracking feature is available to help log metrics, hyperparameters, dependencies, and more to the cloud. Finally, we have ensured the reliability of the implementations by benchmarking commonly employed D4RL datasets providing a transparent source of results that can be reused for robust evaluation tools such as performance profiles, probability of improvement, or expected online performance.</details>

### CS-Isolate: Extracting Hard Confident Examples by Content and Style Isolation

**Authors:** Yexiong Lin, Yu Yao, Xiaolong Shi, Mingming Gong, Xu Shen, Dong Xu, Tongliang Liu

**<details><summary>Abstract**</summary>

Label noise widely exists in large-scale image datasets. To mitigate the side effects of label noise, state-of-the-art methods focus on selecting confident examples by leveraging semi-supervised learning. Existing research shows that the ability to extract hard confident examples, which are close to the decision boundary, significantly influences the generalization ability of the learned classifier.In this paper, we find that a key reason for some hard examples being close to the decision boundary is due to the entanglement of style factors with content factors. The hard examples become more discriminative when we focus solely on content factors, such as semantic information, while ignoring style factors. Nonetheless, given only noisy data, content factors are not directly observed and have to be inferred.To tackle the problem of inferring content factors for classification when learning with noisy labels, our objective is to ensure that the content factors of all examples in the same underlying clean class remain unchanged as their style information changes.To achieve this, we utilize different data augmentation techniques to alter the styles while regularizing content factors based on some confident examples. By training existing methods with our inferred content factors, CS-Isolate proves their effectiveness in learning hard examples on benchmark datasets. The implementation is available at https://github.com/tmllab/2023_NeurIPS_CS-isolate.</details>

### CSLP-AE: A Contrastive Split-Latent Permutation Autoencoder Framework for Zero-Shot Electroencephalography Signal Conversion

**Authors:** Anders Nørskov, Alexander Neergaard Zahid, Morten Mørup

**<details><summary>Abstract**</summary>

Electroencephalography (EEG) is a prominent non-invasive neuroimaging technique providing insights into brain function. Unfortunately, EEG data exhibit a high degree of noise and variability across subjects hampering generalizable signal extraction. Therefore, a key aim in EEG analysis is to extract the underlying neural activation (content) as well as to account for the individual subject variability (style). We hypothesize that the ability to convert EEG signals between tasks and subjects requires the extraction of latent representations accounting for content and style. Inspired by recent advancements in voice conversion technologies, we propose a novel contrastive split-latent permutation autoencoder (CSLP-AE) framework that directly optimizes for EEG conversion. Importantly, the latent representations are guided using contrastive learning to promote the latent splits to explicitly represent subject (style) and task (content). We contrast CSLP-AE to conventional supervised, unsupervised (AE), and self-supervised (contrastive learning) training and find that the proposed approach provides favorable generalizable characterizations of subject and task. Importantly, the procedure also enables zero-shot conversion between unseen subjects. While the present work only considers conversion of EEG, the proposed CSLP-AE provides a general framework for signal conversion and extraction of content (task activation) and style (subject variability) components of general interest for the modeling and analysis of biological signals.</details>

### Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning

**Authors:** Mitsuhiko Nakamoto, Simon Zhai, Anikait Singh, Max Sobol Mark, Yi Ma, Chelsea Finn, Aviral Kumar, Sergey Levine

**<details><summary>Abstract**</summary>

A compelling use case of offline reinforcement learning (RL) is to obtain a policy initialization from existing datasets followed by fast online fine-tuning with limited interaction. However, existing offline RL methods tend to behave poorly during fine-tuning. In this paper, we devise an approach for learning an effective initialization from offline data that also enables fast online fine-tuning capabilities. Our approach, calibrated Q-learning (Cal-QL), accomplishes this by learning a conservative value function initialization that underestimates the value of the learned policy from offline data, while also being calibrated, in the sense that the learned Q-values are at a reasonable scale. We refer to this property as calibration, and define it formally as providing a lower bound on the true value function of the learned policy and an upper bound on the value of some other (suboptimal) reference policy, which may simply be the behavior policy. We show that offline RL algorithms that learn such calibrated value functions lead to effective online fine-tuning, enabling us to take the benefits of offline initializations in online fine-tuning. In practice, Cal-QL can be implemented on top of the conservative Q learning (CQL) for offline RL within a one-line code change. Empirically, Cal-QL outperforms state-of-the-art methods on 9/11 fine-tuning benchmark tasks that we study in this paper. Code and video are available at https://nakamotoo.github.io/Cal-QL</details>

### Calibration by Distribution Matching: Trainable Kernel Calibration Metrics

**Authors:** Charlie Marx, Sofian Zalouk, Stefano Ermon

**<details><summary>Abstract**</summary>

Calibration ensures that probabilistic forecasts meaningfully capture uncertainty by requiring that predicted probabilities align with empirical frequencies. However, many existing calibration methods are specialized for post-hoc recalibration, which can worsen the sharpness of forecasts. Drawing on the insight that calibration can be viewed as a distribution matching task, we introduce kernel-based calibration metrics that unify and generalize popular forms of calibration for both classification and regression. These metrics admit differentiable sample estimates, making it easy to incorporate a calibration objective into empirical risk minimization. Furthermore, we provide intuitive mechanisms to tailor calibration metrics to a decision task, and enforce accurate loss estimation and no regret decisions. Our empirical evaluation demonstrates that employing these metrics as regularizers enhances calibration, sharpness, and decision-making across a range of regression and classification tasks, outperforming methods relying solely on post-hoc recalibration.</details>

### Can Language Models Teach? Teacher Explanations Improve Student Performance via Personalization

**Authors:** Swarnadeep Saha, Peter Hase, Mohit Bansal

**<details><summary>Abstract**</summary>

A hallmark property of explainable AI models is the ability to teach other agents, communicating knowledge of how to perform a task. While Large Language Models (LLMs) perform complex reasoning by generating explanations for their predictions, it is unclear whether they also make good teachers for weaker agents. To address this, we consider a student-teacher framework between two LLM agents and study if, when, and how the teacher should intervene with natural language explanations to improve the student’s performance. Since communication is expensive, we define a budget such that the teacher only communicates explanations for a fraction of the data, after which the student should perform well on its own. We decompose the teaching problem along four axes: (1) if teacher’s test time in- tervention improve student predictions, (2) when it is worth explaining a data point, (3) how the teacher should personalize explanations to better teach the student, and (4) if teacher explanations also improve student performance on future unexplained data. We first show that teacher LLMs can indeed intervene on student reasoning to improve their performance. Next, inspired by the Theory of Mind abilities of effective teachers, we propose building two few-shot mental models of the student. The first model defines an Intervention Function that simulates the utility of an intervention, allowing the teacher to intervene when this utility is the highest and improving student performance at lower budgets. The second model enables the teacher to personalize explanations for a particular student and outperform unpersonalized teachers. We also demonstrate that in multi-turn interactions, teacher explanations generalize and learning from explained data improves student performance on future unexplained data. Finally, we also verify that misaligned teachers can lower student performance to random chance by intentionally misleading them.</details>

### Cascading Bandits: Optimizing Recommendation Frequency in Delayed Feedback Environments

**Authors:** Dairui Wang, Junyu Cao, Yan Zhang, Wei Qi

**<details><summary>Abstract**</summary>

Delayed feedback is a critical problem in dynamic recommender systems. In practice, the feedback result often depends on the frequency of recommendation. Most existing online learning literature fails to consider optimization of the recommendation frequency, and regards the reward from each successfully recommended message to be equal. In this paper, we consider a novel cascading bandits setting, where individual messages from a selected list are sent to a user periodically. Whenever a user does not like a message, she may abandon the system with a probability positively correlated with the recommendation frequency.  A learning agent needs to learn both the underlying message attraction probabilities and users' abandonment probabilities through the randomly delayed feedback. We first show a dynamic programming solution to finding the optimal message sequence in deterministic scenarios, in which the reward is allowed to vary with different messages. Then we propose a polynomial time UCB-based offline learning algorithm, and discuss its performance by characterizing its regret bound. For the online setting, we propose a learning algorithm which allows adaptive content for a given user. Numerical experiment on AmEx dataset confirms the effectiveness of our algorithms.</details>

### Causal Component Analysis

**Authors:** Liang Wendong, Armin Kekić, Julius von Kügelgen, Simon Buchholz, Michel Besserve, Luigi Gresele, Bernhard Schölkopf

**<details><summary>Abstract**</summary>

Independent Component Analysis (ICA) aims to recover independent latent variables from observed mixtures thereof. Causal Representation Learning (CRL) aims instead to infer causally related (thus often statistically _dependent_) latent variables, together with the unknown graph encoding their causal relationships. We introduce an intermediate problem termed _Causal Component Analysis (CauCA)_. CauCA can be viewed as a generalization of ICA, modelling the causal dependence among the latent components, and as a special case of CRL. In contrast to CRL, it presupposes knowledge of the causal graph, focusing solely on learning the unmixing function and the causal mechanisms. Any impossibility results regarding the recovery of the ground truth in CauCA also apply for CRL, while possibility results may serve as a stepping stone for extensions to CRL. We characterize CauCA identifiability from multiple datasets generated through different types of interventions on the latent causal variables. As a corollary, this interventional perspective also leads to new identifiability results for nonlinear ICA—a special case of CauCA with an empty graph—requiring strictly fewer datasets than previous results. We introduce a likelihood-based approach using normalizing flows to estimate both the unmixing function and the causal mechanisms, and demonstrate its effectiveness through extensive synthetic experiments in the CauCA and ICA setting.</details>

### Causal Interpretation of Self-Attention in Pre-Trained Transformers

**Authors:** Raanan Rohekar, Yaniv Gurwicz, Shami Nisimov

**<details><summary>Abstract**</summary>

We propose a causal interpretation of self-attention in the Transformer neural network architecture. We interpret self-attention as a mechanism that estimates a structural equation model for a given input sequence of symbols (tokens). The structural equation model can be interpreted, in turn, as a causal structure over the input symbols under the specific context of the input sequence. Importantly, this interpretation remains valid in the presence of latent confounders. Following this interpretation, we estimate conditional independence relations between input symbols by calculating partial correlations between their corresponding representations in the deepest attention layer. This enables learning the causal structure over an input sequence using existing constraint-based algorithms. In this sense, existing pre-trained Transformers can be utilized for zero-shot causal-discovery. We demonstrate this method by providing causal explanations for the outcomes of Transformers in two tasks: sentiment classification (NLP) and recommendation.</details>

### Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization

**Authors:** Mahyar Fazlyab, Taha Entesari, Aniket Roy, Rama Chellappa

**<details><summary>Abstract**</summary>

To improve the robustness of deep classifiers against adversarial perturbations, many approaches have been proposed, such as designing new architectures with better robustness properties (e.g., Lipschitz-capped networks), or modifying the training process itself (e.g., min-max optimization, constrained learning, or regularization). These approaches, however, might not be effective at increasing the margin in the input (feature) space. In this paper, we propose a differentiable regularizer that is a lower bound on the distance of the data points to the classification boundary. The proposed regularizer requires knowledge of the model's Lipschitz constant along certain directions. To this end, we develop a scalable method for calculating guaranteed differentiable upper bounds on the Lipschitz constant of neural networks accurately and efficiently.  The relative accuracy of the bounds prevents excessive regularization and allows for more direct manipulation of the decision boundary. Furthermore, our Lipschitz bounding algorithm exploits the monotonicity and Lipschitz continuity of the activation layers, and the resulting bounds can be used to design new layers with controllable bounds on their Lipschitz constant. Experiments on the MNIST, CIFAR-10, and Tiny-ImageNet data sets verify that our proposed algorithm obtains competitively improved results compared to the state-of-the-art.</details>

### Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models

**Authors:** Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Jianfeng Gao

**<details><summary>Abstract**</summary>

Large language models (LLMs) have achieved remarkable progress in solving various natural language processing tasks due to emergent reasoning abilities. However, LLMs have inherent limitations as they are incapable of accessing up-to-date information (stored on the Web or in task-specific knowledge bases), using external tools, and performing precise mathematical and logical reasoning. In this paper, we present Chameleon, an AI system that mitigates these limitations by augmenting LLMs with plug-and-play modules for compositional reasoning. Chameleon synthesizes programs by composing various tools (e.g., LLMs, off-the-shelf vision models, web search engines, Python functions, and heuristic-based modules) for accomplishing complex reasoning tasks. At the heart of Chameleon is an LLM-based planner that assembles a sequence of tools to execute to generate the final response. We showcase the effectiveness of Chameleon on two multi-modal knowledge-intensive reasoning tasks: ScienceQA and TabMWP. Chameleon, powered by GPT-4, achieves an 86.54% overall accuracy on ScienceQA, improving the best published few-shot result by 11.37%. On TabMWP, GPT-4-powered Chameleon improves the accuracy by 17.0%, lifting the state of the art to 98.78%. Our analysis also shows that the GPT-4-powered planner exhibits more consistent and rational tool selection via inferring potential constraints from instructions, compared to a ChatGPT-powered planner.</details>

### Chasing Fairness Under Distribution Shift: A Model Weight Perturbation Approach

**Authors:** Zhimeng Jiang, Xiaotian Han, Hongye Jin, Guanchu Wang, Rui Chen, Na Zou, Xia Hu

**<details><summary>Abstract**</summary>

Fairness in machine learning has attracted increasing attention in recent years. The fairness methods improving algorithmic fairness for in-distribution data may not perform well under distribution shifts. In this paper, we first theoretically demonstrate the inherent connection between distribution shift, data perturbation, and model weight perturbation.Subsequently, we analyze the sufficient conditions to guarantee fairness (i.e., low demographic parity) for the target dataset, including fairness for the source dataset, and low prediction difference between the source and target datasets for each sensitive attribute group. Motivated by these sufficient conditions, we propose robust fairness regularization (RFR) by considering the worst case within the model weight perturbation ball for each sensitive attribute group. We evaluate the effectiveness of our proposed RFR algorithm on synthetic and real distribution shifts across various datasets. Experimental results demonstrate that RFR achieves better fairness-accuracy trade-off performance compared with several baselines. The source code is available at \url{https://github.com/zhimengj0326/RFR_NeurIPS23}.</details>

### ChatGPT-Powered Hierarchical Comparisons for Image Classification

**Authors:** Zhiyuan Ren, Yiyang Su, Xiaoming Liu

**<details><summary>Abstract**</summary>

The zero-shot open-vocabulary setting poses challenges for image classification.Fortunately, utilizing a vision-language model like CLIP, pre-trained on image-textpairs, allows for classifying images by comparing embeddings. Leveraging largelanguage models (LLMs) such as ChatGPT can further enhance CLIP’s accuracyby incorporating class-specific knowledge in descriptions. However, CLIP stillexhibits a bias towards certain classes and generates similar descriptions for similarclasses, disregarding their differences. To address this problem, we present anovel image classification framework via hierarchical comparisons. By recursivelycomparing and grouping classes with LLMs, we construct a class hierarchy. Withsuch a hierarchy, we can classify an image by descending from the top to the bottomof the hierarchy, comparing image and text embeddings at each level. Throughextensive experiments and analyses, we demonstrate that our proposed approach isintuitive, effective, and explainable. Code will be released upon publication.</details>

### [Oral] Cinematic Mindscapes: High-quality Video Reconstruction from Brain Activity

**Authors:** Zijiao Chen, Jiaxin Qing, Juan Helen Zhou

**Oral Presentation:** We, Dec 13, 08:15 -- Oral 3A

**<details><summary>Abstract**</summary>

Reconstructing human vision from brain activities has been an appealing task that helps to understand our cognitive process. Even though recent research has seen great success in reconstructing static images from non-invasive brain recordings, work on recovering continuous visual experiences in the form of videos is limited. In this work, we propose Mind-Video that learns spatiotemporal information from continuous fMRI data of the cerebral cortex progressively through masked brain modeling, multimodal contrastive learning with spatiotemporal attention, and co-training with an augmented Stable Diffusion model that incorporates network temporal inflation. We show that high-quality videos of arbitrary frame rates can be reconstructed with Mind-Video using adversarial guidance. The recovered videos were evaluated with various semantic and pixel-level metrics. We achieved an average accuracy of 85% in semantic classification tasks and 0.19 in structural similarity index (SSIM), outperforming the previous state-of-the-art by 45%. We also show that our model is biologically plausible and interpretable, reflecting established physiological processes.</details>

### Circuit as Set of Points

**Authors:** Jialv Zou, Xinggang Wang, Jiahao Guo, Wenyu Liu, Qian Zhang, Chang Huang

**<details><summary>Abstract**</summary>

As the size of circuit designs continues to grow rapidly, artificial intelligence technologies are being extensively used in Electronic Design Automation (EDA) to assist with circuit design.Placement and routing are the most time-consuming parts of the physical design process, and how to quickly evaluate the placement has become a hot research topic. Prior works either transformed circuit designs into images using hand-crafted methods and then used Convolutional Neural Networks (CNN) to extract features, which are limited by the quality of the hand-crafted methods and could not achieve end-to-end training, or treated the circuit design as a graph structure and used Graph Neural Networks (GNN) to extract features, which require time-consuming preprocessing.In our work, we propose a novel perspective for circuit design by treating circuit components as point clouds and using Transformer-based point cloud perception methods to extract features from the circuit. This approach enables direct feature extraction from raw data without any preprocessing, allows for end-to-end training, and results in high performance.Experimental results show that our method achieves state-of-the-art performance in congestion prediction tasks on both the CircuitNet and ISPD2015 datasets, as well as in design rule check (DRC) violation prediction tasks on the CircuitNet dataset.Our method establishes a bridge between the relatively mature point cloud perception methods and the fast-developing EDA algorithms, enabling us to leverage more collective intelligence to solve this task. To facilitate the research of open EDA design, source codes and pre-trained models are released at https://github.com/hustvl/circuitformer.</details>

### CoDA: Collaborative Novel Box Discovery and Cross-modal Alignment for Open-vocabulary 3D Object Detection

**Authors:** Yang Cao, Zeng Yihan, Hang Xu, Dan Xu

**<details><summary>Abstract**</summary>

Open-vocabulary 3D Object Detection (OV-3DDet) aims to detect objects from an arbitrary list of categories within a 3D scene, which remains seldom explored in the literature. There are primarily two fundamental problems in OV-3DDet, *i.e.*, localizing and classifying novel objects. This paper aims at addressing the two problems simultaneously via a unified framework, under the condition of limited base categories. To localize novel 3D objects, we propose an effective 3D Novel Object Discovery strategy, which utilizes both the 3D box geometry priors and 2D semantic open-vocabulary priors to generate pseudo box labels of the novel objects. To classify novel object boxes, we further develop a cross-modal alignment module based on discovered novel boxes, to align feature spaces between 3D pointcloud and image/text modalities. Specifically, the alignment process contains a class-agnostic and a class-discriminative alignment, incorporating not only the base objects with annotations but also the increasingly discovered novel objects, resulting in an iteratively enhanced alignment. The novel box discovery and crossmodal alignment are jointly learned to collaboratively benefit each other. Thenovel object discovery can directly impact the cross-modal alignment, while a better feature alignment can, in turn, boost the localization capability, leading to a unified OV-3DDet framework, named **CoDA**, for simultaneous novel object localization and classification. Extensive experiments on two challenging datasets (*i.e.*, SUN-RGBD and ScanNet) demonstrate the effectiveness of our method and also show a significant mAP improvement upon the best-performing alternative method by 80%. Codes and pre-trained models are released on [the project page](https://yangcaoai.github.io/publications/CoDA.html).</details>

### CoDrug: Conformal Drug Property Prediction with Density Estimation under Covariate Shift

**Authors:** Siddhartha Laghuvarapu, Zhen Lin, Jimeng Sun

**<details><summary>Abstract**</summary>

In drug discovery, it is vital to confirm the predictions of pharmaceutical properties from computational models using costly wet-lab experiments. Hence, obtaining reliable uncertainty estimates is crucial for prioritizing drug molecules for subsequent experimental validation. Conformal Prediction (CP) is a promising tool for creating such prediction sets for molecular properties with a coverage guarantee. However, the exchangeability assumption of CP is often challenged with covariate shift in drug discovery tasks: Most datasets contain limited labeled data, which may not be representative of the vast chemical space from which molecules are drawn. To address this limitation, we propose a method called CoDrug that employs an energy-based model leveraging both training data and unlabelled data, and  Kernel Density Estimation (KDE) to assess the densities of a molecule set. The estimated densities are then used to weigh the molecule samples while building prediction sets and rectifying for distribution shift. In extensive experiments involving realistic distribution drifts in various small-molecule drug discovery tasks,  we demonstrate the ability of CoDrug to provide valid prediction sets and its utility in addressing the distribution shift arising from de novo drug design models. On average, using CoDrug can reduce the coverage gap by over 35% when compared to conformal prediction sets not adjusted for covariate shift.</details>

### Cognitive Steering in Deep Neural Networks via Long-Range Modulatory Feedback Connections

**Authors:** Talia Konkle, George Alvarez

**<details><summary>Abstract**</summary>

Given the rich visual information available in each glance, humans can internally direct their visual attention to enhance goal-relevant information---a capacity often absent in standard vision models.  Here we introduce cognitively and biologically-inspired long-range modulatory pathways to enable `cognitive steering’ in vision models.  First, we show that models equipped with these feedback pathways naturally show improved image recognition, adversarial robustness, and increased brain alignment, relative to baseline models. Further,  these feedback projections from the final layer of the vision backbone provide a meaningful steering interface, where goals can be specified as vectors in the output space.  We show that there are effective ways to steer the model that dramatically improve recognition of categories in composite images of multiple categories, succeeding where baseline feed-forward models without flexible steering fail. And, our multiplicative modulatory motif prevents rampant hallucination of the top-down goal category, dissociating what the model is looking for, from what it is looking at. Thus, these long-range modulatory pathways enable new behavioral capacities for goal-directed visual encoding, offering a flexible communication interface between cognitive and visual systems.</details>

### Combining Behaviors with the Successor Features Keyboard

**Authors:** Wilka Carvalho Carvalho, Andre Saraiva, Angelos Filos, Andrew Lampinen, Loic Matthey, Richard L Lewis, Honglak Lee, Satinder Singh, Danilo Jimenez Rezende, Daniel Zoran

**<details><summary>Abstract**</summary>

The Option Keyboard (OK) was recently proposed as a method for transferring behavioral knowledge across tasks. OK transfers knowledge by adaptively combining subsets of known behaviors using Successor Features (SFs) and Generalized Policy Improvement (GPI).However, it relies on hand-designed state-features and task encodings which are cumbersome to design for every new environment.In this work, we propose the "Successor Features Keyboard" (SFK), which enables transfer with discovered state-features and task encodings.To enable discovery, we propose the "Categorical Successor Feature Approximator" (CSFA), a novel learning algorithm for estimating SFs while jointly discovering state-features and task encodings.With SFK and CSFA, we achieve the first demonstration of transfer with SFs in a challenging 3D environment where all the necessary representations are discovered.We first compare CSFA against other methods for approximating SFs and show that only CSFA discovers representations compatible with SF&GPI at this scale.We then compare SFK against transfer learning baselines and show that it transfers most quickly to long-horizon tasks.</details>

### Compositional Generalization from First Principles

**Authors:** Thaddäus Wiedemer, Prasanna Mayilvahanan, Matthias Bethge, Wieland Brendel

**<details><summary>Abstract**</summary>

Leveraging the compositional nature of our world to expedite learning and facilitate generalization is a hallmark of human perception. In machine learning, on the other hand, achieving compositional generalization has proven to be an elusive goal, even for models with explicit compositional priors. To get a better handle on compositional generalization, we here approach it from the bottom up: Inspired by identifiable representation learning, we investigate compositionality as a property of the data-generating process rather than the data itself. This reformulation enables us to derive mild conditions on only the support of the training distribution and the model architecture, which are sufficient for compositional generalization. We further demonstrate how our theoretical framework applies to real-world scenarios and validate our findings empirically. Our results set the stage for a principled theoretical study of compositional generalization.</details>

### Conditional Adapters: Parameter-efficient Transfer Learning with Fast Inference

**Authors:** Tao Lei, Junwen Bai, Siddhartha Brahma, Joshua Ainslie, Kenton Lee, Yanqi Zhou, Nan Du, Vincent Zhao, Yuexin Wu, Bo Li, Yu Zhang, Ming-Wei Chang

**<details><summary>Abstract**</summary>

We propose Conditional Adapter (CoDA), a parameter-efficient transfer learning method that also improves inference efficiency. CoDA generalizes beyond standard adapter approaches to enable a new way of balancing speed and accuracy using conditional computation.Starting with an existing dense pretrained model, CoDA adds sparse activation together with a small number of new parameters and a light-weight training phase.Our experiments demonstrate that the CoDA approach provides an unexpectedly efficient way to transfer knowledge.Across a variety of language, vision, and speech tasks, CoDA achieves a 2x to 8x inference speed-up compared to the state-of-the-art Adapter approaches with moderate to no accuracy loss and the same parameter efficiency.</details>

### Conditional Matrix Flows for Gaussian Graphical Models

**Authors:** Marcello Massimo Negri, Fabricio Arend Torres, Volker Roth

**<details><summary>Abstract**</summary>

Studying conditional independence among many variables with few observations is a challenging task.Gaussian Graphical Models (GGMs) tackle this problem by encouraging sparsity in the precision matrix through $l_q$ regularization with $q\leq1$.However, most GMMs rely on the $l_1$ norm because the objective is highly non-convex for sub-$l_1$ pseudo-norms.In the frequentist formulation, the $l_1$ norm relaxation provides the solution path as a function of the shrinkage parameter $\lambda$.In the Bayesian formulation, sparsity is instead encouraged through a Laplace prior, but posterior inference for different $\lambda$ requires repeated runs of expensive Gibbs samplers.Here we propose a general framework for variational inference with matrix-variate Normalizing Flow in GGMs, which unifies the benefits of frequentist and Bayesian frameworks.As a key improvement on previous work, we train with one flow a continuum of sparse regression models jointly for all regularization parameters $\lambda$ and all $l_q$ norms, including non-convex sub-$l_1$ pseudo-norms.Within one model we thus have access to (i) the evolution of the posterior for any $\lambda$ and any $l_q$ (pseudo-) norm, (ii) the marginal log-likelihood for model selection, and (iii) the frequentist solution paths through simulated annealing in the MAP limit.</details>

### [Spotlight] Conditional Mutual Information for Disentangled Representations in Reinforcement Learning

**Authors:** Mhairi Dunion, Trevor McInroe, Kevin Sebastian Luck, Josiah Hanna, Stefano Albrecht

**<details><summary>Abstract**</summary>

Reinforcement Learning (RL) environments can produce training data with spurious correlations between features due to the amount of training data or its limited feature coverage. This can lead to RL agents encoding these misleading correlations in their latent representation, preventing the agent from generalising if the correlation changes within the environment or when deployed in the real world. Disentangled representations can improve robustness, but existing disentanglement techniques that minimise mutual information between features require independent features, thus they cannot disentangle correlated features. We propose an auxiliary task for RL algorithms that learns a disentangled representation of high-dimensional observations with correlated features by minimising the conditional mutual information between features in the representation. We demonstrate experimentally, using continuous control tasks, that our approach improves generalisation under correlation shifts, as well as improving the training performance of RL algorithms in the presence of correlated features.</details>

### Conformal Prediction for Time Series with Modern Hopfield Networks

**Authors:** Andreas Auer, Martin Gauch, Daniel Klotz, Sepp Hochreiter

**<details><summary>Abstract**</summary>

To quantify uncertainty, conformal prediction methods are gaining continuously more interest and have already been successfully applied to various domains. However, they are difficult to apply to time series as the autocorrelative structure of time series violates basic assumptions required by conformal prediction. We propose HopCPT, a novel conformal prediction approach for time series that not only copes with temporal structures but leverages them. We show that our approach is theoretically well justified for time series where temporal dependencies are present. In experiments, we demonstrate that our new approach outperforms state-of-the-art conformal prediction methods on multiple real-world time series datasets from four different domains.</details>

### Connecting Multi-modal Contrastive Representations

**Authors:** Zehan Wang, Yang Zhao, Xize 成, Haifeng Huang, Jiageng Liu, Aoxiong Yin, Li Tang, Linjun Li, Yongqi Wang, Ziang Zhang, Zhou Zhao

**<details><summary>Abstract**</summary>

Multi-modal Contrastive Representation (MCR) learning aims to encode different modalities into a semantically aligned shared space. This paradigm shows remarkable generalization ability on numerous downstream tasks across various modalities. However, the reliance on massive high-quality data pairs limits its further development on more modalities. This paper proposes a novel training-efficient method for learning MCR without paired data called Connecting Multi-modal Contrastive Representations (C-MCR). Specifically, given two existing MCRs pre-trained on $(\mathcal{A}$, $\mathcal{B})$ and $(\mathcal{B}$, $\mathcal{C})$ modality pairs, we project them to a new space and use the data from the overlapping modality $\mathcal{B}$ to aligning the two MCRs in the new space. Meanwhile, since the modality pairs $(\mathcal{A}$, $\mathcal{B})$ and $(\mathcal{B}$, $\mathcal{C})$ are already aligned within each MCR, the connection learned by overlapping modality can also be transferred to non-overlapping modality pair $(\mathcal{A}$, $\mathcal{C})$. To unleash the potential of C-MCR, we further introduce a semantic-enhanced inter- and intra-MCR connection method. We first enhance the semantic consistency and completion of embeddings across different modalities for more robust alignment. Then we utilize the inter-MCR alignment to establish the connection, and employ the intra-MCR alignment to better maintain the connection for inputs from non-overlapping modalities. To demonstrate the effectiveness of C-MCR, we take the field of audio-visual and 3D-language learning as examples. Specifically, we connect CLIP and CLAP via texts to derive audio-visual representations, and integrate CLIP and ULIP via images for 3D-language representations. Remarkably, without using any paired data, C-MCR for audio-visual achieves state-of-the-art performance on audio-image retrieval, audio-visual source localization, and counterfactual audio-image recognition tasks. Furthermore, C-MCR for 3D-language also attains advanced zero-shot 3D point cloud classification accuracy on ModelNet40. Our project page is available at \url{https://c-mcr.github.io/C-MCR/}</details>

### Consistent Diffusion Models: Mitigating Sampling Drift by Learning to be Consistent

**Authors:** Giannis Daras, Yuval Dagan, Alex Dimakis, Constantinos Daskalakis

**<details><summary>Abstract**</summary>

Imperfect score-matching leads to a shift between the training and the sampling distribution of diffusion models. Due to the recursive nature of the generation process, errors in previous steps yield sampling iterates that drift away from the training distribution. However, the standard training objective via Denoising Score Matching (DSM) is only designed to optimize over non-drifted data. To train on drifted data, we propose to enforce a \emph{Consistency} property (CP) which states that predictions of the model on its owngenerated data are consistent across time. Theoretically, we show that the differential equation that describes CP together with the one that describes a conservative vector field, have a unique solution given some initial condition. Consequently, if the score is learned well on non-drifted points via DSM (enforcing the true initial condition) then enforcing CP on drifted points propagates true score values. Empirically, we show that enforcing CP improves the generation quality for conditional and unconditional generation on CIFAR-10, and in AFHQ and FFHQ. We open-source our code and models: https://github.com/giannisdaras/cdm.</details>

### Constraint-Conditioned Policy Optimization for Versatile Safe Reinforcement Learning

**Authors:** Yihang Yao, ZUXIN LIU, Zhepeng Cen, Jiacheng Zhu, Wenhao Yu, Tingnan Zhang, DING ZHAO

**<details><summary>Abstract**</summary>

Safe reinforcement learning (RL) focuses on training reward-maximizing agents subject to pre-defined safety constraints. Yet, learning versatile safe policies that can adapt to varying safety constraint requirements during deployment without retraining remains a largely unexplored and challenging area. In this work, we formulate the versatile safe RL problem and consider two primary requirements: training efficiency and zero-shot adaptation capability. To address them, we introduce the Conditioned Constrained Policy Optimization (CCPO) framework, consisting of two key modules: (1) Versatile Value Estimation (VVE) for approximating value functions under unseen threshold conditions, and (2) Conditioned Variational Inference (CVI) for encoding arbitrary constraint thresholds during policy optimization. Our extensive experiments demonstrate that CCPO outperforms the baselines in terms of safety and task performance while preserving zero-shot adaptation capabilities to different constraint thresholds data-efficiently. This makes our approach suitable for real-world dynamic applications.</details>

### Construction of Hierarchical Neural Architecture Search Spaces based on Context-free Grammars

**Authors:** Simon Schrodi, Danny Stoll, Binxin Ru, Rhea Sukthanker, Thomas Brox, Frank Hutter

**<details><summary>Abstract**</summary>

The discovery of neural architectures from simple building blocks is a long-standing goal of Neural Architecture Search (NAS). Hierarchical search spaces are a promising step towards this goal but lack a unifying search space design framework and typically only search over some limited aspect of architectures. In this work, we introduce a unifying search space design framework based on context-free grammars that can naturally and compactly generate expressive hierarchical search spaces that are 100s of orders of magnitude larger than common spaces from the literature. By enhancing and using their properties, we effectively enable search over the complete architecture and can foster regularity. Further, we propose an efficient hierarchical kernel design for a Bayesian Optimization search strategy to efficiently search over such huge spaces. We demonstrate the versatility of our search space design framework and show that our search strategy can be superior to existing NAS approaches. Code is available at https://github.com/automl/hierarchical_nas_construction.</details>

### Context-lumpable stochastic bandits

**Authors:** Chung-Wei Lee, Qinghua Liu, Yasin Abbasi Yadkori, Chi Jin, Tor Lattimore, Csaba Szepesvari

**<details><summary>Abstract**</summary>

We consider a contextual bandit problem with $S $ contexts and $K $ actions. In each round $t=1,2,\dots$ the learnerobserves a random context and chooses an action based on its past experience. The learner then observes a random reward whose mean is a function of the context and the action for the round. Under the assumption that the contexts can be lumped into $r\le \min(S ,K)$ groups such that the mean reward for the various actions is the same for any two contexts that are in the same group, we give an algorithm that outputs an $\epsilon$-optimal policy after using at most $\widetilde O(r (S +K )/\epsilon^2)$ samples with high probability and provide a matching $\widetilde\Omega(r (S +K )/\epsilon^2)$ lower bound. In the regret minimization setting, we give an algorithm whose cumulative regret up to time $T$ is bounded by $\widetilde O(\sqrt{r ^3(S +K )T})$. To the best of our knowledge, we are the first to show the near-optimal sample complexity in the PAC setting and $\widetilde O{\sqrt{\text{poly}(r)(S+K)T}}$ minimax regret in the online setting for this problem.  We also show our algorithms can be applied to more general low-rank bandits and get improved regret bounds in some scenarios.</details>

### Contextually Affinitive Neighborhood Refinery for Deep Clustering

**Authors:** Chunlin Yu, Ye Shi, Jingya Wang

**<details><summary>Abstract**</summary>

Previous endeavors in self-supervised learning have enlightened the research of deep clustering from an instance discrimination perspective. Built upon this foundation, recent studies further highlight the importance of grouping semantically similar instances. One effective method to achieve this is by promoting the semantic structure preserved by neighborhood consistency. However, the samples in the local neighborhood may be limited due to their close proximity to each other,  which may not provide substantial and diverse supervision signals. Inspired by the versatile re-ranking methods in the context of image retrieval, we propose to employ an efficient online re-ranking process to mine more informative neighbors in a Contextually Affinitive (ConAff) Neighborhood, and then encourage the cross-view neighborhood consistency. To further mitigate the intrinsic neighborhood noises near cluster boundaries, we propose a progressively relaxed boundary filtering strategy to circumvent the issues brought by noisy neighbors. Our method can be easily integrated into the generic self-supervised frameworks and outperforms the state-of-the-art methods on several popular benchmarks.</details>

### Contrastive Moments: Unsupervised Halfspace Learning in Polynomial Time

**Authors:** Xinyuan Cao, Santosh Vempala

**<details><summary>Abstract**</summary>

We give a polynomial-time algorithm for learning high-dimensional halfspaces with margins in $d$-dimensional space to within desired TV distance when the ambient distribution is an unknown affine transformation of the $d$-fold product of an (unknown) symmetric one-dimensional logconcave distribution, and the halfspace is introduced by deleting at least an $\epsilon$ fraction of the data in one of the component distributions. Notably, our algorithm does not need labels and establishes the unique (and efficient) identifiability of the hidden halfspace under this distributional assumption.  The sample and time complexity of the algorithm are polynomial in the dimension and $1/\epsilon$. The algorithm uses only the first two moments of *suitable re-weightings* of the empirical distribution, which we call *contrastive moments*; its analysis uses classical facts about generalized Dirichlet polynomials and relies crucially on a new monotonicity property of the moment ratio of truncations of logconcave distributions. Such algorithms, based only on first and second moments were suggested in earlier work, but hitherto eluded rigorous guarantees.Prior work addressed the special case when the underlying distribution is Gaussian via Non-Gaussian Component Analysis. We improve on this by providing polytime guarantees based on Total Variation (TV) distance, in place of existing moment-bound guarantees that can be super-polynomial. Our work is also the first to go beyond Gaussians in this setting.</details>

### Controlling Text-to-Image Diffusion by Orthogonal Finetuning

**Authors:** Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, Bernhard Schölkopf

**<details><summary>Abstract**</summary>

Large text-to-image diffusion models have impressive capabilities in generating photorealistic images from text prompts. How to effectively guide or control these powerful models to perform different downstream tasks becomes an important open problem. To tackle this challenge, we introduce a principled finetuning method -- Orthogonal Finetuning (OFT), for adapting text-to-image diffusion models to downstream tasks. Unlike existing methods, OFT can provably preserve hyperspherical energy which characterizes the pairwise neuron relationship on the unit hypersphere. We find that this property is crucial for preserving the semantic generation ability of text-to-image diffusion models. To improve finetuning stability, we further propose Constrained Orthogonal Finetuning (COFT) which imposes an additional radius constraint to the hypersphere. Specifically, we consider two important finetuning text-to-image tasks: subject-driven generation where the goal is to generate subject-specific images given a few images of a subject and a text prompt, and controllable generation where the goal is to enable the model to take in additional control signals. We empirically show that our OFT framework outperforms existing methods in generation quality and convergence speed.</details>

### Convergence Analysis of Sequential Federated Learning on Heterogeneous Data

**Authors:** Yipeng Li, Xinchen Lyu

**<details><summary>Abstract**</summary>

There are two categories of methods in Federated Learning (FL) for joint training across multiple clients: i) parallel FL (PFL), where clients train models in a parallel manner; and ii) sequential FL (SFL), where clients train models in a sequential manner. In contrast to that of PFL, the convergence theory of SFL on heterogeneous data is still lacking. In this paper, we establish the convergence guarantees of SFL for strongly/general/non-convex objectives on heterogeneous data. The convergence guarantees of SFL are better than that of PFL on heterogeneous data with both full and partial client participation. Experimental results validate the counterintuitive analysis result that SFL outperforms PFL on extremely heterogeneous data in cross-device settings.</details>

### Convergence of Actor-Critic with Multi-Layer Neural Networks

**Authors:** Haoxing Tian, Alex Olshevsky, Yannis Paschalidis

**<details><summary>Abstract**</summary>

The early theory of actor-critic methods considered convergence using linear function approximators for the policy and value functions. Recent work has established convergence using neural network approximators with a single hidden layer. In this work we are taking the natural next step and establish convergence using deep neural networks with an arbitrary number of hidden layers, thus closing a gap between theory and practice. We show that actor-critic updates projected on a ball around the initial condition will converge to a neighborhood where the average of the squared gradients is $\tilde{O} \left( 1/\sqrt{m} \right) + O \left( \epsilon \right)$, with $m$ being the width of the neural network and $\epsilon$ the approximation quality of the best critic neural network over the projected set.</details>

### Convolutional Neural Operators for robust and accurate learning of PDEs

**Authors:** Bogdan Raonic, Roberto Molinaro, Tim De Ryck, Tobias Rohner, Francesca Bartolucci, Rima Alaifari, Siddhartha Mishra, Emmanuel de Bézenac

**<details><summary>Abstract**</summary>

Although very successfully used in conventional machine learning, convolution based neural network architectures -- believed to be inconsistent in function space -- have been largely ignored in the context of learning solution operators of PDEs. Here, we present novel adaptations for convolutional neural networks to demonstrate that they are indeed able to process functions as inputs and outputs. The resulting architecture, termed as convolutional neural operators (CNOs), is designed specifically to preserve its underlying continuous nature, even when implemented in a discretized form on a computer. We prove a universality theorem to show that CNOs can approximate operators arising in PDEs to desired accuracy. CNOs are tested on a novel suite of benchmarks, encompassing a diverse set of PDEs with multi-scale solutions and are observed to significantly outperform baselines, paving the way for an alternative framework for robust and accurate operator learning.</details>

### Core-sets for Fair and Diverse Data Summarization

**Authors:** Sepideh Mahabadi, Stojan Trajanovski

**<details><summary>Abstract**</summary>

We study core-set construction algorithms for the task of Diversity Maximization under fairness/partition constraint. Given a set of points $P$ in a metric space partitioned into $m$ groups, and given $k_1,\ldots,k_m$, the goal of this problem is to pick $k_i$ points from each group $i$ such that the overall diversity of the $k=\sum_i k_i$ picked points is maximized. We consider two natural diversity measures: sum-of-pairwise distances and sum-of-nearest-neighbor distances, and show improved core-set construction algorithms with respect to these measures. More precisely, we show the first constant factor core-set w.r.t. sum-of-pairwise distances whose size is independent of the size of the dataset and the aspect ratio. Second, we show the first core-set w.r.t. the sum-of-nearest-neighbor distances. Finally, we run several experiments showing the effectiveness of our core-set approach. In particular, we apply constrained diversity maximization to summarize a set of timed messages that takes into account the messages' recency. Specifically, the summary should include more recent messages compared to older ones. This is a real task in one of the largest communication platforms, affecting the experience of hundreds of millions daily active users. By utilizing our core-set method for this task, we achieve a 100x speed-up while losing the diversity by only a few percent. Moreover, our approach allows us to improve the space usage of the algorithm in the streaming setting.</details>

### Corruption-Robust Offline Reinforcement Learning with General Function Approximation

**Authors:** Chenlu Ye, Rui Yang, Quanquan Gu, Tong Zhang

**<details><summary>Abstract**</summary>

We investigate the problem of corruption robustness in offline reinforcement learning (RL) with general function approximation, where an adversary can corrupt each sample in the offline dataset, and the corruption level $\zeta\geq0$ quantifies the cumulative corruption amount over $n$ episodes and $H$ steps. Our goal is to find a policy that is robust to such corruption and minimizes the suboptimality gap with respect to the optimal policy for the uncorrupted Markov decision processes (MDPs). Drawing inspiration from the uncertainty-weighting technique from the robust online RL setting \citep{he2022nearly,ye2022corruptionrobust}, we design a new uncertainty weight iteration procedure to efficiently compute on batched samples and propose a corruption-robust algorithm for offline RL. Notably, under the assumption of single policy coverage and the knowledge of $\zeta$, our proposed algorithm achieves a suboptimality bound that is worsened by an additive factor of $\mathcal O(\zeta \cdot (\text CC(\lambda,\hat{\mathcal F},\mathcal Z_n^H))^{1/2} (C(\hat{\mathcal F},\mu))^{-1/2} n^{-1})$ due to the corruption. Here $\text CC(\lambda,\hat{\mathcal F},\mathcal Z_n^H)$ is the coverage coefficient that depends on the regularization parameter $\lambda$, the confidence set $\hat{\mathcal F}$, and the dataset $\mathcal Z_n^H$, and $C(\hat{\mathcal F},\mu)$ is a coefficient that depends on $\hat{\mathcal F}$ and the underlying data distribution $\mu$. When specialized to linear MDPs, the corruption-dependent error term reduces to $\mathcal O(\zeta d n^{-1})$ with $d$ being the dimension of the feature map, which matches the existing lower bound for corrupted linear MDPs. This suggests that our analysis is tight in terms of the corruption-dependent term.</details>

### Counterfactually Fair Representation

**Authors:** Zhiqun Zuo, Mahdi Khalili, Xueru Zhang

**<details><summary>Abstract**</summary>

The use of machine learning models in high-stake applications (e.g., healthcare, lending, college admission) has raised growing concerns due to potential biases against protected social groups. Various fairness notions and methods have been proposed to mitigate such biases. In this work, we focus on Counterfactual Fairness (CF), a fairness notion that is dependent on an underlying causal graph and first proposed by Kusner $\textit{et al.}$; it requires that the outcome an individual perceives is the same in the real world as it would be in a "counterfactual" world, in which the individual belongs to another social group. Learning fair models satisfying CF can be challenging. It was shown in (Kusner $\textit{et al.}$) that a sufficient condition for satisfying CF is to $\textbf{not}$ use features that are descendants of sensitive attributes in the causal graph. This implies a simple method that learns CF models only using non-descendants of sensitive attributes while eliminating all descendants. Although several subsequent works proposed methods that use all features for training CF models, there is no theoretical guarantee that they can satisfy CF. In contrast, this work proposes a new algorithm that trains models using all the available features. We theoretically and empirically show that models trained with this method can satisfy CF.</details>

### [Spotlight] Critical Initialization of Wide and Deep Neural Networks using Partial Jacobians: General Theory and Applications

**Authors:** Darshil Doshi, Tianyu He, Andrey Gromov

**<details><summary>Abstract**</summary>

Deep neural networks are notorious for defying theoretical treatment. However, when the number of parameters in each layer tends to infinity, the network function is a Gaussian process (GP) and quantitatively predictive description is possible. Gaussian approximation allows one to formulate criteria for selecting hyperparameters, such as variances of weights and biases, as well as the learning rate. These criteria rely on the notion of criticality defined for deep neural networks. In this work we describe a new practical way to diagnose criticality. We introduce *partial Jacobians* of a network, defined as derivatives of preactivations in layer $l$ with respect to preactivations in layer $l_0\leq l$. We derive recurrence relations for the norms of partial Jacobians and utilize these relations to analyze criticality of deep fully connected neural networks with LayerNorm and/or residual connections. We derive and implement a simple and cheap numerical test that allows one to select optimal initialization for a broad class of deep neural networks; containing fully connected, convolutional and normalization layers. Using these tools we show quantitatively that proper stacking of the LayerNorm (applied to preactivations) and residual connections leads to an architecture that is critical for any initialization. Finally, we apply our methods to analyze ResNet and MLP-Mixer architectures; demonstrating the everywhere-critical regime.</details>

### Crystal Structure Prediction by Joint Equivariant Diffusion

**Authors:** Rui Jiao, Wenbing Huang, Peijia Lin, Jiaqi Han, Pin Chen, Yutong Lu, Yang Liu

**<details><summary>Abstract**</summary>

Crystal Structure Prediction (CSP) is crucial in various scientific disciplines. While CSP can be addressed by employing currently-prevailing generative models (**e.g.** diffusion models), this task encounters unique challenges owing to the symmetric geometry of crystal structures---the invariance of translation, rotation, and periodicity. To incorporate the above symmetries, this paper proposes DiffCSP, a novel diffusion model to learn the structure distribution from stable crystals. To be specific, DiffCSP jointly generates the lattice and atom coordinates for each crystal by employing a periodic-E(3)-equivariant denoising model, to better model the crystal geometry. Notably, different from related equivariant generative approaches, DiffCSP leverages fractional coordinates other than Cartesian coordinates to represent crystals, remarkably promoting the diffusion and the generation process of atom positions. Extensive experiments verify that our DiffCSP remarkably outperforms existing CSP methods, with a much lower computation cost in contrast to DFT-based methods. Moreover, the superiority of DiffCSP is still observed when it is extended for ab initio crystal generation.</details>

### Curvature Filtrations for Graph Generative Model Evaluation

**Authors:** Joshua Southern, Jeremy Wayland, Michael Bronstein, Bastian Rieck

**<details><summary>Abstract**</summary>

Graph generative model evaluation necessitates understanding differences between graphs on the distributional level. This entails being able to harness salient attributes of graphs in an efficient manner. Curvature constitutes one such property of graphs, and has recently started to prove useful in characterising graphs. Its expressive properties, stability, and practical utility in model evaluation remain largely unexplored, however. We combine graph curvature descriptors with emerging methods from topological data analysis to obtain robust, expressive descriptors for evaluating graph generative models.</details>

### D4Explainer: In-distribution Explanations of Graph Neural Network via Discrete Denoising Diffusion

**Authors:** Jialin Chen, Shirley Wu, Abhijit Gupta, Rex Ying

**<details><summary>Abstract**</summary>

The widespread deployment of Graph Neural Networks (GNNs) sparks significant interest in their explainability, which plays a vital role in model auditing and ensuring trustworthy graph learning. The objective of GNN explainability is to discern the underlying graph structures that have the most significant impact on model predictions. Ensuring that explanations generated are reliable necessitates consideration of the in-distribution property, particularly due to the vulnerability of GNNs to out-of-distribution data. Unfortunately, prevailing explainability methods tend to constrain the generated explanations to the structure of the original graph, thereby downplaying the significance of the in-distribution property and resulting in explanations that lack reliability.To address these challenges, we propose D4Explainer, a novel approach that provides in-distribution GNN explanations for both counterfactual and model-level explanation scenarios. The proposed D4Explainer incorporates generative graph distribution learning into the optimization objective, which accomplishes two goals: 1) generate a collection of diverse counterfactual graphs that conform to the in-distribution property for a given instance, and 2) identify the most discriminative graph patterns that contribute to a specific class prediction, thus serving as model-level explanations. It is worth mentioning that D4Explainer is the first unified framework that combines both counterfactual and model-level explanations.Empirical evaluations conducted on synthetic and real-world datasets provide compelling evidence of the state-of-the-art performance achieved by D4Explainer in terms of explanation accuracy, faithfulness, diversity, and robustness.</details>

### DASpeech: Directed Acyclic Transformer for Fast and High-quality Speech-to-Speech Translation

**Authors:** Qingkai Fang, Yan Zhou, Yang Feng

**<details><summary>Abstract**</summary>

Direct speech-to-speech translation (S2ST) translates speech from one language into another using a single model. However, due to the presence of linguistic and acoustic diversity, the target speech follows a complex multimodal distribution, posing challenges to achieving both high-quality translations and fast decoding speeds for S2ST models. In this paper, we propose DASpeech, a non-autoregressive direct S2ST model which realizes both fast and high-quality S2ST. To better capture the complex distribution of the target speech, DASpeech adopts the two-pass architecture to decompose the generation process into two steps, where a linguistic decoder first generates the target text, and an acoustic decoder then generates the target speech based on the hidden states of the linguistic decoder. Specifically, we use the decoder of DA-Transformer as the linguistic decoder, and use FastSpeech 2 as the acoustic decoder. DA-Transformer models translations with a directed acyclic graph (DAG). To consider all potential paths in the DAG during training, we calculate the expected hidden states for each target token via dynamic programming, and feed them into the acoustic decoder to predict the target mel-spectrogram. During inference, we select the most probable path and take hidden states on that path as input to the acoustic decoder. Experiments on the CVSS Fr$\rightarrow$En benchmark demonstrate that DASpeech can achieve comparable or even better performance than the state-of-the-art S2ST model Translatotron 2, while preserving up to 18.53$\times$ speedup compared to the autoregressive baseline. Compared with the previous non-autoregressive S2ST model, DASpeech does not rely on knowledge distillation and iterative decoding, achieving significant improvements in both translation quality and decoding speed. Furthermore, DASpeech shows the ability to preserve the speaker's voice of the source speech during translation.</details>

### DAW: Exploring the Better Weighting Function for Semi-supervised Semantic Segmentation

**Authors:** Rui Sun, Huayu Mai, Tianzhu Zhang, Feng Wu

**<details><summary>Abstract**</summary>

The critical challenge of semi-supervised semantic segmentation lies in how to fully exploit a large volume of unlabeled data to improve the model’s generalization performance for robust segmentation.  Existing methods tend to employ certain criteria (weighting function) to select pixel-level pseudo labels. However, the trade-off exists between inaccurate yet utilized pseudo-labels, and correct yet discarded pseudo-labels in these methods when handling pseudo-labels without thoughtful consideration of the weighting function, hindering the generalization ability of the model. In this paper, we systematically analyze the trade-off in previous methods when dealing with pseudo-labels. We formally define the trade-off between inaccurate yet utilized pseudo-labels, and correct yet discarded pseudo-labels by explicitly modeling the confidence distribution of correct and inaccurate pseudo-labels, equipped with a unified weighting function. To this end, we propose Distribution-Aware Weighting (DAW) to strive to minimize the negative equivalence impact raised by the trade-off. We find an interesting fact that the optimal solution for the weighting function is a hard step function, with the jump point located at the intersection of the two confidence distributions. Besides, we devise distribution alignment to mitigate the issue of the discrepancy between the prediction distributions of labeled and unlabeled data. Extensive experimental results on multiple benchmarks including mitochondria segmentation demonstrate that DAW performs favorably against state-of-the-art methods.</details>

### [Spotlight] DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization

**Authors:** Zhiqing Sun, Yiming Yang

**<details><summary>Abstract**</summary>

Neural network-based Combinatorial Optimization (CO) methods have shown promising results in solving various NP-complete (NPC) problems without relying on hand-crafted domain knowledge. This paper broadens the current scope of neural solvers for NPC problems by introducing a new graph-based diffusion framework, namely DIFUSCO. It formulates NPC problems into a discrete {0, 1}-vector space and uses graph-based denoising diffusion models to generate high-quality solutions. Specifically, we explore diffusion models with Gaussian and Bernoulli noise, respectively, and also introduce an effective inference schedule to improve the generation quality. We evaluate our methods on two well-studied combinatorial optimization problems: Traveling Salesman Problem (TSP) and Maximal Independent Set (MIS). Experimental results show that DIFUSCO strongly outperforms the previous state-of-the-art neural solvers, improving the performance gap between ground-truth and neural solvers from 1.76% to 0.46% on TSP-500, from 2.46% to 1.17% on TSP-1000, and from 3.19% to 2.58% on TSP-10000. For the MIS problem, DIFUSCO outperforms the previous state-of-the-art neural solver on the challenging SATLIB benchmark. Our code is available at [this url](https://github.com/Edward-Sun/DIFUSCO).</details>

### DSR: Dynamical Surface Representation as Implicit Neural Networks for Protein

**Authors:** Daiwen Sun, He Huang, Yao Li, Xinqi Gong, Qiwei Ye

**<details><summary>Abstract**</summary>

We propose a novel neural network-based approach to modeling protein dynamics using an implicit representation of a protein’s surface in 3D and time. Our method utilizes the zero-level set of signed distance functions (SDFs) to represent protein surfaces, enabling temporally and spatially continuous representations of protein dynamics. Our experimental results demonstrate that our model accurately captures protein dynamic trajectories and can interpolate and extrapolate in 3D and time. Importantly, this is the first study to introduce this method and successfully model large-scale protein dynamics. This approach offers a promising alternative to current methods, overcoming the limitations of first-principles-based and deep learning methods, and provides a more scalable and efficient approach to modeling protein dynamics. Additionally, our surface representation approach simplifies calculations and allows identifying movement trends and amplitudes of protein domains, making it a useful tool for protein dynamics research. Codes are available at https://github.com/Sundw-818/DSR, and we have a project webpage that shows some video results, https://sundw-818.github.io/DSR/.</details>

### Data Quality in Imitation Learning

**Authors:** Suneel Belkhale, Yuchen Cui, Dorsa Sadigh

**<details><summary>Abstract**</summary>

In supervised learning, the question of data quality and curation has been sidelined in recent years in favor of increasingly more powerful and expressive models that can ingest internet-scale data. However, in offline learning for robotics, we simply lack internet scale data, and so high quality datasets are a necessity. This is especially true in imitation learning (IL), a sample efficient paradigm for robot learning using expert demonstrations. Policies learned through IL suffer from state distribution shift at test time due to compounding errors in action prediction, which leads to unseen states that the policy cannot recover from.Instead of designing new algorithms to address distribution shift, an alternative perspective is to develop new ways of assessing and curating datasets. There is growing evidence that the same IL algorithms can have substantially different performance across different datasets. This calls for a formalism for defining metrics of "data quality" that can further be leveraged for data curation.In this work, we take the first step toward formalizing data quality for imitation learning through the lens of distribution shift: a high quality dataset encourages the policy to stay in distribution at test time. We propose two fundamental properties that are necessary for a high quality datasets: i) action divergence: the mismatch between the expert and learned policy at certain states; and ii) transition diversity: the noise present in the system for a given state and action. We investigate the combined effect of these two key properties in imitation learning theoretically, and we empirically analyze models trained on a variety of different data sources. We show that state diversity is not always beneficial, and we demonstrate how action divergence and transition diversity interact in practice.</details>

### Data-Centric Learning from Unlabeled Graphs with Diffusion Model

**Authors:** Gang Liu, Eric Inae, Tong Zhao, Jiaxin Xu, Tengfei Luo, Meng Jiang

**<details><summary>Abstract**</summary>

Graph property prediction tasks are important and numerous. While each task offers a small size of labeled examples, unlabeled graphs have been collected from various sources and at a large scale. A conventional approach is training a model with the unlabeled graphs on self-supervised tasks and then fine-tuning the model on the prediction tasks. However, the self-supervised task knowledge could not be aligned or sometimes conflicted with what the predictions needed. In this paper, we propose to extract the knowledge underlying the large set of unlabeled graphs as a specific set of useful data points to augment each property prediction model. We use a diffusion model to fully utilize the unlabeled graphs and design two new objectives to guide the model's denoising process with each task's labeled data to generate task-specific graph examples and their labels. Experiments demonstrate that our data-centric approach performs significantly better than fifteen existing various methods on fifteen tasks. The performance improvement brought by unlabeled data is visible as the generated labeled examples unlike the self-supervised learning.</details>

### DatasetDM: Synthesizing Data with Perception Annotations Using Diffusion Models

**Authors:** Weijia Wu, Yuzhong Zhao, Hao Chen, Yuchao Gu, Rui Zhao, Yefei He, Hong Zhou, Mike Zheng Shou, Chunhua Shen

**<details><summary>Abstract**</summary>

Current deep networks are very data-hungry and benefit from training on large-scale datasets, which are often time-consuming to collect and annotate. By contrast, synthetic data can be generated infinitely using generative models such as DALL-E and diffusion models, with minimal effort and cost. In this paper, we present DatasetDM, a generic dataset generation model that can produce diverse syntheticimages and the corresponding high-quality perception annotations (e.g., segmentation masks, and depth). Our method builds upon the pre-trained diffusion model and extends text-guided image synthesis to perception data generation. We show that the rich latent code of the diffusion model can be effectively decoded as accurate perception annotations using a decoder module. Training the decoder only needs less than 1% (around 100 images) of manually labeled images, enabling the generation of an infinitely large annotated dataset. Then these synthetic data can be used for training various perception models on downstream tasks. To showcase the power of the proposed approach, we generate datasets with rich dense pixel-wise labels for a wide range of downstream tasks, including semantic15segmentation, instance segmentation, and depth estimation. Notably, it achieves 1) state-of-the-art results on semantic segmentation and instance segmentation; 2) significantly more efficient and robust in domain generalization than the real data; 3) state-of-the-art results in zero-shot segmentation setting; and 4) flexibility for efficient application and novel task composition (e.g., image editing)</details>

### Debiased and Denoised Entity Recognition from Distant Supervision

**Authors:** Haobo Wang, Yiwen Dong, Ruixuan Xiao, Fei Huang, Gang Chen, Junbo Zhao

**<details><summary>Abstract**</summary>

While distant supervision has been extensively explored and exploited in NLP tasks like named entity recognition, a major obstacle stems from the inevitable noisy distant labels tagged unsupervisedly. A few past works approach this problem by adopting a self-training framework with a sample-selection mechanism. In this work, we innovatively identify two types of biases that were omitted by prior work, and these biases lead to inferior performance of the distant-supervised NER setup. First, we characterize the noise concealed in the distant labels as highly structural rather than fully randomized. Second, the self-training framework would ubiquitously introduce an inherent bias that causes erroneous behavior in both sample selection and eventually prediction. To cope with these problems, we propose a novel self-training framework, dubbed DesERT. This framework augments the conventional NER predicative pathway to a dual form that effectively adapts the sample-selection process to conform to its innate distributional-bias structure. The other crucial component of DesERT composes a debiased module aiming to enhance the token representations, hence the quality of the pseudo-labels. Extensive experiments are conducted to validate the DesERT. The results show that our framework establishes a new state-of-art performance, it achieves a +2.22% average F1 score improvement on five standardized benchmarking datasets. Lastly, DesERT demonstrates its effectiveness under a new DSNER benchmark where additional distant supervision comes from the ChatGPT model.</details>

### Debiasing Conditional Stochastic Optimization

**Authors:** Lie He, Shiva Kasiviswanathan

**<details><summary>Abstract**</summary>

In this paper, we study the conditional stochastic optimization (CSO) problem which  covers a variety of applications including  portfolio selection, reinforcement learning, robust learning, causal inference, etc. The sample-averaged gradient of the CSO objective is biased due to its nested structure, and therefore requires a high sample complexity for convergence. We introduce a general stochastic extrapolation technique that effectively reduces the bias. We show that for nonconvex smooth objectives, combining this extrapolation with variance reduction techniques can achieve a significantly better sample complexity than the existing bounds. Additionally, we  develop new algorithms for the  finite-sum variant of the CSO problem that also significantly improve upon existing results. Finally, we believe that our debiasing technique has the potential to be a useful tool for addressing similar challenges in other stochastic optimization problems.</details>

### Debiasing Pretrained Generative Models by Uniformly Sampling Semantic Attributes

**Authors:** Walter Gerych, Kevin Hickey, Luke Buquicchio, Kavin Chandrasekaran, Abdulaziz Alajaji, Elke A. Rundensteiner, Emmanuel Agu

**<details><summary>Abstract**</summary>

Generative models are being increasingly used in science and industry applications. Unfortunately,  they often perpetuate the biases present in their training sets, such as societal biases causing certain groups to be underrepresented in the data. For instance, image generators may overwhelmingly produce images of white people due to few non-white samples in their training data. It is imperative to debias generative models so they synthesize an equal number of instances for each group, while not requiring retraining of the model to avoid  prohibitive expense. We thus propose a *distribution mapping module* that produces samples from a *fair noise distribution*, such that the  pretrained generative model produces *semantically uniform* outputs -  an equal number of instances for each group - when conditioned on these samples. This does *not* involve retraining the generator, nor does it require *any* real training data. Experiments on debiasing generators trained on popular real-world  datasets show that our method outperforms existing approaches.</details>

### [Spotlight] Decentralized Randomly Distributed Multi-agent Multi-armed Bandit with Heterogeneous Rewards

**Authors:** Mengfan Xu, Diego Klabjan

**<details><summary>Abstract**</summary>

We study a decentralized multi-agent multi-armed bandit problem in which multiple clients are connected by time dependent random graphs provided by an environment. The reward distributions of each arm vary across clients and rewards are generated independently over time by an environment based on distributions that include both sub-exponential and sub-gaussian distributions. Each client pulls an arm and communicates with neighbors based on the graph provided by the environment. The goal is to minimize the overall regret of the entire system through collaborations. To this end, we introduce a novel algorithmic framework, which first provides robust simulation methods for generating random graphs using rapidly mixing markov chains or the random graph model, and then combines an averaging-based consensus approach with a newly proposed weighting technique and the upper confidence bound to deliver a UCB-type solution. Our algorithms account for the randomness in the graphs, removing the conventional doubly stochasticity assumption, and only require the knowledge of the number of clients at initialization. We derive optimal instance-dependent regret upper bounds of order $\log{T}$ in both sub-gaussian and sub-exponential environments, and a nearly optimal instance-free regret upper bound of order $\sqrt{T}\log T$ up to a $\log T$ factor. Importantly, our regret bounds hold with high probability and capture graph randomness, whereas prior works consider expected regret under assumptions and require more stringent reward distributions.</details>

### Decision Tree for Locally Private Estimation with Public Data

**Authors:** Yuheng Ma, Han Zhang, Yuchao Cai, Hanfang Yang

**<details><summary>Abstract**</summary>

We propose conducting locally differentially private (LDP) estimation with the aid of a small amount of public data to enhance the performance of private estimation. Specifically, we introduce an efficient algorithm called Locally differentially Private Decision Tree (LPDT) for LDP regression. We first use the public data to grow a decision tree partition and then fit an estimator according to the partition privately.  From a theoretical perspective, we show that LPDT is $\varepsilon$-LDP and has a mini-max optimal convergence rate under a mild assumption of similarity between public and private data, whereas the lower bound of the convergence rate of LPDT without public data is strictly slower, which implies that the public data helps to improve the convergence rates of LDP estimation. We conduct experiments on both synthetic and real-world data to demonstrate the superior performance of LPDT compared with other state-of-the-art LDP regression methods. Moreover, we show that LPDT remains effective despite considerable disparities between public and private data.</details>

### Deep Non-line-of-sight Imaging from Under-scanning Measurements

**Authors:** Yue Li, Yueyi Zhang, Juntian Ye, Feihu Xu, Zhiwei Xiong

**<details><summary>Abstract**</summary>

Active confocal non-line-of-sight (NLOS) imaging has successfully enabled seeing around corners relying on high-quality transient measurements. However, acquiring spatial-dense transient measurement is time-consuming, raising the question of how to reconstruct satisfactory results from under-scanning measurements (USM). The existing solutions, involving the traditional algorithms, however, are hindered by unsatisfactory results or long computing times. To this end, we propose the first deep-learning-based approach to NLOS imaging from USM. Our proposed end-to-end network is composed of two main components: the transient recovery network (TRN) and the volume reconstruction network (VRN). Specifically, TRN takes the under-scanning measurements as input, utilizes a multiple kernel feature extraction module and a multiple feature fusion module, and outputs sufficient-scanning measurements at the high-spatial resolution. Afterwards, VRN incorporates the linear physics prior of the light-path transport model and reconstructs the hidden volume representation. Besides, we introduce regularized constraints that enhance the perception of more local details while suppressing smoothing effects. The proposed method achieves superior performance on both synthetic data and public real-world data, as demonstrated by extensive experimental results with different under-scanning grids. Moreover, the proposed method delivers impressive robustness at an extremely low scanning grid (i.e., 8$\times$8) and offers high-speed inference (i.e., 50 times faster than the existing iterative solution).</details>

### Deep Stochastic Processes via Functional Markov Transition Operators

**Authors:** Jin Xu, Emilien Dupont, Kaspar Märtens, Thomas Rainforth, Yee Whye Teh

**<details><summary>Abstract**</summary>

We introduce Markov Neural Processes (MNPs), a new class of Stochastic Processes (SPs) which are constructed by stacking sequences of neural parameterised Markov transition operators in function space. We prove that these Markov transition operators can preserve the exchangeability and consistency of SPs. Therefore, the proposed iterative construction adds substantial flexibility and expressivity to the original framework of Neural Processes (NPs) without compromising consistency or adding restrictions. Our experiments demonstrate clear advantages of MNPs over baseline models on a variety of tasks.</details>

### Defending Pre-trained Language Models as Few-shot Learners against Backdoor Attacks

**Authors:** Zhaohan Xi, Tianyu Du, Changjiang Li, Ren Pang, Shouling Ji, Jinghui Chen, Fenglong Ma, Ting Wang

**<details><summary>Abstract**</summary>

Pre-trained language models (PLMs) have demonstrated remarkable performance as few-shot learners. However, their security risks under such settings are largely unexplored. In this work, we conduct a pilot study showing that PLMs as few-shot learners are highly vulnerable to backdoor attacks while existing defenses are inadequate due to the unique challenges of few-shot scenarios. To address such challenges, we advocate MDP, a novel lightweight, pluggable, and effective defense for PLMs as few-shot learners. Specifically, MDP leverages the gap between the masking-sensitivity of poisoned and clean samples: with reference to the limited few-shot data as distributional anchors, it compares the representations of given samples under varying masking and identifies poisoned samples as ones with significant variations. We show analytically that MDP creates an interesting dilemma for the attacker to choose between attack effectiveness and detection evasiveness. The empirical evaluation using benchmark datasets and representative attacks validates the efficacy of MDP. The code of MDP is publicly available.</details>

### Delayed Algorithms for Distributed Stochastic Weakly Convex Optimization

**Authors:** Wenzhi Gao, Qi Deng

**<details><summary>Abstract**</summary>

This paper studies delayed stochastic algorithms for weakly convex optimization in a distributed network with workers connected to a master node.  Recently, Xu~et~al.~2022  showed that an inertial stochastic subgradient method   converges at a rate of $\mathcal{O}(\tau_{\text{max}}/\sqrt{K})$ which depends on the maximum information delay $\tau_{\text{max}}$. In this work, we show that the delayed stochastic subgradient method ($\texttt{DSGD}$) obtains a tighter convergence rate which depends on the expected delay $\bar{\tau}$. Furthermore, for an important class of composition weakly convex problems, we develop a new delayed stochastic prox-linear ($\texttt{DSPL}$) method in which the delays only affect the high-order term in the rate and hence,  are negligible after a certain number of $\texttt{DSPL}$  iterations.  In addition, we demonstrate the robustness of our proposed algorithms against arbitrary delays.  By incorporating a simple safeguarding step in both methods, we achieve convergence rates that depend solely on the number of workers, eliminating the effect of delays. Our numerical experiments further confirm the empirical superiority of our proposed methods.</details>

### Demographic Parity Constrained Minimax Optimal Regression under Linear Model

**Authors:** Kazuto Fukuchi, Jun Sakuma

**<details><summary>Abstract**</summary>

We explore the minimax optimal error associated with a demographic parity-constrained regression problem within the context of a linear model. Our proposed model encompasses a broader range of discriminatory bias sources compared to the model presented by Chzhen and Schreuder. Our analysis reveals that the minimax optimal error for the demographic parity-constrained regression problem under our model is characterized by $\Theta(\frac{dM}{n})$, where $n$ denotes the sample size, $d$ represents the dimensionality, and $M$ signifies the number of demographic groups arising from sensitive attributes. Moreover, we demonstrate that the minimax error increases in conjunction with a larger bias present in the model.</details>

### DesCo: Learning Object Recognition with Rich Language Descriptions

**Authors:** Liunian Li, Zi-Yi Dou, Nanyun Peng, Kai-Wei Chang

**<details><summary>Abstract**</summary>

Recent development in vision-language approaches has instigated a paradigm shift in learning visual recognition models from language supervision. These approaches align objects with language queries (e.g. "a photo of a cat") and thus improve the models' adaptability to novel objects and domains. Recent studies have attempted to query these models with complex language expressions that include specifications of fine-grained details, such as colors, shapes, and relations. However, simply incorporating language descriptions into queries does not guarantee accurate interpretation by the models. In fact, our experiments show that GLIP, a state-of-the-art vision-language model for object detection, often disregards contextual information in the language descriptions and instead relies heavily on detecting objects solely by their names. To tackle the challenge, we propose a new description-conditioned (DesCo) paradigm of learning object recognition models with rich language descriptions consisting of two innovations: 1) we employ a large language model as a commonsense knowledge engine to generate rich language descriptions of objects; 2) we design context-sensitive queries to improve the model's ability in deciphering intricate nuances embedded within descriptions and enforce the model to focus on context rather than object names alone.  On two novel object detection benchmarks, LVIS and OminiLabel, under the zero-shot detection setting, our approach achieves 34.8 APr minival (+9.1) and 29.3 AP (+3.6), respectively, surpassing the prior state-of-the-art models, GLIP and FIBER, by a large margin.</details>

### Describe, Explain, Plan and Select: Interactive Planning with LLMs Enables Open-World Multi-Task Agents

**Authors:** Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian (Shawn) Ma, Yitao Liang

**<details><summary>Abstract**</summary>

In this paper, we study the problem of planning in Minecraft, a popular, democratized yet challenging open-ended environment for developing multi-task embodied agents. We've found two primary challenges of empowering such agents with planning: 1) planning in an open-ended world like Minecraft requires precise and multi-step reasoning due to the long-term nature of the tasks, and 2) as vanilla planners do not consider the achievability of the current agent when ordering parallel sub-goals within a complicated plan, the resulting plan could be inefficient. To this end, we propose ``$\underline{D}$escribe, $\underline{E}$xplain, $\underline{P}$lan and $\underline{S}$elect'' ($\textbf{DEPS}$), an interactive planning approach based on Large Language Models (LLMs). Our approach helps with better error correction from the feedback during the long-haul planning, while also bringing the sense of proximity via goal $\textbf{Selector}$, a learnable module that ranks parallel sub-goals based on the estimated steps of completion and improves the original plan accordingly. Our experiments mark the milestone of the first zero-shot multi-task agent that can robustly accomplish 70+ Minecraft tasks and nearly double the overall performances. Further testing reveals our method's general effectiveness in popularly adopted non-open-ended domains as well (i.e., ALFWorld and tabletop manipulation). The ablation and exploratory studies detail how our design beats the counterparts and provide a promising update on the $\texttt{ObtainDiamond}$ grand challenge with our approach.</details>

### Detecting hidden confounding in observational data using multiple environments

**Authors:** Rickard Karlsson, Jesse Krijthe

**<details><summary>Abstract**</summary>

A common assumption in causal inference from observational data is that there is no hidden confounding. Yet it is, in general, impossible to verify the presence of hidden confounding factors from a single dataset. Under the assumption of independent causal mechanisms underlying the data-generating process, we demonstrate a way to detect unobserved confounders when having multiple observational datasets coming from different environments. We present a theory for testable conditional independencies that are only absent when there is hidden confounding and examine cases where we violate its assumptions: degenerate & dependent mechanisms, and faithfulness violations. Additionally, we propose a procedure to test these independencies and study its empirical finite-sample behavior using simulation studies and semi-synthetic data based on a real-world dataset. In most cases, the proposed procedure correctly predicts the presence of hidden confounding, particularly when the confounding bias is large.</details>

### Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models

**Authors:** Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, Zhihua Zhang

**<details><summary>Abstract**</summary>

Due to the ease of training, ability to scale, and high sample quality, diffusion models (DMs) have become the preferred option for generative modeling, with numerous pre-trained models available for a wide variety of datasets. Containing intricate information about data distributions, pre-trained DMs are valuable assets for downstream applications.  In this work, we consider learning from pre-trained DMs and transferring their knowledge to other generative models in a data-free fashion. Specifically, we propose a general framework called Diff-Instruct to instruct the training of arbitrary generative models as long as the generated samples are differentiable with respect to the model parameters. Our proposed Diff-Instruct is built on a rigorous mathematical foundation where the instruction process directly corresponds to minimizing a novel divergence we call Integral Kullback-Leibler (IKL) divergence. IKL is tailored for DMs by calculating the integral of the KL divergence along a diffusion process, which we show to be more robust in comparing distributions with misaligned supports. We also reveal non-trivial connections of our method to existing works such as DreamFusion, and generative adversarial training. To demonstrate the effectiveness and universality of Diff-Instruct, we consider two scenarios: distilling pre-trained diffusion models and refining existing GAN models. The experiments on distilling pre-trained diffusion models show that Diff-Instruct results in state-of-the-art single-step diffusion-based models. The experiments on refining GAN models show that the Diff-Instruct can consistently improve the pre-trained generators of GAN models across various settings.</details>

### DiffAttack: Evasion Attacks Against Diffusion-Based Adversarial Purification

**Authors:** Mintong Kang, Dawn Song, Bo Li

**<details><summary>Abstract**</summary>

Diffusion-based purification defenses leverage diffusion models to remove crafted perturbations of adversarial examples and achieve state-of-the-art robustness. Recent studies show that even advanced attacks cannot break such defenses effectively, since the purification process induces an extremely deep computational graph which poses the potential problem of gradient obfuscation, high memory cost, and unbounded randomness. In this paper, we propose a unified framework DiffAttack to perform effective and efficient attacks against diffusion-based purification defenses, including both DDPM and score-based approaches. In particular, we propose a deviated-reconstruction loss at intermediate diffusion steps to induce inaccurate density gradient estimation to tackle the problem of vanishing/exploding gradients. We also provide a segment-wise forwarding-backwarding algorithm, which leads to memory-efficient gradient backpropagation. We validate the attack effectiveness of DiffAttack compared with existing adaptive attacks on CIFAR-10 and ImageNet. We show that DiffAttack decreases the robust accuracy of models compared with SOTA attacks by over 20\% on CIFAR-10 under $\ell_\infty$ attack $(\epsilon=8/255)$, and over 10\% on ImageNet under $\ell_\infty$ attack $(\epsilon=4/255)$. We conduct a series of ablations studies, and we find 1) DiffAttack with the deviated-reconstruction loss added over uniformly sampled time steps is more effective than that added over only initial/final steps, and 2) diffusion-based purification with a moderate diffusion length is more robust under DiffAttack.</details>

### Differentiable Blocks World: Qualitative 3D Decomposition by Rendering Primitives

**Authors:** Tom Monnier, Jake Austin, Angjoo Kanazawa, Alexei Efros, Mathieu Aubry

**<details><summary>Abstract**</summary>

Given a set of calibrated images of a scene, we present an approach that produces a simple, compact, and actionable 3D world representation by means of 3D primitives. While many approaches focus on recovering high-fidelity 3D scenes, we focus on parsing a scene into mid-level 3D representations made of a small set of textured primitives. Such representations are interpretable, easy to manipulate and suited for physics-based simulations. Moreover, unlike existing primitive decomposition methods that rely on 3D input data, our approach operates directly on images through differentiable rendering. Specifically, we model primitives as textured superquadric meshes and optimize their parameters from scratch with an image rendering loss. We highlight the importance of modeling transparency for each primitive, which is critical for optimization and also enables handling varying numbers of primitives. We show that the resulting textured primitives faithfully reconstruct the input images and accurately model the visible 3D points, while providing amodal shape completions of unseen object regions. We compare our approach to the state of the art on diverse scenes from DTU, and demonstrate its robustness on real-life captures from BlendedMVS and Nerfstudio. We also showcase how our results can be used to effortlessly edit a scene or perform physical simulations. Code and video results are available at https://www.tmonnier.com/DBW.</details>

### [Oral] DiffuseBot: Breeding Soft Robots With Physics-Augmented Generative Diffusion Models

**Authors:** Tsun-Hsuan Johnson Wang, Juntian Zheng, Pingchuan Ma, Yilun Du, Byungchul Kim, Andrew Spielberg, Josh Tenenbaum, Chuang Gan, Daniela Rus

**Oral Presentation:** We, Dec 13, 08:30 -- Oral 3C

**<details><summary>Abstract**</summary>

Nature evolves creatures with a high complexity of morphological and behavioral intelligence, meanwhile computational methods lag in approaching that diversity and efficacy. Co-optimization of artificial creatures' morphology and control in silico shows promise for applications in physical soft robotics and virtual character creation; such approaches, however, require developing new learning algorithms that can reason about function atop pure structure. In this paper, we present DiffuseBot, a physics-augmented diffusion model that generates soft robot morphologies capable of excelling in a wide spectrum of tasks. 

ame bridges the gap between virtually generated content and physical utility by (i) augmenting the diffusion process with a physical dynamical simulation which provides a certificate of performance, and (ii) introducing a co-design procedure that jointly optimizes physical design and control by leveraging information about physical sensitivities from differentiable simulation. We showcase a range of simulated and fabricated robots along with their capabilities. Check our website: https://diffusebot.github.io/</details>

### Diffused Redundancy in Pre-trained Representations

**Authors:** Vedant Nanda, Till Speicher, John Dickerson, Krishna Gummadi, Soheil Feizi, Adrian Weller

**<details><summary>Abstract**</summary>

Representations learned by pre-training a neural network on a large dataset are increasingly used successfully to perform a variety of downstream tasks. In this work, we take a closer look at how features are encoded in such pre-trained representations. We find that learned representations in a given layer exhibit a degree of diffuse redundancy, ie, any randomly chosen subset of neurons in the layer that is larger than a threshold size shares a large degree of similarity with the full layer and is able to perform similarly as the whole layer on a variety of downstream tasks. For example, a linear probe trained on $20\%$ of randomly picked neurons from the penultimate layer of a ResNet50 pre-trained on ImageNet1k achieves an accuracy within $5\%$ of a linear probe trained on the full layer of neurons for downstream CIFAR10 classification. We conduct experiments on different neural architectures (including CNNs and Transformers) pre-trained on both ImageNet1k and ImageNet21k and evaluate a variety of downstream tasks taken from the VTAB benchmark. We find that the loss \& dataset used during pre-training largely govern the degree of diffuse redundancy and the "critical mass" of neurons needed often depends on the downstream task, suggesting that there is a task-inherent redundancy-performance Pareto frontier. Our findings shed light on the nature of representations learned by pre-trained deep neural networks and suggest that entire layers might not be necessary to perform many downstream tasks. We investigate the potential for exploiting this redundancy to achieve efficient generalization for downstream tasks and also draw caution to certain possible unintended consequences. Our code is available at \url{https://github.com/nvedant07/diffused-redundancy}.</details>

### [Spotlight] Diffusion Models and Semi-Supervised Learners Benefit Mutually with Few Labels

**Authors:** Zebin You, Yong Zhong, Fan Bao, Jiacheng Sun, Chongxuan LI, Jun Zhu

**<details><summary>Abstract**</summary>

In an effort to further advance semi-supervised generative and classification tasks, we propose a simple yet effective training strategy called *dual pseudo training* (DPT), built upon strong semi-supervised learners and diffusion models. DPT operates in three stages: training a classifier on partially labeled data to predict pseudo-labels; training a conditional generative model using these pseudo-labels to generate pseudo images; and retraining the classifier with a mix of real and pseudo images. Empirically, DPT consistently achieves SOTA performance of semi-supervised generation and classification across various settings. In particular, with one or two labels per class, DPT achieves a Fréchet Inception Distance (FID) score of 3.08 or 2.52 on ImageNet $256	imes256$. Besides, DPT outperforms competitive semi-supervised baselines substantially on ImageNet classification tasks, *achieving top-1 accuracies of 59.0 (+2.8), 69.5 (+3.0), and 74.4 (+2.0)* with one, two, or five labels per class, respectively. Notably, our results demonstrate that diffusion can generate realistic images with only a few labels (e.g., $<0.1$%) and generative augmentation remains viable for semi-supervised classification. Our code is available at *https://github.com/ML-GSAI/DPT*.</details>

### Diffusion-Based Probabilistic Uncertainty Estimation for Active Domain Adaptation

**Authors:** Zhekai Du, Jingjing Li

**<details><summary>Abstract**</summary>

Active Domain Adaptation (ADA) has emerged as an attractive technique for assisting domain adaptation by actively annotating a small subset of target samples. Most ADA methods focus on measuring the target representativeness beyond traditional active learning criteria to handle the domain shift problem, while leaving the uncertainty estimation to be performed by an uncalibrated deterministic model. In this work, we introduce a probabilistic framework that captures both data-level and prediction-level uncertainties beyond a point estimate. Specifically, we use variational inference to approximate the joint posterior distribution of latent representation and model prediction. The variational objective of labeled data can be formulated by a variational autoencoder and a latent diffusion classifier, and the objective of unlabeled data can be implemented in a knowledge distillation framework. We utilize adversarial learning to ensure an invariant latent space. The resulting diffusion classifier enables efficient sampling of all possible predictions for each individual to recover the predictive distribution. We then leverage a t-test-based criterion upon the sampling and select informative unlabeled target samples based on the p-value, which encodes both prediction variability and cross-category ambiguity. Experiments on both ADA and Source-Free ADA settings show that our method provides more calibrated predictions than previous ADA methods and achieves favorable performance on three domain adaptation datasets.</details>

### Direct Preference-based Policy Optimization without Reward Modeling

**Authors:** Gaon An, Junhyeok Lee, Xingdong Zuo, Norio Kosaka, Kyung-Min Kim, Hyun Oh Song

**<details><summary>Abstract**</summary>

Preference-based reinforcement learning (PbRL) is an approach that enables RL agents to learn from preference, which is particularly useful when formulating a reward function is challenging. Existing PbRL methods generally involve a two-step procedure: they first learn a reward model based on given preference data and then employ off-the-shelf reinforcement learning algorithms using the learned reward model. However, obtaining an accurate reward model solely from preference information, especially when the preference is from human teachers, can be difficult. Instead, we propose a PbRL algorithm that directly learns from preference without requiring any reward modeling. To achieve this, we adopt a contrastive learning framework to design a novel policy scoring metric that assigns a high score to policies that align with the given preferences. We apply our algorithm to offline RL tasks with actual human preference labels and show that our algorithm outperforms or is on par with the existing PbRL methods. Notably, on high-dimensional control tasks, our algorithm surpasses offline RL methods that learn with ground-truth reward information. Finally, we show that our algorithm can be successfully applied to fine-tune large language models.</details>

### Disambiguated Attention Embedding for Multi-Instance Partial-Label Learning

**Authors:** Wei Tang, Weijia Zhang, Min-Ling Zhang

**<details><summary>Abstract**</summary>

In many real-world tasks, the concerned objects can be represented as a multi-instance bag associated with a candidate label set, which consists of one ground-truth label and several false positive labels. Multi-instance partial-label learning (MIPL) is a learning paradigm to deal with such tasks and has achieved favorable performances. Existing MIPL approach follows the instance-space paradigm by assigning augmented candidate label sets of bags to each instance and aggregating bag-level labels from instance-level labels. However, this scheme may be suboptimal as global bag-level information is ignored and the predicted labels of bags are sensitive to predictions of negative instances. In this paper, we study an alternative scheme where a multi-instance bag is embedded into a single vector representation. Accordingly, an intuitive algorithm named DEMIPL, i.e., Disambiguated attention Embedding for Multi-Instance Partial-Label learning, is proposed. DEMIPL employs a disambiguation attention mechanism to aggregate a multi-instance bag into a single vector representation, followed by a momentum-based disambiguation strategy to identify the ground-truth label from the candidate label set. Furthermore, we introduce a real-world MIPL dataset for colorectal cancer classification. Experimental results on benchmark and real-world datasets validate the superiority of DEMIPL against the compared MIPL and partial-label learning approaches.</details>

### Discrete-Smoothness in Online Algorithms with Predictions

**Authors:** Yossi Azar, Debmalya Panigrahi, Noam Touitou

**<details><summary>Abstract**</summary>

In recent years, there has been an increasing focus on designing online algorithms with (machine-learned) predictions. The ideal learning-augmented algorithm is comparable to the optimum when given perfect predictions (consistency), to the best online approximation for arbitrary predictions (robustness), and should interpolate between these extremes as a smooth function of the prediction error. In this paper, we quantify these guarantees in terms of a general property that we call discrete-smoothness, and achieve discrete-smooth algorithms for online covering, specifically the facility location and set cover problems. For set cover, our work improves the results of Bamas, Maggiori, and Svensson (2020) by augmenting consistency and robustness with smoothness guarantees. For facility location, our work improves on prior work by Almanza et al. (2021) by generalizing to nonuniform costs and also providing smoothness guarantees by augmenting consistency and robustness.</details>

### Disentanglement via Latent Quantization

**Authors:** Kyle Hsu, William Dorrell, James Whittington, Jiajun Wu, Chelsea Finn

**<details><summary>Abstract**</summary>

In disentangled representation learning, a model is asked to tease apart a dataset's underlying sources of variation and represent them independently of one another. Since the model is provided with no ground truth information about these sources, inductive biases take a paramount role in enabling disentanglement. In this work, we construct an inductive bias towards encoding to and decoding from an organized latent space. Concretely, we do this by (i) quantizing the latent space into discrete code vectors with a separate learnable scalar codebook per dimension and (ii) applying strong model regularization via an unusually high weight decay. Intuitively, the latent space design forces the encoder to combinatorially construct codes from a small number of distinct scalar values, which in turn enables the decoder to assign a consistent meaning to each value. Regularization then serves to drive the model towards this parsimonious strategy. We demonstrate the broad applicability of this approach by adding it to both basic data-reconstructing (vanilla autoencoder) and latent-reconstructing (InfoGAN) generative models. For reliable evaluation, we also propose InfoMEC, a new set of metrics for disentanglement that is cohesively grounded in information theory and fixes well-established shortcomings in previous metrics. Together with regularization, latent quantization dramatically improves the modularity and explicitness of learned representations on a representative suite of benchmark datasets. In particular, our quantized-latent autoencoder (QLAE) consistently outperforms strong methods from prior work in these key disentanglement properties without compromising data reconstruction.</details>

### Disentangling Voice and Content with Self-Supervision for Speaker Recognition

**Authors:** TIANCHI LIU, Kong Aik Lee, Qiongqiong Wang, Haizhou Li

**<details><summary>Abstract**</summary>

For speaker recognition, it is difficult to extract an accurate  speaker representation from speech because of its mixture of speaker traits and content. This paper proposes a disentanglement framework that simultaneously models speaker traits and content variability in speech. It is realized with the use of three Gaussian inference layers, each consisting of a learnable transition model that extracts distinct speech components. Notably, a strengthened transition model is specifically designed to model complex speech dynamics. We also propose a self-supervision method to dynamically disentangle content without the use of labels other than speaker identities.  The efficacy of the proposed framework is validated via experiments conducted on the VoxCeleb and SITW datasets with 9.56\% and 8.24\% average reductions in EER and minDCF, respectively. Since neither additional model training nor data is specifically needed, it is easily applicable in practical use.</details>

### Distributed Personalized Empirical Risk Minimization

**Authors:** Yuyang Deng, Mohammad Mahdi Kamani, Pouria Mahdavinia, Mehrdad Mahdavi

**<details><summary>Abstract**</summary>

This paper advocates a new paradigm  Personalized Empirical Risk Minimization (PERM) to facilitate learning from heterogeneous  data sources without imposing stringent constraints on computational resources shared by participating devices. In PERM, we aim at  learning  a distinct model for each client by personalizing the aggregation of local empirical losses by effectively estimating the  statistical discrepancy among data distributions, which entails  optimal statistical accuracy for all local distributions and overcomes the data heterogeneity issue.  To learn personalized models at scale,  we propose a distributed algorithm that replaces the standard model averaging with model shuffling to simultaneously optimize PERM objectives for all devices. This also allows to learn distinct model architectures (e.g., neural networks with different number of parameters) for  different clients, thus confining to  underlying  memory and compute resources of individual clients. We rigorously analyze the convergence of proposed algorithm and  conduct experiments  that corroborates the effectiveness of proposed paradigm.</details>

### Distribution Learnability and Robustness

**Authors:** Shai Ben-David, Alex Bie, Gautam Kamath, Tosca Lechner

**<details><summary>Abstract**</summary>

We examine the relationship between learnability and robust learnability for the problem of distribution learning.We show that learnability implies robust learnability if the adversary can only perform additive contamination (and consequently, under Huber contamination), but not if the adversary is allowed to perform subtractive contamination. Thus, contrary to other learning settings (e.g., PAC learning of function classes), realizable learnability does not imply agnostic learnability. We also explore related implications in the context of compression schemes and  differentially private learnability.</details>

### Distributional Learning of Variational AutoEncoder: Application to Synthetic Data Generation

**Authors:** Seunghwan An, Jong-June Jeon

**<details><summary>Abstract**</summary>

The Gaussianity assumption has been consistently criticized as a main limitation of the Variational Autoencoder (VAE) despite its efficiency in computational modeling. In this paper, we propose a new approach that expands the model capacity (i.e., expressive power of distributional family) without sacrificing the computational advantages of the VAE framework. Our VAE model's decoder is composed of an infinite mixture of asymmetric Laplace distribution, which possesses general distribution fitting capabilities for continuous variables. Our model is represented by a special form of a nonparametric M-estimator for estimating general quantile functions, and we theoretically establish the relevance between the proposed model and quantile estimation. We apply the proposed model to synthetic data generation, and particularly, our model demonstrates superiority in easily adjusting the level of data privacy.</details>

### Does Invariant Graph Learning via Environment Augmentation Learn Invariance?

**Authors:** Yongqiang Chen, Yatao Bian, Kaiwen Zhou, Binghui Xie, Bo Han, James Cheng

**<details><summary>Abstract**</summary>

Invariant graph representation learning aims to learn the invariance among data from different environments for out-of-distribution generalization on graphs. As the graph environment partitions are usually expensive to obtain, augmenting the environment information has become the de facto approach. However, the usefulness of the augmented environment information has never been verified. In this work, we find that it is fundamentally impossible to learn invariant graph representations via environment augmentation without additional assumptions. Therefore, we develop a set of minimal assumptions, including variation sufficiency and variation consistency, for feasible invariant graph learning. We then propose a new framework Graph invAriant Learning Assistant (GALA). GALA incorporates an assistant model that needs to be sensitive to graph environment changes or distribution shifts. The correctness of the proxy predictions by the assistant model hence can differentiate the variations in spurious subgraphs. We show that extracting the maximally invariant subgraph to the proxy predictions provably identifies the underlying invariant subgraph for successful OOD generalization under the established minimal assumptions. Extensive experiments on datasets including DrugOOD with various graph distribution shifts confirm the effectiveness of GALA.</details>

### Does Visual Pretraining Help End-to-End Reasoning?

**Authors:** Chen Sun, Calvin Luo, Xingyi Zhou, Anurag Arnab, Cordelia Schmid

**<details><summary>Abstract**</summary>

We aim to investigate whether end-to-end learning of visual reasoning can be achieved with general-purpose neural networks, with the help of visual pretraining. A positive result would refute the common belief that explicit visual abstraction (e.g. object detection) is essential for compositional generalization on visual reasoning, and confirm the feasibility of a neural network ''generalist'' to solve visual recognition and reasoning tasks. We propose a simple and general self-supervised framework which ''compresses'' each video frame into a small set of tokens with a transformer network, and reconstructs the remaining frames based on the compressed temporal context. To minimize the reconstruction loss, the network must learn a compact representation for each image, as well as capture temporal dynamics and object permanence from temporal context. We perform evaluation on two visual reasoning benchmarks, CATER and ACRE. We observe that pretraining is essential to achieve compositional generalization for end-to-end visual reasoning. Our proposed framework outperforms traditional supervised pretraining, including image classification and explicit object detection, by large margins.</details>

### Domain Adaptive Imitation Learning with Visual Observation

**Authors:** Sungho Choi, Seungyul Han, Woojun Kim, Jongseong Chae, Whiyoung Jung, Youngchul Sung

**<details><summary>Abstract**</summary>

In this paper, we consider domain-adaptive imitation learning with visual observation, where an agent in a target domain learns to perform a task by observing expert demonstrations in a source domain. Domain adaptive imitation learning arises in practical scenarios where a robot, receiving visual sensory data, needs to mimic movements by visually observing other robots from different angles or observing robots of different shapes. To overcome the domain shift in cross-domain imitation learning with visual observation, we propose a novel framework for extracting domain-independent behavioral features from input observations that can be used to train the learner, based on dual feature extraction and image reconstruction. Empirical results demonstrate that our approach outperforms previous algorithms for imitation learning from visual observation with domain shift.</details>

### Domain Watermark: Effective and Harmless Dataset Copyright Protection is Closed at Hand

**Authors:** Junfeng Guo, Yiming Li, Lixu Wang, Shu-Tao Xia, Heng Huang, Cong Liu, Bo Li

**<details><summary>Abstract**</summary>

The prosperity of deep neural networks (DNNs) is largely benefited from open-source datasets, based on which users can evaluate and improve their methods. In this paper, we revisit backdoor-based dataset ownership verification (DOV), which is currently the only feasible approach to protect the copyright of open-source datasets. We reveal that these methods are fundamentally harmful given that they could introduce malicious misclassification behaviors to watermarked DNNs by the adversaries. In this paper, we design DOV from another perspective by making watermarked models (trained on the protected dataset) correctly classify some `hard' samples that will be misclassified by the benign model. Our method is inspired by the generalization property of DNNs, where we find a \emph{hardly-generalized domain} for the original dataset (as its \emph{domain watermark}). It can be easily learned with the protected dataset containing modified samples. Specifically, we formulate the domain generation as a bi-level optimization and propose to optimize a set of visually-indistinguishable clean-label modified data with similar effects to domain-watermarked samples from the hardly-generalized domain to ensure watermark stealthiness. We also design a hypothesis-test-guided ownership verification via our domain watermark and provide the theoretical analyses of our method. Extensive experiments on three benchmark datasets are conducted, which verify the effectiveness of our method and its resistance to potential adaptive methods.</details>

### Don’t just prune by magnitude! Your mask topology is a secret weapon

**Authors:** Duc Hoang, Souvik Kundu, Shiwei Liu, Zhangyang "Atlas" Wang

**<details><summary>Abstract**</summary>

Recent years have witnessed significant progress in understanding the relationship between the connectivity of a deep network's architecture as a graph, and the network's performance. A few prior arts connected deep architectures to expander graphs or Ramanujan graphs, and particularly,[7] demonstrated the use of such graph connectivity measures with ranking and relative performance of various obtained sparse sub-networks (i.e. models with prune masks) without the need for training. However, no prior work explicitly explores the role of parameters in the graph's connectivity, making the graph-based understanding of prune masks and the magnitude/gradient-based pruning practice isolated from one another. This paper strives to fill in this gap, by analyzing the Weighted Spectral Gap of Ramanujan structures in sparse neural networks and investigates its correlation with final performance. We specifically examine the evolution of sparse structures under a popular dynamic sparse-to-sparse network training scheme, and intriguingly find that the generated random topologies inherently maximize Ramanujan graphs. We also identify a strong correlation between masks, performance, and the weighted spectral gap. Leveraging this observation, we propose to construct a new "full-spectrum coordinate'' aiming to comprehensively characterize a sparse neural network's promise. Concretely, it consists of the classical Ramanujan's gap (structure), our proposed weighted spectral gap (parameters), and the constituent nested regular graphs within. In this new coordinate system, a sparse subnetwork's L2-distance from its original initialization is found to have nearly linear correlated with its performance. Eventually, we apply this unified perspective to develop a new actionable pruning method, by sampling sparse masks to maximize the L2-coordinate distance. Our method can be augmented with the "pruning at initialization" (PaI) method, and significantly outperforms existing PaI methods. With only a few iterations of training (e.g 500 iterations), we can get LTH-comparable performance as that yielded via "pruning after training", significantly saving pre-training costs. Codes can be found at: https://github.com/VITA-Group/FullSpectrum-PAI.</details>

### [Spotlight] DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data

**Authors:** Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, Phillip Isola

**<details><summary>Abstract**</summary>

Current perceptual similarity metrics operate at the level of pixels and patches. These metrics compare images in terms of their low-level colors and textures, but fail to capture mid-level similarities and differences in image layout, object pose, and semantic content. In this paper, we develop a perceptual metric that assesses images holistically. Our first step is to collect a new dataset of human similarity judgments over image pairs that are alike in diverse ways. Critical to this dataset is that judgments are nearly automatic and shared by all observers. To achieve this we use recent text-to-image models to create synthetic pairs that are perturbed along various dimensions. We observe that popular perceptual metrics fall short of explaining our new data, and we introduce a new metric, DreamSim, tuned to better align with human perception. We analyze how our metric is affected by different visual attributes, and find that it focuses heavily on foreground objects and semantic content while also being sensitive to color and layout. Notably, despite being trained on synthetic data, our metric generalizes to real images, giving strong results on retrieval and reconstruction tasks. Furthermore, our metric outperforms both prior learned metrics and recent large vision models on these tasks. Our project page: https://dreamsim-nights.github.io/</details>

### DropCompute: simple and more robust  distributed synchronous training via compute variance reduction

**Authors:** Niv Giladi, Shahar Gottlieb, moran shkolnik, Asaf Karnieli, Ron Banner, Elad Hoffer, Kfir Y. Levy, Daniel Soudry

**<details><summary>Abstract**</summary>

Background: Distributed training is essential for large scale training of deep neural networks (DNNs). The dominant methods for large scale DNN training are synchronous (e.g. All-Reduce), but these require waiting for all workers in each step. Thus, these methods are limited by the delays caused by straggling workers.Results: We study a typical scenario in which workers are straggling due to variability in compute time. We find an analytical relation between compute time properties and scalability limitations, caused by such straggling workers. With these findings, we propose a simple yet effective decentralized method to reduce the variation among workers and thus improve the robustness of synchronous training. This method can be integrated with the widely used All-Reduce. Our findings are validated on large-scale training tasks using 200 Gaudi Accelerators.</details>

### [Spotlight] Dynamic Tensor Decomposition via Neural Diffusion-Reaction Processes

**Authors:** Zheng Wang, Shikai Fang, Shibo Li, Shandian Zhe

**<details><summary>Abstract**</summary>

Tensor decomposition is an important tool for multiway data analysis. In practice, the data is often sparse yet associated with rich temporal information. Existing methods, however, often under-use the time information and ignore the structural knowledge within the sparsely observed tensor entries. To overcome these limitations and to better capture the underlying temporal structure, we propose Dynamic EMbedIngs fOr dynamic Tensor dEcomposition (DEMOTE). We develop a neural diffusion-reaction process to estimate dynamic embeddings for the entities in each tensor mode. Specifically, based on the observed tensor entries, we build a multi-partite graph to encode the correlation between the entities. We construct a graph diffusion process to co-evolve the embedding trajectories of the correlated entities and use a neural network to construct a reaction process for each individual entity. In this way, our model can capture both the commonalities and personalities during the evolution of the embeddings for different entities. We then use a neural network to model the entry value as a nonlinear function of the embedding trajectories. For model estimation, we combine ODE solvers to develop a stochastic mini-batch learning algorithm. We propose a stratified sampling method to balance the cost of processing each mini-batch so as to improve the overall efficiency. We show the advantage of our approach in both simulation studies and real-world applications. The code is available at https://github.com/wzhut/Dynamic-Tensor-Decomposition-via-Neural-Diffusion-Reaction-Processes.</details>

### Echoes Beyond Points: Unleashing the Power of Raw Radar Data in Multi-modality Fusion

**Authors:** Yang Liu, Feng Wang, Naiyan Wang, ZHAO-XIANG ZHANG

**<details><summary>Abstract**</summary>

Radar is ubiquitous in autonomous driving systems due to its low cost and good adaptability to bad weather. Nevertheless, the radar detection performance is usually inferior because its point cloud is sparse and not accurate due to the poor azimuth and elevation resolution. Moreover, point cloud generation algorithms already drop weak signals to reduce the false targets which may be suboptimal for the use of deep fusion. In this paper, we propose a novel method named EchoFusion to skip the existing radar signal processing pipeline and then incorporate the radar raw data with other sensors. Specifically, we first generate the Bird's Eye View (BEV) queries and then take corresponding spectrum features from radar to fuse with other sensors. By this approach, our method could utilize both rich and lossless distance and speed clues from radar echoes and rich semantic clues from images, making our method surpass all existing methods on the RADIal dataset, and approach the performance of LiDAR. The code will be released on https://github.com/tusen-ai/EchoFusion.</details>

### Ecosystem-level Analysis of Deployed Machine Learning Reveals Homogeneous Outcomes

**Authors:** Connor Toups, Rishi Bommasani, Kathleen Creel, Sarah Bana, Dan Jurafsky, Percy Liang

**<details><summary>Abstract**</summary>

Machine learning is traditionally studied at the model level: researchers measure and improve the accuracy, robustness, bias, efficiency, and other dimensions of specific models. In practice, however, the societal impact of any machine learning model is partially determined by the context into which it is deployed. To capture this, we introduce *ecosystem-level analysis:* rather than analyzing a single model, we consider the collection of models that are deployed in a given context. For example, ecosystem-level analysis in hiring recognizes that a job candidate’s outcomes are determined not only by a single hiring algorithm or firm but instead by the collective decisions of all the firms to which the candidate applied. Across three modalities (text, images, speech) and 11 datasets, we establish a clear trend: deployed machine learning is prone to systemic failure, meaning some users are exclusively misclassified by all models available. Even when individual models improve at the population level over time, we find these improvements rarely reduce the prevalence of systemic failure. Instead, the benefits of these improvements predominantly accrue to individuals who are already correctly classified by other models. In light of these trends, we analyze medical imaging for dermatology, a setting where the costs of systemic failure are especially high. While traditional analyses reveal that both models and humans exhibit racial performance disparities, ecosystem-level analysis reveals new forms of racial disparity in model predictions that do not present in human predictions. These examples demonstrate that ecosystem-level analysis has unique strengths in characterizing the societal impact of machine learning.</details>

### Effective Robustness against Natural Distribution Shifts for Models with Different Training Data

**Authors:** Zhouxing Shi, Nicholas Carlini, Ananth Balashankar, Ludwig Schmidt, Cho-Jui Hsieh, Alex Beutel, Yao Qin

**<details><summary>Abstract**</summary>

``Effective robustness'' measures the extra out-of-distribution (OOD) robustness beyond what can be predicted from the in-distribution (ID) performance. Existing effective robustness evaluations typically use a single test set such as ImageNet to evaluate the ID accuracy. This becomes problematic when evaluating models trained on different data distributions, e.g., comparing models trained on ImageNet vs. zero-shot language-image pre-trained models trained on LAION. In this paper, we propose a new evaluation metric to evaluate and compare the effective robustness of models trained on different data. To do this, we control for the accuracy on multiple ID test sets that cover the training distributions for all the evaluated models. Our new evaluation metric provides a better estimate of effective robustness when there are models with different training data. It may also explain the surprising effective robustness gains of zero-shot CLIP-like models exhibited in prior works that used ImageNet as the only ID test set, while the gains diminish under our new evaluation. Additional artifacts including interactive visualizations are provided at https://shizhouxing.github.io/effective-robustness.</details>

### Efficient Batched Algorithm for Contextual Linear Bandits with Large Action Space via Soft Elimination

**Authors:** Osama Hanna, Lin Yang, Christina Fragouli

**<details><summary>Abstract**</summary>

In this paper, we provide the first efficient batched algorithm for contextual linear bandits with large action spaces. Unlike existing batched algorithms that rely on action elimination, which are not implementable for large action sets, our algorithm only uses a linear optimization oracle over the action set to design the policy. The proposed algorithm achieves a regret upper bound $\tilde{O}(\sqrt{T})$ with high probability,  and uses $O(\log\log T)$ batches, matching the lower bound on the number of batches (Gao et al., 2019). When specialized to linear bandits, our algorithm can achieve a high probability gap-dependent regret bound of $\tilde{O}(1/\Delta_{\min})$ with the optimal  $\log T$ number of batches, where $\Delta_{\min}$ is the minimum reward gap between a suboptimal arm and the optimal. Our result is achieved via a novel soft elimination approach, that entails $\text{``}$shaping$\text{"}$ the action sets at each batch so that we can efficiently identify (near) optimal actions.</details>

### Efficient Beam Tree Recursion

**Authors:** Jishnu Ray Chowdhury, Cornelia Caragea

**<details><summary>Abstract**</summary>

Beam Tree Recursive Neural Network (BT-RvNN) was recently proposed as an extension of Gumbel Tree RvNN and it was shown to achieve state-of-the-art length generalization performance in ListOps while maintaining comparable performance on other tasks. However, although better than previous approaches in terms of memory usage, BT-RvNN can be still exorbitantly expensive. In this paper, we identify the main bottleneck in BT-RvNN's memory usage to be the entanglement of the scorer function and the recursive cell function. We propose strategies to remove this bottleneck and further simplify its memory usage. Overall, our strategies not only reduce the memory usage of BT-RvNN by $10-16$ times but also create a new state-of-the-art in ListOps while maintaining similar performance in other tasks. In addition, we also propose a strategy to utilize the induced latent-tree node representations produced by BT-RvNN to turn BT-RvNN from a sentence encoder of the form...into a token contextualizer of the form.... Thus, our proposals not only open up a path for further scalability of RvNNs but also standardize a way to use BT-RvNNs as another building block in the deep learning toolkit that can be easily stacked or interfaced with other popular models such as Transformers and Structured State Space models. Our code is available at the link: https://github.com/JRC1995/BeamRecursionFamily.</details>

### Efficient Low-rank Backpropagation for Vision Transformer Adaptation

**Authors:** Yuedong Yang, Hung-Yueh Chiang, Guihong Li, Diana Marculescu, Radu Marculescu

**<details><summary>Abstract**</summary>

The increasing scale of vision transformers (ViT) has made the efficient fine-tuning of these large models for specific needs a significant challenge in various applications. This issue originates from the computationally demanding matrix multiplications required during the backpropagation process through linear layers in ViT.In this paper, we tackle this problem by proposing a new Low-rank BackPropagation via Walsh-Hadamard Transformation (LBP-WHT) method. Intuitively, LBP-WHT projects the gradient into a low-rank space and carries out backpropagation. This approach substantially reduces the computation needed for adapting ViT, as matrix multiplication in the low-rank space is far less resource-intensive. We conduct extensive experiments with different models (ViT, hybrid convolution-ViT model) on multiple datasets to demonstrate the effectiveness of our method. For instance, when adapting an EfficientFormer-L1 model on CIFAR100, our LBP-WHT achieves 10.4\% higher accuracy than the state-of-the-art baseline, while requiring 9 MFLOPs less computation.As the first work to accelerate ViT adaptation with low-rank backpropagation, our LBP-WHT method is complementary to many prior efforts and can be combined with them for better performance.</details>

### Efficient Model-Free Exploration in Low-Rank MDPs

**Authors:** Zak Mhammedi, Adam Block, Dylan J Foster, Alexander Rakhlin

**<details><summary>Abstract**</summary>

A major challenge in reinforcement learning is to develop practical, sample-efficient algorithms for exploration in high-dimensional domains where generalization and function approximation is required. Low-Rank Markov Decision Processes---where transition probabilities admit a low-rank factorization based on an unknown feature embedding---offer a simple, yet expressive framework for RL with function approximation, yet existing algorithms either (1) are computationally intractable, or (2) require restrictive statistical assumptions such as latent variable structure or access to model-based function approximation. In this work, we propose the first provably sample-efficient algorithm for exploration in Low-Rank MDPs that is both computationally efficient and model-free, allowing for general function approximation while requiring no structural assumptions beyond a reachability condition that we show is substantially weaker than that assumed in prior work. Our algorithm, SpanRL, uses the notion of a barycentric spanner for the feature embedding as an efficiently computable basis for exploration, performing efficient spanner computation by interleaving representation learning and policy optimization subroutines. Our analysis---which is appealingly simple and modular---carefully combines several techniques, including a new approach to error-tolerant barycentric spanner computation, and a new analysis of a certain minimax representation learning objective found in prior work.</details>

### Efficient Test-Time Adaptation for Super-Resolution with Second-Order Degradation and Reconstruction

**Authors:** Zeshuai Deng, Zhuokun Chen, Shuaicheng Niu, Thomas Li, Bohan Zhuang, Mingkui Tan

**<details><summary>Abstract**</summary>

Image super-resolution (SR) aims to learn a mapping from low-resolution (LR) to high-resolution (HR) using paired HR-LR training images. Conventional SR methods typically gather the paired training data by synthesizing LR images from HR images using a predetermined degradation model, e.g., Bicubic down-sampling. However, the realistic degradation type of test images may mismatch with the training-time degradation type due to the dynamic changes of the real-world scenarios, resulting in inferior-quality SR images. To address this, existing methods attempt to estimate the degradation model and train an image-specific model, which, however, is quite time-consuming and impracticable to handle rapidly changing domain shifts. Moreover, these methods largely concentrate on the estimation of one degradation type (e.g., blur degradation), overlooking other degradation types like noise and JPEG in real-world test-time scenarios, thus limiting their practicality. To tackle these problems, we present an efficient test-time adaptation framework for SR, named SRTTA, which is able to quickly adapt SR models to test domains with different/unknown degradation types. Specifically, we design a second-order degradation scheme to construct paired data based on the degradation type of the test image, which is predicted by a pre-trained degradation classifier. Then, we adapt the SR model by implementing feature-level reconstruction learning from the initial test image to its second-order degraded counterparts, which helps the SR model generate plausible HR images. Extensive experiments are conducted on newly synthesized corrupted DIV2K datasets with 8 different degradations and several real-world datasets, demonstrating that our SRTTA framework achieves an impressive improvement over existing methods with satisfying speed. The source code is available at https://github.com/DengZeshuai/SRTTA.</details>

### Efficient Training of Energy-Based Models Using Jarzynski Equality

**Authors:** Davide Carbone, Mengjian Hua, Simon Coste, Eric Vanden-Eijnden

**<details><summary>Abstract**</summary>

Energy-based models (EBMs) are generative models inspired by statistical physics with a wide range of applications in unsupervised learning.  Their performance is well measured by the cross-entropy (CE) of the model distribution relative to the data distribution. Using the CE as the objective for training is however challenging because the computation of its gradient with respect to the model parameters requires sampling the model distribution. Here we show how results for nonequilibrium thermodynamics based on Jarzynski equality together with tools from sequential Monte-Carlo sampling can be used to perform this computation efficiently and avoid the uncontrolled approximations made using the standard contrastive divergence algorithm.  Specifically, we introduce a modification of the unadjusted Langevin algorithm (ULA) in which each walker acquires a  weight that enables the estimation of the gradient of the cross-entropy at any step during GD, thereby bypassing sampling biases induced by slow mixing of ULA. We illustrate these results with numerical experiments on Gaussian mixture distributions as well as the MNIST and CIFAR-10 datasets. We show that the proposed approach outperforms  methods based on the contrastive divergence algorithm in all the considered situations.</details>

### EgoDistill: Egocentric Head Motion Distillation for Efficient Video Understanding

**Authors:** Shuhan Tan, Tushar Nagarajan, Kristen Grauman

**<details><summary>Abstract**</summary>

Recent advances in egocentric video understanding models are promising, but their heavy computational expense is a barrier for many real-world applications. To address this challenge, we propose EgoDistill, a distillation-based approach that learns to reconstruct heavy ego-centric video clip features by combining the semantics from a sparse set of video frames with head motion from lightweight IMU readings. We further devise a novel IMU-based self-supervised pretraining strategy. Our method leads to significant improvements in efficiency, requiring 200× fewer GFLOPs than equivalent video models. We demonstrate its effectiveness on the Ego4D and EPIC- Kitchens datasets, where our method outperforms state-of-the-art efficient video understanding methods.</details>

### Eliminating Catastrophic Overfitting Via Abnormal Adversarial Examples Regularization

**Authors:** Runqi Lin, Chaojian Yu, Tongliang Liu

**<details><summary>Abstract**</summary>

Single-step adversarial training (SSAT) has demonstrated the potential to achieve both efficiency and robustness. However, SSAT suffers from catastrophic overfitting (CO), a phenomenon that leads to a severely distorted classifier, making it vulnerable to multi-step adversarial attacks. In this work, we observe that some adversarial examples generated on the SSAT-trained network exhibit anomalous behaviour, that is, although these training samples are generated by the inner maximization process, their associated loss decreases instead, which we named abnormal adversarial examples (AAEs). Upon further analysis, we discover a close relationship between AAEs and classifier distortion, as both the number and outputs of AAEs undergo a significant variation with the onset of CO. Given this observation, we re-examine the SSAT process and uncover that before the occurrence of CO, the classifier already displayed a slight distortion, indicated by the presence of few AAEs. Furthermore, the classifier directly optimizing these AAEs will accelerate its distortion, and correspondingly, the variation of AAEs will sharply increase as a result. In such a vicious circle, the classifier rapidly becomes highly distorted and manifests as CO within a few iterations. These observations motivate us to eliminate CO by hindering the generation of AAEs. Specifically, we design a novel method, termed Abnormal Adversarial Examples Regularization (AAER), which explicitly regularizes the variation of AAEs to hinder the classifier from becoming distorted. Extensive experiments demonstrate that our method can effectively eliminate CO and further boost adversarial robustness with negligible additional computational overhead. Our implementation can be found at https://github.com/tmllab/2023_NeurIPS_AAER.</details>

### [Spotlight] EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought

**Authors:** Yao Mu, Qinglong Zhang, Mengkang Hu, Wenhai Wang, Mingyu Ding, Jun Jin, Bin Wang, Jifeng Dai, Yu Qiao, Ping Luo

**<details><summary>Abstract**</summary>

Embodied AI is a crucial frontier in robotics, capable of planning and executing action sequences for robots to accomplish long-horizon tasks in physical environments.In this work, we introduce EmbodiedGPT, an end-to-end multi-modal foundation model for embodied AI, empowering embodied agents with multi-modal understanding and execution capabilities. To achieve this, we have made the following efforts: (i) We craft a large-scale embodied planning dataset, termed EgoCOT. The dataset consists of carefully selected videos from the Ego4D dataset, along with corresponding high-quality language instructions. Specifically, we generate a sequence of sub-goals with the "Chain of Thoughts" mode for effective embodied planning.(ii) We introduce an efficient training approach to EmbodiedGPT for high-quality plan generation, by adapting a 7B large language model (LLM) to the EgoCOT dataset via prefix tuning. (iii) We introduce a paradigm for extracting task-related features from LLM-generated planning queries to form a closed loop between high-level planning and low-level control.Extensive experiments show the effectiveness of EmbodiedGPT on embodied tasks, including embodied planning, embodied control, visual captioning, and visual question answering.Notably, EmbodiedGPT significantly enhances the success rate of the embodied control task by extracting more effective features. It has achieved a remarkable 1.6 times increase in success rate on the Franka Kitchen benchmark and a 1.3 times increase on the Meta-World benchmark, compared to the BLIP-2 baseline fine-tuned with the Ego4D dataset.</details>

### Emergent and Predictable Memorization in Large Language Models

**Authors:** Stella Biderman, USVSN PRASHANTH, Lintang Sutawika, Hailey Schoelkopf, Quentin Anthony, Shivanshu Purohit, Edward Raff

**<details><summary>Abstract**</summary>

Memorization, or the tendency of large language models (LLMs) to output entire sequences from their training data verbatim, is a key concern for deploying language models. In particular, it is vital to minimize a model's memorization of sensitive datapoints such as those containing personal identifiable information (PII). The prevalence of such undesirable memorization can pose issues for model trainers, and may even require discarding an otherwise functional model. We therefore seek to predict which sequences will be memorized before a large model's full train-time by extrapolating the memorization behavior of lower-compute trial runs. We measure memorization in the Pythia model suite and plot scaling laws for forecasting memorization, allowing us to provide equi-compute recommendations to maximize the reliability (recall) of such predictions. We additionally provide further novel discoveries on the distribution of memorization scores across models and data. We release all code and data necessary to reproduce the results in this paper at https://github.com/EleutherAI/pythia.</details>

### Energy Transformer

**Authors:** Benjamin Hoover, Yuchen Liang, Bao Pham, Rameswar Panda, Hendrik Strobelt, Duen Horng Chau, Mohammed Zaki, Dmitry Krotov

**<details><summary>Abstract**</summary>

Our work combines aspects of three promising paradigms in machine learning, namely, attention mechanism, energy-based models, and associative memory. Attention is the power-house driving modern deep learning successes, but it lacks clear theoretical foundations. Energy-based models allow a principled approach to discriminative and generative tasks, but the design of the energy functional is not straightforward. At the same time, Dense Associative Memory models or Modern Hopfield Networks have a well-established theoretical foundation, and allow an intuitive design of the energy function. We propose a novel architecture, called the Energy Transformer (or ET for short), that uses a sequence of attention layers that are purposely designed to minimize a specifically engineered energy function, which is responsible for representing the relationships between the tokens. In this work, we introduce the theoretical foundations of ET, explore its empirical capabilities using the image completion task, and obtain strong quantitative results on the graph anomaly detection and graph classification tasks.</details>

### Enhancing Minority Classes by Mixing: An Adaptative Optimal Transport Approach for Long-tailed Classification

**Authors:** Jintong Gao, He Zhao, Zhuo Li, Dandan Guo

**<details><summary>Abstract**</summary>

Real-world data usually confronts severe class-imbalance problems, where several majority classes have a significantly larger presence in the training set than minority classes. One effective solution is using mixup-based methods to generate synthetic samples to enhance the presence of minority classes. Previous approaches mix the background images from the majority classes and foreground images from theminority classes in a random manner, which ignores the sample-level semantic similarity, possibly resulting in less reasonable or less useful images. In this work, we propose an adaptive image-mixing method based on optimal transport (OT) to incorporate both class-level and sample-level information, which is able to generate semantically reasonable and meaningful mixed images for minority classes. Due toits flexibility, our method can be combined with existing long-tailed classification methods to enhance their performance and it can also serve as a general data augmentation method for balanced datasets. Extensive experiments indicate that our method achieves effective performance for long-tailed classification tasks. The code is available at https://github.com/JintongGao/Enhancing-Minority-Classes-by-Mixing.</details>

### Ensemble-based Deep Reinforcement Learning for Vehicle Routing Problems under Distribution Shift

**Authors:** YUAN JIANG, Zhiguang Cao, Yaoxin Wu, Wen Song, Jie Zhang

**<details><summary>Abstract**</summary>

While performing favourably on the independent and identically distributed (i.i.d.) instances, most of the existing neural methods for vehicle routing problems (VRPs) struggle to generalize in the presence of a distribution shift. To tackle this issue, we propose an ensemble-based deep reinforcement learning method for VRPs, which learns a group of diverse sub-policies to cope with various instance distributions. In particular, to prevent convergence of the parameters to the same one, we enforce diversity across sub-policies by leveraging Bootstrap with random initialization. Moreover, we also explicitly pursue inequality between sub-policies by exploiting regularization terms during training to further enhance diversity. Experimental results show that our method is able to outperform the state-of-the-art neural baselines on randomly generated instances of various distributions, and also generalizes favourably on the benchmark instances from TSPLib and CVRPLib, which confirmed the effectiveness of the whole method and the respective designs.</details>

### [Oral] Entropic Neural Optimal Transport via Diffusion Processes

**Authors:** Nikita Gushchin, Alexander Kolesov, Alexander Korotin, Dmitry Vetrov, Evgeny Burnaev

**Oral Presentation:** We, Dec 13, 08:15 -- Oral 3C

**<details><summary>Abstract**</summary>

We propose a novel neural algorithm for the fundamental problem of computing the entropic optimal transport (EOT) plan between probability distributions which are accessible by samples. Our algorithm is based on the saddle point reformulation of the dynamic version of EOT which is known as the Schrödinger Bridge problem. In contrast to the prior methods for large-scale EOT, our algorithm is end-to-end and consists of a single learning step, has fast inference procedure, and allows handling small values of the entropy regularization coefficient which is of particular importance in some applied problems. Empirically, we show the performance of the method on several large-scale EOT tasks. The code for the ENOT solver can be found at https://github.com/ngushchin/EntropicNeuralOptimalTransport</details>

### Entropy-dissipation Informed Neural Network for McKean-Vlasov Type PDEs

**Authors:** Zebang Shen, Zhenfu Wang

**<details><summary>Abstract**</summary>

The McKean-Vlasov equation (MVE) describes the collective behavior of particles subject to drift, diffusion, and mean-field interaction. In physical systems, the interaction term can be singular, i.e. it diverges when two particles collide. Notable examples of such interactions include the Coulomb interaction, fundamental in plasma physics, and the Biot-Savart interaction, present in the vorticity formulation of the 2D Navier-Stokes equation (NSE) in fluid dynamics. Solving MVEs that involve singular interaction kernels presents a significant challenge, especially when aiming to provide rigorous theoretical guarantees. In this work, we propose a novel approach based on the concept of entropy dissipation in the underlying system. We derive a potential function that effectively controls the KL divergence between a hypothesis solution and the ground truth. Building upon this theoretical foundation, we introduce the Entropy-dissipation Informed Neural Network (EINN) framework for solving MVEs. In EINN, we utilize neural networks (NN) to approximate the underlying velocity field and minimize the proposed potential function. By leveraging the expressive power of NNs, our approach offers a promising avenue for tackling the complexities associated with singular interactions. To assess the empirical performance of our method, we compare EINN with SOTA NN-based MVE solvers. The results demonstrate the effectiveness of our approach in solving MVEs across various example problems.</details>

### [Spotlight] Episodic Multi-Task Learning with Heterogeneous Neural Processes

**Authors:** Jiayi Shen, Xiantong Zhen, Qi Wang, Marcel Worring

**<details><summary>Abstract**</summary>

This paper focuses on the data-insufficiency problem in multi-task learning within an episodic training setup. Specifically, we explore the potential of heterogeneous information across tasks and meta-knowledge among episodes to effectively tackle each task with limited data. Existing meta-learning methods often fail to take advantage of crucial heterogeneous information in a single episode, while multi-task learning models neglect reusing experience from earlier episodes. To address the problem of insufficient data, we develop Heterogeneous Neural Processes (HNPs) for the episodic multi-task setup. Within the framework of hierarchical Bayes, HNPs effectively capitalize on prior experiences as meta-knowledge and capture task-relatedness among heterogeneous tasks, mitigating data-insufficiency. Meanwhile, transformer-structured inference modules are designed to enable efficient inferences toward meta-knowledge and task-relatedness. In this way, HNPs can learn more powerful functional priors for adapting to novel heterogeneous tasks in each meta-test episode. Experimental results show the superior performance of the proposed HNPs over typical baselines, and ablation studies verify the effectiveness of the designed inference modules.</details>

### [Spotlight] Equivariant Neural Operator Learning with Graphon Convolution

**Authors:** Chaoran Cheng, Jian Peng

**<details><summary>Abstract**</summary>

We propose a general architecture that combines the coefficient learning scheme with a residual operator layer for learning mappings between continuous functions in the 3D Euclidean space. Our proposed model is guaranteed to achieve SE(3)-equivariance by design. From the graph spectrum view, our method can be interpreted as convolution on graphons (dense graphs with infinitely many nodes), which we term InfGCN. By leveraging both the continuous graphon structure and the discrete graph structure of the input data, our model can effectively capture the geometric information while preserving equivariance. Through extensive experiments on large-scale electron density datasets, we observed that our model significantly outperformed the current state-of-the-art architectures. Multiple ablation studies were also carried out to demonstrate the effectiveness of the proposed architecture.</details>

### Estimating Koopman operators with sketching to provably learn large scale dynamical systems

**Authors:** Giacomo Meanti, Antoine Chatalic, Vladimir Kostic, Pietro Novelli, Massimiliano Pontil, Lorenzo Rosasco

**<details><summary>Abstract**</summary>

The theory of Koopman operators allows to deploy non-parametric machine learning algorithms to predict and analyze complex dynamical systems.Estimators such as principal component regression (PCR) or reduced rank regression (RRR) in kernel spaces can be shown to provably learn Koopman operators from finite empirical observations of the system's time evolution. Scaling these approaches to very long trajectories is a challenge and requires introducing suitable approximations to make computations feasible. In this paper, we boost the efficiency of different kernel-based Koopman operator estimators using random projections (sketching).We derive, implement and test the new ``sketched'' estimators with extensive experiments on synthetic and large-scale molecular dynamics datasets. Further, we establish non asymptotic error bounds giving a sharp characterization of the trade-offs between statistical learning rates and computational efficiency.Our empirical and theoretical analysis shows that the proposed estimators provide a sound and efficient way to learn large scale dynamical systems.In particular our experiments indicate that the proposed estimators retain the same accuracy of PCR or RRR, while being much faster.</details>

### [Spotlight] Evaluating and Inducing Personality in Pre-trained Language Models

**Authors:** Guangyuan Jiang, Manjie Xu, Song-Chun Zhu, Wenjuan Han, Chi Zhang, Yixin Zhu

**<details><summary>Abstract**</summary>

Standardized and quantified evaluation of machine behaviors is a crux of understanding LLMs. In this study, we draw inspiration from psychometric studies by leveraging human personality theory as a tool for studying machine behaviors. Originating as a philosophical quest for human behaviors, the study of personality delves into how individuals differ in thinking, feeling, and behaving. Toward building and understanding human-like social machines, we are motivated to ask: Can we assess machine behaviors by leveraging human psychometric tests in a **principled** and **quantitative** manner? If so, can we induce a specific personality in LLMs? To answer these questions, we introduce the Machine Personality Inventory (MPI) tool for studying machine behaviors; MPI follows standardizedpersonality tests, built upon the Big Five Personality Factors (Big Five) theory and personality assessment inventories. By systematically evaluating LLMs with MPI, we provide the first piece of evidence demonstrating the efficacy of MPI in studying LLMs behaviors. We further devise a Personality Prompting (P$^2$) method to induce LLMs with specific personalities in a **controllable** way, capable of producing diverse and verifiable behaviors. We hope this work sheds light on future studies by adopting personality as the essential indicator for various downstream tasks, and could further motivate research into equally intriguing human-like machine behaviors.</details>

### [Spotlight] Evaluating the Moral Beliefs Encoded in LLMs

**Authors:** Nino Scherrer, Claudia Shi, Amir Feder, David Blei

**<details><summary>Abstract**</summary>

This paper presents a case study on the design, administration, post-processing,  and evaluation of surveys on large language models (LLMs). It comprises two components:(1) A statistical method for eliciting beliefs encoded in LLMs. We introduce statistical measures and evaluation metrics that quantify the probability of an LLM "making a choice", the associated uncertainty, and the consistency of that choice.(2) We apply this method to study what moral beliefs are encoded in different LLMs, especially in ambiguous cases where the right choice is not obvious.We design a large-scale survey comprising 680 high-ambiguity moral scenarios (e.g., "Should I tell a white lie?") and 687 low-ambiguity moral scenarios (e.g., "Should I stop for a pedestrian on the road?"). Each scenario includes a description, two possible actions, and auxiliary labels indicating violated rules (e.g., "do not kill"). We administer the survey to 28 open- and closed-source LLMs.We find that (a) in unambiguous scenarios, most models ``choose" actions that align with commonsense. In ambiguous cases, most models express uncertainty.(b) Some models are uncertain about choosing the commonsense action because their responses are sensitive to the question-wording.(c) Some models reflect clear preferences in ambiguous scenarios. Specifically, closed-source models tend to agree with each other.</details>

### EvoFed: Leveraging Evolutionary Strategies for Communication-Efficient Federated Learning

**Authors:** Mohammad Mahdi Rahimi, Hasnain Irshad Bhatti, Younghyun Park, Humaira Kousar, Do-Yeon Kim, Jaekyun Moon

**<details><summary>Abstract**</summary>

Federated Learning (FL) is a decentralized machine learning paradigm that enables collaborative model training across dispersed nodes without having to force individual nodes to share data.However, its broad adoption is hindered by the high communication costs of transmitting a large number of model parameters. This paper presents EvoFed, a novel approach that integrates Evolutionary Strategies (ES) with FL to address these challenges.EvoFed employs a concept of `fitness-based information sharing’, deviating significantly from the conventional model-based FL. Rather than exchanging the actual updated model parameters, each node transmits a distance-based similarity measure between the locally updated model and each member of the noise-perturbed model population. Each node, as well as the server, generates an identical population set of perturbed models in a completely synchronized fashion using the same random seeds. With properly chosen noise variance and population size, perturbed models can be combined to closely reflect the actual model updated using the local dataset, allowing the transmitted similarity measures (or fitness values) to carry nearly the complete information about the model parameters.As the population size is typically much smaller than the number of model parameters, the savings in communication load is large. The server aggregates these fitness values and is able to update the global model. This global fitness vector is then disseminated back to the nodes, each of which applies the same update to be synchronized to the global model. Our analysis shows that EvoFed converges, and our experimental results validate that at the cost of increased local processing loads, EvoFed achieves performance comparable to FedAvg while reducing overall communication requirements drastically in various practical settings.</details>

### Evolutionary Neural Architecture Search for Transformer in Knowledge Tracing

**Authors:** Shangshang Yang, Xiaoshan Yu, Ye Tian, Xueming Yan, Haiping Ma, Xingyi Zhang

**<details><summary>Abstract**</summary>

Knowledge tracing (KT) aims to trace students' knowledge states by predicting whether students answer correctly on exercises. Despite the excellent performance of existing Transformer-based KT approaches, they are criticized for the manually selected input features for fusion and the defect of single global context modelling to directly capture students' forgetting behavior in KT, when the related records are distant from the current record in terms of time. To address the issues, this paper first considers adding convolution operations to the Transformer to enhance its local context modelling ability used for students' forgetting behavior, then proposes an evolutionary neural architecture search approach to automate the input feature selection and automatically determine where to apply which operation for achieving the balancing of the local/global context modelling. In the search space, the original global path containing the attention module in Transformer is replaced with the sum of a global path and a local path that could contain different convolutions, and the selection of input features is also considered. To search the best architecture, we employ an effective evolutionary algorithm to explore the search space and also suggest a search space reduction strategy to accelerate the convergence of the algorithm. Experimental results on the two largest and most challenging education datasets demonstrate the effectiveness of the architecture found by the proposed approach.</details>

### Experimental Designs for Heteroskedastic Variance

**Authors:** Justin Weltz, Tanner Fiez, Alexander Volfovsky, Eric Laber, Blake Mason, houssam nassif, Lalit Jain

**<details><summary>Abstract**</summary>

Most linear experimental design problems assume homogeneous variance, while the presence of heteroskedastic noise is present in many realistic settings. Let a learner have access to a finite set of measurement vectors $\mathcal{X}\subset \mathbb{R}^d$ that can be probed to receive noisy linear responses of the form $y=x^{\top}\theta^{\ast}+\eta$. Here $\theta^{\ast}\in \mathbb{R}^d$ is an unknown parameter vector, and $\eta$ is independent mean-zero $\sigma_x^2$-sub-Gaussian noise defined by a flexible heteroskedastic variance model, $\sigma_x^2 = x^{\top}\Sigma^{\ast}x$. Assuming that $\Sigma^{\ast}\in \mathbb{R}^{d\times d}$ is an unknown matrix, we propose, analyze and empirically evaluate a novel design for uniformly bounding estimation error of the variance parameters, $\sigma_x^2$. We demonstrate this method on two adaptive experimental design problems under heteroskedastic noise, fixed confidence transductive best-arm identification and level-set identification and prove the first instance-dependent lower bounds in these settings.Lastly, we construct near-optimal algorithms and demonstrate the large improvements in sample complexity gained from accounting for heteroskedastic variance in these designs empirically.</details>

### Explainable and Efficient Randomized Voting Rules

**Authors:** Soroush Ebadian, Aris Filos-Ratsikas, Mohamad Latifian, Nisarg Shah

**<details><summary>Abstract**</summary>

With a rapid growth in the deployment of AI tools for making critical decisions (or aiding humans in doing so), there is a growing demand to be able to explain to the stakeholders how these tools arrive at a decision. Consequently, voting is frequently used to make such decisions due to its inherent explainability. Recent work suggests that using randomized (as opposed to deterministic) voting rules can lead to significant efficiency gains measured via the distortion framework. However, rules that use intricate randomization can often become too complex to explain to the stakeholders; losing explainability can eliminate the key advantage of voting over black-box AI tools, which may outweigh the efficiency gains.We study the efficiency gains which can be unlocked by using voting rules that add a simple randomization step to a deterministic rule, thereby retaining explainability. We focus on two such families of rules, randomized positional scoring rules and random committee member rules, and show, theoretically and empirically, that they indeed achieve explainability and efficiency simultaneously to some extent.</details>

### Exploiting Contextual Objects and Relations for 3D Visual Grounding

**Authors:** Li Yang, chunfeng yuan, Ziqi Zhang, Zhongang Qi, Yan Xu, Wei Liu, Ying Shan, Bing Li, Weiping Yang, Peng Li, Yan Wang, Weiming Hu

**<details><summary>Abstract**</summary>

3D visual grounding, the task of identifying visual objects in 3D scenes based on natural language inputs, plays a critical role in enabling machines to understand and engage with the real-world environment. However, this task is challenging due to the necessity to capture 3D contextual information to distinguish target objects from complex 3D scenes. The absence of annotations for contextual objects and relations further exacerbates the difficulties. In this paper, we propose a novel model, CORE-3DVG, to address these challenges by explicitly learning about contextual objects and relations. Our method accomplishes 3D visual grounding via three sequential modular networks, including a text-guided object detection network, a relation matching network, and a target identification network. During training, we introduce a pseudo-label self-generation strategy and a weakly-supervised method to facilitate the learning of contextual objects and relations, respectively. The proposed techniques allow the networks to focus more effectively on referred objects within 3D scenes by understanding their context better. We validate our model on the challenging Nr3D, Sr3D, and ScanRefer datasets and demonstrate state-of-the-art performance. Our code will be public at https://github.com/yangli18/CORE-3DVG.</details>

### Exploring Question Decomposition for Zero-Shot VQA

**Authors:** Zaid Khan, Vijay Kumar B G, Samuel Schulter, Manmohan Chandraker, Yun Fu

**<details><summary>Abstract**</summary>

Visual question answering (VQA) has traditionally been treated as a single-step task where each question receives the same amount of effort, unlike natural human question-answering strategies. We explore a question decomposition strategy for VQA to overcome this limitation. We probe the ability of recently developed large vision-language models to use human-written decompositions and produce their own decompositions of visual questions, finding they are capable of learning both tasks from demonstrations alone.However, we show that naive application of model-written decompositions can hurt performance.We introduce a model-driven selective decomposition approach for second-guessing predictions and correcting errors, and validate its effectiveness on eight VQA tasks across three domains, showing consistent improvements in accuracy, including improvements of >20% on medical VQA datasets and boosting the zero-shot performance of BLIP-2 above chance on a VQA reformulation of the challenging Winoground task. Project Site: https://zaidkhan.me/decomposition-0shot-vqa/</details>

### Extracting Reward Functions from Diffusion Models

**Authors:** Felipe Nuti, Tim Franzmeyer, João Henriques

**<details><summary>Abstract**</summary>

Diffusion models have achieved remarkable results in image generation, and have similarly been used to learn high-performing policies in sequential decision-making tasks. Decision-making diffusion models can be trained on lower-quality data, and then be steered with a reward function to generate near-optimal trajectories.We consider the problem of extracting a reward function by comparing a decision-making diffusion model that models low-reward behavior and one that models high-reward behavior; a setting related to inverse reinforcement learning. We first define the notion of a \emph{relative reward function of two diffusion models} and show conditions under which it exists and is unique. We then devise a practical learning algorithm for extracting it by aligning the gradients of a reward function -- parametrized by a neural network -- to the difference in outputs of both diffusion models.Our method finds correct reward functions in navigation environments, and we demonstrate that steering the base model with the learned reward functions results in significantly increased performance in standard locomotion benchmarks.Finally, we demonstrate that our approach generalizes beyond sequential decision-making by learning a reward-like function from two large-scale image generation diffusion models. The extracted reward function successfully assigns lower rewards to harmful images.</details>

### FABind: Fast and Accurate Protein-Ligand Binding

**Authors:** Qizhi Pei, Kaiyuan Gao, Lijun Wu, Jinhua Zhu, Yingce Xia, Shufang Xie, Tao Qin, Kun He, Tie-Yan Liu, Rui Yan

**<details><summary>Abstract**</summary>

Modeling the interaction between proteins and ligands and accurately predicting their binding structures is a critical yet challenging task in drug discovery. Recent advancements in deep learning have shown promise in addressing this challenge, with sampling-based and regression-based methods emerging as two prominent approaches. However, these methods have notable limitations. Sampling-based methods often suffer from low efficiency due to the need for generating multiple candidate structures for selection. On the other hand, regression-based methods offer fast predictions but may experience decreased accuracy. Additionally, the variation in protein sizes often requires external modules for selecting suitable binding pockets, further impacting efficiency. In this work, we propose FABind, an end-to-end model that combines pocket prediction and docking to achieve accurate and fast protein-ligand binding. FABind incorporates a unique ligand-informed pocket prediction module, which is also leveraged for docking pose estimation. The model further enhances the docking process by incrementally integrating the predicted pocket to optimize protein-ligand binding, reducing discrepancies between training and inference. Through extensive experiments on benchmark datasets, our proposed FABind demonstrates strong advantages in terms of effectiveness and efficiency compared to existing methods. Our code is available at https://github.com/QizhiPei/FABind.</details>

### FaceDNeRF: Semantics-Driven Face Reconstruction, Prompt Editing and Relighting with Diffusion Models

**Authors:** Hao ZHANG, Tianyuan DAI, Yanbo Xu, Yu-Wing Tai, Chi-Keung Tang

**<details><summary>Abstract**</summary>

The ability to create high-quality 3D faces from a single image has become increasingly important with wide applications in video conferencing, AR/VR, and advanced video editing in movie industries. In this paper, we propose Face Diffusion NeRF (FaceDNeRF), a new generative method to reconstruct high-quality Face NeRFs from single images, complete with semantic editing and relighting capabilities. FaceDNeRF utilizes high-resolution 3D GAN inversion and expertly trained 2D latent-diffusion model, allowing users to manipulate and construct Face NeRFs in zero-shot learning without the need for explicit 3D data. With carefully designed illumination and identity preserving loss, as well as multi-modal pre-training, FaceDNeRF offers users unparalleled control over the editing process enabling them to create and edit face NeRFs using just single-view images, text prompts, and explicit target lighting. The advanced features of FaceDNeRF have been designed to produce more impressive results than existing 2D editing approaches that rely on 2D segmentation maps for editable attributes. Experiments show that our FaceDNeRF achieves exceptionally realistic results and unprecedented flexibility in editing compared with state-of-the-art 3D face reconstruction and editing methods. Our code will be available at https://github.com/BillyXYB/FaceDNeRF.</details>

### Facilitating Graph Neural Networks with Random Walk on Simplicial Complexes

**Authors:** Cai Zhou, Xiyuan Wang, Muhan Zhang

**<details><summary>Abstract**</summary>

Node-level random walk has been widely used to improve Graph Neural Networks. However, there is limited attention to random walk on edge and, more generally, on $k$-simplices. This paper systematically analyzes how random walk on different orders of simplicial complexes (SC) facilitates GNNs in their theoretical expressivity. First, on $0$-simplices or node level, we establish a connection between existing positional encoding (PE) and structure encoding (SE) methods through the bridge of random walk. Second, on $1$-simplices or edge level, we bridge edge-level random walk and Hodge $1$-Laplacians and design corresponding edge PE respectively. In spatial domain, we directly make use of edge level random walk to construct EdgeRWSE. Based on spectral analysis of Hodge $1$-Laplcians, we propose Hodge1Lap, a permutation equivariant and expressive edge-level positional encoding. Third, we generalize our theory to random walk on higher-order simplices and propose the general principle to design PE on simplices based on random walk and Hodge Laplacians. Inter-level random walk is also introduced to unify a wide range of simplicial networks. Extensive experiments verify the effectiveness of our random walk-based methods.</details>

### Fair Adaptive Experiments

**Authors:** Waverly Wei, Xinwei Ma, Jingshen Wang

**<details><summary>Abstract**</summary>

Randomized experiments have been the gold standard for assessing the effectiveness of a treatment, policy, or intervention, spanning various fields, including social sciences, biomedical studies, and e-commerce. The classical complete randomization approach assigns treatments based on a pre-specified probability and may lead to inefficient use of data. Adaptive experiments improve upon complete randomization by sequentially learning and updating treatment assignment probabilities using accrued evidence during the experiment. Hence, they can help achieve efficient data use and higher estimation efficiency. However, their application can also raise fairness and equity concerns, as assignment probabilities may vary drastically across groups of participants. Furthermore, when treatment is expected to be extremely beneficial to certain groups of participants, it is more appropriate to expose many of these participants to favorable treatment. In response to these challenges, we propose a fair adaptive experiment strategy that simultaneously enhances data use efficiency, achieves an ``envy-free'' treatment assignment guarantee, and improves the overall welfare of participants. An important feature of our proposed strategy is that we do not impose parametric modeling assumptions on the outcome variables, making it more versatile and applicable to a wider array of applications. Through our theoretical investigation, we characterize the convergence rate of the estimated treatment effects and the associated standard deviations at the group level and further prove that our adaptive treatment assignment algorithm, despite not having a closed-form expression, approaches the optimal allocation rule asymptotically. Our proof strategy takes into account the fact that the allocation decisions in our design depend on sequentially accumulated data, which poses a significant challenge in characterizing the properties and conducting statistical inference of our method. We further provide simulation evidence and two synthetic data studies to showcase the performance of our fair adaptive experiment strategy.</details>

### Fairly Recommending with Social Attributes: A Flexible and Controllable Optimization Approach

**Authors:** Jinqiu Jin, Haoxuan Li, Fuli Feng, Sihao Ding, Peng Wu, Xiangnan He

**<details><summary>Abstract**</summary>

Item-side group fairness (IGF) requires a recommendation model to treat different item groups similarly, and has a crucial impact on information diffusion, consumption activity, and market equilibrium. Previous IGF notions only focus on the direct utility of the item exposures, i.e., the exposure numbers across different item groups. Nevertheless, the item exposures also facilitate utility gained from the neighboring users via social influence, called social utility, such as information sharing on the social media. To fill this gap, this paper introduces two social attribute-aware IGF metrics, which require similar user social attributes on the exposed items across the different item groups. In light of the trade-off between the direct utility and social utility, we formulate a new multi-objective optimization problem for training recommender models with flexible trade-off while ensuring controllable accuracy. To solve this problem, we develop a gradient-based optimization algorithm and theoretically show that the proposed algorithm can find Pareto optimal solutions with varying trade-off and guaranteed accuracy. Extensive experiments on two real-world datasets validate the effectiveness of our approach.</details>

### Fairness Continual Learning Approach to Semantic Scene Understanding in Open-World Environments

**Authors:** Thanh-Dat Truong, Hoang-Quan Nguyen, Bhiksha Raj, Khoa Luu

**<details><summary>Abstract**</summary>

Continual semantic segmentation aims to learn new classes while maintaining the information from the previous classes. Although prior studies have shown impressive progress in recent years, the fairness concern in the continual semantic segmentation needs to be better addressed. Meanwhile, fairness is one of the most vital factors in deploying the deep learning model, especially in human-related or safety applications. In this paper,  we present a novel Fairness Continual Learning approach to the semantic segmentation problem.In particular, under the fairness objective, a new fairness continual learning framework is proposed based on class distributions.Then, a novel Prototypical Contrastive Clustering loss is proposed to address the significant challenges in continual learning, i.e., catastrophic forgetting and background shift. Our proposed loss has also been proven as a novel, generalized learning paradigm of knowledge distillation commonly used in continual learning. Moreover, the proposed Conditional Structural Consistency loss further regularized the structural constraint of the predicted segmentation. Our proposed approach has achieved State-of-the-Art performance on three standard scene understanding benchmarks, i.e., ADE20K, Cityscapes, and Pascal VOC, and promoted the fairness of the segmentation model.</details>

### Fairness-guided Few-shot Prompting for Large Language Models

**Authors:** Huan Ma, Changqing Zhang, Yatao Bian, Lemao Liu, Zhirui Zhang, Peilin Zhao, Shu Zhang, Huazhu Fu, Qinghua Hu, Bingzhe Wu

**<details><summary>Abstract**</summary>

Large language models have demonstrated surprising ability to perform in-context learning, i.e., these models can be directly applied to solve numerous downstream tasks by conditioning on a prompt constructed by a few input-output examples. However, prior research has shown that in-context learning can suffer from high instability due to variations in training examples, example order, and prompt formats. Therefore, the construction of an appropriate prompt is essential for improving the performance of in-context learning. In this paper,  we revisit this problem from the view of predictive bias. Specifically, we introduce a metric to evaluate the predictive bias of a fixed prompt against labels or a given attributes. Then we empirically show that prompts with higher bias always lead to unsatisfactory predictive quality. Based on this observation, we propose a novel search strategy based on the greedy search to identify the near-optimal prompt for improving the performance of in-context learning. We perform comprehensive experiments with state-of-the-art mainstream models such as GPT-3 on various downstream tasks. Our results indicate that our method can enhance the model's in-context learning performance in an effective and interpretable manner.</details>

### Fast Attention Requires Bounded Entries

**Authors:** Josh Alman, Zhao Song

**<details><summary>Abstract**</summary>

In modern machine learning, inner product attention computation is a fundamental task for training large language models such as Transformer, GPT-1, BERT, GPT-2, GPT-3 and ChatGPT. Formally, in this problem, one is given as input three matrices $Q, K, V \in [-B,B]^{n \times d}$, and the goal is to construct the matrix $\mathrm{Att}(Q,K,V) := \mathrm{diag}(A {\bf 1}_n)^{-1} A V \in \mathbb{R}^{n \times d}$, where $A = \exp(QK^\top/d)$ is the `attention matrix', and $\exp$ is applied entry-wise. Straightforward methods for this problem explicitly compute the $n \times n$ attention matrix $A$, and hence require time $\Omega(n^2)$ even when $d = n^{o(1)}$ is small. In this paper, we investigate whether faster algorithms are possible by \emph{implicitly} making use of the matrix $A$. We present two results, showing that there is a sharp transition at $B = \Theta(\sqrt{\log n})$.$\bullet$ If $d = O(\log n)$ and $B = o(\sqrt{\log n})$, there is an $n^{1+o(1)}$ time algorithm to approximate $\mathrm{Att}(Q,K,V)$ up to $1/\mathrm{poly}(n)$ additive error.$\bullet$ If $d = O(\log n)$ and $B = \Theta (\sqrt{\log n})$, assuming the Strong Exponential Time Hypothesis from fine-grained complexity theory, it is impossible to approximate $\mathrm{Att}(Q,K,V)$ up to $1/\mathrm{poly}(n)$ additive error in truly subquadratic time $n^{2 - \Omega(1)}$.This gives a theoretical explanation for the phenomenon observed in practice that attention computation is much more efficient when the input matrices have smaller entries.</details>

### Fast Optimal Locally Private Mean Estimation via Random Projections

**Authors:** Hilal Asi, Vitaly Feldman, Jelani Nelson, Huy Nguyen, Kunal Talwar

**<details><summary>Abstract**</summary>

We study the problem of locally private mean estimation of high-dimensional vectors in the Euclidean ball. Existing algorithms for this problem either incur sub-optimal error or have high communication and/or run-time complexity. We propose a new algorithmic framework, namely ProjUnit, for private mean estimation that yields algorithms that are computationally efficient, have low communication complexity, and incur optimal error up to a $1+o(1)$-factor. Our framework is deceptively simple: each randomizer projects its input to a random low-dimensional subspace and then runs an optimal algorithm such a PrivUnitG in the lower dimensional space. We analyze the error of the algorithm in terms of properties of the random projection ensemble, and study two instantiations. We conduct several experiments for private mean estimation and private federated learning which demonstrate that our algorithms obtain nearly the same utility as optimal algorithms while having significantly lower communication and computational cost.</details>

### Fast Rank-1 Lattice Targeted Sampling for Black-box Optimization

**Authors:** Yueming LYU

**<details><summary>Abstract**</summary>

Black-box optimization has gained great attention for its success in recent applications.    However, scaling up to high-dimensional problems with good query efficiency remains challenging. This paper proposes a novel Rank-1 Lattice Targeted Sampling (RLTS) technique to address this issue. Our RLTS benefits from random rank-1 lattice Quasi-Monte Carlo, which enables us to perform fast local exact Gaussian processes (GP)  training and inference with $O(n \log n)$ complexity w.r.t. $n$ batch samples. Furthermore, we developed a fast coordinate searching method with $O(n \log n)$ time complexity for fast targeted sampling. The fast computation enables us to plug our RLTS into the sampling phase of stochastic optimization methods. This improves the query efficiency while  scaling up to higher dimensional problems than Bayesian optimization. Moreover,  to construct rank-1 lattices efficiently, we proposed a closed-form construction.  Extensive experiments on challenging benchmark test functions and black-box prompt fine-tuning for large language models demonstrate the query efficiency of our RLTS technique.</details>

### Fast and Simple Spectral Clustering in Theory and Practice

**Authors:** Peter Macgregor

**<details><summary>Abstract**</summary>

Spectral clustering is a popular and effective algorithm designed to find $k$ clusters in a graph $G$.In the classical spectral clustering algorithm, the vertices of $G$ are embedded into $\mathbb{R}^k$ using $k$ eigenvectors of the graph Laplacian matrix.However, computing this embedding is computationally expensive and dominates the running time of the algorithm.In this paper, we present a simple spectral clustering algorithm based on a vertex embedding with $O(\log(k))$ vectors computed by the power method.The vertex embedding is computed in nearly-linear time with respect to the size of the graph, andthe algorithm provably recovers the ground truth clusters under natural assumptions on the input graph.We evaluate the new algorithm on several synthetic and real-world datasets, finding that it is significantly faster than alternative clustering algorithms,while producing results with approximately the same clustering accuracy.</details>

### Faster Query Times for Fully Dynamic $k$-Center Clustering with Outliers

**Authors:** Leyla Biabani, Annika Hennes, Morteza Monemizadeh, Melanie Schmidt

**<details><summary>Abstract**</summary>

Given a point set $P\subseteq M$ from a metric space $(M,d)$ and numbers $k, z \in N$, the *metric $k$-center problem with $z$ outliers* is to find a set $C^\ast\subseteq P$ of $k$ points such that the maximum distance of all but at most $z$ outlier points of $P$ to their nearest center in ${C}^\ast$ is minimized. We consider this problem in the fully dynamic model, i.e., under insertions and deletions of points, for the case that the metric space has a bounded doubling dimension $dim$. We utilize a hierarchical data structure to maintain the points and their neighborhoods, which enables us to efficiently find the clusters. In particular, our data structure can be queried at any time to generate a $(3+\varepsilon)$-approximate solution for input values of $k$ and $z$ in worst-case query time $\varepsilon^{-O(dim)}k \log{n} \log\log{\Delta}$, where $\Delta$ is the ratio between the maximum and minimum distance between two points in $P$. Moreover, it allows insertion/deletion of a point in worst-case update time $\varepsilon^{-O(dim)}\log{n}\log{\Delta}$. Our result achieves a significantly faster query time with respect to $k$ and $z$ than the current state-of-the-art by Pellizzoni, Pietracaprina, and Pucci, which uses $\varepsilon^{-O(dim)}(k+z)^2\log{\Delta}$ query time to obtain a $(3+\varepsilon)$-approximation.</details>

### Faster Relative Entropy Coding with Greedy Rejection Coding

**Authors:** Gergely Flamich, Stratis Markou, José Miguel Hernández-Lobato

**<details><summary>Abstract**</summary>

Relative entropy coding (REC) algorithms encode a sample from a target distribution $Q$ using a proposal distribution $P$ using as few bits as possible. Unlike entropy coding, REC does not assume discrete distributions and require quantisation.As such, it can be naturally integrated into communication pipelines such as learnt compression and differentially private federated learning. Unfortunately, despite their practical benefits, REC algorithms have not seen widespread application, due to their prohibitively slow runtimes or restrictive assumptions. In this paper, we make progress towards addressing these issues. We introduce Greedy Rejection Coding (GRC), which generalises the rejection sampling-based algorithm of Harsha et al. (2007) to arbitrary probability spaces and partitioning schemes. We first show that GRC terminates almost surely and returns unbiased samples from $Q$, and then focus on two variants of GRC, namely GRCS and GRCD. We show that for continuous $Q$ and $P$ over $\mathbb{R}$ with unimodal $dQ/dP$, the expected runtime of GRCS is upper bounded by $\beta D_{KL}(Q||P) + \mathcal{O}(1)$ where $\beta \approx 4.82$, and its expected codelength is optimal. This makes GRCS the first REC algorithm with guaranteed optimal runtime for this class of distributions, up to the multiplicative constant $\beta$. This significantly improves upon the previous state-of-the-art method, A* coding (Flamich et al., 2022). Under the same assumptions, we experimentally observe and conjecture that the expected runtime and codelength of GRCD are upper bounded by $D_{KL}(Q||P) + \mathcal{O}(1)$. Finally, we evaluate GRC in a compression pipeline with variational autoencoders on MNIST, and show that a modified training objective and a codelength-compression method can further improve compression efficiency.</details>

### Faster approximate subgraph counts with privacy

**Authors:** Dung Nguyen, Mahantesh Halappanavar, Venkatesh Srinivasan, Anil Vullikanti

**<details><summary>Abstract**</summary>

One of the most common problems studied in the context of differential privacy for graph data is counting the number of non-induced embeddings of a subgraph in a given graph. These counts have very high global sensitivity. Therefore, adding noise based on powerful alternative techniques, such as smooth sensitivity and higher-order local sensitivity have been shown to give significantly better accuracy. However, all these alternatives to global sensitivity become computationally very expensive, and to date efficient polynomial time algorithms are known only for few selected subgraphs, such as triangles, $k$-triangles, and $k$-stars.In this paper, we show that good approximations to these sensitivity metrics can be still used to get private algorithms.Using this approach, we show the first quasilinear time and parallel algorithms for privately counting the number of triangles.We also give a private polynomial time algorithm for counting any constant size subgraph using less noise than the global sensitivity; we show this can be improved significantly for counting paths in special classes of graphs.</details>

### Feature Learning for Interpretable, Performant Decision Trees

**Authors:** Jack Good, Torin Kovach, Kyle Miller, Artur Dubrawski

**<details><summary>Abstract**</summary>

Decision trees are regarded for high interpretability arising from their hierarchical partitioning structure built on simple decision rules. However, in practice, this is not realized because axis-aligned partitioning of realistic data results in deep trees, and because ensemble methods are used to mitigate overfitting. Even then, model complexity and performance remain sensitive to transformation of the input, and extensive expert crafting of features from the raw data is common. We propose the first system to alternate sparse feature learning with differentiable decision tree construction to produce small, interpretable trees with good performance. We benchmark against conventional tree-based models and demonstrate several notions of interpretation of a model and its predictions.</details>

### Feature-Learning Networks Are Consistent Across Widths At Realistic Scales

**Authors:** Nikhil Vyas, Alexander Atanasov, Blake Bordelon, Depen Morwani, Sabarish Sainathan, Cengiz Pehlevan

**<details><summary>Abstract**</summary>

We study the effect of width on the dynamics of feature-learning neural networks across a variety of architectures and datasets. Early in training, wide neural networks trained on online data have not only identical loss curves but also agree in their point-wise test predictions throughout training. For simple tasks such as CIFAR-5m this holds throughout training for networks of realistic widths. We also show that structural properties of the models, including internal representations, preactivation distributions, edge of stability phenomena, and large learning rate effects are consistent across large widths. This motivates the hypothesis that phenomena seen in realistic models can be captured by infinite-width, feature-learning limits. For harder tasks (such as ImageNet and language modeling), and later training times, finite-width deviations grow systematically. Two distinct effects cause these deviations across widths. First, the network output has an initialization-dependent variance scaling inversely with width, which can be removed by ensembling networks. We observe, however, that ensembles of narrower networks perform worse than a single wide network. We call this the bias of narrower width. We conclude with a spectral perspective on the origin of this finite-width bias.</details>

### Federated Conditional Stochastic Optimization

**Authors:** Xidong Wu, Jianhui Sun, Zhengmian Hu, Junyi Li, Aidong Zhang, Heng Huang

**<details><summary>Abstract**</summary>

Conditional stochastic optimization has found applications in a wide range of machine learning tasks, such as invariant learning, AUPRC maximization, and meta-learning. As the demand for training models with large-scale distributed data grows in these applications, there is an increasing need for communication-efficient distributed optimization algorithms, such as federated learning algorithms. This paper considers the nonconvex conditional stochastic optimization in federated learning and proposes the first federated conditional stochastic optimization algorithm (FCSG) with a conditional stochastic gradient estimator and a momentum-based algorithm (\emph{i.e.}, FCSG-M). To match the lower bound complexity in the single-machine setting, we design an accelerated algorithm (Acc-FCSG-M) via the variance reduction to achieve the best sample and communication complexity. Compared with the existing optimization analysis for Meta-Learning in FL, federated conditional stochastic optimization considers the sample of tasks. Extensive experimental results on various tasks validate the efficiency of these algorithms.</details>

### Federated Multi-Objective Learning

**Authors:** Haibo Yang, Zhuqing Liu, Jia Liu, Chaosheng Dong, Michinari Momma

**<details><summary>Abstract**</summary>

In recent years, multi-objective optimization (MOO) emerges as a foundational problem underpinning many multi-agent multi-task learning applications. However, existing algorithms in MOO literature remain limited to centralized learning settings, which do not satisfy the distributed nature and data privacy needs of such multi-agent multi-task learning applications. This motivates us to propose a new federated multi-objective learning (FMOL) framework with multiple clients distributively and collaboratively solving an MOO problem while keeping their training data private. Notably, our FMOL framework allows a different set of objective functions across different clients to support a wide range of applications, which advances and generalizes the MOO formulation to the federated learning paradigm for the first time. For this FMOL framework, we propose two new federated multi-objective optimization (FMOO) algorithms called federated multi-gradient descent averaging (FMGDA) and federated stochastic multi-gradient descent averaging (FSMGDA). Both algorithms allow local updates to significantly reduce communication costs, while achieving the {\em same} convergence rates as those of their algorithmic counterparts in the single-objective federated learning. Our extensive experiments also corroborate the efficacy of our proposed FMOO algorithms.</details>

### Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering

**Authors:** Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexandru Coca, Bill Byrne

**<details><summary>Abstract**</summary>

Knowledge-based Visual Question Answering (KB-VQA) requires VQA systems to utilize knowledge from external knowledge bases to answer visually-grounded questions. Retrieval-Augmented Visual Question Answering (RA-VQA), a strong framework to tackle KB-VQA, first retrieves related documents with Dense Passage Retrieval (DPR) and then uses them to answer questions. This paper proposes Fine-grained Late-interaction Multi-modal Retrieval (FLMR) which significantly improves knowledge retrieval in RA-VQA. FLMR addresses two major limitations in RA-VQA's retriever: (1) the image representations obtained via image-to-text transforms can be incomplete and inaccurate and (2) similarity scores between queries and documents are computed with one-dimensional embeddings, which can be insensitive to finer-grained similarities.FLMR overcomes these limitations by obtaining image representations that complement those from the image-to-text transform using a vision model aligned with an existing text-based retriever through a simple alignment network. FLMR also encodes images and questions using multi-dimensional embeddings to capture finer-grained similarities between queries and documents. FLMR significantly improves the original RA-VQA retriever's PRRecall@5 by approximately 8\%. Finally, we equipped RA-VQA with two state-of-the-art large multi-modal/language models to achieve $\sim62$% VQA score in the OK-VQA dataset.</details>

### FineMoGen: Fine-Grained Spatio-Temporal Motion Generation and Editing

**Authors:** Mingyuan Zhang, Huirong Li, Zhongang Cai, Jiawei Ren, Lei Yang, Ziwei Liu

**<details><summary>Abstract**</summary>

Text-driven motion generation has achieved substantial progress with the emergence of diffusion models. However, existing methods still struggle to generate complex motion sequences that correspond to fine-grained descriptions, depicting detailed and accurate spatio-temporal actions.This lack of fine controllability limits the usage of motion generation to a larger audience. To tackle these challenges, we present FineMoGen, a diffusion-based motion generation and editing framework that can synthesize fine-grained motions, with spatial-temporal composition to the user instructions. Specifically, FineMoGen builds upon diffusion model with a novel transformer architecture dubbed Spatio-Temporal Mixture Attention SAMI. SAMI optimizes the generation of the global attention template from two perspectives: 1) explicitly modeling the constraints of spatio-temporal composition; and 2) utilizing sparsely-activated mixture-of-experts to adaptively extract fine-grained features. To facilitate a large-scale study on this new fine-grained motion generation task, we contribute the HuMMan-MoGen dataset, which consists of 2,968 videos and 102,336 fine-grained spatio-temporal descriptions. Extensive experiments validate that FineMoGen  exhibits superior motion generation quality over state-of-the-art methods. Notably, FineMoGen further enables zero-shot motion editing capabilities with the aid of modern large language models (LLM), which faithfully manipulates motion sequences with fine-grained instructions.</details>

### First Order Methods with Markovian Noise: from Acceleration to Variational Inequalities

**Authors:** Aleksandr Beznosikov, Sergey Samsonov, Marina Sheshukova, Alexander Gasnikov, Alexey Naumov, Eric Moulines

**<details><summary>Abstract**</summary>

This paper delves into stochastic optimization problems that involve Markovian noise. We present a unified approach for the theoretical analysis of first-order gradient methods for stochastic optimization and variational inequalities. Our approach covers scenarios for both non-convex and strongly convex minimization problems. To achieve an optimal (linear) dependence on the mixing time of the underlying noise sequence, we use the randomized batching scheme, which is based on the multilevel Monte Carlo method. Moreover, our technique allows us to eliminate the limiting assumptions of previous research on Markov noise, such as the need for a bounded domain and uniformly bounded stochastic gradients. Our extension to variational inequalities under Markovian noise is original. Additionally, we provide lower bounds that match the oracle complexity of our method in the case of strongly convex optimization problems.</details>

### First Order Stochastic Optimization with Oblivious Noise

**Authors:** Ilias Diakonikolas, Sushrut Karmalkar, Jong Ho Park, Christos Tzamos

**<details><summary>Abstract**</summary>

We initiate the study of stochastic optimization with oblivious noise, broadly generalizing the standard heavy-tailed noise setup.In our setting, in addition to random observation noise, the stochastic gradient may be subject to independent \emph{oblivious noise}, which may not have bounded moments and is not necessarily centered. Specifically, we assume access to a noisy oracle for the stochastic gradient of $f$ at $x$,  which returns a vector $\nabla f(\gamma, x) + \xi$, where $\gamma$ is the  bounded variance observation noise and $\xi$ is the oblivious noise that is independent of $\gamma$ and $x$. The only assumption we make on the oblivious noise $\xi$ is that $\Pr[\xi = 0] \ge \alpha$, for some $\alpha \in (0, 1)$.In this setting, it is not information-theoretically possible to recover a single solution close to the target when the fraction of inliers $\alpha$ is less than $1/2$. Our main result is an efficient {\em list-decodable} learner that recovers a small list of candidates at least one of which is close to the true solution. On the other hand, if $\alpha = 1-\epsilon$, where $0< \epsilon < 1/2$ is sufficiently smallconstant, the algorithm recovers a single solution.Along the way, we develop a rejection-sampling-based algorithm to perform noisy location estimation, which may be of independent interest.</details>

### First- and Second-Order Bounds for Adversarial Linear Contextual Bandits

**Authors:** Julia Olkhovskaya, Jack Mayo, Tim van Erven, Gergely Neu, Chen-Yu Wei

**<details><summary>Abstract**</summary>

We consider the adversarial linear contextual bandit setting, whichallows for the loss functions associated with each of $K$ arms to changeover time without restriction. Assuming the $d$-dimensional contexts aredrawn from a fixed known distribution, the worst-case expected regretover the course of $T$ rounds is known to scale as $\tilde O(\sqrt{KdT})$. Under the additional assumption that the density of the contextsis log-concave, we obtain a second-order bound of order $\tildeO(K\sqrt{d V_T})$ in terms of the cumulative second moment of thelearner's losses $V_T$, and a closely related first-order bound of order$\tilde O(K\sqrt{d L_T^*})$ in terms of the cumulative loss of the bestpolicy $L_T^*$. Since $V_T$ or $L_T^*$ may be significantly smaller than$T$, these improve over the worst-case regret whenever the environmentis relatively benign. Our results are obtained using a truncated versionof the continuous exponential weights algorithm over the probabilitysimplex, which we analyse by exploiting a novel connection to the linearbandit setting without contexts.</details>

### Flow Factorized Representation Learning

**Authors:** Yue Song, T. Anderson Keller, Nicu Sebe, Max Welling

**<details><summary>Abstract**</summary>

A prominent goal of representation learning research is to achieve representations which are factorized in a useful manner with respect to the ground truth factors of variation. The fields of disentangled and equivariant representation learning have approached this ideal from a range of complimentary perspectives; however, to date, most approaches have proven to either be ill-specified or insufficiently flexible to effectively separate all realistic factors of interest in a learned latent space. In this work, we propose an alternative viewpoint on such structured representation learning which we call Flow Factorized Representation Learning, and demonstrate it to learn both more efficient and more usefully structured representations than existing frameworks. Specifically, we introduce a generative model which specifies a distinct set of latent probability paths that define different input transformations. Each latent flow is generated by the gradient field of a learned potential following dynamic optimal transport. Our novel setup brings new understandings to both \textit{disentanglement} and \textit{equivariance}. We show that our model achieves higher likelihoods on standard representation learning benchmarks while simultaneously being closer to approximately equivariant models. Furthermore, we demonstrate that the transformations learned by our model are flexibly composable and can also extrapolate to new data, implying a degree of robustness and generalizability approaching the ultimate goal of usefully factorized representation learning.</details>

### Flow Matching for Scalable Simulation-Based Inference

**Authors:** Jonas Wildberger, Maximilian Dax, Simon Buchholz, Stephen Green, Jakob H Macke, Bernhard Schölkopf

**<details><summary>Abstract**</summary>

Neural posterior estimation methods based on discrete normalizing flows have become established tools for simulation-based inference (SBI), but scaling them to high-dimensional problems can be challenging. Building on recent advances in generative modeling, we here present flow matching posterior estimation (FMPE), a technique for SBI using continuous normalizing flows. Like diffusion models, and in contrast to discrete flows, flow matching allows for unconstrained architectures, providing enhanced flexibility for complex data modalities. Flow matching, therefore, enables exact density evaluation, fast training, and seamless scalability to large architectures---making it ideal for SBI. We show that FMPE achieves competitive performance on an established SBI benchmark, and then demonstrate its improved scalability on a challenging scientific problem: for gravitational-wave inference, FMPE outperforms methods based on comparable discrete flows, reducing training time by 30\% with substantially improved accuracy. Our work underscores the potential of FMPE to enhance performance in challenging inference scenarios, thereby paving the way for more advanced applications to scientific problems.</details>

### For SALE: State-Action Representation Learning for Deep Reinforcement Learning

**Authors:** Scott Fujimoto, Wei-Di Chang, Edward Smith, Shixiang (Shane) Gu, Doina Precup, David Meger

**<details><summary>Abstract**</summary>

In reinforcement learning (RL), representation learning is a proven tool for complex image-based tasks, but is often overlooked for environments with low-level states, such as physical control problems. This paper introduces SALE, a novel approach for learning embeddings that model the nuanced interaction between state and action, enabling effective representation learning from low-level states. We extensively study the design space of these embeddings and highlight important design considerations. We integrate SALE and an adaptation of checkpoints for RL into TD3 to form the TD7 algorithm, which significantly outperforms existing continuous control algorithms. On OpenAI gym benchmark tasks, TD7 has an average performance gain of 276.7% and 50.7% over TD3 at 300k and 5M time steps, respectively, and works in both the online and offline settings.</details>

### Formulating Discrete Probability Flow Through Optimal Transport

**Authors:** Pengze Zhang, Hubery Yin, Chen Li, Xiaohua Xie

**<details><summary>Abstract**</summary>

Continuous diffusion models are commonly acknowledged to display a deterministic probability flow, whereas discrete diffusion models do not. In this paper, we aim to establish the fundamental theory for the probability flow of discrete diffusion models. Specifically, we first prove that the continuous probability flow is the Monge optimal transport map under certain conditions, and also present an equivalent evidence for discrete cases. In view of these findings, we are then able to define the discrete probability flow in line with the principles of optimal transport. Finally, drawing upon our newly established definitions, we propose a novel sampling method that surpasses previous discrete diffusion models in its ability to generate more certain outcomes. Extensive experiments on the synthetic toy dataset and the CIFAR-10 dataset have validated the effectiveness of our proposed discrete probability flow. Code is released at: https://github.com/PangzeCheung/Discrete-Probability-Flow.</details>

### Fractal Landscapes in Policy Optimization

**Authors:** Tao Wang, Sylvia Herbert, Sicun Gao

**<details><summary>Abstract**</summary>

Policy gradient lies at the core of deep reinforcement learning (RL) in continuous domains. Despite much success, it is often observed in practice that RL training with policy gradient can fail for many reasons, even on standard control problems with known solutions. We propose a framework for understanding one inherent limitation of the policy gradient approach: the optimization landscape in the policy space can be extremely non-smooth or fractal for certain classes of MDPs, such that there does not exist gradient to be estimated in the first place. We draw on techniques from chaos theory and non-smooth analysis, and analyze the maximal Lyapunov exponents and H\"older exponents of the policy optimization objectives. Moreover, we develop a practical method that can estimate the local smoothness of objective function from samples to identify when the training process has encountered fractal landscapes. We show experiments to illustrate how some failure cases of policy optimization can be explained by such fractal landscapes.</details>

### Frequency Domain-Based Dataset Distillation

**Authors:** Donghyeok Shin, Seungjae Shin, Il-chul Moon

**<details><summary>Abstract**</summary>

This paper presents FreD, a novel parameterization method for dataset distillation, which utilizes the frequency domain to distill a small-sized synthetic dataset from a large-sized original dataset. Unlike conventional approaches that focus on the spatial domain, FreD employs frequency-based transforms to optimize the frequency representations of each data instance. By leveraging the concentration of spatial domain information on specific frequency components, FreD intelligently selects a subset of frequency dimensions for optimization, leading to a significant reduction in the required budget for synthesizing an instance. Through the selection of frequency dimensions based on the explained variance, FreD demonstrates both theoretical and empirical evidence of its ability to operate efficiently within a limited budget, while better preserving the information of the original dataset compared to conventional parameterization methods. Furthermore, Based on the orthogonal compatibility of FreD with existing methods, we confirm that FreD consistently improves the performances of existing distillation methods over the evaluation scenarios with different benchmark datasets. We release the code at https://github.com/sdh0818/FreD.</details>

### From Distribution Learning in Training to Gradient Search in Testing for Combinatorial Optimization

**Authors:** Yang Li, Jinpei Guo, Runzhong Wang, Junchi Yan

**<details><summary>Abstract**</summary>

Extensive experiments have gradually revealed the potential performance bottleneck of modeling Combinatorial Optimization (CO) solving as neural solution prediction tasks. The neural networks, in their pursuit of minimizing the average objective score across the distribution of historical problem instances, diverge from the core target of CO of seeking optimal solutions for every test instance. This calls for an effective search on each problem instance, while the model should serve to provide supporting knowledge that benefits the search. To this end, we propose T2TCO (Training to Testing) framework that first leverages the generative modeling to estimate the high-quality solution distribution for each instance during training, and then conducts a gradient-based search within the solution space during testing. The proposed neural search paradigm consistently leverages generative modeling, specifically diffusion, for graduated solution improvement. It disrupts the local structure of the given solution by introducing noise and reconstructs a lower-cost solution guided by the optimization objective.  Experimental results on Traveling Salesman Problem (TSP) and Maximal Independent Set (MIS) show the significant superiority of T2TCO, demonstrating an average performance gain of 49.15% for TSP solving and 17.27% for MIS solving compared to the previous state-of-the-art.</details>

### [Spotlight] From Tempered to Benign Overfitting in ReLU Neural Networks

**Authors:** Guy Kornowski, Gilad Yehudai, Ohad Shamir

**<details><summary>Abstract**</summary>

Overparameterized neural networks (NNs) are observed to generalize well even when trained to perfectly fit noisy data. This phenomenon motivated a large body of work on "benign overfitting", where interpolating predictors achieve near-optimal performance. Recently, it was conjectured and empirically observed that the behavior of NNs is often better described as "tempered overfitting", where the performance is non-optimal yet also non-trivial, and degrades as a function of the noise level. However, a theoretical justification of this claim for non-linear NNs has been lacking so far. In this work, we provide several results that aim at bridging these complementing views. We study a simple classification setting with 2-layer ReLU NNs, and prove that under various assumptions, the type of overfitting transitions from tempered in the extreme case of one-dimensional data, to benign in high dimensions. Thus, we show that the input dimension has a crucial role on the overfitting profile in this setting, which we also validate empirically for intermediate dimensions. Overall, our results shed light on the intricate connections between the dimension, sample size, architecture and training algorithm on the one hand, and the type of resulting overfitting on the other hand.</details>

### Functional Renyi Differential Privacy for Generative Modeling

**Authors:** Dihong Jiang, Sun Sun, Yaoliang Yu

**<details><summary>Abstract**</summary>

Differential privacy (DP) has emerged as a rigorous notion to quantify data privacy. Subsequently, Renyi differential privacy (RDP) becomes an alternative to the ordinary DP notion in both theoretical and empirical studies, for its convenient compositional rules and flexibility. However, most mechanisms with DP (RDP) guarantees are essentially based on randomizing a fixed, finite-dimensional vector output. In this work, following Hall et al. (2013) we further extend RDP to functional outputs, where the output space can be infinite-dimensional, and develop all necessary tools, *e.g.*, (subsampled) Gaussian mechanism, composition, and post-processing rules, to facilitate its practical adoption. As an illustration, we apply functional RDP (f-RDP) to functions in the reproducing kernel Hilbert space (RKHS) to develop a differentially private generative model (DPGM), where training can be interpreted as iteratively releasing loss functions (in an RKHS) with DP (RDP) guarantees. Empirically, the new training paradigm achieves a significant improvement in privacy-utility trade-off compared to existing alternatives, especially when $\epsilon=0.2$. Our code is available at https://github.com/dihjiang/DP-kernel.</details>

### [Spotlight] Future-Dependent Value-Based Off-Policy Evaluation in POMDPs

**Authors:** Masatoshi Uehara, Haruka Kiyohara, Andrew Bennett, Victor Chernozhukov, Nan Jiang, Nathan Kallus, Chengchun Shi, Wen Sun

**<details><summary>Abstract**</summary>

We study off-policy evaluation (OPE) for partially observable MDPs (POMDPs) with general function approximation. Existing methods such as sequential importance sampling estimators and fitted-Q evaluation suffer from the curse of horizon in POMDPs. To circumvent this problem, we develop a novel model-free OPE method by introducing future-dependent value functions that take future proxies as inputs. Future-dependent value functions play similar roles as classical value functions in fully-observable MDPs. We derive a new off-policy Bellman equation for future-dependent value functions as conditional moment equations that use history proxies as instrumental variables. We further propose a minimax learning method to learn future-dependent value functions using the new Bellman equation. We obtain the PAC result, which implies our OPE estimator is close to the true policy value as long as futures and histories contain sufficient information about latent states, and the Bellman completeness. Our code is available at https://github.com/aiueola/neurips2023-future-dependent-ope</details>

### GRAND-SLAMIN’ Interpretable Additive Modeling with Structural Constraints

**Authors:** Shibal Ibrahim, Gabriel Afriat, Kayhan Behdin, Rahul Mazumder

**<details><summary>Abstract**</summary>

Generalized Additive Models (GAMs) are a family of flexible and interpretable models with old roots in statistics. GAMs are often used with pairwise interactions to improve model accuracy while still retaining flexibility and interpretability but lead to computational challenges as we are dealing with order of $p^2$ terms. It is desirable to restrict the number of components (i.e., encourage sparsity) for easier interpretability, and better computational and statistical properties. Earlier approaches, considering sparse pairwise interactions, have limited scalability, especially when imposing additional structural interpretability constraints. We propose a flexible GRAND-SLAMIN framework that can learn GAMs with interactions under sparsity and additional structural constraints in a differentiable end-to-end fashion. We customize first-order gradient-based optimization to perform sparse backpropagation to exploit sparsity in additive effects for any differentiable loss function in a GPU-compatible manner. Additionally, we establish novel non-asymptotic prediction bounds for our estimators with tree-based shape functions. Numerical experiments on real-world datasets show that our toolkit performs favorably in terms of performance, variable selection and scalability when compared with popular toolkits to fit GAMs with interactions. Our work expands the landscape of interpretable modeling while maintaining prediction accuracy competitive with non-interpretable black-box models. Our code is available at https://github.com/mazumder-lab/grandslamin.</details>

### Generalised f-Mean Aggregation for Graph Neural Networks

**Authors:** Ryan Kortvelesy, Steven D Morad, Amanda Prorok

**<details><summary>Abstract**</summary>

Graph Neural Network (GNN) architectures are defined by their implementations of update and aggregation modules. While many works focus on new ways to parametrise the update modules, the aggregation modules receive comparatively little attention. Because it is difficult to parametrise aggregation functions, currently most methods select a ``standard aggregator'' such as mean, sum, or max. While this selection is often made without any reasoning, it has been shown that the choice in aggregator has a significant impact on performance, and the best choice in aggregator is problem-dependent. Since aggregation is a lossy operation, it is crucial to select the most appropriate aggregator in order to minimise information loss. In this paper, we present GenAgg, a generalised aggregation operator, which parametrises a function space that includes all standard aggregators. In our experiments, we show that GenAgg is able to represent the standard aggregators with much higher accuracy than baseline methods. We also show that using GenAgg as a drop-in replacement for an existing aggregator in a GNN often leads to a significant boost in performance across various tasks.</details>

### [Spotlight] Generalizing Importance Weighting to A Universal Solver for Distribution Shift Problems

**Authors:** Tongtong Fang, Nan Lu, Gang Niu, Masashi Sugiyama

**<details><summary>Abstract**</summary>

Distribution shift (DS) may have two levels: the distribution itself changes, and the support (i.e., the set where the probability density is non-zero) also changes. When considering the support change between the training and test distributions, there can be four cases: (i) they exactly match; (ii) the training support is wider (and thus covers the test support); (iii) the test support is wider; (iv) they partially overlap. Existing methods are good at cases (i) and (ii), while cases (iii) and (iv) are more common nowadays but still under-explored. In this paper, we generalize importance weighting (IW), a golden solver for cases (i) and (ii), to a universal solver for all cases. Specifically, we first investigate why IW might fail in cases (iii) and (iv); based on the findings, we propose generalized IW (GIW) that could handle cases (iii) and (iv) and would reduce to IW in cases (i) and (ii). In GIW, the test support is split into an in-training (IT) part and an out-of-training (OOT) part, and the expected risk is decomposed into a weighted classification term over the IT part and a standard classification term over the OOT part, which guarantees the risk consistency of GIW. Then, the implementation of GIW consists of three components: (a) the split of validation data is carried out by the one-class support vector machine, (b) the first term of the empirical risk can be handled by any IW algorithm given training data and IT validation data, and (c) the second term just involves OOT validation data. Experiments demonstrate that GIW is a universal solver for DS problems, outperforming IW methods in cases (iii) and (iv).</details>

### Generating Images with Multimodal Language Models

**Authors:** Jing Yu Koh, Daniel Fried, Russ Salakhutdinov

**<details><summary>Abstract**</summary>

We propose a method to fuse frozen text-only large language models (LLMs) with pre-trained image encoder and decoder models, by mapping between their embedding spaces. Our model demonstrates a wide suite of multimodal capabilities: image retrieval, novel image generation, and multimodal dialogue. Ours is the first approach capable of conditioning on arbitrarily interleaved image and text inputs to generate coherent image (and text) outputs. To achieve strong performance on image generation, we propose an efficient mapping network to ground the LLM to an off-the-shelf text-to-image generation model. This mapping network translates hidden representations of text into the embedding space of the visual models, enabling us to leverage the strong text representations of the LLM for visual outputs. Our approach outperforms baseline generation models on tasks with longer and more complex language. In addition to novel image generation, our model is also capable of image retrieval from a prespecified dataset, and decides whether to retrieve or generate at inference time. This is done with a learnt decision module which conditions on the hidden representations of the LLM. Our model exhibits a wider range of capabilities compared to prior multimodal language models. It can process image-and-text inputs, and produce retrieved images, generated images, and generated text — outperforming non-LLM based generation models across several text-to-image tasks that measure context dependence.</details>

### Generative Modelling of Stochastic Actions with Arbitrary Constraints in Reinforcement Learning

**Authors:** Changyu CHEN, Ramesha Karunasena, Thanh Nguyen, Arunesh Sinha, Pradeep Varakantham

**<details><summary>Abstract**</summary>

Many problems in Reinforcement Learning (RL) seek an optimal policy with large discrete multidimensional yet unordered action spaces; these include problems in randomized allocation of resources such as placements of multiple security resources and emergency response units, etc. A challenge in this setting is that the underlying action space is categorical (discrete and unordered) and large, for which existing RL methods do not perform well. Moreover, these problems require validity of the realized action (allocation); this validity constraint is often difficult to express compactly in a closed mathematical form. The allocation nature of the problem also prefers stochastic optimal policies, if one exists. In this work, we address these challenges by (1) applying a (state) conditional normalizing flow to compactly represent the stochastic policy — the compactness arises due to the network only producing one sampled action and the corresponding log probability of the action, which is then used by an actor-critic method; and (2) employing an invalid action rejection method (via a valid action oracle) to update the base policy. The action rejection is enabled by a modified policy gradient that we derive. Finally, we conduct extensive experiments to show the scalability of our approach compared to prior methods and the ability to enforce arbitrary state-conditional constraints on the support of the distribution of actions in any state.</details>

### Generative Neural Fields by Mixtures of Neural Implicit Functions

**Authors:** Tackgeun You, Mijeong Kim, Jungtaek Kim, Bohyung Han

**<details><summary>Abstract**</summary>

We propose a novel approach to learning the generative neural fields represented by linear combinations of implicit basis networks. Our algorithm learns basis networks in the form of implicit neural representations and their coefficients in a latent space by either conducting meta-learning or adopting auto-decoding paradigms. The proposed method easily enlarges the capacity of generative neural fields by increasing the number of basis networks while maintaining the size of a network for inference to be small through their weighted model averaging. Consequently, sampling instances using the model is efficient in terms of latency and memory footprint. Moreover, we customize denoising diffusion probabilistic model for a target task to sample latent mixture coefficients, which allows our final model to generate unseen data effectively. Experiments show that our approach achieves competitive generation performance on diverse benchmarks for images, voxel data, and NeRF scenes without sophisticated designs for specific modalities and domains.</details>

### GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization

**Authors:** Vicente Vivanco Cepeda, Gaurav Kumar Nayak, Mubarak Shah

**<details><summary>Abstract**</summary>

Worldwide Geo-localization aims to pinpoint the precise location of images taken anywhere on Earth. This task has considerable challenges due to the immense variation in geographic landscapes. The image-to-image retrieval-based approaches fail to solve this problem on a global scale as it is not feasible to construct a large gallery of images covering the entire world. Instead, existing approaches divide the globe into discrete geographic cells, transforming the problem into a classification task. However, their performance is limited by the predefined classes and often results in inaccurate localizations when an image's location significantly deviates from its class center. To overcome these limitations, we propose GeoCLIP, a novel CLIP-inspired Image-to-GPS retrieval approach that enforces alignment between the image and its corresponding GPS locations. GeoCLIP's location encoder models the Earth as a continuous function by employing positional encoding through random Fourier features and constructing a hierarchical representation that captures information at varying resolutions to yield a semantically rich high-dimensional feature suitable to use even beyond geo-localization. To the best of our knowledge, this is the first work employing GPS encoding for geo-localization. We demonstrate the efficacy of our method via extensive experiments and ablations on benchmark datasets. We achieve competitive performance with just 20% of training data, highlighting its effectiveness even in limited-data settings. Furthermore, we qualitatively demonstrate geo-localization using a text query by leveraging the CLIP backbone of our image encoder. The project webpage is available at: https://vicentevivan.github.io/GeoCLIP</details>

### Geometric Algebra Transformer

**Authors:** Johann Brehmer, Pim de Haan, Sönke Behrends, Taco Cohen

**<details><summary>Abstract**</summary>

Problems involving geometric data arise in physics, chemistry, robotics, computer vision, and many other fields. Such data can take numerous forms, for instance points, direction vectors, translations, or rotations, but to date there is no single architecture that can be applied to such a wide variety of geometric types while respecting their symmetries. In this paper we introduce the Geometric Algebra Transformer (GATr), a general-purpose architecture for geometric data. GATr represents inputs, outputs, and hidden states in the projective geometric (or Clifford) algebra, which offers an efficient 16-dimensional vector-space representation of common geometric objects as well as operators acting on them. GATr is equivariant with respect to E(3), the symmetry group of 3D Euclidean space. As a Transformer, GATr is versatile, efficient, and scalable. We demonstrate GATr in problems from n-body modeling to wall-shear-stress estimation on large arterial meshes to robotic motion planning. GATr consistently outperforms both non-geometric and equivariant baselines in terms of error, data efficiency, and scalability.</details>

### Geometric Analysis of Matrix Sensing over Graphs

**Authors:** Haixiang Zhang, Ying Chen, Javad Lavaei

**<details><summary>Abstract**</summary>

In this work, we consider the problem of matrix sensing over graphs (MSoG). As a general case of matrix completion and matrix sensing problems, the MSoG problem has not been analyzed in the literature and the existing results cannot be directly applied to the MSoG problem. This work provides the first theoretical results on the optimization landscape of the MSoG problem. More specifically, we propose a new condition, named the $\Omega$-RIP condition, to characterize the optimization complexity of the problem. In addition, with an improved regularizer of the incoherence, we prove that the strict saddle property holds for the MSoG problem with high probability under the incoherence condition and the $\Omega$-RIP condition, which guarantees the polynomial-time global convergence of saddle-avoiding methods. Compared with state-of-the-art results, the bounds in this work are tight up to a constant. Besides the theoretical guarantees, we numerically illustrate the close relation between the $\Omega$-RIP condition and the optimization complexity.</details>

### Geometric Neural Diffusion Processes

**Authors:** Emile Mathieu, Vincent Dutordoir, Michael Hutchinson, Valentin De Bortoli, Yee Whye Teh, Richard Turner

**<details><summary>Abstract**</summary>

Denoising diffusion models have proven to be a flexible and effective paradigm for generative modelling.Their recent extension to infinite dimensional Euclidean spaces has allowed for the modelling of stochastic processes.However, many problems in the natural sciences incorporate symmetries and involve data living in non-Euclidean spaces.In this work, we extend the framework of diffusion models to incorporate a series of geometric priors in infinite-dimension modelling.We do so by a) constructing a noising process which admits, as limiting distribution, a geometric Gaussian process that transforms under the symmetry group of interest, and b) approximating the score with a neural network that is equivariant w.r.t. this group.We show that with these conditions, the generative functional model admits the same symmetry.We demonstrate scalability and capacity of the model, using a novel Langevin-based conditional sampler, to fit complex scalar and vector fields, with Euclidean and spherical codomain, on synthetic and real-world weather data.</details>

### GlucoSynth: Generating Differentially-Private Synthetic Glucose Traces

**Authors:** Josephine Lamp, Mark Derdzinski, Christopher Hannemann, Joost van der Linden, Lu Feng, Tianhao Wang, David Evans

**<details><summary>Abstract**</summary>

We focus on the problem of generating high-quality, private synthetic glucose traces, a task generalizable to many other time series sources. Existing methods for time series data synthesis, such as those using Generative Adversarial Networks (GANs), are not able to capture the innate characteristics of glucose data and cannot provide any formal privacy guarantees without severely degrading the utility of the synthetic data. In this paper we present GlucoSynth, a novel privacy-preserving GAN framework to generate synthetic glucose traces. The core intuition behind our approach is to conserve relationships amongst motifs (glucose events) within the traces, in addition to temporal dynamics. Our framework incorporates differential privacy mechanisms to provide strong formal privacy guarantees. We provide a comprehensive evaluation on the real-world utility of the data using 1.2 million glucose traces; GlucoSynth outperforms all previous methods in its ability to generate high-quality synthetic glucose traces with strong privacy guarantees.</details>

### Goal Driven Discovery of Distributional Differences via Language Descriptions

**Authors:** Ruiqi Zhong, Peter Zhang, Steve Li, Jinwoo Ahn, Dan Klein, Jacob Steinhardt

**<details><summary>Abstract**</summary>

Exploring large corpora can generate useful discoveries but is time-consuming for humans.    We formulate a new task, D5, that automatically discovers differences between two large corpora in a goal-driven way.     The task input is a problem comprising a user-specified research goal (“*comparing the side effects of drug A and drug*”) and a corpus pair (two large collections of patients' self-reported reactions after taking each drug).     The output is a goal-related description (discovery) of how these corpora differ (patients taking drug A “*mention feelings of paranoia*” more often).    We build a D5 system, and to quantitatively evaluate its performance, we 1) build a diagnostic benchmark, SynD5, to test whether it can recover known differences between two synthetic corpora, and 2) contribute a meta-dataset, OpenD5, aggregating 675 open-ended problems ranging across business, social sciences, humanities, machine learning, and health.    With both synthetic and real datasets, we confirm that language models can leverage the user-specified goals to propose more relevant candidate discoveries, and they sometimes produce discoveries previously unknown to the authors, including demographic differences in discussion topics, political stances in speech, insights in commercial reviews, and error patterns in NLP models.    Finally, we discuss the limitations of the current D5 system, which discovers correlation rather than causation and has the potential to reinforce societal biases when misused; therefore, practitioners should treat the outputs of our system with caution.</details>

### Goal-Conditioned Predictive Coding for Offline Reinforcement Learning

**Authors:** Zilai Zeng, Ce Zhang, Shijie Wang, Chen Sun

**<details><summary>Abstract**</summary>

Recent work has demonstrated the effectiveness of formulating decision making as supervised learning on offline-collected trajectories. Powerful sequence models, such as GPT or BERT, are often employed to encode the trajectories. However, the benefits of performing sequence modeling on trajectory data remain unclear. In this work, we investigate whether sequence modeling has the ability to condense trajectories into useful representations that enhance policy learning. We adopt a two-stage framework that first leverages sequence models to encode trajectory-level representations, and then learns a goal-conditioned policy employing the encoded representations as its input. This formulation allows us to consider many existing supervised offline RL methods as specific instances of our framework. Within this framework, we introduce Goal-Conditioned Predictive Coding (GCPC), a sequence modeling objective that yields powerful trajectory representations and leads to performant policies. Through extensive empirical evaluations on AntMaze, FrankaKitchen and Locomotion environments, we observe that sequence modeling can have a significant impact on challenging decision making tasks. Furthermore, we demonstrate that GCPC learns a goal-conditioned latent representation encoding the future trajectory, which enables competitive performance on all three benchmarks.</details>

### Goal-conditioned Offline Planning from Curious Exploration

**Authors:** Marco Bagatella, Georg Martius

**<details><summary>Abstract**</summary>

Curiosity has established itself as a powerful exploration strategy in deep reinforcement learning. Notably, leveraging expected future novelty as intrinsic motivation has been shown to efficiently generate exploratory trajectories, as well as a robust dynamics model. We consider the challenge of extracting goal-conditioned behavior from the products of such unsupervised exploration techniques, without any additional environment interaction. We find that conventional goal-conditioned reinforcement learning approaches for extracting a value function and policy fall short in this difficult offline setting. By analyzing the geometry of optimal goal-conditioned value functions, we relate this issue to a specific class of estimation artifacts in learned values. In order to mitigate their occurrence, we propose to combine model-based planning over learned value landscapes with a graph-based value aggregation scheme. We show how this combination can correct both local and global artifacts, obtaining significant improvements in zero-shot goal-reaching performance across diverse simulated environments.</details>

### GradOrth: A Simple yet Efficient Out-of-Distribution Detection with Orthogonal Projection of Gradients

**Authors:** Sima Behpour, Thang Long Doan, Xin Li, Wenbin He, Liang Gou, Liu Ren

**<details><summary>Abstract**</summary>

Detecting out-of-distribution (OOD) data is crucial for ensuring the safe deployment of machine learning models in real-world applications. However, existing OOD detection approaches primarily rely on the feature maps or the full gradient space information to derive OOD scores neglecting the role of \textbf{most important parameters} of the pre-trained network over In-Distribution data. In this study, we propose a novel approach called GradOrth to facilitate OOD detection based on one intriguing observation that the important features to identify OOD data lie in the lower-rank subspace of in-distribution (ID) data.In particular, we identify OOD data by computing the norm of gradient projection on \textit{the subspaces considered \textbf{important} for the in-distribution data}. A large orthogonal projection value (i.e. a small projection value) indicates the sample as OOD as it captures a weak correlation of the in-distribution (ID) data. This simple yet effective method exhibits outstanding performance, showcasing a notable reduction in the average false positive rate at a 95\% true positive rate (FPR95) of up to 8\% when compared to the current state-of-the-art methods.</details>

### Gradient-Based Feature Learning under Structured Data

**Authors:** Alireza Mousavi-Hosseini, Denny Wu, Taiji Suzuki, Murat Erdogdu

**<details><summary>Abstract**</summary>

Recent works have demonstrated that the sample complexity of gradient-based learning of single index models, i.e. functions that depend on a 1-dimensional projection of the input data, is governed by their information exponent. However, these results are only concerned with isotropic data, while in practice the input often contains additional structure which can implicitly guide the algorithm. In this work, we investigate the effect of a spiked covariance structure and reveal several interesting phenomena. First, we show that in the anisotropic setting, the commonly used spherical gradient dynamics may fail to recover the true direction, even when the spike is perfectly aligned with the target direction. Next, we show that appropriate weight normalization that is reminiscent of batch normalization can alleviate this issue. Further, by exploiting the alignment between the (spiked) input covariance and the target, we obtain improved sample complexity compared to the isotropic case. In particular, under the spiked model with a suitably large spike, the sample complexity of gradient-based training can be made independent of the information exponent while also outperforming lower bounds for rotationally invariant kernel methods.</details>

### Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit Diversity Modeling

**Authors:** Haotao Wang, Ziyu Jiang, Yuning You, Yan Han, Gaowen Liu, Jayanth Srinivasa, Ramana Kompella, Zhangyang "Atlas" Wang

**<details><summary>Abstract**</summary>

Graph neural networks (GNNs) have found extensive applications in learning from graph data. However, real-world graphs often possess diverse structures and comprise nodes and edges of varying types. To bolster the generalization capacity of GNNs, it has become customary to augment training graph structures through techniques like graph augmentations and large-scale pre-training on a wider array of graphs. Balancing this diversity while avoiding increased computational costs and the notorious trainability issues of GNNs is crucial. This study introduces the concept of Mixture-of-Experts (MoE) to GNNs, with the aim of augmenting their capacity to adapt to a diverse range of training graph structures, without incurring explosive computational overhead. The proposed Graph Mixture of Experts (GMoE) model empowers individual nodes in the graph to dynamically and adaptively select more general information aggregation experts. These experts are trained to capture distinct subgroups of graph structures and to incorporate information with varying hop sizes, where those with larger hop sizes specialize in gathering information over longer distances. The effectiveness of GMoE is validated through a series of experiments on a diverse set of tasks, including graph, node, and link prediction, using the OGB benchmark. Notably, it enhances ROC-AUC by $1.81\%$ in ogbg-molhiv and by $1.40\%$ in ogbg-molbbbp, when compared to the non-MoE baselines. Our code is publicly available at https://github.com/VITA-Group/Graph-Mixture-of-Experts.</details>

### Grassmann Manifold Flows for Stable Shape Generation

**Authors:** Ryoma Yataka, Kazuki Hirashima, Masashi Shiraishi

**<details><summary>Abstract**</summary>

Recently, studies on machine learning have focused on methods that use symmetry implicit in a specific manifold as an inductive bias.Grassmann manifolds provide the ability to handle fundamental shapes represented as shape spaces, enabling stable shape analysis. In this paper, we present a novel approach in which we establish the theoretical foundations for learning distributions on the Grassmann manifold via continuous normalization flows, with the explicit goal of generating stable shapes.Our approach facilitates more robust generation by effectively eliminating the influence of extraneous transformations, such as rotations and inversions, through learning and generating within a Grassmann manifold designed to accommodate the essential shape information of the object.The experimental results indicated that the proposed method could generate high-quality samples by capturing the data structure.Furthermore, the proposed method significantly outperformed state-of-the-art methods in terms of the log-likelihood or evidence lower bound.The results obtained are expected to stimulate further research in this field, leading to advances for stable shape generation and analysis.</details>

### Guiding The Last Layer in Federated Learning with Pre-Trained Models

**Authors:** Gwen Legate, Nicolas Bernier, Lucas Page-Caccia, Edouard Oyallon, Eugene Belilovsky

**<details><summary>Abstract**</summary>

Federated Learning (FL) is an emerging paradigm that allows a model to be trained across a number of participants without sharing data. Recent works have begun to consider the effects of using pre-trained models as an initialization point for existing FL algorithms; however, these approaches ignore the vast body of efficient transfer learning literature from the centralized learning setting. Here we revisit the problem of FL from a pre-trained model considered in prior work and expand it to a set of computer vision transfer learning problems. We first observe that simply fitting a linear classification head can be efficient in many cases. We then show that in the FL setting, fitting a classifier using the Nearest Class Means (NCM) can be done exactly and  orders of magnitude more efficiently than existing proposals, while obtaining strong performance. Finally, we demonstrate that using a two-stage approach of obtaining the classifier and then fine-tuning the model can yield rapid convergence and improved generalization in the federated setting. We demonstrate the potential our method has to reduce communication and compute costs while achieving better model performance.</details>

### H-nobs: Achieving Certified Fairness and Robustness in Distributed Learning on Heterogeneous Datasets

**Authors:** Guanqiang Zhou, Ping Xu, Yue Wang, Zhi Tian

**<details><summary>Abstract**</summary>

Fairness and robustness are two important goals in the design of modern distributed learning systems. Despite a few prior works attempting to achieve both fairness and robustness, some key aspects of this direction remain underexplored. In this paper, we try to answer three largely unnoticed and unaddressed questions that are of paramount significance to this topic: (i) What makes jointly satisfying fairness and robustness difficult? (ii) Is it possible to establish theoretical guarantee for the dual property of fairness and robustness? (iii) How much does fairness have to sacrifice at the expense of robustness being incorporated into the system? To address these questions, we first identify data heterogeneity as the key difficulty of combining fairness and robustness. Accordingly, we propose a fair and robust framework called H-nobs which can offer certified fairness and robustness through the adoption of two key components, a fairness-promoting objective function and a simple robust aggregation scheme called norm-based screening (NBS). We explain in detail why NBS is the suitable scheme in our algorithm in contrast to other robust aggregation measures. In addition, we derive three convergence theorems for H-nobs in cases of the learning model being nonconvex, convex, and strongly convex respectively, which provide theoretical guarantees for both fairness and robustness. Further, we empirically investigate the influence of the robust mechanism (NBS) on the fairness performance of H-nobs, the very first attempt of such exploration.</details>

### H3T: Efficient Integration of Memory Optimization and Parallelism for Large-scale Transformer Training

**Authors:** Yuzhong Wang, Xu Han, Weilin Zhao, Guoyang Zeng, Zhiyuan Liu, Maosong Sun

**<details><summary>Abstract**</summary>

In recent years, big models based on Transformers have achieved state-of-the-art performance on many artificial intelligence (AI) tasks.Despite the success of these Transformer-based models, their huge parameter size poses a serious challenge to their training, both from the storage and computation perspectives.To this end, memory optimization (e.g., rematerialization and offloading) and parallelism (e.g., data parallelism and model parallelism) are widely explored to make training Transformers more efficient.In this paper, we propose a framework to automatically find an efficient integration of memory optimization and parallelism for High-Throughput Transformer Training (named H3T), which is rarely considered by existing efforts for training big Transformer-based models.Specifically, we design search algorithms to combine appropriate memory optimization strategies and parallelism schemes to achieve a balance between memory overhead and training efficiency.We implement H3T based on an open-source toolkit BMTrain and then use H3T to train the Transformers of different sizes to evaluate the efficiency of H3T.The experimental results show that H3T outperforms the most popular deep learning (DL) toolkit Megatron-DeepSpeed by $1.2	imes \sim 4.3	imes$ training speed while reducing $34.6\% \sim 80.5\%$ of memory overhead.Moreover, H3T can use only 64 NVIDIA A100 GPUs to train GPT-3-175B, which is very difficult for existing DL toolkits. The source code is available at https://github.com/OpenBMB/BMTrain/tree/h3t.</details>

### HEDNet: A Hierarchical Encoder-Decoder Network for 3D Object Detection in Point Clouds

**Authors:** Gang Zhang, Chen Junnan, Guohuan Gao, Jianmin Li, Xiaolin Hu

**<details><summary>Abstract**</summary>

3D object detection in point clouds is important for autonomous driving systems. A primary challenge in 3D object detection stems from the sparse distribution of points within the 3D scene. Existing high-performance methods typically employ 3D sparse convolutional neural networks with small kernels to extract features. To reduce computational costs, these methods resort to submanifold sparse convolutions, which prevent the information exchange among spatially disconnected features. Some recent approaches have attempted to address this problem by introducing large-kernel convolutions or self-attention mechanisms, but they either achieve limited accuracy improvements or incur excessive computational costs. We propose HEDNet, a hierarchical encoder-decoder network for 3D object detection, which leverages encoder-decoder blocks to capture long-range dependencies among features in the spatial space, particularly for large and distant objects. We conducted extensive experiments on the Waymo Open and nuScenes datasets. HEDNet achieved superior detection accuracy on both datasets than previous state-of-the-art methods with competitive efficiency. The code is available at https://github.com/zhanggang001/HEDNet.</details>

### HQA-Attack: Toward High Quality Black-Box Hard-Label Adversarial Attack on Text

**Authors:** Han Liu, Zhi Xu, Xiaotong Zhang, Feng Zhang, Fenglong Ma, Hongyang Chen, Hong Yu, Xianchao Zhang

**<details><summary>Abstract**</summary>

Black-box hard-label adversarial attack on text is a practical and challenging task, as the text data space is inherently discrete and non-differentiable, and only the predicted label is accessible. Research on this problem is still in the embryonic stage and only a few methods are available. Nevertheless, existing methods rely on the complex heuristic algorithm or unreliable gradient estimation strategy, which probably fall into the local optimum and inevitably consume numerous queries, thus are difficult to craft satisfactory adversarial examples with high semantic similarity and low perturbation rate in a limited query budget. To alleviate above issues, we propose a simple yet effective framework to generate high quality textual adversarial examples under the black-box hard-label attack scenarios, named HQA-Attack. Specifically, after initializing an adversarial example randomly, HQA-attack first constantly substitutes original words back as many as possible, thus shrinking the perturbation rate. Then it leverages the synonym set of the remaining changed words to further optimize the adversarial example with the direction which can improve the semantic similarity and satisfy the adversarial condition simultaneously. In addition, during the optimizing procedure, it searches a transition synonym word for each changed word, thus avoiding traversing the whole synonym set and reducing the query number to some extent. Extensive experimental results on five text classification datasets, three natural language inference datasets and two real-world APIs have shown that the proposed HQA-Attack method outperforms other strong baselines significantly.</details>

### Handling Data Heterogeneity via Architectural Design for Federated Visual Recognition

**Authors:** Sara Pieri, Jose Restom, Samuel Horváth, Hisham Cholakkal

**<details><summary>Abstract**</summary>

Federated Learning (FL) is a promising research paradigm that enables the collaborative training of machine learning models among various parties without the need for sensitive information exchange. Nonetheless, retaining data in individual clients introduces fundamental challenges to achieving performance on par with centrally trained models. Our study provides an extensive review of federated learning applied to visual recognition. It underscores the critical role of thoughtful architectural design choices in achieving optimal performance, a factor often neglected in the FL literature. Many existing FL solutions are tested on shallow or simple networks, which may not accurately reflect real-world applications. This practice restricts the transferability of research findings to large-scale visual recognition models. Through an in-depth analysis of diverse cutting-edge architectures such as convolutional neural networks, transformers, and MLP-mixers,  we experimentally demonstrate that architectural choices can substantially enhance FL systems' performance, particularly when handling heterogeneous data.  We study visual recognition models from five different architectural families on four challenging FL datasets. We also re-investigate the inferior performance convolution-based architectures in the FL setting and analyze the influence of normalization layers on the FL performance. Our findings emphasize the importance of architectural design for computer vision tasks in practical scenarios, effectively narrowing the performance gap between federated and centralized learning.</details>

### Hardness of Low Rank Approximation of Entrywise Transformed Matrix Products

**Authors:** Tamas Sarlos, Xingyou Song, David Woodruff, Richard Zhang

**<details><summary>Abstract**</summary>

Inspired by fast algorithms in natural language processing, we study low rank approximation in the entrywise transformed setting where we want to find a good rank $k$ approximation to $f(U \cdot V)$, where $U, V^\top \in \mathbb{R}^{n \times r}$ are given, $r = O(\log(n))$, and $f(x)$ is a general scalar function. Previous work in sublinear low rank approximation has shown that if both (1) $U = V^\top$ and (2) $f(x)$ is a PSD kernel function, then there is an $O(nk^{\omega-1})$ time constant relative error approximation algorithm, where $\omega \approx 2.376$ is the exponent of matrix multiplication. We give the first conditional time hardness results for this problem, demonstrating that both conditions (1) and (2) are in fact necessary for getting better than $n^{2-o(1)}$ time for a relative error low rank approximation for a wide class of functions. We give novel reductions from the Strong Exponential Time Hypothesis (SETH) that rely on lower bounding the leverage scores of flat sparse vectors and hold even when the rank of the transformed matrix $f(UV)$ and the target rank are $n^{o(1)}$, and when $U = V^\top$. Furthermore, even when $f(x) = x^p$ is a simple polynomial, we give runtime lower bounds in the case when $U \neq V^\top$ of the form $\Omega(\min(n^{2-o(1)}, \Omega(2^p)))$. Lastly, we demonstrate that our lower bounds are tight by giving an $O(n \cdot \text{poly}(k, 2^p, 1/\epsilon))$ time relative error approximation algorithm and a fast $O(n \cdot \text{poly}(k, p, 1/\epsilon))$ additive error approximation using fast tensor-based sketching. Additionally, since our low rank algorithms rely on matrix-vector product subroutines, our lower bounds extend to show that computing $f(UV)W$, for even a small matrix $W$, requires $\Omega(n^{2-o(1)})$ time.</details>

### Hardware Resilience Properties of Text-Guided Image Classifiers

**Authors:** Syed Talal Wasim, Kabila Haile Soboka, Abdulrahman Mahmoud, Salman Khan, David Brooks, Gu-Yeon Wei

**<details><summary>Abstract**</summary>

This paper presents a novel method to enhance the reliability of image classification models during deployment in the face of transient hardware errors. By utilizing enriched text embeddings derived from GPT-3 with question prompts per class and CLIP pretrained text encoder, we investigate their impact as an initialization for the classification layer. Our approach achieves a remarkable $5.5	imes$ average increase in hardware reliability (and up to $14	imes$) across various architectures in the most critical layer, with minimal accuracy drop ($0.3\%$ on average) compared to baseline PyTorch models. Furthermore, our method seamlessly integrates with any image classification backbone, showcases results across various network architectures, decreases parameter and FLOPs overhead, and follows a consistent training recipe. This research offers a practical and efficient solution to bolster the robustness of image classification models against hardware failures, with potential implications for future studies in this domain. Our code and models are released at https://github.com/TalalWasim/TextGuidedResilience.</details>

### Harnessing the power of choices in decision tree learning

**Authors:** Guy Blanc, Jane Lange, Chirag Pabbaraju, Colin Sullivan, Li-Yang Tan, Mo Tiwari

**<details><summary>Abstract**</summary>

We propose a simple generalization of standard and empirically successful decision tree learning algorithms such as ID3, C4.5, and CART. These algorithms, which have been central to machine learning for decades, are greedy in nature: they grow a decision tree by iteratively splitting on the best attribute. Our algorithm, Top-$k$, considers the $k$ best attributes as possible splits instead of just the single best attribute. We demonstrate, theoretically and empirically, the power of this simple generalization. We first prove a greediness hierarchy theorem showing that for every $k\in \mathbb{N}$, Top-$(k+1)$ can be dramatically more powerful than Top-$k$: there are data distributions for which the former achieves accuracy $1-\epsilon$, whereas the latter only achieves accuracy $\frac{1}{2}+\epsilon$. We then show, through extensive experiments, that Top-$k$ outperforms the two main approaches to decision tree learning: classic greedy algorithms and more recent ``optimal decision tree'' algorithms. On one hand, Top-$k$ consistently enjoys significant accuracy gains over greedy algorithms across a wide range of benchmarks. On the other hand, Top-$k$ is markedly more scalable than optimal decision tree algorithms and is able to handle dataset and feature set sizes that remain far beyond the reach of these algorithms. The code to reproduce our results is available at https://github.com/SullivanC19/pydl8.5-topk.</details>

### [Spotlight] Hierarchically Gated Recurrent Neural Network for Sequence Modeling

**Authors:** Zhen Qin, Songlin Yang, Yiran Zhong

**<details><summary>Abstract**</summary>

Transformers have surpassed RNNs in popularity due to their superior abilities in parallel training and long-term dependency modeling.Recently, there has been a renewed interest in using linear RNNs for efficient sequence modeling.These linear RNNs often employ gating mechanisms in the output of the linear recurrence layer while ignoring the significance of using forget gates within the recurrence. In this paper, we propose a gated linear RNN model dubbed Hierarchically Gated Recurrent Neural Network (HGRN), which includes forget gates that are lower bounded by a learnable value. The lower bound increases monotonically when moving up layers. This allows the upper layers to model long-term dependencies and the lower layers to model more local, short-term dependencies. Experiments on language modeling, image classification, and long-range arena benchmarks showcase the efficiency and effectiveness of our proposed model. The source code is available at https://github.com/OpenNLPLab/HGRN.</details>

### High dimensional, tabular deep learning with an auxiliary knowledge graph

**Authors:** Camilo Ruiz, Hongyu Ren, Kexin Huang, Jure Leskovec

**<details><summary>Abstract**</summary>

Machine learning models exhibit strong performance on datasets with abundant labeled samples. However, for tabular datasets with extremely high $d$-dimensional features but limited $n$ samples (i.e. $d \gg n$), machine learning models struggle to achieve strong performance due to the risk of overfitting. Here, our key insight is that there is often abundant, auxiliary domain information describing input features which can be structured as a heterogeneous knowledge graph (KG). We propose PLATO, a method that achieves strong performance on tabular data with $d \gg n$ by using an auxiliary KG describing input features to regularize a multilayer perceptron (MLP). In PLATO, each input feature corresponds to a node in the auxiliary KG. In the MLP’s first layer, each input feature also corresponds to a weight vector. PLATO is based on the inductive bias that two input features corresponding to similar nodes in the auxiliary KG should have similar weight vectors in the MLP's first layer. PLATO captures this inductive bias by inferring the weight vector for each input feature from its corresponding node in the KG via a trainable message-passing function. Across 6 $d \gg n$ datasets, PLATO outperforms 13 state-of-the-art baselines by up to 10.19%.</details>

### [Spotlight] Honesty Is the Best Policy: Defining and Mitigating AI Deception

**Authors:** Francis Ward, Francesca Toni, Francesco Belardinelli, Tom Everitt

**<details><summary>Abstract**</summary>

Deceptive agents are a challenge for the safety, trustworthiness, and cooperation of AI systems. We focus on the problem that agents might deceive in order to achieve their goals (for instance, in our experiments with language models, the goal of being evaluated as truthful).There are a number of existing definitions of deception in the literature on game theory and symbolic AI, but there is no overarching theory of deception for learning agents in games. We introduce a formaldefinition of deception in structural causal games, grounded in the philosophyliterature, and applicable to real-world machine learning systems.Several examples and results illustrate that our formal definition aligns with the philosophical and commonsense meaning of deception.Our main technical result is to provide graphical criteria for deception. We show, experimentally, that these results can be used to mitigate deception in reinforcement learning agents and language models.</details>

### How Re-sampling Helps for Long-Tail Learning?

**Authors:** Jiang-Xin Shi, Tong Wei, Yuke Xiang, Yu-Feng Li

**<details><summary>Abstract**</summary>

Long-tail learning has received significant attention in recent years due to the challenge it poses with extremely imbalanced datasets. In these datasets, only a few classes (known as the head classes) have an adequate number of training samples, while the rest of the classes (known as the tail classes) are infrequent in the training data. Re-sampling is a classical and widely used approach for addressing class imbalance issues. Unfortunately, recent studies claim that re-sampling brings negligible performance improvements in modern long-tail learning tasks. This paper aims to investigate this phenomenon systematically. Our research shows that re-sampling can considerably improve generalization when the training images do not contain semantically irrelevant contexts. In other scenarios, however, it can learn unexpected spurious correlations between irrelevant contexts and target labels. We design experiments on two homogeneous datasets, one containing irrelevant context and the other not, to confirm our findings. To prevent the learning of spurious correlations, we propose a new context shift augmentation module that generates diverse training images for the tail class by maintaining a context bank extracted from the head-class images. Experiments demonstrate that our proposed module can boost the generalization and outperform other approaches, including class-balanced re-sampling, decoupled classifier re-training, and data augmentation methods. The source code is available at https://www.lamda.nju.edu.cn/code_CSA.ashx.</details>

### [Spotlight] How to Scale Your EMA

**Authors:** Dan Busbridge, Jason Ramapuram, Pierre Ablin, Tatiana Likhomanenko, Eeshan Gunesh Dhekane, Xavier Suau Cuadros, Russell Webb

**<details><summary>Abstract**</summary>

Preserving training dynamics across batch sizes is an important tool for practical machine learning as it enables the trade-off between batch size and wall-clock time. This trade-off is typically enabled by a scaling rule, for example, in stochastic gradient descent, one should scale the learning rate linearly with the batch size. Another important machine learning tool is the model EMA, a functional copy of a target model, whose parameters move towards those of its target model according to an Exponential Moving Average (EMA) at a rate parameterized by a momentum hyperparameter. This model EMA can improve the robustness and generalization of supervised learning, stabilize pseudo-labeling, and provide a learning signal for Self-Supervised Learning (SSL). Prior works have not considered the optimization of the model EMA when performing scaling, leading to different training dynamics across batch sizes and lower model performance. In this work, we provide a scaling rule for optimization in the presence of a model EMA and demonstrate the rule's validity across a range of architectures, optimizers, and data modalities.  We also show the rule's validity where the model EMA contributes to the optimization of the target model, enabling us to train EMA-based pseudo-labeling and SSL methods at small and large batch sizes. For SSL, we enable training of BYOL up to batch size 24,576 without sacrificing performance, a 6$\times$ wall-clock time reduction under idealized hardware settings.</details>

### How2comm: Communication-Efficient and Collaboration-Pragmatic Multi-Agent Perception

**Authors:** Dingkang Yang, Kun Yang, Yuzheng Wang, Jing Liu, Zhi Xu, Rongbin Yin, Peng Zhai, Lihua Zhang

**<details><summary>Abstract**</summary>

Multi-agent collaborative perception has recently received widespread attention as an emerging application in driving scenarios. Despite the advancements in previous efforts, challenges remain due to various noises in the perception procedure, including communication redundancy, transmission delay, and collaboration heterogeneity. To tackle these issues, we propose 	extit{How2comm}, a collaborative perception framework that seeks a trade-off between perception performance and communication bandwidth. Our novelties lie in three aspects. First, we devise a mutual information-aware communication mechanism to maximally sustain the informative features shared by collaborators. The spatial-channel filtering is adopted to perform effective feature sparsification for efficient communication. Second, we present a flow-guided delay compensation strategy to predict future characteristics from collaborators and eliminate feature misalignment due to temporal asynchrony. Ultimately, a pragmatic collaboration transformer is introduced to integrate holistic spatial semantics and temporal context clues among agents. Our framework is thoroughly evaluated on several LiDAR-based collaborative detection datasets in real-world and simulated scenarios. Comprehensive experiments demonstrate the superiority of How2comm and the effectiveness of all its vital components. The code will be released at https://github.com/ydk122024/How2comm.</details>

### Human spatiotemporal pattern learning as probabilistic program synthesis

**Authors:** Tracey Mills, Josh Tenenbaum, Samuel Cheyette

**<details><summary>Abstract**</summary>

People are adept at learning a wide variety of structured patterns from small amounts of data, presenting a conundrum from the standpoint of the bias-variance tradeoff: what kinds of representations and algorithms support the joint flexibility and data-paucity of human learning? One possibility is that people "learn by programming": inducing probabilistic models to fit observed data. Here, we experimentally test human learning in the domain of structured 2-dimensional patterns, using a task in which participants repeatedly predicted where a dot would move based on its previous trajectory. We evaluate human performance against standard parametric and non-parametric time-series models, as well as two Bayesian program synthesis models whose hypotheses vary in their degree of structure: a compositional Gaussian Process model and a structured "Language of Thought" (LoT) model. We find that signatures of human pattern learning are best explained by the LoT model, supporting the idea that the flexibility and data-efficiency of human structure learning can be understood as probabilistic inference over an expressive space of programs.</details>

### Human-Aligned Calibration for AI-Assisted Decision Making

**Authors:** Nina Corvelo Benz, Manuel Rodriguez

**<details><summary>Abstract**</summary>

Whenever a binary classifier is used to provide decision support, it typically provides both a label prediction and a confidence value. Then, the decision maker is supposed to use the confidence value to calibrate how much to trust the prediction. In this context, it has been often argued that the confidence value should correspond to a well calibrated estimate of the probability that the predicted label matches the ground truth label. However, multiple lines of empirical evidence suggest that decision makers have difficulties at developing a good sense on when to trust a prediction using these confidence values. In this paper, our goal is first to understand why and then investigate how to construct more useful confidence values. We first argue that, for a broad class of utility functions, there exists data distributions for which a rational decision maker is, in general, unlikely to discover the optimal decision policy using the above confidence values—an optimal decision maker would need to sometimes place more (less) trust on predictions with lower (higher) confidence values. However, we then show that, if the confidence values satisfy a natural alignment property with respect to the decision maker’s confidence on her own predictions, there always exists an optimal decision policy under which the level of trust the decision maker would need to place on predictions is monotone on the confidence values, facilitating its discoverability. Further, we show that multicalibration with respect to the decision maker’s confidence on her own prediction is a sufficient condition for alignment. Experiments on a real AI-assisted decision making scenario where a classifier provides decision support to human decision makers validate our theoretical results and suggest that alignment may lead to better decisions.</details>

### [Oral] Human-like Few-Shot Learning via Bayesian Reasoning over Natural Language

**Authors:** Kevin Ellis

**Oral Presentation:** We, Dec 13, 08:30 -- Oral 3A

**<details><summary>Abstract**</summary>

A core tension in models of concept learning is that the model must carefully balance the tractability of inference against the expressivity of the hypothesis class. Humans, however, can efficiently learn a broad range of concepts. We introduce a model of inductive learning that seeks to be human-like in that sense.It implements a Bayesian reasoning process where a language model first proposes candidate hypotheses expressed in natural language, which are then re-weighed by a prior and a likelihood.By estimating the prior from human data, we can predict human judgments on learning problems involving numbers and sets, spanning concepts that are generative, discriminative, propositional, and higher-order.</details>

### HyPoradise: An Open Baseline for Generative Speech Recognition with Large Language Models

**Authors:** CHEN CHEN, Yuchen Hu, Chao-Han Huck Yang, Sabato Marco Siniscalchi, Pin-Yu Chen, Eng-Siong Chng

**<details><summary>Abstract**</summary>

Advancements in deep neural networks have allowed automatic speech recognition (ASR) systems to attain human parity on several publicly available clean speech datasets. However, even state-of-the-art ASR systems experience performance degradation when confronted with adverse conditions, as a well-trained acoustic model is sensitive to variations in the speech domain, e.g., background noise. Intuitively, humans address this issue by relying on their linguistic knowledge: the meaning of ambiguous spoken terms is usually inferred from contextual cues thereby reducing the dependency on the auditory system. Inspired by this observation, we introduce the first open-source benchmark to utilize external large language models (LLMs) for ASR error correction, where N-best decoding hypotheses provide informative elements for true transcription prediction. This approach is a paradigm shift from the traditional language model rescoring strategy that can only select one candidate hypothesis as output transcription. The proposed benchmark contains a novel dataset, "HyPoradise" (HP), encompassing more than 316,000 pairs of N-best hypotheses and corresponding accurate transcriptions across prevalent speech domains. Given this dataset, we examine three types of error correction techniques based on LLMs with varying amounts of labeled hypotheses-transcription pairs, which gains significant word error rate (WER) reduction. Experimental evidence demonstrates the proposed technique achieves a breakthrough by surpassing the upper bound of traditional re-ranking based methods. More surprisingly, LLM with reasonable prompt design can even correct those tokens that are missing in N-best list. We make our results publicly accessible for reproducible pipelines with released pre-trained models, thus providing a new paradigm for ASR error correction with LLMs.</details>

### Hybrid Policy Optimization from Imperfect Demonstrations

**Authors:** Hanlin Yang, Chao Yu, peng sun, Siji Chen

**<details><summary>Abstract**</summary>

Exploration is one of the main challenges in Reinforcement Learning (RL), especially in environments with sparse rewards. Learning from Demonstrations (LfD) is a promising approach to solving this problem by leveraging expert demonstrations. However, expert demonstrations of high quality  are usually costly or even impossible to collect in real-world applications. In this work, we propose a novel RL algorithm called HYbrid Policy Optimization (HYPO), which uses a small number of imperfect demonstrations to accelerate an agent's online learning process. The key idea is to train an offline guider policy using imitation learning in order to instruct an online agent policy to explore efficiently. Through mutual update of the guider policy and the agent policy, the agent can leverage suboptimal demonstrations for efficient exploration while avoiding the conservative policy caused by imperfect demonstrations. Empirical results show that HYPO significantly outperforms several baselines in various challenging tasks, such as MuJoCo with sparse rewards, Google Research Football, and the AirSim drone simulation.</details>

### Hyperbolic VAE via Latent Gaussian Distributions

**Authors:** Seunghyuk Cho, Juyong Lee, Dongwoo Kim

**<details><summary>Abstract**</summary>

We propose a Gaussian manifold variational auto-encoder (GM-VAE) whose latent space consists of a set of Gaussian distributions. It is known that the set of the univariate Gaussian distributions with the Fisher information metric form a hyperbolic space, which we call a Gaussian manifold. To learn the VAE endowed with the Gaussian manifolds, we propose a pseudo-Gaussian manifold normal distribution based on the Kullback-Leibler divergence, a local approximation of the squared Fisher-Rao distance, to define a density over the latent space. We demonstrate the efficacy of GM-VAE on two different tasks: density estimation of image datasets and state representation learning for model-based reinforcement learning. GM-VAE outperforms the other variants of hyperbolic- and Euclidean-VAEs on density estimation tasks and shows competitive performance in model-based reinforcement learning. We observe that our model provides strong numerical stability, addressing a common limitation reported in previous hyperbolic-VAEs. The implementation is available at https://github.com/ml-postech/GM-VAE.</details>

### [Spotlight] ID and OOD Performance Are Sometimes Inversely Correlated on Real-world Datasets

**Authors:** Damien Teney, Yong Lin, Seong Joon Oh, Ehsan Abbasnejad

**<details><summary>Abstract**</summary>

Several studies have compared the in-distribution (ID) and out-of-distribution (OOD) performance of models in computer vision and NLP. They report a frequent positive correlation and some surprisingly never even observe an inverse correlation indicative of a necessary trade-off. The possibility of inverse patterns is important to determine whether ID performance can serve as a proxy for OOD generalization capabilities.This paper shows that inverse correlations between ID and OOD performance do happen with multiple real-world datasets, not only in artificial worst-case settings. We explain theoretically how these cases arise and how past studies missed them because of improper methodologies that examined a biased selection of models.Our observations lead to recommendations that contradict those found in much of the current literature.- High OOD performance sometimes requires trading off ID performance.- Focusing on ID performance alone may not lead to optimal OOD performance. It may produce diminishing (eventually negative) returns in OOD performance.- In these cases, studies on OOD generalization that use ID performance for model selection (a common recommended practice) will necessarily miss the best-performing models, making these studies blind to a whole range of phenomena.</details>

### IDRNet: Intervention-Driven Relation Network for Semantic Segmentation

**Authors:** Zhenchao Jin, Xiaowei Hu, Lingting Zhu, Luchuan Song, Li Yuan, Lequan Yu

**<details><summary>Abstract**</summary>

Co-occurrent visual patterns suggest that pixel relation modeling facilitates dense prediction tasks, which inspires the development of numerous context modeling paradigms, \emph{e.g.}, multi-scale-driven and similarity-driven context schemes. Despite the impressive results, these existing paradigms often suffer from inadequate or ineffective contextual information aggregation due to reliance on large amounts of predetermined priors. To alleviate the issues, we propose a novel \textbf{I}ntervention-\textbf{D}riven \textbf{R}elation \textbf{Net}work (\textbf{IDRNet}), which leverages a deletion diagnostics procedure to guide the modeling of contextual relations among different pixels. Specifically, we first group pixel-level representations into semantic-level representations with the guidance of pseudo labels and further improve the distinguishability of the grouped representations with a feature enhancement module. Next, a deletion diagnostics procedure is conducted to model relations of these semantic-level representations via perceiving the network outputs and the extracted relations are utilized to guide the semantic-level representations to interact with each other.  Finally, the interacted representations are utilized to augment original pixel-level representations for final predictions. Extensive experiments are conducted to validate the effectiveness of IDRNet quantitatively and qualitatively. Notably, our intervention-driven context scheme brings consistent performance improvements to state-of-the-art segmentation frameworks and achieves competitive results on popular benchmark datasets, including ADE20K, COCO-Stuff, PASCAL-Context, LIP, and Cityscapes.</details>

### IPMix: Label-Preserving Data Augmentation Method for Training Robust Classifiers

**Authors:** Zhenglin Huang, Xiaoan Bao, Na Zhang, Qingqi Zhang, Xiao Tu, Biao Wu, Xi Yang

**<details><summary>Abstract**</summary>

Data augmentation has been proven effective for training high-accuracy convolutional neural network classifiers by preventing overfitting. However, building deep neural networks in real-world scenarios requires not only high accuracy on clean data but also robustness when data distributions shift. While prior methods have proposed that there is a trade-off between accuracy and robustness, we propose IPMix, a simple data augmentation approach to improve robustness without hurting clean accuracy. IPMix integrates three levels of data augmentation (image-level, patch-level, and pixel-level) into a coherent and label-preserving technique to increase the diversity of training data with limited computational overhead. To further improve the robustness, IPMix introduces structural complexity at different levels to generate more diverse images and adopts the random mixing method for multi-scale information fusion. Experiments demonstrate that IPMix outperforms state-of-the-art corruption robustness on CIFAR-C and ImageNet-C. In addition, we show that IPMix also significantly improves the other safety measures, including robustness to adversarial perturbations, calibration, prediction consistency, and anomaly detection, achieving state-of-the-art or comparable results on several benchmarks, including ImageNet-R, ImageNet-A, and ImageNet-O.</details>

### ImageBrush: Learning Visual In-Context Instructions for Exemplar-Based Image Manipulation

**Authors:** ya sheng sun, Yifan Yang, Houwen Peng, Yifei Shen, Yuqing Yang, Han Hu, Lili Qiu, Hideki Koike

**<details><summary>Abstract**</summary>

While language-guided image manipulation has made remarkable progress, the challenge of how to instruct the manipulation process faithfully reflecting human intentions persists. An accurate and comprehensive description of a manipulation task using natural language is laborious and sometimes even impossible, primarily due to the inherent uncertainty and ambiguity present in linguistic expressions. Is it feasible to accomplish image manipulation without resorting to external cross-modal language information? If this possibility exists, the inherent modality gap would be effortlessly eliminated. In this paper, we propose a novel  manipulation methodology, dubbed ImageBrush, that learns visual instructions for more accurate image editing.Our key idea is to employ a pair of transformation images as visual instructions, which not only precisely captures human intention but also facilitates accessibility in real-world scenarios. Capturing visual instructions is particularly challenging because it involves extracting the underlying intentions solely from visual demonstrations and then applying this operation to a new image. To address this challenge, we formulate visual instruction learning as a diffusion-based inpainting problem, where the contextual information is fully exploited through an iterative process of generation. A visual prompting encoder is carefully devised to enhance the model's capacity in uncovering human intent behind the visual instructions. Extensive experiments show that our method generates engaging manipulation results conforming to the transformations entailed in demonstrations. Moreover, our model exhibits robust generalization capabilities on various downstream tasks such as pose transfer, image translation and video inpainting.</details>

### ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation

**Authors:** Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, Yuxiao Dong

**<details><summary>Abstract**</summary>

We present a comprehensive solution to learn and improve text-to-image models from human preference feedback.To begin with, we build ImageReward---the first general-purpose text-to-image human preference reward model---to effectively encode human preferences.Its training is based on our systematic annotation pipeline including rating and ranking, which collects 137k expert comparisons to date.In human evaluation, ImageReward outperforms existing scoring models and metrics, making it a promising automatic metric for evaluating text-to-image synthesis.On top of it, we propose Reward Feedback Learning (ReFL), a direct tuning algorithm to optimize diffusion models against a scorer.Both automatic and human evaluation support ReFL's advantages over compared methods.All code and datasets are provided at \url{https://github.com/THUDM/ImageReward}.</details>

### Imbalanced Mixed Linear Regression

**Authors:** Pini Zilber, Boaz Nadler

**<details><summary>Abstract**</summary>

We consider the problem of mixed linear regression (MLR), where each observed sample belongs to one of $K$ unknown linear models. In practical applications, the mixture of the $K$ models may be imbalanced with a significantly different number of samples from each model. Unfortunately, most MLR methods do not perform well in such settings. Motivated by this practical challenge, in this work we propose Mix-IRLS, a novel, simple and fast algorithm for MLR with excellent performance on both balanced and imbalanced mixtures.In contrast to popular approaches that recover the $K$ models simultaneously, Mix-IRLS does it sequentially using tools from robust regression. Empirically, beyond imbalanced mixtures, Mix-IRLS succeeds in a broad range of additional settings where other methods fail, including small sample sizes, presence of outliers, and an unknown number of models $K$. Furthermore, Mix-IRLS outperforms competing methods on several real-world datasets, in some cases by a large margin. We complement our empirical results by deriving a recovery guarantee for Mix-IRLS, which highlights its advantage on imbalanced mixtures.</details>

### Implicit Bias of (Stochastic) Gradient Descent for Rank-1 Linear Neural Network

**Authors:** Bochen Lyu, Zhanxing Zhu

**<details><summary>Abstract**</summary>

Studying the implicit bias of gradient descent (GD) and stochastic gradient descent (SGD) is critical to unveil the underlying mechanism of deep learning. Unfortunately, even for standard linear networks in regression setting, a comprehensive characterization of the implicit bias is still an open problem. This paper proposes to investigate a new proxy model of standard linear network,  rank-1 linear network, where each weight matrix is parameterized as a rank-1 form. For over-parameterized regression problem, we  precisely analyze the implicit bias of GD and SGD---by identifying a “potential” function such that GD converges to its minimizer constrained by zero training error (i.e., interpolation solution), and further characterizing the role of the noise introduced by  SGD in perturbing the form of this potential. Our results explicitly connect the depth of the network and the initialization with the implicit bias of GD and SGD. Furthermore, we emphasize a new implicit bias of SGD jointly induced by stochasticity and over-parameterization, which can reduce the dependence of the SGD's solution on the initialization. Our findings regarding the implicit bias are different from that of a recently popular model, the diagonal linear network. We highlight that the induced bias of our rank-1 model is more consistent with standard linear network while the diagonal one is not. This suggests that the proposed rank-1 linear network might be a plausible proxy for standard linear net.</details>

### Implicit Bias of Gradient Descent for Two-layer ReLU and Leaky ReLU Networks on Nearly-orthogonal Data

**Authors:** Yiwen Kou, Zixiang Chen, Quanquan Gu

**<details><summary>Abstract**</summary>

The implicit bias towards solutions with favorable properties is believed to be a key reason why neural networks trained by gradient-based optimization can generalize well. While the implicit bias of gradient flow has been widely studied for homogeneous neural networks (including ReLU and leaky ReLU networks), the implicit bias of gradient descent is currently only understood for smooth neural networks. Therefore, implicit bias in non-smooth neural networks trained by gradient descent remains an open question. In this paper, we aim to answer this question by studying the implicit bias of gradient descent for training two-layer fully connected (leaky) ReLU neural networks. We showed that when the training data are nearly-orthogonal, for leaky ReLU activation function, gradient descent will find a network with a stable rank that converges to $1$, whereas for ReLU activation function, gradient descent will find a neural network with a stable rank that is upper bounded by a constant. Additionally, we show that gradient descent will find a neural network such that all the training data points have the same normalized margin asymptotically. Experiments on both synthetic and real data backup our theoretical findings.</details>

### Importance-aware Co-teaching for Offline Model-based Optimization

**Authors:** Ye Yuan, Can Chen, Zixuan Liu, Willie Neiswanger, Xue (Steve) Liu

**<details><summary>Abstract**</summary>

Offline model-based optimization aims to find a design that maximizes a property of interest using only an offline dataset, with applications in robot, protein, and molecule design, among others. A prevalent approach is gradient ascent, where a proxy model is trained on the offline dataset and then used to optimize the design. This method suffers from an out-of-distribution issue, where the proxy is not accurate for unseen designs. To mitigate this issue, we explore using a pseudo-labeler to generate valuable data for fine-tuning the proxy. Specifically, we propose $\textit{\textbf{I}mportance-aware \textbf{C}o-\textbf{T}eaching for Offline Model-based Optimization}~(\textbf{ICT})$. This method maintains three symmetric proxies with their mean ensemble as the final proxy, and comprises two steps. The first step is $\textit{pseudo-label-driven co-teaching}$. In this step, one proxy is iteratively selected as the pseudo-labeler for designs near the current optimization point, generating pseudo-labeled data.  Subsequently, a co-teaching process identifies small-loss samples as valuable data and exchanges them between the other two proxies for fine-tuning, promoting knowledge transfer.  This procedure is repeated three times, with a different proxy chosen as the pseudo-labeler each time, ultimately enhancing the ensemble performance.To further improve accuracy of pseudo-labels, we perform a secondary step of $\textit{meta-learning-based sample reweighting}$,which assigns importance weights to samples in the pseudo-labeled dataset and updates them via meta-learning. ICT achieves state-of-the-art results across multiple design-bench tasks, achieving the best mean rank $3.1$ and median rank $2$ among $15$ methods.Our source code can be accessed here.</details>

### Improved Bayes Risk Can Yield Reduced Social Welfare Under Competition

**Authors:** Meena Jagadeesan, Michael Jordan, Jacob Steinhardt, Nika Haghtalab

**<details><summary>Abstract**</summary>

As the scale of machine learning models increases, trends such as scaling laws anticipate consistent downstream improvements in predictive accuracy. However, these trends take the perspective of a single model-provider in isolation, while in reality providers often compete with each other for users. In this work, we demonstrate that competition can fundamentally alter the behavior of these scaling trends, even causing overall predictive accuracy across users to be non-monotonic or decreasing with scale. We define a model of competition for classification tasks, and use data representations as a lens for studying the impact of increases in scale. We find many settings where improving data representation quality (as measured by Bayes risk) decreases the overall predictive accuracy across users (i.e., social welfare) for a marketplace of competing model-providers. Our examples range from closed-form formulas in simple settings to simulations with pretrained representations on CIFAR-10. At a conceptual level, our work suggests that favorable scaling trends for individual model-providers need not translate to downstream improvements in  social welfare in marketplaces with  multiple model providers.</details>

### Improvements on Uncertainty Quantification for Node Classification via Distance Based Regularization

**Authors:** Russell Hart, Linlin Yu, Yifei Lou, Feng Chen

**<details><summary>Abstract**</summary>

Deep neural networks have achieved significant success in the last decades, but they are not well-calibrated and often produce unreliable predictions. A large number of literature relies on uncertainty quantification to evaluate the reliability of a learning model, which is particularly important for applications of out-of-distribution (OOD) detection and misclassification detection. We are interested in uncertainty quantification for interdependent node-level classification. We start our analysis based on graph posterior networks (GPNs) that optimize the uncertainty cross-entropy (UCE)-based loss function. We describe the theoretical limitations of the widely-used UCE loss. To alleviate the identified drawbacks, we propose a distance-based regularization that encourages clustered OOD nodes to remain clustered in the latent space. We conduct extensive comparison experiments on eight standard datasets and demonstrate that the proposed regularization outperforms the state-of-the-art in both OOD detection and misclassification detection.</details>

### Improving Few-Shot Generalization by Exploring and Exploiting Auxiliary Data

**Authors:** Alon Albalak, Colin Raffel, William Yang Wang

**<details><summary>Abstract**</summary>

Few-shot learning is valuable in many real-world applications, but learning a generalizable model without overfitting to the few labeled datapoints is challenging.In this work, we focus on Few-shot Learning with Auxiliary Data (FLAD), a training paradigm that assumes access to auxiliary data during few-shot learning in hopes of improving generalization.Previous works have proposed automated methods for mixing auxiliary and target data, but these methods typically scale linearly (or worse) with the number of auxiliary datasets, limiting their practicality.In this work we relate FLAD to the explore-exploit dilemma that is central to the multi-armed bandit setting and derive algorithms whose computational complexity is independent of the number of auxiliary datasets, allowing us to scale to 100x more auxiliary datasets than prior methods.We propose two algorithms -- EXP3-FLAD and UCB1-FLAD -- and compare them with prior FLAD methods that either explore or exploit, finding that the combination of exploration and exploitation is crucial.Through extensive experimentation we find that our methods outperform all pre-existing FLAD methods by 4% and lead to the first 3 billion parameter language models that outperform the 175 billion parameter GPT-3.Overall, our work suggests that the discovery of better, more efficient mixing strategies for FLAD may provide a viable path towards substantially improving generalization in few-shot learning.</details>

### Improving neural network representations using human similarity judgments

**Authors:** Lukas Muttenthaler, Lorenz Linhardt, Jonas Dippel, Robert Vandermeulen, Katherine Hermann, Andrew Lampinen, Simon Kornblith

**<details><summary>Abstract**</summary>

Deep neural networks have reached human-level performance on many computer vision tasks. However, the objectives used to train these networks enforce only that similar images are embedded at similar locations in the representation space, and do not directly constrain the global structure of the resulting space. Here, we explore the impact of supervising this global structure by linearly aligning it with human similarity judgments. We find that a naive approach leads to large changes in local representational structure that harm downstream performance. Thus, we propose a novel method that aligns the global structure of representations while preserving their local structure. This global-local transform considerably improves accuracy across a variety of few-shot learning and anomaly detection tasks. Our results indicate that human visual representations are globally organized in a way that facilitates learning from few examples, and incorporating this global structure into neural network representations improves performance on downstream tasks.</details>

### Improving the Privacy and Practicality of Objective Perturbation for Differentially Private Linear Learners

**Authors:** Rachel Redberg, Antti Koskela, Yu-Xiang Wang

**<details><summary>Abstract**</summary>

In the arena of privacy-preserving machine learning, differentially private stochastic gradient descent (DP-SGD) has outstripped the objective perturbation mechanism in popularity and interest. Though unrivaled in versatility, DP-SGD requires a non-trivial privacy overhead (for privately tuning the model’s hyperparameters) and a computational complexity which might be extravagant for simple models such as linear and logistic regression. This paper revamps the objective perturbation mechanism with tighter privacy analyses and new computational tools that boost it to perform competitively with DP-SGD on unconstrained convex generalized linear problems.</details>

### Information Geometry of the Retinal Representation Manifold

**Authors:** Xuehao Ding, Dongsoo Lee, Joshua Melander, George Sivulka, Surya Ganguli, Stephen Baccus

**<details><summary>Abstract**</summary>

The ability for the brain to discriminate among visual stimuli is constrained by their retinal representations. Previous studies of visual discriminability have been limited to either low-dimensional artificial stimuli or pure theoretical considerations without a realistic encoding model. Here we propose a novel framework for understanding stimulus discriminability achieved by retinal representations of naturalistic stimuli with the method of information geometry. To model the joint probability distribution of neural responses conditioned on the stimulus, we created a stochastic encoding model of a population of salamander retinal ganglion cells based on a three-layer convolutional neural network model. This model not only accurately captured the mean response to natural scenes but also a variety of second-order statistics. With the model and the proposed theory, we computed the Fisher information metric over stimuli to study the most discriminable stimulus directions. We found that the most discriminable stimulus varied substantially across stimuli, allowing an examination of the relationship between the most discriminable stimulus and the current stimulus. By examining responses generated by the most discriminable stimuli we further found that the most discriminative response mode is often aligned with the most stochastic mode. This finding carries the important implication that under natural scenes, retinal noise correlations are information-limiting rather than increasing information transmission as has been previously speculated. We additionally observed that sensitivity saturates less in the population than for single cells and that as a function of firing rate, Fisher information varies less than sensitivity. We conclude that under natural scenes, population coding benefits from complementary coding and helps to equalize the information carried by different firing rates, which may facilitate decoding of the stimulus under principles of information maximization.</details>

### Information-guided Planning: An Online Approach for Partially Observable Problems

**Authors:** Amokh Varma, Yehia Elkhatib, Leandro Soriano Marcolino

**<details><summary>Abstract**</summary>

This paper presents IB-POMCP, a novel algorithm for online planning under partial observability. Our approach enhances the decision-making process by using estimations of the world belief's entropy to guide a tree search process and surpass the limitations of planning in scenarios with sparse reward configurations. By performing what we denominate as an *information-guided planning process*, the algorithm, which incorporates a novel I-UCB function, shows significant improvements in reward and reasoning time compared to state-of-the-art baselines in several benchmark scenarios, along with theoretical convergence guarantees.</details>

### Initialization Matters: Privacy-Utility Analysis of Overparameterized Neural Networks

**Authors:** Jiayuan Ye, Zhenyu Zhu, Fanghui Liu, Reza Shokri, Volkan Cevher

**<details><summary>Abstract**</summary>

We analytically investigate how over-parameterization of models in randomized machine learning algorithms impacts the information leakage about their training data. Specifically, we prove a privacy bound for the KL divergence between model distributions on worst-case neighboring datasets, and explore its dependence on the initialization, width, and depth of fully connected neural networks. We find that this KL privacy bound is largely determined by the expected squared gradient norm relative to model parameters during training. Notably, for the special setting of linearized network, our analysis indicates that the squared gradient norm (and therefore the escalation of privacy loss) is tied directly to the per-layer variance of the initialization distribution. By using this analysis, we demonstrate that privacy bound improves with increasing depth under certain initializations (LeCun and Xavier), while degrades with increasing depth under other initializations (He and NTK). Our work reveals a complex interplay between privacy and depth that depends on the chosen initialization distribution. We further prove excess empirical risk bounds under a fixed KL privacy budget, and show that the interplay between privacy utility trade-off and depth is similarly affected by the initialization.</details>

### Interpretable Graph Networks Formulate Universal Algebra Conjectures

**Authors:** Francesco Giannini, Stefano Fioravanti, Oguzhan Keskin, Alisia Lupidi, Lucie Charlotte Magister, Pietro Lió, Pietro Barbiero

**<details><summary>Abstract**</summary>

The rise of Artificial Intelligence (AI) recently empowered researchers to investigate hard mathematical problems which eluded traditional approaches for decades. Yet, the use of AI in Universal Algebra (UA)---one of the fields laying the foundations of modern mathematics---is still completely unexplored. This work proposes the first use of AI to investigate UA's conjectures with an equivalent equational and topological characterization. While topological representations would enable the analysis of such properties using graph neural networks, the limited transparency and brittle explainability of these models hinder their straightforward use to empirically validate existing conjectures or to formulate new ones. To bridge these gaps, we propose a general algorithm generating AI-ready datasets based on UA's conjectures, and introduce a novel neural layer to build fully interpretable graph networks. The results of our experiments demonstrate that interpretable graph networks: (i) enhance interpretability without sacrificing task accuracy, (ii) strongly generalize when predicting universal algebra's properties, (iii) generate simple explanations that empirically validate existing conjectures, and (iv) identify subgraphs suggesting the formulation of novel conjectures.</details>

### Interpretable and Explainable Logical Policies via Neurally Guided Symbolic Abstraction

**Authors:** Quentin Delfosse, Hikaru Shindo, Devendra Dhami, Kristian Kersting

**<details><summary>Abstract**</summary>

The limited priors required by neural networks make them the dominating choice to encode and learn policies using reinforcement learning (RL). However, they are also black-boxes, making it hard to understand the agent's behavior, especially when working on the image level. Therefore, neuro-symbolic RL aims at creating policies that are interpretable in the first place.Unfortunately, interpretability is not explainability. To achieve both, we introduce Neurally gUided Differentiable loGic policiEs (NUDGE). NUDGE exploits trained neural network-based agents to guide the search of candidate-weighted logic rules, then uses differentiable logic to train the logic agents. Our experimental evaluation demonstrates that NUDGE agents can induce interpretable and explainable policies while outperforming purely neural ones and showing good flexibility to environments of different initial states and problem sizes.</details>

### Intra-Modal Proxy Learning for Zero-Shot Visual Categorization with CLIP

**Authors:** Qi Qian, Yuanhong Xu, Juhua Hu

**<details><summary>Abstract**</summary>

Vision-language pre-training methods, e.g., CLIP, demonstrate an impressive zero-shot performance on visual categorizations with the class proxy from the text embedding of the class name. However, the modality gap between the text and vision space can result in a sub-optimal performance. We theoretically show that the gap cannot be reduced sufficiently by minimizing the contrastive loss in CLIP and the optimal proxy for vision tasks may reside only in the vision space. Therefore, given unlabeled target vision data, we propose to learn the vision proxy directly with the help from the text proxy for zero-shot transfer. Moreover, according to our theoretical analysis, strategies are developed to further refine the pseudo label obtained by the text proxy to facilitate the intra-modal proxy learning (InMaP) for vision. Experiments on extensive downstream tasks confirm the effectiveness and efficiency of our proposal. Concretely, InMaP can obtain the vision proxy within one minute on a single GPU while improving the zero-shot accuracy from $77.02\%$ to $80.21\%$ on ImageNet with ViT-L/14@336 pre-trained by CLIP.</details>

### Intriguing Properties of Quantization at Scale

**Authors:** Arash Ahmadian, Saurabh Dash, Hongyu Chen, Bharat Venkitesh, Zhen Stephen Gou, Phil Blunsom, Ahmet Üstün, Sara Hooker

**<details><summary>Abstract**</summary>

Emergent properties have been widely adopted as a term to describe behavior not present in smaller models but observed in larger models  (Wei et al., 2022a). Recent work suggests that the trade-off incurred by quantization is also an emergent property, with sharp drops in performance in models over 6B parameters. In this work, we ask _are quantization cliffs in performance solely a factor of scale?_ Against a backdrop of increased research focus on why certain emergent properties surface at scale, this work provides a useful counter-example. We posit that it is possible to optimize for a quantization friendly training recipe that suppresses large activation magnitude outliers. Here, we find that outlier dimensions are not an inherent product of scale, but rather sensitive to the optimization conditions present during pre-training. This both opens up directions for more efficient quantization, and poses the question of whether other emergent properties are inherent or can be altered and conditioned by optimization and architecture design choices. We successfully quantize models ranging in size from 410M to 52B with minimal degradation in performance.</details>

### [Spotlight] Invariant Learning via Probability of Sufficient and Necessary Causes

**Authors:** Mengyue Yang, Yonggang Zhang, Zhen Fang, Yali Du, Furui Liu, Jean-Francois Ton, Jianhong Wang, Jun Wang

**<details><summary>Abstract**</summary>

Out-of-distribution (OOD) generalization is indispensable for learning models in the wild, where testing distribution typically unknown and different from the training. Recent methods derived from causality have shown great potential in achieving OOD generalization. However, existing methods mainly focus on the invariance property of causes, while largely overlooking the property of sufficiency and necessity conditions. Namely, a necessary but insufficient cause (feature) is invariant to distribution shift, yet it may not have required accuracy. By contrast, a sufficient yet unnecessary cause (feature) tends to fit specific data well but may have a risk of adapting to a new domain. To capture the information of sufficient and necessary causes, we employ a classical concept, the probability of sufficiency and necessary causes (PNS), which indicates the probability of whether one is the necessary and sufficient cause. To associate PNS with OOD generalization, we propose PNS risk and formulate an algorithm to learn representation with a high PNS value. We theoretically analyze and prove the generalizability of the PNS risk. Experiments on both synthetic and real-world benchmarks demonstrate the effectiveness of the proposed method. The detailed implementation can be found at the GitHub repository: https://github.com/ymy4323460/CaSN.</details>

### Is Distance Matrix Enough for Geometric Deep Learning?

**Authors:** Zian Li, Xiyuan Wang, Yinan Huang, Muhan Zhang

**<details><summary>Abstract**</summary>

Graph Neural Networks (GNNs) are often used for tasks involving the 3D geometry of a given graph, such as molecular dynamics simulation. While incorporating Euclidean distance into Message Passing Neural Networks (referred to as Vanilla DisGNN) is a straightforward way to learn the geometry, it has been demonstrated that Vanilla DisGNN is geometrically incomplete. In this work, we first construct families of novel and symmetric geometric graphs that Vanilla DisGNN cannot distinguish even when considering all-pair distances, which greatly expands the existing counterexample families. Our counterexamples show the inherent limitation of Vanilla DisGNN to capture symmetric geometric structures. We then propose $k$-DisGNNs, which can effectively exploit the rich geometry contained in the distance matrix. We demonstrate the high expressive power of $k$-DisGNNs from three perspectives: 1. They can learn high-order geometric information that cannot be captured by Vanilla DisGNN. 2. They can unify some existing well-designed geometric models. 3. They are universal function approximators from geometric graphs to scalars (when $k\geq 2$) and vectors (when $k\geq 3$). Most importantly, we establish a connection between geometric deep learning (GDL) and traditional graph representation learning (GRL), showing that those highly expressive GNN models originally designed for GRL can also be applied to GDL with impressive performance, and that existing complicated, equivariant models are not the only solution. Experiments verify our theory. Our $k$-DisGNNs achieve many new state-of-the-art results on MD17.</details>

### Isometric Quotient Variational Auto-Encoders for Structure-Preserving Representation Learning

**Authors:** In Huh, changwook jeong, Jae Myung Choe, YOUNGGU KIM, Daesin Kim

**<details><summary>Abstract**</summary>

We study structure-preserving low-dimensional representation of a data manifold embedded in a high-dimensional observation space based on variational auto-encoders (VAEs). We approach this by decomposing the data manifold $\mathcal{M}$ as $\mathcal{M} = \mathcal{M} / G \times G$, where $G$ and $\mathcal{M} / G$ are a group of symmetry transformations and a quotient space of $\mathcal{M}$ up to $G$, respectively. From this perspective, we define the structure-preserving representation of such a manifold as a latent space $\mathcal{Z}$ which is isometrically isomorphic (i.e., distance-preserving) to the quotient space $\mathcal{M} / G$ rather $\mathcal{M}$ (i.e., symmetry-preserving). To this end, we propose a novel auto-encoding framework, named isometric quotient VAEs (IQVAEs), that can extract the quotient space from observations and learn the Riemannian isometry of the extracted quotient in an unsupervised manner. Empirical proof-of-concept experiments reveal that the proposed method can find a meaningful representation of the learned data and outperform other competitors for downstream tasks.</details>

### Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels

**Authors:** Zifu Wang, Xuefei Ning, Matthew Blaschko

**<details><summary>Abstract**</summary>

Intersection over Union (IoU) losses are surrogates that directly optimize the Jaccard index. Leveraging IoU losses as part of the loss function have demonstrated superior performance in semantic segmentation tasks compared to optimizing pixel-wise losses such as the cross-entropy loss alone. However, we identify a lack of flexibility in these losses to support vital training techniques like label smoothing, knowledge distillation, and semi-supervised learning, mainly due to their inability to process soft labels. To address this, we introduce Jaccard Metric Losses (JMLs), which are identical to the soft Jaccard loss in standard settings with hard labels but are fully compatible with soft labels. We apply JMLs to three prominent use cases of soft labels: label smoothing, knowledge distillation and semi-supervised learning, and demonstrate their potential to enhance model accuracy and calibration. Our experiments show consistent improvements over the cross-entropy loss across 4 semantic segmentation datasets (Cityscapes, PASCAL VOC, ADE20K, DeepGlobe Land) and 13 architectures, including classic CNNs and recent vision transformers. Remarkably, our straightforward approach significantly outperforms state-of-the-art knowledge distillation and semi-supervised learning methods. The code is available at \href{https://github.com/zifuwanggg/JDTLosses}{https://github.com/zifuwanggg/JDTLosses}.</details>

### Jigsaw: Learning to Assemble Multiple Fractured Objects

**Authors:** Jiaxin Lu, Yifan Sun, Qixing Huang

**<details><summary>Abstract**</summary>

Automated assembly of 3D fractures is essential in orthopedics, archaeology, and our daily life. This paper presents Jigsaw, a novel framework for assembling physically broken 3D objects from multiple pieces. Our approach leverages hierarchical features of global and local geometry to match and align the fracture surfaces. Our framework consists of four components: (1) front-end point feature extractor with attention layers, (2) surface segmentation to separate fracture and original parts, (3) multi-parts matching to find correspondences among fracture surface points, and (4) robust global alignment to recover the global poses of the pieces. We show how to jointly learn segmentation and matching and seamlessly integrate feature matching and rigidity constraints. We evaluate Jigsaw on the Breaking Bad dataset and achieve superior performance compared to state-of-the-art methods. Our method also generalizes well to diverse fracture modes, objects, and unseen instances. To the best of our knowledge, this is the first learning-based method designed specifically for 3D fracture assembly over multiple pieces. Our code is available at https://jiaxin-lu.github.io/Jigsaw/.</details>

### Joint Attribute and Model Generalization Learning for Privacy-Preserving Action Recognition

**Authors:** Duo Peng, Li Xu, Qiuhong Ke, Ping Hu, Jun Liu

**<details><summary>Abstract**</summary>

Privacy-Preserving Action Recognition (PPAR) aims to transform raw videos into anonymous ones to prevent privacy leakage while maintaining action clues, which is an increasingly important problem in intelligent vision applications. Despite recent efforts in this task, it is still challenging to deal with novel privacy attributes and novel privacy attack models that are unavailable during the training phase. In this paper, from the perspective of meta-learning (learning to learn), we propose a novel Meta Privacy-Preserving Action Recognition (MPPAR) framework to improve both generalization abilities above (i.e., generalize to *novel privacy attributes* and *novel privacy attack models*) in a unified manner. Concretely, we simulate train/test task shifts by constructing disjoint support/query sets w.r.t. privacy attributes or attack models. Then, a virtual training and testing scheme is applied based on support/query sets to provide feedback to optimize the model's learning toward better generalization. Extensive experiments demonstrate the effectiveness and generalization of the proposed framework compared to state-of-the-arts.</details>

### Joint Data-Task Generation for Auxiliary Learning

**Authors:** Hong Chen, Xin Wang, Yuwei Zhou, Yijian Qin, Chaoyu Guan, Wenwu Zhu

**<details><summary>Abstract**</summary>

Current auxiliary learning methods mainly adopt the methodology of reweighing losses for the manually collected auxiliary data and tasks. However, these methods heavily rely on domain knowledge during data collection, which may be hardly available in reality. Therefore, current methods will become less effective and even do harm to the primary task when unhelpful auxiliary data and tasks are employed. To tackle the problem, we propose a joint data-task generation framework for auxiliary learning (DTG-AuxL), which can bring benefits to the primary task by generating the new auxiliary data and task in a joint manner. The proposed DTG-AuxL framework contains a joint generator and a bi-level optimization strategy. Specifically, the joint generator contains a feature generator and a label generator, which are designed to be applicable  and expressive for various auxiliary learning scenarios. The bi-level optimization strategy optimizes the joint generator and the task learning model, where the joint generator is effectively optimized in the upper level via the implicit gradient from the primary loss and the explicit gradient of our proposed instance regularization, while the task learning model is optimized in the lower level by the generated data and task. Extensive experiments show that our proposed DTG-AuxL framework consistently outperforms existing methods in various auxiliary learning scenarios, particularly when the manually collected auxiliary data and tasks are unhelpful.</details>

### KAKURENBO: Adaptively Hiding Samples in Deep Neural Network Training

**Authors:** Truong Thao Nguyen, Balazs Gerofi, Edgar Josafat Martinez-Noriega, François Trahay, Mohamed Wahib

**<details><summary>Abstract**</summary>

This paper proposes a method for hiding the least-important samples during the training of deep neural networks to increase efficiency, i.e., to reduce the cost of training. Using information about the loss and prediction confidence during training, we adaptively find samples to exclude in a given epoch based on their contribution to the overall learning process, without significantly degrading accuracy. We explore the converge properties when accounting for the reduction in the number of SGD updates.  Empirical results on various large-scale datasets and models used directly in image classification and segmentation show that while the with-replacement importance sampling algorithm performs poorly on large datasets,  our method can reduce total training time by up to 22\% impacting accuracy only by 0.4\% compared to the baseline.</details>

### [Spotlight] Kernel Quadrature with Randomly Pivoted Cholesky

**Authors:** Ethan Epperly, Elvira Moreno

**<details><summary>Abstract**</summary>

This paper presents new quadrature rules for functions in a reproducing kernel Hilbert space using nodes drawn by a sampling algorithm known as randomly pivoted Cholesky. The resulting computational procedure compares favorably to previous kernel quadrature methods, which either achieve low accuracy or require solving a computationally challenging sampling problem. Theoretical and numerical results show that randomly pivoted Cholesky is fast and achieves comparable quadrature error rates to more computationally expensive quadrature schemes based on continuous volume sampling, thinning, and recombination. Randomly pivoted Cholesky is easily adapted to complicated geometries with arbitrary kernels, unlocking new potential for kernel quadrature.</details>

### [Spotlight] Kronecker-Factored Approximate Curvature for Modern Neural Network Architectures

**Authors:** Runa Eschenhagen, Alexander Immer, Richard Turner, Frank Schneider, Philipp Hennig

**<details><summary>Abstract**</summary>

The core components of many modern neural network architectures, such as transformers, convolutional, or graph neural networks, can be expressed as linear layers with *weight-sharing*. Kronecker-Factored Approximate Curvature (K-FAC), a second-order optimisation method, has shown promise to speed up neural network training and thereby reduce computational costs. However, there is currently no framework to apply it to generic architectures, specifically ones with linear weight-sharing layers. In this work, we identify two different settings of linear weight-sharing layers which motivate two flavours of K-FAC -- *expand* and *reduce*. We show that they are exact for deep linear networks with weight-sharing in their respective setting. Notably, K-FAC-reduce is generally faster than K-FAC-expand, which we leverage to speed up automatic hyperparameter selection via optimising the marginal likelihood for a Wide ResNet. Finally, we observe little difference between these two K-FAC variations when using them to train both a graph neural network and a vision transformer. However, both variations are able to reach a fixed validation metric target in $50$-$75$\% of the number of steps of a first-order reference run, which translates into a comparable improvement in wall-clock time. This highlights the potential of applying K-FAC to modern neural network architectures.</details>

### LIMA: Less Is More for Alignment

**Authors:** Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, LILI YU, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, Omer Levy

**<details><summary>Abstract**</summary>

Large language models are trained in two stages: (1) unsupervised pretraining from raw text, to learn general-purpose representations, and (2) large scale instruction tuning and reinforcement learning, to better align to end tasks and user preferences. We measure the relative importance of these two stages by training LIMA, a 65B parameter LLaMa language model fine-tuned with the standard supervised loss on only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling.LIMA demonstrates remarkably strong performance, learning to follow specific response formats from only a handful of examples in the training data, including complex queries that range from planning trip itineraries to speculating about alternate history.Moreover, the model tends to generalize well to unseen tasks that did not appear in the training data.In a controlled human study, responses from LIMA are either equivalent or strictly preferred to GPT-4 in 43\% of cases; this statistic is as high as 58\% when compared to Bard and 65\% versus DaVinci003, which was trained with human feedback.Taken together, these results strongly suggest that almost all knowledge in large language models is learned during pretraining, and only limited instruction tuning data is necessary to teach models to produce high quality output.</details>

### LLM-Pruner: On the Structural Pruning of Large Language Models

**Authors:** Xinyin Ma, Gongfan Fang, Xinchao Wang

**<details><summary>Abstract**</summary>

Large language models (LLMs) have shown remarkable capabilities in language understanding and generation. However, such impressive capability typically comes with a substantial model size, which presents significant challenges in both the deployment, inference, and training stages. With LLM being a general-purpose task solver, we explore its compression in a task-agnostic manner, which aims to preserve the multi-task solving and language generation ability of the original LLM. One challenge to achieving this is the enormous size of the training corpus of LLM, which makes both data transfer and model post-training over-burdensome. Thus, we tackle the compression of LLMs within the bound of two constraints: being task-agnostic and minimizing the reliance on the original training dataset. Our method, named LLM-pruner, adopts structural pruning that selectively removes non-critical coupled structures based on gradient information, maximally preserving the majority of the LLM's functionality. To this end, the performance of pruned models can be efficiently recovered through tuning techniques, LoRA, in merely 3 hours, requiring only 50K data. We validate the LLM-Pruner on three LLMs, including LLaMA, Vicuna, and ChatGLM, and demonstrate that the compressed models still exhibit satisfactory capabilities in zero-shot classification and generation. The code will be made public.</details>

### LLMScore: Unveiling the Power of Large Language Models in Text-to-Image Synthesis Evaluation

**Authors:** Yujie Lu, Xianjun Yang, Xiujun Li, Xin Eric Wang, William Yang Wang

**<details><summary>Abstract**</summary>

Existing automatic evaluation on text-to-image synthesis can only provide an image-text matching score, without considering the object-level compositionality, which results in poor correlation with human judgments. In this work, we propose LLMScore, a new framework that offers evaluation scores with multi-granularity compositionality. LLMScore leverages the large language models (LLMs) to evaluate text-to-image models. Initially, it transforms the image into image-level and object-level visual descriptions. Then an evaluation instruction is fed into the LLMs to measure the alignment between the synthesized image and the text, ultimately generating a score accompanied by a rationale. Our substantial analysis reveals the highest correlation of LLMScore with human judgments on a wide range of datasets (Attribute Binding Contrast, Concept Conjunction, MSCOCO, DrawBench, PaintSkills). Notably, our LLMScore achieves Kendall's tau correlation with human evaluations that is 58.8% and 31.2% higher than the commonly-used text-image matching metrics CLIP and BLIP, respectively.</details>

### Label Robust and Differentially Private Linear Regression: Computational and Statistical Efficiency

**Authors:** Xiyang Liu, Prateek Jain, Weihao Kong, Sewoong Oh, Arun Suggala

**<details><summary>Abstract**</summary>

We study the canonical problem of linear regression under $(\varepsilon,\delta)$-differential privacy when the datapoints are sampled i.i.d.~from a distribution and a fraction of response variables are adversarially corrupted. We provide the first provably efficient -- both computationally and statistically -- method for this problem, assuming standard assumptions on the data distribution. Our algorithm is a variant of the popular differentially private stochastic gradient descent (DP-SGD) algorithm with two key innovations: a full-batch gradient descent to improve sample complexity and a novel adaptive clipping to guarantee robustness. Our method requires only linear time in input size, and still matches the information theoretical optimal sample complexity up to a data distribution dependent condition number factor.  Interestingly, the same algorithm, when applied to a setting where there is no adversarial corruption, still improves upon the existing state-of-the-art and achieves a near optimal sample complexity.</details>

### Label-Only Model Inversion Attacks via Knowledge Transfer

**Authors:** Bao-Ngoc Nguyen, Keshigeyan Chandrasegaran, Milad Abdollahzadeh, Ngai-Man (Man) Cheung

**<details><summary>Abstract**</summary>

In a model inversion (MI) attack, an adversary abuses access to a machine learning (ML) model to infer and reconstruct private training data. Remarkable progress has been made in the white-box and black-box setups, where the adversary has access to the complete model or the model's soft output respectively. However, there is very limited study in the most challenging but practically important setup: Label-only MI attacks, where the adversary only has access to the model's predicted label (hard label) without confidence scores nor any other model information. In this work, we propose LOKT, a novel approach for label-only MI attacks. Our idea is based on transfer of knowledge from the opaque target model to surrogate models. Subsequently, using these surrogate models, our approach can harness advanced white-box attacks. We propose knowledge transfer based on generative modelling, and introduce a new model, Target model-assisted ACGAN (T-ACGAN), for effective knowledge transfer. Our method casts the challenging label-only MI into the more tractable white-box setup. We provide analysis to support that surrogate models based on our approach serve as effective proxies for the target model for MI. Our experiments show that our method significantly outperforms existing SOTA Label-only MI attack by more than 15% across all MI benchmarks. Furthermore, our method compares favorably in terms of query budget. Our study highlights rising privacy threats for ML models even when minimal information (i.e., hard labels) is exposed. Our study highlights rising privacy threats for ML models even when minimal information (i.e., hard labels) is exposed. Our code, demo, models and reconstructed data are available at our project page:https://ngoc-nguyen-0.github.io/lokt/</details>

### Language Models can Solve Computer Tasks

**Authors:** Geunwoo Kim, Pierre Baldi, Stephen McAleer

**<details><summary>Abstract**</summary>

Agents capable of carrying out general tasks on a computer can improve efficiency and productivity by automating repetitive tasks and assisting in complex problem-solving. Ideally, such agents should be able to solve new computer tasks presented to them through natural language commands. However, previous approaches to this problem require large amounts of expert demonstrations and task-specific reward functions, both of which are impractical for new tasks. In this work, we show that a pre-trained large language model (LLM) agent can execute computer tasks guided by natural language using a simple prompting scheme where the agent 	extbf{R}ecursively 	extbf{C}riticizes and 	extbf{I}mproves its output (RCI). The RCI approach significantly outperforms existing LLM methods for automating computer tasks and surpasses supervised learning (SL) and reinforcement learning (RL) approaches on the MiniWoB++ benchmark. We compare multiple LLMs and find that RCI with the InstructGPT-3+RLHF LLM is state-of-the-art on MiniWoB++, using only a handful of demonstrations per task rather than tens of thousands, and without a task-specific reward function. Furthermore, we demonstrate RCI prompting's effectiveness in enhancing LLMs' reasoning abilities on a suite of natural language reasoning tasks, outperforming chain of thought (CoT) prompting with external feedback. We find that RCI combined with CoT performs better than either separately. Our code can be found here: https://github.com/posgnu/rci-agent.</details>

### Large Language Models can Implement Policy Iteration

**Authors:** Ethan Brooks, Logan Walls, Richard L Lewis, Satinder Singh

**<details><summary>Abstract**</summary>

In this work, we demonstrate a method for implementing policy iteration using a large language model. While the application of foundation models to RL has received considerable attention, most approaches rely on either (1) the curation of expert demonstrations (either through manual design or task-specific pretraining) or (2) adaptation to the task of interest using gradient methods (either fine-tuning or training of adapter layers). Both of these techniques have drawbacks. Collecting demonstrations is labor-intensive, and algorithms that rely on them do not outperform the experts from which the demonstrations were derived. All gradient techniques are inherently slow, sacrificing the “few-shot” quality that makes in-context learning attractive to begin with. Our method demonstrates that a large language model can be used to implement policy iteration using the machinery of in-context learning, enabling it to learn to perform RL tasks without expert demonstrations or gradients. Our approach iteratively updates the contents of the prompt from which it derives its policy through trial-and-error interaction with an RL environment. In order to eliminate the role of in-weights learning (on which approaches like Decision Transformer rely heavily), we demonstrate our method using Codex (M. Chen et al. 2021b), a language model with no prior knowledge of the domains on which we evaluate it.</details>

### Latent Diffusion for Language Generation

**Authors:** Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, Kilian Weinberger

**<details><summary>Abstract**</summary>

Diffusion models have achieved great success in modeling continuous data modalities such as images, audio, and video, but have seen limited use in discrete domains such as language. Recent attempts to adapt diffusion to language have presented diffusion as an alternative to existing pretrained language models. We view diffusion and existing language models as complementary. We demonstrate that encoder-decoder language models can be utilized to efficiently learn high-quality language autoencoders. We then demonstrate that continuous diffusion models can be learned in the latent space of the language autoencoder, enabling us to sample continuous latent representations that can be decoded into natural language with the pretrained decoder. We validate the effectiveness of our approach for unconditional, class-conditional, and sequence-to-sequence language generation. We demonstrate across multiple diverse data sets that our latent language diffusion models are significantly more effective than previous diffusion language models. Our code is available at \url{https://github.com/justinlovelace/latent-diffusion-for-language}.</details>

### Latent Field Discovery in Interacting Dynamical Systems with Neural Fields

**Authors:** Miltiadis (Miltos) Kofinas, Erik Bekkers, Naveen Nagaraja, Efstratios Gavves

**<details><summary>Abstract**</summary>

Systems of interacting objects often evolve under the influence of underlying field effects that govern their dynamics, yet previous works have abstracted away from such effects, and assume that systems evolve in a vacuum. In this work, we focus on discovering these fields, and infer them from the observed dynamics alone, without directly observing them. We theorize the presence of latent force fields, and propose neural fields to learn them. Since the observed dynamics constitute the net effect of local object interactions and global field effects, recently popularized equivariant networks are inapplicable, as they fail to capture global information. To address this, we propose to disentangle local object interactions --which are SE(3) equivariant and depend on relative states-- from external global field effects --which depend on absolute states. We model the interactions with equivariant graph networks, and combine them with neural fields in a novel graph network that integrates field forces. Our experiments show that we can accurately discover the underlying fields in charged particles settings, traffic scenes, and gravitational n-body problems, and effectively use them to learn the system and forecast future trajectories.</details>

### Latent exploration for Reinforcement Learning

**Authors:** Alberto Silvio Chiappa, Alessandro Marin Vargas, Ann Huang, Alexander Mathis

**<details><summary>Abstract**</summary>

In Reinforcement Learning, agents learn policies by exploring and interacting with the environment. Due to the curse of dimensionality, learning policies that map high-dimensional sensory input to motor output is particularly challenging. During training, state of the art methods (SAC, PPO, etc.) explore the environment by perturbing the actuation with independent Gaussian noise. While this unstructured exploration has proven successful in numerous tasks, it can be suboptimal for overactuated systems. When multiple actuators, such as motors or muscles, drive behavior, uncorrelated perturbations risk diminishing each other's effect, or modifying the behavior in a task-irrelevant way. While solutions to introduce time correlation across action perturbations exist, introducing correlation across actuators has been largely ignored. Here, we propose LATent TIme-Correlated Exploration (Lattice), a method to inject temporally-correlated noise into the latent state of the policy network, which can be seamlessly integrated with on- and off-policy algorithms. We demonstrate that the noisy actions generated by perturbing the network's activations can be modeled as a multivariate Gaussian distribution with a full covariance matrix. In the PyBullet locomotion tasks, Lattice-SAC achieves state of the art results, and reaches 18\% higher reward than unstructured exploration in the Humanoid environment. In the musculoskeletal control environments of MyoSuite, Lattice-PPO achieves higher reward in most reaching and object manipulation tasks, while also finding more energy-efficient policies with reductions of 20-60\%. Overall, we demonstrate the effectiveness of structured action noise in time and actuator space for complex motor control tasks. The code is available at: https://github.com/amathislab/lattice.</details>

### Laughing Hyena Distillery: Extracting Compact Recurrences From Convolutions

**Authors:** Stefano Massaroli, Michael Poli, Dan Fu, Hermann Kumbong, Rom Parnichkun, David Romero, Aman Timalsina, Quinn McIntyre, Beidi Chen, Atri Rudra, Ce Zhang, Christopher Ré, Stefano Ermon, Yoshua Bengio

**<details><summary>Abstract**</summary>

Recent advances in attention-free sequence models rely on convolutions as alternatives to the attention operator at the core of Transformers. In particular, long convolution sequence models have achieved state-of-the-art performance in many domains, but incur a significant cost during auto-regressive inference workloads -- naively requiring a full pass (or caching of activations) over the input sequence for each generated token -- similarly to attention-based models. In this paper, we seek to enable $\mathcal O(1)$ compute and memory cost per token in any pre-trained long convolution architecture to reduce memory footprint and increase throughput during generation. Concretely, our methods consist in extracting low-dimensional linear state-space models from each convolution layer, building upon rational interpolation and model-order reduction techniques. We further introduce architectural improvements to convolution-based layers such as Hyena: by weight-tying the filters across channels into heads, we achieve higher pre-training quality and reduce the number of filters to be distilled. The resulting model achieves 10x higher throughput than Transformers and 1.5x higher than Hyena at 1.3B parameters, without any loss in quality after distillation.</details>

### Layer-Neighbor Sampling --- Defusing Neighborhood Explosion in GNNs

**Authors:** Muhammed Fatih Balin, Ümit Çatalyürek

**<details><summary>Abstract**</summary>

Graph Neural Networks (GNNs) have received significant attention recently, but training them at a large scale remains a challenge.Mini-batch training coupled with sampling is used to alleviate this challenge.However, existing approaches either suffer from the neighborhood explosion phenomenon or have suboptimal performance. To address these issues, we propose a new sampling algorithm called LAyer-neighBOR sampling (LABOR). It is designed to be a direct replacement for Neighbor Sampling (NS) with the same fanout hyperparameter while sampling up to 7 times fewer vertices, without sacrificing quality.By design, the variance of the estimator of each vertex matches NS from the point of view of a single vertex.Moreover, under the same vertex sampling budget constraints, LABOR converges faster than existing layer sampling approaches and can use up to 112 times larger batch sizes compared to NS.</details>

### Learning Dense Flow Field for Highly-accurate Cross-view Camera Localization

**Authors:** Zhenbo Song, ze xianghui, Jianfeng Lu, Yujiao Shi

**<details><summary>Abstract**</summary>

This paper addresses the problem of estimating the 3-DoF camera pose for a ground-level image with respect to a satellite image that encompasses the local surroundings. We propose a novel end-to-end approach that leverages the learning of dense pixel-wise flow fields in pairs of ground and satellite images to calculate the camera pose. Our approach differs from existing methods by constructing the feature metric at the pixel level, enabling full-image supervision for learning distinctive geometric configurations and visual appearances across views. Specifically, our method employs two distinct convolution networks for ground and satellite feature extraction. Then, we project the ground feature map to the bird's eye view (BEV) using a fixed camera height assumption to achieve preliminary geometric alignment. To further establish the content association between the BEV and satellite features, we introduce a residual convolution block to refine the projected BEV feature. Optical flow estimation is performed on the refined BEV feature map and the satellite feature map using flow decoder networks based on RAFT. After obtaining dense flow correspondences, we apply the least square method to filter matching inliers and regress the ground camera pose. Extensive experiments demonstrate significant improvements compared to state-of-the-art methods. Notably, our approach reduces the median localization error by 89\%, 19\%, 80\%, and 35\% on the KITTI, Ford multi-AV, VIGOR, and Oxford RobotCar datasets, respectively.</details>

### Learning Descriptive Image Captioning via Semipermeable Maximum Likelihood Estimation

**Authors:** Zihao Yue, Anwen Hu, Liang Zhang, Qin Jin

**<details><summary>Abstract**</summary>

Image captioning aims to describe visual content in natural language. As 'a picture is worth a thousand words', there could be various correct descriptions for an image. However, with maximum likelihood estimation as the training objective, the captioning model is penalized whenever its prediction mismatches with the label. For instance, when the model predicts a word expressing richer semantics than the label, it will be penalized and optimized to prefer more concise expressions, referred to as *conciseness optimization*. In contrast, predictions that are more concise than labels lead to *richness optimization*. Such conflicting optimization directions could eventually result in the model generating general descriptions. In this work, we introduce Semipermeable MaxImum Likelihood Estimation (SMILE), which allows richness optimization while blocking conciseness optimization, thus encouraging the model to generate longer captions with more details. Extensive experiments on two mainstream image captioning datasets MSCOCO and Flickr30K demonstrate that SMILE significantly enhances the descriptiveness of generated captions. We further provide in-depth investigations to facilitate a better understanding of how SMILE works.</details>

### Learning Exponential Families from Truncated Samples

**Authors:** Jane Lee, Andre Wibisono, Emmanouil Zampetakis

**<details><summary>Abstract**</summary>

Missing data problems have many manifestations across many scientific fields. A fundamental type of missing data problem arises when samples are \textit{truncated}, i.e., samples that lie in a subset of the support are not observed. Statistical estimation from truncated samples is a classical problem in statistics which dates back to Galton, Pearson, and Fisher. A recent line of work provides the first efficient estimation algorithms for the parameters of a Gaussian distribution and for linear regression with Gaussian noise.In this paper we generalize these results to log-concave exponential families. We provide an estimation algorithm that shows that \textit{extrapolation} is possible for a much larger class of distributions while it maintains a polynomial sample and time complexity on average. Our algorithm is based on Projected Stochastic Gradient Descent and is not only applicable in a more general setting but is also simpler and more efficient than recent algorithms. Our work also has interesting implications for learning general log-concave distributions and sampling given only access to truncated data.</details>

### Learning Fine-grained View-Invariant Representations from Unpaired Ego-Exo Videos via Temporal Alignment

**Authors:** Zihui (Sherry) Xue, Kristen Grauman

**<details><summary>Abstract**</summary>

The egocentric and exocentric viewpoints of a human activity look dramatically different, yet invariant representations to link them are essential for many potential applications in robotics and augmented reality.  Prior work is limited to learning view-invariant features from paired synchronized viewpoints.  We relax that strong data assumption and propose to learn fine-grained action features that are invariant to the viewpoints by aligning egocentric and exocentric videos in time, even when not captured simultaneously or in the same environment. To this end, we propose AE2, a self-supervised embedding approach with two key designs: (1) an object-centric encoder that explicitly focuses on regions corresponding to hands and active objects; (2) a contrastive-based alignment objective that leverages temporally reversed frames as negative samples. For evaluation, we establish a benchmark for fine-grained video understanding in the ego-exo context, comprising four datasets---including an ego tennis forehand dataset we collected, along with dense per-frame labels we annotated for each dataset. On the four datasets, our AE2 method strongly outperforms prior work in a variety of fine-grained downstream tasks, both in regular and cross-view settings.</details>

### Learning Large-scale Neural Fields via Context Pruned Meta-Learning

**Authors:** Jihoon Tack, Subin Kim, Sihyun Yu, Jaeho Lee, Jinwoo Shin, Jonathan Richard Schwarz

**<details><summary>Abstract**</summary>

We introduce an efficient optimization-based meta-learning technique for large-scale neural field training by realizing significant memory savings through automated online context point selection. This is achieved by focusing each learning step on the subset of data with the highest expected immediate improvement in model quality, resulting in the almost instantaneous modeling of global structure and subsequent refinement of high-frequency details. We further improve the quality of our meta-learned initialization by introducing a bootstrap correction resulting in the minimization of any error introduced by reduced context sets while simultaneously mitigating the well-known myopia of optimization-based meta-learning. Finally, we show how gradient re-scaling at meta-test time allows the learning of extremely high-quality neural fields in significantly shortened optimization procedures. Our framework is model-agnostic, intuitive, straightforward to implement, and shows significant reconstruction improvements for a wide range of signals. We provide an extensive empirical evaluation on nine datasets across multiple multiple modalities, demonstrating state-of-the-art results while providing additional insight through careful analysis of the algorithmic components constituting our method. Code is available at https://github.com/jihoontack/GradNCP</details>

### Learning Mask-aware CLIP Representations for Zero-Shot Segmentation

**Authors:** Siyu Jiao, Yunchao Wei, Yaowei Wang, Yao Zhao, Humphrey Shi

**<details><summary>Abstract**</summary>

Recently, pre-trained vision-language models have been increasingly used to tackle the challenging zero-shot segmentation task. Typical solutions follow the paradigm of first generating mask proposals and then adopting CLIP to classify them. To maintain the CLIP's zero-shot transferability, previous practices favour to freeze CLIP during training. However, in the paper, we reveal that CLIP is insensitive to different mask proposals and tends to produce similar predictions for various mask proposals of the same image. This insensitivity results in numerous false positives when classifying mask proposals. This issue mainly relates to the fact that CLIP is trained with image-level supervision. To alleviate this issue, we propose a simple yet effective method, named Mask-aware Fine-tuning (MAFT). Specifically, Image-Proposals CLIP Encoder (IP-CLIP Encoder) is proposed to handle arbitrary numbers of image and mask proposals simultaneously. Then, *mask-aware loss* and *self-distillation loss* are designed to fine-tune IP-CLIP Encoder, ensuring CLIP is responsive to different mask proposals while not sacrificing transferability. In this way, mask-aware representations can be easily learned to make the true positives stand out. Notably, our solution can seamlessly plug into most existing methods without introducing any new parameters during the fine-tuning process. We conduct extensive experiments on the popular zero-shot benchmarks. With MAFT, the performance of the state-of-the-art methods is promoted by a large margin: 50.4\% (+ 8.2\%) on COCO, 81.8\% (+ 3.2\%) on Pascal-VOC, and 8.7\% (+4.3\%) on ADE20K in terms of mIoU for unseen classes. Codes will be provided for reproducibility. Code is available at https://github.com/jiaosiyu1999/MAFT.git.</details>

### Learning Multi-agent Behaviors from Distributed and Streaming Demonstrations

**Authors:** Shicheng Liu, Minghui Zhu

**<details><summary>Abstract**</summary>

This paper considers the problem of inferring the behaviors of multiple interacting experts by estimating their reward functions and constraints where the distributed demonstrated trajectories are sequentially revealed to a group of learners. We formulate the problem as a distributed online bi-level optimization problem where the outer-level problem is to estimate the reward functions and the inner-level problem is to learn the constraints and corresponding policies. We propose a novel ``multi-agent behavior inference from distributed and streaming demonstrations" (MA-BIRDS) algorithm that allows the learners to solve the outer-level and inner-level problems in a single loop through intermittent communications. We formally guarantee that the distributed learners achieve consensus on reward functions, constraints, and policies, the average local regret (over $N$ online iterations) decreases at the rate of $O(1/N^{1-\eta_1}+1/N^{1-\eta_2}+1/N)$, and the cumulative constraint violation increases sub-linearly at the rate of $O(N^{\eta_2}+1)$ where $\eta_1,\eta_2\in (1/2,1)$.</details>

### Learning Time-Invariant Representations for Individual Neurons from Population Dynamics

**Authors:** Lu Mi, Trung Le, Tianxing He, Eli Shlizerman, Uygar Sümbül

**<details><summary>Abstract**</summary>

Neurons can display highly variable dynamics. While such variability presumably supports the wide range of behaviors generated by the organism, their gene expressions are relatively stable in the adult brain. This suggests that neuronal activity is a combination of its time-invariant identity and the inputs the neuron receives from the rest of the circuit. Here, we propose a self-supervised learning based method to assign time-invariant representations to individual neurons based on permutation-, and population size-invariant summary of population recordings. We fit dynamical models to neuronal activity to learn a representation by considering the activity of both the individual and the neighboring population. Our self-supervised approach and use of implicit representations enable robust inference against imperfections such as partial overlap of neurons across sessions, trial-to-trial variability, and limited availability of molecular (transcriptomic) labels for downstream supervised tasks. We demonstrate our method on a public multimodal dataset of mouse cortical neuronal activity and transcriptomic labels. We report >35\% improvement in predicting the transcriptomic subclass identity and >20\% improvement in predicting class identity with respect to the state-of-the-art.</details>

### [Oral] Learning Transformer Programs

**Authors:** Dan Friedman, Alexander Wettig, Danqi Chen

**Oral Presentation:** We, Dec 13, 08:30 -- Oral 3B

**<details><summary>Abstract**</summary>

Recent research in mechanistic interpretability has attempted to reverse-engineer Transformer models by carefully inspecting network weights and activations. However, these approaches require considerable manual effort and still fall short of providing complete, faithful descriptions of the underlying algorithms. In this work, we introduce a procedure for training Transformers that are mechanistically interpretable by design. We build on RASP [Weiss et al., 2021], a programming language that can be compiled into Transformer weights. Instead of compiling human-written programs into Transformers, we design a modified Transformer that can be trained using gradient-based optimization and then automatically converted into a discrete, human-readable program. We refer to these models as Transformer Programs. To validate our approach, we learn Transformer Programs for a variety of problems, including an in-context learning task, a suite of algorithmic problems (e.g. sorting, recognizing Dyck languages), and NLP tasks including named entity recognition and text classification. The Transformer Programs can automatically find reasonable solutions, performing on par with standard Transformers of comparable size; and, more importantly, they are easy to interpret. To demonstrate these advantages, we convert Transformers into Python programs and use off-the-shelf code analysis tools to debug model errors and identify the “circuits” used to solve different sub-problems. We hope that Transformer Programs open a new path toward the goal of intrinsically interpretable machine learning.</details>

### Learning Visual Prior via Generative Pre-Training

**Authors:** Jinheng Xie, Kai Ye, Yudong Li, Yuexiang Li, Kevin Qinghong Lin, Yefeng Zheng, Linlin Shen, Mike Zheng Shou

**<details><summary>Abstract**</summary>

Various stuff and things in visual data possess specific traits, which can be learned by deep neural networks and are implicitly represented as the visual prior, e.g., object location and shape, in the model. Such prior potentially impacts many vision tasks. For example, in conditional image synthesis, spatial conditions failing to adhere to the prior can result in visually inaccurate synthetic results. This work aims to explicitly learn the visual prior and enable the customization of sampling. Inspired by advances in language modeling, we propose to learn Visual prior via Generative Pre-Training, dubbed VisorGPT. By discretizing visual locations, e.g., bounding boxes, human pose, and instance masks, into sequences, VisorGPT can model visual prior through likelihood maximization. Besides, prompt engineering is investigated to unify various visual locations and enable customized sampling of sequential outputs from the learned prior. Experimental results demonstrate the effectiveness of VisorGPT in modeling visual prior and extrapolating to novel scenes, potentially motivating that discrete visual locations can be integrated into the learning paradigm of current language models to further perceive visual world. Code is available at https://sierkinhane.github.io/visor-gpt.</details>

### Learning in the Presence of Low-dimensional Structure: A Spiked Random Matrix Perspective

**Authors:** Jimmy Ba, Murat Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu

**<details><summary>Abstract**</summary>

We consider the learning of a single-index target function $f_*: \mathbb{R}^d\to\mathbb{R}$ under spiked covariance data: $$f_*(\boldsymbol{x}) = \textstyle\sigma_*(\frac{1}{\sqrt{1+\theta}}\langle\boldsymbol{x},\boldsymbol{\mu}\rangle), ~~ \boldsymbol{x}\overset{\small\mathrm{i.i.d.}}{\sim}\mathcal{N}(0,\boldsymbol{I_d} + \theta\boldsymbol{\mu}\boldsymbol{\mu}^\top), ~~ \theta\asymp d^{\beta} \text{ for } \beta\in[0,1), $$ where the link function $\sigma_*:\mathbb{R}\to\mathbb{R}$ is a degree-$p$ polynomial with information exponent $k$ (defined as the lowest degree in the Hermite expansion of $\sigma_*$), and it depends on the projection of input $\boldsymbol{x}$ onto the spike (signal) direction $\boldsymbol{\mu}\in\mathbb{R}^d$. In the proportional asymptotic limit where the number of training examples $n$ and the dimensionality $d$ jointly diverge: $n,d\to\infty, n/d\to\psi\in(0,\infty)$, we ask the following question: how large should the spike magnitude $\theta$ (i.e., the strength of the low-dimensional component) be, in order for $(i)$ kernel methods, $(ii)$ neural networks optimized by gradient descent, to learn $f_*$? We show that for kernel ridge regression, $\beta\ge 1-\frac{1}{p}$ is both sufficient and necessary. Whereas for two-layer neural networks trained with gradient descent, $\beta>1-\frac{1}{k}$ suffices. Our results demonstrate that both kernel methods and neural networks benefit from low-dimensional structures in the data. Further, since $k\le p$ by definition, neural networks can adapt to such structures more effectively.</details>

### Learning to Augment Distributions for Out-of-distribution Detection

**Authors:** Qizhou Wang, Zhen Fang, Yonggang Zhang, Feng Liu, Yixuan Li, Bo Han

**<details><summary>Abstract**</summary>

Open-world classification systems should discern out-of-distribution (OOD) data whose labels deviate from those of in-distribution (ID) cases, motivating recent studies in OOD detection. Advanced works, despite their promising progress, may still fail in the open world, owing to the lacking knowledge about unseen OOD data in advance. Although one can access auxiliary OOD data (distinct from unseen ones) for model training, it remains to analyze how such auxiliary data will work in the open world. To this end, we delve into such a problem from a learning theory perspective, finding that the distribution discrepancy between the auxiliary and the unseen real OOD data is the key to affect the open-world detection performance. Accordingly, we propose Distributional-Augmented OOD Learning (DAOL), alleviating the OOD distribution discrepancy by crafting an OOD distribution set that contains all distributions in a Wasserstein ball centered on the auxiliary OOD distribution. We justify that the predictor trained over the worst OOD data in the ball can shrink the OOD distribution discrepancy, thus improving the open-world detection performance given only the auxiliary OOD data. We conduct extensive evaluations across representative OOD detection setups, demonstrating the superiority of our DAOL over its advanced counterparts.</details>

### [Spotlight] Leveraging sparse and shared feature activations for disentangled representation learning

**Authors:** Marco Fumero, Florian Wenzel, Luca Zancato, Alessandro Achille, Emanuele Rodolà, Stefano Soatto, Bernhard Schölkopf, Francesco Locatello

**<details><summary>Abstract**</summary>

Recovering the latent factors of variation of high dimensional data has so far focused on simple synthetic settings. Mostly building on unsupervised and weakly-supervised objectives, prior work missed out on the positive implications for representation learning on real world data. In this work, we propose to leverage knowledge extracted from a diversified set of supervised tasks to learn a common disentangled representation. Assuming each supervised task only depends on an unknown subset of the factors of variation, we disentangle the feature space of a supervised multi-task model, with features activating sparsely across different tasks and information being shared as appropriate. Importantly, we never directly observe the factors of variations but establish that access to multiple tasks is sufficient for identifiability under sufficiency and minimality assumptions.We validate our approach on six real world distribution shift benchmarks, and different data modalities (images, text), demonstrating how disentangled representations can be transferred to real settings.</details>

### [Spotlight] Lexinvariant Language Models

**Authors:** Qian Huang, Eric Zelikman, Sarah Chen, Yuhuai Wu, Gregory Valiant, Percy Liang

**<details><summary>Abstract**</summary>

Token embeddings, a mapping from discrete lexical symbols to continuous vectors, are at the heart of any language model (LM). However, lexical symbol meanings can also be determined and even redefined by their structural role in a long context. In this paper, we ask: is it possible for a language model to be performant without \emph{any} fixed token embeddings? Such a language model would have to rely entirely on the co-occurence and repetition of tokens in the context rather than the \textit{a priori} identity of any token. To answer this, we study \textit{lexinvariant}language models that are invariant to lexical symbols and therefore do not need fixed token embeddings in practice. First, we prove that we can construct a lexinvariant LM to converge to the true language model at a uniform rate that is polynomial in terms of the context length, with a constant factor that is sublinear in the vocabulary size. Second, to build a lexinvariant LM, we simply encode tokens using random Gaussian vectors, such that each token maps to the same representation within each sequence but different representations across sequences. Empirically, we demonstrate that it can indeed attain perplexity comparable to that of a standard language model, given a sufficiently long context. We further explore two properties of the lexinvariant language models: First, given text generated from a substitution cipher of English, it implicitly implements Bayesian in-context deciphering and infers the mapping to the underlying real tokens with high accuracy. Second, it has on average 4X better accuracy over synthetic in-context reasoning tasks. Finally, we discuss regularizing standard language models towards lexinvariance and potential practical applications.</details>

### Logarithmic Bayes Regret Bounds

**Authors:** Alexia Atsidakou, Branislav Kveton, Sumeet Katariya, Constantine Caramanis, Sujay Sanghavi

**<details><summary>Abstract**</summary>

We derive the first finite-time logarithmic Bayes regret upper bounds for Bayesian bandits. In Gaussian bandits, we obtain $O(c_\Delta \log n)$ and $O(c_h \log^2 n)$ bounds for an upper confidence bound algorithm, where $c_h$ and $c_\Delta$ are constants depending on the prior distribution and the gaps of random bandit instances sampled from it, respectively. The latter bound asymptotically matches the lower bound of Lai (1987). Our proofs are a major technical departure from prior works, while being simple and general. To show the generality of our techniques, we apply them to linear bandits. Our results provide insights on the value of prior in the Bayesian setting, both in the objective and as a side information given to the learner. They significantly improve upon existing $\tilde{O}(\sqrt{n})$ bounds, which have become standard in the literature despite the existing lower bounds.</details>

### Lossy Image Compression with Conditional Diffusion Models

**Authors:** Ruihan Yang, Stephan Mandt

**<details><summary>Abstract**</summary>

This paper outlines an end-to-end optimized lossy image compression framework using diffusion generative models. The approach relies on the transform coding paradigm, where an image is mapped into a latent space for entropy coding and, from there, mapped back to the data space for reconstruction. In contrast to VAE-based neural compression, where the (mean) decoder is a deterministic neural network, our decoder is a conditional diffusion model. Our approach thus introduces an additional "content" latent variable on which the reverse diffusion process is conditioned and uses this variable to store information about the image. The remaining "texture" variables characterizing the diffusion process are synthesized at decoding time. We show that the model's performance can be tuned toward perceptual metrics of interest. Our extensive experiments involving multiple datasets and image quality assessment metrics show that our approach yields stronger reported FID scores than the GAN-based model, while also yielding competitive performance with VAE-based models in several distortion metrics. Furthermore, training the diffusion with  $\mathcal{X}$-parameterization enables high-quality reconstructions in only a handful of decoding steps, greatly affecting the model's practicality.</details>

### Low-shot Object Learning with Mutual Exclusivity Bias

**Authors:** Anh Thai, Ahmad Humayun, Stefan Stojanov, Zixuan Huang, Bikram Boote, James Rehg

**<details><summary>Abstract**</summary>

This paper introduces Low-shot Object Learning with Mutual Exclusivity Bias (LSME), the first computational framing of mutual exclusivity bias, a phenomenon commonly observed in infants during word learning. We provide a novel dataset, comprehensive baselines, and a SOTA method to enable the ML community to tackle this challenging learning task. The goal of LSME is to analyze an RGB image of a scene containing multiple objects and correctly associate a previously-unknown object instance with a provided category label. This association is then used to perform low-shot learning to test category generalization. We provide a data generation pipeline for the LSME problem and conduct a thorough analysis of the factors that contribute to its difficulty. Additionally, we evaluate the performance of multiple baselines, including state-of-the-art foundation models. Finally, we present a baseline approach that outperforms state-of-the-art models in terms of low-shot accuracy. Code and data are available at https://github.com/rehg-lab/LSME.</details>

### MAG-GNN: Reinforcement Learning Boosted Graph Neural Network

**Authors:** Lecheng Kong, Jiarui Feng, Hao Liu, Dacheng Tao, Yixin Chen, Muhan Zhang

**<details><summary>Abstract**</summary>

While Graph Neural Networks (GNNs) recently became powerful tools in graph learning tasks, considerable efforts have been spent on improving GNNs' structural encoding ability. A particular line of work proposed subgraph GNNs that use subgraph information to improve GNNs' expressivity and achieved great success. However, such effectivity sacrifices the efficiency of GNNs by enumerating all possible subgraphs. In this paper, we analyze the necessity of complete subgraph enumeration and show that a model can achieve a comparable level of expressivity by considering a small subset of the subgraphs. We then formulate the identification of the optimal subset as a combinatorial optimization problem and propose Magnetic Graph Neural Network (MAG-GNN), a reinforcement learning (RL) boosted GNN, to solve the problem. Starting with a candidate subgraph set, MAG-GNN employs an RL agent to iteratively update the subgraphs to locate the most expressive set for prediction. This reduces the exponential complexity of subgraph enumeration to the constant complexity of a subgraph search algorithm while keeping good expressivity. We conduct extensive experiments on many datasets, showing that MAG-GNN achieves competitive performance to state-of-the-art methods and even outperforms many subgraph GNNs. We also demonstrate that MAG-GNN effectively reduces the running time of subgraph GNNs.</details>

### MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers

**Authors:** LILI YU, Daniel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, Mike Lewis

**<details><summary>Abstract**</summary>

Autoregressive transformers are spectacular models for short sequences but scale poorly to long sequences such as high-resolution images, podcasts, code, or books. We proposed Megabyte, a multi-scale decoder architecture that enables end-to-end differentiable modeling of sequences of over one million bytes. Megabyte segments sequences into patches and uses a local submodel within patches and a global model between patches. This enables sub-quadratic self-attention, much larger feedforward layers for the same compute, and improved parallelism during decoding---unlocking  better performance at reduced cost for both training and generation. Extensive experiments show that Megabyte allows byte-level models to perform competitively with subword models on long context language modeling, achieve state-of-the-art density estimation on ImageNet, and model audio from raw files. Together, these results establish the  viability of tokenization-free autoregressive sequence modeling at scale.</details>

### [Spotlight] MGDD: A Meta Generator for Fast Dataset Distillation

**Authors:** Songhua Liu, Xinchao Wang

**<details><summary>Abstract**</summary>

Existing dataset distillation (DD) techniques typically rely on iterative strategies to synthesize condensed datasets, where datasets before and after distillation are forward and backward through neural networks a massive number of times. Despite the promising results achieved, the time efficiency of prior approaches is still far from satisfactory. Moreover, when different sizes of synthetic datasets are required, they have to repeat the iterative training procedures, which is highly cumbersome and lacks flexibility. In this paper, different from the time-consuming forward-backward passes, we introduce a generative fashion for dataset distillation with significantly improved efficiency. Specifically, synthetic samples are produced by a generator network conditioned on the initialization of DD, while synthetic labels are obtained by solving a least-squares problem in a feature space. Our theoretical analysis reveals that the errors of synthetic datasets solved in the original space and then processed by any conditional generators are upper-bounded. To find a satisfactory generator efficiently, we propose a meta-learning algorithm, where a meta generator is trained on a large dataset so that only a few steps are required to adapt to a target dataset. The meta generator is termed as MGDD in our approach. Once adapted, it can handle arbitrary sizes of synthetic datasets, even for those unseen during adaptation. Experiments demonstrate that the generator adapted with only a limited number of steps performs on par with those state-of-the-art DD methods and yields $22\times$ acceleration.</details>

### [Spotlight] MMD-Fuse: Learning and Combining Kernels for Two-Sample Testing Without Data Splitting

**Authors:** Felix Biggs, Antonin Schrab, Arthur Gretton

**<details><summary>Abstract**</summary>

We propose novel statistics which maximise the power  of a two-sample test based on the Maximum Mean Discrepancy (MMD), byadapting over the set of kernels used in defining it.For finite sets, this reduces to combining (normalised) MMD values under each of these kernels via a weighted soft maximum.Exponential concentration bounds are proved for our proposed statistics under the null and alternative.We further show how these kernels can be chosen in a data-dependent but permutation-independent way, in a well-calibrated test, avoiding data splitting.This technique applies more broadly to general permutation-based MMD testing, and includes the use of deep kernels with features learnt using unsupervised models such as auto-encoders.We highlight the applicability of our MMD-Fuse tests on both synthetic low-dimensional and real-world high-dimensional data, and compare its performance in terms of power against current state-of-the-art kernel tests.</details>

### Machine learning detects terminal singularities

**Authors:** Tom Coates, Alexander Kasprzyk, Sara Veneziale

**<details><summary>Abstract**</summary>

Algebraic varieties are the geometric shapes defined by systems of polynomial equations; they are ubiquitous across mathematics and science. Amongst these algebraic varieties are Q-Fano varieties: positively curved shapes which have Q-factorial terminal singularities. Q-Fano varieties are of fundamental importance in geometry as they are `atomic pieces’ of more complex shapes – the process of breaking a shape into simpler pieces in this sense is called the Minimal Model Programme.Despite their importance, the classification of Q-Fano varieties remains unknown. In this paper we demonstrate that machine learning can be used to understand this classification. We focus on eight-dimensional positively-curved algebraic varieties that have toric symmetry and Picard rank two, and develop a neural network classifier that predicts with 95% accuracy whether or not such an algebraic variety is Q-Fano. We use this to give a first sketch of the landscape of Q-Fano varieties in dimension eight.How the neural network is able to detect Q-Fano varieties with such accuracy remains mysterious, and hints at some deep mathematical theory waiting to be uncovered. Furthermore, when visualised using the quantum period, an invariant that has played an important role in recent theoretical developments, we observe that the classification as revealed by ML appears to fall within a bounded region, and is stratified by the Fano index. This suggests that it may be possible to state and prove conjectures on completeness in the future.Inspired by the ML analysis, we formulate and prove a new global combinatorial criterion for a positively curved toric variety of Picard rank two to have terminal singularities. Together with the first sketch of the landscape of Q-Fano varieties in higher dimensions, this gives strong new evidence that machine learning can be an essential tool in developing mathematical conjectures and accelerating theoretical discovery.</details>

### Make Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning

**Authors:** Baohao Liao, Shaomu Tan, Christof Monz

**<details><summary>Abstract**</summary>

Parameter-efficient fine-tuning (PEFT) of pre-trained language models (PLMs) has emerged as a highly successful approach, with training only a small number of parameters without sacrificing performance and becoming the de-facto learning paradigm with the increasing size of PLMs. However, existing PEFT methods are not memory-efficient, because they still require caching most of the intermediate activations for the gradient calculation, akin to fine-tuning. One effective way to reduce the activation memory is to apply a reversible model, so the intermediate activations are not necessary to be cached and can be recomputed. Nevertheless, modifying a PLM to its reversible variant is not straightforward, since the reversible model has a distinct architecture from the currently released PLMs. In this paper, we first investigate what is a key factor for the success of existing PEFT methods, and realize that it's essential to preserve the PLM's starting point when initializing a PEFT method. With this finding, we propose memory-efficient fine-tuning (MEFT) that inserts adapters into a PLM, preserving the PLM's starting point and making it reversible without additional pre-training. We evaluate MEFT on the GLUE benchmark and five question-answering tasks with various backbones, BERT, RoBERTa, BART and OPT. MEFT significantly reduces the activation memory up to 84% of full fine-tuning with a negligible amount of trainable parameters. Moreover, MEFT achieves the same score on GLUE and a comparable score on the question-answering tasks as full fine-tuning. A similar finding is also observed for the image classification task.</details>

### [Spotlight] MeCo: Zero-Shot NAS with One Data and Single Forward Pass via Minimum Eigenvalue of Correlation

**Authors:** Tangyu Jiang, Haodi Wang, Rongfang Bie

**<details><summary>Abstract**</summary>

Neural Architecture Search (NAS) is a promising paradigm in automatic architecture engineering. Zero-shot NAS can evaluate the network without training via some specific metrics called zero-cost proxies. Though effective, the existing zero-cost proxies either invoke at least one backpropagation or depend highly on the data and labels. To alleviate the above issues, in this paper, we first reveal how the Pearson correlation matrix of the feature maps impacts the convergence rate and the generalization capacity of an over-parameterized neural network. Enlightened by the theoretical analysis, we propose a novel zero-cost proxy called $\mathsf{MeCo}$, which requires only one random data for a single forward pass. We further propose an optimization approach $\mathsf{MeCo_{opt}}$ to improve the performance of our method. We design comprehensive experiments and extensively evaluate $\mathsf{MeCo}$ on multiple popular benchmarks. $\mathsf{MeCo}$ achieves the highest correlation with the ground truth (e.g., 0.89 on NATS-Bench-TSS with CIFAR-10) among all the state-of-the-art proxies, which is also fully independent of the data and labels. Moreover, we integrate $\mathsf{MeCo}$ with the existing generation method to comprise a complete NAS. The experimental results illustrate that $\mathsf{MeCo}$-based NAS can select the architecture with the highest accuracy and a low search cost. For instance, the best network searched by $\mathsf{MeCo}$-based NAS achieves 97.31% on CIFAR-10, which is 0.04% higher than the baselines under the same settings. Our code is available at https://github.com/HamsterMimi/MeCo</details>

### Meta-learning families of plasticity rules in recurrent spiking networks using simulation-based inference

**Authors:** Basile Confavreux, Poornima Ramesh, Pedro Goncalves, Jakob H Macke, Tim Vogels

**<details><summary>Abstract**</summary>

There is substantial experimental evidence that learning and memory-related behaviours rely on local synaptic changes, but the search for distinct plasticity rules has been driven by human intuition, with limited success for multiple, co-active plasticity rules in biological networks. More recently, automated meta-learning approaches have been used in simplified settings, such as rate networks and small feed-forward spiking networks. Here, we develop a simulation-based inference (SBI) method for sequentially filtering plasticity rules through an increasingly fine mesh of constraints that can be modified on-the-fly. This method, _filter SBI_, allows us to infer entire families of complex and co-active plasticity rules in spiking networks. We first consider flexibly parameterized doublet (Hebbian) rules, and find that the set of inferred rules contains solutions that extend and refine -and also reject- predictions from mean-field theory. Next, we expand the search space of plasticity rules by modelling them as multi-layer perceptrons that combine several plasticity-relevant factors, such as weight, voltage, triplets and co-dependency. Out of the millions of possible rules, we identify thousands of unique rule combinations that satisfy biological constraints like plausible activity and weight dynamics. The resulting rules can be used as a starting point for further investigations into specific network computations, and already suggest refinements and predictions for classical experimental approaches on plasticity. This flexible approach for principled exploration of complex plasticity rules in large recurrent spiking networks presents the most advanced search tool to date for enabling robust predictions and deep insights into the plasticity mechanisms underlying brain function.</details>

### Minimax-Optimal Location Estimation

**Authors:** Shivam Gupta, Jasper Lee, Eric Price, Paul Valiant

**<details><summary>Abstract**</summary>

Location estimation is one of the most basic questions in parametric statistics. Suppose we have a known distribution density $f$, and we get $n$ i.i.d. samples from $f(x-\mu)$ for some unknown shift $\mu$.The task is to estimate $\mu$ to high accuracy with high probability.The maximum likelihood estimator (MLE) is known to be asymptotically optimal as $n \to \infty$, but what is possible for finite $n$?In this paper, we give two location estimators that are optimal under different criteria: 1) an estimator that has minimax-optimal estimation error subject to succeeding with probability $1-\delta$ and 2) a confidence interval estimator which, subject to its output interval containing $\mu$ with probability at least $1-\delta$, has the minimum expected squared interval width among all shift-invariant estimators.The latter construction can be generalized to minimizing the expectation of any loss function on the interval width.</details>

### Minimum Description Length and Generalization Guarantees for Representation Learning

**Authors:** Milad Sefidgaran, Abdellatif Zaidi, Piotr Krasnowski

**<details><summary>Abstract**</summary>

A major challenge in designing efficient statistical supervised learning algorithms is finding representations that perform well not only on available training samples but also on unseen data. While the study of representation learning has spurred much interest, most existing such approaches are heuristic; and very little is known about theoretical generalization guarantees. For example, the information bottleneck method seeks a good generalization by finding a minimal description of the input that is maximally informative about the label variable, where minimality and informativeness are both measured by Shannon’s mutual information. In this paper, we establish a compressibility framework that allows us to derive upper bounds on the generalization error of a representation learning algorithm in terms of the "Minimum Description Length'' (MDL) of the labels or the latent variables (representations).  Rather than the mutual information between the encoder’s input and the representation, which is often believed to reflect the algorithm’s generalization capability in the related literature but in fact, falls short of doing so, our new bounds involve the "multi-letter" relative entropy between the distribution of the representations (or labels) of the training and test sets and a fixed prior. In particular, these new bounds reflect the structure of the encoder and are not vacuous for deterministic algorithms. Our compressibility approach, which is information-theoretic in nature, builds upon that of Blum-Langford for PAC-MDL bounds and introduces two essential ingredients: block-coding and lossy-compression. The latter allows our approach to subsume the so-called geometrical compressibility as a special case. To the best knowledge of the authors, the established generalization bounds are the first of their kind for Information Bottleneck type encoders and representation learning. Finally, we partly exploit the theoretical results by introducing a new data-dependent prior. Numerical simulations illustrate the advantages of well-chosen such priors over classical priors used in IB.</details>

### Minimum norm interpolation by perceptra: Explicit regularization and implicit bias

**Authors:** Jiyoung Park, Ian Pelakh, Stephan Wojtowytsch

**<details><summary>Abstract**</summary>

We investigate how shallow ReLU networks interpolate between known regions. Our analysis shows that empirical risk minimizers converge to a minimum norm interpolant as the number of data points and parameters tends to infinity when a weight decay regularizer is penalized with a coefficient which vanishes at a precise rate as the network width and the number of data points grow. With and without explicit regularization, we numerically study the implicit bias of common optimization algorithms towards known minimum norm interpolants.</details>

### Mitigating Test-Time Bias for Fair Image Retrieval

**Authors:** Fanjie Kong, Shuai Yuan, Weituo Hao, Ricardo Henao

**<details><summary>Abstract**</summary>

We address the challenge of generating fair and unbiased image retrieval results given neutral textual queries (with no explicit gender or race connotations), while maintaining the utility (performance) of the underlying vision-language (VL) model. Previous methods aim to disentangle learned representations of images and text queries from gender and racial characteristics. However, we show these are inadequate at alleviating bias for the desired equal representation result, as there usually exists test-time bias in the target retrieval set. So motivated, we introduce a straightforward technique, Post-hoc Bias Mitigation (PBM), that post-processes the outputs from the pre-trained vision-language model. We evaluate our algorithm on real-world image search datasets, Occupation 1 and 2, as well as two large-scale image-text datasets, MS-COCO and Flickr30k. Our approach achieves the lowest bias, compared with various existing bias-mitigation methods, in text-based image retrieval result while maintaining satisfactory retrieval performance. The source code is publicly available at \url{https://github.com/timqqt/Fair_Text_based_Image_Retrieval}.</details>

### [Spotlight] Mitigating the Popularity Bias of  Graph Collaborative Filtering: A Dimensional Collapse Perspective

**Authors:** Yifei Zhang, Hao Zhu, yankai Chen, Zixing Song, Piotr Koniusz, Irwin King

**<details><summary>Abstract**</summary>

Graph-based Collaborative Filtering (GCF) is widely used in personalized recommendation systems. However, GCF suffers from a fundamental problem where features tend to occupy the embedding space inefficiently (by spanning only a low-dimensional subspace). Such an effect is characterized in GCF by the embedding space being dominated by a few of popular items with the user embeddings highly concentrated around them. This enhances the so-called Matthew effect of the popularity bias where popular items are highly recommend whereas remaining items are ignored. In this paper, we analyze the above effect in GCF and reveal that the simplified graph convolution operation (typically used in GCF) shrinks the singular space of the feature matrix. As typical approaches  (i.e., optimizing the uniformity term) fail to prevent the embedding space degradation, we propose a decorrelation-enhanced GCF objective that promotes feature diversity by leveraging the so-called principle of redundancy reduction in embeddings. However, unlike conventional methods that use the Euclidean geometry to relax hard constraints for decorrelation, we exploit non-Euclidean geometry. Such a choice helps  maintain the range space of the matrix and obtain small condition number, which prevents the embedding space degradation. Our  method  outperforms contrastive-based GCF models on several benchmark datasets and improves the performance for unpopular items.</details>

### Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models

**Authors:** Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, WUYOU XIAO, Rui Zhao, Shuning Chang, Weijia Wu, Yixiao Ge, Ying Shan, Mike Zheng Shou

**<details><summary>Abstract**</summary>

Public large-scale text-to-image diffusion models, such as Stable Diffusion, have gained significant attention from the community. These models can be easily customized for new concepts using low-rank adaptations (LoRAs). However,  the utilization of multiple-concept LoRAs to jointly support multiple customized concepts presents a challenge. We refer to this scenario as decentralized multi-concept customization, which involves single-client concept tuning and center-node concept fusion. In this paper, we propose a new framework called Mix-of-Show that addresses the challenges of decentralized multi-concept customization, including concept conflicts resulting from existing single-client LoRA tuning and identity loss during model fusion. Mix-of-Show adopts an embedding-decomposed LoRA (ED-LoRA) for single-client tuning and gradient fusion for the center node to preserve the in-domain essence of single concepts and support theoretically limitless concept fusion. Additionally, we introduce regionally controllable sampling, which extends spatially controllable sampling (e.g., ControlNet and T2I-Adapter) to address attribute binding and missing object problems in multi-concept sampling. Extensive experiments demonstrate that Mix-of-Show is capable of composing multiple customized concepts with high fidelity, including characters, objects, and scenes.</details>

### Mnemosyne: Learning to Train Transformers with Transformers

**Authors:** Deepali Jain, Krzysztof M Choromanski, Kumar Avinava Dubey, Sumeet Singh, Vikas Sindhwani, Tingnan Zhang, Jie Tan

**<details><summary>Abstract**</summary>

In this work, we propose a new class of learnable optimizers, called Mnemosyne. It is based on the novel spatio-temporal low-rank implicit attention Transformers that can learn to train entire neural network architectures, including other Transformers, without any task-specific optimizer tuning. We show that Mnemosyne: (a) outperforms popular LSTM optimizers (also with new feature engineering to mitigate catastrophic forgetting of LSTMs), (b) can successfully train Transformers while using simple meta-training strategies that require minimal computational resources, (c) matches accuracy-wise SOTA hand-designed optimizers with carefully tuned hyper-parameters (often producing top performing models). Furthermore, Mnemosyne provides space complexity comparable to that of its hand-designed first-order counterparts, which allows it to scale to training larger sets of parameters. We conduct an extensive empirical evaluation of Mnemosyne on: (a) fine-tuning a wide range of Vision Transformers (ViTs) from medium-size architectures to massive ViT-Hs (36 layers, 16 heads), (b) pre-training BERT models and (c) soft prompt-tuning large 11B+ T5XXL models. We complement our results with a comprehensive theoretical analysis of the compact associative memory used by Mnemosyne which we believe was never done before.</details>

### Mobilizing Personalized Federated Learning in Infrastructure-Less and Heterogeneous Environments via Random Walk Stochastic ADMM

**Authors:** Ziba Parsons, Fei Dou, Houyi Du, Zheng Song, Jin Lu

**<details><summary>Abstract**</summary>

This paper explores the challenges of implementing Federated Learning (FL) in practical scenarios featuring isolated nodes with data heterogeneity, which can only be connected to the server through wireless links in an infrastructure-less environment. To overcome these challenges, we propose a novel mobilizing personalized FL approach, which aims to facilitate mobility and resilience. Specifically, we develop a novel optimization algorithm called Random Walk Stochastic Alternating Direction Method of Multipliers (RWSADMM). RWSADMM capitalizes on the server's random movement toward clients and formulates local proximity among their adjacent clients based on hard inequality constraints rather than requiring consensus updates or introducing bias via regularization methods. To mitigate the computational burden on the clients, an efficient stochastic solver of the approximated optimization problem is designed in RWSADMM, which provably converges to the stationary point almost surely in expectation. Our theoretical and empirical results demonstrate the provable fast convergence and substantial accuracy improvements achieved by RWSADMM compared to baseline methods, along with its benefits of reduced communication costs and enhanced scalability.</details>

### Model-free Posterior Sampling via Learning Rate Randomization

**Authors:** Daniil Tiapkin, Denis Belomestny, Daniele Calandriello, Eric Moulines, Remi Munos, Alexey Naumov, Pierre Perrault, Michal Valko, Pierre Ménard

**<details><summary>Abstract**</summary>

In this paper, we introduce Randomized Q-learning (RandQL), a novel randomized model-free algorithm for regret minimization in episodic Markov Decision Processes (MDPs). To the best of our knowledge, RandQL is the first tractable model-free posterior sampling-based algorithm. We analyze the performance of RandQL in both tabular and non-tabular metric space settings. In tabular MDPs, RandQL achieves a regret bound of order $\widetilde{\mathcal{O}}(\sqrt{H^{5}SAT})$, where $H$ is the planning horizon, $S$ is the number of states, $A$ is the number of actions, and $T$ is the number of episodes. For a metric state-action space, RandQL enjoys a regret bound of order $\widetilde{\mathcal{O}}(H^{5/2} T^{(d_z+1)/(d_z+2)})$, where $d_z$ denotes the zooming dimension. Notably, RandQL achieves optimistic exploration without using bonuses, relying instead on a novel idea of learning rate randomization. Our empirical study shows that RandQL outperforms existing approaches on baseline exploration environments.</details>

### Modulated Neural ODEs

**Authors:** Ilze Amanda Auzina, Çağatay Yıldız, Sara Magliacane, Matthias Bethge, Efstratios Gavves

**<details><summary>Abstract**</summary>

Neural ordinary differential equations (NODEs) have been proven useful for learning non-linear dynamics of arbitrary trajectories. However, current NODE methods capture variations across trajectories only via the initial state value or by auto-regressive encoder updates. In this work, we introduce Modulated Neural ODEs (MoNODEs), a novel framework that sets apart dynamics states from underlying static factors of variation and improves the existing NODE methods.  In particular, we introduce *time-invariant modulator variables* that are learned from the data. We incorporate our proposed framework into four existing NODE variants. We test MoNODE on oscillating systems, videos and human walking trajectories, where each trajectory has trajectory-specific modulation. Our framework consistently improves the existing model ability to generalize to new dynamic parameterizations and to perform far-horizon forecasting. In addition, we verify that the proposed modulator variables are informative of the true unknown factors of variation as measured by $R^2$ scores.</details>

### Multi-Player Zero-Sum Markov Games with Networked Separable Interactions

**Authors:** Chanwoo Park, Kaiqing Zhang, Asuman Ozdaglar

**<details><summary>Abstract**</summary>

We study a new class of  Markov games, \textit{(multi-player) zero-sum Markov Games} with {\it Networked separable interactions} (zero-sum NMGs), to model the local interaction structure in non-cooperative multi-agent sequential decision-making. We define a zero-sum NMG as a model where {the payoffs of the auxiliary games associated with each state are zero-sum and} have some separable (i.e., polymatrix) structure across the neighbors over some interaction network. We first identify the necessary and sufficient conditions under which an MG can be presented as a zero-sum NMG, and show that the set of Markov coarse correlated equilibrium (CCE) collapses to the set of Markov Nash equilibrium (NE) in these games, in that the {product of} per-state marginalization of the former for all players yields the latter. Furthermore,  we show that finding approximate Markov \emph{stationary}  CCE  in infinite-horizon discounted zero-sum NMGs is \texttt{PPAD}-hard, unless the underlying network has a ``star topology''. Then, we propose fictitious-play-type dynamics, the classical learning dynamics in normal-form games, for zero-sum NMGs, and establish convergence guarantees to Markov stationary NE under a star-shaped network structure. Finally, in light of the hardness result, we focus on computing a Markov \emph{non-stationary} NE and provide finite-iteration guarantees for a series of value-iteration-based algorithms. We also provide numerical experiments to corroborate our theoretical results.</details>

### Multi-Step Generalized Policy Improvement by Leveraging Approximate Models

**Authors:** Lucas N. Alegre, Ana Bazzan, Ann Nowe, Bruno da Silva

**<details><summary>Abstract**</summary>

We introduce a principled method for performing zero-shot transfer in reinforcement learning (RL) by exploiting approximate models of the environment. Zero-shot transfer in RL has been investigated by leveraging methods rooted in generalized policy improvement (GPI) and successor features (SFs). Although computationally efficient, these methods are model-free: they analyze a library of policies---each solving a particular task---and identify which action the agent should take. We investigate the more general setting where, in addition to a library of policies, the agent has access to an approximate environment model. Even though model-based RL algorithms can identify near-optimal policies, they are typically computationally intensive. We introduce $h$-GPI, a multi-step extension of GPI that interpolates between these extremes---standard model-free GPI and fully model-based planning---as a function of a parameter, $h$, regulating the amount of time the agent has to reason. We prove that $h$-GPI's performance lower bound is strictly better than GPI's, and show that $h$-GPI generally outperforms GPI as $h$ increases. Furthermore, we prove that as $h$ increases, $h$-GPI's performance becomes arbitrarily less susceptible to sub-optimality in the agent's policy library. Finally, we introduce novel bounds characterizing the gains achievable by $h$-GPI as a function of approximation errors in both the agent's policy library and its (possibly learned) model. These bounds strictly generalize those known in the literature. We evaluate $h$-GPI on challenging tabular and continuous-state problems under value function approximation and show that it consistently outperforms GPI and state-of-the-art competing methods under various levels of approximation errors.</details>

### Multiply Robust Federated Estimation of Targeted Average Treatment Effects

**Authors:** Larry Han, Zhu Shen, Jose Zubizarreta

**<details><summary>Abstract**</summary>

Federated or multi-site studies have distinct advantages over single-site studies, including increased generalizability, the ability to study underrepresented populations, and the opportunity to study rare exposures and outcomes. However, these studies are complicated by the need to preserve the privacy of each individual's data, heterogeneity in their covariate distributions, and different data structures between sites. We propose a novel federated approach to derive valid causal inferences for a target population using multi-site data. We adjust for covariate shift and accommodate covariate mismatch between sites by developing a multiply-robust and privacy-preserving nuisance function estimation approach. Our methodology incorporates transfer learning to estimate ensemble weights to combine information from source sites. We show that these learned weights are efficient and optimal under different scenarios. We showcase the finite sample advantages of our approach in terms of efficiency and robustness compared to existing state-of-the-art approaches. We apply our approach to study the treatment effect of percutaneous coronary intervention (PCI) on the duration of hospitalization for patients experiencing acute myocardial infarction (AMI) with data from the Centers for Medicare \& Medicaid Services (CMS).</details>

### Multitask Learning with No Regret: from Improved Confidence Bounds to Active Learning

**Authors:** Pier Giuseppe Sessa, Pierre Laforgue, Nicolò Cesa-Bianchi, Andreas Krause

**<details><summary>Abstract**</summary>

Multitask learning is a powerful framework that enables one to simultaneously learn multiple related tasks by sharing information between them. Quantifying uncertainty in the estimated tasks is of pivotal importance for many downstream applications, such as online or active learning. In this work, we provide novel confidence intervals for multitask regression in the challenging agnostic setting, i.e., when neither the similarity between tasks nor the tasks' features are available to the learner. The obtained intervals do not require i.i.d. data and can be directly applied to bound the regret in online learning. Through a refined analysis of the multitask information gain, we obtain new regret guarantees that, depending on a task similarity parameter, can significantly improve over treating tasks independently. We further propose a novel online learning algorithm that achieves such improved regret without knowing this parameter in advance, i.e., automatically adapting to task similarity. As a second key application of our results, we introduce a novel multitask active learning setup where several tasks must be simultaneously optimized, but only one of them can be queried for feedback by the learner at each round. For this problem, we design a no-regret algorithm that uses our confidence intervals to decide which task should be queried. Finally, we empirically validate our bounds and algorithms on synthetic and real-world (drug discovery) data.</details>

### Mutual-Information Regularized Multi-Agent Policy Iteration

**Authors:** Wang, Deheng Ye, Zongqing Lu

**<details><summary>Abstract**</summary>

Despite the success of cooperative multi-agent reinforcement learning algorithms, most of them focus on a single team composition, which prevents them from being used in more realistic scenarios where dynamic team composition is possible. While some studies attempt to solve this problem via multi-task learning in a fixed set of team compositions, there is still a risk of overfitting to the training set, which may lead to catastrophic performance when facing dramatically varying team compositions during execution. To address this problem, we propose to use mutual information (MI) as an augmented reward to prevent individual policies from relying too much on team-related information and encourage agents to learn policies that are robust in different team compositions. Optimizing this MI-augmented objective in an off-policy manner can be intractable due to the existence of dynamic marginal distribution. To alleviate this problem, we first propose a multi-agent policy iteration algorithm with a fixed marginal distribution and prove its convergence and optimality. Then, we propose to employ the Blahut–Arimoto algorithm and an imaginary team composition distribution for optimization with approximate marginal distribution as the practical implementation. Empirically, our method demonstrates strong zero-shot generalization to dynamic team compositions in complex cooperative tasks.</details>

### NAR-Former V2: Rethinking Transformer for Universal Neural Network Representation Learning

**Authors:** Yun Yi, Haokui Zhang, Rong Xiao, Nannan Wang, Xiaoyu Wang

**<details><summary>Abstract**</summary>

As more deep learning models are being applied in real-world applications, there is a growing need for modeling and learning the representations of neural networks themselves. An effective representation can be used to predict target attributes of networks without the need for actual training and deployment procedures, facilitating efficient network design and deployment. Recently, inspired by the success of Transformer, some Transformer-based representation learning frameworks have been proposed and achieved promising performance in handling cell-structured models. However, graph neural network (GNN) based approaches still dominate the field of learning representation for the entire network. In this paper, we revisit the Transformer and compare it with GNN to analyze their different architectural characteristics. We then propose a modified Transformer-based universal neural network representation learning model NAR-Former V2. It can learn efficient representations from both cell-structured networks and entire networks. Specifically, we first take the network as a graph and design a straightforward tokenizer to encode the network into a sequence. Then, we incorporate the inductive representation learning capability of GNN into Transformer, enabling Transformer to generalize better when encountering unseen architecture. Additionally, we introduce a series of simple yet effective modifications to enhance the ability of the Transformer in learning representation from graph structures. In encoding entire networks and then predicting the latency, our proposed method surpasses the GNN-based method NNLP by a significant margin on the NNLQP dataset. Furthermore, regarding accuracy prediction on the cell-structured NASBench101 and NASBench201 datasets, our method achieves highly comparable performance to other state-of-the-art methods. The code is available at https://github.com/yuny220/NAR-Former-V2.</details>

### NU-MCC: Multiview Compressive Coding with Neighborhood Decoder and Repulsive UDF

**Authors:** Stefan Lionar, Xiangyu Xu, Min Lin, Gim Hee Lee

**<details><summary>Abstract**</summary>

Remarkable progress has been made in 3D reconstruction from single-view RGB-D inputs. MCC is the current state-of-the-art method in this field, which achieves unprecedented success by combining vision Transformers with large-scale training. However, we identified two key limitations of MCC: 1) The Transformer decoder is inefficient in handling large number of query points; 2) The 3D representation struggles to recover high-fidelity details. In this paper, we propose a new approach called NU-MCC that addresses these limitations. NU-MCC includes two key innovations: a Neighborhood decoder and a Repulsive Unsigned Distance Function (Repulsive UDF). First, our Neighborhood decoder introduces center points as an efficient proxy of input visual features, allowing each query point to only attend to a small neighborhood. This design not only results in much faster inference speed but also enables the exploitation of finer-scale visual features for improved recovery of 3D textures. Second, our Repulsive UDF is a novel alternative to the occupancy field used in MCC, significantly improving the quality of 3D object reconstruction. Compared to standard UDFs that suffer from holes in results, our proposed Repulsive UDF can achieve more complete surface reconstruction. Experimental results demonstrate that NU-MCC is able to learn a strong 3D representation, significantly advancing the state of the art in single-view 3D reconstruction. Particularly, it outperforms MCC by 9.7% in terms of the F1-score on the CO3D-v2 dataset with more than 5x faster running speed.</details>

### Nash Regret Guarantees for Linear Bandits

**Authors:** Ayush Sawarni, Soumyabrata Pal, Siddharth Barman

**<details><summary>Abstract**</summary>

We obtain essentially tight upper bounds for a strengthened notion of regret in the stochastic linear bandits framework. The strengthening---referred to as Nash regret---is defined as the difference between the (a priori unknown) optimum and the geometric mean of expected rewards accumulated by the linear bandit algorithm. Since the geometric mean corresponds to the well-studied Nash social welfare (NSW) function, this formulation quantifies the performance of a bandit algorithm as the collective welfare it generates across rounds. NSW is known to satisfy fairness axioms and, hence, an upper bound on Nash regret provides a principled fairness guarantee.    We consider the stochastic linear bandits problem over a horizon of $\mathsf{T}$ rounds and with a set of arms ${\cal X}$ in ambient dimension $d$. Furthermore, we focus on settings in which the stochastic reward---associated with each arm in ${\cal X}$---is a non-negative, sub-Poisson random variable. For this setting, we develop an algorithm that achieves a Nash regret of $O\left( \sqrt{\frac{d}{\mathsf{T}}} \log(\mathsf{T} |{\cal X}|)\right)$. In addition, addressing linear bandit instances in which the set of arms ${\cal X}$ is not necessarily finite, we obtain a Nash regret upper bound of $O\left( \frac{d^\frac{5}{4}}{\sqrt{\mathsf{T}}}  \log(\mathsf{T})\right)$. Since bounded random variables are sub-Poisson, these results hold for bounded, non-negative rewards. Our linear bandit algorithm is built upon the successive elimination method with novel technical insights, including tailored concentration bounds and the use of sampling via John ellipsoid in conjunction with the Kiefer–Wolfowitz optimal design.</details>

### Navigating the Pitfalls of Active Learning Evaluation: A Systematic Framework for Meaningful Performance Assessment

**Authors:** Carsten Lüth, Till Bungert, Lukas Klein, Paul Jaeger

**<details><summary>Abstract**</summary>

Active Learning (AL) aims to reduce the labeling burden by interactively selecting the most informative samples from a pool of unlabeled data. While there has been extensive research on improving AL query methods in recent years, some studies have questioned the effectiveness of AL compared to emerging paradigms such as semi-supervised (Semi-SL) and self-supervised learning (Self-SL), or a simple optimization of classifier configurations. Thus, today’s AL literature presents an inconsistent and contradictory landscape, leaving practitioners uncertain about whether and how to use AL in their tasks. In this work, we make the case that this inconsistency arises from a lack of systematic and realistic evaluation of AL methods. Specifically, we identify five key pitfalls in the current literature that reflect the delicate considerations required for AL evaluation. Further, we present an evaluation framework that overcomes these pitfalls and thus enables meaningful statements about the performance of AL methods. To demonstrate the relevance of our protocol, we present a large-scale empirical study and benchmark for image classification spanning various data sets, query methods, AL settings, and training paradigms. Our findings clarify the inconsistent picture in the literature and enable us to give hands-on recommendations for practitioners. The benchmark is hosted at https://github.com/IML-DKFZ/realistic-al.</details>

### Near-Linear Time Algorithm for the Chamfer Distance

**Authors:** Ainesh Bakshi, Piotr Indyk, Rajesh Jayaram, Sandeep Silwal, Erik Waingarten

**<details><summary>Abstract**</summary>

For any two point sets $A,B \subset \mathbb{R}^d$ of size up to $n$, the Chamfer distance from $A$ to $B$ is defined as $\texttt{CH}(A,B)=\sum_{a \in A} \min_{b \in B} d_X(a,b)$, where $d_X$ is the underlying distance measure (e.g., the Euclidean or Manhattan distance). The Chamfer distance is a popular measure of dissimilarity between point clouds, used in many machine learning, computer vision, and graphics applications, and admits a straightforward $O(d n^2)$-time brute force algorithm. Further, Chamfer distance is often used as a  proxy for the more computationally demanding Earth-Mover (Optimal Transport) Distance. However, the \emph{quadratic} dependence on $n$ in the running time  makes the naive approach intractable for large datasets.We overcome this bottleneck and present the first $(1+\epsilon)$-approximate algorithm for estimating Chamfer distance with a near-linear running time. Specifically, our algorithm runs in time $O(nd \log (n)/\epsilon^2)$ and is implementable. Our experiments demonstrate that it is both accurate and fast on large high-dimensional datasets. We believe that our algorithm will open new avenues for analyzing large high-dimensional point clouds. We also give evidence that if the goal is to report a $(1+\epsilon)$-approximate mapping from $A$ to $B$ (as opposed to just its value), then any sub-quadratic  time algorithm is unlikely to exist.</details>

### Networks are Slacking Off: Understanding Generalization Problem in Image Deraining

**Authors:** Jinjin Gu, Xianzheng Ma, Xiangtao Kong, Yu Qiao, Chao Dong

**<details><summary>Abstract**</summary>

Deep deraining networks consistently encounter substantial generalization issues when deployed in real-world applications, although they are successful in laboratory benchmarks. A prevailing perspective in deep learning encourages using highly complex data for training, with the expectation that richer image background content will facilitate overcoming the generalization problem. However, through comprehensive and systematic experimentation, we discover that this strategy does not enhance the generalization capability of these networks. On the contrary, it exacerbates the tendency of networks to overfit specific degradations. Our experiments reveal that better generalization in a deraining network can be achieved by simplifying the complexity of the training background images. This is because that the networks are ``slacking off'' during training, that is, learning the least complex elements in the image background and degradation to minimize training loss. When the background images are less complex than the rain streaks, the network will prioritize the background reconstruction, thereby suppressing overfitting the rain patterns and leading to improved generalization performance. Our research offers a valuable perspective and methodology for better understanding the generalization problem in low-level vision tasks and displays promising potential for practical application.</details>

### Neural Functional Transformers

**Authors:** Allan Zhou, Kaien Yang, Yiding Jiang, Kaylee Burns, Winnie Xu, Samuel Sokota, J. Zico Kolter, Chelsea Finn

**<details><summary>Abstract**</summary>

The recent success of neural networks as implicit representation of data has driven growing interest in neural functionals: models that can process other neural networks as input by operating directly over their weight spaces. Nevertheless, constructing expressive and efficient neural functional architectures that can handle high-dimensional weight-space objects remains challenging. This paper uses the attention mechanism to define a novel set of permutation equivariant weight-space layers and composes them into deep equivariant models called neural functional Transformers (NFTs). NFTs respect weight-space permutation symmetries while incorporating the advantages of attention, which have exhibited remarkable success across multiple domains. In experiments processing the weights of feedforward MLPs and CNNs, we find that NFTs match or exceed the performance of prior weight-space methods. We also leverage NFTs to develop Inr2Array, a novel method for computing permutation invariant latent representations from the weights of implicit neural representations (INRs). Our proposed method improves INR classification accuracy by up to $+17\%$ over existing methods. We provide an implementation of our layers at https://github.com/AllanYangZhou/nfn.</details>

### Neural Graph Generation from Graph Statistics

**Authors:** Kiarash Zahirnia, Yaochen Hu, Mark Coates, Oliver Schulte

**<details><summary>Abstract**</summary>

We describe a new setting for learning a deep graph generative model (GGM) from aggregate graph statistics, rather than from the graph adjacency matrix. Matching the statistics of observed training graphs is the main approach for learning traditional GGMs (e.g, BTER, Chung-Lu, and  Erdos-Renyi models). Privacy researchers have proposed learning from graph statistics as a way to protect privacy. We develop an architecture for training a deep GGM  to match statistics while preserving local differential privacy guarantees. Empirical evaluation on 8 datasets indicates that our deep GGM model generates more realistic graphs than the traditional GGMs when both are learned from graph statistics only. We also benchmark our deep GGM trained on statistics only, against state-of-the-art deep GGM models that are trained on the entire adjacency matrix. The results show that graph statistics are often sufficient to build a competitive deep GGM that generates realistic graphs while protecting local privacy.</details>

### Neural Image Compression: Generalization, Robustness, and Spectral Biases

**Authors:** Kelsey Lieberman, James Diffenderfer, Charles Godfrey, Bhavya Kailkhura

**<details><summary>Abstract**</summary>

Recent advances in neural image compression (NIC) have produced models that are starting to outperform classic codecs. While this has led to growing excitement about using NIC in real-world applications, the successful adoption of any machine learning system in the wild requires it to generalize (and be robust) to unseen distribution shifts at deployment. Unfortunately, current research lacks comprehensive datasets and informative tools to evaluate and understand NIC performance in real-world settings. To bridge this crucial gap, first, this paper presents a comprehensive benchmark suite to evaluate the out-of-distribution (OOD) performance of image compression methods. Specifically, we provide CLIC-C and Kodak-C by introducing 15 corruptions to the popular CLIC and Kodak benchmarks. Next, we propose spectrally-inspired inspection tools to gain deeper insight into errors introduced by image compression methods as well as their OOD performance. We then carry out a detailed performance comparison of several classic codecs and NIC variants, revealing intriguing findings that challenge our current understanding of the strengths and limitations of NIC. Finally, we corroborate our empirical findings with theoretical analysis, providing an in-depth view of the OOD performance of NIC and its dependence on the spectral properties of the data. Our benchmarks, spectral inspection tools, and findings provide a crucial bridge to the real-world adoption of NIC. We hope that our work will propel future efforts in designing robust and generalizable NIC methods. Code and data will be made available at https://github.com/klieberman/ood_nic.</details>

### Neural Latent Geometry Search: Product Manifold Inference via Gromov-Hausdorff-Informed Bayesian Optimization

**Authors:** Haitz Sáez de Ocáriz Borde, Alvaro Arroyo, Ismael Morales, Ingmar Posner, Xiaowen Dong

**<details><summary>Abstract**</summary>

Recent research indicates that the performance of machine learning models can be improved by aligning the geometry of the latent space with the underlying data structure. Rather than relying solely on Euclidean space, researchers have proposed using hyperbolic and spherical spaces with constant curvature, or combinations thereof, to better model the latent space and enhance model performance. However, little attention has been given to the problem of automatically identifying the optimal latent geometry for the downstream task. We mathematically define this novel formulation and coin it as neural latent geometry search (NLGS). More specifically, we introduce an initial attempt to search for a latent geometry composed of a product of constant curvature model spaces with a small number of query evaluations, under some simplifying assumptions. To accomplish this, we propose a novel notion of distance between candidate latent geometries based on the Gromov-Hausdorff distance from metric geometry. In order to compute the Gromov-Hausdorff distance, we introduce a mapping function that enables the comparison of different manifolds by embedding them in a common high-dimensional ambient space. We then design a graph search space based on the notion of smoothness between latent geometries and employ the calculated distances as an additional inductive bias. Finally, we use Bayesian optimization to search for the optimal latent geometry in a query-efficient manner. This is a general method which can be applied to search for the optimal latent geometry for a variety of models and downstream tasks. We perform experiments on synthetic and real-world datasets to identify the optimal latent geometry for multiple machine learning problems.</details>

### Neural Modulation for Flash Memory: An Unsupervised Learning Framework for Improved Reliability

**Authors:** Jonathan Zedaka, Elisha Halperin, Evgeny Blaichman, Amit Berman

**<details><summary>Abstract**</summary>

Recent years have witnessed a significant increase in the storage density of NAND flash memory, making it a critical component in modern electronic devices. However, with the rise in storage capacity comes an increased likelihood of errors in data storage and retrieval. The growing number of errors poses ongoing challenges for system designers and engineers, in terms of the characterization, modeling, and optimization of NAND-based systems. We present a novel approach for modeling and preventing errors by utilizing the capabilities of generative and unsupervised machine learning methods. As part of our research, we constructed and trained a neural modulator that translates information bits into programming operations on each memory cell in NAND devices. Our modulator, tailored explicitly for flash memory channels, provides a smart writing scheme that reduces programming errors as well as compensates for data degradation over time. Specifically, the modulator is based on an auto-encoder architecture with an additional channel model embedded between the encoder and the decoder. A conditional generative adversarial network (cGAN) was used to construct the channel model. Optimized for the end-of-life work-point, the learned memory system outperforms the prior art by up to 56\% in raw bit error rate (RBER) and extends the lifetime of the flash memory block by up to 25\%.</details>

### Neural Multi-Objective Combinatorial Optimization with Diversity Enhancement

**Authors:** Jinbiao Chen, Zizhen Zhang, Zhiguang Cao, Yaoxin Wu, Yining Ma, Te Ye, Jiahai Wang

**<details><summary>Abstract**</summary>

Most of existing neural methods for multi-objective combinatorial optimization (MOCO) problems solely rely on decomposition, which often leads to repetitive solutions for the respective subproblems, thus a limited Pareto set. Beyond decomposition, we propose a novel neural heuristic with diversity enhancement (NHDE) to produce more Pareto solutions from two perspectives. On the one hand, to hinder duplicated solutions for different subproblems, we propose an indicator-enhanced deep reinforcement learning method to guide the model, and design a heterogeneous graph attention mechanism to capture the relations between the instance graph and the Pareto front graph. On the other hand, to excavate more solutions in the neighborhood of each subproblem, we present a multiple Pareto optima strategy to sample and preserve desirable solutions. Experimental results on classic MOCO problems show that our NHDE is able to generate a Pareto front with higher diversity, thereby achieving superior overall performance. Moreover, our NHDE is generic and can be applied to different neural methods for MOCO.</details>

### Neural Oscillators are Universal

**Authors:** Samuel Lanthaler, T. Konstantin Rusch, Siddhartha Mishra

**<details><summary>Abstract**</summary>

Coupled oscillators are being increasingly used as the basis of machine learning (ML) architectures, for instance in sequence modeling, graph representation learning and in physical neural networks that are used in analog ML devices. We introduce an abstract class of *neural oscillators* that encompasses these architectures and prove that neural oscillators are universal, i.e, they can approximate any continuous and casual operator mapping between time-varying functions, to desired accuracy. This universality result provides theoretical justification for the use of oscillator based ML systems. The proof builds on a fundamental result of independent interest, which shows that a combination of forced harmonic oscillators with a nonlinear read-out suffices to approximate the underlying operators.</details>

### Neural Processes with Stability

**Authors:** Huafeng Liu, Liping Jing, Jian Yu

**<details><summary>Abstract**</summary>

Unlike traditional statistical models depending on hand-specified priors, neural processes (NPs) have recently emerged as a class of powerful neural statistical models that combine the strengths of neural networks and stochastic processes. NPs can define a flexible class of stochastic processes well suited for highly non-trivial functions by encoding contextual knowledge into the function space. However, noisy context points introduce challenges to the algorithmic stability that small changes in training data may significantly change the models and yield lower generalization performance. In this paper, we provide theoretical guidelines for deriving stable solutions with high generalization by introducing the notion of algorithmic stability into NPs, which can be flexible to work with various NPs and achieves less biased approximation with theoretical guarantees. To illustrate the superiority of the proposed model, we perform experiments on both synthetic and real-world data, and the results demonstrate that our approach not only helps to achieve more accurate performance but also improves model robustness.</details>

### Neural-Logic Human-Object Interaction Detection

**Authors:** Liulei Li, Jianan Wei, Wenguan Wang, Yi Yang

**<details><summary>Abstract**</summary>

The interaction decoder utilized in prevalent Transformer-based HOI detectors typically accepts pre-composed human-object pairs as inputs. Though achieving remarkable performance, such a paradigm lacks feasibility and cannot explore novel combinations over entities during decoding. We present LogicHOI, a new HOI detector that leverages neural-logic reasoning and Transformer to infer feasible interactions between. entities. Specifically, we modify. self-attention mechanism in the vanilla Transformer, enabling it to reason over the ⟨ human, action, object ⟩ triplet and constitute novel interactions. Meanwhile, such a reasoning process is guided by two crucial properties for understanding HOI: affordances (the potential actions an object can facilitate) and proxemics (the spatial relations between humans and objects). We formulate these two properties in first-order logic and ground them into continuous space to constrain the learning process of our approach, leading to improved performance and zero-shot generalization capabilities. We evaluate L OGIC HOI on V-COCO and HICO-DET under both normal and zero-shot setups, achieving significant improvements over existing methods.</details>

### NeuralGF: Unsupervised Point Normal Estimation by Learning Neural Gradient Function

**Authors:** Qing Li, Huifang Feng, Kanle Shi, Yue Gao, Yi Fang, Yu-Shen Liu, Zhizhong Han

**<details><summary>Abstract**</summary>

Normal estimation for 3D point clouds is a fundamental task in 3D geometry processing. The state-of-the-art methods rely on priors of fitting local surfaces learned from normal supervision. However, normal supervision in benchmarks comes from synthetic shapes and is usually not available from real scans, thereby limiting the learned priors of these methods. In addition, normal orientation consistency across shapes remains difficult to achieve without a separate post-processing procedure. To resolve these issues, we propose a novel method for estimating oriented normals directly from point clouds without using ground truth normals as supervision. We achieve this by introducing a new paradigm for learning neural gradient functions, which encourages the neural network to fit the input point clouds and yield unit-norm gradients at the points. Specifically, we introduce loss functions to facilitate query points to iteratively reach the moving targets and aggregate onto the approximated surface, thereby learning a global surface representation of the data. Meanwhile, we incorporate gradients into the surface approximation to measure the minimum signed deviation of queries, resulting in a consistent gradient field associated with the surface. These techniques lead to our deep unsupervised oriented normal estimator that is robust to noise, outliers and density variations. Our excellent results on widely used benchmarks demonstrate that our method can learn more accurate normals for both unoriented and oriented normal estimation tasks than the latest methods. The source code and pre-trained model are publicly available.</details>

### Non-autoregressive Machine Translation with Probabilistic Context-free Grammar

**Authors:** Shangtong Gui, Chenze Shao, Zhengrui Ma, xishan zhang, Yunji Chen, Yang Feng

**<details><summary>Abstract**</summary>

Non-autoregressive Transformer(NAT) significantly accelerates the inference of neural machine translation. However, conventional NAT models suffer from limited expression power and performance degradation compared to autoregressive (AT) models due to the assumption of conditional independence among target tokens. To address these limitations, we propose a novel approach called PCFG-NAT, which leverages a specially designed Probabilistic Context-Free Grammar (PCFG) to enhance the ability of NAT models to capture complex dependencies among output tokens. Experimental results on major machine translation benchmarks demonstrate that PCFG-NAT further narrows the gap in translation quality between NAT and AT models. Moreover, PCFG-NAT facilitates a deeper understanding of the generated sentences, addressing the lack of satisfactory explainability in neural machine translation. Code is publicly available at https://github.com/ictnlp/PCFG-NAT.</details>

### Non-stationary Experimental Design under Linear Trends

**Authors:** David Simchi-Levi, Chonghuan Wang, Zeyu Zheng

**<details><summary>Abstract**</summary>

Experimentation has been critical and increasingly popular  across various domains, such as clinical trials and online platforms, due to its widely recognized benefits. One of the primary objectives of classical experiments is to estimate the average treatment effect (ATE) to inform future decision-making. However, in healthcare and many other settings, treatment effects may be non-stationary, meaning that they can change over time, rendering the traditional experimental design inadequate and the classical static ATE uninformative. In this work, we address the problem of non-stationary experimental design under linear trends by considering two objectives: estimating the dynamic treatment effect and minimizing welfare loss within the experiment. We propose an efficient design that can be customized for optimal estimation error rate, optimal regret rate, or the Pareto optimal trade-off between the two objectives. We establish information-theoretical lower bounds that highlight the inherent challenge in estimating dynamic treatment effects and minimizing welfare loss, and also statistically reveal the fundamental trade-off between them.</details>

### Normalization-Equivariant Neural Networks with Application to Image Denoising

**Authors:** Sébastien Herbreteau, Emmanuel Moebel, Charles Kervrann

**<details><summary>Abstract**</summary>

In many information processing systems, it may be desirable to ensure that any change of the input, whether by shifting or  scaling, results in a corresponding change in the system response.  While deep neural networks are gradually replacing all traditional automatic processing methods, they surprisingly do not guarantee such normalization-equivariance (scale + shift) property, which can be detrimental in many applications. To address this issue, we propose a methodology for adapting existing neural networks so that normalization-equivariance holds by design. Our main claim is that not only ordinary convolutional layers, but also all activation functions, including the ReLU (rectified linear unit), which are applied element-wise to the pre-activated neurons, should be completely removed from neural networks and replaced by better conditioned alternatives. To this end, we introduce affine-constrained convolutions and channel-wise sort pooling layers as surrogates and show that these two architectural modifications do preserve normalization-equivariance without loss of performance. Experimental results in image denoising show that normalization-equivariant neural networks, in addition to their better conditioning, also provide much better generalization across noise levels.</details>

### NuTrea: Neural Tree Search for Context-guided Multi-hop KGQA

**Authors:** Hyeong Kyu Choi, Seunghun Lee, Jaewon Chu, Hyunwoo Kim

**<details><summary>Abstract**</summary>

Multi-hop Knowledge Graph Question Answering (KGQA) is a task that involves retrieving nodes from a knowledge graph (KG) to answer natural language questions. Recent GNN-based approaches formulate this task as a KG path searching problem, where messages are sequentially propagated from the seed node towards the answer nodes. However, these messages are past-oriented, and they do not consider the full KG context. To make matters worse, KG nodes often represent pronoun entities and are sometimes encrypted, being uninformative in selecting between paths. To address these problems, we propose Neural Tree Search (NuTrea), a tree search-based GNN model that incorporates the broader KG context. Our model adopts a message-passing scheme that probes the unreached subtree regions to boost the past-oriented embeddings. In addition, we introduce the Relation Frequency-Inverse Entity Frequency (RF-IEF) node embedding that considers the global KG context to better characterize ambiguous KG nodes. The general effectiveness of our approach is demonstrated through experiments on three major multi-hop KGQA benchmark datasets, and our extensive analyses further validate its expressiveness and robustness. Overall, NuTrea provides a powerful means to query the KG with complex natural language questions. Code is available at https://github.com/mlvlab/NuTrea.</details>

### Off-Policy Evaluation for Human Feedback

**Authors:** Qitong Gao, Ge Gao, Juncheng Dong, Vahid Tarokh, Min Chi, Miroslav Pajic

**<details><summary>Abstract**</summary>

Off-policy evaluation (OPE) is important for closing the gap between offline training and evaluation of reinforcement learning (RL), by estimating performance and/or rank of target (evaluation) policies using offline trajectories only. It can improve the safety and efficiency of data collection and policy testing procedures in situations where online deployments are expensive, such as healthcare. However, existing OPE methods fall short in estimating human feedback (HF) signals, as HF may be conditioned over multiple underlying factors and are only sparsely available; as opposed to the agent-defined environmental rewards (used in policy optimization), which are usually determined over parametric functions or distributions. Consequently, the nature of HF signals makes extrapolating accurate OPE estimations to be challenging. To resolve this, we introduce an OPE for HF (OPEHF) framework that revives existing OPE methods in order to accurately evaluate the HF signals. Specifically, we develop an immediate human reward (IHR) reconstruction approach, regularized by environmental knowledge distilled in a latent space that captures the underlying dynamics of state transitions as well as issuing HF signals. Our approach has been tested over *two real-world experiments*, adaptive *in-vivo* neurostimulation and intelligent tutoring, and a simulation environment (visual Q&A). Results show that our approach significantly improves the performance toward estimating HF signals accurately, compared to directly applying (variants of) existing OPE methods.</details>

### On Dynamic Programming Decompositions of Static Risk Measures in Markov Decision Processes

**Authors:** Jia Lin Hau, Erick Delage, Mohammad Ghavamzadeh, Marek Petrik

**<details><summary>Abstract**</summary>

Optimizing static risk-averse objectives in Markov decision processes is difficult because they do not admit standard dynamic programming equations common in Reinforcement Learning (RL) algorithms. Dynamic programming decompositions that augment the state space with discrete risk levels have recently gained popularity in the RL community. Prior work has shown that these decompositions are optimal when the risk level is discretized sufficiently. However, we show that these popular decompositions for Conditional-Value-at-Risk (CVaR) and Entropic-Value-at-Risk (EVaR) are inherently suboptimal regardless of the discretization level. In particular, we show that a saddle point property assumed to hold in prior literature may be violated. However, a decomposition does hold for Value-at-Risk and our proof demonstrates how this risk measure differs from CVaR and EVaR. Our findings are significant because risk-averse algorithms are used in high-stake environments, making their correctness much more critical.</details>

### On Robust Streaming for Learning with Experts: Algorithms and Lower Bounds

**Authors:** David Woodruff, Fred Zhang, Samson Zhou

**<details><summary>Abstract**</summary>

In the online learning with experts problem, an algorithm makes predictions about an outcome on each of $T$ days, given a set of $n$ experts who make predictions on each day. The algorithm is given feedback on the outcomes of each day, including the cost of its prediction and the cost of the expert predictions, and the goal is to make a prediction with the minimum cost, compared to the best expert in hindsight. However, often the predictions made by experts or algorithms at some time influence future outcomes, so that the input is adaptively generated. In this paper, we study robust algorithms for the experts problem under memory constraints. We first give a randomized algorithm that is robust to adaptive inputs that uses $\widetilde{O}\left(\frac{n}{R\sqrt{T}}\right)$ space for  $M=O\left(\frac{R^2 T}{\log^2 n}\right)$, thereby showing a smooth space-regret trade-off. We then show a space lower bound of $\widetilde{\Omega}\left(\frac{nM}{RT}\right)$ for any randomized algorithm that achieves regret $R$ with probability $1-2^{-\Omega(T)}$, when the best expert makes $M$ mistakes. Our result implies that the natural deterministic algorithm, which iterates through pools of experts until each expert in the pool has erred, is optimal up to polylogarithmic factors. Finally, we empirically demonstrate the benefit of using robust procedures against a white-box adversary that has access to the internal state of the algorithm.</details>

### On Single-Index Models beyond Gaussian Data

**Authors:** Aaron Zweig, Loucas PILLAUD-VIVIEN, Joan Bruna

**<details><summary>Abstract**</summary>

Sparse high-dimensional functions have arisen as a rich framework to study the behavior of gradient-descent methods using shallow neural networks, and showcasing its ability to perform feature learning beyond linear models. Amongst those functions, the simplest are single-index models $f(x) = \phi( x \cdot \theta^*)$, where the labels are generated by an arbitrary non-linear link function $\phi$ of an unknown one-dimensional projection $\theta^*$ of the input data. By focusing on Gaussian data, several recent works have built a remarkable picture, where the so-called information exponent (related to the regularity of the link function) controls the required sample complexity. In essence, these tools exploit the stability and spherical symmetry of Gaussian distributions.In this work, we explore extensions of this picture beyond the Gaussian setting, where both stability or symmetry might be violated. Focusing on the planted setting where $\phi$ is known, our main results establish that Stochastic Gradient Descent recovers the unknown direction $\theta^*$ with constant probability in the high-dimensional regime, under mild assumptions that significantly extend ~[Yehudai and Shamir,20].</details>

### On Slicing Optimality for Mutual Information

**Authors:** Ammar Fayad, Majd Ibrahim

**<details><summary>Abstract**</summary>

Measuring dependence between two random variables is of great importance in various domains but is difficult to compute in today's complex environments with high-dimensional data. Recently, slicing methods have shown to be a scalable approach to measuring mutual information (MI) between high-dimensional variables by projecting these variables into one-dimensional spaces. Unfortunately, these methods use uniform distributions of slicing directions, which generally discard informative features between variables and thus lead to inaccurate quantification of dependence. In this paper, we propose a principled framework that searches for an \textit{optimal} distribution of slices for MI. Importantly, we answer theoretical questions about finding the optimal slicing distribution in the context of MI and develop corresponding theoretical analyses. We also develop a practical algorithm, connecting our theoretical results with modern machine learning frameworks. Through comprehensive experiments in benchmark domains, we demonstrate significant gains in our information measure than state-of-the-art baselines.</details>

### On Transfer of Adversarial Robustness from Pretraining to Downstream Tasks

**Authors:** Laura F. Nern, Harsh Raj, Maurice André Georgi, Yash Sharma

**<details><summary>Abstract**</summary>

As large-scale training regimes have gained popularity, the use of pretrained models for downstream tasks has become common practice in machine learning. While pretraining has been shown to enhance the performance of models in practice, the transfer of robustness properties from pretraining to downstream tasks remains poorly understood. In this study, we demonstrate that the robustness of a linear predictor on downstream tasks can be constrained by the robustness of its underlying representation, regardless of the protocol used for pretraining. We prove (i) a bound on the loss that holds independent of any downstream task, as well as (ii) a criterion for robust classification in particular. We validate our theoretical results in practical applications, show how our results can be used for calibrating expectations of downstream robustness, and when our results are useful for optimal transfer learning. Taken together, our results offer an initial step towards characterizing the requirements of the representation function for reliable post-adaptation performance.</details>

### [Spotlight] On quantum backpropagation, information reuse, and cheating measurement collapse

**Authors:** Amira Abbas, Robbie King, Hsin-Yuan Huang, William J. Huggins, Ramis Movassagh, Dar Gilboa, Jarrod McClean

**<details><summary>Abstract**</summary>

The success of modern deep learning hinges on the ability to train neural networks at scale. Through clever reuse of intermediate information, backpropagation facilitates training through gradient computation at a total cost roughly proportional to running the function, rather than incurring an additional factor proportional to the number of parameters -- which can now be in the trillions.  Naively, one expects that quantum measurement collapse entirely rules out the reuse of quantum information as in backpropagation. But recent developments in shadow tomography, which assumes access to multiple copies of a quantum state, have challenged that notion.  Here, we investigate whether parameterized quantum models can train as efficiently as classical neural networks. We show that achieving backpropagation scaling is impossible without access to multiple copies of a state.  With this added ability, we introduce an algorithm with foundations in shadow tomography that matches backpropagation scaling in quantum resources while reducing classical auxiliary computational costs to open problems in shadow tomography. These results highlight the nuance of reusing quantum information for practical purposes and clarify the unique difficulties in training large quantum models, which could alter the course of quantum machine learning.</details>

### On the Adversarial Robustness of Out-of-distribution Generalization Models

**Authors:** Xin Zou, Weiwei Liu

**<details><summary>Abstract**</summary>

Out-of-distribution (OOD) generalization has attracted increasing research attention in recent years, due to its promising experimental results in real-world applications. Interestingly, we find that existing OOD generalization methods are vulnerable to adversarial attacks. This motivates us to study OOD adversarial robustness. We first present theoretical analyses of OOD adversarial robustness in two different complementary settings. Motivated by the theoretical results, we design two algorithms to improve the OOD adversarial robustness. Finally, we conduct experiments to validate the effectiveness of our proposed algorithms.</details>

### On the Asymptotic Learning Curves of Kernel Ridge Regression under Power-law Decay

**Authors:** Yicheng Li, haobo Zhang, Qian Lin

**<details><summary>Abstract**</summary>

The widely observed 'benign overfitting phenomenon' in the neural network literature raises the challenge to the `bias-variance trade-off' doctrine in the statistical learning theory.Since the generalization ability of the 'lazy trained' over-parametrized neural network can be well approximated by that of the neural tangent kernel regression,the curve of the excess risk (namely, the learning curve) of kernel ridge regression attracts increasing attention recently.However, most recent arguments on the learning curve are heuristic and are based on the 'Gaussian design' assumption.In this paper, under mild and more realistic assumptions, we rigorously provide a full characterization of the learning curve in the asymptotic senseunder a power-law decay condition of the eigenvalues of the kernel and also the target function.The learning curve elaborates the effect and the interplay of the choice of the regularization parameter, the source condition and the noise.In particular, our results suggest that the 'benign overfitting phenomenon' exists in over-parametrized neural networks only when the noise level is small.</details>

### On the Complexity of Differentially Private Best-Arm Identification with Fixed Confidence

**Authors:** Achraf Azize, Marc Jourdan, Aymen Al Marjani, Debabrota Basu

**<details><summary>Abstract**</summary>

Best Arm Identification (BAI) problems are progressively used for data-sensitive applications, such as designing adaptive clinical trials, tuning hyper-parameters, and conducting user studies to name a few. Motivated by the data privacy concerns invoked by these applications, we study the problem of BAI with fixed confidence under $\epsilon$-global Differential Privacy (DP). First, to quantify the cost of privacy, we derive a lower bound on the sample complexity of any $\delta$-correct BAI algorithm satisfying $\epsilon$-global DP. Our lower bound suggests the existence of two privacy regimes depending on the privacy budget $\epsilon$. In the high-privacy regime (small $\epsilon$), the hardness depends on a coupled effect of privacy and a novel information-theoretic quantity, called the Total Variation Characteristic Time. In the low-privacy regime (large $\epsilon$), the sample complexity lower bound reduces to the classical non-private lower bound. Second, we propose AdaP-TT, an $\epsilon$-global DP variant of the Top Two algorithm. AdaP-TT runs in *arm-dependent adaptive episodes* and adds *Laplace noise* to ensure a good privacy-utility trade-off. We derive an asymptotic upper bound on the sample complexity of AdaP-TT that matches with the lower bound up to multiplicative constants in the high-privacy regime. Finally, we provide an experimental analysis of AdaP-TT that validates our theoretical results.</details>

### On the Constrained Time-Series Generation Problem

**Authors:** Andrea Coletta, Sriram Gopalakrishnan, Daniel Borrajo, Svitlana Vyetrenko

**<details><summary>Abstract**</summary>

Synthetic time series are often used in practical applications to augment the historical time series dataset, amplify the occurrence of rare events and also create counterfactual scenarios.Distributional-similarity (which we refer to as realism) as well as the satisfaction of certain numerical constraints are common requirements for counterfactual time series generation. For instance, the US Federal Reserve publishes synthetic market stress scenarios given by the constrained time series for financial institutions to assess their performance in hypothetical recessions.Existing approaches for generating constrained time series usually penalize training loss to enforce constraints, and reject non-conforming samples. However, these approaches would require re-training if we change constraints, and rejection sampling can be computationally expensive, or impractical for complex constraints.In this paper, we propose a novel set of methods to tackle the constrained time series generation problem and provide efficient sampling while ensuring the realism of generated time series.  In particular, we frame the problem using a constrained optimization framework and then we propose a set of generative methods including 'GuidedDiffTime', a guided diffusion model. We empirically evaluate our work on several datasets for financial and energy data, where incorporating constraints is critical. We show that our approaches outperform existing work both qualitatively and quantitatively, and that 'GuidedDiffTime' does not require re-training for new constraints, resulting in a significant carbon footprint reduction, up to 92% w.r.t. existing deep learning methods.</details>

### On the Exploration of Local Significant Differences For Two-Sample Test

**Authors:** Zhijian Zhou, Jie Ni, Jia-He Yao, Wei Gao

**<details><summary>Abstract**</summary>

Recent years have witnessed increasing attentions on two-sample test with diverse real applications, while this work takes one more step on the exploration of local significant differences for two-sample test. We propose the ME$_\text{MaBiD}$, an effective test for two-sample testing, and the basic idea is to exploit local information by multiple Mahalanobis kernels and introduce bi-directional hypothesis for testing. On the exploration of local significant differences, we first partition the embedding space into several rectangle regions via a new splitting criterion, which is relevant to test power and data correlation. We then explore local significant differences based on our bi-directional masked $p$-value together with the ME$_\text{MaBiD}$ test. Theoretically, we present the asymptotic distribution and lower bounds of test power for our ME$_\text{MaBiD}$ test, and control the familywise error rate on the exploration of local significant differences. We finally conduct extensive experiments  to validate the effectiveness of our proposed methods on two-sample test and the exploration of local significant differences.</details>

### On the Generalization Error of Stochastic Mirror Descent for Quadratically-Bounded Losses: an Improved Analysis

**Authors:** Ta Duy Nguyen, Alina Ene, Huy Nguyen

**<details><summary>Abstract**</summary>

In this work, we revisit the generalization error of stochastic mirror descent for quadratically bounded losses studied in Telgarsky (2022). Quadratically bounded losses is a broad class of loss functions, capturing both Lipschitz and smooth functions, for both regression and classification problems. We study the high probability generalization for this class of losses on linear predictors in both realizable and non-realizable cases when the data are sampled IID or from a Markov chain. The prior work relies on an intricate coupling argument between the iterates of the original problem and those projected onto a bounded domain. This approach enables blackbox application of concentration inequalities, but also leads to suboptimal guarantees due in part to the use of a union bound across all iterations. In this work, we depart significantly from the prior work of Telgarsky (2022), and introduce a novel approach for establishing high probability generalization guarantees. In contrast to the prior work, our work directly analyzes the moment generating function of a novel supermartingale sequence and leverages the structure of stochastic mirror descent. As a result, we obtain improved bounds in all aforementioned settings. Specifically, in the realizable case and non-realizable case with light-tailed sub-Gaussian data, we improve the bounds by a $\log T$ factor, matching the correct rates of $1/T$ and $1/\sqrt{T}$, respectively. In the more challenging case of heavy-tailed polynomial data, we improve the existing bound by a $\mathrm{poly}\ T$ factor.</details>

### On the Implicit Bias of Linear Equivariant Steerable Networks

**Authors:** Ziyu Chen, Wei Zhu

**<details><summary>Abstract**</summary>

We study the implicit bias of gradient flow on linear equivariant steerable networks in group-invariant binary classification. Our findings reveal that the parameterized predictor converges in direction to the unique group-invariant classifier with a maximum margin defined by the input group action. Under a unitary assumption on the input representation, we establish the equivalence between steerable networks and data augmentation. Furthermore, we demonstrate the improved margin and generalization bound of steerable networks over their non-invariant counterparts.</details>

### On the Overlooked Structure of Stochastic Gradients

**Authors:** Zeke Xie, Qian-Yuan Tang, Mingming Sun, Ping Li

**<details><summary>Abstract**</summary>

Stochastic gradients closely relate to both optimization and generalization of deep neural networks (DNNs). Some works attempted to explain the success of stochastic optimization for deep learning by the arguably heavy-tail properties of gradient noise, while other works presented theoretical and empirical evidence against the heavy-tail hypothesis on gradient noise. Unfortunately, formal statistical tests for analyzing the structure and heavy tails of stochastic gradients in deep learning are still under-explored. In this paper, we mainly make two contributions. First, we conduct formal statistical tests on the distribution of stochastic gradients and gradient noise across both parameters and iterations. Our statistical tests reveal that dimension-wise gradients usually exhibit power-law heavy tails, while iteration-wise gradients and stochastic gradient noise caused by minibatch training usually do not exhibit power-law heavy tails. Second, we further discover that the covariance spectra of stochastic gradients have the power-law structures overlooked by previous studies and present its theoretical implications for training of DNNs. While previous studies believed that the anisotropic structure of stochastic gradients matters to deep learning, they did not expect the gradient covariance can have such an elegant mathematical structure. Our work challenges the existing belief and provides novel insights on the structure of stochastic gradients in deep learning.</details>

### On the Pareto Front of Multilingual Neural Machine Translation

**Authors:** Liang Chen, Shuming Ma, Dongdong Zhang, Furu Wei, Baobao Chang

**<details><summary>Abstract**</summary>

In this work, we study how the performance of a given direction changes with its sampling ratio in Multilingual Neural Machine Translation (MNMT). By training over 200 multilingual models with various model sizes, data sizes, and language directions, we find it interesting that the performance of certain translation direction does not always improve with the increase of its weight in the multi-task optimization objective. Accordingly, scalarization method leads to a multitask trade-off front that deviates from the traditional Pareto front when there exists data imbalance in the training corpus, which poses a great challenge to improve the overall performance of all directions. Based on our observations, we propose the Double Power Law to predict the unique performance trade-off front in MNMT, which is robust across various languages, data adequacy, and the number of tasks. Finally, we formulate the sample ratio selection problem in MNMT as an optimization problem based on the Double Power Law. Extensive experiments show that it achieves better performance than temperature searching and gradient manipulation methods with only 1/5 to 1/2 of the total training budget. We release the code at https://github.com/pkunlp-icler/ParetoMNMT for reproduction.</details>

### On the Stability-Plasticity Dilemma in Continual Meta-Learning: Theory and Algorithm

**Authors:** Qi CHEN, Changjian Shui, Ligong Han, Mario Marchand

**<details><summary>Abstract**</summary>

We focus on Continual Meta-Learning (CML), which targets accumulating and exploiting meta-knowledge on a sequence of non-i.i.d. tasks. The primary challenge is to strike a balance between stability and plasticity, where a model should be stable to avoid catastrophic forgetting in previous tasks and plastic to learn generalizable concepts from new tasks. To address this, we formulate the CML objective as controlling the average excess risk upper bound of the task sequence, which reflects the trade-off between forgetting and generalization. Based on the objective, we introduce a unified theoretical framework for CML in both static and shifting environments, providing guarantees for various task-specific learning algorithms. Moreover, we first present a rigorous analysis of a bi-level trade-off in shifting environments. To approach the optimal trade-off, we propose a novel algorithm that dynamically adjusts the meta-parameter and its learning rate w.r.t environment change. Empirical evaluations on synthetic and real datasets illustrate the effectiveness of the proposed theory and algorithm.</details>

### On the Statistical Consistency of Risk-Sensitive Bayesian Decision-Making

**Authors:** Prateek Jaiswal, Harsha Honnappa, Vinayak Rao

**<details><summary>Abstract**</summary>

We study data-driven decision-making problems in the Bayesian framework, where the expectation in the Bayes risk is replaced by a risk-sensitive entropic risk measure with respect to the posterior distribution. We focus on problems where calculating the posterior distribution is intractable, a typical situation in modern applications with large datasets and complex data generating models. We leverage a dual representation of the entropic risk measure to introduce a novel risk-sensitive variational Bayesian (RSVB) framework for jointly computing a risk-sensitive posterior approximation and the corresponding decision rule. Our general framework includes \textit{loss-calibrated} VB (Lacoste-Julien et al. [2011] ) as a special case. We also study the impact of these computational approximations on the predictive performance of the inferred decision rules. We compute the convergence rates of the RSVB approximate posterior and the corresponding optimal value. We illustrate our theoretical findings in parametric and nonparametric settings with the help of three examples.</details>

### On the Trade-off of Intra-/Inter-class Diversity for Supervised Pre-training

**Authors:** Jieyu Zhang, Bohan Wang, Zhengyu Hu, Pang Wei Koh, Alexander Ratner

**<details><summary>Abstract**</summary>

Pre-training datasets are critical for building state-of-the-art machine learning models, motivating rigorous study on their impact on downstream tasks. In this work, we study the impact of the trade-off between the intra-class diversity (the number of samples per class) and the inter-class diversity (the number of classes) of a supervised pre-training dataset. Empirically, we found that with the size of the pre-training dataset fixed, the best downstream performance comes with a balance on the intra-/inter-class diversity. To understand the underlying mechanism, we show theoretically that the downstream performance depends monotonically on both types of diversity. Notably, our theory reveals that the optimal class-to-sample ratio (#classes / #samples per class) is invariant to the size of the pre-training dataset, which motivates an application of predicting the optimal number of pre-training classes. We demonstrate the effectiveness of this application by an improvement of around 2 points on the downstream tasks when using ImageNet as the pre-training dataset.</details>

### On the spectral bias of two-layer linear networks

**Authors:** Aditya  Vardhan Varre, Maria-Luiza Vladarean, Loucas PILLAUD-VIVIEN, Nicolas Flammarion

**<details><summary>Abstract**</summary>

This paper studies the behaviour of two-layer fully connected networks with linear activations trained with gradient flow on the square loss. We show how the optimization process carries an implicit bias on the parameters that depends on the scale of its initialization.  The main result of the paper is a variational characterization of the loss minimizers retrieved by the gradient flow for a specific initialization shape. This characterization reveals that, in the small scale initialization regime, the linear neural network's hidden layer is biased toward having a low-rank structure. To complement our results, we showcase a hidden mirror flow that tracks the dynamics of the singular values of the weights matrices and describe their time evolution. We support our findings with numerical experiments illustrating the phenomena.</details>

### One Less Reason for Filter Pruning: Gaining Free Adversarial Robustness with Structured Grouped Kernel Pruning

**Authors:** Shaochen (Henry) Zhong, Zaichuan You, Jiamu Zhang, Sebastian Zhao, Zachary LeClaire, Zirui Liu, Daochen Zha, Vipin Chaudhary, Shuai Xu, Xia Hu

**<details><summary>Abstract**</summary>

Densely structured pruning methods utilizing simple pruning heuristics can deliver immediate compression and acceleration benefits with acceptable benign performances. However, empirical findings indicate such naively pruned networks are extremely fragile under simple adversarial attacks. Naturally, we would be interested in knowing if such a phenomenon also holds for carefully designed modern structured pruning methods. If so, then to what extent is the severity? And what kind of remedies are available? Unfortunately, both the questions and the solution remain largely unaddressed: no prior art is able to provide a thorough investigation on the adversarial performance of modern structured pruning methods (spoiler: it is not good), yet the few works that attempt to provide mitigation often do so at various extra costs with only to-be-desired performance.In this work, we answer both questions by fairly and comprehensively investigating the adversarial performance of 10+ popular structured pruning methods. Solution-wise, we take advantage of *Grouped Kernel Pruning (GKP)*'s recent success in pushing densely structured pruning freedom to a more fine-grained level. By mixing up kernel smoothness — a classic robustness-related kernel-level metric — into a modified GKP procedure, we present a one-shot-post-train-weight-dependent GKP method capable of advancing SOTA performance on both the benign and adversarial scale, while requiring no extra (in fact, often less) cost than a standard pruning procedure. Please refer to our [GitHub repository](https://github.com/henryzhongsc/adv_robust_gkp) for code implementation, tool sharing, and model checkpoints.</details>

### OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling

**Authors:** yifan zhang, Qingsong Wen, xue wang, Weiqi Chen, Liang Sun, Zhang Zhang, Liang Wang, Rong Jin, Tieniu Tan

**<details><summary>Abstract**</summary>

Online updating of time series forecasting models aims to address the concept drifting problem by efficiently updating forecasting models based on streaming data. Many algorithms are designed for online time series forecasting, with some exploiting cross-variable dependency while others assume independence among variables. Given every data assumption has its own pros and cons in online time series modeling, we propose **On**line **e**nsembling **Net**work (**OneNet**). It dynamically updates and combines two models, with one focusing on modeling the dependency across the time dimension and the other on cross-variate dependency. Our method incorporates a reinforcement learning-based approach into the traditional online convex programming framework, allowing for the linear combination of the two models with dynamically adjusted weights. OneNet addresses the main shortcoming of classical online learning methods that tend to be slow in adapting to the concept drift. Empirical results show that OneNet reduces online forecasting error by more than $\mathbf{50}\\%$ compared to the State-Of-The-Art (SOTA) method.</details>

### [Spotlight] Online Label Shift: Optimal Dynamic Regret meets Practical Algorithms

**Authors:** Dheeraj Baby, Saurabh Garg, Tzu-Ching Yen, Sivaraman Balakrishnan, Zachary Lipton, Yu-Xiang Wang

**<details><summary>Abstract**</summary>

This paper focuses on supervised and unsupervised online label shift,where the class marginals $Q(y)$ variesbut the class-conditionals $Q(x|y)$ remain invariant. In the unsupervised setting, our goal is to adapt a learner, trained on some offline labeled data, to changing label distributions given unlabeled online data. In the supervised setting, we must both learn a classifier and adapt to the dynamically evolving class marginals given only labeled online data. We develop novel algorithms that reduce the adaptation problem to online regression and guarantee optimal dynamic regret without any prior knowledge of the extent of drift in the label distribution. Our solution is based on bootstrapping the estimates of *online regression oracles* that track the drifting proportions. Experiments across numerous simulated and real-world online label shift scenarios demonstrate the superior performance of our proposed approaches, often achieving 1-3% improvement in accuracy while being sample and computationally efficient. Code is publicly available at https://github.com/Anon-djiwh/OnlineLabelShift</details>

### Online Learning under Adversarial Nonlinear Constraints

**Authors:** Pavel Kolev, Georg Martius, Michael Muehlebach

**<details><summary>Abstract**</summary>

In many applications, learning systems are required to process continuous non-stationary data streams.We study this problem in an online learning framework and propose an algorithm that can deal with adversarial time-varying and nonlinear constraints.As we show in our work, the algorithm called Constraint Violation Velocity Projection (CVV-Pro) achieves $\sqrt{T}$ regret and converges to the feasible set at a rate of $1/\sqrt{T}$, despite the fact that the feasible set is slowly time-varying and a priori unknown to the learner. CVV-Pro only relies on local sparse linear approximations of the feasible set and therefore avoids optimizing over the entire set at each iteration, which is in sharp contrast to projected gradients or Frank-Wolfe methods. We also empirically evaluate our algorithm on two-player games, where the players are subjected to a shared constraint.</details>

### Online Performative Gradient Descent for Learning Nash Equilibria in Decision-Dependent Games

**Authors:** Zihan Zhu, Ethan Fang, Zhuoran Yang

**<details><summary>Abstract**</summary>

We study the multi-agent game within the innovative framework of decision-dependent games, which establishes a feedback mechanism that population data reacts to agents’ actions and further characterizes the strategic interactions between agents. We focus on finding the Nash equilibrium of decision-dependent games in the bandit feedback setting. However, since agents are strategically coupled, traditional gradient-based methods are infeasible without the gradient oracle. To overcome this challenge, we model the strategic interactions by a general parametric model and propose a novel online algorithm, Online Performative Gradient Descent (OPGD), which leverages the ideas of online stochastic approximation and projected gradient descent to learn the Nash equilibrium in the context of function approximation for the unknown gradient. In particular, under mild assumptions on the function classes defined in the parametric model, we prove that OPGD can find the Nash equilibrium efficiently for strongly monotone decision-dependent games. Synthetic numerical experiments validate our theory.</details>

### Open-Vocabulary Semantic Segmentation via Attribute Decomposition-Aggregation

**Authors:** Chaofan Ma, Yang Yuhuan, Chen Ju, Fei Zhang, Ya Zhang, Yanfeng Wang

**<details><summary>Abstract**</summary>

Open-vocabulary semantic segmentation is a challenging task that requires segmenting novel object categories at inference time. Recent works explore vision-language pre-training to handle this task, but suffer from unrealistic assumptions in practical scenarios, \ie, low-quality textual category names.For example, this paradigm assumes that new textual categories will be accurately and completely provided, and exist in lexicons during pre-training.However, exceptions often happen when meet with ambiguity for brief or incomplete names, new words that are not present in the pre-trained lexicons, and difficult-to-describe categories for users.To address these issues, this work proposes a novel attribute decomposition-aggregation framework, inspired by human cognition in understanding new concepts. Specifically, in the decomposition stage, we decouple class names into diverse attribute descriptions to complement semantic contexts from multiple perspectives.Two attribute construction strategies are designed: using large language models for common categories, and involving manually labelling for human-invented categories. In the aggregation stage, we group diverse attributes into an integrated global description, to form a discriminative classifier that distinguishes the target object from others. One hierarchical aggregation architecture is further proposed to achieve multi-level aggregation, leveraging the meticulously designed clustering module.The final result is obtained by computing the similarity between aggregated attributes and images embedding.To evaluate the effectiveness, we annotate three datasets with attribute descriptions, and conduct extensive experiments and ablation studies. The results show the superior performance of attribute decomposition-aggregation.We refer readers to the latest arXiv version at https://arxiv.org/abs/2309.00096 .</details>

### OpenGSL: A Comprehensive Benchmark for Graph Structure Learning

**Authors:** Zhou Zhiyao, Sheng Zhou, Bochao Mao, Xuanyi Zhou, Jiawei Chen, Qiaoyu Tan, Daochen Zha, Yan Feng, Chun Chen, Can Wang

**<details><summary>Abstract**</summary>

Graph Neural Networks (GNNs) have emerged as the *de facto* standard for representation learning on graphs, owing to their ability to effectively integrate graph topology and node attributes. However, the inherent suboptimal nature of node connections, resulting from the complex and contingent formation process of graphs, presents significant challenges in modeling them effectively. To tackle this issue, Graph Structure Learning (GSL), a family of data-centric learning approaches, has garnered substantial attention in recent years. The core concept behind GSL is to jointly optimize the graph structure and the corresponding GNN models. Despite the proposal of numerous GSL methods, the progress in this field remains unclear due to inconsistent experimental protocols, including variations in datasets, data processing techniques, and splitting strategies. In this paper, we introduce OpenGSL, the first comprehensive benchmark for GSL, aimed at addressing this gap. OpenGSL enables a fair comparison among state-of-the-art GSL methods by evaluating them across various popular datasets using uniform data processing and splitting strategies. Through extensive experiments, we observe that existing GSL methods do not consistently outperform vanilla GNN counterparts. We also find that there is no significant correlation between the homophily of the learned structure and task performance, challenging the common belief. Moreover, we observe that the learned graph structure demonstrates a strong generalization ability across different GNN models, despite the high computational and space consumption. We hope that our open-sourced library will facilitate rapid and equitable evaluation and inspire further innovative research in this field. The code of the benchmark can be found in https://github.com/OpenGSL/OpenGSL.</details>

### OpenShape: Scaling Up 3D Shape Representation Towards Open-World Understanding

**Authors:** Minghua Liu, Ruoxi Shi, Kaiming Kuang, Yinhao Zhu, Xuanlin Li, Shizhong Han, Hong Cai, Fatih Porikli, Hao Su

**<details><summary>Abstract**</summary>

We introduce OpenShape, a method for learning multi-modal joint representations of text, image, and point clouds. We adopt the commonly used multi-modal contrastive learning framework for representation alignment, but with a specific focus on scaling up 3D representations to enable open-world 3D shape understanding. To achieve this, we scale up training data by ensembling multiple 3D datasets and propose several strategies to automatically filter and enrich noisy text descriptions. We also explore and compare strategies for scaling 3D backbone networks and introduce a novel hard negative mining module for more efficient training. We evaluate OpenShape on zero-shot 3D classification benchmarks and demonstrate its superior capabilities for open-world recognition. Specifically, OpenShape achieves a zero-shot accuracy of 46.8% on the 1,156-category Objaverse-LVIS benchmark, compared to less than 10% for existing methods. OpenShape also achieves an accuracy of 85.3% on ModelNet40, outperforming previous zero-shot baseline methods by 20% and performing on par with some fully-supervised methods. Furthermore, we show that our learned embeddings encode a wide range of visual and semantic concepts (e.g., subcategories, color, shape, style) and facilitate fine-grained text-3D and image-3D interactions. Due to their alignment with CLIP embeddings, our learned shape representations can also be integrated with off-the-shelf CLIP-based models for various applications, such as point cloud captioning and point cloud-conditioned image generation.</details>

### Opening the Vocabulary of Egocentric Actions

**Authors:** Dibyadip Chatterjee, Fadime Sener, Shugao Ma, Angela Yao

**<details><summary>Abstract**</summary>

Human actions in egocentric videos often feature hand-object interactions composed of a verb (performed by the hand) applied to an object. Despite their extensive scaling up, egocentric datasets still face two limitations — sparsity of action compositions and a closed set of interacting objects. This paper proposes a novel open vocabulary action recognition task. Given a set of verbs and objects observed during training, the goal is to generalize the verbs to an open vocabulary of actions with seen and novel objects. To this end, we decouple the verb and object predictions via an object-agnostic _verb encoder_ and a prompt-based _object encoder_. The prompting leverages CLIP representations to predict an open vocabulary of interacting objects. We create open vocabulary benchmarks on the EPIC-KITCHENS-100 and Assembly101 datasets; whereas closed-action methods fail to generalize, our proposed method is effective. In addition, our object encoder significantly outperforms existing open-vocabulary visual recognition methods in recognizing novel interacting objects.</details>

### Optimal Preconditioning and Fisher Adaptive Langevin Sampling

**Authors:** Michalis Titsias

**<details><summary>Abstract**</summary>

We define  an optimal preconditioning for the Langevin diffusion by analytically optimizing the expected squared jumped distance. This yields as the optimal preconditioning an inverse Fisher information covariance matrix, where the covariance matrix is computed as the outer product of log target gradients averaged under the target.  We apply this result to the Metropolis adjusted Langevin algorithm (MALA)  and derive a computationally efficient adaptive MCMC scheme that learns the preconditioning from the history of gradients produced as the algorithm runs. We show in several experiments that the proposed algorithm is very robust in high dimensions and significantly outperforms other methods, including a closely related adaptive MALA scheme that learns the preconditioning with standard adaptive MCMC as well as the position-dependent  Riemannian manifold MALA sampler.</details>

### Optimal Rates for Bandit Nonstochastic Control

**Authors:** Y. Jennifer Sun, Stephen Newman, Elad Hazan

**<details><summary>Abstract**</summary>

Linear Quadratic Regulator (LQR) and Linear Quadratic Gaussian (LQG) control are foundational and extensively researched problems in optimal control. We investigate LQR and LQG problems with semi-adversarial perturbations and time-varying adversarial bandit loss functions. The best-known sublinear regret algorithm~\cite{gradu2020non} has a $T^{\frac{3}{4}}$ time horizon dependence, and its authors posed an open question about whether a tight rate of $\sqrt{T}$ could be achieved. We answer in the affirmative, giving an algorithm for bandit LQR and LQG which attains optimal regret, up to logarithmic factors. A central component of our method is a new scheme for bandit convex optimization with memory, which is of independent interest.</details>

### Optimal Transport for Treatment Effect Estimation

**Authors:** Hao Wang, Jiajun Fan, Zhichao Chen, Haoxuan Li, Weiming Liu, Tianqiao Liu, Quanyu Dai, Yichao Wang, Zhenhua Dong, Ruiming Tang

**<details><summary>Abstract**</summary>

Estimating individual treatment effects from observational data is challenging due to treatment selection bias. Prevalent methods mainly mitigate this issue by aligning different treatment groups in the latent space, the core of which is the calculation of distribution discrepancy. However, two issues that are often overlooked can render these methods invalid:(1) mini-batch sampling effects (MSE), where the calculated discrepancy is erroneous in non-ideal mini-batches with outcome imbalance and outliers;(2) unobserved confounder effects (UCE), where the unobserved confounders are not considered in the discrepancy calculation.Both of these issues invalidate the calculated discrepancy, mislead the training of estimators, and thus impede the handling of treatment selection bias.To tackle these issues, we propose Entire Space CounterFactual Regression (ESCFR), which is a new take on optimal transport technology in the context of causality.Specifically, based on the canonical optimal transport framework, we propose a relaxed mass-preserving regularizer to address the MSE issue and design a proximal factual outcome regularizer to handle the UCE issue.Extensive experiments demonstrate that ESCFR estimates distribution discrepancy accurately, handles the treatment selection bias effectively, and outperforms prevalent competitors significantly.</details>

### Optimal Treatment Allocation for Efficient Policy Evaluation in Sequential Decision Making

**Authors:** Ting Li, Chengchun Shi, Jianing Wang, Fan Zhou, hongtu zhu

**<details><summary>Abstract**</summary>

A/B testing is critical for modern technological companies to evaluate the effectiveness of newly developed products against standard baselines. This paper studies optimal designs that aim to maximize the amount of information obtained from online experiments to estimate treatment effects accurately. We propose three optimal allocation strategies in a dynamic setting where treatments are sequentially assigned over time. These strategies are designed to minimize the variance of the treatment effect estimator when data follow a non Markov decision process or a (time-varying) Markov decision process. We further develop estimation procedures based on existing off-policy evaluation (OPE) methods and conduct extensive experiments in various environments to demonstrate the effectiveness of the proposed methodologies. In theory, we prove the optimality of the proposed treatment allocation design and establish upper bounds for the mean squared errors of the resulting treatment effect estimators.</details>

### Optimization of Inter-group criteria for clustering with minimum size constraints

**Authors:** Eduardo Laber, Lucas Murtinho

**<details><summary>Abstract**</summary>

Internal measures that are used to assess the quality of a clustering usually take into account intra-group and/or inter-group criteria.There are many papers in the literature that propose algorithms with provable approximation guarantees for optimizing the former. However,  the optimization of inter-group criteria is much less understood.Here, we contribute to the state-of-the-art of this literature by devising algorithms with provable guarantees for the maximization of two natural inter-group criteria, namely the minimum spacing and the minimum spanning tree spacing. The former is the minimum distance between points in different groups while the latter captures separability through the cost of the minimum spanning tree that connects all groups. We obtain results for both the unrestricted case, in which no constraint on the clusters is imposed, and for the constrained case where each group is required to have a minimum number of points. Our constraint is motivated by the fact that the popular Single-Linkage, which optimizes both criteria in the unrestricted case, produces clustering with many tiny groups.To complement our work, we present an empirical study with 10 real datasets that provides evidence that our methods work very well in practical settings.</details>

### Optimization or Architecture: How to Hack Kalman Filtering

**Authors:** Ido Greenberg, Netanel Yannay, Shie Mannor

**<details><summary>Abstract**</summary>

In non-linear filtering, it is traditional to compare non-linear architectures such as neural networks to the standard linear Kalman Filter (KF). We observe that this mixes the evaluation of two separate components: the non-linear architecture, and the parameters optimization method. In particular, the non-linear model is often optimized, whereas the reference KF model is not. We argue that both should be optimized similarly, and to that end present the Optimized KF (OKF). We demonstrate that the KF may become competitive to neural models – if optimized using OKF. This implies that experimental conclusions of certain previous studies were derived from a flawed process. The advantage of OKF over the standard KF is further studied theoretically and empirically, in a variety of problems. Conveniently, OKF can replace the KF in real-world systems by merely updating the parameters.</details>

### Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal

**Authors:** Leah Chrestien, Stefan Edelkamp, Antonin Komenda, Tomas Pevny

**<details><summary>Abstract**</summary>

In imitation learning for planning, parameters of heuristic functions are optimized against a set of solved problem instances. This work revisits the necessary and sufficient conditions of strictly optimally efficient heuristics for forward search algorithms, mainly A* and greedy best-first search, which expand only states on the returned optimal path. It then proposes a family of loss functions based on ranking tailored for a given variant of the forward search algorithm. Furthermore, from a learning theory point of view, it discusses why optimizing cost-to-goal h* is unnecessarily difficult. The experimental comparison on a diverse set of problems unequivocally supports the derived theory.</details>

### Overcoming Recency Bias of Normalization Statistics in Continual Learning: Balance and Adaptation

**Authors:** Yilin Lyu, Liyuan Wang, Xingxing Zhang, Zicheng Sun, Hang Su, Jun Zhu, Liping Jing

**<details><summary>Abstract**</summary>

Continual learning entails learning a sequence of tasks and balancing their knowledge appropriately. With limited access to old training samples, much of the current work in deep neural networks has focused on overcoming catastrophic forgetting of old tasks in gradient-based optimization. However, the normalization layers provide an exception, as they are updated interdependently by the gradient and statistics of currently observed training samples, which require specialized strategies to mitigate recency bias. In this work, we focus on the most popular Batch Normalization (BN) and provide an in-depth theoretical analysis of its sub-optimality in continual learning. Our analysis demonstrates the dilemma between balance and adaptation of BN statistics for incremental tasks, which potentially affects training stability and generalization. Targeting on these particular challenges, we propose Adaptive Balance of BN (AdaB$^2$N), which incorporates appropriately a Bayesian-based strategy to adapt task-wise contributions and a modified momentum to balance BN statistics, corresponding to the training and testing stages. By implementing BN in a continual learning fashion, our approach achieves significant performance gains across a wide range of benchmarks, particularly for the challenging yet realistic online scenarios (e.g., up to 7.6\%, 6.86\% and 4.26\% on Split CIFAR-10, Split CIFAR-100 and Split Mini-ImageNet, respectively). Our code is available at https://github.com/lvyilin/AdaB2N.</details>

### [Spotlight] PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers

**Authors:** Phillip Lippe, Bas Veeling, Paris Perdikaris, Richard Turner, Johannes Brandstetter

**<details><summary>Abstract**</summary>

Time-dependent partial differential equations (PDEs) are ubiquitous in science and engineering. Recently, mostly due to the high computational cost of traditional solution techniques, deep neural network based surrogates have gained increased interest. The practical utility of such neural PDE solvers relies on their ability to provide accurate, stable predictions over long time horizons, which is a notoriously hard problem. In this work, we present a large-scale analysis of common temporal rollout strategies, identifying the neglect of non-dominant spatial frequency information, often associated with high frequencies in PDE solutions, as the primary pitfall limiting stable, accurate rollout performance.  Based on these insights, we draw inspiration from recent advances in diffusion models to introduce PDE-Refiner; a novel model class that enables more accurate modeling of all frequency components via a multistep refinement process. We validate PDE-Refiner on challenging benchmarks of complex fluid dynamics, demonstrating stable and accurate rollouts that consistently outperform state-of-the-art models, including neural, numerical, and hybrid neural-numerical architectures. We further demonstrate that PDE-Refiner greatly enhances data efficiency, since the denoising objective implicitly induces a novel form of spectral data augmentation. Finally, PDE-Refiner's connection to diffusion models enables an accurate and efficient assessment of the model's predictive uncertainty, allowing us to estimate when the surrogate becomes inaccurate.</details>

### PICProp: Physics-Informed Confidence Propagation for Uncertainty Quantification

**Authors:** Qianli Shen, Wai Hoh Tang, Zhun Deng, Apostolos Psaros, Kenji Kawaguchi

**<details><summary>Abstract**</summary>

Standard approaches for uncertainty quantification in deep learning and physics-informed learning have persistent limitations. Indicatively, strong assumptions regarding the data likelihood are required, the performance highly depends on the selection of priors, and the posterior can be sampled only approximately, which leads to poor approximations because of the associated computational cost.This paper introduces and studies confidence interval (CI) estimation for deterministic partial differential equations as a novel problem.That is, to propagate confidence, in the form of CIs, from data locations to the entire domain with probabilistic guarantees.We propose a method, termed Physics-Informed Confidence Propagation (PICProp), based on bi-level optimization to compute a valid CI without making heavy assumptions.We provide a theorem regarding the validity of our method, and computational experiments, where the focus is on physics-informed learning. Code is available at https://github.com/ShenQianli/PICProp.</details>

### PaintSeg: Painting Pixels for Training-free Segmentation

**Authors:** Xiang Li, Chung-Ching Lin, Yinpeng Chen, Zicheng Liu, Jinglu Wang, Rita Singh, Bhiksha Raj

**<details><summary>Abstract**</summary>

The paper introduces PaintSeg, a new unsupervised method for segmenting objects without any training. We propose an adversarial masked contrastive painting (AMCP) process, which creates a contrast between the original image and a painted image in which a masked area is painted using off-the-shelf generative models. During the painting process, inpainting and outpainting are alternated, with the former masking the foreground and filling in the background, and the latter masking the background while recovering the missing part of the foreground object. Inpainting and outpainting, also referred to as I-step and O-step, allow our method to gradually advance the target segmentation mask toward the ground truth without supervision or training. PaintSeg can be configured to work with a variety of prompts, e.g. coarse masks, boxes, scribbles, and points. Our experimental results demonstrate that PaintSeg outperforms existing approaches in coarse mask-prompt, box-prompt, and point-prompt segmentation tasks, providing a training-free solution suitable for unsupervised segmentation. Code: https://github.com/lxa9867/PaintSeg.</details>

### Parameterizing Non-Parametric Meta-Reinforcement Learning Tasks via Subtask Decomposition

**Authors:** Suyoung Lee, Myungsik Cho, Youngchul Sung

**<details><summary>Abstract**</summary>

Meta-reinforcement learning (meta-RL) techniques have demonstrated remarkable success in generalizing deep reinforcement learning across a range of tasks. Nevertheless, these methods often struggle to generalize beyond tasks with parametric variations. To overcome this challenge, we propose Subtask Decomposition and Virtual Training (SDVT), a novel meta-RL approach that decomposes each non-parametric task into a collection of elementary subtasks and parameterizes the task based on its decomposition. We employ a  Gaussian mixture VAE to meta-learn the decomposition process, enabling the agent to reuse policies acquired from common subtasks. Additionally, we propose a virtual training procedure, specifically designed for non-parametric task variability, which generates hypothetical subtask compositions, thereby enhancing generalization to previously unseen subtask compositions. Our method significantly improves performance on the Meta-World ML-10 and ML-45 benchmarks, surpassing current state-of-the-art techniques.</details>

### Partial Multi-Label Learning with Probabilistic Graphical Disambiguation

**Authors:** Jun-Yi Hang, Min-Ling Zhang

**<details><summary>Abstract**</summary>

In partial multi-label learning (PML), each training example is associated with a set of candidate labels, among which only some labels are valid. As a common strategy to tackle PML problem, disambiguation aims to recover the ground-truth labeling information from such inaccurate annotations. However, existing approaches mainly rely on heuristics or ad-hoc rules to disambiguate candidate labels, which may not be universal enough in complicated real-world scenarios. To provide a principled way for disambiguation, we make a first attempt to explore the probabilistic graphical model for PML problem, where a directed graph is tailored to infer latent ground-truth labeling information from the generative process of partial multi-label data. Under the framework of stochastic gradient variational Bayes, a unified variational lower bound is derived for this graphical model, which is further relaxed probabilistically so that the desired prediction model can be induced with simultaneously identified ground-truth labeling information. Comprehensive experiments on multiple synthetic and real-world data sets show that our approach outperforms the state-of-the-art counterparts.</details>

### Payoff-based Learning with Matrix Multiplicative Weights in Quantum Games

**Authors:** Kyriakos Lotidis, Panayotis Mertikopoulos, Nicholas Bambos, Jose Blanchet

**<details><summary>Abstract**</summary>

In this paper, we study the problem of learning in quantum games - and other classes of semidefinite games - with scalar, payoff-based feedback.For concreteness, we focus on the widely used matrix multiplicative weights (MMW) algorithm and, instead of requiring players to have full knowledge of the game (and/or each other's chosen states), we introduce a suite of minimal-information matrix multiplicative weights (3MW) methods tailored to different information frameworks.The main difficulty to attaining convergence in this setting is that, in contrast to classical finite games, quantum games have an infinite continuum of pure states (the quantum equivalent of pure strategies), so standard importance-weighting techniques for estimating payoff vectors cannot be employed.Instead, we borrow ideas from bandit convex optimization and we design a zeroth-order gradient sampler adapted to the semidefinite geometry of the problem at hand.As a first result, we show that the 3MW method with deterministic payoff feedback retains the $\mathcal{O}(1/\sqrt{T})$ convergence rate of the vanilla, full information MMW algorithm in quantum min-max games, even though the players only observe a single scalar.Subsequently, we relax the algorithm's information requirements even further and we provide a 3MW method that only requires players to observe a random realization of their payoff observable, and converges to equilibrium at an $\mathcal{O}(T^{-1/4})$ rate.Finally, going beyond zero-sum games, we show that a regularized variant of the proposed 3MW method guarantees local convergence with high probability to all equilibria that satisfy a certain first-order stability condition.</details>

### Penalising the biases in norm regularisation enforces sparsity

**Authors:** Etienne Boursier, Nicolas Flammarion

**<details><summary>Abstract**</summary>

Controlling the parameters' norm often yields good generalisation when training neural networks. Beyond simple intuitions, the relation between regularising parameters' norm and obtained estimators remains theoretically misunderstood. For one hidden ReLU layer networks with unidimensional data, this work shows the parameters' norm required to represent a function is given by the total variation of its second derivative, weighted by a $\sqrt{1+x^2}$ factor. Notably, this weighting factor disappears when the norm of bias terms is not regularised. The presence of this additional weighting factor is of utmost significance as it is shown to enforce the uniqueness and sparsity (in the number of kinks) of the minimal norm interpolator. Conversely, omitting the bias' norm  allows for non-sparse solutions.Penalising the bias terms in the regularisation, either explicitly or implicitly, thus leads to sparse estimators.</details>

### Perceptual adjustment queries and an inverted measurement paradigm for low-rank metric learning

**Authors:** Austin Xu, Andrew McRae, Jingyan Wang, Mark Davenport, Ashwin Pananjady

**<details><summary>Abstract**</summary>

We introduce a new type of query mechanism for collecting human feedback, called the perceptual adjustment query (PAQ). Being both informative and cognitively lightweight, the PAQ adopts an inverted measurement scheme, and combines advantages from both cardinal and ordinal queries. We showcase the PAQ in the metric learning problem, where we collect PAQ measurements to learn an unknown Mahalanobis distance. This gives rise to a high-dimensional, low-rank matrix estimation problem to which standard matrix estimators cannot be applied. Consequently, we develop a two-stage estimator for metric learning from PAQs, and provide sample complexity guarantees for this estimator. We present numerical simulations demonstrating the performance of the estimator and its notable properties.</details>

### Performance Bounds for Policy-Based Average Reward Reinforcement Learning Algorithms

**Authors:** Yashaswini Murthy, Mehrdad Moharrami, R. Srikant

**<details><summary>Abstract**</summary>

Many policy-based reinforcement learning (RL) algorithms can be viewed as instantiations of approximate policy iteration (PI), i.e., where policy improvement and policy evaluation are both performed approximately. In applications where the average reward objective is the meaningful performance metric, often discounted reward formulations are used with the discount factor being close to $1,$ which is equivalent to making the expected horizon very large. However, the corresponding theoretical bounds for error performance scale with the square of the horizon. Thus, even after dividing the total reward by the length of the horizon, the corresponding performance bounds for average reward problems go to infinity. Therefore, an open problem has been to obtain meaningful performance bounds for approximate PI and RL algorithms for the average-reward setting.  In this paper, we solve this open problem by obtaining the first non-trivial finite time error bounds for average-reward MDPs which go to zero in the limit as policy evaluation and policy improvement errors go to zero.</details>

### Perturbation Towards Easy Samples Improves Targeted Adversarial Transferability

**Authors:** Junqi Gao, Biqing Qi, Yao Li, Zhichang Guo, Dong Li, Yuming Xing, Dazhi Zhang

**<details><summary>Abstract**</summary>

The transferability of adversarial perturbations provides an effective shortcut for black-box attacks. Targeted perturbations have greater practicality but are more difficult to transfer between models. In this paper, we experimentally and theoretically demonstrated that neural networks trained on the same dataset have more consistent performance in High-Sample-Density-Regions (HSDR) of each class instead of low sample density regions. Therefore, in the target setting, adding perturbations towards HSDR of the target class is more effective in improving transferability. However, density estimation is challenging in high-dimensional scenarios. Further theoretical and experimental verification demonstrates that easy samples with low loss are more likely to be located in HSDR. Perturbations towards such easy samples in the target class can avoid density estimation for HSDR location. Based on the above facts, we verified that adding perturbations to easy samples in the target class improves targeted adversarial transferability of existing attack methods. A generative targeted attack strategy named Easy Sample Matching Attack (ESMA) is proposed, which has a higher success rate for targeted attacks and outperforms the SOTA generative method. Moreover, ESMA requires only $5\%$ of the storage space and much less computation time comparing to the current SOTA, as ESMA attacks all classes with only one model instead of seperate models for each class. Our code is available at https://github.com/gjq100/ESMA</details>

### Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation

**Authors:** Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, Omer Levy

**<details><summary>Abstract**</summary>

The ability to collect a large dataset of human preferences from text-to-image users is usually limited to companies, making such datasets inaccessible to the public. To address this issue, we create a web app that enables text-to-image users to generate images and specify their preferences. Using this web app we build Pick-a-Pic, a large, open dataset of text-to-image prompts and real users’ preferences over generated images. We leverage this dataset to train a CLIP-based scoring function, PickScore, which exhibits superhuman performance on the task of predicting human preferences. Then, we test PickScore’s ability to perform model evaluation and observe that it correlates better with human rankings than other automatic evaluation metrics. Therefore, we recommend using PickScore for evaluating future text-to-image generation models, and using Pick-a-Pic prompts as a more relevant dataset than MS-COCO. Finally, we demonstrate how PickScore can enhance existing text-to-image models via ranking.</details>

### PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change

**Authors:** Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, Subbarao Kambhampati

**<details><summary>Abstract**</summary>

Generating plans of action, and reasoning about change have long been considered a core competence of intelligent agents. It is thus no surprise that evaluating the planning and reasoning capabilities of large language models (LLMs) has become a hot topic of research. Most claims about LLM planning capabilities are however based on common sense tasks–where it becomes hard to tell whether LLMs are planning or merely retrieving from their vast world knowledge.  There is a strong need for systematic and extensible planning benchmarks with sufficient diversity to evaluate whether LLMs have innate planning capabilities. Motivated by this, we propose PlanBench, an extensible benchmark suite based on the kinds of domains used in the automated planning community, especially in the International Planning Competition, to test the capabilities of LLMs in planning or reasoning about actions and change. PlanBench provides sufficient diversity in both the task domains and the specific planning capabilities. Our studies also show that on many critical capabilities–including plan generation–LLM performance falls quite short, even with the SOTA models. PlanBench can thus function as a useful marker of progress of LLMs in planning and reasoning.</details>

### Policy Gradient for Rectangular Robust Markov Decision Processes

**Authors:** Navdeep Kumar, Esther Derman, Matthieu Geist, Kfir Y. Levy, Shie Mannor

**<details><summary>Abstract**</summary>

Policy gradient methods have become a standard for training reinforcement learning agents in a scalable and efficient manner. However, they do not account for transition uncertainty, whereas learning robust policies can be computationally expensive. In this paper, we introduce robust policy gradient (RPG), a policy-based method that efficiently solves rectangular robust Markov decision processes (MDPs). We provide a closed-form expression for the worst occupation measure. Incidentally, we find that the worst kernel is a rank-one perturbation of the nominal. Combining the worst occupation measure with a robust Q-value estimation yields an explicit form of the robust gradient. Our resulting RPG can be estimated from data with the same time complexity as its non-robust equivalent. Hence, it relieves the computational burden of convex optimization problems required for training robust policies by current policy gradient approaches.</details>

### Practical Differentially Private Hyperparameter Tuning with Subsampling

**Authors:** Antti Koskela, Tejas Kulkarni

**<details><summary>Abstract**</summary>

Tuning the hyperparameters of differentially private (DP) machine learning (ML) algorithms often requires use of sensitive data and this may leak private information via hyperparameter values. Recently, Papernot and Steinke (2022) proposed a certain class of DP hyperparameter tuning algorithms, where the number of random search samples is randomized. Commonly, these algorithms still considerably increase the DP privacy parameter $\varepsilon$ over non-tuned DP ML model training and can be computationally heavy as evaluating each hyperparameter candidate requires a new training run. We focus on lowering both the DP bounds and the compute cost of these methods by using only a random subset of the sensitive data for the hyperparameter tuning and by appropriately extrapolating the optimal values to a larger dataset. We carry out a Rényi differential privacy analysis for the proposed method and experimentally show that it consistently leads to better privacy-utility trade-off than the baseline method by Papernot and Steinke.</details>

### PreDiff: Precipitation Nowcasting with Latent Diffusion Models

**Authors:** Zhihan Gao, Xingjian Shi, Boran Han, Hao Wang, Xiaoyong Jin, Danielle Maddix, Yi Zhu, Mu Li, Yuyang (Bernie) Wang

**<details><summary>Abstract**</summary>

Earth system forecasting has traditionally relied on complex physical models that are computationally expensive and require significant domain expertise.In the past decade, the unprecedented increase in spatiotemporal Earth observation data has enabled data-driven forecasting models using deep learning techniques.These models have shown promise for diverse Earth system forecasting tasks but either struggle with handling uncertainty or neglect domain-specific prior knowledge, resulting in averaging possible futures to blurred forecasts or generating physically implausible predictions.To address these limitations, we propose a two-stage pipeline for probabilistic spatiotemporal forecasting: 1) We develop *PreDiff*, a conditional latent diffusion model capable of probabilistic forecasts. 2) We incorporate an explicit knowledge alignment mechanism to align forecasts with domain-specific physical constraints. This is achieved by estimating the deviation from imposed constraints at each denoising step and adjusting the transition distribution accordingly.We conduct empirical studies on two datasets: N-body MNIST, a synthetic dataset with chaotic behavior, and SEVIR, a real-world precipitation nowcasting dataset. Specifically, we impose the law of conservation of energy in N-body MNIST and anticipated precipitation intensity in SEVIR. Experiments demonstrate the effectiveness of PreDiff in handling uncertainty, incorporating domain-specific prior knowledge, and generating forecasts that exhibit high operational utility.</details>

### Prediction and Control in Continual Reinforcement Learning

**Authors:** Nishanth Anand, Doina Precup

**<details><summary>Abstract**</summary>

Temporal difference (TD) learning is often used to update the estimate of the value function which is used by RL agents to extract useful policies. In this paper, we focus on value function estimation in continual reinforcement learning. We propose to decompose the value function into two components which update at different timescales: a _permanent_ value function, which holds general knowledge that persists over time, and a _transient_ value function, which allows quick adaptation to new situations. We establish theoretical results showing that our approach is well suited for continual learning and draw connections to the complementary learning systems (CLS) theory from neuroscience. Empirically, this approach improves performance significantly on both prediction and control problems.</details>

### PrimDiffusion: Volumetric Primitives Diffusion for 3D Human Generation

**Authors:** Zhaoxi Chen, Fangzhou Hong, Haiyi Mei, Guangcong Wang, Lei Yang, Ziwei Liu

**<details><summary>Abstract**</summary>

We present PrimDiffusion, the first diffusion-based framework for 3D human generation. Devising diffusion models for 3D human generation is difficult due to the intensive computational cost of 3D representations and the articulated topology of 3D humans. To tackle these challenges, our key insight is operating the denoising diffusion process directly on a set of volumetric primitives, which models the human body as a number of small volumes with radiance and kinematic information. This volumetric primitives representation marries the capacity of volumetric representations with the efficiency of primitive-based rendering. Our PrimDiffusion framework has three appealing properties: **1)** compact and expressive parameter space for the diffusion model, **2)** flexible representation that incorporates human prior, and **3)** decoder-free rendering for efficient novel-view and novel-pose synthesis. Extensive experiments validate that PrimDiffusion outperforms state-of-the-art methods in 3D human generation. Notably, compared to GAN-based methods, our PrimDiffusion supports real-time rendering of high-quality 3D humans at a resolution of $512\times512$ once the denoising process is done. We also demonstrate the flexibility of our framework on training-free conditional generation such as texture transfer and 3D inpainting.</details>

### [Spotlight] Private estimation algorithms for stochastic block models and mixture models

**Authors:** Hongjie Chen, Vincent Cohen-Addad, Tommaso d’Orsi, Alessandro Epasto, Jacob Imola, David Steurer, Stefan Tiegel

**<details><summary>Abstract**</summary>

We introduce general tools for designing efficient private estimation algorithms, in the high-dimensional settings, whose statistical guarantees almost match those of the best known non-private algorithms.To illustrate our techniques, we consider two problems: recovery of stochastic block models and learning mixtures of spherical Gaussians.For the former, we present the first efficient $(\epsilon, \delta)$-differentially private algorithm for both weak recovery and exact recovery. Previously known algorithms achieving comparable guarantees required quasi-polynomial time. For the latter, we design an  $(\epsilon, \delta)$-differentially private algorithm that recovers the centers of the $k$-mixture when the minimum separation is at least $O(k^{1/t}\sqrt{t})$. For all choices of $t$, this algorithm requires sample complexity $n\geq k^{O(1)}d^{O(t)}$ and time complexity $(nd)^{O(t)}$. Prior work required either an additional additive $\Omega(\sqrt{\log n})$ term in the minimum separation or an explicit upper bound on the Euclidean norm of the centers.</details>

### Prompt-augmented Temporal Point Process for Streaming Event Sequence

**Authors:** Siqiao Xue, Yan Wang, Zhixuan Chu, Xiaoming Shi, Caigao JIANG, Hongyan Hao, Gangwei Jiang, Xiaoyun Feng, James Zhang, Jun Zhou

**<details><summary>Abstract**</summary>

Neural Temporal Point Processes (TPPs)  are the prevalent paradigm for modeling continuous-time event sequences, such as user activities on the web and financial transactions. In real world applications, the event data typically comes in a streaming manner, where the distribution of the patterns may shift over time. Under the privacy and memory constraints commonly seen in real scenarios, how to continuously monitor a TPP to learn the streaming event sequence is an important yet under-investigated problem. In this work, we approach this problem by adopting Continual Learning (CL), which aims to enable a model to continuously learn a sequence of tasks without catastrophic forgetting. While CL for event sequence is less well studied, we present a simple yet effective framework, PromptTPP, by integrating the base TPP with a continuous-time retrieval prompt pool. In our proposed framework, prompts are small learnable parameters, maintained in a memory space and jointly optimized with the base TPP so that the model is properly instructed to learn event streams arriving sequentially without buffering past examples or task-specific attributes. We formalize a novel and realistic experimental setup for modeling event streams, where PromptTPP consistently sets state-of-the-art performance across two real user behavior datasets.</details>

### PromptIR: Prompting for All-in-One Image Restoration

**Authors:** Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, Fahad Shahbaz Khan

**<details><summary>Abstract**</summary>

Image restoration involves recovering a high-quality clean image from its degraded version. Deep learning-based methods have significantly improved image restoration performance, however, they have limited generalization ability to different degradation types and levels. This restricts their real-world application since it requires training individual models for each specific degradation and knowing the input degradation type to apply the relevant model. We present a prompt-based learning approach, PromptIR, for All-In-One image restoration that can effectively restore images from various types and levels of degradation. In particular, our method uses prompts to encode degradation-specific information, which is then used to dynamically guide the restoration network. This allows our method to generalize to different degradation types and levels, while still achieving state-of-the-art results on image denoising, deraining, and dehazing. Overall, PromptIR offers a generic and efficient plugin module with few lightweight prompts that can be used to restore images of various types and levels of degradation with no prior information on the corruptions present in the image. Our code and pre-trained models are available here: https://github.com/va1shn9v/PromptIR</details>

### Propagating Knowledge Updates to LMs Through Distillation

**Authors:** Shankar Padmanabhan, Yasumasa Onoe, Michael Zhang, Greg Durrett, Eunsol Choi

**<details><summary>Abstract**</summary>

Modern language models have the capacity to store and use immense amounts of knowledge about real-world entities, but it remains unclear how to update such knowledge stored in model parameters. While prior methods for updating knowledge in LMs successfully inject atomic facts, updated LMs fail to make inferences based on injected facts. In this work, we demonstrate that a context distillation-based approach can both impart knowledge about entities \emph{and} propagate that knowledge to enable broader inferences. Our approach consists of two stages: transfer set generation and distillation on the transfer set. We first generate a transfer set by prompting a language model to generate continuations from the entity definition. Then, we update the model parameters so that the distribution of the LM (the 'student') matches the distribution of the LM conditioned on the definition (the 'teacher') on the transfer set. Our experiments demonstrate that this approach is more effective at propagating knowledge updates than fine-tuning and other gradient-based knowledge-editing methods. Moreover, it does not  compromise performance in other contexts, even when injecting the definitions of up to 150 entities at once.</details>

### Proportional Response: Contextual Bandits for Simple and Cumulative Regret Minimization

**Authors:** Sanath Kumar Krishnamurthy, Ruohan Zhan, Susan Athey, Emma Brunskill

**<details><summary>Abstract**</summary>

In many applications, e.g. in healthcare and e-commerce, the goal of a contextual bandit may be to learn an optimal treatment assignment policy at the end of the experiment. That is, to minimize simple regret. However, this objective remains understudied. We propose a new family of computationally efficient bandit algorithms for the stochastic contextual bandit setting, where a tuning parameter determines the weight placed on cumulative regret minimization (where we establish near-optimal minimax guarantees) versus simple regret minimization (where we establish state-of-the-art guarantees). Our algorithms work with any function class, are robust to model misspecification, and can be used in continuous arm settings. This flexibility comes from constructing and relying on “conformal arm sets" (CASs). CASs provide a set of arms for every context, encompassing the context-specific optimal arm with a certain probability across the context distribution. Our positive results on simple and cumulative regret guarantees are contrasted with a negative result, which shows that no algorithm can achieve instance-dependent simple regret guarantees while simultaneously achieving minimax optimal cumulative regret guarantees.</details>

### [Spotlight] Protein Design with Guided Discrete Diffusion

**Authors:** Nate Gruver, Samuel Stanton, Nathan Frey, Tim G. J. Rudner, Isidro Hotzel, Julien Lafrance-Vanasse, Arvind Rajpal, Kyunghyun Cho, Andrew Wilson

**<details><summary>Abstract**</summary>

A popular approach to protein design is to combine a generative model with a discriminative model for conditional sampling. The generative model samples plausible sequences while the discriminative model guides a search for sequences with high fitness. Given its broad success in conditional sampling, classifier-guided diffusion modeling is a promising foundation for protein design, leading many to develop guided diffusion models for structure with inverse folding to recover sequences. In this work, we propose diffusioN Optimized Sampling (NOS), a guidance method for discrete diffusion models that follows gradients in the hidden states of the denoising network. NOS makes it possible to perform design directly in sequence space, circumventing significant limitations of structure-based methods, including scarce data and challenging inverse design. Moreover, we use NOS to generalize LaMBO, a Bayesian optimization procedure for sequence design that facilitates multiple objectives and edit-based constraints. The resulting method, LaMBO-2, enables discrete diffusions and  stronger performance with limited edits through a novel application of saliency maps. We apply LaMBO-2 to a real-world protein design task, optimizing antibodies for higher expression yield and binding affinity to several therapeutic targets under locality and developability constraints, attaining a 99\% expression rate and 40\% binding rate in exploratory in vitro experiments.</details>

### Prototype-based Aleatoric Uncertainty Quantification for Cross-modal Retrieval

**Authors:** Hao Li, Jingkuan Song, Lianli Gao, Xiaosu Zhu, Hengtao Shen

**<details><summary>Abstract**</summary>

Cross-modal Retrieval methods build similarity relations between vision and language modalities by jointly learning a common representation space. However, the predictions are often unreliable due to the Aleatoric uncertainty, which is induced by low-quality data, e.g., corrupt images, fast-paced videos, and non-detailed texts. In this paper, we propose a novel Prototype-based Aleatoric Uncertainty Quantification (PAU) framework to provide trustworthy predictions by quantifying the uncertainty arisen from the inherent data ambiguity. Concretely, we first construct a set of various learnable prototypes for each modality to represent the entire semantics subspace. Then Dempster-Shafer Theory and Subjective Logic Theory are utilized to build an evidential theoretical framework by associating evidence with Dirichlet Distribution parameters. The PAU model induces accurate uncertainty and reliable predictions for cross-modal retrieval. Extensive experiments are performed on four major benchmark datasets of MSR-VTT, MSVD, DiDeMo, and MS-COCO, demonstrating the effectiveness of our method. The code is accessible at https://github.com/leolee99/PAU.</details>

### [Spotlight] Provably Bounding Neural Network Preimages

**Authors:** Suhas Kotha, Christopher Brix, J. Zico Kolter, Krishnamurthy Dvijotham, Huan Zhang

**<details><summary>Abstract**</summary>

Most work on the formal verification of neural networks has focused on bounding the set of outputs that correspond to a given set of inputs (for example, bounded perturbations of a nominal input). However, many use cases of neural network verification require solving the inverse problem, or over-approximating the set of inputs that lead to certain outputs. We present the INVPROP algorithm for verifying properties over the preimage of a linearly constrained output set, which can be combined with branch-and-bound to increase precision. Contrary to other approaches, our efficient algorithm is GPU-accelerated and does not require a linear programming solver. We demonstrate our algorithm for identifying safe control regions for a dynamical system via backward reachability analysis, verifying adversarial robustness, and detecting out-of-distribution inputs to a neural network. Our results show that in certain settings, we find over-approximations over $2500	imes$ tighter than prior work while being $2.5	imes$ faster. By strengthening robustness verification with output constraints, we consistently verify more properties than the previous state-of-the-art on multiple benchmarks, including a large model with 167k neurons in VNN-COMP 2023. Our algorithm has been incorporated into the $\alpha,eta$-CROWN verifier, available at https://abcrown.org.</details>

### [Spotlight] Provably Fast Finite Particle Variants of SVGD via Virtual Particle Stochastic Approximation

**Authors:** Aniket Das, Dheeraj Nagaraj

**<details><summary>Abstract**</summary>

Stein Variational Gradient Descent (SVGD) is a popular particle-based variational inference algorithm with impressive empirical performance across various domains. Although the population (i.e, infinite-particle) limit dynamics of SVGD is well characterized, its behavior in the finite-particle regime is far less understood. To this end, our work introduces the notion of *virtual particles* to develop novel stochastic approximations of population-limit SVGD dynamics in the space of probability measures, that are exactly realizable using finite particles. As a result, we design two computationally efficient variants of SVGD, namely VP-SVGD and GB-SVGD, with provably fast finite-particle convergence rates. Our algorithms can be viewed as specific random-batch approximations of SVGD, which are computationally more efficient than ordinary SVGD. We show that the $n$ particles output by VP-SVGD and GB-SVGD, run for $T$ steps with batch-size $K$, are at-least as good as i.i.d samples from a distribution whose Kernel Stein Discrepancy to the target is at most $O(\tfrac{d^{1/3}}{(KT)^{1/6}})$ under standard assumptions. Our results also hold under a mild growth condition on the potential function, which is much weaker than the isoperimetric (e.g. Poincare Inequality) or information-transport conditions (e.g. Talagrand's Inequality $\mathsf{T}_1$) generally considered in prior works. As a corollary, we analyze the convergence of the empirical measure (of the particles output by VP-SVGD and GB-SVGD) to the target distribution and demonstrate a **double exponential improvement** over the best known finite-particle analysis of SVGD. Beyond this, our results present the **first known oracle complexities for this setting with polynomial dimension dependence**, thereby completely eliminating the curse of dimensionality exhibited by previously known finite-particle rates.</details>

### Provably Robust Temporal Difference Learning for Heavy-Tailed Rewards

**Authors:** Semih Cayci, Atilla Eryilmaz

**<details><summary>Abstract**</summary>

In a broad class of reinforcement learning applications, stochastic rewards have heavy-tailed distributions, which lead to infinite second-order moments for stochastic (semi)gradients in policy evaluation and direct policy optimization. In such instances, the existing RL methods may fail miserably due to frequent statistical outliers. In this work, we establish that temporal difference (TD) learning with a dynamic gradient clipping mechanism, and correspondingly operated natural actor-critic (NAC), can be provably robustified against heavy-tailed reward distributions. It is shown in the framework of linear function approximation that a favorable tradeoff between bias and variability of the stochastic gradients can be achieved with this dynamic gradient clipping mechanism. In particular, we prove that robust versions of TD learning achieve sample complexities of order $\mathcal{O}(\varepsilon^{-\frac{1}{p}})$ and $\mathcal{O}(\varepsilon^{-1-\frac{1}{p}})$ with and without the full-rank assumption on the feature matrix, respectively, under heavy-tailed rewards with finite moments of order $(1+p)$ for some $p\in(0,1]$, both in expectation and with high probability. We show that a robust variant of NAC based on Robust TD learning achieves $\tilde{\mathcal{O}}(\varepsilon^{-4-\frac{2}{p}})$ sample complexity. We corroborate our theoretical results with numerical experiments.</details>

### Pseudo-Likelihood Inference

**Authors:** Theo Gruner, Boris Belousov, Fabio Muratore, Daniel Palenicek, Jan Peters

**<details><summary>Abstract**</summary>

Simulation-Based Inference (SBI) is a common name for an emerging family of approaches that infer the model parameters when the likelihood is intractable. Existing SBI methods either approximate the likelihood, such as Approximate Bayesian Computation (ABC) or directly model the posterior, such as Sequential Neural Posterior Estimation (SNPE). While ABC is efficient on low-dimensional problems, on higher-dimensional tasks, it is generally outperformed by SNPE, which leverages function approximation. In this paper, we propose Pseudo-Likelihood Inference (PLI), a new method that brings neural approximation into ABC, making it competitive on challenging Bayesian system identification tasks. By utilizing integral probability metrics, we introduce a smooth likelihood kernel with an adaptive bandwidth that is updated based on information-theoretic trust regions. Thanks to this formulation, our method (i) allows for optimizing neural posteriors via gradient descent, (ii) does not rely on summary statistics, and (iii) enables multiple observations as input. In comparison to SNPE, it leads to improved performance when more data is available. The effectiveness of PLI is evaluated on four classical SBI benchmark tasks and on a highly dynamic physical system, showing particular advantages on stochastic simulations and multi-modal posterior landscapes.</details>

### [Spotlight] QuACK: Accelerating Gradient-Based Quantum Optimization with Koopman Operator Learning

**Authors:** Di Luo, Jiayu Shen, Rumen Dangovski, Marin Soljacic

**<details><summary>Abstract**</summary>

Quantum optimization, a key application of quantum computing, has traditionally been stymied by the linearly increasing complexity of gradient calculations with an increasing number of parameters. This work bridges the gap between Koopman operator theory, which has found utility in applications because it allows for a linear representation of nonlinear dynamical systems, and natural gradient methods in quantum optimization, leading to a significant acceleration of gradient-based quantum optimization. We present Quantum-circuit Alternating Controlled Koopman learning (QuACK), a novel framework that leverages an alternating algorithm for efficient prediction of gradient dynamics on quantum computers. We demonstrate QuACK's remarkable ability to accelerate gradient-based optimization across a range of applications in quantum optimization and machine learning. In fact, our empirical studies, spanning quantum chemistry, quantum condensed matter, quantum machine learning, and noisy environments, have shown accelerations of more than 200x speedup in the overparameterized regime, 10x speedup in the smooth regime, and 3x speedup in the non-smooth regime. With QuACK, we offer a robust advancement that harnesses the advantage of gradient-based quantum optimization for practical benefits.</details>

### Quantifying & Modeling Multimodal Interactions: An Information Decomposition Framework

**Authors:** Paul Pu Liang, Yun Cheng, Xiang Fan, Chun Kai Ling, Suzanne Nie, Richard Chen, Zihao Deng, Nicholas Allen, Randy Auerbach, Faisal Mahmood, Russ Salakhutdinov, Louis-Philippe Morency

**<details><summary>Abstract**</summary>

The recent explosion of interest in multimodal applications has resulted in a wide selection of datasets and methods for representing and integrating information from different modalities. Despite these empirical advances, there remain fundamental research questions: How can we quantify the interactions that are necessary to solve a multimodal task? Subsequently, what are the most suitable multimodal models to capture these interactions? To answer these questions, we propose an information-theoretic approach to quantify the degree of redundancy, uniqueness, and synergy relating input modalities with an output task. We term these three measures as the PID statistics of a multimodal distribution (or PID for short), and introduce two new estimators for these PID statistics that scale to high-dimensional distributions. To validate PID estimation, we conduct extensive experiments on both synthetic datasets where the PID is known and on large-scale multimodal benchmarks where PID estimations are compared with human annotations. Finally, we demonstrate their usefulness in (1) quantifying interactions within multimodal datasets, (2) quantifying interactions captured by multimodal models, (3) principled approaches for model selection, and (4) three real-world case studies engaging with domain experts in pathology, mood prediction, and robotic perception where our framework helps to recommend strong multimodal models for each application.</details>

### Quantum speedups for stochastic optimization

**Authors:** Aaron Sidford, Chenyi Zhang

**<details><summary>Abstract**</summary>

We consider the problem of minimizing a continuous function given given access to a natural quantum generalization of a stochastic gradient oracle. We provide two new methods for the special case of minimizing a Lipschitz convex function. Each method obtains a dimension versus accuracy trade-off which is provably unachievable classically and we prove that one method is asymptotically optimal in low-dimensional settings. Additionally, we provide quantum algorithms for computing a critical point of a smooth non-convex function at rates not known to be achievable classically. To obtain these results we build upon the quantum multivariate mean estimation result of Cornelissen et al. and provide a general quantum variance reduction technique of independent interest.</details>

### R-divergence for Estimating Model-oriented Distribution Discrepancy

**Authors:** Zhilin Zhao, Longbing Cao

**<details><summary>Abstract**</summary>

Real-life data are often non-IID due to complex distributions and interactions, and the sensitivity to the distribution of samples can differ among learning models. Accordingly, a key question for any supervised or unsupervised model is whether the probability distributions of two given datasets can be considered identical. To address this question, we introduce R-divergence, designed to assess model-oriented distribution discrepancies. The core insight is that two distributions are likely identical if their optimal hypothesis yields the same expected risk for each distribution. To estimate the distribution discrepancy between two datasets, R-divergence learns a minimum hypothesis on the mixed data and then gauges the empirical risk difference between them. We evaluate the test power across various unsupervised and supervised tasks and find that R-divergence achieves state-of-the-art performance. To demonstrate the practicality of R-divergence, we employ R-divergence to train robust neural networks on samples with noisy labels.</details>

### RADAR: Robust AI-Text Detection via Adversarial Learning

**Authors:** Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**<details><summary>Abstract**</summary>

Recent advances in large language models (LLMs) and the intensifying popularity of ChatGPT-like applications have blurred the boundary of high-quality text generation between humans and machines. However, in addition to the anticipated revolutionary changes to our technology and society, the difficulty of distinguishing LLM-generated texts (AI-text) from human-generated texts poses new challenges of misuse and fairness, such as fake content generation, plagiarism, and false accusations of innocent writers. While existing works show that current AI-text detectors are not robust to LLM-based paraphrasing, this paper aims to bridge this gap by proposing a new framework called RADAR, which jointly trains a $\underline{r}$obust  $\underline{A}$I-text  $\underline{d}$etector via  $\underline{a}$dversarial lea$\underline{r}$ning. RADAR is based on adversarial training of a paraphraser and a detector. The paraphraser's goal is to generate realistic content to evade AI-text detection.RADAR uses the feedback from the detector to update the paraphraser, and vice versa.Evaluated with 8 different LLMs (Pythia, Dolly 2.0, Palmyra, Camel, GPT-J, Dolly 1.0, LLaMA, and Vicuna) across 4 datasets, experimental results show that RADAR significantly outperforms existing AI-text detection methods, especially when paraphrasing is in place. We also identify the strong transferability of RADAR from instruction-tuned LLMs to other LLMs, and evaluate the improved capability of RADAR via GPT-3.5-Turbo.</details>

### REFINE: A Fine-Grained Medication Recommendation System Using Deep Learning and Personalized Drug Interaction Modeling

**Authors:** Suman Bhoi, Mong Li Lee, Wynne Hsu, Ngiap Chuan Tan

**<details><summary>Abstract**</summary>

Patients with co-morbidities often require multiple medications to manage their conditions. However, existing medication recommendation systems only offer class-level medications and regard all interactions among drugs to have the same level of severity. This limits their ability to provide personalized and safe recommendations tailored to individual needs. In this work, we introduce a deep learning-based fine-grained medication recommendation system called REFINE, which is designed to improve treatment outcomes and minimize adverse drug interactions. In order to better characterize patients’ health conditions, we model the trend in medication dosage titrations and lab test responses, and adapt the vision transformer to obtain effective patient representations. We also model drug interaction severity levels as weighted graphs to learn safe drug combinations and design a balanced loss function to avoid overly conservative recommendations and miss medications that might be needed for certain conditions. Extensive experiments on two real-world datasets show that REFINE outperforms state-of-the-art techniques.</details>

### RRHF: Rank Responses to Align Language Models with Human Feedback

**Authors:** Hongyi Yuan, Zheng Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, Fei Huang

**<details><summary>Abstract**</summary>

Reinforcement Learning from Human Feedback (RLHF) facilitates the alignment of large language models with human preferences, significantly enhancing the quality of interactions between humans and models. InstructGPT implements RLHF through several stages, including Supervised Fine-Tuning (SFT), reward model training, and Proximal Policy Optimization (PPO). However, PPO is sensitive to hyperparameters and requires multiple models in its standard implementation, making it hard to train and scale up to larger parameter counts.In contrast, we propose a novel learning paradigm called RRHF, which scores sampled responses from different sources via a logarithm of conditional probabilities and learns to align these probabilities with human preferences through ranking loss.RRHF can leverage sampled responses from various sources including the model responses from itself, other large language model responses, and human expert responses to learn to rank them.RRHF only needs 1 to 2 models during tuning and can efficiently align language models with human preferences robustly without complex hyperparameter tuning. Additionally, RRHF can be considered an extension of SFT and reward model training while being simpler than PPO in terms of coding, model counts, and hyperparameters. We evaluate RRHF on the Helpful and Harmless dataset, demonstrating comparable alignment performance with PPO by reward model score and human labeling.Extensive experiments show that the performance of RRHF is highly related to sampling quality which suggests RRHF is a best-of-$n$ learner.</details>

### [Spotlight] Randomized Sparse Neural Galerkin Schemes for Solving Evolution Equations with Deep Networks

**Authors:** Jules Berman, Benjamin Peherstorfer

**<details><summary>Abstract**</summary>

Training neural networks sequentially in time to approximate solution fields of time-dependent partial differential equations can be beneficial for preserving causality and other physics properties; however, the sequential-in-time training is numerically challenging because training errors quickly accumulate and amplify over time. This work introduces Neural Galerkin schemes that update randomized sparse subsets of network parameters at each time step. The randomization avoids overfitting locally in time and so helps prevent the error from accumulating quickly over the sequential-in-time training, which is motivated by dropout that addresses a similar issue of overfitting due to neuron co-adaptation. The sparsity of the update reduces the computational costs of training without losing expressiveness because many of the network parameters are redundant locally at each time step. In numerical experiments with a wide range of evolution equations, the proposed scheme with randomized sparse updates is up to two orders of magnitude more accurate at a fixed computational budget and up to two orders of magnitude faster at a fixed accuracy than schemes with dense updates.</details>

### Randomized and Deterministic Maximin-share Approximations for Fractionally Subadditive Valuations

**Authors:** Hannaneh Akrami, Kurt Mehlhorn, Masoud Seddighin, Golnoosh Shahkarami

**<details><summary>Abstract**</summary>

We consider the problem of guaranteeing maximin-share ($\MMS$) when allocating a set of indivisible items to a set of agents with fractionally subadditive ($\XOS$) valuations.  For $\XOS$ valuations, it has been previously shown that for some instances no allocation can guarantee a fraction better than $1/2$ of maximin-share to all the agents. Also, a deterministic allocation exists that guarantees $0.219225$ of the maximin-share of each agent. Our results involve both deterministic and randomized allocations. On the deterministic side, we improve the best approximation guarantee for fractionally subadditive valuations to $3/13 = 0.230769$. We develop new ideas on allocating large items in our allocation algorithm which might be of independent interest. Furthermore, we investigate randomized algorithms and the Best-of-both-worlds fairness guarantees. We propose a randomized allocation that is $1/4$-$\MMS$ ex-ante and $1/8$-$\MMS$ ex-post for $\XOS$ valuations. Moreover, we prove an upper bound of $3/4$ on the ex-ante guarantee for this class of valuations.</details>

### [Spotlight] RePo: Resilient Model-Based Reinforcement Learning by Regularizing Posterior Predictability

**Authors:** Chuning Zhu, Max Simchowitz, Siri Gadipudi, Abhishek Gupta

**<details><summary>Abstract**</summary>

Visual model-based RL methods typically encode image observations into low-dimensional representations in a manner that does not eliminate redundant information. This leaves them susceptible to spurious variations -- changes in task-irrelevant components such as background distractors or lighting conditions. In this paper, we propose a visual model-based RL method that learns a latent representation resilient to such spurious variations. Our training objective encourages the representation to be maximally predictive of dynamics and reward, while constraining the information flow from the observation to the latent representation. We demonstrate that this objective significantly bolsters the resilience of visual model-based RL methods to visual distractors, allowing them to operate in dynamic environments. We then show that while the learned encoder is able to operate in dynamic environments, it is not invariant under significant distribution shift. To address this, we propose a simple reward-free alignment procedure that enables test time adaptation of the encoder. This allows for quick adaptation to widely differing environments without having to relearn the dynamics and policy. Our effort is a step towards making model-based RL a practical and useful tool for dynamic, diverse domains and we show its effectiveness in simulation tasks with significant spurious variations.</details>

### Reading Relevant Feature from Global Representation Memory for Visual Object Tracking

**Authors:** Xinyu Zhou, Pinxue Guo, Lingyi Hong, Jinglun Li, Wei Zhang, Weifeng Ge, Wenqiang Zhang

**<details><summary>Abstract**</summary>

Reference features from a template or historical frames are crucial for visual object tracking. Prior works utilize all features from a fixed template or memory for visual object tracking. However, due to the dynamic nature of videos, the required reference historical information for different search regions at different time steps is also inconsistent. Therefore, using all features in the template and memory can lead to redundancy and impair tracking performance. To alleviate this issue, we propose a novel tracking paradigm, consisting of a relevance attention mechanism and a global representation memory, which can adaptively assist the search region in selecting the most relevant historical information from reference features. Specifically, the proposed relevance attention mechanism in this work differs from previous approaches in that it can dynamically choose and build the optimal global representation memory for the current frame by accessing cross-frame information globally. Moreover, it can flexibly read the relevant historical information from the constructed memory to reduce redundancy and counteract the negative effects of harmful information. Extensive experiments validate the effectiveness of the proposed method, achieving competitive performance on five challenging datasets with 71 FPS.</details>

### Recaptured Raw Screen Image and Video Demoiréing via Channel and Spatial Modulations

**Authors:** Yijia Cheng, Yijia Cheng, Xin Liu, Jingyu Yang

**<details><summary>Abstract**</summary>

Capturing screen contents by smartphone cameras has become a common way for information sharing. However, these images and videos are often degraded by moiré patterns, which are caused by frequency aliasing between the camera filter array and digital display grids. We observe that the moiré patterns in raw domain is simpler than those in sRGB domain, and the moiré patterns in raw color channels have different properties. Therefore, we propose an image and video demoiréing network tailored for raw inputs. We introduce a color-separated feature branch, and it is fused with the traditional feature-mixed branch via channel and spatial modulations. Specifically, the channel modulation utilizes modulated color-separated features to enhance the color-mixed features. The spatial modulation utilizes the feature with large receptive field to modulate the feature with small receptive field. In addition, we build the first well-aligned raw video demoiréing (RawVDemoiré) dataset and propose an efficient temporal alignment method by inserting alternating patterns. Experiments demonstrate that our method achieves state-of-the-art performance for both image and video demoiréing. Our dataset and code will be released after the acceptance of this work.</details>

### Red Teaming Deep Neural Networks with Feature Synthesis Tools

**Authors:** Stephen Casper, Tong Bu, Yuxiao Li, Jiawei Li, Kevin Zhang, Kaivalya Hariharan, Dylan Hadfield-Menell

**<details><summary>Abstract**</summary>

Interpretable AI tools are often motivated by the goal of understanding model behavior in out-of-distribution (OOD) contexts. Despite the attention this area of study receives, there are comparatively few cases where these tools have identified previously unknown bugs in models. We argue that this is due, in part, to a common feature of many interpretability methods: they analyze model behavior by using a particular dataset. This only allows for the study of the model in the context of features that the user can sample in advance. To address this, a growing body of research involves interpreting models using feature synthesis methods that do not depend on a dataset. In this paper, we benchmark the usefulness of interpretability tools for model debugging. Our key insight is that we can implant human-interpretable trojans into models and then evaluate these tools based on whether they can help humans discover them. This is analogous to finding OOD bugs, except the ground truth is known, allowing us to know when a user's interpretation is correct. We make four contributions. (1) We propose trojan discovery as an evaluation task for interpretability tools and introduce a benchmark with 12 trojans of 3 different types. (2) We demonstrate the difficulty of this benchmark with a preliminary evaluation of 16 state-of-the-art feature attribution/saliency tools. Even under ideal conditions, given direct access to data with the trojan trigger, these methods still often fail to identify bugs. (3) We evaluate 7 feature-synthesis methods on our benchmark. (4) We introduce and evaluate 2 new variants of the best-performing method from the previous evaluation.</details>

### ResMem: Learn what you can and memorize the rest

**Authors:** Zitong Yang, MICHAL LUKASIK, Vaishnavh Nagarajan, Zonglin Li, Ankit Rawat, Manzil Zaheer, Aditya Menon, Sanjiv Kumar

**<details><summary>Abstract**</summary>

The impressive generalization performance of modern neural networks is attributed in part to their ability to implicitly memorize complex training patterns.Inspired by this, we explore a novel mechanism to improve model generalization via explicit memorization.Specifically, we propose the residual-memorization (ResMem) algorithm, a new method that augments an existing prediction model (e.g., a neural network) by fitting the model's residuals with a nearest-neighbor based regressor.The final prediction is then the sum of the original model and the fitted residual regressor.By construction, ResMem can explicitly memorize the training labels.We start by formulating a stylized linear regression problem and rigorously show that ResMem results in a more favorable test risk over a base linear neural network.Then, we empirically show that ResMem consistently improves the test set generalization of the original prediction model across standard vision and natural language processing benchmarks.</details>

### Residual Q-Learning: Offline and Online Policy Customization without Value

**Authors:** Chenran Li, Chen Tang, Haruki Nishimura, Jean Mercat, Masayoshi TOMIZUKA, Wei Zhan

**<details><summary>Abstract**</summary>

Imitation Learning (IL) is a widely used framework for learning imitative behavior from demonstrations. It is especially appealing for solving complex real-world tasks where handcrafting reward function is difficult, or when the goal is to mimic human expert behavior. However, the learned imitative policy can only follow the behavior in the demonstration. When applying the imitative policy, we may need to customize the policy behavior to meet different requirements coming from diverse downstream tasks. Meanwhile, we still want the customized policy to maintain its imitative nature. To this end, we formulate a new problem setting called policy customization. It defines the learning task as training a policy that inherits the characteristics of the prior policy while satisfying some additional requirements imposed by a target downstream task. We propose a novel and principled approach to interpret and determine the trade-off between the two task objectives. Specifically, we formulate the customization problem as a Markov Decision Process (MDP) with a reward function that combines 1) the inherent reward of the demonstration; and 2) the add-on reward specified by the downstream task. We propose a novel framework, Residual Q-learning, which can solve the formulated MDP by leveraging the prior policy without knowing the inherent reward or value function of the prior policy. We derive a family of residual Q-learning algorithms that can realize offline and online policy customization, and show that the proposed algorithms can effectively accomplish policy customization tasks in various environments. Demo videos and code are available on our website: https://sites.google.com/view/residualq-learning.</details>

### ResoNet: Noise-Trained Physics-Informed MRI Off-Resonance Correction

**Authors:** Alfredo De Goyeneche Macaya, Shreya Ramachandran, Ke Wang, Ekin Karasan, Joseph Y. Cheng, Stella X. Yu, Michael Lustig

**<details><summary>Abstract**</summary>

Magnetic Resonance Imaging (MRI) is a powerful medical imaging modality that offers diagnostic information without harmful ionizing radiation. Unlike optical imaging, MRI sequentially samples the spatial Fourier domain (k-space) of the image. Measurements are collected in multiple shots, or readouts, and in each shot, data along a smooth trajectory is sampled.Conventional MRI data acquisition relies on sampling k-space row-by-row in short intervals, which is slow and inefficient. More efficient, non-Cartesian sampling trajectories (e.g., Spirals) use longer data readout intervals, but are more susceptible to magnetic field inhomogeneities, leading to off-resonance artifacts. Spiral trajectories cause off-resonance blurring in the image, and the mathematics of this blurring resembles that of optical blurring, where magnetic field variation corresponds to depth and readout duration to aperture size. Off-resonance blurring is a system issue with a physics-based, accurate forward model. We present a physics-informed deep learning framework for off-resonance correction in MRI, which is trained exclusively on synthetic, noise-like data with representative marginal statistics. Our approach allows for fat/water separation and is compatible with parallel imaging acceleration. Through end-to-end training using synthetic randomized data (i.e., noise-like images, coil sensitivities, field maps), we train the network to reverse off-resonance effects across diverse anatomies and contrasts without retraining. We demonstrate the effectiveness of our approach through results on phantom and in-vivo data. This work has the potential to facilitate the clinical adoption of non-Cartesian sampling trajectories, enabling efficient, rapid, and motion-robust MRI scans. Code is publicly available at: https://github.com/mikgroup/ResoNet.</details>

### ResoNet: a Physics-Informed DL Framework for Off-Resonance Correction in MRI Trained with Noise

**Authors:** Alfredo De Goyeneche Macaya, Shreya Ramachandran, Ke Wang, Ekin Karasan, Joseph Y. Cheng, Stella X. Yu, Michael Lustig

**<details><summary>Abstract**</summary>

Magnetic Resonance Imaging (MRI) is a powerful medical imaging modality that offers diagnostic information without harmful ionizing radiation. Unlike optical imaging, MRI sequentially samples the spatial Fourier domain (k-space) of the image. Measurements are collected in multiple shots, or readouts, and in each shot, data along a smooth trajectory is sampled.Conventional MRI data acquisition relies on sampling k-space row-by-row in short intervals, which is slow and inefficient. More efficient, non-Cartesian sampling trajectories (e.g., Spirals) use longer data readout intervals, but are more susceptible to magnetic field inhomogeneities, leading to off-resonance artifacts. Spiral trajectories cause off-resonance blurring in the image, and the mathematics of this blurring resembles that of optical blurring, where magnetic field variation corresponds to depth and readout duration to aperture size. Off-resonance blurring is a system issue with a physics-based, accurate forward model. We present a physics-informed deep learning framework for off-resonance correction in MRI, which is trained exclusively on synthetic, noise-like data with representative marginal statistics. Our approach allows for fat/water separation and is compatible with parallel imaging acceleration. Through end-to-end training using synthetic randomized data (i.e., noise-like images, coil sensitivities, field maps), we train the network to reverse off-resonance effects across diverse anatomies and contrasts without retraining. We demonstrate the effectiveness of our approach through results on phantom and in-vivo data. This work has the potential to facilitate the clinical adoption of non-Cartesian sampling trajectories, enabling efficient, rapid, and motion-robust MRI scans. Code is publicly available at: https://github.com/mikgroup/ResoNet.</details>

### Responsible AI (RAI) Games and Ensembles

**Authors:** Yash Gupta, Runtian Zhai, Arun Suggala, Pradeep Ravikumar

**<details><summary>Abstract**</summary>

Several recent works have studied the societal effects of AI; these include issues such as fairness, robustness, and safety.  In many of these objectives, a learner seeks to minimize its worst-case loss over a set of predefined distributions (known as uncertainty sets), with usual examples being perturbed versions of the empirical distribution. In other words, the aforementioned problems can be written as min-max problems over these uncertainty sets. In this work, we provide a general framework for studying these problems, which we refer to as Responsible AI (RAI) games. We provide two classes of algorithms for solving these games:  (a) game-play based algorithms, and (b) greedy stagewise estimation algorithms. The former class is motivated by online learning and game theory, whereas the latter class is motivated by the classical statistical literature on boosting, and regression. We empirically demonstrate the applicability and competitive performance of our techniques for solving several RAI problems, particularly around subpopulation shift.</details>

### [Spotlight] Restless Bandits with Average Reward: Breaking the Uniform Global Attractor Assumption

**Authors:** Yige Hong, Qiaomin Xie, Yudong Chen, Weina Wang

**<details><summary>Abstract**</summary>

We study the infinite-horizon Restless Bandit problem with the average reward criterion, under both discrete-time and continuous-time settings.A fundamental goal is to design computationally efficient policies that achieve a diminishing optimality gap as the number of arms, $N$, grows large. Existing results on asymptotic optimality all rely on the uniform global attractor property (UGAP), a complex and challenging-to-verify assumption. In this paper, we propose a general, simulation-based framework, Follow-the-Virtual-Advice, that converts any single-armed policy into a policy for the original $N$-armed problem. This is done by simulating the single-armed policy on each arm and carefully steering the real state towards the simulated state. Our framework can be instantiated to produce a policy with an $O(1/\sqrt{N})$ optimality gap. In the discrete-time setting, our result holds under a simpler synchronization assumption, which covers some problem instances that violate UGAP. More notably, in the continuous-time setting, we do not require \emph{any} additional assumptions beyond the standard unichain condition. In both settings, our work is the first asymptotic optimality result that does not require UGAP.</details>

### Rethinking the Backward Propagation for Adversarial Transferability

**Authors:** Wang Xiaosen, Kangheng Tong, Kun He

**<details><summary>Abstract**</summary>

Transfer-based attacks generate adversarial examples on the surrogate model, which can mislead other black-box models without access, making it promising to attack real-world applications. Recently, several works have been proposed to boost adversarial transferability, in which the surrogate model is usually overlooked. In this work, we identify that non-linear layers (e.g., ReLU, max-pooling, etc.) truncate the gradient during backward propagation, making the gradient w.r.t. input image imprecise to the loss function. We hypothesize and empirically validate that such truncation undermines the transferability of adversarial examples. Based on these findings, we propose a novel method called Backward Propagation Attack (BPA) to increase the relevance between the gradient w.r.t. input image and loss function so as to generate adversarial examples with higher transferability. Specifically, BPA adopts a non-monotonic function as the derivative of ReLU and incorporates softmax with temperature to smooth the derivative of max-pooling, thereby mitigating the information loss during the backward propagation of gradients. Empirical results on the ImageNet dataset demonstrate that not only does our method substantially boost the adversarial transferability, but it is also general to existing transfer-based attacks. Code is available at https://github.com/Trustworthy-AI-Group/RPA.</details>

### Reward Imputation with Sketching for Contextual Batched Bandits

**Authors:** Xiao Zhang, Ninglu Shao, Zihua Si, Jun Xu, Wenhan Wang, Hanjing Su, Ji-Rong Wen

**<details><summary>Abstract**</summary>

Contextual batched bandit (CBB) is a setting where a batch of rewards is observed from the environment at the end of each episode, but the rewards of the non-executed actions are unobserved, resulting in partial-information feedback. Existing approaches for CBB often ignore the rewards of the non-executed actions, leading to underutilization of feedback information. In this paper, we propose an efficient approach called Sketched Policy Updating with Imputed Rewards (SPUIR) that completes the unobserved rewards using sketching, which approximates the full-information feedbacks. We formulate reward imputation as an imputation regularized ridge regression problem that captures the feedback mechanisms of both executed and non-executed actions. To reduce time complexity, we solve the regression problem using randomized sketching. We prove that our approach achieves an instantaneous regret with controllable bias and smaller variance than approaches without reward imputation. Furthermore, our approach enjoys a sublinear regret bound against the optimal policy. We also present two extensions, a rate-scheduled version and a version for nonlinear rewards, making our approach more practical. Experimental results show that SPUIR outperforms state-of-the-art baselines on synthetic, public benchmark, and real-world datasets.</details>

### Reward Scale Robustness for Proximal Policy Optimization via DreamerV3 Tricks

**Authors:** Ryan Sullivan, Akarsh Kumar, Shengyi Huang, John Dickerson, Joseph Suarez

**<details><summary>Abstract**</summary>

Most reinforcement learning methods rely heavily on dense, well-normalized environment rewards. DreamerV3 recently introduced a model-based method with a number of tricks that mitigate these limitations, achieving state-of-the-art on a wide range of benchmarks with a single set of hyperparameters. This result sparked discussion about the generality of the tricks, since they appear to be applicable to other reinforcement learning algorithms. Our work applies DreamerV3's tricks to PPO and is the first such empirical study outside of the original work. Surprisingly, we find that the tricks presented do not transfer as general improvements to PPO. We use a high quality PPO reference implementation and present extensive ablation studies totaling over 10,000 A100 hours on the Arcade Learning Environment and the DeepMind Control Suite. Though our experiments demonstrate that these tricks do not generally outperform PPO, we identify cases where they succeed and offer insight into the relationship between the implementation tricks. In particular, PPO with these tricks performs comparably to PPO on Atari games with reward clipping and significantly outperforms PPO without reward clipping.</details>

### Reward-agnostic Fine-tuning: Provable Statistical Benefits of Hybrid Reinforcement Learning

**Authors:** Gen Li, Wenhao Zhan, Jason Lee, Yuejie Chi, Yuxin Chen

**<details><summary>Abstract**</summary>

This paper studies tabular reinforcement learning (RL) in the hybrid setting, which assumes access to both an offline dataset and online interactions with the unknown environment. A central question boils down to how to efficiently utilize online data to strengthen and complement the offline dataset and enable effective policy fine-tuning. Leveraging recent advances in reward-agnostic exploration and offline RL, we design a three-stage hybrid RL algorithm that beats the best of both worlds --- pure offline RL and pure online RL --- in terms of sample complexities. The proposed algorithm does not require any reward information during data collection. Our theory is developed based on a new notion called **single-policy partial concentrability**, which captures the trade-off between distribution mismatch and miscoverage and guides the interplay between offline and online data.</details>

### RiskQ: Risk-sensitive Multi-Agent Reinforcement Learning Value Factorization

**Authors:** Siqi Shen, Chennan Ma, Chao Li, Weiquan Liu, Yongquan Fu, Songzhu Mei, Xinwang Liu, Cheng Wang

**<details><summary>Abstract**</summary>

Multi-agent systems are characterized by environmental uncertainty, varying policies of agents, and partial observability, which result in significant risks. In the context of Multi-Agent Reinforcement Learning (MARL), learning coordinated and decentralized policies that are sensitive to risk is challenging. To formulate the coordination requirements in risk-sensitive MARL, we introduce the Risk-sensitive Individual-Global-Max (RIGM) principle as a generalization of the Individual-Global-Max (IGM) and Distributional IGM (DIGM) principles. This principle requires that the collection of risk-sensitive action selections of each agent should be equivalent to the risk-sensitive action selection of the central policy. Current MARL value factorization methods do not satisfy the RIGM principle for common risk metrics such as the Value at Risk (VaR) metric or distorted risk measurements. Therefore, we propose RiskQ to address this limitation, which models the joint return distribution by modeling quantiles of it as weighted quantile mixtures of per-agent return distribution utilities. RiskQ satisfies the RIGM principle for the VaR and distorted risk metrics. We show that RiskQ can obtain promising performance through extensive experiments. The source code of RiskQ is available in https://github.com/xmu-rl-3dv/RiskQ.</details>

### RoboCLIP: One Demonstration is Enough to Learn Robot Policies

**Authors:** Sumedh Sontakke, Jesse Zhang, Séb Arnold, Karl Pertsch, Erdem Bıyık, Dorsa Sadigh, Chelsea Finn, Laurent Itti

**<details><summary>Abstract**</summary>

Reward specification is a notoriously difficult problem in reinforcement learning, requiring extensive expert supervision to design robust reward functions. Imitation learning (IL) methods attempt to circumvent these problems by utilizing expert demonstrations instead of using an extrinsic reward function but typically require a large number of in-domain expert demonstrations. Inspired by advances in the field of Video-and-Language Models (VLMs), we present RoboCLIP, an online imitation learning method that uses a single demonstration (overcoming the large data requirement) in the form of a video demonstration or a textual description of the task to generate rewards without manual reward function design. Additionally, RoboCLIP can also utilize out-of-domain demonstrations, like videos of humans solving the task for reward generation, circumventing the need to have the same demonstration and deployment domains. RoboCLIP utilizes pretrained VLMs without any finetuning for reward generation. Reinforcement learning agents trained with RoboCLIP rewards demonstrate 2-3 times higher zero-shot performance than competing imitation learning methods on downstream robot manipulation tasks, doing so using only one video/text demonstration. Visit our website at https://sites.google.com/view/roboclip/home for experiment videos.</details>

### Robust Concept Erasure via Kernelized Rate-Distortion Maximization

**Authors:** Somnath Basu Roy Chowdhury, Nicholas Monath, Kumar Avinava Dubey, Amr Ahmed, Snigdha Chaturvedi

**<details><summary>Abstract**</summary>

Distributed representations provide a vector space that captures meaningful relationships between data instances. The distributed nature of these representations, however, entangles together multiple attributes or concepts of data instances (e.g., the topic or sentiment of a text, characteristics of the author (age, gender, etc), etc).  Recent work has proposed the task of concept erasure, in which rather than making a concept predictable, the goal is to remove an attribute from distributed representations while retaining other information from the original representation space as much as possible.  In this paper, we propose a new distance metric learning-based objective, the Kernelized Rate-Distortion Maximizer (KRaM), for performing concept erasure. KRaM fits a transformation of representations to match a specified distance measure (defined by a labeled concept to erase) using a modified rate-distortion function. Specifically, KRaM's objective function aims to make instances with similar concept labels dissimilar in the learned representation space while retaining other information.  We find that optimizing KRaM effectively erases various types of concepts—categorical, continuous, and vector-valued variables—from data representations across diverse domains. We also provide a theoretical analysis of several properties of KRaM's objective. To assess the quality of the learned representations, we propose an alignment score to evaluate their similarity with the original representation space. Additionally, we conduct experiments to showcase KRaM's efficacy in various settings, from erasing binary gender variables in word embeddings to vector-valued variables in GPT-3 representations.</details>

### Robust Contrastive Language-Image Pretraining against Data Poisoning and Backdoor Attacks

**Authors:** Wenhan Yang, Jingdong Gao, Baharan Mirzasoleiman

**<details><summary>Abstract**</summary>

Contrastive vision-language representation learning has achieved state-of-the-art performance for zero-shot classification, by learning from millions of image-caption pairs crawled from the internet. However, the massive data that powers large multimodal models such as CLIP, makes them extremely vulnerable to various types of targeted data poisoning and backdoor attacks. Despite this vulnerability, robust contrastive vision-language pre-training against such attacks has remained unaddressed. In this work, we propose RoCLIP, the first effective method for robust pre-training multimodal vision-language models against targeted data poisoning and backdoor attacks. RoCLIP effectively breaks the association between poisoned image-caption pairs by considering a relatively large and varying pool of random captions, and matching every image with the text that is most similar to it in the pool instead of its own caption, every few epochs.It also leverages image and text augmentations to further strengthen the defense and improve the performance of the model. Our extensive experiments show that RoCLIP renders state-of-the-art targeted data poisoning and backdoor attacks ineffective during pre-training CLIP models. In particular, RoCLIP decreases the success rate for targeted data poisoning attacks from 93.75% to 12.5% and that of backdoor attacks down to 0%, while improving the model's linear probe performance by 10% and maintains a similar zero shot performance compared to CLIP. By increasing the frequency of matching, RoCLIP is able to defend strong attacks, which add up to 1% poisoned examples to the data, and successfully maintain a low attack success rate of 12.5%, while trading off the performance on some tasks.</details>

### Robust Multi-Agent Reinforcement Learning via Adversarial Regularization: Theoretical Foundation and Stable Algorithms

**Authors:** Alexander Bukharin, Yan Li, Yue Yu, Qingru Zhang, Zhehui Chen, Simiao Zuo, Chao Zhang, Songan Zhang, Tuo Zhao

**<details><summary>Abstract**</summary>

Multi-Agent Reinforcement Learning (MARL) has shown promising results across several domains. Despite this promise, MARL policies often lack robustness and are therefore sensitive to small changes in their environment. This presents a serious concern for the real world deployment of MARL algorithms, where the testing environment may slightly differ from the training environment. In this work we show that we can gain robustness by controlling a policy’s Lipschitz constant, and under mild conditions, establish the existence of a Lipschitz and close-to-optimal policy. Motivated by these insights, we propose a new robust MARL framework, ERNIE, that promotes the Lipschitz continuity of the policies with respect to the state observations and actions by adversarial regularization. The ERNIE framework provides robustness against noisy observations, changing transition dynamics, and malicious actions of agents. However, ERNIE’s adversarial regularization may introduce some training instability. To reduce this instability, we reformulate adversarial regularization as a Stackelberg game. We demonstrate the effectiveness of the proposed framework with extensive experiments in traffic light control and particle environments. In addition, we extend ERNIE to mean-field MARL with a formulation based on distributionally robust optimization that outperforms its non-robust counterpart and is of independent interest. Our code is available at https://github.com/abukharin3/ERNIE.</details>

### Robust covariance estimation with missing values and cell-wise contamination

**Authors:** Grégoire Pacreau, Karim Lounici

**<details><summary>Abstract**</summary>

Large datasets are often affected by cell-wise outliers in the form of missing or erroneous data. However, discarding any samples containing outliers may result in a dataset that is too small to accurately estimate the covariance matrix. Moreover, the robust procedures designed to address this problem require the invertibility of the covariance operator and thus are not effective on high-dimensional data. In this paper, we propose an unbiased estimator for the covariance in the presence of missing values that does not require any imputation step and still achieves near minimax statistical accuracy with the operator norm. We also advocate for its use in combination with cell-wise outlier detection methods to tackle cell-wise contamination in a high-dimensional and low-rank setting, where state-of-the-art methods may suffer from numerical instability and long computation times. To complement our theoretical findings, we conducted an experimental study which demonstrates the superiority of our approach over the state of the art both in low and high dimension settings.</details>

### Robust low-rank training via approximate orthonormal constraints

**Authors:** Dayana Savostianova, Emanuele Zangrando, Gianluca Ceruti, Francesco Tudisco

**<details><summary>Abstract**</summary>

With the growth of model and data sizes, a broad effort has been made to design pruning techniques that reduce the resource demand of deep learning pipelines, while retaining model performance. In order to reduce both inference and training costs, a prominent line of work uses low-rank matrix factorizations to represent the network weights. Although able to retain accuracy,  we observe that low-rank methods tend to compromise model robustness against adversarial perturbations. By modeling robustness in terms of the condition number of the neural network, we argue that this loss of robustness is due to the exploding singular values of the low-rank weight matrices. Thus, we introduce a robust low-rank training algorithm that maintains the network's weights on the low-rank matrix manifold while  simultaneously enforcing approximate orthonormal constraints. The resulting model reduces both training and inference costs while ensuring well-conditioning and thus better adversarial robustness, without compromising model accuracy. This is shown by extensive numerical evidence and by our main approximation theorem that shows the computed robust low-rank network well-approximates the ideal full model, provided a highly performing low-rank sub-network exists.</details>

### Robustifying Generalizable Implicit Shape Networks with a Tunable Non-Parametric Model

**Authors:** Amine Ouasfi, Adnane Boukhayma

**<details><summary>Abstract**</summary>

Feedforward generalizable models for implicit shape reconstruction from unoriented point cloud present multiple advantages, including high performance and inference speed. However, they still suffer from generalization issues, ranging from underfitting the input point cloud, to misrepresenting samples outside of the training data distribution, or with toplogies unseen at training.  We propose here an efficient mechanism to remedy some of these limitations at test time. We combine the inter-shape data prior of the network with an intra-shape regularization prior of a Nyström Kernel Ridge Regression, that we further adapt by fitting its hyperprameters to the current shape. The resulting shape function defined in a shape specific Reproducing Kernel Hilbert Space benefits from desirable stability and efficiency properties and grants a shape adaptive expressiveness-robustness trade-off. We demonstrate the improvement obtained through our method  with respect to baselines and the state-of-the-art using synthetic and real data.</details>

### S-CLIP: Semi-supervised Vision-Language Learning using Few Specialist Captions

**Authors:** Sangwoo Mo, Minkyu Kim, Kyungmin Lee, Jinwoo Shin

**<details><summary>Abstract**</summary>

Vision-language models, such as contrastive language-image pre-training (CLIP), have demonstrated impressive results in natural image domains. However, these models often struggle when applied to specialized domains like remote sensing, and adapting to such domains is challenging due to the limited number of image-text pairs available for training. To address this, we propose S-CLIP, a semi-supervised learning method for training CLIP that utilizes additional unpaired images. S-CLIP employs two pseudo-labeling strategies specifically designed for contrastive learning and the language modality. The caption-level pseudo-label is given by a combination of captions of paired images, obtained by solving an optimal transport problem between unpaired and paired images. The keyword-level pseudo-label is given by a keyword in the caption of the nearest paired image, trained through partial label learning that assumes a candidate set of labels for supervision instead of the exact one. By combining these objectives, S-CLIP significantly enhances the training of CLIP using only a few image-text pairs, as demonstrated in various specialist domains, including remote sensing, fashion, scientific figures, and comics. For instance, S-CLIP improves CLIP by 10% for zero-shot classification and 4% for image-text retrieval on the remote sensing benchmark, matching the performance of supervised CLIP while using three times fewer image-text pairs.</details>

### [Spotlight] SE(3) Equivariant Augmented Coupling Flows

**Authors:** Laurence Midgley, Vincent Stimper, Javier Antorán, Emile Mathieu, Bernhard Schölkopf, José Miguel Hernández-Lobato

**<details><summary>Abstract**</summary>

Coupling normalizing flows allow for fast sampling and density evaluation, making them the tool of choice for probabilistic modeling of physical systems. However, the standard coupling architecture precludes endowing flows that operate on the Cartesian coordinates of atoms with the SE(3) and permutation invariances of physical systems. This work proposes a coupling flow that preserves SE(3) and permutation equivariance by performing coordinate splits along additional augmented dimensions. At each layer, the flow maps atoms' positions into learned SE(3) invariant bases, where we apply standard flow transformations, such as monotonic rational-quadratic splines, before returning to the original basis. Crucially, our flow preserves fast sampling and density evaluation, and may be used to produce unbiased estimates of expectations with respect to the target distribution via importance sampling. When trained on the DW4, LJ13, and QM9-positional datasets, our flow is competitive with equivariant continuous normalizing flows, while allowing sampling more than an order of magnitude faster. Moreover, to the best of our knowledge, we are the first to learn the full Boltzmann distribution of alanine dipeptide by only modeling the Cartesian positions of its atoms. Lastly, we demonstrate that our flow can be trained to approximately sample from the Boltzmann distribution of the DW4 and LJ13 particle systems using only their energy functions.</details>

### SEGA: Instructing Text-to-Image Models using Semantic Guidance

**Authors:** Manuel Brack, Felix Friedrich, Dominik Hintersdorf, Lukas Struppek, Patrick Schramowski, Kristian Kersting

**<details><summary>Abstract**</summary>

Text-to-image diffusion models have recently received a lot of interest for their astonishing ability to produce high-fidelity images from text only. However, achieving one-shot generation that aligns with the user’s intent is nearly impossible, yet small changes to the input prompt often result in very different images. This leaves the user with little semantic control. To put the user in control, we show how to interact with the diffusion process to flexibly steer it along semantic directions. This semantic guidance (SEGA) generalizes to any generative architecture using classifier-free guidance. More importantly, it allows for subtle and extensive edits, composition and style changes, and optimizing the overall artistic conception. We demonstrate SEGA’s effectiveness on both latent and pixel-based diffusion models such as Stable Diffusion, Paella, and DeepFloyd-IF using a variety of tasks, thus providing strong evidence for its versatility and flexibility.</details>

### SHAP-IQ: Unified Approximation of any-order Shapley Interactions

**Authors:** Fabian Fumagalli, Maximilian Muschalik, Patrick Kolpaczki, Eyke Hüllermeier, Barbara Hammer

**<details><summary>Abstract**</summary>

Predominately in explainable artificial intelligence (XAI) research, the Shapley value (SV) is applied to determine feature attributions for any black box model. Shapley interaction indices extend the SV to define any-order feature interactions. Defining a unique Shapley interaction index is an open research question and, so far, three definitions have been proposed, which differ by their choice of axioms. Moreover, each definition requires a specific approximation technique. Here, we propose SHAPley Interaction Quantification (SHAP-IQ), an efficient sampling-based approximator to compute Shapley interactions for arbitrary cardinal interaction indices (CII), i.e. interaction indices that satisfy the linearity, symmetry and dummy axiom. SHAP-IQ is based on a novel representation and, in contrast to existing methods, we provide theoretical guarantees for its approximation quality, as well as estimates for the variance of the point estimates. For the special case of SV, our approach reveals a novel representation of the SV and corresponds to Unbiased KernelSHAP with a greatly simplified calculation. We illustrate the computational efficiency and effectiveness by explaining language, image classification and high-dimensional synthetic models.</details>

### SHOT: Suppressing the Hessian along the Optimization Trajectory for Gradient-Based Meta-Learning

**Authors:** JunHoo Lee, Jayeon Yoo, Nojun Kwak

**<details><summary>Abstract**</summary>

In this paper, we hypothesize that gradient-based meta-learning (GBML) implicitly suppresses the Hessian along the optimization  trajectory in the inner loop. Based on this hypothesis, we introduce an algorithm called  SHOT (Suppressing the Hessian along the Optimization Trajectory) that minimizes the distance between the parameters of the target and reference models to suppress the Hessian in the inner loop. Despite dealing with  high-order terms, SHOT does not increase the computational complexity of the baseline model much.  It is agnostic to both the algorithm and architecture used in GBML, making it highly  versatile and applicable to any GBML baseline. To validate the effectiveness of SHOT,  we conduct empirical tests on standard few-shot learning tasks and qualitatively  analyze its dynamics. We confirm our hypothesis empirically and demonstrate that SHOT  outperforms the corresponding baseline.</details>

### SLIBO-Net: Floorplan Reconstruction via Slicing Box Representation with Local Geometry Regularization

**Authors:** Jheng-Wei Su, Kuei-Yu Tung, Chi-Han Peng, Peter Wonka, Hung-Kuo (James) Chu

**<details><summary>Abstract**</summary>

This paper focuses on improving the reconstruction of 2D floorplans from unstructured 3D point clouds. We identify opportunities for enhancement over the existing methods in three main areas: semantic quality, efficient representation, and local geometric details. To address these, we presents SLIBO-Net, an innovative approach to reconstructing 2D floorplans from unstructured 3D point clouds. We propose a novel transformer-based architecture that employs an efficient floorplan representation, providing improved room shape supervision and allowing for manageable token numbers. By incorporating geometric priors as a regularization mechanism and post-processing step, we enhance the capture of local geometric details. We also propose a scale-independent evaluation metric, correcting the discrepancy in error treatment between varying floorplan sizes. Our approach notably achieves a new state-of-the-art on the Structure3D dataset. The resultant floorplans exhibit enhanced semantic plausibility, substantially improving the overall quality and realism of the reconstructions. Our code and dataset are available online.</details>

### SLM: A Smoothed First-Order Lagrangian Method for Structured Constrained Nonconvex Optimization

**Authors:** Songtao Lu

**<details><summary>Abstract**</summary>

Functional constrained optimization (FCO) has emerged as a powerful tool for solving various machine learning problems. However, with the rapid increase in applications of neural networks in recent years, it has become apparent that both the objective and constraints often involve nonconvex functions, which poses significant challenges in obtaining high-quality solutions. In this work, we focus on a class of nonconvex FCO problems with nonconvex constraints, where the two optimization variables are nonlinearly coupled in the inequality constraint. Leveraging the primal-dual optimization framework, we propose a smoothed first-order Lagrangian method (SLM) for solving this class of problems. We establish the theoretical convergence guarantees of SLM to the Karush-Kuhn-Tucker (KKT) solutions through quantifying dual error bounds. By establishing connections between this structured FCO and equilibrium-constrained nonconvex problems (also known as bilevel optimization), we apply the proposed SLM to tackle bilevel optimization problems where the lower-level problem is nonconvex. Numerical results obtained from both toy examples and hyper-data cleaning problems demonstrate the superiority of SLM compared to benchmark methods.</details>

### SLaM: Student-Label Mixing for  Distillation with Unlabeled Examples

**Authors:** Vasilis Kontonis, Fotis Iliopoulos, Khoa Trinh, Cenk Baykal, Gaurav Menghani, Erik Vee

**<details><summary>Abstract**</summary>

Knowledge distillation with unlabeled examples is a powerful training paradigm for generating compact and lightweight student models in applications where the amount of labeled data is limited but one has access to a large pool of unlabeled data. In this setting, a large teacher model generates "soft" pseudo-labels for the unlabeled dataset which are then used for training the student model. Despite its success in a wide variety of applications, a  shortcoming of this approach is that the teacher's pseudo-labels are often noisy, leading to impaired student performance. In this paper, we present a principled method for knowledge distillation with unlabeled examples that we call Student-Label Mixing (SLaM) and we show that it consistently improves over prior approaches by evaluating it on several standard benchmarks. Finally, we show that SLaM comes with theoretical guarantees; along the way we give an algorithm improving the best-known sample complexity for learning halfspaces with margin under random classification noise, and provide the first convergence analysis for so-called ``forward loss-adjustment" methods.</details>

### SNAP: Self-Supervised Neural Maps for Visual Positioning and Semantic Understanding

**Authors:** Paul-Edouard Sarlin, Eduard Trulls, Marc Pollefeys, Jan Hosang, Simon Lynen

**<details><summary>Abstract**</summary>

Semantic 2D maps are commonly used by humans and machines for navigation purposes, whether it's walking or driving. However, these maps have limitations: they lack detail, often contain inaccuracies, and are difficult to create and maintain, especially in an automated fashion. Can we use  _raw imagery_ to automatically create _better maps_ that can be easily interpreted by both humans and machines? We introduce SNAP, a deep network that learns rich 2D _neural_ maps from ground-level and overhead images. We train our model to align neural maps estimated from different inputs, supervised only with camera poses over tens of millions of StreetView images. SNAP can resolve the location of challenging image queries beyond the reach of traditional methods, outperforming the state of the art in localization by a large margin. Moreover, our neural maps encode not only geometry and appearance but also high-level semantics, discovered without explicit supervision. This enables effective pre-training for data-efficient semantic scene understanding, with the potential to unlock cost-efficient creation of more detailed maps.</details>

### SOAR: Improved Indexing for Approximate Nearest Neighbor Search

**Authors:** Philip Sun, David Simcha, Dave Dopson, Ruiqi Guo, Sanjiv Kumar

**<details><summary>Abstract**</summary>

This paper introduces SOAR: **S**pilling with **O**rthogonality-**A**mplified **R**esiduals, a novel data indexing technique for approximate nearest neighbor (ANN) search. SOAR extends upon previous approaches to ANN search, such as spill trees, that utilize multiple redundant representations while partitioning the data to reduce the probability of missing a nearest neighbor during search. Rather than training and computing these redundant representations independently, however, SOAR uses an *orthogonality-amplified residual* loss, which optimizes each representation to compensate for cases where other representations perform poorly. This drastically improves the overall index quality, resulting in state-of-the-art ANN benchmark performance while maintaining fast indexing times and low memory consumption.</details>

### SOAR: Improved Quantization for Approximate Nearest Neighbor Search

**Authors:** Philip Sun, David Simcha, Dave Dopson, Ruiqi Guo, Sanjiv Kumar

**<details><summary>Abstract**</summary>

This paper introduces SOAR: Spilling with Orthogonality-Amplified Residuals, a novel data indexing technique for approximate nearest neighbor (ANN) search. SOAR extends upon previous approaches to ANN search, such as spill trees, that utilize multiple redundant representations while partitioning the data to reduce the probability of missing a nearest neighbor during search. Rather than training and computing these redundant representations independently, however, SOAR uses an orthogonality-amplified residual loss, which optimizes each representation to compensate for cases where other representations perform poorly. This drastically improves the overall index quality, resulting in state-of-the-art ANN benchmark performance while maintaining fast indexing times and low memory consumption.</details>

### SOC: Semantic-Assisted  Object Cluster for Referring Video Object Segmentation

**Authors:** Zhuoyan Luo, Yicheng Xiao, Yong Liu, Shuyan Li, Yitong Wang, Yansong Tang, Xiu Li, Yujiu Yang

**<details><summary>Abstract**</summary>

This paper studies referring video object segmentation (RVOS) by boosting video-level visual-linguistic alignment. Recent approaches model the RVOS task as a sequence prediction problem and perform multi-modal interaction as well as segmentation for each frame separately. However, the lack of a global view of video content leads to difficulties in effectively utilizing inter-frame relationships and understanding textual descriptions of object temporal variations. To address this issue, we propose Semantic-assisted Object Cluster (SOC), which aggregates video content and textual guidance for unified temporal modeling and cross-modal alignment. By associating a group of frame-level object embeddings with language tokens, SOC facilitates joint space learning across modalities and time steps. Moreover, we present multi-modal contrastive supervision to help construct well-aligned joint space at the video level. We conduct extensive experiments on popular RVOS benchmarks, and our method outperforms state-of-the-art competitors on all benchmarks by a remarkable margin. Besides, the emphasis on temporal coherence enhances the segmentation stability and adaptability of our method in processing text expressions with temporal variations. Code is available at https://github.com/RobertLuo1/NeurIPS2023_SOC.</details>

### STREAMER: Streaming Representation Learning and Event Segmentation in a Hierarchical Manner

**Authors:** Ramy Mounir, Sujal Vijayaraghavan, Sudeep Sarkar

**<details><summary>Abstract**</summary>

We present a novel self-supervised approach for hierarchical representation learning and segmentation of perceptual inputs in a streaming fashion. Our research addresses how to semantically group streaming inputs into chunks at various levels of a hierarchy while simultaneously learning, for each chunk, robust global representations throughout the domain. To achieve this, we propose STREAMER, an architecture that is trained layer-by-layer, adapting to the complexity of the input domain. In our approach, each layer is trained with two primary objectives: making accurate predictions into the future and providing necessary information to other levels for achieving the same objective. The event hierarchy is constructed by detecting prediction error peaks at different levels, where a detected boundary triggers a bottom-up information flow. At an event boundary, the encoded representation of inputs at one layer becomes the input to a higher-level layer. Additionally, we design a communication module that facilitates top-down and bottom-up exchange of information during the prediction process. Notably, our model is fully self-supervised and trained in a streaming manner, enabling a single pass on the training data. This means that the model encounters each input only once and does not store the data. We evaluate the performance of our model on the egocentric EPIC-KITCHENS dataset, specifically focusing on temporal event segmentation. Furthermore, we conduct event retrieval experiments using the learned representations to demonstrate the high quality of our video event representations. Illustration videos and code are available on our project page: https://ramymounir.com/publications/streamer</details>

### SUBP: Soft Uniform Block Pruning for 1$\times$N Sparse CNNs Multithreading Acceleration

**Authors:** JINGYANG XIANG, Siqi Li, Jun Chen, Guang Dai, Shipeng Bai, Yukai Ma, Yong Liu

**<details><summary>Abstract**</summary>

The study of sparsity in Convolutional Neural Networks (CNNs) has become widespread to compress and accelerate models in environments with limited resources. By constraining N consecutive weights along the output channel to be group-wise non-zero, the recent network with 1$\times$N sparsity has received tremendous popularity for its three outstanding advantages: 1) A large amount of storage space saving by a \emph{Block Sparse Row} matrix. 2) Excellent performance at a high sparsity. 3) Significant speedups on CPUs with Advanced Vector Extensions. Recent work requires selecting and fine-tuning 1$	imes$N sparse weights based on dense pre-trained weights, leading to the problems such as expensive training cost and memory access, sub-optimal model quality, as well as unbalanced workload across threads (different sparsity across output channels). To overcome them, this paper proposes a novel \emph{	extbf{S}oft \textbf{U}niform \textbf{B}lock \textbf{P}runing} (SUBP) approach to train a uniform 1$\times$N sparse structured network from scratch. Specifically, our approach tends to repeatedly allow pruned blocks to regrow to the network based on block angular redundancy and importance sampling in a uniform manner throughout the training process. It not only makes the model less dependent on pre-training, reduces the model redundancy and the risk of pruning the important blocks permanently but also achieves balanced workload. Empirically, on ImageNet, comprehensive experiments across various CNN architectures show that our SUBP consistently outperforms existing 1$\times$N and structured sparsity methods based on pre-trained models or training from scratch. Source codes and models are available at \url{https://github.com/JingyangXiang/SUBP}.</details>

### Sample Complexity of Goal-Conditioned Hierarchical Reinforcement Learning

**Authors:** Arnaud Robert, Ciara Pike-Burke, Aldo Faisal

**<details><summary>Abstract**</summary>

Hierarchical Reinforcement Learning (HRL) algorithms can perform planning at multiple levels of abstraction. Empirical results have shown that state or temporal abstractions might significantly improve the sample efficiency of algorithms. Yet, we still do not have a complete understanding of the basis of those efficiency gains nor any theoretically grounded design rules. In this paper, we derive a lower bound on the sample complexity for the considered class of goal-conditioned HRL algorithms. The proposed lower bound empowers us to quantify the benefits of hierarchical decomposition and leads to the design of a simple Q-learning-type algorithm that leverages hierarchical decompositions. We empirically validate our theoretical findings by investigating the sample complexity of the proposed hierarchical algorithm on a spectrum of tasks (hierarchical $n$-rooms, Gymnasium's Taxi). The hierarchical $n$-rooms tasks were designed to allow us to dial their complexity over multiple orders of magnitude. Our theory and algorithmic findings provide a step towards answering the foundational question of quantifying the improvement hierarchical decomposition offers over monolithic solutions in reinforcement learning.</details>

### [Spotlight] Sample Efficient Reinforcement Learning in Mixed Systems through Augmented Samples and Its Applications to Queueing Networks

**Authors:** Honghao Wei, Xin Liu, Weina Wang, Lei Ying

**<details><summary>Abstract**</summary>

This paper considers a class of reinforcement learning problems, which involve systems with two types of states: stochastic and pseudo-stochastic. In such systems, stochastic states follow a stochastic transition kernel while the transitions of pseudo-stochastic states are deterministic {\em given} the stochastic states/transitions. We refer to such systems as mixed systems, which are widely used in various applications, including Manufacturing systems, communication networks, and queueing networks. We propose a sample-efficient RL method that accelerates learning by generating augmented data samples. The proposed algorithm is data-driven (model-free), but it learns the policy from data samples from both real and augmented samples. This method significantly improves learning by reducing the sample complexity such that the dataset only needs to have sufficient coverage of the stochastic states. We analyze the sample complexity of the proposed method under Fitted Q Iteration (FQI) and demonstrate that the optimality gap decreases as  $O\left(\sqrt{\frac{1}{n}}+\sqrt{\frac{1}{m}}\right),$ where $n$ represents the number of real samples, and $m$ is the number of augmented samples per real sample. It is important to note that without augmented samples, the optimality gap is $O(1)$ due to the insufficient data coverage of the pseudo-stochastic states. Our experimental results on multiple queueing network applications confirm that the proposed method indeed significantly accelerates both deep Q-learning and deep policy gradient.</details>

### Scalable Primal-Dual Actor-Critic Method for Safe Multi-Agent RL with General Utilities

**Authors:** Donghao Ying, Yunkai Zhang, Yuhao Ding, Alec Koppel, Javad Lavaei

**<details><summary>Abstract**</summary>

We investigate safe multi-agent reinforcement learning, where agents seek to collectively maximize an aggregate sum of local objectives while satisfying their own safety constraints. The objective and constraints are described by general utilities, i.e., nonlinear functions of the long-term state-action occupancy measure, which encompass broader decision-making goals such as risk, exploration, or imitations. The exponential growth of the state-action space size with the number of agents presents challenges for global observability, further exacerbated by the global coupling arising from agents' safety constraints. To tackle this issue, we propose a primal-dual method utilizing shadow reward and $\kappa$-hop neighbor truncation under a form of correlation decay property, where $\kappa$ is the communication radius. In the exact setting, our algorithm converges to a first-order stationary point (FOSP) at the rate of $\mathcal{O}\left(T^{-2/3}\right)$. In the sample-based setting, we demonstrate that, with high probability, our algorithm requires $\widetilde{\mathcal{O}}\left(\epsilon^{-3.5}\right)$ samples to achieve an $\epsilon$-FOSP with an approximation error of $\mathcal{O}(\phi_0^{2\kappa})$, where $\phi_0\in (0,1)$. Finally, we demonstrate the effectiveness of our model through extensive numerical experiments.</details>

### [Spotlight] Scale Alone Does not Improve Mechanistic Interpretability in Vision Models

**Authors:** Roland S. Zimmermann, Thomas Klein, Wieland Brendel

**<details><summary>Abstract**</summary>

In light of the recent widespread adoption of AI systems, understanding the internal information processing of neural networks has become increasingly critical. Most recently, machine vision has seen remarkable progress by scaling neural networks to unprecedented levels in dataset and model size. We here ask whether this extraordinary increase in scale also positively impacts the field of mechanistic interpretability. In other words, has our understanding of the inner workings of scaled neural networks improved as well? We use a psychophysical paradigm to quantify one form of mechanistic interpretability for a diverse suite of nine models and find no scaling effect for interpretability - neither for model nor dataset size. Specifically, none of the investigated state-of-the-art models are easier to interpret than the GoogLeNet model from almost a decade ago. Latest-generation vision models appear even less interpretable than older architectures, hinting at a regression rather than improvement, with modern models sacrificing interpretability for accuracy. These results highlight the need for models explicitly designed to be mechanistically interpretable and the need for more helpful interpretability methods to increase our understanding of networks at an atomic level. We release a dataset containing more than 130'000 human responses from our psychophysical evaluation of 767 units across nine models. This dataset facilitates research on automated instead of human-based interpretability evaluations, which can ultimately be leveraged to directly optimize the mechanistic interpretability of models.</details>

### Scaling MLPs: A Tale of Inductive Bias

**Authors:** Gregor Bachmann, Sotiris Anagnostidis, Thomas Hofmann

**<details><summary>Abstract**</summary>

In this work we revisit the most fundamental building block in deep learning, the multi-layer perceptron (MLP), and study the limits of its performance on vision tasks. Empirical insights into MLPs are important for multiple reasons. (1) Given the recent narrative "less inductive bias is better", popularized due to transformers eclipsing convolutional models, it is natural to explore the limits of this hypothesis. To that end, MLPs offer an ideal test bed, as they lack any vision-specific inductive bias. (2) MLPs have almost exclusively been the main protagonist in the deep learning theory literature due to their mathematical simplicity, serving as a proxy to explain empirical phenomena observed for more complex architectures. Surprisingly, experimental datapoints for MLPs are very difficult to find in the literature, especially when coupled with large pre-training protocols. This discrepancy between practice and theory is worrying: \textit{Do MLPs reflect the empirical advances exhibited by practical models?} Or do theorists need to rethink the role of MLPs as a proxy? We provide insights into both these aspects.We show that the performance of MLPs drastically improves with scale (95% on CIFAR10, 82% on CIFAR100, 58% on ImageNet ReaL), highlighting that lack of inductive bias can indeed be compensated. We observe that MLPs mimic the behaviour of their modern counterparts faithfully, with some components in the learning setting however exhibiting stronger or unexpected behaviours. Due to their inherent computational efficiency, large pre-training experiments become more accessible for academic researchers. All of our experiments were run on a single GPU.</details>

### Scaling Riemannian Diffusion Models

**Authors:** Aaron Lou, Minkai Xu, Adam Farris, Stefano Ermon

**<details><summary>Abstract**</summary>

Riemannian diffusion models draw inspiration from standard Euclidean space diffusion models to learn distributions on general manifolds. Unfortunately, the additional geometric complexity renders the diffusion transition term inexpressible in closed form, so prior methods resort to imprecise approximations of the score matching training objective that degrade performance and preclude applications in high dimensions. In this work, we reexamine these approximations and propose several practical improvements. Our key observation is that most relevant manifolds are symmetric spaces, which are much more amenable to computation. By leveraging and combining various ans\"{a}tze, we can quickly compute relevant quantities to high precision. On low dimensional datasets, our correction produces a noticeable improvement and is competitive with other techniques. Additionally, we show that our method enables us to scale to high dimensional tasks on nontrivial manifolds, including $SU(n)$ lattices in the context of lattice quantum chromodynamics (QCD). Finally, we apply our models to contrastively learned hyperspherical embeddings, curbing the representation collapse problem in the projection head and closing the gap between theory and practice.</details>

### SceneScape: Text-Driven Consistent Scene Generation

**Authors:** Rafail Fridman, Amit Abecasis, Yoni Kasten, Tali Dekel

**<details><summary>Abstract**</summary>

We present a method for text-driven perpetual view generation -- synthesizing long-term videos of various scenes solely, given an input text prompt describing the scene and camera poses. We introduce a novel framework that generates such videos in an online fashion by combining the generative power of a pre-trained text-to-image model with the geometric priors learned by a pre-trained monocular depth prediction model. To tackle the pivotal challenge of achieving 3D consistency, i.e., synthesizing videos that depict geometrically-plausible scenes, we deploy an online test-time training to encourage the predicted depth map of the current frame to be geometrically consistent with the synthesized scene. The depth maps are used to construct a \emph{unified} mesh representation of the scene, which is progressively constructed along the video generation process. In contrast to previous works, which are applicable only to limited domains, our method generates diverse scenes, such as walkthroughs in spaceships, caves, or ice castles.</details>

### Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time

**Authors:** Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, Anshumali Shrivastava

**<details><summary>Abstract**</summary>

Large language models(LLMs) have sparked a new wave of exciting AI applications. Hosting these models at scale requires significant memory resources. One crucial memory bottleneck for the deployment stems from the context window. It is commonly recognized that model weights are memory hungry; however, the size of key-value embedding stored during the generation process (KV cache) can easily surpass the model size. The enormous size of the KV cache puts constraints on the inference batch size, which is crucial for high throughput inference workload. Inspired by an interesting observation of the attention scores, we hypothesize the persistence of importance: only pivotal tokens, which had a substantial influence at one step, will significantly influence future generations. Based on our empirical verification and theoretical analysis around this hypothesis, we propose scissorhands, a system that maintains the memory usage of the KV cache at a fixed budget without finetuning the model. In essence, Scissorhands manages the KV cache by storing the pivotal tokens with a higher probability. We validate that scissorhands reduces the inference memory usage of the KV cache by up to 5$\times$ without compromising model quality. We further demonstrate that scissorhands can be combined with 4-bit quantization, traditionally used to compress model weights, to achieve up to 20$\times$ compression.</details>

### [Spotlight] Score-based Generative Models with Lévy Processes

**Authors:** EUN BI YOON, Keehun Park, Sungwoong Kim, Sungbin Lim

**<details><summary>Abstract**</summary>

Investigating the optimal stochastic process beyond Gaussian for noise injection in a score-based generative model remains an open question. Brownian motion is a light-tailed process with continuous paths, which leads to a slow convergence rate for the Number of Function Evaluation (NFE). Recent studies have shown that diffusion models suffer from mode-collapse issues on imbalanced data.In order to overcome the limitations of Brownian motion, we introduce a novel score-based generative model referred to as Lévy-Itō Model (LIM). This model utilizes isotropic $\alpha$-stable Lévy processes. We first derive an exact reverse-time stochastic differential equation driven by the Lévy process and develop the corresponding fractional denoising score matching. The proposed generative model takes advantage of the heavy-tailed properties of the Lévy process. Our experimental results show LIM allows for faster and more diverse sampling while maintaining high fidelity compared to existing diffusion models across various image datasets such as CIFAR10, CelebA, and imbalanced dataset CIFAR10LT. Comparing our results to those of DDPM with 3.21 Fréchet Inception Distance (FID) and 0.6437 Recall on the CelebA dataset, we achieve 1.58 FID and 0.7006 Recall using the same architecture. LIM shows the best performance in NFE 500 with $2\times$ faster total wall-clock time than the baseline.</details>

### SegRefiner: Towards Model-Agnostic Segmentation Refinement with Discrete Diffusion Process

**Authors:** Mengyu Wang, Henghui Ding, Jun Hao Liew, Jiajun Liu, Yao Zhao, Yunchao Wei

**<details><summary>Abstract**</summary>

In this paper, we explore a principal way to enhance the quality of object masks produced by different segmentation models. We propose a model-agnostic solution called SegRefiner, which offers a novel perspective on this problem by interpreting segmentation refinement as a data generation process. As a result, the refinement process can be smoothly implemented through a series of denoising diffusion steps. Specifically, SegRefiner takes coarse masks as inputs and refines them using a discrete diffusion process. By predicting the label and corresponding states-transition probabilities for each pixel, SegRefiner progressively refines the noisy masks in a conditional denoising manner. To assess the effectiveness of SegRefiner, we conduct comprehensive experiments on various segmentation tasks, including semantic segmentation, instance segmentation, and dichotomous image segmentation. The results demonstrate the superiority of our SegRefiner from multiple aspects. Firstly, it consistently improves both the segmentation metrics and boundary metrics across different types of coarse masks. Secondly, it outperforms previous model-agnostic refinement methods by a significant margin. Lastly, it exhibits a strong capability to capture extremely fine details when refining high-resolution images. The source code and trained models are available at [SegRefiner.git](https://github.com/MengyuWang826/SegRefiner)</details>

### Selectively Sharing Experiences Improves Multi-Agent Reinforcement Learning

**Authors:** Matthias Gerstgrasser, Tom Danino, Sarah Keren

**<details><summary>Abstract**</summary>

We present a novel multi-agent RL approach, Selective Multi-Agent Prioritized Experience Relay, in which agents share with other agents a limited number of transitions they observe during training. The intuition behind this is that even a small number of relevant experiences from other agents could help each agent learn. Unlike many other multi-agent RL algorithms, this approach allows for largely decentralized training, requiring only a limited communication channel between agents. We show that our approach outperforms baseline no-sharing decentralized training and state-of-the art multi-agent RL algorithms. Further, sharing only a small number of highly relevant experiences outperforms sharing all experiences between agents, and the performance uplift from selective experience sharing is robust across a range of hyperparameters and DQN variants.</details>

### Semantic Image Synthesis with Unconditional Generator

**Authors:** JungWoo Chae, Hyunin Cho, Sooyeon Go, Kyungmook Choi, Youngjung Uh

**<details><summary>Abstract**</summary>

Semantic image synthesis (SIS) aims to generate realistic images according to semantic masks given by a user. Although recent methods produce high quality results with fine spatial control, SIS requires expensive pixel-level annotation of the training images. On the other hand, manipulating intermediate feature maps in a pretrained unconditional generator such as StyleGAN supports coarse spatial control without heavy annotation. In this paper, we introduce a new approach, for reflecting user's detailed guiding masks on a pretrained unconditional generator. Our method converts a user's guiding mask to a proxy mask through a semantic mapper. Then the proxy mask conditions the resulting image through a rearranging network based on cross-attention mechanism. The proxy mask is simple clustering of intermediate feature maps in the generator. The semantic mapper and the rearranging network are easy to train (less than half an hour). Our method is useful for many tasks: semantic image synthesis, spatially editing real images, and unaligned local transplantation. Last but not least, it is generally applicable to various datasets such as human faces, animal faces, and churches.</details>

### Sequential Preference Ranking for Efficient Reinforcement Learning from Human Feedback

**Authors:** Minyoung Hwang, Gunmin Lee, Hogun Kee, Chan Woo Kim, Kyungjae Lee, Songhwai Oh

**<details><summary>Abstract**</summary>

Reinforcement learning from human feedback (RLHF) alleviates the problem of designing a task-specific reward function in reinforcement learning by learning it from human preference. However, existing RLHF models are considered inefficient as they produce only a single preference data from each human feedback. To tackle this problem, we propose a novel RLHF framework called SeqRank, that uses sequential preference ranking to enhance the feedback efficiency. Our method samples trajectories in a sequential manner by iteratively selecting a defender from the set of previously chosen trajectories $\mathcal{K}$ and a challenger from the set of unchosen trajectories $\mathcal{U}\setminus\mathcal{K}$, where $\mathcal{U}$ is the replay buffer. We propose two trajectory comparison methods with different defender sampling strategies: (1) sequential pairwise comparison that selects the most recent trajectory and (2) root pairwise comparison that selects the most preferred trajectory from $\mathcal{K}$. We construct a data structure and rank trajectories by preference to augment additional queries. The proposed method results in at least 39.2% higher average feedback efficiency than the baseline and also achieves a balance between feedback efficiency and data dependency. We examine the convergence of the empirical risk and the generalization bound of the reward model with Rademacher complexity. While both trajectory comparison methods outperform conventional pairwise comparison, root pairwise comparison improves the average reward in locomotion tasks and the average success rate in manipulation tasks by 29.0% and 25.0%, respectively. The source code and the videos are provided in the supplementary material.</details>

### Sheaf Hypergraph Networks

**Authors:** Iulia Duta, Giulia Cassarà, Fabrizio Silvestri, Pietro Lió

**<details><summary>Abstract**</summary>

Higher-order relations are widespread in nature, with numerous phenomena involving complex interactions that extend beyond simple pairwise connections. As a result, advancements in higher-order processing can accelerate the growth of various fields requiring structured data. Current approaches typically represent these interactions using hypergraphs.We enhance this representation by introducing cellular sheaves for hypergraphs, a mathematical construction that adds extra structure to the conventional hypergraph while maintaining their local, higher-order connectivity. Drawing inspiration from existing Laplacians in the literature, we develop two unique formulations of sheaf hypergraph Laplacians: linear and non-linear. Our theoretical analysis demonstrates that incorporating sheaves into the hypergraph Laplacian provides a more expressive inductive bias than standard hypergraph diffusion, creating a powerful instrument for effectively modelling complex data structures.We employ these sheaf hypergraph Laplacians to design two categories of models: Sheaf Hypergraph Neural Networks and Sheaf Hypergraph Convolutional Networks. These models generalize classical Hypergraph Networks often found in the literature. Through extensive experimentation, we show that this generalization significantly improves performance, achieving top results on multiple benchmark datasets for hypergraph node classification.</details>

### Should Under-parameterized Student Networks Copy or Average Teacher Weights?

**Authors:** Berfin Simsek, Amire Bendjeddou, Wulfram Gerstner, Johanni Brea

**<details><summary>Abstract**</summary>

Any continuous function $f^*$ can be approximated arbitrarily well by a neural network with sufficiently many neurons $k$. We consider the case when $f^*$ itself is a neural network with one hidden layer and $k$ neurons. Approximating $f^*$ with a neural network with $n< k$ neurons can thus be seen as fitting an under-parameterized "student" network with $n$ neurons to a "teacher" network with $k$ neurons. As the student has fewer neurons than the teacher, it is unclear, whether each of the $n$ student neurons should copy one of the teacher neurons or rather average a group of teacher neurons. For shallow neural networks with erf activation function and for the standard Gaussian input distribution, we prove that "copy-average" configurations are critical points if the teacher's incoming vectors are orthonormal and its outgoing weights are unitary. Moreover, the optimum among such configurations is reached when $n-1$ student neurons each copy one teacher neuron and the $n$-th student neuron averages the remaining $k-n+1$ teacher neurons. For the student network with $n=1$ neuron, we provide additionally a closed-form solution of the non-trivial critical point(s) for commonly used activation functions through solving an equivalent constrained optimization problem. Empirically, we find for the erf activation function that gradient flow converges either to the optimal copy-average critical point or to another point where each student neuron approximately copies a different teacher neuron. Finally, we find similar results for the ReLU activation function, suggesting that the optimal solution of underparameterized networks has a universal structure.</details>

### Simple, Scalable and Effective Clustering via One-Dimensional Projections

**Authors:** Moses Charikar, Monika Henzinger, Lunjia Hu, Maximilian Vötsch, Erik Waingarten

**<details><summary>Abstract**</summary>

Clustering is a fundamental problem in unsupervised machine learning with many applications in data analysis. Popular clustering algorithms such as Lloyd's algorithm and $k$-means++ can take $\Omega(ndk)$ time when clustering $n$ points in a $d$-dimensional space (represented by an $n\times d$ matrix $X$) into $k$ clusters. On massive datasets with moderate to large $k$, the multiplicative $k$ factor can become very expensive. We introduce a simple randomized clustering algorithm that provably runs in expected time $O(\mathsf{nnz}(X) + n\log n)$ for arbitrary $k$. Here $\mathsf{nnz}(X)$ is the total number of non-zero entries in the input dataset $X$, which is upper bounded by $nd$ and can be significantly smaller for sparse datasets. We prove that our algorithm achieves approximation ratio $\widetilde{O}(k^4)$ on any input dataset for the $k$-means objective, and our experiments show that the quality of the clusters found by our algorithm is usually much better than this worst-case bound. We use our algorithm for $k$-means clustering and for coreset construction; our experiments show that it gives a new tradeoff between running time and cluster quality compared to previous state-of-the-art methods for these tasks. Our theoretical analysis is based on novel results of independent interest. We show that the approximation ratio achieved after a random one-dimensional projection can be lifted to the original points and that $k$-means++ seeding can be implemented in expected time $O(n\log n)$ in one dimension.</details>

### Simplicity Bias in 1-Hidden Layer Neural Networks

**Authors:** Depen Morwani, Jatin Batra, Prateek Jain, Praneeth Netrapalli

**<details><summary>Abstract**</summary>

Recent works have demonstrated that neural networks exhibit extreme *simplicity bias* (SB). That is,  they learn *only the simplest* features  to solve a task at hand, even in the presence of other, more robust but more complex features. Due to the lack of a general and rigorous definition of *features*, these works showcase SB on *semi-synthetic* datasets such as Color-MNIST , MNIST-CIFAR where defining features is relatively easier. In this work, we rigorously define as well as thoroughly establish SB for *one hidden layer* neural networks in the infinite width regime. More concretely, (i) we define SB as the network essentially being a function of a low dimensional projection of the inputs (ii) theoretically, we show that when the data is linearly separable, the network primarily depends on only the linearly separable ($1$-dimensional) subspace even in the presence of an arbitrarily large number of other, more complex features which could have led to a significantly more robust classifier,  (iii) empirically, we show that models trained on *real* datasets such as Imagenet and Waterbirds-Landbirds indeed depend on a low dimensional projection of the inputs, thereby demonstrating SB on these datasets, iv) finally, we present a natural ensemble approach that encourages diversity in  models by training successive models on features not used by earlier models, and demonstrate that it yields models that are significantly more robust to Gaussian noise.</details>

### Sketchy: Memory-efficient Adaptive Regularization with Frequent Directions

**Authors:** Vladimir Feinberg, Xinyi Chen, Y. Jennifer Sun, Rohan Anil, Elad Hazan

**<details><summary>Abstract**</summary>

Adaptive regularization methods that exploit more than the diagonal entries exhibit state of the art performance for many tasks, but can be prohibitive in terms of memory and running time. We find the spectra of the Kronecker-factored gradient covariance matrix in deep learning (DL) training tasks are concentrated on a small leading eigenspace that changes throughout training, motivating a low-rank sketching approach. We describe a generic method for reducing memory and compute requirements of maintaining a matrix preconditioner using the Frequent Directions (FD) sketch. While previous approaches have explored applying FD for second-order optimization, we present a novel analysis which allows efficient interpolation between resource requirements and the degradation in regret guarantees with rank $k$: in the online convex optimization (OCO) setting over dimension $d$, we match full-matrix $d^2$ memory regret using only $dk$ memory up to additive error in the bottom $d-k$ eigenvalues of the gradient covariance. Further, we show extensions of our work to Shampoo, resulting in a method competitive in quality with Shampoo and Adam, yet requiring only sub-linear memory for tracking second moments.</details>

### [Spotlight] Skill-it! A data-driven skills framework for understanding and training language models

**Authors:** Mayee Chen, Nicholas Roberts, Kush Bhatia, Jue WANG, Ce Zhang, Frederic Sala, Christopher Ré

**<details><summary>Abstract**</summary>

The quality of training data impacts the performance of pre-trained large language models (LMs). Given a fixed budget of tokens, we study how to best select data that leads to good downstream model performance across tasks. We develop a new framework based on a simple hypothesis: just as humans acquire interdependent skills in a deliberate order, language models also follow a natural order when learning a set of skills from their training data. If such an order exists, it can be utilized for improved understanding of LMs and for data-efficient training. Using this intuition, our framework formalizes the notion of a skill and of an ordered set of skills in terms of the associated data. First, using both synthetic and real data, we demonstrate that these ordered skill sets exist, and that their existence enables more advanced skills to be learned with less data when we train on their prerequisite skills. Second, using our proposed framework, we introduce an online data sampling algorithm, Skill-It, over mixtures of skills for both continual pre-training and fine-tuning regimes, where the objective is to efficiently learn multiple skills in the former and an individual skill in the latter. On the LEGO synthetic in the continual pre-training setting, Skill-It obtains 37.5 points higher accuracy than random sampling. On the Natural Instructions dataset in the fine-tuning setting, Skill-It reduces the validation loss on the target skill by 13.6% versus training on data associated with the target skill itself. We apply our skills framework on the RedPajama dataset to continually pre-train a 3B-parameter LM, achieving higher accuracy on the LM Evaluation Harness with 1B tokens than the baseline approach of sampling uniformly over data sources with 3B tokens.</details>

### [Spotlight] SlotDiffusion: Object-Centric Generative Modeling with Diffusion Models

**Authors:** Ziyi Wu, Jingyu Hu, Wuyue Lu, Igor Gilitschenski, Animesh Garg

**<details><summary>Abstract**</summary>

Object-centric learning aims to represent visual data with a set of object entities (a.k.a. slots), providing structured representations that enable systematic generalization.Leveraging advanced architectures like Transformers, recent approaches have made significant progress in unsupervised object discovery.In addition, slot-based representations hold great potential for generative modeling, such as controllable image generation and object manipulation in image editing.However, current slot-based methods often produce blurry images and distorted objects, exhibiting poor generative modeling capabilities.In this paper, we focus on improving slot-to-image decoding, a crucial aspect for high-quality visual generation.We introduce SlotDiffusion -- an object-centric Latent Diffusion Model (LDM) designed for both image and video data.Thanks to the powerful modeling capacity of LDMs, SlotDiffusion surpasses previous slot models in unsupervised object segmentation and visual generation across six datasets.Furthermore, our learned object features can be utilized by existing object-centric dynamics models, improving video prediction quality and downstream temporal reasoning tasks.Finally, we demonstrate the scalability of SlotDiffusion to unconstrained real-world datasets such as PASCAL VOC and COCO, when integrated with self-supervised pre-trained image encoders.</details>

### Smooth Flipping Probability for Differential Private Sign Random Projection Methods

**Authors:** Ping Li, Xiaoyun Li

**<details><summary>Abstract**</summary>

We develop a series of differential privacy (DP) algorithms from a family of random projection (RP) and sign random projection (SignRP) methods. We first show how to improve the previous DP-RP approach using the ``optimal Gaussian mechanism''. Then, we propose a series of DP-SignRP algorithms that leverage the robustness of the ``sign flipping probability'' of random projections. That is, given $x = \sum_{i=1}^p u_i w_{i}$ where $u$ is a $p$-dimensional data vector and $w$ is a symmetric random vector,  $sign(x)$ only has a fairly small probability to be flipped if there is a small modification on data $u$, depending on the specific distribution of $w$. This robustness leads to our novel design of ``smooth flipping probability'' for SignRP-type algorithms with better utility than using the standard randomized response mechanism. Retrieval and classification experiments demonstrate that,  among the presented DP-RP  algorithms,  \textbf{DP-SignOPORP} (where OPORP  is an improvement over the celebrated  count-sketch algorithms), performs the best in general.In the industrial practice, DP methods were not very popular for machine learning or search, largely because the performance typically would  drop substantially if DP is applied. Since our proposed new DP algorithms have significantly improved the  performance, it is anticipated that our work will motivate a wide adoption of DP in practice. Finally, we   stress that, since our  methods are applied to the original data (i.e., feature vectors), the privacy of  downstream tasks is naturally protected.</details>

### Speculative Decoding with Big Little Decoder

**Authors:** Sehoon Kim, Karttikeya Mangalam, Suhong Moon, Jitendra Malik, Michael Mahoney, Amir Gholami, Kurt Keutzer

**<details><summary>Abstract**</summary>

The recent emergence of Large Language Models based on the Transformer architecture has enabled dramatic advancements in the field of Natural Language Processing. However, these models have long inference latency, which limits their deployment and makes them prohibitively expensive for various real-time applications. The inference latency is further exacerbated by autoregressive generative tasks, as models need to run iteratively to generate tokens sequentially without leveraging token-level parallelization. To address this, we propose Big Little Decoder (BiLD), a framework that can improve inference efficiency and latency for a wide range of text generation applications. The BiLD framework contains two models with different sizes that collaboratively generate text. The small model runs autoregressively to generate text with a low inference cost, and the large model is only invoked occasionally to refine the small model’s inaccurate predictions in a non-autoregressive manner. To coordinate the small and large models, BiLD introduces two simple yet effective policies: (1) the fallback policy that determines when to hand control over to the large model; and (2) the rollback policy that determines when the large model needs to correct the small model's inaccurate predictions. To evaluate our framework across different tasks and models, we apply BiLD to various text generation scenarios encompassing machine translation on IWSLT 2017 De-En and WMT 2014 De-En, and summarization on XSUM and CNN/DailyMail. On an NVIDIA T4 GPU, our framework achieves a speedup of up to 2.12x speedup with minimal generation quality degradation. Furthermore, our framework is fully plug-and-play and can be applied without any modifications in the training process or model architecture. Our code is open-sourced.</details>

### Spiking PointNet: Spiking Neural Networks for Point Clouds

**Authors:** Dayong Ren, Zhe Ma, Yuanpei Chen, Weihang Peng, Xiaode Liu, Yuhan Zhang, Yufei Guo

**<details><summary>Abstract**</summary>

Recently, Spiking Neural Networks (SNNs), enjoying extreme energy efficiency, have drawn much research attention on 2D visual recognition and shown gradually increasing application potential. However, it still remains underexplored whether SNNs can be generalized to 3D recognition. To this end, we present Spiking PointNet in the paper, the first spiking neural model for efficient deep learning on point clouds. We discover that the two huge obstacles limiting the application of SNNs in point clouds are: the intrinsic optimization obstacle of SNNs that impedes the training of a big spiking model with large time steps, and the expensive memory and computation cost of PointNet that makes training a big spiking point model unrealistic. To solve the problems simultaneously, we present a trained-less but learning-more paradigm for Spiking PointNet with theoretical justifications and in-depth experimental analysis. In specific, our Spiking PointNet is trained with only a single time step but can obtain better performance with multiple time steps inference, compared to the one trained directly with multiple time steps. We conduct various experiments on ModelNet10, ModelNet40 to demonstrate the effectiveness of Sipiking PointNet. Notably, our Spiking PointNet even can outperform its ANN counterpart, which is rare in the SNN field thus providing a potential research direction for the following work. Moreover, Spiking PointNet shows impressive speedup and storage saving in the training phase. Our code is open-sourced at https://github.com/DayongRen/Spiking-PointNet.</details>

### [Spotlight] Squared Neural Families: A New Class of Tractable Density Models

**Authors:** Russell Tsuchida, Cheng Soon Ong, Dino Sejdinovic

**<details><summary>Abstract**</summary>

Flexible models for probability distributions are an essential ingredient in many machine learning tasks. We develop and investigate a new class of probability distributions, which we call a Squared Neural Family (SNEFY), formed by squaring the 2-norm of a neural network and normalising it with respect to a base measure. Following the reasoning similar to the well established connections between infinitely wide neural networks and Gaussian processes, we show that SNEFYs admit closed form normalising constants in many cases of interest, thereby resulting in flexible yet fully tractable density models. SNEFYs strictly generalise classical exponential families, are closed under conditioning, and have tractable marginal distributions. Their utility is illustrated on a variety of density estimation, conditional density estimation, and density estimation with missing data tasks.</details>

### Stability-penalty-adaptive follow-the-regularized-leader: Sparsity, game-dependency, and best-of-both-worlds

**Authors:** Taira Tsuchiya, Shinji Ito, Junya Honda

**<details><summary>Abstract**</summary>

Adaptivity to the difficulties of a problem is a key property in sequential decision-making problems to broaden the applicability of algorithms. Follow-the-regularized-leader (FTRL) has recently emerged as one of the most promising approaches for obtaining various types of adaptivity in bandit problems. Aiming to further generalize this adaptivity, we develop a generic adaptive learning rate, called stability-penalty-adaptive (SPA) learning rate for FTRL. This learning rate yields a regret bound jointly depending on stability and penalty of the algorithm, into which the regret of FTRL is typically decomposed. With this result, we establish several algorithms with three types of adaptivity: sparsity, game-dependency, and best-of-both-worlds (BOBW). Despite the fact that sparsity appears frequently in real problems, existing sparse multi-armed bandit algorithms with $k$-arms assume that the sparsity level $s \leq k$ is known in advance, which is often not the case in real-world scenarios. To address this issue, we first establish $s$-agnostic algorithms with regret bounds of $\tilde{O}(\sqrt{sT})$ in the adversarial regime for $T$ rounds, which matches the existing lower bound up to a logarithmic factor. Meanwhile, BOBW algorithms aim to achieve a near-optimal regret in both the stochastic and adversarial regimes. Leveraging the SPA learning rate and the technique for $s$-agnostic algorithms combined with a new analysis to bound the variation in FTRL output in response to changes in a regularizer, we establish the first BOBW algorithm with a sparsity-dependent bound. Additionally, we explore partial monitoring and demonstrate that the proposed SPA learning rate framework allows us to achieve a game-dependent bound and the BOBW simultaneously.</details>

### StableFDG: Style and Attention Based Learning for Federated Domain Generalization

**Authors:** Jungwuk Park, Dong-Jun Han, Jinho Kim, Shiqiang Wang, Christopher Brinton, Jaekyun Moon

**<details><summary>Abstract**</summary>

Traditional federated learning (FL) algorithms operate under the assumption that the data distributions at training (source domains) and testing (target domain) are the same. The fact that domain shifts often occur in practice necessitates equipping FL methods with a domain generalization (DG) capability. However, existing DG algorithms face fundamental challenges in FL setups due to the lack of samples/domains in each client’s local dataset. In this paper, we propose StableFDG, a style and attention based learning strategy for accomplishing federated domain generalization, introducing two key contributions. The first is style-based learning, which enables each client to explore novel styles beyond the original source domains in its local dataset, improving domain diversity based on the proposed style sharing, shifting, and exploration strategies. Our second contribution is an attention-based feature highlighter, which captures the similarities between the features of data samples in the same class, and emphasizes the important/common characteristics to better learn the domain-invariant characteristics of each class in data-poor FL scenarios. Experimental results show that StableFDG outperforms existing baselines on various DG benchmark datasets, demonstrating its efficacy.</details>

### State Regularized Policy Optimization on Data with Dynamics Shift

**Authors:** Zhenghai Xue, Qingpeng Cai, Shuchang Liu, Dong Zheng, Peng Jiang, Kun Gai, Bo An

**<details><summary>Abstract**</summary>

In many real-world scenarios, Reinforcement Learning (RL) algorithms are trained on data with dynamics shift, i.e., with different underlying environment dynamics. A majority of current methods address such issue by training context encoders to identify environment parameters. Data with dynamics shift are separated according to their environment parameters to train the corresponding policy.However, these methods can be sample inefficient as data are used \textit{ad hoc}, and policies trained for one dynamics cannot benefit from data collected in all other environments with different dynamics. In this paper, we find that in many environments with similar structures and different dynamics, optimal policies have similar stationary state distributions. We exploit such property and learn the stationary state distribution from data with dynamics shift for efficient data reuse. Such distribution is used to regularize the policy trained in a new environment, leading to the SRPO (\textbf{S}tate \textbf{R}egularized \textbf{P}olicy \textbf{O}ptimization) algorithm. To conduct theoretical analyses, the intuition of similar environment structures is characterized by the notion of homomorphous MDPs. We then demonstrate a lower-bound performance guarantee on policies regularized by the stationary state distribution. In practice, SRPO can be an add-on module to context-based algorithms in both online and offline RL settings.Experimental results show that SRPO can make several context-based algorithms far more data efficient and significantly improve their overall performance.</details>

### [Spotlight] State Sequences Prediction via Fourier Transform for Representation Learning

**Authors:** Mingxuan Ye, Yufei Kuang, Jie Wang, Yang Rui, Wengang Zhou, Houqiang Li, Feng Wu

**<details><summary>Abstract**</summary>

While deep reinforcement learning (RL) has been demonstrated effective in solving complex control tasks, sample efficiency remains a key challenge due to the large amounts of data required for remarkable performance. Existing research explores the application of representation learning for data-efficient RL, e.g., learning predictive representations by predicting long-term future states. However, many existing methods do not fully exploit the structural information inherent in sequential state signals, which can potentially improve the quality of long-term decision-making but is difficult to discern in the time domain. To tackle this problem, we propose State Sequences Prediction via Fourier Transform (SPF), a novel method that exploits the frequency domain of state sequences to extract the underlying patterns in time series data for learning expressive representations efficiently. Specifically, we theoretically analyze the existence of structural information in state sequences, which is closely related to policy performance and signal regularity, and then propose to predict the Fourier transform of infinite-step future state sequences to extract such information. One of the appealing features of SPF is that it is simple to implement while not requiring storage of infinite-step future states as prediction targets. Experiments demonstrate that the proposed method outperforms several state-of-the-art algorithms in terms of both sample efficiency and performance.</details>

### Statistically Valid Variable Importance Assessment through Conditional Permutations

**Authors:** Ahmad CHAMMA, Denis Engemann, Bertrand Thirion

**<details><summary>Abstract**</summary>

Variable importance assessment has become a crucial step in machine-learning applications when using complex learners, such as deep neural networks, on large-scale data. Removal-based importance assessment is currently the reference approach, particularly when statistical guarantees are sought to justify variable inclusion. It is often implemented with variable permutation schemes. On the flip side, these approaches risk misidentifying unimportant variables as important in the presence of correlations among covariates. Here we develop a systematic approach for studying Conditional Permutation Importance (CPI) that is model agnostic and computationally lean,  as well as reusable benchmarks of state-of-the-art variable importance estimators. We show theoretically and empirically that \textit{CPI} overcomes the limitations of standard permutation importance by providing accurate type-I error control. When used with a deep neural network, \textit{CPI} consistently showed top accuracy across benchmarks. An experiment on real-world data analysis in a large-scale medical dataset showed that \textit{CPI} provides a more parsimonious selection of statistically significant variables. Our results suggest that \textit{CPI} can be readily used as drop-in replacement for permutation-based methods.</details>

### Stochastic Approximation Approaches to Group Distributionally Robust Optimization

**Authors:** Lijun Zhang, Peng Zhao, Zhen-Hua Zhuang, Tianbao Yang, Zhi-Hua Zhou

**<details><summary>Abstract**</summary>

This paper investigates group distributionally robust optimization (GDRO), with the purpose to learn a model that performs well over $m$ different distributions. First, we formulate GDRO as a stochastic convex-concave saddle-point problem, and demonstrate that stochastic mirror descent (SMD), using $m$ samples in each iteration, achieves an $O(m (\log m)/\epsilon^2)$ sample complexity for finding an $\epsilon$-optimal solution, which matches the $\Omega(m/\epsilon^2)$ lower bound up to a logarithmic factor. Then, we make use of techniques from online learning to reduce the number of samples required in each round from $m$ to $1$, keeping the same sample complexity. Specifically, we cast GDRO as a two-players game where one player simply performs SMD and the other executes an online algorithm for non-oblivious multi-armed bandits. Next, we consider a more practical scenario where the number of samples that can be drawn from each distribution is different, and propose a novel formulation of weighted GDRO, which allows us to derive distribution-dependent convergence rates.  Denote by $n_i$ the sample budget for the $i$-th distribution, and assume $n_1 \geq n_2 \geq \cdots \geq n_m$. In the first approach, we incorporate non-uniform sampling into SMD such that the sample budget is satisfied in expectation, and prove that the excess risk of the $i$-th distribution decreases at an $O(\sqrt{n_1 \log m}/n_i)$ rate. In the second approach, we use mini-batches to meet the budget exactly and also reduce the variance in stochastic gradients, and then leverage stochastic mirror-prox algorithm, which can exploit small variances, to optimize a carefully designed weighted GDRO problem. Under appropriate conditions, it attains an $O((\log m)/\sqrt{n_i})$ convergence rate, which almost matches the optimal $O(\sqrt{1/n_i})$ rate of only learning from the $i$-th distribution with $n_i$ samples.</details>

### Stochastic Optimal Control for Collective Variable Free Sampling of Molecular Transition Paths

**Authors:** Lars Holdijk, Yuanqi Du, Ferry Hooft, Priyank Jaini, Berend Ensing, Max Welling

**<details><summary>Abstract**</summary>

We consider the problem of sampling transition paths between two given metastable states of a molecular system, eg. a folded and unfolded protein or products and reactants of a chemical reaction. Due to the existence of high energy barriers separating the states, these transition paths are unlikely to be sampled with standard Molecular Dynamics (MD) simulation. Traditional methods to augment MD with a bias potential to increase the probability of the transition rely on a dimensionality reduction step based on Collective Variables (CVs). Unfortunately, selecting appropriate CVs requires chemical intuition and traditional methods are therefore not always applicable to larger systems. Additionally, when incorrect CVs are used, the bias potential might not be minimal and bias the system along dimensions irrelevant to the transition. Showing a formal relation between the problem of sampling molecular transition paths, the Schrodinger bridge problem and stochastic optimal control with neural network policies, we propose a machine learning method for sampling said transitions. Unlike previous non-machine learning approaches our method, named PIPS, does not depend on CVs. We show that our method successful generates low energy transitions for Alanine Dipeptide as well as the larger Polyproline and Chignolin proteins.</details>

### Strategic Distribution Shift of Interacting Agents via Coupled Gradient Flows

**Authors:** Lauren Conger, Franca Hoffmann, Eric Mazumdar, Lillian Ratliff

**<details><summary>Abstract**</summary>

We propose a novel framework for analyzing the dynamics of distribution shift in real-world systems that captures the feedback loop between learning algorithms and the distributions on which they are deployed. Prior work largely models feedback-induced distribution shift as adversarial or via an overly simplistic distribution-shift structure. In contrast, we propose a coupled partial differential equation model that captures fine-grained changes in the distribution over time by accounting for complex dynamics that arise due to strategic responses to algorithmic decision-making, non-local endogenous population interactions, and other exogenous sources of distribution shift. We consider two common settings in machine learning: cooperative settings with information asymmetries, and competitive settings where a learner faces strategic users. For both of these settings, when the algorithm retrains via gradient descent, we prove asymptotic convergence of the retraining procedure to a steady-state, both in finite and in infinite dimensions, obtaining explicit rates in terms of the model parameters. To do so we derive new results on the convergence of coupled PDEs that extends what is known on multi-species systems. Empirically, we show that our approach captures well-documented forms of distribution shifts like polarization and disparate impacts that simpler models cannot capture.</details>

### Streaming Factor Trajectory Learning for Temporal Tensor Decomposition

**Authors:** Shikai Fang, Xin Yu, Shibo Li, Zheng Wang, Mike Kirby, Shandian Zhe

**<details><summary>Abstract**</summary>

Practical tensor data is often along with time information. Most existing temporal decomposition approaches estimate a set of fixed factors for the objects in each tensor mode, and hence cannot capture the temporal evolution of the objects' representation. More important, we lack an effective approach to capture such evolution from streaming data, which is common in real-world applications.  To address these issues, we propose Streaming Factor Trajectory Learning (SFTL) for temporal tensor decomposition. We use Gaussian processes (GPs) to model the trajectory of  factors so as to flexibly estimate their temporal evolution. To address the computational challenges in handling streaming data, we convert the GPs into a state-space prior by constructing an equivalent stochastic differential equation (SDE).  We develop an efficient online filtering algorithm to estimate a decoupled running posterior of the involved factor states upon receiving new data. The decoupled estimation enables us to conduct standard Rauch-Tung-Striebel smoothing to compute the full posterior of all the  trajectories in parallel, without the need for revisiting any previous data. We have shown the advantage of SFTL in both synthetic tasks and real-world applications.</details>

### [Spotlight] Streaming PCA for Markovian Data

**Authors:** Syamantak Kumar, Purnamrita Sarkar

**<details><summary>Abstract**</summary>

Since its inception in 1982, Oja's algorithm has become an established method for streaming principle component analysis (PCA). We study the problem of streaming PCA, where the data-points are sampled from an irreducible, aperiodic, and reversible Markov chain starting in stationarity. Our goal is to estimate the top eigenvector of the unknown covariance matrix of the stationary distribution. This setting has implications in scenarios where data can solely be sampled from a Markov Chain Monte Carlo (MCMC) type algorithm, and the objective is to perform inference on parameters of the stationary distribution. Most convergence guarantees for Oja's algorithm in the literature assume that the data-points are sampled IID. For data streams with Markovian dependence, one typically downsamples the data to get a "nearly" independent data stream. In this paper, we obtain the first near-optimal rate for Oja's algorithm on the entire data, where we remove the logarithmic dependence on the sample size, $n$, resulting from throwing data away in downsampling strategies.</details>

### Structured Neural-PI Control with End-to-End Stability and Output Tracking Guarantees

**Authors:** Wenqi Cui, Yan Jiang, Baosen Zhang, Yuanyuan Shi

**<details><summary>Abstract**</summary>

We study the optimal control of multiple-input and multiple-output dynamical systems via the design of neural network-based controllers with  stability and output tracking guarantees. While neural network-based nonlinear controllers have shown superior performance in various applications, their lack of provable guarantees has restricted their adoption in high-stake real-world applications. This paper bridges the gap between neural network-based controllers and the need for stabilization guarantees. Using equilibrium-independent passivity, a property present in a wide range of physical systems, we propose neural Proportional-Integral (PI) controllers that have provable guarantees of stability and zero steady-state output tracking error. The key structure is the strict monotonicity on proportional and integral terms, which is parameterized as gradients of strictly convex neural networks (SCNN). We construct SCNN with tunable softplus-$\beta$ activations, which yields universal approximation capability and is also useful in incorporating communication constraints. In addition, the SCNNs serve as Lyapunov functions, giving us end-to-end performance guarantees.  Experiments on traffic and power networks demonstrate that the proposed approach improves both transient and steady-state performances, while unstructured neural networks lead to unstable behaviors.</details>

### StyleDrop: Text-to-Image Synthesis of Any Style

**Authors:** Kihyuk Sohn, Lu Jiang, Jarred Barber, Kimin Lee, Nataniel Ruiz, Dilip Krishnan, Huiwen Chang, Yuanzhen Li, Irfan Essa, Michael Rubinstein, Yuan Hao, Glenn Entis, Irina Blok, Daniel Castro Chin

**<details><summary>Abstract**</summary>

Pre-trained large text-to-image models synthesize impressive images with an appropriate use of text prompts. However, ambiguities inherent in natural language, and out-of-distribution effects make it hard to synthesize arbitrary image styles, leveraging a specific design pattern, texture or material. In this paper, we introduce *StyleDrop*, a method that enables the synthesis of images that faithfully follow a specific style using a text-to-image model. StyleDrop is extremely versatile and captures nuances and details of a user-provided style, such as color schemes, shading, design patterns, and local and global effects. StyleDrop works by efficiently learning a new style by fine-tuning very few trainable parameters (less than 1\% of total model parameters), and improving the quality via iterative training with either human or automated feedback. Better yet, StyleDrop is able to deliver impressive results even when the user supplies only a *single* image specifying the desired style. An extensive study shows that, for the task of style tuning text-to-image models, StyleDrop on Muse convincingly outperforms other methods, including DreamBooth and textual inversion on Imagen or Stable Diffusion. More results are available at our project website: [https://styledrop.github.io](https://styledrop.github.io).</details>

### StyleGAN knows Normal, Depth, Albedo, and More

**Authors:** Anand Bhattad, Daniel McKee, Derek Hoiem, David Forsyth

**<details><summary>Abstract**</summary>

Intrinsic images, in the original sense, are image-like maps of scene properties like depth, normal, albedo, or shading. This paper demonstrates that StyleGAN can easily be induced to produce intrinsic images.  The procedure is straightforward. We show that if StyleGAN produces $G({\bf w})$ from latent ${\bf w}$, then for each type of intrinsic image, there is a fixed offset ${\bf d}_c$ so that $G({\bf w}+{\bf d}_c)$ is that type of intrinsic image for $G({\bf w})$. Here ${\bf d}_c$ is {\em independent of ${\bf w}$}.  The StyleGAN we used was pretrained by others, so this property is not some accident of our training regime.  We show that there are image transformations StyleGAN will {\em not} produce in this fashion, so StyleGAN is not a generic image regression engine.  It is conceptually exciting that an image generator should ``know'' and represent intrinsic images. There may also be practical advantages to using a generative model to produce intrinsic images. The intrinsic images obtained from StyleGAN compare well both qualitatively and quantitatively with those obtained by using SOTA image regression techniques; but StyleGAN's intrinsic images are robust to relighting effects, unlike SOTA methods.</details>

### Sub-optimality of the Naive Mean Field approximation for proportional high-dimensional Linear Regression

**Authors:** Jiaze Qiu

**<details><summary>Abstract**</summary>

The Naïve Mean Field (NMF) approximation is widely employed in modern Machine Learning due to the huge computational gains it bestows on the statistician. Despite its popularity in practice, theoretical guarantees for high-dimensional problems are only available under strong structural assumptions (e.g. sparsity). Moreover, existing theory often does not explain empirical observations noted in the existing literature.  In this paper, we take a step towards addressing these problems by deriving sharp asymptotic characterizations for the NMF approximation in high-dimensional linear regression. Our results apply to a wide class of natural priors and allow for model mismatch (i.e. the underlying statistical model can be different from the fitted model).  We work under an iid Gaussian design and the proportional asymptotic regime, where the number of features and number of observations grow at a proportional rate. As a consequence of our asymptotic characterization, we establish two concrete corollaries: (a) we establish the inaccuracy of the NMF approximation for the log-normalizing constant in this regime, and (b) we provide theoretical results backing the empirical observation that the NMF approximation can be overconfident in terms of uncertainty quantification.Our results utilize recent advances in the theory of Gaussian comparison inequalities. To the best of our knowledge, this is the first application of these ideas to the analysis of Bayesian variational inference problems. Our theoretical results are corroborated by numerical experiments. Lastly, we believe our results can be generalized to non-Gaussian designs and provide empirical evidence to support it.</details>

### Switching Temporary Teachers for Semi-Supervised Semantic Segmentation

**Authors:** Jaemin Na, Jung-Woo Ha, Hyung Jin Chang, Joon Chung, Wonjun Hwang

**<details><summary>Abstract**</summary>

The teacher-student framework, prevalent in semi-supervised semantic segmentation, mainly employs the exponential moving average (EMA) to update a single teacher's weights based on the student's. However, EMA updates raise a problem in that the weights of the teacher and student are getting coupled, causing a potential performance bottleneck. Furthermore, this problem may become more severe when training with more complicated labels such as segmentation masks but with few annotated data. This paper introduces Dual Teacher, a simple yet effective approach that employs dual temporary teachers aiming to alleviate the coupling problem for the student. The temporary teachers work in shifts and are progressively improved, so consistently prevent the teacher and student from becoming excessively close. Specifically, the temporary teachers periodically take turns generating pseudo-labels to train a student model and maintain the distinct characteristics of the student model for each epoch. Consequently, Dual Teacher achieves competitive performance on the PASCAL VOC, Cityscapes, and ADE20K benchmarks with remarkably shorter training times than state-of-the-art methods. Moreover, we demonstrate that our approach is model-agnostic and compatible with both CNN- and Transformer-based models. Code is available at https://github.com/naver-ai/dual-teacher.</details>

### SyncDiffusion: Coherent Montage via Synchronized Joint Diffusions

**Authors:** Yuseung Lee, Kunho Kim, Hyunjin Kim, Minhyuk Sung

**<details><summary>Abstract**</summary>

The remarkable capabilities of pretrained image diffusion models have been utilized not only for generating fixed-size images but also for creating panoramas. However, naive stitching of multiple images often results in visible seams. Recent techniques have attempted to address this issue by performing joint diffusions in multiple windows and averaging latent features in overlapping regions. However, these approaches, which focus on seamless montage generation, often yield incoherent outputs by blending different scenes within a single image. To overcome this limitation, we propose SyncDiffusion, a plug-and-play module that synchronizes multiple diffusions through gradient descent from a perceptual similarity loss. Specifically, we compute the gradient of the perceptual loss using the predicted denoised images at each denoising step, providing meaningful guidance for achieving coherent montages. Our experimental results demonstrate that our method produces significantly more coherent outputs compared to previous methods (66.35% vs. 33.65% in our user study) while still maintaining fidelity (as assessed by GIQA) and compatibility with the input prompt (as measured by CLIP score). We further demonstrate the versatility our method across three plug-and-play applications: layout-guided image generation, conditional image generation and 360-degree panorama generation. Our project page is at https://syncdiffusion.github.io.</details>

### Synthetic-to-Real Pose Estimation with Geometric Reconstruction

**Authors:** Qiuxia Lin, Kerui Gu, Linlin Yang, Angela Yao

**<details><summary>Abstract**</summary>

Pose estimation is remarkably successful under supervised learning, but obtaining annotations, especially for new deployments, is costly and time-consuming. This work tackles adapting models trained on synthetic data to real-world target domains with only unlabelled data. A common approach is model fine-tuning with pseudo-labels from the target domain; yet many pseudo-labelling strategies cannot provide sufficient high-quality pose labels. This work proposes a reconstruction-based strategy as a complement to pseudo-labelling for synthetic-to-real domain adaptation. We generate the driving image by geometrically transforming a base image according to the predicted keypoints and enforce a reconstruction loss to refine the predictions. It provides a novel solution to effectively correct confident yet inaccurate keypoint locations through image reconstruction in domain adaptation. Our approach outperforms the previous state-of-the-arts by 8% for PCK on four large-scale hand and human real-world datasets. In particular, we excel on endpoints such as fingertips and head, with 7.2% and 29.9% improvements in PCK.</details>

### Tanimoto Random Features for Scalable Molecular Machine Learning

**Authors:** Austin Tripp, Sergio Bacallado, Sukriti Singh, José Miguel Hernández-Lobato

**<details><summary>Abstract**</summary>

The Tanimoto coefficient is commonly used to measure the similarity between molecules represented as discrete fingerprints,either as a distance metric or a positive definite kernel. While many kernel methods can be accelerated using random feature approximations, at present there is a lack of such approximations for the Tanimoto kernel. In this paper we propose two kinds of novel random features to allow this kernel to scale to large datasets, and in the process discover a novel extension of the kernel to real-valued vectors. We theoretically characterize these random features, and provide error bounds on the spectral norm of the Gram matrix. Experimentally, we show that these random features are effective at approximating the Tanimoto coefficient of real-world datasetsand are useful for molecular property prediction and optimization tasks. Future updates to this work will be available at http://arxiv.org/abs/2306.14809.</details>

### Task-Robust Pre-Training for Worst-Case Downstream Adaptation

**Authors:** Jianghui Wang, Yang Chen, Xingyu Xie, Cong Fang, Zhouchen Lin

**<details><summary>Abstract**</summary>

Pre-training has achieved remarkable success when transferred to downstream tasks. In machine learning, we care about not only the good performance of a model but also its behavior under reasonable shifts of condition. The same philosophy holds when pre-training a foundation model. However,  the foundation model may not uniformly behave well for a series of related downstream tasks. This happens, for example, when conducting mask recovery regression where the recovery ability or the training instances diverge like pattern features are extracted dominantly on pre-training, but semantic features are also required on a downstream task. This paper considers pre-training a model that guarantees a uniformly good performance over the downstream tasks. We call this goal as *downstream-task robustness*.Our method first separates the upstream task into several representative ones and applies a simple minimax loss for pre-training. We then design an efficient algorithm to solve the minimax lossand prove its convergence in the convex setting. In the experiments, we show both on large-scale natural language processing and computer vision datasets our method increases the metrics on worse-case downstream tasks. Additionally, some theoretical explanations for why our loss is beneficial are provided. Specifically, we show fewer samples are inherently required for the most challenging downstream task in some cases.</details>

### Task-aware world model learning with meta weighting via bi-level optimization

**Authors:** Huining Yuan, Hongkun Dou, Xingyu Jiang, Yue Deng

**<details><summary>Abstract**</summary>

Aligning the world model with the environment for the agent’s specific task is crucial in model-based reinforcement learning. While value-equivalent models may achieve better task awareness than maximum-likelihood models, they sacrifice a large amount of semantic information and face implementation issues. To combine the benefits of both types of models, we propose Task-aware Environment Modeling Pipeline with bi-level Optimization (TEMPO), a bi-level model learning framework that introduces an additional level of optimization on top of a maximum-likelihood model by incorporating a meta weighter network that weights each training sample. The meta weighter in the upper level learns to generate novel sample weights by minimizing a proposed task-aware model loss. The model in the lower level focuses on important samples while maintaining rich semantic information in state representations. We evaluate TEMPO on a variety of continuous and discrete control tasks from the DeepMind Control Suite and Atari video games. Our results demonstrate that TEMPO achieves state-of-the-art performance regarding asymptotic performance, training stability, and convergence speed.</details>

### TaskMet: Task-driven Metric Learning for Model Learning

**Authors:** Dishank Bansal, Ricky T. Q. Chen, Mustafa Mukadam, Brandon Amos

**<details><summary>Abstract**</summary>

Deep learning models are often used with some downstream task. Models solely trained to achieve accurate predictions may struggle to perform well on the desired downstream tasks. We propose using the task loss to learn a metric which parameterizes a loss to train the model. This approach does not alter the optimal prediction model itself, but rather changes the model learning to emphasize the information important for the downstream task. This enables us to achieve the best of both worlds: a prediction model trained in the original prediction space while also being valuable for the desired downstream task. We validate our approach through experiments conducted in two main settings: 1) decision-focused model learning scenarios involving portfolio optimization and budget allocation, and 2) reinforcement learning in noisy environments with distracting states.</details>

### Template-free Articulated Neural Point Clouds for Reposable View Synthesis

**Authors:** Lukas Uzolas, Elmar Eisemann, Petr Kellnhofer

**<details><summary>Abstract**</summary>

Dynamic Neural Radiance Fields (NeRFs) achieve remarkable visual quality when synthesizing novel views of time-evolving 3D scenes. However, the common reliance on backward deformation fields makes reanimation of the captured object poses challenging. Moreover, the state of the art dynamic models are often limited by low visual fidelity, long reconstruction time or specificity to narrow application domains.  In this paper, we present a novel method utilizing a point-based representation and Linear Blend Skinning (LBS) to jointly learn a Dynamic NeRF and an associated skeletal model from even sparse multi-view video. Our forward-warping approach achieves state-of-the-art visual fidelity when synthesizing novel views and poses while significantly reducing the necessary learning time when compared to existing work. We demonstrate the versatility of our representation on a variety of articulated objects from common datasets and obtain reposable 3D reconstructions without the need of object-specific skeletal templates.</details>

### Temporal Robustness against Data poisoning

**Authors:** Wenxiao Wang, Soheil Feizi

**<details><summary>Abstract**</summary>

Data poisoning considers cases when an adversary manipulates the behavior of machine learning algorithms through malicious training data. Existing threat models of data poisoning center around a single metric, the number of poisoned samples. In consequence, if attackers can poison more samples than expected with affordable overhead, as in many practical scenarios, they may be able to render existing defenses ineffective in a short time. To address this issue, we leverage timestamps denoting the birth dates of data, which are often available but neglected in the past. Benefiting from these timestamps, we propose a temporal threat model of data poisoning with two novel metrics, earliness and duration, which respectively measure how long an attack started in advance and how long an attack lasted. Using these metrics, we define the notions of temporal robustness against data poisoning, providing a meaningful sense of protection even with unbounded amounts of poisoned samples when the attacks are temporally bounded. We present a benchmark with an evaluation protocol simulating continuous data collection and periodic deployments of updated models, thus enabling empirical evaluation of temporal robustness. Lastly, we develop and also empirically verify a baseline defense, namely temporal aggregation, offering provable temporal robustness and highlighting the potential of our temporal threat model for data poisoning.</details>

### Testing the General Deductive Reasoning Capacity of Large Language Models Using OOD Examples

**Authors:** Abulhair Saparov, Richard Yuanzhe Pang, Vishakh Padmakumar, Nitish Joshi, Mehran Kazemi, Najoung Kim, He He

**<details><summary>Abstract**</summary>

Given the intractably large size of the space of proofs, any model that is capable of general deductive reasoning must generalize to proofs of greater complexity. Recent studies have shown that large language models (LLMs) possess some abstract deductive reasoning ability given chain-of-thought prompts. However, they have primarily been tested on proofs using modus ponens or of a specific size, and from the same distribution as the in-context examples. To measure the general deductive reasoning ability of LLMs, we test on a broad set of deduction rules and measure their ability to generalize to more complex proofs from simpler demonstrations from multiple angles: depth-, width-, and compositional generalization. To facilitate systematic exploration, we construct a new synthetic and programmable reasoning dataset that enables control over deduction rules and proof complexity. Our experiments on four LLMs of various sizes and training objectives show that they are able to generalize to compositional proofs. However, they have difficulty generalizing to longer proofs, and they require explicit demonstrations to produce hypothetical subproofs, specifically in proof by cases and proof by contradiction.</details>

### The Gain from Ordering in Online Learning

**Authors:** Vasilis Kontonis, Mingchen Ma, Christos Tzamos

**<details><summary>Abstract**</summary>

We study fixed-design online learning where the learner is allowed to choose the order of the datapoints in order to minimize their regret (aka self-directed online learning).  We focus on the fundamental task of online linear regression: the learner is given a dataset $X$ with $n$ examples in $d$ dimensions and at step $t$  they select a point $x_t \in X$, predict a value $\widetilde y_t$,  and suffer loss $(\widetilde y_t - w^\ast \cdot x_t)^2$.  The goal is to design algorithms that order the examples and achieve better regret  than random- or worst-order online algorithms.For an arbitrary dataset $X$, we show that, under the Exponential Time Hypothesis, no efficient algorithm can approximate the optimal (best-order)  regret  within a factor of $d^{1/\poly(\log \log d)}$.We then show that, for structured datasets, we can bypass the above hardness result and achieve nearly optimal regret. When the examples of $X$ are drawn i.i.d.\ from the uniform distribution on the sphere, we present an algorithm based on the greedy heuristic of  selecting ``easiest'' examples first that achieves a $\log d$-approximation of the optimal regret.</details>

### The Impact of Positional Encoding on Length Generalization in Transformers

**Authors:** Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, Siva Reddy

**<details><summary>Abstract**</summary>

Length generalization, the ability to generalize from small training context sizes to larger ones, is a critical challenge in the development of Transformer-based language models. Positional encoding (PE) has been identified as a major factor influencing length generalization, but the exact impact of different PE schemes on extrapolation in downstream tasks remains unclear. In this paper, we conduct a systematic empirical study comparing the length generalization performance of decoder-only Transformers with five different position encoding approaches including Absolute Position Embedding (APE), T5's Relative PE, ALiBi, and Rotary, in addition to Transformers without positional encoding (NoPE). Our evaluation encompasses a battery of reasoning and mathematical tasks. Our findings reveal that the most commonly used positional encoding methods, such as ALiBi, Rotary, and APE, are not well suited for length generalization in downstream tasks. More importantly, NoPE outperforms other explicit positional encoding methods while requiring no additional computation. We theoretically demonstrate that NoPE can represent both absolute and relative PEs, but when trained with SGD, it mostly resembles T5's relative PE attention patterns. Finally, we find that scratchpad is not always helpful to solve length generalization and its format highly impacts the model's performance. Overall, our work suggests that explicit position embeddings are not essential for decoder-only Transformers to generalize well to longer sequences.</details>

### The Shaped Transformer: Attention Models in the Infinite Depth-and-Width Limit

**Authors:** Lorenzo Noci, Chuning Li, Mufan Li, Bobby He, Thomas Hofmann, Chris Maddison, Dan Roy

**<details><summary>Abstract**</summary>

In deep learning theory, the covariance matrix of the representations serves as aproxy to examine the network’s trainability. Motivated by the success of Transform-ers, we study the covariance matrix of a modified Softmax-based attention modelwith skip connections in the proportional limit of infinite-depth-and-width. Weshow that at initialization the limiting distribution can be described by a stochasticdifferential equation (SDE) indexed by the depth-to-width ratio. To achieve awell-defined stochastic limit, the Transformer’s attention mechanism is modifiedby centering the Softmax output at identity, and scaling the Softmax logits by awidth-dependent temperature parameter. We examine the stability of the networkthrough the corresponding SDE, showing how the scale of both the drift and diffu-sion can be elegantly controlled with the aid of residual connections. The existenceof a stable SDE implies that the covariance structure is well-behaved, even for verylarge depth and width, thus preventing the notorious issues of rank degeneracyin deep attention models. Finally, we show, through simulations, that the SDEprovides a surprisingly good description of the corresponding finite-size model.We coin the name shaped Transformer for these architectural modifications.</details>

### The Transient Nature of Emergent In-Context Learning in Transformers

**Authors:** Aaditya Singh, Stephanie Chan, Ted Moskovitz, Erin Grant, Andrew Saxe, Felix Hill

**<details><summary>Abstract**</summary>

Transformer neural networks can exhibit a surprising capacity for in-context learning (ICL), despite not being explicitly trained for it.  Prior work has provided a deeper understanding of how ICL emerges in transformers, e.g., through the lens of mechanistic interpretability, Bayesian inference, or by examining the distributional properties of training data. However, in each of these cases, ICL is treated largely as a persistent phenomenon; namely, once ICL emerges, it is assumed to persist asymptotically. Here, we show that the emergence of ICL during transformer training is, in fact, often transient. We train transformers on synthetic data designed so that both ICL or in-weights learning (IWL) strategies can lead to correct predictions. We find that ICL first emerges, then disappears and gives way to IWL,all while the training loss decreases, indicating an asymptotic preference for IWL. The transient nature of ICL is observed in transformers across a range of model sizes and datasets, raising the question of how much to “overtrain” transformers when seeking compact, cheaper-to-run models. We find that L2 regularization may offer a path to more persistent ICL that removes the need for early stopping based on ICL-style validation tasks.</details>

### Three Towers: Flexible Contrastive Learning with Pretrained Image Models

**Authors:** Jannik Kossen, Mark Collier, Basil Mustafa, Xiao Wang, Xiaohua Zhai, Lucas Beyer, Andreas Steiner, Jesse Berent, Rodolphe Jenatton, Effrosyni Kokiopoulou

**<details><summary>Abstract**</summary>

We introduce Three Towers (3T), a flexible method to improve the contrastive learning of vision-language models by incorporating pretrained image classifiers. While contrastive models are usually trained from scratch, LiT (Zhai et al., 2022) has recently shown performance gains from using pretrained classifier embeddings. However, LiT directly replaces the image tower with the frozen embeddings, excluding any potential benefits from training the image tower contrastively. With 3T, we propose a more flexible strategy that allows the image tower to benefit from both pretrained embeddings and contrastive training. To achieve this, we introduce a third tower that contains the frozen pretrained embeddings, and we encourage alignment between this third tower and the main image-text towers. Empirically, 3T consistently improves over LiT and the CLIP-style from-scratch baseline for retrieval tasks. For classification, 3T reliably improves over the from-scratch baseline, and while it underperforms relative to LiT for JFT-pretrained models, it outperforms LiT for ImageNet-21k and Places365 pretraining.</details>

### Time-Independent Information-Theoretic Generalization Bounds for SGLD

**Authors:** Futoshi Futami, Masahiro Fujisawa

**<details><summary>Abstract**</summary>

We provide novel information-theoretic generalization bounds for stochastic gradient Langevin dynamics (SGLD) under the assumptions of smoothness and dissipativity, which are widely used in sampling and non-convex optimization studies.Our bounds are time-independent and decay to zero as the sample size increases, regardless of the number of iterations and whether the step size is fixed.Unlike previous studies, we derive the generalization error bounds by focusing on the time evolution of the Kullback--Leibler divergence, which is related to the stability of datasets and is the upper bound of the mutual information between output parameters and an input dataset.Additionally, we establish the first information-theoretic generalization bound when the training and test loss are the same by showing that a loss function of SGLD is sub-exponential.This bound is also time-independent and removes the problematic step size dependence in existing work, leading to an improved excess risk bound by combining our analysis with the existing non-convex optimization error bounds.</details>

### [Spotlight] Timewarp: Transferable Acceleration of Molecular Dynamics by Learning Time-Coarsened Dynamics

**Authors:** Leon Klein, Andrew Foong, Tor Fjelde, Bruno Mlodozeniec, Marc Brockschmidt, Sebastian Nowozin, Frank Noe, Ryota Tomioka

**<details><summary>Abstract**</summary>

*Molecular dynamics* (MD) simulation is a widely used technique to simulate molecular systems, most commonly at the all-atom resolution where equations of motion are integrated with timesteps on the order of femtoseconds ($1\textrm{fs}=10^{-15}\textrm{s}$). MD is often used to compute equilibrium properties, which requires sampling from an equilibrium distribution such as the Boltzmann distribution. However, many important processes, such as binding and folding, occur over timescales of milliseconds or beyond, and cannot be efficiently sampled with conventional MD.Furthermore, new MD simulations need to be performed for each molecular system studied.We present *Timewarp*, an enhanced sampling method which uses a normalising flow as a proposal distribution in a Markov chain Monte Carlo method targeting the Boltzmann distribution. The flow is trained offline on MD trajectories and learns to make large steps in time, simulating the molecular dynamics of $10^{5} - 10^{6} \textrm{fs}$.Crucially, Timewarp is *transferable* between molecular systems: once trained, we show that it generalises to unseen small peptides (2-4 amino acids) at all-atom resolution, exploring their metastable states and providing wall-clock acceleration of sampling compared to standard MD.Our method constitutes an important step towards general, transferable algorithms for accelerating MD.</details>

### Token-Scaled Logit Distillation for Ternary Weight Generative Language Models

**Authors:** Minsoo Kim, Sihwa Lee, Janghwan Lee, Sukjin Hong, Du-Seong Chang, Wonyong Sung, Jungwook Choi

**<details><summary>Abstract**</summary>

Generative Language Models (GLMs) have shown impressive performance in tasks such as text generation, understanding, and reasoning. However, the large model size poses challenges for practical deployment. To solve this problem, Quantization-Aware Training (QAT) has become increasingly popular. However, current QAT methods for generative models have resulted in a noticeable loss of accuracy. To counteract this issue, we propose a novel knowledge distillation method specifically designed for GLMs. Our method, called token-scaled logit distillation, prevents overfitting and provides superior learning from the teacher model and ground truth. This research marks the first evaluation of ternary weight quantization-aware training of large-scale GLMs with less than 1.0 degradation in perplexity and achieves enhanced accuracy in tasks like common-sense QA and arithmetic reasoning as well as natural language understanding. Our code is available at https://github.com/aiha-lab/TSLD.</details>

### [Oral] Toolformer: Language Models Can Teach Themselves to Use Tools

**Authors:** Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom

**Oral Presentation:** We, Dec 13, 08:15 -- Oral 3B

**<details><summary>Abstract**</summary>

Language models (LMs) exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller specialized models excel. In this paper, we show that LMs can teach themselves to *use external tools* via simple APIs and achieve the best of both worlds. We introduce *Toolformer*, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. This is done in a self-supervised way, requiring nothing more than a handful of demonstrations for each API. We incorporate a range of tools, including a calculator, a Q&A system, a search engine, a translation system, and a calendar. Toolformer achieves substantially improved zero-shot performance across a variety of downstream tasks, often competitive with much larger models, without sacrificing its core language modeling abilities.</details>

### [Oral] ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings

**Authors:** Shibo Hao, Tianyang Liu, Zhen Wang, Zhiting Hu

**Oral Presentation:** We, Dec 13, 08:00 -- Oral 3B

**<details><summary>Abstract**</summary>

Integrating large language models (LLMs) with various tools has led to increased attention in the field. Existing approaches either involve fine-tuning the LLM, which is both computationally costly and limited to a fixed set of tools, or prompting LLMs by in-context tool demonstrations. Although the latter method offers adaptability to new tools, it struggles with the inherent context length constraint of LLMs when many new tools are presented, and mastering a new set of tools with few-shot examples remains challenging, resulting in suboptimal performance. To address these limitations, we propose a novel solution, named **ToolkenGPT**, wherein LLMs effectively learn to master tools as predicting tokens through **tool embeddings** for solving complex tasks. In this framework, each tool is transformed into vector embeddings and plugged into the language model head. Once the function is triggered during text generation, the LLM enters a special function mode to execute the tool calls. Our experiments show that function embeddings effectively help LLMs understand tool use and improve on several tasks, including numerical reasoning, knowledge-based question answering and embodied decision-making.</details>

### Toward Better PAC-Bayes Bounds for Uniformly Stable Algorithms

**Authors:** Sijia Zhou, Yunwen Lei, Ata Kaban

**<details><summary>Abstract**</summary>

We give sharper bounds for uniformly stable randomized algorithms in a PAC-Bayesian framework, which improve the existing results by up to a factor of $\sqrt{n}$ (ignoring a log factor), where $n$ is the sample size. The key idea is to bound the moment generating function of the generalization gap using concentration of weakly dependent random variables due to Bousquet et al (2020). We introduce an assumption of sub-exponential stability parameter, which allows a general treatment that we instantiate in two applications: stochastic gradient descent and randomized coordinate descent. Our results eliminate the requirement of strong convexity from previous results, and hold for non-smooth convex problems.</details>

### Towards Characterizing the First-order Query Complexity of Learning (Approximate) Nash Equilibria in Zero-sum Matrix Games

**Authors:** Hedi Hadiji, Sarah Sachs, Tim van Erven, Wouter Koolen

**<details><summary>Abstract**</summary>

In the first-order query model for zero-sum $K\times K$ matrix games, players observe the expected pay-offs for all their possible actions under the randomized action played by their opponent. This classical model has received renewed interest after the discovery by Rakhlin and Sridharan that $\epsilon$-approximate Nash equilibria can be computed efficiently from $O(\frac{\ln K}{\epsilon})$ instead of $O(\frac{\ln K}{\epsilon^2})$ queries. Surprisingly, the optimal number of such queries, as a function of both $\epsilon$ and $K$, is not known. We make progress on this question on two fronts. First, we fully characterise the query complexity of learning exact equilibria ($\epsilon=0$), by showing that they require a number of queries that is linear in $K$, which means that it is essentially as hard as querying the whole matrix, which can also be done with $K$ queries. Second, for $\epsilon > 0$, the current query complexity upper bound stands at $O(\min(\frac{\ln(K)}{\epsilon} , K))$. We argue that, unfortunately, obtaining a matching lower bound is not possible with existing techniques: we prove that no lower bound can be derived by constructing hard matrices whose entries take values in a known countable set, because such matrices can be fully identified by a single query. This rules out, for instance, reducing to an optimization problem over the hypercube by encoding it as a binary payoff matrix. We then introduce a new technique for lower bounds, which allows us to obtain lower bounds of order $\tilde\Omega(\log(\frac{1}{K\epsilon})$ for any $\epsilon \leq 1 / (cK^4)$, where $c$ is a constant independent of $K$. We further discuss possible future directions to improve on our techniques in order to close the gap with the upper bounds.</details>

### Towards Free Data Selection with General-Purpose Models

**Authors:** Yichen Xie, Mingyu Ding, Masayoshi TOMIZUKA, Wei Zhan

**<details><summary>Abstract**</summary>

A desirable data selection algorithm can efficiently choose the most informative samples to maximize the utility of limited annotation budgets. However, current approaches, represented by active learning methods, typically follow a cumbersome pipeline that iterates the time-consuming model training and batch data selection repeatedly. In this paper, we challenge this status quo by designing a distinct data selection pipeline that utilizes existing general-purpose models to select data from various datasets with a single-pass inference without the need for additional training or supervision. A novel free data selection (FreeSel) method is proposed following this new pipeline. Specifically, we define semantic patterns extracted from inter-mediate features of the general-purpose model to capture subtle local information in each image. We then enable the selection of all data samples in a single pass through distance-based sampling at the fine-grained semantic pattern level. FreeSel bypasses the heavy batch selection process, achieving a significant improvement in efficiency and being 530x faster than existing active learning methods. Extensive experiments verify the effectiveness of FreeSel on various computer vision tasks.</details>

### Towards Generic Semi-Supervised Framework for Volumetric Medical Image Segmentation

**Authors:** Haonan Wang, Xiaomeng Li

**<details><summary>Abstract**</summary>

Volume-wise labeling in 3D medical images is a time-consuming task that requires expertise. As a result, there is growing interest in using semi-supervised learning (SSL) techniques to train models with limited labeled data. However, the challenges and practical applications extend beyond SSL to settings such as unsupervised domain adaptation (UDA) and semi-supervised domain generalization (SemiDG). This work aims to develop a generic SSL framework that can handle all three settings. We identify two main obstacles to achieving this goal in the existing SSL framework: 1) the weakness of capturing distribution-invariant features; and 2) the tendency for unlabeled data to be overwhelmed by labeled data, leading to over-fitting to the labeled data during training. To address these issues, we propose an Aggregating & Decoupling framework. The aggregating part consists of a Diffusion encoder that constructs a "common knowledge set" by extracting distribution-invariant features from aggregated information from multiple distributions/domains. The decoupling part consists of three decoders that decouple the training process with labeled and unlabeled data, thus avoiding over-fitting to labeled data, specific domains and classes. We evaluate our proposed framework on four benchmark datasets for SSL, Class-imbalanced SSL, UDA and SemiDG. The results showcase notable improvements compared to state-of-the-art methods across all four settings, indicating the potential of our framework to tackle more challenging SSL scenarios. Code and models are available at: https://github.com/xmed-lab/GenericSSL.</details>

### Towards Label Position Bias in Graph Neural Networks

**Authors:** Haoyu Han, Xiaorui Liu, Feng Shi, MohamadAli Torkamani, Charu Aggarwal, Jiliang Tang

**<details><summary>Abstract**</summary>

Graph Neural Networks (GNNs) have emerged as a powerful tool for semi-supervised node classification tasks. However, recent studies have revealed various biases in GNNs stemming from both node features and graph topology. In this work, we uncover a new bias - label position bias, which indicates that the node closer to the labeled nodes tends to perform better. We introduce a new metric, the Label Proximity Score, to quantify this bias, and find that it is closely related to performance disparities. To address the label position bias, we propose a novel optimization framework for learning a label position unbiased graph structure, which can be applied to existing GNNs. Extensive experiments demonstrate that our proposed method not only outperforms backbone methods but also significantly mitigates the issue of label position bias in GNNs.</details>

### Towards Optimal Effective Resistance Estimation

**Authors:** Rajat Vadiraj Dwaraknath, Ishani Karmarkar, Aaron Sidford

**<details><summary>Abstract**</summary>

We provide new algorithms and conditional hardness for the problem of estimating effective resistances in $n$-node $m$-edge undirected, expander graphs. We provide an $\widetilde{O}(m\epsilon^{-1})$-time algorithm that produces with high probability, an $\widetilde{O}(n\epsilon^{-1})$-bit sketch from which the effective resistance between any pair of nodes can be estimated, to $(1 \pm \epsilon)$-multiplicative accuracy, in $\widetilde{O}(1)$-time. Consequently, we obtain an $\widetilde{O}(m\epsilon^{-1})$-time algorithm for estimating the effective resistance of all edges in such graphs, improving (for sparse graphs) on the previous fastest runtimes of $\widetilde{O}(m\epsilon^{-3/2})$ [Chu et. al. 2018] and $\widetilde{O}(n^2\epsilon^{-1})$ [Jambulapati, Sidford, 2018] for general graphs and $\widetilde{O}(m + n\epsilon^{-2})$ for expanders [Li, Sachdeva 2022]. We complement this result by showing a conditional lower bound that a broad set of algorithms for computing such estimates of the effective resistances between all pairs of nodes require $\widetilde{\Omega}(n^2 \epsilon^{-1/2})$-time, improving upon the previous best such lower bound of $\widetilde{\Omega}(n^2 \epsilon^{-1/13})$ [Musco et. al. 2017]. Further, we leverage the tools underlying these results to obtain improved algorithms and conditional hardness for more general problems of sketching the pseudoinverse of positive semidefinite matrices and estimating functions of their eigenvalues.</details>

### [Spotlight] Towards Symmetry-Aware Generation of Periodic Materials

**Authors:** Youzhi Luo, Chengkai Liu, Shuiwang Ji

**<details><summary>Abstract**</summary>

We consider the problem of generating periodic materials with deep models. While symmetry-aware molecule generation has been studied extensively, periodic materials possess different symmetries, which have not been completely captured by existing methods.In this work, we propose SyMat, a novel material generation approach that can capture physical symmetries of periodic material structures. SyMat generates atom types and lattices of materials through generating atom type sets, lattice lengths and lattice angles with a variational auto-encoder model. In addition, SyMat employs a score-based diffusion model to generate atom coordinates of materials, in which a novel symmetry-aware probabilistic model is used in the coordinate diffusion process. We show that SyMat is theoretically invariant to all symmetry transformations on materials and demonstrate that SyMat achieves promising performance on random generation and property optimization tasks. Our code is publicly available as part of the AIRS library (https://github.com/divelab/AIRS).</details>

### Towards a Unified Framework of Contrastive Learning for Disentangled Representations

**Authors:** Stefan Matthes, Zhiwei Han, Hao Shen

**<details><summary>Abstract**</summary>

Contrastive learning has recently emerged as a promising approach for learning data representations that discover and disentangle the explanatory factors of the data.Previous analyses of such approaches have largely focused on individual contrastive losses, such as noise-contrastive estimation (NCE) and InfoNCE, and rely on specific assumptions about the data generating process.This paper extends the theoretical guarantees for disentanglement to a broader family of contrastive methods, while also relaxing the assumptions about the data distribution.Specifically, we prove identifiability of the true latents for four contrastive losses studied in this paper, without imposing common independence assumptions.The theoretical findings are validated on several benchmark datasets.Finally, practical limitations of these methods are also investigated.</details>

### Trade-off Between Efficiency and Consistency for Removal-based Explanations

**Authors:** Yifan Zhang, Haowei He, Zhiquan Tan, Yang Yuan

**<details><summary>Abstract**</summary>

In the current landscape of explanation methodologies, most predominant approaches, such as SHAP and LIME, employ removal-based techniques to evaluate the impact of individual features by simulating various scenarios with specific features omitted. Nonetheless, these methods primarily emphasize efficiency in the original context, often resulting in general inconsistencies. In this paper, we demonstrate that such inconsistency is an inherent aspect of these approaches by establishing the Impossible Trinity Theorem, which posits that interpretability, efficiency, and consistency cannot hold simultaneously. Recognizing that the attainment of an ideal explanation remains elusive, we propose the utilization of interpretation error as a metric to gauge inefficiencies and inconsistencies. To this end, we present two novel algorithms founded on the standard polynomial basis, aimed at minimizing interpretation error. Our empirical findings indicate that the proposed methods achieve a substantial reduction in interpretation error, up to 31.8 times lower when compared to alternative techniques.</details>

### Trading-off price for data quality to achieve fair online allocation

**Authors:** Mathieu Molina, Nicolas Gast, Patrick Loiseau, Vianney Perchet

**<details><summary>Abstract**</summary>

We consider the problem of online allocation subject to a long-term fairness penalty. Contrary to existing works, however, we do not assume that the decision-maker observes the protected attributes---which is often unrealistic in practice. Instead they can purchase data that help estimate them from sources of different quality; and hence reduce the fairness penalty at some cost. We model this problem as a multi-armed bandit problem where each arm corresponds to the choice of a data source, coupled with the fair online allocation problem. We propose an algorithm that jointly solves both problems and show that it has a regret bounded by $\mathcal{O}(\sqrt{T})$. A key difficulty is that the rewards received by selecting a source are correlated by the fairness penalty, which leads to a need for randomization (despite a stochastic setting). Our algorithm takes into account contextual information available before the source selection, and can adapt to many different fairness notions.</details>

### Training Fully Connected Neural Networks is $\exists\mathbb{R}$-Complete

**Authors:** Daniel Bertschinger, Christoph Hertrich, Paul Jungeblut, Tillmann Miltzow, Simon Weber

**<details><summary>Abstract**</summary>

We consider the algorithmic problem of finding the optimal weights and biases for a two-layer fully connected neural network to fit a given set of data points, also known as empirical risk minimization. We show that the problem is $\exists\mathbb{R}$-complete. This complexity class can be defined as the set of algorithmic problems that are polynomial-time equivalent to finding real roots of a multivariate polynomial with integer coefficients. Furthermore, we show that arbitrary algebraic numbers are required as weights to be able to train some instances to optimality, even if all data points are rational. Our result already applies to fully connected instances with two inputs, two outputs, and one hidden layer of ReLU neurons. Thereby, we strengthen a result by Abrahamsen, Kleist and Miltzow [NeurIPS 2021]. A consequence of this is that a combinatorial search algorithm like the one by Arora, Basu, Mianjy and Mukherjee [ICLR 2018] is impossible for networks with more than one output dimension, unless $\text{NP} = \exists\mathbb{R}$.</details>

### Training Private Models That Know What They Don’t Know

**Authors:** Stephan Rabanser, Anvith Thudi, Abhradeep Guha Thakurta, Krishnamurthy Dvijotham, Nicolas Papernot

**<details><summary>Abstract**</summary>

Training reliable deep learning models which avoid making overconfident but incorrect predictions is a longstanding challenge. This challenge is further exacerbated when learning has to be differentially private: protection provided to sensitive data comes at the price of injecting additional randomness into the learning process. In this work, we conduct a thorough empirical investigation of selective classifiers---that can abstain under uncertainty---under a differential privacy constraint. We find that some popular selective prediction approaches are ineffective in a differentially private setting because they increase the risk of privacy leakage. At the same time, we identify that a recent approach that only uses checkpoints produced by an off-the-shelf private learning algorithm stands out as particularly suitable under DP. Further, we show that differential privacy does not just harm utility but also degrades selective classification performance. To analyze this effect across privacy levels, we propose a novel evaluation mechanism which isolates selective prediction performance across model utility levels at full coverage. Our experimental results show that recovering the performance level attainable by non-private models is possible but comes at a considerable coverage cost as the privacy budget decreases.</details>

### Training Transformers with 4-bit Integers

**Authors:** Haocheng Xi, ChangHao Li, Jianfei Chen, Jun Zhu

**<details><summary>Abstract**</summary>

Quantizing the activation, weight, and gradient to 4-bit is promising to accelerate neural network training. However, existing 4-bit training methods require custom numerical formats which are not supported by contemporary hardware. In this work, we propose a training method for transformers with all matrix multiplications implemented with the INT4 arithmetic. Training with an ultra-low INT4 precision is challenging. To achieve this, we carefully analyze the specific structures of activation and gradients in transformers to propose dedicated quantizers for them. For forward propagation, we identify the challenge of outliers and propose a Hadamard quantizer to suppress the outliers. For backpropagation, we leverage the structural sparsity of gradients by proposing bit splitting and leverage score sampling techniques to quantize gradients accurately. Our algorithm achieves competitive accuracy on a wide range of tasks including natural language understanding, machine translation, and image classification. Unlike previous 4-bit training methods, our algorithm can be implemented on the current generation of GPUs. Our prototypical linear operator implementation is up to 2.2 times faster than the FP16 counterparts and speeds up the training by 17.8\% on average for sufficiently large models. Our code is available at https://github.com/xijiu9/Train\_Transformers\_with\_INT4.</details>

### Training biologically plausible recurrent neural networks on cognitive tasks with long-term dependencies

**Authors:** Wayne Soo, Vishwa Goudar, Xiao-Jing Wang

**<details><summary>Abstract**</summary>

Training recurrent neural networks (RNNs) has become a go-to approach for generating and evaluating mechanistic neural hypotheses for cognition. The ease and efficiency of training RNNs with backpropagation through time and the availability of robustly supported deep learning libraries has made RNN modeling more approachable and accessible to neuroscience. Yet, a major technical hindrance remains. Cognitive processes such as working memory and decision making involve neural population dynamics over a long period of time within a behavioral trial and across trials. It is difficult to train RNNs to accomplish tasks where neural representations and dynamics have long temporal dependencies without gating mechanisms such as LSTMs or GRUs which currently lack experimental support and prohibit direct comparison between RNNs and biological neural circuits. We tackled this problem based on the idea of specialized skip-connections through time to support the emergence of task-relevant dynamics, and subsequently reinstitute biological plausibility by reverting to the original architecture. We show that this approach enables RNNs to successfully learn cognitive tasks that prove impractical if not impossible to learn using conventional methods. Over numerous tasks considered here, we achieve less training steps and shorter wall-clock times, particularly in tasks that require learning long-term dependencies via temporal integration over long timescales or maintaining a memory of past events in hidden-states. Our methods expand the range of experimental tasks that biologically plausible RNN models can learn, thereby supporting the development of theory for the emergent neural mechanisms of computations involving long-term dependencies.</details>

### Trajectory Alignment: Understanding the Edge of Stability Phenomenon via Bifurcation Theory

**Authors:** Minhak Song, Chulhee Yun

**<details><summary>Abstract**</summary>

Cohen et al. (2021) empirically study the evolution of the largest eigenvalue of the loss Hessian, also known as sharpness, along the gradient descent (GD) trajectory and observe the Edge of Stability (EoS) phenomenon. The sharpness increases at the early phase of training (referred to as progressive sharpening), and eventually saturates close to the threshold of $2 / \text{(step size)}$. In this paper, we start by demonstrating through empirical studies that when the EoS phenomenon occurs, different GD trajectories (after a proper reparameterization) align on a specific bifurcation diagram independent of initialization. We then rigorously prove this trajectory alignment phenomenon for a two-layer fully-connected linear network and a single-neuron nonlinear network trained with a single data point. Our trajectory alignment analysis establishes both progressive sharpening and EoS phenomena, encompassing and extending recent findings in the literature.</details>

### Transfer Learning with Affine Model Transformation

**Authors:** Shunya Minami, Kenji Fukumizu, Yoshihiro Hayashi, Ryo Yoshida

**<details><summary>Abstract**</summary>

Supervised transfer learning has received considerable attention due to its potential to boost the predictive power of machine learning in scenarios where data are scarce. Generally, a given set of source models and a dataset from a target domain are used to adapt the pre-trained models to a target domain by statistically learning domain shift and domain-specific factors. While such procedurally and intuitively plausible methods have achieved great success in a wide range of real-world applications, the lack of a theoretical basis hinders further methodological development. This paper presents a general class of transfer learning regression called affine model transfer, following the principle of expected-square loss minimization. It is shown that the affine model transfer broadly encompasses various existing methods, including the most common procedure based on neural feature extractors. Furthermore, the current paper clarifies theoretical properties of the affine model transfer such as generalization error and excess risk. Through several case studies, we demonstrate the practical benefits of modeling and estimating inter-domain commonality and domain-specific factors separately with the affine-type transfer models.</details>

### Transfer learning for atomistic simulations using GNNs and kernel mean embeddings

**Authors:** John Falk, Luigi Bonati, Pietro Novelli, Michele Parrinello, Massimiliano Pontil

**<details><summary>Abstract**</summary>

Interatomic potentials learned using machine learning methods have been successfully applied to atomistic simulations. However, accurate models require large training datasets, while generating reference calculations is computationally demanding. To bypass this difficulty, we propose a transfer learning algorithm that leverages the ability of graph neural networks (GNNs) to represent chemical environments together with kernel mean embeddings. We extract a feature map from GNNs pre-trained on the OC20 dataset and use it to learn the potential energy surface from system-specific datasets of catalytic processes. Our method is further enhanced by incorporating into the kernel the chemical species information, resulting in improved performance and interpretability. We test our approach on a series of realistic datasets of increasing complexity, showing excellent generalization and transferability performance, and improving on methods that rely on GNNs or ridge regression alone, as well as similar fine-tuning approaches.</details>

### Transferable Adversarial Robustness for Categorical Data via Universal Robust Embeddings

**Authors:** Klim Kireev, Maksym Andriushchenko, Carmela Troncoso, Nicolas Flammarion

**<details><summary>Abstract**</summary>

Research on adversarial robustness is primarily focused on image and text data. Yet, many scenarios in which lack of robustness can result in serious risks, such as fraud detection, medical diagnosis, or recommender systems often do not rely on images or text but instead on tabular data. Adversarial robustness in tabular data poses two serious challenges. First, tabular datasets often contain categorical features, and therefore cannot be tackled directly with existing optimization procedures. Second, in the tabular domain, algorithms that are not based on deep networks are widely used and offer great performance, but algorithms to enhance robustness are tailored to neural networks (e.g. adversarial training).In this paper, we tackle both challenges. We present a method that allows us to train adversarially robust deep networks for tabular data and to transfer this robustness to other classifiers via universal robust embeddings tailored to categorical data. These embeddings, created using a bilevel alternating minimization framework, can be transferred to boosted trees or random forests making them robust without the need for adversarial training while preserving their high accuracy on tabular data. We show that our methods outperform existing techniques within a practical threat model suitable for tabular data.</details>

### Transformers are uninterpretable with myopic methods: a case study with bounded Dyck grammars

**Authors:** Kaiyue Wen, Yuchen Li, Bingbin Liu, Andrej Risteski

**<details><summary>Abstract**</summary>

Transformer interpretability aims to understand the algorithm implemented by a learned Transformer by examining various aspects of the model, such as the weight matrices or the attention patterns.In this work, through a combination of theoretical results and carefully controlled experiments on synthetic data, we take a critical viewof methods that exclusively focus on individual parts of the model, rather than consider the network as a whole.We consider a simple synthetic setup of learning a (bounded) Dyck language. Theoretically, we show that the set of models that (exactly or approximately) solve this task satisfy a structural characterization derived from ideas in formal languages (the pumping lemma).We use this characterization to show that the set of optima is qualitatively rich; in particular, the attention pattern of a single layer can be "nearly randomized", while preserving the functionality of the network.We also show via extensive experiments that these constructions are not merely a theoretical artifact: even with severe constraints to the architecture of the model, vastly different solutions can be reached via standard training. Thus, interpretability claims based on inspecting individual heads or weight matrices in the Transformer can be misleading.</details>

### Transformers learn to implement preconditioned gradient descent for in-context learning

**Authors:** Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, Suvrit Sra

**<details><summary>Abstract**</summary>

Motivated by the striking ability of transformers for in-context learning, several works demonstrate that transformers can implement algorithms like gradient descent. By a careful construction of weights, these works show that multiple layers of transformers are expressive enough to simulate gradient descent iterations. Going beyond the question of expressivity, we ask: \emph{Can transformers can learn to implement such algorithms by training over random problem instances?} To our knowledge, we make the first theoretical progress toward this question via analysis of the loss landscape for linear transformers trained over random instances of linear regression. For a single attention layer, we prove the global minimum of the training objective implements a single iteration of preconditioned gradient descent.Notably, the preconditioning matrix not only adapts to the input distribution but also to the variance induced by data inadequacy.  For a transformer with $k$ attention layers, we prove certain critical points of the training objective implement $k$ iterations of preconditioned gradient descent. Our results call for future theoretical studies on learning algorithms by training transformers.</details>

### [Spotlight] Transition-constant Normalization for Image Enhancement

**Authors:** Jie Huang, man zhou, Jinghao Zhang, Gang Yang, Mingde Yao, Chongyi Li, Zhiwei Xiong, Feng Zhao

**<details><summary>Abstract**</summary>

Normalization techniques that capture image style by statistical representation have become a popular component in deep neural networks.Although image enhancement can be considered as a form of style transformation, there has been little exploration of how normalization affect the enhancement performance. To fully leverage the potential of normalization, we present a novel Transition-Constant Normalization (TCN) for various image enhancement tasks.Specifically, it consists of two streams of normalization operations arranged under an invertible constraint, along with a feature sub-sampling operation that satisfies the normalization constraint.TCN enjoys several merits, including being parameter-free, plug-and-play, and incurring no additional computational costs.We provide various formats to utilize TCN for image enhancement, including seamless integration with enhancement networks, incorporation into encoder-decoder architectures for downsampling, and implementation of efficient architectures.Through extensive experiments on multiple image enhancement tasks, like low-light enhancement, exposure correction, SDR2HDR translation, and image dehazing, our TCN consistently demonstrates performance improvements.Besides, it showcases extensive ability in other tasks including pan-sharpening and medical segmentation.The code is available at 	extit{	extcolor{blue}{https://github.com/huangkevinj/TCNorm}}.</details>

### [Spotlight] Tree-Based Diffusion Schrödinger Bridge with Applications to Wasserstein Barycenters

**Authors:** Maxence Noble, Valentin De Bortoli, Arnaud Doucet, Alain Durmus

**<details><summary>Abstract**</summary>

Multi-marginal Optimal Transport (mOT), a generalization of OT, aims at minimizing the integral of a cost function with respect to a distribution with some prescribed marginals. In this paper, we consider an entropic version of mOT  with a tree-structured quadratic cost, i.e., a function that can be written as  a sum of pairwise cost functions between the nodes of a tree. To address this  problem, we develop Tree-based Diffusion Schr\"odinger Bridge (TreeDSB), an  extension of the Diffusion Schr\"odinger Bridge (DSB) algorithm. TreeDSB  corresponds to a dynamic and continuous state-space counterpart of the  multimarginal Sinkhorn algorithm. A notable use case of our methodology is to  compute Wasserstein barycenters which can be recast as the solution of a mOT  problem on a star-shaped tree. We demonstrate that our methodology can be applied in high-dimensional settings such as image interpolation and  Bayesian fusion.</details>

### Trial matching: capturing variability with data-constrained spiking neural networks

**Authors:** Christos Sourmpis, Carl Petersen, Wulfram Gerstner, Guillaume Bellec

**<details><summary>Abstract**</summary>

Simultaneous behavioral and electrophysiological recordings call for new methods to reveal the interactions between neural activity and behavior. A milestone would be an interpretable model of the co-variability of spiking activity and behavior across trials. Here, we model a mouse cortical sensory-motor pathway in a tactile detection task reported by licking with a large recurrent spiking neural network (RSNN), fitted to the recordings via gradient-based optimization. We focus specifically on the difficulty to match the trial-to-trial variability in the data. Our solution relies on optimal transport to define a distance between the distributions of generated and recorded trials. The technique is applied to artificial data and neural recordings covering six cortical areas. We find that the resulting RSNN can generate realistic cortical activity and predict jaw movements across the main modes of trial-to-trial variability. Our analysis also identifies an unexpected mode of variability in the data corresponding to task-irrelevant movements of the mouse.</details>

### Truly Scale-Equivariant Deep Nets with Fourier Layers

**Authors:** Md Ashiqur Rahman, Raymond A. Yeh

**<details><summary>Abstract**</summary>

In computer vision, models must be able to adapt to changes in image resolution to effectively carry out tasks such as image segmentation; This is known as scale-equivariance. Recent works have made progress in developing scale-equivariant convolutional neural networks, e.g., through weight-sharing and kernel resizing. However, these networks are not truly scale-equivariant in practice. Specifically, they do not consider anti-aliasing as they formulate the down-scaling operation in the continuous domain. To address this shortcoming, we directly formulate down-scaling in the discrete domain with consideration of anti-aliasing. We then propose a novel architecture based on Fourier layers to achieve truly scale-equivariant deep nets, i.e., absolute zero equivariance-error. Following prior works, we test this model on MNIST-scale and STL-10 datasets. Our proposed model achieves competitive classification performance while maintaining zero equivariance-error.</details>

### Truncating Trajectories in Monte Carlo Policy Evaluation: an Adaptive Approach

**Authors:** Riccardo Poiani, Nicole Nobili, Alberto Maria Metelli, Marcello Restelli

**<details><summary>Abstract**</summary>

Policy evaluation via Monte Carlo (MC) simulation is at the core of many MC Reinforcement Learning (RL) algorithms (e.g., policy gradient methods). In this context, the designer of the learning system specifies an interaction budget that the agent usually spends by collecting trajectories of *fixed length* within a simulator. However, is this data collection strategy the best option? To answer this question, in this paper, we consider as quality index the variance of an unbiased policy return estimator that uses trajectories of different lengths, i.e., *truncated*. We first derive a closed-form expression of this variance that clearly shows the sub-optimality of the fixed-length trajectory schedule. Furthermore, it suggests that adaptive data collection strategies that spend the available budget sequentially might be able to allocate a larger portion of transitions in timesteps in which more accurate sampling is required to reduce the variance of the final estimate. Building on these findings, we present an *adaptive* algorithm called **R**obust and **I**terative **D**ata collection strategy **O**ptimization (RIDO). The main intuition behind RIDO is to split the available interaction budget into mini-batches. At each round, the agent determines the most convenient schedule of trajectories that minimizes an empirical and robust estimate of the estimator's variance. After discussing the theoretical properties of our method, we conclude by assessing its performance across multiple domains. Our results show that RIDO can adapt its trajectory schedule toward timesteps where more sampling is required to increase the quality of the final estimation.</details>

### Two-Stage Learning to Defer with Multiple Experts

**Authors:** Anqi Mao, Christopher Mohri, Mehryar Mohri, Yutao Zhong

**<details><summary>Abstract**</summary>

We study a two-stage scenario for learning to defer with multiple experts, which is crucial in practice for many applications.  In this scenario, a predictor is derived in a first stage by training with a common loss function such as cross-entropy.  In the second stage, a deferral function is learned to assign the most suitable expert to each input.  We design a new family of surrogate loss functions for this scenario both in the score-based and the predictor-rejector settings and prove that they are supported by $H$-consistency bounds, which implies their Bayes-consistency.  Moreover, we show that, for a constant cost function, our two-stage surrogate losses are realizable $H$-consistent. While the main focus of this work is a theoretical analysis, we also report the results of several experiments on CIFAR-10 and SVHN datasets.</details>

### Type-to-Track: Retrieve Any Object via Prompt-based Tracking

**Authors:** Pha Nguyen, Kha Gia Quach, Kris Kitani, Khoa Luu

**<details><summary>Abstract**</summary>

One of the recent trends in vision problems is to use natural language captions to describe the objects of interest. This approach can overcome some limitations of traditional methods that rely on bounding boxes or category annotations. This paper introduces a novel paradigm for Multiple Object Tracking called Type-to-Track, which allows users to track objects in videos by typing natural language descriptions. We present a new dataset for that Grounded Multiple Object Tracking task, called GroOT, that contains videos with various types of objects and their corresponding textual captions describing their appearance and action in detail. Additionally, we introduce two new evaluation protocols and formulate evaluation metrics specifically for this task. We develop a new efficient method that models a transformer-based eMbed-ENcoDE-extRact framework (MENDER) using the third-order tensor decomposition. The experiments in five scenarios show that our MENDER approach outperforms another two-stage design in terms of accuracy and efficiency, up to 14.7\% accuracy and $4\times$ speed faster.</details>

### Unbounded Differentially Private Quantile and Maximum Estimation

**Authors:** David Durfee

**<details><summary>Abstract**</summary>

In this work we consider the problem of differentially private computation ofquantiles for the data, especially the highest quantiles such as maximum, butwith an unbounded range for the dataset. We show that this can be doneefficiently through a simple invocation of $\texttt{AboveThreshold}$, asubroutine that is iteratively called in the fundamental Sparse VectorTechnique, even when there is no upper bound on the data. In particular, weshow that this procedure can give more accurate and robust estimates on thehighest quantiles with applications towards clipping that is essential fordifferentially private sum and mean estimation. In addition, we show how twoinvocations can handle the fully unbounded data setting. Within our study, weshow that an improved analysis of $\texttt{AboveThreshold}$ can improve theprivacy guarantees for the widely used Sparse Vector Technique that is ofindependent interest. We give a more general characterization of privacy lossfor $\texttt{AboveThreshold}$ which we immediately apply to our method forimproved privacy guarantees. Our algorithm only requires one $O(n)$ passthrough the data, which can be unsorted, and each subsequent query takes $O(1)$time. We empirically compare our unbounded algorithm with the state-of-the-artalgorithms in the bounded setting. For inner quantiles, we find that our methodoften performs better on non-synthetic datasets. For the maximal quantiles,which we apply to differentially private sum computation, we find that ourmethod performs significantly better.</details>

### Uncovering Prototypical Knowledge for Weakly Open-Vocabulary Semantic Segmentation

**Authors:** Fei Zhang, Tianfei Zhou, Boyang Li, Hao He, Chaofan Ma, Tianjiao Zhang, Jiangchao Yao, Ya Zhang, Yanfeng Wang

**<details><summary>Abstract**</summary>

This paper studies the problem of weakly open-vocabulary semantic segmentation (WOVSS), which learns to segment objects of arbitrary classes using mere image-text pairs. Existing works turn to enhance the vanilla vision transformer by introducing explicit grouping recognition, i.e., employing several group tokens/centroids to cluster the image tokens and perform the group-text alignment. Nevertheless, these methods suffer from a granularity inconsistency regarding the usage of group tokens, which are aligned in the all-to-one v.s. one-to-one manners during the training and inference phases, respectively. We argue that this discrepancy arises from the lack of elaborate supervision for each group token. To bridge this granularity gap, this paper explores explicit supervision for the group tokens from the prototypical knowledge. To this end, this paper proposes the non-learnable prototypical regularization (NPR) where non-learnable prototypes are estimated from source features to serve as supervision and enable contrastive matching of the group tokens. This regularization encourages the group tokens to segment objects with less redundancy and capture more comprehensive semantic regions, leading to increased compactness and richness. Based on NPR, we propose the prototypical guidance segmentation network (PGSeg) that incorporates multi-modal regularization by leveraging prototypical sources from both images and texts at different levels, progressively enhancing the segmentation capability with diverse prototypical patterns. Experimental results show that our proposed method achieves state-of-the-art performance on several benchmark datasets.</details>

### [Oral] Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation

**Authors:** Diederik Kingma, Ruiqi Gao

**Oral Presentation:** We, Dec 13, 08:00 -- Oral 3C

**<details><summary>Abstract**</summary>

To achieve the highest perceptual quality, state-of-the-art diffusion models are optimized with objectives that typically look very different from the maximum likelihood and the Evidence Lower Bound (ELBO) objectives. In this work, we reveal that diffusion model objectives are actually closely related to the ELBO.Specifically, we show that all commonly used diffusion model objectives equate to a weighted integral of ELBOs over different noise levels, where the weighting depends on the specific objective used. Under the condition of monotonic weighting, the connection is even closer: the diffusion objective then equals the ELBO, combined with simple data augmentation, namely Gaussian noise perturbation. We show that this condition holds for a number of state-of-the-art diffusion models. In experiments, we explore new monotonic weightings and demonstrate their effectiveness, achieving state-of-the-art FID scores on the high-resolution ImageNet benchmark.</details>

### Understanding Few-Shot Learning: Measuring Task Relatedness and Adaptation Difficulty via Attributes

**Authors:** Minyang Hu, Hong Chang, Zong Guo, Bingpeng MA, Shiguang Shan, Xilin Chen

**<details><summary>Abstract**</summary>

Few-shot learning (FSL) aims to learn novel tasks with very few labeled samples by leveraging experience from \emph{related} training tasks. In this paper, we try to understand FSL by exploring two key questions: (1) How to quantify the relationship between \emph{ training} and \emph{novel} tasks? (2) How does the relationship affect the \emph{adaptation difficulty} on novel tasks for different models? To answer the first question, we propose Task Attribute Distance (TAD) as a metric to quantify the task relatedness via attributes. Unlike other metrics, TAD is independent of models, making it applicable to different FSL models. To address the second question, we utilize TAD metric to establish a theoretical connection between task relatedness and task adaptation difficulty. By deriving the generalization error bound on a novel task, we discover how TAD measures the adaptation difficulty on novel tasks for different models. To validate our theoretical results, we conduct experiments on three benchmarks. Our experimental results confirm that TAD metric effectively quantifies the task relatedness and reflects the adaptation difficulty on novel tasks for various FSL methods, even if some of them do not learn attributes explicitly or human-annotated attributes are not provided. Our code is available at \href{https://github.com/hu-my/TaskAttributeDistance}{https://github.com/hu-my/TaskAttributeDistance}.</details>

### Understanding and Improving Ensemble Adversarial Defense

**Authors:** Yian Deng, Tingting Mu

**<details><summary>Abstract**</summary>

The strategy of ensemble has become popular in adversarial defense, which trains multiple base classifiers to defend against adversarial attacks in a cooperative manner. Despite the empirical success, theoretical explanations on why an ensemble of adversarially trained classifiers is more robust than single ones remain unclear. To fill in this gap, we develop a new error theory dedicated to understanding ensemble adversarial defense,  demonstrating a provable 0-1 loss reduction on challenging sample sets in adversarial defense scenarios. Guided by this theory, we propose an effective approach to improve ensemble adversarial defense, named interactive global adversarial training (iGAT). The proposal includes (1) a probabilistic distributing rule that selectively allocates to different base classifiers adversarial examples that are globally challenging to the ensemble, and (2) a regularization term to rescue the severest weaknesses of the base classifiers. Being tested over various existing ensemble adversarial defense techniques,  iGAT is capable of boosting their performance by up to 17\%  evaluated using  CIFAR10 and CIFAR100 datasets under both white-box and black-box attacks.</details>

### Understanding the Limitations of Deep Models for Molecular property prediction: Insights and Solutions

**Authors:** Jun Xia, Lecheng Zhang, Xiao Zhu, Yue Liu, Zhangyang Gao, Bozhen Hu, Cheng Tan, Jiangbin Zheng, Siyuan Li, Stan Z. Li

**<details><summary>Abstract**</summary>

Molecular Property Prediction (MPP) is a crucial task in the AI-driven Drug Discovery (AIDD) pipeline, which has recently gained considerable attention thanks to advancements in deep learning. However, recent research has revealed that deep models struggle to beat traditional non-deep ones on MPP. In this study, we benchmark 12 representative models (3 non-deep models and 9 deep models) on 15 molecule datasets. Through the most comprehensive study to date, we make the following key observations: \textbf{(\romannumeral 1)} Deep models are generally unable to outperform non-deep ones; \textbf{(\romannumeral 2)} The failure of deep models on MPP cannot be solely attributed to the small size of molecular datasets; \textbf{(\romannumeral 3)} In particular, some traditional models including XGB and RF that use molecular fingerprints as inputs tend to perform better than other competitors. Furthermore, we conduct extensive empirical investigations into the unique patterns of molecule data and inductive biases of various models underlying these phenomena. These findings stimulate us to develop a simple-yet-effective feature mapping method for molecule data prior to feeding them into deep models. Empirically, deep models equipped with this mapping method can beat non-deep ones in most MoleculeNet datasets. Notably, the effectiveness is further corroborated by extensive experiments on cutting-edge dataset related to COVID-19 and activity cliff datasets.</details>

### Unified 3D Segmenter As Prototypical Classifiers

**Authors:** Zheyun Qin, Cheng Han, Qifan Wang, Xiushan Nie, Yilong Yin, Lu Xiankai

**<details><summary>Abstract**</summary>

The task of point cloud segmentation, comprising semantic, instance, and panoptic segmentation, has been mainly tackled by designing task-specific network architectures, which often lack the flexibility to generalize across tasks, thus resulting in a fragmented research landscape. In this paper, we introduce ProtoSEG, a prototype-based model that unifies semantic, instance, and panoptic segmentation tasks. Our approach treats these three homogeneous tasks as a classification problem with different levels of granularity. By leveraging a Transformer architecture, we extract point embeddings to optimize prototype-class distances and dynamically learn class prototypes to accommodate the end tasks. Our prototypical design enjoys simplicity and transparency, powerful representational learning, and ad-hoc explainability. Empirical results demonstrate that ProtoSEG outperforms concurrent well-known specialized architectures on 3D point cloud benchmarks, achieving 72.3%, 76.4% and 74.2% mIoU for semantic segmentation on S3DIS, ScanNet V2 and SemanticKITTI, 66.8% mCov and 51.2% mAP for instance segmentation on S3DIS and ScanNet V2, 62.4% PQ for panoptic segmentation on SemanticKITTI, validating the strength of our concept and the effectiveness of our algorithm. The code and models are available at https://github.com/zyqin19/PROTOSEG.</details>

### Unifying GANs and Score-Based Diffusion as Generative Particle Models

**Authors:** Jean-Yves Franceschi, Mike Gartrell, Ludovic Dos Santos, Thibaut Issenhuth, Emmanuel de Bézenac, Mickael Chen, Alain Rakotomamonjy

**<details><summary>Abstract**</summary>

Particle-based deep generative models, such as gradient flows and score-based diffusion models, have recently gained traction thanks to their striking performance. Their principle of displacing particle distributions using differential equations is conventionally seen as opposed to the previously widespread generative adversarial networks (GANs), which involve training a pushforward generator network. In this paper we challenge this interpretation, and propose a novel framework that unifies particle and adversarial generative models by framing generator training as a generalization of particle models. This suggests that a generator is an optional addition to any such generative model. Consequently, integrating a generator into a score-based diffusion model and training a GAN without a generator naturally emerge from our framework. We empirically test the viability of these original models as proofs of concepts of potential applications of our framework.</details>

### Universality laws for Gaussian mixtures in generalized linear models

**Authors:** Yatin Dandi, Ludovic Stephan, Florent Krzakala, Bruno Loureiro, Lenka Zdeborová

**<details><summary>Abstract**</summary>

A recent line of work in high-dimensional statistics working under the Gaussian mixture hypothesis has led to a number of results in the context of empirical risk minimization, Bayesian uncertainty quantification, separation of kernel methods and neural networks, ensembling and fluctuation of random features. We provide rigorous proofs for the applicability of these results to a general class of datasets $(\mathbf{x_i},y_i, {i=1,\dots,n})$ containing independent samples from a mixture distribution $\sum_{c\in\mathcal{C}} \rho_{c}P_{c}^{\mathbf{x}}$. Specifically, we consider the hypothesis class of generalized linear models $\hat{y} = F(\mathbf{\Theta}^{\top}\mathbf{x})$ and investigate the asymptotic joint statistics of a family of generalized linear estimators $(\mathbf{\Theta}^{(1)}, \dots, \mathbf{\Theta}^{(M)})$, obtained either from (a) minimizing an empirical risk $\hat{R_n}^{(m)}(\mathbf{\Theta}^{(m)};\mathbf{X},\mathbf{y})$ or (b) sampling from the associated Gibbs measure $\exp(-\beta n \hat{R_n}^{(m)}(\mathbf{\Theta}^{(m)};\mathbf{X},\mathbf{y}))$. Our main contribution is to characterize under which conditions the asymptotic joint statistics of this family depends (on a weak sense) only on the means and covariances of the class conditional features distribution $P_{c}^{\mathbf{x}}$. This allows us to prove the universality of different quantities of interest, including training, generalization errors, as well as the geometrical properties and correlations of the estimators.</details>

### Unlocking Feature Visualization for Deep Network with MAgnitude Constrained Optimization

**Authors:** Thomas FEL, Thibaut Boissin, Victor Boutin, Agustin PICARD, Paul Novello, Julien Colin, Drew Linsley, Tom ROUSSEAU, Remi Cadene, Lore Goetschalckx, Laurent Gardes, Thomas Serre

**<details><summary>Abstract**</summary>

Feature visualization has gained significant popularity as an explainability method, particularly after the influential work by Olah et al. in 2017. Despite its success, its widespread adoption has been limited due to issues in scaling to deeper neural networks and the reliance on tricks to generate interpretable images. Here, we describe MACO, a simple approach to address these shortcomings. It consists in optimizing solely an image's phase spectrum while keeping its magnitude constant to ensure that the generated explanations lie in the space of natural images. Our approach yields significantly better results -- both qualitatively and quantitatively -- unlocking efficient and interpretable feature visualizations for state-of-the-art neural networks. We also show that our approach exhibits an attribution mechanism allowing to augment feature visualizations with spatial importance. Furthermore, we enable quantitative evaluation of feature visualizations by introducing 3 metrics: transferability, plausibility, and alignment with natural images. We validate our method on various applications and we introduce a website featuring MACO visualizations for all classes of the ImageNet dataset, which will be made available upon acceptance. Overall, our study unlocks feature visualizations for the largest, state-of-the-art classification networks without resorting to any parametric prior image model, effectively advancing a field that has been stagnating since 2017 (Olah et al, 2017).</details>

### Utilitarian Algorithm Configuration

**Authors:** Devon Graham, Kevin Leyton-Brown, Tim Roughgarden

**<details><summary>Abstract**</summary>

We present the first nontrivial procedure for configuring heuristic algorithms to maximize the utility provided to their end users while also offering theoretical guarantees about performance. Existing procedures seek configurations that minimize expected runtime. However, very recent theoretical work argues that expected runtime minimization fails to capture algorithm designers' preferences. Here we show that the utilitarian objective also confers significant algorithmic benefits. Intuitively, this is because mean runtime is dominated by extremely long runs even when they are incredibly rare; indeed, even when an algorithm never gives rise to such long runs, configuration procedures that provably minimize mean runtime must perform a huge number of experiments to demonstrate this fact. In contrast, utility is bounded and monotonically decreasing in runtime, allowing for meaningful empirical bounds on a configuration's performance. This paper builds on this idea to describe effective and theoretically sound configuration procedures. We prove upper bounds on the runtime of these procedures that are similar to theoretical lower bounds, while also demonstrating their performance empirically.</details>

### VCC: Scaling Transformers to 128K Tokens or More by Prioritizing Important Tokens

**Authors:** Zhanpeng Zeng, Cole Hawkins, Mingyi Hong, Aston Zhang, Nikolaos Pappas, Vikas Singh, Shuai Zheng

**<details><summary>Abstract**</summary>

Transformers are central in modern natural language processing and computer vision applications. Despite recent works devoted to reducing the quadratic cost of such models with respect to sequence length, dealing with ultra long sequences (e.g., $>$16K tokens) remains challenging. Applications such as answering questions based on a book or summarizing a scientific article are inefficient or infeasible. Here, we propose to significantly improve the efficiency of Transformers for ultra long sequences, by compressing the sequence into a much smaller representation at each layer. Specifically, by exploiting the fact that in many tasks, only a small subset of special tokens, which we call VIP-tokens, are most relevant to the final prediction, we propose a VIP-token centric compression (VCC) scheme which selectively compresses the sequence based on their impact on approximating the representation of the VIP-tokens. Compared with competitive baselines, our algorithm is not only efficient (achieving more than $3	imes$ compute efficiency gain compared to baselines on 4K and 16K lengths), but also offers competitive/better performance on a large number of tasks. Further, we show that our algorithm scales to 128K tokens (or more) while consistently offering accuracy improvement. Code is available at https://github.com/mlpen/VCC.</details>

### VanillaNet: the Power of Minimalism in Deep Learning

**Authors:** Hanting Chen, Yunhe Wang, Jianyuan Guo, Dacheng Tao

**<details><summary>Abstract**</summary>

At the heart of foundation models is the philosophy of "more is different", exemplified by the astonishing success in computer vision and natural language processing. However, the challenges of optimization and inherent complexity of transformer models call for a paradigm shift towards simplicity. In this study, we introduce VanillaNet, a neural network architecture that embraces elegance in design. By avoiding high depth, shortcuts, and intricate operations like self-attention, VanillaNet is refreshingly concise yet remarkably powerful. Each layer is carefully crafted to be compact and straightforward, with nonlinear activation functions pruned after training to restore the original architecture. VanillaNet overcomes the challenges of inherent complexity, making it ideal for resource-constrained environments. Its easy-to-understand and highly simplified architecture opens new possibilities for efficient deployment. Extensive experimentation demonstrates that VanillaNet delivers performance on par with renowned deep neural networks and vision transformers, showcasing the power of minimalism in deep learning. This visionary journey of VanillaNet has significant potential to redefine the landscape and challenge the status quo of foundation model, setting a new path for elegant and effective model design. Pre-trained models and codes are available at https://github.com/huawei-noah/VanillaNet and https://gitee.com/mindspore/models/tree/master/research/cv/vanillanet</details>

### Variational Gaussian processes for linear inverse problems

**Authors:** Thibault RANDRIANARISOA, Botond Szabo

**<details><summary>Abstract**</summary>

By now Bayesian methods are routinely used in practice for solving inverse problems. In inverse problems the parameter or signal of interest is observed only indirectly, as an image of a given map, and the observations are typically further corrupted with noise. Bayes offers a natural way to regularize these problems via the prior distribution and provides a probabilistic solution, quantifying the remaining uncertainty in the problem. However, the computational costs of standard, sampling based Bayesian approaches can be overly large in such complex models. Therefore, in practice variational Bayes is becoming increasingly popular. Nevertheless, the theoretical understanding of these methods is still relatively limited, especially in context of inverse problems.In our analysis we investigate variational Bayesian methods for Gaussian process priors to solve linear inverse problems. We consider both mildly and severely ill-posed inverse problems and work with the popular inducing variable variational Bayes approach proposed by Titsias [Titsias, 2009]. We derive posterior contraction rates for the variational posterior in general settings and show that the minimax estimation rate can be attained by correctly tunned procedures. As specific examples we consider a collection of inverse problems including the heat equation, Volterra operator and Radon transform and inducing variable methods based on population and empirical spectral features.</details>

### Variational Monte Carlo on a Budget — Fine-tuning pre-trained Neural Wavefunctions

**Authors:** Michael Scherbela, Leon Gerard, Philipp Grohs

**<details><summary>Abstract**</summary>

Obtaining accurate solutions to the Schrödinger equation is the key challenge in computational quantum chemistry. Deep-learning-based Variational Monte Carlo (DL-VMC) has recently outperformed conventional approaches in terms of accuracy, but only at large computational cost.Whereas in many domains models are trained once and subsequently applied for inference, accurate DL-VMC so far requires a full optimization for every new problem instance, consuming thousands of GPUhs even for small molecules.We instead propose a DL-VMC model which has been pre-trained using self-supervised wavefunction optimization on a large and chemically diverse set of molecules. Applying this model to new molecules without any optimization, yields wavefunctions and absolute energies that outperform established methods such as CCSD(T)-2Z.To obtain accurate relative energies, only few fine-tuning steps of this base model are required.We accomplish this with a fully end-to-end machine-learned model, consisting of an improved geometry embedding architecture and an existing SE(3)-equivariant model to represent molecular orbitals. Combining this architecture with continuous sampling of geometries, we improve zero-shot accuracy by two orders of magnitude compared to the state of the art.We extensively evaluate the accuracy, scalability and limitations of our base model on a wide variety of test systems.</details>

### Variational Weighting for Kernel Density Ratios

**Authors:** Sangwoong Yoon, Frank Park, Gunsu YUN, Iljung Kim, Yung-Kyun Noh

**<details><summary>Abstract**</summary>

Kernel density estimation (KDE) is integral to a range of generative and discriminative tasks in machine learning. Drawing upon tools from the multidimensional calculus of variations, we derive an optimal weight function that reduces bias in standard kernel density estimates for density ratios, leading to improved estimates of prediction posteriors and information-theoretic measures. In the process, we shed light on some fundamental aspects of density estimation, particularly from the perspective of algorithms that employ KDEs as their main building blocks.</details>

### VeriX: Towards Verified Explainability of Deep Neural Networks

**Authors:** Min Wu, Haoze Wu, Clark Barrett

**<details><summary>Abstract**</summary>

We present **VeriX** (**Veri**fied e**X**plainability), a system for producing *optimal robust explanations* and generating *counterfactuals* along decision boundaries of machine learning models. We build such explanations and counterfactuals iteratively using constraint solving techniques and a heuristic based on feature-level sensitivity ranking. We evaluate our method on image recognition benchmarks and a real-world scenario of autonomous aircraft taxiing.</details>

### ViCA-NeRF: View-Consistency-Aware 3D Editing of Neural Radiance Fields

**Authors:** Jiahua Dong, Yu-Xiong Wang

**<details><summary>Abstract**</summary>

We introduce ViCA-NeRF, a view-consistency-aware method for 3D editing with text instructions. In addition to the implicit NeRF modeling, our key insight is to exploit two sources of regularization that {\em explicitly} propagate the editing information across different views, thus ensuring multi-view consistency. As {\em geometric regularization}, we leverage the depth information derived from the NeRF model to establish image correspondence between different views. As {\em learned regularization}, we align the latent codes in the 2D diffusion model between edited and unedited images, enabling us to edit key views and propagate the update to the whole scene. Incorporating these two regularizations, our ViCA-NeRF framework consists of two stages. In the initial stage, we blend edits from different views to create a preliminary 3D edit. This is followed by a second stage of NeRF training that is dedicated to further refining the scene's appearance. Experiments demonstrate that ViCA-NeRF provides more flexible, efficient editing with higher levels of consistency and details, compared with the state of the art.</details>

### Video Prediction Models as Rewards for Reinforcement Learning

**Authors:** Alejandro Escontrela, Ademi Adeniji, Wilson Yan, Ajay Jain, Xue Bin Peng, Ken Goldberg, Youngwoon Lee, Danijar Hafner, Pieter Abbeel

**<details><summary>Abstract**</summary>

Specifying reward signals that allow agents to learn complex behaviors is a long-standing challenge in reinforcement learning.A promising approach is to extract preferences for behaviors from unlabeled videos, which are widely available on the internet. We present Video Prediction Rewards (VIPER), an algorithm that leverages pretrained video prediction models as action-free reward signals for reinforcement learning. Specifically, we first train an autoregressive transformer on expert videos and then use the video prediction likelihoods as reward signals for a reinforcement learning agent. VIPER enables expert-level control without programmatic task rewards across a wide range of DMC, Atari, and RLBench tasks. Moreover, generalization of the video prediction model allows us to derive rewards for an out-of-distribution environment where no expert data is available, enabling cross-embodiment generalization for tabletop manipulation. We see our work as starting point for scalable reward specification from unlabeled videos that will benefit from the rapid advances in generative modeling. Source code and datasets are available on the project website: https://ViperRL.com</details>

### Visual Explanations of Image-Text Representations via Multi-Modal Information Bottleneck Attribution

**Authors:** Ying Wang, Tim G. J. Rudner, Andrew Wilson

**<details><summary>Abstract**</summary>

Vision-language pretrained models have seen remarkable success, but their application to safety-critical settings is limited by their lack of interpretability. To improve the interpretability of vision-language models such as CLIP, we propose a multi-modal information bottleneck (M2IB) approach that learns latent representations that compress irrelevant information while preserving relevant visual and textual features. We demonstrate how M2IB can be applied to attribution analysis of vision-language pretrained models, increasing attribution accuracy and improving the interpretability of such models when applied to safety-critical domains such as healthcare. Crucially, unlike commonly used unimodal attribution methods, M2IB does not require ground truth labels, making it possible to audit representations of vision-language pretrained models when multiple modalities but no ground-truth data is available. Using CLIP as an example, we demonstrate the effectiveness of M2IB attribution and show that it outperforms gradient-based, perturbation-based, and attention-based attribution methods both qualitatively and quantitatively.</details>

### Vocabulary-free Image Classification

**Authors:** Alessandro Conti, Enrico Fini, Massimiliano Mancini, Paolo Rota, Yiming Wang, Elisa Ricci

**<details><summary>Abstract**</summary>

Recent advances in large vision-language models have revolutionized the image classification paradigm. Despite showing impressive zero-shot capabilities, a pre-defined set of categories, a.k.a. the vocabulary, is assumed at test time for composing the textual prompts. However, such assumption can be impractical when the semantic context is unknown and evolving. We thus formalize a novel task, termed as Vocabulary-free Image Classification (VIC), where we aim to assign to an input image a class that resides in an unconstrained language-induced semantic space, without the prerequisite of a known vocabulary. VIC is a challenging task as the semantic space is extremely large, containing millions of concepts, with hard-to-discriminate fine-grained categories. In this work, we first empirically verify that representing this semantic space by means of an external vision-language database is the most effective way to obtain semantically relevant content for classifying the image. We then propose Category Search from External Databases (CaSED), a method that exploits a pre-trained vision-language model and an external vision-language database to address VIC in a training-free manner. CaSED first extracts a set of candidate categories from captions retrieved from the database based on their semantic similarity to the image, and then assigns to the image the best matching candidate category according to the same vision-language model. Experiments on benchmark datasets validate that CaSED outperforms other complex vision-language frameworks, while being efficient with much fewer parameters, paving the way for future research in this direction.</details>

### [Spotlight] VoxDet: Voxel Learning for Novel Instance Detection

**Authors:** Bowen Li, Jiashun Wang, Yaoyu Hu, Chen Wang, Sebastian Scherer

**<details><summary>Abstract**</summary>

Detecting unseen instances based on multi-view templates is a challenging problem due to its open-world nature. Traditional methodologies, which primarily rely on $2 \mathrm{D}$ representations and matching techniques, are often inadequate in handling pose variations and occlusions. To solve this, we introduce VoxDet, a pioneer 3D geometry-aware framework that fully utilizes the strong 3D voxel representation and reliable voxel matching mechanism. VoxDet first ingeniously proposes template voxel aggregation (TVA) module, effectively transforming multi-view 2D images into 3D voxel features. By leveraging associated camera poses, these features are aggregated into a compact 3D template voxel. In novel instance detection, this voxel representation demonstrates heightened resilience to occlusion and pose variations. We also discover that a $3 \mathrm{D}$ reconstruction objective helps to pre-train the 2D-3D mapping in TVA. Second, to quickly align with the template voxel, VoxDet incorporates a Query Voxel Matching (QVM) module. The 2D queries are first converted into their voxel representation with the learned 2D-3D mapping. We find that since the 3D voxel representations encode the geometry, we can first estimate the relative rotation and then compare the aligned voxels, leading to improved accuracy and efficiency. In addition to method, we also introduce the first instance detection benchmark, RoboTools, where 20 unique instances are video-recorded with camera extrinsic. RoboTools also provides 24 challenging cluttered scenarios with more than $9 \mathrm{k}$ box annotations. Exhaustive experiments are conducted on the demanding LineMod-Occlusion, YCB-video, and RoboTools benchmarks, where VoxDet outperforms various $2 \mathrm{D}$ baselines remarkably with faster speed. To the best of our knowledge, VoxDet is the first to incorporate implicit 3D knowledge for 2D novel instance detection tasks.</details>

### Wasserstein distributional robustness of neural networks

**Authors:** Xingjian Bai, Guangyi He, Yifan Jiang, Jan Obloj

**<details><summary>Abstract**</summary>

Deep neural networks are known to be vulnerable to adversarial attacks (AA). For an image recognition task, this means that a small perturbation of the original can result in the image being misclassified. Design of such attacks as well as methods of adversarial training against them are subject of intense research. We re-cast the problem using techniques of Wasserstein distributionally robust optimization (DRO) and obtain novel contributions leveraging recent insights from DRO sensitivity analysis. We consider a set of distributional threat models. Unlike the traditional pointwise attacks, which assume a uniform bound on perturbation of each input data point, distributional threat models allow attackers to perturb inputs in a non-uniform way. We link these more general attacks with questions of out-of-sample performance and Knightian uncertainty. To evaluate the distributional robustness of neural networks, we propose a first-order AA algorithm and its multistep version. Our attack algorithms include Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) as special cases. Furthermore, we provide a new asymptotic estimate of the adversarial accuracy against distributional threat models. The bound is fast to compute and first-order accurate, offering new insights even for the pointwise AA. It also naturally yields out-of-sample performance guarantees. We conduct numerical experiments on CIFAR-10, CIFAR-100, ImageNet datasets using DNNs on RobustBench to illustrate our theoretical results. Our code is available at https://github.com/JanObloj/W-DRO-Adversarial-Methods.</details>

### Weakly Coupled Deep Q-Networks

**Authors:** Ibrahim El Shar, Daniel Jiang

**<details><summary>Abstract**</summary>

We propose weakly coupled deep Q-networks (WCDQN), a novel deep reinforcement learning algorithm that enhances performance in a class of structured problems called weakly coupled Markov decision processes (WCMDP). WCMDPs consist of multiple independent subproblems connected by an action space constraint, which is a structural property that frequently emerges in practice. Despite this appealing structure, WCMDPs quickly become intractable as the number of subproblems grows. WCDQN employs a single network to train multiple DQN ``subagents,'' one for each subproblem, and then combine their solutions to establish an upper bound on the optimal action value. This guides the main DQN agent towards optimality. We show that the tabular version, weakly coupled Q-learning (WCQL), converges almost surely to the optimal action value. Numerical experiments show faster convergence compared to DQN and related techniques in settings with as many as 10 subproblems, $3^{10}$ total actions, and a continuous state space.</details>

### Weakly Supervised 3D Open-vocabulary Segmentation

**Authors:** Kunhao Liu, Fangneng Zhan, Jiahui Zhang, MUYU XU, Yingchen Yu, Abdulmotaleb El Saddik, Christian Theobalt, Eric Xing, Shijian Lu

**<details><summary>Abstract**</summary>

Open-vocabulary segmentation of 3D scenes is a fundamental function of human perception and thus a crucial objective in computer vision research. However, this task is heavily impeded by the lack of large-scale and diverse 3D open-vocabulary segmentation datasets for training robust and generalizable models. Distilling knowledge from pre-trained 2D open-vocabulary segmentation models helps but it compromises the open-vocabulary feature as the 2D models are mostly finetuned with close-vocabulary datasets. We tackle the challenges in 3D open-vocabulary segmentation by exploiting pre-trained foundation models CLIP and DINO in a weakly supervised manner. Specifically, given only the open-vocabulary text descriptions of the objects in a scene, we distill the open-vocabulary multimodal knowledge and object reasoning capability of CLIP and DINO into a neural radiance field (NeRF), which effectively lifts 2D features into view-consistent 3D segmentation. A notable aspect of our approach is that it does not require any manual segmentation annotations for either the foundation models or the distillation process. Extensive experiments show that our method even outperforms fully supervised models trained with segmentation annotations in certain scenes, suggesting that 3D open-vocabulary segmentation can be effectively learned from 2D images and text-image pairs. Code is available at https://github.com/Kunhao-Liu/3D-OVS.</details>

### Weakly-Supervised Concealed Object Segmentation with SAM-based Pseudo Labeling and Multi-scale Feature Grouping

**Authors:** Chunming He, Kai Li, Yachao Zhang, Guoxia Xu, Longxiang Tang, Yulun Zhang, Zhenhua Guo, Xiu Li

**<details><summary>Abstract**</summary>

Weakly-Supervised Concealed Object Segmentation (WSCOS) aims to  segment objects well blended with surrounding environments using sparsely-annotated data for model training. It remains a challenging task since (1) it is hard to distinguish concealed objects from the background due to the intrinsic similarity and (2) the sparsely-annotated training data only provide weak supervision for model learning. In this paper, we propose a new WSCOS method to address these two challenges. To tackle the intrinsic similarity challenge, we design a multi-scale feature grouping module that first groups features at different granularities and then aggregates these grouping results. By grouping similar features together, it encourages segmentation coherence, helping obtain complete segmentation results for both single and multiple-object images. For the weak supervision challenge, we utilize the recently-proposed vision foundation model, ``Segment Anything Model (SAM)'', and use the provided sparse annotations as prompts to generate segmentation masks, which are used to train the model. To alleviate the impact of low-quality segmentation masks, we further propose a series of strategies, including multi-augmentation result ensemble, entropy-based pixel-level weighting, and entropy-based image-level selection. These strategies help provide more reliable supervision to train the segmentation model. We verify the effectiveness of our method on various WSCOS tasks, and experiments demonstrate that our method achieves state-of-the-art performance on these tasks.</details>

### What Knowledge Gets Distilled in Knowledge Distillation?

**Authors:** Utkarsh Ojha, Yuheng Li, Anirudh Sundara Rajan, Yingyu Liang, Yong Jae Lee

**<details><summary>Abstract**</summary>

Knowledge distillation aims to transfer useful information from a teacher network to a student network, with the primary goal of improving the student's performance for the task at hand. Over the years, there has a been a deluge of novel techniques and use cases of knowledge distillation. Yet, despite the various improvements, there seems to be a glaring gap in the community's fundamental understanding of the process. Specifically, what is the knowledge that gets distilled in knowledge distillation?  In other words, in what ways does the student become similar to the teacher? Does it start to localize objects in the same way? Does it get fooled by the same adversarial samples? Does its data invariance properties become similar? Our work presents a comprehensive study to try to answer these questions. We show that existing methods can indeed indirectly distill these properties beyond improving task performance. We further study why knowledge distillation might work this way, and show that our findings have practical implications as well.</details>

### What functions can Graph Neural Networks compute on random graphs? The role of Positional Encoding

**Authors:** Nicolas Keriven, Samuel Vaiter

**<details><summary>Abstract**</summary>

We aim to deepen the theoretical understanding of Graph Neural Networks (GNNs) on large graphs, with a focus on their expressive power.Existing analyses relate this notion to the graph isomorphism problem, which is mostly relevant for graphs of small sizes, or studied graph classification or regression tasks, while prediction tasks on \emph{nodes} are far more relevant on large graphs. Recently, several works showed that, on very general random graphs models, GNNs converge to certains functions as the number of nodes grows.In this paper, we provide a more complete and intuitive description of the function space generated by equivariant GNNs for node-tasks, through general notions of convergence that encompass several previous examples. We emphasize the role of input node features, and study the impact of \emph{node Positional Encodings} (PEs), a recent line of work that has been shown to yield state-of-the-art results in practice. Through the study of several examples of PEs on large random graphs, we extend previously known universality results to significantly more general models. Our theoretical results hint at some normalization tricks, which is shown numerically to have a positive impact on GNN generalization on synthetic and real data. Our proofs contain new concentration inequalities of independent interest.</details>

### When Can We Track Significant Preference Shifts in Dueling Bandits?

**Authors:** Joe Suk, Arpit Agarwal

**<details><summary>Abstract**</summary>

The $K$-armed dueling bandits problem, where the feedback is in the form of noisy pairwise preferences, has been widely studied due its applications in information retrieval, recommendation systems, etc. Motivated by concerns that user preferences/tastes can evolve over time, we consider the problem of _dueling bandits with distribution shifts_. Specifically, we study the recent notion of _significant shifts_ (Suk and Kpotufe, 2022), and ask whether one can design an _adaptive_ algorithm for the dueling problem with $O(\sqrt{K\tilde{L}T})$ dynamic regret,where $\tilde{L}$ is the (unknown) number of significant shifts in preferences. We show that the answer to this question depends on the properties of underlying preference distributions. Firstly,  we give an impossibility result that rules out any algorithm with $O(\sqrt{K\tilde{L}T})$ dynamic regret under the well-studied Condorcet and SST classes of preference distributions. Secondly, we show that $\text{SST}\cap \text{STI}$ is the largest amongst popular classes of preference distributions where it is possible to design such an algorithm. Overall, our results provides an almost complete resolution of the above question for the hierarchy of  distribution classes.</details>

### Where2Explore: Few-shot Affordance Learning for Unseen Novel Categories of Articulated Objects

**Authors:** Chuanruo Ning, Ruihai Wu, Haoran Lu, Kaichun Mo, Hao Dong

**<details><summary>Abstract**</summary>

Articulated object manipulation is a fundamental yet challenging task in robotics. Due to significant geometric and semantic variations across object categories, previous manipulation models struggle to generalize to novel categories. Few-shot learning is a promising solution for alleviating this issue by allowing robots to perform a few interactions with unseen objects. However, extant approaches often necessitate costly and inefficient test-time interactions with each unseen instance. Recognizing this limitation, we observe that despite their distinct shapes, different categories often share similar local geometries essential for manipulation, such as pullable handles and graspable edges - a factor typically underutilized in previous few-shot learning works. To harness this commonality, we introduce 'Where2Explore', an affordance learning framework that effectively explores novel categories with minimal interactions on a limited number of instances. Our framework explicitly estimates the geometric similarity across different categories, identifying local areas that differ from shapes in the training categories for efficient exploration while concurrently transferring affordance knowledge to similar parts of the objects. Extensive experiments in simulated and real-world environments demonstrate our framework's capacity for efficient few-shot exploration and generalization.</details>

### Why Did This Model Forecast This Future? Information-Theoretic Saliency for Counterfactual Explanations of Probabilistic Regression Models

**Authors:** Chirag Raman, Alec Nonnemaker, Amelia Villegas-Morcillo, Hayley Hung, Marco Loog

**<details><summary>Abstract**</summary>

We propose a post hoc saliency-based explanation framework for counterfactual reasoning in probabilistic multivariate time-series forecasting (regression) settings. Building upon Miller's framework of explanations derived from research in multiple social science disciplines, we establish a conceptual link between counterfactual reasoning and saliency-based explanation techniques. To address the lack of a principled notion of saliency, we leverage a unifying definition of information-theoretic saliency grounded in preattentive human visual cognition and extend it to forecasting settings. Specifically, we obtain a closed-form expression for commonly used density functions to identify which observed timesteps appear salient to an underlying model in making its probabilistic forecasts. We empirically validate our framework in a principled manner using synthetic data to establish ground-truth saliency that is unavailable for real-world data. Finally, using real-world data and forecasting models, we demonstrate how our framework can assist domain experts in forming new data-driven hypotheses about the causal relationships between features in the wild.</details>

### Window-Based Distribution Shift Detection for Deep Neural Networks

**Authors:** Guy Bar Shalom, Yonatan Geifman, Ran El-Yaniv

**<details><summary>Abstract**</summary>

To deploy and operate deep neural models in production, the quality of their predictions, which might be contaminated benignly or manipulated maliciously by input distributional deviations, must be monitored and assessed. Specifically, we study the case of monitoring the healthy operation of a deep neural network (DNN) receiving a stream of data, with the aim of detecting input distributional deviations over which the quality of the network's predictions is potentially damaged. Using selective prediction principles, we propose a distribution deviation detection method for DNNs. The proposed method is derived from a tight coverage generalization bound computed over a sample of instances drawn from the true underlying distribution. Based on this bound, our detector continuously monitors the operation of the network over a test window and fires off an alarm whenever a deviation is detected. Our novel detection method performs on-par or better than the state-of-the-art, while consuming substantially lower computation time (five orders of magnitude reduction) and space complexity. Unlike previous methods, which require at least linear dependence on the size of the source distribution for each detection, rendering them inapplicable to ``Google-Scale'' datasets, our approach eliminates this dependence, making it suitable for real-world applications. Code is available at [https://github.com/BarSGuy/Window-Based-Distribution-Shift-Detection](https://github.com/BarSGuy/Window-Based-Distribution-Shift-Detection).</details>

### Zero-One Laws of Graph Neural Networks

**Authors:** Sam Adam-Day, Iliant, Ismail Ceylan

**<details><summary>Abstract**</summary>

Graph neural networks (GNNs) are the de facto standard deep learning architectures for machine learning on graphs.  This has led to a large body of work analyzing the capabilities and limitations of these models,  particularly pertaining to their representation and extrapolation capacity.  We offer a novel theoretical perspective on the representation and extrapolation capacity of GNNs, by answering the question: how do GNNs behave as the number of graph nodes become very large? Under mild assumptions, we show that when we draw graphs of increasing size from the Erdős–Rényi model, the probability that such graphs are mapped to a particular output by a class of GNN classifiers tends to either zero or one. This class includes the popular graph convolutional network architecture. The result establishes `zero-one laws' for these GNNs, and analogously to other convergence laws,  entails theoretical limitations on their capacity.  We empirically verify our results, observing that the theoretical asymptotic limits are evident already on relatively small graphs.</details>

### Zero-Regret Performative Prediction Under Inequality Constraints

**Authors:** Wenjing YAN, Xuanyu Cao

**<details><summary>Abstract**</summary>

Performative prediction is a recently proposed framework where predictions guide decision-making and hence influence future data distributions. Such performative phenomena are ubiquitous in various areas, such as transportation, finance, public policy, and recommendation systems. To date, work on performative prediction has only focused on unconstrained problems, neglecting the fact that many real-world learning problems are subject to constraints. This paper bridges this gap by studying performative prediction under inequality constraints. Unlike most existing work that provides only performative stable points, we aim to find the optimal solutions. Anticipating performative gradient is a challenging task, due to the agnostic performative effect on data distributions. To address this issue, we first develop a robust primal-dual framework that requires only approximate gradients up to a certain accuracy, yet delivers the same order of performance as the stationary stochastic primal-dual algorithm without performativity. Based on this framework, we then propose an adaptive primal-dual algorithm for location families. Our analysis demonstrates that the proposed adaptive primal-dual algorithm attains $\mathcal{O}(\sqrt{T})$ regret and constraint violations, using only $\sqrt{T} + 2T$ samples, where $T$ is the time horizon. To our best knowledge, this is the first study and analysis on the optimality of the performative prediction problem under inequality constraints. Finally, we validate the effectiveness of our algorithm and theoretical results through numerical simulations.</details>

### Zero-sum Polymatrix Markov Games: Equilibrium Collapse and Efficient Computation of Nash Equilibria

**Authors:** Fivos Kalogiannis, Ioannis Panageas

**<details><summary>Abstract**</summary>

The works of (Daskalakis et al., 2009, 2022; Jin et al., 2022; Deng et al., 2023) indicate that computing Nash equilibria in multi-player Markov games is a computationally hard task.  This fact raises the question of whether or not computational intractability can be circumvented if one focuses on specific classes of Markov games. One such example is two-player zero-sum Markov games, in which efficient ways to compute a Nash equilibrium are known. Inspired by zero-sum polymatrix normal-form games (Cai et al., 2016), we define a class of zero-sum multi-agent Markov games in which there are only pairwise interactions described by a graph that changes per state.For this class of Markov games, we show that an $\epsilon$-approximate Nash equilibrium can be found efficiently. To do so, we generalize the techniques of (Cai et al., 2016), by showing that the set of coarse-correlated equilibria collapses to the set of Nash equilibria. Afterwards, it is possible to use any algorithm in the literature that computes approximate coarse-correlated equilibria Markovian policies to get an approximate Nash equilibrium.</details>

### [Spotlight] ZoomTrack: Target-aware Non-uniform Resizing for Efficient Visual Tracking

**Authors:** Yutong Kou, Jin Gao, Bing Li, Gang Wang, Weiming Hu, Yizheng Wang, Liang Li

**<details><summary>Abstract**</summary>

Recently, the transformer has enabled the speed-oriented trackers to approach state-of-the-art (SOTA) performance with high-speed thanks to the smaller input size or the lighter feature extraction backbone, though they still substantially lag behind their corresponding performance-oriented versions. In this paper, we demonstrate that it is possible to narrow or even close this gap while achieving high tracking speed based on the smaller input size. To this end, we non-uniformly resize the cropped image to have a smaller input size while the resolution of the area where the target is more likely to appear is higher and vice versa. This enables us to solve the dilemma of attending to a larger visual field while retaining more raw information for the target despite a smaller input size. Our formulation for the non-uniform resizing can be efficiently solved through quadratic programming (QP) and naturally integrated into most of the crop-based local trackers. Comprehensive experiments on five challenging datasets based on two kinds of transformer trackers, \ie, OSTrack and TransT, demonstrate consistent improvements over them. In particular, applying our method to the speed-oriented version of OSTrack even outperforms its performance-oriented counterpart by 0.6\% AUC on TNL2K, while running 50\% faster and saving over 55\% MACs. Codes and models are available at https://github.com/Kou-99/ZoomTrack.</details>

### f-Policy Gradients: A General Framework for Goal-Conditioned RL using f-Divergences

**Authors:** Siddhant Agarwal, Ishan Durugkar, Peter Stone, Amy Zhang

**<details><summary>Abstract**</summary>

Goal-Conditioned Reinforcement Learning (RL) problems often have access to sparse rewards where the agent receives a reward signal only when it has achieved the goal, making policy optimization a difficult problem. Several works augment this sparse reward with a learned dense reward function, but this can lead to sub-optimal policies if the reward is misaligned. Moreover, recent works have demonstrated that effective shaping rewards for a particular problem can depend on the underlying learning algorithm. This paper introduces a novel way to encourage exploration called $f$-Policy Gradients, or $f$-PG. $f$-PG minimizes the f-divergence between the agent's state visitation distribution and the goal, which we show can lead to an optimal policy. We derive gradients for various f-divergences to optimize this objective. Our learning paradigm provides dense learning signals for exploration in sparse reward settings. We further introduce an entropy-regularized policy optimization objective, that we call $state$-MaxEnt RL (or $s$-MaxEnt RL) as a special case of our objective. We show that several metric-based shaping rewards like L2 can be used with $s$-MaxEnt RL, providing a common ground to study such metric-based shaping rewards with efficient exploration. We find that $f$-PG has better performance compared to standard policy gradient methods on a challenging gridworld as well as the Point Maze and FetchReach environments. More information on our website https://agarwalsiddhant10.github.io/projects/fpg.html.</details>

### iSCAN: Identifying Causal Mechanism Shifts among Nonlinear Additive Noise Models

**Authors:** Tianyu Chen, Kevin Bello, Bryon Aragam, Pradeep Ravikumar

**<details><summary>Abstract**</summary>

Structural causal models (SCMs) are widely used in various disciplines to represent causal relationships among variables in complex systems.Unfortunately, the underlying causal structure is often unknown, and estimating it from data remains a challenging task. In many situations, however, the end goal is to localize the changes (shifts) in the causal mechanisms between related datasets instead of learning the full causal structure of the individual datasets. Some applications include root cause analysis, analyzing gene regulatory network structure changes between healthy and cancerous individuals, or explaining distribution shifts. This paper focuses on identifying the causal mechanism shifts in two or more related datasets over the same set of variables---*without estimating the entire DAG structure of each SCM*.Prior work under this setting assumed linear models with Gaussian noises; instead, in this work we assume that each SCM belongs to the more general class of *nonlinear* additive noise models (ANMs).A key technical contribution of this work is to show that the Jacobian of the score function for the *mixture distribution* allows for the identification of shifts under general non-parametric functional mechanisms.Once the shifted variables are identified, we leverage recent work to estimate the structural differences, if any, for the shifted variables.Experiments on synthetic and real-world data are provided to showcase the applicability of this approach.Code implementing the proposed method is open-source and publicly available at https://github.com/kevinsbello/iSCAN.</details>

### “Why Not Looking backward?” A Robust Two-Step Method to Automatically Terminate Bayesian Optimization

**Authors:** Shuang Li, Ke Li, Wei Li

**<details><summary>Abstract**</summary>

Bayesian Optimization (BO) is a powerful method for tackling expensive black-box optimization problems. As a sequential model-based optimization strategy, BO iteratively explores promising solutions until a predetermined budget, either iterations or time, is exhausted. The decision on when to terminate BO significantly influences both the quality of solutions and its computational efficiency. In this paper, we propose a simple, yet theoretically grounded, two-step method for automatically terminating BO. Our core concept is to proactively identify if the search is within a convex region by examining previously observed samples. BO is halted once the local regret within this convex region falls below a predetermined threshold. To enhance numerical stability, we propose an approximation method for calculating the termination indicator by solving a bilevel optimization problem. We conduct extensive empirical studies on diverse benchmark problems, including synthetic functions, reinforcement learning, and hyperparameter optimization. Experimental results demonstrate that our proposed method saves up to $\approx 80\%$ computational budget yet is with an order of magnitude smaller performance degradation, comparing against the other peer methods. In addition, our proposed termination method is robust in terms of the setting of its termination criterion.</details>

