## Poster Session 5: Thursday, Dec 14, 08:45 CT

### $H$-Consistency Bounds: Characterization and Extensions

**Authors:** Anqi Mao, Mehryar Mohri, Yutao Zhong

**<details><summary>Abstract**</summary>

A series of recent publications by Awasthi et al. have introduced the key notion of *$H$-consistency bounds* for surrogate loss functions.  These are upper bounds on the zero-one estimation error of any predictor in a hypothesis set, expressed in terms of its surrogate loss estimation error. They are both non-asymptotic and hypothesis set-specific and thus stronger and more informative than Bayes-consistency. However, determining if they hold and deriving these bounds have required a specific proof and analysis for each surrogate loss. Can we derive more general tools and characterizations? This paper provides both a general characterization and an extension of $H$-consistency bounds for multi-class classification. We present new and tight $H$-consistency bounds for both the family of constrained losses and that of comp-sum losses, which covers the familiar cross-entropy, or logistic loss applied to the outputs of a neural network. We further extend our analysis beyond the completeness assumptions adopted in previous studies and cover more realistic bounded hypothesis sets.  Our characterizations are based on error transformations, which are explicitly defined for each formulation. We illustrate the application of our general results through several special examples. A by-product of our analysis is the observation that a recently derived multi-class $H$-consistency bound for cross-entropy reduces to an excess bound and is not significant. Instead, we prove a much stronger and more significant guarantee.</details>

### $SE(3)$  Equivariant Convolution and Transformer in Ray Space

**Authors:** Yinshuang Xu, Jiahui Lei, Kostas Daniilidis

**<details><summary>Abstract**</summary>

3D reconstruction and novel view rendering can greatly benefit from geometric priors when the input views are not sufficient in terms of coverage and inter-view baselines. Deep learning of geometric priors from 2D images requires each image to be represented in a $2D$ canonical frame and the prior to be learned in a given or learned $3D$ canonical frame. In this paper, given only the relative poses of the cameras, we show how to learn priors from multiple views equivariant to coordinate frame transformations by proposing an $SE(3)$-equivariant convolution and transformer in the space of rays in 3D. We model the ray space as a homogeneous space of $SE(3)$ and introduce the $SE(3)$-equivariant convolution in ray space. Depending on the output domain of the convolution, we present convolution-based $SE(3)$-equivariant maps from ray space to ray space and to $\mathbb{R}^3$. Our mathematical framework allows us to go beyond convolution to $SE(3)$-equivariant attention in the ray space. We showcase how to tailor and adapt the equivariant convolution and transformer in the tasks of equivariant $3D$ reconstruction and equivariant neural rendering from multiple views. We demonstrate $SE(3)$-equivariance by obtaining robust results in roto-translated datasets without performing transformation augmentation.</details>

### $p$-value Adjustment for Monotonous, Unbiased, and Fast Clustering Comparison

**Authors:** Kai Klede, Thomas Altstidl, Dario Zanca, Bjoern Eskofier

**<details><summary>Abstract**</summary>

Popular metrics for clustering comparison, like the Adjusted Rand Index and the Adjusted Mutual Information, are type II biased. The Standardized Mutual Information removes this bias but suffers from counterintuitive non-monotonicity and poor computational efficiency. We introduce the $p$-value adjusted Rand Index ($\operatorname{PMI}_2$), the first cluster comparison method that is type II unbiased and provably monotonous. The $\operatorname{PMI}_2$ has fast approximations that outperform the Standardized Mutual information. We demonstrate its unbiased clustering selection, approximation quality, and runtime efficiency on synthetic benchmarks. In experiments on image and social network datasets, we show how the $\operatorname{PMI}_2$ can help practitioners choose better clustering and community detection algorithms.</details>

### (Provable) Adversarial Robustness for Group Equivariant Tasks: Graphs, Point Clouds, Molecules, and More

**Authors:** Jan Schuchardt, Yan Scholten, Stephan Günnemann

**<details><summary>Abstract**</summary>

A machine learning model is traditionally considered robust if its prediction remains (almost) constant under input perturbations with small norm. However, real-world tasks like molecular property prediction or point cloud segmentation have inherent equivariances, such as rotation or permutation equivariance. In such tasks, even perturbations with large norm do not necessarily change an input's semantic content. Furthermore, there are perturbations for which a model's prediction explicitly needs to change. For the first time, we propose a sound notion of adversarial robustness that accounts for task equivariance. We then demonstrate that provable robustness can be achieved by (1) choosing a model that matches the task's equivariances (2) certifying traditional adversarial robustness. Certification methods are, however, unavailable for many models, such as those with continuous equivariances. We close this gap by developing the framework of equivariance-preserving randomized smoothing, which enables architecture-agnostic certification. We additionally derive the first architecture-specific graph edit distance certificates, i.e. sound robustness guarantees for isomorphism equivariant tasks like node classification. Overall, a sound notion of robustness is an important prerequisite for future work at the intersection of robust  and geometric machine learning.</details>

### 2Direction: Theoretically Faster Distributed Training with Bidirectional Communication Compression

**Authors:** Alexander Tyurin, Peter Richtarik

**<details><summary>Abstract**</summary>

We consider distributed convex optimization problems in the regime when the communication between the server and the workers is expensive in both uplink and downlink directions. We develop a new and provably accelerated method, which we call 2Direction, based on fast bidirectional compressed communication and a new bespoke error-feedback mechanism which may be of independent interest. Indeed, we find that the EF and EF21-P mechanisms (Seide et al., 2014; Gruntkowska et al., 2023) that have considerable success in the design of efficient non-accelerated methods are not appropriate for accelerated methods. In particular, we prove that 2Direction improves the previous state-of-the-art communication complexity $\widetilde{\Theta}\left(K \times \left(\frac{L}{\alpha \mu} + \frac{L_{\max} \omega}{n \mu} + \omega\right)\right)$ (Gruntkowska et al., 2023) to $\widetilde{\Theta}(K \times (\sqrt{\frac{L (\omega + 1)}{\alpha \mu}} + \sqrt{\frac{L_{\max} \omega^2}{n \mu}} + \frac{1}{\alpha} + \omega))$ in the $\mu$--strongly-convex setting, where $L$ and $L_{\max}$ are smoothness constants, $n$ is \# of workers, $\omega$ and $\alpha$ are compression errors of the Rand$K$ and Top$K$ sparsifiers (as examples), $K$ is \# of coordinates/bits that the server and workers send to each other. Moreover, our method is the first that improves upon the communication complexity of the vanilla accelerated gradient descent method (AGD). We obtain similar improvements in the general convex regime as well. Finally, our theoretical findings are corroborated by experimental evidence.</details>

### A Causal Framework for Decomposing Spurious Variations

**Authors:** Drago Plecko, Elias Bareinboim

**<details><summary>Abstract**</summary>

One of the fundamental challenges found throughout the data sciences is to explain why things happen in specific ways, or through which mechanisms a certain variable $X$ exerts influences over another variable $Y$. In statistics and machine learning, significant efforts have been put into developing machinery to estimate correlations across variables efficiently. In causal inference, a large body of literature is concerned with the decomposition of causal effects under the rubric of mediation analysis. However, many variations are spurious in nature, including different phenomena throughout the applied sciences. Despite the statistical power to estimate correlations and the identification power to decompose causal effects, there is still little understanding of the properties of spurious associations and how they can be decomposed in terms of the underlying causal mechanisms. In this manuscript, we develop formal tools for decomposing spurious variations in both Markovian and Semi-Markovian models. We prove the first results that allow a non-parametric decomposition of spurious effects and provide sufficient conditions for the identification of such decompositions. The described approach has several applications, ranging from explainable and fair AI to questions in epidemiology and medicine, and we empirically demonstrate its use on a real-world dataset.</details>

### A Computation and Communication Efficient Method for Distributed Nonconvex Problems in the Partial Participation Setting

**Authors:** Alexander Tyurin, Peter Richtarik

**<details><summary>Abstract**</summary>

We present a new method that includes three key components of distributed optimization and federated learning: variance reduction of stochastic gradients, partial participation, and compressed communication. We prove that the new method has optimal oracle complexity and state-of-the-art communication complexity in the partial participation setting. Regardless of the communication compression feature, our method successfully combines variance reduction and partial participation: we get the optimal oracle complexity, never need the participation of all nodes, and do not require the bounded gradients (dissimilarity) assumption.</details>

### A Finite-Sample Analysis of Payoff-Based Independent Learning in Zero-Sum Stochastic Games

**Authors:** Zaiwei Chen, Kaiqing Zhang, Eric Mazumdar, Asuman Ozdaglar, Adam Wierman

**<details><summary>Abstract**</summary>

In this work, we study two-player zero-sum stochastic games and develop a variant of the smoothed best-response learning dynamics that combines independent learning dynamics for matrix games with the minimax value iteration for stochastic games. The resulting learning dynamics are payoff-based, convergent, rational, and symmetric between the two players. Our theoretical results present to the best of our knowledge the first last-iterate finite-sample analysis of such independent learning dynamics. To establish the results, we develop a coupled Lyapunov drift approach to capture the evolution of multiple sets of coupled and stochastic iterates, which might be of independent interest.</details>

### A Fractional Graph Laplacian Approach to Oversmoothing

**Authors:** Sohir Maskey, Raffaele Paolino, Aras Bacho, Gitta Kutyniok

**<details><summary>Abstract**</summary>

Graph neural networks (GNNs) have shown state-of-the-art performances in various applications. However, GNNs often struggle to capture long-range dependencies in graphs due to oversmoothing. In this paper, we generalize the concept of oversmoothing from undirected to directed graphs. To this aim, we extend the notion of Dirichlet energy by considering a directed symmetrically normalized Laplacian. As vanilla graph convolutional networks are prone to oversmooth, we adopt a neural graph ODE framework. Specifically, we propose fractional graph Laplacian neural ODEs, which describe non-local dynamics. We prove that our approach allows propagating information between distant nodes while maintaining a low probability of long-distance jumps. Moreover, we show that our method is more flexible with respect to the convergence of the graph’s Dirichlet energy, thereby mitigating oversmoothing. We conduct extensive experiments on synthetic and real-world graphs, both directed and undirected, demonstrating our method’s versatility across diverse graph homophily levels. Ourcode is available at https://github.com/RPaolino/fLode</details>

### A Framework for Fast and Stable Representations of Multiparameter Persistent Homology Decompositions

**Authors:** David Loiseaux, Mathieu Carrière, Andrew Blumberg

**<details><summary>Abstract**</summary>

Topological data analysis (TDA) is an area of data science that focuses on using invariants from algebraic topology to provide multiscale shape descriptors for geometric data sets such as point clouds. One of the most important such descriptors is persistent homology, which encodes the change in shape as a filtration parameter changes; a typical parameter is the feature scale. For many data sets, it is useful to simultaneously vary multiple filtration parameters, for example feature scale and density. While the theoretical properties of single parameter persistent homology are well understood, less is known about the multiparameter case.  A central question is the problem of representing multiparameter persistent homology by elements of a vector space for integration with standard machine learning algorithms. Existing approaches to this problem either ignore most of the multiparameter information to reduce to the one-parameter case or are heuristic and potentially unstable in the face of noise. In this article, we introduce a new general representation framework that leverages recent results on decompositions of multiparameter persistent homology. This framework is rich in information, fast to compute, and encompasses previous approaches. Moreover, we establish theoretical stability guarantees under this framework as well as efficient algorithms for practical computation, making this framework an applicable and versatile tool for analyzing geometric and point cloud data. We validate our stability results and algorithms with numerical experiments that demonstrate statistical convergence, prediction accuracy, and fast running times on several real data sets.</details>

### A General Framework for Robust G-Invariance in G-Equivariant Networks

**Authors:** Sophia Sanborn, Nina Miolane

**<details><summary>Abstract**</summary>

We introduce a general method for achieving robust group-invariance in group-equivariant convolutional neural networks ($G$-CNNs), which we call the $G$-triple-correlation ($G$-TC) layer. The approach leverages the theory of the triple-correlation on groups, which is the unique, lowest-degree polynomial invariant map that is also \textit{complete}. Many commonly used invariant maps\textemdash such as the \texttt{max}\textemdash are incomplete: they remove both group and signal structure. A complete invariant, by contrast, removes only the variation due to the actions of the group, while preserving all information about the structure of the signal. The completeness of the triple correlation endows the $G$-TC layer with strong robustness, which can be observed in its resistance to invariance-based adversarial attacks. In addition, we observe that it yields measurable improvements in classification accuracy over standard Max $G$-Pooling in $G$-CNN architectures. We provide a general and efficient implementation of the method for any discretized group, which requires only a table defining the group's product structure. We demonstrate the benefits of this method for $G$-CNNs defined on both commutative and non-commutative groups\textemdash $SO(2)$, $O(2)$, $SO(3)$, and $O(3)$ (discretized as the cyclic $C8$, dihedral $D16$, chiral octahedral $O$ and full octahedral $O_h$ groups)\textemdash acting on $\mathbb{R}^2$ and $\mathbb{R}^3$ on both $G$-MNIST and $G$-ModelNet10 datasets.</details>

### A Hierarchical Training Paradigm for Antibody Structure-sequence Co-design

**Authors:** Fang Wu, Stan Z. Li

**<details><summary>Abstract**</summary>

Therapeutic antibodies are an essential and rapidly flourishing drug modality. The binding specificity between antibodies and antigens is decided by complementarity-determining regions (CDRs) at the tips of these Y-shaped proteins. In this paper, we propose a \textbf{h}ierarchical \textbf{t}raining \textbf{p}aradigm (HTP) for the antibody sequence-structure co-design. HTP consists of four levels of training stages, each corresponding to a specific protein modality within a particular protein domain. Through carefully crafted tasks in different stages, HTP seamlessly and effectively integrates geometric graph neural networks (GNNs) with large-scale protein language models to excavate evolutionary information from not only geometric structures but also vast antibody and non-antibody sequence databases, which determines ligand binding pose and strength. Empirical experiments show HTP sets the new state-of-the-art performance in the co-design problem as well as the fix-backbone design. Our research offers a hopeful path to unleash the potential of deep generative architectures and seeks to illuminate the way forward for the antibody sequence and structure co-design challenge.</details>

### A Pseudo-Semantic Loss for Autoregressive Models with Logical Constraints

**Authors:** Kareem Ahmed, Kai-Wei Chang, Guy Van den Broeck

**<details><summary>Abstract**</summary>

Neuro-symbolic AI bridges the gap between purely symbolic and neural approaches to learning. This often requires maximizing the likelihood of a symbolic constraint w.r.t the neural network's output distribution. Such output distributions are typically assumed to be fully-factorized. This limits the applicability of neuro-symbolic learning to the more expressive auto-regressive distributions, e.g., transformers. Under such distributions, computing the likelihood of even simple constraints is #P-hard. Instead of attempting to enforce the constraint on the entire likelihood distribution, we propose to do so on a random, local approximation thereof. More precisely, we approximate the likelihood of the constraint with the pseudolikelihood of the constraint centered around a model sample. Our approach is factorizable, allowing us to reuse solutions to sub-problems---a main tenet for the efficient computation of neuro-symbolic losses. It also provides a local, high fidelity approximation of the likelihood: it exhibits low entropy and KL-divergence around the model sample. We tested our approach on Sudoku and shortest-path prediction cast as auto-regressive generation, and observe that we greatly improve upon the base model's ability to predict logically-consistent outputs. We also tested our approach on the task of detoxifying large language models. We observe that using a simple constraint disallowing a list of toxic words, we are able to steer the model's outputs away from toxic generations, achieving SoTA compared to previous approaches.</details>

### [Oral] A Rigorous Link between Deep Ensembles and (Variational) Bayesian Methods

**Authors:** Veit David Wild, Sahra Ghalebikesabi, Dino Sejdinovic, Jeremias Knoblauch

**Oral Presentation:** Th, Dec 14, 08:15 -- Oral 5C

**<details><summary>Abstract**</summary>

We establish the first mathematically rigorous link between Bayesian, variational Bayesian, and ensemble methods. A key step towards this it to reformulate the non-convex optimisation problem typically encountered in deep learning as a convex optimisation in the space of probability measures. On a technical level, our contribution amounts to studying generalised variational inference through the lense of Wasserstein gradient flows. The result is a unified theory of various seemingly disconnected approaches that are commonly used for uncertainty quantification in deep learning---including deep ensembles and (variational) Bayesian methods. This offers a fresh perspective on the reasons behind the success of deep ensembles over procedures based on parameterised variational inference, and allows the derivation of new ensembling schemes with convergence guarantees. We showcase this by proposing a family of interacting deep ensembles with direct parallels to the interactions of particle systems in thermodynamics, and use our theory to prove the convergence of these algorithms to a well-defined global minimiser on the space of probability measures.</details>

### A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models

**Authors:** Alexander Reisach, Myriam Tami, Christof Seiler, Antoine Chambaz, Sebastian Weichwald

**<details><summary>Abstract**</summary>

Additive Noise Models (ANMs) are a common model class for causal discovery from observational data. Due to a lack of real-world data for which an underlying ANM is known, ANMs with randomly sampled parameters are commonly used to simulate data for the evaluation of causal discovery algorithms. While some parameters may be fixed by explicit assumptions, fully specifying an ANM requires choosing all parameters. Reisach et al. (2021) show that, for many ANM parameter choices, sorting the variables by increasing variance yields an ordering close to a causal order and introduce ‘var-sortability’ to quantify this alignment. Since increasing variances may be unrealistic and cannot be exploited when data scales are arbitrary, ANM data are often rescaled to unit variance in causal discovery benchmarking.We show that synthetic ANM data are characterized by another pattern that is scale-invariant and thus persists even after standardization: the explainable fraction of a variable’s variance, as captured by the coefficient of determination $R^2$, tends to increase along the causal order. The result is high ‘$R^2$-sortability’, meaning that sorting the variables by increasing $R^2$ yields an ordering close to a causal order. We propose a computationally efficient baseline algorithm termed ‘$R^2$-SortnRegress’ that exploits high $R^2$-sortability and that can match and exceed the performance of established causal discovery algorithms. We show analytically that sufficiently high edge weights lead to a relative decrease of the noise contributions along causal chains, resulting in increasingly deterministic relationships and high $R^2$. We characterize $R^2$-sortability on synthetic data with different simulation parameters and find high values in common settings. Our findings reveal high $R^2$-sortability as an assumption about the data generating process relevant to causal discovery and implicit in many ANM sampling schemes. It should be made explicit, as its prevalence in real-world data is an open question. For causal discovery benchmarking, we provide implementations of $R^2$-sortability, the $R^2$-SortnRegress algorithm, and ANM simulation procedures in our library CausalDisco at https://causaldisco.github.io/CausalDisco/.</details>

### A Single 2D Pose with Context is Worth Hundreds for 3D Human Pose Estimation

**Authors:** Qitao Zhao, Ce Zheng, Mengyuan Liu, Chen Chen

**<details><summary>Abstract**</summary>

The dominant paradigm in 3D human pose estimation that lifts a 2D pose sequence to 3D heavily relies on long-term temporal clues (i.e., using a daunting number of video frames) for improved accuracy, which incurs performance saturation, intractable computation and the non-causal problem. This can be attributed to their inherent inability to perceive spatial context as plain 2D joint coordinates carry no visual cues. To address this issue, we propose a straightforward yet powerful solution: leveraging the $	extit{readily available}$ intermediate visual representations produced by off-the-shelf (pre-trained) 2D pose detectors -- no finetuning on the 3D task is even needed. The key observation is that, while the pose detector learns to localize 2D joints, such representations (e.g., feature maps) implicitly encode the joint-centric spatial context thanks to the regional operations in backbone networks. We design a simple baseline named $	extbf{Context-Aware PoseFormer}$ to showcase its effectiveness. $	extit{Without access to any temporal information}$, the proposed method significantly outperforms its context-agnostic counterpart, PoseFormer, and other state-of-the-art methods using up to $	extit{hundreds of}$ video frames regarding both speed and precision. $	extit{Project page:}$ https://qitaozhao.github.io/ContextAware-PoseFormer</details>

### A Tale of Two Features: Stable Diffusion Complements DINO for Zero-Shot Semantic Correspondence

**Authors:** Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, Ming-Hsuan Yang

**<details><summary>Abstract**</summary>

Text-to-image diffusion models have made significant advances in generating and editing high-quality images. As a result, numerous approaches have explored the ability of diffusion model features to understand and process single images for downstream tasks, e.g., classification, semantic segmentation, and stylization. However, significantly less is known about what these features reveal across multiple, different images and objects. In this work, we exploit Stable Diffusion (SD) features for semantic and dense correspondence and discover that with simple post-processing, SD features can perform quantitatively similar to SOTA representations. Interestingly, the qualitative analysis reveals that SD features have very different properties compared to existing representation learning features, such as the recently released DINOv2: while DINOv2 provides sparse but accurate matches, SD features provide high-quality spatial information but sometimes inaccurate semantic matches. We demonstrate that a simple fusion of these two features works surprisingly well, and a zero-shot evaluation using nearest neighbors on these fused features provides a significant performance gain over state-of-the-art methods on benchmark datasets, e.g., SPair-71k, PF-Pascal, and TSS. We also show that these correspondences can enable interesting applications such as instance swapping in two images. Project page: https://sd-complements-dino.github.io/.</details>

### A Theoretical Analysis of the Test Error of Finite-Rank Kernel Ridge Regression

**Authors:** Tin Sum Cheng, Aurelien Lucchi, Anastasis Kratsios, Ivan Dokmanić, David Belius

**<details><summary>Abstract**</summary>

Existing statistical learning guarantees for general kernel regressors often yield loose bounds when used with finite-rank kernels. Yet, finite-rank kernels naturally appear in a number of machine learning problems, e.g. when fine-tuning a pre-trained deep neural network's last layer to adapt it to a novel task when performing transfer learning.  We address this gap for finite-rank kernel ridge regression (KRR) by deriving sharp non-asymptotic upper and lower bounds for the KRR test error of any finite-rank KRR. Our bounds are tighter than previously derived bounds on finite-rank KRR and, unlike comparable results, they also remain valid for any regularization parameters.</details>

### A Trichotomy for Transductive Online Learning

**Authors:** Steve Hanneke, Shay Moran, Jonathan Shafer

**<details><summary>Abstract**</summary>

We present new upper and lower bounds on the number of learner mistakes in the `transductive' online learning setting of Ben-David, Kushilevitz and Mansour (1997).    This setting is similar to standard online learning, except that the adversary fixes a sequence of instances $x_1,\dots,x_n$ to be labeled at the start of the game, and this sequence is known to the learner.    Qualitatively, we prove a \emph{trichotomy}, stating that the minimal number of mistakes made by the learner as $n$ grows can take only one of precisely three possible values: $n$, $\Theta\left(\log (n)\right)$, or $\Theta(1)$.    Furthermore, this behavior is determined by a combination of the VC dimension and the Littlestone dimension.    Quantitatively, we show a variety of bounds relating the number of mistakes to well-known combinatorial dimensions.    In particular, we improve the known lower bound on the constant in the $\Theta(1)$ case from $\Omega\left(\sqrt{\log(d)}\right)$ to $\Omega(\log(d))$ where $d$ is the Littlestone dimension.    Finally, we extend our results to cover multiclass classification and the agnostic setting.</details>

### A Unified Fast Gradient Clipping Framework for DP-SGD

**Authors:** Weiwei Kong, Andres Munoz Medina

**<details><summary>Abstract**</summary>

A well-known numerical bottleneck in the differentially-private stochastic gradient descent (DP-SGD) algorithm is the computation of the gradient norm for each example in a large input batch. When the loss function in DP-SGD is consists of an intermediate linear operation, existing methods in the literature have proposed decompositions of gradients that are amenable to fast norm computations. In this paper, we present a framework that generalizes the above approach to arbitrary (possibly nonlinear) intermediate operations. Moreover, we show that for certain operations, such as fully-connected and embedding layer computations, further improvements to the runtime and storage costs of existing decompositions can be deduced using certain components of our framework. Finally, preliminary numerical experiments are given to demonstrate the substantial effects of the aforementioned improvements.</details>

### A Unified Model and Dimension for Interactive Estimation

**Authors:** Nataly Brukhim, Miro Dudik, Aldo Pacchiano, Robert Schapire

**<details><summary>Abstract**</summary>

We study an abstract framework for interactive learning called interactive estimation in which the goal is to estimate a target from its ``similarity'' to points queried by the learner.We introduce a combinatorial measure called Dissimilarity dimension which largely captures learnability in our model.We present a simple, general, and broadly-applicable algorithm, for which we obtain both regret and PAC generalization bounds that are polynomial in the new dimension. We show that our framework subsumes and thereby unifies two classic learning models:statistical-query learning and structured bandits. We also delineate how the Dissimilarity dimension is related to well-known parameters for both frameworks, in some cases yielding significantly improved analyses.</details>

### A fast heuristic to optimize time-space tradeoff for large models

**Authors:** Akifumi Imanishi, Zijian Xu, Masayuki Takagi, Sixue Wang, Emilio Castillo

**<details><summary>Abstract**</summary>

Training large-scale neural networks is heavily constrained by GPU memory. In order to circumvent this limitation, gradient checkpointing, or recomputation is a powerful technique. There is active research in this area with methods such as Checkmake or Moccasin. However, both Checkmate and Moccasin rely on mixed integer linear programming or constraint programming, resulting in limited scalability due to their exponentially large search space.This paper proposes a novel algorithm for recomputation (FastSA) based on a simulated annealing heuristic that achieves comparable or even better solutions than state-of-the-art alternatives. FastSA can optimize computational graphs with thousands of nodes within 3 to 30 seconds, several orders of magnitude faster than current solutions.We applied FastSA to PyTorch models and verified its effectiveness through popular large vision and text models, including recent language models with the transformer architecture. The results demonstrate significant memory reductions by 73% with extra 18% computational overheads on average. Our experiments demonstrate the practicality and effectiveness of our recomputation algorithm, further highlighting its potential for wide application in various deep learning domains.</details>

### A new perspective on building efficient and expressive 3D equivariant graph neural networks

**Authors:** weitao Du, Yuanqi Du, Limei Wang, Dieqiao Feng, Guifeng Wang, Shuiwang Ji, Carla Gomes, Zhi-Ming Ma

**<details><summary>Abstract**</summary>

Geometric deep learning enables the encoding of physical symmetries in modeling 3D objects. Despite rapid progress in encoding 3D symmetries into Graph Neural Networks (GNNs), a comprehensive evaluation of the expressiveness of these network architectures through a local-to-global analysis lacks today. In this paper, we propose a local hierarchy of 3D isomorphism to evaluate the expressive power of equivariant GNNs and investigate the process of representing global geometric information from local patches. Our work leads to two crucial modules for designing expressive and efficient geometric GNNs; namely local substructure encoding (	extbf{LSE}) and frame transition encoding (	extbf{FTE}). To demonstrate the applicability of our theory, we propose LEFTNet which effectively implements these modules and achieves state-of-the-art performance on both scalar-valued and vector-valued molecular property prediction tasks. We further point out future design space for 3D equivariant graph neural networks. Our codes are available at \url{https://github.com/yuanqidu/LeftNet}</details>

### A polar prediction model for learning to represent visual transformations

**Authors:** Pierre-Étienne Fiquet, Eero Simoncelli

**<details><summary>Abstract**</summary>

All organisms make temporal predictions, and their evolutionary fitness level depends on the accuracy of these predictions. In the context of visual perception, the motions of both the observer and objects in the scene structure the dynamics of sensory signals, allowing for partial prediction of future signals based on past ones. Here, we propose a self-supervised representation-learning framework that extracts and exploits the regularities of natural videos to compute accurate predictions. We motivate the polar architecture by appealing to the Fourier shift theorem and its group-theoretic generalization, and we optimize its parameters on next-frame prediction. Through controlled experiments, we demonstrate that this approach can discover the representation of simple transformation groups acting in data. When trained on natural video datasets, our framework achieves better prediction performance than traditional motion compensation and rivals conventional deep networks, while maintaining interpretability and speed. Furthermore, the polar computations can be restructured into components resembling normalized simple and direction-selective complex cell models of primate V1 neurons. Thus, polar prediction offers a principled framework for understanding how the visual system represents sensory inputs in a form that simplifies temporal prediction.</details>

### AD-PT: Autonomous Driving Pre-Training with Large-scale Point Cloud Dataset

**Authors:** Jiakang Yuan, Bo Zhang, Xiangchao Yan, Botian Shi, Tao Chen, Yikang LI, Yu Qiao

**<details><summary>Abstract**</summary>

It is a long-term vision for Autonomous Driving (AD) community that the perception models can learn from a large-scale point cloud dataset, to obtain unified representations that can achieve promising results on different tasks or benchmarks. Previous works mainly focus on the self-supervised pre-training pipeline, meaning that they perform the pre-training and fine-tuning on the same benchmark, which is difficult to attain the performance scalability and cross-dataset application for the pre-training checkpoint.  In this paper, for the first time, we are committed to building a large-scale pre-training point-cloud dataset with diverse data distribution, and meanwhile learning generalizable representations from such a diverse pre-training dataset. We formulate the point-cloud pre-training task as a semi-supervised problem, which leverages the few-shot labeled and massive unlabeled point-cloud data to generate the unified backbone representations that can be directly applied to many baseline models and benchmarks, decoupling the AD-related pre-training process and downstream fine-tuning task. During the period of backbone pre-training, by enhancing the scene- and instance-level distribution diversity and exploiting the backbone's ability to learn from unknown instances, we achieve significant performance gains on a series of downstream perception benchmarks including Waymo, nuScenes, and KITTI, under different baseline models like PV-RCNN++, SECOND, CenterPoint.</details>

### [Spotlight] AIMS: All-Inclusive Multi-Level Segmentation for Anything

**Authors:** Lu Qi, Jason Kuen, Weidong Guo, Jiuxiang Gu, Zhe Lin, Bo Du, Yu Xu, Ming-Hsuan Yang

**<details><summary>Abstract**</summary>

Despite the progress of image segmentation for accurate visual entity segmentation, completing the diverse requirements of image editing applications for different-level region-of-interest selections remains unsolved. In this paper, we propose a new task, All-Inclusive Multi-Level Segmentation (AIMS), which segments visual regions into three levels: part, entity, and relation (two entities with some semantic relationships). We also build a unified AIMS model through multi-dataset multi-task training to address the two major challenges of annotation inconsistency and task correlation. Specifically, we propose task complementarity, association, and prompt mask encoder for three-level predictions. Extensive experiments demonstrate the effectiveness and generalization capacity of our method compared to other state-of-the-art methods on a single dataset or the concurrent work on segment anything. We will make our code and training model publicly available.</details>

### [Spotlight] AMDP: An Adaptive Detection Procedure for False Discovery Rate Control in High-Dimensional Mediation Analysis

**Authors:** Jiarong Ding, Xuehu ZHU

**<details><summary>Abstract**</summary>

High-dimensional mediation analysis is often associated with a multiple testing problem for detecting significant mediators. Assessing the uncertainty of this detecting process via false discovery rate (FDR) has garnered great interest. To control the FDR in multiple testing, two essential steps are involved: ranking and selection. Existing approaches either construct p-values without calibration or disregard the joint information across tests, leading to conservation in FDR control or non-optimal ranking rules for multiple hypotheses. In this paper, we develop an adaptive mediation detection procedure (referred to as "AMDP") to identify relevant mediators while asymptotically controlling the FDR in high-dimensional mediation analysis. AMDP produces the optimal rule for ranking hypotheses and proposes a data-driven strategy to determine the threshold for mediator selection. This novel method captures information from the proportions of composite null hypotheses and the distribution of p-values, which turns the high dimensionality into an advantage instead of a limitation. The numerical studies on synthetic and real data sets illustrate the performances of AMDP compared with existing approaches.</details>

### ARTIC3D: Learning Robust Articulated 3D Shapes from Noisy Web Image Collections

**Authors:** Chun-Han Yao, Amit Raj, Wei-Chih Hung, Michael Rubinstein, Yuanzhen Li, Ming-Hsuan Yang, Varun Jampani

**<details><summary>Abstract**</summary>

Estimating 3D articulated shapes like animal bodies from monocular images is inherently challenging due to the ambiguities of camera viewpoint, pose, texture, lighting, etc. We propose ARTIC3D, a self-supervised framework to reconstruct per-instance 3D shapes from a sparse image collection in-the-wild. Specifically, ARTIC3D is built upon a skeleton-based surface representation and is further guided by 2D diffusion priors from Stable Diffusion. First, we enhance the input images with occlusions/truncation via 2D diffusion to obtain cleaner mask estimates and semantic features. Second, we perform diffusion-guided 3D optimization to estimate shape and texture that are of high-fidelity and faithful to input images. We also propose a novel technique to calculate more stable image-level gradients via diffusion models compared to existing alternatives. Finally, we produce realistic animations by fine-tuning the rendered shape and texture under rigid part transformations. Extensive evaluations on multiple existing datasets as well as newly introduced noisy web image collections with occlusions and truncation demonstrate that ARTIC3D outputs are more robust to noisy images, higher quality in terms of shape and texture details, and more realistic when animated.</details>

### AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web

**Authors:** Michael Schlichtkrull, Zhijiang Guo, Andreas Vlachos

**<details><summary>Abstract**</summary>

Existing datasets for automated fact-checking have substantial limitations, such as relying on artificial claims, lacking annotations for evidence and intermediate reasoning, or including evidence published after the claim. In this paper we introduce AVeriTeC, a new dataset of 4,568 real-world claims covering fact-checks by 50 different organizations. Each claim is annotated with question-answer pairs supported by evidence available online, as well as textual justifications explaining how the evidence combines to produce a verdict. Through a multi-round annotation process, we avoid common pitfalls including context dependence, evidence insufficiency, and temporal leakage, and reach a substantial inter-annotator agreement of $\kappa=0.619$ on verdicts. We develop a baseline as well as an evaluation scheme for verifying claims through question-answering against the open web.</details>

### Active Observing in Continuous-time Control

**Authors:** Samuel Holt, Alihan Hüyük, Mihaela van der Schaar

**<details><summary>Abstract**</summary>

The control of continuous-time environments while actively deciding when to take costly observations in time is a crucial yet unexplored problem, particularly relevant to real-world scenarios such as medicine, low-power systems, and resource management. Existing approaches either rely on continuous-time control methods that take regular, expensive observations in time or discrete-time control with costly observation methods, which are inapplicable to continuous-time settings due to the compounding discretization errors introduced by time discretization. In this work, we are the first to formalize the continuous-time control problem with costly observations. Our key theoretical contribution shows that observing at regular time intervals is not optimal in certain environments, while irregular observation policies yield higher expected utility.  This perspective paves the way for the development of novel methods that can take irregular observations in continuous-time control with costly observations. We empirically validate our theoretical findings in various continuous-time environments, including a cancer simulation, by constructing a simple initial method to solve this new problem, with a heuristic threshold on the variance of reward rollouts in an offline continuous-time model-based model predictive control (MPC) planner. Although determining the optimal method remains an open problem, our work offers valuable insights and understanding of this unique problem, laying the foundation for future research in this area.</details>

### Active Vision Reinforcement Learning under Limited Visual Observability

**Authors:** Jinghuan Shang, Michael Ryoo

**<details><summary>Abstract**</summary>

In this work, we investigate Active Vision Reinforcement Learning (ActiveVision-RL), where an embodied agent simultaneously learns action policy for the task while also controlling its visual observations in partially observable environments. We denote the former as motor policy and the latter as sensory policy. For example, humans solve real world tasks by hand manipulation (motor policy) together with eye movements (sensory policy). ActiveVision-RL poses challenges on coordinating two policies given their mutual influence. We propose SUGARL, Sensorimotor Understanding Guided Active Reinforcement Learning, a framework that models motor and sensory policies separately, but jointly learns them using with an intrinsic sensorimotor reward. This learnable reward is assigned by sensorimotor reward module, incentivizes the sensory policy to select observations that are optimal to infer its own motor action, inspired by the sensorimotor stage of humans. Through a series of experiments, we show the effectiveness of our method across a range of observability conditions and its adaptability to existed RL algorithms. The sensory policies learned through our method are observed to exhibit effective active vision strategies.</details>

### Activity Grammars for Temporal Action Segmentation

**Authors:** Dayoung Gong, Joonseok Lee, Deunsol Jung, Suha Kwak, Minsu Cho

**<details><summary>Abstract**</summary>

Sequence prediction on temporal data requires the ability to understand compositional structures of multi-level semantics beyond individual and contextual properties of parts. The task of temporal action segmentation remains challenging for the reason, aiming at translating an untrimmed activity video into a sequence of action segments. This paper addresses the problem by introducing an effective activity grammar to guide neural predictions for temporal action segmentation.  We propose a novel grammar induction algorithm, dubbed KARI, that extracts a powerful context-free grammar from action sequence data. We also develop an efficient generalized parser, dubbed BEP, that transforms frame-level probability distributions into a reliable sequence of actions according to the induced grammar with recursive rules. Our approach can be combined with any neural network for temporal action segmentation to enhance the sequence prediction and discover its compositional structure. Experimental results demonstrate that our method significantly improves temporal action segmentation in terms of both performance and interpretability on two standard benchmarks, Breakfast and 50 Salads.</details>

### AdaPlanner: Adaptive Planning from Feedback with Language Models

**Authors:** Haotian Sun, Yuchen Zhuang, Lingkai Kong, Bo Dai, Chao Zhang

**<details><summary>Abstract**</summary>

Large language models (LLMs) have recently demonstrated the potential in acting as autonomous agents for sequential decision-making tasks. However, most existing methods either take actions greedily without planning or rely on static plans that are not adaptable to environmental feedback. Consequently, the sequential decision-making performance of LLM agents degenerates with problem complexity and plan horizons increase. We propose a closed-loop approach, AdaPlanner, which allows the LLM agent to refine its self-generated plan adaptively in response to environmental feedback. In AdaPlanner, the LLM agent adaptively refines its plan from feedback with both in-plan and out-of-plan refinement strategies. To mitigate hallucination, we develop a code-style LLM prompt structure that facilitates plan generation across a variety of tasks, environments, and agent capabilities. Furthermore, we propose a skill discovery mechanism that leverages successful plans as few-shot exemplars, enabling the agent to plan and refine with fewer task demonstrations. Our experiments in the ALFWorld and MiniWoB++ environments demonstrate that AdaPlanner outperforms state-of-the-art baselines by 3.73% and 4.11% while utilizing 2x and 600x fewer samples, respectively. The implementation of AdaPlanner is available at https://github.com/haotiansun14/AdaPlanner.</details>

### AdaVAE: Bayesian Structural Adaptation for Variational Autoencoders

**Authors:** Paribesh Regmi, Rui Li

**<details><summary>Abstract**</summary>

The neural network structures of generative models and their corresponding inference models paired in variational autoencoders (VAEs) play a critical role in the models' generative performance. However, powerful VAE network structures are hand-crafted and fixed prior to training, resulting in a one-size-fits-all approach that requires heavy computation to tune for given data. Moreover, existing VAE regularization methods largely overlook the importance of network structures and fail to prevent overfitting in deep VAE models with cascades of hidden layers. To address these issues, we propose a Bayesian inference framework that automatically adapts VAE network structures to data and prevent overfitting as they grow deeper. We model the number of hidden layers with a beta process to infer the most plausible encoding/decoding network depths warranted by data and perform layer-wise dropout regularization with a conjugate Bernoulli process. We develop a scalable estimator that performs joint inference on both VAE network structures and latent variables. Our experiments show that the inference framework effectively prevents overfitting in both shallow and deep VAE models, yielding state-of-the-art performance. We demonstrate that our framework is compatible with different types of VAE backbone networks and can be applied to various VAE variants, further improving their performance.</details>

### Adaptive Online Replanning with Diffusion Models

**Authors:** Siyuan Zhou, Yilun Du, Shun Zhang, Mengdi Xu, Yikang Shen, Wei Xiao, Dit-Yan Yeung, Chuang Gan

**<details><summary>Abstract**</summary>

Diffusion models have risen a promising approach to data-driven planning, and have demonstrated impressive robotic control, reinforcement learning, and video planning performance. Given an effective planner, an important question to consider is replanning -- when given plans should be regenerated due to both action execution error and external environment changes.  Direct plan execution, without replanning, is problematic as errors from individual actions rapidly accumulate and environments are partially observable and stochastic. Simultaneously, replanning at each timestep incurs a substantial computational cost, and may prevent successful task execution, as different generated plans prevent consistent progress to any particular goal. In this paper, we explore how we may effectively replan with diffusion models. We propose a principled approach to determine when to replan, based on the diffusion model's estimated likelihood of existing generated plans. We further present an approach to replan existing trajectories to ensure that new plans follow the same goal state as the original trajectory, which may efficiently bootstrap off previously generated plans.  We illustrate how a combination of our proposed additions significantly improves the performance of diffusion planners leading to 38\% gains over past diffusion planning approaches on Maze2D and further enables handling of stochastic and long-horizon robotic control tasks.</details>

### Adaptive recurrent vision performs zero-shot computation scaling to unseen difficulty levels

**Authors:** Vijay Veerabadran, Srinivas Ravishankar, Yuan Tang, Ritik Raina, Virginia de Sa

**<details><summary>Abstract**</summary>

Humans solving algorithmic (or) reasoning problems typically exhibit solution times that grow as a function of problem difficulty. Adaptive recurrent neural networks have been shown to exhibit this property for various language-processing tasks. However, little work has been performed to assess whether such adaptive computation can also enable vision models to extrapolate solutions beyond their training distribution's difficulty level, with prior work focusing on very simple tasks. In this study, we investigate a critical functional role of such adaptive processing using recurrent neural networks: to dynamically scale computational resources conditional on input requirements that allow for zero-shot generalization to novel difficulty levels not seen during training using two challenging visual reasoning tasks: PathFinder and Mazes. We combine convolutional recurrent neural networks (ConvRNNs) with a learnable halting mechanism based on Graves (2016). We explore various implementations of such adaptive ConvRNNs (AdRNNs) ranging from tying weights across layers to more sophisticated biologically inspired recurrent networks that possess lateral connections and gating. We show that 1) AdRNNs learn to dynamically halt processing early (or late) to solve easier (or harder) problems, 2) these RNNs zero-shot generalize to more difficult problem settings not shown during training by dynamically increasing the number of recurrent iterations at test time. Our study provides modeling evidence supporting the hypothesis that recurrent processing enables the functional advantage of adaptively allocating compute resources conditional on input requirements and hence allowing generalization to harder difficulty levels of a visual reasoning problem without training.</details>

### [Spotlight] Adversarial Counterfactual Environment Model Learning

**Authors:** Xiong-Hui Chen, Yang Yu, Zhengmao Zhu, ZhiHua Yu, Chen Zhenjun, Chenghe Wang, Yinan Wu, Rong-Jun Qin, Hongqiu Wu, Ruijin Ding, Huang Fangsheng

**<details><summary>Abstract**</summary>

An accurate environment dynamics model is crucial for various downstream tasks, such as counterfactual prediction, off-policy evaluation, and offline reinforcement learning. Currently, these models were learned through empirical risk minimization (ERM) by step-wise fitting of historical transition data.However, we first show that, particularly in the sequential decision-making setting, this approach may catastrophically fail to predict counterfactual action effects due to the selection bias of behavior policies during data collection.To tackle this problem, we introduce a novel model-learning objective called adversarial weighted empirical risk minimization (AWRM).  AWRM incorporates an adversarial policy that exploits the model to generate a data distribution that weakens the model's prediction accuracy, and subsequently, the model is learned under this adversarial data distribution.We implement a practical algorithm, GALILEO, for AWRM and evaluate it on two synthetic tasks, three continuous-control tasks, and  \textit{a real-world application}. The experiments demonstrate that GALILEO can accurately predict counterfactual actions and improve various downstream tasks, including offline policy evaluation and improvement, as well as online decision-making.</details>

### Adversarial Examples Are Not Real Features

**Authors:** Ang Li, Yifei Wang, Yiwen Guo, Yisen Wang

**<details><summary>Abstract**</summary>

The existence of adversarial examples has been a mystery for years and attracted much interest. A well-known theory by \citet{ilyas2019adversarial} explains adversarial vulnerability from a data perspective by showing that one can extract non-robust features from adversarial examples and these features alone are useful for classification. However, the explanation remains quite counter-intuitive since non-robust features are mostly noise features to humans. In this paper, we re-examine the theory from a larger context by incorporating multiple learning paradigms. Notably, we find that contrary to their good usefulness under supervised learning, non-robust features attain poor usefulness when transferred to other self-supervised learning paradigms, such as contrastive learning, masked image modeling, and diffusion models. It reveals that non-robust features are not really as useful as robust or natural features that enjoy good transferability between these paradigms. Meanwhile, for robustness, we also show that naturally trained encoders from robust features are largely non-robust under AutoAttack. Our cross-paradigm examination suggests that the non-robust features are not really useful but more like paradigm-wise shortcuts, and robust features alone might be insufficient to attain reliable model robustness. Code is available at \url{https://github.com/PKU-ML/AdvNotRealFeatures}.</details>

### Adversarial Learning for Feature Shift Detection and Correction

**Authors:** Míriam Barrabés, Daniel Mas Montserrat, Margarita Geleta, Xavier Giró-i-Nieto, Alexander Ioannidis

**<details><summary>Abstract**</summary>

Data shift is a phenomenon present in many real-world applications, and while there are multiple methods attempting to detect shifts, the task of localizing and correcting the features originating such shifts has not been studied in depth. Feature shifts can occur in many datasets, including in multi-sensor data, where some sensors are malfunctioning, or in tabular and structured data, including biomedical, financial, and survey data, where faulty standardization and data processing pipelines can lead to erroneous features. In this work, we explore using the principles of adversarial learning, where the information from several discriminators trained to distinguish between two distributions is used to both detect the corrupted features and fix them in order to remove the distribution shift between datasets. We show that mainstream supervised classifiers, such as random forest or gradient boosting trees, combined with simple iterative heuristics, can localize and correct feature shifts, outperforming current statistical and neural network-based techniques. The code is available at https://github.com/AI-sandbox/DataFix.</details>

### [Spotlight] Adversarial Training from Mean Field Perspective

**Authors:** Soichiro Kumano, Hiroshi Kera, Toshihiko Yamasaki

**<details><summary>Abstract**</summary>

Although adversarial training is known to be effective against adversarial examples, training dynamics are not well understood. In this study, we present the first theoretical analysis of adversarial training in random deep neural networks without any assumptions on data distributions. We introduce a new theoretical framework based on mean field theory, which addresses the limitations of existing mean field-based approaches. Based on the framework, we derive the (empirically tight) upper bounds of $\ell_q$ norm-based adversarial loss with $\ell_p$ norm-based adversarial examples for various values of $p$ and $q$. Moreover, we prove that networks without shortcuts are generally not adversarially trainable and that adversarial training reduces network capacity. We also show that the network width alleviates these issues. Furthermore, the various impacts of input and output dimensions on the upper bounds and time evolution of weight variance are presented.</details>

### Affinity-Aware Graph Networks

**Authors:** Ameya Velingker, Ali Sinop, Ira Ktena, Petar Veličković, Sreenivas Gollapudi

**<details><summary>Abstract**</summary>

Graph Neural Networks (GNNs) have emerged as a powerful technique for learning on relational data. Owing to the relatively limited number of message passing steps they perform—and hence a smaller receptive field—there has been significant interest in improving their expressivity by incorporating structural aspects of the underlying graph. In this paper, we explore the use of affinity measures as features in graph neural networks, in particular measures arising from random walks, including effective resistance, hitting and commute times. We propose message passing networks based on these features and evaluate their performance on a variety of node and graph property prediction tasks. Our architecture has low computational complexity, while our features are invariant to the permutations of the underlying graph. The measures we compute allow the network to exploit the connectivity properties of the graph, thereby allowing us to outperform relevant benchmarks for a wide variety of tasks, often with significantly fewer message passing steps. On one of the largest publicly available graph regression datasets, OGB-LSC-PCQM4Mv1, we obtain the best known single-model validation MAE at the time of writing.</details>

### Aggregating Capacity in FL through Successive Layer Training for Computationally-Constrained Devices

**Authors:** Kilian Pfeiffer, Ramin Khalili, Joerg Henkel

**<details><summary>Abstract**</summary>

Federated learning (FL) is usually performed on resource-constrained edge devices, e.g., with limited memory for the computation. If the required memory to train a model exceeds this limit, the device will be excluded from the training. This can lead to a lower accuracy as valuable data and computation resources are excluded from training, also causing bias and unfairness. The FL training process should be adjusted to such constraints. The state-of-the-art techniques propose training subsets of the FL model at constrained devices, reducing their resource requirements for training. However, these techniques largely limit the co-adaptation among parameters of the model and are highly inefficient, as we show: it is actually better to train a smaller (less accurate) model by the system where all the devices can train the model end-to-end than applying such techniques. We propose a new method that enables successive freezing and training of the parameters of the FL model at devices, reducing the training’s resource requirements at the devices while still allowing enough co-adaptation between parameters. We show through extensive experimental evaluation that our technique greatly improves the accuracy of the trained model (by 52.4 p.p. ) compared with the state of the art, efficiently aggregating the computation capacity available on distributed devices.</details>

### Algorithmic Regularization in Tensor Optimization: Towards a Lifted Approach in Matrix Sensing

**Authors:** Ziye Ma, Javad Lavaei, Somayeh Sojoudi

**<details><summary>Abstract**</summary>

Gradient descent (GD) is crucial for generalization in machine learning models, as it induces implicit regularization, promoting compact representations. In this work, we examine the role of GD in inducing implicit regularization for tensor optimization, particularly within the context of the lifted matrix sensing framework. This framework has been recently proposed to address the non-convex matrix sensing problem by transforming spurious solutions into strict saddles when optimizing over symmetric, rank-1 tensors. We show that, with sufficiently small initialization scale, GD applied to this lifted problem results in approximate rank-1 tensors and critical points with escape directions. Our findings underscore the significance of the tensor parametrization of matrix sensing, in combination with first-order methods, in achieving global optimality in such problems.</details>

### [Spotlight] Alleviating the Semantic Gap for Generalized fMRI-to-Image Reconstruction

**Authors:** Tao Fang, Qian Zheng, Gang Pan

**<details><summary>Abstract**</summary>

Although existing fMRI-to-image reconstruction methods could predict high-quality images, they do not explicitly consider the semantic gap between training and testing data, resulting in reconstruction with unstable and uncertain semantics. This paper addresses the problem of generalized fMRI-to-image reconstruction by explicitly alleviates the semantic gap. Specifically, we leverage the pre-trained CLIP model to map the training data to a compact feature representation, which essentially extends the sparse semantics of training data to dense ones, thus alleviating the semantic gap of the instances nearby known concepts (i.e., inside the training super-classes). Inspired by the robust low-level representation in fMRI data, which could help alleviate the semantic gap for instances that far from the known concepts (i.e., outside the training super-classes), we leverage structural information as a general cue to guide image reconstruction. Further, we quantify the semantic uncertainty based on probability density estimation and achieve Generalized fMRI-to-image reconstruction by adaptively integrating Expanded Semantics and Structural information (GESS) within a diffusion process. Experimental results demonstrate that the proposed GESS model outperforms state-of-the-art methods, and we propose a generalized scenario split strategy to evaluate the advantage of GESS in closing the semantic gap.</details>

### An Adaptive Algorithm for Learning with Unknown Distribution Drift

**Authors:** Alessio Mazzetto, Eli Upfal

**<details><summary>Abstract**</summary>

We develop and analyze a general technique for learning with an unknown distribution drift. Given a sequence of independent observations from the last $T$ steps of a drifting distribution, our algorithm agnostically learns a family of functions with respect to the current distribution at time $T$. Unlike previous work, our technique does not require prior knowledge about the magnitude of the drift. Instead, the algorithm adapts to the sample data. Without explicitly estimating the drift, the algorithm learns a family of functions with almost the same error as a learning algorithm that knows the magnitude of the drift in advance. Furthermore, since our algorithm adapts to the data, it can guarantee a better learning error than an algorithm that relies on loose bounds on the drift. We demonstrate the application of our technique in two fundamental learning scenarios: binary classification and linear regression.</details>

### An active learning framework for multi-group mean estimation

**Authors:** Abdellah Aznag, Rachel Cummings, Adam N. Elmachtoub

**<details><summary>Abstract**</summary>

We consider a fundamental problem where there are multiple groups whose data distributions are unknown, and an analyst would like to learn the mean of each group. We consider an active learning framework to sequentially collect $T$ samples with bandit, each period observing a sample from a chosen group. After observing a sample, the analyst may update their estimate of the mean and variance of that group and choose the next group accordingly.  The objective is to dynamically collect samples to minimize the $p$-norm of the vector of variances of our mean estimators after $T$ rounds. We propose an algorithm, Variance-UCB, that selects groups according to a an upper bound on the variance estimate adjusted to the $p$-norm chosen. We show that the regret of Variance-UCB is $O(T^{-2})$ for finite $p$, and prove that no algorithm can do better. When $p$ is infinite, we recover the $O(T^{-1.5})$ obtained in \cite{activelearning, carpentier2011upper} and provide a new lower bound showing that no algorithm can do better.</details>

### An information-theoretic quantification of the content of communication between brain regions

**Authors:** Marco Celotto, Jan Bím, Alejandro Tlaie, Vito De Feo, Alessandro Toso, Stefan Lemke, Daniel Chicharro, Hamed Nili, Malte Bieler, Ileana Hanganu-Opatz, Tobias Donner, Andrea Brovelli, Stefano Panzeri

**<details><summary>Abstract**</summary>

Quantifying the amount, content and direction of communication between brain regions is key to understanding brain function. Traditional methods to analyze brain activity based on the Wiener-Granger causality principle quantify the overall information propagated by neural activity between simultaneously recorded brain regions, but do not reveal the information flow about specific features of interest (such as sensory stimuli). Here, we develop a new information theoretic measure termed Feature-specific Information Transfer (FIT), quantifying how much information about a specific feature flows between two regions. FIT merges the Wiener-Granger causality principle with information-content specificity. We first derive FIT and prove analytically its key properties. We then illustrate and test them with simulations of neural activity, demonstrating that FIT identifies, within the total information propagated between regions, the information that is transmitted about specific features. We then analyze three neural datasets obtained with different recording methods, magneto- and electro-encephalography, and spiking activity, to demonstrate the ability of FIT to uncover the content and direction of information flow between brain regions beyond what can be discerned with traditional analytical methods. FIT can improve our understanding of how brain regions communicate by uncovering previously unaddressed feature-specific information flow.</details>

### Analyzing the Sample Complexity of Self-Supervised Image Reconstruction Methods

**Authors:** Tobit Klug, Dogukan Atik, Reinhard Heckel

**<details><summary>Abstract**</summary>

Supervised training of deep neural networks on pairs of clean image and noisy measurement achieves state-of-the-art performance for many image reconstruction tasks, but such training pairs are difficult to collect. Self-supervised methods enable training based on noisy measurements only, without clean images. In this work, we investigate the cost of self-supervised training in terms of sample complexity for a class of self-supervised methods that enable the computation of unbiased estimates of gradients of the supervised loss, including noise2noise methods. We analytically show that a model trained with such self-supervised training is as good as the same model trained in a supervised fashion, but self-supervised training requires more examples than supervised training. We then study self-supervised denoising and accelerated MRI empirically and characterize the cost of self-supervised training in terms of the number of additional samples required, and find that the performance gap between self-supervised and supervised training vanishes as a function of the training examples, at a problem-dependent rate, as predicted by our theory.</details>

### Are Diffusion Models Vision-And-Language Reasoners?

**Authors:** Benno Krojer, Elinor Poole-Dayan, Vikram Voleti, Chris Pal, Siva Reddy

**<details><summary>Abstract**</summary>

Text-conditioned image generation models have recently shown immense qualitative success using denoising diffusion processes. However, unlike discriminative vision-and-language models, it is a non-trivial task to subject these diffusion-based generative models to automatic fine-grained quantitative evaluation of high-level phenomena such as compositionality.Towards this goal, we perform two innovations. First, we transform diffusion-based models (in our case, Stable Diffusion) for any image-text matching (ITM) task using a novel method called DiffusionITM.Second, we introduce the Generative-Discriminative Evaluation Benchmark (GDBench) benchmark with 7 complex vision-and-language tasks, bias evaluation and detailed analysis.We find that Stable Diffusion + DiffusionITM is competitive on many tasks and outperforms CLIP on compositional tasks like like CLEVR and Winoground.We further boost its compositional performance with a transfer setup by fine-tuning on MS-COCO while retaining generative capabilities. We also measure the stereotypical bias in diffusion models, and find that Stable Diffusion 2.1 is, for the most part, less biased than Stable Diffusion 1.5.Overall, our results point in an exciting direction bringing discriminative and generative model evaluation closer. We will release code and benchmark setup soon.</details>

### Asynchronous Proportional Response Dynamics: Convergence in Markets with Adversarial Scheduling

**Authors:** Yoav Kolumbus, Menahem Levy, Noam Nisan

**<details><summary>Abstract**</summary>

We study Proportional Response Dynamics (PRD) in linear Fisher markets, where participants act asynchronously. We model this scenario as a sequential process in which at each step, an adversary selects a subset of the players to update their bids, subject to liveness constraints. We show that if every bidder individually applies the PRD update rule whenever they are included in the group of bidders selected by the adversary, then, in the generic case, the entire dynamic converges to a competitive equilibrium of the market. Our proof technique reveals additional properties of linear Fisher markets, such as the uniqueness of the market equilibrium for generic parameters and the convergence of associated no swap regret dynamics and best response dynamics under certain conditions.</details>

### AutoGO: Automated Computation Graph Optimization for Neural Network Evolution

**Authors:** Mohammad Salameh, Keith Mills, Negar Hassanpour, Fred Han, Shuting Zhang, Wei Lu, Shangling Jui, CHUNHUA ZHOU, Fengyu Sun, Di Niu

**<details><summary>Abstract**</summary>

Optimizing Deep Neural Networks (DNNs) to obtain high-quality models for efficient real-world deployment has posed multi-faceted challenges to machine learning engineers. Existing methods either search for neural architectures in heuristic design spaces or apply low-level adjustments to computation primitives to improve inference efficiency on hardware. We present Automated Graph Optimization (AutoGO), a framework to evolve neural networks in a low-level Computation Graph (CG) of primitive operations to improve both its performance and hardware friendliness. Through a tokenization scheme, AutoGO performs variable-sized segment mutations, making both primitive changes and larger-grained changes to CGs. We introduce our segmentation and mutation algorithms, efficient frequent segment mining technique, as well as a pretrained context-aware predictor to estimate the impact of segment replacements. Extensive experimental results show that AutoGO can automatically evolve several typical large convolutional networks to achieve significant task performance improvement and FLOPs reduction on a range of CV tasks, ranging from Classification, Semantic Segmentation, Human Pose Estimation, to Super Resolution, yet without introducing any newer primitive operations. We also demonstrate the lightweight deployment results of AutoGO-optimized super-resolution and denoising U-Nets on a cycle simulator for a Neural Processing Unit (NPU), achieving PSNR improvement and latency/power reduction simultaneously. Code available at https://github.com/Ascend-Research/AutoGO.</details>

### Auxiliary Losses for Learning Generalizable Concept-based Models

**Authors:** Ivaxi Sheth, Samira Ebrahimi Kahou

**<details><summary>Abstract**</summary>

The increasing use of neural networks in various applications has lead to increasing apprehensions, underscoring the necessity to understand their operations beyond mere final predictions. As a solution to enhance model transparency, Concept Bottleneck Models (CBMs) have gained popularity since their introduction. CBMs essentially limit the latent space of a model to human-understandable high-level concepts.  While beneficial, CBMs have been reported to often learn irrelevant concept representations that consecutively damage model performance. To overcome the performance trade-off, we propose a cooperative-Concept Bottleneck Model (coop-CBM). The concept representation of our model is particularly meaningful when fine-grained concept labels are absent. Furthermore, we introduce the concept orthogonal loss (COL) to encourage the separation between the concept representations and to reduce the intra-concept distance. This paper presents extensive experiments on real-world datasets for image classification tasks, namely CUB, AwA2, CelebA and TIL. We also study the performance of coop-CBM models under various distributional shift settings. We show that our proposed method achieves higher accuracy in all distributional shift settings even compared to the black-box models with the highest concept accuracy.</details>

### BIRD: Generalizable Backdoor Detection and Removal for Deep Reinforcement Learning

**Authors:** Xuan Chen, Wenbo Guo, Guanhong Tao, Xiangyu Zhang, Dawn Song

**<details><summary>Abstract**</summary>

Backdoor attacks pose a severe threat to the supply chain management of deep reinforcement learning (DRL) policies. Despite initial defenses proposed in recent studies, these methods have very limited generalizability and scalability. To address this issue, we propose BIRD, a technique to detect and remove backdoors from a pretrained DRL policy in a clean environment without requiring any knowledge about the attack specifications and accessing its training process. By analyzing the unique properties and behaviors of backdoor attacks, we formulate trigger restoration as an optimization problem and design a novel metric to detect backdoored policies. We also design a finetuning method to remove the backdoor, while maintaining the agent's performance in the clean environment. We evaluate BIRD against three backdoor attacks in ten different single-agent or multi-agent environments. Our results verify the effectiveness, efficiency, and generalizability of BIRD, as well as its robustness to different attack variations and adaptions.</details>

### BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing

**Authors:** DONGXU LI, Junnan Li, Steven Hoi

**<details><summary>Abstract**</summary>

Subject-driven text-to-image generation models create novel renditions of an input subject based on text prompts. Existing models suffer from lengthy fine-tuning and difficulties preserving the subject fidelity. To overcome these limitations, we introduce BLIP-Diffusion, a new subject-driven image generation model that supports multimodal control which consumes inputs of subject images and text prompts. Unlike other subject-driven generation models, BLIP-Diffusion introduces a new multimodal encoder which is pre-trained to provide subject representation. We first pre-train the multimodal encoder following BLIP-2 to produce visual representation aligned with the text.Then we design a subject representation learning task which enables a diffusion model to leverage such visual representation and generates new subject renditions. Compared with previous methods such as DreamBooth, our model enables zero-shot subject-driven generation, and efficient fine-tuning for customized subject with up to 20x speedup. We also demonstrate that BLIP-Diffusion can be flexibly combined with existing techniques such as ControlNet and prompt-to-prompt to enable novel subject-driven generation and editing applications. Implementations are available at: https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion.</details>

### Bayesian Active Causal Discovery with Multi-Fidelity Experiments

**Authors:** Zeyu Zhang, Chaozhuo Li, Xu Chen, Xing Xie

**<details><summary>Abstract**</summary>

This paper studies the problem of active causal discovery when the experiments can be done based on multi-fidelity oracles, where higher fidelity experiments are more precise and expensive, while the lower ones are cheaper but less accurate. In this paper, we formally define the task of multi-fidelity active causal discovery, and design a probabilistic model for solving this problem. In specific, we first introduce a mutual-information based acquisition function to determine which variable should be intervened at which fidelity, and then a cascading model is proposed to capture the correlations between different fidelity oracles. Beyond the above basic framework, we also extend it to the batch intervention scenario. We find that the theoretical foundations behind the widely used and efficient greedy method do not hold in our problem. To solve this problem, we introduce a new concept called $\epsilon$-submodular, and design a constraint based fidelity model to theoretically validate the greedy method. We conduct extensive experiments to demonstrate the effectiveness of our model.</details>

### Bayesian nonparametric (non-)renewal processes for analyzing neural spike train variability

**Authors:** David Liu, Mate Lengyel

**<details><summary>Abstract**</summary>

Neural spiking activity is generally variable, non-stationary, and exhibits complex dependencies on covariates, such as sensory input or behavior. These dependencies have been proposed to be signatures of specific computations, and so characterizing them with quantitative rigor is critical for understanding neural computations. Approaches based on point processes provide a principled statistical framework for modeling neural spiking activity. However, currently, they only allow the instantaneous mean, but not the instantaneous variability, of responses to depend on covariates. To resolve this limitation, we propose a scalable Bayesian approach generalizing modulated renewal processes using sparse variational Gaussian processes. We leverage pathwise conditioning for computing nonparametric priors over conditional interspike interval distributions and rely on automatic relevance determination to detect lagging interspike interval dependencies beyond renewal order.  After systematically validating our method on synthetic data, we apply it to two foundational datasets of animal navigation: head direction cells in freely moving mice and hippocampal place cells in rats running along a linear track. Our model exhibits competitive or better predictive power compared to state-of-the-art baselines, and outperforms them in terms of capturing interspike interval statistics. These results confirm the importance of modeling covariate-dependent spiking variability, and further analyses of our fitted models reveal rich patterns of variability modulation beyond the temporal resolution of flexible count-based approaches.</details>

### [Spotlight] Best Arm Identification with Fixed Budget: A Large Deviation Perspective

**Authors:** Po-An Wang, Ruo-Chun Tzeng, Alexandre Proutiere

**<details><summary>Abstract**</summary>

We consider the problem of identifying the best arm in stochastic Multi-Armed Bandits (MABs) using a fixed sampling budget. Characterizing the minimal instance-specific error probability for this problem constitutes one of the important remaining open problems in MABs. When arms are selected using a static sampling strategy, the error probability decays exponentially with the number of samples at a rate that can be explicitly derived via Large Deviation techniques. Analyzing the performance of algorithms with adaptive sampling strategies is however much more challenging. In this paper, we establish a connection between the Large Deviation Principle (LDP) satisfied by the empirical proportions of arm draws and that satisfied by the empirical arm rewards. This connection holds for any adaptive algorithm, and is leveraged (i) to improve error probability upper bounds of some existing algorithms, such as the celebrated SR (Successive Rejects) algorithm \cite{audibert2010best}, and (ii) to devise and analyze new algorithms. In particular, we present CR (Continuous Rejects), a truly adaptive algorithm that can reject arms in {\it any} round based on the observed empirical gaps between the rewards of various arms. Applying our Large Deviation results, we prove that CR enjoys better performance guarantees than existing algorithms, including SR. Extensive numerical experiments confirm this observation.</details>

### Better with Less: A Data-Active Perspective on Pre-Training Graph Neural Networks

**Authors:** Jiarong Xu, Renhong Huang, XIN JIANG, Yuxuan Cao, Carl Yang, Chunping Wang, YANG YANG

**<details><summary>Abstract**</summary>

Pre-training on graph neural networks (GNNs) aims to learn transferable knowledge for downstream tasks with unlabeled data, and it has recently become an active research area. The success of graph pre-training models is often attributed to the massive amount of input data. In this paper, however, we identify the curse of big data phenomenon in graph pre-training: more training data do not necessarily lead to better downstream performance. Motivated by this observation, we propose a better-with-less framework for graph pre-training: fewer, but carefully chosen data are fed into a GNN model to enhance pre-training. The proposed pre-training pipeline is called the data-active graph pre-training (APT) framework, and is composed of a graph selector and a pre-training model. The graph selector chooses the most representative and instructive data points based on the inherent properties of graphs as well as predictive uncertainty. The proposed predictive uncertainty, as feedback from the pre-training model, measures the confidence level of the model in the data. When fed with the chosen data, on the other hand, the pre-training model grasps an initial understanding of the new, unseen data, and at the same time attempts to remember the knowledge learned from previous data. Therefore, the integration and interaction between these two components form a unified framework (APT), in which graph pre-training is performed in a progressive and iterative way. Experiment results show that the proposed APT is able to obtain an efficient pre-training model with fewer training data and better downstream performance.</details>

### Beyond Average Return in Markov Decision Processes

**Authors:** Alexandre Marthe, Aurélien Garivier, Claire Vernade

**<details><summary>Abstract**</summary>

What are the functionals of the reward that can be computed and optimized exactly in Markov Decision Processes?In the finite-horizon, undiscounted setting, Dynamic Programming (DP) can only handle these operations efficiently for certain classes of statistics. We summarize the characterization of these classes for policy evaluation, and give a new answer for the planning problem. Interestingly, we prove that only generalized means can be optimized exactly, even in the more general framework of Distributional Reinforcement Learning (DistRL).DistRL permits, however, to evaluate other functionals approximately. We provide error bounds on the resulting estimators, and discuss the potential of this approach as well as its limitations.These results contribute to advancing the theory of Markov Decision Processes by examining overall characteristics of the return, and particularly risk-conscious strategies.</details>

### Beyond Deep Ensembles: A Large-Scale Evaluation of Bayesian Deep Learning under Distribution Shift

**Authors:** Florian Seligmann, Philipp Becker, Michael Volpp, Gerhard Neumann

**<details><summary>Abstract**</summary>

Bayesian deep learning (BDL) is a promising approach to achieve well-calibrated predictions on distribution-shifted data. Nevertheless, there exists no large-scale survey that evaluates recent SOTA methods on diverse, realistic, and challenging benchmark tasks in a systematic manner. To provide a clear picture of the current state of BDL research, we evaluate modern BDL algorithms on real-world datasets from the WILDS collection containing challenging classification and regression tasks, with a focus on generalization capability and calibration under distribution shift. We compare the algorithms on a wide range of large, convolutional and transformer-based neural network architectures. In particular, we investigate a signed version of the expected calibration error that reveals whether the methods are over- or underconfident, providing further insight into the behavior of the methods. Further, we provide the first systematic evaluation of BDL for fine-tuning large pre-trained models, where training from scratch is prohibitively expensive. Finally, given the recent success of Deep Ensembles, we extend popular single-mode posterior approximations to multiple modes by the use of ensembles.   While we find that ensembling single-mode approximations generally improves the generalization capability and calibration of the models by a significant margin, we also identify a failure mode of ensembles when finetuning large transformer-based language models.  In this setting, variational inference based approaches such as last-layer Bayes By Backprop outperform other methods in terms of accuracy by a large margin, while modern approximate inference algorithms such as SWAG achieve the best calibration.</details>

### Beyond Pretrained Features: Noisy Image Modeling Provides Adversarial Defense

**Authors:** Zunzhi You, Daochang Liu, Bohyung Han, Chang Xu

**<details><summary>Abstract**</summary>

Recent advancements in masked image modeling (MIM) have made it a prevailing framework for self-supervised visual representation learning. The MIM pretrained models, like most deep neural network methods, remain vulnerable to adversarial attacks, limiting their practical application, and this issue has received little research attention. In this paper, we investigate how this powerful self-supervised learning paradigm can provide adversarial robustness to downstream classifiers. During the exploration, we find that noisy image modeling (NIM), a simple variant of MIM that adopts denoising as the pre-text task, reconstructs noisy images surprisingly well despite severe corruption. Motivated by this observation, we propose an adversarial defense method, referred to as De^3, by exploiting the pretrained decoder for denoising. Through De^3, NIM is able to enhance adversarial robustness beyond providing pretrained features. Furthermore, we incorporate a simple modification, sampling the noise scale hyperparameter from random distributions, and enable the defense to achieve a better and tunable trade-off between accuracy and robustness. Experimental results demonstrate that, in terms of adversarial robustness, NIM is superior to MIM thanks to its effective denoising capability. Moreover, the defense provided by NIM achieves performance on par with adversarial training while offering the extra tunability advantage. Source code and models are available at https://github.com/youzunzhi/NIM-AdvDef.</details>

### BiSLS/SPS: Auto-tune Step Sizes for Stable Bi-level Optimization

**Authors:** Chen Fan, Gaspard Choné-Ducasse, Mark Schmidt, Christos Thrampoulidis

**<details><summary>Abstract**</summary>

The popularity of bi-level optimization (BO) in deep learning has spurred a growing interest in studying gradient-based BO algorithms.However, existing algorithms involve two coupled learning rates that can be affected by approximation errors when computing hypergradients, making careful fine-tuning necessary to ensure fast convergence. To alleviate this issue, we investigate the use of recently proposed adaptive step-size methods, namely stochastic line search (SLS) and stochastic Polyak step size (SPS), for computing both the upper and lower-level learning rates. First, we revisit the use of SLS and SPS in single-level optimization without the additional interpolation condition that is typically assumed in prior works. For such settings, we investigate new variants of SLS and SPS that improve upon existing suggestions in the literature and are simpler to implement. Importantly, these two variants can be seen as special instances of general family of methods with an envelope-type step-size. This unified envelope strategy allows for the extension of the algorithms and their convergence guarantees to BO settings. Finally, our extensive experiments demonstrate that the new algorithms, which are available in both SGD and Adam versions, can find large learning rates with minimal tuning and converge faster than corresponding vanilla SGD or Adam BO algorithms that require fine-tuning.</details>

### Bicriteria Approximation Algorithms for the Submodular Cover Problem

**Authors:** Wenjing Chen, Victoria Crawford

**<details><summary>Abstract**</summary>

In this paper, we consider the optimization problem Submodular Cover (SCP), which is to find a minimum cardinality subset of a finite universe $U$ such that the value of a submodular function $f$ is above an input threshold $\tau$. In particular, we consider several variants of SCP including the general case, the case where $f$ is additionally assumed to be monotone, and finally the case where $f$ is a regularized monotone submodular function. Our most significant contributions are that: (i) We propose a scalable algorithm for monotone SCP that achieves nearly the same approximation guarantees as the standard greedy algorithm in significantly faster time; (ii) We are the first to develop an algorithm for general SCP that achieves a solution arbitrarily close to being feasible; and finally (iii) we are the first to develop algorithms for regularized SCP. Our algorithms are then demonstrated to be effective in an extensive experimental section on data summarization and graph cut, two applications of SCP.</details>

### Binarized Spectral Compressive Imaging

**Authors:** Yuanhao Cai, Yuxin Zheng, Jing Lin, Xin Yuan, Yulun Zhang, Haoqian Wang

**<details><summary>Abstract**</summary>

Existing deep learning models for hyperspectral image (HSI) reconstruction achieve good performance but require powerful hardwares with enormous memory and computational resources. Consequently, these methods can hardly be deployed on resource-limited mobile devices. In this paper, we propose a novel method, Binarized Spectral-Redistribution Network (BiSRNet), for efficient and practical HSI restoration from compressed measurement in snapshot compressive imaging (SCI) systems. Firstly, we redesign a compact and easy-to-deploy base model to be binarized. Then we present the basic unit, Binarized Spectral-Redistribution Convolution (BiSR-Conv). BiSR-Conv can adaptively redistribute the HSI representations before binarizing activation and uses a scalable hyperbolic tangent function to closer approximate the Sign function in backpropagation. Based on our BiSR-Conv, we customize four binarized convolutional modules to address the dimension mismatch and propagate full-precision information throughout the whole network. Finally, our BiSRNet is derived by using the proposed techniques to binarize the base model. Comprehensive quantitative and qualitative experiments manifest that our proposed BiSRNet outperforms state-of-the-art binarization algorithms. Code and models are publicly available at https://github.com/caiyuanhao1998/BiSCI</details>

### Black-Box Differential Privacy for Interactive ML

**Authors:** Haim Kaplan, Yishay Mansour, Shay Moran, Kobbi Nissim, Uri Stemmer

**<details><summary>Abstract**</summary>

In this work we revisit an interactive variant of joint differential privacy, recently introduced by Naor et al. [2023], and generalize it towards handling online processes in which existing privacy definitions seem too restrictive. We study basic properties of this definition and demonstrate that it satisfies (suitable variants) of group privacy, composition, and post processing.In order to demonstrate the advantages of this privacy definition compared to traditional forms of differential privacy,we consider the basic setting of online classification. We show that any (possibly non-private) learning rule can be effectively transformed to a private learning rule with only a polynomial overhead in the mistake bound. This demonstrates a stark difference with traditional forms of differential privacy, such as the one studied  by Golowich and Livni [2021], where only a double exponential overhead in the mistake bound is known (via an information theoretic upper bound).</details>

### Block Broyden's Methods for Solving Nonlinear Equations

**Authors:** Chengchang Liu, Cheng Chen, Luo Luo, John C.S. Lui

**<details><summary>Abstract**</summary>

This paper studies quasi-Newton methods for solving nonlinear equations. We propose block variants of both good and bad Broyden's methods, which enjoy explicit local superlinear convergence rates.Our block good Broyden's method has faster condition-number-free convergence rate than existing Broyden's methods because it takes the advantage of multiple rank modification on the Jacobian estimator.On the other hand, our block bad Broyden's method directly estimates the inverse of the Jacobian provably, which reduces the computational cost of the iteration.Our theoretical results provide some new insights on why good Broyden's method outperforms bad Broyden's method in most of the cases.The empirical results also demonstrate the superiority of our methods and validate our theoretical analysis.</details>

### Bootstrapped Training of Score-Conditioned Generator for Offline Design of Biological Sequences

**Authors:** Minsu Kim, Federico Berto, Sungsoo Ahn, Jinkyoo Park

**<details><summary>Abstract**</summary>

We study the problem of optimizing biological sequences, e.g., proteins, DNA, and RNA, to maximize a black-box score function that is only evaluated in an offline dataset. We propose a novel solution, bootstrapped training of score-conditioned generator (BootGen) algorithm. Our algorithm repeats a two-stage process. In the first stage, our algorithm trains the biological sequence generator with rank-based weights to enhance the accuracy of sequence generation based on high scores. The subsequent stage involves bootstrapping, which augments the training dataset with self-generated data labeled by a proxy score function. Our key idea is to align the score-based generation with a proxy score function, which distills the knowledge of the proxy score function to the generator. After training, we aggregate samples from multiple bootstrapped generators and proxies to produce a diverse design. Extensive experiments show that our method outperforms competitive baselines on biological sequential design tasks. We provide reproducible source code: https://github.com/kaist-silab/bootgen.</details>

### Brain Dissection: fMRI-trained Networks Reveal Spatial Selectivity in the Processing of Natural Images

**Authors:** Gabriel Sarch, Michael Tarr, Katerina Fragkiadaki, Leila Wehbe

**<details><summary>Abstract**</summary>

The alignment between deep neural network (DNN) features and cortical responses currently provides the most accurate quantitative explanation for higher visual areas. At the same time, these model features have been critiqued as uninterpretable explanations, trading one black box (the human brain) for another (a neural network). In this paper, we train networks to directly predict, from scratch, brain responses to images from a large-scale dataset of natural scenes (Allen et. al., 2021). We then use "network dissection" (Bau et. al., 2017), an explainable AI technique used for enhancing neural network interpretability by identifying and localizing the most significant features in images for individual units of a trained network, and which has been used to study category selectivity in the human brain (Khosla & Wehbe, 2022). We adapt this approach to create a hypothesis-neutral model that is then used to explore the tuning properties of specific visual regions beyond category selectivity, which we call "brain dissection". We use brain dissection to examine a range of ecologically important, intermediate properties, including depth, surface normals, curvature, and object relations across sub-regions of the parietal, lateral, and ventral visual streams, and scene-selective regions. Our findings reveal distinct preferences in brain regions for interpreting visual scenes, with ventro-lateral areas favoring closer and curvier features, medial and parietal areas opting for more varied and flatter 3D elements, and the parietal region uniquely preferring spatial relations. Scene-selective regions exhibit varied preferences, as the retrosplenial complex prefers distant and outdoor features, while the occipital and parahippocampal place areas favor proximity, verticality, and in the case of the OPA, indoor elements. Such findings show the potential of using explainable AI to uncover spatial feature selectivity across the visual cortex, contributing to a deeper, more fine-grained understanding of the functional characteristics of human visual cortex when viewing natural scenes.</details>

### [Spotlight] Break It Down:  Evidence for Structural Compositionality in Neural Networks

**Authors:** Michael Lepori, Thomas Serre, Ellie Pavlick

**<details><summary>Abstract**</summary>

Though modern neural networks have achieved impressive performance in both vision and language tasks, we know little about the functions that they implement. One possibility is that neural networks implicitly break down complex tasks into subroutines, implement modular solutions to these subroutines, and compose them into an overall solution to a task --- a property we term structural compositionality.  Another possibility is that they may simply learn to match new inputs to learned templates, eliding task decomposition entirely. Here, we leverage model pruning techniques to investigate this question in both vision and language across a variety of architectures, tasks, and pretraining regimens. Our results demonstrate that models oftentimes implement solutions to subroutines via modular subnetworks, which can be ablated while maintaining the functionality of other subnetworks. This suggests that neural networks may be able to learn compositionality, obviating the need for specialized symbolic mechanisms.</details>

### Bypassing the Simulator: Near-Optimal Adversarial Linear Contextual Bandits

**Authors:** Haolin Liu, Chen-Yu Wei, Julian Zimmert

**<details><summary>Abstract**</summary>

We consider the adversarial linear contextual bandit problem, where the loss vectors are selected fully adversarially and the per-round action set (i.e. the context) is drawn from a fixed distribution. Existing methods for this problem either require access to a simulator to generate free i.i.d. contexts, achieve a sub-optimal regret no better than $\tilde{\mathcal{O}}(T^{\frac{5}{6}})$, or are computationally inefficient. We greatly improve these results by achieving a regret of $\tilde{\mathcal{O}}(\sqrt{T})$ without a simulator, while maintaining computational efficiency when the action set in each round is small. In the special case of sleeping bandits with adversarial loss and stochastic arm availability, our result answers affirmatively the open question by [SGV20]  on whether there exists a polynomial-time algorithm with  $poly(d)\sqrt{T}$ regret. Our approach naturally handles the case where the loss is linear up to an additive misspecification error, and our regret shows near-optimal dependence on the magnitude of the error.</details>

### C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models

**Authors:** Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, jiayi lei, Yao Fu, Maosong Sun, Junxian He

**<details><summary>Abstract**</summary>

New NLP benchmarks are urgently needed to align with the rapid development of large language models (LLMs). We present C-Eval, the first comprehensive Chinese evaluation suite designed to assess advanced knowledge and reasoning abilities of foundation models in a Chinese context. C-Eval comprises multiple-choice questions across four difficulty levels: middle school, high school, college, and professional. The questions span 52 diverse disciplines, ranging from humanities to science and engineering. C-Eval is accompanied by C-Eval Hard, a subset of very challenging subjects in C-Eval that requires advanced reasoning abilities to solve. We conduct a comprehensive evaluation of the most advanced LLMs on C-Eval, including both English- and Chinese-oriented models. Results indicate that only GPT-4 could achieve an average accuracy of over 60%, suggesting that there is still significant room for improvement for current LLMs. We anticipate C-Eval will help analyze important strengths and shortcomings of foundation models, and foster their development and growth for Chinese users.</details>

### CARE: Modeling Interacting Dynamics Under Temporal Environmental Variation

**Authors:** Xiao Luo, Haixin Wang, Zijie Huang, Huiyu Jiang, Abhijeet Gangan, Song Jiang, Yizhou Sun

**<details><summary>Abstract**</summary>

Modeling interacting dynamical systems, such as fluid dynamics and intermolecular interactions, is a fundamental research problem for understanding and simulating complex real-world systems. Many of these systems can be naturally represented by dynamic graphs, and graph neural network-based approaches have been proposed and shown promising performance. However, most of these approaches assume the underlying dynamics does not change over time, which is unfortunately untrue. For example, a molecular dynamics can be affected by the environment temperature over the time. In this paper, we take an attempt to provide a probabilistic view for time-varying dynamics and propose a model Context-attended Graph ODE (CARE) for modeling time-varying interacting dynamical systems. In our CARE, we explicitly use a context variable to model time-varying environment and construct an encoder to initialize the context variable from historical trajectories. Furthermore, we employ a neural ODE model to depict the dynamic evolution of the context variable inferred from system states. This context variable is incorporated into a coupled ODE to simultaneously drive the evolution of systems. Comprehensive experiments on four datasets demonstrate the effectiveness of our proposed CARE compared with several state-of-the-art approaches.</details>

### CAST: Cross-Attention in Space and Time for Video Action Recognition

**Authors:** Dongho Lee, Jongseo Lee, Jinwoo Choi

**<details><summary>Abstract**</summary>

Recognizing human actions in videos requires spatial and temporal understanding. Most existing action recognition models lack a balanced spatio-temporal understanding of videos. In this work, we propose a novel two-stream architecture, called Cross-Attention in Space and Time (CAST), that achieves a balanced spatio-temporal understanding of videos using only RGB input. Our proposed bottleneck cross-attention mechanism enables the spatial and temporal expert models to exchange information and make synergistic predictions, leading to improved performance. We validate the proposed method with extensive experiments on public benchmarks with different characteristics: EPIC-Kitchens-100, Something-Something-V2, and Kinetics-400. Our method consistently shows favorable performance across these datasets, while the performance of existing methods fluctuates depending on the dataset characteristics. The code is available at https://github.com/KHU-VLL/CAST.</details>

### CLIP4HOI: Towards Adapting CLIP for Practical Zero-Shot HOI Detection

**Authors:** Yunyao Mao, Jiajun Deng, Wengang Zhou, Li Li, Yao Fang, Houqiang Li

**<details><summary>Abstract**</summary>

Zero-shot Human-Object Interaction (HOI) detection aims to identify both seen and unseen HOI categories. A strong zero-shot HOI detector is supposed to be not only capable of discriminating novel interactions but also robust to positional distribution discrepancy between seen and unseen categories when locating human-object pairs. However, top-performing zero-shot HOI detectors rely on seen and predefined unseen categories to distill knowledge from CLIP and jointly locate human-object pairs without considering the potential positional distribution discrepancy, leading to impaired transferability. In this paper, we introduce CLIP4HOI, a novel framework for zero-shot HOI detection. CLIP4HOI is developed on the vision-language model CLIP and ameliorates the above issues in the following two aspects. First, to avoid the model from overfitting to the joint positional distribution of seen human-object pairs, we seek to tackle the problem of zero-shot HOI detection in a disentangled two-stage paradigm. To be specific, humans and objects are independently identified and all feasible human-object pairs are processed by Human-Object interactor for pairwise proposal generation. Second, to facilitate better transferability, the CLIP model is elaborately adapted into a fine-grained HOI classifier for proposal discrimination, avoiding data-sensitive knowledge distillation. Finally, experiments on prevalent benchmarks show that our CLIP4HOI outperforms previous approaches on both rare and unseen categories, and sets a series of state-of-the-art records under a variety of zero-shot settings.</details>

### CLeAR: Continual Learning on Algorithmic Reasoning for Human-like Intelligence

**Authors:** Bong Gyun Kang, HyunGi Kim, Dahuin Jung, Sungroh Yoon

**<details><summary>Abstract**</summary>

Continual learning (CL) aims to incrementally learn multiple tasks that are presented sequentially. The significance of CL lies not only in the practical importance but also in studying the learning mechanisms of humans who are excellent continual learners. While most research on CL has been done on structured data such as images, there is a lack of research on CL for abstract logical concepts such as counting, sorting, and arithmetic, which humans learn gradually over time in the real world. In this work, for the first time, we introduce novel algorithmic reasoning (AR) methodology for continual tasks of abstract concepts: CLeAR. Our methodology proposes a one-to-many mapping of input distribution to a shared mapping space, which allows the alignment of various tasks of different dimensions and shared semantics. Our tasks of abstract logical concepts, in the form of formal language, can be classified into Chomsky hierarchies based on their difficulty. In this study, we conducted extensive experiments consisting of 15 tasks with various levels of Chomsky hierarchy, ranging from in-hierarchy to inter-hierarchy scenarios. CLeAR not only achieved near zero forgetting but also improved accuracy during following tasks, a phenomenon known as backward transfer, while previous CL methods designed for image classification drastically failed.</details>

### CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders

**Authors:** Anthony Fuller, Koreen Millard, James Green

**<details><summary>Abstract**</summary>

A vital and rapidly growing application, remote sensing offers vast yet sparsely labeled, spatially aligned multimodal data; this makes self-supervised learning algorithms invaluable. We present CROMA: a framework that combines contrastive and reconstruction self-supervised objectives to learn rich unimodal and multimodal representations. Our method separately encodes masked-out multispectral optical and synthetic aperture radar samples—aligned in space and time—and performs cross-modal contrastive learning. Another encoder fuses these sensors, producing joint multimodal encodings that are used to predict the masked patches via a lightweight decoder. We show that these objectives are complementary when leveraged on spatially aligned multimodal data. We also introduce X- and 2D-ALiBi, which spatially biases our cross- and self-attention matrices. These strategies improve representations and allow our models to effectively extrapolate to images up to $17.6\times$ larger at test-time. CROMA outperforms the current SoTA multispectral model, evaluated on: four classification benchmarks—finetuning (avg.$\uparrow$ 1.8%), linear (avg.$\uparrow$ 2.4%) and nonlinear (avg.$\uparrow$ 1.4%) probing, $k$NN classification (avg.$\uparrow$ 3.5%), and $K$-means clustering (avg.$\uparrow$ 8.4%); and three segmentation benchmarks (avg.$\uparrow$ 6.4%). CROMA’s rich, optionally multimodal representations can be widely leveraged across remote sensing applications.</details>

### CWCL: Cross-Modal Transfer with Continuously Weighted Contrastive Loss

**Authors:** Rakshith Sharma Srinivasa, Jaejin Cho, Chouchang Yang, Yashas Malur Saidutta, Ching-Hua Lee, Yilin Shen, Hongxia Jin

**<details><summary>Abstract**</summary>

This paper considers contrastive training for cross-modal 0-shot transfer wherein a pre-trained model in one modality is used for representation learning in another domain using pairwise data. The learnt models in the latter domain can then be used for a diverse set of tasks in a 0-shot way, similar to Contrastive Language-Image Pre-training (CLIP) and Locked-image Tuning (LiT) that have recently gained considerable attention. Classical contrastive training employs sets of positive and negative examples to align similar and repel dissimilar training data samples. However, similarity amongst training examples has a more continuous nature, thus calling for a more `non-binary' treatment. To address this, we propose a new contrastive loss function called Continuously Weighted Contrastive Loss (CWCL) that employs a continuous measure of similarity. With CWCL, we seek to transfer the structure of the embedding space from one modality to another. Owing to the continuous nature of similarity in the proposed loss function, these models outperform existing methods for 0-shot transfer across multiple models, datasets and modalities. By using publicly available datasets, we achieve 5-8% (absolute) improvement over previous state-of-the-art methods in 0-shot image classification and 20-30% (absolute) improvement in 0-shot speech-to-intent classification and keyword classification.</details>

### [Spotlight] Calibrated Stackelberg Games: Learning Optimal Commitments Against Calibrated Agents

**Authors:** Nika Haghtalab, Chara Podimata, Kunhe Yang

**<details><summary>Abstract**</summary>

In this paper, we introduce a generalization of the standard Stackelberg Games (SGs) framework: _Calibrated Stackelberg Games_. In CSGs, a principal repeatedly interacts with an agent who (contrary to standard SGs) does not have direct access to the principal's action but instead best responds to _calibrated forecasts_ about it. CSG is a powerful modeling tool that goes beyond assuming that agents use ad hoc and highly specified algorithms for interacting in strategic settings to infer the principal's actions  and thus more robustly addresses real-life applications that SGs were originally intended to capture. Along with CSGs, we also introduce a stronger notion of calibration, termed _adaptive calibration_, that provides fine-grained any-time calibration guarantees against adversarial sequences. We give a general approach for obtaining adaptive calibration algorithms and specialize them for finite CSGs. In our main technical result, we show that in CSGs, the principal can achieve utility that converges to the optimum Stackelberg value of the game both in _finite_ and _continuous_ settings and that no higher utility is achievable. Two prominent and immediate applications of our results are the settings of learning in Stackelberg Security Games and strategic classification, both against _calibrated_ agents.</details>

### Can Pre-Trained Text-to-Image Models Generate Visual Goals for Reinforcement Learning?

**Authors:** Jialu Gao, Kaizhe Hu, Guowei Xu, Huazhe Xu

**<details><summary>Abstract**</summary>

Pre-trained text-to-image generative models can produce diverse, semantically rich, and realistic images from natural language descriptions. Compared with language, images usually convey information with more details and less ambiguity. In this study, we propose Learning from the Void (LfVoid), a method that leverages the power of pre-trained text-to-image models and advanced image editing techniques to guide robot learning. Given natural language instructions, LfVoid can edit the original observations to obtain goal images, such as "wiping" a stain off a table. Subsequently, LfVoid trains an ensembled goal discriminator on the generated image to provide reward signals for a reinforcement learning agent, guiding it to achieve the goal. The ability of LfVoid to learn with zero in-domain training on expert demonstrations or true goal observations (the void) is attributed to the utilization of knowledge from web-scale generative models. We evaluate LfVoid across three simulated tasks and validate its feasibility in the corresponding real-world scenarios. In addition, we offer insights into the key considerations for the effective integration of visual generative models into robot learning workflows. We posit that our work represents an initial step towards the broader application of pre-trained visual generative models in the robotics field. Our project page: https://lfvoid-rl.github.io/.</details>

### Causal Discovery from Subsampled Time Series with Proxy Variables

**Authors:** Mingzhou Liu, Xinwei Sun, Lingjing Hu, Yizhou Wang

**<details><summary>Abstract**</summary>

Inferring causal structures from time series data is the central interest of many scientific inquiries. A major barrier to such inference is the problem of subsampling, *i.e.*, the frequency of measurement is much lower than that of causal influence. To overcome this problem, numerous methods have been proposed, yet either was limited to the linear case or failed to achieve identifiability. In this paper, we propose a constraint-based algorithm that can identify the entire causal structure from subsampled time series, without any parametric constraint. Our observation is that the challenge of subsampling arises mainly from hidden variables at the unobserved time steps. Meanwhile, every hidden variable has an observed proxy, which is essentially itself at some observable time in the future, benefiting from the temporal structure. Based on these, we can leverage the proxies to remove the bias induced by the hidden variables and hence achieve identifiability. Following this intuition, we propose a proxy-based causal discovery algorithm. Our algorithm is nonparametric and can achieve full causal identification. Theoretical advantages are reflected in synthetic and real-world experiments.</details>

### Causal Effect Identification in Uncertain Causal Networks

**Authors:** Sina Akbari, Fateme Jamshidi, Ehsan Mokhtarian, Matthew Vowels, Jalal Etesami, Negar Kiyavash

**<details><summary>Abstract**</summary>

Causal identification is at the core of the causal inference literature, where complete algorithms have been proposed to identify causal queries of interest. The validity of these algorithms hinges on the restrictive assumption of having access to a correctly specified causal structure. In this work, we study the setting where a probabilistic model of the causal structure is available. Specifically, the edges in a causal graph exist with uncertainties which may, for example, represent degree of belief from domain experts. Alternatively, the uncertainty about an edge may reflect the confidence of a particular statistical test. The question that naturally arises in this setting is: Given such a probabilistic graph and a specific causal effect of interest, what is the subgraph which has the highest plausibility and for which the causal effect is identifiable? We show that answering this question reduces to solving an NP-hard combinatorial optimization problem which we call the edge ID problem. We propose efficient algorithms to approximate this problem and evaluate them against both real-world networks and randomly generated graphs.</details>

### Causal de Finetti: On the Identification of Invariant Causal Structure in Exchangeable Data

**Authors:** Siyuan Guo, Viktor Toth, Bernhard Schölkopf, Ferenc Huszar

**<details><summary>Abstract**</summary>

Constraint-based causal discovery methods leverage conditional independence tests to infer causal relationships in a wide variety of applications. Just as the majority of machine learning methods, existing work focuses on studying $\textit{independent and identically distributed}$ data. However, it is known that even with infinite $i.i.d.\$ data, constraint-based methods can only identify causal structures up to broad Markov equivalence classes, posing a fundamental limitation for causal discovery. In this work, we observe that exchangeable data contains richer conditional independence structure than $i.i.d.\$ data, and show how the richer structure can be leveraged for causal discovery. We first present causal de Finetti theorems, which state that exchangeable distributions with certain non-trivial conditional independences can always be represented as $\textit{independent causal mechanism (ICM)}$ generative processes. We then present our main identifiability theorem, which shows that given data from an ICM generative process, its unique causal structure can be identified through performing conditional independence tests. We finally develop a causal discovery algorithm and demonstrate its applicability to inferring causal relationships from multi-environment data.</details>

### Certifiably Robust Graph Contrastive Learning

**Authors:** Minhua Lin, Teng Xiao, Enyan Dai, Xiang Zhang, Suhang Wang

**<details><summary>Abstract**</summary>

Graph Contrastive Learning (GCL) has emerged as a popular unsupervised graph representation learning method. However, it has been shown that GCL is vulnerable to adversarial attacks on both the graph structure and node attributes. Although empirical approaches have been proposed to enhance the robustness of GCL, the certifiable robustness of GCL is still remain unexplored. In this paper, we develop the first certifiably robust framework in GCL. Specifically, we first propose a unified criteria to evaluate and certify the robustness of GCL. We then introduce a novel technique, RES (Randomized Edgedrop Smoothing), to ensure certifiable robustness for any GCL model, and this certified robustness can be provably preserved in downstream tasks. Furthermore, an effective training method is proposed for robust GCL. Extensive experiments on real-world datasets demonstrate the effectiveness of our proposed method in providing effective certifiable robustness and enhancing the robustness of any GCL model. The source code of RES is available at https://github.com/ventr1c/RES-GCL.</details>

### ChimpACT: A Longitudinal Dataset for Understanding Chimpanzee Behaviors

**Authors:** Xiaoxuan Ma, Stephan Kaufhold, Jiajun Su, Wentao Zhu, Jack Terwilliger, Andres Meza, Yixin Zhu, Federico Rossano, Yizhou Wang

**<details><summary>Abstract**</summary>

Understanding the behavior of non-human primates is crucial for improving animal welfare, modeling social behavior, and gaining insights into distinctively human and phylogenetically shared behaviors. However, the lack of datasets on non-human primate behavior hinders in-depth exploration of primate social interactions, posing challenges to research on our closest living relatives. To address these limitations, we present ChimpACT, a comprehensive dataset for quantifying the longitudinal behavior and social relations of chimpanzees within a social group. Spanning from 2015 to 2018, ChimpACT features videos of a group of over 20 chimpanzees residing at the Leipzig Zoo, Germany, with a particular focus on documenting the developmental trajectory of one young male, Azibo. ChimpACT is both comprehensive and challenging, consisting of 163 videos with a cumulative 160,500 frames, each richly annotated with detection, identification, pose estimation, and fine-grained spatiotemporal behavior labels. We benchmark representative methods of three tracks on ChimpACT: (i) tracking and identification, (ii) pose estimation, and (iii) spatiotemporal action detection of the chimpanzees. Our experiments reveal that ChimpACT offers ample opportunities for both devising new methods and adapting existing ones to solve fundamental computer vision tasks applied to chimpanzee groups, such as detection, pose estimation, and behavior analysis, ultimately deepening our comprehension of communication and sociality in non-human primates.</details>

### Class-Conditional Conformal Prediction with Many Classes

**Authors:** Tiffany Ding, Anastasios Angelopoulos, Stephen Bates, Michael Jordan, Ryan Tibshirani

**<details><summary>Abstract**</summary>

Standard conformal prediction methods provide a marginal coverage guarantee,which means that for a random test point, the conformal prediction set contains the true label with a user-specified probability. In many classificationproblems, we would like to obtain a stronger guarantee--that for test pointsof a specific class, the prediction set contains the true label with thesame user-chosen probability. For the latter goal, existing conformal predictionmethods do not work well when there is a limited amount of labeled data perclass, as is often the case in real applications where the number of classes islarge. We propose a method called clustered conformal prediction thatclusters together classes having "similar" conformal scores and performs conformal prediction at the cluster level. Based on empirical evaluation acrossfour image data sets with many (up to 1000) classes, we find that clusteredconformal typically outperforms existing methods in terms of class-conditionalcoverage and set size metrics.</details>

### Classification of Heavy-tailed Features in High Dimensions: a Superstatistical Approach

**Authors:** Urte Adomaityte, Gabriele Sicuro, Pierpaolo Vivo

**<details><summary>Abstract**</summary>

We characterise the learning of a mixture of two clouds of data points with generic centroids via empirical risk minimisation in the high dimensional regime, under the assumptions of generic convex loss and convex regularisation. Each cloud of data points is obtained via a double-stochastic process, where the sample is obtained from a Gaussian distribution whose variance is itself a random parameter sampled from a scalar distribution $\varrho$. As a result, our analysis covers a large family of data distributions, including the case of power-law-tailed distributions with no covariance, and allows us to test recent ''Gaussian universality'' claims. We study the generalisation performance of the obtained estimator, we analyse the role of regularisation, and we analytically characterise the separability transition.</details>

### [Oral] Clifford Group Equivariant Neural Networks

**Authors:** David Ruhe, Johannes Brandstetter, Patrick Forré

**Oral Presentation:** Th, Dec 14, 08:15 -- Oral 5A

**<details><summary>Abstract**</summary>

We introduce Clifford Group Equivariant Neural Networks: a novel approach for constructing $\mathrm{O}(n)$- and $\mathrm{E}(n)$-equivariant models. We identify and study the *Clifford group*: a subgroup inside the Clifford algebra tailored to achieve several favorable properties. Primarily, the group's action forms an orthogonal automorphism that extends beyond the typical vector space to the entire Clifford algebra while respecting the multivector grading. This leads to several non-equivalent subrepresentations corresponding to the multivector decomposition. Furthermore, we prove that the action respects not just the vector space structure of the Clifford algebra but also its multiplicative structure, i.e., the geometric product. These findings imply that every polynomial in multivectors, including their grade projections, constitutes an equivariant map with respect to the Clifford group, allowing us to parameterize equivariant neural network layers. An advantage worth mentioning is that we obtain expressive layers that can elegantly generalize to inner-product spaces of any dimension. We demonstrate, notably from a single core implementation, state-of-the-art performance on several distinct tasks, including a three-dimensional $n$-body experiment, a four-dimensional Lorentz-equivariant high-energy physics experiment, and a five-dimensional convex hull experiment.</details>

### CluB: Cluster Meets BEV for LiDAR-Based 3D Object Detection

**Authors:** Yingjie Wang, Jiajun Deng, Yuenan Hou, Yao Li, Yu Zhang, Jianmin Ji, Wanli Ouyang, Yanyong Zhang

**<details><summary>Abstract**</summary>

Currently, LiDAR-based 3D detectors are broadly categorized into two groups, namely, BEV-based detectors and cluster-based detectors.BEV-based detectors capture the contextual information from the Bird's Eye View (BEV) and fill their center voxels via feature diffusion with a stack of convolution layers, which, however, weakens the capability of presenting an object with the center point.On the other hand, cluster-based detectors exploit the voting mechanism and aggregate the foreground points into object-centric clusters for further prediction.In this paper, we explore how to effectively combine these two complementary representations into a unified framework.Specifically, we propose a new 3D object detection framework, referred to as CluB, which incorporates an auxiliary cluster-based branch into the BEV-based detector by enriching the object representation at both feature and query levels.Technically, CluB is comprised of two steps.First, we construct a cluster feature diffusion module to establish the association between cluster features and BEV features in a subtle and adaptive fashion. Based on that, an imitation loss is introduced to distill object-centric knowledge from the cluster features to the BEV features.Second, we design a cluster query generation module to leverage the voting centers directly from the cluster branch, thus enriching the diversity of object queries.Meanwhile, a direction loss is employed to encourage a more accurate voting center for each cluster.Extensive experiments are conducted on Waymo and nuScenes datasets, and our CluB achieves state-of-the-art performance on both benchmarks.</details>

### CoPriv: Network/Protocol Co-Optimization for Communication-Efficient Private Inference

**Authors:** Wenxuan Zeng, Meng Li, Haichuan Yang, Wen-jie Lu, Runsheng Wang, Ru Huang

**<details><summary>Abstract**</summary>

Deep neural network (DNN) inference based on secure 2-party computation (2PC) can offer cryptographically-secure privacy protection but suffers from orders of magnitude latency overhead due to enormous communication. Previous works heavily rely on a proxy metric of ReLU counts to approximate the communication overhead and focus on reducing the ReLUs to improve the communication efficiency. However, we observe these works achieve limited communication reduction for state-of-the-art (SOTA) 2PC protocols due to the ignorance of other linear and non-linear operations, which now contribute to the majority of communication. In this work, we present CoPriv, a framework that jointly optimizes the 2PC inference protocol and the DNN architecture. CoPriv features a new 2PC protocol for convolution based on Winograd transformation and develops DNN-aware optimization to significantly reduce the inference communication. CoPriv further develops a 2PC-aware network optimization algorithm that is compatible with the proposed protocol and simultaneously reduces the communication for all the linear and non-linear operations. We compare CoPriv with the SOTA 2PC protocol, CrypTFlow2, and demonstrate 2.1× communication reduction for both ResNet-18 and ResNet-32 on CIFAR-100. We also compare CoPriv with SOTA network optimization methods, including SNL, MetaPruning, etc. CoPriv achieves 9.98× and 3.88× online and total communication reduction with a higher accuracy compare to SNL, respectively. CoPriv also achieves 3.87× online communication reduction with more than 3% higher accuracy compared to MetaPruning.</details>

### [Spotlight] Coherent Soft Imitation Learning

**Authors:** Joe Watson, Sandy Huang, Nicolas Heess

**<details><summary>Abstract**</summary>

Imitation learning methods seek to learn from an expert either through behavioral cloning (BC) for the policy or inverse reinforcement learning (IRL) for the reward.Such methods enable agents to learn complex tasks from humans that are difficult to capture with hand-designed reward functions.Choosing between BC or IRL for imitation depends on the quality and state-action coverage of the demonstrations, as well as additional access to the Markov decision process. Hybrid strategies that combine BC and IRL are rare, as initial policy optimization against inaccurate rewards diminishes the benefit of pretraining the policy with BC.Our work derives an imitation method that captures the strengths of both BC and IRL.In the entropy-regularized (`soft') reinforcement learning setting, we show that the behavioral-cloned policy can be used as both a shaped reward and a critic hypothesis space by inverting the regularized policy update. This coherency facilitates fine-tuning cloned policies using the reward estimate and additional interactions with the environment.This approach conveniently achieves imitation learning through initial behavioral cloning and subsequent refinement via RL with online or offline data sources.The simplicity of the approach enables graceful scaling to high-dimensional and vision-based tasks, with stable learning and minimal hyperparameter tuning, in contrast to adversarial approaches.For the open-source implementation and simulation results, see https://joemwatson.github.io/csil/.</details>

### Collaborative Alignment of NLP Models

**Authors:** Fereshte Khani, Marco Tulio Ribeiro

**<details><summary>Abstract**</summary>

Despite substantial advancements, Natural Language Processing (NLP) models often require post-training adjustments to enforce business rules, rectify undesired behavior, and align with user values.  These adjustments involve operationalizing "concepts"—dictating desired model responses to certain inputs. However, it's difficult for a single entity to enumerate and define all possible concepts, indicating a need for a multi-user, collaborative model alignment framework. Moreover, the exhaustive delineation of a concept is challenging, and an improper approach can create shortcuts or interfere with original data or other concepts.To address these challenges, we introduce CoAlign, a framework that enables multi-user interaction with the model, thereby mitigating individual limitations. CoAlign aids users in operationalizing their concepts using Large Language Models, and relying on the principle that NLP models exhibit simpler behaviors in local regions. Our main insight is learning a \emph{local} model for each concept, and a \emph{global} model to integrate the original data with all concepts.We then steer a large language model to generate instances within concept boundaries where local and global disagree.Our experiments show CoAlign is effective at helping multiple users operationalize concepts and avoid interference for a variety of scenarios, tasks, and models.</details>

### Combating Bilateral Edge Noise for Robust Link Prediction

**Authors:** Zhanke Zhou, Jiangchao Yao, Jiaxu Liu, Xiawei Guo, Quanming Yao, LI He, Liang Wang, Bo Zheng, Bo Han

**<details><summary>Abstract**</summary>

Although link prediction on graphs has achieved great success with the development of graph neural networks (GNNs), the potential robustness under the edge noise is still less investigated. To close this gap, we first conduct an empirical study to disclose that the edge noise bilaterally perturbs both input topology and target label, yielding severe performance degradation and representation collapse. To address this dilemma, we propose an information-theory-guided principle, Robust Graph Information Bottleneck (RGIB), to extract reliable supervision signals and avoid representation collapse. Different from the basic information bottleneck, RGIB further decouples and balances the mutual dependence among graph topology, target labels, and representation, building new learning objectives for robust representation against the bilateral noise. Two instantiations, RGIB-SSL and RGIB-REP, are explored to leverage the merits of different methodologies, i.e., self-supervised learning and data reparameterization, for implicit and explicit data denoising, respectively. Extensive experiments on six datasets and three GNNs with diverse noisy scenarios verify the effectiveness of our RGIB instantiations. The code is publicly available at: https://github.com/tmlr-group/RGIB.</details>

### Combating Representation Learning Disparity with Geometric Harmonization

**Authors:** Zhihan Zhou, Jiangchao Yao, Feng Hong, Ya Zhang, Bo Han, Yanfeng Wang

**<details><summary>Abstract**</summary>

Self-supervised learning (SSL) as an effective paradigm of representation learning has achieved tremendous success on various curated datasets in diverse scenarios. Nevertheless, when facing the long-tailed distribution in real-world applications, it is still hard for existing methods to capture transferable and robust representation. The attribution is that the vanilla SSL methods that pursue the sample-level uniformity easily leads to representation learning disparity, where head classes with the huge sample number dominate the feature regime but tail classes with the small sample number passively collapse. To address this problem, we propose a novel Geometric Harmonization (GH) method to encourage the category-level uniformity in representation learning, which is more benign to the minority and almost does not hurt the majority under long-tailed distribution. Specially, GH measures the population statistics of the embedding space on top of self-supervised learning, and then infer an fine-grained instance-wise calibration to constrain the space expansion of head classes and avoid the passive collapse of tail classes. Our proposal does not alter the setting of SSL and can be easily integrated into existing methods in a low-cost manner. Extensive results on a range of benchmark datasets show the effectiveness of \methodspace with high tolerance to the distribution skewness.</details>

### Complex Query Answering on Eventuality Knowledge Graph with Implicit Logical Constraints

**Authors:** Jiaxin Bai, Xin Liu, Weiqi Wang, Chen Luo, Yangqiu Song

**<details><summary>Abstract**</summary>

Querying knowledge graphs (KGs) using deep learning approaches can naturally leverage the reasoning and generalization ability to learn to infer better answers. Traditional neural complex query answering (CQA) approaches mostly work on entity-centric KGs. However, in the real world, we also need to make logical inferences about events, states, and activities (i.e., eventualities or situations) to push learning systems from System I to System II, as proposed by Yoshua Bengio. Querying logically from an EVentuality-centric KG (EVKG) can naturally provide references to such kind of intuitive and logical inference. Thus, in this paper, we propose a new framework to leverage neural methods to answer complex logical queries based on an EVKG, which can satisfy not only traditional first-order logic constraints but also implicit logical constraints over eventualities concerning their occurrences and orders. For instance, if we know that *Food is bad* happens before *PersonX adds soy sauce*, then *PersonX adds soy sauce* is unlikely to be the cause of *Food is bad* due to implicit temporal constraint. To facilitate consistent reasoning on EVKGs, we propose Complex Eventuality Query Answering (CEQA), a more rigorous definition of CQA that considers the implicit logical constraints governing the temporal order and occurrence of eventualities. In this manner, we propose to leverage theorem provers for constructing benchmark datasets to ensure the answers satisfy implicit logical constraints. We also propose a Memory-Enhanced Query Encoding (MEQE) approach to significantly improve the performance of state-of-the-art neural query encoders on the CEQA task.</details>

### Complex-valued Neurons Can Learn More but Slower than Real-valued Neurons via Gradient Descent

**Authors:** Jin-Hui Wu, Shao-Qun Zhang, Yuan Jiang, Zhi-Hua Zhou

**<details><summary>Abstract**</summary>

Complex-valued neural networks potentially possess better representations and performance than real-valued counterparts when dealing with some complicated tasks such as acoustic analysis, radar image classification, etc. Despite empirical successes, it remains unknown theoretically when and to what extent complex-valued neural networks outperform real-valued ones. We take one step in this direction by comparing the learnability of real-valued neurons and complex-valued neurons via gradient descent. We show that a complex-valued neuron can efficiently learn functions expressed by any one real-valued neuron and any one complex-valued neuron with convergence rate $O(t^{-3})$ and $O(t^{-1})$ where $t$ is the iteration index of gradient descent, respectively, whereas a two-layer real-valued neural network with finite width cannot learn a single non-degenerate complex-valued neuron. We prove that a complex-valued neuron learns a real-valued neuron with rate $\Omega (t^{-3})$, exponentially slower than the $O(\mathrm{e}^{- c t})$ rate of learning one real-valued neuron using a real-valued neuron with a constant $c$. We further verify and extend these results via simulation experiments in more general settings.</details>

### Composable Coresets for Determinant Maximization: Greedy is Almost Optimal

**Authors:** Siddharth Gollapudi, Sepideh Mahabadi, Varun Sivashankar

**<details><summary>Abstract**</summary>

Given a set of $n$ vectors in $\mathbb{R}^d$, the goal of the \emph{determinant maximization} problem is to pick $k$ vectors with the maximum volume. Determinant maximization is the MAP-inference task for determinantal point processes (DPP) and has recently received considerable attention for modeling diversity. As most applications for the problem use large amounts of data, this problem has been studied in the relevant \textit{composable coreset} setting.In particular, [Indyk-Mahabadi-OveisGharan-Rezaei--SODA'20, ICML'19] showed that one can get composable coresets with optimal approximation factor of $\tilde O(k)^k$ for the problem, and that a local search algorithm achieves an almost optimal approximation guarantee of $O(k)^{2k}$.In this work, we show that the widely-used Greedy algorithm also provides composable coresets with an almost optimal approximation factor of $O(k)^{3k}$, which improves over the previously known guarantee of $C^{k^2}$, and supports the prior experimental results showing the practicality of the greedy algorithm as a coreset.Our main result follows by showing a local optimality property for Greedy:swapping a single point from the greedy solution with a vector that was not picked by the greedy algorithm can increase the volume by a factor of at most $(1+\sqrt{k})$. This is tight up to the additive constant $1$. Finally, our experiments show that the local optimality of the greedy algorithm is even lower than the theoretical bound on real data sets.</details>

### Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task

**Authors:** Maya Okawa, Ekdeep S Lubana, Robert Dick, Hidenori Tanaka

**<details><summary>Abstract**</summary>

Modern generative models exhibit unprecedented capabilities to generate extremely realistic data. However, given the inherent compositionality of real world, reliable use of these models in practical applications mandates they exhibit the ability to compose their capabilities, generating and reasoning over entirely novel samples never seen in the training distribution. Prior work demonstrates recent vision diffusion models exhibit intriguing compositional generalization abilities, but also fail rather unpredictably. What are the reasons underlying this behavior? Which concepts does the model generally find difficult to compose to form novel data? To address these questions, we perform a controlled study of compositional generalization in conditional diffusion models in a synthetic setting, varying different attributes of the training data and measuring the model's ability to generate samples out-of-distribution. Our results show that: (i) the compositional structure of the data-generating process governs the order in which capabilities and an ability to compose them emerges; (ii) learning individual concepts impacts performance on compositional tasks, multiplicatively explaining sudden emergence; and (iii) learning and composing capabilities is difficult under correlations. We hope our study inspires further grounded research on understanding capabilities and compositionality in generative models from a data-centric perspective.</details>

### Computing Approximate $\ell_p$ Sensitivities

**Authors:** Swati Padmanabhan, David Woodruff, Richard Zhang

**<details><summary>Abstract**</summary>

Recent works in dimensionality reduction for regression tasks have introduced the notion of sensitivity, an estimate of the importance of a specific datapoint in a dataset, offering provable guarantees on the quality of the approximation after removing low-sensitivity datapoints via subsampling. However, fast algorithms for approximating sensitivities, which we show is equivalent to approximate regression, are known for only the $\ell_2$ setting, in which they are popularly termed leverage scores. In this work, we provide the first efficient algorithms for approximating $\ell_p$ sensitivities and other summary statistics of a given matrix. In particular, for a given $n \times d$ matrix, we compute $\alpha$-approximation to its $\ell_1$ sensitivities at the cost of $n/\alpha$ sensitivity computations. For estimating the total $\ell_p$ sensitivity (i.e. the sum of $\ell_p$ sensitivities), we provide an algorithm based on importance sampling of $\ell_p$ Lewis weights, which computes a constant factor approximation at the cost of roughly $\sqrt{d}$ sensitivity computations, with no polynomial dependence on $n$. Furthermore, we estimate the maximum $\ell_1$ sensitivity up to a $\sqrt{d}$ factor in $O(d)$ sensitivity computations. We also generalize these results to $\ell_p$ norms.  Lastly, we experimentally show that for a wide class of structured matrices in real-world datasets, the total sensitivity can be quickly approximated and is significantly smaller than the theoretical prediction, demonstrating that real-world datasets have on average low intrinsic effective dimensionality.</details>

### Computing Optimal Nash Equilibria in Multiplayer Games

**Authors:** Youzhi Zhang, Bo An, Venkatramanan Subrahmanian

**<details><summary>Abstract**</summary>

Designing efficient algorithms to compute a Nash Equilibrium (NE) in multiplayer games is still an open challenge. In this paper, we focus on computing an NE that optimizes a given objective function. For example, when there is a team of players independently playing against an adversary in a game (e.g., several groups in a forest trying to interdict illegal loggers in green security games), these team members may need to find an NE minimizing the adversary’s utility. Finding an optimal NE in multiplayer games can be formulated as a mixed-integer bilinear program by introducing auxiliary variables to represent bilinear terms, leading to a huge number of bilinear terms, making it hard to solve. To overcome this challenge, we first propose a general framework for this formulation based on a set of correlation plans. We then develop a novel algorithm called CRM based on this framework, which uses correlation plans with their relations to strictly reduce the feasible solution space after the convex relaxation of bilinear terms while minimizing the number of correlation plans to significantly reduce the number of bilinear terms. We show that our techniques can significantly reduce the time complexity and CRM can be several orders of magnitude faster than the state-of-the-art baseline.</details>

### Content-based Unrestricted Adversarial Attack

**Authors:** Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, Wenqiang Zhang

**<details><summary>Abstract**</summary>

Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic, demonstrating their ability to deceive human perception and deep neural networks with stealth and success. However, current works usually sacrifice unrestricted degrees and subjectively select some image content to guarantee the photorealism of unrestricted adversarial examples, which limits its attack performance. To ensure the photorealism of adversarial examples and boost attack performance, we propose a novel unrestricted attack framework called Content-based Unrestricted Adversarial Attack. By leveraging a low-dimensional manifold that represents natural images, we map the images onto the manifold and optimize them along its adversarial direction. Therefore, within this framework, we implement Adversarial Content Attack (ACA) based on Stable Diffusion and can generate high transferable unrestricted adversarial examples with various adversarial contents. Extensive experimentation and visualization demonstrate the efficacy of ACA, particularly in surpassing state-of-the-art attacks by an average of 13.3-50.4\% and 16.8-48.0\% in normally trained models and defense methods, respectively.</details>

### Contextual Gaussian Process Bandits with Neural Networks

**Authors:** Haoting Zhang, Jinghai He, Rhonda Righter, Zuo-Jun Shen, Zeyu Zheng

**<details><summary>Abstract**</summary>

Contextual decision-making problems have witnessed extensive applications in various fields such as online content recommendation, personalized healthcare, and autonomous vehicles, where a core practical challenge is to select a suitable surrogate model for capturing unknown complicated reward functions. It is often the case that both high approximation accuracy and explicit uncertainty quantification are desired. In this work, we propose a neural network-accompanied Gaussian process (NN-AGP) model, which leverages neural networks to approximate the unknown and potentially complicated reward function regarding the contextual variable, and maintains a Gaussian process surrogate model with respect to the decision variable. Our model is shown to outperform existing approaches by offering better approximation accuracy thanks to the use of neural networks and possessing explicit uncertainty quantification from the Gaussian process. We also analyze the maximum information gain of the NN-AGP model and prove the regret bounds for the corresponding algorithms. Moreover, we conduct the experiments on both synthetic and practical problems, illustrating the effectiveness of our approach.</details>

### Continual Learning for Instruction Following from Realtime Feedback

**Authors:** Alane Suhr, Yoav Artzi

**<details><summary>Abstract**</summary>

We propose and deploy an approach to  continually train an instruction-following agent from feedback provided by users during collaborative interactions. During interaction, human users instruct an agent using natural language, and provide realtime binary feedback as they observe the agent following their instructions. We design a contextual bandit learning approach, converting  user feedback to immediate reward. We evaluate through thousands of human-agent interactions, demonstrating 15.4% absolute improvement in instruction execution accuracy over time. We also show our approach is robust to several design variations, and that the feedback signal is roughly equivalent to the learning signal of supervised demonstration data.</details>

### Continuous-time Analysis of Anchor Acceleration

**Authors:** Jaewook Suh, Jisun Park, Ernest Ryu

**<details><summary>Abstract**</summary>

Recently, the anchor acceleration, an acceleration mechanism distinct from Nesterov's, has been discovered for minimax optimization and fixed-point problems, but its mechanism is not understood well, much less so than Nesterov acceleration. In this work, we analyze continuous-time models of anchor acceleration. We provide tight, unified analyses for characterizing the convergence rate as a function of the anchor coefficient $\beta(t)$, thereby providing insight into the anchor acceleration mechanism and its accelerated $\mathcal{O}(1/k^2)$-convergence rate. Finally, we present an adaptive method inspired by the continuous-time analyses and establish its effectiveness through theoretical analyses and experiments.</details>

### Contrast Everything: A Hierarchical Contrastive Framework for Medical Time-Series

**Authors:** Yihe Wang, Yu Han, Haishuai Wang, Xiang Zhang

**<details><summary>Abstract**</summary>

Contrastive representation learning is crucial in medical time series analysis as it alleviates dependency on labor-intensive, domain-specific, and scarce expert annotations. However, existing contrastive learning methods primarily focus on one single data level, which fails to fully exploit the intricate nature of medical time series. To address this issue, we present COMET, an innovative hierarchical framework that leverages data consistencies at all inherent levels in medical time series. Our meticulously designed model systematically captures data consistency from four potential levels: observation, sample, trial, and patient levels. By developing contrastive loss at multiple levels, we can learn effective representations that preserve comprehensive data consistency, maximizing information utilization in a self-supervised manner. We conduct experiments in the challenging patient-independent setting. We compare COMET against six baselines using three diverse datasets, which include ECG signals for myocardial infarction and EEG signals for Alzheimer’s and Parkinson’s diseases. The results demonstrate that COMET consistently outperforms all baselines, particularly in setup with 10% and 1% labeled data fractions across all datasets. These results underscore the significant impact of our framework in advancing contrastive representation learning techniques for medical time series. The source code is available at https://github.com/DL4mHealth/COMET.</details>

### Contrastive Retrospection: honing in on critical steps for rapid learning and generalization in RL

**Authors:** Chen Sun, Wannan Yang, Thomas Jiralerspong, Dane Malenfant, Benjamin Alsbury-Nealy, Yoshua Bengio, Blake Richards

**<details><summary>Abstract**</summary>

In real life, success is often contingent upon multiple critical steps that are distant in time from each other and from the final reward. These critical steps are challenging to identify with traditional reinforcement learning (RL) methods that rely on the Bellman equation for credit assignment. Here, we present a new RL algorithm that uses offline contrastive learning to hone in on these critical steps. This algorithm, which we call Contrastive Retrospection (ConSpec), can be added to any existing RL algorithm. ConSpec learns a set of prototypes for the critical steps in a task by a novel contrastive loss and delivers an intrinsic reward when the current state matches one of the prototypes. The prototypes in ConSpec provide two key benefits for credit assignment: (i) They enable rapid identification of all the critical steps. (ii) They do so in a readily interpretable manner, enabling out-of-distribution generalization when sensory features are altered. Distinct from other contemporary RL approaches to credit assignment, ConSpec takes advantage of the fact that it is easier to retrospectively identify the small set of steps that success is contingent upon (and ignoring other states) than it is to prospectively predict reward at every taken step. ConSpec greatly improves learning in a diverse set of RL tasks. The code is available at the link: https://github.com/sunchipsster1/ConSpec</details>

### Contrastive Training of Complex-Valued Autoencoders for Object Discovery

**Authors:** Aleksandar Stanić, Anand Gopalakrishnan, Kazuki Irie, Jürgen Schmidhuber

**<details><summary>Abstract**</summary>

Current state-of-the-art object-centric models use slots and attention-based routing for binding. However, this class of models has several conceptual limitations: the number of slots is hardwired; all slots have equal capacity; training has high computational cost; there are no object-level relational factors within slots. Synchrony-based models in principle can address these limitations by using complex-valued activations which store binding information in their phase components. However, working examples of such synchrony-based models have been developed only very recently, and are still limited to toy grayscale datasets and simultaneous storage of less than three objects in practice. Here we introduce architectural modifications and a novel contrastive learning method that greatly improve the state-of-the-art synchrony-based model. For the first time, we obtain a class of synchrony-based models capable of discovering objects in an unsupervised manner in multi-object color datasets and simultaneously representing more than three objects.</details>

### Convolutional Visual Prompt for Robust Visual Perception

**Authors:** Yun-Yun Tsai, Chengzhi Mao, Junfeng Yang

**<details><summary>Abstract**</summary>

Vision models are often vulnerable to out-of-distribution (OOD) samples without adapting. While visual prompts offer a lightweight method of input-space adaptation for large-scale vision models, they rely on a high-dimensional additive vector and labeled data. This leads to overfitting when adapting models in a self-supervised test-time setting without labels. We introduce convolutional visual prompts (CVP) for label-free test-time adaptation for robust visual perception. The structured nature of CVP demands fewer trainable parameters, less than 1\% compared to standard visual prompts, combating overfitting. Extensive experiments and analysis on a wide variety of OOD visual perception tasks show that our approach is effective, improving robustness by up to 5.87\% over several large-scale models.</details>

### [Spotlight] Coop: Memory is not a Commodity

**Authors:** Jianhao Zhang, Shihan Ma, Peihong Liu, Jinhui Yuan

**<details><summary>Abstract**</summary>

Tensor rematerialization allows the training of deep neural networks (DNNs) under limited memory budgets by checkpointing the models and recomputing the evicted tensors as needed. However, the existing tensor rematerialization techniques overlook the memory system in deep learning frameworks and implicitly assume that free memory blocks at different addresses are identical. Under this flawed assumption, discontiguous tensors are evicted, among which some are not used to allocate the new tensor. This leads to severe memory fragmentation and increases the cost of potential rematerializations.To address this issue, we propose to evict tensors within a sliding window to ensure all evictions are contiguous and are immediately used. Furthermore, we proposed cheap tensor partitioning and recomputable in-place to further reduce the rematerialization cost by optimizing the tensor allocation.We named our method \name/ as it is a co-optimization of tensor allocation and tensor rematerialization. We evaluated \name/ on eight representative DNNs. The experimental results demonstrate that \name/ achieves up to $2\times$ memory saving and hugely reduces compute overhead, search latency, and memory fragmentation compared to the state-of-the-art baselines.</details>

### CorresNeRF: Image Correspondence Priors for Neural Radiance Fields

**Authors:** Yixing Lao, Xiaogang Xu, zhipeng cai, Xihui Liu, Hengshuang Zhao

**<details><summary>Abstract**</summary>

Neural implicit representations in Neural Radiance Fields (NeRF) have achieved impressive results in novel view synthesis and surface reconstruction tasks. However, their performance suffers under challenging scenarios with sparse input views. We present CorresNeRF, a method to leverage image correspondence priors computed by off-the-shelf methods to supervise the training of NeRF. These correspondence priors are first augmented and filtered with our adaptive algorithm. Then they are injected into the training process by adding loss terms on the reprojection error and depth error of the correspondence points. We evaluate our methods on novel view synthesis and surface reconstruction tasks with density-based and SDF-based neural implicit representations across different datasets. We show that this simple yet effective technique can be applied as a plug-and-play module to improve the performance of NeRF under sparse-view settings across different NeRF variants. Our experiments show that we outperform previous methods in both photometric and geometric metrics. The source code is available at https://github.com/yxlao/corres-nerf.</details>

### [Spotlight] Counterfactual Evaluation of Peer-Review Assignment Policies

**Authors:** Martin Saveski, Steven Jecmen, Nihar Shah, Johan Ugander

**<details><summary>Abstract**</summary>

Peer review assignment algorithms aim to match research papers to suitable expert reviewers, working to maximize the quality of the resulting reviews. A key challenge in designing effective assignment policies is evaluating how changes to the assignment algorithm map to changes in review quality. In this work, we leverage recently proposed policies that introduce randomness in peer-review assignment—in order to mitigate fraud—as a valuable opportunity to evaluate counterfactual assignment policies. Specifically, we exploit how such randomized assignments provide a positive probability of observing the reviews of many assignment policies of interest. To address challenges in applying standard off-policy evaluation methods, such as violations of positivity, we introduce novel methods for partial identification based on monotonicity and Lipschitz smoothness assumptions for the mapping between reviewer-paper covariates and outcomes. We apply our methods to peer-review data from two computer science venues: the TPDP'21 workshop (95 papers and 35 reviewers) and the AAAI'22 conference (8,450 papers and 3,145 reviewers). We consider estimates of (i) the effect on review quality when changing weights in the assignment algorithm, e.g., weighting reviewers' bids vs. textual similarity (between the review's past papers and the submission), and (ii) the "cost of randomization", capturing the difference in expected quality between the perturbed and unperturbed optimal match. We find that placing higher weight on text similarity results in higher review quality and that introducing randomization in the reviewer-paper assignment only marginally reduces the review quality. Our methods for partial identification may be of independent interest, while our off-policy approach can likely find use in evaluating a broad class of algorithmic matching systems.</details>

### Creating Multi-Level Skill Hierarchies in Reinforcement Learning

**Authors:** Joshua B. Evans, Özgür Şimşek

**<details><summary>Abstract**</summary>

What is a useful skill hierarchy for an autonomous agent? We propose an answer based on a graphical representation of how the interaction between an agent and its environment may unfold. Our approach uses modularity maximisation as a central organising principle to expose the structure of the interaction graph at multiple levels of abstraction. The result is a collection of skills that operate at varying time scales, organised into a hierarchy, where skills that operate over longer time scales are composed of skills that operate over shorter time scales. The entire skill hierarchy is generated automatically, with no human input, including the skills themselves (their behaviour, when they can be called, and when they terminate) as well as the dependency structure between them. In a wide range of environments, this approach generates skill hierarchies that are intuitively appealing and that considerably improve the learning performance of the agent.</details>

### Creating a Public Repository for Joining Private Data

**Authors:** James Cook, Milind Shyani, Nina Mishra

**<details><summary>Abstract**</summary>

How can one publish a dataset with sensitive attributes in a way that both preserves privacy and enables joins with other datasets on those same sensitive attributes? This problem arises in many contexts, e.g., a hospital and an airline may want to jointly determine whether people who take long-haul flights are more likely to catch respiratory infections. If they join their data by a common keyed user identifier such as email address, they can determine the answer, though it breaks privacy.  This paper shows how the hospital can generate a private sketch and how the airline can privately join with the hospital's sketch by email address. The proposed solution satisfies pure differential privacy and gives approximate answers to linear queries and optimization problems over those joins. Whereas prior work such as secure function evaluation requires sender/receiver interaction, a distinguishing characteristic of the proposed approach is that it is non-interactive. Consequently, the sketch can be published to a repository for any organization to join with, facilitating data discovery. The accuracy of the method is demonstrated through both theoretical analysis and extensive empirical evidence.</details>

### Cross-Domain Policy Adaptation via Value-Guided Data Filtering

**Authors:** Kang Xu, Chenjia Bai, Xiaoteng Ma, Dong Wang, Bin Zhao, Zhen Wang, Xuelong Li, Wei Li

**<details><summary>Abstract**</summary>

Generalizing policies across different domains with dynamics mismatch poses a significant challenge in reinforcement learning. For example, a robot learns the policy in a simulator, but when it is deployed in the real world, the dynamics of the environment may be different. Given the source and target domain with dynamics mismatch, we consider the online dynamics adaptation problem, in which case the agent can access sufficient source domain data while online interactions with the target domain are limited. Existing research has attempted to solve the problem from the dynamics discrepancy perspective. In this work, we reveal the limitations of these methods and explore the problem from the value difference perspective via a novel insight on the value consistency across domains. Specifically, we present the Value-Guided Data Filtering (VGDF) algorithm, which selectively shares transitions from the source domain based on the proximity of paired value targets across the two domains. Empirical results on various environments with kinematic and morphology shifts demonstrate that our method achieves superior performance compared to prior approaches.</details>

### Cross-links Matter for Link Prediction: Rethinking the Debiased GNN from a Data Perspective

**Authors:** Zihan Luo, Hong Huang, Jianxun Lian, Xiran Song, Xing Xie, Hai Jin

**<details><summary>Abstract**</summary>

Recently, the bias-related issues in GNN-based link prediction have raised widely spread concerns. In this paper, we emphasize the bias on links across different node clusters, which we call cross-links, after considering its significance in both easing information cocoons and preserving graph connectivity. Instead of following the objective-oriented mechanism in prior works with compromised utility, we empirically find that existing GNN models face severe data bias between internal-links (links within the same cluster) and cross-links, and this inspires us to rethink the bias issue on cross-links from a data perspective. Specifically, we design a simple yet effective twin-structure framework, which can be easily applied to most of GNNs to mitigate the bias as well as boost their utility in an end-to-end manner. The basic idea is to generate debiased node embeddings as demonstrations, and fuse them into the embeddings of original GNNs. In particular, we learn debiased node embeddings with the help of augmented supervision signals, and a novel dynamic training strategy is designed to effectively fuse debiased node embeddings with the original node embeddings. Experiments on three datasets with six common GNNs show that our framework can not only alleviate the bias between internal-links and cross-links, but also boost the overall accuracy. Comparisons with other state-of-the-art methods also verify the superiority of our method.</details>

### Curriculum Learning for Graph Neural Networks: Which Edges Should We Learn First

**Authors:** Zheng Zhang, Junxiang Wang, Liang Zhao

**<details><summary>Abstract**</summary>

Graph Neural Networks (GNNs) have achieved great success in representing data with dependencies by recursively propagating and aggregating messages along the edges. However, edges in real-world graphs often have varying degrees of difficulty, and some edges may even be noisy to the downstream tasks. Therefore, existing GNNs may lead to suboptimal learned representations because they usually treat every edge in the graph equally. On the other hand, Curriculum Learning (CL), which mimics the human learning principle of learning data samples in a meaningful order, has been shown to be effective in improving the generalization ability and robustness of representation learners by gradually proceeding from easy to more difficult samples during training. Unfortunately, existing CL strategies are designed for independent data samples and cannot trivially generalize to handle data dependencies. To address these issues, we propose a novel CL strategy to gradually incorporate more edges into training according to their difficulty from easy to hard, where the degree of difficulty is measured by how well the edges are expected given the model training status. We demonstrate the strength of our proposed method in improving the generalization ability and robustness of learned representations through extensive experiments on nine synthetic datasets and nine real-world datasets. The code for our proposed method is available at https://github.com/rollingstonezz/Curriculum_learning_for_GNNs</details>

### D-CIPHER: Discovery of Closed-form Partial Differential Equations

**Authors:** Krzysztof Kacprzyk, Zhaozhi Qian, Mihaela van der Schaar

**<details><summary>Abstract**</summary>

Closed-form differential equations, including partial differential equations and higher-order ordinary differential equations, are one of the most important tools used by scientists to model and better understand natural phenomena. Discovering these equations directly from data is challenging because it requires modeling relationships between various derivatives that are not observed in the data (equation-data mismatch) and it involves searching across a huge space of possible equations. Current approaches make strong assumptions about the form of the equation and thus fail to discover many well-known phenomena. Moreover, many of them resolve the equation-data mismatch by estimating the derivatives, which makes them inadequate for noisy and infrequent observations. To this end, we propose D-CIPHER, which is robust to measurement artifacts and can uncover a new and very general class of differential equations. We further design a novel optimization procedure, CoLLie, to help D-CIPHER search through this class efficiently. Finally, we demonstrate empirically that it can discover many well-known equations that are beyond the capabilities of current methods.</details>

### DDCoT: Duty-Distinct Chain-of-Thought Prompting for Multimodal Reasoning in Language Models

**Authors:** Ge Zheng, Bin Yang, Jiajin Tang, Hong-Yu Zhou, Sibei Yang

**<details><summary>Abstract**</summary>

A long-standing goal of AI systems is to perform complex multimodal reasoning like humans. Recently, large language models (LLMs) have made remarkable strides in such multi-step reasoning on the language modality solely by leveraging the chain of thought (CoT) to mimic human thinking. However, the transfer of these advancements to multimodal contexts introduces heightened challenges, including but not limited to the impractical need for labor-intensive annotation and the limitations in terms of flexibility, generalizability, and explainability. To evoke CoT reasoning in multimodality, this work first conducts an in-depth analysis of these challenges posed by multimodality and presents two key insights: “keeping critical thinking” and “letting everyone do their jobs” in multimodal CoT reasoning. Furthermore, this study proposes a novel DDCoT prompting that maintains a critical attitude through negative-space prompting and incorporates multimodality into reasoning by first dividing the reasoning responsibility of LLMs into reasoning and recognition and then integrating the visual recognition capability of visual models into the joint reasoning process. The rationales generated by DDCoT not only improve the reasoning abilities of both large and small language models in zero-shot prompting and fine-tuning learning, significantly outperforming state-of-the-art methods but also exhibit impressive generalizability and explainability.</details>

### DESSERT: An Efficient Algorithm for Vector Set Search with Vector Set Queries

**Authors:** Joshua Engels, Benjamin Coleman, Vihan Lakshman, Anshumali Shrivastava

**<details><summary>Abstract**</summary>

We study the problem of $\text{\emph{vector set search}}$ with $\text{\emph{vector set queries}}$. This task is analogous to traditional near-neighbor search, with the exception that both the query and each element in the collection are $\text{\textit{sets}}$ of vectors. We identify this problem as a core subroutine for semantic search applications and find that existing solutions are unacceptably slow. Towards this end, we present a new approximate search algorithm, DESSERT ($\text{\bf D}$ESSERT $\text{\bf E}$ffeciently $\text{\bf S}$earches $\text{\bf S}$ets of $\text{\bf E}$mbeddings via $\text{\bf R}$etrieval $\text{\bf T}$ables). DESSERT is a general tool with strong theoretical guarantees and excellent empirical performance. When we integrate DESSERT into ColBERT, a state-of-the-art semantic search model, we find a 2-5x speedup on the MS MARCO and LoTTE retrieval benchmarks with minimal loss in recall, underscoring the effectiveness and practical applicability of our proposal.</details>

### DRAUC: An Instance-wise Distributionally Robust AUC Optimization Framework

**Authors:** Siran Dai, Qianqian Xu, Zhiyong Yang, Xiaochun Cao, Qingming Huang

**<details><summary>Abstract**</summary>

The Area Under the ROC Curve (AUC) is a widely employed metric in long-tailed classification scenarios. Nevertheless, most existing methods primarily assume that training and testing examples are drawn i.i.d. from the same distribution, which is often unachievable in practice. Distributionally Robust Optimization (DRO) enhances model performance by optimizing it for the local worst-case scenario, but directly integrating AUC optimization with DRO results in an intractable optimization problem. To tackle this challenge, methodically we propose an instance-wise surrogate loss of Distributionally Robust AUC (DRAUC) and build our optimization framework on top of it. Moreover, we highlight that conventional DRAUC may induce label bias, hence introducing distribution-aware DRAUC as a more suitable metric for robust AUC learning. Theoretically, we affirm that the generalization gap between the training loss and testing error diminishes if the training set is sufficiently large. Empirically, experiments on corrupted benchmark datasets demonstrate the effectiveness of our proposed method. Code is available at: https://github.com/EldercatSAM/DRAUC.</details>

### DaTaSeg: Taming a Universal Multi-Dataset Multi-Task Segmentation Model

**Authors:** Xiuye Gu, Yin Cui, Jonathan Huang, Abdullah Rashwan, Xuan Yang, Xingyi Zhou, Golnaz Ghiasi, Weicheng Kuo, Huizhong Chen, Liang-Chieh Chen, David Ross

**<details><summary>Abstract**</summary>

Observing the close relationship among panoptic, semantic and instance segmentation tasks, we propose to train a universal multi-dataset multi-task segmentation model: DaTaSeg. We use a shared representation (mask proposals with class predictions) for all tasks. To tackle task discrepancy, we adopt different merge operations and post-processing for different tasks. We also leverage weak-supervision, allowing our segmentation model to benefit from cheaper bounding box annotations. To share knowledge across datasets, we use text embeddings from the same semantic embedding space as classifiers and share all network parameters among datasets. We train DaTaSeg on ADE semantic, COCO panoptic, and Objects365 detection datasets. DaTaSeg improves performance on all datasets, especially small-scale datasets, achieving 54.0 mIoU on ADE semantic and 53.5 PQ on COCO panoptic. DaTaSeg also enables weakly-supervised knowledge transfer on ADE panoptic and Objects365 instance segmentation. Experiments show DaTaSeg scales with the number of training datasets and enables open-vocabulary segmentation through direct transfer. In addition, we annotate an Objects365 instance segmentation set of 1,000 images and release it as a public evaluation benchmark on https://laoreja.github.io/dataseg.</details>

### [Oral] DataComp: In search of the next generation of multimodal datasets

**Authors:** Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim Entezari, Giannis Daras, Sarah Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga Saukh, Alexander Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont, Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, Ludwig Schmidt

**Oral Presentation:** Th, Dec 14, 08:30 -- Oral 5D

**<details><summary>Abstract**</summary>

Multimodal datasets are a critical component in recent breakthroughs such as CLIP, Stable Diffusion and GPT-4, yet their design does not receive the same research attention as model architectures or training algorithms. To address this shortcoming in the machine learning ecosystem, we introduce DataComp, a testbed for dataset experiments centered around a new candidate pool of 12.8 billion image-text pairs from Common Crawl. Participants in our benchmark design new filtering techniques or curate new data sources and then evaluate their new dataset by running our standardized CLIP training code and testing the resulting model on 38 downstream test sets. Our benchmark consists of multiple compute scales spanning four orders of magnitude, which enables the study of scaling trends and makes the benchmark accessible to researchers with varying resources. Our baseline experiments show that the DataComp workflow leads to better training sets. Our best baseline, DataComp-1B, enables training a CLIP ViT-L/14 from scratch to 79.2% zero-shot accuracy on ImageNet, outperforming OpenAI's CLIP ViT-L/14 by 3.7 percentage points while using the same training procedure and compute. We release \datanet and all accompanying code at www.datacomp.ai.</details>

### Debiasing Scores and Prompts of 2D Diffusion for View-consistent Text-to-3D Generation

**Authors:** Susung Hong, Donghoon Ahn, Seungryong Kim

**<details><summary>Abstract**</summary>

Existing score-distilling text-to-3D generation techniques, despite their considerable promise, often encounter the view inconsistency problem. One of the most notable issues is the Janus problem, where the most canonical view of an object (	extit{e.g}., face or head) appears in other views. In this work, we explore existing frameworks for score-distilling text-to-3D generation and identify the main causes of the view inconsistency problem---the embedded bias of 2D diffusion models. Based on these findings, we propose two approaches to debias the score-distillation frameworks for view-consistent text-to-3D generation. Our first approach, called score debiasing, involves cutting off the score estimated by 2D diffusion models and gradually increasing the truncation value throughout the optimization process. Our second approach, called prompt debiasing, identifies conflicting words between user prompts and view prompts using a language model, and adjusts the discrepancy between view prompts and the viewing direction of an object. Our experimental results show that our methods improve the realism of the generated 3D objects by significantly reducing artifacts and achieve a good trade-off between faithfulness to the 2D diffusion models and 3D consistency with little overhead. Our project page is available at~\url{https://susunghong.github.io/Debiased-Score-Distillation-Sampling/}.</details>

### Deconstructing Data Reconstruction: Multiclass, Weight Decay and General Losses

**Authors:** Gon Buzaglo, Niv Haim, Gilad Yehudai, Gal Vardi, Yakir Oz, Yaniv Nikankin, Michal Irani

**<details><summary>Abstract**</summary>

Memorization of training data is an active research area, yet our understanding of the inner workings of neural networks is still in its infancy.Recently, Haim et al. 2022 proposed a scheme to reconstruct training samples from multilayer perceptron binary classifiers, effectively demonstrating that a large portion of training samples are encoded in the parameters of such networks.In this work, we extend their findings in several directions, including reconstruction from multiclass and convolutional neural networks. We derive a more general reconstruction scheme which is applicable to a wider range of loss functions such as regression losses. Moreover, we study the various factors that contribute to networks' susceptibility to such reconstruction schemes. Intriguingly, we observe that using weight decay during training increases reconstructability both in terms of quantity and quality. Additionally, we examine the influence of the number of neurons relative to the number of training samples on the reconstructability.Code: https://github.com/gonbuzaglo/decoreco</details>

### [Spotlight] Deep Reinforcement Learning with Plasticity Injection

**Authors:** Evgenii Nikishin, Junhyuk Oh, Georg Ostrovski, Clare Lyle, Razvan Pascanu, Will Dabney, Andre Barreto

**<details><summary>Abstract**</summary>

A growing body of evidence suggests that neural networks employed in deep reinforcement learning (RL) gradually lose their plasticity, the ability to learn from new data; however, the analysis and mitigation of this phenomenon is hampered by the complex relationship between plasticity, exploration, and performance in RL. This paper introduces plasticity injection, a minimalistic intervention that increases the network plasticity without changing the number of trainable parameters or biasing the predictions. The applications of this intervention are two-fold: first, as a diagnostic tool — if injection increases the performance, we may conclude that an agent's network was losing its plasticity. This tool allows us to identify a subset of Atari environments where the lack of plasticity causes performance plateaus, motivating future studies on understanding and combating plasticity loss. Second, plasticity injection can be used to improve the computational efficiency of RL training if the agent has to re-learn from scratch due to exhausted plasticity or by growing the agent's network dynamically without compromising performance. The results on Atari show that plasticity injection attains stronger performance compared to alternative methods while being computationally efficient.</details>

### Demo2Code: From Summarizing Demonstrations to Synthesizing Code via Extended Chain-of-Thought

**Authors:** Yuki Wang, Gonzalo Gonzalez-Pumariega, Yash Sharma, Sanjiban Choudhury

**<details><summary>Abstract**</summary>

Language instructions and demonstrations are two natural ways for users to teach robots personalized tasks. Recent progress in Large Language Models (LLMs) has shown impressive performance in translating language instructions into code for robotic tasks. However, translating demonstrations into task code continues to be a challenge due to the length and complexity of both demonstrations and code, making learning a direct mapping intractable. This paper presents Demo2Code, a novel framework that generates robot task code from demonstrations via an extended chain-of-thought and defines a common latent specification to connect the two. Our framework employs a robust two-stage process: (1) a recursive summarization technique that condenses demonstrations into concise specifications, and (2) a code synthesis approach that expands each function recursively from the generated specifications. We conduct extensive evaluation on various robot task benchmarks, including a novel game benchmark Robotouille, designed to simulate diverse cooking tasks in a kitchen environment.</details>

### Demystifying Structural Disparity in Graph Neural Networks: Can One Size Fit All?

**Authors:** Haitao Mao, Zhikai Chen, Wei Jin, Haoyu Han, Yao Ma, Tong Zhao, Neil Shah, Jiliang Tang

**<details><summary>Abstract**</summary>

Recent studies on Graph Neural Networks(GNNs) provide both empirical and theoretical evidence supporting their effectiveness in capturing structural patterns on both homophilic and certain heterophilic graphs. Notably, most real-world homophilic and heterophilic graphs are comprised of a mixture of nodes in both homophilic and heterophilic structural patterns, exhibiting a structural disparity. However, the analysis of GNN performance with respect to nodes exhibiting different structural patterns, e.g., homophilic nodes in heterophilic graphs, remains rather limited. In the present study, we provide evidence that Graph Neural Networks(GNNs) on node classification typically perform admirably on homophilic nodes within homophilic graphs and heterophilic nodes within heterophilic graphs while struggling on the opposite node set, exhibiting a performance disparity. We theoretically and empirically identify effects of GNNs on testing nodes exhibiting distinct structural patterns. We then propose a rigorous, non-i.i.d PAC-Bayesian generalization bound for GNNs, revealing reasons for the performance disparity, namely the aggregated feature distance and homophily ratio difference between training and testing nodes. Furthermore, we demonstrate the practical implications of our new findings via (1) elucidating the effectiveness of deeper GNNs; and (2) revealing an over-looked distribution shift factor on graph out-of-distribution problem and proposing a new scenario accordingly.</details>

### Demystifying the Optimal Performance of Multi-Class Classification

**Authors:** Minoh Jeong, Martina Cardone, Alex Dytso

**<details><summary>Abstract**</summary>

Classification is a fundamental task in science and engineering on which machine learning methods have shown outstanding performances. However, it is challenging to determine whether such methods have achieved the Bayes error rate, that is, the lowest error rate attained by any classifier. This is mainly due to the fact that the Bayes error rate is not known in general and hence, effectively estimating it is paramount. Inspired by the work by Ishida et al. (2023), we propose an estimator for the Bayes error rate of supervised multi-class classification problems. We analyze several theoretical aspects of such estimator, including its consistency, unbiasedness, convergence rate, variance, and robustness. We also propose a denoising method that reduces the noise that potentially corrupts the data labels, and we improve the robustness of the proposed estimator to outliers by incorporating the median-of-means estimator. Our analysis demonstrates the consistency, asymptotic unbiasedness, convergence rate, and robustness of the proposed estimators. Finally, we validate the effectiveness of our theoretical results via experiments both on synthetic data under various noise settings and on real data.</details>

### Design from Policies: Conservative Test-Time Adaptation for Offline Policy Optimization

**Authors:** Jinxin Liu, Hongyin Zhang, Zifeng Zhuang, Yachen Kang, Donglin Wang, Bin Wang

**<details><summary>Abstract**</summary>

In this work, we decouple the iterative bi-level offline RL (value estimation and policy extraction) from the offline training phase, forming a non-iterative bi-level paradigm and avoiding the iterative error propagation over two levels. Specifically, this non-iterative paradigm allows us to conduct inner-level optimization (value estimation) in training, while performing outer-level optimization (policy extraction) in testing. Naturally, such a paradigm raises three core questions that are not fully answered by prior non-iterative offline RL counterparts like reward-conditioned policy: (q1) What information should we transfer from the inner-level to the outer-level? (q2) What should we pay attention to when exploiting the transferred information for safe/confident outer-level optimization? (q3) What are the benefits of concurrently conducting outer-level optimization during testing? Motivated by model-based optimization (MBO), we propose DROP (design from policies), which fully answers the above questions. Specifically, in the inner-level, DROP decomposes offline data into multiple subsets, and learns an MBO score model (a1). To keep safe exploitation to the score model in the outer-level, we explicitly learn a behavior embedding and introduce a conservative regularization (a2). During testing, we show that DROP permits deployment adaptation, enabling an adaptive inference across states (a3). Empirically, we evaluate DROP on various tasks, showing that DROP gains comparable or better performance compared to prior methods.</details>

### Detecting Any Human-Object Interaction Relationship: Universal HOI Detector with Spatial Prompt Learning on Foundation Models

**Authors:** Yichao Cao, Qingfei Tang, Xiu Su, Song Chen, Shan You, Xiaobo Lu, Chang Xu

**<details><summary>Abstract**</summary>

Human-object interaction (HOI) detection aims to comprehend the intricate relationships between humans and objects, predicting  triplets, and serving as the foundation for numerous computer vision tasks. The complexity and diversity of human-object interactions in the real world, however, pose significant challenges for both annotation and recognition, particularly in recognizing interactions within an open world context. This study explores the universal interaction recognition in an open-world setting through the use of Vision-Language (VL) foundation models and large language models (LLMs). The proposed method is dubbed as UniHOI. We conduct a deep analysis of the three hierarchical features inherent in visual HOI detectors and propose a method for high-level relation extraction aimed at VL foundation models, which we call HO prompt-based learning. Our design includes an HO Prompt-guided Decoder (HOPD), facilitates the association of high-level relation representations in the foundation model with various HO pairs within the image. Furthermore, we utilize a LLM (i.e. GPT) for interaction interpretation, generating a richer linguistic understanding for complex HOIs. For open-category interaction recognition, our method supports either of two input types: interaction phrase or interpretive sentence.  Our efficient architecture design and learning methods effectively unleash the potential of the VL foundation models and LLMs, allowing UniHOI to surpass all existing methods with a substantial margin, under both supervised and zero-shot settings. The code and pre-trained weights will be made publicly available.</details>

### Diff-Foley: Synchronized Video-to-Audio Synthesis with Latent Diffusion Models

**Authors:** Simian Luo, Chuanhao Yan, Chenxu Hu, Hang Zhao

**<details><summary>Abstract**</summary>

The Video-to-Audio (V2A) model has recently gained attention for its practical application in generating audio directly from silent videos, particularly in video/film production. However, previous methods in V2A have limited generation quality in terms of temporal synchronization and audio-visual relevance. We present Diff-Foley, a synchronized Video-to-Audio synthesis method with a latent diffusion model (LDM) that generates high-quality audio with improved synchronization and audio-visual relevance. We adopt contrastive audio-visual pretraining (CAVP) to learn more temporally and semantically aligned features, then train an LDM with CAVP-aligned visual features on spectrogram latent space. The CAVP-aligned features enable LDM to capture the subtler audio-visual correlation via a cross-attention module. We further significantly improve sample quality with `double guidance'. Diff-Foley achieves state-of-the-art V2A performance on current large scale V2A dataset. Furthermore, we demonstrate Diff-Foley practical applicability and adaptability via customized downstream finetuning. Project Page: https://diff-foley.github.io/</details>

### DiffComplete: Diffusion-based Generative 3D Shape Completion

**Authors:** Ruihang Chu, Enze Xie, Shentong Mo, Zhenguo Li, Matthias Niessner, Chi-Wing Fu, Jiaya Jia

**<details><summary>Abstract**</summary>

We introduce a new diffusion-based approach for shape completion on 3D range scans. Compared with prior deterministic and probabilistic methods, we strike a balance between realism, multi-modality, and high fidelity. We propose DiffComplete by casting shape completion as a generative task conditioned on the incomplete shape. Our key designs are two-fold. First, we devise a hierarchical feature aggregation mechanism to inject conditional features in a spatially-consistent manner. So, we can capture both local details and broader contexts of the conditional inputs to control the shape completion. Second, we propose an occupancy-aware fusion strategy in our model to enable the completion of multiple partial shapes and introduce higher flexibility on the input conditions. DiffComplete sets a new SOTA performance (e.g., 40% decrease on $l_1$ error) on two large-scale 3D shape completion benchmarks. Our completed shapes not only have a realistic outlook compared with the deterministic methods but also exhibit high similarity to the ground truths compared with the probabilistic alternatives. Further, DiffComplete has strong generalizability on objects of entirely unseen classes for both synthetic and real data, eliminating the need for model re-training in various applications.</details>

### DiffVL: Scaling Up Soft Body Manipulation using Vision-Language Driven Differentiable Physics

**Authors:** Zhiao Huang, Feng Chen, Yewen Pu, Chunru Lin, Hao Su, Chuang Gan

**<details><summary>Abstract**</summary>

Combining gradient-based trajectory optimization with differentiable physics simulation is an efficient technique for solving soft-body manipulation problems.Using a well-crafted optimization objective, the solver can quickly converge onto a valid trajectory.However, writing the appropriate objective functions requires expert knowledge, making it difficult to collect a large set of naturalistic problems from non-expert users.We introduce DiffVL, a method that enables non-expert users to communicate soft-body manipulation tasks -- a combination of vision and natural language, given in multiple stages -- that can be readily leveraged by a differential physics solver. We have developed GUI tools that enable non-expert users to specify 100 tasks inspired by real-life soft-body manipulations from online videos, which we'll make public.We leverage large language models to translate task descriptions into machine-interpretable optimization objectives. The optimization objectives can help differentiable physics solvers to solve these long-horizon multistage tasks that are challenging for previous baselines.</details>

### Differentiable Clustering with Perturbed Spanning Forests

**Authors:** Lawrence Stewart, Francis Bach, Felipe Llinares-Lopez, Quentin Berthet

**<details><summary>Abstract**</summary>

We introduce a differentiable clustering method based on stochastic perturbations of minimum-weight spanning forests. This allows us to include clustering in end-to-end trainable pipelines, with efficient gradients. We show that our method performs well even in difficult settings, such as data sets with high noise and challenging geometries. We also formulate an ad hoc loss to efficiently learn from partial clustering data using this operation. We demonstrate its performance on several data sets for supervised and semi-supervised tasks.</details>

### Diffusion Model is an Effective Planner and Data Synthesizer for Multi-Task Reinforcement Learning

**Authors:** Haoran He, Chenjia Bai, Kang Xu, Zhuoran Yang, Weinan Zhang, Dong Wang, Bin Zhao, Xuelong Li

**<details><summary>Abstract**</summary>

Diffusion models have demonstrated highly-expressive generative capabilities in vision and NLP. Recent studies in reinforcement learning (RL) have shown that diffusion models are also powerful in modeling complex policies or trajectories in offline datasets. However, these works have been limited to single-task settings where a generalist agent capable of addressing multi-task predicaments is absent. In this paper, we aim to investigate the effectiveness of a single diffusion model in modeling large-scale multi-task offline data, which can be challenging due to diverse and multimodal data distribution. Specifically, we propose Multi-Task Diffusion Model (\textsc{MTDiff}), a diffusion-based method that incorporates Transformer backbones and prompt learning for generative planning and data synthesis in multi-task offline settings. \textsc{MTDiff} leverages vast amounts of knowledge available in multi-task data and performs implicit knowledge sharing among tasks. For generative planning, we find \textsc{MTDiff} outperforms state-of-the-art algorithms across 50 tasks on Meta-World and 8 maps on Maze2D. For data synthesis, \textsc{MTDiff} generates high-quality data for testing tasks given a single demonstration as a prompt, which enhances the low-quality datasets for even unseen tasks.</details>

### Diffusion Schrödinger Bridge Matching

**Authors:** Yuyang Shi, Valentin De Bortoli, Andrew Campbell, Arnaud Doucet

**<details><summary>Abstract**</summary>

Solving transport problems, i.e. finding a map transporting one given distribution to another, has numerous applications in machine learning. Novel mass transport methods motivated by generative modeling have recently been proposed, e.g. Denoising Diffusion Models (DDMs) and Flow Matching Models (FMMs) implement such a transport through a Stochastic Differential Equation (SDE) or an Ordinary Differential Equation (ODE). However, while it is desirable in many applications to approximate the deterministic dynamic Optimal Transport (OT) map which admits attractive properties, DDMs and FMMs are not guaranteed to provide transports close to the OT map. In contrast, Schrödinger bridges (SBs) compute stochastic dynamic mappings which recover entropy-regularized versions of OT. Unfortunately, existing numerical methods approximating SBs either scale poorly with dimension or accumulate errors across iterations. In this work, we introduce Iterative Markovian Fitting (IMF), a new methodology for solving SB problems, and Diffusion Schrödinger Bridge Matching (DSBM), a novel numerical algorithm for computing IMF iterates. DSBM significantly improves over previous SB numerics and recovers as special/limiting cases various recent transport methods. We demonstrate the performance of DSBM on a variety of problems.</details>

### Diffusion with Forward Models: Solving Stochastic Inverse Problems Without Direct Supervision

**Authors:** Ayush Tewari, Tianwei Yin, George Cazenavette, Semon Rezchikov, Josh Tenenbaum, Fredo Durand, Bill Freeman, Vincent Sitzmann

**<details><summary>Abstract**</summary>

Denoising diffusion models are a powerful type of generative models used to capture complex distributions of real-world signals. However, their applicability is limited to scenarios where training samples are readily available, which is not always the case in real-world applications. For example, in inverse graphics, the goal is to generate samples from a distribution of 3D scenes that align with a given image, but ground-truth 3D scenes are unavailable and only 2D images are accessible. To address this limitation, we propose a novel class of denoising diffusion probabilistic models that learn to sample from distributions of signals that are never directly observed. Instead, these signals are measured indirectly through a known differentiable forward model, which produces partial observations of the unknown signal. Our approach involves integrating the forward model directly into the denoising process. A key contribution of our work is the integration of a differentiable forward model into the denoising process. This integration effectively connects the generative modeling of observations with the generative modeling of the underlying signals, allowing for end-to-end training of a conditional generative model over signals. During inference, our approach enables sampling from the distribution of underlying signals that are consistent with a given partial observation. We demonstrate the effectiveness of our method on three challenging computer vision tasks. For instance, in the context of inverse graphics, our model enables direct sampling from the distribution of 3D scenes that align with a single 2D input image.</details>

### Direct Training of SNN using Local Zeroth Order Method

**Authors:** Bhaskar Mukhoty, Velibor Bojkovic, William de Vazelhes, Xiaohan Zhao, Giulia De Masi, Huan Xiong, Bin Gu

**<details><summary>Abstract**</summary>

Spiking neural networks are becoming increasingly popular for their low energy requirement in real-world tasks with accuracy comparable to traditional ANNs. SNN training algorithms face the loss of gradient information and non-differentiability due to the Heaviside function in minimizing the model loss over model parameters. To circumvent this problem, the surrogate method employs a differentiable approximation of the Heaviside function in the backward pass, while the forward pass continues to use the Heaviside as the spiking function. We propose to use the zeroth-order technique at the local or neuron level in training SNNs, motivated by its regularizing and potential energy-efficient effects and establish a theoretical connection between it and the existing surrogate methods. We perform experimental validation of the technique on standard static datasets (CIFAR-10, CIFAR-100, ImageNet-100) and neuromorphic datasets (DVS-CIFAR-10, DVS-Gesture, N-Caltech-101, NCARS) and obtain results that offer improvement over the state-of-the-art results. The proposed method also lends itself to efficient implementations of the back-propagation method, which could provide 3-4 times overall speedup in training time. The code is available at \url{https://github.com/BhaskarMukhoty/LocalZO}.</details>

### Directional diffusion models for graph representation learning

**Authors:** Run Yang, Yuling Yang, Fan Zhou, Qiang Sun

**<details><summary>Abstract**</summary>

Diffusion models have achieved remarkable success in diverse domains such as image synthesis, super-resolution, and 3D molecule generation. Surprisingly, the application of diffusion models in graph learning has garnered little attention. In this paper, we aim to bridge this gap by exploring the use of diffusion models for unsupervised graph representation learning. Our investigation commences with the identification of anisotropic structures within graphs and the recognition of a crucial limitation in the vanilla forward diffusion process when dealing with these anisotropic structures. The original forward diffusion process continually adds isotropic Gaussian noise to the data, which may excessively dilute anisotropic signals, leading to rapid signal-to-noise conversion. This rapid conversion poses challenges for training denoising neural networks and obstructs the acquisition of semantically meaningful representations during the reverse process. To overcome this challenge, we introduce a novel class of models termed {\it directional diffusion models}. These models adopt data-dependent, anisotropic, and directional noises in the forward diffusion process. In order to assess the effectiveness of our proposed models, we conduct extensive experiments on 12 publicly available datasets, with a particular focus on two distinct graph representation learning tasks. The experimental results unequivocally establish the superiority of our models over state-of-the-art baselines, underscoring their effectiveness in capturing meaningful graph representations. Our research not only sheds light on the intricacies of the forward process in diffusion models but also underscores the vast potential of these models in addressing a wide spectrum of graph-related tasks. Our code is available at \url{https://github.com/statsle/DDM}.</details>

### Discovering General Reinforcement Learning Algorithms with Adversarial Environment Design

**Authors:** Matthew T Jackson, Minqi Jiang, Jack Parker-Holder, Risto Vuorio, Chris Lu, Greg Farquhar, Shimon Whiteson, Jakob Foerster

**<details><summary>Abstract**</summary>

The past decade has seen vast progress in deep reinforcement learning (RL) on the back of algorithms manually designed by human researchers. Recently, it has been shown that it is possible to meta-learn update rules, with the hope of discovering algorithms that can perform well on a wide range of RL tasks. Despite impressive initial results from algorithms such as Learned Policy Gradient (LPG), there remains a generalization gap when these algorithms are applied to unseen environments. In this work, we examine how characteristics of the meta-training distribution impact the generalization performance of these algorithms. Motivated by this analysis and building on ideas from Unsupervised Environment Design (UED), we propose a novel approach for automatically generating curricula to maximize the regret of a meta-learned optimizer, in addition to a novel approximation of regret, which we name algorithmic regret (AR). The result is our method, General RL Optimizers Obtained Via Environment Design (GROOVE). In a series of experiments, we show that GROOVE achieves superior generalization to LPG, and evaluate AR against baseline metrics from UED, identifying it as a critical component of environment design in this setting. We believe this approach is a step towards the discovery of truly general RL algorithms, capable of solving a wide range of real-world environments.</details>

### Discriminative Feature Attributions: Bridging Post Hoc Explainability and Inherent Interpretability

**Authors:** Usha Bhalla, Suraj Srinivas, Himabindu Lakkaraju

**<details><summary>Abstract**</summary>

With the increased deployment of machine learning models in various real-world applications, researchers and practitioners alike have emphasized the need for explanations of model behaviour. To this end, two broad strategies have been outlined in prior literature to explain models. Post hoc explanation methods explain the behaviour of complex black-box models by identifying features critical to model predictions; however, prior work has shown that these explanations may not be faithful, in that they incorrectly attribute high importance to features that are unimportant or non-discriminative for the underlying task. Inherently interpretable models, on the other hand, circumvent these issues by explicitly encoding explanations into model architecture, meaning their explanations are naturally faithful, but they often exhibit poor predictive performance due to their limited expressive power. In this work, we identify a key reason for the lack of faithfulness of feature attributions: the lack of robustness of the underlying black-box models, especially the erasure of unimportant distractor features in the input. To address this issue, we propose Distractor Erasure Tuning (DiET), a method that adapts black-box models to be robust to distractor erasure, thus providing discriminative and faithful attributions. This strategy naturally combines the ease-of-use of post hoc explanations with the faithfulness of inherently interpretable models. We perform extensive experiments on semi-synthetic and real-world datasets, and show that DiET produces models that (1) closely approximate the original black-box models they are intended to explain, and (2) yield explanations that match approximate ground truths available by construction.</details>

### Distilling Out-of-Distribution Robustness from Vision-Language Foundation Models

**Authors:** Andy Zhou, Jindong Wang, Yu-Xiong Wang, Haohan Wang

**<details><summary>Abstract**</summary>

We propose a conceptually simple and lightweight framework for improving the robustness of vision models through the combination of knowledge distillation and data augmentation. We address the conjecture that larger models do not make for better teachers by showing strong gains in out-of-distribution robustness when distilling from pretrained foundation models. Following this finding, we propose Discrete Adversarial Distillation (DAD), which leverages a robust teacher to generate adversarial examples and a VQGAN to discretize them, creating more informative samples than standard data augmentation techniques. We provide a theoretical framework for the use of a robust teacher in the knowledge distillation with data augmentation setting and demonstrate strong gains in out-of-distribution robustness and clean accuracy across different student architectures. Notably, our method adds minor computational overhead compared to similar techniques and can be easily combined with other data augmentations for further improvements.</details>

### Distributional Pareto-Optimal Multi-Objective Reinforcement Learning

**Authors:** Xin-Qiang Cai, Pushi Zhang, Li Zhao, Jiang Bian, Masashi Sugiyama, Ashley Llorens

**<details><summary>Abstract**</summary>

Multi-objective reinforcement learning (MORL) has been proposed to learn control policies over multiple competing objectives with each possible preference over returns. However, current MORL algorithms fail to account for distributional preferences over the multi-variate returns, which are particularly important in real-world scenarios such as autonomous driving. To address this issue, we extend the concept of Pareto-optimality in MORL into distributional Pareto-optimality, which captures the optimality of return distributions, rather than the expectations. Our proposed method, called Distributional Pareto-Optimal Multi-Objective Reinforcement Learning~(DPMORL), is capable of learning distributional Pareto-optimal policies that balance multiple objectives while considering the return uncertainty. We evaluated our method on several benchmark problems and demonstrated its effectiveness in discovering distributional Pareto-optimal policies and satisfying diverse distributional preferences compared to existing MORL methods.</details>

### Distributional Policy Evaluation: a Maximum Entropy approach to Representation Learning

**Authors:** Riccardo Zamboni, Alberto Maria Metelli, Marcello Restelli

**<details><summary>Abstract**</summary>

The Maximum Entropy (Max-Ent) framework has been effectively employed in a variety of Reinforcement Learning (RL) tasks. In this paper, we first propose a novel Max-Ent framework for policy evaluation in a distributional RL setting, named *Distributional Maximum Entropy Policy Evaluation* (D-Max-Ent PE). We derive a generalization-error bound that depends on the complexity of the representation employed, showing that this framework can explicitly take into account the features used to represent the state space while evaluating a policy. Then, we exploit these favorable properties to drive the representation learning of the state space in a Structural Risk Minimization fashion. We employ state-aggregation functions as feature functions and we specialize the D-Max-Ent approach into an algorithm, named *D-Max-Ent Progressive Factorization*, which constructs a progressively finer-grained representation of the state space by balancing the trade-off between preserving information (bias) and reducing the effective number of states, i.e., the complexity of the representation space (variance). Finally, we report the results of some illustrative numerical simulations, showing that the proposed algorithm matches the expected theoretical behavior and highlighting the relationship between aggregations and sample regimes.</details>

### Diversify \& Conquer: Outcome-directed Curriculum RL via Out-of-Distribution Disagreement

**Authors:** Daesol Cho, Seungjae Lee, H. Jin Kim

**<details><summary>Abstract**</summary>

Reinforcement learning (RL) often faces the challenges of uninformed search problems where the agent should explore without access to the domain knowledge such as characteristics of the environment or external rewards. To tackle these challenges, this work proposes a new approach for curriculum RL called $\textbf{D}$iversify for $\textbf{D}$isagreement \& $\textbf{C}$onquer ($\textbf{D2C}$). Unlike previous curriculum learning methods, D2C requires only a few examples of desired outcomes and works in any environment, regardless of its geometry or the distribution of the desired outcome examples. The proposed method performs diversification of the goal-conditional classifiers to identify similarities between visited and desired outcome states and ensures that the classifiers disagree on states from out-of-distribution, which enables quantifying the unexplored region and designing an arbitrary goal-conditioned intrinsic reward signal in a simple and intuitive way. The proposed method then employs bipartite matching to define a curriculum learning objective that produces a sequence of well-adjusted intermediate goals, which enable the agent to automatically explore and conquer the unexplored region. We present experimental results demonstrating that D2C outperforms prior curriculum RL methods in both quantitative and qualitative aspects, even with the arbitrarily distributed desired outcome examples.</details>

### Do SSL Models Have Déjà Vu? A Case of Unintended Memorization in Self-supervised Learning

**Authors:** Casey Meehan, Florian Bordes, Pascal Vincent, Kamalika Chaudhuri, Chuan Guo

**<details><summary>Abstract**</summary>

Self-supervised learning (SSL) algorithms can produce useful image representations by learning to associate different parts of natural images with one another. However, when taken to the extreme, SSL models can unintendedly memorize specific parts in individual training samples rather than learning semantically meaningful associations. In this work, we perform a systematic study of the unintended memorization of image-specific information in SSL models -- which we refer to as déjà vu memorization. Concretely, we show that given the trained model and a crop of a training image containing only the background (e.g., water, sky, grass), it is possible to infer the foreground object with high accuracy or even visually reconstruct it. Furthermore, we show that déjà vu memorization is common to different SSL algorithms, is exacerbated by certain design choices, and cannot be detected by conventional techniques for evaluating representation quality. Our study of déjà vu memorization reveals previously unknown privacy risks in SSL models, as well as suggests potential practical mitigation strategies.</details>

### Domain Re-Modulation for Few-Shot Generative Domain Adaptation

**Authors:** Yi Wu, Ziqiang Li, Chaoyue Wang, Heliang Zheng, Shanshan Zhao, Bin Li, Dacheng Tao

**<details><summary>Abstract**</summary>

In this study, we delve into the task of few-shot Generative Domain Adaptation (GDA), which involves transferring a pre-trained generator from one domain to a new domain using only a few reference images. Inspired by the way human brains acquire knowledge in new domains, we present an innovative generator structure called $	extbf{Domain Re-Modulation (DoRM)}$. DoRM not only meets the criteria of $	extit{high quality}$, $	extit{large synthesis diversity}$, and $	extit{cross-domain consistency}$, which were achieved by previous research in GDA, but also incorporates $	extit{memory}$ and $	extit{domain association}$, akin to how human brains operate. Specifically, DoRM freezes the source generator and introduces new mapping and affine modules (M\&A modules) to capture the attributes of the target domain during GDA. This process resembles the formation of new synapses in human brains. Consequently, a linearly combinable domain shift occurs in the style space. By incorporating multiple new M\&A modules, the generator gains the capability to perform high-fidelity multi-domain and hybrid-domain generation. Moreover, to maintain cross-domain consistency more effectively, we introduce a similarity-based structure loss. This loss aligns the auto-correlation map of the target image with its corresponding auto-correlation map of the source image during training. Through extensive experiments, we demonstrate the superior performance of our DoRM and similarity-based structure loss in few-shot GDA, both quantitatively and qualitatively. Code will be available at https://github.com/wuyi2020/DoRM.</details>

### Don't be so Monotone: Relaxing Stochastic Line Search in Over-Parameterized Models

**Authors:** Leonardo Galli, Holger Rauhut, Mark Schmidt

**<details><summary>Abstract**</summary>

Recent works have shown that line search methods can speed up Stochastic Gradient Descent (SGD) and Adam in modern over-parameterized settings. However, existing line searches may take steps that are smaller than necessary since they require a monotone decrease of the (mini-)batch objective function. We explore nonmonotone line search methods to relax this condition and possibly accept larger step sizes. Despite the lack of a monotonic decrease, we prove the same fast rates of convergence as in the monotone case. Our experiments show that nonmonotone methods improve the speed of convergence and generalization properties of SGD/Adam even beyond the previous monotone line searches. We propose a POlyak NOnmonotone Stochastic (PoNoS) method, obtained by combining a nonmonotone line search with a Polyak initial step size. Furthermore, we develop a new resetting technique that in the majority of the iterations reduces the amount of backtracks to zero while still maintaining a large initial step size. To the best of our knowledge, a first runtime comparison shows that the epoch-wise advantage of line-search-based methods gets reflected in the overall computational time.</details>

### Double Randomized Underdamped Langevin with Dimension-Independent Convergence Guarantee

**Authors:** Yuanshi Liu, Cong Fang, Tong Zhang

**<details><summary>Abstract**</summary>

This paper focuses on the high-dimensional sampling of log-concave distributions with composite structures: $p^*(\mathrm{d}x)\propto \exp(-g(x)-f(x))\mathrm{d}x$. We develop a double randomization technique, which leads to a fast underdamped Langevin algorithm with a dimension-independent convergence guarantee. We prove that the algorithm enjoys an overall $\tilde{\mathcal{O}}\left(\frac{\left(\mathrm{tr}(H)\right)^{1/3}}{\epsilon^{2/3}}\right)$ iteration complexity to reach an $\epsilon$-tolerated sample whose distribution $p$ admits $W_2(p,p^*)\leq \epsilon$.  Here,  $H$ is an upper bound of the Hessian matrices for $f$ and does not explicitly depend on dimension $d$. For the posterior sampling over linear models with normalized data, we show a clear superiority of convergence rate which is dimension-free and outperforms the previous best-known results by a $d^{1/3}$ factor. The analysis to achieve a faster convergence rate brings new insights into high-dimensional sampling.</details>

### Doubly Robust Augmented Transfer for Meta-Reinforcement Learning

**Authors:** Yuankun Jiang, Nuowen Kan, Chenglin Li, Wenrui Dai, Junni Zou, Hongkai Xiong

**<details><summary>Abstract**</summary>

Meta-reinforcement learning (Meta-RL), though enabling a fast adaptation to learn new skills by exploiting the common structure shared among different tasks, suffers performance degradation in the sparse-reward setting. Current hindsight-based sample transfer approaches can alleviate this issue by transferring relabeled trajectories from other tasks to a new task so as to provide informative experience for the target reward function, but are unfortunately constrained with the unrealistic assumption that tasks differ only in reward functions. In this paper, we propose a doubly robust augmented transfer (DRaT) approach, aiming at addressing the more general sparse reward meta-RL scenario with both dynamics mismatches and varying reward functions across tasks. Specifically, we design a doubly robust augmented estimator for efficient value-function evaluation, which tackles dynamics mismatches with the optimal importance weight of transition distributions achieved by minimizing the theoretically derived upper bound of mean squared error (MSE) between the estimated values of transferred samples and their true values in the target task. Due to its intractability, we then propose an interval-based approximation to this optimal importance weight, which is guaranteed to cover the optimum with a constrained and sample-independent upper bound on the MSE approximation error. Based on our theoretical findings, we finally develop a DRaT algorithm for transferring informative samples across tasks during the training of meta-RL. We implement DRaT on an off-policy meta-RL baseline, and empirically show that it significantly outperforms other hindsight-based approaches on various sparse-reward MuJoCo locomotion tasks with varying dynamics and reward functions.</details>

### Doubly-Robust Self-Training

**Authors:** Banghua Zhu, Mingyu Ding, Philip Jacobson, Ming Wu, Wei Zhan, Michael Jordan, Jiantao Jiao

**<details><summary>Abstract**</summary>

Self-training is a well-established technique in semi-supervised learning, which leverages unlabeled data by generating pseudo-labels and incorporating them with a limited labeled dataset for training. The effectiveness of self-training heavily relies on the accuracy of these pseudo-labels. In this paper, we introduce doubly-robust self-training, an innovative semi-supervised algorithm that provably balances between two extremes. When pseudo-labels are entirely incorrect, our method reduces to a training process solely using labeled data. Conversely, when pseudo-labels are completely accurate, our method transforms into a training process utilizing all pseudo-labeled data and labeled data, thus increasing the effective sample size. Through empirical evaluations on both the ImageNet dataset for image classification and the nuScenes autonomous driving dataset for 3D object detection, we demonstrate the superiority of the doubly-robust loss over the self-training baseline.</details>

### DreamHuman: Animatable 3D Avatars from Text

**Authors:** Nikos Kolotouros, Thiemo Alldieck, Andrei Zanfir, Eduard Bazavan, Mihai Fieraru, Cristian Sminchisescu

**<details><summary>Abstract**</summary>

We present \emph{DreamHuman}, a method to generate realistic animatable 3D human avatar models entirely from textual descriptions. Recent text-to-3D methods have made considerable strides in generation, but are still lacking in important aspects. Control and often spatial resolution remain limited, existing methods produce fixed rather than 3D human models that can be placed in different poses (i.e. re-posable or animatable), and anthropometric consistency for complex structures like people remains a challenge. \emph{DreamHuman} connects large text-to-image synthesis models, neural radiance fields, and statistical human body models in a novel optimization framework. This makes it possible to generate dynamic 3D human avatars with high-quality textures and learnt per-instance rigid and non rigid geometric deformations. We demonstrate that our method is capable to generate a wide variety of animatable, realistic 3D human models from text. These have diverse appearance, clothing, skin tones and body shapes, and outperform both generic text-to-3D approaches and previous text-based 3D avatar generators in visual fidelity.</details>

### DrugCLIP: Contrasive Protein-Molecule Representation Learning for Virtual Screening

**Authors:** Bowen Gao, Bo Qiang, Haichuan Tan, Yinjun Jia, Minsi Ren, Minsi Lu, Jingjing Liu, Wei-Ying Ma, Yanyan Lan

**<details><summary>Abstract**</summary>

Virtual screening, which identifies potential drugs from vast compound databases to bind with a particular protein pocket, is a critical step in AI-assisted drug discovery. Traditional docking methods are highly time-consuming, and can only work with a restricted search library in real-life applications. Recent supervised learning approaches using scoring functions for binding-affinity prediction, although promising, have not yet surpassed docking methods due to their strong dependency on limited data with reliable binding-affinity labels. In this paper, we propose a novel contrastive learning framework, DrugCLIP, by reformulating virtual screening as a dense retrieval task and employing contrastive learning to align representations of binding protein pockets and molecules from a large quantity of pairwise data without explicit binding-affinity scores. We also introduce a biological-knowledge inspired data augmentation strategy to learn better protein-molecule representations. Extensive experiments show that DrugCLIP significantly outperforms traditional docking and supervised learning methods on diverse virtual screening benchmarks with highly reduced computation time, especially in zero-shot setting.</details>

### DynPoint: Dynamic Neural Point For View Synthesis

**Authors:** Kaichen Zhou, Jia-Xing Zhong, Sangyun Shin, Kai Lu, Yiyuan Yang, Andrew Markham, Niki Trigoni

**<details><summary>Abstract**</summary>

The introduction of neural radiance fields has greatly improved the effectiveness of view synthesis for monocular videos. However, existing algorithms face difficulties when dealing with uncontrolled or lengthy scenarios, and require extensive training time specific to each new scenario.To tackle these limitations, we propose DynPoint, an algorithm designed to facilitate the rapid synthesis of novel views for unconstrained monocular videos. Rather than encoding the entirety of the scenario information into a latent representation, DynPoint concentrates on predicting the explicit 3D correspondence between neighboring frames to realize information aggregation.Specifically, this correspondence prediction is achieved through the estimation of consistent depth and scene flow information across frames.Subsequently, the acquired correspondence is utilized to aggregate information from multiple reference frames to a target frame, by constructing hierarchical neural point clouds. The resulting framework enables swift and accurate view synthesis for desired views of target frames. The experimental results obtained demonstrate the considerable acceleration of training time achieved - typically an order of magnitude - by our proposed method while yielding comparable outcomes compared to prior approaches. Furthermore, our method exhibits strong robustness in handling long-duration videos without learning a canonical representation of video content.</details>

### [Spotlight] Dynamics of Finite Width Kernel and Prediction Fluctuations in Mean Field Neural Networks

**Authors:** Blake Bordelon, Cengiz Pehlevan

**<details><summary>Abstract**</summary>

We analyze the dynamics of finite width effects in wide but finite feature learning neural networks. Starting from a dynamical mean field theory  description of infinite width deep neural network kernel and prediction dynamics, we provide a characterization of the $\mathcal{O}(1/\sqrt{\text{width}})$ fluctuations of the DMFT order parameters over random initializations of the network weights. Our results, while perturbative in width, unlike prior analyses, are non-perturbative in the strength of feature learning. In the lazy limit of network training, all kernels are random but static in time and the prediction variance has a universal form. However, in the rich, feature learning regime, the fluctuations of the kernels and predictions are dynamically coupled with a variance that can be computed self-consistently. In two layer networks, we show how feature learning can dynamically reduce the variance of the final tangent kernel and final network predictions. We also show how initialization variance can slow down online learning in wide but finite networks. In deeper networks, kernel variance can dramatically accumulate through subsequent layers at large feature learning strengths, but feature learning continues to improve the signal-to-noise ratio of the feature kernels. In discrete time, we demonstrate that large learning rate phenomena such as edge of stability effects can be well captured by infinite width dynamics and that initialization variance can decrease dynamically. For CNNs trained on CIFAR-10, we empirically find significant corrections to both the bias and variance of network dynamics due to finite width.</details>

### DäRF: Boosting Radiance Fields from Sparse Input Views with Monocular Depth Adaptation

**Authors:** Jiuhn Song, Seonghoon Park, Honggyu An, Seokju Cho, Min-Seop Kwak, Sungjin Cho, Seungryong Kim

**<details><summary>Abstract**</summary>

Neural radiance field (NeRF) shows powerful performance in novel view synthesis and 3D geometry reconstruction, but it suffers from critical performance degradation when the number of known viewpoints is drastically reduced. Existing works attempt to overcome this problem by employing external priors, but their success is limited to certain types of scenes or datasets. Employing monocular depth estimation (MDE) networks, pretrained on large-scale RGB-D datasets, with powerful generalization capability may be a key to solving this problem: however, using MDE in conjunction with NeRF comes with a new set of challenges due to various ambiguity problems exhibited by monocular depths. In this light, we propose a novel framework, dubbed DäRF, that achieves robust NeRF reconstruction with a handful of real-world images by combining the strengths of NeRF and monocular depth estimation through online complementary training. Our framework imposes the MDE network's powerful geometry prior to NeRF representation at both seen and unseen viewpoints to enhance its robustness and coherence. In addition, we overcome the ambiguity problems of monocular depths through patch-wise scale-shift fitting and geometry distillation, which adapts the MDE network to produce depths aligned accurately with NeRF geometry. Experiments show our framework achieves state-of-the-art results both quantitatively and qualitatively, demonstrating consistent and reliable performance in both indoor and outdoor real-world datasets.</details>

### ELDEN: Exploration via Local Dependencies

**Authors:** Zizhao Wang, Jiaheng Hu, Peter Stone, Roberto Martín-Martín

**<details><summary>Abstract**</summary>

Tasks with large state space and sparse rewards present a longstanding challenge to reinforcement learning. In these tasks, an agent needs to explore the state space efficiently until it finds a reward. To deal with this problem, the community has proposed to augment the reward function with intrinsic reward, a bonus signal that encourages the agent to visit interesting states. In this work, we propose a new way of defining interesting states for environments with factored state spaces and complex chained dependencies, where an agent's actions may change the value of one entity that, in order, may affect the value of another entity. Our insight is that, in these environments, interesting states for exploration are states where the agent is uncertain whether (as opposed to how) entities such as the agent or objects have some influence on each other. We present ELDEN, Exploration via Local DepENdencies, a novel intrinsic reward that encourages the discovery of new interactions between entities. ELDEN utilizes a novel scheme --- the partial derivative of the learned dynamics to model the local dependencies between entities accurately and computationally efficiently. The uncertainty of the predicted dependencies is then used as an intrinsic reward to encourage exploration toward new interactions. We evaluate the performance of ELDEN on four different domains with complex dependencies, ranging from 2D grid worlds to 3D robotic tasks. In all domains, ELDEN correctly identifies local dependencies and learns successful policies, significantly outperforming previous state-of-the-art exploration methods.</details>

### ESSEN: Improving Evolution State Estimation for Temporal Networks using Von Neumann Entropy

**Authors:** Qiyao Huang, Yingyue Zhang, Zhihong Zhang, Edwin Hancock

**<details><summary>Abstract**</summary>

Temporal networks are widely used as abstract graph representations for real-world dynamic systems. Indeed, recognizing the network evolution states is crucial in understanding and analyzing temporal networks. For instance, social networks will generate the clustering and formation of tightly-knit groups or communities over time, relying on the triadic closure theory. However, the existing methods often struggle to account for the time-varying nature of these network structures, hindering their performance when applied to networks with complex evolution states. To mitigate this problem, we propose a novel framework called ESSEN, an Evolution StateS awarE Network, to measure temporal network evolution using von Neumann entropy and thermodynamic temperature. The developed framework utilizes a von Neumann entropy aware attention mechanism and network evolution state contrastive learning in the graph encoding. In addition, it employs a unique decoder the so-called Mixture of Thermodynamic Experts (MoTE) for decoding. ESSEN extracts local and global network evolution information using thermodynamic features and adaptively recognizes the network evolution states. Moreover, the proposed method is evaluated on link prediction tasks under both transductive and inductive settings, with the corresponding results demonstrating its effectiveness compared to various state-of-the-art baselines.</details>

### EV-Eye: Rethinking High-frequency Eye Tracking through the Lenses of Event Cameras

**Authors:** Guangrong Zhao, Yurun Yang, Jingwei Liu, Ning Chen, Yiran Shen, Hongkai Wen, Guohao Lan

**<details><summary>Abstract**</summary>

In this paper, we present EV-Eye, a first-of-its-kind large scale multimodal eye tracking dataset aimed at inspiring research on high-frequency eye/gaze tracking. EV-Eye utilizes an emerging bio-inspired event camera to capture independent pixel-level intensity changes induced by eye movements, achieving sub-microsecond latency. Our dataset was curated over a two-week period and collected from 48 participants encompassing diverse genders and age groups. It comprises over 1.5 million near-eye grayscale images and 2.7 billion event samples generated by two DAVIS346 event cameras. Additionally, the dataset contains 675 thousands scene images and 2.7 million gaze references captured by Tobii Pro Glasses 3 eye tracker for cross-modality validation. Compared with existing event-based high-frequency eye tracking datasets, our dataset is significantly larger in size, and the gaze references involve more natural eye movement patterns, i.e., fixation, saccade and smooth pursuit. Alongside the event data, we also present a hybrid eye tracking method as benchmark, which leverages both the near-eye grayscale images and event data for robust and high-frequency eye tracking. We show that our method achieves higher accuracy for both pupil and gaze estimation tasks compared to the existing solution.</details>

### Easy Learning from Label Proportions

**Authors:** Róbert Busa-Fekete, Heejin Choi, Travis Dick, Claudio Gentile, Andres Munoz Medina

**<details><summary>Abstract**</summary>

We consider the problem of Learning from Label Proportions (LLP), a weakly supervised classification setup where instances are grouped into i.i.d. “bags”, and only the frequency of class labels at each bag is available.  Albeit, the objective of the learner is to achieve low task loss at an individual instance level.  Here we propose EASYLLP, a flexible and simple-to-implement debiasing approach based on aggregate labels, which operates on arbitrary loss functions. Our technique allows us to accurately estimate the expected loss of an arbitrary model at an individual level.  We elucidate the differences between our method and standard methods based on label proportion matching, in terms of applicability and optimality conditions. We showcase the flexibility of our approach compared to alternatives by applying our method to popular learning frameworks, like Empirical Risk Minimization (ERM) and Stochastic Gradient Descent (SGD) with provable guarantees on instance level performance.  Finally, we validate our theoretical results on multiple datasets, empirically illustrating the conditions under which our algorithm is expected to perform better or worse than previous LLP approaches</details>

### Efficient Adversarial Contrastive Learning via Robustness-Aware Coreset Selection

**Authors:** Xilie Xu, Jingfeng ZHANG, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**<details><summary>Abstract**</summary>

Adversarial contrastive learning (ACL) does not require expensive data annotations but outputs a robust representation that withstands adversarial attacks and also generalizes to a wide range of downstream tasks. However, ACL needs tremendous running time to generate the adversarial variants of all training data, which limits its scalability to large datasets. To speed up ACL, this paper proposes a robustness-aware coreset selection (RCS) method. RCS does not require label information and searches for an informative subset that minimizes a representational divergence, which is the distance of the representation between natural data and their virtual adversarial variants. The vanilla solution of RCS via traversing all possible subsets is computationally prohibitive. Therefore, we theoretically transform RCS into a surrogate problem of submodular maximization, of which the greedy search is an efficient solution with an optimality guarantee for the original problem. Empirically, our comprehensive results corroborate that RCS can speed up ACL by a large margin without significantly hurting the robustness transferability. Notably, to the best of our knowledge, we are the first to conduct ACL efficiently on the large-scale ImageNet-1K dataset to obtain an effective robust representation via RCS. Our source code is at https://github.com/GodXuxilie/Efficient_ACL_via_RCS.</details>

### Efficient Data Subset Selection to Generalize Training Across Models: Transductive and Inductive Networks

**Authors:** Eeshaan Jain, Tushar Nandy, Gaurav Aggarwal, Ashish Tendulkar, Rishabh Iyer, Abir De

**<details><summary>Abstract**</summary>

Existing subset selection methods for efficient learning predominantly employ discrete combinatorial and model-specific approaches, which lack generalizability--- for each new model, the algorithm has to be executed from the beginning. Therefore, for an unseen architecture, one cannot use the subset chosen for a different model. In this work, we propose $\texttt{SubSelNet}$, a non-adaptive subset selection framework, which tackles these problems. Here, we first introduce an attention-based neural gadget that leverages the graph structure of architectures and acts as a surrogate to trained deep neural networks for quick model prediction. Then, we use these predictions to build subset samplers. This naturally provides us two variants of $\texttt{SubSelNet}$. The first variant is transductive (called Transductive-$\texttt{SubSelNet}$), which computes the subset separately for each model by solving a small optimization problem. Such an optimization is still super fast, thanks to the replacement of explicit model training by the model approximator. The second variant is inductive (called Inductive-$\texttt{SubSelNet}$), which computes the subset using a trained subset selector, without any optimization.  Our experiments show that our model outperforms several methods across several real datasets.</details>

### Efficient Hyper-parameter Optimization with Cubic Regularization

**Authors:** Zhenqian Shen, Hansi Yang, Yong Li, James Kwok, Quanming Yao

**<details><summary>Abstract**</summary>

As hyper-parameters are ubiquitous and can significantly affect the model performance, hyper-parameter optimization is extremely important in machine learning. In this paper, we consider a sub-class of hyper-parameter optimization problems, where the hyper-gradients are not available. Such problems frequently appear when the performance metric is non-differentiable or the hyper-parameter is not continuous. However, existing algorithms, like Bayesian optimization and reinforcement learning, often get trapped in local optimals with poor performance. To address the above limitations, we propose to use cubic regularization to accelerate convergence and avoid saddle points. First, we adopt stochastic relaxation, which allows obtaining gradient and Hessian information without hyper-gradients. Then, we exploit the rich curvature information by cubic regularization. Theoretically, we prove that the proposed method can converge to approximate second-order stationary points, and the convergence is also guaranteed when the lower-level problem is inexactly solved. Experiments on synthetic and real-world data demonstrate the effectiveness of our proposed method.</details>

### Efficient Sampling of Stochastic Differential Equations with Positive Semi-Definite Models

**Authors:** Anant Raj, Umut Simsekli, Alessandro Rudi

**<details><summary>Abstract**</summary>

This paper deals with the problem of efficient sampling from a stochastic differential equation, given the drift function and the diffusion matrix. The proposed approach leverages a recent model for probabilities (Rudi and Ciliberto, 2021) (the positive semi-definite -- PSD model) from which it is possible to obtain independent and identically distributed (i.i.d.) samples at precision $\varepsilon$ with a cost that is $m^2 d \log(1/\varepsilon)$ where $m$ is the dimension of the model, $d$ the dimension of the space. The proposed approach consists in: first, computing the PSD model that satisfies the Fokker-Planck equation (or its fractional variant) associated with the SDE, up to error $\varepsilon$, and then sampling from the resulting PSD model. Assuming some regularity of the Fokker-Planck solution (i.e. $\beta$-times differentiability plus some geometric condition on its zeros) We obtain an algorithm that: (a) in the preparatory phase obtains a PSD model with L2 distance $\varepsilon$ from the solution of the equation, with a model of dimension $m = \varepsilon^{-(d+1)/(\beta-2s)} (\log(1/\varepsilon))^{d+1}$ where $1/2\leq s\leq1$ is the fractional power to the Laplacian, and total computational complexity of $O(m^{3.5} \log(1/\varepsilon))$ and then (b) for Fokker-Planck equation, it is able to produce i.i.d.\ samples with error $\varepsilon$ in Wasserstein-1 distance, with a cost that is $O(d \varepsilon^{-2(d+1)/\beta-2} \log(1/\varepsilon)^{2d+3})$ per sample. This means that, if the probability associated with the SDE is somewhat regular, i.e. $\beta \geq 4d+2$, then the algorithm requires $O(\varepsilon^{-0.88} \log(1/\varepsilon)^{4.5d})$ in the preparatory phase, and $O(\varepsilon^{-1/2}\log(1/\varepsilon)^{2d+2})$ for each sample. Our results suggest that as the true solution gets smoother, we can circumvent the curse of dimensionality without requiring any sort of convexity.</details>

### Efficient Testable Learning of Halfspaces with Adversarial Label Noise

**Authors:** Ilias Diakonikolas, Daniel Kane, Vasilis Kontonis, Sihan Liu, Nikos Zarifis

**<details><summary>Abstract**</summary>

We give the first polynomial-time algorithm for the testable learning of halfspaces in the presence of adversarial label noise under the Gaussian distribution. In the recently introduced testable learning model, one is required to produce a tester-learner such that if the data passes the tester, then one can trust the output of the robust learner on the data. Our tester-learner runs in time $\text{poly}(d/\epsilon)$ and outputs a halfspace with misclassification error $O(\text{opt})+\epsilon$, where $\text{opt}$ is the 0-1 error of the best fitting halfspace. At a technical level, our algorithm employs an iterative soft localization technique enhanced with appropriate testers to ensure that the data distribution is sufficiently similar to a Gaussian. Finally, our algorithm can be readily adapted to yield an efficient and testable active learner requiring only $d ~ \text{polylog}(1/\epsilon)$ labeled examples.</details>

### [Oral] EgoEnv: Human-centric environment representations from egocentric video

**Authors:** Tushar Nagarajan, Santhosh Kumar Ramakrishnan, Ruta Desai, James Hillis, Kristen Grauman

**Oral Presentation:** Th, Dec 14, 08:15 -- Oral 5D

**<details><summary>Abstract**</summary>

First-person video highlights a camera-wearer's activities in the context of their persistent environment. However, current video understanding approaches reason over visual features from short video clips that are detached from the underlying physical space and  capture only what is immediately visible. To facilitate human-centric environment understanding, we present an approach that links egocentric video and the environment by learning representations that are predictive of the camera-wearer's (potentially unseen) local surroundings. We train such models using videos from agents in simulated 3D environments where the environment is fully observable, and test them on human-captured real-world videos from unseen environments. On two human-centric video tasks, we show that models equipped with our environment-aware features consistently outperform their counterparts with traditional clip features. Moreover, despite being trained exclusively on simulated videos, our approach successfully handles real-world videos from HouseTours and Ego4D, and achieves state-of-the-art results on the Ego4D NLQ challenge.</details>

### Eliminating Domain Bias for Federated Learning in Representation Space

**Authors:** Jianqing Zhang, Yang Hua, Jian Cao, Hao Wang, Tao Song, Zhengui XUE, Ruhui Ma, Haibing Guan

**<details><summary>Abstract**</summary>

Recently, federated learning (FL) is popular for its privacy-preserving and collaborative learning abilities. However, under statistically heterogeneous scenarios, we observe that biased data domains on clients cause a representation bias phenomenon and further degenerate generic representations during local training, i.e., the representation degeneration phenomenon. To address these issues, we propose a general framework Domain Bias Eliminator (DBE) for FL. Our theoretical analysis reveals that DBE can promote bi-directional knowledge transfer between server and client, as it reduces the domain discrepancy between server and client in representation space. Besides, extensive experiments on four datasets show that DBE can greatly improve existing FL methods in both generalization and personalization abilities. The DBE-equipped FL method can outperform ten state-of-the-art personalized FL methods by a large margin. Our code is public at https://github.com/TsingZ0/DBE.</details>

### Emergent Communication for Rules Reasoning

**Authors:** Yuxuan Guo, Yifan Hao, Rui Zhang, Enshuai Zhou, Zidong Du, xishan zhang, Xinkai Song, Yuanbo Wen, Yongwei Zhao, Xuehai Zhou, Jiaming Guo, Qi Yi, Shaohui Peng, Di Huang, Ruizhi Chen, Qi Guo, Yunji Chen

**<details><summary>Abstract**</summary>

Research on emergent communication between deep-learning-based agents has received extensive attention due to its inspiration for linguistics and artificial intelligence.   However, previous attempts have hovered around emerging communication under perception-oriented environmental settings,  that forces agents to describe low-level perceptual features intra image or symbol contexts.  In this work, inspired by the classic human reasoning test (namely Raven's Progressive Matrix), we propose the Reasoning Game, a cognition-oriented environment that encourages agents to reason and communicate high-level rules, rather than perceived low-level contexts.  Moreover, we propose 1) an unbiased dataset (namely rule-RAVEN) as a benchmark to avoid overfitting, 2) and a two-stage curriculum agent training method as a baseline for more stable convergence in the Reasoning Game,  where contexts and semantics are bilaterally drifting.  Experimental results show that, in the Reasoning Game, a semantically stable and compositional language emerges to solve reasoning problems.  The emerged language helps agents apply the extracted rules to the generalization of unseen context attributes, and to the transfer between different context attributes or even tasks.</details>

### End-to-End Meta-Bayesian Optimisation with Transformer Neural Processes

**Authors:** Alexandre Maraval, Matthieu Zimmer, Antoine Grosnit, Haitham Bou Ammar

**<details><summary>Abstract**</summary>

Meta-Bayesian optimisation (meta-BO) aims to improve the sample efficiency of Bayesian optimisation by leveraging data from related tasks. While previous methods successfully meta-learn either a surrogate model or an acquisition function independently, joint training of both components remains an open challenge. This paper proposes the first end-to-end differentiable meta-BO framework that generalises neural processes to learn acquisition functions via transformer architectures. We enable this end-to-end framework with reinforcement learning (RL) to tackle the lack of labelled acquisition data. Early on, we notice that training transformer-based neural processes from scratch with RL is challenging due to insufficient supervision, especially when rewards are sparse. We formalise this claim with a combinatorial analysis showing that the widely used notion of regret as a reward signal exhibits a logarithmic sparsity pattern in trajectory lengths. To tackle this problem, we augment the RL objective with an auxiliary task that guides part of the architecture to learn a valid probabilistic model as an inductive bias. We demonstrate that our method achieves state-of-the-art regret results against various baselines in experiments on standard hyperparameter optimisation tasks and also outperforms others in the real-world problems of mixed-integer programming tuning, antibody design, and logic synthesis for electronic design automation.</details>

### Energy-Based Cross Attention for Bayesian Context Update in Text-to-Image Diffusion Models

**Authors:** Geon Yeong Park, Jeongsol Kim, Beomsu Kim, Sang Wan Lee, Jong Chul Ye

**<details><summary>Abstract**</summary>

Despite the remarkable performance of text-to-image diffusion models in image generation tasks, recent studies have raised the issue that generated images sometimes cannot capture the intended semantic contents of the text prompts, which phenomenon is often called semantic misalignment. To address this, here we present a novel energy-based model (EBM) framework for adaptive context control by modeling the posterior of context vectors. Specifically, we first formulate EBMs of latent image representations and text embeddings in each cross-attention layer of the denoising autoencoder. Then, we obtain the gradient of the log posterior of context vectors, which can be updated and transferred to the subsequent cross-attention layer, thereby implicitly minimizing a nested hierarchy of energy functions. Our latent EBMs further allow zero-shot compositional generation as a linear combination of cross-attention outputs from different contexts. Using extensive experiments, we demonstrate that the proposed method is highly effective in handling various image generation tasks, including multi-concept generation, text-guided image inpainting, and real and synthetic image editing. Code: https://github.com/EnergyAttention/Energy-Based-CrossAttention.</details>

### Energy-based learning algorithms for analog computing: a comparative study

**Authors:** Benjamin Scellier, Maxence Ernoult, Jack Kendall, Suhas Kumar

**<details><summary>Abstract**</summary>

Energy-based learning algorithms have recently gained a surge of interest due to their compatibility with analog (post-digital) hardware. Existing algorithms include contrastive learning (CL), equilibrium propagation (EP) and coupled learning (CpL), all consisting in contrasting two states, and differing in the type of perturbation used to obtain the second state from the first one. However, these algorithms have never been explicitly compared on equal footing with same models and datasets, making it difficult to assess their scalability and decide which one to select in practice. In this work, we carry out a comparison of seven learning algorithms, namely CL and different variants of EP and CpL depending on the signs of the perturbations. Specifically, using these learning algorithms, we train deep convolutional Hopfield networks (DCHNs) on five vision tasks (MNIST, F-MNIST, SVHN, CIFAR-10 and CIFAR-100). We find that, while all algorithms yield comparable performance on MNIST, important differences in performance arise as the difficulty of the task increases. Our key findings reveal that negative perturbations are better than positive ones, and highlight the centered variant of EP (which uses two perturbations of opposite sign) as the best-performing algorithm. We also endorse these findings with theoretical arguments. Additionally, we establish new SOTA results with DCHNs on all five datasets, both in performance and speed. In particular, our DCHN simulations are 13.5 times faster with respect to Laborieux et al. (2021), which we achieve thanks to the use of a novel energy minimisation algorithm based on asynchronous updates, combined with reduced precision (16 bits).</details>

### Enhancing Robot Program Synthesis Through Environmental Context

**Authors:** Tianyi Chen, Qidi Wang, Zhen Dong, Liwei Shen, Xin Peng

**<details><summary>Abstract**</summary>

Program synthesis aims to automatically generate an executable program that conforms to the given specification. Recent advancements have demonstrated that deep neural methodologies and large-scale pretrained language models are highly proficient in capturing program semantics.For robot programming, prior works have facilitated program synthesis by incorporating global environments. However, the assumption of acquiring a comprehensive understanding of the entire environment is often excessively challenging to achieve.In this work, we present a framework that learns to synthesize a program by rectifying potentially erroneous code segments, with the aid of partially observed environments. To tackle the issue of inadequate attention to partial observations, we propose to first learn an environment embedding space that can implicitly evaluate the impacts of each program token based on the precondition. Furthermore, by employing a graph structure, the model can aggregate both environmental and syntactic information flow and furnish smooth program rectification guidance.Extensive experimental evaluations and ablation studies on the partially observed VizDoom domain authenticate that our method offers superior generalization capability across various tasks and greater robustness when encountering noises.</details>

### Entropy-based Training Methods for Scalable Neural Implicit Samplers

**Authors:** Weijian Luo, Boya Zhang, Zhihua Zhang

**<details><summary>Abstract**</summary>

Efficiently sampling from un-normalized target distributions is a fundamental problem in scientific computing and machine learning. Traditional approaches such as Markov Chain Monte Carlo (MCMC) guarantee asymptotically unbiased samples from such distributions but suffer from computational inefficiency, particularly when dealing with high-dimensional targets, as they require numerous iterations to generate a batch of samples. In this paper, we introduce an efficient and scalable neural implicit sampler that overcomes these limitations. The implicit sampler can generate large batches of samples with low computational costs by leveraging a neural transformation that directly maps easily sampled latent vectors to target samples without the need for iterative procedures. To train the neural implicit samplers, we introduce two novel methods: the KL training method and the Fisher training method. The former method minimizes the Kullback-Leibler divergence, while the latter minimizes the Fisher divergence between the sampler and the target distributions. By employing the two training methods, we effectively optimize the neural implicit samplers to learn and generate from the desired target distribution. To demonstrate the effectiveness, efficiency, and scalability of our proposed samplers, we evaluate them on three sampling benchmarks with different scales. These benchmarks include sampling from 2D targets, Bayesian inference, and sampling from high-dimensional energy-based models (EBMs). Notably, in the experiment involving high-dimensional EBMs, our sampler produces samples that are comparable to those generated by MCMC-based methods while being more than 100 times more efficient, showcasing the efficiency of our neural sampler. Besides the theoretical contributions and strong empirical performances, the proposed neural samplers and corresponding training methods will shed light on further research on developing efficient samplers for various applications beyond the ones explored in this study.</details>

### Environment-Aware Dynamic Graph Learning for Out-of-Distribution Generalization

**Authors:** Haonan Yuan, Qingyun Sun, Xingcheng Fu, Ziwei Zhang, Cheng Ji, Hao Peng, Jianxin Li

**<details><summary>Abstract**</summary>

Dynamic graph neural networks (DGNNs) are increasingly pervasive in exploiting spatio-temporal patterns on dynamic graphs. However, existing works fail to generalize under distribution shifts, which are common in real-world scenarios. As the generation of dynamic graphs is heavily influenced by latent environments, investigating their impacts on the out-of-distribution (OOD) generalization is critical. However, it remains unexplored with the following two major challenges: **(1)** How to properly model and infer the complex environments on dynamic graphs with distribution shifts? **(2)** How to discover invariant patterns given inferred spatio-temporal environments? To solve these challenges, we propose a novel **E**nvironment-**A**ware dynamic **G**raph **LE**arning (**EAGLE**) framework for OOD generalization by modeling complex coupled environments and exploiting spatio-temporal invariant patterns. Specifically, we first design the environment-aware EA-DGNN to model environments by multi-channel environments disentangling. Then, we propose an environment instantiation mechanism for environment diversification with inferred distributions. Finally, we discriminate spatio-temporal invariant patterns for out-of-distribution prediction by the invariant pattern recognition mechanism and perform fine-grained causal interventions node-wisely with a mixture of instantiated environment samples. Experiments on real-world and synthetic dynamic graph datasets demonstrate the superiority of our method against state-of-the-art baselines under distribution shifts. To the best of our knowledge, we are the first to study OOD generalization on dynamic graphs from the environment learning perspective.</details>

### Equivariant Adaptation of Large Pretrained Models

**Authors:** Arnab Kumar Mondal, Siba Smarak Panigrahi, Oumar Kaba, Sai Rajeswar Mudumba, Siamak Ravanbakhsh

**<details><summary>Abstract**</summary>

Equivariant networks are specifically designed to ensure consistent behavior with respect to a set of input transformations, leading to higher sample efficiency and more accurate and robust predictions. However, redesigning each component of prevalent deep neural network architectures to achieve chosen equivariance is a difficult problem and can result in a computationally expensive network during both training and inference. A recently proposed alternative towards equivariance that removes the architectural constraints is to use a simple canonicalization network that transforms the input to a canonical form before feeding it to an unconstrained prediction network. We show here that this approach can effectively be used to make a large pretrained network equivariant. However, we observe that the produced canonical orientations can be misaligned with those of the training distribution, hindering performance. Using dataset-dependent priors to inform the canonicalization function, we are able to make large pretrained models equivariant while maintaining their performance. This significantly improves the robustness of these models to deterministic transformations of the data, such as rotations. We believe this equivariant adaptation of large pretrained models can help their domain-specific applications with known symmetry priors.</details>

### Equivariant Flow Matching with Hybrid Probability Transport for 3D Molecule Generation

**Authors:** Yuxuan Song, Jingjing Gong, Minkai Xu, Ziyao Cao, Yanyan Lan, Stefano Ermon, Hao Zhou, Wei-Ying Ma

**<details><summary>Abstract**</summary>

The generation of 3D molecules requires simultaneously deciding the categorical features (atom types) and continuous features (atom coordinates). Deep generative models, especially Diffusion Models (DMs), have demonstrated effectiveness in generating feature-rich geometries. However, existing DMs typically suffer from unstable probability dynamics with inefficient sampling speed. In this paper, we introduce geometric flow matching, which enjoys the advantages of both equivariant modeling and stabilized probability dynamics. More specifically, we propose a hybrid probability path where the coordinates probability path is regularized by an equivariant optimal transport, and the information between different modalities is aligned. Experimentally, the proposed method could consistently achieve better performance on multiple molecule generation benchmarks with 4.75$\times$ speed up of sampling on average.</details>

### [Spotlight] Error Bounds for Learning with Vector-Valued Random Features

**Authors:** Samuel Lanthaler, Nicholas H. Nelsen

**<details><summary>Abstract**</summary>

This paper provides a comprehensive error analysis of learning with vector-valued random features (RF). The theory is developed for RF ridge regression in a fully general infinite-dimensional input-output setting, but nonetheless applies to and improves existing finite-dimensional analyses. In contrast to comparable work in the literature, the approach proposed here relies on a direct analysis of the underlying risk functional and completely avoids the explicit RF ridge regression solution formula in terms of random matrices. This removes the need for concentration results in random matrix theory or their generalizations to random operators. The main results established in this paper include strong consistency of vector-valued RF estimators under model misspecification and minimax optimal convergence rates in the well-specified setting. The parameter complexity (number of random features) and sample complexity (number of labeled data) required to achieve such rates are comparable with Monte Carlo intuition and free from logarithmic factors.</details>

### Ess-InfoGAIL: Semi-supervised Imitation Learning from Imbalanced Demonstrations

**Authors:** Huiqiao Fu, Kaiqiang Tang, Yuanyang Lu, Yiming Qi, Guizhou Deng, Flood Sung, Chunlin Chen

**<details><summary>Abstract**</summary>

Imitation learning aims to reproduce expert behaviors without relying on an explicit reward signal. However, real-world demonstrations often present challenges, such as multi-modal, data imbalance, and expensive labeling processes. In this work, we propose a novel semi-supervised imitation learning architecture that learns disentangled behavior representations from imbalanced demonstrations using limited labeled data. Specifically, our method consists of three key components. First, we adapt the concept of semi-supervised generative adversarial networks to the imitation learning context. Second, we employ a learnable latent distribution to align the generated and expert data distributions. Finally, we utilize a regularized information maximization approach in conjunction with an approximate label prior to further improve the semi-supervised learning performance. Experimental results demonstrate the efficiency of our method in learning multi-modal behaviors from imbalanced demonstrations compared to baseline methods.</details>

### [Oral] Ethical Considerations for Responsible Data Curation

**Authors:** Jerone Andrews, Dora Zhao, William Thong, Apostolos Modas, Orestis Papakyriakopoulos, Alice Xiang

**Oral Presentation:** Th, Dec 14, 08:30 -- Oral 5B

**<details><summary>Abstract**</summary>

Human-centric computer vision (HCCV) data curation practices often neglect privacy and bias concerns, leading to dataset retractions and unfair models. HCCV datasets constructed through nonconsensual web scraping lack crucial metadata for comprehensive fairness and robustness evaluations. Current remedies are post hoc, lack persuasive justification for adoption, or fail to provide proper contextualization for appropriate application. Our research focuses on proactive, domain-specific recommendations, covering purpose, privacy and consent, and diversity, for curating HCCV evaluation datasets, addressing privacy and bias concerns. We adopt an ante hoc reflective perspective, drawing from current practices, guidelines, dataset withdrawals, and audits, to inform our considerations and recommendations.</details>

### [Oral] Evaluating Post-hoc Explanations for Graph Neural Networks via Robustness Analysis

**Authors:** Junfeng Fang, Wei Liu, Yuan Gao, Zemin Liu, An Zhang, Xiang Wang, Xiangnan He

**Oral Presentation:** Th, Dec 14, 08:30 -- Oral 5A

**<details><summary>Abstract**</summary>

This work studies the evaluation of explaining graph neural networks (GNNs), which is crucial to the credibility of post-hoc explainability in practical usage. Conventional evaluation metrics, and even explanation methods -- which mainly follow the paradigm of feeding the explanatory subgraph and measuring output difference -- always suffer from the notorious out-of-distribution (OOD) issue. In this work, we endeavor to confront the issue by introducing a novel evaluation metric, termed **O**OD-resistant **A**dversarial **R**obustness (OAR). Specifically, we draw inspiration from the notion of adversarial robustness and evaluate post-hoc explanation subgraphs by calculating their robustness under attack. On top of that, an elaborate OOD reweighting block is inserted into the pipeline to confine the evaluation process to the original data distribution. For applications involving large datasets, we further devise a **Sim**plified version of **OAR** (SimOAR), which achieves a significant improvement in computational efficiency at the cost of a small amount of performance. Extensive empirical studies validate the effectiveness of our OAR and SimOAR.</details>

### Evaluating Robustness and Uncertainty of Graph Models Under Structural Distributional Shifts

**Authors:** Gleb Bazhenov, Denis Kuznedelev, Andrey Malinin, Artem Babenko, Liudmila Prokhorenkova

**<details><summary>Abstract**</summary>

In reliable decision-making systems based on machine learning, models have to be robust to distributional shifts or provide the uncertainty of their predictions. In node-level problems of graph learning, distributional shifts can be especially complex since the samples are interdependent. To evaluate the performance of graph models, it is important to test them on diverse and meaningful distributional shifts. However, most graph benchmarks considering distributional shifts for node-level problems focus mainly on node features, while structural properties are also essential for graph problems. In this work, we propose a general approach for inducing diverse distributional shifts based on graph structure. We use this approach to create data splits according to several structural node properties: popularity, locality, and density. In our experiments, we thoroughly evaluate the proposed distributional shifts and show that they can be quite challenging for existing graph models. We also reveal that simple models often outperform more sophisticated methods on the considered structural shifts. Finally, our experiments provide evidence that there is a trade-off between the quality of learned representations for the base classification task under structural distributional shift and the ability to separate the nodes from different distributions using these representations.</details>

### EvoPrompting: Language Models for Code-Level Neural Architecture Search

**Authors:** Angelica Chen, David Dohan, David So

**<details><summary>Abstract**</summary>

Given the recent impressive accomplishments of language models (LMs) for code generation, we explore the use of LMs as general adaptive mutation and crossover operators for an evolutionary neural architecture search (NAS) algorithm.While NAS still proves too difficult a task for LMs to succeed at solely through prompting, we find that the combination of evolutionary prompt engineering with soft prompt-tuning, a method we term EvoPrompting, consistently finds diverse and high performing models. We first demonstrate that EvoPrompting is effective on the computationally efficient MNIST-1D dataset, where EvoPrompting produces convolutional architecture variants that outperform both those designed by human experts and naive few-shot prompting in terms of accuracy and model size. We then apply our method to searching for graph neural networks on the CLRS Algorithmic Reasoning Benchmark, where EvoPrompting is able to design *novel* architectures that outperform current state-of-the-art models on 21 out of 30 algorithmic reasoning tasks while maintaining similar model size. EvoPrompting is successful at designing accurate and efficient neural network architectures across a variety of machine learning tasks, while also being general enough for easy adaptation to other tasks beyond neural network design.</details>

### Exact Generalization Guarantees for (Regularized) Wasserstein Distributionally Robust Models

**Authors:** Waïss Azizian, Franck Iutzeler, Jérôme Malick

**<details><summary>Abstract**</summary>

Wasserstein distributionally robust estimators have emerged as powerful models for prediction and decision-making under uncertainty. These estimators provide attractive generalization guarantees: the robust objective obtained from the training distribution is an exact upper bound on the true risk with high probability. However, existing guarantees either suffer from the curse of dimensionality, are restricted to specific settings, or lead to spurious error terms. In this paper, we show that these generalization guarantees actually hold on general classes of models, do not suffer from the curse of dimensionality, and can even cover distribution shifts at testing. We also prove that these results carry over to the newly-introduced regularized versions of Wasserstein distributionally robust problems.</details>

### [Spotlight] Explore In-Context Learning for 3D Point Cloud Understanding

**Authors:** Zhongbin Fang, Xiangtai Li, Xia Li, Joachim M Buhmann, Chen Change Loy, Mengyuan Liu

**<details><summary>Abstract**</summary>

With the rise of large-scale models trained on broad data, in-context learning has become a new learning paradigm that has demonstrated significant potential in natural language processing and computer vision tasks. Meanwhile, in-context learning is still largely unexplored in the 3D point cloud domain. Although masked modeling has been successfully applied for in-context learning in 2D vision, directly extending it to 3D point clouds remains a formidable challenge. In the case of point clouds, the tokens themselves are the point cloud positions (coordinates) that are masked during inference. Moreover, position embedding in previous works may inadvertently introduce information leakage. To address these challenges, we introduce a novel framework, named Point-In-Context, designed especially for in-context learning in 3D point clouds, where both inputs and outputs are modeled as coordinates for each task. Additionally, we propose the Joint Sampling module, carefully designed to work in tandem with the general point sampling operator, effectively resolving the aforementioned technical issues. We conduct extensive experiments to validate the versatility and adaptability of our proposed methods in handling a wide range of tasks. Furthermore, with a more effective prompt selection strategy, our framework surpasses the results of individually trained models.</details>

### Extensible Prompts for Language Models on Zero-shot Language Style Customization

**Authors:** Tao Ge, Hu Jing, Li Dong, Shaoguang Mao, Yan Xia, Xun Wang, Si-Qing Chen, Furu Wei

**<details><summary>Abstract**</summary>

We propose eXtensible Prompt (X-Prompt) for prompting a large language model (LLM) beyond natural language (NL). X-Prompt instructs an LLM with not only NL but also an extensible vocabulary of imaginary words. Registering new imaginary words allows us to instruct the LLM to comprehend concepts that are difficult to describe with NL words, thereby making a prompt more descriptive. Also, these imaginary words are designed to be out-of-distribution (OOD) robust so that they can be (re)used like NL words in various prompts, distinguishing X-Prompt from soft prompt that is for fitting in-distribution data. We propose context-augmented learning (CAL) to learn imaginary words for general usability, enabling them to work properly in OOD (unseen) prompts. We experiment X-Prompt for zero-shot language style customization as a case study. The promising results of X-Prompt demonstrate its potential to facilitate advanced interaction beyond the natural language interface, bridging the communication gap between humans and LLMs.</details>

### FGPrompt: Fine-grained Goal Prompting for Image-goal Navigation

**Authors:** Xinyu Sun, Peihao Chen, Jugang Fan, Jian Chen, Thomas Li, Mingkui Tan

**<details><summary>Abstract**</summary>

Learning to navigate to an image-specified goal is an important but challenging task for autonomous systems like household robots. The agent is required to well understand and reason the location of the navigation goal from a picture shot in the goal position. Existing methods try to solve this problem by learning a navigation policy, which captures semantic features of the goal image and observation image independently and lastly fuses them for predicting a sequence of navigation actions. However, these methods suffer from two major limitations. 1) They may miss detailed information in the goal image, and thus fail to reason the goal location. 2) More critically, it is hard to focus on the goal-relevant regions in the observation image, because they attempt to understand observation without goal conditioning. In this paper, we aim to overcome these limitations by designing a Fine-grained Goal Prompting (\sexyname) method for image-goal navigation. In particular, we leverage fine-grained and high-resolution feature maps in the goal image as prompts to perform conditioned embedding, which preserves detailed information in the goal image and guides the observation encoder to pay attention to goal-relevant regions. Compared with existing methods on the image-goal navigation benchmark, our method brings significant performance improvement on 3 benchmark datasets (\textit{i.e.,} Gibson, MP3D, and HM3D). Especially on Gibson, we surpass the state-of-the-art success rate by 8\% with only 1/50 model size.</details>

### FIND: A Function Description Benchmark for Evaluating Interpretability Methods

**Authors:** Sarah Schwettmann, Tamar Shaham, Joanna Materzynska, Neil Chowdhury, Shuang Li, Jacob Andreas, David Bau, Antonio Torralba

**<details><summary>Abstract**</summary>

Labeling neural network submodules with human-legible descriptions is useful for many downstream tasks: such descriptions can surface failures, guide interventions, and perhaps even explain important model behaviors. To date, most mechanistic descriptions of trained networks have involved small models, narrowly delimited phenomena, and large amounts of human labor. Labeling all human-interpretable sub-computations in models of increasing size and complexity will almost certainly require tools that can generate and validate descriptions automatically. Recently, techniques that use learned models in-the-loop for labeling have begun to gain traction, but methods for evaluating their efficacy are limited and ad-hoc. How should we validate and compare open-ended labeling tools? This paper introduces FIND (Function INterpretation and Description), a benchmark suite for evaluating the building blocks of automated interpretability methods. FIND contains functions that resemble components of trained neural networks, and accompanying descriptions of the kind we seek to generate. The functions are procedurally constructed across textual and numeric domains, and involve a range of real-world complexities, including noise, composition, approximation, and bias. We evaluate methods that use pretrained language models (LMs) to produce code-based and natural language descriptions of function behavior. Additionally, we introduce a new interactive method in which an Automated Interpretability Agent (AIA) generates function descriptions. We find that an AIA, built with an off-the-shelf LM augmented with black-box access to functions, can sometimes infer function structure—acting as a scientist by forming hypotheses, proposing experiments, and updating descriptions in light of new data. However, FIND also reveals that LM-based descriptions capture global function behavior while missing local details. These results suggest that FIND will be useful for characterizing the performance of more sophisticated interpretability methods before they are applied to real-world models.</details>

### FLuID: Mitigating Stragglers in Federated Learning using Invariant Dropout

**Authors:** Irene Wang, Prashant Nair, Divya Mahajan

**<details><summary>Abstract**</summary>

Federated Learning (FL) allows machine learning models to train locally on individual mobile devices, synchronizing model updates via a shared server. This approach safeguards user privacy; however, it also generates a heterogeneous training environment due to the varying performance capabilities across devices. As a result, “straggler” devices with lower performance often dictate the overalltraining time in FL. In this work, we aim to alleviate this performance bottleneck due to stragglers by dynamically balancing the training load across the system. We introduce Invariant Dropout, a method that extracts a sub-model based on the weight update threshold, thereby minimizing potential impacts on accuracy. Building on this dropout technique, we develop an adaptive training framework, Federated Learning using Invariant Dropout (FLuID). FLuID offers a lightweight sub-model extraction to regulate computational intensity, thereby reducing the load on straggler devices without affecting model quality. Our method leverages neuron updates from non-straggler devices to construct a tailored sub-model for each straggler based on client performance profiling. Furthermore, FLuID can dynamically adapt to changes in stragglers as runtime conditions shift. We evaluate FLuID using five real-world mobile clients. The evaluations show that Invariant Dropout maintains baseline model efficiency while alleviating the performance bottleneck of stragglers through a dynamic, runtime approach.</details>

### FOCAL: Contrastive Learning for Multimodal Time-Series Sensing Signals in Factorized Orthogonal Latent Space

**Authors:** Shengzhong Liu, Tomoyoshi Kimura, Dongxin Liu, Ruijie Wang, Jinyang Li, Suhas Diggavi, Mani Srivastava, Tarek Abdelzaher

**<details><summary>Abstract**</summary>

This paper proposes a novel contrastive learning framework, called FOCAL, for extracting comprehensive features from multimodal time-series sensing signals through self-supervised training. Existing multimodal contrastive frameworks mostly rely on the shared information between sensory modalities, but do not explicitly consider the exclusive modality information that could be critical to understanding the underlying sensing physics. Besides, contrastive frameworks for time series have not handled the temporal information locality appropriately. FOCAL solves these challenges by making the following contributions: First, given multimodal time series, it encodes each modality into a factorized latent space consisting of shared features and private features that are orthogonal to each other. The shared space emphasizes feature patterns consistent across sensory modalities through a modal-matching objective. In contrast, the private space extracts modality-exclusive information through a transformation-invariant objective. Second, we propose a temporal structural constraint for modality features, such that the average distance between temporally neighboring samples is no larger than that of temporally distant samples. Extensive evaluations are performed on four multimodal sensing datasets with two backbone encoders and two classifiers to demonstrate the superiority of FOCAL. It consistently outperforms the state-of-the-art baselines in downstream tasks with a clear margin, under different ratios of available labels. The code and self-collected dataset are available at https://github.com/tomoyoshki/focal.</details>

### False Discovery Proportion control for aggregated Knockoffs

**Authors:** Alexandre Blain, Bertrand Thirion, Olivier Grisel, Pierre Neuvial

**<details><summary>Abstract**</summary>

Controlled variable selection is an important analytical step in various scientific fields, such as brain imaging or genomics. In these high-dimensional data settings, considering too many variables leads to poor models and high costs, hence the need for statistical guarantees on false positives. Knockoffs are a popular statistical tool for conditional variable selection in high dimension. However, they control for the expected proportion of false discoveries (FDR) and not the actual proportion of false discoveries (FDP). We present a new method, KOPI, that controls the proportion of false discoveries for Knockoff-based inference. The proposed method also relies on a new type of aggregation to address the undesirable randomness associated with classical Knockoff inference. We demonstrate FDP control and substantial power gains over existing Knockoff-based methods in various simulation settings and achieve good sensitivity/specificity tradeoffs on brain imaging data.</details>

### Fast Optimal Transport through Sliced Generalized Wasserstein Geodesics

**Authors:** Guillaume Mahey, Laetitia Chapel, Gilles Gasso, Clément Bonet, Nicolas Courty

**<details><summary>Abstract**</summary>

Wasserstein distance (WD) and the associated optimal transport plan have been proven useful in many applications where probability measures are at stake. In this paper, we propose a new proxy of the squared WD, coined $\textnormal{min-SWGG}$, that is based on the transport map induced by an optimal one-dimensional projection of the two input distributions. We draw connections between  $\textnormal{min-SWGG}$, and Wasserstein generalized geodesics in which the pivot measure is supported on a line. We notably provide a new closed form for the exact Wasserstein distance in the particular case of one of the distributions supported on a line allowing us to derive a fast computational scheme that is amenable to gradient descent optimization. We show that  $\textnormal{min-SWGG}$, is an upper bound of WD and that it has a complexity similar to as Sliced-Wasserstein, with the additional feature of providing an associated transport plan. We also investigate some theoretical properties such as metricity, weak convergence, computational and topological properties. Empirical evidences support the benefits of  $\textnormal{min-SWGG}$, in various contexts, from gradient flows, shape matching and image colorization, among others.</details>

### Fast Partitioned Learned Bloom Filter

**Authors:** Atsuki Sato, Yusuke Matsui

**<details><summary>Abstract**</summary>

A Bloom filter is a memory-efficient data structure for approximate membership queries used in numerous fields of computer science.Recently, learned Bloom filters that achieve better memory efficiency using machine learning models have attracted attention.One such filter, the partitioned learned Bloom filter (PLBF), achieves excellent memory efficiency.However, PLBF requires a $\mathcal{O}(N^3k)$ time complexity to construct the data structure, where $N$ and $k$ are the hyperparameters of PLBF.One can improve memory efficiency by increasing $N$, but the construction time becomes extremely long.Thus, we propose two methods that can reduce the construction time while maintaining the memory efficiency of PLBF.First, we propose fast PLBF, which can construct the same data structure as PLBF with a smaller time complexity $\mathcal{O}(N^2k)$.Second, we propose fast PLBF++, which can construct the data structure with even smaller time complexity $\mathcal{O}(Nk\log N + Nk^2)$.Fast PLBF++ does not necessarily construct the same data structure as PLBF.Still, it is almost as memory efficient as PLBF, and it is proved that fast PLBF++ has the same data structure as PLBF when the distribution satisfies a certain constraint.Our experimental results from real-world datasets show that (i) fast PLBF and fast PLBF++ can construct the data structure up to 233 and 761 times faster than PLBF, (ii) fast PLBF can achieve the same memory efficiency as PLBF, and (iii) fast PLBF++ can achieve almost the same memory efficiency as PLBF.The codes are available at [this https URL](https://github.com/atsukisato/FastPLBF).</details>

### Faster Differentially Private Convex Optimization via Second-Order Methods

**Authors:** Arun Ganesh, Mahdi Haghifam, Thomas Steinke, Abhradeep Guha Thakurta

**<details><summary>Abstract**</summary>

Differentially private (stochastic) gradient descent is the workhorse of DP private machine learning in both the convex and non-convex settings. Without privacy constraints, second-order methods, like Newton's method, converge faster than first-order methods like gradient descent. In this work, we investigate the prospect of using the second-order information from the loss function to accelerate DP convex optimization. We first develop a private variant of the regularized cubic Newton method of Nesterov and Polyak, and show that for the class of strongly convex loss functions, our algorithm has quadratic convergence and achieves the optimal excess loss. We then design a practical second-order DP algorithm for the unconstrained logistic regression problem. We theoretically and empirically study the performance of our algorithm. Empirical results show our algorithm consistently achieves the best excess loss compared to other baselines and is 10-40x faster than DP-GD/DP-SGD for challenging datasets.</details>

### Feature Likelihood Score: Evaluating the Generalization of Generative Models Using Samples

**Authors:** Marco Jiralerspong, Joey Bose, Ian Gemp, Chongli Qin, Yoram Bachrach, Gauthier Gidel

**<details><summary>Abstract**</summary>

The past few years have seen impressive progress in the development of deep generative models capable of producing high-dimensional, complex, and photo-realistic data. However, current methods for evaluating such models remain incomplete: standard likelihood-based metrics do not always apply and rarely correlate with perceptual fidelity, while sample-based metrics, such as FID, are insensitive to overfitting, i.e., inability to generalize beyond the training set. To address these limitations, we propose a new metric called the Feature Likelihood Score (FLS), a parametric sample-based score that uses density estimation to provide a comprehensive trichotomic evaluation accounting for novelty (i.e., different from the training samples), fidelity, and diversity of generated samples.  We empirically demonstrate the ability of FLS to identify specific overfitting problem cases, where previously proposed metrics fail. We also extensively evaluate FLS on various image datasets and model classes, demonstrating its ability to match intuitions of previous metrics like FID while offering a more comprehensive evaluation of generative models.</details>

### FedFed: Feature Distillation against Data Heterogeneity in Federated Learning

**Authors:** Zhiqin Yang, Yonggang Zhang, Yu Zheng, Xinmei Tian, Hao Peng, Tongliang Liu, Bo Han

**<details><summary>Abstract**</summary>

Federated learning (FL) typically faces data heterogeneity, i.e., distribution shifting among clients. Sharing clients' information has shown great potentiality in mitigating data heterogeneity, yet incurs a dilemma in preserving privacy and promoting model performance. To alleviate the dilemma, we raise a fundamental question: Is it possible to share partial features in the data to tackle data heterogeneity?In this work, we give an affirmative answer to this question by proposing a novel approach called **Fed**erated **Fe**ature **d**istillation (FedFed).Specifically, FedFed partitions data into performance-sensitive features (i.e., greatly contributing to model performance) and performance-robust features (i.e., limitedly contributing to model performance).The performance-sensitive features are globally shared to mitigate data heterogeneity, while the performance-robust features are kept locally.FedFed enables clients to train models over local and shared data. Comprehensive experiments demonstrate the efficacy of FedFed in promoting model performance.</details>

### FedGCN: Convergence-Communication Tradeoffs in Federated Training of Graph Convolutional Networks

**Authors:** Yuhang Yao, Weizhao Jin, Srivatsan Ravi, Carlee Joe-Wong

**<details><summary>Abstract**</summary>

Methods for training models on graphs distributed across multiple clients have recently grown in popularity, due to the size of these graphs as well as regulations on keeping data where it is generated. However, the cross-client edges naturally exist among clients. Thus, distributed methods for training a model on a single graph incur either significant communication overhead between clients or a loss of available information to the training. We introduce the Federated Graph Convolutional Network (FedGCN) algorithm, which uses federated learning to train GCN models for semi-supervised node classification with fast convergence and little communication. Compared to prior methods that require extra communication among clients at each training round, FedGCN clients only communicate with the central server in one pre-training step, greatly reducing communication costs and allowing the use of homomorphic encryption to further enhance privacy. We theoretically analyze the tradeoff between FedGCN's convergence rate and communication cost under different data distributions. Experimental results show that our FedGCN algorithm achieves better model accuracy with 51.7\% faster convergence on average and at least 100$\times$ less communication compared to prior work.</details>

### FedGame: A Game-Theoretic Defense against Backdoor Attacks in Federated Learning

**Authors:** Jinyuan Jia, Zhuowen Yuan, Dinuka Sahabandu, Luyao Niu, Arezoo Rajabi, Bhaskar Ramasubramanian, Bo Li, Radha Poovendran

**<details><summary>Abstract**</summary>

Federated learning (FL) provides a distributed training paradigm where multiple clients can jointly train a global model without sharing their local data. However, recent studies have shown that FL offers an additional surface for backdoor attacks. For instance, an attacker can compromise a subset of clients and thus corrupt the global model to misclassify an input with a backdoor trigger as the adversarial target. Existing defenses for FL against backdoor attacks usually detect and exclude the corrupted information from the compromised clients based on a static attacker model. However, such defenses are inadequate against dynamic attackers who strategically adapt their attack strategies. To bridge this gap, we model the strategic interactions between the defender and dynamic attackers as a minimax game. Based on the analysis of the game, we design an interactive defense mechanism FedGame. We prove that under mild assumptions, the global model trained with FedGame under backdoor attacks is close to that trained without attacks. Empirically, we compare FedGame with multiple state-of-the-art baselines on several benchmark datasets under various attacks. We show that FedGame can effectively defend against strategic attackers and achieves significantly higher robustness than baselines. Our code is available at: https://github.com/AI-secure/FedGame.</details>

### Finding Safe Zones of Markov Decision Processes Policies

**Authors:** Lee Cohen, Yishay Mansour, Michal Moshkovitz

**<details><summary>Abstract**</summary>

Given a policy of a Markov Decision Process, we define a SafeZone as a subset of states, such that most of the policy's trajectories are confined to this subset. The quality of a SafeZone is parameterized by the number of states and the escape probability, i.e., the probability that a random trajectory will leave the subset. SafeZones are especially interesting when they have a small number of states and low escape probability. We study the complexity of finding optimal SafeZones, and show that in general, the problem is computationally hard. For this reason, we concentrate on finding approximate SafeZones. Our main result is a bi-criteria approximation learning algorithm with a factor of almost $2$  approximation for both the escape probability and \newprob size, using a polynomial size sample complexity.</details>

### Fine-Grained Visual Prompting

**Authors:** Lingfeng Yang, Yueze Wang, Xiang Li, Xinlong Wang, Jian Yang

**<details><summary>Abstract**</summary>

Vision-Language Models (VLMs), such as CLIP, have demonstrated impressive zero-shot transfer capabilities in image-level visual perception. However, these models have shown limited performance in instance-level tasks that demand precise localization and recognition. Previous works have suggested that incorporating visual prompts, such as colorful boxes or circles, can improve the ability of models to recognize objects of interest. Nonetheless, compared to language prompting, visual prompting designs are rarely explored. Existing approaches, which employ coarse visual cues such as colorful boxes or circles, often result in sub-optimal performance due to the inclusion of irrelevant and noisy pixels. In this paper, we carefully study the visual prompting designs by exploring more fine-grained markings, such as segmentation masks and their variations. In addition, we introduce a new zero-shot framework that leverages pixel-level annotations acquired from a generalist segmentation model for fine-grained visual prompting. Consequently, our investigation reveals that a straightforward application of blur outside the target mask, referred to as the Blur Reverse Mask, exhibits exceptional effectiveness. This proposed prompting strategy leverages the precise mask annotations to reduce focus on weakly related regions while retaining spatial coherence between the target and the surrounding background. Our **F**ine-**G**rained **V**isual **P**rompting (**FGVP**) demonstrates superior performance in zero-shot comprehension of referring expressions on the RefCOCO, RefCOCO+, and RefCOCOg benchmarks. It outperforms prior methods by an average margin of 3.0\% to 4.6\%, with a maximum improvement of 12.5\% on the RefCOCO+ testA subset. The part detection experiments conducted on the PACO dataset further validate the preponderance of FGVP over existing visual prompting techniques. Code is available at https://github.com/ylingfeng/FGVP.</details>

### Flat Seeking Bayesian Neural Networks

**Authors:** Van-Anh Nguyen, Tung-Long Vuong, Hoang Phan, Thanh-Toan Do, Dinh Phung, Trung Le

**<details><summary>Abstract**</summary>

Bayesian Neural Networks (BNNs) provide a probabilistic interpretation for deep learning models by imposing a prior distribution over model parameters and inferring a posterior distribution based on observed data. The model sampled from the posterior distribution can be used for providing ensemble predictions and quantifying prediction uncertainty. It is well-known that deep learning models with lower sharpness have better generalization ability. However, existing posterior inferences are not aware of sharpness/flatness in terms of formulation, possibly leading to high sharpness for the models sampled from them. In this paper, we develop theories, the Bayesian setting, and the variational inference approach for the sharpness-aware posterior. Specifically, the models sampled from our sharpness-aware posterior, and the optimal approximate posterior estimating this sharpness-aware posterior, have better flatness, hence possibly possessing higher generalization ability. We conduct experiments by leveraging the sharpness-aware posterior with state-of-the-art Bayesian Neural Networks, showing that the flat-seeking counterparts outperform their baselines in all metrics of interest.</details>

### Flexible Attention-Based Multi-Policy Fusion for Efficient Deep Reinforcement Learning

**Authors:** Zih-Yun Chiu, Yi-Lin Tuan, William Yang Wang, Michael Yip

**<details><summary>Abstract**</summary>

Reinforcement learning (RL) agents have long sought to approach the efficiency of human learning. Humans are great observers who can learn by aggregating external knowledge from various sources, including observations from others' policies of attempting a task. Prior studies in RL have incorporated external knowledge policies to help agents improve sample efficiency. However, it remains non-trivial to perform arbitrary combinations and replacements of those policies, an essential feature for generalization and transferability. In this work, we present Knowledge-Grounded RL (KGRL), an RL paradigm fusing multiple knowledge policies and aiming for human-like efficiency and flexibility. We propose a new actor architecture for KGRL, Knowledge-Inclusive Attention Network (KIAN), which allows free knowledge rearrangement due to embedding-based attentive action prediction. KIAN also addresses entropy imbalance, a problem arising in maximum entropy KGRL that hinders an agent from efficiently exploring the environment, through a new design of policy distributions. The experimental results demonstrate that KIAN outperforms alternative methods incorporating external knowledge policies and achieves efficient and flexible learning. Our implementation is available at https://github.com/Pascalson/KGRL.git .</details>

### Flow-Attention-based Spatio-Temporal Aggregation Network for 3D Mask Detection

**Authors:** Yuxin Cao, Yian Li, Yumeng Zhu, Derui Wang, Minhui Xue

**<details><summary>Abstract**</summary>

Anti-spoofing detection has become a necessity for face recognition systems due to the security threat posed by spoofing attacks. Despite great success in traditional attacks, most deep-learning-based methods perform poorly in 3D masks, which can highly simulate real faces in appearance and structure, suffering generalizability insufficiency while focusing only on the spatial domain with single frame input. This has been mitigated by the recent introduction of a biomedical technology called rPPG (remote photoplethysmography). However, rPPG-based methods are sensitive to noisy interference and require at least one second (> 25 frames) of observation time, which induces high computational overhead. To address these challenges, we propose a novel 3D mask detection framework, called FASTEN (Flow-Attention-based Spatio-Temporal aggrEgation Network). We tailor the network for focusing more on fine-grained details in large movements, which can eliminate redundant spatio-temporal feature interference and quickly capture splicing traces of 3D masks in fewer frames. Our proposed network contains three key modules: 1) a facial optical flow network to obtain non-RGB inter-frame flow information; 2) flow attention to assign different significance to each frame; 3) spatio-temporal aggregation to aggregate high-level spatial features and temporal transition features. Through extensive experiments, FASTEN only requires five frames of input and outperforms eight competitors for both intra-dataset and cross-dataset evaluations in terms of multiple detection metrics. Moreover, FASTEN has been deployed in real-world mobile devices for practical 3D mask detection.</details>

### Flow-Based Feature Fusion for Vehicle-Infrastructure Cooperative 3D Object Detection

**Authors:** Haibao Yu, Yingjuan Tang, Enze Xie, Jilei Mao, Ping Luo, Zaiqing Nie

**<details><summary>Abstract**</summary>

Cooperatively utilizing both ego-vehicle and infrastructure sensor data can significantly enhance autonomous driving perception abilities. However, the uncertain temporal asynchrony and limited communication conditions that are present in traffic environments can lead to fusion misalignment and constrain the exploitation of infrastructure data. To address these issues in vehicle-infrastructure cooperative 3D (VIC3D) object detection, we propose the Feature Flow Net (FFNet), a novel cooperative detection framework. FFNet is a flow-based feature fusion framework that uses a feature flow prediction module to predict future features and compensate for asynchrony. Instead of transmitting feature maps extracted from still-images, FFNet transmits feature flow, leveraging the temporal coherence of sequential infrastructure frames. Furthermore, we introduce a self-supervised training approach that enables FFNet to generate feature flow with feature prediction ability from raw infrastructure sequences. Experimental results demonstrate that our proposed method outperforms existing cooperative detection methods while only requiring about 1/100 of the transmission cost of raw data and covers all latency in one model on the DAIR-V2X dataset. The code is available https://github.com/haibao-yu/FFNet-VIC3D.</details>

### FlowPG: Action-constrained Policy Gradient with Normalizing Flows

**Authors:** Janaka Brahmanage, Jiajing LING, Akshat Kumar

**<details><summary>Abstract**</summary>

Action-constrained reinforcement learning (ACRL) is a popular approach for solving safety-critical and resource-allocation related decision making problems. A major challenge in ACRL is to ensure agent taking a valid action satisfying constraints in each RL step. Commonly used approach of using a projection layer on top of the policy network requires solving an optimization program which can result in longer training time, slow convergence, and zero gradient problem. To address this, first we use a normalizing flow model to learn an invertible, differentiable mapping between the feasible action space and the support of a simple distribution on a latent variable, such as Gaussian. Second, learning the flow model requires sampling from the feasible action space, which is also challenging. We develop multiple methods, based on Hamiltonian Monte-Carlo and probabilistic sentential decision diagrams for such action sampling for convex and non-convex constraints. Third, we integrate the learned normalizing flow with the DDPG algorithm. By design, a well-trained normalizing flow will transform policy output into a valid action without requiring an optimization solver. Empirically, our approach results in significantly fewer constraint violations (upto an order-of-magnitude for several instances) and is multiple times faster on a variety of continuous control tasks.</details>

### Focus on Query: Adversarial Mining Transformer for Few-Shot Segmentation

**Authors:** Yuan Wang, Naisong Luo, Tianzhu Zhang

**<details><summary>Abstract**</summary>

Few-shot segmentation (FSS) aims to segment objects of new categories given only a handful of annotated samples. Previous works focus their efforts on exploring the support information while paying less attention to the mining of the critical query branch. In this paper, we rethink the importance of support information and propose a new query-centric FSS model Adversarial Mining Transformer (AMFormer), which achieves accurate query image segmentation with only rough support guidance or even weak support labels. The proposed AMFormer enjoys several merits. First, we design an object mining transformer (G) that can achieve the expansion of incomplete region activated by support clue, and a detail mining transformer (D) to discriminate the detailed local difference between the expanded mask and the ground truth. Second, we propose to train G and D via an adversarial process, where G is optimized to generate more accurate masks approaching ground truth to fool D. We conduct extensive experiments on commonly used Pascal-5i and COCO-20i benchmarks and achieve state-of-the-art results across all settings. In addition, the decent performance with weak support labels in our query-centric paradigm may inspire the development of more general FSS models.</details>

### Focused Transformer: Contrastive Training for Context Scaling

**Authors:** Szymon Tworkowski, Konrad Staniszewski, Mikołaj Pacek, Yuhuai Wu, Henryk Michalewski, Piotr Miłoś

**<details><summary>Abstract**</summary>

Large language models have an exceptional capability to incorporate new information in a contextual manner. However, the full potential of such an approach is often restrained due to a limitation in the effective context length. One solution to this issue is to endow an attention layer with access to an additional context, which comprises of (key, value) pairs. Yet, as the number of documents increases, the proportion of relevant keys to irrelevant ones decreases, leading the model to focus more on the irrelevant keys. We identify a significant challenge, dubbed the distraction issue, where keys linked to different semantic values might overlap, making them hard to distinguish. To tackle this problem, we introduce the Focused Transformer (FoT), a technique that employs a training process inspired by contrastive learning. This novel approach enhances the structure of the (key, value) space, enabling an extension of the context length. Our method allows for fine-tuning pre-existing, large-scale models to lengthen their effective context. This is demonstrated by our fine-tuning of $3 B$ and $7 B$ OpenLLaMA checkpoints. The resulting models, which we name LongLLaMA, exhibit advancements in tasks requiring a long context. We further illustrate that our LongLLaMA models adeptly manage a $256 k$ context length for passkey retrieval.</details>

### [Spotlight] Follow-ups Also Matter: Improving Contextual Bandits via Post-serving Contexts

**Authors:** Chaoqi Wang, Ziyu Ye, Zhe Feng, Haifeng Xu

**<details><summary>Abstract**</summary>

Standard contextual bandit problem assumes that all the relevant contexts are observed before the algorithm chooses an arm. This modeling paradigm, while useful, often falls short when dealing with problems in which additional valuable contexts can be observed after arm selection. For example, content recommendation platforms like Youtube, Instagram, Tiktok receive much additional features about a user's reward after the user clicks a content (e.g., how long the user stayed, what is the user's watch speed, etc.). To improve online learning efficiency in these applications,  we  study a novel contextual bandit problem with post-serving contexts and design a new algorithm, poLinUCB,  that achieves tight regret under standard assumptions. Core to our technical proof is a robustified and generalized version of the well-known Elliptical Potential Lemma (EPL), which can accommodate  noise in data. Such robustification is necessary for tackling our problem, though we believe it could also be of general interest.Extensive empirical tests on both synthetic and real-world datasets  demonstrate the significant benefit of utilitzing post-serving contexts as well as the superior performance of  our   algorithm over the state-of-the-art approaches.</details>

### Formalizing locality for normative synaptic plasticity models

**Authors:** Colin Bredenberg, Ezekiel Williams, Cristina Savin, Blake Richards, Guillaume Lajoie

**<details><summary>Abstract**</summary>

In recent years, many researchers have proposed new models for synaptic plasticity in the brain based on principles of machine learning. The central motivation has been the development of learning algorithms that are able to learn difficult tasks while qualifying as "biologically plausible". However, the concept of a biologically plausible learning algorithm is only heuristically defined as an algorithm that is potentially implementable by biological neural networks. Further, claims that neural circuits could implement any given algorithm typically rest on an amorphous concept of "locality" (both in space and time). As a result, it is unclear what many proposed local learning algorithms actually predict biologically, and which of these are consequently good candidates for experimental investigation. Here, we address this lack of clarity by proposing formal and operational definitions of locality. Specifically, we define different classes of locality, each of which makes clear what quantities cannot be included in a learning rule if an algorithm is to qualify as local with respect to a given (biological) constraint. We subsequently use this framework to distill testable predictions from various classes of biologically plausible synaptic plasticity models that are robust to arbitrary choices about neural network architecture. Therefore, our framework can be used to guide claims of biological plausibility and to identify potential means of experimentally falsifying a proposed learning algorithm for the brain.</details>

### FouriDown: Factoring Down-Sampling into Shuffling and Superposing

**Authors:** Qi Zhu, man zhou, Jie Huang, Naishan Zheng, Hongzhi Gao, Chongyi Li, Yuan Xu, Feng Zhao

**<details><summary>Abstract**</summary>

Spatial down-sampling techniques, such as strided convolution, Gaussian, and Nearest down-sampling, are essential in deep neural networks. In this study, we revisit the working mechanism of the spatial down-sampling family and analyze the biased effects caused by the static weighting strategy employed in previous approaches. To overcome this limitation, we propose a novel  down-sampling paradigm in the Fourier domain, abbreviated as FouriDown, which unifies existing down-sampling techniques. Drawing inspiration from the signal sampling theorem, we parameterize the non-parameter static weighting down-sampling operator as a learnable and context-adaptive operator within a unified Fourier function. Specifically, we organize the corresponding frequency positions of the 2D plane in a physically-closed manner within a single channel dimension. We then perform point-wise channel shuffling based on an indicator that determines whether a channel's signal frequency bin is susceptible to aliasing, ensuring the consistency of the weighting parameter learning. FouriDown, as a generic operator, comprises four key components: 2D discrete Fourier transform, context shuffling rules, Fourier weighting-adaptively superposing rules, and 2D inverse Fourier transform. These components can be easily integrated into existing image restoration networks. To demonstrate the efficacy of FouriDown, we conduct extensive experiments on image de-blurring and low-light image enhancement. The results consistently show that FouriDown can provide significant performance improvements. We will make the code publicly available to facilitate further exploration and application of FouriDown.</details>

### Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator

**Authors:** Hanzhuo Huang, Yufan Feng, Cheng Shi, Lan Xu, Jingyi Yu, Sibei Yang

**<details><summary>Abstract**</summary>

Text-to-video is a rapidly growing research area that aims to generate a semantic, identical, and temporal coherence sequence of frames that accurately align with the input text prompt. This study focuses on zero-shot text-to-video generation considering the data- and cost-efficient. To generate a semantic-coherent video, exhibiting a rich portrayal of temporal semantics such as the whole process of flower blooming rather than a set of ``moving images'', we propose a novel Free-Bloom pipeline that harnesses large language models (LLMs) as the director to generate a semantic-coherence prompt sequence, while pre-trained latent diffusion models (LDMs) as the animator to generate the high fidelity frames. Furthermore, to ensure temporal and identical coherence while maintaining semantic coherence, we propose a series of annotative modifications to adapting LDMs in the reverse process, including joint noise sampling, step-aware attention shift, and dual-path interpolation. Without any video data and training requirements, Free-Bloom generates vivid and high-quality videos, awe-inspiring in generating complex scenes with semantic meaningful frame sequences.  In addition, Free-Bloom is naturally compatible with LDMs-based extensions.</details>

### Frequency-Enhanced Data Augmentation for Vision-and-Language Navigation

**Authors:** Keji He, Chenyang Si, Zhihe Lu, Yan Huang, Liang Wang, Xinchao Wang

**<details><summary>Abstract**</summary>

Vision-and-Language Navigation (VLN) is a challenging task that requires an agent to navigate through complex environments based on natural language instructions. In contrast to conventional approaches, which primarily focus on the spatial domain exploration, we propose a paradigm shift toward the Fourier domain. This alternative perspective aims to enhance visual-textual matching, ultimately improving the agent's ability to understand and execute navigation tasks based on the given instructions. In this study, we first explore the significance of high-frequency information in VLN and provide evidence that it is instrumental in bolstering visual-textual matching processes. Building upon this insight, we further propose a sophisticated and versatile Frequency-enhanced Data Augmentation (FDA) technique to improve the VLN model's capability of capturing critical high-frequency information. Specifically, this approach requires the agent to navigate in environments where only a subset of high-frequency visual information corresponds with the provided textual instructions, ultimately fostering the agent's ability to selectively discern and capture pertinent high-frequency features according to the given instructions. Promising results on R2R, RxR, CVDN and REVERIE demonstrate that our FDA can be readily integrated with existing VLN approaches, improving performance without adding extra parameters, and keeping models simple and efficient. The code is available at https://github.com/hekj/FDA.</details>

### Frequency-domain MLPs are More Effective Learners in Time Series Forecasting

**Authors:** Kun Yi, Qi Zhang, Wei Fan, Shoujin Wang, Pengyang Wang, Hui He, Ning An, Defu Lian, Longbing Cao, Zhendong Niu

**<details><summary>Abstract**</summary>

Time series forecasting has played the key role in different industrial, including finance, traffic, energy, and healthcare domains. While existing literatures have designed many sophisticated architectures based on RNNs, GNNs, or Transformers, another kind of approaches based on multi-layer perceptrons (MLPs) are proposed with simple structure, low complexity, and superior performance. However, most MLP-based forecasting methods suffer from the point-wise mappings and information bottleneck, which largely hinders the forecasting performance. To overcome this problem, we explore a novel direction of applying MLPs in the frequency domain for time series forecasting. We investigate the learned patterns of frequency-domain MLPs and discover their two inherent characteristic benefiting forecasting, (i) global view: frequency spectrum makes MLPs own a complete view for signals and learn global dependencies more easily, and (ii) energy compaction: frequency-domain MLPs concentrate on smaller key part of frequency components with compact signal energy. Then, we propose FreTS, a simple yet effective architecture built upon Frequency-domain MLPs for Time Series forecasting. FreTS mainly involves two stages, (i) Domain Conversion, that transforms time-domain signals into complex numbers of frequency domain; (ii) Frequency Learning, that performs our redesigned MLPs for the learning of real and imaginary part of frequency components. The above stages operated on both inter-series and intra-series scales further contribute to channel-wise and time-wise dependency learning. Extensive experiments on 13 real-world benchmarks (including 7 benchmarks for short-term forecasting and 6 benchmarks for long-term forecasting) demonstrate our consistent superiority over state-of-the-art methods. Code is available at this repository: https://github.com/aikunyi/FreTS.</details>

### From Trainable Negative Depth to Edge Heterophily in Graphs

**Authors:** Yuchen Yan, Yuzhong Chen, Huiyuan Chen, Minghua Xu, Mahashweta Das, Hao Yang, Hanghang Tong

**<details><summary>Abstract**</summary>

Finding the proper depth $d$ of a graph convolutional network (GCN) that provides strong representation ability has drawn significant attention, yet nonetheless  largely  remains an open problem for the graph learning community. Although noteworthy progress has been made, the depth or the number of layers of a corresponding GCN is realized by a series of graph convolution operations, which naturally makes $d$ a positive integer ($d \in \mathbb{N}+$). An interesting question is whether breaking the constraint of $\mathbb{N}+$ by making $d$ a real number ($d \in \mathbb{R}$) can bring new insights into graph learning mechanisms. In this work, by redefining GCN's depth $d$ as a trainable parameter continuously adjustable within $(-\infty,+\infty)$, we open a new door of controlling its signal processing capability to model graph homophily/heterophily (nodes with similar/dissimilar labels/attributes tend to be inter-connected). A simple and powerful GCN model TEDGCN, is proposed to retain the simplicity of GCN and meanwhile automatically search for the optimal $d$ without the prior knowledge regarding whether the input graph is homophilic or heterophilic. Negative-valued $d$ intrinsically enables high-pass frequency filtering functionality via augmented topology for graph heterophily. Extensive experiments demonstrate the superiority of TEDGCN on node classification tasks for a variety of homophilic and heterophilic graphs.</details>

### From ViT Features to Training-free Video Object Segmentation via Streaming-data Mixture Models

**Authors:** Roy Uziel, Or Dinari, Oren Freifeld

**<details><summary>Abstract**</summary>

In the task of semi-supervised video object segmentation, the input is the binary mask of an object in the first frame, and the desired output consists of the corresponding masks of that object in the subsequent frames. Existing leading solutions have two main drawbacks: 1) an expensive and typically-supervised training on videos; 2) a large memory footprint during inference. Here we present a training-free solution, with a low-memory footprint, that yields state-of-the-art results. The proposed method combines pre-trained deep learning-based features (trained on still images) with more classical methods for streaming-data clustering. Designed to adapt to temporal concept drifts and generalize to diverse video content without relying on annotated images or videos, the method eliminates the need for additional training or fine-tuning, ensuring fast inference and immediate applicability to new videos. Concretely, we represent an object via a dynamic ensemble of temporally- and spatially-coherent mixtures over a representation built from pre-trained ViT features and positional embeddings. A convolutional conditional random field further improves spatial coherence and helps reject outliers. We demonstrate the efficacy of the method on key benchmarks: the DAVIS-2017 and YouTube-VOS 2018 validation datasets. Moreover, by the virtue of the low-memory footprint of the compact cluster-based representation, the method scales gracefully to high-resolution ViT features. Our code is available at https://github.com/BGU-CS-VIL/Training-Free-VOS</details>

### Front-door Adjustment Beyond Markov Equivalence with Limited Graph Knowledge

**Authors:** Abhin Shah, Karthikeyan Shanmugam, Murat Kocaoglu

**<details><summary>Abstract**</summary>

Causal effect estimation from data typically requires assumptions about the cause-effect relations either explicitly in the form of a causal graph structure within the Pearlian framework, or implicitly in terms of (conditional) independence statements between counterfactual variables within the potential outcomes framework. When the treatment variable and the outcome variable are confounded, front-door adjustment is an important special case where, given the graph, causal effect of the treatment on the target can be estimated using post-treatment variables. However, the exact formula for front-door adjustment depends on the structure of the graph, which is difficult to learn in practice. In this work, we provide testable conditional independence statements to compute the causal effect using front-door-like adjustment without knowing the graph under limited structural side information. We show that our method is applicable in scenarios where knowing the Markov equivalence class is not sufficient for causal effect estimation. We demonstrate the effectiveness of our method on a class of random graphs as well as real causal fairness benchmarks.</details>

### Functional-Group-Based Diffusion for Pocket-Specific Molecule Generation and Elaboration

**Authors:** Haitao Lin, Yufei Huang, Odin Zhang, Yunfan Liu, Lirong Wu, Siyuan Li, Zhiyuan Chen, Stan Z. Li

**<details><summary>Abstract**</summary>

In recent years, AI-assisted drug design methods have been proposed to generate molecules given the pockets' structures of target proteins. Most of them are  {\em atom-level-based} methods, which consider atoms as basic components and generate atom positions and types. In this way, however, it is hard to generate realistic fragments with complicated structures. To solve this, we propose \textsc{D3FG}, a {\em functional-group-based} diffusion model for pocket-specific molecule generation and elaboration. \textsc{D3FG} decomposes molecules into two categories of components: functional groups defined as rigid bodies and linkers as mass points. And the two kinds of components can together form complicated fragments that enhance ligand-protein interactions. To be specific, in the diffusion process, \textsc{D3FG} diffuses the data distribution of the positions, orientations, and types of the components into a prior distribution; In the generative process, the noise is gradually removed from the three variables by denoisers parameterized with designed equivariant graph neural networks.  In the experiments, our method can generate molecules with more realistic 3D structures, competitive affinities toward the protein targets, and better drug properties. Besides, \textsc{D3FG} as a solution to a new task of molecule elaboration, could generate molecules with high affinities based on existing ligands and the hotspots of target proteins.</details>

### GALOPA: Graph Transport Learning with Optimal Plan Alignment

**Authors:** Yejiang Wang, Yuhai Zhao, Daniel Zhengkui Wang, Ling Li

**<details><summary>Abstract**</summary>

Self-supervised learning on graph aims to learn graph representations in an unsupervised manner. While graph contrastive learning (GCL - relying on graph augmentation for creating perturbation views of anchor graphs and maximizing/minimizing similarity for positive/negative pairs) is a popular self-supervised method, it faces challenges in finding label-invariant augmented graphs and determining the exact extent of similarity between sample pairs to be achieved. In this work, we propose an alternative self-supervised solution that (i) goes beyond the label invariance assumption without distinguishing between positive/negative samples, (ii) can calibrate the encoder for preserving not only the structural information inside the graph, but the matching information between different graphs, (iii) learns isometric embeddings that preserve the distance between graphs, a by-product of our objective. Motivated by optimal transport theory, this scheme relays on an observation that the optimal transport plans between node representations at the output space, which measure the matching probability between two distributions, should be consistent to the plans between the corresponding graphs at the input space. The experimental findings include: (i) The plan alignment strategy significantly outperforms the counterpart using the transport distance; (ii) The proposed model shows superior performance using only node attributes as calibration signals, without relying on edge information; (iii) Our model maintains robust results even under high perturbation rates; (iv) Extensive experiments on various benchmarks validate the effectiveness of the proposed method.</details>

### GNNEvaluator: Evaluating GNN Performance On Unseen Graphs Without Labels

**Authors:** Xin Zheng, Miao Zhang, Chunyang Chen, Soheila Molaei, Chuan Zhou, Shirui Pan

**<details><summary>Abstract**</summary>

Evaluating the performance of graph neural networks (GNNs) is an essential task for practical GNN model deployment and serving, as deployed GNNs face significant performance uncertainty when inferring on unseen and unlabeled test graphs, due to mismatched training-test graph distributions. In this paper, we study a *new* problem, **GNN model evaluation**, that aims to assess the performance of a specific GNN model trained on labeled and observed graphs, by precisely estimating its performance (e.g., node classification accuracy) on unseen graphs without labels. Concretely, we propose a two-stage GNN model evaluation framework, including (1) DiscGraph set construction and (2) GNNEvaluator training and inference. The DiscGraph set captures wide-range and diverse graph data distribution discrepancies through a discrepancy measurement function, which exploits the GNN outputs of latent node embeddings and node class predictions. Under the effective training supervision from the DiscGraph set, GNNEvaluator learns to precisely estimate node classification accuracy of the to-be-evaluated GNN model and makes an accurate inference for evaluating GNN model performance. Extensive experiments on real-world unseen and unlabeled test graphs demonstrate the effectiveness of our proposed method for GNN model evaluation.</details>

### GPEX, A Framework For Interpreting Artificial Neural Networks

**Authors:** Gilbert Bigras, Nilanjan Ray

**<details><summary>Abstract**</summary>

The analogy between Gaussian processes (GPs) and deep artificial neural networks (ANNs) has received a lot of interest, and has shown promise to unbox the blackbox of deep ANNs. Existing theoretical works put strict assumptions on the ANN (e.g. requiring all intermediate layers to be wide, or using specific activation functions). Accommodating those theoretical assumptions is hard in recent deep architectures, and those theoretical conditions need refinement as new deep architectures emerge. In this paper we derive an evidence lower-bound that encourages the GP's posterior to match the ANN's output without any requirement on the ANN. Using our method we find out that on 5 datasets, only a subset of those theoretical assumptions are sufficient. Indeed, in our experiments we used a normal ResNet-18 or feed-forward backbone with a single wide layer in the end. One limitation of training GPs is the lack of scalability with respect to the number of inducing points. We use novel computational techniques that allow us to train GPs with hundreds of thousands of inducing points and with GPU acceleration. As shown in our experiments, doing so has been essential to get a close match between the GPs and the ANNs on 5 datasets. We implement our method as a publicly available tool called GPEX: https://github.com/amirakbarnejad/gpex. On 5 datasets (4 image datasets, and 1 biological dataset) and ANNs with 2 types of functionality (classifier or attention-mechanism) we were able to find GPs whose outputs closely match those of the corresponding ANNs. After matching the GPs to the ANNs, we used the GPs' kernel functions to explain the ANNs' decisions. We provide more than 200 explanations (around 30 in the paper and the rest in the supplementary) which are highly interpretable by humans and show the ability of the obtained GPs to unbox the ANNs' decisions.</details>

### GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks

**Authors:** Zhonghang Li, Lianghao Xia, Yong Xu, Chao Huang

**<details><summary>Abstract**</summary>

In recent years, there has been a rapid development of spatio-temporal prediction techniques in response to the increasing demands of traffic management and travel planning. While advanced end-to-end models have achieved notable success in improving predictive performance, their integration and expansion pose significant challenges. This work aims to address these challenges by introducing a spatio-temporal pre-training framework that seamlessly integrates with downstream baselines and enhances their performance. The framework is built upon two key designs: (i) We propose a spatio-temporal mask autoencoder as a pre-training model for learning spatio-temporal dependencies. The model incorporates customized parameter learners and hierarchical spatial pattern encoding networks. These modules are specifically designed to capture spatio-temporal customized representations and intra- and inter-cluster region semantic relationships, which have often been neglected in existing approaches. (ii) We introduce an adaptive mask strategy as part of the pre-training mechanism. This strategy guides the mask autoencoder in learning robust spatio-temporal representations and facilitates the modeling of different relationships, ranging from intra-cluster to inter-cluster, in an easy-to-hard training manner. Extensive experiments conducted on representative benchmarks demonstrate the effectiveness of our proposed method. We have made our model implementation publicly available at https://github.com/HKUDS/GPT-ST.</details>

### Game Solving with Online Fine-Tuning

**Authors:** Ti-Rong Wu, Hung Guei, Ting Han Wei, Chung-Chin Shih, Jui-Te Chin, I-Chen Wu

**<details><summary>Abstract**</summary>

Game solving is a similar, yet more difficult task than mastering a game. Solving a game typically means to find the game-theoretic value (outcome given optimal play), and optionally a full strategy to follow in order to achieve that outcome. The AlphaZero algorithm has demonstrated super-human level play, and its powerful policy and value predictions have also served as heuristics in game solving. However, to solve a game and obtain a full strategy, a winning response must be found for all possible moves by the losing player. This includes very poor lines of play from the losing side, for which the AlphaZero self-play process will not encounter. AlphaZero-based heuristics can be highly inaccurate when evaluating these out-of-distribution positions, which occur throughout the entire search. To address this issue, this paper investigates applying online fine-tuning while searching and proposes two methods to learn tailor-designed heuristics for game solving. Our experiments show that using online fine-tuning can solve a series of challenging 7x7 Killall-Go problems, using only 23.54\% of computation time compared to the baseline without online fine-tuning. Results suggest that the savings scale with problem size. Our method can further be extended to any tree search algorithm for problem solving. Our code is available at https://rlg.iis.sinica.edu.tw/papers/neurips2023-online-fine-tuning-solver.</details>

### Generalization bounds for neural ordinary differential equations and deep residual networks

**Authors:** Pierre Marion

**<details><summary>Abstract**</summary>

Neural ordinary differential equations (neural ODEs) are a popular family of continuous-depth deep learning models.  In this work, we consider a large family of parameterized ODEs with continuous-in-time parameters, which include time-dependent neural ODEs.  We derive a generalization bound for this class by a Lipschitz-based argument. By leveraging the analogy between neural ODEs and deep residual networks, our approach yields in particular a generalization bound for a class of deep residual networks. The bound involves the magnitude of the difference between successive weight matrices. We illustrate numerically how this quantity affects the generalization capability of neural networks.</details>

### Generalized Belief Transport

**Authors:** Junqi Wang, PEI WANG, Patrick Shafto

**<details><summary>Abstract**</summary>

Human learners have ability to adopt appropriate learning approaches depending on constraints such as prior on the hypothesis, urgency of decision, and drift of the environment. However, existing learning models are typically considered individually rather than in relation to one and other. To build agents that have the ability to move between different modes of learning over time, it is important to understand how learning models are related as points in a broader space of possibilities. We introduce a mathematical framework, Generalized Belief Transport (GBT), that unifies and generalizes prior models, including Bayesian inference, cooperative communication and classification, as parameterizations of three learning constraints within Unbalanced Optimal Transport (UOT). We visualize the space of learning models encoded by GBT as a cube which includes classic learning models as special points. We derive critical properties of this parameterized space including proving continuity and differentiability which is the basis for model interpolation, and study limiting behavior of the parameters, which allows attaching learning models on the boundaries. Moreover, we investigate the long-run behavior of GBT, explore convergence properties of models in GBT mathematical and computationally, document the ability to learn in the presence of distribution drift, and formulate conjectures about general behavior. We conclude with open questions and implications for more unified models of learning.</details>

### Generalized test utilities for long-tail performance in extreme multi-label classification

**Authors:** Erik Schultheis, Marek Wydmuch, Wojciech Kotlowski, Rohit Babbar, Krzysztof Dembczynski

**<details><summary>Abstract**</summary>

Extreme multi-label classification (XMLC) is a task of selecting a small subset of relevant labels from a very large set of possible labels. As such, it is characterized by long-tail labels, i.e., most labels have very few positive instances. With standard performance measures such as precision@k, a classifier can ignore tail labels and still report good performance. However, it is often argued that correct predictions in the tail are more "interesting" or "rewarding," but the community has not yet settled on a metric capturing this intuitive concept. The existing propensity-scored metrics fall short on this goal by confounding the problems of long-tail and missing labels. In this paper, we analyze generalized metrics budgeted "at k" as an alternative solution. To tackle the challenging problem of optimizing these metrics, we formulate it in the \emph{expected test utility} (ETU) framework, which aims at optimizing the expected performance on a given test set. We derive optimal prediction rules and construct their computationally efficient approximations with provable regret guarantees and being robust against model misspecification. Our algorithm, based on block coordinate descent, scales effortlessly to XMLC problems and obtains promising results in terms of long-tail performance.</details>

### Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion

**Authors:** Zhengyi Yang, Jiancan Wu, Zhicai Wang, Xiang Wang, Yancheng Yuan, Xiangnan He

**<details><summary>Abstract**</summary>

Sequential recommendation aims to recommend the next item that matches a user’sinterest, based on the sequence of items he/she interacted with before. Scrutinizingprevious studies, we can summarize a common learning-to-classify paradigm—given a positive item, a recommender model performs negative sampling to addnegative items and learns to classify whether the user prefers them or not, based onhis/her historical interaction sequence. Although effective, we reveal two inherentlimitations: (1) it may differ from human behavior in that a user could imaginean oracle item in mind and select potential items matching the oracle; and (2)the classification is limited in the candidate pool with noisy or easy supervisionfrom negative samples, which dilutes the preference signals towards the oracleitem. Yet, generating the oracle item from the historical interaction sequence ismostly unexplored. To bridge the gap, we reshape sequential recommendationas a learning-to-generate paradigm, which is achieved via a guided diffusionmodel, termed DreamRec. Specifically, for a sequence of historical items, itapplies a Transformer encoder to create guidance representations. Noising targetitems explores the underlying distribution of item space; then, with the guidance ofhistorical interactions, the denoising process generates an oracle item to recoverthe positive item, so as to cast off negative sampling and depict the true preferenceof the user directly. We evaluate the effectiveness of DreamRec through extensiveexperiments and comparisons with existing methods. Codes and data are open-sourcedat https://github.com/YangZhengyi98/DreamRec.</details>

### Generating Behaviorally Diverse Policies with Latent Diffusion Models

**Authors:** Shashank Hegde, Sumeet Batra, K.R. Zentner, Gaurav Sukhatme

**<details><summary>Abstract**</summary>

Recent progress in Quality Diversity Reinforcement Learning (QD-RL) has enabled learning a collection of behaviorally diverse, high performing policies. However, these methods typically involve storing thousands of policies, which results in high space-complexity and poor scaling to additional behaviors. Condensing the archive into a single model while retaining the performance and coverage of theoriginal collection of policies has proved challenging. In this work, we propose using diffusion models to distill the archive into a single generative model over policy parameters. We show that our method achieves a compression ratio of 13x while recovering 98% of the original rewards and 89% of the original humanoid archive coverage. Further, the conditioning mechanism of diffusion models allowsfor flexibly selecting and sequencing behaviors, including using language. Project website: https://sites.google.com/view/policydiffusion/home.</details>

### Generative Category-level Object Pose Estimation via Diffusion Models

**Authors:** Jiyao Zhang, Mingdong Wu, Hao Dong

**<details><summary>Abstract**</summary>

Object pose estimation plays a vital role in embodied AI and computer vision, enabling intelligent agents to comprehend and interact with their surroundings. Despite the practicality of category-level pose estimation, current approaches encounter challenges with partially observed point clouds, known as the multihypothesis issue. In this study, we propose a novel solution by reframing categorylevel object pose estimation as conditional generative modeling, departing from traditional point-to-point regression. Leveraging score-based diffusion models, we estimate object poses by sampling candidates from the diffusion model and aggregating them through a two-step process: filtering out outliers via likelihood estimation and subsequently mean-pooling the remaining candidates. To avoid the costly integration process when estimating the likelihood, we introduce an alternative method that distils an energy-based model from the original score-based model, enabling end-to-end likelihood estimation. Our approach achieves state-of-the-art performance on the REAL275 dataset, surpassing 50% and 60% on strict 5 ◦ 2cm and 5 ◦ 5cm metrics, respectively. Furthermore, our method demonstrates strong generalization to novel categories without the need for fine-tuning and can readily adapt to object pose tracking tasks, yielding comparable results to the current state-of-the-art baselines. Our checkpoints and demonstrations can be found at https://sites.google.com/view/genpose.</details>

### Generator Born from Classifier

**Authors:** Runpeng Yu, Xinchao Wang

**<details><summary>Abstract**</summary>

In this paper, we make a bold attempt toward an ambitious task: given a pre-trained classifier, we aim to reconstruct an image generator, without relying on any data samples. From a black-box perspective, this challenge seems intractable, since it inevitably involves identifying the inverse function for a classifier, which is, by nature, an information extraction process. As such, we resort to leveraging the knowledge encapsulated within the parameters of the neural network. Grounded on the theory of Maximum-Margin Bias of gradient descent, we propose a novel learning paradigm, in which the generator is trained to ensure that the convergence conditions of the network parameters are satisfied over the generated distribution of the samples. Empirical validation from various image generation tasks substantiates the efficacy of our strategy.</details>

### Geodesic Multi-Modal Mixup for Robust Fine-Tuning

**Authors:** Changdae Oh, Junhyuk So, Hoyoon Byun, YongTaek Lim, Minchul Shin, Jong-June Jeon, Kyungwoo Song

**<details><summary>Abstract**</summary>

Pre-trained multi-modal models, such as CLIP, provide transferable embeddings and show promising results in diverse applications. However, the analysis of learned multi-modal embeddings is relatively unexplored, and the embedding transferability can be improved. In this work, we observe that CLIP holds separated embedding subspaces for two different modalities, and then we investigate it through the lens of \textit{uniformity-alignment} to measure the quality of learned representation. Both theoretically and empirically, we show that CLIP retains poor uniformity and alignment even after fine-tuning. Such a lack of alignment and uniformity might restrict the transferability and robustness of embeddings. To this end, we devise a new fine-tuning method for robust representation equipping better alignment and uniformity. First, we propose a \textit{Geodesic Multi-Modal Mixup} that mixes the embeddings of image and text to generate hard negative samples on the hypersphere. Then, we fine-tune the model on hard negatives as well as original negatives and positives with contrastive loss. Based on the theoretical analysis about hardness guarantee and limiting behavior, we justify the use of our method. Extensive experiments on retrieval, calibration, few- or zero-shot classification (under distribution shift), embedding arithmetic, and image captioning further show that our method provides transferable representations, enabling robust model adaptation on diverse tasks.</details>

### Geometry-Aware Adaptation for Pretrained Models

**Authors:** Nicholas Roberts, Xintong Li, Dyah Adila, Sonia Cromp, Tzu-Heng Huang, Jitian Zhao, Frederic Sala

**<details><summary>Abstract**</summary>

Machine learning models---including prominent zero-shot models---are often trained on datasets whose labels are only a small proportion of a larger label space. Such spaces are commonly equipped with a metric that relates the labels via distances between them. We propose a simple approach to exploit this information to adapt the trained model to reliably predict new classes---or, in the case of zero-shot prediction, to improve its performance---without any additional training. Our technique is a drop-in replacement of the standard prediction rule, swapping $\text{argmax}$ with the Fréchet mean. We provide a comprehensive theoretical analysis for this approach, studying (i) learning-theoretic results trading off label space diameter, sample complexity, and model dimension, (ii) characterizations of the full range of scenarios in which it is possible to predict any unobserved class, and (iii) an optimal active learning-like next class selection procedure to obtain optimal training classes for when it is not possible to predict the entire range of unobserved classes. Empirically, using easily-available external metrics, our proposed approach, Loki, gains up to 29.7% relative improvement over SimCLR on ImageNet and scales to hundreds of thousands of classes. When no such metric is available, Loki can use self-derived metrics from class embeddings and obtains a 10.5% improvement on pretrained zero-shot models such as CLIP.</details>

### Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design

**Authors:** Ibrahim Alabdulmohsin, Xiaohua Zhai, Alexander Kolesnikov, Lucas Beyer

**<details><summary>Abstract**</summary>

Scaling laws have been recently employed to derive compute-optimal model size (number of parameters) for a given compute duration. We advance and refine such methods to infer compute-optimal model shapes, such as width and depth, and successfully implement this in vision transformers. Our shape-optimized vision transformer, SoViT, achieves results competitive with models that exceed twice its size, despite being pre-trained with an equivalent amount of  compute. For example, SoViT-400m/14 achieves 90.3% fine-tuning accuracy on ILSRCV2012, surpassing the much larger ViT-g/14 and approaching ViT-G/14 under identical settings, with also less than half the inference cost. We conduct a thorough evaluation across multiple tasks, such as image classification, captioning, VQA and zero-shot transfer, demonstrating the effectiveness of our model across a broad range of domains and identifying limitations. Overall, our findings challenge the prevailing approach of blindly scaling up vision models and pave a path for a more informed scaling.</details>

### Global Convergence Analysis of Local SGD for Two-layer Neural Network without Overparameterization

**Authors:** Yajie Bao, Amarda Shehu, Mingrui Liu

**<details><summary>Abstract**</summary>

Local SGD, a cornerstone algorithm in federated learning, is widely used in training deep neural networks and shown to have strong empirical performance. A theoretical understanding of such performance on nonconvex loss landscapes is currently lacking. Analysis of the global convergence of SGD is challenging, as the noise depends on the model parameters. Indeed, many works narrow their focus to GD and rely on injecting noise to enable convergence to the local or global optimum. When expanding the focus to local SGD, existing analyses in the nonconvex case can only guarantee finding stationary points or assume the neural network is overparameterized so as to guarantee convergence to the global minimum through neural tangent kernel analysis. In this work, we provide the first global convergence analysis of the vanilla local SGD for two-layer neural networks \emph{without overparameterization} and \textit{without injecting noise}, when the input data is Gaussian. The main technical ingredients of our proof are \textit{a self-correction mechanism} and \textit{a new exact recursive characterization of the direction of global model parameters}. The self-correction mechanism guarantees the algorithm reaches a good region even if the initialization is in a bad region. A good (bad) region means updating the model by gradient descent will move closer to (away from) the optimal solution. The main difficulty in establishing a self-correction mechanism is to cope with the gradient dependency between two layers. To address this challenge, we divide the landscape of the objective into several regions to carefully control the interference of two layers during the correction process. As a result, we show that local SGD can correct the two layers and enter the good region in polynomial time. After that, we establish a new exact recursive characterization of the direction of global parameters, which is the key to showing convergence to the global minimum with linear speedup in the number of machines and reduced communication rounds. Experiments on synthetic data confirm theoretical results.</details>

### Global Update Tracking: A Decentralized Learning Algorithm for Heterogeneous Data

**Authors:** Sai Aparna Aketi, Abolfazl Hashemi, Kaushik Roy

**<details><summary>Abstract**</summary>

Decentralized learning enables the training of deep learning models over large distributed datasets generated at different locations, without the need for a central server. However, in practical scenarios, the data distribution across these devices can be significantly different, leading to a degradation in model performance. In this paper, we focus on designing a decentralized learning algorithm that is less susceptible to variations in data distribution across devices. We propose Global Update Tracking (GUT), a novel tracking-based method that aims to mitigate the impact of heterogeneous data in decentralized learning without introducing any communication overhead. We demonstrate the effectiveness of the proposed technique through an exhaustive set of experiments on various Computer Vision datasets (CIFAR-10, CIFAR-100, Fashion MNIST, and ImageNette), model architectures, and network topologies. Our experiments show that the proposed method achieves state-of-the-art performance for decentralized learning on heterogeneous data via a 1-6% improvement in test accuracy compared to other existing techniques.</details>

### Global-correlated 3D-decoupling Transformer for Clothed Avatar Reconstruction

**Authors:** Zechuan Zhang, Li Sun, Zongxin Yang, Ling Chen, Yi Yang

**<details><summary>Abstract**</summary>

Reconstructing 3D clothed human avatars from single images is a challenging task, especially when encountering complex poses and loose clothing. Current methods exhibit limitations in performance, largely attributable to their dependence on insufficient 2D image features and inconsistent query methods. Owing to this, we present the Global-correlated 3D-decoupling Transformer for clothed Avatar reconstruction (GTA), a novel transformer-based architecture that reconstructs clothed human avatars from monocular images. Our approach leverages transformer architectures by utilizing a Vision Transformer model as an encoder for capturing global-correlated image features. Subsequently, our innovative 3D-decoupling decoder employs cross-attention to decouple tri-plane features, using learnable embeddings as queries for cross-plane generation. To effectively enhance feature fusion with the tri-plane 3D feature and human body prior, we propose a hybrid prior fusion strategy combining spatial and prior-enhanced queries, leveraging the benefits of spatial localization and human body prior knowledge. Comprehensive experiments on CAPE and THuman2.0 datasets illustrate that our method outperforms state-of-the-art approaches in both geometry and texture reconstruction, exhibiting high robustness to challenging poses and loose clothing, and producing higher-resolution textures. Codes are available at https://github.com/River-Zhang/GTA.</details>

### [Oral] Going beyond persistent homology using persistent homology

**Authors:** Johanna Immonen, Amauri Souza, Vikas Garg

**Oral Presentation:** Th, Dec 14, 08:00 -- Oral 5A

**<details><summary>Abstract**</summary>

Representational limits of message-passing graph neural networks (MP-GNNs), e.g., in terms of the Weisfeiler-Leman (WL) test for isomorphism, are well understood. Augmenting these graph models with topological features via persistent homology (PH) has gained prominence, but identifying the class of attributed graphs that PH can recognize remains open.  We introduce a novel concept of color-separating sets to provide a complete resolution to this important problem. Specifically, we establish the necessary and sufficient conditions for distinguishing graphs based on the persistence of their connected components, obtained from filter functions on vertex and edge colors. Our constructions expose the limits of vertex- and edge-level PH, proving that neither category subsumes the other. Leveraging these theoretical insights, we propose RePHINE for learning topological features on graphs. RePHINE efficiently combines vertex- and edge-level PH, achieving a scheme that is provably more powerful than both. Integrating RePHINE into MP-GNNs boosts their expressive power, resulting in gains over standard PH on several benchmarks for graph classification.</details>

### Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism

**Authors:** Chengcheng Wang, Wei He, Ying Nie, Jianyuan Guo, Chuanjian Liu, Yunhe Wang, Kai Han

**<details><summary>Abstract**</summary>

In the past years, YOLO-series models have emerged as the leading approaches in the area of real-time object detection. Many studies pushed up the baseline to a higher level by modifying the architecture, augmenting data and designing new losses. However, we find previous models still suffer from information fusion problem, although Feature Pyramid Network (FPN) and Path Aggregation Network (PANet) have alleviated this. Therefore, this study provides an advanced Gatherand-Distribute mechanism (GD) mechanism, which is realized with convolution and self-attention operations. This new designed model named as Gold-YOLO, which boosts the multi-scale feature fusion capabilities and achieves an ideal balance between latency and accuracy across all model scales. Additionally, we implement MAE-style pretraining in the YOLO-series for the first time, allowing YOLOseries models could be to benefit from unsupervised pretraining. Gold-YOLO-N attains an outstanding 39.9% AP on the COCO val2017 datasets and 1030 FPS on a T4 GPU, which outperforms the previous SOTA model YOLOv6-3.0-N with similar FPS by +2.4%. The PyTorch code is available at https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO, and the MindSpore code is available at https://gitee.com/mindspore/models/tree/master/research/cv/Gold_YOLO.</details>

### Gradient Flossing: Improving Gradient Descent through Dynamic Control of Jacobians

**Authors:** Rainer Engelken

**<details><summary>Abstract**</summary>

Training recurrent neural networks (RNNs) remains a challenge due to the instability of gradients across long time horizons, which can lead to exploding and vanishing gradients. Recent research has linked these problems to the values of Lyapunov exponents for the forward-dynamics, which describe the growth or shrinkage of infinitesimal perturbations. Here, we propose gradient flossing, a novel approach to tackling gradient instability by pushing Lyapunov exponents of the forward dynamics toward zero during learning. We achieve this by regularizing Lyapunov exponents through backpropagation using differentiable linear algebra. This enables us to "floss" the gradients, stabilizing them and thus improving network training. We show that gradient flossing controls not only the gradient norm but also the condition number of the long-term Jacobian, facilitating multidimensional error feedback propagation. We find that applying gradient flossing before training enhances both the success rate and convergence speed for tasks involving long time horizons.For challenging tasks, we show that gradient flossing during training can further increase the time horizon that can be bridged by backpropagation through time. Moreover, we demonstrate the effectiveness of our approach on various RNN architectures and tasks of variable temporal complexity. Additionally, we provide a simple implementation of our gradient flossing algorithm that can be used in practice. Our results indicate that gradient flossing via regularizing Lyapunov exponents can significantly enhance the effectiveness of RNN training and mitigate the exploding and vanishing gradients problem.</details>

### Gradient Informed Proximal Policy Optimization

**Authors:** Sanghyun Son, Laura Zheng, Ryan Sullivan, Yi-Ling Qiao, Ming Lin

**<details><summary>Abstract**</summary>

We introduce a novel policy learning method that integrates analytical gradients from differentiable environments with the Proximal Policy Optimization (PPO) algorithm. To incorporate analytical gradients into the PPO framework, we introduce the concept of an α-policy that stands as a locally superior policy. By adaptively modifying the α value, we can effectively manage the influence of analytical policy gradients during learning. To this end, we suggest metrics for assessing the variance and bias of analytical gradients, reducing dependence on these gradients when high variance or bias is detected. Our proposed approach outperforms baseline algorithms in various scenarios, such as function optimization, physics simulations, and traffic control environments. Our code can be found online: https://github.com/SonSang/gippo.</details>

### Graph-Structured Gaussian Processes for Transferable Graph Learning

**Authors:** Jun Wu, Lisa Ainsworth, Andrew Leakey, Haixun Wang, Jingrui He

**<details><summary>Abstract**</summary>

Transferable graph learning involves knowledge transferability from a source graph to a relevant target graph. The major challenge of transferable graph learning is the distribution shift between source and target graphs induced by individual node attributes and complex graph structures. To solve this problem, in this paper, we propose a generic graph-structured Gaussian process framework (GraphGP) for adaptively transferring knowledge across graphs with either homophily or heterophily assumptions. Specifically, GraphGP is derived from a novel graph structure-aware neural network in the limit on the layer width. The generalization analysis of GraphGP explicitly investigates the connection between knowledge transferability and graph domain similarity. Extensive experiments on several transferable graph learning benchmarks demonstrate the efficacy of GraphGP over state-of-the-art Gaussian process baselines.</details>

### GraphPatcher: Mitigating Degree Bias for Graph Neural Networks via Test-time Augmentation

**Authors:** Mingxuan Ju, Tong Zhao, Wenhao Yu, Neil Shah, Yanfang Ye

**<details><summary>Abstract**</summary>

Recent studies have shown that graph neural networks (GNNs) exhibit strong biases towards the node degree: they usually perform satisfactorily on high-degree nodes with rich neighbor information but struggle with low-degree nodes. Existing works tackle this problem by deriving either designated GNN architectures or training strategies specifically for low-degree nodes. Though effective, these approaches unintentionally create an artificial out-of-distribution scenario, where models mainly or even only observe low-degree nodes during the training, leading to a downgraded performance for high-degree nodes that GNNs originally perform well at. In light of this, we propose a test-time augmentation framework, namely GraphPatcher, to enhance test-time generalization of any GNNs on low-degree nodes. Specifically, GraphPatcher iteratively generates virtual nodes to patch artificially created low-degree nodes via corruptions, aiming at progressively reconstructing target GNN's predictions over a sequence of increasingly corrupted nodes. Through this scheme, GraphPatcher not only learns how to enhance low-degree nodes (when the neighborhoods are heavily corrupted) but also preserves the original superior performance of GNNs on high-degree nodes (when lightly corrupted). Additionally, GraphPatcher is model-agnostic and can also mitigate the degree bias for either self-supervised or supervised GNNs. Comprehensive experiments are conducted over seven benchmark datasets and GraphPatcher consistently enhances common GNNs' overall performance by up to 3.6% and low-degree performance by up to 6.5%, significantly outperforming state-of-the-art baselines. The source code is publicly available at https://github.com/jumxglhf/GraphPatcher.</details>

### [Spotlight] Grounding Neural Inference with Satisfiability Modulo Theories

**Authors:** Zifan Wang, Saranya Vijayakumar, Kaiji Lu, Vijay Ganesh, Somesh Jha, Matt Fredrikson

**<details><summary>Abstract**</summary>

Recent techniques that integrate solver layers into Deep Neural Networks (DNNs) have shown promise in bridging a long-standing gap between inductive learning and symbolic reasoning techniques. In this paper we present a set of techniques for integrating Satisfiability Modulo Theories (SMT) solvers into the forward and backward passes of a deep network layer, called SMTLayer.Using this approach, one can encode rich domain knowledge into the network in the form of mathematical formulas.In the forward pass, the solver uses symbols produced by prior layers, along with these formulas, to construct inferences; in the backward pass, the solver informs updates to the network, driving it towards representations that are compatible with the solver's theory.Notably, the solver need not be differentiable. We implement SMTLayer as a Pytorch module, and our empirical results show that it leads to models that 1) require fewer training samples than conventional models, 2) that are robust to certain types of covariate shift, and 3) that ultimately learn representations that are consistent with symbolic knowledge, and thus naturally interpretable.</details>

### H-InDex: Visual Reinforcement Learning with Hand-Informed Representations for Dexterous Manipulation

**Authors:** Yanjie Ze, Yuyao Liu, Ruizhe Shi, Jiaxin Qin, Zhecheng Yuan, Jiashun Wang, Huazhe Xu

**<details><summary>Abstract**</summary>

Human hands possess remarkable dexterity and have long served as a source of inspiration for robotic manipulation. In this work, we propose a human $	extbf{H}$and-$	extbf{In}$formed visual representation learning framework to solve difficult $	extbf{Dex}$terous manipulation tasks ($	extbf{H-InDex}$) with reinforcement learning. Our framework consists of three stages: $	extit{(i)}$ pre-training representations with 3D human hand pose estimation, $	extit{(ii)}$ offline adapting representations with self-supervised keypoint detection, and $	extit{(iii)}$ reinforcement learning with exponential moving average BatchNorm. The last two stages only modify $0.36$% parameters of the pre-trained representation in total, ensuring the knowledge from pre-training is maintained to the full extent. We empirically study $	extbf{12}$ challenging dexterous manipulation tasks and find that $	extbf{H-InDex}$ largely surpasses strong baseline methods and the recent visual foundation models for motor control. Code and videos are available at https://yanjieze.com/H-InDex .</details>

### Hierarchical Gaussian Mixture based Task Generative Model for Robust Meta-Learning

**Authors:** Yizhou Zhang, Jingchao Ni, Wei Cheng, Zhengzhang Chen, Liang Tong, Haifeng Chen, Yan Liu

**<details><summary>Abstract**</summary>

Meta-learning enables quick adaptation of machine learning models to new tasks with limited data. While tasks could come from varying distributions in reality, most of the existing meta-learning methods consider both training and testing tasks as from the same uni-component distribution, overlooking two critical needs of a practical solution: (1) the various sources of tasks may compose a multi-component mixture distribution, and (2) novel tasks may come from a distribution that is unseen during meta-training. In this paper, we demonstrate these two challenges can be solved jointly by modeling the density of task instances. We develop a meta-training framework underlain by a novel Hierarchical Gaussian Mixture based Task Generative Model (HTGM). HTGM extends the widely used empirical process of sampling tasks to a theoretical model, which learns task embeddings, fits the mixture distribution of tasks, and enables density-based scoring of novel tasks. The framework is agnostic to the encoder and scales well with large backbone networks. The model parameters are learned end-to-end by maximum likelihood estimation via an Expectation-Maximization (EM) algorithm. Extensive experiments on benchmark datasets indicate the effectiveness of our method for both sample classification and novel task detection.</details>

### [Spotlight] High-Fidelity Audio Compression with Improved RVQGAN

**Authors:** Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, Kundan Kumar

**<details><summary>Abstract**</summary>

Language models have been successfully used to model natural signals, such as images, speech, and music. A key component of these models is a high quality neural compression model that can compress high-dimensional natural signals into lower dimensional discrete tokens. To that end, we introduce a high-fidelity universal neural audio compression algorithm that achieves ~90x compression of 44.1 KHz audio into tokens at just 8kbps bandwidth. We achieve this by combining advances in high-fidelity audio generation with better vector quantization techniques from the image domain, along with improved adversarial and reconstruction losses. We compress all domains (speech, environment, music, etc.) with a single universal model, making it widely applicable to generative modeling of all audio. We compare with competing audio compression algorithms, and find our method outperforms them significantly. We provide thorough ablations for every design choice, as well as open-source code and trained model weights. We hope our work can lay the foundation for the next generation of high-fidelity audio modeling.</details>

### History Filtering in Imperfect Information Games: Algorithms and Complexity

**Authors:** Christopher Solinas, Doug Rebstock, Nathan Sturtevant, Michael Buro

**<details><summary>Abstract**</summary>

Historically applied exclusively to perfect information games, depth-limited search with value functions has been key to recent advances in AI for imperfect information games. Most prominent approaches with strong theoretical guarantees require *subgame decomposition* - a process in which a subgame is computed from public information and player beliefs. However, subgame decomposition can itself require non-trivial computations, and its tractability depends on the existence of efficient algorithms for either full enumeration or generation of the histories that form the root of the subgame. Despite this, no formal analysis of the tractability of such computations has been established in prior work, and application domains have often consisted of games, such as poker, for which enumeration is trivial on modern hardware.Applying these ideas to more complex domains requires understanding their cost. In this work, we introduce and analyze the computational aspects and tractability of filtering histories for subgame decomposition. We show that constructing a single history from the root of the subgame is generally intractable, and then provide a necessary and sufficient condition for efficient enumeration. We also introduce a novel Markov Chain Monte Carlo-based generation algorithm for trick-taking card games - a domain where enumeration is often prohibitively expensive. Our experiments demonstrate its improved scalability in the trick-taking card game *Oh Hell*.These contributions clarify when and how depth-limited search via subgame decomposition can be an effective tool for sequential decision-making in imperfect information settings.</details>

### Homotopy-based training of NeuralODEs for accurate dynamics discovery

**Authors:** Joon-Hyuk Ko, Hankyul Koh, Nojun Park, Wonho Jhe

**<details><summary>Abstract**</summary>

Neural Ordinary Differential Equations (NeuralODEs) present an attractive way to extract dynamical laws from time series data, as they bridge neural networks with the differential equation-based modeling paradigm of the physical sciences. However, these models often display long training times and suboptimal results, especially for longer duration data. While a common strategy in the literature imposes strong constraints to the NeuralODE architecture to inherently promote stable model dynamics, such methods are ill-suited for dynamics discovery as the unknown governing equation is not guaranteed to satisfy the assumed constraints. In this paper, we develop a new training method for NeuralODEs, based on synchronization and homotopy optimization, that does not require changes to the model architecture. We show that synchronizing the model dynamics and the training data tames the originally irregular loss landscape, which homotopy optimization can then leverage to enhance training. Through benchmark experiments, we demonstrate our method achieves competitive or better training loss while often requiring less than half the number of training epochs compared to other model-agnostic techniques. Furthermore, models trained with our method display better extrapolation capabilities, highlighting the effectiveness of our method.</details>

### How Does Adaptive Optimization Impact Local Neural Network Geometry?

**Authors:** Kaiqi Jiang, Dhruv Malik, Yuanzhi Li

**<details><summary>Abstract**</summary>

Adaptive optimization methods are well known to achieve superior convergence relative to vanilla gradient methods. The traditional viewpoint in optimization, particularly in convex optimization, explains this improved performance by arguing that, unlike vanilla gradient schemes, adaptive algorithms mimic the behavior of a second-order method by adapting to the *global* geometry of the loss function. We argue that in the context of neural network optimization, this traditional viewpoint is insufficient. Instead, we advocate for a *local* trajectory analysis. For iterate trajectories produced by running a generic optimization algorithm OPT, we introduce $R^{\text{OPT}}\_{\text{med}}$, a statistic that is analogous to the condition number of the loss Hessian evaluated at the iterates. Through extensive experiments on language models where adaptive algorithms converge faster than vanilla gradient methods like SGD, we show that adaptive methods such as Adam bias the trajectories towards regions where $R^{\text{Adam}}_{\text{med}}$ is small, where one might expect faster optimization. By contrast, SGD (with momentum) biases the trajectories towards regions where $R^{\text{SGD}}\_{\text{med}}$ is comparatively large. We complement these empirical observations with a theoretical result that provably demonstrates this phenomenon in the simplified setting of a two-layer linear network. We view our findings as evidence for the need of a new explanation of the success of adaptive methods, one that is different than the conventional wisdom.</details>

### How do Minimum-Norm Shallow Denoisers Look in Function Space?

**Authors:** Chen Zeno, Greg Ongie, Yaniv Blumenfeld, Nir Weinberger, Daniel Soudry

**<details><summary>Abstract**</summary>

Neural network (NN) denoisers are an essential building block in many common tasks, ranging from image reconstruction to image generation. However, the success of these models is not well understood from a theoretical perspective. In this paper, we aim to characterize the functions realized by shallow ReLU NN denoisers --- in the common theoretical setting of interpolation (i.e., zero training loss) with a minimal representation cost (i.e., minimal $\ell^2$ norm weights). First, for univariate data, we derive a closed form for the NN denoiser function, find it is contractive toward the clean data points, and prove it generalizes better than the empirical MMSE estimator at a low noise level. Next, for multivariate data, we find the NN denoiser functions in a closed form under various geometric assumptions on the training data: data contained in a low-dimensional subspace, data contained in a union of one-sided rays, or several types of simplexes. These functions decompose into a sum of simple rank-one piecewise linear interpolations aligned with edges and/or faces connecting training samples. We empirically verify this alignment phenomenon on synthetic data and real images.</details>

### How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model

**Authors:** Michael Hanna, Ollie Liu, Alexandre Variengien

**<details><summary>Abstract**</summary>

Pre-trained language models can be surprisingly adept at tasks they were not explicitly trained on, but how they implement these capabilities is poorly understood. In this paper, we investigate the basic mathematical abilities often acquired by pre-trained language models. Concretely, we use mechanistic interpretability techniques to explain the (limited) mathematical abilities of GPT-2 small. As a case study, we examine its ability to take in sentences such as "The war lasted from the year 1732 to the year 17", and predict valid two-digit end years (years > 32). We first identify a circuit, a small subset of GPT-2 small's computational graph that computes this task's output. Then, we explain the role of each circuit component, showing that GPT-2 small's final multi-layer perceptrons boost the probability of end years greater than the start year. Finally, we find related tasks that activate our circuit. Our results suggest that GPT-2 small computes greater-than using a complex but general mechanism that activates across diverse contexts.</details>

### How to Select Which Active Learning Strategy is Best Suited for Your Specific Problem and Budget

**Authors:** Guy Hacohen, Daphna Weinshall

**<details><summary>Abstract**</summary>

In the domain of Active Learning (AL), a learner actively selects which unlabeled examples to seek labels from an oracle, while operating within predefined budget constraints. Importantly, it has been recently shown that distinct query strategies are better suited for different conditions and budgetary constraints. In practice, the determination of the most appropriate AL strategy for a given situation remains an open problem. To tackle this challenge, we propose a practical derivative-based method that dynamically identifies the best strategy for a given budget. Intuitive motivation for our approach is provided by the theoretical analysis of a simplified scenario. We then introduce a method to dynamically select an AL strategy, which takes into account the unique characteristics of the problem and the available budget. Empirical results showcase the effectiveness of our approach across diverse budgets and computer vision tasks.</details>

### [Spotlight] HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution

**Authors:** Eric Nguyen, Michael Poli, Marjan Faizi, Armin Thomas, Michael Wornow, Callum Birch-Sykes, Stefano Massaroli, Aman Patel, Clayton Rabideau, Yoshua Bengio, Stefano Ermon, Christopher Ré, Stephen Baccus

**<details><summary>Abstract**</summary>

Genomic (DNA) sequences encode an enormous amount of information for gene regulation and protein synthesis. Similar to natural language models, researchers have proposed foundation models in genomics to learn generalizable features from unlabeled genome data that can then be fine-tuned for downstream tasks such as identifying regulatory elements. Due to the quadratic scaling of attention, previous Transformer-based genomic models have used 512 to 4k tokens as context (<0.001% of the human genome), significantly limiting the modeling of long-range interactions in DNA. In addition, these methods rely on tokenizers or fixed k-mers to aggregate meaningful DNA units, losing single nucleotide resolution (i.e. DNA "characters") where subtle genetic variations can completely alter protein function via single nucleotide polymorphisms (SNPs). Recently, Hyena, a large language model based on implicit convolutions was shown to match attention in quality while allowing longer context lengths and lower time complexity. Leveraging Hyena’s new long-range capabilities, we present HyenaDNA, a genomic foundation model pretrained on the human reference genome with context lengths of up to 1 million tokens at the single nucleotide-level – an up to 500x increase over previous dense attention-based models. HyenaDNA scales sub-quadratically in sequence length (training up to 160x faster than Transformer), uses single nucleotide tokens, and has full global context at each layer. We explore what longer context enables - including the first use of in-context learning in genomics for simple adaptation to novel tasks without updating pretrained model weights. On fine-tuned benchmarks from the Nucleotide Transformer, HyenaDNA reaches state-of-the-art (SotA) on 12 of 18 datasets using a model with orders of magnitude less parameters and pretraining data.1 On the GenomicBenchmarks, HyenaDNA surpasses SotA on 7 of 8 datasets on average by +10 accuracy points. Code at https://github.com/HazyResearch/hyena-dna.</details>

### IEBins: Iterative Elastic Bins for Monocular Depth Estimation

**Authors:** Shuwei Shao, Zhongcai Pei, Xingming Wu, Zhong Liu, Weihai Chen, Zhengguo Li

**<details><summary>Abstract**</summary>

Monocular depth estimation (MDE) is a fundamental topic of geometric computer vision and a core technique for many downstream applications. Recently, several methods reframe the MDE as a classification-regression problem where a linear combination of probabilistic distribution and bin centers is used to predict depth. In this paper, we propose a novel concept of iterative elastic bins (IEBins) for the classification-regression-based MDE. The proposed IEBins aims to search for high-quality depth by progressively optimizing the search range, which involves multiple stages and each stage performs a finer-grained depth search in the target bin on top of its previous stage. To alleviate the possible error accumulation during the iterative process, we utilize a novel elastic target bin to replace the original target bin, the width of which is adjusted elastically based on the depth uncertainty. Furthermore, we develop a dedicated framework composed of a feature extractor and an iterative optimizer that has powerful temporal context modeling capabilities benefiting from the GRU-based architecture. Extensive experiments on the KITTI, NYU-Depth-v2 and SUN RGB-D datasets demonstrate that the proposed method surpasses prior state-of-the-art competitors. The source code is publicly available at https://github.com/ShuweiShao/IEBins.</details>

### ISP: Multi-Layered Garment Draping with Implicit Sewing Patterns

**Authors:** Ren Li, Benoit Guillard, Pascal Fua

**<details><summary>Abstract**</summary>

Many approaches to draping individual garments on human body models are realistic, fast, and yield outputs that are differentiable with respect to the body shape on which they are draped. However, they are either unable to handle multi-layered clothing, which is prevalent in everyday dress, or restricted to bodies in T-pose. In this paper, we introduce a parametric garment representation model that addresses these limitations. As in models used by clothing designers, each garment consists of individual 2D panels. Their 2D shape is defined by a Signed Distance Function and 3D shape by a 2D to 3D mapping. The 2D parameterization enables easy detection of potential collisions and the 3D parameterization handles complex shapes effectively. We show that this combination is faster and yields higher quality reconstructions than purely implicit surface representations, and makes the recovery of layered garments from images possible thanks to its differentiability. Furthermore, it supports rapid editing of garment shapes and texture by modifying individual 2D panels.</details>

### Identifiability Guarantees for Causal Disentanglement from Soft Interventions

**Authors:** Jiaqi Zhang, Kristjan Greenewald, Chandler Squires, Akash Srivastava, Karthikeyan Shanmugam, Caroline Uhler

**<details><summary>Abstract**</summary>

Causal disentanglement aims to uncover a representation of data using latent variables that are interrelated through a causal model. Such a representation is identifiable if the latent model that explains the data is unique. In this paper, we focus on the scenario where unpaired observational and interventional data are available, with each intervention changing the mechanism of a latent variable. When the causal variables are fully observed, statistically consistent algorithms have been developed to identify the causal model under faithfulness assumptions. We here show that identifiability can still be achieved with unobserved causal variables, given a generalized notion of faithfulness. Our results guarantee that we can recover the latent causal model up to an equivalence class and predict the effect of unseen combinations of interventions, in the limit of infinite data. We implement our causal disentanglement framework by developing an autoencoding variational Bayes algorithm and apply it to the problem of predicting combinatorial perturbation effects in genomics.</details>

### Imagine That! Abstract-to-Intricate Text-to-Image Synthesis with Scene Graph Hallucination Diffusion

**Authors:** Shengqiong Wu, Hao Fei, Hanwang Zhang, Tat-Seng Chua

**<details><summary>Abstract**</summary>

In this work, we investigate the task of text-to-image (T2I) synthesis under the abstract-to-intricate setting, i.e., generating intricate visual content from simple abstract text prompts. Inspired by human imagination intuition, we propose a novel scene-graph hallucination (SGH) mechanism for effective abstract-to-intricate T2I synthesis. SGH carries out scene hallucination by expanding the initial scene graph (SG) of the input prompt with more feasible specific scene structures, in which the structured semantic representation of SG ensures high controllability of the intrinsic scene imagination. To approach the T2I synthesis, we deliberately build an SG-based hallucination diffusion system. First, we implement the SGH module based on the discrete diffusion technique, which evolves the SG structure by iteratively adding new scene elements. Then, we utilize another continuous-state diffusion model as the T2I synthesizer, where the overt image-generating process is navigated by the underlying semantic scene structure induced from the SGH module. On the benchmark COCO dataset, our system outperforms the existing best-performing T2I model by a significant margin, especially improving on the abstract-to-intricate T2I generation. Further in-depth analyses reveal how our methods advance.</details>

### Implicit Differentiable Outlier Detection Enable Robust Deep Multimodal Analysis

**Authors:** Zhu Wang, Sourav Medya, Sathya Ravi

**<details><summary>Abstract**</summary>

Deep network models are often purely inductive during both training and inference on unseen data. When these models are used for prediction, but they may fail to capture important semantic information and implicit dependencies within datasets. Recent advancements have shown that combining multiple modalities in large-scale vision and language settings can improve understanding and generalization performance. However, as the model size increases, fine-tuning and deployment become computationally expensive, even for a small number of downstream tasks. Moreover, it is still unclear how domain or prior modal knowledge can be specified in a backpropagation friendly manner, especially in large-scale and noisy settings. To address these challenges, we propose a simplified alternative of combining features from pretrained deep networks and freely available semantic explicit knowledge. In order to remove irrelevant explicit knowledge that does not correspond well to the images, we introduce an implicit Differentiable Out-of-Distribution (OOD) detection layer. This layer addresses outlier detection by solving for fixed points of a differentiable function and using the last iterate of fixed point solver to backpropagate. In practice, we apply our model on several vision and language downstream tasks including visual question answering, visual reasoning, and image-text retrieval on different datasets. Our experiments show that it is possible to design models that perform similarly to state-of-the-art results but with significantly fewer samples and less training time. Our models and code are available here: https://github.com/ellenzhuwang/implicit_vkood</details>

### [Spotlight] Improved Convergence in High Probability of Clipped Gradient Methods with Heavy Tailed Noise

**Authors:** Ta Duy Nguyen, Thien H Nguyen, Alina Ene, Huy Nguyen

**<details><summary>Abstract**</summary>

In this work, we study the convergence in high probability of clipped gradient methods when the noise distribution has heavy tails, i.e., with bounded $p$th moments, for some $1</details>

### Improving Compositional Generalization using Iterated Learning and Simplicial Embeddings

**Authors:** Yi Ren, Samuel Lavoie, Michael Galkin, Danica J. Sutherland, Aaron Courville

**<details><summary>Abstract**</summary>

Compositional generalization, the ability of an agent to generalize to unseen combinations of latent factors, is easy for humans but hard for deep neural networks. A line of research in cognitive science has hypothesized a process, "iterated learning," to help explain how human language developed this ability; the theory rests on simultaneous pressures towards compressibility (when an ignorant agent learns from an informed one) and expressivity (when it uses the representation for downstream tasks). Inspired by this process, we propose to improve the compositional generalization of deep networks by using iterated learning on models with simplicial embeddings, which can approximately discretize representations. This approach is further motivated by an analysis of compositionality based on Kolmogorov complexity. We show that this combination of changes improves compositional generalization over other approaches, demonstrating these improvements both on vision tasks with well-understood latent factors and on real molecular graph prediction tasks where the latent structure is unknown.</details>

### Improving Language Plasticity via Pretraining with Active Forgetting

**Authors:** Yihong Chen, Kelly Marchisio, Roberta Raileanu, David Adelani, Sebastian Riedel, Mikel Artetxe

**<details><summary>Abstract**</summary>

Pretrained language models (PLMs) are today the primary model for natural language processing. Despite their impressive downstream performance, it can be difficult to apply PLMs to new languages, a barrier to making their capabilities universally accessible. While prior work has shown it possible to address this issue by learning a new embedding layer for the new language, doing so is both data and compute inefficient. We propose to use an active forgetting mechanism during pretraining, as a simple way of creating PLMs that can quickly adapt to new languages. Concretely, by resetting the embedding layer every K updates during pretraining, we encourage the PLM to improve its ability of learning new embeddings within limited number of updates, similar to a meta-learning effect. Experiments with RoBERTa show that models pretrained with our forgetting mechanism not only demonstrate faster convergence during language adaptation, but also outperform standard ones in a low-data regime, particularly for languages that are distant from English. Code will be available at https://github.com/facebookresearch/language-model-plasticity.</details>

### In Defense of Softmax Parametrization for Calibrated and Consistent Learning to Defer

**Authors:** Yuzhou Cao, Hussein Mozannar, Lei Feng, Hongxin Wei, Bo An

**<details><summary>Abstract**</summary>

Enabling machine learning classifiers to defer their decision to a downstream expert when the expert is more accurate will ensure improved safety and performance. This objective can be achieved with the learning-to-defer framework which aims to jointly learn how to classify and how to defer to the expert. In recent studies, it has been theoretically shown that popular estimators for learning to defer parameterized with softmax provide unbounded estimates for the likelihood of deferring which makes them uncalibrated. However, it remains unknown whether this is due to the widely used softmax parameterization and if we can find a softmax-based estimator that is both statistically consistent and possesses a valid probability estimator. In this work, we first show that the cause of the miscalibrated and unbounded estimator in prior literature is due to the symmetric nature of the surrogate losses used and not due to softmax. We then propose a novel statistically consistent asymmetric softmax-based surrogate loss that can produce valid estimates without the issue of unboundedness. We further analyze the non-asymptotic properties of our proposed method and empirically validate its performance and calibration on benchmark datasets.</details>

### Incentivizing Honesty among Competitors in Collaborative Learning and Optimization

**Authors:** Florian E. Dorner, Nikola Konstantinov, Georgi Pashaliev, Martin Vechev

**<details><summary>Abstract**</summary>

Collaborative learning techniques have the potential to enable training machine learning models that are superior to models trained on a single entity’s data. However, in many cases, potential participants in such collaborative schemes are competitors on a downstream task, such as firms that each aim to attract customers by providing the best recommendations. This can incentivize dishonest updates that damage other participants' models, potentially undermining the benefits of collaboration. In this work, we formulate a game that models such interactions and study two learning tasks within this framework: single-round mean estimation and multi-round SGD on strongly-convex objectives. For a natural class of player actions, we show that rational clients are incentivized to strongly manipulate their updates, preventing learning. We then propose mechanisms that incentivize honest communication and ensure learning quality comparable to full cooperation. Lastly, we empirically demonstrate the effectiveness of our incentive scheme on a standard non-convex federated learning benchmark. Our work shows that explicitly modeling the incentives and actions of dishonest clients, rather than assuming them malicious, can enable strong robustness guarantees for collaborative learning.</details>

### [Spotlight] Individual Arbitrariness and Group Fairness

**Authors:** Carol Long, Hsiang Hsu, Wael Alghamdi, Flavio Calmon

**<details><summary>Abstract**</summary>

Machine learning tasks may admit multiple competing models that achieve similar performance yet produce conflicting outputs for individual samples---a phenomenon known as predictive multiplicity. We demonstrate that fairness interventions in machine learning optimized solely for group fairness and accuracy can exacerbate predictive multiplicity. Consequently, state-of-the-art fairness interventions can mask high predictive multiplicity behind favorable group fairness and accuracy metrics. We argue that a third axis of ``arbitrariness'' should be considered  when deploying models to aid decision-making in applications of individual-level impact.To address this challenge, we propose an ensemble  algorithm applicable to any fairness intervention that provably ensures  more consistent predictions.</details>

### Individualized Dosing Dynamics via Neural Eigen Decomposition

**Authors:** Stav Belogolovsky, Ido Greenberg, Danny Eytan, Shie Mannor

**<details><summary>Abstract**</summary>

Dosing models often use differential equations to model biological dynamics. Neural differential equations in particular can learn to predict the derivative of a process, which permits predictions at irregular points of time. However, this temporal flexibility often comes with a high sensitivity to noise, whereas medical problems often present high noise and limited data. Moreover, medical dosing models must generalize reliably over individual patients and changing treatment policies. To address these challenges, we introduce the Neural Eigen Stochastic Differential Equation algorithm (NESDE). NESDE provides individualized modeling (using a hypernetwork over patient-level parameters); generalization to new treatment policies (using decoupled control); tunable expressiveness according to the noise level (using piecewise linearity); and fast, continuous, closed-form prediction (using spectral representation). We demonstrate the robustness of NESDE in both synthetic and real medical problems, and use the learned dynamics to publish simulated medical gym environments.</details>

### Inferring Hybrid Neural Fluid Fields from Videos

**Authors:** Hong-Xing Yu, Yang Zheng, Yuan Gao, Yitong Deng, Bo Zhu, Jiajun Wu

**<details><summary>Abstract**</summary>

We study recovering fluid density and velocity from sparse multiview videos. Existing neural dynamic reconstruction methods predominantly rely on optical flows; therefore, they cannot accurately estimate the density and uncover the underlying velocity due to the inherent visual ambiguities of fluid velocity, as fluids are often shapeless and lack stable visual features. The challenge is further pronounced by the turbulent nature of fluid flows, which calls for properly designed fluid velocity representations. To address these challenges, we propose hybrid neural fluid fields (HyFluid), a neural approach to jointly infer fluid density and velocity fields. Specifically, to deal with visual ambiguities of fluid velocity, we introduce a set of physics-based losses that enforce inferring a physically plausible velocity field, which is divergence-free and drives the transport of density. To deal with the turbulent nature of fluid velocity, we design a hybrid neural velocity representation that includes a base neural velocity field that captures most irrotational energy and a vortex particle-based velocity that models residual turbulent velocity. We show that our method enables recovering vortical flow details. Our approach opens up possibilities for various learning and reconstruction applications centered around 3D incompressible flow, including fluid re-simulation and editing, future prediction, and neural dynamic scene composition. Project website: https://kovenyu.com/HyFluid/</details>

### Inferring the Future by Imagining the Past

**Authors:** Kartik Chandra, Tony Chen, Tzu-Mao Li, Jonathan Ragan-Kelley, Josh Tenenbaum

**<details><summary>Abstract**</summary>

A single panel of a comic book can say a lot: it can depict not only where the characters currently are, but also their motions, their motivations, their emotions, and what they might do next. More generally, humans routinely infer complex sequences of past and future events from a *static snapshot* of a *dynamic scene*, even in situations they have never seen before.In this paper, we model how humans make such rapid and flexible inferences. Building on a long line of work in cognitive science, we offer a Monte Carlo algorithm whose inferences correlate well with human intuitions in a wide variety of domains, while only using a small, cognitively-plausible number of samples. Our key technical insight is a surprising connection between our inference problem and Monte Carlo path tracing, which allows us to apply decades of ideas from the computer graphics community to this seemingly-unrelated theory of mind task.</details>

### InfoPrompt: Information-Theoretic Soft Prompt Tuning for Natural Language Understanding

**Authors:** Junda Wu, Tong Yu, Rui Wang, Zhao Song, Ruiyi Zhang, Handong Zhao, Chaochao Lu, Shuai Li, Ricardo Henao

**<details><summary>Abstract**</summary>

Soft prompt tuning achieves superior performances across a wide range of few-shot tasks. However, the performances of prompt tuning can be highly sensitive to the initialization of the prompts. We have also empirically observed that conventional prompt tuning methods cannot encode and learn sufficient task-relevant information from prompt tokens. In this work, we develop an information-theoretic framework that formulates soft prompt tuning as maximizing the mutual information between prompts and other model parameters (or encoded representations). This novel view helps us to develop a more efficient, accurate and robust soft prompt tuning method, InfoPrompt. With this framework, we develop two novel mutual information based loss functions, to (i) explore proper prompt initialization for the downstream tasks and learn sufficient task-relevant information from prompt tokens and (ii) encourage the output representation from the pretrained language model to be more aware of the task-relevant information captured in the learnt prompts. Extensive experiments validate that InfoPrompt can significantly accelerate the convergence of the prompt tuning and outperform traditional prompt tuning methods. Finally, we provide a formal theoretical result to show that a gradient descent type algorithm can be used to train our mutual information loss.</details>

### Initialization-Dependent Sample Complexity of Linear Predictors and Neural Networks

**Authors:** Roey Magen, Ohad Shamir

**<details><summary>Abstract**</summary>

We provide several new results on the sample complexity of vector-valued linear predictors (parameterized by a matrix), and more generally neural networks. Focusing on size-independent bounds, where only the Frobenius norm distance of the parameters from some fixed reference matrix $W_0$ is controlled, we show that the sample complexity behavior can be surprisingly different than what we may expect considering the well-studied setting of scalar-valued linear predictors. This also leads to new sample complexity bounds for feed-forward neural networks, tackling some open questions in the literature, and establishing a new convex linear prediction problem that is provably learnable without uniform convergence.</details>

### InsActor: Instruction-driven Physics-based Characters

**Authors:** Jiawei Ren, Mingyuan Zhang, Cunjun Yu, Xiao Ma, Liang Pan, Ziwei Liu

**<details><summary>Abstract**</summary>

Generating animation of physics-based characters with intuitive control has long been a desirable task with numerous applications. However, generating physically simulated animations that reflect high-level human instructions remains a difficult problem due to the complexity of physical environments and the richness of human language. In this paper, we present $	extbf{InsActor}$, a principled generative framework that leverages recent advancements in diffusion-based human motion models to produce instruction-driven animations of physics-based characters.Our framework empowers InsActor to capture complex relationships between high-level human instructions and character motions by employing diffusion policies for flexibly conditioned motion planning.To overcome invalid states and infeasible state transitions in planned motions, InsActor discovers low-level skills and maps plans to latent skill sequences in a compact latent space. Extensive experiments demonstrate that InsActor achieves state-of-the-art results on various tasks, including instruction-driven motion generation and instruction-driven waypoint heading. Notably, the ability of InsActor to generate physically simulated animations using high-level human instructions makes it a valuable tool, particularly in executing long-horizon tasks with a rich set of instructions. Our project page is available at [jiawei-ren.github.io/projects/insactor/index.html](https://jiawei-ren.github.io/projects/insactor/index.html)</details>

### Interpretable Prototype-based Graph Information Bottleneck

**Authors:** Sangwoo Seo, Sungwon Kim, Chanyoung Park

**<details><summary>Abstract**</summary>

The success of Graph Neural Networks (GNNs) has led to a need for understanding their decision-making process and providing explanations for their predictions, which has given rise to explainable AI (XAI) that offers transparent explanations for black-box models. Recently, the use of prototypes has successfully improved the explainability of models by learning prototypes to imply training graphs that affect the prediction. However, these approaches tend to provide prototypes with excessive information from the entire graph, leading to the exclusion of key substructures or the inclusion of irrelevant substructures, which can limit both the interpretability and the performance of the model in downstream tasks. In this work, we propose a novel framework of explainable GNNs, called interpretable Prototype-based Graph Information Bottleneck (PGIB) that incorporates prototype learning within the information bottleneck framework to provide prototypes with the key subgraph from the input graph that is important for the model prediction. This is the first work that incorporates prototype learning into the process of identifying the key subgraphs that have a critical impact on the prediction performance. Extensive experiments, including qualitative analysis, demonstrate that PGIB outperforms state-of-the-art methods in terms of both prediction performance and explainability.</details>

### Inverse Dynamics Pretraining Learns Good Representations for Multitask Imitation

**Authors:** David Brandfonbrener, Ofir Nachum, Joan Bruna

**<details><summary>Abstract**</summary>

In recent years, domains such as natural language processing and image recognition have popularized the paradigm of using large datasets to pretrain representations that can be effectively transferred to downstream tasks. In this work we evaluate how such a paradigm should be done in imitation learning, where both pretraining and finetuning data are trajectories collected by experts interacting with an unknown environment. Namely, we consider a setting where the pretraining corpus consists of multitask demonstrations and the task for each demonstration is set by an unobserved latent context variable. The goal is to use the pretraining corpus to learn a low dimensional representation of the high dimensional (e.g., visual) observation space which can be transferred to a novel context for finetuning on a limited dataset of demonstrations. Among a variety of possible pretraining objectives, we argue that inverse dynamics modeling -- i.e.,  predicting an action given the observations appearing before and after it in the demonstration -- is well-suited to this setting. We provide empirical evidence of this claim through evaluations on a variety of simulated visuomotor manipulation problems. While previous work has attempted various theoretical explanations regarding the benefit of inverse dynamics modeling, we find that these arguments are insufficient to explain the empirical advantages often observed in our settings, and so we derive a novel analysis using a simple but general environment model.</details>

### Inverse Preference Learning: Preference-based RL without a Reward Function

**Authors:** Joey Hejna, Dorsa Sadigh

**<details><summary>Abstract**</summary>

Reward functions are difficult to design and often hard to align with human intent. Preference-based Reinforcement Learning (RL) algorithms address these problems by learning reward functions from human feedback. However, the majority of preference-based RL methods na\"ively combine supervised reward models with off-the-shelf RL algorithms. Contemporary approaches have sought to improve performance and query complexity by using larger and more complex reward architectures such as transformers. Instead of using highly complex architectures, we develop a new and parameter-efficient algorithm, Inverse Preference Learning (IPL), specifically designed for learning from offline preference data. Our key insight is that for a fixed policy, the $Q$-function encodes all information about the reward function, effectively making them interchangeable. Using this insight, we completely eliminate the need for a learned reward function. Our resulting algorithm is simpler and more parameter-efficient. Across a suite of continuous control and robotics benchmarks, IPL attains competitive performance compared to more complex approaches that leverage transformer-based and non-Markovian reward functions while having fewer algorithmic hyperparameters and learned network parameters. Our code is publicly released.</details>

### Is Heterogeneity Notorious? Taming Heterogeneity to Handle Test-Time Shift in Federated Learning

**Authors:** Yue Tan, Chen Chen, Weiming Zhuang, Xin Dong, Lingjuan Lyu, Guodong Long

**<details><summary>Abstract**</summary>

Federated learning (FL) is an effective machine learning paradigm where multiple clients can train models based on heterogeneous data in a decentralized manner without accessing their private data. However, existing FL systems undergo performance deterioration due to feature-level test-time shifts, which are well investigated in centralized settings but rarely studied in FL. The common non-IID issue in FL usually refers to inter-client heterogeneity during training phase, while the test-time shift refers to the intra-client heterogeneity during test phase. Although the former is always deemed to be notorious for FL, there is still a wealth of useful information delivered by heterogeneous data sources, which may potentially help alleviate the latter issue. To explore the possibility of using inter-client heterogeneity in handling intra-client heterogeneity, we firstly propose a contrastive learning-based FL framework, namely FedICON, to capture invariant knowledge among heterogeneous clients and consistently tune the model to adapt to test data. In FedICON, each client performs sample-wise supervised contrastive learning during the local training phase, which enhances sample-wise invariance encoding ability. Through global aggregation, the invariance extraction ability can be mutually boosted among inter-client heterogeneity. During the test phase, our test-time adaptation procedure leverages unsupervised contrastive learning to guide the model to smoothly generalize to test data under intra-client heterogeneity. Extensive experiments validate the effectiveness of the proposed FedICON in taming heterogeneity to handle test-time shift problems.</details>

### Iterative Reachability Estimation for Safe Reinforcement Learning

**Authors:** Milan Ganai, Zheng Gong, Chenning Yu, Sylvia Herbert, Sicun Gao

**<details><summary>Abstract**</summary>

Ensuring safety is important for the practical deployment of reinforcement learning (RL). Various challenges must be addressed, such as handling stochasticity in the environments, providing rigorous guarantees of persistent state-wise safety satisfaction, and avoiding overly conservative behaviors that sacrifice performance. We propose a new framework, Reachability Estimation for Safe Policy Optimization (RESPO), for safety-constrained RL in general stochastic settings. In the feasible set where there exist violation-free policies, we optimize for rewards while maintaining persistent safety. Outside this feasible set, our optimization produces the safest behavior by guaranteeing entrance into the feasible set whenever possible with the least cumulative discounted violations. We introduce a class of algorithms using our novel reachability estimation function to optimize in our proposed framework and in similar frameworks such as those concurrently handling multiple hard and soft constraints. We theoretically establish that our algorithms almost surely converge to locally optimal policies of our safe optimization framework. We evaluate the proposed methods on a diverse suite of safe RL environments from Safety Gym, PyBullet, and MuJoCo, and show the benefits in improving both reward performance and safety compared with state-of-the-art baselines.</details>

### Joint Feature and Differentiable $ k $-NN Graph Learning using Dirichlet Energy

**Authors:** Lei Xu, Lei Chen, Rong Wang, Feiping Nie, Xuelong Li

**<details><summary>Abstract**</summary>

Feature selection (FS) plays an important role in machine learning, which extracts important features and accelerates the learning process. In this paper, we propose a deep FS method that simultaneously conducts feature selection and differentiable $ k $-NN graph learning  based on the Dirichlet Energy. The Dirichlet Energy identifies important features by measuring their smoothness on the graph structure, and facilitates the learning of a new graph that reflects the inherent structure in new feature subspace. We employ Optimal Transport theory to address the non-differentiability issue of learning $ k $-NN graphs in neural networks, which theoretically makes our method applicable to other graph neural networks for dynamic graph learning. Furthermore, the proposed framework is interpretable, since all modules are designed algorithmically. We validate the effectiveness of our model with extensive experiments on both synthetic and real-world datasets.</details>

### K-Nearest-Neighbor Local Sampling Based Conditional Independence Testing

**Authors:** Shuai Li, Yingjie Zhang, Hongtu Zhu, Christina Wang, Hai Shu, Ziqi Chen, Zhuoran Sun, Yanfeng Yang

**<details><summary>Abstract**</summary>

Conditional independence (CI) testing is a fundamental task in statistics and machine learning, but its effectiveness is hindered by the challenges posed by high-dimensional conditioning variables and limited data samples. This article introduces a novel testing approach to address these challenges and enhance control of the type I error while achieving high power under alternative hypotheses. The proposed approach incorporates a computationally efficient classifier-based conditional mutual information (CMI) estimator, capable of capturing intricate dependence structures among variables. To approximate a distribution encoding the null hypothesis, a $k$-nearest-neighbor local sampling strategy is employed. An important advantage of this approach is its ability to operate without assumptions about distribution forms or feature dependencies. Furthermore, it eliminates the need to derive asymptotic null distributions for the estimated CMI and avoids dataset splitting, making it particularly suitable for small datasets. The method presented in this article demonstrates asymptotic control of the type I error and consistency against all alternative hypotheses. Extensive analyses using both synthetic and real data highlight the computational efficiency of the proposed test. Moreover, it outperforms existing state-of-the-art methods in terms of type I and II errors, even in scenarios with high-dimensional conditioning sets. Additionally, the proposed approach exhibits robustness in the presence of heavy-tailed data.</details>

### Kernel-Based Tests for Likelihood-Free Hypothesis Testing

**Authors:** Patrik Robert Gerber, Tianze Jiang, Yury Polyanskiy, Rui Sun

**<details><summary>Abstract**</summary>

Given $n$ observations from two balanced classes, consider the task of labeling an additional $m$ inputs that are known to all belong to \emph{one} of the two classes. Special cases of this problem are well-known: with completeknowledge of class distributions ($n=\infty$) theproblem is solved optimally by the likelihood-ratio test; when$m=1$ it corresponds to binary classification; and when $m\approx n$ it is equivalent to two-sample testing. The intermediate settings occur in the field of likelihood-free inference, where labeled samples are obtained by running forward simulations and the unlabeled sample is collected experimentally. In recent work it was discovered that there is a fundamental trade-offbetween $m$ and $n$: increasing the data sample $m$ reduces the amount $n$ of training/simulationdata needed. In this work we (a) introduce a generalization where unlabeled samples come from a mixture of the two classes -- a case often encountered in practice; (b) study the minimax sample complexity for non-parametric classes of densities under \textit{maximum meandiscrepancy} (MMD) separation; and (c) investigate the empirical performance of kernels parameterized by neural networks on two tasks: detectionof the Higgs boson and detection of planted DDPM generated images amidstCIFAR-10 images. For both problems we confirm the existence of the theoretically predicted asymmetric $m$ vs $n$ trade-off.</details>

### [Spotlight] Kernelized Cumulants: Beyond Kernel Mean Embeddings

**Authors:** Patric Bonnier, Harald Oberhauser, Zoltan Szabo

**<details><summary>Abstract**</summary>

In $\mathbb{R}^d$, it is well-known that cumulants provide an alternative to moments that can achieve the same goals with numerous benefits such as lower variance estimators. In this paper we extend cumulants to reproducing kernel Hilbert spaces (RKHS) using tools from tensor algebras and show that they are computationally tractable by a kernel trick. These kernelized cumulants provide a new set of all-purpose statistics; the classical maximum mean discrepancy and Hilbert-Schmidt independence criterion arise as the degree one objects in our general construction. We argue both theoretically and empirically (on synthetic, environmental, and traffic data analysis) that going beyond degree one has several advantages and can be achieved with the same computational complexity and minimal overhead in our experiments.</details>

### Knowledge Distillation for High Dimensional Search Index

**Authors:** Zepu Lu, Jin Chen, Defu Lian, ZAIXI ZHANG, Yong Ge, Enhong Chen

**<details><summary>Abstract**</summary>

Lightweight compressed models are prevalent in Approximate Nearest Neighbor Search (ANNS) and Maximum Inner Product Search (MIPS) owing to their superiority of retrieval efficiency in large-scale datasets. However, results given by compressed methods are less accurate due to the curse of dimension and the limitations of optimization objectives (e.g., lacking interactions between queries and documents). Thus, we are encouraged to design a new learning algorithm for the compressed search index on high dimensions to improve retrieval performance. In this paper, we propose a novel KnowledgeDistillation for high dimensional search index framework (KDindex), with the aim of efficiently learning lightweight indexes by distilling knowledge from high-precision ANNS and MIPS models such as graph-based indexes. Specifically, the student is guided to keep the same ranking order of the top-k relevant results yielded by the teacher model, which acts as the additional supervision signals between queries and documents to learn the similarities between documents. Furthermore, to avoid the trivial solutions that all candidates are partitioned to the same centroid, the reconstruction loss that minimizes the compressed error, and the posting list balance strategy that equally allocates the candidates, are integrated into the learning objective. Experiment results demonstrate that KDindex outperforms existing learnable quantization-based indexes and is 40× lighter than the state-of-the-art non-exhaustive methods while achieving comparable recall quality.</details>

### L2T-DLN: Learning to Teach with Dynamic Loss Network

**Authors:** Zhaoyang Hai, Liyuan Pan, Xiabi Liu, Zhengzheng Liu, Mirna Yunita

**<details><summary>Abstract**</summary>

With the concept of teaching being introduced to the machine learning community, a teacher model start using dynamic loss functions to teach the training of a student model. The dynamic intends to set adaptive loss functions to different phases of student model learning. In existing works, the teacher model 1) merely determines the loss function based on the present states of the student model, e.g., disregards the experience of the teacher; 2) only utilizes the states of the student model, e.g., training iteration number and loss/accuracy from training/validation sets, while ignoring the states of the loss function. In this paper, we first formulate the loss adjustment as a temporal task by designing a teacher model with memory units, and, therefore, enables the student learning to be guided by the experience of the teacher model. Then, with a Dynamic Loss Network, we can additionally use the states of the loss to assist the teacher learning in enhancing the interactions between the teacher and the student model.   Extensive experiments demonstrate our approach can enhance student learning and improve the performance of various deep models on real-world tasks, including classification, objective detection, and semantic segmentation scenario.</details>

### LART: Neural Correspondence Learning with Latent Regularization Transformer for 3D Motion Transfer

**Authors:** Haoyu Chen, Hao Tang, Radu Timofte, Luc V Gool, Guoying Zhao

**<details><summary>Abstract**</summary>

3D motion transfer aims at transferring the motion from a dynamic input sequence to a static 3D object and outputs an identical motion of the target with high-fidelity and realistic visual effects. In this work, we propose a novel 3D Transformer framework called LART for 3D motion transfer. With carefully-designed architectures, LART is able to implicitly learn the correspondence via a flexible geometry perception. Thus, unlike other existing methods, LART does not require any key point annotations or pre-defined correspondence between the motion source and target meshes and can also handle large-size full-detailed unseen 3D targets. Besides, we introduce a novel latent metric regularization on the Transformer for better motion generation. Our rationale lies in the observation that the decoded motions can be approximately expressed as linearly geometric distortion at the frame level. The metric preservation of motions could be translated to the formation of linear paths in the underlying latent space as a rigorous constraint to control the synthetic motions occurring in the construction of the latent space. The proposed LART shows a high learning efficiency with the need for a few samples from the AMASS dataset to generate motions with plausible visual effects. The experimental results verify the potential of our generative model in applications of motion transfer, content generation, temporal interpolation, and motion denoising. The code is made available: https://github.com/mikecheninoulu/LART.</details>

### LEPARD: Learning Explicit Part Discovery for 3D Articulated Shape Reconstruction

**Authors:** Di Liu, Anastasis Stathopoulos, Qilong Zhangli, Yunhe Gao, Dimitris Metaxas

**<details><summary>Abstract**</summary>

Reconstructing the 3D articulated shape of an animal from a single in-the-wild image is a challenging task. We propose LEPARD, a learning-based framework that discovers semantically meaningful 3D parts and reconstructs 3D shapes in a part-based manner. This is advantageous as 3D parts are robust to pose variations due to articulations and their shape is typically simpler than the overall shape of the object. In our framework, the parts are explicitly represented as parameterized primitive surfaces with global and local deformations in 3D that deform to match the image evidence. We propose a kinematics-inspired optimization to guide each transformation of the primitive deformation given 2D evidence. Similar to recent approaches, LEPARD is only trained using off-the-shelf deep features from DINO and does not require any form of 2D or 3D annotations. Experiments on 3D animal shape reconstruction, demonstrate significant improvement over existing alternatives in terms of both the overall reconstruction performance as well as the ability to discover semantically meaningful and consistent parts.</details>

### LaFTer: Label-Free Tuning of Zero-shot Classifier using Language and Unlabeled Image Collections

**Authors:** Muhammad Jehanzeb Mirza, Leonid Karlinsky, Wei Lin, Horst Possegger, Mateusz Kozinski, Rogerio Feris, Horst Bischof

**<details><summary>Abstract**</summary>

Recently, large-scale pre-trained Vision and Language (VL) models have set a new state-of-the-art (SOTA) in zero-shot visual classification enabling open-vocabulary recognition of potentially unlimited set of categories defined as simple language prompts. However, despite these great advances, the performance of these zero-shot classifiers still falls short of the results of dedicated (closed category set) classifiers trained with supervised fine-tuning. In this paper we show, for the first time, how to reduce this gap without any labels and without any paired VL data, using an unlabeled image collection and a set of texts auto-generated using a Large Language Model (LLM) describing the categories of interest and effectively substituting labeled visual instances of those categories. Using our label-free approach, we are able to attain significant performance improvements over the zero-shot performance of the base VL model and other contemporary methods and baselines on a wide variety of datasets, demonstrating absolute improvement of up to $11.7\%$ ($3.8\%$ on average) in the label-free setting. Moreover, despite our approach being label-free, we observe $1.3\%$ average gains over leading few-shot prompting baselines that do use 5-shot supervision.</details>

### Label Correction of Crowdsourced Noisy Annotations with an Instance-Dependent Noise Transition Model

**Authors:** Hui GUO, Boyu Wang, Grace Yi

**<details><summary>Abstract**</summary>

The predictive ability of supervised learning algorithms hinges on the quality of annotated examples, whose labels often come from multiple crowdsourced annotators with diverse expertise. To aggregate noisy crowdsourced annotations, many existing methods employ an annotator-specific instance-independent noise transition matrix  to characterize the labeling skills of each annotator. Learning an instance-dependent noise transition model, however, is challenging and remains relatively less explored. To address this problem, in this paper, we formulate the noise transition model in a Bayesian framework and subsequently design a new label correction algorithm. Specifically, we approximate the instance-dependent noise transition matrices using a Bayesian network with a hierarchical spike and slab prior. To theoretically characterize the distance between the noise transition model and the true instance-dependent noise transition matrix, we provide a posterior-concentration theorem that ensures the posterior consistency in terms of the Hellinger distance. We further formulate the label correction process as a hypothesis testing problem and propose a novel algorithm to infer the true label from the noisy annotations based on the pairwise likelihood ratio test. Moreover, we establish an information-theoretic bound on the Bayes error for the proposed method. We validate the effectiveness of our approach through experiments on benchmark and real-world datasets.</details>

### Label Poisoning is All You Need

**Authors:** Rishi Jha, Jonathan Hayase, Sewoong Oh

**<details><summary>Abstract**</summary>

In a backdoor attack, an adversary injects corrupted data into a model's training dataset in order to gain control over its predictions on images with a specific attacker-defined trigger. A typical corrupted training example requires altering both the image, by applying the trigger, and the label. Models trained on clean images, therefore, were considered safe from backdoor attacks. However, in some common machine learning scenarios, the training labels are provided by potentially malicious third-parties. This includes crowd-sourced annotation and knowledge distillation. We, hence, investigate a fundamental question: can we launch a successful backdoor attack by only corrupting labels? We introduce a novel approach to design label-only backdoor attacks, which we call FLIP, and demonstrate its strengths on three datasets (CIFAR-10, CIFAR-100, and Tiny-ImageNet) and four architectures (ResNet-32, ResNet-18, VGG-19, and Vision Transformer). With only 2% of CIFAR-10 labels corrupted, FLIP achieves a near-perfect attack success rate of 99.4% while suffering only a 1.8% drop in the clean test accuracy. Our approach builds upon the recent advances in trajectory matching, originally introduced for dataset distillation.</details>

### Label-Retrieval-Augmented Diffusion Models for Learning from Noisy Labels

**Authors:** Jian Chen, Ruiyi Zhang, Tong Yu, Rohan Sharma, Zhiqiang Xu, Tong Sun, Changyou Chen

**<details><summary>Abstract**</summary>

Learning from noisy labels is an important and long-standing problem in machine learning for real applications. One of the main research lines focuses on learning a label corrector to purify potential noisy labels. However, these methods typically rely on strict assumptions and are limited to certain types of label noise. In this paper, we reformulate the label-noise problem from a generative-model perspective, *i.e.*, labels are generated by gradually refining an initial random guess. This new perspective immediately enables existing powerful diffusion models to seamlessly learn the stochastic generative process. Once the generative uncertainty is modeled, we can perform classification inference using maximum likelihood estimation of labels. To mitigate the impact of noisy labels, we propose the **L**abel-**R**etrieval-**A**ugmented (LRA) diffusion model, which leverages neighbor consistency to effectively construct pseudo-clean labels for diffusion training. Our model is flexible and general, allowing easy incorporation of different types of conditional information, *e.g.*, use of pre-trained models, to further boost model performance. Extensive experiments are conducted for evaluation. Our model achieves new state-of-the-art (SOTA) results on all the standard real-world benchmark datasets. Remarkably, by incorporating conditional information from the powerful CLIP model, our method can boost the current SOTA accuracy by 10-20 absolute points in many cases. Code is available: https://anonymous.4open.science/r/LRA-diffusion-5F2F</details>

### Labeling Neural Representations with Inverse Recognition

**Authors:** Kirill Bykov, Laura Kopf, Shinichi Nakajima, Marius Kloft, Marina Höhne

**<details><summary>Abstract**</summary>

Deep Neural Networks (DNNs) demonstrated remarkable capabilities in learning complex hierarchical data representations, but the nature of these representations remains largely unknown. Existing global explainability methods, such as Network Dissection, face limitations such as reliance on segmentation masks, lack of statistical significance testing, and high computational demands. We propose Inverse Recognition (INVERT), a scalable approach for linking the learned representations to human-interpretable concepts based on the ability to differentiate between concepts. In contrast to prior work, INVERT is capable of handling diverse types of neurons, exhibits less computational complexity, and does not rely on the availability of segmentation masks. Moreover, INVERT provides an interpretable metric assessing the alignment between the representation and its corresponding explanation and delivering a measure of statistical significance, emphasizing its utility and credibility.We demonstrate the applicability of INVERT in various scenarios, including the identification of representations affected by spurious correlations, and the interpretation of the hierarchical structure of decision-making within the models.</details>

### LambdaBeam: Neural Program Search with Higher-Order Functions and Lambdas

**Authors:** Kensen Shi, Hanjun Dai, Wen-Ding Li, Kevin Ellis, Charles Sutton

**<details><summary>Abstract**</summary>

Search is an important technique in program synthesis that allows for adaptive strategies such as focusing on particular search directions based on execution results. Several prior works have demonstrated that neural models are effective at guiding program synthesis searches. However, a common drawback of those approaches is the inability to handle iterative loops, higher-order functions, or lambda functions, thus limiting prior neural searches from synthesizing longer and more general programs. We address this gap by designing a search algorithm called LambdaBeam that can construct arbitrary lambda functions that compose operations within a given DSL. We create semantic vector representations of the execution behavior of the lambda functions and train a neural policy network to choose which lambdas to construct during search, and pass them as arguments to higher-order functions to perform looping computations. Our experiments show that LambdaBeam outperforms neural, symbolic, and LLM-based techniques in an integer list manipulation domain.</details>

### Landscape Surrogate: Learning Decision Losses for Mathematical Optimization Under Partial Information

**Authors:** Arman Zharmagambetov, Brandon Amos, Aaron Ferber, Taoan Huang, Bistra Dilkina, Yuandong Tian

**<details><summary>Abstract**</summary>

Recent works in learning-integrated optimization have shown promise in settings where the optimization problem is only partially observed or where general-purpose optimizers perform poorly without expert tuning. By learning an optimizer $\mathbf{g}$ to tackle these challenging problems with $f$ as the objective, the optimization process can be substantially accelerated by leveraging past experience. The optimizer can be trained with supervision from known optimal solutions or implicitly by optimizing the compound function $f\circ \mathbf{g}$. The implicit approach may not require optimal solutions as labels and is capable of handling problem uncertainty; however, it is slow to train and deploy due to frequent calls to optimizer $\mathbf{g}$ during both training and testing. The training is further challenged by sparse gradients of $\mathbf{g}$, especially for combinatorial solvers. To address these challenges, we propose using a smooth and learnable **Landscape Surrogate** $\mathcal{M}$ as a replacement for $f\circ \mathbf{g}$. This surrogate, learnable by neural networks, can be computed faster than the solver $\mathbf{g}$, provides dense and smooth gradients during training, can generalize to unseen optimization problems, and is efficiently learned via alternating optimization. We test our approach on both synthetic problems, including shortest path and multidimensional knapsack, and real-world problems such as portfolio optimization, achieving comparable or superior objective values compared to state-of-the-art baselines while reducing the number of calls to $\mathbf{g}$. Notably, our approach outperforms existing methods for computationally expensive high-dimensional problems.</details>

### Language-based Action Concept Spaces Improve Video Self-Supervised Learning

**Authors:** Kanchana Ranasinghe, Michael S Ryoo

**<details><summary>Abstract**</summary>

Recent contrastive language image pre-training has led to learning highly transferable and robust image representations. However, adapting these models to video domain with minimal supervision remains an open problem. We explore a simple step in that direction, using language tied self-supervised learning to adapt an image CLIP model to the video domain. A backbone modified for temporal modeling is trained under self-distillation settings with train objectives operating in an action concept space. Feature vectors of various action concepts extracted from a language encoder using relevant textual prompts construct this space. A large language model aware of actions and their attributes generates the relevant textual prompts.We introduce two train objectives, concept distillation and concept alignment, that retain generality of original representations while enforcing relations between actions and their attributes. Our approach improves zero-shot and linear probing performance on three action recognition benchmarks.</details>

### Large Language Models Are Zero-Shot Time Series Forecasters

**Authors:** Nate Gruver, Marc Finzi, Shikai Qiu, Andrew Wilson

**<details><summary>Abstract**</summary>

By encoding time series as a string of numerical digits, we can frame time series forecasting as next-token prediction in text. Developing this approach, we find that large language models (LLMs) such as GPT-3 and LLaMA-2 can surprisingly zero-shot extrapolate time series at a level comparable to or exceeding the performance of purpose-built time series models trained on the downstream tasks. To facilitate this performance, we propose procedures for effectively tokenizing time series data and converting discrete distributions over tokens into highly flexible densities over continuous values. We argue the success of LLMs for time series stems from their ability to naturally represent multimodal distributions, in conjunction with biases for simplicity, and repetition, which align with the salient features in many time series, such as repeated seasonal trends. We also show how LLMs can naturally handle missing data without imputation through non-numerical text, accommodate textual side information, and answer questions to help explain predictions.  While we find that increasing model size generally improves performance on time series, we show GPT-4 can perform worse than GPT-3 because of how it tokenizes numbers, and poor uncertainty calibration, which is likely the result of alignment interventions such as RLHF.</details>

### Large language models implicitly learn to straighten neural sentence trajectories to construct a predictive representation of natural language.

**Authors:** Eghbal Hosseini, Evelina Fedorenko

**<details><summary>Abstract**</summary>

Predicting upcoming events is critical to our ability to effectively interact with ourenvironment and conspecifics. In natural language processing, transformer models,which are trained on next-word prediction, appear to construct a general-purposerepresentation of language that can support diverse downstream tasks. However, westill lack an understanding of how a predictive objective shapes such representations.Inspired by recent work in vision neuroscience Hénaff et al. (2019), here we test ahypothesis about predictive representations of autoregressive transformer models.In particular, we test whether the neural trajectory of a sequence of words in asentence becomes progressively more straight as it passes through the layers of thenetwork. The key insight behind this hypothesis is that straighter trajectories shouldfacilitate prediction via linear extrapolation. We quantify straightness using a 1-dimensional curvature metric, and present four findings in support of the trajectorystraightening hypothesis: i) In trained models, the curvature progressively decreasesfrom the first to the middle layers of the network. ii) Models that perform better onthe next-word prediction objective, including larger models and models trained onlarger datasets, exhibit greater decreases in curvature, suggesting that this improvedability to straighten sentence neural trajectories may be the underlying driver ofbetter language modeling performance. iii) Given the same linguistic context, thesequences that are generated by the model have lower curvature than the groundtruth (the actual continuations observed in a language corpus), suggesting thatthe model favors straighter trajectories for making predictions. iv) A consistentrelationship holds between the average curvature and the average surprisal ofsentences in the middle layers of models, such that sentences with straighter neuraltrajectories also have lower surprisal. Importantly, untrained models don’t exhibitthese behaviors. In tandem, these results support the trajectory straighteninghypothesis and provide a possible mechanism for how the geometry of the internalrepresentations of autoregressive models supports next word prediction.</details>

### Latent Space Translation via Semantic Alignment

**Authors:** Valentino Maiorca, Luca Moschella, Antonio Norelli, Marco Fumero, Francesco Locatello, Emanuele Rodolà

**<details><summary>Abstract**</summary>

While different neural models often exhibit latent spaces that are alike when exposed to semantically related data, this intrinsic similarity is not always immediately discernible. Towards a better understanding of this phenomenon, our work shows how representations learned from these neural modules can be translated between different pre-trained networks via simpler transformations than previously thought. An advantage of this approach is the ability to estimate these transformations using standard, well-understood algebraic procedures that have closed-form solutions. Our method directly estimates a transformation between two given latent spaces, thereby enabling effective stitching of encoders and decoders without additional training. We extensively validate the adaptability of this translation procedure in different experimental settings: across various trainings, domains, architectures (e.g., ResNet, CNN, ViT), and in multiple downstream tasks (classification, reconstruction). Notably, we show how it is possible to zero-shot stitch text encoders and vision decoders, or vice-versa, yielding surprisingly good classification performance in this multimodal setting.</details>

### Learn to Categorize or Categorize to Learn? Self-Coding for Generalized Category Discovery

**Authors:** Sarah Rastegar, Hazel Doughty, Cees Snoek

**<details><summary>Abstract**</summary>

In the quest for unveiling novel categories at test time, we confront the inherent limitations of traditional supervised recognition models that are restricted by a predefined category set. While strides have been made in the realms of self-supervised and open-world learning towards test-time category discovery, a crucial yet often overlooked question persists: what exactly delineates a category? In this paper, we conceptualize a category through the lens of optimization, viewing it as an optimal solution to a well-defined problem. Harnessing this unique conceptualization, we propose a novel, efficient and self-supervised method capable of discovering previously unknown categories at test time. A salient feature of our approach is the assignment of minimum length category codes to individual data instances, which encapsulates the implicit category hierarchy prevalent in real-world datasets. This mechanism affords us enhanced control over category granularity, thereby equipping our model to handle fine-grained categories adeptly. Experimental evaluations, bolstered by state-of-the-art benchmark comparisons, testify to the efficacy of our solution in managing unknown categories at test time. Furthermore, we fortify our proposition with a theoretical foundation, providing proof of its optimality. Our code is available at: https://github.com/SarahRastegar/InfoSieve.</details>

### Learning Cuts via Enumeration Oracles

**Authors:** Daniel Thuerck, Boro Sofranac, Marc E Pfetsch, Sebastian Pokutta

**<details><summary>Abstract**</summary>

Cutting-planes are one of the most important building blocks for solving large-scale integer programming (IP) problems to (near) optimality. The majority of cutting plane approaches rely on explicit rules to derive valid inequalities that can separate the target point from the feasible set. Local cuts, on the other hand, seek to directly derive the facets of the underlying polyhedron and use them as cutting planes. However, current approaches rely on solving Linear Programming (LP) problems in order to derive such a hyperplane. In this paper, we present a novel generic approach for learning the facets of the underlying polyhedron by accessing it implicitly via an enumeration oracle in a reduced dimension. This is achieved by embedding the oracle in a variant of the Frank-Wolfe algorithm which is capable of generating strong cutting planes, effectively turning the enumeration oracle into a separation oracle. We demonstrate the effectiveness of our approach with a case study targeting the multidimensional knapsack problem (MKP).</details>

### Learning Dictionary for Visual Attention

**Authors:** Yingjie Liu, Xuan Liu, Hui Yu, XUAN TANG, Xian Wei

**<details><summary>Abstract**</summary>

Recently, the attention mechanism has shown outstanding competence in capturing global structure information and long-range relationships within data, thus enhancing the performance of deep vision models on various computer vision tasks. In this work, we propose a novel dictionary learning-based attention (\textit{Dic-Attn}) module, which models this issue as a decomposition and reconstruction problem with the sparsity prior, inspired by sparse coding in the human visual perception system. The proposed \textit{Dic-Attn} module decomposes the input into a dictionary and corresponding sparse representations, allowing for the disentanglement of underlying nonlinear structural information in visual data and the reconstruction of an attention embedding. By applying transformation operations in the spatial and channel domains, the module dynamically selects the dictionary's atoms and sparse representations. Finally, the updated dictionary and sparse representations capture the global contextual information and reconstruct the attention maps. The proposed \textit{Dic-Attn} module is designed with plug-and-play compatibility, allowing for integration into deep attention encoders. Our approach offers an intuitive and elegant means to exploit the discriminative information from data, promoting visual attention construction. Extensive experimental results on various computer vision tasks, e.g., image and point cloud classification, validate that our method achieves promising performance, and shows a strong competitive comparison with state-of-the-art attention methods.</details>

### Learning Efficient Coding of Natural Images with Maximum Manifold Capacity Representations

**Authors:** Thomas Yerxa, Yilun Kuang, Eero Simoncelli, SueYeon Chung

**<details><summary>Abstract**</summary>

The efficient coding hypothesis proposes that the response properties of sensory systems are adapted to the statistics of their inputs such that they capture maximal information about the environment, subject to biological constraints. While elegant, information theoretic properties are notoriously difficult to measure in practical settings or to employ as objective functions in optimization. This difficulty has necessitated that computational models designed to test the hypothesis employ several different information metrics ranging from approximations and lower bounds to proxy measures like reconstruction error. Recent theoretical advances have characterized a novel and ecologically relevant efficiency metric, the ``manifold capacity,” which is the number of object categories that may be represented in a linearly separable fashion. However, calculating manifold capacity is a computationally intensive iterative procedure that until now has precluded its use as an objective. Here we outline the simplifying assumptions that allow manifold capacity to be optimized directly, yielding Maximum Manifold Capacity Representations (MMCR). The resulting method is closely related to and inspired by advances in the field of self supervised learning (SSL), and we demonstrate that MMCRs are competitive with state of the art results on standard SSL benchmarks. Empirical analyses reveal differences between MMCRs and representations learned by other SSL frameworks, and suggest a mechanism by which manifold compression gives rise to class separability.  Finally we evaluate a set of SSL methods on a suite of neural predicitivity benchmarks, and find MMCRs are higly competitive as models of the ventral stream.</details>

### Learning Energy-Based Prior Model with Diffusion-Amortized MCMC

**Authors:** Peiyu Yu, Yaxuan Zhu, Sirui Xie, Xiaojian (Shawn) Ma, Ruiqi Gao, Song-Chun Zhu, Ying Nian Wu

**<details><summary>Abstract**</summary>

Latent space EBMs, also known as energy-based priors, have drawn growing interests in the field of generative modeling due to its flexibility in the formulation and strong modeling power of the latent space. However, the common practice of learning latent space EBMs with non-convergent short-run MCMC for prior and posterior sampling is hindering the model from further progress; the degenerate MCMC sampling quality in practice often leads to degraded generation quality and instability in training, especially with highly multi-modal and/or high-dimensional target distributions. To remedy this sampling issue, in this paper we introduce a simple but effective diffusion-based amortization method for long-run MCMC sampling and develop a novel learning algorithm for the latent space EBM based on it. We provide theoretical evidence that the learned amortization of MCMC is a valid long-run MCMC sampler. Experiments on several image modeling benchmark datasets demonstrate the superior performance of our method compared with strong counterparts.</details>

### Learning Invariant Molecular Representation in Latent Discrete Space

**Authors:** Xiang Zhuang, Qiang Zhang, Keyan Ding, Yatao Bian, Xiao Wang, Jingsong Lv, Hongyang Chen, Huajun Chen

**<details><summary>Abstract**</summary>

Molecular representation learning lays the foundation for drug discovery. However, existing methods suffer from poor out-of-distribution (OOD) generalization, particularly when data for training and testing originate from different environments. To address this issue, we propose a new framework for learning molecular representations that exhibit invariance and robustness against distribution shifts. Specifically, we propose a strategy called ``first-encoding-then-separation'' to identify invariant molecule features in the latent space, which deviates from conventional practices. Prior to the separation step, we introduce a residual vector quantization module that mitigates the over-fitting to training data distributions while preserving the expressivity of encoders. Furthermore, we design a task-agnostic self-supervised learning objective to encourage precise invariance identification, which enables our method widely applicable to a variety of tasks, such as regression and multi-label classification. Extensive experiments on 18 real-world molecular datasets demonstrate that our model achieves stronger generalization against state-of-the-art baselines in the presence of various distribution shifts. Our code is available at https://github.com/HICAI-ZJU/iMoLD.</details>

### Learning List-Level Domain-Invariant Representations for Ranking

**Authors:** Ruicheng Xian, Honglei Zhuang, Zhen Qin, Hamed Zamani, Jing Lu, Ji Ma, Kai Hui, Han Zhao, Xuanhui Wang, Michael Bendersky

**<details><summary>Abstract**</summary>

Domain adaptation aims to transfer the knowledge learned on (data-rich) source domains to (low-resource) target domains, and a popular method is invariant representation learning, which matches and aligns the data distributions on the feature space. Although this method is studied extensively and applied on classification and regression problems, its adoption on ranking problems is sporadic, and the few existing implementations lack theoretical justifications. This paper revisits invariant representation learning for ranking. Upon reviewing prior work, we found that they implement what we call item-level alignment, which aligns the distributions of the items being ranked from all lists in aggregate but ignores their list structure. However, the list structure should be leveraged, because it is intrinsic to ranking problems where the data and the metrics are defined and computed on lists, not the items by themselves. To close this discrepancy, we propose list-level alignment—learning domain-invariant representations at the higher level of lists. The benefits are twofold: it leads to the first domain adaptation generalization bound for ranking, in turn providing theoretical support for the proposed method, and it achieves better empirical transfer performance for unsupervised domain adaptation on ranking tasks, including passage reranking.</details>

### Learning Modulated Transformation in GANs

**Authors:** Ceyuan Yang, Qihang Zhang, Yinghao Xu, Jiapeng Zhu, Yujun Shen, Bo Dai

**<details><summary>Abstract**</summary>

The success of style-based generators largely benefits from style modulation,which helps take care of the cross-instance variation within data. However, theinstance-wise stochasticity is typically introduced via regular convolution, wherekernels interact with features at some fixed locations, limiting its capacity formodeling geometric variation. To alleviate this problem, we equip the generatorin generative adversarial networks (GANs) with a plug-and-play module, termedas modulated transformation module (MTM). This module predicts spatial offsetsunder the control of latent codes, based on which the convolution operation canbe applied at variable locations for different instances, and hence offers the modelan additional degree of freedom to handle geometry deformation. Extensiveexperiments suggest that our approach can be faithfully generalized to variousgenerative tasks, including image generation, 3D-aware image synthesis, andvideo generation, and get compatible with state-of-the-art frameworks withoutany hyper-parameter tuning. It is noteworthy that, towards human generation onthe challenging TaiChi dataset, we improve the FID of StyleGAN3 from 21.36 to13.60, demonstrating the efficacy of learning modulated geometry transformation.Code and models are available at https://github.com/limbo0000/mtm.</details>

### Learning Motion Refinement for Unsupervised Face Animation

**Authors:** Jiale Tao, Shuhang Gu, Wen Li, Lixin Duan

**<details><summary>Abstract**</summary>

Unsupervised face animation aims to generate a human face video based on theappearance of a source image, mimicking the motion from a driving video. Existingmethods typically adopted a prior-based motion model (e.g., the local affine motionmodel or the local thin-plate-spline motion model). While it is able to capturethe coarse facial motion, artifacts can often be observed around the tiny motionin local areas (e.g., lips and eyes), due to the limited ability of these methodsto model the finer facial motions. In this work, we design a new unsupervisedface animation approach to learn simultaneously the coarse and finer motions. Inparticular, while exploiting the local affine motion model to learn the global coarsefacial motion, we design a novel motion refinement module to compensate forthe local affine motion model for modeling finer face motions in local areas. Themotion refinement is learned from the dense correlation between the source anddriving images. Specifically, we first construct a structure correlation volume basedon the keypoint features of the source and driving images. Then, we train a modelto generate the tiny facial motions iteratively from low to high resolution. Thelearned motion refinements are combined with the coarse motion to generate thenew image. Extensive experiments on widely used benchmarks demonstrate thatour method achieves the best results among state-of-the-art baselines.</details>

### Learning Probabilistic Symmetrization for Architecture Agnostic Equivariance

**Authors:** Jinwoo Kim, Dat Nguyen, Ayhan Suleymanzade, Hyeokjun An, Seunghoon Hong

**<details><summary>Abstract**</summary>

We present a novel framework to overcome the limitations of equivariant architectures in learning functions with group symmetries. In contrary to equivariant architectures, the framework uses an arbitrary backbone (such as an MLP or a transformer) and symmetrizes it to be equivariant to given group by employing a small equivariant network that parameterizes the probabilistic distribution underlying the symmetrization. The distribution is end-to-end trained with the backbone which can maximize performance while reducing sample complexity of symmetrization. We show that this approach ensures not only equivariance to the given group but also universal approximation ability in expectation. We implement our method on a simple patch-based transformer backbone initialized from pretrained vision transformer, and test it for a wide range of symmetry groups including permutation and Euclidean groups and their combinations. Empirical tests show competitive results against tailored equivariant architectures, suggesting the potential for learning equivariant functions for diverse groups using a non-equivariant universal backbone. We further show evidence of enhanced learning in symmetric modalities, like graphs, when pretrained from non-symmetric modalities, like vision.</details>

### Learning Provably Robust Estimators for Inverse Problems via Jittering

**Authors:** Anselm Krainovic, Mahdi Soltanolkotabi, Reinhard Heckel

**<details><summary>Abstract**</summary>

Deep neural networks provide excellent performance for inverse problems such as denoising. However, neural networks can be sensitive to adversarial or worst-case perturbations. This raises the question of whether such networks can be trained efficiently to be worst-case robust. In this paper, we investigate whether jittering, a simple regularization technique that adds isotropic Gaussian noise during training, is effective for learning worst-case robust estimators for inverse problems. While well studied for prediction in classification tasks, the effectiveness of jittering for inverse problems has not been systematically investigated. In this paper, we present a novel analytical characterization of the optimal $\ell_2$-worst-case robust estimator for linear denoising and show that jittering yields optimal robust denoisers. Furthermore, we examine jittering empirically via training deep neural networks (U-nets) for natural image denoising, deconvolution, and accelerated magnetic resonance imaging (MRI). The results show that jittering significantly enhances the worst-case robustness, but can be suboptimal for inverse problems beyond denoising. Moreover, our results imply that training on real data which often contains slight noise is somewhat robustness enhancing.</details>

### Learning Re-sampling Methods with Parameter Attribution for Image Super-resolution

**Authors:** Xiaotong Luo, Yuan Xie, Yanyun Qu

**<details><summary>Abstract**</summary>

Single image super-resolution (SISR) has made a significant breakthrough benefiting from the prevalent rise of deep neural networks and large-scale training samples. The mainstream deep SR models primarily focus on network architecture design as well as optimization schemes, while few pay attention to the training data. In fact, most of the existing SR methods train the model on uniformly sampled patch pairs from the whole image. However, the uneven image content makes the training data present an unbalanced distribution, i.e., the easily reconstructed region (smooth) occupies the majority of the data, while the hard reconstructed region (edge or texture) has rarely few samples. Based on this phenomenon, we consider rethinking the current paradigm of merely using uniform data sampling way for training SR models. In this paper, we propose a simple yet effective Bi-Sampling Parameter Attribution (BSPA) method for accurate image SR. Specifically, the bi-sampling consists of uniform sampling and inverse sampling, which is introduced to reconcile the unbalanced inherent data bias. The former aims to keep the intrinsic data distribution, and the latter is designed to enhance the feature extraction ability of the model on the hard samples. Moreover, integrated gradient is introduced to attribute the contribution of each parameter in the alternate models trained by both sampling data so as to filter the trivial parameters for further dynamic refinement. By progressively decoupling the allocation of parameters, the SR model can learn a more compact representation. Extensive experiments on publicly available datasets demonstrate that our proposal can effectively boost the performance of baseline methods from the data re-sampling view.</details>

### Learning Reliable Logical Rules with SATNet

**Authors:** Zhaoyu Li, Jinpei Guo, Yuhe Jiang, Xujie Si

**<details><summary>Abstract**</summary>

Bridging logical reasoning and deep learning is crucial for advanced AI systems. In this work, we present a new framework that addresses this goal by generating interpretable and verifiable logical rules through differentiable learning, without relying on pre-specified logical structures. Our approach builds upon SATNet, a differentiable MaxSAT solver that learns the underlying rules from input-output examples. Despite its efficacy, the learned weights in SATNet are not straightforwardly interpretable, failing to produce human-readable rules. To address this, we propose a novel specification method called ``maximum equality'', which enables the interchangeability between the learned weights of SATNet and a set of propositional logical rules in weighted MaxSAT form. With the decoded weighted MaxSAT formula, we further introduce several effective verification techniques to validate it against the ground truth rules. Experiments on stream transformations and Sudoku problems show that our decoded rules are highly reliable: using exact solvers on them could achieve 100% accuracy, whereas the original SATNet fails to give correct solutions in many cases. Furthermore, we formally verify that our decoded logical rules are functionally equivalent to the ground truth ones.</details>

### Learning Repeatable Speech Embeddings Using An Intra-class Correlation Regularizer

**Authors:** Jianwei Zhang, Suren Jayasuriya, Visar Berisha

**<details><summary>Abstract**</summary>

A good supervised embedding for a specific machine learning task is only sensitive to changes in the label of interest and is invariant to other confounding factors. We leverage the concept of repeatability from measurement theory to describe this property and propose to use the intra-class correlation coefficient (ICC) to evaluate the repeatability of embeddings. We then propose a novel regularizer, the ICC regularizer, as a complementary component for contrastive losses to guide deep neural networks to produce embeddings with higher repeatability. We use simulated data to explain why the ICC regularizer works better on minimizing the intra-class variance than the contrastive loss alone. We implement the ICC regularizer and apply it to three speech tasks: speaker verification, voice style conversion, and a clinical application for detecting dysphonic voice. The experimental results demonstrate that adding an ICC regularizer can improve the repeatability of learned embeddings compared to only using the contrastive loss; further, these embeddings lead to improved performance in these downstream tasks.</details>

### Learning Score-based Grasping Primitive for Human-assisting Dexterous Grasping

**Authors:** Tianhao Wu, Mingdong Wu, Jiyao Zhang, Yunchong Gan, Hao Dong

**<details><summary>Abstract**</summary>

The use of anthropomorphic robotic hands for assisting individuals in situations where human hands may be unavailable or unsuitable has gained significant importance. In this paper, we propose a novel task called human-assisting dexterous grasping that aims to train a policy for controlling a robotic hand's fingers to assist users in grasping objects. Unlike conventional dexterous grasping, this task presents a more complex challenge as the policy needs to adapt to diverse user intentions, in addition to the object's geometry. We address this challenge by proposing an approach consisting of two sub-modules: a hand-object-conditional grasping primitive called Grasping Gradient Field (GraspGF), and a history-conditional residual policy. GraspGF learns 'how' to grasp by estimating the gradient of a synthesised success grasping example set, while the residual policy determines 'when' and at what speed the grasping action should be executed based on the trajectory history. Experimental results demonstrate the superiority of our proposed method compared to baselines, highlighting the user-awareness and practicality in real-world applications. The codes and demonstrations can be viewed at https://sites.google.com/view/graspgf.</details>

### Learning To Dive In Branch And Bound

**Authors:** Max Paulus, Andreas Krause

**<details><summary>Abstract**</summary>

Primal heuristics are important for solving mixed integer linear programs, because they find feasible solutions that facilitate branch and bound search. A prominent group of primal heuristics are diving heuristics. They iteratively modify and resolve linear programs to conduct a depth-first search from any node in the search tree. Existing divers rely on generic decision rules that fail to exploit structural commonality between similar problem instances that often arise in practice. Therefore, we propose L2Dive to learn specific diving heuristics with graph neural networks: We train generative models to predict variable assignments and leverage the duality of linear programs to make diving decisions based on the model's predictions. L2Dive is fully integrated into the open-source solver SCIP. We find that L2Dive outperforms standard divers to find better feasible solutions on a range of combinatorial optimization problems. For real-world applications from server load balancing and neural network verification, L2Dive improves the primal-dual integral by up to 7% (35%) on average over a tuned (default) solver baseline and reduces average solving time by 20% (29%).</details>

### Learning Topology-Agnostic EEG Representations with Geometry-Aware Modeling

**Authors:** Ke Yi, Yansen Wang, Kan Ren, Dongsheng Li

**<details><summary>Abstract**</summary>

Large-scale pre-training has shown great potential to enhance models on downstream tasks in vision and language. Developing similar techniques for scalp electroencephalogram (EEG) is suitable since unlabelled data is plentiful. Meanwhile, various sampling channel selections and inherent structural and spatial information bring challenges and avenues to improve existing pre-training strategies further. In order to break boundaries between different EEG resources and facilitate cross-dataset EEG pre-training, we propose to map all kinds of channel selections to a unified topology. We further introduce MMM, a pre-training framework with Multi-dimensional position encoding, Multi-level channel hierarchy, and Multi-stage pre-training strategy built on the unified topology to obtain topology-agnostic representations. Experiments demonstrate that our approach yields impressive improvements over previous state-of-the-art techniques on emotional recognition benchmark datasets.</details>

### Learning and processing the ordinal information of temporal sequences in recurrent neural circuits

**Authors:** xiaolong zou, Zhikun Chu, Qinghai Guo, Jie Cheng, Bo Ho, Si Wu, Yuanyuan Mi

**<details><summary>Abstract**</summary>

Temporal sequence processing is fundamental in brain cognitive functions. Experimental data has indicated that the representations of ordinal information and contents of temporal sequences are disentangled in the brain, but the neural mechanism underlying this disentanglement remains largely unclear. Here, we investigate how recurrent neural circuits learn to represent the abstract order structure of temporal sequences, and how this disentangled representation of order structure from that of contents facilitates the processing of temporal sequences. We show that with an appropriate learn protocol, a recurrent neural circuit can learn a set of tree-structured attractor states to encode the corresponding tree-structured orders of given temporal sequences. This abstract temporal order template can then be bound with different contents, allowing for flexible and robust temporal sequence processing. Using a transfer learning task, we demonstrate that the reuse of a temporal order template facilitates the acquisition of new temporal sequences of the same or similar ordinal structure. Using a key-word spotting task, we demonstrate that the attractor representation of order structure improves the robustness of temporal sequence discrimination, if the ordinal information is the key to differentiate different sequences. We hope this study gives us insights into the neural mechanism of representing the ordinal information of temporal sequences in the brain, and helps us to develop brain-inspired temporal sequence processing algorithms.</details>

### Learning from Both Structural and Textual Knowledge for Inductive Knowledge Graph Completion

**Authors:** Kunxun Qi, Jianfeng Du, Hai Wan

**<details><summary>Abstract**</summary>

Learning rule-based systems plays a pivotal role in knowledge graph completion (KGC). Existing rule-based systems restrict the input of the system to structural knowledge only, which may omit some useful knowledge for reasoning, e.g., textual knowledge. In this paper, we propose a two-stage framework that imposes both structural and textual knowledge to learn rule-based systems. In the first stage, we compute a set of triples with confidence scores (called \emph{soft triples}) from a text corpus by distant supervision, where a textual entailment model with multi-instance learning is exploited to estimate whether a given triple is entailed by a set of sentences. In the second stage, these soft triples are used to learn a rule-based model for KGC. To mitigate the negative impact of noise from soft triples, we propose a new formalism for rules to be learnt, named \emph{text enhanced rules} or \emph{TE-rules} for short. To effectively learn TE-rules, we propose a neural model that simulates the inference of TE-rules. We theoretically show that any set of TE-rules can always be interpreted by a certain parameter assignment of the neural model. We introduce three new datasets to evaluate the effectiveness of our method. Experimental results demonstrate that the introduction of soft triples and TE-rules results in significant performance improvements in inductive link prediction.</details>

### Learning to Compress Prompts with Gist Tokens

**Authors:** Jesse Mu, Xiang Li, Noah Goodman

**<details><summary>Abstract**</summary>

Prompting is the primary way to utilize the multitask capabilities of language models (LMs), but prompts occupy valuable space in the input context window, and repeatedly encoding the same prompt is computationally inefficient. Finetuning and distillation methods allow for specialization of LMs without prompting, but require retraining the model for each task. To avoid this trade-off entirely, we present gisting, which trains an LM to compress prompts into smaller sets of "gist" tokens which can be cached and reused for compute efficiency. Gist models can be trained with no additional cost over standard instruction finetuning by simply modifying Transformer attention masks to encourage prompt compression. On decoder (LLaMA-7B) and encoder-decoder (FLAN-T5-XXL) LMs, gisting enables up to 26x compression of prompts, resulting in up to 40% FLOPs reductions, 4.2% wall time speedups, and storage savings, all with minimal loss in output quality.</details>

### Learning to Configure Separators in Branch-and-Cut

**Authors:** Sirui Li, Wenbin Ouyang, Max Paulus, Cathy Wu

**<details><summary>Abstract**</summary>

Cutting planes are crucial in solving mixed integer linear programs (MILP) as they facilitate bound improvements on the optimal solution. Modern MILP solvers rely on a variety of separators to generate a diverse set of cutting planes by invoking the separators frequently during the solving process. This work identifies that MILP solvers can be drastically accelerated by appropriately selecting separators to activate. As the combinatorial separator selection space imposes challenges for machine learning, we *learn to separate* by proposing a novel data-driven strategy to restrict the selection space and a learning-guided algorithm on the restricted space. Our method predicts instance-aware separator configurations which can dynamically adapt during the solve, effectively accelerating the open source MILP solver SCIP by improving the relative solve time up to 72% and 37% on synthetic and real-world MILP benchmarks. Our work complements recent work on learning to select cutting planes and highlights the importance of separator management.</details>

### Learning to Group Auxiliary Datasets for Molecule

**Authors:** Tinglin Huang, Ziniu Hu, Rex Ying

**<details><summary>Abstract**</summary>

The limited availability of annotations in small molecule datasets presents a challenge to machine learning models. To address this, one common strategy is to collaborate with additional auxiliary datasets. However, having more data does not always guarantee improvements. Negative transfer can occur when the knowledge in the target dataset differs or contradicts that of the auxiliary molecule datasets. In light of this, identifying the auxiliary molecule datasets that can benefit the target dataset when jointly trained remains a critical and unresolved problem. Through an empirical analysis, we observe that combining graph structure similarity and task similarity can serve as a more reliable indicator for identifying high-affinity auxiliary datasets. Motivated by this insight, we propose MolGroup, which separates the dataset affinity into task and structure affinity to predict the potential benefits of each auxiliary molecule dataset. MolGroup achieves this by utilizing a routing mechanism optimized through a bi-level optimization framework. Empowered by the meta gradient, the routing mechanism is optimized toward maximizing the target dataset's performance and quantifies the affinity as the gating score. As a result, MolGroup is capable of predicting the optimal combination of auxiliary datasets for each target dataset. Our extensive experiments demonstrate the efficiency and effectiveness of MolGroup, showing an average improvement of 4.41%/3.47% for GIN/Graphormer trained with the group of molecule datasets selected by MolGroup on 11 target molecule datasets.</details>

### Learning to Influence Human Behavior with Offline Reinforcement Learning

**Authors:** Joey Hong, Sergey Levine, Anca Dragan

**<details><summary>Abstract**</summary>

When interacting with people, AI agents do not just influence the state of the world -- they also influence the actions people take in response to the agent, and even their underlying intentions and strategies. Accounting for and leveraging this influence has mostly been studied in settings where it is sufficient to assume that human behavior is near-optimal: competitive games, or general-sum settings like autonomous driving alongside human drivers. Instead, we focus on influence in settings where there is a need to capture human suboptimality. For instance, imagine a collaborative task in which, due either to cognitive biases or lack of information, people do not perform very well -- how could an agent influence them towards more optimal behavior? Assuming near-optimal human behavior will not work here, and so the agent needs to learn from real human data. But experimenting online with humans is potentially unsafe, and creating a high-fidelity simulator of the environment is often impractical. Hence, we  focus on learning from an offline dataset of human-human interactions. Our observation is that offline reinforcement learning (RL) can learn to effectively influence suboptimal humans by extending and combining elements of observed human-human behavior. We demonstrate that offline RL can solve two challenges with effective influence. First, we show that by learning from a dataset of suboptimal human-human interaction on a variety of tasks -- none of which contains examples of successful influence -- an agent can learn influence strategies to steer humans towards better performance even on new tasks. Second, we show that by also modeling and conditioning on human behavior, offline RL can learn to affect not just the human's actions but also their underlying strategy, and adapt to changes in their strategy.</details>

### Learning with Explanation Constraints

**Authors:** Rattana Pukdee, Dylan Sam, J. Zico Kolter, Maria-Florina Balcan, Pradeep Ravikumar

**<details><summary>Abstract**</summary>

As larger deep learning models are hard to interpret, there has been a recent focus on generating explanations of these black-box models. In contrast, we may have apriori explanations of how models should behave. In this paper, we formalize this notion as learning from \emph{explanation constraints} and provide a learning theoretic framework to analyze how such explanations can improve the learning of our models. One may naturally ask, ``When would these explanations be helpful?"Our first key contribution addresses this question via a class of models that satisfies these explanation constraints in expectation over new data. We provide a characterization of the benefits of these models (in terms of the reduction of their Rademacher complexities) for a canonical class of explanations given by gradient information in the settings of both linear models and two layer neural networks. In addition, we provide an algorithmic solution for our framework, via a variational approximation that achieves better performance and satisfies these constraints more frequently, when compared to simpler augmented Lagrangian methods to incorporate these explanations. We demonstrate the benefits of our approach over a large array of synthetic and real-world experiments.</details>

### Leveraging Vision-Centric Multi-Modal Expertise for 3D Object Detection

**Authors:** Linyan Huang, Zhiqi Li, Chonghao Sima, Wenhai Wang, Jingdong Wang, Yu Qiao, Hongyang Li

**<details><summary>Abstract**</summary>

Current research is primarily dedicated to advancing the accuracy of camera-only 3D object detectors (apprentice) through the knowledge transferred from LiDAR- or multi-modal-based counterparts (expert). However, the presence of the domain gap between LiDAR and camera features, coupled with the inherent incompatibility in temporal fusion, significantly hinders the effectiveness of distillation-based enhancements for apprentices. Motivated by the success of uni-modal distillation, an apprentice-friendly expert model would predominantly rely on camera features, while still achieving comparable performance to multi-modal models. To this end, we introduce VCD, a framework to improve the camera-only apprentice model, including an apprentice-friendly multi-modal expert and temporal-fusion-friendly distillation supervision. The multi-modal expert VCD-E adopts an identical structure as that of the camera-only apprentice in order to alleviate the feature disparity, and leverages LiDAR input as a depth prior to reconstruct the 3D scene, achieving the performance on par with other heterogeneous multi-modal experts. Additionally, a fine-grained trajectory-based distillation module is introduced with the purpose of individually rectifying the motion misalignment for each object in the scene. With those improvements, our camera-only apprentice VCD-A sets new state-of-the-art on nuScenes with a score of 63.1% NDS. The code will be released at https://github.com/OpenDriveLab/Birds-eye-view-Perception.</details>

### Leveraging the two-timescale regime to demonstrate convergence of neural networks

**Authors:** Pierre Marion, Raphaël Berthier

**<details><summary>Abstract**</summary>

We study the training dynamics of shallow neural networks, in a two-timescale regime in which the stepsizes for the inner layer are much smaller than those for the outer layer. In this regime, we prove convergence of the gradient flow to a global optimum of the non-convex optimization problem in a simple univariate setting. The number of neurons need not be asymptotically large for our result to hold, distinguishing our result from popular recent approaches such as the neural tangent kernel or mean-field regimes. Experimental illustration is provided, showing that the stochastic gradient descent behaves according to our description of the gradient flow and thus converges to a global optimum in the two-timescale regime, but can fail outside of this regime.</details>

### Lift Yourself Up: Retrieval-augmented Text Generation with Self-Memory

**Authors:** Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu, Dongyan Zhao, Rui Yan

**<details><summary>Abstract**</summary>

With direct access to human-written reference as memory, retrieval-augmented generation has achieved much progress in a wide range of text generation tasks. Since better memory would typically prompt better generation (we define this as primal problem). The traditional approach for memory retrieval involves selecting memory that exhibits the highest similarity to the input. However, this method is constrained by the quality of the fixed corpus from which memory is retrieved. In this paper, by exploring the duality of the primal problem: better generation also prompts better memory, we propose a novel framework, selfmem, which addresses this limitation by iteratively employing a retrieval-augmented generator to create an unbounded memory pool and using a memory selector to choose one output as memory for the subsequent generation round. This enables the model to leverage its own output, referred to as self-memory, for improved generation. We evaluate the effectiveness of selfmem on three distinct text generation tasks: neural machine translation, abstractive text summarization, and dialogue generation, under two generation paradigms: fine-tuned small model and few-shot LLM. Our approach achieves state-of-the-art results in four directions in JRC-Acquis translation dataset, 50.3 ROUGE-1 in XSum, and 62.9 ROUGE-1 in BigPatent, demonstrating the potential of self-memory in enhancing retrieval-augmented generation models. Furthermore, we conduct thorough analyses of each component in the selfmem framework to identify current system bottlenecks and provide insights for future research.</details>

### Limits, approximation and size transferability for GNNs on sparse graphs via graphops

**Authors:** Thien Le, Stefanie Jegelka

**<details><summary>Abstract**</summary>

Can graph neural networks generalize to graphs that are different from the graphs they were trained on, e.g., in size? In this work, we study this question from a theoretical perspective. While recent work established such transferability and approximation results via graph limits, e.g., via graphons, these only apply nontrivially to dense graphs. To include frequently encountered sparse graphs such as bounded-degree or power law graphs, we take a perspective of taking limits of operators derived from graphs, such as the aggregation operation that makes up GNNs. This leads to the recently introduced limit notion of graphops (Backhausz and Szegedy, 2022). We demonstrate how the operator perspective allows us to develop quantitative bounds on the distance between a finite GNN and its limit on an infinite graph, as well as the distance between the GNN on graphs of different sizes that share structural properties, under a regularity assumption verified for various graph sequences. Our results hold for dense and sparse graphs, and various notions of graph limits.</details>

### Linear Time Algorithms for k-means with Multi-Swap Local Search

**Authors:** Junyu Huang, Qilong Feng, Ziyun Huang, Jinhui Xu, Jianxin Wang

**<details><summary>Abstract**</summary>

The local search methods have been widely used to solve the clustering problems. In practice, local search algorithms for clustering problems mainly adapt the single-swap strategy, which enables them to handle large-scale datasets and achieve linear running time in the data size. However, compared with multi-swap local search algorithms, there is a considerable gap on the approximation ratios of the single-swap local search algorithms. Although the current multi-swap local search algorithms provide small constant approximation, the proposed algorithms tend to have large polynomial running time, which cannot be used to handle large-scale datasets. In this paper, we propose a multi-swap local search algorithm for the $k$-means problem with linear running time in the data size. Given a swap size $t$, our proposed algorithm can achieve a $(50(1+\frac{1}{t})+\epsilon)$-approximation, which improves the current best result 509 (ICML 2019) with linear running time in the data size. Our proposed method, compared with previous multi-swap local search algorithms, is the first one to achieve linear running time in the data size. To obtain a more practical algorithm for the problem with better clustering quality and running time, we propose a sampling-based method which accelerates the process of clustering cost update during swaps. Besides, a recombination mechanism is proposed to find potentially better solutions. Empirical experiments show that our proposed algorithms achieve better performances compared with branch and bound solver (NeurIPS 2022) and other existing state-of-the-art local search algorithms on both small and large datasets.</details>

### [Spotlight] Locality Sensitive Hashing in Fourier Frequency Domain For Soft Set Containment Search

**Authors:** Indradyumna Roy, Rishi Agarwal, Soumen Chakrabarti, Anirban Dasgupta, Abir De

**<details><summary>Abstract**</summary>

In many search applications related to passage retrieval, text entailment, and subgraph search, the query and each 'document' is a set of elements, with a document being relevant if it contains the query. These elements are not represented by atomic IDs, but by  embedded representations, thereby extending set containment to *soft* set containment. Recent applications address soft set containment by encoding sets into fixed-size vectors and checking for elementwise *vector* *dominance*. This 0/1 property can be relaxed to an asymmetric *hinge* *distance* for scoring and ranking candidate documents. Here we focus on data-sensitive, trainable indices for fast retrieval of relevant documents. Existing LSH methods are designed for mostly symmetric or few  simple asymmetric distance functions, which are not suitable for hinge distance. Instead, we transform hinge distance into a proposed *dominance* *similarity* measure, to which we then apply a Fourier transform, thereby expressing dominance similarity as an expectation of inner products of functions in the frequency domain. Next, we approximate the expectation with an importance-sampled estimate. The overall consequence is that now we can use a traditional LSH, but in the frequency domain. To ensure that the LSH uses hash bits efficiently, we learn hash functions that are sensitive to both corpus and query distributions, mapped to the frequency domain. Our experiments show that the proposed asymmetric dominance similarity is critical to the targeted applications, and that our LSH, which we call FourierHashNet, provides a better query time vs. retrieval quality trade-off, compared to several baselines. Both the Fourier transform and the trainable hash codes contribute to performance gains.</details>

### Look Beneath the Surface: Exploiting Fundamental Symmetry for Sample-Efficient Offline RL

**Authors:** Peng Cheng, Xianyuan Zhan, zhihao wu, Wenjia Zhang, Youfang Lin, Shou cheng Song, Han Wang, Li Jiang

**<details><summary>Abstract**</summary>

Offline reinforcement learning (RL) offers an appealing approach to real-world tasks by learning policies from pre-collected datasets without interacting with the environment. However, the performance of existing offline RL algorithms heavily depends on the scale and state-action space coverage of datasets. Real-world data collection is often expensive and uncontrollable, leading to small and narrowly covered datasets and posing significant challenges for practical deployments of offline RL. In this paper, we provide a new insight that leveraging the fundamental symmetry of system dynamics can substantially enhance offline RL performance under small datasets. Specifically, we propose a Time-reversal symmetry (T-symmetry) enforced Dynamics Model (TDM), which establishes consistency between a pair of forward and reverse latent dynamics. TDM provides both well-behaved representations for small datasets and a new reliability measure for OOD samples based on compliance with the T-symmetry. These can be readily used to construct a new offline RL algorithm (TSRL) with less conservative policy constraints and a reliable latent space data augmentation procedure. Based on extensive experiments, we find TSRL achieves great performance on small benchmark datasets with as few as 1% of the original samples, which significantly outperforms the recent offline RL algorithms in terms of data efficiency and generalizability. Code is available at:https://github.com/pcheng2/TSRL</details>

### Lookaround Optimizer: $k$ steps around, 1 step average

**Authors:** Jiangtao Zhang, Shunyu Liu, Jie Song, Tongtian Zhu, Zhengqi Xu, Mingli Song

**<details><summary>Abstract**</summary>

Weight Average (WA) is an active research topic due to its simplicity in ensembling deep networks and the effectiveness in promoting generalization. Existing weight average approaches, however, are often carried out along only one training trajectory in a post-hoc manner (i.e., the weights are averaged after the entire training process is finished), which significantly degrades the diversity between networks and thus impairs the effectiveness. In this paper, inspired by weight average, we propose Lookaround, a straightforward yet effective SGD-based optimizer leading to flatter minima with better generalization. Specifically, Lookaround iterates two steps during the whole training period: the around step and the average step. In each iteration, 1) the around step starts from a common point and trains multiple networks simultaneously, each on transformed data by a different data augmentation, and 2) the average step averages these trained networks to get the averaged network, which serves as the starting point for the next iteration. The around step improves the functionality diversity while the average step guarantees the weight locality of these networks during the whole training, which is essential for WA to work. We theoretically explain the superiority of Lookaround by convergence analysis, and make extensive experiments to evaluate Lookaround on popular benchmarks including CIFAR and ImageNet with both CNNs and ViTs, demonstrating clear superiority over state-of-the-arts. Our code is available at https://github.com/Ardcy/Lookaround.</details>

### Lower Bounds on Adaptive Sensing for Matrix Recovery

**Authors:** Praneeth Kacham, David Woodruff

**<details><summary>Abstract**</summary>

We study lower bounds on adaptive sensing algorithms for recovering low rank matrices using linear measurements. Given an $n \times n$ matrix $A$, a general linear measurement $S(A)$, for an $n \times n$ matrix $S$, is just the inner product of $S$ and $A$, each treated as $n^2$-dimensional vectors. By performing as few linear measurements as possible on a rank-$r$ matrix $A$, we hope to construct a matrix $\hat{A}$ that satisfies $|A - \hat{A}|\_F^2 \le c |A|\_F^2$, for a small constant $c$. Here $|A|\_F$ denotes the Frobenius norm $(\sum_{i,j} A_{i,j}^2)^{1/2}$. It is commonly assumed that when measuring $A$ with $S$, the response is corrupted with an independent Gaussian random variable of mean $0$ and variance $\sigma^2$. Candès and Plan (IEEE Trans. Inform. Theory 2011) study non-adaptive algorithms for low rank matrix recovery using random linear measurements. They use the restricted isometry property (RIP) of Random Gaussian Matrices to give tractable algorithms to estimate $A$ from the measurements.At the edge of the noise level where recovery is information-theoretically feasible, it is known that their non-adaptive algorithms need to perform $\Omega(n^2)$ measurements, which amounts to reading the entire matrix. An important question is whether adaptivity helps in decreasing the overall number of measurements. While for the related problem of sparse recovery, adaptive algorithms have been extensively studied, as far as we are aware adaptive algorithms and lower bounds on them seem largely unexplored for matrix recovery. We show that any adaptive algorithm that uses $k$ linear measurements in each round and outputs an approximation as in (1) with probability $\ge 9/10$ must run for $t = \Omega(\log(n^2/k)/\log\log n)$ rounds. Our lower bound shows that any adaptive algorithm which uses $n^{2-\beta}$ ($\beta > 0$ is arbitrary constant) linear measurements in each round must run for $\Omega(\log n/\log\log n)$ rounds. Our techniques also readily extend to obtain lower bounds on adaptive algorithms for tensor recovery. Our hard distribution also allows us to give a measurement-vs-rounds trade-off for many sensing problems in numerical linear algebra, such as spectral norm low rank approximation, Frobenius norm low rank approximation, singular vector approximation, and more.</details>

### MIMONets: Multiple-Input-Multiple-Output Neural Networks Exploiting Computation in Superposition

**Authors:** Nicolas Menet, Michael Hersche, Geethan Karunaratne, Luca Benini, Abu Sebastian, Abbas Rahimi

**<details><summary>Abstract**</summary>

With the advent of deep learning, progressively larger neural networks have been designed to solve complex tasks. We take advantage of these capacity-rich models to lower the cost of inference by exploiting computation in superposition. To reduce the computational burden per input, we propose Multiple-Input-Multiple-Output Neural Networks (MIMONets) capable of handling many inputs at once. MIMONets augment various deep neural network architectures with variable binding mechanisms to represent an arbitrary number of inputs in a compositional data structure via fixed-width distributed representations. Accordingly, MIMONets adapt nonlinear neural transformations to process the data structure holistically, leading to a speedup nearly proportional to the number of superposed input items in the data structure. After processing in superposition, an unbinding mechanism recovers each transformed input of interest. MIMONets also provide a dynamic trade-off between accuracy and throughput by an instantaneous on-demand switching between a set of accuracy-throughput operating points, yet within a single set of fixed parameters. We apply the concept of MIMONets to both CNN and Transformer architectures resulting in MIMOConv and MIMOFormer, respectively. Empirical evaluations show that MIMOConv achieves $\approx 2$–$4	imes$ speedup at an accuracy delta within [+0.68, -3.18]% compared to WideResNet CNNs on CIFAR10 and CIFAR100. Similarly, MIMOFormer can handle $2$–$4$ inputs at once while maintaining a high average accuracy within a [-1.07, -3.43]% delta on the long range arena benchmark. Finally, we provide mathematical bounds on the interference between superposition channels in MIMOFormer. Our code is available at https://github.com/IBM/multiple-input-multiple-output-nets.</details>

### MKOR: Momentum-Enabled Kronecker-Factor-Based Optimizer Using Rank-1 Updates

**Authors:** Mohammad Mozaffari, Sikan Li, Zhao Zhang, Maryam Mehri Dehnavi

**<details><summary>Abstract**</summary>

This work proposes a Momentum-Enabled Kronecker-Factor-Based Optimizer Using Rank-1 updates, called MKOR, that improves the training time and convergence properties of deep neural networks (DNNs). Second-order techniques, while enjoying higher convergence rates vs first-order counterparts, have cubic complexity with respect to either the model size and/or the training batch size. Hence they exhibit poor scalability and performance in transformer models, e.g. large language models (LLMs), because the batch sizes in these models  scale by the attention mechanism sequence length, leading to large model size and batch sizes. MKOR's complexity is quadratic with respect to the model size, alleviating the computation bottlenecks in  second-order methods. Because of their high computation complexity, state-of-the-art implementations of second-order methods can only afford to update the second order information infrequently, and thus do not fully exploit the promise of better convergence from these updates. By reducing the communication complexity of the second-order updates as well as achieving a linear communication complexity, MKOR increases the frequency of second order updates. We also propose a hybrid version of MKOR (called MKOR-H) that mid-training falls backs to a first order optimizer if the second order updates  no longer accelerate convergence.  Our experiments show that MKOR outperforms state -of-the-art first order methods, e.g. the LAMB optimizer, and best implementations of second-order methods, i.e. KAISA/KFAC, up to 2.57x and 1.85x respectively on BERT-Large-Uncased on 64 GPUs.</details>

### Macro Placement by Wire-Mask-Guided Black-Box Optimization

**Authors:** Yunqi Shi, Ke Xue, Song Lei, Chao Qian

**<details><summary>Abstract**</summary>

The development of very large-scale integration (VLSI) technology has posed new challenges for electronic design automation (EDA) techniques in chip floorplanning. During this process, macro placement is an important subproblem, which tries to determine the positions of all macros with the aim of minimizing half-perimeter wirelength (HPWL) and avoiding overlapping. Previous methods include packing-based, analytical and reinforcement learning methods. In this paper, we propose a new black-box optimization (BBO) framework (called WireMask-BBO) for macro placement, by using a wire-mask-guided greedy procedure for objective evaluation. Equipped with different BBO algorithms, WireMask-BBO empirically achieves significant improvements over previous methods, i.e., achieves significantly shorter HPWL by using much less time. Furthermore, it can fine-tune existing placements by treating them as initial solutions, which can bring up to 50% improvement in HPWL. WireMask-BBO has the potential to significantly improve the quality and efficiency of chip floorplanning, which makes it appealing to researchers and practitioners in EDA and will also promote the application of BBO. Our code is available at https://github.com/lamda-bbo/WireMask-BBO.</details>

### Managing Temporal Resolution in Continuous Value Estimation: A Fundamental Trade-off

**Authors:** Zichen Zhang, Johannes Kirschner, Junxi Zhang, Francesco Zanini, Alex Ayoub, Masood Dehghan, Dale Schuurmans

**<details><summary>Abstract**</summary>

A default assumption in reinforcement learning (RL) and optimal control is that observations arrive at discrete time points on a fixed clock cycle. Yet, many applications involve continuous-time systems where the time discretization, in principle, can be managed. The impact of time discretization on RL methods has not been fully characterized in existing theory, but a more detailed analysis of its effect could reveal opportunities for improving data-efficiency. We address this gap by analyzing Monte-Carlo policy evaluation for LQR systems and uncover a fundamental trade-off between approximation and statistical error in value estimation. Importantly, these two errors behave differently to time discretization, leading to an optimal choice of temporal resolution for a given data budget. These findings show that managing the temporal resolution can provably improve policy evaluation efficiency in LQR systems with finite data. Empirically, we demonstrate the trade-off in numerical simulations of LQR instances and standard RL benchmarks for non-linear continuous control.</details>

### Marginal Density Ratio for Off-Policy Evaluation in Contextual Bandits

**Authors:** Muhammad Faaiz Taufiq, Arnaud Doucet, Rob Cornish, Jean-Francois Ton

**<details><summary>Abstract**</summary>

Off-Policy Evaluation (OPE) in contextual bandits is crucial for assessing new policies using existing data without costly experimentation. However, current OPE methods, such as Inverse Probability Weighting (IPW) and Doubly Robust (DR) estimators, suffer from high variance, particularly in cases of low overlap between target and behaviour policies or large action and context spaces. In this paper, we introduce a new OPE estimator for contextual bandits, the Marginal Ratio (MR) estimator, which focuses on the shift in the marginal distribution of outcomes $Y$ instead of the policies themselves. Through rigorous theoretical analysis, we demonstrate the benefits of the MR estimator compared to conventional methods like IPW and DR in terms of variance reduction. Additionally, we establish a connection between the MR estimator and the state-of-the-art Marginalized Inverse Propensity Score (MIPS) estimator, proving that MR achieves lower variance among a generalized family of MIPS estimators. We further illustrate the utility of the MR estimator in causal inference settings, where it exhibits enhanced performance in estimating Average Treatment Effects (ATE). Our experiments on synthetic and real-world datasets corroborate our theoretical findings and highlight the practical advantages of the MR estimator in OPE for contextual bandits.</details>

### [Spotlight] Masked Space-Time Hash Encoding for Efficient Dynamic Scene Reconstruction

**Authors:** Feng Wang, Zilong Chen, Guokang Wang, Yafei Song, Huaping Liu

**<details><summary>Abstract**</summary>

In this paper, we propose the Masked Space-Time Hash encoding (MSTH), a novel method for efficiently reconstructing dynamic 3D scenes from multi-view or monocular videos. Based on the observation that dynamic scenes often contain substantial static areas that result in redundancy in storage and computations, MSTH represents a dynamic scene as a weighted combination of a 3D hash encoding and a 4D hash encoding. The weights for the two components are represented by a learnable mask which is guided by an uncertainty-based objective to reflect the spatial and temporal importance of each 3D position. With this design, our method can reduce the hash collision rate by avoiding redundant queries and modifications on static areas, making it feasible to represent a large number of space-time voxels by hash tables with small size.Besides, without the requirements to fit the large numbers of temporally redundant features independently, our method is easier to optimize and converge rapidly with only twenty minutes of training for a 300-frame dynamic scene. We evaluate our method on extensive dynamic scenes. As a result, MSTH obtains consistently better results than previous state-of-the-art methods with only 20 minutes of training time and 130 MB of memory storage.</details>

### Maximum Independent Set: Self-Training through Dynamic Programming

**Authors:** Lorenzo Brusca, Lars C.P.M. Quaedvlieg, Stratis Skoulakis, Grigorios Chrysos, Volkan Cevher

**<details><summary>Abstract**</summary>

This work presents a graph neural network (GNN) framework for solving the maximum independent set (MIS) problem, inspired by dynamic programming (DP). Specifically, given a graph, we propose a DP-like recursive algorithm based on GNNs that firstly constructs two smaller sub-graphs, predicts the one with the larger MIS, and then uses it in the next recursive call. To train our algorithm, we require annotated comparisons of different graphs concerning their MIS size. Annotating the comparisons with the output of our algorithm leads to a self-training process that results in more accurate self-annotation of the comparisons and vice versa. We provide numerical evidence showing the superiority of our method vs prior methods in multiple synthetic and real-world datasets.</details>

### Meek Separators and Their Applications in Targeted Causal Discovery

**Authors:** Kirankumar Shiragur, Jiaqi Zhang, Caroline Uhler

**<details><summary>Abstract**</summary>

Learning causal structures from interventional data is a fundamental problem with broad applications across various fields. While many previous works have focused on recovering the entire causal graph, in practice, there are scenarios where learning only part of the causal graph suffices. This is called \emph{targeted} causal discovery. In our work, we focus on two such well-motivated problems: subset search and causal matching. We aim to minimize the number of interventions in both cases.Towards this, we introduce the \emph{Meek separator}, which is a subset of vertices that, when intervened, decomposes the remaining unoriented edges into smaller connected components. We then present an efficient algorithm to find Meek separators that are of small sizes. Such a procedure is helpful in designing various divide-and-conquer-based approaches. In particular, we propose two randomized algorithms that achieve logarithmic approximation for subset search and causal matching, respectively. Our results provide the first known average-case provable guarantees for both problems. We believe that this opens up possibilities to design near-optimal methods for many other targeted causal structure learning problems arising from various applications.</details>

### [Spotlight] Memory Efficient Optimizers with 4-bit States

**Authors:** Bingrui Li, Jianfei Chen, Jun Zhu

**<details><summary>Abstract**</summary>

Optimizer states are a major source of memory consumption for training neural networks, limiting the maximum trainable model within given memory budget. Compressing the optimizer states from 32-bit floating points to lower bitwidth is promising to reduce the training memory footprint, while the current lowest achievable bitwidth is 8-bit. In this work, we push optimizer states bitwidth down to 4-bit through a detailed empirical analysis of first and second moments. Specifically, we find that moments have complicated outlier patterns, that current block-wise quantization cannot accurately approximate. We use a smaller block size and propose to utilize both row-wise and column-wise information for better quantization. We further identify a zero point problem of quantizing the second moment, and solve this problem with a linear quantizer that excludes the zero point. Our 4-bit optimizers are evaluated on a wide variety of benchmarks including natural language understanding, machine translation, image classification, and instruction tuning. On all the tasks our optimizers can achieve comparable accuracy with their full-precision counterparts, while enjoying better memory efficiency.</details>

### Memory-Constrained Algorithms for Convex Optimization

**Authors:** Moise Blanchard, Junhui Zhang, Patrick Jaillet

**<details><summary>Abstract**</summary>

We propose a family of recursive cutting-plane algorithms to solve feasibility problems with constrained memory, which can also be used for first-order convex optimization. Precisely, in order to find a point within a ball of radius $\epsilon$ with a separation oracle in dimension $d$---or to minimize $1$-Lipschitz convex functions to accuracy $\epsilon$ over the unit ball---our algorithms use $\mathcal O(\frac{d^2}{p}\ln \frac{1}{\epsilon})$ bits of memory, and make $\mathcal O((C\frac{d}{p}\ln \frac{1}{\epsilon})^p)$ oracle calls. The family is parametrized by $p\in[d]$ and provides an oracle-complexity/memory trade-off in the sub-polynomial regime $\ln\frac{1}{\epsilon}\gg\ln d$. While several works gave lower-bound trade-offs (impossibility results)---we explicit here their dependence with $\ln\frac{1}{\epsilon}$, showing that these also hold in any sub-polynomial regime---to the best of our knowledge this is the first class of algorithms that provides a positive trade-off between gradient descent and cutting-plane methods in any regime with $\epsilon\leq 1/\sqrt d$. The algorithms divide the $d$ variables into $p$ blocks and optimize over blocks sequentially, with approximate separation vectors constructed using a variant of Vaidya's method. In the regime $\epsilon \leq d^{-\Omega(d)}$, our algorithm with $p=d$ achieves the information-theoretic optimal memory usage and improves the oracle-complexity of gradient descent.</details>

### Meta-Learning Adversarial Bandit Algorithms

**Authors:** Misha Khodak, Ilya Osadchiy, Keegan Harris, Maria-Florina Balcan, Kfir Y. Levy, Ron Meir, Steven Wu

**<details><summary>Abstract**</summary>

We study online meta-learning with bandit feedback, with the goal of improving performance across multiple tasks if they are similar according to some natural similarity measure.  As the first to target the adversarial online-within-online partial-information setting, we design meta-algorithms that combine outer learners to simultaneously tune the initialization and other hyperparameters of an inner learner for two important cases:  multi-armed bandits (MAB) and bandit linear optimization (BLO).  For MAB, the meta-learners initialize and set hyperparameters of the Tsallis-entropy generalization of Exp3, with the task-averaged regret improving if the entropy of the optima-in-hindsight is small.  For BLO, we learn to initialize and tune online mirror descent (OMD) with self-concordant barrier regularizers, showing that task-averaged regret varies directly with an action space-dependent measure they induce.  Our guarantees rely on proving that unregularized follow-the-leader combined with two levels of low-dimensional hyperparameter tuning is enough to learn a sequence of affine functions of non-Lipschitz and sometimes non-convex Bregman divergences bounding the regret of OMD.</details>

### Metropolis Sampling for Constrained Diffusion Models

**Authors:** Nic Fishman, Leo Klarner, Emile Mathieu, Michael Hutchinson, Valentin De Bortoli

**<details><summary>Abstract**</summary>

Denoising diffusion models have recently emerged as the predominant paradigm for generative modelling on image domains. In addition, their extension to Riemannian manifolds has facilitated a range of applications across the natural sciences. While many of these problems stand to benefit from the ability to specify arbitrary, domain-informed constraints, this setting is not covered by the existing (Riemannian) diffusion model methodology. Recent work has attempted to address this issue by constructing novel noising processes based on the reflected Brownian motion and logarithmic barrier methods. However, the associated samplers are either computationally burdensome or only apply to convex subsets of Euclidean space. In this paper, we introduce an alternative, simple noising scheme based on Metropolis sampling that affords substantial gains in computational efficiency and empirical performance compared to the earlier samplers. Of independent interest, we prove that this new process corresponds to a valid discretisation of the reflected Brownian motion. We demonstrate the scalability and flexibility of our approach on a range of problem settings with convex and non-convex constraints, including applications from geospatial modelling, robotics and protein design.</details>

### [Spotlight] Minimum-Risk Recalibration of Classifiers

**Authors:** Zeyu Sun, Dogyoon Song, Alfred Hero

**<details><summary>Abstract**</summary>

Recalibrating probabilistic classifiers is vital for enhancing the reliability and accuracy of predictive models. Despite the development of numerous recalibration algorithms, there is still a lack of a comprehensive theory that integrates calibration and sharpness (which is essential for maintaining predictive power). In this paper, we introduce the concept of minimum-risk recalibration within the framework of mean-squared-error (MSE) decomposition, offering a principled approach for evaluating and recalibrating probabilistic classifiers. Using this framework, we analyze the uniform-mass binning (UMB) recalibration method and establish a finite-sample risk upper bound of order $\tilde{O}(B/n + 1/B^2)$ where $B$ is the number of bins and $n$ is the sample size. By balancing calibration and sharpness, we further determine that the optimal number of bins for UMB scales with $n^{1/3}$, resulting in a risk bound of approximately $O(n^{-2/3})$. Additionally, we tackle the challenge of label shift by proposing a two-stage approach that adjusts the recalibration function using limited labeled data from the target domain. Our results show that transferring a calibrated classifier requires significantly fewer target samples compared to recalibrating from scratch. We validate our theoretical findings through numerical simulations, which confirm the tightness of the proposed bounds, the optimal number of bins, and the effectiveness of label shift adaptation.</details>

### Modality-Independent Teachers Meet Weakly-Supervised Audio-Visual Event Parser

**Authors:** Yung-Hsuan Lai, Yen-Chun Chen, Frank Wang

**<details><summary>Abstract**</summary>

Audio-visual learning has been a major pillar of multi-modal machine learning, where the community mostly focused on its $\textit{modality-aligned}$ setting, $\textit{i.e.}$, the audio and visual modality are $\textit{both}$ assumed to signal the prediction target.With the Look, Listen, and Parse dataset (LLP), we investigate the under-explored $\textit{unaligned}$ setting, where the goal is to recognize audio and visual events in a video with only weak labels observed.Such weak video-level labels only tell what events happen without knowing the modality they are perceived (audio, visual, or both).To enhance learning in this challenging setting, we incorporate large-scale contrastively pre-trained models as the modality teachers. A simple, effective, and generic method, termed $\textbf{V}$isual-$\textbf{A}$udio $\textbf{L}$abel Elab$\textbf{or}$ation (VALOR), is innovated to harvest modality labels for the training events.Empirical studies show that the harvested labels significantly improve an attentional baseline by $\textbf{8.0}$ in average F-score (Type@AV).Surprisingly, we found that modality-independent teachers outperform their modality-fused counterparts since they are noise-proof from the other potentially unaligned modality.Moreover, our best model achieves the new state-of-the-art on all metrics of LLP by a substantial margin ($\textbf{+5.4}$ F-score for Type@AV). VALOR is further generalized to Audio-Visual Event Localization and achieves the new state-of-the-art as well.</details>

### [Spotlight] Model Spider: Learning to Rank Pre-Trained Models Efficiently

**Authors:** Yi-Kai Zhang, Ting-Ji Huang, Yao-Xiang Ding, De-Chuan Zhan, Han-Jia Ye

**<details><summary>Abstract**</summary>

Figuring out which Pre-Trained Model (PTM) from a model zoo fits the target task is essential to take advantage of plentiful model resources. With the availability of numerous heterogeneous PTMs from diverse fields, efficiently selecting the most suitable one is challenging due to the time-consuming costs of carrying out forward or backward passes over all PTMs. In this paper, we propose Model Spider, which tokenizes both PTMs and tasks by summarizing their characteristics into vectors to enable efficient PTM selection. By leveraging the approximated performance of PTMs on a separate set of training tasks, Model Spider learns to construct representation and measure the fitness score between a model-task pair via their representation. The ability to rank relevant PTMs higher than others generalizes to new tasks. With the top-ranked PTM candidates, we further learn to enrich task repr. with their PTM-specific semantics to re-rank the PTMs for better selection. Model Spider balances efficiency and selection ability, making PTM selection like a spider preying on a web. Model Spider exhibits promising performance across diverse model zoos, including visual models and Large Language Models (LLMs). Code is available at https://github.com/zhangyikaii/Model-Spider.</details>

### Monitor-Guided Decoding of Code LMs with Static Analysis of Repository Context

**Authors:** Lakshya A Agrawal, Aditya Kanade, Navin Goyal, Shuvendu Lahiri, Sriram Rajamani

**<details><summary>Abstract**</summary>

Language models of code (LMs) work well when the surrounding code provides sufficient context. This is not true when it becomes necessary to use types, functionality or APIs defined elsewhere in the repository or a linked library, especially those not seen during training. LMs suffer from limited awareness of such global context and end up hallucinating.Integrated development environments (IDEs) assist developers in understanding repository context using static analysis. We extend this assistance, enjoyed by developers, to LMs. We propose monitor-guided decoding (MGD) where a monitor uses static analysis to guide the decoding. We construct a repository-level dataset PragmaticCode for method-completion in Java and evaluate MGD on it. On models of varying parameter scale, by monitoring for type-consistent object dereferences, MGD consistently improves compilation rates and agreement with ground truth. Further, LMs with fewer parameters, when augmented with MGD, can outperform larger LMs. With MGD, SantaCoder-1.1B achieves better compilation rate and next-identifier match than the much larger text-davinci-003 model.We also conduct a generalizability study to evaluate the ability of MGD to generalize to multiple programming languages (Java, C# and Rust), coding scenarios (e.g., correct number of arguments to method calls), and to enforce richer semantic constraints (e.g., stateful API protocols). Our data and implementation are available at https://github.com/microsoft/monitors4codegen.</details>

### Moral Responsibility for AI Systems

**Authors:** Sander Beckers

**<details><summary>Abstract**</summary>

As more and more decisions that have a significant ethical dimension are being outsourced to AI systems, it is important to have a definition of _moral responsibility_ that can be applied to AI systems. Moral responsibility for an outcome of an agent who performs some action is commonly taken to involve both a _causal condition_ and an _epistemic condition_: the action should cause the outcome, and the agent should have been aware - in some form or other - of the possible moral consequences of their action. This paper presents a formal definition of both conditions within the framework of causal models. I compare my approach to the existing approaches of Braham and van Hees (BvH) and of Halpern and Kleiman-Weiner (HK). I then generalize my definition into a _degree of responsibility_.</details>

### Most Neural Networks Are Almost Learnable

**Authors:** Amit Daniely, Nati Srebro, Gal Vardi

**<details><summary>Abstract**</summary>

We present a PTAS for learning random constant-depth networks. We show that for any fixed $\epsilon>0$ and depth $i$, there is a poly-time algorithm that for any distribution on $\sqrt{d} \cdot \mathbb{S}^{d-1}$ learns random Xavier networks of depth $i$, up to an additive error of $\epsilon$. The algorithm runs in time and sample complexity of $(\bar{d})^{\mathrm{poly}(\epsilon^{-1})}$, where $\bar d$ is the size of the network. For some  cases of sigmoid and ReLU-like activations the bound can be improved to $(\bar{d})^{\mathrm{polylog}(\epsilon^{-1})}$, resulting in a quasi-poly-time algorithm for learning constant depth random networks.</details>

### Multi Time Scale World Models

**Authors:** Vaisakh Shaj Kumar, SALEH GHOLAM ZADEH, Ozan Demir, Luiz Douat, Gerhard Neumann

**<details><summary>Abstract**</summary>

Intelligent agents use internal world models to reason and make predictions about different courses of their actions at many scales. Devising learning paradigms and architectures that allow machines to learn world models that operate at multiple levels of temporal abstractions while dealing with complex uncertainty predictions is a major technical hurdle. In this work, we propose a probabilistic formalism to learn multi-time scale world models which we call the Multi Time Scale State Space (MTS3) model. Our model uses a  computationally efficient inference scheme on multiple time scales for highly accurate long-horizon predictions and uncertainty estimates over several seconds into the future. Our experiments, which focus on action conditional long horizon future predictions, show that MTS3 outperforms recent methods on several system identification benchmarks including complex simulated and real-world dynamical systems.</details>

### Multi-Agent Meta-Reinforcement Learning: Sharper Convergence Rates with Task Similarity

**Authors:** Weichao Mao, Haoran Qiu, Chen Wang, Hubertus Franke, Zbigniew Kalbarczyk, Ravishankar Iyer, Tamer Basar

**<details><summary>Abstract**</summary>

Multi-agent reinforcement learning (MARL) has primarily focused on solving a single task in isolation, while in practice the environment is often evolving, leaving many related tasks to be solved. In this paper, we investigate the benefits of meta-learning in solving multiple MARL tasks collectively. We establish the first line of theoretical results for meta-learning in a wide range of fundamental MARL settings, including learning Nash equilibria in two-player zero-sum Markov games and Markov potential games, as well as learning coarse correlated equilibria in general-sum Markov games. Under natural notions of task similarity, we show that meta-learning achieves provable sharper convergence to various game-theoretical solution concepts than learning each task separately. As an important intermediate step, we develop multiple MARL algorithms with initialization-dependent convergence guarantees. Such algorithms integrate optimistic policy mirror descents with stage-based value updates, and their refined convergence guarantees (nearly) recover the best known results even when a good initialization is unknown. To our best knowledge, such results are also new and might be of independent interest. We further provide numerical simulations to corroborate our theoretical findings.</details>

### Multi-Fidelity Multi-Armed Bandits Revisited

**Authors:** Xuchuang Wang, Qingyun Wu, Wei Chen, John C.S. Lui

**<details><summary>Abstract**</summary>

We study the multi-fidelity multi-armed bandit ($\texttt{MF-MAB}$), an extension of the canonical multi-armed bandit (MAB) problem.$\texttt{MF-MAB}$ allows each arm to be pulled with different costs (fidelities) and observation accuracy.We study both the best arm identification with fixed confidence ($\texttt{BAI}$) and the regret minimization objectives.For $\texttt{BAI}$, we present (a) a cost complexity lower bound, (b) an algorithmic framework with two alternative fidelity selection procedures,and (c) both procedures' cost complexity upper bounds.From both cost complexity bounds of $\texttt{MF-MAB}$,one can recover the standard sample complexity bounds of the classic (single-fidelity) MAB.For regret minimization of $\texttt{MF-MAB}$, we propose a new regret definition, prove its problem-independent regret lower bound $\Omega(K^{1/3}\Lambda^{2/3})$ and problem-dependent lower bound $\Omega(K\log \Lambda)$, where $K$ is the number of arms and $\Lambda$ is the decision budget in terms of cost, and devise an elimination-based algorithm whose worst-cost regret upper bound matches its corresponding lower bound up to some logarithmic terms and, whose problem-dependent bound matches its corresponding lower bound in terms of $\Lambda$.</details>

### Multi-Head Adapter Routing for Cross-Task Generalization

**Authors:** Lucas Page-Caccia, Edoardo Maria Ponti, Zhan Su, Matheus Pereira, Nicolas Le Roux, Alessandro Sordoni

**<details><summary>Abstract**</summary>

Parameter-efficient fine-tuning (PEFT) for cross-task generalization consists in pre-training adapters on a multi-task training set before few-shot adaptation to test tasks. Polytropon [Ponti et al., 2023] ($\texttt{Poly}$) jointly learns an inventory of adapters and a *routing* function that selects a (variable-size) subset of adapters for each task during both pre-training and few-shot adaptation. In this paper, we investigate the role that adapter routing plays in its success and design new variants based on our findings.First, we build on the intuition that finer-grained routing provides more expressivity. Hence,we propose  $\texttt{MHR}$ (Multi-Head Routing) which combines *subsets* of adapter parameters and outperforms $\texttt{Poly}$ under a comparable parameter budget; by only fine-tuning the routing function and not the adapters ($\texttt{MHR}$-$z$) we achieve competitive performance with extreme parameter efficiency. Second, we find that  $\texttt{Poly}$/$\texttt{MHR}$ performance is a result of better multi-task optimization, rather than modular inductive biases that facilitate adapter recombination and local adaptation, as previously hypothesized. In fact, we find that $\texttt{MHR}$ exhibits high gradient alignment between training tasks. We find that routing is most beneficial during multi-task pre-training rather than during few-shot adaptation and propose $\texttt{MHR}$-$\mu$, which discards routing and fine-tunes the average of the pre-trained adapters on each downstream tasks. This establishes $\texttt{MHR}$-$\mu$ as an effective method for single-adapter fine-tuning. We also show that $\texttt{MHR}$-$\mu$ can be used as an effective zero-shot transfer method by training the average of the pre-trained adapters for a few additional steps on the multi-task training set: this yields gains up to 3\% on absolute accuracy w.r.t. the baselines. Code is available at .</details>

### Multi-Objective Intrinsic Reward Learning for Conversational Recommender Systems

**Authors:** Zhendong Chu, Nan Wang, Hongning Wang

**<details><summary>Abstract**</summary>

Conversational Recommender Systems (CRS) actively elicit user preferences to generate adaptive recommendations. Mainstream reinforcement learning-based CRS solutions heavily rely on handcrafted reward functions, which may not be aligned with user intent in CRS tasks. Therefore, the design of task-specific rewards is critical to facilitate CRS policy learning, which remains largely under-explored in the literature. In this work, we propose a novel approach to address this challenge by learning intrinsic rewards from interactions with users. Specifically, we formulate intrinsic reward learning as a multi-objective bi-level optimization problem. The inner level optimizes the CRS policy augmented by the learned intrinsic rewards, while the outer level drives the intrinsic rewards to optimize two CRS-specific objectives: maximizing the success rate and minimizing the number of turns to reach a successful recommendation}in conversations. To evaluate the effectiveness of our approach, we conduct extensive experiments on three public CRS benchmarks. The results show that our algorithm significantly improves CRS performance by exploiting informative learned intrinsic rewards.</details>

### Multi-modal Queried Object Detection in the Wild

**Authors:** Yifan Xu, Mengdan Zhang, Chaoyou Fu, Peixian Chen, Xiaoshan Yang, Ke Li, Changsheng Xu

**<details><summary>Abstract**</summary>

We introduce MQ-Det, an efficient architecture and pre-training strategy design to utilize both textual description with open-set generalization and visual exemplars with rich description granularity as category queries, namely, Multi-modal Queried object Detection, for real-world detection with both open-vocabulary categories and various granularity. MQ-Det incorporates vision queries into existing well-established language-queried-only detectors. A plug-and-play gated class-scalable perceiver module upon the frozen detector is proposed to augment category text with class-wise visual information. To address the learning inertia problem brought by the frozen detector, a vision conditioned masked language prediction strategy is proposed. MQ-Det's simple yet effective architecture and training strategy design is compatible with most language-queried object detectors, thus yielding versatile applications. Experimental results demonstrate that multi-modal queries largely boost open-world detection. For instance, MQ-Det significantly improves the state-of-the-art open-set detector GLIP by +7.8% AP on the LVIS benchmark via multi-modal queries without any downstream finetuning, and averagely +6.3% AP on 13 few-shot downstream tasks, with merely additional 3% modulating time required by GLIP. Code is available at https://github.com/YifanXu74/MQ-Det.</details>

### Multi-resolution Spectral Coherence for Graph Generation with Score-based Diffusion

**Authors:** Hyuna Cho, Minjae Jeong, Sooyeon Jeon, Sungsoo Ahn, Won Hwa Kim

**<details><summary>Abstract**</summary>

Successful graph generation depends on the accurate estimation of the joint distribution of graph components such as nodes and edges from training data. While recent deep neural networks have demonstrated sampling of realistic graphs together with diffusion models, however, they still suffer from oversmoothing problems which are inherited from conventional graph convolution and thus high-frequency characteristics of nodes and edges become intractable. To overcome such issues and generate graphs with high fidelity, this paper introduces a novel approach that captures the dependency between nodes and edges at multiple resolutions in the spectral space. By modeling the joint distribution of node and edge signals in a shared graph wavelet space, together with a score-based diffusion model, we propose a Wavelet Graph Diffusion Model (Wave-GD) which lets us sample synthetic graphs with real-like frequency characteristics of nodes and edges. Experimental results on four representative benchmark datasets validate the superiority of the Wave-GD over existing approaches, highlighting its potential for a wide range of applications that involve graph data.</details>

### Multi-task learning with summary statistics

**Authors:** Parker Knight, Rui Duan

**<details><summary>Abstract**</summary>

Multi-task learning has emerged as a powerful machine learning paradigm for integrating data from multiple sources, leveraging similarities between tasks to improve overall model performance. However, the application of multi-task learning to real-world settings is hindered by data-sharing constraints, especially in healthcare settings. To address this challenge, we propose a flexible multi-task learning framework utilizing summary statistics from various sources. Additionally, we present an adaptive parameter selection approach based on a variant of Lepski's method, allowing for data-driven tuning parameter selection when only summary statistics are accessible. Our systematic non-asymptotic analysis characterizes the performance of the proposed methods under various regimes of the source datasets' sample complexity and overlap.  We demonstrate our theoretical findings and the performance of the method  through extensive simulations. This work offers a more flexible tool for training related models across various domains, with  practical implications in genetic risk prediction and many other fields.</details>

### MultiMoDN—Multimodal, Multi-Task, Interpretable Modular Networks

**Authors:** Vinitra Swamy, Malika Satayeva, Jibril Frej, Thierry Bossy, Thijs Vogels, Martin Jaggi, Tanja Käser, Mary-Anne Hartley

**<details><summary>Abstract**</summary>

Predicting multiple real-world tasks in a single model often requires a particularly diverse feature space. Multimodal (MM) models aim to extract the synergistic predictive potential of multiple data types to create a shared feature space with aligned semantic meaning across inputs of drastically varying sizes (i.e. images, text, sound). Most current MM architectures fuse these representations in parallel, which not only limits their interpretability but also creates a dependency on modality availability. We present MultiModN, a multimodal, modular network that fuses latent representations in a sequence of any number, combination, or type of modality while providing granular real-time predictive feedback on any number or combination of predictive tasks. MultiModN's composable pipeline is interpretable-by-design, as well as innately multi-task and robust to the fundamental issue of biased missingness. We perform four experiments on several benchmark MM datasets across 10 real-world tasks (predicting medical diagnoses, academic performance, and weather), and show that MultiModN's sequential MM fusion does not compromise performance compared with a baseline of parallel fusion. By simulating the challenging bias of missing not-at-random (MNAR), this work shows that, contrary to MultiModN, parallel fusion baselines erroneously learn MNAR and suffer catastrophic failure when faced with different patterns of MNAR at inference. To the best of our knowledge, this is the first inherently MNAR-resistant approach to MM modeling. In conclusion, MultiModN provides granular insights, robustness, and flexibility without compromising performance.</details>

### Multiclass Boosting: Simple and Intuitive Weak Learning Criteria

**Authors:** Nataly Brukhim, Amit Daniely, Yishay Mansour, Shay Moran

**<details><summary>Abstract**</summary>

We study a generalization of boosting to the multiclass setting.We introduce a weak learning condition for multiclass classification that captures the original notion of weak learnability as being “slightly better than random guessing”. We give a simple and efficient boosting algorithm, that does not require realizability assumptions and its sample and oracle complexity bounds are independent of the number of classes. In addition, we utilize our new boosting technique in several theoretical applications within the context of List PAC Learning. First, we establish an equivalence to weak PAC learning. Furthermore, we present a new result on boosting for list learners, as well as provide a novel proof for the characterization of multiclass PAC learning and List PAC learning. Notably, our technique gives rise to simplified algorithms and analysis compared to previous works.</details>

### NPCL: Neural Processes for Uncertainty-Aware Continual Learning

**Authors:** Saurav Jha, Dong Gong, He Zhao, Lina Yao

**<details><summary>Abstract**</summary>

Continual learning (CL) aims to train deep neural networks efficiently on streaming data while limiting the forgetting caused by new tasks. However, learning transferable knowledge with less interference between tasks is difficult, and real-world deployment of CL models is limited by their inability to measure predictive uncertainties. To address these issues, we propose handling CL tasks with neural processes (NPs), a class of meta-learners that encode different tasks into probabilistic distributions over functions all while providing reliable uncertainty estimates. Specifically, we propose an NP-based CL approach (NPCL) with task-specific modules arranged in a hierarchical latent variable model. We tailor regularizers on the learned latent distributions to alleviate forgetting. The uncertainty estimation capabilities of the NPCL can also be used to handle the task head/module inference challenge in CL. Our experiments show that the NPCL outperforms previous CL approaches. We validate the effectiveness of uncertainty estimation in the NPCL for identifying novel data and evaluating instance-level model confidence. Code is available at https://github.com/srvCodes/NPCL.</details>

### NVFi: Neural Velocity Fields for 3D Physics Learning from Dynamic Videos

**Authors:** Jinxi Li, Ziyang Song, Bo Yang

**<details><summary>Abstract**</summary>

In this paper, we aim to model 3D scene dynamics from multi-view videos. Unlike the majority of existing works which usually focus on the common task of novel view synthesis within the training time period, we propose to simultaneously learn the geometry, appearance, and physical velocity of 3D scenes only from video frames, such that multiple desirable applications can be supported, including future frame extrapolation, unsupervised 3D semantic scene decomposition, and dynamic motion transfer. Our method consists of three major components, 1) the keyframe dynamic radiance field, 2) the interframe velocity field, and 3) a joint keyframe and interframe optimization module which is the core of our framework to effectively  train both networks. To validate our method, we further introduce two dynamic 3D datasets: 1) Dynamic Object dataset, and 2) Dynamic Indoor Scene dataset. We conduct extensive experiments on multiple datasets, demonstrating the superior performance of our method over all baselines, particularly in the critical tasks of future frame extrapolation and unsupervised 3D semantic scene decomposition.</details>

### NeRF Revisited: Fixing Quadrature Instability in Volume Rendering

**Authors:** Mikaela Angelina Uy, Kiyohiro Nakayama, Guandao Yang, Rahul Thomas, Leonidas Guibas, Ke Li

**<details><summary>Abstract**</summary>

Neural radiance fields (NeRF) rely on volume rendering to synthesize novel views. Volume rendering requires evaluating an integral along each ray, which is numerically approximated with a finite sum that corresponds to the exact integral along the ray under piecewise constant volume density. As a consequence, the rendered result is unstable w.r.t. the choice of samples along the ray, a phenomenon that we dub quadrature instability. We propose a mathematically principled solution by reformulating the sample-based rendering equation so that it corresponds to the exact integral under piecewise linear volume density. This simultaneously resolves multiple issues: conflicts between samples along different rays, imprecise hierarchical sampling, and non-differentiability of quantiles of ray termination distances w.r.t. model parameters. We demonstrate several benefits over the classical sample-based rendering equation, such as sharper textures, better geometric reconstruction, and stronger depth supervision. Our proposed formulation can be also be used as a drop-in replacement to the volume rendering equation of existing NeRF-based methods. Our project page can be found at pl-nerf.github.io.</details>

### Near-Optimal Algorithms for Gaussians with Huber Contamination: Mean Estimation and Linear Regression

**Authors:** Ilias Diakonikolas, Daniel Kane, Ankit Pensia, Thanasis Pittas

**<details><summary>Abstract**</summary>

We study the fundamental problems of Gaussian mean estimation and linear regression with Gaussian covariates in the presence of Huber contamination. Our main contribution is the design of the first sample near-optimal and almost linear-time algorithms with optimal error guarantees for both these problems. Specifically, for Gaussian robust mean estimation on $\mathbb R^d$ with contamination parameter $\epsilon \in (0, \epsilon_0)$ for a small absolute constant $\epsilon_0$, we give an algorithm with sample complexity $n = \tilde{O}(d/\epsilon^2)$ and almost linear runtime that approximates the target mean within $\ell_2$-error $O(\epsilon)$. This improves on prior work that achieved this error guarantee with polynomially suboptimal sample and time complexity. For robust linear regression, we give the first algorithm with sample complexity $n = \tilde{O}(d/\epsilon^2)$ and almost linear runtime that approximates the target regressor within $\ell_2$-error $O(\epsilon)$. This is the first polynomial sample and time algorithm achieving the optimal error guarantee, answering an open question in the literature. At the technical level, we develop a methodology that yields almost-linear time algorithms for multi-directional filtering that may be of broader interest.</details>

### Neural (Tangent Kernel) Collapse

**Authors:** Mariia Seleznova, Dana Weitzner, Raja Giryes, Gitta Kutyniok, Hung-Hsu Chou

**<details><summary>Abstract**</summary>

This work bridges two important concepts: the Neural Tangent Kernel (NTK), which captures the evolution of deep neural networks (DNNs) during training, and the Neural Collapse (NC) phenomenon, which refers to the emergence of symmetry and structure in the last-layer features of well-trained classification DNNs. We adopt the natural assumption that the empirical NTK develops a block structure aligned with the class labels, i.e., samples within the same class have stronger correlations than samples from different classes. Under this assumption, we derive the dynamics of DNNs trained with mean squared (MSE) loss and break them into interpretable phases. Moreover, we identify an invariant that captures the essence of the dynamics, and use it to prove the emergence of NC in DNNs with block-structured NTK. We provide large-scale numerical experiments on three common DNN architectures and three benchmark datasets to support our theory.</details>

### [Spotlight] Neural Foundations of Mental Simulation: Future Prediction of Latent Representations on Dynamic Scenes

**Authors:** Aran Nayebi, Rishi Rajalingham, Mehrdad Jazayeri, Guangyu Robert Yang

**<details><summary>Abstract**</summary>

Humans and animals have a rich and flexible understanding of the physical world, which enables them to infer the underlying dynamical trajectories of objects and events, plausible future states, and use that to plan and anticipate the consequences of actions.However, the neural mechanisms underlying these computations are unclear.We combine a goal-driven modeling approach with dense neurophysiological data and high-throughput human behavioral readouts that contain thousands of comparisons to directly impinge on this question.Specifically, we construct and evaluate several classes of sensory-cognitive networks to predict the future state of rich, ethologically-relevant environments, ranging from self-supervised end-to-end models with pixel-wise or object-slot objectives, to models that future predict in the latent space of purely static image-pretrained or dynamic video-pretrained foundation models.We find that ``scale is \emph{not} all you need'', and that many state-of-the-art machine learning models fail to perform well on our neural and behavioral benchmarks for future prediction.In fact, only one class of models matches these data well overall.We find that neural responses are currently best predicted by models trained to predict the future state of their environment in the \emph{latent} space of pretrained foundation models optimized for \emph{dynamic} scenes in a self-supervised manner.These models also approach the neurons' ability to predict the environmental state variables that are visually hidden from view, despite not being explicitly trained to do so.Finally, we find that not all foundation model latents are equal.Notably, models that future predict in the latent space of video foundation models that are optimized to support a \emph{diverse} range of egocentric sensorimotor tasks, reasonably match \emph{both} human behavioral error patterns and neural dynamics across all environmental scenarios that we were able to test.Overall, these findings suggest that the neural mechanisms and behaviors of primate mental simulation have strong inductive biases associated with them, and are thus far most consistent with being optimized to future predict on \emph{reusable} visual representations that are useful for Embodied AI more generally.</details>

### Neural Harmonics: Bridging Spectral Embedding and Matrix Completion in Self-Supervised Learning

**Authors:** Marina Munkhoeva, Ivan Oseledets

**<details><summary>Abstract**</summary>

Self-supervised methods received tremendous attention thanks to their seemingly heuristic approach to learning representations that respect the semantics of the data without any apparent supervision in the form of labels. A growing body of literature is already being published in an attempt to build a coherent and theoretically grounded understanding of the workings of a zoo of losses used in modern self-supervised representation learning methods. In this paper, we attempt to provide an understanding from the perspective of a Laplace operator and connect the inductive bias stemming from the augmentation process to a low-rank matrix completion problem.To this end, we leverage the results from low-rank matrix completion to provide theoretical analysis on the convergence of modern SSL methods and a key property that affects their downstream performance.</details>

### Neural Lighting Simulation for Urban Scenes

**Authors:** Ava Pun, Gary Sun, Jingkang Wang, Yun Chen, Ze Yang, Sivabalan Manivasagam, Wei-Chiu Ma, Raquel Urtasun

**<details><summary>Abstract**</summary>

Different outdoor illumination conditions drastically alter the appearance of urban scenes, and they can harm the performance of image-based robot perception systems if not seen during training. Camera simulation provides a cost-effective solution to create a large dataset of images captured under different lighting conditions. Towards this goal, we propose LightSim, a neural lighting camera simulation system that enables diverse, realistic, and controllable data generation. LightSim automatically builds lighting-aware digital twins at scale from collected raw sensor data and decomposes the scene into dynamic actors and static background with accurate geometry, appearance, and estimated scene lighting. These digital twins enable actor insertion, modification, removal, and rendering from a new viewpoint, all in a lighting-aware manner. LightSim then combines physically-based and learnable deferred rendering to perform realistic relighting of modified scenes, such as altering the sun location and modifying the shadows or changing the sun brightness, producing spatially- and temporally-consistent camera videos. Our experiments show that LightSim generates more realistic relighting results than prior work. Importantly, training perception models on data generated by LightSim can significantly improve their performance. Our project page is available at https://waabi.ai/lightsim/.</details>

### Neural Relation Graph: A Unified Framework for Identifying Label Noise and Outlier Data

**Authors:** Jang-Hyun Kim, Sangdoo Yun, Hyun Oh Song

**<details><summary>Abstract**</summary>

Diagnosing and cleaning data is a crucial step for building robust machine learning systems. However, identifying problems within large-scale datasets with real-world distributions is challenging due to the presence of complex issues such as label errors, under-representation, and outliers. In this paper, we propose a unified approach for identifying the problematic data by utilizing a largely ignored source of information: a relational structure of data in the feature-embedded space. To this end, we present scalable and effective algorithms for detecting label errors and outlier data based on the relational graph structure of data. We further introduce a visualization tool that provides contextual information of a data point in the feature-embedded space, serving as an effective tool for interactively diagnosing data. We evaluate the label error and outlier/out-of-distribution (OOD) detection performances of our approach on the large-scale image, speech, and language domain tasks, including ImageNet, ESC-50, and SST2. Our approach achieves state-of-the-art detection performance on all tasks considered and demonstrates its effectiveness in debugging large-scale real-world datasets across various domains. We release codes at https://github.com/snu-mllab/Neural-Relation-Graph.</details>

### Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction

**Authors:** Chih-Yu (Andrew) Lai, Fan-Keng Sun, Zhengqi Gao, Jeffrey H Lang, Duane Boning

**<details><summary>Abstract**</summary>

Time series anomaly detection is challenging due to the complexity and variety of patterns that can occur. One major difficulty arises from modeling time-dependent relationships to find contextual anomalies while maintaining detection accuracy for point anomalies. In this paper, we propose a framework for unsupervised time series anomaly detection that utilizes point-based and sequence-based reconstruction models. The point-based model attempts to quantify point anomalies, and the sequence-based model attempts to quantify both point and contextual anomalies. Under the formulation that the observed time point is a two-stage deviated value from a nominal time point, we introduce a nominality score calculated from the ratio of a combined value of the reconstruction errors. We derive an induced anomaly score by further integrating the nominality score and anomaly score, then theoretically prove the superiority of the induced anomaly score over the original anomaly score under certain conditions. Extensive studies conducted on several public datasets show that the proposed framework outperforms most state-of-the-art baselines for time series anomaly detection.</details>

### Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization

**Authors:** Quanqi Hu, Dixian Zhu, Tianbao Yang

**<details><summary>Abstract**</summary>

This paper investigates new families of compositional optimization problems, called non-smooth weakly-convex finite-sum coupled compositional optimization (NSWC FCCO). There has been a growing interest in FCCO due to its wide-ranging applications in machine learning and AI, as well as its ability to address the shortcomings of stochastic algorithms based on empirical risk minimization. However, current research on FCCO presumes that both the inner and outer functions are smooth, limiting their potential to tackle a more diverse set of problems. Our research expands on this area by examining non-smooth weakly-convex FCCO, where the outer function is weakly convex and non-decreasing, and the inner function is weakly-convex. We analyze a single-loop algorithm and establish its complexity for finding an $\epsilon$-stationary point of the Moreau envelop of the objective function. Additionally, we also extend the algorithm for solving novel non-smooth weakly-convex tri-level finite-sum coupled compositional optimization problems,  which feature a nested arrangement of three functions. Lastly, we explore the applications of our algorithms in deep learning for two-way partial AUC maximization and multi-instance two-way partial AUC maximization, using empirical studies to showcase the effectiveness of the proposed algorithms.</details>

### Non-adversarial training of Neural SDEs with signature kernel scores

**Authors:** Zacharia Issa, Blanka Horvath, Maud Lemercier, Cristopher Salvi

**<details><summary>Abstract**</summary>

Neural SDEs are continuous-time generative models for sequential data. State-of-the-art performance for irregular time series generation has been previously obtained by training these models adversarially as GANs. However, as typical for GAN architectures, training is notoriously unstable, often suffers from mode collapse, and requires specialised techniques such as weight clipping and gradient penalty to mitigate these issues. In this paper, we introduce a novel class of scoring rules on pathspace based on signature kernels and use them as objective for training Neural SDEs non-adversarially. By showing strict properness of such kernel scores and consistency of the corresponding estimators, we provide existence and uniqueness guarantees for the minimiser. With this formulation, evaluating the generator-discriminator pair amounts to solving a system of linear path-dependent PDEs which allows for memory-efficient adjoint-based backpropagation. Moreover, because the proposed kernel scores are well-defined for paths with values in infinite dimensional spaces of functions, our framework can be easily extended to generate spatiotemporal data. Our procedure significantly outperforms alternative ways of training Neural SDEs on a variety of tasks including the simulation of rough volatility models, the conditional probabilistic forecasts of real-world forex pairs where the conditioning variable is an observed past trajectory, and the mesh-free generation of limit order book dynamics.</details>

### Norm-guided latent space exploration for text-to-image generation

**Authors:** Dvir Samuel, Rami Ben-Ari, Nir Darshan, Haggai Maron, Gal Chechik

**<details><summary>Abstract**</summary>

Text-to-image diffusion models show great potential in synthesizing a large variety of concepts in new compositions and scenarios. However, the latent space of initial seeds is still not well understood and its structure was shown to impact the generation of various concepts. Specifically, simple operations like interpolation and finding the centroid of a set of seeds perform poorly when using standard Euclidean or spherical metrics in the latent space. This paper makes the observation that, in current training procedures, diffusion models observed inputs with a narrow range of norm values. This has strong implications for methods that rely on seed manipulation for image generation, with applications to few-shot and long-tail learning tasks. To address this issue, we propose a novel method for interpolating between two seeds and demonstrate that it defines a new non-Euclidean metric that takes into account a norm-based prior on seeds. We describe a simple yet efficient algorithm for approximating this interpolation procedure and use it to further define centroids in the latent seed space. We show that our new interpolation and centroid techniques significantly enhance the generation of rare concept images. This further leads to state-of-the-art performance on few-shot and long-tail benchmarks, improving prior approaches in terms of generation speed, image quality, and semantic content.</details>

### Normalization Layers Are All That Sharpness-Aware Minimization Needs

**Authors:** Maximilian Mueller, Tiffany Vlaar, David Rolnick, Matthias Hein

**<details><summary>Abstract**</summary>

Sharpness-aware minimization (SAM) was proposed to reduce sharpness of minima and has been shown to enhance generalization performance in various settings. In this work we show that perturbing only the affine normalization parameters (typically comprising 0.1% of the total parameters) in the adversarial step of SAM can outperform perturbing all of the parameters. This finding generalizesto different SAM variants and both ResNet (Batch Normalization) and Vision Transformer (Layer Normalization) architectures. We consider alternative sparse perturbation approaches and find that these do not achieve similar performance enhancement at such extreme sparsity levels, showing that this behaviour is unique to the normalization layers. Although our findings reaffirm the effectivenessof SAM in improving generalization performance, they cast doubt on whether this is solely caused by reduced sharpness.</details>

### Not All Out-of-Distribution Data Are Harmful to Open-Set Active Learning

**Authors:** Yang Yang, Yuxuan Zhang, XIN SONG, Yi Xu

**<details><summary>Abstract**</summary>

Active learning (AL) methods have been proven to be an effective way to reduce the labeling effort by intelligently selecting valuable instances for annotation. Despite their great success with in-distribution (ID) scenarios, AL methods suffer from performance degradation in many real-world applications because out-of-distribution (OOD) instances are always inevitably contained in unlabeled data, which may lead to inefficient sampling. Therefore, several attempts have been explored open-set AL by strategically selecting pure ID instances while filtering OOD instances. However, concentrating solely on selecting pseudo-ID instances may cause the training constraint of the ID classifier and OOD detector. To address this issue, we propose a simple yet effective sampling scheme, Progressive Active Learning (PAL), which employs a progressive sampling mechanism to leverage the active selection of valuable OOD instances. The proposed PAL measures unlabeled instances by synergistically evaluating instances' informativeness and representativeness, and thus it can balance the pseudo-ID and pseudo-OOD instances in each round to enhance both the capacity of the ID classifier and the OOD detector. %Meanwhile, PAL measures unlabeled instances by synergistically evaluating instances' informativeness and representativeness, which can more effectively estimate the values of instances. Extensive experiments on various open-set AL scenarios demonstrate the effectiveness of the proposed PAL, compared with the state-of-the-art methods. The code is available at \url{https://github.com/njustkmg/PAL}.</details>

### Object-Centric Slot Diffusion

**Authors:** Jindong Jiang, Fei Deng, Gautam Singh, Sungjin Ahn

**<details><summary>Abstract**</summary>

The recent success of transformer-based image generative models in object-centric learning highlights the importance of powerful image generators for handling complex scenes. However, despite the high expressiveness of diffusion models in image generation, their integration into object-centric learning remains largely unexplored in this domain. In this paper, we explore the feasibility and potential of integrating diffusion models into object-centric learning and investigate the pros and cons of this approach. We introduce Latent Slot Diffusion (LSD), a novel model that serves dual purposes: it is the first object-centric learning model to replace conventional slot decoders with a latent diffusion model conditioned on object slots, and it is also the first unsupervised compositional conditional diffusion model that operates without the need for supervised annotations like text. Through experiments on various object-centric tasks, including the first application of the FFHQ dataset in this field, we demonstrate that LSD significantly outperforms state-of-the-art transformer-based decoders, particularly in more complex scenes, and exhibits superior unsupervised compositional generation quality. In addition, we conduct a preliminary investigation into the integration of pre-trained diffusion models in LSD and demonstrate its effectiveness in real-world image segmentation and generation. Project page is available at https://latentslotdiffusion.github.io</details>

### Offline Minimax Soft-Q-learning Under Realizability and Partial Coverage

**Authors:** Masatoshi Uehara, Nathan Kallus, Jason Lee, Wen Sun

**<details><summary>Abstract**</summary>

We consider offline reinforcement learning (RL) where we only have only access to offline data. In contrast to numerous offline RL algorithms that necessitate the uniform coverage of the offline data over state and action space, we propose value-based algorithms with PAC guarantees under partial coverage, specifically, coverage of offline data against a single policy, and realizability of soft Q-function (a.k.a., entropy-regularized Q-function) and another function, which is defined as a solution to a saddle point of certain minimax optimization problem). Furthermore, we show the analogous result for Q-functions instead of soft Q-functions. To attain these guarantees, we use novel algorithms with minimax loss functions to accurately estimate soft Q-functions and Q-functions with -convergence guarantees measured on the offline data. We introduce these loss functions by casting the estimation problems into nonlinear convex optimization problems and taking the Lagrange functions.</details>

### On Certified Generalization in Structured Prediction

**Authors:** Bastian Boll, Christoph Schnörr

**<details><summary>Abstract**</summary>

In structured prediction, target objects have rich internal structure which does not factorize into independent components and violates common i.i.d. assumptions. This challenge becomes apparent through the exponentially large output space in applications such as image segmentation or scene graph generation.We present a novel PAC-Bayesian risk bound for structured prediction wherein the rate of generalization scales not only with the number of structured examples but also with their size.The underlying assumption, conforming to ongoing research on generative models, is that data are generated by the Knothe-Rosenblatt rearrangement of a factorizing reference measure. This allows to explicitly distill the structure between random output variables into a Wasserstein dependency matrix. Our work makes a preliminary step towards leveraging powerful generative models to establish generalization bounds for discriminative downstream tasks in the challenging setting of structured prediction.</details>

### [Spotlight] On Learning Necessary and Sufficient Causal Graphs

**Authors:** Hengrui Cai, Yixin Wang, Michael Jordan, Rui Song

**<details><summary>Abstract**</summary>

The causal revolution has stimulated interest in understanding complex relationships in various fields. Most of the existing methods aim to discover causal relationships among all variables within a complex large-scale graph. However, in practice, only a small subset of variables in the graph are relevant to the outcomes of interest. Consequently, causal estimation with the full causal graph---particularly given limited data---could lead to numerous *falsely discovered, spurious* variables that exhibit high correlation with, but exert no causal impact on, the target outcome. In this paper, we propose learning a class of *necessary and sufficient causal graphs (NSCG)* that exclusively comprises causally relevant variables for an outcome of interest, which we term *causal features*. The key idea is to employ *probabilities of causation* to systematically evaluate the importance of features in the causal graph, allowing us to identify a subgraph relevant to the outcome of interest. To learn NSCG from data, we develop a *necessary and sufficient causal structural learning (NSCSL)* algorithm, by establishing theoretical properties and relationships between probabilities of causation and natural causal effects of features. Across empirical studies of simulated and real data, we demonstrate that NSCSL outperforms existing algorithms and can reveal crucial yeast genes for target heritable traits of interest.</details>

### On Masked Pre-training and the Marginal Likelihood

**Authors:** Pablo Moreno-Muñoz, Pol Garcia Recasens, Søren Hauberg

**<details><summary>Abstract**</summary>

Masked pre-training removes random input dimensions and learns a model that can predict the missing values. Empirical results indicate that this intuitive form of self-supervised learning yields models that generalize very well to new domains. A theoretical understanding is, however, lacking. This paper shows that masked pre-training with a suitable cumulative scoring function corresponds to maximizing the model's marginal likelihood, which is de facto the Bayesian model selection measure of generalization. Beyond shedding light on the success of masked pre-training, this insight also suggests that Bayesian models can be trained with appropriately designed self-supervision. Empirically, we confirm the developed theory and explore the main learning principles of masked pre-training in large language models.</details>

### On Private and Robust Bandits

**Authors:** Yulian Wu, Xingyu Zhou, Youming Tao, Di Wang

**<details><summary>Abstract**</summary>

We study private and robust multi-armed bandits (MABs), where the agent receives Huber's contaminated heavy-tailed rewards and meanwhile needs to ensure differential privacy. We consider both the finite $k$-th raw moment and the finite $k$-th central moment settings for heavy-tailed rewards distributions with $k\ge 2$. We first present its minimax lower bound, characterizing the information-theoretic limit of regret with respect to privacy budget, contamination level, and heavy-tailedness. Then, we propose a meta-algorithm that builds on a private and robust mean estimation sub-routine \texttt{PRM} that essentially relies on reward truncation and the Laplace mechanism.  For the above two different heavy-tailed settings, we give corresponding schemes of \texttt{PRM}, which enable us to achieve nearly-optimal regrets.  Moreover, our two proposed truncation-based or histogram-based \texttt{PRM} schemes achieve the optimal trade-off between estimation accuracy, privacy and robustness. Finally, we support our theoretical results and show the effectiveness of our algorithms with experimental studies.</details>

### On permutation symmetries in Bayesian neural network posteriors: a variational perspective

**Authors:** Simone Rossi, Ankit Singh, Thomas Hannagan

**<details><summary>Abstract**</summary>

The elusive nature of gradient-based optimization in neural networks is tied to their loss landscape geometry, which is poorly understood. However recent work has brought solid evidence that there is essentially no loss barrier between the local solutions of gradient descent, once accounting for weight-permutations that leave the network's computation unchanged. This raises questions for approximate inference in Bayesian neural networks (BNNs), where we are interested in marginalizing over multiple points in the loss landscape.In this work, we first extend the formalism of marginalized loss barrier and solution interpolation to BNNs, before proposing a matching algorithm to search for linearly connected solutions. This is achieved by aligning the distributions of two independent approximate Bayesian solutions with respect to permutation matrices. Building on the work of Ainsworth et al. (2023), we frame the problem as a combinatorial optimization one, using an approximation to the sum of bilinear assignment problem. We then experiment on a variety of architectures and datasets, finding nearly zero marginalized loss barriers for linearly connected solutions.</details>

### On the Consistency of Maximum Likelihood Estimation of Probabilistic Principal Component Analysis

**Authors:** Arghya Datta, Sayak Chakrabarty

**<details><summary>Abstract**</summary>

Probabilistic principal component analysis (PPCA) is currently one of the most used statistical tools to reduce the ambient dimension of the data. From multidimensional scaling to the imputation of missing data, PPCA has a broad spectrum of applications ranging from science and engineering to quantitative finance.\\Despite this wide applicability in various fields, hardly any theoretical guarantees exist to justify the soundness of the maximal likelihood (ML) solution for this model. In fact, it is well known that the maximum likelihood estimation (MLE) can only recover the true model parameters up to a rotation. The main obstruction is posed by the inherent identifiability nature of the PPCA model resulting from the rotational symmetry of the parameterization. To resolve this ambiguity, we propose a novel approach using quotient topological spaces and in particular, we show that the maximum likelihood solution is consistent in an appropriate quotient Euclidean space. Furthermore, our consistency results encompass a more general class of estimators beyond the MLE. Strong consistency of the ML estimate and consequently strong covariance estimation of the PPCA model have also been established under a compactness assumption.</details>

### On the Identifiability of Sparse ICA without Assuming Non-Gaussianity

**Authors:** Ignavier Ng, Yujia Zheng, Xinshuai Dong, Kun Zhang

**<details><summary>Abstract**</summary>

Independent component analysis (ICA) is a fundamental statistical tool used to reveal hidden generative processes from observed data. However, traditional ICA approaches struggle with the rotational invariance inherent in Gaussian distributions, often necessitating the assumption of non-Gaussianity in the underlying sources. This may limit their applicability in broader contexts. To accommodate Gaussian sources, we develop an identifiability theory that relies on second-order statistics without imposing further preconditions on the distribution of sources, by introducing novel assumptions on the connective structure from sources to observed variables. Different from recent work that focuses on potentially restrictive connective structures, our proposed assumption of structural variability is both considerably less restrictive and provably necessary. Furthermore, we propose two estimation methods based on second-order statistics and sparsity constraint. Experimental results are provided to validate our identifiability theory and estimation methods.</details>

### On the Properties of Kullback-Leibler Divergence Between Multivariate Gaussian Distributions

**Authors:** Yufeng Zhang, Jialu Pan, Li Ken Li, Wanwei Liu, Zhenbang Chen, Xinwang Liu, J Wang

**<details><summary>Abstract**</summary>

Kullback-Leibler (KL) divergence is one of the most important measures to calculate the difference between probability distributions. In this paper, we theoretically study several properties of KL divergence between multivariate Gaussian distributions. Firstly, for any two $n$-dimensional Gaussian distributions $\mathcal{N}_1$ and $\mathcal{N}_2$, we prove that when $KL(\mathcal{N}_2||\mathcal{N}_1)\leq \varepsilon\ (\varepsilon>0)$ the supremum of $KL(\mathcal{N}_1||\mathcal{N}_2)$ is $(1/2)\left((-W_{0}(-e^{-(1+2\varepsilon)}))^{-1}+\log(-W_{0}(-e^{-(1+2\varepsilon)})) -1 \right)$, where $W_0$ is the principal branch of Lambert $W$ function.For small $\varepsilon$, the supremum is $\varepsilon + 2\varepsilon^{1.5} + O(\varepsilon^2)$. This quantifies the approximate symmetry of small KL divergence between Gaussian distributions. We further derive the infimum of $KL(\mathcal{N}_1||\mathcal{N}_2)$ when $KL(\mathcal{N}_2||\mathcal{N}_1)\geq M\ (M>0)$. We give the conditions when the supremum and infimum can be attained. Secondly, for any three $n$-dimensional Gaussian distributions $\mathcal{N}_1$, $\mathcal{N}_2$, and $\mathcal{N}_3$, we theoretically show that an upper bound of $KL(\mathcal{N}_1||\mathcal{N}_3)$ is $3\varepsilon_1+3\varepsilon_2+2\sqrt{\varepsilon_1\varepsilon_2}+o(\varepsilon_1)+o(\varepsilon_2)$ when $KL(\mathcal{N}_1||\mathcal{N}_2)\leq \varepsilon_1$ and $KL(\mathcal{N}_2||\mathcal{N}_3)\leq \varepsilon_2$ ($\varepsilon_1,\varepsilon_2\ge 0$). This reveals that KL divergence between Gaussian distributions follows a relaxed triangle inequality. Note that, all these bounds in the theorems presented in this work are independent of the dimension $n$. Finally, we discuss several applications of our theories in deep learning, reinforcement learning, and sample complexity research.</details>

### On the Robustness of Mechanism Design under Total Variation Distance

**Authors:** Anuran Makur, Marios Mertzanidis, Alexandros Psomas, Athina Terzoglou

**<details><summary>Abstract**</summary>

We study the problem of designing mechanisms when agents' valuation functions are drawn from unknown and correlated prior distributions. In particular, we are given a prior distribution $D$, and we are interested in designing a (truthful) mechanism that has good performance for all "true distributions" that are close to $D$ in Total Variation (TV) distance. We show that DSIC and BIC mechanisms in this setting are strongly robust with respect to TV distance, for any bounded objective function $\mathcal{O}$, extending a recent result of Brustle et al. ([BCD20], EC 2020). At the heart of our result is a fundamental duality property of total variation distance. As direct applications of our result, we (i) demonstrate how to find approximately revenue-optimal and approximately BIC mechanisms for weakly dependent prior distributions; (ii) show how to find correlation-robust mechanisms when only ``noisy'' versions of marginals are accessible, extending recent results of Bei et. al. ([BGLT19], SODA 2019); (iii) prove that prophet-inequality type guarantees are preserved for correlated priors, recovering a variant of a result of D{\"u}tting and Kesselheim ([DK19], EC 2019) as a special case; (iv) give a new necessary condition for a correlated distribution to witness an infinite separation in revenue between simple and optimal mechanisms, complementing recent results of Psomas et al. ([PSCW22], NeurIPS 2022); (v) give a new condition for simple mechanisms to approximate revenue-optimal mechanisms for the case of a single agent whose type is drawn from a correlated distribution that can be captured by a Markov Random Field, complementing recent results of Cai and Oikonomou ([CO21], EC 2021).</details>

### On the Role of Noise in the Sample Complexity of Learning Recurrent Neural Networks: Exponential Gaps for Long Sequences

**Authors:** Alireza F. Pour, Hassan Ashtiani

**<details><summary>Abstract**</summary>

We consider the class of noisy multi-layered sigmoid recurrent neural networks with $w$ (unbounded) weights for classification of sequences of length $T$, where independent noise distributed according to $\mathcal{N}(0,\sigma^2)$ is added to the output of each neuron in the network. Our main result shows that the sample complexity of PAC learning this class can be bounded by $O (w\log(T/\sigma))$. For the non-noisy version of the same class (i.e., $\sigma=0$), we prove a lower bound of $\Omega (wT)$ for the sample complexity.   Our results indicate an exponential gap in the dependence of sample complexity on $T$ for noisy versus non-noisy networks. Moreover, given the mild logarithmic dependence of the upper bound on $1/\sigma$, this gap still holds even for numerically negligible values of $\sigma$.</details>

### On the choice of Perception Loss Function for Learned Video Compression

**Authors:** Sadaf Salehkalaibar, Truong Buu Phan, Jun Chen, Wei Yu, Ashish Khisti

**<details><summary>Abstract**</summary>

We study causal, low-latency, sequential video compression when the output is subjected to both a mean squared-error (MSE) distortion loss as well as a perception loss to target realism. Motivated by prior approaches, we consider two different perception loss functions (PLFs). The first, PLF-JD,  considers the joint distribution (JD) of all the video frames up to the current one, while the second metric, PLF-FMD,  considers the framewise marginal distributions (FMD) between the source and reconstruction. Using information theoretic analysis and deep-learning based experiments, we demonstrate that the choice of PLF can have a significant effect on the reconstruction, especially at low-bit rates. In particular, while the reconstruction based on PLF-JD can better preserve the temporal correlation across frames, it also imposes a significant penalty in distortion  compared to PLF-FMD and further makes it more difficult to recover from errors  made in the earlier output frames. Although the choice of PLF decisively affects  reconstruction quality, we also demonstrate that it may not be essential to commit to a particular PLF during encoding and the choice of PLF can be delegated to the decoder. In particular, encoded representations generated by training a system to minimize the MSE (without requiring either PLF) can be  {\em near universal}  and can generate close to optimal reconstructions for either choice of PLF at the decoder.  We validate our results using (one-shot) information-theoretic analysis, detailed study of the rate-distortion-perception tradeoff of the Gauss-Markov source model as well as deep-learning based experiments on moving MNIST and KTH datasets.</details>

### On the impact of activation  and normalization in obtaining  isometric embeddings at initialization

**Authors:** Amir Joudaki, Hadi Daneshmand, Francis Bach

**<details><summary>Abstract**</summary>

In this paper, we explore the structure of the penultimate Gram matrix in deep neural networks, which contains the pairwise inner products of outputs corresponding to a batch of inputs. In several architectures it has been observed that this Gram matrix becomes degenerate with depth at initialization, which dramatically slows training. Normalization layers, such as batch or layer normalization, play a pivotal role in preventing the rank collapse issue. Despite promising advances, the existing theoretical results do not extend to layer normalization, which is widely used in transformers, and can not quantitatively characterize the role of non-linear activations.  To bridge this gap, we prove that layer normalization, in conjunction with activation layers, biases the Gram matrix of a multilayer perceptron towards the identity matrix at an exponential rate with depth at initialization. We quantify this rate using the Hermite expansion of the activation function.</details>

### [Spotlight] One Fits All: Power General Time Series Analysis by Pretrained LM

**Authors:** Tian Zhou, Peisong Niu, xue wang, Liang Sun, Rong Jin

**<details><summary>Abstract**</summary>

Although we have witnessed great success of pre-trained models in natural language processing (NLP) and computer vision (CV), limited progress has been made for general time series analysis. Unlike NLP and CV where a unified model can be used to perform different tasks, specially designed approach still dominates in each time series analysis task such as classification, anomaly detection, forecasting, and few-shot learning. The main challenge that blocks the development of pre-trained model for time series analysis is the lack of a large amount of data for training. In this work, we address this challenge by leveraging language or CV models, pre-trained from billions of tokens, for time series analysis. Specifically, we refrain from altering the self-attention and feedforward layers of the residual blocks in the pre-trained language or image model. This model, known as the Frozen Pretrained Transformer (FPT), is evaluated through fine-tuning on all major types of tasks involving time series. Our results demonstrate that pre-trained models on natural language or images can lead to a comparable or state-of-the-art performance in all main time series analysis tasks, as illustrated in Figure1. We also found both theoretically and empirically that the self-attention module behaviors similarly to principle component analysis (PCA), an observation that helps explains how transformer bridges the domain gap and a crucial step towards understanding the universality of a pre-trained transformer. The code is publicly available at https://anonymous.4open.science/r/Pretrained-LM-for-TSForcasting-C561.</details>

### One Risk to Rule Them All: A Risk-Sensitive Perspective on Model-Based Offline Reinforcement Learning

**Authors:** Marc Rigter, Bruno Lacerda, Nick Hawes

**<details><summary>Abstract**</summary>

Offline reinforcement learning (RL) is suitable for safety-critical domains where online exploration is not feasible. In such domains, decision-making should take into consideration the risk of catastrophic outcomes. In other words, decision-making should be *risk-averse*. An additional challenge of offline RL is avoiding *distributional shift*, i.e. ensuring that  state-action pairs visited by the policy remain near those in the dataset. Previous offline RL algorithms that consider risk combine offline RL techniques (to avoid distributional shift), with risk-sensitive RL algorithms (to achieve risk-aversion). In this work, we propose risk-aversion as a mechanism to jointly address *both* of these issues. We propose a model-based approach, and use an ensemble of models to estimate epistemic uncertainty, in addition to aleatoric uncertainty. We train a policy that is risk-averse, and avoids high uncertainty actions. Risk-aversion to epistemic uncertainty prevents distributional shift, as areas not covered by the dataset have high epistemic uncertainty. Risk-aversion to aleatoric uncertainty discourages actions that are risky due to environment stochasticity. Thus, by considering epistemic uncertainty via a model ensemble and introducing risk-aversion, our algorithm (1R2R) avoids distributional shift in addition to achieving risk-aversion to aleatoric risk. Our experiments show that 1R2R achieves strong performance on deterministic benchmarks, and outperforms existing approaches for risk-sensitive objectives in stochastic domains.</details>

### [Spotlight] Online (Multinomial) Logistic Bandit: Improved Regret and Constant Computation Cost

**Authors:** Yu-Jie Zhang, Masashi Sugiyama

**<details><summary>Abstract**</summary>

This paper investigates the logistic bandit problem, a variant of the generalized linear bandit model that utilizes a logistic model to depict the feedback from an action. While most existing research focuses on the binary logistic bandit problem, the multinomial case, which considers more than two possible feedback values, offers increased practical relevance and adaptability for use in complex decision-making problems such as reinforcement learning. In this paper, we provide an algorithm that enjoys both statistical and computational efficiency for the logistic bandit problem. In the binary case, our method improves the state-of-the-art binary logistic bandit method by reducing the per-round computation cost from $\mathcal{O}(\log T)$ to $\mathcal{O}(1)$ with respect to the time horizon $T$, while still preserving the minimax optimal guarantee up to logarithmic factors. In the multinomial case, with $K+1$ potential feedback values, our algorithm achieves an $\tilde{\mathcal{O}}(K\sqrt{T})$ regret bound with $\mathcal{O}(1)$ computational cost per round. The result not only improves the $\tilde{\mathcal{O}}(K\sqrt{\kappa T})$ bound for the best-known tractable algorithm—where the large constant $\kappa$ increases exponentially with the diameter of the parameter domain—but also reduces the $\mathcal{O}(T)$ computational complexity demanded by the previous method.</details>

### Online Ad Allocation with Predictions

**Authors:** Fabian Spaeh, Alina Ene

**<details><summary>Abstract**</summary>

Display Ads and the generalized assignment problem are two well-studied online packing problems with important applications in ad allocation and other areas. In both problems, ad impressions arrive online and have to be allocated immediately to budget-constrained advertisers. Worst-case algorithms that achieve the ideal competitive ratio are known for both problems, but might act overly conservative given the predictable and usually tame nature of real-world input. Given this discrepancy, we develop an algorithm for both problems that incorporate machine-learned predictions and can thus improve the performance beyond the worst-case. Our algorithm is based on the work of Feldman et al. (2009) and similar in nature to Mahdian et al. (2007) who were the first to develop a learning-augmented algorithm for the related, but more structured Ad Words problem. We use a novel analysis to show that our algorithm is able to capitalize on a good prediction, while being robust against poor predictions. We experimentally evaluate our algorithm on synthetic and real-world data on a wide range of predictions. Our algorithm is consistently outperforming the worst-case algorithm without predictions.</details>

### Online Adaptive Policy Selection in Time-Varying Systems: No-Regret via Contractive Perturbations

**Authors:** Yiheng Lin, James A. Preiss, Emile Anand, Yingying Li, Yisong Yue, Adam Wierman

**<details><summary>Abstract**</summary>

We study online adaptive policy selection in systems with time-varying costs and dynamics. We develop the Gradient-based Adaptive Policy Selection (GAPS) algorithm together with a general analytical framework for online policy selection via online optimization. Under our proposed notion of contractive policy classes, we show that GAPS approximates the behavior of an ideal online gradient descent algorithm on the policy parameters while requiring less information and computation. When convexity holds, our algorithm is the first to achieve optimal policy regret. When convexity does not hold, we provide the first local regret bound for online policy selection. Our numerical experiments show that GAPS can adapt to changing environments more quickly than existing benchmarks.</details>

### Open Compound Domain Adaptation with Object Style Compensation for Semantic Segmentation

**Authors:** Tingliang Feng, Hao Shi, Xueyang Liu, Wei Feng, Liang Wan, Yanlin Zhou, Di Lin

**<details><summary>Abstract**</summary>

Many methods of semantic image segmentation have borrowed the success of open compound domain adaptation. They minimize the style gap between the images of source and target domains, more easily predicting the accurate pseudo annotations for target domain's images that train segmentation network. The existing methods globally adapt the scene style of the images, whereas the object styles of different categories or instances are adapted improperly. This paper proposes the Object Style Compensation, where we construct the Object-Level Discrepancy Memory with multiple sets of discrepancy features. The discrepancy features in a set capture the style changes of the same category's object instances adapted from target to source domains. We learn the discrepancy features from the images of source and target domains, storing the discrepancy features in memory. With this memory, we select appropriate discrepancy features for compensating the style information of the object instances of various categories, adapting the object styles to a unified style of source domain. Our method enables a more accurate computation of the pseudo annotations for target domain's images, thus yielding state-of-the-art results on different datasets.</details>

### Open Visual Knowledge Extraction via Relation-Oriented Multimodality Model Prompting

**Authors:** Hejie Cui, Xinyu Fang, Zihan Zhang, Ran Xu, Xuan Kan, Xin Liu, Yue Yu, Manling Li, Yangqiu Song, Carl Yang

**<details><summary>Abstract**</summary>

Images contain rich relational knowledge that can help machines understand the world. Existing methods on visual knowledge extraction often rely on the pre-defined format (e.g., sub-verb-obj tuples) or vocabulary (e.g., relation types), restricting the expressiveness of the extracted knowledge. In this work, we take a first exploration to a new paradigm of open visual knowledge extraction. To achieve this, we present OpenVik which consists of an open relational region detector to detect regions potentially containing relational knowledge and a visual knowledge generator that generates format-free knowledge by prompting the large multimodality model with the detected region of interest. We also explore two data enhancement techniques for diversifying the generated format-free visual knowledge. Extensive knowledge quality evaluations highlight the correctness and uniqueness of the extracted open visual knowledge by OpenVik. Moreover, integrating our extracted knowledge across various visual reasoning applications shows consistent improvements, indicating the real-world applicability of OpenVik.</details>

### [Spotlight] Optimal Exploration for Model-Based RL in Nonlinear Systems

**Authors:** Andrew Wagenmaker, Guanya Shi, Kevin Jamieson

**<details><summary>Abstract**</summary>

Learning to control unknown nonlinear dynamical systems is a fundamental problem in reinforcement learning and control theory. A commonly applied approach is to first explore the environment (exploration), learn an accurate model of it (system identification), and then compute an optimal controller with the minimum cost on this estimated system (policy optimization). While existing work has shown that it is possible to learn a uniformly good model of the system (Mania et al., 2020), in practice, if we aim to learn a good controller with a low cost on the actual system, certain system parameters may be significantly more critical than others, and we therefore ought to focus our exploration on learning such parameters.In this work, we consider the setting of nonlinear dynamical systems and seek to formally quantify, in such settings, (a) which parameters are most relevant to learning a good controller, and (b) how we can best explore so as to minimize uncertainty in such parameters. Inspired by recent work in linear systems (Wagenmaker et al., 2021), we show that minimizing the controller loss in nonlinear systems translates to estimating the system parameters in a particular, task-dependent metric. Motivated by this, we develop an algorithm able to efficiently explore the system to reduce uncertainty in this metric, and prove a lower bound showing that our approach learns a controller at a near-instance-optimal rate. Our algorithm relies on a general reduction from policy optimization to optimal experiment design in arbitrary systems, and may be of independent interest. We conclude with experiments demonstrating the effectiveness of our method in realistic nonlinear robotic systems.</details>

### [Spotlight] Optimal Guarantees for Algorithmic Reproducibility and Gradient Complexity in Convex Optimization

**Authors:** Liang Zhang, Junchi YANG, Amin Karbasi, Niao He

**<details><summary>Abstract**</summary>

Algorithmic reproducibility measures the deviation in outputs of machine learning algorithms upon minor changes in the training process. Previous work suggests that first-order methods would need to trade-off convergence rate (gradient complexity) for better reproducibility. In this work, we challenge this perception and demonstrate that both optimal reproducibility and near-optimal convergence guarantees can be achieved for smooth convex minimization and smooth convex-concave minimax problems under various error-prone oracle settings. Particularly, given the inexact initialization oracle, our regularization-based algorithms achieve the best of both worlds -- optimal reproducibility and near-optimal gradient complexity -- for minimization and minimax optimization. With the inexact gradient oracle, the near-optimal guarantees also hold for minimax optimization. Additionally, with the stochastic gradient oracle, we show that stochastic gradient descent ascent is optimal in terms of both reproducibility and gradient complexity. We believe our results contribute to an enhanced understanding of the reproducibility-convergence trade-off in the context of convex optimization.</details>

### Optimality in Mean Estimation: Beyond Worst-Case, Beyond Sub-Gaussian, and Beyond $1+\alpha$ Moments

**Authors:** Trung Dang, Jasper Lee, Maoyuan 'Raymond' Song, Paul Valiant

**<details><summary>Abstract**</summary>

There is growing interest in improving our algorithmic understanding of fundamental statistical problems such as mean estimation, driven by the goal of understanding the fundamental limits of what we can extract from limited and valuable data.The state of the art results for mean estimation in $\mathbb{R}$ are 1) the optimal sub-Gaussian mean estimator by [Lee and Valiant, 2022], attaining the optimal sub-Gaussian error constant for all distributions with finite but unknown variance, and 2) the analysis of the median-of-means algorithm by [Bubeck, Cesa-Bianchi and Lugosi, 2013] and a matching lower bound by [Devroye, Lerasle, Lugosi, and Oliveira, 2016], characterizing the big-O optimal errors for distributions that have tails heavy enough that only a $1+\alpha$ moment exists for some $\alpha \in (0,1)$.Both of these results, however, are optimal only in the worst case.Motivated by the recent effort in the community to go "beyond the worst-case analysis" of algorithms, we initiate the fine-grained study of the mean estimation problem:Is it possible for algorithms to leverage *beneficial* features/quirks of their input distribution to *beat* the sub-Gaussian rate, without explicit knowledge of these features?We resolve this question, finding an unexpectedly nuanced answer: "Yes in limited regimes, but in general no".Given a distribution $p$, assuming *only* that it has a finite mean and absent any additional assumptions,we show how to construct a distribution $q_{n,\delta}$ such that the means of $p$ and $q$ are well-separated, yet $p$ and $q$ are impossible to distinguish with $n$ samples with probability $1-\delta$, and $q$ further preserves the finiteness of moments of $p$.Moreover, the variance of $q$ is at most twice the variance of $p$ if it exists.The main consequence of our result is that, no reasonable estimator can asymptotically achieve better than the sub-Gaussian error rate for any distribution, up to constant factors, which matches the worst-case result of [Lee and Valiant, 2022].More generally, we introduce a new definitional framework to analyze the fine-grained optimality of algorithms, which we call "neighborhood optimality", interpolating between the unattainably strong "instance optimality" and the trivially weak admissibility/Pareto optimality definitions.As an application of the new framework, we show that the median-of-means algorithm is neighborhood optimal, up to constant factors.It is an open question to find a neighborhood-optimal estimator *without* constant factor slackness.</details>

### Optimistic Rates for Multi-Task Representation Learning

**Authors:** Austin Watkins, Enayat Ullah, Thanh Nguyen-Tang, Raman Arora

**<details><summary>Abstract**</summary>

We study the problem of transfer learning via Multi-Task Representation Learning (MTRL), wherein multiple source tasks are used to learn a good common representation, and a predictor is trained on top of it for the target task. Under standard regularity assumptions on the loss function and task diversity, we provide new statistical rates on the excess risk of the target task, which demonstrate the benefit of representation learning. Importantly, our rates are optimistic, i.e., they interpolate between the standard $O(m^{-1/2})$ rate and the fast $O(m^{-1})$ rate, depending on the difficulty of the learning task, where $m$ is the number of samples for the target task. Besides the main result, we make several new contributions, including giving optimistic rates for excess risk of source tasks (multi-task learning (MTL)), a local Rademacher complexity theorem for MTRL and MTL, as well as a chain rule for local Rademacher complexity for composite predictor classes.</details>

### [Oral] Optimizing Solution-Samplers for Combinatorial Problems: The Landscape of Policy-Gradient Method

**Authors:** Constantine Caramanis, Dimitris Fotakis, Alkis Kalavasis, Vasilis Kontonis, Christos Tzamos

**Oral Presentation:** Th, Dec 14, 08:30 -- Oral 5C

**<details><summary>Abstract**</summary>

Deep Neural Networks and Reinforcement Learning methods have empirically shown great promise in tackling challenging combinatorial problems. In those methods a deep neural network is used as a solution generator which is then trained by gradient-based methods (e.g., policy gradient) to successively obtain better solution distributions.In this work we introduce a novel theoretical framework for analyzing the effectiveness of such methods. We ask whether there exist generative models that (i) are expressive enough to generate approximately optimal solutions; (ii) have a tractable, i.e, polynomial in the size of the input, number of parameters; (iii) their optimization landscape is benign in the sense that it does not contain sub-optimal stationary points. Our main contribution is a positive answer to this question. Our result holds for a broad class of combinatorial problems including Max- and Min-Cut, Max-$k$-CSP, Maximum-Weight-Bipartite-Matching, and the Traveling Salesman Problem. As a byproduct of our analysis we introduce a novel regularization process over vanilla gradient descent and provide theoretical and experimental evidence that it helps address vanishing-gradient issues and escape bad stationary points.</details>

### PCF-GAN: generating sequential data via the characteristic function of measures on the path space

**Authors:** Hang Lou, Siran Li, Hao Ni

**<details><summary>Abstract**</summary>

Generating high-fidelity time series data using generative adversarial networks (GANs) remains a challenging task, as it is difficult to capture the temporal dependence of joint probability distributions induced by time-series data. Towards this goal, a key step is the development of an effective discriminator to distinguish between time series distributions. We propose the so-called PCF-GAN, a novel GAN that incorporates the path characteristic function (PCF) as the principled representation of time series distribution into the discriminator to enhance its generative performance.  On the one hand, we establish theoretical foundations of the PCF distance by proving its characteristicity, boundedness, differentiability with respect to generator parameters, and weak continuity, which ensure the stability and feasibility of training the PCF-GAN. On the other hand, we design efficient initialisation and optimisation schemes for PCFs to strengthen the discriminative power and accelerate training efficiency. To further boost the capabilities of complex time series generation, we integrate the auto-encoder structure via sequential embedding into the PCF-GAN, which provides additional reconstruction functionality. Extensive numerical experiments on various datasets demonstrate the consistently superior performance of PCF-GAN over state-of-the-art baselines, in both generation and reconstruction quality.</details>

### PDF: Point Diffusion Implicit Function for Large-scale Scene Neural Representation

**Authors:** Yuhan Ding, Fukun Yin, Jiayuan Fan, Hui Li, Xin Chen, Wen Liu, Chongshan Lu, Gang Yu, Tao Chen

**<details><summary>Abstract**</summary>

Recent advances in implicit neural representations have achieved impressive results by sampling and fusing individual points along sampling rays in the sampling space. However, due to the explosively growing sampling space, finely representing and synthesizing detailed textures remains a challenge for unbounded large-scale outdoor scenes. To alleviate the dilemma of using individual points to perceive the entire colossal space, we explore learning the surface distribution of the scene to provide structural priors and reduce the samplable space and propose a Point Diffusion implicit Function, PDF, for large-scale scene neural representation. The core of our method is a large-scale point cloud super-resolution diffusion module that enhances the sparse point cloud reconstructed from several training images into a dense point cloud as an explicit prior. Then in the rendering stage, only sampling points with prior points within the sampling radius are retained. That is, the sampling space is reduced from the unbounded space to the scene surface. Meanwhile, to fill in the background of the scene that cannot be provided by point clouds, the region sampling based on Mip-NeRF 360 is employed to model the background representation. Expensive experiments have demonstrated the effectiveness of our method for large-scale scene novel view synthesis, which outperforms relevant state-of-the-art baselines.</details>

### PDP: Parameter-free Differentiable Pruning is All You Need

**Authors:** Minsik Cho, Saurabh Adya, Devang Naik

**<details><summary>Abstract**</summary>

DNN pruning is a popular way to reduce the size of a model, improve the inferencelatency, and minimize the power consumption on DNN accelerators. However,existing approaches might be too complex, expensive or ineffective to apply toa variety of vision/language tasks, DNN architectures and to honor structuredpruning constraints. In this paper, we propose an efficient yet effective train-timepruning scheme, Parameter-free Differentiable Pruning (PDP), which offers state-of-the-art qualities in model size, accuracy, and training cost. PDP uses a dynamicfunction of weights during training to generate soft pruning masks for the weightsin a parameter-free manner for a given pruning target. While differentiable, thesimplicity and efficiency of PDP make it universal enough to deliver state-of-the-artrandom/structured/channel pruning results on various vision and natural languagetasks. For example, for MobileNet-v1, PDP can achieve 68.2% top-1 ImageNet1kaccuracy at 86.6% sparsity, which is 1.7% higher accuracy than those from thestate-of-the-art algorithms. Also, PDP yields over 83.1% accuracy on Multi-GenreNatural Language Inference with 90% sparsity for BERT, while the next best fromthe existing techniques shows 81.5% accuracy. In addition, PDP can be applied tostructured pruning, such as N:M pruning and channel pruning. For 1:4 structuredpruning of ResNet18, PDP improved the top-1 ImageNet1k accuracy by over 3.6%over the state-of-the-art. For channel pruning of ResNet50, PDP reduced the top-1ImageNet1k accuracy by 0.6% from the state-of-the-art.</details>

### PHOTOSWAP: Personalized Subject Swapping in Images

**Authors:** Jing Gu, Yilin Wang, Nanxuan Zhao, Tsu-Jui Fu, Wei Xiong, Qing Liu, Zhifei Zhang, HE Zhang, Jianming Zhang, HyunJoon Jung, Xin Eric Wang

**<details><summary>Abstract**</summary>

In an era where images and visual content dominate our digital landscape, the ability to manipulate and personalize these images has become a necessity.Envision seamlessly substituting a tabby cat lounging on a sunlit window sill in a photograph with your own playful puppy, all while preserving the original charm and composition of the image. We present \emph{Photoswap}, a novel approach that enables this immersive image editing experience through personalized subject swapping in existing images.\emph{Photoswap} first learns the visual concept of the subject from reference images and then swaps it into the target image using pre-trained diffusion models in a training-free manner. We establish that a well-conceptualized visual subject can be seamlessly transferred to any image with appropriate self-attention and cross-attention manipulation, maintaining the pose of the swapped subject and the overall coherence of the image. Comprehensive experiments underscore the efficacy and controllability of \emph{Photoswap} in personalized subject swapping. Furthermore, \emph{Photoswap} significantly outperforms baseline methods in human ratings across subject swapping, background preservation, and overall quality, revealing its vast application potential, from entertainment to professional editing.</details>

### PID-Inspired Inductive Biases for Deep Reinforcement Learning in Partially Observable Control Tasks

**Authors:** Ian Char, Jeff Schneider

**<details><summary>Abstract**</summary>

Deep reinforcement learning (RL) has shown immense potential for learning to control systems through data alone. However, one challenge deep RL faces is that the full state of the system is often not observable. When this is the case, the policy needs to leverage the history of observations to infer the current state. At the same time, differences between the training and testing environments makes it critical for the policy not to overfit to the sequence of observations it sees at training time. As such, there is an important balancing act between having the history encoder be flexible enough to extract relevant information, yet be robust to changes in the environment. To strike this balance, we look to the PID controller for inspiration. We assert the PID controller's success shows that only summing and differencing are needed to accumulate information over time for many control tasks. Following this principle, we propose two architectures for encoding history: one that directly uses PID features and another that extends these core ideas and can be used in arbitrary control tasks. When compared with prior approaches, our encoders produce policies that are often more robust and achieve better performance on a variety of tracking tasks. Going beyond tracking tasks, our policies achieve 1.7x better performance on average over previous state-of-the-art methods on a suite of locomotion control tasks.</details>

### POP-3D: Open-Vocabulary 3D Occupancy Prediction from Images

**Authors:** Antonin Vobecky, Oriane Siméoni, David Hurych, Spyridon Gidaris, Andrei Bursuc, Patrick Pérez, Josef Sivic

**<details><summary>Abstract**</summary>

We describe an approach to predict open-vocabulary 3D semantic voxel occupancy map from input 2D images with the objective of enabling 3D grounding, segmentation and retrieval of free-form language queries. This is a challenging problem because of the 2D-3D ambiguity and the open-vocabulary nature of the target tasks, where obtaining annotated training data in 3D is difficult. The contributions of this work are three-fold. First, we design a new model architecture for open-vocabulary 3D semantic occupancy prediction.  The architecture consists of a 2D-3D encoder together with occupancy prediction and 3D-language heads. The output is a dense voxel map of 3D grounded language embeddings enabling a range of open-vocabulary tasks. Second, we develop a tri-modal self-supervised learning algorithm that leverages three modalities: (i) images, (ii) language and (iii) LiDAR point clouds, and enables training the proposed architecture using a strong pre-trained vision-language model without the need for any 3D manual language annotations. Finally, we demonstrate quantitatively the strengths of the proposed model on several open-vocabulary tasks: Zero-shot 3D semantic segmentation using existing datasets; 3D grounding and retrieval of free-form language queries, using a small dataset that we propose as an extension of nuScenes.</details>

### PRED: Pre-training via Semantic Rendering on LiDAR Point Clouds

**Authors:** Hao Yang, Haiyang Wang, Di Dai, Liwei Wang

**<details><summary>Abstract**</summary>

Pre-training is crucial in 3D-related fields such as autonomous driving where point cloud annotation is costly and challenging. Many recent studies on point cloud pre-training, however, have overlooked the issue of incompleteness, where only a fraction of the points are captured by LiDAR, leading to ambiguity during the training phase. On the other hand, images offer more comprehensive information and richer semantics that can bolster point cloud encoders in addressing the incompleteness issue inherent in point clouds. Yet, incorporating images into point cloud pre-training presents its own challenges due to occlusions, potentially causing misalignments between points and pixels. In this work, we propose PRED, a novel image-assisted pre-training framework for outdoor point clouds in an occlusion-aware manner. The main ingredient of our framework is a Birds-Eye-View (BEV) feature map conditioned semantic rendering, leveraging the semantics of images for supervision through neural rendering. We further enhance our model's performance by incorporating point-wise masking with a high mask ratio (95%). Extensive experiments demonstrate PRED's superiority over prior point cloud pre-training methods, providing significant improvements on various large-scale datasets for 3D perception tasks. Codes will be available at https://github.com/PRED4pc/PRED.</details>

### Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies

**Authors:** Wei Fang, Zhaofei Yu, Zhaokun Zhou, Ding Chen, Yanqi Chen, Zhengyu Ma, Timothée Masquelier, Yonghong Tian

**<details><summary>Abstract**</summary>

Vanilla spiking neurons in Spiking Neural Networks (SNNs) use charge-fire-reset neuronal dynamics, which can only be simulated serially and can hardly learn long-time dependencies. We find that when removing reset, the neuronal dynamics can be reformulated in a non-iterative form and parallelized. By rewriting neuronal dynamics without reset to a general formulation, we propose the Parallel Spiking Neuron (PSN), which generates hidden states that are independent of their predecessors, resulting in parallelizable neuronal dynamics and extremely high simulation speed. The weights of inputs in the PSN are fully connected, which maximizes the utilization of temporal information. To avoid the use of future inputs for step-by-step inference, the weights of the PSN can be masked, resulting in the masked PSN. By sharing weights across time-steps based on the masked PSN, the sliding PSN is proposed to handle sequences of varying lengths. We evaluate the PSN family on simulation speed and temporal/static data classification, and the results show the overwhelming advantage of the PSN family in efficiency and accuracy. To the best of our knowledge, this is the first study about parallelizing spiking neurons and can be a cornerstone for the spiking deep learning research. Our codes are available at https://github.com/fangwei123456/Parallel-Spiking-Neuron.</details>

### [Spotlight] Parallel Submodular Function Minimization

**Authors:** Deeparnab Chakrabarty, Andrei Graur, Haotian Jiang, Aaron Sidford

**<details><summary>Abstract**</summary>

We consider the parallel complexity of submodular function minimization (SFM).     We provide a pair of methods which obtain two new query versus depth trade-offs a submodular function defined on subsets of $n$ elements that has integer values between $-M$ and $M$. The first method has depth $2$ and query complexity $n^{O(M)}$ and the second method has depth $\widetilde{O}(n^{1/3} M^{2/3})$ and query complexity $O(\mathrm{poly}(n, M))$. Despite a line of work on improved parallel lower bounds for SFM, prior to our work the only known algorithms for parallel SFM either followed from more general methods for sequential SFM or highly-parallel minimization of convex $\ell_2$-Lipschitz functions. Interestingly, to obtain our second result we provide the first highly-parallel algorithm for minimizing $\ell_\infty$-Lipschitz function over the hypercube which obtains near-optimal depth for obtaining constant accuracy.</details>

### Parameter-efficient Tuning of Large-scale Multimodal Foundation Model

**Authors:** Haixin Wang, Xinlong Yang, Jianlong Chang, Dian Jin, Jinan Sun, Shikun Zhang, Xiao Luo, Qi Tian

**<details><summary>Abstract**</summary>

Driven by the progress of large-scale pre-training, parameter-efficient transfer learning has gained immense popularity across different subfields of Artificial Intelligence. The core is to adapt the model to downstream tasks with only a small set of parameters. Recently, researchers have leveraged such proven techniques in multimodal tasks and achieve promising results. However, two critical issues remain unresolved: how to further reduce the complexity with lightweight design and how to boost alignment between modalities under extremely low parameters. In this paper, we propose A gracefUl pRompt framewOrk for cRoss-modal trAnsfer (AURORA) to overcome these challenges. Considering the redundancy in existing architectures, we first utilize the mode approximation to generate 0.1M trainable parameters to implement the multimodal parameter-efficient tuning, which explores the low intrinsic dimension with only 0.04% parameters of the pre-trained model. Then, for better modality alignment, we propose the Informative Context Enhancement and Gated Query Transformation module under extremely few parameters scenes. A thorough evaluation on six cross-modal benchmarks shows that it not only outperforms the state-of-the-art but even outperforms the full fine-tuning approach. Our code is available at: https://github.com/WillDreamer/Aurora.</details>

### [Spotlight] Parsel🐍: Algorithmic Reasoning with Language Models by Composing Decompositions

**Authors:** Eric Zelikman, Qian Huang, Gabriel Poesia, Noah Goodman, Nick Haber

**<details><summary>Abstract**</summary>

Despite recent success in large language model (LLM) reasoning, LLMs struggle with hierarchical multi-step reasoning tasks like generating complex programs. For these tasks, humans often start with a high-level algorithmic design and implement each part gradually. We introduce Parsel, a framework enabling automatic implementation and validation of complex algorithms with code LLMs. With Parsel, we automatically decompose algorithmic tasks into hierarchical natural language function descriptions and then search over combinations of possible function implementations using tests. We show that Parsel can be used across domains requiring hierarchical reasoning, including program synthesis and robotic planning. We find that, using Parsel, LLMs solve more competition-level problems in the APPS dataset, resulting in pass rates over 75\% higher than prior results from directly sampling AlphaCode and Codex, while often using a smaller sample budget. Moreover, with automatically generated tests, we find that Parsel can improve the state-of-the-art pass@1 performance on HumanEval from 67\% to 85\%. We also find that LLM-generated robotic plans using Parsel are more than twice as likely to be considered accurate than directly generated plans. Lastly, we explore how Parsel addresses LLM limitations and discuss how Parsel may be useful for human programmers. We release our code at https://github.com/ezelikman/parsel.</details>

### Partial Matrix Completion

**Authors:** Elad Hazan, Adam Tauman Kalai, Varun Kanade, Clara Mohri, Y. Jennifer Sun

**<details><summary>Abstract**</summary>

The matrix completion problem involves reconstructing a low-rank matrix by using a given set of revealed (and potentially noisy) entries. Although existing methods address the completion of the entire matrix, the accuracy of the completed entries can vary significantly across the matrix, due to differences in the sampling distribution. For instance, users may rate movies primarily from their country or favorite genres, leading to inaccurate predictions for the majority of completed entries.We propose a novel formulation of the problem as Partial Matrix Completion, where the objective is to complete a substantial subset of the entries with high confidence. Our algorithm efficiently handles the unknown and arbitrarily complex nature of the sampling distribution, ensuring high accuracy for all completed entries and sufficient coverage across the matrix. Additionally, we introduce an online version of the problem and present a low-regret efficient algorithm based on iterative gradient updates. Finally, we conduct a preliminary empirical evaluation of our methods.</details>

### Particle-based Variational Inference with Generalized Wasserstein Gradient Flow

**Authors:** Ziheng Cheng, Shiyue Zhang, Longlin Yu, Cheng Zhang

**<details><summary>Abstract**</summary>

Particle-based variational inference methods (ParVIs) such as Stein variational gradient descent (SVGD) update the particles based on the kernelized Wasserstein gradient flow for the Kullback-Leibler (KL) divergence. However, the design of kernels is often non-trivial and can be restrictive for the flexibility of the method. Recent works show that functional gradient flow approximations with quadratic form regularization terms can improve performance. In this paper, we propose a ParVI framework, called generalized Wasserstein gradient descent (GWG), based on a generalized Wasserstein gradient flow of the KL divergence, which can be viewed as a functional gradient method with a broader class of regularizers induced by convex functions. We show that GWG exhibits strong convergence guarantees. We also provide an adaptive version that automatically chooses Wasserstein metric to accelerate convergence. In experiments, we demonstrate the effectiveness and efficiency of the proposed framework on both simulated and real data problems.</details>

### Parts of Speech–Grounded Subspaces in Vision-Language Models

**Authors:** James Oldfield, Christos Tzelepis, Yannis Panagakis, Mihalis Nicolaou, Ioannis Patras

**<details><summary>Abstract**</summary>

Latent image representations arising from vision-language models have proved immensely useful for a variety of downstream tasks. However, their utility is limited by their entanglement with respect to different visual attributes. For instance, recent work has shown that CLIP image representations are often biased toward specific visual properties (such as objects or actions) in an unpredictable manner. In this paper, we propose to separate representations of the different visual modalities in CLIP’s joint vision-language space by leveraging the association between parts of speech and specific visual modes of variation (e.g. nouns relate to objects, adjectives describe appearance). This is achieved by formulating an appropriate component analysis model that learns subspaces capturing variability corresponding to a specific part of speech, while jointly minimising variability to the rest. Such a subspace yields disentangled representations of the different visual properties of an image or text in closed form while respecting the underlying geometry of the manifold on which the representations lie. What’s more, we show the proposed model additionally facilitates learning subspaces corresponding to specific visual appearances (e.g. artists’ painting styles), which enables the selective removal of entire visual themes from CLIP-based text-to-image synthesis. We validate the model both qualitatively, by visualising the subspace projections with a text-to-image model and by preventing the imitation of artists’ styles, and quantitatively, through class invariance metrics and improvements to baseline zero-shot classification.</details>

### Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models

**Authors:** Zhendong Wang, Yifan Jiang, Huangjie Zheng, Peihao Wang, Pengcheng He, Zhangyang "Atlas" Wang, Weizhu Chen, Mingyuan Zhou

**<details><summary>Abstract**</summary>

Diffusion models are powerful, but they require a lot of time and data to train. We propose Patch Diffusion, a generic patch-wise training framework, to significantly reduce the training time costs while improving data efficiency, which thus helps democratize diffusion model training to broader users. At the core of our innovations is a new conditional score function at the patch level, where the patch location in the original image is included as additional coordinate channels, while the patch size is randomized and diversified throughout training to encode the cross-region dependency at multiple scales. Sampling with our method is as easy as in the original diffusion model. Through Patch Diffusion, we could achieve $\mathbf{\ge 2	imes}$ faster training, while maintaining comparable or better generation quality. Patch Diffusion meanwhile improves the performance of diffusion models trained on relatively small datasets, $e.g.$, as few as 5,000 images to train from scratch. We achieve outstanding FID scores in line with state-of-the-art benchmarks: 1.77 on CelebA-64$	imes$64, 1.93 on AFHQv2-Wild-64$	imes$64, and 2.72 on ImageNet-256$	imes$256. We share our code and pre-trained models at https://github.com/Zhendong-Wang/Patch-Diffusion.</details>

### Pengi: An Audio Language Model for Audio Tasks

**Authors:** Soham Deshmukh, Benjamin Elizalde, Rita Singh, Huaming Wang

**<details><summary>Abstract**</summary>

In the domain of audio processing, Transfer Learning has facilitated the rise of Self-Supervised Learning and Zero-Shot Learning techniques. These approaches have led to the development of versatile models capable of tackling a wide array of tasks, while delivering state-of-the-art performance. However, current models inherently lack the capacity to produce the requisite language for open-ended tasks, such as Audio Captioning or Audio Question Answering. We introduce Pengi, a novel Audio Language Model that leverages Transfer Learning by framing all audio tasks as text-generation tasks. It takes as input, an audio recording, and text, and generates free-form text as output. The input audio is represented as a sequence of continuous embeddings by an audio encoder. A text encoder does the same for the corresponding text input. Both sequences are combined as a prefix to prompt a pre-trained frozen language model. The unified architecture of Pengi enables open-ended tasks and close-ended tasks without any additional fine-tuning or task-specific extensions. When evaluated on 21 downstream tasks, our approach yields state-of-the-art performance in several of them. Our results show that connecting language models with audio models is a major step towards general-purpose audio understanding.</details>

### Performance Scaling via Optimal Transport: Enabling Data Selection from Partially Revealed Sources

**Authors:** Feiyang Kang, Hoang Anh Just, Anit Kumar Sahu, Ruoxi Jia

**<details><summary>Abstract**</summary>

Traditionally, data selection has been studied in settings where all samples from prospective sources are fully revealed to a machine learning developer. However, in practical data exchange scenarios, data providers often reveal only a limited subset of samples before an acquisition decision is made. Recently, there have been efforts to fit scaling functions that predict model performance at any *size and data source composition* using the limited available samples. However, these scaling functions are usually black-box, computationally expensive to fit, highly susceptible to overfitting, or/and difficult to optimize for data selection. This paper proposes a framework called **, which predicts model performance and supports data selection decisions based on partial samples of prospective data sources. Our approach distinguishes itself from existing work by introducing a novel *two-stage* performance inference process. In the first stage, we leverage the Optimal Transport distance to predict the model's performance for any data mixture ratio within the range of disclosed data sizes. In the second stage, we extrapolate the performance to larger undisclosed data sizes based on a novel parameter-free mapping technique inspired by neural scaling laws. We further derive an efficient gradient-based method to select data sources based on the projected model performance. Evaluation over a diverse range of applications (e.g., vision, text, fine-tuning, noisy data sources, etc.) demonstrates that ** significantly improves existing performance scaling approaches in terms of both the accuracy of performance inference and the computation costs associated with constructing the performance predictor. Also, ** outperforms by a wide margin in data selection effectiveness compared to a range of other off-the-shelf solutions. We provide ** an open-source toolkit.</details>

### Permutation Equivariant Neural Functionals

**Authors:** Allan Zhou, Kaien Yang, Kaylee Burns, Adriano Cardace, Yiding Jiang, Samuel Sokota, J. Zico Kolter, Chelsea Finn

**<details><summary>Abstract**</summary>

This work studies the design of neural networks that can process the weights or gradients of other neural networks, which we refer to as *neural functional networks* (NFNs). Despite a wide range of potential applications, including learned optimization, processing implicit neural representations, network editing, and policy evaluation, there are few unifying principles for designing effective architectures that process the weights of other networks. We approach the design of neural functionals through the lens of symmetry, in particular by focusing on the permutation symmetries that arise in the weights of deep feedforward networks because hidden layer neurons have no inherent order. We introduce a framework for building *permutation equivariant* neural functionals, whose architectures encode these symmetries as an inductive bias. The key building blocks of this framework are *NF-Layers* (neural functional layers) that we constrain to be permutation equivariant through an appropriate parameter sharing scheme. In our experiments, we find that permutation equivariant neural functionals are effective on a diverse set of tasks that require processing the weights of MLPs and CNNs, such as predicting classifier generalization, producing "winning ticket" sparsity masks for initializations, and classifying or editing implicit neural representations (INRs). In addition, we provide code for our models and experiments at https://github.com/AllanYangZhou/nfn.</details>

### Point Cloud Completion with Pretrained Text-to-Image Diffusion Models

**Authors:** Yoni Kasten, Ohad Rahamim, Gal Chechik

**<details><summary>Abstract**</summary>

Point cloud data collected in real-world applications are often incomplete. This is because they are observed from partial viewpoints, which capture only a specific perspective or angle, or due to occlusion and low resolution. Existing completion approaches rely on datasets of specific predefined objects to guide the completion of incomplete, and possibly noisy, point clouds. However, these approaches perform poorly with Out-Of-Distribution (OOD) objects, which are either absent from the dataset or poorly represented. In recent years, the field of text-guided image generation has made significant progress, leading to major breakthroughs in text guided shape generation. We describe an approach called SDS-Complete that uses a pre-trained text-to-image diffusion model and leverages the text semantic of a given incomplete point cloud of an object, to obtain a complete surface representation. SDS-Complete can complete a variety of objects at test time optimization without the need for an expensive collection of 3D information. We evaluate SDS-Complete on incomplete scanned objects, captured by real-world depth sensors and LiDAR scanners, and demonstrate that is effective in handling objects which are typically absent from common datasets.</details>

### PointGPT: Auto-regressively Generative Pre-training from Point Clouds

**Authors:** Guangyan Chen, Meiling Wang, Yi Yang, Kai Yu, Li Yuan, Yufeng Yue

**<details><summary>Abstract**</summary>

Large language models (LLMs) based on the generative pre-training transformer (GPT) have demonstrated remarkable effectiveness across a diverse range of downstream tasks. Inspired by the advancements of the GPT, we present PointGPT, a novel approach that extends the concept of GPT to point clouds, addressing the challenges associated with disorder properties, low information density, and task gaps. Specifically, a point cloud auto-regressive generation task is proposed to pre-train transformer models. Our method partitions the input point cloud into multiple point patches and arranges them in an ordered sequence based on their spatial proximity. Then, an extractor-generator based transformer decode, with a dual masking strategy, learns latent representations conditioned on the preceding point patches, aiming to predict the next one in an auto-regressive manner. To explore scalability and enhance performance, a larger pre-training dataset is collected. Additionally, a subsequent post-pre-training stage is introduced, incorporating a labeled hybrid dataset. Our scalable approach allows for learning high-capacity models that generalize well, achieving state-of-the-art performance on various downstream tasks. In particular, our approach achieves classification accuracies of 94.9% on the ModelNet40 dataset and 93.4% on the ScanObjectNN dataset, outperforming all other transformer models. Furthermore, our method also attains new state-of-the-art accuracies on all four few-shot learning benchmarks. Codes are available at https://github.com/CGuangyan-BIT/PointGPT.</details>

### Policy Finetuning in Reinforcement Learning via Design of Experiments using Offline Data

**Authors:** Ruiqi Zhang, Andrea Zanette

**<details><summary>Abstract**</summary>

In some applications of reinforcement learning, a dataset of pre-collected experience is already availablebut it is also possible to acquire some additional online data to help improve the quality of the policy.However, it may be preferable to gather additional data with a single, non-reactive exploration policyand avoid the engineering costs associated with switching policies. In this paper we propose an algorithm with provable guarantees that can leverage an offline dataset to design a single non-reactive policy for exploration. We theoretically analyze the algorithm and measure the quality of the final policy as a function of the local coverage of the original dataset and the amount of additional data collected.</details>

### Post-processing Private Synthetic Data for Improving Utility on Selected Measures

**Authors:** Hao Wang, Shivchander Sudalairaj, John Henning, Kristjan Greenewald, Akash Srivastava

**<details><summary>Abstract**</summary>

Existing private synthetic data generation algorithms are agnostic to downstream tasks. However, end users may have specific requirements that the synthetic data must satisfy. Failure to meet these requirements could significantly reduce the utility of the data for downstream use. We introduce a post-processing technique that improves the utility of the synthetic data with respect to measures selected by the end user, while preserving strong privacy guarantees and dataset quality. Our technique involves resampling from the synthetic data to filter out samples that do not meet the selected utility measures, using an efficient stochastic first-order algorithm to find optimal resampling weights. Through comprehensive numerical experiments, we demonstrate that our approach consistently improves the utility of synthetic data across multiple benchmark datasets and state-of-the-art synthetic data generation algorithms.</details>

### Practical and Asymptotically Exact Conditional Sampling in Diffusion Models

**Authors:** Luhuan Wu, Brian Trippe, Christian Naesseth, John Cunningham, David Blei

**<details><summary>Abstract**</summary>

Diffusion models have been successful on a range of conditional generation tasks including molecular design and text-to-image generation. However, these achievements have primarily depended on task-specific conditional training or error-prone heuristic approximations. Ideally, a conditional generation method should provide exact samples for a broad range of conditional distributions without requiring taskspecific training. To this end, we introduce the Twisted Diffusion Sampler, or TDS. TDS is a sequential Monte Carlo (SMC) algorithm that targets the conditional distributions of diffusion models. The main idea is to use twisting, an SMC technique that enjoys good computational efficiency, to incorporate heuristic approximations without compromising asymptotic exactness. We first find in simulation and in conditional image generation tasks that TDS provides a computational statistical trade-off, yielding more accurate approximations with many particles but with empirical improvements over heuristics with as few as two particles. We then turn to motif-scaffolding, a core task in protein design, using a TDS extension to Riemannian diffusion models; on benchmark test cases, TDS allows flexible conditioning criteria and often outperforms the state-of-the-art, conditionally trained model.</details>

### Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting

**Authors:** Marcel Kollovieh, Abdul Fatir Ansari, Michael Bohlke-Schneider, Jasper Zschiegner, Hao Wang, Yuyang (Bernie) Wang

**<details><summary>Abstract**</summary>

Diffusion models have achieved state-of-the-art performance in generative modeling tasks across various domains. Prior works on time series diffusion models have primarily focused on developing conditional models tailored to specific forecasting or imputation tasks. In this work, we explore the potential of task-agnostic, unconditional diffusion models for several time series applications. We propose TSDiff, an unconditionally-trained diffusion model for time series. Our proposed self-guidance mechanism enables conditioning TSDiff for downstream tasks during inference, without requiring auxiliary networks or altering the training procedure. We demonstrate the effectiveness of our method on three different time series tasks: forecasting, refinement, and synthetic data generation. First, we show that TSDiff is competitive with several task-specific conditional forecasting methods (*predict*). Second, we leverage the learned implicit probability density of TSDiff to iteratively refine the predictions of base forecasters with reduced computational overhead over reverse diffusion (*refine*). Notably, the generative performance of the model remains intact — downstream forecasters trained on synthetic samples from TSDiff outperform forecasters that are trained on samples from other state-of-the-art generative time series models, occasionally even outperforming models trained on real data (*synthesize*).</details>

### Predicting Global Label Relationship Matrix for Graph Neural Networks under Heterophily

**Authors:** Langzhang Liang, Xiangjing Hu, Zenglin Xu, Zixing Song, Irwin King

**<details><summary>Abstract**</summary>

Graph Neural Networks (GNNs) have been shown to achieve remarkable performance on node classification tasks by exploiting both graph structures and node features. The majority of existing GNNs rely on the implicit homophily assumption. Recent studies have demonstrated that GNNs may struggle to model heterophilous graphs where nodes with different labels are more likely connected. To address this issue, we propose a generic GNN applicable to both homophilous and heterophilous graphs, namely Low-Rank Graph Neural Network (LRGNN). Our analysis demonstrates that a signed graph's global label relationship matrix has a low rank. This insight inspires us to predict the label relationship matrix by solving a robust low-rank matrix approximation problem, as prior research has proven that low-rank approximation could achieve perfect recovery under certain conditions. The experimental results reveal that the solution bears a strong resemblance to the label relationship matrix, presenting two advantages for graph modeling: a block diagonal structure and varying distributions of within-class and between-class entries.</details>

### [Spotlight] Prefix-Tree Decoding for Predicting Mass Spectra from Molecules

**Authors:** Samuel Goldman, John Bradshaw, Jiayi Xin, Connor Coley

**<details><summary>Abstract**</summary>

Computational predictions of mass spectra from molecules have enabled the discovery of clinically relevant metabolites. However, such predictive tools are still limited as they occupy one of two extremes, either operating  (a) by fragmenting molecules combinatorially with overly rigid constraints on potential rearrangements and poor time complexity or (b) by decoding lossy and nonphysical discretized spectra vectors. In this work, we use a new intermediate strategy for predicting mass spectra from molecules by treating mass spectra as sets of molecular formulae, which are themselves multisets of atoms. After first encoding an input molecular graph, we decode a set of molecular subformulae, each of which specify a predicted peak in the mass spectrum, the intensities of which are predicted by a second model. Our key insight is to overcome the combinatorial possibilities for molecular subformulae by decoding the formula set using a prefix tree structure, atom-type by atom-type, representing a general method for ordered multiset decoding. We show promising empirical results on mass spectra prediction tasks.</details>

### [Spotlight] Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision

**Authors:** Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, Chuang Gan

**<details><summary>Abstract**</summary>

Recent AI-assistant agents, such as ChatGPT, predominantly rely on supervised fine-tuning (SFT) with human annotations and reinforcement learning from human feedback (RLHF) to align the output of large language models (LLMs) with human intentions, ensuring they are helpful, ethical, and reliable. However, this dependence can significantly constrain the true potential of AI-assistant agents due to the high cost of obtaining human supervision and the related issues on quality, reliability, diversity, self-consistency, and undesirable biases. To address these challenges, we propose a novel approach called SELF-ALIGN, which combines principle-driven reasoning and the generative power of LLMs for the self-alignment of AI agents with minimal human supervision. Our approach encompasses four stages: first, we use an LLM to generate synthetic prompts, and a topic-guided method to augment the prompt diversity; second, we use a small set of human-written principles for AI models to follow, and guide the LLM through in-context learning from demonstrations (of principles application) to produce helpful, ethical, and reliable responses to user's queries; third, we fine-tune the original LLM with the high-quality self-aligned responses so that the resulting model can generate desirable responses for each query directly without the principle set and the demonstrations anymore; and finally, we offer a refinement step to address the issues of overly-brief or indirect responses. Applying SELF-ALIGN to the LLaMA-65b base language model, we develop an AI assistant named Dromedary. With fewer than 300 lines of human annotations (including < 200 seed prompts, 16 generic principles, and 5 exemplars for in-context learning). Dromedary significantly surpasses the performance of several state-of-the-art AI systems, including Text-Davinci-003 and Alpaca, on benchmark datasets with various settings.</details>

### Principled Weight Initialisation for Input-Convex Neural Networks

**Authors:** Pieter-Jan Hoedt, Günter Klambauer

**<details><summary>Abstract**</summary>

Input-Convex Neural Networks (ICNNs) are networks that guarantee convexity in their input-output mapping. These networks have been successfully applied for energy-based modelling, optimal transport problems and learning invariances.The convexity of ICNNs is achieved by using non-decreasing convex activation functions and non-negative weights. Because of these peculiarities, previous initialisation strategies, which implicitly assume centred weights, are not effective for ICNNs. By studying signal propagation through layers with non-negative weights, we are able to derive a principled weight initialisation for ICNNs. Concretely, we generalise signal propagation theory by removing the assumption that weights are sampled from a centred distribution. In a set of experiments, we demonstrate that our principled initialisation effectively accelerates learning in ICNNs and leads to better generalisation. Moreover, we find that, in contrast to common belief, ICNNs can be trained without skip-connections when initialised correctly. Finally, we apply ICNNs to a real-world drug discovery task and show that they allow for more effective molecular latent space exploration.</details>

### Prioritizing Samples in Reinforcement Learning with Reducible Loss

**Authors:** Shivakanth Sujit, Somjit Nath, Pedro Braga, Samira Ebrahimi Kahou

**<details><summary>Abstract**</summary>

Most reinforcement learning algorithms take advantage of an experience replay buffer to repeatedly train on samples the agent has observed in the past. Not all samples carry the same amount of significance and simply assigning equal importance to each of the samples is a naïve strategy. In this paper, we propose a method to prioritize samples based on how much we can learn from a sample. We define the learn-ability of a sample as the steady decrease of the training loss associated with this sample over time. We develop an algorithm to prioritize samples with high learn-ability, while assigning lower priority to those that are hard-to-learn, typically caused by noise or stochasticity. We empirically show that across multiple domains our method is more robust than random sampling and also better than just prioritizing with respect to the training loss, i.e. the temporal difference loss, which is used in prioritized experience replay.</details>

### Privacy Amplification via Compression: Achieving the Optimal Privacy-Accuracy-Communication Trade-off in Distributed Mean Estimation

**Authors:** Wei-Ning Chen, Dan Song, Ayfer Ozgur, Peter Kairouz

**<details><summary>Abstract**</summary>

Privacy and communication constraints are two major bottlenecks in federated learning (FL) and analytics (FA). We study the optimal accuracy of mean and frequency estimation (canonical models for FL and FA respectively) under joint communication and $(\varepsilon, \delta)$-differential privacy (DP) constraints. We consider both the central and the multi-message shuffled DP models. We show that in order to achieve the optimal $\ell_2$ error under $(\varepsilon, \delta)$-DP, it is sufficient for each client to send $\Theta\left( n \min\left(\varepsilon, \varepsilon^2\right)\right)$ bits for FL %{\color{blue}(assuming the dimension $d \gg n \min\left(\varepsilon, \varepsilon^2\right)$)} and $\Theta\left(\log\left( n\min\left(\varepsilon, \varepsilon^2\right) \right)\right)$ bits for FA to the server, where $n$ is the number of participating clients.  Without compression, each client needs $O(d)$ bits and $O\left(\log d\right)$ bits for the mean and frequency estimation problems respectively (where $d$ corresponds to the number of trainable parameters in FL or the domain size in FA), meaning that we can get significant savings in the regime $ n \min\left(\varepsilon, \varepsilon^2\right)  = o(d)$, which is often the relevant regime in practice. We propose two different ways to leverage compression for privacy amplification and achieve the optimal privacy-communication-accuracy trade-offs. In both cases, each client communicates only partial information about its sample and we show that privacy is amplified by randomly selecting the part contributed by each client. In the first method, the random selection is revealed to the server, which results in a central DP guarantee with optimal privacy-communication-accuracy trade-offs.  In the second method, the random data parts from the clients are  shuffled by a secure shuffler resulting in a multi-message shuffling scheme with the same optimal trade-offs. As a result, we establish the optimal three-way trade-offs between privacy, communication, and accuracy for both the central DP and multi-message shuffling frameworks.</details>

### [Spotlight] Private Distribution Learning with Public Data: The View from Sample Compression

**Authors:** Shai Ben-David, Alex Bie, Clément L Canonne, Gautam Kamath, Vikrant Singhal

**<details><summary>Abstract**</summary>

We study the problem of private distribution learning with access to public data. In this setup, which we refer to as *public-private learning*, the learner is given public and private samples drawn from an unknown distribution $p$ belonging to a class $\mathcal Q$, with the goal of outputting an estimate of $p$ while adhering to privacy constraints (here, pure differential privacy) only with respect to the private samples.     We show that the public-private learnability of a class $\mathcal Q$ is connected to the existence of a sample compression scheme for $\mathcal Q$, as well as to an intermediate notion we refer to as \emph{list learning}. Leveraging this connection: (1) approximately recovers previous results on Gaussians over $\mathbb R^d$; and (2) leads to new ones, including sample complexity upper bounds for arbitrary $k$-mixtures of Gaussians over $\mathbb R^d$, results for agnostic and distribution-shift resistant learners, as well as closure properties for public-private learnability under taking mixtures and products of distributions. Finally, via the connection to list learning, we show that for Gaussians in $\mathbb R^d$, at least $d$ public samples are necessary for private learnability, which is close to the known upper bound of $d+1$ public samples.</details>

### Private Federated Frequency Estimation: Adapting to the Hardness of the Instance

**Authors:** Jingfeng Wu, Wennan Zhu, Peter Kairouz, Vladimir Braverman

**<details><summary>Abstract**</summary>

In federated frequency estimation (FFE), multiple clients work together to estimate the frequency of their local data by communicating with a server, while maintaining the security constraint of $\mathtt{secsum}$ where the server can only access the sum of client-held vectors. For FFE with a single communication round, it is known that count sketch is nearly information-theoretically optimal [Chen et al., 2022]. However, when multiple communication rounds are allowed, we propose a new sketch algorithm that is provably more accurate than a naive adaptation of count sketch. Furthermore, we show that both our sketch algorithm and count sketch can achieve better accuracy when the problem instance is simpler. Therefore, we propose a two-phase approach to enable the use of a smaller sketch size for simpler problems. Finally, we provide mechanisms to make our proposed algorithm differentially private. We verify the performance of our methods through experiments conducted on real datasets.</details>

### Projection-Free Online Convex Optimization via Efficient Newton Iterations

**Authors:** Khashayar Gatmiry, Zak Mhammedi

**<details><summary>Abstract**</summary>

This paper presents new projection-free algorithms for Online Convex Optimization (OCO) over a convex domain $\mathcal{K} \subset \mathbb{R}^d$. Classical OCO algorithms (such as Online Gradient Descent) typically need to perform Euclidean projections onto the convex set $\mathcal{K}$ to ensure feasibility of their iterates. Alternative algorithms, such as those based on the Frank-Wolfe method, swap potentially-expensive Euclidean projections onto $\mathcal{K}$ for linear optimization over $\mathcal{K}$. However, such algorithms have a sub-optimal regret in OCO compared to projection-based algorithms. In this paper, we look at a third type of algorithms that output approximate Newton iterates using a self-concordant barrier for the set of interest. The use of a self-concordant barrier automatically ensures feasibility without the need of projections. However, the computation of the Newton iterates requires a matrix inverse, which can still be expensive. As our main contribution, we show how the stability of the Newton iterates can be leveraged to only compute the inverse Hessian a vanishing fractions of the rounds, leading to a new efficient projection-free OCO algorithm with a state-of-the-art regret bound.</details>

### ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation

**Authors:** Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan LI, Hang Su, Jun Zhu

**<details><summary>Abstract**</summary>

Score distillation sampling (SDS) has shown great promise in text-to-3D generation by distilling pretrained large-scale text-to-image diffusion models, but suffers from over-saturation, over-smoothing, and low-diversity problems. In this work, we propose to model the 3D parameter as a random variable instead of a constant as in SDS and present *variational score distillation* (VSD), a principled particle-based variational framework to explain and address the aforementioned issues in text-to-3D generation. We show that SDS is a special case of VSD and leads to poor samples with both small and large CFG weights. In comparison, VSD works well with various CFG weights as ancestral sampling from diffusion models and simultaneously improves the diversity and sample quality with a common CFG weight (i.e., 7.5). We further present various improvements in the design space for text-to-3D such as distillation time schedule and density initialization, which are orthogonal to the distillation algorithm yet not well explored. Our overall approach, dubbed *ProlificDreamer*, can generate high rendering resolution (i.e., 512$\times$512) and high-fidelity NeRF with rich structure and complex effects (e.g., smoke and drops). Further, initialized from NeRF, meshes fine-tuned by VSD are meticulously detailed and photo-realistic.</details>

### Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition

**Authors:** Shuhuai Ren, Aston Zhang, Yi Zhu, Shuai Zhang, Shuai Zheng, Mu Li, Alexander Smola, Xu Sun

**<details><summary>Abstract**</summary>

This work proposes POMP, a prompt pre-training method for vision-language models. Being memory and computation efficient, POMP enables the learned prompt to condense semantic information for a rich set of visual concepts with over twenty-thousand classes. Once pre-trained, the prompt with a strong transferable ability can be directly plugged into a variety of visual recognition tasks including image classification, semantic segmentation, and object detection, to boost recognition performances in a zero-shot manner. Empirical evaluation shows that POMP achieves state-of-the-art performances on 21 datasets, e.g., 67.0% average accuracy on 10 classification datasets (+3.1% compared to CoOp) and 84.4 hIoU on open-vocabulary Pascal VOC segmentation (+6.9 compared to ZSSeg).</details>

### PromptRestorer: A Prompting Image Restoration Method with Degradation Perception

**Authors:** Cong Wang, Jinshan Pan, Wei Wang, Jiangxin Dong, Mengzhu Wang, Yakun Ju, Junyang Chen, Xiao-Ming Wu

**<details><summary>Abstract**</summary>

We show that raw degradation features can effectively guide deep restoration models, providing accurate degradation priors to facilitate better restoration. While networks that do not consider them for restoration forget gradually degradation during the learning process, model capacity is severely hindered. To address this, we propose a Prompt}ing image Restorer, termed as PromptRestorer. Specifically, PromptRestorer contains two branches: a restoration branch and a prompting branch. The former is used to restore images, while the latter perceives degradation priors to prompt the restoration branch with reliable perceived content to guide the restoration process for better recovery. To better perceive the degradation which is extracted by a pre-trained model from given degradation observations, we propose a prompting degradation perception modulator, which adequately considers the characters of the self-attention mechanism and pixel-wise modulation, to better perceive the degradation priors from global and local perspectives. To control the propagation of the perceived content for the restoration branch, we propose gated degradation perception propagation, enabling the restoration branch to adaptively learn more useful features for better recovery. Extensive experimental results show that our PromptRestorer achieves state-of-the-art results on 4 image restoration tasks, including image deraining, deblurring, dehazing, and desnowing.</details>

### ProtoDiff: Learning to Learn Prototypical Networks by Task-Guided Diffusion

**Authors:** Yingjun Du, Zehao Xiao, Shengcai Liao, Cees Snoek

**<details><summary>Abstract**</summary>

Prototype-based meta-learning has emerged as a powerful technique for addressing few-shot learning challenges. However, estimating a deterministic prototype using a simple average function from a limited number of examples remains a fragile process. To overcome this limitation, we introduce ProtoDiff, a novel framework that leverages a task-guided diffusion model during the meta-training phase to gradually generate prototypes, thereby providing efficient class representations. Specifically,  a set of prototypes is optimized to achieve per-task prototype overfitting, enabling accurately obtaining the overfitted prototypes for individual tasks.Furthermore, we introduce a task-guided diffusion process within the prototype space, enabling the meta-learning of a generative process that transitions from a vanilla prototype to an overfitted prototype. ProtoDiff gradually generates task-specific prototypes from random noise during the meta-test stage, conditioned on the limited samples available for the new task. Furthermore, to expedite training and enhance ProtoDiff's performance, we propose the utilization of residual prototype learning, which leverages the sparsity of the residual prototype. We conduct thorough ablation studies to demonstrate its ability to accurately capture the underlying prototype distribution and enhance generalization. The new state-of-the-art performance on within-domain, cross-domain, and few-task few-shot classiﬁcation further substantiates the beneﬁt of ProtoDiff.</details>

### Prototypical Variational Autoencoder for 3D Few-shot Object Detection

**Authors:** Weiliang Tang, Biqi YANG, Xianzhi Li, Yun-Hui Liu, Pheng-Ann Heng, Chi-Wing Fu

**<details><summary>Abstract**</summary>

Few-Shot 3D Point Cloud Object Detection (FS3D) is a challenging task, aiming to detect 3D objects of novel classes using only limited annotated samples for training. Considering that the detection performance highly relies on the quality of the latent features, we design a VAE-based prototype learning scheme, named prototypical VAE (P-VAE), to learn a probabilistic latent space for enhancing the diversity and distinctiveness of the sampled features. The network encodes a multi-center GMM-like posterior, in which each distribution centers at a prototype. For regularization, P-VAE incorporates a reconstruction task to preserve geometric information. To adopt P-VAE for the detection framework, we formulate Geometric-informative Prototypical VAE (GP-VAE) to handle varying geometric components and Class-specific Prototypical VAE (CP-VAE) to handle varying object categories. In the first stage, we harness GP-VAE to aid feature extraction from the input scene. In the second stage, we cluster the geometric-informative features into per-instance features and use CP-VAE to refine each instance feature with category-level guidance. Experimental results show the top performance of our approach over the state of the arts on two FS3D benchmarks. Quantitative ablations and qualitative prototype analysis further demonstrate that our probabilistic modeling can significantly boost prototype learning for FS3D.</details>

### Provable Guarantees for Generative Behavior Cloning: Bridging Low-Level Stability and High-Level Behavior

**Authors:** Adam Block, Ali Jadbabaie, Daniel Pfrommer, Max Simchowitz, Russ Tedrake

**<details><summary>Abstract**</summary>

We propose a theoretical framework for studying behavior cloning of complex expert demonstrations using generative modeling.Our framework invokes low-level controllers - either learned or implicit in position-command control - to stabilize imitation around expert demonstrations. We show that with (a) a suitable low-level stability guarantee and (b) a powerful enough generative model as our imitation learner,  pure supervised behavior cloning can generate trajectories matching the per-time step distribution of essentially arbitrary expert trajectories in an optimal transport cost. Our analysis relies on a stochastic continuity property of the learned policy we call "total variation continuity" (TVC). We then show that TVC can be ensured with minimal degradation of accuracy by combining a popular data-augmentation regimen with a novel algorithmic trick: adding augmentation noise at execution time. We instantiate our guarantees for policies parameterized by diffusion models and prove that if the learner accurately estimates the score of the (noise-augmented) expert policy, then the distribution of imitator trajectories is close to the demonstrator distribution in a natural optimal transport distance. Our analysis constructs intricate couplings between noise-augmented trajectories, a technique that may be of independent interest. We conclude by empirically validating our algorithmic recommendations, and discussing implications for future research directions for better behavior cloning with generative modeling.</details>

### Provable convergence guarantees for black-box variational inference

**Authors:** Justin Domke, Robert Gower, Guillaume Garrigos

**<details><summary>Abstract**</summary>

Black-box variational inference is widely used in situations where there is no proof that its stochastic optimization succeeds. We suggest this is due to a theoretical gap in existing stochastic optimization proofs—namely the challenge of gradient estimators with unusual noise bounds, and a composite non-smooth objective. For dense Gaussian variational families, we observe that existing gradient estimators based on reparameterization satisfy a quadratic noise bound and give novel convergence guarantees for proximal and projected stochastic gradient descent using this bound. This provides rigorous guarantees that methods similar to those used in practice converge on realistic inference problems.</details>

### Provably Efficient Offline Goal-Conditioned Reinforcement Learning with General Function Approximation and Single-Policy Concentrability

**Authors:** Hanlin Zhu, Amy Zhang

**<details><summary>Abstract**</summary>

Goal-conditioned reinforcement learning (GCRL) refers to learning general-purpose skills that aim to reach diverse goals. In particular, offline GCRL only requires purely pre-collected datasets to perform training tasks without additional interactions with the environment. Although offline GCRL has become increasingly prevalent and many previous works have demonstrated its empirical success, the theoretical understanding of efficient offline GCRL algorithms is not well established, especially when the state space is huge and the offline dataset only covers the policy we aim to learn. In this paper, we provide a rigorous theoretical analysis of an existing empirically successful offline GCRL algorithm. We prove that under slight modification, this algorithm enjoys an $\tilde{O}(\text{poly}(1/\epsilon))$ sample complexity (where $\epsilon$ is the desired suboptimality of the learned policy) with general function approximation thanks to the property of (semi-)strong convexity of the objective functions. We only require nearly minimal assumptions on the dataset (single-policy concentrability) and the function class (realizability). Moreover, this algorithm consists of two uninterleaved optimization steps, which we refer to as $V$-learning and policy learning, and is computationally stable since it does not involve minimax optimization. We also empirically validate our theory by showing that the modified algorithm outperforms the previous algorithm in various real-world environments.To the best of our knowledge, this is the first algorithm that is both provably efficient with general function approximation and single-policy concentrability, and empirically successful without requiring solving minimax optimization problems.</details>

### Provably Fast Convergence of Independent Natural Policy Gradient for Markov Potential Games

**Authors:** Youbang Sun, Tao Liu, Ruida Zhou, P. R. Kumar, Shahin Shahrampour

**<details><summary>Abstract**</summary>

This work studies an independent natural policy gradient (NPG) algorithm for the multi-agent reinforcement learning problem in Markov potential games. It is shown that, under mild technical assumptions and the introduction of the \textit{suboptimality gap}, the independent NPG method with an oracle providing exact policy evaluation asymptotically reaches an $\epsilon$-Nash Equilibrium (NE) within $\mathcal{O}(1/\epsilon)$ iterations. This improves upon the previous best result of $\mathcal{O}(1/\epsilon^2)$ iterations and is of the same order, $\mathcal{O}(1/\epsilon)$, that is achievable for the single-agent case. Empirical results for a synthetic potential game and a congestion game are presented to verify the theoretical bounds.</details>

### Q-DM: An Efficient Low-bit Quantized Diffusion Model

**Authors:** Yanjing Li, Sheng Xu, Xianbin Cao, Xiao Sun, Baochang Zhang

**<details><summary>Abstract**</summary>

Denoising diffusion generative models are capable of generating high-quality data, but suffers from the computation-costly generation process, due to a iterative noise estimation using full-precision  networks. As an intuitive solution, quantization can significantly reduce the computational and memory consumption by low-bit parameters and operations. However, low-bit noise estimation networks in diffusion models (DMs) remain unexplored yet and perform much worse than the full-precision counterparts as observed in our experimental studies. In this paper, we first identify that the bottlenecks of low-bit quantized DMs come from a large distribution oscillation  on activations  and accumulated quantization error caused by the multi-step denoising process. To address these issues, we first develop a Timestep-aware Quantization (TaQ) method and a Noise-estimating Mimicking (NeM) scheme for low-bit quantized DMs (Q-DM) to effectively eliminate such oscillation and accumulated error respectively, leading to well-performed low-bit DMs. In this way, we propose an efficient Q-DM to calculate low-bit DMs by considering both training and inference process in the same framework. We evaluate our methods on popular DDPM and DDIM models. Extensive experimental results show that our method achieves a much better performance than the prior arts. For example, the 4-bit Q-DM theoretically accelerates the 1000-step DDPM by 7.8x and achieves a FID score of 5.17, on the unconditional CIFAR-10 dataset.</details>

### QH9: A Quantum Hamiltonian Prediction Benchmark for QM9 Molecules

**Authors:** Haiyang Yu, Meng Liu, Youzhi Luo, Alex Strasser, Xiaofeng Qian, Xiaoning Qian, Shuiwang Ji

**<details><summary>Abstract**</summary>

Supervised machine learning approaches have been increasingly used in accelerating electronic structure prediction as surrogates of first-principle computational methods, such as density functional theory (DFT). While numerous quantum chemistry datasets focus on chemical properties and atomic forces, the ability to achieve accurate and efficient prediction of the Hamiltonian matrix is highly desired, as it is the most important and fundamental physical quantity that determines the quantum states of physical systems and chemical properties. In this work, we generate a new Quantum Hamiltonian dataset, named as QH9, to provide precise Hamiltonian matrices for 2,399 molecular dynamics trajectories and 130,831 stable molecular geometries, based on the QM9 dataset. By designing benchmark tasks with various molecules, we show that current machine learning models have the capacity to predict Hamiltonian matrices for arbitrary molecules. Both the QH9 dataset and the baseline models are provided to the community through an open-source benchmark, which can be highly valuable for developing machine learning methods and accelerating molecular and materials design for scientific and technological applications. Our benchmark is publicly available at \url{https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench}.</details>

### RAPHAEL: Text-to-Image Generation via Large Mixture of Diffusion Paths

**Authors:** Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, Ping Luo

**<details><summary>Abstract**</summary>

Text-to-image generation has recently witnessed remarkable achievements. We introduce a text-conditional image diffusion model, termed RAPHAEL, to generate highly artistic images, which accurately portray the text prompts, encompassing multiple nouns, adjectives, and verbs. This is achieved by stacking tens of mixture-of-experts (MoEs) layers, i.e., space-MoE and time-MoE layers, enabling billions of diffusion paths (routes) from the network input to the output. Each path intuitively functions as a "painter" for depicting a particular textual concept onto a specified image region at a diffusion timestep. Comprehensive experiments reveal that RAPHAEL outperforms recent cutting-edge models, such as Stable Diffusion, ERNIE-ViLG 2.0, DeepFloyd, and DALL-E 2, in terms of both image quality and aesthetic appeal. Firstly, RAPHAEL exhibits superior performance in switching images across diverse styles, such as Japanese comics, realism, cyberpunk, and ink illustration. Secondly, a single model with three billion parameters, trained on 1,000 A100 GPUs for two months, achieves a state-of-the-art zero-shot FID score of 6.61 on the COCO dataset. Furthermore, RAPHAEL significantly surpasses its counterparts in human evaluation on the ViLG-300 benchmark. We believe that RAPHAEL holds the potential to propel the frontiers of image generation research in both academia and industry, paving the way for future breakthroughs in this rapidly evolving field. More details can be found on an anonymous webpage: https://raphaelpainting.github.io/.</details>

### RETVec: Resilient and Efficient Text Vectorizer

**Authors:** Elie Bursztein, Marina Zhang, Owen Vallis, XINYU JIA, Alexey Kurakin

**<details><summary>Abstract**</summary>

This paper describes RETVec, an efficient, resilient, and multilingual text vectorizer designed for neural-based text processing. RETVec combines a novel character encoding with an optional small embedding model to embed words into a 256-dimensional vector space. The RETVec embedding model is pre-trained using pair-wise metric learning to be robust against typos and character-level adversarial attacks. In this paper, we evaluate and compare RETVec to state-of-the-art vectorizers and word embeddings on popular model architectures and datasets. These comparisons demonstrate that RETVec leads to competitive, multilingual models that are significantly more resilient to typos and adversarial text attacks. RETVec is available under the Apache 2 license at https://github.com/google-research/retvec.</details>

### RGMIL: Guide Your Multiple-Instance Learning Model with Regressor

**Authors:** Zhaolong Du, Shasha Mao, Yimeng Zhang, Shuiping Gou, Licheng Jiao, Lin Xiong

**<details><summary>Abstract**</summary>

In video analysis, an important challenge is insufficient annotated data due to the rare occurrence of the critical patterns, and we need to provide discriminative frame-level representation with limited annotation in some applications. Multiple Instance Learning (MIL) is suitable for this scenario. However, many MIL models paid attention to analyzing the relationships between instance representations and aggregating them, but neglecting the critical information from the MIL problem itself, which causes difficultly achieving ideal instance-level performance compared with the supervised model.To address this issue, we propose the $	extbf{	extit{Regressor-Guided MIL network} (RGMIL)}$, which effectively produces discriminative instance-level representations in a general multi-classification scenario. In the proposed method, we make full use of the $	extit{regressor}$ through our newly introduced $	extit{aggregator}$, $	extbf{	extit{Regressor-Guided Pooling} (RGP)}$. RGP focuses on simulating the correct inference process of humans while facing similar problems without introducing new parameters, and the MIL problem can be accurately described through the critical information from the $	extit{regressor}$ in our method. In experiments, RGP shows dominance on more than 20 MIL benchmark datasets, with the average bag-level classification accuracy close to 1. We also perform a series of comprehensive experiments on the MMNIST dataset. Experimental results illustrate that our $	extit{aggregator}$ outperforms existing methods under different challenging circumstances. Instance-level predictions are even possible under the guidance of RGP information table in a long sequence. RGMIL also presents comparable instance-level performance with S-O-T-A supervised models in complicated applications. Statistical results demonstrate the assumption that a MIL model can compete with a supervised model at the instance level, as long as a structure that accurately describes the MIL problem is provided. The codes are available on $\url{https://github.com/LMBDA-design/RGMIL}$.</details>

### RH-BrainFS: Regional Heterogeneous Multimodal Brain Networks Fusion Strategy

**Authors:** Hongting Ye, Yalu Zheng, Yueying Li, Ke Zhang, Youyong Kong, Yonggui Yuan

**<details><summary>Abstract**</summary>

Multimodal fusion has become an important research technique in neuroscience that completes downstream tasks by extracting complementary information from multiple modalities. Existing multimodal research on brain networks mainly focuses on two modalities, structural connectivity (SC) and functional connectivity (FC). Recently, extensive literature has shown that the relationship between SC and FC is complex and not a simple one-to-one mapping. The coupling of structure and function at the regional level is heterogeneous. However, all previous studies have neglected the modal regional heterogeneity between SC and FC and fused their representations via "simple patterns", which are inefficient ways of multimodal fusion and affect the overall performance of the model. In this paper, to alleviate the issue of regional heterogeneity of multimodal brain networks, we propose a novel Regional Heterogeneous multimodal Brain networks Fusion Strategy (RH-BrainFS). Briefly, we introduce a brain subgraph networks module to extract regional characteristics of brain networks, and further use a new transformer-based fusion bottleneck module to alleviate the issue of regional heterogeneity between SC and FC. To the best of our knowledge, this is the first paper to explicitly state the issue of structural-functional modal regional heterogeneity and to propose asolution. Extensive experiments demonstrate that the proposed method outperforms several state-of-the-art methods in a variety of neuroscience tasks.</details>

### RS-Del: Edit Distance Robustness Certificates for Sequence Classifiers via Randomized Deletion

**Authors:** Zhuoqun Huang, Neil G Marchant, Keane Lucas, Lujo Bauer, Olga Ohrimenko, Benjamin Rubinstein

**<details><summary>Abstract**</summary>

Randomized smoothing is a leading approach for constructing classifiers that are certifiably robust against adversarial examples. Existing work on randomized smoothing has focused on classifiers with continuous inputs, such as images, where $\ell_p$-norm bounded adversaries are commonly studied. However, there has been limited work for classifiers with discrete or variable-size inputs, such as for source code, which require different threat models and smoothing mechanisms. In this work, we adapt randomized smoothing for discrete sequence classifiers to provide certified robustness against edit distance-bounded adversaries. Our proposed smoothing mechanism randomized deletion (RS-Del) applies random deletion edits, which are (perhaps surprisingly) sufficient to confer robustness against adversarial deletion, insertion and substitution edits. Our proof of certification deviates from the established Neyman-Pearson approach, which is intractable in our setting, and is instead organized around longest common subsequences. We present a case study on malware detection—a binary classification problem on byte sequences where classifier evasion is a well-established threat model. When applied to the popular MalConv malware detection model, our smoothing mechanism RS-Del achieves a certified accuracy of 91% at an edit distance radius of 128 bytes.</details>

### Rank-1 Matrix Completion with Gradient Descent and Small Random Initialization

**Authors:** Daesung Kim, Hye Won Chung

**<details><summary>Abstract**</summary>

The nonconvex formulation of the matrix completion problem has received significant attention in recent years due to its affordable complexity compared to the convex formulation. Gradient Descent (GD) is a simple yet efficient baseline algorithm for solving nonconvex optimization problems. The success of GD has been witnessed in many different problems in both theory and practice when it is combined with random initialization. However, previous works on matrix completion require either careful initialization or regularizers to prove the convergence of GD. In this paper, we study the rank-1 symmetric matrix completion and prove that GD converges to the ground truth when small random initialization is used. We show that in a logarithmic number of iterations, the trajectory enters the region where local convergence occurs. We provide an upper bound on the initialization size that is sufficient to guarantee the convergence, and show that a larger initialization can be used as more samples are available. We observe that the implicit regularization effect of GD plays a critical role in the analysis, and for the entire trajectory, it prevents each entry from becoming much larger than the others.</details>

### Reconstructing the Mind's Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors

**Authors:** Paul Scotti, Atmadeep Banerjee, Jimmie Goode, Stepan Shabalin, Alex Nguyen, ethan cohen, Aidan Dempster, Nathalie Verlinde, Elad Yundler, David Weisberg, Kenneth Norman, Tanishq Abraham

**<details><summary>Abstract**</summary>

We present MindEye, a novel fMRI-to-image approach to retrieve and reconstruct viewed images from brain activity. Our model comprises two parallel submodules that are specialized for retrieval (using contrastive learning) and reconstruction (using a diffusion prior). MindEye can map fMRI brain activity to any high dimensional multimodal latent space, like CLIP image space, enabling image reconstruction using generative models that accept embeddings from this latent space. We comprehensively compare our approach with other existing methods, using both qualitative side-by-side comparisons and quantitative evaluations, and show that MindEye achieves state-of-the-art performance in both reconstruction and retrieval tasks. In particular, MindEye can retrieve the exact original image even among highly similar candidates indicating that its brain embeddings retain fine-grained image-specific information. This allows us to accurately retrieve images even from large-scale databases like LAION-5B. We demonstrate through ablations that MindEye's performance improvements over previous methods result from specialized submodules for retrieval and reconstruction, improved training techniques, and training models with orders of magnitude more parameters. Furthermore, we show that MindEye can better preserve low-level image features in the reconstructions by using img2img, with outputs from a separate autoencoder. All code is available on GitHub.</details>

### Recurrent Hypernetworks are Surprisingly Strong in Meta-RL

**Authors:** Jacob Beck, Risto Vuorio, Zheng Xiong, Shimon Whiteson

**<details><summary>Abstract**</summary>

Deep reinforcement learning (RL) is notoriously impractical to deploy due to sample inefficiency. Meta-RL directly addresses this sample inefficiency by learning to perform few-shot learning when a distribution of related tasks is available for meta-training. While many specialized meta-RL methods have been proposed, recent work suggests that end-to-end learning in conjunction with an off-the-shelf sequential model, such as a recurrent network, is a surprisingly strong baseline. However, such claims have been controversial due to limited supporting evidence, particularly in the face of prior work establishing precisely the opposite. In this paper, we conduct an empirical investigation. While we likewise find that a recurrent network can achieve strong performance, we demonstrate that the use of hypernetworks is crucial to maximizing their potential. Surprisingly, when combined with hypernetworks, the recurrent baselines that are far simpler than existing specialized methods actually achieve the strongest performance of all methods evaluated.</details>

### Recursion in Recursion: Two-Level Nested Recursion for Length Generalization with Scalability

**Authors:** Jishnu Ray Chowdhury, Cornelia Caragea

**<details><summary>Abstract**</summary>

Binary Balanced Tree Recursive Neural Networks (BBT-RvNNs) enforce sequence composition according to a preset balanced binary tree structure. Thus, their non-linear recursion depth (which is the tree depth) is just $\log_2 n$ ($n$ being the sequence length). Such logarithmic scaling makes BBT-RvNNs efficient and scalable on long sequence tasks such as Long Range Arena (LRA). However, such computational efficiency comes at a cost because BBT-RvNNs cannot solve simple arithmetic tasks like ListOps. On the flip side, RvNN models (e.g., Beam Tree RvNN) that do succeed on ListOps (and other structure-sensitive tasks like formal logical inference) are generally several times more expensive (in time and space) than even Recurrent Neural Networks. In this paper, we introduce a novel framework --- Recursion in Recursion (RIR) to strike a balance between the two sides - getting some of the benefits from both worlds. In RIR, we use a form of two-level nested recursion - where the outer recursion is a $k$-ary balanced tree model with another recursive model (inner recursion) implementing its cell function. For the inner recursion, we choose Beam Tree RvNNs. To adjust Beam Tree RvNNs within RIR we also propose a novel strategy of beam alignment. Overall, this entails that the total recursive depth in RIR is upper-bounded by $k log_k n$. Our best RIR-based model is the first model that demonstrates high ($geq 90$) length-generalization performance on ListOps while at the same time being scalable enough to be trainable on long sequence inputs from LRA (it can reduce the memory usage of the original Beam Tree RvNN by hundreds of times). Moreover, in terms of accuracy in the LRA language tasks, it performs competitively with Structured State Space Models (SSMs) without any special initialization - outperforming Transformers by a large margin. On the other hand, while SSMs can marginally outperform RIR on LRA, they (SSMs) fail to length-generalize on ListOps. Our code is available at: https://github.com/JRC1995/BeamRecursionFamily/</details>

### Reduced Policy Optimization for Continuous Control with Hard Constraints

**Authors:** Shutong Ding, Jingya Wang, Yali Du, Ye Shi

**<details><summary>Abstract**</summary>

Recent advances in constrained reinforcement learning (RL) have endowed reinforcement learning with certain safety guarantees. However, deploying existing constrained RL algorithms in continuous control tasks with general hard constraints remains challenging, particularly in those situations with non-convex hard constraints. Inspired by the generalized reduced gradient (GRG) algorithm, a classical constrained optimization technique, we propose a reduced policy optimization (RPO) algorithm that combines RL with GRG to address general hard constraints. RPO partitions actions into basic actions and nonbasic actions following the GRG method and outputs the basic actions via a policy network. Subsequently, RPO calculates the nonbasic actions by solving equations based on equality constraints using the obtained basic actions. The policy network is then updated by implicitly differentiating nonbasic actions with respect to basic actions. Additionally, we introduce an action projection procedure based on the reduced gradient and apply a modified Lagrangian relaxation technique to ensure inequality constraints are satisfied. To the best of our knowledge, RPO is the first attempt that introduces GRG to RL as a way of efficiently handling both equality and inequality hard constraints. It is worth noting that there is currently a lack of RL environments with complex hard constraints, which motivates us to develop three new benchmarks: two robotics manipulation tasks and a smart grid operation control task. With these benchmarks, RPO achieves better performance than previous constrained RL algorithms in terms of both cumulative reward and constraint violation. We believe RPO, along with the new benchmarks, will open up new opportunities for applying RL to real-world problems with complex constraints.</details>

### Reducing Blackwell and Average Optimality to Discounted MDPs via the Blackwell Discount Factor

**Authors:** Julien Grand-Clément, Marek Petrik

**<details><summary>Abstract**</summary>

We introduce the Blackwell discount factor for Markov Decision Processes (MDPs). Classical objectives for MDPs include discounted, average, and Blackwell optimality. Many existing approaches to computing average-optimal policies solve for discount-optimal policies with a discount factor close to $1$, but they only work under strong or hard-to-verify assumptions on the MDP structure  such as unichain or ergodicity. We are the first to highlight the shortcomings of the classical definition of Blackwell optimality, which does not lead to simple algorithms for computing Blackwell-optimal policies and overlooks the pathological behaviors of optimal policies as regards the discount factors. To resolve this issue, in this paper, we show that when the discount factor is larger than  the Blackwell discount factor $\gamma_{\sf bw}$, all discount-optimal policies become Blackwell- and average-optimal, and we derive a general upper bound on $\gamma_{\sf bw}$. Our upper bound on $\gamma_{\sf bw}$, parametrized by the bit-size of the rewards and transition probabilities of the MDP instance, provides the first reduction from average and Blackwell optimality to discounted optimality, without any assumptions, along with new polynomial-time algorithms. Our work brings new ideas from polynomials and algebraic numbers to the analysis of MDPs. Our results also apply to robust MDPs, enabling the first algorithms to compute robust Blackwell-optimal policies.</details>

### Reducing Shape-Radiance Ambiguity in Radiance Fields with a Closed-Form Color Estimation Method

**Authors:** Qihang Fang, Yafei Song, Keqiang Li, Liefeng Bo

**<details><summary>Abstract**</summary>

Neural radiance field (NeRF) enables the synthesis of cutting-edge realistic novel view images of a 3D scene. It includes density and color fields to model the shape and radiance of a scene, respectively. Supervised by the photometric loss in an end-to-end training manner, NeRF inherently suffers from the shape-radiance ambiguity problem, i.e., it can perfectly fit training views but does not guarantee decoupling the two fields correctly. To deal with this issue, existing works have incorporated prior knowledge to provide an independent supervision signal for the density field, including total variation loss, sparsity loss, distortion loss, etc. These losses are based on general assumptions about the density field, e.g., it should be smooth, sparse, or compact, which are not adaptive to a specific scene. In this paper, we propose a more adaptive method to reduce the shape-radiance ambiguity. The key is a rendering method that is only based on the density field. Specifically, we first estimate the color field based on the density field and posed images in a closed form. Then NeRF's rendering process can proceed. We address the problems in estimating the color field, including occlusion and non-uniformly distributed views. Afterward, it is applied to regularize NeRF's density field. As our regularization is guided by photometric loss, it is more adaptive compared to existing ones. Experimental results show that our method improves the density field of NeRF both qualitatively and quantitatively. Our code is available at https://github.com/qihangGH/Closed-form-color-field.</details>

### RegBN: Batch Normalization of Multimodal Data with Regularization

**Authors:** Morteza Ghahremani Boozandani, Christian Wachinger

**<details><summary>Abstract**</summary>

Recent years have witnessed a surge of interest in integrating high-dimensional data captured by multisource sensors, driven by the impressive success of neural networks in integrating multimodal data. However, the integration of heterogeneous multimodal data poses a significant challenge, as confounding effects and dependencies among such heterogeneous data sources introduce unwanted variability and bias, leading to suboptimal performance of multimodal models. Therefore, it becomes crucial to normalize the low- or high-level features extracted from data modalities before their fusion takes place. This paper introduces RegBN, a novel approach for multimodal Batch Normalization with REGularization. RegBN uses the Frobenius norm as a regularizer term to address the side effects of confounders and underlying dependencies among different data sources. The proposed method generalizes well across multiple modalities and eliminates the need for learnable parameters, simplifying training and inference. We validate the effectiveness of RegBN on eight databases from five research areas, encompassing diverse modalities such as language, audio, image, video, depth, tabular, and 3D MRI. The proposed method demonstrates broad applicability across different architectures such as multilayer perceptrons, convolutional neural networks, and vision transformers, enabling effective normalization of both low- and high-level features in multimodal neural networks. RegBN is available at https://mogvision.github.io/RegBN.</details>

### Regression with Cost-based Rejection

**Authors:** Xin Cheng, Yuzhou Cao, Haobo Wang, Hongxin Wei, Bo An, Lei Feng

**<details><summary>Abstract**</summary>

Learning with rejection is an important framework that can refrain from making predictions to avoid critical mispredictions by balancing between prediction and rejection. Previous studies on cost-based rejection only focused on the classification setting, which cannot handle the continuous and infinite target space in the regression setting. In this paper, we investigate a novel regression problem called regression with cost-based rejection, where the model can reject to make predictions on some examples given certain rejection costs. To solve this problem, we first formulate the expected risk for this problem and then derive the Bayes optimal solution, which shows that the optimal model should reject to make predictions on the examples whose variance is larger than the rejection cost when the mean squared error is used as the evaluation metric. Furthermore, we propose to train the model by a surrogate loss function that considers rejection as binary classification and we provide conditions for the model consistency, which implies that the Bayes optimal solution can be recovered by our proposed surrogate loss. Extensive experiments demonstrate the effectiveness of our proposed method.</details>

### [Spotlight] Regret Matching+: (In)Stability and Fast Convergence in Games

**Authors:** Gabriele Farina, Julien Grand-Clément, Christian Kroer, Chung-Wei Lee, Haipeng Luo

**<details><summary>Abstract**</summary>

Regret Matching$^+$ (RM$^+$) and its variants are important algorithms for solving large-scale games.However, a theoretical understanding of their success in practice is still a mystery.Moreover, recent advances on fast convergence in games are limited to no-regret algorithms such as online mirror descent, which satisfy stability.In this paper, we first give counterexamples showing that RM+ and its predictive version can be unstable, which might cause other players to suffer large regret. We then provide two fixes: restarting and chopping off the positive orthant that RM$^+$ works in.We show that these fixes are sufficient to get $O(T^{1/4})$ individual regret and $O(1)$ social regret in normal-form games via RM$^+$ with predictions.We also apply our stabilizing techniques to clairvoyant updates in the uncoupled learning setting for RM$^+$ and prove desirable results akin to recent works for Clairvoyant online mirror descent. Our experiments show the advantages of our algorithms over vanilla RM$^+$-based algorithms in matrix and extensive-form games.</details>

### Regret-Optimal Model-Free Reinforcement Learning for Discounted MDPs with Short Burn-In Time

**Authors:** Xiang Ji, Gen Li

**<details><summary>Abstract**</summary>

A crucial problem in reinforcement learning is learning the optimal policy. We study this in tabular infinite-horizon discounted Markov decision processes under the online setting. The existing algorithms either fail to achieve regret optimality or have to incur a high memory and computational cost. In addition, existing optimal algorithms all require a long burn-in time in order to achieve optimal sample efficiency, i.e., their optimality is not guaranteed unless sample size surpasses a high threshold. We address both open problems by introducing a model-free algorithm that employs variance reduction and a novel technique that switches the execution policy in a slow-yet-adaptive manner. This is the first regret-optimal model-free algorithm in the discounted setting, with the additional benefit of a low burn-in time.</details>

### Reinforcement Learning with Simple Sequence Priors

**Authors:** Tankred Saanum, Noemi Elteto, Peter Dayan, Marcel Binz, Eric Schulz

**<details><summary>Abstract**</summary>

In reinforcement learning (RL), simplicity is typically quantified on an action-by-action basis -- but this timescale ignores temporal regularities, like repetitions, often present in sequential strategies. We therefore propose an RL algorithm that learns to solve tasks with sequences of actions that are compressible. We explore two possible sources of simple action sequences: Sequences that can be learned by autoregressive models, and sequences that are compressible with off-the-shelf data compression algorithms. Distilling these preferences into sequence priors, we derive a novel information-theoretic objective that incentivizes agents to learn policies that maximize rewards while conforming to these priors. We show that the resulting RL algorithm leads to faster learning, and attains higher returns than state-of-the-art model-free approaches in a series of continuous control tasks from the DeepMind Control Suite. These priors also produce a powerful information-regularized agent that is robust to noisy observations and can perform open-loop control.</details>

### Reining Generalization in Offline Reinforcement Learning via Representation Distinction

**Authors:** Yi Ma, Hongyao Tang, Dong Li, Zhaopeng Meng

**<details><summary>Abstract**</summary>

Offline Reinforcement Learning (RL) aims to address the challenge of distribution shift between the dataset and the learned policy, where the value of out-of-distribution (OOD) data may be erroneously estimated due to overgeneralization. It has been observed that a considerable portion of the benefits derived from the conservative terms designed by existing offline RL approaches originates from their impact on the learned representation. This observation prompts us to scrutinize the learning dynamics of offline RL, formalize the process of generalization, and delve into the prevalent overgeneralization issue in offline RL. We then investigate the potential to rein the generalization from the representation perspective to enhance offline RL. Finally, we present  Representation Distinction (RD), an innovative plug-in method for improving offline RL algorithm performance by explicitly differentiating between the representations of in-sample and OOD state-action pairs generated by the learning policy. Considering scenarios in which the learning policy mirrors the behavioral policy and similar samples may be erroneously distinguished, we suggest a dynamic adjustment mechanism for RD based on an OOD data generator to prevent data representation collapse and further enhance policy performance. We demonstrate the efficacy of our approach by applying RD to specially-designed backbone algorithms and widely-used offline RL algorithms. The proposed RD method significantly improves their performance across various continuous control tasks on D4RL datasets, surpassing several state-of-the-art offline RL algorithms.</details>

### Relative Entropic Optimal Transport: a (Prior-aware) Matching Perspective to (Unbalanced) Classification

**Authors:** Liangliang Shi, Haoyu Zhen, Gu Zhang, Junchi Yan

**<details><summary>Abstract**</summary>

Classification is a fundamental problem in machine learning, and considerable efforts have been recently devoted to the demanding long-tailed setting due to its prevalence in nature. Departure from the Bayesian framework, this paper rethinks classification from a matching perspective by studying the matching probability between samples and labels with optimal transport (OT) formulation. Specifically, we first propose a new variant of optimal transport, called Relative Entropic Optimal Transport (RE-OT), which guides the coupling solution to a known prior information matrix. We gives some theoretical results and their proof for RE-OT and surprisingly find RE-OT can help to deblur for barycenter images. Then we adopt inverse RE-OT for training long-tailed data and find that the loss derived from RE-OT has a similar form to Softmax-based cross-entropy loss, indicating a close connection between optimal transport and classification and the potential for transferring concepts between these two academic fields, such as barycentric projection in OT, which can map the labels back to the feature space. We further derive an epoch-varying RE-OT loss, and do the experiments on unbalanced image classification,  molecule classification, instance segmentation and representation learning. Experimental results show its effectiveness.</details>

### Representation Learning via Consistent Assignment of Views over Random Partitions

**Authors:** Thalles Santos Silva, Adín Ramírez Rivera

**<details><summary>Abstract**</summary>

We present Consistent Assignment of Views over Random Partitions (CARP), a self-supervised clustering method for representation learning of visual features. CARP learns prototypes in an end-to-end online fashion using gradient descent without additional non-differentiable modules to solve the cluster assignment problem. CARP optimizes a new pretext task based on random partitions of prototypes that regularizes the model and enforces consistency between views' assignments.  Additionally, our method improves training stability and prevents collapsed solutions in joint-embedding training. Through an extensive evaluation, we demonstrate that CARP's representations are suitable for learning downstream tasks. We evaluate CARP's representations capabilities in 17 datasets across many standard protocols, including linear evaluation, few-shot classification, $k$-NN, $k$-means, image retrieval, and copy detection. We compare CARP performance to 11 existing self-supervised methods. We extensively ablate our method and demonstrate that our proposed random partition pretext task improves the quality of the learned representations by devising multiple random classification tasks.In transfer learning tasks, CARP achieves the best performance on average against many SSL methods trained for a longer time.</details>

### Resilient Constrained Learning

**Authors:** Ignacio Hounie, Alejandro Ribeiro, Luiz F. O. Chamon

**<details><summary>Abstract**</summary>

When deploying machine learning solutions, they must satisfy multiple requirements beyond accuracy, such as fairness, robustness, or safety. These requirements are imposed during training either implicitly, using penalties, or explicitly, using constrained optimization methods based on Lagrangian duality. Either way, specifying requirements is hindered by the presence of compromises and limited prior knowledge about the data. Furthermore, their impact on performance can often only be evaluated by actually solving the learning problem. This paper presents a constrained learning approach that adapts the requirements while simultaneously solving the learning task. To do so, it relaxes the learning constraints in a way that contemplates how much they affect the task at hand by balancing the performance gains obtained from the relaxation against a user-defined cost of that relaxation. We call this approach resilient constrained learning after the term used to describe ecological systems that adapt to disruptions by modifying their operation. We show conditions under which this balance can be achieved and introduce a practical algorithm to compute it, for which we derive approximation and generalization guarantees. We showcase the advantages of this resilient learning method in image classification tasks involving multiple potential invariances and in federated learning under distribution shift.</details>

### Resilient Multiple Choice Learning: A learned scoring scheme with application to audio scene analysis

**Authors:** Victor Letzelter, Mathieu Fontaine, Mickael Chen, Patrick Pérez, Slim Essid, Gaël Richard

**<details><summary>Abstract**</summary>

We introduce Resilient Multiple Choice Learning (rMCL), an extension of the MCL approach for conditional distribution estimation in regression settings where multiple targets may be sampled for each training input.Multiple Choice Learning is a simple framework to tackle multimodal density estimation, using the Winner-Takes-All (WTA) loss for a set of hypotheses. In regression settings, the existing MCL variants focus on merging the hypotheses, thereby eventually sacrificing the diversity of the predictions. In contrast, our method relies on a novel learned scoring scheme underpinned by a mathematical framework based on Voronoi tessellations of the output space, from which we can derive a probabilistic interpretation.After empirically validating rMCL with experiments on synthetic data, we further assess its merits on the sound source localization problem, demonstrating its practical usefulness and the relevance of its interpretation.</details>

### Resolving the Tug-of-War: A Separation of Communication and Learning in Federated Learning

**Authors:** Junyi Li, Heng Huang

**<details><summary>Abstract**</summary>

Federated learning (FL) is a promising privacy-preserving machine learning paradigm over distributed data. In this paradigm, each client trains the parameter of a model locally and the server aggregates the parameter from clients periodically. Therefore, we perform the learning and communication over the same set of parameters. However, we find that learning and communication have fundamentally divergent requirements for parameter selection, akin to two opposite teams in a tug-of-war game. To mitigate this discrepancy, we introduce FedSep, a novel two-layer federated learning framework. FedSep consists of separated communication and learning layers for each client and the two layers are connected through decode/encode operations. In particular, the decoding operation is formulated as a minimization problem. We view FedSep as a federated bilevel optimization problem and propose an efficient algorithm to solve it. Theoretically, we demonstrate that its convergence matches that of the standard FL algorithms. The separation of communication and learning in FedSep offers innovative solutions to various challenging problems in FL, such as Communication-Efficient FL and Heterogeneous-Model FL. Empirical validation shows the superior performance of FedSep over various baselines in these tasks.</details>

### Response Length Perception and  Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline

**Authors:** Zangwei Zheng, Xiaozhe Ren, Fuzhao Xue, Yang Luo, Xin Jiang, Yang You

**<details><summary>Abstract**</summary>

Large language models (LLMs) have revolutionized the field of AI, demonstrating unprecedented capacity across various tasks. However, the inference process for LLMs comes with significant computational costs. In this paper, we propose an efficient LLM inference pipeline that harnesses the power of LLMs. Our approach begins by tapping into the potential of LLMs to accurately perceive and predict the response length with minimal overhead. By leveraging this information, we introduce an efficient sequence scheduling technique that groups queries with similar response lengths into micro-batches. We evaluate our approach on real-world instruction datasets using the LLaMA-based model, and our results demonstrate an impressive 86% improvement in inference throughput without compromising effectiveness. Notably, our method is orthogonal to other inference acceleration techniques, making it a valuable addition to many existing toolkits (e.g., FlashAttention, Quantization) for LLM inference.</details>

### [Oral] Rethinking Bias Mitigation: Fairer Architectures Make for Fairer Face Recognition

**Authors:** Samuel Dooley, Rhea Sukthanker, John Dickerson, Colin White, Frank Hutter, Micah Goldblum

**Oral Presentation:** Th, Dec 14, 08:15 -- Oral 5B

**<details><summary>Abstract**</summary>

Face recognition systems are widely deployed in safety-critical applications, including law enforcement, yet they exhibit bias across a range of socio-demographic dimensions, such as gender and race. Conventional wisdom dictates that model biases arise from biased training data. As a consequence, previous works on bias mitigation largely focused on pre-processing the training data, adding penalties to prevent bias from effecting the model during training, or post-processing predictions to debias them, yet these approaches have shown limited success on hard problems such as face recognition. In our work, we discover that biases are actually inherent to neural network architectures themselves. Following this reframing, we conduct the first neural architecture search for fairness, jointly with a search for hyperparameters. Our search outputs a suite of models which Pareto-dominate all other high-performance architectures and existing bias mitigation methods in terms of accuracy and fairness, often by large margins, on the two most widely used datasets for face identification, CelebA and VGGFace2. Furthermore, these models generalize to other datasets and sensitive attributes. We release our code, models and raw data files at https://github.com/dooleys/FR-NAS.</details>

### Rethinking Semi-Supervised Medical Image Segmentation: A Variance-Reduction Perspective

**Authors:** Chenyu You, Weicheng Dai, Yifei Min, Fenglin Liu, David Clifton, S. Kevin Zhou, Lawrence Staib, James Duncan

**<details><summary>Abstract**</summary>

For medical image segmentation, contrastive learning is the dominant practice to improve the quality of visual representations by contrasting semantically similar and dissimilar pairs of samples. This is enabled by the observation that without accessing ground truth labels, negative examples with truly dissimilar anatomical features, if sampled, can significantly improve the performance. In reality, however, these samples may come from similar anatomical features and the models may struggle to distinguish the minority tail-class samples, making the tail classes more prone to misclassification, both of which typically lead to model collapse. In this paper, we propose $\texttt{ARCO}$, a semi-supervised contrastive learning (CL) framework with stratified group theory for medical image segmentation. In particular, we first propose building $\texttt{ARCO}$ through the concept of variance-reduced estimation, and show that certain variance-reduction techniques are particularly beneficial in pixel/voxel-level segmentation tasks with extremely limited labels. Furthermore, we theoretically prove these sampling techniques are universal in variance reduction. Finally, we experimentally validate our approaches on eight benchmarks, i.e., five 2D/3D medical and three semantic segmentation datasets, with different label settings, and our methods consistently outperform state-of-the-art semi-supervised methods. Additionally, we augment the CL frameworks with these sampling techniques and demonstrate significant gains over previous methods. We believe our work is an important step towards semi-supervised medical image segmentation by quantifying the limitation of current self-supervision objectives for accomplishing such challenging safety-critical tasks.</details>

### Rethinking Tokenizer and Decoder in Masked Graph Modeling for Molecules

**Authors:** ZHIYUAN LIU, Yaorui Shi, An Zhang, Enzhi Zhang, Kenji Kawaguchi, Xiang Wang, Tat-Seng Chua

**<details><summary>Abstract**</summary>

Masked graph modeling excels in the self-supervised representation learning of molecular graphs. Scrutinizing previous studies, we can reveal a common scheme consisting of three key components: (1) graph tokenizer, which breaks a molecular graph into smaller fragments (\ie subgraphs) and converts them into tokens; (2) graph masking, which corrupts the graph with masks; (3) graph autoencoder, which first applies an encoder on the masked graph to generate the representations, and then employs a decoder on the representations to recover the tokens of the original graph. However, the previous MGM studies focus extensively on graph masking and encoder, while there is limited understanding of tokenizer and decoder. To bridge the gap, we first summarize popular molecule tokenizers at the granularity of node, edge, motif, and Graph Neural Networks (GNNs), and then examine their roles as the MGM's reconstruction targets. Further, we explore the potential of adopting an expressive decoder in MGM. Our results show that a subgraph-level tokenizer and a sufficiently expressive decoder with remask decoding have a \yuan{large impact on the encoder's representation learning}. Finally, we propose a novel MGM method SimSGT, featuring a Simple GNN-based Tokenizer (SGT) and an effective decoding strategy. We empirically validate that our method outperforms the existing molecule self-supervised learning methods. Our codes and checkpoints are available at https://github.com/syr-cn/SimSGT.</details>

### Revisiting Visual Model Robustness: A Frequency Long-Tailed Distribution View

**Authors:** Zhiyu Lin, Yifei Gao, Yunfan Yang, Jitao Sang

**<details><summary>Abstract**</summary>

A widely discussed hypothesis regarding the cause of visual models' lack of robustness is that they can exploit human-imperceptible high-frequency components (HFC) in images, which in turn leads to model vulnerabilities, such as the adversarial examples. However, (1) inconsistent findings regarding the validation of this hypothesis reflect in a limited understanding of HFC, and (2) solutions inspired by the hypothesis tend to involve a robustness-accuracy trade-off and leaning towards suppressing the model's learning on HFC. In this paper, inspired by the long-tailed characteristic observed in frequency spectrum, we first formally define the HFC from long-tailed perspective and then revisit the relationship between HFC and model robustness. In the frequency long-tailed scenario, experimental results on common datasets and various network structures consistently indicate that models in standard training exhibit high sensitivity to HFC. We investigate the reason of the sensitivity, which reflects in model's under-fitting behavior on HFC. Furthermore, the cause of the model's under-fitting behavior is attributed to the limited information content in HFC. Based on these findings, we propose a Balance Spectrum Sampling (BaSS) strategy, which effectively counteracts the long-tailed effect and enhances the model's learning on HFC. Extensive experimental results demonstrate that our method achieves a substantially better robustness-accuracy trade-off when combined with existing defense methods, while also indicating the potential of encouraging HFC learning in improving model performance.</details>

### Reward Finetuning for Faster and More Accurate Unsupervised Object Discovery

**Authors:** Katie Luo, Zhenzhen Liu, Xiangyu Chen, Yurong You, Sagie Benaim, Cheng Perng Phoo, Mark Campbell, Wen Sun, Bharath Hariharan, Kilian Weinberger

**<details><summary>Abstract**</summary>

Recent advances in machine learning have shown that Reinforcement Learning from Human Feedback (RLHF) can improve machine learning models and align them with human preferences. Although very successful for Large Language Models (LLMs), these advancements have not had a comparable impact in research for autonomous vehicles—where alignment with human expectations can be imperative. In this paper, we propose to adapt similar RL-based methods to unsupervised object discovery, i.e. learning to detect objects from LiDAR points without any training labels. Instead of labels, we use simple heuristics to mimic human feedback. More explicitly, we combine multiple heuristics into a simple reward function that positively correlates its score with bounding box accuracy, i.e., boxes containing objects are scored higher than those without. We start from the detector’s own predictions to explore the space and reinforce boxes with high rewards through gradient updates. Empirically, we demonstrate that our approach is not only more accurate, but also orders of magnitudes faster to train compared to prior works on object discovery. Code is available at https://github.com/katieluo88/DRIFT.</details>

### Riemannian stochastic optimization methods avoid strict saddle points

**Authors:** Ya-Ping Hsieh, Mohammad Reza Karimi Jaghargh, Andreas Krause, Panayotis Mertikopoulos

**<details><summary>Abstract**</summary>

Many modern machine learning applications - from online principal component analysis to covariance matrix identification and dictionary learning - can be formulated as minimization problems on Riemannian manifolds, typically solved with a Riemannian stochastic gradient method (or some variant thereof). However, in many cases of interest, the resulting minimization problem is _not_ geodesically convex, so the convergence of the chosen solver to a desirable solution - i.e., a local minimizer - is by no means guaranteed. In this paper, we study precisely this question, that is, whether stochastic Riemannian optimization algorithms are guaranteed to avoid saddle points with probability $1$. For generality, we study a family of retraction-based methods which, in addition to having a potentially much lower per-iteration cost relative to Riemannian gradient descent, include other widely used algorithms, such as natural policy gradient methods and mirror descent in ordinary convex spaces. In this general setting, we show that, under mild assumptions for the ambient manifold and the oracle providing gradient information, the policies under study avoid strict saddle points / submanifolds with probability $1$, from any initial condition. This result provides an important sanity check for the use of gradient methods on manifolds as it shows that, almost always, the end state of a stochastic Riemannian algorithm can only be a local minimizer.</details>

### Robust Matrix Sensing in the Semi-Random Model

**Authors:** Xing Gao, Yu Cheng

**<details><summary>Abstract**</summary>

Low-rank matrix recovery is a fundamental problem in machine learning with numerous applications. In practice, the problem can be solved by convex optimization namely nuclear norm minimization, or by non-convex optimization as it is well-known that for low-rank matrix problems like matrix sensing and matrix completion, all local optima of the natural non-convex objectives are also globally optimal under certain ideal assumptions.In this paper, we study new approaches for matrix sensing in a semi-random model where an adversary can add any number of arbitrary sensing matrices. More precisely, the problem is to recover a low-rank matrix $X^\star$ from linear measurements $b_i = \langle A_i, X^\star \rangle$, where an unknown subset of the sensing matrices satisfies the Restricted Isometry Property (RIP) and the rest of the $A_i$'s are chosen adversarially.It is known that in the semi-random model, existing non-convex objectives can have bad local optima. To fix this, we present a descent-style algorithm that provably recovers the ground-truth matrix $X^\star$. For the closely-related problem of semi-random matrix completion, prior work [CG18] showed that all bad local optima can be eliminated by reweighting the input data. However, the analogous approach for matrix sensing requires reweighting a set of matrices to satisfy RIP, which is a condition that is NP-hard to check. Instead, we build on the framework proposed in [KLL$^+$23] for semi-random sparse linear regression, where the algorithm in each iteration reweights the input based on the current solution, and then takes a weighted gradient step that is guaranteed to work well locally. Our analysis crucially exploits the connection between sparsity in vector problems and low-rankness in matrix problems, which may have other applications in obtaining robust algorithms for sparse and low-rank problems.</details>

### Robust Mean Estimation Without Moments for Symmetric Distributions

**Authors:** Gleb Novikov, David Steurer, Stefan Tiegel

**<details><summary>Abstract**</summary>

We study the problem of robustly estimating the mean or location parameter without moment assumptions.Known computationally efficient algorithms rely on strong distributional assumptions, such as sub-Gaussianity, or (certifiably) bounded moments.Moreover, the guarantees that they achieve in the heavy-tailed setting are weaker than those for sub-Gaussian distributions with known covariance.In this work, we show that such a tradeoff, between error guarantees and heavy-tails, is not necessary for symmetric distributions.We show that for a large class of symmetric distributions, the same error as in the Gaussian setting can be achieved efficiently.The distributions we study include products of arbitrary symmetric one-dimensional distributions, such as product Cauchy distributions, as well as elliptical distributions, a vast generalization of the Gaussian distribution.For product distributions and elliptical distributions with known scatter (covariance) matrix, we show that given an $\varepsilon$-corrupted sample, we can with probability at least $1-\delta$ estimate its location up to error $O(\varepsilon \sqrt{\log(1/\varepsilon)})$ using $\tfrac{d\log(d) + \log(1/\delta)}{\varepsilon^2 \log(1/\varepsilon)}$ samples.This result matches the best-known guarantees for the Gaussian distribution and known SQ lower bounds (up to the $\log(d)$ factor).For elliptical distributions with unknown scatter (covariance) matrix, we propose a sequence of efficient algorithms that approaches this optimal error.Specifically, for every $k \in \mathbb{N}$, we design an estimator using time and samples $\tilde{O}({d^k})$ achieving error $O(\varepsilon^{1-\frac{1}{2k}})$.This matches the error and running time guarantees when assuming certifiably bounded moments of order up to $k$.For unknown covariance, such error bounds of $o(\sqrt{\varepsilon})$ are not even known for (general) sub-Gaussian distributions.Our algorithms are based on a generalization of the well-known filtering technique [DK22].More specifically, we show how this machinery can be combined with Huber-loss-based techniques to work with projections of the noise that behave more nicely than the initial noise.Moreover, we show how sum-of-squares proofs can be used to obtain algorithmic guarantees even for distributions without a first moment.We believe that this approach may find other applications in future works.</details>

### Rubik's Cube: High-Order Channel Interactions with a Hierarchical Receptive Field

**Authors:** Naishan Zheng, man zhou, Chong Zhou, Chen Change Loy

**<details><summary>Abstract**</summary>

Image restoration techniques, spanning from the convolution to the transformer paradigm, have demonstrated robust spatial representation capabilities to deliver high-quality performance.Yet, many of these methods, such as convolution and the Feed Forward Network (FFN) structure of transformers, primarily leverage the basic first-order channel interactions and have not maximized the potential benefits of higher-order modeling. To address this limitation, our research dives into understanding relationships within the channel dimension and introduces a simple yet efficient, high-order channel-wise operator tailored for image restoration. Instead of merely mimicking high-order spatial interaction, our approach offers several added benefits: Efficiency: It adheres to the zero-FLOP and zero-parameter principle, using a spatial-shifting mechanism across channel-wise groups. Simplicity: It turns the favorable channel interaction and aggregation capabilities into element-wise multiplications and convolution units with $1 	imes 1$ kernel. Our new formulation expands the first-order channel-wise interactions seen in previous works to arbitrary high orders, generating a hierarchical receptive field akin to a Rubik's cube through the combined action of shifting and interactions. Furthermore, our proposed Rubik's cube convolution is a flexible operator that can be incorporated into existing image restoration networks, serving as a drop-in replacement for the standard convolution unit with fewer parameters overhead. We conducted experiments across various low-level vision tasks, including image denoising, low-light image enhancement, guided image super-resolution, and image de-blurring. The results consistently demonstrate that our Rubik's cube operator enhances performance across all tasks. Code is publicly available at https://github.com/zheng980629/RubikCube.</details>

### SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models

**Authors:** Shuchen Xue, Mingyang Yi, Weijian Luo, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, Zhi-Ming Ma

**<details><summary>Abstract**</summary>

Diffusion Probabilistic Models (DPMs) have achieved considerable success in generation tasks. As sampling from DPMs is equivalent to solving diffusion SDE or ODE which is time-consuming, numerous fast sampling methods built upon improved differential equation solvers are proposed. The majority of such techniques consider solving the diffusion ODE due to its superior efficiency. However, stochastic sampling could offer additional advantages in generating diverse and high-quality data. In this work, we engage in a comprehensive analysis of stochastic sampling from two aspects: variance-controlled diffusion SDE and linear multi-step SDE solver. Based on our analysis, we propose SA-Solver, which is an improved efficient stochastic Adams method for solving diffusion SDE to generate data with high quality. Our experiments show that SA-Solver achieves: 1) improved or comparable performance compared with the existing state-of-the-art (SOTA) sampling methods for few-step sampling; 2) SOTA FID on substantial benchmark datasets under a suitable number of function evaluations (NFEs).</details>

### SAME: Uncovering GNN Black Box with Structure-aware Shapley-based Multipiece Explanations

**Authors:** Ziyuan Ye, Rihan Huang, Qilin Wu, Quanying Liu

**<details><summary>Abstract**</summary>

Post-hoc explanation techniques on graph neural networks (GNNs) provide economical solutions for opening the black-box graph models without model retraining. Many GNN explanation variants have achieved state-of-the-art explaining results on a diverse set of benchmarks, while they rarely provide theoretical analysis for their inherent properties and explanatory capability. In this work, we propose $\underline{	ext{S}}$tructure-$\underline{	ext{A}}$ware Shapley-based $\underline{	ext{M}}$ultipiece $\underline{	ext{E}}$xplanation (SAME) method to address the structure-aware feature interactions challenges for GNNs explanation. Specifically, SAME leverages an expansion-based Monte Carlo tree search to explore the multi-grained structure-aware connected substructure. Afterward, the explanation results are encouraged to be informative of the graph properties by optimizing the combination of distinct single substructures. With the consideration of fair feature interactions in the process of investigating multiple connected important substructures, the explanation provided by SAME has the potential to be as explainable as the theoretically optimal explanation obtained by the Shapley value within polynomial time. Extensive experiments on real-world and synthetic benchmarks show that SAME improves the previous state-of-the-art fidelity performance by 12.9\% on BBBP, 7.01\% on MUTAG, 42.3\% on Graph-SST2, 38.9\% on Graph-SST5, 11.3\% on BA-2Motifs and 18.2\% on BA-Shapes under the same testing condition. Code is available at https://github.com/same2023neurips/same.</details>

### SOL: Sampling-based Optimal Linear bounding of arbitrary scalar functions

**Authors:** Yuriy Biktairov, Jyotirmoy Deshmukh

**<details><summary>Abstract**</summary>

Finding tight linear bounds for activation functions in neural networksis an essential part of several state of the art neural network robustness certification tools. An activation function is an arbitrary, nonlinear,scalar function $f: \mathbb{R}^d \rightarrow \mathbb{R}$. In the existing work on robustness certification, such bounds have been computed using human ingenuity for a handful of the most popular activation functions. While a number of heuristics have been proposed for bounding arbitrary functions,no analysis of the tightness optimality for general scalar functions has been offered yet, to the best of our knowledge. We fill this gap by formulating a concise optimality criterion for tightness of the approximation which allows us tobuild optimal bounds for any function convex in the region of interest $R$. Fora more general class of functions Lipshitz-continuous in $R$ we propose a sampling-based approach (SOL) which, given an instance of the bounding problem, efficiently computes the tightest linear bounds within a given $\varepsilon > 0$ threshold. We leverage an adaptive sampling technique to iteratively build a setof sample points suitable for representing the target activation function. While the theoretical worst case time complexity of our approach is$O(\varepsilon^{-2d})$,it typically only takes $O(\log^{\beta} \frac{1}{\varepsilon})$ time for some $\beta \ge 1$ and isthus sufficiently fast in practice. We provide empirical evidence of SOL's practicalityby incorporating it into a robustness certifier and observing that itproduces similar or higher certification rates while taking as low as quarter of the time compared to the other methods.</details>

### SPA: A Graph Spectral Alignment Perspective for Domain Adaptation

**Authors:** Zhiqing Xiao, Haobo Wang, Ying Jin, Lei Feng, Gang Chen, Fei Huang, Junbo Zhao

**<details><summary>Abstract**</summary>

Unsupervised domain adaptation (UDA) is a pivotal form in machine learning to extend the in-domain model to the distinctive target domains where the data distributions differ. Most prior works focus on capturing the inter-domain transferability but largely overlook rich intra-domain structures, which empirically results in even worse discriminability. In this work, we introduce a novel graph SPectral Alignment (SPA) framework to tackle the tradeoff. The core of our method is briefly condensed as follows: (i)-by casting the DA problem to graph primitives, SPA composes a coarse graph alignment mechanism with a novel spectral regularizer towards aligning the domain graphs in eigenspaces; (ii)-we further develop a fine-grained message propagation module --- upon a novel neighbor-aware self-training mechanism --- in order for enhanced discriminability in the target domain. On standardized benchmarks, the extensive experiments of SPA demonstrate that its performance has surpassed the existing cutting-edge DA methods. Coupled with dense model analysis, we conclude that our approach indeed possesses superior efficacy, robustness, discriminability, and transferability. Code and data are available at: https://github.com/CrownX/SPA.</details>

### SPQR: Controlling Q-ensemble Independence with Spiked Random Model for Reinforcement Learning

**Authors:** Dohyeok Lee, Seungyub Han, Taehyun Cho, Jungwoo Lee

**<details><summary>Abstract**</summary>

Alleviating overestimation bias is a critical challenge for deep reinforcement learning to achieve successful performance on more complex tasks or offline datasets containing out-of-distribution data. In order to overcome overestimation bias, ensemble methods for Q-learning have been investigated to exploit the diversity of multiple Q-functions. Since network initialization has been the predominant approach to promote diversity in Q-functions, heuristically designed diversity injection methods have been studied in the literature. However, previous studies have not attempted to approach guaranteed independence over an ensemble from a theoretical perspective. By introducing a novel regularization loss for Q-ensemble independence based on random matrix theory, we propose spiked Wishart Q-ensemble independence regularization (SPQR) for reinforcement learning. Specifically, we modify the intractable hypothesis testing criterion for the Q-ensemble independence into a tractable KL divergence between the spectral distribution of the Q-ensemble and the target Wigner's semicircle distribution. We implement SPQR in several online and offline ensemble Q-learning algorithms. In the experiments, SPQR outperforms the baseline algorithms in both online and offline RL benchmarks.</details>

### SPRING: Studying Papers and Reasoning to play Games

**Authors:** Yue Wu, So Yeon Min, Shrimai Prabhumoye, Yonatan Bisk, Russ Salakhutdinov, Amos Azaria, Tom Mitchell, Yuanzhi Li

**<details><summary>Abstract**</summary>

Open-world survival games pose significant challenges for AI algorithms due to their multi-tasking, deep exploration, and goal prioritization requirements. Despite reinforcement learning (RL) being popular for solving games, its high sample complexity limits its effectiveness in complex open-world games like Crafter or Minecraft. We propose a novel approach, SPRING, to read Crafter's original academic paper and use the knowledge learned to reason and play the game through a large language model (LLM).Prompted with the LaTeX source as game context and a description of the agent's current observation, our SPRING framework employs a directed acyclic graph (DAG) with game-related questions as nodes and dependencies as edges. We identify the optimal action to take in the environment by traversing the DAG and calculating LLM responses for each node in topological order, with the LLM's answer to final node directly translating to environment actions.In our experiments, we study the quality of in-context "reasoning" induced by different forms of prompts under the setting of the Crafter environment. Our experiments suggest that LLMs, when prompted with consistent chain-of-thought, have great potential in completing sophisticated high-level trajectories. Quantitatively, SPRING with GPT-4 outperforms all state-of-the-art RL baselines, trained for 1M steps, without any training. Finally, we show the potential of Crafter as a test bed for LLMs. Code at github.com/holmeswww/SPRING</details>

### SQ Lower Bounds for Learning Mixtures of Linear Classifiers

**Authors:** Ilias Diakonikolas, Daniel Kane, Yuxin Sun

**<details><summary>Abstract**</summary>

We study the problem of learning mixtures of linear classifiers under Gaussian covariates.Given sample access to a mixture of $r$ distributions on $\mathbb{R}^n$ of the form $(\mathbf{x},y_{\ell})$, $\ell \in [r]$,where $\mathbf{x}\sim\mathcal{N}(0,\mathbf{I}_n)$ and$y_\ell=\mathrm{sign}(\langle\mathbf{v}_{\ell},\mathbf{x}\rangle)$for an unknown unit vector $\mathbf{v}_{\ell}$,the goal is to learn the underlying distribution in total variation distance. Our main result is a Statistical Query (SQ) lower bound suggesting that known algorithms for this problem are essentially best possible,even for the special case of uniform mixtures.In particular, we show that the complexity of any SQ algorithm for the problem is $n^{\mathrm{poly}(1/\Delta) \log(r)}$,where $\Delta$ is a lower bound on the pairwise $\ell_2$-separation between the $\mathbf{v}_{\ell}$'s.The key technical ingredient underlying our result is a new construction of spherical designs on the unit sphere that may be of independent interest.</details>

### [Spotlight] Saddle-to-Saddle Dynamics in Diagonal Linear Networks

**Authors:** Scott Pesme, Nicolas Flammarion

**<details><summary>Abstract**</summary>

In this paper we fully describe the trajectory of gradient flow over $2$-layer diagonal linear networks for the regression setting in the limit of vanishing initialisation. We show that the limiting flow successively jumps from a saddle of the training loss to another until reaching the minimum $\ell_1$-norm solution. We explicitly characterise the visited saddles as well as the jump times through a recursive algorithm reminiscent of the LARS algorithm used for computing the Lasso path.  Starting from the zero vector, coordinates are successively activated until the minimum $\ell_1$-norm solution is recovered, revealing an incremental learning. Our proof leverages a convenient arc-length time-reparametrisation which enables to keep track of the transitions between the jumps. Our analysis requires negligible assumptions on the data, applies to both under and overparametrised settings and covers complex cases where there is no monotonicity of the number of active coordinates. We provide numerical experiments to support our findings.</details>

### Sample Complexity for Quadratic Bandits: Hessian Dependent Bounds and Optimal Algorithms

**Authors:** Qian Yu, Yining Wang, Baihe Huang, Qi Lei, Jason Lee

**<details><summary>Abstract**</summary>

In stochastic zeroth-order optimization, a problem of practical relevance is understanding how to fully exploit the local geometry of the underlying objective function. We consider a fundamental setting in which the objective function is quadratic, and provide the first tight characterization of the optimal Hessian-dependent sample complexity. Our contribution is twofold. First, from an information-theoretic point of view, we prove tight lower bounds on Hessian-dependent complexities by introducing a concept called \emph{energy allocation}, which captures the interaction between the searching algorithm and the geometry of objective functions. A matching upper bound is obtained by solving the optimal energy spectrum. Then, algorithmically, we show the existence of a Hessian-independent algorithm that universally achieves the asymptotic optimal sample complexities for all Hessian instances. The optimal sample complexities achieved by our algorithm remain valid for heavy-tailed noise distributions, which are enabled by a truncation method.</details>

### Sample based Explanations via Generalized Representers

**Authors:** Che-Ping Tsai, Chih-Kuan Yeh, Pradeep Ravikumar

**<details><summary>Abstract**</summary>

We propose a general class of sample based explanations of machine learning models, which we term generalized representers. To measure the effect of a training sample on a model's test prediction, generalized representers use two components: a global sample importance that quantifies the importance of the training point to the model and is invariant to test samples, and a local sample importance that measures similarity between the training sample and the test point with a kernel. A key contribution of the paper is to show that generalized representers are the only class of sample based explanations satisfying a natural set of axiomatic properties. We discuss approaches to extract global importances given a kernel, and also natural choices of kernels given modern non-linear models. As we show, many popular existing sample based explanations could be cast as generalized representers with particular choices of kernels and approaches to extract global importances. Additionally, we conduct empirical comparisons of different generalized representers on two image classification datasets.</details>

### Sample-Conditioned Hypothesis Stability Sharpens Information-Theoretic Generalization Bounds

**Authors:** Ziqiao Wang, Yongyi Mao

**<details><summary>Abstract**</summary>

We present new information-theoretic generalization guarantees through the a novel construction of the "neighboring-hypothesis" matrix and a new family of stability notions termed sample-conditioned hypothesis (SCH) stability.  Our approach yields sharper bounds that improve upon previous information-theoretic bounds in various learning scenarios. Notably, these bounds address the limitations of existing information-theoretic bounds in the context of stochastic convex optimization (SCO) problems, as explored in the recent work by Haghifam et al. (2023).</details>

### [Oral] Sampling from Gaussian Process Posteriors using Stochastic Gradient Descent

**Authors:** Jihao Andreas Lin, Javier Antorán, Shreyas Padhy, David Janz, José Miguel Hernández-Lobato, Alexander Terenin

**Oral Presentation:** Th, Dec 14, 08:00 -- Oral 5C

**<details><summary>Abstract**</summary>

Gaussian processes are a powerful framework for quantifying uncertainty and for sequential decision-making but are limited by the requirement of solving linear systems. In general, this has a cubic cost in dataset size and is sensitive to conditioning. We explore stochastic gradient algorithms as a computationally efficient method of approximately solving these linear systems: we develop low-variance optimization objectives for sampling from the posterior and extend these to inducing points. Counterintuitively, stochastic gradient descent often produces accurate predictions, even in cases where it does not converge quickly to the optimum. We explain this through a spectral characterization of the implicit bias from non-convergence. We show that stochastic gradient descent produces predictive distributions close to the true posterior both in regions with sufficient data coverage, and in regions sufficiently far away from the data. Experimentally, stochastic gradient descent achieves state-of-the-art performance on sufficiently large-scale or ill-conditioned regression tasks. Its uncertainty estimates match the performance of significantly more expensive baselines on a large-scale Bayesian~optimization~task.</details>

### Scaling Laws for Hyperparameter Optimization

**Authors:** Arlind Kadra, Maciej Janowski, Martin Wistuba, Josif Grabocka

**<details><summary>Abstract**</summary>

Hyperparameter optimization is an important subfield of machine learning that focuses on tuning the hyperparameters of a chosen algorithm to achieve peak performance. Recently, there has been a stream of methods that tackle the issue of hyperparameter optimization, however, most of the methods do not exploit the dominant power law nature of learning curves for Bayesian optimization. In this work, we propose Deep Power Laws (DPL), an ensemble of neural network models conditioned to yield predictions that follow a power-law scaling pattern. Our method dynamically decides which configurations to pause and train incrementally by making use of gray-box evaluations. We compare our method against 7 state-of-the-art competitors on 3 benchmarks related to tabular, image, and NLP datasets covering 59 diverse tasks. Our method achieves the best results across all benchmarks by obtaining the best any-time results compared to all competitors.</details>

### [Spotlight] Scaling Open-Vocabulary Object Detection

**Authors:** Matthias Minderer, Alexey Gritsenko, Neil Houlsby

**<details><summary>Abstract**</summary>

Open-vocabulary object detection has benefited greatly from pretrained vision-language models, but is still limited by the amount of available detection training data. While detection training data can be expanded by using Web image-text pairs as weak supervision, this has not been done at scales comparable to image-level pretraining. Here, we scale up detection data with self-training, which uses an existing detector to generate pseudo-box annotations on image-text pairs. Major challenges in scaling self-training are the choice of label space, pseudo-annotation filtering, and training efficiency. We present the OWLv2 model and OWL-ST self-training recipe, which address these challenges. OWLv2 surpasses the performance of previous state-of-the-art open-vocabulary detectors already at comparable training scales (~10M examples). However, with OWL-ST, we can scale to over 1B examples, yielding further large improvement: With an L/14 architecture, OWL-ST improves AP on LVIS rare classes, for which the model has seen no human box annotations, from 31.2% to 44.6% (43% relative improvement). OWL-ST unlocks Web-scale training for open-world localization, similar to what has been seen for image classification and language modelling. Code and checkpoints are available on GitHub.</details>

### Scan and Snap: Understanding Training Dynamics and Token Composition in 1-layer Transformer

**Authors:** Yuandong Tian, Yiping Wang, Beidi Chen, Simon Du

**<details><summary>Abstract**</summary>

Transformer architecture has shown impressive performance in multiple research domains and has become the backbone of many neural network models. However, there is limited understanding on how it works. In particular, with a simple predictive loss,  how the representation emerges from the gradient \emph{training dynamics} remains a mystery. In this paper, for 1-layer transformer with one self-attention layer plus one decoder layer, we analyze its SGD training dynamics for the task of next token prediction in a mathematically rigorous manner. We open the black box of the dynamic process of how the self-attention layer combines input tokens, and reveal the nature of underlying inductive bias. More specifically, with the assumption (a) no positional encoding, (b) long input sequence, and (c) the decoder layer learns faster than the self-attention layer, we prove that self-attention acts as a \emph{discriminative scanning algorithm}:  starting from uniform attention, it gradually attends more to distinct key tokens for a specific next token to be predicted, and pays less attention to common key tokens that occur across different next tokens. Among distinct tokens, it progressively drops attention weights, following the order of low to high co-occurrence between the key and the query token in the training set. Interestingly, this procedure does not lead to winner-takes-all, but stops due to a \emph{phase transition} that is controllable by the learning rate of the decoder layer, leaving (almost) fixed token combination. We verify this \textbf{\emph{scan and snap}} dynamics on synthetic and real-world data (WikiText-103).</details>

### Scattering Vision Transformer: Spectral Mixing Matters

**Authors:** Badri Patro, Vijay Agneeswaran

**<details><summary>Abstract**</summary>

Vision transformers have gained significant attention and achieved state-of-the-art performance in various computer vision tasks, including image classification, instance segmentation, and object detection. However, challenges remain in addressing attention complexity and effectively capturing fine-grained information within images. Existing solutions often resort to down-sampling operations, such as pooling, to reduce computational cost. Unfortunately, such operations are non-invertible and can result in information loss. In this paper, we present a novel approach called Scattering Vision Transformer (SVT) to tackle these challenges. SVT incorporates a spectrally scattering network that enables the capture of intricate image details. SVT overcomes the invertibility issue associated with down-sampling operations by separating low-frequency and high-frequency components. Furthermore, SVT introduces a unique spectral gating network utilizing Einstein multiplication for token and channel mixing, effectively reducing complexity. We show that SVT achieves state-of-the-art performance on the ImageNet dataset with a significant reduction in a number of parameters and FLOPS. SVT shows 2\% improvement over LiTv2 and iFormer. SVT-H-S reaches 84.2\% top-1 accuracy, while SVT-H-B reaches 85.2\% (state-of-art for base versions) and SVT-H-L reaches 85.7\% (again state-of-art for large versions). SVT also shows comparable results in other vision tasks such as instance segmentation. SVT also outperforms other transformers in transfer learning on standard datasets such as CIFAR10, CIFAR100, Oxford Flower, and Stanford Car datasets. The project page is available on this webpage.\url{https://badripatro.github.io/svt/}.</details>

### [Spotlight] Segment Any Point Cloud Sequences by Distilling Vision Foundation Models

**Authors:** Youquan Liu, Lingdong Kong, Jun CEN, Runnan Chen, Wenwei Zhang, Liang Pan, Kai Chen, Ziwei Liu

**<details><summary>Abstract**</summary>

Recent advancements in vision foundation models (VFMs) have opened up new possibilities for versatile and efficient visual perception. In this work, we introduce Seal, a novel framework that harnesses VFMs for segmenting diverse automotive point cloud sequences. Seal exhibits three appealing properties: i) Scalability: VFMs are directly distilled into point clouds, obviating the need for annotations in either 2D or 3D during pretraining. ii) Consistency: Spatial and temporal relationships are enforced at both the camera-to-LiDAR and point-to-segment regularization stages, facilitating cross-modal representation learning. iii) Generalizability: Seal enables knowledge transfer in an off-the-shelf manner to downstream tasks involving diverse point clouds, including those from real/synthetic, low/high-resolution, large/small-scale, and clean/corrupted datasets. Extensive experiments conducted on eleven different point cloud datasets showcase the effectiveness and superiority of Seal. Notably, Seal achieves a remarkable 45.0% mIoU on nuScenes after linear probing, surpassing random initialization by 36.9% mIoU and outperforming prior arts by 6.1% mIoU. Moreover, Seal demonstrates significant performance gains over existing methods across 20 different few-shot fine-tuning tasks on all eleven tested point cloud datasets. The code is available at this link.</details>

### [Spotlight] Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models

**Authors:** Alvin Heng, Harold Soh

**<details><summary>Abstract**</summary>

The recent proliferation of large-scale text-to-image models has led to growing concerns that such models may be misused to generate harmful, misleading, and inappropriate content. Motivated by this issue, we derive a technique inspired by continual learning to selectively forget concepts in pretrained deep generative models. Our method, dubbed Selective Amnesia, enables controllable forgetting  where a user can specify how a concept should be forgotten. Selective Amnesia can be applied to conditional variational likelihood models, which encompass a variety of popular deep generative frameworks, including variational autoencoders and large-scale text-to-image diffusion models. Experiments across different models demonstrate that our approach induces forgetting on a variety of concepts, from entire classes in standard datasets to celebrity and nudity prompts in text-to-image models.</details>

### Self-Predictive Universal AI

**Authors:** Elliot Catt, Jordi Grau-Moya, Marcus Hutter, Matthew Aitchison, Tim Genewein, Grégoire Delétang, Kevin Li, Joel Veness

**<details><summary>Abstract**</summary>

Reinforcement Learning (RL) algorithms typically utilize learning and/or planning techniques to derive effective policies. The integration of both approaches has proven to be highly successful in addressing complex sequential decision-making challenges, as evidenced by algorithms such as AlphaZero and MuZero, which consolidate the planning process into a parametric search-policy. AIXI, the most potent theoretical universal agent, leverages planning through comprehensive search as its primary means to find an optimal policy. Here we define an alternative universal agent, which we call Self-AIXI, that on the contrary to AIXI, maximally exploits learning to obtain good policies. It does so by self-predicting its own stream of action data, which is generated, similarly to other TD(0) agents, by taking an action maximization step over the current on-policy (universal mixture-policy) Q-value estimates. We prove that Self-AIXI converges to AIXI, and inherits a series of properties like maximal Legg-Hutter intelligence and the self-optimizing property.</details>

### Self-Supervised Motion Magnification by Backpropagating Through Optical Flow

**Authors:** Zhaoying Pan, Daniel Geng, Andrew Owens

**<details><summary>Abstract**</summary>

This paper presents a simple, self-supervised method for magnifying subtle motions in video: given an input video and a magnification factor, we manipulate the video such that its new optical flow is scaled by the desired amount. To train our model, we propose a loss function that estimates the optical flow of the generated video and penalizes how far if deviates from the given magnification factor. Thus, training involves differentiating through a pretrained optical flow network. Since our model is self-supervised, we can further improve its performance through test-time adaptation, by finetuning it on the input video. It can also be easily extended to magnify the motions of only user-selected objects. Our approach avoids the need for synthetic magnification datasets that have been used to train prior learning-based approaches. Instead, it leverages the existing capabilities of off-the-shelf motion estimators. We demonstrate the effectiveness of our method through evaluations of both visual quality and quantitative metrics on a range of real-world and synthetic videos, and we show our method works for both supervised and unsupervised optical flow methods.</details>

### Semi-Supervised Contrastive Learning for Deep Regression with Ordinal Rankings from Spectral Seriation

**Authors:** Weihang Dai, Yao DU, Hanru Bai, Kwang-Ting Cheng, Xiaomeng Li

**<details><summary>Abstract**</summary>

Contrastive learning methods can be applied to deep regression by enforcing label distance relationships in feature space. However, these methods are limited to labeled data only unlike for classification, where unlabeled data can be used for contrastive pretraining. In this work, we extend contrastive regression methods to allow unlabeled data to be used in a semi-supervised setting, thereby reducing the reliance on manual annotations. We observe that the feature similarity matrix between unlabeled samples still reflect inter-sample relationships, and that an accurate ordinal relationship can be recovered through spectral seriation algorithms if the level of error is within certain bounds. By using the recovered ordinal relationship for contrastive learning on unlabeled samples, we can allow more data to be used for feature representation learning, thereby achieve more robust results. The ordinal rankings can also be used to supervise predictions on unlabeled samples, which can serve as an additional training signal. We provide theoretical guarantees and empirical support through experiments on different datasets, demonstrating that our method can surpass existing state-of-the-art semi-supervised deep regression methods. To the best of our knowledge, this work is the first to explore using unlabeled data to perform contrastive learning for regression.</details>

### Sensitivity in Translation Averaging

**Authors:** Lalit Manam, Venu Madhav Govindu

**<details><summary>Abstract**</summary>

In 3D computer vision, translation averaging solves for absolute translations given a set of pairwise relative translation directions. While there has been much work on robustness to outliers and studies on the uniqueness of the solution, this paper deals with a distinctly different problem of sensitivity in translation averaging under uncertainty. We first analyze sensitivity in estimating scales corresponding to relative directions under small perturbations of the relative directions. Then, we formally define the conditioning of the translation averaging problem, which assesses the reliability of estimated translations based solely on the input directions. We give a sufficient criterion to ensure that the problem is well-conditioned. Subsequently, we provide an efficient algorithm to identify and remove combinations of directions which make the problem ill-conditioned while ensuring uniqueness of the solution. We demonstrate the utility of such analysis in global structure-from-motion pipelines for obtaining 3D reconstructions, which reveals the benefits of filtering the ill-conditioned set of directions in translation averaging in terms of reduced translation errors, a higher number of 3D points triangulated and faster convergence of bundle adjustment.</details>

### Shape Non-rigid Kinematics (SNK): A Zero-Shot Method for Non-Rigid Shape Matching via Unsupervised Functional Map Regularized Reconstruction

**Authors:** Souhaib Attaiki, Maks Ovsjanikov

**<details><summary>Abstract**</summary>

We present Shape Non-rigid Kinematics (SNK), a novel zero-shot method for non-rigid shape matching that eliminates the need for extensive training or ground truth data.SNK operates on a single pair of shapes, and employs a reconstruction-based strategy using an encoder-decoder architecture, which deforms the source shape to closely match the target shape. During the process, an unsupervised functional map is predicted and converted into a point-to-point map, serving as a supervisory mechanism for the reconstruction. To aid in training, we have designed a new decoder architecture that generates smooth, realistic deformations. SNK demonstrates competitive results on traditional benchmarks, simplifying the shape-matching process without compromising accuracy. Our code can be found online: https://github.com/pvnieo/SNK</details>

### Sharp Bounds for Generalized Causal Sensitivity Analysis

**Authors:** Dennis Frauen, Valentyn Melnychuk, Stefan Feuerriegel

**<details><summary>Abstract**</summary>

Causal inference from observational data is crucial for many disciplines such as medicine and economics. However, sharp bounds for causal effects under relaxations of the unconfoundedness assumption (causal sensitivity analysis) are subject to ongoing research. So far, works with sharp bounds are restricted to fairly simple settings (e.g., a single binary treatment). In this paper, we propose a unified framework for causal sensitivity analysis under unobserved confounding in various settings. For this, we propose a flexible generalization of the marginal sensitivity model (MSM) and then derive sharp bounds for a large class of causal effects. This includes (conditional) average treatment effects, effects for mediation analysis and path analysis, and distributional effects. Furthermore, our sensitivity model is applicable to discrete, continuous, and time-varying treatments. It allows us to interpret the partial identification problem under unobserved confounding as a distribution shift in the latent confounders while evaluating the causal effect of interest. In the special case of a single binary treatment, our bounds for (conditional) average treatment effects coincide with recent optimality results for causal sensitivity analysis. Finally, we propose a scalable algorithm to estimate our sharp bounds from observational data.</details>

### ShiftAddViT: Mixture of Multiplication Primitives Towards Efficient Vision Transformer

**Authors:** Haoran You, Huihong Shi, Yipin Guo, Celine Lin

**<details><summary>Abstract**</summary>

Vision Transformers (ViTs) have shown impressive performance and have become a unified backbone for multiple vision tasks. However, both the attention mechanism and multi-layer perceptrons (MLPs) in ViTs are not sufficiently efficient due to dense multiplications, leading to costly training and inference. To this end, we propose to reparameterize pre-trained ViTs with a mixture of multiplication primitives, e.g., bitwise shifts and additions, towards a new type of multiplication-reduced model, dubbed $	extbf{ShiftAddViT}$, which aims to achieve end-to-end inference speedups on GPUs without requiring training from scratch. Specifically, all $	exttt{MatMuls}$ among queries, keys, and values are reparameterized using additive kernels, after mapping queries and keys to binary codes in Hamming space. The remaining MLPs or linear layers are then reparameterized with shift kernels. We utilize TVM to implement and optimize those customized kernels for practical hardware deployment on GPUs. We find that such a reparameterization on (quadratic or linear) attention maintains model accuracy, while inevitably leading to accuracy drops when being applied to MLPs. To marry the best of both worlds, we further propose a new mixture of experts (MoE) framework to reparameterize MLPs by taking multiplication or its primitives as experts, e.g., multiplication and shift, and designing a new latency-aware load-balancing loss. Such a loss helps to train a generic router for assigning a dynamic amount of input tokens to different experts according to their latency. In principle, the faster the experts run, the more input tokens they are assigned. Extensive experiments on various 2D/3D Transformer-based vision tasks consistently validate the effectiveness of our proposed ShiftAddViT, achieving up to $	extbf{5.18$	imes$}$ latency reductions on GPUs and $	extbf{42.9}$% energy savings, while maintaining a comparable accuracy as original or efficient ViTs. Codes and models are available at https://github.com/GATECH-EIC/ShiftAddViT.</details>

### [Spotlight] SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling

**Authors:** Jiaxiang Dong, Haixu Wu, Haoran Zhang, Li Zhang, Jianmin Wang, Mingsheng Long

**<details><summary>Abstract**</summary>

Time series analysis is widely used in extensive areas. Recently, to reduce labeling expenses and benefit various tasks, self-supervised pre-training has attracted immense interest. One mainstream paradigm is masked modeling, which successfully pre-trains deep models by learning to reconstruct the masked content based on the unmasked part. However, since the semantic information of time series is mainly contained in temporal variations, the standard way of randomly masking a portion of time points will seriously ruin vital temporal variations of time series, making the reconstruction task too difficult to guide representation learning. We thus present SimMTM, a Simple pre-training framework for Masked Time-series Modeling. By relating masked modeling to manifold learning, SimMTM proposes to recover masked time points by the weighted aggregation of multiple neighbors outside the manifold, which eases the reconstruction task by assembling ruined but complementary temporal variations from multiple masked series. SimMTM further learns to uncover the local structure of the manifold, which is helpful for masked modeling. Experimentally, SimMTM achieves state-of-the-art fine-tuning performance compared to the most advanced time series pre-training methods in two canonical time series analysis tasks: forecasting and classification, covering both in- and cross-domain settings.</details>

### Similarity-based cooperative equilibrium

**Authors:** Caspar Oesterheld, Johannes Treutlein, Roger Grosse, Vincent Conitzer, Jakob Foerster

**<details><summary>Abstract**</summary>

As machine learning agents act more autonomously in the world, they will increasingly interact with each other. Unfortunately, in many social dilemmas like the one-shot Prisoner’s Dilemma, standard game theory predicts that ML agents will fail to cooperate with each other. Prior work has shown that one way to enable cooperative outcomes in the one-shot Prisoner’s Dilemma is to make the agents mutually transparent to each other, i.e., to allow them to access one another’s source code (Rubinstein, 1998; Tennenholtz, 2004) – or weights in the case of ML agents. However, full transparency is often unrealistic, whereas partial transparency is commonplace. Moreover, it is challenging for agents to learn their way to cooperation in the full transparency setting. In this paper, we introduce a more realistic setting in which agents only observe a single number indicating how similar they are to each other. We prove that this allows for the same set of cooperative outcomes as the full transparency setting. We also demonstrate experimentally that cooperation can be learned using simple ML methods.</details>

### Simple and Controllable Music Generation

**Authors:** Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, Gabriel Synnaeve, Yossi Adi, Alexandre Defossez

**<details><summary>Abstract**</summary>

We tackle the task of conditional music generation. We introduce MusicGen, a single Language Model (LM) that operates over several streams of compressed discrete music representation, i.e., tokens. Unlike prior work, MusicGen is comprised of a single-stage transformer LM together with efficient token interleaving patterns, which eliminates the need for cascading several models, e.g., hierarchically or upsampling. Following this approach, we demonstrate how MusicGen can generate high-quality samples, both mono and stereo, while being conditioned on textual description or melodic features, allowing better controls over the generated output. We conduct extensive empirical evaluation, considering both automatic and human studies, showing the proposed approach is superior to the evaluated baselines on a standard text-to-music benchmark. Through ablation studies, we shed light over the importance of each of the components comprising MusicGen. Music samples, code, and models are available at https://github.com/facebookresearch/audiocraft</details>

### Simplifying Neural Network Training Under Class Imbalance

**Authors:** Ravid Shwartz-Ziv, Micah Goldblum, Yucen Li, C. Bayan Bruss, Andrew Wilson

**<details><summary>Abstract**</summary>

Real-world datasets are often highly class-imbalanced, which can adversely impact the performance of deep learning models. The majority of research on training neural networks under class imbalance has focused on specialized loss functions and sampling techniques. Notably, we demonstrate that simply tuning existing components of standard deep learning pipelines, such as the batch size, data augmentation, architecture size, pre-training, optimizer, and label smoothing, can achieve state-of-the-art performance without any specialized loss functions or samplers. We also provide key prescriptions and considerations for training under class imbalance, and an understanding of why imbalance methods succeed or fail.</details>

### Simultaneous embedding of multiple attractor manifolds in a recurrent neural network using constrained gradient optimization

**Authors:** Haggai Agmon, Yoram Burak

**<details><summary>Abstract**</summary>

The storage of continuous variables in working memory is hypothesized to be sustained in the brain by the dynamics of recurrent neural networks (RNNs) whose steady states form continuous manifolds. In some cases, it is thought that the synaptic connectivity supports multiple attractor manifolds, each mapped to a different context or task. For example, in hippocampal area CA3, positions in distinct environments are represented by distinct sets of population activity patterns, each forming a continuum. It has been argued that the embedding of multiple continuous attractors in a single RNN inevitably causes detrimental interference: quenched noise in the synaptic connectivity disrupts the continuity of each attractor, replacing it by a discrete set of steady states that can be conceptualized as lying on local minima of an abstract energy landscape. Consequently, population activity patterns exhibit systematic drifts towards one of these discrete minima, thereby degrading the stored memory over time. Here we show that it is possible to dramatically attenuate these detrimental interference effects by adjusting the synaptic weights. Synaptic weight adjustment are derived from a loss function that quantifies the roughness of the energy landscape along each of the embedded attractor manifolds. By minimizing this loss function, the stability of states can be dramatically improved, without compromising the capacity.</details>

### Single-Pass Pivot Algorithm for Correlation Clustering. Keep it simple!

**Authors:** Konstantin Makarychev, Sayak Chakrabarty

**<details><summary>Abstract**</summary>

We show that a simple single-pass semi-streaming variant of the Pivot algorithm for Correlation Clustering gives a (3+eps)-approximation using O(n/eps) words of memory. This is a slight improvement over the recent results of Cambus, Kuhn, Lindy, Pai, and Uitto, who gave a (3+eps)-approximation using O(n log n) words of memory, and Behnezhad, Charikar, Ma, and Tan, who gave a 5-approximation using O(n) words of memory. One of the main contributions of our paper is that the algorithm and its analysis are simple and easy to understand.</details>

### Sketching Algorithms for Sparse Dictionary Learning: PTAS and Turnstile Streaming

**Authors:** Gregory Dexter, Petros Drineas, David Woodruff, Taisuke Yasuda

**<details><summary>Abstract**</summary>

Sketching algorithms have recently proven to be a powerful approach both for designing low-space streaming algorithms as well as fast polynomial time approximation schemes (PTAS). In this work, we develop new techniques to extend the applicability of sketching-based approaches to the sparse dictionary learning and the Euclidean $k$-means clustering problems. In particular, we initiate the study of the challenging setting where the dictionary/clustering assignment for each of the $n$ input points must be output, which has surprisingly received little attention in prior work. On the fast algorithms front, we obtain a new approach for designing PTAS's for the $k$-means clustering problem, which generalizes to the first PTAS for the sparse dictionary learning problem. On the streaming algorithms front, we obtain new upper bounds and lower bounds for dictionary learning and $k$-means clustering. In particular, given a design matrix $\mathbf A\in\mathbb R^{n\times d}$ in a turnstile stream, we show an $\tilde O(nr/\epsilon^2 + dk/\epsilon)$ space upper bound for $r$-sparse dictionary learning of size $k$, an $\tilde O(n/\epsilon^2 + dk/\epsilon)$ space upper bound for $k$-means clustering, as well as an $\tilde O(n)$ space upper bound for $k$-means clustering on random order row insertion streams with a natural "bounded sensitivity" assumption. On the lower bounds side, we obtain a general $\tilde\Omega(n/\epsilon + dk/\epsilon)$ lower bound for $k$-means clustering, as well as an $\tilde\Omega(n/\epsilon^2)$ lower bound for algorithms which can estimate the cost of a single fixed set of candidate centers.</details>

### Slot-guided Volumetric Object Radiance Fields

**Authors:** DI QI, Tong Yang, Xiangyu Zhang

**<details><summary>Abstract**</summary>

We present a novel framework for 3D object-centric representation learning. Our approach effectively decomposes complex scenes into individual objects from a single image in an unsupervised fashion. This method, called \underline{s}lot-guided \underline{V}olumetric \underline{O}bject \underline{R}adiance \underline{F}ields~(sVORF), composes volumetric object radiance fields with object slots as a guidance to implement unsupervised 3D scene decomposition. Specifically, sVORF obtains object slots from a single image via a transformer module, maps these slots to volumetric object radiance fields with a hypernetwork and composes object radiance fields with the guidance of object slots at a 3D location. Moreover, sVORF significantly reduces memory requirement due to small-sized pixel rendering during training. We demonstrate the effectiveness of our approach by showing top results in scene decomposition and generation tasks of complex synthetic datasets (e.g., Room-Diverse). Furthermore, we also confirm the potential of sVORF to segment objects in real-world scenes (e.g., the LLFF dataset).  We hope our approach can provide preliminary understanding of the physical world and help ease future research in 3D object-centric representation learning.</details>

### Social Motion Prediction with Cognitive Hierarchies

**Authors:** Wentao Zhu, Jason Qin, Yuke Lou, Hang Ye, Xiaoxuan Ma, Hai Ci, Yizhou Wang

**<details><summary>Abstract**</summary>

Humans exhibit a remarkable capacity for anticipating the actions of others and planning their own actions accordingly. In this study, we strive to replicate this ability by addressing the social motion prediction problem. We introduce a new benchmark, a novel formulation, and a cognition-inspired framework. We present Wusi, a 3D multi-person motion dataset under the context of team sports, which features intense and strategic human interactions and diverse pose distributions. By reformulating the problem from a multi-agent reinforcement learning perspective, we incorporate behavioral cloning and generative adversarial imitation learning to boost learning efficiency and generalization. Furthermore, we take into account the cognitive aspects of the human social action planning process and develop a cognitive hierarchy framework to predict strategic human social interactions. We conduct comprehensive experiments to validate the effectiveness of our proposed dataset and approach.</details>

### Soft-Unification in Deep Probabilistic Logic

**Authors:** Jaron Maene, Luc De Raedt

**<details><summary>Abstract**</summary>

A fundamental challenge in neuro-symbolic AI is to devise primitives that fuse the logical and neural concepts. The Neural Theorem Prover has proposed the notion of soft-unification to turn the symbolic comparison between terms (i.e. unification) into a comparison in embedding space. It has been shown that soft-unification is a powerful mechanism that can be used to learn logic rules in an end-to-end differentiable manner. We study soft-unification from a conceptual point and outline several desirable properties of this operation. These include non-redundancy in the proof, well-defined proof scores, and non-sparse gradients. Unfortunately, these properties are not satisfied by previous systems such as the Neural Theorem Prover. Therefore, we introduce a more principled framework called DeepSoftLog based on probabilistic rather than fuzzy semantics. Our experiments demonstrate that DeepSoftLog can outperform the state-of-the-art on neuro-symbolic benchmarks, highlighting the benefits of these properties.</details>

### Softmax Output Approximation for Activation Memory-Efficient Training of Attention-based Networks

**Authors:** Changhyeon Lee, Seulki Lee

**<details><summary>Abstract**</summary>

In this paper, we propose to approximate the softmax output, which is the key product of the attention mechanism, to reduce its activation memory usage when training attention-based networks (aka Transformers). During the forward pass of the network, the proposed softmax output approximation method stores only a small fraction of the entire softmax output required for back-propagation and evicts the rest of the softmax output from memory. Then, during the backward pass, the evicted softmax activation output is approximated to compose the gradient to perform back-propagation for model training. Considering most attention-based models heavily rely on the softmax-based attention module that usually takes one of the biggest portions of the network, approximating the softmax activation output can be a simple yet effective way to decrease the training memory requirement of many attention-based networks. The experiment with various attention-based models and relevant tasks, i.e., machine translation, text classification, and sentiment analysis, shows that it curtails the activation memory usage of the softmax-based attention module by up to 84% (6.2× less memory) in model training while achieving comparable or better performance, e.g., up to 5.4% higher classification accuracy.</details>

### SpatialRank: Urban Event Ranking with NDCG Optimization on Spatiotemporal Data

**Authors:** BANG AN, Xun Zhou, YONGJIAN ZHONG, Tianbao Yang

**<details><summary>Abstract**</summary>

The problem of urban event ranking aims at predicting the top-$k$ most risky locations of future events such as traffic accidents and crimes. This problem is of fundamental importance to public safety and urban administration especially when limited resources are available. The problem is, however, challenging due to complex and dynamic spatio-temporal correlations between locations, uneven distribution of urban events in space, and the difficulty to correctly rank nearby locations with similar features. Prior works on event forecasting mostly aim at accurately predicting the actual risk score or counts of events for all the locations. Rankings obtained as such usually have low quality due to prediction errors. Learning-to-rank methods directly optimize measures such as Normalized Discounted Cumulative Gain (NDCG), but cannot handle the spatiotemporal autocorrelation existing among locations. Due to the common assumption that items are independent. In this paper, we bridge the gap by proposing a novel spatial event ranking approach named SpatialRank. SpatialRank features  adaptive graph convolution layers that dynamically learn the spatiotemporal dependencies across locations from data. In addition, the model optimizes through surrogates a hybrid NDCG loss with a spatial component to better rank neighboring spatial locations. We design an importance-sampling with a spatial filtering algorithm to effectively evaluate the loss during training. Comprehensive experiments on three real-world datasets demonstrate that SpatialRank can effectively identify the top riskiest locations of crimes and traffic accidents and outperform state-of-art methods in terms of NDCG by up to 12.7%.</details>

### Spatially Resolved Gene Expression Prediction from Histology Images via Bi-modal Contrastive Learning

**Authors:** Ronald Xie, Kuan Pang, Sai Chung, Catia Perciani, Sonya MacParland, Bo Wang, Gary Bader

**<details><summary>Abstract**</summary>

Histology imaging is an important tool in medical diagnosis and research, enabling the examination of tissue structure and composition at the microscopic level. Understanding the underlying molecular mechanisms of tissue architecture is critical in uncovering disease mechanisms and developing effective treatments.Gene expression profiling provides insight into the molecular processes underlying tissue architecture, but the process can be time-consuming and expensive. We present BLEEP (Bi-modaL Embedding for Expression Prediction), a bi-modal embedding framework capable of generating spatially resolved gene expression profiles of whole-slide Hematoxylin and eosin (H&E) stained histology images. BLEEP uses contrastive learning to construct a low-dimensional joint embedding space from a reference dataset using paired image and expression profiles at micrometer resolution. With this approach, the gene expression of any query image patch can be imputed using the expression profiles from the reference dataset. We demonstrate BLEEP’s effectiveness in gene expression prediction by benchmarking its performance on a human liver tissue dataset captured using the 10x Visium platform, where it achieves significant improvements over existing methods. Our results demonstrate the potential of BLEEP to provide insights into the molecular mechanisms underlying tissue architecture, with important implications in diagnosis and research of various diseases. The proposed approach can significantly reduce the time and cost associated with gene expression profiling, opening up new avenues for high-throughput analysis of histology images for both research and clinical applications.</details>

### Spectral Evolution and Invariance in Linear-width Neural Networks

**Authors:** Zhichao Wang, Andrew Engel, Anand D Sarwate, Ioana Dumitriu, Tony Chiang

**<details><summary>Abstract**</summary>

We investigate the spectral properties of linear-width feed-forward neural networks, where the sample size is asymptotically proportional to network width. Empirically, we show that the spectra of weight in this high dimensional regime are invariant when trained by gradient descent for small constant learning rates; we provide a theoretical justification for this observation and prove the invariance of the bulk spectra for both conjugate and neural tangent kernels. We demonstrate similar characteristics when training with stochastic gradient descent with small learning rates. When the learning rate is large, we exhibit the emergence of an outlier whose corresponding eigenvector is aligned with the training data structure. We also show that after adaptive gradient training, where a lower test error and feature learning emerge, both weight and kernel matrices exhibit heavy tail behavior. Simple examples are provided to explain when heavy tails can have better generalizations. We exhibit different spectral properties such as invariant bulk, spike, and heavy-tailed distribution from a two-layer neural network using different training strategies, and then correlate them to the feature learning. Analogous phenomena also appear when we train conventional neural networks with real-world data. We conclude that monitoring the evolution of the spectra during training is an essential step toward understanding the training dynamics and feature learning.</details>

### Spontaneous symmetry breaking in generative diffusion models

**Authors:** Gabriel Raya, Luca Ambrogioni

**<details><summary>Abstract**</summary>

Generative diffusion models have recently emerged as a leading approach for generating high-dimensional data. In this paper, we show that the dynamics of these models exhibit a spontaneous symmetry breaking that divides the generative dynamics into two distinct phases: 1) A linear steady-state dynamics around a central fixed-point and 2) an attractor dynamics directed towards the data manifold. These two "phases'' are separated by the change in stability of the central fixed-point, with the resulting window of instability being responsible for the diversity of the generated samples. Using both theoretical and empirical evidence, we show that an accurate simulation of the early dynamics does not significantly contribute to the final generation, since early fluctuations are reverted to the central fixed point. To leverage this insight, we propose a Gaussian late initialization scheme, which significantly improves model performance, achieving up to 3x FID improvements on fast samplers, while also increasing sample diversity (e.g., racial composition of generated CelebA images). Our work offers a new way to understand the generative dynamics of diffusion models that has the potential to bring about higher performance and less biased fast-samplers.</details>

### Spuriosity Rankings: Sorting Data to Measure and Mitigate Biases

**Authors:** Mazda Moayeri, Wenxiao Wang, Sahil Singla, Soheil Feizi

**<details><summary>Abstract**</summary>

We present a simple but effective method to measure and mitigate model biases caused by reliance on spurious cues. Instead of requiring costly changes to one's data or model training, our method better utilizes the data one already has by sorting them. Specifically, we rank images within their classes based on spuriosity (the degree to which common spurious cues are present), proxied via deep neural features of an interpretable network. With spuriosity rankings, it is easy to identify minority subpopulations (i.e. low spuriosity images) and assess model bias as the gap in accuracy between high and low spuriosity images. One can even efficiently remove a model's bias at little cost to accuracy by finetuning its classification head on low spuriosity images, resulting in fairer treatment of samples regardless of spuriosity. We demonstrate our method on ImageNet, annotating $5000$ class-feature dependencies ($630$ of which we find to be spurious) and generating a dataset of $325k$ soft segmentations for these features along the way. Having computed spuriosity rankings via the identified spurious neural features, we assess biases for $89$ diverse models and find that class-wise biases are highly correlated across models. Our results suggest that model bias due to spurious feature reliance is influenced far more by what the model is trained on than how it is trained.</details>

### Stability and Generalization of the Decentralized Stochastic Gradient Descent Ascent Algorithm

**Authors:** Miaoxi Zhu, Li Shen, Bo Du, Dacheng Tao

**<details><summary>Abstract**</summary>

The growing size of available data has attracted increasing interest in solving minimax problems in a decentralized manner for various machine learning tasks. Previous theoretical research has primarily focused on the convergence rate and communication complexity of decentralized minimax algorithms, with little attention given to their generalization. In this paper, we investigate the primal-dual generalization bound of the decentralized stochastic gradient descent ascent (D-SGDA) algorithm using the approach of algorithmic stability under both convex-concave and nonconvex-nonconcave settings. Our theory refines the algorithmic stability in a decentralized manner and demonstrates that the decentralized structure does not destroy the stability and generalization of D-SGDA, implying that it can generalize as well as the vanilla SGDA in certain situations. Our results analyze the impact of different topologies on the generalization bound of the D-SGDA algorithm beyond trivial factors such as sample sizes, learning rates, and iterations. We also evaluate the optimization error and balance it with the generalization gap to obtain the optimal population risk of D-SGDA in the convex-concave setting. Additionally, we perform several numerical experiments which validate our theoretical findings.</details>

### State-Action Similarity-Based Representations for Off-Policy Evaluation

**Authors:** Brahma Pavse, Josiah Hanna

**<details><summary>Abstract**</summary>

In reinforcement learning, off-policy evaluation (OPE) is the problem of estimating the expected return of an evaluation policy given a fixed dataset that was collected by running one or more different policies. One of the more empirically successful algorithms for OPE has been the fitted q-evaluation (FQE) algorithm that uses temporal difference updates to learn an action-value function, which is then used to estimate the expected return of the evaluation policy. Typically, the original fixed dataset is fed directly into FQE to learn the action-value function of the evaluation policy. Instead, in this paper, we seek to enhance the data-efficiency of FQE by first transforming the fixed dataset using a learned encoder, and then feeding the transformed dataset into FQE. To learn such an encoder, we introduce an OPE-tailored state-action behavioral similarity metric, and use this metric and the fixed dataset to learn an encoder that models this metric. Theoretically, we show that this metric allows us to bound the error in the resulting OPE estimate. Empirically, we show that other state-action similarity metrics lead to representations that cannot represent the action-value function of the evaluation policy, and that our state-action representation method boosts the data-efficiency of FQE and lowers OPE error relative to other OPE-based representation learning methods on challenging OPE tasks. We also empirically show that the learned representations significantly mitigate divergence of FQE under varying distribution shifts. Our code is available here: https://github.com/Badger-RL/ROPE.</details>

### State-space models with layer-wise nonlinearity are universal approximators with exponential decaying memory

**Authors:** Shida Wang, Beichen Xue

**<details><summary>Abstract**</summary>

State-space models have gained popularity in sequence modelling due to their simple and efficient network structures. However, the absence of nonlinear activation along the temporal direction limits the model's capacity. In this paper, we prove that stacking state-space models with layer-wise nonlinear activation is sufficient to approximate any continuous sequence-to-sequence relationship. Our findings demonstrate that the addition of layer-wise nonlinear activation enhances the model's capacity to learn complex sequence patterns. Meanwhile, it can be seen both theoretically and empirically that the state-space models do not fundamentally resolve the issue of exponential decaying memory. Theoretical results are justified by numerical verifications.</details>

### Statistical Insights into HSIC in High Dimensions

**Authors:** Tao Zhang, Yaowu Zhang, Tingyou Zhou

**<details><summary>Abstract**</summary>

Measuring the nonlinear dependence between random vectors and testing for their statistical independence is a fundamental problem in statistics. One of the most popular dependence measures is the Hilbert-Schmidt independence criterion (HSIC), which has attracted increasing attention in recent years. However, most existing works have focused on either fixed or very high-dimensional covariates. In this work, we bridge the gap between these two scenarios and provide statistical insights into the performance of HSIC when the dimensions grow at different rates. We first show that, under the null hypothesis, the rescaled HSIC converges in distribution to a standard normal distribution. Then we provide a general condition for the HSIC based tests to have nontrivial power in high dimensions. By decomposing this condition, we illustrate how the ability of HSIC to measure nonlinear dependence changes with increasing dimensions. Moreover, we demonstrate that, depending on the sample size, the covariate dimensions and the dependence structures within covariates, the HSIC can capture different types of associations between random vectors. We also conduct extensive numerical studies to validate our theoretical results.</details>

### Statistical Knowledge Assessment for Large Language Models

**Authors:** Qingxiu Dong, Jingjing Xu, Lingpeng Kong, Zhifang Sui, Lei Li

**<details><summary>Abstract**</summary>

Given varying prompts regarding a factoid question, can a large language model (LLM) reliably generate factually correct answers? Existing LLMs may generate distinct responses for different prompts. In this paper, we study the problem of quantifying knowledge contained in an LLM regarding a given set of facts. We propose KaRR, a statistical approach to assess factual knowledge for LLMs. The main idea is to estimate the ratio of LLM generating text corresponding to the answer entity given diverse prompts of the subject and the querying relation, versus it generating by random chances. Our assessment suite contains a comprehensive set of 994,123 entities and 600 relations, with 1,395,905 text aliases. We use our method to evaluate 20 LLMs of various sizes, including LLaMA, Alpaca, OPT, etc. Experiments show that our results have a strong correlation (0.43 Kendall's $\tau$) with the results of human assessment on LLMs. Our results reveal that the knowledge in LLMs with the same backbone architecture adheres to the scaling law, while tuning on instruction-following data sometimes compromises the model's capability to generate factually correct text reliably.</details>

### Stochastic Collapse: How Gradient Noise Attracts SGD Dynamics Towards Simpler Subnetworks

**Authors:** Feng Chen, Daniel Kunin, Atsushi Yamamura, Surya Ganguli

**<details><summary>Abstract**</summary>

In this work, we reveal a strong implicit bias of stochastic gradient descent (SGD) that drives overly expressive networks to much simpler subnetworks, thereby dramatically reducing the number of independent parameters, and improving generalization. To reveal this bias, we identify _invariant sets_, or subsets of parameter space that remain unmodified by SGD. We focus on two classes of invariant sets that correspond to simpler (sparse or low-rank) subnetworks and commonly appear in modern architectures. Our analysis uncovers that SGD exhibits a property of _stochastic attractivity_ towards these simpler invariant sets. We establish a sufficient condition for stochastic attractivity based on a competition between the loss landscape's curvature around the invariant set and the noise introduced by stochastic gradients. Remarkably, we find that an increased level of noise strengthens attractivity, leading to the emergence of attractive invariant sets associated with saddle-points or local maxima of the train loss. We observe empirically the existence of attractive invariant sets in trained deep neural networks, implying that SGD dynamics often collapses to simple subnetworks with either vanishing or redundant neurons. We further demonstrate how this simplifying process of _stochastic collapse_ benefits generalization in a linear teacher-student framework. Finally, through this analysis, we mechanistically explain why early training with large learning rates for extended periods benefits subsequent generalization.</details>

### Strategyproof Voting under Correlated Beliefs

**Authors:** Daniel Halpern, Rachel Li, Ariel Procaccia

**<details><summary>Abstract**</summary>

In voting theory, when voters have ranked preferences over candidates, the celebrated Gibbard-Satterthwaite Theorem essentially rules out the existence of reasonable strategyproof methods for picking a winner. What if we weaken strategyproofness to only hold for Bayesian voters with beliefs over others' preferences? When voters believe other participants' rankings are drawn independently from a fixed distribution, the impossibility persists. However, it is quite reasonable for a voter to believe that other votes are correlated, either to each other or to their own ranking. We consider such beliefs induced by classic probabilistic models in social choice such as the Mallows, Placket-Luce, and Thurstone-Mosteller models. We single out the plurality rule (choosing the candidate ranked first most often) as a particularly promising choice as it is strategyproof for a large class of beliefs containing the specific ones we introduce. Further, we show that plurality is unique among positional scoring rules in having this property: no other scoring rule is strategyproof for beliefs induced by the Mallows model when there are a sufficient number of voters. Finally, we give examples of prominent non-scoring voting rules failing to be strategyproof on beliefs in this class, further bolstering the case for plurality.</details>

### Streaming Algorithms and Lower Bounds for Estimating Correlation Clustering Cost

**Authors:** Sepehr Assadi, Vihan Shah, Chen Wang

**<details><summary>Abstract**</summary>

Correlation clustering is a fundamental optimization problem at the intersection of machine learning and theoretical computer science. Motivated by applications to big data processing, recent years have witnessed a flurry of results on this problem in the streaming model. In this model, the algorithm needs to process the input $n$-vertex graph by making one or few passes over the stream of its edges and using a limited memory, much smaller than the input size. All previous work on streaming correlation clustering have focused on semi-streaming algorithms with $\Omega(n)$ memory, whereas in this work, we study streaming algorithms with much smaller memory requirement of only $\text{polylog}{(n)}$ bits. This stringent memory requirement is in the same spirit of classical streaming algorithms that instead of recovering a full solution to the problem---which can be prohibitively large with such small memory as is the case in our problem---, aimed to learn certain statistical properties of their inputs. In our case, this translates to determining the ``(correlation) clusterability'' of input graphs, or more precisely, estimating the cost of the optimal correlation clustering solution. As our main result, we present two novel algorithms that in only $\text{polylog}{(n)}$ space are able to estimate the optimal correlation clustering cost up to some constant multiplicative factor plus some extra additive error. One of the algorithms outputs a $3$-multiplicative approximation plus $o(n^2)$ additive approximation, and the other one improves the additive error further down at the cost of increasing the multiplicative factor to some large constant. We then present new lower bounds that justify this mix of both multiplicative and additive error approximation in our algorithms.</details>

### Structure of universal formulas

**Authors:** Dmitry Yarotsky

**<details><summary>Abstract**</summary>

By universal formulas we understand parameterized analytic expressions that have a fixed complexity, but nevertheless can approximate any continuous function on a compact set. There exist various examples of such formulas, including some in the form of neural networks. In this paper we analyze the essential structural elements of these highly expressive models. We introduce a hierarchy of expressiveness classes connecting the global approximability property to the weaker property of infinite VC dimension, and prove a series of classification results for several increasingly complex functional families. In particular, we introduce a general family of polynomially-exponentially-algebraic functions that, as we prove, is subject to polynomial constraints. As a consequence, we show that fixed-size neural networks with not more than one layer of neurons having transcendental activations (e.g., sine or standard sigmoid) cannot in general approximate functions on arbitrary finite sets. On the other hand, we give examples of functional families, including two-hidden-layer neural networks, that approximate functions on arbitrary finite sets, but fail to do that  on the whole domain of definition.</details>

### [Oral] Students Parrot Their Teachers: Membership Inference on Model Distillation

**Authors:** Matthew Jagielski, Milad Nasr, Katherine Lee, Christopher A. Choquette-Choo, Nicholas Carlini, Florian Tramer

**Oral Presentation:** Th, Dec 14, 08:00 -- Oral 5B

**<details><summary>Abstract**</summary>

Model distillation is frequently proposed as a technique to reduce the privacy leakage of machine learning. These empirical privacy defenses rely on the intuition that distilled ``student'' models protect the privacy of training data, as they only interact with this data indirectly through a ``teacher'' model. In this work, we design membership inference attacks to systematically study the privacy provided by knowledge distillation to both the teacher and student training sets. Our new attacks show that distillation alone provides only limited privacy across a number of domains. We explain the success of our attacks on distillation by showing that membership inference attacks on a private dataset can succeed even if the target model is never queried on any actual training points, but only on inputs whose predictions are highly influenced by training data. Finally, we show that our attacks are strongest when student and teacher sets are similar, or when the attacker can poison the teacher set.</details>

### StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models

**Authors:** Yinghao Aaron Li, Cong Han, Vinay Raghavan, Gavin Mischler, Nima Mesgarani

**<details><summary>Abstract**</summary>

In this paper, we present StyleTTS 2, a text-to-speech (TTS) model that leverages style diffusion and adversarial training with large speech language models (SLMs) to achieve human-level TTS synthesis. StyleTTS 2 differs from its predecessor by modeling styles as a latent random variable through diffusion models to generate the most suitable style for the text without requiring reference speech, achieving efficient latent diffusion while benefiting from the diverse speech synthesis offered by diffusion models. Furthermore, we employ large pre-trained SLMs, such as WavLM, as discriminators with our novel differentiable duration modeling for end-to-end training, resulting in improved speech naturalness. StyleTTS 2 surpasses human recordings on the single-speaker LJSpeech dataset and matches it on the multispeaker VCTK dataset as judged by native English speakers. Moreover, when trained on the LibriTTS dataset, our model outperforms previous publicly available models for zero-shot speaker adaptation. This work achieves the first human-level TTS on both single and multispeaker datasets, showcasing the potential of style diffusion and adversarial training with large SLMs. The audio demos and source code are available at https://styletts2.github.io/.</details>

### Successor-Predecessor Intrinsic Exploration

**Authors:** Changmin Yu, Neil Burgess, Maneesh Sahani, Samuel J Gershman

**<details><summary>Abstract**</summary>

Exploration is essential in reinforcement learning, particularly in environments where external rewards are sparse. Here we focus on exploration with intrinsic rewards, where the agent transiently augments the external rewards with self-generated intrinsic rewards. Although the study of intrinsic rewards has a long history, existing methods focus on composing the intrinsic reward based on measures of future prospects of states, ignoring the information contained in the retrospective structure of transition sequences. Here we argue that the agent can utilise retrospective information to generate explorative behaviour with structure-awareness, facilitating efficient exploration based on global instead of local information. We propose Successor-Predecessor Intrinsic Exploration (SPIE), an exploration algorithm based on a novel intrinsic reward combining prospective and retrospective information. We show that SPIE yields more efficient and ethologically plausible exploratory behaviour in environments with sparse rewards and bottleneck states  than competing methods. We also implement SPIE in deep reinforcement learning agents, and show that the resulting agent achieves stronger empirical performance than existing methods on sparse-reward Atari games.</details>

### Supported Value Regularization for Offline Reinforcement Learning

**Authors:** Yixiu Mao, Hongchang Zhang, Chen Chen, Yi Xu, Xiangyang Ji

**<details><summary>Abstract**</summary>

Offline reinforcement learning suffers from the extrapolation error and value overestimation caused by out-of-distribution (OOD) actions. To mitigate this issue, value regularization approaches aim to penalize the learned value functions to assign lower values to OOD actions. However, existing value regularization methods lack a proper distinction between the regularization effects on in-distribution (ID) and OOD actions, and fail to guarantee optimal convergence results of the policy. To this end, we propose Supported Value Regularization (SVR), which penalizes the Q-values for all OOD actions while maintaining standard Bellman updates for ID ones. Specifically, we utilize the bias of importance sampling to compute the summation of Q-values over the entire OOD region, which serves as the penalty for policy evaluation. This design automatically separates the regularization for ID and OOD actions without manually distinguishing between them. In tabular MDP, we show that the policy evaluation operator of SVR is a contraction, whose fixed point outputs unbiased Q-values for ID actions and underestimated Q-values for OOD actions. Furthermore, the policy iteration with SVR guarantees strict policy improvement until convergence to the optimal support-constrained policy in the dataset. Empirically, we validate the theoretical properties of SVR in a tabular maze environment and demonstrate its state-of-the-art performance on a range of continuous control tasks in the D4RL benchmark.</details>

### SutraNets: Sub-series Autoregressive Networks for Long-Sequence, Probabilistic Forecasting

**Authors:** Shane Bergsma, Tim Zeyl, Lei Guo

**<details><summary>Abstract**</summary>

We propose SutraNets, a novel method for neural probabilistic forecasting of long-sequence time series. SutraNets use an autoregressive generative model to factorize the likelihood of long sequences into products of conditional probabilities. When generating long sequences, most autoregressive approaches suffer from harmful error accumulation, as well as challenges in modeling long-distance dependencies. SutraNets treat long, univariate prediction as multivariate prediction over lower-frequency sub-series. Autoregression proceeds across time and across sub-series in order to ensure coherent multivariate (and, hence, high-frequency univariate) outputs. Since sub-series can be generated using fewer steps, SutraNets effectively reduce error accumulation and signal path distances. We find SutraNets to significantly improve forecasting accuracy over competitive alternatives on six real-world datasets, including when we vary the number of sub-series and scale up the depth and width of the underlying sequence models.</details>

### Swarm Reinforcement Learning for Adaptive Mesh Refinement

**Authors:** Niklas Freymuth, Philipp Dahlinger, Tobias Würth, Simon Reisch, Luise Kärger, Gerhard Neumann

**<details><summary>Abstract**</summary>

The Finite Element Method, an important technique in engineering, is aided by Adaptive Mesh Refinement (AMR), which dynamically refines mesh regions to allow for a favorable trade-off between computational speed and simulation accuracy. Classical methods for AMR depend on task-specific heuristics or expensive error estimators, hindering their use for complex simulations. Recent learned AMR methods tackle these problems, but so far scale only to simple toy examples. We formulate AMR as a novel Adaptive Swarm Markov Decision Process in which a mesh is modeled as a system of simple collaborating agents that may split into multiple new agents. This framework allows for a spatial reward formulation that simplifies the credit assignment problem, which we combine with Message Passing Networks to propagate information between neighboring mesh elements. We experimentally validate the effectiveness of our approach, Adaptive Swarm Mesh Refinement (ASMR), showing that it learns reliable, scalable, and efficient refinement strategies on a set of challenging problems. Our approach significantly speeds up computation, achieving up to 30-fold improvement compared to uniform refinements in complex simulations. Additionally, we outperform learned baselines and achieve a refinement quality that is on par with a traditional error-based AMR strategy without expensive oracle information about the error signal.</details>

### Switching Autoregressive Low-rank Tensor Models

**Authors:** Hyun Dong Lee, andrew warrington, Joshua Glaser, Scott Linderman

**<details><summary>Abstract**</summary>

An important problem in time-series analysis is modeling systems with time-varying dynamics.  Probabilistic models with joint continuous and discrete latent states offer interpretable, efficient, and experimentally useful descriptions of such data.  Commonly used models include autoregressive hidden Markov models (ARHMMs) and switching linear dynamical systems (SLDSs), each with its own advantages and disadvantages.  ARHMMs permit exact inference and easy parameter estimation, but are parameter intensive when modeling long dependencies, and hence are prone to overfitting.  In contrast, SLDSs can capture long-range dependencies in a parameter efficient way through Markovian latent dynamics, but present an intractable likelihood and a challenging parameter estimation task.  In this paper, we propose _switching autoregressive low-rank tensor_ SALT models, which retain the advantages of both approaches while ameliorating the weaknesses.  SALT parameterizes the tensor of an ARHMM with a low-rank factorization to control the number of parameters and allow longer range dependencies without overfitting.  We prove theoretical and discuss practical connections between SALT, linear dynamical systems, and SLDSs.  We empirically demonstrate quantitative advantages of SALT models on a range of simulated and real prediction tasks, including behavioral and neural datasets.  Furthermore, the learned low-rank tensor provides novel insights into temporal dependencies within each discrete state.</details>

### Symbolic Discovery of Optimization Algorithms

**Authors:** Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V Le

**<details><summary>Abstract**</summary>

We present a method to formulate algorithm discovery as program search, and apply it to discover optimization algorithms for deep neural network training. We leverage efficient search techniques to explore an infinite and sparse program space. To bridge the large generalization gap between proxy and target tasks, we also introduce program selection and simplification strategies.Our method discovers a simple and effective optimization algorithm, $\textbf{Lion}$ ($\textit{Evo$\textbf{L}$ved S$\textbf{i}$gn M$\textbf{o}$me$\textbf{n}$tum}$). It is more memory-efficient than Adam as it only keeps track of the momentum. Different from adaptive optimizers, its update has the same magnitude for each parameter calculated through the sign operation.We compare Lion with widely used optimizers, such as Adam and Adafactor, for training a variety of models on different tasks. On image classification, Lion boosts the accuracy of ViT by up to 2\% on ImageNet and saves up to 5x the pre-training compute on JFT. On vision-language contrastive learning, we achieve 88.3\% $\textit{zero-shot}$ and 91.1\% $\textit{fine-tuning}$ accuracy on ImageNet, surpassing the previous best results by 2\% and 0.1\%, respectively. On diffusion models, Lion outperforms Adam by achieving a better FID score and reducing the training compute by up to 2.3x. For autoregressive, masked language modeling, and fine-tuning, Lion exhibits a similar or better performance compared to Adam. Our analysis of Lion reveals that its performance gain grows with the training batch size. It also requires a smaller learning rate than Adam due to the larger norm of the update produced by the sign function. Additionally, we examine the limitations of Lion and identify scenarios where its improvements are small or not statistically significant.</details>

### Synthetic Combinations: A Causal Inference Framework for Combinatorial Interventions

**Authors:** Abhineet Agarwal, Anish Agarwal, Suhas Vijaykumar

**<details><summary>Abstract**</summary>

We consider a setting where there are $N$ heterogeneous units and $p$ interventions. Our goal is to learn unit-specific potential outcomes for any combination of these $p$ interventions, i.e., $N \times 2^p$ causal parameters. Choosing a combination of interventions is a problem that naturally arises in a variety of applications such as factorial design experiments and recommendation engines (e.g., showing a set of movies that maximizes engagement for a given user). Running $N \times 2^p$ experiments to estimate the various parameters is likely expensive and/or infeasible as $N$ and $p$ grow. Further, with observational data there is likely confounding, i.e., whether or not a unit is seen under a combination is correlated with its potential outcome under that combination. We study this problem under a novel model that imposes latent structure across both units and combinations of interventions. Specifically, we assume latent similarity in potential outcomes across units (i.e., the matrix of potential outcomes is approximately rank $r$) and regularity in how combinations of interventions interact (i.e., the coefficients in the Fourier expansion of the potential outcomes is approximately $s$ sparse). We establish identification for all $N \times 2^p$ parameters despite unobserved confounding. We propose an estimation procedure, Synthetic Combinations, and establish finite-sample consistency under precise conditions on the observation pattern. We show that Synthetic Combinations is able to consistently estimate unit-specific potential outcomes given a total of $\text{poly}(r) \times \left( N + s^2p\right)$ observations. In comparison, previous methods that do not exploit structure across both units and combinations have poorer sample complexity scaling as $\min(N \times s^2p, \ \ r \times (N + 2^p))$.</details>

### TWIGMA: A dataset of AI-Generated Images with Metadata From Twitter

**Authors:** Yiqun Chen, James Zou

**<details><summary>Abstract**</summary>

Recent progress in generative artificial intelligence (gen-AI) has enabled the generation of photo-realistic and artistically-inspiring photos at a single click, catering to millions of users online. To explore how people use gen-AI models such as DALLE and StableDiffusion, it is critical to understand the themes, contents, and variations present in the AI-generated photos. In this work, we introduce TWIGMA (TWItter Generative-ai images with MetadatA), a comprehensive dataset encompassing over 800,000 gen-AI images collected from Jan 2021 to March 2023 on Twitter, with associated metadata (e.g., tweet text, creation date, number of likes). Through a comparative analysis of TWIGMA with natural images and human artwork, we find that gen-AI images possess distinctive characteristics and exhibit, on average, lower variability when compared to their non-gen-AI counterparts. Additionally, we find that the similarity between a gen-AI image and natural images is inversely correlated with the number of likes. Finally, we observe a longitudinal shift in the themes of AI-generated images on Twitter, with users increasingly sharing artistically sophisticated content such as intricate human portraits, whereas their interest in simple subjects such as natural scenes and animals has decreased. Our analyses and findings underscore the significance of TWIGMA as a unique data resource for studying AI-generated images.</details>

### TabMT: Generating tabular data with masked transformers

**Authors:** Manbir Gulati, Paul Roysdon

**<details><summary>Abstract**</summary>

Autoregressive and Masked Transformers are incredibly effective as generative models and classifiers.    While these models are most prevalent in NLP, they also exhibit strong performance in other domains, such as vision.     This work contributes to the exploration of transformer-based models in synthetic data generation for diverse application domains.     In this paper, we present TabMT, a novel Masked Transformer design for generating synthetic tabular data.     TabMT effectively addresses the unique challenges posed by heterogeneous data fields and is natively able to handle missing data.     Our design leverages improved masking techniques to allow for generation and demonstrates state-of-the-art performance from extremely small to extremely large tabular datasets.     We evaluate TabMT for privacy-focused applications and find that it is able to generate high quality data with superior privacy tradeoffs.</details>

### Tackling Heavy-Tailed Rewards in Reinforcement Learning with Function Approximation: Minimax Optimal and Instance-Dependent Regret Bounds

**Authors:** Jiayi Huang, Han Zhong, Liwei Wang, Lin Yang

**<details><summary>Abstract**</summary>

While numerous works have focused on devising efficient algorithms for reinforcement learning (RL) with uniformly bounded rewards, it remains an open question whether sample or time-efficient algorithms for RL with large state-action space exist when the rewards are \emph{heavy-tailed}, i.e., with only finite $(1+\epsilon)$-th moments for some $\epsilon\in(0,1]$. In this work, we address the challenge of such rewards in RL with linear function approximation. We first design an algorithm, \textsc{Heavy-OFUL}, for heavy-tailed linear bandits, achieving an \emph{instance-dependent} $T$-round regret of $\tilde{O}\big(d T^{\frac{1-\epsilon}{2(1+\epsilon)}} \sqrt{\sum_{t=1}^T \nu_t^2} + d T^{\frac{1-\epsilon}{2(1+\epsilon)}}\big)$, the \emph{first} of this kind. Here, $d$ is the feature dimension, and $\nu_t^{1+\epsilon}$ is the $(1+\epsilon)$-th central moment of the reward at the $t$-th round. We further show the above bound is minimax optimal when applied to the worst-case instances in stochastic and deterministic linear bandits. We then extend this algorithm to the RL settings with linear function approximation. Our algorithm, termed as \textsc{Heavy-LSVI-UCB}, achieves the \emph{first} computationally efficient \emph{instance-dependent} $K$-episode regret of $\tilde{O}(d \sqrt{H \mathcal{U}^*} K^\frac{1}{1+\epsilon} + d \sqrt{H \mathcal{V}^* K})$. Here, $H$ is length of the episode, and $\mathcal{U}^*, \mathcal{V}^*$ are instance-dependent quantities scaling with the central moment of reward and value functions, respectively. We also provide a matching minimax lower bound $\Omega(d H K^{\frac{1}{1+\epsilon}} + d \sqrt{H^3 K})$ to demonstrate the optimality of our algorithm in the worst case. Our result is achieved via a novel robust self-normalized concentration inequality that may be of independent interest in handling heavy-tailed noise in general online regression problems.</details>

### Taylor TD-learning

**Authors:** Michele Garibbo, Maxime Robeyns, Laurence Aitchison

**<details><summary>Abstract**</summary>

Many reinforcement learning approaches rely on temporal-difference (TD) learning to learn a critic.However, TD-learning updates can be high variance.Here, we introduce a model-based RL framework, Taylor TD, which reduces this variance in continuous state-action settings. Taylor TD uses a first-order Taylor series expansion of TD updates.This expansion allows Taylor TD to analytically integrate over stochasticity in the action-choice, and some stochasticity in the state distribution for the initial state and action of each TD update.We include theoretical and empirical evidence that Taylor TD updates are indeed lower variance than standard TD updates. Additionally, we show Taylor TD has the same stable learning guarantees as standard TD-learning with linear function approximation under a reasonable assumption.Next, we combine Taylor TD with the TD3 algorithm, forming TaTD3.We show TaTD3 performs as well, if not better, than several state-of-the art model-free and model-based baseline algorithms on a set of standard benchmark tasks.</details>

### Team-PSRO for Learning Approximate TMECor in Large Team Games via Cooperative Reinforcement Learning

**Authors:** Stephen McAleer, Gabriele Farina, Gaoyue Zhou, Mingzhi Wang, Yaodong Yang, Tuomas Sandholm

**<details><summary>Abstract**</summary>

Recent algorithms have achieved superhuman performance at a number of two-player zero-sum games such as poker and go. However, many real-world situations are multi-player games. Zero-sum two-team games, such as bridge and football, involve two teams where each member of the team shares the same reward with every other member of that team, and each team has the negative of the reward of the other team. A popular solution concept in this setting, called TMECor, assumes that teams can jointly correlate their strategies before play, but are not able to communicate during play. This setting is harder than two-player zero-sum games because each player on a team has different information and must use their public actions to signal to other members of the team. Prior works either have game-theoretic guarantees but only work in very small games, or are able to scale to large games but do not have game-theoretic guarantees. In this paper we introduce two algorithms: Team-PSRO, an extension of PSRO from two-player games to team games, and Team-PSRO Mix-and-Match which improves upon Team PSRO by better using population policies. In Team-PSRO, in every iteration both teams learn a joint best response to the opponent's meta-strategy via reinforcement learning. As the reinforcement learning joint best response approaches the optimal best response, Team-PSRO is guaranteed to converge to a TMECor. In experiments on Kuhn poker and Liar's Dice, we show that a tabular version of Team-PSRO converges to TMECor, and a version of Team PSRO using deep cooperative reinforcement learning beats self-play reinforcement learning in the large game of Google Research Football.</details>

### [Spotlight] Temperature Balancing, Layer-wise Weight Analysis, and Neural Network Training

**Authors:** Yefan Zhou, TIANYU PANG, Keqin Liu, charles martin, Michael Mahoney, Yaoqing Yang

**<details><summary>Abstract**</summary>

Regularization in modern machine learning is crucial, and it can take various forms in algorithmic design: training set, model family, error function, regularization terms, and optimizations. In particular, the learning rate, which can be interpreted as a temperature-like parameter within the statistical mechanics of learning, plays a crucial role in neural network training. Indeed, many widely adopted training strategies basically just define the decay of the learning rate over time. This process can be interpreted as decreasing a temperature, using either a global learning rate (for the entire model) or a learning rate that varies for each parameter. This paper proposes TempBalance, a straightforward yet effective layer-wise learning rate method. TempBalance is based on Heavy-Tailed Self-Regularization (HT-SR) Theory, an approach which characterizes the implicit self-regularization of different layers in trained models. We demonstrate the efficacy of using HT-SR-motivated metrics to guide the scheduling and balancing of temperature across all network layers during model training, resulting in improved performance during testing. We implement TempBalance on CIFAR10, CIFAR100, SVHN, and TinyImageNet datasets using ResNets, VGGs and WideResNets with various depths and widths. Our results show that TempBalance significantly outperforms ordinary SGD and carefully-tuned spectral norm regularization. We also show that TempBalance outperforms a number of state-of-the-art optimizers and learning rate schedulers.</details>

### Test-Time Amendment with a Coarse Classifier for Fine-Grained Classification

**Authors:** Kanishk Jain, Shyamgopal Karthik, Vineet Gandhi

**<details><summary>Abstract**</summary>

We investigate the problem of reducing mistake severity for fine-grained classification. Fine-grained classification can be challenging, mainly due to the requirement of knowledge or domain expertise for accurate annotation. However, humans are particularly adept at performing coarse classification as it requires relatively low levels of expertise. To this end, we present a novel approach for Post-Hoc Correction called Hierarchical Ensembles (HiE) that utilizes label hierarchy to improve the performance of fine-grained classification at test-time using the coarse-grained predictions. By only requiring the parents of leaf nodes, our method significantly reduces avg. mistake severity while improving top-1 accuracy on the iNaturalist-19 and tieredImageNet-H datasets, achieving a new state-of-the-art on both benchmarks. We also investigate the efficacy of our approach in the semi-supervised setting. Our approach brings notable gains in top-1 accuracy while significantly decreasing the severity of mistakes as training data decreases for the fine-grained classes. The simplicity and post-hoc nature of HiE renders it practical to be used with any off-the-shelf trained model to improve its predictions further.</details>

### Test-time Adaptation of Discriminative Models via Diffusion Generative Feedback

**Authors:** Mihir Prabhudesai, Tsung-Wei Ke, Alex Li, Deepak Pathak, Katerina Fragkiadaki

**<details><summary>Abstract**</summary>

The advancements in generative modeling, particularly the advent of diffusion models, have sparked a fundamental question: how can these models be effectively used for discriminative tasks? In this work, we find that generative models can be great test-time adapters for discriminative models. Our method, Diffusion-TTA, adapts pre-trained discriminative models such as image classifiers, segmenters and depth predictors, to each unlabelled example in the test set using generative feedback from a diffusion model. We achieve this by modulating the conditioning of the diffusion model using the output of the discriminative model. We then maximize the image likelihood objective by backpropagating the gradients to discriminative model’s parameters. We show Diffusion-TTA significantly enhances the accuracy of various large-scale pre-trained discriminative models, such as, ImageNet classifiers, CLIP models, image pixel labellers and image depth predictors. Diffusion-TTA outperforms existing test-time adaptation methods, including TTT-MAE and TENT, and particularly shines in online adaptation setups, where the discriminative model is continually adapted to each example in the test set. We provide access to code, results, and visualizations on our website: https://diffusion-tta.github.io/</details>

### Text-to-Image Diffusion Models are Zero Shot Classifiers

**Authors:** Kevin Clark, Priyank Jaini

**<details><summary>Abstract**</summary>

The excellent generative capabilities of text-to-image diffusion models suggest they learn informative representations of image-text data.However, what knowledge their representations capture is not fully understood, and they have not been thoroughly explored on downstream tasks.We investigate diffusion models by proposing a method for evaluating them as zero-shot classifiers.The key idea is using a diffusion model's ability to denoise a noised image given a text description of a label as a proxy for that label's likelihood.We apply our method to Stable Diffusion and Imagen, using it to probe fine-grained aspects of the models' knowledge and comparing them with CLIP's zero-shot abilities. They perform competitively with CLIP on a wide range of zero-shot image classification datasets. Additionally, they achieve state-of-the-art results on shape/texture bias tests and can successfully perform attribute binding while CLIP cannot.Although generative pre-training is prevalent in NLP, visual foundation models often use other methods such as contrastive learning. Based on our findings, we argue that generative pre-training should be explored as a compelling alternative for vision and vision-language problems.</details>

### The CLIP Model is Secretly an Image-to-Prompt Converter

**Authors:** Yuxuan Ding, Chunna Tian, Haoxuan Ding, Lingqiao Liu

**<details><summary>Abstract**</summary>

The Stable Diffusion model is a prominent text-to-image generation model that relies on a text prompt as its input, which is encoded using the Contrastive Language-Image Pre-Training (CLIP). However, text prompts have limitations when it comes to incorporating implicit information from reference images. Existing methods have attempted to address this limitation by employing expensive training procedures involving millions of training samples for image-to-image generation. In contrast, this paper demonstrates that the CLIP model, as utilized in Stable Diffusion, inherently possesses the ability to instantaneously convert images into text prompts. Such an image-to-prompt conversion can be achieved by utilizing a linear projection matrix that is calculated in a closed form. Moreover, the paper showcases that this capability can be further enhanced by either utilizing a small amount of similar-domain training data (approximately 100 images) or incorporating several online training steps (around 30 iterations) on the reference images. By leveraging these approaches, the proposed method offers a simple and flexible solution to bridge the gap between images and text prompts. This methodology can be applied to various tasks such as image variation and image editing, facilitating more effective and seamless interaction between images and textual prompts.</details>

### The Learnability of In-Context Learning

**Authors:** Noam Wies, Yoav Levine, Amnon Shashua

**<details><summary>Abstract**</summary>

In-context learning is a surprising and important phenomenon that emerged when modern language models were scaled to billions of learned parameters.   Without modifying a large language model's weights, it can be tuned to perform various downstream natural language tasks simply by including concatenated training examples of these tasks in its input.  Though disruptive for many practical applications of large language models, this emergent learning paradigm is not well understood from a theoretical perspective. In this paper, we propose a first-of-its-kind PAC based framework for in-context learnability, and use it to provide the first finite sample complexity results for the in-context learning setup.  Our framework includes an initial pretraining phase, which fits a function to the pretraining distribution, and then a second in-context learning phase, which keeps this function constant and concatenates training examples of the downstream task in its input.  We use our framework in order to prove that, under mild assumptions, when the pretraining distribution is a mixture of latent tasks (a model often considered for natural language pretraining), these tasks can be efficiently learned via in-context learning, even though the model's weights are unchanged and the input significantly diverges from the pretraining distribution.  Our theoretical analysis reveals that in this setting, in-context learning is more about identifying the task than about learning it, a result which is in line with a series of recent empirical findings.   We hope that the in-context learnability framework presented in this paper will facilitate future progress towards a deeper understanding of this important new learning paradigm.</details>

### The Rise of AI Language Pathologists: Exploring Two-level Prompt Learning for Few-shot Weakly-supervised Whole Slide Image Classification

**Authors:** Linhao Qu, xiaoyuan luo, Kexue Fu, Manning Wang, Zhijian Song

**<details><summary>Abstract**</summary>

This paper introduces the novel concept of few-shot weakly supervised learning for pathology Whole Slide Image (WSI) classification, denoted as FSWC. A solution is proposed based on prompt learning and the utilization of a large language model, GPT-4. Since a WSI is too large and needs to be divided into patches for processing, WSI classification is commonly approached as a Multiple Instance Learning (MIL) problem. In this context, each WSI is considered a bag, and the obtained patches are treated as instances. The objective of FSWC is to classify both bags and instances with only a limited number of labeled bags. Unlike conventional few-shot learning problems, FSWC poses additional challenges due to its weak bag labels within the MIL framework. Drawing inspiration from the recent achievements of vision-language models (V-L models) in downstream few-shot classification tasks, we propose a two-level prompt learning MIL framework tailored for pathology, incorporating language prior knowledge. Specifically, we leverage CLIP to extract instance features for each patch, and introduce a prompt-guided pooling strategy to aggregate these instance features into a bag feature. Subsequently, we employ a small number of labeled bags to facilitate few-shot prompt learning based on the bag features. Our approach incorporates the utilization of GPT-4 in a question-and-answer mode to obtain language prior knowledge at both the instance and bag levels, which are then integrated into the instance and bag level language prompts. Additionally, a learnable component of the language prompts is trained using the available few-shot labeled data. We conduct extensive experiments on three real WSI datasets encompassing breast cancer, lung cancer, and cervical cancer, demonstrating the notable performance of the proposed method in bag and instance classification. All codes will be made publicly accessible.</details>

### [Spotlight] Theoretical and Practical Perspectives on what Influence Functions Do

**Authors:** Andrea Schioppa, Katja Filippova, Ivan Titov, Polina Zablotskaia

**<details><summary>Abstract**</summary>

Influence functions (IF) have been seen as a technique for explaining model predictions through the lens of the training data. Their utility is assumed to be in identifying training examples "responsible" for a prediction so that, for example, correcting a prediction is possible by intervening on those examples (removing or editing them) and retraining the model. However, recent empirical studies have shown that the existing methods of estimating IF predict the leave-one-out-and-retrain effect poorly. In order to understand the mismatch between the theoretical promise and the practical results, we analyse five assumptions made by IF methods which are problematic for modern-scale deep neural networks and which concern convexity, numeric stability, training trajectory and parameter divergence. This allows us to clarify what can be expected theoretically from IF. We show that while most assumptions can be addressed successfully, the parameter divergence poses a clear limitation on the predictive power of IF: influence fades over training time even with deterministic training. We illustrate this theoretical result with BERT and ResNet models.Another conclusion from the theoretical analysis is that IF are still useful for model debugging and correcting even though some of the assumptions made in prior work do not hold: using natural language processing and computer vision tasks, we verify that mis-predictions can be successfully corrected by taking only a few fine-tuning steps on influential examples.</details>

### Theoretically Guaranteed Bidirectional Data Rectification for Robust Sequential Recommendation

**Authors:** Yatong Sun, Bin Wang, Zhu Sun, Xiaochun Yang, Yan Wang

**<details><summary>Abstract**</summary>

Sequential recommender systems (SRSs) are typically trained to predict the next item as the target given its preceding (and succeeding) items as the input. Such a paradigm assumes that every input-target pair is reliable for training. However, users can be induced to click on items that are inconsistent with their true preferences, resulting in unreliable instances, i.e., mismatched input-target pairs. Current studies on mitigating this issue suffer from two limitations: (i) they discriminate instance reliability according to models trained with unreliable data, yet without theoretical guarantees that such a seemingly contradictory solution can be effective; and (ii) most methods can only tackle either unreliable input or targets but fail to handle both simultaneously. To fill the gap, we theoretically unveil the relationship between SRS predictions and instance reliability, whereby two error-bounded strategies are proposed to rectify unreliable targets and input, respectively. On this basis, we devise a model-agnostic Bidirectional Data Rectification (BirDRec) framework, which can be flexibly implemented with most existing SRSs for robust training against unreliable data. Additionally, a rectification sampling strategy is devised and a self-ensemble mechanism is adopted to reduce the (time and space) complexity of BirDRec. Extensive experiments on four real-world datasets verify the generality, effectiveness, and efficiency of our proposed BirDRec.</details>

### Thinker: Learning to Plan and Act

**Authors:** Stephen Chung, Ivan Anokhin, David Krueger

**<details><summary>Abstract**</summary>

We propose the Thinker algorithm, a novel approach that enables reinforcement learning agents to autonomously interact with and utilize a learned world model. The Thinker algorithm wraps the environment with a world model and introduces new actions designed for interacting with the world model. These model-interaction actions enable agents to perform planning by proposing alternative plans to the world model before selecting a final action to execute in the environment. This approach eliminates the need for handcrafted planning algorithms by enabling the agent to learn how to plan autonomously and allows for easy interpretation of the agent's plan with visualization. We demonstrate the algorithm's effectiveness through experimental results in the game of Sokoban and the Atari 2600 benchmark, where the Thinker algorithm achieves state-of-the-art performance and competitive results, respectively. Visualizations of agents trained with the Thinker algorithm demonstrate that they have learned to plan effectively with the world model to select better actions. Thinker is the first work showing that an RL agent can learn to plan with a learned world model in complex environments.</details>

### Toward Re-Identifying Any Animal

**Authors:** Bingliang Jiao, Lingqiao Liu, Liying Gao, Ruiqi Wu, Guosheng Lin, PENG WANG, Yanning Zhang

**<details><summary>Abstract**</summary>

The current state of re-identification (ReID) models poses limitations to their applicability in the open world, as they are primarily designed and trained for specific categories like person or vehicle. In light of the importance of ReID technology for tracking wildlife populations and migration patterns, we propose a new task called ``Re-identify Any Animal in the Wild'' (ReID-AW). This task aims to develop a ReID model capable of handling any unseen wildlife category it encounters. To address this challenge, we have created a comprehensive dataset called Wildlife-71, which includes ReID data from 71 different wildlife categories. This dataset is the first of its kind to encompass multiple object categories in the realm of ReID. Furthermore, we have developed a universal re-identification model named UniReID specifically for the ReID-AW task. To enhance the model's adaptability to the target category, we employ a dynamic prompting mechanism using category-specific visual prompts. These prompts are generated based on knowledge gained from a set of pre-selected images within the target category. Additionally, we leverage explicit semantic knowledge derived from the large-scale pre-trained language model, GPT-4. This allows UniReID to focus on regions that are particularly useful for distinguishing individuals within the target category. Extensive experiments have demonstrated the remarkable generalization capability of our UniReID model. It showcases promising performance in handling arbitrary wildlife categories, offering significant advancements in the field of ReID for wildlife conservation and research purposes.</details>

### Towards Optimal Caching and Model Selection for Large Model Inference

**Authors:** Banghua Zhu, Ying Sheng, Lianmin Zheng, Clark Barrett, Michael Jordan, Jiantao Jiao

**<details><summary>Abstract**</summary>

Large Language Models (LLMs) and other large foundation models have achieved impressive results, but their size exacerbates existing resource consumption and latency challenges. In particular, the large-scale deployment of these models is hindered by the significant resource requirements during inference. In this paper, we study  two  approaches for mitigating these challenges: employing a cache to store previous queries and learning a model selector to choose from an ensemble of models for query processing.Theoretically, we provide an optimal algorithm for jointly optimizing both approaches  to reduce the inference cost in both offline and online tabular settings. By combining a caching algorithm, namely Greedy Dual Size with Frequency (GDSF) or Least Expected Cost (LEC), with a model selector, we achieve optimal rates in both offline and online settings. Empirically, simulations show that our caching and model selection algorithm greatly improves over the baselines, with up to $50\times$ improvement over the baseline when the ratio between the maximum cost and minimum cost is $100$.  Experiments on real datasets show a $4.3\times$ improvement in FLOPs over the baseline when the ratio for FLOPs is $10$, and a $1.8\times$ improvement in latency when the ratio for average latency is $1.85$.</details>

### Towards robust and generalizable representations of extracellular data using contrastive learning

**Authors:** Ankit Vishnubhotla, Charlotte Loh, Akash Srivastava, Liam Paninski, Cole Hurwitz

**<details><summary>Abstract**</summary>

Contrastive learning is quickly becoming an essential tool in neuroscience for extracting robust and meaningful representations of neural activity. Despite numerous applications to neuronal population data, there has been little exploration of how these methods can be adapted to key primary data analysis tasks such as spike sorting or cell-type classification. In this work, we propose a novel contrastive learning framework, CEED (Contrastive Embeddings for Extracellular Data), for high-density extracellular recordings. We demonstrate that through careful design of the network architecture and data augmentations, it is possible to generically extract representations that far outperform current specialized approaches. We validate our method across multiple high-density extracellular recordings. All code used to run CEED can be found at https://github.com/ankitvishnu23/CEED.</details>

### Towards the Difficulty for a Deep Neural Network to Learn Concepts of Different Complexities

**Authors:** Dongrui Liu, Huiqi Deng, Xu Cheng, Qihan Ren, Kangrui Wang, Quanshi Zhang

**<details><summary>Abstract**</summary>

This paper theoretically explains the intuition that simple concepts are more likely to be learned by deep neural networks (DNNs) than complex concepts. In fact, recent studies have observed [24, 15] and proved [26] the emergence of interactive concepts in a DNN, i.e., it is proven that a DNN usually only encodes a small number of interactive concepts, and can be considered to use their interaction effects to compute inference scores. Each interactive concept is encoded by the DNN to represent the collaboration between a set of input variables. Therefore, in this study, we aim to theoretically explain that interactive concepts involving more input variables (i.e., more complex concepts) are more difficult to learn. Our finding clarifies the exact conceptual complexity that boosts the learning difficulty.</details>

### Train Once and Explain Everywhere: Pre-training Interpretable Graph Neural Networks

**Authors:** Jun Yin, Chaozhuo Li, Hao Yan, Jianxun Lian, Senzhang Wang

**<details><summary>Abstract**</summary>

Intrinsic interpretable graph neural networks aim to provide transparent predictions by identifying the influential fraction of the input graph that guides the model prediction, i.e., the explanatory subgraph. However, current interpretable GNNs mostly are dataset-specific and hard to generalize to different graphs. A more generalizable GNN interpretation model which can effectively distill the universal structural patterns of different graphs is until-now unexplored. Motivated by the great success of recent pre-training techniques, we for the first time propose the Pre-training Interpretable Graph Neural Network ($\pi$-GNN) to distill the universal interpretability of GNNs by pre-training over synthetic graphs with ground-truth explanations. Specifically, we introduce a structural pattern learning module to extract diverse universal structure patterns and integrate them together to comprehensively represent the graphs of different types. Next, a hypergraph refining module is proposed to identify the explanatory subgraph by incorporating the universal structure patterns with local edge interactions. Finally, the task-specific predictor is cascaded with the pre-trained $\pi$-GNN model and fine-tuned over downstream tasks. Extensive experiments demonstrate that $\pi$-GNN significantly surpasses the leading interpretable GNN baselines with up to 9.98\% interpretation improvement and 16.06\% classification accuracy improvement. Meanwhile, $\pi$-GNN pre-trained on graph classification task also achieves the top-tier interpretation performance on node classification task, which further verifies its promising generalization performance among different downstream tasks. Our code and datasets are available at https://anonymous.4open.science/r/PI-GNN-F86C</details>

### Training Transitive and Commutative Multimodal Transformers with LoReTTa

**Authors:** Manuel Tran, Yashin Dicente Cid, Amal Lahiani, Fabian Theis, Tingying Peng, Eldad Klaiman

**<details><summary>Abstract**</summary>

Training multimodal foundation models is challenging due to the limited availability of multimodal datasets. While many public datasets pair images with text, few combine images with audio or text with audio. Even rarer are datasets that align all three modalities at once. Critical domains such as healthcare, infrastructure, or transportation are particularly affected by missing modalities. This makes it difficult to integrate all modalities into a large pre-trained neural network that can be used out-of-the-box or fine-tuned for different downstream tasks. We introduce LoReTTa ($\textbf{L}$inking m$\textbf{O}$dalities with a t$\textbf{R}$ansitive and commutativ$\textbf{E}$ pre-$\textbf{T}$raining s$\textbf{T}$r$\textbf{A}$tegy) to address this understudied problem. Our self-supervised framework unifies causal modeling and masked modeling with the rules of commutativity and transitivity. This allows us to transition within and between modalities. As a result, our pre-trained models are better at exploring the true underlying joint probability distribution. Given a dataset containing only the disjoint combinations $(A, B)$ and $(B, C)$, LoReTTa can model the relation $A \leftrightarrow C$ with $A \leftrightarrow B \leftrightarrow C$. In particular, we show that a transformer pre-trained with LoReTTa can handle any mixture of modalities at inference time, including the never-seen pair $(A, C)$ and the triplet $(A, B, C)$. We extensively evaluate our approach on a synthetic, medical, and reinforcement learning dataset. Across different domains, our universal multimodal transformer consistently outperforms strong baselines such as GPT, BERT, and CLIP on tasks involving the missing modality tuple.</details>

### Transformed Low-Rank Parameterization Can Help Robust Generalization for Tensor Neural Networks

**Authors:** Andong Wang, Chao Li, Mingyuan Bai, Zhong Jin, Guoxu Zhou, Qibin Zhao

**<details><summary>Abstract**</summary>

Multi-channel learning has gained significant attention in recent applications, where neural networks with t-product layers (t-NNs) have shown promising performance through novel feature mapping in the transformed domain. However, despite the practical success of t-NNs, the theoretical analysis of their generalization remains unexplored. We address this gap by deriving upper bounds on the generalization error of t-NNs in both standard and adversarial settings. Notably, it reveals that t-NNs compressed with exact transformed low-rank parameterization can achieve tighter adversarial generalization bounds compared to non-compressed models. While exact transformed low-rank weights are rare in practice, the analysis demonstrates that through adversarial training with gradient flow, highly over-parameterized t-NNs with the ReLU activation can be implicitly regularized towards a transformed low-rank parameterization under certain conditions. Moreover, this paper establishes sharp adversarial generalization bounds for t-NNs with approximately transformed low-rank weights. Our analysis highlights the potential of transformed low-rank parameterization in enhancing the robust generalization of t-NNs, offering valuable insights for further research and development.</details>

### Transformers over Directed Acyclic Graphs

**Authors:** Yuankai Luo, Veronika Thost, Lei Shi

**<details><summary>Abstract**</summary>

Transformer models have recently gained popularity in graph representation learning as they have the potential to learn complex relationships beyond the ones captured by regular graph neural networks.The main research question is how to inject the structural bias of graphs into the transformer architecture,and several proposals have been made for undirected molecular graphs and, recently, also for larger network graphs.In this paper, we study transformers over directed acyclic graphs (DAGs) and propose architecture adaptations tailored to DAGs: (1) An attention mechanism that is considerably more efficient than the regular quadratic complexity of transformers and at the same time faithfully captures the DAG structure, and (2) a positional encoding of the DAG's partial order, complementing the former.We rigorously evaluate our approach over various types of tasks, ranging from classifying source code graphs to nodes in citation networks, and show that it is effective in two important aspects: in making graph transformers generally outperform graph neural networks tailored to DAGs and in improving SOTA graph transformer performance in terms of both quality and efficiency.</details>

### [Spotlight] Transient Neural Radiance Fields for Lidar View Synthesis and 3D Reconstruction

**Authors:** Anagh Malik, Parsa Mirdehghan, Sotiris Nousias, Kyros Kutulakos, David Lindell

**<details><summary>Abstract**</summary>

Neural radiance fields (NeRFs) have become a ubiquitous tool for modeling scene appearance and geometry from multiview imagery. Recent work has also begun to explore how to use additional supervision from lidar or depth sensor measurements in the NeRF framework. However, previous lidar-supervised NeRFs focus on rendering conventional camera imagery and use lidar-derived point cloud data as auxiliary supervision; thus, they fail to incorporate the underlying image formation model of the lidar. Here, we propose a novel method for rendering transient NeRFs that take as input the raw, time-resolved photon count histograms measured by a single-photon lidar system, and we seek to render such histograms from novel views. Different from conventional NeRFs, the approach relies on a time-resolved version of the volume rendering equation to render the lidar measurements and capture transient light transport phenomena at picosecond timescales. We evaluate our method on a first-of-its-kind dataset of simulated and captured transient multiview scans from a prototype single-photon lidar. Overall, our work brings NeRFs to a new dimension of imaging at transient timescales, newly enabling rendering of transient imagery from novel views. Additionally, we show that our approach recovers improved geometry and conventional appearance compared to point cloud-based supervision when training on few input viewpoints. Transient NeRFs may be especially useful for applications which seek to simulate raw lidar measurements for downstream tasks in autonomous driving, robotics, and remote sensing.</details>

### TriRE: A Multi-Mechanism Learning Paradigm for Continual Knowledge Retention and Promotion

**Authors:** Preetha Vijayan, Prashant Bhat, Bahram Zonooz, Elahe Arani

**<details><summary>Abstract**</summary>

Continual learning (CL) has remained a persistent challenge for deep neural networks due to catastrophic forgetting (CF) of previously learned tasks. Several techniques such as weight regularization, experience rehearsal, and parameter isolation have been proposed to alleviate CF. Despite their relative success, these research directions have predominantly remained orthogonal and suffer from several shortcomings, while missing out on the advantages of competing strategies. On the contrary, the brain continually learns, accommodates, and transfers knowledge across tasks by simultaneously leveraging several neurophysiological processes, including neurogenesis, active forgetting, neuromodulation, metaplasticity, experience rehearsal, and context-dependent gating, rarely resulting in CF. Inspired by how the brain exploits multiple mechanisms concurrently, we propose TriRE, a novel CL paradigm that encompasses retaining the most prominent neurons for each task, revising and solidifying the extracted knowledge of current and past tasks, and actively promoting less active neurons for subsequent tasks through rewinding and relearning. Across CL settings, TriRE significantly reduces task interference and surpasses different CL approaches considered in isolation.</details>

### Triple Eagle: Simple, Fast and Practical Budget-Feasible Mechanisms

**Authors:** Kai Han, You Wu, He Huang, Shuang Cui

**<details><summary>Abstract**</summary>

We revisit the classical problem of designing Budget-Feasible Mechanisms (BFMs) for submodular valuation functions, which has been extensively studied since the seminal paper of Singer [FOCS'10] due to its wide applications in crowdsourcing and social marketing. We propose TripleEagle, a novel algorithmic framework for designing BFMs, based on which we present several simple yet effective BFMs that achieve better approximation ratios than the state-of-the-art work. Moreover, our BFMs are the first in the literature to achieve linear complexities while ensuring obvious strategyproofness, making them more practical than the previous BFMs. We conduct extensive experiments to evaluate the empirical performance of our BFMs, and the experimental results strongly demonstrate the efficiency and effectiveness of our approach.</details>

### Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection

**Authors:** Hezhe Qiao, Guansong Pang

**<details><summary>Abstract**</summary>

We reveal a one-class homophily phenomenon, which is one prevalent property we find empirically in real-world graph anomaly detection (GAD) datasets, i.e., normal nodes tend to have strong connection/affinity with each other, while the homophily in abnormal nodes is significantly weaker than normal nodes. However, this anomaly-discriminative property is ignored by existing GAD methods that are typically built using a conventional anomaly detection objective, such as data reconstruction.In this work, we explore this property to introduce a novel unsupervised anomaly scoring measure for GAD -- local node affinity-- that assigns a larger anomaly score to nodes that are less affiliated with their neighbors, with the affinity defined as similarity on node attributes/representations. We further propose Truncated Affinity Maximization (TAM) that learns tailored node representations for our anomaly measure by maximizing the local affinity of nodes to their neighbors. Optimizing on the original graph structure can be biased by non-homophily edges(i.e., edges connecting normal and abnormal nodes). Thus, TAM is instead optimized on truncated graphs where non-homophily edges are removed iteratively to mitigate this bias. The learned representations result in significantly stronger local affinity for normal nodes than abnormal nodes. Extensive empirical results on 10 real-world GAD datasets show that TAM substantially outperforms seven competing models, achieving over 10% increase in AUROC/AUPRC compared to the best contenders on challenging datasets. Our code is available at https://github.com/mala-lab/TAM-master/.</details>

### Trust Region-Based Safe Distributional Reinforcement Learning for Multiple Constraints

**Authors:** Dohyeong Kim, Kyungjae Lee, Songhwai Oh

**<details><summary>Abstract**</summary>

In safety-critical robotic tasks, potential failures must be reduced, and multiple constraints must be met, such as avoiding collisions, limiting energy consumption, and maintaining balance.Thus, applying safe reinforcement learning (RL) in such robotic tasks requires to handle multiple constraints and use risk-averse constraints rather than risk-neutral constraints.To this end, we propose a trust region-based safe RL algorithm for multiple constraints called a safe distributional actor-critic (SDAC).Our main contributions are as follows: 1) introducing a gradient integration method to manage infeasibility issues in multi-constrained problems, ensuring theoretical convergence, and 2) developing a TD($\lambda$) target distribution to estimate risk-averse constraints with low biases. We evaluate SDAC through extensive experiments involving multi- and single-constrained robotic tasks.While maintaining high scores, SDAC shows 1.93 times fewer steps to satisfy all constraints in multi-constrained tasks and 1.78 times fewer constraint violations in single-constrained tasks compared to safe RL baselines.Code is available at: https://github.com/rllab-snu/Safe-Distributional-Actor-Critic.</details>

### Two Sides of One Coin: the Limits of Untuned SGD and the Power of Adaptive Methods

**Authors:** Junchi YANG, Xiang Li, Ilyas Fatkhullin, Niao He

**<details><summary>Abstract**</summary>

The classical analysis of Stochastic Gradient Descent (SGD) with polynomially decaying stepsize $\eta_t = \eta/\sqrt{t}$ relies on well-tuned $\eta$ depending on  problem parameters such as Lipschitz smoothness constant, which is often unknown in practice. In this work, we prove that SGD with arbitrary $\eta > 0$, referred to as untuned SGD, still attains an order-optimal convergence rate $\widetilde{\mathcal{O}}(T^{-1/4})$ in terms of gradient norm for minimizing smooth objectives. Unfortunately, it comes at the expense of a catastrophic exponential dependence on the smoothness constant, which we show is unavoidable for this scheme even in the noiseless setting. We then examine three families of adaptive methods — Normalized SGD (NSGD), AMSGrad, and AdaGrad — unveiling their power in preventing such exponential dependency in  the absence of information about the smoothness parameter and boundedness of stochastic gradients. Our results provide  theoretical justification for the advantage of adaptive methods over untuned SGD in alleviating the issue with large gradients.</details>

### UE4-NeRF:Neural Radiance Field for Real-Time Rendering of Large-Scale Scene

**Authors:** Jiaming Gu, Minchao Jiang, Hongsheng Li, Xiaoyuan Lu, Guangming Zhu, Syed Afaq Ali Shah, Liang Zhang, Mohammed Bennamoun

**<details><summary>Abstract**</summary>

Neural Radiance Fields (NeRF) is a novel implicit 3D reconstruction method that shows immense potential and has been gaining increasing attention. It enables the reconstruction of 3D scenes solely from a set of photographs. However, its real-time rendering capability, especially for interactive real-time rendering of large-scale scenes, still has significant limitations. To address these challenges, in this paper, we propose a novel neural rendering system called UE4-NeRF, specifically designed for real-time rendering of large-scale scenes. We partitioned each large scene into different sub-NeRFs. In order to represent the partitioned independent scene, we initialize polygonal meshes by constructing multiple regular octahedra within the scene and the vertices of the polygonal faces are continuously optimized during the training process. Drawing inspiration from Level of Detail (LOD) techniques, we trained meshes of varying levels of detail for different observation levels. Our approach combines with the rasterization pipeline in Unreal Engine 4 (UE4), achieving real-time rendering of large-scale scenes at 4K resolution with a frame rate of up to 43 FPS. Rendering within UE4 also facilitates scene editing in subsequent stages. Furthermore, through experiments, we have demonstrated that our method achieves rendering quality comparable to state-of-the-art approaches. Project page: https://jamchaos.github.io/UE4-NeRF/.</details>

### Unbiased constrained sampling with Self-Concordant Barrier Hamiltonian Monte Carlo

**Authors:** Maxence Noble, Valentin De Bortoli, Alain Durmus

**<details><summary>Abstract**</summary>

In this paper, we propose Barrier Hamiltonian Monte Carlo (BHMC), a version of the  HMC algorithm which aims at sampling from a Gibbs distribution $\pi$ on a manifold  $\mathsf{M}$, endowed with a Hessian metric $\mathfrak{g}$ derived from a self-concordant  barrier. Our method relies on Hamiltonian  dynamics which comprises $\mathfrak{g}$. Therefore, it incorporates the constraints defining  $\mathsf{M}$ and is able to exploit its underlying geometry. However,   the corresponding Hamiltonian dynamics is defined via non separable Ordinary Differential Equations (ODEs) in contrast to the Euclidean case. It implies unavoidable bias in existing generalization of HMC to Riemannian manifolds. In this paper, we propose a new filter step, called ``involution checking step'', to address this problem. This step is implemented in two versions of BHMC, coined continuous BHMC (c-bHMC) and  numerical BHMC (n-BHMC) respectively.  Our main results establish that these two new algorithms  generate reversible Markov  chains with respect to $\pi$ and do not suffer from any bias in comparison to previous implementations. Our conclusions are supported by numerical experiments where  we consider target distributions defined on polytopes.</details>

### Uncertainty Quantification via Neural Posterior Principal Components

**Authors:** Elias Nehme, Omer Yair, Tomer Michaeli

**<details><summary>Abstract**</summary>

Uncertainty quantification is crucial for the deployment of image restoration models in safety-critical domains, like autonomous driving and biological imaging. To date, methods for uncertainty visualization have mainly focused on per-pixel estimates. Yet, a heatmap of per-pixel variances is typically of little practical use, as it does not capture the strong correlations between pixels. A more natural measure of uncertainty corresponds to the variances along the principal components (PCs) of the posterior distribution. Theoretically, the PCs can be computed by applying PCA on samples generated from a conditional generative model for the input image. However, this requires generating a very large number of samples at test time, which is painfully slow with the current state-of-the-art (diffusion) models. In this work, we present a method for predicting the PCs of the posterior distribution for any input image, in a single forward pass of a neural network. Our method can either wrap around a pre-trained model that was trained to minimize the mean square error (MSE), or can be trained from scratch to output both a predicted image and the posterior PCs. We showcase our method on multiple inverse problems in imaging, including denoising, inpainting, super-resolution, and biological image-to-image translation. Our method reliably conveys instance-adaptive uncertainty directions, achieving uncertainty quantification comparable with posterior samplers while being orders of magnitude faster. Code and examples are available on our [webpage](https://eliasnehme.github.io/NPPC/).</details>

### Uncertainty-Aware Alignment  Network  for Cross-Domain Video-Text Retrieval

**Authors:** Xiaoshuai Hao, Wanqian Zhang

**<details><summary>Abstract**</summary>

Video-text retrieval is an important but challenging research task in the multimedia community.  In this paper, we address the challenge task of Unsupervised Domain Adaptation Video-text Retrieval (UDAVR), assuming that training (source) data and testing (target) data are from different domains. Previous approaches are mostly derived from classification based domain adaptation methods, which are neither multi-modal nor suitable for retrieval task.  In addition, as to the pairwise misalignment issue in target domain, i.e., no pairwise annotations between target videos and texts, the existing method assumes that a video corresponds to a text. Yet we empirically find that in the real scene, one text usually corresponds to multiple videos and vice versa. To tackle this one-to-many issue, we propose a novel method named Uncertainty-aware Alignment Network (UAN). Specifically, we first introduce the multimodal mutual information module to balance the minimization of domain shift in a smooth manner. To tackle the multimodal uncertainties pairwise misalignment in target domain, we propose the Uncertainty-aware Alignment Mechanism (UAM) to fully exploit the semantic information of both modalities in target domain. Extensive experiments in the context of domain-adaptive video-text retrieval demonstrate that our proposed method consistently outperforms multiple baselines, showing a superior generalization ability for target data.</details>

### Unconstrained Dynamic Regret via Sparse Coding

**Authors:** Zhiyu Zhang, Ashok Cutkosky, Yannis Paschalidis

**<details><summary>Abstract**</summary>

Motivated by the challenge of nonstationarity in sequential decision making, we study Online Convex Optimization (OCO) under the coupling of two problem structures: the domain is unbounded, and the comparator sequence $u_1,\ldots,u_T$ is arbitrarily time-varying. As no algorithm can guarantee low regret simultaneously against all comparator sequences, handling this setting requires moving from minimax optimality to comparator adaptivity. That is, sensible regret bounds should depend on certain complexity measures of the comparator relative to one's prior knowledge. This paper achieves a new type of such adaptive regret bounds leveraging a sparse coding framework. The complexity of the comparator is measured by its energy and its sparsity on a user-specified dictionary, which offers considerable versatility. For example, equipped with a wavelet dictionary, our framework improves the state-of-the-art bound (Jacobsen & Cutkosky, 2022) by adapting to both ($i$) the magnitude of the comparator average $||\bar u||=||\sum_{t=1}^Tu_t/T||$, rather than the maximum $\max_t||u_t||$; and ($ii$) the comparator variability $\sum_{t=1}^T||u_t-\bar u||$, rather than the uncentered sum $\sum_{t=1}^T||u_t||$. Furthermore, our proof is simpler due to decoupling function approximation from regret minimization.</details>

### [Spotlight] Understanding Multi-phase Optimization Dynamics and Rich Nonlinear Behaviors of ReLU Networks

**Authors:** Mingze Wang, Chao Ma

**<details><summary>Abstract**</summary>

The training process of ReLU neural networks often exhibits complicated nonlinear phenomena. The nonlinearity of models and non-convexity of loss pose significant challenges for theoretical analysis. Therefore, most previous theoretical works on the optimization dynamics of neural networks focus either on local analysis (like the end of training) or approximate linear models (like Neural Tangent Kernel). In this work, we conduct a complete theoretical characterization of the training process of a two-layer ReLU network trained by Gradient Flow on a linearly separable data. In this specific setting, our analysis captures the whole optimization process starting from random initialization to final convergence. Despite the relatively simple model and data that we studied, we reveal four different phases from the whole training process showing a general simplifying-to-complicating learning trend.Specific nonlinear behaviors can also be precisely identified and captured theoretically, such asinitial condensation, saddle-to-plateau dynamics, plateau escape, changes of activation patterns, learning with increasing complexity, etc.</details>

### Understanding and Improving Feature Learning for Out-of-Distribution Generalization

**Authors:** Yongqiang Chen, Wei Huang, Kaiwen Zhou, Yatao Bian, Bo Han, James Cheng

**<details><summary>Abstract**</summary>

A common explanation for the failure of out-of-distribution (OOD) generalization is that the model trained with empirical risk minimization (ERM) learns spurious features instead of invariant features. However, several recent studies challenged this explanation and found that deep networks may have already learned sufficiently good features for OOD generalization. Despite the contradictions at first glance, we theoretically show that ERM essentially learns both spurious and invariant features, while ERM tends to learn spurious features faster if the spurious correlation is stronger. Moreover, when fed the ERM learned features to the OOD objectives, the invariant feature learning quality significantly affects the final OOD performance, as OOD objectives rarely learn new features. Therefore, ERM feature learning can be a bottleneck to OOD generalization. To alleviate the reliance, we propose Feature Augmented Training (FeAT), to enforce the model to learn richer features ready for OOD generalization. FeAT iteratively augments the model to learn new features while retaining the already learned features. In each round, the retention and augmentation operations are performed on different subsets of the training data that capture distinct features. Extensive experiments show that FeAT effectively learns richer features thus boosting the performance of various OOD objectives.</details>

### UniTSFace: Unified Threshold Integrated Sample-to-Sample Loss for Face Recognition

**Authors:** qiufu li, Xi Jia, Jiancan Zhou, Linlin Shen, Jinming Duan

**<details><summary>Abstract**</summary>

Sample-to-class-based face recognition models can not fully explore the cross-sample relationship among large amounts of facial images, while sample-to-sample-based models require sophisticated pairing processes for training. Furthermore, neither method satisfies the requirements of real-world face verification applications, which expect a unified threshold separating positive from negative facial pairs. In this paper, we propose a unified threshold integrated sample-to-sample based loss (USS loss), which features an explicit unified threshold for distinguishing positive from negative pairs. Inspired by our USS loss, we also derive the sample-to-sample based softmax and BCE losses, and discuss their relationship. Extensive evaluation on multiple benchmark datasets, including MFR, IJB-C, LFW, CFP-FP, AgeDB, and MegaFace, demonstrates that the proposed USS loss is highly efficient and can work seamlessly with sample-to-class-based losses. The embedded loss (USS and sample-to-class Softmax loss) overcomes the pitfalls of previous approaches and the trained facial model UniTSFace exhibits exceptional performance, outperforming state-of-the-art methods, such as CosFace, ArcFace, VPL, AnchorFace, and UNPG. Our code is available at https://github.com/CVI-SZU/UniTSFace.</details>

### [Spotlight] Unified Embedding: Battle-Tested Feature Representations for Web-Scale ML Systems

**Authors:** Benjamin Coleman, Wang-Cheng Kang, Matthew Fahrbach, Ruoxi Wang, Lichan Hong, Ed Chi, Derek Cheng

**<details><summary>Abstract**</summary>

Learning high-quality feature embeddings efficiently and effectively is critical for the performance of web-scale machine learning systems. A typical model ingests hundreds of features with vocabularies on the order of millions to billions of tokens. The standard approach is to represent each feature value as a $d$-dimensional embedding, which introduces hundreds of billions of parameters for extremely high-cardinality features. This bottleneck has led to substantial progress in alternative embedding algorithms. Many of these methods, however, make the assumption that each feature uses an independent embedding table. This work introduces a simple yet highly effective framework, Feature Multiplexing, where one single representation space is used for many different categorical features. Our theoretical and empirical analysis reveals that multiplexed embeddings can be decomposed into components from each constituent feature, allowing models to distinguish between features. We show that multiplexed representations give Pareto-optimal space-accuracy tradeoffs for three public benchmark datasets. Further, we propose a highly practical approach called Unified Embedding with three major benefits: simplified feature configuration, strong adaptation to dynamic data distributions, and compatibility with modern hardware. Unified embedding gives significant improvements in offline and online metrics compared to highly competitive baselines across five web-scale search, ads, and recommender systems, where it serves billions of users across the world in industry-leading products.</details>

### Unified Enhancement of Privacy Bounds for Mixture Mechanisms via $f$-Differential Privacy

**Authors:** Chendi Wang, Buxin Su, Jiayuan Ye, Reza Shokri, Weijie Su

**<details><summary>Abstract**</summary>

Differentially private (DP) machine learning algorithms incur many sources of randomness, such as random initialization, random batch subsampling, and shuffling. However, such randomness is difficult to take into account when proving differential privacy bounds because it induces mixture distributions for the algorithm's output that are difficult to analyze. This paper focuses on improving privacy bounds for shuffling models and one-iteration  differentially private gradient descent (DP-GD) with random initializations using $f$-DP. We derive a closed-form expression of the trade-off function for shuffling models that outperforms the most up-to-date results based on $(\epsilon,\delta)$-DP.Moreover, we investigate the effects of random initialization on the privacy of one-iteration DP-GD. Our numerical computations of the trade-off function indicate that random initialization can enhance the privacy of DP-GD.Our analysis of $f$-DP guarantees for these mixture mechanisms relies on an inequality for trade-off functions introduced in this paper. This inequality implies the joint convexity of $F$-divergences. Finally, we study an $f$-DP analog of the advanced joint convexity of the hockey-stick divergence related to $(\epsilon,\delta)$-DP  and apply it to analyze the privacy of mixture mechanisms.</details>

### Unified Off-Policy Learning to Rank: a Reinforcement Learning Perspective

**Authors:** Zeyu Zhang, Yi Su, Hui Yuan, Yiran Wu, Rishab Balasubramanian, Qingyun Wu, Huazheng Wang, Mengdi Wang

**<details><summary>Abstract**</summary>

Off-policy Learning to Rank (LTR) aims to optimize a ranker from data collected by a deployed logging policy. However, existing off-policy learning to rank methods often make strong assumptions about how users generate the click data, i.e., the click model, and hence need to tailor their methods specifically under different click models. In this paper, we unified the ranking process under general stochastic click models as a Markov Decision Process (MDP), and the optimal ranking could be learned with offline reinforcement learning (RL) directly. Building upon this, we leverage offline RL techniques for off-policy LTR and propose the Click Model-Agnostic Unified Off-policy Learning to Rank (CUOLR) method, which could be easily applied to a wide range of click models. Through a dedicated formulation of the MDP, we show that offline RL algorithms can adapt to various click models without complex debiasing techniques and prior knowledge of the model. Results on various large-scale datasets demonstrate that CUOLR consistently outperforms the state-of-the-art off-policy learning to rank algorithms while maintaining consistency and robustness under different click models.</details>

### Unleash the Potential of Image Branch for Cross-modal 3D Object Detection

**Authors:** Yifan Zhang, Qijian Zhang, Junhui Hou, Yixuan Yuan, Guoliang Xing

**<details><summary>Abstract**</summary>

To achieve reliable and precise scene understanding, autonomous vehicles typically incorporate multiple sensing modalities to capitalize on their complementary attributes. However, existing cross-modal 3D detectors do not fully utilize the image domain information to address the bottleneck issues of the LiDAR-based detectors. This paper presents a new cross-modal 3D object detector, namely UPIDet, which aims to unleash the potential of the image branch from two aspects. First, UPIDet introduces a new 2D auxiliary task called normalized local coordinate map estimation. This approach enables the learning of local spatial-aware features from the image modality to supplement sparse point clouds. Second, we discover that the representational capability of the point cloud backbone can be enhanced through the gradients backpropagated from the training objectives of the image branch, utilizing a succinct and effective point-to-pixel module. Extensive experiments and ablation studies validate the effectiveness of our method. Notably, we achieved the top rank in the highly competitive cyclist class of the KITTI benchmark at the time of submission. The source code is available at https://github.com/Eaphan/UPIDet.</details>

### Unleashing the Power of Graph Data Augmentation on Covariate Distribution Shift

**Authors:** Yongduo Sui, Qitian Wu, Jiancan Wu, Qing Cui, Longfei Li, Jun Zhou, Xiang Wang, Xiangnan He

**<details><summary>Abstract**</summary>

The issue of distribution shifts is emerging as a critical concern in graph representation learning. From the perspective of invariant learning and stable learning, a recently well-established paradigm for out-of-distribution generalization, stable features of the graph are assumed to causally determine labels, while environmental features tend to be unstable and can lead to the two primary types of distribution shifts. The correlation shift is often caused by the spurious correlation between environmental features and labels that differs between the training and test data; the covariate shift often stems from the presence of new environmental features in test data. However, most strategies, such as invariant learning or graph augmentation, typically struggle with limited training environments or perturbed stable features, thus exposing limitations in handling the problem of covariate shift. To address this challenge, we propose a simple-yet-effective data augmentation strategy, Adversarial Invariant Augmentation (AIA), to handle the covariate shift on graphs. Specifically, given the training data, AIA aims to extrapolate and generate new environments, while concurrently preserving the original stable features during the augmentation process. Such a design equips the graph classification model with an enhanced capability to identify stable features in new environments, thereby effectively tackling the covariate shift in data. Extensive experiments with in-depth empirical analysis demonstrate the superiority of our approach. The implementation codes are publicly available at https://github.com/yongduosui/AIA.</details>

### Unleashing the Power of Randomization in Auditing Differentially Private ML

**Authors:** Krishna Pillutla, Galen Andrew, Peter Kairouz, H. Brendan McMahan, Alina Oprea, Sewoong Oh

**<details><summary>Abstract**</summary>

We present a rigorous methodology for auditing differentially private machine learning by adding multiple carefully designed examples called canaries. We take a first principles approach based on three key components. First, we introduce Lifted Differential Privacy (LiDP) that expands the definition of differential privacy to handle randomized datasets. This gives us the freedom to design randomized canaries. Second, we audit LiDP by trying to distinguish between the model trained with $K$ canaries versus $K-1$ canaries in the dataset, leaving one canary out. By drawing the canaries i.i.d., LiDP can leverage the symmetry in the design and reuse each privately trained model to run multiple statistical tests, one for each canary. Third, we introduce novel confidence intervals that take advantage of the multiple test statistics by adapting to the empirical higher-order correlations. Together, this new recipe demonstrates significant improvements in sample complexity, both theoretically and empirically, using synthetic and real data. Further, recent advances in designing stronger canaries can be readily incorporated in the new framework.</details>

### Unlimiformer: Long-Range Transformers with Unlimited Length Input

**Authors:** Amanda Bertsch, Uri Alon, Graham Neubig, Matthew Gormley

**<details><summary>Abstract**</summary>

Since the proposal of transformers, these models have been limited to bounded input lengths, because of their need to attend to every token in the input. In this work, we propose Unlimiformer: a general approach that wraps any existing pretrained encoder-decoder transformer, and offloads the cross-attention computation to a single $k$-nearest-neighbor ($k$NN) index, while the returned $k$NN distances are the attention dot-product scores. This $k$NN index can be kept on either the GPU or CPU memory and queried in sub-linear time; this way, we can index practically unlimited input sequences, while every attention head in every decoder layer retrieves its top-$k$ keys, instead of attending to every key. We evaluate Unlimiformer on several long-document and book-summarization benchmarks, showing that it can process even **500k** token-long inputs from the BookSum dataset, without any input truncation at test time. We demonstrate that Unlimiformer improves pretrained models such as BART and Longformer by extending them to unlimited inputs without additional learned weights and without modifying their code. Our code and models are publicly available at https://github.com/abertsch72/unlimiformer , and support LLaMA-2 as well.</details>

### Unsupervised Learning for Solving the Travelling Salesman Problem

**Authors:** Yimeng Min, Yiwei Bai, Carla Gomes

**<details><summary>Abstract**</summary>

We propose UTSP, an Unsupervised Learning (UL) framework for solving the Travelling Salesman Problem (TSP). We train a Graph Neural Network (GNN) using a surrogate loss. The GNN outputs a heat map representing the probability for each edge to be part of the optimal path. We then apply local search to generate our final prediction based on the heat map. Our loss function consists of two parts: one pushes the model to find the shortest path and the other serves as a surrogate for the constraint that the route should form a Hamiltonian Cycle. Experimental results show that UTSP outperforms the existing data-driven TSP heuristics.Our approach is parameter efficient as well as data efficient: the model takes  $\sim$ 10\% of the number of parameters and $\sim$ 0.2\% of training samples compared with Reinforcement Learning or Supervised Learning methods.</details>

### Use perturbations when learning from explanations

**Authors:** Juyeon Heo, Vihari Piratla, Matthew Wicker, Adrian Weller

**<details><summary>Abstract**</summary>

Machine learning from explanations (MLX) is an approach to learning that uses human-provided explanations of relevant or irrelevant features for each input to ensure that model predictions are right for the right reasons. Existing MLX approaches rely on local model interpretation methods and require strong model smoothing to align model and human explanations, leading to sub-optimal performance. We recast MLX as a robustness problem, where human explanations specify a lower dimensional manifold from which perturbations can be drawn, and show both theoretically and empirically how this approach alleviates the need for strong model smoothing. We consider various approaches to achieving robustness, leading to improved performance over prior MLX methods. Finally, we show how to combine robustness with an earlier MLX method, yielding state-of-the-art results on both synthetic and real-world benchmarks.</details>

### Using Imperfect Surrogates for Downstream Inference: Design-based Supervised Learning for Social Science Applications of Large Language Models

**Authors:** Naoki Egami, Musashi Hinck, Brandon Stewart, Hanying Wei

**<details><summary>Abstract**</summary>

In computational social science (CSS), researchers analyze documents to explain social and political phenomena. In most scenarios, CSS researchers first obtain labels for documents and then explain labels using interpretable regression analyses in the second step. One increasingly common way to annotate documents cheaply at scale is through large language models (LLMs). However, like other scalable ways of producing annotations, such surrogate labels are often imperfect and biased. We present a new algorithm for using imperfect annotation surrogates for downstream statistical analyses while guaranteeing statistical properties—like asymptotic unbiasedness and proper uncertainty quantification—which are fundamental to CSS research. We show that direct use of surrogate labels in downstream statistical analyses leads to substantial bias and invalid confidence intervals, even with high surrogate accuracy of 80-90\%. To address this, we build on debiased machine learning to propose the design-based supervised learning (DSL) estimator. DSL employs a doubly-robust procedure to combine surrogate labels with a smaller number of high-quality, gold-standard labels. Our approach guarantees valid inference for downstream statistical analyses, even when surrogates are arbitrarily biased and without requiring stringent assumptions, by controlling the probability of sampling documents for gold-standard labeling. Both our theoretical analysis and experimental results show that DSL provides valid statistical inference while achieving root mean squared errors comparable to existing alternatives that focus only on prediction without inferential guarantees.</details>

### VLATTACK: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models

**Authors:** Ziyi Yin, Muchao Ye, Tianrong Zhang, Tianyu Du, Jinguo Zhu, Han Liu, Jinghui Chen, Ting Wang, Fenglong Ma

**<details><summary>Abstract**</summary>

Vision-Language (VL) pre-trained models have shown their superiority on many multimodal tasks. However, the adversarial robustness of such models has not been fully explored. Existing approaches mainly focus on exploring the adversarial robustness under the white-box setting, which is unrealistic. In this paper, we aim to investigate a new yet practical task to craft image and text perturbations using pre-trained VL models to attack black-box fine-tuned models on different downstream tasks. Towards this end, we propose VLATTACK to generate adversarial samples by fusing perturbations of images and texts from both single-modal and multi-modal levels. At the single-modal level, we propose a new block-wise similarity attack (BSA) strategy to learn image perturbations for disrupting universal representations. Besides, we adopt an existing text attack strategy to generate text perturbations independent of the image-modal attack. At the multi-modal level, we design a novel iterative cross-search attack (ICSA) method to update adversarial image-text pairs periodically, starting with the outputs from the single-modal level.  We conduct extensive experiments to attack three widely-used VL pretrained models for six tasks on eight datasets. Experimental results show that the proposed VLATTACK framework achieves the highest attack success rates on all tasks compared with state-of-the-art baselines, which reveals a significant blind spot in the deployment of pre-trained VL models.</details>

### VOCE: Variational Optimization with Conservative Estimation for Offline Safe Reinforcement Learning

**Authors:** Jiayi Guan, Guang Chen, Jiaming Ji, Long Yang, ao zhou, Zhijun Li, changjun jiang

**<details><summary>Abstract**</summary>

Offline safe reinforcement learning (RL) algorithms promise to learn policies that satisfy safety constraints directly in offline datasets without interacting with the environment. This arrangement is particularly important in scenarios with high sampling costs and potential dangers, such as autonomous driving and robotics. However, the influence of safety constraints and out-of-distribution (OOD) actions have made it challenging for previous methods to achieve high reward returns while ensuring safety. In this work, we propose a Variational Optimization with Conservative Eestimation algorithm (VOCE) to solve the problem of optimizing safety policies in the offline dataset. Concretely, we reframe the problem of offline safe RL using probabilistic inference, which introduces variational distributions to make the optimization of policies more flexible. Subsequently, we utilize pessimistic estimation methods to estimate the Q-value of cost and reward, which mitigates the extrapolation errors induced by OOD actions. Finally, extensive experiments demonstrate that the VOCE algorithm achieves competitive performance across multiple experimental tasks, particularly outperforming state-of-the-art algorithms in terms of safety.</details>

### [Spotlight] VaRT: Variational Regression Trees

**Authors:** Sebastian Salazar

**<details><summary>Abstract**</summary>

Decision trees are a well-established tool in machine learning for classification and regression tasks. In this paper, we introduce a novel non-parametric Bayesian model that uses variational inference to approximate a posterior distribution over the space of stochastic decision trees. We evaluate the model's performance on 18 datasets and demonstrate its competitiveness with other state-of-the-art methods in regression tasks. We also explore its application to causal inference problems. We provide a fully vectorized implementation of our algorithm in PyTorch.</details>

### Variational Gaussian Processes with Decoupled Conditionals

**Authors:** Xinran Zhu, Kaiwen Wu, Natalie Maus, Jacob Gardner, David Bindel

**<details><summary>Abstract**</summary>

Variational Gaussian processes (GPs) approximate exact GP inference by using a small set of inducing points to form a sparse approximation of the true posterior, with the fidelity of the model increasing with additional inducing points. Although the approximation error in principle can be reduced through the use of more inducing points, this leads to scaling optimization challenges and computational complexity. To achieve scalability, inducing point methods typically introduce conditional independencies and then approximations to the training and test conditional distributions. In this paper, we consider an alternative approach to modifying the training and test conditionals, in which we make them more flexible. In particular, we investigate decoupling the parametric form of the predictive mean and covariance in the conditionals, and learn independent parameters for predictive mean and covariance. We derive new evidence lower bounds (ELBO) under these more flexible conditionals, and provide two concrete examples of applying the decoupled conditionals. Empirically, we find this additional flexibility leads to improved model performance on a variety of regression tasks and Bayesian optimization (BO) applications.</details>

### Versatile Energy-Based Probabilistic Models for High Energy Physics

**Authors:** Taoli Cheng, Aaron Courville

**<details><summary>Abstract**</summary>

As a classical generative modeling approach, energy-based models have the natural advantage of flexibility in the form of the energy function. Recently, energy-based models have achieved great success in modeling high-dimensional data in computer vision and natural language processing. In line with these advancements, we build a multi-purpose energy-based probabilistic model for High Energy Physics events at the Large Hadron Collider.  This framework builds on a powerful generative model and describes higher-order inter-particle interactions. It suits different encoding architectures and builds on implicit generation. As for applicational aspects, it can serve as a powerful parameterized event generator for physics simulation, a generic anomalous signal detector free from spurious correlations, and an augmented event classifier for particle identification.</details>

### Video Timeline Modeling For News Story Understanding

**Authors:** Meng Liu, Mingda Zhang, Jialu Liu, Hanjun Dai, Ming-Hsuan Yang, Shuiwang Ji, Zheyun Feng, Boqing Gong

**<details><summary>Abstract**</summary>

In this paper, we present a novel problem, namely video timeline modeling. Our objective is to create a video-associated timeline from a set of videos related to a specific topic, thereby facilitating the content and structure understanding of the story being told. This problem has significant potential in various real-world applications, for instance, news story summarization. To bootstrap research in this area, we curate a realistic benchmark dataset, YouTube-News-Timeline, consisting of over $12$k timelines and $300$k YouTube news videos. Additionally, we propose a set of quantitative metrics to comprehensively evaluate and compare methodologies. With such a testbed, we further develop and benchmark several deep learning approaches to tackling this problem. We anticipate that this exploratory work will pave the way for further research in video timeline modeling. The assets are available via https://github.com/google-research/google-research/tree/master/video_timeline_modeling.</details>

### VideoComposer: Compositional Video Synthesis with Motion Controllability

**Authors:** Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, Jingren Zhou

**<details><summary>Abstract**</summary>

The pursuit of controllability as a higher standard of visual content creation has yielded remarkable progress in customizable image synthesis. However, achieving controllable video synthesis remains challenging due to the large variation of temporal dynamics and the requirement of cross-frame temporal consistency. Based on the paradigm of compositional generation, this work presents VideoComposer that allows users to flexibly compose a video with textual conditions, spatial conditions, and more importantly temporal conditions. Specifically, considering the characteristic of video data, we introduce the motion vector from compressed videos as an explicit control signal to provide guidance regarding temporal dynamics. In addition, we develop a Spatio-Temporal Condition encoder (STC-encoder) that serves as a unified interface to effectively incorporate the spatial and temporal relations of sequential inputs, with which the model could make better use of temporal conditions and hence achieve higher inter-frame consistency. Extensive experimental results suggest that VideoComposer is able to control the spatial and temporal patterns simultaneously within a synthesized video in various forms, such as text description, sketch sequence, reference video, or even simply hand-crafted motions. The code and models are publicly available athttps://videocomposer.github.io.</details>

### Visual Instruction Inversion: Image Editing via Image Prompting

**Authors:** Thao Nguyen, Yuheng Li, Utkarsh Ojha, Yong Jae Lee

**<details><summary>Abstract**</summary>

Text-conditioned image editing has emerged as a powerful tool for editing images.However, in many situations, language can be ambiguous and ineffective in describing specific image edits.When faced with such challenges, visual prompts can be a more informative and intuitive way to convey ideas.We present a method for image editing via visual prompting.Given pairs of example that represent the "before" and "after" images of an edit, our goal is to learn a text-based editing direction that can be used to perform the same edit on new images.We leverage the rich, pretrained editing capabilities of text-to-image diffusion models by inverting visual prompts into editing instructions.Our results show that with just one example pair, we can achieve competitive results compared to state-of-the-art text-conditioned image editing frameworks.</details>

### [Oral] Visual Instruction Tuning

**Authors:** Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee

**Oral Presentation:** Th, Dec 14, 08:00 -- Oral 5D

**<details><summary>Abstract**</summary>

Instruction tuning large language models (LLMs) using machine-generated instruction-following data has been shown to improve zero-shot capabilities on new tasks, but the idea is less explored in the multimodal field. We present the first attempt to use language-only GPT-4 to generate multimodal language-image instruction-following data. By instruction tuning on such generated data, we introduce LLaVA: Large Language and Vision Assistant, an end-to-end trained large multimodal model that connects a vision encoder and an LLM for general-purpose visual and language understanding. To facilitate future research on visual instruction following, we construct two evaluation benchmarks with diverse and challenging application-oriented tasks. Our experiments show that LLaVA demonstrates impressive multimodal chat abilities, sometimes exhibiting the behaviors of multimodal GPT-4 on unseen images/instructions, and yields a 85.1% relative score compared with GPT-4 on a synthetic multimodal instruction-following dataset. When fine-tuned on Science QA, the synergy of LLaVA and GPT-4 achieves a new state-of-the-art accuracy of 92.53%. We make GPT-4 generated visual instruction tuning data, our model, and code publicly available.</details>

### Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale

**Authors:** Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, Wei-Ning Hsu

**<details><summary>Abstract**</summary>

Large-scale generative models such as GPT and DALL-E have revolutionized the research community. These models not only generate high fidelity outputs, but are also generalists which can solve tasks not explicitly taught. In contrast, speech generative models are still primitive in terms of scale and task generalization. In this paper, we present Voicebox, the most versatile text-guided generative model for speech at scale. Voicebox is a non-autoregressive flow-matching model trained to infill speech, given audio context and text, trained on over 50K hours of speech that are not filtered or enhanced. Similar to GPT, Voicebox can perform many different tasks through in-context learning, but is more flexible as it can also condition on future context. Voicebox can be used for mono or cross-lingual zero-shot text-to-speech synthesis, noise removal, content editing, style conversion, and diverse sample generation. In particular, Voicebox outperforms the state-of-the-art zero-shot TTS model VALL-E on both intelligibility (5.9\% vs 1.9\% word error rates) and audio similarity (0.580 vs 0.681) while being up to 20 times faster. Audio samples can be found in \url{https://voicebox.metademolab.com}.</details>

### Wasserstein Gradient Flows for Optimizing Gaussian Mixture Policies

**Authors:** Hanna Ziesche, Leonel Rozo

**<details><summary>Abstract**</summary>

Robots often rely on a repertoire of previously-learned motion policies for performing tasks of diverse complexities.  When facing unseen task conditions or when new task requirements arise, robots must adapt their motion policies accordingly. In this context, policy optimization is the \emph{de facto} paradigm to adapt robot policies as a function of task-specific objectives.  Most commonly-used motion policies carry particular structures that are often overlooked in policy optimization algorithms.  We instead propose to leverage the structure of probabilistic policies by casting the policy optimization as an optimal transport problem. Specifically, we focus on robot motion policies that build on Gaussian mixture models (GMMs) and formulate the policy optimization as a Wassertein gradient flow over the GMMs space. This naturally allows us to constrain the policy updates via the $L^2$-Wasserstein distance between GMMs to enhance the stability of the policy optimization process. Furthermore, we leverage the geometry of the Bures-Wasserstein manifold to optimize the Gaussian distributions of the GMM policy via Riemannian optimization. We evaluate our approach on common robotic settings: Reaching motions, collision-avoidance behaviors, and multi-goal tasks. Our results show that our method outperforms common policy optimization baselines in terms of task success rate and low-variance solutions.</details>

### [Spotlight] Wasserstein Quantum Monte Carlo: A Novel Approach for Solving the Quantum Many-Body Schrödinger Equation

**Authors:** Kirill Neklyudov, Jannes Nys, Luca Thiede, Juan Carrasquilla, Qiang Liu, Max Welling, Alireza Makhzani

**<details><summary>Abstract**</summary>

Solving the quantum many-body Schrödinger equation is a fundamental and challenging problem in the fields of quantum physics, quantum chemistry, and material sciences. One of the common computational approaches to this problem is Quantum Variational Monte Carlo (QVMC), in which ground-state solutions are obtained by minimizing the energy of the system within a restricted family of parameterized wave functions. Deep learning methods partially address the limitations of traditional QVMC by representing a rich family of wave functions in terms of neural networks. However, the optimization objective in QVMC remains notoriously hard to minimize and requires second-order optimization methods such as natural gradient. In this paper, we first reformulate energy functional minimization in the space of Born distributions corresponding to particle-permutation (anti-)symmetric wave functions, rather than the space of wave functions. We then interpret QVMC as the Fisher--Rao gradient flow in this distributional space, followed by a projection step onto the variational manifold. This perspective provides us with a principled framework to derive new QMC algorithms, by endowing the distributional space with better metrics, and following the projected gradient flow induced by those metrics. More specifically, we propose ``Wasserstein Quantum Monte Carlo'' (WQMC), which uses the gradient flow induced by the Wasserstein metrics, rather than Fisher--Rao metric, and corresponds to *transporting* the probability mass, rather than *teleporting* it. We demonstrate empirically that the dynamics of WQMC results in faster convergence to the ground state of molecular systems.</details>

### What Makes Good Examples for Visual In-Context Learning?

**Authors:** Yuanhan Zhang, Kaiyang Zhou, Ziwei Liu

**<details><summary>Abstract**</summary>

Large vision models with billions of parameters and trained on broad data have great potential in numerous downstream applications. However, these models are typically difficult to adapt due to their large parameter size and sometimes lack of accesss to their weights---entities able to develop large vision models often provide APIs only. In this paper, we study how to better utilize large vision models through the lens of in-context learning, a concept that has been well-known in natural language processing but has only been studied very recently in computer vision. In-context learning refers to the ability to perform inference on tasks never seen during training by simply conditioning on in-context examples (i.e., input-output pairs) without updating any internal model parameters. To demystify in-context learning in computer vision, we conduct an extensive research and identify a critical problem: downstream performance is highly sensitivie to the choice of visual in-context examples. To address this problem, we propose a prompt retrieval framework specifically for large vision models, allowing the selection of in-context examples to be fully automated. Concretely, we provide two implementations: (i) an unsupervised prompt retrieval method based on nearest example search using an off-the-shelf model, and (ii) a supervised prompt retrieval method, which trains a neural network to choose examples that directly maximize in-context learning performance. Both methods do not require access to the internal weights of large vision models. Our results demonstrate that our methods can bring non-trivial improvements to visual in-context learning in comparison to the commonly-used random selection. Code and models will be released.</details>

### What Planning Problems Can A Relational Neural Network Solve?

**Authors:** Jiayuan Mao, Tomás Lozano-Pérez, Josh Tenenbaum, Leslie Kaelbling

**<details><summary>Abstract**</summary>

Goal-conditioned policies are generally understood to be "feed-forward" circuits, in the form of neural networks that map from the current state and the goal specification to the next action to take. However, under what circumstances such a policy can be learned and how efficient the policy will be are not well understood. In this paper, we present a circuit complexity analysis for relational neural networks (such as graph neural networks and transformers) representing policies for planning problems, by drawing connections with serialized goal regression search (S-GRS). We show that there are three general classes of planning problems, in terms of the growth of circuit width and depth as a function of the number of objects and planning horizon, providing constructive proofs. We also illustrate the utility of this analysis for designing neural networks for policy learning.</details>

### When Do Graph Neural Networks Help with Node Classification? Investigating the Homophily Principle on Node Distinguishability

**Authors:** Sitao Luan, Chenqing Hua, Minkai Xu, Qincheng Lu, Jiaqi Zhu, Xiao-Wen Chang, Jie Fu, Jure Leskovec, Doina Precup

**<details><summary>Abstract**</summary>

Homophily principle, i.e., nodes with the same labels are more likely to be connected, has been believed to be the main reason for the performance superiority of Graph Neural Networks (GNNs) over Neural Networks on node classification tasks. Recent research suggests that, even in the absence of homophily, the advantage of GNNs still exists as long as nodes from the same class share similar neighborhood patterns. However, this argument only considers intra-class Node Distinguishability (ND) but neglects inter-class ND, which provides incomplete understanding of homophily on GNNs. In this paper, we first demonstrate such deficiency with examples and argue that an ideal situation for ND is to have smaller intra-class ND than inter-class ND. To formulate this idea and study ND deeply, we propose Contextual Stochastic Block Model for Homophily (CSBM-H) and define two metrics, Probabilistic Bayes Error (PBE) and negative generalized Jeffreys divergence, to quantify ND. With the metrics, we visualize and analyze how graph filters, node degree distributions and class variances influence ND, and investigate the combined effect of intra- and inter-class ND. Besides, we discovered the mid-homophily pitfall, which occurs widely in graph datasets. Furthermore, we verified that, in real-work tasks, the superiority of GNNs is indeed closely related to both intra- and inter-class ND regardless of homophily levels. Grounded in this observation, we propose a new hypothesis-testing based performance metric beyond homophily, which is non-linear, feature-based and can provide statistical threshold value for GNNs' the superiority. Experiments indicate that it is significantly more effective than the existing homophily metrics on revealing the advantage and disadvantage of graph-aware modes on both synthetic and benchmark real-world datasets.</details>

### When are ensembles really effective?

**Authors:** Ryan Theisen, Hyunsuk Kim, Yaoqing Yang, Liam Hodgkinson, Michael Mahoney

**<details><summary>Abstract**</summary>

Ensembling has a long history in statistical data analysis, with many impactful applications. However, in many modern machine learning settings, the benefits of ensembling are less ubiquitous and less obvious. We study, both theoretically and empirically, the fundamental question of when ensembling yields significant performance improvements in classification tasks. Theoretically, we prove new results relating the \emph{ensemble improvement rate} (a measure of how much ensembling decreases the error rate versus a single model, on a relative scale) to the \emph{disagreement-error ratio}. We show that ensembling improves performance significantly whenever the disagreement rate is large relative to the average error rate; and that, conversely, one classifier is often enough whenever the disagreement rate is low relative to the average error rate. On the way to proving these results, we derive, under a mild condition called \emph{competence}, improved upper and lower bounds on the average test error rate of the majority vote classifier.To complement this theory, we study ensembling empirically in a variety of settings, verifying the predictions made by our theory, and identifying practical scenarios where ensembling does and does not result in large performance improvements. Perhaps most notably, we demonstrate a distinct difference in behavior between interpolating models (popular in current practice) and non-interpolating models (such as tree-based methods, where ensembling is popular), demonstrating that ensembling helps considerably more in the latter case than in the former.</details>

### Worst-case Performance of Popular Approximate Nearest Neighbor Search Implementations: Guarantees and Limitations

**Authors:** Piotr Indyk, Haike Xu

**<details><summary>Abstract**</summary>

Graph-based approaches to nearest neighbor search are popular and powerful tools for handling large datasets in practice, but they have limited theoretical guarantees. We study the worst-case performance of recent graph-based approximate nearest neighbor search algorithms, such as HNSW, NSG and DiskANN. For DiskANN, we show that its "slow preprocessing'' version provably supports approximate nearest neighbor search query with constant approximation ratio and poly-logarithmic query time, on data sets with bounded "intrinsic'' dimension. For the other data structure variants studied, including DiskANN with "fast preprocessing'',  HNSW and NSG,  we present a family of instances on which the empirical query time required to achieve a "reasonable'' accuracy is linear in instance size. For example, for DiskANN, we show that the query procedure can take at least  $0.1 n$ steps on instances of size $n$ before it encounters any of the $5$ nearest neighbors of the query.</details>

### xTrimoGene: An Efficient and Scalable Representation Learner for Single-Cell RNA-Seq Data

**Authors:** Jing Gong, Minsheng Hao, Xingyi Cheng, Xin Zeng, Chiming Liu, Jianzhu Ma, Xuegong Zhang, Taifeng Wang, Le Song

**<details><summary>Abstract**</summary>

Advances in high-throughput sequencing technology have led to significant progress in measuring gene expressions at the single-cell level. The amount of publicly available single-cell RNA-seq (scRNA-seq) data is already surpassing 50M records for humans with each record measuring 20,000 genes. This highlights the need for unsupervised representation learning to fully ingest these data, yet classical transformer architectures are prohibitive to train on such data in terms of both computation and memory. To address this challenge, we propose a novel asymmetric encoder-decoder transformer for scRNA-seq data, called xTrimoGene$^\alpha$ (or xTrimoGene for short), which leverages the sparse characteristic of the data to scale up the pre-training. This scalable design of xTrimoGene reduces FLOPs by one to two orders of magnitude compared to classical transformers while maintaining high accuracy, enabling us to train the largest transformer models over the largest scRNA-seq dataset today. Our experiments also show that the performance of xTrimoGene improves as we scale up the model sizes, and it also leads to SOTA performance over various downstream tasks, such as cell type annotation, perturb-seq effect prediction, and drug combination prediction. xTrimoGene model is now available for use as a service via the following link: https://api.biomap.com/xTrimoGene/apply.</details>

