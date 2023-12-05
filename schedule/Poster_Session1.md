## Poster Session 1: Tuesday, Dec 12, 08:45 CT### (S)GD over Diagonal Linear Networks: Implicit bias, Large Stepsizes and Edge of Stability

**Authors:** Mathieu Even, Scott Pesme, Suriya Gunasekar, Nicolas Flammarion

**<details><summary>Abstract**</summary>

In this paper, we investigate the impact of stochasticity and large stepsizes on the implicit regularisation of gradient descent (GD) and stochastic gradient descent (SGD) over $2$-layer diagonal linear networks. We prove the convergence of GD and SGD with macroscopic stepsizes in an overparametrised regression setting and characterise their solutions through an implicit regularisation problem. Our crisp characterisation leads to qualitative insights about the impact of stochasticity and stepsizes on the recovered solution. Specifically, we show that large stepsizes consistently benefit SGD for sparse regression problems, while they can hinder the recovery of sparse solutions for GD. These effects are magnified for stepsizes in a tight window just below the divergence threshold, in the ``edge of stability'' regime. Our findings are supported by experimental results.</details>

### 3D Indoor Instance Segmentation in an Open-World

**Authors:** Mohamed El Amine Boudjoghra, Salwa Al Khatib, Jean Lahoud, Hisham Cholakkal, Rao Anwer, Salman Khan, Fahad Shahbaz Khan

**<details><summary>Abstract**</summary>

Existing 3D instance segmentation methods typically assume that all semantic classes to be segmented would be available during training and only seen categories are segmented at inference. We argue that such a closed-world assumption is restrictive and explore for the first time 3D indoor instance segmentation in an open-world setting, where the model is allowed to distinguish a set of known classes as well as identify an unknown object as unknown and then later incrementally learning the semantic category of the unknown when the corresponding category labels are available. To this end, we introduce an open-world 3D indoor instance segmentation method, where an auto-labeling scheme is employed to produce pseudo-labels during training and induce separation to separate known and unknown category labels. We further improve the pseudo-labels quality at inference by adjusting the unknown class probability based on the objectness score distribution. We also introduce carefully curated open-world splits leveraging realistic scenarios based on inherent object distribution, region-based indoor scene exploration and randomness aspect of open-world classes. Extensive experiments reveal the efficacy of the proposed contributions leading to promising open-world 3D instance segmentation performance. Code and splits are available at: https://github.com/aminebdj/3D-OWIS.</details>

### A Batch-to-Online Transformation under Random-Order Model

**Authors:** Jing Dong, Yuichi Yoshida

**<details><summary>Abstract**</summary>

We introduce a transformation framework that can be utilized to develop online algorithms with low $\epsilon$-approximate regret in the random-order model from offline approximation algorithms. We first give a general reduction theorem that transforms an offline approximation algorithm with low average sensitivity to an online algorithm with low $\epsilon$-approximate regret. We then demonstrate that offline approximation algorithms can be transformed into a low-sensitivity version using a coreset construction method. To showcase the versatility of our approach, we apply it to various problems, including online $(k,z)$-clustering, online matrix approximation, and online regression, and successfully achieve polylogarithmic $\epsilon$-approximate regret for each problem. Moreover, we show that in all three cases, our algorithm also enjoys low inconsistency, which may be desired in some online applications.</details>

### A Comprehensive Study on Text-attributed Graphs: Benchmarking and Rethinking

**Authors:** Hao Yan, Chaozhuo Li, Ruosong Long, Chao Yan, Jianan Zhao, Wenwen Zhuang, Jun Yin, Peiyan Zhang, Weihao Han, Hao Sun, Weiwei Deng, Qi Zhang, Lichao Sun, Xing Xie, Senzhang Wang

**<details><summary>Abstract**</summary>

Text-attributed graphs (TAGs) are prevalent in various real-world scenarios, where each node is associated with a text description. The cornerstone of representation learning on TAGs lies in the seamless integration of textual semantics within individual nodes and the topological connections across nodes. Recent advancements in pre-trained language models (PLMs) and graph neural networks (GNNs) have facilitated effective learning on TAGs, garnering increased research interest. However, the absence of meaningful benchmark datasets and standardized evaluation procedures for TAGs has impeded progress in this field. In this paper, we propose CS-TAG, a comprehensive and diverse collection of challenging benchmark datasets for TAGs. The CS-TAG datasets are notably large in scale and encompass a wide range of domains, spanning from citation networks to purchase graphs. In addition to building the datasets, we conduct extensive benchmark experiments over CS-TAG with various learning paradigms, including PLMs, GNNs, PLM-GNN co-training methods, and the proposed novel topological pre-training of language models. In a nutshell, we provide an overview of the CS-TAG datasets, standardized evaluation procedures, and present baseline experiments. The entire CS-TAG project is publicly accessible at \url{https://github.com/sktsherlock/TAG-Benchmark}.</details>

### A Fast and Accurate Estimator for Large Scale Linear Model via Data Averaging

**Authors:** Rui Wang, Yanyan Ouyang, Yu Panpan, Wangli Xu

**<details><summary>Abstract**</summary>

This work is concerned with the estimation problem of linear model when thesample size is extremely large and the data dimension can vary with the samplesize. In this setting, the least square estimator based on the full data is not feasiblewith limited computational resources. Many existing methods for this problem arebased on the sketching technique which uses the sketched data to perform leastsquare estimation. We derive fine-grained lower bounds of the conditional meansquared error for sketching methods. For sampling methods, our lower boundprovides an attainable optimal convergence rate. Our result implies that when thedimension is large, there is hardly a sampling method can have a faster convergencerate than the uniform sampling method. To achieve a better statistical performance,we propose a new sketching method based on data averaging. The proposedmethod reduces the original data to a few averaged observations. These averagedobservations still satisfy the linear model and are used to estimate the regressioncoefficients. The asymptotic behavior of the proposed estimation procedure isstudied. Our theoretical results show that the proposed method can achieve afaster convergence rate than the optimal convergence rate for sampling methods.Theoretical and numerical results show that the proposed estimator has goodstatistical performance as well as low computational cost.</details>

### A Guide Through the Zoo of Biased SGD

**Authors:** Yury Demidovich, Grigory Malinovsky, Igor Sokolov, Peter Richtarik

**<details><summary>Abstract**</summary>

Stochastic Gradient Descent (SGD) is arguably the most important single algorithm in modern machine learning. Although SGD with unbiased gradient estimators has been studied extensively over at least half a century, SGD variants relying on biased estimators are rare. Nevertheless, there has been an increased interest in this topic in recent years. However, existing literature on SGD with biased estimators lacks coherence since each new paper relies on a different set of assumptions, without any clear understanding of how they are connected, which may lead to confusion. We address this gap by establishing connections among the existing assumptions, and presenting a comprehensive map of the underlying relationships. Additionally, we introduce a new set of assumptions that is provably weaker than all previous assumptions, and use it to present a thorough analysis of BiasedSGD in both convex and non-convex settings, offering advantages over previous results. We also provide examples where biased estimators outperform their unbiased counterparts or where unbiased versions are simply not available. Finally, we demonstrate the effectiveness of our framework through experimental results that validate our theoretical findings.</details>

### A High-Resolution Dataset for Instance Detection with Multi-View Object Capture

**Authors:** QIANQIAN SHEN, Yunhan Zhao, Nahyun Kwon, Jeeeun Kim, Yanan Li, Shu Kong

**<details><summary>Abstract**</summary>

Instance detection (InsDet) is a long-lasting problem in robotics and computer vision, aiming to detect object instances (predefined by some visual examples) in a cluttered scene. Despite its practical significance, its advancement is overshadowed by Object Detection, which aims to detect objects belonging to some predefined classes. One major reason is that current InsDet datasets are too small in scale by today's standards. For example, the popular InsDet dataset GMU (published in 2016) has only 23 instances, far less than  COCO (80 classes), a well-known object detection dataset published in 2014. We are motivated to introduce a new InsDet dataset and protocol. First, we define a realistic setup for InsDet: training data consists of multi-view instance captures, along with diverse scene images allowing synthesizing training images by pasting instance images on them with free box annotations. Second, we release a real-world database, which contains multi-view capture of 100 object instances, and high-resolution (6k$\times$8k) testing images. Third, we extensively study baseline methods for InsDet on our dataset, analyze their performance and suggest future work. Somewhat surprisingly, using the off-the-shelf  class-agnostic segmentation model (Segment Anything Model, SAM) and the self-supervised feature representation DINOv2 performs the best, achieving $>$10 AP better than end-to-end trained InsDet models that repurpose object detectors (e.g., FasterRCNN and RetinaNet).</details>

### A Multi-modal Global Instance Tracking Benchmark (MGIT): Better Locating Target in Complex Spatio-temporal and Causal Relationship

**Authors:** Shiyu Hu, Dailing Zhang, wu meiqi, Xiaokun Feng, Xuchen Li, Xin Zhao, Kaiqi Huang

**<details><summary>Abstract**</summary>

Tracking an arbitrary moving target in a video sequence is the foundation for high-level tasks like video understanding. Although existing visual-based trackers have demonstrated good tracking capabilities in short video sequences, they always perform poorly in complex environments, as represented by the recently proposed global instance tracking task, which consists of longer videos with more complicated narrative content. Recently, several works have introduced natural language into object tracking, desiring to address the limitations of relying only on a single visual modality. However, these selected videos are still short sequences with uncomplicated spatio-temporal and causal relationships, and the provided semantic descriptions are too simple to characterize video content.To address these issues, we (1) first propose a new multi-modal global instance tracking benchmark named MGIT. It consists of 150 long video sequences with a total of 2.03 million frames, aiming to fully represent the complex spatio-temporal and causal relationships coupled in longer narrative content. (2) Each video sequence is annotated with three semantic grains (i.e., action, activity, and story) to model the progressive process of human cognition. We expect this multi-granular annotation strategy can provide a favorable environment for multi-modal object tracking research and long video understanding. (3) Besides, we execute comparative experiments on existing multi-modal object tracking benchmarks, which not only explore the impact of different annotation methods, but also validate that our annotation method is a feasible solution for coupling human understanding into semantic labels. (4) Additionally, we conduct detailed experimental analyses on MGIT, and hope the explored performance bottlenecks of existing algorithms can support further research in multi-modal object tracking. The proposed benchmark, experimental results, and toolkit will be released gradually on http://videocube.aitestunion.com/.</details>

### A Path to Simpler Models Starts With Noise

**Authors:** Lesia Semenova, Harry Chen, Ronald Parr, Cynthia Rudin

**<details><summary>Abstract**</summary>

The Rashomon set is the set of models that perform approximately equally well on a given dataset, and the Rashomon ratio is the fraction of all models in a given hypothesis space that are in the Rashomon set. Rashomon ratios are often large for tabular datasets in criminal justice, healthcare, lending, education, and in other areas, which has practical implications about whether simpler models can attain the same level of accuracy as more complex models. An open question is why Rashomon ratios often tend to be large. In this work, we propose and study a mechanism of the data generation process, coupled with choices usually made by the analyst during the learning process, that determines the size of the Rashomon ratio. Specifically, we demonstrate that noisier datasets lead to larger Rashomon ratios through the way that practitioners train models. Additionally, we introduce a measure called pattern diversity, which captures the average difference in predictions between distinct classification patterns in the Rashomon set, and motivate why it tends to increase with label noise. Our results explain a key aspect of why simpler models often tend to perform as well as black box models on complex, noisier datasets.</details>

### A Performance-Driven Benchmark for Feature Selection in Tabular Deep Learning

**Authors:** Valeriia Cherepanova, Roman Levin, Gowthami Somepalli, Jonas Geiping, C. Bayan Bruss, Andrew Wilson, Tom Goldstein, Micah Goldblum

**<details><summary>Abstract**</summary>

Academic tabular benchmarks often contain small sets of curated features. In contrast, data scientists typically collect as many features as possible into their datasets, and even engineer new features from existing ones. To prevent over-fitting in subsequent downstream modeling, practitioners commonly use automated feature selection methods that identify a reduced subset of informative features. Existing benchmarks for tabular feature selection consider classical downstream models, toy synthetic datasets, or do not evaluate feature selectors on the basis of downstream performance. We construct a challenging feature selection benchmark evaluated on downstream neural networks including transformers, using real datasets and multiple methods for generating extraneous features. We also propose an input-gradient-based analogue of LASSO for neural networks that outperforms classical feature selection methods on challenging problems such as selecting from corrupted or second-order features.</details>

### A Theory of Link Prediction via Relational Weisfeiler-Leman on Knowledge Graphs

**Authors:** Xingyue Huang, Miguel Romero, Ismail Ceylan, Pablo Barceló

**<details><summary>Abstract**</summary>

Graph neural networks are prominent models for representation learning over graph-structured data. While the capabilities and limitations of these models are well-understood for simple graphs, our understanding remains incomplete in the context of knowledge graphs. Our goal is to provide a systematic understanding of the landscape of graph neural networks for knowledge graphs pertaining to the prominent task of link prediction. Our analysis entails a unifying perspective on seemingly unrelated models and unlocks a series of other models. The expressive power of various models is characterized via a corresponding relational Weisfeiler-Leman algorithm. This analysis is extended to provide a precise logical characterization of the class of functions captured by a class of graph neural networks. The theoretical findings presented in this paper explain the benefits of some widely employed practical design choices, which are validated empirically.</details>

### A Theory of Unsupervised Translation Motivated by Understanding Animal Communication

**Authors:** Shafi Goldwasser, David Gruber, Adam Tauman Kalai, Orr Paradise

**<details><summary>Abstract**</summary>

Neural networks are capable of translating between languages—in some cases even between two languages where there is little or no access to parallel translations, in what is known as Unsupervised Machine Translation (UMT). Given this progress, it is intriguing to ask whether machine learning tools can ultimately enable understanding animal communication, particularly that of highly intelligentanimals. We propose a theoretical framework for analyzing UMT when no parallel translations are available and when it cannot be assumed that the source and target corpora address related subject domains or posses similar linguistic structure. Weexemplify this theory with two stylized models of language, for which our framework provides bounds on necessary sample complexity; the bounds are formally proven and experimentally verified on synthetic data. These bounds show that the error rates are inversely related to the language complexity and amount of common ground. This suggests that unsupervised translation of animal communication may be feasible if the communication system is sufficiently complex.</details>

### [Oral] A U-turn on Double Descent: Rethinking Parameter Counting in Statistical Learning

**Authors:** Alicia Curth, Alan Jeffares, Mihaela van der Schaar

**Oral Presentation:** Tu, Dec 12, 08:30 -- Oral 1D

**<details><summary>Abstract**</summary>

Conventional statistical wisdom established a well-understood relationship between model complexity and prediction error, typically presented as a _U-shaped curve_ reflecting a transition between under- and overfitting regimes. However, motivated by the success of overparametrized neural networks, recent influential work has suggested this theory to be generally incomplete, introducing an additional regime that exhibits a second descent in test error as the parameter count $p$ grows past sample size $n$  -- a  phenomenon dubbed  _double descent_. While most attention has naturally been given to the deep-learning setting, double descent was shown to emerge more generally across non-neural models: known cases include _linear regression, trees, and boosting_. In this work, we take a closer look at the evidence surrounding these more classical statistical machine learning methods and challenge the claim that observed cases of  double descent truly extend the limits of a traditional U-shaped complexity-generalization curve therein. We show that once careful consideration is given to _what is being plotted_ on the x-axes of their double descent plots, it becomes apparent that there are implicitly multiple, distinct complexity axes along which the parameter count grows. We demonstrate that the second descent appears exactly (and _only_) when and where the transition between these underlying axes occurs, and that its location is thus _not_ inherently tied to the interpolation threshold $p=n$. We then gain further insight by adopting a classical nonparametric statistics perspective. We interpret the investigated methods as _smoothers_ and propose a generalized measure for the _effective_ number of parameters they use _on unseen examples_, using which we find that their apparent double descent curves do indeed fold back into more traditional convex shapes -- providing a resolution to the ostensible tension between double descent and traditional statistical intuition.</details>

### A Unified Approach for Maximizing Continuous DR-submodular Functions

**Authors:** Mohammad Pedramfar, Christopher Quinn, Vaneet Aggarwal

**<details><summary>Abstract**</summary>

This paper presents a unified approach for maximizing continuous DR-submodular functions that encompasses a range of settings and oracle access types. Our approach includes a Frank-Wolfe type offline algorithm for both monotone and non-monotone functions, with different restrictions on the general convex set. We consider settings where the oracle provides access to either the gradient of the function or only the function value, and where the oracle access is either deterministic or stochastic. We determine the number of required oracle accesses in all cases. Our approach gives new/improved results for nine out of the sixteen considered cases, avoids computationally expensive projections in three cases, with the proposed framework matching performance of state-of-the-art approaches in the remaining four cases. Notably, our approach for the stochastic function value-based oracle enables the first regret bounds with bandit feedback for stochastic DR-submodular functions.</details>

### A Unified Discretization Framework for Differential Equation Approach with Lyapunov Arguments for Convex Optimization

**Authors:** Kansei Ushiyama, Shun Sato, Takayasu Matsuo

**<details><summary>Abstract**</summary>

The differential equation (DE) approach for convex optimization, which relates optimization methods to specific continuous DEs with rate-revealing Lyapunov functionals, has gained increasing interest since the seminal paper by Su--Boyd--Candès (2014).However, the approach still lacks a crucial component to make it truly useful: there is no general, consistent way to transition back to discrete optimization methods. Consequently, even if we derive insights from continuous DEs, we still need to perform individualized and tedious calculations for the analysis of each method.This paper aims to bridge this gap by introducing a new concept called  ``weak discrete gradient'' (wDG), which consolidates the conditions required for discrete versions of gradients in the DE approach arguments.We then define abstract optimization methods using wDG and provide abstract convergence theories that parallel those in continuous DEs.We demonstrate that many typical optimization methods and their convergence rates can be derived as special cases of this abstract theory.The proposed unified discretization framework for the differential equation approach to convex optimization provides an easy environment for developing new optimization methods and achieving competitive convergence rates with state-of-the-art methods, such as Nesterov's accelerated gradient.</details>

### A Unifying Perspective on Multi-Calibration: Game Dynamics for Multi-Objective Learning

**Authors:** Nika Haghtalab, Michael Jordan, Eric Zhao

**<details><summary>Abstract**</summary>

We provide a unifying framework for the design and analysis of multi-calibrated predictors. By placing the multi-calibration problem in the general setting of multi-objective learning---where learning guarantees must hold simultaneously over a set of distributions and loss functions---we exploit connections to game dynamics to achieve state-of-the-art guarantees for a diverse set of multi-calibration learning problems. In addition to shedding light on existing multi-calibration guarantees and greatly simplifying their analysis, our approach also yields improved guarantees, such as error tolerances that scale with the square-root of group size versus the constant tolerances guaranteed by prior works, and improving the complexity of $k$-class multi-calibration by an exponential factor of $k$ versus Gopalan et al.. Beyond multi-calibration, we use these game dynamics to address emerging considerations in the study of group fairness and multi-distribution learning.</details>

### A unified framework for information-theoretic generalization bounds

**Authors:** Yifeng Chu, Maxim Raginsky

**<details><summary>Abstract**</summary>

This paper presents a general methodology for deriving information-theoretic generalization bounds for learning algorithms. The main technical tool is a probabilistic decorrelation lemma based on a change of measure and a relaxation of Young's inequality in $L_{\psi_p}$ Orlicz spaces. Using the decorrelation lemma in combination with other techniques, such as symmetrization, couplings, and chaining in the space of probability measures, we obtain new upper bounds on the generalization error, both in expectation and in high probability, and recover as special cases many of the existing generalization bounds, including the ones based on mutual information, conditional mutual information, stochastic chaining, and PAC-Bayes inequalities. In addition, the Fernique--Talagrand upper bound on the expected supremum of a subgaussian process emerges as a special case.</details>

### ALGO: Synthesizing Algorithmic Programs with Generated Oracle Verifiers

**Authors:** Kexun Zhang, Danqing Wang, Jingtao Xia, William Yang Wang, Lei Li

**<details><summary>Abstract**</summary>

Large language models (LLMs) excel at implementing code from functionality descriptions but struggle with algorithmic problems that require not only implementation but also identification of the suitable algorithm. Moreover, LLM-generated programs lack guaranteed correctness and require human verification. To address these challenges, we propose ALGO, a framework that synthesizes Algorithmic programs with LLM-Generated Oracles to guide the generation and verify their correctness. ALGO first generates a reference oracle by prompting an LLM to exhaustively enumerate all the combinations of relevant variables. This oracle is then utilized to guide an arbitrary search strategy in exploring the algorithm space and to verify the synthesized algorithms. Our study shows that the LLM-generatedoracles are correct for 88% of the cases. With the oracles as verifiers, ALGO can be integrated with any existing code generation model in a model-agnostic manner to enhance its performance. Experiments show that when equipped with ALGO, we achieve an 8× better one-submission pass rate over the Codex model and a 2.6× better one-submission pass rate over CodeT, the current state-of-the-art model on CodeContests. We can also get 1.3× better pass rate over the ChatGPT Code Interpreter on unseen problems. The problem set we used for testing, the prompts we used, the verifier and solution programs, and the test cases generated by ALGOare available at https://github.com/zkx06111/ALGO.</details>

### AND: Adversarial Neural Degradation for Learning Blind Image Super-Resolution

**Authors:** Fangzhou Luo, Xiaolin Wu, Yanhui Guo

**<details><summary>Abstract**</summary>

Learnt deep neural networks for image super-resolution fail easily if the assumed degradation model in training mismatches that of the real degradation source at the inference stage. Instead of attempting to exhaust all degradation variants in simulation, which is unwieldy and impractical, we propose a novel adversarial neural degradation (AND) model that can, when trained in conjunction with a deep restoration neural network under a minmax criterion, generate a wide range of highly nonlinear complex degradation effects without any explicit supervision. The AND model has a unique advantage over the current state of the art in that it can generalize much better to unseen degradation variants and hence deliver significantly improved restoration performance on real-world images.</details>

### ANTN: Bridging Autoregressive Neural Networks and Tensor Networks for Quantum Many-Body Simulation

**Authors:** Zhuo Chen, Laker Newhouse, Eddie Chen, Di Luo, Marin Soljacic

**<details><summary>Abstract**</summary>

Quantum many-body physics simulation has important impacts on understanding fundamental science and has applications to quantum materials design and quantum technology. However, due to the exponentially growing size of the Hilbert space with respect to the particle number, a direct simulation is intractable. While representing quantum states with tensor networks and neural networks are the two state-of-the-art methods for approximate simulations, each has its own limitations in terms of expressivity and inductive bias. To address these challenges, we develop a novel architecture, Autoregressive Neural TensorNet (ANTN), which bridges tensor networks and autoregressive neural networks. We show that Autoregressive Neural TensorNet parameterizes normalized wavefunctions, allows for exact sampling, generalizes the expressivity of tensor networks and autoregressive neural networks, and inherits a variety of symmetries from autoregressive neural networks. We demonstrate our approach on quantum state learning as well as finding the ground state of the challenging 2D $J_1$-$J_2$ Heisenberg model with different systems sizes and coupling parameters, outperforming both tensor networks and autoregressive neural networks. Our work opens up new opportunities for quantum many-body physics simulation, quantum technology design, and generative modeling in artificial intelligence.</details>

### [Spotlight] AbDiffuser: full-atom generation of in-vitro functioning antibodies

**Authors:** Karolis Martinkus, Jan Ludwiczak, WEI-CHING LIANG, Julien Lafrance-Vanasse, Isidro Hotzel, Arvind Rajpal, Yan Wu, Kyunghyun Cho, Richard Bonneau, Vladimir Gligorijevic, Andreas Loukas

**<details><summary>Abstract**</summary>

We introduce AbDiffuser, an equivariant and physics-informed diffusion model for the joint generation of antibody 3D structures and sequences. AbDiffuser is built on top of a new representation of protein structure, relies on a novel architecture for aligned proteins, and utilizes strong diffusion priors to improve the denoising process. Our approach improves protein diffusion by taking advantage of domain knowledge and physics-based constraints; handles sequence-length changes; and reduces memory complexity by an order of magnitude, enabling backbone and side chain generation. We validate AbDiffuser in silico and in vitro. Numerical experiments showcase the ability of AbDiffuser to generate antibodies that closely track the sequence and structural properties of a reference set. Laboratory experiments confirm that all 16 HER2 antibodies discovered were expressed at high levels and that 57.1% of the selected designs were tight binders.</details>

### AbdomenAtlas-8K: Annotating 8,000 CT Volumes for Multi-Organ Segmentation in Three Weeks

**Authors:** Chongyu Qu, Tiezheng Zhang, Hualin Qiao, jie liu, Yucheng Tang, Alan Yuille, Zongwei Zhou

**<details><summary>Abstract**</summary>

Annotating medical images, particularly for organ segmentation, is laborious and time-consuming. For example, annotating an abdominal organ requires an estimated rate of 30-60 minutes per CT volume based on the expertise of an annotator and the size, visibility, and complexity of the organ. Therefore, publicly available datasets for multi-organ segmentation are often limited in data size and organ diversity. This paper proposes an active learning procedure to expedite the annotation process for organ segmentation and creates the largest multi-organ dataset (by far) with the spleen, liver, kidneys, stomach, gallbladder, pancreas, aorta, and IVC annotated in 8,448 CT volumes, equating to 3.2 million slices. The conventional annotation methods would take an experienced annotator up to 1,600 weeks (or roughly 30.8 years) to complete this task. In contrast, our annotation procedure has accomplished this task in three weeks (based on an 8-hour workday, five days a week) while maintaining a similar or even better annotation quality. This achievement is attributed to three unique properties of our method: (1) label bias reduction using multiple pre-trained segmentation models, (2) effective error detection in the model predictions, and (3) attention guidance for annotators to make corrections on the most salient errors. Furthermore, we summarize the taxonomy of common errors made by AI algorithms and annotators. This allows for continuous improvement of AI and annotations, significantly reducing the annotation costs required to create large-scale datasets for a wider variety of medical imaging tasks. Code and dataset are available at https://github.com/MrGiovanni/AbdomenAtlas</details>

### [Oral] Abide by the law and follow the flow: conservation laws for gradient flows

**Authors:** Sibylle Marcotte, Remi Gribonval, Gabriel Peyré

**Oral Presentation:** Tu, Dec 12, 08:15 -- Oral 1D

**<details><summary>Abstract**</summary>

Understanding the geometric properties of gradient descent dynamics is a key ingredient in deciphering the recent success of very large machine learning models. A striking observation is that trained over-parameterized models retain some properties of the optimization initialization. This "implicit bias" is believed to be responsible for some favorable properties of the trained models and could explain their good generalization properties. The purpose of this article is threefold. First, we rigorously expose the definition and basic properties of "conservation laws", that define quantities conserved during gradient flows of a given model (e.g. of a ReLU network with a given architecture) with any training data and any loss. Then we explain how to find the maximal number of independent conservation lawsby performing finite-dimensional algebraic manipulations on the Lie algebra generated by the Jacobian of the model. Finally, we provide algorithms to: a) compute a family of polynomial laws; b) compute the maximal number of (not necessarily polynomial) independent conservation laws. We provide showcase examples that we fully work out theoretically. Besides, applying the two algorithms confirms for a number of ReLU network architectures that all known laws are recovered by the algorithm, and that there are no other independent laws. Such computational tools pave the way to understanding desirable properties of optimization initialization in large machine learning models.</details>

### Accelerated Zeroth-order Method for Non-Smooth Stochastic Convex Optimization Problem with Infinite Variance

**Authors:** Nikita Kornilov, Ohad Shamir, Aleksandr Lobanov, Darina Dvinskikh, Alexander Gasnikov, Innokentiy Shibaev, Eduard Gorbunov, Samuel Horváth

**<details><summary>Abstract**</summary>

In this paper, we consider non-smooth stochastic convex optimization with two function evaluations per round under infinite noise variance. In the classical setting when noise has finite variance, an optimal algorithm, built upon the batched accelerated gradient method, was proposed in (Gasnikov et. al., 2022). This optimality is defined in terms of iteration and oracle complexity, as well as the maximal admissible level of adversarial noise. However, the assumption of finite variance is burdensome and it might not hold in many practical scenarios. To address this, we demonstrate how to adapt a refined clipped version of the accelerated gradient (Stochastic Similar Triangles) method from (Sadiev et al., 2023) for a two-point zero-order oracle. This adaptation entails extending the batching technique to accommodate infinite variance — a non-trivial task that stands as a distinct contribution of this paper.</details>

### Accelerating Exploration with Unlabeled Prior Data

**Authors:** Qiyang Li, Jason Zhang, Dibya Ghosh, Amy Zhang, Sergey Levine

**<details><summary>Abstract**</summary>

Learning to solve tasks from a sparse reward signal is a major challenge for standard reinforcement learning (RL) algorithms. However, in the real world, agents rarely need to solve sparse reward tasks entirely from scratch. More often, we might possess prior experience to draw on that provides considerable guidance about which actions and outcomes are possible in the world, which we can use to explore more effectively for new tasks. In this work, we study how prior data without reward labels may be used to guide and accelerate exploration for an agent solving a new sparse reward task. We propose a simple approach that learns a reward model from online experience, labels the unlabeled prior data with optimistic rewards, and then uses it concurrently alongside the online data for downstream policy and critic optimization. This general formula leads to rapid exploration in several challenging sparse-reward domains where tabula rasa exploration is insufficient, including the AntMaze domain, Adroit hand manipulation domain, and a visual simulated robotic manipulation domain. Our results highlight the ease of incorporating unlabeled prior data into existing online RL algorithms, and the (perhaps surprising) effectiveness of doing so.</details>

### Accountability in Offline Reinforcement Learning: Explaining Decisions with a Corpus of Examples

**Authors:** Hao Sun, Alihan Hüyük, Daniel Jarrett, Mihaela van der Schaar

**<details><summary>Abstract**</summary>

Learning controllers with offline data in decision-making systems is an essential area of research due to its potential to reduce the risk of applications in real-world systems. However, in responsibility-sensitive settings such as healthcare, decision accountability is of paramount importance, yet has not been adequately addressed by the literature.This paper introduces the Accountable Offline Controller (AOC) that employs the offline dataset as the Decision Corpus and performs accountable control based on a tailored selection of examples, referred to as the Corpus Subset. AOC operates effectively in low-data scenarios, can be extended to the strictly offline imitation setting, and displays qualities of both conservation and adaptability.We assess AOC's performance in both simulated and real-world healthcare scenarios, emphasizing its capability to manage offline control tasks with high levels of performance while maintaining accountability.</details>

### AdaptSSR: Pre-training User Model with Augmentation-Adaptive Self-Supervised Ranking

**Authors:** Yang Yu, Qi Liu, Kai Zhang, Yuren Zhang, Chao Song, Min Hou, Yuqing Yuan, Zhihao Ye, ZAIXI ZHANG, Sanshi Lei Yu

**<details><summary>Abstract**</summary>

User modeling, which aims to capture users' characteristics or interests, heavily relies on task-specific labeled data and suffers from the data sparsity issue. Several recent studies tackled this problem by pre-training the user model on massive user behavior sequences with a contrastive learning task. Generally, these methods assume different views of the same behavior sequence constructed via data augmentation are semantically consistent, i.e., reflecting similar characteristics or interests of the user, and thus maximizing their agreement in the feature space. However, due to the diverse interests and heavy noise in user behaviors, existing augmentation methods tend to lose certain characteristics of the user or introduce noisy behaviors. Thus, forcing the user model to directly maximize the similarity between the augmented views may result in a negative transfer. To this end, we propose to replace the contrastive learning task with a new pretext task: Augmentation-Adaptive SelfSupervised Ranking (AdaptSSR), which alleviates the requirement of semantic consistency between the augmented views while pre-training a discriminative user model. Specifically, we adopt a multiple pairwise ranking loss which trains the user model to capture the similarity orders between the implicitly augmented view, the explicitly augmented view, and views from other users. We further employ an in-batch hard negative sampling strategy to facilitate model training. Moreover, considering the distinct impacts of data augmentation on different behavior sequences, we design an augmentation-adaptive fusion mechanism to automatically adjust the similarity order constraint applied to each sample based on the estimated similarity between the augmented views. Extensive experiments on both public and industrial datasets with six downstream tasks verify the effectiveness of AdaptSSR.</details>

### Adaptive Test-Time Personalization for Federated Learning

**Authors:** Wenxuan Bao, Tianxin Wei, Haohan Wang, Jingrui He

**<details><summary>Abstract**</summary>

Personalized federated learning algorithms have shown promising results in adapting models to various distribution shifts. However, most of these methods require labeled data on testing clients for personalization, which is usually unavailable in real-world scenarios. In this paper, we introduce a novel setting called test-time personalized federated learning (TTPFL), where clients locally adapt a global model in an unsupervised way without relying on any labeled data during test-time. While traditional test-time adaptation (TTA) can be used in this scenario, most of them inherently assume training data come from a single domain, while they come from multiple clients (source domains) with different distributions. Overlooking these domain interrelationships can result in suboptimal generalization. Moreover, most TTA algorithms are designed for a specific kind of distribution shift and lack the flexibility to handle multiple kinds of distribution shifts in FL. In this paper, we find that this lack of flexibility partially results from their pre-defining which modules to adapt in the model. To tackle this challenge, we propose a novel algorithm called ATP to adaptively learns the adaptation rates for each module in the model from distribution shifts among source domains. Theoretical analysis proves the strong generalization of ATP. Extensive experiments demonstrate its superiority in handling various distribution shifts including label shift, image corruptions, and domain shift, outperforming existing TTA methods across multiple datasets and model architectures. Our code is available at https://github.com/baowenxuan/ATP.</details>

### Add and Thin: Diffusion for Temporal Point Processes

**Authors:** David Lüdke, Marin Biloš, Oleksandr Shchur, Marten Lienen, Stephan Günnemann

**<details><summary>Abstract**</summary>

Autoregressive neural networks within the temporal point process (TPP) framework have become the standard for modeling continuous-time event data. Even though these models can expressively capture event sequences in a one-step-ahead fashion, they are inherently limited for long-term forecasting applications due to the accumulation of errors caused by their sequential nature. To overcome these limitations, we derive ADD-THIN, a principled probabilistic denoising diffusion model for TPPs that operates on entire event sequences. Unlike existing diffusion approaches, ADD-THIN naturally handles data with discrete and continuous components. In experiments on synthetic and real-world datasets, our model matches the state-of-the-art TPP models in density estimation and strongly outperforms them in forecasting.</details>

### Advancing Bayesian Optimization via Learning Correlated Latent Space

**Authors:** Seunghun Lee, Jaewon Chu, Sihyeon Kim, Juyeon Ko, Hyunwoo Kim

**<details><summary>Abstract**</summary>

Bayesian optimization is a powerful method for optimizing black-box functions with limited function evaluations. Recent works have shown that optimization in a latent space through deep generative models such as variational autoencoders leads to effective and efficient Bayesian optimization for structured or discrete data. However, as the optimization does not take place in the input space, it leads to an inherent gap that results in potentially suboptimal solutions. To alleviate the discrepancy, we propose Correlated latent space Bayesian Optimization (CoBO), which focuses on learning correlated latent spaces characterized by a strong correlation between the distances in the latent space and the distances within the objective function. Specifically, our method introduces Lipschitz regularization, loss weighting, and trust region recoordination to minimize the inherent gap around the promising areas. We demonstrate the effectiveness of our approach on several optimization tasks in discrete data, such as molecule design and arithmetic expression fitting, and achieve high performance within a small budget.</details>

### Adversarial Attacks on Online Learning to Rank with Click Feedback

**Authors:** Jinhang Zuo, Zhiyao Zhang, Zhiyong Wang, Shuai Li, Mohammad Hajiesmaili, Adam Wierman

**<details><summary>Abstract**</summary>

Online learning to rank (OLTR) is a sequential decision-making problem where a learning agent selects an ordered list of items and receives feedback through user clicks. Although potential attacks against OLTR algorithms may cause serious losses in real-world applications, there is limited knowledge about adversarial attacks on OLTR. This paper studies attack strategies against multiple variants of OLTR. Our first result provides an attack strategy against the UCB algorithm on classical stochastic bandits with binary feedback, which solves the key issues caused by bounded and discrete feedback that previous works cannot handle. Building on this result, we design attack algorithms against UCB-based OLTR algorithms in position-based and cascade models. Finally, we propose a general attack strategy against any algorithm under the general click model. Each attack algorithm manipulates the learning agent into choosing the target attack item $T-o(T)$ times, incurring a cumulative cost of $o(T)$. Experiments on synthetic and real data further validate the effectiveness of our proposed attack algorithms.</details>

### Adversarial Examples Exist in Two-Layer ReLU Networks for Low Dimensional Linear Subspaces

**Authors:** Odelia Melamed, Gilad Yehudai, Gal Vardi

**<details><summary>Abstract**</summary>

Despite a great deal of research, it is still not well-understood why trained neural networks are highly vulnerable to adversarial examples.In this work we focus on two-layer neural networks trained using data which lie on a low dimensional linear subspace.We show that standard gradient methods lead to non-robust neural networks, namely, networks which have large gradients in directions orthogonal to the data subspace, and are susceptible to small adversarial $L_2$-perturbations in these directions.Moreover, we show that decreasing the initialization scale of the training algorithm, or adding $L_2$ regularization, can make the trained network more robust to adversarial perturbations orthogonal to the data.</details>

### Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions

**Authors:** Lukas Gosch, Simon Geisler, Daniel Sturm, Bertrand Charpentier, Daniel Zügner, Stephan Günnemann

**<details><summary>Abstract**</summary>

Despite its success in the image domain, adversarial training did not (yet) stand out as an effective defense for Graph Neural Networks (GNNs) against graph structure perturbations. In the pursuit of fixing adversarial training  (1) we show and overcome fundamental theoretical as well as practical limitations of the adopted graph learning setting in prior work; (2) we reveal that flexible GNNs based on learnable graph diffusion are able to adjust to adversarial perturbations, while the learned message passing scheme is naturally interpretable; (3) we introduce the first attack for structure perturbations that, while targeting multiple nodes at once, is capable of handling global (graph-level) as well as local (node-level) constraints. Including these contributions, we demonstrate that adversarial training is a state-of-the-art defense against adversarial structure perturbations.</details>

### Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors

**Authors:** Tom Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, Marzyeh Ghassemi

**<details><summary>Abstract**</summary>

Deployed language models decay over time due to shifting inputs, changing user needs, or emergent world-knowledge gaps. When such problems are identified, we want to make targeted edits while avoiding expensive retraining. However, current model editors, which modify such behaviors of pre-trained models, degrade model performance quickly across multiple, sequential edits. We propose GRACE, a 	extit{lifelong} model editing method, which implements spot-fixes on streaming errors of a deployed model, ensuring minimal impact on unrelated inputs. GRACE writes new mappings into a pre-trained model's latent space, creating a discrete, local codebook of edits without altering model weights. This is the first method enabling thousands of sequential edits using only streaming errors. Our experiments on T5, BERT, and GPT models show GRACE's state-of-the-art performance in making and retaining edits, while generalizing to unseen inputs. Our code is available at [github.com/thartvigsen/grace](https://www.github.com/thartvigsen/grace}).</details>

### Agnostically Learning Single-Index Models using Omnipredictors

**Authors:** Aravind Gollakota, Parikshit Gopalan, Adam Klivans, Konstantinos Stavropoulos

**<details><summary>Abstract**</summary>

We give the first result for agnostically learning Single-Index Models (SIMs) with arbitrary monotone and Lipschitz activations. All prior work either held only in the realizable setting or required the activation to be known. Moreover, we only require the marginal to have bounded second moments, whereas all prior work required stronger distributional assumptions (such as anticoncentration or boundedness). Our algorithm is based on recent work by Gopalan et al. [2023] on Omniprediction using predictors satisfying calibrated multiaccuracy. Our analysis is simple and relies on the relationship between Bregman divergences (or matching losses) and $\ell_p$ distances. We also provide new guarantees for standard algorithms like GLMtron and logistic regression in the agnostic setting.</details>

### AirDelhi: Fine-Grained Spatio-Temporal Particulate Matter Dataset From Delhi For ML based Modeling

**Authors:** Sachin Chauhan, Zeel Bharatkumar Patel, Sayan Ranu, Rijurekha Sen, Nipun Batra

**<details><summary>Abstract**</summary>

Air pollution poses serious health concerns in developing countries, such as India, necessitating large-scale measurement for correlation analysis, policy recommendations, and informed decision-making. However, fine-grained data collection is costly. Specifically, static sensors for pollution measurement cost several thousand dollars per unit, leading to inadequate deployment and coverage. To complement the existing sparse static sensor network, we propose a mobile sensor network utilizing lower-cost PM2.5 sensors mounted on public buses in the Delhi-NCR region of India. Through this exercise, we introduce a novel dataset AirDelhi comprising PM2.5 and PM10 measurements. This dataset is made publicly available, at https://www.cse.iitd.ac.in/pollutiondata, serving as a valuable resource for machine learning (ML) researchers and environmentalists. We present three key contributions with the release of this dataset. Firstly, through in-depth statistical analysis, we demonstrate that the released dataset significantly differs from existing pollution datasets, highlighting its uniqueness and potential for new insights. Secondly, the dataset quality been validated against existing expensive sensors. Thirdly, we conduct a benchmarking exercise (https://github.com/sachin-iitd/DelhiPMDatasetBenchmark), evaluating state-of-the-art methods for interpolation, feature imputation, and forecasting on this dataset, which is the largest publicly available PM dataset to date. The results of the benchmarking exercise underscore the substantial disparities in accuracy between the proposed dataset and other publicly available datasets. This finding highlights the complexity and richness of our dataset, emphasizing its value for advancing research in the field of air pollution.</details>

### AircraftVerse: A Large-Scale Multimodal Dataset of Aerial Vehicle Designs

**Authors:** Adam Cobb, Anirban Roy, Daniel Elenius, Frederick Heim, Brian Swenson, Sydney Whittington, James Walker, Theodore Bapty, Joseph Hite, Karthik Ramani, Christopher McComb, Susmit Jha

**<details><summary>Abstract**</summary>

We present AircraftVerse, a publicly available aerial vehicle design dataset. Aircraft design encompasses different physics domains and, hence, multiple modalities of representation. The evaluation of these designs requires the use of scientific analytical and simulation models ranging from computer-aided design tools for structural and manufacturing analysis, computational fluid dynamics tools for drag and lift computation, battery models for energy estimation, and simulation models for flight control and dynamics. AircraftVerse contains $27{,}714$ diverse air vehicle designs - the largest corpus of designs with this level of complexity. Each design comprises the following artifacts: a symbolic design tree describing topology, propulsion subsystem, battery subsystem, and other design details; a STandard for the Exchange of Product (STEP) model data; a 3D CAD design using a stereolithography (STL) file format; a 3D point cloud for the shape of the design; and evaluation results from high fidelity state-of-the-art physics models that characterize performance metrics such as maximum flight distance and hover-time. We also present baseline surrogate models that use different modalities of design representation to predict design performance metrics, which we provide as part of our dataset release. Finally, we discuss the potential impact of this dataset on the use of learning in aircraft design, and more generally, in the emerging field of deep learning for scientific design. AircraftVerse is accompanied by a datasheet as suggested in the recent literature, and it is released under Creative Commons Attribution-ShareAlike (CC BY-SA) license. The dataset with baseline models are hosted at http://doi.org/10.5281/zenodo.6525446, code at https://github.com/SRI-CSL/AircraftVerse, and the dataset description at https://uavdesignverse.onrender.com/.</details>

### Aligning Gradient and Hessian for Neural Signed Distance Function

**Authors:** Ruian Wang, Zixiong Wang, Yunxiao Zhang, Shuangmin Chen, Shiqing Xin, Changhe Tu, Wenping Wang

**<details><summary>Abstract**</summary>

The Signed Distance Function (SDF), as an implicit surface representation, provides a crucial method for reconstructing a watertight surface from unorganized point clouds. The SDF has a fundamental relationship with the principles of surface vector calculus. Given a smooth surface, there exists a thin-shell space in which the SDF is differentiable everywhere such that the gradient of the SDF is an eigenvector of its Hessian matrix, with a corresponding eigenvalue of zero. In this paper, we introduce a method to directly learn the SDF from point clouds in the absence of normals. Our motivation is grounded in a fundamental observation: aligning the gradient and the Hessian of the SDF provides a more efficient mechanism to govern gradient directions. This, in turn, ensures that gradient changes more accurately reflect the true underlying variations in shape. Extensive experimental results demonstrate its ability to accurately recover the underlying shape while effectively suppressing the presence of ghost geometry.</details>

### [Spotlight] Aligning Synthetic Medical Images with Clinical Knowledge using Human Feedback

**Authors:** Shenghuan Sun, Greg Goldgof, Atul Butte, Ahmed Alaa

**<details><summary>Abstract**</summary>

Generative models capable of precisely capturing nuanced clinical features in medical images hold great promise for facilitating clinical data sharing, enhancing rare disease datasets, and efficiently synthesizing (annotated) medical images at scale. Despite their potential, assessing the quality of synthetic medical images remains a challenge. While modern generative models can synthesize visually-realistic medical images, the clinical plausibility of these images may be called into question. Domain-agnostic scores, such as FID score, precision, and recall, cannot incorporate clinical knowledge and are, therefore, not suitable for assessing clinical sensibility. Additionally, there are numerous unpredictable ways in which generative models may fail to synthesize clinically plausible images, making it challenging to anticipate potential failures and design automated scores for their detection. To address these challenges, this paper introduces a pathologist-in-the-loop framework for generating clinically-plausible synthetic medical images. Our framework comprises three steps: (1) pretraining a conditional diffusion model to generate medical images conditioned on a clinical concept, (2) expert pathologist evaluation of the generated images to assess whether they satisfy clinical desiderata, and (3) training a reward model that predicts human feedback on new samples, which we use to incorporate expert knowledge into the finetuning objective of the diffusion model. Our results show that human feedback significantly improves the quality of synthetic images in terms of fidelity, diversity, utility in downstream applications, and plausibility as evaluated by experts. We also demonstrate that human feedback can teach the model new clinical concepts not annotated in the original training data. Our results demonstrate the value of incorporating human feedback in clinical applications where generative models may struggle to capture extensive domain knowledge from raw data alone.</details>

### AllSim: Simulating and Benchmarking Resource Allocation Policies in Multi-User Systems

**Authors:** Jeroen Berrevoets, Daniel Jarrett, Alex Chan, Mihaela van der Schaar

**<details><summary>Abstract**</summary>

Numerous real-world systems, ranging from healthcare to energy grids, involve users competing for finite and potentially scarce resources. Designing policies for resource allocation in such real-world systems is challenging for many reasons, including the changing nature of user types and their (possibly urgent) need for resources. Researchers have developed numerous machine learning solutions for determining resource allocation policies in these challenging settings. However, a key limitation has been the absence of good methods and test-beds for benchmarking these policies; almost all resource allocation policies are benchmarked in environments which are either completely synthetic or do not allow _any_ deviation from historical data. In this paper we introduce AllSim, which is a benchmarking environment for realistically simulating the impact and utility of policies for resource allocation in systems in which users compete for such scarce resources. Building such a benchmarking environment is challenging because it needs to successfully take into account _the entire collective_ of potential users and the impact a resource allocation policy has on all the other users in the system. AllSim's benchmarking environment is modular (each component being parameterized individually), learnable (informed by historical data), and customizable (adaptable to changing conditions). These, when interacting with an allocation policy, produce a dataset of simulated outcomes for evaluation and comparison of such policies. We believe AllSim is an essential step towards a more systematic evaluation of policies for scarce resource allocation compared to current approaches for benchmarking such methods.</details>

### Alpha-divergence Variational Inference Meets Importance Weighted Auto-Encoders: Methodology and Asymptotics

**Authors:** Kamélia Daudel, Joe Benton, Yuyang Shi, Arnaud Doucet

**<details><summary>Abstract**</summary>

Several algorithms involving the Variational Rényi (VR) bound have been proposed to minimize an alpha-divergence between a target posterior distribution and a variational distribution. Despite promising empirical results, those algorithms resort to biased stochastic gradient descent procedures and thus lack theoretical guarantees. In this paper, we formalize and study the VR-IWAE bound, a generalization of the importance weighted auto-encoder (IWAE) bound. We show that the VR-IWAE bound enjoys several desirable properties and notably leads to the same stochastic gradient descent procedure as the VR bound in the reparameterized case, but this time by relying on unbiased gradient estimators. We then provide two complementary theoretical analyses of the VR-IWAE bound and thus of the standard IWAE bound. Those analyses shed light on the benefits or lack thereof of these bounds. Lastly, we illustrate our theoretical claims over toy and real-data examples.</details>

### [Spotlight] Alternating Updates for Efficient Transformers

**Authors:** Cenk Baykal, Dylan Cutler, Nishanth Dikkala, Nikhil Ghosh, Rina Panigrahy, Xin Wang

**<details><summary>Abstract**</summary>

It has been well established that increasing scale in deep transformer networks leads to improved quality and performance. However, this increase in scale often comes with prohibitive increases in compute cost and inference latency. We introduce Alternating Updates (AltUp), a simple-to-implement method to increase a model's capacity without the computational burden. AltUp enables the widening of the learned representation, i.e., the token embedding, while only incurring a negligible increase in latency. AltUp achieves this by working on a subblock of the widened representation at each layer and using a predict-and-correct mechanism to update the inactivated blocks. We present extensions of AltUp, such as its applicability to the sequence dimension, and demonstrate how AltUp can be synergistically combined with existing approaches, such as Sparse Mixture-of-Experts models, to obtain efficient models with even higher capacity. Our experiments on benchmark transformer models and language tasks demonstrate the consistent effectiveness of AltUp on a diverse set of scenarios. Notably, on SuperGLUE and SQuAD benchmarks, AltUp enables up to $87\%$ speedup relative to the dense baselines at the same accuracy.</details>

### Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation

**Authors:** Wei Jin, Haitao Mao, Zheng Li, Haoming Jiang, Chen Luo, Hongzhi Wen, Haoyu Han, Hanqing Lu, Zhengyang Wang, Ruirui Li, Zhen Li, Monica Cheng, Rahul Goutam, Haiyang Zhang, Karthik Subbian, Suhang Wang, Yizhou Sun, Jiliang Tang, Bing Yin, Xianfeng Tang

**<details><summary>Abstract**</summary>

Modeling customer shopping intentions is a crucial task for e-commerce, as it directly impacts user experience and engagement. Thus, accurately understanding customer preferences is essential for providing personalized recommendations. Session-based recommendation, which utilizes customer session data to predict their next interaction, has become increasingly popular. However, existing session datasets have limitations in terms of item attributes, user diversity, and dataset scale. As a result, they cannot comprehensively capture the spectrum of user behaviors and preferences.To bridge this gap, we present the Amazon Multilingual Multi-locale Shopping Session Dataset, namely Amazon-M2. It is the first multilingual dataset consisting of millions of user sessions from six different locales, where the major languages of products are English, German, Japanese, French, Italian, and Spanish.Remarkably, the dataset can help us enhance personalization and understanding of user preferences, which can benefit various existing tasks as well as enable new tasks. To test the potential of the dataset, we introduce three tasks in this work:(1) next-product recommendation, (2) next-product recommendation with domain shifts, and (3) next-product title generation.With the above tasks, we benchmark a range of algorithms on our proposed dataset, drawing new insights for further research and practice. In addition, based on the proposed dataset and tasks, we hosted a competition in the KDD CUP 2023 https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge and have attracted thousands of users and submissions. The winning solutions and the associated workshop can be accessed at our website~https://kddcup23.github.io/.</details>

### Amortized Reparametrization: Efficient and Scalable Variational Inference for Latent SDEs

**Authors:** Kevin Course, Prasanth Nair

**<details><summary>Abstract**</summary>

We consider the problem of inferring latent stochastic differential equations (SDEs) with a time and memory cost that scales independently with the amount of data, the total length of the time series, and the stiffness of the approximate differential equations. This is in stark contrast to typical methods for inferring latent differential equations which, despite their constant memory cost, have a time complexity that is heavily dependent on the stiffness of the approximate differential equation. We achieve this computational advancement by removing the need to solve differential equations when approximating gradients using a novel amortization strategy coupled with a recently derived reparametrization of expectations under linear SDEs. We show that, in practice, this allows us to achieve similar performance to methods based on adjoint sensitivities with more than an order of magnitude fewer evaluations of the model in training.</details>

### An Inverse Scaling Law for CLIP Training

**Authors:** Xianhang Li, Zeyu Wang, Cihang Xie

**<details><summary>Abstract**</summary>

CLIP, one of the pioneering foundation models that connect images and text, has enabled many recent breakthroughs in computer vision. However, its associated training cost is prohibitively high, imposing a significant barrier to its widespread exploration. In this paper, we present a surprising finding that there exists an inverse scaling law for CLIP training, whereby the larger the image/text encoders used, the shorter the sequence length of image/text tokens that can be applied in training. Moreover, we showcase that the strategy for reducing image/text token length plays a crucial role in determining the quality of this scaling law.As a result of this finding, we are able to successfully train CLIP even with limited computational resources. For example, using 8 A100 GPUs, our CLIP models achieve zero-shot top-1 ImageNet-1k accuracies of 63.2% in ~2 days, 67.8% in ~3 days, and 69.3% in ~4 days. Our method also works well when scaling up --- with G/14, we register a new record of 83.0% ImageNet-1k zero-shot accuracy, and meanwhile accelerate the training by ~33x compared to its OpenCLIP counterpart.By reducing the computation barrier associated with CLIP, we hope to inspire more research in this field, particularly from academics. Our code is available at https://github.com/UCSC-VLAA/CLIPA.</details>

### Analysis of Variance of Multiple Causal Networks

**Authors:** Zhongli Jiang, Dabao Zhang

**<details><summary>Abstract**</summary>

Constructing a directed cyclic graph (DCG) is challenged by both algorithmic difficulty and computational burden. Comparing multiple DCGs is even more difficult, compounded by the need of identifying variational causalities across graphs. We propose to unify multiple DCGs with a single structural model and develop a limited-information-based method to simultaneously construct multiple networks and infer their disparities, which can be visualized by appropriate correspondence analysis. The algorithm provides DCGs with robust non-asymptotic theoretical properties. It is designed with two sequential stages, each of which involves parallel computation tasks that are scalable to the network complexity. Taking advantage of high-performance clusters, our method makes it possible to evaluate the statistical significance of DCGs using the bootstrap method. We demonstrated the effectiveness of our method by applying it to synthetic and real datasets.</details>

### Analyzing Vision Transformers for Image Classification in Class Embedding Space

**Authors:** Martina G. Vilas, Timothy Schaumlöffel, Gemma Roig

**<details><summary>Abstract**</summary>

Despite the growing use of transformer models in computer vision, a mechanistic understanding of these networks is still needed. This work introduces a method to reverse-engineer Vision Transformers trained to solve image classification tasks. Inspired by previous research in NLP, we demonstrate how the inner representations at any level of the hierarchy can be projected onto the learned class embedding space to uncover how these networks build categorical representations for their predictions. We use our framework to show how image tokens develop class-specific representations that depend on attention mechanisms and contextual information, and give insights on how self-attention and MLP layers differentially contribute to this categorical composition. We additionally demonstrate that this method (1) can be used to determine the parts of an image that would be important for detecting the class of interest, and (2) exhibits significant advantages over traditional linear probing approaches. Taken together, our results position our proposed framework as a powerful tool for mechanistic interpretability and explainability research.</details>

### Anonymous Learning via Look-Alike Clustering: A Precise Analysis of Model Generalization

**Authors:** Adel Javanmard, Vahab Mirrokni

**<details><summary>Abstract**</summary>

While personalized recommendations systems have become increasingly popular, ensuring user data protection remains a top concern in the development of these learning systems. A common approach to enhancing privacy involves training models using anonymous data rather than individual data. In this paper, we explore a natural technique called "look-alike clustering", which involves replacing sensitive features of individuals with the cluster's average values. We provide a precise analysis of how training models using anonymous cluster centers affects their generalization capabilities. We focus on an asymptotic regime where the size of the training set grows in proportion to the features dimension. Our analysis is based on the  Convex Gaussian Minimax Theorem (CGMT) and allows us to theoretically understand the role of different model components on the generalization error. In addition, we demonstrate that in certain high-dimensional regimes, training over anonymous cluster centers acts as a regularization and improves generalization error of the trained models. Finally, we corroborate our asymptotic theory with finite-sample numerical experiments where we observe a perfect match when the sample size is only of order of a few hundreds.</details>

### Approximate inference of marginals using the IBIA framework

**Authors:** Shivani Bathla, Vinita Vasudevan

**<details><summary>Abstract**</summary>

Exact inference of marginals in probabilistic graphical models (PGM) is known to be intractable, necessitating the use of approximate methods. Most of the existing variational techniques perform iterative message passing in loopy graphs which is slow to converge for many benchmarks. In this paper, we propose a new algorithm for marginal inference that is based on the incremental build-infer-approximate (IBIA) paradigm. Our algorithm converts the PGM into a sequence of linked clique tree forests (SLCTF) with bounded clique sizes, and then uses a heuristic belief update algorithm to infer the marginals. For the special case of Bayesian networks, we show that if the incremental build step in IBIA uses the topological order of variables then (a) the prior marginals are consistent in all CTFs in the SLCTF  and (b) the posterior marginals are consistent once all evidence variables are added to the SLCTF. In our approach, the belief propagation step is non-iterative and the accuracy-complexity trade-off is controlled using user-defined clique size bounds. Results for several benchmark sets from recent UAI competitions show that our method gives either better or comparable accuracy than existing variational and sampling based methods, with smaller runtimes.</details>

### Approximately Equivariant Graph Networks

**Authors:** Ningyuan Huang, Ron Levie, Soledad Villar

**<details><summary>Abstract**</summary>

Graph neural networks (GNNs) are commonly described as being permutation equivariant with respect to node relabeling in the graph. This symmetry of GNNs is often compared to the translation equivariance of Euclidean convolution neural networks (CNNs). However, these two symmetries are fundamentally different: The translation equivariance of CNNs corresponds to symmetries of the fixed domain acting on the image signals (sometimes known as active symmetries), whereas in GNNs any permutation acts on both the graph signals and the graph domain (sometimes described as passive symmetries). In this work, we focus on the active symmetries of GNNs, by considering a learning setting where signals are supported on a fixed graph. In this case, the natural symmetries of GNNs are the automorphisms of the graph. Since real-world graphs tend to be asymmetric, we relax the notion of symmetries by formalizing approximate symmetries via graph coarsening. We present a bias-variance formula that quantifies the tradeoff between the loss in expressivity and the gain in the regularity of the learned estimator, depending on the chosen symmetry group. To illustrate our approach, we conduct extensive experiments on image inpainting, traffic flow prediction, and human pose estimation with different choices of symmetries. We show theoretically and empirically that the best generalization performance can be achieved by choosing a suitably larger group than the graph automorphism, but smaller than the permutation group.</details>

### Arbitrarily Scalable Environment Generators via Neural Cellular Automata

**Authors:** Yulun Zhang, Matthew Fontaine, Varun Bhatt, Stefanos Nikolaidis, Jiaoyang Li

**<details><summary>Abstract**</summary>

We study the problem of generating arbitrarily large environments to improve the throughput of multi-robot systems. Prior work proposes Quality Diversity (QD) algorithms as an effective method for optimizing the environments of automated warehouses. However, these approaches optimize only relatively small environments, falling short when it comes to replicating real-world warehouse sizes. The challenge arises from the exponential increase in the search space as the environment size increases. Additionally, the previous methods have only been tested with up to 350 robots in simulations, while practical warehouses could host thousands of robots. In this paper, instead of optimizing environments, we propose to optimize Neural Cellular Automata (NCA) environment generators via QD algorithms. We train a collection of NCA generators with QD algorithms in small environments and then generate arbitrarily large environments from the generators at test time. We show that NCA environment generators maintain consistent, regularized patterns regardless of environment size, significantly enhancing the scalability of multi-robot systems in two different domains with up to 2,350 robots. Additionally, we demonstrate that our method scales a single-agent reinforcement learning policy to arbitrarily large environments with similar patterns. We include the source code at https://github.com/lunjohnzhang/warehouse_env_gen_nca_public.</details>

### Are GATs Out of Balance?

**Authors:** Nimrah Mustafa, Aleksandar Bojchevski, Rebekka Burkholz

**<details><summary>Abstract**</summary>

While the expressive power and computational capabilities of graph neural networks (GNNs) have been theoretically studied, their optimization and learning dynamics, in general, remain largely unexplored. Our study undertakes the Graph Attention Network (GAT), a popular GNN architecture in which a node's neighborhood aggregation is weighted by parameterized attention coefficients. We derive a conservation law of GAT gradient flow dynamics, which explains why a high portion of parameters in GATs with standard initialization struggle to change during training. This effect is amplified in deeper GATs, which perform significantly worse than their shallow counterparts. To alleviate this problem, we devise an initialization scheme that balances the GAT network. Our approach i) allows more effective propagation of gradients and in turn enables trainability of deeper networks, and ii) attains a considerable speedup in training and convergence time in comparison to the standard initialization. Our main theorem serves as a stepping stone to studying the learning dynamics of positive homogeneous models with attention mechanisms.</details>

### Are These the Same Apple? Comparing Images Based on Object Intrinsics

**Authors:** Klemen Kotar, Stephen Tian, Hong-Xing Yu, Dan Yamins, Jiajun Wu

**<details><summary>Abstract**</summary>

The human visual system can effortlessly recognize an object under different extrinsic factors such as lighting, object poses, and background, yet current computer vision systems often struggle with these variations. An important step to understanding and improving artificial vision systems is to measure image similarity purely based on intrinsic object properties that define object identity. This problem has been studied in the computer vision literature as re-identification, though mostly restricted to specific object categories such as people and cars. We propose to extend it to general object categories, exploring an image similarity metric based on object intrinsics. To benchmark such measurements, we collect the Common paired objects Under differenT Extrinsics (CUTE) dataset of 18, 000 images of 180 objects under different extrinsic factors such as lighting, poses, and imaging conditions. While existing methods such as LPIPS and CLIP scores do not measure object intrinsics well, we find that combining deep features learned from contrastive self-supervised learning with foreground filtering is a simple yet effective approach to approximating the similarity. We conduct an extensive survey of pre-trained features and foreground extraction methods to arrive at a strong baseline that best measures intrinsic object-centric image similarity among current methods. Finally, we demonstrate that our approach can aid in downstream applications such as acting as an analog for human subjects and improving generalizable re-identification. Please see our project website at https://s-tian.github.io/projects/cute/ for visualizations of the data and demos of our metric.</details>

### Are Vision Transformers More Data Hungry Than Newborn Visual Systems?

**Authors:** Lalit Pandey, Samantha Wood, Justin Wood

**<details><summary>Abstract**</summary>

Vision transformers (ViTs) are top-performing models on many computer vision benchmarks and can accurately predict human behavior on object recognition tasks. However, researchers question the value of using ViTs as models of biological learning because ViTs are thought to be more “data hungry” than brains, with ViTs requiring more training data than brains to reach similar levels of performance. To test this assumption, we directly compared the learning abilities of ViTs and animals, by performing parallel controlled-rearing experiments on ViTs and newborn chicks. We first raised chicks in impoverished visual environments containing a single object, then simulated the training data available in those environments by building virtual animal chambers in a video game engine. We recorded the first-person images acquired by agents moving through the virtual chambers and used those images to train self-supervised ViTs that leverage time as a teaching signal, akin to biological visual systems. When ViTs were trained “through the eyes” of newborn chicks, the ViTs solved the same view-invariant object recognition tasks as the chicks. Thus, ViTs were not more data hungry than newborn chicks: both learned view-invariant object representations in impoverished visual environments. The flexible and generic attention-based learning mechanism in ViTs—combined with the embodied data streams available to newborn animals—appears sufficient to drive the development of animal-like object recognition.</details>

### Asymmetric Certified Robustness via Feature-Convex Neural Networks

**Authors:** Samuel Pfrommer, Brendon Anderson, Julien Piet, Somayeh Sojoudi

**<details><summary>Abstract**</summary>

Real-world adversarial attacks on machine learning models often feature an asymmetric structure wherein adversaries only attempt to induce false negatives (e.g., classify a spam email as not spam). We formalize the asymmetric robustness certification problem and correspondingly present the feature-convex neural network architecture, which composes an input-convex neural network (ICNN) with a Lipschitz continuous feature map in order to achieve asymmetric adversarial robustness. We consider the aforementioned binary setting with one "sensitive" class, and for this class we prove deterministic, closed-form, and easily-computable certified robust radii for arbitrary $\ell_p$-norms. We theoretically justify the use of these models by characterizing their decision region geometry, extending the universal approximation theorem for ICNN regression to the classification setting, and proving a lower bound on the probability that such models perfectly fit even unstructured uniformly distributed data in sufficiently high dimensions. Experiments on Malimg malware classification and subsets of the MNIST, Fashion-MNIST, and CIFAR-10 datasets show that feature-convex classifiers attain substantial certified $\ell_1$, $\ell_2$, and $\ell_{\infty}$-radii while being far more computationally efficient than competitive baselines.</details>

### Augmentation-free Dense Contrastive Distillation for Efficient Semantic Segmentation

**Authors:** Jiawei Fan, Chao Li, Xiaolong Liu, Meina Song, Anbang Yao

**<details><summary>Abstract**</summary>

In recent years, knowledge distillation methods based on contrastive learning have achieved promising results on image classification and object detection tasks. However, in this line of research, we note that less attention is paid to semantic segmentation. Existing methods heavily rely on data augmentation and memory buffer, which entail high computational resource demands when applying them to handle semantic segmentation that requires to preserve high-resolution feature maps for making dense pixel-wise predictions. In order to address this problem, we present Augmentation-free Dense Contrastive Knowledge Distillation (Af-DCD), a new contrastive distillation learning paradigm to train compact and accurate deep neural networks for semantic segmentation applications. Af-DCD leverages a masked feature mimicking strategy, and formulates a novel contrastive learning loss via taking advantage of tactful feature partitions across both channel and spatial dimensions, allowing to effectively transfer dense and structured local knowledge learnt by the teacher model to a target student model while maintaining training efficiency. Extensive experiments on five mainstream benchmarks with various teacher-student network pairs demonstrate the effectiveness of our approach. For instance, DeepLabV3-Res18|DeepLabV3-MBV2 model trained by Af-DCD reaches 77.03\%|76.38\% mIOU on Cityscapes dataset when choosing DeepLabV3-Res101 as the teacher, setting new performance records. Besides that, Af-DCD achieves an absolute mIOU improvement of 3.26\%|3.04\%|2.75\%|2.30\%|1.42\% compared with individually trained counterpart on Cityscapes|Pascal VOC|Camvid|ADE20K|COCO-Stuff-164K. Code is available at https://github.com/OSVAI/Af-DCD.</details>

### Automatic Integration for Spatiotemporal Neural Point Processes

**Authors:** Zihao Zhou, Rose Yu

**<details><summary>Abstract**</summary>

Learning continuous-time point processes is essential to many discrete event forecasting tasks. However, integration poses a major challenge, particularly for spatiotemporal point processes (STPPs), as it involves calculating the likelihood through triple integrals over space and time. Existing methods for integrating STPP either assume a parametric form of the intensity function, which lacks flexibility; or approximating the intensity with Monte Carlo sampling, which introduces numerical errors. Recent work by Omi et al. proposes a dual network approach for efficient integration of flexible intensity function. However, their method only focuses on the 1D temporal point process. In this paper, we introduce a novel paradigm: `Auto-STPP` (Automatic Integration for Spatiotemporal Neural Point Processes) that extends the dual network approach to 3D STPP. While previous work provides a foundation, its direct extension overly restricts the intensity function and leads to computational challenges. In response, we introduce a decomposable parametrization for the integral network using ProdNet. This approach, leveraging the product of simplified univariate graphs, effectively sidesteps the computational complexities inherent in multivariate computational graphs. We prove the consistency of `Auto-STPP` and validate it on synthetic data and benchmark real-world datasets. `Auto-STPP` shows a significant advantage in recovering complex intensity functions from irregular spatiotemporal events, particularly when the intensity is sharply localized. Our code is open-source at https://github.com/Rose-STL-Lab/AutoSTPP.</details>

### Bayesian Learning of Optimal Policies in Markov Decision Processes with Countably Infinite State-Space

**Authors:** Saghar Adler, Vijay Subramanian

**<details><summary>Abstract**</summary>

Models of many real-life applications, such as queueing models of communication networks or computing systems, have a countably infinite state-space. Algorithmic and learning procedures that have been developed to produce optimal policies mainly focus on finite state settings, and do not directly apply to these models. To overcome this lacuna, in this work we study the problem of optimal control of a family of discrete-time countable state-space Markov Decision Processes (MDPs) governed by an unknown parameter $\theta\in\Theta$,  and defined on a countably-infinite state-space $\mathcal X=\mathbb{Z}_+^d$, with finite action space $\mathcal A$, and an unbounded cost function. We take a Bayesian perspective with the random unknown parameter $\boldsymbol{\theta}^*$ generated via a given fixed prior distribution on $\Theta$. To optimally control the unknown MDP, we propose an algorithm based on Thompson sampling with dynamically-sized episodes: at the beginning of each episode, the posterior distribution formed via Bayes' rule is used to produce a parameter estimate, which then decides the policy applied during the episode. To ensure the stability of the Markov chain obtained by following the policy chosen for each parameter, we impose ergodicity assumptions. From this condition and using the solution of the average cost Bellman equation, we establish an $\tilde O(dh^d\sqrt{|\mathcal A|T})$ upper bound on the Bayesian regret of our algorithm, where $T$ is the time-horizon. Finally, to elucidate the applicability of our algorithm, we consider two different queueing models with unknown dynamics, and show that our algorithm can be applied to develop approximately optimal control algorithms.</details>

### Bayesian Optimization with Cost-varying Variable Subsets

**Authors:** Sebastian Tay, Chuan Sheng Foo, Daisuke Urano, Richalynn Leong, Bryan Kian Hsiang Low

**<details><summary>Abstract**</summary>

We introduce the problem of Bayesian optimization with cost-varying variable subsets (BOCVS) where in each iteration, the learner chooses a subset of query variables and specifies their values while the rest are randomly sampled. Each chosen subset has an associated cost. This presents the learner with the novel challenge of balancing between choosing more informative subsets for more directed learning versus leaving some variables to be randomly sampled to reduce incurred costs. This paper presents a novel Gaussian process upper confidence bound-based algorithm for solving the BOCVS problem that is provably no-regret. We analyze how the availability of cheaper control sets helps in exploration and reduces overall regret. We empirically show that our proposed algorithm can find significantly better solutions than comparable baselines with the same budget.</details>

### [Spotlight] Bayesian target optimisation for high-precision holographic optogenetics

**Authors:** Marcus Triplett, Marta Gajowa, Hillel Adesnik, Liam Paninski

**<details><summary>Abstract**</summary>

Two-photon optogenetics has transformed our ability to probe the structure and function of neural circuits. However, achieving precise optogenetic control of neural ensemble activity has remained fundamentally constrained by the problem of off-target stimulation (OTS): the inadvertent activation of nearby non-target neurons due to imperfect confinement of light onto target neurons. Here we propose a novel computational approach to this problem called Bayesian target optimisation. Our approach uses nonparametric Bayesian inference to model neural responses to optogenetic stimulation, and then optimises the laser powers and optical target locations needed to achieve a desired activity pattern with minimal OTS. We validate our approach in simulations and using data from in vitro experiments, showing that Bayesian target optimisation considerably reduces OTS across all conditions we test. Together, these results establish our ability to overcome OTS, enabling optogenetic stimulation with substantially improved precision.</details>

### [Spotlight] Behavior Alignment via Reward Function Optimization

**Authors:** Dhawal Gupta, Yash Chandak, Scott Jordan, Philip Thomas, Bruno da Silva

**<details><summary>Abstract**</summary>

Designing reward functions for efficiently guiding reinforcement learning (RL) agents toward specific behaviors is a complex task.This is challenging since it requires the identification of reward structures that are not sparse and that avoid inadvertently inducing undesirable behaviors. Naively modifying the reward structure to offer denser and more frequent feedback can lead to unintended outcomes and promote behaviors that are not aligned with the designer's intended goal. Although potential-based reward shaping is often suggested as a remedy, we systematically investigate settings where deploying it often significantly impairs performance. To address these issues, we introduce a new framework that uses a bi-level objective to learn \emph{behavior alignment reward functions}. These functions integrate auxiliary rewards reflecting a designer's heuristics and domain knowledge with the environment's primary rewards. Our approach automatically determines the most effective way to blend these types of feedback, thereby enhancing robustness against heuristic reward misspecification. Remarkably, it can also adapt an agent's policy optimization process to mitigate suboptimalities resulting from limitations and biases inherent in the underlying RL algorithms. We evaluate our method's efficacy on a diverse set of tasks, from small-scale experiments to high-dimensional control challenges. We investigate heuristic auxiliary rewards of varying quality---some of which are beneficial and others detrimental to the learning process. Our results show that our framework offers a robust and principled way to integrate designer-specified heuristics. It not only addresses key shortcomings of existing approaches but also consistently leads to high-performing solutions, even when given misaligned or poorly-specified auxiliary reward functions.</details>

### Beta Diffusion

**Authors:** Mingyuan Zhou, Tianqi Chen, Zhendong Wang, Huangjie Zheng

**<details><summary>Abstract**</summary>

We introduce beta diffusion, a novel generative modeling method that integrates demasking and denoising to generate data within bounded ranges. Using scaled and shifted beta distributions, beta diffusion utilizes multiplicative transitions over time to create both forward and reverse diffusion processes, maintaining beta distributions in both the forward marginals and the reverse conditionals, given the data at any point in time. Unlike traditional diffusion-based generative models relying on additive Gaussian noise and reweighted evidence lower bounds (ELBOs), beta diffusion is multiplicative and optimized with KL-divergence upper bounds (KLUBs) derived from the convexity of the KL divergence. We demonstrate that the proposed KLUBs are more effective for optimizing beta diffusion compared to negative ELBOs, which can also be derived as the KLUBs of the same KL divergence with its two arguments swapped. The loss function of beta diffusion, expressed in terms of Bregman divergence, further supports the efficacy of KLUBs for optimization. Experimental results on both synthetic data and natural images demonstrate the unique capabilities of beta diffusion in generative modeling of range-bounded data and validate the effectiveness of KLUBs in optimizing diffusion models, thereby making them valuable additions to the family of diffusion-based generative models and the optimization techniques used to train them.</details>

### Better Private Linear Regression Through Better Private Feature Selection

**Authors:** Travis Dick, Jennifer Gillenwater, Matthew Joseph

**<details><summary>Abstract**</summary>

Existing work on differentially private linear regression typically assumes that end users can precisely set data bounds or algorithmic hyperparameters. End users often struggle to meet these requirements without directly examining the data (and violating privacy). Recent work has attempted to develop solutions that shift these burdens from users to algorithms, but they struggle to provide utility as the feature dimension grows. This work extends these algorithms to higher-dimensional problems by introducing a differentially private feature selection method based on Kendall rank correlation. We prove a utility guarantee for the setting where features are normally distributed and conduct experiments across 25 datasets. We find that adding this private feature selection step before regression significantly broadens the applicability of ``plug-and-play'' private linear regression algorithms at little additional cost to privacy, computation, or decision-making by the end user.</details>

### Binarized Neural Machine Translation

**Authors:** Yichi Zhang, Ankush Garg, Yuan Cao, Lukasz Lew, Behrooz Ghorbani, Zhiru Zhang, Orhan Firat

**<details><summary>Abstract**</summary>

The rapid scaling of language models is motivating research using low-bitwidth quantization.In this work, we propose a novel binarization technique for Transformers applied to machine translation (BMT), the first of its kind. We identify and address the problem of inflated dot-product variance when using one-bit weights and activations. Specifically, BMT leverages additional LayerNorms and residual connections to improve binarization quality. Experiments on the WMT dataset show that a one-bit weight-only Transformer can achieve the same quality as a float one, while being 16$\times$ smaller in size. One-bit activations incur varying degrees of quality drop, but mitigated by the proposed architectural changes. We further conduct a scaling law study using production-scale translation datasets, which shows that one-bit weight Transformers scale and generalize well in both in-domain and out-of-domain settings. Implementation in JAX/Flax will be open sourced.</details>

### Binary Classification with Confidence Difference

**Authors:** Wei Wang, Lei Feng, Yuchen Jiang, Gang Niu, Min-Ling Zhang, Masashi Sugiyama

**<details><summary>Abstract**</summary>

Recently, learning with soft labels has been shown to achieve better performance than learning with hard labels in terms of model generalization, calibration, and robustness. However, collecting pointwise labeling confidence for all training examples can be challenging and time-consuming in real-world scenarios. This paper delves into a novel weakly supervised binary classification problem called confidence-difference (ConfDiff) classification. Instead of pointwise labeling confidence, we are given only unlabeled data pairs with confidence difference that specifies the difference in the probabilities of being positive. We propose a risk-consistent approach to tackle this problem and show that the estimation error bound achieves the optimal convergence rate. We also introduce a risk correction approach to mitigate overfitting problems, whose consistency and convergence rate are also proven. Extensive experiments on benchmark data sets and a real-world recommender system data set validate the effectiveness of our proposed approaches in exploiting the supervision information of the confidence difference.</details>

### Black-box Backdoor Defense via Zero-shot Image Purification

**Authors:** Yucheng Shi, Mengnan Du, Xuansheng Wu, Zihan Guan, Jin Sun, Ninghao Liu

**<details><summary>Abstract**</summary>

Backdoor attacks inject poisoned samples into the training data, resulting in the misclassification of the poisoned input during a model's deployment. Defending against such attacks is challenging, especially for real-world black-box models where only query access is permitted. In this paper, we propose a novel defense framework against backdoor attacks through Zero-shot Image Purification (ZIP). Our framework can be applied to poisoned models without requiring internal information about the model or any prior knowledge of the clean/poisoned samples. Our defense framework involves two steps. First, we apply a linear transformation (e.g., blurring) on the poisoned image to destroy the backdoor pattern. Then, we use a pre-trained diffusion model to recover the missing semantic information removed by the transformation. In particular, we design a new reverse process by using the transformed image to guide the generation of high-fidelity purified images, which works in zero-shot settings. We evaluate our ZIP framework on multiple datasets with different types of attacks. Experimental results demonstrate the superiority of our ZIP framework compared to state-of-the-art backdoor defense baselines. We believe that our results will provide valuable insights for future defense methods for black-box models. Our code is available at https://github.com/sycny/ZIP.</details>

### Blurred-Dilated Method for Adversarial Attacks

**Authors:** Yang Deng, Weibin Wu, Jianping Zhang, Zibin Zheng

**<details><summary>Abstract**</summary>

Deep neural networks (DNNs) are vulnerable to adversarial attacks, which lead to incorrect predictions. In black-box settings, transfer attacks can be conveniently used to generate adversarial examples. However, such examples tend to overfit the specific architecture and feature representations of the source model, resulting in poor attack performance against other target models. To overcome this drawback, we propose a novel model modification-based transfer attack: Blurred-Dilated method (BD) in this paper. In summary, BD works by reducing downsampling while introducing BlurPool and dilated convolutions in the source model. Then BD employs the modified source model to generate adversarial samples. We think that BD can more comprehensively preserve the feature information than the original source model. It thus enables more thorough destruction of the image features, which can improve the transferability of the generated adversarial samples. Extensive experiments on the ImageNet dataset show that adversarial examples generated by BD achieve significantly higher transferability than the state-of-the-art baselines. Besides, BD can be conveniently combined with existing black-box attack techniques to further improve their performance.</details>

### BoardgameQA: A Dataset for Natural Language Reasoning with Contradictory Information

**Authors:** Mehran Kazemi, Quan Yuan, Deepti Bhatia, Najoung Kim, Xin Xu, Vaiva Imbrasaite, Deepak Ramachandran

**<details><summary>Abstract**</summary>

Automated reasoning with unstructured natural text is a key requirement for many potential applications of NLP and for developing robust AI systems. Recently, Language Models (LMs) have demonstrated complex reasoning capacities even without any finetuning. However, existing evaluation for automated reasoning assumes access to a consistent and coherent set of information over which models reason. When reasoning in the real-world, the available information is frequently inconsistent or contradictory, and therefore models need to be equipped with a strategy to resolve such conflicts when they arise. One widely-applicable way of resolving conflicts is to impose preferences over information sources (e.g., based on source credibility or information recency) and adopt the source with higher preference. In this paper, we formulate the problem of reasoning with contradictory information guided by preferences over sources as the classical problem of defeasible reasoning, and develop a dataset called BoardgameQA for measuring the reasoning capacity of LMs in this setting. BoardgameQA also incorporates reasoning with implicit background knowledge, to better reflect reasoning problems in downstream applications. We benchmark various LMs on BoardgameQA and the results reveal a significant gap in the reasoning capacity of state-of-the-art LMs on this problem, showing that reasoning with conflicting information does not surface out-of-the-box in LMs. While performance can be improved with finetuning, it nevertheless remains poor.</details>

### Boosting with Tempered Exponential Measures

**Authors:** Richard Nock, Ehsan Amid, Manfred Warmuth

**<details><summary>Abstract**</summary>

One of the most popular ML algorithms, AdaBoost, can bederived from the dual of a relative entropyminimization problem subject to the fact that the positive weightson the examples sum to one. Essentially, harder examples receive higher probabilities. We generalize this setup to the recently introduced *temperedexponential measure*s (TEMs) where normalization is enforced on a specific power of the measure and not the measure itself.TEMs are indexed by a parameter $t$ and generalize exponential families ($t=1$). Our algorithm, $t$-AdaBoost, recovers AdaBoost as a special case ($t=1$). We show that $t$-AdaBoost retains AdaBoost's celebrated exponential convergence rate when $t\in [0,1)$ while allowing a slight improvement of the rate's hidden constant compared to $t=1$. $t$-AdaBoost partially computes on a generalization of classical arithmetic over the reals and brings notable properties like guaranteed bounded leveraging coefficients for $t\in [0,1)$. From the loss that $t$-AdaBoost minimizes (a generalization of the exponential loss), we show how to derive a new family of *tempered* losses for the induction of domain-partitioning classifiers like decision trees. Crucially, strict properness is ensured for all while their boosting rates span the full known spectrum. Experiments using $t$-AdaBoost+trees display that significant leverage can be achieved by tuning $t$.</details>

### Bottleneck Structure in Learned Features: Low-Dimension vs Regularity Tradeoff

**Authors:** Arthur Jacot

**<details><summary>Abstract**</summary>

Previous work has shown that DNNs withlarge depth $L$ and $L_{2}$-regularization are biased towards learninglow-dimensional representations of the inputs, which can be interpretedas minimizing a notion of rank $R^{(0)}(f)$ of the learned function$f$, conjectured to be the Bottleneck rank. We compute finite depthcorrections to this result, revealing a measure $R^{(1)}$ of regularitywhich bounds the pseudo-determinant of the Jacobian $\left\|Jf(x)\right\|\_\+$and is subadditive under composition and addition. This formalizesa balance between learning low-dimensional representations and minimizingcomplexity/irregularity in the feature maps, allowing the networkto learn the `right' inner dimension. Finally, we prove the conjecturedbottleneck structure in the learned features as $L\to\infty$: forlarge depths, almost all hidden representations are approximately$R^{(0)}(f)$-dimensional, and almost all weight matrices $W_{\ell}$have $R^{(0)}(f)$ singular values close to 1 while the others are$O(L^{-\frac{1}{2}})$. Interestingly, the use of large learning ratesis required to guarantee an order $O(L)$ NTK which in turns guaranteesinfinite depth convergence of the representations of almost all layers.</details>

### Bounded rationality in structured  density estimation

**Authors:** Tianyuan Teng, Kevin Li, Hang Zhang

**<details><summary>Abstract**</summary>

Learning to accurately represent environmental uncertainty is crucial for adaptive and optimal behaviors in various cognitive tasks. However, it remains unclear how the human brain, constrained by finite cognitive resources, constructs an internal model from an infinite space of probability distributions. In this study, we explore how these learned distributions deviate from the ground truth, resulting in observable inconsistency in a novel structured density estimation task. During each trial, human participants were asked to form and report the latent probability distribution functions underlying sequentially presented independent observations. As the number of observations increased, the reported predictive density became closer to the ground truth. Nevertheless, we observed an intriguing inconsistency in human structure estimation, specifically a large error in the number of reported clusters. Such inconsistency is invariant to the scale of the distribution and persists across stimulus modalities. We modeled uncertainty learning as approximate Bayesian inference in a nonparametric mixture prior of distributions. Human reports were best explained under resource rationality embodied in a decaying tendency towards model expansion. Our study offers insights into human cognitive processes under uncertainty and lays the groundwork for further exploration of resource-rational representations in the brain under more complex tasks.</details>

### Bounding the Invertibility of Privacy-preserving Instance Encoding using Fisher Information

**Authors:** Kiwan Maeng, Chuan Guo, Sanjay Kariyappa, G. Edward Suh

**<details><summary>Abstract**</summary>

Privacy-preserving instance encoding aims to encode raw data into feature vectors without revealing their privacy-sensitive information. When designed properly, these encodings can be used for downstream ML applications such as training and inference with limited privacy risk. However, the vast majority of existing schemes do not theoretically justify that their encoding is non-invertible, and their privacy-enhancing properties are only validated empirically against a limited set of attacks. In this paper, we propose a theoretically-principled measure for the invertibility of instance encoding based on Fisher information that is broadly applicable to a wide range of popular encoders. We show that dFIL can be used to bound the invertibility of encodings both theoretically and empirically, providing an intuitive interpretation of the privacy of instance encoding.</details>

### BubbleML: A Multiphase Multiphysics Dataset and Benchmarks for Machine Learning

**Authors:** Sheikh Md Shakeel Hassan, Arthur Feeney, Akash Dhruv, Jihoon Kim, Youngjoon Suh, Jaiyoung Ryu, Yoonjin Won, Aparna Chandramowlishwaran

**<details><summary>Abstract**</summary>

In the field of phase change phenomena, the lack of accessible and diverse datasets suitable for machine learning (ML) training poses a significant challenge. Existing experimental datasets are often restricted, with limited availability and sparse ground truth, impeding our understanding of this complex multiphysics phenomena. To bridge this gap, we present the BubbleML dataset which leverages physics-driven simulations to provide accurate ground truth information for various boiling scenarios, encompassing nucleate pool boiling, flow boiling, and sub-cooled boiling. This extensive dataset covers a wide range of parameters, including varying gravity conditions, flow rates, sub-cooling levels, and wall superheat, comprising 79 simulations.  BubbleML is validated against experimental observations and trends, establishing it as an invaluable resource for ML research. Furthermore, we showcase its potential to facilitate the exploration of diverse downstream tasks by introducing two benchmarks: (a) optical flow analysis to capture bubble dynamics, and (b) neural PDE solvers for learning temperature and flow dynamics. The BubbleML dataset and its benchmarks aim to catalyze progress in ML-driven research on multiphysics phase change phenomena, providing robust baselines for the development and comparison of state-of-the-art techniques and models.</details>

### CAP:  Correlation-Aware Pruning for Highly-Accurate Sparse Vision Models

**Authors:** Denis Kuznedelev, Eldar Kurtić, Elias Frantar, Dan Alistarh

**<details><summary>Abstract**</summary>

Driven by significant improvements in architectural design and training pipelines, computer visionhas recently experienced dramatic progress in terms of accuracy on classic benchmarks such as ImageNet. These highly-accurate models are challenging to deploy,  as they appear harder to compress using standard techniques such as pruning. We address this issue by introducing the Correlation Aware Pruner (CAP), a new unstructured pruning framework which significantly pushes the compressibility limits for state-of-the-art architectures.Our method is based on two technical advancements: a new theoretically-justified pruner, which can handle complex weight correlations accurately and efficiently during the pruning process itself,  and an efficient finetuning procedure for post-compression recovery. We validate our approach via extensive experiments on several modern vision models such as Vision Transformers (ViT), modern CNNs, and ViT-CNN hybrids, showing for the first time that these can be pruned to high sparsity levels (e.g. $\geq 75$%) with low impact on accuracy ($\leq 1$% relative drop). Our approach is also compatible with structured pruning and quantization, and can lead to practical speedups of 1.5 to 2.4x without accuracy loss. To further showcase CAP's accuracy and scalability, we use it to show for the first time that extremely-accurate large vision models, trained via self-supervised techniques,  can also be pruned to moderate sparsities, with negligible accuracy loss.</details>

### CAT-Walk: Inductive Hypergraph Learning via Set Walks

**Authors:** Ali Behrouz, Farnoosh Hashemi, Sadaf Sadeghian, Margo Seltzer

**<details><summary>Abstract**</summary>

Temporal hypergraphs provide a powerful paradigm for modeling time-dependent, higher-order interactions in complex systems. Representation learning for hypergraphs is essential for extracting patterns of the higher-order interactions that are critically important in real-world problems in social network analysis, neuroscience, finance, etc. However, existing methods are typically designed only for specific tasks or static hypergraphs. We present CAT-Walk, an inductive method that learns the underlying dynamic laws that govern the temporal and structural processes underlying a temporal hypergraph. CAT-Walk introduces a temporal, higher-order walk on hypergraphs, SetWalk, that extracts higher-order causal patterns. CAT-Walk uses a novel adaptive and permutation invariant pooling strategy, SetMixer, along with a set-based anonymization process that hides the identity of hyperedges. Finally, we present a simple yet effective neural network model to encode hyperedges. Our evaluation on 10 hypergraph benchmark datasets shows that CAT-Walk attains outstanding performance on temporal hyperedge prediction benchmarks in both inductive and transductive settings. It also shows competitive performance with state-of-the-art methods for node classification. (https://github.com/ubc-systopia/CATWalk)</details>

### [Spotlight] CLIP-OGD: An Experimental Design for Adaptive Neyman Allocation in Sequential Experiments

**Authors:** Jessica Dai, Paula Gradu, Christopher Harshaw

**<details><summary>Abstract**</summary>

From clinical development of cancer therapies to investigations into partisan bias, adaptive sequential designs have become increasingly popular method for causal inference, as they offer the possibility of improved precision over their non-adaptive counterparts. However, even in simple settings (e.g. two treatments) the extent to which adaptive designs can improve precision is not sufficiently well understood. In this work, we study the problem of Adaptive Neyman Allocation in a design-based potential outcomes framework, where the experimenter seeks to construct an adaptive design which is nearly as efficient as the optimal (but infeasible) non-adaptive Neyman design, which has access to all potential outcomes. Motivated by connections to online optimization, we propose Neyman Ratio and Neyman Regret as two (equivalent) performance measures of adaptive designs for this problem. We present Clip-OGD, an adaptive design which achieves $\widetilde{\mathcal{O}}(\sqrt{T})$ expected Neyman regret and thereby recovers the optimal Neyman variance in large samples. Finally, we construct a conservative variance estimator which facilitates the development of asymptotically valid confidence intervals. To complement our theoretical results, we conduct simulations using data from a microeconomic experiment.</details>

### CLadder: A Benchmark to Assess Causal Reasoning Capabilities of Language Models

**Authors:** Zhijing Jin, Yuen Chen, Felix Leeb, Luigi Gresele, Ojasv Kamal, Zhiheng LYU, Kevin Blin, Fernando Gonzalez Adauto, Max Kleiman-Weiner, Mrinmaya Sachan, Bernhard Schölkopf

**<details><summary>Abstract**</summary>

The ability to perform causal reasoning is widely considered a core feature of intelligence. In this work, we investigate whether large language models (LLMs) can coherently reason about causality. Much of the existing work in natural language processing (NLP) focuses on evaluating _commonsense_ causal reasoning in LLMs, thus failing to assess whether a model can perform causal inference in accordance with a set of well-defined _formal rules_. To address this, we propose a new NLP task, _causal inference in natural language_, inspired by the _“causal inference engine”_ postulated by Judea Pearl et al. We compose a large dataset, CLadder, with 10K samples: based on a collection of causal graphs and queries (associational, interventional, and counterfactual), we obtain symbolic questions and ground-truth answers, through an oracle causal inference engine. These are then translated into natural language. We evaluate multiple LLMs on our dataset, and we introduce and evaluate a bespoke chain-of-thought prompting strategy, CausalCoT. We show that our task is highly challenging for LLMs, and we conduct an in-depth analysis to gain deeper insight into the causal reasoning abilities of LLMs</details>

### COOM: A Game Benchmark for Continual Reinforcement Learning

**Authors:** Tristan Tomilin, Meng Fang, Yudi Zhang, Mykola Pechenizkiy

**<details><summary>Abstract**</summary>

The advancement of continual reinforcement learning (RL) has been facing various obstacles, including standardized metrics and evaluation protocols, demanding computational requirements, and a lack of widely accepted standard benchmarks. In response to these challenges, we present COOM ($\textbf{C}$ontinual D$\textbf{OOM}$), a continual RL benchmark tailored for embodied pixel-based RL. COOM presents a meticulously crafted suite of task sequences set within visually distinct 3D environments, serving as a robust evaluation framework to assess crucial aspects of continual RL, such as catastrophic forgetting, knowledge transfer, and sample-efficient learning. Following an in-depth empirical evaluation of popular continual learning (CL) methods, we pinpoint their limitations, provide valuable insight into the benchmark and highlight unique algorithmic challenges. This makes our work the first to benchmark image-based CRL in 3D environments with embodied perception. The primary objective of the COOM benchmark is to offer the research community a valuable and cost-effective challenge. It seeks to deepen our comprehension of the capabilities and limitations of current and forthcoming CL methods in an RL setting. The code and environments are open-sourced and accessible on GitHub.</details>

### CORNN: Convex optimization of recurrent neural networks for rapid inference of neural dynamics

**Authors:** Fatih Dinc, Adam Shai, Mark Schnitzer, Hidenori Tanaka

**<details><summary>Abstract**</summary>

Advances in optical and electrophysiological recording technologies have made it possible to record the dynamics of thousands of neurons, opening up new possibilities for interpreting and controlling large neural populations in behaving animals. A promising way to extract computational principles from these large datasets is to train data-constrained recurrent neural networks (dRNNs). Performing this training in real-time could open doors for research techniques and medical applications to model and control interventions at single-cell resolution and drive desired forms of animal behavior. However, existing training algorithms for dRNNs are inefficient and have limited scalability, making it a challenge to analyze large neural recordings even in offline scenarios. To address these issues, we introduce a training method termed Convex Optimization of Recurrent Neural Networks (CORNN). In studies of simulated recordings, CORNN attained training speeds $\sim$100-fold faster than traditional optimization approaches while maintaining or enhancing modeling accuracy. We further validated CORNN on simulations with thousands of cells that performed simple computations such as those of a 3-bit flip-flop or the execution of a timed response. Finally, we showed that CORNN can robustly reproduce network dynamics and underlying attractor structures despite mismatches between generator and inference models, severe subsampling of observed neurons, or mismatches in neural time-scales. Overall, by training dRNNs with millions of parameters in subminute processing times on a standard computer, CORNN constitutes a first step towards real-time network reproduction constrained on large-scale neural recordings and a powerful computational tool for advancing the understanding of neural computation.</details>

### CP-SLAM: Collaborative Neural Point-based SLAM System

**Authors:** Jiarui Hu, Mao Mao, Hujun Bao, Guofeng Zhang, Zhaopeng Cui

**<details><summary>Abstract**</summary>

This paper presents a collaborative implicit neural simultaneous localization and mapping (SLAM) system with RGB-D image sequences, which consists of complete front-end and back-end modules including odometry, loop detection, sub-map fusion, and global refinement. In order to enable all these modules in a unified framework, we propose a novel neural point based 3D scene representation in which each point maintains a learnable neural feature for scene encoding and is associated with a certain keyframe. Moreover, a distributed-to-centralized learning strategy is proposed for the collaborative implicit SLAM to improve consistency and cooperation. A novel global optimization framework is also proposed to improve the system accuracy like traditional bundle adjustment. Experiments on various datasets demonstrate the superiority of the proposed method in both camera tracking and mapping.</details>

### CSOT: Curriculum and Structure-Aware Optimal Transport for Learning with Noisy Labels

**Authors:** Wanxing Chang, Ye Shi, Jingya Wang

**<details><summary>Abstract**</summary>

Learning with noisy labels (LNL) poses a significant challenge in training a well-generalized model while avoiding overfitting to corrupted labels.Recent advances have achieved impressive performance by identifying clean labels and correcting corrupted labels for training.However, the current approaches rely heavily on the model’s predictions and evaluate each sample independently without considering either the global or local structure of the sample distribution.These limitations typically result in a suboptimal solution for the identification and correction processes, which eventually leads to models overfitting to incorrect labels.In this paper, we propose a novel optimal transport (OT) formulation, called Curriculum and Structure-aware Optimal Transport (CSOT). CSOT concurrently considers the inter- and intra-distribution structure of the samples to construct a robust denoising and relabeling allocator.During the training process, the allocator incrementally assigns reliable labels to a fraction of the samples with the highest confidence. These labels have both global discriminability and local coherence.Notably, CSOT is a new OT formulation with a nonconvex objective function and curriculum constraints, so it is not directly compatible with classical OT solvers. Here, we develop a lightspeed computational method that involves a scaling iteration within a generalized conditional gradient framework to solve CSOT efficiently.Extensive experiments demonstrate the superiority of our method over the current state-of-the-arts in LNL.</details>

### [Spotlight] Can semi-supervised learning use all the data effectively? A lower bound perspective

**Authors:** Alexandru Tifrea, Gizem Yüce, Amartya Sanyal, Fanny Yang

**<details><summary>Abstract**</summary>

Prior theoretical and empirical works have established that semi-supervised learning algorithms can leverage the unlabeled data to improve over the labeled sample complexity of supervised learning (SL) algorithms. However, existing theoretical work focuses on regimes where the unlabeled data is sufficient to learn a good decision boundary using unsupervised learning (UL) alone. This begs the question: Can SSL algorithms simultaneously improve upon both UL and SL? To this end, we derive a tight lower bound for 2-Gaussian mixture models that explicitly depends on the labeled and the unlabeled dataset size as well as the signal-to-noise ratio of the mixture distribution. Surprisingly, our result implies that no SSL algorithm improves upon the minimax-optimal statistical error rates of SL or UL algorithms for these distributions. Nevertheless, in our real-world experiments, SSL algorithms can often outperform UL and SL algorithms. In summary, our work suggests that while it is possible to prove the performance gains of SSL algorithms, this would require careful tracking of constants in the theoretical analysis.</details>

### Causal Context Connects Counterfactual Fairness to Robust Prediction and Group Fairness

**Authors:** Jacy Anthis, Victor Veitch

**<details><summary>Abstract**</summary>

Counterfactual fairness requires that a person would have been classified in the same way by an AI or other algorithmic system if they had a different protected class, such as a different race or gender. This is an intuitive standard, as reflected in the U.S. legal system, but its use is limited because counterfactuals cannot be directly observed in real-world data. On the other hand, group fairness metrics (e.g., demographic parity or equalized odds) are less intuitive but more readily observed. In this paper, we use \textit{causal context} to bridge the gaps between counterfactual fairness, robust prediction, and group fairness. First, we motivate counterfactual fairness by showing that there is not necessarily a fundamental trade-off between fairness and accuracy because, under plausible conditions, the counterfactually fair predictor is in fact accuracy-optimal in an unbiased target distribution. Second, we develop a correspondence between the causal graph of the data-generating process and which, if any, group fairness metrics are equivalent to counterfactual fairness. Third, we show that in three common fairness contexts—measurement error, selection on label, and selection on predictors—counterfactual fairness is equivalent to demographic parity, equalized odds, and calibration, respectively. Counterfactual fairness can sometimes be tested by measuring relatively simple group fairness metrics.</details>

### Cause-Effect Inference in Location-Scale Noise Models: Maximum Likelihood vs. Independence Testing

**Authors:** Xiangyu Sun, Oliver Schulte

**<details><summary>Abstract**</summary>

A fundamental problem of causal discovery is cause-effect inference, to learn the correct causal direction between two random variables. Significant progress has been made through modelling the effect as a function of its cause and a noise term, which allows us to leverage assumptions about the generating function class. The recently introduced heteroscedastic location-scale noise functional models (LSNMs) combine expressive power with identifiability guarantees. LSNM model selection based on maximizing likelihood achieves state-of-the-art accuracy, when the noise distributions are correctly specified. However, through an extensive empirical evaluation, we demonstrate that the accuracy deteriorates sharply when the form of the noise distribution is misspecified by the user. Our analysis shows that the failure occurs mainly when the conditional variance in the anti-causal direction is smaller than that in the causal direction. As an alternative, we find that causal model selection through residual independence testing is much more robust to noise misspecification and misleading conditional variance.</details>

### Censored Sampling of Diffusion Models Using 3 Minutes of Human Feedback

**Authors:** TaeHo Yoon, Kibeom Myoung, Keon Lee, Jaewoong Cho, Albert No, Ernest Ryu

**<details><summary>Abstract**</summary>

Diffusion models have recently shown remarkable success in high-quality image generation. Sometimes, however, a pre-trained diffusion model exhibits partial misalignment in the sense that the model can generate good images, but it sometimes outputs undesirable images. If so, we simply need to prevent the generation of the bad images, and we call this task censoring. In this work, we present censored generation with a pre-trained diffusion model using a reward model trained on minimal human feedback. We show that censoring can be accomplished with extreme human feedback efficiency and that labels generated with a mere few minutes of human feedback are sufficient.</details>

### Certification of Distributional Individual Fairness

**Authors:** Matthew Wicker, Vihari Piratla, Adrian Weller

**<details><summary>Abstract**</summary>

Providing formal guarantees of algorithmic fairness is of paramount importance to socially responsible deployment of machine learning algorithms. In this work, we study formal guarantees, i.e., certificates, for individual fairness (IF) of neural networks. We start by introducing a novel convex approximation of IF constraints that exponentially decreases the computational cost of providing formal guarantees of local individual fairness. We highlight that prior methods are constrained by their focus on global IF certification and can therefore only scale to models with a few dozen hidden neurons, thus limiting their practical impact. We propose to certify \textit{distributional} individual fairness which ensures that for a given empirical distribution and all distributions within a $\gamma$-Wasserstein ball, the neural network has guaranteed individually fair predictions. Leveraging developments in quasi-convex optimization, we provide novel and efficient certified bounds on distributional individual fairness and show that our method allows us to certify and regularize neural networks that are several orders of magnitude larger than those considered by prior works. Moreover, we study real-world distribution shifts and find our bounds to be a scalable, practical, and sound source of IF guarantees.</details>

### [Oral] Characteristic Circuits

**Authors:** Zhongjie Yu, Martin Trapp, Kristian Kersting

**Oral Presentation:** Tu, Dec 12, 08:30 -- Oral 1C

**<details><summary>Abstract**</summary>

In many real-world scenarios it is crucial to be able to reliably and efficiently reason under uncertainty while capturing complex relationships in data.  Probabilistic circuits (PCs), a prominent family of tractable probabilistic models, offer a remedy to this challenge by composing simple, tractable distributions into a high-dimensional probability distribution.   However, learning PCs on heterogeneous data is challenging and densities of some parametric distributions are not available in closed form, limiting their potential use.   We introduce characteristic circuits (CCs), a family of tractable probabilistic models providing a unified formalization of distributions over heterogeneous data in the spectral domain.  The one-to-one relationship between characteristic functions and probability measures enables us to learn high-dimensional distributions on heterogeneous data domains and facilitates efficient probabilistic inference even when no closed-form density function is available.   We show that the structure and parameters of CCs can be learned efficiently from the data and find that CCs outperform state-of-the-art density estimators for heterogeneous data domains on common benchmark data sets.</details>

### Characterizing Graph Datasets for Node Classification: Homophily-Heterophily Dichotomy and Beyond

**Authors:** Oleg Platonov, Denis Kuznedelev, Artem Babenko, Liudmila Prokhorenkova

**<details><summary>Abstract**</summary>

Homophily is a graph property describing the tendency of edges to connect similar nodes; the opposite is called heterophily. It is often believed that heterophilous graphs are challenging for standard message-passing graph neural networks (GNNs), and much effort has been put into developing efficient methods for this setting. However, there is no universally agreed-upon measure of homophily in the literature. In this work, we show that commonly used homophily measures have critical drawbacks preventing the comparison of homophily levels across different datasets. For this, we formalize desirable properties for a proper homophily measure and verify which measures satisfy which properties. In particular, we show that a measure that we call adjusted homophily satisfies more desirable properties than other popular homophily measures while being rarely used in graph machine learning literature. Then, we go beyond the homophily-heterophily dichotomy and propose a new characteristic that allows one to further distinguish different sorts of heterophily. The proposed label informativeness (LI) characterizes how much information a neighbor's label provides about a node's label. We prove that this measure satisfies important desirable properties. We also observe empirically that LI better agrees with GNN performance compared to homophily measures, which confirms that it is a useful characteristic of the graph structure.</details>

### Cheaply Estimating Inference Efficiency Metrics for Autoregressive Transformer Models

**Authors:** Deepak Narayanan, Keshav Santhanam, Peter Henderson, Rishi Bommasani, Tony Lee, Percy Liang

**<details><summary>Abstract**</summary>

Large language models (LLMs) are highly capable but also computationally expensive. Characterizing the _fundamental tradeoff_ between inference efficiency and model capabilities requires a metric that is comparable across models from different providers.Unfortunately, raw runtimes measured through black-box APIs do not satisfy this property: model providers can implement software and hardware optimizations orthogonal to the model, and shared infrastructure introduces performance contention.We propose a new metric for inference efficiency called _idealized runtime_, that puts models on equal footing as though they were served on uniform hardware and software without performance contention, and a cost model to efficiently estimate this metric for autoregressive Transformer models.We also propose variants of the idealized runtime that incorporate the number and type of accelerators needed to serve the model.Using these metrics, we compare ten LLMs developed in 2022 to provide the first analysis of inference efficiency-capability tradeoffs; we make several observations from this analysis, including the fact that the superior inference runtime performance of certain APIs is often a byproduct of optimizations within the API rather than the underlying model. Our code is open sourced at https://github.com/stanford-crfm/helm-efficiency.</details>

### ChessGPT: Bridging Policy Learning and Language Modeling

**Authors:** Xidong Feng, Yicheng Luo, Ziyan Wang, Hongrui Tang, Mengyue Yang, Kun Shao, David Mguni, Yali Du, Jun Wang

**<details><summary>Abstract**</summary>

When solving decision-making tasks, humans typically depend on information from two key sources: (1) Historical policy data, which provides interaction replay from the environment, and (2) Analytical insights in natural language form, exposing the invaluable thought process or strategic considerations. Despite this, the majority of preceding research focuses on only one source: they either use historical replay exclusively to directly learn policy or value functions, or engaged in language model training utilizing mere language corpus. In this paper, we argue that a powerful autonomous agent should cover both sources. Thus, we propose ChessGPT, a GPT model bridging policy learning and language modeling by integrating data from these two sources in Chess games. Specifically, we build a large-scale game and language dataset related to chess. Leveraging the dataset, we showcase two model examples ChessCLIP and ChessGPT, integrating policy learning and language modeling. Finally, we propose a full evaluation framework for evaluating language model's chess ability. Experimental results validate our model and dataset's effectiveness. We open source our code, model, and dataset at https://github.com/waterhorse1/ChessGPT.</details>

### CityRefer: Geography-aware 3D Visual Grounding Dataset on  City-scale Point Cloud Data

**Authors:** Taiki Miyanishi, Fumiya Kitamori, Shuhei Kurita, Jungdae Lee, Motoaki Kawanabe, Nakamasa Inoue

**<details><summary>Abstract**</summary>

City-scale 3D point cloud is a promising way to express detailed and complicated outdoor structures. It encompasses both the appearance and geometry features of segmented city components, including cars, streets, and buildings that can be utilized for attractive applications such as user-interactive navigation of autonomous vehicles and drones. However, compared to the extensive text annotations available for images and indoor scenes, the scarcity of text annotations for outdoor scenes poses a significant challenge for achieving these applications. To tackle this problem, we introduce the CityRefer dataset for city-level visual grounding. The dataset consists of 35k natural language descriptions of 3D objects appearing in SensatUrban city scenes and 5k landmarks labels synchronizing with OpenStreetMap. To ensure the quality and accuracy of the dataset, all descriptions and labels in the CityRefer dataset are manually verified. We also have developed a baseline system that can learn encoded language descriptions, 3D object instances, and geographical information about the city's landmarks to perform visual grounding on the CityRefer dataset. To the best of our knowledge, the CityRefer dataset is the largest city-level visual grounding dataset for localizing specific 3D objects.</details>

### Classical Simulation of Quantum Circuits: Parallel Environments and Benchmark

**Authors:** Xiao-Yang Liu, Zeliang Zhang

**<details><summary>Abstract**</summary>

Google's  quantum supremacy announcement has received broad questions from academia and industry due to the debatable estimate of 10,000 years' running time for the classical simulation task on the Summit supercomputer. Has quantum supremacy already come? Or will it come in one or two decades later? To avoid hasty advertisements of  quantum supremacy by tech giants or quantum startups and eliminate the cost of dedicating a team to the classical simulation task, we advocate an open-source approach to maintain a trustable benchmark performance.  In this paper, we take a reinforcement learning approach for the classical simulation of quantum circuits and demonstrate its great potential by reporting an estimated simulation time of less than 4 days, a speedup of 5.40x over the state-of-the-art method.  Specifically, we formulate the classical simulation task as a tensor network contraction ordering problem using the K-spin Ising model and employ a novel Hamiltonina-based reinforcement learning algorithm. Then, we establish standard criteria to evaluate the performance of classical simulation of quantum circuits.  We develop a dozen of massively parallel environments to simulate quantum circuits.  We open-source our parallel gym environments and benchmarks. We hope the AI/ML community and quantum physics community will collaborate to maintain reference curves for validating an unequivocal first demonstration of empirical quantum supremacy.</details>

### ClimateSet: A Large-Scale Climate Model Dataset for Machine Learning

**Authors:** Julia Kaltenborn, Charlotte Lange, Venkatesh Ramesh, Philippe Brouillard, Yaniv Gurwicz, Chandni Nagda, Jakob Runge, Peer Nowack, David Rolnick

**<details><summary>Abstract**</summary>

Climate models have been key for assessing the impact of climate change and simulating future climate scenarios. The machine learning (ML) community has taken an increased interest in supporting climate scientists’ efforts on various tasks such as climate model emulation, downscaling, and prediction tasks. Many of those tasks have been addressed on datasets created with single climate models. However, both the climate science and ML communities have suggested that to address those tasks at scale, we need large, consistent, and ML-ready climate model datasets. Here, we introduce ClimateSet, a dataset containing the inputs and outputs of 36 climate models from the Input4MIPs and CMIP6 archives. In addition, we provide a modular dataset pipeline for retrieving and preprocessing additional climate models and scenarios. We showcase the potential of our dataset by using it as a benchmark for ML-based climate model emulation. We gain new insights about the performance and generalization capabilities of the different ML models by analyzing their performance across different climate models. Furthermore, the dataset can be used to train an ML emulator on several climate models instead of just one. Such a “super emulator” can quickly project new climate change scenarios, complementing existing scenarios already provided to policymakers. We believe ClimateSet will create the basis needed for the ML community to tackle climate-related tasks at scale.</details>

### Cola: A Benchmark for Compositional Text-to-image Retrieval

**Authors:** Arijit Ray, Filip Radenovic, Abhimanyu Dubey, Bryan Plummer, Ranjay Krishna, Kate Saenko

**<details><summary>Abstract**</summary>

Compositional reasoning is a hallmark of human visual intelligence. Yet, despite the size of large vision-language models, they struggle to represent simple compositions by combining objects with their attributes. To measure this lack of compositional capability, we design Cola, a text-to-image retrieval benchmark to Compose Objects Localized with Attributes. To solve Cola, a model must retrieve images with the correct configuration of attributes and objects and avoid choosing a distractor image with the same objects and attributes but in the wrong configuration. Cola contains about 1.2k composed queries of 168 objects and 197 attributes on around 30K images. Our human evaluation finds that Cola is 83.33% accurate, similar to contemporary compositionality benchmarks. Using Cola as a testbed, we explore empirical modeling designs to adapt pre-trained vision-language models to reason compositionally. We explore 6 adaptation strategies on 2 seminal vision-language models, using compositionality-centric test benchmarks - Cola and CREPE. We find the optimal adaptation strategy is to train a multi-modal attention layer that jointly attends over the frozen pre-trained image and language features. Surprisingly, training multimodal layers on CLIP performs better than tuning a larger FLAVA model with already pre-trained multimodal layers. Furthermore, our adaptation strategy improves CLIP and FLAVA to comparable levels, suggesting that training multimodal layers using contrastive attribute-object data is key, as opposed to using them pre-trained. Lastly, we show that Cola is harder than a closely related contemporary benchmark, CREPE, since simpler fine-tuning strategies without multimodal layers suffice on CREPE, but not on Cola. However, we still see a significant gap between our best adaptation and human accuracy, suggesting considerable room for further research. Project page: https://cs-people.bu.edu/array/research/cola/</details>

### Collapsed Inference for Bayesian Deep Learning

**Authors:** Zhe Zeng, Guy Van den Broeck

**<details><summary>Abstract**</summary>

Bayesian neural networks (BNNs) provide a formalism to quantify and calibrate uncertainty in deep learning. Current inference approaches for BNNs often resort to few-sample estimation for scalability, which can harm predictive performance, while its alternatives tend to be computationally prohibitively expensive. We tackle this challenge by revealing a previously unseen connection between inference on BNNs and volume computation problems. With this observation, we introduce a novel collapsed inference scheme that performs Bayesian model averaging using collapsed samples. It improves over a Monte-Carlo sample by limiting sampling to a subset of the network weights while pairing it with some closed-form conditional distribution over the rest. A collapsed sample represents uncountably many models drawn from the approximate posterior and thus yields higher sample efficiency. Further, we show that the marginalization of a collapsed sample can be solved analytically and efficiently despite the non-linearity of neural networks by leveraging existing volume computation solvers. Our proposed use of collapsed samples achieves a balance between scalability and accuracy. On various regression and classification tasks, our collapsed Bayesian deep learning approach demonstrates significant improvements over existing methods and sets a new state of the art in terms of uncertainty estimation as well as predictive performance.</details>

### Computational Guarantees for Doubly Entropic Wasserstein Barycenters

**Authors:** Tomas Vaskevicius, Lénaïc Chizat

**<details><summary>Abstract**</summary>

We study the computation of doubly regularized Wasserstein barycenters, a recently introduced family of entropic barycenters governed by inner and outer regularization strengths. Previous research has demonstrated that various regularization parameter choices unify several notions of entropy-penalized barycenters while also revealing new ones, including a special case of debiased barycenters.  In this paper, we propose and analyze an algorithm for computing doubly regularized Wasserstein barycenters. Our procedure builds on damped Sinkhorn iterations followed by exact maximization/minimization steps and guarantees convergence for any choice of regularization parameters. An inexact variant of our algorithm, implementable using approximate Monte Carlo sampling, offers the first non-asymptotic convergence guarantees for approximating Wasserstein barycenters between discrete point clouds in the free-support/grid-free setting.</details>

### ConRad: Image Constrained Radiance Fields for 3D Generation from a Single Image

**Authors:** Senthil Purushwalkam, Nikhil Naik

**<details><summary>Abstract**</summary>

We present a novel method for reconstructing 3D objects from a single RGB image. Our method leverages the latest image generation models to infer the hidden 3D structure while remaining faithful to the input image. While existing methods obtain impressive results in generating 3D models from text prompts, they do not provide an easy approach for conditioning on input RGB data. Naive extensions of these methods often lead to improper alignment in appearance between the input image and the 3D reconstructions. We address these challenges by introducing Image Constrained Radiance Fields (ConRad), a novel variant of neural radiance fields. ConRad is an efficient 3D representation that explicitly captures the appearance of an input image in one viewpoint. We propose a training algorithm that leverages the single RGB image in conjunction with pretrained Diffusion Models to optimize the parameters of a ConRad representation. Extensive experiments show that ConRad representations can simplify preservation of image details while producing a realistic 3D reconstruction. Compared to existing state-of-the-art baselines, we show that our 3D reconstructions remain more faithful to the input and produce more consistent 3D models while demonstrating significantly improved quantitative performance on a ShapeNet object benchmark.</details>

### Concept Algebra for (Score-Based) Text-Controlled Generative Models

**Authors:** Zihao Wang, Lin Gui, Jeffrey Negrea, Victor Veitch

**<details><summary>Abstract**</summary>

This paper concerns the structure of learned representations in text-guided generative models, focusing on score-based models. A key property of such models is that they can compose disparate concepts in a 'disentangled' manner.This suggests these models have internal representations that encode concepts in a 'disentangled' manner. Here, we focus on the idea that concepts are encoded as subspaces of some representation space.  We formalize what this means, show there's a natural choice for the representation, and develop a simple method for identifying the part of the representation corresponding to a given concept. In particular, this allows us to manipulate the concepts expressed by the model through algebraic manipulation of the representation. We demonstrate the idea with examples using Stable Diffusion.</details>

### [Spotlight] Conditional independence testing under misspecified inductive biases

**Authors:** Felipe Maia Polo, Yuekai Sun, Moulinath Banerjee

**<details><summary>Abstract**</summary>

Conditional independence (CI) testing is a fundamental and challenging task in modern statistics and machine learning. Many modern methods for CI testing rely on powerful supervised learning methods to learn regression functions or Bayes predictors as an intermediate step; we refer to this class of tests as regression-based tests. Although these methods are guaranteed to control Type-I error when the supervised learning methods accurately estimate the regression functions or Bayes predictors of interest, their behavior is less understood when they fail due to misspecified inductive biases; in other words, when the employed models are not flexible enough or when the training algorithm does not induce the desired predictors. Then, we study the performance of regression-based CI tests under misspecified inductive biases. Namely, we propose new approximations or upper bounds for the testing errors of three regression-based tests that depend on misspecification errors. Moreover, we introduce the Rao-Blackwellized Predictor Test (RBPT), a regression-based CI test robust against misspecified inductive biases. Finally, we conduct experiments with artificial and real data, showcasing the usefulness of our theory and methods.</details>

### Connected Superlevel Set in (Deep) Reinforcement Learning and its Application to Minimax Theorems

**Authors:** Sihan Zeng, Thinh Doan, Justin Romberg

**<details><summary>Abstract**</summary>

The aim of this paper is to improve the understanding of the optimization landscape for policy optimization problems in reinforcement learning. Specifically, we show that the superlevel set of the objective function with respect to the policy parameter is always a connected set both in the tabular setting and under policies represented by a class of neural networks. In addition, we show that the optimization objective as a function of the policy parameter and reward satisfies a stronger “equiconnectedness” property. To our best knowledge, these are novel and previously unknown discoveries.We present an application of the connectedness of these superlevel sets to the derivation of minimax theorems for robust reinforcement learning. We show that any minimax optimization program which is convex on one side and is equiconnected on the other side observes the minimax equality (i.e. has a Nash equilibrium). We find that this exact structure is exhibited by an interesting class of robust reinforcement learning problems under an adversarial reward attack, and the validity of its minimax equality immediately follows. This is the first time such a result is established in the literature.</details>

### Conservative Offline Policy Adaptation in Multi-Agent Games

**Authors:** Chengjie Wu, Pingzhong Tang, Jun Yang, Yujing Hu, Tangjie Lv, Changjie Fan, Chongjie Zhang

**<details><summary>Abstract**</summary>

Prior research on policy adaptation in multi-agent games has often relied on online interaction with the target agent in training, which can be expensive and impractical in real-world scenarios. Inspired by recent progress in offline reinforcement learn- ing, this paper studies offline policy adaptation, which aims to utilize the target agent’s behavior data to exploit its weakness or enable effective cooperation. We investigate its distinct challenges of distributional shift and risk-free deviation, and propose a novel learning objective, conservative offline adaptation, that optimizes the worst-case performance against any dataset consistent proxy models. We pro- pose an efficient algorithm called Constrained Self-Play (CSP) that incorporates dataset information into regularized policy learning. We prove that CSP learns a near-optimal risk-free offline adaptation policy upon convergence. Empirical results demonstrate that CSP outperforms non-conservative baselines in various environments, including Maze, predator-prey, MuJoCo, and Google Football.</details>

### Constructing Non-isotropic Gaussian Diffusion Model Using Isotropic Gaussian Diffusion Model for Image Editing

**Authors:** Xi Yu, Xiang Gu, Haozhi Liu, Jian Sun

**<details><summary>Abstract**</summary>

Score-based diffusion models (SBDMs) have achieved state-of-the-art results in image generation. In this paper, we propose a Non-isotropic Gaussian Diffusion Model (NGDM) for image editing, which requires editing the source image while preserving the image regions irrelevant to the editing task. We construct NGDM by adding independent Gaussian noises with different variances to different image pixels. Instead of specifically training the NGDM, we rectify the NGDM into an isotropic Gaussian diffusion model with different pixels having different total forward diffusion time. We propose to reverse the diffusion by designing a sampling method that starts at different time for different pixels for denoising to generate images using the pre-trained isotropic Gaussian diffusion model. Experimental results show that NGDM achieves state-of-the-art performance for image editing tasks, considering the trade-off between the fidelity to the source image and alignment with the desired editing target.</details>

### Contextual Bandits and Imitation Learning with Preference-Based Active Queries

**Authors:** Ayush Sekhari, Karthik Sridharan, Wen Sun, Runzhe Wu

**<details><summary>Abstract**</summary>

We consider the problem of contextual bandits and imitation learning, where the learner lacks direct knowledge of the executed action's reward. Instead, the learner can actively request the expert at each round to compare two actions and receive noisy preference feedback. The learner's objective is two-fold: to minimize regret associated with the executed actions, while simultaneously, minimizing the number of comparison queries made to the expert. In this paper, we assume that the learner has access to a function class that can represent the expert's preference model under appropriate link functions and present an algorithm that leverages an online regression oracle with respect to this function class. For the contextual bandit setting, our algorithm achieves a regret bound that combines the best of both worlds, scaling as $O(\min\\{\sqrt{T}, d/\Delta\\})$, where $T$ represents the number of interactions, $d$ represents the eluder dimension of the function class, and $\Delta$ represents the minimum preference of the optimal action over any suboptimal action under all contexts. Our algorithm does not require the knowledge of $\Delta$, and the obtained regret bound is comparable to what can be achieved in the standard contextual bandits setting where the learner observes reward signals at each round. Additionally, our algorithm makes only $O(\min\\{T, d^2/\Delta^2\\})$ queries to the expert. We then extend our algorithm to the imitation learning setting, where the agent engages with an unknown environment in episodes of length $H$, and provide similar guarantees regarding regret and query complexity. Interestingly, with preference-based feedback, our imitation learning algorithm can learn a policy outperforming a sub-optimal expert, matching the result from interactive imitation learning algorithms [Ross and Bagnell, 2014] that require access to the expert's actions and also reward signals.</details>

### ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling

**Authors:** Yuqi Chen, Kan Ren, Yansen Wang, Yuchen Fang, Weiwei Sun, Dongsheng Li

**<details><summary>Abstract**</summary>

Modeling continuous-time dynamics on irregular time series is critical to account for data evolution and correlations that occur continuously. Traditional methods including recurrent neural networks or Transformer models leverage inductive bias via powerful neural architectures to capture complex patterns. However, due to their discrete characteristic, they have limitations in generalizing to continuous-time data paradigms. Though neural ordinary differential equations (Neural ODEs) and their variants have shown promising results in dealing with irregular time series, they often fail to capture the intricate correlations within these sequences. It is challenging yet demanding to concurrently model the relationship between input data points and capture the dynamic changes of the continuous-time system. To tackle this problem, we propose ContiFormer that extends the relation modeling of vanilla Transformer to the continuous-time domain, which explicitly incorporates the modeling abilities of continuous dynamics of Neural ODEs with the attention mechanism of Transformers. We mathematically characterize the expressive power of ContiFormer and illustrate that, by curated designs of function hypothesis, many Transformer variants specialized in irregular time series modeling can be covered as a special case of ContiFormer. A wide range of experiments on both synthetic and real-world datasets have illustrated the superior modeling capacities and prediction performance of ContiFormer on irregular time series data. The project link is https://seqml.github.io/contiformer/.</details>

### Continuous-Time Functional Diffusion Processes

**Authors:** Giulio Franzese, Giulio Corallo, Simone Rossi, Markus Heinonen, Maurizio Filippone, Pietro Michiardi

**<details><summary>Abstract**</summary>

We introduce Functional Diffusion Processes (FDPs), which generalize score-based diffusion models to infinite-dimensional function spaces. FDPs require a new mathematical framework to describe the forward and backward dynamics, and several extensions to derive practical training objectives. These include infinite-dimensional versions of Girsanov theorem, in order to be able to compute an ELBO, and of the sampling theorem, in order to guarantee that functional evaluations in a countable set of points are equivalent to infinite-dimensional functions. We use FDPs to build a new breed of generative models in function spaces, which do not require specialized network architectures, and that can work with any kind of continuous data.Our results on real data show that FDPs achieve high-quality image generation, using a simple MLP architecture with orders of magnitude fewer parameters than existing diffusion models.</details>

### Contrastive Modules with Temporal Attention for Multi-Task Reinforcement Learning

**Authors:** Siming Lan, Rui Zhang, Qi Yi, Jiaming Guo, Shaohui Peng, Yunkai Gao, Fan Wu, Ruizhi Chen, Zidong Du, Xing Hu, xishan zhang, Ling Li, Yunji Chen

**<details><summary>Abstract**</summary>

In the field of multi-task reinforcement learning, the modular principle, which involves specializing functionalities into different modules and combining them appropriately, has been widely adopted as a promising approach to prevent the negative transfer problem that performance degradation due to conflicts between tasks. However, most of the existing multi-task RL methods only combine shared modules at the task level, ignoring that there may be conflicts within the task. In addition, these methods do not take into account that without constraints, some modules may learn similar functions, resulting in restricting the model's expressiveness and generalization capability of modular methods.In this paper, we propose the Contrastive Modules with Temporal Attention(CMTA) method to address these limitations. CMTA constrains the modules to be different from each other by contrastive learning and combining shared modules at a finer granularity than the task level with temporal attention, alleviating the negative transfer within the task and improving the generalization ability and the performance for multi-task RL.We conducted the experiment on Meta-World, a multi-task RL benchmark containing various robotics manipulation tasks. Experimental results show that CMTA outperforms learning each task individually for the first time and achieves substantial performance improvements over the baselines.</details>

### Coupled Reconstruction of Cortical Surfaces by Diffeomorphic Mesh Deformation

**Authors:** Hao Zheng, Hongming Li, Yong Fan

**<details><summary>Abstract**</summary>

Accurate reconstruction of cortical surfaces from brain magnetic resonance images (MRIs) remains a challenging task due to the notorious partial volume effect in brain MRIs and the cerebral cortex's thin and highly folded patterns. Although many promising deep learning-based cortical surface reconstruction methods have been developed, they typically fail to model the interdependence between inner (white matter) and outer (pial) cortical surfaces, which can help generate cortical surfaces with spherical topology. To robustly reconstruct the cortical surfaces with topological correctness, we develop a new deep learning framework to jointly reconstruct the inner, outer, and their in-between (midthickness) surfaces and estimate cortical thickness directly from 3D MRIs. Our method first estimates the midthickness surface and then learns three diffeomorphic flows jointly to optimize the midthickness surface and deform it inward and outward to the inner and outer cortical surfaces respectively, regularized by topological correctness. Our method also outputs a cortex thickness value for each surface vertex, estimated from its diffeomorphic deformation trajectory. Our method has been evaluated on two large-scale neuroimaging datasets, including ADNI and OASIS, achieving state-of-the-art cortical surface reconstruction performance in terms of accuracy, surface regularity, and computation efficiency.</details>

### Cross-modal Prompts: Adapting Large Pre-trained Models for Audio-Visual Downstream Tasks

**Authors:** Haoyi Duan, Yan Xia, Zhou Mingze, Li Tang, Jieming Zhu, Zhou Zhao

**<details><summary>Abstract**</summary>

In recent years, the deployment of large-scale pre-trained models in audio-visual downstream tasks has yielded remarkable outcomes. However, these models, primarily trained on single-modality unconstrained datasets, still encounter challenges in feature extraction for multi-modal tasks, leading to suboptimal performance. This limitation arises due to the introduction of irrelevant modality-specific information during encoding, which adversely affects the performance of downstream tasks. To address this challenge, this paper proposes a novel Dual-Guided Spatial-Channel-Temporal (DG-SCT) attention mechanism. This mechanism leverages audio and visual modalities as soft prompts to dynamically adjust the parameters of pre-trained models based on the current multi-modal input features. Specifically, the DG-SCT module incorporates trainable cross-modal interaction layers into pre-trained audio-visual encoders, allowing adaptive extraction of crucial information from the current modality across spatial, channel, and temporal dimensions, while preserving the frozen parameters of large-scale pre-trained models. Experimental evaluations demonstrate that our proposed model achieves state-of-the-art results across multiple downstream tasks, including AVE, AVVP, AVS, and AVQA. Furthermore, our model exhibits promising performance in challenging few-shot and zero-shot scenarios. The source code and pre-trained models are available at https://github.com/haoyi-duan/DG-SCT.</details>

### CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion

**Authors:** Yangruibo Ding, Zijian Wang, Wasi Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, Bing Xiang

**<details><summary>Abstract**</summary>

Code completion models have made significant progress in recent years, yet current popular evaluation datasets, such as HumanEval and MBPP, predominantly focus on code completion tasks within a single file. This over-simplified setting falls short of representing the real-world software development scenario where repositories span multiple files with numerous cross-file dependencies, and accessing and understanding cross-file context is often required to complete the code correctly. To fill in this gap, we propose CrossCodeEval, a diverse and multilingual code completion benchmark that necessitates an in-depth cross-file contextual understanding to complete the code accurately. CrossCodeEval is built on a diverse set of real-world, open-sourced, permissively-licensed repositories in four popular programming languages: Python, Java, TypeScript, and C#. To create examples that strictly require cross-file context for accurate completion, we propose a straightforward yet efficient static-analysis-based approach to pinpoint the use of cross-file context within the current file. Extensive experiments on state-of-the-art code language models like CodeGen and StarCoder demonstrate that CrossCodeEval is extremely challenging when the relevant cross-file context is absent, and we see clear improvements when adding these context into the prompt. However, despite such improvements,  the pinnacle of performance remains notably unattained even with the highest-performing model,  indicating that CrossCodeEval is also capable of assessing model's capability in leveraging extensive context to make better code completion. Finally, we benchmarked various methods in retrieving cross-file context, and show that CrossCodeEval can also be used to measure the capability of code retrievers.</details>

### Curve Your Enthusiasm: Concurvity Regularization in Differentiable Generalized Additive Models

**Authors:** Julien Siems, Konstantin Ditschuneit, Winfried Ripken, Alma Lindborg, Maximilian Schambach, Johannes Otterbach, Martin Genzel

**<details><summary>Abstract**</summary>

Generalized Additive Models (GAMs) have recently experienced a resurgence in popularity due to their interpretability, which arises from expressing the target value as a sum of non-linear transformations of the features. Despite the current enthusiasm for GAMs, their susceptibility to concurvity — i.e., (possibly non-linear) dependencies between the features — has hitherto been largely overlooked. Here, we demonstrate how concurvity can severly impair the interpretability of GAMs and propose a remedy: a conceptually simple, yet effective regularizer which penalizes pairwise correlations of the non-linearly transformed feature variables. This procedure is applicable to any differentiable additive model, such as Neural Additive Models or NeuralProphet, and enhances interpretability by eliminating ambiguities due to self-canceling feature contributions. We validate the effectiveness of our regularizer in experiments on synthetic as well as real-world datasets for time-series and tabular data. Our experiments show that concurvity in GAMs can be reduced without significantly compromising prediction quality, improving interpretability and reducing variance in the feature importances.</details>

### Customizable Image Synthesis with Multiple Subjects

**Authors:** Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli Zhao, Jingren Zhou, Yang Cao

**<details><summary>Abstract**</summary>

Synthesizing images with user-specified subjects has received growing attention due to its practical applications. Despite the recent success in single subject customization, existing algorithms suffer from high training cost and low success rate along with increased number of subjects. Towards controllable image synthesis with multiple subjects as the constraints, this work studies how to efficiently represent a particular subject as well as how to appropriately compose different subjects. We find that the text embedding regarding the subject token already serves as a simple yet effective representation that supports arbitrary combinations without any model tuning. Through learning a residual on top of the base embedding, we manage to robustly shift the raw subject to the customized subject given various text conditions. We then propose to employ layout, a very abstract and easy-to-obtain prior, as the spatial guidance for subject arrangement. By rectifying the activations in the cross-attention map, the layout appoints and separates the location of different subjects in the image, significantly alleviating the interference across them. Using cross-attention map as the intermediary, we could strengthen the signal of target subjects and weaken the signal of irrelevant subjects within a certain region, significantly alleviating the interference across subjects. Both qualitative and quantitative experimental results demonstrate our superiority over state-of-the-art alternatives under a variety of settings for multi-subject customization.</details>

### DELIFFAS: Deformable Light Fields for Fast Avatar Synthesis

**Authors:** Youngjoong Kwon, Lingjie Liu, Henry Fuchs, Marc Habermann, Christian Theobalt

**<details><summary>Abstract**</summary>

Generating controllable and photorealistic digital human avatars is a long-standing and important problem in Vision and Graphics. Recent methods have shown great progress in terms of either photorealism or inference speed while the combination of the two desired properties still remains unsolved. To this end, we propose a novel method, called DELIFFAS, which parameterizes the appearance of the human as a surface light field that is attached to a controllable and deforming human mesh model. At the core, we represent the light field around the human with a deformable two-surface parameterization, which enables fast and accurate inference of the human appearance. This allows perceptual supervision on the full image compared to previous approaches that could only supervise individual pixels or small patches due to their slow runtime. Our carefully designed human representation and supervision strategy leads to state-of-the-art synthesis results and inference time. The video results and code are available at https://vcai.mpi-inf.mpg.de/projects/DELIFFAS.</details>

### DIFFER:Decomposing Individual Reward for Fair Experience Replay in Multi-Agent Reinforcement Learning

**Authors:** Xunhan Hu, Jian Zhao, Wengang Zhou, Ruili Feng, Houqiang Li

**<details><summary>Abstract**</summary>

Cooperative multi-agent reinforcement learning (MARL) is a challenging task, as agents must learn complex and diverse individual strategies from a shared team reward. However, existing methods struggle to distinguish and exploit important individual experiences, as they lack an effective way to decompose the team reward into individual rewards. To address this challenge, we propose DIFFER, a powerful theoretical framework for decomposing individual rewards to enable fair experience replay in MARL.By enforcing the invariance of network gradients, we establish a partial differential equation whose solution yields the underlying individual reward function. The individual TD-error can then be computed from the solved closed-form individual rewards, indicating the importance of each piece of experience in the learning task and guiding the training process. Our method elegantly achieves an equivalence to the original learning framework when individual experiences are homogeneous, while also adapting to achieve more muscular efficiency and fairness when diversity is observed.Our extensive experiments on popular benchmarks validate the effectiveness of our theory and method, demonstrating significant improvements in learning efficiency and fairness. Code is available in supplement material.</details>

### DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction

**Authors:** Mohammadreza Pourreza, Davood Rafiei

**<details><summary>Abstract**</summary>

There is currently a significant gap between the performance of fine-tuned models and prompting approaches using Large Language Models (LLMs) on the challenging task of text-to-SQL, as evaluated on datasets such as Spider. To improve the performance of LLMs in the reasoning process, we study how decomposing the task into smaller sub-tasks can be effective. In particular, we show that breaking down the generation problem into sub-problems and feeding the solutions of those sub-problems into LLMs can be an effective approach for significantly improving their performance.  Our experiments with three LLMs show that this approach consistently improves their simple few-shot performance by roughly 10%, pushing the accuracy of LLMs towards SOTA or surpassing it. On the holdout test set of Spider, the SOTA, in terms of execution accuracy, was 79.9 and the new SOTA at the time of  this writing using our approach is 85.3. Our approach with in-context learning beats many heavily fine-tuned models by at least 5%. Additionally, when evaluated on the BIRD benchmark, our approach achieved an execution accuracy of 55.9%, setting a new SOTA on its holdout test set.</details>

### DISCS: A Benchmark for Discrete Sampling

**Authors:** Katayoon Goshvadi, Haoran Sun, Xingchao Liu, Azade Nova, Ruqi Zhang, Will Grathwohl, Dale Schuurmans, Hanjun Dai

**<details><summary>Abstract**</summary>

Sampling in discrete spaces, with critical applications in simulation and optimization, has recently been boosted by significant advances in gradient-based approaches that exploit modern accelerators like GPUs. However, two key challenges are hindering further advancement in research on discrete sampling. First, since there is no consensus on experimental settings and evaluation setups, the empirical results in different research papers are often not comparable. Second, implementing samplers and target distributions often requires a nontrivial amount of effort in terms of calibration and parallelism. To tackle these challenges, we propose DISCS (DISCrete Sampling), a tailored package and benchmark that supports unified and efficient experiment implementation and evaluations for discrete sampling in three types of tasks: sampling from classical graphical models and energy based generative models, and sampling for solving combinatorial optimization. Throughout the comprehensive evaluations in DISCS, we gained new insights into scalability, design principles for proposal distributions, and lessons for adaptive sampling design. DISCS efficiently implements representative discrete samplers in existing research works as baselines and offers a simple interface that researchers can conveniently add new discrete samplers and directly compare their performance with the benchmark result in a calibrated setup.</details>

### DVSOD: RGB-D Video Salient Object Detection

**Authors:** Jingjing Li, Wei Ji, Size Wang, Wenbo Li, Li cheng

**<details><summary>Abstract**</summary>

Salient object detection (SOD) aims to identify standout elements in a scene, with recent advancements primarily focused on integrating depth data (RGB-D) or temporal data from videos to enhance SOD in complex scenes. However, the unison of two types of crucial information remains largely underexplored due to data constraints. To bridge this gap, we in this work introduce the DViSal dataset, fueling further research in the emerging field of RGB-D video salient object detection (DVSOD). Our dataset features 237 diverse RGB-D videos alongside comprehensive annotations, including object and instance-level markings, as well as bounding boxes and scribbles. These resources enable a broad scope for potential research directions. We also conduct benchmarking experiments using various SOD models, affirming the efficacy of multimodal video input for salient object detection. Lastly, we highlight some intriguing findings and promising future research avenues. To foster growth in this field, our dataset and benchmark results are publicly accessible at: https://dvsod.github.io/.</details>

### Data Market Design through Deep Learning

**Authors:** Sai Srivatsa Ravindranath, Yanchen Jiang, David Parkes

**<details><summary>Abstract**</summary>

The  _data market design_ problem is a problem in economic theory to find a set of signaling schemes (statistical experiments) to maximize expected revenue to the information seller, where each experiment reveals some of the information known to a seller and has a corresponding price. Each buyer has their own decision to make in a world environment, and their subjective expected value for the information associated with a particular experiment comes from the improvement in this decision and depends on their prior and value for different outcomes. In a setting with multiple buyers, a buyer's expected value for an experiment may also depend on the information sold to others. We introduce the application of deep learning for the design of revenue-optimal data markets, looking to expand the frontiers of what can be understood and achieved. Relative to earlier work on deep learning for auction design, we must learn signaling schemes rather than allocation rules and handle  _obedience constraints_  — these arising from modeling the downstream actions of buyers — in addition to incentive constraints on bids.  Our experiments demonstrate that this new deep learning framework can almost precisely replicate all known solutions from theory, expand to more complex settings, and be used to establish the optimality of new designs for data markets and make conjectures in regard to the structure of optimal designs.</details>

### Data-Driven Network Neuroscience: On Data Collection and Benchmark

**Authors:** Jiaxing Xu, Yunhan Yang, David Huang, Sophi Shilpa Gururajapathy, Yiping Ke, Miao Qiao, Alan Wang, Haribalan Kumar, Josh McGeown, Eryn Kwon

**<details><summary>Abstract**</summary>

This paper presents a comprehensive and quality collection of functional human brain network data for potential research in the intersection of neuroscience, machine learning, and graph analytics. Anatomical and functional MRI images have been used to understand the functional connectivity of the human brain and are particularly important in identifying underlying neurodegenerative conditions such as Alzheimer's, Parkinson's, and Autism. Recently, the study of the brain in the form of brain networks using machine learning and graph analytics has become increasingly popular, especially to predict the early onset of these conditions. A brain network, represented as a graph, retains rich structural and positional information that traditional examination methods are unable to capture. However, the lack of publicly accessible brain network data prevents researchers from data-driven explorations. One of the main difficulties lies in the complicated domain-specific preprocessing steps and the exhaustive computation required to convert the data from MRI images into brain networks. We bridge this gap by collecting a large amount of MRI images from public databases and a private source, working with domain experts to make sensible design choices, and preprocessing the MRI images to produce a collection of brain network datasets. The datasets originate from 6 different sources, cover 4 brain conditions, and consist of a total of 2,702 subjects. We test our graph datasets on 12 machine learning models to provide baselines and validate the data quality on a recent graph analysis model. To lower the barrier to entry and promote the research in this interdisciplinary field, we release our brain network data and complete preprocessing details including codes at https://doi.org/10.17608/k6.auckland.21397377 and https://github.com/brainnetuoa/data_driven_network_neuroscience.</details>

### Data-Informed Geometric Space Selection

**Authors:** Shuai Zhang, Wenqi Jiang

**<details><summary>Abstract**</summary>

Geometric representation learning (e.g., hyperbolic and spherical geometry) has proven to be efficacious in solving many intricate machine learning tasks. The fundamental challenge of geometric representation learning lies in aligning the inherent geometric bias with the underlying structure of the data, which is a rarely explored topic in the literature. Existing methods heavily rely on heuristic assumptions on the data structure to decide the type of geometry to be adopted, which often leads to suboptimal performance. This work aims to automate the alignment process via a data-informed strategy such that we optimize model performance with minimal overhead. Specifically, a sparse gating mechanism is employed to enable each input data point $\mathit{p}$ to select $K$ geometric spaces from a given candidate geometric space pool with $N$ ($K</details>

### DataPerf: Benchmarks for Data-Centric AI Development

**Authors:** Mark Mazumder, Colby Banbury, Xiaozhe Yao, Bojan Karlaš, William Gaviria Rojas, Sudnya Diamos, Greg Diamos, Lynn He, Alicia Parrish, Hannah Rose Kirk, Jessica Quaye, Charvi Rastogi, Douwe Kiela, David Jurado, David Kanter, Rafael Mosquera, Will Cukierski, Juan Ciro, Lora Aroyo, Bilge Acun, Lingjiao Chen, Mehul Raje, Max Bartolo, Evan Sabri Eyuboglu, Amirata Ghorbani, Emmett Goodman, Addison Howard, Oana Inel, Tariq Kane, Christine R. Kirkpatrick, D. Sculley, Tzu-Sheng Kuo, Jonas Mueller, Tristan Thrush, Joaquin Vanschoren, Margaret Warren, Adina Williams, Serena Yeung, Newsha Ardalani, Praveen Paritosh, Ce Zhang, James Zou, Carole-Jean Wu, Cody Coleman, Andrew Ng, Peter Mattson, Vijay Janapa Reddi

**<details><summary>Abstract**</summary>

Machine learning research has long focused on models rather than datasets, and prominent datasets are used for common ML tasks without regard to the breadth, difficulty, and faithfulness of the underlying problems. Neglecting the fundamental importance of data has given rise to inaccuracy, bias, and fragility in real-world applications, and research is hindered by saturation across existing dataset benchmarks. In response, we present DataPerf, a community-led benchmark suite for evaluating ML datasets and data-centric algorithms. We aim to foster innovation in data-centric AI through competition, comparability, and reproducibility. We enable the ML community to iterate on datasets, instead of just architectures, and we provide an open, online platform with multiple rounds of challenges to support this iterative development. The first iteration of DataPerf contains five benchmarks covering a wide spectrum of data-centric techniques, tasks, and modalities in vision, speech, acquisition, debugging, and diffusion prompting, and we support hosting new contributed benchmarks from the community. The benchmarks, online evaluation platform, and baseline implementations are open source, and the MLCommons Association will maintain DataPerf to ensure long-term benefits to academia and industry.</details>

### Decentralized Matrix Sensing: Statistical Guarantees and Fast Convergence

**Authors:** Marie Maros, Gesualdo Scutari

**<details><summary>Abstract**</summary>

We explore the matrix sensing problem from near-isotropic linear measurements, distributed across a network of agents modeled as an undirected graph, with no centralized node. We provide the first study of statistical, computational/communication guarantees for a decentralized gradient algorithm that solves the (nonconvex) Burer-Monteiro type decomposition associated to the low-rank matrix estimation. With small random initialization, the algorithm displays an approximate two-phase convergence: (i) a spectral phase that aligns the iterates' column space with the underlying low-rank matrix, mimicking centralized spectral initialization (not directly implementable over networks); and (ii) a local refinement phase that diverts the iterates from certain degenerate saddle points, while ensuring swift convergence to the underlying low-rank matrix. Central to our analysis is a novel "in-network" Restricted Isometry Property which accommodates for the decentralized nature of the optimization, revealing an intriguing interplay between sample complexity and network connectivity, topology, and communication complexity.</details>

### [Oral] DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models

**Authors:** Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Rylan Schaeffer, Sang Truong, Simran Arora, Mantas Mazeika, Dan Hendrycks, Zinan Lin, Yu Cheng, Sanmi Koyejo, Dawn Song, Bo Li

**Oral Presentation:** Tu, Dec 12, 08:30 -- Oral 1B

**<details><summary>Abstract**</summary>

Generative Pre-trained Transformer (GPT) models have exhibited exciting progress in capabilities, capturing the interest of practitioners and the public alike. Yet, while the literature on the trustworthiness of GPT models remains limited, practitioners have proposed employing capable GPT models for sensitive applications to healthcare and finance – where mistakes can be costly. To this end, this work proposes a comprehensive trustworthiness evaluation for large language models with a focus on GPT-4 and GPT-3.5, considering diverse perspectives – including toxicity, stereotype bias, adversarial robustness, out-of-distribution robustness, robustness on adversarial demonstrations, privacy, machine ethics, and fairness. Based on our evaluations, we discover previously unpublished vulnerabilities to trustworthiness threats. For instance, we find that GPT models can be easily misled to generate toxic and biased outputs and leak private information in both training data and conversation history. We also find that although GPT-4 is usually more trustworthy than GPT-3.5 on standard benchmarks, GPT-4 is more vulnerable given jailbreaking system or user prompts, potentially due to the reason that GPT-4 follows the (misleading) instructions more precisely. Our work illustrates a comprehensive trustworthiness evaluation of GPT models and sheds light on the trustworthiness gaps. Our benchmark is publicly available at https://decodingtrust.github.io/.</details>

### Deductive Verification of Chain-of-Thought Reasoning

**Authors:** Zhan Ling, Yunhao Fang, Xuanlin Li, Zhiao Huang, Mingu Lee, Roland Memisevic, Hao Su

**<details><summary>Abstract**</summary>

Large Language Models (LLMs) significantly benefit from Chain-of-thought (CoT) prompting in performing various reasoning tasks. While CoT allows models to produce more comprehensive reasoning processes, its emphasis on intermediate reasoning steps can inadvertently introduce hallucinations and accumulated errors, thereby limiting models’ ability to solve complex reasoning tasks. Inspired by how humans engage in careful and meticulous deductive logical reasoning processes to solve tasks, we seek to enable language models to perform explicit and rigorous deductive reasoning, and also ensure the trustworthiness of their reasoning process through self-verification. However, directly verifying the validity of an entire deductive reasoning process is challenging, even with advanced models like ChatGPT. In light of this, we propose to decompose a reasoning verification process into a series of step-by-step subprocesses, each only receiving their necessary context and premises. To facilitate this procedure, we propose Natural Program, a natural language-based deductive reasoning format. Our approach enables models to generate precise reasoning steps where subsequent steps are more rigorously grounded on prior steps. It also empowers language models to carry out reasoning self-verification in a step-by-step manner. By integrating this verification process into each deductive reasoning stage, we significantly enhance the rigor and trustfulness of generated reasoning steps. Along this process, we also improve the answer correctness on complex reasoning tasks.</details>

### Deep Contract Design via Discontinuous Networks

**Authors:** Tonghan Wang, Paul Duetting, Dmitry Ivanov, Inbal Talgam-Cohen, David Parkes

**<details><summary>Abstract**</summary>

Contract design involves a principal who establishes contractual agreements about payments for outcomes that arise from the actions of an agent. In this paper, we initiate the study of deep learning for the automated design of optimal contracts. We introduce a novel representation: the Discontinuous ReLU (DeLU) network, which models the principal's utility as a discontinuous piecewise affine function of the design of a contract where each piece corresponds to the agent taking a particular action. DeLU networks implicitly learn closed-form expressions for the incentive compatibility constraints of the agent and the utility maximization objective of the principal, and support parallel inference on each piece through linear programming or interior-point methods that solve for optimal contracts. We provide empirical results that demonstrate success in approximating the principal's utility function with a small number of training samples and scaling to find approximately optimal contracts on problems with a large number of actions and outcomes.</details>

### Deep Optimal Transport: A Practical Algorithm for Photo-realistic Image Restoration

**Authors:** Theo Adrai, Guy Ohayon, Michael Elad, Tomer Michaeli

**<details><summary>Abstract**</summary>

We propose an image restoration algorithm that can control the perceptual quality and/or the mean square error (MSE) of any pre-trained model, trading one over the other at test time. Our algorithm is few-shot: Given about a dozen images restored by the model, it can significantly improve the perceptual quality and/or the MSE of the model for newly restored images without further training. Our approach is motivated by a recent theoretical result that links between the minimum MSE (MMSE) predictor and the predictor that minimizes the MSE under a perfect perceptual quality constraint. Specifically, it has been shown that the latter can be obtained by optimally transporting the output of the former, such that its distribution matches that of the source data. Thus, to improve the perceptual quality of a predictor that was originally trained to minimize MSE, we approximate the optimal transport by a linear transformation in the latent space of a variational auto-encoder, which we compute in closed-form using empirical means and covariances. Going beyond the theory, we find that applying the same procedure on models that were initially trained to achieve high perceptual quality, typically improves their perceptual quality even further. And by interpolating the results with the original output of the model, we can improve their MSE on the expense of perceptual quality. We illustrate our method on a variety of degradations applied to general content images with arbitrary dimensions.</details>

### Degraded Polygons Raise Fundamental Questions of Neural Network Perception

**Authors:** Leonard Tang, Dan Ley

**<details><summary>Abstract**</summary>

It is well-known that modern computer vision systems often exhibit behaviors misaligned with those of humans: from adversarial attacks to image corruptions, deeplearning vision models suffer in a variety of settings that humans capably handle. Inlight of these phenomena, here we introduce another, orthogonal perspective studying the human-machine vision gap. We revisit the task of recovering images underdegradation, first introduced over 30 years ago in the Recognition-by-Componentstheory of human vision. Specifically, we study the performance and behavior ofneural networks on the seemingly simple task of classifying regular polygons atvarying orders of degradation along their perimeters. To this end, we implement theAutomated Shape Recoverability Testfor rapidly generating large-scale datasetsof perimeter-degraded regular polygons, modernizing the historically manual creation of image recoverability experiments. We then investigate the capacity ofneural networks to recognize and recover such degraded shapes when initializedwith different priors. Ultimately, we find that neural networks’ behavior on thissimple task conflicts with human behavior, raising a fundamental question of therobustness and learning capabilities of modern computer vision models</details>

### [Spotlight] Demystifying Oversmoothing in Attention-Based Graph Neural Networks

**Authors:** Xinyi Wu, Amir Ajorlou, Zihui Wu, Ali Jadbabaie

**<details><summary>Abstract**</summary>

Oversmoothing in Graph Neural Networks (GNNs) refers to the phenomenon where increasing network depth leads to homogeneous node representations. While previous work has established that Graph Convolutional Networks (GCNs) exponentially lose expressive power, it remains controversial whether the graph attention mechanism can mitigate oversmoothing. In this work, we provide a definitive answer to this question through a rigorous mathematical analysis, by viewing attention-based GNNs as nonlinear time-varying dynamical systems and incorporating tools and techniques from the theory of products of inhomogeneous matrices and the joint spectral radius. We establish that, contrary to popular belief, the graph attention mechanism cannot prevent oversmoothing and loses expressive power exponentially. The proposed framework extends the existing results on oversmoothing for symmetric GCNs to a significantly broader class of GNN models, including random walk GCNs, Graph Attention Networks (GATs) and (graph) transformers. In particular, our analysis accounts for asymmetric, state-dependent and time-varying aggregation operators and a wide range of common nonlinear activation functions, such as ReLU, LeakyReLU, GELU and SiLU.</details>

### Density of States Prediction of Crystalline Materials via Prompt-guided Multi-Modal Transformer

**Authors:** Namkyeong Lee, Heewoong Noh, Sungwon Kim, Dongmin Hyun, Gyoung S. Na, Chanyoung Park

**<details><summary>Abstract**</summary>

The density of states (DOS) is a spectral property of crystalline materials, which provides fundamental insights into various characteristics of the materials.While previous works mainly focus on obtaining high-quality representations of crystalline materials for DOS prediction, we focus on predicting the DOS from the obtained representations by reflecting the nature of DOS: DOS determines the general distribution of states as a function of energy.That is, DOS is not solely determined by the crystalline material but also by the energy levels, which has been neglected in previous works.In this paper, we propose to integrate heterogeneous information obtained from the crystalline materials and the energies via a multi-modal transformer, thereby modeling the complex relationships between the atoms in the crystalline materials and various energy levels for DOS prediction.Moreover, we propose to utilize prompts to guide the model to learn the crystal structural system-specific interactions between crystalline materials and energies.Extensive experiments on two types of DOS, i.e., Phonon DOS and Electron DOS, with various real-world scenarios demonstrate the superiority of DOSTransformer.The source code for DOSTransformer is available at https://github.com/HeewoongNoh/DOSTransformer.</details>

### DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models

**Authors:** XiMing Xing, Chuang Wang, Haitao Zhou, Jing Zhang, Qian Yu, Dong Xu

**<details><summary>Abstract**</summary>

Even though trained mainly on images, we discover that pretrained diffusion models show impressive power in guiding sketch synthesis. In this paper, we present DiffSketcher, an innovative algorithm that creates 	extit{vectorized} free-hand sketches using natural language input. DiffSketcher is developed based on a pre-trained text-to-image diffusion model. It performs the task by directly optimizing a set of Bézier curves with an extended version of the score distillation sampling (SDS) loss, which allows us to use a raster-level diffusion model as a prior for optimizing a parametric vectorized sketch generator. Furthermore, we explore attention maps embedded in the diffusion model for effective stroke initialization to speed up the generation process. The generated sketches demonstrate multiple levels of abstraction while maintaining recognizability, underlying structure, and essential visual details of the subject drawn. Our experiments show that DiffSketcher achieves greater quality than prior work. The code and demo of DiffSketcher can be found at https://ximinng.github.io/DiffSketcher-project/.</details>

### Differentiable Neuro-Symbolic Reasoning on Large-Scale Knowledge Graphs

**Authors:** CHEN SHENGYUAN, Yunfeng Cai, Huang Fang, Xiao Huang, Mingming Sun

**<details><summary>Abstract**</summary>

Knowledge graph (KG) reasoning utilizes two primary techniques, i.e., rule-based and KG-embedding based. The former provides precise inferences, but inferring via concrete rules is not scalable. The latter enables efficient reasoning at the cost of ambiguous inference accuracy. Neuro-symbolic reasoning seeks to amalgamate the advantages of both techniques. The crux of this approach is replacing the predicted existence of all possible triples (i.e., truth scores inferred from rules) with a suitable approximation grounded in embedding representations. However, constructing an effective approximation of all possible triples' truth scores is a challenging task, because it needs to balance the tradeoff between accuracy and efficiency, while compatible with both the rule-based and KG-embedding models. To this end, we proposed a differentiable framework - DiffLogic. Instead of directly approximating all possible triples, we design a tailored filter to adaptively select essential triples based on the dynamic rules and weights. The truth scores assessed by KG-embedding are continuous, so we employ a continuous Markov logic network named probabilistic soft logic (PSL). It employs the truth scores of essential triples to assess the overall agreement among rules, weights, and observed triples. PSL enables end-to-end differentiable optimization, so we can alternately update embedding and weighted rules. On benchmark datasets, we empirically show that DiffLogic surpasses baselines in both effectiveness and efficiency.</details>

### Differentiable Sampling of Categorical Distributions Using the CatLog-Derivative Trick

**Authors:** Lennert De Smet, Emanuele Sansone, Pedro Zuidberg Dos Martires

**<details><summary>Abstract**</summary>

Categorical random variables can faithfully represent the discrete and uncertain aspects of data as part of a discrete latent variable model. Learning in such models necessitates taking gradients with respect to the parameters of the categorical probability distributions, which is often intractable due to their combinatorial nature. A popular technique to estimate these otherwise intractable gradients is the Log-Derivative trick. This trick forms the basis of the well-known REINFORCE gradient estimator and its many extensions. While the Log-Derivative trick allows us to differentiate through samples drawn from categorical distributions, it does not take into account the discrete nature of the distribution itself. Our first contribution addresses this shortcoming by introducing the CatLog-Derivative trick -- a variation of the Log-Derivative trick tailored towards categorical distributions. Secondly, we use the CatLog-Derivative trick to introduce IndeCateR, a novel and unbiased gradient estimator for the important case of products of independent categorical distributions with provably lower variance than REINFORCE. Thirdly, we empirically show that IndeCateR can be efficiently implemented and that its gradient estimates have significantly lower bias and variance for the same number of samples compared to the state of the art.</details>

### Differentiable sorting for censored time-to-event data.

**Authors:** Andre Vauvelle, Benjamin Wild, Roland Eils, Spiros Denaxas

**<details><summary>Abstract**</summary>

Survival analysis is a crucial semi-supervised task in machine learning with significant real-world applications, especially in healthcare. The most common approach to survival analysis, Cox’s partial likelihood, can be interpreted as a ranking model optimized on a lower bound of the concordance index.  We follow these connections further, with listwise ranking losses that allow for a relaxation of the pairwise independence assumption. Given the inherent transitivity of ranking, we explore differentiable sorting networks as a means to introduce a stronger transitive inductive bias during optimization. Despite their potential, current differentiable sorting methods cannot account for censoring, a crucial aspect of many real-world datasets. We propose a novel method, Diffsurv, to overcome this limitation by extending differentiable sorting methods to handle censored tasks. Diffsurv predicts matrices of possible permutations that accommodate the label uncertainty introduced by censored samples. Our experiments reveal that Diffsurv outperforms established baselines in various simulated and real-world risk prediction scenarios. Furthermore, we demonstrate the algorithmic advantages of Diffsurv by presenting a novel method for top-k risk prediction that surpasses current methods.</details>

### [Spotlight] Differentially Private Approximate Near Neighbor Counting in High Dimensions

**Authors:** Alexandr Andoni, Piotr Indyk, Sepideh Mahabadi, Shyam Narayanan

**<details><summary>Abstract**</summary>

Range counting (e.g., counting the number of data points falling into a given query ball) under differential privacy has been studied extensively. However, the current algorithms for this problem are subject to the following dichotomy. One class of algorithms suffers from an additive error that is a fixed polynomial in the number of points. Another class of algorithms allows for polylogarithmic additive error, but the error grows exponentially in the dimension. To achieve the latter, the problem is relaxed to allow a “fuzzy” definition of the range boundary, e.g., a count of the points in a ball of radius $r$ might also include points in a ball of radius $cr$ for some $c>1$. In this paper we present an efficient algorithm that offers a sweet spot between these two classes. The algorithm has an additive error that is an arbitrary small power of the data set size, depending on how fuzzy the range boundary is, as well as a small ($1+o(1)$) multiplicative error. Crucially, the amount of noise added has no dependence on the dimension. Our algorithm introduces a variant of Locality-Sensitive Hashing, utilizing it in a novel manner.</details>

### [Spotlight] Differentially Private Image Classification by Learning Priors from Random Processes

**Authors:** Xinyu Tang, Ashwinee Panda, Vikash Sehwag, Prateek Mittal

**<details><summary>Abstract**</summary>

In privacy-preserving machine learning, differentially private stochastic gradient descent (DP-SGD) performs worse than SGD due to per-sample gradient clipping and noise addition.A recent focus in private learning research is improving the performance of DP-SGD on private data by incorporating priors that are learned on real-world public data.In this work, we explore how we can improve the privacy-utility tradeoff of DP-SGD by learning priors from images generated by random processes and transferring these priors to private data. We propose DP-RandP, a three-phase approach. We attain new state-of-the-art accuracy when training from scratch on CIFAR10, CIFAR100, MedMNIST and ImageNet for a range of privacy budgets $\\varepsilon \\in [1, 8]$. In particular, we improve the previous best reported accuracy on CIFAR10 from $60.6 \\%$ to $72.3 \\%$ for $\\varepsilon=1$.</details>

### Diffusion Probabilistic Models for Structured Node Classification

**Authors:** Hyosoon Jang, Seonghyun Park, Sangwoo Mo, Sungsoo Ahn

**<details><summary>Abstract**</summary>

This paper studies structured node classification on graphs, where the predictions should consider dependencies between the node labels. In particular, we focus on solving the problem for partially labeled graphs where it is essential to incorporate the information in the known label for predicting the unknown labels. To address this issue, we propose a novel framework leveraging the diffusion probabilistic model for structured node classification (DPM-SNC). At the heart of our framework is the extraordinary capability of DPM-SNC to (a) learn a joint distribution over the labels with an expressive reverse diffusion process and (b) make predictions conditioned on the known labels utilizing manifold-constrained sampling. Since the DPMs lack training algorithms for partially labeled data, we design a novel training algorithm to apply DPMs, maximizing a new variational lower bound. We also theoretically analyze how DPMs benefit node classification by enhancing the expressive power of GNNs based on proposing AGG-WL, which is strictly more powerful than the classic 1-WL test. We extensively verify the superiority of our DPM-SNC in diverse scenarios, which include not only the transductive setting on partially labeled graphs but also the inductive setting and unlabeled graphs.</details>

### Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability

**Authors:** Haotian Xue, Alexandre Araujo, Bin Hu, Yongxin Chen

**<details><summary>Abstract**</summary>

Neural networks are known to be susceptible to adversarial samples: small variations of natural examples crafted to deliberatelymislead the models. While they can be easily generated using gradient-based techniques in digital and physical scenarios, they often differ greatly from the actual data distribution of natural images, resulting in a trade-off between strength and stealthiness. In this paper, we propose a novel framework dubbed Diffusion-Based Projected Gradient Descent (Diff-PGD) for generating realistic adversarial samples. By exploiting a gradient guided by a diffusion model, Diff-PGD ensures that adversarial samples remain close to the original data distribution while maintaining their effectiveness. Moreover, our framework can be easily customized for specific tasks such as digital attacks, physical-world attacks, and style-based attacks. Compared with existing methods for generating natural-style adversarial samples, our framework enables the separation of optimizing adversarial loss from other surrogate losses (e.g. content/smoothness/style loss), making it more stable and controllable. Finally, we demonstrate that the samples generated using Diff-PGD have better transferability and anti-purification power than traditional gradient-based methods.</details>

### Diffusion-SS3D: Diffusion Model for Semi-supervised 3D Object Detection

**Authors:** Cheng-Ju Ho, Chen-Hsuan Tai, Yen-Yu Lin, Ming-Hsuan Yang, Yi-Hsuan Tsai

**<details><summary>Abstract**</summary>

Semi-supervised object detection is crucial for 3D scene understanding, efficiently addressing the limitation of acquiring large-scale 3D bounding box annotations. Existing methods typically employ a teacher-student framework with pseudo-labeling to leverage unlabeled point clouds. However, producing reliable pseudo-labels in a diverse 3D space still remains challenging. In this work, we propose Diffusion-SS3D, a new perspective of enhancing the quality of pseudo-labels via the diffusion model for semi-supervised 3D object detection. Specifically, we include noises to produce corrupted 3D object size and class label distributions, and then utilize the diffusion model as a denoising process to obtain bounding box outputs. Moreover, we integrate the diffusion model into the teacher-student framework, so that the denoised bounding boxes can be used to improve pseudo-label generation, as well as the entire semi-supervised learning process. We conduct experiments on the ScanNet and SUN RGB-D benchmark datasets to demonstrate that our approach achieves state-of-the-art performance against existing methods. We also present extensive analysis to understand how our diffusion model design affects performance in semi-supervised learning. The source code will be available at https://github.com/luluho1208/Diffusion-SS3D.</details>

### Diplomat: A Dialogue Dataset for Situated PragMATic Reasoning

**Authors:** Hengli Li, Song-Chun Zhu, Zilong Zheng

**<details><summary>Abstract**</summary>

The ability to discern and comprehend pragmatic meanings is a cornerstone of social and emotional intelligence, referred to as pragmatic reasoning. Despite the strides made in the development of Large Language Models (LLMs), such as ChatGPT, these models grapple with capturing the nuanced and ambiguous facets of language, falling short of the aspiration to build human-like conversational agents. In this work, we introduce a novel benchmark, the **DiPlomat**, which delves into the fundamental components of conversational pragmatic reasoning, encompassing situational context reasoning, open-world knowledge acquisition, and unified figurative language understanding. We start by collecting a new human-annotated dialogue dataset, composed of 4,177 multi-turn dialogues and a vocabulary of 48,900 words. Along with the dataset, two tasks are proposed to evaluate machines' pragmatic reasoning capabilities, namely, Pragmatic Reasoning and Identification(PIR) and Conversational Question Answering (CQA). Furthermore, we probe into a zero-shot natural language inference task, where the significance of context in pragmatic reasoning is underscored. Experimental findings illustrate the existing limitations of current prevailing LLMs in the realm of pragmatic reasoning, shedding light on the pressing need for further research to facilitate the emergence of emotional intelligence within human-like conversational agents.</details>

### Directed Cyclic Graph for Causal Discovery from Multivariate Functional Data

**Authors:** Saptarshi Roy, Raymond K. W. Wong, Yang Ni

**<details><summary>Abstract**</summary>

Discovering causal relationship using multivariate functional data has received a significant amount of attention very recently. In this article, we introduce a functional linear structural equation model for causal structure learning when the underlying graph involving the multivariate functions may have cycles. To enhance interpretability, our model involves a low-dimensional causal embedded space such that all the relevant causal information in the multivariate functional data is preserved in this lower-dimensional subspace. We prove that the proposed model is causally identifiable under standard assumptions that are often made in the causal discovery literature. To carry out inference of our model, we develop a fully Bayesian framework with suitable prior specifications and uncertainty quantification through posterior summaries. We illustrate the superior performance of our method over existing methods in terms of causal graph estimation through extensive simulation studies. We also demonstrate the proposed method using a brain EEG dataset.</details>

### Dis-inhibitory neuronal circuits can control the sign of synaptic plasticity

**Authors:** Julian Rossbroich, Friedemann Zenke

**<details><summary>Abstract**</summary>

How neuronal circuits achieve credit assignment remains a central unsolved question in systems neuroscience. Various studies have suggested plausible solutions for back-propagating error signals through multi-layer networks. These purely functionally motivated models assume distinct neuronal compartments to represent local error signals that determine the sign of synaptic plasticity. However, this explicit error modulation is inconsistent with phenomenological plasticity models in which the sign depends primarily on postsynaptic activity. Here we show how a plausible microcircuit model and Hebbian learning rule derived within an adaptive control theory framework can resolve this discrepancy. Assuming errors are encoded in top-down dis-inhibitory synaptic afferents, we show that error-modulated learning emerges naturally at the circuit level when recurrent inhibition explicitly influences Hebbian plasticity. The same learning rule accounts for experimentally observed plasticity in the absence of inhibition and performs comparably to back-propagation of error (BP) on several non-linearly separable benchmarks. Our findings bridge the gap between functional and experimentally observed plasticity rules and make concrete predictions on inhibitory modulation of excitatory plasticity.</details>

### Discriminative Calibration: Check Bayesian Computation from Simulations and Flexible Classifier

**Authors:** Yuling Yao, Justin Domke

**<details><summary>Abstract**</summary>

To check the accuracy of Bayesian computations, it is common to use rank-based simulation-based calibration (SBC). However, SBC has drawbacks: The test statistic is somewhat ad-hoc, interactions are difficult to examine, multiple testing is a challenge, and the resulting p-value is not a divergence metric. We propose to replace the marginal rank test with a flexible classification approach that learns test statistics from data. This measure typically has a higher statistical power than the SBC test and returns an interpretable divergence measure of miscalibration, computed from classification accuracy. This approach can be used with different data generating processes to address simulation-based inference or traditional inference methods like Markov chain Monte Carlo or variational inference. We illustrate an automated implementation using neural networks and statistically-inspired features, and validate the method with numerical and real data experiments.</details>

### Disentangled Counterfactual Learning for Physical Audiovisual Commonsense Reasoning

**Authors:** Changsheng Lv, Shuai Zhang, Yapeng Tian, Mengshi Qi, Huadong Ma

**<details><summary>Abstract**</summary>

In this paper, we propose a Disentangled Counterfactual Learning (DCL) approach for physical audiovisual commonsense reasoning. The task aims to infer objects’ physics commonsense based on both video and audio input, with the main challenge is how to imitate the reasoning ability of humans. Most of the current methods fail to take full advantage of different characteristics in multi-modal data, and lacking causal reasoning ability in models impedes the progress of implicit physical knowledge inferring. To address these issues, our proposed DCL method decouples videos into static (time-invariant) and dynamic (time-varying) factors in the latent space by the disentangled sequential encoder, which adopts a variational autoencoder (VAE) to maximize the mutual information with a contrastive loss function. Furthermore, we introduce a counterfactual learning module to augment the model’s reasoning ability by modeling physical knowledge relationships among different objects under counterfactual intervention. Our proposed method is a plug-and-play module that can be incorporated into any baseline. In experiments, we show that our proposed method improves baseline methods and achieves state-of-the-art performance. Our source code is available at https://github.com/Andy20178/DCL.</details>

### Disentangled Wasserstein Autoencoder for T-Cell Receptor Engineering

**Authors:** Tianxiao Li, Hongyu Guo, Filippo Grazioli, Mark Gerstein, Martin Renqiang Min

**<details><summary>Abstract**</summary>

In protein biophysics, the separation between the functionally important residues (forming the active site or binding surface) and those that create the overall structure (the fold) is a well-established and fundamental concept. Identifying and modifying those functional sites is critical for protein engineering but computationally non-trivial, and requires significant domain knowledge. To automate this process from a data-driven perspective, we propose a disentangled Wasserstein autoencoder with an auxiliary classifier, which isolates the function-related patterns from the rest with theoretical guarantees. This enables one-pass protein sequence editing and improves the understanding of the resulting sequences and editing actions involved. To demonstrate its effectiveness, we apply it to T-cell receptors (TCRs), a well-studied structure-function case. We show that our method can be used to alter the function of TCRs without changing the structural backbone, outperforming several competing methods in generation quality and efficiency, and requiring only 10\% of the running time needed by baseline models. To our knowledge, this is the first approach that utilizes disentangled representations for TCR engineering.</details>

### Disentangling Cognitive Diagnosis with Limited Exercise Labels

**Authors:** Xiangzhi Chen, Le Wu, Fei Liu, Lei Chen, Kun Zhang, Richang Hong, Meng Wang

**<details><summary>Abstract**</summary>

Cognitive diagnosis is an important task in intelligence education, which aims at measuring students’ proficiency in specific knowledge concepts. Given a fully labeled exercise-concept matrix, most existing models focused on mining students' response records for cognitive diagnosis. Despite their success, due to the huge cost of labeling exercises, a more practical scenario is that limited exercises are labeled with concepts. Performing cognitive diagnosis with limited exercise labels is under-explored and remains pretty much open. In this paper, we propose Disentanglement based Cognitive Diagnosis (DCD) to address the challenges of limited exercise labels. Specifically, we utilize students' response records to model student proficiency, exercise difficulty and exercise label distribution.   Then, we introduce two novel modules - group-based disentanglement and limited-labeled alignment modules - to disentangle the factors relevant to concepts and align them with real limited labels.  Particularly, we introduce the tree-like structure of concepts with negligible cost for group-based disentangling, as concepts of different levels exhibit different independence relationships.Extensive experiments on widely used benchmarks demonstrate the superiority of our proposed model.</details>

### [Spotlight] Distribution-Free Statistical Dispersion Control for Societal Applications

**Authors:** Zhun Deng, Thomas Zollo, Jake Snell, Toniann Pitassi, Richard Zemel

**<details><summary>Abstract**</summary>

Explicit finite-sample statistical guarantees on model performance are an important ingredient in responsible machine learning.  Previous work has focused mainly on bounding either the expected loss of a predictor or the probability that an individual prediction will incur a loss value in a specified range.  However, for many high-stakes applications it is crucial to understand and control the \textit{dispersion} of a loss distribution, or the extent to which different members of a population experience unequal effects of algorithmic decisions. We initiate the study of distribution-free control of statistical dispersion measures with societal implications and propose a simple yet flexible framework that allows us to handle a much richer class of statistical functionals beyond previous work. Our methods are verified through experiments in toxic comment detection, medical imaging, and film recommendation.</details>

### [Spotlight] Distributionally Robust Linear Quadratic Control

**Authors:** Bahar Taskesen, Dan Iancu, Çağıl Koçyiğit, Daniel Kuhn

**<details><summary>Abstract**</summary>

Linear-Quadratic-Gaussian (LQG) control is a fundamental control paradigm that is studied in various fields such as engineering, computer science, economics, and neuroscience. It involves controlling a system with linear dynamics and imperfect observations, subject to additive noise, with the goal of minimizing a quadratic cost function for the state and control variables. In this work, we consider a generalization of the discrete-time, finite-horizon LQG problem, where the noise distributions are unknown and belong to Wasserstein ambiguity sets centered at nominal (Gaussian) distributions. The objective is to minimize a worst-case cost across all distributions in the ambiguity set, including non-Gaussian distributions. Despite the added complexity, we prove that a control policy that is linear in the observations is optimal for this problem, as in the classic LQG problem. We propose a numerical solution method that efficiently characterizes this optimal control policy. Our method uses the Frank-Wolfe algorithm to identify the least-favorable distributions within the Wasserstein ambiguity sets and computes the controller's optimal policy using Kalman filter estimation under these distributions.</details>

### Diverse Conventions for Human-AI Collaboration

**Authors:** Bidipta Sarkar, Andy Shih, Dorsa Sadigh

**<details><summary>Abstract**</summary>

Conventions are crucial for strong performance in cooperative multi-agent games, because they allow players to coordinate on a shared strategy without explicit communication. Unfortunately, standard multi-agent reinforcement learning techniques, such as self-play, converge to conventions that are arbitrary and non-diverse, leading to poor generalization when interacting with new partners. In this work, we present a technique for generating diverse conventions by (1) maximizing their rewards during self-play, while (2) minimizing their rewards when playing with previously discovered conventions (cross-play), stimulating conventions to be semantically different. To ensure that learned policies act in good faith despite the adversarial optimization of cross-play, we introduce mixed-play, where an initial state is randomly generated by sampling self-play and cross-play transitions and the player learns to maximize the self-play reward from this initial state. We analyze the benefits of our technique on various multi-agent collaborative games, including Overcooked, and find that our technique can adapt to the conventions of humans, surpassing human-level performance when paired with real users.</details>

### DoWG Unleashed: An Efficient Universal Parameter-Free Gradient Descent Method

**Authors:** Ahmed Khaled, Konstantin Mishchenko, Chi Jin

**<details><summary>Abstract**</summary>

This paper proposes a new easy-to-implement parameter-free gradient-based optimizer: DoWG (Distance over Weighted Gradients). We prove that DoWG is efficient---matching the convergence rate of optimally tuned gradient descent in convex optimization up to a logarithmic factor without tuning any parameters, and universal---automatically adapting to both smooth and nonsmooth problems. While popular algorithms following the AdaGrad framework compute a running average of the squared gradients, DoWG maintains a new distance-based weighted version of the running average, which is crucial to achieve the desired properties. To complement our theory, we also show empirically that DoWG trains at the edge of stability, and validate its effectiveness on practical machine learning tasks.</details>

### [Spotlight] Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models

**Authors:** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun

**<details><summary>Abstract**</summary>

Language models learn a great quantity of factual information during pretraining, and recent work localizes this information to specific model weights like mid-layer MLP weights. In this paper, we find that we can change how a fact is stored in a model by editing weights that are in a different location than where existing methods suggest that the fact is stored. This is surprising because we would expect that localizing facts to specific model parameters would tell us where to manipulate knowledge in models, and this assumption has motivated past work on model editing methods. Specifically, we show that localization conclusions from representation denoising (also known as Causal Tracing) do not provide any insight into which model MLP layer would be best to edit in order to override an existing stored fact with a new one. This finding raises questions about how past work relies on Causal Tracing to select which model layers to edit. Next, we consider several variants of the editing problem, including erasing and amplifying facts. For one of our editing problems, editing performance does relate to localization results from representation denoising, but we find that which layer we edit is a far better predictor of performance. Our results suggest, counterintuitively, that better mechanistic understanding of how pretrained language models work may not always translate to insights about how to best change their behavior.</details>

### Domain Agnostic Fourier Neural Operators

**Authors:** Ning Liu, Siavash Jafarzadeh, Yue Yu

**<details><summary>Abstract**</summary>

Fourier neural operators (FNOs) can learn highly nonlinear mappings between function spaces, and have recently become a popular tool for learning responses of complex physical systems. However, to achieve good accuracy and efficiency, FNOs rely on the Fast Fourier transform (FFT), which is restricted to modeling problems on rectangular domains. To lift such a restriction and permit FFT on irregular geometries as well as topology changes, we introduce domain agnostic Fourier neural operator (DAFNO), a novel neural operator architecture for learning surrogates with irregular geometries and evolving domains. The key idea is to incorporate a smoothed characteristic function in the integral layer architecture of FNOs, and leverage FFT to achieve rapid computations, in such a way that the geometric information is explicitly encoded in the architecture. In our empirical evaluation, DAFNO has achieved state-of-the-art accuracy as compared to baseline neural operator models on two benchmark datasets of material modeling and airfoil simulation. To further demonstrate the capability and generalizability of DAFNO in handling complex domains with topology changes, we consider a brittle material fracture evolution problem. With only one training crack simulation sample, DAFNO has achieved generalizability to unseen loading scenarios and substantially different crack patterns from the trained scenario. Our code and data accompanying this paper are available at https://github.com/ningliu-iga/DAFNO.</details>

### Don’t Stop Pretraining? Make Prompt-based Fine-tuning Powerful Learner

**Authors:** Zhengxiang Shi, Aldo Lipani

**<details><summary>Abstract**</summary>

Language models (LMs) trained on vast quantities of unlabelled data have greatly advanced the field of natural language processing (NLP). In this study, we re-visit the widely accepted notion in NLP that continued pre-training LMs on task-related texts improves the performance of fine-tuning (FT) in downstream tasks. Through experiments on eight single-sentence tasks and eight sentence-pair tasks in both semi-supervised and fully-supervised settings, we find that conventional continued pre-training does not consistently provide benefits and can even be detrimental for sentence-pair tasks or when prompt-based FT is used. To tackle these issues, we propose Prompt-based Continued Pre-training (PCP), which combines the idea of instruction tuning with conventional continued pre-training. Our approach aims to improve the performance of prompt-based FT by presenting both task-related texts and prompt templates to LMs through unsupervised pre-training objectives before fine-tuning for the target task. Our empirical evaluations on 21 benchmarks demonstrate that the PCP consistently improves the performance of state-of-the-art prompt-based FT approaches (up to 20.1% absolute) in both semi-supervised and fully-supervised settings, even with only hundreds of unlabelled examples. Additionally, prompt-based FT with PCP outperforms state-of-the-art semi-supervised approaches with greater simplicity, eliminating the need for an iterative process and extra data augmentation. Our further analysis explores the performance lower bound of the PCP and reveals that the advantages of PCP persist across different sizes of models and datasets.</details>

### Don’t blame Dataset Shift! Shortcut Learning due to Gradients and Cross Entropy

**Authors:** Aahlad Manas Puli, Lily Zhang, Yoav Wald, Rajesh Ranganath

**<details><summary>Abstract**</summary>

Common explanations for shortcut learning assume that the shortcut improves prediction only under the training distribution. Thus, models trained in the typical way by minimizing log-loss using gradient descent, which we call default-ERM, should utilize the shortcut. However, even when the stable feature determines the label in the training distribution and the shortcut does not provide any additional information, like in perception tasks, default-ERM exhibits shortcut learning. Why are such solutions preferred when the loss can be driven to zero when using the stable feature alone? By studying a linear perception task, we show that default-ERM’s preference for maximizing the margin, even without overparameterization, leads to models that depend more on the shortcut than the stable feature. This insight suggests that default-ERM’s implicit inductive bias towards max-margin may be unsuitable for perception tasks. Instead, we consider inductive biases toward uniform margins. We show that uniform margins guarantee sole dependence on the perfect stable feature in the linear perception task and suggest alternative loss functions, termed margin control (MARG-CTRL), that encourage uniform-margin solutions. MARG-CTRL techniques mitigate shortcut learning on a variety of vision and language tasks, showing that changing inductive biases can remove the need for complicated shortcut-mitigating methods in perception tasks.</details>

### DreamSparse: Escaping from Plato’s Cave with 2D Diffusion Model Given Sparse Views

**Authors:** Paul Yoo, Jiaxian Guo, Yutaka Matsuo, Shixiang (Shane) Gu

**<details><summary>Abstract**</summary>

Synthesizing novel view images from a few views is a challenging but practical problem. Existing methods often struggle with producing high-quality results or necessitate per-object optimization in such few-view settings due to the insufficient information provided. In this work, we explore leveraging the strong 2D priors in pre-trained diffusion models for synthesizing novel view images. 2D diffusion models, nevertheless, lack 3D awareness, leading to distorted image synthesis and compromising the identity. To address these problems, we propose $	extit{DreamSparse}$, a framework that enables the frozen pre-trained diffusion model to generate geometry and identity-consistent novel view images. Specifically, DreamSparse incorporates a geometry module designed to capture features about spatial information from sparse views as a 3D prior. Subsequently, a spatial guidance model is introduced to convert rendered feature maps as spatial information for the generative process. This information is then used to guide the pre-trained diffusion model toencourage the synthesis of geometrically consistent images without further tuning. Leveraging the strong image priors in the pre-trained diffusion models, DreamSparse is capable of synthesizing high-quality novel views for both object and object-centric scene-level images and generalising to open-set images.Experimental results demonstrate that our framework can effectively synthesize novel view images from sparse views and outperforms baselines in both trained and open-set category images. More results can be found on our project page: https://sites.google.com/view/dreamsparse-webpage.</details>

### DreamWaltz: Make a Scene with Complex 3D Animatable Avatars

**Authors:** Yukun Huang, Jianan Wang, Ailing Zeng, He CAO, Xianbiao Qi, Yukai Shi, Zheng-Jun Zha, Lei Zhang

**<details><summary>Abstract**</summary>

We present DreamWaltz, a novel framework for generating and animating complex 3D avatars given text guidance and parametric human body prior. While recent methods have shown encouraging results for text-to-3D generation of common objects, creating high-quality and animatable 3D avatars remains challenging. To create high-quality 3D avatars, DreamWaltz proposes 3D-consistent occlusion-aware Score Distillation Sampling (SDS) to optimize implicit neural representations with canonical poses. It provides view-aligned supervision via 3D-aware skeleton conditioning which enables complex avatar generation without artifacts and multiple faces. For animation, our method learns an animatable 3D avatar representation from abundant image priors of diffusion model conditioned on various poses, which could animate complex non-rigged avatars given arbitrary poses without retraining. Extensive evaluations demonstrate that DreamWaltz is an effective and robust approach for creating 3D avatars that can take on complex shapes and appearances as well as novel poses for animation. The proposed framework further enables the creation of complex scenes with diverse compositions, including avatar-avatar, avatar-object and avatar-scene interactions. See https://dreamwaltz3d.github.io/ for more vivid 3D avatar and animation results.</details>

### Dual Mean-Teacher: An Unbiased Semi-Supervised Framework for Audio-Visual Source Localization

**Authors:** Yuxin Guo, Shijie Ma, Hu Su, Zhiqing Wang, Yuhao Zhao, Wei Zou, Siyang Sun, Yun Zheng

**<details><summary>Abstract**</summary>

Audio-Visual Source Localization (AVSL) aims to locate sounding objects within video frames given the paired audio clips. Existing methods predominantly rely on self-supervised contrastive learning of audio-visual correspondence. Without any bounding-box annotations, they struggle to achieve precise localization, especially for small objects, and suffer from blurry boundaries and false positives. Moreover, the naive semi-supervised method is poor in effectively utilizing the abundance of unlabeled audio-visual pairs. In this paper, we propose a novel Semi-Supervised Learning framework for AVSL, namely Dual Mean-Teacher (DMT), comprising two teacher-student structures to circumvent the confirmation bias issue. Specifically, two teachers, pre-trained on limited labeled data, are employed to filter out noisy samples via the consensus between their predictions, and then generate high-quality pseudo-labels by intersecting their confidence maps. The optimal utilization of both labeled and unlabeled data combined with this unbiased framework enable DMT to outperform current state-of-the-art methods by a large margin, with CIoU of $	extbf{90.4\%}$ and $	extbf{48.8\%}$ on Flickr-SoundNet and VGG-Sound Source, obtaining $\textbf{8.9\%}$ and $\textbf{9.6\%}$ improvements respectively, given only $3\%$ of data positional-annotated. We also extend our framework to some existing AVSL methods and consistently boost their performance. Our code is publicly available at https://github.com/gyx-gloria/DMT.</details>

### DynaDojo: An Extensible Platform for Benchmarking Scaling in Dynamical System Identification

**Authors:** Logan M Bhamidipaty, Tommy Bruzzese, Caryn Tran, Rami Ratl Mrad, Maxinder S. Kanwal

**<details><summary>Abstract**</summary>

Modeling complex dynamical systems poses significant challenges, with traditional methods struggling to work on a variety of systems and scale to high-dimensional dynamics. In response, we present DynaDojo, a novel benchmarking platform designed for data-driven dynamical system identification. DynaDojo provides diagnostics on three ways an algorithm’s performance scales: across the number of training samples, the complexity of a dynamical system, and a target error to achieve. Furthermore, DynaDojo enables studying out-of-distribution generalization (by providing unique test conditions for each system) and active learning (by supporting closed-loop control). Through its user-friendly and easily extensible API, DynaDojo accommodates a wide range of user-defined \texttt{Algorithms}, \texttt{Systems}, and \texttt{Challenges} (evaluation metrics). The platform also prioritizes resource-efficient training with parallel processing strategies for running on a cluster. To showcase its utility, in DynaDojo 0.9, we include implementations of 7 baseline algorithms and 20 dynamical systems, along with several demos exhibiting insights researchers can glean using our platform. This work aspires to make DynaDojo a unifying benchmarking platform for system identification, paralleling the role of OpenAI’s Gym in reinforcement learning.</details>

### Dynamic Pricing and Learning with Bayesian Persuasion

**Authors:** Shipra Agrawal, Yiding Feng, Wei Tang

**<details><summary>Abstract**</summary>

We consider a novel dynamic pricing and learning setting where in addition to setting prices of products in sequential rounds, the seller also ex-ante commits to ‘advertising schemes’. That is, in the beginning of each round the seller can decide what kind of signal they will provide to the buyer about the product’s quality upon realization. Using the popular Bayesian persuasion framework to model the effect of these signals on the buyers’ valuation and purchase responses, we formulate the problem of finding an optimal design of the advertising scheme along with a pricing scheme that maximizes the seller’s expected revenue. Without any apriori knowledge of the buyers’ demand function, our goal is to design an online algorithm that can use past purchase responses to adaptively learn the optimal pricing and advertising strategy. We study the regret of the algorithm when compared to the optimal clairvoyant price and advertisingscheme. Our main result is a computationally efficient online algorithm that achieves an $O(T^{2/3}(m \log T )^{1/3})$ regret bound when the valuation function is linear in the product quality. Here $m$ is the cardinality of the discrete product quality domain and $T$ is the time horizon. This result requires some natural monotonicity and Lipschitz assumptions on the valuation function, but no Lipschitz or smoothness assumption on the buyers’ demand function. For constant $m$, our result matches the regret lower bound for dynamic pricing within logarithmic factors, which is a special case of our problem. We also obtain several improved results for the widely considered special case of additive valuations, including an $\tilde{O}(T^{2/3})$ regret bound independent of $m$ when $m\le T^{1/3}$.</details>

### Dynamics Generalisation in Reinforcement Learning via Adaptive Context-Aware Policies

**Authors:** Michael Beukman, Devon Jarvis, Richard Klein, Steven James, Benjamin Rosman

**<details><summary>Abstract**</summary>

While reinforcement learning has achieved remarkable successes in several domains, its real-world application is limited due to many methods failing to generalise to unfamiliar conditions. In this work, we consider the problem of generalising to new transition dynamics, corresponding to cases in which the environment's response to the agent's actions differs. For example, the gravitational force exerted on a robot depends on its mass and changes the robot's mobility. Consequently, in such cases, it is necessary to condition an agent's actions on extrinsic state information and pertinent contextual information reflecting how the environment responds. While the need for context-sensitive policies has been established, the manner in which context is incorporated architecturally has received less attention. Thus, in this work, we present an investigation into how context information should be incorporated into behaviour learning to improve generalisation.  To this end, we introduce a neural network architecture, the Decision Adapter, which generates the weights of an adapter module and conditions the behaviour of an agent on the context information. We show that the Decision Adapter is a useful generalisation of a previously proposed architecture and empirically demonstrate that it results in superior generalisation performance compared to previous approaches in several environments. Beyond this, the Decision Adapter is more robust to irrelevant distractor variables than several alternative methods.</details>

### EHRXQA: A Multi-Modal Question Answering Dataset for Electronic Health Records with Chest X-ray Images

**Authors:** Seongsu Bae, Daeun Kyung, Jaehee Ryu, Eunbyeol Cho, Gyubok Lee, Sunjun Kweon, Jungwoo Oh, Lei Ji, Eric Chang, Tackeun Kim, Edward Choi

**<details><summary>Abstract**</summary>

Electronic Health Records (EHRs), which contain patients' medical histories in various multi-modal formats, often overlook the potential for joint reasoning across imaging and table modalities underexplored in current EHR Question Answering (QA) systems. In this paper, we introduce EHRXQA, a novel multi-modal question answering dataset combining structured EHRs and chest X-ray images. To develop our dataset, we first construct two uni-modal resources: 1) The MIMIC- CXR-VQA dataset, our newly created medical visual question answering (VQA) benchmark, specifically designed to augment the imaging modality in EHR QA, and 2) EHRSQL (MIMIC-IV), a refashioned version of a previously established table-based EHR QA dataset. By integrating these two uni-modal resources, we successfully construct a multi-modal EHR QA dataset that necessitates both uni-modal and cross-modal reasoning. To address the unique challenges of multi-modal questions within EHRs, we propose a NeuralSQL-based strategy equipped with an external VQA API. This pioneering endeavor enhances engagement with multi-modal EHR sources and we believe that our dataset can catalyze advances in real-world medical scenarios such as clinical decision-making and research. EHRXQA is available at https://github.com/baeseongsu/ehrxqa.</details>

### EMBERSim: A Large-Scale Databank for Boosting Similarity Search in Malware Analysis

**Authors:** Dragos Georgian Corlatescu, Alexandru Dinu, Mihaela Petruta Gaman, Paul Sumedrea

**<details><summary>Abstract**</summary>

In recent years there has been a shift from heuristics based malware detection towards machine learning, which proves to be more robust in the current heavily adversarial threat landscape. While we acknowledge machine learning to be better equipped to mine for patterns in the increasingly high amounts of similar-looking files, we also note a remarkable scarcity of the data available for similarity targeted research. Moreover, we observe that the focus in the few related works falls on quantifying similarity in malware, often overlooking the clean data. This one-sided quantification is especially dangerous in the context of detection bypass. We propose to address the deficiencies in the space of similarity research on binary files, starting from EMBER — one of the largest malware classification datasets. We enhance EMBER with similarity information as well as malware class tags, to enable further research in the similarity space. Our contribution is threefold: (1) we publish EMBERSim, an augmented version of EMBER, that includes similarity informed tags; (2) we enrich EMBERSim with automatically determined malware class tags using the open-source tool AVClass on VirusTotal data and (3) we describe and share the implementation for our class scoring technique and leaf similarity method.</details>

### Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks

**Authors:** Steven Adriaensen, Herilalaina Rakotoarison, Samuel Müller, Frank Hutter

**<details><summary>Abstract**</summary>

Learning curve extrapolation aims to predict model performance in later epochs of training, based on the performance in earlier epochs.In this work, we argue that, while the inherent uncertainty in the extrapolation of learning curves warrants a Bayesian approach, existing methods are (i) overly restrictive, and/or (ii) computationally expensive. We describe the first application of prior-data fitted neural networks (PFNs) in this context. A PFN is a transformer, pre-trained on data generated from a prior, to perform approximate Bayesian inference in a single forward pass. We propose LC-PFN, a PFN trained to extrapolate 10 million artificial right-censored learning curves generated from a parametric prior proposed in prior art using MCMC. We demonstrate that LC-PFN can approximate the posterior predictive distribution more accurately than MCMC, while being over 10 000 times faster. We also show that the same LC-PFN achieves competitive performance extrapolating a total of 20 000 real learning curves from four learning curve benchmarks (LCBench, NAS-Bench-201, Taskset, and PD1) that stem from training a wide range of model architectures (MLPs, CNNs, RNNs, and Transformers) on 53 different datasets with varying input modalities (tabular, image, text, and protein data). Finally, we investigate its potential in the context of model selection and find that a simple LC-PFN based predictive early stopping criterion obtains 2 - 6x speed-ups on 45 of these datasets, at virtually no overhead.</details>

### Efficient Learning of Linear Graph Neural Networks via Node Subsampling

**Authors:** Seiyun Shin, Ilan Shomorony, Han Zhao

**<details><summary>Abstract**</summary>

Graph Neural Networks (GNNs) are a powerful class of machine learning models with applications in recommender systems, drug discovery, social network analysis, and computer vision. One challenge with their implementation is that GNNs often take large-scale graphs as inputs, which imposes significant computational/storage costs in the training and testing phases. In particular, the message passing operations of a GNN require multiplication of the graph adjacency matrix $A \in \mathbb{R}^{n \times n}$ and the data matrix $X \in \mathbb{R}^{n \times d}$, and the $O(n^2 d)$ time complexity can be prohibitive for large $n$. Thus, a natural question is whether it is possible to perform the GNN operations in (quasi-)linear time by avoiding the full computation of $A X$. To study this question, we consider the setting of a regression task on a two-layer Linear Graph Convolutional Network (GCN). We develop an efficient training algorithm based on (1) performing node subsampling, (2) estimating the leverage scores of $A X$ based on the subsampled graph, and (3) performing leverage score sampling on $A X$. We show that our proposed scheme learns the regression model observing only $O(nd\epsilon^{-2}\log n)$ entries of $A$ in time $O(nd^2 \epsilon^{-2}\log n)$, with the guarantee that the learned weights deviate by at most $\epsilon$ under the $\ell_2$ norm from the model learned using the entire adjacency matrix $A$. We present empirical results for regression problems on real-world graphs and show that our algorithm significantly outperforms other baseline sampling strategies that exploit the same number of observations.</details>

### Efficient Neural Music Generation

**Authors:** Max W. Y. Lam, Qiao Tian, Tang Li, Zongyu Yin, Siyuan Feng, Ming Tu, Yuliang Ji, Rui Xia, Mingbo Ma, Xuchen Song, Jitong Chen, Wang Yuping, Yuxuan Wang

**<details><summary>Abstract**</summary>

Recent progress in music generation has been remarkably advanced by the state-of-the-art MusicLM, which comprises a hierarchy of three LMs, respectively, for semantic, coarse acoustic, and fine acoustic modelings. Yet, sampling with the MusicLM requires processing through these LMs one by one to obtain the fine-grained acoustic tokens, making it computationally expensive and prohibitive for a real-time generation. Efficient music generation with a quality on par with MusicLM remains a significant challenge.In this paper, we present **M**e**L**o**D**y (**M** for music; **L** for LM; **D** for diffusion), an LM-guided diffusion model that generates music audios of state-of-the-art quality meanwhile reducing 95.7\% to 99.6\% forward passes in MusicLM, respectively, for sampling 10s to 30s music. MeLoDy inherits the highest-level LM from MusicLM for semantic modeling, and applies a novel dual-path diffusion (DPD) model and an audio VAE-GAN to efficiently decode the conditioning semantic tokens into waveform. DPD is proposed to simultaneously model the coarse and fine acoustics by incorporating the semantic information into segments of latents effectively via cross-attention at each denoising step. Our experimental results suggest the superiority of MeLoDy, not only in its practical advantages on sampling speed and infinitely continuable generation, but also in its state-of-the-art musicality, audio quality, and text correlation.Our samples are available at https://Efficient-MeLoDy.github.io/.</details>

### Efficient Policy Adaptation with Contrastive Prompt Ensemble for Embodied Agents

**Authors:** wonje choi, Woo Kyung Kim, SeungHyun Kim, Honguk Woo

**<details><summary>Abstract**</summary>

For embodied reinforcement learning (RL) agents interacting with the environment, it is desirable to have rapid policy adaptation to unseen visual observations, but achieving zero-shot adaptation capability is considered as a challenging problem in the RL context. To address the problem, we present a novel contrastive prompt ensemble (ConPE) framework which utilizes a pretrained vision-language model and a set of visual prompts, thus enables efficient policy learning and adaptation upon a wide range of environmental and physical changes encountered by embodied agents. Specifically, we devise a guided-attention-based ensemble approach with multiple visual prompts on the vision-language model to construct robust state representations. Each prompt is contrastively learned in terms of an individual domain factors that significantly affects the agent's egocentric perception and observation. For a given task, the attention-based ensemble and policy are jointly learned so that the resulting state representations not only generalize to various domains but are also optimized for learning the task. Through experiments, we show that ConPE outperforms other state-of-the-art algorithms for several embodied agent tasks including navigation in AI2THOR, manipulation in Metaworld, and autonomous driving in CARLA, while also improving the sample efficiency of policy learning and adaptation.</details>

### Efficient Subgame Refinement for Extensive-form Games

**Authors:** Zhenxing Ge, Zheng Xu, Tianyu Ding, Wenbin Li, Yang Gao

**<details><summary>Abstract**</summary>

Subgame solving is an essential technique in addressing large imperfect information games, with various approaches developed to enhance the performance of refined strategies in the abstraction of the target subgame. However, directly applying existing subgame solving techniques may be difficult, due to the intricate nature and substantial size of many real-world games. To overcome this issue, recent subgame solving methods allow for subgame solving on limited knowledge order subgames, increasing their applicability in large games; yet this may still face obstacles due to extensive information set sizes. To address this challenge, we propose a generative subgame solving (GS2) framework, which utilizes a generation function to identify a subset of the earliest-reached nodes, reducing the size of the subgame. Our method is supported by a theoretical analysis and employs a diversity-based generation function to enhance safety. Experiments conducted on medium-sized games as well as the challenging large game of GuanDan demonstrate a significant improvement over the blueprint.</details>

### Egocentric Planning for Scalable Embodied Task Achievement

**Authors:** Xiatoian Liu, Hector Palacios, Christian Muise

**<details><summary>Abstract**</summary>

Embodied agents face significant challenges when tasked with performing actions in diverse environments, particularly in generalizing across object types and executing suitable actions to accomplish tasks. Furthermore, agents should exhibit robustness, minimizing the execution of illegal actions. In this work, we present Egocentric Planning, an innovative approach that combines symbolic planning and Object-oriented POMDPs to solve tasks in complex environments, harnessing existing models for visual perception and natural language processing. We evaluated our approach in ALFRED, a simulated environment designed for domestic tasks, and demonstrated its high scalability, achieving an impressive 36.07\% unseen success rate in the ALFRED benchmark and winning the ALFRED challenge at CVPR Embodied AI workshop. Our method requires reliable perception and the specification or learning of a symbolic description of the preconditions and effects of the agent's actions, as well as what object types reveal information about others. It can naturally scale to solve new tasks beyond ALFRED, as long as they can be solved using the available skills. This work offers a solid baseline for studying end-to-end and hybrid methods that aim to generalize to new tasks, including recent approaches relying on LLMs, but often struggle to scale to long sequences of actions or produce robust plans for novel tasks.</details>

### Eliciting User Preferences for Personalized Multi-Objective Decision Making through Comparative Feedback

**Authors:** Han Shao, Lee Cohen, Avrim Blum, Yishay Mansour, Aadirupa Saha, Matthew Walter

**<details><summary>Abstract**</summary>

In this work, we propose a multi-objective decision making framework that accommodates different user preferences over objectives, where preferences are learned via policy comparisons. Our model consists of a known Markov decision process with a vector-valued reward function, with each user having an unknown preference vector that expresses the relative importance of each objective. The goal is to efficiently compute a near-optimal policy for a given user. We consider two user feedback models. We first address the case where a user is provided with two policies and returns their preferred policy as feedback. We then move to a different user feedback model, where a user is instead provided with two small weighted sets of representative trajectories and selects the preferred one. In both cases, we suggest an algorithm that finds a nearly optimal policy for the user using a number of comparison queries that scales quasilinearly in the number of objectives.</details>

### Embedding Space Interpolation Beyond Mini-Batch, Beyond Pairs and Beyond Examples

**Authors:** Shashanka Venkataramanan, Ewa Kijak, laurent amsaleg, Yannis Avrithis

**<details><summary>Abstract**</summary>

Mixup refers to interpolation-based data augmentation, originally motivated as a way to go beyond empirical risk minimization (ERM). Its extensions mostly focus on the definition of interpolation and the space (input or feature) where it takes place, while the augmentation process itself is less studied. In most methods, the number of generated examples is limited to the mini-batch size and the number of examples being interpolated is limited to two (pairs), in the input space.We make progress in this direction by introducing MultiMix, which generates an arbitrarily large number of interpolated examples beyond the mini-batch size and interpolates the entire mini-batch in the embedding space. Effectively, we sample on the entire convex hull of the mini-batch rather than along linear segments between pairs of examples.On sequence data, we further extend to Dense MultiMix. We densely interpolate features and target labels at each spatial location and also apply the loss densely. To mitigate the lack of dense labels, we inherit labels from examples and weight interpolation factors by attention as a measure of confidence.Overall, we increase the number of loss terms per mini-batch by orders of magnitude at little additional cost. This is only possible because of interpolating in the embedding space. We empirically show that our solutions yield significant improvement over state-of-the-art mixup methods on four different benchmarks, despite interpolation being only linear. By analyzing the embedding space, we show that the classes are more tightly clustered and uniformly spread over the embedding space, thereby explaining the improved behavior.</details>

### End-To-End Latent Variational Diffusion Models for Inverse Problems in High Energy Physics

**Authors:** Alexander Shmakov, Kevin Greif, Michael Fenton, Aishik Ghosh, Pierre Baldi, Daniel Whiteson

**<details><summary>Abstract**</summary>

High-energy collisions at the Large Hadron Collider (LHC) provide valuable insights into open questions in particle physics. However, detector effects must be corrected before measurements can be compared to certain theoretical predictions or measurements from other detectors. Methods to solve this inverse problem of mapping detector observations to theoretical quantities of the underlying collision are essential parts of many physics analyses at the LHC. We investigate and compare various generative deep learning methods to approximate this inverse mapping. We introduce a novel unified architecture, termed latent variational diffusion models, which combines the latent learning of cutting-edge generative art approaches with an end-to-end variational framework. We demonstrate the effectiveness of this approach for reconstructing global distributions of theoretical kinematic quantities, as well as for ensuring the adherence of the learned posterior distributions to known physics constraints. Our unified approach achieves a distribution-free distance to the truth of over 20 times smaller than non-latent state-of-the-art baseline and 3 times smaller than traditional latent diffusion models.</details>

### Energy-Based Sliced Wasserstein Distance

**Authors:** Khai Nguyen, Nhat Ho

**<details><summary>Abstract**</summary>

The sliced Wasserstein (SW) distance has been widely recognized as a statistically effective and computationally efficient metric between two probability measures. A key component of the SW distance is the slicing distribution. There are two existing approaches for choosing this distribution. The first approach is using a fixed prior distribution. The second approach is optimizing for the best distribution which belongs to a parametric family of distributions and can maximize the expected distance. However, both approaches have their limitations. A fixed prior distribution is non-informative in terms of highlighting projecting directions that can discriminate two general probability measures. Doing optimization for the best distribution is often expensive and unstable. Moreover, designing the parametric family of the candidate distribution could be easily misspecified. To address the issues, we propose to design the slicing distribution as an energy-based distribution that is parameter-free and has the density proportional to an energy function of the projected one-dimensional Wasserstein distance. We then derive a novel sliced Wasserstein variant, energy-based sliced Waserstein (EBSW) distance, and investigate its topological, statistical, and computational properties via importance sampling, sampling importance resampling, and Markov Chain methods. Finally, we conduct experiments on point-cloud gradient flow, color transfer, and point-cloud reconstruction to show the favorable performance of the EBSW.</details>

### Enhancing Adversarial Robustness via Score-Based Optimization

**Authors:** Boya Zhang, Weijian Luo, Zhihua Zhang

**<details><summary>Abstract**</summary>

Adversarial attacks have the potential to mislead deep neural network classifiers by introducing slight perturbations. Developing algorithms that can mitigate the effects of these attacks is crucial for ensuring the safe use of artificial intelligence. Recent studies have suggested that score-based diffusion models are effective in adversarial defenses. However, existing diffusion-based defenses rely on the sequential simulation of the reversed stochastic differential equations of diffusion models, which are computationally inefficient and yield suboptimal results. In this paper, we introduce a novel adversarial defense scheme named ScoreOpt, which optimizes adversarial samples at test-time, towards original clean data  in the direction guided by score-based priors. We conduct comprehensive experiments on multiple datasets, including CIFAR10, CIFAR100 and ImageNet. Our experimental results demonstrate that our approach outperforms existing adversarial defenses in terms of both robustness performance and inference speed.</details>

### Equivariant Neural Simulators for Stochastic Spatiotemporal Dynamics

**Authors:** Koen Minartz, Yoeri Poels, Simon Koop, Vlado Menkovski

**<details><summary>Abstract**</summary>

Neural networks are emerging as a tool for scalable data-driven simulation of high-dimensional dynamical systems, especially in settings where numerical methods are infeasible or computationally expensive. Notably, it has been shown that incorporating domain symmetries in deterministic neural simulators can substantially improve their accuracy, sample efficiency, and parameter efficiency. However, to incorporate symmetries in probabilistic neural simulators that can simulate stochastic phenomena, we need a model that produces equivariant distributions over trajectories, rather than equivariant function approximations. In this paper, we propose Equivariant Probabilistic Neural Simulation (EPNS), a framework for autoregressive probabilistic modeling of equivariant distributions over system evolutions. We use EPNS to design models for a stochastic n-body system and stochastic cellular dynamics. Our results show that EPNS considerably outperforms existing neural network-based methods for probabilistic simulation. More specifically, we demonstrate that incorporating equivariance in EPNS improves simulation quality, data efficiency, rollout stability, and uncertainty quantification. We conclude that EPNS is a promising method for efficient and effective data-driven probabilistic simulation in a diverse range of domains.</details>

### Equivariant Spatio-Temporal Attentive Graph Networks to Simulate Physical Dynamics

**Authors:** Liming Wu, Zhichao Hou, Jirui Yuan, Yu Rong, Wenbing Huang

**<details><summary>Abstract**</summary>

Learning to represent and simulate the dynamics of physical systems is a crucial yet challenging task. Existing equivariant Graph Neural Network (GNN) based methods have encapsulated the symmetry of physics, \emph{e.g.}, translations, rotations, etc, leading to better generalization ability. Nevertheless, their frame-to-frame formulation of the task overlooks the non-Markov property mainly incurred by unobserved dynamics in the environment. In this paper, we reformulate dynamics simulation as a spatio-temporal prediction task, by employing the trajectory in the past period to recover the Non-Markovian interactions. We propose Equivariant Spatio-Temporal Attentive Graph Networks (ESTAG), an equivariant version of spatio-temporal GNNs, to fulfil our purpose. At its core, we design a novel Equivariant Discrete Fourier Transform (EDFT) to extract periodic patterns from the history frames, and then construct an Equivariant Spatial Module (ESM) to accomplish spatial message passing, and an Equivariant Temporal Module (ETM) with the forward attention and equivariant pooling mechanisms to aggregate temporal message. We evaluate our model on three real datasets corresponding to the molecular-, protein- and macro-level. Experimental results verify the effectiveness of ESTAG compared to typical spatio-temporal GNNs and equivariant GNNs.</details>

### Equivariant flow matching

**Authors:** Leon Klein, Andreas Krämer, Frank Noe

**<details><summary>Abstract**</summary>

Normalizing flows are a class of deep generative models that are especially interesting for modeling probability distributions in physics, where the exact likelihood of flows allows reweighting to known target energy functions and computing unbiased observables. For instance, Boltzmann generators tackle the long-standing sampling problem in statistical physics by training flows to produce equilibrium samples of many-body systems such as small molecules and proteins. To build effective models for such systems, it is crucial to incorporate the symmetries of the target energy into the model, which can be achieved by equivariant continuous normalizing flows (CNFs). However, CNFs can be computationally expensive to train and generate samples from, which has hampered their scalability and practical application.In this paper, we introduce equivariant flow matching, a new training objective for equivariant CNFs that is based on the recently proposed optimal transport flow matching. Equivariant flow matching exploits the physical symmetries of the target energy for efficient, simulation-free training of equivariant CNFs.We demonstrate the effectiveness of flow matching on rotation and permutation invariant many-particle systems and a small molecule, alanine dipeptide, where for the first time we obtain a Boltzmann generator with significant sampling efficiency without relying on tailored internal coordinate featurization. Our results show that the equivariant flow matching objective yields flows with shorter integration paths, improved sampling efficiency, and higher scalability compared to existing methods.</details>

### Estimating Causal Effects Identifiable from a Combination of Observations and Experiments

**Authors:** Yonghan Jung, Ivan Diaz, Jin Tian, Elias Bareinboim

**<details><summary>Abstract**</summary>

Learning cause and effect relations is arguably one of the central challenges found throughout the data sciences.Formally, determining whether a collection of observational and interventional distributions can be combined to learn a target causal relation is known as the problem of generalized identification (or g-identification) [Lee et al., 2019]. Although g-identification has been well understood and solved in theory, it turns out to be challenging to apply these results in practice, in particular when considering the estimation of the target distribution from finite samples. In this paper, we develop a new, general estimator that exhibits multiply robustness properties for g-identifiable causal functionals. Specifically, we show that any g-identifiable causal effect can be expressed as a function of generalized multi-outcome sequential back-door adjustments that are amenable to estimation. We then construct a corresponding estimator for the g-identification expression that exhibits robustness properties to bias. We analyze the asymptotic convergence properties of the estimator. Finally, we illustrate the use of the proposed estimator in experimental studies. Simulation results corroborate the theory.</details>

### Estimating Noise Correlations Across Continuous Conditions With Wishart Processes

**Authors:** Amin Nejatbakhsh, Isabel Garon, Alex Williams

**<details><summary>Abstract**</summary>

The signaling capacity of a neural population depends on the scale and orientation of its covariance across trials. Estimating this "noise" covariance is challenging and is thought to require a large number of stereotyped trials. New approaches are therefore needed to interrogate the structure of neural noise across rich, naturalistic behaviors and sensory experiences, with few trials per condition. Here, we exploit the fact that conditions are smoothly parameterized in many experiments and leverage Wishart process models to pool statistical power from trials in neighboring conditions. We demonstrate that these models perform favorably on experimental data from the mouse visual cortex and monkey motor cortex relative to standard covariance estimators. Moreover, they produce smooth estimates of covariance as a function of stimulus parameters, enabling estimates of noise correlations in entirely unseen conditions as well as continuous estimates of Fisher information—a commonly used measure of signal fidelity. Together, our results suggest that Wishart processes are broadly applicable tools for quantification and uncertainty estimation of noise correlations in trial-limited regimes, paving the way toward understanding the role of noise in complex neural computations and behavior.</details>

### Euler-Lagrange Analysis of Generative Adversarial Networks

**Authors:** Siddarth Asokan, Chandra Seelamantula

**<details><summary>Abstract**</summary>

We consider Generative Adversarial Networks (GANs) and address the underlying functional optimization problem ab initio within a variational setting. Strictly speaking, the optimization of the generator and discriminator functions must be carried out in accordance with the Euler-Lagrange conditions, which become particularly relevant in scenarios where the  optimization cost involves regularizers comprising the derivatives of these functions. Considering Wasserstein GANs (WGANs) with a gradient-norm penalty, we show that the optimal discriminator is the solution to a Poisson differential equation. In principle, the optimal discriminator can be obtained in closed form without having to train a neural network. We illustrate this by employing a Fourier-series approximation to solve the Poisson differential equation. Experimental results based on synthesized Gaussian data demonstrate superior convergence behavior of the proposed approach in comparison with the baseline WGAN variants that employ weight-clipping, gradient or Lipschitz penalties on the discriminator on low-dimensional data. We also analyze the truncation error of the Fourier-series approximation  and the estimation error of the Fourier coefficients in a high-dimensional setting. We demonstrate applications to real-world images considering latent-space prior matching in Wasserstein autoencoders and present performance comparisons on  benchmark datasets such as MNIST, SVHN, CelebA, CIFAR-10, and Ukiyo-E. We demonstrate that the proposed approach achieves comparable reconstruction error and Frechet inception distance with faster convergence and up to two-fold improvement in image sharpness.</details>

### Evaluating Self-Supervised Learning for Molecular Graph Embeddings

**Authors:** Hanchen Wang, Jean Kaddour, Shengchao Liu, Jian Tang, Joan Lasenby, Qi Liu

**<details><summary>Abstract**</summary>

Graph Self-Supervised Learning (GSSL) provides a robust pathway for acquiring embeddings without expert labelling, a capability that carries profound implications for molecular graphs due to the staggering number of potential molecules and the high cost of obtaining labels. However, GSSL methods are designed not for optimisation within a specific domain but rather for transferability across a variety of downstream tasks. This broad applicability complicates their evaluation. Addressing this challenge, we present "Molecular Graph Representation Evaluation" (MOLGRAPHEVAL), generating detailed profiles of molecular graph embeddings with interpretable and diversified attributes. MOLGRAPHEVAL offers a suite of probing tasks grouped into three categories: (i) generic graph, (ii) molecular substructure, and (iii) embedding space properties. By leveraging MOLGRAPHEVAL to benchmark existing GSSL methods against both current downstream datasets and our suite of tasks, we uncover significant inconsistencies between inferences drawn solely from existing datasets and those derived from more nuanced probing. These findings suggest that current evaluation methodologies fail to capture the entirety of the landscape.</details>

### Evaluating the Robustness of Interpretability Methods through Explanation Invariance and Equivariance

**Authors:** Jonathan Crabbé, Mihaela van der Schaar

**<details><summary>Abstract**</summary>

Interpretability methods are valuable only if their explanations faithfully describe the explained model. In this work, we consider neural networks whose predictions are invariant under a specific symmetry group. This includes popular architectures, ranging from convolutional to graph neural networks. Any explanation that faithfully explains this type of model needs to be in agreement with this invariance property. We formalize this intuition through the notion of explanation invariance and equivariance by leveraging the formalism from geometric deep learning. Through this rigorous formalism, we derive (1) two metrics to measure the robustness of any interpretability method with respect to the model symmetry group; (2) theoretical robustness guarantees for some popular interpretability methods and (3) a systematic approach to increase the invariance of any interpretability method with respect to a symmetry group. By empirically measuring our metrics for explanations of models associated with various modalities and symmetry groups, we derive a set of 5 guidelines to allow users and developers of interpretability methods to produce robust explanations.</details>

### [Oral] Exact Bayesian Inference on Discrete Models via Probability Generating Functions: A Probabilistic Programming Approach

**Authors:** Fabian Zaiser, Andrzej Murawski, Luke Ong

**Oral Presentation:** Tu, Dec 12, 08:15 -- Oral 1C

**<details><summary>Abstract**</summary>

We present an exact Bayesian inference method for discrete statistical models, which can find exact solutions to a large class of discrete inference problems, even with infinite support and continuous priors.To express such models, we introduce a probabilistic programming language that supports discrete and continuous sampling, discrete observations, affine functions, (stochastic) branching, and conditioning on discrete events.Our key tool is *probability generating functions*:they provide a compact closed-form representation of distributions that are definable by programs, thus enabling the exact computation of posterior probabilities, expectation, variance, and higher moments.Our inference method is provably correct and fully automated in a tool called *Genfer*, which uses automatic differentiation (specifically, Taylor polynomials), but does not require computer algebra.Our experiments show that Genfer is often faster than the existing exact inference tools PSI, Dice, and Prodigy.On a range of real-world inference problems that none of these exact tools can solve, Genfer's performance is competitive with approximate Monte Carlo methods, while avoiding approximation errors.</details>

### Expanding Small-Scale Datasets with Guided Imagination

**Authors:** Yifan Zhang, Daquan Zhou, Bryan Hooi, Kai Wang, Jiashi Feng

**<details><summary>Abstract**</summary>

The power of DNNs relies heavily on the quantity and quality of training data. However, collecting and annotating data on a large scale is often expensive and time-consuming. To address this issue, we explore a new task, termed dataset expansion, aimed at expanding a ready-to-use small dataset by automatically creating new labeled samples. To this end, we present a Guided Imagination Framework (GIF) that leverages cutting-edge generative models like DALL-E2 and Stable Diffusion (SD) to "imagine" and create informative new data from the input seed data. Specifically, GIF conducts data imagination by optimizing the latent features of the seed data in the semantically meaningful space of the prior model, resulting in the creation of photo-realistic images with new content. To guide the imagination towards creating informative samples for model training, we introduce two key criteria, i.e., class-maintained information boosting and sample diversity promotion. These criteria are verified to be essential for effective dataset expansion: GIF-SD obtains 13.5\% higher model accuracy on natural image datasets than unguided expansion with SD. With these essential criteria, GIF successfully expands small datasets in various scenarios, boosting model accuracy by 36.9\% on average over six natural image datasets and by 13.5\% on average over three medical datasets. The source code is available at https://github.com/Vanint/DatasetExpansion.</details>

### [Spotlight] Explaining the Uncertain: Stochastic Shapley Values for Gaussian Process Models

**Authors:** Siu Lun Chau, Krikamol Muandet, Dino Sejdinovic

**<details><summary>Abstract**</summary>

We present a novel approach for explaining Gaussian processes (GPs) that can utilize the full analytical covariance structure present in GPs. Our method is based on the popular solution concept of Shapley values extended to stochastic cooperative games, resulting in explanations that are random variables. The GP explanations generated using our approach satisfy similar favorable axioms to standard Shapley values and possess a tractable covariance function across features and data observations. This covariance allows for quantifying explanation uncertainties and studying the statistical dependencies between explanations. We further extend our framework to the problem of predictive explanation, and propose a Shapley prior over the explanation function to predict Shapley values for new data based on previously computed ones. Our extensive illustrations demonstrate the effectiveness of the proposed approach.</details>

### Exploring Why Object Recognition Performance Degrades Across Income Levels and Geographies with Factor Annotations

**Authors:** Laura Gustafson, Megan Richards, Melissa Hall, Caner Hazirbas, Diane Bouchacourt, Mark Ibrahim

**<details><summary>Abstract**</summary>

Despite impressive advances in object-recognition, deep learning systems’ performance degrades significantly across geographies and lower income levels---raising pressing concerns of inequity. Addressing such performance gaps remains a challenge, as little is understood about why performance degrades across incomes or geographies.We take a step in this direction by annotating images from Dollar Street, a popular benchmark of geographically and economically diverse images, labeling each image with factors such as color, shape, and background. These annotations unlock a new granular view into how objects differ across incomes/regions. We then use these object differences to pinpoint model vulnerabilities across incomes and regions.We study a range of modern vision models, finding that performance disparities are most associated with differences in _texture, occlusion_, and images with _darker lighting_.We illustrate how insights from our factor labels can surface mitigations to improve models' performance disparities.As an example, we show that mitigating a model's vulnerability to texture can improve performance on the lower income level.**We release all the factor annotations along with an interactive dashboardto facilitate research into more equitable vision systems.**</details>

### Exponentially Convergent Algorithms for Supervised Matrix Factorization

**Authors:** Joowon Lee, Hanbaek Lyu, Weixin Yao

**<details><summary>Abstract**</summary>

Supervised matrix factorization (SMF) is a classical machine learning method that simultaneously seeks feature extraction and classification tasks, which are not necessarily a priori aligned objectives. Our goal is to use SMF to learn low-rank latent factors that offer interpretable, data-reconstructive, and class-discriminative features, addressing challenges posed by high-dimensional data. Training SMF model involves solving a nonconvex and possibly constrained optimization with at least three blocks of parameters. Known algorithms are either heuristic or provide weak convergence guarantees for special cases. In this paper, we provide a novel framework that `lifts' SMF as a low-rank matrix estimation problem in a combined factor space and propose an efficient algorithm that provably converges exponentially fast to a global minimizer of the objective with arbitrary initialization under mild assumptions. Our framework applies to a wide range of SMF-type problems for multi-class classification with auxiliary features. To showcase an application, we demonstrate that our algorithm successfully identified well-known cancer-associated gene groups for various cancers.</details>

### [Spotlight] Extraction and Recovery of Spatio-Temporal Structure in Latent Dynamics Alignment with Diffusion Model

**Authors:** Yule Wang, Zijing Wu, Chengrui Li, Anqi Wu

**<details><summary>Abstract**</summary>

In the field of behavior-related brain computation, it is necessary to align raw neural signals against the drastic domain shift among them. A foundational framework within neuroscience research posits that trial-based neural population activities rely on low-dimensional latent dynamics, thus focusing on the latter greatly facilitates the alignment procedure. Despite this field's progress, existing methods ignore the intrinsic spatio-temporal structure during the alignment phase. Hence, their solutions usually lead to poor quality in latent dynamics structures and overall performance. To tackle this problem, we propose an alignment method ERDiff, which leverages the expressivity of the diffusion model to preserve the spatio-temporal structure of latent dynamics. Specifically, the latent dynamics structures of the source domain are first extracted by a diffusion model. Then, under the guidance of this diffusion model, such structures are well-recovered through a maximum likelihood alignment procedure in the target domain. We first demonstrate the effectiveness of our proposed method on a synthetic dataset. Then, when applied to neural recordings from the non-human primate motor cortex, under both cross-day and inter-subject settings, our method consistently manifests its capability of preserving the spatio-temporal structure of latent dynamics and outperforms existing approaches in alignment goodness-of-fit and neural decoding performance.</details>

### FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation

**Authors:** Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, Lu Hou

**<details><summary>Abstract**</summary>

Recently, open-domain text-to-video (T2V) generation models have made remarkable progress. However, the promising results are mainly shown by the qualitative cases of generated videos, while the quantitative evaluation of T2V models still faces two critical problems. Firstly, existing studies lack fine-grained evaluation of T2V models on different categories of text prompts. Although some benchmarks have categorized the prompts, their categorization either only focuses on a single aspect or fails to consider the temporal information in video generation. Secondly, it is unclear whether the automatic evaluation metrics are consistent with human standards. To address these problems, we propose **FETV**, a benchmark for **F**ine-grained **E**valuation of **T**ext-to-**V**ideo generation. FETV is multi-aspect, categorizing the prompts based on three orthogonal aspects: the major content, the attributes to control and the prompt complexity. FETV is also temporal-aware, which introduces several temporal categories tailored for video generation. Based on FETV, we conduct comprehensive manual evaluations of four representative T2V models, revealing their pros and cons on different categories of prompts from different aspects. We also extend FETV as a testbed to evaluate the reliability of automatic T2V metrics. The multi-aspect categorization of FETV enables fine-grained analysis of the metrics' reliability in different scenarios. We find that existing automatic metrics (e.g., CLIPScore and FVD) correlate poorly with human evaluation. To address this problem, we explore several solutions to improve CLIPScore and FVD, and develop two automatic metrics that exhibit significant higher correlation with humans than existing metrics. Benchmark page: https://github.com/llyx97/FETV.</details>

### Face Reconstruction from Facial Templates by Learning Latent Space of a Generator Network

**Authors:** Hatef Otroshi Shahreza, Sébastien Marcel

**<details><summary>Abstract**</summary>

In this paper, we focus on the template inversion attack against face recognition systems and propose a new method to reconstruct face images from facial templates. Within a generative adversarial network (GAN)-based framework, we learn a mapping from facial templates to the intermediate latent space of a pre-trained face generation network, from which we can generate high-resolution realistic reconstructed face images. We show that our proposed method can be applied in whitebox and blackbox attacks against face recognition systems. Furthermore, we evaluate the transferability of our attack when the adversary uses the reconstructed face image to impersonate the underlying subject in an attack against another face recognition system. Considering the adversary's knowledge and the target face recognition system, we define five different attacks and evaluate the vulnerability of state-of-the-art face recognition systems. Our experiments show that our proposed method achieves high success attack rates in whitebox and blackbox scenarios. Furthermore, the reconstructed face images are transferable and can be used to enter target face recognition systems with a different feature extractor model. We also explore important areas in the reconstructed face images that can fool the target face recognition system.</details>

### FaceComposer: A Unified Model for Versatile Facial Content Creation

**Authors:** Jiayu Wang, Kang Zhao, Yifeng Ma, Shiwei Zhang, Yingya Zhang, Yujun Shen, Deli Zhao, Jingren Zhou

**<details><summary>Abstract**</summary>

This work presents FaceComposer, a unified generative model that accomplishes a variety of facial content creation tasks, including text-conditioned face synthesis, text-guided face editing, face animation etc. Based on the latent diffusion framework, FaceComposer follows the paradigm of compositional generation and employs diverse face-specific conditions, e.g., Identity Feature and Projected Normalized Coordinate Code, to release the model creativity at all possible. To support text control and animation, we clean up some existing face image datasets and collect around 500 hours of talking-face videos, forming a high-quality large-scale multi-modal face database. A temporal self-attention module is incorporated into the U-Net structure, which allows learning the denoising process on the mixture of images and videos. Extensive experiments suggest that our approach not only achieves comparable or even better performance than state-of-the-arts on each single task, but also facilitates some combined tasks with one-time forward, demonstrating its potential in serving as a foundation generative model in face domain. We further develop an interface such that users can enjoy our one-step service to create, edit, and animate their own characters. Code, dataset, model, and interface will be made publicly available.</details>

### Facing Off World Model Backbones: RNNs, Transformers, and S4

**Authors:** Fei Deng, Junyeong Park, Sungjin Ahn

**<details><summary>Abstract**</summary>

World models are a fundamental component in model-based reinforcement learning (MBRL). To perform temporally extended and consistent simulations of the future in partially observable environments, world models need to possess long-term memory. However, state-of-the-art MBRL agents, such as Dreamer, predominantly employ recurrent neural networks (RNNs) as their world model backbone, which have limited memory capacity. In this paper, we seek to explore alternative world model backbones for improving long-term memory. In particular, we investigate the effectiveness of Transformers and Structured State Space Sequence (S4) models, motivated by their remarkable ability to capture long-range dependencies in low-dimensional sequences and their complementary strengths. We propose S4WM, the first world model compatible with parallelizable SSMs including S4 and its variants. By incorporating latent variable modeling, S4WM can efficiently generate high-dimensional image sequences through latent imagination. Furthermore, we extensively compare RNN-, Transformer-, and S4-based world models across four sets of environments, which we have tailored to assess crucial memory capabilities of world models, including long-term imagination, context-dependent recall, reward prediction, and memory-based reasoning. Our findings demonstrate that S4WM outperforms Transformer-based world models in terms of long-term memory, while exhibiting greater efficiency during training and imagination. These results pave the way for the development of stronger MBRL agents.</details>

### Failure-Aware Gaussian Process Optimization with Regret Bounds

**Authors:** Shogo Iwazaki, Shion Takeno, Tomohiko Tanabe, Mitsuru Irie

**<details><summary>Abstract**</summary>

Real-world optimization problems often require black-box optimization with observation failure, where we can obtain the objective function value if we succeed, otherwise, we can only obtain a fact of failure. Moreover, this failure region can be complex by several latent constraints, whose number is also unknown. For this problem, we propose a failure-aware Gaussian process upper confidence bound (F-GP-UCB), which only requires a mild assumption for the observation failure that an optimal solution lies on an interior of a feasible region. Furthermore, we show that the number of successful observations grows linearly, by which we provide the first regret upper bounds and the convergence of F-GP-UCB. We demonstrate the effectiveness of F-GP-UCB in several benchmark functions, including the simulation function motivated by material synthesis experiments.</details>

### Fair Allocation of Indivisible Chores: Beyond Additive Costs

**Authors:** Bo Li, Fangxiao Wang, Yu Zhou

**<details><summary>Abstract**</summary>

We study the maximin share (MMS) fair allocation of $m$ indivisible tasks to $n$ agents who have costs for completing the assigned tasks.It is known that exact MMS fairness cannot be guaranteed, and so far the best-known approximation for additive cost functions is $\frac{13}{11}$ by Huang and Segal-Halevi [EC, 2023]; however, beyond additivity, very little is known. In this work, we first prove that no algorithm can ensure better than $\min\{n,\frac{\log m}{\log \log m}\}$-approximation if the cost functions are submodular. This result also shows a sharp contrast with the allocation of goods where constant approximations exist as shown by Barman and Krishnamurthy [TEAC, 2020] and Ghodsi et al. [AIJ, 2022]. We then prove that for subadditive costs, there always exists an allocation that is $\min\{n,\lceil\log m\rceil\}$-approximation, and thus the approximation ratio is asymptotically tight.Besides multiplicative approximation, we also consider the ordinal relaxation, 1-out-of-$d$ MMS, which was recently proposed by Hosseini et al. [JAIR and AAMAS, 2022]. Our impossibility result implies that for any $d\ge 2$, a 1-out-of-$d$ MMS allocation may not exist.Due to these hardness results for general subadditive costs, we turn to studying two specific subadditive costs, namely, bin packing and job scheduling. For both settings, we show that constant approximate allocations exist for both multiplicative and ordinal relaxations of MMS.</details>

### Fair Streaming Principal Component Analysis: Statistical and Algorithmic Viewpoint

**Authors:** Junghyun Lee, Hanseul Cho, Se-Young Yun, Chulhee Yun

**<details><summary>Abstract**</summary>

Fair Principal Component Analysis (PCA) is a problem setting where we aim to perform PCA while making the resulting representation fair in that the projected distributions, conditional on the sensitive attributes, match one another. However, existing approaches to fair PCA have two main problems: theoretically, there has been no statistical foundation of fair PCA in terms of learnability; practically, limited memory prevents us from using existing approaches, as they explicitly rely on full access to the entire data. On the theoretical side, we rigorously formulate fair PCA using a new notion called probably approximately fair and optimal (PAFO) learnability. On the practical side, motivated by recent advances in streaming algorithms for addressing memory limitation, we propose a new setting called fair streaming PCA along with a memory-efficient algorithm, fair noisy power method (FNPM). We then provide its statistical guarantee in terms of PAFO-learnability, which is the first of its kind in fair PCA literature. We verify our algorithm in the CelebA dataset without any pre-processing; while the existing approaches are inapplicable due to memory limitations, by turning it into a streaming setting, we show that our algorithm performs fair PCA efficiently and effectively.</details>

### [Spotlight] Faith and Fate: Limits of Transformers on Compositionality

**Authors:** Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang (Lorraine) Li, Liwei Jiang, Bill Yuchen Lin, Sean Welleck, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena Hwang, Soumya Sanyal, Xiang Ren, Allyson Ettinger, Zaid Harchaoui, Yejin Choi

**<details><summary>Abstract**</summary>

Transformer large language models (LLMs) have sparked admiration for their exceptional performance on tasks that demand intricate multi-step reasoning. Yet, these models simultaneously show failures on surprisingly trivial problems. This begs the question: Are these errors incidental, or do they signal more substantial limitations?In an attempt to demystify transformer LLMs, we investigate the limits of these models across three representative compositional tasks---multi-digit multiplication, logic grid puzzles, and a classic dynamic programming problem. These tasks require breaking problems down into sub-steps and synthesizing these steps into a precise answer.  We formulate compositional tasks as computation graphs to systematically quantify the level of complexity, and break down reasoning steps into intermediate sub-procedures. Our empirical findings suggest that transformer LLMs solve compositional tasks by reducing multi-step compositional reasoning into linearized subgraph matching, without necessarily developing systematic problem-solving skills.  To round off our empirical study, we provide theoretical arguments on abstract multi-step reasoning problems that highlight how autoregressive generations' performance can rapidly decay with increased task complexity.</details>

### Fantastic Weights and How to Find Them: Where to Prune in Dynamic Sparse Training

**Authors:** Aleksandra Nowak, Bram Grooten, Decebal Constantin Mocanu, Jacek Tabor

**<details><summary>Abstract**</summary>

Dynamic Sparse Training (DST) is a rapidly evolving area of research that seeks to optimize the sparse initialization of a neural network by adapting its topology during training.  It has been shown that under specific conditions, DST is able to outperform dense models. The key components of this framework are the pruning and growing criteria, which are repeatedly applied during the training process to adjust the network’s sparse connectivity. While the growing criterion's impact on DST performance is relatively well studied, the influence of the pruning criterion remains overlooked. To address this issue, we design and perform an extensive empirical analysis of various pruning criteria to better understand their impact on the dynamics of DST solutions. Surprisingly, we find that most of the studied methods yield similar results. The differences become more significant in the low-density regime, where the best performance is predominantly given by the simplest technique: magnitude-based pruning.</details>

### Fast Asymptotically Optimal Algorithms for Non-Parametric Stochastic Bandits

**Authors:** Dorian Baudry, Fabien Pesquerel, Rémy Degenne, Odalric-Ambrym Maillard

**<details><summary>Abstract**</summary>

We consider the problem of regret minimization in non-parametric stochastic bandits. When the rewards are known to be bounded from above, there exists asymptotically optimal algorithms, with asymptotic regret depending on an infimum of Kullback-Leibler divergences (KL). These algorithms are computationally expensive and require storing all past rewards, thus simpler but non-optimal algorithms are often used instead. We introduce several methods to approximate the infimum KL which reduce drastically the computational and memory costs of existing optimal algorithms, while keeping their regret guaranties. We apply our findings to design new variants of the MED and IMED algorithms, and demonstrate their interest with extensive numerical simulations.</details>

### Fast Conditional Mixing of MCMC Algorithms for Non-log-concave Distributions

**Authors:** Xiang Cheng, Bohan Wang, Jingzhao Zhang, Yusong Zhu

**<details><summary>Abstract**</summary>

MCMC algorithms offer empirically efficient tools for sampling from a target distribution $\pi(x) \propto \exp(-V(x))$. However, on the theory side, MCMC algorithms suffer from slow mixing rate when $\pi(x)$ is non-log-concave. Our work examines this gap and shows that when Poincar\'e-style inequality holds on a subset $\mathcal{X}$ of the state space, the conditional distribution of MCMC iterates over $\mathcal{X}$ mixes fast to the true conditional distribution. This fast mixing guarantee can hold in cases when global mixing is provably slow. We formalize the statement and quantify the conditional mixing rate. We further show that conditional mixing can have interesting implications for sampling from mixtures of Gaussians, parameter estimation for Gaussian mixture models, and Gibbs-sampling with well-connected local minima.</details>

### [Spotlight] Feature Adaptation for Sparse Linear Regression

**Authors:** Jonathan Kelner, Frederic Koehler, Raghu Meka, Dhruv Rohatgi

**<details><summary>Abstract**</summary>

Sparse linear regression is a central problem in high-dimensional statistics. We study the correlated random design setting, where the covariates are drawn from a multivariate Gaussian $N(0,\Sigma)$, and we seek an estimator with small excess risk. If the true signal is $t$-sparse, information-theoretically, it is possible to achieve strong recovery guarantees with only $O(t\log n)$ samples. However, computationally efficient algorithms have sample complexity linear in (some variant of) the *condition number* of $\Sigma$. Classical algorithms such as the Lasso can require significantly more samples than necessary even if there is only a single sparse approximate dependency among the covariates.We provide a polynomial-time algorithm that, given $\Sigma$, automatically adapts the Lasso to tolerate a small number of approximate dependencies. In particular, we achieve near-optimal sample complexity for constant sparsity and if $\Sigma$ has few ``outlier'' eigenvalues.Our algorithm fits into a broader framework of *feature adaptation* for sparse linear regression with ill-conditioned covariates. With this framework, we additionally provide the first polynomial-factor improvement over brute-force search for constant sparsity $t$ and arbitrary covariance $\Sigma$.</details>

### Feature Dropout: Revisiting the Role of Augmentations in Contrastive Learning

**Authors:** Alex Tamkin, Margalit Glasgow, Xiluo He, Noah Goodman

**<details><summary>Abstract**</summary>

What role do augmentations play in contrastive learning? Recent work suggests that good augmentations are label-preserving with respect to a specific downstream task. We complicate this picture by showing that label-destroying augmentations can be useful in the foundation model setting, where the goal is to learn diverse, general-purpose representations for multiple downstream tasks. We perform contrastive learning experiments on a range of image and audio datasets with multiple downstream tasks (e.g. for digits superimposed on photographs, predicting the class of one vs. the other). We find that Viewmaker Networks, a recently proposed model for learning augmentations for contrastive learning, produce label-destroying augmentations that stochastically destroy features needed for different downstream tasks. These augmentations are interpretable (e.g. altering shapes, digits, or letters added to images) and surprisingly often result in better performance compared to expert-designed augmentations, despite not preserving label information. To support our empirical results, we theoretically analyze a simple contrastive learning setting with a linear model. In this setting, label-destroying augmentations are crucial for preventing one set of features from suppressing the learning of features useful for another downstream task. Our results highlight the need for analyzing the interaction between multiple downstream tasks when trying to explain the success of foundation models.</details>

### FedL2P: Federated Learning to Personalize

**Authors:** Royson Lee, Minyoung Kim, Da Li, Xinchi Qiu, Timothy Hospedales, Ferenc Huszar, Nicholas Lane

**<details><summary>Abstract**</summary>

Federated learning (FL) research has made progress in developing algorithms for distributed learning of global models, as well as algorithms for local personalization of those common models to the specifics of each client’s local data distribution. However, different FL problems may require different personalization strategies, and it may not even be possible to define an effective one-size-fits-all personalization strategy for all clients: Depending on how similar each client’s optimal predictor is to that of the global model, different personalization strategies may be preferred. In this paper, we consider the federated meta-learning problem of learning personalization strategies. Specifically, we consider meta-nets that induce the batch-norm and learning rate parameters for each client given local data statistics. By learning these meta-nets through FL, we allow the whole FL network to collaborate in learning a customized personalization strategy for each client. Empirical results show that this framework improves on a range of standard hand-crafted personalization baselines in both label and feature shift situations.</details>

### Fine-Grained Cross-View Geo-Localization Using a Correlation-Aware Homography Estimator

**Authors:** Xiaolong Wang, Runsen Xu, Zhuofan Cui, Zeyu Wan, Yu Zhang

**<details><summary>Abstract**</summary>

In this paper, we introduce a novel approach to fine-grained cross-view geo-localization. Our method aligns a warped ground image with a corresponding GPS-tagged satellite image covering the same area using homography estimation. We first employ a differentiable spherical transform, adhering to geometric principles, to accurately align the perspective of the ground image with the satellite map. This transformation effectively places ground and aerial images in the same view and on the same plane, reducing the task to an image alignment problem. To address challenges such as occlusion, small overlapping range, and seasonal variations, we propose a robust correlation-aware homography estimator to align similar parts of the transformed ground image with the satellite image. Our method achieves sub-pixel resolution and meter-level GPS accuracy by mapping the center point of the transformed ground image to the satellite image using a homography matrix and determining the orientation of the ground camera using a point above the central axis. Operating at a speed of 30 FPS, our method outperforms state-of-the-art techniques, reducing the mean metric localization error by 21.3\% and 32.4\% in same-area and cross-area generalization tasks on the VIGOR benchmark, respectively, and by 34.4\% on the KITTI benchmark in same-area evaluation.</details>

### Fine-grained Expressivity of Graph Neural Networks

**Authors:** Jan Böker, Ron Levie, Ningyuan Huang, Soledad Villar, Christopher Morris

**<details><summary>Abstract**</summary>

Numerous recent works have analyzed the expressive power of message-passing graph neural networks (MPNNs), primarily utilizing combinatorial techniques such as the $1$-dimensional Weisfeiler--Leman test ($1$-WL) for the graph isomorphism problem. However, the graph isomorphism objective is inherently binary, not giving insights into the degree of similarity between two given graphs. This work resolves this issue by considering continuous extensions of both $1$-WL and MPNNs to graphons. Concretely, we show that the continuous variant of $1$-WL delivers an accurate topological characterization of the expressive power of MPNNs on graphons, revealing which graphs these networks can distinguish and the level of difficulty in separating them. We identify the finest topology where MPNNs separate points and prove a universal approximation theorem. Consequently, we provide a theoretical framework for graph and graphon similarity combining various topological variants of classical characterizations of the $1$-WL. In particular, we characterize the expressive power of MPNNs in terms of the tree distance, which is a graph distance based on the concept of fractional isomorphisms, and substructure counts via tree homomorphisms, showing that these concepts have the same expressive power as the $1$-WL and MPNNs on graphons. Empirically, we validate our theoretical findings by showing that randomly initialized MPNNs, without training, exhibit competitive performance compared to their trained counterparts. Moreover, we evaluate different MPNN architectures based on their ability to preserve graph distances, highlighting the significance of our continuous $1$-WL test in understanding MPNNs' expressivity.</details>

### Finite-Time Analysis of Whittle Index based Q-Learning for Restless Multi-Armed Bandits with Neural Network Function Approximation

**Authors:** GUOJUN XIONG, Jian Li

**<details><summary>Abstract**</summary>

Whittle index policy is a heuristic to the intractable restless multi-armed bandits (RMAB) problem. Although it is provably asymptotically optimal, finding Whittle indices remains difficult.  In this paper, we present Neural-Q-Whittle, a Whittle index based Q-learning algorithm for RMAB with neural network function approximation, which is an example of  nonlinear two-timescale stochastic approximation with Q-function values updated on a faster timescale and Whittle indices on a slower timescale.  Despite the empirical success of deep Q-learning, the non-asymptotic convergence rate of Neural-Q-Whittle, which couples neural networks with two-timescale Q-learning largely remains unclear.  This paper provides a finite-time analysis of Neural-Q-Whittle, where data are generated from a Markov chain, and Q-function is approximated by a ReLU neural network. Our analysis leverages a Lyapunov drift approach to capture the evolution of two coupled parameters, and the nonlinearity in value function approximation further requires us to characterize the approximation error.  Combing these provide Neural-Q-Whittle  with $\mathcal{O}(1/k^{2/3})$ convergence rate, where $k$ is the number of iterations.</details>

### FlatMatch: Bridging Labeled Data and Unlabeled Data with Cross-Sharpness for Semi-Supervised Learning

**Authors:** Zhuo Huang, Li Shen, Jun Yu, Bo Han, Tongliang Liu

**<details><summary>Abstract**</summary>

Semi-Supervised Learning (SSL) has been an effective way to leverage abundant unlabeled data with extremely scarce labeled data. However, most SSL methods are commonly based on instance-wise consistency between different data transformations. Therefore, the label guidance on labeled data is hard to be propagated to unlabeled data. Consequently, the learning process on labeled data is much faster than on unlabeled data which is likely to fall into a local minima that does not favor unlabeled data, leading to sub-optimal generalization performance. In this paper, we propose FlatMatch which minimizes a cross-sharpness measure to ensure consistent learning performance between the two datasets. Specifically, we increase the empirical risk on labeled data to obtain a worst-case model which is a failure case needing to be enhanced. Then, by leveraging the richness of unlabeled data, we penalize the prediction difference (i.e., cross-sharpness) between the worst-case model and the original model so that the learning direction is beneficial to generalization on unlabeled data. Therefore, we can calibrate the learning process without being limited to insufficient label information. As a result, the mismatched learning performance can be mitigated, further enabling the effective exploitation of unlabeled data and improving SSL performance. Through comprehensive validation, we show FlatMatch achieves state-of-the-art results in many SSL settings.</details>

### ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning

**Authors:** Junguang Jiang, Baixu Chen, Junwei Pan, Ximei Wang, Dapeng Liu, Jie Jiang, Mingsheng Long

**<details><summary>Abstract**</summary>

Auxiliary-Task Learning (ATL) aims to improve the performance of the target task by leveraging the knowledge obtained from related tasks. Occasionally, learning multiple tasks simultaneously results in lower accuracy than learning only the target task, which is known as negative transfer. This problem is often attributed to the gradient conflicts among tasks, and is frequently tackled by coordinating the task gradients in previous works. However, these optimization-based methods largely overlook the auxiliary-target generalization capability. To better understand the root cause of negative transfer, we experimentally investigate it from both optimization and generalization perspectives. Based on our findings, we introduce ForkMerge, a novel approach that periodically forks the model into multiple branches, automatically searches the varying task weights by minimizing target validation errors, and dynamically merges all branches to filter out detrimental task-parameter updates. On a series of auxiliary-task learning benchmarks, ForkMerge outperforms existing methods and effectively mitigates negative transfer.</details>

### Fragment-based Pretraining and Finetuning on Molecular Graphs

**Authors:** Kha-Dinh Luong, Ambuj K Singh

**<details><summary>Abstract**</summary>

Property prediction on molecular graphs is an important application of Graph Neural Networks (GNNs). Recently, unlabeled molecular data has become abundant, which facilitates the rapid development of self-supervised learning for GNNs in the chemical domain. In this work, we propose pretraining GNNs at the fragment level, a promising middle ground to overcome the limitations of node-level and graph-level pretraining. Borrowing techniques from recent work on principal subgraph mining, we obtain a compact vocabulary of prevalent fragments from a large pretraining dataset. From the extracted vocabulary, we introduce several fragment-based contrastive and predictive pretraining tasks. The contrastive learning task jointly pretrains two different GNNs: one on molecular graphs and the other on fragment graphs, which represents higher-order connectivity within molecules. By enforcing consistency between the fragment embedding and the aggregated embedding of the corresponding atoms from the molecular graphs, we ensure that the embeddings capture structural information at multiple resolutions. The structural information of fragment graphs is further exploited to extract auxiliary labels for graph-level predictive pretraining. We employ both the pretrained molecular-based and fragment-based GNNs for downstream prediction, thus utilizing the fragment information during finetuning. Our graph fragment-based pretraining (GraphFP) advances the performances on 5 out of 8 common molecular benchmarks and improves the performances on long-range biological benchmarks by at least 11.5%. Code is available at: https://github.com/lvkd84/GraphFP.</details>

### Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization

**Authors:** Kamil Dreczkowski, Antoine Grosnit, Haitham Bou Ammar

**<details><summary>Abstract**</summary>

This paper introduces a modular framework for Mixed-variable and Combinatorial Bayesian Optimization (MCBO) to address the lack of systematic benchmarking and standardized evaluation in the field. Current MCBO papers often introduce non-diverse or non-standard benchmarks to evaluate their methods, impeding the proper assessment of different MCBO primitives and their combinations. Additionally, papers introducing a solution for a single MCBO primitive often omit benchmarking against baselines that utilize the same methods for the remaining primitives. This omission is primarily due to the significant implementation overhead involved, resulting in a lack of controlled assessments and an inability to showcase the merits of a contribution effectively.To overcome these challenges, our proposed framework enables an effortless combination of Bayesian Optimization components, and provides a diverse set of synthetic and real-world benchmarking tasks. Leveraging this flexibility, we implement 47 novel MCBO algorithms and benchmark them against seven existing MCBO solvers and five standard black-box optimization algorithms on ten tasks, conducting over 4000 experiments. Our findings reveal a superior combination of MCBO primitives outperforming existing approaches and illustrate the significance of model fit and the use of a trust region. We make our MCBO library available under the MIT license at \url{https://github.com/huawei-noah/HEBO/tree/master/MCBO}.</details>

### FreeMask: Synthetic Images with Dense Annotations Make Stronger Segmentation Models

**Authors:** Lihe Yang, Xiaogang Xu, Bingyi Kang, Yinghuan Shi, Hengshuang Zhao

**<details><summary>Abstract**</summary>

Semantic segmentation has witnessed tremendous progress due to the proposal of various advanced network architectures. However, they are extremely hungry for delicate annotations to train, and the acquisition is laborious and unaffordable. Therefore, we present FreeMask in this work, which resorts to synthetic images from generative models to ease the burden of both data collection and annotation procedures. Concretely, we first synthesize abundant training images conditioned on the semantic masks provided by realistic datasets. This yields extra well-aligned image-mask training pairs for semantic segmentation models. We surprisingly observe that, solely trained with synthetic images, we already achieve comparable performance with real ones (e.g., 48.3 vs. 48.5 mIoU on ADE20K, and 49.3 vs. 50.5 on COCO-Stuff). Then, we investigate the role of synthetic images by joint training with real images, or pre-training for real images. Meantime, we design a robust filtering principle to suppress incorrectly synthesized regions. In addition, we propose to inequally treat different semantic masks to prioritize those harder ones and sample more corresponding synthetic images for them. As a result, either jointly trained or pre-trained with our filtered and re-sampled synthesized images, segmentation models can be greatly enhanced, e.g., from 48.7 to 52.0 on ADE20K.</details>

### Function Space Bayesian Pseudocoreset for Bayesian Neural Networks

**Authors:** Balhae Kim, Hyungi Lee, Juho Lee

**<details><summary>Abstract**</summary>

A Bayesian pseudocoreset is a compact synthetic dataset summarizing essential information of a large-scale dataset and thus can be used as a proxy dataset for scalable Bayesian inference. Typically, a Bayesian pseudocoreset is constructed by minimizing a divergence measure between the posterior conditioning on the pseudocoreset and the posterior conditioning on the full dataset. However, evaluating the divergence can be challenging, particularly for the models like deep neural networks having high-dimensional parameters. In this paper, we propose a novel Bayesian pseudocoreset construction method that operates on a function space. Unlike previous methods, which construct and match the coreset and full data posteriors in the space of model parameters (weights), our method constructs variational approximations to the coreset posterior on a function space and matches it to the full data posterior in the function space. By working directly on the function space, our method could bypass several challenges that may arise when working on a weight space, including limited scalability and multi-modality issue. Through various experiments, we demonstrate that the Bayesian pseudocoresets constructed from our method enjoys enhanced uncertainty quantification and better robustness across various model architectures.</details>

### [Spotlight] GLIME: General, Stable and Local LIME Explanation

**Authors:** Zeren Tan, Yang Tian, Jian Li

**<details><summary>Abstract**</summary>

As black-box machine learning models become more complex and are applied in high-stakes settings, the need for providing explanations for their predictions becomes crucial. Although Local Interpretable Model-agnostic Explanations (LIME) \cite{ribeiro2016should} is a widely adopted method for understanding model behavior, it suffers from instability with respect to random seeds \cite{zafar2019dlime, shankaranarayana2019alime, bansal2020sam} and exhibits low local fidelity (i.e., how the explanation explains model's local behaviors) \cite{rahnama2019study, laugel2018defining}. Our study demonstrates that this instability is caused by small sample weights, resulting in the dominance of regularization and slow convergence. Additionally, LIME's sampling approach is non-local and biased towards the reference, leading to diminished local fidelity and instability to references. To address these challenges, we propose \textsc{Glime}, an enhanced framework that extends LIME and unifies several previous methods. Within the \textsc{Glime} framework, we derive an equivalent formulation of LIME that achieves significantly faster convergence and improved stability. By employing a local and unbiased sampling distribution, \textsc{Glime} generates explanations with higher local fidelity compared to LIME, while being independent of the reference choice. Moreover, \textsc{Glime} offers users the flexibility to choose sampling distribution based on their specific scenarios.</details>

### GNeSF: Generalizable Neural Semantic Fields

**Authors:** Hanlin Chen, Chen Li, Mengqi Guo, Zhiwen Yan, Gim Hee Lee

**<details><summary>Abstract**</summary>

3D scene segmentation based on neural implicit representation has emerged recently with the advantage of training only on 2D supervision. However, existing approaches still requires expensive per-scene optimization that prohibits generalization to novel scenes during inference. To circumvent this problem, we introduce a \textit{generalizable} 3D segmentation framework based on implicit representation. Specifically, our framework takes in multi-view image features and semantic maps as the inputs instead of only spatial information to avoid overfitting to scene-specific geometric and semantic information. We propose a novel soft voting mechanism to aggregate the 2D semantic information from different views for each 3D point. In addition to the image features, view difference information is also encoded in our framework to predict the voting scores. Intuitively, this allows the semantic information from nearby views to contribute more compared to distant ones. Furthermore, a visibility module is also designed to detect and filter out detrimental information from occluded views. Due to the generalizability of our proposed method, we can synthesize semantic maps or conduct 3D semantic segmentation for novel scenes with solely 2D semantic supervision. Experimental results show that our approach achieves comparable performance with scene-specific approaches. More importantly, our approach can even outperform existing strong supervision-based approaches with only 2D annotations.</details>

### GSLB: The Graph Structure Learning Benchmark

**Authors:** Zhixun Li, Liang Wang, Xin Sun, Yifan Luo, Yanqiao Zhu, Dingshuo Chen, Yingtao Luo, Xiangxin Zhou, Qiang Liu, Shu Wu, Liang Wang, Jeffrey Yu

**<details><summary>Abstract**</summary>

Graph Structure Learning (GSL) has recently garnered considerable attention due to its ability to optimize both the parameters of Graph Neural Networks (GNNs) and the computation graph structure simultaneously. Despite the proliferation of GSL methods developed in recent years, there is no standard experimental setting or fair comparison for performance evaluation, which creates a great obstacle to understanding the progress in this field. To fill this gap, we systematically analyze the performance of GSL in different scenarios and develop a comprehensive Graph Structure Learning Benchmark (GSLB) curated from 20 diverse graph datasets and 16 distinct GSL algorithms. Specifically, GSLB systematically investigates the characteristics of GSL in terms of three dimensions: effectiveness, robustness, and complexity. We comprehensively evaluate state-of-the-art GSL algorithms in node- and graph-level tasks, and analyze their performance in robust learning and model complexity. Further, to facilitate reproducible research, we have developed an easy-to-use library for training, evaluating, and visualizing different GSL methods. Empirical results of our extensive experiments demonstrate the ability of GSL and reveal its potential benefits on various downstream tasks, offering insights and opportunities for future research. The code of GSLB is available at: https://github.com/GSL-Benchmark/GSLB.</details>

### Gaussian Differential Privacy on Riemannian Manifolds

**Authors:** Yangdi Jiang, Xiaotian Chang, Yi Liu, Lei Ding, Linglong Kong, Bei Jiang

**<details><summary>Abstract**</summary>

We develop an advanced approach for extending Gaussian Differential Privacy (GDP) to general Riemannian manifolds. The concept of GDP stands out as a prominent privacy definition that strongly warrants extension to manifold settings, due to its central limit properties. By harnessing the power of the renowned Bishop-Gromov theorem in geometric analysis, we propose a Riemannian Gaussian distribution that integrates the Riemannian distance, allowing us to achieve GDP in Riemannian manifolds with bounded Ricci curvature. To the best of our knowledge, this work marks the first instance of extending the GDP framework to accommodate general Riemannian manifolds, encompassing curved spaces, and circumventing the reliance on tangent space summaries. We provide a simple algorithm to evaluate the privacy budget $\mu$ on any one-dimensional manifold and introduce a versatile Markov Chain Monte Carlo (MCMC)-based algorithm to calculate $\mu$ on any Riemannian manifold with constant curvature. Through simulations on one of the most prevalent manifolds in statistics, the unit sphere $S^d$, we demonstrate the superior utility of our Riemannian Gaussian mechanism in comparison to the previously proposed Riemannian Laplace mechanism for implementing GDP.</details>

### GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image

**Authors:** Mingjian Zhu, Hanting Chen, Qiangyu YAN, Xudong Huang, Guanyu Lin, Wei Li, Zhijun Tu, Hailin Hu, Jie Hu, Yunhe Wang

**<details><summary>Abstract**</summary>

The extraordinary ability of generative models to generate photographic images has intensified concerns about the spread of disinformation, thereby leading to the demand for detectors capable of distinguishing between AI-generated fake images and real images. However, the lack of large datasets containing images from the most advanced image generators poses an obstacle to the development of such detectors. In this paper, we introduce the GenImage dataset, which has the following advantages: 1) Plenty of Images, including over one million pairs of AI-generated fake images and collected real images. 2) Rich Image Content, encompassing a broad range of image classes. 3) State-of-the-art Generators, synthesizing images with advanced diffusion models and GANs. The aforementioned advantages allow the detectors trained on GenImage to undergo a thorough evaluation and demonstrate strong applicability to diverse images. We conduct a comprehensive analysis of the dataset and propose two tasks for evaluating the detection method in resembling real-world scenarios. The cross-generator image classification task measures the performance of a detector trained on one generator when tested on the others. The degraded image classification task assesses the capability of the detectors in handling degraded images such as low-resolution, blurred, and compressed images. With the GenImage dataset, researchers can effectively expedite the development and evaluation of superior AI-generated image detectors in comparison to prevailing methodologies.</details>

### GenS: Generalizable Neural Surface Reconstruction from Multi-View Images

**Authors:** Rui Peng, Xiaodong Gu, Luyang Tang, Shihe Shen, Fanqi Yu, Ronggang Wang

**<details><summary>Abstract**</summary>

Combining the signed distance function (SDF) and differentiable volume rendering has emerged as a powerful paradigm for surface reconstruction from multi-view images without 3D supervision. However, current methods are impeded by requiring long-time per-scene optimizations and cannot generalize to new scenes. In this paper, we present GenS, an end-to-end generalizable neural surface reconstruction model, which performs well in both sparse and dense settings. Unlike coordinate-based methods that train a separate network for each scene, we construct a generalized multi-scale volume to directly encode all scenes. Compared with existing solutions, our representation is more powerful, which can recover high-frequency details while maintaining global smoothness. Meanwhile, we introduce a multi-scale feature-metric consistency to impose the multi-view consistency in a more discriminative multi-scale feature space, which is robust to the failures of the photometric consistency. And the learnable feature can be self-enhanced to continuously improve the matching accuracy and mitigate aggregation ambiguity. Furthermore, we design a view contrast loss to force the model to be robust to those regions covered by few viewpoints through distilling the geometric prior from dense input to sparse input. Extensive experiments on popular benchmarks show that our model can generalize well to new scenes and outperform existing state-of-the-art methods even those employing ground-truth depth supervision. Code will be available at https://github.com/prstrive/GenS.</details>

### Generalizable Lightweight Proxy for Robust NAS against Diverse Perturbations

**Authors:** Hyeonjeong Ha, Minseon Kim, Sung Ju Hwang

**<details><summary>Abstract**</summary>

Recent neural architecture search (NAS) frameworks have been successful in finding optimal architectures for given conditions (e.g., performance or latency). However, they search for optimal architectures in terms of their performance on clean images only, while robustness against various types of perturbations or corruptions is crucial in practice. Although there exist several robust NAS frameworks that tackle this issue by integrating adversarial training into one-shot NAS, however, they are limited in that they only consider robustness against adversarial attacks and require significant computational resources to discover optimal architectures for a single task, which makes them impractical in real-world scenarios. To address these challenges, we propose a novel lightweight robust zero-cost proxy that considers the consistency across features, parameters, and gradients of both clean and perturbed images at the initialization state. Our approach facilitates an efficient and rapid search for neural architectures capable of learning generalizable features that exhibit robustness across diverse perturbations. The experimental results demonstrate that our proxy can rapidly and efficiently search for neural architectures that are consistently robust against various perturbations on multiple benchmark datasets and diverse search spaces, largely outperforming existing clean zero-shot NAS and robust NAS with reduced search cost.</details>

### Generalizable One-shot 3D Neural Head Avatar

**Authors:** Xueting Li, Shalini De Mello, Sifei Liu, Koki Nagano, Umar Iqbal, Jan Kautz

**<details><summary>Abstract**</summary>

We present a method that reconstructs and animates a 3D head avatar from a single-view portrait image. Existing methods either involve time-consuming optimization for a specific person with multiple images, or they struggle to synthesize intricate appearance details beyond the facial region. To address these limitations, we propose a framework that not only generalizes to unseen identities based on a single-view image without requiring person-specific optimization, but also captures characteristic details within and beyond the face area (e.g. hairstyle, accessories, etc.). At the core of our method are three branches that produce three tri-planes representing the coarse 3D geometry, detailed appearance of a source image, as well as the expression of a target image. By applying volumetric rendering to the combination of the three tri-planes followed by a super-resolution module, our method yields a high fidelity image of the desired identity, expression and pose. Once trained, our model enables efficient 3D head avatar reconstruction and animation via a single forward pass through a network. Experiments show that the proposed approach generalizes well to unseen validation datasets, surpassing SOTA baseline methods by a large margin on head avatar reconstruction and animation.</details>

### Geometry-Informed Neural Operator for Large-Scale 3D PDEs

**Authors:** Zongyi Li, Nikola Kovachki, Chris Choy, Boyi Li, Jean Kossaifi, Shourya Otta, Mohammad Amin Nabian, Maximilian Stadler, Christian Hundt, Kamyar Azizzadenesheli, Animashree Anandkumar

**<details><summary>Abstract**</summary>

We propose the geometry-informed neural operator (GINO), a highly efficient approach to learning the solution operator of large-scale partial differential equations with varying geometries. GINO uses a signed distance function (SDF) representation of the input shape and neural operators based on graph and Fourier architectures to learn the solution operator. The graph neural operator handles irregular grids and transforms them into and from regular latent grids on which Fourier neural operator can be efficiently applied. We provide an efficient implementation of GINO using an optimized hashing approach, which allows efficient learning  in a shared, compressed latent space with reduced computation and memory costs. GINO is discretization-invariant, meaning the trained model can be applied to arbitrary discretizations of the continuous domain and applies to any shape or resolution.  To empirically validate the performance of our method on large-scale simulation, we generate the industry-standard aerodynamics dataset of 3D vehicle geometries with Reynolds numbers as high as five million. For this large-scale 3D fluid simulation, numerical methods are expensive to compute surface pressure. We successfully trained GINO to predict the pressure on car surfaces using only five hundred data points. The cost-accuracy experiments show a 26,000x speed-up compared to optimized GPU-based computational fluid dynamics (CFD) simulators on computing the drag coefficient. When tested on new combinations of geometries and boundary conditions (inlet velocities), GINO obtains a one-fourth reduction in error rate compared to deep neural network approaches.</details>

### Glance and Focus: Memory Prompting for Multi-Event Video Question Answering

**Authors:** Ziyi Bai, Ruiping Wang, Xilin Chen

**<details><summary>Abstract**</summary>

Video Question Answering (VideoQA) has emerged as a vital tool to evaluate agents’ ability to understand human daily behaviors. Despite the recent success of large vision language models in many multi-modal tasks, complex situation reasoning over videos involving multiple human-object interaction events still remains challenging. In contrast, humans can easily tackle it by using a series of episode memories as anchors to quickly locate question-related key moments for reasoning. To mimic this effective reasoning strategy, we propose the Glance- Focus model. One simple way is to apply an action detection model to predict a set of actions as key memories. However, these actions within a closed set vocabulary are hard to generalize to various video domains. Instead of that, we train an Encoder-Decoder to generate a set of dynamic event memories at the glancing stage. Apart from using supervised bipartite matching to obtain the event memories, we further design an unsupervised memory generation method to get rid of dependence on event annotations. Next, at the focusing stage, these event memories act as a bridge to establish the correlation between the questions with high-level event concepts and low-level lengthy video content. Given the question, the model first focuses on the generated key event memory, then focuses on the most relevant moment for reasoning through our designed multi-level cross- attention mechanism. We conduct extensive experiments on four Multi-Event VideoQA benchmarks including STAR, EgoTaskQA, AGQA, and NExT-QA. Our proposed model achieves state-of-the-art results, surpassing current large models in various challenging reasoning tasks. The code and models are available at https://github.com/ByZ0e/Glance-Focus.</details>

### Global Identifiability of  $\ell_1$-based Dictionary Learning via Matrix Volume Optimization

**Authors:** Jingzhou Hu, Kejun Huang

**<details><summary>Abstract**</summary>

We propose a novel formulation for dictionary learning that minimizes the determinant of the dictionary matrix, also known as its volume, subject to the constraint that each row of the sparse coefficient matrix has unit $\ell_1$ norm. The main motivation for the proposed formulation is that it provides global identifiability guarantee of the groundtruth dictionary and sparse coefficient matrices, up to the inherent and inconsequential permutation and scaling ambiguity, if a set of vectors obtained from the coefficient matrix lies inside the $\ell_\infty$ norm ball but contains the $\ell_2$ norm ball in their convex hull. Unlike existing work on identifiability of dictionary learning, our result is global, meaning that a globally optimal solution to our proposed formulation has to be a permuted and rescaled version of the groundtruth factors. Another major improvement in our result is that there is no additional assumption on the dictionary matrix other than it is nonsingular, unlike most other work that require the atoms of the dictionary to be mutually incoherent. We also provide a probabilistic analysis and show that if the sparse coefficient matrix is generated from the widely adopted Bernoulli-Gaussian model, then it is globally identifiable if the sample size is bigger than a constant times $k\log k$, where $k$ is the number atoms in the dictionary, with overwhelming probability. The bound is essentially the same as those local identifiability results, but we show that it is also global. Finally, we propose algorithms to solve the new proposed formulation, specifically one based on the linearized-ADMM with efficient per-iteration updates. The proposed algorithms exhibit surprisingly effective performance in correctly and efficiently recovering the dictionary, as demonstrated in the numerical experiments.</details>

### Global Optimality and Finite Sample Analysis of Softmax Off-Policy Actor Critic under State Distribution Mismatch

**Authors:** Shangtong Zhang, Remi Tachet des Combes, Romain Laroche

**<details><summary>Abstract**</summary>

In this paper,  we establish the global optimality and convergence rate of an off-policy actor critic algorithm in the tabular setting without using density ratio to correct the discrepancy between the state distribution of the behavior policy and that of the target policy.  Our work goes beyond existing works on the optimality of policy gradient methods in that  existing works use the exact policy gradient for updating the policy parameters   while we use an approximate and stochastic update step.  Our update step is not a gradient update because we do not use a density ratio to correct the state distribution,  which aligns well with what practitioners do.  Our update is approximate because we use a learned critic instead of the true value function.  Our update is stochastic because at each step the update is done for only the current state action pair.  Moreover,  we remove several restrictive assumptions from existing works in our analysis.  Central to our work is the finite sample analysis of a generic stochastic approximation algorithm with time-inhomogeneous update operators on time-inhomogeneous Markov chains,  based on its uniform contraction properties.</details>

### Global Structure-Aware Diffusion Process for Low-light Image Enhancement

**Authors:** Jinhui HOU, Zhiyu Zhu, Junhui Hou, Hui LIU, Huanqiang Zeng, Hui Yuan

**<details><summary>Abstract**</summary>

This paper studies a diffusion-based framework to address the low-light image enhancement problem. To harness the capabilities of diffusion models, we delve into this intricate process and advocate for the regularization of its inherent ODE-trajectory. To be specific, inspired by the recent research that low curvature ODE-trajectory results in a stable and effective diffusion process, we formulate a curvature regularization term anchored in the intrinsic non-local structures of image data, i.e., global structure-aware regularization, which gradually facilitates the preservation of complicated details and the augmentation of contrast during the diffusion process. This incorporation mitigates the adverse effects of noise and artifacts resulting from the diffusion process, leading to a more precise and flexible enhancement. To additionally promote learning in challenging regions, we introduce an uncertainty-guided regularization technique, which wisely relaxes constraints on the most extreme regions of the image. Experimental evaluations reveal that the proposed diffusion-based framework, complemented by rank-informed regularization, attains distinguished performance in low-light enhancement. The outcomes indicate substantial advancements in image quality, noise suppression, and contrast amplification in comparison with state-of-the-art methods. We believe this innovative approach will stimulate further exploration and advancement in low-light image processing, with potential implications for other applications of diffusion models. The code is publicly available at https://github.com/jinnh/GSAD.</details>

### Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces

**Authors:** Martin Ryner, Jan Kronqvist, Johan Karlsson

**<details><summary>Abstract**</summary>

This paper presents a framework for computing the Gromov-Wasserstein problem between two sets of points in low dimensional spaces, where the discrepancy is the squared Euclidean norm.The Gromov-Wasserstein problem is a generalization of the optimal transport problem that finds the assignment between two sets preserving pairwise distances as much as possible. This can be used to quantify the similarity between two formations or shapes, a common problem in AI and machine learning.The problem can be formulated as a Quadratic Assignment Problem (QAP), which is in general computationally intractable even for small problems. Our framework addresses this challenge by reformulating the QAP as an optimization problem with a low-dimensional domain, leveraging the fact that the problem can be expressed as a concave quadratic optimization problem with low rank. The method scales well with the number of points, and it can be used to find the global solution for large-scale problems with thousands of points.We compare the computational complexity of our approach with state-of-the-art methods on synthetic problems and apply it to a near-symmetrical problem which is of particular interest in computational biology.</details>

### [Spotlight] GloptiNets: Scalable Non-Convex Optimization with Certificates

**Authors:** Gaspard Beugnot, Julien Mairal, Alessandro Rudi

**<details><summary>Abstract**</summary>

We present a novel approach to non-convex optimization with certificates, which handles smooth functions on the hypercube or on the torus. Unlike traditional methods that rely on algebraic properties, our algorithm exploits the regularity of the target function intrinsic in the decay of its Fourier spectrum. By defining a tractable family of models, we allow {\em at the same time} to obtain precise certificates and to leverage the advanced and powerful computational techniques developed to optimize neural networks. In this way the scalability of our approach is naturally enhanced by parallel computing with GPUs. Our approach, when applied to the case of polynomials of moderate dimensions but with thousands of coefficients, outperforms the state-of-the-art optimization methods with certificates, as the ones based on Lasserre's hierarchy, addressing problems intractable for the competitors.</details>

### GlyphControl: Glyph Conditional Control for Visual Text Generation

**Authors:** Yukang Yang, Dongnan Gui, YUHUI YUAN, Weicong Liang, Haisong Ding, Han Hu, Kai Chen

**<details><summary>Abstract**</summary>

Recently, there has been an increasing interest in developing diffusion-based text-to-image generative models capable of generating coherent and well-formed visual text. In this paper, we propose a novel and efficient approach called GlyphControl to address this task. Unlike existing methods that rely on character-aware text encoders like ByT5 and require retraining of text-to-image models, our approach leverages additional glyph conditional information to enhance the performance of the off-the-shelf Stable-Diffusion model in generating accurate visual text. By incorporating glyph instructions, users can customize the content, location, and size of the generated text according to their specific requirements. To facilitate further research in visual text generation, we construct a training benchmark dataset called LAION-Glyph. We evaluate the effectiveness of our approach by measuring OCR-based metrics, CLIP score, and FID of the generated visual text. Our empirical evaluations demonstrate that GlyphControl outperforms the recent DeepFloyd IF approach in terms of OCR accuracy, CLIP score, and FID, highlighting the efficacy of our method.</details>

### GlyphControl: Glyph Conditional Controllable Visual Text Generation

**Authors:** Yukang Yang, Dongnan Gui, YUHUI YUAN, Weicong Liang, Haisong Ding, Han Hu, Kai Chen

**<details><summary>Abstract**</summary>

Recently, there has been an increasing interest in developing diffusion-based text-to-image generative models capable of generating coherent and well-formed visual text. In this paper, we propose a novel and efficient approach called GlyphControl to address this task. Unlike existing methods that rely on character-aware text encoders like ByT5 and require retraining of text-to-image models, our approach leverages additional glyph conditional information to enhance the performance of the off-the-shelf Stable-Diffusion model in generating accurate visual text. By incorporating glyph instructions, users can customize the content, location, and size of the generated text according to their specific requirements. To facilitate further research in visual text generation, we construct a training benchmark dataset called LAION-Glyph. We evaluate the effectiveness of our approach by measuring OCR-based metrics, FID, and CLIP scores of the generated visual text. Our empirical evaluations demonstrate that GlyphControl outperforms the recent DeepFloyd IF approach in terms of OCR accuracy, FID, and CLIP scores, highlighting the efficacy of our method.</details>

### Gradient Descent with Linearly Correlated Noise: Theory and Applications to Differential Privacy

**Authors:** Anastasiia Koloskova, Ryan McKenna, Zachary Charles, John Rush, H. Brendan McMahan

**<details><summary>Abstract**</summary>

We study gradient descent under linearly correlated noise. Our work is motivated by recent practical methods for optimization with differential privacy (DP), such as DP-FTRL, which achieve strong performance in settings where privacy amplification techniques are infeasible (such as in federated learning). These methods inject privacy noise through a matrix factorization mechanism, making the noise *linearly correlated* over iterations. We propose a simplified setting that distills key facets of these methods and isolates the impact of linearly correlated noise. We analyze the behavior of gradient descent in this setting, for both convex and non-convex functions. Our analysis is demonstrably tighter than prior work and recovers multiple important special cases exactly (including anticorrelated perturbed gradient descent). We use our results to develop new, effective matrix factorizations for differentially private optimization, and highlight the benefits of these factorizations theoretically and empirically.</details>

### Grounded Decoding: Guiding Text Generation with Grounded Models for Embodied Agents

**Authors:** Wenlong Huang, Fei Xia, Dhruv Shah, Danny Driess, Andy Zeng, Yao Lu, Pete Florence, Igor Mordatch, Sergey Levine, Karol Hausman, brian ichter

**<details><summary>Abstract**</summary>

Recent progress in large language models (LLMs) has demonstrated the ability to learn and leverage Internet-scale knowledge through pre-training with autoregressive models. Unfortunately, applying such models to settings with embodied agents, such as robots, is challenging due to their lack of experience with the physical world, inability to parse non-language observations, and ignorance of rewards or safety constraints that robots may require. On the other hand, language-conditioned robotic policies that learn from interaction data can provide the necessary grounding that allows the agent to be correctly situated in the real world, but such policies are limited by the lack of high-level semantic understanding due to the limited breadth of the interaction data available for training them. Thus, if we want to make use of the semantic knowledge in a language model while still situating it in an embodied setting, we must construct an action sequence that is both likely according to the language model and also realizable according to grounded models of the environment. We frame this as a problem similar to probabilistic filtering: decode a sequence that both has high probability under the language model and high probability under a set of grounded model objectives. We demonstrate how such grounded models can be obtained across three simulation and real-world domains, and that the proposed decoding strategy is able to solve complex, long-horizon embodiment tasks in a robotic setting by leveraging the knowledge of both models.</details>

### Guide Your Agent with Adaptive Multimodal Rewards

**Authors:** Changyeon Kim, Younggyo Seo, Hao Liu, Lisa Lee, Jinwoo Shin, Honglak Lee, Kimin Lee

**<details><summary>Abstract**</summary>

Developing an agent capable of adapting to unseen environments remains a difficult challenge in imitation learning. This work presents Adaptive Return-conditioned Policy (ARP), an efficient framework designed to enhance the agent's generalization ability using natural language task descriptions and pre-trained multimodal encoders. Our key idea is to calculate a similarity between visual observations and natural language instructions in the pre-trained multimodal embedding space (such as CLIP) and use it as a reward signal. We then train a return-conditioned policy using expert demonstrations labeled with multimodal rewards. Because the multimodal rewards provide adaptive signals at each timestep, our ARP effectively mitigates the goal misgeneralization. This results in superior generalization performances even when faced with unseen text instructions, compared to existing text-conditioned policies. To improve the quality of rewards, we also introduce a fine-tuning method for pre-trained multimodal encoders, further enhancing the performance. Video demonstrations and source code are available on the project website: \url{https://sites.google.com/view/2023arp}.</details>

### H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

**Authors:** Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang "Atlas" Wang, Beidi Chen

**<details><summary>Abstract**</summary>

Large Language Models (LLMs), despite their recent impressive accomplishments, are notably cost-prohibitive to deploy, particularly for applications involving long-content generation, such as dialogue systems and story writing. Often, a large amount of transient state information, referred to as the $\mathsf{KV}$ $\mathsf{cache}$, is stored in GPU memory in addition to model parameters, scaling linearly with the sequence length and batch size. In this paper, we introduce a novel approach for implementing the  $\mathsf{KV}$ $\mathsf{cache}$ which significantly reduces its memory footprint.  Our approach is based on the noteworthy observation that a small portion of tokens contributes most of the value when computing attention scores.  We call these tokens Heavy Hitters ($\mathsf{H_2}$). Through a comprehensive investigation, we find that ($i$) the emergence of $\mathsf{H_2}$ is natural and strongly correlates with the frequent co-occurrence of tokens in the text, and ($ii$) removing them results in significant performance degradation. Based on these insights, we propose Heavy Hitter Oracle ($\mathsf{H_2O}$), a $\mathsf{KV}$ $\mathsf{cache}$ eviction policy that dynamically retains a balance of recent and $\mathsf{H_2}$ tokens. We formulate the $\mathsf{KV}$ $\mathsf{cache}$ eviction as a dynamic submodular problem and prove (under mild assumptions) a theoretical guarantee for our novel eviction algorithm which could help guide future work. We validate the accuracy of our algorithm with OPT, LLaMA, and GPT-NeoX across a wide range of tasks. Our implementation of $\mathsf{H_2O}$ with 20\% heavy hitters improves the throughput over three leading inference systems DeepSpeed Zero-Inference, Hugging Face Accelerate, and FlexGen by up to $29\times$, $29\times$, and $3\times$ on OPT-6.7B and OPT-30B. With the same batch size, $\mathsf{H_2O}$ can reduce the latency by up to $1.9\times$.</details>

### HA-ViD: A Human Assembly Video Dataset for Comprehensive Assembly Knowledge Understanding

**Authors:** Hao Zheng, Regina Lee, Yuqian Lu

**<details><summary>Abstract**</summary>

Understanding comprehensive assembly knowledge from videos is critical for futuristic ultra-intelligent industry. To enable technological breakthrough, we present HA-ViD – the first human assembly video dataset that features representative industrial assembly scenarios, natural procedural knowledge acquisition process, and consistent human-robot shared annotations. Specifically, HA-ViD captures diverse collaboration patterns of real-world assembly, natural human behaviors and learning progression during assembly, and granulate action annotations to subject, action verb, manipulated object, target object, and tool. We provide 3222 multi-view and multi-modality videos), 1.5M frames, 96K temporal labels and 2M spatial labels. We benchmark four foundational video understanding tasks: action recognition, action segmentation, object detection and multi-object tracking. Importantly, we analyze their performance and the further reasoning steps for comprehending knowledge in assembly progress, process efficiency, task collaboration, skill parameters and human intention. Details of HA-ViD is available at: https://iai-hrc.github.io/ha-vid.</details>

### HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception

**Authors:** Junkun Yuan, Xinyu Zhang, Hao Zhou, Jian Wang, Zhongwei Qiu, Zhiyin Shao, Shaofeng Zhang, Sifan Long, Kun Kuang, Kun Yao, Junyu Han, Errui Ding, Lanfen Lin, Fei Wu, Jingdong Wang

**<details><summary>Abstract**</summary>

Model pre-training is essential in human-centric perception. In this paper, we first introduce masked image modeling (MIM) as a pre-training approach for this task. Upon revisiting the MIM training strategy, we reveal that human structure priors offer significant potential. Motivated by this insight, we further incorporate an intuitive human structure prior - human parts - into pre-training. Specifically, we employ this prior to guide the mask sampling process. Image patches, corresponding to human part regions, have high priority to be masked out. This encourages the model to concentrate more on body structure information during pre-training, yielding substantial benefits across a range of human-centric perception tasks. To further capture human characteristics, we propose a structure-invariant alignment loss that enforces different masked views, guided by the human part prior, to be closely aligned for the same image. We term the entire method as HAP. HAP simply uses a plain ViT as the encoder yet establishes new state-of-the-art performance on 11 human-centric benchmarks, and on-par result on one dataset. For example, HAP achieves 78.1% mAP on MSMT17 for person re-identification, 86.54% mA on PA-100K for pedestrian attribute recognition, 78.2% AP on MS COCO for 2D pose estimation, and 56.0 PA-MPJPE on 3DPW for 3D pose and shape estimation.</details>

### HT-Step: Aligning Instructional Articles with How-To Videos

**Authors:** Triantafyllos Afouras, Effrosyni Mavroudi, Tushar Nagarajan, Huiyu Wang, Lorenzo Torresani

**<details><summary>Abstract**</summary>

We introduce HT-Step, a large-scale dataset containing temporal annotations of instructional article steps in cooking videos. It includes 122k segment-level annotations over 20k narrated videos (approximately 2.3k hours) of the HowTo100M dataset.Each annotation provides a temporal interval, and a categorical step label from a taxonomy of 4,958 unique steps automatically mined from wikiHow articles which include rich descriptions of each step.Our dataset significantly surpasses existing labeled step datasets in terms of scale, number of tasks, and richness of natural language step descriptions. Based on these annotations, we introduce a strongly supervised benchmark for aligning instructional articles with how-to videos and present a comprehensive evaluation of baseline methods for this task.By publicly releasing these annotations and defining rigorous evaluation protocols and metrics,we hope to significantly accelerate research in the field of procedural activity understanding.</details>

### Harnessing Hard Mixed Samples with Decoupled Regularizer

**Authors:** Zicheng Liu, Siyuan Li, Ge Wang, Lirong Wu, Cheng Tan, Stan Z. Li

**<details><summary>Abstract**</summary>

Mixup is an efficient data augmentation approach that improves the generalization of neural networks by smoothing the decision boundary with mixed data. Recently, dynamic mixup methods have improved previous \textit{static} policies effectively (e.g., linear interpolation) by maximizing target-related salient regions in mixed samples, but excessive additional time costs are not acceptable. These additional computational overheads mainly come from optimizing the mixed samples according to the mixed labels. However, we found that the extra optimizing step may be redundant because label-mismatched mixed samples are informative hard mixed samples for deep models to localize discriminative features. In this paper, we thus are not trying to propose a more complicated dynamic mixup policy but rather an efficient mixup objective function with decoupled regularizer, named decoupled mixup (DM). The primary effect is that DM can adaptively utilize those hard mixed samples to mine discriminative features without losing the original smoothness of mixup. As a result, DM enables static mixup methods to achieve comparable or even exceed the performance of dynamic methods without any extra computation. This also leads to an interesting objective design problem for mixup training that we need to focus on both smoothing the decision boundaries and identifying discriminative features. Extensive experiments on supervised and semi-supervised learning benchmarks across seven datasets validate the effectiveness of DM.</details>

### Hidden Poison: Machine Unlearning Enables Camouflaged Poisoning Attacks

**Authors:** Jimmy Di, Jack Douglas, Jayadev Acharya, Gautam Kamath, Ayush Sekhari

**<details><summary>Abstract**</summary>

We introduce camouflaged data poisoning attacks, a new attack vector that arises in the context of machine unlearning and other settings when model retraining may be induced. An adversary first adds a few carefully crafted points to the training dataset such that the impact on the model's predictions is minimal. The adversary subsequently triggers a request to remove a subset of the introduced points at which point the attack is unleashed and the model's predictions are negatively affected. In particular, we consider clean-label targeted attacks (in which the goal is to cause the model to misclassify a specific test point) on datasets including CIFAR-10, Imagenette, and Imagewoof. This attack is realized by constructing camouflage datapoints that mask the effect of a poisoned dataset. We demonstrate efficacy of our attack when unlearning is performed via retraining from scratch, the idealized setting of machine unlearning which other efficient methods attempt to emulate, as well as against the approximate unlearning approach of Graves et al. (2021).</details>

### Hierarchical Adaptive Value Estimation for Multi-modal Visual Reinforcement Learning

**Authors:** Yangru Huang, Peixi Peng, Yifan Zhao, Haoran Xu, Mengyue Geng, Yonghong Tian

**<details><summary>Abstract**</summary>

Integrating RGB frames with alternative modality inputs is gaining increasing traction in many vision-based reinforcement learning (RL) applications. Existing multi-modal vision-based RL methods usually follow a Global Value Estimation (GVE) pipeline, which uses a fused modality feature to obtain a unified global environmental description. However, such a feature-level fusion paradigm with a single critic may fall short in policy learning as it tends to overlook the distinct values of each modality. To remedy this, this paper proposes a Local modality-customized Value Estimation (LVE) paradigm, which dynamically estimates the contribution and adjusts the importance weight of each modality from a value-level perspective. Furthermore, a task-contextual re-fusion process is developed to achieve a task-level re-balance of estimations from both feature and value levels. To this end, a Hierarchical Adaptive Value Estimation (HAVE) framework is formed, which adaptively coordinates the contributions of individual modalities as well as their collective efficacy. Agents trained by HAVE are able to exploit the unique characteristics of various modalities while capturing their intricate interactions, achieving substantially improved performance. We specifically highlight the potency of our approach within the challenging landscape of autonomous driving, utilizing the CARLA benchmark with neuromorphic event and depth data to demonstrate HAVE's capability and the effectiveness of its distinct components.</details>

### Hierarchical Open-vocabulary Universal Image Segmentation

**Authors:** Xudong Wang, Shufan Li, Konstantinos Kallidromitis, Yusuke Kato, Kazuki Kozuka, Trevor Darrell

**<details><summary>Abstract**</summary>

Open-vocabulary image segmentation aims to partition an image into semantic regions according to arbitrary text descriptions. However, complex visual scenes can be naturally decomposed into simpler parts and abstracted at multiple lev4 els of granularity, introducing inherent segmentation ambiguity. Unlike existing methods that typically sidestep this ambiguity and treat it as an external factor, our approach actively incorporates a hierarchical representation encompassing different semantic-levels into the learning process. We propose a decoupled text-image fusion mechanism and representation learning modules for both “things” and “stuff”. Additionally, we systematically examine the differences that exist in the textual and visual features between these types of categories. Our resulting model, named HIPIE, tackles HIerarchical, oPen-vocabulary, and unIvErsal segmentation tasks within a unified framework. Benchmarked on diverse datasets, e.g., ADE20K,COCO, Pascal-VOC Part, and RefCOCO/RefCOCOg, HIPIE achieves the state-of14 the-art results at various levels of image comprehension, including semantic-level (e.g., semantic segmentation), instance-level (e.g., panoptic/referring segmentationand object detection), as well as part-level (e.g., part/subpart segmentation) tasks.</details>

### Hierarchical Semi-Implicit Variational Inference with Application to Diffusion Model Acceleration

**Authors:** Longlin Yu, Tianyu Xie, Yu Zhu, Tong Yang, Xiangyu Zhang, Cheng Zhang

**<details><summary>Abstract**</summary>

Semi-implicit variational inference (SIVI) has been introduced to expand the analytical variational families by defining expressive semi-implicit distributions in a hierarchical manner. However, the single-layer architecture commonly used in current SIVI methods can be insufficient when the target posterior has complicated structures. In this paper, we propose hierarchical semi-implicit variational inference, called HSIVI, which generalizes SIVI to allow more expressive multi-layer construction of semi-implicit distributions. By introducing auxiliary distributions that interpolate between a simple base distribution and the target distribution, the conditional layers can be trained by progressively matching these auxiliary distributions one layer after another. Moreover, given pre-trained score networks, HSIVI can be used to accelerate the sampling process of diffusion models with the score matching objective. We show that HSIVI significantly enhances the expressiveness of SIVI on several Bayesian inference problems with complicated target distributions. When used for diffusion model acceleration, we show that HSIVI can produce high quality samples comparable to or better than the existing fast diffusion model based samplers with a small number of function evaluations on various datasets.</details>

### High-dimensional Contextual Bandit Problem without Sparsity

**Authors:** Junpei Komiyama, Masaaki Imaizumi

**<details><summary>Abstract**</summary>

In this research, we investigate the high-dimensional linear contextual bandit problem where the number of features $p$ is greater than the budget $T$, or it may even be infinite. Differing from the majority of previous works in this field, we do not impose sparsity on the regression coefficients. Instead, we rely on recent findings on overparameterized models, which enables us to analyze the performance of the minimum-norm interpolating estimator when data distributions have small effective ranks. We propose an explore-then-commit (EtC) algorithm to address this problem and examine its performance. Through our analysis, we derive the optimal rate of the ETC algorithm in terms of $T$ and show that this rate can be achieved by balancing exploration and exploitation. Moreover, we introduce an adaptive explore-then-commit (AEtC) algorithm that adaptively finds the optimal balance. We assess the performance of the proposed algorithms through a series of simulations.</details>

### Holistic Evaluation of Text-to-Image Models

**Authors:** Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, Agrim Gupta, Yunzhi Zhang, Deepak Narayanan, Hannah Teufel, Marco Bellagente, Minguk Kang, Taesung Park, Jure Leskovec, Jun-Yan Zhu, Fei-Fei Li, Jiajun Wu, Stefano Ermon, Percy Liang

**<details><summary>Abstract**</summary>

The stunning qualitative improvement of text-to-image models has led to their widespread attention and adoption. However, we lack a comprehensive quantitative understanding of their capabilities and risks. To fill this gap, we introduce a new benchmark, Holistic Evaluation of Text-to-Image Models (HEIM). Whereas previous evaluations focus mostly on image-text alignment and image quality, we identify 12 aspects, including text-image alignment, image quality, aesthetics, originality, reasoning, knowledge, bias, toxicity, fairness, robustness, multilinguality, and efficiency. We curate 62 scenarios encompassing these aspects and evaluate 26 state-of-the-art text-to-image models on this benchmark. Our results reveal that no single model excels in all aspects, with different models demonstrating different strengths. We release the generated images and human evaluation results for full transparency at https://crfm.stanford.edu/heim/latest and the code at https://github.com/stanford-crfm/helm, which is integrated with the HELM codebase</details>

### Holistic Transfer: Towards Non-Disruptive Fine-Tuning with Partial Target Data

**Authors:** Cheng-Hao Tu, Hong-You Chen, Zheda Mai, Jike Zhong, Vardaan Pahuja, Tanya Berger-Wolf, Song Gao, Charles Stewart, Yu Su, Wei-Lun (Harry) Chao

**<details><summary>Abstract**</summary>

We propose a learning problem involving adapting a pre-trained source model to the target domain for classifying all classes that appeared in the source data, using target data that covers only a partial label space. This problem is practical, as it is unrealistic for the target end-users to collect data for all classes prior to adaptation. However, it has received limited attention in the literature. To shed light on this issue, we construct benchmark datasets and conduct extensive experiments to uncover the inherent challenges. We found a dilemma --- on the one hand, adapting to the new target domain is important to claim better performance; on the other hand, we observe that preserving the classification accuracy of classes missing in the target adaptation data is highly challenging, let alone improving them. To tackle this, we identify two key directions: 1) disentangling domain gradients from classification gradients, and 2) preserving class relationships. We present several effective solutions that maintain the accuracy of the missing classes and enhance the overall performance, establishing solid baselines for holistic transfer of pre-trained models with partial target data.</details>

### How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources

**Authors:** Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Chandu, David Wadden, Kelsey MacMillan, Noah Smith, Iz Beltagy, Hannaneh Hajishirzi

**<details><summary>Abstract**</summary>

In this work we explore recent advances in instruction-tuning language models on a range of open instruction-following datasets. Despite recent claims that open models can be on par with state-of-the-art proprietary models, these claims are often accompanied by limited evaluation, making it difficult to compare models across the board and determine the utility of various resources. We provide a large set of instruction-tuned models from 6.7B to 65B parameters in size, trained on 12 instruction datasets ranging from manually curated (e.g., OpenAssistant) to synthetic and distilled (e.g., Alpaca) and systematically evaluate them on their factual knowledge, reasoning, multilinguality, coding, safety, and open-ended instruction following abilities through a collection of automatic, model-based, and human-based metrics. We further introduce Tülu, our best performing instruction-tuned model suite finetuned on a combination of high-quality open resources.Our experiments show that different instruction-tuning datasets can uncover or enhance specific skills, while no single dataset (or combination) provides the best performance across all evaluations. Interestingly, we find that model and human preference-based evaluations fail to reflect differences in model capabilities exposed by benchmark-based evaluations, suggesting the need for the type of systemic evaluation performed in this work. Our evaluations show that the best model in any given evaluation reaches on average 87% of ChatGPT performance, and 73% of GPT-4 performance, suggesting that further investment in building better base models and instruction-tuning data is required to close the gap. We release our instruction-tuned models, including a fully finetuned 65B Tülu, along with our code, data, and evaluation framework to facilitate future research.</details>

### How to Fine-tune the Model: Unified Model Shift and Model Bias Policy Optimization

**Authors:** Hai Zhang, Hang Yu, Junqiao Zhao, Di Zhang, xiao zhang, Hongtu Zhou, Chang Huang, Chen Ye

**<details><summary>Abstract**</summary>

Designing and deriving effective model-based reinforcement learning (MBRL) algorithms with a performance improvement guarantee is challenging, mainly attributed to the high coupling between model learning and policy optimization. Many prior methods that rely on return discrepancy to guide model learning ignore the impacts of model shift, which can lead to performance deterioration due to excessive model updates. Other methods use performance difference bound to explicitly consider model shift. However, these methods rely on a fixed threshold to constrain model shift, resulting in a heavy dependence on the threshold and a lack of adaptability during the training process. In this paper, we theoretically derive an optimization objective that can unify model shift and model bias and then formulate a fine-tuning process. This process adaptively adjusts the model updates to get a performance improvement guarantee while avoiding model overfitting. Based on these, we develop a straightforward algorithm USB-PO (Unified model Shift and model Bias Policy Optimization). Empirical results show that USB-PO achieves state-of-the-art performance on several challenging benchmark tasks.</details>

### [Oral] How to Turn Your Knowledge Graph Embeddings into Generative Models

**Authors:** Lorenzo Loconte, Nicola Di Mauro, Robert Peharz, Antonio Vergari

**Oral Presentation:** Tu, Dec 12, 08:00 -- Oral 1C

**<details><summary>Abstract**</summary>

Some of the most successful knowledge graph embedding (KGE) models for link prediction – CP, RESCAL, TuckER, ComplEx – can be interpreted as energy-based models. Under this perspective they are not amenable for exact maximum-likelihood estimation (MLE), sampling and struggle to integrate logical constraints. This work re-interprets the score functions of these KGEs as circuits – constrained computational graphs allowing efficient marginalisation. Then, we design two recipes to obtain efficient generative circuit models by either restricting their activations to be non-negative or squaring their outputs. Our interpretation comes with little or no loss of performance for link prediction, while the circuits framework unlocks exact learning by MLE, efficient sampling of new triples, and guarantee that logical constraints are satisfied by design. Furthermore, our models scale more gracefully than the original KGEs on graphs with millions of entities.</details>

### HubRouter: Learning Global Routing via Hub Generation and Pin-hub Connection

**Authors:** Xingbo Du, Chonghua Wang, Ruizhe Zhong, Junchi Yan

**<details><summary>Abstract**</summary>

Global Routing (GR) is a core yet time-consuming task in VLSI systems. It recently attracted efforts from the machine learning community, especially generative models, but they suffer from the non-connectivity of generated routes. We argue that the inherent non-connectivity can harm the advantage of its one-shot generation and has to be post-processed by traditional approaches. Thus, we propose a novel definition, called hub, which represents the key point in the route. Equipped with hubs, global routing is transferred from a pin-pin connection problem to a hub-pin connection problem. Specifically, to generate definitely-connected routes, this paper proposes a two-phase learning scheme named HubRouter, which includes 1) hub-generation phase: A condition-guided hub generator using deep generative models; 2) pin-hub-connection phase: An RSMT construction module that connects the hubs and pins using an actor-critic model. In the first phase, we incorporate typical generative models into a multi-task learning framework to perform hub generation and address the impact of sensitive noise points with stripe mask learning. During the second phase, HubRouter employs an actor-critic model to finish the routing, which is efficient and has very slight errors. Experiments on simulated and real-world global routing benchmarks are performed to show our approach's efficiency, particularly HubRouter outperforms the state-of-the-art generative global routing methods in wirelength, overflow, and running time. Moreover, HubRouter also shows strength in other applications, such as RSMT construction and interactive path replanning.</details>

### Human-in-the-Loop Optimization for Deep Stimulus Encoding in Visual Prostheses

**Authors:** Jacob Granley, Tristan Fauvel, Matthew Chalk, Michael Beyeler

**<details><summary>Abstract**</summary>

Neuroprostheses show potential in restoring lost sensory function and enhancing human capabilities, but the sensations produced by current devices often seem unnatural or distorted. Exact placement of implants and differences in individual perception lead to significant variations in stimulus response, making personalized stimulus optimization a key challenge. Bayesian optimization could be usedto optimize patient-specific stimulation parameters with limited noisy observations, but is not feasible for high-dimensional stimuli. Alternatively, deep learning models can optimize stimulus encoding strategies, but typically assume perfect knowledge of patient-specific variations. Here we propose a novel, practically feasible approach that overcomes both of these fundamental limitations. First, a deep encoder network is trained to produce optimal stimuli for any individual patient by inverting a forward model mapping electrical stimuli to visual percepts. Second, a preferential Bayesian optimization strategy utilizes this encoder to learn the optimal patient-specific parameters for a new patient, using a minimal number of pairwise comparisons between candidate stimuli. We demonstrate the viability of this approach on a novel, state-of-the-art visual prosthesis model. Our approach quickly learns a personalized stimulus encoder and leads to dramatic improvements in the quality of restored vision, outperforming existing encoding strategies. Further, this approach is robust to noisy patient feedback and misspecifications in the underlying forward model. Overall, our results suggest that combining the strengths of deep learning and Bayesian optimization could significantly improve the perceptual experience of patients fitted with visual prostheses and may prove a viable solution for a range of neuroprosthetic technologies</details>

### IMP-MARL: a Suite of Environments for Large-scale Infrastructure Management Planning via MARL

**Authors:** Pascal Leroy, Pablo G. Morato, Jonathan Pisane, Athanasios Kolios, Damien Ernst

**<details><summary>Abstract**</summary>

We introduce IMP-MARL, an open-source suite of multi-agent reinforcement learning (MARL) environments for large-scale Infrastructure Management Planning (IMP), offering a platform for benchmarking the scalability of cooperative MARL methods in real-world engineering applications.In IMP, a multi-component engineering system is subject to a risk of failure due to its components' damage condition.Specifically, each agent plans inspections and repairs for a specific system component, aiming to minimise maintenance costs while cooperating to minimise system failure risk.With IMP-MARL, we release several environments including one related to offshore wind structural systems, in an effort to meet today's needs to improve management strategies to support sustainable and reliable energy systems.Supported by IMP practical engineering environments featuring up to 100 agents, we conduct a benchmark campaign, where the scalability and performance of state-of-the-art cooperative MARL methods are compared against expert-based heuristic policies. The results reveal that centralised training with decentralised execution methods scale better with the number of agents than fully centralised or decentralised RL approaches, while also outperforming expert-based heuristic policies in most IMP environments.Based on our findings, we additionally outline remaining cooperation and scalability challenges that future MARL methods should still address.Through IMP-MARL, we encourage the implementation of new environments and the further development of MARL methods.</details>

### [Spotlight] Imitation Learning from Imperfection: Theoretical Justifications and Algorithms

**Authors:** Ziniu Li, Tian Xu, Zeyu Qin, Yang Yu, Zhi-Quan Luo

**<details><summary>Abstract**</summary>

Imitation learning (IL) algorithms excel in acquiring high-quality policies from expert data for sequential decision-making tasks. But, their effectiveness is hampered when faced with limited expert data. To tackle this challenge, a novel framework called (offline) IL with supplementary data has emerged, which enhances learning by incorporating an additional yet imperfect dataset obtained inexpensively from sub-optimal policies. Nonetheless, learning becomes challenging due to the potential inclusion of out-of-expert-distribution samples. In this work, we pioneer the mathematical formalization of this framework, uncovering its limitations. Our theoretical analysis reveals that a naive approach—applying the behavioral cloning (BC) algorithm concept to the combined set of expert and supplementary data—may fall short of vanilla BC, which solely relies on expert data. This deficiency arises due to the distribution shift between the two data sources. To address this issue, we propose a new importance-sampling-based technique for selecting data within the expert distribution. We prove that the proposed method theoretically eliminates the gap of the naive approach, highlighting its efficacy when handling imperfect data. Empirical studies demonstrate that our method outperforms previous state-of-the-art methods in tasks including robotics locomotion control, Atari video games, and image classification. Overall, our work underscores the potential of improving IL by leveraging diverse data sources through effective data selection.</details>

### [Spotlight] Implicit Bias of Gradient Descent for Logistic Regression at the Edge of Stability

**Authors:** Jingfeng Wu, Vladimir Braverman, Jason Lee

**<details><summary>Abstract**</summary>

Recent research has observed that in machine learning optimization, gradient descent (GD) often operates at the edge of stability (EoS) [Cohen et al., 2021], where the stepsizes are set to be large, resulting in non-monotonic losses induced by the GD iterates. This paper studies the convergence and implicit bias of constant-stepsize GD for logistic regression on linearly separable data in the EoS regime. Despite the presence of local oscillations, we prove that the logistic loss can be minimized by GD with any constant stepsize over a long time scale. Furthermore, we prove that with any constant stepsize, the GD iterates tend to infinity when projected to a max-margin direction (the hard-margin SVM direction) and converge to a fixed vector that minimizes a strongly convex potential when projected to the orthogonal complement of the max-margin direction. In contrast, we also show that in the EoS regime, GD iterates may diverge catastrophically under the exponential loss, highlighting the superiority of the logistic loss. These theoretical findings are in line with numerical simulations and complement existing theories on the convergence and implicit bias of GD for logistic regression, which are only applicable when the stepsizes are sufficiently small.</details>

### Implicit Contrastive Representation Learning with Guided Stop-gradient

**Authors:** Byeongchan Lee, Sehyun Lee

**<details><summary>Abstract**</summary>

In self-supervised representation learning, Siamese networks are a natural architecture for learning transformation-invariance by bringing representations of positive pairs closer together. But it is prone to collapse into a degenerate solution. To address the issue, in contrastive learning, a contrastive loss is used to prevent collapse by moving representations of negative pairs away from each other. But it is known that algorithms with negative sampling are not robust to a reduction in the number of negative samples. So, on the other hand, there are algorithms that do not use negative pairs. Many positive-only algorithms adopt asymmetric network architecture consisting of source and target encoders as a key factor in coping with collapse. By exploiting the asymmetric architecture, we introduce a methodology to implicitly incorporate the idea of contrastive learning. As its implementation, we present a novel method guided stop-gradient. We apply our method to benchmark algorithms SimSiam and BYOL and show that our method stabilizes training and boosts performance. We also show that the algorithms with our method work well with small batch sizes and do not collapse even when there is no predictor. The code is available in the supplementary material.</details>

### Improved Best-of-Both-Worlds Guarantees for Multi-Armed Bandits: FTRL with General Regularizers and Multiple Optimal Arms

**Authors:** Tiancheng Jin, Junyan Liu, Haipeng Luo

**<details><summary>Abstract**</summary>

We study the problem of designing adaptive multi-armed bandit algorithms that perform optimally in both the stochastic setting and the adversarial setting simultaneously (often known as a best-of-both-world guarantee). A line of recent works shows that when configured and analyzed properly, the Follow-the-Regularized-Leader (FTRL) algorithm, originally designed for the adversarial setting, can in fact optimally adapt to the stochastic setting as well. Such results, however, critically rely on an assumption that there exists one unique optimal arm. Recently, Ito [2021] took the first step to remove such an undesirable uniqueness assumption for one particular FTRL algorithm withthe 1/2-Tsallis entropy regularizer. In this work, we significantly improve and generalize this result, showing that uniqueness is unnecessary for FTRL with a broad family of regularizers and a new learning rate schedule. For some regularizers, our regret bounds also improve upon prior results even when uniqueness holds. We further provide an application of our results to the decoupled exploration and exploitation problem, demonstrating that our techniques are broadly applicable.</details>

### Improved Communication Efficiency in Federated Natural Policy Gradient via ADMM-based Gradient Updates

**Authors:** Guangchen Lan, Han Wang, James Anderson, Christopher Brinton, Vaneet Aggarwal

**<details><summary>Abstract**</summary>

Federated reinforcement learning (FedRL) enables agents to collaboratively train a global policy without sharing their individual data. However, high communication overhead remains a critical bottleneck, particularly for natural policy gradient (NPG) methods, which are second-order. To address this issue, we propose the FedNPG-ADMM framework, which leverages the alternating direction method of multipliers (ADMM) to approximate global NPG directions efficiently. We theoretically demonstrate that using ADMM-based gradient updates reduces communication complexity from $\mathcal{O}({d^{2}})$ to $\mathcal{O}({d})$ at each iteration, where $d$ is the number of model parameters. Furthermore, we show that achieving an $\epsilon$-error stationary convergence requires $\mathcal{O}(\frac{1}{(1-\gamma)^{2}{\epsilon}})$ iterations for discount factor $\gamma$, demonstrating that FedNPG-ADMM maintains the same convergence rate as standard FedNPG. Through evaluation of the proposed algorithms in MuJoCo environments, we demonstrate that FedNPG-ADMM maintains the reward performance of standard FedNPG, and that its convergence rate improves when the number of federated agents increases.</details>

### Improving Adversarial Robustness via Information Bottleneck Distillation

**Authors:** Huafeng Kuang, Hong Liu, Yongjian Wu, Shin'ichi Satoh, Rongrong Ji

**<details><summary>Abstract**</summary>

Previous studies have shown that optimizing the information bottleneck can significantly improve the robustness of deep neural networks. Our study closely examines the information bottleneck principle and proposes an Information Bottleneck Distillation approach. This specially designed, robust distillation technique utilizes prior knowledge obtained from a robust pre-trained model to boost information bottlenecks. Specifically, we propose two distillation strategies that align with the two optimization processes of the information bottleneck. Firstly, we use a robust soft-label distillation method to increase the mutual information between latent features and output prediction. Secondly, we introduce an adaptive feature distillation method that automatically transfers relevant knowledge from the teacher model to the student model, thereby reducing the mutual information between the input and latent features. We conduct extensive experiments to evaluate our approach's robustness against state-of-the-art adversarial attackers such as PGD-attack and AutoAttack. Our experimental results demonstrate the effectiveness of our approach in significantly improving adversarial robustness. Our code is available at https://github.com/SkyKuang/IBD.</details>

### Improving Diffusion-Based Image Synthesis with Context Prediction

**Authors:** Ling Yang, Jingwei Liu, Shenda Hong, Zhilong Zhang, Zhilin Huang, Zheming Cai, Wentao Zhang, Bin CUI

**<details><summary>Abstract**</summary>

Diffusion models are a new class of generative models, and have dramatically promoted image generation with unprecedented quality and diversity. Existing diffusion models mainly try to reconstruct input image from a corrupted one with a pixel-wise or feature-wise constraint along spatial axes. However, such point-based reconstruction may fail to make each predicted pixel/feature fully preserve its neighborhood context, impairing diffusion-based image synthesis. As a powerful source of automatic supervisory signal, context has been well studied for learning representations. Inspired by this, we for the first time propose ConPreDiff to improve diffusion-based image synthesis with context prediction. We explicitly reinforce each point to predict its neighborhood context (i.e., multi-stride pixels/features) with a context decoder at the end of diffusion denoising blocks in training stage, and remove the decoder for inference. In this way, each point can better reconstruct itself by preserving its semantic connections with neighborhood context. This new paradigm of ConPreDiff can generalize to arbitrary discrete and continuous diffusion backbones without introducing extra parameters in sampling procedure. Extensive experiments are conducted on unconditional image generation, text-to-image generation and image inpainting tasks. Our ConPreDiff consistently outperforms previous methods and achieves new SOTA text-to-image generation results on MS-COCO, with a zero-shot FID score of 6.21.</details>

### Improving multimodal datasets with image captioning

**Authors:** Thao Nguyen, Samir Yitzhak Gadre, Gabriel Ilharco, Sewoong Oh, Ludwig Schmidt

**<details><summary>Abstract**</summary>

Massive web datasets play a key role in the success of large vision-language models like CLIP and Flamingo. However, the raw web data is noisy, and existing filtering methods to reduce noise often come at the expense of data diversity. Our work focuses on caption quality as one major source of noise, and studies how generated captions can increase the utility of web-scraped datapoints with nondescript text. Through exploring different mixing strategies for raw and generated captions, we outperform the best filtering method proposed by the DataComp benchmark by 2% on ImageNet and 4% on average across 38 tasks, given a candidate pool of 128M image-text pairs. Our best approach is also 2x better at Flickr and MS-COCO retrieval. We then analyze what makes synthetic captions an effective source of text supervision. In experimenting with different image captioning models, we also demonstrate that the performance of a model on standard image captioning benchmarks (e.g., NoCaps CIDEr) is not a reliable indicator of the utility of the captions it generates for multimodal training. Finally, our experiments with using generated captions at DataComp's large scale (1.28B image-text pairs) offer insights into the limitations of synthetic text, as well as the importance of image curation with increasing training data quantity. The synthetic captions used in our experiments are now available on HuggingFace.</details>

### Information Design in Multi-Agent Reinforcement Learning

**Authors:** Yue Lin, Wenhao Li, Hongyuan Zha, Baoxiang Wang

**<details><summary>Abstract**</summary>

Reinforcement learning (RL) is inspired by the way human infants and animals learn from the environment. The setting is somewhat idealized because, in actual tasks, other agents in the environment have their own goals and behave adaptively to the ego agent. To thrive in those environments, the agent needs to influence other agents so their actions become more helpful and less harmful. Research in computational economics distills two ways to influence others directly: by providing tangible goods (mechanism design) and by providing information (information design). This work investigates information design problems for a group of RL agents. The main challenges are two-fold. One is the information provided will immediately affect the transition of the agent trajectories, which introduces additional non-stationarity. The other is the information can be ignored, so the sender must provide information that the receiver is willing to respect. We formulate the Markov signaling game, and develop the notions of signaling gradient and the extended obedience constraints that address these challenges. Our algorithm is efficient on various mixed-motive tasks and provides further insights into computational economics. Our code is publicly available at https://github.com/YueLin301/InformationDesignMARL.</details>

### Inserting Anybody in Diffusion Models via Celeb Basis

**Authors:** Ge Yuan, Xiaodong Cun, Yong Zhang, Maomao Li, Chenyang Qi, Xintao Wang, Ying Shan, Huicheng Zheng

**<details><summary>Abstract**</summary>

Exquisite demand exists for customizing the pretrained large text-to-image model, $e.g.$ Stable Diffusion, to generate innovative concepts, such as the users themselves. However, the newly-added concept from previous customization methods often shows weaker combination abilities than the original ones even given several images during training. We thus propose a new personalization method that allows for the seamless integration of a unique individual into the pre-trained diffusion model using just $one\ facial\ photograph$ and only $1024\ learnable\ parameters$ under $3\ minutes$. So we can effortlessly generate stunning images of this person in any pose or position, interacting with anyone and doing anything imaginable from text prompts. To achieve this, we first analyze and build a well-defined celeb basis from the embedding space of the pre-trained large text encoder. Then, given one facial photo as the target identity, we generate its own embedding by optimizing the weight of this basis and locking all other parameters. Empowered by the proposed celeb basis, the new identity in our customized model showcases a better concept combination ability than previous personalization methods. Besides, our model can also learn several new identities at once and interact with each other where the previous customization model fails to. Project page is at: http://celeb-basis.github.io. Code is at: https://github.com/ygtxr1997/CelebBasis.</details>

### Intensity Profile Projection: A Framework for Continuous-Time Representation Learning for Dynamic Networks

**Authors:** Alexander Modell, Ian Gallagher, Emma Ceccherini, Nick Whiteley, Patrick Rubin-Delanchy

**<details><summary>Abstract**</summary>

We present a new representation learning framework, Intensity Profile Projection, for continuous-time dynamic network data. Given triples $(i,j,t)$, each representing a time-stamped ($t$) interaction between two entities ($i,j$), our procedure returns a continuous-time trajectory for each node, representing its behaviour over time. The framework consists of three stages: estimating pairwise intensity functions, e.g. via kernel smoothing; learning a projection which minimises a notion of intensity reconstruction error; and constructing evolving node representations via the learned projection. The trajectories satisfy two properties, known as structural and temporal coherence, which we see as fundamental for reliable inference. Moreoever, we develop estimation theory providing tight control on the error of any estimated trajectory, indicating that the representations could even be used in quite noise-sensitive follow-on analyses. The theory also elucidates the role of smoothing as a bias-variance trade-off, and shows how we can reduce the level of smoothing as the signal-to-noise ratio increases on account of the algorithm `borrowing strength' across the network.</details>

### Interaction Measures, Partition Lattices and Kernel Tests for High-Order Interactions

**Authors:** Zhaolu Liu, Robert Peach, Pedro A.M Mediano, Mauricio Barahona

**<details><summary>Abstract**</summary>

Models that rely solely on pairwise relationships often fail to capture the complete statistical structure of the complex multivariate data found in diverse domains, such as socio-economic, ecological, or biomedical systems. Non-trivial dependencies between groups of more than two variables can play a significant role in the analysis and modelling of such systems, yet extracting such high-order interactions from data remains challenging. Here, we introduce a hierarchy of $d$-order ($d \geq 2$) interaction measures, increasingly inclusive of possible factorisations of the joint probability distribution, and define non-parametric, kernel-based tests to establish systematically the statistical significance of $d$-order interactions. We also establish mathematical links with lattice theory, which elucidate the derivation of the interaction measures and their composite permutation tests; clarify the connection of simplicial complexes with kernel matrix centring; and provide a means to enhance computational efficiency. We illustrate our results numerically with validations on synthetic data, and through an application to neuroimaging data.</details>

### Interpretability at Scale: Identifying Causal Mechanisms in Alpaca

**Authors:** Zhengxuan Wu, Atticus Geiger, Thomas Icard, Christopher Potts, Noah Goodman

**<details><summary>Abstract**</summary>

Obtaining human-interpretable explanations of large, general-purpose language models is an urgent goal for AI safety. However, it is just as important that our interpretability methods are faithful to the causal dynamics underlying model behavior and able to robustly generalize to unseen inputs. Distributed Alignment Search (DAS) is a powerful gradient descent method grounded in a theory of causal abstraction that uncovered perfect alignments between interpretable symbolic algorithms and small deep learning models fine-tuned for specific tasks. In the present paper, we scale DAS significantly by replacing the remaining brute-force search steps with learned parameters -- an approach we call Boundless DAS. This enables us to efficiently search for interpretable causal structure in large language models while they follow instructions. We apply Boundless DAS to the Alpaca model (7B parameters), which, off the shelf, solves a simple numerical reasoning problem. With Boundless DAS, we discover that Alpaca does this by implementing a causal model with two interpretable boolean variables. Furthermore, we find that the alignment of neural representations with these variables is robust to changes in inputs and instructions. These findings mark a first step toward deeply understanding the inner-workings of our largest and most widely deployed language models.</details>

### Intervention Generalization: A View from Factor Graph Models

**Authors:** Gecia Bravo-Hermsdorff, David Watson, Jialin Yu, Jakob Zeitler, Ricardo Silva

**<details><summary>Abstract**</summary>

One of the goals of causal inference is to generalize from past experiments and observational data to novel conditions. While it is in principle possible to eventually learn a mapping from a novel experimental condition to an outcome of interest, provided a sufficient variety of experiments is available in the training data, coping with a large combinatorial space of possible interventions is hard. Under a typical sparse experimental design, this mapping is ill-posed without relying on heavy regularization or prior distributions. Such assumptions may or may not be reliable, and can be hard to defend or test. In this paper, we take a close look at how to warrant a leap from past experiments to novel conditions based on minimal assumptions about the factorization of the distribution of the manipulated system, communicated in the well-understood language of factor graph models. A postulated interventional factor model (IFM) may not always be informative, but it conveniently abstracts away a need for explicitly modeling unmeasured confounding and feedback mechanisms, leading to directly testable claims. Given an IFM and datasets from a collection of experimental regimes, we derive conditions for identifiability of the expected outcomes of new regimes never observed in these training data. We implement our framework using several efficient algorithms, and apply them on a range of semi-synthetic experiments.</details>

### Invariant Anomaly Detection under Distribution Shifts: A Causal Perspective

**Authors:** João Carvalho, Mengtao Zhang, Robin Geyer, Carlos Cotrini, Joachim M Buhmann

**<details><summary>Abstract**</summary>

Anomaly detection (AD) is the machine learning task of identifying highly discrepant abnormal samples by solely relying on the consistency of the normal training samples. Under the constraints of a distribution shift, the assumption that training samples and test samples are drawn from the same distribution breaks down. In this work, by leveraging tools from causal inference we attempt to increase the resilience of anomaly detection models to different kinds of distribution shifts. We begin by elucidating a simple yet necessary statistical property that ensures invariant representations, which is critical for robust AD under both domain and covariate shifts. From this property, we derive a regularization term which, when minimized, leads to partial distribution invariance across environments. Through extensive experimental evaluation on both synthetic and real-world tasks, covering a range of six different AD methods, we demonstrated significant improvements in out-of-distribution performance. Under both covariate and domain shift, models regularized with our proposed term showed marked increased robustness. Code is available at: https://github.com/JoaoCarv/invariant-anomaly-detection</details>

### Investigating how ReLU-networks encode symmetries

**Authors:** Georg Bökman, Fredrik Kahl

**<details><summary>Abstract**</summary>

Many data symmetries can be described in terms of group equivariance and the most common way of encoding group equivariances in neural networks is by building linear layers that are group equivariant.In this work we investigate whether equivariance of a network implies that all layers are equivariant.On the theoretical side we find cases where equivariance implies layerwise equivariance, but alsodemonstrate that this is not the case generally.Nevertheless, we conjecture that CNNs that are trained to be equivariant will exhibit layerwise equivariance and explain how this conjecture is a weaker version of the recent permutation conjecture by Entezari et al.\ [2022].We perform quantitative experiments with VGG-nets on CIFAR10 and qualitative experiments with ResNets on ImageNet to illustrate and support our theoretical findings. These experiments are not only of interest for understanding how group equivariance is encoded in ReLU-networks, but they also give a new perspective on Entezari et al.'s permutation conjecture as we find that itis typically easier to merge a network with a group-transformed version of itself than merging two different networks.</details>

### [Spotlight] Is Learning in Games Good for the Learners?

**Authors:** William Brown, Jon Schneider, Kiran Vodrahalli

**<details><summary>Abstract**</summary>

We consider a number of questions related to tradeoffs between reward and regret in repeated gameplay between two agents. To facilitate this,  we introduce a notion of generalized equilibrium which allows for asymmetric regret constraints, and yields polytopes of feasible values for each agent and pair of regret constraints, where we show that any such equilibrium is reachable by a pair of algorithms which maintain their regret guarantees against arbitrary opponents. As a central example, we highlight the case one agent is no-swap and the other's regret is unconstrained. We show that this captures an extension of Stackelberg equilibria with a matching optimal value, and that there exists a wide class of games where a player can significantly increase their utility by deviating from a no-swap-regret algorithm against a no-swap learner (in fact, almost any game without pure Nash equilibria is of this form). Additionally, we make use of generalized equilibria to consider tradeoffs in terms of the opponent's algorithm choice. We give a tight characterization for the maximal reward obtainable against some no-regret learner, yet we also show a class of games in which this is bounded away from the value obtainable against the class of common "mean-based" no-regret algorithms. Finally, we consider the question of learning reward-optimal strategies via repeated play with a no-regret agent when the game is initially unknown. Again we show tradeoffs depending on the opponent's learning algorithm: the Stackelberg strategy is learnable in exponential time with any no-regret agent (and in polynomial time with any no-adaptive-regret agent) for any game where it is learnable via queries, and there are games where it is learnable in polynomial time against any no-swap-regret agent but requires exponential time against a mean-based no-regret agent.</details>

### Joint Bayesian Inference of Graphical Structure and Parameters with a Single Generative Flow Network

**Authors:** Tristan Deleu, Mizu Nishikawa-Toomey, Jithendaraa Subramanian, Nikolay Malkin, Laurent Charlin, Yoshua Bengio

**<details><summary>Abstract**</summary>

Generative Flow Networks (GFlowNets), a class of generative models over discrete and structured sample spaces, have been previously applied to the problem of inferring the marginal posterior distribution over the directed acyclic graph (DAG) of a Bayesian Network, given a dataset of observations. Based on recent advances extending this framework to non-discrete sample spaces, we propose in this paper to approximate the joint posterior over not only the structure of a Bayesian Network, but also the parameters of its conditional probability distributions. We use a single GFlowNet whose sampling policy follows a two-phase process: the DAG is first generated sequentially one edge at a time, and then the corresponding parameters are picked once the full structure is known. Since the parameters are included in the posterior distribution, this leaves more flexibility for the local probability models of the Bayesian Network, making our approach applicable even to non-linear models parametrized by neural networks. We show that our method, called JSP-GFN, offers an accurate approximation of the joint posterior, while comparing favorably against existing methods on both simulated and real data.</details>

### Joint Prompt Optimization of Stacked LLMs using Variational Inference

**Authors:** Alessandro Sordoni, Eric Yuan, Marc-Alexandre Côté, Matheus Pereira, Adam Trischler, Ziang Xiao, Arian Hosseini, Friederike Niedtner, Nicolas Le Roux

**<details><summary>Abstract**</summary>

Large language models (LLMs) can be seen as atomic units of computation mapping sequences to a distribution over sequences. Thus, they can be seen as stochastic language layers in a language network, where the learnable parameters are the natural language prompts at each layer. By stacking two such layers and feeding the output of one layer to the next, we obtain a Deep Language Network (DLN). We first show how to effectively perform prompt optimization for a 1-Layer language network (DLN-1). Then, we present an extension that applies to 2-layer DLNs (DLN-2), where two prompts must be learned. The key idea is to consider the output of the first layer as a latent variable, which requires inference, and prompts to be learned as the parameters of the generative distribution. We first test the effectiveness of DLN-1 in multiple reasoning and natural language understanding tasks. Then, we show that DLN-2 can reach higher performance than a single layer, showing promise that we might reach comparable performance to GPT-4, even when each LLM in the network is smaller and less powerful.</details>

### JourneyDB: A Benchmark for Generative Image Understanding

**Authors:** Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, Jifeng Dai, Yu Qiao, Limin Wang, Hongsheng Li

**<details><summary>Abstract**</summary>

While recent advancements in vision-language models have had a transformative impact on multi-modal comprehension, the extent to which these models possess the ability to comprehend generated images remains uncertain. Synthetic images, in comparison to real data, encompass a higher level of diversity in terms of both content and style, thereby presenting significant challenges for the models to fully grasp. In light of this challenge, we introduce a comprehensive dataset, referred to as JourneyDB, that caters to the domain of generative images within the context of multi-modal visual understanding. Our meticulously curated dataset comprises 4 million distinct and high-quality generated images, each paired with the corresponding text prompts that were employed in their creation. Furthermore, we additionally introduce an external subset with results of another 22 text-to-image generative models, which makes JourneyDB a comprehensive benchmark for evaluating the comprehension of generated images. On our dataset, we have devised four benchmarks to assess the performance of generated image comprehension in relation to both content and style interpretation. These benchmarks encompass prompt inversion, style retrieval, image captioning, and visual question answering. Lastly, we evaluate the performance of state-of-the-art multi-modal models when applied to the JourneyDB dataset, providing a comprehensive analysis of their strengths and limitations in comprehending generated content. We anticipate that the proposed dataset and benchmarks will facilitate further research in the field of generative content understanding. The dataset is publicly available at https://journeydb.github.io.</details>

### Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

**Authors:** Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph Gonzalez, Ion Stoica

**<details><summary>Abstract**</summary>

Evaluating large language model (LLM) based chat assistants is challenging due to their broad capabilities and the inadequacy of existing benchmarks in measuring human preferences.To address this, we explore using strong LLMs as judges to evaluate these models on more open-ended questions.We examine the usage and limitations of LLM-as-a-judge, including position, verbosity, and self-enhancement biases, as well as limited reasoning ability, and propose solutions to mitigate some of them.We then verify the agreement between LLM judges and human preferences by introducing two benchmarks: MT-bench, a multi-turn question set; and Chatbot Arena, a crowdsourced battle platform.Our results reveal that strong LLM judges like GPT-4 can match both controlled and crowdsourced human preferences well, achieving over 80\% agreement, the same level of agreement between humans.Hence, LLM-as-a-judge is a scalable and explainable way to approximate human preferences, which are otherwise very expensive to obtain.Additionally, we show our benchmark and traditional benchmarks complement each other by evaluating several variants of LLaMA and Vicuna.The MT-bench questions, 3K expert votes, and 30K conversations with human preferences are publicly available at https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge.</details>

### Katakomba: Tools and Benchmarks for Data-Driven NetHack

**Authors:** Vladislav Kurenkov, Alexander Nikulin, Denis Tarasov, Sergey Kolesnikov

**<details><summary>Abstract**</summary>

NetHack is known as the frontier of reinforcement learning research where learning-based methods still need to catch up to rule-based solutions. One of the promising directions for a breakthrough is using pre-collected datasets similar to recent developments in robotics, recommender systems, and more under the umbrella of offline reinforcement learning (ORL). Recently, a large-scale NetHack dataset was released; while it was a necessary step forward, it has yet to gain wide adoption in the ORL community. In this work, we argue that there are three major obstacles for adoption: tool-wise, implementation-wise, and benchmark-wise. To address them, we develop an open-source library that provides workflow fundamentals familiar to the ORL community: pre-defined D4RL-style tasks, uncluttered baseline implementations, and reliable evaluation tools with accompanying configs and logs synced to the cloud.</details>

### Kernel Stein Discrepancy thinning: a theoretical perspective of pathologies and a practical fix with regularization

**Authors:** Clement Benard, Brian Staber, Sébastien Da Veiga

**<details><summary>Abstract**</summary>

Stein thinning is a promising algorithm proposed by (Riabiz et al., 2022) for post-processing outputs of Markov chain Monte Carlo (MCMC). The main principle is to greedily minimize the kernelized Stein discrepancy (KSD), which only requires the gradient of the log-target distribution, and is thus well-suited for Bayesian inference. The main advantages of Stein thinning are the automatic remove of the burn-in period, the correction of the bias introduced by recent MCMC algorithms, and the asymptotic properties of convergence towards the target distribution. Nevertheless, Stein thinning suffers from several empirical pathologies, which may result in poor approximations, as observed in the literature. In this article, we conduct a theoretical analysis of these pathologies, to clearly identify the mechanisms at stake, and suggest improved strategies. Then, we introduce the regularized Stein thinning algorithm to alleviate the identified pathologies. Finally, theoretical guarantees and extensive experiments show the high efficiency of the proposed algorithm. An implementation of regularized Stein thinning as the kernax library in python and JAX is available at https://gitlab.com/drti/kernax.</details>

### Kernelized Reinforcement Learning with Order Optimal Regret Bounds

**Authors:** Sattar Vakili, Julia Olkhovskaya

**<details><summary>Abstract**</summary>

Modern reinforcement learning (RL) has shown empirical success in various real world settings with complex models and large state-action spaces. The existing analytical results, however, typically focus on settings with a small number of state-actions or simple models such as linearly modeled state-action value functions. To derive RL policies that efficiently handle large state-action spaces with more general value functions, some recent works have considered nonlinear function approximation using kernel ridge regression. We propose $\pi$-KRVI, an optimistic modification of least-squares value iteration, when the action-value function is represented by an RKHS. We prove the first order-optimal regret guarantees under a general setting. Our results show a significant polynomial in the number of episodes improvement over the state of the art. In particular, with highly non-smooth kernels (such as Neural Tangent kernel or some Matérn kernels) the existing results lead to trivial (superlinear in the number of episodes) regret bounds. We show a sublinear regret bound that is order optimal in the cases where a lower bound on regret is known (which includes the kernels mentioned above).</details>

### [Spotlight] Kiki or Bouba? Sound Symbolism in Vision-and-Language Models

**Authors:** Morris Alper, Hadar Averbuch-Elor

**<details><summary>Abstract**</summary>

Although the mapping between sound and meaning in human language is assumed to be largely arbitrary, research in cognitive science has shown that there are non-trivial correlations between particular sounds and meanings across languages and demographic groups, a phenomenon known as sound symbolism. Among the many dimensions of meaning, sound symbolism is particularly salient and well-demonstrated with regards to cross-modal associations between language and the visual domain. In this work, we address the question of whether sound symbolism is reflected in vision-and-language models such as CLIP and Stable Diffusion. Using zero-shot knowledge probing to investigate the inherent knowledge of these models, we find strong evidence that they do show this pattern, paralleling the well-known kiki-bouba effect in psycholinguistics. Our work provides a novel method for demonstrating sound symbolism and understanding its nature using computational tools. Our code will be made publicly available.</details>

### Kissing to Find a Match: Efficient Low-Rank Permutation Representation

**Authors:** Hannah Dröge, Zorah Lähner, Yuval Bahat, Onofre Martorell Nadal, Felix Heide, Michael Moeller

**<details><summary>Abstract**</summary>

Permutation matrices play a key role in matching and assignment problems across the fields, especially in computer vision and robotics. However, memory for explicitly representing permutation matrices grows quadratically with the size of the problem, prohibiting large problem instances. In this work, we propose to tackle the curse of dimensionality of large  permutation matrices by approximating them using low-rank matrix factorization, followed by a nonlinearity. To this end, we rely on the Kissing number theory to infer the minimal rank required for representing a permutation matrix of a given size, which is significantly smaller than the problem size. This leads to a drastic reduction in computation and memory costs, e.g., up to $3$ orders of magnitude less memory for a problem of size $n=20000$, represented using $8.4\times10^5$ elements in two small matrices instead of using a single huge matrix with $4\times 10^8$ elements. The proposed representation allows for accurate representations of large permutation matrices, which in turn enables handling large problems that would have been infeasible otherwise. We demonstrate the applicability and merits of the proposed approach through a series of experiments on a range of problems that involve predicting permutation matrices, from linear and quadratic assignment to shape matching problems.</details>

### Knowledge Diffusion for Distillation

**Authors:** Tao Huang, Yuan Zhang, Mingkai Zheng, Shan You, Fei Wang, Chen Qian, Chang Xu

**<details><summary>Abstract**</summary>

The representation gap between teacher and student is an emerging topic in knowledge distillation (KD). To reduce the gap and improve the performance, current methods often resort to complicated training schemes, loss functions, and feature alignments, which are task-specific and feature-specific. In this paper, we state that the essence of these methods is to discard the noisy information and distill the valuable information in the feature, and propose a novel KD method dubbed DiffKD, to explicitly denoise and match features using diffusion models. Our approach is based on the observation that student features typically contain more noises than teacher features due to the smaller capacity of student model. To address this, we propose to denoise student features using a diffusion model trained by teacher features. This allows us to perform better distillation between the refined clean feature and teacher feature. Additionally, we introduce a light-weight diffusion model with a linear autoencoder to reduce the computation cost and an adaptive noise matching module to improve the denoising performance. Extensive experiments demonstrate that DiffKD is effective across various types of features and achieves state-of-the-art performance consistently on image classification, object detection, and semantic segmentation tasks. Code is available at https://github.com/hunto/DiffKD.</details>

### Koopman Kernel Regression

**Authors:** Petar Bevanda, Max Beier, Armin Lederer, Stefan Sosnowski, Eyke Hüllermeier, Sandra Hirche

**<details><summary>Abstract**</summary>

Many machine learning approaches for decision making, such as reinforcement learning, rely on simulators or predictive models to forecast the time-evolution of quantities of interest, e.g., the state of an agent or the reward of a policy. Forecasts of such complex phenomena are commonly described by highly nonlinear dynamical systems, making their use in optimization-based decision-making challenging.Koopman operator theory offers a beneficial paradigm for addressing this problem by characterizing forecasts via linear time-invariant (LTI) ODEs -- turning multi-step forecasting into sparse matrix multiplications.Though there exists a variety of learning approaches, they usually lack crucial learning-theoretic guarantees, making the behavior of the obtained models with increasing data and dimensionality unclear.We address the aforementioned by deriving a novel reproducing kernel Hilbert space (RKHS) over trajectories that solely spans transformations into LTI dynamical systems. The resulting Koopman Kernel Regression (KKR) framework enables the use of statistical learning tools from function approximation for novel convergence results and generalization error bounds under weaker assumptions than existing work. Our experiments demonstrate superior forecasting performance compared to Koopman operator and sequential data predictors in RKHS.</details>

### KuaiSim: A Comprehensive Simulator for Recommender Systems

**Authors:** Kesen Zhao, Shuchang Liu, Qingpeng Cai, Xiangyu Zhao, Ziru Liu, Dong Zheng, Peng Jiang, Kun Gai

**<details><summary>Abstract**</summary>

Reinforcement Learning (RL)-based recommender systems (RSs) have garnered considerable attention due to their ability to learn optimal recommendation policies and maximize long-term user rewards. However, deploying RL models directly in online environments and generating authentic data through A/B tests can pose challenges and require substantial resources. Simulators offer an alternative approach by providing training and evaluation environments for RS models, reducing reliance on real-world data. Existing simulators have shown promising results but also have limitations such as simplified user feedback, lacking consistency with real-world data, the challenge of simulator evaluation, and difficulties in migration and expansion across RSs.To address these challenges, we propose KuaiSim, a comprehensive user environment that provides user feedback with multi-behavior and cross-session responses.The resulting simulator can support three levels of recommendation problems: the request level list-wise recommendation task, the whole-session level sequential recommendation task, and the cross-session level retention optimization task. For each task, KuaiSim also provides evaluation protocols and baseline recommendation algorithms that further serve as benchmarks for future research. We also restructure existing competitive simulators on the Kuairand Dataset and compare them against KuaiSim to future assess their performance and behavioral differences. Furthermore, to showcase KuaiSim's flexibility in accommodating different datasets, we demonstrate its versatility and robustness when deploying it on the ML-1m dataset. The implementation code is available online to ease reproducibility ootnote{https://github.com/Applied-Machine-Learning-Lab/KuaiSim}.</details>

### L-C2ST: Local Diagnostics for Posterior Approximations in Simulation-Based Inference

**Authors:** Julia Linhart, Alexandre Gramfort, Pedro Rodrigues

**<details><summary>Abstract**</summary>

Many recent works in simulation-based inference  (SBI) rely on deep generative models to approximate complex, high-dimensional posterior distributions. However, evaluating whether or not these approximations can be trusted remains a challenge. Most approaches evaluate the posterior estimator only in expectation  over the observation space. This limits their interpretability and is not sufficient to identify for which observations the approximation can be trusted or should be improved. Building upon the well-known classifier two-sample test (C2ST), we introduce $\ell$-C2ST, a new method that allows for a local evaluation of the posterior estimator at any given observation. It offers theoretically grounded and easy to interpret -- e.g. graphical -- diagnostics, and unlike C2ST, does not require access to samples from the true posterior. In the case of normalizing flow-based posterior estimators, $\ell$-C2ST can be specialized to offer better statistical power, while being computationally more efficient. On standard SBI benchmarks, $\ell$-C2ST  provides comparable results to C2ST and outperforms alternative local approaches such as coverage tests based on highest predictive density (HPD). We further highlight the importance of local evaluation and the benefit of interpretability of $\ell$-C2ST on a challenging application from computational neuroscience.</details>

### LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning

**Authors:** Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone

**<details><summary>Abstract**</summary>

Lifelong learning offers a promising paradigm of building a generalist agent that learns and adapts over its lifespan. Unlike traditional lifelong learning problems in image and text domains, which primarily involve the transfer of declarative knowledge of entities and concepts, lifelong learning in decision-making (LLDM) also necessitates the transfer of procedural knowledge, such as actions and behaviors. To advance research in LLDM, we introduce LIBERO, a novel benchmark of lifelong learning for robot manipulation. Specifically, LIBERO highlights five key research topics in LLDM: 1) how to efficiently transfer declarative knowledge, procedural knowledge, or the mixture of both; 2) how to design effective policy architectures and 3) effective algorithms for LLDM; 4) the robustness of a lifelong learner with respect to task ordering; and 5) the effect of model pretraining for LLDM. We develop an extendible procedural generation pipeline that can in principle generate infinitely many tasks. For benchmarking purpose, we create four task suites (130 tasks in total) that we use to investigate the above-mentioned research topics. To support sample-efficient learning, we provide high-quality human-teleoperated demonstration data for all tasks. Our extensive experiments present several insightful or even unexpected discoveries: sequential finetuning outperforms existing lifelong learning methods in forward transfer, no single visual encoder architecture excels at all types of knowledge transfer, and naive supervised pretraining can hinder agents' performance in the subsequent LLDM.</details>

### Language Model Alignment with Elastic Reset

**Authors:** Michael Noukhovitch, Samuel Lavoie, Florian Strub, Aaron Courville

**<details><summary>Abstract**</summary>

Finetuning language models with reinforcement learning (RL), e.g. from human feedback (HF), is a prominent method for alignment. But optimizing against a reward model can improve on reward while degrading performance in other areas, a phenomenon known as reward hacking, alignment tax, or language drift. First, we argue that commonly-used test metrics are insufficient and instead measure how different algorithms tradeoff between reward and drift. The standard method modified the reward with a Kullback-Lieber (KL) penalty between the online and initial model. We propose Elastic Reset, a new algorithm that achieves higher reward with less drift without explicitly modifying the training objective. We periodically reset the online model to an exponentially moving average (EMA) of itself, then reset the EMA model to the initial model. Through the use of an EMA, our model recovers quickly after resets and achieves higher reward with less drift in the same number of steps. We demonstrate that fine-tuning language models with Elastic Reset leads to state-of-the-art performance on a small scale pivot-translation benchmark, outperforms all baselines in a medium-scale RLHF-like IMDB mock sentiment task and leads to a more performant and more aligned technical QA chatbot with LLaMA-7B. Code available https://github.com/mnoukhov/elastic-reset</details>

### Language Quantized AutoEncoders: Towards Unsupervised Text-Image Alignment

**Authors:** Hao Liu, Wilson Yan, Pieter Abbeel

**<details><summary>Abstract**</summary>

Recent progress in scaling up large language models has shown impressive capabilities in performing few-shot learning across a wide range of natural language tasks. However, a key limitation is that these language models fundamentally lack grounding to visual perception - a crucial attribute needed to extend to real world tasks such as in visual-question answering and robotics. While prior works have largely connected image to text through pretraining or fine-tuning, learning such alignments are generally costly due to a combination of curating massive datasets and large computational burdens. In order to resolve these limitations, we propose a simple yet effective approach called Language-Quantized AutoEncoder (LQAE), a modification of VQ-VAE that learns to align text-image data in an unsupervised manner by leveraging pretrained language model denoisers (e.g., BERT). Our main idea is to encode images as sequences of text tokens by directly quantizing image embeddings using a pretrained language codebook. We then feed a masked version of the quantized embeddings into a BERT to reconstruct the original input. By doing so, LQAE learns to represent similar images with similar clusters of text tokens, thereby aligning these two modalities without the use of aligned text-image pairs. We show LQAE learns text-aligned image tokens that enable few-shot multi-modal learning with large language models, outperforming baseline methods in tasks such as image classification and VQA while requiring as few as 1-10 image-text pairs.</details>

### Language Semantic Graph Guided Data-Efficient Learning

**Authors:** Wenxuan Ma, Shuang Li, lincan Cai, Jingxuan Kang

**<details><summary>Abstract**</summary>

Developing generalizable models that can effectively learn from limited data and with minimal reliance on human supervision is a significant objective within the machine learning community, particularly in the era of deep neural networks. Therefore, to achieve data-efficient learning, researchers typically explore approaches that can leverage more related or unlabeled data without necessitating additional manual labeling efforts, such as Semi-Supervised Learning (SSL), Transfer Learning (TL), and Data Augmentation (DA).SSL leverages unlabeled data in the training process, while TL enables the transfer of expertise from related data distributions. DA broadens the dataset by synthesizing new data from existing examples. However, the significance of additional knowledge contained within labels has been largely overlooked in research. In this paper, we propose a novel perspective on data efficiency that involves exploiting the semantic information contained in the labels of the available data. Specifically, we introduce a Language Semantic Graph (LSG) which is constructed from labels manifest as natural language descriptions. Upon this graph, an auxiliary graph neural network is trained to extract high-level semantic relations and then used to guide the training of the primary model, enabling more adequate utilization of label knowledge. Across image, video, and audio modalities, we utilize the LSG method in both TL and SSL scenarios and illustrate its versatility in significantly enhancing performance compared to other data-efficient learning approaches. Additionally, our in-depth analysis shows that the LSG method also expedites the training process.</details>

### Large sample spectral analysis of graph-based multi-manifold clustering

**Authors:** Nicolas Garcia Trillos, Pengfei He, Chenghui Li

**<details><summary>Abstract**</summary>

In this work we study statistical properties of graph-based algorithms for multi-manifold clustering (MMC). In MMC the goal is to retrieve the multi-manifold structure underlying a given Euclidean data set when this one is assumed to be obtained by sampling a distribution on a union of manifolds $\M = \M_1 \cup\dots  \cup \M_N$ that may intersect with each other and that may have different dimensions. We investigate sufficient conditions that similarity graphs on data sets must satisfy in order for their corresponding graph Laplacians to capture the right geometric information to solve the MMC problem. Precisely, we provide high probability error bounds for the spectral approximation of a tensorized Laplacian on $\M$ with a suitable graph Laplacian built from the observations; the recovered tensorized Laplacian contains all geometric information of all the individual underlying manifolds. We provide an example of a family of similarity graphs, which we call annular proximity graphs with angle constraints, satisfying these sufficient conditions. We contrast our family of graphs with other constructions in the literature based on the alignment of tangent planes. Extensive numerical experiments expand the insights that our theory provides on the MMC problem.</details>

### LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting

**Authors:** Xu Liu, Yutong Xia, Yuxuan Liang, Junfeng Hu, Yiwei Wang, LEI BAI, Chao Huang, Zhenguang Liu, Bryan Hooi, Roger Zimmermann

**<details><summary>Abstract**</summary>

Road traffic forecasting plays a critical role in smart city initiatives and has experienced significant advancements thanks to the power of deep learning in capturing non-linear patterns of traffic data. However, the promising results achieved on current public datasets may not be applicable to practical scenarios due to limitations within these datasets. First, the limited sizes of them may not reflect the real-world scale of traffic networks. Second, the temporal coverage of these datasets is typically short, posing hurdles in studying long-term patterns and acquiring sufficient samples for training deep models. Third, these datasets often lack adequate metadata for sensors, which compromises the reliability and interpretability of the data. To mitigate these limitations, we introduce the LargeST benchmark dataset. It encompasses a total number of 8,600 sensors in California with a 5-year time coverage and includes comprehensive metadata. Using LargeST, we perform in-depth data analysis to extract data insights, benchmark well-known baselines in terms of their performance and efficiency, and identify challenges as well as opportunities for future research. We release the datasets and baseline implementations at: https://github.com/liuxu77/LargeST.</details>

### LayoutPrompter: Awaken the Design Ability of Large Language Models

**Authors:** Jiawei Lin, Jiaqi Guo, Shizhao Sun, Zijiang Yang, Jian-Guang Lou, Dongmei Zhang

**<details><summary>Abstract**</summary>

Conditional graphic layout generation, which automatically maps user constraints to high-quality layouts, has attracted widespread attention today. Although recent works have achieved promising performance, the lack of versatility and data efficiency hinders their practical applications. In this work, we propose LayoutPrompter, which leverages large language models (LLMs) to address the above problems through in-context learning. LayoutPrompter is made up of three key components, namely input-output serialization, dynamic exemplar selection and layout ranking. Specifically, the input-output serialization component meticulously designs the input and output formats for each layout generation task. Dynamic exemplar selection is responsible for selecting the most helpful prompting exemplars for a given input. And a layout ranker is used to pick the highest quality layout from multiple outputs of LLMs. We conduct experiments on all existing layout generation tasks using four public datasets. Despite the simplicity of our approach, experimental results show that LayoutPrompter can compete with or even outperform state-of-the-art approaches on these tasks without any model training or fine-tuning. This demonstrates the effectiveness of this versatile and training-free approach. In addition, the ablation studies show that LayoutPrompter is significantly superior to the training-based baseline in a low-data regime, further indicating the data efficiency of LayoutPrompter. Our project is available at https://github.com/microsoft/LayoutGeneration/tree/main/LayoutPrompter.</details>

### [Oral] LeanDojo: Theorem Proving with Retrieval-Augmented Language Models

**Authors:** Kaiyu Yang, Aidan Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan J Prenger, Animashree Anandkumar

**Oral Presentation:** Tu, Dec 12, 08:00 -- Oral 1B

**<details><summary>Abstract**</summary>

Large language models (LLMs) have shown promise in proving formal theorems using proof assistants such as Lean. However, existing methods are difficult to reproduce or build on, due to private code, data, and large compute requirements. This has created substantial barriers to research on machine learning methods for theorem proving. This paper removes these barriers by introducing LeanDojo: an open-source Lean playground consisting of toolkits, data, models, and benchmarks. LeanDojo extracts data from Lean and enables interaction with the proof environment programmatically. It contains fine-grained annotations of premises in proofs, providing valuable data for premise selection—a key bottleneck in theorem proving. Using this data, we develop ReProver (Retrieval-Augmented Prover): an LLM-based prover augmented with retrieval for selecting premises from a vast math library. It is inexpensive and needs only one GPU week of training. Our retriever leverages LeanDojo's program analysis capability to identify accessible premises and hard negative examples, which makes retrieval much more effective. Furthermore, we construct a new benchmark consisting of 98,734 theorems and proofs extracted from Lean's math library. It features challenging data split requiring the prover to generalize to theorems relying on novel premises that are never used in training. We use this benchmark for training and evaluation, and experimental results demonstrate the effectiveness of ReProver over non-retrieval baselines and GPT-4. We thus provide the first set of open-source LLM-based theorem provers without any proprietary datasets and release it under a permissive MIT license to facilitate further research.</details>

### Learning Adaptive Tensorial Density Fields for Clean Cryo-ET Reconstruction

**Authors:** Yuanhao Wang, Ramzi Idoughi, Wolfgang Heidrich

**<details><summary>Abstract**</summary>

We present a novel learning-based framework for reconstructing 3D structures from tilt-series cryo-Electron Tomography (cryo-ET) data. Cryo-ET is a powerful imaging technique that can achieve near-atomic resolutions. Still, it suffers from challenges such as missing-wedge acquisition, large data size, and high noise levels. Our framework addresses these challenges by using an adaptive tensorial-based representation for the 3D density field of the scanned sample. First, we optimize a quadtree structure to partition the volume of interest. Then, we learn a vector-matrix factorization of the tensor representing the density field in each node. Moreover, we use a loss function that combines a differentiable tomographic formation model with three regularization terms: total variation, boundary consistency constraint, and an isotropic Fourier prior. Our framework allows us to query the density at any location using the learned representation and obtain a high-quality 3D tomogram. We demonstrate the superiority of our framework over existing methods using synthetic and real data. Thus, our framework boosts the quality of the reconstruction while reducing the computation time and the memory footprint. The code is available at https://github.com/yuanhaowang1213/adaptivetensordf.</details>

### Learning Curves for Deep Structured Gaussian Feature Models

**Authors:** Jacob Zavatone-Veth, Cengiz Pehlevan

**<details><summary>Abstract**</summary>

In recent years, significant attention in deep learning theory has been devoted to analyzing when models that interpolate their training data can still generalize well to unseen examples. Many insights have been gained from studying models with multiple layers of Gaussian random features, for which one can compute precise generalization asymptotics. However, few works have considered the effect of weight anisotropy; most assume that the random features are generated using independent and identically distributed Gaussian weights, and allow only for structure in the input data. Here, we use the replica trick from statistical physics to derive learning curves for models with many layers of structured Gaussian features. We show that allowing correlations between the rows of the first layer of features can aid generalization, while structure in later layers is generally detrimental. Our results shed light on how weight structure affects generalization in a simple class of solvable models.</details>

### Learning DAGs from Data with Few Root Causes

**Authors:** Panagiotis Misiakos, Chris Wendler, Markus Püschel

**<details><summary>Abstract**</summary>

We present a novel perspective and algorithm for learning directed acyclic graphs (DAGs) from data generated by a linear structural equation model (SEM). First, we show that a linear SEM can be viewed as a linear transform that, in prior work, computes the data from a dense input vector of random valued root causes (as we will call them) associated with the nodes. Instead, we consider the case of (approximately) few root causes and also introduce noise in the measurement of the data. Intuitively, this means that the DAG data is produced by few data generating events whose effect percolates through the DAG. We prove identifiability in this new setting and show that the true DAG is the global minimizer of the $L^0$-norm of the vector of root causes. For data satisfying the few root causes assumption, we show superior performance compared to prior DAG learning methods.</details>

### Learning Domain-Aware Detection Head with Prompt Tuning

**Authors:** Haochen Li, Rui Zhang, Hantao Yao, Xinkai Song, Yifan Hao, Yongwei Zhao, Ling Li, Yunji Chen

**<details><summary>Abstract**</summary>

Domain adaptive object detection (DAOD) aims to generalize detectors trained on an annotated source domain to an unlabelled target domain. However, existing methods focus on reducing the domain bias of the detection backbone by inferring a discriminative visual encoder, while ignoring the domain bias in the detection head. Inspired by the high generalization of vision-language models (VLMs), applying a VLM as the robust detection backbone following a domain-aware detection head is a reasonable way to learn the discriminative detector for each domain, rather than reducing the domain bias in traditional methods. To achieve the above issue, we thus propose a novel DAOD framework named Domain-Aware detection head with Prompt tuning (DA-Pro), which applies the learnable domain-adaptive prompt to generate the dynamic detection head for each domain. Formally, the domain-adaptive prompt consists of the domain-invariant tokens, domain-specific tokens, and the domain-related textual description along with the class label. Furthermore, two constraints between the source and target domains are applied to ensure that the domain-adaptive prompt can capture the domains-shared and domain-specific knowledge. A prompt ensemble strategy is also proposed to reduce the effect of prompt disturbance. Comprehensive experiments over multiple cross-domain adaptation tasks demonstrate that using the domain-adaptive prompt can produce an effectively domain-related detection head for boosting domain-adaptive object detection. Our code is available at https://github.com/Therock90421/DA-Pro.</details>

### Learning Efficient Surrogate Dynamic Models with Graph Spline Networks

**Authors:** Chuanbo Hua, Federico Berto, Michael Poli, Stefano Massaroli, Jinkyoo Park

**<details><summary>Abstract**</summary>

While complex simulations of physical systems have been widely used in engineering and scientific computing, lowering their often prohibitive computational requirements has only recently been tackled by deep learning approaches. In this paper, we present GraphSplineNets, a novel deep-learning method to speed up the forecasting of physical systems by reducing the grid size and number of iteration steps of deep surrogate models. Our method uses two differentiable orthogonal spline collocation methods to efficiently predict response at any location in time and space. Additionally, we introduce an adaptive collocation strategy in space to prioritize sampling from the most important regions. GraphSplineNets improve the accuracy-speedup tradeoff in forecasting various dynamical systems with increasing complexity, including the heat equation, damped wave propagation, Navier-Stokes equations, and real-world ocean currents in both regular and irregular domains.</details>

### Learning From Biased Soft Labels

**Authors:** Hua Yuan, Yu Shi, Ning Xu, Xu Yang, Xin Geng, Yong Rui

**<details><summary>Abstract**</summary>

Since the advent of knowledge distillation, many researchers have been intrigued by the $\textit{dark knowledge}$ hidden in the soft labels generated by the teacher model. This prompts us to scrutinize the circumstances under which these soft labels are effective. Predominant existing theories implicitly require that the soft labels are close to the ground-truth labels. In this paper, however, we investigate whether biased soft labels are still effective. Here, bias refers to the discrepancy between the soft labels and the ground-truth labels. We present two indicators to measure the effectiveness of the soft labels. Based on the two indicators, we propose moderate conditions to ensure that, the biased soft label learning problem is both $\textit{classifier-consistent}$ and $\textit{Empirical Risk Minimization}$ (ERM) $\textit{learnable}$, which can be applicable even for large-biased soft labels. We further design a heuristic method to train Skillful but Bad Teachers (SBTs), and these teachers with accuracy less than 30\% can teach students to achieve accuracy over 90\% on CIFAR-10, which is comparable to models trained on the original data. The proposed indicators adequately measure the effectiveness of the soft labels generated in this process. Moreover, our theoretical framework can be adapted to elucidate the effectiveness of soft labels in three weakly-supervised learning paradigms, namely incomplete supervision, partial label learning and learning with additive noise. Experimental results demonstrate that our indicators can measure the effectiveness of biased soft labels generated by teachers or in these weakly-supervised learning paradigms.</details>

### [Spotlight] Learning Generalizable Agents via Saliency-guided Features Decorrelation

**Authors:** Sili Huang, Yanchao Sun, Jifeng Hu, Siyuan Guo, Hechang Chen, Yi Chang, Lichao Sun, Bo Yang

**<details><summary>Abstract**</summary>

In visual-based Reinforcement Learning (RL), agents often struggle to generalize well to environmental variations in the state space that were not observed during training. The variations can arise in both task-irrelevant features, such as background noise, and task-relevant features, such as robot configurations, that are related to the optimal decisions. To achieve generalization in both situations, agents are required to accurately understand the impact of changed features on the decisions, i.e., establishing the true associations between changed features and decisions in the policy model. However, due to the inherent correlations among features in the state space, the associations between features and decisions become entangled, making it difficult for the policy to distinguish them. To this end, we propose Saliency-Guided Features Decorrelation (SGFD) to eliminate these correlations through sample reweighting. Concretely, SGFD consists of two core techniques: Random Fourier Functions (RFF) and the saliency map. RFF is utilized to estimate the complex non-linear correlations in high-dimensional images, while the saliency map is designed to identify the changed features. Under the guidance of the saliency map, SGFD employs sample reweighting to minimize the estimated correlations related to changed features, thereby achieving decorrelation in visual RL tasks. Our experimental results demonstrate that SGFD can generalize well on a wide range of test environments and significantly outperforms state-of-the-art methods in handling both task-irrelevant variations and task-relevant variations.</details>

### Learning Mixtures of Gaussians Using the DDPM Objective

**Authors:** Kulin Shah, Sitan Chen, Adam Klivans

**<details><summary>Abstract**</summary>

Recent works have shown that diffusion models can learn essentially any distribution provided one can perform score estimation.Yet it remains poorly understood under what settings score estimation is possible, let alone when practical gradient-based algorithms for this task can provably succeed. In this work, we give the first provably efficient results for one of the most fundamental distribution families, Gaussian mixture models.We prove that GD on the denoising diffusion probabilistic model (DDPM) objective can efficiently recover the ground truth parameters of the mixture model in the following two settings:1. We show GD with random initialization learns mixtures of two spherical Gaussians in $d$ dimensions with $1/\text{poly}(d)$-separated centers.2. We show GD with a warm start learns mixtures of $K$ spherical Gaussians with $\Omega(\sqrt{\log(\min(K,d))})$-separated centers.A key ingredient in our proofs is a new connection between score-based methods and two other approaches to distribution learning, EM and spectral methods.</details>

### Learning Rate Free Bayesian Inference in Constrained Domains

**Authors:** Louis Sharrock, Lester Mackey, Christopher Nemeth

**<details><summary>Abstract**</summary>

We introduce a suite of new particle-based algorithms for sampling on constrained domains which are entirely learning rate free. Our approach leverages coin betting ideas from convex optimisation, and the viewpoint of constrained sampling as a mirrored optimisation problem on the space of probability measures. Based on this viewpoint, we also introduce a unifying framework for several existing constrained sampling algorithms, including mirrored Langevin dynamics and mirrored Stein variational gradient descent. We demonstrate the performance of our algorithms on a range of numerical examples, including sampling from targets on the simplex, sampling with fairness constraints, and constrained sampling problems in post-selection inference. Our results indicate that our algorithms achieve competitive performance with existing constrained sampling methods, without the need to tune any hyperparameters.</details>

### Learning Shared Safety Constraints from Multi-task Demonstrations

**Authors:** Konwoo Kim, Gokul Swamy, ZUXIN LIU, DING ZHAO, Sanjiban Choudhury, Steven Wu

**<details><summary>Abstract**</summary>

Regardless of the particular task we want to perform in an environment, there are often shared safety constraints we want our agents to respect. For example, regardless of whether it is making a sandwich or clearing the table, a kitchen robot should not break a plate. Manually specifying such a constraint can be both time-consuming and error-prone. We show how to learn constraints from expert demonstrations of safe task completion by extending inverse reinforcement learning (IRL) techniques to the space of constraints. Intuitively, we learn constraints that forbid highly rewarding behavior that the expert could have taken but chose not to. Unfortunately, the constraint learning problem is rather ill-posed and typically leads to overly conservative constraints that forbid all behavior that the expert did not take. We counter this by leveraging diverse demonstrations that naturally occur in multi-task setting to learn a tighter set of constraints. We validate our method with simulation experiments on high-dimensional continuous control tasks.</details>

### Learning World Models with Identifiable Factorization

**Authors:** Yuren Liu, Biwei Huang, Zhengmao Zhu, Honglong Tian, Mingming Gong, Yang Yu, Kun Zhang

**<details><summary>Abstract**</summary>

Extracting a stable and compact representation of the environment is crucial for efficient reinforcement learning in high-dimensional, noisy, and non-stationary environments.  Different categories of information coexist in such environments -- how to effectively extract and disentangle the information remains a challenging problem. In this paper, we propose IFactor, a general framework to model four distinct categories of latent state variables that capture various aspects of information within the RL system, based on their interactions with actions and rewards. Our analysis establishes block-wise identifiability of these latent variables, which not only provides a stable and compact representation but also discloses that all reward-relevant factors are significant for policy learning. We further present a practical approach to learning the world model with identifiable blocks, ensuring the removal of redundancies but retaining minimal and sufficient information for policy optimization. Experiments in synthetic worlds demonstrate that our method accurately identifies the ground-truth latent variables, substantiating our theoretical findings. Moreover, experiments in variants of the DeepMind Control Suite and RoboDesk showcase the superior performance of our approach over baselines.</details>

### Learning better with Dale’s Law: A Spectral Perspective

**Authors:** Pingsheng Li, Jonathan Cornford, Arna Ghosh, Blake Richards

**<details><summary>Abstract**</summary>

Most recurrent neural networks (RNNs) do not include a fundamental constraint of real neural circuits: Dale's Law, which implies that neurons must be excitatory (E) or inhibitory (I). Dale's Law is generally absent from RNNs because simply partitioning a standard network's units into E and I populations impairs learning. However, here we extend a recent feedforward bio-inspired EI network architecture, named Dale's ANNs, to recurrent networks, and demonstrate that good performance is possible while respecting Dale's Law. This begs the question: What makes some forms of EI network learn poorly and others learn well? And, why does the simple approach of incorporating Dale's Law impair learning?  Historically the answer was thought to be the sign constraints on EI network parameters, and this was a motivation behind Dale's ANNs. However, here we show the spectral properties of the recurrent weight matrix at initialisation are more impactful on network performance than sign constraints. We find that simple EI partitioning results in a singular value distribution that is multimodal and dispersed, whereas standard RNNs have an unimodal, more clustered singular value distribution, as do recurrent Dale's ANNs. We also show that the spectral properties and performance of partitioned EI networks are worse for small networks with fewer I units, and we present normalised SVD entropy as a measure of spectrum pathology that correlates with performance. Overall, this work sheds light on a long-standing mystery in neuroscience-inspired AI and computational neuroscience, paving the way for greater alignment between neural networks and biology.</details>

### [Spotlight] Learning from Active Human Involvement through Proxy Value Propagation

**Authors:** Zhenghao Peng, Wenjie Mo, Chenda Duan, Quanyi Li, Bolei Zhou

**<details><summary>Abstract**</summary>

Learning from active human involvement enables the human subject to actively intervene and demonstrate to the AI agent during training. The interaction and corrective feedback from human brings safety and AI alignment to the learning process. In this work, we propose a new reward-free active human involvement method called Proxy Value Propagation for policy optimization. Our key insight is that a proxy value function can be designed to express human intents, wherein state- action pairs in the human demonstration are labeled with high values, while those agents’ actions that are intervened receive low values. Through the TD-learning framework, labeled values of demonstrated state-action pairs are further propagated to other unlabeled data generated from agents’ exploration. The proxy value function thus induces a policy that faithfully emulates human behaviors. Human- in-the-loop experiments show the generality and efficiency of our method. With minimal modification to existing reinforcement learning algorithms, our method can learn to solve continuous and discrete control tasks with various human control devices, including the challenging task of driving in Grand Theft Auto V. Demo video and code are available at: https://metadriverse.github.io/pvp.</details>

### Learning from Visual Observation via Offline Pretrained State-to-Go Transformer

**Authors:** Bohan Zhou, Ke Li, Jiechuan Jiang, Zongqing Lu

**<details><summary>Abstract**</summary>

Learning from visual observation (LfVO), aiming at recovering policies from only visual observation data, is promising yet a challenging problem. Existing LfVO approaches either only adopt inefficient online learning schemes or require additional task-specific information like goal states, making them not suited for open-ended tasks. To address these issues, we propose a two-stage framework for learning from visual observation. In the first stage, we introduce and pretrain State-to-Go (STG) Transformer offline to predict and differentiate latent transitions of demonstrations. Subsequently, in the second stage, the STG Transformer provides intrinsic rewards for downstream reinforcement learning tasks where an agent learns merely from intrinsic rewards. Empirical results on Atari and Minecraft show that our proposed method outperforms baselines and in some tasks even achieves performance comparable to the policy learned from environmental rewards. These results shed light on the potential of utilizing video-only data to solve difficult visual reinforcement learning tasks rather than relying on complete offline datasets containing states, actions, and rewards. The project’s website and code can befound at https://sites.google.com/view/stgtransformer.</details>

### Learning to Parameterize Visual Attributes for Open-set Fine-grained Retrieval

**Authors:** Shijie Wang, Jianlong Chang, Haojie Li, Zhihui Wang, Wanli Ouyang, Qi Tian

**<details><summary>Abstract**</summary>

Open-set fine-grained retrieval is an emerging challenging task that allows to retrieve unknown categories beyond the training set. The best solution for handling unknown categories is to represent them using a set of visual attributes learnt from known categories, as widely used in zero-shot learning. Though important, attribute modeling usually requires significant manual annotations and thus is labor-intensive. Therefore, it is worth to investigate how to transform retrieval models trained by image-level supervision from category semantic extraction to attribute modeling. To this end, we propose a novel Visual Attribute Parameterization Network (VAPNet) to learn visual attributes from known categories and parameterize them into the retrieval model, without the involvement of any attribute annotations.In this way, VAPNet could utilize its parameters to parse a set of visual attributes from unknown categories and precisely represent them.Technically, VAPNet explicitly attains some semantics with rich details via making use of local image patches and distills the visual attributes from these discovered semantics. Additionally, it integrates the online refinement of these visual attributes into the training process to iteratively enhance their quality. Simultaneously, VAPNet treats these attributes as supervisory signals to tune the retrieval models, thereby achieving attribute parameterization. Extensive experiments on open-set fine-grained retrieval datasets validate the superior performance of our VAPNet over existing solutions.</details>

### Learning to Search Feasible and Infeasible Regions of Routing Problems with Flexible Neural k-Opt

**Authors:** Yining Ma, Zhiguang Cao, Yeow Meng Chee

**<details><summary>Abstract**</summary>

In this paper, we present Neural k-Opt (NeuOpt), a novel learning-to-search (L2S) solver for routing problems. It learns to perform flexible k-opt exchanges based on a tailored action factorization method and a customized recurrent dual-stream decoder. As a pioneering work to circumvent the pure feasibility masking scheme and enable the autonomous exploration of both feasible and infeasible regions, we then propose the Guided Infeasible Region Exploration (GIRE) scheme, which supplements the NeuOpt policy network with feasibility-related features and leverages reward shaping to steer reinforcement learning more effectively. Additionally, we equip NeuOpt with Dynamic Data Augmentation (D2A) for more diverse searches during inference. Extensive experiments on the Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP) demonstrate that our NeuOpt not only significantly outstrips existing (masking-based) L2S solvers, but also showcases superiority over the learning-to-construct (L2C) and learning-to-predict (L2P) solvers. Notably, we offer fresh perspectives on how neural solvers can handle VRP constraints. Our code is available: https://github.com/yining043/NeuOpt.</details>

### Leave No Stone Unturned: Mine Extra Knowledge for Imbalanced Facial Expression Recognition

**Authors:** Yuhang Zhang, Yaqi Li, lixiong Qin, Xuannan Liu, Weihong Deng

**<details><summary>Abstract**</summary>

Facial expression data is characterized by a significant imbalance, with most collected data showing happy or neutral expressions and fewer instances of fear or disgust. This imbalance poses challenges to facial expression recognition (FER) models, hindering their ability to fully understand various human emotional states. Existing FER methods typically report overall accuracy on highly imbalanced test sets but exhibit low performance in terms of the mean accuracy across all expression classes. In this paper, our aim is to address the imbalanced FER problem. Existing methods primarily focus on learning knowledge of minor classes solely from minor-class samples. However, we propose a novel approach to extract extra knowledge related to the minor classes from both major and minor class samples. Our motivation stems from the belief that FER resembles a distribution learning task, wherein a sample may contain information about multiple classes. For instance, a sample from the major class surprise might also contain useful features of the minor class fear. Inspired by that, we propose a novel method that leverages re-balanced attention maps to regularize the model, enabling it to extract transformation invariant information about the minor classes from all training samples. Additionally, we introduce re-balanced smooth labels to regulate the cross-entropy loss, guiding the model to pay more attention to the minor classes by utilizing the extra information regarding the label distribution of the imbalanced training data. Extensive experiments on different datasets and backbones show that the two proposed modules work together to regularize the model and achieve state-of-the-art performance under the imbalanced FER task. Code is available at https://github.com/zyh-uaiaaaa.</details>

### LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models

**Authors:** Neel Guha, Julian Nyarko, Daniel Ho, Christopher Ré, Adam Chilton, Aditya K, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel Rockmore, Diego Zambrano, Dmitry Talisman, Enam Hoque, Faiz Surani, Frank Fagan, Galit Sarfaty, Gregory Dickinson, Haggai Porat, Jason Hegland, Jessica Wu, Joe Nudell, Joel Niklaus, John Nay, Jonathan Choi, Kevin Tobia, Margaret Hagan, Megan Ma, Michael Livermore, Nikon Rasumov-Rahe, Nils Holzenberger, Noam Kolt, Peter Henderson, Sean Rehaag, Sharad Goel, Shang Gao, Spencer Williams, Sunny Gandhi, Tom Zur, Varun Iyer, Zehua Li

**<details><summary>Abstract**</summary>

The advent of large language models (LLMs) and their adoption by the legal community has given rise to the question: what types of legal reasoning can LLMs perform? To enable greater study of this question, we present LegalBench: a collaboratively constructed legal reasoning benchmark consisting of 162 tasks covering six different types of legal reasoning. LegalBench was built through an interdisciplinary process, in which we collected tasks designed and hand-crafted by legal professionals. Because these subject matter experts took a leading role in construction, tasks either measure legal reasoning capabilities that are practically useful, or measure reasoning skills that lawyers find interesting. To enable cross-disciplinary conversations about LLMs in the law, we additionally show how popular legal frameworks for describing legal reasoning—which distinguish between its many forms—correspond to LegalBench tasks, thus giving lawyers and LLM developers a common vocabulary. This paper describes LegalBench, presents an empirical evaluation of 20 open-source and commercial LLMs, and illustrates the types of research explorations LegalBench enables.</details>

### LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios

**Authors:** Yazhe Niu, YUAN PU, Zhenjie Yang, Xueyan Li, Tong Zhou, Jiyuan Ren, Shuai Hu, Hongsheng Li, Yu Liu

**<details><summary>Abstract**</summary>

Building agents based on tree-search planning capabilities with learned models has achieved remarkable success in classic decision-making problems, such as Go and Atari.However, it has been deemed challenging or even infeasible to extend Monte Carlo Tree Search (MCTS) based algorithms to diverse real-world applications, especially when these environments involve complex action spaces and significant simulation costs, or inherent stochasticity.In this work, we introduce LightZero, the first unified benchmark for deploying MCTS/MuZero in general sequential decision scenarios. Specificially, we summarize the most critical challenges in designing a general MCTS-style decision-making solver, then decompose the tightly-coupled algorithm and system design of tree-search RL methods into distinct sub-modules.By incorporating more appropriate exploration and optimization strategies, we can significantly enhance these sub-modules and construct powerful LightZero agents to tackle tasks across a wide range of domains, such as board games, Atari, MuJoCo, MiniGrid and GoBigger.Detailed benchmark results reveal the significant potential of such methods in building scalable and efficient decision intelligence.The code is available as part of OpenDILab at https://github.com/opendilab/LightZero.</details>

### LinGCN: Structural Linearized Graph Convolutional Network for Homomorphically Encrypted Inference

**Authors:** Hongwu Peng, Ran Ran, Yukui Luo, Jiahui Zhao, Shaoyi Huang, Kiran Thorat, Tong Geng, Chenghong Wang, Xiaolin Xu, Wujie Wen, Caiwen Ding

**<details><summary>Abstract**</summary>

The growth of Graph Convolution Network (GCN) model sizes has revolutionized numerous applications, surpassing human performance in areas such as personal healthcare and financial systems. The  deployment of GCNs in the cloud raises privacy concerns due to potential adversarial attacks on client data. To address security concerns, Privacy-Preserving Machine Learning (PPML) using Homomorphic Encryption (HE) secures sensitive client data. However, it introduces substantial computational overhead in practical applications. To tackle those challenges, we present LinGCN, a framework designed to reduce multiplication depth and optimize the performance of HE based GCN inference. LinGCN is structured around three key elements: (1) A differentiable structural linearization algorithm, complemented by a parameterized discrete indicator function, co-trained with model weights to meet the optimization goal. This strategy promotes fine-grained node-level non-linear location selection, resulting in a model with minimized multiplication depth. (2) A compact node-wise polynomial replacement policy with a second-order trainable activation function, steered towards superior convergence by a two-level distillation approach from an all-ReLU based teacher model. (3) an enhanced HE solution that enables finer-grained operator fusion for node-wise activation functions, further reducing multiplication level consumption in HE-based inference. Our experiments on the NTU-XVIEW skeleton joint dataset reveal that LinGCN excels in latency, accuracy, and scalability for homomorphically encrypted inference, outperforming solutions such as CryptoGCN. Remarkably, LinGCN achieves a 14.2× latency speedup relative to CryptoGCN, while preserving an inference accuracy of ~75\% and notably reducing multiplication depth. Additionally, LinGCN proves scalable for larger models, delivering a substantial 85.78\% accuracy with 6371s latency, a 10.47\% accuracy improvement over CryptoGCN.</details>

### [Spotlight] List and Certificate Complexities in Replicable Learning

**Authors:** Peter Dixon, A. Pavan, Jason Vander Woude, N. V. Vinodchandran

**<details><summary>Abstract**</summary>

We investigate replicable learning algorithms. Informally  a learning algorithm is replicable if the algorithm outputs the same canonical hypothesis over multiple runs with high probability, even when different runs observe a different set of samples from the unknown data distribution. In general, such a strong notion of replicability is not achievable. Thus we consider two feasible notions of replicability called {\em list replicability} and {\em certificate replicability}. Intuitively, these notions capture the degree of (non) replicability. The goal is to design learning algorithms with optimal list and certificate complexities while minimizing the sample complexity.  Our contributions are the following.1. We first study the learning task of estimating the biases of $d$ coins, up to an additive error of $\varepsilon$, by observing samples. For this task, we design a $(d+1)$-list replicable algorithm. To complement this result, we establish that the list complexity is optimal, i.e there are no learning algorithms with a list size smaller than $d+1$ for this task. We also design learning algorithms with certificate complexity $\tilde{O}(\log d)$.   The sample complexity of both these algorithms is $\tilde{O}(\frac{d^2}{\varepsilon^2})$ where $\varepsilon$ is the approximation error parameter (for a constant error probability).  2. In the PAC model, we show that any hypothesis class that is learnable with $d$-nonadaptive statistical queries can be learned via a $(d+1)$-list replicable algorithm and also via a $\tilde{O}(\log d)$-certificate replicable algorithm. The sample complexity of both these algorithms is $\tilde{O}(\frac{d^2}{\nu^2})$ where $\nu$ is the approximation error of the statistical query. We also show that for the concept class \dtep, the list complexity is exactly $d+1$ with respect to the uniform distribution.   To establish our upper bound results we use rounding schemes induced by geometric partitions with certain properties. We use Sperner/KKM Lemma to establish the lower bound results.</details>

### LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning

**Authors:** Atsuyuki Miyai, Qing Yu, Go Irie, Kiyoharu Aizawa

**<details><summary>Abstract**</summary>

We present a novel vision-language prompt learning approach for few-shot out-of-distribution (OOD) detection. Few-shot OOD detection aims to detect OOD images from classes that are unseen during training using only a few labeled in-distribution (ID) images. While prompt learning methods such as CoOp have shown effectiveness and efficiency in few-shot ID classification, they still face limitations in OOD detection due to the potential presence of ID-irrelevant information in text embeddings. To address this issue, we introduce a new approach called $	extbf{Lo}$cal regularized $	extbf{Co}$ntext $	extbf{Op}$timization (LoCoOp), which performs OOD regularization that utilizes the portions of CLIP local features as OOD features during training. CLIP's local features have a lot of ID-irrelevant nuisances (	extit{e.g.}, backgrounds), and by learning to push them away from the ID class text embeddings, we can remove the nuisances in the ID class text embeddings and enhance the separation between ID and OOD. Experiments on the large-scale ImageNet OOD detection benchmarks demonstrate the superiority of our LoCoOp over zero-shot, fully supervised detection methods and prompt learning methods. Notably, even in a one-shot setting -- just one label per class, LoCoOp outperforms existing zero-shot and fully supervised detection methods. The code is available via https://github.com/AtsuMiyai/LoCoOp.</details>

### LoRA: A Logical Reasoning Augmented Dataset for Visual Question Answering

**Authors:** Jingying Gao, Qi Wu, Alan Blair, Maurice Pagnucco

**<details><summary>Abstract**</summary>

The capacity to reason logically is a hallmark of human cognition. Humans excel at integrating multimodal information for locigal reasoning, as exemplified by the Visual Question Answering (VQA) task, which is a challenging multimodal task. VQA tasks and large vision-and-language models aim to tackle reasoning problems, but the accuracy, consistency and fabrication of the generated answers is hard to evaluate in the absence of a VQA dataset that can offer formal, comprehensive and systematic complex logical reasoning questions. To address this gap, we present LoRA, a novel Logical Reasoning Augmented VQA dataset that requires formal and complex description logic reasoning based on a food-and-kitchen knowledge base. Our main objective in creating LoRA is to enhance the complex and formal logical reasoning capabilities of VQA models, which are not adequately measured by existing VQA datasets. We devise strong and flexible programs to automatically generate 200,000 diverse description logic reasoning questions based on the SROIQ Description Logic, along with realistic kitchen scenes and ground truth answers. We fine-tune the latest transformer VQA models and evaluate the zero-shot performance of the state-of-the-art large vision-and-language models on LoRA. The results reveal that LoRA presents a unique challenge in logical reasoning, setting a systematic and comprehensive evaluation standard.</details>

### Locality-Aware Generalizable Implicit Neural Representation

**Authors:** Doyup Lee, Chiheon Kim, Minsu Cho, WOOK SHIN HAN

**<details><summary>Abstract**</summary>

Generalizable implicit neural representation (INR) enables a single continuous function, i.e., a coordinate-based neural network, to represent multiple data instances by modulating its weights or intermediate features using latent codes. However, the expressive power of the state-of-the-art modulation is limited due to its inability to localize and capture fine-grained details of data entities such as specific pixels and rays. To address this issue, we propose a novel framework for generalizable INR that combines a transformer encoder with a locality-aware INR decoder. The transformer encoder predicts a set of latent tokens from a data instance to encode local information into each latent token. The locality-aware INR decoder extracts a modulation vector by selectively aggregating the latent tokens via cross-attention for a coordinate input and then predicts the output by progressively decoding with coarse-to-fine modulation through multiple frequency bandwidths. The selective token aggregation and the multi-band feature modulation enable us to learn locality-aware representation in spatial and spectral aspects, respectively. Our framework significantly outperforms previous generalizable INRs and validates the usefulness of the locality-aware latents for downstream tasks such as image generation.</details>

### LogSpecT: Feasible Graph Learning Model from Stationary Signals with Recovery Guarantees

**Authors:** Shangyuan LIU, Linglingzhi Zhu, Anthony Man-Cho So

**<details><summary>Abstract**</summary>

Graph learning from signals is a core task in graph signal processing (GSP). A significant subclass of graph signals called the stationary graph signals that broadens the concept of stationarity of data defined on regular domains to signals on graphs is gaining increasing popularity in the GSP community. The most commonly used model to learn graphs from these stationary signals is SpecT, which forms the foundation for nearly all the subsequent, more advanced models. Despite its strengths, the practical formulation of the model, known as rSpecT, has been identified to be susceptible to the choice of hyperparameters. More critically, it may suffer from infeasibility as an optimization problem. In this paper, we introduce the first condition that ensures the infeasibility of rSpecT and design a novel model called LogSpecT, along with its practical formulation rLogSpecT to overcome this issue. Contrary to rSpecT, our novel practical model rLogSpecT is always feasible. Furthermore, we provide recovery guarantees of rLogSpecT from modern optimization tools related to epi-convergence, which could be of independent interest and significant for various learning problems. To demonstrate the practical advantages of rLogSpecT, a highly efficient algorithm based on the linearized alternating direction method of multipliers (L-ADMM) that allows closed-form solutions for each subproblem is proposed with convergence guarantees. Extensive numerical results on both synthetic and real networks not only corroborate the stability of our proposed methods, but also highlight their comparable and even superior performance than existing models.</details>

### Logarithmic-Regret Quantum Learning Algorithms for Zero-Sum Games

**Authors:** Minbo Gao, Zhengfeng Ji, Tongyang Li, Qisheng Wang

**<details><summary>Abstract**</summary>

We propose the first online quantum algorithm for zero-sum games with $\widetilde O(1)$ regret under the game setting. Moreover, our quantum algorithm computes an $\varepsilon$-approximate Nash equilibrium of an $m \times n$ matrix zero-sum game in quantum time $\widetilde O(\sqrt{m+n}/\varepsilon^{2.5})$. Our algorithm uses standard quantum inputs and generates classical outputs with succinct descriptions, facilitating end-to-end applications. As an application, we obtain a fast quantum linear programming solver. Technically, our online quantum algorithm "quantizes" classical algorithms based on the optimistic multiplicative weight update method. At the heart of our algorithm is a fast quantum multi-sampling procedure for the Gibbs sampling problem, which may be of independent interest.</details>

### Long Sequence Hopfield Memory

**Authors:** Hamza Chaudhry, Jacob Zavatone-Veth, Dmitry Krotov, Cengiz Pehlevan

**<details><summary>Abstract**</summary>

Sequence memory is an essential attribute of natural and artificial intelligence that enables agents to encode, store, and retrieve complex sequences of stimuli and actions. Computational models of sequence memory have been proposed where recurrent Hopfield-like neural networks are trained with temporally asymmetric Hebbian rules. However, these networks suffer from limited sequence capacity (maximal length of the stored sequence) due to interference between the memories. Inspired by recent work on Dense Associative Memories, we expand the sequence capacity of these models by introducing a nonlinear interaction term, enhancing separation between the patterns. We derive novel scaling laws for sequence capacity with respect to network size, significantly outperforming existing scaling laws for models based on traditional Hopfield networks, and verify these theoretical results with numerical simulation. Moreover, we introduce a generalized pseudoinverse rule to recall sequences of highly correlated patterns. Finally, we extend this model to store sequences with variable timing between states' transitions and describe a biologically-plausible implementation, with connections to motor neuroscience.</details>

### Lovász Principle for Unsupervised Graph Representation Learning

**Authors:** Ziheng Sun, Chris Ding, Jicong Fan

**<details><summary>Abstract**</summary>

This paper focuses on graph-level representation learning that aims to represent graphs as vectors that can be directly utilized in downstream tasks such as graph classification. We propose a novel graph-level representation learning principle called Lovász principle, which is motivated by the Lovász number in graph theory. The Lovász number of a graph is a real number that is an upper bound for graph Shannon capacity and is strongly connected with various global characteristics of the graph. Specifically, we show that the handle vector for computing the Lovász number is potentially a suitable choice for graph representation, as it captures a graph's global properties, though a direct application of the handle vector is difficult and problematic. We propose to use neural networks to address the problems and hence provide the Lovász principle. Moreover, we propose an enhanced Lovász principle that is able to exploit the subgraph Lovász numbers directly and efficiently. The experiments demonstrate that our Lovász principles achieve competitive performance compared to the baselines in unsupervised and semi-supervised graph-level representation learning tasks. The code of our Lovász principles is publicly available on GitHub.</details>

### M5HisDoc: A Large-scale Multi-style Chinese Historical Document Analysis Benchmark

**Authors:** Yongxin Shi, Chongyu Liu, Dezhi Peng, Cheng Jian, Jiarong Huang, Lianwen Jin

**<details><summary>Abstract**</summary>

Recognizing and organizing text in correct reading order plays a crucial role in historical document analysis and preservation. While existing methods have shown promising performance, they often struggle with challenges such as diverse layouts, low image quality, style variations, and distortions. This is primarily due to the lack of consideration for these issues in the current benchmarks, which hinders the development and evaluation of historical document analysis and recognition (HDAR) methods in complex real-world scenarios. To address this gap, this paper introduces a complex multi-style Chinese historical document analysis benchmark, named M5HisDoc. The M5 indicates five properties of style, ie., Multiple layouts, Multiple document types, Multiple calligraphy styles, Multiple backgrounds, and Multiple challenges. The M5HisDoc dataset consists of two subsets, M5HisDoc-R (Regular) and M5HisDoc-H (Hard). The M5HisDoc-R subset comprises 4,000 historical document images. To ensure high-quality annotations, we meticulously perform manual annotation and triple-checking. To replicate real-world conditions for historical document analysis applications, we incorporate image rotation, distortion, and resolution reduction into M5HisDoc-R subset to form a new challenging subset named M5HisDoc-H, which contains the same number of images as M5HisDoc-R. The dataset exhibits diverse styles, significant scale variations, dense texts, and an extensive character set. We conduct benchmarking experiments on five tasks: text line detection, text line recognition, character detection, character recognition, and reading order prediction. We also conduct cross-validation with other benchmarks. Experimental results demonstrate that the M5HisDoc dataset can offer new challenges and great opportunities for future research in this field, thereby providing deep insights into the solution for HDAR. The dataset is available at https://github.com/HCIILAB/M5HisDoc.</details>

### MAViL: Masked Audio-Video Learners

**Authors:** Po-Yao Huang, Vasu Sharma, Hu Xu, Chaitanya Ryali, haoqi fan, Yanghao Li, Shang-Wen Li, Gargi Ghosh, Jitendra Malik, Christoph Feichtenhofer

**<details><summary>Abstract**</summary>

We present Masked Audio-Video Learners (MAViL) to learn audio-visual representations with three complementary forms of self-supervision: (1) reconstructing masked raw audio and video inputs, (2) intra-modal and inter-modal contrastive learning with masking, and (3) self-training to predict aligned and contextualized audio-video representations learned from the first two objectives. Empirically, MAViL achieves state-of-the-art audio-video classification performance on AudioSet (53.3 mAP) and VGGSound (67.1\% accuracy), surpassing recent self-supervised models and supervised models that utilize external labeled data. Notably, pre-training with MAViL not only enhances performance in multimodal classification and retrieval tasks, but it also improves the representations of each modality in isolation, without relying on information from the other modality during uni-modal fine-tuning or inference. The code and models are available at https://github.com/facebookresearch/MAViL.</details>

### MCUFormer: Deploying Vision Tranformers on Microcontrollers with Limited Memory

**Authors:** Yinan Liang, Ziwei Wang, Xiuwei Xu, Yansong Tang, Jie Zhou, Jiwen Lu

**<details><summary>Abstract**</summary>

Due to the high price and heavy energy consumption of GPUs, deploying deep models on IoT devices such as microcontrollers makes significant contributions for ecological AI. Conventional methods successfully enable convolutional neural network inference of high resolution images on microcontrollers, while the framework for vision transformers that achieve the state-of-the-art performance in many vision applications still remains unexplored. In this paper, we propose a hardware-algorithm co-optimizations method called MCUFormer to deploy vision transformers on microcontrollers with extremely limited memory, where we jointly design transformer architecture and construct the inference operator library to fit the memory resource constraint. More specifically, we generalize the one-shot network architecture search (NAS) to discover the optimal architecture with highest task performance given the memory budget from the microcontrollers, where we enlarge the existing search space of vision transformers by considering the low-rank decomposition dimensions and patch resolution for memory reduction. For the construction of the inference operator library of vision transformers, we schedule the memory buffer during inference through operator integration, patch embedding decomposition, and token overwriting, allowing the memory buffer to be fully utilized to adapt to the forward pass of the vision transformer. Experimental results demonstrate that our MCUFormer achieves 73.62\% top-1 accuracy on ImageNet for image classification with 320KB memory on STM32F746 microcontroller. Code is available at https://github.com/liangyn22/MCUFormer.</details>

### MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection

**Authors:** Junho Song, Keonwoo Kim, Jeonglyul Oh, Sungzoon Cho

**<details><summary>Abstract**</summary>

Detecting anomalies in real-world multivariate time series data is challenging due to complex temporal dependencies and inter-variable correlations. Recently, reconstruction-based deep models have been widely used to solve the problem. However, these methods still suffer from an over-generalization issue and fail to deliver consistently high performance. To address this issue, we propose the MEMTO, a memory-guided Transformer using a reconstruction-based approach. It is designed to incorporate a novel memory module that can learn the degree to which each memory item should be updated in response to the input data. To stabilize the training procedure, we use a two-phase training paradigm which involves using K-means clustering for initializing memory items. Additionally, we introduce a bi-dimensional deviation-based detection criterion that calculates anomaly scores considering both input space and latent space. We evaluate our proposed method on five real-world datasets from diverse domains, and it achieves an average anomaly detection F1-score of 95.74%, significantly outperforming the previous state-of-the-art methods. We also conduct extensive experiments to empirically validate the effectiveness of our proposed model's key components.</details>

### MIMEx: Intrinsic Rewards from Masked Input Modeling

**Authors:** Toru Lin, Allan Jabri

**<details><summary>Abstract**</summary>

Exploring in environments with high-dimensional observations is hard. One promising approach for exploration is to use intrinsic rewards, which often boils down to estimating "novelty" of states, transitions, or trajectories with deep networks. Prior works have shown that conditional prediction objectives such as masked autoencoding can be seen as stochastic estimation of pseudo-likelihood. We show how this perspective naturally leads to a unified view on existing intrinsic reward approaches: they are special cases of conditional prediction, where the estimation of novelty can be seen as pseudo-likelihood estimation with different mask distributions. From this view, we propose a general framework for deriving intrinsic rewards -- Masked Input Modeling for Exploration (MIMEx) -- where the mask distribution can be flexibly tuned to control the difficulty of the underlying conditional prediction task. We demonstrate that MIMEx can achieve superior results when compared against competitive baselines on a suite of challenging sparse-reward visuomotor tasks.</details>

### Markovian Sliced Wasserstein Distances: Beyond Independent Projections

**Authors:** Khai Nguyen, Tongzheng Ren, Nhat Ho

**<details><summary>Abstract**</summary>

Sliced Wasserstein (SW) distance suffers from redundant projections due to independent uniform random projecting directions. To partially overcome the issue, max K sliced Wasserstein  (Max-K-SW) distance ($K\geq 1$), seeks the best discriminative orthogonal projecting directions. Despite being able to reduce the number of projections, the metricity of the Max-K-SW cannot be guaranteed in practice due to the non-optimality of the optimization. Moreover,  the orthogonality constraint is also computationally expensive and might not be effective. To address the problem, we introduce a new family of SW distances, named Markovian sliced Wasserstein (MSW) distance, which imposes a first-order Markov structure on projecting directions. We discuss various members of the MSW by specifying the Markov structure including the prior distribution, the transition distribution, and the burning and thinning technique. Moreover, we investigate the theoretical properties of  MSW including topological properties (metricity, weak convergence, and connection to other distances), statistical properties (sample complexity, and Monte Carlo estimation error), and computational properties (computational complexity and memory complexity). Finally, we compare MSW distances with previous SW variants in various applications such as gradient flows, color transfer, and deep generative modeling to demonstrate the favorable performance of the MSW.</details>

### Mathematical Capabilities of ChatGPT

**Authors:** Simon Frieder, Luca Pinchetti, Chevalier, Ryan-Rhys Griffiths, Tommaso Salvatori, Thomas Lukasiewicz, Philipp Petersen, Julius Berner

**<details><summary>Abstract**</summary>

We investigate the mathematical capabilities of two iterations of ChatGPT (released 9-January-2023 and 30-January-2023) and of GPT-4 by testing them on publicly available datasets, as well as hand-crafted ones, using a novel methodology. In contrast to formal mathematics, where large databases of formal proofs are available (e.g., mathlib, the Lean Mathematical Library), current datasets of natural-language mathematics used to benchmark language models either cover only elementary mathematics or are very small. We address this by publicly releasing two new datasets: GHOSTS and miniGHOSTS. These are the first natural-language datasets curated by working researchers in mathematics that (1) aim to cover graduate-level mathematics, (2) provide a holistic overview of the mathematical capabilities of language models, and (3) distinguish multiple dimensions of mathematical reasoning. These datasets test on 1636 human expert evaluations whether ChatGPT and GPT-4 can be helpful assistants to professional mathematicians by emulating use cases that arise in the daily professional activities of mathematicians. We benchmark the models on a range of fine-grained performance metrics. For advanced mathematics, this is the most detailed evaluation effort to date. We find that ChatGPT and GPT-4 can be used most successfully as mathematical assistants for querying facts, acting as mathematical search engines and knowledge base interfaces. GPT-4 can additionally be used for undergraduate-level mathematics but fails on graduate-level difficulty. Contrary to many positive reports in the media about GPT-4 and ChatGPT's exam-solving abilities (a potential case of selection bias), their overall mathematical performance is well below the level of a graduate student. Hence, if you aim to use ChatGPT to pass a graduate-level math exam, you would be better off copying from your average peer!</details>

### Matrix Compression via Randomized Low Rank and Low Precision Factorization

**Authors:** Rajarshi Saha, Varun Srivastava, Mert Pilanci

**<details><summary>Abstract**</summary>

Matrices are exceptionally useful in various fields of study as they provide a convenient framework to organize and manipulate data in a structured manner. However, modern matrices can  involve billions of elements, making their storage and processing quite demanding in terms of computational resources and memory usage. Although prohibitively large, such matrices are often approximately low rank. We propose an algorithm that exploits this structure to obtain a low rank decomposition of any matrix $\mathbf{A}$ as $\mathbf{A} \approx \mathbf{L}\mathbf{R}$, where $\mathbf{L}$ and $\mathbf{R}$ are the low rank factors. The total number of elements in $\mathbf{L}$ and $\mathbf{R}$ can be significantly less than that in $\mathbf{A}$. Furthermore, the entries of $\mathbf{L}$ and $\mathbf{R}$ are quantized to low precision formats -- compressing $\mathbf{A}$ by giving us a low rank and low precision factorization. Our algorithm first computes an approximate basis of the range space of $\mathbf{A}$ by randomly sketching its columns, followed by a quantization of the vectors constituting this basis. It then computes approximate projections of the columns of $\mathbf{A}$ onto this quantized basis. We derive upper bounds on the approximation error of our algorithm, and analyze the impact of target rank and quantization bit-budget. The tradeoff between compression ratio and approximation accuracy allows for flexibility in choosing these parameters based on specific application requirements. We empirically demonstrate the efficacy of our algorithm in image compression, nearest neighbor classification of image and text embeddings, and compressing the layers of LlaMa-$7$b. Our results illustrate that we can achieve compression ratios as aggressive as one bit per matrix coordinate, all while surpassing or maintaining the performance of traditional compression techniques.</details>

### Maximum State Entropy Exploration using Predecessor and Successor Representations

**Authors:** Arnav Kumar Jain, Lucas Lehnert, Irina Rish, Glen Berseth

**<details><summary>Abstract**</summary>

Animals have a developed ability to explore that aids them in important tasks such as locating food, exploring for shelter, and finding misplaced items. These exploration skills necessarily track where they have been so that they can plan for finding items with relative efficiency. Contemporary exploration algorithms often learn a less efficient exploration strategy because they either condition only on the current state or simply rely on making random open-loop exploratory moves. In this work, we propose $\eta\psi$-Learning, a method to learn efficient exploratory policies by conditioning on past episodic experience to make the next exploratory move. Specifically, $\eta\psi$-Learning learns an exploration policy that maximizes the entropy of the state visitation distribution of a single trajectory. Furthermore, we demonstrate how variants of the predecessor representation and successor representations can be combined to predict the state visitation entropy. Our experiments demonstrate the efficacy of $\eta\psi$-Learning to strategically explore the environment and maximize the state coverage with limited samples.</details>

### May the Force be with You: Unified Force-Centric Pre-Training for 3D Molecular Conformations

**Authors:** Rui Feng, Qi Zhu, Huan Tran, Binghong Chen, Aubrey Toland, Rampi Ramprasad, Chao Zhang

**<details><summary>Abstract**</summary>

Recent works have shown the promise of learning pre-trained models for 3D molecular representation.However, existing pre-training models focus predominantly on equilibrium data and largely overlook off-equilibrium conformations.It is challenging to extend these methods to off-equilibrium data because their training objective relies on assumptions ofconformations being the local energy minima. We address this gap by proposing a force-centric pretraining model for 3D molecular conformations covering both equilibrium and off-equilibrium data.For off-equilibrium data, our model learns directly from their atomic forces. For equilibrium data, we introduce zero-force regularization and forced-based denoising techniques to approximate near-equilibrium forces.We obtain a unified pre-trained model for 3D molecular representation with over 15 million diverse conformations. Experiments show that, with our pre-training objective, we increase forces accuracy by around 3 times compared to the un-pre-trained Equivariant Transformer model. By incorporating regularizations on equilibrium data, we solved the problem of unstable MD simulations in vanilla Equivariant Transformers, achieving state-of-the-art simulation performance with 2.45 times faster inference time than NequIP.  As a powerful molecular encoder, our pre-trained model achieves on-par performance with state-of-the-art property prediction tasks.</details>

### Metis: Understanding and Enhancing In-Network Regular Expressions

**Authors:** Zhengxin Zhang, Yucheng Huang, Guanglin Duan, Qing Li, Dan Zhao, Yong Jiang, Lianbo Ma, Xi Xiao, Hengyang Xu

**<details><summary>Abstract**</summary>

Regular expressions (REs) offer one-shot solutions for many networking tasks, e.g., network intrusion detection. However, REs purely rely on expert knowledge and cannot utilize labeled data for better accuracy. Today, neural networks (NNs) have shown superior accuracy and flexibility, thanks to their ability to learn from rich labeled data. Nevertheless, NNs are often incompetent in cold-start scenarios and too complex for deployment on network devices. In this paper, we propose Metis, a general framework that converts REs to network device affordable models for superior accuracy and throughput by taking advantage of REs' expert knowledge and NNs' learning ability. In Metis, we convert REs to byte-level recurrent neural networks (BRNNs) without training. The BRNNs preserve expert knowledge from REs and offer adequate accuracy in cold-start scenarios. When rich labeled data is available, the performance of BRNNs can be improved by training. Furthermore, we design a semi-supervised knowledge distillation to transform the BRNNs into pooling soft random forests (PSRFs) that can be deployed on network devices. To the best of our knowledge, this is the first method to employ model inference as an alternative to RE matching in network scenarios. We collect network traffic data on our campus for three weeks and evaluate Metis on them. Experimental results show that Metis is more accurate than original REs and other baselines, achieving superior throughput when deployed on network devices.</details>

### Michelangelo: Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation

**Authors:** Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, BIN FU, Tao Chen, Gang Yu, Shenghua Gao

**<details><summary>Abstract**</summary>

We present a novel alignment-before-generation approach to tackle the challenging task of generating general 3D shapes based on 2D images or texts. Directly learning a conditional generative model from images or texts to 3D shapes is prone to producing inconsistent results with the conditions because 3D shapes have an additional dimension whose distribution significantly differs from that of 2D images and texts. To bridge the domain gap among the three modalities and facilitate multi-modal-conditioned 3D shape generation, we explore representing 3D shapes in a shape-image-text-aligned space. Our framework comprises two models: a Shape-Image-Text-Aligned Variational Auto-Encoder (SITA-VAE) and a conditional Aligned Shape Latent Diffusion Model (ASLDM). The former model encodes the 3D shapes into the shape latent space aligned to the image and text and reconstructs the fine-grained 3D neural fields corresponding to given shape embeddings via the transformer-based decoder. The latter model learns a probabilistic mapping function from the image or text space to the latent shape space. Our extensive experiments demonstrate that our proposed approach can generate higher-quality and more diverse 3D shapes that better semantically conform to the visual or textural conditional inputs, validating the effectiveness of the shape-image-text-aligned space for cross-modality 3D shape generation.</details>

### Mitigating Source Bias for Fairer Weak Supervision

**Authors:** Changho Shin, Sonia Cromp, Dyah Adila, Frederic Sala

**<details><summary>Abstract**</summary>

Weak supervision enables efficient development of training sets by reducing the need for ground truth labels. However, the techniques that make weak supervision attractive---such as integrating any source of signal to estimate unknown labels---also entail the danger that the produced pseudolabels are highly biased. Surprisingly, given everyday use and the potential for increased bias, weak supervision has not been studied from the point of view of fairness. We begin such a study, starting with the observation that even when a fair model can be built from a dataset with access to ground-truth labels, the corresponding dataset labeled via weak supervision can be arbitrarily unfair. To address this, we propose and empirically validate a model for source unfairness in weak supervision, then introduce a simple counterfactual fairness-based technique that can mitigate these biases. Theoretically, we show that it is possible for our approach to simultaneously improve both accuracy and fairness---in contrast to standard fairness approaches that suffer from tradeoffs. Empirically, we show that our technique improves accuracy on weak supervision baselines by as much as 32\% while reducing demographic parity gap by 82.5\%. A simple extension of our method aimed at maximizing performance produces state-of-the-art performance in five out of ten datasets in the WRENCH benchmark.</details>

### Mixed Samples as Probes for Unsupervised Model Selection in Domain Adaptation

**Authors:** Dapeng Hu, Jian Liang, Jun Hao Liew, Chuhui Xue, Song Bai, Xinchao Wang

**<details><summary>Abstract**</summary>

Unsupervised domain adaptation (UDA) has been widely applied in improving model generalization on unlabeled target data. However, accurately selecting the best UDA model for the target domain is challenging due to the absence of labeled target data and domain distribution shifts. Traditional model selection approaches involve training extra models with source data to estimate the target validation risk. Recent studies propose practical methods that are based on measuring various properties of model predictions on target data. Although effective for some UDA models, these methods often lack stability and may lead to poor selections for other UDA models.In this paper, we present MixVal, an innovative model selection method that operates solely with unlabeled target data during inference. MixVal leverages mixed target samples with pseudo labels to directly probe the learned target structure by each UDA model. Specifically, MixVal employs two distinct types of probes: the intra-cluster mixed samples for evaluating neighborhood density and the inter-cluster mixed samples for investigating the classification boundary. With this comprehensive probing strategy, MixVal elegantly combines the strengths of two state-of-the-art model selection methods, Entropy and SND. We extensively evaluate MixVal on 11 UDA methods across 4 adaptation settings, including classification and segmentation tasks. Experimental results consistently demonstrate that MixVal achieves state-of-the-art performance and maintains exceptional stability in model selection. Code is available at \url{https://github.com/LHXXHB/MixVal}.</details>

### MoCa: Measuring Human-Language Model Alignment on Causal and Moral Judgment Tasks

**Authors:** Allen Nie, Yuhui Zhang, Atharva Shailesh Amdekar, Chris Piech, Tatsunori Hashimoto, Tobias Gerstenberg

**<details><summary>Abstract**</summary>

Human commonsense understanding of the physical and social world is organized around intuitive theories. These theories support making causal and moral judgments. When something bad happens, we naturally ask: who did what, and why? A rich literature in cognitive science has studied people's causal and moral intuitions. This work has revealed a number of factors that systematically influence people's judgments, such as the violation of norms and whether the harm is avoidable or inevitable. We collected a dataset of stories from 24 cognitive science papers and developed a system to annotate each story with the factors they investigated. Using this dataset, we test whether large language models (LLMs) make causal and moral judgments about text-based scenarios that align with those of human participants. On the aggregate level, alignment has improved with more recent LLMs. However, using statistical analyses, we find that LLMs weigh the different factors quite differently from human participants. These results show how curated, challenge datasets combined with insights from cognitive science can help us go beyond comparisons based merely on aggregate metrics: we uncover LLMs implicit tendencies and show to what extent these align with human intuitions.</details>

### Mode Connectivity in Auction Design

**Authors:** Christoph Hertrich, Yixin Tao, László A. Végh

**<details><summary>Abstract**</summary>

Optimal auction design is a fundamental problem in algorithmic game theory. This problem is notoriously difficult already in very simple settings. Recent work in differentiable economics showed that neural networks can efficiently learn known optimal auction mechanisms and discover interesting new ones. In an attempt to theoretically justify their empirical success, we focus on one of the first such networks, RochetNet, and a generalized version for affine maximizer auctions. We prove that they satisfy mode connectivity, i.e., locally optimal solutions are connected by a simple, piecewise linear path such that every solution on the path is almost as good as one of the two local optima. Mode connectivity has been recently investigated as an intriguing empirical and theoretically justifiable property of neural networks used for prediction problems. Our results give the first such analysis in the context of differentiable economics, where neural networks are used directly for solving non-convex optimization problems.</details>

### [Spotlight] Model Sparsity Can Simplify Machine Unlearning

**Authors:** jinghan jia, Jiancheng Liu, Parikshit Ram, Yuguang Yao, Gaowen Liu, Yang Liu, PRANAY SHARMA, Sijia Liu

**<details><summary>Abstract**</summary>

In response to recent data regulation requirements, machine unlearning (MU) has emerged as a critical process to remove the influence of specific examples from a given model. Although exact unlearning can be achieved through complete model retraining using the remaining dataset, the associated computational costs have driven the development of efficient, approximate unlearning techniques. Moving beyond data-centric MU approaches, our study introduces a novel model-based perspective: model sparsification via weight pruning, which is capable of reducing the gap between exact unlearning and approximate unlearning. We show in both theory and practice that model sparsity can boost the multi-criteria unlearning performance of an approximate unlearner, closing the approximation gap, while continuing to be efficient. This leads to a new MU paradigm, termed prune first, then unlearn, which infuses a sparse prior to the unlearning process. Building on this insight, we also develop a sparsity-aware unlearning method that utilizes sparsity regularization to enhance the training process of approximate unlearning. Extensive experiments show that our proposals consistently benefit MU in various unlearning scenarios. A notable highlight is the 77% unlearning efficacy gain of fine-tuning (one of the simplest approximate unlearning methods) when using our proposed sparsity-aware unlearning method. Furthermore, we showcase the practical impact of our proposed MU methods through two specific use cases: defending against backdoor attacks, and enhancing transfer learning through source class removal. These applications demonstrate the versatility and effectiveness of our approaches in addressing a variety of machine learning challenges beyond unlearning for data privacy. Codes are available at https://github.com/OPTML-Group/Unlearn-Sparse.</details>

### Model and Feature Diversity for Bayesian Neural Networks in Mutual Learning

**Authors:** Van Cuong Pham, Cuong C Nguyen, Trung Le, Dinh Phung, Gustavo Carneiro, Thanh-Toan Do

**<details><summary>Abstract**</summary>

Bayesian Neural Networks (BNNs) offer probability distributions for model parameters, enabling uncertainty quantification in predictions. However, they often underperform compared to deterministic neural networks. Utilizing mutual learning can effectively enhance the performance of peer BNNs. In this paper, we propose a novel approach to improve BNNs performance through deep mutual learning. The proposed approaches aim to increase diversity in both network parameter distributions and feature distributions, promoting peer networks to acquire distinct features that capture different characteristics of the input, which enhances the effectiveness of mutual learning. Experimental results demonstrate significant improvements in the classification accuracy, negative log-likelihood, and expected calibration error when compared to traditional mutual learning for BNNs.</details>

### Model-Based Control with Sparse Neural Dynamics

**Authors:** Ziang Liu, Genggeng Zhou, Jeff He, Tobia Marcucci, Fei-Fei Li, Jiajun Wu, Yunzhu Li

**<details><summary>Abstract**</summary>

Learning predictive models from observations using deep neural networks (DNNs) is a promising new approach to many real-world planning and control problems. However, common DNNs are too unstructured for effective planning, and current control methods typically rely on extensive sampling or local gradient descent. In this paper, we propose a new framework for integrated model learning and predictive control that is amenable to efficient optimization algorithms. Specifically, we start with a ReLU neural model of the system dynamics and, with minimal losses in prediction accuracy, we gradually sparsify it by removing redundant neurons. This discrete sparsification process is approximated as a continuous problem, enabling an end-to-end optimization of both the model architecture and the weight parameters. The sparsified model is subsequently used by a mixed-integer predictive controller, which represents the neuron activations as binary variables and employs efficient branch-and-bound algorithms. Our framework is applicable to a wide variety of DNNs, from simple multilayer perceptrons to complex graph neural dynamics. It can efficiently handle tasks involving complicated contact dynamics, such as object pushing, compositional object sorting, and manipulation of deformable objects. Numerical and hardware experiments show that, despite the aggressive sparsification, our framework can deliver better closed-loop performance than existing state-of-the-art methods.</details>

### Model-Free Active Exploration in Reinforcement Learning

**Authors:** Alessio Russo, Alexandre Proutiere

**<details><summary>Abstract**</summary>

We study the problem of exploration in Reinforcement Learning and present a novel model-free solution. We adopt an information-theoretical viewpoint and start from the  instance-specific lower bound of the number of samples that have to be collected to identify a nearly-optimal policy. Deriving this lower bound along with the optimal exploration strategy entails solving an intricate optimization problem and requires a model of the system. In turn, most existing sample optimal exploration algorithms rely on estimating the model. We derive an approximation of the instance-specific lower bound that only involves quantities that can be inferred using model-free approaches. Leveraging this approximation, we devise an ensemble-based model-free exploration strategy  applicable to both tabular and continuous Markov decision processes. Numerical results demonstrate that our strategy is able to identify efficient policies faster than state-of-the-art exploration approaches.</details>

### Modelling Cellular Perturbations with the Sparse Additive Mechanism Shift Variational Autoencoder

**Authors:** Michael Bereket, Theofanis Karaletsos

**<details><summary>Abstract**</summary>

Generative models of observations under interventions have been a vibrant topic of interest across machine learning and the sciences in recent years. For example, in drug discovery, there is a need to model the effects of diverse interventions on cells in order to characterize unknown biological mechanisms of action. We propose the Sparse Additive Mechanism Shift Variational Autoencoder, SAMS-VAE, to combine compositionality, disentanglement, and interpretability for perturbation models. SAMS-VAE models the latent state of a perturbed sample as the sum of a local latent variable capturing sample-specific variation and sparse global variables of latent intervention effects. Crucially, SAMS-VAE sparsifies these global latent variables for individual perturbations to identify disentangled, perturbation-specific latent subspaces that are flexibly composable. We evaluate SAMS-VAE both quantitatively and qualitatively on a range of tasks using two popular single cell sequencing datasets.In order to measure perturbation-specific model-properties, we also introduce a framework for evaluation of perturbation models based on average treatment effects with links to posterior predictive checks. SAMS-VAE outperforms comparable models in terms of generalization across in-distribution and out-of-distribution tasks, including a combinatorial reasoning task under resource paucity, and yields interpretable latent structures which correlate strongly to known biological mechanisms. Our results suggest SAMS-VAE is an interesting addition to the modeling toolkit for machine learning-driven scientific discovery.</details>

### Module-wise Adaptive Distillation for Multimodality Foundation Models

**Authors:** Chen Liang, Jiahui Yu, Ming-Hsuan Yang, Matthew Brown, Yin Cui, Tuo Zhao, Boqing Gong, Tianyi Zhou

**<details><summary>Abstract**</summary>

Pre-trained multimodal foundation models have demonstrated remarkable generalizability but pose challenges for deployment due to their large sizes. One effective approach to reducing their sizes is layerwise distillation, wherein small student models are trained to match the hidden representations of large teacher models at each layer. Motivated by our observation that certain architecture components, referred to as modules, contribute more significantly to the student's performance than others, we propose to track the contributions of individual modules by recording the loss decrement after distillation each module and choose the module with a greater contribution to distill more frequently. Such an approach can be naturally formulated as a multi-armed bandit (MAB) problem, where modules and loss decrements are considered as arms and rewards, respectively. We then develop a modified-Thompson sampling algorithm named OPTIMA to address the nonstationarity of module contributions resulting from model updating. Specifically, we leverage the observed contributions in recent history to estimate the changing contribution of each module and select modules based on these estimations to maximize the cumulative contribution. We evaluate the effectiveness of OPTIMA through distillation experiments on various multimodal understanding and image captioning tasks, using the CoCa-Large model \citep{yu2022coca} as the teacher model.</details>

### Moment Matching Denoising Gibbs Sampling

**Authors:** Mingtian Zhang, Alex Hawkins-Hooker, Brooks Paige, David Barber

**<details><summary>Abstract**</summary>

Energy-Based Models (EBMs) offer a versatile framework for modelling complex data distributions. However, training and sampling from EBMs continue to pose significant challenges. The widely-used Denoising Score Matching (DSM) method for scalable EBM training suffers from inconsistency issues, causing the energy model to learn a noisy data distribution. In this work, we propose an efficient sampling framework: (pseudo)-Gibbs sampling with moment matching, which enables effective sampling from the underlying clean model when given a noisy model that has been well-trained via DSM.  We explore the benefits of our approach compared to related methods and demonstrate how to scale the method to high-dimensional datasets.</details>

### Monte Carlo Tree Search with Boltzmann Exploration

**Authors:** Michael Painter, Mohamed Baioumy, Nick Hawes, Bruno Lacerda

**<details><summary>Abstract**</summary>

Monte-Carlo Tree Search (MCTS) methods, such as Upper Confidence Bound applied to Trees (UCT), are instrumental to automated planning techniques. However, UCT can be slow to explore an optimal action when it initially appears inferior to other actions. Maximum ENtropy Tree-Search (MENTS) incorporates the maximum entropy principle into an MCTS approach, utilising Boltzmann policies to sample actions, naturally encouraging more exploration. In this paper, we highlight a major limitation of MENTS: optimal actions for the maximum entropy objective do not necessarily correspond to optimal actions for the original objective. We introduce two algorithms, Boltzmann Tree Search (BTS) and Decaying ENtropy Tree-Search (DENTS), that address these limitations and preserve the benefits of Boltzmann policies, such as allowing actions to be sampled faster by using the Alias method. Our empirical analysis shows that our algorithms show consistent high performance across several benchmark domains, including the game of Go.</details>

### Mr. HiSum: A Large-scale Dataset for Video Highlight Detection and Summarization

**Authors:** Jinhwan Sul, Jihoon Han, Joonseok Lee

**<details><summary>Abstract**</summary>

Video highlight detection is a task to automatically select the most engaging moments from a long video. This problem is highly challenging since it aims to learn a general way of finding highlights from a variety of videos in the real world.The task has an innate subjectivity because the definition of a highlight differs across individuals. Therefore, to detect consistent and meaningful highlights, prior benchmark datasets have been labeled by multiple (5-20) raters. Due to the high cost of manual labeling, most existing public benchmarks are in extremely small scale, containing only a few tens or hundreds of videos. This insufficient benchmark scale causes multiple issues such as unstable evaluation or high sensitivity in traintest splits. We present Mr. HiSum, a large-scale dataset for video highlight detection and summarization, containing 31,892 videos and reliable labels aggregated over 50,000+ users per video. We empirically prove reliability of the labels as frame importance by cross-dataset transfer and user study.</details>

### Multi-Swap k-Means++

**Authors:** Lorenzo Beretta, Vincent Cohen-Addad, Silvio Lattanzi, Nikos Parotsidis

**<details><summary>Abstract**</summary>

The $k$-means++ algorithm of Arthur and Vassilvitskii (SODA 2007) is often the practitioners' choice algorithm for optimizing the popular $k$-means clustering objective and is known to give an $O(\log k)$-approximation in expectation. To obtain higher quality solutions, Lattanzi and Sohler (ICML 2019) proposed augmenting $k$-means++ with $O(k \log \log k)$ local-search steps obtained through the $k$-means++ sampling distribution to yield a $c$-approximation to the $k$-means clustering problem, where $c$ is a large absolute constant. Here we generalize and extend their local-search algorithm by considering larger and more sophisticated local-search neighborhoods hence allowing to  swap multiple centers at the same time. Our algorithm achieves a $9 + \varepsilon$ approximation ratio, which is the best possible for local search. Importantly we show that our algorithm is practical, namely easy to implement and fast enough to run on a variety of classic datasets, and outputs solutions of better cost.</details>

### Multi-task Representation Learning for Pure Exploration in Bilinear Bandits

**Authors:** Subhojyoti Mukherjee, Qiaomin Xie, Josiah Hanna, Robert Nowak

**<details><summary>Abstract**</summary>

We study multi-task representation learning for the problem of pure exploration in bilinear bandits. In bilinear bandits, an action takes theform of a pair of arms from two different entity types and the reward is a bilinear function of the known feature vectors of the arms. In the \textit{multi-task bilinear bandit problem}, we aim to find optimal actions for multiple tasks that share a common low-dimensional linear representation. The objective is to leverage this characteristic to expedite the process of identifying the best pair of arms for all tasks. We propose the algorithm GOBLIN that uses an experimental design approach to optimize sample allocations for learning the global representation as well as minimize the number of samples needed to identify the optimal pair of arms in individual tasks. To the best of our knowledge, this is the first study to give sample complexity analysis for pure exploration in bilinear bandits with shared representation. Our results demonstrate that by learning the shared representation across tasks, we achieve significantly improved sample complexity compared to the traditional approach of solving tasks independently.</details>

### Multimodal C4: An Open, Billion-scale Corpus of Images Interleaved with Text

**Authors:** Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, Yejin Choi

**<details><summary>Abstract**</summary>

In-context vision and language models like Flamingo support arbitrarily interleaved sequences of images and text as input.This format not only enables few-shot learning via interleaving independent supervised (image, text) examples, but also, more complex prompts involving interaction between images, e.g., ``What do image A and image B have in common?''To support this interface, pretraining occurs over web corpora that similarly contain interleaved images+text.To date, however, large-scale data of this form have not been publicly available.We release Multimodal C4, an augmentation of the popular text-only C4 corpus with images interleaved.We use a linear assignment algorithm to place images into longer bodies of text using CLIP features, a process that we show outperforms alternatives.Multimodal C4 spans everyday topics like cooking, travel, technology, etc. A manual inspection of a random sample of documents shows that a vast majority (88\%) of images are topically relevant, and that linear assignment frequently selects individual sentences specifically well-aligned with each image (80\%). After filtering NSFW images, ads, etc., the resulting corpus consists of 101.2M documents with 571M images interleaved in 43B English tokens.</details>

### Multimodal Deep Learning Model Unveils Behavioral Dynamics of V1 Activity in Freely Moving Mice

**Authors:** Aiwen Xu, Yuchen Hou, Cristopher Niell, Michael Beyeler

**<details><summary>Abstract**</summary>

Despite their immense success as a model of macaque visual cortex, deep convolutional neural networks (CNNs) have struggled to predict activity in visual cortex of the mouse, which is thought to be strongly dependent on the animal’s behavioral state. Furthermore, most computational models focus on predicting neural responses to static images presented under head fixation, which are dramatically different from the dynamic, continuous visual stimuli that arise during movement in the real world. Consequently, it is still unknown how natural visual input and different behavioral variables may integrate over time to generate responses in primary visual cortex (V1). To address this, we introduce a multimodal recurrent neural network that integrates gaze-contingent visual input with behavioral and temporal dynamics to explain V1 activity in freely moving mice. We show that the model achieves state-of-the-art predictions of V1 activity during free exploration and demonstrate the importance of each component in an extensive ablation study. Analyzing our model using maximally activating stimuli and saliency maps, we reveal new insights into cortical function, including the prevalence of mixed selectivity for behavioral variables in mouse V1. In summary, our model offers a comprehensive deep-learning framework for exploring the computational principles underlying V1 neurons in freely-moving animals engaged in natural behavior.</details>

### Near Optimal Reconstruction of Spherical Harmonic Expansions

**Authors:** Amir Zandieh, Insu Han, Haim Avron

**<details><summary>Abstract**</summary>

We propose an algorithm for robust recovery of the spherical harmonic expansion of functions defined on the $d$-dimensional unit sphere $\mathbb{S}^{d-1}$ using a near-optimal number of function evaluations. We show that for any $f\in L^2(\mathbb{S}^{d-1})$, the number of evaluations of $f$ needed to recover its degree-$q$ spherical harmonic expansion equals the dimension of the space of spherical harmonics of degree at most $q$, up to a logarithmic factor. Moreover, we develop a simple yet efficient kernel regression-based algorithm to recover degree-$q$ expansion of $f$ by only evaluating the function on uniformly sampled points on $\mathbb{S}^{d-1}$. Our algorithm is built upon the connections between spherical harmonics and Gegenbauer polynomials. Unlike the prior results on fast spherical harmonic transform, our proposed algorithm works efficiently using a nearly optimal number of samples in any dimension $d$. Furthermore, we illustrate the empirical performance of our algorithm on numerical examples.</details>

### Near-Optimal Bounds for Learning Gaussian Halfspaces with Random Classification Noise

**Authors:** Ilias Diakonikolas, Jelena Diakonikolas, Daniel Kane, Puqian Wang, Nikos Zarifis

**<details><summary>Abstract**</summary>

We study the problem of learning general (i.e., not necessarily homogeneous) halfspaces with Random Classification Noise under the Gaussian distribution. We establish nearly-matching algorithmic and Statistical Query (SQ) lower bound results revealing a surprising information-computation gap for this basic problem. Specifically, the sample complexity of this learning problem is $\widetilde{\Theta}(d/\epsilon)$, where $d$ is the dimension and $\epsilon$ is the excess error. Our positive result is a computationally efficient learning algorithm with sample complexity$\tilde{O}(d/\epsilon + d/\max(p, \epsilon))^2)$, where $p$ quantifies the bias of the target halfspace. On the lower bound side, we show that any efficient SQ algorithm (or low-degree test)for the problem requires sample complexity at least $\Omega(d^{1/2}/(\max(p, \epsilon))^2)$. Our lower bound suggests that this quadratic dependence on $1/\epsilon$ is inherent for efficient algorithms.</details>

### Neural Fields with Hard Constraints of Arbitrary Differential Order

**Authors:** Fangcheng Zhong, Kyle Fogarty, Param Hanji, Tianhao Wu, Alejandro Sztrajman, Andrew Spielberg, Andrea Tagliasacchi, Petra Bosilj, Cengiz Oztireli

**<details><summary>Abstract**</summary>

While deep learning techniques have become extremely popular for solving a broad range of optimization problems, methods to enforce hard constraints during optimization, particularly on deep neural networks, remain underdeveloped. Inspired by the rich literature on meshless interpolation and its extension to spectral collocation methods in scientific computing, we develop a series of approaches for enforcing hard constraints on neural fields, which we refer to as Constrained Neural Fields (CNF). The constraints can be specified as a linear operator applied to the neural field and its derivatives. We also design specific model representations and training strategies for problems where standard models may encounter difficulties, such as conditioning of the system, memory consumption, and capacity of the network when being constrained. Our approaches are demonstrated in a wide range of real-world applications. Additionally, we develop a framework that enables highly efficient model and constraint specification, which can be readily applied to any downstream task where hard constraints need to be explicitly satisfied during optimization.</details>

### Neural Polarizer: A Lightweight and Effective Backdoor Defense via Purifying Poisoned Features

**Authors:** Mingli Zhu, Shaokui Wei, Hongyuan Zha, Baoyuan Wu

**<details><summary>Abstract**</summary>

Recent studies have demonstrated the susceptibility of deep neural networks to backdoor attacks. Given a backdoored model, its prediction of a poisoned sample with trigger will be dominated by the trigger information, though trigger information and benign information coexist. Inspired by the mechanism of the optical polarizer that a polarizer could pass light waves with particular polarizations while filtering light waves with other polarizations, we propose a novel backdoor defense method by inserting a learnable neural polarizer into the backdoored model as an intermediate layer, in order to purify the poisoned sample via filtering trigger information while maintaining benign information. The neural polarizer is instantiated as one lightweight linear transformation layer, which is learned through solving a well designed bi-level optimization problem, based on a limited clean dataset. Compared to other fine-tuning-based defense methods which often adjust all parameters of the backdoored model, the proposed method only needs to learn one additional layer, such that it is more efficient and requires less clean data. Extensive experiments demonstrate the effectiveness and efficiency of our method in removing backdoors across various neural network architectures and datasets, especially in the case of very limited clean data. Codes are available at \href{https://github.com/SCLBD/BackdoorBench}{https://github.com/SCLBD/BackdoorBench} (PyTorch) and \href{https://github.com/JulieCarlon/NPD-MindSpore}{https://github.com/JulieCarlon/NPD-MindSpore} (MindSpore).</details>

### NeuroEvoBench:  Benchmarking Evolutionary Optimizers for Deep Learning Applications

**Authors:** Robert Lange, Yujin Tang, Yingtao Tian

**<details><summary>Abstract**</summary>

Recently, the Deep Learning community has become interested in evolutionary optimization (EO) as a means to address hard optimization problems, e.g. meta-learning through long inner loop unrolls or optimizing non-differentiable operators. One core reason for this trend has been the recent innovation in hardware acceleration and compatible software -- making distributed population evaluations much easier than before. Unlike for gradient descent-based methods though, there is a lack of hyperparameter understanding and best practices for EO – arguably due to severely less `graduate student descent' and benchmarking being performed for EO methods. Additionally, classical benchmarks from the evolutionary community provide few practical insights for Deep Learning applications. This poses challenges for newcomers to hardware-accelerated EO and hinders significant adoption. Hence, we establish a new benchmark of EO methods (NEB) tailored toward Deep Learning applications and exhaustively evaluate traditional and meta-learned EO. We investigate core scientific questions including resource allocation, fitness shaping, normalization, regularization & scalability of EO. The benchmark is open-sourced at https://github.com/neuroevobench/neuroevobench under Apache-2.0 license.</details>

### [Spotlight] Newton–Cotes Graph Neural Networks: On the Time Evolution of Dynamic Systems

**Authors:** Lingbing Guo, Weiqing Wang, Zhuo Chen, Ningyu Zhang, Zequn Sun, Yixuan Lai, Qiang Zhang, Huajun Chen

**<details><summary>Abstract**</summary>

Reasoning system dynamics is one of the most important analytical approaches for many scientific studies. With the initial state of a system as input, the recent graph neural networks (GNNs)-based methods are capable of predicting the future state distant in time with high accuracy. Although these methods have diverse designs in modeling the coordinates and interacting forces of the system, we show that they actually share a common paradigm that learns the integration of the velocity over the interval between the initial and terminal coordinates. However, their integrand is constant w.r.t. time. Inspired by this observation, we propose a new approach to predict the integration based on several velocity estimations with Newton–Cotes formulas and prove its effectiveness theoretically. Extensive experiments on several benchmarks empirically demonstrate consistent and significant improvement compared with the state-of-the-art methods.</details>

### No-Regret Learning with Unbounded Losses: The Case of Logarithmic Pooling

**Authors:** Eric Neyman, Tim Roughgarden

**<details><summary>Abstract**</summary>

For each of $T$ time steps, $m$ experts report probability distributions over $n$ outcomes; we wish to learn to aggregate these forecasts in a way that attains a no-regret guarantee. We focus on the fundamental and practical aggregation method known as *logarithmic pooling* -- a weighted average of log odds -- which is in a certain sense the optimal choice of pooling method if one is interested in minimizing log loss (as we take to be our loss function). We consider the problem of learning the best set of parameters (i.e. expert weights) in an online adversarial setting.  We assume (by necessity) that the adversarial choices of outcomes and forecasts are consistent, in the sense that experts report calibrated forecasts. Imposing this constraint creates a (to our knowledge) novel semi-adversarial setting in which the adversary retains a large amount of flexibility. In this setting, we present an algorithm based on online mirror descent that learns expert weights in a way that attains $O(\sqrt{T} \log T)$ expected regret as compared with the best weights in hindsight.</details>

### Non-Convex Bilevel Optimization with Time-Varying Objective Functions

**Authors:** Sen Lin, Daouda Sow, Kaiyi Ji, Yingbin Liang, Ness Shroff

**<details><summary>Abstract**</summary>

Bilevel optimization has become a powerful tool in a wide variety of machine learning problems. However, the current nonconvex bilevel optimization considers an offline dataset and static functions, which may not work well in emerging online applications with streaming data and time-varying functions. In this work, we study online bilevel optimization (OBO) where the functions can be time-varying and the agent continuously updates the decisions with online streaming data. To deal with the function variations and the unavailability of the true hypergradients in OBO, we propose a single-loop online bilevel optimizer with window averaging (SOBOW), which updates the outer-level decision based on a window average of the most recent hypergradient estimations stored in the memory. Compared to existing algorithms, SOBOW is computationally efficient and does not need to know previous functions. To handle the unique technical difficulties rooted in single-loop update and function variations for OBO, we develop a novel analytical technique that disentangles the complex couplings between decision variables, and carefully controls the hypergradient estimation error. We show that SOBOW can achieve a sublinear bilevel local regret under mild conditions. Extensive experiments across multiple domains corroborate the effectiveness of SOBOW.</details>

### Norm-based Generalization Bounds for Sparse Neural Networks

**Authors:** Tomer Galanti, Mengjia Xu, Liane Galanti, Tomaso Poggio

**<details><summary>Abstract**</summary>

In this paper, we derive norm-based generalization bounds for sparse ReLU neural networks, including convolutional neural networks. These bounds differ from previous ones because they consider the sparse structure of the neural network architecture and the norms of the convolutional filters, rather than the norms of the (Toeplitz) matrices associated with the convolutional layers. Theoretically, we demonstrate that these bounds are significantly tighter than standard norm-based generalization bounds. Empirically, they offer relatively tight estimations of generalization for various simple classification problems. Collectively, these findings suggest that the sparsity of the underlying target function and the model's architecture plays a crucial role in the success of deep learning.</details>

### NurViD: A Large Expert-Level Video Database for Nursing Procedure Activity Understanding

**Authors:** Ming Hu, Lin Wang, Siyuan Yan, Don Ma, Qingli Ren, Peng Xia, Wei Feng, Peibo Duan, Lie Ju, Zongyuan Ge

**<details><summary>Abstract**</summary>

The application of deep learning to nursing procedure activity understanding has the potential to greatly enhance the quality and safety of nurse-patient interactions. By utilizing the technique, we can facilitate training and education, improve quality control, and enable operational compliance monitoring. However, the development of automatic recognition systems in this field is currently hindered by the scarcity of appropriately labeled datasets. The existing video datasets pose several limitations: 1) these datasets are small-scale in size to support comprehensive investigations of nursing activity; 2) they primarily focus on single procedures, lacking expert-level annotations for various nursing procedures and action steps; and 3) they lack temporally localized annotations, which prevents the effective localization of targeted actions within longer video sequences. To mitigate these limitations, we propose NurViD, a large video dataset with expert-level annotation for nursing procedure activity understanding. NurViD consists of over 1.5k videos totaling 144 hours, making it approximately four times longer than the existing largest nursing activity datasets. Notably, it encompasses 51 distinct nursing procedures and 177 action steps, providing a much more comprehensive coverage compared to existing datasets that primarily focus on limited procedures. To evaluate the efficacy of current deep learning methods on nursing activity understanding, we establish three benchmarks on NurViD: procedure recognition on untrimmed videos, procedure and action recognition on trimmed videos, and action detection. Our benchmark and code will be available at https://github.com/minghu0830/NurViD-benchmark.</details>

### ODE-based Recurrent Model-free Reinforcement Learning for POMDPs

**Authors:** Xuanle Zhao, Duzhen Zhang, Han Liyuan, Tielin Zhang, Bo Xu

**<details><summary>Abstract**</summary>

Neural ordinary differential equations (ODEs) are widely recognized as the standard for modeling physical mechanisms, which help to perform approximate inference in unknown physical or biological environments. In partially observable (PO) environments, how to infer unseen information from raw observations puzzled the agents. By using a recurrent policy with a compact context, context-based reinforcement learning provides a flexible way to extract unobservable information from historical transitions. To help the agent extract more dynamics-related information, we present a novel ODE-based recurrent model combines with model-free reinforcement learning (RL) framework to solve partially observable Markov decision processes (POMDPs). We experimentally demonstrate the efficacy of our methods across various PO continuous control and meta-RL tasks. Furthermore, our experiments illustrate that our method is robust against irregular observations, owing to the ability of ODEs to model irregularly-sampled time series.</details>

### OceanBench: The Sea Surface Height Edition

**Authors:** J. Emmanuel Johnson, Quentin Febvre, Anastasiia Gorbunova, Sam Metref, Maxime Ballarotta, Julien Le Sommer, ronan fablet

**<details><summary>Abstract**</summary>

The ocean is a crucial component of the Earth's system. It profoundly influences human activities and plays a critical role in climate regulation. Our understanding has significantly improved over the last decades with the advent of satellite remote sensing data, allowing us to capture essential sea surface quantities over the globe, e.g., sea surface height (SSH). Despite their ever-increasing abundance, ocean satellite data presents challenges for information extraction due to their sparsity and irregular sampling, signal complexity, and noise. Machine learning (ML) techniques have demonstrated their capabilities in dealing with large-scale, complex signals. Therefore we see an opportunity for these ML models to harness the full extent of the information contained in ocean satellite data. However, data representation and relevant evaluation metrics can be the defining factors when determining the success of applied ML. The processing steps from the raw observation data to a ML-ready state and from model outputs to interpretable quantities require domain expertise, which can be a significant barrier to entry for ML researchers. In addition, imposing fixed processing steps, like committing to specific variables, regions, and geometries, will narrow the scope of ML models and their potential impact on real-world applications. OceanBench is a unifying framework that provides standardized processing steps that comply with domain-expert standards. It is designed with a flexible and pedagogical abstraction: it a) provides plug-and-play data and pre-configured pipelines for ML researchers to benchmark their models w.r.t. ML and domain-related baselines and b) provides a transparent and configurable framework for researchers to customize and extend the pipeline for their tasks. In this work, we demonstrate the OceanBench framework through a first edition dedicated to SSH interpolation challenges. We provide datasets and ML-ready benchmarking pipelines for the long-standing problem of interpolating observations from simulated ocean satellite data, multi-modal and multi-sensor fusion issues, and transfer-learning to real ocean satellite observations. The OceanBench framework is available at https://github.com/jejjohnson/oceanbench and the dataset registry is available at https://github.com/quentinf00/oceanbench-data-registry.</details>

### Offline Imitation Learning with Variational Counterfactual Reasoning

**Authors:** Zexu Sun, Bowei He, Jinxin Liu, Xu Chen, Chen Ma, Shuai Zhang

**<details><summary>Abstract**</summary>

In offline imitation learning (IL), an agent aims to learn an optimal expert behavior policy without additional online environment interactions. However, in many real-world scenarios, such as robotics manipulation, the offline dataset is collected from suboptimal behaviors without rewards. Due to the scarce expert data, the agents usually suffer from simply memorizing poor trajectories and are vulnerable to the variations in the environments, lacking the capability of generalizing to new environments.To automatically generate high-quality expert data and improve the generalization ability of the agent, we propose a framework named \underline{O}ffline \underline{I}mitation \underline{L}earning with \underline{C}ounterfactual data \underline{A}ugmentation (OILCA) by doing counterfactual inference. In particular, we leverage identifiable variational autoencoder to generate \textit{counterfactual} samples for expert data augmentation. We theoretically analyze the influence of the generated expert data and the improvement of generalization. Moreover, we conduct extensive experiments to demonstrate that our approach significantly outperforms various baselines on both \textsc{DeepMind Control Suite} benchmark for in-distribution performance and \textsc{CausalWorld} benchmark for out-of-distribution generalization.</details>

### Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization

**Authors:** Xiangsen Wang, Haoran Xu, Yinan Zheng, Xianyuan Zhan

**<details><summary>Abstract**</summary>

Offline reinforcement learning (RL) has received considerable attention in recent years due to its attractive capability of learning policies from offline datasets without environmental interactions. Despite some success in the single-agent setting, offline multi-agent RL (MARL) remains to be a challenge. The large joint state-action space and the coupled multi-agent behaviors pose extra complexities for offline policy optimization. Most existing offline MARL studies simply apply offline data-related regularizations on individual agents, without fully considering the multi-agent system at the global level. In this work, we present OMIGA, a new offline multi-agent RL algorithm with implicit global-to-local value regularization. OMIGA provides a principled framework to convert global-level value regularization into equivalent implicit local value regularizations and simultaneously enables in-sample learning, thus elegantly bridging multi-agent value decomposition and policy learning with offline regularizations. Based on comprehensive experiments on the offline multi-agent MuJoCo and StarCraft II micro-management tasks, we show that OMIGA achieves superior performance over the state-of-the-art offline MARL methods in almost all tasks.</details>

### Offline Reinforcement Learning with Differential Privacy

**Authors:** Dan Qiao, Yu-Xiang Wang

**<details><summary>Abstract**</summary>

The offline reinforcement learning (RL) problem is often motivated by the need to learn data-driven decision policies in financial, legal and healthcare applications.  However, the learned policy could retain sensitive information of individuals in the training data (e.g., treatment and outcome of patients), thus susceptible to various privacy risks. We design offline RL algorithms with differential privacy guarantees which provably prevent such risks. These algorithms also enjoy strong instance-dependent learning bounds under both tabular and linear Markov Decision Process (MDP) settings. Our theory and simulation suggest that the privacy guarantee comes at (almost) no drop in utility comparing to the non-private counterpart for a medium-size dataset.</details>

### On Differentially Private Sampling from Gaussian and Product Distributions

**Authors:** Badih Ghazi, Xiao Hu, Ravi Kumar, Pasin Manurangsi

**<details><summary>Abstract**</summary>

We study the problem, where given a dataset of $n$ i.i.d. samples from an unknown distribution $P$, we seek to generate a sample from a distribution that is close to $P$ in total variation distance, under the constraint of differential privacy. We study the settings where $P$ is a multi-dimensional Gaussian distribution with different assumptions: known covariance, unknown bounded covariance, and unknown unbounded covariance. We present new differentially private sampling algorithms, and show that they achieve near-optimal sample complexity in the first two settings. Moreover, when $P$ is a product distribution on the binary hypercube, we obtain a pure-DP algorithm whereas only an approximate-DP algorithm (with slightly worse sample complexity) was previously known.</details>

### On Sparse Modern Hopfield Model

**Authors:** Jerry Yao-Chieh Hu, Donglin Yang, Dennis Wu, Chenwei Xu, Bo-Yu Chen, Han Liu

**<details><summary>Abstract**</summary>

We introduce the sparse modern Hopfield model as a sparse extension of the modern Hopfield model.Like its dense counterpart, the sparse modern Hopfield model equips a memory-retrieval dynamics whose one-step approximation corresponds to the sparse attention mechanism. Theoretically, our key contribution is a principled derivation of a closed-form sparse Hopfield energy using the convex conjugate of the sparse entropic regularizer.Building upon this, we derive the sparse memory retrieval dynamics from the sparse energy function and show its one-step approximation is equivalent to the sparse-structured attention.Importantly, we provide a sparsity-dependent memory retrieval error bound which is provably tighter than its dense analog.The conditions for the benefits of sparsity to arise are therefore identified and discussed.In addition, we show that the sparse modern Hopfield model maintains the robust theoretical properties of its dense counterpart, including rapid fixed point convergence and exponential memory capacity.Empirically, we use both synthetic and real-world datasets to demonstrate that the sparse Hopfield model outperforms its dense counterpart in many situations.</details>

### On skip connections and normalisation layers in deep optimisation

**Authors:** Lachlan MacDonald, Jack Valmadre, Hemanth Saratchandran, Simon Lucey

**<details><summary>Abstract**</summary>

We introduce a general theoretical framework, designed for the study of gradient optimisation of deep neural networks, that encompasses ubiquitous architecture choices including batch normalisation, weight normalisation and skip connections.  Our framework determines the curvature and regularity properties of multilayer loss landscapes in terms of their constituent layers, thereby elucidating the roles played by normalisation layers and skip connections in globalising these properties. We then demonstrate the utility of this framework in two respects. First, we give the only proof of which we are aware that a class of deep neural networks can be trained using gradient descent to global optima even when such optima only exist at infinity, as is the case for the cross-entropy cost.  Second, we identify a novel causal mechanism by which skip connections accelerate training, which we verify predictively with ResNets on MNIST, CIFAR10, CIFAR100 and ImageNet.</details>

### On the Convergence and Sample Complexity Analysis of Deep Q-Networks with $\epsilon$-Greedy Exploration

**Authors:** Shuai Zhang, Hongkang Li, Meng Wang, Miao Liu, Pin-Yu Chen, Songtao Lu, Sijia Liu, Keerthiram Murugesan, Subhajit Chaudhury

**<details><summary>Abstract**</summary>

This paper provides a theoretical understanding of deep Q-Network (DQN) with the $\varepsilon$-greedy exploration in deep reinforcement learning.Despite the tremendous empirical achievement of the DQN, its theoretical characterization remains underexplored.First, the exploration strategy is either impractical or ignored in the existing analysis.  Second, in contrast to conventional Q-learning algorithms, the DQN employs the target network and experience replay to acquire an unbiased estimation of the mean-square Bellman error (MSBE) utilized in training  the Q-network. However,the existing theoretical analysis of DQNs lacks convergence analysis or bypasses the technical challenges by deploying a significantly overparameterized neural network, which is not computationally efficient. This paper provides the first theoretical convergence and sample complexity analysis of the  practical setting of DQNs with $\epsilon$-greedy policy. We prove an iterative procedure with decaying $\epsilon$ converges to the optimal Q-value function geometrically. Moreover, a higher level of $\epsilon$ values enlarges the region of convergence but slows down the convergence, while the opposite holds for a lower level of $\epsilon$ values. Experiments justify our established theoretical insights on DQNs.</details>

### On the Convergence to a Global Solution of Shuffling-Type Gradient Algorithms

**Authors:** Lam Nguyen, Trang H. Tran

**<details><summary>Abstract**</summary>

Stochastic gradient descent (SGD) algorithm is the method of choice in many machine learning tasks thanks to its scalability and efficiency in dealing with large-scale problems. In this paper, we focus on the shuffling version of SGD which matches the mainstream practical heuristics. We show the convergence to a global solution of shuffling SGD for a class of non-convex functions under over-parameterized settings. Our analysis employs more relaxed non-convex assumptions than previous literature. Nevertheless, we maintain the desired computational complexity as shuffling SGD has achieved in the general convex setting.</details>

### On the Generalization Properties of Diffusion Models

**Authors:** Puheng Li, Zhong Li, Huishuai Zhang, Jiang Bian

**<details><summary>Abstract**</summary>

Diffusion models are a class of generative models that serve to establish a stochastic transport map between an empirically observed, yet unknown, target distribution and a known prior. Despite their remarkable success in real-world applications, a theoretical understanding of their generalization capabilities remains underdeveloped. This work embarks on a comprehensive theoretical exploration of the generalization attributes of diffusion models. We establish the theoretical estimates of the generalization gap that evolves in tandem with the training dynamics of score-based diffusion models, suggesting a polynomially small generalization error ($O(n^{-2/5}+m^{-4/5})$) on both the sample size $n$ and the model capacity $m$, evading the curse of dimensionality (i.e., independent of the data dimension) when *early-stopped*. Furthermore, we extend our quantitative analysis to a *data-dependent* scenario, wherein target distributions are portrayed as a succession of densities with progressively increasing distances between modes. This precisely elucidates the *adverse* effect of "*modes shift*'' in ground truths on the model generalization. Furthermore, these estimates are not solely theoretical constructs but have also been confirmed through numerical simulations. Our findings contribute to the rigorous understanding of diffusion models' generalization properties and provide insights that may guide practical applications.</details>

### On the Identifiability and Interpretability of Gaussian Process Models

**Authors:** Jiawen Chen, Wancen Mu, Yun Li, Didong Li

**<details><summary>Abstract**</summary>

In this paper, we critically examine the prevalent practice of using additive mixtures of Mat\'ern kernels in single-output Gaussian process (GP) models and explore the properties of multiplicative mixtures of Mat\'ern kernels for multi-output GP models. For the single-output case, we derive a series of theoretical results showing that the smoothness of a mixture of Mat\'ern kernels is determined by the least smooth component and that a GP with such a kernel is effectively equivalent to the least smooth kernel component. Furthermore, we demonstrate that none of the mixing weights or parameters within individual kernel components are identifiable. We then turn our attention to multi-output GP models and analyze the identifiability of the covariance matrix $A$ in the multiplicative kernel $K(x,y) = AK_0(x,y)$, where $K_0$ is a standard single output kernel such as Mat\'ern. We show that $A$ is identifiable up to a multiplicative constant, suggesting that multiplicative mixtures are well suited for multi-output tasks. Our findings are supported by extensive simulations and real applications for both single- and multi-output settings. This work provides insight into kernel selection and interpretation for GP models, emphasizing the importance of choosing appropriate kernel structures for different tasks.</details>

### On the Interplay between Social Welfare and Tractability of Equilibria

**Authors:** Ioannis Anagnostides, Tuomas Sandholm

**<details><summary>Abstract**</summary>

Computational tractability and social welfare (aka. efficiency) of equilibria are two fundamental but in general orthogonal considerations in algorithmic game theory. Nevertheless, we show that when (approximate) full efficiency can be guaranteed via a smoothness argument a la Roughgarden, Nash equilibria are approachable under a family of no-regret learning algorithms, thereby enabling fast and decentralized computation. We leverage this connection to obtain new convergence results in large games---wherein the number of players $n \gg 1$---under the well-documented property of full efficiency via smoothness in the limit. Surprisingly, our framework unifies equilibrium computation in disparate classes of problems including games with vanishing strategic sensitivity and two-player zero-sum games, illuminating en route an immediate but overlooked equivalence between smoothness and a well-studied condition in the optimization literature known as the Minty property. Finally, we establish that a family of no-regret dynamics attains a welfare bound that improves over the smoothness framework while at the same time guaranteeing convergence to the set of coarse correlated equilibria. We show this by employing the clairvoyant mirror descent algortihm recently introduced by Piliouras et al.</details>

### [Spotlight] On the Minimax Regret for Online Learning with Feedback Graphs

**Authors:** Khaled Eldowa, Emmanuel Esposito, Tom Cesari, Nicolò Cesa-Bianchi

**<details><summary>Abstract**</summary>

In this work, we improve on the upper and lower bounds for the regret of online learning with strongly observable undirected feedback graphs. The best known upper bound for this problem is $\mathcal{O}\bigl(\sqrt{\alpha T\ln K}\bigr)$, where $K$ is the number of actions, $\alpha$ is the independence number of the graph, and $T$ is the time horizon. The $\sqrt{\ln K}$ factor is known to be necessary when $\alpha = 1$ (the experts case). On the other hand, when $\alpha = K$ (the bandits case), the minimax rate is known to be $\Theta\bigl(\sqrt{KT}\bigr)$, and a lower bound $\Omega\bigl(\sqrt{\alpha T}\bigr)$ is known to hold for any $\alpha$. Our improved upper bound $\mathcal{O}\bigl(\sqrt{\alpha T(1+\ln(K/\alpha))}\bigr)$ holds for any $\alpha$ and matches the lower bounds for bandits and experts, while interpolating intermediate cases. To prove this result, we use FTRL with $q$-Tsallis entropy for a carefully chosen value of $q \in [1/2, 1)$ that varies with $\alpha$. The analysis of this algorithm requires a new bound on the variance term in the regret. We also show how to extend our techniques to time-varying graphs, without requiring prior knowledge of their independence numbers. Our upper bound is complemented by an improved $\Omega\bigl(\sqrt{\alpha T(\ln K)/(\ln\alpha)}\bigr)$ lower bound for all $\alpha > 1$, whose analysis relies on a novel reduction to multitask learning. This shows that a logarithmic factor is necessary as soon as $\alpha < K$.</details>

### [Spotlight] On the Planning Abilities of Large Language Models - A Critical Investigation

**Authors:** Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, Subbarao Kambhampati

**<details><summary>Abstract**</summary>

Intrigued by the claims of emergent reasoning capabilities in LLMs trained on general web corpora, in this paper, we set out to investigate their planning capabilities. We aim to evaluate (1) the effectiveness of LLMs in generating plans autonomously in commonsense planning tasks and (2) the potential of LLMs as a source of heuristic guidance for other agents (AI planners) in their planning tasks. We conduct a systematic study by generating a suite of instances on domains similar to the ones employed in the International Planning Competition and evaluate LLMs in two distinct modes: autonomous and heuristic. Our findings reveal that LLMs’ ability to generate executable plans autonomously is rather limited, with the best model (GPT-4) having an average success rate of ~12% across the domains. However, the results in the heuristic mode show more promise. In the heuristic mode, we demonstrate that LLM-generated plans can improve the search process for underlying sound planners and additionally show that external verifiers can help provide feedback on the generated plans and back-prompt the LLM for better plan generation.</details>

### On the Power of SVD in the Stochastic Block Model

**Authors:** Xinyu Mao, Jiapeng Zhang

**<details><summary>Abstract**</summary>

A popular heuristic method for improving clustering results is to apply dimensionality reduction before running clustering algorithms.It has been observed that spectral-based dimensionality reduction tools, such as PCA or SVD, improve the performance of clustering algorithms in many applications. This phenomenon indicates that spectral method not only serves as a dimensionality reduction tool, but also contributes to the clustering procedure in some sense. It is an interesting question to understand the behavior of spectral steps in clustering problems.As an initial step in this direction, this paper studies the power of vanilla-SVD algorithm in the stochastic block model (SBM). We show that, in the symmetric setting, vanilla-SVD algorithm recovers all clusters correctly. This result answers an open question posed by Van Vu (Combinatorics Probability and Computing, 2018) in the symmetric setting.</details>

### On the Sublinear Regret of GP-UCB

**Authors:** Justin Whitehouse, Aaditya Ramdas, Steven Wu

**<details><summary>Abstract**</summary>

In the kernelized bandit problem, a learner aims to sequentially compute the optimum of a function lying in a reproducing kernel Hilbert space given only noisy evaluations at sequentially chosen points. In particular, the learner aims to minimize regret, which is a measure of the suboptimality of the choices made.Arguably the most popular algorithm is the Gaussian Process Upper Confidence Bound (GP-UCB) algorithm, which involves acting based on a simple linear estimator of the unknown function.Despite its popularity, existing analyses of GP-UCB give a suboptimal regret rate, which fails to be sublinear for many commonly used kernels such as the Matern kernel. This has led to a longstanding open question: are existing regret analyses for GP-UCB tight, or can bounds be improved by using more sophisticated analytical techniques?In this work, we resolve this open question and show that GP-UCB enjoys nearly optimal regret. In particular, our results yield sublinear regret rates for the Matern kernel, improving over the state-of-the-art analyses and partially resolving a COLT open problem posed by Vakili et al. Our improvements rely on a key technical contribution --- regularizing kernel ridge estimators in proportion to the smoothness of the underlying kernel $k$. Applying this key idea together with a largely overlooked concentration result in separable Hilbert spaces (for which we provide an independent, simplified derivation), we are able to provide a tighter analysis of the GP-UCB algorithm.</details>

### One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization

**Authors:** Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, Hao Su

**<details><summary>Abstract**</summary>

Single image 3D reconstruction is an important but challenging task that requires extensive knowledge of our natural world. Many existing methods solve this problem by optimizing a neural radiance field under the guidance of 2D diffusion models but suffer from lengthy optimization time, 3D inconsistency results, and poor geometry. In this work, we propose a novel method that takes a single image of any object as input and generates a full 360-degree 3D textured mesh in a single feed-forward pass. Given a single image, we first use a view-conditioned 2D diffusion model, Zero123, to generate multi-view images for the input view, and then aim to lift them up to 3D space. Since traditional reconstruction methods struggle with inconsistent multi-view predictions, we build our 3D reconstruction module upon an SDF-based generalizable neural surface reconstruction method and propose several critical training strategies to enable the reconstruction of 360-degree meshes. Without costly optimizations, our method reconstructs 3D shapes in significantly less time than existing methods. Moreover, our method favors better geometry, generates more 3D consistent results, and adheres more closely to the input image. We evaluate our approach on both synthetic data and in-the-wild images and demonstrate its superiority in terms of both mesh quality and runtime. In addition, our approach can seamlessly support the text-to-3D task by integrating with off-the-shelf text-to-image diffusion models.</details>

### One-Pass Distribution Sketch for Measuring Data Heterogeneity in Federated Learning

**Authors:** Zichang Liu, Zhaozhuo Xu, Benjamin Coleman, Anshumali Shrivastava

**<details><summary>Abstract**</summary>

Federated learning (FL) is a machine learning paradigm where multiple client devices train models collaboratively without data exchange. Data heterogeneity problem is naturally inherited in FL since data in different clients follow diverse distributions.  To mitigate the negative influence of data heterogeneity, we need to start by measuring it across clients. However, the efficient measurement between distributions is a challenging problem, especially in high dimensionality. In this paper, we propose a one-pass distribution sketch to represent the client data distribution. Our sketching algorithm only requires a single pass of the client data, which is efficient in terms of time and memory. Moreover, we show in both theory and practice that the distance between two distribution sketches represents the divergence between their corresponding distributions. Furthermore, we demonstrate with extensive experiments that our distribution sketch improves the client selection in the FL training. We also showcase that our distribution sketch is an efficient solution to the cold start problem in FL for new clients with unlabeled data.</details>

### Online Convex Optimization with Unbounded Memory

**Authors:** Raunak Kumar, Sarah Dean, Robert Kleinberg

**<details><summary>Abstract**</summary>

Online convex optimization (OCO) is a widely used framework in online  learning. In each round, the learner chooses a decision in a convex set and an  adversary chooses a convex loss function, and then the learner suffers the  loss associated with their current decision. However, in many applications the  learner's loss depends not only on the current decision but on the entire  history of decisions until that point. The OCO framework and its existing  generalizations do not capture this, and they can only be applied to many  settings of interest after a long series of approximation arguments. They also  leave open the question of whether the dependence on memory is tight because  there are no non-trivial lower bounds. In this work we introduce a  generalization of the OCO framework, ``Online Convex Optimization with  Unbounded Memory'', that captures long-term dependence on past decisions. We  introduce the notion of $p$-effective memory capacity, $H_p$, that quantifies  the maximum influence of past decisions on present losses. We prove an  $O(\sqrt{H_p T})$ upper bound on the policy regret and a matching (worst-case)  lower bound. As a special case, we prove the first non-trivial lower bound for  OCO with finite memory~\citep{anavaHM2015online}, which could be of  independent interest, and also improve existing upper bounds. We demonstrate  the broad applicability of our framework by using it to derive regret bounds,  and to improve and simplify existing regret bound derivations, for a variety  of online learning problems including online linear control and an online  variant of performative prediction.</details>

### Online Map Vectorization for Autonomous Driving: A Rasterization Perspective

**Authors:** Gongjie Zhang, Jiahao Lin, Shuang Wu, yilin song, Zhipeng Luo, Yang Xue, Shijian Lu, Zuoguan Wang

**<details><summary>Abstract**</summary>

High-definition (HD) vectorized map is essential for autonomous driving, providing detailed and precise environmental information for advanced perception and planning. However, current map vectorization methods often exhibit deviations, and the existing evaluation metric for map vectorization lacks sufficient sensitivity to detect these deviations. To address these limitations, we propose integrating the philosophy of rasterization into map vectorization. Specifically, we introduce a new rasterization-based evaluation metric, which has superior sensitivity and is better suited to real-world autonomous driving scenarios. Furthermore, we propose MapVR (Map Vectorization via Rasterization), a novel framework that applies differentiable rasterization to vectorized outputs and then performs precise and geometry-aware supervision on rasterized HD maps. Notably, MapVR designs tailored rasterization strategies for various geometric shapes, enabling effective adaptation to a wide range of map elements. Experiments show that incorporating rasterization into map vectorization greatly enhances performance with no extra computational cost during inference, leading to more accurate map perception and ultimately promoting safer autonomous driving. Codes are available at https://github.com/ZhangGongjie/MapVR. A standalone map vectorization evaluation toolkit is available at https://github.com/jiahaoLjh/MapVectorizationEvalToolkit.</details>

### Online Nonstochastic Model-Free Reinforcement Learning

**Authors:** Udaya Ghai, Arushi Gupta, Wenhan Xia, Karan Singh, Elad Hazan

**<details><summary>Abstract**</summary>

We investigate robust model-free reinforcement learning algorithms designed for environments that may be dynamic or even adversarial. Traditional state-based policies often struggle to accommodate the challenges imposed by the presence of unmodeled disturbances in such settings. Moreover, optimizing linear state-based policies pose an obstacle for efficient optimization, leading to nonconvex objectives, even in benign environments like linear dynamical systems.Drawing inspiration from recent advancements in model-based control, we intro- duce a novel class of policies centered on disturbance signals. We define several categories of these signals, which we term pseudo-disturbances, and develop corresponding policy classes based on them. We provide efficient and practical algorithms for optimizing these policies.Next, we examine the task of online adaptation of reinforcement learning agents in the face of adversarial disturbances. Our methods seamlessly integrate with any black-box model-free approach, yielding provable regret guarantees when dealing with linear dynamics. These regret guarantees unconditionally improve the best-known results for bandit linear control in having no dependence on the state-space dimension. We evaluate our method over various standard RL benchmarks and demonstrate improved robustness.</details>

### Online PCA in Converging Self-consistent Field Equations

**Authors:** Xihan Li, Xiang Chen, Rasul Tutunov, Haitham Bou Ammar, Lei Wang, Jun Wang

**<details><summary>Abstract**</summary>

Self-consistent Field (SCF) equation is a type of nonlinear eigenvalue problem in which the matrix to be eigen-decomposed is a function of its own eigenvectors. It is of great significance in computational science for its connection to the Schrödinger equation. Traditional fixed-point iteration methods for solving such equations suffer from non-convergence issues. In this work, we present a novel perspective on such SCF equations as a principal component analysis (PCA) for non-stationary time series, in which a distribution and its own top principal components are mutually updated over time, and the equilibrium state of the model corresponds to the solution of the SCF equations. By the new perspective, online PCA techniques are able to engage in so as to enhance the convergence of the model towards the equilibrium state, acting as a new set of tools for converging the SCF equations. With several numerical adaptations, we then develop a new algorithm for converging the SCF equation, and demonstrated its high convergence capacity with experiments on both synthesized and real electronic structure scenarios.</details>

### [Oral] Online RL in Linearly $q^\pi$-Realizable MDPs Is as Easy as in Linear MDPs If You Learn What to Ignore

**Authors:** Gellert Weisz, András György, Csaba Szepesvari

**Oral Presentation:** Tu, Dec 12, 08:30 -- Oral 1A

**<details><summary>Abstract**</summary>

We consider online reinforcement learning (RL) in episodic Markov decision processes (MDPs) under the linear $q^\pi$-realizability assumption, where it is assumed that the action-values of all policies can be  expressed as linear functions of state-action features. This class is known to be more general than  linear MDPs, where the transition kernel and the reward function are assumed to be linear functions of the feature vectors. As our first contribution, we show that the difference between the two classes is the presence of states in linearly $q^\pi$-realizable MDPs where for any policy, all the actions have  approximately equal values, and skipping over these states by following an arbitrarily fixed policy in those states transforms the problem to a linear MDP. Based on this observation, we derive a novel (computationally inefficient) learning algorithm for linearly $q^\pi$-realizable MDPs that simultaneously learns what states should be skipped over and runs another learning algorithm on the linear MDP hidden in the problem. The method returns an $\epsilon$-optimal policy after $\text{polylog}(H, d)/\epsilon^2$ interactions with the MDP, where $H$ is the time horizon and $d$ is the dimension of the feature vectors, giving the first polynomial-sample-complexity online RL algorithm for this setting. The results are proved for the misspecified case, where the sample complexity is shown to degrade gracefully with the misspecification error.</details>

### Online learning of long-range dependencies

**Authors:** Nicolas Zucchet, Robert Meier, Simon Schug, Asier Mujika, Joao Sacramento

**<details><summary>Abstract**</summary>

Online learning holds the promise of enabling efficient long-term credit assignment in recurrent neural networks. However, current algorithms fall short of offline backpropagation by either not being scalable or failing to learn long-range dependencies. Here we present a high-performance online learning algorithm that merely doubles the memory and computational requirements of a single inference pass. We achieve this by leveraging independent recurrent modules in multi-layer networks, an architectural motif that has recently been shown to be particularly powerful. Experiments on synthetic memory problems and on the challenging long-range arena benchmark suite reveal that our algorithm performs competitively, establishing a new standard for what can be achieved through online learning. This ability to learn long-range dependencies offers a new perspective on learning in the brain and opens a promising avenue in neuromorphic computing.</details>

### [Oral] OpenAssistant Conversations - Democratizing Large Language Model Alignment

**Authors:** Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi Rui Tam, Keith Stevens, Abdullah Barhoum, Duc Nguyen, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, Alexander Mattick

**Oral Presentation:** Tu, Dec 12, 08:15 -- Oral 1B

**<details><summary>Abstract**</summary>

Aligning large language models (LLMs) with human preferences has proven to drastically improve usability and has driven rapid adoption as demonstrated by ChatGPT.Alignment techniques such as supervised fine-tuning (\textit{SFT}) and  reinforcement learning from human feedback (\textit{RLHF}) greatly reduce the required skill and domain knowledge to effectively harness the capabilities of LLMs, increasing their accessibility and utility across various domains.However, state-of-the-art alignment techniques like \textit{RLHF} rely on high-quality human feedback data, which is expensive to create and often remains proprietary.In an effort to democratize research on large-scale alignment, we release OpenAssistant Conversations, a human-generated, human-annotated assistant-style conversation corpus consisting of 161,443 messages in 35 different languages, annotated with 461,292 quality ratings, resulting in over 10,000 complete and fully annotated conversation trees.The corpus is a product of a worldwide crowd-sourcing effort involving over 13,500 volunteers.Models trained on OpenAssistant Conversations show consistent improvements on standard benchmarks over respective base models.We release our code\footnote{\git} and data\footnote{\data} under a fully permissive licence.</details>

### OpenDataVal: a Unified Benchmark for Data Valuation

**Authors:** Kevin Jiang, Weixin Liang, James Zou, Yongchan Kwon

**<details><summary>Abstract**</summary>

Assessing the quality and impact of individual data points is critical for improving model performance and mitigating undesirable biases within the training dataset. Several data valuation algorithms have been proposed to quantify data quality, however, there lacks a systemic and standardized benchmarking system for data valuation. In this paper, we introduce *OpenDataVal*, an easy-to-use and unified benchmark framework that empowers researchers and practitioners to apply and compare various data valuation algorithms. *OpenDataVal* provides an integrated environment that includes (i) a diverse collection of image, natural language, and tabular datasets, (ii) implementations of eleven different state-of-the-art data valuation algorithms, and (iii) a prediction model API that can import any models in scikit-learn. Furthermore, we propose four downstream machine learning tasks for evaluating the quality of data values. We perform benchmarking analysis using *OpenDataVal*, quantifying and comparing the efficacy of state-of-the-art data valuation approaches. We find that no single algorithm performs uniformly best across all tasks, and an appropriate algorithm should be employed for a user's downstream task. *OpenDataVal* is publicly available at https://opendataval.github.io with comprehensive documentation. Furthermore, we provide a leaderboard where researchers can evaluate the effectiveness of their own data valuation algorithms.</details>

### OpenIllumination: A Multi-Illumination Dataset for Inverse Rendering Evaluation on Real Objects

**Authors:** Isabella Liu, Linghao Chen, Ziyang Fu, Liwen Wu, Haian Jin, Zhong Li, Chin Ming Ryan Wong, Yi Xu, Ravi Ramamoorthi, Zexiang Xu, Hao Su

**<details><summary>Abstract**</summary>

We introduce OpenIllumination, a real-world dataset containing over 108K images of 64 objects with diverse materials, captured under 72 camera views and a large number of different illuminations. For each image in the dataset, we provide accurate camera parameters, illumination ground truth, and foreground segmentation masks. Our dataset enables the quantitative evaluation of most inverse rendering and material decomposition methods for real objects. We examine several state-of-the-art inverse rendering methods on our dataset and compare their performances. The dataset and code can be found on the project page: https://oppo-us-research.github.io/OpenIllumination.</details>

### OpenMask3D: Open-Vocabulary 3D Instance Segmentation

**Authors:** Ayca Takmaz, Elisabetta Fedele, Robert Sumner, Marc Pollefeys, Federico Tombari, Francis Engelmann

**<details><summary>Abstract**</summary>

We introduce the task of open-vocabulary 3D instance segmentation. Current approaches for 3D instance segmentation can typically only recognize object categories from a pre-defined closed set of classes that are annotated in the training datasets. This results in important limitations for real-world applications where one might need to perform tasks guided by novel, open-vocabulary queries related to a wide variety of objects. Recently, open-vocabulary 3D scene understanding methods have emerged to address this problem by learning queryable features for each point in the scene. While such a representation can be directly employed to perform semantic segmentation, existing methods cannot separate multiple object instances. In this work, we address this limitation, and propose OpenMask3D, which is a zero-shot approach for open-vocabulary 3D instance segmentation. Guided by predicted class-agnostic 3D instance masks, our model aggregates per-mask features via multi-view fusion of CLIP-based image embeddings. Experiments and ablation studies on ScanNet200 and Replica show that OpenMask3D outperforms other open-vocabulary methods, especially on the long-tail distribution. Qualitative experiments further showcase OpenMask3D’s ability to segment object properties based on free-form queries describing geometry, affordances, and materials.</details>

### Operator Learning with Neural Fields: Tackling PDEs on General Geometries

**Authors:** Louis Serrano, Lise Le Boudec, Armand Kassaï Koupaï, Thomas X Wang, Yuan Yin, Jean-Noël Vittaut, Patrick Gallinari

**<details><summary>Abstract**</summary>

Machine learning approaches for solving partial differential equations require learning mappings between function spaces. While convolutional or graph neural networks are constrained to discretized functions, neural operators present a promising milestone toward mapping functions directly. Despite impressive results they still face challenges with respect to the domain geometry and typically rely on some form of discretization. In order to alleviate such limitations, we present CORAL, a new method that leverages coordinate-based networks for solving PDEs on general geometries. CORAL is designed to remove constraints on the input mesh, making it applicable to any spatial sampling and geometry. Its ability extends to diverse problem domains, including PDE solving, spatio-temporal forecasting, and inverse problems like geometric design. CORAL demonstrates robust performance across multiple resolutions and performs well in both convex and non-convex domains, surpassing or performing on par with state-of-the-art models.</details>

### Optimal Convergence Rate for Exact Policy Mirror Descent in Discounted Markov Decision Processes

**Authors:** Emmeran Johnson, Ciara Pike-Burke, Patrick Rebeschini

**<details><summary>Abstract**</summary>

Policy Mirror Descent (PMD) is a general family of algorithms that covers a wide range of novel and fundamental methods in reinforcement learning. Motivated by the instability of policy iteration (PI) with inexact policy evaluation, unregularised PMD algorithmically regularises the policy improvement step of PI without regularising the objective function. With exact policy evaluation, PI is known to converge linearly with a rate given by the discount factor $\gamma$ of a Markov Decision Process. In this work, we bridge the gap between PI and PMD with exact policy evaluation and show that the dimension-free $\gamma$-rate of PI can be achieved by the general family of unregularised PMD algorithms under an adaptive step-size. We show that both the rate and step-size are unimprovable for PMD: we provide matching lower bounds that demonstrate that the $\gamma$-rate is optimal for PMD methods as well as PI and that the adaptive step-size is necessary to achieve it. Our work is the first to relate PMD to rate-optimality and step-size necessity. Our study of the convergence of PMD avoids the use of the performance difference lemma, which leads to a direct analysis of independent interest. We also extend the analysis to the inexact setting and establish the first dimension-optimal sample complexity for unregularised PMD under a generative model, improving upon the best-known result.</details>

### Optimal Extragradient-Based Algorithms for Stochastic Variational Inequalities with Separable Structure

**Authors:** Angela Yuan, Chris Junchi Li, Gauthier Gidel, Michael Jordan, Quanquan Gu, Simon Du

**<details><summary>Abstract**</summary>

We consider the problem of solving stochastic monotone variational inequalities with a separable structure using a stochastic first-order oracle. Building on standard extragradient for variational inequalities we propose a novel algorithm---stochastic \emph{accelerated gradient-extragradient} (AG-EG)---for strongly monotone variational inequalities (VIs). Our approach combines the strengths of extragradient and Nesterov acceleration. By showing that its iterates remain in a bounded domain and applying scheduled restarting, we prove that AG-EG has an optimal convergence rate for strongly monotone VIs. Furthermore, when specializing to the particular case of bilinearly coupled strongly-convex-strongly-concave saddle-point problems, including bilinear games, our algorithm achieves fine-grained convergence rates that match the respective lower bounds, with the stochasticity being characterized by an additive statistical error term that is optimal up to a constant prefactor.</details>

### Optimal privacy guarantees for a relaxed threat model: Addressing sub-optimal adversaries in differentially private machine learning

**Authors:** Georgios Kaissis, Alexander Ziller, Stefan Kolek, Anneliese Riess, Daniel Rueckert

**<details><summary>Abstract**</summary>

Differentially private mechanisms restrict the membership inference capabilities of powerful (optimal) adversaries against machine learning models. Such adversaries are rarely encountered in practice. In this work, we examine a more realistic threat model relaxation, where (sub-optimal) adversaries lack access to the exact model training database, but may possess related or partial data. We then formally characterise and experimentally validate adversarial membership inference capabilities in this setting in terms of hypothesis testing errors. Our work helps users to interpret the privacy properties of sensitive data processing systems under realistic threat model relaxations and choose appropriate noise levels for their use-case.</details>

### Optimal testing using combined test statistics across independent studies

**Authors:** Lasse Vuursteen, Botond Szabo, Aad van der Vaart, Harry van Zanten

**<details><summary>Abstract**</summary>

Combining test statistics from independent trials or experiments is a popular method of meta-analysis. However, there is very limited theoretical understanding of the power of the combined test, especially in high-dimensional models considering composite hypotheses tests. We derive a mathematical framework to study standard {meta-analysis} testing approaches in the context of the many normal means model, which serves as the platform to investigate more complex models.We introduce a natural and mild restriction on the meta-level combination functions of the local trials. This allows us to mathematically quantify the cost of compressing $m$ trials into real-valued test statistics and combining these. We then derive minimax lower and matching upper bounds for the separation rates of standard combination methods for e.g. p-values and e-values, quantifying the loss relative to using the full, pooled data. We observe an elbow effect, revealing that in certain cases combining the locally optimal tests in each trial results in a sub-optimal {meta-analysis} method and develop approaches to achieve the global optima. We also explore the possible gains of allowing limited coordination between the trial designs. Our results connect meta-analysis with bandwidth constraint distributed inference and build on recent information theoretic developments in the latter field.</details>

### Optimistic Active Exploration of Dynamical Systems

**Authors:** Bhavya, Lenart Treven, Cansu Sancaktar, Sebastian Blaes, Stelian Coros, Andreas Krause

**<details><summary>Abstract**</summary>

Reinforcement learning algorithms commonly seek to optimize policies for solving one particular task. How should we explore an unknown dynamical system such that the estimated model allows us to solve multiple downstream tasks in a zero-shot manner? In this paper, we address this challenge, by developing an algorithm -- OPAX -- for active exploration. OPAX uses well-calibrated probabilistic models to quantify the epistemic uncertainty about the unknown dynamics. It optimistically---w.r.t. to plausible dynamics---maximizes the information gain between the unknown dynamics and state observations. We show how the resulting optimization problem can be reduced to an optimal control problem that can be solved at each episode using standard approaches.  We analyze our algorithm for general models, and, in the case of Gaussian process dynamics, we give a sample complexity bound andshow that the epistemic uncertainty converges to zero. In our experiments, we compare OPAX with other heuristic active exploration approaches on several environments. Our experiments show that OPAX is not only theoretically sound but also performs well for zero-shot planning on novel downstream tasks.</details>

### Optimistic Exploration in Reinforcement Learning Using Symbolic Model Estimates

**Authors:** Sarath Sreedharan, Michael Katz

**<details><summary>Abstract**</summary>

There has been an increasing interest in using symbolic models along with reinforcement learning (RL) problems, where these coarser abstract models are used as a way to provide RL agents with higher level guidance. However, most of these works are inherently limited by their assumption of having an access to a symbolic approximation of the underlying problem. To address this issue, we introduce a new method for learning optimistic symbolic approximations of the underlying world model. We will see how these representations, coupled with fast diverse planners developed by the automated planning community, provide us with a new paradigm for optimistic exploration in sparse reward settings. We investigate the possibility of speeding up the learning process by generalizing learned model dynamics across similar actions with minimal human input. Finally, we evaluate the method, by testing it on multiple benchmark domains and compare it with other RL strategies.</details>

### Optimizing over trained GNNs via symmetry breaking

**Authors:** Shiqiang Zhang, Juan Campos, Christian Feldmann, David Walz, Frederik Sandfort, Miriam Mathea, Calvin Tsay, Ruth Misener

**<details><summary>Abstract**</summary>

Optimization over trained machine learning models has applications including: verification, minimizing neural acquisition functions, and integrating a trained surrogate into a larger decision-making problem. This paper formulates and solves optimization problems constrained by trained graph neural networks (GNNs). To circumvent the symmetry issue caused by graph isomorphism, we propose two types of symmetry-breaking constraints: one indexing a node 0 and one indexing the remaining nodes by lexicographically ordering their neighbor sets. To guarantee that adding these constraints will not remove all symmetric solutions, we construct a graph indexing algorithm and prove that the resulting graph indexing satisfies the proposed symmetry-breaking constraints. For the classical GNN architectures considered in this paper, optimizing over a GNN with a fixed graph is equivalent to optimizing over a dense neural network. Thus, we study the case where the input graph is not fixed, implying that each edge is a decision variable, and develop two mixed-integer optimization formulations. To test our symmetry-breaking strategies and optimization formulations, we consider an application in molecular design.</details>

### Oracle Complexity of Single-Loop Switching Subgradient Methods for Non-Smooth Weakly Convex Functional Constrained Optimization

**Authors:** Yankun Huang, Qihang Lin

**<details><summary>Abstract**</summary>

We consider a non-convex constrained optimization problem, where the objective function is weakly convex and the constraint function is either convex or weakly convex. To solve this problem, we consider the classical switching subgradient method, which is an intuitive and easily implementable first-order method whose oracle complexity was only known for convex problems. This paper provides the first analysis on the oracle complexity of the switching subgradient method for finding a nearly stationary point of non-convex problems. Our results are derived separately for convex and weakly convex constraints. Compared to existing approaches, especially the double-loop methods, the switching gradient method can be applied to non-smooth problems and achieves the same complexity using only a single loop, which saves the effort on tuning the number of inner iterations.</details>

### Order Matters in the Presence of Dataset Imbalance for Multilingual Learning

**Authors:** Dami Choi, Derrick Xin, Hamid Dadkhahi, Justin Gilmer, Ankush Garg, Orhan Firat, Chih-Kuan Yeh, Andrew Dai, Behrooz Ghorbani

**<details><summary>Abstract**</summary>

In this paper, we empirically study the optimization dynamics of multi-task learning, particularly focusing on those that govern a collection of tasks with significant data imbalance. We present a simple yet effective method of pre-training on high-resource tasks, followed by fine-tuning on a mixture of high/low-resource tasks. We provide a thorough empirical study and analysis of this method's benefits showing that it achieves consistent improvements relative to the performance trade-off profile of standard static weighting. We analyze under what data regimes this method is applicable and show its improvements empirically in neural machine translation (NMT) and multi-lingual language modeling.</details>

### [Oral] Ordering-based Conditions for Global Convergence of Policy Gradient Methods

**Authors:** Jincheng Mei, Bo Dai, Alekh Agarwal, Mohammad Ghavamzadeh, Csaba Szepesvari, Dale Schuurmans

**Oral Presentation:** Tu, Dec 12, 08:00 -- Oral 1A

**<details><summary>Abstract**</summary>

We prove that, for finite-arm bandits with linear function approximation, the global convergence of policy gradient (PG) methods depends on inter-related properties between the policy update and the representation. textcolor{blue}{First}, we establish a few key observations that frame the study: \textbf{(i)} Global convergence can be achieved under linear function approximation without policy or reward realizability, both for the standard Softmax PG and natural policy gradient (NPG). \textbf{(ii)} Approximation error is not a key quantity for characterizing global convergence in either algorithm. \textbf{(iii)} The conditions on the representation that imply global convergence are different between these two algorithms. Overall, these observations call into question approximation error as an appropriate quantity for characterizing the global convergence of PG methods under linear function approximation. \textcolor{blue}{Second}, motivated by these observations, we establish new general results: \textbf{(i)} NPG with linear function approximation achieves global convergence \emph{if and only if} the projection of the reward  onto the representable space preserves the optimal action's rank, a quantity that is not strongly related to approximation error. \textbf{(ii)} The global convergence of Softmax PG occurs if the representation satisfies a non-domination condition and can preserve the ranking of rewards, which goes well beyond policy or reward realizability. We provide experimental results to support these theoretical findings.</details>

### Orthogonal Non-negative Tensor Factorization based Multi-view Clustering

**Authors:** Jing Li, Quanxue Gao, QIANQIAN WANG, Ming Yang, Wei Xia

**<details><summary>Abstract**</summary>

Multi-view clustering (MVC) based on non-negative matrix factorization (NMF) and its variants have attracted much attention due to their advantages in clustering interpretability. However, existing NMF-based multi-view clustering methods perform NMF on each view respectively and ignore the impact of between-view. Thus, they can't well exploit the within-view spatial structure and between-view complementary information. To resolve this issue, we present orthogonal non-negative tensor factorization (Orth-NTF) and develop a novel multi-view clustering based on Orth-NTF with one-side orthogonal constraint. Our model directly performs Orth-NTF on the 3rd-order tensor which is composed of anchor graphs of views. Thus, our model directly considers the between-view relationship. Moreover, we use the tensor Schatten $p$-norm regularization as a rank approximation of the 3rd-order tensor which characterizes the cluster structure of multi-view data and exploits the between-view complementary information. In addition, we provide an optimization algorithm for the proposed method and prove mathematically that the algorithm always converges to the stationary KKT point. Extensive experiments on various benchmark datasets indicate that our proposed method is able to achieve satisfactory clustering performance.</details>

### PIXIU: A Comprehensive Benchmark, Instruction Dataset and Large Language Model for Finance

**Authors:** Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, Jimin Huang

**<details><summary>Abstract**</summary>

Although large language models (LLMs) have shown great performance in natural language processing (NLP) in the financial domain, there are no publicly available financially tailored LLMs, instruction tuning datasets, and evaluation benchmarks, which is critical for continually pushing forward the open-source development of financial artificial intelligence (AI). This paper introduces PIXIU, a comprehensive framework including the first financial LLM based on fine-tuning LLaMA with instruction data, the first instruction data with 128K data samples to support the fine-tuning, and an evaluation benchmark with 8 tasks and 15 datasets. We first construct the large-scale multi-task instruction data considering a variety of financial tasks, financial document types, and financial data modalities. We then propose a financial LLM called FinMA by fine-tuning LLaMA with the constructed dataset to be able to follow instructions for various financial tasks. To support the evaluation of financial LLMs, we propose a standardized benchmark that covers a set of critical financial tasks, including six financial NLP tasks and two financial prediction tasks. With this benchmark, we conduct a detailed analysis of FinMA and several existing LLMs, uncovering their strengths and weaknesses in handling critical financial tasks. The model, datasets, benchmark, and experimental results are open-sourced to facilitate future research in financial AI.</details>

### PLANNER: Generating Diversified Paragraph via Latent Language Diffusion Model

**Authors:** Yizhe Zhang, Jiatao Gu, Zhuofeng Wu, Shuangfei Zhai, Joshua Susskind, Navdeep Jaitly

**<details><summary>Abstract**</summary>

Autoregressive models for text sometimes generate repetitive and low-quality output because errors accumulate during the steps of generation. This issue is often attributed to exposure bias -- the difference between how a model is trained, and how it is used during inference. Denoising diffusion models provide an alternative approach in which a model can revisit and revise its output. However, they can be computationally expensive and prior efforts on text have led to models that produce less fluent output compared to autoregressive models, especially for longer text and paragraphs. In this paper, we propose PLANNER, a model that combines latent semantic diffusion with autoregressive generation, to generate fluent text while exercising global control over paragraphs. The model achieves this by combining an autoregressive "decoding" module with a "planning" module  that uses latent diffusion to generate semantic paragraph embeddings in a coarse-to-fine manner. The proposed method is evaluated on various conditional generation tasks, and results on semantic generation, text completion and summarization show its effectiveness in generating high-quality long-form text in an efficient manner.</details>

### PRIOR: Personalized Prior for Reactivating the Information Overlooked in Federated Learning.

**Authors:** Mingjia Shi, Yuhao Zhou, Kai Wang, Huaizheng Zhang, Shudong Huang, Qing Ye, Jiancheng Lv

**<details><summary>Abstract**</summary>

Classical federated learning (FL) enables training machine learning models without sharing data for privacy preservation, but heterogeneous data characteristic degrades the performance of the localized model. Personalized FL (PFL) addresses this by synthesizing personalized models from a global model via training on local data. Such a global model may overlook the specific information that the clients have been sampled. In this paper, we propose a novel scheme to inject personalized prior knowledge into the global model in each client, which attempts to mitigate the introduced incomplete information problem in PFL. At the heart of our proposed approach is a framework, the $\textit{PFL with Bregman Divergence}$ (pFedBreD), decoupling the personalized prior from the local objective function regularized by Bregman divergence for greater adaptability in personalized scenarios. We also relax the mirror descent (RMD) to extract the prior explicitly to provide optional strategies. Additionally, our pFedBreD is backed up by a convergence analysis. Sufficient experiments demonstrate that our method reaches the $\textit{state-of-the-art}$ performances on 5 datasets and outperforms other methods by up to 3.5% across 8 benchmarks. Extensive analyses verify the robustness and necessity of proposed designs. The code will be made public.</details>

### [Spotlight] PRODIGY: Enabling In-context Learning Over Graphs

**Authors:** Qian Huang, Hongyu Ren, Peng Chen, Gregor Kržmanc, Daniel Zeng, Percy Liang, Jure Leskovec

**<details><summary>Abstract**</summary>

In-context learning is the ability of a pretrained model to  adapt to novel and diverse downstream tasks by conditioning on prompt examples, without optimizing any parameters. While large language models have demonstrated this ability, how in-context learning could be performed over graphs is unexplored. In this paper, we develop \textbf{Pr}etraining \textbf{O}ver \textbf{D}iverse \textbf{I}n-Context  \textbf{G}raph S\textbf{y}stems (PRODIGY), the first pretraining framework that enables in-context learning over graphs. The key idea of our framework is to formulate in-context learning over graphs with a novel \emph{prompt graph} representation, which connects prompt examples and queries. We then propose a graph neural network architecture over the prompt graph and a corresponding family of in-context pretraining objectives. With PRODIGY, the pretrained model can directly perform novel downstream classification tasks on unseen graphs via in-context learning. We provide empirical evidence of the effectiveness of our framework by showcasing its strong in-context learning performance on tasks involving citation networks and knowledge graphs. Our approach outperforms the in-context learning accuracy of contrastive pretraining baselines with hard-coded adaptation by 18\% on average across all setups. Moreover, it also outperforms standard finetuning with limited data by 33\% on average with in-context learning.</details>

### PackQViT: Faster Sub-8-bit Vision Transformers via Full and Packed Quantization on the Mobile

**Authors:** Peiyan Dong, LEI LU, Chao Wu, Cheng Lyu, Geng Yuan, Hao Tang, Yanzhi Wang

**<details><summary>Abstract**</summary>

While Vision Transformers (ViTs) have undoubtedly made impressive strides in computer vision (CV), their intricate network structures necessitate substantial computation and memory resources. A decision-making process for CV tasks typically entails performing computations with low latency, which is a tricky problem for ViT models.Model quantization is a widely-used technique to optimize the hardware efficiency of deep neural networks.Full quantization under Sub-8-bit precision, in particular, is a promising solution to reduce inference latency significantly. Unfortunately, current commodity hardware, such as CPUs and GPUs, still struggles to efficiently execute these sub-8-bit quantized networks, as their SIMD instructions only support a granularity of 8 bits or wider.Also, there is a scarcity of literature that presents a full quantization paradigm for ViTs.In this paper, we propose an activation-aware fully sub-8-bit quantization-aware training (QAT) framework called PackQViT for efficient yet accurate ViT acceleration on mobile devices to facilitate real-time AI-powered decision-making.Specifically, in revisiting data activation within the ViT dataflow, two characteristics are relevant to quantization strategy and precision: the long-tailed distribution and systematic channel-wise outliers.In response, we employ either log2 quantization or clipping to address the long-tailed distribution and incorporate outlier-aware training for residual link quantization to regulate the various channel-wise outliers more consistently.Notably, due to the systematic fixed pattern, outlier-aware training approach can predict the channel indices and regularized scales of outliers in advance, thus avoiding the runtime data-adaptive selection during inference.Furthermore, we employ Int-$2^{n}$-Softmax, Int-LayerNorm, and Integer GELU to enable integer-only computation flow. Finally, we develop a SIMD-based 4-bit packed multiplier to achieve end-to-end ViT acceleration on mobile phones.Compared to prior studies on ViT quantization using 8-bit precision, PackQViT surpasses other works by an improved accuracy ranging from 0.4\% to 17.9\% for various widely used ViTs on ImageNet dataset; under 4-bit precision, PackQViT demonstrates 0.4%$\sim$2.8% higher accuracy. Compared to the baseline multiplier, our implementations on the Realme GT Android smartphone with Snapdragon 870 SoC CPU achieve 2.6x$\sim$3.7x speedup under 8-bit scenario and 3.8x$\sim$5.9x speedup under 4-bit which ensures practical real-time performance.</details>

### ParaFuzz: An Interpretability-Driven Technique for Detecting Poisoned Samples in NLP

**Authors:** Lu Yan, Zhuo Zhang, Guanhong Tao, Kaiyuan Zhang, Xuan Chen, Guangyu Shen, Xiangyu Zhang

**<details><summary>Abstract**</summary>

Backdoor attacks have emerged as a prominent threat to natural language processing (NLP) models, where the presence of specific triggers in the input can lead poisoned models to misclassify these inputs to predetermined target classes. Current detection mechanisms are limited by their inability to address more covert backdoor strategies, such as style-based attacks. In this work, we propose an innovative test-time poisoned sample detection framework that hinges on the interpretability of model predictions, grounded in the semantic meaning of inputs.We contend that triggers (e.g., infrequent words) are not supposed to fundamentally alter the underlying semantic meanings of poisoned samples as they want to stay stealthy. Based on this observation, we hypothesize that while the model's predictions for paraphrased clean samples should remain stable, predictions for poisoned samples should revert to their true labels upon the mutations applied to triggers during the paraphrasing process.We employ ChatGPT, a state-of-the-art large language model, as our paraphraser and formulate the trigger-removal task as a prompt engineering problem. We adopt fuzzing, a technique commonly used for unearthing software vulnerabilities, to discover optimal paraphrase prompts that can effectively eliminate triggers while concurrently maintaining input semantics.Experiments on 4 types of backdoor attacks, including the subtle style backdoors, and 4 distinct datasets demonstrate that our approach surpasses baseline methods, including STRIP, RAP, and ONION, in precision and recall.</details>

### [Spotlight] Partial Counterfactual Identification of Continuous Outcomes with a Curvature Sensitivity Model

**Authors:** Valentyn Melnychuk, Dennis Frauen, Stefan Feuerriegel

**<details><summary>Abstract**</summary>

Counterfactual inference aims to answer retrospective "what if" questions and thus belongs to the most fine-grained type of inference in Pearl's causality ladder. Existing methods for counterfactual inference with continuous outcomes aim at point identification and thus make strong and unnatural assumptions about the underlying structural causal model. In this paper, we relax these assumptions and aim at partial counterfactual identification of continuous outcomes, i.e., when the counterfactual query resides in an ignorance interval with informative bounds. We prove that, in general, the ignorance interval of the counterfactual queries has non-informative bounds, already when functions of structural causal models are continuously differentiable. As a remedy, we propose a novel sensitivity model called Curvature Sensitivity Model. This allows us to obtain informative bounds by bounding the curvature of level sets of the functions. We further show that existing point counterfactual identification methods are special cases of our Curvature Sensitivity Model when the bound of the curvature is set to zero. We then propose an implementation of our Curvature Sensitivity Model in the form of a novel deep generative model, which we call Augmented Pseudo-Invertible Decoder. Our implementation employs (i) residual normalizing flows with (ii) variational augmentations. We empirically demonstrate the effectiveness of our Augmented Pseudo-Invertible Decoder. To the best of our knowledge, ours is the first partial identification model for Markovian structural causal models with continuous outcomes.</details>

### [Spotlight] Participatory Personalization in Classification

**Authors:** Hailey Joren, Chirag Nagpal, Katherine Heller, Berk Ustun

**<details><summary>Abstract**</summary>

Machine learning models are often personalized based on information that is protected, sensitive, self-reported, or costly to acquire. These models use information about people, but do not facilitate nor inform their *consent*. Individuals cannot opt out of reporting information that a model needs to personalize their predictions nor tell if they benefit from personalization in the first place. We introduce a new family of prediction models, called participatory systems, that let individuals opt into personalization at prediction time. We present a model-agnostic algorithm to learn participatory systems for supervised learning tasks where models are personalized with categorical group attributes. We conduct a comprehensive empirical study of participatory systems in clinical prediction tasks, comparing them to common approaches for personalization and imputation. Our results show that participatory systems can facilitate and inform consent in a way that improves performance and privacy across all groups who report personal data.</details>

### Path following algorithms for $\ell_2$-regularized $M$-estimation with approximation guarantee

**Authors:** Yunzhang Zhu, Renxiong Liu

**<details><summary>Abstract**</summary>

Many modern machine learning algorithms are formulated as regularized M-estimation problems, in which a regularization (tuning) parameter controls a trade-off between model fit to the training data and model complexity. To select the ``best'' tuning parameter value that achieves a good trade-off, an approximated solution path needs to be computed. In practice, this is often done through selecting a grid of tuning parameter values and solving the regularized problem at the selected grid points. However, given any desired level of accuracy, it is often not clear how to choose the grid points and also how accurately one should solve the regularized problems at the selected gird points, both of which can greatly impact the overall amount of computation. In the context of  $\ell_2$-regularized $M$-estimation problem, we propose a novel grid point selection scheme and an adaptive stopping criterion for any given optimization algorithm that produces an approximated solution path with approximation error guarantee. Theoretically, we prove that the proposed solution path can approximate the exact solution path to arbitrary level of accuracy, while saving the overall computation as much as possible. Numerical results also corroborate with our theoretical analysis.</details>

### Penguin: Parallel-Packed Homomorphic Encryption for Fast Graph Convolutional Network Inference

**Authors:** Ran Ran, Nuo Xu, Tao Liu, Wei Wang, Gang Quan, Wujie Wen

**<details><summary>Abstract**</summary>

he marriage of Graph Convolutional Network (GCN) and Homomorphic Encryption (HE) enables the inference of graph data on the cloud with significantly enhanced client data privacy. However, the tremendous computation and memory overhead associated with HE operations challenges the practicality of HE-based GCN inference. GCN inference involves a sequence of expensive matrix-matrix multiplications, and we observe that directly applying the state-of-the-art HE-based secure matrix-matrix multiplication solutions to accelerate HE-GCN inference is far less efficient as it does not exploit the unique aggregation mechanism of two-dimension graph node-features in GCN layer computation. As a result, in this paper, we propose a novel HE-based ciphertext packing technique, i.e., Penguin, that can take advantage of the unique computation pattern during the HE-GCN inference to significantly reduce the computation and memory overhead associated with HE operations.Specifically, Penguin employs (i) an effective two-dimension parallel packing technique for feature ciphertext with optimal graph node partitioning and graph feature interleaving, and (ii) an interleaved assembly technique that can effectively make use of the blank slots to merge ciphertexts after feature reduction and significantly reduce the costly rotation operation.We provide theoretical analysis and experimental validation to demonstrate the speedup achieved by Penguin in accelerating GCN inference using popular GCN models and datasets. Our results show that Penguin can achieve up to $\sim10\times$ speedup and around $\sim79$% reduction in computational memory overhead, significantly outperforming state-of-the-art solutions. To the best of our knowledge, this is the first work that can ensure the protection of both graph structure and features when accelerating HE-GCN inference on encrypted data. Our code is publicly available at https://github.com/ranran0523/Penguin.</details>

### Percentile Criterion Optimization in Offline Reinforcement Learning

**Authors:** Cyrus Cousins, Elita Lobo, Marek Petrik, Yair Zick

**<details><summary>Abstract**</summary>

In reinforcement learning, robust policies for high-stakes decision-making problems with limited data are usually computed by optimizing the percentile criterion. The percentile criterion is optimized by constructing an uncertainty set that contains the true model with high probability and optimizing the policy for the worst model in the set. Since the percentile criterion is non-convex, constructing these sets itself is challenging. Existing works use Bayesian credible regions as uncertainty sets, but they are often unnecessarily large and result in learning overly conservative policies. To overcome these shortcomings, we propose a novel Value-at-Risk based dynamic programming algorithm to optimize the percentile criterion without explicitly constructing any uncertainty sets. Our theoretical and empirical results show that our algorithm implicitly constructs much smaller uncertainty sets and learns less-conservative robust policies.</details>

### Persuading Farsighted Receivers in MDPs: the Power of Honesty

**Authors:** Martino Bernasconi, Matteo Castiglioni, Alberto Marchesi, Mirco Mutti

**<details><summary>Abstract**</summary>

Bayesian persuasion studies the problem faced by an informed sender who strategically discloses information to influence the behavior of an uninformed receiver. Recently, a growing attention has been devoted to settings where the sender and the receiver interact sequentially, in which the receiver's decision-making problem is usually modeled as a Markov decision process (MDP). However, the literature focuses on computing optimal information-revelation policies (a.k.a. signaling schemes) under the restrictive assumption that the receiver acts myopically, selecting actions to maximize the one-step utility and disregarding future rewards. This is justified by the fact that, when the receiver is farsighted and thus considers future rewards, finding an optimal Markovian signaling scheme is NP-hard. In this paper, we show that Markovian signaling schemes do not constitute the "right" class of policies. Indeed, differently from most of the MDPs settings, we show that Markovian signaling schemes are not optimal, and general history-dependent signaling schemes should be considered. Moreover, we also show that history-dependent signaling schemes circumvent the negative complexity results affecting Markovian signaling schemes. Formally, we design an algorithm that computes an optimal and $\epsilon$-persuasive history-dependent signaling scheme in time polynomial in ${1}/{\epsilon}$ and in the instance size. The crucial challenge is that general history-dependent signaling schemes cannot be represented in polynomial space. Nevertheless, we introduce a convenient subclass of history-dependent signaling schemes, called promise-form, which are as powerful as general history-dependent ones and efficiently representable. Intuitively, promise-form signaling schemes compactly encode histories in the form of honest promises on future receiver's rewards.</details>

### [Spotlight] Plug-and-Play Stability for Intracortical Brain-Computer Interfaces: A One-Year Demonstration of Seamless Brain-to-Text Communication

**Authors:** Chaofei Fan, Nick Hahn, Foram Kamdar, Donald Avansino, Guy Wilson, Leigh Hochberg, Krishna V Shenoy, Jaimie Henderson, Francis Willett

**<details><summary>Abstract**</summary>

Intracortical brain-computer interfaces (iBCIs) have shown promise for restoring rapid communication to people with neurological disorders such as amyotrophic lateral sclerosis (ALS). However, to maintain high performance over time, iBCIs typically need frequent recalibration to combat changes in the neural recordings that accrue over days. This requires iBCI users to stop using the iBCI and engage in supervised data collection, making the iBCI system hard to use. In this paper, we propose a method that enables self-recalibration of communication iBCIs without interrupting the user. Our method leverages large language models (LMs) to automatically correct errors in iBCI outputs. The self-recalibration process uses these corrected outputs ("pseudo-labels") to continually update the iBCI decoder online. Over a period of more than one year (403 days), we evaluated our Continual Online Recalibration with Pseudo-labels (CORP) framework with one clinical trial participant. CORP achieved  a stable decoding accuracy of 93.84% in an online handwriting iBCI task, significantly outperforming other baseline methods. Notably, this is the longest-running iBCI stability demonstration involving a human participant. Our results provide the first  evidence for long-term stabilization of a plug-and-play, high-performance communication iBCI, addressing a major barrier for the clinical translation of iBCIs.</details>

### Policy Optimization for Continuous Reinforcement Learning

**Authors:** HANYANG ZHAO, Wenpin Tang, David Yao

**<details><summary>Abstract**</summary>

We study reinforcement learning (RL) in the setting of continuous time and space, for an infinite horizon with a discounted objective and the underlying dynamics driven by a stochastic differential equation. Built upon recent advances in the continuous approach to RL, we develop a notion of occupation time (specifically for a discounted objective),  and show how it can be effectively used to derive performance difference and local approximation formulas. We further extend these results to illustrate their applications in the PG (policy gradient) and TRPO/PPO (trust region policy optimization/ proximal policy optimization) methods,  which have been familiar and powerful tools in the discrete RL setting but under-developed in continuous RL. Through numerical experiments, we demonstrate the effectiveness and advantages of our approach.</details>

### Policy Optimization in a Noisy Neighborhood: On Return Landscapes in Continuous Control

**Authors:** Nate Rahn, Pierluca D'Oro, Harley Wiltzer, Pierre-Luc Bacon, Marc Bellemare

**<details><summary>Abstract**</summary>

Deep reinforcement learning agents for continuous control are known to exhibit significant instability in their performance over time. In this work, we provide a fresh perspective on these behaviors by studying the return landscape: the mapping between a policy and a return. We find that popular algorithms traverse noisy neighborhoods of this landscape, in which a single update to the policy parameters leads to a wide range of returns. By taking a distributional view of these returns, we map the landscape, characterizing failure-prone regions of policy space and revealing a hidden dimension of policy quality. We show that the landscape exhibits surprising structure by finding simple paths in parameter space which improve the stability of a policy. To conclude, we develop a distribution-aware procedure which finds such paths, navigating away from noisy neighborhoods in order to improve the robustness of a policy. Taken together, our results provide new insight into the optimization, evaluation, and design of agents.</details>

### Policy Space Diversity for Non-Transitive Games

**Authors:** Jian Yao, Weiming Liu, Haobo Fu, Yaodong Yang, Stephen McAleer, Qiang Fu, Wei Yang

**<details><summary>Abstract**</summary>

Policy-Space Response Oracles (PSRO) is an influential algorithm framework for approximating a Nash Equilibrium (NE) in multi-agent non-transitive games. Many previous studies have been trying to promote policy diversity in PSRO. A major weakness with existing diversity metrics is that a more diverse (according to their diversity metrics) population does not necessarily mean (as we proved in the paper) a better approximation to a NE. To alleviate this problem, we propose a new diversity metric, the improvement of which guarantees a better approximation to a NE. Meanwhile, we develop a practical and well-justified method to optimize our diversity metric using only state-action samples. By incorporating our diversity regularization into the best response solving of PSRO, we obtain a new PSRO variant, \textit{Policy Space Diversity} PSRO (PSD-PSRO). We present the convergence property of PSD-PSRO. Empirically, extensive experiments on single-state games, Leduc, and Goofspiel demonstrate that PSD-PSRO is more effective in producing significantly less exploitable policies than state-of-the-art PSRO variants.</details>

### Polynomially Over-Parameterized Convolutional Neural Networks Contain Structured Strong Winning Lottery Tickets

**Authors:** Arthur da Cunha, Francesco D'Amore, Natale

**<details><summary>Abstract**</summary>

The Strong Lottery Ticket Hypothesis (SLTH) states that randomly-initialised neural networks likely contain subnetworks that perform well without any training. Although unstructured pruning has been extensively studied in this context, its structured counterpart, which can deliver significant computational and memory efficiency gains, has been largely unexplored. One of the main reasons for this gap is the limitations of the underlying mathematical tools used in formal analyses of the SLTH.In this paper, we overcome these limitations: we leverage recent advances in the multidimensional generalisation of the Random Subset-Sum Problem and obtain a variant that admits the stochastic dependencies that arise when addressing structured pruning in the SLTH. We apply this result to prove, for a wide class of random Convolutional Neural Networks, the existence of structured subnetworks that can approximate any sufficiently smaller network.This result provides the first sub-exponential bound around the SLTH for structured pruning, opening up new avenues for further research on the hypothesis and contributing to the understanding of the role of over-parameterization in deep learning.</details>

### PopSign ASL v1.0: An Isolated American Sign Language Dataset Collected via Smartphones

**Authors:** Thad Starner, Sean Forbes, Matthew So, David Martin, Rohit Sridhar, Gururaj Deshpande, Sam Sepah, Sahir Shahryar, Khushi Bhardwaj, Tyler Kwok, Daksh Sehgal, Saad Hassan, Bill Neubauer, Sofia Vempala, Alec Tan, Jocelyn Heath, Unnathi Kumar, Priyanka Mosur, Tavenner Hall, Rajandeep Singh, Christopher Cui, Glenn Cameron, Sohier Dane, Garrett Tanzer

**<details><summary>Abstract**</summary>

PopSign is a smartphone-based bubble-shooter game that helps hearing parentsof deaf infants learn sign language. To help parents practice their ability to sign,PopSign is integrating sign language recognition as part of its gameplay. Fortraining the recognizer, we introduce the PopSign ASL v1.0 dataset that collectsexamples of 250 isolated American Sign Language (ASL) signs using Pixel 4Asmartphone selfie cameras in a variety of environments. It is the largest publiclyavailable, isolated sign dataset by number of examples and is the first dataset tofocus on one-handed, smartphone signs. We collected over 210,000 examplesat 1944x2592 resolution made by 47 consenting Deaf adult signers for whomAmerican Sign Language is their primary language. We manually reviewed 217,866of these examples, of which 175,023 (approximately 700 per sign) were the signintended for the educational game. 39,304 examples were recognizable as a signbut were not the desired variant or were a different sign. We provide a training setof 31 signers, a validation set of eight signers, and a test set of eight signers. Abaseline LSTM model for the 250-sign vocabulary achieves 82.1% accuracy (81.9%class-weighted F1 score) on the validation set and 84.2% (83.9% class-weightedF1 score) on the test set. Gameplay suggests that accuracy will be sufficient forcreating educational games involving sign language recognition.</details>

### PrObeD: Proactive Object Detection Wrapper

**Authors:** Vishal Asnani, Abhinav Kumar, Suya You, Xiaoming Liu

**<details><summary>Abstract**</summary>

Previous research in $2D$ object detection focuses on various tasks, including detecting objects in generic and camouflaged images. These works are regarded as passive works for object detection as they take the input image as is. However, convergence to global minima is not guaranteed to be optimal in neural networks; therefore, we argue that the trained weights in the object detector are not optimal. To rectify this problem, we propose a wrapper based on proactive schemes, PrObeD, which enhances the performance of these object detectors by learning a signal. PrObeD consists of an encoder-decoder architecture, where the encoder network generates an image-dependent signal termed templates to encrypt the input images, and the decoder recovers this template from the encrypted images. We propose that learning the optimum template results in an object detector with an improved detection performance. The template acts as a mask to the input images to highlight semantics useful for the object detector. Finetuning the object detector with these encrypted images enhances the detection performance for both generic and camouflaged. Our experiments on MS-COCO, CAMO, COD$10$K, and NC$4$K datasets show improvement over different detectors after applying PrObeD. Our models/codes are available at https://github.com/vishal3477/Proactive-Object-Detection.</details>

### Practical Contextual Bandits with Feedback Graphs

**Authors:** Mengxiao Zhang, Yuheng Zhang, Olga Vrousgou, Haipeng Luo, Paul Mineiro

**<details><summary>Abstract**</summary>

While contextual bandit has a mature theory, effectively leveraging different feedback patterns to enhance the pace of learning remains unclear. Bandits with feedback graphs, which interpolates between the full information and bandit regimes, provides a promising framework to mitigate the statistical complexity of learning. In this paper, we propose and analyze an approach to contextual bandits with feedback graphs based upon reduction to regression.  The resulting algorithms are computationally practical and achieve established minimax rates, thereby reducing the statistical complexity in real-world applications.</details>

### [Spotlight] Practical Sharpness-Aware Minimization Cannot Converge All the Way to Optima

**Authors:** Dongkuk Si, Chulhee Yun

**<details><summary>Abstract**</summary>

Sharpness-Aware Minimization (SAM) is an optimizer that takes a descent step based on the gradient at a perturbation $y_t = x_t + \rho \frac{\nabla f(x_t)}{\lVert \nabla f(x_t) \rVert}$ of the current point $x_t$. Existing studies prove convergence of SAM for smooth functions, but they do so by assuming decaying perturbation size $\rho$ and/or no gradient normalization in $y_t$, which is detached from practice. To address this gap, we study deterministic/stochastic versions of SAM with practical configurations (i.e., constant $\rho$ and gradient normalization in $y_t$) and explore their convergence properties on smooth functions with (non)convexity assumptions.Perhaps surprisingly, in many scenarios, we find out that SAM has limited capability to converge to global minima or stationary points.For smooth strongly convex functions, we show that while deterministic SAM enjoys tight global convergence rates of $\tilde \Theta(\frac{1}{T^2})$, the convergence bound of stochastic SAM suffers an inevitable additive term $\mathcal O(\rho^2)$, indicating convergence only up to neighborhoods of optima.In fact, such $\mathcal O(\rho^2)$ factors arise for stochastic SAM in all the settings we consider, and also for deterministic SAM in nonconvex cases; importantly, we prove by examples that such terms are unavoidable.Our results highlight vastly different characteristics of SAM with vs. without decaying perturbation size or gradient normalization, and suggest that the intuitions gained from one version may not apply to the other.</details>

### [Spotlight] Pre-Training Protein Encoder via Siamese Sequence-Structure Diffusion Trajectory Prediction

**Authors:** Zuobai Zhang, Minghao Xu, Aurelie Lozano, Vijil Chenthamarakshan, Payel Das, Jian Tang

**<details><summary>Abstract**</summary>

Self-supervised pre-training methods on proteins have recently gained attention, with most approaches focusing on either protein sequences or structures, neglecting the exploration of their joint distribution, which is crucial for a comprehensive understanding of protein functions by integrating co-evolutionary information and structural characteristics. In this work, inspired by the success of denoising diffusion models in generative tasks, we propose the DiffPreT approach to pre-train a protein encoder by sequence-structure joint diffusion modeling. DiffPreT guides the encoder to recover the native protein sequences and structures from the perturbed ones along the joint diffusion trajectory, which acquires the joint distribution of sequences and structures. Considering the essential protein conformational variations, we enhance DiffPreT by a method called Siamese Diffusion Trajectory Prediction (SiamDiff) to capture the correlation between different conformers of a protein. SiamDiff attains this goal by maximizing the mutual information between representations of diffusion trajectories of structurally-correlated conformers. We study the effectiveness of DiffPreT and SiamDiff on both atom- and residue-level structure-based protein understanding tasks. Experimental results show that the performance of DiffPreT is consistently competitive on all tasks, and SiamDiff achieves new state-of-the-art performance, considering the mean ranks on all tasks. Code will be released upon acceptance.</details>

### [Spotlight] Precise asymptotic generalization for multiclass classification with overparameterized linear models

**Authors:** David Wu, Anant Sahai

**<details><summary>Abstract**</summary>

We study the asymptotic generalization of an overparameterized linear model for multiclass classification under the Gaussian covariates bi-level model introduced in Subramanian et al. (NeurIPS'22), where the number of data points, features, and classes all grow together. We fully resolve the conjecture posed in Subramanian et al. '22, matching the predicted regimes for which the model does and does not generalize. Furthermore, our new lower bounds are akin to an information-theoretic strong converse: they establish that the misclassification rate goes to 0 or 1 asymptotically. One surprising consequence of our tight results is that the min-norm interpolating classifier can be asymptotically suboptimal relative to noninterpolating classifiers in the regime where the min-norm interpolating regressor is known to be optimal. The key to our tight analysis is a new variant of the Hanson-Wright inequality which is broadly useful for multiclass problems with sparse labels. As an application, we show that the same type of analysis can be used to analyze the related multi-label classification problem under the same bi-level ensemble.</details>

### Precision-Recall Divergence Optimization for Generative Modeling with GANs and Normalizing Flows

**Authors:** Alexandre Verine, Benjamin Negrevergne, Muni Sreenivas Pydi, Yann Chevaleyre

**<details><summary>Abstract**</summary>

Achieving a balance between image quality (precision) and diversity (recall) is a significant challenge in the domain of generative models. Current state-of-the-art models primarily rely on optimizing heuristics, such as the Fr\'echet Inception Distance. While recent developments have introduced principled methods for evaluating precision and recall, they have yet to be successfully integrated into the training of generative models. Our main contribution is a novel training method for generative models, such as Generative Adversarial Networks and Normalizing Flows, which explicitly optimizes a user-defined trade-off between precision and recall.  More precisely, we show that achieving a specified precision-recall trade-off corresponds to minimizing a unique $f$-divergence from a family we call the \mbox{\em PR-divergences}. Conversely, any $f$-divergence can be written as a linear combination of PR-divergences and  corresponds to a weighted precision-recall trade-off. Through comprehensive evaluations, we show that our approach improves the performance of existing state-of-the-art models like BigGAN in terms of either precision or recall when tested on datasets such as ImageNet.</details>

### Predicting mutational effects on protein-protein binding via a side-chain diffusion probabilistic model

**Authors:** Shiwei Liu, Tian Zhu, Milong Ren, Chungong Yu, Dongbo Bu, Haicang Zhang

**<details><summary>Abstract**</summary>

Many crucial biological processes rely on networks of protein-protein interactions. Predicting the effect of amino acid mutations on protein-protein binding is important in protein engineering, including therapeutic discovery. However, the scarcity of annotated experimental data on binding energy poses a significant challenge for developing computational approaches, particularly deep learning-based methods. In this work, we propose SidechainDiff, a novel representation learning-based approach that leverages unlabelled experimental protein structures. SidechainDiff utilizes a Riemannian diffusion model to learn the generative process of side-chain conformations and can also give the structural context representations of mutations on the protein-protein interface. Leveraging the learned representations, we achieve state-of-the-art performance in predicting the mutational effects on protein-protein binding. Furthermore, SidechainDiff is the first diffusion-based generative model for side-chains, distinguishing it from prior efforts that have predominantly focused on the generation of protein backbone structures.</details>

### Primal-Attention: Self-attention through Asymmetric Kernel SVD in Primal Representation

**Authors:** Yingyi Chen, Qinghua Tao, Francesco Tonin, Johan Suykens

**<details><summary>Abstract**</summary>

Recently, a new line of works has emerged to understand and improve self-attention in Transformers by treating it as a kernel machine. However, existing works apply the methods for symmetric kernels to the asymmetric self-attention, resulting in a nontrivial gap between the analytical understanding and numerical implementation. In this paper, we provide a new perspective to represent and optimize self-attention through asymmetric Kernel Singular Value Decomposition (KSVD), which is also motivated by the low-rank property of self-attention normally observed in deep layers. Through asymmetric KSVD, i) a primal-dual representation of self-attention is formulated, where the optimization objective is cast to maximize the projection variances in the attention outputs; ii) a novel attention mechanism, i.e., Primal-Attention, is proposed via the primal representation of KSVD, avoiding explicit computation of the kernel matrix in the dual; iii) with KKT conditions, we prove that the stationary solution to the KSVD optimization in Primal-Attention yields a zero-value objective. In this manner, KSVD optimization can be implemented by simply minimizing a regularization loss, so that low-rank property is promoted without extra decomposition. Numerical experiments show state-of-the-art performance of our Primal-Attention with improved efficiency. Moreover, we demonstrate that the deployed KSVD optimization regularizes Primal-Attention with a sharper singular value decay than that of the canonical self-attention, further verifying the great potential of our method. To the best of our knowledge, this is the first work that provides a primal-dual representation for the asymmetric kernel in self-attention and successfully applies it to modelling and optimization.</details>

### ProBio: A Protocol-guided Multimodal Dataset for Molecular Biology Lab

**Authors:** Jieming Cui, Ziren Gong, Baoxiong Jia, Siyuan Huang, Zilong Zheng, Jianzhu Ma, Yixin Zhu

**<details><summary>Abstract**</summary>

The challenge of replicating research results has posed a significant impediment to the field of molecular biology. The advent of modern intelligent systems has led to notable progress in various domains. Consequently, we embarked on an investigation of intelligent monitoring systems as a means of tackling the issue of the reproducibility crisis. Specifically, we first curate a comprehensive multimodal dataset, named ProBio, as an initial step towards this objective. This dataset comprises fine-grained hierarchical annotations intended for the purpose of studying activity understanding in BioLab. Next, we devise two challenging benchmarks, transparent solution tracking and multimodal action recognition, to emphasize the unique characteristics and difficulties associated with activity understanding in BioLab settings. Finally, we provide a thorough experimental evaluation of contemporary video understanding models and highlight their limitations in this specialized domain to identify potential avenues for future research. We hope \dataset with associated benchmarks may garner increased focus on modern AI techniques in the realm of molecular biology.</details>

### [Spotlight] ProPILE: Probing Privacy Leakage in Large Language Models

**Authors:** Siwon Kim, Sangdoo Yun, Hwaran Lee, Martin Gubri, Sungroh Yoon, Seong Joon Oh

**<details><summary>Abstract**</summary>

The rapid advancement and widespread use of large language models (LLMs) have raised significant concerns regarding the potential leakage of personally identifiable information (PII). These models are often trained on vast quantities of web-collected data, which may inadvertently include sensitive personal data. This paper presents ProPILE, a novel probing tool designed to empower data subjects, or the owners of the PII, with awareness of potential PII leakage in LLM-based services. ProPILE lets data subjects formulate prompts based on their own PII to evaluate the level of privacy intrusion in LLMs. We demonstrate its application on the OPT-1.3B model trained on the publicly available Pile dataset. We show how hypothetical data subjects may assess the likelihood of their PII being included in the Pile dataset being revealed. ProPILE can also be leveraged by LLM service providers to effectively evaluate their own levels of PII leakage with more powerful prompts specifically tuned for their in-house models. This tool represents a pioneering step towards empowering the data subjects for their awareness and control over their own data on the web.</details>

### Probabilistic Exponential Integrators

**Authors:** Nathanael Bosch, Philipp Hennig, Filip Tronarp

**<details><summary>Abstract**</summary>

Probabilistic solvers provide a flexible and efficient framework for simulation, uncertainty quantification, and inference in dynamical systems. However, like standard solvers, they suffer performance penalties for certain stiff systems, where small steps are required not for reasons of numerical accuracy but for the sake of stability. This issue is greatly alleviated in semi-linear problems by the probabilistic exponential integrators developed in this paper. By including the fast, linear dynamics in the prior, we arrive at a class of probabilistic integrators with favorable properties. Namely, they are proven to be L-stable, and in a certain case reduce to a classic exponential integrator---with the added benefit of providing a probabilistic account of the numerical error. The method is also generalized to arbitrary non-linear systems by imposing piece-wise semi-linearity on the prior via Jacobians of the vector field at the previous estimates, resulting in probabilistic exponential Rosenbrock methods. We evaluate the proposed methods on multiple stiff differential equations and demonstrate their improved stability and efficiency over established probabilistic solvers. The present contribution thus expands the range of problems that can be effectively tackled within probabilistic numerics.</details>

### Probabilistic Inference in Reinforcement Learning Done Right

**Authors:** Jean Tarbouriech, Tor Lattimore, Brendan O'Donoghue

**<details><summary>Abstract**</summary>

A popular perspective in Reinforcement learning (RL) casts the problem as probabilistic inference on a graphical model of the Markov decision process (MDP). The core object of study is the probability of each state-action pair being visited under the optimal policy. Previous approaches to approximate this quantity can be arbitrarily poor, leading to algorithms that do not implement genuine statistical inference and consequently do not perform well in challenging problems. In this work, we undertake a rigorous Bayesian treatment of the posterior probability of state-action optimality and clarify how it flows through the MDP. We first reveal that this quantity can indeed be used to generate a policy that explores efficiently, as measured by regret. Unfortunately, computing it is intractable, so we derive a new variational Bayesian approximation yielding a tractable convex optimization problem and establish that the resulting policy also explores efficiently. We call our approach VAPOR and show that it has strong connections to Thompson sampling, K-learning, and maximum entropy exploration. We conclude with some experiments demonstrating the performance advantage of a deep RL version of VAPOR.</details>

### Projection-Free Methods for Stochastic Simple Bilevel Optimization with Convex Lower-level Problem

**Authors:** Jincheng Cao, Ruichen Jiang, Nazanin Abolfazli, Erfan Yazdandoost Hamedani, Aryan Mokhtari

**<details><summary>Abstract**</summary>

In this paper, we study a class of stochastic bilevel optimization problems, also known as stochastic simple bilevel optimization, where we minimize a smooth stochastic objective function over the optimal solution set of another stochastic convex optimization problem. We introduce novel stochastic bilevel optimization methods that locally approximate the solution set of the lower-level problem via a stochastic cutting plane, and then run a conditional gradient update with variance reduction techniques to control the error induced by using stochastic gradients. For the case that the upper-level function is convex, our method requires $\mathcal{O}(\max\\{1/\epsilon_f^{2},1/\epsilon_g^{2}\\}) $ stochastic oracle queries to obtain a solution that is $\epsilon_f$-optimal for the upper-level and $\epsilon_g$-optimal for the lower-level. This guarantee improves the previous best-known complexity of $\mathcal{O}(\max\\{1/\epsilon_f^{4},1/\epsilon_g^{4}\\})$. Moreover, for the case that the upper-level function is non-convex, our method requires at most $\mathcal{O}(\max\\{1/\epsilon_f^{3},1/\epsilon_g^{3}\\}) $ stochastic oracle queries to find an $(\epsilon_f, \epsilon_g)$-stationary point. In the finite-sum setting, we show that the number of stochastic oracle calls required by our method are  $\mathcal{O}(\sqrt{n}/\epsilon)$ and $\mathcal{O}(\sqrt{n}/\epsilon^{2})$ for the convex and non-convex settings, respectively, where $\epsilon=\min \\{\epsilon_f,\epsilon_g\\}$.</details>

### ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design

**Authors:** Pascal Notin, Aaron Kollasch, Daniel Ritter, Lood van Niekerk, Steffanie Paul, Han Spinner, Nathan Rollins, Ada Shaw, Rose Orenbuch, Ruben Weitzman, Jonathan Frazer, Mafalda Dias, Dinko Franceschi, Yarin Gal, Debora Marks

**<details><summary>Abstract**</summary>

Predicting the effects of mutations in proteins is critical to many applications, from understanding genetic disease to designing novel proteins to address our most pressing challenges in climate, agriculture and healthcare. Despite an increase in machine learning-based protein modeling methods, assessing their effectiveness is problematic due to the use of distinct, often contrived, experimental datasets and variable performance across different protein families. Addressing these challenges requires scale. To that end we introduce ProteinGym v1.0, a large-scale and holistic set of benchmarks specifically designed for protein fitness prediction and design. It encompasses both a broad collection of over 250 standardized deep mutational scanning assays, spanning millions of mutated sequences, as well as curated clinical datasets providing high-quality expert annotations about mutation effects. We devise a robust evaluation framework that combines metrics for both fitness prediction and design, factors in known limitations of the underlying experimental methods, and covers both zero-shot and supervised settings. We report the performance of a diverse set of over 40 high-performing models from various subfields (eg., mutation effects, inverse folding) into a unified benchmark. We open source the corresponding codebase, datasets, MSAs, structures, predictions and develop a user-friendly website that facilitates comparisons across all settings.</details>

### Provable Guarantees for Neural Networks via Gradient Feature Learning

**Authors:** Zhenmei Shi, Junyi Wei, Yingyu Liang

**<details><summary>Abstract**</summary>

Neural networks have achieved remarkable empirical performance, while the current theoretical analysis is not adequate for understanding their success, e.g., the Neural Tangent Kernel approach fails to capture their key feature learning ability, while recent analyses on feature learning are typically problem-specific. This work proposes a unified analysis framework for two-layer networks trained by gradient descent. The framework is centered around the principle of feature learning from gradients, and its effectiveness is demonstrated by applications in several prototypical problems, such as mixtures of Gaussians and parity functions.The framework also sheds light on interesting network learning phenomena such as feature learning beyond kernels and the lottery ticket hypothesis.</details>

### [Spotlight] Provable Guarantees for Nonlinear Feature Learning in Three-Layer Neural Networks

**Authors:** Eshaan Nichani, Alex Damian, Jason Lee

**<details><summary>Abstract**</summary>

One of the central questions in the theory of deep learning is to understand how neural networks learn hierarchical features. The ability of deep networks to extract salient features is crucial to both their outstanding generalization ability and the modern deep learning paradigm of pretraining and finetuneing. However, this feature learning process remains poorly understood from a theoretical perspective, with existing analyses largely restricted to two-layer networks. In this work we show that three-layer neural networks have provably richer feature learning capabilities than two-layer networks. We analyze the features learned by a three-layer network trained with layer-wise gradient descent, and present a general purpose theorem which upper bounds the sample complexity and width needed to achieve low test error when the target has specific hierarchical structure. We instantiate our framework in specific statistical learning settings -- single-index models and functions of quadratic features -- and show that in the latter setting three-layer networks obtain a sample complexity improvement over all existing guarantees for two-layer networks. Crucially, this sample complexity improvement relies on the ability of three-layer networks to efficiently learn *nonlinear* features. We then establish a concrete optimization-based depth separation by constructing a function which is efficiently learnable via gradient descent on a three-layer network, yet cannot be learned efficiently by a two-layer network. Our work makes progress towards understanding the provable benefit of three-layer neural networks over two-layer networks in the feature learning regime.</details>

### [Spotlight] Provable benefits of annealing for estimating normalizing constants: Importance Sampling, Noise-Contrastive Estimation, and beyond

**Authors:** Omar Chehab, Aapo Hyvarinen, Andrej Risteski

**<details><summary>Abstract**</summary>

Recent research has developed several Monte Carlo methods for estimating the normalization constant (partition function) based on the idea of annealing. This means sampling successively from a path of distributions which interpolate between a tractable "proposal" distribution and the unnormalized "target" distribution. Prominent estimators in this family include annealed importance sampling and annealed noise-contrastive estimation (NCE). Such methods hinge on a number of design choices: which estimator to use, which path of distributions to use and whether to use a path at all; so far, there is no definitive theory on which choices are efficient. Here, we evaluate each design choice by the asymptotic estimation error it produces. First, we show that using NCE is more efficient than the importance sampling estimator, but in the limit of infinitesimal path steps, the difference vanishes. Second, we find that using the geometric path brings down the estimation error from an exponential to a polynomial function of the parameter distance between the target and proposal distributions. Third, we find that the arithmetic path, while rarely used, can offer optimality properties over the universally-used geometric path. In fact, in a particular limit, the optimal path is arithmetic. Based on this theory, we finally propose a two-step estimator to approximate the optimal path in an efficient way.</details>

### Provably (More) Sample-Efficient Offline RL with Options

**Authors:** Xiaoyan Hu, Ho-fung Leung

**<details><summary>Abstract**</summary>

The options framework yields empirical success in long-horizon planning problems of reinforcement learning (RL). Recent works show that options help improve the sample efficiency in online RL. However, these results are no longer applicable to scenarios where exploring the environment online is risky, e.g., automated driving and healthcare. In this paper, we provide the first analysis of the sample complexity for offline RL with options, where the agent learns from a dataset without further interaction with the environment. We derive a novel information-theoretic lower bound, which generalizes the one for offline learning with actions. We propose the PEssimistic Value Iteration for Learning with Options (PEVIO) algorithm and establish near-optimal suboptimality bounds for two popular data-collection procedures, where the first one collects state-option transitions and the second one collects state-action transitions. We show that compared to offline RL with actions, using options not only enjoys a faster finite-time convergence rate (to the optimal value) but also attains a better performance when either the options are carefully designed or the offline data is limited. Based on these results, we analyze the pros and cons of the data-collection procedures.</details>

### Provably Efficient Algorithm for Nonstationary Low-Rank MDPs

**Authors:** Yuan Cheng, Jing Yang, Yingbin Liang

**<details><summary>Abstract**</summary>

Reinforcement learning (RL) under changing environment models many real-world applications via nonstationary Markov Decision Processes (MDPs), and hence gains considerable interest. However, theoretical studies on nonstationary MDPs in the literature have mainly focused on tabular and linear (mixture) MDPs, which do not capture the nature of unknown representation in deep RL. In this paper, we make the first effort to investigate nonstationary RL under episodic low-rank MDPs, where both transition kernels and rewards may vary over time, and the low-rank model contains unknown representation in addition to the linear state embedding function. We first propose a parameter-dependent policy optimization algorithm called PORTAL,and further improve PORTAL to its parameter-free version of Ada-PORTAL, which is able to tune its hyper-parameters adaptively without any prior knowledge of nonstationarity. For both algorithms, we provide upper bounds on the average dynamic suboptimality gap, which show that as long as the nonstationarity is not significantly large, PORTAL and Ada-PORTAL are sample-efficient and can achieve arbitrarily small average dynamic suboptimality gap with polynomial sample complexity.</details>

### Provably Efficient Offline Reinforcement Learning in Regular Decision Processes

**Authors:** Roberto Cipollone, Anders Jonsson, Alessandro Ronca, Mohammad Sadegh Talebi

**<details><summary>Abstract**</summary>

This paper deals with offline (or batch) Reinforcement Learning (RL) in episodic Regular Decision Processes (RDPs). RDPs are the subclass of Non-Markov Decision Processes where the dependency on the history of past events can be captured by a finite-state automaton. We consider a setting where the automaton that underlies the RDP is unknown, and a learner strives to learn a near-optimal policy using pre-collected data, in the form of non-Markov sequences of observations, without further exploration. We present RegORL, an algorithm that suitably combines automata learning techniques and state-of-the-art algorithms for offline RL in MDPs. RegORL has a modular design allowing one to use any off-the-shelf offline RL algorithm in MDPs. We report a non-asymptotic high-probability sample complexity bound for RegORL to yield an $\varepsilon$-optimal policy, which makes appear a notion of concentrability relevant for RDPs. Furthermore, we present a sample complexity lower bound for offline RL in RDPs. To our best knowledge, this is the first work presenting a provably efficient algorithm for offline learning in RDPs.</details>

### Provably Safe Reinforcement Learning with Step-wise Violation Constraints

**Authors:** Nuoya Xiong, Yihan Du, Longbo Huang

**<details><summary>Abstract**</summary>

We investigate a novel safe reinforcement learning problem with step-wise violation constraints. Our problem differs from existing works in that we focus on stricter step-wise violation constraints and do not assume the existence of safe actions, making our formulation more suitable for safety-critical applications that need to ensure safety in all decision steps but may not always possess safe actions, e.g., robot control and autonomous driving.We propose an efficient algorithm SUCBVI, which guarantees $\widetilde{\mathcal{O}}(\sqrt{ST})$ or gap-dependent $\widetilde{\mathcal{O}}(S/\mathcal{C}_{\mathrm{gap}} + S^2AH^2)$ step-wise violation and $\widetilde{\mathcal{O}}(\sqrt{H^3SAT})$ regret. Lower bounds are provided to validate the optimality in both violation and  regret performance with respect to the number of states $S$ and the total number of steps $T$. Moreover, we further study an innovative safe reward-free exploration problem with step-wise violation constraints. For this problem, we design algorithm SRF-UCRL to find a near-optimal safe policy, which achieves nearly state-of-the-art  sample complexity $\widetilde{\mathcal{O}}((\frac{S^2AH^2}{\varepsilon}+\frac{H^4SA}{\varepsilon^2})(\log(\frac{1}{\delta})+S))$, and guarantees $\widetilde{\mathcal{O}}(\sqrt{ST})$ violation during exploration.  Experimental results demonstrate the  superiority of our algorithms in safety performance and corroborate our theoretical results.</details>

### Punctuation-level Attack: Single-shot and Single Punctuation Can Fool Text Models

**Authors:** wenqiang wang, Chongyang Du, Tao Wang, Kaihao Zhang, Wenhan Luo, Lin Ma, Wei Liu, Xiaochun Cao

**<details><summary>Abstract**</summary>

The adversarial attacks have attracted increasing attention in various fields including natural language processing. The current textual attacking models primarily focus on fooling models by adding character-/word-/sentence-level perturbations, ignoring their influence on human perception. In this paper, for the first time in the community, we propose a novel mode of textual attack, punctuation-level attack. With various types of perturbations, including insertion, displacement, deletion, and replacement, the punctuation-level attack achieves promising fooling rates against SOTA models on typical textual tasks and maintains minimal influence on human perception and understanding of the text by mere perturbation of single-shot single punctuation. Furthermore, we propose a search method named Text Position Punctuation Embedding and Paraphrase (TPPEP) to accelerate the pursuit of optimal position to deploy the attack, without exhaustive search, and we present a mathematical interpretation of TPPEP. Thanks to the integrated Text Position Punctuation Embedding (TPPE), the punctuation attack can be applied at a constant cost of time. Experimental results on public datasets and SOTA models demonstrate the effectiveness of the punctuation attack and the proposed TPPE. We additionally apply the single punctuation attack to summarization, semantic-similarity-scoring, and text-to-image tasks, and achieve encouraging results.</details>

### Quantification of Uncertainty with Adversarial Models

**Authors:** Kajetan Schweighofer, Lukas Aichberger, Mykyta Ielanskyi, Günter Klambauer, Sepp Hochreiter

**<details><summary>Abstract**</summary>

Quantifying uncertainty is important for actionable predictions in real-world applications. A crucial part of predictive uncertainty quantification is the estimation of epistemic uncertainty, which is defined as an integral of the product between a divergence function and the posterior. Current methods such as Deep Ensembles or MC dropout underperform at estimating the epistemic uncertainty, since they primarily consider the posterior when sampling models. We suggest Quantification of Uncertainty with Adversarial Models (QUAM) to better estimate the epistemic uncertainty. QUAM identifies regions where the whole product under the integral is large, not just the posterior. Consequently, QUAM has lower approximation error of the epistemic uncertainty compared to previous methods. Models for which the product is large correspond to adversarial models (not adversarial examples!). Adversarial models have both a high posterior as well as a high divergence between their predictions and that of a reference model. Our experiments show that QUAM excels in capturing epistemic uncertainty for deep learning models and outperforms previous methods on challenging tasks in the vision domain.</details>

### Query-based Temporal Fusion with Explicit Motion for 3D Object Detection

**Authors:** Jinghua Hou, Zhe Liu, dingkang liang, Zhikang Zou, Xiaoqing Ye, Xiang Bai

**<details><summary>Abstract**</summary>

Effectively utilizing temporal information to improve 3D detection performance is vital for autonomous driving vehicles. Existing methods either conduct temporal fusion based on the dense BEV features or sparse 3D proposal features. However, the former does not pay more attention to foreground objects, leading to more computation costs and sub-optimal performance. The latter implements time-consuming operations to generate sparse 3D proposal features, and the performance is limited by the quality of 3D proposals. In this paper, we propose a simple and effective Query-based Temporal Fusion Network (QTNet). The main idea is to exploit the object queries in previous frames to enhance the representation of current object queries by the proposed Motion-guided Temporal Modeling (MTM) module, which utilizes the spatial position information of object queries along the temporal dimension to construct their relevance between adjacent frames reliably. Experimental results show our proposed QTNet outperforms BEV-based or proposal-based manners on the nuScenes dataset. Besides, the MTM is a plug-and-play module, which can be integrated into some advanced LiDAR-only or multi-modality 3D detectors and even brings new SOTA performance with negligible computation cost and latency on the nuScenes dataset. These experiments powerfully illustrate the superiority and generalization of our method. The code is available at https://github.com/AlmoonYsl/QTNet.</details>

### RD-Suite: A Benchmark for Ranking Distillation

**Authors:** Zhen Qin, Rolf Jagerman, Rama Kumar Pasumarthi, Honglei Zhuang, He Zhang, Aijun Bai, Kai Hui, Le Yan, Xuanhui Wang

**<details><summary>Abstract**</summary>

The distillation of ranking models has become an important topic in both academia and industry. In recent years, several advanced methods have been proposed to tackle this problem, often leveraging ranking information from teacher rankers that is absent in traditional classification settings. To date, there is no well-established consensus on how to evaluate this class of models. Moreover, inconsistent benchmarking on a wide range of tasks and datasets make it difficult to assess or invigorate advances in this field. This paper first examines representative prior arts on ranking distillation, and raises three questions to be answered around methodology and reproducibility. To that end, we propose a systematic and unified benchmark, Ranking Distillation Suite (RD-Suite), which is a suite of tasks with 4 large real-world datasets, encompassing two major modalities (textual and numeric) and two applications (standard distillation and distillation transfer). RD-Suite consists of benchmark results that challenge some of the common wisdom in the field, and the release of datasets with teacher scores and evaluation scripts for future research. RD-Suite paves the way towards better understanding of ranking distillation, facilities more research in this direction, and presents new challenges.</details>

### REASONER: An Explainable Recommendation Dataset with Comprehensive Labeling Ground Truths

**Authors:** Xu Chen, Jingsen Zhang, Lei Wang, Quanyu Dai, Zhenhua Dong, Ruiming Tang, Rui Zhang, Li Chen, Xin Zhao, Ji-Rong Wen

**<details><summary>Abstract**</summary>

Explainable recommendation has attracted much attention from the industry and academic communities. It has shown great potential to improve the recommendation persuasiveness, informativeness and user satisfaction. In the past few years, while a lot of promising explainable recommender models have been proposed, the datasets used to evaluate them still suffer from several limitations, for example, the explanation ground truths are not labeled by the real users, the explanations are mostly single-modal and around only one aspect. To bridge these gaps, in this paper, we build a new explainable recommendation dataset, which, to our knowledge, is the first contribution that provides a large amount of real user labeled multi-modal and multi-aspect explaination ground truths. In specific, we firstly develop a video recommendation platform, where a series of questions around the recommendation explainability are carefully designed. Then, we recruit about 3000 high-quality labelers with different backgrounds to use the system, and collect their behaviors and feedback to our questions. In this paper, we detail the construction process of our dataset and also provide extensive analysis on its characteristics. In addition, we develop a library, where ten well-known explainable recommender models are implemented in a unified framework. Based on this library, we build several benchmarks for different explainable recommendation tasks. At last, we present many new opportunities brought by our dataset, which are expected to promote the field of explainable recommendation. Our dataset, library and the related documents have been released at https://reasoner2023.github.io/.</details>

### RECESS Vaccine for Federated Learning: Proactive Defense Against Model Poisoning Attacks

**Authors:** Haonan Yan, Wenjing Zhang, Qian Chen, Xiaoguang Li, Wenhai Sun, HUI LI, Xiaodong Lin

**<details><summary>Abstract**</summary>

Model poisoning attacks greatly jeopardize the application of federated learning (FL). The effectiveness of existing defenses is susceptible to the latest model poisoning attacks, leading to a decrease in prediction accuracy. Besides, these defenses are intractable to distinguish benign outliers from malicious gradients, which further compromises the model generalization. In this work, we propose a novel defense including detection and aggregation, named RECESS, to serve as a “vaccine” for FL against model poisoning attacks. Different from the passive analysis in previous defenses, RECESS proactively queries each participating client with a delicately constructed aggregation gradient, accompanied by the detection of malicious clients according to their responses with higher accuracy. Further, RECESS adopts a newly proposed trust scoring based mechanism to robustly aggregate gradients. Rather than previous methods of scoring in each iteration, RECESS takes into account the correlation of clients’ performance over multiple iterations to estimate the trust score, bringing in a significant increase in detection fault tolerance. Finally, we extensively evaluate RECESS on typical model architectures and four datasets under various settings including white/black-box, cross-silo/device FL, etc. Experimental results show the superiority of RECESS in terms of reducing accuracy loss caused by the latest model poisoning attacks over five classic and two state-of-the-art defenses.</details>

### RaLEs: a Benchmark for Radiology Language Evaluations

**Authors:** Juan M Zambrano Chaves, Nandita Bhaskhar, Maayane Attias, Jean-Benoit Delbrouck, Daniel Rubin, Andreas Loening, Curtis Langlotz, Akshay Chaudhari

**<details><summary>Abstract**</summary>

The radiology report is the main form of communication between radiologists and other clinicians. Prior work in natural language processing in radiology reports has shown the value of developing methods tailored for individual tasks such as identifying reports with critical results or disease detection. Meanwhile, English and biomedical natural language understanding benchmarks such as the General Language Understanding and Evaluation as well as Biomedical Language Understanding and Reasoning Benchmark have motivated the development of models that can be easily adapted to address many tasks in those domains. Here, we characterize the radiology report as a distinct domain and introduce RaLEs, the Radiology Language Evaluations, as a benchmark for natural language understanding and generation in radiology. RaLEs is comprised of seven natural language understanding and generation evaluations including the extraction of anatomical and disease entities and their relations, procedure selection, and report summarization. We characterize the performance of models designed for the general, biomedical, clinical and radiology domains across these tasks. We find that advances in the general and biomedical domains do not necessarily translate to radiology, and that improved models from the general domain can perform comparably to smaller clinical-specific models. The limited performance of existing pre-trained models on RaLEs highlights the opportunity to improve domain-specific self-supervised models for natural language processing in radiology. We propose RaLEs as a benchmark to promote and track the development of such domain-specific radiology language models.</details>

### ReContrast: Domain-Specific Anomaly Detection via Contrastive Reconstruction

**Authors:** Jia Guo, shuai lu, Lize Jia, Weihang Zhang, Huiqi Li

**<details><summary>Abstract**</summary>

Most advanced unsupervised anomaly detection (UAD) methods rely on modeling feature representations of frozen encoder networks pre-trained on large-scale datasets, e.g. ImageNet. However, the features extracted from the encoders that are borrowed from natural image domains coincide little with the features required in the target UAD domain, such as industrial inspection and medical imaging. In this paper, we propose a novel epistemic UAD method, namely ReContrast, which optimizes the entire network to reduce biases towards the pre-trained image domain and orients the network in the target domain. We start with a feature reconstruction approach that detects anomalies from errors. Essentially, the elements of contrastive learning are elegantly embedded in feature reconstruction to prevent the network from training instability, pattern collapse, and identical shortcut, while simultaneously optimizing both the encoder and decoder on the target domain. To demonstrate our transfer ability on various image domains, we conduct extensive experiments across two popular industrial defect detection benchmarks and three medical image UAD tasks, which shows our superiority over current state-of-the-art methods.</details>

### Read and Reap the Rewards: Learning to Play Atari with the Help of Instruction Manuals

**Authors:** Yue Wu, Yewen Fan, Paul Pu Liang, Amos Azaria, Yuanzhi Li, Tom Mitchell

**<details><summary>Abstract**</summary>

High sample complexity has long been a challenge for RL. On the other hand, humans learn to perform tasks not only from interaction or demonstrations, but also by reading unstructured text documents, e.g., instruction manuals. Instruction manuals and wiki pages are among the most abundant data that could inform agents of valuable features and policies or task-specific environmental dynamics and reward structures. Therefore, we hypothesize that the ability to utilize human-written instruction manuals to assist learning policies for specific tasks should lead to a more efficient and better-performing agent. We propose the Read and Reward framework. Read and Reward speeds up RL algorithms on Atari games by reading manuals released by the Atari game developers. Our framework consists of a QA Extraction module that extracts and summarizes relevant information from the manual and a Reasoning module that evaluates object-agent interactions based on information from the manual. An auxiliary reward is then provided to a standard A2C RL agent, when interaction is detected. Experimentally, various RL algorithms obtain significant improvement in performance and training speed when assisted by our design. Code at github.com/Holmeswww/RnR</details>

### Reconciling Competing Sampling Strategies of Network Embedding

**Authors:** Yuchen Yan, Baoyu Jing, Lihui Liu, Ruijie Wang, Jinning Li, Tarek Abdelzaher, Hanghang Tong

**<details><summary>Abstract**</summary>

Network embedding plays a significant role in a variety of applications. To capture the topology of the network, most of the existing network embedding algorithms follow a sampling training procedure, which maximizes the similarity (e.g., embedding vectors' dot product) between positively sampled node pairs and minimizes the similarity between negatively sampled node pairs in the embedding space. Typically, close node pairs function as positive samples while distant node pairs are usually considered as negative samples. However, under different or even competing sampling strategies, some methods champion sampling distant node pairs as positive samples to encapsulate longer distance information in link prediction, whereas others advocate adding close nodes into the negative sample set to boost the performance of node recommendation. In this paper, we seek to understand the intrinsic relationships between these competing strategies. To this end, we identify two properties (discrimination and monotonicity) that given any node pair proximity distribution, node embeddings should embrace.Moreover, we quantify the empirical error of the trained similarity score w.r.t. the sampling strategy, which leads to an important finding that the discrimination property and the monotonicity property for all node pairs can not be satisfied simultaneously in real-world applications. Guided by such analysis, a simple yet novel model (SENSEI) is proposed, which seamlessly fulfills the discrimination property and the partial monotonicity within the top-$K$ ranking list. Extensive experiments show that SENSEI outperforms the state-of-the-arts in plain network embedding.</details>

### Recovering Unbalanced Communities in the Stochastic Block Model with Application to Clustering with a Faulty Oracle

**Authors:** Chandra Sekhar Mukherjee, Pan Peng, Jiapeng Zhang

**<details><summary>Abstract**</summary>

The stochastic block model (SBM) is a fundamental model for studying graph clustering or community detection in networks. It has received great attention in the last decade and the balanced case, i.e., assuming all clusters have large size, has been well studied. However, our understanding of SBM with unbalanced communities (arguably, more relevant in practice) is still limited. In this paper, we provide a simple SVD-based algorithm for recovering the communities in the SBM with communities of varying sizes.We improve upon a result of Ailon, Chen and Xu [ICML 2013; JMLR 2015] by removing the assumption that there is a large interval such that the sizes of clusters do not fall in, and also remove the dependency of the size of the recoverable clusters on the number of underlying clusters. We further complement our theoretical improvements with experimental comparisons.Under the planted clique conjecture, the size of the clusters that can be recovered by our algorithm is nearly optimal (up to poly-logarithmic factors) when the probability parameters are constant. As a byproduct, we obtain an efficient clustering algorithm with sublinear query complexity in a faulty oracle model, which is capable of detecting all clusters larger than $\tilde{\Omega}({\sqrt{n}})$, even in the presence of $\Omega(n)$ small clusters in the graph. In contrast, previous efficient algorithms that use a sublinear number of queries are incapable of recovering any large clusters if there are more than $\tilde{\Omega}(n^{2/5})$ small clusters.</details>

### Refining Diffusion Planner for Reliable Behavior Synthesis by Automatic Detection of Infeasible Plans

**Authors:** Kyowoon Lee, Seongun Kim, Jaesik Choi

**<details><summary>Abstract**</summary>

Diffusion-based planning has shown promising results in long-horizon, sparse-reward tasks by training trajectory diffusion models and conditioning the sampled trajectories using auxiliary guidance functions. However, due to their nature as generative models, diffusion models are not guaranteed to generate feasible plans, resulting in failed execution and precluding planners from being useful in safety-critical applications. In this work, we propose a novel approach to refine unreliable plans generated by diffusion models by providing refining guidance to error-prone plans. To this end, we suggest a new metric named restoration gap for evaluating the quality of individual plans generated by the diffusion model. A restoration gap is estimated by a gap predictor which produces restoration gap guidance to refine a diffusion planner. We additionally present an attribution map regularizer to prevent adversarial refining guidance that could be generated from the sub-optimal gap predictor, which enables further refinement of infeasible plans. We demonstrate the effectiveness of our approach on three different benchmarks in offline control settings that require long-horizon planning. We also illustrate that our approach presents explainability by presenting the attribution maps of the gap predictor and highlighting error-prone transitions, allowing for a deeper understanding of the generated plans.</details>

### [Spotlight] Regularization properties of adversarially-trained linear regression

**Authors:** Antonio Ribeiro, Dave Zachariah, Francis Bach, Thomas Schön

**<details><summary>Abstract**</summary>

State-of-the-art machine learning models can be vulnerable to very small input perturbations that are adversarially constructed. Adversarial training is an effective approach to defend against it. Formulated as a min-max problem, it searches for the best solution when the training data were corrupted by the worst-case attacks. Linear models are among the simple models where vulnerabilities can be observed and are the focus of our study. In this case, adversarial training leads to a convex optimization problem which can be formulated as the minimization of a finite sum. We provide a comparative analysis between the solution of adversarial training in linear regression and other regularization methods. Our main findings are that: (A) Adversarial training yields the  minimum-norm  interpolating solution in the overparameterized regime (more parameters than data), as long as the maximum disturbance radius is smaller than a threshold. And, conversely, the minimum-norm interpolator is the solution to adversarial training with a given radius. (B) Adversarial training can be equivalent to parameter shrinking methods (ridge regression and Lasso). This happens in the underparametrized region, for an appropriate choice of adversarial radius and zero-mean symmetrically distributed covariates. (C) For $\ell_\infty$-adversarial training---as in square-root Lasso---the choice of adversarial radius for optimal bounds does not depend on the additive noise variance. We confirm our theoretical findings with numerical examples.</details>

### Reinforcement Learning with Fast and Forgetful Memory

**Authors:** Steven D Morad, Ryan Kortvelesy, Stephan Liwicki, Amanda Prorok

**<details><summary>Abstract**</summary>

Nearly all real world tasks are inherently partially observable, necessitating the use of memory in Reinforcement Learning (RL). Most model-free approaches summarize the trajectory into a latent Markov state using memory models borrowed from Supervised Learning (SL), even though RL tends to exhibit different training and efficiency characteristics. Addressing this discrepancy, we introduce Fast and Forgetful Memory, an algorithm-agnostic memory model designed specifically for RL. Our approach constrains the model search space via strong structural priors inspired by computational psychology. It is a drop-in replacement for recurrent neural networks (RNNs) in recurrent RL algorithms, achieving greater reward than RNNs across various recurrent benchmarks and algorithms _without changing any hyperparameters_. Moreover, Fast and Forgetful Memory exhibits training speeds two orders of magnitude faster than RNNs, attributed to its logarithmic time and linear space complexity. Our implementation is available at https://github.com/proroklab/ffm.</details>

### [Spotlight] Reinforcement-Enhanced Autoregressive Feature Transformation: Gradient-steered Search in Continuous Space for Postfix Expressions

**Authors:** Dongjie Wang, Meng Xiao, Min Wu, pengfei wang, Yuanchun Zhou, Yanjie Fu

**<details><summary>Abstract**</summary>

Feature transformation aims to generate new pattern-discriminative feature space from original features to improve downstream machine learning (ML) task performances. However, the discrete search space for the optimal feature explosively grows on the basis of combinations of features and operations from low-order forms to high-order forms. Existing methods, such as exhaustive search, expansion reduction, evolutionary algorithms, reinforcement learning, and iterative greedy, suffer from large search space. Overly emphasizing efficiency in algorithm design usually sacrifice stability or robustness. To fundamentally fill this gap, we reformulate discrete feature transformation as a continuous space optimization task and develop an embedding-optimization-reconstruction framework. This framework includes four steps: 1) reinforcement-enhanced data preparation, aiming to prepare high-quality transformation-accuracy training data; 2) feature transformation operation sequence embedding, intending to encapsulate the knowledge of prepared training data within a continuous space; 3) gradient-steered optimal embedding search, dedicating to uncover potentially superior embeddings within the learned space; 4) transformation operation sequence reconstruction, striving to reproduce the feature transformation solution to pinpoint the optimal feature space. Finally, extensive experiments and case studies are performed to demonstrate the effectiveness and robustness of the proposed method. The code and data are publicly accessible https://www.dropbox.com/sh/imh8ckui7va3k5u/AACulQegVx0MuywYyoCqSdVPa?dl=0.</details>

### Reliable Off-Policy Learning for Dosage Combinations

**Authors:** Jonas Schweisthal, Dennis Frauen, Valentyn Melnychuk, Stefan Feuerriegel

**<details><summary>Abstract**</summary>

Decision-making in personalized medicine such as cancer therapy or critical care must often make choices for dosage combinations, i.e., multiple continuous treatments. Existing work for this task has modeled the effect of multiple treatments independently, while estimating the joint effect has received little attention but comes with non-trivial challenges. In this paper, we propose a novel method for reliable off-policy learning for dosage combinations. Our method proceeds along three steps: (1) We develop a tailored neural network that estimates the individualized dose-response function while accounting for the joint effect of multiple dependent dosages. (2) We estimate the generalized propensity score using conditional normalizing flows in order to detect regions with limited overlap in the shared covariate-treatment space. (3) We present a gradient-based learning algorithm to find the optimal, individualized dosage combinations. Here, we ensure reliable estimation of the policy value by avoiding regions with limited overlap. We finally perform an extensive evaluation of our method to show its effectiveness. To the best of our knowledge, ours is the first work to provide a method for reliable off-policy learning for optimal dosage combinations.</details>

### Renku: a platform for sustainable data science

**Authors:** Rok Roškar, Chandrasekhar Ramakrishnan, Michele Volpi, Fernando Perez-Cruz, Lilian Gasser, Firat Ozdemir, Patrick Paitz, Mohammad Alisafaee, Philipp Fischer, Ralf Grubenmann, Eliza Harris, Tasko Olevski, Carl Remlinger, Luis Salamanca, Elisabet Capon Garcia, Lorenzo Cavazzi, Jakub Chrobasik, Darlin Cordoba Osnas, Alessandro Degano, Jimena Dupre, Wesley Johnson, Eike Kettner, Laura Kinkead, Sean Murphy, Flora Thiebaut, Olivier Verscheure

**<details><summary>Abstract**</summary>

Data and code working together is fundamental to machine learning (ML), but the context around datasets and interactions between datasets and code are in general captured only rudimentarily. Context such as how the dataset was prepared and created, what source data were used, what code was used in processing, how the dataset evolved, and where it has been used and reused can provide much insight, but this information is often poorly documented. That is unfortunate since it makes datasets into black-boxes with potentially hidden characteristics that have downstream consequences. We argue that making dataset preparation more accessible and dataset usage easier to record and document would have significant benefits for the ML community: it would allow for greater diversity in datasets by inviting modification to published sources, simplify use of alternative datasets and, in doing so, make results more transparent and robust, while allowing for all contributions to be adequately credited. We present a platform, Renku, designed to support and encourage such sustainable development and use of data, datasets, and code, and we demonstrate its benefits through a few illustrative projects which span the spectrum from dataset creation to dataset consumption and showcasing.</details>

### Reproducibility Study of ”CartoonX: Cartoon Explanations of Image Classifiers”

**Authors:** Aditya Patra, Sina Taslimi, Luke Chin A Foeng, Pratik Kayal

**<details><summary>Abstract**</summary>

In this reproducibility study, we verify the claims and contributions in Cartoon Explanations of Image Classifiers by Kolek et al.. These include (i) A proposed technique named CartoonX used to extract visual explanations for predictions via image classification networks, (ii) CartoonX being able to reveal piece‐wise smooth regions of the image, unlike previous methods, which extract relevant pixel‐sparse regions, and (iii) CartoonX achieving lower distortion values than these methods.</details>

### Reproducibility study of 'Proto2Proto: Can you recognise the car, the way I do?'

**Authors:** Gerson de Kleuver, David Bikker, Wenhua Hu, Bram Veenman

**<details><summary>Abstract**</summary>

Scope of Reproducibility — This paper analyses the reproducibility of the study Proto2Proto: Can you recognize the car, the way I do? The main contributions and claims of the study are: 1) Using Proto2Proto, a shallower student model is more faithful to the teacher in terms of interpretability than a baseline student model while also showing the same or better accuracy; 2) Global Explanation loss forces student prototypes to be close to teacher prototypes; 3) Patch‐Prototype Correspondence loss enforces the local representations of the student to be similar to those of the teacher; 4) The proposed evaluation metrics determine the faithfulness of the student to the teacher in terms of interpretability.Methodology — A public code repository was available for the paper, which provided a working but incomplete and minimally documented codebase. With some modifications we were able to carry out the experiments that were best supported by the codebase. We spent a total of 60 computational GPU hours on reproduction.Results — The results we were able to produce support claim 1, albeit weakly. Further results are in line with the paper, but we found them to go against claim 3. In addition, we carried out a theoretical analysis which provides support for claim 4. Finally, we were unable to carry out our intended experiment to verify claim 2. What was easy — The original paper was clearly structured and understandable. The experiments for which configurations were provided were simple to conduct.What was difficult — The public codebase contained minimal documentation. Moreover, the use of variable names did not correspond between the code and the paper. Furthermore, the codebase lacked elements vital to reproducing some experiments. Another significant constraint were the computational requirements needed to reproduce the original experiments. Finally, the code required to reproduce one of the visualizations was not provided.Communication with original authors — We contacted the authors to ask for trained model weights and missing hyperparameters for several experiments. We did not receive aresponse.</details>

### [Spotlight] ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting

**Authors:** Zongsheng Yue, Jianyi Wang, Chen Change Loy

**<details><summary>Abstract**</summary>

Diffusion-based image super-resolution (SR) methods are mainly limited by the low inference speed due to the requirements of hundreds or even thousands of sampling steps. Existing acceleration sampling techniques inevitably sacrifice performance to some extent, leading to over-blurry SR results. To address this issue, we propose a novel and efficient diffusion model for SR that significantly reduces the number of diffusion steps, thereby eliminating the need for post-acceleration during inference and its associated performance deterioration. Our method constructs a Markov chain that transfers between the high-resolution image and the low-resolution image by shifting the residual between them, substantially improving the transition efficiency. Additionally, an elaborate noise schedule is developed to flexibly control the shifting speed and the noise strength during the diffusion process. Extensive experiments demonstrate that the proposed method obtains superior or at least comparable performance to current state-of-the-art methods on both synthetic and real-world datasets, \textit{\textbf{even only with 20 sampling steps}}. Our code and model will be made publicly.</details>

### Rethinking Conditional Diffusion Sampling with Progressive Guidance

**Authors:** Anh-Dung Dinh, Daochang Liu, Chang Xu

**<details><summary>Abstract**</summary>

This paper tackles two critical challenges encountered in classifier guidance for diffusion generative models, i.e., the lack of diversity and the presence of adversarial effects. These issues often result in a scarcity of diverse samples or the generation of non-robust features. The underlying cause lies in the mechanism of classifier guidance, where discriminative gradients push samples to be recognized as conditions aggressively. This inadvertently suppresses information with common features among relevant classes, resulting in a limited pool of features with less diversity or the absence of robust features for image construction.We propose a generalized classifier guidance method called Progressive Guidance, which mitigates the problems by allowing relevant classes' gradients to contribute to shared information construction when the image is noisy in early sampling steps. In the later sampling stage, we progressively enhance gradients to refine the details in the image toward the primary condition. This helps to attain a high level of diversity and robustness compared to the vanilla classifier guidance. Experimental results demonstrate that our proposed method further improves the image quality while offering a significant level of diversity as well as robust features.</details>

### Rethinking the Role of Token Retrieval in Multi-Vector Retrieval

**Authors:** Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, Vincent Zhao

**<details><summary>Abstract**</summary>

Multi-vector retrieval models such as ColBERT [Khattab et al., 2020] allow token-level interactions between queries and documents, and hence achieve state of the art on many information retrieval benchmarks. However, their non-linear scoring function cannot be scaled to millions of documents, necessitating a three-stage process for inference: retrieving initial candidates via token retrieval, accessing all token vectors, and scoring the initial candidate documents. The non-linear scoring function is applied over all token vectors of each candidate document, making the inference process complicated and slow. In this paper, we aim to simplify the multi-vector retrieval by rethinking the role of token retrieval. We present XTR, ConteXtualized Token Retriever, which introduces a simple, yet novel, objective function that encourages the model to retrieve the most important document tokens first. The improvement to token retrieval allows XTR to rank candidates only using the retrieved tokens rather than all tokens in the document, and enables a newly designed scoring stage that is two-to-three orders of magnitude cheaper than that of ColBERT. On the popular BEIR benchmark, XTR advances the state-of-the-art by 2.8 nDCG@10 without any distillation. Detailed analysis confirms our decision to revisit the token retrieval stage, as XTR demonstrates much better recall of the token retrieval stage compared to ColBERT.</details>

### Reverse Engineering Self-Supervised Learning

**Authors:** Ido Ben-Shaul, Ravid Shwartz-Ziv, Tomer Galanti, Shai Dekel, Yann LeCun

**<details><summary>Abstract**</summary>

Understanding the learned representation and underlying mechanisms of Self-Supervised Learning (SSL) often poses a challenge. In this paper, we ‘reverse engineer’ SSL, conducting an in-depth empirical analysis of its learned internal representations, encompassing diverse models, architectures, and hyperparameters. Our study reveals an intriguing process within the SSL training: an inherent facilitation of semantic label-based clustering, which is surprisingly driven by the regularization component of the SSL objective. This clustering not only enhances downstream classification, but also compresses the information. We further illustrate that the alignment of the SSL-trained representation is more pronounced with semantic classes rather than random functions. Remarkably, the learned representations align with semantic classes across various hierarchical levels, with this alignment intensifying when going deeper into the network. This ‘reverse engineering’ approach provides valuable insights into the inner mechanism of SSL and their influences on the performance across different class sets.</details>

### Revisit Weakly-Supervised Audio-Visual Video Parsing from the Language Perspective

**Authors:** Yingying Fan, Yu Wu, Bo Du, Yutian Lin

**<details><summary>Abstract**</summary>

We focus on the weakly-supervised audio-visual video parsing task (AVVP), which aims to identify and locate all the events in audio/visual modalities. Previous works only concentrate on video-level overall label denoising across modalities, but overlook the segment-level label noise, where adjacent video segments (i.e., 1-second video clips) may contain different events. However, recognizing events on the segment is challenging because its label could be any combination of events that occur in the video. To address this issue, we consider tackling AVVP from the language perspective, since language could freely describe how various events appear in each segment beyond fixed labels. Specifically, we design language prompts to describe all cases of event appearance for each video. Then, the similarity between language prompts and segments is calculated, where the event of the most similar prompt is regarded as the segment-level label. In addition, to deal with the mislabeled segments, we propose to perform dynamic re-weighting on the unreliable segments to adjust their labels. Experiments show that our simple yet effective approach outperforms state-of-the-art methods by a large margin.</details>

### Revisiting Adversarial Training for ImageNet: Architectures, Training and Generalization across Threat Models

**Authors:** Naman Deep Singh, Francesco Croce, Matthias Hein

**<details><summary>Abstract**</summary>

While adversarial training has been extensively studied for ResNet architectures and low resolution datasets like CIFAR-10, much less is known for ImageNet. Given the recent debate about whether transformers are more robust than convnets, we revisit adversarial training on ImageNet comparing ViTs and ConvNeXts. Extensive experiments show that minor changes in architecture, most notably replacing PatchStem with ConvStem, and training scheme have a significant impact on the achieved robustness. These changes not only increase robustness in the seen $\ell_\infty$-threat model, but even more so improve generalization to unseen $\ell_1/\ell_2$-attacks. Our modified ConvNeXt, ConvNeXt + ConvStem, yields the most robust $\ell_\infty$-models across different ranges of model parameters and FLOPs, while our ViT + ConvStem yields the best generalization to unseen threat models.</details>

### Reward-Directed Conditional Diffusion: Provable Distribution Estimation and Reward Improvement

**Authors:** Hui Yuan, Kaixuan Huang, Chengzhuo Ni, Minshuo Chen, Mengdi Wang

**<details><summary>Abstract**</summary>

We explore the methodology and theory of reward-directed generation via conditional diffusion models. Directed generation aims to generate samples with desired properties as measured by a reward function, which has broad applications in generative AI, reinforcement learning, and computational biology. We consider the common learning scenario where the dataset consists of majorly unlabeled data and a small set of data with noisy reward labels. Our approach leverages a learned reward function on the smaller data set as a pseudolabeler to label the unlabelled data. After pseudo-labelling, a conditional diffusion model (CDM) is trained on the data and samples are generated by setting a target value $a$ as the condition in CDM. From a theoretical standpoint, we show that this directed generator can effectively learn and sample from the reward-conditioned data distribution: 1. our model is capable of recovering the data's latent subspace representation. 2. the model generates samples moving closer to the user-specified target. The improvement in rewards of samples is influenced by a interplay between the strength of the reward signal, the distribution shift, and the cost of off-support extrapolation. We provide empirical results to validate our theory and highlight the relationship between the strength of extrapolation and the quality of generated samples.</details>

### Rewarded soups: towards Pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards

**Authors:** Alexandre Rame, Guillaume Couairon, Corentin Dancette, Jean-Baptiste Gaya, Mustafa Shukor, Laure Soulier, Matthieu Cord

**<details><summary>Abstract**</summary>

Foundation models are first pre-trained on vast unsupervised datasets and then fine-tuned on labeled data. Reinforcement learning, notably from human feedback (RLHF), can further align the network with the intended usage. Yet the imperfections in the proxy reward may hinder the training and lead to suboptimal results; the diversity of objectives in real-world tasks and human opinions exacerbate the issue. This paper proposes embracing the heterogeneity of diverse rewards by following a multi-policy strategy. Rather than focusing on a single a priori reward, we aim for Pareto-optimal generalization across the entire space of preferences. To this end, we propose rewarded soup, first specializing multiple networks independently (one for each proxy reward) and then interpolating their weights linearly. This succeeds empirically because we show that the weights remain linearly connected when fine-tuned on diverse rewards from a shared pre-trained initialization. We demonstrate the effectiveness of our approach for text-to-text (summarization, Q&A, helpful assistant, review), text-image (image captioning, text-to-image generation, visual grounding), and control (locomotion) tasks. We hope to enhance the alignment of deep models, and how they interact with the world in all its diversity.</details>

### Rewrite Caption Semantics: Bridging Semantic Gaps for Language-Supervised Semantic Segmentation

**Authors:** Yun Xing, Jian Kang, Aoran Xiao, Jiahao Nie, Ling Shao, Shijian Lu

**<details><summary>Abstract**</summary>

Vision-Language Pre-training has demonstrated its remarkable zero-shot recognition ability and potential to learn generalizable visual representations from languagesupervision. Taking a step ahead, language-supervised semantic segmentation enables spatial localization of textual inputs by learning pixel grouping solely from image-text pairs. Nevertheless, the state-of-the-art suffers from a clear semantic gap between visual and textual modalities: plenty of visual concepts appeared in images are missing in their paired captions. Such semantic misalignment circulates in pre-training, leading to inferior zero-shot performance in dense predictions due to insufficient visual concepts captured in textual representations. To close such semantic gap, we propose Concept Curation (CoCu), a pipeline that leverages CLIP to compensate for the missing semantics. For each image-text pair, we establish a concept archive that maintains potential visually-matched concepts with our proposed vision-driven expansion and text-to-vision-guided ranking. Relevant concepts can thus be identified via cluster-guided sampling and fed into pre-training, thereby bridging the gap between visual and textual semantics. Extensive experiments over a broad suite of 8 segmentation benchmarks show that CoCu achieves superb zero-shot transfer performance and greatly boosts language-supervised segmentation baseline by a large margin, suggesting the value of closing semantic gap in pre-training data.</details>

### Rigorous Runtime Analysis of MOEA/D for Solving Multi-Objective Minimum Weight Base Problems

**Authors:** Anh Viet Do, Aneta Neumann, Frank Neumann, Andrew Sutton

**<details><summary>Abstract**</summary>

We study the multi-objective minimum weight base problem, an abstraction of classical NP-hard combinatorial problems such as the multi-objective minimum spanning tree problem. We prove some important properties of the convex hull of the non-dominated front, such as its approximation quality and an upper bound on the number of extreme points. Using these properties, we give the first run-time analysis of the MOEA/D algorithm for this problem, an evolutionary algorithm that effectively optimizes by decomposing the objectives into single-objective components. We show that the MOEA/D, given an appropriate decomposition setting, finds all extreme points within expected fixed-parameter polynomial time, in the oracle model. Experiments are conducted on random bi-objective minimum spanning tree instances, and the results agree with our theoretical findings. Furthermore, compared with a previously studied evolutionary algorithm for the problem GSEMO, MOEA/D finds all extreme points much faster across all instances.</details>

### Risk-Averse Model Uncertainty for Distributionally Robust Safe Reinforcement Learning

**Authors:** James Queeney, Mouhacine Benosman

**<details><summary>Abstract**</summary>

Many real-world domains require safe decision making in uncertain environments. In this work, we introduce a deep reinforcement learning framework for approaching this important problem. We consider a distribution over transition models, and apply a risk-averse perspective towards model uncertainty through the use of coherent distortion risk measures. We provide robustness guarantees for this framework by showing it is equivalent to a specific class of distributionally robust safe reinforcement learning problems. Unlike existing approaches to robustness in deep reinforcement learning, however, our formulation does not involve minimax optimization. This leads to an efficient, model-free implementation of our approach that only requires standard data collection from a single training environment. In experiments on continuous control tasks with safety constraints, we demonstrate that our framework produces robust performance and safety at deployment time across a range of perturbed test environments.</details>

### RoboDepth: Robust Out-of-Distribution Depth Estimation under Corruptions

**Authors:** Lingdong Kong, Shaoyuan Xie, Hanjiang Hu, Lai Xing Ng, Benoit Cottereau, Wei Tsang Ooi

**<details><summary>Abstract**</summary>

Depth estimation from monocular images is pivotal for real-world visual perception systems. While current learning-based depth estimation models train and test on meticulously curated data, they often overlook out-of-distribution (OoD) situations. Yet, in practical settings -- especially safety-critical ones like autonomous driving -- common corruptions can arise. Addressing this oversight, we introduce a comprehensive robustness test suite, RoboDepth, encompassing 18 corruptions spanning three categories: i) weather and lighting conditions; ii) sensor failures and movement; and iii) data processing anomalies. We subsequently benchmark 42 depth estimation models across indoor and outdoor scenes to assess their resilience to these corruptions. Our findings underscore that, in the absence of a dedicated robustness evaluation framework, many leading depth estimation models may be susceptible to typical corruptions. We delve into design considerations for crafting more robust depth estimation models, touching upon pre-training, augmentation, modality, model capacity, and learning paradigms. We anticipate our benchmark will establish a foundational platform for advancing robust OoD depth estimation.</details>

### Robust Data Pruning under Label Noise via Maximizing Re-labeling Accuracy

**Authors:** Dongmin Park, Seola Choi, Doyoung Kim, Hwanjun Song, Jae-Gil Lee

**<details><summary>Abstract**</summary>

Data pruning, which aims to downsize a large training set into a small informative subset, is crucial for reducing the enormous computational costs of modern deep learning. Though large-scale data collections invariably contain annotation noise and numerous robust learning methods have been developed, data pruning for the noise-robust learning scenario has received little attention. With state-of-the-art Re-labeling methods that self-correct erroneous labels while training, it is challenging to identify which subset induces the most accurate re-labeling of erroneous labels in the entire training set. In this paper, we formalize the problem of data pruning with re-labeling. We first show that the likelihood of a training example being correctly re-labeled is proportional to the prediction confidence of its neighborhood in the subset. Therefore, we propose a novel data pruning algorithm, Prune4Rel, that finds a subset maximizing the total neighborhood confidence of all training examples, thereby maximizing the re-labeling accuracy and generalization performance. Extensive experiments on four real and one synthetic noisy datasets show that Prune4Rel outperforms the baselines with Re-labeling models by up to 9.1% as well as those with a standard model by up to 21.6%.</details>

### [Spotlight] Robust Distributed Learning: Tight Error Bounds and Breakdown Point under Data Heterogeneity

**Authors:** Youssef Allouah, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot, Geovani Rizk

**<details><summary>Abstract**</summary>

The theory underlying robust distributed learning algorithms, designed to resist adversarial machines, matches empirical observations when data is homogeneous. Under data heterogeneity however, which is the norm in practical scenarios, established lower bounds on the learning error are essentially vacuous and greatly mismatch empirical observations. This is because the heterogeneity model considered is too restrictive and does not cover basic learning tasks such as least-squares regression. We consider in this paper a more realistic heterogeneity model, namely $(G,B)$-gradient dissimilarity, and show that it covers a larger class of learning problems than existing theory. Notably, we show that the breakdown point under heterogeneity is lower than the classical fraction $\frac{1}{2}$. We also prove a new lower bound on the learning error of any distributed learning algorithm. We derive a matching upper bound for a robust variant of distributed gradient descent, and empirically show that our analysis reduces the gap between theory and practice.</details>

### Robust Learning for Smoothed Online Convex Optimization with Feedback Delay

**Authors:** Pengfei Li, Jianyi Yang, Adam Wierman, Shaolei Ren

**<details><summary>Abstract**</summary>

We study a general form of Smoothed Online Convex Optimization, a.k.a. SOCO, including multi-step switching costs and feedback delay. We propose a novel machine learning (ML) augmented online algorithm, Robustness-Constrained Learning (RCL), which combines untrusted ML predictions with a trusted expert online algorithm via constrained projection to robustify the ML prediction. Specifically, we prove that RCL is able to guarantee $(1+\lambda)$-competitiveness against any given expert for any $\lambda>0$, while also explicitly training the ML model in a robustification-aware manner to improve the average-case performance. Importantly, RCL is the first ML-augmented algorithm with a provable robustness guarantee in the case of multi-step switching cost and feedback delay. We demonstrate the improvement of RCL in both robustness and average performance using battery management as a case study.</details>

### SANFlow: Semantic-Aware Normalizing Flow for Anomaly Detection

**Authors:** Daehyun Kim, Sungyong Baik, Tae Hyun Kim

**<details><summary>Abstract**</summary>

Visual anomaly detection, the task of detecting abnormal characteristics in images, is challenging due to the rarity and unpredictability of anomalies. In order to reliably model the distribution of normality and detect anomalies, a few works have attempted to exploit the density estimation ability of normalizing flow (NF). However, previous NF-based methods have relied solely on the capability of NF and forcibly transformed the distribution of all features to a single distribution (e.g., unit normal distribution), when features can have different semantic information and thus follow different distributions. We claim that forcibly learning to transform such diverse distributions to a single distribution with a single network will cause the learning difficulty, limiting the capacity of a network to discriminate normal and abnormal data. As such, we propose to transform the distribution of features at each location of a given image to different distributions. In particular, we train NF to map normal data distribution to distributions with the same mean but different variances at each location of the given image. To enhance the discriminability, we also train NF to map abnormal data distribution to a distribution with a mean that is different from that of normal data, where abnormal data is synthesized with data augmentation. The experimental results outline the effectiveness of the proposed framework in improving the density modeling and thus anomaly detection performance.</details>

### SEENN: Towards Temporal Spiking Early Exit Neural Networks

**Authors:** Yuhang Li, Tamar Geller, Youngeun Kim, Priyadarshini Panda

**<details><summary>Abstract**</summary>

Spiking Neural Networks (SNNs) have recently become more popular as a biologically plausible substitute for traditional Artificial Neural Networks (ANNs). SNNs are cost-efficient and deployment-friendly because they process input in both spatial and temporal manner using binary spikes. However, we observe that the information capacity in SNNs is affected by the number of timesteps, leading to an accuracy-efficiency tradeoff. In this work, we study a fine-grained adjustment of the number of timesteps in SNNs. Specifically, we treat the number of timesteps as a variable conditioned on different input samples to reduce redundant timesteps for certain data. We call our method Spiking Early-Exit Neural Networks (**SEENNs**). To determine the appropriate number of timesteps, we propose SEENN-I which uses a confidence score thresholding to filter out the uncertain predictions, and SEENN-II which determines the number of timesteps by reinforcement learning. Moreover, we demonstrate that SEENN is compatible with both the directly trained SNN and the ANN-SNN conversion. By dynamically adjusting the number of timesteps, our SEENN achieves a remarkable reduction in the average number of timesteps during inference. For example, our SEENN-II ResNet-19 can achieve **96.1**\% accuracy with an average of **1.08** timesteps on the CIFAR-10 test dataset. Code is shared at https://github.com/Intelligent-Computing-Lab-Yale/SEENN.</details>

### SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation

**Authors:** Zhongang Cai, Wanqi Yin, Ailing Zeng, CHEN WEI, Qingping SUN, Wang Yanjun, Hui En Pang, Haiyi Mei, Mingyuan Zhang, Lei Zhang, Chen Change Loy, Lei Yang, Ziwei Liu

**<details><summary>Abstract**</summary>

Expressive human pose and shape estimation (EHPS) unifies body, hands, and face motion capture with numerous applications. Despite encouraging progress, current state-of-the-art methods still depend largely on a confined set of training datasets. In this work, we investigate scaling up EHPS towards the first generalist foundation model (dubbed SMPLer-X), with up to ViT-Huge as the backbone and training with up to 4.5M instances from diverse data sources. With big data and the large model, SMPLer-X exhibits strong performance across diverse test benchmarks and excellent transferability to even unseen environments. 1) For the data scaling, we perform a systematic investigation on 32 EHPS datasets, including a wide range of scenarios that a model trained on any single dataset cannot handle. More importantly, capitalizing on insights obtained from the extensive benchmarking process, we optimize our training scheme and select datasets that lead to a significant leap in EHPS capabilities. 2) For the model scaling, we take advantage of vision transformers to study the scaling law of model sizes in EHPS. Moreover, our finetuning strategy turn SMPLer-X into specialist models, allowing them to achieve further performance boosts. Notably, our foundation model SMPLer-X consistently delivers state-of-the-art results on seven benchmarks such as AGORA (107.2 mm NMVE), UBody (57.4 mm PVE), EgoBody (63.6 mm PVE), and EHF (62.3 mm PVE without finetuning).</details>

### SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities

**Authors:** Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty

**<details><summary>Abstract**</summary>

Many approaches in machine learning rely on a weighted graph to encode thesimilarities between samples in a dataset. Entropic affinities (EAs), which are notably used in the popular Dimensionality Reduction (DR) algorithm t-SNE, are particular instances of such graphs. To ensure robustness to heterogeneous sampling densities, EAs assign a kernel bandwidth parameter to every sample in such a way that the entropy of each row in the affinity matrix is kept constant at a specific value, whose exponential is known as perplexity. EAs are inherently asymmetric and row-wise stochastic, but they are used in DR approaches after undergoing heuristic symmetrization methods that violate both the row-wise constant entropy and stochasticity properties. In this work, we uncover a novel characterization of EA as an optimal transport problem, allowing a natural symmetrization that can be computed efficiently using dual ascent. The corresponding novel affinity matrix derives advantages from symmetric doubly stochastic normalization in terms of clustering performance, while also effectively controlling the entropy of each row thus making it particularly robust to varying noise levels. Following, we present a new DR algorithm, SNEkhorn, that leverages this new affinity matrix. We show its clear superiority to state-of-the-art approaches with several indicators on both synthetic and real-world datasets.</details>

### SSL4EO-L: Datasets and Foundation Models for Landsat Imagery

**Authors:** Adam Stewart, Nils Lehmann, Isaac Corley, Yi Wang, Yi-Chia Chang, Nassim Ait Ait Ali Braham, Shradha Sehgal, Caleb Robinson, Arindam Banerjee

**<details><summary>Abstract**</summary>

The Landsat program is the longest-running Earth observation program in history, with 50+ years of data acquisition by 8 satellites. The multispectral imagery captured by sensors onboard these satellites is critical for a wide range of scientific fields. Despite the increasing popularity of deep learning and remote sensing, the majority of researchers still use decision trees and random forests for Landsat image analysis due to the prevalence of small labeled datasets and lack of foundation models. In this paper, we introduce SSL4EO-L, the first ever dataset designed for Self-Supervised Learning for Earth Observation for the Landsat family of satellites (including 3 sensors and 2 product levels) and the largest Landsat dataset in history (5M image patches). Additionally, we modernize and re-release the L7 Irish and L8 Biome cloud detection datasets, and introduce the first ML benchmark datasets for Landsats 4–5 TM and Landsat 7 ETM+ SR. Finally, we pre-train the first foundation models for Landsat imagery using SSL4EO-L and evaluate their performance on multiple semantic segmentation tasks. All datasets and model weights are available via the TorchGeo library, making reproducibility and experimentation easy, and enabling scientific advancements in the burgeoning field of remote sensing for a multitude of downstream applications.</details>

### STXD: Structural and Temporal Cross-Modal Distillation for Multi-View 3D Object Detection

**Authors:** Sujin Jang, Dae Ung Jo, Sung Ju Hwang, Dongwook Lee, Daehyun Ji

**<details><summary>Abstract**</summary>

3D object detection (3DOD) from multi-view images is an economically appealing alternative to expensive LiDAR-based detectors, but also an extremely challenging task due to the absence of precise spatial cues. Recent studies have leveraged the teacher-student paradigm for cross-modal distillation, where a strong LiDAR-modality teacher transfers useful knowledge to a multi-view-based image-modality student. However, prior approaches have only focused on minimizing global distances between cross-modal features, which may lead to suboptimal knowledge distillation results. Based on these insights, we propose a novel structural and temporal cross-modal knowledge distillation (STXD) framework for multi-view 3DOD. First, STXD reduces redundancy of the feature components of the student by regularizing the cross-correlation of cross-modal features, while maximizing their similarities. Second, to effectively transfer temporal knowledge, STXD encodes temporal relations of features across a sequence of frames via similarity maps. Lastly, STXD also adopts a response distillation method to further enhance the quality of knowledge distillation at the output-level. Our extensive experiments demonstrate that STXD significantly improves the NDS and mAP of the based student detectors by 2.8%~4.5% on the nuScenes testing dataset.</details>

### SUPA: A Lightweight Diagnostic Simulator for Machine Learning in Particle Physics

**Authors:** Atul Kumar Sinha, Daniele Paliotta, Bálint Máté, John Raine, Tobias Golling, François Fleuret

**<details><summary>Abstract**</summary>

Deep learning methods have gained popularity in high energy physics for fast modeling of particle showers in detectors. Detailed simulation frameworks such as the gold standard \textsc{Geant4} are computationally intensive, and current deep generative architectures work on discretized, lower resolution versions of the detailed simulation. The development of models that work at higher spatial resolutions is currently hindered by the complexity of the full simulation data, and by the lack of simpler, more interpretable benchmarks. Our contribution is \textsc{SUPA}, the SUrrogate PArticle propagation simulator, an algorithm and software package for generating data by simulating simplified particle propagation, scattering and shower development in matter. The generation is extremely fast and easy to use compared to \textsc{Geant4}, but still exhibits the key characteristics and challenges of the detailed simulation. The proposed simulator generates thousands of particle showers per second on a desktop machine, a speed up of up to 6 orders of magnitudes over \textsc{Geant4}, and stores detailed geometric information about the shower propagation. \textsc{\textsc{SUPA}} provides much greater flexibility for setting initial conditions and defining multiple benchmarks for the development of models. Moreover, interpreting particle showers as point clouds creates a connection to geometric machine learning and provides challenging and fundamentally new datasets for the field.</details>

### Scalable Transformer for PDE Surrogate Modeling

**Authors:** Zijie Li, Dule Shu, Amir Barati Farimani

**<details><summary>Abstract**</summary>

Transformer has shown state-of-the-art performance on various applications and has recently emerged as a promising tool for surrogate modeling of partial differential equations (PDEs). Despite the introduction of linear-complexity attention, applying Transformer to problems with a large number of grid points can be numerically unstable and computationally expensive. In this work, we propose Factorized Transformer (FactFormer), which is based on an axial factorized kernel integral. Concretely, we introduce a learnable projection operator that decomposes the input function into multiple sub-functions with one-dimensional domain. These sub-functions are then evaluated and used to compute the instance-based kernel with an axial factorized scheme. We showcase that the proposed model is able to simulate 2D Kolmogorov flow on a $256\times 256$ grid and 3D smoke buoyancy on a $64\times64\times64$ grid with good accuracy and efficiency. The proposed factorized scheme can serve as a computationally efficient low-rank surrogate for the full attention scheme when dealing with multi-dimensional problems.</details>

### Scalarization for Multi-Task and Multi-Domain Learning at Scale

**Authors:** Amelie Royer, Tijmen Blankevoort, Babak Ehteshami Bejnordi

**<details><summary>Abstract**</summary>

Training a single model on multiple input domains and/or output tasks allows for compressing information from multiple sources into a unified backbone hence improves model efficiency. It also enables potential positive knowledge transfer across tasks/domains, leading to improved accuracy and data-efficient training. However, optimizing such networks is a challenge, in particular due to discrepancies between the different tasks or domains: Despite several hypotheses and solutions proposed over the years, recent work has shown that uniform scalarization training, i.e., simply minimizing the average of the task losses,  yields on-par performance with more costly SotA optimization methods. This raises the issue of how well we understand the training dynamics of multi-task and multi-domain networks. In this work, we first devise a large-scale unified analysis of multi-domain and multi-task learning to better understand the dynamics of scalarization across varied task/domain combinations and model sizes. Following these insights, we then propose to leverage population-based training to efficiently search for the optimal scalarization weights when dealing with a large number of tasks or domains.</details>

### Scaling Up Differentially Private LASSO Regularized Logistic Regression via Faster Frank-Wolfe Iterations

**Authors:** Edward Raff, Amol Khanna, Fred Lu

**<details><summary>Abstract**</summary>

To the best of our knowledge, there are no methods today for training differentially private regression models on sparse input data. To remedy this, we adapt the Frank-Wolfe algorithm for $L_1$ penalized linear regression to be aware of sparse inputs and to use them effectively. In doing so, we reduce the training time of the algorithm from $\mathcal{O}( T D S + T N S)$ to $\mathcal{O}(N S + T \sqrt{D} \log{D}  + T S^2)$, where $T$ is the number of iterations and a sparsity rate $S$ of a dataset with $N$ rows and $D$ features. Our results demonstrate that this procedure can reduce runtime by a factor of up to $2,200\times$, depending on the value of the privacy parameter $\epsilon$ and the sparsity of the dataset.</details>

### ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling

**Authors:** Quanyi Li, Zhenghao Peng, Lan Feng, Zhizheng Liu, Chenda Duan, Wenjie Mo, Bolei Zhou

**<details><summary>Abstract**</summary>

Large-scale driving datasets such as Waymo Open Dataset and nuScenes substantially accelerate autonomous driving research, especially for perception tasks such as 3D detection and trajectory forecasting. Since the driving logs in these datasets contain HD maps and detailed object annotations which accurately reflect the real-world complexity of traffic behaviors, we can harvest a massive number of complex traffic scenarios and recreate their digital twins in simulation. Compared to the hand-crafted scenarios often used in existing simulators, data-driven scenarios collected from the real world can facilitate many research opportunities in machine learning and autonomous driving. In this work, we present ScenarioNet, an open-source platform for large-scale traffic scenario modeling and simulation. ScenarioNet defines a unified scenario description format and collects a large-scale repository of real-world traffic scenarios from the heterogeneous data in various driving datasets including Waymo, nuScenes, Lyft L5, and nuPlan datasets. These scenarios can be further replayed and interacted with in multiple views from Bird-Eye-View layout to realistic 3D rendering in MetaDrive simulator. This provides a benchmark for evaluating the safety of autonomous driving stacks in simulation before their real-world deployment. We further demonstrate the strengths of ScenarioNet on large-scale scenario generation, imitation learning, and reinforcement learning in both single-agent and multi-agent settings. Code, demo videos, and website are available at https://github.com/metadriverse/scenarionet</details>

### Score-based Data Assimilation

**Authors:** François Rozet, Gilles Louppe

**<details><summary>Abstract**</summary>

Data assimilation, in its most comprehensive form, addresses the Bayesian inverse problem of identifying plausible state trajectories that explain noisy or incomplete observations of stochastic dynamical systems. Various approaches have been proposed to solve this problem, including particle-based and variational methods. However, most algorithms depend on the transition dynamics for inference, which becomes intractable for long time horizons or for high-dimensional systems with complex dynamics, such as oceans or atmospheres. In this work, we introduce score-based data assimilation for trajectory inference. We learn a score-based generative model of state trajectories based on the key insight that the score of an arbitrarily long trajectory can be decomposed into a series of scores over short segments. After training, inference is carried out using the score model, in a non-autoregressive manner by generating all states simultaneously. Quite distinctively, we decouple the observation model from the training procedure and use it only at inference to guide the generative process, which enables a wide range of zero-shot observation scenarios. We present theoretical and empirical evidence supporting the effectiveness of our method.</details>

### Selective Sampling and Imitation Learning via Online Regression

**Authors:** Ayush Sekhari, Karthik Sridharan, Wen Sun, Runzhe Wu

**<details><summary>Abstract**</summary>

We consider the problem of Imitation Learning (IL) by actively querying noisy expert for feedback. While imitation learning has been empirically successful, much of prior work assumes access to noiseless expert feedback which is not practical in many applications. In fact, when one only has access to noisy expert feedback, algorithms that rely on purely offline data (non-interactive IL) can be shown to need a prohibitively large number of samples to be successful. In contrast, in this work, we provide an interactive algorithm for IL that uses selective sampling to actively query the noisy expert for feedback. Our contributions are twofold: First,  we provide a new selective sampling algorithm that works with general function classes and multiple actions, and obtains the best-known bounds for the regret and the number of queries. Next, we extend this analysis to the problem of IL with noisy expert feedback and provide a new IL algorithm that  makes limited queries.  Our algorithm for selective sampling leverages function approximation, and relies on an online regression oracle w.r.t.~the given model class to predict actions, and to decide whether to query the expert for its label. On the theoretical side, the regret bound of our algorithm is upper bounded by the regret of the online regression oracle, while the query complexity additionally depends on the eluder dimension of the model class. We complement this with a  lower bound that demonstrates that our results are tight. We extend our selective sampling algorithm for IL with general function approximation and provide bounds on both the regret and the number of queries made to the noisy expert. A key novelty here is that our regret and query complexity bounds only depend on the number of times the optimal policy (and not the noisy expert, or the learner) go to states that have a small margin.</details>

### Self-Correcting Bayesian Optimization through Bayesian Active Learning

**Authors:** Carl Hvarfner, Erik Hellsten, Frank Hutter, Luigi Nardi

**<details><summary>Abstract**</summary>

Gaussian processes are the model of choice in Bayesian optimization and active learning. Yet, they are highly dependent on cleverly chosen hyperparameters to reach their full potential, and little effort is devoted to finding good hyperparameters in the literature. We demonstrate the impact of selecting good hyperparameters for GPs and present two acquisition functions that explicitly prioritize hyperparameter learning. Statistical distance-based Active Learning (SAL) considers the average disagreement between samples from the posterior, as measured by a statistical distance. SAL outperforms the state-of-the-art in Bayesian active learning on several test functions. We then introduce Self-Correcting Bayesian Optimization (SCoreBO), which extends SAL to perform Bayesian optimization and active learning simultaneously. SCoreBO learns the model hyperparameters at improved rates compared to vanilla BO, while outperforming the latest Bayesian optimization methods on traditional benchmarks. Moreover, we demonstrate the importance of self-correction on atypical Bayesian optimization tasks.</details>

### Semi-Implicit Denoising Diffusion Models (SIDDMs)

**Authors:** yanwu xu, Mingming Gong, Shaoan Xie, Wei Wei, Matthias Grundmann, Kayhan Batmanghelich, Tingbo Hou

**<details><summary>Abstract**</summary>

Despite the proliferation of generative models, achieving fast sampling during inference without compromising sample diversity and quality remains challenging. Existing models such as Denoising Diffusion Probabilistic Models (DDPM) deliver high-quality, diverse samples but are slowed by an inherently high number of iterative steps. The Denoising Diffusion Generative Adversarial Networks (DDGAN) attempted to circumvent this limitation by integrating a GAN model for larger jumps in the diffusion process. However, DDGAN encountered scalability limitations when applied to large datasets. To address these limitations, we introduce a novel approach that tackles the problem by matching implicit and explicit factors. More specifically, our approach involves utilizing an implicit model to match the marginal distributions of noisy data and the explicit conditional distribution of the forward diffusion. This combination allows us to effectively match the joint denoising distributions. Unlike DDPM but similar to DDGAN, we do not enforce a parametric distribution for the reverse step, enabling us to take large steps during inference. Similar to the DDPM but unlike DDGAN, we take advantage of the exact form of the diffusion process. We demonstrate that our proposed method obtains comparable generative performance to diffusion-based models and vastly superior results to models with a small number of sampling steps.</details>

### [Spotlight] Semi-Supervised Domain Generalization with Known and Unknown Classes

**Authors:** Lei Zhang, Ji-Fu Li, Wei Wang

**<details><summary>Abstract**</summary>

Semi-Supervised Domain Generalization (SSDG) aims to learn a model that is generalizable to an unseen target domain with only a few labels, and most existing SSDG methods assume that unlabeled training and testing samples are all known classes. However, a more realistic scenario is that known classes may be mixed with some unknown classes in unlabeled training and testing data. To deal with such a scenario, we propose the Class-Wise Adaptive Exploration and Exploitation (CWAEE) method. In particular, we explore unlabeled training data by using one-vs-rest classifiers and class-wise adaptive thresholds to detect known and unknown classes, and exploit them by adopting consistency regularization on augmented samples based on Fourier Transformation to improve the unseen domain generalization. The experiments conducted on real-world datasets verify the effectiveness and superiority of our method.</details>

### Sequential Memory with Temporal Predictive Coding

**Authors:** Mufeng Tang, Helen Barron, Rafal Bogacz

**<details><summary>Abstract**</summary>

Forming accurate memory of sequential stimuli is a fundamental function of biological agents. However, the computational mechanism underlying sequential memory in the brain remains unclear. Inspired by neuroscience theories and recent successes in applying predictive coding (PC) to \emph{static} memory tasks, in this work we propose a novel PC-based model for \emph{sequential} memory, called \emph{temporal predictive coding} (tPC). We show that our tPC models can memorize and retrieve sequential inputs accurately with a biologically plausible neural implementation. Importantly, our analytical study reveals that tPC can be viewed as a classical Asymmetric Hopfield Network (AHN) with an implicit statistical whitening process, which leads to more stable performance in sequential memory tasks of structured inputs. Moreover, we find that tPC exhibits properties consistent with behavioral observations and theories in neuroscience, thereby strengthening its biological relevance. Our work establishes a possible computational mechanism underlying sequential memory in the brain that can also be theoretically interpreted using existing memory model frameworks.</details>

### [Spotlight] Sharp Spectral Rates for Koopman Operator Learning

**Authors:** Vladimir Kostic, Karim Lounici, Pietro Novelli, Massimiliano Pontil

**<details><summary>Abstract**</summary>

Non-linear dynamical systems can be handily described by the associated Koopman operator, whose action evolves every observable of the system forward in time. Learning the Koopman operator and its spectral decomposition from data is enabled by a number of algorithms. In this work we present for the first time non-asymptotic learning bounds for the Koopman eigenvalues and eigenfunctions. We focus on time-reversal-invariant stochastic dynamical systems, including the important example of Langevin dynamics. We analyze two popular estimators: Extended Dynamic Mode Decomposition (EDMD) and Reduced Rank Regression (RRR). Our results critically hinge on novel {minimax} estimation bounds for the operator norm error, that may be of independent interest. Our spectral learning bounds are driven by the simultaneous control of the operator norm error and a novel metric distortion functional of the estimated eigenfunctions. The bounds indicates that both EDMD and RRR have similar variance, but EDMD suffers from a larger bias which might be detrimental to its learning rate. Our results shed new light on the emergence of spurious eigenvalues, an issue which is well known empirically. Numerical experiments illustrate the implications of the bounds in practice.</details>

### [Oral] Sharpness Minimization Algorithms Do Not Only Minimize Sharpness To Achieve Better Generalization

**Authors:** Kaiyue Wen, Zhiyuan Li, Tengyu Ma

**Oral Presentation:** Tu, Dec 12, 08:00 -- Oral 1D

**<details><summary>Abstract**</summary>

Despite extensive studies, the underlying reason as to why overparameterizedneural networks can generalize remains elusive. Existing theory shows that common stochastic optimizers prefer flatter minimizers of the training loss, and thusa natural potential explanation is that flatness implies generalization. This workcritically examines this explanation. Through theoretical and empirical investigation, we identify the following three scenarios for two-layer ReLU networks: (1)flatness provably implies generalization; (2) there exist non-generalizing flattestmodels and sharpness minimization algorithms fail to generalize poorly, and (3)perhaps most strikingly, there exist non-generalizing flattest models, but sharpnessminimization algorithms still generalize. Our results suggest that the relationshipbetween sharpness and generalization subtly depends on the data distributionsand the model architectures and sharpness minimization algorithms do not onlyminimize sharpness to achieve better generalization. This calls for the search forother explanations for the generalization of over-parameterized neural networks</details>

### SheetCopilot: Bringing Software Productivity to the Next Level through Large Language Models

**Authors:** Hongxin Li, Jingran Su, Yuntao Chen, Qing Li, ZHAO-XIANG ZHANG

**<details><summary>Abstract**</summary>

Computer end users have spent billions of hours completing daily tasks like tabular data processing and project timeline scheduling. Most of these tasks are repetitive and error-prone, yet most end users lack the skill to automate these burdensome works. With the advent of large language models (LLMs), directing software with natural language user requests become a reachable goal. In this work, we propose a SheetCopilot agent that takes natural language task and control spreadsheet to fulfill the requirements. We propose a set of atomic actions as an abstraction of spreadsheet software functionalities. We further design a state machine-based task planning framework for LLMs to robustly interact with spreadsheets. We curate a representative dataset containing 221 spreadsheet control tasks and establish a fully automated evaluation pipeline for rigorously benchmarking the ability of LLMs in software control tasks. Our SheetCopilot correctly completes 44.3\% of tasks for a single generation, outperforming the strong code generation baseline by a wide margin. Our project page: https://sheetcopilot.github.io/.</details>

### Should We Learn Most Likely Functions or Parameters?

**Authors:** Shikai Qiu, Tim G. J. Rudner, Sanyam Kapoor, Andrew Wilson

**<details><summary>Abstract**</summary>

Standard regularized training procedures correspond to maximizing a posterior distribution over parameters, known as maximum a posteriori (MAP) estimation. However, model parameters are of interest only insomuch as they combine with the functional form of a model to provide a function that can make good predictions. Moreover, the most likely parameters under the parameter posterior do not generally correspond to the most likely function induced by the parameter posterior. In fact, we can re-parametrize a model such that any setting of parameters can maximize the parameter posterior. As an alternative, we investigate the benefits and drawbacks of directly estimating the most likely function implied by the model and the data. We show that this procedure leads to pathological solutions when using neural networks and prove conditions under which the procedure is well-behaved, as well as a scalable approximation. Under these conditions, we find that function-space MAP estimation can lead to flatter minima, better generalization, and improved robustness to overfitting.</details>

### Similarity, Compression and Local Steps: Three Pillars of Efficient Communications for Distributed Variational Inequalities

**Authors:** Aleksandr Beznosikov, Martin Takac, Alexander Gasnikov

**<details><summary>Abstract**</summary>

Variational inequalities are a broad and flexible class of problems that includes minimization, saddle point, and fixed point problems as special cases. Therefore, variational inequalities are used in various applications ranging from equilibrium search to adversarial learning. With the increasing size of data and models, today's instances demand parallel and distributed computing for real-world machine learning problems, most of which can be represented as variational inequalities. Meanwhile, most distributed approaches have a significant bottleneck -- the cost of communications. The three main techniques to reduce the total number of communication rounds and the cost of one such round are the similarity of local functions, compression of transmitted information, and local updates. In this paper, we combine all these approaches. Such a triple synergy did not exist before for variational inequalities and saddle problems, nor even for minimization problems. The methods presented in this paper have the best theoretical guarantees of communication complexity and are significantly ahead of other methods for distributed variational inequalities. The theoretical results are confirmed by adversarial learning experiments on synthetic and real datasets.</details>

### SituatedGen: Incorporating Geographical and Temporal Contexts into Generative Commonsense Reasoning

**Authors:** Yunxiang Zhang, Xiaojun Wan

**<details><summary>Abstract**</summary>

Recently, commonsense reasoning in text generation has attracted much attention. Generative commonsense reasoning is the task that requires machines, given a group of keywords, to compose a single coherent sentence with commonsense plausibility. While existing datasets targeting generative commonsense reasoning focus on everyday scenarios, it is unclear how well machines reason under specific geographical and temporal contexts. We formalize this challenging task as SituatedGen, where machines with commonsense should generate a pair of contrastive sentences given a group of keywords including geographical or temporal entities. We introduce a corresponding English dataset consisting of 8,268 contrastive sentence pairs, which are built upon several existing commonsense reasoning benchmarks with minimal manual labor. Experiments show that state-of-the-art generative language models struggle to generate sentences with commonsense plausibility and still lag far behind human performance. Our dataset is publicly available at https://github.com/yunx-z/situated_gen.</details>

### Slimmed Asymmetrical Contrastive Learning and Cross Distillation for Lightweight Model Training

**Authors:** Jian Meng, Li Yang, Kyungmin Lee, Jinwoo Shin, Deliang Fan, Jae-sun Seo

**<details><summary>Abstract**</summary>

Contrastive learning (CL) has been widely investigated with various learning mechanisms and achieves strong capability in learning representations of data in a self-supervised manner using unlabeled data. A common fashion of contrastive learning on this line is employing mega-sized encoders to achieve comparable performance as the supervised learning counterpart. Despite the success of the labelless training, current contrastive learning algorithms *failed* to achieve good performance with lightweight (compact) models, e.g., MobileNet, while the requirements of the heavy encoders impede the energy-efficient computation, especially for resource-constrained AI applications. Motivated by this, we propose a new self-supervised CL scheme, named SACL-XD, consisting of two technical components, **S**limmed **A**symmetrical **C**ontrastive **L**earning (SACL) and **Cross**-**D**istillation (XD), which collectively enable efficient CL with compact models. While relevant prior works employed a strong pre-trained model as the teacher of unsupervised knowledge distillation to a lightweight encoder, our proposed method trains CL models from scratch and outperforms them even without such an expensive requirement. Compared to the SoTA lightweight CL training (distillation) algorithms, SACL-XD achieves 1.79% ImageNet-1K accuracy improvement on MobileNet-V3 with 64$\times$ training FLOPs reduction.</details>

### [Spotlight] Slow and Weak Attractor Computation Embedded in Fast and Strong E-I Balanced Neural Dynamics

**Authors:** Xiaohan Lin, Liyuan Li, Boxin Shi, Tiejun Huang, Yuanyuan Mi, Si Wu

**<details><summary>Abstract**</summary>

Attractor networks require neuronal connections to be highly structured in order to maintain attractor states that represent information, while excitation and inhibition balanced networks (E-INNs) require neuronal connections to be random and sparse to generate irregular neuronal firings. Despite being regarded as canonical models of neural circuits, both types of networks are usually studied in isolation, and it remains unclear how they coexist in the brain, given their very different structural demands. In this study, we investigate the compatibility of continuous attractor neural networks (CANNs) and E-INNs. In line with recent experimental data, we find that a neural circuit can exhibit both the traits of CANNs and E-INNs if the neuronal synapses consist of two sets: one set is strong and fast for irregular firing, and the other set is weak and slow for attractor dynamics. Our results from simulations and theoretical analysis reveal that the network also exhibits enhanced performance compared to the case of using only one set of synapses, with accelerated convergence of attractor states and retained E-I balanced condition for localized input. We also apply the network model to solve a real-world tracking problem and demonstrate that it can track fast-moving objects well. We hope that this study provides insight into how structured neural computations are realized by irregular firings of neurons.</details>

### Small Transformers Compute Universal Metric Embeddings

**Authors:** Anastasis Kratsios, Valentin Debarnot, Ivan Dokmanić

**<details><summary>Abstract**</summary>

We study representations of data from an arbitrary metric space $\mathcal{X}$ in the space of univariate Gaussian mixtures equipped with a transport metric (Delon and Desolneux 2020).  We prove embedding guarantees for feature maps implemented by small neural networks called probabilistic transformers.  Our guarantees are of memorization type: we prove that a probabilistic transformer of depth about $n\log(n)$ and width about $n^2$ can bi-H\"older embed any $n$-point dataset from $\mathcal{X}$ with low metric distortion, thus avoiding the curse of dimensionality.  We further derive probabilistic bi-Lipschitz guarantees, which trade off the amount of distortion and the probability that a randomly chosen pair of points embeds with that distortion.  If the geometry of $\mathcal{X}$ is sufficiently regular, we obtain stronger bi-Lipschitz guarantees for all points.  As applications, we derive neural embedding guarantees for datasets from Riemannian manifolds, metric trees, and certain types of combinatorial graphs. When instead embedding into multivariate Gaussian mixtures, we show that probabilistic transformers compute bi-Hölder embeddings with arbitrarily small distortion.  Our results show that any finite metric dataset, from vertices on a graph to functions a function space, can be faithfully represented in a single representation space, and that the representation can be implemented by a simple transformer architecture. Thus one may only need a modular set of machine learning tools compatible with this one representation space, many of which already exist, for downstream supervised and unsupervised learning from a great variety of data types.</details>

### Smooth, exact rotational symmetrization for deep learning on point clouds

**Authors:** Sergey Pozdnyakov, Michele Ceriotti

**<details><summary>Abstract**</summary>

Point clouds are versatile representations of 3D objects and have found widespread application in science and engineering. Many successful deep-learning models have been proposed that use them as input. The domain of chemical and materials modeling is especially challenging because exact compliance with physical constraints is highly desirable for a model to be usable in practice. These constraints include smoothness and invariance with respect to translations, rotations, and permutations of identical atoms. If these requirements are not rigorously fulfilled, atomistic simulations might lead to absurd outcomes even if the model has excellent accuracy. Consequently, dedicated architectures, which achieve invariance by restricting their design space, have been developed. General-purpose point-cloud models are more varied but often disregard rotational symmetry. We propose a general symmetrization method that adds rotational equivariance to any given model while preserving all the other requirements.Our approach simplifies the development of better atomic-scale ML schemes by relaxing the constraints on the design space and making it possible to incorporate ideas that proved effective in other domains.We demonstrate this idea by introducing the Point Edge Transformer (PET) architecture, which is not intrinsically equivariant but achieves state-of-the-art performance on several benchmark datasets of molecules and solids. A-posteriori application of our general protocol makes PET exactly equivariant, with minimal changes to its accuracy.</details>

### SmoothHess: ReLU Network Feature Interactions via Stein's Lemma

**Authors:** Max Torop, Aria Masoomi, Davin Hill, Kivanc Kose, Stratis Ioannidis, Jennifer Dy

**<details><summary>Abstract**</summary>

Several recent methods for interpretability model feature interactions by looking at the Hessian of a neural network. This poses a challenge for ReLU networks, which are piecewise-linear and thus have a zero Hessian almost everywhere. We propose SmoothHess, a method of estimating second-order interactions through Stein's Lemma. In particular, we estimate the Hessian of the network convolved with a Gaussian through an efficient sampling algorithm, requiring only network gradient calls. SmoothHess is applied post-hoc, requires no modifications to the ReLU network architecture, and the extent of smoothing can be controlled explicitly. We provide a non-asymptotic bound on the sample complexity of our estimation procedure. We validate the superior ability of SmoothHess to capture interactions on benchmark datasets and a real-world medical spirometry dataset.</details>

### [Spotlight] Smoothed Analysis of Sequential Probability Assignment

**Authors:** Alankrita Bhatt, Nika Haghtalab, Abhishek Shetty

**<details><summary>Abstract**</summary>

We initiate the study of smoothed analysis for the sequential probability assignment problem with contexts. We study information-theoretically optimal minmax rates as well as a framework for algorithmic reduction involving the maximum likelihood estimator oracle. Our approach establishes a general-purpose reduction from minimax rates for sequential probability assignment for smoothed adversaries to minimax rates for transductive learning. This leads to optimal (logarithmic) fast rates for parametric classes and classes with finite VC dimension. On the algorithmic front, we develop an algorithm that efficiently taps into the MLE oracle, for general classes of functions. We show that under general conditions this algorithmic approach yields sublinear regret.</details>

### SoTTA: Robust Test-Time Adaptation on Noisy Data Streams

**Authors:** Taesik Gong, Yewon Kim, Taeckyung Lee, Sorn Chottananurak, Sung-Ju Lee

**<details><summary>Abstract**</summary>

Test-time adaptation (TTA) aims to address distributional shifts between training and testing data using only unlabeled test data streams for continual model adaptation. However, most TTA methods assume benign test streams, while test samples could be unexpectedly diverse in the wild. For instance, an unseen object or noise could appear in autonomous driving. This leads to a new threat to existing TTA algorithms; we found that prior TTA algorithms suffer from those noisy test samples as they blindly adapt to incoming samples. To address this problem, we present Screening-out Test-Time Adaptation (SoTTA), a novel TTA algorithm that is robust to noisy samples. The key enabler of SoTTA is two-fold: (i) input-wise robustness via high-confidence uniform-class sampling that effectively filters out the impact of noisy samples and (ii) parameter-wise robustness via entropy-sharpness minimization that improves the robustness of model parameters against large gradients from noisy samples. Our evaluation with standard TTA benchmarks with various noisy scenarios shows that our method outperforms state-of-the-art TTA methods under the presence of noisy samples and achieves comparable accuracy to those methods without noisy samples. The source code is available at https://github.com/taeckyung/SoTTA.</details>

### SoundCam: A Dataset for Finding Humans Using Room Acoustics

**Authors:** Mason Wang, Samuel Clarke, Jui-Hsien Wang, Ruohan Gao, Jiajun Wu

**<details><summary>Abstract**</summary>

A room’s acoustic properties are a product of the room’s geometry, the objects within the room, and their specific positions. A room’s acoustic properties can be characterized by its impulse response (RIR) between a source and listener location, or roughly inferred from recordings of natural signals present in the room. Variations in the positions of objects in a room can effect measurable changes in the room’s acoustic properties, as characterized by the RIR. Existing datasets of RIRs either do not systematically vary positions of objects in an environment, or they consist of only simulated RIRs. We present SoundCam, the largest dataset of unique RIRs from in-the-wild rooms publicly released to date. It includes 5,000 10-channel real-world measurements of room impulse responses and 2,000 10-channel recordings of music in three different rooms, including a controlled acoustic lab, an in-the-wild living room, and a conference room, with different humans in positions throughout each room. We show that these measurements can be used for interesting tasks, such as detecting and identifying humans, and tracking their positions.</details>

### Species196: A One-Million Semi-supervised Dataset for Fine-grained Species Recognition

**Authors:** Wei He, Kai Han, Ying Nie, Chengcheng Wang, Yunhe Wang

**<details><summary>Abstract**</summary>

The development of foundation vision models has pushed the general visual recognition to a high level, but cannot well address the fine-grained recognition in specialized domain such as invasive species classification. Identifying and managing invasive species has strong social and ecological value. Currently, most invasive species datasets are limited in scale and cover a narrow range of species, which restricts the development of deep-learning based invasion biometrics systems. To fill the gap of this area, we introduced Species196, a large-scale semi-supervised dataset of 196-category invasive species. It collects over 19K images with expert-level accurate annotations (Species196-L), and 1.2M unlabeled images of invasive species (Species196-U). The dataset provides four experimental settings for benchmarking the existing models and algorithms, namely, supervised learning, semi-supervised learning and self-supervised pretraining. To facilitate future research on these four learning paradigms, we conduct an empirical study of the representative methods on the introduced dataset. The dataset will be made publicly available at https://species-dataset.github.io/.</details>

### SpokenWOZ: A Large-Scale Speech-Text Benchmark for Spoken Task-Oriented Dialogue Agents

**Authors:** Shuzheng Si, Wentao Ma, Haoyu Gao, Yuchuan Wu, Ting-En Lin, Yinpei Dai, Hangyu Li, Rui Yan, Fei Huang, Yongbin Li

**<details><summary>Abstract**</summary>

Task-oriented dialogue (TOD) models have made significant progress in recent years. However, previous studies primarily focus on datasets written by annotators, which has resulted in a gap between academic research and real-world spoken con- versation scenarios. While several small-scale spoken TOD datasets are proposed to address robustness issues such as ASR errors, they ignore the unique challenges in spoken conversation. To tackle the limitations, we introduce SpokenWOZ, a large-scale speech-text dataset for spoken TOD, containing 8 domains, 203k turns, 5.7k dialogues and 249 hours of audios from human-to-human spoken conversations. SpokenWOZ further incorporates common spoken characteristics such as word-by-word processing and reasoning in spoken language. Based on these characteristics, we present cross-turn slot and reasoning slot detection as new challenges. We conduct experiments on various baselines, including text-modal models, newly proposed dual-modal models, and LLMs, e.g., ChatGPT. The results show that the current models still have substantial room for improvement in spoken conversation, where the most advanced dialogue state tracker only achieves 25.65% in joint goal accuracy and the SOTA end-to-end model only correctly completes the user request in 52.1% of dialogues. Our dataset, code, and leaderboard are available at https://spokenwoz.github.io/SpokenWOZ-github.io/.</details>

### Stability Guarantees for Feature Attributions with Multiplicative Smoothing

**Authors:** Anton Xue, Rajeev Alur, Eric Wong

**<details><summary>Abstract**</summary>

Explanation methods for machine learning models tend not to provide any formal guarantees and may not reflect the underlying decision-making process.In this work, we analyze stability as a property for reliable feature attribution methods. We prove that relaxed variants of stability are guaranteed if the model is sufficiently Lipschitz with respect to the masking of features. We develop a smoothing method called Multiplicative Smoothing (MuS) to achieve such a model.We show that MuS overcomes the theoretical limitations of standard smoothing techniques and can be integrated with any classifier and feature attribution method.We evaluate MuS on vision and language models with various feature attribution methods, such as LIME and SHAP, and demonstrate that MuS endows feature attributions with non-trivial stability guarantees.</details>

### Stabilized Neural Differential Equations for Learning Dynamics with Explicit Constraints

**Authors:** Alistair White, Niki Kilbertus, Maximilian Gelbrecht, Niklas Boers

**<details><summary>Abstract**</summary>

Many successful methods to learn dynamical systems from data have recently been introduced. However, ensuring that the inferred dynamics preserve known constraints, such as conservation laws or restrictions on the allowed system states, remains challenging. We propose stabilized neural differential equations (SNDEs), a method to enforce arbitrary manifold constraints for neural differential equations. Our approach is based on a stabilization term that, when added to the original dynamics, renders the constraint manifold provably asymptotically stable. Due to its simplicity, our method is compatible with all common neural differential equation (NDE) models and broadly applicable. In extensive empirical evaluations, we demonstrate that SNDEs outperform existing methods while broadening the types of constraints that can be incorporated into NDE training.</details>

### Stabilizing the Optimization of Neural Signed Distance Functions and Finer Shape Representation

**Authors:** Huizong Yang, Yuxin Sun, Ganesh Sundaramoorthi, Anthony Yezzi

**<details><summary>Abstract**</summary>

We present new insights and a novel paradigm for learning implicit neural representations (INR) of shapes. In particular, we shed light on the popular eikonal loss used for imposing a signed distance function constraint in INR. We show analytically that as the representation power of the network increases, the optimization approaches a partial differential equation (PDE) in the continuum limit that is unstable. We show that this instability can manifest in existing network optimization, leading to irregularities in the reconstructed surface and/or convergence to sub-optimal local minima, and thus fails to capture fine geometric and topological structure. We show analytically how other terms added to the loss, currently used in the literature for other purposes, can actually eliminate these instabilities. However, such terms can over-regularize the surface, preventing the representation of fine shape detail. Based on a similar PDE theory for the continuum limit, we introduce a new regularization term that still counteracts the eikonal instability but without over-regularizing. Furthermore, since stability is now guaranteed in the continuum limit, this stabilization also allows for considering new network structures that are able to represent finer shape detail. We introduce such a structure based on quadratic layers. Experiments on multiple benchmark data sets show that our new regularization and network are able to capture more precise shape details and more accurate topology than existing state-of-the-art.</details>

### Stable Bias: Evaluating Societal Representations in Diffusion Models

**Authors:** Sasha Luccioni, Christopher Akiki, Margaret Mitchell, Yacine Jernite

**<details><summary>Abstract**</summary>

As machine learning-enabled Text-to-Image (TTI) systems are becoming increasingly prevalent and seeing growing adoption as commercial services, characterizing the social biases they exhibit is a necessary first step to lowering their risk of discriminatory outcomes. This evaluation, however, is made more difficult by the synthetic nature of these systems’ outputs: common definitions of diversity are grounded in social categories of people living in the world, whereas the artificial depictions of fictive humans created by these systems have no inherent gender or ethnicity. To address this need, we propose a new method for exploring the social biases in TTI systems. Our approach relies on characterizing the variation in generated images triggered by enumerating gender and ethnicity markers in the prompts, and comparing it to the variation engendered by spanning different professions. This allows us to (1) identify specific bias trends, (2) provide targeted scores to directly compare models in terms of diversity and representation, and (3) jointly model interdependent social variables to support a multidimensional analysis. We leverage this method to analyze images generated by 3 popular TTI systems (Dall·E 2 , Stable Diffusion v 1.4 and 2) and find that while all of their outputs show correlations with US labor demographics, they also consistently under-represent marginalized identities to different extents. We also release the datasets and low-code interactive bias exploration platforms developed forthis work, as well as the necessary tools to similarly evaluate additional TTI systems.</details>

### [Spotlight] Stable Nonconvex-Nonconcave Training via Linear Interpolation

**Authors:** Thomas Pethick, Wanyun Xie, Volkan Cevher

**<details><summary>Abstract**</summary>

This paper presents a theoretical analysis of linear interpolation as a principled method for stabilizing (large-scale) neural network training. We argue that instabilities in the optimization process are often caused by the nonmonotonicity of the loss landscape and show how linear interpolation can help by leveraging the theory of nonexpansive operators. We construct a new optimization scheme called relaxed approximate proximal point (RAPP), which is the first explicit method to achieve last iterate convergence rates for the full range of cohypomonotone problems. The construction extends to constrained and regularized settings. By replacing the inner optimizer in RAPP we rediscover the family of Lookahead algorithms for which we establish convergence in cohypomonotone problems even when the base optimizer is taken to be gradient descent ascent. The range of cohypomonotone problems in which Lookahead converges is further expanded by exploiting that Lookahead inherits the properties of the base optimizer. We corroborate the results with experiments on generative adversarial networks which demonstrates the benefits of the linear interpolation present in both RAPP and Lookahead.</details>

### State2Explanation: Concept-Based Explanations to Benefit Agent Learning and User Understanding

**Authors:** Devleena Das, Sonia Chernova, Been Kim

**<details><summary>Abstract**</summary>

As more non-AI experts use complex AI systems for daily tasks, there has been an increasing effort to develop methods that produce explanations of AI decision making that are understandable by non-AI experts. Towards this effort, leveraging higher-level concepts and producing concept-based explanations have become a popular method. Most concept-based explanations have been developed for classification techniques, and we posit that the few existing methods for sequential decision making are limited in scope. In this work, we first contribute a desiderata for defining ``concepts'' in sequential decision making settings. Additionally, inspired by the Protege Effect which states explaining knowledge often reinforces one's self-learning, we explore how concept-based explanations of an RL agent's decision making can in turn improve the agent's learning rate, as well as improve end-user understanding of the agent's decision making. To this end, we contribute a unified framework, State2Explanation (S2E), that involves learning a joint embedding model between state-action pairs and concept-based explanations, and leveraging such learned model to both (1) inform reward shaping during an agent's training, and (2) provide explanations to end-users at deployment for improved task performance. Our experimental validations, in Connect 4 and Lunar Lander, demonstrate the success of S2E in providing a dual-benefit, successfully informing reward shaping and improving agent learning rate, as well as significantly improving end user task performance at deployment time.</details>

### StateMask: Explaining Deep Reinforcement Learning through State Mask

**Authors:** Zelei Cheng, Xian Wu, Jiahao Yu, Wenhai Sun, Wenbo Guo, Xinyu Xing

**<details><summary>Abstract**</summary>

Despite the promising performance of deep reinforcement learning (DRL) agents in many challenging scenarios, the black-box nature of these agents greatly limits their applications in critical domains. Prior research has proposed several explanation techniques to understand the deep learning-based policies in RL. Most existing methods explain why an agent takes individual actions rather than pinpointing the critical steps to its final reward. To fill this gap, we propose StateMask, a novel method to identify the states most critical to the agent's final reward. The high-level idea of StateMask is to learn a mask net that blinds a target agent and forces it to take random actions at some steps without compromising the agent's performance. Through careful design, we can theoretically ensure that the masked agent performs similarly to the original agent. We evaluate StateMask in various popular RL environments and show its superiority over existing explainers in explanation fidelity. We also show that StateMask  has better utilities, such as launching adversarial attacks and patching policy errors.</details>

### [Spotlight] Statistical Guarantees for Variational Autoencoders using PAC-Bayesian Theory

**Authors:** Sokhna Diarra Mbacke, Florence Clerc, Pascal Germain

**<details><summary>Abstract**</summary>

Since their inception, Variational Autoencoders (VAEs) have become central in machine learning. Despite their widespread use, numerous questions regarding their theoretical properties remain open. Using PAC-Bayesian theory, this work develops statistical guarantees for VAEs. First, we derive the first PAC-Bayesian bound for posterior distributions conditioned on individual samples from the data-generating distribution. Then, we utilize this result to develop generalization guarantees for the VAE's reconstruction loss, as well as upper bounds on the distance between the input and the regenerated distributions. More importantly, we provide upper bounds on the Wasserstein distance between the input distribution and the distribution defined by the VAE's generative model.</details>

### [Spotlight] Stochastic Multi-armed Bandits: Optimal Trade-off among Optimality, Consistency, and Tail Risk

**Authors:** David Simchi-Levi, Zeyu Zheng, Feng Zhu

**<details><summary>Abstract**</summary>

We consider the stochastic multi-armed bandit problem and fully characterize the interplays among three desired properties for policy design: worst-case optimality, instance-dependent consistency, and light-tailed risk. We show how the order of expected regret exactly affects the decaying rate of the regret tail probability for both the worst-case and instance-dependent scenario. A novel policy is proposed to achieve the optimal regret tail risk for any regret threshold. Concretely, for any given $\alpha\in[1/2, 1)$ and $\beta\in[0, 1)$, our policy achieves a worst-case expected regret of $\tilde O(T^\alpha)$ and instance-dependent expected regret of $\tilde O(T^\beta)$, while enjoys a probability of incurring an $\Omega(T^\delta)$ regret that decays exponentially with a polynomial $T$ term. Such decaying rate is proved to be best achievable. We also generalize our analysis to the stochastic multi-armed bandit problem with non-stationary baseline rewards, where in each time period $t$, the decision maker pulls one of $K$ arms and collects a reward which is the sum of three terms: the mean of the pulled arm, an independent noise, and a non-stationary baseline reward as a function of $t$. Our results reveal insights on the trade-off between expected regret and tail risk for both worst-case and instance-dependent scenario, indicating that more sub-optimality and inconsistency leaves space for more light-tailed risk of incurring a large regret.</details>

### StoryBench: A Multifaceted Benchmark for Continuous Story Visualization

**Authors:** Emanuele Bugliarello, H. Hernan Moraldo, Ruben Villegas, Mohammad Babaeizadeh, Mohammad Taghi Saffar, Han Zhang, Dumitru Erhan, Vittorio Ferrari, Pieter-Jan Kindermans, Paul Voigtlaender

**<details><summary>Abstract**</summary>

Generating video stories from text prompts is a complex task. In addition to having high visual quality, videos need to realistically adhere to a sequence of text prompts whilst being consistent throughout the frames. Creating a benchmark for video generation requires data annotated over time, which contrasts with the single caption used often in video datasets. To fill this gap, we collect comprehensive human annotations on three existing datasets, and introduce StoryBench: a new, challenging multi-task benchmark to reliably evaluate forthcoming text-to-video models. Our benchmark includes three video generation tasks of increasing difficulty: action execution, where the next action must be generated starting from a conditioning video; story continuation, where a sequence of actions must be executed starting from a conditioning video; and story generation, where a video must be generated from only text prompts. We evaluate small yet strong text-to-video baselines, and show the benefits of training on story-like data algorithmically generated from existing video captions. Finally, we establish guidelines for human evaluation of video stories, and reaffirm the need of better automatic metrics for video generation. StoryBench aims at encouraging future research efforts in this exciting new area.</details>

### StressID: a Multimodal Dataset for Stress Identification

**Authors:** Hava Chaptoukaev, Valeriya Strizhkova, Michele Panariello, Bianca Dalpaos, Aglind Reka, Valeria Manera, Susanne Thümmler, Esma ISMAILOVA, Nicholas W., francois bremond, Massimiliano Todisco, Maria A Zuluaga, Laura M. Ferrari

**<details><summary>Abstract**</summary>

StressID is a new dataset specifically designed for stress identification fromunimodal and multimodal data. It contains videos of facial expressions, audiorecordings, and physiological signals. The video and audio recordings are acquiredusing an RGB camera with an integrated microphone. The physiological datais composed of electrocardiography (ECG), electrodermal activity (EDA), andrespiration signals that are recorded and monitored using a wearable device. Thisexperimental setup ensures a synchronized and high-quality multimodal data col-lection. Different stress-inducing stimuli, such as emotional video clips, cognitivetasks including mathematical or comprehension exercises, and public speakingscenarios, are designed to trigger a diverse range of emotional responses. Thefinal dataset consists of recordings from 65 participants who performed 11 tasks,as well as their ratings of perceived relaxation, stress, arousal, and valence levels.StressID is one of the largest datasets for stress identification that features threedifferent sources of data and varied classes of stimuli, representing more than39 hours of annotated data in total. StressID offers baseline models for stressclassification including a cleaning, feature extraction, and classification phase foreach modality. Additionally, we provide multimodal predictive models combiningvideo, audio, and physiological inputs. The data and the code for the baselines areavailable at https://project.inria.fr/stressid/.</details>

### Structure Learning with Adaptive Random Neighborhood Informed MCMC

**Authors:** Xitong Liang, Alberto Caron, Samuel Livingstone, Jim Griffin

**<details><summary>Abstract**</summary>

In this paper, we introduce a novel MCMC sampler, PARNI-DAG, for a fully-Bayesian approach to the problem of structure learning under observational data. Under the assumption of causal sufficiency, the algorithm allows for approximate sampling directly from the posterior distribution on Directed Acyclic Graphs (DAGs). PARNI-DAG performs efficient sampling of DAGs via locally informed, adaptive random neighborhood proposal that results in better mixing properties. In addition, to ensure better scalability with the number of nodes, we couple PARNI-DAG with a pre-tuning procedure of the sampler's parameters that exploits a skeleton graph derived through some constraint-based or scoring-based algorithms. Thanks to these novel features, PARNI-DAG quickly converges to high-probability regions and is less likely to get stuck in local modes in the presence of high correlation between nodes in high-dimensional settings. After introducing the technical novelties in PARNI-DAG, we empirically demonstrate its mixing efficiency and accuracy in learning DAG structures on a variety of experiments.</details>

### Subject-driven Text-to-Image Generation via Apprenticeship Learning

**Authors:** wenhu chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, William Cohen

**<details><summary>Abstract**</summary>

Recent text-to-image generation models like DreamBooth have made remarkable progress in generating highly customized images of a target subject, by fine-tuning an ``expert model'' for a given subject from a few examples.However, this process is expensive, since a new expert model must be learned for each subject. In this paper, we present SuTI, a Subject-driven Text-to-Image generator that replaces subject-specific fine tuning with {in-context} learning.Given a few demonstrations of a new subject, SuTI can instantly generate novel renditions of the subject in different scenes, without any subject-specific optimization.SuTI is powered by {apprenticeship learning}, where a single apprentice model is learned from data generated by a massive number of subject-specific expert models. Specifically, we mine millions of image clusters from the Internet, each centered around a specific visual subject. We adopt these clusters to train a massive number of expert models, each specializing in a different subject. The apprentice model SuTI then learns to imitate the behavior of these fine-tuned experts. SuTI can generate high-quality and customized subject-specific images 20x faster than optimization-based SoTA methods. On the challenging DreamBench and DreamBench-v2, our human evaluation shows that SuTI significantly outperforms existing models like InstructPix2Pix, Textual Inversion, Imagic, Prompt2Prompt, Re-Imagen and DreamBooth.</details>

### SubseasonalClimateUSA: A Dataset for Subseasonal Forecasting and Benchmarking

**Authors:** Soukayna Mouatadid, Paulo Orenstein, Genevieve Flaspohler, Miruna Oprescu, Judah Cohen, Franklyn Wang, Sean Knight, Maria Geogdzhayeva, Sam Levang, Ernest Fraenkel, Lester Mackey

**<details><summary>Abstract**</summary>

Subseasonal forecasting of the weather two to six weeks in advance is critical for resource allocation and climate adaptation but poses many challenges for the forecasting community. At this forecast horizon, physics-based dynamical models have limited skill, and the targets for prediction depend in a complex manner on both local weather variables and global climate variables. Recently, machine learning methods have shown promise in advancing the state of the art but only at the cost of complex data curation, integrating expert knowledge with aggregation across multiple relevant data sources, file formats, and temporal and spatial resolutions.To streamline this process and accelerate future development, we introduce SubseasonalClimateUSA, a curated dataset for training and benchmarking subseasonal forecasting models in the United States. We use this dataset to benchmark a diverse suite of models, including operational dynamical models, classical meteorological baselines, and ten state-of-the-art machine learning and deep learning-based methods from the literature. Overall, our benchmarks suggest simple and effective ways to extend the accuracy of current operational models. SubseasonalClimateUSA is regularly updated and accessible via the https://github.com/microsoft/subseasonal_data/ Python package.</details>

### [Spotlight] Subspace Identification for Multi-Source Domain Adaptation

**Authors:** Zijian Li, Ruichu Cai, Guangyi Chen, Boyang Sun, Zhifeng Hao, Kun Zhang

**<details><summary>Abstract**</summary>

Multi-source domain adaptation (MSDA) methods aim to transfer knowledge from multiple labeled source domains to an unlabeled target domain. Although current methods achieve target joint distribution identifiability by enforcing minimal changes across domains, they often necessitate stringent conditions, such as an adequate number of domains, monotonic transformation of latent variables, and invariant label distributions. These requirements are challenging to satisfy in real-world applications. To mitigate the need for these strict assumptions, we propose a subspace identification theory that guarantees the disentanglement of domain-invariant and domain-specific variables under less restrictive constraints regarding domain numbers and transformation properties and thereby facilitating domain adaptation by minimizing the impact of domain shifts on invariant variables. Based on this theory, we develop a Subspace Identification Guarantee (SIG) model that leverages variational inference. Furthermore, the SIG model incorporates class-aware conditional alignment to accommodate target shifts where label distributions change with the domain. Experimental results demonstrate that our SIG model outperforms existing MSDA techniques on various benchmark datasets, highlighting its effectiveness in real-world applications.</details>

### [Spotlight] Supervised Pretraining Can Learn In-Context Reinforcement Learning

**Authors:** Jonathan Lee, Annie Xie, Aldo Pacchiano, Yash Chandak, Chelsea Finn, Ofir Nachum, Emma Brunskill

**<details><summary>Abstract**</summary>

Large transformer models trained on diverse datasets have shown a remarkable ability to learn in-context, achieving high few-shot performance on tasks they were not explicitly trained to solve. In this paper, we study the in-context learning capabilities of transformers in decision-making problems, i.e., reinforcement learning (RL) for bandits and Markov decision processes. To do so, we introduce and study the Decision-Pretrained Transformer (DPT), a supervised pretraining method where a transformer predicts an optimal action given a query state and an in-context dataset of interactions from a diverse set of tasks. While simple, this procedure produces a model with several surprising capabilities. We find that the trained transformer can solve a range of RL problems in-context, exhibiting both exploration online and conservatism offline, despite not being explicitly trained to do so. The model also generalizes beyond the pretraining distribution to new tasks and automatically adapts its decision-making strategies to unknown structure. Theoretically, we show DPT can be viewed as an efficient implementation of Bayesian posterior sampling, a provably sample-efficient RL algorithm. We further leverage this connection to provide guarantees on the regret of the in-context algorithm yielded by DPT, and prove that it can learn faster than algorithms used to generate the pretraining data. These results suggest a promising yet simple path towards instilling strong in-context decision-making abilities in transformers.</details>

### Supply-Side Equilibria in Recommender Systems

**Authors:** Meena Jagadeesan, Nikhil Garg, Jacob Steinhardt

**<details><summary>Abstract**</summary>

Algorithmic recommender systems such as Spotify and Netflix affect not only consumer behavior but also *producer incentives*. Producers seek to create content that will be shown by the recommendation algorithm, which can impact both the diversity and quality of their content. In this work, we investigate the resulting supply-side equilibria in personalized content recommender systems. We model the decisions of producers as choosing *multi-dimensional* content vectors and users as having *heterogenous* preferences, which contrasts with classical low-dimensional models. Multi-dimensionality and heterogeneity creates the potential for *specialization*, where different producers create different types of content at equilibrium. Using a duality argument, we derive necessary and sufficient conditions for whether specialization occurs. Then, we characterize the distribution of content at equilibrium in concrete settings with two populations of users. Lastly, we show that specialization can enable producers to achieve *positive profit at equilibrium*, which means that specialization can reduce the competitiveness of the marketplace. At a conceptual level, our analysis of supply-side competition takes a step towards elucidating how personalized recommendations shape the marketplace of digital goods.</details>

### SustainGym: Reinforcement Learning Environments for Sustainable Energy Systems

**Authors:** Christopher Yeh, Victor Li, Rajeev Datta, Julio Arroyo, Nicolas Christianson, Chi Zhang, Yize Chen, Mohammad Mehdi Hosseini, Azarang Golmohammadi, Yuanyuan Shi, Yisong Yue, Adam Wierman

**<details><summary>Abstract**</summary>

The lack of standardized benchmarks for reinforcement learning (RL) in sustainability applications has made it difficult to both track progress on specific domains and identify bottlenecks for researchers to focus their efforts. In this paper, we present SustainGym, a suite of five environments designed to test the performance of RL algorithms on realistic sustainable energy system tasks, ranging from electric vehicle charging to carbon-aware data center job scheduling. The environments test RL algorithms under realistic distribution shifts as well as in multi-agent settings. We show that standard off-the-shelf RL algorithms leave significant room for improving performance and highlight the challenges ahead for introducing RL to real-world sustainability tasks.</details>

### Swap Agnostic Learning, or Characterizing Omniprediction via Multicalibration

**Authors:** Parikshit Gopalan, Michael Kim, Omer Reingold

**<details><summary>Abstract**</summary>

We introduce and study the notion of Swap Agnostic Learning.The problem can be phrased as a game between a *predictor* and an *adversary*:  first, the predictor selects a hypothesis $h$; then, the adversary plays in response, and for each level set of the predictor, selects a loss-minimizing hypothesis $c_v \in \mathcal{C}$; the predictor wins if $h$ competes with the adaptive adversary's loss.Despite the strength of the adversary, our main result demonstrates the feasibility Swap Agnostic Learning for any convex loss.Somewhat surprisingly, the result follows by proving an *equivalence* between Swap Agnostic Learning and swap variants of the recent notions Omniprediction (ITCS'22) and Multicalibration (ICML'18).Beyond this equivalence, we establish further connections to the literature on Outcome Indistinguishability (STOC'20, ITCS'23), revealing a unified notion of OI that captures all existing notions of omniprediction and multicalibration.</details>

### SwapPrompt: Test-Time Prompt Adaptation for Vision-Language Models

**Authors:** XIAOSONG MA, Jie ZHANG, Song Guo, Wenchao Xu

**<details><summary>Abstract**</summary>

Test-time adaptation (TTA) is a special and practical setting in unsupervised domain adaptation, which allows a pre-trained model in a source domain to adapt to unlabeled test data in another target domain. To avoid the computation-intensive backbone fine-tuning process, the zero-shot generalization potentials of the emerging pre-trained vision-language models (e.g., CLIP, CoOp) are leveraged to only tune the run-time prompt for unseen test domains. However, existing solutions have yet to fully exploit the representation capabilities of pre-trained models as they only focus on the entropy-based optimization and the performance is far below the supervised prompt adaptation methods, e.g., CoOp. In this paper, we propose SwapPrompt, a novel framework that can effectively leverage the self-supervised contrastive learning to facilitate the test-time prompt adaptation. SwapPrompt employs a dual prompts paradigm, i.e., an online prompt and a target prompt that averaged from the online prompt to retain historical information. In addition, SwapPrompt applies a swapped prediction mechanism, which takes advantage of the representation capabilities of pre-trained models to enhance the online prompt via contrastive learning. Specifically, we use the online prompt together with an augmented view of the input image to predict the class assignment generated by the target prompt together with an alternative augmented view of the same image. The proposed SwapPrompt can be easily deployed on vision-language models without additional requirement, and experimental results show that it achieves state-of-the-art test-time adaptation performance on ImageNet and nine other datasets. It is also shown that SwapPrompt can even achieve comparable performance with supervised prompt adaptation methods.</details>

### SwiFT: Swin 4D fMRI Transformer

**Authors:** Peter Kim, Junbeom Kwon, Sunghwan Joo, Sangyoon Bae, Donggyu Lee, Yoonho Jung, Shinjae Yoo, Jiook Cha, Taesup Moon

**<details><summary>Abstract**</summary>

Modeling spatiotemporal brain dynamics from high-dimensional data, such as functional Magnetic Resonance Imaging (fMRI), is a formidable task in neuroscience. Existing approaches for fMRI analysis utilize hand-crafted features, but the process of feature extraction risks losing essential information in fMRI scans. To address this challenge, we present SwiFT (Swin 4D fMRI Transformer), a Swin Transformer architecture that can learn brain dynamics directly from fMRI volumes in a memory and computation-efficient manner. SwiFT achieves this by implementing a 4D window multi-head self-attention mechanism and absolute positional embeddings. We evaluate SwiFT using multiple large-scale resting-state fMRI datasets, including the Human Connectome Project (HCP), Adolescent Brain Cognitive Development (ABCD), and UK Biobank (UKB) datasets, to predict sex, age, and cognitive intelligence. Our experimental outcomes reveal that SwiFT consistently outperforms recent state-of-the-art models. Furthermore, by leveraging its end-to-end learning capability, we show that contrastive loss-based self-supervised pre-training of SwiFT can enhance performance on downstream tasks. Additionally, we employ an explainable AI method to identify the brain regions associated with sex classification. To our knowledge, SwiFT is the first Swin Transformer architecture to process dimensional spatiotemporal brain functional data in an end-to-end fashion. Our work holds substantial potential in facilitating scalable learning of functional brain imaging in neuroscience research by reducing the hurdles associated with applying Transformer models to high-dimensional fMRI.</details>

### SynMob: Creating High-Fidelity Synthetic GPS Trajectory Dataset for Urban Mobility Analysis

**Authors:** Yuanshao Zhu, Yongchao Ye, Ying Wu, Xiangyu Zhao, James Yu

**<details><summary>Abstract**</summary>

Urban mobility analysis has been extensively studied in the past decade using a vast amount of GPS trajectory data, which reveals hidden patterns in movement and human activity within urban landscapes. Despite its significant value, the availability of such datasets often faces limitations due to privacy concerns, proprietary barriers, and quality inconsistencies. To address these challenges, this paper presents a synthetic trajectory dataset with high fidelity, offering a general solution to these data accessibility issues. Specifically, the proposed dataset adopts a diffusion model as its synthesizer, with the primary aim of accurately emulating the spatial-temporal behavior of the original trajectory data. These synthesized data can retain the geo-distribution and statistical properties characteristic of real-world datasets. Through rigorous analysis and case studies, we validate the high similarity and utility between the proposed synthetic trajectory dataset and real-world counterparts. Such validation underscores the practicality of synthetic datasets for urban mobility analysis and advocates for its wider acceptance within the research community. Finally, we publicly release the trajectory synthesizer and datasets, aiming to enhance the quality and availability of synthetic trajectory datasets and encourage continued contributions to this rapidly evolving field. The dataset is released for public online availability https://github.com/Applied-Machine-Learning-Lab/SynMob.</details>

### Synthcity: a benchmark framework for diverse use cases of tabular synthetic data

**Authors:** Zhaozhi Qian, Rob Davis, Mihaela van der Schaar

**<details><summary>Abstract**</summary>

Accessible high-quality data is the bread and butter of machine learning research, and the demand for data has exploded as larger and more advanced ML models are built across different domains. Yet, real data often contain sensitive information, are subject to various biases, and are costly to acquire, which compromise their quality and accessibility. Synthetic data have thus emerged as a complement to, sometimes even a replacement for, real data for ML training. However, the landscape of synthetic data research has been fragmented due to the diverse range of data modalities, such as tabular, time series, and images, and the wide array of use cases, including privacy preservation, fairness considerations, and data augmentation. This fragmentation poses practical challenges when comparing and selecting synthetic data generators in for different problem settings. To this end, we develop Synthcity, an open-source Python library that allows researchers and practitioners to perform one-click benchmarking of synthetic data generators across data modalities and use cases. Beyond benchmarking, Synthcity serves as a centralized toolkit for accessing cutting-edge data generators. In addition, Synthcity’s flexible plug-in style API makes it easy to incorporate additional data generators into the framework. Using examples of tabular data generation and data augmentation, we illustrate the general applicability of Synthcity, and the insight one can obtain.</details>

### T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation

**Authors:** Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, Xihui Liu

**<details><summary>Abstract**</summary>

Despite the stunning ability to generate high-quality images by recent text-to-image models, current approaches often struggle to effectively compose objects with different attributes and relationships into a complex and coherent scene. We propose T2I-CompBench, a comprehensive benchmark for open-world compositional text-to-image generation, consisting of 6,000 compositional text prompts from 3 categories (attribute binding, object relationships, and complex compositions) and 6 sub-categories (color binding, shape binding, texture binding, spatial relationships, non-spatial relationships, and complex compositions). We further propose several evaluation metrics specifically designed to evaluate compositional text-to-image generation and explore the potential and limitations of multimodal LLMs for evaluation. We introduce a new approach, Generative mOdel finetuning with Reward-driven Sample selection (GORS), to boost the compositional text-to-image generation abilities of pretrained text-to-image models. Extensive experiments and evaluations are conducted to benchmark previous methods on T2I-CompBench, and to validate the effectiveness of our proposed evaluation metrics and GORS approach. Project page is available at https://karine-h.github.io/T2I-CompBench/.</details>

### TFLEX: Temporal Feature-Logic Embedding Framework for Complex Reasoning over Temporal Knowledge Graph

**Authors:** Xueyuan Lin, Haihong E, Chengjin Xu, Gengxian Zhou, Haoran Luo, Tianyi Hu, Fenglong Su, Ningyuan Li, Mingzhi Sun

**<details><summary>Abstract**</summary>

Multi-hop logical reasoning over knowledge graph plays a fundamental role in many artificial intelligence tasks.  Recent complex query embedding methods for reasoning focus on static KGs, while temporal knowledge graphs have not been fully explored.  Reasoning over TKGs has two challenges: 1. The query should answer entities or timestamps; 2. The operators should consider both set logic on entity set and temporal logic on timestamp set.To bridge this gap, we introduce the multi-hop logical reasoning problem on TKGs and then propose the first temporal complex query embedding named Temporal Feature-Logic Embedding framework (TFLEX) to answer the temporal complex queries.  Specifically, we utilize fuzzy logic to compute the logic part of the Temporal Feature-Logic embedding, thus naturally modeling all first-order logic operations on the entity set.  In addition, we further extend fuzzy logic on timestamp set to cope with three extra temporal operators (**After**, **Before** and **Between**).Experiments on numerous query patterns demonstrate the effectiveness of our method.</details>

### TRIAGE: Characterizing and auditing training data for improved regression

**Authors:** Nabeel Seedat, Jonathan Crabbé, Zhaozhi Qian, Mihaela van der Schaar

**<details><summary>Abstract**</summary>

Data quality is crucial for robust machine learning algorithms, with the recent interest in data-centric AI emphasizing the importance of training data characterization. However, current data characterization methods are largely focused on classification settings, with regression settings largely understudied. To address this, we introduce TRIAGE, a novel data characterization framework tailored to regression tasks and compatible with a broad class of regressors. TRIAGE utilizes conformal predictive distributions to provide a model-agnostic scoring method, the TRIAGE score. We operationalize the score to analyze individual samples' training dynamics and characterize samples as under-, over-, or well-estimated by the model. We show that TRIAGE's characterization is consistent and highlight its utility to improve performance via data sculpting/filtering, in multiple regression settings. Additionally, beyond sample level, we show TRIAGE enables new approaches to dataset selection and feature acquisition. Overall, TRIAGE highlights the value unlocked by data characterization in real-world regression applications.</details>

### Taking the neural sampling code very seriously: A data-driven approach for evaluating generative models of the visual system

**Authors:** Suhas Shrinivasan, Konstantin-Klemens Lurz, Kelli Restivo, George Denfield, Andreas Tolias, Edgar Walker, Fabian Sinz

**<details><summary>Abstract**</summary>

Prevailing theories of perception hypothesize that the brain implements perception via Bayesian inference in a generative model of the world.One prominent theory, the Neural Sampling Code (NSC), posits that neuronal responses to a stimulus represent samples from the posterior distribution over latent world state variables that cause the stimulus.Although theoretically elegant, NSC does not specify the exact form of the generative model or prescribe how to link the theory to recorded neuronal activity.Previous works assume simple generative models and test their qualitative agreement with neurophysiological data.Currently, there is no precise alignment of the normative theory with neuronal recordings, especially in response to natural stimuli, and a quantitative, experimental evaluation of models under NSC has been lacking.Here, we propose a novel formalization of NSC, that (a) allows us to directly fit NSC generative models to recorded neuronal activity in response to natural images, (b) formulate richer and more flexible generative models, and (c) employ standard metrics to quantitatively evaluate different generative models under NSC.Furthermore, we derive a stimulus-conditioned predictive model of neuronal responses from the trained generative model using our formalization that we compare to neural system identification models.We demonstrate our approach by fitting and comparing classical- and flexible deep learning-based generative models on population recordings from the macaque primary visual cortex (V1) to natural images, and show that the flexible models outperform classical models in both their generative- and predictive-model performance.Overall, our work is an important step towards a quantitative evaluation of NSC. It provides a framework that lets us \textit{learn} the generative model directly from neuronal population recordings, paving the way for an experimentally-informed understanding of probabilistic computational principles underlying perception and behavior.</details>

### Taming Local Effects in Graph-based Spatiotemporal Forecasting

**Authors:** Andrea Cini, Ivan Marisca, Daniele Zambon, Cesare Alippi

**<details><summary>Abstract**</summary>

Spatiotemporal graph neural networks have shown to be effective in time series forecasting applications, achieving better performance than standard univariate predictors in several settings. These architectures take advantage of a graph structure and relational inductive biases to learn a single (global) inductive model to predict any number of the input time series, each associated with a graph node. Despite the gain achieved in computational and data efficiency w.r.t. fitting a set of local models, relying on a single global model can be a limitation whenever some of the time series are generated by a different spatiotemporal stochastic process. The main objective of this paper is to understand the interplay between globality and locality in graph-based spatiotemporal forecasting, while contextually proposing a methodological framework to rationalize the practice of including trainable node embeddings in such architectures. We ascribe to trainable node embeddings the role of amortizing the learning of specialized components. Moreover, embeddings allow for 1) effectively combining the advantages of shared message-passing layers with node-specific parameters and 2) efficiently transferring the learned model to new node sets. Supported by strong empirical evidence, we provide insights and guidelines for specializing graph-based models to the dynamics of each time series and show how this aspect plays a crucial role in obtaining accurate predictions.</details>

### Task-aware Distributed Source Coding under Dynamic Bandwidth

**Authors:** Po-han Li, Sravan Kumar Ankireddy, Ruihan (Philip) Zhao, Hossein Nourkhiz Mahjoub, Ehsan Moradi Pari, Ufuk Topcu, Sandeep Chinchali, Hyeji Kim

**<details><summary>Abstract**</summary>

Efficient compression of correlated data is essential to minimize communication overload in multi-sensor networks. In such networks, each sensor independently compresses the data and transmits them to a central node. A decoder at the central node decompresses and passes the data to a pre-trained machine learning-based task model to generate the final output. Due to limited communication bandwidth, it is important for the compressor to learn only the features that are relevant to the task. Additionally, the final performance depends heavily on the total available bandwidth. In practice, it is common to encounter varying availability in bandwidth. Since higher bandwidth results in better performance, it is essential for the compressor to dynamically take advantage of the maximum available bandwidth at any instant. In this work, we propose a novel distributed compression framework composed of independent encoders and a joint decoder, which we call neural distributed principal component analysis (NDPCA). NDPCA flexibly compresses data from multiple sources to any available bandwidth with a single model, reducing compute and storage overhead. NDPCA achieves this by learning low-rank task representations and efficiently distributing bandwidth among sensors, thus providing a graceful trade-off between performance and bandwidth. Experiments show that NDPCA improves the success rate of multi-view robotic arm manipulation by 9% and the accuracy of object detection tasks on satellite imagery by 14% compared to an autoencoder with uniform bandwidth allocation.</details>

### TempME: Towards the Explainability of Temporal Graph Neural Networks via Motif Discovery

**Authors:** Jialin Chen, Rex Ying

**<details><summary>Abstract**</summary>

Temporal graphs are widely used to model dynamic systems with time-varying interactions. In real-world scenarios, the underlying mechanisms of generating future interactions in dynamic systems are typically governed by a set of recurring substructures within the graph, known as temporal motifs. Despite the success and prevalence of current temporal graph neural networks (TGNN), it remains uncertain which temporal motifs are recognized as the significant indications that trigger a certain prediction from the model, which is a critical challenge for advancing the explainability and trustworthiness of current TGNNs. To address this challenge, we propose a novel approach, called **Temp**oral **M**otifs **E**xplainer (**TempME**), which uncovers the most pivotal temporal motifs guiding the prediction of TGNNs.  Derived from the information bottleneck principle, TempME extracts the most interaction-related motifs while minimizing the amount of contained information to preserve the sparsity and succinctness of the explanation. Events in the explanations generated by TempME are verified to be more spatiotemporally correlated than those of existing approaches, providing more understandable insights. Extensive experiments validate the superiority of TempME, with up to 8.21% increase in terms of explanation accuracy across six real-world datasets and up to 22.96% increase in boosting the prediction Average Precision of current TGNNs.</details>

### Temporal Dynamic Quantization for Diffusion Models

**Authors:** Junhyuk So, Jungwon Lee, Daehyun Ahn, Hyungjun Kim, Eunhyeok Park

**<details><summary>Abstract**</summary>

Diffusion model has gained popularity in vision applications due to its remarkable generative performance and versatility. However, its high storage and computation demands, resulting from the model size and iterative generation, hinder its use on mobile devices. Existing quantization techniques struggle to maintain performance even in 8-bit precision due to the diffusion model's unique property of temporal variation in activation. We introduce a novel quantization method that dynamically adjusts the quantization interval based on time step information, significantly improving output quality. Unlike conventional dynamic quantization techniques, our approach has no computational overhead during inference and is compatible with both post-training quantization (PTQ) and quantization-aware training (QAT). Our extensive experiments demonstrate substantial improvements in output quality with the quantized model across various configurations.</details>

### Temporal Graph Benchmark for Machine Learning on Temporal Graphs

**Authors:** Shenyang Huang, Farimah Poursafaei, Jacob Danovitch, Matthias Fey, Weihua Hu, Emanuele Rossi, Jure Leskovec, Michael Bronstein, Guillaume Rabusseau, Reihaneh Rabbany

**<details><summary>Abstract**</summary>

We present the Temporal Graph Benchmark (TGB), a collection of challenging and diverse benchmark datasets for realistic, reproducible, and robust evaluation of machine learning models on temporal graphs. TGB datasets are of large scale, spanning years in duration, incorporate both node and edge-level prediction tasks and cover a diverse set of domains including social, trade, transaction, and transportation networks. For both tasks, we design evaluation protocols based on realistic use-cases. We extensively benchmark each dataset and find that the performance of common models can vary drastically across datasets. In addition, on dynamic node property prediction tasks, we show that simple methods often achieve superior performance compared to existing temporal graph models. We believe that these findings open up opportunities for future research on temporal graphs. Finally, TGB provides an automated machine learning pipeline for reproducible and accessible temporal graph research, including data loading, experiment setup and performance evaluation. TGB will be maintained and updated on a regular basis and welcomes community feedback. TGB datasets, data loaders, example codes, evaluation setup, and leaderboards are publicly available at https://tgb.complexdatalab.com/.</details>

### The Benefits of Being Distributional: Small-Loss Bounds for Reinforcement Learning

**Authors:** Kaiwen Wang, Kevin Zhou, Runzhe Wu, Nathan Kallus, Wen Sun

**<details><summary>Abstract**</summary>

While distributional reinforcement learning (DistRL) has been empirically effective, the question of when and why it is better than vanilla, non-distributional RL has remained unanswered.This paper explains the benefits of DistRL through the lens of small-loss bounds, which are instance-dependent bounds that scale with optimal achievable cost.Particularly, our bounds converge much faster than those from non-distributional approaches if the optimal cost is small.As warmup, we propose a distributional contextual bandit (DistCB) algorithm, which we show enjoys small-loss regret bounds and empirically outperforms the state-of-the-art on three real-world tasks.In online RL, we propose a DistRL algorithm that constructs confidence sets using maximum likelihood estimation. We prove that our algorithm enjoys novel small-loss PAC bounds in low-rank MDPs.As part of our analysis, we introduce the $\ell_1$ distributional eluder dimension which may be of independent interest. Then, in offline RL, we show that pessimistic DistRL enjoys small-loss PAC bounds that are novel to the offline setting and are more robust to bad single-policy coverage.</details>

### The Best of Both Worlds in Network Population Games: Reaching Consensus and Convergence to Equilibrium

**Authors:** Shuyue Hu, Harold Soh, Georgios Piliouras

**<details><summary>Abstract**</summary>

Reaching consensus and convergence to equilibrium are two major challenges of multi-agent systems. Although each has attracted significant attention, relatively few studies address both challenges at the same time. This paper examines the connection between the notions of consensus and equilibrium in a multi-agent system where multiple interacting sub-populations coexist. We argue that consensus can be seen as an intricate component of intra-population stability, whereas equilibrium can be seen as encoding inter-population stability. We show that smooth fictitious play, a well-known learning model in game theory, can achieve both consensus and convergence to equilibrium in diverse multi-agent settings. Moreover, we show that the consensus formation process plays a crucial role in the seminal thorny problem of equilibrium selection in multi-agent learning.</details>

### The Cambridge Law Corpus: A Corpus for Legal AI Research

**Authors:** Andreas Östling, Holli Sargeant, Huiyuan Xie, Ludwig Bull, Alexander Terenin, Leif Jonsson, Måns Magnusson, Felix Steffek

**<details><summary>Abstract**</summary>

We introduce the Cambridge Law Corpus (CLC), a corpus for legal AI research. It consists of over 250 000 court cases from the UK. Most cases are from the 21st century, but the corpus includes cases as old as the 16th century. This paper presents the first release of the corpus, containing the raw text and meta-data. Together with the corpus, we provide annotations on case outcomes for 638 cases, done by legal experts. Using our annotated data, we have trained and evaluated case outcome extraction with GPT-3, GPT-4 and RoBERTa models to provide benchmarks. We include an extensive legal and ethical discussion to address the potentially sensitive nature of this material. As a consequence, the corpus will only be released for research purposes under certain restrictions.</details>

### The Crucial Role of Normalization in Sharpness-Aware Minimization

**Authors:** Yan Dai, Kwangjun Ahn, Suvrit Sra

**<details><summary>Abstract**</summary>

Sharpness-Aware Minimization (SAM) is a recently proposed gradient-based optimizer (Foret et al., ICLR 2021) that greatly improves the prediction performance of deep neural networks. Consequently, there has been a surge of interest in explaining its empirical success. We focus, in particular, on understanding ***the role played by normalization***, a key component of the SAM updates. We theoretically and empirically study the effect of normalization in SAM for both convex and non-convex functions, revealing two key roles played by normalization: i) it helps in stabilizing the algorithm; and ii) it enables the algorithm to drift along a continuum (manifold) of minima -- a property identified by recent theoretical works that is the key to better performance. We further argue that these two properties of normalization make SAM robust against the choice of hyper-parameters, supporting the practicality of SAM. Our conclusions are backed by various experiments.</details>

### The Double-Edged Sword of Implicit Bias: Generalization vs. Robustness in ReLU Networks

**Authors:** Spencer Frei, Gal Vardi, Peter Bartlett, Nati Srebro

**<details><summary>Abstract**</summary>

In this work, we study the implications of the implicit bias of gradient flow on generalization and adversarial robustness in ReLU networks.  We focus on a setting where the data consists of clusters and the correlations between cluster means are small, and show that in two-layer ReLU networks gradient flow is biased towards solutions that generalize well, but are vulnerable to adversarial examples. Our results hold even in cases where the network is highly overparameterized.  Despite the potential for harmful overfitting in such settings, we prove that the implicit bias of gradient flow prevents it.  However, the implicit bias also leads to non-robust solutions (susceptible to small adversarial $\ell_2$-perturbations), even though robust networks that fit the data exist.</details>

### [Spotlight] The Equivalence of Dynamic and Strategic Stability under Regularized Learning in Games

**Authors:** Victor Boone, Panayotis Mertikopoulos

**<details><summary>Abstract**</summary>

In this paper, we examine the long-run behavior of regularized, no-regret learning in finite N-player games. A well-known result in the field states that the empirical frequencies of play under no-regret learning converge to the game’s set of coarse correlated equilibria; however, our understanding of how the players' _actual strategies_ evolve over time is much more limited – and, in many cases, non-existent. This issue is exacerbated further by a series of recent results showing that _only_ strict Nash equilibria are stable and attracting under regularized learning, thus making the relation between learning and _pointwise_ solution concepts particularly elusive. In lieu of this, we take a more general approach and instead seek to characterize the _setwise_ rationality properties of the players' day-to-day trajectory of play. To do so, we focus on one of the most stringent criteria of setwise strategic stability, namely that any unilateral deviation from the set in question incurs a cost to the deviator – a property known as _closedness under better replies_ (club). In so doing, we obtain a remarkable equivalence between strategic and dynamic stability: _a product of pure strategies is closed under better replies if and only if its span is stable and attracting under regularized learning._ In addition, we estimate the rate of convergence to such sets, and we show that methods based on entropic regularization (like the exponential weights algorithm) converge at a geometric rate, while projection-based methods converge within a finite number of iterations, even with bandit, payoff-based feedback.</details>

### The Harvard USPTO Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications

**Authors:** Mirac Suzgun, Luke Melas-Kyriazi, Suproteem Sarkar, Scott D Kominers, Stuart Shieber

**<details><summary>Abstract**</summary>

Innovation is a major driver of economic and social development, and information about many kinds of innovation is embedded in semi-structured data from patents and patent applications. Though the impact and novelty of innovations expressed in patent data are difficult to measure through traditional means, machine learning offers a promising set of techniques for evaluating novelty, summarizing contributions, and embedding semantics. In this paper, we introduce the Harvard USPTO Patent Dataset (HUPD), a large-scale, well-structured, and multi-purpose corpus of English-language patent applications filed to the United States Patent and Trademark Office (USPTO) between 2004 and 2018. With more than 4.5 million patent documents, HUPD is two to three times larger than comparable corpora. Unlike other NLP patent datasets, HUPD contains the inventor-submitted versions of patent applications, not the final versions of granted patents, allowing us to study patentability at the time of filing using NLP methods for the first time. It is also novel in its inclusion of rich structured data alongside the text of patent filings: By providing each application’s metadata along with all of its text fields, HUPD enables researchers to perform new sets of NLP tasks that leverage variation in structured covariates. As a case study on the types of research HUPD makes possible, we introduce a new task to the NLP community -- patent acceptance prediction. We additionally show the structured metadata provided in HUPD allows us to conduct explicit studies of concept shifts for this task. We find that performance on patent acceptance prediction decays when models trained in one context are evaluated on different innovation categories and over time. Finally, we demonstrate how HUPD can be used for three additional tasks: Multi-class classification of patent subject areas, language modeling, and abstractive summarization. Put together, our publicly-available dataset aims to advance research extending language and classification models to diverse and dynamic real-world data distributions.</details>

### The Target-Charging Technique for Privacy Analysis across Interactive Computations

**Authors:** Edith Cohen, Xin Lyu

**<details><summary>Abstract**</summary>

We propose the \emph{Target Charging Technique} (TCT), a unified privacy analysis framework for interactive settings where a sensitive dataset is accessed multiple times using differentially private algorithms. Unlike traditional composition, where privacy guarantees deteriorate quickly with the number of accesses, TCT allows computations that don't hit a specified \emph{target}, often the vast majority, to be essentially free (while incurring instead a small overhead on those that do hit their targets). TCT generalizes tools such as the sparse vector technique and top-k selection from private candidates and extends their remarkable privacy enhancement benefits from noisy Lipschitz functions to general private algorithms.</details>

### The probability flow ODE is provably fast

**Authors:** Sitan Chen, Sinho Chewi, Holden Lee, Yuanzhi Li, Jianfeng Lu, Adil Salim

**<details><summary>Abstract**</summary>

We provide the first polynomial-time convergence guarantees for the probabilistic flow ODE implementation (together with a corrector step) of score-based generative modeling. Our analysis is carried out in the wake of recent results obtaining such guarantees for the SDE-based implementation (i.e., denoising diffusion probabilistic modeling or DDPM), but requires the development of novel techniques for studying deterministic dynamics without contractivity. Through the use of a specially chosen corrector step based on the underdamped Langevin diffusion, we obtain better dimension dependence than prior works on DDPM ($O(\sqrt d)$ vs. $O(d)$, assuming smoothness of the data distribution), highlighting potential advantages of the ODE framework.</details>

### Theoretical Analysis of the Inductive Biases in Deep Convolutional Networks

**Authors:** Zihao Wang, Lei Wu

**<details><summary>Abstract**</summary>

In this paper, we provide a theoretical analysis of the inductive biases in convolutional neural networks (CNNs). We start by examining the universality of CNNs, i.e., the ability to approximate any continuous functions.  We prove that a depth of $\mathcal{O}(\log d)$ suffices for deep CNNs to achieve this universality, where $d$ in the input dimension.  Additionally, we establish  that learning sparse functions with  CNNs requires only $\widetilde{\mathcal{O}}(\log^2d)$ samples, indicating that deep CNNs can efficiently capture {\em long-range} sparse correlations. These results are made possible through a novel combination of the multichanneling and downsampling when increasing the network depth. We also delve into the distinct roles  of weight sharing and locality in CNNs. To this end, we compare the performance of CNNs, locally-connected networks (LCNs), and fully-connected networks (FCNs) on a simple regression task, where LCNs can be viewed as CNNs without weight sharing. On the one hand,  we  prove that LCNs require ${\Omega}(d)$ samples while CNNs need only $\widetilde{\mathcal{O}}(\log^2d)$ samples,  highlighting the  critical role of weight sharing. On the other hand, we prove that FCNs require $\Omega(d^2)$ samples, whereas LCNs need only $\widetilde{\mathcal{O}}(d)$ samples,  underscoring the importance  of locality. These provable separations quantify the difference between the two biases, and the major observation behind our proof is that weight sharing and locality break different symmetries in the learning process.</details>

### Thrust: Adaptively Propels Large Language Models with External Knowledge

**Authors:** Xinran Zhao, Hongming Zhang, Xiaoman Pan, Wenlin Yao, Dong Yu, Jianshu Chen

**<details><summary>Abstract**</summary>

Although large-scale pre-trained language models (PTLMs) are shown to encode rich knowledge in their model parameters, the inherent knowledge in PTLMs can be opaque or static, making external knowledge necessary. However, the existing information retrieval techniques could be costly and may even introduce noisy and sometimes misleading knowledge. To address these challenges, we propose the instance-level adaptive propulsion of external knowledge (IAPEK), where we only conduct the retrieval when necessary. To achieve this goal, we propose to model whether a PTLM contains enough knowledge to solve an instance with a novel metric, Thrust, which leverages the representation distribution of a small amount of seen instances. Extensive experiments demonstrate that Thrust is a good measurement of models' instance-level knowledgeability. Moreover, we can achieve higher cost-efficiency with the Thrust score as the retrieval indicator than the naive usage of external knowledge on 88% of the evaluated tasks with 26% average performance improvement. Such findings shed light on the real-world practice of knowledge-enhanced LMs with a limited budget for knowledge seeking due to computation latency or costs.</details>

### [Spotlight] Tight Risk Bounds for Gradient Descent on Separable Data

**Authors:** Matan Schliserman, Tomer Koren

**<details><summary>Abstract**</summary>

We study the generalization properties of unregularized gradient methods applied to separable linear classification---a setting that has received considerable attention since the pioneering work of Soudry et al. (2018).We establish tight upper and lower (population) risk bounds for gradient descent in this setting, for any smooth loss function, expressed in terms of its tail decay rate.Our bounds take the form $\Theta(r_{\ell,T}^2 / \gamma^2 T + r_{\ell,T}^2 / \gamma^2 n)$, where $T$ is the number of gradient steps, $n$ is size of the training set, $\gamma$ is the data margin, and $r_{\ell,T}$ is a complexity term that depends on the tail decay rate of the loss function (and on $T$).Our upper bound greatly improves the existing risk bounds due to Shamir (2021) and Schliserman and Koren (2022), that either applied to specific loss functions or imposed extraneous technical assumptions, and applies to virtually any convex and smooth loss function.Our risk lower bound is the first in this context and establish the tightness of our general upper bound for any given tail decay rate and in all parameter regimes.The proof technique used to show these results is also markedly simpler compared to previous work, and is straightforward to extend to other gradient methods; we illustrate this by providing analogous results for Stochastic Gradient Descent.</details>

### To Stay or Not to Stay in the Pre-train Basin: Insights on Ensembling in Transfer Learning

**Authors:** Ildus Sadrtdinov, Dmitrii Pozdeev, Dmitry Vetrov, Ekaterina Lobacheva

**<details><summary>Abstract**</summary>

Transfer learning and ensembling are two popular techniques for improving the performance and robustness of neural networks. Due to the high cost of pre-training, ensembles of models fine-tuned from a single pre-trained checkpoint are often used in practice. Such models end up in the same basin of the loss landscape, which we call the pre-train basin, and thus have limited diversity. In this work, we show that ensembles trained from a single pre-trained checkpoint may be improved by better exploring the pre-train basin, however, leaving the basin results in losing the benefits of transfer learning and in degradation of the ensemble quality. Based on the analysis of existing exploration methods, we propose a more effective modification of the Snapshot Ensembles (SSE) for transfer learning setup, StarSSE, which results in stronger ensembles and uniform model soups.</details>

### ToolQA: A Dataset for LLM Question Answering with External Tools

**Authors:** Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, Chao Zhang

**<details><summary>Abstract**</summary>

Large Language Models (LLMs) have demonstrated impressive performance in various NLP tasks, but they still suffer from challenges such as hallucination and weak numerical reasoning. To overcome these challenges, external tools can be used to enhance LLMs' question-answering abilities. However, current evaluation methods do not distinguish between questions that can be answered using LLMs' internal knowledge and those that require external information through tool use. To address this issue, we introduce a new dataset called ToolQA, which is designed to faithfully evaluate LLMs' ability to use external tools for question answering. Our development of ToolQA involved a scalable, automated process for dataset curation, along with 13 specialized tools designed for interaction with external knowledge in order to answer questions. Importantly, we strive to minimize the overlap between our benchmark data and LLMs' pre-training data, enabling a more precise evaluation of LLMs' tool-use reasoning abilities. We conducted an in-depth diagnosis of existing tool-use LLMs to highlight their strengths, weaknesses, and potential improvements. Our findings set a new benchmark for evaluating LLMs and suggest new directions for future advancements. Our data and code are freely available for the broader scientific community on GitHub.</details>

### TopoSRL: Topology preserving self-supervised Simplicial Representation Learning

**Authors:** Hiren Madhu, Sundeep Prabhakar Chepuri

**<details><summary>Abstract**</summary>

In this paper, we introduce $\texttt{TopoSRL}$, a novel self-supervised learning (SSL) method for simplicial complexes to effectively capture higher-order interactions and preserve topology in the learned representations. $\texttt{TopoSRL}$ addresses the limitations of existing graph-based SSL methods that typically concentrate on pairwise relationships, neglecting long-range dependencies crucial to capture topological information. We propose a new simplicial augmentation technique that generates two views of the simplicial complex that enriches the representations while being efficient. Next, we propose a new simplicial contrastive loss function that contrasts the generated simplices to preserve local and global information present in the simplicial complexes. Extensive experimental results demonstrate the superior performance of $\texttt{TopoSRL}$ compared to state-of-the-art graph SSL techniques and supervised simplicial neural models across various datasets corroborating the efficacy of $\texttt{TopoSRL}$ in processing simplicial complex data in a self-supervised setting.</details>

### Topological RANSAC for instance verification and retrieval without fine-tuning

**Authors:** Guoyuan An, Ju-hyeong Seon, Inkyu An, Yuchi Huo, Sung-eui Yoon

**<details><summary>Abstract**</summary>

This paper presents an innovative approach to enhancing explainable image retrieval, particularly in situations where a fine-tuning set is unavailable. The widely-used SPatial verification (SP) method, despite its efficacy, relies on a spatial model and the hypothesis-testing strategy for instance recognition, leading to inherent limitations, including the assumption of planar structures and neglect of topological relations among features. To address these shortcomings, we introduce a pioneering technique that replaces the spatial model with a topological one within the RANSAC process. We propose bio-inspired saccade and fovea functions to verify the topological consistency among features, effectively circumventing the issues associated with SP's spatial model. Our experimental results demonstrate that our method significantly outperforms SP, achieving state-of-the-art performance in non-fine-tuning retrieval. Furthermore, our approach can enhance performance when used in conjunction with fine-tuned features. Importantly, our method retains high explainability and is lightweight, offering a practical and adaptable solution for a variety of real-world applications.</details>

### Topology-Aware Uncertainty for Image Segmentation

**Authors:** Saumya Gupta, Yikai Zhang, Xiaoling Hu, Prateek Prasanna, Chao Chen

**<details><summary>Abstract**</summary>

Segmentation of curvilinear structures such as vasculature and road networks is challenging due to relatively weak signals and complex geometry/topology. To facilitate and accelerate large scale annotation, one has to adopt semi-automatic approaches such as proofreading by experts. In this work, we focus on uncertainty estimation for such tasks, so that highly uncertain, and thus error-prone structures can be identified for human annotators to verify. Unlike most existing works, which provide pixel-wise uncertainty maps, we stipulate it is crucial to estimate uncertainty in the units of topological structures, e.g., small pieces of connections and branches. To achieve this, we leverage tools from topological data analysis, specifically discrete Morse theory (DMT), to first capture the structures, and then reason about their uncertainties. To model the uncertainty, we (1) propose a joint prediction model that estimates the uncertainty of a structure while taking the neighboring structures into consideration (inter-structural uncertainty); (2) propose a novel Probabilistic DMT to model the inherent uncertainty within each structure (intra-structural uncertainty) by sampling its representations via a perturb-and-walk scheme. On various 2D and 3D datasets, our method produces better structure-wise uncertainty maps compared to existing works. Code available at: https://github.com/Saumya-Gupta-26/struct-uncertainty</details>

### Towards Efficient Pre-Trained Language Model via Feature Correlation Distillation

**Authors:** Kun Huang, Xin Guo, Meng Wang

**<details><summary>Abstract**</summary>

Knowledge Distillation (KD) has emerged as a promising approach for compressing large Pre-trained Language Models (PLMs). The performance of KD relies on how to effectively formulate and transfer the knowledge from the teacher model to the student model. Prior arts mainly focus on directly aligning output features from the transformer block, which may impose overly strict constraints on the student model's learning process and complicate the training process by introducing extra parameters and computational cost. Moreover, our analysis indicates that the different relations within self-attention, as adopted in other works, involves more computation complexities and can easily be constrained by the number of heads, potentially leading to suboptimal solutions.  To address these issues, we propose a novel approach that builds relationships directly from output features. Specifically, we introduce token-level and sequence-level relations concurrently  to fully exploit the knowledge from the teacher model. Furthermore, we propose a correlation-based distillation loss to alleviate the exact match properties inherent in traditional KL divergence or MSE loss functions. Our method, dubbed FCD, presents a simple yet effective method to compress various architectures (BERT, RoBERTa, and GPT) and model sizes (base-size and large-size).  Extensive experimental results demonstrate that our distilled, smaller language models significantly surpass existing KD methods across various NLP tasks.</details>

### Towards Foundation Models for Scientific Machine Learning: Characterizing Scaling and Transfer Behavior

**Authors:** Shashank Subramanian, Peter Harrington, Kurt Keutzer, Wahid Bhimji, Dmitriy Morozov, Michael Mahoney, Amir Gholami

**<details><summary>Abstract**</summary>

Pre-trained machine learning (ML) models have shown great performance for awide range of applications, in particular in natural language processing (NLP)and computer vision (CV). Here, we study how pre-training could be used forscientific machine learning (SciML) applications, specifically in the context oftransfer learning. We study the transfer behavior of these models as (i) the pretrainedmodel size is scaled, (ii) the downstream training dataset size is scaled,(iii) the physics parameters are systematically pushed out of distribution, and (iv)how a single model pre-trained on a mixture of different physics problems canbe adapted to various downstream applications. We find that—when fine-tunedappropriately—transfer learning can help reach desired accuracy levels with ordersof magnitude fewer downstream examples (across different tasks that can even beout-of-distribution) than training from scratch, with consistent behaviour across awide range of downstream examples. We also find that fine-tuning these modelsyields more performance gains as model size increases, compared to training fromscratch on new downstream tasks. These results hold for a broad range of PDElearning tasks. All in all, our results demonstrate the potential of the “pre-train andfine-tune” paradigm for SciML problems, demonstrating a path towards buildingSciML foundation models. Our code is available as open-source.</details>

### [Spotlight] Towards In-context Scene Understanding

**Authors:** Ivana Balazevic, David Steiner, Nikhil Parthasarathy, Relja Arandjelović, Olivier Henaff

**<details><summary>Abstract**</summary>

In-context learning––the ability to configure a model's behavior with different prompts––has revolutionized the field of natural language processing, alleviating the need for task-specific models and paving the way for generalist models capable of assisting with any query. Computer vision, in contrast, has largely stayed in the former regime: specialized decoders and finetuning protocols are generally required to perform dense tasks such as semantic segmentation and depth estimation. In this work we explore a simple mechanism for in-context learning of such scene understanding tasks: nearest neighbor retrieval from a prompt of annotated features. We propose a new pretraining protocol––leveraging attention within and across images––which yields representations particularly useful in this regime. The resulting Hummingbird model, suitably prompted, performs various scene understanding tasks without modification while approaching the performance of specialists that have been finetuned for each task. Moreover, Hummingbird can be configured to perform new tasks much more efficiently than finetuned models, raising the possibility of scene understanding in the interactive assistant regime.</details>

### Towards Last-Layer Retraining for Group Robustness with Fewer Annotations

**Authors:** Tyler LaBonte, Vidya Muthukumar, Abhishek Kumar

**<details><summary>Abstract**</summary>

Empirical risk minimization (ERM) of neural networks is prone to over-reliance on spurious correlations and poor generalization on minority groups. The recent deep feature reweighting (DFR) technique achieves state-of-the-art group robustness via simple last-layer retraining, but it requires held-out group and class annotations to construct a group-balanced reweighting dataset. In this work, we examine this impractical requirement and find that last-layer retraining can be surprisingly effective with no group annotations (other than for model selection) and only a handful of class annotations. We first show that last-layer retraining can greatly improve worst-group accuracy even when the reweighting dataset has only a small proportion of worst-group data. This implies a "free lunch" where holding out a subset of training data to retrain the last layer can substantially outperform ERM on the entire dataset with no additional data, annotations, or computation for training. To further improve group robustness, we introduce a lightweight method called selective last-layer finetuning (SELF), which constructs the reweighting dataset using misclassifications or disagreements. Our experiments present the first evidence that model disagreement upsamples worst-group data, enabling SELF to nearly match DFR on four well-established benchmarks across vision and language tasks with no group annotations and less than 3% of the held-out class annotations.</details>

### Towards Last-layer Retraining for Group Robustness with Fewer Annotations

**Authors:** Tyler LaBonte, Vidya Muthukumar, Abhishek Kumar

**<details><summary>Abstract**</summary>

Empirical risk minimization (ERM) of neural networks is prone to over-reliance on spurious correlations and poor generalization on minority groups. The recent deep feature reweighting (DFR) technique achieves state-of-the-art group robustness via simple last-layer retraining, but it requires held-out group and class annotations to construct a group-balanced reweighting dataset. In this work, we examine this impractical requirement and find that last-layer retraining can be surprisingly effective with no group annotations (other than for model selection) and only a handful of class annotations. We first show that last-layer retraining can greatly improve worst-group accuracy even when the reweighting dataset has only a small proportion of worst-group data. This implies a "free lunch" where holding out a subset of training data to retrain the last layer can substantially outperform ERM on the entire dataset with no additional data, annotations, or computation for training. To further improve group robustness, we introduce a lightweight method called selective last-layer finetuning (SELF), which constructs the reweighting dataset using misclassifications or disagreements. Our experiments present the first evidence that model disagreement upsamples worst-group data, enabling SELF to nearly match DFR on four well-established benchmarks across vision and language tasks with no group annotations and less than 3% of the held-out class annotations.</details>

### Towards Unbounded Machine Unlearning

**Authors:** Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, Eleni Triantafillou

**<details><summary>Abstract**</summary>

Deep machine unlearning is the problem of 'removing' from a trained neural network a subset of its training set. This problem is very timely and has many applications, including the key tasks of removing biases (RB), resolving confusion (RC) (caused by mislabelled data in trained models), as well as allowing users to exercise their 'right to be forgotten' to protect User Privacy (UP). This paper is the first, to our knowledge, to study unlearning for different applications (RB, RC, UP), with the view that each has its own desiderata, definitions for 'forgetting' and associated metrics for forget quality. For UP, we propose a novel adaptation of a strong Membership Inference Attack for unlearning. We also propose SCRUB, a novel unlearning algorithm, which is the only method that is consistently a top performer for forget quality across the different application-dependent metrics for RB, RC, and UP. At the same time, SCRUB is also consistently a top performer on metrics that measure model utility (i.e. accuracy on retained data and generalization), and is more efficient than previous work. The above are substantiated through a comprehensive empirical evaluation against previous state-of-the-art.</details>

### Towards a Unified Analysis of Kernel-based Methods Under Covariate Shift

**Authors:** Xingdong Feng, Xin HE, Caixing Wang, Chao Wang, Jingnan Zhang

**<details><summary>Abstract**</summary>

Covariate shift occurs prevalently in practice, where the input distributions of the source and target data are substantially different. Despite its practical importance in various learning problems, most of the existing methods only focus on some specific learning tasks and are not well validated theoretically and numerically. To tackle this problem, we propose a unified analysis of general nonparametric methods in a reproducing kernel Hilbert space (RKHS) under covariate shift.  Our theoretical results are established for a general loss belonging to a rich loss function family, which includes many commonly used methods as special cases, such as mean regression, quantile regression, likelihood-based classification, and margin-based classification. Two types of covariate shift problems are the focus of this paper and the sharp convergence rates are established for a general loss function to provide a unified theoretical analysis, which concurs with the optimal results in literature where the squared loss is used. Extensive numerical studies on synthetic and real examples confirm our theoretical findings and further illustrate the effectiveness of our proposed method.</details>

### Towards a fuller understanding of neurons with Clustered Compositional Explanations

**Authors:** Biagio La Rosa, Leilani Gilpin, Roberto Capobianco

**<details><summary>Abstract**</summary>

Compositional Explanations is a method for identifying logical formulas of concepts that approximate the neurons' behavior. However, these explanations are linked to the small spectrum of neuron activations (i.e., the highest ones) used to check the alignment, thus lacking completeness. In this paper, we propose a generalization, called Clustered Compositional Explanations, that combines  Compositional Explanations with clustering and a novel search heuristic to approximate a broader spectrum of the neuron behavior. We define and address the problems connected to the application of these methods to multiple ranges of activations, analyze the insights retrievable by using our algorithm, and propose desiderata qualities that can be used to study the explanations returned by different algorithms.</details>

### TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs

**Authors:** Mangpo Phothilimtha, Sami Abu-El-Haija, Kaidi Cao, Bahare Fatemi, Michael Burrows, Charith Mendis, Bryan Perozzi

**<details><summary>Abstract**</summary>

Precise hardware performance models play a crucial role in code optimizations. They can assist compilers in making heuristic decisions or aid autotuners in identifying the optimal configuration for a given program. For example, the autotuner for XLA, a machine learning compiler, discovered 10–20\% speedup on state-of-the-art models serving substantial production traffic at Google. Although there exist a few datasets for program performance prediction, they target small sub-programs such as basic blocks or kernels. This paper introduces TpuGraphs, a performance prediction dataset on full tensor programs, represented as computational graphs, running on Tensor Processing Units (TPUs). Each graph in the dataset represents the main computation of a machine learning workload, e.g., a training epoch or an inference step. Each data sample contains a computational graph, a compilation configuration, and the execution time of the graph when compiled with the configuration. The graphs in the dataset are collected from open-source machine learning programs, featuring popular model architectures (e.g., ResNet, EfficientNet, Mask R-CNN, and Transformer). TpuGraphs provides 25x more graphs than the largest graph property prediction dataset (with comparable graph sizes), and 770x larger graphs on average compared to existing performance prediction datasets on machine learning programs. This graph-level prediction task on large graphs introduces new challenges in learning, ranging from scalability, training efficiency, to model quality.</details>

### [Spotlight] Train Once, Get a Family: State-Adaptive Balances for Offline-to-Online Reinforcement Learning

**Authors:** Shenzhi Wang, Qisen Yang, Jiawei Gao, Matthieu Lin, HAO CHEN, Liwei Wu, Ning Jia, Shiji Song, Gao Huang

**<details><summary>Abstract**</summary>

Offline-to-online reinforcement learning (RL) is a training paradigm that combines pre-training on a pre-collected dataset with fine-tuning in an online environment. However, the incorporation of online fine-tuning can intensify the well-known distributional shift problem. Existing solutions tackle this problem by imposing a policy constraint on the policy improvement objective in both offline and online learning. They typically advocate a single balance between policy improvement and constraints across diverse data collections. This one-size-fits-all manner may not optimally leverage each collected sample due to the significant variation in data quality across different states. To this end, we introduce Family Offline-to-Online RL (FamO2O), a simple yet effective framework that empowers existing algorithms to determine state-adaptive improvement-constraint balances. FamO2O utilizes a universal model to train a family of policies with different improvement/constraint intensities, and a balance model to select a suitable policy for each state. Theoretically, we prove that state-adaptive balances are necessary for achieving a higher policy performance upper bound. Empirically, extensive experiments show that FamO2O offers a statistically significant improvement over various existing methods, achieving state-of-the-art performance on the D4RL benchmark. Codes are available at https://github.com/LeapLabTHU/FamO2O.</details>

### Training Chain-of-Thought via Latent-Variable Inference

**Authors:** Matthew Douglas Hoffman, Du Phan, David Dohan, Sholto Douglas, Tuan Anh Le, Aaron Parisi, Pavel Sountsov, Charles Sutton, Sharad Vikram, Rif A. Saurous

**<details><summary>Abstract**</summary>

Large language models (LLMs) solve problems more accurately and interpretably when instructed to work out the answer step by step using a "chain-of-thought" (CoT) prompt. One can also improve LLMs' performance on a specific task by supervised fine-tuning, i.e., by using gradient ascent on some tunable parameters to maximize the average log-likelihood of correct answers from a labeled training set. Naively combining CoT with supervised tuning requires supervision not just of the correct answers, but also of detailed rationales that lead to those answers; these rationales are expensive to produce by hand. Instead, we propose a fine-tuning strategy that tries to maximize the \emph{marginal} log-likelihood of generating a correct answer using CoT prompting, approximately averaging over all possible rationales. The core challenge is sampling from the posterior over rationales conditioned on the correct answer; we address it using a simple Markov-chain Monte Carlo (MCMC) expectation-maximization (EM) algorithm inspired by the self-taught reasoner (STaR), memoized wake-sleep, Markovian score climbing, and persistent contrastive divergence. This algorithm also admits a novel control-variate technique that drives the variance of our gradient estimates to zero as the model improves. Applying our technique to GSM8K and the tasks in BIG-Bench Hard, we find that this MCMC-EM fine-tuning technique typically improves the model's accuracy on held-out examples more than STaR or prompt-tuning with or without CoT.</details>

### [Spotlight] Training shallow ReLU networks on noisy data using hinge loss: when do we overfit and is it benign?

**Authors:** Erin George, Michael Murray, William Swartworth, Deanna Needell

**<details><summary>Abstract**</summary>

We study benign overfitting in two-layer ReLU networks trained using gradient descent and hinge loss on noisy data for binary classification. In particular, we consider linearly separable data for which a relatively small proportion of labels are corrupted or flipped. We identify conditions on the margin of the clean data that give rise to three distinct training outcomes: benign overfitting, in which zero loss is achieved and with high probability test data is classified correctly; overfitting, in which zero loss is achieved but test data is misclassified with probability lower bounded by a constant; and non-overfitting, in which clean points, but not corrupt points, achieve zero loss and again with high probability test data is classified correctly. Our analysis provides a fine-grained description of the dynamics of neurons throughout training and reveals two distinct phases: in the first phase clean points achieve close to zero loss, in the second phase clean points oscillate on the boundary of zero loss while corrupt points either converge towards zero loss or are eventually zeroed by the network. We prove these results using a combinatorial approach that involves bounding the number of clean versus corrupt updates during these phases of training.</details>

### Training-free Diffusion Model Adaptation for Variable-Sized Text-to-Image Synthesis

**Authors:** Zhiyu Jin, Xuli Shen, Bin Li, Xiangyang Xue

**<details><summary>Abstract**</summary>

Diffusion models (DMs) have recently gained attention with state-of-the-art performance in text-to-image synthesis. Abiding by the tradition in deep learning, DMs are trained and evaluated on the images with fixed sizes. However, users are demanding for various images with specific sizes and various aspect ratio. This paper focuses on adapting text-to-image diffusion models to handle such variety while maintaining visual fidelity. First we observe that, during the synthesis, lower resolution images suffer from incomplete object portrayal, while higher resolution images exhibit repetitively disordered presentation. Next, we establish a statistical relationship indicating that attention entropy changes with token quantity, suggesting that models aggregate spatial information in proportion to image resolution. The subsequent interpretation on our observations is that objects are incompletely depicted due to limited spatial information for low resolutions, while repetitively disorganized presentation arises from redundant spatial information for high resolutions. From this perspective, we propose a scaling factor to alleviate the change of attention entropy and mitigate the defective pattern observed. Extensive experimental results validate the efficacy of the proposed scaling factor, enabling models to achieve better visual effects, image quality, and text alignment. Notably, these improvements are achieved without additional training or fine-tuning techniques.</details>

### Transformers learn through gradual rank increase

**Authors:** Emmanuel Abbe, Samy Bengio, Enric Boix-Adsera, Etai Littwin, Joshua Susskind

**<details><summary>Abstract**</summary>

We identify incremental learning dynamics in transformers, where the difference between trained and initial weights progressively increases in rank. We rigorously prove this occurs under the simplifying assumptions of diagonal weight matrices and small initialization. Our experiments support the theory and also show that phenomenon can occur in practice without the simplifying assumptions.</details>

### Tree-Rings Watermarks: Invisible Fingerprints for Diffusion Images

**Authors:** Yuxin Wen, John Kirchenbauer, Jonas Geiping, Tom Goldstein

**<details><summary>Abstract**</summary>

Watermarking the outputs of generative models is a crucial technique for tracing copyright and preventing potential harm from AI-generated content. In this paper, we introduce a novel technique called Tree-Ring Watermarking that robustly fingerprints diffusion model outputs.  Unlike existing methods that perform post-hoc modifications to images after sampling, Tree-Ring Watermarking subtly influences the entire sampling process, resulting in a model fingerprint that is invisible to humans. The watermark embeds a pattern into the initial noise vector used for sampling. These patterns are structured in Fourier space so that they are invariant to convolutions, crops, dilations, flips, and rotations.  After image generation, the watermark signal is detected by inverting the diffusion process to retrieve the noise vector, which is then checked for the embedded signal.  We demonstrate that this technique can be easily applied to arbitrary diffusion models, including text-conditioned Stable Diffusion, as a plug-in with negligible loss in FID. Our watermark is semantically hidden in the image space and is far more robust than watermarking alternatives that are currently deployed.</details>

### Triangulation Residual Loss for Data-efficient 3D Pose Estimation

**Authors:** Jiachen Zhao, Tao Yu, Liang An, Yipeng Huang, Fang Deng, Qionghai Dai

**<details><summary>Abstract**</summary>

This paper presents Triangulation Residual loss (TR loss) for multiview 3D pose estimation in a data-efficient manner. Existing 3D supervised models usually require large-scale 3D annotated datasets, but the amount of existing data is still insufficient to train supervised models to achieve ideal performance, especially for animal pose estimation. To employ unlabeled multiview data for training, previous epipolar-based consistency provides a self-supervised loss that considers only the local consistency in pairwise views, resulting in limited performance and heavy calculations. In contrast, TR loss enables self-supervision with global multiview geometric consistency. Starting from initial 2D keypoint estimates, the TR loss can fine-tune the corresponding 2D detector without 3D supervision by simply minimizing the smallest singular value of the triangulation matrix in an end-to-end fashion. Our method achieves the state-of-the-art 25.8mm MPJPE and competitive 28.7mm MPJPE with only 5\% 2D labeled training data on the Human3.6M dataset. Experiments on animals such as mice demonstrate our TR loss's data-efficient training ability.</details>

### Turbulence in Focus: Benchmarking Scaling Behavior of 3D Volumetric Super-Resolution with BLASTNet 2.0 Data

**Authors:** Wai Tong Chung, Bassem Akoush, Pushan Sharma, Alex Tamkin, Ki Sung Jung, Jacqueline Chen, Jack Guo, Davy Brouzet, Mohsen Talei, Bruno Savard, Alexei Poludnenko, Matthias Ihme

**<details><summary>Abstract**</summary>

Analysis of compressible turbulent flows is essential for applications related to propulsion, energy generation, and the environment. Here, we present BLASTNet 2.0, a 2.2 TB network-of-datasets containing 744 full-domain samples from 34 high-fidelity direct numerical simulations, which addresses the current limited availability of 3D high-fidelity reacting and non-reacting compressible turbulent flow simulation data. With this data, we benchmark a total of 49 variations of five deep learning approaches for 3D super-resolution - which can be applied for improving scientific imaging, simulations, turbulence models, as well as in computer vision applications. We perform neural scaling analysis on these models to examine the performance of different machine learning (ML) approaches, including two scientific ML techniques. We demonstrate that (i) predictive performance can scale with model size and cost, (ii) architecture matters significantly, especially for smaller models, and (iii) the benefits of physics-based losses can persist with increasing model size. The outcomes of this benchmark study are anticipated to offer insights that can aid the design of 3D super-resolution models, especially for turbulence models, while this data is expected to foster ML methods for a broad range of flow physics applications. This data is publicly available with download links and browsing tools consolidated at https://blastnet.github.io.</details>

### UUKG: Unified Urban Knowledge Graph Dataset for Urban Spatiotemporal Prediction

**Authors:** Yansong Ning, Hao Liu, Hao Wang, Zhenyu Zeng, Hui Xiong

**<details><summary>Abstract**</summary>

Accurate Urban SpatioTemporal Prediction (USTP) is of great importance to the development and operation of the smart city. As an emerging building block, multi-sourced urban data are usually integrated as urban knowledge graphs (UrbanKGs) to provide critical knowledge for urban spatiotemporal prediction models. However, existing UrbanKGs are often tailored for specific downstream prediction tasks and are not publicly available, which limits the potential advancement. This paper presents UUKG, the unified urban knowledge graph dataset for knowledge-enhanced urban spatiotemporal predictions. Specifically, we first construct UrbanKGs consisting of millions of triplets for two metropolises by connecting heterogeneous urban entities such as administrative boroughs, POIs, and road segments. Moreover, we conduct qualitative and quantitative analysis on constructed UrbanKGs and uncover diverse high-order structural patterns, such as hierarchies and cycles, that can be leveraged to benefit downstream USTP tasks. To validate and facilitate the use of UrbanKGs, we implement and evaluate 15 KG embedding methods on the KG completion task and integrate the learned KG embeddings into 9 spatiotemporal models for five different USTP tasks. The extensive experimental results not only provide benchmarks of knowledge-enhanced USTP models under different task settings but also highlight the potential of state-of-the-art high-order structure-aware UrbanKG embedding methods. We hope the proposed UUKG fosters research on urban knowledge graphs and broad smart city applications. The dataset and source code are available at https://github.com/usail-hkust/UUKG/.</details>

### [Spotlight] Uncovering motifs of concurrent signaling across multiple neuronal populations

**Authors:** Evren Gokcen, Anna Jasper, Alison Xu, Adam Kohn, Christian Machens, Byron M Yu

**<details><summary>Abstract**</summary>

Modern recording techniques now allow us to record from distinct neuronal populations in different brain networks. However, especially as we consider multiple (more than two) populations, new conceptual and statistical frameworks are needed to characterize the multi-dimensional, concurrent flow of signals among these populations. Here, we develop a dimensionality reduction framework that determines (1) the subset of populations described by each latent dimension, (2) the direction of signal flow among those populations, and (3) how those signals evolve over time within and across experimental trials. We illustrate these features in simulation, and further validate the method by applying it to previously studied recordings from neuronal populations in macaque visual areas V1 and V2. Then we study interactions across select laminar compartments of areas V1, V2, and V3d, recorded simultaneously with multiple Neuropixels probes. Our approach uncovered signatures of selective communication across these three areas that related to their retinotopic alignment. This work advances the study of concurrent signaling across multiple neuronal populations.</details>

### Understanding and Mitigating Copying in Diffusion Models

**Authors:** Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, Tom Goldstein

**<details><summary>Abstract**</summary>

Images generated by diffusion models like Stable Diffusion are increasingly widespread. Recent works and even lawsuits have shown that these models are prone to replicating their training data, unbeknownst to the user. In this paper, we first analyze this memorization problem in text-to-image diffusion models. While it is widely believed that duplicated images in the training set are responsible for content replication at inference time, we observe that the text conditioning of the model plays a similarly important role. In fact, we see in our experiments that data replication often does not happen for unconditional models, while it is common in the text-conditional case. Motivated by our findings, we then propose several techniques for reducing data replication at both training and inference time by randomizing and augmenting image captions in the training set. Code is available at https://github.com/somepago/DCR.</details>

### Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models

**Authors:** Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, Kwan-Yee K. Wong

**<details><summary>Abstract**</summary>

Text-to-Image diffusion models have made tremendous progress over the past two years, enabling the generation of highly realistic images based on open-domain text descriptions. However, despite their success, text descriptions often struggle to adequately convey detailed controls, even when composed of long and complex texts. Moreover, recent studies have also shown that these models face challenges in understanding such complex texts and generating the corresponding images. Therefore, there is a growing need to enable more control modes beyond text description. In this paper, we introduce Uni-ControlNet, a unified framework that allows for the simultaneous utilization of different local controls (e.g., edge maps, depth map, segmentation masks) and global controls (e.g., CLIP image embeddings) in a flexible and composable manner within one single model. Unlike existing methods, Uni-ControlNet only requires the fine-tuning of two additional adapters upon frozen pre-trained text-to-image diffusion models, eliminating the huge cost of training from scratch. Moreover, thanks to some dedicated adapter designs, Uni-ControlNet only necessitates a constant number (i.e., 2) of adapters, regardless of the number of local or global controls used. This not only reduces the fine-tuning costs and model size, making it more suitable for real-world deployment, but also facilitate composability of different conditions. Through both quantitative and qualitative comparisons, Uni-ControlNet demonstrates its superiority over existing methods in terms of controllability, generation quality and composability. Code is available at https://github.com/ShihaoZhaoZSH/Uni-ControlNet.</details>

### UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models

**Authors:** Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, Jiwen Lu

**<details><summary>Abstract**</summary>

Diffusion probabilistic models (DPMs) have demonstrated a very promising ability in high-resolution image synthesis. However, sampling from a pre-trained DPM is time-consuming due to the multiple evaluations of the denoising network, making it more and more important to accelerate the sampling of DPMs. Despite recent progress in designing fast samplers, existing methods still cannot generate satisfying images in many applications where fewer steps (e.g., $<$10) are favored. In this paper, we develop a unified corrector (UniC) that can be applied after any existing DPM sampler to increase the order of accuracy without extra model evaluations, and derive a unified predictor (UniP) that supports arbitrary order as a byproduct. Combining UniP and UniC, we propose a unified predictor-corrector framework called UniPC for the fast sampling of DPMs, which has a unified analytical form for any order and can significantly improve the sampling quality over previous methods, especially in extremely few steps. We evaluate our methods through extensive experiments including both unconditional and conditional sampling using pixel-space and latent-space DPMs. Our UniPC can achieve 3.87 FID on CIFAR10 (unconditional) and 7.51 FID on ImageNet 256$	imes$256 (conditional) with only 10 function evaluations. Code is available at https://github.com/wl-zhao/UniPC.</details>

### UniT: A Unified Look at Certified Robust Training against Text Adversarial Perturbation

**Authors:** Muchao Ye, Ziyi Yin, Tianrong Zhang, Tianyu Du, Jinghui Chen, Ting Wang, Fenglong Ma

**<details><summary>Abstract**</summary>

Recent years have witnessed a surge of certified robust training pipelines against text adversarial perturbation constructed by synonym substitutions. Given a base model, existing pipelines provide prediction certificates either in the discrete word space or the continuous latent space. However, they are isolated from each other with a structural gap. We observe that existing training frameworks need unification to provide stronger certified robustness. Additionally,  they mainly focus on building the certification process but neglect to improve the robustness of the base model. To mitigate the aforementioned limitations, we propose a unified framework named UniT that enables us to train flexibly in either fashion by working in the word embedding space. It can provide a stronger robustness guarantee obtained directly from the word embedding space without extra modules. In addition, we introduce the decoupled regularization (DR) loss to improve the robustness of the base model, which includes two separate robustness regularization terms for the feature extraction and classifier modules. Experimental results on widely used text classification datasets further demonstrate the effectiveness of the designed unified framework and the proposed DR loss for improving the certified robust accuracy.</details>

### Unified Lower Bounds for Interactive High-dimensional Estimation under Information Constraints

**Authors:** Jayadev Acharya, Clément L Canonne, Ziteng Sun, Himanshu Tyagi

**<details><summary>Abstract**</summary>

We consider distributed parameter estimation using interactive protocols subject to local information constraints such as bandwidth limitations, local differential privacy, and restricted measurements. We provide a unified framework enabling us to derive a variety of (tight) minimax lower bounds for different parametric families of distributions, both continuous and discrete, under any $\ell_p$ loss. Our lower bound framework is versatile and yields “plug-and-play” bounds that are widely applicable to a large range of estimation problems, and, for the prototypical case of the Gaussian family, circumvents limitations of previous techniques. In particular, our approach recovers bounds obtained using data processing inequalities and Cramér–Rao bounds, two other alternative approaches for proving lower bounds in our setting of interest. Further, for the families considered, we complement our lower bounds with matching upper bounds.</details>

### Unified Segment-to-Segment Framework for Simultaneous Sequence Generation

**Authors:** Shaolei Zhang, Yang Feng

**<details><summary>Abstract**</summary>

Simultaneous sequence generation is a pivotal task for real-time scenarios, such as streaming speech recognition, simultaneous machine translation and simultaneous speech translation, where the target sequence is generated while receiving the source sequence. The crux of achieving high-quality generation with low latency lies in identifying the optimal moments for generating, accomplished by learning a mapping between the source and target sequences. However, existing methods often rely on task-specific heuristics for different sequence types, limiting the model’s capacity to adaptively learn the source-target mapping and hindering the exploration of multi-task learning for various simultaneous tasks. In this paper, we propose a unified segment-to-segment framework (Seg2Seg) for simultaneous sequence generation, which learns the mapping in an adaptive and unified manner. During the process of simultaneous generation, the model alternates between waiting for a source segment and generating a target segment, making the segment serve as the natural bridge between the source and target. To accomplish this, Seg2Seg introduces a latent segment as the pivot between source to target and explores all potential source-target mappings via the proposed expectation training, thereby learning the optimal moments for generating. Experiments on multiple simultaneous generation tasks demonstrate that Seg2Seg achieves state-of-the-art performance and exhibits better generality across various tasks.</details>

### Unsupervised Polychromatic Neural Representation for CT Metal Artifact Reduction

**Authors:** Qing Wu, Lixuan Chen, Ce Wang, Hongjiang Wei, S. Kevin Zhou, Jingyi Yu, Yuyao Zhang

**<details><summary>Abstract**</summary>

Emerging neural reconstruction techniques based on tomography (e.g., NeRF, NeAT, and NeRP) have started showing unique capabilities in medical imaging. In this work, we present a novel Polychromatic neural representation (Polyner) to tackle the challenging problem of CT imaging when metallic implants exist within the human body. CT metal artifacts arise from the drastic variation of metal's attenuation coefficients at various energy levels of the X-ray spectrum, leading to a nonlinear metal effect in CT measurements. Recovering CT images from metal-affected measurements hence poses a complicated nonlinear inverse problem where empirical models adopted in previous metal artifact reduction (MAR) approaches lead to signal loss and strongly aliased reconstructions. Polyner instead models the MAR problem from a nonlinear inverse problem perspective. Specifically, we first derive a polychromatic forward model to accurately simulate the nonlinear CT acquisition process. Then, we incorporate our forward model into the implicit neural representation to accomplish reconstruction. Lastly, we adopt a regularizer to preserve the physical properties of the CT images across different energy levels while effectively constraining the solution space. Our Polyner is an unsupervised method and does not require any external training data. Experimenting with multiple datasets shows that our Polyner achieves comparable or better performance than supervised methods on in-domain datasets while demonstrating significant performance improvements on out-of-domain datasets. To the best of our knowledge, our Polyner is the first unsupervised MAR method that outperforms its supervised counterparts. The code for this work is available at: https://github.com/iwuqing/Polyner.</details>

### VPP: Efficient Conditional 3D Generation via Voxel-Point Progressive Representation

**Authors:** Zekun Qi, Muzhou Yu, Runpei Dong, Kaisheng Ma

**<details><summary>Abstract**</summary>

Conditional 3D generation is undergoing a significant advancement, enabling the free creation of 3D content from inputs such as text or 2D images. However, previous approaches have suffered from low inference efficiency, limited generation categories, and restricted downstream applications. In this work, we revisit the impact of different 3D representations on generation quality and efficiency. We propose a progressive generation method through Voxel-Point Progressive Representation (VPP). VPP leverages structured voxel representation in the proposed Voxel Semantic Generator and the sparsity of unstructured point representation in the Point Upsampler, enabling efficient generation of multi-category objects. VPP can generate high-quality 8K point clouds within 0.2 seconds. Additionally, the masked generation Transformer allows for various 3D downstream tasks, such as generation, editing, completion, and pre-training. Extensive experiments demonstrate that VPP efficiently generates high-fidelity and diverse 3D shapes across different categories, while also exhibiting excellent representation transfer performance. Codes will be released at https://github.com/qizekun/VPP.</details>

### VTaC: A  Benchmark Dataset of Ventricular Tachycardia Alarms from ICU Monitors

**Authors:** Li-wei Lehman, Benjamin Moody, Harsh Deep, Feng Wu, Hasan Saeed, Lucas McCullum, Diane Perry, Tristan Struja, Qiao Li, Gari Clifford, Roger Mark

**<details><summary>Abstract**</summary>

False arrhythmia alarms in intensive care units (ICUs) are a continuing problem despite considerable effort from industrial and academic algorithm developers. Of all life-threatening arrhythmias, ventricular tachycardia (VT) stands out as the most challenging arrhythmia to detect reliably. We introduce a new annotated VT alarm database, VTaC (Ventricular Tachycardia annotated alarms from ICUs)  consisting of  over 5,000 waveform recordings with VT alarms triggered by bedside monitors in the ICU. Each VT alarm waveform in the dataset has been labeled by at least two independent human expert annotators. The dataset encompasses data collected from ICUs in two major US hospitals and includes data from three leading bedside monitor manufacturers, providing a diverse and representative collection of alarm waveform data. Each waveform recording comprises at least two electrocardiogram (ECG) leads and one or more pulsatile waveforms, such as photoplethysmogram (PPG or PLETH) and arterial blood pressure (ABP) waveforms. We demonstrate the utility of this new benchmark dataset for the task of false arrhythmia alarm reduction, and present performance of multiple machine learning approaches, including conventional supervised machine learning, deep learning, semi-supervised learning, and generative approaches for the task of VT false alarm reduction.</details>

### Validated Image Caption Rating Dataset

**Authors:** Lothar D Narins, Andrew Scott, Aakash Gautam, Anagha Kulkarni, Mar Castanon, Benjamin Kao, Shasta Ihorn, Yue-Ting Siu, James M. Mason, Alexander Blum, Ilmi Yoon

**<details><summary>Abstract**</summary>

We present a new high-quality validated image caption rating (VICR) dataset. How well a caption fits an image can be difficult to assess due to the subjective nature of caption quality. How do we evaluate whether a caption is good? We generated a new dataset to help answer this question by using our new image caption rating system, which consists of a novel robust rating scale and gamified approach to gathering human ratings. We show that our approach is consistent and teachable. 113 participants were involved in generating the dataset, which is composed of 68,217 ratings among 15,646 image-caption pairs. Our new dataset has greater inter-rater agreement than the state of the art, and custom machine learning rating predictors that were trained on our dataset outperform previous metrics. We improve over Flickr8k-Expert in Kendall's $W$ by 12\% and in Fleiss' $\kappa$ by 19\%, and thus provide a new benchmark dataset for image caption rating.</details>

### Variational Imbalanced Regression: Fair Uncertainty Quantification via Probabilistic Smoothing

**Authors:** Ziyan Wang, Hao Wang

**<details><summary>Abstract**</summary>

Existing regression models tend to fall short in both accuracy and uncertainty estimation when the label distribution is imbalanced. In this paper, we propose a probabilistic deep learning model, dubbed variational imbalanced regression (VIR), which not only performs well in imbalanced regression but naturally produces reasonable uncertainty estimation as a byproduct. Different from typical variational autoencoders assuming I.I.D. representations (a data point's representation is not directly affected by other data points), our VIR borrows data with similar regression labels to compute the latent representation's variational distribution; furthermore, different from deterministic regression models producing point estimates, VIR predicts the entire normal-inverse-gamma distributions and modulates the associated conjugate distributions to impose probabilistic reweighting on the imbalanced data, thereby providing better uncertainty estimation. Experiments in several real-world datasets show that our VIR can outperform state-of-the-art imbalanced regression models in terms of both accuracy and uncertainty estimation. Code will soon be available at https://github.com/Wang-ML-Lab/variational-imbalanced-regression.</details>

### Video-Mined Task Graphs for Keystep Recognition in Instructional Videos

**Authors:** Kumar Ashutosh, Santhosh Kumar Ramakrishnan, Triantafyllos Afouras, Kristen Grauman

**<details><summary>Abstract**</summary>

Procedural activity understanding requires perceiving human actions in terms of a broader task, where multiple keysteps are performed in sequence across a long video to reach a final goal state---such as the steps of a recipe or the steps of a DIY fix-it task.  Prior work largely treats keystep recognition in isolation of this broader structure, or else rigidly confines keysteps to align with a particular sequential script.  We propose discovering a task graph automatically from how-to videos to represent probabilistically how people tend to execute keysteps, then leverage this graph to regularize keystep recognition in novel videos.  On multiple datasets of real-world instructional video, we show the impact: more reliable zero-shot keystep localization and improved video representation learning, exceeding the state of the art.</details>

### VisIT-Bench: A Dynamic Benchmark for Evaluating Instruction-Following Vision-and-Language Models

**Authors:** Yonatan Bitton, Hritik Bansal, Jack Hessel, Rulin Shao, Wanrong Zhu, Anas Awadalla, Josh Gardner, Rohan Taori, Ludwig Schmidt

**<details><summary>Abstract**</summary>

We introduce VisIT-Bench (Visual InsTruction Benchmark), a benchmark for evaluating instruction-following vision-language models for real-world use. Our starting point is curating 70 "instruction families" that we envision instruction tuned vision-language models should be able to address. Extending beyond evaluations like VQAv2 and COCO, tasks range from basic recognition to game playing and creative generation. Following curation, our dataset comprises 592 test queries, each with a human-authored instruction-conditioned caption. These descriptions surface instruction-specific factors, e.g., for an instruction asking about the accessibility of a storefront for wheelchair users, the instruction-conditioned caption describes ramps/potential obstacles. These descriptions enable 1) collecting human-verified reference outputs for each instance; and 2) automatic evaluation of candidate multimodal generations using a text-only LLM, aligning with human judgment. We quantify quality gaps between models and references using both human and automatic evaluations; e.g., the top-performing instruction-following model wins against the GPT-4 reference in just 27% of the comparison. VisIT-Bench is dynamic to participate, practitioners simply submit their model's response on the project website; Data, code and leaderboard is available at https://visit-bench.github.io/.</details>

### WalkLM: A Uniform Language Model Fine-tuning Framework for Attributed Graph Embedding

**Authors:** Yanchao Tan, Zihao Zhou, Hang Lv, Weiming Liu, Carl Yang

**<details><summary>Abstract**</summary>

Graphs are widely used to model interconnected entities and improve downstream predictions in various real-world applications. However, real-world graphs nowadays are often associated with complex attributes on multiple types of nodes and even links that are hard to model uniformly, while the widely used graph neural networks (GNNs) often require sufficient training toward specific downstream predictions to achieve strong performance. In this work, we take a fundamentally different approach than GNNs, to simultaneously achieve deep joint modeling of complex attributes and flexible structures of real-world graphs and obtain unsupervised generic graph representations that are not limited to specific downstream predictions. Our framework, built on a natural integration of language models (LMs) and random walks (RWs), is straightforward, powerful and data-efficient. Specifically, we first perform attributed RWs on the graph and design an automated program to compose roughly meaningful textual sequences directly from the attributed RWs; then we fine-tune an LM using the RW-based textual sequences and extract embedding vectors from the LM, which encapsulates both attribute semantics and graph structures. In our experiments, we evaluate the learned node embeddings towards different downstream prediction tasks on multiple real-world attributed graph datasets and observe significant improvements over a comprehensive set of state-of-the-art unsupervised node embedding methods. We believe this work opens a door for more sophisticated technical designs and empirical evaluations toward the leverage of LMs for the modeling of real-world graphs.</details>

### Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research

**Authors:** Cole Gulino, Justin Fu, Wenjie Luo, George Tucker, Eli Bronstein, Yiren Lu, Jean Harb, Xinlei Pan, Yan Wang, Xiangyu Chen, John Co-Reyes, Rishabh Agarwal, Rebecca Roelofs, Yao Lu, Nico Montali, Paul Mougin, Zoey Yang, Brandyn White, Aleksandra Faust, Rowan McAllister, Dragomir Anguelov, Benjamin Sapp

**<details><summary>Abstract**</summary>

Simulation is an essential tool to develop and benchmark autonomous vehicle planning software in a safe and cost-effective manner. However, realistic simulation requires accurate modeling of multi-agent interactive behaviors to be trustworthy, behaviors which can be highly nuanced and complex. To address these challenges, we introduce Waymax, a new data-driven simulator for autonomous driving in multi-agent scenes, designed for large-scale simulation and testing.  Waymax uses publicly-released, real-world driving data (e.g., the Waymo Open Motion Dataset) to initialize or play back a diverse set of multi-agent simulated scenarios.   It runs entirely on hardware accelerators such as TPUs/GPUs and supports in-graph simulation for training, making it suitable for modern large-scale, distributed machine learning workflows. To support online training and evaluation, Waymax includes several learned and hard-coded behavior models that allow for realistic interaction within simulation. To supplement Waymax, we benchmark a suite of popular imitation and reinforcement learning algorithms with ablation studies on different design decisions, where we highlight the effectiveness of routes as guidance for planning agents and the ability of RL to overfit against simulated agents.</details>

### Weitzman's Rule for Pandora's Box with Correlations

**Authors:** Evangelia Gergatsouli, Christos Tzamos

**<details><summary>Abstract**</summary>

Pandora’s Box is a central problem in decision making under uncertainty that can model various real life scenarios. In this problem we are given n boxes, each with a fixed opening cost, and an unknown value drawn from a known distribution, only revealed if we pay the opening cost. Our goal is to find a strategy for opening boxes to minimize the sum of the value selected and the opening cost paid.In this work we revisit Pandora’s Box when the value distributions are correlated, first studied in [CGT+20]. We show that the optimal algorithm for the independent case, given by Weitzman’s rule, directly works for the correlated case. In fact, our algorithm results in significantly improved approximation guarantees compared to the previous work, while also being substantially simpler. We also show how to implement the rule given only sample access to the correlated distribution of values. Specifically, we find that a number of samples that is polynomial in the number of boxes is sufficient for the algorithm to work.</details>

### What Distributions are Robust to Indiscriminate Poisoning Attacks for Linear Learners?

**Authors:** Fnu Suya, Xiao Zhang, Yuan Tian, David Evans

**<details><summary>Abstract**</summary>

We study indiscriminate poisoning for linear learners where an adversary injects a few  crafted examples into the training data with the goal of forcing the induced model to incur higher test error. Inspired by the observation that linear learners on some datasets are able to resist the best known attacks even without any defenses, we further investigate whether datasets can be inherently robust to indiscriminate poisoning attacks for linear learners. For theoretical Gaussian distributions, we rigorously characterize the behavior of an optimal poisoning attack, defined as the poisoning strategy that attains the maximum risk of the induced model at a given poisoning budget. Our results prove that linear learners can indeed be robust to indiscriminate poisoning if the class-wise data distributions are well-separated with low variance and the size of the constraint set containing all permissible poisoning points is also small. These findings largely explain the drastic variation in empirical attack performance of the state-of-the-art poisoning attacks on linear learners across benchmark datasets, making an important initial step towards understanding the underlying reasons some learning tasks are vulnerable to data poisoning attacks.</details>

### What Truly Matters in Trajectory Prediction for Autonomous Driving?

**Authors:** Tran Phong, Haoran Wu, Cunjun Yu, Panpan Cai, Sifa Zheng, David Hsu

**<details><summary>Abstract**</summary>

Trajectory prediction plays a vital role in the performance of autonomous driving systems, and prediction accuracy, such as average displacement error (ADE) or final displacement error (FDE), is widely used as a performance metric. However, a significant disparity exists between the accuracy of predictors on fixed datasets and driving performance when the predictors are used downstream for vehicle control, because of a dynamics gap. In the real world, the prediction algorithm influences the behavior of the ego vehicle, which, in turn, influences the behaviors of other vehicles nearby. This interaction results in predictor-specific dynamics that directly impacts prediction results. In fixed datasets, since other vehicles' responses are predetermined, this interaction effect is lost, leading to a significant dynamics gap. This paper studies the overlooked significance of this dynamics gap. We also examine several other factors contributing to the disparity between prediction performance and driving performance. The findings highlight the trade-off between the predictor's computational efficiency and prediction accuracy in determining real-world driving performance. In summary, an interactive, task-driven evaluation protocol for trajectory prediction is crucial to capture its effectiveness for autonomous driving. Source code along with experimental settings is available online (https://whatmatters23.github.io/).</details>

### What You See is What You Read? Improving Text-Image Alignment Evaluation

**Authors:** Michal Yarom, Yonatan Bitton, Soravit Changpinyo, Roee Aharoni, Jonathan Herzig, Oran Lang, Eran Ofek, Idan Szpektor

**<details><summary>Abstract**</summary>

Automatically determining whether a text and a corresponding image are semantically aligned is a significant challenge for vision-language models, with applications in generative text-to-image and image-to-text tasks. In this work, we study methods for automatic text-image alignment evaluation. We first introduce SeeTRUE: a comprehensive evaluation set, spanning multiple datasets from both text-to-image and image-to-text generation tasks, with human judgements for whether a given text-image pair is semantically aligned. We then describe two automatic methods to determine alignment: the first involving a pipeline based on question generation and visual question answering models, and the second employing an end-to-end classification approach by finetuning multimodal pretrained models. Both methods surpass prior approaches in various text-image alignment tasks, with significant improvements in challenging cases that involve complex composition or unnatural images. Finally, we demonstrate how our approaches can localize specific misalignments between an image and a given text, and how they can be used to automatically re-rank candidates in text-to-image generation.</details>

### What a MESS: Multi-Domain Evaluation of Zero-Shot Semantic Segmentation

**Authors:** Benedikt Blumenstiel, Johannes Jakubik, Hilde Kuehne, Michael Vössing

**<details><summary>Abstract**</summary>

While semantic segmentation has seen tremendous improvements in the past, there are still significant labeling efforts necessary and the problem of limited generalization to classes that have not been present during training. To address this problem, zero-shot semantic segmentation makes use of large self-supervised vision-language models, allowing zero-shot transfer to unseen classes. In this work, we build a benchmark for Multi-domain Evaluation of Zero-Shot Semantic Segmentation (MESS), which allows a holistic analysis of performance across a wide range of domain-specific datasets such as medicine, engineering, earth monitoring, biology, and agriculture. To do this, we reviewed 120 datasets, developed a taxonomy, and classified the datasets according to the developed taxonomy. We select a representative subset consisting of 22 datasets and propose it as the MESS benchmark. We evaluate eight recently published models on the proposed MESS benchmark and analyze characteristics for the performance of zero-shot transfer models. The toolkit is available at https://github.com/blumenstiel/MESS.</details>

### What’s Left? Concept Grounding with Logic-Enhanced Foundation Models

**Authors:** Joy Hsu, Jiayuan Mao, Josh Tenenbaum, Jiajun Wu

**<details><summary>Abstract**</summary>

Recent works such as VisProg and ViperGPT have smartly composed foundation models for visual reasoning—using large language models (LLMs) to produce programs that can be executed by pre-trained vision-language models. However, they operate in limited domains, such as 2D images, not fully exploiting the generalization of language: abstract concepts like “*left*” can also be grounded in 3D, temporal, and action data, as in moving to your *left*. This limited generalization stems from these inference-only methods’ inability to learn or adapt pre-trained models to a new domain. We propose the **L**ogic-**E**nhanced **F**ounda**T**ion Model (**LEFT**), a unified framework that *learns* to ground and reason with concepts across domains with a differentiable, domain-independent, first-order logic-based program executor. LEFT has an LLM interpreter that outputs a program represented in a general, logic-based reasoning language, which is shared across all domains and tasks. LEFT’s executor then executes the program with trainable domain-specific grounding modules. We show that LEFT flexibly learns concepts in four domains: 2D images, 3D scenes, human motions, and robotic manipulation. It exhibits strong reasoning ability in a wide variety of tasks, including those that are complex and not seen during training, and can be easily applied to new domains.</details>

### [Oral] When Demonstrations meet Generative World Models: A Maximum Likelihood Framework for Offline Inverse Reinforcement Learning

**Authors:** Siliang Zeng, Chenliang Li, Alfredo Garcia, Mingyi Hong

**Oral Presentation:** Tu, Dec 12, 08:15 -- Oral 1A

**<details><summary>Abstract**</summary>

Offline inverse reinforcement learning (Offline IRL) aims to recover the structure of rewards and environment dynamics that underlie observed actions in a fixed, finite set of demonstrations from an expert agent. Accurate models of expertise in executing a task has applications in safety-sensitive applications such as clinical decision making and autonomous driving. However, the structure of an expert's preferences implicit in observed actions is closely linked to the expert's model of the environment dynamics (i.e. the ``world''). Thus, inaccurate models of the world obtained from finite data with limited coverage could compound inaccuracy in estimated rewards. To address this issue, we propose a bi-level optimization formulation of the estimation task wherein the upper level is likelihood maximization based upon a conservative model of the expert's policy (lower level). The policy model is conservative in that it maximizes reward subject to a penalty that is increasing in the uncertainty of the estimated model of the world. We propose a new algorithmic framework to solve the bi-level optimization problem formulation and provide statistical and computational guarantees of performance for the associated optimal reward estimator. Finally,  we demonstrate that the proposed algorithm outperforms the state-of-the-art offline IRL and imitation learning benchmarks by a large margin, over the continuous control tasks in MuJoCo and different datasets in the D4RL benchmark.</details>

### When Does Confidence-Based Cascade Deferral Suffice?

**Authors:** Wittawat Jitkrittum, Neha Gupta, Aditya Menon, Harikrishna Narasimhan, Ankit Rawat, Sanjiv Kumar

**<details><summary>Abstract**</summary>

Cascades are a classical strategy to enable inference cost to vary adaptively across samples, wherein a sequence of classifiers are invoked in turn.  A deferral rule determines whether to invoke the next classifier in the sequence, or to terminate prediction.  One simple deferral rule employs the confidence of the current classifier, e.g., based on the maximum predicted softmax probability.  Despite being oblivious to the structure of the cascade --- e.g., not modelling the errors of downstream models --- such confidence-based deferral often works remarkably well in practice.  In this paper, we seek to better understand the conditions under which confidence-based deferral may fail, and when alternate deferral strategies can perform better.  We first present a theoretical characterisation of the optimal deferral rule, which precisely characterises settings under which confidence-based deferral may suffer.  We then study post-hoc deferral mechanisms, and demonstrate they can significantly improve upon confidence-based deferral in settings where (i)  downstream models are specialists that only work well on a subset of inputs, (ii) samples are subject to label noise, and (iii) there is distribution shift between the train and test set.</details>

### When is Agnostic Reinforcement Learning Statistically Tractable?

**Authors:** Zeyu Jia, Gene Li, Alexander Rakhlin, Ayush Sekhari, Nati Srebro

**<details><summary>Abstract**</summary>

We study the problem of agnostic PAC reinforcement learning (RL): given a policy class $\Pi$, how many rounds of interaction with an unknown MDP (with a potentially large state and action space) are required to learn an $\epsilon$-suboptimal policy with respect to \(\Pi\)? Towards that end, we introduce a new complexity measure, called the \emph{spanning capacity}, that depends solely on the set \(\Pi\) and is independent of the MDP dynamics. With a generative model, we show that the spanning capacity characterizes PAC learnability for every policy class $\Pi$. However, for online RL, the situation is more subtle. We show there exists a policy class $\Pi$ with a bounded spanning capacity that requires a superpolynomial number of samples to learn. This reveals a surprising separation for agnostic learnability between generative access and online access models (as well as between deterministic/stochastic MDPs under online access). On the positive side, we identify an additional \emph{sunflower} structure which in conjunction with bounded spanning capacity enables statistically efficient online RL via a new algorithm called POPLER, which takes inspiration from classical importance sampling methods as well as recent developments for reachable-state identification and policy evaluation in reward-free exploration.</details>

### Where Did I Come From? Origin Attribution of AI-Generated Images

**Authors:** Zhenting Wang, Chen Chen, Yi Zeng, Lingjuan Lyu, Shiqing Ma

**<details><summary>Abstract**</summary>

Image generation techniques have been gaining increasing attention recently, but concerns have been raised about the potential misuse and intellectual property (IP) infringement associated with image generation models. It is, therefore, necessary to analyze the origin of images by inferring if a specific image was generated by a particular model, i.e., origin attribution. Existing methods only focus on specific types of generative models and require additional procedures during the training phase or generation phase. This makes them unsuitable for pre-trained models that lack these specific operations and may impair generation quality. To address this problem, we first develop an alteration-free and model-agnostic origin attribution method via reverse-engineering on image generation models, i.e., inverting the input of a particular model for a specific image. Given a particular model, we first analyze the differences in the hardness of reverse-engineering tasks for generated samples of the given model and other images. Based on our analysis, we then propose a method that utilizes the reconstruction loss of reverse-engineering to infer the origin. Our proposed method effectively distinguishes between generated images of a specific generative model and other images, i.e., images generated by other models and real images.</details>

### Why Does Sharpness-Aware Minimization Generalize Better Than SGD?

**Authors:** Zixiang Chen, Junkai Zhang, Yiwen Kou, Xiangning Chen, Cho-Jui Hsieh, Quanquan Gu

**<details><summary>Abstract**</summary>

The challenge of overfitting, in which the model memorizes the training data and fails to generalize to test data, has become increasingly significant in the training of large neural networks. To tackle this challenge, Sharpness-Aware Minimization (SAM) has emerged as a promising training method, which can improve the generalization of neural networks even in the presence of label noise. However, a deep understanding of how SAM works, especially in the setting of nonlinear neural networks and classification tasks, remains largely missing. This paper fills this gap by demonstrating why SAM generalizes better than Stochastic Gradient Descent (SGD) for a certain data model and two-layer convolutional ReLU networks. The loss landscape of our studied problem is nonsmooth, thus current explanations for the success of SAM based on the Hessian information are insufficient. Our result explains the benefits of SAM, particularly its ability to prevent noise learning in the early stages, thereby facilitating more effective learning of features. Experiments on both synthetic and real data corroborate our theory.</details>

### WordScape: a Pipeline to extract multilingual, visually rich Documents with Layout Annotations from Web Crawl Data

**Authors:** Maurice Weber, Carlo Siebenschuh, Rory Butler, Anton Alexandrov, Valdemar Thanner, Georgios Tsolakis, Haris Jabbar, Ian Foster, Bo Li, Rick Stevens, Ce Zhang

**<details><summary>Abstract**</summary>

We introduce WordScape, a novel pipeline for the creation of cross-disciplinary, multilingual corpora comprising millions of pages with annotations for document layout detection. Relating visual and textual items on document pages has gained further significance with the advent of multimodal models. Various approaches proved effective for visual question answering or layout segmentation. However, the interplay of text, tables, and visuals remains challenging for a variety of document understanding tasks. In particular, many models fail to generalize well to diverse domains and new languages due to insufficient availability of training data. WordScape addresses these limitations. Our automatic annotation pipeline parses the Open XML structure of Word documents obtained from the web, jointly providing layout-annotated document images and their textual representations. In turn, WordScape offers unique properties as it (1) leverages the ubiquity of the Word file format on the internet, (2) is readily accessible through the Common Crawl web corpus, (3) is adaptive to domain-specific documents, and (4) offers culturally and linguistically diverse document pages with natural semantic structure and high-quality text. Together with the pipeline, we will additionally release 9.5M urls to word documents which can be processed using WordScape to create a dataset of over 40M pages. Finally, we investigate the quality of text and layout annotations extracted by WordScape, assess the impact on document understanding benchmarks, and demonstrate that manual labeling costs can be substantially reduced.</details>

### Wyze Rule: Federated Rule Dataset for Rule Recommendation Benchmarking

**Authors:** Mohammad Mahdi Kamani, Yuhang Yao, Hanjia Lyu, Zhongwei Cheng, Lin Chen, Liangju Li, Carlee Joe-Wong, Jiebo Luo

**<details><summary>Abstract**</summary>

In the rapidly evolving landscape of smart home automation, the potential of IoT devices is vast. In this realm, rules are the main tool utilized for this automation, which are predefined conditions or triggers that establish connections between devices, enabling seamless automation of specific processes. However, one significant challenge researchers face is the lack of comprehensive datasets to explore and advance the field of smart home rule recommendations. These datasets are essential for developing and evaluating intelligent algorithms that can effectively recommend rules for automating processes while preserving the privacy of the users, as it involves personal information about users' daily lives. To bridge this gap, we present the Wyze Rule Dataset, a large-scale dataset designed specifically for smart home rule recommendation research. Wyze Rule encompasses over 1 million rules gathered from a diverse user base of 300,000 individuals from Wyze Labs, offering an extensive and varied collection of real-world data. With a focus on federated learning, our dataset is tailored to address the unique challenges of a cross-device federated learning setting in the recommendation domain, featuring a large-scale number of clients with widely heterogeneous data. To establish a benchmark for comparison and evaluation, we have meticulously implemented multiple baselines in both centralized and federated settings. Researchers can leverage these baselines to gauge the performance and effectiveness of their rule recommendation systems, driving advancements in the domain. The Wyze Rule Dataset is publicly accessible through [HuggingFace](https://huggingface.co/datasets/wyzelabs/RuleRecommendation)'s dataset API.</details>

### XAGen: 3D Expressive Human Avatars Generation

**Authors:** Zhongcong XU, Jianfeng Zhang, Jun Hao Liew, Jiashi Feng, Mike Zheng Shou

**<details><summary>Abstract**</summary>

Recent advances in 3D-aware GAN models have enabled the generation of realistic and controllable human body images. However, existing methods focus on the control of major body joints, neglecting the manipulation of expressive attributes, such as facial expressions, jaw poses, hand poses, and so on. In this work, we present XAGen, the first 3D generative model for human avatars capable of expressive control over body, face, and hands. To enhance the fidelity of small-scale regions like face and hands, we devise a multi-scale and multi-part 3D representation that models fine details. Based on this representation, we propose a multi-part rendering technique that disentangles the synthesis of body, face, and hands to ease model training and enhance geometric quality. Furthermore, we design multi-part discriminators that evaluate the quality of the generated avatars with respect to their appearance and fine-grained control capabilities. Experiments show that XAGen surpasses state-of-the-art methods in terms of realism, diversity, and expressive control abilities. Code and data will be made available at https://showlab.github.io/xagen.</details>

### Zero-Shot Anomaly Detection via Batch Normalization

**Authors:** Aodong Li, Chen Qiu, Marius Kloft, Padhraic Smyth, Maja Rudolph, Stephan Mandt

**<details><summary>Abstract**</summary>

Anomaly detection (AD) plays a crucial role in many safety-critical application domains. The challenge of adapting an anomaly detector to drift in the normal data distribution, especially when no training data is available for the "new normal," has led to the development of zero-shot AD techniques. In this paper, we propose a simple yet effective method called Adaptive Centered Representations (ACR) for zero-shot batch-level AD. Our approach trains off-the-shelf deep anomaly detectors (such as deep SVDD) to adapt to a set of inter-related training data distributions in combination with batch normalization, enabling automatic zero-shot generalization for unseen AD tasks. This simple recipe, batch normalization plus meta-training, is a highly effective and versatile tool. Our results demonstrate the first zero-shot AD results for tabular data and outperform existing methods in zero-shot anomaly detection and segmentation on image data from specialized domains.</details>

### Zero-shot Visual Relation Detection via Composite Visual Cues from Large Language Models

**Authors:** Lin Li, Jun Xiao, Guikun Chen, Jian Shao, Yueting Zhuang, Long Chen

**<details><summary>Abstract**</summary>

Pretrained vision-language models, such as CLIP, have demonstrated strong generalization capabilities, making them promising tools in the realm of zero-shot visual recognition. Visual relation detection (VRD) is a typical task that identifies relationship (or interaction) types between object pairs within an image. However, naively utilizing CLIP with prevalent class-based prompts for zero-shot VRD has several weaknesses, e.g., it struggles to distinguish between different fine-grained relation types and it neglects essential spatial information of two objects. To this end, we propose a novel method for zero-shot VRD: RECODE, which solves RElation detection via COmposite DEscription prompts. Specifically, RECODE first decomposes each predicate category into subject, object, and spatial components. Then, it leverages large language models (LLMs) to generate description-based prompts (or visual cues) for each component. Different visual cues enhance the discriminability of similar relation categories from different perspectives, which significantly boosts performance in VRD. To dynamically fuse different cues, we further introduce a chain-of-thought method that prompts LLMs to generate reasonable weights for different visual cues. Extensive experiments on four VRD benchmarks have demonstrated the effectiveness and interpretability of RECODE.</details>

### Zeroth-Order Methods for Nondifferentiable, Nonconvex, and Hierarchical Federated Optimization

**Authors:** Yuyang Qiu, Uday Shanbhag, Farzad Yousefian

**<details><summary>Abstract**</summary>

Federated learning (FL) has emerged as an enabling framework for communication-efficient decentralized training. We study three broadly applicable problem classes in FL: (i) Nondifferentiable nonconvex optimization; (ii) Federated bilevel optimization; (iii) Federated minimax problems. Notably, in an implicit sense, both (ii) and (iii) are instances of (i). However, these hierarchical problems are often complicated by the absence of a closed-form expression for the implicit objective function. Unfortunately, research on these problems has been limited and afflicted by reliance on strong assumptions, including the need for differentiability and L-smoothness of the implicit function. We address this shortcoming by making the following contributions. In (i), by leveraging convolution-based smoothing and Clarke’s subdifferential calculus, we devise a randomized smoothing-enabled zeroth-order FL method and derive communication and iteration complexity guarantees for computing an approximate Clarke stationary point. To contend with (ii) and (iii), we devise a unifying randomized implicit zeroth-order FL framework, equipped with explicit communication and iteration complexities. Importantly, our method utilizes delays during local steps to skip calling the inexact lower-level FL oracle. This results in significant reduction in communication overhead when addressing hierarchical problems. We empirically validate the theory on nonsmooth and hierarchical ML problems.</details>

### Zeroth-Order Methods for Nonsmooth, Nonconvex, and Hierarchical Federated Optimization

**Authors:** Yuyang Qiu, Uday Shanbhag, Farzad Yousefian

**<details><summary>Abstract**</summary>

Federated learning (FL) has emerged as an enabling framework for communication-efficient decentralized training. We study three broadly applicable problem classes in FL: (i) Nondifferentiable nonconvex optimization; (ii) Federated bilevel optimization; (iii) Federated minimax problems. Notably, in an implicit sense, both (ii) and (iii) are instances of (i). However, these hierarchical problems are often complicated by the absence of a closed-form expression for the implicit objective function. Unfortunately, research on these problems has been limited and afflicted by reliance on strong assumptions, including the need for differentiability and L-smoothness of the implicit function. We address this shortcoming by making the following contributions. In (i), by leveraging convolution-based smoothing and Clarke’s subdifferential calculus, we devise a randomized smoothing-enabled zeroth-order FL method and derive communication and iteration complexity guarantees for computing an approximate Clarke stationary point. To contend with (ii) and (iii), we devise a unifying randomized implicit zeroth-order FL framework, equipped with explicit communication and iteration complexities. Importantly, our method utilizes delays during local steps to skip calling the inexact lower-level FL oracle. This results in significant reduction in communication overhead when addressing hierarchical problems. We empirically validate the theory on nonsmooth and hierarchical ML problems.</details>

### [Re] Hierarchical Shrinkage: Improving the Accuracy and Interpretability of Tree-Based Methods

**Authors:** Domen Mohorčič, David Ocepek

**<details><summary>Abstract**</summary>

Scope of ReproducibilityThe paper presents a novel post-hoc regularization technique for tree-based models, called Hierarchical Shrinkage (Agarwal 2022). Our main goal is to confirm the claims that it substantially increases the predictive performance of both decision trees and random forests, that it is faster than other regularization techniques, and that it makes the interpretation of random forests simpler.MethodologyIn our reproduction, we used the Hierarchical Shrinkage, provided by the authors in the Python package imodels. We also used their function for obtaining pre-cleaned data sets. While the algorithm code and clean datasets were provided we re-implemented the experiments as well as added additional experiments to further test the validity of the claims. The results were tested by applying Hierarchical Shrinkage to different tree models and comparing them to the authors' results.ResultsWe managed to reproduce most of the results the authors get. The method works well and most of the claims are supported. The method does increase the predictive performance of tree-based models most of the time, but not always. When compared to other regularization techniques the Hierarchical Shrinkage outperforms them when used on decision trees, but not when used on random forests. Since the method is applied after learning, it is extremely fast. And it does simplify decision boundaries for random forests, making them easier to interpret.What was easyThe use of the official code for Hierarchical Shrinkage was straightforward and used the same function naming convention as other machine learning libraries. The function for acquiring already clean data sets saved a lot of time.What was difficultThe authors also provided the code for their experiments in a separate library, but the code did not run out of the box and we had no success reproducing the results with it. The code was inconsistent with the paper methodology. We had the most problems with hyperparameter tuning. The authors did not specify how they tuned the hyperparameters for the used RF regularizers.Communication with original authorsWe did not contact the authors of the original paper</details>

### [Re] Masked Autoencoders Are Small Scale Vision Learners: A Reproduction Under Resource Constraints

**Authors:** Athanasios Charisoudis, Simon Ekman von Huth, Emil Jansson

**<details><summary>Abstract**</summary>

Scope of Reproducibility — The Masked Autoencoder (MAE) was recently proposed as aframework for efficient self‐supervised pre‐training in Computer Vision [1]. In this pa‐per, we attempt a replication of the MAE under significant computational constraints.Specifically, we target the claim that masking out a large part of the input image yieldsa nontrivial and meaningful self‐supervisory task, which allows training models thatgeneralize well. We also present the Semantic Masked Autoencoder (SMAE), a novel yetsimple extension of MAE which uses perceptual loss to improve encoder embeddings.Methodology — The datasets and backbones we rely on are significantly smaller than thoseused by [1]. Our main experiments are performed on Tiny ImageNet (TIN) [2] and trans‐fer learning is performed on a low‐resolution version of CUB‐200‐2011 [3]. We use aViT‐Lite [4] as backbone. We also compare the MAE to DINO, an alternative frame‐work for self‐supervised learning [5]. The ViT, MAE, as well as perceptual loss wereimplemented from scratch, without consulting the original authors’ code. Our code isavailable at https://github.com/MLReproHub/SMAE. The computational budget for ourreproduction and extension was approximately 150 GPU hours.Results — This paper successfully reproduces the claim that the MAE poses a nontrivialand meaningful self‐supervisory task. We show that models trained with this frame‐work generalize well to new datasets and conclude that the MAE is reproducible withexception for some hyperparameter choices. We also demonstrate that MAE performswell with smaller backbones and datasets. Finally, our results suggest that the SMAEextension improves the downstream classification accuracy of the MAE on CUB (+5 pp)when coupled with an appropriate masking strategy.What was easy — Given prior experience with a deep learning framework, re‐implementingthe paper was relatively straightforward, with sufficient details given in the paper.What was difficult — We faced challenges implementing efficient patch shuffling and tun‐ing hyperparameters. The hyperparameter choices from [1] did not translate well to asmaller dataset and backbone.Communication with original authors — We have not had contact with the original authors.</details>

### [Re] VAE Approximation Error: ELBO and Exponential Families

**Authors:** Volodymyr Kyrylov, Navdeep Singh Bedi, Qianbo Zang

**<details><summary>Abstract**</summary>

Scope of Reproducibility — Exponential family variational autoencoders struggle with reconstruction when encoders output limited information. We reproduce two experiments in which we first train the decoder and encoder separately. Then, we train both modules jointly using ELBO and observe the degradation of reconstruction. We verify how the theoretical insight into the design of the approximate posterior and decoder distributions for a discrete VAE in a semantic hashing application influences the choice of input features to improve overall performance.Methodology — We implement and train a synthetic experiment from scratch on a laptop. We use a mix of authors’ code and publicly available code to reproduce a GAN reinterpreted as a VAE. We consult authors’ code to reproduce semantic hashing experiment and make our own implementation. We train models the USI HPC cluster on machines with GTX 1080 Ti or A100 GPUs and 128 GiB of RAM. We spend under 100 GPU hours for all experiments.Results — We observe expected qualitative behavior predicted by the theory on all experiments. We improve the best semantic hashing model’s test performance by 5 nats by using a modern method for gradient estimation of discrete random variables.What was easy — Following experiment recipes was easy once we worked out the theory.What was difficult — The paper enables verification of exponential family distributions VAE designs of arbitrary complexity, which require probabilistic modeling skills. We contribute elaboration on the implementation details of the synthetic experiment and provide code.Communication with original authors — We are extremely grateful to the authors for discussing the paper, encouraging us to implement experiments on our own, and suggesting directions for improving results over e‐mail.</details>

### rPPG-Toolbox: Deep Remote PPG Toolbox

**Authors:** Xin Liu, Girish Narayanswamy, Akshay Paruchuri, Xiaoyu Zhang, Jiankai Tang, Yuzhe Zhang, Roni Sengupta, Shwetak Patel, Yuntao Wang, Daniel McDuff

**<details><summary>Abstract**</summary>

Camera-based physiological measurement is a fast growing field of computer vision. Remote photoplethysmography (rPPG) utilizes imaging devices (e.g., cameras) to measure the peripheral blood volume pulse (BVP) via photoplethysmography, and enables cardiac measurement via webcams and smartphones. However, the task is non-trivial with important pre-processing, modeling and post-processing steps required to obtain state-of-the-art results. Replication of results and benchmarking of new models is critical for scientific progress; however, as with many other applications of deep learning, reliable codebases are not easy to find or use. We present a comprehensive toolbox, rPPG-Toolbox, unsupervised and supervised rPPG models with support for public benchmark datasets, data augmentation and systematic evaluation: https://github.com/ubicomplab/rPPG-Toolbox.</details>

