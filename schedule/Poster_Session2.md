## Poster Session 2: Tuesday, Dec 12, 15:15 CT### (Almost) Provable Error Bounds Under Distribution Shift via Disagreement Discrepancy

**Authors:** Elan Rosenfeld, Saurabh Garg

**<details><summary>Abstract**</summary>

We derive a new, (almost) guaranteed upper bound on the error of deep neural networks under distribution shift using unlabeled test data. Prior methods are either vacuous in practice or accurate on average but heavily underestimate error for a sizeable fraction of shifts. In particular, the latter only give guarantees based on complex continuous measures such as test calibration, which cannot be identified without labels, and are therefore unreliable. Instead, our bound requires a simple, intuitive condition which is well justified by prior empirical works and holds in practice effectively 100\% of the time. The bound is inspired by $\mathcal{H}\Delta\mathcal{H}$-divergence but is easier to evaluate and substantially tighter, consistently providing non-vacuous test error upper bounds. Estimating the bound requires optimizing one multiclass classifier to disagree with another, for which some prior works have used sub-optimal proxy losses; we devise a "disagreement loss" which is theoretically justified and performs better in practice. We expect this loss can serve as a drop-in replacement for future methods which require maximizing multiclass disagreement. Across a wide range of natural and synthetic distribution shift benchmarks, our method gives valid error bounds while achieving average accuracy comparable to—though not better than—competitive estimation baselines.</details>

### 3D-Aware Visual Question Answering about Parts, Poses and Occlusions

**Authors:** Xingrui Wang, Wufei Ma, Zhuowan Li, Adam Kortylewski, Alan Yuille

**<details><summary>Abstract**</summary>

Despite rapid progress in Visual question answering (\textit{VQA}), existing datasets and models mainly focus on testing reasoning in 2D.  However, it is important that VQA models also understand the 3D structure of visual scenes, for example to support tasks like navigation or manipulation.  This includes an understanding of the 3D object pose, their parts and occlusions.   In this work, we introduce the task of 3D-aware VQA, which focuses on challenging questions that require a compositional reasoning over the 3D structure of visual scenes.   We address 3D-aware VQA from both the dataset and the model perspective.   First, we introduce Super-CLEVR-3D, a compositional reasoning dataset that contains questions about object parts, their 3D poses, and occlusions.   Second, we propose PO3D-VQA, a 3D-aware VQA model that marries two powerful ideas: probabilistic neural symbolic program execution for reasoning and deep neural networks with 3D generative representations of objects for robust visual recognition.  Our experimental results show our model PO3D-VQA outperforms existing methods significantly, but we still observe a significant performance gap compared to 2D VQA benchmarks, indicating that 3D-aware VQA remains an important open research area.</details>

### A Combinatorial Algorithm for Approximating the Optimal Transport in the Parallel and MPC Settings

**Authors:** Nathaniel Lahn, Sharath Raghvendra, Kaiyi Zhang

**<details><summary>Abstract**</summary>

Optimal Transport is a popular distance metric for measuring similarity between distributions. Exact and approximate combinatorial algorithms for computing the optimal transport distance are hard to parallelize. This has motivated the development of numerical solvers (e.g. Sinkhorn method) that can exploit GPU parallelism and produce approximate solutions. We introduce the first parallel combinatorial algorithm to find an additive $\varepsilon$-approximation of the OT distance. The parallel complexity of our algorithm is $O(\log(n)/ \varepsilon^2)$ where $n$ is the total support size for the input distributions. In Massive Parallel Computation (MPC) frameworks such as Hadoop and MapReduce, our algorithm computes an $\varepsilon$-approximate transport plan in $O(\log (\log (n/\varepsilon))/\varepsilon^2)$ rounds with $O(n/\varepsilon)$ space per machine; all prior algorithms in the MPC framework take $\Omega(\log n)$ rounds. We also provide a GPU-friendly matrix-based interpretation of our algorithm where each step of the algorithm is row or column manipulation of the matrix. Experiments suggest that our combinatorial algorithm is faster than the state-of-the-art approximate solvers in the GPU, especially for higher values of $n$.</details>

### A Competitive Algorithm for Agnostic Active Learning

**Authors:** Yihan Zhou, Eric Price

**<details><summary>Abstract**</summary>

For some hypothesis classes and input distributions, \emph{active}  agnostic learning needs exponentially fewer samples than passive  learning; for other classes and distributions, it offers little to  no improvement.  The most popular algorithms for agnostic active  learning express their performance in terms of a parameter called  the disagreement coefficient, but it is known that these algorithms  are inefficient on some inputs.  We take a different approach to agnostic active learning, getting an  algorithm that is \emph{competitive} with the optimal algorithm for  any binary hypothesis class $H$ and distribution $\mathcal{D}_X$ over $X$.  In particular, if any algorithm can use $m^*$ queries to get  $O(\eta)$ error, then our algorithm uses $O(m^* \log H)$ queries to  get $O(\eta)$ error.  Our algorithm lies in the vein of the  splitting-based approach of Dasgupta [2004], which gets a similar  result for the realizable ($\eta = 0$) setting.  We also show that it is NP-hard to do better than our algorithm's  $O(\log H)$ overhead in general.</details>

### A Computationally Efficient Sparsified Online Newton Method

**Authors:** Fnu Devvrit, Sai Surya Duvvuri, Rohan Anil, Vineet Gupta, Cho-Jui Hsieh, Inderjit Dhillon

**<details><summary>Abstract**</summary>

Second-order methods hold significant promise for enhancing the convergence of deep neural network training; however, their large memory and computational demands have limited their practicality. Thus there is a need for scalable second-order methods that can efficiently train large models. In this paper, we introduce the Sparsified Online Newton~(SONew) method, a memory-efficient second-order algorithm that yields a sparsified yet effective preconditioner. The algorithm emerges from a novel use of the LogDet matrix divergence measure; we combine it with sparsity constraints to minimize regret in the online convex optimization framework. Empirically, we test our method on large scale benchmarks of up to 1B parameters. We achieve up to $30\%$ faster convergence, $3.4\%$ relative improvement in validation performance, and $80\%$ relative improvement in training loss, in comparison to memory efficient optimizers including first order methods. Powering the method is a surprising fact -- imposing structured sparsity patterns, like tridiagonal and banded structure, requires little to no overhead, making it as efficient and parallelizable as first-order methods. In wall-clock time, tridiagonal SONew is only about $3\%$ slower per step than first-order methods but gives overall gains due to much faster convergence. In contrast, one of the state-of-the-art (SOTA) memory-intensive second-order methods, Shampoo, is unable to scale to large benchmarks. Additionally, while Shampoo necessitates significant engineering efforts to scale to large benchmarks, SONew offers a more straightforward implementation, increasing its practical appeal. SONew code is available at: https://github.com/devvrit/SONew</details>

### [Spotlight] A Cross-Moment Approach for Causal Effect Estimation

**Authors:** Yaroslav Kivva, Saber Salehkaleybar, Negar Kiyavash

**<details><summary>Abstract**</summary>

We consider the problem of estimating the causal effect of a treatment on an outcome in  linear structural causal models (SCM) with latent confounders when we have access to a single proxy variable.Several methods (such as difference-in-difference (DiD) estimator or negative outcome control) have been proposed in this setting in the literature. However, these approaches require either restrictive assumptions on the data generating model or having access to at least two proxy variables.We propose a method to estimate the causal effect using cross moments between the treatment, the outcome, and the proxy variable. In particular, we show that the causal effect can be identified with simple arithmetic operations on the cross moments if the latent confounder in linear SCM is non-Gaussian.In this setting, DiD estimator provides an unbiased estimate only in the special case where the latent confounder has exactly the same direct causal effects on the outcomes in the pre-treatment and post-treatment phases. This translates to the common trend assumption in DiD, which we effectively relax.Additionally, we provide an impossibility result that shows the causal effect cannot be identified if the observational distribution over the treatment, the outcome, and the proxy is jointly Gaussian. Our experiments on both synthetic and real-world datasets showcase the effectivenessof the proposed approach in estimating the causal effect.</details>

### A Data-Free Approach to Mitigate Catastrophic Forgetting in Federated Class Incremental Learning for Vision Tasks

**Authors:** Sara Babakniya, Zalan Fabian, Chaoyang He, Mahdi Soltanolkotabi, Salman Avestimehr

**<details><summary>Abstract**</summary>

Deep learning models often suffer from forgetting previously learned information when trained on new data. This problem is exacerbated in federated learning (FL), where the data is distributed and can change independently for each user. Many solutions are proposed to resolve this catastrophic forgetting in a centralized setting. However, they do not apply directly to FL because of its unique complexities, such as privacy concerns and resource limitations. To overcome these challenges, this paper presents a framework for \textbf{federated class incremental learning} that utilizes a generative model to synthesize samples from past distributions. This data can be later exploited alongside the training data to mitigate catastrophic forgetting. To preserve privacy, the generative model is trained on the server using data-free methods at the end of each task without requesting data from clients. Moreover, our solution does not demand the users to store old data or models, which gives them the freedom to join/leave the training at any time. Additionally, we introduce SuperImageNet, a new regrouping of the ImageNet dataset specifically tailored for federated continual learning. We demonstrate significant improvements compared to existing baselines through extensive experiments on multiple datasets.</details>

### [Spotlight] A Deep Instance Generative Framework for MILP Solvers Under Limited Data Availability

**Authors:** Zijie Geng, Xijun Li, Jie Wang, Xiao Li, Yongdong Zhang, Feng Wu

**<details><summary>Abstract**</summary>

In the past few years, there has been an explosive surge in the use of machine learning (ML) techniques to address combinatorial optimization (CO) problems, especially mixed-integer linear programs (MILPs). Despite the achievements, the limited availability of real-world instances often leads to sub-optimal decisions and biased solver assessments, which motivates a suite of synthetic MILP instance generation techniques. However, existing methods either rely heavily on expert-designed formulations or struggle to capture the rich features of real-world instances. To tackle this problem, we propose G2MILP, *the first* deep generative framework for MILP instances. Specifically, G2MILP represents MILP instances as bipartite graphs, and applies a masked variational autoencoder to iteratively corrupt and replace parts of the original graphs to generate new ones. The appealing feature of G2MILP is that it can learn to generate novel and realistic MILP instances without prior expert-designed formulations, while preserving the structures and computational hardness of real-world datasets, simultaneously. Thus the generated instances can facilitate downstream tasks for enhancing MILP solvers under limited data availability. We design a suite of benchmarks to evaluate the quality of the generated MILP instances. Experiments demonstrate that our method can produce instances that closely resemble real-world datasets in terms of both structures and computational hardness. The deliverables are released at [https://miralab-ustc.github.io/L2O-G2MILP](https://miralab-ustc.github.io/L2O-G2MILP).</details>

### A Diffusion-Model of Joint Interactive Navigation

**Authors:** Matthew Niedoba, Jonathan Lavington, Yunpeng Liu, Vasileios Lioutas, Justice Sefas, Xiaoxuan Liang, Dylan Green, Setareh Dabiri, Berend Zwartsenberg, Adam Scibior, Frank Wood

**<details><summary>Abstract**</summary>

Simulation of autonomous vehicle systems requires that simulated traffic participants exhibit diverse and realistic behaviors. The use of prerecorded real-world traffic scenarios in simulation ensures realism but the rarity of safety critical events makes large scale collection of driving scenarios expensive. In this paper, we present DJINN -- a diffusion based method of generating traffic scenarios. Our approach jointly diffuses the trajectories of all agents, conditioned on a flexible set of state observations from the past, present, or future. On popular trajectory forecasting datasets, we report state of the art performance on joint trajectory metrics. In addition, we demonstrate how DJINN flexibly enables direct test-time sampling from a variety of valuable conditional distributions including goal-based sampling, behavior-class sampling, and scenario editing.</details>

### A Finite-Particle Convergence Rate for Stein Variational Gradient Descent

**Authors:** Jiaxin Shi, Lester Mackey

**<details><summary>Abstract**</summary>

We provide the first finite-particle convergence rate for Stein variational gradient descent (SVGD), a popular algorithm for approximating a probability distribution with a collection of particles. Specifically, whenever the target distribution is sub-Gaussian with a Lipschitz score, SVGD with $n$ particles and an appropriate step size sequence drives the kernel Stein discrepancy to zero at an order ${1/}{\sqrt{\log\log n}}$ rate. We suspect that the dependence on $n$ can be improved, and we hope that our explicit, non-asymptotic proof strategy will serve as a template for future refinements.</details>

### A General Framework for Equivariant Neural Networks on Reductive Lie Groups

**Authors:** Ilyes Batatia, Mario Geiger, Jose Munoz, Tess Smidt, Lior Silberman, Christoph Ortner

**<details><summary>Abstract**</summary>

Reductive Lie Groups, such as the orthogonal groups, the Lorentz group, or the unitary groups, play essential roles across scientific fields as diverse as high energy physics, quantum mechanics, quantum chromodynamics, molecular dynamics, computer vision, and imaging. In this paper, we present a general Equivariant Neural Network architecture capable of respecting the symmetries of the finite-dimensional representations of any reductive Lie Group. Our approach generalizes the successful ACE and MACE architectures for atomistic point clouds to any data equivariant to a reductive Lie group action. We also introduce the lie-nn software library, which provides all the necessary tools to develop and implement such general G-equivariant neural networks. It implements routines for the reduction of generic tensor products of representations into irreducible representations, making it easy to apply our architecture to a wide range of problems and groups. The generality and performance of our approach are demonstrated by applying it to the tasks of top quark decay tagging (Lorentz group) and shape recognition (orthogonal group).</details>

### A General Theory of Correct, Incorrect, and Extrinsic Equivariance

**Authors:** Dian Wang, Xupeng Zhu, Jung Yeon Park, Mingxi Jia, Guanang Su, Robert Platt, Robin Walters

**<details><summary>Abstract**</summary>

Although equivariant machine learning has proven effective at many tasks, success depends heavily on the assumption that the ground truth function is symmetric over the entire domain matching the symmetry in an equivariant neural network. A missing piece in the equivariant learning literature is the analysis of equivariant networks when symmetry exists only partially in the domain. In this work, we present a general theory for such a situation. We propose pointwise definitions of correct, incorrect, and extrinsic equivariance, which allow us to quantify continuously the degree of each type of equivariance a function displays. We then study the impact of various degrees of incorrect or extrinsic symmetry on model error. We prove error lower bounds for invariant or equivariant networks in classification or regression settings with partially incorrect symmetry. We also analyze the potentially harmful effects of extrinsic equivariance. Experiments validate these results in three different environments.</details>

### [Oral] A Measure-Theoretic Axiomatisation of Causality

**Authors:** Junhyung Park, Simon Buchholz, Bernhard Schölkopf, Krikamol Muandet

**Oral Presentation:** Tu, Dec 12, 13:55 -- Oral 2C

**<details><summary>Abstract**</summary>

Causality is a central concept in a wide range of research areas, yet there is still no universally agreed axiomatisation of causality. We view causality both as an extension of probability theory and as a study of what happens when one intervenes on a system, and argue in favour of taking Kolmogorov's measure-theoretic axiomatisation of probability as the starting point towards an axiomatisation of causality. To that end, we propose the notion of a causal space, consisting of a probability space along with a collection of transition probability kernels, called causal kernels, that encode the causal information of the space. Our proposed framework is not only rigorously grounded in measure theory, but it also sheds light on long-standing limitations of existing frameworks including, for example, cycles, latent variables and stochastic processes.</details>

### A Reduction-based Framework for Sequential Decision Making with Delayed Feedback

**Authors:** Yunchang Yang, Han Zhong, Tianhao Wu, Bin Liu, Liwei Wang, Simon Du

**<details><summary>Abstract**</summary>

We study stochastic delayed feedback in general single-agent and multi-agent sequential decision making, which includes bandits, single-agent Markov decision processes (MDPs), and Markov games (MGs). We propose a novel reduction-based framework, which turns any multi-batched algorithm for sequential decision making with instantaneous feedback into a sample-efficient algorithm that can handle stochastic delays in sequential decision making. By plugging different multi-batched algorithms into our framework, we provide several examples demonstrating that our framework not only matches or improves existing results for bandits, tabular MDPs, and tabular MGs, but also provides the first line of studies on delays in sequential decision making with function approximation. In summary, we provide a complete set of sharp results for single-agent and multi-agent sequential decision making with delayed feedback.</details>

### A Riemannian Exponential Augmented Lagrangian Method for Computing the Projection Robust Wasserstein Distance

**Authors:** Bo Jiang, Ya-Feng Liu

**<details><summary>Abstract**</summary>

Projection robust Wasserstein (PRW) distance is recently proposed to efficiently mitigate the curse of dimensionality in the classical Wasserstein distance.  In this paper, by equivalently reformulating the computation of the PRW distance as an optimization problem over the Cartesian product of the Stiefel manifold and the Euclidean space with additional nonlinear inequality constraints, we propose a Riemannian exponential augmented Lagrangian method (REALM) for solving this problem. Compared with the existing Riemannian exponential penalty-based approaches, REALM can potentially avoid too small penalty parameters and exhibit more stable numerical performance. To solve the subproblems in REALM efficiently, we design an inexact Riemannian Barzilai-Borwein method with Sinkhorn iteration  (iRBBS), which selects the stepsizes adaptively rather than tuning the stepsizes in efforts as done in the existing methods. We show that iRBBS can return an $\epsilon$-stationary point of the original PRW distance problem within  $\mathcal{O}(\epsilon^{-3})$ iterations, which matches the best known iteration complexity result. Extensive numerical results demonstrate that our proposed methods outperform the state-of-the-art solvers for computing the PRW distance.</details>

### A Simple Solution for Offline Imitation from Observations and Examples with Possibly Incomplete Trajectories

**Authors:** Kai Yan, Alex Schwing, Yu-Xiong Wang

**<details><summary>Abstract**</summary>

Offline imitation from observations aims to solve MDPs where only task-specific expert states and task-agnostic non-expert state-action pairs are available. Offline imitation is useful in real-world scenarios where arbitrary interactions are costly and expert actions are unavailable. The state-of-the-art ‘DIstribution Correction Estimation’ (DICE) methods minimize divergence of state occupancy between expert and learner policies and retrieve a policy with weighted behavior cloning; however, their results are unstable when learning from incomplete trajectories, due to a non-robust optimization in the dual domain. To address the issue, in this paper, we propose Trajectory-Aware Imitation Learning from Observations (TAILO). TAILO uses a discounted sum along the future trajectory as the weight for weighted behavior cloning. The terms for the sum are scaled by the output of a discriminator, which aims to identify expert states. Despite simplicity, TAILO works well if there exist trajectories or segments of expert behavior in the task-agnostic data, a common assumption in prior work. In experiments across multiple testbeds, we find TAILO to be more robust and effective, particularly with incomplete trajectories.</details>

### [Spotlight] A Spectral Theory of Neural Prediction and Alignment

**Authors:** Abdulkadir Canatar, Jenelle Feather, Albert Wakhloo, SueYeon Chung

**<details><summary>Abstract**</summary>

The representations of neural networks are often compared to those of biological systems by performing regression between the neural network responses and those measured from biological systems. Many different state-of-the-art deep neural networks yield similar neural predictions, but it remains unclear how to differentiate among models that perform equally well at predicting neural responses. To gain insight into this, we use a recent theoretical framework that relates the generalization error from regression to the spectral properties of the model and the target. We apply this theory to the case of regression between model activations and neural responses and decompose the neural prediction error in terms of the model eigenspectra, alignment of model eigenvectors and neural responses, and the training set size. Using this decomposition, we introduce geometrical measures to interpret the neural prediction error. We test a large number of deep neural networks that predict visual cortical activity and show that there are multiple types of geometries that result in low neural prediction error as measured via regression. The work demonstrates that carefully decomposing representational metrics can provide interpretability of how models are capturing neural activity and points the way towards improved models of neural activity.</details>

### A State Representation for Diminishing Rewards

**Authors:** Ted Moskovitz, Samo Hromadka, Ahmed Touati, Diana Borsa, Maneesh Sahani

**<details><summary>Abstract**</summary>

A common setting in multitask reinforcement learning (RL) demands that an agent rapidly adapt to various stationary reward functions randomly sampled from a fixed distribution. In such situations, the successor representation (SR) is a popular framework which supports rapid policy evaluation by decoupling a policy's expected discounted, cumulative state occupancies from a specific reward function. However, in the natural world, sequential tasks are rarely independent, and instead reflect shifting priorities based on the availability and subjective perception of rewarding stimuli. Reflecting this disjunction, in this paper we study the phenomenon of diminishing marginal utility and introduce a novel state representation, the $\lambda$ representation ($\lambda$R) which, surprisingly, is required for policy evaluation in this setting and which generalizes the SR as well as several other state representations from the literature. We establish the $\lambda$R's formal properties and examine its normative advantages in the context of machine learning, as well as its usefulness for studying natural behaviors, particularly foraging.</details>

### A Sublinear-Time Spectral Clustering Oracle with Improved Preprocessing Time

**Authors:** Ranran Shen, Pan Peng

**<details><summary>Abstract**</summary>

We address the problem of designing a sublinear-time spectral clustering oracle for graphs that exhibit strong clusterability. Such graphs contain $k$ latent clusters, each characterized by a large inner conductance (at least $\varphi$) and a small outer conductance (at most $\varepsilon$). Our aim is to preprocess the graph to enable clustering membership queries, with the key requirement that both preprocessing and query answering should be performed in sublinear time, and the resulting partition should be consistent with a $k$-partition that is close to the ground-truth clustering. Previous oracles have relied on either a $\textrm{poly}(k)\log n$ gap between inner and outer conductances or exponential (in $k/\varepsilon$) preprocessing time. Our algorithm relaxes these assumptions, albeit at the cost of a slightly higher misclassification ratio. We also show that our clustering oracle is robust against a few random edge deletions. To validate our theoretical bounds, we conducted experiments on synthetic networks.</details>

### A Theoretical Analysis of Optimistic Proximal Policy Optimization in Linear Markov Decision Processes

**Authors:** Han Zhong, Tong Zhang

**<details><summary>Abstract**</summary>

The proximal policy optimization (PPO) algorithm stands as one of the most prosperous methods in the field of reinforcement learning (RL). Despite its success, the theoretical understanding of PPO remains deficient. Specifically, it is unclear whether PPO or its optimistic variants can effectively solve linear Markov decision processes (MDPs), which are arguably the simplest models in RL with function approximation.     To bridge this gap, we propose an optimistic variant of PPO for episodic adversarial linear MDPs with full-information feedback, and establish a $\tilde{\mathcal{O}}(d^{3/4}H^2K^{3/4})$ regret for it. Here $d$ is the ambient dimension of linear MDPs, $H$ is the length of each episode, and $K$ is the number of episodes. Compared with existing policy-based algorithms, we achieve the state-of-the-art regret bound in both stochastic linear MDPs and adversarial linear MDPs with full information. Additionally, our algorithm design features a novel multi-batched updating mechanism and the theoretical analysis utilizes a new covering number argument of value and policy classes, which might be of independent interest.</details>

### A Theory of Multimodal Learning

**Authors:** Zhou Lu

**<details><summary>Abstract**</summary>

Human perception of the empirical world involves recognizing the diverse appearances, or 'modalities', of underlying objects. Despite the longstanding consideration of this perspective in philosophy and cognitive science, the study of multimodality remains relatively under-explored within the field of machine learning. Nevertheless, current studies of multimodal machine learning are limited to empirical practices, lacking theoretical foundations beyond heuristic arguments. An intriguing finding from the practice of multimodal learning is that a model trained on multiple modalities can outperform a finely-tuned unimodal model, even on unimodal tasks. This paper provides a theoretical framework that explains this phenomenon, by studying generalization properties of multimodal learning algorithms. We demonstrate that multimodal learning allows for a superior generalization bound compared to unimodal learning, up to a factor of $O(\sqrt{n})$, where $n$ represents the sample size. Such advantage occurs when both connection and heterogeneity exist between the modalities.</details>

### A Theory of Transfer-Based Black-Box Attacks: Explanation and Implications

**Authors:** Yanbo Chen, Weiwei Liu

**<details><summary>Abstract**</summary>

Transfer-based attacks are a practical method of black-box adversarial attacks, in which the attacker aims to craft adversarial examples from a source (surrogate) model that is transferable to the target model. A wide range of empirical works has tried to explain the transferability of adversarial examples from different angles. However, these works only provide ad hoc explanations without quantitative analyses. The theory behind transfer-based attacks remains a mystery.This paper studies transfer-based attacks under a unified theoretical framework. We propose an explanatory model, called the manifold attack model, that formalizes popular beliefs and explains the existing empirical results. Our model explains why adversarial examples are transferable even when the source model is inaccurate. Moreover, our model implies that the existence of transferable adversarial examples depends on the “curvature” of the data manifold, which quantitatively explains why the success rates of transfer-based attacks are hard to improve. We also discuss the expressive power and the possible extensions of our model in general applications.</details>

### A Unified Detection Framework for Inference-Stage Backdoor Defenses

**Authors:** Xun Xian, Ganghua Wang, Jayanth Srinivasa, Ashish Kundu, Xuan Bi, Mingyi Hong, Jie Ding

**<details><summary>Abstract**</summary>

Backdoor attacks involve inserting poisoned samples during training, resulting in a model containing a hidden backdoor that can trigger specific behaviors without impacting performance on normal samples. These attacks are challenging to detect, as the backdoored model appears normal until activated by the backdoor trigger, rendering them particularly stealthy. In this study, we devise a unified inference-stage detection framework to defend against backdoor attacks. We first rigorously formulate the inference-stage backdoor detection problem, encompassing various existing methods, and discuss several challenges and limitations. We then propose a framework with provable guarantees on the false positive rate or the probability of misclassifying a clean sample.  Further, we derive the most powerful detection rule to maximize the detection power, namely the rate of accurately identifying a backdoor sample, given a false positive rate under classical learning scenarios. Based on the theoretically optimal detection rule, we suggest a practical and effective approach for real-world applications based on the latent representations of backdoored deep nets. We extensively evaluate our method on 14 different backdoor attacks using Computer Vision (CV) and Natural Language Processing (NLP) benchmark datasets. The experimental findings align with our theoretical results. We significantly surpass the state-of-the-art methods, e.g., up to 300\% improvement on the detection power as evaluated by AUCROC, over the state-of-the-art defense against advanced adaptive backdoor attacks.</details>

### A Unified Framework for U-Net Design and Analysis

**Authors:** Christopher Williams, Fabian Falck, George Deligiannidis, Chris C Holmes, Arnaud Doucet, Saifuddin Syed

**<details><summary>Abstract**</summary>

U-Nets are a go-to neural architecture across numerous tasks for continuous signals on a square such as images and Partial Differential Equations (PDE), however their design and architecture is understudied. In this paper, we provide a framework for designing and analysing general U-Net architectures. We present theoretical results which characterise the role of the encoder and decoder in a U-Net, their high-resolution scaling limits and their conjugacy to ResNets via preconditioning. We propose Multi-ResNets, U-Nets with a simplified, wavelet-based encoder without learnable parameters. Further, we show how to design novel U-Net architectures which encode function constraints, natural bases, or the geometry of the data. In diffusion models, our framework enables us to identify that high-frequency information is dominated by noise exponentially faster, and show how U-Nets with average pooling exploit this. In our experiments, we demonstrate how Multi-ResNets achieve competitive and often superior performance compared to classical U-Nets in image segmentation, PDE surrogate modelling, and generative modelling with diffusion models. Our U-Net framework paves the way to study the theoretical properties of U-Nets and design natural, scalable neural architectures for a multitude of problems beyond the square.</details>

### A Unified Solution for Privacy and Communication Efficiency in Vertical Federated Learning

**Authors:** Ganyu Wang, Bin Gu, Qingsong Zhang, Xiang Li, Boyu Wang, Charles Ling

**<details><summary>Abstract**</summary>

Vertical Federated Learning (VFL) is a collaborative machine learning paradigm that enables multiple participants to jointly train a model on their private data without sharing it.To make VFL practical, privacy security and communication efficiency should both be satisfied. Recent research has shown that Zero-Order Optimization (ZOO) in VFL can effectively conceal the internal information of the model without adding costly privacy protective add-ons, making it a promising approach for privacy and efficiency.However, there are still two key problems that have yet to be resolved. First, the convergence rate of ZOO-based VFL is significantly slower compared to gradient-based VFL, resulting in low efficiency in model training and more communication round, which hinders its application on large neural networks. Second, although ZOO-based VFL has demonstrated resistance to state-of-the-art (SOTA) attacks, its privacy guarantee lacks a theoretical explanation.To address these challenges, we propose a novel cascaded hybrid optimization approach that employs a zeroth-order (ZO) gradient on the most critical output layer of the clients, with other parts utilizing the first-order (FO) gradient. This approach preserves the privacy protection of ZOO while significantly enhancing convergence.Moreover, we theoretically prove that applying ZOO to the VFL is equivalent to adding Gaussian Mechanism to the gradient information, which offers an implicit differential privacy guarantee. Experimental results demonstrate that our proposed framework achieves similar utility as the Gaussian mechanism under the same privacy budget, while also having significantly lower communication costs compared with SOTA communication-efficient VFL frameworks.</details>

### A normative theory of social conflict

**Authors:** Sergey Shuvaev, Evgeny Amelchenko, Dmitry Smagin, Natalia Kudryavtseva, Grigori Enikolopov, Alex Koulakov

**<details><summary>Abstract**</summary>

Social conflict is a survival mechanism yielding both normal and pathological behaviors. To understand its underlying principles, we collected behavioral and whole-brain neural data from mice advancing through stages of social conflict. We modeled the animals’ interactions as a normal-form game using Bayesian inference to account for the partial observability of animals’ strengths. We find that our behavioral and neural data are consistent with the first-level Theory of Mind (1-ToM) model where mice form “primary” beliefs about the strengths of all mice involved and “secondary” beliefs that estimate the beliefs of their opponents. Our model identifies the brain regions that carry the information about these beliefs and offers a framework for studies of social behaviors in partially observable settings.</details>

### AGD: an Auto-switchable Optimizer using Stepwise Gradient Difference for Preconditioning Matrix

**Authors:** Yun Yue, Zhiling Ye, Jiadi Jiang, Yongchao Liu, Ke Zhang

**<details><summary>Abstract**</summary>

Adaptive optimizers, such as Adam, have achieved remarkable success in deep learning. A key component of these optimizers is the so-called preconditioning matrix, providing enhanced gradient information and regulating the step size of each gradient direction. In this paper, we propose a novel approach to designing the preconditioning matrix by utilizing the gradient difference between two successive steps as the diagonal elements. These diagonal elements are closely related to the Hessian and can be perceived as an approximation of the inner product between the Hessian row vectors and difference of the adjacent parameter vectors. Additionally, we introduce an auto-switching function that enables the preconditioning matrix to switch dynamically between Stochastic Gradient Descent (SGD) and the adaptive optimizer. Based on these two techniques, we develop a new optimizer named AGD that enhances the generalization performance. We evaluate AGD on public datasets of Natural Language Processing (NLP), Computer Vision (CV), and Recommendation Systems (RecSys). Our experimental results demonstrate that AGD outperforms the state-of-the-art (SOTA) optimizers, achieving highly competitive or significantly better predictive performance. Furthermore, we analyze how AGD is able to switch automatically between SGD and the adaptive optimizer and its actual effects on various scenarios. The code is available at https://github.com/intelligent-machine-learning/dlrover/tree/master/atorch/atorch/optimizers.</details>

### AI for Interpretable Chemistry: Predicting Radical Mechanistic Pathways via Contrastive Learning

**Authors:** Mohammadamin Tavakoli, Pierre Baldi, Ann Marie Carlton, Yin Ting Chiu, Alexander Shmakov, David Van Vranken

**<details><summary>Abstract**</summary>

Deep learning-based reaction predictors have undergone significant architectural evolution. However, their reliance on reactions from the US Patent Office results in a lack of interpretable predictions and limited generalizability to other chemistry domains, such as radical and atmospheric chemistry. To address these challenges, we introduce a new reaction predictor system, RMechRP, that leverages contrastive learning in conjunction with mechanistic pathways, the most interpretable representation of chemical reactions. Specifically designed for radical reactions, RMechRP provides different levels of interpretation of chemical reactions. We develop and train multiple deep-learning models using RMechDB, a public database of radical reactions, to establish the first benchmark for predicting radical reactions. Our results demonstrate the effectiveness of RMechRP in providing accurate and interpretable predictions of radical reactions, and its potential for various applications in atmospheric chemistry.</details>

### AMAG: Additive, Multiplicative and Adaptive Graph Neural Network For Forecasting Neuron Activity

**Authors:** Jingyuan Li, Leo Scholl, Trung Le, Pavithra Rajeswaran, Amy Orsborn, Eli Shlizerman

**<details><summary>Abstract**</summary>

Latent Variable Models (LVMs) propose to model the dynamics of neural populations by capturing low-dimensional structures that represent features involved in neural activity. Recent LVMs are based on deep learning methodology where a deep neural network is trained to reconstruct the same neural activity given as input and as a result to build the latent representation. Without taking past or future activity into account such a task is non-causal. In contrast, the task of forecasting neural activity based on given input extends the reconstruction task. LVMs that are trained on such a task could potentially capture temporal causality constraints within its latent representation. Forecasting has received less attention than reconstruction due to recording challenges such as limited neural measurements and trials. In this work, we address modeling neural population dynamics via the forecasting task and improve forecasting performance by including a prior, which consists of pairwise neural unit interaction as a multivariate dynamic system. Our proposed model---Additive, Multiplicative, and Adaptive Graph Neural Network (AMAG)---leverages additive and multiplicative message-passing operations analogous to the interactions in neuronal systems and adaptively learns the interaction among neural units to forecast their future activity. We demonstrate the advantage of AMAG compared to non-GNN based methods on synthetic data and multiple modalities of neural recordings (field potentials from penetrating electrodes or surface-level micro-electrocorticography) from four rhesus macaques. Our results show the ability of AMAG to recover ground truth spatial interactions and yield estimation for future dynamics of the neural population.</details>

### AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation

**Authors:** Tong Wu, Zhihao Fan, Xiao Liu, Hai-Tao Zheng, Yeyun Gong, yelong shen, Jian Jiao, Juntao Li, zhongyu wei, Jian Guo, Nan Duan, Weizhu Chen

**<details><summary>Abstract**</summary>

Diffusion models have gained significant attention in the realm of image generation due to their exceptional performance. Their success has been recently expanded to text generation via generating all tokens within a sequence concurrently. However, natural language exhibits a far more pronounced sequential dependency in comparison to images, and the majority of existing language models are trained with a left-to-right auto-regressive approach.To account for the inherent sequential characteristic of natural language, we introduce Auto-Regressive Diffusion (AR-Diffusion). AR-Diffusion ensures that the generation of tokens on the right depends on the generated ones on the left, a mechanism achieved through employing a dynamic number of denoising steps that vary based on token position. This results in tokens on the left undergoing fewer denoising steps than those on the right, thereby enabling them to generate earlier and subsequently influence the generation of tokens on the right.In a series of experiments on various text generation tasks, including text summarization, machine translation, and common sense generation, AR-Diffusion clearly demonstrated its superiority over existing diffusion language models and that it can be $100	imes\sim600	imes$ faster when achieving comparable results. Our code is available at https://github.com/microsoft/ProphetNet/tree/master/AR-diffusion.</details>

### ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition

**Authors:** Aashaka Desai, Lauren Berger, Fyodor Minakov, Nessa Milano, Chinmay Singh, Kriston Pumphrey, Richard Ladner, Hal Daumé III, Alex X Lu, Naomi Caselli, Danielle Bragg

**<details><summary>Abstract**</summary>

Sign languages are used as a primary language by approximately 70 million D/deaf people world-wide. However, most communication technologies operate in spoken and written languages, creating inequities in access. To help tackle this problem, we release ASL Citizen, the first crowdsourced Isolated Sign Language Recognition (ISLR) dataset, collected with consent and containing 83,399 videos for 2,731 distinct signs filmed by 52 signers in a variety of environments. We propose that this dataset be used for sign language dictionary retrieval for American Sign Language (ASL), where a user demonstrates a sign to their webcam to retrieve matching signs from a dictionary. We show that training supervised machine learning classifiers with our dataset advances the state-of-the-art on metrics relevant for dictionary retrieval, achieving 63\% accuracy and a recall-at-10 of 91\%, evaluated entirely on videos of users who are not present in the training or validation sets.</details>

### AVIS: Autonomous Visual Information Seeking with Large Language Model Agent

**Authors:** Ziniu Hu, Ahmet Iscen, Chen Sun, Kai-Wei Chang, Yizhou Sun, David Ross, Cordelia Schmid, Alireza Fathi

**<details><summary>Abstract**</summary>

In this paper, we propose an autonomous information seeking visual question answering framework, AVIS. Our method leverages a Large Language Model (LLM) to dynamically strategize the utilization of external tools and to investigate their outputs via tree search, thereby acquiring the indispensable knowledge needed to provide answers to the posed questions. Responding to visual questions that necessitate external knowledge, such as "What event is commemorated by the building depicted in this image?", is a complex task. This task presents a combinatorial search space that demands a sequence of actions, including invoking APIs, analyzing their responses, and making informed decisions. We conduct a user study to collect a variety of instances of human decision-making when faced with this task. This data is then used to design a system comprised of three components: an LLM-powered planner that dynamically determines which tool to use next, an LLM-powered reasoner that analyzes and extracts key information from the tool outputs, and a working memory component that retains the acquired information throughout the process. The collected user behavior serves as a guide for our system in two key ways. First, we create a transition graph by analyzing the sequence of decisions made by users. This graph delineates distinct states and confines the set of actions available at each state. Second, we use examples of user decision-making to provide our LLM-powered planner and reasoner with relevant contextual instances, enhancing their capacity to make informed decisions. We show that AVIS achieves state-of-the-art results on knowledge-based visual question answering benchmarks such as Infoseek and OK-VQA.</details>

### Accelerated On-Device Forward Neural Network Training with Module-Wise Descending Asynchronism

**Authors:** Xiaohan Zhao, Hualin Zhang, Zhouyuan Huo, Bin Gu

**<details><summary>Abstract**</summary>

On-device learning faces memory constraints when optimizing or fine-tuning on edge devices with limited resources. Current techniques for training deep models on edge devices rely heavily on backpropagation. However, its high memory usage calls for a reassessment of its dominance.In this paper, we propose forward gradient descent (FGD) as a potential solution to overcome the memory capacity limitation in on-device learning. However, FGD's dependencies across layers hinder parallel computation and can lead to inefficient resource utilization.To mitigate this limitation, we propose AsyncFGD, an asynchronous framework that decouples dependencies, utilizes module-wise stale parameters, and maximizes parallel computation. We demonstrate its convergence to critical points through rigorous theoretical analysis.Empirical evaluations conducted on NVIDIA's AGX Orin, a popular embedded device,  show that AsyncFGD reduces memory consumption and enhances hardware efficiency, offering a novel approach to on-device learning.</details>

### Accelerating Molecular Graph Neural Networks via Knowledge Distillation

**Authors:** Filip Ekström Kelvinius, Dimitar Georgiev, Artur Toshev, Johannes Gasteiger

**<details><summary>Abstract**</summary>

Recent advances in graph neural networks (GNNs) have enabled more comprehensive modeling of molecules and molecular systems, thereby enhancing the precision of molecular property prediction and molecular simulations. Nonetheless, as the field has been progressing to bigger and more complex architectures, state-of-the-art GNNs have become largely prohibitive for many large-scale applications. In this paper, we explore the utility of knowledge distillation (KD) for accelerating molecular GNNs. To this end, we devise KD strategies that facilitate the distillation of hidden representations in directional and equivariant GNNs, and evaluate their performance on the regression task of energy and force prediction. We validate our protocols across different teacher-student configurations and datasets, and demonstrate that they can consistently boost the predictive accuracy of student models without any modifications to their architecture. Moreover, we conduct comprehensive optimization of various components of our framework, and investigate the potential of data augmentation to further enhance performance. All in all, we manage to close the gap in predictive accuracy between teacher and student models by as much as 96.7\% and 62.5\% for energy and force prediction respectively, while fully preserving the inference throughput of the more lightweight models.</details>

### Accelerating Monte Carlo Tree Search with Probability Tree State Abstraction

**Authors:** Yangqing Fu, Ming Sun, Buqing Nie, Yue Gao

**<details><summary>Abstract**</summary>

Monte Carlo Tree Search (MCTS) algorithms such as AlphaGo and MuZero have achieved superhuman performance in many challenging tasks. However, the computational complexity of MCTS-based algorithms is influenced by the size of the search space. To address this issue, we propose a novel probability tree state abstraction (PTSA) algorithm to improve the search efficiency of MCTS. A general tree state abstraction with path transitivity is defined. In addition, the probability tree state abstraction is proposed for fewer mistakes during the aggregation step. Furthermore, the theoretical guarantees of the transitivity and aggregation error bound are justified. To evaluate the effectiveness of the PTSA algorithm, we integrate it with state-of-the-art MCTS-based algorithms, such as Sampled MuZero and Gumbel MuZero. Experimental results on different tasks demonstrate that our method can accelerate the training process of state-of-the-art algorithms with 10%-45% search space reduction.</details>

### Accelerating Motion Planning via Optimal Transport

**Authors:** An T. Le, Georgia Chalvatzaki, Armin Biess, Jan Peters

**<details><summary>Abstract**</summary>

Motion planning is still an open problem for many disciplines, e.g., robotics, autonomous driving, due to their need for high computational resources that hinder real-time, efficient decision-making. A class of methods striving to provide smooth solutions is gradient-based trajectory optimization. However, those methods usually suffer from bad local minima, while for many settings, they may be inapplicable due to the absence of easy-to-access gradients of the optimization objectives. In response to these issues, we introduce Motion Planning via Optimal Transport (MPOT)---a \textit{gradient-free} method that optimizes a batch of smooth trajectories over highly nonlinear costs, even for high-dimensional tasks, while imposing smoothness through a Gaussian Process dynamics prior via the planning-as-inference perspective. To facilitate batch trajectory optimization, we introduce an original zero-order and highly-parallelizable update rule----the Sinkhorn Step, which uses the regular polytope family for its search directions. Each regular polytope, centered on trajectory waypoints, serves as a local cost-probing neighborhood, acting as a \textit{trust region} where the Sinkhorn Step ``transports'' local waypoints toward low-cost regions. We theoretically show that Sinkhorn Step guides the optimizing parameters toward local minima regions of non-convex objective functions. We then show the efficiency of MPOT in a range of problems from low-dimensional point-mass navigation to high-dimensional whole-body robot motion planning, evincing its superiority compared to popular motion planners, paving the way for new applications of optimal transport in motion planning.</details>

### Accelerating Reinforcement Learning with Value-Conditional State Entropy Exploration

**Authors:** Dongyoung Kim, Jinwoo Shin, Pieter Abbeel, Younggyo Seo

**<details><summary>Abstract**</summary>

A promising technique for exploration is to maximize the entropy of visited state distribution, i.e., state entropy, by encouraging uniform coverage of visited state space. While it has been effective for an unsupervised setup, it tends to struggle in a supervised setup with a task reward, where an agent prefers to visit high-value states to exploit the task reward. Such a preference can cause an imbalance between the distributions of high-value states and low-value states, which biases exploration towards low-value state regions as a result of the state entropy increasing when the distribution becomes more uniform. This issue is exacerbated when high-value states are narrowly distributed within the state space, making it difficult for the agent to complete the tasks. In this paper, we present a novel exploration technique that maximizes the value-conditional state entropy, which separately estimates the state entropies that are conditioned on the value estimates of each state, then maximizes their average. By only considering the visited states with similar value estimates for computing the intrinsic bonus, our method prevents the distribution of low-value states from affecting exploration around high-value states, and vice versa. We demonstrate that the proposed alternative to the state entropy baseline significantly accelerates various reinforcement learning algorithms across a variety of tasks within MiniGrid, DeepMind Control Suite, and Meta-World benchmarks. Source code is available at https://sites.google.com/view/rl-vcse.</details>

### Accurate Interpolation for Scattered Data through Hierarchical Residual Refinement

**Authors:** Shizhe Ding, Boyang Xia, Dongbo Bu

**<details><summary>Abstract**</summary>

Accurate interpolation algorithms are highly desired in various theoretical and engineering scenarios. Unlike the traditional numerical algorithms that have exact zero-residual constraints on observed points, the neural network-based interpolation methods exhibit non-zero residuals at these points. These residuals, which provide observations of an underlying residual function, can guide predicting interpolation functions, but have not been exploited by the existing approaches. To fill this gap, we propose Hierarchical INTerpolation Network (HINT), which utilizes the residuals on observed points to guide target function estimation in a hierarchical fashion. HINT consists of several sequentially arranged lightweight interpolation blocks. The first interpolation block estimates the main component of the target function, while subsequent blocks predict the residual components using observed points residuals of the preceding blocks. The main component and residual components are accumulated to form the final interpolation results. Furthermore, under the assumption that finer residual prediction requires a more focused attention range on observed points, we utilize hierarchical local constraints in correlation modeling between observed and target points. Extensive experiments demonstrate that HINT outperforms existing interpolation algorithms significantly in terms of interpolation accuracy across a wide variety of datasets, which underscores its potential for practical scenarios.</details>

### Active Bipartite Ranking

**Authors:** James Cheshire, Vincent Laurent, Stephan Clémençon

**<details><summary>Abstract**</summary>

In this paper, we develop an active learning framework for the bipartite ranking problem.Motivated by numerous applications, ranging from supervised anomaly detection to credit-scoring through the design of medical diagnosis support systems, and usually formulated as the problem of optimizing (a scalar summary of) the ROC curve, bipartite ranking has been the subject of much attention in the passive context. Various dedicated algorithms have been recently proposed and studied by the machine-learning community. In contrast, active bipartite ranking rule is poorly documented in the literature. Due to its global nature, a strategy for labeling sequentially data points that are difficult to rank w.r.t. to the others is required. This learning task is much more complex than binary classification, for which many active algorithms have been designed. It is the goal of this article to provide a rigorous formulation of such a selective sampling approach. We propose a dedicated algorithm, referred to as active-rank, which aims to minimise the distance between the ROC curve of the ranking function built and the optimal one, w.r.t. the sup norm. We show that, for a fixed confidence level $\epsilon$ and probability $\delta$,  active-rank is PAC$(\epsilon,\delta)$. In addition, we provide a problem dependent upper bound on the expected sampling time of  active-rank and also demonstrate a problem dependent lower bound on the expected sampling time of any PAC$(\epsilon,\delta)$ algorithm. Beyond the theoretical analysis carried out, numerical results are presented, providing strong empirical evidence of the performance of the algorithm proposed, which compares favorably with more naive approaches.</details>

### Adapting Fairness Interventions to Missing Values

**Authors:** Raymond Feng, Flavio Calmon, Hao Wang

**<details><summary>Abstract**</summary>

Missing values in real-world data pose a significant and unique challenge to algorithmic fairness. Different demographic groups may be unequally affected by missing data, and the standard procedure for handling missing values where first data is imputed, then the imputed data is used for classification—a procedure referred to as "impute-then-classify"—can exacerbate discrimination. In this paper, we analyze how missing values affect algorithmic fairness. We first prove that training a classifier from imputed data can significantly worsen the achievable values of group fairness and average accuracy. This is because imputing data results in the loss of the missing pattern of the data, which often conveys information about the predictive label. We present scalable and adaptive algorithms for fair classification with missing values. These algorithms can be combined with any preexisting fairness-intervention algorithm to handle all possible missing patterns while preserving information encoded within the missing patterns. Numerical experiments with state-of-the-art fairness interventions demonstrate that our adaptive algorithms consistently achieve higher fairness and accuracy than impute-then-classify across different datasets.</details>

### Adapting Neural Link Predictors for Data-Efficient Complex Query Answering

**Authors:** Erik Arakelyan, Pasquale Minervini, Daniel Daza, Michael Cochez, Isabelle Augenstein

**<details><summary>Abstract**</summary>

Answering complex queries on incomplete knowledge graphs is a challenging task where a model needs to answer complex logical queries in the presence of missing knowledge. Prior work in the literature has proposed to address this problem by designing architectures trained end-to-end for the complex query answering task with a reasoning process that is hard to interpret while requiring data and resource-intensive training. Other lines of research have proposed re-using simple neural link predictors to answer complex queries, reducing the amount of training data by orders of magnitude while providing interpretable answers. The neural link predictor used in such approaches is not explicitly optimised for the complex query answering task, implying that its scores are not calibrated to interact together. We propose to address these problems via CQD$^{\mathcal{A}}$, a parameter-efficient score \emph{adaptation} model optimised to re-calibrate neural link prediction scores for the complex query answering task. While the neural link predictor is frozen, the adaptation component -- which only increases the number of model parameters by $0.03\%$ -- is trained on the downstream complex query answering task. Furthermore, the calibration component enables us to support reasoning over queries that include atomic negations, which was previously impossible with link predictors. In our experiments, CQD$^{\mathcal{A}}$ produces significantly more accurate results than current state-of-the-art methods, improving from $34.4$ to $35.1$ Mean Reciprocal Rank values averaged across all datasets and query types while using $\leq 30\%$ of the available training query types. We further show that CQD$^{\mathcal{A}}$ is data-efficient, achieving competitive results with only $1\%$ of the complex training queries and robust in out-of-domain evaluations. Source code and datasets are available at https://github.com/EdinburghNLP/adaptive-cqd.</details>

### Adaptive Contextual Perception: How To Generalize To New Backgrounds and Ambiguous Objects

**Authors:** Zhuofan Ying, Peter Hase, Mohit Bansal

**<details><summary>Abstract**</summary>

Biological vision systems make adaptive use of context to recognize objects in new settings with novel contexts as well as occluded or blurry objects in familiar settings. In this paper, we investigate how vision models adaptively use context for out-of-distribution (OOD) generalization and leverage our analysis results to improve model OOD generalization. First, we formulate two distinct OOD settings where the contexts are either beneficial Object-Disambiguation or irrelevant Background-Invariance, reflecting the diverse contextual challenges faced in biological vision. We then analyze model performance in these two different OOD settings and demonstrate that models that excel in one setting tend to struggle in the other. Notably, prior works on learning causal features improve on one setting but hurt on the other. This underscores the importance of generalizing across both OOD settings, as this ability is crucial for both human cognition and robust AI systems. Next, to better understand the model properties contributing to OOD generalization, we use representational geometry analysis and our own probing methods to examine a population of models, and we discover that those with more factorized representations and appropriate feature weighting are more successful in handling Object-Disambiguation and Background-Invariance tests. We further validate these findings through causal intervention, manipulating representation factorization and feature weighting to demonstrate their causal effect on performance. Motivated by our analysis results, we propose new augmentation methods aimed at enhancing model generalization. The proposed methods outperform strong baselines, yielding improvements in both in-distribution and OOD tests. We conclude that, in order to replicate the generalization abilities of biological vision, computer vision models must have factorized object vs. background representations and appropriately weigh both kinds of features.</details>

### Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective

**Authors:** Zhiding Liu, Mingyue Cheng, Zhi Li, Zhenya Huang, Qi Liu, Yanhu Xie, Enhong Chen

**<details><summary>Abstract**</summary>

Deep learning models have progressively advanced time series forecasting due to their powerful capacity in capturing sequence dependence. Nevertheless, it is still challenging to make accurate predictions due to the existence of non-stationarity in real-world data, denoting the data distribution rapidly changes over time. To mitigate such a dilemma, several efforts have been conducted by reducing the non-stationarity with normalization operation. However, these methods typically overlook the distribution discrepancy between the input series and the horizon series, and assume that all time points within the same instance share the same statistical properties, which is too ideal and may lead to suboptimal relative improvements. To this end, we propose a novel slice-level adaptive normalization, referred to \textbf{SAN}, which is a novel scheme for empowering time series forecasting with more flexible normalization and denormalization. SAN includes two crucial designs. First, SAN tries to eliminate the non-stationarity of time series in units of a local temporal slice (i.e., sub-series) rather than a global instance. Second, SAN employs a slight network module to independently model the evolving trends of statistical properties of raw time series. Consequently, SAN could serve as a general model-agnostic plugin and better alleviate the impact of the non-stationary nature of time series data. We instantiate the proposed SAN on four widely used forecasting models and test their prediction results on benchmark datasets to evaluate its effectiveness. Also, we report some insightful findings to deeply analyze and understand our proposed SAN. We make our codes publicly available.</details>

### Adaptive Selective Sampling for Online Prediction with Experts

**Authors:** Rui Castro, Fredrik Hellström, Tim van Erven

**<details><summary>Abstract**</summary>

We consider online prediction of a binary sequence with expert advice. For this setting, we devise label-efficient forecasting algorithms, which use a selective sampling scheme that enables collecting much fewer labels than standard procedures. For the general case without a perfect expert, we prove best-of-both-worlds guarantees, demonstrating that the proposed forecasting algorithm always queries sufficiently many labels in the worst case to obtain optimal regret guarantees, while simultaneously querying much fewer labels in more benign settings. Specifically, for a scenario where one expert is strictly better than the others in expectation, we show that the label complexity of the label-efficient forecaster is roughly upper-bounded by the square root of the number of rounds. Finally, we present numerical experiments empirically showing that the normalized regret of the label-efficient forecaster can asymptotically match known minimax rates for pool-based active learning, suggesting it can optimally adapt to benign settings.</details>

### Adaptive Topological Feature via Persistent Homology: Filtration Learning for Point Clouds

**Authors:** Naoki Nishikawa, Yuichi Ike, Kenji Yamanishi

**<details><summary>Abstract**</summary>

Machine learning for point clouds has been attracting much attention, with many applications in various fields, such as shape recognition and material science. For enhancing the accuracy of such machine learning methods, it is often effective to incorporate global topological features, which are typically extracted by persistent homology. In the calculation of persistent homology for a point cloud, we choose a filtration for the point cloud, an increasing sequence of spaces. Since the performance of machine learning methods combined with persistent homology is highly affected by the choice of a filtration, we need to tune it depending on data and tasks. In this paper, we propose a framework that learns a filtration adaptively with the use of neural networks. In order to make the resulting persistent homology isometry-invariant, we develop a neural network architecture with such invariance. Additionally, we show a theoretical result on a finite-dimensional approximation of filtration functions, which justifies the proposed network architecture. Experimental results demonstrated the efficacy of our framework in several classification tasks.</details>

### Adaptive Uncertainty Estimation via High-Dimensional Testing on Latent Representations

**Authors:** Tsai Hor Chan, Kin Wai Lau, Jiajun Shen, Guosheng Yin, Lequan Yu

**<details><summary>Abstract**</summary>

Uncertainty estimation aims to evaluate the confidence of a trained deep neural network. However, existing uncertainty estimation approaches rely on low-dimensional distributional assumptions and thus suffer from the high dimensionality of latent features. Existing approaches tend to focus on uncertainty on discrete classification probabilities, which leads to poor generalizability to uncertainty estimation for other tasks. Moreover, most of the literature requires seeing the out-of-distribution (OOD) data in the training for better estimation of uncertainty, which limits the uncertainty estimation performance in practice because the OOD data are typically unseen. To overcome these limitations, we propose a new framework using data-adaptive high-dimensional hypothesis testing for uncertainty estimation, which leverages the statistical properties of the feature representations. Our method directly operates on latent representations and thus does not require retraining the feature encoder under a modified objective. The test statistic relaxes the feature distribution assumptions to high dimensionality, and it is more discriminative to uncertainties in the latent representations. We demonstrate that encoding features with Bayesian neural networks can enhance testing performance and lead to more accurate uncertainty estimation. We further introduce a family-wise testing procedure to determine the optimal threshold of OOD detection, which minimizes the false discovery rate (FDR). Extensive experiments validate the satisfactory performance of our framework on uncertainty estimation and task-specific prediction over a variety of competitors. The experiments on the OOD detection task also show satisfactory performance of our method when the OOD data are unseen in the training. Codes are available at https://github.com/HKU-MedAI/bnn_uncertainty.</details>

### [Oral] Additive Decoders for Latent Variables Identification and Cartesian-Product Extrapolation

**Authors:** Sébastien Lachapelle, Divyat Mahajan, Ioannis Mitliagkas, Simon Lacoste-Julien

**Oral Presentation:** Tu, Dec 12, 14:10 -- Oral 2B

**<details><summary>Abstract**</summary>

We tackle the problems of latent variables identification and "out-of-support'' image generation in representation learning. We show that both are possible for a class of decoders that we call additive, which are reminiscent of decoders used for object-centric representation learning (OCRL) and well suited for images that can be decomposed as a sum of object-specific images. We provide conditions under which exactly solving the reconstruction problem using an additive decoder is guaranteed to identify the blocks of latent variables up to permutation and block-wise invertible transformations. This guarantee relies only on very weak assumptions about the distribution of the latent factors, which might present statistical dependencies and have an almost arbitrarily shaped support. Our result provides a new setting where nonlinear independent component analysis (ICA) is possible and adds to our theoretical understanding of OCRL methods. We also show theoretically that additive decoders can generate novel images by recombining observed factors of variations in novel ways, an ability we refer to as Cartesian-product extrapolation. We show empirically that additivity is crucial for both identifiability and extrapolation on simulated data.</details>

### Addressing Negative Transfer in Diffusion Models

**Authors:** Hyojun Go, Kim, Yunsung Lee, Seunghyun Lee, Shinhyeok Oh, Hyeongdon Moon, Seungtaek Choi

**<details><summary>Abstract**</summary>

Diffusion-based generative models have achieved remarkable success in various domains. It trains a shared model on denoising tasks that encompass different noise levels simultaneously, representing a form of multi-task learning (MTL). However, analyzing and improving diffusion models from an MTL perspective remains under-explored. In particular, MTL can sometimes lead to the well-known phenomenon of 	extit{negative transfer}, which results in the performance degradation of certain tasks due to conflicts between tasks. In this paper, we first aim to analyze diffusion training from an MTL standpoint, presenting two key observations: 	extbf{(O1)} the task affinity between denoising tasks diminishes as the gap between noise levels widens, and 	extbf{(O2)} negative transfer can arise even in diffusion training. Building upon these observations, we aim to enhance diffusion training by mitigating negative transfer. To achieve this, we propose leveraging existing MTL methods, but the presence of a huge number of denoising tasks makes this computationally expensive to calculate the necessary per-task loss or gradient. To address this challenge, we propose clustering the denoising tasks into small task clusters and applying MTL methods to them. Specifically, based on 	extbf{(O2)}, we employ interval clustering to enforce temporal proximity among denoising tasks within clusters. We show that interval clustering can be solved using dynamic programming, utilizing signal-to-noise ratio, timestep, and task affinity for clustering objectives. Through this, our approach addresses the issue of negative transfer in diffusion models by allowing for efficient computation of MTL methods. We validate the proposed clustering and its integration with MTL methods through various experiments, demonstrating improved sample quality of diffusion models. Our project page is available at https://gohyojun15.github.io/ANT_diffusion.</details>

### Adjustable Robust Reinforcement Learning for Online 3D Bin Packing

**Authors:** Yuxin Pan, Yize Chen, Fangzhen Lin

**<details><summary>Abstract**</summary>

Designing effective policies for the online 3D bin packing problem (3D-BPP) has been a long-standing challenge, primarily due to the unpredictable nature of incoming box sequences and stringent physical constraints.  While current deep reinforcement learning (DRL) methods for online 3D-BPP have shown promising results in optimizing average performance over an underlying box sequence distribution, they often fail in real-world settings where some worst-case scenarios can materialize. Standard robust DRL algorithms tend to overly prioritize optimizing the worst-case performance at the expense of performance under normal problem instance distribution. To address these issues, we first introduce a permutation-based attacker to investigate the practical robustness of both DRL-based and heuristic methods proposed for solving online 3D-BPP. Then, we propose an adjustable robust reinforcement learning (AR2L) framework that allows efficient adjustment of robustness weights to achieve the desired balance of the policy's performance in average and worst-case environments. Specifically, we formulate the objective function as a weighted sum of expected and worst-case returns, and derive the lower performance bound  by relating to the return under a mixture dynamics. To realize this lower bound, we adopt an iterative procedure that searches for the associated mixture dynamics and improves the corresponding policy. We integrate this procedure into two popular robust adversarial algorithms to develop the exact and approximate AR2L algorithms. Experiments demonstrate that AR2L is versatile in the sense that it improves policy robustness while maintaining an acceptable level of performance for the nominal case.</details>

### Adversarial Model for Offline Reinforcement Learning

**Authors:** Mohak Bhardwaj, Tengyang Xie, Byron Boots, Nan Jiang, Ching-An Cheng

**<details><summary>Abstract**</summary>

We propose a novel model-based offline Reinforcement Learning (RL) framework, called Adversarial Model for Offline Reinforcement Learning (ARMOR), which can robustly learn policies to improve upon an arbitrary reference policy regardless of data coverage. ARMOR is designed to optimize policies for the worst-case performance relative to the reference policy through adversarially training a Markov decision process model. In theory, we prove that ARMOR, with a well-tuned hyperparameter, can compete with the best policy within data coverage when the reference policy is supported by the data. At the same time, ARMOR is robust to hyperparameter choices: the policy learned by ARMOR, with any admissible hyperparameter, would never degrade the performance of the reference policy, even when the reference policy is not covered by the dataset. To validate these properties in practice, we design a scalable implementation of ARMOR, which by adversarial training, can optimize policies without using model ensembles in contrast to typical model-based methods. We show that ARMOR achieves competent performance with both state-of-the-art offline model-free and model-based RL algorithms and can robustly improve the reference policy over various hyperparameter choices.</details>

### Adversarially Robust Distributed Count Tracking via Partial Differential Privacy

**Authors:** Zhongzheng Xiong, Xiaoyi Zhu, zengfeng Huang

**<details><summary>Abstract**</summary>

We study the distributed tracking model, also known as distributed functional monitoring. This model involves $k$ sites each receiving a stream of items and communicating with the central server. The server's task is to track a function of all items received thus far continuously, with minimum communication cost. For count tracking, it is known that there is a $\sqrt{k}$ gap in communication between deterministic and randomized algorithms. However, existing randomized algorithms assume an "oblivious adversary" who constructs the entire input streams before the algorithm starts. Here we consider adaptive adversaries who can choose new items based on previous answers from the algorithm. Deterministic algorithms are trivially robust to adaptive adversaries, while randomized ones may not. Therefore, we investigate whether the $\sqrt{k}$ advantage of randomized algorithms is from randomness itself or the oblivious adversary assumption. We provide an affirmative answer to this question by giving a robust algorithm with optimal communication. Existing robustification techniques do not yield optimal bounds due to the inherent challenges of the distributed nature of the problem. To address this, we extend the differential privacy framework by introducing "partial differential privacy" and proving a new generalization theorem. This theorem may have broader applications beyond robust count tracking, making it of independent interest.</details>

### Adversarially Robust Learning with Uncertain Perturbation Sets

**Authors:** Tosca Lechner, Vinayak Pathak, Ruth Urner

**<details><summary>Abstract**</summary>

In many real-world settings exact perturbation sets to be used by an adversary are not plausibly available to a learner. While prior literature has studied both scenarios with completely known and completely unknown perturbation sets, we propose an in-between setting of learning with respect to a class of perturbation sets. We show that in this setting we can improve on previous results with completely unknown perturbation sets, while still addressing the concerns of not having perfect knowledge of these sets in real life. In particular, we give the first positive results for the learnability of infinite Littlestone classes when having access to a perfect-attack oracle. We also consider a setting of learning with abstention, where predictions are considered robustness violations, only when the wrong prediction is made within the perturbation set. We show there are classes for which perturbation-set unaware learning without query access is possible, but abstention is required.</details>

### Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization

**Authors:** Jameel Abdul Samadh, Mohammad Hanan Gani, Noor Hussein, Muhammad Uzair Khattak, Muhammad Muzammal Naseer, Fahad Shahbaz Khan, Salman Khan

**<details><summary>Abstract**</summary>

The promising zero-shot generalization of vision-language models such as CLIP has led to their adoption using prompt learning for numerous downstream tasks. Previous works have shown test-time prompt tuning using entropy minimization to adapt text prompts for unseen domains. While effective, this overlooks the key cause for performance degradation to unseen domains -- distribution shift. In this work, we explicitly handle this problem by aligning the out-of-distribution (OOD) test sample statistics to those of the source data using prompt tuning. We use a single test sample to adapt multi-modal prompts at test time by minimizing the feature distribution shift to bridge the gap in the test domain. Evaluating against the domain generalization benchmark, our method improves zero-shot top-1 accuracy beyond existing prompt-learning techniques, with a 3.08% improvement over the baseline MaPLe. In cross-dataset generalization with unseen categories across 10 datasets, our method improves consistently across all datasets compared to the existing state-of-the-art. Our source code and models are available at [https://jameelhassan.github.io/promptalign](https://jameelhassan.github.io/promptalign)</details>

### [Spotlight] AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback

**Authors:** Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, Tatsunori Hashimoto

**<details><summary>Abstract**</summary>

Large language models (LLMs) such as ChatGPT have seen widespread adoption due to their ability to follow user instructions well.Developing these LLMs involves a complex yet poorly understood workflow requiring training with human feedback. Replicating and understanding this instruction-following process faces three major challenges: the high cost of data collection, the lack of trustworthy evaluation, and the absence of reference method implementations. We address these bottlenecks with AlpacaFarm, a simulator that enables research and development for learning from feedback at a low cost. First, we design LLM based simulator for human feedback that is 45x cheaper than crowdworkers and displays high agreement with humans. Second, we identify an evaluation dataset representative of real-world instructions and propose an automatic evaluation procedure. Third, we contribute reference implementations for several methods (PPO, best-of-n, expert iteration, among others) that learn from pairwise feedback. Finally, as an end-to-end validation of AlpacaFarm, we train and evaluate eleven models on 10k pairs of human feedback and show that rankings of models trained in AlpacaFarm match rankings of models trained on human data. As a demonstration of the research possible in AlpacaFarm, we find that methods that use a reward model can substantially improve over supervised fine-tuning and that our reference PPO implementation leads to a +10% win-rate improvement against Davinci003.</details>

### AmadeusGPT: a natural language interface for interactive animal behavioral analysis

**Authors:** shaokai ye, Jessy Lauer, Mu Zhou, Alexander Mathis, Mackenzie Mathis

**<details><summary>Abstract**</summary>

The process of quantifying and analyzing animal behavior involves translating the naturally occurring descriptive language of their actions into machine-readable code. Yet, codifying behavior analysis is often challenging without deep understanding of animal behavior and technical machine learning knowledge. To limit this gap, we introduce AmadeusGPT: a natural language interface that turns natural language descriptions of behaviors into machine-executable code. Large-language models (LLMs) such as GPT3.5 and GPT4 allow for interactive language-based queries that are potentially well suited for making interactive behavior analysis. However, the comprehension capability of these LLMs is limited by the context window size, which prevents it from remembering distant conversations. To overcome the context window limitation, we implement a novel dual-memory mechanism to allow communication between short-term and long-term memory using symbols as context pointers for retrieval and saving. Concretely, users directly use language-based definitions of behavior and our augmented GPT develops code based on the core AmadeusGPT API, which contains machine learning, computer vision, spatio-temporal reasoning, and visualization modules. Users then can interactively refine results, and seamlessly add new behavioral modules as needed. We used the MABe 2022 behavior challenge tasks to benchmark AmadeusGPT and show excellent performance. Note, an end-user would not need to write any code to achieve this. Thus, collectively AmadeusGPT presents a novel way to merge deep biological knowledge, large-language models, and core computer vision modules into a more naturally intelligent system. Code and demos can be found at: https://github.com/AdaptiveMotorControlLab/AmadeusGPT</details>

### An Alternative to Variance: Gini Deviation for Risk-averse Policy Gradient

**Authors:** Yudong Luo, Guiliang Liu, Pascal Poupart, Yangchen Pan

**<details><summary>Abstract**</summary>

Restricting the variance of a policy’s return is a popular choice in risk-averse Reinforcement Learning (RL) due to its clear mathematical definition and easy interpretability. Traditional methods directly restrict the total return variance. Recent methods restrict the per-step reward variance as a proxy. We thoroughly examine the limitations of these variance-based methods, such as sensitivity to numerical scale and hindering of policy learning, and propose to use an alternative risk measure, Gini deviation, as a substitute. We study various properties of this new risk measure and derive a policy gradient algorithm to minimize it. Empirical evaluation in domains where risk-aversion can be clearly defined, shows that our algorithm can mitigate the limitations of variance-based risk measures and achieves high return with low risk in terms of variance and Gini deviation when others fail to learn a reasonable policy.</details>

### An Efficient End-to-End Training Approach for Zero-Shot Human-AI Coordination

**Authors:** Xue Yan, Jiaxian Guo, Xingzhou Lou, Jun Wang, Haifeng Zhang, Yali Du

**<details><summary>Abstract**</summary>

The goal of zero-shot human-AI coordination is to develop an agent that can collaborate with humans without relying on human data. Prevailing two-stage population-based methods require a diverse population of mutually distinct policies to simulate diverse human behaviors. The necessity of such populations severely limits their computational efficiency. To address this issue, we propose E3T, an **E**fficient **E**nd-to-**E**nd **T**raining approach for zero-shot human-AI coordination. E3T employs a mixture of ego policy and random policy to construct the partner policy, making it both coordination-skilled and diverse. In this way, the ego agent is end-to-end trained with this mixture policy without the need of a pre-trained population, thus significantly improving the training efficiency. In addition, a partner modeling module is proposed to predict the partner's action from historical information. With the predicted partner's action, the ego policy is able to adapt its policy and take actions accordingly when collaborating with humans of different behavior patterns. Empirical results on the Overcooked environment show that our method significantly improves the training efficiency while preserving comparable or superior performance than the population-based baselines. Demo videos are available at https://sites.google.com/view/e3t-overcooked.</details>

### An Improved Relaxation for Oracle-Efficient Adversarial Contextual Bandits

**Authors:** Kiarash Banihashem, MohammadTaghi Hajiaghayi, Suho Shin, Max Springer

**<details><summary>Abstract**</summary>

We present an oracle-efficient relaxation  for the adversarial contextual bandits problem,  where the contexts are sequentially drawn i.i.d from a known distribution and the cost sequence is chosen by an online adversary.  Our algorithm has a regret bound of $O(T^{\frac{2}{3}}(K\log(|\Pi|))^{\frac{1}{3}})$ and makes at most $O(K)$ calls per round to an offline optimization oracle,  where $K$ denotes the number of actions, $T$ denotes the number of rounds and $\Pi$ denotes   the set of policies.  This is the first result to improve the prior best bound of $O((TK)^{\frac{2}{3}}(\log(|\Pi|))^{\frac{1}{3}})$ as obtained by   Syrgkanis et al.  at NeurIPS 2016, and the first to match the original bound of   Langford and Zhang at NeurIPS 2007  which was obtained for the stochastic case.</details>

### Analyzing Generalization of Neural Networks through Loss Path Kernels

**Authors:** Yilan Chen, Wei Huang, Hao Wang, Charlotte Loh, Akash Srivastava, Lam Nguyen, Lily Weng

**<details><summary>Abstract**</summary>

Deep neural networks have been increasingly used in real-world applications, making it critical to ensure their ability to adapt to new, unseen data. In this paper, we study the generalization capability of neural networks trained with (stochastic) gradient flow. We establish a new connection between the loss dynamics of gradient flow and general kernel machines by proposing a new kernel, called loss path kernel. This kernel measures the similarity between two data points by evaluating the agreement between loss gradients along the path determined by the gradient flow. Based on this connection, we derive a new generalization upper bound that applies to general neural network architectures. This new bound is tight and strongly correlated with the true generalization error. We apply our results to guide the design of neural architecture search (NAS) and demonstrate favorable performance compared with state-of-the-art NAS algorithms through numerical experiments.</details>

### Any-to-Any Generation via Composable Diffusion

**Authors:** Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, Mohit Bansal

**<details><summary>Abstract**</summary>

We present Composable Diffusion (CoDi), a novel generative model capable of generating any combination of output modalities, such as language, image, video, or audio, from any combination of input modalities. Unlike existing generative AI systems, CoDi can generate multiple modalities in parallel and its input is not limited to a subset of modalities like text or image. Despite the absence of training datasets for many combinations of modalities, we propose to align modalities in both the input and output space. This allows CoDi to freely condition on any input combination and generate any group of modalities, even if they are not present in the training data. CoDi employs a novel composable generation strategy which involves building a shared multimodal space by bridging alignment in the diffusion process, enabling the synchronized generation of intertwined modalities, such as temporally aligned video and audio. Highly customizable and flexible, CoDi achieves strong joint-modality generation quality, and outperforms or is on par with the unimodal state-of-the-art for single-modality synthesis.</details>

### Attention as Implicit Structural Inference

**Authors:** Ryan Singh, Christopher L Buckley

**<details><summary>Abstract**</summary>

Attention mechanisms play a crucial role in cognitive systems by allowing them to flexibly allocate cognitive resources. Transformers, in particular, have become a dominant architecture in machine learning, with attention as their central innovation. However, the underlying intuition and formalism of attention in Transformers is based on ideas of keys and queries in database management systems. In this work, we pursue a structural inference perspective, building upon, and bringing together, previous theoretical descriptions of attention such as; Gaussian Mixture Models, alignment mechanisms and Hopfield Networks. Specifically, we demonstrate that attention can be viewed as inference over an implicitly defined set of possible adjacency structures in a graphical model, revealing the generality of such a mechanism. This perspective unifies different attentional architectures in machine learning and suggests potential modifications and generalizations of attention. Here we investigate two and demonstrate their behaviour on explanatory toy problems: (a) extending the value function to incorporate more nodes of a graphical model yielding a mechanism with a bias toward attending  multiple tokens; (b) introducing a geometric prior (with conjugate hyper-prior) over the adjacency structures producing a mechanism which dynamically scales the context window depending on input. Moreover, by describing a link between structural inference and precision-regulation in Predictive Coding Networks, we discuss how this framework can bridge the gap between attentional mechanisms in machine learning and Bayesian conceptions of attention in Neuroscience. We hope by providing a new lens on attention architectures our work can guide the development of new and improved attentional mechanisms.</details>

### Augmenting Language Models with Long-Term Memory

**Authors:** Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, Furu Wei

**<details><summary>Abstract**</summary>

Existing large language models (LLMs) can only afford fix-sized inputs due to the input length limit, preventing them from utilizing rich long-context information from past inputs. To address this, we propose a framework, Language Models Augmented with Long-Term Memory (LongMem), which enables LLMs to memorize long history. We design a novel decoupled network architecture with the original backbone LLM frozen as a memory encoder and an adaptive residual side-network as a memory retriever and reader. Such a decoupled memory design can easily cache and update long-term past contexts for memory retrieval without suffering from memory staleness. Enhanced with memory-augmented adaptation training, LongMem can thus memorize long past context and use long-term memory for language modeling. The proposed memory retrieval module can handle unlimited-length context in its memory bank to benefit various downstream tasks. Typically, LongMem can enlarge the long-form memory to 65k tokens and thus cache many-shot extra demonstration examples as long-form memory for in-context learning. Experiments show that our method outperforms strong long-context models on ChapterBreak, a challenging long-context modeling benchmark, and achieves remarkable improvements on memory-augmented in-context learning over LLMs. The results demonstrate that the proposed method is effective in helping language models to memorize and utilize long-form contents.</details>

### Automatic Grouping for Efficient Cooperative Multi-Agent Reinforcement Learning

**Authors:** Yifan Zang, Jinmin He, Kai Li, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng

**<details><summary>Abstract**</summary>

Grouping is ubiquitous in natural systems and is essential for promoting efficiency in team coordination. This paper proposes a novel formulation of Group-oriented Multi-Agent Reinforcement Learning (GoMARL), which learns automatic grouping without domain knowledge for efficient cooperation. In contrast to existing approaches that attempt to directly learn the complex relationship between the joint action-values and individual utilities, we empower grouping as a bridge to model the connection between small sets of agents and encourage cooperation among them, thereby improving the learning efficiency of the whole team. In particular, we factorize the joint action-values as a combination of group-wise values, which guide agents to improve their policies in a fine-grained fashion. We present an automatic grouping mechanism to generate dynamic groups and group action-values. We further introduce a hierarchical control for policy learning that drives the agents in the same group to specialize in similar policies and possess diverse strategies for various groups. Experiments on the StarCraft II micromanagement tasks and Google Research Football scenarios verify our method's effectiveness. Extensive component studies show how grouping works and enhances performance.</details>

### BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization

**Authors:** Darko Drakulic, Sofia Michel, Florian Mai, Arnaud Sors, Jean-Marc Andreoli

**<details><summary>Abstract**</summary>

Despite the success of neural-based combinatorial optimization methods for end-to-end heuristic learning, out-of-distribution generalization remains a challenge. In this paper, we present a novel formulation of Combinatorial Optimization Problems (COPs) as Markov Decision Processes (MDPs) that effectively leverages common symmetries of COPs to improve out-of-distribution robustness. Starting from a direct MDP formulation of a constructive method, we introduce a generic way to reduce the state space, based on Bisimulation Quotienting (BQ) in MDPs. Then, for COPs with a recursive nature, we specialize the bisimulation and show how the reduced state exploits the symmetries of these problems and facilitates MDP solving. Our approach is principled and we prove that an optimal policy for the proposed BQ-MDP actually solves the associated COPs. We illustrate our approach on five classical problems: the Euclidean and Asymmetric Traveling Salesman, Capacitated Vehicle Routing, Orienteering and Knapsack Problems. Furthermore, for each problem, we introduce a simple attention-based policy network for the BQ-MDPs, which we train by imitation of (near) optimal solutions of small instances from a single distribution. We obtain new state-of-the-art results for the five COPs on both synthetic and realistic benchmarks. Notably, in contrast to most existing neural approaches, our learned policies show excellent generalization performance to much larger instances than seen during training, without any additional search procedure. Our code is available at: [link](https://github.com/naver/bq-nco).</details>

### Balancing Risk and Reward: A Batched-Bandit Strategy for Automated Phased Release

**Authors:** Yufan Li, Jialiang Mao, Iavor Bojinov

**<details><summary>Abstract**</summary>

Phased releases are a common strategy in the technology industry for gradually releasing new products or updates through a sequence of A/B tests in which the number of treated units gradually grows until full deployment or deprecation. Performing phased releases in a principled way requires selecting the proportion of units assigned to the new release in a way that balances the risk of an adverse effect with the need to iterate and learn from the experiment rapidly. In this paper, we formalize this problem and propose an algorithm that automatically determines the release percentage at each stage in the schedule, balancing the need to control risk while maximizing ramp-up speed. Our framework models the challenge as a constrained batched bandit problem that ensures that our pre-specified experimental budget is not depleted with high probability. Our proposed algorithm leverages an adaptive Bayesian approach in which the maximal number of units assigned to the treatment is determined by the posterior distribution, ensuring that the probability of depleting the remaining budget is low. Notably, our approach analytically solves the ramp sizes by inverting probability bounds, eliminating the need for challenging rare-event Monte Carlo simulation. It only requires computing means and variances of outcome subsets, making it highly efficient and parallelizable.</details>

### [Spotlight] Banana: Banach Fixed-Point Network for Pointcloud Segmentation with Inter-Part Equivariance

**Authors:** Congyue Deng, Jiahui Lei, William B Shen, Kostas Daniilidis, Leonidas Guibas

**<details><summary>Abstract**</summary>

Equivariance has gained strong interest as a desirable network property that inherently ensures robust generalization. However, when dealing with complex systems such as articulated objects or multi-object scenes, effectively capturing inter-part transformations poses a challenge, as it becomes entangled with the overall structure and local transformations. The interdependence of part assignment and per-part group action necessitates a novel equivariance formulation that allows for their co-evolution. In this paper, we present Banana, a Banach fixed-point network for equivariant segmentation with inter-part equivariance by construction. Our key insight is to iteratively solve a fixed-point problem, where point-part assignment labels and per-part SE(3)-equivariance co-evolve simultaneously. We provide theoretical derivations of both per-step equivariance and global convergence, which induces an equivariant final convergent state. Our formulation naturally provides a strict definition of inter-part equivariance that generalizes to unseen inter-part configurations. Through experiments conducted on both articulated objects and multi-object scans, we demonstrate the efficacy of our approach in achieving strong generalization under inter-part transformations, even when confronted with substantial changes in pointcloud geometry and topology.</details>

### Bayesian Learning via Q-Exponential Process

**Authors:** Shuyi Li, Michael O'Connor, Shiwei Lan

**<details><summary>Abstract**</summary>

Regularization is one of the most fundamental topics in optimization, statistics and machine learning. To get sparsity in estimating a parameter $u\in\mathbb{R}^d$, an $\ell_q$ penalty term, $\Vert u\Vert_q$, is usually added to the objective function. What is the probabilistic distribution corresponding to such $\ell_q$ penalty? What is the \emph{correct} stochastic process corresponding to $\Vert u\Vert_q$ when we model functions $u\in L^q$? This is important for statistically modeling high-dimensional objects such as images, with penalty to preserve certainty properties, e.g. edges in the image.In this work, we generalize the $q$-exponential distribution (with density proportional to) $\exp{(- \frac{1}{2}|u|^q)}$ to a stochastic process named \emph{$Q$-exponential (Q-EP) process} that corresponds to the $L_q$ regularization of functions. The key step is to specify consistent multivariate $q$-exponential distributions by choosing from a large family of elliptic contour distributions. The work is closely related to Besov process which is usually defined in terms of series. Q-EP can be regarded as a definition of Besov process with explicit probabilistic formulation, direct control on the correlation strength, and tractable prediction formula. From the Bayesian perspective, Q-EP provides a flexible prior on functions with sharper penalty ($q<2$) than the commonly used Gaussian process (GP, $q=2$).We compare GP, Besov and Q-EP in modeling functional data, reconstructing images and solving inverse problems and demonstrate the advantage of our proposed methodology.</details>

### Bayesian Metric Learning for Uncertainty Quantification in Image Retrieval

**Authors:** Frederik Warburg, Marco Miani, Silas Brack, Søren Hauberg

**<details><summary>Abstract**</summary>

We propose a Bayesian encoder for metric learning. Rather than relying on neural amortization as done in prior works, we learn a distribution over the network weights with the Laplace Approximation. We first prove that the contrastive loss is a negative log-likelihood on the spherical space. We propose three methods that ensure a positive definite covariance matrix. Lastly, we present a novel decomposition of the Generalized Gauss-Newton approximation. Empirically, we show that our Laplacian Metric Learner (LAM) yields well-calibrated uncertainties, reliably detects out-of-distribution examples, and has state-of-the-art predictive performance.</details>

### Beyond Geometry: Comparing the Temporal Structure of Computation in Neural Circuits with Dynamical Similarity Analysis

**Authors:** Mitchell Ostrow, Adam Eisen, Leo Kozachkov, Ila Fiete

**<details><summary>Abstract**</summary>

How can we tell whether two neural networks utilize the same internal processes for a particular computation? This question is pertinent for multiple subfields of neuroscience and machine learning, including neuroAI, mechanistic interpretability, and brain-machine interfaces. Standard approaches for comparing neural networks focus on the spatial geometry of latent states. Yet in recurrent networks, computations are implemented at the level of dynamics, and two networks performing the same computation with equivalent dynamics need not exhibit the same geometry. To bridge this gap, we introduce a novel similarity metric that compares two systems at the level of their dynamics, called Dynamical Similarity Analysis (DSA). Our method incorporates two components: Using recent advances in data-driven dynamical systems theory, we learn a high-dimensional linear system that accurately captures core features of the original nonlinear dynamics. Next, we compare different systems passed through this embedding using a novel extension of Procrustes Analysis that accounts for how vector fields change under orthogonal transformation. In four case studies, we demonstrate that our method disentangles conjugate and non-conjugate recurrent neural networks (RNNs), while geometric methods fall short. We additionally show that our method can distinguish learning rules in an unsupervised manner. Our method opens the door to comparative analyses of the essential temporal structure of computation in neural circuits.</details>

### Beyond Invariance: Test-Time Label-Shift Adaptation for Addressing "Spurious" Correlations

**Authors:** Qingyao Sun, Kevin Murphy, Sayna Ebrahimi, Alexander D'Amour

**<details><summary>Abstract**</summary>

Changes in the data distribution at test time can have deleterious effects on the performance of predictive models $p(y|x)$.We consider situations where there are additional meta-data labels (such as group labels), denoted by $z$, that can account for such changes in the distribution.In particular, we assume that the prior distribution $p(y,z)$, which models the dependence between the class label $y$ and the "nuisance" factors $z$, may change across domains, either due to a change in the correlation between these terms, or a change in one of their marginals.However, we assume that the generative model for features $p(x|y,z)$ is invariant across domains.We note that this corresponds to an expanded version of the widely used "label shift" assumption, where the labels now also include the nuisance factors $z$. Based on this observation, we propose a test-time label shift correction that adapts to changes in the joint distribution $p(y, z)$ using EM applied to unlabeled samples from the target domain distribution, $p_t(x)$.Importantly, we are able to avoid fitting a generative model $p(x|y,z)$, and merely need to reweight the outputs of a discriminative model $p_s(y,z|x)$ trained on the source distribution.We evaluate our method, which we call "Test-Time Label-Shift Adaptation" (TTLSA), on several standard image and text datasets, as well as the CheXpert chest X-ray dataset, and show that it improves performance over methods that target invariance to changes in the distribution, as well as baseline empirical risk minimization methods.Code for reproducing experiments is available at https://github.com/nalzok/test-time-label-shift.</details>

### Beyond Normal: On the Evaluation of Mutual Information Estimators

**Authors:** Paweł Czyż, Frederic Grabowski, Julia Vogt, Niko Beerenwinkel, Alexander Marx

**<details><summary>Abstract**</summary>

Mutual information is a general statistical dependency measure which has found applications in representation learning, causality, domain generalization and computational biology. However, mutual information estimators are typically evaluated on simple families of probability distributions, namely multivariate normal distribution and selected distributions with one-dimensional random variables. In this paper, we show how to construct a diverse family of distributions with known ground-truth mutual information and propose a language-independent benchmarking platform for mutual information estimators. We discuss the general applicability and limitations of classical and neural estimators in settings involving high dimensions, sparse interactions, long-tailed distributions, and high mutual information. Finally, we provide guidelines for practitioners on how to select appropriate estimator adapted to the difficulty of problem considered and issues one needs to consider when applying an estimator to a new data set.</details>

### Beyond probability partitions: Calibrating neural networks with semantic aware grouping

**Authors:** Jia-Qi Yang, De-Chuan Zhan, Le Gan

**<details><summary>Abstract**</summary>

Research has shown that deep networks tend to be overly optimistic about their predictions, leading to an underestimation of prediction errors. Due to the limited nature of data, existing studies have proposed various methods based on model prediction probabilities to bin the data and evaluate calibration error. We propose a more generalized definition of calibration error called Partitioned Calibration Error (PCE), revealing that the key difference among these calibration error metrics lies in how the data space is partitioned. We put forth an intuitive proposition that an accurate model should be calibrated across any partition, suggesting that the input space partitioning can extend beyond just the partitioning of prediction probabilities, and include partitions directly related to the input. Through semantic-related partitioning functions, we demonstrate that the relationship between model accuracy and calibration lies in the granularity of the partitioning function. This highlights the importance of partitioning criteria for training a calibrated and accurate model. To validate the aforementioned analysis, we propose a method that involves jointly learning a semantic aware grouping function based on deep model features and logits to partition the data space into subsets. Subsequently, a separate calibration function is learned for each subset. Experimental results demonstrate that our approach achieves significant performance improvements across multiple datasets and network architectures, thus highlighting the importance of the partitioning function for calibration.</details>

### Bias in Evaluation Processes: An Optimization-Based Model

**Authors:** L. Elisa Celis, Amit Kumar, Anay Mehrotra, Nisheeth K. Vishnoi

**<details><summary>Abstract**</summary>

Biases with respect to socially-salient attributes of individuals have been well documented in evaluation processes used in settings such as admissions and hiring. We view such an evaluation process as a transformation of a  distribution of the true utility of an individual for a task to an observed distribution and model it as a solution to a loss minimization problem subject to an information constraint. Our model has two parameters that have been identified as factors leading to biases: the resource-information trade-off parameter in the information constraint and the risk-averseness parameter in the loss function.  We characterize the distributions that arise from our model and study the effect of the parameters on the observed distribution. The outputs of our model enrich the class of distributions that can be used to capture variation across groups in the observed evaluations. We empirically validate our model by fitting real-world datasets and use it to study the effect of interventions in a downstream selection task. These results contribute to an understanding of the emergence of bias in evaluation processes and provide tools to guide the deployment of interventions to mitigate biases.</details>

### Bicriteria Multidimensional Mechanism Design with Side Information

**Authors:** Siddharth Prasad, Maria-Florina Balcan, Tuomas Sandholm

**<details><summary>Abstract**</summary>

We develop a versatile new methodology for multidimensional mechanism design that incorporates side information about agent types to generate high social welfare and high revenue simultaneously. Prominent sources of side information in practice include predictions from a machine-learning model trained on historical agent data, advice from domain experts, and even the mechanism designer's own gut instinct. In this paper we adopt a prior-free perspective that makes no assumptions on the correctness, accuracy, or source of the side information. First, we design a meta-mechanism that integrates input side information with an improvement of the classical VCG mechanism. The welfare, revenue, and incentive properties of our meta-mechanism are characterized by novel constructions we introduce based on the notion of a weakest competitor, which is an agent that has the smallest impact on welfare. We show that our meta-mechanism, when carefully instantiated, simultaneously achieves strong welfare and revenue guarantees parameterized by errors in the side information. When the side information is highly informative and accurate, our mechanism achieves welfare and revenue competitive with the total social surplus, and its performance decays continuously and gradually as the quality of the side information decreases. Finally, we apply our meta-mechanism to a setting where each agent's type is determined by a constant number of parameters. Specifically, agent types lie on constant-dimensional subspaces (of the potentially high-dimensional ambient type space) that are known to the mechanism designer. We use our meta-mechanism to obtain the first known welfare and revenue guarantees in this setting.</details>

### Bilevel Coreset Selection in Continual Learning: A New Formulation and Algorithm

**Authors:** Jie Hao, Kaiyi Ji, Mingrui Liu

**<details><summary>Abstract**</summary>

Coreset is a small set that provides a data summary for a large dataset, such that training solely on the small set achieves competitive performance compared with a large dataset. In rehearsal-based continual learning, the coreset is typically used in the memory replay buffer to stand for representative samples in previous tasks, and the coreset selection procedure is typically formulated as a bilevel problem. However, the typical bilevel formulation for coreset selection explicitly performs optimization over discrete decision variables with greedy search, which is computationally expensive. Several works consider other formulations to address this issue, but they ignore the nested nature of bilevel optimization problems and may not solve the bilevel coreset selection problem accurately. To address these issues, we propose a new bilevel formulation, where the inner problem tries to find a model which minimizes the expected training error sampled from a given probability distribution, and the outer problem aims to learn the probability distribution with approximately $K$ (coreset size) nonzero entries such that learned model in the inner problem minimizes the training error over the whole data. To ensure the learned probability has approximately $K$ nonzero entries, we introduce a novel regularizer based on the smoothed top-$K$ loss in the upper problem. We design a new optimization algorithm that provably converges to the $\epsilon$-stationary point with $O(1/\epsilon^4)$ computational complexity. We conduct extensive experiments in various settings in continual learning, including balanced data, imbalanced data, and label noise, to show that our proposed formulation and new algorithm significantly outperform competitive baselines. From bilevel optimization point of view, our algorithm significantly improves the vanilla greedy coreset selection method in terms of running time on continual learning benchmark datasets. The code is available at https://github.com/MingruiLiu-ML-Lab/Bilevel-Coreset-Selection-via-Regularization.</details>

### [Spotlight] Birth of a Transformer: A Memory Viewpoint

**Authors:** Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Herve Jegou, Leon Bottou

**<details><summary>Abstract**</summary>

Large language models based on transformers have achieved great empirical successes. However, as they are deployed more widely, there is a growing need to better understand their internal mechanisms in order to make them more reliable. These models appear to store vast amounts of knowledge from their training data, and to adapt quickly to new information provided in their context or prompt. We study how transformers balance these two types of knowledge by considering a synthetic setup where tokens are generated from either global or context-specific bigram distributions. By a careful empirical analysis of the training process on a simplified two-layer transformer, we illustrate the fast learning of global bigrams and the slower development of an "induction head" mechanism for the in-context bigrams. We highlight the role of weight matrices as associative memories, provide theoretical insights on how gradients enable their learning during training, and study the role of data-distributional properties.</details>

### Block-Coordinate Methods and Restarting for Solving Extensive-Form Games

**Authors:** Darshan Chakrabarti, Jelena Diakonikolas, Christian Kroer

**<details><summary>Abstract**</summary>

Coordinate descent methods are popular in machine learning and optimization for their simple sparse updates and excellent practical performance. In the context of large-scale sequential game solving, these same properties would be attractive, but until now no such methods were known, because the strategy spaces do not satisfy the typical separable block structure exploited by such methods.We present the first cyclic coordinate-descent-like method for the polytope of sequence-form strategies, which form the strategy spaces for the players in an extensive-form game (EFG). Our method exploits the recursive structure of the proximal update induced by what are known as dilated regularizers, in order to allow for a pseudo block-wise update.We show that our method enjoys a O(1/T) convergence rate to a two-player zero-sum Nash equilibrium, while avoiding the worst-case polynomial scaling with the number of blocks common to cyclic methods. We empirically show that our algorithm usually performs better than other state-of-the-art first-order methods (i.e., mirror prox), and occasionally can even beat CFR$^+$, a state-of-the-art algorithm for numerical equilibrium computation in zero-sum EFGs. We then introduce a restarting heuristic for EFG solving. We show empirically that restarting can lead to speedups, sometimes huge, both for our cyclic method, as well as for existing methods such as mirror prox and predictive CFR$^+$.</details>

### Boosting Adversarial Transferability by Achieving Flat Local Maxima

**Authors:** Zhijin Ge, Wang Xiaosen, Hongying Liu, Fanhua Shang, Yuanyuan Liu

**<details><summary>Abstract**</summary>

Transfer-based attack adopts the adversarial examples generated on the surrogate model to attack various models, making it applicable in the physical world and attracting increasing interest. Recently, various adversarial attacks have emerged to boost adversarial transferability from different perspectives. In this work, inspired by the observation that flat local minima are correlated with good generalization, we assume and empirically validate that adversarial examples at a flat local region tend to have good transferability by introducing a penalized gradient norm to the original loss function. Since directly optimizing the gradient regularization norm is computationally expensive and intractable for generating adversarial examples, we propose an approximation optimization method to simplify the gradient update of the objective function. Specifically, we randomly sample an example and adopt a first-order procedure to approximate the curvature of the second-order Hessian matrix, which makes computing more efficient by interpolating two Jacobian matrices. Meanwhile, in order to obtain a more stable gradient direction, we randomly sample multiple examples and average the gradients of these examples to reduce the variance due to random sampling during the iterative process. Extensive experimental results on the ImageNet-compatible dataset show that the proposed method can generate adversarial examples at flat local regions, and significantly improve the adversarial transferability on either normally trained models or adversarially trained models than the state-of-the-art attacks. Our codes are available at: https://github.com/Trustworthy-AI-Group/PGN.</details>

### [Spotlight] Bootstrapping Vision-Language Learning with Decoupled Language Pre-training

**Authors:** Yiren Jian, Chongyang Gao, Soroush Vosoughi

**<details><summary>Abstract**</summary>

We present a novel methodology aimed at optimizing the application of frozen large language models (LLMs) for resource-intensive vision-language (VL) pre-training. The current paradigm uses visual features as prompts to guide language models, with a focus on determining the most relevant visual features for corresponding text. Our approach diverges by concentrating on the language component, specifically identifying the optimal prompts to align with visual features. We introduce the Prompt-Transformer (P-Former), a model that predicts these ideal prompts, which is trained exclusively on linguistic data, bypassing the need for image-text pairings. This strategy subtly bifurcates the end-to-end VL training process into an additional, separate stage. Our experiments reveal that our framework significantly enhances the performance of a robust image-to-text baseline (BLIP-2), and effectively narrows the performance gap between models trained with either 4M or 129M image-text pairs. Importantly, our framework is modality-agnostic and flexible in terms of architectural design, as validated by its successful application in a video learning task using varied base modules. The code will be made available at https://github.com/yiren-jian/BLIText.</details>

### Bounding training data reconstruction in DP-SGD

**Authors:** Jamie Hayes, Borja Balle, Saeed Mahloujifar

**<details><summary>Abstract**</summary>

Differentially private training offers a protection which is usually interpreted as a guarantee against membership inference attacks. By proxy, this guarantee extends to other threats like reconstruction attacks attempting to extract complete training examples. Recent works provide evidence that if one does not need to protect against membership attacks but instead only wants to protect against a training data reconstruction, then utility of private models can be improved because less noise is required to protect against these more ambitious attacks. We investigate this question further in the context of DP-SGD, a standard algorithm for private deep learning, and provide an upper bound on the success of any reconstruction attack against DP-SGD together with an attack that empirically matches the predictions of our bound. Together, these two results open the door to fine-grained investigations on how to set the privacy parameters of DP-SGD in practice to protect against reconstruction attacks. Finally, we use our methods to demonstrate that different settings of the DP-SGD parameters leading to same DP guarantees can results in significantly different success rates for reconstruction, indicating that the DP guarantee alone might not be a good proxy for controlling the protection against reconstruction attacks.</details>

### Breaking the Communication-Privacy-Accuracy Tradeoff with $f$-Differential Privacy

**Authors:** Richeng Jin, Zhonggen Su, caijun zhong, Zhaoyang Zhang, Tony Quek, Huaiyu Dai

**<details><summary>Abstract**</summary>

We consider a federated data analytics problem in which a server coordinates the collaborative data analysis of multiple users with privacy concerns and limited communication capability. The commonly adopted compression schemes introduce information loss into local data while improving communication efficiency, and it remains an open problem whether such discrete-valued mechanisms provide any privacy protection. In this paper, we study the local differential privacy guarantees of discrete-valued mechanisms with finite output space through the lens of $f$-differential privacy (DP). More specifically, we advance the existing literature by deriving tight $f$-DP guarantees for a variety of discrete-valued mechanisms, including the binomial noise and the binomial mechanisms that are proposed for privacy preservation, and the sign-based methods that are proposed for data compression, in closed-form expressions. We further investigate the amplification in privacy by sparsification and propose a ternary stochastic compressor. By leveraging compression for privacy amplification, we improve the existing methods by removing the dependency of accuracy (in terms of mean square error) on communication cost in the popular use case of distributed mean estimation, therefore breaking the three-way tradeoff between privacy, communication, and accuracy.</details>

### [Oral] Bridging Discrete and Backpropagation: Straight-Through and Beyond

**Authors:** Liyuan Liu, Chengyu Dong, Xiaodong Liu, Bin Yu, Jianfeng Gao

**Oral Presentation:** Tu, Dec 12, 14:25 -- Oral 2A

**<details><summary>Abstract**</summary>

Backpropagation, the cornerstone of deep learning, is limited to computing gradients for continuous variables. This limitation poses challenges for problems involving discrete latent variables. To address this issue, we propose a novel approach to approximate the gradient of parameters involved in generating discrete latent variables. First, we examine the widely used Straight-Through (ST) heuristic and demonstrate that it works as a first-order approximation of the gradient. Guided by our findings, we propose ReinMax, which achieves second-order accuracy by integrating Heun’s method, a second-order numerical method for solving ODEs. ReinMax does not require Hessian or other second-order derivatives, thus having negligible computation overheads. Extensive experimental results on various tasks demonstrate the superiority of ReinMax over the state of the art.</details>

### Budgeting Counterfactual for Offline RL

**Authors:** Yao Liu, Pratik Chaudhari, Rasool Fakoor

**<details><summary>Abstract**</summary>

The main challenge of offline reinforcement learning, where data is limited, arises from a sequence of counterfactual reasoning dilemmas within the realm of potential actions: What if we were to choose a different course of action? These circumstances frequently give rise to extrapolation errors, which tend to accumulate exponentially with the problem horizon. Hence, it becomes crucial to acknowledge that not all decision steps are equally important to the final outcome, and to budget the number of counterfactual decisions a policy make in order to control the extrapolation. Contrary to existing approaches that use regularization on either the policy or value function, we propose an approach to explicitly bound the amount of out-of-distribution actions during training. Specifically, our method utilizes dynamic programming to decide where to extrapolate and where not to, with an upper bound on the decisions different from behavior policy. It balances between the potential for improvement from taking out-of-distribution actions and the risk of making errors due to extrapolation. Theoretically, we justify our method by the constrained optimality of the fixed point solution to our $Q$ updating rules. Empirically, we show that the overall performance of our method is better than the state-of-the-art offline RL methods on tasks in the widely-used D4RL benchmarks.</details>

### CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society

**Authors:** Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, Bernard Ghanem

**<details><summary>Abstract**</summary>

The rapid advancement of chat-based language models has led to remarkable progress in complex task-solving. However, their success heavily relies on human input to guide the conversation, which can be challenging and time-consuming. This paper explores the potential of building scalable techniques to facilitate autonomous cooperation among communicative agents, and provides insight into their “cognitive” processes. To address the challenges of achieving autonomous cooperation, we propose a novel communicative agent framework named role-playing . Our approach involves using inception prompting to guide chat agents toward task completion while maintaining consistency with human intentions. We showcase how role-playing can be used to generate conversational data for studying the behaviors and capabilities of a society of agents, providing a valuable resource for investigating conversational language models. In particular, we conduct comprehensive studies on instruction-following cooperation in multi-agent settings. Our contributions include introducing a novel communicative agent framework, offering a scalable approach for studying the cooperative behaviors and capabilities of multi-agent systems, and open-sourcing our library to support research on communicative agents and beyond: https://github.com/camel-ai/camel.</details>

### CAPro: Webly Supervised Learning with Cross-modality Aligned Prototypes

**Authors:** Yulei Qin, Xingyu Chen, Yunhang Shen, Chaoyou Fu, Yun Gu, Ke Li, Xing Sun, Rongrong Ji

**<details><summary>Abstract**</summary>

Webly supervised learning has attracted increasing attention for its effectiveness in exploring publicly accessible data at scale without manual annotation. However, most existing methods of learning with web datasets are faced with challenges from label noise, and they have limited assumptions on clean samples under various noise. For instance, web images retrieved with queries of ”tiger cat“ (a cat species) and ”drumstick“ (a musical instrument) are almost dominated by images of tigers and chickens, which exacerbates the challenge of fine-grained visual concept learning. In this case, exploiting both web images and their associated texts is a requisite solution to combat real-world noise. In this paper, we propose Cross-modality Aligned Prototypes (CAPro), a unified prototypical contrastive learning framework to learn visual representations with correct semantics. For one thing, we leverage textual prototypes, which stem from the distinct concept definition of classes, to select clean images by text matching and thus disambiguate the formation of visual prototypes. For another, to handle missing and mismatched noisy texts, we resort to the visual feature space to complete and enhance individual texts and thereafter improve text matching. Such semantically aligned visual prototypes are further polished up with high-quality samples, and engaged in both cluster regularization and noise removal. Besides, we propose collective bootstrapping to encourage smoother and wiser label reference from appearance-similar instances in a manner of dictionary look-up. Extensive experiments on WebVision1k and NUS-WIDE (Web) demonstrate that CAPro well handles realistic noise under both single-label and multi-label scenarios. CAPro achieves new state-of-the-art performance and exhibits robustness to open-set recognition. Codes are available at https://github.com/yuleiqin/capro.</details>

### CRoSS: Diffusion Model Makes Controllable, Robust and Secure Image Steganography

**Authors:** Jiwen Yu, Xuanyu Zhang, Youmin Xu, Jian Zhang

**<details><summary>Abstract**</summary>

Current image steganography techniques are mainly focused on cover-based methods, which commonly have the risk of leaking secret images and poor robustness against degraded container images. Inspired by recent developments in diffusion models, we discovered that two properties of diffusion models, the ability to achieve translation between two images without training, and robustness to noisy data, can be used to improve security and natural robustness in image steganography tasks. For the choice of diffusion model, we selected Stable Diffusion, a type of conditional diffusion model, and fully utilized the latest tools from open-source communities, such as LoRAs and ControlNets, to improve the controllability and diversity of container images. In summary, we propose a novel image steganography framework, named Controllable, Robust and Secure Image Steganography (CRoSS), which has significant advantages in controllability, robustness, and security compared to cover-based image steganography methods. These benefits are obtained without additional training. To our knowledge, this is the first work to introduce diffusion models to the field of image steganography. In the experimental section, we conducted detailed experiments to demonstrate the advantages of our proposed CRoSS framework in controllability, robustness, and security.</details>

### Cal-DETR: Calibrated Detection Transformer

**Authors:** Muhammad Akhtar Munir, Salman Khan, Muhammad Haris Khan, Mohsen Ali, Fahad Shahbaz Khan

**<details><summary>Abstract**</summary>

Albeit revealing impressive predictive performance for several computer vision tasks, deep neural networks (DNNs) are prone to making overconfident predictions. This limits the adoption and wider utilization of DNNs in many safety-critical applications. There have been recent efforts toward calibrating DNNs, however, almost all of them focus on the classification task. Surprisingly, very little attention has been devoted to calibrating modern DNN-based object detectors, especially detection transformers, which have recently demonstrated promising detection performance and are influential in many decision-making systems. In this work, we address the problem by proposing a mechanism for calibrated detection transformers (Cal-DETR), particularly for Deformable-DETR, UP-DETR, and DINO. We pursue the train-time calibration route and make the following contributions. First, we propose a simple yet effective approach for quantifying uncertainty in transformer-based object detectors. Second, we develop an uncertainty-guided logit modulation mechanism that leverages the uncertainty to modulate the class logits. Third, we develop a logit mixing approach that acts as a regularizer with detection-specific losses and is also complementary to the uncertainty-guided logit modulation technique to further improve the calibration performance. Lastly, we conduct extensive experiments across three in-domain and four out-domain scenarios. Results corroborate the effectiveness of Cal-DETR against the competing train-time methods in calibrating both in-domain and out-domain detections while maintaining or even improving the detection performance. Our codebase and pre-trained models can be accessed at \url{https://github.com/akhtarvision/cal-detr}.</details>

### Calibrate and Boost Logical Expressiveness of GNN Over Multi-Relational and Temporal Graphs

**Authors:** Yeyuan Chen, Dingmin Wang

**<details><summary>Abstract**</summary>

As a powerful framework for graph representation learning, Graph Neural Networks (GNNs) have garnered significant attention in recent years. However, to the best of our knowledge, there has been no formal analysis of the logical expressiveness of GNNs as Boolean node classifiers over multi-relational graphs, where each edge carries a specific relation type. In this paper, we investigate $\mathcal{FOC}_2$, a fragment of first-order logic with two variables and counting quantifiers. On the negative side, we demonstrate that the R$^2$-GNN architecture, which extends the local message passing GNN by incorporating global readout, fails to capture $\mathcal{FOC}_2$ classifiers in the general case. Nevertheless, on the positive side, we establish that R$^2$-GNNs models are equivalent to $\mathcal{FOC}_2$ classifiers under certain restricted yet reasonable scenarios. To address the limitations of R$^2$-GNNs regarding expressiveness, we propose a simple graph transformation technique, akin to a preprocessing step, which can be executed in linear time. This transformation enables R$^2$-GNNs to effectively capture any $\mathcal{FOC}_2$ classifiers when applied to the "transformed" input graph. Moreover, we extend our analysis of expressiveness and graph transformation to temporal graphs, exploring several temporal GNN architectures and providing an expressiveness hierarchy for them. To validate our findings, we implement R$^2$-GNNs and the graph transformation technique and conduct empirical tests in node classification tasks against various well-known GNN architectures that support multi-relational or temporal graphs. Our experimental results consistently demonstrate that R$^2$-GNN with the graph transformation outperforms the baseline methods on both synthetic and real-world datasets</details>

### CamoPatch: An Evolutionary Strategy for Generating Camoflauged Adversarial Patches

**Authors:** Phoenix Williams, Ke Li

**<details><summary>Abstract**</summary>

Deep neural networks (DNNs) have demonstrated vulnerabilities to adversarial examples, which raises concerns about their reliability in safety-critical applications. While the majority of existing methods generate adversarial examples by making small modifications to the entire image, recent research has proposed a practical alternative known as adversarial patches. Adversarial patches have shown to be highly effective in causing DNNs to misclassify by distorting a localized area (patch) of the image. However, existing methods often produce clearly visible distortions since they do not consider the visibility of the patch. To address this, we propose a novel method for constructing adversarial patches that approximates the appearance of the area it covers. We achieve this by using a set of semi-transparent, RGB-valued circles, drawing inspiration from the computational art community. We utilize an evolutionary strategy to optimize the properties of each shape, and employ a simulated annealing approach to optimize the patch's location. Our approach achieves better or comparable performance to state-of-the-art methods on ImageNet DNN classifiers while achieving a lower $l_2$ distance from the original image. By minimizing the visibility of the patch, this work further highlights the vulnerabilities of DNNs to adversarial patches.</details>

### Cascading Contextual Assortment Bandits

**Authors:** Hyun-jun Choi, Rajan Udwani, Min-hwan Oh

**<details><summary>Abstract**</summary>

We present a new combinatorial bandit model, the \textit{cascading contextual assortment bandit}. This model serves as a generalization of both existing cascading bandits and assortment bandits, broadening their applicability in practice. For this model, we propose our first UCB bandit algorithm, UCB-CCA. We prove that this algorithm achieves a $T$-step regret upper-bound of $\tilde{\mathcal{O}}(\frac{1}{\kappa}d\sqrt{T})$, sharper than existing bounds for cascading contextual bandits by eliminating dependence on cascade length $K$. To improve the dependence on problem-dependent constant $\kappa$, we introduce our second algorithm, UCB-CCA+, which leverages a new Bernstein-type concentration result. This algorithm achieves $\tilde{\mathcal{O}}(d\sqrt{T})$ without dependence on $\kappa$ in the leading term. We substantiate our theoretical claims with numerical experiments, demonstrating the practical efficacy of our proposed methods.</details>

### Causal Fairness for Outcome Control

**Authors:** Drago Plecko, Elias Bareinboim

**<details><summary>Abstract**</summary>

As society transitions towards an AI-based decision-making infrastructure, an ever-increasing number of decisions once under control of humans are now delegated to automated systems. Even though such developments make various parts of society more efficient, a large body of evidence suggests that a great deal of care needs to be taken to make such automated decision-making systems fair and equitable, namely, taking into account sensitive attributes such as gender, race, and religion. In this paper, we study a specific decision-making task called outcome control in which an automated system aims to optimize an outcome variable $Y$ while being fair and equitable. The interest in such a setting ranges from interventions related to criminal justice and welfare, all the way to clinical decision-making and public health. In this paper, we first analyze through causal lenses the notion of benefit, which captures how much a specific individual would benefit from a positive decision, counterfactually speaking, when contrasted with an alternative, negative one. We introduce the notion of benefit fairness, which can be seen as the minimal fairness requirement in decision-making, and develop an algorithm for satisfying it. We then note that the benefit itself may be influenced by the protected attribute, and propose causal tools which can be used to analyze this. Finally, if some of the variations of the protected attribute in the benefit are considered as discriminatory, the notion of benefit fairness may need to be strengthened, which leads us to articulating a notion of causal benefit fairness. Using this notion, we develop a new optimization procedure capable of maximizing $Y$ while ascertaining causal fairness in the decision process.</details>

### Causal discovery from observational and interventional data across multiple environments

**Authors:** Adam Li, Amin Jaber, Elias Bareinboim

**<details><summary>Abstract**</summary>

A fundamental problem in many sciences is the learning of causal structure underlying a system, typically through observation and experimentation. Commonly, one even collects data across multiple domains, such as gene sequencing from different labs, or neural recordings from different species. Although there exist methods for learning the equivalence class of causal diagrams from observational and experimental data, they are meant to operate in a single domain. In this paper, we develop a fundamental approach to structure learning in non-Markovian systems (i.e. when there exist latent confounders) leveraging observational and interventional data collected from multiple domains. Specifically, we start by showing that learning from observational data in multiple domains is equivalent to learning from interventional data with unknown targets in a single domain. But there are also subtleties when considering observational and experimental data. Using causal invariances derived from do-calculus, we define a property called S-Markov that connects interventional distributions from multiple-domains to graphical criteria on a selection diagram. Leveraging the S-Markov property, we introduce a new constraint-based causal discovery algorithm, S-FCI, that can learn from observational and interventional data from different domains. We prove that the algorithm is sound and subsumes existing constraint-based causal discovery algorithms.</details>

### [Oral] Causal normalizing flows: from theory to practice

**Authors:** Adrián Javaloy, Pablo Sanchez-Martin, Isabel Valera

**Oral Presentation:** Tu, Dec 12, 14:25 -- Oral 2C

**<details><summary>Abstract**</summary>

In this work, we deepen on the use of normalizing flows for causal reasoning. Specifically, we first leverage recent results on non-linear ICA to show that causal models are identifiable from observational data given a causal ordering, and thus can be recovered using autoregressive normalizing flows (NFs). Second, we analyze different design and learning choices for *causal normalizing flows* to capture the underlying causal data-generating process. Third, we describe how to implement the *do-operator* in causal NFs, and thus, how to answer interventional and counterfactual questions. Finally, in our experiments, we validate our design and training choices through a comprehensive ablation study; compare causal NFs to other approaches for approximating causal models; and empirically demonstrate that causal NFs can be used to address real-world problems—where the presence of mixed discrete-continuous data and partial knowledge on the causal graph is the norm. The code for this work can be found at https://github.com/psanch21/causal-flows.</details>

### Causes and Effects of Unanticipated Numerical Deviations in Neural Network Inference Frameworks

**Authors:** Alex Schlögl, Nora Hofer, Rainer Böhme

**<details><summary>Abstract**</summary>

Hardware-specific optimizations in machine learning (ML) frameworks can cause numerical deviations of inference results. Quite surprisingly, despite using a fixed trained model and fixed input data, inference results are not consistent across platforms, and sometimes not even deterministic on the same platform. We study the causes of these numerical deviations for convolutional neural networks (CNN) onrealistic end-to-end inference pipelines and in isolated experiments. Results from 75 distinct platforms suggest that the main causes of deviations on CPUs are differences in SIMD use, and the selection of convolution algorithms at runtime on GPUs. We link the causes and propagation effects to properties of the ML model and evaluate potential mitigations. We make our research code publicly available.</details>

### Certified Minimax Unlearning with Generalization Rates and Deletion Capacity

**Authors:** Jiaqi Liu, Jian Lou, Zhan Qin, Kui Ren

**<details><summary>Abstract**</summary>

We study the problem of $(\epsilon,\delta)$-certified machine unlearning for minimax models. Most of the existing works focus on unlearning from standard statistical learning models that have a single variable and their unlearning steps hinge on the \emph{direct Hessian-based conventional Newton} update. We develop a new $(\epsilon,\delta)$-certified machine unlearning algorithm for minimax models. It proposes a minimax unlearning step consisting of a \emph{total-Hessian-based complete Newton} update and the Gaussian mechanism borrowed from differential privacy. To obtain the unlearning certification, our method injects calibrated Gaussian noises by carefully analyzing the ``sensitivity'' of the minimax unlearning step (i.e., the closeness between the minimax unlearning variables and the retraining-from-scratch variables). We derive the generalization rates in terms of population strong and weak primal-dual risk for three different cases of loss functions, i.e., (strongly-)convex-(strongly-)concave losses. We also provide the deletion capacity to guarantee that a desired population risk can be maintained as long as the number of deleted samples does not exceed the derived amount. With training samples $n$ and model dimension $d$, it yields the order $\mathcal O(n/d^{1/4})$, which shows a strict gap over the baseline method of differentially private minimax learning that has $\mathcal O(n/d^{1/2})$. In addition, our rates of generalization and deletion capacity match the state-of-the-art rates derived previously for standard statistical learning models.</details>

### Change point detection and inference in multivariate non-parametric models under mixing conditions

**Authors:** Carlos Misael Madrid Padilla, Haotian Xu, Daren Wang, OSCAR HERNAN MADRID PADILLA, Yi Yu

**<details><summary>Abstract**</summary>

This paper addresses the problem of localizing and inferring multiple change points, in non-parametric multivariate time series settings. Specifically, we consider a multivariate time series with potentially short-range dependence, whose underlying distributions have Hölder smooth densities and can change over time in a piecewise-constant manner. The change points, which correspond to the times when the distribution changes, are unknown. We present the limiting distributions of the change point estimators under the scenarios where the minimal jump size vanishes or remains constant.  Such results have not been revealed in the literature in non-parametric change point settings. As byproducts, we develop a sharp estimator that can accurately localize the change points in multivariate non-parametric time series, and a consistent block-type long-run variance estimator.  Numerical studies are provided to complement our theoretical findings.</details>

### [Spotlight] Characterizing the Optimal $0-1$ Loss for Multi-class Classification with a Test-time Attacker

**Authors:** Sihui Dai, Wenxin Ding, Arjun Nitin Bhagoji, Daniel Cullina, Heather Zheng, Ben Zhao, Prateek Mittal

**<details><summary>Abstract**</summary>

Finding classifiers robust to adversarial examples is critical for their safedeployment. Determining the robustness of the best possible classifier under agiven threat model for a fixed data distribution and comparing it to thatachieved by state-of-the-art training methods is thus an important diagnostictool. In this paper, we find achievable information-theoretic lower bounds onrobust loss in the presence of a test-time attacker for *multi-classclassifiers on any discrete dataset*. We provide a general framework for findingthe optimal $0-1$ loss that revolves around the construction of a conflicthypergraph from the data and adversarial constraints. The prohibitive cost ofthis formulation in practice leads us to formulate other variants of the attacker-classifiergame that more efficiently determine the range of the optimal loss. Ourvaluation shows, for the first time, an analysis of the gap to optimalrobustness for classifiers in the multi-class setting on benchmark datasets.</details>

### Closing the Computational-Statistical Gap in Best Arm Identification for Combinatorial Semi-bandits

**Authors:** Ruo-Chun Tzeng, Po-An Wang, Alexandre Proutiere, Chi-Jen Lu

**<details><summary>Abstract**</summary>

We study the best arm identification problem in combinatorial semi-bandits in the fixed confidence setting. We present Perturbed Frank-Wolfe Sampling (P-FWS), an algorithm that (i) runs in polynomial time, (ii) achieves the instance-specific minimal sample complexity in the high confidence regime, and (iii) enjoys polynomial sample complexity guarantees in the moderate confidence regime. To our best knowledge, existing algorithms cannot achieve (ii) and (iii) simultaneously in vanilla bandits. With P-FWS, we close the computational-statistical gap in best arm identification in combinatorial semi-bandits. The design of P-FWS starts from the optimization problem that defines the information-theoretical and instance-specific sample complexity lower bound. P-FWS solves this problem in an online manner using, in each round, a single iteration of the Frank-Wolfe algorithm. Structural properties of the problem are leveraged to make the P-FWS successive updates computationally efficient. In turn, P-FWS only relies on a simple linear maximization oracle.</details>

### Clustering the Sketch: Dynamic Compression for Embedding Tables

**Authors:** Henry Tsang, Thomas Ahle

**<details><summary>Abstract**</summary>

Embedding tables are used by machine learning systems  to work with categorical features.In modern Recommendation Systems, these tables can be very large, necessitating the development of new methods for fitting them in memory, even during training.We suggest Clustered Compositional Embeddings (CCE) which combines clustering-based compression like quantization to codebooks with dynamic methods like The Hashing Trick and Compositional Embeddings [Shi et al., 2020].Experimentally CCE achieves the best of both worlds: The high compression rate of codebook-based quantization, but \emph{dynamically} like hashing-based methods, so it can be used during training.Theoretically, we prove that CCE is guaranteed to converge to the optimal codebook and give a tight bound for the number of iterations required.</details>

### CoLLAT: On Adding Fine-grained Audio Understanding to Language Models using Token-Level Locked-Language Tuning

**Authors:** Dadallage A R Silva, Spencer Whitehead, Christopher Lengerich, Hugh Leather

**<details><summary>Abstract**</summary>

Humans can easily understand various audio concepts, but conventional audio classification models fail due to their inability to predict unseen classes during training. To address this challenge, recent literature has explored contrastive language-audio pretraining to learn an audio understanding model using natural language supervision from a pretrained language model. However, despite their reasonable zero-shot performance in audio understanding, these models typically fail to achieve optimal performance while preserving the text understanding capabilities of the pretrained language model. They also perform poorly when comprehending audio clips with multiple audio concepts. To bridge these gaps, we propose $CoLLAT$: $Co$ntrastive $L$ocked $L$anguage and $A$udio $T$uning. This is a framework to effectively learn an audio understanding model with a locked language model, which is learned using a novel pretraining objective for audio-to-text grounding to yield fine-grained audio understanding. Our extensive experiments, which include several downstream applications such as audio classification, cross-modal retrieval, and audio-guided image generation, demonstrate that $CoLLAT$ yields state-of-the-art performance for audio understanding. Additionally, it unlocks audio guidance to applications built on top of pretrained language models.</details>

### Collaborative Score Distillation for Consistent Visual Editing

**Authors:** Subin Kim, Kyungmin Lee, June Suk Choi, Jongheon Jeong, Kihyuk Sohn, Jinwoo Shin

**<details><summary>Abstract**</summary>

Generative priors of large-scale text-to-image diffusion models enable a wide range of new generation and editing applications on diverse visual modalities. However, when adapting these priors to complex visual modalities, often represented as multiple images (e.g., video or 3D scene), achieving consistency across a set of images is challenging. In this paper, we address this challenge with a novel method, Collaborative Score Distillation (CSD). CSD is based on the Stein Variational Gradient Descent (SVGD). Specifically, we propose to consider multiple samples as “particles” in the SVGD update and combine their score functions to distill generative priors over a set of images synchronously. Thus, CSD facilitates the seamless integration of information across 2D images, leading to a consistent visual synthesis across multiple samples. We show the effectiveness of CSD in a variety of editing tasks, encompassing the visual editing of panorama images, videos, and 3D scenes. Our results underline the competency of CSD as a versatile method for enhancing inter-sample consistency, thereby broadening the applicability of text-to-image diffusion models.</details>

### ComSL: A Composite Speech-Language Model for End-to-End Speech-to-Text Translation

**Authors:** Chenyang Le, Yao Qian, Long Zhou, Shujie LIU, Yanmin Qian, Michael Zeng, Xuedong Huang

**<details><summary>Abstract**</summary>

Joint speech-language training is challenging due to the large demand for training data and GPU consumption, as well as the modality gap between speech and language. We present ComSL, a speech-language model built atop a composite architecture of public pre-trained speech-only and language-only models and optimized data-efficiently for spoken language tasks. Particularly, we propose to incorporate cross-modality learning into transfer learning and conduct them simultaneously for downstream tasks in a multi-task learning manner. Our approach has demonstrated effectiveness in end-to-end speech-to-text translation tasks, achieving a new state-of-the-art average BLEU score of 31.5 on the multilingual speech to English text translation task for 21 languages, as measured on the public CoVoST2 evaluation set.</details>

### Combinatorial Group Testing with Selfish Agents

**Authors:** Georgios Chionas, Dariusz Kowalski, Piotr Krysta

**<details><summary>Abstract**</summary>

We study the Combinatorial Group Testing (CGT) problem in a novel game-theoretic framework, with a solution concept of Adversarial Equilibrium (AE). In this new framework, we have $n$ selfish agents corresponding to the elements of the universe $[n] =\{0,1,\ldots,n-1\}$ and a hidden set $K \subseteq [n]$ of active agents of size $|K| = k \ll n$. In each round of the game, each active agent decides if it is present in a query $Q \subseteq [n]$, and all agents receive feedback on $Q \cap K$. The goal of each active agent is to assure that its id could be learned from the feedback as early as possible. We present a comprehensive set of results in this new game, where we design and analyze adaptive algorithmic strategies of agents which are AE's. In particular, if $k$ is known to the agents, then we design adaptive AE strategies with provably near optimal learning time of $O(k \log(n/k))$. In the case of unknown $k$, we design an adaptive AE strategies with learning time of order $n^k$, and we prove a lower bound of $\Omega(n)$ on the learning time of any such algorithmic strategies. This shows a strong separations between the two models of known and unknown $k$, as well as between the classic CGT, i.e., without selfish agents, and our game theoretic CGT model.</details>

### [Spotlight] Common Ground in Cooperative Communication

**Authors:** Xiaoran Hao, Yash Jhaveri, Patrick Shafto

**<details><summary>Abstract**</summary>

Cooperative communication plays a fundamental role in theories of human-human interaction--cognition, culture, development, language, etc.--as well as human-robot interaction. The core challenge in cooperative communication is the problem of common ground: having enough shared knowledge and understanding to successfully communicate. Prior models of cooperative communication, however, uniformly assume the strongest form of common ground, perfect and complete knowledge sharing, and, therefore, fail to capture the core challenge of cooperative communication. We propose a general theory of cooperative communication that is mathematically principled and explicitly defines a spectrum of common ground possibilities, going well beyond that of perfect and complete knowledge sharing, on spaces that permit arbitrary representations of data and hypotheses. Our framework is a strict generalization of prior models of cooperative communication. After considering a parametric form of common ground and viewing the data selection and hypothesis inference processes of communication as encoding and decoding, we establish a connection to variational autoencoding, a powerful model in modern machine learning. Finally, we carry out a series of empirical simulations to support and elaborate on our theoretical results.</details>

### CommonScenes: Generating Commonsense 3D Indoor Scenes with Scene Graphs

**Authors:** Guangyao Zhai, Evin Pınar Örnek, Shun-Cheng Wu, Yan Di, Federico Tombari, Nassir Navab, Benjamin Busam

**<details><summary>Abstract**</summary>

Controllable scene synthesis aims to create interactive environments for various industrial use cases. Scene graphs provide a highly suitable interface to facilitate these applications by abstracting the scene context in a compact manner. Existing methods, reliant on retrieval from extensive databases or pre-trained shape embeddings, often overlook scene-object and object-object relationships, leading to inconsistent results due to their limited generation capacity. To address this issue, we present CommonScenes, a fully generative model that converts scene graphs into corresponding controllable 3D scenes, which are semantically realistic and conform to commonsense. Our pipeline consists of two branches, one predicting the overall scene layout via a variational auto-encoder and the other generating compatible shapes via latent diffusion, capturing global scene-object and local inter-object relationships while preserving shape diversity. The generated scenes can be manipulated by editing the input scene graph and sampling the noise in the diffusion model. Due to lacking a scene graph dataset offering high-quality object-level meshes with relations, we also construct SG-FRONT, enriching the off-the-shelf indoor dataset 3D-FRONT with additional scene graph labels. Extensive experiments are conducted on SG-FRONT where CommonScenes shows clear advantages over other methods regarding generation consistency, quality, and diversity. Codes and the dataset will be released upon acceptance.</details>

### Communication-Efficient Federated Bilevel Optimization with Global and Local Lower Level Problems

**Authors:** Junyi Li, Feihu Huang, Heng Huang

**<details><summary>Abstract**</summary>

Bilevel Optimization has witnessed notable progress recently with new emerging efficient algorithms. However, its application in the Federated Learning setting remains relatively underexplored, and the impact of Federated Learning's inherent challenges on the convergence of bilevel algorithms remain obscure.In this work, we investigate Federated Bilevel Optimization problems and propose a communication-efficient algorithm, named FedBiOAcc. The algorithm leverages an efficient estimation of the hyper-gradient in the distributed setting and utilizes the momentum-based variance-reduction acceleration. Remarkably, FedBiOAcc achieves a communication complexity $O(\epsilon^{-1})$, a sample complexity $O(\epsilon^{-1.5})$ and the linear speed up with respect to the number of clients. We also analyze a special case of the Federated Bilevel Optimization problems, where lower level problems are locally managed by clients. We prove that FedBiOAcc-Local, a modified version of FedBiOAcc, converges at the same rate for this type of problems. Finally, we validate the proposed algorithms through two real-world tasks: Federated Data-cleaning and Federated Hyper-representation Learning. Empirical results show superior performance of our algorithms.</details>

### Comparing Causal Frameworks: Potential Outcomes, Structural Models, Graphs, and Abstractions

**Authors:** Duligur Ibeling, Thomas Icard

**<details><summary>Abstract**</summary>

The aim of this paper is to make clear and precise the relationship between the Rubin causal model (RCM) and structural causal model (SCM) frameworks for causal inference. Adopting a neutral logical perspective, and drawing on previous work, we show what is required for an RCM to be representable by an SCM. A key result then shows that every RCM---including those that violate algebraic principles implied by the SCM framework---emerges as an abstraction of some representable RCM. Finally, we illustrate the power of this ameliorative perspective by pinpointing an important role for SCM principles in classic applications of RCMs; conversely, we offer a characterization of the algebraic constraints implied by a graph, helping to substantiate further comparisons between the two frameworks.</details>

### [Spotlight] Complexity Matters: Rethinking the Latent Space for Generative Modeling

**Authors:** Tianyang Hu, Fei Chen, Haonan Wang, Jiawei Li, Wenjia Wang, Jiacheng Sun, Zhenguo Li

**<details><summary>Abstract**</summary>

In generative modeling, numerous successful approaches leverage a low-dimensional latent space, e.g., Stable Diffusion models the latent space induced by an encoder and generates images through a paired decoder. Although the selection of the latent space is empirically pivotal, determining the optimal choice and the process of identifying it remain unclear. In this study, we aim to shed light on this under-explored topic by rethinking the latent space from the perspective of model complexity. Our investigation starts with the classic generative adversarial networks (GANs). Inspired by the GAN training objective, we propose a novel "distance" between the latent and data distributions, whose minimization coincides with that of the generator complexity. The minimizer of this distance is characterized as the optimal data-dependent latent that most effectively capitalizes on the generator's capacity. Then, we consider parameterizing such a latent distribution by an encoder network and propose a two-stage training strategy called Decoupled Autoencoder (DAE), where the encoder is only updated in the first stage with an auxiliary decoder and then frozen in the second stage while the actual decoder is being trained. DAE can improve the latent distribution and as a result, improve the generative performance. Our theoretical analyses are corroborated by comprehensive experiments on various models such as VQGAN and Diffusion Transformer, where our modifications yield significant improvements in sample quality with decreased model complexity.</details>

### Composing Parameter-Efficient Modules with Arithmetic Operation

**Authors:** Jinghan Zhang, shiqi chen, Junteng Liu, Junxian He

**<details><summary>Abstract**</summary>

As an efficient alternative to conventional full fine-tuning, parameter-efficient fine-tuning (PEFT) is becoming the prevailing method to adapt pretrained language models. In PEFT, a lightweight module is learned on each dataset while the underlying pretrained language model remains unchanged, resulting in multiple compact modules representing diverse skills when applied to various domains and tasks. In this paper, we propose to compose these parameter-efficient modules through linear arithmetic operations in the weight space, thereby integrating different module capabilities. Specifically, we first define an addition and negation operator for the module, and then further compose these two basic operators to perform flexible arithmetic. Our approach requires no additional training and enables highly flexible module composition.  We apply different arithmetic operations to compose the parameter-efficient modules for (1) distribution generalization, (2) multi-tasking, (3) detoxifying, and (4) domain transfer. Additionally, we extend our approach to detoxify Alpaca-LoRA, the latest instruction-tuned large language model based on LLaMA. Empirical results demonstrate that our approach produces new and effective parameter-efficient modules that significantly outperform existing ones across all settings.</details>

### Compressed Video Prompt Tuning

**Authors:** Bing Li, Jiaxin Chen, Xiuguo Bao, Di Huang

**<details><summary>Abstract**</summary>

Compressed videos offer a compelling alternative to raw videos, showing the possibility to significantly reduce the on-line computational and storage cost. However, current approaches to compressed video processing generally follow the resource-consuming pre-training and fine-tuning paradigm, which does not fully take advantage of such properties, making them not favorable enough for widespread applications. Inspired by recent successes of prompt tuning techniques in computer vision, this paper presents the first attempt to build a prompt based representation learning framework, which enables effective and efficient adaptation of pre-trained raw video models to compressed video understanding tasks. To this end, we propose a novel prompt tuning approach, namely Compressed Video Prompt Tuning (CVPT), emphatically dealing with the challenging issue caused by the inconsistency between pre-training and downstream data modalities. Specifically, CVPT replaces the learnable prompts with compressed modalities (\emph{e.g.} Motion Vectors and Residuals) by re-parameterizing them into conditional prompts followed by layer-wise refinement. The conditional prompts exhibit improved adaptability and generalizability to instances compared to conventional individual learnable ones, and the Residual prompts enhance the noisy motion cues in the Motion Vector prompts for further fusion with the visual cues from I-frames. Additionally, we design Selective Cross-modal Complementary Prompt (SCCP) blocks. After inserting them into the backbone, SCCP blocks leverage semantic relations across diverse levels and modalities to improve cross-modal interactions between prompts and input flows. Extensive evaluations on HMDB-51, UCF-101 and Something-Something v2 demonstrate that CVPT remarkably outperforms the state-of-the-art counterparts, delivering a much better balance between accuracy and efficiency.</details>

### [Spotlight] Compression with Bayesian Implicit Neural Representations

**Authors:** Zongyu Guo, Gergely Flamich, Jiajun He, Zhibo Chen, José Miguel Hernández-Lobato

**<details><summary>Abstract**</summary>

Many common types of data can be represented as functions that map coordinates to signal values, such as pixel locations to RGB values in the case of an image. Based on this view, data can be compressed by overfitting a compact neural network to its functional representation and then encoding the network weights. However, most current solutions for this are inefficient, as quantization to low-bit precision substantially degrades the reconstruction quality. To address this issue, we propose overfitting variational Bayesian neural networks to the data and compressing an approximate posterior weight sample using relative entropy coding instead of quantizing and entropy coding it. This strategy enables direct optimization of the rate-distortion performance by minimizing the $\beta$-ELBO, and target different rate-distortion trade-offs for a given network architecture by adjusting $\beta$. Moreover, we introduce an iterative algorithm for learning prior weight distributions and employ a progressive refinement process for the variational posterior that significantly enhances performance. Experiments show that our method achieves strong performance on image and audio compression while retaining simplicity.</details>

### Computational Complexity of Learning Neural Networks: Smoothness and Degeneracy

**Authors:** Amit Daniely, Nati Srebro, Gal Vardi

**<details><summary>Abstract**</summary>

Understanding when neural networks can be learned efficientlyis a fundamental question in learning theory.Existing hardness results suggest that assumptions on both the input distribution and the network's weights are necessary for obtaining efficient algorithms. Moreover, it was previously shown that depth-$2$ networks can be efficiently learned under the assumptions that the input distribution is Gaussian, and the weight matrix is non-degenerate. In this work, we study whether such assumptions may suffice for learning deeper networks and prove negative results. We show that learning depth-$3$ ReLU networks under the Gaussian input distribution is hard even in the smoothed-analysis framework, where a random noise is added to the network's parameters. It implies that learning depth-$3$ ReLU networks under the Gaussian distribution is hard even if the weight matrices are non-degenerate. Moreover, we consider depth-$2$ networks, and show hardness of learning in the smoothed-analysis framework, where both the network parameters and the input distribution are smoothed. Our hardness results are under a well-studied assumption on the existence of local pseudorandom generators.</details>

### [Spotlight] Computing a human-like reaction time metric from stable recurrent vision models

**Authors:** Lore Goetschalckx, Alekh Karkada Ashok, Aarit Ahuja, David Sheinberg, Thomas Serre

**<details><summary>Abstract**</summary>

The meteoric rise in the adoption of deep neural networks as computational models of vision has inspired efforts to ``align” these models with humans. One dimension of interest for alignment includes behavioral choices, but moving beyond characterizing choice patterns to capturing temporal aspects of visual decision-making has been challenging. Here, we sketch a general-purpose methodology to construct computational accounts of reaction times from a stimulus-computable, task-optimized model. Specifically, we introduce a novel metric leveraging insights from subjective logic theory summarizing evidence accumulation in recurrent vision models. We demonstrate that our metric aligns with patterns of human reaction times for stimulus manipulations across four disparate visual decision-making tasks spanning perceptual grouping, mental simulation, and scene categorization. This work paves the way for exploring the temporal alignment of model and human visual strategies in the context of various other cognitive tasks toward generating testable hypotheses for neuroscience. Links to the code and data can be found on the project page: https://serre-lab.github.io/rnn_rts_site/.</details>

### Coneheads: Hierarchy Aware Attention

**Authors:** Albert Tseng, Tao Yu, Toni Liu, Christopher De Sa

**<details><summary>Abstract**</summary>

Attention networks such as transformers have achieved state-of-the-art performance in many domains. These networks rely heavily on the dot product attention operator, which computes the similarity between two points by taking their inner product.However, the inner product does not explicitly model the complex structural properties of real world datasets, such as hierarchies between data points.To remedy this, we introduce cone attention, a drop-in replacement for dot product attention based on hyperbolic entailment cones.Cone attention associates two points by the depth of their lowest common ancestor in a hierarchy defined by hyperbolic cones, which intuitively measures the divergence of two points and gives a $\textit{hierarchy aware}$ similarity score.We test cone attention on a wide variety of models and tasks and show that it improves task-level performance over dot product attention and other baselines, and is able to match dot-product attention with significantly fewer parameters.Our results suggest that cone attention is an effective way to capture hierarchical relationships when calculating attention.</details>

### [Oral] Conformal Meta-learners for Predictive Inference of Individual Treatment Effects

**Authors:** Ahmed Alaa, Zaid Ahmad, Mark van der Laan

**Oral Presentation:** Tu, Dec 12, 14:10 -- Oral 2C

**<details><summary>Abstract**</summary>

We investigate the problem of machine learning-based (ML) predictive inference on individual treatment effects (ITEs). Previous work has focused primarily on developing ML-based “meta-learners” that can provide point estimates of the conditional average treatment effect (CATE)—these are model-agnostic approaches for combining intermediate nuisance estimates to produce estimates of CATE. In this paper, we develop conformal meta-learners, a general framework for issuing predictive intervals for ITEs by applying the standard conformal prediction (CP) procedure on top of CATE meta-learners. We focus on a broad class of meta-learners based on two-stage pseudo-outcome regression and develop a stochastic ordering framework to study their validity. We show that inference with conformal meta-learners is marginally valid if their (pseudo-outcome) conformity scores stochastically dominate “oracle” conformity scores evaluated on the unobserved ITEs. Additionally, we prove that commonly used CATE meta-learners, such as the doubly-robust learner, satisfy a model- and distribution-free stochastic (or convex) dominance condition, making their conformal inferences valid for practically-relevant levels of target coverage. Whereas existing procedures conduct inference on nuisance parameters (i.e., potential outcomes) via weighted CP, conformal meta-learners enable direct inference on the target parameter (ITE). Numerical experiments show that conformal meta-learners provide valid intervals with competitive efficiency while retaining the favorable point estimation properties of CATE meta-learners.</details>

### Conformalized matrix completion

**Authors:** Yu Gui, Rina Barber, Cong Ma

**<details><summary>Abstract**</summary>

Matrix completion aims to estimate missing entries in a data matrix, using the assumption of a low-complexity structure (e.g., low-rankness) so that imputation is possible. While many effective estimation algorithms exist in the literature, uncertainty quantification for this problem has proved to be challenging, and existing methods are extremely sensitive to model misspecification. In this work, we propose a distribution-free method for predictive inference in the matrix completion problem. Our method adapts the framework of conformal prediction, which provides prediction intervals with guaranteed distribution-free validity in the setting of regression, to the problem of matrix completion. Our resulting method, conformalized matrix completion (cmc), offers provable predictive coverage regardless of the accuracy of the low-rank model. Empirical results on simulated and real data demonstrate that cmc is robust to model misspecification while matching the performance of existing model-based methods when the model is correct.</details>

### Conservative State Value Estimation for Offline Reinforcement Learning

**Authors:** Liting Chen, Jie Yan, Zhengdao Shao, Lu Wang, Qingwei Lin, Saravanakumar Rajmohan, Thomas Moscibroda, Dongmei Zhang

**<details><summary>Abstract**</summary>

Offline reinforcement learning faces a significant challenge of value over-estimation due to the distributional drift between the dataset and the current learned policy, leading to learning failure in practice. The common approach is to incorporate a penalty term to reward or value estimation in the Bellman iterations. Meanwhile, to avoid extrapolation on out-of-distribution (OOD) states and actions, existing methods focus on conservative Q-function estimation. In this paper, we propose Conservative State Value Estimation (CSVE), a new approach that learns conservative V-function via directly imposing penalty on OOD states. Compared to prior work, CSVE allows more effective state value estimation with conservative guarantees and further better policy optimization. Further, we apply CSVE and develop a practical actor-critic algorithm in which the critic does the conservative value estimation by additionally sampling and penalizing the states around the dataset, and the actor applies advantage weighted updates extended with state exploration to improve the policy. We evaluate in classic continual control tasks of D4RL, showing that our method performs better than the conservative Q-function learning methods and is strongly competitive among recent SOTA methods.</details>

### [Spotlight] Constant Approximation for Individual Preference Stable Clustering

**Authors:** Anders Aamand, Justin Chen, Allen Liu, Sandeep Silwal, Pattara Sukprasert, Ali Vakilian, Fred Zhang

**<details><summary>Abstract**</summary>

Individual preference (IP) stability, introduced by Ahmadi et al. (ICML 2022), is a natural clustering objective inspired by stability and fairness constraints. A clustering is $\alpha$-IP stable if the average distance of every data point to its own cluster is at most $\alpha$ times the average distance to any other cluster. Unfortunately, determining if a dataset admits a $1$-IP stable clustering is NP-Hard. Moreover, before this work, it was unknown if an $o(n)$-IP stable clustering always exists, as the prior state of the art only guaranteed an $O(n)$-IP stable clustering. We close this gap in understanding and show that an $O(1)$-IP stable clustering always exists for general metrics, and we give an efficient algorithm which outputs such a clustering.  We also introduce generalizations of IP stability beyond average distance and give efficient near optimal algorithms in the cases where we consider the maximum and minimum distances within and between clusters.</details>

### [Spotlight] Convex and Non-convex Optimization Under Generalized Smoothness

**Authors:** Haochuan Li, Jian Qian, Yi Tian, Alexander Rakhlin, Ali Jadbabaie

**<details><summary>Abstract**</summary>

Classical analysis of convex and non-convex optimization methods often requires the Lipschitz continuity of the gradient, which limits the analysis to functions bounded by quadratics. Recent work relaxed this requirement to a non-uniform smoothness condition with the Hessian norm  bounded by an affine function of the gradient norm, and proved convergence in the non-convex setting via gradient clipping, assuming bounded noise. In this paper, we further generalize this non-uniform smoothness condition and develop a simple, yet powerful analysis technique that bounds the gradients along the trajectory, thereby leading to  stronger results for both convex and non-convex optimization problems. In particular, we obtain the classical convergence rates for (stochastic) gradient descent and Nesterov's accelerated gradient method in the convex and/or non-convex setting under this general smoothness condition. The new analysis approach does not require gradient clipping and allows heavy-tailed noise with bounded variance in the stochastic setting.</details>

### Convex-Concave Zero-Sum Stochastic Stackelberg Games

**Authors:** Denizalp Goktas, Arjun Prakash, Amy Greenwald

**<details><summary>Abstract**</summary>

Zero-sum stochastic Stackelberg games can be used to model a large class of problems, ranging from economics to human robot interaction. In this paper, we develop policy gradient methods to solve these games from noisy gradient estimates computed from observed trajectories of play. We prove that our algorithms converge to Stackelberg equilibrium in polynomial time when the games are convex-concave. We also prove that reach-avoid problems are naturally modeled as convex-concave zero-sum stochastic Stackelberg games. Finally, we run experiments which demonstrate that modeling reach-avoid problems as Stackelberg games leads to solutions which are safer, thus less likely to result in collisions, and liver, thus more likely to reach their goals, than alternative solutions, in particular Nash equilibrium.</details>

### Convolution Monge Mapping Normalization for learning on sleep data

**Authors:** Théo Gnassounou, Rémi Flamary, Alexandre Gramfort

**<details><summary>Abstract**</summary>

In many machine learning applications on signals and biomedical data, especially electroencephalogram (EEG), one major challenge is the variability of the data across subjects, sessions, and hardware devices. In this work, we propose a new method called Convolutional Monge Mapping Normalization ($\texttt{CMMN}$), which consists in filtering the signals in order to adapt their power spectrum density (PSD) to a Wasserstein barycenter estimated on training data. $\texttt{CMMN}$ relies on novel closed-form solutions for optimal transport mappings and barycenters and provides individual test time adaptation to new data without needing to retrain a prediction model. Numerical experiments on sleep EEG data show that $\texttt{CMMN}$ leads to significant and consistent performance gains independent from the neural network architecture when adapting between subjects, sessions, and even datasets collected with different hardware. Notably our performance gain is on par with much more numerically intensive Domain Adaptation (DA) methods and can be used in conjunction with those for even better performances.</details>

### Convolutional State Space Models for Long-Range Spatiotemporal Modeling

**Authors:** Jimmy Smith, Shalini De Mello, Jan Kautz, Scott Linderman, Wonmin Byeon

**<details><summary>Abstract**</summary>

Effectively modeling long spatiotemporal sequences is challenging due to the need to model complex spatial correlations and long-range temporal dependencies simultaneously. ConvLSTMs attempt to address this by updating tensor-valued states with recurrent neural networks, but their sequential computation makes them slow to train. In contrast, Transformers can process an entire spatiotemporal sequence, compressed into tokens, in parallel. However, the cost of attention scales quadratically in length, limiting their scalability to longer sequences. Here, we address the challenges of prior methods and introduce convolutional state space models (ConvSSM) that combine the tensor modeling ideas of ConvLSTM with the long sequence modeling approaches of state space methods such as S4 and S5. First, we demonstrate how parallel scans can be applied to convolutional recurrences to achieve subquadratic parallelization and fast autoregressive generation. We then establish an equivalence between the dynamics of ConvSSMs and SSMs, which motivates parameterization and initialization strategies for modeling long-range dependencies. The result is ConvS5, an efficient ConvSSM variant for long-range spatiotemporal modeling. ConvS5 significantly outperforms Transformers and ConvLSTM on a long horizon Moving-MNIST experiment while training $3\times$ faster than ConvLSTM and generating samples $400\times$ faster than Transformers. In addition,  ConvS5 matches or exceeds the performance of state-of-the-art methods on challenging DMLab, Minecraft and Habitat prediction benchmarks and enables new directions for modeling long spatiotemporal sequences.</details>

### Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Convolutional CLIP

**Authors:** Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, Liang-Chieh Chen

**<details><summary>Abstract**</summary>

Open-vocabulary segmentation is a challenging task requiring segmenting and recognizing objects from an open set of categories in diverse environments. One way to address this challenge is to leverage multi-modal models, such as CLIP, to provide image and text features in a shared embedding space, which effectively bridges the gap between closed-vocabulary and open-vocabulary recognition.Hence, existing methods often adopt a two-stage framework to tackle the problem, where the inputs first go through a mask generator and then through the CLIP model along with the predicted masks. This process involves extracting features from raw images multiple times, which can be ineffective and inefficient. By contrast, we propose to build everything into a single-stage framework using a _shared **F**rozen **C**onvolutional **CLIP** backbone_, which not only significantly simplifies the current two-stage pipeline, but also remarkably yields a better accuracy-cost trade-off. The resulting single-stage system, called FC-CLIP, benefits from the following observations: the _frozen_ CLIP backbone maintains the ability of open-vocabulary classification and can also serve as a strong mask generator, and the _convolutional_ CLIP generalizes well to a larger input resolution than the one used during contrastive image-text pretraining. Surprisingly, FC-CLIP advances state-of-the-art results on various benchmarks, while running practically fast. Specifically, when training on COCO panoptic data only and testing in a zero-shot manner, FC-CLIP achieve 26.8 PQ, 16.8 AP, and 34.1 mIoU on ADE20K, 18.2 PQ, 27.9 mIoU on Mapillary Vistas, 44.0 PQ, 26.8 AP, 56.2 mIoU on Cityscapes, outperforming the prior art under the same setting by +4.2 PQ, +2.4 AP, +4.2 mIoU on ADE20K, +4.0 PQ on Mapillary Vistas and +20.1 PQ on Cityscapes, respectively. Additionally, the training and testing time of FC-CLIP is 7.5x and 6.6x significantly faster than the same prior art, while using 5.9x fewer total model parameters. Meanwhile, FC-CLIP also sets a new state-of-the-art performance across various open-vocabulary semantic segmentation datasets. Code and models are available at https://github.com/bytedance/fc-clip</details>

### Counterfactual Conservative Q Learning for Offline Multi-agent Reinforcement Learning

**Authors:** Jianzhun Shao, Yun Qu, Chen Chen, Hongchang Zhang, Xiangyang Ji

**<details><summary>Abstract**</summary>

Offline multi-agent reinforcement learning is challenging due to the coupling effect of both distribution shift issue common in offline setting and the high dimension issue common in multi-agent setting, making the action out-of-distribution (OOD) and value overestimation phenomenon excessively severe. To mitigate this problem, we propose a novel multi-agent offline RL algorithm, named CounterFactual Conservative Q-Learning (CFCQL) to conduct conservative value estimation. Rather than regarding all the agents as a high dimensional single one and directly applying single agent conservative methods to it, CFCQL calculates conservative regularization for each agent separately in a counterfactual way and then linearly combines them to realize an overall conservative value estimation. We prove that it still enjoys the underestimation property and the performance guarantee as those single agent conservative methods do, but the induced regularization and safe policy improvement bound are independent of the agent number, which is therefore theoretically superior to the direct treatment referred to above, especially when the agent number is large. We further conduct experiments on four environments including both discrete and continuous action settings on both existing and our man-made datasets, demonstrating that CFCQL outperforms existing methods on most datasets and even with a remarkable margin on some of them.</details>

### [Spotlight] Counterfactual Memorization in Neural Language Models

**Authors:** Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramer, Nicholas Carlini

**<details><summary>Abstract**</summary>

Modern neural language models that are widely used in various NLP tasks risk memorizing sensitive information from their training data.Understanding this memorization is important in real world applications and also from a learning-theoretical perspective. An open question in previous studies of language model memorization is how to filter out ``common'' memorization. In fact, most memorization criteria strongly correlate with the number of occurrences in the training set, capturing memorized familiar phrases, public knowledge, templated texts, or other repeated data.We formulate a notion of counterfactual memorization which characterizes how a model's predictions change if a particular document is omitted during training.We identify and study counterfactually-memorized training examples in standard text datasets.We estimate the influence of each memorized training example on the validation set and on generated texts, showing how this can provide direct evidence of the source of memorization at test time.</details>

### Counterfactually Comparing Abstaining Classifiers

**Authors:** Yo Joong Choe, Aditya Gangrade, Aaditya Ramdas

**<details><summary>Abstract**</summary>

Abstaining classifiers have the option to abstain from making predictions on inputs that they are unsure about. These classifiers are becoming increasingly popular in high-stake decision-making problems, as they can withhold uncertain predictions to improve their reliability and safety. When evaluating black-box abstaining classifier(s), however, we lack a principled approach that accounts for what the classifier would have predicted on its abstentions. These missing predictions matter when they can eventually be utilized, either directly or as a backup option in a failure mode. In this paper, we introduce a novel approach and perspective to the problem of evaluating and comparing abstaining classifiers by treating abstentions as missing data. Our evaluation approach is centered around defining the counterfactual score of an abstaining classifier, defined as the expected performance of the classifier had it not been allowed to abstain. We specify the conditions under which the counterfactual score is identifiable: if the abstentions are stochastic, and if the evaluation data is independent of the training data (ensuring that the predictions are missing at random), then the score is identifiable. Note that, if abstentions are deterministic, then the score is unidentifiable because the classifier can perform arbitrarily poorly on its abstentions. Leveraging tools from observational causal inference, we then develop nonparametric and doubly robust methods to efficiently estimate this quantity under identification. Our approach is examined in both simulated and real data experiments.</details>

### Counting Distinct Elements in the Turnstile Model with Differential Privacy under Continual Observation

**Authors:** Palak Jain, Iden Kalemaj, Sofya Raskhodnikova, Satchit Sivakumar, Adam Smith

**<details><summary>Abstract**</summary>

Privacy is a central challenge for systems that learn from sensitive data sets, especially when a system's outputs must be continuously updated to reflect changing data. We consider the achievable error for differentially private continual release of a basic  statistic---the number of distinct items---in a stream where items may be both inserted and deleted (the turnstile model). With only insertions, existing algorithms have additive error just polylogarithmic in the length of the stream $T$. We uncover a much richer landscape in the turnstile model, even without considering memory restrictions. We show that every differentially private mechanism that handles insertions and deletions has worst-case additive error  at least $T^{1/4}$ even under  a relatively weak, event-level privacy definition. Then, we identify a parameter of the input stream, its maximum flippancy, that is low for natural data streams and for which we give tight parameterized error guarantees. Specifically, the maximum flippancy is the largest number of times that the contribution of a single item to the distinct elements count changes over the course of the stream. We present an item-level differentially private mechanism that, for all turnstile streams with  maximum flippancy  $w$, continually outputs the number of distinct elements with an $O(\sqrt{w} \cdot \mathsf{poly}\log T)$ additive error,  without requiring prior knowledge of $w$. We prove that this is the best achievable error bound  that depends only on $w$, for a large range of values of $w$. When $w$ is small, the error of our mechanism is similar to the polylogarithmic in $T$ error in the insertion-only setting, bypassing the hardness in the turnstile model.</details>

### Cross-Scale MAE: A Tale of Multiscale Exploitation in Remote Sensing

**Authors:** Maofeng Tang, Andrei Cozma, Konstantinos Georgiou, Hairong Qi

**<details><summary>Abstract**</summary>

Remote sensing images present unique challenges to image analysis due to the extensive geographic coverage, hardware limitations, and misaligned multi-scale images. This paper revisits the classical multi-scale representation learning problem but under the general framework of self-supervised learning for remote sensing image understanding. We present Cross-Scale MAE, a self-supervised model built upon the Masked Auto-Encoder (MAE). During pre-training, Cross-Scale MAE employs scale augmentation techniques and enforces cross-scale consistency constraints through both contrastive and generative losses, to ensure consistent and meaningful representations well-suited for a wide range of downstream tasks. Further, our implementation leverages the xFormers library to accelerate network pre training on a single GPU while maintaining the quality of learned representations. Experimental evaluations demonstrate that Cross-Scale MAE exhibits superior performance compared to standard MAE and other state-of-the-art remote sensing MAE methods.</details>

### DAC-DETR: Divide the Attention Layers and Conquer

**Authors:** Zhengdong Hu, Yifan Sun, Jingdong Wang, Yi Yang

**<details><summary>Abstract**</summary>

This paper reveals a characteristic of DEtection Transformer (DETR) that negatively impacts its training efficacy, i.e., the cross-attention and self-attention layers in DETR decoder have contrary impacts on the object queries (though both impacts are important). Specifically, we observe the cross-attention tends to gather multiple queries around the same object, while the self-attention disperses these queries far away. To improve the training efficacy, we propose a Divide-And-Conquer DETR (DAC-DETR) that divides the cross-attention out from this contrary for better conquering. During training, DAC-DETR employs an auxiliary decoder that focuses on learning the cross-attention layers. The auxiliary decoder, while sharing all the other parameters, has NO self-attention layers and employs one-to-many label assignment to improve the gathering effect. Experiments show that DAC-DETR brings remarkable improvement over popular DETRs. For example, under the 12 epochs training scheme on MS-COCO, DAC-DETR improves Deformable DETR (ResNet-50) by +3.4 AP and achieves 50.9 (ResNet-50) / 58.1 AP (Swin-Large) based on some popular methods (i.e., DINO and an IoU-related loss). Our code will be made available at https://github.com/huzhengdongcs/DAC-DETR.</details>

### DDF-HO: Hand-Held Object Reconstruction via Conditional Directed Distance Field

**Authors:** Chenyangguang Zhang, Yan Di, Ruida Zhang, Guangyao Zhai, Fabian Manhardt, Federico Tombari, Xiangyang Ji

**<details><summary>Abstract**</summary>

Reconstructing hand-held objects from a single RGB image is an important and challenging problem. Existing works utilizing Signed Distance Fields (SDF) reveal limitations in comprehensively capturing the complex hand-object interactions, since SDF is only reliable within the proximity of the target, and hence, infeasible to simultaneously encode local hand and object cues. To address this issue, we propose DDF-HO, a novel approach leveraging Directed Distance Field (DDF) as the shape representation. Unlike SDF, DDF maps a ray in 3D space, consisting of an origin and a direction, to corresponding DDF values, including a binary visibility signal determining whether the ray intersects the objects and a distance value measuring the distance from origin to target in the given direction. We randomly sample multiple rays and collect local to global geometric features for them by introducing a novel 2D ray-based feature aggregation scheme and a 3D intersection-aware hand pose embedding, combining 2D-3D features to model hand-object interactions. Extensive experiments on synthetic and real-world datasets demonstrate that DDF-HO consistently outperforms all baseline methods by a large margin, especially under Chamfer Distance, with about 80% leap forward. Codes are available at https://github.com/ZhangCYG/DDFHO.</details>

### DELTA: Diverse Client Sampling for Fasting Federated Learning

**Authors:** Lin Wang, Yongxin Guo, Tao Lin, Xiaoying Tang

**<details><summary>Abstract**</summary>

Partial client participation has been widely adopted in Federated Learning (FL) to reduce the communication burden efficiently. However, an inadequate client sampling scheme can lead to the selection of unrepresentative subsets, resulting in significant variance in model updates and slowed convergence. Existing sampling methods are either biased or can be further optimized for faster convergence.In this paper, we present DELTA, an unbiased sampling scheme designed to alleviate these issues. DELTA characterizes the effects of client diversity and local variance, and samples representative clients with valuable information for global model updates. In addition, DELTA is a proven optimal unbiased sampling scheme that minimizes variance caused by partial client participation and outperforms other unbiased sampling schemes in terms of convergence.  Furthermore, to address full-client gradient dependence, we provide a practical version of DELTA depending on the available clients' information, and also analyze its convergence. Our results are validated through experiments on both synthetic and real-world datasets.</details>

### DFRD: Data-Free Robustness Distillation for Heterogeneous Federated Learning

**Authors:** kangyang Luo, Shuai Wang, Yexuan Fu, Xiang Li, Yunshi Lan, Ming Gao

**<details><summary>Abstract**</summary>

Federated Learning (FL) is a privacy-constrained decentralized machine learning paradigm in which clients enable collaborative training without compromising private data. However, how to learn a robust global model in the data-heterogeneous and model-heterogeneous FL scenarios is challenging. To address it, we resort to data-free knowledge distillation to propose a new FL method (namely DFRD).DFRD equips a conditional generator on the server to approximate the training space of the local models uploaded by clients, and systematically investigates its training in terms of fidelity, transferability and diversity. To overcome the catastrophic forgetting of the global model caused by the distribution shifts of the generator across communication rounds, we maintain an exponential moving average copy of the generator on the server.  Additionally, we propose dynamic weighting and label sampling to accurately extract knowledge from local models. Finally, our extensive experiments on various image classification tasks illustrate that DFRD achieves significant performance gains compared to SOTA baselines.</details>

### DOSE: Diffusion Dropout with Adaptive Prior for Speech Enhancement

**Authors:** Wenxin Tai, Yue Lei, Fan Zhou, Goce Trajcevski, Ting Zhong

**<details><summary>Abstract**</summary>

Speech enhancement (SE) aims to improve the intelligibility and quality of speech in the presence of non-stationary additive noise. Deterministic deep learning models have traditionally been used for SE, but recent studies have shown that generative approaches, such as denoising diffusion probabilistic models (DDPMs), can also be effective. However, incorporating condition information into DDPMs for SE remains a challenge. We propose a model-agnostic method called DOSE that employs two efficient condition-augmentation techniques to address this challenge, based on two key insights: (1) We force the model to prioritize the condition factor when generating samples by training it with dropout operation; (2) We inject the condition information into the sampling process by providing an informative adaptive prior. Experiments demonstrate that our approach yields substantial improvements in high-quality and stable speech generation, consistency with the condition factor, and inference efficiency. Codes are publicly available at https://github.com/ICDM-UESTC/DOSE.</details>

### DP-HyPO: An Adaptive Private Framework for Hyperparameter Optimization

**Authors:** Hua Wang, Sheng Gao, Huanyu Zhang, Weijie Su, Milan Shen

**<details><summary>Abstract**</summary>

Hyperparameter optimization, also known as hyperparameter tuning, is a widely recognized technique for improving model performance. Regrettably, when training private ML models, many practitioners often overlook the privacy risks associated with hyperparameter optimization, which could potentially expose sensitive information about the underlying dataset.Currently, the sole existing approach to allow privacy-preserving hyperparameter optimization is to uniformly and randomly select hyperparameters for a number of runs, subsequently reporting the best-performing hyperparameter.In contrast, in non-private settings, practitioners commonly utilize "adaptive" hyperparameter optimization methods such as Gaussian Process-based optimization, which select the next candidate based on information gathered from previous outputs.This substantial contrast between private and non-private hyperparameter optimization underscores a critical concern. In our paper, we introduce DP-HyPO, a pioneering framework for "adaptive" private hyperparameter optimization, aiming to bridge the gap between private and non-private hyperparameter optimization. To accomplish this, we provide a comprehensive differential privacy analysis of our framework. Furthermore, we empirically demonstrate the effectiveness of DP-HyPO on a diverse set of real-world datasets.</details>

### DP-Mix: Mixup-based Data Augmentation for Differentially Private Learning

**Authors:** Wenxuan Bao, Francesco Pittaluga, Vijay Kumar B G, Vincent Bindschaedler

**<details><summary>Abstract**</summary>

Data augmentation techniques, such as simple image transformations and combinations, are highly effective at improving the generalization of computer vision models, especially when training data is limited. However, such techniques are fundamentally incompatible with differentially private learning approaches, due to the latter’s built-in assumption that each training image’s contribution to the learned model is bounded. In this paper, we investigate why naive applications of multi-sample data augmentation techniques, such as mixup, fail to achieve good performance and propose two novel data augmentation techniques specifically designed for the constraints of differentially private learning. Our first technique, DP-Mix_Self, achieves SoTA classification performance across a range of datasets and settings by performing mixup on self-augmented data. Our second technique, DP-Mix_Diff, further improves performance by incorporating synthetic data from a pre-trained diffusion model into the mixup process. We open-source the code at https://github.com/wenxuan-Bao/DP-Mix.</details>

### DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics

**Authors:** Kaiwen Zheng, Cheng Lu, Jianfei Chen, Jun Zhu

**<details><summary>Abstract**</summary>

Diffusion probabilistic models (DPMs) have exhibited excellent performance for high-fidelity image generation while suffering from inefficient sampling. Recent works accelerate the sampling procedure by proposing fast ODE solvers that leverage the specific ODE form of DPMs. However, they highly rely on specific parameterization during inference (such as noise/data prediction), which might not be the optimal choice. In this work, we propose a novel formulation towards the optimal parameterization during sampling that minimizes the first-order discretization error of the ODE solution. Based on such formulation, we propose \textit{DPM-Solver-v3}, a new fast ODE solver for DPMs by introducing several coefficients efficiently computed on the pretrained model, which we call \textit{empirical model statistics}. We further incorporate multistep methods and a predictor-corrector framework, and propose some techniques for improving sample quality at small numbers of function evaluations (NFE) or large guidance scales. Experiments show that DPM-Solver-v3 achieves consistently better or comparable performance in both unconditional and conditional sampling with both pixel-space and latent-space DPMs, especially in 5$\sim$10 NFEs. We achieve FIDs of 12.21 (5 NFE), 2.51 (10 NFE) on unconditional CIFAR10, and MSE of 0.55 (5 NFE, 7.5 guidance scale) on Stable Diffusion, bringing a speed-up of 15\%$\sim$30\% compared to previous state-of-the-art training-free methods. Code is available at \url{https://github.com/thu-ml/DPM-Solver-v3}.</details>

### Data-Dependent Bounds for Online Portfolio Selection Without Lipschitzness and Smoothness

**Authors:** Chung-En Tsai, Ying-Ting Lin, Yen-Huan Li

**<details><summary>Abstract**</summary>

This work introduces the first small-loss and gradual-variation regret bounds for online portfolio selection, marking the first instances of data-dependent bounds for online convex optimization with non-Lipschitz, non-smooth losses. The algorithms we propose exhibit sublinear regret rates in the worst cases and achieve logarithmic regrets when the data is "easy," with per-round time almost linear in the number of investment alternatives. The regret bounds are derived using novel smoothness characterizations of the logarithmic loss, a local norm-based analysis of following the regularized leader (FTRL) with self-concordant regularizers, which are not necessarily barriers, and an implicit variant of optimistic FTRL with the log-barrier.</details>

### Data-driven Optimal Filtering for Linear Systems with Unknown Noise Covariances

**Authors:** Shahriar Talebi, Amirhossein Taghvaei, Mehran Mesbahi

**<details><summary>Abstract**</summary>

This paper examines learning the optimal filtering policy, known as the Kalman gain, for a linear system with unknown noise covariance matrices using noisy output data. The learning problem is formulated as a stochastic policy optimiza- tion problem, aiming to minimize the output prediction error. This formulation provides a direct bridge between data-driven optimal control and, its dual, op- timal filtering. Our contributions are twofold. Firstly, we conduct a thorough convergence analysis of the stochastic gradient descent algorithm, adopted for the filtering problem, accounting for biased gradients and stability constraints. Secondly, we carefully leverage a combination of tools from linear system theory and high-dimensional statistics to derive bias-variance error bounds that scale logarithmically with problem dimension, and, in contrast to subspace methods, the length of output trajectories only affects the bias term.</details>

### De novo Drug Design using Reinforcement Learning with Multiple GPT Agents

**Authors:** Xiuyuan Hu, Guoqing Liu, Yang Zhao, Hao Zhang

**<details><summary>Abstract**</summary>

*De novo* drug design is a pivotal issue in pharmacology and a new area of focus in AI for science research. A central challenge in this field is to generate molecules with specific properties while also producing a wide range of diverse candidates. Although advanced technologies such as transformer models and reinforcement learning have been applied in drug design, their potential has not been fully realized. Therefore, we propose MolRL-MGPT, a reinforcement learning algorithm with multiple GPT agents for drug molecular generation. To promote molecular diversity, we encourage the agents to collaborate in searching for desirable molecules in diverse directions. Our algorithm has shown promising results on the GuacaMol benchmark and exhibits efficacy in designing inhibitors against SARS-CoV-2 protein targets. The codes are available at: https://github.com/HXYfighter/MolRL-MGPT.</details>

### [Spotlight] Debias Coarsely, Sample Conditionally: Statistical Downscaling through Optimal Transport and Probabilistic Diffusion Models

**Authors:** Zhong Yi Wan, Ricardo Baptista, Anudhyan Boral, Yi-Fan Chen, John Anderson, Fei Sha, Leonardo Zepeda-Núñez

**<details><summary>Abstract**</summary>

We introduce a two-stage probabilistic framework for statistical downscaling using unpaired data. Statistical downscaling seeks a probabilistic map to transform low-resolution data from a biased coarse-grained numerical scheme to high-resolution data that is consistent with a high-fidelity scheme. Our framework tackles the problem bycomposing two transformations: (i) a debiasing step via an optimal transport map, and (ii) an upsampling step achieved by a probabilistic diffusion model with a posteriori conditional sampling. This approach characterizes a conditional distribution without needing paired data, and faithfully recovers relevant physical statistics from biased samples. We demonstrate the utility of the proposed approach on one- and two-dimensional fluid flow problems, which are representative of the core difficulties present in numerical simulations of weather and climate. Our method produces realistic high-resolution outputs from low-resolution inputs, by upsampling resolutions of $8	imes$ and $16	imes$. Moreover, our procedure correctly matches the statistics of physical quantities, even when the low-frequency content of the inputs and outputs do not match, a crucial but difficult-to-satisfy assumption needed by current state-of-the-art alternatives. Code for this work is available at: https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/probabilistic_diffusion.</details>

### Deciphering Spatio-Temporal Graph Forecasting: A Causal Lens and Treatment

**Authors:** Yutong Xia, Yuxuan Liang, Haomin Wen, Xu Liu, Kun Wang, Zhengyang Zhou, Roger Zimmermann

**<details><summary>Abstract**</summary>

Spatio-Temporal Graph (STG) forecasting is a fundamental task in many real-world applications. Spatio-Temporal Graph Neural Networks have emerged as the most popular method for STG forecasting, but they often struggle with temporal out-of-distribution (OoD) issues and dynamic spatial causation. In this paper, we propose a novel framework called CaST to tackle these two challenges via causal treatments. Concretely, leveraging a causal lens, we first build a structural causal model to decipher the data generation process of STGs. To handle the temporal OoD issue, we employ the back-door adjustment by a novel disentanglement block to separate the temporal environments from input data. Moreover, we utilize the front-door adjustment and adopt edge-level convolution to model the ripple effect of causation. Experiments results on three real-world datasets demonstrate the effectiveness of CaST, which consistently outperforms existing methods with good interpretability. Our source code is available at https://github.com/yutong-xia/CaST.</details>

### Decision-Aware Actor-Critic with Function Approximation and Theoretical Guarantees

**Authors:** Sharan Vaswani, Amirreza Kazemi, Reza Babanezhad Harikandeh, Nicolas Le Roux

**<details><summary>Abstract**</summary>

Actor-critic (AC) methods are widely used in reinforcement learning (RL), and benefit from the flexibility of using any policy gradient method as the actor and value-based method as the critic. The critic is usually trained by minimizing the TD error, an objective that is potentially decorrelated with the true goal of achieving a high reward with the actor. We address this mismatch by designing a joint objective for training the actor and critic in a decision-aware fashion. We use the proposed objective to design a generic, AC algorithm that can easily handle any function approximation. We explicitly characterize the conditions under which the resulting algorithm guarantees monotonic policy improvement, regardless of the choice of the policy and critic parameterization. Instantiating the generic algorithm results in an actor that involves maximizing a sequence of surrogate functions (similar to TRPO, PPO), and a critic that involves minimizing a closely connected objective. Using simple bandit examples, we provably establish the benefit of the proposed critic objective over the standard squared error. Finally, we empirically demonstrate the benefit of our decision-aware actor-critic framework on simple RL problems.</details>

### Decorate3D: Text-Driven High-Quality Texture Generation for Mesh Decoration in the Wild

**Authors:** Yanhui Guo, Xinxin Zuo, Peng Dai, Juwei Lu, Xiaolin Wu, Li cheng, Youliang Yan, Songcen Xu, Xiaofei Wu

**<details><summary>Abstract**</summary>

This paper presents Decorate3D, a versatile and user-friendly method for the creation and editing of 3D objects using images. Decorate3D models a real-world object of interest by neural radiance field (NeRF) and decomposes the NeRF representation into an explicit mesh representation, a view-dependent texture, and a diffuse UV texture. Subsequently, users can either manually edit the UV or provide a prompt for the automatic generation of a new 3D-consistent texture. To achieve high-quality 3D texture generation, we propose a structure-aware score distillation sampling method to optimize a neural UV texture based on user-defined text and empower an image diffusion model with 3D-consistent generation capability. Furthermore, we introduce a few-view resampling training method and utilize a super-resolution model to obtain refined high-resolution UV textures (2048$	imes$2048) for 3D texturing. Extensive experiments collectively validate the superior performance of Decorate3D in retexturing real-world 3D objects. Project page: https://decorate3d.github.io/Decorate3D/.</details>

### Deep learning with kernels through RKHM and the Perron-Frobenius operator

**Authors:** Yuka Hashimoto, Masahiro Ikeda, Hachem Kadri

**<details><summary>Abstract**</summary>

Reproducing kernel Hilbert $C^*$-module (RKHM) is a generalization of reproducing kernel Hilbert space (RKHS) by means of $C^*$-algebra, and the Perron-Frobenius operator is a linear operator related to the composition of functions. Combining these two concepts, we present deep RKHM, a deep learning framework for kernel methods. We derive a new Rademacher generalization bound in this setting and provide a theoretical interpretation of benign overfitting by means of Perron-Frobenius operators. By virtue of $C^*$-algebra, the dependency of the bound on output dimension is milder than existing bounds. We show that $C^*$-algebra is a suitable tool for deep learning with kernels, enabling us to take advantage of the product structure of operators and to provide a clear connection with convolutional neural networks. Our theoretical analysis provides a new lens through which one can design and analyze deep kernel methods.</details>

### Defending against Data-Free Model Extraction by  Distributionally Robust Defensive Training

**Authors:** Zhenyi Wang, Li Shen, Tongliang Liu, Tiehang Duan, Yanjun Zhu, Donglin Zhan, DAVID DOERMANN, Mingchen Gao

**<details><summary>Abstract**</summary>

Data-Free Model Extraction (DFME) aims to clone a black-box model without knowing its original training data distribution, making it much easier for attackers to steal commercial models. Defense against DFME faces several challenges: (i) effectiveness; (ii) efficiency; (iii) no prior on the attacker's query data distribution and strategy. However, existing defense methods: (1) are highly computation and memory inefficient; or (2) need strong assumptions about attack data distribution; or (3) can only delay the attack or prove a model theft after the model stealing has happened. In this work, we propose a Memory and Computation efficient defense approach, named MeCo, to prevent DFME from happening while maintaining the model utility simultaneously by distributionally robust defensive training on the target victim model. Specifically, we randomize the input so that it: (1) causes a mismatch of the knowledge distillation loss for attackers; (2) disturbs the zeroth-order gradient estimation; (3) changes the label prediction for the attack query data. Therefore, the attacker can only extract misleading information from the black-box model. Extensive experiments on defending against both decision-based and score-based DFME demonstrate that MeCo can significantly reduce the effectiveness of existing DFME methods and substantially improve running efficiency.</details>

### [Spotlight] Dense and Aligned Captions (DAC) Promote Compositional Reasoning in VL Models

**Authors:** Sivan Doveh, Assaf Arbelle, Sivan Harary, Roei Herzig, Donghyun Kim, Paola Cascante-Bonilla, Amit Alfassy, Rameswar Panda, Raja Giryes, Rogerio Feris, Shimon Ullman, Leonid Karlinsky

**<details><summary>Abstract**</summary>

Vision and Language (VL) models offer an effective method for aligning representation spaces of images and text allowing for numerous applications such as cross-modal retrieval, visual and multi-hop question answering, captioning, and many more. However, the aligned image-text spaces learned by all the popular VL models are still suffering from the so-called 'object bias' - their representations behave as 'bags of nouns' mostly ignoring or downsizing the attributes, relations, and states of objects described/appearing in texts/images. Although some great attempts at fixing these `compositional reasoning' issues were proposed in the recent literature, the problem is still far from being solved. In this paper, we uncover two factors limiting the VL models' compositional reasoning performance. These two factors are properties of the paired VL dataset used for finetuning (or pre-training) the VL model: (i) the caption quality, or in other words 'image-alignment', of the texts; and (ii) the 'density' of the captions in the sense of mentioning all the details appearing on the image. We propose a fine-tuning approach for automatically treating these factors on a standard collection of paired VL data (CC3M). Applied to CLIP, we demonstrate its significant compositional reasoning performance increase of up to $\sim27$\% over the base model, up to $\sim20$\% over the strongest baseline, and by $6.7$\% on average. Our code is provided in the Supplementary and would be released upon acceptance.</details>

### Described Object Detection: Liberating Object Detection with Flexible Expressions

**Authors:** Chi Xie, Zhao Zhang, Yixuan Wu, Feng Zhu, Rui Zhao, Shuang Liang

**<details><summary>Abstract**</summary>

Detecting objects based on language information is a popular task that includes Open-Vocabulary object Detection (OVD) and Referring Expression Comprehension (REC). In this paper, we advance them to a more practical setting called *Described Object Detection* (DOD) by expanding category names to flexible language expressions for OVD and overcoming the limitation of REC only grounding the pre-existing object. We establish the research foundation for DOD by constructing a *Description Detection Dataset* ($D^3$). This dataset features flexible language expressions, whether short category names or long descriptions, and annotating all described objects on all images without omission. By evaluating previous SOTA methods on $D^3$, we find some troublemakers that fail current REC, OVD, and bi-functional methods. REC methods struggle with confidence scores, rejecting negative instances, and multi-target scenarios, while OVD methods face constraints with long and complex descriptions. Recent bi-functional methods also do not work well on DOD due to their separated training procedures and inference strategies for REC and OVD tasks. Building upon the aforementioned findings, we propose a baseline that largely improves REC methods by reconstructing the training data and introducing a binary classification sub-task, outperforming existing methods. Data and code are available at https://github.com/shikras/d-cube and related works are tracked in https://github.com/Charles-Xie/awesome-described-object-detection.</details>

### Designing Robust Transformers using Robust Kernel Density Estimation

**Authors:** Xing Han, Tongzheng Ren, Tan Nguyen, Khai Nguyen, Joydeep Ghosh, Nhat Ho

**<details><summary>Abstract**</summary>

Transformer-based architectures have recently exhibited remarkable successes across different domains beyond just powering large language models. However, existing approaches typically focus on predictive accuracy and computational cost, largely ignoring certain other practical issues such as robustness to contaminated samples. In this paper, by re-interpreting the self-attention mechanism as a non-parametric kernel density estimator, we adapt classical robust kernel density estimation methods to develop novel classes of transformers that are resistant to adversarial attacks and data contamination. We first propose methods that down-weight outliers in RKHS when computing the self-attention operations. We empirically show that these methods produce improved performance over existing state-of-the-art methods, particularly on image data under adversarial attacks. Then we leverage the median-of-means principle to obtain another efficient approach that results in noticeably enhanced performance and robustness on language modeling and time series classification tasks. Our methods can be combined with existing transformers to augment their robust properties, thus promising to impact a wide variety of applications.</details>

### Differentiable Random Partition Models

**Authors:** Thomas Sutter, Alain Ryser, Joram Liebeskind, Julia Vogt

**<details><summary>Abstract**</summary>

Partitioning a set of elements into an unknown number of mutually exclusive subsets is essential in many machine learning problems.However, assigning elements, such as samples in a dataset or neurons in a network layer, to an unknown and discrete number of subsets is inherently non-differentiable, prohibiting end-to-end gradient-based optimization of parameters.We overcome this limitation by proposing a novel two-step method for inferring partitions, which allows its usage in variational inference tasks.This new approach enables reparameterized gradients with respect to the parameters of the new random partition model.Our method works by inferring the number of elements per subset and, second, by filling these subsets in a learned order.We highlight the versatility of our general-purpose approach on three different challenging experiments: variational clustering, inference of shared and independent generative factors under weak supervision, and multitask learning.</details>

### Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes

**Authors:** Ali Younis, Erik Sudderth

**<details><summary>Abstract**</summary>

Particle filters flexibly represent multiple posterior modes nonparametrically, via a collection of weighted samples, but have classically been applied to tracking problems with known dynamics and observation likelihoods.  Such generative models may be inaccurate or unavailable for high-dimensional observations like images.  We instead leverage training data to discriminatively learn particle-based representations of uncertainty in latent object states, conditioned on arbitrary observations via deep neural network encoders.   While prior discriminative particle filters have used heuristic relaxations of discrete particle resampling, or biased learning by truncating gradients at resampling steps, we achieve unbiased and low-variance gradient estimates by representing posteriors as continuous mixture densities.  Our theory and experiments expose dramatic failures of existing reparameterization-based estimators for mixture gradients, an issue we address via an importance-sampling gradient estimator. Unlike standard recurrent neural networks, our mixture density particle filter represents multimodal uncertainty in continuous latent states, improving accuracy and robustness.  On a range of challenging tracking and robot localization problems, our approach achieves dramatic improvements in accuracy, will also showing much greater stability across multiple training runs.</details>

### Differentially Private Statistical Inference through $\beta$-Divergence One Posterior Sampling

**Authors:** Jack Jewson, Sahra Ghalebikesabi, Chris C Holmes

**<details><summary>Abstract**</summary>

Differential privacy guarantees allow the results of a statistical analysis involving sensitive data to be released without compromising the privacy of any individual taking part. Achieving such guarantees generally requires the injection of noise, either directly into parameter estimates or into the estimation process. Instead of artificially introducing perturbations, sampling from Bayesian posterior distributions has been shown to be a special case of the exponential mechanism, producing consistent,and efficient private estimates without altering the data generative process. The application of current approaches has, however, been limited by their strong bounding assumptions which do not hold for basic models, such as simple linear regressors.To ameliorate this, we propose $\beta$D-Bayes, a posterior sampling  scheme from a generalised posterior targeting the minimisation of the $\beta$-divergence between the model and the data generating process. This provides private estimation that is generally applicable without requiring changes to the underlying model and consistently learns the data generating parameter. We show that $\beta$D-Bayes produces more precise inference estimation for the same privacy guarantees, and further facilitates differentially private estimation of complex classifiers, and continuous regression models such as neural networks, which goes beyond what has been currently possible with private posterior sampling.</details>

### Diffused Task-Agnostic Milestone Planner

**Authors:** Mineui Hong, Minjae Kang, Songhwai Oh

**<details><summary>Abstract**</summary>

Addressing decision-making problems using sequence modeling to predict future trajectories shows promising results in recent years.In this paper, we take a step further to leverage the sequence predictive method in wider areas such as long-term planning, vision-based control, and multi-task decision-making.To this end, we propose a method to utilize a diffusion-based generative sequence model to plan a series of milestones in a latent space and to have an agent to follow the milestones to accomplish a given task.The proposed method can learn control-relevant, low-dimensional latent representations of milestones, which makes it possible to efficiently perform long-term planning and vision-based control.Furthermore, our approach exploits generation flexibility of the diffusion model, which makes it possible to plan diverse trajectories for multi-task decision-making.We demonstrate the proposed method across offline reinforcement learning (RL) benchmarks and an visual manipulation environment.The results show that our approach outperforms offline RL methods in solving long-horizon, sparse-reward tasks and multi-task problems,while also achieving the state-of-the-art performance on the most challenging vision-based manipulation benchmark.</details>

### Diffusion Model for Graph Inverse Problems: Towards Effective Source Localization on Complex Networks

**Authors:** Xin Yan, Hui Fang, Qiang He

**<details><summary>Abstract**</summary>

Information diffusion problems, such as the spread of epidemics or rumors, are widespread in society. The inverse problems of graph diffusion, which involve locating the sources and identifying the paths of diffusion based on currently observed diffusion graphs, are crucial to controlling the spread of information. The problem of localizing the source of diffusion is highly ill-posed, presenting a major obstacle in accurately assessing the uncertainty involved. Besides, while comprehending how information diffuses through a graph is crucial, there is a scarcity of research on reconstructing the paths of information propagation. To tackle these challenges, we propose a probabilistic model called DDMSL (Discrete Diffusion Model for Source Localization). Our approach is based on the natural diffusion process of information propagation over complex networks, which can be formulated using a message-passing function. First, we model the forward diffusion of information using Markov chains. Then, we design a reversible residual network to construct a denoising-diffusion model in discrete space for both source localization and reconstruction of information diffusion paths. We provide rigorous theoretical guarantees for DDMSL and demonstrate its effectiveness through extensive experiments on five real-world datasets.</details>

### Diffusion Representation for Asymmetric Kernels via Magnetic Transform

**Authors:** Mingzhen He, FAN He, Ruikai Yang, Xiaolin Huang

**<details><summary>Abstract**</summary>

As a nonlinear dimension reduction technique, the diffusion map (DM) has been widely used. In DM, kernels play an important role for capturing the nonlinear relationship of data. However, only symmetric kernels can be used now, which prevents the use of DM in directed graphs, trophic networks, and other real-world scenarios where the intrinsic and extrinsic geometries in data are asymmetric. A promising technique is the magnetic transform which  converts an asymmetric matrix to a Hermitian one. However, we are facing essential problems, including how diffusion distance could be preserved and how divergence could be avoided during diffusion process. Via theoretical proof, we successfully establish a diffusion representation framework with the magnetic transform, named MagDM. The effectiveness and robustness for dealing data endowed with asymmetric proximity are demonstrated on three synthetic datasets and two trophic networks.</details>

### Direct Diffusion Bridge using Data Consistency for Inverse Problems

**Authors:** Hyungjin Chung, Jeongsol Kim, Jong Chul Ye

**<details><summary>Abstract**</summary>

Diffusion model-based inverse problem solvers have shown impressive performance, but are limited in speed, mostly as they require reverse diffusion sampling starting from noise. Several recent works have tried to alleviate this problem by building a diffusion process, directly bridging the clean and the corrupted for specific inverse problems. In this paper, we first unify these existing works under the name Direct Diffusion Bridges (DDB), showing that while motivated by different theories, the resulting algorithms only differ in the choice of parameters. Then, we highlight a critical limitation of the current DDB framework, namely that it does not ensure data consistency. To address this problem, we propose a modified inference procedure that imposes data consistency without the need for fine-tuning. We term the resulting method data Consistent DDB (CDDB), which outperforms its inconsistent counterpart in terms of both perception and distortion metrics, thereby effectively pushing the Pareto-frontier toward the optimum. Our proposed method achieves state-of-the-art results on both evaluation criteria, showcasing its superiority over existing methods. Code is open-sourced [here](https://github.com/HJ-harry/CDDB).</details>

### Direction-oriented Multi-objective Learning: Simple and Provable Stochastic Algorithms

**Authors:** Peiyao Xiao, Hao Ban, Kaiyi Ji

**<details><summary>Abstract**</summary>

Multi-objective optimization (MOO) has become an influential framework in many machine learning problems with multiple objectives such as learning with multiple criteria and multi-task learning (MTL). In this paper, we propose a new direction-oriented multi-objective formulation by regularizing the common descent direction within a neighborhood of a direction that optimizes a linear combination of objectives such as the average loss in MTL or a weighted loss that places higher emphasis on some tasks than the others. This formulation includes GD and MGDA as special cases, enjoys the direction-oriented benefit as in CAGrad, and facilitates the design of stochastic algorithms. To solve this problem, we propose Stochastic Direction-oriented Multi-objective Gradient descent (SDMGrad) with simple SGD type of updates, and its variant SDMGrad-OS with an efficient objective sampling. We develop a comprehensive convergence analysis for the proposed methods with different loop sizes and regularization coefficients. We show that both SDMGrad and SDMGrad-OS achieve improved sample complexities to find an $\epsilon$-accurate Pareto stationary point while achieving a small $\epsilon$-level distance toward a conflict-avoidant (CA) direction. For a constant-level CA distance, their sample complexities match the best known $\mathcal{O}(\epsilon^{-2})$ without bounded function value assumption. Extensive experiments show that our methods achieve competitive or improved performance compared to existing gradient manipulation approaches in a series of tasks on multi-task supervised learning and reinforcement learning. Code is available at https://github.com/ml-opt-lab/sdmgrad.</details>

### DisDiff: Unsupervised Disentanglement of Diffusion Probabilistic Models

**Authors:** Tao Yang, Yuwang Wang, Yan Lu, Nanning Zheng

**<details><summary>Abstract**</summary>

Targeting to understand the underlying explainable factors behind observations and modeling the conditional generation process on these factors, we connect disentangled representation learning to diffusion probabilistic models (DPMs) to take advantage of the remarkable modeling ability of DPMs. We propose a new task, disentanglement of (DPMs): given a pre-trained DPM, without any annotations of the factors, the task is to automatically discover the inherent factors behind the observations and disentangle the gradient fields of DPM into sub-gradient fields, each conditioned on the representation of each discovered factor. With disentangled DPMs, those inherent factors can be automatically discovered, explicitly represented and clearly injected into the diffusion process via the sub-gradient fields. To tackle this task, we devise an unsupervised approach, named DisDiff, and for the first time achieving disentangled representation learning in the framework of DPMs. Extensive experiments on synthetic and real-world datasets demonstrate the effectiveness of DisDiff.</details>

### [Spotlight] Distance-Restricted Folklore Weisfeiler-Leman GNNs with Provable Cycle Counting Power

**Authors:** Junru Zhou, Jiarui Feng, Xiyuan Wang, Muhan Zhang

**<details><summary>Abstract**</summary>

The ability of graph neural networks (GNNs) to count certain graph substructures, especially cycles, is important for the success of GNNs on a wide range of tasks. It has been recently used as a popular metric for evaluating the expressive power of GNNs. Many of the proposed GNN models with provable cycle counting power are based on subgraph GNNs, i.e., extracting a bag of subgraphs from the input graph, generating representations for each subgraph, and using them to augment the representation of the input graph. However, those methods require heavy preprocessing, and suffer from high time and memory costs. In this paper, we overcome the aforementioned limitations of subgraph GNNs by proposing a novel class of GNNs---$d$-Distance-Restricted FWL(2) GNNs, or $d$-DRFWL(2) GNNs, based on the well-known FWL(2) algorithm. As a heuristic method for graph isomorphism testing, FWL(2) colors all node pairs in a graph and performs message passing among those node pairs. In order to balance the expressive power and complexity, $d$-DRFWL(2) GNNs simplify FWL(2) by restricting the range of message passing to node pairs whose mutual distances are at most $d$. This way, $d$-DRFWL(2) GNNs exploit graph sparsity while avoiding the expensive subgraph extraction operations in subgraph GNNs, making both the time and space complexity lower. We theoretically investigate both the discriminative power and the cycle counting power of $d$-DRFWL(2) GNNs. Our most important finding is that $d$-DRFWL(2) GNNs have provably strong cycle counting power even with $d=2$: they can count all 3, 4, 5, 6-cycles. Since 6-cycles (e.g., benzene rings) are ubiquitous in organic molecules, being able to detect and count them is crucial for achieving robust and generalizable performance on molecular tasks. Experiments on both synthetic datasets and molecular datasets verify our theory. To the best of our knowledge, 2-DRFWL(2) GNN is the most efficient GNN model to date (both theoretically and empirically) that can count up to 6-cycles.</details>

### Diversify Your Vision Datasets with Automatic Diffusion-based Augmentation

**Authors:** Lisa Dunlap, Alyssa Umino, Han Zhang, Jiezhi Yang, Joseph Gonzalez, Trevor Darrell

**<details><summary>Abstract**</summary>

Many fine-grained classification tasks, like rare animal identification, have limited training data and consequently classifiers trained on these datasets often fail to generalize to variations in the domain like changes in weather or location. As such, we explore how natural language descriptions of the domains seen in training data can be used with large vision models trained on diverse pretraining datasets to generate useful variations of the training data. We introduce ALIA (Automated Language-guided Image Augmentation), a method which utilizes large vision and language models to automatically generate natural language descriptions of a dataset's domains and augment the training data via language-guided image editing. To maintain data integrity, a model trained on the original dataset filters out minimal image edits and those which corrupt class-relevant information. The resulting dataset is visually consistent with the original training data and offers significantly enhanced diversity. We show that ALIA is able to surpasses traditional data augmentation and text-to-image generated data on fine-grained classification tasks, including cases of domain generalization and contextual bias. Code is available at https://github.com/lisadunlap/ALIA.</details>

### Divide, Evaluate, and Refine: Evaluating and Improving Text-to-Image Alignment with Iterative VQA Feedback

**Authors:** Jaskirat Singh, Liang Zheng

**<details><summary>Abstract**</summary>

The field of text-conditioned image generation has made unparalleled progress with the recent advent of latent diffusion models. While revolutionary, as the complexity of given text input increases, the current state of art diffusion models may still fail in generating images that accurately convey the semantics of the given prompt. Furthermore, such misalignments are often left undetected by pretrained multi-modal models such as CLIP.  To address these problems, in this paper, we explore a simple yet effective decompositional approach towards both evaluation and improvement of text-to-image alignment. In particular,  we first introduce a Decompositional-Alignment-Score which given a complex caption decomposes it into a set of disjoint assertions. The alignment of each assertion with generated images is then measured using a VQA model. Finally, alignment scores for different assertions are combined aposteriori to give the final text-to-image alignment score. Experimental analysis reveals that the proposed alignment metric shows a significantly higher correlation with human ratings as opposed to traditional CLIP, BLIP scores. Furthermore, we also find that the assertion level alignment scores also provide useful feedback which can then be used in a simple iterative procedure to gradually increase the expressivity of different assertions in the final image outputs. Human user studies indicate that the proposed approach surpasses previous state-of-the-art by 8.7% in overall text-to-image alignment accuracy.</details>

### Django: Detecting Trojans in Object Detection Models via Gaussian Focus Calibration

**Authors:** Guangyu Shen, Siyuan Cheng, Guanhong Tao, Kaiyuan Zhang, Yingqi Liu, Shengwei An, Shiqing Ma, Xiangyu Zhang

**<details><summary>Abstract**</summary>

Object detection models are vulnerable to backdoor or trojan attacks, where an attacker can inject malicious triggers into the model, leading to altered behavior during inference. As a defense mechanism, trigger inversion leverages optimization to reverse-engineer triggers and identify compromised models. While existing trigger inversion methods assume that each instance from the support set is equally affected by the injected trigger, we observe that the poison effect can vary significantly across bounding boxes in object detection models due to its dense prediction nature, leading to an undesired optimization objective misalignment issue for existing trigger reverse-engineering methods. To address this challenge, we propose the first object detection backdoor detection framework Django (Detecting Trojans in Object Detection Models via Gaussian Focus Calibration). It leverages a dynamic Gaussian weighting scheme that prioritizes more vulnerable victim boxes and assigns appropriate coefficients to calibrate the optimization objective during trigger inversion. In addition, we combine Django with a novel label proposal pre-processing technique to enhance its efficiency. We evaluate Django on 3 object detection image datasets, 3 model architectures, and 2 types of attacks, with a total of 168 models. Our experimental results show that Django outperforms 6 state-of-the-art baselines, with up to 38% accuracy improvement and 10x reduced overhead. The code is available at https://github.com/PurduePAML/DJGO.</details>

### Double Auctions with Two-sided Bandit Feedback

**Authors:** Soumya Basu, Abishek Sankararaman

**<details><summary>Abstract**</summary>

Double Auction enables decentralized transfer of goods between multiple buyers and sellers, thus underpinning functioning of many online marketplaces. Buyers and sellers compete in these markets through bidding, but do not often know their own valuation a-priori. As the allocation and pricing happens through bids, the profitability of participants, hence sustainability of such markets, depends crucially on learning respective valuations through repeated interactions. We initiate the study of Double Auction markets under bandit feedback on both buyers' and sellers' side. We show with confidence bound based bidding, and `Average Pricing' there is an efficient price discovery among the participants.  In particular, the regret on combined valuation of the buyers and the sellers -- a.k.a. the social regret -- is $O(\log(T)/\Delta)$ in $T$ rounds, where $\Delta$ is the minimum price gap. Moreover, the buyers and sellers exchanging goods attain $O(\sqrt{T})$ regret, individually. The buyers and sellers who do not benefit from exchange in turn only experience $O(\log{T}/ \Delta)$  regret individually in $T$ rounds.  We augment our upper bound by showing that $\omega(\sqrt{T})$ individual regret, and $\omega(\log{T})$ social regret is unattainable in certain Double Auction markets. Our paper is the first to provide decentralized learning algorithms in a two-sided market where \emph{both sides have uncertain preference} that need to be learned.</details>

### Dual Self-Awareness Value Decomposition Framework without Individual Global Max for Cooperative MARL

**Authors:** Zhiwei Xu, Bin Zhang, dapeng li, Guangchong Zhou, Zeren Zhang, Guoliang Fan

**<details><summary>Abstract**</summary>

Value decomposition methods have gained popularity in the field of cooperative multi-agent reinforcement learning. However, almost all existing methods follow the principle of Individual Global Max (IGM) or its variants, which limits their problem-solving capabilities. To address this, we propose a dual self-awareness value decomposition framework, inspired by the notion of dual self-awareness in psychology, that entirely rejects the IGM premise. Each agent consists of an ego policy for action selection and an alter ego value function to solve the credit assignment problem. The value function factorization can ignore the IGM assumption by utilizing an explicit search procedure. On the basis of the above, we also suggest a novel anti-ego exploration mechanism to avoid the algorithm becoming stuck in a local optimum. As the first fully IGM-free value decomposition method, our proposed framework achieves desirable performance in various cooperative tasks.</details>

### Dynamic Non-monotone Submodular Maximization

**Authors:** Kiarash Banihashem, Leyla Biabani, Samira Goudarzi, MohammadTaghi Hajiaghayi, Peyman Jabbarzade, Morteza Monemizadeh

**<details><summary>Abstract**</summary>

Maximizing submodular functions has been increasingly used in many applications of machine learning, such as data summarization, recommendation systems, and feature selection. Moreover, there has been a growing interest in both submodular maximization and dynamic algorithms. In 2020, Monemizadeh and Lattanzi, Mitrovic, Norouzi-Fard, Tarnawski, and Zadimoghaddam initiated developing dynamic algorithms for the monotone submodular maximization problem under the cardinality constraint $k$. In 2022, Chen and Peng studied the complexity of this problem and raised an important open question: "\emph{Can we extend [fully dynamic] results (algorithm or hardness) to non-monotone submodular maximization?}". We affirmatively answer their question by demonstrating a reduction from maximizing a non-monotone submodular function under the cardinality constraint $k$ to maximizing a monotone submodular function under the same constraint. Through this reduction, we obtain the first dynamic algorithms to solve the non-monotone submodular maximization problem under the cardinality constraint $k$. Our algorithms maintain an $(8+\epsilon)$-approximate of the solution and use expected amortized $O(\epsilon^{-3}k^3\log^3(n)\log(k))$ or $O(\epsilon^{-1}k^2\log^3(k))$ oracle queries per update, respectively. Furthermore, we showcase the benefits of our dynamic algorithm for video summarization and max-cut problems on several real-world data sets.</details>

### Dynamic Prompt Learning: Addressing Cross-Attention Leakage for Text-Based Image Editing

**Authors:** kai wang, Fei Yang, Shiqi Yang, Muhammad Atif Butt, Joost van de Weijer

**<details><summary>Abstract**</summary>

Large-scale text-to-image generative models have been a ground-breaking development in generative AI, with diffusion models showing their astounding ability to synthesize convincing images following an input text prompt. The goal of image editing research is to give users control over the generated images by modifying the text prompt. Current image editing techniques are susceptible to unintended modifications of regions outside the targeted area, such as on the background or on distractor objects which have some semantic or visual relationship with the targeted object. According to our experimental findings, inaccurate cross-attention maps are at the root of this problem. Based on this observation, we propose $\textit{Dynamic Prompt Learning}$ ($DPL$) to force cross-attention maps to focus on correct $\textit{noun}$ words in the text prompt. By updating the dynamic tokens for nouns in the textual input with the proposed leakage repairment losses, we achieve fine-grained image editing over particular objects while preventing undesired changes to other image regions. Our method $DPL$, based on the publicly available $\textit{Stable Diffusion}$, is extensively evaluated on a wide range of images, and consistently obtains superior results both quantitatively (CLIP score, Structure-Dist) and qualitatively (on user-evaluation). We show improved prompt editing results for Word-Swap, Prompt Refinement, and Attention Re-weighting, especially for complex multi-object scenes.</details>

### Dynamic Regret of Adversarial Linear Mixture MDPs

**Authors:** Long-Fei Li, Peng Zhao, Zhi-Hua Zhou

**<details><summary>Abstract**</summary>

We study reinforcement learning in episodic inhomogeneous MDPs with adversarial full-information rewards and the unknown transition kernel. We consider the linear mixture MDPs whose transition kernel is a linear mixture model and choose the \emph{dynamic regret} as the performance measure. Denote by $d$ the dimension of the feature mapping, $H$ the horizon, $K$ the number of episodes, $P_T$ the non-stationary measure, we propose a novel algorithm that enjoys an $\widetilde{\mathcal{O}}\big(\sqrt{d^2 H^3K} + \sqrt{H^4(K+P_T)(1+P_T)}\big)$ dynamic regret under the condition that $P_T$ is known, which improves previously best-known dynamic regret for adversarial linear mixture MDP and adversarial tabular MDPs. We also establish an $\Omega\big(\sqrt{d^2 H^3 K} + \sqrt{H K (H+P_T)}\big)$ lower bound, indicating our algorithm is \emph{optimal} in $K$ and $P_T$. Furthermore, when the non-stationary measure $P_T$ is unknown, we design an online ensemble algorithm with a meta-base structure, which is proved to achieve an $\widetilde{\mathcal{O}}\big(\sqrt{d^2 H^3K} + \sqrt{H^4(K+P_T)(1+P_T) + H^2 S_T^2}\big)$ dynamic regret and here $S_T$ is the expected switching number of the best base-learner. The result can be optimal under certain regimes.</details>

### Dynamic Sparsity Is Channel-Level Sparsity Learner

**Authors:** Lu Yin, Gen Li, Meng Fang, Li Shen, Tianjin Huang, Zhangyang "Atlas" Wang, Vlado Menkovski, Xiaolong Ma, Mykola Pechenizkiy, Shiwei Liu

**<details><summary>Abstract**</summary>

Sparse training has received an upsurging interest in machine learning due to its tantalizing saving potential for both the entire training process as well as the inference. Dynamic sparse training (DST) as a leading approach can train deep neural networks at high sparsity from scratch to match the performance of their dense counterparts. However, most if not all DST prior arts demonstrate their effectiveness on unstructured sparsity with highly irregular sparse patterns, which receives limited support in common hardware. This limitation hinders the usage of DST in practice. In this paper, we propose Channel-aware dynamic sparse (Chase), that for the first time seamlessly translates the promise of unstructured dynamic sparsity to GPU-friendly channel-level sparsity (not fine-grained N:M or group sparsity) during one end-to-end training process, without any ad-hoc operations. The resulting small sparse networks can be directly accelerated by commodity hardware, without using any particularly sparsity-aware hardware accelerators. This appealing outcome is partially motivated by a hidden phenomenon of dynamic sparsity: off-the-shelf unstructured DST implicitly involves biased parameter reallocation across channels, with a large fraction of channels (up to 60%) being sparser than others. By progressively identifying and removing these channels during training, our approach transfers unstructured sparsity to channel-wise sparsity. Our experimental results demonstrate that Chase achieves 1.7x inference throughput speedup on common GPU devices without compromising accuracy with ResNet-50 on ImageNet. We release our code in https://github.com/luuyin/chase.</details>

### Effective Targeted Attacks for Adversarial Self-Supervised Learning

**Authors:** Minseon Kim, Hyeonjeong Ha, Sooel Son, Sung Ju Hwang

**<details><summary>Abstract**</summary>

Recently, unsupervised adversarial training (AT) has been highlighted as a means of achieving robustness in models without any label information. Previous studies in unsupervised AT have mostly focused on implementing self-supervised learning (SSL) frameworks, which maximize the instance-wise classification loss to generate adversarial examples. However, we observe that simply maximizing the self-supervised training loss with an untargeted adversarial attack often results in generating ineffective adversaries that may not help improve the robustness of the trained model, especially for non-contrastive SSL frameworks without negative examples. To tackle this problem, we propose a novel positive mining for targeted adversarial attack to generate effective adversaries for adversarial SSL frameworks. Specifically, we introduce an algorithm that selects the most confusing yet similar target example for a given instance based on entropy and similarity, and subsequently perturbs the given instance towards the selected target. Our method demonstrates significant enhancements in robustness when applied to non-contrastive SSL frameworks, and less but consistent robustness improvements with contrastive SSL frameworks, on the benchmark datasets.</details>

### Effectively Learning Initiation Sets in Hierarchical Reinforcement Learning

**Authors:** Akhil Bagaria, Ben Abbatematteo, Omer Gottesman, Matt Corsaro, Sreehari Rammohan, George Konidaris

**<details><summary>Abstract**</summary>

An agent learning an option in hierarchical reinforcement learning must solve three problems: identify the option's subgoal (termination condition), learn a policy, and learn where that policy will succeed (initiation set). The termination condition is typically identified first, but the option policy and initiation set must be learned simultaneously, which is challenging because the initiation set depends on the option policy, which changes as the agent learns. Consequently, data obtained from option execution becomes invalid over time, leading to an inaccurate initiation set that subsequently harms downstream task performance. We highlight three issues---data non-stationarity, temporal credit assignment, and pessimism---specific to learning initiation sets, and propose to address them using tools from off-policy value estimation and classification. We show that our method learns higher-quality initiation sets faster than existing methods (in MiniGrid and Montezuma's Revenge), can automatically discover promising grasps for robot manipulation (in Robosuite), and improves the performance of a state-of-the-art option discovery method in a challenging maze navigation task in MuJoCo.</details>

### Efficient Adversarial Attacks on Online Multi-agent Reinforcement Learning

**Authors:** Guanlin Liu, Lifeng LAI

**<details><summary>Abstract**</summary>

Due to the broad range of applications of multi-agent reinforcement learning (MARL), understanding the effects of adversarial attacks against MARL model is essential for the safe applications of this model. Motivated by this, we investigate the impact of adversarial attacks on MARL. In the considered setup, there is an exogenous attacker who is able to modify the rewards before the agents receive them or manipulate the actions before the environment receives them. The attacker aims to guide each agent into a target policy or maximize the cumulative rewards under some specific reward function chosen by the attacker, while minimizing the amount of the manipulation on feedback and action. We first show the limitations of the action poisoning only attacks and the reward poisoning only attacks. We then introduce a mixed attack strategy with both the action poisoning and reward poisoning. We show that the mixed attack strategy can efficiently attack MARL agents even if the attacker has no prior information about the underlying environment and the agents’ algorithms.</details>

### Efficient Diffusion Policies For Offline Reinforcement Learning

**Authors:** Bingyi Kang, Xiao Ma, Chao Du, Tianyu Pang, Shuicheng Yan

**<details><summary>Abstract**</summary>

Offline reinforcement learning (RL) aims to learn optimal policies from offline datasets, where the parameterization of policies is crucial but often overlooked. Recently, Diffsuion-QL significantly boosts the performance of offline RL by representing a policy with a diffusion model, whose success relies on a parametrized Markov Chain with hundreds of steps for sampling. However, Diffusion-QL suffers from two critical limitations. 1) It is computationally inefficient to forward and backward through the whole Markov chain during training. 2) It is incompatible with maximum likelihood-based RL algorithms (e.g., policy gradient methods) as the likelihood of diffusion models is intractable. Therefore, we propose efficient diffusion policy (EDP) to overcome these two challenges. EDP approximately constructs actions from corrupted ones at training to avoid running the sampling chain. We conduct extensive experiments on the D4RL benchmark. The results show that EDP can reduce the diffusion policy training time from 5 days to 5 hours on gym-locomotion tasks. Moreover, we show that EDP is compatible with various offline RL algorithms (TD3, CRR, and IQL) and achieves new state-of-the-art on D4RL by large margins over previous methods.</details>

### Efficient Meta Neural Heuristic for Multi-Objective Combinatorial Optimization

**Authors:** Jinbiao Chen, Jiahai Wang, Zizhen Zhang, Zhiguang Cao, Te Ye, Siyuan Chen

**<details><summary>Abstract**</summary>

Recently, neural heuristics based on deep reinforcement learning have exhibited promise in solving multi-objective combinatorial optimization problems (MOCOPs). However, they are still struggling to achieve high learning efficiency and solution quality. To tackle this issue, we propose an efficient meta neural heuristic (EMNH), in which a meta-model is first trained and then fine-tuned with a few steps to solve corresponding single-objective subproblems. Specifically, for the training process, a (partial) architecture-shared multi-task model is leveraged to achieve parallel learning for the meta-model, so as to speed up the training; meanwhile, a scaled symmetric sampling method with respect to the weight vectors is designed to stabilize the training. For the fine-tuning process, an efficient hierarchical method is proposed to systematically tackle all the subproblems. Experimental results on the multi-objective traveling salesman problem (MOTSP), multi-objective capacitated vehicle routing problem (MOCVRP), and multi-objective knapsack problem (MOKP) show that, EMNH is able to outperform the state-of-the-art neural heuristics in terms of solution quality and learning efficiency, and yield competitive solutions to the strong traditional heuristics while consuming much shorter time.</details>

### [Spotlight] Efficient Online Clustering with Moving Costs

**Authors:** Dimitris Christou, Stratis Skoulakis, Volkan Cevher

**<details><summary>Abstract**</summary>

In this work we consider an online learning problem, called Online $k$-Clustering with Moving Costs, at which a learner maintains a set of $k$ facilities over $T$ rounds so as to minimize the connection cost of an adversarially selected sequence of clients. The learner is informed on the positions of the clients at each round $t$ only after its facility-selection and can use this information to update its decision in the next round. However, updating the facility positions comes with an additional moving cost based on the moving distance of the facilities. We present the first $\mathcal{O}(\log n)$-regret polynomial-time online learning algorithm guaranteeing that the overall cost (connection $+$ moving) is at most $\mathcal{O}(\log n)$ times the time-averaged connection cost of the best fixed solution. Our work improves on the recent result of (Fotakis et al., 2021) establishing $\mathcal{O}(k)$-regret guarantees only on the connection cost.</details>

### Efficient Potential-based Exploration in Reinforcement Learning using Inverse Dynamic Bisimulation Metric

**Authors:** Yiming Wang, Ming Yang, Renzhi Dong, Binbin Sun, Furui Liu, Leong Hou U

**<details><summary>Abstract**</summary>

Reward shaping is an effective technique for integrating domain knowledge into reinforcement learning (RL). However, traditional approaches like potential-based reward shaping totally rely on manually designing shaping reward functions, which significantly restricts exploration efficiency and introduces human cognitive biases.While a number of RL methods have been proposed to boost exploration by designing an intrinsic reward signal as exploration bonus. Nevertheless, these methods heavily rely on the count-based episodic term in their exploration bonus which falls short in scalability. To address these limitations, we propose a general end-to-end potential-based exploration bonus for deep RL via potentials of state discrepancy, which motivates the agent to discover novel states and provides them with denser rewards without manual intervention. Specifically, we measure the novelty of adjacent states by calculating their distance using the bisimulation metric-based potential function, which enhances agent's exploration and ensures policy invariance. In addition, we offer a theoretical guarantee on our inverse dynamic bisimulation metric, bounding the value difference and ensuring that the agent explores states with higher TD error, thus significantly improving training efficiency. The proposed approach is named \textbf{LIBERTY} (exp\textbf{L}oration v\textbf{I}a \textbf{B}isimulation m\textbf{E}t\textbf{R}ic-based s\textbf{T}ate discrepanc\textbf{Y}) which is comprehensively evaluated on the MuJoCo and the Arcade Learning Environments. Extensive experiments have verified the superiority and scalability of our algorithm compared with other competitive methods.</details>

### Efficient Uncertainty Quantification and Reduction for Over-Parameterized Neural Networks

**Authors:** Ziyi Huang, Henry Lam, Haofeng Zhang

**<details><summary>Abstract**</summary>

Uncertainty quantification (UQ) is important for reliability assessment and enhancement of machine learning models. In deep learning, uncertainties arise not only from data, but also from the training procedure that often injects substantial noises and biases. These hinder the attainment of statistical guarantees and, moreover, impose computational challenges on UQ due to the need for repeated network retraining. Building upon the recent neural tangent kernel theory, we create statistically guaranteed schemes to principally \emph{characterize}, and \emph{remove}, the uncertainty of over-parameterized neural networks with very low computation effort. In particular, our approach, based on what we call a procedural-noise-correcting (PNC) predictor, removes the procedural uncertainty by using only \emph{one} auxiliary network that is trained on a suitably labeled dataset, instead of many retrained networks employed in deep ensembles. Moreover, by combining our PNC predictor with suitable light-computation resampling methods, we build several approaches to construct asymptotically exact-coverage confidence intervals using as low as four trained networks without additional overheads.</details>

### Embracing the chaos: analysis and diagnosis of numerical instability in variational flows

**Authors:** Zuheng Xu, Trevor Campbell

**<details><summary>Abstract**</summary>

In this paper, we investigate the impact of numerical instability on the reliability of sampling, density evaluation, and evidence lower bound (ELBO) estimation in variational flows. We first empirically demonstrate that common flows can exhibit a catastrophic accumulation of error: the numerical flow map deviates significantly from the exact map---which affects sampling---and the numerical inverse flow map does not accurately recover the initial input---which affects density and ELBO computations. Surprisingly though, we find that results produced by flows are often accurate enough for applications despite the presence of serious numerical instability. In this work, we treat variational flows as chaotic dynamical systems, and leverage shadowing theory to elucidate this behavior via theoretical guarantees on the error of sampling, density evaluation, and ELBO estimation. Finally, we develop and empirically test a diagnostic procedure that can be used to validate results produced by numerically unstable flows in practice.</details>

### Embroid: Unsupervised Prediction Smoothing Can Improve Few-Shot Classification

**Authors:** Neel Guha, Mayee Chen, Kush Bhatia, Azalia Mirhoseini, Frederic Sala, Christopher Ré

**<details><summary>Abstract**</summary>

Recent work has shown that language models' (LMs) prompt-based learning capabilities make them well suited for automating data labeling in domains where manual annotation is expensive. The challenge is that while writing an initial prompt is cheap, improving a prompt is costly---practitioners often require significant labeled data in order to evaluate the impact of prompt modifications. Our work asks whether it is possible to improve prompt-based learning _without_ additional labeled data. We approach this problem by attempting to modify the predictions of a prompt, rather than the prompt itself. Our intuition is that accurate predictions should also be consistent: samples which are similar under some feature representation should receive the same prompt prediction. We propose Embroid, a method which computes multiple representations of a dataset under different embedding functions, and uses the consistency between the LM predictions for neighboring samples to identify mispredictions. Embroid then uses these neighborhoods to create additional predictions for each sample, and combines these predictions with a simple latent variable graphical model in order to generate a final corrected prediction. In addition to providing a theoretical analysis of Embroid, we conduct a rigorous empirical evaluation across six different LMs and up to 95 different tasks. We find that (1) Embroid substantially improves performance over original prompts (e.g., by an average of 7.3 points on GPT-JT), (2) also realizes improvements for more sophisticated prompting strategies (e.g., chain-of-thought), and (3) can be specialized to domains like law through the embedding functions.</details>

### [Oral] Emergence of Shape Bias in Convolutional Neural Networks through Activation Sparsity

**Authors:** Tianqin Li, Ziqi Wen, Yangfan Li, Tai Sing Lee

**Oral Presentation:** Tu, Dec 12, 14:25 -- Oral 2B

**<details><summary>Abstract**</summary>

Current deep-learning models for object recognition are known to be heavily biased toward texture. In contrast, human visual systems are known to be biased toward shape and structure. What could be the design principles in human visual systems that led to this difference? How could we introduce more shape bias into the deep learning models? In this paper, we report that sparse coding, a ubiquitous principle in the brain, can in itself introduce shape bias into the network. We found that enforcing the sparse coding constraint using a non-differential Top-K operation can lead to the emergence of structural encoding in neurons in convolutional neural networks, resulting in a smooth decomposition of objects into parts and subparts and endowing the networks with shape bias. We demonstrated this emergence of shape bias and its functional benefits for different network structures with various datasets. For object recognition convolutional neural networks, the shape bias leads to greater robustness against style and pattern change distraction. For the image synthesis generative adversary networks, the emerged shape bias leads to more coherent and decomposable structures in the synthesized images. Ablation studies suggest that sparse codes tend to encode structures, whereas the more distributed codes tend to favor texture. Our code is host at the github repository: \url{https://github.com/Crazy-Jack/nips2023_shape_vs_texture}</details>

### Energy Discrepancies: A Score-Independent Loss for Energy-Based Models

**Authors:** Tobias Schröder, Zijing Ou, Jen Lim, Yingzhen Li, Sebastian Vollmer, Andrew Duncan

**<details><summary>Abstract**</summary>

Energy-based models are a simple yet powerful class of probabilistic models, but their widespread adoption has been limited by the computational burden of training them. We propose a novel loss function called Energy Discrepancy (ED) which does not rely on the computation of scores or expensive Markov chain Monte Carlo. We show that energy discrepancy approaches the explicit score matching and negative log-likelihood loss under different limits, effectively interpolating between both. Consequently, minimum energy discrepancy estimation overcomes the problem of nearsightedness encountered in score-based estimation methods, while also enjoying theoretical guarantees. Through numerical experiments, we demonstrate that ED learns low-dimensional data distributions faster and more accurately than explicit score matching or contrastive divergence. For high-dimensional image data, we describe how the manifold hypothesis puts limitations on our approach and demonstrate the effectiveness of energy discrepancy by training the energy-based model as a prior of a variational decoder model.</details>

### Energy Guided Diffusion for Generating Neurally Exciting Images

**Authors:** Pawel Pierzchlewicz, Konstantin Willeke, Arne Nix, Pavithra Elumalai, Kelli Restivo, Tori Shinn, Cate Nealley, Gabrielle Rodriguez, Saumil Patel, Katrin Franke, Andreas Tolias, Fabian Sinz

**<details><summary>Abstract**</summary>

In recent years, most exciting inputs (MEIs) synthesized from encoding models of neuronal activity have become an established method for studying tuning properties of biological and artificial visual systems.    However, as we move up the visual hierarchy, the complexity of neuronal computations increases.     Consequently, it becomes more challenging to model neuronal activity, requiring more complex models.    In this study, we introduce a novel readout architecture inspired by the mechanism of visual attention. This new architecture, which we call attention readout, together with a data-driven convolutional core outperforms previous task-driven models in predicting the activity  of neurons in macaque area V4.    However, as our predictive network becomes deeper and more complex, synthesizing MEIs via straightforward gradient ascent (GA) can struggle to produce qualitatively good results and overfit to idiosyncrasies of a more complex model, potentially decreasing the MEI's model-to-brain transferability.    To solve this problem, we propose a diffusion-based method for generating MEIs via Energy Guidance (EGG).    We show that for models of macaque V4, EGG generates single neuron MEIs that generalize better across varying model architectures than the state-of-the-art GA, while at the same time reducing computational costs by a factor of 4.7x, facilitating experimentally challenging closed-loop experiments.    Furthermore, EGG diffusion can be used to generate other neurally exciting images, like most exciting naturalistic images that are on par with a selection of highly activating natural images, or image reconstructions that generalize better across architectures.    Finally, EGG is simple to implement, requires no retraining of the diffusion model, and can easily be generalized to provide other characterizations of the visual system, such as invariances.    Thus, EGG provides a general and flexible framework to study the coding properties of the visual system in the context of natural images.</details>

### Energy-Based Models for Anomaly Detection: A Manifold Diffusion Recovery Approach

**Authors:** Sangwoong Yoon, Young-Uk Jin, Yung-Kyun Noh, Frank Park

**<details><summary>Abstract**</summary>

We present a new method of training energy-based models (EBMs) for anomaly detection that leverages low-dimensional structures within data. The proposed algorithm, Manifold Projection-Diffusion Recovery (MPDR), first perturbs a data point along a low-dimensional manifold that approximates the training dataset. Then, EBM is trained to maximize the probability of recovering the original data. The training involves the generation of negative samples via MCMC, as in conventional EBM training, but from a different distribution concentrated near the manifold. The resulting near-manifold negative samples are highly informative, reflecting relevant modes of variation in data. An energy function of MPDR effectively learns accurate boundaries of the training data distribution and excels at detecting out-of-distribution samples. Experimental results show that MPDR exhibits strong performance across various anomaly detection tasks involving diverse data types, such as images, vectors, and acoustic signals.</details>

### Energy-Efficient Scheduling with Predictions

**Authors:** Eric Balkanski, Noemie Perivier, Clifford Stein, Hao-Ting Wei

**<details><summary>Abstract**</summary>

An  important goal of modern scheduling systems is to efficiently manage power usage. In energy-efficient scheduling,   the operating system controls the speed at which a machine is processing jobs  with the dual objective of  minimizing energy consumption and optimizing the quality of service cost of the resulting schedule. Since  machine-learned predictions  about future  requests can often be learned from historical data, a recent line of work  on learning-augmented algorithms aims to achieve improved performance guarantees by leveraging predictions.   In particular, for energy-efficient scheduling, Bamas et. al. [NeurIPS '20] and Antoniadis et. al. [SWAT '22]  designed algorithms with predictions for the  energy minimization with deadlines problem and achieved an improved competitive ratio when the prediction error is small while also maintaining  worst-case bounds even when the prediction error is arbitrarily large.In this paper, we consider a general setting for energy-efficient scheduling and provide a flexible learning-augmented algorithmic framework that takes as input an offline and an online algorithm for the desired energy-efficient scheduling problem. We show that, when the prediction error is small, this framework gives improved competitive ratios for many different energy-efficient scheduling problems, including  energy minimization with deadlines, while also maintaining a bounded competitive ratio regardless of the prediction error. Finally, we empirically demonstrate that this framework achieves an improved performance on real and synthetic datasets.</details>

### Enhancing CLIP with CLIP: Exploring Pseudolabeling for Limited-Label Prompt Tuning

**Authors:** Cristina Menghini, Andrew Delworth, Stephen Bach

**<details><summary>Abstract**</summary>

Fine-tuning vision-language models (VLMs) like CLIP to downstream tasks is often necessary to optimize their performance. However, a major obstacle is the limited availability of labeled data. We study the use of pseudolabels, i.e., heuristic labels for unlabeled data, to enhance CLIP via prompt tuning. Conventional pseudolabeling trains a model on labeled data and then generates labels for unlabeled data. VLMs' zero-shot capabilities enable a ``second generation'' of pseudolabeling approaches that do not require task-specific training on labeled data. By using zero-shot pseudolabels as a source of supervision, we observe that learning paradigms such as semi-supervised, transductive zero-shot, and unsupervised learning can all be seen as optimizing the same loss function. This unified view enables the development of versatile training strategies that are applicable across learning paradigms. We investigate them on image classification tasks where CLIP exhibits limitations, by varying prompt modalities, e.g., textual or visual prompts, and learning paradigms. We find that(1) unexplored prompt tuning strategies that iteratively refine pseudolabels consistently improve CLIP accuracy, by 19.5 points in semi-supervised learning, by 28.4 points in transductive zero-shot learning, and by 15.2 points in unsupervised learning, and (2) unlike conventional semi-supervised pseudolabeling, which exacerbates model biases toward classes with higher-quality pseudolabels, prompt tuning leads to a more equitable distribution of per-class accuracy. The code to reproduce the experiments is at https://github.com/BatsResearch/menghini-neurips23-code.</details>

### Enhancing User Intent Capture in Session-Based Recommendation with Attribute Patterns

**Authors:** Xin Liu, Zheng Li, Yifan Gao, Jingfeng Yang, Tianyu Cao, Zhengyang Wang, Bing Yin, Yangqiu Song

**<details><summary>Abstract**</summary>

The goal of session-based recommendation in E-commerce is to predict the next item that an anonymous user will purchase based on the browsing and purchase history. However, constructing global or local transition graphs to supplement session data can lead to noisy correlations and user intent vanishing. In this work, we propose the Frequent Attribute Pattern Augmented Transformer (FAPAT) that characterizes user intents by building attribute transition graphs and matching attribute patterns. Specifically, the frequent and compact attribute patterns are served as memory to augment session representations, followed by a gate and a transformer block to fuse the whole session information. Through extensive experiments on two public benchmarks and 100 million industrial data in three domains, we demonstrate that FAPAT consistently outperforms state-of-the-art methods by an average of 4.5% across various evaluation metrics (Hits, NDCG, MRR). Besides evaluating the next-item prediction, we estimate the models' capabilities to capture user intents via predicting items' attributes and period-item recommendations.</details>

### Estimating Propensity for Causality-based Recommendation without Exposure Data

**Authors:** Zhongzhou Liu, Yuan Fang, Min Wu

**<details><summary>Abstract**</summary>

Causality-based recommendation systems focus on the causal effects of user-item interactions resulting from item exposure (i.e., which items are recommended or exposed to the user), as opposed to conventional correlation-based recommendation. They are gaining popularity due to their multi-sided benefits to users, sellers and platforms alike. However, existing causality-based recommendation methods require additional input in the form of exposure data and/or propensity scores (i.e., the probability of exposure) for training. Such data, crucial for modeling causality in recommendation, are often not available in real-world situations due to technical or privacy constraints. In this paper, we bridge the gap by proposing a new framework, called Propensity Estimation for Causality-based Recommendation (PropCare). It can estimate the propensity and exposure from a more practical setup, where only interaction data are available *without* any ground truth on exposure or propensity in training and inference. We demonstrate that, by relating the pairwise characteristics between propensity and item popularity, PropCare enables competitive causality-based recommendation given only the conventional interaction data. We further present a theoretical analysis on the bias of the causal effect under our model estimation.  Finally, we empirically evaluate PropCare through both quantitative and qualitative experiments.</details>

### Every Parameter Matters: Ensuring the Convergence of Federated Learning with Dynamic Heterogeneous Models Reduction

**Authors:** Hanhan Zhou, Tian Lan, Guru Prasadh Venkataramani, Wenbo Ding

**<details><summary>Abstract**</summary>

Cross-device Federated Learning (FL) faces significant challenges where low-end clients that could potentially make unique contributions are excluded from training large models due to their resource bottlenecks. Recent research efforts have focused on model-heterogeneous FL, by extracting reduced-size models from the global model and applying them to local clients accordingly. Despite the empirical success, general theoretical guarantees of convergence on this method remain an open question. This paper presents a unifying framework for heterogeneous FL algorithms with online model extraction and provides a general convergence analysis for the first time. In particular, we prove that under certain sufficient conditions and for both IID and non-IID data, these algorithms converge to a stationary point of standard FL for general smooth cost functions. Moreover, we introduce the concept of minimum coverage index, together with model reduction noise, which will determine the convergence of heterogeneous federated learning, and therefore we advocate for a holistic approach that considers both factors to enhance the efficiency of heterogeneous federated learning.</details>

### Evolving Connectivity for Recurrent Spiking Neural Networks

**Authors:** Guan Wang, Yuhao Sun, Sijie Cheng, Sen Song

**<details><summary>Abstract**</summary>

Recurrent spiking neural networks (RSNNs) hold great potential for advancing artificial general intelligence, as they draw inspiration from the biological nervous system and show promise in modeling complex dynamics.However, the widely-used surrogate gradient-based training methods for RSNNs are inherently inaccurate and unfriendly to neuromorphic hardware.To address these limitations, we propose the evolving connectivity (EC) framework, an inference-only method for training RSNNs.The EC framework reformulates weight-tuning as a search into parameterized connection probability distributions, and employs Natural Evolution Strategies (NES) for optimizing these distributions.Our EC framework circumvents the need for gradients and features hardware-friendly characteristics, including sparse boolean connections and high scalability.We evaluate EC on a series of standard robotic locomotion tasks, where it achieves comparable performance with deep neural networks and outperforms gradient-trained RSNNs, even solving the complex 17-DoF humanoid task.Additionally, the EC framework demonstrates a two to three fold speedup in efficiency compared to directly evolving parameters.By providing a performant and hardware-friendly alternative, the EC framework lays the groundwork for further energy-efficient applications of RSNNs and advances the development of neuromorphic devices.Our code is publicly available at https://github.com/imoneoi/EvolvingConnectivity.</details>

### Evolving Standardization for Continual Domain Generalization over Temporal Drift

**Authors:** Mixue Xie, Shuang Li, Longhui Yuan, Chi Liu, Zehui Dai

**<details><summary>Abstract**</summary>

The capability of generalizing to out-of-distribution data is crucial for the deployment of machine learning models in the real world. Existing domain generalization (DG) mainly embarks on offline and discrete scenarios, where multiple source domains are simultaneously accessible and the distribution shift among domains is abrupt and violent. Nevertheless, such setting may not be universally applicable to all real-world applications, as there are cases where the data distribution gradually changes over time due to various factors, e.g., the process of aging. Additionally, as the domain constantly evolves, new domains will continually emerge. Re-training and updating models with both new and previous domains using existing DG methods can be resource-intensive and inefficient. Therefore, in this paper, we present a problem formulation for Continual Domain Generalization over Temporal Drift (CDGTD). CDGTD addresses the challenge of gradually shifting data distributions over time, where domains arrive sequentially and models can only access the data of the current domain. The goal is to generalize to unseen domains that are not too far into the future. To this end, we propose an Evolving Standardization (EvoS) method, which characterizes the evolving pattern of feature distribution and mitigates the distribution shift by standardizing features with generated statistics of corresponding domain. Specifically, inspired by the powerful ability of transformers to model sequence relations, we design a multi-scale attention module (MSAM) to learn the evolving pattern under sliding time windows of different lengths. MSAM can generate statistics of current domain based on the statistics of previous domains and the learned evolving pattern. Experiments on multiple real-world datasets including images and texts validate the efficacy of our EvoS.</details>

### ExPT: Synthetic Pretraining for Few-Shot Experimental Design

**Authors:** Tung Nguyen, Sudhanshu Agrawal, Aditya Grover

**<details><summary>Abstract**</summary>

Experimental design is a fundamental problem in many science and engineering fields. In this problem, sample efficiency is crucial due to the time, money, and safety costs of real-world design evaluations. Existing approaches either rely on active data collection or access to large, labeled datasets of past experiments, making them impractical in many real-world scenarios. In this work, we address the more challenging yet realistic setting of few-shot experimental design, where only a few labeled data points of input designs and their corresponding values are available. We approach this problem as a conditional generation task, where a model conditions on a few labeled examples and the desired output to generate an optimal input design. To this end, we introduce Experiment Pretrained Transformers (ExPT), a foundation model for few-shot experimental design that employs a novel combination of synthetic pretraining with in-context learning. In ExPT, we only assume knowledge of a finite collection of unlabelled data points from the input domain and pretrain a transformer neural network to optimize diverse synthetic functions defined over this domain. Unsupervised pretraining allows ExPT to adapt to any design task at test time in an in-context fashion by conditioning on a few labeled data points from the target task and generating the candidate optima. We evaluate ExPT on few-shot experimental design in challenging domains and demonstrate its superior generality and performance compared to existing methods. The source code is available at https://github.com/tung-nd/ExPT.git.</details>

### Exact Optimality of Communication-Privacy-Utility Tradeoffs in Distributed Mean Estimation

**Authors:** Berivan Isik, Wei-Ning Chen, Ayfer Ozgur, Tsachy Weissman, Albert No

**<details><summary>Abstract**</summary>

We study the mean estimation problem under communication and local differential privacy constraints. While previous work has proposed order-optimal algorithms for the same problem (i.e., asymptotically optimal as we spend more bits), exact optimality (in the non-asymptotic setting) still has not been achieved. In this work, we take a step towards characterizing the exact-optimal approach in the presence of shared randomness (a random variable shared between the server and the user) and identify several conditions for exact optimality. We prove that one of the conditions is to utilize a rotationally symmetric shared random codebook. Based on this, we propose a randomization mechanism where the codebook is a randomly rotated simplex -- satisfying the properties of the exact-optimal codebook. The proposed mechanism is based on a $k$-closest encoding which we prove to be exact-optimal for the randomly rotated simplex codebook.</details>

### Exact Verification of ReLU Neural Control Barrier Functions

**Authors:** Hongchao Zhang, Junlin Wu, Yevgeniy Vorobeychik, Andrew Clark

**<details><summary>Abstract**</summary>

Control Barrier Functions (CBFs) are a popular approach for safe control of nonlinear systems. In CBF-based control, the desired safety properties of the system are mapped to nonnegativity of a CBF, and the control input is chosen to ensure that the CBF remains nonnegative for all time. Recently, machine learning methods that represent CBFs as neural networks (neural control barrier functions, or NCBFs) have shown great promise due to the universal representability of neural networks. However, verifying that a learned CBF guarantees safety remains a challenging research problem. This paper presents novel exact conditions and algorithms for verifying safety of feedforward NCBFs with ReLU activation functions. The key challenge in doing so is that, due to the piecewise linearity of the ReLU function, the NCBF will be nondifferentiable at certain points, thus invalidating traditional safety verification methods that assume a smooth barrier function. We resolve this issue by leveraging a generalization of Nagumo's theorem for proving invariance of sets with nonsmooth boundaries to derive necessary and sufficient conditions for safety. Based on this condition, we propose an algorithm for safety verification of NCBFs that first decomposes the NCBF into piecewise linear segments and then solves a nonlinear program to verify safety of each segment as well as the intersections of the linear segments. We mitigate the complexity by only considering the boundary of the safe region and by pruning the segments with Interval Bound Propagation (IBP) and linear relaxation. We evaluate our approach through numerical studies with comparison to state-of-the-art SMT-based methods. Our code is available at https://github.com/HongchaoZhang-HZ/exactverif-reluncbf-nips23.</details>

### Explaining Predictive Uncertainty with Information Theoretic Shapley Values

**Authors:** David Watson, Joshua O'Hara, Niek Tax, Richard Mudd, Ido Guy

**<details><summary>Abstract**</summary>

Researchers in explainable artificial intelligence have developed numerous methods for helping users understand the predictions of complex supervised learning models. By contrast, explaining the $\textit{uncertainty}$ of model outputs has received relatively little attention. We adapt the popular Shapley value framework to explain various types of predictive uncertainty, quantifying each feature's contribution to the conditional entropy of individual model outputs. We consider games with modified characteristic functions and find deep connections between the resulting Shapley values and fundamental quantities from information theory and conditional independence testing. We outline inference procedures for finite sample error rate control with provable guarantees, and implement efficient algorithms that perform well in a range of experiments on real and simulated data. Our method has applications to covariate shift detection, active learning, feature selection, and active feature-value acquisition.</details>

### Explaining V1 Properties with a Biologically Constrained Deep Learning Architecture

**Authors:** Galen Pogoncheff, Jacob Granley, Michael Beyeler

**<details><summary>Abstract**</summary>

Convolutional neural networks (CNNs) have recently emerged as promising models of the ventral visual stream, despite their lack of biological specificity.While current state-of-the-art models of the primary visual cortex (V1) have surfaced from training with adversarial examples and extensively augmented data, these models are still unable to explain key neural properties observed in V1 that arise from biological circuitry.To address this gap, we systematically incorporated neuroscience-derived architectural components into CNNs to identify a set of mechanisms and architectures that more comprehensively explain V1 activity.Upon enhancing task-driven CNNs with architectural components that simulate center-surround antagonism, local receptive fields, tuned normalization, and cortical magnification, we uncover models with latent representations that yield state-of-the-art explanation of V1 neural activity and tuning properties.Moreover, analyses of the learned parameters of these components and stimuli that maximally activate neurons of the evaluated networks provide support for their role in explaining neural properties of V1.Our results highlight an important advancement in the field of NeuroAI, as we systematically establish a set of architectural components that contribute to unprecedented explanation of V1.The neuroscience insights that could be gleaned from increasingly accurate in-silico models of the brain have the potential to greatly advance the fields of both neuroscience and artificial intelligence.</details>

### Exploiting Connections between Lipschitz Structures for Certifiably Robust Deep Equilibrium Models

**Authors:** Aaron Havens, Alexandre Araujo, Siddharth Garg, Farshad Khorrami, Bin Hu

**<details><summary>Abstract**</summary>

Recently, deep equilibrium models (DEQs) have drawn increasing attention from the machine learning community. However, DEQs are much less understood in terms of certified robustness than their explicit network counterparts. In this paper, we advance the understanding of certified robustness of DEQs via exploiting the connections between various Lipschitz network parameterizations for both explicit and implicit models. Importantly, we show that various popular Lipschitz network structures, including convex potential layers (CPL), SDP-based Lipschitz layers (SLL), almost orthogonal layers (AOL), Sandwich layers, and monotone DEQs (MonDEQ) can all be reparameterized as special cases of the Lipschitz-bounded equilibrium networks (LBEN) without changing the prescribed Lipschitz constant in the original network parameterization. A key feature of our reparameterization technique is that it preserves the Lipschitz prescription used in different structures. This opens the possibility of achieving improved certified robustness of DEQs via a combination of network reparameterization, structure-preserving regularization, and LBEN-based fine-tuning. We also support our theoretical understanding with new empirical results, which show that our proposed method improves the certified robust accuracy of DEQs on classification tasks. All codes and experiments are made available at \url{https://github.com/AaronHavens/ExploitingLipschitzDEQ}.</details>

### Exploiting hidden structures in non-convex games for convergence to Nash equilibrium

**Authors:** Iosif Sakos, Panayotis Mertikopoulos, Georgios Piliouras

**<details><summary>Abstract**</summary>

A wide array of modern machine learning applications – from adversarial models to multi-agent reinforcement learning – can be formulated as non-cooperative games whose Nash equilibria represent the system’s desired operational states. Despite having a highly non-convex loss landscape, many cases of interest possess a latent convex structure that could potentially be leveraged to yield convergence to an equilibrium. Driven by this observation, our paper proposes a flexible first-order method that successfully exploits such “hidden structures” and achieves convergence under minimal assumptions for the transformation connecting the players’ control variables to the game’s latent, convex-structured layer. The proposed method – which we call preconditioned hidden gradient descent (PHGD) – hinges on a judiciously chosen gradient preconditioning scheme related to natural gradient methods. Importantly, we make no separability assumptions for the game’s hidden structure, and we provide explicit convergence rate guarantees for both deterministic and stochastic environments.</details>

### Explore to Generalize in Zero-Shot RL

**Authors:** Ev Zisselman, Itai Lavie, Daniel Soudry, Aviv Tamar

**<details><summary>Abstract**</summary>

We study zero-shot generalization in reinforcement learning - optimizing a policy on a set of training tasks to perform well on a similar but unseen test task. To mitigate overfitting, previous work explored different notions of invariance to the task. However, on problems such as the ProcGen Maze, an adequate solution that is invariant to the task visualization does not exist, and therefore invariance-based approaches fail. Our insight is that learning a policy that effectively $	extit{explores}$ the domain is harder to memorize than a policy that maximizes reward for a specific task, and therefore we expect such learned behavior to generalize well; we indeed demonstrate this empirically on several domains that are difficult for invariance-based approaches. Our $	extit{Explore to Generalize}$ algorithm (ExpGen) builds on this insight: we train an additional ensemble of agents that optimize reward. At test time, either the ensemble agrees on an action, and we generalize well, or we take exploratory actions, which generalize well and drive us to a novel part of the state space, where the ensemble may potentially agree again. We show that our approach is the state-of-the-art on tasks of the ProcGen challenge that have thus far eluded effective generalization, yielding a success rate of 83% on the Maze task and 74% on Heist with $200$ training levels. ExpGen can also be combined with an invariance based approach to gain the best of both worlds, setting new state-of-the-art results on ProcGen.Code available at [https://github.com/EvZissel/expgen](https://github.com/EvZissel/expgen).</details>

### [Spotlight] Exploring Loss Functions for Time-based Training Strategy in Spiking Neural Networks

**Authors:** Yaoyu Zhu, Wei Fang, Xiaodong Xie, Tiejun Huang, Zhaofei Yu

**<details><summary>Abstract**</summary>

Spiking Neural Networks (SNNs) are considered promising brain-inspired energy-efficient models due to their event-driven computing paradigm.The spatiotemporal spike patterns used to convey information in SNNs consist of both rate coding and temporal coding, where the temporal coding is crucial to biological-plausible learning rules such as spike-timing-dependent-plasticity.The time-based training strategy is proposed to better utilize the temporal information in SNNs and learn in an asynchronous fashion.However, some recent works train SNNs by the time-based scheme with rate-coding-dominated loss functions.In this paper, we first map rate-based loss functions to time-based counterparts and explain why they are also applicable to the time-based training scheme.After that, we infer that loss functions providing adequate positive overall gradients help training by theoretical analysis.Based on this, we propose the enhanced counting loss to replace the commonly used mean square counting loss.In addition, we transfer the training of scale factor in weight standardization into thresholds.Experiments show that our approach outperforms previous time-based training methods in most datasets. Our work provides insights for training SNNs with time-based schemes and offers a fresh perspective on the correlation between rate coding and temporal coding.Our code is available at https://github.com/zhuyaoyu/SNN-temporal-training-losses.</details>

### Exploring and Interacting with the Set of Good Sparse Generalized Additive Models

**Authors:** Chudi Zhong, Zhi Chen, Jiachang Liu, Margo Seltzer, Cynthia Rudin

**<details><summary>Abstract**</summary>

In real applications, interaction between machine learning models and domain experts is critical; however, the classical machine learning paradigm that usually produces only a single model does not facilitate such interaction. Approximating and exploring the Rashomon set, i.e., the set of all near-optimal models, addresses this practical challenge by providing the user with a searchable space containing a diverse set of models from which domain experts can choose. We present algorithms to efficiently and accurately approximate the Rashomon set of sparse, generalized additive models with ellipsoids for fixed support sets and use these ellipsoids to approximate Rashomon sets for many different support sets. The approximated Rashomon set serves as a cornerstone to solve practical challenges such as (1) studying the variable importance for the model class; (2) finding models under user-specified constraints (monotonicity, direct editing); and (3) investigating sudden changes in the shape functions. Experiments demonstrate the fidelity of the approximated Rashomon set and its effectiveness in solving practical challenges.</details>

### Exploring the Optimal Choice for Generative Processes in Diffusion Models: Ordinary vs Stochastic Differential Equations

**Authors:** Yu Cao, Jingrun Chen, Yixin Luo, Xiang ZHOU

**<details><summary>Abstract**</summary>

The diffusion model has shown remarkable success in computer vision, but it remains unclear whether the ODE-based probability flow or the SDE-based diffusion model is more superior and under what circumstances. Comparing the two is challenging due to dependencies on data distributions, score training, and other numerical issues. In this paper, we study the problem mathematically for two limiting scenarios: the zero diffusion (ODE) case and the large diffusion case. We first introduce a pulse-shape error to perturb the score function and analyze error accumulation of sampling quality, followed by a thorough analysis for generalization to arbitrary error.  Our findings indicate that when the perturbation occurs at the end of the generative process, the ODE model outperforms the SDE model with a large diffusion coefficient. However, when the perturbation occurs earlier, the SDE model outperforms the ODE model. We demonstrate that the error of sample generation due to the pulse-shape perturbation is exponentially suppressed as the diffusion term's magnitude increases to infinity. Numerical validation of this phenomenon is provided using Gaussian, Gaussian mixture, and Swiss roll distribution, as well as realistic datasets like MNIST and CIFAR-10.</details>

### Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models

**Authors:** George Stein, Jesse Cresswell, Rasa Hosseinzadeh, Yi Sui, Brendan Ross, Valentin Villecroze, Zhaoyan Liu, Anthony Caterini, Eric Taylor, Gabriel Loaiza-Ganem

**<details><summary>Abstract**</summary>

We systematically study a wide variety of generative models spanning semantically-diverse image datasets to understand and improve the feature extractors and metrics used to evaluate them.Using best practices in psychophysics, we measure human perception of image realism for generated samples by conducting the largest experiment evaluating generative models to date, and find that no existing metric strongly correlates with human evaluations.Comparing to 17 modern metrics for evaluating the overall performance, fidelity, diversity, rarity, and memorization of generative models, we find that the state-of-the-art perceptual realism of diffusion models as judged by humans is not reflected in commonly reported metrics such as FID. This discrepancy is not explained by diversity in generated samples, though one cause is over-reliance on Inception-V3.We address these flaws through a study of alternative self-supervised feature extractors, find that the semantic information encoded by individual networks strongly depends on their training procedure, and show that DINOv2-ViT-L/14 allows for much richer evaluation of generative models. Next, we investigate data memorization, and find that generative models do memorize training examples on simple, smaller datasets like CIFAR10, but not necessarily on more complex datasets like ImageNet. However, our experiments show that current metrics do not properly detect memorization: none in the literature is able to separate memorization from other phenomena such as underfitting or mode shrinkage. To facilitate further development of generative models and their evaluation we release all generated image datasets, human evaluation data, and a modular library to compute 17 common metrics for 9 different encoders at https://github.com/layer6ai-labs/dgm-eval.</details>

### FIRAL: An Active Learning Algorithm for Multinomial Logistic Regression

**Authors:** Youguang Chen, George Biros

**<details><summary>Abstract**</summary>

We investigate theory and algorithms for pool-based active learning for multiclass classification using multinomial logistic regression.  Using finite sample analysis, we prove that the Fisher Information Ratio (FIR)  lower and upper bounds  the excess risk. Based on our theoretical analysis, we propose an active learning algorithm that  employs regret minimization to minimize the FIR. To verify our derived excess risk bounds, we conduct experiments on synthetic datasets. Furthermore, we compare FIRAL with five other methods and found that our scheme  outperforms them: it consistently produces the smallest classification error in the multiclass logistic regression setting, as demonstrated through experiments on MNIST, CIFAR-10, and 50-class ImageNet.</details>

### FLSL: Feature-level Self-supervised Learning

**Authors:** Qing Su, Anton Netchaev, Hai Li, Shihao Ji

**<details><summary>Abstract**</summary>

Current self-supervised learning (SSL) methods (e.g., SimCLR, DINO, VICReg, MOCOv3) target primarily on representations at instance level and do not generalize well to dense prediction tasks, such as object detection and segmentation. Towards aligning SSL with dense predictions, this paper demonstrates for the first time the underlying mean-shift clustering process of Vision Transformers (ViT), which aligns well with natural image semantics (e.g., a world of objects and stuffs). By employing transformer for joint embedding and clustering, we propose a bi-level feature clustering SSL method, coined Feature-Level Self-supervised Learning (FLSL). We present the formal definition of the FLSL problem and construct the objectives from the mean-shift and k-means perspectives. We show that FLSL promotes remarkable semantic cluster representations and learns an embedding scheme amenable to intra-view and inter-view feature clustering. Experiments show that FLSL yields significant improvements in dense prediction tasks, achieving 44.9 (+2.8)% AP and 46.5% AP in object detection, as well as 40.8 (+2.3)% AP and 42.1% AP in instance segmentation on MS-COCO, using Mask R-CNN with ViT-S/16 and ViT-S/8 as backbone, respectively. FLSL consistently outperforms existing SSL methods across additional benchmarks, including UAV object detection on UAVDT, and video instance segmentation on DAVIS 2017. We conclude by presenting visualization and various ablation studies to better understand the success of FLSL. The source code is available at https://github.com/ISL-CV/FLSL.</details>

### Fairness Aware Counterfactuals for Subgroups

**Authors:** Loukas Kavouras, Konstantinos Tsopelas, Giorgos Giannopoulos, Dimitris Sacharidis, Eleni Psaroudaki, Nikolaos Theologitis, Dimitrios Rontogiannis, Dimitris Fotakis, Ioannis Emiris

**<details><summary>Abstract**</summary>

In this work, we present Fairness Aware Counterfactuals for Subgroups (FACTS), a framework for auditing subgroup fairness through counterfactual explanations. We start with revisiting (and generalizing) existing notions and introducing new, more refined notions of subgroup fairness. We aim to (a) formulate different aspects of the difficulty of individuals in certain subgroups to achieve recourse, i.e. receive the desired outcome, either at the micro level, considering members of the subgroup individually, or at the macro level, considering the subgroup as a whole, and (b) introduce notions of subgroup fairness that are robust, if not totally oblivious, to the cost of achieving recourse. We accompany these notions with an efficient, model-agnostic, highly parameterizable, and explainable framework for evaluating subgroup fairness. We demonstrate the advantages, the wide applicability, and the efficiency of our approach through a thorough experimental evaluation on different benchmark datasets.</details>

### Fantastic Robustness Measures: The Secrets of Robust Generalization

**Authors:** Hoki Kim, Jinseong Park, Yujin Choi, Jaewook Lee

**<details><summary>Abstract**</summary>

Adversarial training has become the de-facto standard method for improving the robustness of models against adversarial examples. However, robust overfitting remains a significant challenge, leading to a large gap between the robustness on the training and test datasets. To understand and improve robust generalization, various measures have been developed, including margin, smoothness, and flatness-based measures. In this study, we present a large-scale analysis of robust generalization to empirically verify whether the relationship between these measures and robust generalization remains valid in diverse settings. We demonstrate when and how these measures effectively capture the robust generalization gap by comparing over 1,300 models trained on CIFAR-10 under the $L_\infty$ norm and further validate our findings through an evaluation of more than 100 models from RobustBench across CIFAR-10, CIFAR-100, and ImageNet. We hope this work can help the community better understand adversarial robustness and motivate the development of more robust defense methods against adversarial attacks.</details>

### [Spotlight] Fast Approximation of Similarity Graphs with Kernel Density Estimation

**Authors:** Peter Macgregor, He Sun

**<details><summary>Abstract**</summary>

Constructing a similarity graph from a set $X$ of data points in $ \mathbb{R}^d$ is the first step of many modern clustering algorithms. However, typical constructions of a similarity graph have high time complexity, and a quadratic space dependency with respect to $|X|$. We address this limitation and present a new algorithmic framework that constructs a sparse approximation of the fully connected similarity graph while preserving its cluster structure. Our presented algorithm is based on the kernel density estimation problem, and is applicable for arbitrary kernel functions. We compare our designed algorithm with the  well-known implementations from the scikit-learn library and the FAISS library,  and find that our method significantly outperforms the implementation from both libraries on a variety of datasets.</details>

### Fast Bellman Updates for Wasserstein Distributionally Robust MDPs

**Authors:** Zhuodong Yu, Ling Dai, Shaohang Xu, Siyang Gao, Chin Pang Ho

**<details><summary>Abstract**</summary>

Markov decision processes (MDPs) often suffer from the sensitivity issue under model ambiguity. In recent years, robust MDPs have emerged as an effective framework to overcome this challenge. Distributionally robust MDPs extend the robust MDP framework by incorporating distributional information of the uncertain model parameters to alleviate the conservative nature of robust MDPs. This paper proposes a computationally efficient solution framework for solving distributionally robust MDPs with Wasserstein ambiguity sets. By exploiting the specific problem structure, the proposed framework decomposes the optimization problems associated with distributionally robust Bellman updates into smaller subproblems, which can be solved efficiently. The overall complexity of the proposed algorithm is quasi-linear in both the numbers of states and actions when the distance metric of the Wasserstein distance is chosen to be $L_1$, $L_2$, or $L_{\infty}$ norm, and so the computational cost of distributional robustness is substantially reduced. Our numerical experiments demonstrate that the proposed algorithms outperform other state-of-the-art solution methods.</details>

### Fast Projected Newton-like Method for Precision Matrix Estimation under Total Positivity

**Authors:** Jian-Feng CAI, Daniel Palomar, Jiaxi Ying

**<details><summary>Abstract**</summary>

We study the problem of estimating precision matrices in Gaussian distributions that are multivariate totally positive of order two ($\mathrm{MTP}_2$). The precision matrix in such a distribution is an M-matrix. This problem can be formulated as a sign-constrained log-determinant program. Current algorithms are designed using the block coordinate descent method or the proximal point algorithm, which becomes computationally challenging in high-dimensional cases due to the requirement to solve numerous nonnegative quadratic programs or large-scale linear systems. To address this issue, we propose a novel algorithm based on the two-metric projection method, incorporating a carefully designed search direction and variable partitioning scheme. Our algorithm substantially reduces computational complexity, and its theoretical convergence is established. Experimental results on synthetic and real-world datasets demonstrate that our proposed algorithm provides a significant improvement in computational efficiency compared to the state-of-the-art methods.</details>

### Fast Scalable and Accurate Discovery of DAGs Using the Best Order Score Search and Grow Shrink Trees

**Authors:** Bryan Andrews, Joseph Ramsey, Ruben Sanchez Romero, Jazmin Camchong, Erich Kummerfeld

**<details><summary>Abstract**</summary>

Learning graphical conditional independence structures is an important machine learning problem and a cornerstone of causal discovery. However, the accuracy and execution time of learning algorithms generally struggle to scale to problems with hundreds of highly connected variables---for instance, recovering brain networks from fMRI data. We introduce the best order score search (BOSS) and grow-shrink trees (GSTs) for learning directed acyclic graphs (DAGs) in this paradigm. BOSS greedily searches over permutations of variables, using GSTs to construct and score DAGs from permutations. GSTs efficiently cache scores to eliminate redundant calculations. BOSS achieves state-of-the-art performance in accuracy and execution time, comparing favorably to a variety of combinatorial and gradient-based learning algorithms under a broad range of conditions. To demonstrate its practicality, we apply BOSS to two sets of resting-state fMRI data: simulated data with pseudo-empirical noise distributions derived from randomized empirical fMRI cortical signals and clinical data from 3T fMRI scans processed into cortical parcels. BOSS is available for use within the TETRAD project which includes Python and R wrappers.</details>

### [Spotlight] Faster Margin Maximization Rates for Generic Optimization Methods

**Authors:** Guanghui Wang, Zihao Hu, Vidya Muthukumar, Jacob Abernethy

**<details><summary>Abstract**</summary>

First-order optimization methods tend to inherently favor certain solutions over others when minimizing a given training objective with multiple local optima. This phenomenon, known as \emph{implicit bias}, plays a critical role in understanding the generalization capabilities of optimization algorithms. Recent research has revealed that gradient-descent-based methods exhibit an implicit bias for the $\ell_2$-maximal margin classifier in the context of separable binary classification. In contrast, generic optimization methods, such as mirror descent and steepest descent, have been shown to converge to maximal margin classifiers defined by alternative geometries. However, while gradient-descent-based algorithms demonstrate fast implicit bias rates, the implicit bias rates of generic optimization methods have been relatively slow. To address this limitation, in this paper, we present a series of state-of-the-art implicit bias rates for mirror descent and steepest descent algorithms. Our primary technique involves transforming a generic optimization algorithm into an online learning dynamic that solves a regularized bilinear game, providing a unified framework for analyzing the implicit bias of various optimization methods. The accelerated rates are derived leveraging the regret bounds of online learning algorithms within this game framework.</details>

### FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning

**Authors:** Dipam Goswami, Yuyang Liu, Bartłomiej Twardowski, Joost van de Weijer

**<details><summary>Abstract**</summary>

Exemplar-free class-incremental learning (CIL) poses several challenges since it prohibits the rehearsal of data from previous tasks and thus suffers from catastrophic forgetting. Recent approaches to incrementally learning the classifier by freezing the feature extractor after the first task have gained much attention. In this paper, we explore prototypical networks for CIL, which generate new class prototypes using the frozen feature extractor and classify the features based on the Euclidean distance to the prototypes. In an analysis of the feature distributions of classes, we show that classification based on Euclidean metrics is successful for jointly trained features. However, when learning from non-stationary data, we observe that the Euclidean metric is suboptimal and that feature distributions are heterogeneous. To address this challenge, we revisit the anisotropic Mahalanobis distance for CIL. In addition, we empirically show that modeling the feature covariance relations is better than previous attempts at sampling features from normal distributions and training a linear classifier. Unlike existing methods, our approach generalizes to both many- and few-shot CIL settings, as well as to domain-incremental settings. Interestingly, without updating the backbone network, our method obtains state-of-the-art results on several standard continual learning benchmarks. Code is available at https://github.com/dipamgoswami/FeCAM.</details>

### Federated Learning with Manifold Regularization and Normalized Update Reaggregation

**Authors:** Xuming An, Li Shen, Han Hu, Yong Luo

**<details><summary>Abstract**</summary>

Federated Learning (FL) is an emerging collaborative machine learning framework where multiple clients train the global model without sharing their own datasets. In FL, the model inconsistency caused by the local data heterogeneity across clients results in the near-orthogonality of client updates, which leads to the global update norm reduction and slows down the convergence.  Most previous works focus on eliminating the difference of parameters (or gradients) between the local and global models, which may fail to reflect the model inconsistency due to the complex structure of the machine learning model and the Euclidean space's limitation in meaningful geometric representations.In this paper, we propose FedMRUR by adopting the manifold model fusion scheme and a new global optimizer to alleviate the negative impacts.Concretely, FedMRUR adopts a hyperbolic graph manifold regularizer enforcing the representations of the data in the local and global models are close to each other in a low-dimensional subspace. Because the machine learning model has the graph structure, the distance in hyperbolic space can reflect the model bias better than the Euclidean distance.In this way, FedMRUR exploits the manifold structures of the representations to significantly reduce the model inconsistency.FedMRUR also aggregates the client updates norms as the global update norm, which can appropriately enlarge each client's contribution to the global update, thereby mitigating the norm reduction introduced by the near-orthogonality of client updates.Furthermore, we theoretically prove that our algorithm can achieve a linear speedup property $\mathcal{O}(\frac{1}{\sqrt{SKT}})$ for non-convex setting under partial client participation, where $S$ is the participated clients number, $K$ is the local interval and $T$ is the total number of communication rounds.Experiments demonstrate that FedMRUR can achieve a new state-of-the-art (SOTA) accuracy with less communication.</details>

### Few-Shot Class-Incremental Learning via Training-Free Prototype Calibration

**Authors:** Qi-wei Wang, Da-Wei Zhou, Yi-Kai Zhang, De-Chuan Zhan, Han-Jia Ye

**<details><summary>Abstract**</summary>

Real-world scenarios are usually accompanied by continuously appearing classes with scare labeled samples, which require the machine learning model to incrementally learn new classes and maintain the knowledge of base classes. In this Few-Shot Class-Incremental Learning (FSCIL) scenario, existing methods either introduce extra learnable components or rely on a frozen feature extractor to mitigate catastrophic forgetting and overfitting problems. However, we find a tendency for existing methods to misclassify the samples of new classes into base classes, which leads to the poor performance of new classes. In other words, the strong discriminability of base classes distracts the classification of new classes. To figure out this intriguing phenomenon, we observe that although the feature extractor is only trained on base classes, it can surprisingly represent the *semantic similarity* between the base and *unseen* new classes. Building upon these analyses, we propose a *simple yet effective* Training-frEE calibratioN (TEEN) strategy to enhance the discriminability of new classes by fusing the new prototypes (i.e., mean features of a class) with weighted base prototypes. In addition to standard benchmarks in FSCIL, TEEN demonstrates remarkable performance and consistent improvements over baseline methods in the few-shot learning scenario. Code is available at: https://github.com/wangkiw/TEEN</details>

### Find What You Want: Learning Demand-conditioned Object Attribute Space for Demand-driven Navigation

**Authors:** Hongcheng Wang, Andy Guan Hong Chen, Xiaoqi Li, Mingdong Wu, Hao Dong

**<details><summary>Abstract**</summary>

The task of Visual Object Navigation (VON) involves an agent's ability to locate a particular object within a given scene. To successfully accomplish the VON task, two essential conditions must be fulfiled: 1) the user knows the name of the desired object; and 2) the user-specified object actually is present within the scene. To meet these conditions, a simulator can incorporate predefined object names and positions into the metadata of the scene. However, in real-world scenarios, it is often challenging to ensure that these conditions are always met. Humans in an unfamiliar environment may not know which objects are present in the scene, or they may mistakenly specify an object that is not actually present. Nevertheless, despite these challenges, humans may still have a demand for an object, which could potentially be fulfilled by other objects present within the scene in an equivalent manner. Hence, this paper proposes Demand-driven Navigation (DDN), which leverages the user's demand as the task instruction and prompts the agent to find an object which matches the specified demand. DDN aims to relax the stringent conditions of VON by focusing on fulfilling the user's demand rather than relying solely on specified object names. This paper proposes a method of acquiring textual attribute features of objects by extracting common sense knowledge from a large language model (LLM). These textual attribute features are subsequently aligned with visual attribute features using Contrastive Language-Image Pre-training (CLIP). Incorporating the visual attribute features as prior knowledge, enhances the navigation process. Experiments on AI2Thor with the ProcThor dataset demonstrate that the visual attribute features improve the agent's navigation performance and outperform the baseline methods commonly used in the VON and VLN task and methods with LLMs. The codes and demonstrations can be viewed at https://sites.google.com/view/demand-driven-navigation.</details>

### Finding Local Minima Efficiently in Decentralized Optimization

**Authors:** Wenhan Xian, Heng Huang

**<details><summary>Abstract**</summary>

In this paper we study the second-order optimality of decentralized stochastic algorithm that escapes saddle point efficiently for nonconvex optimization problems. We propose a new pure gradient-based decentralized stochastic algorithm PEDESTAL with a novel convergence analysis framework to address the technical challenges unique to the decentralized stochastic setting. Our method is the first decentralized stochastic algorithm to achieve second-order optimality with non-asymptotic analysis. We provide theoretical guarantees with the gradient complexity of $\tilde{O} (\epsilon^{-3})$ to find $O(\epsilon, \sqrt{\epsilon})$-second-order stationary point, which matches state-of-the-art results of centralized counterparts or decentralized methods to find first-order stationary point. We also conduct two decentralized tasks in our experiments, a matrix sensing task with synthetic data and a matrix factorization task with a real-world dataset to validate the performance of our method.</details>

### [Spotlight] Fine-Grained Human Feedback Gives Better Rewards for Language Model Training

**Authors:** Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj (Raj) Ammanabrolu, Noah Smith, Mari Ostendorf, Hannaneh Hajishirzi

**<details><summary>Abstract**</summary>

Language models (LMs) often exhibit undesirable text generation behaviors, including generating false, toxic, or irrelevant outputs. Reinforcement learning from human feedback (RLHF)---where human preference judgments on LM outputs are transformed into a learning signal---has recently shown promise in addressing these issues. However, such holistic feedback conveys limited information on long text outputs; it does not indicate which aspects of the outputs influenced user preference; e.g., which parts contain what type(s) of errors. In this paper, we use fine-grained human feedback (e.g., which sentence is false, which sub-sentence is irrelevant) as an explicit training signal. We introduce Fine-Grained RLHF, a framework that enables training and learning from reward functions that are fine-grained in two respects: (1) density, providing a reward after every segment (e.g., a sentence) is generated; and (2) incorporating multiple reward models associated with different feedback types (e.g., factual incorrectness, irrelevance, and information incompleteness). We conduct experiments on detoxification and long-form question answering to illustrate how learning with this reward function leads to improved performance, supported by both automatic and human evaluation. Additionally, we show that LM behaviors can be customized using different combinations of fine-grained reward models. We release all data, collected human feedback, and codes at https://FineGrainedRLHF.github.io.</details>

### Fine-Grained Theoretical Analysis of Federated Zeroth-Order Optimization

**Authors:** Jun Chen, Hong Chen, Bin Gu, Hao Deng

**<details><summary>Abstract**</summary>

Federated zeroth-order optimization (FedZO) algorithm enjoys the advantages of both zeroth-order optimization and federated learning, and has shown exceptional performance on black-box attack and softmax regression tasks. However, there is no generalization analysis for FedZO, and its analysis on computing convergence rate is slower than the corresponding first-order optimization setting. This paper aims to establish systematic theoretical assessments of FedZO by developing the analysis technique of on-average model stability. We establish the first generalization error bound of FedZO under the Lipschitz continuity and smoothness conditions. Then, refined generalization and optimization bounds are provided by replacing bounded gradient with heavy-tailed gradient noise and utilizing the second-order Taylor expansion for gradient approximation. With the help of a new error decomposition strategy, our theoretical analysis is also extended to the asynchronous case. For FedZO, our fine-grained analysis fills the theoretical gap on the generalization guarantees and polishes the convergence characterization of the computing algorithm.</details>

### Finite Population Regression Adjustment and Non-asymptotic Guarantees for Treatment Effect Estimation

**Authors:** Mehrdad Ghadiri, David Arbour, Tung Mai, Cameron Musco, Anup Rao

**<details><summary>Abstract**</summary>

The design and analysis of randomized experiments is fundamental to many areas, from the physical and social sciences to industrial settings. Regression adjustment is a popular technique to reduce the variance of estimates obtained from experiments, by utilizing information contained in auxiliary covariates. While there is a large literature within the statistics community studying various approaches to regression adjustment and their asymptotic properties, little focus has been given to approaches in the finite population setting with non-asymptotic accuracy bounds. Further, prior work typically assumes that an entire population is exposed to an experiment, whereas practitioners often seek to minimize the number of subjects exposed to an experiment, for ethical and pragmatic reasons.In this work, we study the problems of estimating the sample mean, individual treatment effects, and average treatment effect with regression adjustment. We propose approaches that use techniques from randomized numerical linear algebra to sample a subset of the population on which to perform an experiment. We give non-asymptotic accuracy bounds for our methods and demonstrate that they compare favorably with prior approaches.</details>

### Finite-Time Analysis of Single-Timescale Actor-Critic

**Authors:** Xuyang Chen, Lin Zhao

**<details><summary>Abstract**</summary>

Actor-critic methods have achieved significant success in many challenging applications. However, its finite-time convergence is still poorly understood in the most practical single-timescale form. Existing works on analyzing single-timescale actor-critic have been limited to i.i.d. sampling or tabular setting for simplicity. We investigate the more practical online single-timescale actor-critic algorithm on continuous state space, where the critic assumes linear function approximation and updates with a single Markovian sample per actor step.  Previous analysis has been unable to establish the convergence for such a challenging scenario. We demonstrate that the online single-timescale actor-critic method provably finds an $\epsilon$-approximate stationary point with $\widetilde{\mathcal{O}}(\epsilon^{-2})$ sample complexity under standard assumptions, which can be further improved to $\mathcal{O}(\epsilon^{-2})$ under the i.i.d. sampling. Our novel framework systematically evaluates and controls the error propagation between the actor and critic. It offers a promising approach for analyzing other single-timescale reinforcement learning algorithms as well.</details>

### Flow: Per-instance Personalized Federated Learning

**Authors:** Kunjal Panchal, Sunav Choudhary, Nisarg Parikh, Lijun Zhang, Hui Guan

**<details><summary>Abstract**</summary>

Federated learning (FL) suffers from data heterogeneity, where the diverse data distributions across clients make it challenging to train a single global model effectively. Existing personalization approaches aim to address the data heterogeneity issue by creating a personalized model for each client from the global model that fits their local data distribution. However, these personalized models may achieve lower accuracy than the global model in some clients, resulting in limited performance improvement compared to that without personalization. To overcome this limitation, we propose a per-instance personalization FL algorithm Flow. Flow creates dynamic personalized models that are adaptive not only to each client’s data distributions but also to each client’s data instances. The personalized model allows each instance to dynamically determine whether it prefers the local parameters or its global counterpart to make correct predictions, thereby improving clients’accuracy. We provide theoretical analysis on the convergence of Flow and empirically demonstrate the superiority of Flow in improving clients’ accuracy compared to state-of-the-art personalization approaches on both vision and language-based tasks.</details>

### Foundation Model is Efficient Multimodal Multitask Model Selector

**Authors:** fanqing meng, Wenqi Shao, zhanglin peng, Chonghe Jiang, Kaipeng Zhang, Yu Qiao, Ping Luo

**<details><summary>Abstract**</summary>

This paper investigates an under-explored but important problem: given a collection of pre-trained neural networks, predicting their performance on each multi-modal task without fine-tuning them, such as image recognition, referring, captioning, visual question answering, and text question answering.A brute-force approach is to finetune all models on all target datasets, bringing high computational costs. Although recent-advanced approaches employed lightweight metrics to measure models’ transferability, they often depend heavily on the prior knowledge of a single task, making them inapplicable in a multi-modal multi-task scenario. To tackle this issue, we propose an efficient multi-task model selector (EMMS), which employs large-scale foundation models to transform diverse label formats such as categories, texts, and bounding boxes of different downstream tasks into a unified noisy label embedding. EMMS can estimate a model’s transferability through a simple weighted linear regression, which can be efficiently solved by an alternating minimization algorithm with a convergence guarantee. Extensive experiments on 5 downstream tasks with 24 datasets show that EMMS is fast, effective, and generic enough to assess the transferability of pre-trained models, making it the first model selection method in the multi-task scenario. For instance, compared with the state- of-the-art method LogME enhanced by our label embeddings, EMMS achieves 9.0%, 26.3%, 20.1%, 54.8%, 12.2% performance gain on image recognition, referring, captioning, visual question answering, and text question answering, while bringing 5.13×, 6.29×, 3.59×, 6.19×, and 5.66× speedup in wall-clock time, respectively. The code is available at https://github.com/OpenGVLab/Multitask-Model-Selector.</details>

### From Cloze to Comprehension: Retrofitting Pre-trained Masked Language Models to Pre-trained Machine Reader

**Authors:** Weiwen Xu, Xin Li, Wenxuan Zhang, Meng Zhou, Wai Lam, Luo Si, Lidong Bing

**<details><summary>Abstract**</summary>

We present Pre-trained Machine Reader (PMR), a novel method for retrofitting pre-trained masked language models (MLMs) to pre-trained machine reading comprehension (MRC) models without acquiring labeled data.PMR can resolve the discrepancy between model pre-training and downstream fine-tuning of existing MLMs.To build the proposed PMR, we constructed a large volume of general-purpose and high-quality MRC-style training data by using Wikipedia hyperlinks and designed a Wiki Anchor Extraction task to guide the MRC-style pre-training.Apart from its simplicity, PMR effectively solves extraction tasks, such as Extractive Question Answering and Named Entity Recognition. PMR shows tremendous improvements over existing approaches, especially in low-resource scenarios.When applied to the sequence classification task in the MRC formulation, PMR enables the extraction of high-quality rationales to explain the classification process, thereby providing greater prediction explainability. PMR also has the potential to serve as a unified model for tackling various extraction and classification tasks in the MRC formulation.</details>

### [Spotlight] From Pixels to UI Actions: Learning to Follow Instructions via Graphical User Interfaces

**Authors:** Peter Shaw, Mandar Joshi, James Cohan, Jonathan Berant, Panupong Pasupat, Hexiang Hu, Urvashi Khandelwal, Kenton Lee, Kristina N Toutanova

**<details><summary>Abstract**</summary>

Much of the previous work towards digital agents for graphical user interfaces (GUIs) has relied on text-based representations (derived from HTML or other structured data sources), which are not always readily available. These input representations have been often coupled with custom, task-specific action spaces.  This paper focuses on creating agents that interact with the digital world using the same conceptual interface that humans commonly use — via pixel-based screenshots and a generic action space corresponding to keyboard and mouse actions. Building upon recent progress in pixel-based pretraining, we show, for the first time, that it is possible for such agents to outperform human crowdworkers on the MiniWob++ benchmark of GUI-based instruction following tasks.</details>

### GUST: Combinatorial Generalization by Unsupervised Grouping with Neuronal Coherence

**Authors:** Hao Zheng, Hui Lin, Rong Zhao

**<details><summary>Abstract**</summary>

Dynamically grouping sensory information into structured entities is essential for understanding the world of combinatorial nature. However, the grouping ability and therefore combinatorial generalization are still challenging artificial neural networks. Inspired by the evidence that successful grouping is indicated by neuronal coherence in the human brain, we introduce GUST (Grouping Unsupervisely by Spike Timing network), an iterative network architecture with biological constraints to bias the network towards a dynamical state of neuronal coherence that softly reflects the grouping information in the temporal structure of its spiking activity. We evaluate and analyze the model on synthetic datasets. Interestingly, the segregation ability is directly learned from superimposed stimuli with a succinct unsupervised objective. Two learning stages are present, from coarsely perceiving global features to additionally capturing local features. Further, the learned symbol-like building blocks can be systematically composed to represent novel scenes in a bio-plausible manner.</details>

### Gaussian Membership Inference Privacy

**Authors:** Tobias Leemann, Martin Pawelczyk, Gjergji Kasneci

**<details><summary>Abstract**</summary>

We propose a novel and practical privacy notion called $f$-Membership Inference Privacy ($f$-MIP), which explicitly considers the capabilities of realistic adversaries under the membership inference attack threat model.  Consequently, $f$-MIP offers interpretable privacy guarantees and improved utility (e.g., better classification accuracy). In particular, we derive a parametric family of $f$-MIP guarantees that we refer to as $\mu$-Gaussian Membership Inference Privacy ($\mu$-GMIP) by theoretically analyzing likelihood ratio-based membership inference attacks on stochastic gradient descent (SGD). Our analysis highlights that models trained with standard SGD already offer an elementary level of MIP.  Additionally, we show how $f$-MIP can be amplified by adding noise to gradient updates. Our analysis further yields an analytical membership inference attack that offers two distinct advantages over previous approaches. First, unlike existing state-of-the-art attacks that require training hundreds of shadow models, our attack does not require any shadow model.  Second, our analytical attack enables straightforward auditing of our privacy notion $f$-MIP. Finally, we quantify how various hyperparameters (e.g., batch size, number of model parameters) and specific data characteristics determine an attacker's ability to accurately infer a point's membership in the training set. We demonstrate the effectiveness of our method on models trained on vision and tabular datasets.</details>

### Gaussian Mixture Solvers for Diffusion Models

**Authors:** Hanzhong Guo, Cheng Lu, Fan Bao, Tianyu Pang, Shuicheng Yan, Chao Du, Chongxuan LI

**<details><summary>Abstract**</summary>

Recently, diffusion models have achieved great success in generative tasks. Sampling from diffusion models is equivalent to solving the reverse diffusion stochastic differential equations (SDEs) or the corresponding probability flow ordinary differential equations (ODEs). In comparison, SDE-based solvers can generate samples of higher quality and are suited for image translation tasks like stroke-based synthesis. During inference, however, existing SDE-based solvers are severely constrained by the efficiency-effectiveness dilemma. Our investigation suggests that this is because the Gaussian assumption in the reverse transition kernel is frequently violated (even in the case of simple mixture data) given a limited number of discretization steps. To overcome this limitation, we introduce a novel class of SDE-based solvers called \emph{Gaussian Mixture Solvers (GMS)} for diffusion models. Our solver estimates the first three-order moments and optimizes the parameters of a Gaussian mixture transition kernel using generalized methods of moments in each step during sampling. Empirically, our solver outperforms numerous SDE-based solvers in terms of sample quality in image generation and stroke-based synthesis in various diffusion models, which validates the motivation and effectiveness of GMS. Our code is available at https://github.com/Guohanzhong/GMS.</details>

### [Spotlight] Gaussian Partial Information Decomposition: Bias Correction and Application to High-dimensional Data

**Authors:** Praveen Venkatesh, Corbett Bennett, Sam Gale, Tamina Ramirez, Greggory Heller, Severine Durand, Shawn Olsen, Stefan Mihalas

**<details><summary>Abstract**</summary>

Recent advances in neuroscientific experimental techniques have enabled us to simultaneously record the activity of thousands of neurons across multiple brain regions. This has led to a growing need for computational tools capable of analyzing how task-relevant information is represented and communicated between several brain regions. Partial information decompositions (PIDs) have emerged as one such tool, quantifying how much unique, redundant and synergistic information two or more brain regions carry about a task-relevant message. However, computing PIDs is computationally challenging in practice, and statistical issues such as the bias and variance of estimates remain largely unexplored. In this paper, we propose a new method for efficiently computing and estimating a PID definition on multivariate Gaussian distributions. We show empirically that our method satisfies an intuitive additivity property, and recovers the ground truth in a battery of canonical examples, even at high dimensionality. We also propose and evaluate, for the first time, a method to correct the bias in PID estimates at finite sample sizes. Finally, we demonstrate that our Gaussian PID effectively characterizes inter-areal interactions in the mouse brain, revealing higher redundancy between visual areas when a stimulus is behaviorally relevant.</details>

### Gaussian Process Probes (GPP) for Uncertainty-Aware Probing

**Authors:** Zi Wang, Alexander Ku, Jason Baldridge, Tom Griffiths, Been Kim

**<details><summary>Abstract**</summary>

Understanding which concepts models can and cannot represent has been fundamental to many tasks: from effective and responsible use of models to detecting out of distribution data. We introduce Gaussian process probes (GPP), a unified and simple framework for probing and measuring uncertainty about concepts represented by models. As a Bayesian extension of linear probing methods, GPP asks what kind of distribution over classifiers (of concepts) is induced by the model. This distribution can be used to measure both what the model represents and how confident the probe is about what the model represents.  GPP can be applied to any pre-trained  model with vector representations of inputs (e.g., activations). It does not require access to training data, gradients, or the architecture. We validate GPP on datasets containing both synthetic and real images. Our experiments show it can (1) probe a model's representations of concepts even with a very small number of examples, (2) accurately measure both epistemic uncertainty (how confident the probe is) and aleatory uncertainty (how fuzzy the concepts are to the model), and (3) detect out of distribution data using those uncertainty measures as well as classic methods do. By using Gaussian processes to expand what probing can offer, GPP provides a data-efficient, versatile and uncertainty-aware tool for understanding and evaluating the capabilities of machine learning models.</details>

### [Spotlight] Generalization in the Face of Adaptivity: A Bayesian Perspective

**Authors:** Moshe Shenfeld, Katrina Ligett

**<details><summary>Abstract**</summary>

Repeated use of a data sample via adaptively chosen queries can rapidly lead to overfitting, wherein the empirical evaluation of queries on the sample significantly deviates from their mean with respect to the underlying data distribution. It turns out that simple noise addition algorithms suffice to prevent this issue, and differential privacy-based analysis of these algorithms shows that they can handle an asymptotically optimal number of queries.  However, differential privacy's worst-case nature entails scaling such noise to the range of the queries even for highly-concentrated queries, or introducing more complex algorithms.In this paper, we prove that straightforward noise-addition algorithms already provide variance-dependent guarantees that also extend to unbounded queries. This improvement stems from a novel characterization that illuminates the core problem of adaptive data analysis. We show that the harm of adaptivity results from the covariance between the new query and a Bayes factor-based measure of how much information about the data sample was encoded in the responses given to past queries. We then leverage this characterization to introduce a new data-dependent stability notion that can bound this covariance.</details>

### Generalized Logit Adjustment: Calibrating Fine-tuned Models by Removing Label Bias in Foundation Models

**Authors:** Beier Zhu, Kaihua Tang, QIANRU SUN, Hanwang Zhang

**<details><summary>Abstract**</summary>

Foundation models like CLIP allow zero-shot transfer on various tasks without additional training data. Yet, the zero-shot performance is less competitive than a fully supervised one. Thus, to enhance the performance, fine-tuning and ensembling are also commonly adopted to better fit the downstream tasks. However, we argue that such prior work has overlooked the inherent biases in foundation models. Due to the highly imbalanced Web-scale training set, these foundation models are inevitably skewed toward frequent semantics, and thus the subsequent fine-tuning or ensembling is still biased. In this study, we systematically examine the biases in foundation models and demonstrate the efficacy of our proposed Generalized Logit Adjustment (GLA) method. Note that bias estimation in foundation models is challenging, as most pre-train data cannot be explicitly assessed like in traditional long-tailed classification tasks.To this end, GLA has an optimization-based bias estimation approach for debiasing foundation models. As our work resolves a fundamental flaw in the pre-training, the proposed GLA demonstrates significant improvements across a diverse range of tasks: it achieves 1.5 pp accuracy gains on ImageNet, an large average improvement (1.4-4.6 pp) on 11 few-shot datasets, 2.4 pp gains on long-tailed classification. Codes are in https://github.com/BeierZhu/GLA.</details>

### Generalized Semi-Supervised Learning via Self-Supervised Feature Adaptation

**Authors:** Jiachen Liang, RuiBing Hou, Hong Chang, Bingpeng MA, Shiguang Shan, Xilin Chen

**<details><summary>Abstract**</summary>

Traditional semi-supervised learning (SSL) assumes that the feature distributions of labeled and unlabeled data are consistent which rarely holds in realistic scenarios. In this paper, we propose a novel SSL setting, where unlabeled samples are drawn from a mixed distribution that deviates from the feature distribution of labeled samples.Under this setting, previous SSL methods tend to predict wrong pseudo-labels with the model fitted on labeled data, resulting in noise accumulation. To tackle this issue, we propose \emph{Self-Supervised Feature Adaptation} (SSFA), a generic framework for improving SSL performance when labeled and unlabeled data come from different distributions. SSFA decouples the prediction of pseudo-labels from the current model to improve the quality of pseudo-labels. Particularly, SSFA incorporates a self-supervised task into the SSL framework and uses it to adapt the feature extractor of the model to the unlabeled data. In this way, the extracted features better fit the distribution of unlabeled data, thereby generating high-quality pseudo-labels. Extensive experiments show that our proposed SSFA is applicable to various pseudo-label-based SSL learners and significantly improves performance in labeled, unlabeled, and even unseen distributions.</details>

### Generalized equivalences between subsampling and ridge regularization

**Authors:** Pratik Patil, Jin-Hong Du

**<details><summary>Abstract**</summary>

We establish precise structural and risk equivalences between subsampling and ridge regularization for ensemble ridge estimators. Specifically, we prove that linear and quadratic functionals of subsample ridge estimators, when fitted with different ridge regularization levels $\lambda$ and subsample aspect ratios $\psi$, are asymptotically equivalent along specific paths in the $(\lambda,\psi)$-plane (where $\psi$ is the ratio of the feature dimension to the subsample size). Our results only require bounded moment assumptions on feature and response distributions and allow for arbitrary joint distributions. Furthermore, we provide a data-dependent method to determine the equivalent paths of $(\lambda,\psi)$. An indirect implication of our equivalences is that optimally tuned ridge regression exhibits a monotonic prediction risk in the data aspect ratio. This resolves a recent open problem raised by Nakkiran et al. for general data distributions under proportional asymptotics, assuming a mild regularity condition that maintains regression hardness through linearized signal-to-noise ratios.</details>

### Generative Modeling through the Semi-dual Formulation of Unbalanced Optimal Transport

**Authors:** Jaemoo Choi, Jaewoong Choi, Myungjoo Kang

**<details><summary>Abstract**</summary>

Optimal Transport (OT) problem investigates a transport map that bridges two distributions while minimizing a given cost function. In this regard, OT between tractable prior distribution and data has been utilized for generative modeling tasks. However, OT-based methods are susceptible to outliers and face optimization challenges during training. In this paper, we propose a novel generative model based on the semi-dual formulation of Unbalanced Optimal Transport (UOT). Unlike OT, UOT relaxes the hard constraint on distribution matching. This approach provides better robustness against outliers, stability during training, and faster convergence. We validate these properties empirically through experiments. Moreover, we study the theoretical upper-bound of divergence between distributions in UOT. Our model outperforms existing OT-based generative models, achieving FID scores of 2.97 on CIFAR-10 and 5.80 on CelebA-HQ-256. The code is available at \url{https://github.com/Jae-Moo/UOTM}.</details>

### GeoPhy: Differentiable Phylogenetic Inference via Geometric Gradients of Tree Topologies

**Authors:** Takahiro Mimori, Michiaki Hamada

**<details><summary>Abstract**</summary>

Phylogenetic inference, grounded in molecular evolution models, is essential for understanding the evolutionary relationships in biological data. Accounting for the uncertainty of phylogenetic tree variables, which include tree topologies and evolutionary distances on branches, is crucial for accurately inferring species relationships from molecular data and tasks requiring variable marginalization. Variational Bayesian methods are key to developing scalable, practical models; however, it remains challenging to conduct phylogenetic inference without restricting the combinatorially vast number of possible tree topologies. In this work, we introduce a novel, fully differentiable formulation of phylogenetic inference that leverages a unique representation of topological distributions in continuous geometric spaces. Through practical considerations on design spaces and control variates for gradient estimations, our approach, GeoPhy, enables variational inference without limiting the topological candidates. In experiments using real benchmark datasets, GeoPhy significantly outperformed other approximate Bayesian methods that considered whole topologies.</details>

### GeoTMI: Predicting Quantum Chemical Property with Easy-to-Obtain Geometry via Positional Denoising

**Authors:** Hyeonsu Kim, Jeheon Woo, SEONGHWAN KIM, Seokhyun Moon, Jun Hyeong Kim, Woo Youn Kim

**<details><summary>Abstract**</summary>

As quantum chemical properties have a dependence on their geometries, graph neural networks (GNNs) using 3D geometric information have achieved high prediction accuracy in many tasks. However, they often require 3D geometries obtained from high-level quantum mechanical calculations, which are practically infeasible, limiting their applicability to real-world problems. To tackle this, we propose a new training framework, GeoTMI, that employs denoising process to predict properties accurately using easy-to-obtain geometries (corrupted versions of correct geometries, such as those obtained from low-level calculations). Our starting point was the idea that the correct geometry is the best description of the target property. Hence, to incorporate information of the correct, GeoTMI aims to maximize mutual information between three variables: the correct and the corrupted geometries and the property. GeoTMI also explicitly updates the corrupted input to approach the correct geometry as it passes through the GNN layers, contributing to more effective denoising. We investigated the performance of the proposed method using 3D GNNs for three prediction tasks: molecular properties, a chemical reaction property, and relaxed energy in a heterogeneous catalytic system. Our results showed consistent improvements in accuracy across various tasks, demonstrating the effectiveness and robustness of GeoTMI.</details>

### Geometric Transformer with Interatomic Positional Encoding

**Authors:** Yusong Wang, Shaoning Li, Tong Wang, Bin Shao, Nanning Zheng, Tie-Yan Liu

**<details><summary>Abstract**</summary>

The widespread adoption of Transformer architectures in various data modalities has opened new avenues for the applications in molecular modeling. Nevertheless, it remains elusive that whether the Transformer-based architecture can do molecular modeling as good as equivariant GNNs. In this paper, by designing Interatomic Positional Encoding (IPE) thatparameterizes atomic environments as Transformer's positional encodings,we propose Geoformer, a novel geometric Transformer to effectively model molecular structures for various molecular property prediction. We evaluate Geoformer on several benchmarks, including the QM9 dataset and the recently proposed Molecule3D dataset. Compared with both Transformers and equivariant GNN models, Geoformer outperforms the state-of-the-art (SoTA) algorithms on QM9, and achieves the best performance on Molecule3D for both random and scaffold splits.By introducing IPE, Geoformer paves the way for molecular geometric modeling based on Transformer architecture.</details>

### Global Optimality in Bivariate Gradient-based DAG Learning

**Authors:** Chang Deng, Kevin Bello, Pradeep Ravikumar, Bryon Aragam

**<details><summary>Abstract**</summary>

Recently, a new class of non-convex optimization problems motivated by the statistical problem of learning an acyclic directed graphical model from data has attracted significant interest. While existing work uses standard first-order optimization schemes to solve this problem, proving the global optimality of such approaches has proven elusive. The difficulty lies in the fact that unlike other non-convex problems in the literature, this problem is not "benign", and possesses multiple spurious solutions that standard approaches can easily get trapped in. In this paper, we prove that a simple path-following optimization scheme globally converges to the global minimum of the population loss in the bivariate setting.</details>

### Globally injective and bijective neural operators

**Authors:** Takashi Furuya, Michael Puthawala, Matti Lassas, Maarten V. de Hoop

**<details><summary>Abstract**</summary>

Recently there has been great interest in operator learning, where networks learn operators between function spaces from an essentially infinite-dimensional perspective. In this work we present results for when the operators learned by these networks are injective and surjective. As a warmup, we combine prior work in both the finite-dimensional ReLU and operator learning setting by giving sharp conditions under which ReLU layers with linear neural operators are injective. We then consider the case when the activation function is pointwise bijective and obtain sufficient conditions for the layer to be injective. We remark that this question, while trivial in the finite-rank setting, is subtler in the infinite-rank setting and is proven using tools from Fredholm theory. Next, we prove that our supplied injective neural operators are universal approximators and that their implementation, with finite-rank neural networks, are still injective. This ensures that injectivity is not 'lost' in the transcription from analytical operators to their finite-rank implementation with networks. Finally, we conclude with an increase in abstraction and consider general conditions when subnetworks, which may have many layers, are injective and surjective and provide an exact inversion from a 'linearization.’ This section uses general arguments from Fredholm theory and Leray-Schauder degree theory for non-linear integral equations to analyze the mapping properties of neural operators in function spaces. These results apply to subnetworks formed from the layers considered in this work, under natural conditions. We believe that our work has applications in Bayesian uncertainty quantification where injectivity enables likelihood estimation and in inverse problems where surjectivity and injectivity corresponds to existence and uniqueness of the solutions, respectively.</details>

### Gradient-Free Kernel Stein Discrepancy

**Authors:** Matthew Fisher, Chris Oates

**<details><summary>Abstract**</summary>

Stein discrepancies have emerged as a powerful statistical tool, being applied to fundamental statistical problems including parameter inference, goodness-of-fit testing, and sampling.  The canonical Stein discrepancies require the derivatives of a statistical model to be computed, and in return provide theoretical guarantees of convergence detection and control.  However, for complex statistical models, the stable numerical computation of derivatives can require bespoke algorithmic development and render Stein discrepancies impractical.  This paper focuses on posterior approximation using Stein discrepancies, and introduces a collection of non-canonical Stein discrepancies that are gradient-free, meaning that derivatives of the statistical model are not required.  Sufficient conditions for convergence detection and control are established, and applications to sampling and variational inference are presented.</details>

### Grammar Prompting for Domain-Specific Language Generation with  Large Language Models

**Authors:** Bailin Wang, Zi Wang, Xuezhi Wang, Yuan Cao, Rif A. Saurous, Yoon Kim

**<details><summary>Abstract**</summary>

Large language models (LLMs)  can learn to perform a wide range of natural language tasks from just a  handful of in-context examples. However, for generating strings from highly structured  languages (e.g., semantic parsing to complex domain-specific languages), it is challenging for the LLM to generalize from just a few exemplars. We propose \emph{grammar prompting}, a simple approach to enable LLMs to use external knowledge and domain-specific constraints, expressed through a grammar in Backus--Naur Form (BNF), during in-context learning. Grammar prompting augments each demonstration example with a specialized grammar that is minimally sufficient for generating the particular output example, where the specialized grammar is a subset of the full DSL grammar.  For inference, the LLM first predicts a BNF grammar given a test input, and then generates the output according to the rules of the grammar. Experiments demonstrate that grammar prompting can enable LLMs to perform competitively on a diverse set of DSL generation tasks, including semantic parsing (SMCalFlow, Overnight, GeoQuery), PDDL planning, and SMILES-based molecule generation.</details>

### Graph Contrastive Learning with Stable and Scalable Spectral Encoding

**Authors:** Deyu Bo, Yuan Fang, Yang Liu, Chuan Shi

**<details><summary>Abstract**</summary>

Graph contrastive learning (GCL) aims to learn representations by capturing the agreements between different graph views. Traditional GCL methods generate views in the spatial domain, but it has been recently discovered that the spectral domain also plays a vital role in complementing spatial views. However, existing spectral-based graph views either ignore the eigenvectors that encode valuable positional information or suffer from high complexity when trying to address the instability of spectral features. To tackle these challenges, we first design an informative, stable, and scalable spectral encoder, termed EigenMLP, to learn effective representations from the spectral features. Theoretically, EigenMLP is invariant to the rotation and reflection transformations on eigenvectors and robust against perturbations. Then, we propose a spatial-spectral contrastive framework (Sp$^{2}$GCL) to capture the consistency between the spatial information encoded by graph neural networks and the spectral information learned by EigenMLP, thus effectively fusing these two graph views. Experiments on the node- and graph-level datasets show that our method not only learns effective graph representations but also achieves a 2--10x speedup over other spectral-based methods.</details>

### Greedy Pruning with Group Lasso Provably Generalizes for Matrix Sensing

**Authors:** Nived Rajaraman, Fnu Devvrit, Aryan Mokhtari, Kannan Ramchandran

**<details><summary>Abstract**</summary>

Pruning schemes have been widely used in practice to reduce the complexity of trained models with a massive number of parameters. In fact, several practical studies have shown that if the pruned model is fine-tuned with some gradient-based updates it generalizes well to new samples. Although the above pipeline, which we refer to as pruning + fine-tuning, has been extremely successful in lowering the complexity of trained models, there is very little known about the theory behind this success. In this paper we address this issue by investigating the pruning + fine-tuning framework on the overparameterized matrix sensing problem with the ground truth denoted $U_\star \in \mathbb{R}^{d \times r}$ and the overparameterized model $U \in \mathbb{R}^{d \times k}$ with $k \gg r$. We study the approximate local minima of the mean square error, augmented with a smooth version of a group Lasso regularizer, $\sum_{i=1}^{k} \lVert Ue_i \rVert_2 $. In particular, we provably show that pruning all the columns below a certain explicit $\ell_2$-norm threshold results in a solution $U_{\text{prune}}$ which has the minimum number of columns $r$, yet close to the ground truth in training loss. Moreover, in the subsequent fine-tuning phase, gradient descent initialized at $U_{\text{prune}}$ converges at a linear rate to its limit. While our analysis provides insights into the role of regularization in pruning, we also show that running gradient descent in the absence of regularization results in models which {are not suitable for greedy pruning}, i.e., many columns could have their $\ell_2$ norm comparable to that of the maximum. Lastly, we show that our results also extend for the training and pruning of two-layer neural networks with quadratic activation functions. To the best of our knowledge, our results provide the first rigorous insights on why greedy pruning + fine-tuning leads to smaller models which also generalize well.</details>

### Group Robust Classification Without Any Group Information

**Authors:** Christos Tsirigotis, Joao Monteiro, Pau Rodriguez, David Vazquez, Aaron Courville

**<details><summary>Abstract**</summary>

Empirical risk minimization (ERM) is sensitive to spurious correlations present in training data, which poses a significant risk when deploying systems trained under this paradigm in high-stake applications. While the existing literature focuses on maximizing group-balanced or worst-group accuracy, estimating these quantities is hindered by costly bias annotations. This study contends that current bias-unsupervised approaches to group robustness continue to rely on group information to achieve optimal performance. Firstly, these methods implicitly assume that all group combinations are represented during training. To illustrate this, we introduce a systematic generalization task on the MPI3D dataset and discover that current algorithms fail to improve the ERM baseline when combinations of observed attribute values are missing. Secondly, bias labels are still crucial for effective model selection, restricting the practicality of these methods in real-world scenarios. To address these limitations, we propose a revised methodology for training and validating debiased models in an entirely bias-unsupervised manner. We achieve this by employing pretrained self-supervised models to reliably extract bias information, which enables the integration of a logit adjustment training loss with our validation criterion. Our empirical analysis on synthetic and real-world tasks provides evidence that our approach overcomes the identified challenges and consistently enhances robust accuracy, attaining performance which is competitive with or outperforms that of state-of-the-art methods, which, conversely, rely on bias labels for validation.</details>

### HeadSculpt: Crafting 3D Head Avatars with Text

**Authors:** Xiao Han, Yukang Cao, Kai Han, Xiatian Zhu, Jiankang Deng, Yi-Zhe Song, Tao Xiang, Kwan-Yee K. Wong

**<details><summary>Abstract**</summary>

Recently, text-guided 3D generative methods have made remarkable advancements in producing high-quality textures and geometry, capitalizing on the proliferation of large vision-language and image diffusion models. However, existing methods still struggle to create high-fidelity 3D head avatars in two aspects: (1) They rely mostly on a pre-trained text-to-image diffusion model whilst missing the necessary 3D awareness and head priors. This makes them prone to inconsistency and geometric distortions in the generated avatars. (2) They fall short in fine-grained editing. This is primarily due to the inherited limitations from the pre-trained 2D image diffusion models, which become more pronounced when it comes to 3D head avatars. In this work, we address these challenges by introducing a versatile coarse-to-fine pipeline dubbed HeadSculpt for crafting (i.e., generating and editing) 3D head avatars from textual prompts. Specifically, we first equip the diffusion model with 3D awareness by leveraging landmark-based control and a learned textual embedding representing the back view appearance of heads, enabling 3D-consistent head avatar generations. We further propose a novel identity-aware editing score distillation strategy to optimize a textured mesh with a high-resolution differentiable rendering technique. This enables identity preservation while following the editing instruction.We showcase HeadSculpt's superior fidelity and editing capabilities through comprehensive experiments and comparisons with existing methods.</details>

### HiBug: On Human-Interpretable Model Debug

**Authors:** Muxi Chen, YU LI, Qiang Xu

**<details><summary>Abstract**</summary>

Machine learning models can frequently produce systematic errors on critical subsets (or slices) of data that share common attributes. Discovering and explaining such model bugs is crucial for reliable model deployment. However, existing bug discovery and interpretation methods usually involve heavy human intervention and annotation, which can be cumbersome and have low bug coverage.In this paper, we propose HiBug, an automated framework for interpretable model debugging. Our approach utilizes large pre-trained models, such as chatGPT, to suggest human-understandable attributes that are related to the targeted computer vision tasks. By leveraging pre-trained vision-language models, we can efficiently identify common visual attributes of underperforming data slices using human-understandable terms. This enables us to uncover rare cases in the training data, identify spurious correlations in the model, and use the interpretable debug results to select or generate new training data for model improvement. Experimental results demonstrate the efficacy of the HiBug framework.</details>

### HiNeRV: Video Compression with Hierarchical Encoding-based Neural Representation

**Authors:** Ho Man Kwan, Ge Gao, Fan Zhang, Andrew Gower, David Bull

**<details><summary>Abstract**</summary>

Learning-based video compression is currently a popular research topic, offering the potential to compete with conventional standard video codecs. In this context, Implicit Neural Representations (INRs) have previously been used to represent and compress image and video content, demonstrating relatively high decoding speed compared to other methods. However, existing INR-based methods have failed to deliver rate quality performance comparable with the state of the art in video compression. This is mainly due to the simplicity of the employed network architectures, which limit their representation capability. In this paper, we propose HiNeRV, an INR that combines light weight layers with novel hierarchical positional encodings. We employs depth-wise convolutional, MLP and interpolation layers to build the deep and wide network architecture with high capacity. HiNeRV is also a unified representation encoding videos in both frames and patches at the same time, which offers higher performance and flexibility than existing methods. We further build a video codec based on HiNeRV and a refined pipeline for training, pruning and quantization that can better preserve HiNeRV's performance during lossy model compression. The proposed method has been evaluated on both UVG and MCL-JCV datasets for video compression, demonstrating significant improvement over all existing INRs baselines and competitive performance when compared to learning-based codecs (72.3\% overall bit rate saving over HNeRV and 43.4\% over DCVC on the UVG dataset, measured in PSNR).</details>

### [Spotlight] Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality

**Authors:** Liyuan Wang, Jingyi Xie, Xingxing Zhang, Mingyi Huang, Hang Su, Jun Zhu

**<details><summary>Abstract**</summary>

Prompt-based continual learning is an emerging direction in leveraging pre-trained knowledge for downstream continual learning, and has almost reached the performance pinnacle under supervised pre-training. However, our empirical research reveals that the current strategies fall short of their full potential under the more realistic self-supervised pre-training, which is essential for handling vast quantities of unlabeled data in practice. This is largely due to the difficulty of task-specific knowledge being incorporated into instructed representations via prompt parameters and predicted by uninstructed representations at test time. To overcome the exposed sub-optimality, we conduct a theoretical analysis of the continual learning objective in the context of pre-training, and decompose it into hierarchical components: within-task prediction, task-identity inference, and task-adaptive prediction. Following these empirical and theoretical insights, we propose Hierarchical Decomposition (HiDe-)Prompt, an innovative approach that explicitly optimizes the hierarchical components with an ensemble of task-specific prompts and statistics of both uninstructed and instructed representations, further with the coordination of a contrastive regularization strategy. Our extensive experiments demonstrate the superior performance of HiDe-Prompt and its robustness to pre-training paradigms in continual learning (e.g., up to 15.01% and 9.61% lead on Split CIFAR-100 and Split ImageNet-R, respectively).</details>

### Hierarchical Randomized Smoothing

**Authors:** Yan Scholten, Jan Schuchardt, Aleksandar Bojchevski, Stephan Günnemann

**<details><summary>Abstract**</summary>

Real-world data is complex and often consists of objects that can be decomposed into multiple entities (e.g. images into pixels, graphs into interconnected nodes). Randomized smoothing is a powerful framework for making models provably robust against small changes to their inputs - by guaranteeing robustness of the majority vote when randomly adding noise before classification. Yet, certifying robustness on such complex data via randomized smoothing is challenging when adversaries do not arbitrarily perturb entire objects (e.g. images) but only a subset of their entities (e.g. pixels). As a solution, we introduce hierarchical randomized smoothing: We partially smooth objects by adding random noise only on a randomly selected subset of their entities. By adding noise in a more targeted manner than existing methods we obtain stronger robustness guarantees while maintaining high accuracy. We initialize hierarchical smoothing using different noising distributions, yielding novel robustness certificates for discrete and continuous domains. We experimentally demonstrate the importance of hierarchical smoothing in image and node classification, where it yields superior robustness-accuracy trade-offs. Overall, hierarchical smoothing is an important contribution towards models that are both - certifiably robust to perturbations and accurate.</details>

### Hierarchical VAEs provide a normative account of motion processing in the primate brain

**Authors:** Hadi Vafaii, Jacob Yates, Daniel Butts

**<details><summary>Abstract**</summary>

The relationship between perception and inference, as postulated by Helmholtz in the 19th century, is paralleled in modern machine learning by generative models like Variational Autoencoders (VAEs) and their hierarchical variants. Here, we evaluate the role of hierarchical inference and its alignment with brain function in the domain of motion perception. We first introduce a novel synthetic data framework, Retinal Optic Flow Learning (ROFL), which enables control over motion statistics and their causes. We then present a new hierarchical VAE and test it against alternative models on two downstream tasks: (i) predicting ground truth causes of retinal optic flow (e.g., self-motion); and (ii) predicting the responses of neurons in the motion processing pathway of primates. We manipulate the model architectures (hierarchical versus non-hierarchical), loss functions, and the causal structure of the motion stimuli. We find that hierarchical latent structure in the model leads to several improvements. First, it improves the linear decodability of ground truth variables and does so in a sparse and disentangled manner. Second, our hierarchical VAE outperforms previous state-of-the-art models in predicting neuronal responses and exhibits sparse latent-to-neuron relationships. These results depend on the causal structure of the world, indicating that alignment between brains and artificial neural networks depends not only on architecture but also on matching ecologically relevant stimulus statistics. Taken together, our results suggest that hierarchical Bayesian inference underlines the brain's understanding of the world, and hierarchical VAEs can effectively model this understanding.</details>

### [Spotlight] Hierarchical clustering with dot products recovers hidden tree structure

**Authors:** Annie Gray, Alexander Modell, Patrick Rubin-Delanchy, Nick Whiteley

**<details><summary>Abstract**</summary>

In this paper we offer a new perspective on the well established agglomerative clustering algorithm, focusing on recovery of hierarchical structure. We recommend a simple variant of the standard algorithm, in which clusters are merged by maximum average dot product and not, for example, by minimum distance or within-cluster variance. We demonstrate that the tree output by this algorithm provides a bona fide estimate of generative hierarchical structure in data, under a generic probabilistic graphical model. The key technical innovations are to understand how hierarchical information in this model translates into tree geometry which can be recovered from data, and to characterise the benefits of simultaneously growing sample size and data dimension. We demonstrate superior tree recovery performance with real data over existing approaches such as UPGMA, Ward's method, and HDBSCAN.</details>

### High Precision Causal Model Evaluation with Conditional Randomization

**Authors:** Chao Ma, Cheng Zhang

**<details><summary>Abstract**</summary>

The gold standard for causal model evaluation involves comparing model predictions with true effects estimated from randomized controlled trials (RCT). However, RCTs are not always feasible or ethical to perform. In contrast, conditionally randomized experiments based on inverse probability weighting (IPW) offer a more realistic approach but may suffer from high estimation variance. To tackle this challenge and enhance causal model evaluation in real-world conditional randomization settings, we introduce a novel low-variance estimator for causal error, dubbed as the pairs estimator. By applying the same IPW estimator to both the model and true experimental effects, our estimator effectively cancels out the variance due to IPW and achieves a smaller asymptotic variance. Empirical studies demonstrate the improved of our estimator, highlighting its potential on achieving near-RCT performance. Our method offers a simple yet powerful solution to evaluate causal inference models in conditional randomization settings without complicated modification of the IPW estimator itself, paving the way for more robust and reliable model assessments.</details>

### HotBEV: Hardware-oriented Transformer-based Multi-View 3D Detector for BEV Perception

**Authors:** Peiyan Dong, Zhenglun Kong, Xin Meng, Pinrui Yu, Yifan Gong, Geng Yuan, Hao Tang, Yanzhi Wang

**<details><summary>Abstract**</summary>

The bird's-eye-view (BEV) perception plays a critical role in autonomous driving systems, involving the accurate and efficient detection and tracking of objects from a top-down perspective. To achieve real-time decision-making in self-driving scenarios, low-latency computation is essential. While recent approaches to BEV detection have focused on improving detection precision using Lift-Splat-Shoot (LSS)-based or transformer-based schemas, the substantial computational and memory burden of these approaches increases the risk of system crashes when multiple on-vehicle tasks run simultaneously. Unfortunately, there is a dearth of literature on efficient BEV detector paradigms, let alone achieving realistic speedups.Unlike existing works that focus on reducing computation costs, this paper focuses on developing an efficient model design that prioritizes actual on-device latency.To achieve this goal, we propose a latency-aware design methodology that considers key hardware properties, such as memory access cost and degree of parallelism.Given the prevalence of GPUs as the main computation platform for autonomous driving systems, we develop a theoretical latency prediction model and introduce efficient building operators.By leveraging these operators and following an effective local-to-global visual modeling process, we propose a hardware-oriented backbone that is also optimized for strong feature capturing and fusing.Using these insights, we present a new hardware-oriented framework for efficient yet accurate camera-view BEV detectors.Experiments show that HotBEV achieves a 2\%$\sim$23\% NDS gain, and 2\%$\sim$7.8\% mAP gain with a 1.1$\times$$\sim$3.4$\times$ speedups compared to existing works on V100;On multiple GPU devices such as GPU GTX 2080 and the low-end GTX 1080, HotBEV achieves 1.1$\times$$\sim$6.3$\times$ faster than others.</details>

### How a Student becomes a Teacher: learning and forgetting through Spectral methods

**Authors:** Lorenzo Giambagli, Lorenzo Buffoni, Lorenzo Chicchi, Duccio Fanelli

**<details><summary>Abstract**</summary>

In theoretical Machine Learning, the teacher-student paradigm is often employed as an effective metaphor for real-life tuition.  A student network is trained on data generated by a fixed teacher network until it matches the instructor’s ability to cope with the assigned task. The above scheme proves particularly relevant when the student network is overparameterized (namely, when larger layer sizes are employed) as compared to the underlying teacher network. Under these operating conditions, it is tempting to speculate that the student ability to handle the given task could be eventually stored in a sub-portion of the whole network. This latter should be to some extent reminiscent of the frozen teacher structure, according to suitable metrics, while being approximately invariant across different architectures of the student candidate network. Unfortunately, state-of-the-art conventional learning techniques could not help in identifying the existence of such an invariant subnetwork, due to the inherent degree of non-convexity that characterizes the examined problem. In this work, we take a decisive leap forward by proposing a radically different optimization scheme which builds on a spectral representation of the linear transfer of information between layers. The gradient is hence calculated with respect to both eigenvalues and eigenvectors with negligible increase in terms of computational and complexity load, as compared to standard training algorithms. Working in this framework, we could isolate a stable student substructure, that mirrors the true complexity of the teacher in terms of computing neurons, path distribution and topological attributes. When pruning unimportant nodes of the trained student, as follows a ranking that reflects the optimized eigenvalues, no degradation in the recorded performance is seen above a threshold that corresponds to the effective teacher size. The observed behavior can be pictured as a genuine second-order phase transition that bears universality traits.</details>

### How many samples are needed to leverage smoothness?

**Authors:** Vivien Cabannes, Stefano Vigogna

**<details><summary>Abstract**</summary>

A core principle in statistical learning is that smoothness of target functions allows to break the curse of dimensionality. However, learning a smooth function seems to require enough samples close to one another to get meaningful estimate of high-order derivatives, which would be hard in machine learning problems where the ratio between number of data and input dimension is relatively small. By deriving new lower bounds on the generalization error, this paper formalizes such an intuition, before investigating the role of constants and transitory regimes which are usually not depicted beyond classical learning theory statements while they play a dominant role in practice.</details>

### HyP-NeRF: Learning Improved NeRF Priors using a HyperNetwork

**Authors:** Bipasha Sen, Gaurav Singh, Aditya Agarwal, Rohith Agaram, Madhava Krishna, Srinath Sridhar

**<details><summary>Abstract**</summary>

Neural Radiance Fields (NeRF) have become an increasingly popular representation to capture high-quality appearance and shape of scenes and objects. However, learning generalizable NeRF priors over categories of scenes or objects has been challenging due to the high dimensionality of network weight space. To address the limitations of existing work on generalization, multi-view consistency and to improve quality, we propose HyP-NeRF, a latent conditioning method for learning generalizable category-level NeRF priors using hypernetworks. Rather than using hypernetworks to estimate only the weights of a NeRF, we estimate both the weights and the multi-resolution hash encodings resulting in significant quality gains. To improve quality even further, we incorporate a denoise and finetune strategy that denoises images rendered from NeRFs estimated by the hypernetwork and finetunes it while retaining multiview consistency. These improvements enable us to use HyP-NeRF as a generalizable prior for multiple downstream tasks including NeRF reconstruction from single-view or cluttered scenes and text-to-NeRF. We provide qualitative comparisons and evaluate HyP-NeRF on three tasks: generalization, compression, and retrieval, demonstrating our state-of-the-art results.</details>

### [Spotlight] HyTrel: Hypergraph-enhanced  Tabular Data Representation Learning

**Authors:** Pei Chen, Soumajyoti Sarkar, Leonard Lausen, Balasubramaniam Srinivasan, Sheng Zha, Ruihong Huang, George Karypis

**<details><summary>Abstract**</summary>

Language models pretrained on large collections of tabular data have demonstrated their effectiveness in several downstream tasks.However, many of these models do not take into account the row/column permutation invariances, hierarchical structure, etc. that exist in tabular data. To alleviate these limitations, we propose HyTrel, a tabular language model, that captures the permutation invariances and three more structural properties of tabular data by using hypergraphs--where the table cells make up the nodes and the cells occurring jointly together in each row, column, and the entire table are used to form three different types of hyperedges. We show thatHyTrel is maximally invariant under certain conditions for tabular data, i.e., two tables obtain the same representations via HyTreliff the two tables are identical up to permutation. Our empirical results demonstrate that HyTrel consistently outperforms other competitive baselines on four downstream tasks with minimal pretraining, illustrating the advantages of incorporating inductive biases associated with tabular data into the representations. Finally, our qualitative analyses showcase that HyTrel can assimilate the table structure to generate robust representations for the cells, rows, columns, and the entire table.</details>

### Hypervolume Maximization: A Geometric View of Pareto Set Learning

**Authors:** Xiaoyuan Zhang, Xi Lin, Bo Xue, Yifan Chen, Qingfu Zhang

**<details><summary>Abstract**</summary>

This paper presents a novel approach to multiobjective algorithms aimed at modeling the Pareto set using neural networks. Whereas previous methods mainly focused on identifying a finite number of solutions, our approach allows for the direct modeling of the entire Pareto set. Furthermore, we establish an equivalence between learning the complete Pareto set and maximizing the associated hypervolume, which enables the convergence analysis of hypervolume (as a new metric) for Pareto set learning. Specifically, our new analysis framework reveals the connection between the learned Pareto solution and its representation in a polar coordinate system. We evaluate our proposed approach on various benchmark problems and real-world problems, and the encouraging results make it a potentially viable alternative to existing multiobjective algorithms. Code is available at \url{https://github.com/xzhang2523/hvpsl/tree/master}.</details>

### Hypothesis Selection with Memory Constraints

**Authors:** Maryam Aliakbarpour, Mark Bun, Adam Smith

**<details><summary>Abstract**</summary>

Hypothesis selection is a fundamental problem in learning theory and statistics. Given a dataset and a finite set of candidate distributions, the goal is to select a distribution that matches the data as well as possible. More specifically, suppose we have sample access to an unknown distribution $P$ over a domain $\mathcal{X}$ that we know is well-approximated by one of a a class of $n$  distributions (a.k.a. hypotheses), $\mathcal{H} \coloneqq \{H_1, H_2, \ldots, H_n\}$. The goal is to design an algorithm that outputs a distribution $\hat{H} \in \mathcal{H}$ whose total variation distance from $P$ is nearly minimal.In this work, we study the hypothesis selection problem under memory constraints. We consider a model where samples from $P$ are presented in a stream and we access each sample $x$ via ``PDF-comparison'' queries that allow us to compare the probability densities of any pair of hypothesesat the domain point $x$ (i.e., is $H_i(x) < H_j(x)$?). This model allows us to study how much memory is needed at any point in time to store information about the portion of the stream seen so far.Our main result is an algorithm that achieves a nearly optimal tradeoff between memory usage and the number of samples required. In particular, given $b$ bits of memory (for $b$ roughly between $\log n$ and $n$), our algorithm solves the hypothesis selection problem with $s$ samples, where $b \cdot s = O(n \log n)$. This result is optimal up to an $O(\log n)$ factor, for all $b$.</details>

### IDEA: An Invariant Perspective for Efficient Domain Adaptive Image Retrieval

**Authors:** Haixin Wang, Hao Wu, Jinan Sun, Shikun Zhang, Chong Chen, Xian-Sheng Hua, Xiao Luo

**<details><summary>Abstract**</summary>

In this paper, we investigate the problem of unsupervised domain adaptive hashing, which leverage knowledge from a label-rich source domain to expedite learning to hash on a label-scarce target domain. Although numerous existing approaches attempt to incorporate transfer learning techniques into deep hashing frameworks, they often neglect the essential invariance for adequate alignment between these two domains. Worse yet, these methods fail to distinguish between causal and non-causal effects embedded in images, rendering cross-domain retrieval ineffective. To address these challenges, we propose an Invariance-acquired Domain AdaptivE HAshing (IDEA) model. Our IDEA first decomposes each image into a causal feature representing label information, and a non-causal feature indicating domain information. Subsequently, we generate discriminative hash codes using causal features with consistency learning on both source and target domains. More importantly, we employ a generative model for synthetic samples to simulate the intervention of various non-causal effects, ultimately minimizing their impact on hash codes for domain invariance. Comprehensive experiments conducted on benchmark datasets validate the superior performance of our IDEA compared to a variety of competitive baselines.</details>

### Idempotent Learned Image Compression with Right-Inverse

**Authors:** Yanghao Li, Tongda Xu, Yan Wang, Jingjing Liu, Ya-Qin Zhang

**<details><summary>Abstract**</summary>

We consider the problem of idempotent learned image compression (LIC).The idempotence of codec refers to the stability of codec to re-compression.To achieve idempotence, previous codecs adopt invertible transforms such as DCT and normalizing flow.In this paper, we first identify that invertibility of transform is sufficient but not necessary for idempotence. Instead, it can be relaxed into right-invertibility. And such relaxation allows wider family of transforms.Based on this identification, we implement an idempotent codec using our proposed blocked convolution and null-space enhancement.Empirical results show that we achieve state-of-the-art rate-distortion performance among idempotent codecs. Furthermore, our codec can be extended into near-idempotent codec by relaxing the right-invertibility. And this near-idempotent codec has significantly less quality decay after $50$ rounds of re-compression compared with other near-idempotent codecs.</details>

### Identifiable Contrastive Learning with Automatic Feature Importance Discovery

**Authors:** Qi Zhang, Yifei Wang, Yisen Wang

**<details><summary>Abstract**</summary>

Existing contrastive learning methods rely on pairwise sample contrast $z_x^	op z_{x'}$ to learn data representations, but the learned features often lack clear interpretability from a human perspective. Theoretically, it lacks feature identifiability and different initialization may lead to totally different features. In this paper, we study a new method named tri-factor contrastive learning (triCL) that involves a 3-factor contrast in the form of $z_x^\top S z_{x'}$, where $S=	ext{diag}(s_1,\dots,s_k)$ is a learnable diagonal matrix that automatically captures the importance of each feature. We show that by this simple extension, triCL can not only obtain identifiable features that eliminate randomness but also obtain more interpretable features that are ordered according to the importance matrix $S$. We show that features with high importance have nice interpretability by capturing common classwise features, and obtain superior performance when evaluated for image retrieval using a few features. The proposed triCL objective is general and can be applied to different contrastive learning methods like SimCLR and CLIP. We believe that it is a better alternative to existing 2-factor contrastive learning by improving its identifiability and interpretability with minimal overhead. Code is available at https://github.com/PKU-ML/Tri-factor-Contrastive-Learning.</details>

### Im-Promptu: In-Context Composition from Image Prompts

**Authors:** Bhishma Dedhia, Michael Chang, Jake Snell, Tom Griffiths, Niraj Jha

**<details><summary>Abstract**</summary>

Large language models are few-shot learners that can solve diverse tasks from a handful of demonstrations. This implicit understanding of tasks suggests that the attention mechanisms over word tokens may play a role in analogical reasoning. In this work, we investigate whether analogical reasoning can enable in-context composition over composable elements of visual stimuli. First, we introduce a suite of three benchmarks to test the generalization properties of a visual in-context learner. We formalize the notion of an analogy-based in-context learner and use it to design a meta-learning framework called Im-Promptu. Whereas the requisite token granularity for language is well established, the appropriate compositional granularity for enabling in-context generalization in visual stimuli is usually unspecified. To this end, we use Im-Promptu to train multiple agents with different levels of compositionality, including vector representations, patch representations, and object slots. Our experiments reveal tradeoffs between extrapolation abilities and the degree of compositionality, with non-compositional representations extending learned composition rules to unseen domains but performing poorly on combinatorial tasks. Patch-based representations require patches to contain entire objects for robust extrapolation. At the same time, object-centric tokenizers coupled with a cross-attention module generate consistent and high-fidelity solutions, with these inductive biases being particularly crucial for compositional generalization. Lastly, we demonstrate a use case of Im-Promptu as an intuitive programming interface for image generation.</details>

### Imitation Learning from Vague Feedback

**Authors:** Xin-Qiang Cai, Yu-Jie Zhang, Chao-Kai Chiang, Masashi Sugiyama

**<details><summary>Abstract**</summary>

Imitation learning from human feedback studies how to train well-performed imitation agents with an annotator's relative comparison of two demonstrations (one demonstration is better/worse than the other), which is usually easier to collect than the perfect expert data required by traditional imitation learning. However, in many real-world applications, it is still expensive or even impossible to provide a clear pairwise comparison between two demonstrations with similar quality. This motivates us to study the problem of imitation learning with vague feedback, where the data annotator can only distinguish the paired demonstrations correctly when their quality differs significantly, i.e., one from the expert and another from the non-expert. By modeling the underlying demonstration pool as a mixture of expert and non-expert data, we show that the expert policy distribution can be recovered when the proportion $\alpha$ of expert data is known. We also propose a mixture proportion estimation method for the unknown $\alpha$ case. Then, we integrate the recovered expert policy distribution with generative adversarial imitation learning to form an end-to-end algorithm. Experiments show that our methods outperform standard and preference-based imitation learning methods on various tasks.</details>

### Implicit Convolutional Kernels for Steerable CNNs

**Authors:** Maksim Zhdanov, Nico Hoffmann, Gabriele Cesa

**<details><summary>Abstract**</summary>

Steerable convolutional neural networks (CNNs) provide a general framework for building neural networks equivariant to translations and transformations of an origin-preserving group $G$, such as reflections and rotations. They rely on standard convolutions with $G$-steerable kernels obtained by analytically solving the group-specific equivariance constraint imposed onto the kernel space. As the solution is tailored to a particular group $G$, implementing a kernel basis does not generalize to other symmetry transformations, complicating the development of general group equivariant models. We propose using implicit neural representation via multi-layer perceptrons (MLPs) to parameterize $G$-steerable kernels. The resulting framework offers a simple and flexible way to implement Steerable CNNs and generalizes to any group $G$ for which a $G$-equivariant MLP can be built. We prove the effectiveness of our method on multiple tasks, including N-body simulations, point cloud classification and molecular property prediction.</details>

### Implicit Transfer Operator Learning: Multiple Time-Resolution Models for Molecular Dynamics

**Authors:** Mathias Schreiner, Ole Winther, Simon Olsson

**<details><summary>Abstract**</summary>

Computing properties of molecular systems rely on estimating expectations of the (unnormalized) Boltzmann distribution. Molecular dynamics (MD) is a broadly adopted technique to approximate such quantities. However, stable simulations rely on very small integration time-steps ($10^{-15}\,\mathrm{s}$), whereas convergence of some moments, e.g. binding free energy or rates, might rely on sampling processes on time-scales as long as $10^{-1}\, \mathrm{s}$, and these simulations must be repeated for every molecular system independently. Here, we present Implict Transfer Operator (ITO) Learning, a framework to learn surrogates of the simulation process with multiple time-resolutions. We implement ITO with denoising diffusion probabilistic models with a new SE(3) equivariant architecture and show the resulting models can generate self-consistent stochastic dynamics across multiple time-scales, even when the system is only partially observed. Finally, we present a coarse-grained CG-SE3-ITO model which can quantitatively model all-atom molecular dynamics using only coarse molecular representations. As such, ITO provides an important step towards multiple time- and space-resolution acceleration of MD. Code is available at\ \href{https://github.com/olsson-group/ito}{https://github.com/olsson-group/ito}.</details>

### Importance Weighted Actor-Critic for Optimal Conservative Offline Reinforcement Learning

**Authors:** Hanlin Zhu, Paria Rashidinejad, Jiantao Jiao

**<details><summary>Abstract**</summary>

We propose A-Crab (Actor-Critic Regularized by Average Bellman error), a new practical algorithm for offline reinforcement learning (RL) in complex environments with insufficient data coverage. Our algorithm combines the marginalized importance sampling framework with the actor-critic paradigm, where the critic returns evaluations of the actor (policy) that are pessimistic relative to the offline data and have a small average (importance-weighted) Bellman error. Compared to existing methods, our algorithm simultaneously offers a number of advantages:(1) It achieves the optimal statistical rate of $1/\sqrt{N}$---where $N$ is the size of offline dataset---in converging to the best policy covered in the offline dataset, even when combined with general function approximators.(2) It relies on a weaker \textit{average} notion of policy coverage (compared to the $\ell_\infty$ single-policy concentrability) that exploits the structure of policy visitations.(3) It outperforms the data-collection behavior policy over a wide range of specific hyperparameters. We provide both theoretical analysis and experimental results to validate the effectiveness of our proposed algorithm. The code is available at https://github.com/zhuhl98/ACrab.</details>

### Improved Bayesian Regret Bounds for Thompson Sampling in Reinforcement Learning

**Authors:** Ahmadreza Moradipari, Mohammad Pedramfar, Modjtaba Shokrian Zini, Vaneet Aggarwal

**<details><summary>Abstract**</summary>

In this paper, we prove state-of-the-art Bayesian regret bounds for Thompson Sampling in reinforcement learning in a multitude of settings. We present a refined analysis of the information ratio, and show an upper bound of order $\widetilde{O}(H\sqrt{d_{l_1}T})$ in the time inhomogeneous reinforcement learning problem where $H$ is the episode length and $d_{l_1}$ is the Kolmogorov $l_1-$dimension of the space of environments. We then find concrete bounds of $d_{l_1}$ in a variety of settings, such as tabular, linear and finite mixtures, and discuss how our results improve the state-of-the-art.</details>

### [Spotlight] Improved Frequency Estimation Algorithms with and without Predictions

**Authors:** Anders Aamand, Justin Chen, Huy Nguyen, Sandeep Silwal, Ali Vakilian

**<details><summary>Abstract**</summary>

Estimating frequencies of elements appearing in a data stream is a key task in large-scale data analysis. Popular sketching approaches to this problem (e.g., CountMin and CountSketch) come with worst-case guarantees that probabilistically bound the error of the estimated frequencies for any possible input. The work of Hsu et al.~(2019) introduced the idea of using machine learning to tailor sketching algorithms to the specific data distribution they are being run on. In particular, their learning-augmented frequency estimation algorithm uses a learned heavy-hitter oracle which predicts which elements will appear many times in the stream. We give a novel algorithm, which in some parameter regimes, already theoretically outperforms the learning based algorithm of Hsu et al. *without* the use of any predictions. Augmenting our algorithm with heavy-hitter predictions further reduces the error and improves upon the state of the art. Empirically, our algorithms achieve superior performance in all experiments compared to prior approaches.</details>

### Improving Graph Matching with Positional Reconstruction Encoder-Decoder Network

**Authors:** Yixiao Zhou, Ruiqi Jia, Hongxiang Lin, Hefeng Quan, Yumeng Zhao, Xiaoqing Lyu

**<details><summary>Abstract**</summary>

Deriving from image matching and understanding, semantic keypoint matching aims at establishing correspondence between keypoint sets in images. As graphs are powerful tools to represent points and their complex relationships, graph matching provides an effective way to find desired semantic keypoint correspondences. Recent deep graph matching methods have shown excellent performance, but there is still a lack of exploration and utilization of spatial information of keypoints as nodes in graphs. More specifically, existing methods are insufficient to capture the relative spatial relations through current graph construction approaches from the locations of semantic keypoints. To address these issues, we introduce a positional reconstruction encoder-decoder (PR-EnDec) to model intrinsic graph spatial structure, and present an end-to-end graph matching network PREGM based on PR-EnDec. Our PR-EnDec consists of a positional encoder that learns effective node spatial embedding with the affine transformation invariance, and a spatial relation decoder that further utilizes the high-order spatial information by reconstructing the locational structure of graphs contained in the node coordinates. Extensive experimental results on three public keypoint matching datasets demonstrate the effectiveness of our proposed PREGM.</details>

### Improving the Knowledge Gradient Algorithm

**Authors:** Le Yang, Siyang Gao, Chin Pang Ho

**<details><summary>Abstract**</summary>

The knowledge gradient (KG) algorithm is a popular policy for the best arm identification (BAI) problem. It is built on the simple idea of always choosing the measurement that yields the greatest expected one-step improvement in the estimate of the best mean of the arms. In this research, we show that this policy has limitations, causing the algorithm not asymptotically optimal. We next provide a remedy for it, by following the manner of one-step look ahead of KG, but instead choosing the measurement that yields the greatest one-step improvement in the probability of selecting the best arm. The new policy is called improved knowledge gradient (iKG). iKG can be shown to be asymptotically optimal. In addition, we show that compared to KG, it is easier to extend iKG to variant problems of BAI, with the $\epsilon$-good arm identification and feasible arm identification as two examples. The superior performances of iKG on these problems are further demonstrated using numerical examples.</details>

### [Spotlight] In-Context Impersonation Reveals Large Language Models' Strengths and Biases

**Authors:** Leonard Salewski, Stephan Alaniz, Isabel Rio-Torto, Eric Schulz, Zeynep Akata

**<details><summary>Abstract**</summary>

In everyday conversations, humans can take on different roles and adapt their vocabulary to their chosen roles. We explore whether LLMs can take on, that is impersonate, different roles when they generate text in-context. We ask LLMs to assume different personas before solving vision and language tasks. We do this by prefixing the prompt with a persona that is associated either with a social identity or domain expertise. In a multi-armed bandit task, we find that LLMs pretending to be children of different ages recover human-like developmental stages of exploration. In a language-based reasoning task, we find that LLMs impersonating domain experts perform better than LLMs impersonating non-domain experts. Finally, we test whether LLMs' impersonations are complementary to visual information when describing different categories. We find that impersonation can improve performance: an LLM prompted to be a bird expert describes birds better than one prompted to be a car expert. However, impersonation can also uncover LLMs' biases: an LLM prompted to be a man describes cars better than one prompted to be a woman. These findings demonstrate that LLMs are capable of taking on diverse roles and that this in-context impersonation can be used to uncover their strengths and hidden biases. Our code is available at https://github.com/ExplainableML/in-context-impersonation.</details>

### [Spotlight] In-Context Learning Unlocked for Diffusion Models

**Authors:** Zhendong Wang, Yifan Jiang, Yadong Lu, yelong shen, Pengcheng He, Weizhu Chen, Zhangyang "Atlas" Wang, Mingyuan Zhou

**<details><summary>Abstract**</summary>

We present Prompt Diffusion, a framework for enabling in-context learning in diffusion-based generative models. Given a pair of task-specific example images, such as depth from/to image and scribble from/to image, and a text guidance, our model automatically understands the underlying task and performs the same task on a new query image following the text guidance. To achieve this, we propose a vision-language prompt that can model a wide range of vision-language tasks and a diffusion model that takes it as input. The diffusion model is trained jointly on six different tasks using these prompts. The resulting Prompt Diffusion model becomes the first diffusion-based vision-language foundation model capable of in-context learning. It demonstrates high-quality in-context generation for the trained tasks and effectively generalizes to new, unseen vision tasks using their respective prompts. Our model also shows compelling text-guided image editing results. Our framework aims to facilitate research into in-context learning for computer vision. We share our code and pre-trained models at https://github.com/Zhendong-Wang/Prompt-Diffusion.</details>

### Incentives in Federated Learning: Equilibria, Dynamics, and Mechanisms for Welfare Maximization

**Authors:** Aniket Murhekar, Zhuowen Yuan, Bhaskar Ray Chaudhury, Bo Li, Ruta Mehta

**<details><summary>Abstract**</summary>

Federated learning (FL) has emerged as a powerful scheme to facilitate the collaborative learning of models amongst a set of agents holding their own private data.  Although the agents benefit from the global model trained on shared data, by participating in federated learning, they may also incur costs (related to privacy and communication) due to data sharing. In this paper, we model a collaborative FL framework, where every agent attempts to achieve an optimal trade-off between her learning payoff and data sharing cost. We show the existence of Nash equilibrium (NE) under mild assumptions on agents' payoff and costs. Furthermore, we show that agents can discover the NE via best response dynamics. However, some of the NE may be bad in terms of overall welfare for the agents, implying little incentive for some fraction of the agents to participate in the learning. To remedy this, we design a budget-balanced mechanism involving payments to the agents, that ensures that any $p$-mean welfare function of the agents' utilities is maximized at NE. In addition, we introduce a FL protocol FedBR-BG that incorporates our budget-balanced mechanism, utilizing best response dynamics. Our empirical validation on MNIST and CIFAR-10 substantiates our theoretical analysis. We show that FedBR-BG outperforms the basic best-response-based protocol without additional incentivization, the standard federated learning protocol FedAvg, as well as a recent baseline MWFed in terms of achieving superior $p$-mean welfare.</details>

### Incentivized Communication for Federated Bandits

**Authors:** Zhepei Wei, Chuanhao Li, Haifeng Xu, Hongning Wang

**<details><summary>Abstract**</summary>

Most existing works on federated bandits take it for granted that all clients are altruistic about sharing their data with the server for the collective good whenever needed. Despite their compelling theoretical guarantee on performance and communication efficiency, this assumption is overly idealistic and oftentimes violated in practice, especially when the algorithm is operated over self-interested clients, who are reluctant to share data without explicit benefits. Negligence of such self-interested behaviors can significantly affect the learning efficiency and even the practical operability of federated bandit learning. In light of this, we aim to spark new insights into this under-explored research area by formally introducing an incentivized communication problem for federated bandits, where the server shall motivate clients to share data by providing incentives. Without loss of generality, we instantiate this bandit problem with the contextual linear setting and propose the first incentivized communication protocol, namely, Inc-FedUCB, that achieves near-optimal regret with provable communication and incentive cost guarantees. Extensive empirical experiments on both synthetic and real-world datasets further validate the effectiveness of the proposed method across various environments.</details>

### Incomplete Multimodality-Diffused Emotion Recognition

**Authors:** Yuanzhi Wang, Yong Li, Zhen Cui

**<details><summary>Abstract**</summary>

Human multimodal emotion recognition (MER) aims to perceive and understand human emotions via various heterogeneous modalities, such as language, vision, and acoustic. Compared with unimodality, the complementary information in the multimodalities facilitates robust emotion understanding. Nevertheless,  in real-world scenarios, the missing modalities hinder multimodal understanding and result in degraded MER performance. In this paper, we propose an Incomplete Multimodality-Diffused emotion recognition (IMDer) method to mitigate the challenge of MER under incomplete multimodalities. To recover the missing modalities, IMDer exploits the score-based diffusion model that maps the input Gaussian noise into the desired distribution space of the missing modalities and recovers missing data abided by their original distributions. Specially, to reduce semantic ambiguity between the missing and the recovered modalities, the available modalities are embedded as the condition to guide and refine the diffusion-based recovering process. In contrast to previous work,  the diffusion-based modality recovery mechanism in IMDer allows to simultaneously reach both distribution consistency and semantic disambiguation. Feature visualization of the recovered modalities illustrates the consistent modality-specific distribution and semantic alignment. Besides, quantitative experimental results verify that IMDer obtains state-of-the-art MER accuracy under various missing modality patterns.</details>

### Inconsistency, Instability, and Generalization Gap of Deep Neural Network Training

**Authors:** Rie Johnson, Tong Zhang

**<details><summary>Abstract**</summary>

As deep neural networks are highly expressive, it is important to find solutions with small generalization gap (the difference between the performance on the training data and unseen data).  Focusing on the stochastic nature of training, we first present a theoretical analysis in which the bound of generalization gap depends on what we call inconsistency and instability of model outputs, which can be estimated on unlabeled data.  Our empirical study based on this analysis shows that instability and inconsistency are strongly predictive of generalization gap in various settings.  In particular, our finding indicates that inconsistency is a more reliable indicator of generalization gap than the sharpness of the loss landscape.  Furthermore, we show that algorithmic reduction of inconsistency leads to superior performance.  The results also provide a theoretical basis for existing methods such as co-distillation and ensemble.</details>

### [Spotlight] Information Maximization Perspective of Orthogonal Matching Pursuit with Applications to Explainable AI

**Authors:** Aditya Chattopadhyay, Ryan Pilgrim, Rene Vidal

**<details><summary>Abstract**</summary>

Information Pursuit (IP) is a classical active testing algorithm for predicting an output by sequentially and greedily querying the input in order of information gain. However, IP is computationally intensive since it involves estimating mutual information in high-dimensional spaces. This paper explores Orthogonal Matching Pursuit (OMP) as an alternative to IP for greedily selecting the queries. OMP is a classical signal processing algorithm for sequentially encoding a signal in terms of dictionary atoms chosen in order of correlation gain. In each iteration, OMP selects the atom that is most correlated with the signal residual (the signal minus its reconstruction thus far). Our first contribution is to establish a fundamental connection between IP and OMP, where we prove that IP with random projections of dictionary atoms as queries ``almost'' reduces to OMP, with the difference being that IP selects atoms in order of normalized correlation gain. We call this version IP-OMP and present simulations indicating that this difference does not have any appreciable effect on the sparse code recovery rate of IP-OMP compared to that of OMP for random Gaussian dictionaries. Inspired by this connection, our second contribution is to explore the utility of IP-OMP for generating explainable predictions, an area in which IP has recently gained traction. More specifically, we propose a simple explainable AI algorithm which encodes an image as a sparse combination of semantically meaningful dictionary atoms that are defined as text embeddings of interpretable concepts. The final prediction is made using the weights of this sparse combination, which serve as an explanation. Empirically, our proposed algorithm is not only competitive with existing explainability methods but also computationally less expensive.</details>

### Information Maximizing Curriculum: A Curriculum-Based Approach for Learning Versatile Skills

**Authors:** Denis Blessing, Onur Celik, Xiaogang Jia, Moritz Reuss, Maximilian Li, Rudolf Lioutikov, Gerhard Neumann

**<details><summary>Abstract**</summary>

Imitation learning uses data for training policies to solve complex tasks. However,when the training data is collected from human demonstrators, it often leadsto multimodal distributions because of the variability in human actions. Mostimitation learning methods rely on a maximum likelihood (ML) objective to learna parameterized policy, but this can result in suboptimal or unsafe behavior dueto the mode-averaging property of the ML objective. In this work, we proposeInformation Maximizing Curriculum, a curriculum-based approach that assignsa weight to each data point and encourages the model to specialize in the data itcan represent, effectively mitigating the mode-averaging problem by allowing themodel to ignore data from modes it cannot represent. To cover all modes and thus,enable versatile behavior, we extend our approach to a mixture of experts (MoE)policy, where each mixture component selects its own subset of the training datafor learning. A novel, maximum entropy-based objective is proposed to achievefull coverage of the dataset, thereby enabling the policy to encompass all modeswithin the data distribution. We demonstrate the effectiveness of our approach oncomplex simulated control tasks using versatile human demonstrations, achievingsuperior performance compared to state-of-the-art methods.</details>

### Information Theoretic Lower Bounds for Information Theoretic Upper Bounds

**Authors:** Roi Livni

**<details><summary>Abstract**</summary>

We examine the relationship between the mutual information between the output model and the empirical sample and the algorithm's generalization in the context of stochastic convex optimization. Despite increasing interest in information-theoretic generalization bounds, it is uncertain if these bounds can provide insight into the exceptional performance of various learning algorithms. Our study of stochastic convex optimization reveals that, for true risk minimization, dimension-dependent mutual information is necessary. This indicates that existing information-theoretic generalization bounds fall short in capturing the generalization capabilities of algorithms like SGD and regularized ERM, which have dimension-independent sample complexity.</details>

### Inner Product-based Neural Network Similarity

**Authors:** Wei Chen, Zichen Miao, Qiang Qiu

**<details><summary>Abstract**</summary>

Analyzing representational similarity among neural networks (NNs) is essential for interpreting or transferring deep models. In application scenarios where numerous NN models are learned, it becomes crucial to assess model similarities in computationally efficient ways. In this paper, we propose a new paradigm for reducing NN representational similarity to filter subspace distance. Specifically, when convolutional filters are decomposed as a linear combination of a set of filter subspace elements, denoted as filter atoms, and have those decomposed atom coefficients shared across networks, NN representational similarity can be significantly simplified as calculating the cosine distance among respective filter atoms, to achieve millions of times computation reduction over popular probing-based methods. We provide both theoretical and empirical evidence that such simplified filter subspace-based similarity preserves a strong linear correlation with other popular probing-based metrics, while being significantly more efficient to obtain and robust to probing data. We further validate the effectiveness of the proposed method in various application scenarios where numerous models exist, such as federated and continual learning as well as analyzing training dynamics. We hope our findings can help further explorations of real-time large-scale representational similarity analysis in neural networks.</details>

### Instructing Goal-Conditioned Reinforcement Learning Agents with Temporal Logic Objectives

**Authors:** Wenjie Qiu, Wensen Mao, He Zhu

**<details><summary>Abstract**</summary>

Goal-conditioned reinforcement learning (RL) is a powerful approach for learning general-purpose skills by reaching diverse goals. However, it has limitations when it comes to task-conditioned policies, where goals are specified by temporally extended instructions written in the Linear Temporal Logic (LTL) formal language. Existing approaches for finding LTL-satisfying policies rely on sampling a large set of LTL instructions during training to adapt to unseen tasks at inference time. However, these approaches do not guarantee generalization to out-of-distribution LTL objectives, which may have increased complexity. In this paper, we propose a novel approach to address this challenge. We show that simple goal-conditioned RL agents can be instructed to follow arbitrary LTL specifications without additional training over the LTL task space. Unlike existing approaches that focus on LTL specifications expressible as regular expressions, our technique is unrestricted and generalizes to $\omega$-regular expressions. Experiment results demonstrate the effectiveness of our approach in adapting goal-conditioned RL agents to satisfy complex temporal logic task specifications zero-shot.</details>

### Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts

**Authors:** Eduard Tulchinskii, Kristian Kuznetsov, Laida Kushnareva, Daniil Cherniavskii, Sergey Nikolenko, Evgeny Burnaev, Serguei Barannikov, Irina Piontkovskaya

**<details><summary>Abstract**</summary>

Rapidly increasing quality of AI-generated content makes it difficult to distinguish between human and AI-generated texts, which may lead to undesirable consequences for society. Therefore, it becomes increasingly important to study the properties of human texts that are invariant over text domains and various proficiency of human writers, can be easily calculated for any language, and can robustly separate natural and AI-generated texts regardless of the generation model and sampling method. In this work, we propose such an invariant of human texts, namely the intrinsic dimensionality of the manifold underlying the set of embeddings of a given text sample. We show that the average intrinsic dimensionality of fluent texts in natural language is hovering around the value $9$ for several alphabet-based languages and around $7$ for Chinese, while the average intrinsic dimensionality of AI-generated texts for each language is $\approx 1.5$ lower, with a clear statistical separation between human-generated and AI-generated distributions. This property allows us to build a score-based artificial text detector. The proposed detector's accuracy is stable over text domains, generator models, and human writer proficiency levels, outperforming SOTA detectors in model-agnostic and cross-domain scenarios by a significant margin.</details>

### Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation

**Authors:** Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, LINGMING ZHANG

**<details><summary>Abstract**</summary>

Program synthesis has been long studied with recent approaches focused on directly using the power of Large Language Models (LLMs) to generate code. Programming benchmarks, with curated synthesis problems and test-cases, are used to measure the performance of various LLMs on code synthesis. However, these test-cases can be limited in both quantity and quality for fully assessing the functional correctness of the generated code. Such limitation in the existing benchmarks begs the following question: In the era of LLMs, is the code generated really correct? To answer this, we propose EvalPlus – a code synthesis evaluation framework to rigorously benchmark the functional correctness of LLM-synthesized code. EvalPlus augments a given evaluation dataset with large amounts of test-cases newly produced by an automatic test input generator, powered by both LLM- and mutation-based strategies. While EvalPlus is general, we extend the test-cases of the popular HumanEval benchmark by 80x to build HumanEval+. Our extensive evaluation across 26 popular LLMs (e.g., GPT-4 and ChatGPT) demonstrates that HumanEval+ is able to catch significant amounts of previously undetected wrong code synthesized by LLMs, reducing the pass@k by up-to 19.3-28.9%. We also surprisingly found that test insufficiency can lead to mis-ranking. For example, both WizardCoder-CodeLlama and Phind-CodeLlama now outperform ChatGPT on HumanEval+, while none of them could on HumanEval. Our work not only indicates that prior popular code synthesis evaluation results do not accurately reflect the true performance of LLMs for code synthesis, but also opens up a new direction to improve such programming benchmarks through automated testing. We have open-sourced our tools, enhanced datasets as well as all LLM-generated code at https://github.com/evalplus/evalplus to facilitate and accelerate future LLM-for-code research.</details>

### Iteratively Learn Diverse Strategies with State Distance Information

**Authors:** Wei Fu, Weihua Du, Jingwei Li, Sunli Chen, Jingzhao Zhang, YI WU

**<details><summary>Abstract**</summary>

In complex reinforcement learning (RL) problems, policies with similar rewards may have substantially different behaviors. It remains a fundamental challenge to optimize rewards while also discovering as many *diverse* strategies as possible, which can be crucial in many practical applications. Our study examines two design choices for tackling this challenge, i.e., *diversity measure* and *computation framework*. First, we find that with existing diversity measures, visually indistinguishable policies can still yield high diversity scores. To accurately capture the behavioral difference, we propose to incorporate the state-space distance information into the diversity measure. In addition, we examine two common computation frameworks for this problem, i.e., population-based training (PBT) and iterative learning (ITR). We show that although PBT is the precise problem formulation, ITR can achieve comparable diversity scores with higher computation efficiency, leading to improved solution quality in practice. Based on our analysis, we further combine ITR with two tractable realizations of the state-distance-based diversity measures and develop a novel diversity-driven RL algorithm, *State-based Intrinsic-reward Policy Optimization* (SIPO), with provable convergence properties. We empirically examine SIPO across three domains from robot locomotion to multi-agent games. In all of our testing environments, SIPO consistently produces strategically diverse and human-interpretable policies that cannot be discovered by existing baselines.</details>

### Joint processing of linguistic properties in brains and language models

**Authors:** SUBBAREDDY OOTA, Manish Gupta, Mariya Toneva

**<details><summary>Abstract**</summary>

Language models have been shown to be very effective in predicting brain recordings of subjects experiencing complex language stimuli. For a deeper understanding of this alignment, it is important to understand the correspondence between the detailed processing of linguistic information by the human brain versus language models. We investigate this correspondence via a direct approach, in which we eliminate information related to specific linguistic properties in the language model representations and observe how this intervention affects the alignment with fMRI brain recordings obtained while participants listened to a story. We investigate a range of linguistic properties (surface, syntactic, and semantic) and find that the elimination of each one results in a significant decrease in brain alignment. Specifically, we find that syntactic properties (i.e. Top Constituents and Tree Depth) have the largest effect on the trend of brain alignment across model layers. These findings provide clear evidence for the role of specific linguistic information in the alignment between brain and language models, and open new avenues for mapping the joint information processing in both systems. We make the code publicly available [https://github.com/subbareddy248/linguistic-properties-brain-alignment].</details>

### Keep Various Trajectories: Promoting Exploration of Ensemble Policies in Continuous Control

**Authors:** Chao Li, Chen GONG, Qiang He, Xinwen Hou

**<details><summary>Abstract**</summary>

The combination of deep reinforcement learning (DRL) with ensemble methods has been proved to be highly effective in addressing complex sequential decision-making problems. This success can be primarily attributed to the utilization of multiple models, which enhances both the robustness of the policy and the accuracy of value function estimation. However, there has been limited analysis of the empirical success of current ensemble RL methods thus far. Our new analysis reveals that the sample efficiency of previous ensemble DRL algorithms may be limited by sub-policies that are not as diverse as they could be. Motivated by these findings, our study introduces a new ensemble RL algorithm, termed \textbf{T}rajectories-awar\textbf{E} \textbf{E}nsemble exploratio\textbf{N} (TEEN). The primary goal of TEEN is to  maximize the expected return while promoting more diverse trajectories. Through extensive experiments, we demonstrate that TEEN not only enhances the sample diversity of the ensemble policy compared to using sub-policies alone but also improves the performance over ensemble RL algorithms. On average, TEEN outperforms the baseline ensemble DRL algorithms by 41\% in performance on the tested representative environments.</details>

### Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors

**Authors:** Yong Liu, Chenyu Li, Jianmin Wang, Mingsheng Long

**<details><summary>Abstract**</summary>

Real-world time series are characterized by intrinsic non-stationarity that poses a principal challenge for deep forecasting models. While previous models suffer from complicated series variations induced by changing temporal distribution, we tackle non-stationary time series with modern Koopman theory that fundamentally considers the underlying time-variant dynamics. Inspired by Koopman theory of portraying complex dynamical systems, we disentangle time-variant and time-invariant components from intricate non-stationary series by Fourier Filter and design Koopman Predictor to advance respective dynamics forward. Technically, we propose Koopa as a novel Koopman forecaster composed of stackable blocks that learn hierarchical dynamics. Koopa seeks measurement functions for Koopman embedding and utilizes Koopman operators as linear portraits of implicit transition. To cope with time-variant dynamics that exhibits strong locality, Koopa calculates context-aware operators in the temporal neighborhood and is able to utilize incoming ground truth to scale up forecast horizon. Besides, by integrating Koopman Predictors into deep residual structure, we ravel out the binding reconstruction loss in previous Koopman forecasters and achieve end-to-end forecasting objective optimization. Compared with the state-of-the-art model, Koopa achieves competitive performance while saving 77.3% training time and 76.0% memory.</details>

### LANCE: Stress-testing Visual Models by Generating Language-guided Counterfactual Images

**Authors:** Viraj Prabhu, Sriram Yenamandra, Prithvijit Chattopadhyay, Judy Hoffman

**<details><summary>Abstract**</summary>

We propose an automated algorithm to stress-test a trained visual model by generating language-guided counterfactual test images (LANCE). Our method leverages recent progress in large language modeling and text-based image editing to augment an IID test set with a suite of diverse, realistic, and challenging test images without altering model weights. We benchmark the performance of a diverse set of pre-trained models on our generated data and observe significant and consistent performance drops. We further analyze model sensitivity across different types of edits, and demonstrate its applicability at surfacing previously unknown class-level model biases in ImageNet. Code is available at https://github.com/virajprabhu/lance.</details>

### LD2: Scalable Heterophilous Graph Neural Network with Decoupled Embeddings

**Authors:** Ningyi Liao, Siqiang Luo, Xiang Li, Jieming Shi

**<details><summary>Abstract**</summary>

Heterophilous Graph Neural Network (GNN) is a family of GNNs that specializes in learning graphs under heterophily, where connected nodes tend to have different labels. Most existing heterophilous models incorporate iterative non-local computations to capture node relationships. However, these approaches have limited application to large-scale graphs due to their high computational costs and challenges in adopting minibatch schemes. In this work, we study the scalability issues of heterophilous GNN and propose a scalable model, LD2, which simplifies the learning process by decoupling graph propagation and generating expressive embeddings prior to training. Theoretical analysis demonstrates that LD2 achieves optimal time complexity in training, as well as a memory footprint that remains independent of the graph scale. We conduct extensive experiments to showcase that our model is capable of lightweight minibatch training on large-scale heterophilous graphs, with up to $15\times$ speed improvement and efficient memory utilization, while maintaining comparable or better performance than the baselines.</details>

### Label-efficient Segmentation via Affinity Propagation

**Authors:** Wentong Li, Yuqian Yuan, Song Wang, Wenyu Liu, Dongqi Tang, Jian liu, Jianke Zhu, Lei Zhang

**<details><summary>Abstract**</summary>

Weakly-supervised segmentation with label-efficient sparse annotations has attracted increasing research attention to reduce the cost of laborious pixel-wise labeling process, while the pairwise affinity modeling techniques play an essential role in this task. Most of the existing approaches focus on using the local appearance kernel to model the neighboring pairwise potentials. However, such a local operation fails to capture the long-range dependencies and ignores the  topology of objects. In this work, we formulate the affinity modeling as an affinity propagation process, and propose a local and a global pairwise affinity terms to generate accurate soft pseudo labels. An efficient algorithm is also developed to reduce significantly the computational cost. The proposed approach can be conveniently plugged into existing segmentation networks. Experiments on three typical label-efficient segmentation tasks, i.e. box-supervised instance segmentation, point/scribble-supervised semantic segmentation and CLIP-guided semantic segmentation, demonstrate the superior performance of the proposed approach.</details>

### Langevin Quasi-Monte Carlo

**Authors:** Sifan Liu

**<details><summary>Abstract**</summary>

Langevin Monte Carlo (LMC) and its stochastic gradient versions are powerful algorithms for sampling from complex high-dimensional distributions. To sample from a distribution with density $\pi(\theta)\propto \exp(-U(\theta)) $, LMC iteratively generates the next sample by taking a step in the gradient direction $\nabla U$ with added Gaussian perturbations. Expectations w.r.t. the target distribution $\pi$ are estimated by averaging over LMC samples. In ordinary Monte Carlo, it is well known that the estimation error can be substantially reduced by replacing independent random samples by quasi-random samples like low-discrepancy sequences. In this work, we show that the estimation error of LMC can also be reduced by using quasi-random samples. Specifically, we propose to use completely uniformly distributed (CUD) sequences with certain low-discrepancy property to generate the Gaussian perturbations. Under smoothness and convexity conditions, we prove that LMC with a low-discrepancy CUD sequence achieves smaller error than standard LMC. The theoretical analysis is supported by compelling numerical experiments, which demonstrate the effectiveness of our approach.</details>

### Language Model Tokenizers Introduce Unfairness Between Languages

**Authors:** Aleksandar Petrov, Emanuele La Malfa, Philip Torr, Adel Bibi

**<details><summary>Abstract**</summary>

Recent language models have shown impressive multilingual performance, even when not explicitly trained for it.Despite this, there are concerns about the quality of their outputs across different languages.In this paper, we show how disparity in the treatment of different languages arises at the tokenization stage, well before a model is even invoked.The same text translated into different languages can have drastically different tokenization lengths, with differences up to 15 times in some cases.These disparities persist even for tokenizers that are intentionally trained for multilingual support.Character-level and byte-level models also exhibit over 4 times the difference in the encoding length for some language pairs.This induces unfair treatment for some language communities in regard to the cost of accessing commercial language services, the processing time and latency, as well as the amount of content that can be provided as context to the models.Therefore, we make the case that we should train future language models using multilingually fair subword tokenizers.</details>

### Language Models are Weak Learners

**Authors:** Hariharan Manikandan, Yiding Jiang, J. Zico Kolter

**<details><summary>Abstract**</summary>

A central notion in practical and theoretical machine learning is that of a *weak learner*, classifiers that achieve better-than-random performance (on any given distribution over data), even by a small margin.  Such weak learners form the practical basis for canonical machine learning methods such as boosting.  In this work, we illustrate that prompt-based large language models can operate effectively as said weak learners.  Specifically, we illustrate the use of a large language model (LLM) as a weak learner in a boosting algorithm applied to tabular data.  We show that by providing (properly sampled according to the distribution of interest) text descriptions of tabular data samples, LLMs can produce a summary of the samples that serves as a template for classification, and achieves the aim of acting as a weak learner on this task.  We incorporate these models into a boosting approach, which in many settings can leverage the knowledge within the LLM to outperform traditional tree-based boosting.  The model outperforms both few-shot learning and occasionally even more involved fine-tuning procedures, particularly for some tasks involving small numbers of data points.  The results illustrate the potential for prompt-based LLMs to function not just as few-shot learners themselves, but as components of larger machine learning models.</details>

### Laplacian Canonization: A Minimalist Approach to Sign and Basis Invariant Spectral Embedding

**Authors:** George Ma, Yifei Wang, Yisen Wang

**<details><summary>Abstract**</summary>

Spectral embedding is a powerful graph embedding technique that has received a lot of attention recently due to its effectiveness on Graph Transformers. However, from a theoretical perspective, the universal expressive power of spectral embedding comes at the price of losing two important invariance properties of graphs, sign and basis invariance, which also limits its effectiveness on graph data. To remedy this issue, many previous methods developed costly approaches to learn new invariants and suffer from high computation complexity. In this work, we explore a minimal approach that resolves the ambiguity issues by directly finding canonical directions for the eigenvectors, named Laplacian Canonization (LC). As a pure pre-processing method, LC is light-weighted and can be applied to any existing GNNs. We provide a thorough investigation, from theory to algorithm, on this approach, and discover an efficient algorithm named Maximal Axis Projection (MAP) that works for both sign and basis invariance and successfully canonizes more than 90\% of all eigenvectors. Experiments on real-world benchmark datasets like ZINC, MOLTOX21, and MOLPCBA show that MAP consistently outperforms existing methods while bringing minimal computation overhead. Code is available at \url{https://github.com/PKU-ML/LaplacianCanonization}.</details>

### Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning

**Authors:** Xinyi Wang, Wanrong Zhu, Michael Saxon, Mark Steyvers, William Yang Wang

**<details><summary>Abstract**</summary>

In recent years, pre-trained large language models (LLMs) have demonstrated remarkable efficiency in achieving an inference-time few-shot learning capability known as in-context learning. However, existing literature has highlighted the sensitivity of this capability to the selection of few-shot demonstrations. Current understandings of the underlying mechanisms by which this capability arises from regular language model pretraining objectives remain disconnected from the real-world LLMs. This study aims to examine the in-context learning phenomenon through a Bayesian lens, viewing real-world LLMs as latent variable models. On this premise, we propose an algorithm to select optimal demonstrations from a set of annotated data with a small LM, and then directly generalize the selected demonstrations to larger LMs. We demonstrate significant improvement over baselines, averaged over eight GPT models on eight real-world text classification datasets. We also demonstrate the real-world usefulness of our algorithm on GSM8K, a math word problem dataset. Our empirical findings support our hypothesis that LLMs implicitly infer a latent variable containing task information.</details>

### Large Language Models are Visual Reasoning Coordinators

**Authors:** Liangyu Chen, Bo Li, Sheng Shen, Jingkang Yang, Chunyuan Li, Kurt Keutzer, Trevor Darrell, Ziwei Liu

**<details><summary>Abstract**</summary>

Visual reasoning requires multimodal perception and commonsense cognition of the world. Recently, multiple vision-language models (VLMs) have been proposed with excellent commonsense reasoning ability in various domains. However, how to harness the collective power of these complementary VLMs is rarely explored. Existing methods like ensemble still struggle to aggregate these models with the desired higher-order communications. In this work, we propose Cola, a novel paradigm that coordinates multiple VLMs for visual reasoning. Our key insight is that a large language model (LLM) can efficiently coordinate multiple VLMs by facilitating natural language communication that leverages their distinct and complementary capabilities. Extensive experiments demonstrate that our instruction tuning variant, Cola-FT, achieves state-of-the-art performance on visual question answering (VQA), outside knowledge VQA, visual entailment, and visual spatial reasoning tasks. Moreover, we show that our in-context learning variant, Cola-Zero, exhibits competitive performance in zero and few-shot settings, without finetuning. Through systematic ablation studies and visualizations, we validate that a coordinator LLM indeed comprehends the instruction prompts as well as the separate functionalities of VLMs; it then coordinates them to enable impressive visual reasoning capabilities.</details>

### Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering

**Authors:** Noah Hollmann, Samuel Müller, Frank Hutter

**<details><summary>Abstract**</summary>

As the field of automated machine learning (AutoML) advances, it becomes increasingly important to incorporate domain knowledge into these systems.We present an approach for doing so by harnessing the power of large language models (LLMs). Specifically, we introduce Context-Aware Automated Feature Engineering (CAAFE), a feature engineering method for tabular datasets that utilizes an LLM to iteratively generate additional semantically meaningful features for tabular datasets based on the description of the dataset. The method produces both Python code for creating new features and explanations for the utility of the generated features.Despite being methodologically simple, CAAFE improves performance on 11 out of 14 datasets -- boosting mean ROC AUC performance from 0.798 to 0.822 across all dataset - similar to the improvement achieved by using a random forest instead of logistic regression on our datasets. Furthermore, CAAFE is interpretable by providing a textual explanation for each generated feature.CAAFE paves the way for more extensive semi-automation in data science tasks and emphasizes the significance of context-aware solutions that can extend the scope of AutoML systems to semantic AutoML. We release our [code](url{https://github.com/automl/CAAFE), a simple [demo](url{https://colab.research.google.com/drive/1mCA8xOAJZ4MaB_alZvyARTMjhl6RZf0a)</details>

### Large language models transition from integrating across position-yoked, exponential windows to structure-yoked, power-law windows

**Authors:** David Skrill, Samuel Norman-Haignere

**<details><summary>Abstract**</summary>

Modern language models excel at integrating across long temporal scales needed to encode linguistic meaning and show non-trivial similarities to biological neural systems. Prior work suggests that human brain responses to language exhibit hierarchically organized "integration windows" that substantially constrain the overall influence of an input token (e.g., a word) on the neural response. However, little prior work has attempted to use integration windows to characterize computations in large language models (LLMs). We developed a simple word-swap procedure for estimating integration windows from black-box language models that does not depend on access to gradients or knowledge of the model architecture (e.g., attention weights). Using this method, we show that trained LLMs exhibit stereotyped integration windows that are well-fit by a convex combination of an exponential and a power-law function, with a partial transition from exponential to power-law dynamics across network layers. We then introduce a metric for quantifying the extent to which these integration windows vary with structural boundaries (e.g., sentence boundaries), and using this metric, we show that integration windows become increasingly yoked to structure at later network layers. None of these findings were observed in an untrained model, which as expected integrated uniformly across its input. These results suggest that LLMs learn to integrate information in natural language using a stereotyped pattern: integrating across position-yoked, exponential windows at early layers, followed by structure-yoked, power-law windows at later layers. The methods we describe in this paper provide a general-purpose toolkit for understanding temporal integration in language models, facilitating cross-disciplinary research at the intersection of biological and artificial intelligence.</details>

### Learning Energy-based Model via Dual-MCMC Teaching

**Authors:** Jiali Cui, Tian Han

**<details><summary>Abstract**</summary>

This paper studies the fundamental learning problem of the energy-based model (EBM). Learning the EBM can be achieved using the maximum likelihood estimation (MLE), which typically involves the Markov Chain Monte Carlo (MCMC) sampling, such as the Langevin dynamics. However, the noise-initialized Langevin dynamics can be challenging in practice and hard to mix. This motivates the exploration of joint training with the generator model where the generator model serves as a complementary model to bypass MCMC sampling. However, such a method can be less accurate than the MCMC and result in biased EBM learning. While the generator can also serve as an initializer model for better MCMC sampling, its learning can be biased since it only matches the EBM and has no access to empirical training examples. Such biased generator learning may limit the potential of learning the EBM. To address this issue, we present a joint learning framework that interweaves the maximum likelihood learning algorithm for both the EBM and the complementary generator model. In particular, the generator model is learned by MLE to match both the EBM and the empirical data distribution, making it a more informative initializer for MCMC sampling of EBM. Learning generator with observed examples typically requires inference of the generator posterior. To ensure accurate and efficient inference, we adopt the MCMC posterior sampling and introduce a complementary inference model to initialize such latent MCMC sampling. We show that three separate models can be seamlessly integrated into our joint framework through two (dual-) MCMC teaching, enabling effective and efficient EBM learning.</details>

### [Spotlight] Learning Functional Transduction

**Authors:** Mathieu Chalvidal, Thomas Serre, Rufin VanRullen

**<details><summary>Abstract**</summary>

Research in statistical learning has polarized into two general approaches to perform regression analysis: Transductive methods construct estimates directly based on exemplar data using generic relational principles which might suffer from the curse of dimensionality. Conversely, inductive methods can potentially fit highly complex functions at the cost of compute-intensive solution searches. In this work, we leverage the theory of vector-valued Reproducing Kernel Banach Spaces (RKBS) to propose a hybrid approach: We show that transductive regression systems can be meta-learned with gradient descent to form efficient _in-context_ neural approximators of function defined over both finite and infinite-dimensional spaces (operator regression). Once trained, our _Transducer_ can almost instantaneously capture new functional relationships and produce original image estimates, given a few pairs of input and output examples. We demonstrate the benefit of our meta-learned transductive approach to model physical systems influenced by varying external factors with little data at a fraction of the usual deep learning training costs for partial differential equations and climate modeling applications.</details>

### Learning Interpretable Low-dimensional Representation via Physical Symmetry

**Authors:** Xuanjie Liu, Daniel Chin, Yichen Huang, Gus Xia

**<details><summary>Abstract**</summary>

We have recently seen great progress in learning interpretable music representations, ranging from basic factors, such as pitch and timbre, to high-level concepts, such as chord and texture. However, most methods rely heavily on music domain knowledge. It remains an open question what general computational principles *give rise to* interpretable representations, especially low-dim factors that agree with human perception. In this study, we take inspiration from modern physics and use *physical symmetry* as a self-consistency constraint for the latent space. Specifically, it requires the prior model that characterises the dynamics of the latent states to be *equivariant* with respect to certain group transformations. We show that physical symmetry leads the model to learn a *linear* pitch factor from unlabelled monophonic music audio in a self-supervised fashion. In addition, the same methodology can be applied to computer vision, learning a 3D Cartesian space from videos of a simple moving object without labels. Furthermore, physical symmetry naturally leads to *representation augmentation*, a new technique which improves sample efficiency.</details>

### Learning Invariant Representations with a Nonparametric Nadaraya-Watson Head

**Authors:** Alan Wang, Minh Nguyen, Mert Sabuncu

**<details><summary>Abstract**</summary>

Machine learning models will often fail when deployed in an environment with a data distribution that is different than the training distribution. When multiple environments are available during training, many methods exist that learn representations which are invariant across the different distributions, with the hope that these representations will be transportable to unseen domains. In this work, we present a nonparametric strategy for learning invariant representations based on the recently-proposed Nadaraya-Watson (NW) head. The NW head makes a prediction by comparing the learned representations of the query to the elements of a support set that consists of labeled data. We demonstrate that by manipulating the support set, one can encode different causal assumptions. In particular, restricting the support set to a single environment encourages the model to learn invariant features that do not depend on the environment. We present a causally-motivated setup for our modeling and training strategy and validate on three challenging real-world domain generalization tasks in computer vision.</details>

### Learning Large Graph Property Prediction via Graph Segment Training

**Authors:** Kaidi Cao, Mangpo Phothilimtha, Sami Abu-El-Haija, Dustin Zelle, Yanqi Zhou, Charith Mendis, Jure Leskovec, Bryan Perozzi

**<details><summary>Abstract**</summary>

Learning to predict properties of large graphs is challenging because each prediction requires the knowledge of an entire graph, while the amount of memory available during training is bounded. Here we propose Graph Segment Training (GST), a general framework that utilizes a divide-and-conquer approach to allow learning large graph property prediction with a constant memory footprint. GST first divides a large graph into segments and then backpropagates through only a few segments sampled per training iteration. We refine the GST paradigm by introducing a historical embedding table to efficiently obtain embeddings for segments not sampled for backpropagation. To mitigate the staleness of historical embeddings, we design two novel techniques. First, we finetune the prediction head to fix the input distribution shift. Second, we introduce Stale Embedding Dropout to drop some stale embeddings during training to reduce bias. We evaluate our complete method GST-EFD (with all the techniques together) on two large graph property prediction benchmarks: MalNet and TpuGraphs. Our experiments show that GST-EFD is both memory-efficient and fast, while offering a slight boost on test accuracy over a typical full graph training regime.</details>

### Learning Large-Scale MTP$_2$ Gaussian Graphical Models via Bridge-Block Decomposition

**Authors:** Xiwen WANG, Jiaxi Ying, Daniel Palomar

**<details><summary>Abstract**</summary>

This paper studies the problem of learning the large-scale Gaussian graphical models that are multivariate totally positive of order two ($\text{MTP}_2$). By introducing the concept of bridge, which commonly exists in large-scale sparse graphs, we show that the entire problem can be equivalently optimized through (1) several smaller-scaled sub-problems induced by a \emph{bridge-block decomposition} on the thresholded sample covariance graph and (2) a set of explicit solutions on entries corresponding to  \emph{bridges}. From practical aspect, this simple and provable discipline can be applied to break down a large problem into small tractable ones, leading to enormous reduction on the computational complexity and substantial improvements for all existing algorithms.  The synthetic and real-world experiments demonstrate that our proposed method presents a significant speed-up compared to the state-of-the-art benchmarks.</details>

### [Oral] Learning Linear Causal Representations from Interventions under General Nonlinear Mixing

**Authors:** Simon Buchholz, Goutham Rajendran, Elan Rosenfeld, Bryon Aragam, Bernhard Schölkopf, Pradeep Ravikumar

**Oral Presentation:** Tu, Dec 12, 13:40 -- Oral 2C

**<details><summary>Abstract**</summary>

We study the problem of learning causal representations from unknown, latent interventions in a general setting, where the latent distribution is Gaussian but the mixing function is completely general. We prove strong identifiability results given unknown single-node interventions, i.e., without having access to the intervention targets. This generalizes prior works which have focused on weaker classes, such as linear maps or paired counterfactual data. This is also the first instance of identifiability from non-paired interventions for deep neural network embeddings and general causal structures. Our proof relies on carefully uncovering the high-dimensional geometric structure present in the data distribution after a non-linear density transformation, which we capture by analyzing quadratic forms of precision matrices of the latent distributions. Finally, we propose a contrastive algorithm to identify the latent variables in practice and evaluate its performance on various tasks.</details>

### Learning Neural Implicit through Volume Rendering with Attentive Depth Fusion Priors

**Authors:** Pengchong Hu, Zhizhong Han

**<details><summary>Abstract**</summary>

Learning neural implicit representations has achieved remarkable performance in 3D reconstruction from multi-view images. Current methods use volume rendering to render implicit representations into either RGB or depth images that are supervised by the multi-view ground truth. However, rendering a view each time suffers from incomplete depth at holes and unawareness of occluded structures from the depth supervision, which severely affects the accuracy of geometry inference via volume rendering. To resolve this issue, we propose to learn neural implicit representations from multi-view RGBD images through volume rendering with an attentive depth fusion prior. Our prior allows neural networks to sense coarse 3D structures from the Truncated Signed Distance Function (TSDF) fused from all available depth images for rendering. The TSDF enables accessing the missing depth at holes on one depth image and the occluded parts that are invisible from the current view. By introducing a novel attention mechanism, we allow neural networks to directly use the depth fusion prior with the inferred occupancy as the learned implicit function. Our attention mechanism works with either a one-time fused TSDF that represents a whole scene or an incrementally fused TSDF that represents a partial scene in the context of Simultaneous Localization and Mapping (SLAM). Our evaluations on widely used benchmarks including synthetic and real-world scans show our superiority over the latest neural implicit methods.</details>

### Learning Sample Difficulty from Pre-trained Models for Reliable Prediction

**Authors:** Peng Cui, Dan Zhang, Zhijie Deng, Yinpeng Dong, Jun Zhu

**<details><summary>Abstract**</summary>

Large-scale pre-trained models have achieved remarkable success in many applications, but how to leverage them to improve the prediction reliability of downstream models is undesirably under-explored. Moreover, modern neural networks have been found to be poorly calibrated and make overconfident predictions regardless of inherent sample difficulty and data uncertainty. To address this issue, we propose to utilize large-scale pre-trained models to guide downstream model training with sample difficulty-aware entropy regularization. Pre-trained models that have been exposed to large-scale datasets and do not overfit the downstream training classes enable us to measure each training sample’s difficulty via feature-space Gaussian modeling and relative Mahalanobis distance computation. Importantly, by adaptively penalizing overconfident prediction based on the sample difficulty, we simultaneously improve accuracy and uncertainty calibration across challenging benchmarks (e.g., +0.55% ACC and −3.7% ECE on ImageNet1k using ResNet34), consistently surpassing competitive baselines for reliable prediction. The improved uncertainty estimate further improves selective classification (abstaining from erroneous predictions) and out-of-distribution detection.</details>

### Learning Space-Time Continuous Latent Neural PDEs from Partially Observed States

**Authors:** Valerii Iakovlev, Markus Heinonen, Harri Lähdesmäki

**<details><summary>Abstract**</summary>

We introduce a novel grid-independent model for learning partial differential equations (PDEs) from noisy and partial observations on irregular spatiotemporal grids. We propose a space-time continuous latent neural PDE model with an efficient probabilistic framework and a novel encoder design for improved data efficiency and grid independence. The latent state dynamics are governed by a PDE model that combines the collocation method and the method of lines. We employ amortized variational inference for approximate posterior estimation and utilize a multiple shooting technique for enhanced training speed and stability. Our model demonstrates state-of-the-art performance on complex synthetic and real-world datasets, overcoming limitations of previous approaches and effectively handling partially-observed data. The proposed model outperforms recent methods, showing its potential to advance data-driven PDE modeling and enabling robust, grid-independent modeling of complex partially-observed dynamic processes across various domains.</details>

### Learning Unseen Modality Interaction

**Authors:** Yunhua Zhang, Hazel Doughty, Cees Snoek

**<details><summary>Abstract**</summary>

Multimodal learning assumes all modality combinations of interest are available during training to learn cross-modal correspondences. In this paper, we challenge this modality-complete assumption for multimodal learning and instead strive for generalization to unseen modality combinations during inference. We pose the problem of unseen modality interaction and introduce a first solution. It exploits a module that projects the multidimensional features of different modalities into a common space with rich information preserved. This allows the information to be accumulated with a simple summation operation across available modalities. To reduce overfitting to less discriminative modality combinations during training, we further improve the model learning with pseudo-supervision indicating the reliability of a modality’s prediction. We demonstrate that our approach is effective for diverse tasks and modalities by evaluating it for multimodal video classification, robot state regression, and multimedia retrieval. Project website: https://xiaobai1217.github.io/Unseen-Modality-Interaction/.</details>

### Learning and Collusion in Multi-unit Auctions

**Authors:** Simina Branzei, Mahsa Derakhshan, Negin Golrezaei, Yanjun Han

**<details><summary>Abstract**</summary>

In a carbon auction, licenses for CO2 emissions are allocated among multiple interested players. Inspired by this setting, we consider repeated multi-unit auctions with uniform pricing, which are widely used in practice. Our contribution is to analyze these auctions in both the offline and online settings, by designing efficient bidding algorithms with low regret and giving regret lower bounds. We also analyze the quality of the equilibria  in  two  main variants of the auction, finding that one  variant is susceptible to collusion among the bidders while the other is not.</details>

### Learning the Efficient Frontier

**Authors:** Philippe Chatigny, Ivan Sergienko, Ryan Ferguson, Jordan Weir, Maxime Bergeron

**<details><summary>Abstract**</summary>

The efficient frontier (EF) is a fundamental resource allocation problem where one has to find an optimal portfolio maximizing a reward at a given level of risk. This optimal solution is traditionally found by solving a convex optimization problem. In this paper, we introduce NeuralEF: a fast neural approximation framework that robustly forecasts the result of the EF convex optimizations problems with respect to heterogeneous linear constraints and variable number of optimization inputs. By reformulating an optimization problem as a sequence to sequence problem, we show that NeuralEF is a viable solution to accelerate large-scale simulation while handling discontinuous behavior.</details>

### Learning threshold neurons via edge of stability

**Authors:** Kwangjun Ahn, Sebastien Bubeck, Sinho Chewi, Yin Tat Lee, Felipe Suarez, Yi Zhang

**<details><summary>Abstract**</summary>

Existing analyses of neural network training often operate under the unrealistic assumption of an extremely small learning rate. This lies in stark contrast to practical wisdom and empirical studies, such as the work of J. Cohen et al. (ICLR 2021), which exhibit startling new phenomena (the "edge of stability"' or "unstable convergence") and potential benefits for generalization in the large learning rate regime. Despite a flurry of recent works on this topic, however, the latter effect is still poorly understood. In this paper, we take a step towards understanding genuinely non-convex training dynamics with large learning rates by performing a detailed analysis of gradient descent for simplified models of two-layer neural networks. For these models, we provably establish the edge of stability phenomenon and discover a sharp phase transition for the step size below which the neural network fails to learn ``threshold-like'' neurons (i.e., neurons with a non-zero first-layer bias). This elucidates one possible mechanism by which the edge of stability can in fact lead to better generalization, as threshold neurons are basic building blocks with useful inductive bias for many tasks.</details>

### Learning to Discover Skills through Guidance

**Authors:** HYUNSEUNG KIM, BYUNG KUN LEE, Hojoon Lee, Dongyoon Hwang, Sejik Park, Kyushik Min, Jaegul Choo

**<details><summary>Abstract**</summary>

In the field of unsupervised skill discovery (USD), a major challenge is limited exploration, primarily due to substantial penalties when skills deviate from their initial trajectories. To enhance exploration, recent methodologies employ auxiliary rewards to maximize the epistemic uncertainty or entropy of states. However, we have identified that the effectiveness of these rewards declines as the environmental complexity rises. Therefore, we present a novel USD algorithm, skill **disco**very with gui**dance** (**DISCO-DANCE**), which (1) selects the guide skill that possesses the highest potential to reach unexplored states, (2) guides other skills to follow guide skill, then (3) the guided skills are dispersed to maximize their discriminability in unexplored states. Empirical evaluation demonstrates that DISCO-DANCE outperforms other USD baselines in challenging environments, including two navigation benchmarks and a continuous control benchmark. Qualitative visualizations and code of DISCO-DANCE are available at https://mynsng.github.io/discodance/.</details>

### [Spotlight] Learning to Receive Help: Intervention-Aware Concept Embedding Models

**Authors:** Mateo Espinosa Zarlenga, Katie Collins, Krishnamurthy Dvijotham, Adrian Weller, Zohreh Shams, Mateja Jamnik

**<details><summary>Abstract**</summary>

Concept Bottleneck Models (CBMs) tackle the opacity of neural architectures by constructing and explaining their predictions using a set of high-level concepts. A special property of these models is that they permit concept interventions, wherein users can correct mispredicted concepts and thus improve the model's performance. Recent work, however, has shown that intervention efficacy can be highly dependent on the order in which concepts are intervened on and on the model's architecture and training hyperparameters. We argue that this is rooted in a CBM's lack of train-time incentives for the model to be appropriately receptive to concept interventions. To address this, we propose Intervention-aware Concept Embedding models (IntCEMs), a novel CBM-based architecture and training paradigm that improves a model's receptiveness to test-time interventions. Our model learns a concept intervention policy in an end-to-end fashion from where it can sample meaningful intervention trajectories at train-time. This conditions IntCEMs to effectively select and receive concept interventions when deployed at test-time. Our experiments show that IntCEMs significantly outperform state-of-the-art concept-interpretable models when provided with test-time concept interventions, demonstrating the effectiveness of our approach.</details>

### Learning via Wasserstein-Based High Probability Generalisation Bounds

**Authors:** Paul Viallard, Maxime Haddouche, Umut Simsekli, Benjamin Guedj

**<details><summary>Abstract**</summary>

Minimising upper bounds on the population risk or the generalisation gap has been widely used in structural risk minimisation (SRM) -- this is in particular at the core of PAC-Bayesian learning. Despite its successes and unfailing surge of interest in recent years, a limitation of the PAC-Bayesian framework is that most bounds involve a Kullback-Leibler (KL) divergence term (or its variations), which might exhibit erratic behavior and fail to capture the underlying geometric structure of the learning problem -- hence restricting its use in practical applications.As a remedy, recent studies have attempted to replace the KL divergence in the PAC-Bayesian bounds with the Wasserstein distance. Even though these bounds alleviated the aforementioned issues to a certain extent, they either hold in expectation, are for bounded losses, or are nontrivial to minimize in an SRM framework.  In this work, we contribute to this line of research and prove novel Wasserstein distance-based PAC-Bayesian generalisation bounds for both batch learning with independent and identically distributed (i.i.d.) data, and online learning with potentially non-i.i.d. data. Contrary to previous art, our bounds are stronger in the sense that (i) they hold with high probability, (ii) they apply to unbounded (potentially heavy-tailed) losses, and (iii) they lead to optimizable training objectives that can be used in SRM. As a result we derive novel Wasserstein-based PAC-Bayesian learning algorithms and we illustrate their empirical advantage on a variety of experiments.</details>

### Lightweight Vision Transformer with Bidirectional Interaction

**Authors:** Qihang Fan, Huaibo Huang, Xiaoqiang Zhou, Ran He

**<details><summary>Abstract**</summary>

Recent advancements in vision backbones have significantly improved their performance by simultaneously modeling images’ local and global contexts. However, the bidirectional interaction between these two contexts has not been well explored and exploited, which is important in the human visual system. This paper proposes a **F**ully **A**daptive **S**elf-**A**ttention (FASA) mechanism for vision transformer to model the local and global information as well as the bidirectional interaction between them in context-aware ways. Specifically, FASA employs self-modulated convolutions to adaptively extract local representation while utilizing self-attention in down-sampled space to extract global representation. Subsequently, it conducts a bidirectional adaptation process between local and global representation to model their interaction. In addition, we introduce a fine-grained downsampling strategy to enhance the down-sampled self-attention mechanism for finer-grained global perception capability. Based on FASA, we develop a family of lightweight vision backbones, **F**ully **A**daptive **T**ransformer (FAT) family. Extensive experiments on multiple vision tasks demonstrate that FAT achieves impressive performance. Notably, FAT accomplishes a **77.6%** accuracy on ImageNet-1K using only **4.5M** parameters and **0.7G** FLOPs, which surpasses the most advanced ConvNets and Transformers with similar model size and computational costs. Moreover, our model exhibits faster speed on modern GPU compared to other models.</details>

### Likelihood Ratio Confidence Sets for Sequential Decision Making

**Authors:** Nicolas Emmenegger, Mojmir Mutny, Andreas Krause

**<details><summary>Abstract**</summary>

Certifiable, adaptive uncertainty estimates for unknown quantities are an essential ingredient of sequential decision-making algorithms. Standard approaches rely on problem-dependent concentration results and are limited to a specific combination of parameterization, noise family, and estimator. In this paper, we revisit the likelihood-based inference principle and propose to use \emph{likelihood ratios} to construct \emph{any-time valid} confidence sequences without requiring specialized treatment in each application scenario. Our method is especially suitable for problems with well-specified likelihoods, and the resulting sets always maintain the prescribed coverage in a model-agnostic manner. The size of the sets depends on a choice of estimator sequence in the likelihood ratio. We discuss how to provably choose the best sequence of estimators and shed light on connections to online convex optimization with algorithms such as Follow-the-Regularized-Leader. To counteract the initially large bias of the estimators, we propose a reweighting scheme that also opens up deployment in non-parametric settings such as RKHS function classes. We provide a \emph{non-asymptotic} analysis of the likelihood ratio confidence sets size for generalized linear models, using insights from convex duality and online learning. We showcase the practical strength of our method on generalized linear bandit problems, survival analysis, and bandits with various additive noise distributions.</details>

### [Oral] Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment

**Authors:** Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, Gal Chechik

**Oral Presentation:** Tu, Dec 12, 13:55 -- Oral 2B

**<details><summary>Abstract**</summary>

Text-conditioned image generation models often generate incorrect associations between entities and their visual attributes. This reflects an impaired mapping between linguistic binding of entities and modifiers in the prompt and visual binding of the corresponding elements in the generated image. As one example, a query like ``a pink sunflower and a yellow flamingo'' may incorrectly produce an image of a yellow sunflower and a pink flamingo. To remedy this issue, we propose SynGen, an approach which first syntactically analyses the prompt to identify entities and their modifiers, and then uses a novel loss function that encourages the cross-attention maps to agree with the linguistic binding reflected by the syntax. Specifically, we encourage large overlap between attention maps of entities and their modifiers, and small overlap with other entities and modifier words. The loss is optimized during inference, without retraining or fine-tuning the model. Human evaluation on three datasets, including one new and challenging set, demonstrate significant improvements of SynGen compared with current state of the art methods. This work highlights how making use of sentence structure during inference can efficiently and substantially improve the faithfulness of text-to-image generation.</details>

### [Spotlight] LinkerNet: Fragment Poses and Linker Co-Design with 3D Equivariant Diffusion

**Authors:** Jiaqi Guan, Xingang Peng, PeiQi Jiang, Yunan Luo, Jian Peng, Jianzhu Ma

**<details><summary>Abstract**</summary>

Targeted protein degradation techniques, such as PROteolysis TArgeting Chimeras (PROTACs), have emerged as powerful tools for selectively removing disease-causing proteins. One challenging problem in this field is designing a linker to connect different molecular fragments to form a stable drug-candidate molecule. Existing models for linker design assume that the relative positions of the fragments are known, which may not be the case in real scenarios. In this work, we address a more general problem where the poses of the fragments are *unknown* in 3D space. We develop a 3D equivariant diffusion model that jointly learns the generative process of both fragment poses and the 3D structure of the linker. By viewing fragments as rigid bodies, we design a fragment pose prediction module inspired by the Newton-Euler equations in rigid body mechanics. Empirical studies on ZINC and PROTAC-DB datasets demonstrate that our model can generate chemically valid, synthetically-accessible,  and low-energy molecules under both unconstrained and constrained generation settings.</details>

### Live Graph Lab: Towards Open, Dynamic and Real Transaction Graphs with NFT

**Authors:** Zhen Zhang, Bingqiao Luo, Shengliang Lu, Bingsheng He

**<details><summary>Abstract**</summary>

Numerous studies have been conducted to investigate the properties of large-scale temporal graphs. Despite the ubiquity of these graphs in real-world scenarios, it's usually impractical for us to obtain the whole real-time graphs due to privacy concerns and technical limitations. In this paper, we introduce the concept of {\it Live Graph Lab} for temporal graphs, which enables open, dynamic and real transaction graphs from blockchains. Among them, Non-fungible tokens (NFTs) have become one of the most prominent parts of blockchain over the past several years. With more than \$40 billion market capitalization, this decentralized ecosystem produces massive, anonymous and real transaction activities, which naturally forms a complicated transaction network. However, there is limited understanding about the characteristics of this emerging NFT ecosystem from a temporal graph analysis perspective. To mitigate this gap, we instantiate a live graph with NFT transaction network and investigate its dynamics to provide new observations and insights. Specifically, through downloading and parsing the NFT transaction activities, we obtain a temporal graph with more than 4.5 million nodes and 124 million edges. Then, a series of measurements are presented to understand the properties of the NFT ecosystem. Through comparisons with social, citation, and web networks, our analyses give intriguing findings and point out potential directions for future exploration. Finally, we also study machine learning models in this live graph to enrich the current datasets and provide new opportunities for the graph community. The source codes and dataset are available at https://livegraphlab.github.io.</details>

### Long-Term Fairness with Unknown Dynamics

**Authors:** Tongxin Yin, Reilly Raab, Mingyan Liu, Yang Liu

**<details><summary>Abstract**</summary>

While machine learning can myopically reinforce social inequalities, it may also be used to dynamically seek equitable outcomes. In this paper, we formalize long-term fairness as an online reinforcement learning problem for a policy affecting human populations. This formulation accommodates dynamical control objectives, such as achieving equitable population states, that cannot be incorporated into static formulations of fairness. We demonstrate that algorithmic solutions to the proposed fairness problem can adapt to unknown dynamics and, by sacrificing short-term incentives, drive the policy-population system towards more desirable equilibria. For the proposed setting, we develop an algorithm that adapts recent work in online learning and prove that this algorithm achieves simultaneous probabilistic bounds on cumulative loss and cumulative violations of fairness. In the classification setting subject to group fairness, we compare our proposed algorithm to several baselines, including the repeated retraining of myopic or distributionally robust classifiers, and to a deep reinforcement learning algorithm that lacks fairness guarantees. Our experiments model human populations according to evolutionary game theory and integrate real-world datasets.</details>

### Lookup Table meets Local Laplacian Filter: Pyramid Reconstruction Network for Tone Mapping

**Authors:** Feng Zhang, Ming Tian, Zhiqiang Li, Bin Xu, Qingbo Lu, Changxin Gao, Nong Sang

**<details><summary>Abstract**</summary>

Tone mapping aims to convert high dynamic range (HDR) images to low dynamic range (LDR) representations, a critical task in the camera imaging pipeline. In recent years, 3-Dimensional LookUp Table (3D LUT) based methods have gained attention due to their ability to strike a favorable balance between enhancement performance and computational efficiency. However, these methods often fail to deliver satisfactory results in local areas since the look-up table is a global operator for tone mapping, which works based on pixel values and fails to incorporate crucial local information. To this end, this paper aims to address this issue by exploring a novel strategy that integrates global and local operators by utilizing closed-form Laplacian pyramid decomposition and reconstruction. Specifically, we employ image-adaptive 3D LUTs to manipulate the tone in the low-frequency image by leveraging the specific characteristics of the frequency information. Furthermore, we utilize local Laplacian filters to refine the edge details in the high-frequency components in an adaptive manner. Local Laplacian filters are widely used to preserve edge details in photographs, but their conventional usage involves manual tuning and fixed implementation within camera imaging pipelines or photo editing tools. We propose to learn parameter value maps progressively for local Laplacian filters from annotated data using a lightweight network. Our model achieves simultaneous global tone manipulation and local edge detail preservation in an end-to-end manner. Extensive experimental results on two benchmark datasets demonstrate that the proposed method performs favorably against state-of-the-art methods.</details>

### Loss Decoupling for Task-Agnostic Continual Learning

**Authors:** Yan-Shuo Liang, Wu-Jun Li

**<details><summary>Abstract**</summary>

Continual learning requires the model to learn multiple tasks in a sequential order. To perform continual learning, the model must possess the abilities to maintain performance on old tasks (stability) and adapt itself to learn new tasks (plasticity). Task-agnostic problem in continual learning is a challenging problem, in which task identities are not available in the inference stage and hence the model must learn to distinguish all the classes in all the tasks. In task-agnostic problem, the model needs to learn two new objectives for learning a new task, including distinguishing new classes from old classes and distinguishing between different new classes. For task-agnostic problem, replay-based methods are commonly used. These methods update the model with both saved old samples and new samples for continual learning. Most existing replay-based methods mix the two objectives in task-agnostic problem together, inhibiting the models from achieving a good trade-off between stability and plasticity. In this paper, we propose a simple yet effective method, called loss decoupling (LODE), for task-agnostic continual learning. LODE separates the two objectives for the new task by decoupling the loss of the new task. As a result, LODE can assign different weights for different objectives, which provides a way to obtain a better trade-off between stability and plasticity than those methods with coupled loss. Experiments show that LODE can outperform existing state-of-the-art replay-based methods on multiple continual learning datasets.</details>

### LuminAIRe: Illumination-Aware Conditional Image Repainting for Lighting-Realistic Generation

**Authors:** Jiajun Tang, Haofeng Zhong, Shuchen Weng, Boxin Shi

**<details><summary>Abstract**</summary>

We present the ilLumination-Aware conditional Image Repainting (LuminAIRe) task to address the unrealistic lighting effects in recent conditional image repainting (CIR) methods. The environment lighting and 3D geometry conditions are explicitly estimated from given background images and parsing masks using a parametric lighting representation and learning-based priors. These 3D conditions are then converted into illumination images through the proposed physically-based illumination rendering and illumination attention module. With the injection of illumination images, physically-correct lighting information is fed into the lighting-realistic generation process and repainted images with harmonized lighting effects in both foreground and background regions can be acquired, whose superiority over the results of state-of-the-art methods is confirmed through extensive experiments. For facilitating and validating the LuminAIRe task, a new dataset Car-LuminAIRe with lighting annotations and rich appearance variants is collected.</details>

### M$^{2}$SODAI: Multi-Modal Maritime Object Detection Dataset With RGB and Hyperspectral Image Sensors

**Authors:** Jonggyu Jang, Sangwoo Oh, Youjin Kim, Dongmin Seo, Youngchol Choi, Hyun Jong Yang

**<details><summary>Abstract**</summary>

Object detection in aerial images is a growing area of research, with maritime object detection being a particularly important task for reliable surveillance, monitoring, and active rescuing. Notwithstanding astonishing advances of computer visiontechnologies, detecting ships and floating matters in these images are challenging due to factors such as object distance. What makes it worse is pervasive sea surface effects such as sunlight reflection, wind, and waves. Hyperspectral image (HSI) sensors, providing more than 100 channels in wavelengths of visible and near-infrared, can extract intrinsic information of materials from a few pixels of HSIs.The advent of HSI sensors motivates us to leverage HSIs to circumvent false positives due to the sea surface effects.Unfortunately, there are few public HSI datasets due to the high cost and labor involved in collecting them, hindering object detection research based on HSIs. We have collected and annotated a new dataset called ``Multi-Modal Ship and flOating matter Detection in Aerial Images (M$^{2}$SODAI),'', which includes synchronized image pairs of RGB and HSI data, along with bounding box labels for nearly 6,000 instances per category. We also propose a new multi-modal extension of the feature pyramid network called DoubleFPN.Extensive experiments on our benchmark demonstrate that fusion of RGB and HSI data can enhance mAP, especially in the presence of the sea surface effects.</details>

### MG-ViT: A Multi-Granularity Method for Compact and Efficient Vision Transformers

**Authors:** Yu Zhang, Yepeng Liu, Duoqian Miao, Qi Zhang, Yiwei Shi, Liang Hu

**<details><summary>Abstract**</summary>

Vision Transformer (ViT) faces obstacles in wide application due to its huge computational cost. Almost all existing studies on compressing ViT adopt the manner of splitting an image with a single granularity, with very few exploration of splitting an image with multi-granularity. As we know, important information often randomly concentrate in few regions of an image, necessitating multi-granularity attention allocation to an image. Enlightened by this, we introduce the multi-granularity strategy to compress ViT, which is simple but effective. We propose a two-stage multi-granularity framework, MG-ViT, to balance ViT’s performance and computational cost. In single-granularity inference stage, an input image is split into a small number of patches for simple inference. If necessary, multi-granularity inference stage will be instigated, where the important patches are further subsplit into multi-finer-grained patches for subsequent inference. Moreover, prior studies on compression only for classification, while we extend the multi-granularity strategy to hierarchical ViT for downstream tasks such as detection and segmentation. Extensive experiments Prove the effectiveness of the multi-granularity strategy. For instance, on ImageNet, without any loss of performance, MG-ViT reduces 47\% FLOPs of LV-ViT-S and 56\% FLOPs of DeiT-S.</details>

### MMGP: a Mesh Morphing Gaussian Process-based machine learning method for regression of physical problems under nonparametrized geometrical variability

**Authors:** Fabien Casenave, Brian Staber, Xavier Roynard

**<details><summary>Abstract**</summary>

When learning simulations for modeling physical phenomena in industrial designs, geometrical variabilities are of prime interest. While classical regression techniques prove effective for parameterized geometries, practical scenarios often involve the absence of shape parametrization during the inference stage, leaving us with only mesh discretizations as available data. Learning simulations from such mesh-based representations poses significant challenges, with recent advances relying heavily on deep graph neural networks to overcome the limitations of conventional machine learning approaches. Despite their promising results, graph neural networks exhibit certain drawbacks, including their dependency on extensive datasets and limitations in providing built-in predictive uncertainties or handling large meshes. In this work, we propose a machine learning method that do not rely on graph neural networks. Complex geometrical shapes and variations with fixed topology are dealt with using well-known mesh morphing onto a common support, combined with classical dimensionality reduction techniques and Gaussian processes. The proposed methodology can easily deal with large meshes without the need for explicit shape parameterization and provides crucial predictive uncertainties, which are essential for informed decision-making. In the considered numerical experiments, the proposed method is competitive with respect to existing graph neural networks, regarding training efficiency and accuracy of the predictions.</details>

### Many-body Approximation for Non-negative Tensors

**Authors:** Kazu Ghalamkari, Mahito Sugiyama, Yoshinobu Kawahara

**<details><summary>Abstract**</summary>

We present an alternative approach to decompose non-negative tensors, called many-body approximation. Traditional decomposition methods assume low-rankness in the representation, resulting in difficulties in global optimization and target rank selection. We avoid these problems by energy-based modeling of tensors, where a tensor and its mode correspond to a probability distribution and a random variable, respectively. Our model can be globally optimized in terms of the KL divergence minimization by taking the interaction between variables (that is, modes), into account that can be tuned more intuitively than ranks. Furthermore, we visualize interactions between modes as tensor networks and reveal a nontrivial relationship between many-body approximation and low-rank approximation. We demonstrate the effectiveness of our approach in tensor completion and approximation.</details>

### Mask Propagation for Efficient Video Semantic Segmentation

**Authors:** Yuetian Weng, Mingfei Han, Haoyu He, Mingjie Li, Lina Yao, Xiaojun Chang, Bohan Zhuang

**<details><summary>Abstract**</summary>

Video Semantic Segmentation (VSS) involves assigning a semantic label to each pixel in a video sequence. Prior work in this field has demonstrated promising results by extending image semantic segmentation models to exploit temporal relationships across video frames; however, these approaches often incur significant computational costs. In this paper, we propose an efficient mask propagation framework for VSS, called MPVSS. Our approach first employs a strong query-based image segmentor on sparse key frames to generate accurate binary masks and class predictions. We then design a flow estimation module utilizing the learned queries to generate a set of segment-aware flow maps, each associated with a mask prediction from the key frame. Finally, the mask-flow pairs are warped to serve as the mask predictions for the non-key frames. By reusing predictions from key frames, we circumvent the need to process a large volume of video frames individually with resource-intensive segmentors, alleviating temporal redundancy and significantly reducing computational costs. Extensive experiments on VSPW and Cityscapes demonstrate that our mask propagation framework achieves SOTA accuracy and efficiency trade-offs. For instance, our best model with Swin-L backbone outperforms the SOTA MRCFA using MiT-B5 by 4.0% mIoU, requiring only 26% FLOPs on the VSPW dataset. Moreover, our framework reduces up to 4× FLOPs compared to the per-frame Mask2Former baseline with only up to 2% mIoU degradation on the Cityscapes validation set. Code is available at https://github.com/ziplab/MPVSS.</details>

### Masked Image Residual Learning for Scaling Deeper Vision Transformers

**Authors:** Guoxi Huang, Hongtao Fu, Adrian G. Bors

**<details><summary>Abstract**</summary>

Deeper Vision Transformers (ViTs) are more challenging to train. We expose a degradation problem in deeper layers of ViT when using masked image modeling (MIM) for pre-training.To ease the training of deeper ViTs, we introduce a self-supervised learning framework called  $\textbf{M}$asked $\textbf{I}$mage $\textbf{R}$esidual $\textbf{L}$earning ($\textbf{MIRL}$), which significantly alleviates the degradation problem, making scaling ViT along depth a promising direction for performance upgrade. We reformulate the pre-training objective for deeper layers of ViT as learning to recover the residual of the masked image.We provide extensive empirical evidence showing that deeper ViTs can be effectively optimized using MIRL and easily gain accuracy from increased depth. With the same level of computational complexity as ViT-Base and ViT-Large, we instantiate $4.5{\times}$ and $2{\times}$ deeper ViTs, dubbed ViT-S-54 and ViT-B-48.The deeper ViT-S-54, costing $3{\times}$ less than ViT-Large, achieves performance on par with ViT-Large.ViT-B-48 achieves 86.2\% top-1 accuracy on ImageNet. On one hand, deeper ViTs pre-trained with MIRL exhibit excellent generalization capabilities on downstream tasks, such as object detection and semantic segmentation. On the other hand, MIRL demonstrates high pre-training efficiency. With less pre-training time, MIRL yields competitive performance compared to other approaches.</details>

### [Spotlight] Maximize to Explore: One Objective Function Fusing Estimation, Planning, and Exploration

**Authors:** Zhihan Liu, Miao Lu, WEI XIONG, Han Zhong, Hao Hu, Shenao Zhang, Sirui Zheng, Zhuoran Yang, Zhaoran Wang

**<details><summary>Abstract**</summary>

In reinforcement learning (RL), balancing exploration and exploitation is crucial for achieving an optimal policy in a sample-efficient way. To this end, existing sample- efficient algorithms typically consist of three components: estimation, planning, and exploration. However, to cope with general function approximators, most of them involve impractical algorithmic components to incentivize exploration, such as data-dependent level-set constraints or complicated sampling procedures. To address this challenge, we propose an easy-to-implement RL framework called Maximize to Explore (MEX), which only needs to optimize unconstrainedly a single objective that integrates the estimation and planning components while balancing exploration and exploitation automatically. Theoretically, we prove that the MEX achieves a sublinear regret with general function approximators and is extendable to the zero-sum Markov game setting. Meanwhile, we adapt deep RL baselines to design practical versions of MEX in both the model-based and model-free settings, which outperform baselines in various MuJoCo environments with sparse reward by a stable margin. Compared with existing sample-efficient algorithms with general function approximators, MEX achieves similar sample efficiency while also enjoying a lower computational cost and is more compatible with modern deep RL methods.</details>

### MeGraph: Capturing Long-Range Interactions by Alternating Local and Hierarchical Aggregation on Multi-Scaled Graph Hierarchy

**Authors:** Honghua Dong, Jiawei Xu, Yu Yang, Rui Zhao, Shiwen Wu, Chun Yuan, Xiu Li, Chris Maddison, Lei Han

**<details><summary>Abstract**</summary>

Graph neural networks, which typically exchange information between local neighbors, often struggle to capture long-range interactions (LRIs) within the graph. Building a graph hierarchy via graph pooling methods is a promising approach to address this challenge; however, hierarchical information propagation cannot entirely take over the role of local information aggregation. To balance locality and hierarchy, we integrate the local and hierarchical structures, represented by intra- and inter-graphs respectively, of a multi-scale graph hierarchy into a single mega graph. Our proposed MeGraph model consists of multiple layers alternating between local and hierarchical information aggregation on the mega graph. Each layer first performs local-aware message-passing on graphs of varied scales via the intra-graph edges, then fuses information across the entire hierarchy along the bidirectional pathways formed by inter-graph edges. By repeating this fusion process, local and hierarchical information could intertwine and complement each other. To evaluate our model, we establish a new Graph Theory Benchmark designed to assess LRI capture ability, in which MeGraph demonstrates dominant performance. Furthermore, MeGraph exhibits superior or equivalent performance to state-of-the-art models on the Long Range Graph Benchmark. The experimental results on commonly adopted real-world datasets further demonstrate the broad applicability of MeGraph.</details>

### [Spotlight] Mean-field Langevin dynamics: Time-space discretization, stochastic gradient, and variance reduction

**Authors:** Taiji Suzuki, Denny Wu, Atsushi Nitanda

**<details><summary>Abstract**</summary>

The mean-field Langevin dynamics (MFLD) is a nonlinear generalization of the Langevin dynamics that incorporates a distribution-dependent drift, and it naturally arises from the optimization of two-layer neural networks via (noisy) gradient descent. Recent works have shown that MFLD globally minimizes an entropy-regularized convex functional in the space of measures. However, all prior analyses assumed the infinite-particle or continuous-time limit, and cannot handle stochastic gradient updates. We provide a general framework to prove a uniform-in-time propagation of chaos for MFLD that takes into account the errors due to finite-particle approximation, time-discretization, and stochastic gradient. To demonstrate the wide applicability of our framework, we establish quantitative convergence rate guarantees to the regularized global optimal solution for $(i)$ a wide range of learning problems such as mean-field neural network and MMD minimization, and $(ii)$ different gradient estimators including SGD and SVRG. Despite the generality of our results, we achieve an improved convergence rate in both the SGD and SVRG settings when specialized to the standard Langevin dynamics.</details>

### Med-UniC: Unifying Cross-Lingual Medical Vision-Language Pre-Training by Diminishing Bias

**Authors:** Zhongwei Wan, Che Liu, Mi Zhang, Jie Fu, Benyou Wang, Sibo Cheng, Lei Ma, César Quilodrán-Casas, Rossella Arcucci

**<details><summary>Abstract**</summary>

The scarcity of data presents a critical obstacle to the efficacy of medical vision-language pre-training (VLP). A potential solution lies in the combination of datasets from various language communities.Nevertheless, the main challenge stems from the complexity of integrating diverse syntax and semantics, language-specific medical terminology, and culture-specific implicit knowledge. Therefore, one crucial aspect to consider is the presence of community bias caused by different languages.This paper presents a novel framework named Unifying Cross-Lingual Medical Vision-Language Pre-Training (\textbf{Med-UniC}), designed to integrate multi-modal medical data from the two most prevalent languages, English and Spanish. Specifically, we propose \textbf{C}ross-lingual \textbf{T}ext Alignment \textbf{R}egularization (\textbf{CTR}) to explicitly unify cross-lingual semantic representations of medical reports originating from diverse language communities. \textbf{CTR} is optimized through latent language disentanglement, rendering our optimization objective to not depend on negative samples, thereby significantly mitigating the bias from determining positive-negative sample pairs within analogous medical reports. Furthermore, it ensures that the cross-lingual representation is not biased toward any specific language community.\textbf{Med-UniC} reaches superior performance across 5 medical image tasks and 10 datasets encompassing over 30 diseases, offering a versatile framework for unifying multi-modal medical data within diverse linguistic communities.The experimental outcomes highlight the presence of community bias in cross-lingual VLP. Reducing this bias enhances the performance not only in vision-language tasks but also in uni-modal visual tasks.</details>

### Meet in the Middle: A New Pre-training Paradigm

**Authors:** Anh Nguyen, Nikos Karampatziakis, Weizhu Chen

**<details><summary>Abstract**</summary>

Most language models (LMs) are trained and applied in an autoregressive left-to-right fashion, predicting the next token from the preceding ones. However, this ignores that the full sequence is available during training. In this paper, we introduce ``Meet in the Middle'' (MIM) a new pre-training paradigm that improves data efficiency by training in two directions, left-to-right and right-to-left, and encouraging the respective modelsto agree on their token distribution for each position. While the primary outcome is an improved left-to-right LM,we also obtain secondary benefits in the infilling task. There, we leverage the two pre-trained directions to propose an infilling procedure that builds the completion simultaneously from both sides. We conduct extensive experiments on both programming and natural languages and show that MIM significantly surpasses existing pre-training paradigms, in both left-to-right generation as well as infilling.Code and models available at https://github.com/microsoft/Meet-in-the-Middle</details>

### Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization

**Authors:** Jeonghoon Kim, Jung Hyun Lee, Sungdong Kim, Joonsuk Park, Kang Min Yoo, Se Jung Kwon, Dongsoo Lee

**<details><summary>Abstract**</summary>

Large language models (LLMs) face the challenges in fine-tuning and deployment due to their high memory demands and computational costs. While parameter-efficient fine-tuning (PEFT) methods aim to reduce the memory usage of the optimizer state during fine-tuning, the inherent size of pre-trained LLM weights continues to be a pressing concern.  Even though quantization techniques are widely proposed to ease memory demands and accelerate LLM inference, most of these techniques are geared towards the deployment phase.To bridge this gap, this paper presents Parameter-Efficient and Quantization-aware Adaptation (PEQA) – a simple yet effective method that combines the advantages of PEFT with quantized LLMs. By updating solely the quantization scales, PEQA can be directly applied to quantized LLMs, ensuring seamless task transitions. Parallel to existing PEFT methods, PEQA significantly reduces the memory overhead associated with the optimizer state. Furthermore, it leverages the advantages of quantization to substantially reduce model sizes. Even after fine-tuning, the quantization structure of a PEQA-tuned LLM remains intact, allowing for accelerated inference on the deployment stage.We employ PEQA-tuning for task-specific adaptation on LLMs with up to $65$ billion parameters. To assess the logical reasoning and language comprehension of PEQA-tuned LLMs, we fine-tune low-bit quantized LLMs using a instruction dataset. Our results show that even when LLMs are quantized to below 4-bit precision, their capabilities in language modeling, few-shot in-context learning, and comprehension can be resiliently restored to (or even improved over) their full-precision original performances with PEQA.</details>

### Mip-Grid: Anti-aliased Grid Representations for Neural Radiance Fields

**Authors:** Seungtae Nam, Daniel Rho, Jong Hwan Ko, Eunbyung Park

**<details><summary>Abstract**</summary>

Despite the remarkable achievements of neural radiance fields (NeRF) in representing 3D scenes and generating novel view images, the aliasing issue, rendering 'jaggies' or 'blurry' images at varying camera distances, remains unresolved in most existing approaches. The recently proposed mip-NeRF has effectively addressed this challenge by introducing integrated positional encodings (IPE). However, it relies on MLP architecture to represent the radiance fields, missing out on the fast training speed offered by the latest grid-based methods. In this work, we present mip-Grid, a novel approach that integrates anti-aliasing techniques into grid-based representations for radiance fields, mitigating the aliasing artifacts while enjoying fast training time. Notably, the proposed method uses a single-scale shared grid representation and a single-sampling approach, which only introduces minimal additions to the model parameters and computational costs. To handle scale ambiguity, mip-Grid generates multiple grids by applying simple convolution operations over the shared grid and uses the scale-aware coordinate to retrieve the appropriate features from the generated multiple grids. To test the effectiveness, we incorporated the proposed approach into the two recent representative grid-based methods, TensoRF and K-Planes. The experimental results demonstrated that mip-Grid greatly improved the rendering performance of both methods and showed comparable performance to mip-NeRF on multi-scale datasets while achieving significantly faster training time.</details>

### Mitigating Over-smoothing in Transformers via Regularized Nonlocal Functionals

**Authors:** Tam Nguyen, Tan Nguyen, Richard Baraniuk

**<details><summary>Abstract**</summary>

Transformers have achieved remarkable success in a wide range of natural language processing and computer vision applications. However, the representation capacity of a deep transformer model is degraded due to the over-smoothing issue in which the token representations become identical when the model's depth grows. In this work, we show that self-attention layers in transformers minimize a functional which promotes smoothness, thereby causing token uniformity. We then propose a novel regularizer that penalizes the norm of the difference between the smooth output tokens from self-attention and the input tokens to preserve the fidelity of the tokens. Minimizing the resulting regularized energy functional, we derive the Neural Transformer with a Regularized Nonlocal Functional (NeuTRENO), a novel class of transformer models that can mitigate the over-smoothing issue. We empirically demonstrate the advantages of NeuTRENO over the baseline transformers and state-of-the-art methods in reducing the over-smoothing of token representations on various practical tasks, including object classification, image segmentation, and language modeling.</details>

### Mitigating the Effect of Incidental Correlations on Part-based Learning

**Authors:** Gaurav Bhatt, Deepayan Das, Leonid Sigal, Vineeth N Balasubramanian

**<details><summary>Abstract**</summary>

Intelligent systems possess a crucial characteristic of breaking complicated problems into smaller reusable components or parts and adjusting to new tasks using these part representations. However, current part-learners encounter difficulties in dealing with incidental correlations resulting from the limited observations of objects that may appear only in specific arrangements or with specific backgrounds. These incidental correlations may have a detrimental impact on the generalization and interpretability of learned part representations. This study asserts that part-based representations could be more interpretable and generalize better with limited data, employing two innovative regularization methods. The first regularization separates foreground and background information's generative process via a unique mixture-of-parts formulation. Structural constraints are imposed on the parts using a weakly-supervised loss, guaranteeing that the mixture-of-parts for foreground and background entails soft, object-agnostic masks. The second regularization assumes the form of a distillation loss, ensuring the invariance of the learned parts to the incidental background correlations. Furthermore, we incorporate sparse and orthogonal constraints to facilitate learning high-quality part representations.By reducing the impact of incidental background correlations on the learned parts, we exhibit state-of-the-art (SoTA) performance on few-shot learning tasks on benchmark datasets, including MiniImagenet, TieredImageNet, and FC100. We also demonstrate that the part-based representations acquired through our approach generalize better than existing techniques, even under domain shifts of the background and common data corruption on the ImageNet-9 dataset.</details>

### MixFormerV2: Efficient Fully Transformer Tracking

**Authors:** Yutao Cui, Tianhui Song, Gangshan Wu, Limin Wang

**<details><summary>Abstract**</summary>

Transformer-based trackers have achieved strong accuracy on the standard benchmarks. However, their efficiency remains an obstacle to practical deployment on both GPU and CPU platforms. In this paper, to overcome this issue, we propose a fully transformer tracking framework, coined as \emph{MixFormerV2}, without any dense convolutional operation and complex score prediction module. Our key design is to introduce four special prediction tokens and concatenate them with the tokens from target template and search areas. Then, we apply the unified transformer backbone on these mixed token sequence. These prediction tokens are able to capture the complex correlation between target template and search area via mixed attentions. Based on them, we can easily predict the tracking box and estimate its confidence score through simple MLP heads. To further improve the efficiency of MixFormerV2, we present a new distillation-based model reduction paradigm, including dense-to-sparse distillation and deep-to-shallow distillation. The former one aims to transfer knowledge from the dense-head based MixViT to our fully transformer tracker, while the latter one is used to prune some layers of the backbone. We instantiate two types of MixForemrV2, where the MixFormerV2-B achieves an AUC of 70.6\% on LaSOT and AUC of 56.7\% on TNL2k with a high GPU speed of 165 FPS, and the MixFormerV2-S surpasses FEAR-L by 2.7\% AUC on LaSOT with a real-time CPU speed.</details>

### Model-Free Reinforcement Learning with the Decision-Estimation Coefficient

**Authors:** Dylan J Foster, Noah Golowich, Jian Qian, Alexander Rakhlin, Ayush Sekhari

**<details><summary>Abstract**</summary>

We consider the problem of interactive decision making, encompassing structured bandits and reinforcementlearning with general function approximation. Recently, Foster et al. (2021) introduced theDecision-Estimation Coefficient, a measure of statistical complexity that lower bounds the optimal regret for interactive decisionmaking, as well as a meta-algorithm, Estimation-to-Decisions, which achieves upperbounds in terms of the same quantity. Estimation-to-Decisions is a reduction, which liftsalgorithms for (supervised) online estimation into algorithms fordecision making. In this paper, we show that by combining Estimation-to-Decisions witha specialized form of "optimistic" estimation introduced byZhang (2022), it is possible to obtain guaranteesthat improve upon those of Foster et al. (2021) byaccommodating more lenient notions of estimation error. We use this approach to derive regret bounds formodel-free reinforcement learning with value function approximation, and give structural results showing when it can and cannot help more generally.</details>

### Modeling Dynamics over Meshes with Gauge Equivariant Nonlinear Message Passing

**Authors:** Jung Yeon Park, Lawson Wong, Robin Walters

**<details><summary>Abstract**</summary>

Data over non-Euclidean manifolds, often discretized as surface meshes, naturally arise in computer graphics and biological and physical systems. In particular, solutions to partial differential equations (PDEs) over manifolds depend critically on the underlying geometry. While graph neural networks have been successfully applied to PDEs, they do not incorporate surface geometry and do not consider local gauge symmetries of the manifold. Alternatively, recent works on gauge equivariant convolutional and attentional architectures on meshes leverage the underlying geometry but underperform in modeling surface PDEs with complex nonlinear dynamics. To address these issues, we introduce a new gauge equivariant architecture using nonlinear message passing. Our novel architecture achieves higher performance than either convolutional or attentional networks on domains with highly complex and nonlinear dynamics. However, similar to the non-mesh case, design trade-offs favor convolutional, attentional, or message passing networks for different tasks; we investigate in which circumstances our message passing method provides the most benefit.</details>

### Module-wise Training of Neural Networks via the Minimizing Movement Scheme

**Authors:** Skander Karkar, Ibrahim Ayed, Emmanuel de Bézenac, Patrick Gallinari

**<details><summary>Abstract**</summary>

Greedy layer-wise or module-wise training of neural networks is compelling in constrained and on-device settings where memory is limited, as it circumvents a number of problems of end-to-end back-propagation. However, it suffers from a stagnation problem, whereby early layers overfit and deeper layers stop increasing the test accuracy after a certain depth. We propose to solve this issue by introducing a simple module-wise regularization inspired by the minimizing movement scheme for gradient flows in distribution space. We call the method TRGL for Transport Regularized Greedy Learning and study it theoretically, proving that it leads to greedy modules that are regular and that progressively solve the task. Experimentally, we show improved accuracy of module-wise training of various architectures such as ResNets, Transformers and VGG, when our regularization is added, superior to that of other module-wise training methods and often to end-to-end training, with as much as 60% less memory usage.</details>

### Momentum Provably Improves Error Feedback!

**Authors:** Ilyas Fatkhullin, Alexander Tyurin, Peter Richtarik

**<details><summary>Abstract**</summary>

Due to the high communication overhead when training machine learning models in a distributed environment, modern algorithms invariably rely on lossy communication compression. However, when untreated, the errors caused by compression propagate, and can lead to severely unstable behavior, including exponential divergence. Almost a decade ago, Seide et al. [2014] proposed an error feedback (EF) mechanism, which we refer to as EF14, as an immensely effective heuristic for mitigating this issue. However, despite steady algorithmic and theoretical  advances in the EF field in the last decade, our understanding is far from complete. In this work we address one of the most pressing issues. In particular, in the canonical nonconvex setting, all known variants of EF rely on very large batch sizes to converge, which can be prohibitive in practice. We propose a surprisingly simple fix which removes this issue both theoretically, and in practice: the application of Polyak's momentum to the latest incarnation of EF due to Richtárik et al. [2021] known as EF21. Our algorithm, for which we coin the name EF21-SGDM, improves the communication and sample complexities of previous error feedback algorithms under standard smoothness and bounded variance assumptions, and does not require any further strong assumptions such as bounded gradient dissimilarity. Moreover, we propose a double momentum version of our method that improves the complexities even further. Our proof seems to be novel even when compression is removed form the method, and as such, our proof technique is of independent interest in the study of nonconvex stochastic optimization enriched with Polyak's momentum.</details>

### [Oral] Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture

**Authors:** Dan Fu, Simran Arora, Jessica Grogan, Isys Johnson, Evan Sabri Eyuboglu, Armin Thomas, Benjamin Spector, Michael Poli, Atri Rudra, Christopher Ré

**Oral Presentation:** Tu, Dec 12, 13:40 -- Oral 2A

**<details><summary>Abstract**</summary>

Machine learning models are increasingly being scaled in both sequence length and model dimension to reach longer contexts and better performance. However, existing architectures such as Transformers scale quadratically along both these axes. We ask: are there performant architectures that can scale sub-quadratically along sequence length and model dimension? We introduce Monarch Mixer (M2), a new architecture that uses the same sub-quadratic primitive along both sequence length and model dimension: Monarch matrices, a simple class of expressive structured matrices that captures many linear transforms, achieves high hardware efficiency on GPUs, and scales sub-quadratically. As a proof of concept, we explore the performance of M2 in three domains: non-causal BERT-style language modeling, ViT-style image classification, and causal GPT-style language modeling. For non-causal BERT-style modeling, M2 matches BERT-base and BERT-large in downstream GLUE quality with up to 27% fewer parameters, and achieves up to 9.1$\times$ higher throughput at sequence length 4K. On ImageNet, M2 outperforms ViT-b by 1% in accuracy, with only half the parameters. Causal GPT-style models introduce a technical challenge: enforcing causality via masking introduces a quadratic bottleneck. To alleviate this bottleneck, we develop a novel theoretical view of Monarch matrices based on multivariate polynomial evaluation and interpolation, which lets us parameterize M2 to be causal while remaining sub-quadratic. Using this parameterization, M2 matches GPT-style Transformers at 360M parameters in pretraining perplexity on The PILE—showing for the first time that it may be possible to match Transformer quality without attention or MLPs.</details>

### MosaicBERT: A Bidirectional Encoder Optimized for Fast Pretraining

**Authors:** Jacob Portes, Alexander Trott, Sam Havens, DANIEL KING, Abhinav Venigalla, Moin Nadeem, Nikhil Sardana, Daya Khudia, Jonathan Frankle

**<details><summary>Abstract**</summary>

Although BERT-style encoder models are heavily used in NLP research, many researchers do not pretrain their own BERTs from scratch due to the high cost of training. In the past half-decade since BERT first rose to prominence, many advances have been made with other transformer architectures and training configurations that have yet to be systematically incorporated into BERT. Here, we introduce MosaicBERT, a BERT-style encoder architecture and training recipe that is empirically optimized for fast pretraining. This efficient architecture incorporates FlashAttention, Attention with Linear Biases (ALiBi), Gated Linear Units (GLU), a module to dynamically remove padded tokens, and low precision LayerNorm into the classic transformer encoder block. The training recipe includes a 30% masking ratio for the Masked Language Modeling (MLM) objective, bfloat16 precision, and vocabulary size optimized for GPU throughput, in addition to best-practices from RoBERTa and other encoder models. When pretrained from scratch on the C4 dataset, this base model achieves a downstream average GLUE score of 79.6 in 1.13 hours on 8 A100 80 GB GPUs at a cost of roughly $20. We plot extensive accuracy vs. pretraining speed Pareto curves and show that MosaicBERT base and large are consistently Pareto optimal when compared to a competitive BERT base and large. This empirical speed up in pretraining enables researchers and engineers to pretrain custom BERT-style models at low cost instead of finetune on existing generic models. We open source our model weights and code.</details>

### MuSe-GNN: Learning Unified Gene Representation From Multimodal Biological Graph Data

**Authors:** Tianyu Liu, Yuge Wang, Rex Ying, Hongyu Zhao

**<details><summary>Abstract**</summary>

Discovering genes with similar functions across diverse biomedical contexts poses a significant challenge in gene representation learning due to data heterogeneity. In this study, we resolve this problem by introducing a novel model called Multimodal Similarity Learning Graph Neural Network, which combines Multimodal Machine Learning and Deep Graph Neural Networks to learn gene representations from single-cell sequencing and spatial transcriptomic data. Leveraging 82 training datasets from 10 tissues, three sequencing techniques, and three species, we create informative graph structures for model training and gene representations generation, while incorporating regularization with weighted similarity learning and contrastive learning to learn cross-data gene-gene relationships. This novel design ensures that we can offer gene representations containing functional similarity across different contexts in a joint space. Comprehensive benchmarking analysis shows our model's capacity to effectively capture gene function similarity across multiple modalities, outperforming state-of-the-art methods in gene representation learning by up to $\textbf{100.4}$%. Moreover, we employ bioinformatics tools in conjunction with gene representations to uncover pathway enrichment, regulation causal networks, and functions of disease-associated genes. Therefore, our model efficiently produces unified gene representations for the analysis of gene functions, tissue functions, diseases, and species evolution.</details>

### Multi-Agent Learning with Heterogeneous Linear Contextual Bandits

**Authors:** Anh Do, Thanh Nguyen-Tang, Raman Arora

**<details><summary>Abstract**</summary>

As trained intelligent systems become increasingly pervasive, multiagent learning has emerged as a popular framework for studying complex interactions between autonomous agents. Yet, a formal understanding of how and when learners in heterogeneous environments benefit from sharing their respective experiences is far from complete. In this paper, we seek answers to these questions in the context of linear contextual bandits. We present a novel distributed learning algorithm based on the upper confidence bound (UCB) algorithm, which we refer to as H-LINUCB, wherein agents cooperatively minimize the group regret under the coordination of a central server. In the setting where the level of heterogeneity or dissimilarity across the environments is known to the agents, we show that H-LINUCB is provably optimal in regimes where the tasks are highly similar or highly dissimilar.</details>

### Multi-Modal Inverse Constrained Reinforcement Learning from a Mixture of Demonstrations

**Authors:** Guanren Qiao, Guiliang Liu, Pascal Poupart, Zhiqiang Xu

**<details><summary>Abstract**</summary>

Inverse Constraint Reinforcement Learning (ICRL) aims to recover the underlying constraints respected by expert agents in a data-driven manner. Existing ICRL algorithms typically assume that the demonstration data is generated by a single type of expert. However, in practice, demonstrations often comprise a mixture of trajectories collected from various expert agents respecting different constraints, making it challenging to explain expert behaviors with a unified constraint function. To tackle this issue, we propose a Multi-Modal Inverse Constrained Reinforcement Learning (MMICRL) algorithm for simultaneously estimating multiple constraints corresponding to different types of experts. MMICRL constructs a flow-based density estimator that enables unsupervised expert identification from demonstrations, so as to infer the agent-specific constraints. Following these constraints, MMICRL imitates expert policies with a novel multi-modal constrained policy optimization objective that minimizes the agent-conditioned policy entropy and maximizes the unconditioned one. To enhance robustness, we incorporate this objective into the contrastive learning framework. This approach enables imitation policies to capture the diversity of behaviors among expert agents. Extensive experiments in both discrete and continuous environments show that MMICRL outperforms other baselines in terms of constraint recovery and control performance.</details>

### [Spotlight] Multi-Object Representation Learning via Feature Connectivity and Object-Centric Regularization

**Authors:** Alex Foo, Wynne Hsu, Mong Li Lee

**<details><summary>Abstract**</summary>

Discovering object-centric representations from images has the potential to greatly improve the robustness, sample efficiency and interpretability of machine learning algorithms. Current works on multi-object images typically follow a generative approach that optimizes for input reconstruction and fail to scale to real-world datasets despite significant increases in model capacity. We address this limitation by proposing a novel method that leverages feature connectivity to cluster neighboring pixels likely to belong to the same object. We further design two object-centric regularization terms to refine object representations in the latent space, enabling our approach to scale to complex real-world images. Experimental results on simulated, real-world, complex texture and common object images demonstrate a substantial improvement in the quality of discovered objects compared to state-of-the-art methods, as well as the sample efficiency and generalizability of our approach. We also show that the discovered object-centric representations can accurately predict key object properties in downstream tasks, highlighting the potential of our method to advance the field of multi-object representation learning.</details>

### Multi-body SE(3) Equivariance for Unsupervised Rigid Segmentation and Motion Estimation

**Authors:** Jia-Xing Zhong, Ta-Ying Cheng, Yuhang He, Kai Lu, Kaichen Zhou, Andrew Markham, Niki Trigoni

**<details><summary>Abstract**</summary>

A truly generalizable approach to rigid segmentation and motion estimation is fundamental to 3D understanding of articulated objects and moving scenes. In view of the closely intertwined relationship between segmentation and motion estimates, we present an SE(3) equivariant architecture and a training strategy to tackle this task in an unsupervised manner. Our architecture is composed of two interconnected, lightweight heads. These heads predict segmentation masks using point-level invariant features and estimate motion from SE(3) equivariant features, all without the need for category information. Our training strategy is unified and can be implemented online, which jointly optimizes the predicted segmentation and motion by leveraging the interrelationships among scene flow, segmentation mask, and rigid transformations. We conduct experiments on four datasets to demonstrate the superiority of our method. The results show that our method excels in both model performance and computational efficiency, with only 0.25M parameters and 0.92G FLOPs. To the best of our knowledge, this is the first work designed for category-agnostic part-level SE(3) equivariance in dynamic point clouds.</details>

### MultiFusion: Fusing Pre-Trained Models for Multi-Lingual, Multi-Modal Image Generation

**Authors:** Marco Bellagente, Manuel Brack, Hannah Teufel, Felix Friedrich, Björn Deiseroth, Constantin Eichenberg, Andrew Dai, Robert Baldock, Souradeep Nanda, Koen Oostermeijer, Andres Felipe Cruz-Salinas, Patrick Schramowski, Kristian Kersting, Samuel Weinbach

**<details><summary>Abstract**</summary>

The recent popularity of text-to-image diffusion models (DM) can largely be attributed to the intuitive interface they provide to users. The intended generation can be expressed in natural language, with the model producing faithful interpretations of text prompts. However, expressing complex or nuanced ideas in text alone can be difficult. To ease image generation, we propose MultiFusion that allows one to express complex and nuanced concepts with arbitrarily interleaved inputs of multiple modalities and languages. MultiFusion leverages pre-trained models and aligns them for integration into a cohesive system, thereby avoiding the need for extensive training from scratch. Our experimental results demonstrate the efficient transfer of capabilities from individual modules to the downstream model. Specifically, the fusion of all independent components allows the image generation module to utilize multilingual, interleaved multimodal inputs despite being trained solely on monomodal data in a single language.</details>

### Mutual Information Regularized Offline Reinforcement Learning

**Authors:** Xiao Ma, Bingyi Kang, Zhongwen Xu, Min Lin, Shuicheng Yan

**<details><summary>Abstract**</summary>

The major challenge of offline RL is the distribution shift that appears when out-of-distribution actions are queried, which makes the policy improvement direction biased by extrapolation errors. Most existing methods address this problem by penalizing the policy or value for deviating from the behavior policy during policy improvement or evaluation. In this work, we propose a novel MISA framework to approach offline RL from the perspective of Mutual Information between States and Actions in the dataset by directly constraining the policy improvement direction. MISA constructs lower bounds of mutual information parameterized by the policy and Q-values. We show that optimizing this lower bound is equivalent to maximizing the likelihood of a one-step improved policy on the offline dataset. Hence, we constrain the policy improvement direction to lie in the data manifold. The resulting algorithm simultaneously augments the policy evaluation and improvement by adding mutual information regularizations. MISA is a general framework that unifies conservative Q-learning (CQL) and behavior regularization methods (e.g., TD3+BC) as special cases. We introduce 3 different variants of MISA, and empirically demonstrate that tighter mutual information lower bound gives better offline RL performance. In addition, our extensive experiments show MISA significantly outperforms a wide range of baselines on various tasks of the D4RL benchmark, e.g., achieving 742.9 total points on gym-locomotion tasks. Our code is attached and will be released upon publication.</details>

### NAS-X: Neural Adaptive Smoothing via Twisting

**Authors:** Dieterich Lawson, Michael Li, Scott Linderman

**<details><summary>Abstract**</summary>

Sequential latent variable models (SLVMs) are essential tools in statistics and machine learning, with applications ranging from healthcare to neuroscience. As their flexibility increases, analytic inference and model learning can become challenging, necessitating approximate methods. Here we introduce neural adaptive smoothing via twisting (NAS-X), a method that extends reweighted wake-sleep (RWS) to the sequential setting by using smoothing sequential Monte Carlo (SMC) to estimate intractable posterior expectations. Combining RWS and smoothing SMC allows NAS-X to provide low-bias and low-variance gradient estimates, and fit both discrete and continuous latent variable models. We illustrate the theoretical advantages of NAS-X over previous methods and explore these advantages empirically in a variety of tasks, including a challenging application to mechanistic models of neuronal dynamics. These experiments show that NAS-X substantially outperforms previous VI- and RWS-based methods in inference and model learning, achieving lower parameter error and tighter likelihood bounds.</details>

### Navigating Data Heterogeneity in Federated Learning: A Semi-Supervised Approach for Object Detection

**Authors:** Taehyeon Kim, Eric Lin, Junu Lee, Christian Lau, Vaikkunth Mugunthan

**<details><summary>Abstract**</summary>

Federated Learning (FL) has emerged as a potent framework for training models across distributed data sources while maintaining data privacy. Nevertheless, it faces challenges with limited high-quality labels and non-IID client data, particularly in applications like autonomous driving. To address these hurdles, we navigate the uncharted waters of Semi-Supervised Federated Object Detection (SSFOD). We present a pioneering SSFOD framework, designed for scenarios where labeled data reside only at the server while clients possess unlabeled data. Notably, our method represents the inaugural implementation of SSFOD for clients with 0% labeled non-IID data, a stark contrast to previous studies that maintain some subset of labels at each client. We propose FedSTO, a two-stage strategy encompassing Selective Training followed by Orthogonally enhanced full-parameter training, to effectively address data shift (e.g. weather conditions) between server and clients. Our contributions include selectively refining the backbone of the detector to avert overfitting, orthogonality regularization to boost representation divergence, and local EMA-driven pseudo label assignment to yield high-quality pseudo labels. Extensive validation on prominent autonomous driving datasets (BDD100K, Cityscapes, and SODA10M) attests to the efficacy of our approach, demonstrating state-of-the-art results. Remarkably, FedSTO, using just 20-30% of labels, performs nearly as well as fully-supervised centralized training methods.</details>

### NeRF-IBVS: Visual Servo Based on NeRF for Visual Localization and Navigation

**Authors:** Yuanze Wang, Yichao Yan, Dianxi Shi, Wenhan Zhu, Jianqiang Xia, Tan Jeff, Songchang Jin, KE GAO, XIAOBO LI, Xiaokang Yang

**<details><summary>Abstract**</summary>

Visual localization is a fundamental task in computer vision and robotics. Training existing visual localization methods requires a large number of posed images to generalize to novel views, while state-of-the-art methods generally require dense ground truth 3D labels for supervision. However, acquiring a  large number of posed images and dense 3D labels in the real world is challenging and costly. In this paper, we present a novel visual localization method that achieves accurate localization while using only a few posed images compared to other localization methods. To achieve this, we first use a few posed images with coarse pseudo-3D labels provided by NeRF to train a coordinate regression network. Then a coarse pose is estimated from the regression network with PNP. Finally, we use the image-based visual servo (IBVS) with the scene prior provided by NeRF for pose optimization. Furthermore, our method can provide effective navigation prior, which enable navigation based on IBVS without using custom markers and depth sensor. Extensive experiments on 7-Scenes  and 12-Scenes datasets demonstrate that our method outperforms state-of-the-art methods under the same setting, with only 5\% to 25\% training data.  Furthermore, our framework can be naturally extended to the visual navigation task based on IBVS, and its effectiveness is verified in simulation experiments.</details>

### Near-Optimal $k$-Clustering in the Sliding Window Model

**Authors:** David Woodruff, Peilin Zhong, Samson Zhou

**<details><summary>Abstract**</summary>

Clustering is an important technique for identifying structural information in large-scale data analysis, where the underlying dataset may be too large to store. In many applications, recent data can provide more accurate information and thus older data past a certain time is expired. The sliding window model captures these desired properties and thus there has been substantial interest in clustering in the sliding window model. In this paper, we give the first algorithm that achieves near-optimal $(1+\varepsilon)$-approximation to $(k,z)$-clustering in the sliding window model. Our algorithm uses $\frac{k}{\min(\varepsilon^4,\varepsilon^{2+z})}\,\text{polylog}\frac{n\Delta}{\varepsilon}$ words of space when the points are from $[\Delta]^d$, thus significantly improving on works by Braverman et. al. (SODA 2016), Borassi et. al. (NeurIPS 2021), and Epasto et. al. (SODA 2022).Along the way, we develop a data structure for clustering called an online coreset, which outputs a coreset not only for the end of a stream, but also for all prefixes of the stream. Our online coreset samples $\frac{k}{\min(\varepsilon^4,\varepsilon^{2+z})}\,\text{polylog}\frac{n\Delta}{\varepsilon}$ points from the stream. We then show that any online coreset requires $\Omega\left(\frac{k}{\varepsilon^2}\log n\right)$ samples, which shows a separation between the problem of constructing an offline coreset, i.e., constructing online coresets is strictly harder. Our results also extend to general metrics on $[\Delta]^d$ and are near-optimal in light of a $\Omega\left(\frac{k}{\varepsilon^{2+z}}\right)$ lower bound for the size of an offline coreset.</details>

### Nearest Neighbour with Bandit Feedback

**Authors:** Stephen Pasteris, Chris Hicks, Vasilios Mavroudis

**<details><summary>Abstract**</summary>

In this paper we adapt the nearest neighbour rule to the contextual bandit problem. Our algorithm handles the fully adversarial setting in which no assumptions at all are made about the data-generation process. When combined with a sufficiently fast data-structure for (perhaps approximate) adaptive nearest neighbour search, such as a navigating net, our algorithm is extremely efficient - having a per trial running time polylogarithmic in both the number of trials and actions, and taking only quasi-linear space. We give generic regret bounds for our algorithm and further analyse them when applied to the stochastic bandit problem in euclidean space. A side result of this paper is that, when applied to the online classification problem with stochastic labels, our algorithm can, under certain conditions, have sublinear regret whilst only finding a single nearest neighbour per trial - in stark contrast to the k-nearest neighbours algorithm.</details>

### [Oral] Nearly Tight Bounds For Differentially Private Multiway Cut

**Authors:** Mina Dalirrooyfard, Slobodan Mitrovic, Yuriy Nevmyvaka

**Oral Presentation:** Tu, Dec 12, 13:40 -- Oral 2D

**<details><summary>Abstract**</summary>

Finding min $s$-$t$ cuts in graphs is a basic algorithmic tool, with applications in image segmentation, community detection, reinforcement learning, and data clustering. In this problem, we are given two nodes as terminals and the goal is to remove the smallest number of edges from the graph so that these two terminals are disconnected. We study the complexity of differential privacy for the min $s$-$t$ cut problem and show nearly tight lower and upper bounds where we achieve privacy at no cost for running time efficiency. We also develop a differentially private algorithm for the multiway $k$-cut problem, in which we are given $k$ nodes as terminals that we would like to disconnect.    As a function of $k$, we obtain privacy guarantees that are exponentially more efficient than applying the advanced composition theorem to known algorithms for multiway $k$-cut.    Finally, we empirically evaluate the approximation of our differentially private min $s$-$t$ cut algorithm and show that it almost matches the quality of the output of non-private ones.</details>

### Neural Circuits for Fast Poisson Compressed Sensing in the Olfactory Bulb

**Authors:** Jacob Zavatone-Veth, Paul Masset, William Tong, Joseph D. Zak, Venkatesh Murthy, Cengiz Pehlevan

**<details><summary>Abstract**</summary>

Within a single sniff, the mammalian olfactory system can decode the identity and concentration of odorants wafted on turbulent plumes of air. Yet, it must do so given access only to the noisy, dimensionally-reduced representation of the odor world provided by olfactory receptor neurons. As a result, the olfactory system must solve a compressed sensing problem, relying on the fact that only a handful of the millions of possible odorants are present in a given scene. Inspired by this principle, past works have proposed normative compressed sensing models for olfactory decoding. However, these models have not captured the unique anatomy and physiology of the olfactory bulb, nor have they shown that sensing can be achieved within the 100-millisecond timescale of a single sniff. Here, we propose a rate-based Poisson compressed sensing circuit model for the olfactory bulb. This model maps onto the neuron classes of the olfactory bulb, and recapitulates salient features of their connectivity and physiology. For circuit sizes comparable to the human olfactory bulb, we show that this model can accurately detect tens of odors within the timescale of a single sniff. We also show that this model can perform Bayesian posterior sampling for accurate uncertainty estimation. Fast inference is possible only if the geometry of the neural code is chosen to match receptor properties, yielding a distributed neural code that is not axis-aligned to individual odor identities. Our results illustrate how normative modeling can help us map function onto specific neural circuits to generate new hypotheses.</details>

### Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity

**Authors:** Joel Ye, Jennifer Collinger, Leila Wehbe, Robert Gaunt

**<details><summary>Abstract**</summary>

The neural population spiking activity recorded by intracortical brain-computer interfaces (iBCIs) contain rich structure. Current models of such spiking activity are largely prepared for individual experimental contexts, restricting data volume to that collectable within a single session and limiting the effectiveness of deep neural networks (DNNs). The purported challenge in aggregating neural spiking data is the pervasiveness of context-dependent shifts in the neural data distributions. However, large scale unsupervised pretraining by nature spans heterogeneous data, and has proven to be a fundamental recipe for successful representation learning across deep learning. We thus develop Neural Data Transformer 2 (NDT2), a spatiotemporal Transformer for neural spiking activity, and demonstrate that pretraining can leverage motor BCI datasets that span sessions, subjects, and experimental tasks. NDT2 enables rapid adaptation to novel contexts in downstream decoding tasks and opens the path to deployment of pretrained DNNs for iBCI control. Code: https://github.com/joel99/context_general_bci</details>

### Neural Frailty Machine: Beyond proportional hazard assumption in neural survival regressions

**Authors:** Ruofan Wu, Jiawei Qiao, Mingzhe Wu, Wen Yu, Ming Zheng, Tengfei LIU, Tianyi Zhang, Weiqiang Wang

**<details><summary>Abstract**</summary>

We present neural frailty machine (NFM), a powerful and flexible neural modeling framework for survival regressions. The NFM framework utilizes the classical idea of multiplicative frailty in survival analysis as a principled way of extending the proportional hazard assumption, at the same time being able to leverage the strong approximation power of neural architectures for handling nonlinear covariate dependence. Two concrete models are derived under the framework that extends neural proportional hazard models and nonparametric hazard regression models. Both models allow efficient training under the likelihood objective. Theoretically, for both proposed models, we establish statistical guarantees of neural function approximation with respect to nonparametric components via characterizing their rate of convergence. Empirically, we provide synthetic experiments that verify our theoretical statements. We also conduct experimental evaluations over $6$ benchmark datasets of different scales, showing that the proposed NFM models achieve predictive performance comparable to or sometimes surpassing state-of-the-art survival models. Our code is publicly availabel at https://github.com/Rorschach1989/nfm</details>

### Neural Lyapunov Control for Discrete-Time Systems

**Authors:** Junlin Wu, Andrew Clark, Yiannis Kantaros, Yevgeniy Vorobeychik

**<details><summary>Abstract**</summary>

While ensuring stability for linear systems is well understood, it remains a major challenge for nonlinear systems. A general approach in such cases is to compute a combination of a Lyapunov function and an associated control policy. However, finding Lyapunov functions for general nonlinear systems is a challenging task. To address this challenge, several methods have been proposed that represent Lyapunov functions using neural networks. However, such approaches either focus on continuous-time systems, or highly restricted classes of nonlinear dynamics. We propose the first approach for learning neural Lyapunov control in a broad class of discrete-time systems. Three key ingredients enable us to effectively learn provably stable control policies. The first is a novel mixed-integer linear programming approach for verifying the discrete-time Lyapunov stability conditions, leveraging the particular structure of these conditions. The second is a novel approach for computing verified sublevel sets. The third is a heuristic gradient-based method for quickly finding counterexamples to significantly speed up Lyapunov function learning. Our experiments on four standard benchmarks demonstrate that our approach significantly outperforms state-of-the-art baselines. For example, on the path tracking benchmark, we outperform recent neural Lyapunov control baselines by an order of magnitude in both running time and the size of the region of attraction, and on two of the four benchmarks (cartpole and PVTOL), ours is the first automated approach to return a provably stable controller. Our code is available at: https://github.com/jlwu002/nlc_discrete.</details>

### Neural Priming for Sample-Efficient Adaptation

**Authors:** Matthew Wallingford, Vivek Ramanujan, Alex Fang, Aditya Kusupati, Roozbeh Mottaghi, Aniruddha Kembhavi, Ludwig Schmidt, Ali Farhadi

**<details><summary>Abstract**</summary>

We propose Neural Priming, a technique for adapting large pretrained models to distribution shifts and downstream tasks given few or no labeled examples. Presented with class names or unlabeled test samples, Neural Priming enables the model to recall and conditions its parameters on relevant data seen throughout pretraining, thereby priming it for the test distribution. Neural Priming can be performed at test time in even for pretraining datasets as large as LAION-2B. Performing lightweight updates on the recalled data significantly improves accuracy across a variety of distribution shift and transfer learning benchmarks. Concretely, in the zero-shot setting, we see a 2.45% improvement in accuracy on ImageNet and 3.81% accuracy improvement on average across standard transfer learning benchmarks. Further, using our test time inference scheme, we see a 1.41% accuracy improvement on ImageNetV2. These results demonstrate the effectiveness of Neural Priming in addressing the common challenge of limited labeled data and changing distributions. Code and models are open-sourced at [https://www.github.com/RAIVNLab/neural-priming](https://www.github.com/RAIVNLab/neural-priming).</details>

### Neural Sculpting: Uncovering hierarchically modular task structure in neural networks through pruning and network analysis

**Authors:** Shreyas Malakarjun Patil, Loizos Michael, Constantine Dovrolis

**<details><summary>Abstract**</summary>

Natural target functions and tasks typically exhibit hierarchical modularity -- they can be broken down into simpler sub-functions that are organized in a hierarchy. Such sub-functions have two important features: they have a distinct set of inputs (input-separability) and they are reused as inputs higher in the hierarchy (reusability). Previous studies have established that hierarchically modular neural networks, which are inherently sparse, offer benefits such as learning efficiency, generalization, multi-task learning, and transfer. However, identifying the underlying sub-functions and their hierarchical structure for a given task can be challenging. The high-level question in this work is: if we learn a task using a sufficiently deep neural network, how can we uncover the underlying hierarchy of sub-functions in that task? As a starting point, we examine the domain of Boolean functions, where it is easier to determine whether a task is hierarchically modular. We propose an approach based on iterative unit and edge pruning (during training), combined with network analysis for module detection and hierarchy inference. Finally, we demonstrate that this method can uncover the hierarchical modularity of a wide range of Boolean functions and two vision tasks based on the MNIST digits dataset.</details>

### Neural approximation of Wasserstein distance via a universal architecture for symmetric and factorwise group invariant functions

**Authors:** Samantha Chen, Yusu Wang

**<details><summary>Abstract**</summary>

Learning distance functions between complex objects, such as the Wasserstein distance to compare point sets, is a common goal in machine learning applications. However, functions on such complex objects (e.g., point sets and graphs) are often required to be invariant to a wide variety of group actions e.g. permutation or rigid transformation. Therefore, continuous and symmetric *product* functions (such as distance functions) on such complex objects must also be invariant to the *product* of such group actions. We call these functions symmetric and factor-wise group invariant functions (or SGFI functions} in short).In this paper, we first present a general neural network architecture for approximating SFGI functions. The main contribution of this paper combines this general NN with a sketching idea in order to develop a specific and efficient neural network which can approximate the $p$-th Wasserstein distance between point sets.Very importantly, the required model complexity is *independent* of the sizes of input point sets. On the theoretical front, to the best of our knowledge, this is the first result showing that there exists a neural network with the capacity to approximate Wasserstein distance with bounded model complexity. Our work provides an interesting integration of sketching ideas for geometric problems with universal approximation of symmetric functions. On the empirical front, we present a range of results showing that our newly proposed neural network architecture performs comparatively or better than other models (including a SOTA Siamese Autoencoder based approach). In particular, our NN generalizes significantly better and trains much faster than the SOTA Siamese AE.Finally, this line of investigation could be useful in exploring effective neural network design for solving a broad range of geometric optimization problems (e.g., $k$-means in a metric space).</details>

### Neuro-symbolic Learning Yielding Logical Constraints

**Authors:** Zenan Li, Yunpeng Huang, Zhaoyu Li, Yuan Yao, Jingwei Xu, Taolue Chen, Xiaoxing Ma, Jian Lu

**<details><summary>Abstract**</summary>

Neuro-symbolic systems combine the abilities of neural perception and logical reasoning. However, end-to-end learning of neuro-symbolic systems is still an unsolved challenge. This paper proposes a natural framework that fuses neural network training, symbol grounding, and logical constraint synthesis into a coherent and efficient end-to-end learning process. The capability of this framework comes from the improved interactions between the neural and the symbolic parts of the system in both the training and inference stages. Technically, to bridge the gap between the continuous neural network and the discrete logical constraint, we introduce a difference-of-convex programming technique to relax the logical constraints while maintaining their precision. We also employ cardinality constraints as the language for logical constraint learning and incorporate a trust region method to avoid the degeneracy of logical constraint in learning. Both theoretical analyses and empirical evaluations substantiate the effectiveness of the proposed framework.</details>

### New Bounds for Hyperparameter Tuning of Regression Problems Across Instances

**Authors:** Maria-Florina Balcan, Anh Nguyen, Dravyansh Sharma

**<details><summary>Abstract**</summary>

The task of tuning regularization coefficients in regularized regression models with provable guarantees across problem instances still poses a significant challenge in the literature. This paper investigates the sample complexity of tuning regularization parameters in linear and logistic regressions under $\ell_1$ and $\ell_2$-constraints in the data-driven setting. For the linear regression problem, by more carefully exploiting the structure of the dual function class, we provide a new upper bound for the pseudo-dimension of the validation loss function class, which significantly improves the best-known results on the problem. Remarkably, we also instantiate the first matching lower bound, proving our results are tight. For tuning the regularization parameters of logistic regression, we introduce a new approach to studying the learning guarantee via an approximation of the validation loss function class. We examine the pseudo-dimension of the approximation class and construct a uniform error bound between the validation loss function class and its approximation, which allows us to instantiate the first learning guarantee for the problem of tuning logistic regression regularization coefficients.</details>

### [Spotlight] No Change, No Gain: Empowering Graph Neural Networks with Expected Model Change Maximization for Active Learning

**Authors:** Zixing Song, Yifei Zhang, Irwin King

**<details><summary>Abstract**</summary>

Graph Neural Networks (GNNs) are crucial for machine learning applications with graph-structured data, but their success depends on sufficient labeled data. We present a novel active learning (AL) method for GNNs, extending the Expected Model Change Maximization (EMCM) principle to improve prediction performance on unlabeled data. By presenting a Bayesian interpretation for the node embeddings generated by GNNs under the semi-supervised setting, we efficiently compute the closed-form EMCM acquisition function as the selection criterion for AL without re-training. Our method establishes a direct connection with expected prediction error minimization, offering theoretical guarantees for AL performance. Experiments demonstrate our method's effectiveness compared to existing approaches, in terms of both accuracy and efficiency.</details>

### No Representation Rules Them All in Category Discovery

**Authors:** Sagar Vaze, Andrea Vedaldi, Andrew Zisserman

**<details><summary>Abstract**</summary>

In this paper we tackle the problem of Generalized Category Discovery (GCD). Specifically, given a dataset with labelled and unlabelled images, the task is to cluster all images in the unlabelled subset, whether or not they belong to the labelled categories. Our first contribution is to recognise that most existing GCD benchmarks only contain labels for a single clustering of the data, making it difficult to ascertain whether models are leveraging the available labels to solve the GCD task, or simply solving an unsupervised clustering problem. As such, we present a synthetic dataset, named 'Clevr-4', for category discovery. Clevr-4 contains four equally valid partitions of the data, i.e based on object 'shape', 'texture' or 'color' or 'count'. To solve the task, models are required to extrapolate the taxonomy specified by labelled set, rather than simply latch onto a single natural grouping of the data. We use this dataset to demonstrate the limitations of unsupervised clustering in the GCD setting, showing that even very strong unsupervised models fail on Clevr-4. We further use Clevr-4 to examine the weaknesses of existing GCD algorithms, and propose a new method which addresses these shortcomings, leveraging consistent findings from the representation learning literature to do so. Our simple solution, which is based on `Mean Teachers' and termed $\mu$GCD, substantially outperforms implemented baselines on Clevr-4. Finally, when we transfer these findings to real data on the challenging Semantic Shift Benchmark suite, we find that $\mu$GCD outperforms all prior work, setting a new state-of-the-art.</details>

### No-Regret Learning in Dynamic Competition with Reference Effects Under Logit Demand

**Authors:** Mengzi Amy Guo, Donghao Ying, Javad Lavaei, Zuo-Jun Shen

**<details><summary>Abstract**</summary>

This work is dedicated to the algorithm design in a competitive framework, with the primary goal of learning a stable equilibrium. We consider the dynamic price competition between two firms operating within an opaque marketplace, where each firm lacks information about its competitor. The demand follows the multinomial logit (MNL) choice model, which depends on the consumers' observed price and their reference price, and consecutive periods in the repeated games are connected by reference price updates. We use the notion of stationary Nash equilibrium (SNE), defined as the fixed point of the equilibrium pricing policy for the single-period game, to simultaneously capture the long-run market equilibrium and stability. We propose the online projected gradient ascent algorithm (OPGA), where the firms adjust prices using the first-order derivatives of their log-revenues that can be obtained from the market feedback mechanism. Despite the absence of typical properties required for the convergence of online games, such as strong monotonicity and variational stability, we demonstrate that under diminishing step-sizes, the price and reference price paths generated by OPGA converge to the unique SNE, thereby achieving the no-regret learning and a stable market. Moreover, with appropriate step-sizes, we prove that this convergence exhibits a rate of $\mathcal{O}(1/t)$.</details>

### No-Regret Online Prediction with Strategic Experts

**Authors:** Omid Sadeghi, Maryam Fazel

**<details><summary>Abstract**</summary>

We study a generalization of the online binary prediction with expert advice framework where at each round, the learner is allowed to pick $m\geq 1$ experts from a pool of $K$ experts and the overall utility is a modular or submodular function of the chosen experts. We focus on the setting in which experts act strategically and aim to maximize their influence on the algorithm's predictions by potentially misreporting their beliefs about the events. Among others, this setting finds applications in forecasting competitions where the learner seeks not only to make predictions by aggregating different forecasters but also to rank them according to their relative performance. Our goal is to design algorithms that satisfy the following two requirements: 1) \emph{Incentive-compatible}: Incentivize the experts to report their beliefs truthfully, and 2) \emph{No-regret}: Achieve sublinear regret with respect to the true beliefs of the best fixed set of $m$ experts in hindsight. Prior works have studied this framework when $m=1$ and provided incentive-compatible no-regret algorithms for the problem. We first show that a simple reduction of our problem to the $m=1$ setting is neither efficient nor effective. Then, we provide algorithms that utilize the specific structure of the utility functions to achieve the two desired goals.</details>

### No-Regret Online Reinforcement Learning with Adversarial Losses and Transitions

**Authors:** Tiancheng Jin, Junyan Liu, Chloé Rouyer, William Chang, Chen-Yu Wei, Haipeng Luo

**<details><summary>Abstract**</summary>

Existing online learning algorithms for adversarial Markov Decision Processes achieve $\mathcal{O}(\sqrt{T})$ regret after $T$ rounds of interactions even if the loss functions are chosen arbitrarily by an adversary, with the caveat that the transition function has to be fixed.This is because it has been shown that adversarial transition functions make no-regret learning impossible.Despite such impossibility results, in this work, we develop algorithms that can handle both adversarial losses and adversarial transitions, with regret increasing smoothly in the degree of maliciousness of the adversary.More concretely, we first propose an algorithm that enjoys $\widetilde{\mathcal{O}}(\sqrt{T} + C^{P})$ regret where $C^{P}$ measures how adversarial the transition functions are and can be at most $\mathcal{O}(T)$.While this algorithm itself requires knowledge of $C^{P}$, we further develop a black-box reduction approach that removes this requirement.Moreover, we also show that further refinements of the algorithm not only maintains the same regret bound, but also simultaneously adapts to easier environments (where losses are generated in a certain stochastically constrained manner as in [Jin et al. 2021]) and achieves $\widetilde{\mathcal{O}}(U + \sqrt{UC^{L}}  + C^{P})$ regret, where $U$ is some standard gap-dependent coefficient and $C^{L}$ is the amount of corruption on losses.</details>

### Noether Embedding: Efficient Learning of Temporal Regularities

**Authors:** Chi Gao, Zidong Zhou, Luping Shi

**<details><summary>Abstract**</summary>

Learning to detect and encode temporal regularities (TRs) in events is a prerequisite for human-like intelligence. These regularities should be formed from limited event samples and stored as easily retrievable representations. Existing event embeddings, however, cannot effectively decode TR validity with well-trained vectors, let alone satisfy the efficiency requirements. We develop Noether Embedding (NE) as the first efficient TR learner with event embeddings. Specifically, NE possesses the intrinsic time-translation symmetries of TRs indicated as conserved local energies in the embedding space. This structural bias reduces the calculation of each TR validity to embedding each event sample, enabling NE to achieve data-efficient TR formation insensitive to sample size and time-efficient TR retrieval in constant time complexity. To comprehensively evaluate the TR learning capability of embedding models, we define complementary tasks of TR detection and TR query, formulate their evaluation metrics, and assess embeddings on classic ICEWS14, ICEWS18, and GDELT datasets. Our experiments demonstrate that NE consistently achieves about double the F1 scores for detecting valid TRs compared to classic embeddings, and it provides over ten times higher confidence scores for querying TR intervals. Additionally, we showcase NE's potential applications in social event prediction, personal decision-making, and memory-constrained scenarios.</details>

### Noise-Adaptive Thompson Sampling for Linear Contextual Bandits

**Authors:** Ruitu Xu, Yifei Min, Tianhao Wang

**<details><summary>Abstract**</summary>

Linear contextual bandits represent a fundamental class of models with numerous real-world applications, and it is critical to develop algorithms that can effectively manage noise with unknown variance, ensuring provable guarantees for both worst-case constant-variance noise and deterministic reward scenarios. In this paper, we study linear contextual bandits with heteroscedastic noise and propose the first noise-adaptive Thompson sampling-style algorithm that achieves a variance-dependent regret upper bound of $\widetilde O\Big(d^{3/2} + d^{3/2} \sqrt{\sum_{t=1}^T \sigma_t^2}\Big)$, where $d$ is the dimension of the context vectors and $\sigma_t^2$ is the variance of the reward in round $t$. This recovers the existing $\widetilde O(d^{3/2}\sqrt{T})$ regret guarantee in the constant-variance regime and further improves to $\widetilde O(d^{3/2})$ in the deterministic regime, thus achieving a smooth interpolation in between. Our approach utilizes a stratified sampling procedure to overcome the too-conservative optimism in the linear Thompson sampling algorithm for linear contextual bandits.</details>

### [Spotlight] Non-Asymptotic Analysis of a UCB-based Top Two Algorithm

**Authors:** Marc Jourdan, Rémy Degenne

**<details><summary>Abstract**</summary>

A Top Two sampling rule for bandit identification is a method which selects the next arm to sample from among two candidate arms, a *leader* and a *challenger*. Due to their simplicity and good empirical performance, they have received increased attention in recent years. However, for fixed-confidence best arm identification, theoretical guarantees for Top Two methods have only been obtained in the asymptotic regime, when the error level vanishes. In this paper, we derive the first non-asymptotic upper bound on the expected sample complexity of a Top Two algorithm, which holds for any error level. Our analysis highlights sufficient properties for a regret minimization algorithm to be used as leader. These properties are satisfied by the UCB algorithm, and our proposed UCB-based Top Two algorithm simultaneously enjoys non-asymptotic guarantees and competitive empirical performance.</details>

### [Spotlight] OKRidge: Scalable Optimal k-Sparse Ridge Regression

**Authors:** Jiachang Liu, Sam Rosen, Chudi Zhong, Cynthia Rudin

**<details><summary>Abstract**</summary>

We consider an important problem in scientific discovery, namely identifying sparse governing equations for nonlinear dynamical systems. This involves solving sparse ridge regression problems to provable optimality in order to determine which terms drive the underlying dynamics. We propose a fast algorithm, OKRidge, for sparse ridge regression, using a novel lower bound calculation involving, first, a saddle point formulation, and from there, either solving (i) a linear system or (ii) using an ADMM-based approach, where the proximal operators can be efficiently evaluated by solving another linear system and an isotonic regression problem. We also propose a method to warm-start our solver, which leverages a beam search. Experimentally, our methods attain provable optimality with run times that are orders of magnitude faster than those of the existing MIP formulations solved by the commercial solver Gurobi.</details>

### Object-centric Learning with Cyclic Walks between Parts and Whole

**Authors:** Ziyu Wang, Mike Zheng Shou, Mengmi Zhang

**<details><summary>Abstract**</summary>

Learning object-centric representations from complex natural environments enables both humans and machines with reasoning abilities from low-level perceptual features. To capture compositional entities of the scene, we proposed cyclic walks between perceptual features extracted from vision transformers and object entities. First, a slot-attention module interfaces with these perceptual features and produces a finite set of slot representations. These slots can bind to any object entities in the scene via inter-slot competitions for attention. Next, we establish entity-feature correspondence with cyclic walks along high transition probability based on the pairwise similarity between perceptual features (aka "parts") and slot-binded object representations (aka "whole"). The whole is greater than its parts and the parts constitute the whole. The part-whole interactions form cycle consistencies, as supervisory signals, to train the slot-attention module. Our rigorous experiments on 	extit{seven} image datasets in \textit{three} \textit{unsupervised} tasks demonstrate that the networks trained with our cyclic walks can disentangle foregrounds and backgrounds, discover objects, and segment semantic objects in complex scenes. In contrast to object-centric models attached with a decoder for the pixel-level or feature-level reconstructions, our cyclic walks provide strong learning signals, avoiding computation overheads and enhancing memory efficiency. Our source code and data are available at: \href{https://github.com/ZhangLab-DeepNeuroCogLab/Parts-Whole-Object-Centric-Learning/}{link}.</details>

### On Computing Pairwise Statistics with Local Differential Privacy

**Authors:** Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Adam Sealfon

**<details><summary>Abstract**</summary>

We study the problem of computing pairwise statistics, i.e., ones of the form $\binom{n}{2}^{-1} \sum_{i \ne j} f(x_i, x_j)$, where $x_i$ denotes the input to the $i$th user, with differential privacy (DP) in the local model. This formulation captures important metrics such as Kendall's $\tau$ coefficient, Area Under Curve, Gini's mean difference, Gini's entropy, etc. We give several novel and generic algorithms for the problem, leveraging techniques from DP algorithms for linear queries.</details>

### On Convergence of Polynomial Approximations to the Gaussian Mixture Entropy

**Authors:** Caleb Dahlke, Jason Pacheco

**<details><summary>Abstract**</summary>

Gaussian mixture models (GMMs) are fundamental to machine learning due to their flexibility as approximating densities. However, uncertainty quantification of GMMs remains a challenge as differential entropy lacks a closed form.  This paper explores polynomial approximations, specifically Taylor and Legendre, to the GMM entropy from a theoretical and practical perspective. We provide new analysis of a widely used approach due to Huber et al.(2008) and show that the series diverges under simple conditions.  Motivated by this divergence we provide a novel Taylor series that is provably convergent to the true entropy of any GMM.  We demonstrate a method for selecting a center such that the series converges from below, providing a lower bound on GMM entropy. Furthermore, we demonstrate that orthogonal polynomial series result in more accurate polynomial approximations. Experimental validation supports our theoretical results while showing that our method is comparable in computation to Huber et al. We also show that in application, the use of these polynomial approximations, such as in Nonparametric Variational Inference by Gershamn et al. (2012), rely on the convergence of the methods in computing accurate approximations. This work contributes useful analysis to existing methods while introducing a novel approximation supported by firm theoretical guarantees.</details>

### On Proper Learnability between Average- and Worst-case Robustness

**Authors:** Vinod Raman, UNIQUE SUBEDI, Ambuj Tewari

**<details><summary>Abstract**</summary>

Recently, Montasser at al. (2019) showed that finite VC dimension is not sufficient for proper adversarially robust PAC learning. In light of this hardness, there is a growing effort to study what type of relaxations to the adversarially robust PAC learning setup can enable proper learnability. In this work, we initiate the study of proper learning under relaxations of the worst-case robust loss. We give a family of robust loss relaxations under which VC classes are properly PAC learnable with sample complexity close to what one would require in the standard PAC learning setup. On the other hand, we show that for an existing and natural relaxation of the worst-case robust loss, finite VC dimension is not sufficient for proper learning. Lastly, we give new generalization guarantees for the adversarially robust empirical risk minimizer.</details>

### On student-teacher deviations in distillation: does it pay to disobey?

**Authors:** Vaishnavh Nagarajan, Aditya Menon, Srinadh Bhojanapalli, Hossein Mobahi, Sanjiv Kumar

**<details><summary>Abstract**</summary>

Knowledge distillation (KD) has been widely used to improve the test accuracy of a "student" network, by training it to mimic the soft probabilities of a trained "teacher" network. Yet, it has been shown in recent work that, despite being trained to fit the teacher's probabilities, the student may not only significantly deviate from the teacher probabilities, but may also outdo than the teacher in performance. Our work aims to reconcile this seemingly paradoxical observation. Specifically, we characterize the precise nature of the student-teacher deviations, and argue how they _can_ co-occur with better generalization.  First, through experiments on  image and language data, we identify that these probability deviations correspond to the student systematically _exaggerating_ the confidence levels of the teacher.Next, we theoretically and empirically establish another form of exaggeration in some simple settings: KD exaggerates the implicit bias of gradient descent in converging faster along the top eigendirections of the data. Finally, we tie these two observations together: we demonstrate that the exaggerated bias of KD can simultaneously result in both (a) the exaggeration of confidence and (b) the improved generalization of the student, thus offering a resolution to the apparent paradox.  Our analysis brings existing theory and practice closer by considering the role of gradient descent in KD and by demonstrating the exaggerated bias effect in both theoretical and empirical settings.</details>

### [Spotlight] On the Connection between Pre-training Data Diversity and Fine-tuning Robustness

**Authors:** Vivek Ramanujan, Thao Nguyen, Sewoong Oh, Ali Farhadi, Ludwig Schmidt

**<details><summary>Abstract**</summary>

Pre-training has been widely adopted in deep learning to improve model performance, especially when the training data for a target task is limited. In our work, we seek to understand the implications of this training strategy on the generalization properties of downstream models. More specifically, we ask the following question: how do properties of the pre-training distribution affect the robustness of a fine-tuned model? The properties we explore include the label space, label semantics, image diversity, data domains, and data quantity of the pre-training distribution. We find that the primary factor influencing downstream effective robustness (Taori et al., 2020) is data quantity, while other factors have limited significance. For example, reducing the number of ImageNet pre-training classes by 4x while increasing the number of images per class by 4x (that is, keeping total data quantity fixed) does not impact the robustness of fine-tuned models. We demonstrate our findings on pre-training distributions drawn from various natural and synthetic data sources, primarily using the iWildCam-WILDS distribution shift as a test for robustness.</details>

### On the Convergence of Encoder-only Shallow Transformers

**Authors:** Yongtao Wu, Fanghui Liu, Grigorios Chrysos, Volkan Cevher

**<details><summary>Abstract**</summary>

In this paper, we aim to build the global convergence theory of encoder-only shallow Transformers under a realistic setting from the perspective of architectures, initialization, and scaling under a finite width regime. The difficulty lies in how to tackle the softmax in self-attention mechanism, the core ingredient of Transformer. In particular, we diagnose the scaling scheme, carefully tackle the input/output of softmax, and prove that quadratic overparameterization is sufficient for global convergence of our shallow Transformers under commonly-used He/LeCun initialization in practice. Besides, neural tangent kernel (NTK) based analysis is also given, which facilitates a comprehensive comparison. Our theory demonstrates the separation on the importance of different scaling schemes and initialization. We believe our results can pave the way for a better understanding of modern Transformers, particularly on training dynamics.</details>

### On the Convergence of No-Regret Learning Dynamics in Time-Varying Games

**Authors:** Ioannis Anagnostides, Ioannis Panageas, Gabriele Farina, Tuomas Sandholm

**<details><summary>Abstract**</summary>

Most of the literature on learning in games has focused on the restrictive setting where the underlying repeated game does not change over time. Much less is known about the convergence of no-regret learning algorithms in dynamic multiagent settings. In this paper, we characterize the convergence of optimistic gradient descent (OGD) in time-varying games. Our framework yields sharp convergence bounds for the equilibrium gap of OGD in zero-sum games parameterized on natural variation measures of the sequence of games, subsuming known results for static games. Furthermore, we establish improved second-order variation bounds under strong convexity-concavity, as long as each game is repeated multiple times. Our results also apply to time-varying general-sum multi-player games via a bilinear formulation of correlated equilibria, which has novel implications for meta-learning and for obtaining refined variation-dependent regret bounds, addressing questions left open in prior papers. Finally, we leverage our framework to also provide new insights on dynamic regret guarantees in static games.</details>

### On the Exploitability of Instruction Tuning

**Authors:** Manli Shu, Jiongxiao Wang, Chen Zhu, Jonas Geiping, Chaowei Xiao, Tom Goldstein

**<details><summary>Abstract**</summary>

Instruction tuning is an effective technique to align large language models (LLMs) with human intent. In this work, we investigate how an adversary can exploit instruction tuning by injecting specific instruction-following examples into the training data that intentionally changes the model's behavior. For example, an adversary can achieve content injection by injecting training examples that mention target content and eliciting such behavior from downstream models. To achieve this goal, we propose \textit{AutoPoison}, an automated data poisoning pipeline. It naturally and coherently incorporates versatile attack goals into poisoned data with the help of an oracle LLM. We showcase two example attacks: content injection and over-refusal attacks, each aiming to induce a specific exploitable behavior. We quantify and benchmark the strength and the stealthiness of our data poisoning scheme. Our results show that AutoPoison allows an adversary to change a model's behavior by poisoning only a small fraction of data while maintaining a high level of stealthiness in the poisoned examples. We hope our work sheds light on how data quality affects the behavior of instruction-tuned models and raises awareness of the importance of data quality for responsible deployments of LLMs.</details>

### On the Importance of Feature Separability in Predicting Out-Of-Distribution Error

**Authors:** RENCHUNZI XIE, Hongxin Wei, Lei Feng, Yuzhou Cao, Bo An

**<details><summary>Abstract**</summary>

Estimating the generalization performance is practically challenging on out-of-distribution (OOD) data without ground-truth labels. While previous methods emphasize the connection between distribution difference and OOD accuracy, we show that a large domain gap not necessarily leads to a low test accuracy. In this paper, we investigate this problem from the perspective of feature separability empirically and theoretically. Specifically, we propose a dataset-level score based upon feature dispersion to estimate the test accuracy under distribution shift. Our method is inspired by desirable properties of features in representation learning: high inter-class dispersion and high intra-class compactness. Our analysis shows that inter-class dispersion is strongly correlated with the model accuracy, while intra-class compactness does not reflect the generalization performance on OOD data. Extensive experiments demonstrate the superiority of our method in both prediction performance and computational efficiency.</details>

### On the Last-iterate Convergence in Time-varying Zero-sum Games: Extra Gradient Succeeds where Optimism Fails

**Authors:** Yi Feng, Hu Fu, Qun Hu, Ping Li, Ioannis Panageas, bo peng, Xiao Wang

**<details><summary>Abstract**</summary>

Last-iterate convergence has received extensive study in two player zero-sum games starting from bilinear, convex-concave up to settings that satisfy the MVI condition. Typical methods that exhibit last-iterate convergence for the aforementioned games include extra-gradient (EG) and optimistic gradient descent ascent (OGDA). However, all the established last-iterate convergence results hold for the restrictive setting where the underlying repeated game does not change over time.Recently, a line of research has focused on regret analysis of OGDA  in time-varying games, i.e., games where payoffs evolve with time; the last-iterate behavior of OGDA and EG in time-varying environments remains unclear though. In this paper, we study the last-iterate behavior of various algorithms in two types of unconstrained, time-varying, bilinear zero-sum games: periodic and convergent perturbed games. These models expand upon the usual repeated game formulation and incorporate external environmental factors, such as the seasonal effects on species competition and vanishing external noise. In periodic games, we prove that EG will converge while OGDA and momentum method will diverge. This is quite surprising, as to the best of our knowledge, it is the first result that indicates EG and OGDA have qualitatively different last-iterate behaviors and do not exhibit similar behavior. In convergent perturbed games, we prove all these algorithms converge as long as the game itself stabilizes with a faster rate than $1/t$.</details>

### On the Powerfulness of Textual Outlier Exposure for Visual OoD Detection

**Authors:** Sangha Park, Jisoo Mok, Dahuin Jung, Saehyung Lee, Sungroh Yoon

**<details><summary>Abstract**</summary>

Successful detection of Out-of-Distribution (OoD) data is becoming increasingly important to ensure safe deployment of neural networks. One of the main challenges in OoD detection is that neural networks output overconfident predictions on OoD data, make it difficult to determine OoD-ness of data solely based on their predictions. Outlier exposure addresses this issue by introducing an additional loss that encourages low-confidence predictions on OoD data during training. While outlier exposure has shown promising potential in improving OoD detection performance, all previous studies on outlier exposure have been limited to utilizing visual outliers. Drawing inspiration from the recent advancements in vision-language pre-training, this paper venture out to the uncharted territory of textual outlier exposure. First, we uncover the benefits of using textual outliers by replacing real or virtual outliers in the image-domain with textual equivalents. Then, we propose various ways of generating preferable textual outliers. Our extensive experiments demonstrate that generated textual outliers achieve competitive performance on large-scale OoD and hard OoD benchmarks. Furthermore, we conduct empirical analyses of textual outliers to provide primary criteria for designing advantageous textual outliers: near-distribution, descriptiveness, and inclusion of visual semantics.</details>

### On the Relationship Between Relevance and Conflict in Online Social Link Recommendations

**Authors:** Yanbang Wang, Jon Kleinberg

**<details><summary>Abstract**</summary>

In an online social network, link recommendations are a way for users to discover relevant links to people they may know, thereby potentially increasing their engagement on the platform. However, the addition of links to a social network can also have an effect on the level of conflict in the network --- expressed in terms of polarization and disagreement. To date, however, we have very little understanding of how these two implications of link formation relate to each other: are the goals of high relevance and conflict reduction aligned, or are the links that users are most likely to accept fundamentally different from the ones with the greatest potential for reducing conflict? Here we provide the first analysis of this question, using the recently popular Friedkin-Johnsen model of opinion dynamics. We first present a surprising result on how link additions shift the level of opinion conflict, followed by explanation work that relates the amount of shift to structural features of the added links. We then characterize the gap in conflict reduction between the set of links achieving the largest reduction and the set of links achieving the highest relevance. The gap is measured on real-world data, based on instantiations of relevance defined by 13 link recommendation algorithms. We find that some, but not all, of the more accurate algorithms actually lead to better reduction of conflict. Our work suggests that social links recommended for increasing user engagement may not be as conflict-provoking as people might have thought.</details>

### [Spotlight] On the Variance, Admissibility, and Stability of Empirical Risk Minimization

**Authors:** Gil Kur, Eli Putterman, Alexander Rakhlin

**<details><summary>Abstract**</summary>

It is well known that Empirical Risk Minimization (ERM) may attain minimax suboptimal rates in terms of the mean squared error (Birgé and Massart, 1993). In this paper, we prove that, under relatively mild assumptions, the suboptimality of ERM must be due to its bias. Namely, the variance error term of ERM (in terms of the bias and variance decomposition) enjoys the minimax rate. In the fixed design setting, we provide an elementary proof of this result using the probabilistic method. Then, we extend our proof to the random design setting for various models. In addition, we provide a simple proof of Chatterjee’s admissibility theorem (Chatterjee, 2014, Theorem 1.4), which states that in the fixed design setting, ERM cannot be ruled out as an optimal method, and then we extend this result to the random design setting. We also show that our estimates imply stability of ERM, complementing the main result of Caponnetto and Rakhlin (2006) for non-Donsker classes. Finally, we highlight the somewhat irregular nature of the loss landscape of ERM in the non-Donsker regime, by showing that functions can be close to ERM, in terms of $L_2$ distance, while still being far from almost-minimizers of the empirical loss.</details>

### On the explainable properties of 1-Lipschitz Neural Networks: An Optimal Transport Perspective

**Authors:** Mathieu Serrurier, Franck Mamalet, Thomas FEL, Louis Béthune, Thibaut Boissin

**<details><summary>Abstract**</summary>

Input gradients have a pivotal role in a variety of applications, including adversarial attack algorithms for evaluating model robustness, explainable AI techniques for generating saliency maps, and counterfactual explanations. However, saliency maps generated by traditional neural networks are often noisy and provide limited insights. In this paper, we demonstrate that, on the contrary, the saliency maps of 1-Lipschitz neural networks, learnt with the dual loss of an optimal transportation problem, exhibit desirable XAI properties:They are highly concentrated on the essential parts of the image with low noise, significantly outperforming state-of-the-art explanation approaches across various models and metrics. We also prove that these maps align unprecedentedly well with human explanations on ImageNet. To explain the particularly beneficial properties of the saliency map for such models, we prove this gradient encodes  both the direction of the transportation plan and the direction towards the nearest adversarial attack. Following the gradient down to the decision boundary is no longer considered an adversarial attack, but rather a counterfactual explanation that explicitly transports the input from one class to another.  Thus, Learning with such a loss jointly optimizes the classification objective and the alignment of the gradient , i.e. the saliency map, to the transportation plan direction. These networks were previously known to be certifiably robust by design, and we demonstrate that they scale well for large problems and models, and are tailored for explainability using a fast and straightforward method.</details>

### One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation

**Authors:** Zhiwei Hao, Jianyuan Guo, Kai Han, Yehui Tang, Han Hu, Yunhe Wang, Chang Xu

**<details><summary>Abstract**</summary>

Knowledge distillation (KD) has proven to be a highly effective approach for enhancing model performance through a teacher-student training scheme. However, most existing distillation methods are designed under the assumption that the teacher and student models belong to the same model family, particularly the hint-based approaches. By using centered kernel alignment (CKA) to compare the learned features between heterogeneous teacher and student models, we observe significant feature divergence. This divergence illustrates the ineffectiveness of previous hint-based methods in cross-architecture distillation. To tackle the challenge in distilling heterogeneous models, we propose a simple yet effective one-for-all KD framework called OFA-KD, which significantly improves the distillation performance between heterogeneous architectures. Specifically, we project intermediate features into an aligned latent space such as the logits space, where architecture-specific information is discarded. Additionally, we introduce an adaptive target enhancement scheme to prevent the student from being disturbed by irrelevant information. Extensive experiments with various architectures, including CNN, Transformer, and MLP, demonstrate the superiority of our OFA-KD framework in enabling distillation between heterogeneous architectures. Specifically, when equipped with our OFA-KD, the student models achieve notable performance improvements, with a maximum gain of 8.0% on the CIFAR-100 dataset and 0.7% on the ImageNet-1K dataset. PyTorch code and checkpoints can be found at https://github.com/Hao840/OFAKD.</details>

### [Spotlight] One-step differentiation of iterative algorithms

**Authors:** Jerome Bolte, Edouard Pauwels, Samuel Vaiter

**<details><summary>Abstract**</summary>

In appropriate frameworks, automatic differentiation is transparent to the user, at the cost of being a significant computational burden when the number of operations is large. For iterative algorithms, implicit differentiation alleviates this issue but requires custom implementation of Jacobian evaluation. In this paper, we study one-step differentiation, also known as Jacobian-free backpropagation, a method as easy as automatic differentiation and as performant as implicit differentiation for fast algorithms (e.g. superlinear optimization methods). We provide a complete theoretical approximation analysis with specific examples (Newton's method, gradient descent) along with its consequences in bilevel optimization. Several numerical examples illustrate the well-foundness of the one-step estimator.</details>

### Online Ad Procurement in Non-stationary Autobidding Worlds

**Authors:** Jason Cheuk Nam Liang, Haihao Lu, Baoyu Zhou

**<details><summary>Abstract**</summary>

Today's online advertisers procure digital ad impressions through interacting with autobidding platforms: advertisers convey high level procurement goals via setting levers such as budget, target return-on-investment, max cost per click, etc.. Then ads platforms subsequently procure impressions on advertisers' behalf, and report final procurement conversions (e.g. click) to advertisers. In practice, advertisers may receive minimal information on platforms' procurement details, and procurement outcomes  are subject to non-stationary factors like seasonal patterns, occasional system corruptions, and market trends which make it difficult for advertisers to optimize lever decisions effectively. Motivated by this,  we present an online learning framework that helps advertisers dynamically optimize ad platform lever decisions while subject to general long-term constraints in a realistic bandit feedback environment with non-stationary procurement outcomes. In particular, we introduce a primal-dual algorithm for online decision making with multi-dimension decision variables, bandit feedback and long-term uncertain constraints. We show that our algorithm achieves low regret in many worlds when procurement outcomes are generated through procedures that are stochastic, adversarial, adversarially corrupted, periodic, and ergodic, respectively, without having to know which procedure is the ground truth. Finally, we emphasize that our proposed algorithm and theoretical results extend beyond the applications of online advertising.</details>

### Online Clustering of Bandits with Misspecified User Models

**Authors:** Zhiyong Wang, Jize Xie, Xutong Liu, Shuai Li, John C.S. Lui

**<details><summary>Abstract**</summary>

The contextual linear bandit is an important online learning problem where given arm features, a learning agent selects an arm at each round to maximize the cumulative rewards in the long run. A line of works, called the clustering of bandits (CB), utilize the collaborative effect over user preferences and have shown significant improvements over classic linear bandit algorithms. However, existing CB algorithms require well-specified linear user models and can fail when this critical assumption does not hold. Whether robust CB algorithms can be designed for more practical scenarios with misspecified user models remains an open problem. In this paper, we are the first to present the important problem of clustering of bandits with misspecified user models (CBMUM), where the expected rewards in user models can be perturbed away from perfect linear models. We devise two robust CB algorithms, RCLUMB and RSCLUMB (representing the learned clustering structure with dynamic graph and sets, respectively), that can accommodate the inaccurate user preference estimations and erroneous clustering caused by model misspecifications. We prove regret upper bounds of $O(\epsilon_*T\sqrt{md\log T}  + d\sqrt{mT}\log T)$ for our algorithms under milder assumptions than previous CB works, which match the lower bound asymptotically in $T$ up to logarithmic factors, and also match the state-of-the-art results in several degenerate cases. Our regret analysis is novel and different from the typical proof flow of previous CB works. The techniques in proving the regret caused by misclustering users are quite general and may be of independent interest. Experiments on both synthetic and real-world data show our outperformance over previous algorithms.</details>

### Online POMDP Planning with Anytime Deterministic Guarantees

**Authors:** Moran Barenboim, Vadim Indelman

**<details><summary>Abstract**</summary>

Autonomous agents operating in real-world scenarios frequently encounter uncertainty and make decisions based on incomplete information. Planning under uncertainty can be mathematically formalized using partially observable Markov decision processes (POMDPs). However, finding an optimal plan for POMDPs can be computationally expensive and is feasible only for small tasks. In recent years, approximate algorithms, such as tree search and sample-based methodologies, have emerged as state-of-the-art POMDP solvers for larger problems. Despite their effectiveness, these algorithms offer only probabilistic and often asymptotic guarantees toward the optimal solution due to their dependence on sampling. To address these limitations, we derive a deterministic relationship between a simplified solution that iseasier to obtain and the theoretically optimal one. First, we derive bounds for selecting a subset of the observations to branch from while computing a complete belief at each posterior node. Then, since a complete belief update may be computationally demanding, we extend the bounds to support reduction of both the state and the observation spaces. We demonstrate how our guarantees can be integrated with existing state-of-the-art solvers that sample a subset of states and observations. As a result, the returned solution holds deterministic bounds relative to the optimal policy. Lastly, we substantiate our findings with supporting experimental results.</details>

### Online Pricing for Multi-User Multi-Item Markets

**Authors:** Yigit Efe Erginbas, Thomas Courtade, Kannan Ramchandran, Soham Phade

**<details><summary>Abstract**</summary>

Online pricing has been the focus of extensive research in recent years, particularly in the context of selling an item to sequentially arriving users. However, what if a provider wants to maximize revenue by selling multiple items to multiple users in each round? This presents a complex problem, as the provider must intelligently offer the items to those users who value them the most without exceeding their highest acceptable prices. In this study, we tackle this challenge by designing online algorithms that can efficiently offer and price items while learning user valuations from accept/reject feedback. We focus on three user valuation models (fixed valuations, random experiences, and random valuations) and provide algorithms with nearly-optimal revenue regret guarantees. In particular, for any market setting with $N$ users, $M$ items, and load $L$ (which roughly corresponds to the maximum number of simultaneous allocations possible), our algorithms achieve regret of order $O(NM\log\log(LT))$ under fixed valuations model, $\widetilde{O}(\sqrt{NMLT})$ under random experiences model and $\widetilde{O}(\sqrt{NMLT})$ under random valuations model in $T$ rounds.</details>

### Optimal Excess Risk Bounds for Empirical Risk Minimization on $p$-Norm Linear Regression

**Authors:** Ayoub El Hanchi, Murat Erdogdu

**<details><summary>Abstract**</summary>

We study the performance of empirical risk minimization on the $p$-norm linear regression problem for $p \in (1, \infty)$. We show that, in the realizable case, under no moment assumptions, and up to a distribution-dependent constant, $O(d)$ samples are enough to exactly recover the target. Otherwise, for $p \in [2, \infty)$, and under weak moment assumptions on the target and the covariates, we prove a high probability excess risk bound on the empirical risk minimizer whose leading term matches, up to a constant that depends only on $p$, the asymptotically exact rate. We extend this result to the case $p \in (1, 2)$ under mild assumptions that guarantee the existence of the Hessian of the risk at its minimizer.</details>

### Optimal Time Complexities of Parallel Stochastic Optimization Methods Under a Fixed Computation Model

**Authors:** Alexander Tyurin, Peter Richtarik

**<details><summary>Abstract**</summary>

Parallelization is a popular strategy for improving the performance of methods. Optimization methods are no exception: design of efficient parallel optimization methods and tight analysis of their theoretical properties are important research endeavors. While the minimax complexities are well known for sequential optimization methods, the theory of parallel optimization methods is less explored. In this paper, we propose a new protocol that generalizes the classical oracle framework approach. Using this protocol, we establish minimax complexities for parallel optimization methods that have access to an unbiased stochastic gradient oracle with bounded variance. We consider a fixed computation model characterized by each worker requiring a fixed but worker-dependent time to calculate stochastic gradient. We prove lower bounds and develop optimal algorithms that attain them. Our results have surprising consequences for the literature of asynchronous optimization methods.</details>

### Optimal Transport Model Distributional Robustness

**Authors:** Van-Anh Nguyen, Trung Le, Anh Bui, Thanh-Toan Do, Dinh Phung

**<details><summary>Abstract**</summary>

Distributional robustness is a promising framework for training deep learning models that are less vulnerable to adversarial examples and data distribution shifts. Previous works have mainly focused on exploiting distributional robustness in the data space. In this work, we explore an optimal transport-based distributional robustness framework in model spaces. Specifically, we examine a model distribution within a Wasserstein ball centered on a given model distribution that maximizes the loss. We have developed theories that enable us to learn the optimal robust center model distribution. Interestingly, our developed theories allow us to flexibly incorporate the concept of sharpness awareness into training, whether it's a single model, ensemble models, or Bayesian Neural Networks, by considering specific forms of the center model distribution. These forms include a Dirac delta distribution over a single model, a uniform distribution over several models, and a general Bayesian Neural Network. Furthermore, we demonstrate that Sharpness-Aware Minimization (SAM) is a specific case of our framework when using a Dirac delta distribution over a single model, while our framework can be seen as a probabilistic extension of SAM. To validate the effectiveness of our framework in the aforementioned settings, we conducted extensive experiments, and the results reveal remarkable improvements compared to the baselines.</details>

### [Spotlight] Optimal Transport-Guided Conditional Score-Based Diffusion Model

**Authors:** Xiang Gu, Liwei Yang, Jian Sun, Zongben Xu

**<details><summary>Abstract**</summary>

Conditional score-based diffusion model (SBDM) is for conditional generation of target data with paired data as condition, and has achieved great success in image translation. However, it requires the paired data as condition, and there would be insufficient paired data provided in real-world applications. To tackle the applications with partially paired or even unpaired dataset, we propose a novel Optimal Transport-guided Conditional Score-based diffusion model (OTCS) in this paper. We build the coupling relationship for the unpaired or partially paired dataset based on $L_2$-regularized unsupervised or semi-supervised optimal transport, respectively. Based on the coupling relationship, we develop the objective for training the conditional score-based model for unpaired or partially paired settings, which is based on a reformulation and generalization of the conditional SBDM for paired setting. With the estimated coupling relationship, we effectively train the conditional score-based model by designing  a ``resampling-by-compatibility'' strategy to choose the sampled data with high compatibility as guidance. Extensive experiments on unpaired super-resolution and semi-paired image-to-image translation demonstrated the effectiveness of the proposed OTCS model. From the viewpoint of optimal transport, OTCS provides an approach to transport data across distributions, which is a challenge for OT on large-scale datasets. We theoretically prove that OTCS realizes the data transport in OT with a theoretical bound.</details>

### Optimal cross-learning for contextual bandits with unknown context distributions

**Authors:** Jon Schneider, Julian Zimmert

**<details><summary>Abstract**</summary>

We consider the problem of designing contextual bandit algorithms in the ``cross-learning'' setting of Balseiro et al., where the learner observes the loss for the action they play in all possible contexts, not just the context of the current round. We specifically consider the setting where losses are chosen adversarially and contexts are sampled i.i.d. from an unknown distribution. In this setting, we resolve an open problem of Balseiro et al. by providing an efficient algorithm with a nearly tight (up to logarithmic factors) regret bound of $\widetilde{O}(\sqrt{TK})$, independent of the number of contexts. As a consequence, we obtain the first nearly tight regret bounds for the problems of learning to bid in first-price auctions (under unknown value distributions) and sleeping bandits with a stochastic action set.At the core of our algorithm is a novel technique for coordinating the execution of a learning algorithm over multiple epochs in such a way to remove correlations between estimation of the unknown distribution and the actions played by the algorithm. This technique may be of independent interest for other learning problems involving estimation of an unknown context distribution.</details>

### Optimization and Bayes: A Trade-off for Overparameterized Neural Networks

**Authors:** Zhengmian Hu, Heng Huang

**<details><summary>Abstract**</summary>

This paper proposes a novel algorithm, Transformative Bayesian Learning (TansBL), which bridges the gap between empirical risk minimization (ERM) and Bayesian learning for neural networks. We compare ERM, which uses gradient descent to optimize, and Bayesian learning with importance sampling for their generalization and computational complexity. We derive the first algorithm-dependent PAC-Bayesian generalization bound for infinitely wide networks based on an exact KL divergence between the trained posterior distribution obtained by infinitesimal step size gradient descent and a Gaussian prior. Moreover, we show how to transform gradient-based optimization into importance sampling by incorporating a weight. While Bayesian learning has better generalization, it suffers from low sampling efficiency. Optimization methods, on the other hand, have good sampling efficiency but poor generalization. Our proposed algorithm TansBL enables a trade-off between generalization and sampling efficiency.</details>

### Out-of-distribution Detection Learning with Unreliable Out-of-distribution Sources

**Authors:** Haotian Zheng, Qizhou Wang, Zhen Fang, Xiaobo Xia, Feng Liu, Tongliang Liu, Bo Han

**<details><summary>Abstract**</summary>

Out-of-distribution (OOD) detection discerns OOD data where the predictor cannot make valid predictions as in-distribution (ID) data, thereby increasing the reliability of open-world classification. However, it is typically hard to collect real out-of-distribution (OOD) data for training a predictor capable of discerning ID and OOD patterns. This obstacle gives rise to *data generation-based learning methods*, synthesizing OOD data via data generators for predictor training without requiring any real OOD data. Related methods typically pre-train a generator on ID data and adopt various selection procedures to find those data likely to be the OOD cases. However, generated data may still coincide with ID semantics, i.e., mistaken OOD generation remains, confusing the predictor between ID and OOD data. To this end, we suggest that generated data (with mistaken OOD generation) can be used to devise an *auxiliary OOD detection task* to facilitate real OOD detection. Specifically, we can ensure that learning from such an auxiliary task is beneficial if the ID and the OOD parts have disjoint supports, with the help of a well-designed training procedure for the predictor. Accordingly, we propose a powerful data generation-based learning method named *Auxiliary Task-based OOD Learning* (ATOL) that can relieve the mistaken OOD generation. We conduct extensive experiments under various OOD detection setups, demonstrating the effectiveness of our method against its advanced counterparts.</details>

### [Spotlight] Outlier-Robust Gromov-Wasserstein for Graph Data

**Authors:** Lemin Kong, Jiajin Li, Jianheng Tang, Anthony Man-Cho So

**<details><summary>Abstract**</summary>

Gromov-Wasserstein (GW) distance is a powerful tool for comparing and aligning probability distributions supported on different metric spaces. Recently, GW has become the main modeling technique for aligning heterogeneous data for a wide range of graph learning tasks. However, the GW distance is known to be highly sensitive to outliers, which can result in large inaccuracies if the outliers are given the same weight as other samples in the objective function. To mitigate this issue, we introduce a new and robust version of the GW distance called RGW. RGW features optimistically perturbed marginal constraints within a Kullback-Leibler divergence-based ambiguity set. To make the benefits of RGW more accessible in practice, we develop a computationally efficient and theoretically provable procedure using Bregman proximal alternating linearized minimization algorithm. Through extensive experimentation, we validate our theoretical results and demonstrate the effectiveness of RGW on real-world graph learning tasks, such as subgraph matching and partial shape correspondence.</details>

### [Spotlight] PAC Learning Linear Thresholds from Label Proportions

**Authors:** Anand Brahmbhatt, Rishi Saket, Aravindan Raghuveer

**<details><summary>Abstract**</summary>

Learning from label proportions (LLP) is a generalization of supervised learning in which the training data is available as sets or bags of feature-vectors (instances) along with the average instance-label of each bag. The goal is to train a good instance classifier. While most previous works on LLP have focused on training models on such training data, computational learnability of LLP was onlyrecently explored by Saket (2021, 2022) who showed worst case intractability of properly learning linear threshold functions (LTFs) from label proportions. However, their work did not rule out efficient algorithms for this problem for natural distributions.In this work we show that it is indeed possible to efficiently learn LTFs using LTFs when given access to random bags of some label proportion in which feature-vectors are, conditioned on their labels, independently sampled from a Gaussian distribution $N(µ, Σ)$. Our work shows that a certain matrix – formed using covariances of the differences of feature-vectors sampled from the bags with and without replacement – necessarily has its principal component, after a transformation, in the direction of the normal vector of the LTF. Our algorithm estimates the means and covariance matrices using subgaussian concentration bounds which we show can be applied to efficiently sample bags for approximating the normal direction. Using this in conjunction with novel generalization error bounds in the bag setting, we show that a low error hypothesis LTF can be identified. For some special cases of the $N(0, I)$ distribution we provide a simpler mean estimation based algorithm. We include an experimental evaluation of our learning algorithms along with a comparison with those of Saket (2021, 2022) and random LTFs, demonstrating the effectiveness of our techniques.</details>

### [Spotlight] PAPR: Proximity Attention Point Rendering

**Authors:** Yanshu Zhang, Shichong Peng, Alireza Moazeni, Ke Li

**<details><summary>Abstract**</summary>

Learning accurate and parsimonious point cloud representations of scene surfaces from scratch remains a challenge in 3D representation learning. Existing point-based methods often suffer from the vanishing gradient problem or require a large number of points to accurately model scene geometry and texture. To address these limitations, we propose Proximity Attention Point Rendering (PAPR), a novel method that consists of a point-based scene representation and a differentiable renderer. Our scene representation uses a point cloud where each point is characterized by its spatial position, foreground score, and view-independent feature vector. The renderer selects the relevant points for each ray and produces accurate colours using their associated features. PAPR effectively learns point cloud positions to represent the correct scene geometry, even when the initialization drastically differs from the target geometry. Notably, our method captures fine texture details while using only a parsimonious set of points. We also demonstrate four practical applications of our method: geometry editing, object manipulation, texture transfer, and exposure control. More results and code are available on our project website at https://zvict.github.io/papr/.</details>

### PERFOGRAPH: A Numerical Aware Program Graph Representation for Performance Optimization and Program Analysis

**Authors:** Ali TehraniJamsaz, Quazi Ishtiaque Mahmud, Le Chen, Nesreen K. Ahmed, Ali Jannesari

**<details><summary>Abstract**</summary>

The remarkable growth and significant success of machine learning have expanded its applications into programming languages and program analysis. However, a key challenge in adopting the latest machine learning methods is the representation of programming languages which has a direct impact on the ability of machine learning methods to reason about programs. The absence of numerical awareness, aggregate data structure information, and improper way of presenting variables in previous representation works have limited their performances.  To overcome the limitations and challenges of current program representations, we propose a novel graph-based program representation called PERFOGRAPH. PERFOGRAPH can capture numerical information and the aggregate data structure by introducing new nodes and edges. Furthermore, we propose an adapted embedding method to incorporate numerical awareness.These enhancements make PERFOGRAPH a highly flexible and scalable representation that can effectively capture programs' intricate dependencies and semantics. Consequently, it serves as a powerful tool for various applications such as program analysis, performance optimization, and parallelism discovery. Our experimental results demonstrate that PERFOGRAPH outperforms existing representations and sets new state-of-the-art results by reducing the error rate by 7.4% (AMD dataset) and 10% (NVIDIA dataset) in the well-known Device Mapping challenge. It also sets new state-of-the-art results in various performance optimization tasks like Parallelism Discovery and Numa and Prefetchers Configuration prediction.</details>

### PETAL: Physics Emulation Through Averaged Linearizations for Solving Inverse Problems

**Authors:** Jihui Jin, Etienne Ollivier, Richard Touret, Matthew McKinley, Karim Sabra, Justin Romberg

**<details><summary>Abstract**</summary>

Inverse problems describe the task of recovering an underlying signal of interest given observables. Typically, the observables are related via some non-linear forward model applied to the underlying unknown signal. Inverting the non-linear forward model can be computationally expensive, as it often involves computing and inverting a linearization at a series of estimates. Rather than inverting the physics-based model, we instead train a surrogate forward model (emulator) and leverage modern auto-grad libraries to solve for the input within a classical optimization framework. Current methods to train emulators are done in a black box supervised machine learning fashion and fail to take advantage of any existing knowledge of the forward model. In this article, we propose a simple learned weighted average model that embeds linearizations of the forward model around various reference points into the model itself, explicitly incorporating known physics. Grounding the learned model with physics based linearizations improves the forward modeling accuracy and provides richer physics based gradient information during the inversion process leading to more accurate signal recovery. We demonstrate the efficacy on an ocean acoustic tomography (OAT) example that aims to recover ocean sound speed profile (SSP) variations from acoustic observations (e.g. eigenray arrival times) within simulation of ocean dynamics in the Gulf of Mexico.</details>

### POMDP Planning for Object Search in Partially Unknown Environment

**Authors:** Yongbo Chen, Hanna Kurniawati

**<details><summary>Abstract**</summary>

Efficiently searching for target objects in complex environments that contain various types of furniture, such as shelves, tables, and beds, is crucial for mobile robots, but it poses significant challenges due to various factors such as localization errors, limited field of view, and visual occlusion. To address this problem, we propose a Partially Observable Markov Decision Process (POMDP) formulation with a growing state space for object search in a 3D region. We solve this POMDP by carefully designing a perception module and developing a planning algorithm, called Growing Partially Observable Monte-Carlo Planning (GPOMCP), based on online Monte-Carlo tree search and belief tree reuse with a novel upper confidence bound. We have demonstrated that belief tree reuse is reasonable and achieves good performance when the belief differences are limited. Additionally, we introduce a guessed target object with an updating grid world to guide the search in the information-less and reward-less cases, like the absence of any detected objects. We tested our approach using Gazebo simulations on four scenarios of target finding in a realistic indoor living environment with the Fetch robot simulator. Compared to the baseline approaches, which are based on POMCP, our results indicate that our approach enables the robot to find the target object with a higher success rate faster while using the same computational requirements.</details>

### PTQD: Accurate Post-Training Quantization for Diffusion Models

**Authors:** Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, Bohan Zhuang

**<details><summary>Abstract**</summary>

Diffusion models have recently dominated image synthesis and other related generative tasks. However, the iterative denoising process is expensive in computations at inference time, making diffusion models less practical for low-latency and scalable real-world applications. Post-training quantization of diffusion models can significantly reduce the model size and accelerate the sampling process without requiring any re-training. Nonetheless, applying existing post-training quantization methods directly to low-bit diffusion models can significantly impair the quality of generated samples. Specifically, for each denoising step, quantization noise leads to deviations in the estimated mean and mismatches with the predetermined variance schedule. Moreover, as the sampling process proceeds, the quantization noise may accumulate, resulting in a low signal-to-noise ratio (SNR) during the later denoising steps. To address these challenges, we propose a unified formulation for the quantization noise and diffusion perturbed noise in the quantized denoising process. Specifically, we first disentangle the quantization noise into its correlated and residual uncorrelated parts regarding its full-precision counterpart. The correlated part can be easily corrected by estimating the correlation coefficient. For the uncorrelated part, we subtract the bias from the quantized results to correct the mean deviation and calibrate the denoising variance schedule to absorb the excess variance resulting from quantization. Moreover, we introduce a mixed-precision scheme for selecting the optimal bitwidth for each denoising step, which prioritizes lower bitwidths to expedite early denoising steps, while ensuring that higher bitwidths maintain a high signal-to-noise ratio (SNR) in the later steps. Extensive experiments demonstrate that our method outperforms previous post-training quantized diffusion models in generating high-quality samples, with only a $0.06$ increase in FID score compared to full-precision LDM-4 on ImageNet $256	imes256$, while saving $19.9	imes$ bit operations. Code is available at [https://github.com/ziplab/PTQD](https://github.com/ziplab/PTQD).</details>

### PUe: Biased Positive-Unlabeled Learning Enhancement by Causal Inference

**Authors:** Xutao Wang, Hanting Chen, Tianyu Guo, Yunhe Wang

**<details><summary>Abstract**</summary>

Positive-Unlabeled (PU) learning aims to achieve high-accuracy binary classification with limited labeled positive examples and numerous unlabeled ones. Existing cost-sensitive-based methods often rely on strong assumptions that examples with an observed positive label were selected entirely at random. In fact, the uneven distribution of labels is prevalent in real-world PU problems, indicating that most actual positive and unlabeled data are subject to selection bias. In this paper, we propose a PU learning enhancement (PUe) algorithm based on causal inference theory, which employs normalized propensity scores and normalized inverse probability weighting (NIPW) techniques to reconstruct the loss function, thus obtaining a consistent, unbiased estimate of the classifier and enhancing the model's performance. Moreover, we investigate and propose a method for estimating propensity scores in deep learning using regularization techniques when the labeling mechanism is unknown. Our experiments on three benchmark datasets demonstrate the proposed PUe algorithm significantly improves the accuracy of classifiers on non-uniform label distribution datasets compared to advanced cost-sensitive PU methods. Codes are available at https://github.com/huawei-noah/Noah-research/tree/master/PUe and https://gitee.com/mindspore/models/tree/master/research/cv/PUe.</details>

### Pairwise Causality Guided Transformers for Event Sequences

**Authors:** Xiao Shou, Debarun Bhattacharjya, Tian Gao, Dharmashankar Subramanian, Oktie Hassanzadeh, Kristin P Bennett

**<details><summary>Abstract**</summary>

Although pairwise causal relations have been extensively studied in observational longitudinal analyses across many disciplines, incorporating knowledge of causal pairs into deep learning models for temporal event sequences remains largely unexplored. In this paper, we propose a novel approach for enhancing the performance of transformer-based models in multivariate event sequences by injecting pairwise qualitative causal knowledge such as `event Z amplifies future occurrences of event Y'. We establish a new framework for causal inference in temporal event sequences using a transformer architecture, providing a theoretical justification for our approach, and show how to obtain unbiased estimates of the proposed measure. Experimental results demonstrate that our approach outperforms several state-of-the-art models in terms of prediction accuracy by effectively leveraging knowledge about causal pairs. We also consider a unique application where we extract knowledge around sequences of societal events by generating them from a large language model, and demonstrate how a causal knowledge graph can help with event prediction in such sequences. Overall, our framework offers a practical means of improving the performance of transformer-based models in multivariate event sequences by explicitly exploiting pairwise causal information.</details>

### Parameterizing Context: Unleashing the Power of Parameter-Efficient Fine-Tuning and In-Context Tuning for Continual Table Semantic Parsing

**Authors:** Yongrui Chen, Shenyu Zhang, Guilin Qi, Xinnan Guo

**<details><summary>Abstract**</summary>

Continual table semantic parsing aims to train a parser on a sequence of tasks, where each task requires the parser to translate natural language into SQL based on task-specific tables but only offers limited training examples. Conventional methods tend to suffer from overfitting with limited supervision, as well as catastrophic forgetting due to parameter updates.Despite recent advancements that partially alleviate these issues through semi-supervised data augmentation and retention of a few past examples, the performance is still limited by the volume of unsupervised data and stored examples.To overcome these challenges, this paper introduces a novel method integrating parameter-efficient fine-tuning (PEFT) and in-context tuning (ICT) for training a continual table semantic parser. Initially, we present a task-adaptive PEFT framework capable of fully circumventing catastrophic forgetting, which is achieved by freezing the pre-trained model backbone and fine-tuning small-scale prompts. Building on this, we propose a teacher-student framework-based solution. The teacher addresses the few-shot problem using ICT, which procures contextual information by demonstrating a few training examples. In turn, the student leverages the proposed PEFT framework to learn from the teacher's output distribution, and subsequently compresses and saves the contextual information to the prompts, eliminating the need to store any training examples.Experimental evaluations on two benchmarks affirm the superiority of our method over prevalent few-shot and continual learning baselines across various metrics.</details>

### Partial Label Learning with Dissimilarity Propagation guided Candidate Label Shrinkage

**Authors:** Yuheng Jia, Fuchao Yang, Yongqiang Dong

**<details><summary>Abstract**</summary>

In partial label learning (PLL), each sample is associated with a group of candidate labels, among which only one label is correct. The key of PLL is to disambiguate the candidate label set to find the ground-truth label. To this end, we first construct a constrained regression model to capture the confidence of the candidate labels, and multiply the label confidence matrix by its transpose to build a second-order similarity matrix, whose elements indicate the pairwise similarity relationships of samples globally. Then we develop a semantic dissimilarity matrix by considering the complement of the intersection of the candidate label set, and further propagate the initial dissimilarity relationships to the whole data set by leveraging the local geometric structure of samples. The similarity and dissimilarity matrices form an adversarial relationship, which is further utilized to shrink the solution space of the label confidence matrix and promote the dissimilarity matrix. We finally extend the proposed model to a kernel version to exploit the non-linear structure of samples and solve the proposed model by the inexact augmented Lagrange multiplier method. By exploiting the adversarial prior, the proposed method can significantly outperformstate-of-the-art PLL algorithms when evaluated on 10 artificial and 7 real-world partial label data sets. We also prove the effectiveness of our method with some theoretical guarantees. The code is publicly available at https://github.com/Yangfc-ML/DPCLS.</details>

### Passive learning of active causal strategies in agents and language models

**Authors:** Andrew Lampinen, Stephanie Chan, Ishita Dasgupta, Andrew Nam, Jane Wang

**<details><summary>Abstract**</summary>

What can be learned about causality and experimentation from passive data? This question is salient given recent successes of passively-trained language models in interactive domains such as tool use. Passive learning is inherently limited. However, we show that purely passive learning can in fact allow an agent to learn generalizable strategies for determining and using causal structures, as long as the agent can intervene at test time. We formally illustrate that learning a strategy of first experimenting, then seeking goals, can allow generalization from passive learning in principle. We then show empirically that agents trained via imitation on expert data can indeed generalize at test time to infer and use causal links which are never present in the training data; these agents can also generalize experimentation strategies to novel variable sets never observed in training.We then show that strategies for causal intervention and exploitation can be generalized from passive data even in a more complex environment with high-dimensional observations, with the support of natural language explanations. Explanations can even allow passive learners to generalize out-of-distribution from perfectly-confounded training data. Finally, we show that language models, trained only on passive next-word prediction, can generalize causal intervention strategies from a few-shot prompt containing explanations and reasoning. These results highlight the surprising power of passive learning of active causal strategies, and have implications for understanding the behaviors and capabilities of language models.</details>

### Phase diagram of early training dynamics in deep neural networks: effect of the learning rate, depth, and width

**Authors:** Dayal Singh Kalra, Maissam Barkeshli

**<details><summary>Abstract**</summary>

We systematically analyze optimization dynamics in deep neural networks (DNNs) trained with stochastic gradient descent (SGD) and study the effect of learning rate $\eta$, depth $d$, and width $w$ of the neural network. By analyzing the maximum eigenvalue $\lambda^H_t$ of the Hessian of the loss, which is a measure of sharpness of the loss landscape, we find that the dynamics can show four distinct regimes: (i) an early time transient regime, (ii) an intermediate saturation regime, (iii) a progressive sharpening regime, and (iv) a late time "edge of stability" regime. The early and intermediate regimes (i) and (ii) exhibit a rich phase diagram depending on $\eta \equiv c / \lambda_0^H $, $d$, and $w$. We identify several critical values of $c$, which separate qualitatively distinct phenomena in the early time dynamics of training loss and sharpness. Notably, we discover the opening up of a "sharpness reduction" phase, where sharpness decreases at early times, as $d$ and $ 1/w$ are increased.</details>

### PoET: A generative model of protein families as sequences-of-sequences

**Authors:** Timothy Truong Jr, Tristan Bepler

**<details><summary>Abstract**</summary>

Generative protein language models are a natural way to design new proteins with desired functions. However, current models are either difficult to direct to produce a protein from a specific family of interest, or must be trained on a large multiple sequence alignment (MSA) from the specific family of interest, making them unable to benefit from transfer learning across families. To address this, we propose **P**r**o**tein **E**volutionary **T**ransformer (PoET), an autoregressive generative model of whole protein families that learns to generate sets of related proteins as sequences-of-sequences across tens of millions of natural protein sequence clusters. PoET can be used as a retrieval-augmented language model to generate and score arbitrary modifications conditioned on any protein family of interest, and can extrapolate from short context lengths to generalize well even for small families. This is enabled by a unique Transformer layer; we model tokens sequentially within sequences while attending between sequences order invariantly, allowing PoET to scale to context lengths beyond those used during training. In extensive experiments on deep mutational scanning datasets, we show that PoET outperforms existing protein language models and evolutionary sequence models for variant function prediction across proteins of all MSA depths. We also demonstrate PoET's ability to controllably generate new protein sequences.</details>

### Pointwise uncertainty quantification for sparse variational Gaussian process regression with a Brownian motion prior

**Authors:** Luke Travis, Kolyan Ray

**<details><summary>Abstract**</summary>

We study pointwise estimation and uncertainty quantification for a sparse variational Gaussian process method with eigenvector inducing variables. For a rescaled Brownian motion prior, we derive theoretical guarantees and limitations for the frequentist size and coverage of pointwise credible sets. For sufficiently many inducing variables, we precisely characterize the asymptotic frequentist coverage, deducing when credible sets from this variational method are conservative and when overconfident/misleading. We numerically illustrate the applicability of our results and discuss connections with other common Gaussian process priors.</details>

### Polyhedron Attention Module: Learning Adaptive-order Interactions

**Authors:** Tan Zhu, Fei Dou, Xinyu Wang, Jin Lu, Jinbo Bi

**<details><summary>Abstract**</summary>

Learning feature interactions can be the key for multivariate predictive modeling. ReLU-activated neural networks create piecewise linear prediction models, and other nonlinear activation functions lead to models with only high-order feature interactions. Recent methods incorporate candidate polynomial terms of fixed orders into deep learning, which is subject to the issue of combinatorial explosion, or learn the orders that are difficult to adapt to different regions of the feature space. We propose a Polyhedron Attention Module (PAM) to create piecewise polynomial models where the input space is split into polyhedrons which define the different pieces and on each piece the hyperplanes that define the polyhedron boundary multiply to form the interactive terms, resulting in interactions of adaptive order to each piece. PAM is interpretable to identify important interactions in predicting a target. Theoretic analysis shows that PAM has stronger expression capability than ReLU-activated networks. Extensive experimental results demonstrate the superior classification performance of PAM on massive datasets of the click-through rate prediction and PAM can learn meaningful interaction effects in a medical problem.</details>

### Post Hoc Explanations of Language Models Can Improve Language Models

**Authors:** Satyapriya Krishna, Jiaqi Ma, Dylan Slack, Asma Ghandeharioun, Sameer Singh, Himabindu Lakkaraju

**<details><summary>Abstract**</summary>

Large Language Models (LLMs) have demonstrated remarkable capabilities in performing complex tasks. Moreover, recent research has shown that incorporating human-annotated rationales (e.g., Chain-of-Thought prompting) during in-context learning can significantly enhance the performance of these models, particularly on tasks that require reasoning capabilities. However, incorporating such rationales poses challenges in terms of scalability as this requires a high degree of human involvement. In this work, we present a novel framework, Amplifying Model Performance by Leveraging In-Context Learning with Post Hoc Explanations (AMPLIFY), which addresses the aforementioned challenges by automating the process of rationale generation. To this end, we leverage post hoc explanation methods which output attribution scores (explanations) capturing the influence of each of the input features on model predictions. More specifically, we construct automated natural language rationales that embed insights from post hoc explanations to provide corrective signals to LLMs. Extensive experimentation with real-world datasets demonstrates that our framework, AMPLIFY, leads to prediction accuracy improvements of about 10-25% over a wide range of tasks, including those where prior approaches which rely on human-annotated rationales such as Chain-of-Thought prompting fall short. Our work makes one of the first attempts at highlighting the potential of post hoc explanations as valuable tools for enhancing the effectiveness of LLMs. Furthermore, we conduct additional empirical analyses and ablation studies to demonstrate the impact of each of the components of AMPLIFY, which, in turn, lead to critical insights for refining in context learning.</details>

### Posthoc privacy guarantees for collaborative inference with modified Propose-Test-Release

**Authors:** Abhishek Singh, Praneeth Vepakomma, Vivek Sharma, Ramesh Raskar

**<details><summary>Abstract**</summary>

Cloud-based machine learning inference is an emerging paradigm where users query by sending their data through a service provider who runs an ML model on that data and returns back the answer. Due to increased concerns over data privacy, recent works have proposed Collaborative Inference (CI) to learn a privacy-preserving encoding of sensitive user data before it is shared with an untrusted service provider. Existing works so far evaluate the privacy of these encodings through empirical reconstruction attacks. In this work, we develop a new framework that provides formal privacy guarantees for an arbitrarily trained neural network by linking its local Lipschitz constant with its local sensitivity. To guarantee privacy using local sensitivity, we extend the Propose-Test-Release (PTR) framework to make it tractable for neural network queries. We verify the efficacy of our framework experimentally on real-world datasets and elucidate the role of Adversarial Representation Learning (ARL) in improving the privacy-utility trade-off.</details>

### Pre-training Contextualized World Models with In-the-wild Videos for Reinforcement Learning

**Authors:** Jialong Wu, Haoyu Ma, Chaoyi Deng, Mingsheng Long

**<details><summary>Abstract**</summary>

Unsupervised pre-training methods utilizing large and diverse datasets have achieved tremendous success across a range of domains. Recent work has investigated such unsupervised pre-training methods for model-based reinforcement learning (MBRL) but is limited to domain-specific or simulated data. In this paper, we study the problem of pre-training world models with abundant in-the-wild videos for efficient learning of downstream visual control tasks. However, in-the-wild videos are complicated with various contextual factors, such as intricate backgrounds and textured appearance, which precludes a world model from extracting shared world knowledge to generalize better. To tackle this issue, we introduce Contextualized World Models (ContextWM) that explicitly separate context and dynamics modeling to overcome the complexity and diversity of in-the-wild videos and facilitate knowledge transfer between distinct scenes. Specifically, a contextualized extension of the latent dynamics model is elaborately realized by incorporating a context encoder to retain contextual information and empower the image decoder, which encourages the latent dynamics model to concentrate on essential temporal variations. Our experiments show that in-the-wild video pre-training equipped with ContextWM can significantly improve the sample efficiency of MBRL in various domains, including robotic manipulation, locomotion, and autonomous driving. Code is available at this repository: https://github.com/thuml/ContextWM.</details>

### Preference-grounded Token-level Guidance for Language Model Fine-tuning

**Authors:** Shentao Yang, Shujian Zhang, Congying Xia, Yihao Feng, Caiming Xiong, Mingyuan Zhou

**<details><summary>Abstract**</summary>

Aligning language models (LMs) with preferences is an important problem in natural language generation. A key challenge is that preferences are typically provided at the *sequence level* while LM training and generation both occur at the *token level*. There is, therefore, a *granularity mismatch* between the preference and the LM training losses, which may complicate the learning problem. In this paper, we address this issue by developing an alternate training process, where we iterate between grounding the sequence-level preference into token-level training guidance, and improving the LM with the learned guidance. For guidance learning, we design a framework that extends the pairwise-preference learning in imitation learning to both variable-length LM generation and the utilization of the preference among multiple generations. For LM training, based on the amount of supervised data, we present two *minimalist* learning objectives that utilize the learned guidance. In experiments, our method performs competitively on two distinct representative LM tasks --- discrete-prompt generation and text summarization.</details>

### Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression

**Authors:** Allan Raventós, Mansheej Paul, Feng Chen, Surya Ganguli

**<details><summary>Abstract**</summary>

Pretrained transformers exhibit the remarkable ability of in-context learning (ICL): they can learn tasks from just a few examples provided in the prompt without updating any weights. This raises a foundational question: can ICL solve fundamentally _new_ tasks that are very different from those seen during pretraining? To probe this question, we examine ICL’s performance on linear regression while varying the diversity of tasks in the pretraining dataset. We empirically demonstrate a _task diversity threshold_ for the emergence of ICL. Below this threshold, the pretrained transformer cannot solve unseen regression tasks, instead behaving like a Bayesian estimator with the _non-diverse pretraining task distribution_ as the prior. Beyond this threshold, the transformer significantly outperforms this estimator; its behavior aligns with that of ridge regression, corresponding to a Gaussian prior over _all tasks_, including those not seen during pretraining. Thus, when pretrained on data with task diversity greater than the threshold, transformers _can_ optimally solve fundamentally new tasks in-context. Importantly, this capability hinges on it deviating from the Bayes optimal estimator with the pretraining distribution as the prior. This study also explores the effect of regularization, model capacity and task structure and underscores, in a concrete example, the critical role of task diversity, alongside data and model scale, in the emergence of ICL.</details>

### [Oral] Privacy Auditing with One (1) Training Run

**Authors:** Thomas Steinke, Milad Nasr, Matthew Jagielski

**Oral Presentation:** Tu, Dec 12, 13:55 -- Oral 2D

**<details><summary>Abstract**</summary>

We propose a scheme for auditing differentially private machine learning systems with a single training run. This exploits the parallelism of being able to add or remove multiple training examples independently. We analyze this using the connection between differential privacy and statistical generalization, which avoids the cost of group privacy. Our auditing scheme requires minimal assumptions about the algorithm and can be applied in the black-box or white-box setting. We demonstrate the effectiveness of our framework by applying it to DP-SGD, where we can achieve meaningful empirical privacy lower bounds by training only one model. In contrast, standard methods would require training hundreds of models.</details>

### [Oral] Private Everlasting Prediction

**Authors:** Moni Naor, Kobbi Nissim, Uri Stemmer, Chao Yan

**Oral Presentation:** Tu, Dec 12, 14:10 -- Oral 2D

**<details><summary>Abstract**</summary>

A private learner is trained on a sample of labeled points and generatesa hypothesis that can be used for predicting the labels of newly sampled points while protecting the privacy of the training set [Kasiviswannathan et al., FOCS 2008]. Research uncovered that private learners may need to exhibit significantly higher sample complexity than non-private learners as is the case with, e.g., learning of one-dimensional threshold functions [Bun et al., FOCS 2015, Alon et al., STOC 2019].We explore prediction as an alternative to learning. Instead of putting forward a hypothesis, a predictor answers a stream of classification queries. Earlier work has considered a private prediction model with just a single classification query [Dwork and Feldman, COLT 2018]. We observe that when answering a stream of queries, a predictor must modify the hypothesis it uses over time, and, furthermore, that it must use the queries for this modification, hence introducing potential privacy risks with respect to the queries themselves.We introduce private everlasting prediction taking into account the privacy of both the training set and the (adaptively chosen) queries made to the predictor. We then present a generic construction of private everlasting predictors in the PAC model. The sample complexity of the initial training sample in our construction is quadratic (up to polylog factors) in the VC dimension of the concept class. Our construction allows prediction for all concept classes with finite VC dimension, and in particular threshold functions with constant size initial training sample, even when considered over infinite domains, whereas it is known that the sample complexity of privately learning threshold functions must grow as a function of the domain size and hence is impossible for infinite domains.</details>

### Probabilistic Weight Fixing: Large-scale training of neural network weight uncertainties for quantisation.

**Authors:** Chris Subia-Waud, Srinandan Dasmahapatra

**<details><summary>Abstract**</summary>

Weight-sharing quantization has emerged as a technique to reduce energy expenditure during inference in large neural networks by constraining their weights to a limited set of values. However, existing methods often assume weights are treated solely based on value, neglecting the unique role of weight position. This paper proposes a probabilistic framework based on Bayesian neural networks (BNNs) and a variational relaxation to identify which weights can be moved to which cluster center and to what degree based on their individual position-specific learned uncertainty distributions. We introduce a new initialization setting and a regularization term, enabling the training of BNNs with complex dataset-model combinations. Leveraging the flexibility of weight values from probability distributions, we enhance noise resilience and compressibility. Our iterative clustering procedure demonstrates superior compressibility and higher accuracy compared to state-of-the-art methods on both ResNet models and the more complex transformer-based architectures. In particular, our method outperforms the state-of-the-art quantization method top-1 accuracy by 1.6\% on ImageNet using DeiT-Tiny, with its 5 million+ weights now represented by only 296 unique values. Code available at https://github.com/subiawaud/PWFN.</details>

### Projection Regret: Reducing Background Bias for Novelty Detection via Diffusion Models

**Authors:** Sungik Choi, Hankook Lee, Honglak Lee, Moontae Lee

**<details><summary>Abstract**</summary>

Novelty detection is a fundamental task of machine learning which aims to detect abnormal (*i.e.* out-of-distribution (OOD)) samples. Since diffusion models have recently emerged as the de facto standard generative framework with surprising generation results, novelty detection via diffusion models has also gained much attention. Recent methods have mainly utilized the reconstruction property of in-distribution samples. However, they often suffer from detecting OOD samples that share similar background information to the in-distribution data. Based on our observation that diffusion models can *project* any sample to an in-distribution sample with similar background information, we propose *Projection Regret (PR)*, an efficient novelty detection method that mitigates the bias of non-semantic information. To be specific, PR computes the perceptual distance between the test image and its diffusion-based projection to detect abnormality. Since the perceptual distance often fails to capture semantic changes when the background information is dominant, we cancel out the background bias by comparing it against recursive projections. Extensive experiments demonstrate that PR outperforms the prior art of generative-model-based novelty detection methods by a significant margin.</details>

### ProteinNPT: Improving protein property prediction and design with non-parametric transformers

**Authors:** Pascal Notin, Ruben Weitzman, Debora Marks, Yarin Gal

**<details><summary>Abstract**</summary>

Protein design holds immense potential for optimizing naturally occurring sequences, with broad applications in drug discovery, material design, and sustainability. However, computational methods for protein engineering are confronted with significant challenges, including an expansive design space, sparse functional regions, and scarcity of available labels. Furthermore, real-life design scenarios often necessitate the simultaneous optimization of multiple properties, exacerbating label sparsity issues. In this paper, we present ProteinNPT, a non-parametric transformer variant tailored for protein sequences and particularly suited to label-scarce and multi-task optimization settings. We first expand the ProteinGym benchmark to evaluate models in supervised settings and develop several cross-validation schemes for robust assessment. Subsequently, we reimplement existing top-performing baselines, introduce several extensions of these baselines by integrating diverse branches of protein engineering literature, and demonstrate that ProteinNPT consistently outperforms all of them across a diverse set of protein property prediction tasks. Finally, we demonstrate the value of our approach for iterative protein design in several in silico Bayesian optimization experiments.</details>

### Provable Advantage of Curriculum Learning on Parity Targets with Mixed Inputs

**Authors:** Emmanuel Abbe, Elisabetta Cornacchia, Aryo Lotfi

**<details><summary>Abstract**</summary>

Experimental results have shown that curriculum learning, i.e., presenting simpler examples before more complex ones, can improve the efficiency of learning. Some recent theoretical results also showed that changing the sampling distribution can help neural networks learn parities, with formal results only for large learning rates and one-step arguments. Here we show a separation result in the number of training steps with standard (bounded) learning rates on a common sample distribution: if the data distribution is a mixture of sparse and dense inputs, there exists a regime in which a 2-layer ReLU neural network trained by a  curriculum noisy-GD (or SGD) algorithm that uses sparse examples first, can learn parities of sufficiently large degree, while any fully connected neural network of possibly larger width or depth trained by noisy-GD on the unordered samples cannot learn without additional steps. We also provide experimental results supporting the qualitative separation beyond the specific regime of the theoretical results.</details>

### Pruning vs Quantization: Which is Better?

**Authors:** Andrey Kuzmin, Markus Nagel, Mart van Baalen, Arash Behboodi, Tijmen Blankevoort

**<details><summary>Abstract**</summary>

Neural network pruning and quantization techniques are almost as old as neural networks themselves. However, to date, only ad-hoc comparisons between the two have been published. In this paper, we set out to answer the question of which is better: neural network quantization or pruning? By answering this question, we hope to inform design decisions made on neural network hardware going forward. We provide an extensive comparison between the two techniques for compressing deep neural networks. First, we give an analytical comparison of expected quantization and pruning error for general data distributions.Then, we provide lower and upper bounds for the per-layer pruning and quantization error in trained networks and compare these to empirical error after optimization.Finally, we provide an extensive experimental comparison for training 8 large-scale models trained on 3 tasks and provide insights into the representations learned during fine-tuning with quantization and pruning in the loop.Our results show that in most cases quantization outperforms pruning. Only in some scenarios with a very high compression ratio, compression might be beneficial from an accuracy standpoint.</details>

### [Oral] QLoRA: Efficient Finetuning of Quantized LLMs

**Authors:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer

**Oral Presentation:** Tu, Dec 12, 13:55 -- Oral 2A

**<details><summary>Abstract**</summary>

We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters~(LoRA). Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU. QLoRA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information-theoretically optimal for normally distributed weights (b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) Paged Optimziers to manage memory spikes. We use QLoRA to finetune more than 1,000 models, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5), and model scales that would be infeasible to run with regular finetuning (e.g. 33B and 65B parameter models). Our results show that QLoRA finetuning on a small, high-quality dataset leads to state-of-the-art results, even when using smaller models than the previous SoTA. We provide a detailed analysis of chatbot performance based on both human and GPT-4 evaluations, showing that GPT-4 evaluations are a cheap and reasonable alternative to human evaluation. Furthermore, we find that current chatbot benchmarks are not trustworthy to accurately evaluate the performance levels of chatbots. A lemon-picked analysis demonstrates where Guanaco fails compared to ChatGPT. We release all of our models and code, including CUDA kernels for 4-bit training.</details>

### [Spotlight] QuantSR: Accurate Low-bit Quantization for Efficient Image Super-Resolution

**Authors:** Haotong Qin, Yulun Zhang, Yifu Ding, Yifan liu, Xianglong Liu, Martin Danelljan, Fisher Yu

**<details><summary>Abstract**</summary>

Low-bit quantization in image super-resolution (SR) has attracted copious attention in recent research due to its ability to reduce parameters and operations significantly. However, many quantized SR models suffer from accuracy degradation compared to their full-precision counterparts, especially at ultra-low bit widths (2-4 bits), limiting their practical applications. To address this issue, we propose a novel quantized image SR network, called QuantSR, which achieves accurate and efficient SR processing under low-bit quantization. To overcome the representation homogeneity caused by quantization in the network, we introduce the Redistribution-driven Learnable Quantizer (RLQ). This is accomplished through an inference-agnostic efficient redistribution design, which adds additional information in both forward and backward passes to improve the representation ability of quantized networks. Furthermore, to achieve flexible inference and break the upper limit of accuracy, we propose the Depth-dynamic Quantized Architecture (DQA). Our DQA allows for the trade-off between efficiency and accuracy during inference through weight sharing. Our comprehensive experiments show that QuantSR outperforms existing state-of-the-art quantized SR networks in terms of accuracy while also providing more competitive computational efficiency. In addition, we demonstrate the scheme's satisfactory architecture generality by providing QuantSR-C and QuantSR-T for both convolution and Transformer versions, respectively. Our code and models are released at https://github.com/htqin/QuantSR .</details>

### Quantifying the Cost of Learning in Queueing Systems

**Authors:** Daniel Freund, Thodoris Lykouris, Wentao Weng

**<details><summary>Abstract**</summary>

Queueing systems are widely applicable stochastic models with use cases in communication networks, healthcare, service systems, etc. Although their optimal control has been extensively studied, most existing approaches assume perfect knowledge of the system parameters. Of course, this assumption rarely holds in practice where there is parameter uncertainty, thus motivating a recent line of work on bandit learning for queueing systems. This nascent stream of research focuses on the asymptotic performance of the proposed algorithms. In this paper, we argue that an asymptotic metric, which focuses on late-stage performance, is insufficient to capture the intrinsic statistical complexity of learning in queueing systems which typically occurs in the early stage. Instead, we propose the *Cost of Learning in Queueing (CLQ)*, a new metric that quantifies the maximum increase in time-averaged queue length caused by parameter uncertainty.We characterize the CLQ of a single-queue multi-server system, and then extend these results to multi-queue multi-server systems and networks of queues. In establishing our results, we propose a unified analysis framework for CLQ that bridges Lyapunov and bandit analysis, provides guarantees for a wide range of algorithms, and could be of independent interest.</details>

### Random-Access Infinite Context Length for Transformers

**Authors:** Amirkeivan Mohtashami, Martin Jaggi

**<details><summary>Abstract**</summary>

While Transformers have shown remarkable success in natural language processing, their attention mechanism's large memory requirements have limited their ability to handle longer contexts. Prior approaches, such as recurrent memory or retrieval-based augmentation, have either compromised the random-access flexibility of attention (i.e., the capability to select any token in the entire context) or relied on separate mechanisms for relevant context retrieval, which may not be compatible with the model's attention. In this paper, we present a novel approach that allows access to the complete context while retaining random-access flexibility, closely resembling running attention on the entire context. Our method uses a landmark token to represent each block of the input and trains the attention to use it for selecting relevant blocks, enabling retrieval of blocks directly through the attention mechanism instead of by relying on a separate mechanism. Our approach seamlessly integrates with specialized data structures and the system's memory hierarchy, enabling processing of arbitrarily long context lengths. We demonstrate that our method can obtain comparable performance with Transformer-XL while significantly reducing the number of retrieved tokens in each step. Finally, we show that fine-tuning LLaMA 7B with our method successfully extends its context length capacity to over 32k tokens, allowing for inference at the context lengths of GPT-4. We release the implementation of landmark attention and the code to reproduce our experiments at https://github.com/epfml/landmark-attention/.</details>

### Re-Think and Re-Design Graph Neural Networks in Spaces of Continuous Graph Diffusion Functionals

**Authors:** Tingting Dan, Jiaqi Ding, Ziquan Wei, Shahar Kovalsky, Minjeong Kim, Won Hwa Kim, Guorong Wu

**<details><summary>Abstract**</summary>

Graphs are ubiquitous in various domains, such as social networks and biological systems. Despite the great successes of graph neural networks (GNNs) in modeling and analyzing complex graph data, the inductive bias of locality assumption, which involves exchanging information only within neighboring connected nodes, restricts GNNs in capturing long-range dependencies and global patterns in graphs. Inspired by the classic Brachistochrone problem, we seek how to devise a new inductive bias for cutting-edge graph application and present a general framework through the lens of variational analysis. The backbone of our framework is a two-way mapping between the discrete GNN model and continuous diffusion functional, which allows us to design application-specific objective function in the continuous domain and engineer discrete deep model with mathematical guarantees. First, we address over-smoothing in current GNNs. Specifically, our inference reveals that the existing layer-by-layer models of graph embedding learning are equivalent to a ${\ell _2}$-norm integral functional of graph gradients, which is the underlying cause of the over-smoothing problem. Similar to edge-preserving filters in image denoising, we introduce the total variation (TV) to promote alignment of the graph diffusion pattern with the global information present in community topologies. On top of this, we devise a new selective mechanism for inductive bias that can be easily integrated into existing GNNs and effectively address the trade-off between model depth and over-smoothing. Second, we devise a novel generative adversarial network (GAN) to predict the spreading flows in the graph through a neural transport equation. To avoid the potential issue of vanishing flows, we tailor the objective function to minimize the transportation within each community while maximizing the inter-community flows. Our new GNN models achieve state-of-the-art (SOTA) performance on graph learning benchmarks such as Cora, Citeseer, and Pubmed.</details>

### ReHLine: Regularized Composite ReLU-ReHU Loss Minimization with Linear Computation and Linear Convergence

**Authors:** Ben Dai, Yixuan Qiu

**<details><summary>Abstract**</summary>

Empirical risk minimization (ERM) is a crucial framework that offers a general approach to handling a broad range of machine learning tasks. In this paper, we propose a novel algorithm, called ReHLine, for minimizing a set of regularized ERMs with convex piecewise linear-quadratic loss functions and optional linear constraints. The proposed algorithm can effectively handle diverse combinations of loss functions, regularization, and constraints, making it particularly well-suited for complex domain-specific problems. Examples of such problems include FairSVM, elastic net regularized quantile regression, Huber minimization, etc. In addition, ReHLine enjoys a provable linear convergence rate and exhibits a per-iteration computational complexity that scales linearly with the sample size. The algorithm is implemented with both Python and R interfaces, and its performance is benchmarked on various tasks and datasets. Our experimental results demonstrate that ReHLine significantly surpasses generic optimization solvers in terms of computational efficiency on large-scale datasets. Moreover, it also outperforms specialized solvers such as Liblinear in SVMs, hqreg in Huber minimization, and Lightning (SAGA, SAG, SDCA, SVRG) in smoothed SVMs, exhibiting exceptional flexibility and efficiency. The source code, project page, accompanying software, and the Python/R interface can be accessed through the link: https://github.com/softmin/ReHLine.</details>

### ReTR: Modeling Rendering Via Transformer for Generalizable Neural Surface Reconstruction

**Authors:** Yixun Liang, Hao He, Yingcong Chen

**<details><summary>Abstract**</summary>

Generalizable neural surface reconstruction techniques have attracted great attention in recent years. However, they encounter limitations of low confidence depth distribution and inaccurate surface reasoning due to the oversimplified volume rendering process employed. In this paper, we present Reconstruction TRansformer (ReTR), a novel framework that leverages the transformer architecture to redesign the rendering process, enabling complex render interaction modeling. It introduces a learnable $	extit{meta-ray token}$ and utilizes the cross-attention mechanism to simulate the interaction of rendering process with sampled points and render the observed color. Meanwhile, by operating within a high-dimensional feature space rather than the color space, ReTR mitigates sensitivity to projected colors in source views. Such improvements result in accurate surface assessment with high confidence. We demonstrate the effectiveness of our approach on various datasets, showcasing how our method outperforms the current state-of-the-art approaches in terms of reconstruction quality and generalization ability. $	extit{Our code is available at }$ https://github.com/YixunLiang/ReTR.</details>

### Real-World Image Super-Resolution as Multi-Task Learning

**Authors:** Wenlong Zhang, Xiaohui Li, Guangyuan SHI, Xiangyu Chen, Yu Qiao, Xiaoyun Zhang, Xiao-Ming Wu, Chao Dong

**<details><summary>Abstract**</summary>

In this paper, we take a new look at real-world image super-resolution (real-SR) from a multi-task learning perspective. We demonstrate that the conventional formulation of real-SR can be viewed as solving multiple distinct degradation tasks using a single shared model. This poses a challenge known as task competition or task conflict in multi-task learning, where certain tasks dominate the learning process, resulting in poor performance on other tasks. This problem is exacerbated in the case of real-SR, due to the involvement of numerous degradation tasks. To address the issue of task competition in real-SR, we propose a task grouping approach. Our approach efficiently identifies the degradation tasks where a real-SR model falls short and groups these unsatisfactory tasks into multiple task groups for further training. By grouping similar tasks together, our approach mitigates task competition and facilitates effective knowledge transfer. Extensive experiments demonstrate our method achieves significantly enhanced performance across a wide range of degradation scenarios.</details>

### Real3D-AD: A Dataset of Point Cloud Anomaly Detection

**Authors:** Jiaqi Liu, Guoyang Xie, Ruitao Chen, Xinpeng Li, Jinbao Wang, Yong Liu, Chengjie Wang, Feng Zheng

**<details><summary>Abstract**</summary>

High-precision point cloud anomaly detection is the gold standard for identifying the defects of advancing machining and precision manufacturing. Despite some methodological advances in this area, the scarcity of datasets and the lack of a systematic benchmark hinder its development. We introduce Real3D-AD, a challenging high-precision point cloud anomaly detection dataset, addressing the limitations in the field. With 1,254 high-resolution 3D items (from forty thousand to millions of points for each item), Real3D-AD is the largest dataset for high-precision 3D industrial anomaly detection to date. Real3D-AD surpasses existing 3D anomaly detection datasets available in terms of point cloud resolution (0.0010mm-0.0015mm), $360^{\circ}$ degree coverage and perfect prototype. Additionally, we present a comprehensive benchmark for Real3D-AD, revealing the absence of baseline methods for high-precision point cloud anomaly detection. To address this, we propose Reg3D-AD, a registration-based 3D anomaly detection method incorporating a novel feature memory bank that preserves local and global representations. Extensive experiments on the Real3D-AD dataset highlight the effectiveness of Reg3D-AD. For reproducibility and accessibility, we provide the Real3D-AD dataset, benchmark source code, and Reg3D-AD on our website: https://github.com/M-3LAB/Real3D-AD.</details>

### Recommender Systems with Generative Retrieval

**Authors:** Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, Maciej Kula, Ed Chi, Mahesh Sathiamoorthy

**<details><summary>Abstract**</summary>

Modern recommender systems perform large-scale retrieval by embedding queries and item candidates in the same unified space, followed by approximate nearest neighbor search to select top candidates given a query embedding. In this paper, we propose a novel generative retrieval approach, where the retrieval model autoregressively decodes the identifiers of the target candidates. To that end, we create semantically meaningful tuple of codewords to serve as a Semantic ID for each item. Given Semantic IDs for items in a user session, a Transformer-based sequence-to-sequence model is trained to predict the Semantic ID of the next item that the user will interact with. We show that recommender systems trained with the proposed paradigm significantly outperform the current SOTA models on various datasets. In addition, we show that incorporating Semantic IDs into the sequence-to-sequence model enhances its ability to generalize, as evidenced by the improved retrieval performance observed for items with no prior interaction history.</details>

### Recovering Simultaneously Structured Data via Non-Convex Iteratively Reweighted Least Squares

**Authors:** Christian Kümmerle, Johannes Maly

**<details><summary>Abstract**</summary>

We propose a new algorithm for the problem of recovering data that adheres to multiple, heterogenous low-dimensional structures from linear observations. Focussing on data matrices that are simultaneously row-sparse and low-rank, we propose and analyze an iteratively reweighted least squares (IRLS) algorithm that is able to leverage both structures. In particular, it optimizes a combination of non-convex surrogates for row-sparsity and rank, a balancing of which is built into the algorithm. We prove locally quadratic convergence of the iterates to a simultaneously structured data matrix in a regime of minimal sample complexity (up to constants and a logarithmic factor), which is known to be impossible for a combination of convex surrogates. In experiments, we show that the IRLS method exhibits favorable empirical convergence, identifying simultaneously row-sparse and low-rank matrices from fewer measurements than state-of-the-art methods.</details>

### Recurrent Temporal Revision Graph Networks

**Authors:** Yizhou Chen, Anxiang Zeng, Qingtao Yu, Kerui Zhang, Cao Yuanpeng, Kangle Wu, Guangda Huzhang, Han Yu, Zhiming Zhou

**<details><summary>Abstract**</summary>

Temporal graphs offer more accurate modeling of many real-world scenarios than static graphs. However, neighbor aggregation, a critical building block of graph networks, for temporal graphs, is currently straightforwardly extended from that of static graphs. It can be computationally expensive when involving all historical neighbors during such aggregation. In practice, typically only a subset of the most recent neighbors are involved. However, such subsampling leads to incomplete and biased neighbor information. To address this limitation, we propose a novel framework for temporal neighbor aggregation that uses the recurrent neural network with node-wise hidden states to integrate information from all historical neighbors for each node to acquire the complete neighbor information. We demonstrate the superior theoretical expressiveness of the proposed framework as well as its state-of-the-art performance in real-world applications. Notably, it achieves a significant +9.4% improvement on averaged precision in a real-world Ecommerce dataset over existing methods on 2-layer models.</details>

### Reference-Based POMDPs

**Authors:** Edward Kim, Yohan Karunanayake, Hanna Kurniawati

**<details><summary>Abstract**</summary>

Making good decisions in partially observable and non-deterministic scenarios is a crucial capability for robots. A Partially Observable Markov Decision Process (POMDP) is a general framework for the above problem. Despite advances in POMDP solving, problems with long planning horizons and evolving environments remain difficult to solve even by the best approximate solvers today. To alleviate this difficulty, we propose a slightly modified POMDP problem, called a Reference-Based POMDP, where the POMDP objective function is slightly modified to balance between maximizing the expected total reward and being close to a given reference (stochastic) policy. The optimal policy of a Reference-Based POMDP can be computed via iterative expectations using the given reference policy, thereby avoiding exhaustive enumeration of actions at each belief node of the search tree. We demonstrate theoretically that the standard POMDP under stochastic policies is related to the Reference-Based POMDP under suitable conditions. To demonstrate the feasibility of exploiting the Reference-Based POMDP formulation, we present a basic algorithm RefSolver. Results from experiments on long-horizon navigation problems indicate that this basic algorithm substantially outperforms POMCP.</details>

### Refined Mechanism Design for Approximately Structured Priors via Active Regression

**Authors:** Christos Boutsikas, Petros Drineas, Marios Mertzanidis, Alexandros Psomas, Paritosh Verma

**<details><summary>Abstract**</summary>

We consider the problem of a revenue-maximizing seller with a large number of items $m$ for sale to $n$ strategic bidders, whose valuations are drawn independently from high-dimensional, unknown prior distributions. It is well-known that optimal and even approximately-optimal mechanisms for this setting are notoriously difficult to characterize or compute, and, even when they can be found, are often rife with various counter-intuitive properties. In this paper, following a model introduced recently by Cai and Daskalakis [CD22], we consider the case that bidders' prior distributions can be well-approximated by a topic model. We design an active learning component, responsible for interacting with the bidders and outputting low-dimensional approximations of their types, and a mechanism design component, responsible for robustifying mechanisms for the low-dimensional model to work for the approximate types of the former component. On the active learning front, we cast our problem in the framework of Randomized Linear Algebra (RLA) for regression problems, allowing us to import several breakthrough results from that line of research, and adapt them to our setting. On the mechanism design front, we remove many restrictive assumptions of prior work on the type of access needed to the underlying distributions and the associated mechanisms. To the best of our knowledge, our work is the first to formulate connections between mechanism design, and RLA for active learning of regression problems, opening the door for further applications of randomized linear algebra primitives to mechanism design.</details>

### Regularity as Intrinsic Reward for Free Play

**Authors:** Cansu Sancaktar, Justus Piater, Georg Martius

**<details><summary>Abstract**</summary>

We propose regularity as a novel reward signal for intrinsically-motivated reinforcement learning. Taking inspiration from child development, we postulate that striving for structure and order helps guide exploration towards a subspace of tasks that are not favored by naive uncertainty-based intrinsic rewards. Our generalized formulation of Regularity as Intrinsic Reward (RaIR) allows us to operationalize it within model-based reinforcement learning. In a synthetic environment, we showcase the plethora of structured patterns that can emerge from pursuing this regularity objective. We also demonstrate the strength of our method in a multi-object robotic manipulation environment. We incorporate RaIR into free play and use it to complement the model’s epistemic uncertainty as an intrinsic reward. Doing so, we witness the autonomous construction of towers and other regular structures during free play, which leads to a substantial improvement in zero-shot downstream task performance on assembly tasks.</details>

### Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models

**Authors:** Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, Kimin Lee

**<details><summary>Abstract**</summary>

Learning from human feedback has been shown to improve text-to-image models. These techniques first learn a reward function that captures what humans care about in the task and then improve the models based on the learned reward function. Even though relatively simple approaches (e.g., rejection sampling based on reward scores) have been investigated, fine-tuning text-to-image models with the reward function remains challenging. In this work, we propose using online reinforcement learning (RL) to fine-tune text-to-image models. We focus on diffusion models, defining the fine-tuning task as an RL problem, and updating the pre-trained text-to-image diffusion models using policy gradient to maximize the feedback-trained reward. Our approach, coined DPOK, integrates policy optimization with KL regularization. We conduct an analysis of KL regularization for both RL fine-tuning and supervised fine-tuning. In our experiments, we show that DPOK is generally superior to supervised fine-tuning with respect to both image-text alignment and image quality. Our code is available at https://github.com/google-research/google-research/tree/master/dpok.</details>

### Reliable learning in challenging environments

**Authors:** Maria-Florina Balcan, Steve Hanneke, Rattana Pukdee, Dravyansh Sharma

**<details><summary>Abstract**</summary>

The problem of designing learners that provide guarantees that their predictions are provably correct is of increasing importance in machine learning. However, learning theoretic guarantees have only been considered in very specific settings.  In this work, we consider the design and analysis of reliable learners in challenging test-time environments as encountered in modern machine learning problems: namely adversarial test-time attacks (in several variations) and natural distribution shifts.  In this work, we provide a reliable learner with provably optimal guarantees in such settings. We discuss computationally feasible implementations of the learner and further show that our algorithm achieves strong positive performance guarantees on several natural examples: for example, linear separators under log-concave distributions or smooth boundary classifiers under smooth probability distributions.</details>

### Removing Hidden Confounding in Recommendation: A Unified Multi-Task Learning Approach

**Authors:** Haoxuan Li, Kunhan Wu, Chunyuan Zheng, Yanghao Xiao, Hao Wang, Zhi Geng, Fuli Feng, Xiangnan He, Peng Wu

**<details><summary>Abstract**</summary>

In recommender systems, the collected data used for training is always subject to selection bias, which poses a great challenge for unbiased learning. Previous studies proposed various debiasing methods based on observed user and item features, but ignored the effect of hidden confounding. To address this problem, recent works suggest the use of sensitivity analysis for worst-case control of the unknown true propensity, but only valid when the true propensity is near to the nominal propensity within a finite bound. In this paper, we first perform theoretical analysis to reveal the possible failure of previous approaches, including propensity-based, multi-task learning, and bi-level optimization methods, in achieving unbiased learning when hidden confounding is present. Then, we propose a unified multi-task learning approach to remove hidden confounding, which uses a few unbiased ratings to calibrate the learned nominal propensities and nominal error imputations from biased data. We conduct extensive experiments on three publicly available benchmark datasets containing a fully exposed large-scale industrial dataset, validating the effectiveness of the proposed methods in removing hidden confounding.</details>

### Replicability in Reinforcement Learning

**Authors:** Amin Karbasi, Grigoris Velegkas, Lin Yang, Felix Zhou

**<details><summary>Abstract**</summary>

We initiate the mathematical study of replicability as an   algorithmic property in the context of reinforcement learning (RL).  We focus on the fundamental setting of discounted tabular MDPs with access to a generative model.  Inspired by Impagliazzo et al. [2022], we say that an RL algorithm is replicable if,  with high probability,  it outputs the exact same policy  after two executions on i.i.d. samples drawn from the generator  when its internal randomness  is the same.  We first provide   an efficient $\rho$-replicable algorithm for $(\varepsilon, \delta)$-optimal policy estimation  with sample and time complexity $\widetilde O\left(\frac{N^3\cdot\log(1/\delta)}{(1-\gamma)^5\cdot\varepsilon^2\cdot\rho^2}\right)$,  where $N$ is the number of state-action pairs.  Next,  for the subclass of deterministic algorithms,  we provide a lower bound of order $\Omega\left(\frac{N^3}{(1-\gamma)^3\cdot\varepsilon^2\cdot\rho^2}\right)$.  Then, we study a relaxed version of replicability proposed  by Kalavasis et al. [2023] called TV indistinguishability.  We design a computationally efficient TV indistinguishable algorithm for policy estimation  whose sample complexity is $\widetilde O\left(\frac{N^2\cdot\log(1/\delta)}{(1-\gamma)^5\cdot\varepsilon^2\cdot\rho^2}\right)$.  At the cost of $\exp(N)$ running time,  we transform these TV indistinguishable algorithms to $\rho$-replicable ones without increasing their sample complexity.  Finally,  we introduce the notion of approximate-replicability  where we only require that two outputted policies are close  under an appropriate statistical divergence (e.g., Renyi)  and show an improved sample complexity of $\widetilde O\left(\frac{N\cdot\log(1/\delta)}{(1-\gamma)^5\cdot\varepsilon^2\cdot\rho^2}\right)$.</details>

### Replicable Clustering

**Authors:** Hossein Esfandiari, Amin Karbasi, Vahab Mirrokni, Grigoris Velegkas, Felix Zhou

**<details><summary>Abstract**</summary>

We design replicable algorithms in the context of statistical clustering under the recently introduced notion of replicability from Impagliazzo et al. [2022]. According to this definition, a clustering algorithm is replicable if, with high probability, its output induces the exact same partition of the sample space after two executions on different inputs drawn from the same distribution, when its internal randomness is shared across the executions. We propose such algorithms for the statistical $k$-medians, statistical $k$-means, and statistical $k$-centers problems by utilizing approximation routines for their combinatorial counterparts in a black-box manner. In particular, we demonstrate a replicable $O(1)$-approximation algorithm for statistical Euclidean $k$-medians ($k$-means) with $\operatorname{poly}(d)$ sample complexity. We also describe an $O(1)$-approximation algorithm with an additional $O(1)$-additive error for statistical Euclidean $k$-centers, albeit with $\exp(d)$ sample complexity. In addition, we provide experiments on synthetic distributions in 2D using the $k$-means++ implementation from sklearn as a black-box that validate our theoretical results.</details>

### Representation Equivalent Neural Operators: a Framework for Alias-free Operator Learning

**Authors:** Francesca Bartolucci, Emmanuel de Bézenac, Bogdan Raonic, Roberto Molinaro, Siddhartha Mishra, Rima Alaifari

**<details><summary>Abstract**</summary>

Recently, operator learning, or learning mappings between infinite-dimensional function spaces, has garnered significant attention, notably in relation to learning partial differential equations from data. Conceptually clear when outlined on paper, neural operators necessitate discretization in the transition to computer implementations. This step can compromise their integrity, often causing them to deviate from the underlying operators. This research offers a fresh take on neural operators with a framework Representation equivalent Neural Operators (ReNO) designed to address these issues. At its core is the concept of operator aliasing, which measures inconsistency between neural operators and their discrete representations. We explore this for widely-used operator learning techniques. Our findings detail how aliasing introduces errors when handling different discretizations and grids and loss of crucial continuous structures. More generally, this framework not only sheds light on existing challenges but, given its constructive and broad nature, also potentially offers tools for developing new neural operators.</details>

### Representational Strengths and Limitations of Transformers

**Authors:** Clayton Sanford, Daniel Hsu, Matus Telgarsky

**<details><summary>Abstract**</summary>

Attention layers, as commonly used in transformers, form the backbone of modern deep learning, yet there is no mathematical description of their benefits and deficiencies as compared with other architectures. In this work we establish both positive and negative results on the representation power of attention layers, with a focus on intrinsic complexity parameters such as width, depth, and embedding dimension. On the positive side, we present a sparse averaging task, where recurrent networks and feedforward networks all have complexity scaling polynomially in the input size, whereas transformers scale merely logarithmically in the input size; furthermore, we use the same construction to show the necessity and role of a large embedding dimension in a transformer. On the negative side, we present a triple detection task, where attention layers in turn have complexity scaling linearly in the input size; as this scenario seems rare in practice, we also present natural variants that can be efficiently solved by attention layers. The proof techniques emphasize the value of communication complexity in the analysis of transformers and related models, and the role of sparse averaging as a prototypical attention task, which even finds use in the analysis of triple detection.</details>

### Resetting the Optimizer in Deep RL: An Empirical Study

**Authors:** Kavosh Asadi, Rasool Fakoor, Shoham Sabach

**<details><summary>Abstract**</summary>

We focus on the task of approximating the optimal value function in deep reinforcement learning. This iterative process is comprised of solving a sequence of optimization problems where the loss function changes per iteration. The common approach to solving this sequence of problems is to employ modern variants of the stochastic gradient descent algorithm such as Adam. These optimizers maintain their own internal parameters such as estimates of the first-order and the second-order moments of the gradient, and update them over time. Therefore, information obtained in previous iterations is used to solve the optimization problem in the current iteration. We demonstrate that this can contaminate the moment estimates because the optimization landscape can change arbitrarily from one iteration to the next one. To hedge against this negative effect, a simple idea is to reset the internal parameters of the optimizer when starting a new iteration. We empirically investigate this resetting idea by employing various optimizers in conjunction with the Rainbow algorithm. We demonstrate that this simple modification significantly improves the performance of deep RL on the Atari benchmark.</details>

### Retaining Beneficial Information from Detrimental Data for Neural Network Repair

**Authors:** Long-Kai Huang, Peilin Zhao, Junzhou Huang, Sinno Pan

**<details><summary>Abstract**</summary>

The performance of deep learning models heavily relies on the quality of the training data. Inadequacies in the training data, such as corrupt input or noisy labels, can lead to the failure of model generalization. Recent studies propose repairing the model by identifying the training samples that contribute to the failure and removing their influence from the model. However, it is important to note that the identified data may contain both beneficial and detrimental information. Simply erasing the information of the identified data from the model can have a negative impact on its performance, especially when accurate data is mistakenly identified as detrimental and removed. To overcome this challenge, we propose a novel approach that leverages the knowledge obtained from a retained clean set. Our method first identifies harmful data by utilizing the clean set, then separates the beneficial and detrimental information within the identified data. Finally, we utilize the extracted beneficial information to enhance the model's performance. Through empirical evaluations, we demonstrate that our method outperforms baseline approaches in both identifying harmful data and rectifying model failures. Particularly in scenarios where identification is challenging and a significant amount of benign data is involved, our method improves performance while the baselines deteriorate due to the erroneous removal of beneficial information.</details>

### Rethinking Gauss-Newton for learning over-parameterized models

**Authors:** Michael Arbel, Romain Menegaux, Pierre Wolinski

**<details><summary>Abstract**</summary>

This work studies the global convergence and implicit bias of Gauss Newton's (GN) when optimizing over-parameterized one-hidden layer networks in the mean-field regime. We first establish a global convergence result for GN in the continuous-time limit exhibiting a faster convergence rate compared to GD due to improved conditioning. We then perform an empirical study on a synthetic regression task to investigate the implicit bias of GN's method.While GN is consistently faster than GD in finding a global optimum, the learned model generalizes well on test data when starting from random initial weights with a small variance and using a small step size to slow down convergence. Specifically, our study shows that such a setting results in a hidden learning phenomenon, where the dynamics are able to recover features with good generalization properties despite the model having sub-optimal training and test performances due to an under-optimized linear layer. This study exhibits a trade-off between the convergence speed of GN and the generalization ability of the learned solution.</details>

### Rethinking Semi-Supervised Imbalanced Node Classification from Bias-Variance Decomposition

**Authors:** Divin Yan, Gengchen Wei, Chen Yang, Shengzhong Zhang, zengfeng Huang

**<details><summary>Abstract**</summary>

This paper introduces a new approach to address the issue of class imbalance in graph neural networks (GNNs) for learning on graph-structured data. Our approach integrates imbalanced node classification and Bias-Variance Decomposition, establishing a theoretical framework that closely relates data imbalance to model variance. We also leverage graph augmentation technique to estimate the variance and design a regularization term to alleviate the impact of imbalance. Exhaustive tests are conducted on multiple benchmarks, including naturally imbalanced datasets and public-split class-imbalanced datasets, demonstrating that our approach outperforms state-of-the-art methods in various imbalanced scenarios. This work provides a novel theoretical perspective for addressing the problem of imbalanced node classification in GNNs.</details>

### Reusable Slotwise Mechanisms

**Authors:** Trang Nguyen, Amin Mansouri, Kanika Madan, Khuong Duy Nguyen, Kartik Ahuja, Dianbo Liu, Yoshua Bengio

**<details><summary>Abstract**</summary>

Agents with the ability to comprehend and reason about the dynamics of objects would be expected to exhibit improved robustness and generalization in novel scenarios. However, achieving this capability necessitates not only an effective scene representation but also an understanding of the mechanisms governing interactions among object subsets. Recent studies have made significant progress in representing scenes using object slots. In this work, we introduce Reusable Slotwise Mechanisms, or RSM, a framework that models object dynamics by leveraging communication among slots along with a modular architecture capable of dynamically selecting reusable mechanisms for predicting the future states of each object slot. Crucially, RSM leverages the Central Contextual Information (CCI), enabling selected mechanisms to access the remaining slots through a bottleneck, effectively allowing for modeling of higher order and complex interactions that might require a sparse subset of objects. Experimental results demonstrate the superior performance of RSM compared to state-of-the-art methods across various future prediction and related downstream tasks, including Visual Question Answering and action planning. Furthermore, we showcase RSM’s Out-of-Distribution generalization ability to handle scenes in intricate scenarios.</details>

### Reversible and irreversible bracket-based dynamics for deep graph neural networks

**Authors:** Anthony Gruber, Kookjin Lee, Nathaniel Trask

**<details><summary>Abstract**</summary>

Recent works have shown that physics-inspired architectures allow the training of deep graph neural networks (GNNs) without oversmoothing. The role of these physics is unclear, however, with successful examples of both reversible (e.g., Hamiltonian) and irreversible (e.g., diffusion) phenomena producing comparable results despite diametrically opposed mechanisms, and further complications arising due to empirical departures from mathematical theory. This work presents a series of novel GNN architectures based upon structure-preserving bracket-based dynamical systems, which are provably guaranteed to either conserve energy or generate positive dissipation with increasing depth. It is shown that the theoretically principled framework employed here allows for inherently explainable constructions, which contextualize departures from theory in current architectures and better elucidate the roles of reversibility and irreversibility in network performance. Code is available at the Github repository \url{https://github.com/natrask/BracketGraphs}.</details>

### Revisit the Power of Vanilla Knowledge Distillation: from Small Scale to Large Scale

**Authors:** Zhiwei Hao, Jianyuan Guo, Kai Han, Han Hu, Chang Xu, Yunhe Wang

**<details><summary>Abstract**</summary>

The tremendous success of large models trained on extensive datasets demonstrates that scale is a key ingredient in achieving superior results. Therefore, the reflection on the rationality of designing knowledge distillation (KD) approaches for limited-capacity architectures solely based on small-scale datasets is now deemed imperative. In this paper, we identify the small data pitfall that presents in previous KD methods, which results in the underestimation of the power of vanilla KD framework on large-scale datasets such as ImageNet-1K. Specifically, we show that employing stronger data augmentation techniques and using larger datasets can directly decrease the gap between vanilla KD and other meticulously designed KD variants. This highlights the necessity of designing and evaluating KD approaches in the context of practical scenarios, casting off the limitations of small-scale datasets. Our investigation of the vanilla KD and its variants in more complex schemes, including stronger training strategies and different model capacities, demonstrates that vanilla KD is elegantly simple but astonishingly effective in large-scale scenarios. Without bells and whistles, we obtain state-of-the-art ResNet-50, ViT-S, and ConvNeXtV2-T models for ImageNet, which achieve 83.1%, 84.3%, and 85.0% top-1 accuracy, respectively. PyTorch code and checkpoints can be found at https://github.com/Hao840/vanillaKD.</details>

### Revisiting Implicit Differentiation for Learning Problems in Optimal Control

**Authors:** Ming Xu, Timothy L. Molloy, Stephen Gould

**<details><summary>Abstract**</summary>

This paper proposes a new method for differentiating through optimal trajectories arising from non-convex, constrained discrete-time optimal control (COC) problems using the implicit function theorem (IFT). Previous works solve a differential Karush-Kuhn-Tucker (KKT) system for the trajectory derivative, and achieve this efficiently by solving an auxiliary Linear Quadratic Regulator (LQR) problem. In contrast, we directly evaluate the matrix equations which arise from applying variable elimination on the Lagrange multiplier terms in the (differential) KKT system. By appropriately accounting for the structure of the terms within the resulting equations, we show that the trajectory derivatives scale linearly with the number of timesteps. Furthermore, our approach allows for easy parallelization, significantly improved scalability with model size, direct computation of vector-Jacobian products and improved numerical stability compared to prior works. As an additional contribution, we unify prior works, addressing claims that computing trajectory derivatives using IFT scales quadratically with the number of timesteps. We evaluate our method on a both synthetic benchmark and four challenging, learning from demonstration benchmarks including a 6-DoF maneuvering quadrotor and 6-DoF rocket powered landing.</details>

### Revisiting Out-of-distribution Robustness in NLP: Benchmarks, Analysis, and LLMs Evaluations

**Authors:** Lifan Yuan, Yangyi Chen, Ganqu Cui, Hongcheng Gao, FangYuan Zou, Xingyi Cheng, Heng Ji, Zhiyuan Liu, Maosong Sun

**<details><summary>Abstract**</summary>

This paper reexamines the research on out-of-distribution (OOD) robustness in the field of NLP. We find that the distribution shift settings in previous studies commonly lack adequate challenges, hindering the accurate evaluation of OOD robustness. To address these issues, we propose a benchmark construction protocol that ensures clear differentiation and challenging distribution shifts. Then we introduceBOSS, a Benchmark suite for Out-of-distribution robustneSS evaluation covering 5 tasks and 20 datasets. Based on BOSS, we conduct a series of experiments on pretrained language models for analysis and evaluation of OOD robustness. First, for vanilla fine-tuning, we examine the relationship between in-distribution (ID) and OOD performance. We identify three typical types that unveil the inner learningmechanism, which could potentially facilitate the forecasting of OOD robustness, correlating with the advancements on ID datasets. Then, we evaluate 5 classic methods on BOSS and find that, despite exhibiting some effectiveness in specific cases, they do not offer significant improvement compared to vanilla fine-tuning. Further, we evaluate 5 LLMs with various adaptation paradigms and find that when sufficient ID data is available, fine-tuning domain-specific models outperform LLMs on ID examples significantly. However, in the case of OOD instances, prioritizing LLMs with in-context learning yields better results. We identify that both fine-tuned small models and LLMs face challenges in effectively addressing downstream tasks. The code is public at https://github.com/lifan-yuan/OOD_NLP.</details>

### Revisiting Scalarization in Multi-Task Learning: A Theoretical Perspective

**Authors:** Yuzheng Hu, Ruicheng Xian, Qilong Wu, Qiuling Fan, Lang Yin, Han Zhao

**<details><summary>Abstract**</summary>

Linear scalarization, i.e., combining all loss functions by a weighted sum, has been the default choice in the literature of multi-task learning (MTL) since its inception. In recent years, there is a surge of interest in developing Specialized Multi-Task Optimizers (SMTOs) that treat MTL as a multi-objective optimization problem. However, it remains open whether there is a fundamental advantage of SMTOs over scalarization. In fact, heated debates exist in the community comparing these two types of algorithms, mostly from an empirical perspective. To approach the above question, in this paper, we revisit scalarization from a theoretical perspective. We focus on linear MTL models and study whether scalarization is capable of fully exploring the Pareto front. Our findings reveal that, in contrast to recent works that claimed empirical advantages of scalarization, scalarization is inherently incapable of full exploration, especially for those Pareto optimal solutions that strike the balanced trade-offs between multiple tasks. More concretely, when the model is under-parametrized, we reveal a multi-surface structure of the feasible region and identify necessary and sufficient conditions for full exploration. This leads to the conclusion that scalarization is in general incapable of tracing out the Pareto front. Our theoretical results partially answer the open questions in Xin et al. (2021), and provide a more intuitive explanation on why scalarization fails beyond non-convexity. We additionally perform experiments on a real-world dataset using both scalarization and state-of-the-art SMTOs. The experimental results not only corroborate our theoretical findings, but also unveil the potential of SMTOs in finding balanced solutions, which cannot be achieved by scalarization.</details>

### Riemannian Projection-free Online Learning

**Authors:** Zihao Hu, Guanghui Wang, Jacob Abernethy

**<details><summary>Abstract**</summary>

The projection operation is a critical component in a wide range of optimization algorithms, such as online gradient descent (OGD), for enforcing constraints and achieving optimal regret bounds. However, it suffers from computational complexity limitations in high-dimensional settings or when dealing with ill-conditioned constraint sets. Projection-free algorithms address this issue by replacing the projection oracle with more efficient optimization subroutines. But to date, these methods have been developed primarily in the Euclidean setting, and while there has been growing interest in optimization on  Riemannian manifolds, there has been essentially no work in trying to utilize projection-free tools here. An apparent issue is that non-trivial affine functions  are generally non-convex in such domains. In this paper, we present methods for obtaining sub-linear regret guarantees in online geodesically convex optimization  on curved spaces for two scenarios: when we have access to (a) a separation oracle or (b) a linear optimization oracle. For geodesically convex losses, and  when a separation oracle is available, our algorithms achieve $O(T^{\frac{1}{2}})$, $O(T^{\frac{3}{4}})$ and $O(T^{\frac{1}{2}})$ adaptive regret guarantees in the full  information setting, the bandit setting with one-point feedback and the bandit setting with two-point feedback, respectively. When a linear optimization oracle is   available, we obtain regret rates of $O(T^{\frac{3}{4}})$ for geodesically convex losses and $O(T^{\frac{2}{3}}\log T)$ for strongly geodesically convex losses.</details>

### Robust Knowledge Transfer in Tiered Reinforcement Learning

**Authors:** Jiawei Huang, Niao He

**<details><summary>Abstract**</summary>

In this paper, we study the Tiered Reinforcement Learning setting, a parallel transfer learning framework, where the goal is to transfer knowledge from the low-tier (source) task to the high-tier (target) task to reduce the exploration risk of the latter while solving the two tasks in parallel. Unlike previous work, we do not assume the low-tier and high-tier tasks share the same dynamics or reward functions, and focus on robust knowledge transfer without prior knowledge on the task similarity. We identify a natural and necessary condition called the ``Optimal Value Dominance'' for our objective. Under this condition, we propose novel online learning algorithms such that, for the high-tier task, it can achieve constant regret on partial states depending on the task similarity and retain near-optimal regret when the two tasks are dissimilar, while for the low-tier task, it can keep near-optimal without making sacrifice. Moreover, we further study the setting with multiple low-tier tasks, and propose a novel transfer source selection mechanism, which can ensemble the information from all low-tier tasks and allow provable benefits on a much larger state-action space.</details>

### Robust Learning with Progressive Data Expansion Against Spurious Correlation

**Authors:** Yihe Deng, Yu Yang, Baharan Mirzasoleiman, Quanquan Gu

**<details><summary>Abstract**</summary>

While deep learning models have shown remarkable performance in various tasks, they are susceptible to learning non-generalizable _spurious features_ rather than the core features that are genuinely correlated to the true label. In this paper, beyond existing analyses of linear models, we theoretically examine the learning process of a two-layer nonlinear convolutional neural network in the presence of spurious features. Our analysis suggests that imbalanced data groups and easily learnable spurious features can lead to the dominance of spurious features during the learning process. In light of this, we propose a new training algorithm called **PDE** that efficiently enhances the model's robustness for a better worst-group performance. PDE begins with a group-balanced subset of training data and progressively expands it to facilitate the learning of the core features. Experiments on synthetic and real-world benchmark datasets confirm the superior performance of our method on models such as ResNets and Transformers. On average, our method achieves a $2.8$ \% improvement in worst-group accuracy compared with the state-of-the-art method, while enjoying up to $10\times$ faster training efficiency.</details>

### Robust Lipschitz Bandits to Adversarial Corruptions

**Authors:** Yue Kang, Cho-Jui Hsieh, Thomas Chun Man Lee

**<details><summary>Abstract**</summary>

Lipschitz bandit is a variant of stochastic bandits that deals with a continuous arm set defined on a metric space, where the reward function is subject to a Lipschitz constraint. In this paper, we introduce a new problem of Lipschitz bandits in the presence of adversarial corruptions where an adaptive adversary corrupts the stochastic rewards up to a total budget $C$. The budget is measured by the sum of corruption levels across the time horizon $T$. We consider both weak and strong adversaries, where the weak adversary is unaware of the current action before the attack, while the strong one can observe it. Our work presents the first line of robust Lipschitz bandit algorithms that can achieve sub-linear regret under both types of adversary, even when the total budget of corruption $C$ is unrevealed to the agent. We provide a lower bound under each type of adversary, and show that our algorithm is optimal under the strong case. Finally, we conduct experiments to illustrate the effectiveness of our algorithms against two classic kinds of attacks.</details>

### Robust Second-Order Nonconvex Optimization and Its Application to Low Rank Matrix Sensing

**Authors:** Shuyao Li, Yu Cheng, Ilias Diakonikolas, Jelena Diakonikolas, Rong Ge, Stephen Wright

**<details><summary>Abstract**</summary>

Finding an approximate second-order stationary point (SOSP) is a well-studied and fundamental problem in stochastic nonconvex optimization with many applications in machine learning.However, this problem is poorly understood in the presence of outliers, limiting the use of existing nonconvex algorithms in adversarial settings.In this paper, we study the problem of finding SOSPs in the strong contamination model, where a constant fraction of datapoints are arbitrarily corrupted.We introduce a general framework for efficiently finding an approximate SOSP with \emph{dimension-independent} accuracy guarantees, using $\widetilde{O}({D^2}/{\epsilon})$ samples where $D$ is the ambient dimension and $\epsilon$ is the fraction of corrupted datapoints.As a concrete application of our framework, we apply it to the problem of low rank matrix sensing, developing efficient and provably robust algorithms that can tolerate corruptions in both the sensing matrices and the measurements.In addition, we establish a Statistical Query lower bound providing evidence that the quadratic dependence on $D$ in the sample complexity is necessary for computationally efficient algorithms.</details>

### Robustness Guarantees for Adversarially Trained Neural Networks

**Authors:** Poorya Mianjy, Raman Arora

**<details><summary>Abstract**</summary>

We study robust adversarial training of two-layer neural networks as a bi-level optimization problem. In particular, for the inner loop that implements the adversarial attack during training using projected gradient descent (PGD), we propose maximizing a \emph{lower bound} on the $0/1$-loss by reflecting a surrogate loss about the origin. This allows us to give a convergence guarantee for the inner-loop PGD attack. Furthermore, assuming the data is linearly separable, we provide precise iteration complexity results for end-to-end adversarial training, which holds for any width and initialization. We provide empirical evidence to support our theoretical results.</details>

### [Oral] Rotating Features for Object Discovery

**Authors:** Sindy Löwe, Phillip Lippe, Francesco Locatello, Max Welling

**Oral Presentation:** Tu, Dec 12, 13:40 -- Oral 2B

**<details><summary>Abstract**</summary>

The binding problem in human cognition, concerning how the brain represents and connects objects within a fixed network of neural connections, remains a subject of intense debate. Most machine learning efforts addressing this issue in an unsupervised setting have focused on slot-based methods, which may be limiting due to their discrete nature and difficulty to express uncertainty. Recently, the Complex AutoEncoder was proposed as an alternative that learns continuous and distributed object-centric representations. However, it is only applicable to simple toy data. In this paper, we present Rotating Features, a generalization of complex-valued features to higher dimensions, and a new evaluation procedure for extracting objects from distributed representations. Additionally, we show the applicability of our approach to pre-trained features. Together, these advancements enable us to scale distributed object-centric representations from simple toy to real-world data. We believe this work advances a new paradigm for addressing the binding problem in machine learning and has the potential to inspire further innovation in the field.</details>

### SALSA VERDE: a machine learning attack on LWE with sparse small secrets

**Authors:** Cathy Li, Emily Wenger, Zeyuan Allen-Zhu, Francois Charton, Kristin E. Lauter

**<details><summary>Abstract**</summary>

Learning with Errors (LWE) is a hard math problem used in post-quantum cryptography. Homomorphic Encryption (HE) schemes rely on the hardness of the LWE problem for their security, and two LWE-based cryptosystems were recently standardized by NIST for digital signatures and key exchange (KEM).  Thus, it is critical to continue assessing the security of LWE and specific parameter choices. For example, HE uses secrets with small entries, and the HE community has considered standardizing small sparse secrets to improve efficiency and functionality.  However, prior work, SALSA and PICANTE, showed that ML attacks can recover sparse binary secrets. Building on these, we propose VERDE, an improved ML attack that can recover sparse binary, ternary, and narrow Gaussian secrets. Using improved preprocessing and secret recovery techniques, VERDE can attack LWE with larger dimensions ($n=512$) and smaller moduli ($\log_2 q=12$ for $n=256$), using less time and power. We propose novel architectures for scaling. Finally, we develop a theory that explains the success of ML LWE attacks.</details>

### SAMoSSA:  Multivariate Singular Spectrum Analysis with Stochastic Autoregressive Noise

**Authors:** Abdullah Alomar, Munther Dahleh, Sean Mann, Devavrat Shah

**<details><summary>Abstract**</summary>

The well-established practice of time series analysis involves estimating deterministic, non-stationary trend and seasonality components followed by learning the residual stochastic, stationary components. Recently, it has been shown that one can learn the deterministic non-stationary components accurately using  multivariate Singular Spectrum Analysis (mSSA) in the absence of a correlated stationary component; meanwhile, in the absence of deterministic non-stationary components, the Autoregressive (AR) stationary component can also be learnt readily, e.g. via Ordinary Least Squares (OLS). However, a theoretical underpinning of multi-stage learning algorithms involving both deterministic and stationary components has been absent in the literature despite its pervasiveness. We resolve this open question by establishing desirable theoretical guarantees for a natural two-stage algorithm, where mSSA is first applied to estimate the non-stationary components despite the presence of a correlated stationary AR component, which is subsequently learned from the residual time series. We provide a finite-sample forecasting consistency bound for the proposed algorithm, SAMoSSA, which is data-driven and thus requires minimal parameter tuning. To establish theoretical guarantees, we overcome three hurdles: (i) we characterize the spectra of Page matrices of stable AR processes, thus extending the analysis of mSSA; (ii) we extend the analysis of AR process identification in the presence of arbitrary bounded perturbations; (iii) we characterize the out-of-sample or forecasting error, as opposed to solely considering model identification. Through representative empirical studies, we validate the superior performance of SAMoSSA compared to existing baselines. Notably, SAMoSSA's ability to account for AR noise structure yields improvements ranging from 5% to 37% across various benchmark datasets.</details>

### [Spotlight] SPAE: Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs

**Authors:** Lijun Yu, Yong Cheng, Zhiruo Wang, Vivek Kumar, Wolfgang Macherey, Yanping Huang, David Ross, Irfan Essa, Yonatan Bisk, Ming-Hsuan Yang, Kevin Murphy, Alexander Hauptmann, Lu Jiang

**<details><summary>Abstract**</summary>

In this work, we introduce Semantic Pyramid AutoEncoder (SPAE) for enabling frozen LLMs to perform both understanding and generation tasks involving non-linguistic modalities such as images or videos. SPAE converts between raw pixels and interpretable lexical tokens (or words) extracted from the LLM's vocabulary. The resulting tokens capture both the rich semantic meaning and the fine-grained details needed for visual reconstruction, effectively translating the visual content into a language comprehensible to the LLM, and empowering it to perform a wide array of multimodal tasks. Our approach is validated through in-context learning experiments with frozen PaLM 2 and GPT 3.5 on a diverse set of image understanding and generation tasks.Our method marks the first successful attempt to enable a frozen LLM to generate image content while surpassing state-of-the-art performance in image understanding tasks, under the same setting, by over 25%.</details>

### SaVeNet: A Scalable Vector Network for Enhanced Molecular Representation Learning

**Authors:** Sarp Aykent, Tian Xia

**<details><summary>Abstract**</summary>

Geometric representation learning of molecules is challenging yet essential for applications in multiple domains. Despite the impressive breakthroughs made by geometric deep learning in various molecular representation learning tasks, effectively capturing complicated geometric features across spatial dimensions is still underexplored due to the significant difficulties in modeling efficient geometric representations and learning the inherent correlation in 3D structural modeling. These include computational inefficiency, underutilization of vectorial embeddings, and limited generalizability to integrate various geometric properties. To address the raised concerns, we introduce an efficient and effective framework, Scalable Vector Network (SaVeNet), designed to accommodate a range of geometric requirements without depending on costly embeddings. In addition, the proposed framework scales effectively with introduced direction noise. Theoretically, we analyze the desired properties (i.e., invariance and equivariant) and framework efficiency of the SaVeNet. Empirically, we conduct a comprehensive series of experiments to evaluate the efficiency and expressiveness of the proposed model. Our efficiency-focused experiments underscore the model's empirical superiority over existing methods. Experimental results on synthetic and real-world datasets demonstrate the expressiveness of our model, which achieves state-of-the-art performance across various tasks within molecular representation learning.</details>

### Sampling from Structured Log-Concave Distributions via a Soft-Threshold Dikin Walk

**Authors:** Oren Mangoubi, Nisheeth K. Vishnoi

**<details><summary>Abstract**</summary>

Given a Lipschitz or smooth convex function $f:K \to \mathbb{R}^d$ for a bounded polytope $K:=${ $\theta \in \mathbb{R}^d: A\theta \leq b$}, where $A\in  \mathbb{R}^{m\times d}$ and $b \in \mathbb{R}^m$, we consider the problem of sampling from the log-concave distribution $\pi(\theta) \propto e^{-f(\theta)}$ constrained to $K$. Interest in this problem derives from its applications to  Bayesian inference and differential privacy. We present  a generalization of the Dikin walk  to this setting that requires at most $O((md + d L^2 R^2) \times md^{\omega-1} \log(\frac{w}{\delta}))$ arithmetic operations to sample from $\pi$ within error $\delta>0$ in the total variation distance from a $w$-warm start. Here $L$ is the Lipschitz constant of $f$, $K$ is contained in a ball of radius $R$ and contains a ball of smaller radius $r$, and $\omega \approx 2.37$ is the matrix-multiplication constant. This improves on the running time of prior works for a range of structured settings important for the aforementioned inference and privacy applications. Technically, we depart from previous Dikin walks by adding a soft-threshold regularizer derived from the Lipschitz or smoothness properties of $f$  to a barrier function for $K$ that allows our version of the Dikin walk to propose updates  that have a high Metropolis acceptance ratio for $f$, while at the same time remaining inside the polytope $K$.</details>

### Saving 100x Storage: Prototype Replay for Reconstructing Training Sample Distribution in Class-Incremental Semantic Segmentation

**Authors:** Jinpeng Chen, Runmin Cong, Yuxuan LUO, Horace Ip, Sam Kwong

**<details><summary>Abstract**</summary>

Existing class-incremental semantic segmentation (CISS) methods mainly tackle catastrophic forgetting and background shift, but often overlook another crucial issue. In CISS, each step focuses on different foreground classes, and the training set for a single step only includes images containing pixels of the current foreground classes, excluding images without them. This leads to an overrepresentation of these foreground classes in the single-step training set, causing the classification biased towards these classes. To address this issue, we present STAR, which preserves the main characteristics of each past class by storing a compact prototype and necessary statistical data, and aligns the class distribution of single-step training samples with the complete dataset by replaying these prototypes and repeating background pixels with appropriate frequency. Compared to the previous works that replay raw images, our method saves over 100 times the storage while achieving better performance. Moreover, STAR incorporates an old-class features maintaining (OCFM) loss, keeping old-class features unchanged while preserving sufficient plasticity for learning new classes. Furthermore, a similarity-aware discriminative (SAD) loss is employed to specifically enhance the feature diversity between similar old-new class pairs. Experiments on two public datasets, Pascal VOC 2012 and ADE20K, reveal that our model surpasses all previous state-of-the-art methods.</details>

### Scale-teaching: Robust Multi-scale Training for Time Series Classification with Noisy Labels

**Authors:** Zhen Liu, ma peitian, Dongliang Chen, Wenbin Pei, Qianli Ma

**<details><summary>Abstract**</summary>

Deep Neural Networks (DNNs) have been criticized because they easily overfit noisy (incorrect) labels. To improve the robustness of DNNs, existing methods for image data regard samples with small training losses as correctly labeled data (small-loss criterion). Nevertheless, time series' discriminative patterns are easily distorted by external noises (i.e., frequency perturbations) during the recording process. This results in training losses of some time series samples that do not meet the small-loss criterion. Therefore, this paper proposes a deep learning paradigm called Scale-teaching to cope with time series noisy labels. Specifically, we design a fine-to-coarse cross-scale fusion mechanism for learning discriminative patterns by utilizing time series at different scales to train multiple DNNs simultaneously. Meanwhile, each network is trained in a cross-teaching manner by using complementary information from different scales to select small-loss samples as clean labels. For unselected large-loss samples, we introduce multi-scale embedding graph learning via label propagation to correct their labels by using selected clean samples. Experiments on multiple benchmark time series datasets demonstrate the superiority of the proposed Scale-teaching paradigm over state-of-the-art methods in terms of effectiveness and robustness.</details>

### ScaleLong: Towards More Stable Training of Diffusion Model via Scaling Network Long Skip Connection

**Authors:** Zhongzhan Huang, Pan Zhou, Shuicheng Yan, Liang Lin

**<details><summary>Abstract**</summary>

In diffusion models, UNet is the most popular network backbone, since its long skip connects (LSCs) to connect distant network blocks can aggregate long-distant information and alleviate vanishing gradient. Unfortunately, UNet often suffers from unstable training in diffusion models which can be alleviated by scaling its LSC coefficients smaller. However, theoretical understandings of the instability of UNet in diffusion models and also the performance improvement of LSC scaling remain absent yet. To solve this issue, we theoretically show that the coefficients of LSCs in UNet have big effects on the stableness of the forward and backward propagation and robustness of UNet. Specifically, the hidden feature and gradient of UNet at any layer can oscillate and their oscillation ranges are actually large which explains the instability of UNet training. Moreover, UNet is also provably sensitive to perturbed input, and predicts an output distant from the desired output, yielding oscillatory loss and thus oscillatory gradient. Besides, we also observe the theoretical benefits of the LSC coefficient scaling of UNet in the stableness of hidden features and gradient and also robustness. Finally,   inspired by our theory, we propose an effective coefficient scaling framework ScaleLong  that scales the coefficients of LSC  in UNet and better improve the  training stability of UNet. Experimental results on CIFAR10, CelebA, ImageNet and COCO show that our methods are superior to stabilize training, and yield about 1.5x training acceleration on different diffusion models with UNet or UViT backbones.</details>

### [Oral] Scaling Data-Constrained Language Models

**Authors:** Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, Colin Raffel

**Oral Presentation:** Tu, Dec 12, 14:10 -- Oral 2A

**<details><summary>Abstract**</summary>

The current trend of scaling language models involves increasing both parameter count and training dataset size. Extrapolating this trend suggests that training dataset size may soon be limited by the amount of text data available on the internet. Motivated by this limit, we investigate scaling language models in data-constrained regimes. Specifically, we run a large set of experiments varying the extent of data repetition and compute budget, ranging up to 900 billion training tokens and 9 billion parameter models. We find that with constrained data for a fixed compute budget, training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data. However, with more repetition, the value of adding compute eventually decays to zero. We propose and empirically validate a scaling law for compute optimality that accounts for the decreasing value of repeated tokens and excess parameters. Finally, we experiment with approaches mitigating data scarcity, including augmenting the training dataset with code data or removing commonly used filters. Models and datasets from our 400 training runs are freely available at https://github.com/huggingface/datablations.</details>

### Scaling laws for language encoding models in fMRI

**Authors:** Richard Antonello, Aditya Vaidya, Alexander Huth

**<details><summary>Abstract**</summary>

Representations from transformer-based unidirectional language models are known to be effective at predicting brain responses to natural language. However, most studies comparing language models to brains have used GPT-2 or similarly sized language models. Here we tested whether larger open-source models such as those from the OPT and LLaMA families are better at predicting brain responses recorded using fMRI. Mirroring scaling results from other contexts, we found that brain prediction performance scales logarithmically with model size from 125M to 30B parameter models, with ~15% increased encoding performance as measured by correlation with a held-out test set across 3 subjects. Similar log-linear behavior was observed when scaling the size of the fMRI training set. We also characterized scaling for acoustic encoding models that use HuBERT, WavLM, and Whisper, and we found comparable improvements with model size. A noise ceiling analysis of these large, high-performance encoding models showed that performance is nearing the theoretical maximum for brain areas such as the precuneus and higher auditory cortex. These results suggest that increasing scale in both models and data will yield incredibly effective models of language processing in the brain, enabling better scientific understanding as well as applications such as decoding.</details>

### Scenario Diffusion: Controllable Driving Scenario Generation With Diffusion

**Authors:** Ethan Pronovost, Meghana Reddy Ganesina, Noureldin Hendy, Zeyu Wang, Andres Morales, Kai Wang, Nick Roy

**<details><summary>Abstract**</summary>

Automated creation of synthetic traffic scenarios is a key part of scaling the safety validation of autonomous vehicles (AVs). In this paper, we propose Scenario Diffusion, a novel diffusion-based architecture for generating traffic scenarios that enables controllable scenario generation. We combine latent diffusion, object detection and trajectory regression to generate distributions of synthetic agent poses, orientations and trajectories simultaneously. This distribution is conditioned on the map and sets of tokens describing the desired scenario to provide additional control over the generated scenario. We show that our approach has sufficient expressive capacity to model diverse traffic patterns and generalizes to different geographical regions.</details>

### [Spotlight] Schema-learning and rebinding as mechanisms of in-context learning and emergence

**Authors:** Sivaramakrishnan Swaminathan, Antoine Dedieu, Rajkumar Vasudeva Raju, Murray Shanahan, Miguel Lazaro-Gredilla, Dileep George

**<details><summary>Abstract**</summary>

In-context learning (ICL) is one of the most powerful and most unexpected capabilities to emerge in recent transformer-based large language models (LLMs). Yet the mechanisms that underlie it are poorly understood. In this paper, we demonstrate that comparable ICL capabilities can be acquired by an alternative sequence prediction learning method using clone-structured causal graphs (CSCGs). Moreover, a key property of CSCGs is that, unlike transformer-based LLMs, they are {\em interpretable}, which considerably simplifies the task of explaining how ICL works. Specifically, we show that it uses a combination of (a) learning template (schema) circuits for pattern completion, (b) retrieving relevant templates in a context-sensitive manner, and (c) rebinding of novel tokens to appropriate slots in the templates. We go on to marshall evidence for the hypothesis that similar mechanisms underlie ICL in LLMs. For example, we find that, with CSCGs as with LLMs, different capabilities emerge at different levels of overparameterization, suggesting that overparameterization helps in learning more complex template (schema) circuits. By showing how ICL can be achieved with small models and datasets, we open up a path to novel architectures, and take a vital step towards a more general understanding of the mechanics behind this important capability.</details>

### [Spotlight] Score-based Generative Modeling through Stochastic Evolution Equations in Hilbert Spaces

**Authors:** Sungbin Lim, EUN BI YOON, Taehyun Byun, Taewon Kang, Seungwoo Kim, Kyungjae Lee, Sungjoon Choi

**<details><summary>Abstract**</summary>

Continuous-time score-based generative models consist of a pair of stochastic differential equations (SDEs)—a forward SDE that smoothly transitions data into a noise space and a reverse SDE that incrementally eliminates noise from a Gaussian prior distribution to generate data distribution samples—are intrinsically connected by the time-reversal theory on diffusion processes. In this paper, we investigate the use of stochastic evolution equations in Hilbert spaces, which expand the applicability of SDEs in two aspects: sample space and evolution operator, so they enable encompassing recent variations of diffusion models, such as generating functional data or replacing drift coefficients with image transformation. To this end, we derive a generalized time-reversal formula to build a bridge between probabilistic diffusion models and stochastic evolution equations and propose a score-based generative model called Hilbert Diffusion Model (HDM). Combining with Fourier neural operator, we verify the superiority of HDM for sampling functions from functional datasets with a power of kernel two-sample test of 4.2 on Quadratic, 0.2 on Melbourne, and 3.6 on Gridwatch, which outperforms existing diffusion models formulated in function spaces. Furthermore, the proposed method shows its strength in motion synthesis tasks by utilizing the Wiener process with values in Hilbert space. Finally, our empirical results on image datasets also validate a connection between HDM and diffusion models using heat dissipation, revealing the potential for exploring evolution operators and sample spaces.</details>

### Score-based Source Separation with Applications to Digital Communication Signals

**Authors:** Tejas Jayashankar, Gary C.F. Lee, Alejandro Lancho, Amir Weiss, Yury Polyanskiy, Gregory Wornell

**<details><summary>Abstract**</summary>

We propose a new method for separating superimposed sources using diffusion-based generative models. Our method relies only on separately trained statistical priors of independent sources to establish a new objective function guided by $	extit{maximum a posteriori}$ estimation with an $	extit{$\alpha$-posterior}$, across multiple levels of Gaussian smoothing. Motivated by applications in radio-frequency (RF) systems, we are interested in sources with underlying discrete nature and the recovery of encoded bits from a signal of interest, as measured by the bit error rate (BER). Experimental results with RF mixtures demonstrate that our method results in a BER reduction of 95\% over classical and existing learning-based methods. Our analysis demonstrates that our proposed method yields solutions that asymptotically approach the modes of an underlying discrete distribution. Furthermore, our method can be viewed as a multi-source extension to the recently proposed score distillation sampling scheme, shedding additional light on its use beyond conditional sampling. The project webpage is available at https://alpha-rgs.github.io.</details>

### Segment Anything in 3D with NeRFs

**Authors:** Jiazhong Cen, Zanwei Zhou, Jiemin Fang, chen yang, Wei Shen, Lingxi Xie, Dongsheng Jiang, XIAOPENG ZHANG, Qi Tian

**<details><summary>Abstract**</summary>

Recently, the Segment Anything Model (SAM) emerged as a powerful vision foundation model which is capable to segment anything in 2D images. This paper aims to generalize SAM to segment 3D objects. Rather than replicating the data acquisition and annotation procedure which is costly in 3D, we design an efficient solution, leveraging the Neural Radiance Field (NeRF) as a cheap and off-the-shelf prior that connects multi-view 2D images to the 3D space. We refer to the proposed solution as SA3D, for Segment Anything in 3D. It is only required to provide a manual segmentation prompt (e.g., rough points) for the target object in a single view, which is used to generate its 2D mask in this view with SAM. Next, SA3D alternately performs mask inverse rendering and cross-view self-prompting across various views to iteratively complete the 3D mask of the target object constructed with voxel grids. The former projects the 2D mask obtained by SAM in the current view onto 3D mask with guidance of the density distribution learned by the NeRF; The latter extracts reliable prompts automatically as the input to SAM from the NeRF-rendered 2D mask in another view. We show in experiments that SA3D adapts to various scenes and achieves 3D segmentation within minutes. Our research offers a generic and efficient methodology to lift a 2D vision foundation model to 3D, as long as the 2D model can steadily address promptable segmentation across multiple views.</details>

### Selectivity Drives Productivity: Efficient Dataset Pruning for Enhanced Transfer Learning

**Authors:** Yihua Zhang, Yimeng Zhang, Aochuan Chen, jinghan jia, Jiancheng Liu, Gaowen Liu, Mingyi Hong, Shiyu Chang, Sijia Liu

**<details><summary>Abstract**</summary>

Massive data is often considered essential for deep learning applications, but it also incurs significant computational and infrastructural costs. Therefore, dataset pruning (DP) has emerged as an effective way to improve data efficiency by identifying and removing redundant training samples without sacrificing performance. In this work, we aim to address the problem of DP for transfer learning, i.e., how to prune a source dataset for improved pretraining efficiency and lossless finetuning accuracy on downstream target tasks. To our best knowledge, the problem of DP for transfer learning remains open, as previous studies have primarily addressed DP and transfer learning as separate problems. By contrast, we establish a unified viewpoint to integrate DP with transfer learning and find that existing DP methods are not suitable for the transfer learning paradigm. We then propose two new DP methods, label mapping and feature mapping, for supervised and self-supervised pretraining settings respectively, by revisiting the DP problem through the lens of source-target domain mapping. Furthermore, we demonstrate the effectiveness of our approach on numerous transfer learning tasks. We show that source data classes can be pruned by up to $40\%\sim 80\%$ without sacrificing the downstream performance, resulting in a significant $2\sim 5\times$ speed-up during the pretraining stage. Besides, our proposal exhibits broad applicability and can improve other computationally intensive transfer learning techniques, such as adversarial pretraining.</details>

### Self-Adaptive Motion Tracking against On-body Displacement of Flexible Sensors

**Authors:** Chengxu Zuo, Fang Jiawei, Shihui Guo, Yipeng Qin

**<details><summary>Abstract**</summary>

Flexible sensors are promising for ubiquitous sensing of human status due to their flexibility and easy integration as wearable systems. However, on-body displacement of sensors is inevitable since the device cannot be firmly worn at a fixed position across different sessions. This displacement issue causes complicated patterns and significant challenges to subsequent machine learning algorithms. Our work proposes a novel self-adaptive motion tracking network to address this challenge. Our network consists of three novel components: i) a light-weight learnable Affine Transformation layer whose parameters can be tuned to efficiently adapt to unknown displacements; ii) a Fourier-encoded LSTM network for better pattern identification; iii) a novel sequence discrepancy loss equipped with auxiliary regressors for unsupervised tuning of Affine Transformation parameters.</details>

### Self-Consistent Velocity Matching of Probability Flows

**Authors:** Lingxiao Li, Samuel Hurault, Justin Solomon

**<details><summary>Abstract**</summary>

We present a discretization-free scalable framework for solving a large class of mass-conserving partial differential equations (PDEs), including the time-dependent Fokker-Planck equation and the Wasserstein gradient flow. The main observation is that the time-varying velocity field of the PDE solution needs to be self-consistent: it must satisfy a fixed-point equation involving the probability flow characterized by the same velocity field. Instead of directly minimizing the residual of the fixed-point equation with neural parameterization, we use an iterative formulation with a biased gradient estimator that bypasses significant computational obstacles with strong empirical performance. Compared to existing approaches, our method does not suffer from temporal or spatial discretization, covers a wider range of PDEs, and scales to high dimensions. Experimentally, our method recovers analytical solutions accurately when they are available and achieves superior performance in high dimensions with less training time compared to alternatives.</details>

### Self-Evaluation Guided Beam Search for Reasoning

**Authors:** Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, Michael Xie

**<details><summary>Abstract**</summary>

Breaking down a problem into intermediate steps has demonstrated impressive performance in Large Language Model (LLM) reasoning. However, the growth of the reasoning chain introduces uncertainty and error accumulation, making it challenging to elicit accurate final results. To tackle this challenge of uncertainty in multi-step reasoning, we introduce a stepwise self-evaluation mechanism to guide and calibrate the reasoning process of LLMs. We propose a decoding algorithm integrating the self-evaluation guidance via stochastic beam search. The self-evaluation guidance serves as a better-calibrated automatic criterion, facilitating an efficient search in the reasoning space and resulting in superior prediction quality. Stochastic beam search balances exploitation and exploration of the search space with temperature-controlled randomness. Our approach surpasses the corresponding Codex-backboned baselines in few-shot accuracy by $6.34$%, $9.56$%, and $5.46$% on the GSM8K, AQuA, and StrategyQA benchmarks, respectively. Experiment results with Llama-2 on arithmetic reasoning demonstrate the efficiency of our method in outperforming the baseline methods with comparable computational budgets. Further analysis in multi-step reasoning finds our self-evaluation guidance pinpoints logic failures and leads to higher consistency and robustness. Our code is publicly available at [https://guideddecoding.github.io/](https://guideddecoding.github.io/).</details>

### Self-Supervised Learning of Representations for Space Generates Multi-Modular Grid Cells

**Authors:** Rylan Schaeffer, Mikail Khona, Tzuhsuan Ma, Cristobal Eyzaguirre, Sanmi Koyejo, Ila Fiete

**<details><summary>Abstract**</summary>

To solve the spatial problems of mapping, localization and navigation, the mammalian lineage has developed striking spatial representations. One important spatial representation is the Nobel-prize winning grid cells: neurons that represent self-location, a local and aperiodic quantity, with seemingly bizarre non-local and spatially periodic activity patterns of a few discrete periods. Why has the mammalian lineage learnt this peculiar grid representation? Mathematical analysis suggests that this multi-periodic representation has excellent properties as an algebraic code with high capacity and intrinsic error-correction, but to date, synthesis of multi-modular grid cells in deep recurrent neural networks remains absent. In this work, we begin by identifying key insights from four families of approaches to answering the grid cell question: dynamical systems, coding theory, function optimization and supervised deep learning. We then leverage our insights to propose a new approach that elegantly combines the strengths of all four approaches. Our approach is a self-supervised learning (SSL) framework - including data, data augmentations, loss functions and a network architecture - motivated from a normative perspective, with no access to supervised position information. Without making assumptions about internal or readout representations, we show that multiple grid cell modules can emerge in networks trained on our SSL framework and that the networks generalize significantly beyond their training distribution. This work contains insights for neuroscientists interested in the origins of grid cells as well as machine learning researchers interested in novel SSL frameworks.</details>

### Self-supervised Object-Centric Learning for Videos

**Authors:** Görkay Aydemir, Weidi Xie, Fatma Guney

**<details><summary>Abstract**</summary>

Unsupervised multi-object segmentation has shown impressive results on images by utilizing powerful semantics learned from self-supervised pretraining. An additional modality such as depth or motion is often used to facilitate the segmentation in video sequences. However, the performance improvements observed in synthetic sequences, which rely on the robustness of an additional cue, do not translate to more challenging real-world scenarios. In this paper, we propose the first fully unsupervised method for segmenting multiple objects in real-world sequences. Our object-centric learning framework spatially binds objects to slots on each frame and then relates these slots across frames. From these temporally-aware slots, the training objective is to reconstruct the middle frame in a high-level semantic feature space. We propose a masking strategy by dropping a significant portion of tokens in the feature space for efficiency and regularization. Additionally, we address over-clustering by merging slots based on similarity. Our method can successfully segment multiple instances of complex and high-variety classes in YouTube videos.</details>

### Semantic segmentation of sparse irregular point clouds for leaf/wood discrimination

**Authors:** Yuchen BAI, Jean-Baptiste Durand, Grégoire Vincent, Florence Forbes

**<details><summary>Abstract**</summary>

Lidar (Light Detection and Ranging) has become an essential part of the remote sensing toolbox used for biosphere monitoring. In particular, Lidar provides the opportunity to map forest leaf area with unprecedented accuracy, while leaf area has remained an important source of uncertainty affecting models of gas exchanges between the vegetation and the atmosphere. Unmanned Aerial Vehicles (UAV) are easy to mobilize and therefore allow frequent revisits to track the response of vegetation to climate change. However, miniature sensors embarked on UAVs usually provide point clouds of limited density, which are further affected by a strong decrease in density from top to bottom of the canopy due to progressively stronger occlusion. In such a context, discriminating leaf points from wood points presents a significant challenge due in particular to strong class imbalance and spatially irregular sampling intensity. Here we introduce a neural network model based on the Pointnet ++ architecture which makes use of point geometry only (excluding any spectral information). To cope with local data sparsity, we propose an innovative sampling scheme which strives to preserve local important geometric information. We also propose a loss function adapted to the severe class imbalance. We show that our model outperforms state-of-the-art alternatives on UAV point clouds. We discuss future possible improvements, particularly regarding much denser point clouds acquired from below the canopy.</details>

### Setting the Trap: Capturing and Defeating Backdoors in Pretrained Language Models through Honeypots

**Authors:** Ruixiang Tang, Jiayi Yuan, Yiming Li, Zirui Liu, Rui Chen, Xia Hu

**<details><summary>Abstract**</summary>

In the field of natural language processing, the prevalent approach involves fine-tuning pretrained language models (PLMs) using local samples. Recent research has exposed the susceptibility of PLMs to backdoor attacks, wherein the adversaries can embed malicious prediction behaviors by manipulating a few training samples. In this study, our objective is to develop a backdoor-resistant tuning procedure that yields a backdoor-free model, no matter whether the fine-tuning dataset contains poisoned samples. To this end, we propose and integrate an \emph{honeypot module} into the original PLM, specifically designed to absorb backdoor information exclusively. Our design is motivated by the observation that lower-layer representations in PLMs carry sufficient backdoor features while carrying minimal information about the original tasks. Consequently, we can impose penalties on the information acquired by the honeypot module to inhibit backdoor creation during the fine-tuning process of the stem network. Comprehensive experiments conducted on benchmark datasets substantiate the effectiveness and robustness of our defensive strategy. Notably, these results indicate a substantial reduction in the attack success rate ranging from 10\% to 40\% when compared to prior state-of-the-art methods.</details>

### Sharpness-Aware Minimization Leads to Low-Rank Features

**Authors:** Maksym Andriushchenko, Dara Bahri, Hossein Mobahi, Nicolas Flammarion

**<details><summary>Abstract**</summary>

Sharpness-aware minimization (SAM) is a recently proposed method that minimizes the sharpness of the training loss of a neural network. While its generalization improvement is well-known and is the primary motivation, we uncover an additional intriguing effect of SAM: reduction of the feature rank which happens at different layers of a neural network. We show that this low-rank effect occurs very broadly: for different architectures such as fully-connected networks, convolutional networks, vision transformers and for different objectives such as regression, classification, language-image contrastive training. To better understand this phenomenon, we provide a mechanistic understanding of how low-rank features arise in a simple two-layer network. We observe that a significant number of activations gets entirely pruned by SAM which directly contributes to the rank reduction. We confirm this effect theoretically and check that it can also occur in deep networks, although the overall rank reduction mechanism can be more complex, especially for deep networks with pre-activation skip connections and self-attention layers.</details>

### [Spotlight] Should I Stop or Should I Go: Early Stopping with Heterogeneous Populations

**Authors:** Hammaad Adam, Fan Yin, Huibin Hu, Neil Tenenholtz, Lorin Crawford, Lester Mackey, Allison Koenecke

**<details><summary>Abstract**</summary>

Randomized experiments often need to be stopped prematurely due to the treatment having an unintended harmful effect. Existing methods that determine when to stop an experiment early are typically applied to the data in aggregate and do not account for treatment effect heterogeneity. In this paper, we study the early stopping of experiments for harm on heterogeneous populations. We first establish that current methods often fail to stop experiments when the treatment harms a minority group of participants. We then use causal machine learning to develop CLASH, the first broadly-applicable method for heterogeneous early stopping. We demonstrate CLASH's performance on simulated and real data and show that it yields effective early stopping for both clinical trials and A/B tests.</details>

### Single-Stage Visual Query Localization in Egocentric Videos

**Authors:** Hanwen Jiang, Santhosh Kumar Ramakrishnan, Kristen Grauman

**<details><summary>Abstract**</summary>

Visual Query Localization on long-form egocentric videos requires spatio-temporal search and localization of visually specified objects and is vital to build episodic memory systems. Prior work develops complex multi-stage pipelines that leverage well-established object detection and tracking methods to perform VQL. However, each stage is independently trained and the complexity of the pipeline results in slow inference speeds. We propose VQLoC, a novel single-stage VQL framework that is end-to-end trainable. Our key idea is to first build a holistic understanding of the query-video relationship and then perform spatio-temporal localization in a single shot manner. Specifically, we establish the query-video relationship by jointly considering query-to-frame correspondences between the query and each video frame and frame-to-frame correspondences between nearby video frames. Our experiments demonstrate that our approach outperforms prior VQL methods by $20$% accuracy while obtaining a $10\times$ improvement in inference speed. VQLoC is also the top entry on the Ego4D VQ2D challenge leaderboard.</details>

### Solving Inverse Physics Problems with Score Matching

**Authors:** Benjamin Holzschuh, Simona Vegetti, Nils Thuerey

**<details><summary>Abstract**</summary>

We propose to solve inverse problems involving the temporal evolution of physics systems by leveraging recent advances from diffusion models. Our method moves the system's current state backward in time step by step by combining an approximate inverse physics simulator and a learned correction function. A central insight of our work is that training the learned correction with a single-step loss is equivalent to a score matching objective, while recursively predicting longer parts of the trajectory during training relates to maximum likelihood training of a corresponding probability flow.We highlight the advantages of our algorithm compared to standard denoising score matching and implicit score matching, as well as fully learned baselines for a wide range of inverse physics problems. The resulting inverse solver has excellent accuracy and temporal stability and, in contrast to other learned inverse solvers, allows for sampling the posterior of the solutions. Code and experiments are available at https://github.com/tum-pbs/SMDP.</details>

### Sparse Deep Learning for Time Series Data: Theory and Applications

**Authors:** Mingxuan Zhang, Yan Sun, Faming Liang

**<details><summary>Abstract**</summary>

Sparse deep learning has become a popular technique for improving the performance of deep neural networks in areas such as uncertainty quantification, variable selection, and large-scale network compression. However, most existing research has focused on problems where the observations are independent and identically distributed (i.i.d.), and there has been little work on the problems where the observations are dependent, such as time series data and sequential data in natural language processing. This paper aims to address this gap by studying the theory for sparse deep learning with dependent data. We show that sparse recurrent neural networks (RNNs) can be consistently estimated, and their predictions are asymptotically normally distributed under appropriate assumptions, enabling the prediction uncertainty to be correctly quantified. Our numerical results show that sparse deep learning outperforms state-of-the-art methods, such as conformal predictions, in prediction uncertainty quantification for time series data. Furthermore, our results indicate that the proposed method can consistently identify the autoregressive order for time series data and outperform existing methods in large-scale model compression. Our proposed method has important practical implications in fields such as finance, healthcare, and energy, where both accurate point estimates and prediction uncertainty quantification are of concern.</details>

### Spike-driven Transformer

**Authors:** Man Yao, JiaKui Hu, Zhaokun Zhou, Li Yuan, Yonghong Tian, Bo Xu, Guoqi Li

**<details><summary>Abstract**</summary>

Spiking Neural Networks (SNNs) provide an energy-efficient deep learning option due to their unique spike-based event-driven (i.e., spike-driven) paradigm. In this paper, we incorporate the spike-driven paradigm into Transformer by the proposed Spike-driven Transformer with four unique properties: (1) Event-driven, no calculation is triggered when the input of Transformer is zero; (2) Binary spike communication, all matrix multiplications associated with the spike matrix can be transformed into sparse additions; (3) Self-attention with linear complexity at both token and channel dimensions; (4) The operations between spike-form Query, Key, and Value are mask and addition. Together, there are only sparse addition operations in the Spike-driven Transformer. To this end, we design a novel Spike-Driven Self-Attention (SDSA), which exploits only mask and addition operations without any multiplication, and thus having up to $87.2\times$ lower computation energy than vanilla self-attention. Especially in SDSA, the matrix multiplication between Query, Key, and Value is designed as the mask operation. In addition, we rearrange all residual connections in the vanilla Transformer before the activation functions to ensure that all neurons transmit binary spike signals. It is shown that the Spike-driven Transformer can achieve 77.1\% top-1 accuracy on ImageNet-1K, which is the state-of-the-art result in the SNN field.</details>

### Spuriosity Didn’t Kill the Classifier: Using Invariant Predictions to Harness Spurious Features

**Authors:** Cian Eastwood, Shashank Singh, Andrei L Nicolicioiu, Marin Vlastelica Pogančić, Julius von Kügelgen, Bernhard Schölkopf

**<details><summary>Abstract**</summary>

To avoid failures on out-of-distribution data, recent works have sought to extract features that have an invariant or stable relationship with the label across domains, discarding "spurious" or unstable features whose relationship with the label changes across domains. However, unstable features often carry complementary information that could boost performance if used correctly in the test domain. In this work, we show how this can be done without test-domain labels. In particular, we prove that pseudo-labels based on stable features provide sufficient guidance for doing so, provided that stable and unstable features are conditionally independent given the label. Based on this theoretical insight, we propose Stable Feature Boosting (SFB), an algorithm for: (i) learning a predictor that separates stable and conditionally-independent unstable features; and (ii) using the stable-feature predictions to adapt the unstable-feature predictions in the test domain. Theoretically, we prove that SFB can learn an asymptotically-optimal predictor without test-domain labels. Empirically, we demonstrate the effectiveness of SFB on real and synthetic data.</details>

### [Spotlight] Squeeze, Recover and Relabel: Dataset Condensation at ImageNet Scale From A New Perspective

**Authors:** Zeyuan Yin, Eric Xing, Zhiqiang Shen

**<details><summary>Abstract**</summary>

We present a new dataset condensation framework termed Squeeze, Recover and Relabel (SRe$^2$L) that decouples the bilevel optimization of model and synthetic data during training, to handle varying scales of datasets, model architectures and image resolutions for efficient dataset condensation. The proposed method demonstrates flexibility across diverse dataset scales and exhibits multiple advantages in terms of arbitrary resolutions of synthesized images, low training cost and memory consumption with high-resolution synthesis, and the ability to scale up to arbitrary evaluation network architectures. Extensive experiments are conducted on Tiny-ImageNet and full ImageNet-1K datasets. Under 50 IPC, our approach achieves the highest 42.5\% and 60.8\% validation accuracy on Tiny-ImageNet and ImageNet-1K, outperforming all previous state-of-the-art methods by margins of 14.5\% and 32.9\%, respectively. Our approach also surpasses MTT in terms of speed by approximately 52$\times$ (ConvNet-4) and 16$\times$ (ResNet-18) faster with less memory consumption of 11.6$	imes$ and 6.4$	imes$ during data synthesis. Our code and condensed datasets of 50, 200 IPC with 4K recovery budget are available at https://github.com/VILA-Lab/SRe2L.</details>

### Stable Vectorization of Multiparameter Persistent Homology using Signed Barcodes as Measures

**Authors:** David Loiseaux, Luis Scoccola, Mathieu Carrière, Magnus Bakke Botnan, Steve OUDOT

**<details><summary>Abstract**</summary>

Persistent homology (PH) provides  topological descriptors for geometric data, such as weighted graphs, which are interpretable, stable to perturbations, and invariant under, e.g., relabeling. Most applications of PH focus on the one-parameter case---where the descriptors summarize the changes in topology of data as it is filtered by a single quantity of interest---and there is now a wide array of methods enabling the use of one-parameter PH descriptors in data science, which rely on the stable vectorization of these descriptors as elements of a Hilbert space. Although the multiparameter PH (MPH) of data that is filtered by several quantities of interest encodes much richer information than its one-parameter counterpart, the scarceness of stability results for MPH descriptors has so far limited the available options for the stable vectorization of MPH. In this paper, we aim to bring together the best of both worlds by showing how the interpretation of signed barcodes---a recent family of MPH descriptors---as signed Radon measures leads to natural extensions of vectorization strategies from one parameter to multiple parameters. The resulting feature vectors are easy to define and to compute, and provably stable. While, as a proof of concept, we focus on simple choices of signed barcodes and vectorizations, we already see notable performance improvements when comparing our feature vectors to state-of-the-art topology-based methods on various types of data.</details>

### Stanford-ORB: A Real-World 3D Object Inverse Rendering Benchmark

**Authors:** Zhengfei Kuang, Yunzhi Zhang, Hong-Xing Yu, Samir Agarwala, Elliott / Shangzhe Wu, Jiajun Wu

**<details><summary>Abstract**</summary>

We introduce Stanford-ORB, a new real-world 3D Object inverse Rendering Benchmark. Recent advances in inverse rendering have enabled a wide range of real-world applications in 3D content generation, moving rapidly from research and commercial use cases to consumer devices. While the results continue to improve, there is no real-world benchmark that can quantitatively assess and compare the performance of various inverse rendering methods. Existing real-world datasets typically only consist of the shape and multi-view images of objects, which are not sufficient for evaluating the quality of material recovery and object relighting. Methods capable of recovering material and lighting often resort to synthetic data for quantitative evaluation, which on the other hand does not guarantee generalization to complex real-world environments. We introduce a new dataset of real-world objects captured under a variety of natural scenes with ground-truth 3D scans, multi-view images, and environment lighting. Using this dataset, we establish the first comprehensive real-world evaluation benchmark for object inverse rendering tasks from in-the-wild scenes, and compare the performance of various existing methods. All data, code, and models can be accessed at https://stanfordorb.github.io/</details>

### Statistical Analysis of Quantum State Learning Process in Quantum Neural Networks

**Authors:** Hao-Kai Zhang, Chenghong Zhu, Mingrui Jing, Xin Wang

**<details><summary>Abstract**</summary>

Quantum neural networks (QNNs) have been a promising framework in pursuing near-term quantum advantage in various fields, where many applications can be viewed as learning a quantum state that encodes useful data. As a quantum analog of probability distribution learning, quantum state learning is theoretically and practically essential in quantum machine learning. In this paper, we develop a no-go theorem for learning an unknown quantum state with QNNs even starting from a high-fidelity initial state. We prove that when the loss value is lower than a critical threshold, the probability of avoiding local minima vanishes exponentially with the qubit count, while only grows polynomially with the circuit depth. The curvature of local minima is concentrated to the quantum Fisher information times a loss-dependent constant, which characterizes the sensibility of the output state with respect to parameters in QNNs. These results hold for any circuit structures, initialization strategies, and work for both fixed ansatzes and adaptive methods. Extensive numerical simulations are performed to validate our theoretical results. Our findings place generic limits on good initial guesses and adaptive methods for improving the learnability and scalability of QNNs, and deepen the understanding of prior information's role in QNNs.</details>

### Statistical and Computational Trade-off in Multi-Agent Multi-Armed Bandits

**Authors:** Filippo Vannella, Alexandre Proutiere, Jaeseong Jeong

**<details><summary>Abstract**</summary>

We study the problem of regret minimization in Multi-Agent Multi-Armed Bandits (MAMABs) where the rewards are defined through a factor graph. We derive an instance-specific regret lower bound and characterize the minimal expected number of times each global action should be explored. Unfortunately, this bound and the corresponding optimal exploration process are obtained by solving a combinatorial optimization problem with a set of variables and constraints exponentially growing with the number of agents. We approximate the regret lower bound problem via Mean Field techniques to reduce the number of variables and constraints. By tuning the latter, we explore the trade-off between achievable regret and complexity. We devise Efficient Sampling for MAMAB (ESM), an algorithm whose regret asymptotically matches the corresponding approximated lower bound. We assess the regret and computational complexity of ESM numerically, using both synthetic and real-world experiments in radio communications networks.</details>

### Strategic Classification under Unknown Personalized Manipulation

**Authors:** Han Shao, Avrim Blum, Omar Montasser

**<details><summary>Abstract**</summary>

We study the fundamental mistake bound and sample complexity in the strategic classification, where agents can strategically manipulate their feature vector up to an extent in order to be predicted as positive. For example, given a classifier determining college admission, student candidates may try to take easier classes to improve their GPA, retake SAT and change schools in an effort to fool the classifier. *Ball manipulations* are a widely studied class of manipulations in the literature, where agents can modify their feature vector within a bounded radius ball. Unlike most prior work, our work consider manipulations to be *personalized*, meaning that agents can have different levels of manipulation abilities (e.g., varying radii for ball manipulations), and *unknown* to the learner.We formalize the learning problem in an interaction model where the learner first deploys a classifier and the agent manipulates the feature vector within their manipulation set to game the deployed classifier. We investigate various scenarios in terms of the information available to the learner during the interaction, such as observing the original feature vector before or after deployment, observing the manipulated feature vector, or not seeing either the original or the manipulated feature vector. We begin by providing online mistake bounds and PAC sample complexity in these scenarios for ball manipulations. We also explore non-ball manipulations and show that, even in the simplest scenario where both the original and the manipulated feature vectors are revealed, the mistake bounds and sample complexity are lower bounded by $\Omega(|\mathcal H|)$ when the target function belongs to a known class $\mathcal H$.</details>

### Strategic Data Sharing between Competitors

**Authors:** Nikita Tsoy, Nikola Konstantinov

**<details><summary>Abstract**</summary>

Collaborative learning techniques have significantly advanced in recent years, enabling private model training across multiple organizations. Despite this opportunity, firms face a dilemma when considering data sharing with competitors—while collaboration can improve a company’s machine learning model, it may also benefit competitors and hence reduce profits. In this work, we introduce a general framework for analyzing this data-sharing trade-off. The framework consists of three components, representing the firms’ production decisions, the effect of additional data on model quality, and the data-sharing negotiation process, respectively. We then study an instantiation of the framework, based on a conventional market model from economic theory, to identify key factors that affect collaboration incentives. Our findings indicate a profound impact of market conditions on the data-sharing incentives. In particular, we find that reduced competition, in terms of the similarities between the firms’ products, and harder learning tasks foster collaboration.</details>

### StreamNet: Memory-Efficient Streaming Tiny Deep Learning Inference on the Microcontroller

**Authors:** Hong-Sheng Zheng, Yu-Yuan Liu, Chen-Fong Hsu, Tsung Tai Yeh

**<details><summary>Abstract**</summary>

With the emerging Tiny Machine Learning (TinyML) inference applications, there is a growing interest when deploying TinyML models on the low-power Microcontroller Unit (MCU). However, deploying TinyML models on MCUs reveals several challenges due to the MCU’s resource constraints, such as small flash memory, tight SRAM memory budget, and slow CPU performance. Unlike typical layer-wise inference, patch-based inference reduces the peak usage of SRAM memory on MCUs by saving small patches rather than the entire tensor in the SRAM memory. However, the processing of patch-based inference tremendously increases the amount of MACs against the layer-wise method. Thus, this notoriously computational overhead makes patch-based inference undesirable on MCUs. This work designs StreamNet that employs the stream buffer to eliminate the redundant computation of patch-based inference. StreamNet uses 1D and 2D streaming processing and provides an parameter selection algorithm that automatically improve the performance of patch-based inference with minimal requirements on the MCU’s SRAM memory space. In 10 TinyML models, StreamNet-2D achieves a geometric mean of 7.3X speedup and saves 81\% of MACs over the state-of-the-art patch-based inference.</details>

### Strong and Precise Modulation of Human Percepts via Robustified ANNs

**Authors:** Guy Gaziv, Michael Lee, James J DiCarlo

**<details><summary>Abstract**</summary>

The visual object category reports of artificial neural networks (ANNs) are notoriously sensitive to tiny, adversarial image perturbations. Because human category reports (aka human percepts) are thought to be insensitive to those same small-norm perturbations -- and locally stable in general -- this argues that ANNs are incomplete scientific models of human visual perception. Consistent with this, we show that when small-norm image perturbations are generated by standard ANN models, human object category percepts are indeed highly stable.  However, in this very same "human-presumed-stable" regime, we find that robustified ANNs reliably discover low-norm image perturbations that strongly disrupt human percepts. These previously undetectable human perceptual disruptions are massive in amplitude, approaching the same level of sensitivity seen in robustified ANNs.  Further, we show that robustified ANNs support precise perceptual state interventions: they guide the construction of low-norm image perturbations that strongly alter human category percepts toward specific prescribed percepts.  In sum, these contemporary models of biological visual processing are now accurate enough to guide strong and precise interventions on human perception.</details>

### Structure from Duplicates: Neural Inverse Graphics from a Pile of Objects

**Authors:** Tianhang Cheng, Wei-Chiu Ma, Kaiyu Guan, Antonio Torralba, Shenlong Wang

**<details><summary>Abstract**</summary>

Abstract Our world is full of identical objects (\emph{e.g.}, cans of coke, cars of same model). These duplicates, when seen together, provide additional and strong cues for us to effectively reason about 3D. Inspired by this observation, we introduce Structure from Duplicates (SfD), a novel inverse graphics framework that reconstructs geometry, material, and illumination from a single image containing multiple identical objects. SfD begins by identifying multiple instances of an object within an image, and then jointly estimates the 6DoF pose for all instances. An inverse graphics pipeline is subsequently employed to jointly reason about the shape, material of the object, and the environment light, while adhering to the shared geometry and material constraint across instances.Our primary contributions involve utilizing object duplicates as a robust prior for single-image inverse graphics and proposing an in-plane rotation-robust Structure from Motion (SfM) formulation for joint 6-DoF object pose estimation. By leveraging multi-view cues from a single image, SfD generates more realistic and detailed 3D reconstructions, significantly outperforming existing single image reconstruction models and multi-view reconstruction approaches with a similar or greater number of observations.</details>

### Structured Federated Learning through Clustered Additive Modeling

**Authors:** Jie Ma, Tianyi Zhou, Guodong Long, Jing Jiang, Chengqi Zhang

**<details><summary>Abstract**</summary>

Heterogeneous federated learning without assuming any structure is challenging due to the conflicts among non-identical data distributions of clients. In practice, clients often comprise near-homogeneous clusters so training a server-side model per cluster mitigates the conflicts. However, FL with client clustering often suffers from “clustering collapse'', i.e., one cluster's model excels on increasing clients, and reduces to single-model FL. Moreover, cluster-wise models hinder knowledge sharing between clusters and each model depends on fewer clients. Furthermore, the static clustering assumption on data may not hold for dynamically changing models, which are sensitive to cluster imbalance/initialization or outliers. To address these challenges, we propose ''Clustered Additive Modeling  (CAM)'', which applies a globally shared model $\Theta_g$ on top of the cluster-wise models $\Theta_{1:K}$, i.e., $y=h(x;\Theta_g)+f(x;\Theta_k)$ for clients of cluster-$k$. The global model captures the features shared by all clusters so $\Theta_{1:K}$ are enforced to focus on the difference among clusters. To train CAM, we develop a novel Fed-CAM algorithm that alternates between client clustering and training global/cluster models to predict the residual of each other.  We can easily modify any existing clustered FL methods by CAM and significantly improve their performance without ‘’clustering collapse'' in different non-IID settings. We also provide a convergence analysis of Fed-CAM algorithm.</details>

### Structured State Space Models for In-Context Reinforcement Learning

**Authors:** Chris Lu, Yannick Schroecker, Albert Gu, Emilio Parisotto, Jakob Foerster, Satinder Singh, Feryal Behbahani

**<details><summary>Abstract**</summary>

Structured state space sequence (S4) models have recently achieved state-of-the-art performance on long-range sequence modeling tasks. These models also have fast inference speeds and parallelisable training, making them potentially useful in many reinforcement learning settings. We propose a modification to a variant of S4 that enables us to initialise and reset the hidden state in parallel, allowing us to tackle reinforcement learning tasks. We show that our modified architecture runs asymptotically faster than Transformers in sequence length and performs better than RNN's on a simple memory-based task. We evaluate our modified architecture on a set of partially-observable environments and find that, in practice, our model outperforms RNN's while also running over five times faster. Then, by leveraging the model’s ability to handle long-range sequences, we achieve strong performance on a challenging meta-learning task in which the agent is given a randomly-sampled continuous control environment, combined with a randomly-sampled linear projection of the environment's observations and actions. Furthermore, we show the resulting model can adapt to out-of-distribution held-out tasks. Overall, the results presented in this paper show that structured state space models are fast and performant for in-context reinforcement learning tasks. We provide code at https://github.com/luchris429/s5rl.</details>

### Subclass-Dominant Label Noise: A Counterexample for the Success of Early Stopping

**Authors:** Yingbin Bai, Zhongyi Han, Erkun Yang, Jun Yu, Bo Han, Dadong Wang, Tongliang Liu

**<details><summary>Abstract**</summary>

In this paper, we empirically investigate a previously overlooked and widespread type of label noise, subclass-dominant label noise (SDN). Our findings reveal that, during the early stages of training, deep neural networks can rapidly memorize mislabeled examples in SDN. This phenomenon poses challenges in effectively selecting confident examples using conventional early stopping techniques. To address this issue, we delve into the properties of SDN and observe that long-trained representations are superior at capturing the high-level semantics of mislabeled examples, leading to a clustering effect where similar examples are grouped together. Based on this observation, we propose a novel method called NoiseCluster that leverages the geometric structures of long-trained representations to identify and correct SDN. Our experiments demonstrate that NoiseCluster outperforms state-of-the-art baselines on both synthetic and real-world datasets, highlighting the importance of addressing SDN in learning with noisy labels. The code is available at https://github.com/tmllab/2023_NeurIPS_SDN.</details>

### Survival Permanental Processes for Survival Analysis with Time-Varying Covariates

**Authors:** Hideaki Kim

**<details><summary>Abstract**</summary>

Survival or time-to-event data with time-varying covariates are common in practice, and exploring the non-stationarity in covariates is essential to accurately analyzing the nonlinear dependence of time-to-event outcomes on covariates. Traditional survival analysis methods such as Cox proportional hazards model have been extended to address the time-varying covariates through a counting process formulation, although sophisticated machine learning methods that can accommodate time-varying covariates have been limited. In this paper, we propose a non-parametric Bayesian survival model to analyze the nonlinear dependence of time-to-event outcomes on time-varying covariates. We focus on a computationally feasible Cox process called permanental process, which assumes the square root of hazard function to be generated from a Gaussian process, and tailor it for survival data with time-varying covariates. We verify that the proposed model holds with the representer theorem, a beneficial property for functional analysis, which offers us a fast Bayesian estimation algorithm that scales linearly with the number of observed events without relying on Markov Chain Monte Carlo computation. We evaluate our algorithm on synthetic and real-world data, and show that it achieves comparable predictive accuracy while being tens to hundreds of times faster than state-of-the-art methods.</details>

### Symmetry-Informed Geometric Representation for Molecules, Proteins, and Crystalline Materials

**Authors:** Shengchao Liu, weitao Du, Yanjing Li, Zhuoxinran Li, Zhiling Zheng, Chenru Duan, Zhi-Ming Ma, Omar Yaghi, Animashree Anandkumar, Christian Borgs, Jennifer Chayes, Hongyu Guo, Jian Tang

**<details><summary>Abstract**</summary>

Artificial intelligence for scientific discovery has recently generated significant interest within the machine learning and scientific communities, particularly in the domains of chemistry, biology, and material discovery. For these scientific problems, molecules serve as the fundamental building blocks, and machine learning has emerged as a highly effective and powerful tool for modeling their geometric structures. Nevertheless, due to the rapidly evolving process of the field and the knowledge gap between science ({\eg}, physics, chemistry, \& biology) and machine learning communities, a benchmarking study on geometrical representation for such data has not been conducted. To address such an issue, in this paper, we first provide a unified view of the current symmetry-informed geometric methods, classifying them into three main categories: invariance, equivariance with spherical frame basis, and equivariance with vector frame basis. Then we propose a platform, coined Geom3D, which enables benchmarking the effectiveness of geometric strategies. Geom3D contains 16 advanced symmetry-informed geometric representation models and 14 geometric pretraining methods over 52 diverse tasks, including small molecules, proteins, and crystalline materials. We hope that Geom3D can, on the one hand, eliminate barriers for machine learning researchers interested in exploring scientific problems; and, on the other hand, provide valuable guidance for researchers in computational chemistry, structural biology, and materials science, aiding in the informed selection of representation techniques for specific applications. The source code is available on \href{https://github.com/chao1224/Geom3D}{the GitHub repository}.</details>

### Synthetic Experience Replay

**Authors:** Cong Lu, Philip Ball, Yee Whye Teh, Jack Parker-Holder

**<details><summary>Abstract**</summary>

A key theme in the past decade has been that when large neural networks and large datasets combine they can produce remarkable results. In deep reinforcement learning (RL), this paradigm is commonly made possible through experience replay, whereby a dataset of past experiences is used to train a policy or value function. However, unlike in supervised or self-supervised learning, an RL agent has to collect its own data, which is often limited. Thus, it is challenging to reap the benefits of deep learning, and even small neural networks can overfit at the start of training. In this work, we leverage the tremendous recent progress in generative modeling and propose Synthetic Experience Replay (SynthER), a diffusion-based approach to flexibly upsample an agent's collected experience. We show that SynthER is an effective method for training RL agents across offline and online settings, in both proprioceptive and pixel-based environments. In offline settings, we observe drastic improvements when upsampling small offline datasets and see that additional synthetic data also allows us to effectively train larger networks. Furthermore, SynthER enables online agents to train with a much higher update-to-data ratio than before, leading to a significant increase in sample efficiency, without any algorithmic changes. We believe that synthetic training data could open the door to realizing the full potential of deep learning for replay-based RL algorithms from limited data. Finally, we open-source our code at https://github.com/conglu1997/SynthER.</details>

### Systematic Visual Reasoning through Object-Centric Relational Abstraction

**Authors:** Taylor Webb, Shanka Subhra Mondal, Jonathan D Cohen

**<details><summary>Abstract**</summary>

Human visual reasoning is characterized by an ability to identify abstract patterns from only a small number of examples, and to systematically generalize those patterns to novel inputs. This capacity depends in large part on our ability to represent complex visual inputs in terms of both objects and relations. Recent work in computer vision has introduced models with the capacity to extract object-centric representations, leading to the ability to process multi-object visual inputs, but falling short of the systematic generalization displayed by human reasoning. Other recent models have employed inductive biases for relational abstraction to achieve systematic generalization of learned abstract rules, but have generally assumed the presence of object-focused inputs. Here, we combine these two approaches, introducing Object-Centric Relational Abstraction (OCRA), a model that extracts explicit representations of both objects and abstract relations, and achieves strong systematic generalization in tasks (including a novel dataset, CLEVR-ART, with greater visual complexity) involving complex visual displays.</details>

### TMT-VIS: Taxonomy-aware Multi-dataset Joint Training for Video Instance Segmentation

**Authors:** Rongkun Zheng, Lu Qi, Xi Chen, Yi Wang, Kun Wang, Yu Qiao, Hengshuang Zhao

**<details><summary>Abstract**</summary>

Training on large-scale datasets can boost the performance of video instance segmentation while the annotated datasets for VIS are hard to scale up due to the high labor cost. What we possess are numerous isolated filed-specific datasets, thus, it is appealing to jointly train models across the aggregation of datasets to enhance data volume and diversity. However, due to the heterogeneity in category space, as mask precision increase with the data volume, simply utilizing multiple datasets will dilute the attention of models on different taxonomy. Thus, increasing the data scale and enriching taxonomy space while improving classification precision is important. In this work, we analyze that providing extra taxonomy information can help models concentrate on specific taxonomy, and propose our model named Taxonomy-aware Multi-dataset Joint Training for Video Instance Segmentation (TMT-VIS) to address this vital challenge. Specifically, we design a two-stage taxonomy aggregation module that first compiles taxonomy information from input videos and then aggregates these taxonomy priors into instance queries before the transformer decoder. We conduct extensive experimental evaluations on four popular and challenging benchmarks, including YouTube-VIS 2019, YouTube-VIS 2021, OVIS, and UVO. Our model shows significant improvement over the baseline solutions, and sets new state-of-the-art records on all these benchmarks. These appealing and encouraging results demonstrate the effectiveness and generality of our proposed approach. The code and trained models will be publicly available.</details>

### Tame a Wild Camera: In-the-Wild Monocular Camera Calibration

**Authors:** Shengjie Zhu, Abhinav Kumar, Masa Hu, Xiaoming Liu

**<details><summary>Abstract**</summary>

3D sensing for monocular in-the-wild images, e.g., depth estimation and 3D object detection, has become increasingly important.However, the unknown intrinsic parameter hinders their development and deployment.Previous methods for the monocular camera calibration rely on specific 3D objects or strong geometry prior, such as using a checkerboard or imposing a Manhattan World assumption.This work instead calibrates intrinsic via exploiting the monocular 3D prior.Given an undistorted image as input, our method calibrates the complete 4 Degree-of-Freedom (DoF) intrinsic parameters.First, we show intrinsic is determined by the two well-studied monocular priors: monocular depthmap and surface normal map.However, this solution necessitates a low-bias and low-variance depth estimation.Alternatively, we introduce the incidence field, defined as the incidence rays between points in 3D space and pixels in the 2D imaging plane.We show that: 1) The incidence field is a pixel-wise parametrization of the intrinsic invariant to image cropping and resizing.2) The incidence field is a learnable monocular 3D prior, determined pixel-wisely by up-to-sacle monocular depthmap and surface normal.With the estimated incidence field, a robust RANSAC algorithm recovers intrinsic.We show the effectiveness of our method through superior performance on synthetic and zero-shot testing datasets.Beyond calibration, we demonstrate downstream applications in image manipulation detection \& restoration, uncalibrated two-view pose estimation, and 3D sensing.</details>

### Tanh Works Better with Asymmetry

**Authors:** Dongjin Kim, Woojeong Kim, Suhyun Kim

**<details><summary>Abstract**</summary>

Batch Normalization is commonly located in front of activation functions, as proposed by the original paper. Swapping the order, i.e., using Batch Normalization after activation functions, has also been attempted, but its performance is generally not much different from the conventional order when ReLU or a similar activation function is used. However, in the case of bounded activation functions like Tanh, we discovered that the swapped order achieves considerably better performance than the conventional order on various benchmarks and architectures. This paper reports this remarkable phenomenon and closely examines what contributes to this performance improvement. By looking at the output distributions of individual activation functions, not the whole layers, we found that many of them are asymmetrically saturated. The experiments designed to induce a different degree of asymmetric saturation support the hypothesis that asymmetric saturation helps improve performance. In addition, Batch Normalization after bounded activation functions relocates the asymmetrically saturated output of activation functions near zero, enabling the swapped model to have high sparsity, further improving performance. Extensive experiments with Tanh, LeCun Tanh, and Softsign show that the swapped models achieve improved performance with a high degree of asymmetric saturation. Finally, based on this investigation, we test a Tanh function shifted to be asymmetric. This shifted Tanh function that is manipulated to have consistent asymmetry shows even higher accuracy than the original Tanh used in the swapped order, confirming the asymmetry's importance. The code is available at https://github.com/hipros/tanh_works_better_with_asymmetry.</details>

### Tempo Adaptation in Non-stationary Reinforcement Learning

**Authors:** Hyunin Lee, Yuhao Ding, Jongmin Lee, Ming Jin, Javad Lavaei, Somayeh Sojoudi

**<details><summary>Abstract**</summary>

We first raise and tackle a ``time synchronization'' issue between the agent and the environment in non-stationary reinforcement learning (RL), a crucial factor hindering its real-world applications. In reality, environmental changes occur over wall-clock time ($t$) rather than episode progress ($k$), where wall-clock time signifies the actual elapsed time within the fixed duration $t \in [0, T]$. In existing works, at episode $k$, the agent rolls a trajectory and trains a policy before transitioning to episode $k+1$. In the context of the time-desynchronized environment, however, the agent at time $t_{k}$ allocates $\Delta t$ for trajectory generation and training, subsequently moves to the next episode at $t_{k+1}=t_{k}+\Delta t$. Despite a fixed total number of episodes ($K$), the agent accumulates different trajectories influenced by the choice of interaction times ($t_1,t_2,...,t_K$), significantly impacting the suboptimality gap of the policy. We propose a Proactively Synchronizing Tempo ($\texttt{ProST}$) framework that computes a suboptimal sequence {$t_1,t_2,...,t_K$} (= { $t_{1:K}$}) by minimizing an upper bound on its performance measure, i.e., the dynamic regret. Our main contribution is that we show that a suboptimal {$t_{1:K}$} trades-off between the policy training time (agent tempo) and how fast the environment changes (environment tempo). Theoretically, this work develops a suboptimal {$t_{1:K}$} as a function of the degree of the environment's non-stationarity while also achieving a sublinear dynamic regret. Our experimental evaluation on various high-dimensional non-stationary environments shows that the $\texttt{ProST}$ framework achieves a higher online return at suboptimal {$t_{1:K}$} than the existing methods.</details>

### TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular Potentials

**Authors:** Guillem Simeon, Gianni De Fabritiis

**<details><summary>Abstract**</summary>

The development of efficient machine learning models for molecular systems representation is becoming crucial in scientific research. We introduce TensorNet, an innovative O(3)-equivariant message-passing neural network architecture that leverages Cartesian tensor representations. By using Cartesian tensor atomic embeddings, feature mixing is simplified through matrix product operations. Furthermore, the cost-effective decomposition of these tensors into rotation group irreducible representations allows for the separate processing of scalars, vectors, and tensors when necessary. Compared to higher-rank spherical tensor models, TensorNet demonstrates state-of-the-art performance with significantly fewer parameters. For small molecule potential energies, this can be achieved even with a single interaction layer. As a result of all these properties, the model's computational cost is substantially decreased. Moreover, the accurate prediction of vector and tensor molecular quantities on top of potential energies and forces is possible. In summary, TensorNet's framework opens up a new space for the design of state-of-the-art equivariant models.</details>

### Test-time Training for Matching-based Video Object Segmentation

**Authors:** Juliette Bertrand, Giorgos Kordopatis Zilos, Yannis Kalantidis, Giorgos Tolias

**<details><summary>Abstract**</summary>

The video object segmentation (VOS) task involves the segmentation of an object over time based on a single initial mask. Current state-of-the-art approaches use a memory of previously processed frames and rely on matching to estimate segmentation masks of subsequent frames. Lacking any adaptation mechanism, such methods are prone to test-time distribution shifts. This work focuses on matching-based VOS under distribution shifts such as video corruptions, stylization, and sim-to-real transfer. We explore test-time training strategies that are agnostic to the specific task as well as strategies that are designed specifically for VOS. This includes a variant based on mask cycle consistency tailored to matching-based VOS methods. The experimental results on common benchmarks demonstrate that the proposed test-time training yields significant improvements in performance. In particular for the sim-to-real scenario and despite using only a single test video, our approach manages to recover a substantial portion of the performance gain achieved through training on real videos. Additionally, we introduce DAVIS-C, an augmented version of the popular DAVIS test set, featuring extreme distribution shifts like image-/video-level corruptions and stylizations. Our results illustrate that test-time training enhances performance even in these challenging cases.</details>

### TextDiffuser: Diffusion Models as Text Painters

**Authors:** Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, Furu Wei

**<details><summary>Abstract**</summary>

Diffusion models have gained increasing attention for their impressive generation abilities but currently struggle with rendering accurate and coherent text. To address this issue, we introduce TextDiffuser, focusing on generating images with visually appealing text that is coherent with backgrounds. TextDiffuser consists of two stages: first, a Transformer model generates the layout of keywords extracted from text prompts, and then diffusion models generate images conditioned on the text prompt and the generated layout. Additionally, we contribute the first large-scale text images dataset with OCR annotations, MARIO-10M, containing 10 million image-text pairs with text recognition, detection, and character-level segmentation annotations. We further collect the MARIO-Eval benchmark to serve as a comprehensive tool for evaluating text rendering quality. Through experiments and user studies, we demonstrate that TextDiffuser is flexible and controllable to create high-quality text images using text prompts alone or together with text template images, and conduct text inpainting to reconstruct incomplete images with text. We will make the code, model and dataset publicly available.</details>

### The Adversarial Consistency of Surrogate Risks for Binary Classification

**Authors:** Natalie Frank, Jonathan Niles-Weed

**<details><summary>Abstract**</summary>

We study the consistency of surrogate risks for robust binary classification.It is common to learn robust classifiers by adversarial training, which seeks to minimize the expected $0$-$1$ loss when each example can be maliciously corrupted within a small ball.We give a simple and complete characterization of the set of surrogate loss functions that are \emph{consistent}, i.e., that can replace the $0$-$1$ loss without affecting the minimizing sequences of the original adversarial risk, for any data distribution.We also prove a quantitative version of adversarial consistency for the $\rho$-margin loss.Our results reveal that the class of adversarially consistent surrogates is substantially smaller than in the standard setting, where many common surrogates are known to be consistent.</details>

### [Spotlight] The Behavior and Convergence of Local Bayesian Optimization

**Authors:** Kaiwen Wu, Kyurae Kim, Roman Garnett, Jacob Gardner

**<details><summary>Abstract**</summary>

A recent development in Bayesian optimization is the use of local optimization strategies, which can deliver strong empirical performance on high-dimensional problems compared to traditional global strategies. The "folk wisdom" in the literature is that the focus on local optimization sidesteps the curse of dimensionality; however, little is known concretely about the expected behavior or convergence of Bayesian local optimization routines. We first study the behavior of the local approach, and find that the statistics of individual local solutions of Gaussian process sample paths are surprisingly good compared to what we would expect to recover from global methods. We then present the first rigorous analysis of such a Bayesian local optimization algorithm recently proposed by Müller et al. (2021), and derive convergence rates in both the noisy and noiseless settings.</details>

### The Contextual Lasso: Sparse Linear Models via Deep Neural Networks

**Authors:** Ryan Thompson, Amir Dezfouli, robert kohn

**<details><summary>Abstract**</summary>

Sparse linear models are one of several core tools for interpretable machine learning, a field of emerging importance as predictive models permeate decision-making in many domains. Unfortunately, sparse linear models are far less flexible as functions of their input features than black-box models like deep neural networks. With this capability gap in mind, we study a not-uncommon situation where the input features dichotomize into two groups: explanatory features, which are candidates for inclusion as variables in an interpretable model, and contextual features, which select from the candidate variables and determine their effects. This dichotomy leads us to the contextual lasso, a new statistical estimator that fits a sparse linear model to the explanatory features such that the sparsity pattern and coefficients vary as a function of the contextual features. The fitting process learns this function nonparametrically via a deep neural network. To attain sparse coefficients, we train the network with a novel lasso regularizer in the form of a projection layer that maps the network's output onto the space of $\ell_1$-constrained linear models. An extensive suite of experiments on real and synthetic data suggests that the learned models, which remain highly transparent, can be sparser than the regular lasso without sacrificing the predictive power of a standard deep neural network.</details>

### [Spotlight] The Exact Sample Complexity Gain from Invariances for Kernel Regression

**Authors:** Behrooz Tahmasebi, Stefanie Jegelka

**<details><summary>Abstract**</summary>

In practice, encoding invariances into models improves sample complexity. In this work, we study this phenomenon from a theoretical perspective. In particular, we provide minimax optimal rates for kernel ridge regression on compact manifolds, with a target function that is invariant to a group action on the manifold. Our results hold for any smooth compact Lie group action, even groups of positive dimension. For a finite group, the gain effectively multiplies the number of samples by the group size. For groups of positive dimension, the gain is observed by a reduction in the manifold's dimension, in addition to a factor proportional to the volume of the quotient space. Our proof takes the viewpoint of differential geometry, in contrast to the more common strategy of using invariant polynomials. This new geometric viewpoint on learning with invariances may be of independent interest.</details>

### [Spotlight] The Goldilocks of Pragmatic Understanding: Fine-Tuning Strategy Matters for Implicature Resolution by LLMs

**Authors:** Laura Ruis, Akbir Khan, Stella Biderman, Sara Hooker, Tim Rocktäschel, Edward Grefenstette

**<details><summary>Abstract**</summary>

Despite widespread use of LLMs as conversational agents, evaluations of performance fail to capture a crucial aspect of communication: interpreting language in context---incorporating its pragmatics. Humans interpret language using beliefs and prior knowledge about the world. For example, we intuitively understand the response "I wore gloves" to the question "Did you leave fingerprints?" as meaning "No". To investigate whether LLMs have the ability to make this type of inference, known as an implicature, we design a simple task and evaluate four categories of widely used state-of-the-art models. We find that, despite only evaluating on utterances that require a binary inference (yes or no), models in three of these categories perform close to random. However, LLMs instruction-tuned at the example-level perform significantly better. These results suggest that certain fine-tuning strategies are far better at inducing pragmatic understanding in models. We present our findings as the starting point for further research into evaluating how LLMs interpret language in context and to drive the development of more pragmatic and useful models of human discourse.</details>

### The Grand Illusion: The Myth of Software Portability and Implications for ML Progress.

**Authors:** Fraser Mince, Dzung Dinh, Jonas Kgomo, Neil Thompson, Sara Hooker

**<details><summary>Abstract**</summary>

Pushing the boundaries of machine learning often requires exploring different hardware and software combinations. However, this ability to experiment with different systems can be at odds with the drive for efficiency, which has produced increasingly specialized AI hardware and incentivized consolidation around a narrow set of ML frameworks. Exploratory research can be further restricted if software and hardware are co-evolving, making it even harder to stray away from a given tooling stack. While this friction increasingly impacts the rate of innovation in machine learning, to our knowledge the lack of portability in tooling has not been quantified. In this work we ask: How portable are popular ML software frameworks? We conduct a large scale study of the portability of mainstream ML frameworks across different hardware types. Our findings paint an uncomfortable picture -- frameworks can lose more than 40% of their key functions when ported to other hardware. Worse, even when functions are portable, the slowdown in their performance can be extreme. Collectively, our results reveal how costly straying from a narrow set of hardware-software combinations can be - and thus how specialization incurs an exploration cost that can impede innovation in machine learning research.</details>

### [Spotlight] The Pick-to-Learn Algorithm: Empowering Compression for Tight Generalization Bounds and Improved Post-training Performance

**Authors:** Dario Paccagnan, Marco Campi, Simone Garatti

**<details><summary>Abstract**</summary>

Generalization bounds are valuable both for theory and applications. On the one hand, they shed light on the mechanisms that underpin the learning processes; on the other, they certify how well a learned model performs against unseen inputs.  In this work we build upon a recent breakthrough in compression theory to develop a new framework yielding tight generalization bounds of wide practical applicability.  The core idea is to embed any given learning algorithm into a suitably-constructed meta-algorithm (here called Pick-to-Learn, P2L) in order to instill desirable compression properties. When applied to the MNIST classification dataset and to a synthetic regression problem, P2L not only attains generalization bounds that compare favorably with the state of the art (test-set and PAC-Bayes bounds), but it also learns models with better post-training performance.</details>

### The Quantization Model of Neural Scaling

**Authors:** Eric Michaud, Ziming Liu, Uzay Girit, Max Tegmark

**<details><summary>Abstract**</summary>

We propose the Quantization Model of neural scaling laws, explaining both the observed power law dropoff of loss with model and data size, and also the sudden emergence of new capabilities with scale.  We derive this model from what we call the Quantization Hypothesis, where network knowledge and skills are "quantized" into discrete chunks (quanta). We show that when quanta are learned in order of decreasing use frequency, then a power law in use frequencies explains observed power law scaling of loss. We validate this prediction on toy datasets, then study how scaling curves decompose for large language models.  Using language model gradients, we automatically decompose model behavior into a diverse set of skills (quanta). We tentatively find that the frequency at which these quanta are used in the training distribution roughly follows a power law corresponding with the empirical scaling exponent for language models, a prediction of our theory.</details>

### Three Iterations of (d − 1)-WL Test Distinguish Non Isometric Clouds of d-dimensional Points

**Authors:** Valentino Delle Rose, Alexander Kozachinskiy, Cristobal Rojas, Mircea Petrache, Pablo Barceló

**<details><summary>Abstract**</summary>

The Weisfeiler-Lehman (WL) test is a fundamental iterative algorithm for checking the isomorphism of graphs. It has also been observed that it underlies the design of several graph neural network architectures, whose capabilities and performance can be understood in terms of the expressive power of this test. Motivated by recent developments in machine learning applications to datasets involving three-dimensional objects, we study when the WL test is {\em complete} for clouds of Euclidean points represented by complete distance graphs, i.e., when it can distinguish, up to isometry, any arbitrary such cloud. Our main result states that the $(d-1)$-dimensional WL test is complete for point clouds in $d$-dimensional Euclidean space, for any $d\ge 2$, and only three iterations of the test suffice. Our result is tight for $d = 2, 3$. We also observe that the $d$-dimensional WL test only requires one iteration to achieve completeness.</details>

### Time-uniform confidence bands for the CDF under nonstationarity

**Authors:** Paul Mineiro, Steven Howard

**<details><summary>Abstract**</summary>

Estimation of a complete univariate distribution from a sequence of observations is a useful primitive for both manual and automated decision making. This problem has received extensive attention in the i.i.d. setting, but the arbitrary data dependent setting remains largely unaddressed. We present computationally felicitous time-uniform and value-uniform bounds on the CDF of the running averaged conditional distribution of a sequence of real-valued random variables. Consistent with known impossibility results, our CDF bounds are always valid but sometimes trivial when the instance is too hard, and we give an instance-dependent convergence guarantee.  The importance-weighted extension is appropriate for estimating complete counterfactual distributions of rewards given data from a randomized experiment, e.g., from an A/B test or a contextual bandit.</details>

### To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis

**Authors:** Fuzhao Xue, Yao Fu, Wangchunshu Zhou, Zangwei Zheng, Yang You

**<details><summary>Abstract**</summary>

Recent research has highlighted the importance of dataset size in scaling language models. However, large language models (LLMs) are notoriously token-hungry during pre-training, and high-quality text data on the web is likely to be approaching its scaling limit for LLMs. To further enhance LLMs, a straightforward approach is to repeat the pre-training data for additional epochs. In this study, we empirically investigate three key aspects under this approach. First, we explore the consequences of repeating pre-training data, revealing that the model is susceptible to overfitting, leading to multi-epoch degradation. Second, we examine the key factors contributing to multi-epoch degradation, finding that significant factors include dataset size, model parameters, and training objectives, while less influential factors consist of dataset quality and model FLOPs. Finally, we explore whether widely used regularization can alleviate multi-epoch degradation. Most regularization techniques do not yield significant improvements, except for dropout, which demonstrates remarkable effectiveness but requires careful tuning when scaling up the model size. Additionally, we discover that leveraging mixture-of-experts (MoE) enables cost-effective and efficient hyper-parameter tuning for computationally intensive dense LLMs with comparable trainable parameters, potentially impacting efficient LLM development on a broader scale.</details>

### TopP&R: Robust Support Estimation Approach for Evaluating Fidelity and Diversity in Generative Models

**Authors:** Pum Jun Kim, Yoojin Jang, Jisu Kim, Jaejun Yoo

**<details><summary>Abstract**</summary>

We propose a robust and reliable evaluation metric for generative models called Topological Precision and Recall (TopP&R, pronounced “topper”), which systematically estimates supports by retaining only topologically and statistically significant features with a certain level of confidence. Existing metrics, such as Inception Score (IS), Frechet Inception Distance (FID), and various Precision and Recall (P&R) variants, rely heavily on support estimates derived from sample features. However, the reliability of these estimates has been overlooked, even though the quality of the evaluation hinges entirely on their accuracy. In this paper, we demonstrate that current methods not only fail to accurately assess sample quality when support estimation is unreliable, but also yield inconsistent results. In contrast, TopP&R reliably evaluates the sample quality and ensures statistical consistency in its results. Our theoretical and experimental findings reveal that TopP&R provides a robust evaluation, accurately capturing the true trend of change in samples, even in the presence of outliers and non-independent and identically distributed (Non-IID) perturbations where other methods result in inaccurate support estimations. To our knowledge, TopP&R is the first evaluation metric specifically focused on the robust estimation of supports, offering statistical consistency under noise conditions.</details>

### Topological Obstructions and How to Avoid Them

**Authors:** Babak Esmaeili, Robin Walters, Heiko Zimmermann, Jan-Willem van de Meent

**<details><summary>Abstract**</summary>

Incorporating geometric inductive biases into models can aid interpretability and generalization, but encoding to a specific geometric structure can be challenging due to the imposed topological constraints. In this paper, we theoretically and empirically characterize obstructions to training encoders with geometric latent spaces. We show that local optima can arise due to singularities (e.g. self-intersection) or due to an incorrect degree or winding number. We then discuss how normalizing flows can potentially circumvent these obstructions by defining multimodal variational distributions. Inspired by this observation, we propose a new flow-based model that maps data points to multimodal distributions over geometric spaces and empirically evaluate our model on 2 domains. We observe improved stability during training and a higher chance of converging to a homeomorphic encoder.</details>

### Towards A Richer 2D Understanding of Hands at Scale

**Authors:** Tianyi Cheng, Dandan Shan, Ayda Hassen, Richard Higgins, David Fouhey

**<details><summary>Abstract**</summary>

As humans, we learn a lot about how to interact with the world by observing others interacting with their hands. To help AI systems obtain a better understanding of hand interactions, we introduce a new model that produces a rich understanding of hand interaction. Our system produces a richer output than past systems at a larger scale. Our outputs include boxes and segments for hands, in-contact objects, and second objects touched by tools as well as contact and grasp type. Supporting this method are annotations of 257K images, 401K hands, 288K objects, and 19K second objects spanning four datasets. We show that our method provides rich information and performs and generalizes well.</details>

### Towards Accelerated Model Training via Bayesian Data Selection

**Authors:** Zhijie Deng, Peng Cui, Jun Zhu

**<details><summary>Abstract**</summary>

Mislabeled, duplicated, or biased data in real-world scenarios can lead to prolonged training and even hinder model convergence. Traditional solutions prioritizing easy or hard samples lack the flexibility to handle such a variety simultaneously. Recent work has proposed a more reasonable data selection principle by examining the data's impact on the model's generalization loss. However, its practical adoption relies on less principled approximations and additional holdout data. This work solves these problems by leveraging a lightweight Bayesian treatment and incorporating off-the-shelf zero-shot predictors built on large-scale pre-trained models. The resulting algorithm is efficient and easy to implement. We perform extensive empirical studies on challenging benchmarks with considerable data noise and imbalance in the online batch selection scenario, and observe superior training efficiency over competitive baselines. Notably, on the challenging WebVision benchmark, our method can achieve similar predictive performance with significantly fewer training iterations than leading data selection methods.</details>

### Towards Anytime Classification in Early-Exit Architectures by Enforcing Conditional Monotonicity

**Authors:** Metod Jazbec, James Allingham, Dan Zhang, Eric Nalisnick

**<details><summary>Abstract**</summary>

Modern predictive models are often deployed to environments in which computational budgets are dynamic. Anytime algorithms  are well-suited to such environments as, at any point during computation, they can output a prediction whose quality is a function of computation time. Early-exit neural networks have garnered attention in the context of anytime computation due to their capability to provide intermediate predictions at various stages throughout the network. However, we demonstrate that current early-exit networks are not directly applicable to anytime settings, as the quality of predictions for individual data points is not guaranteed to improve with longer computation. To address this shortcoming, we propose an elegant post-hoc modification, based on the Product-of-Experts, that encourages an early-exit network to become gradually confident. This gives our deep models the property of *conditional monotonicity* in the prediction quality---an essential building block towards truly anytime predictive modeling using early-exit architectures. Our empirical results on standard image-classification tasks demonstrate that such behaviors can be achieved while preserving competitive accuracy on average.</details>

### [Spotlight] Towards Automated Circuit Discovery for Mechanistic Interpretability

**Authors:** Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, Adrià Garriga-Alonso

**<details><summary>Abstract**</summary>

Through considerable effort and intuition, several recent works have reverse-engineered nontrivial behaviors oftransformer models. This paper systematizes the mechanistic interpretability process they followed. First, researcherschoose a metric and dataset that elicit the desired model behavior. Then, they apply activation patching to find whichabstract neural network units are involved in the behavior. By varying the dataset, metric, and units underinvestigation, researchers can understand the functionality of each component.We automate one of the process' steps: finding the connections between the abstract neural network units that form a circuit. We propose several algorithms and reproduce previous interpretability results to validate them. Forexample, the ACDC algorithm rediscovered 5/5 of the component types in a circuit in GPT-2 Small that computes theGreater-Than operation. ACDC selected 68 of the 32,000 edges in GPT-2 Small, all of which were manually found byprevious work. Our code is available at https://github.com/ArthurConmy/Automatic-Circuit-Discovery</details>

### Towards Data-Algorithm Dependent Generalization: a Case Study on Overparameterized Linear Regression

**Authors:** Jing Xu, Jiaye Teng, Yang Yuan, Andrew Yao

**<details><summary>Abstract**</summary>

One of the major open problems in machine learning is to characterize generalization in the overparameterized regime, where most traditional generalization bounds become inconsistent even for overparameterized linear regression. In many scenarios, this failure can be attributed to obscuring the crucial interplay between the training algorithm and the underlying data distribution. This paper demonstrate that the generalization behavior of overparameterized model should be analyzed in a both data-relevant and algorithm-relevant manner. To make a formal characterization, We introduce a notion called data-algorithm compatibility, which considers the generalization behavior of the entire data-dependent training trajectory, instead of traditional last-iterate analysis.  We validate our claim by studying the setting of solving overparameterized linear regression with gradient descent. Specifically, we perform a data-dependent trajectory analysis and derive a sufficient condition for compatibility in such a setting. Our theoretical results demonstrate that if we take early stopping iterates into consideration, generalization can hold with significantly weaker restrictions on the problem instance than the previous last-iterate analysis.</details>

### Towards Efficient Image Compression Without Autoregressive Models

**Authors:** Muhammad Salman Ali, Yeongwoong Kim, Maryam Qamar, Sung-Chang Lim, Donghyun Kim, Chaoning Zhang, Sung-Ho Bae, Hui Yong Kim

**<details><summary>Abstract**</summary>

Recently, learned image compression (LIC) has garnered increasing interest with its rapidly improving performance surpassing conventional codecs. A key ingredient of LIC is a hyperprior-based entropy model, where the underlying joint probability of the latent image features is modeled as a product of Gaussian distributions from each latent element. Since latents from the actual images are not spatially independent, autoregressive (AR) context based entropy models were proposed to handle the discrepancy between the assumed distribution and the actual distribution. Though the AR-based models have proven effective, the computational complexity is significantly increased due to the inherent sequential nature of the algorithm. In this paper, we present a novel alternative to the AR-based approach that can provide a significantly better trade-off between performance and complexity. To minimize the discrepancy, we introduce a correlation loss that forces the latents to be spatially decorrelated and better fitted to the independent probability model. Our correlation loss is proved to act as a general plug-in for the hyperprior (HP) based learned image compression methods. The performance gain from our correlation loss is ‘free’ in terms of computation complexity for both inference time and decoding time. To our knowledge, our method  gives the best trade-off between the complexity and performance: combined with the Checkerboard-CM, it attains **90%** and when combined with ChARM-CM, it attains **98%** of the AR-based BD-Rate gains yet is around **50 times** and **30 times** faster than AR-based methods respectively</details>

### Towards Label-free Scene Understanding by Vision Foundation Models

**Authors:** Runnan Chen, Youquan Liu, Lingdong Kong, Nenglun Chen, Xinge ZHU, Yuexin Ma, Tongliang Liu, Wenping Wang

**<details><summary>Abstract**</summary>

Vision foundation models such as Contrastive Vision-Language Pre-training (CLIP) and Segment Anything (SAM) have demonstrated impressive zero-shot performance on image classification and segmentation tasks. However, the incorporation of CLIP and SAM for label-free scene understanding has yet to be explored. In this paper, we investigate the potential of vision foundation models in enabling networks to comprehend 2D and 3D worlds without labelled data. The primary challenge lies in effectively supervising networks under extremely noisy pseudo labels, which are generated by CLIP and further exacerbated during the propagation from the 2D to the 3D domain. To tackle these challenges, we propose a novel Cross-modality Noisy Supervision (CNS) method that leverages the strengths of CLIP and SAM to supervise 2D and 3D networks simultaneously. In particular, we introduce a prediction consistency regularization to co-train 2D and 3D networks, then further impose the networks' latent space consistency using the SAM's robust feature representation. Experiments conducted on diverse indoor and outdoor datasets demonstrate the superior performance of our method in understanding 2D and 3D open environments. Our 2D and 3D network achieves label-free semantic segmentation with 28.4\% and 33.5\% mIoU on ScanNet, improving 4.7\% and 7.9\%, respectively. For nuImages and nuScenes datasets, the performance is 22.1\% and 26.8\% with improvements of 3.5\% and 6.0\%, respectively. Code is available. (https://github.com/runnanchen/Label-Free-Scene-Understanding)</details>

### Towards Stable Backdoor Purification through Feature Shift Tuning

**Authors:** Rui Min, Zeyu Qin, Li Shen, Minhao Cheng

**<details><summary>Abstract**</summary>

It has been widely observed that deep neural networks (DNN) are vulnerable to backdoor attacks where attackers could manipulate the model behavior maliciously by tampering with a small set of training samples. Although a line of defense methods is proposed to mitigate this threat, they either require complicated modifications to the training process or heavily rely on the specific model architecture, which makes them hard to deploy into real-world applications. Therefore, in this paper, we instead start with fine-tuning, one of the most common and easy-to-deploy backdoor defenses, through comprehensive evaluations against diverse attack scenarios. Observations made through initial experiments show that in contrast to the promising defensive results on high poisoning rates, vanilla tuning methods completely fail at low poisoning rate scenarios. Our analysis shows that with the low poisoning rate, the entanglement between backdoor and clean features undermines the effect of tuning-based defenses. Therefore, it is necessary to disentangle the backdoor and clean features in order to improve backdoor purification. To address this, we introduce Feature Shift Tuning (FST), a method for tuning-based backdoor purification. Specifically, FST encourages feature shifts by actively deviating the classifier weights from the originally compromised weights. Extensive experiments demonstrate that our FST provides consistently stable performance under different attack settings. Without complex parameter adjustments, FST also achieves much lower tuning costs, only $10$ epochs. Our codes are available at https://github.com/AISafety-HKUST/stable_backdoor_purification.</details>

### Towards Understanding the Dynamics of Gaussian-Stein Variational Gradient Descent

**Authors:** Tianle Liu, Promit Ghosal, Krishnakumar Balasubramanian, Natesh Pillai

**<details><summary>Abstract**</summary>

Stein Variational Gradient Descent (SVGD) is a nonparametric particle-based deterministic sampling algorithm. Despite its wide usage, understanding the theoretical properties of SVGD has remained a challenging problem. For sampling from a Gaussian target, the SVGD dynamics with a bilinear kernel will remain Gaussian as long as the initializer is Gaussian. Inspired by this fact, we undertake a detailed theoretical study of the Gaussian-SVGD, i.e., SVGD projected to the family of Gaussian distributions via the bilinear kernel, or equivalently Gaussian variational inference (GVI) with SVGD. We present a complete picture by considering both the mean-field PDE and discrete particle systems. When the target is strongly log-concave, the mean-field Gaussian-SVGD dynamics is proven to converge linearly to the Gaussian distribution closest to the target in KL divergence. In the finite-particle setting, there is both uniform in time convergence to the mean-field limit and linear convergence in time to the equilibrium if the target is Gaussian. In the general case, we propose a density-based and a particle-based implementation of the Gaussian-SVGD, and show that several recent algorithms for GVI, proposed from different perspectives, emerge as special cases of our unified framework. Interestingly, one of the new particle-based instance from this framework empirically outperforms existing approaches. Our results make concrete contributions towards obtaining a deeper understanding of both SVGD and GVI.</details>

### TradeMaster: A Holistic Quantitative Trading Platform Empowered by Reinforcement Learning

**Authors:** Shuo Sun, Molei Qin, Wentao Zhang, Haochong Xia, Chuqiao Zong, Jie Ying, Yonggang Xie, Lingxuan Zhao, Xinrun Wang, Bo An

**<details><summary>Abstract**</summary>

The financial markets, which involve over \$90 trillion market capitals, attract the attention of innumerable profit-seeking investors globally. Recent explosion of reinforcement learning in financial trading (RLFT) research has shown stellar performance on many quantitative trading tasks. However, it is still challenging to deploy reinforcement learning (RL) methods into real-world financial markets due to the highly composite nature of this domain, which entails design choices and interactions between components that collect financial data, conduct feature engineering, build market environments, make investment decisions, evaluate model behaviors and offers user interfaces. Despite the availability of abundant financial data and advanced RL techniques, a remarkable gap still exists between the potential and realized utilization of RL in financial trading. In particular, orchestrating an RLFT project lifecycle poses challenges in engineering (i.e. hard to build), benchmarking (i.e. hard to compare) and usability (i.e. hard to optimize, maintain and use). To overcome these challenges, we introduce TradeMaster, a holistic open-source RLFT platform that serves as a i) software toolkit, ii) empirical benchmark, and iii) user interface. Our ultimate goal is to provide infrastructures for transparent and reproducible RLFT research and facilitate their real-world deployment with industry impact. TradeMaster will be updated continuously and welcomes contributions from both RL and finance communities.</details>

### Train 'n Trade: Foundations of Parameter Markets

**Authors:** Tzu-Heng Huang, Harit Vishwakarma, Frederic Sala

**<details><summary>Abstract**</summary>

Organizations typically train large models individually. This is costly and time-consuming, particularly for large-scale foundation models. Such vertical production is known to be suboptimal. Inspired by this economic insight, we ask whether it is possible to leverage others' expertise by trading the constituent parts in models, i.e., sets of weights, as if they were market commodities. While recent advances in aligning and interpolating models suggest that doing so may be possible, a number of fundamental questions must be answered to create viable parameter markets. In this work, we address these basic questions, propose a framework containing the infrastructure necessary for market operations to take place, study strategies for exchanging parameters, and offer means for agents to monetize parameters. Excitingly, compared to agents who train siloed models from scratch, we show that it is possible to mutually gain by using the market, even in competitive settings. This suggests that the notion of parameter markets may be a useful paradigm for improving large-scale model training in the future.</details>

### Training Energy-Based Normalizing Flow with Score-Matching Objectives

**Authors:** Chen-Hao Chao, Wei-Fang Sun, Yen-Chang Hsu, Zsolt Kira, Chun-Yi Lee

**<details><summary>Abstract**</summary>

In this paper, we establish a connection between the parameterization of flow-based and energy-based generative models, and present a new flow-based modeling approach called energy-based normalizing flow (EBFlow). We demonstrate that by optimizing EBFlow with score-matching objectives, the computation of Jacobian determinants for linear transformations can be entirely bypassed. This feature enables the use of arbitrary linear layers in the construction of flow-based models without increasing the computational time complexity of each training iteration from $\mathcal{O}(D^2L)$ to $\mathcal{O}(D^3L)$ for an $L$-layered model that accepts $D$-dimensional inputs. This makes the training of EBFlow more efficient than the commonly-adopted maximum likelihood training method. In addition to the reduction in runtime, we enhance the training stability and empirical performance of EBFlow through a number of techniques developed based on our analysis of the score-matching methods. The experimental results demonstrate that our approach achieves a significant speedup compared to maximum likelihood estimation while outperforming prior methods with a noticeable margin in terms of negative log-likelihood (NLL).</details>

### Training neural operators to preserve invariant measures of chaotic attractors

**Authors:** Ruoxi Jiang, Peter Y. Lu, Elena Orlova, Rebecca Willett

**<details><summary>Abstract**</summary>

Chaotic systems make long-horizon forecasts difficult because small perturbations in initial conditions cause trajectories to diverge at an exponential rate. In this setting, neural operators trained to minimize squared error losses, while capable of accurate short-term forecasts, often fail to reproduce statistical or structural properties of the dynamics over longer time horizons and can yield degenerate results. In this paper, we propose an alternative framework designed to preserve invariant measures of chaotic attractors that characterize the time-invariant statistical properties of the dynamics. Specifically, in the multi-environment setting (where each sample trajectory is governed by slightly different dynamics),  we consider two novel approaches to training with noisy data. First, we propose a loss based on the optimal transport distance between the observed dynamics and the neural operator outputs. This approach requires expert knowledge of the underlying physics to determine what statistical features should be included in the optimal transport loss. Second, we show that a  contrastive learning framework, which does not require any specialized prior knowledge, can preserve statistical properties of the dynamics nearly as well as the optimal transport approach. On a variety of chaotic systems, our method is shown empirically to preserve invariant measures of chaotic attractors.</details>

### Training on Foveated Images Improves Robustness to Adversarial Attacks

**Authors:** Muhammad Shah, Aqsa Kashaf, Bhiksha Raj

**<details><summary>Abstract**</summary>

Deep neural networks (DNNs) have been shown to be vulnerable to adversarial attacks-- subtle,  perceptually indistinguishable perturbations of inputs that change the response of the model. In the context of vision, we hypothesize that an important contributor to the robustness of human visual perception is constant exposure to low-fidelity visual stimuli in our peripheral vision. To investigate this hypothesis, we develop RBlur, an image transform that simulates the loss in fidelity of peripheral vision by blurring the image and reducing its color saturation based on the distance from a given fixation point. We show that compared to DNNs trained on the original images, DNNs trained on images transformed by RBlur are substantially more robust to adversarial attacks, as well as other, non-adversarial, corruptions, achieving up to 25% higher accuracy on perturbed data.</details>

### Transformer-based Planning for Symbolic Regression

**Authors:** Parshin Shojaee, Kazem Meidani, Amir Barati Farimani, Chandan Reddy

**<details><summary>Abstract**</summary>

Symbolic regression (SR) is a challenging task in machine learning that involves finding a mathematical expression for a function based on its values. Recent advancements in SR have demonstrated the effectiveness of pre-trained transformer-based models in generating equations as sequences, leveraging large-scale pre-training on synthetic datasets and offering notable advantages in terms of inference time over classical Genetic Programming (GP) methods. However, these models primarily rely on supervised pre-training goals borrowed from text generation and overlook equation discovery objectives like accuracy and complexity. To address this, we propose TPSR, a Transformer-based Planning strategy for Symbolic Regression that incorporates Monte Carlo Tree Search into the transformer decoding process. Unlike conventional decoding strategies, TPSR enables the integration of non-differentiable feedback, such as fitting accuracy and complexity, as external sources of knowledge into the transformer-based equation generation process. Extensive experiments on various datasets show that our approach outperforms state-of-the-art methods, enhancing the model's fitting-complexity trade-off, extrapolation abilities, and robustness to noise</details>

### Tuning Multi-mode Token-level Prompt Alignment across Modalities

**Authors:** Dongsheng Wang, Miaoge Li, Xinyang Liu, MingSheng Xu, Bo Chen, Hanwang Zhang

**<details><summary>Abstract**</summary>

Advancements in prompt tuning of vision-language models have underscored their potential in enhancing open-world visual concept comprehension. However, prior works only primarily focus on single-mode (only one prompt for each modality) and holistic level (image or sentence) semantic alignment, which fails to capture the sample diversity, leading to sub-optimal prompt discovery. To address the limitation, we propose a multi-mode token-level tuning framework that leverages the optimal transportation to learn and align a set of prompt tokens across modalities. Specifically, we rely on two essential factors: 1) multi-mode prompts discovery, which guarantees diverse semantic representations, and 2) token-level alignment, which helps explore fine-grained similarity. Consequently, the similarity can be calculated as a hierarchical transportation problem between the modality-specific sets. Extensive experiments on popular image recognition benchmarks show the superior generalization and few-shot abilities of our approach. The qualitative analysis demonstrates that the learned prompt tokens have the ability to capture diverse visual concepts.</details>

### Two Heads are Better Than One: A Simple Exploration Framework for Efficient Multi-Agent Reinforcement Learning

**Authors:** Jiahui Li, Kun Kuang, Baoxiang Wang, Xingchen Li, Fei Wu, Jun Xiao, Long Chen

**<details><summary>Abstract**</summary>

Exploration strategy plays an important role in reinforcement learning, especially in sparse-reward tasks. In cooperative multi-agent reinforcement learning~(MARL), designing a suitable exploration strategy is much more challenging due to the large state space and the complex interaction among agents. Currently, mainstream exploration methods in MARL either contribute to exploring the unfamiliar states which are large and sparse, or measuring the interaction among agents with high computational costs. We found an interesting phenomenon that different kinds of exploration plays a different role in different MARL scenarios, and choosing a suitable one is often more effective than designing an exquisite algorithm. In this paper, we propose a exploration method that incorporate the \underline{C}uri\underline{O}sity-based and \underline{IN}fluence-based exploration~(COIN) which is simple but effective in various situations. First, COIN measures the influence of each agent on the other agents based on mutual information theory and designs it as intrinsic rewards which are applied to each individual value function. Moreover, COIN computes the curiosity-based intrinsic rewards via prediction errors which are added to the extrinsic reward. For integrating the two kinds of intrinsic rewards, COIN utilizes a novel framework in which they complement each other and lead to a sufficient and effective exploration on cooperative MARL tasks. We perform extensive experiments on different challenging benchmarks, and results across different scenarios show the superiority of our method.</details>

### Two-Stage Predict+Optimize for MILPs with Unknown Parameters in Constraints

**Authors:** Xinyi Hu, Jasper Lee, Jimmy Lee

**<details><summary>Abstract**</summary>

Consider the setting of constrained optimization, with some parameters unknown at solving time and requiring prediction from relevant features. Predict+Optimize is a recent framework for end-to-end training supervised learning models for such predictions, incorporating information about the optimization problem in the training process in order to yield better predictions in terms of the quality of the predicted solution under the true parameters. Almost all prior works have focused on the special case where the unknowns appear only in the optimization objective and not the constraints. Hu et al. proposed the first adaptation of Predict+Optimize to handle unknowns appearing in constraints, but the framework has somewhat ad-hoc elements, and they provided a training algorithm only for covering and packing linear programs. In this work, we give a new simpler and more powerful framework called Two-Stage Predict+Optimize, which we believe should be the canonical framework for the Predict+Optimize setting. We also give a training algorithm usable for all mixed integer linear programs, vastly generalizing the applicability of the framework. Experimental results demonstrate the superior prediction performance of our training framework over all classical and state-of-the-art methods.</details>

### UltraRE: Enhancing RecEraser for Recommendation Unlearning via Error Decomposition

**Authors:** Yuyuan Li, Chaochao Chen, Yizhao Zhang, Weiming Liu, Lingjuan Lyu, Xiaolin Zheng, Dan Meng, Jun Wang

**<details><summary>Abstract**</summary>

With growing concerns regarding privacy in machine learning models, regulations have committed to granting individuals the right to be forgotten while mandating companies to develop non-discriminatory machine learning systems, thereby fueling the study of the machine unlearning problem. Our attention is directed toward a practical unlearning scenario, i.e., recommendation unlearning. As the state-of-the-art framework, i.e., RecEraser, naturally achieves full unlearning completeness, our objective is to enhance it in terms of model utility and unlearning efficiency. In this paper, we rethink RecEraser from an ensemble-based perspective and focus on its three potential losses, i.e., redundancy, relevance, and combination. Under the theoretical guidance of the above three losses, we propose a new framework named UltraRE, which simplifies and powers RecEraser for recommendation tasks. Specifically, for redundancy loss, we incorporate transport weights in the clustering algorithm to optimize the equilibrium between collaboration and balance while enhancing efficiency; for relevance loss, we ensure that sub-models reach convergence on their respective group data; for combination loss, we simplify the combination estimator without compromising its efficacy. Extensive experiments on three real-world datasets demonstrate the effectiveness of UltraRE.</details>

### Unbiased Compression Saves Communication in Distributed Optimization: When and How Much?

**Authors:** Yutong He, Xinmeng Huang, Kun Yuan

**<details><summary>Abstract**</summary>

Communication compression is a common technique in distributed optimization that can alleviate communication overhead  by transmitting compressed gradients and model parameters. However, compression can introduce information distortion, which slows down convergence and incurs more communication rounds to achieve desired solutions. Given the trade-off between lower per-round communication costs and additional rounds of communication, it is unclear whether communication compression reduces the total communication cost. This paper explores the conditions under which unbiased compression, a widely used form of compression, can reduce the total communication cost, as well as the extent to which it can do so. To this end, we present the first theoretical formulation for characterizing the total communication cost in distributed optimization with communication compression. We demonstrate that unbiased compression alone does not necessarily save the total communication cost, but this outcome can be achieved if the compressors used by all workers are further assumed independent. We establish lower bounds on the communication rounds required by algorithms using independent unbiased compressors to minimize smooth convex functions, and show that these lower bounds are tight by refining the analysis for ADIANA. Our results reveal that using independent unbiased compression can reduce the total communication cost by a factor of up to $\Theta(\\sqrt{\\min\\{n, \\kappa\\}})$, where $n$ is the number of workers and $\\kappa$ is the condition number of the functions being minimized. These theoretical findings are supported by experimental results.</details>

### Unbiased learning of deep generative models with structured discrete representations

**Authors:** Henry C Bendekgey, Gabe Hope, Erik Sudderth

**<details><summary>Abstract**</summary>

By composing graphical models with deep learning architectures, we learn generative models with the strengths of both frameworks. The structured variational autoencoder (SVAE) inherits structure and interpretability from graphical models, and flexible likelihoods for high-dimensional data from deep learning, but poses substantial optimization challenges.  We propose novel algorithms for learning SVAEs, and are the first to demonstrate the SVAE's ability to handle multimodal uncertainty when data is missing by incorporating discrete latent variables.  Our memory-efficient implicit differentiation scheme makes the SVAE tractable to learn via gradient descent, while demonstrating robustness to incomplete optimization. To more rapidly learn accurate graphical model parameters, we derive a method for computing natural gradients without manual derivations, which avoids biases found in prior work.  These optimization innovations enable the first comparisons of the SVAE to state-of-the-art time series models, where the SVAE performs competitively while learning interpretable and structured discrete data representations.</details>

### Uncertainty-Aware Instance Reweighting for Off-Policy Learning

**Authors:** Xiaoying Zhang, Junpu Chen, Hongning Wang, Hong Xie, Yang Liu, John C.S. Lui, Hang Li

**<details><summary>Abstract**</summary>

Off-policy learning, referring to the procedure of policy optimization with access only to logged feedback data, has shown importance in various important real-world applications, such as search engines and recommender systems. While the ground-truth logging policy is usually unknown, previous work simply takes its estimated value for the off-policy learning, ignoring the negative impact from both high bias and high variance resulted from such an estimator. And these impact is often magnified on samples with small and inaccurately estimated logging probabilities. The contribution of this work is to explicitly model the uncertainty in the estimated logging policy, and propose an Uncertainty-aware Inverse Propensity Score estimator (UIPS) for improved off-policy learning, with a theoretical convergence guarantee. Experiment results on the synthetic and real-world recommendation datasets demonstrate that UIPS significantly improves the quality of the discovered policy, when compared against an extensive list of state-of-the-art baselines.</details>

### Uncoupled and Convergent Learning in Two-Player Zero-Sum Markov Games with Bandit Feedback

**Authors:** Yang Cai, Haipeng Luo, Chen-Yu Wei, Weiqiang Zheng

**<details><summary>Abstract**</summary>

We revisit the problem of learning in two-player zero-sum Markov games, focusing on developing an algorithm that is *uncoupled*, *convergent*, and *rational*, with non-asymptotic convergence rates to Nash equilibrium. We start from the case of stateless matrix game with bandit feedback as a warm-up, showing an $\tilde{\mathcal{O}}(t^{-\frac{1}{8}})$ last-iterate convergence rate. To the best of our knowledge, this is the first result that obtains finite last-iterate convergence rate given access to only bandit feedback. We extend our result to the case of irreducible Markov games, providing a last-iterate convergence rate of $\tilde{\mathcal{O}}(t^{-\frac{1}{9+\varepsilon}})$ for any $\varepsilon>0$. Finally, we study Markov games without any assumptions on the dynamics, and show a *path convergence* rate, a new notion of convergence we defined, of $\tilde{\mathcal{O}}(t^{-\frac{1}{10}})$. Our algorithm removes the synchronization and prior knowledge requirement of Wei et al. (2021), which pursued the same goals as us for irreducible Markov games. Our algorithm is related to Chen et al. (2021) and Cen et al. (2021)'s and also builds on the entropy regularization technique. However, we remove their requirement of communications on the entropy values, making our algorithm entirely uncoupled.</details>

### Uncovering and Quantifying Social Biases in Code Generation

**Authors:** Yan Liu, Xiaokang Chen, Yan Gao, Zhe Su, Fengji Zhang, Daoguang Zan, Jian-Guang Lou, Pin-Yu Chen, Tsung-Yi Ho

**<details><summary>Abstract**</summary>

With the popularity of automatic code generation tools, such as Copilot, the study of the potential hazards of these tools is gaining importance. In this work, we explore the social bias problem in pre-trained code generation models. We propose a new paradigm to construct code prompts and successfully uncover social biases in code generation models. To quantify the severity of social biases in generated code, we develop a dataset along with three metrics to evaluate the overall social bias and fine-grained unfairness across different demographics. Experimental results on three pre-trained code generation models (Codex, InCoder, and CodeGen) with varying sizes, reveal severe social biases. Moreover, we conduct analysis to provide useful insights for further choice of code generation models with low social bias.</details>

### Understanding Contrastive Learning via Distributionally Robust Optimization

**Authors:** Junkang Wu, Jiawei Chen, Jiancan Wu, Wentao Shi, Xiang Wang, Xiangnan He

**<details><summary>Abstract**</summary>

This study reveals the inherent tolerance of contrastive learning (CL) towards sampling bias, wherein negative samples may encompass similar semantics (\eg labels). However, existing theories fall short in providing explanations for this phenomenon. We bridge this research gap by analyzing CL through the lens of distributionally robust optimization (DRO), yielding several key insights: (1) CL essentially conducts DRO over the negative sampling distribution, thus enabling robust performance across a variety of potential distributions and demonstrating robustness to sampling bias; (2) The design of the temperature $\tau$ is not merely heuristic but acts as a Lagrange Coefficient, regulating the size of the potential distribution set; (3) A theoretical connection is established between DRO and mutual information, thus presenting fresh evidence for ``InfoNCE as an estimate of MI'' and a new estimation approach for $\phi$-divergence-based generalized mutual information. We also identify CL's potential shortcomings, including over-conservatism and sensitivity to outliers, and introduce a novel Adjusted InfoNCE loss (ADNCE) to mitigate these issues. It refines potential distribution, improving performance and accelerating convergence. Extensive experiments on various domains (image, sentence, and graph) validate the effectiveness of the proposal.</details>

### Understanding, Predicting and Better Resolving Q-Value Divergence in Offline-RL

**Authors:** Yang Yue, Rui Lu, Bingyi Kang, Shiji Song, Gao Huang

**<details><summary>Abstract**</summary>

The divergence of the Q-value estimation has been a prominent issue offline reinforcement learning (offline RL), where the agent has no access to real dynamics. Traditional beliefs attribute this instability to querying out-of-distribution actions when bootstrapping value targets. Though this issue can be alleviated with policy constraints or conservative Q estimation, a theoretical understanding of the underlying mechanism causing the divergence has been absent. In this work, we aim to thoroughly comprehend this mechanism and attain an improved solution. We first identify a fundamental pattern, \emph{self-excitation}, as the primary cause of Q-value estimation divergence in offline RL. Then, we propose a novel Self-Excite Eigenvalue Measure (SEEM) metric based on Neural Tangent Kernel (NTK) to measure the evolving property of Q-network at training, which provides an intriguing explanation of the emergence of divergence. For the first time, our theory can reliably decide whether the training will diverge at an early stage, and even predict the order of the growth for the estimated Q-value, the model's norm, and the crashing step when an SGD optimizer is used. The experiments demonstrate perfect alignment with this theoretic analysis. Building on our insights, we propose to resolve divergence from a novel perspective, namely improving the model's architecture for better extrapolating behavior. Through extensive empirical studies, we identify LayerNorm as a good solution to effectively avoid divergence without introducing detrimental bias, leading to superior performance. Experimental results prove that it can still work in some most challenging settings, i.e. using only 1$\%$ transitions of the dataset, where all previous methods fail. Moreover, it can be easily plugged into modern offline RL methods and achieve SOTA results on many challenging tasks. We also give unique insights into its effectiveness.</details>

### Undirected Probabilistic Model for Tensor Decomposition

**Authors:** Zerui Tao, Toshihisa Tanaka, Qibin Zhao

**<details><summary>Abstract**</summary>

Tensor decompositions (TDs) serve as a powerful tool for analyzing multiway data. Traditional TDs incorporate prior knowledge about the data into the model, such as a directed generative process from latent factors to observations. In practice, selecting proper structural or distributional assumptions beforehand is crucial for obtaining a promising TD representation. However, since such prior knowledge is typically unavailable in real-world applications, choosing an appropriate TD model can be challenging. This paper aims to address this issue by introducing a flexible TD framework that discards the structural and distributional assumptions, in order to learn as much information from the data. Specifically, we construct a TD model that captures the joint probability of the data and latent tensor factors through a deep energy-based model (EBM). Neural networks are then employed to parameterize the joint energy function of tensor factors and tensor entries. The flexibility of EBM and neural networks enables the learning of underlying structures and distributions. In addition, by designing the energy function, our model unifies the learning process of different types of tensors, such as static tensors and dynamic tensors with time stamps. The resulting model presents a doubly intractable nature due to the presence of latent tensor factors and the unnormalized probability function. To efficiently train the model, we derive a variational upper bound of the conditional noise-contrastive estimation objective that learns the unnormalized joint probability by distinguishing data from conditional noises. We show advantages of our model on both synthetic and several real-world datasets.</details>

### [Spotlight] Unexpected Improvements to Expected Improvement for Bayesian Optimization

**Authors:** Sebastian Ament, Samuel Daulton, David Eriksson, Maximilian Balandat, Eytan Bakshy

**<details><summary>Abstract**</summary>

Expected Improvement (EI) is arguably the most popular acquisition function in Bayesian optimization and has found countless successful applications, but its performance is often exceeded by that of more recent methods. Notably, EI and its variants, including for the parallel and multi-objective settings, are challenging to optimize because their acquisition values vanish numerically in many regions. This difficulty generally increases as the number of observations, dimensionality of the search space, or the number of constraints grow, resulting in performance that is inconsistent across the literature and most often sub-optimal. Herein, we propose LogEI, a new family of acquisition functions whose members either have identical or approximately equal optima as their canonical counterparts, but are substantially easier to optimize numerically. We demonstrate that numerical pathologies manifest themselves in “classic” analytic EI, Expected Hypervolume Improvement (EHVI), as well as their constrained, noisy, and parallel variants, and propose corresponding reformulations that remedy these pathologies. Our empirical results show that members of the LogEI family of acquisition functions substantially improve on the optimization performance of their canonical counterparts and surprisingly, are on par with or exceed the performance of recent state-of-the-art acquisition functions, highlighting the understated role of numerical optimization in the literature.</details>

### Uni3DETR: Unified 3D Detection Transformer

**Authors:** Zhenyu Wang, Ya-Li Li, Xi Chen, Hengshuang Zhao, Shengjin Wang

**<details><summary>Abstract**</summary>

Existing point cloud based 3D detectors are designed for the particular scene, either indoor or outdoor ones. Because of the substantial differences in object distribution and point density within point clouds collected from various environments, coupled with the intricate nature of 3D metrics, there is still a lack of a unified network architecture that can accommodate diverse scenes. In this paper, we propose Uni3DETR, a unified 3D detector that addresses indoor and outdoor 3D detection within the same framework. Specifically, we employ the detection transformer with point-voxel interaction for object prediction, which leverages voxel features and points for cross-attention and behaves resistant to the discrepancies from data. We then propose the mixture of query points, which sufficiently exploits global information for dense small-range indoor scenes and local information for large-range sparse outdoor ones. Furthermore, our proposed decoupled IoU provides an easy-to-optimize training target for localization by disentangling the $xy$ and $z$ space. Extensive experiments validate that Uni3DETR exhibits excellent performance consistently on both indoor and outdoor 3D detection. In contrast to previous specialized detectors, which may perform well on some particular datasets but suffer a substantial degradation  on different scenes, Uni3DETR demonstrates the strong generalization ability under heterogeneous conditions (Fig. 1).</details>

### Uniform-in-Time Wasserstein Stability Bounds for (Noisy) Stochastic Gradient Descent

**Authors:** Lingjiong Zhu, Mert Gurbuzbalaban, Anant Raj, Umut Simsekli

**<details><summary>Abstract**</summary>

Algorithmic stability is an important notion that has proven powerful for deriving generalization bounds for practical algorithms. The last decade has witnessed an increasing number of stability bounds for different algorithms applied on different classes of loss functions. While these bounds have illuminated various properties of optimization algorithms, the analysis of each case typically required a different proof technique with significantly different mathematical tools. In this study, we make a novel connection between learning theory and applied probability and introduce a unified guideline for proving Wasserstein stability bounds for stochastic optimization algorithms. We illustrate our approach on stochastic gradient descent (SGD) and we obtain time-uniform  stability bounds (i.e., the bound does not increase with the number of iterations) for strongly convex losses and non-convex losses with additive noise, where we recover similar results to the prior art or extend them to more general cases by using a single proof technique. Our approach is flexible and can be generalizable to other popular optimizers, as it mainly requires developing Lyapunov functions, which are often readily available in the literature. It also illustrates that ergodicity is an important component for obtaining time-uniform bounds --  which might not be achieved for convex or non-convex losses unless additional noise is injected to the iterates. Finally, we slightly stretch our analysis technique and prove time-uniform bounds for SGD under convex and non-convex losses (without additional additive noise), which, to our knowledge, is novel.</details>

### [Spotlight] Unifying Predictions of Deterministic and Stochastic Physics in Mesh-reduced Space with Sequential Flow Generative Model

**Authors:** Luning Sun, Xu Han, Han Gao, Jian-Xun Wang, Liping Liu

**<details><summary>Abstract**</summary>

Accurate prediction of dynamical systems in unstructured meshes has recently shown successes in scientific simulations. Many dynamical systems have a nonnegligible level of stochasticity introduced by various factors (e.g. chaoticity), so there is a need for a unified framework that captures both deterministic and stochastic components in the rollouts of these systems. Inspired by regeneration learning, we propose a new model that combines generative and sequential networks to model dynamical systems. Specifically, we use an autoencoder to learn compact representations of full-space physical variables in a low-dimensional space. We then integrate a transformer with a conditional normalizing flow model to model the temporal sequence of latent representations. We evaluate the new model in both deterministic and stochastic systems. The model outperforms several competitive baseline models and makes more accurate predictions of deterministic systems. Its own prediction error is also reflected in its uncertainty estimations. When predicting stochastic systems, the proposed model generates high-quality rollout samples. The mean and variance of these samples well match the statistics of samples computed from expensive numerical simulations.</details>

### Universal Gradient Descent Ascent Method for Nonconvex-Nonconcave Minimax Optimization

**Authors:** Taoli Zheng, Linglingzhi Zhu, Anthony Man-Cho So, Jose Blanchet, Jiajin Li

**<details><summary>Abstract**</summary>

Nonconvex-nonconcave minimax optimization has received intense attention over the last decade due to its broad applications in machine learning. Most existing algorithms rely on one-sided information, such as the convexity (resp. concavity) of the primal (resp. dual) functions, or other specific structures, such as the Polyak-Łojasiewicz (PŁ) and Kurdyka-Łojasiewicz (KŁ) conditions. However, verifying these regularity conditions is challenging in practice. To meet this challenge, we propose a novel universally applicable single-loop algorithm, the doubly smoothed gradient descent ascent method (DS-GDA), which naturally balances the primal and dual updates. That is, DS-GDA with the same hyperparameters is able to uniformly solve nonconvex-concave, convex-nonconcave, and nonconvex-nonconcave problems with one-sided KŁ properties, achieving convergence with $\mathcal{O}(\epsilon^{-4})$ complexity. Sharper (even optimal) iteration complexity can be obtained when the KŁ exponent is known. Specifically, under the one-sided KŁ condition with exponent $\theta\in(0,1)$, DS-GDA converges with an iteration complexity of $\mathcal{O}(\epsilon^{-2\max\\{2\theta,1\\}})$. They all match the corresponding best results in the literature. Moreover, we show that DS-GDA is practically applicable to general nonconvex-nonconcave problems even without any regularity conditions, such as the PŁ condition, KŁ condition, or weak Minty variational inequalities condition.  For various challenging nonconvex-nonconcave examples in the literature, including *Forsaken*, *Bilinearly-coupled minimax*, *Sixth-order polynomial*, and *PolarGame*, the proposed DS-GDA can all get rid of limit cycles. To the best of our knowledge, this is the first first-order algorithm to achieve convergence on all of these formidable problems.</details>

### [Spotlight] Universal Online Learning with Gradient Variations: A Multi-layer Online Ensemble Approach

**Authors:** Yu-Hu Yan, Peng Zhao, Zhi-Hua Zhou

**<details><summary>Abstract**</summary>

In this paper, we propose an online convex optimization approach with two different levels of adaptivity. On a higher level, our approach is agnostic to the unknown types and curvatures of the online functions, while at a lower level, it can exploit the unknown niceness of the environments and attain problem-dependent guarantees. Specifically, we obtain $\mathcal{O}(\log V_T)$, $\mathcal{O}(d \log V_T)$ and $\hat{\mathcal{O}}(\sqrt{V_T})$ regret bounds for strongly convex, exp-concave and convex loss functions, respectively, where $d$ is the dimension, $V_T$ denotes problem-dependent gradient variations and the $\hat{\mathcal{O}}(\cdot)$-notation omits $\log V_T$ factors. Our result not only safeguards the worst-case guarantees but also directly implies the small-loss bounds in analysis. Moreover, when applied to adversarial/stochastic convex optimization and game theory problems, our result enhances the existing universal guarantees. Our approach is based on a multi-layer online ensemble framework incorporating novel ingredients, including a carefully designed optimism for unifying diverse function types and cascaded corrections for algorithmic stability. Notably, despite its multi-layer structure, our algorithm necessitates only one gradient query per round, making it favorable when the gradient evaluation is time-consuming. This is facilitated by a novel regret decomposition equipped with carefully designed surrogate losses.</details>

### Universal Prompt Tuning for Graph Neural Networks

**Authors:** Taoran Fang, Yunchao Zhang, YANG YANG, Chunping Wang, Lei Chen

**<details><summary>Abstract**</summary>

In recent years, prompt tuning has sparked a research surge in adapting pre-trained models. Unlike the unified pre-training strategy employed in the language field, the graph field exhibits diverse pre-training strategies, posing challenges in designing appropriate prompt-based tuning methods for graph neural networks. While some pioneering work has devised specialized prompting functions for models that employ edge prediction as their pre-training tasks, these methods are limited to specific pre-trained GNN models and lack broader applicability. In this paper, we introduce a universal prompt-based tuning method called Graph Prompt Feature (GPF) for pre-trained GNN models under any pre-training strategy. GPF operates on the input graph's feature space and can theoretically achieve an equivalent effect to any form of prompting function. Consequently, we no longer need to illustrate the prompting function corresponding to each pre-training strategy explicitly. Instead, we employ GPF to obtain the prompted graph for the downstream task in an adaptive manner. We provide rigorous derivations to demonstrate the universality of GPF and make guarantee of its effectiveness. The experimental results under various pre-training strategies indicate that our method performs better than fine-tuning, with an average improvement of about 1.4% in full-shot scenarios and about 3.2% in few-shot scenarios. Moreover, our method significantly outperforms existing specialized prompt-based tuning methods when applied to models utilizing the pre-training strategy they specialize in. These numerous advantages position our method as a compelling alternative to fine-tuning for downstream adaptations.</details>

### Unsupervised Behavior Extraction via Random Intent Priors

**Authors:** Hao Hu, Yiqin Yang, Jianing Ye, Ziqing Mai, Chongjie Zhang

**<details><summary>Abstract**</summary>

Reward-free data is abundant and contains rich prior knowledge of human behaviors, but it is not well exploited by offline reinforcement learning (RL) algorithms. In this paper, we propose UBER, an unsupervised approach to extract useful behaviors from offline reward-free datasets via diversified rewards. UBER assigns different pseudo-rewards sampled from a given prior distribution to different agents to extract a diverse set of behaviors, and reuse them as candidate policies to facilitate the learning of new tasks. Perhaps surprisingly, we show that rewards generated from random neural networks are sufficient to extract diverse and useful behaviors, some even close to expert ones. We provide both empirical and theoretical evidences to justify the use of random priors for the reward function. Experiments on multiple benchmarks showcase UBER's ability to learn effective and diverse behavior sets that enhance sample efficiency for online RL, outperforming existing baselines. By reducing reliance on human supervision, UBER broadens the applicability of RL to real-world scenarios with abundant reward-free data.</details>

### Unsupervised Optical Flow Estimation with Dynamic Timing Representation for Spike Camera

**Authors:** Lujie Xia, Ziluo Ding, Rui Zhao, Jiyuan Zhang, Lei Ma, Zhaofei Yu, Tiejun Huang, Ruiqin Xiong

**<details><summary>Abstract**</summary>

Efficiently selecting an appropriate spike stream data length to extract precise information is the key to the spike vision tasks. To address this issue, we propose a dynamic timing representation for spike streams. Based on multi-layers architecture, it applies dilated convolutions on temporal dimension to extract features on multi-temporal scales with few parameters. And we design layer attention to dynamically fuse these features. Moreover, we propose an unsupervised learning method for optical flow estimation in a spike-based manner to break the dependence on labeled data. In addition, to verify the robustness, we also build a spike-based synthetic validation dataset for extreme scenarios in autonomous driving, denoted as SSES dataset. It consists of various corner cases. Experiments show that our method can predict optical flow from spike streams in different high-speed scenes, including real scenes. For instance, our method achieves $15\%$ and $19\%$ error reduction on PHM dataset compared to the best spike-based work, SCFlow, in $\Delta t=10$ and $\Delta t=20$ respectively, using the same settings as in previous works. The source code and dataset are available at \href{https://github.com/Bosserhead/USFlow}{https://github.com/Bosserhead/USFlow}.</details>

### Unsupervised Protein-Ligand Binding Energy Prediction via Neural Euler's Rotation Equation

**Authors:** Wengong Jin, Siranush Sarkizova, Xun Chen, Nir HaCohen, Caroline Uhler

**<details><summary>Abstract**</summary>

Protein-ligand binding prediction is a fundamental problem in AI-driven drug discovery. Previous work focused on supervised learning methods for small molecules where binding affinity data is abundant, but it is hard to apply the same strategy to other ligand classes like antibodies where labelled data is limited. In this paper, we explore unsupervised approaches and reformulate binding energy prediction as a generative modeling task. Specifically, we train an energy-based model on a set of unlabelled protein-ligand complexes using SE(3) denoising score matching (DSM) and interpret its log-likelihood as binding affinity. Our key contribution is a new equivariant rotation prediction network called Neural Euler's Rotation Equations (NERE) for SE(3) DSM. It predicts a rotation by modeling the force and torque between protein and ligand atoms, where the force is defined as the gradient of an energy function with respect to atom coordinates. Using two protein-ligand and antibody-antigen binding affinity prediction benchmarks, we show that NERE outperforms all unsupervised baselines (physics-based potentials and protein language models) in both cases and surpasses supervised baselines in the antibody case.</details>

### [Oral] User-Level Differential Privacy With Few Examples Per User

**Authors:** Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Raghu Meka, Chiyuan Zhang

**Oral Presentation:** Tu, Dec 12, 14:25 -- Oral 2D

**<details><summary>Abstract**</summary>

Previous work on user-level differential privacy (DP) [Ghazi et al. NeurIPS 2021, Bun et al. STOC 2023] obtained generic algorithms that work for various learning tasks. However, their focus was on the *example-rich* regime, where the users have so many examples that each user could themselves solve the problem. In this work we consider the *example-scarce* regime, where each user has only a few examples, and obtain the following results:* For approximate-DP, we give a generic transformation of any item-level DP algorithm to a user-level DP algorithm. Roughly speaking, the latter gives a (multiplicative) savings of $O_{\varepsilon,\delta}(\sqrt{m})$ in terms of the number of users required for achieving the same utility, where $m$ is the number of examples per user. This algorithm, while recovering most known bounds for specific problems, also gives new bounds, e.g., for PAC learning. * For pure-DP, we present a simple technique for adapting the exponential mechanism [McSherry & Talwar, FOCS 2007] to the user-level setting. This gives new bounds for a variety of tasks, such as private PAC learning, hypothesis selection, and distribution learning. For some of these problems, we show that our bounds are near-optimal.</details>

### V-InFoR: A Robust Graph Neural Networks Explainer for Structurally Corrupted Graphs

**Authors:** Senzhang Wang, Jun Yin, Chaozhuo Li, Xing Xie, Jianxin Wang

**<details><summary>Abstract**</summary>

GNN explanation method aims to identify an explanatory subgraph which contains the most informative components of the full graph. However, a major limitation of existing GNN explainers is that they are not robust to the structurally corrupted graphs, e.g., graphs with noisy or adversarial edges. On the one hand, existing GNN explainers mostly explore explanations based on either the raw graph features or the learned latent representations, both of which can be easily corrupted. On the other hand, the corruptions in graphs are irregular in terms of the structural properties, e.g., the size or connectivity of graphs, which makes the rigorous constraints used by previous GNN explainers unfeasible. To address these issues, we propose a robust GNN explainer called V-InfoR. Specifically, a robust graph representation extractor, which takes insights of variational inference, is proposed to infer the latent distribution of graph representations. Instead of directly using the corrupted raw features or representations of each single graph, we sample the graph representations from the inferred distribution for the downstream explanation generator, which can effectively eliminate the minor corruption. We next formulate the explanation exploration as a graph information bottleneck (GIB) optimization problem. As a more general method that does not need any rigorous structural constraints, our GIB-based method can adaptively capture both the regularity and irregularity of the severely corrupted graphs for explanation. Extensive evaluations on both synthetic and real-world datasets indicate that V-InfoR significantly improves the GNN explanation performance for the structurally corrupted graphs. Code and dataset are available at https://anonymous.4open.science/r/V-InfoR-EF88</details>

### VPGTrans: Transfer Visual Prompt Generator across LLMs

**Authors:** Ao Zhang, Hao Fei, Yuan Yao, Wei Ji, Li Li, Zhiyuan Liu, Tat-Seng Chua

**<details><summary>Abstract**</summary>

Since developing a new multimodal LLM (MLLM) by pre-training on tremendous image-text pairs from scratch can be exceedingly resource-consuming, connecting an existing LLM with a comparatively lightweight visual prompt generator (VPG) becomes a feasible paradigm. However, further tuning the VPG component of the MLLM still incurs significant computational costs, such as thousands of GPU hours and millions of training data points. An alternative solution is transferring an existing VPG from one MLLM to the target MLLM. In this work, we investigate VPG transferability across LLMs for the first time, aiming to reduce the cost of VPG training.  Specifically, we explore VPG transfer across different LLM sizes (e.g., small-to-large) and types.  We identify key factors to maximize transfer efficiency, based on which we develop a simple yet highly effective two-stage transfer framework, called VPGTrans. Notably, it enables VPG transfer from BLIP-2 OPT 2.7B to BLIP-2 OPT 6.7B with less than 10% of the GPU hours using only 10.7% of the training data compared to training a VPG for OPT 6.7B from scratch. Furthermore, we provide a series of intriguing findings and discuss potential explanations behind them. Finally, we showcase the practical value of our VPGTrans approach, by customizing two novel MLLMs, including VL-LLaMA and VL-Vicuna, with recently released LLaMA and Vicuna LLMs.</details>

### Variational Inference with Gaussian Score Matching

**Authors:** Chirag Modi, Robert Gower, Charles Margossian, Yuling Yao, David Blei, Lawrence Saul

**<details><summary>Abstract**</summary>

Variational inference (VI) is a method to approximate the computationally intractable posterior distributions that arise in  Bayesian statistics. Typically, VI fits a simple parametric distribution to be close to the target posterior, optimizing an appropriate objective such as the evidence lower bound (ELBO).   In this work, we present a new approach to VI. Our method is based on the principle of score matching---namely, that if two distributions are equal then their score functions (i.e., gradients of the log density) are equal at every point on their support. With this principle, we develop score-matching VI, an iterative algorithm that seeks to match the scores between the variational approximation and the exact posterior. At each iteration, score-matching VI solves an inner optimization, one that minimally adjusts the current variational estimate to match the scores at a newly sampled value of the latent variables. We show that when the variational family is a  Gaussian, this inner optimization enjoys a closed-form solution, which we call Gaussian score matching VI (GSM-VI). GSM-VI is a ``black box'' variational algorithm in that it only requires a differentiable joint distribution, and as such it can be applied to a wide class of models. We compare GSM-VI to black box variational inference (BBVI), which has similar requirements but instead optimizes the ELBO.   We first study how GSM-VI behaves as a function of the problem dimensionality, the condition number of the target covariance matrix (when the target is Gaussian), and the degree of mismatch between the approximating and exact posterior distribution. We then study GSM-VI on a collection of real-world Bayesian inference problems from the posteriorDB database of datasets and models. We find that GSM-VI is faster than BBVI and equally or more accurate. Specifically, over a wide range of target posteriors, GSM-VI requires 10-100x fewer gradient evaluations than BBVI to obtain a comparable quality of approximation.</details>

### ViSt3D: Video Stylization with 3D CNN

**Authors:** Ayush Pande, Gaurav Sharma

**<details><summary>Abstract**</summary>

Visual stylization has been a very popular research area in recent times. While image stylization has seen a rapid advancement in the recent past, video stylization, while being more challenging, is relatively less explored. The immediate method of stylizing videos by stylizing each frame independently has been tried with some success. To the best of our knowledge, we present the first approach to video stylization using 3D CNN directly, building upon insights from 2D image stylization. Stylizing video is highly challenging, as the appearance and video motion, which includes both camera and subject motions, are inherently entangled in the representations learnt by a 3D CNN. Hence, a naive extension of 2D CNN stylization methods to 3D CNN does not work. To perform stylization with 3D CNN, we propose to explicitly disentangle motion and appearance, stylize the appearance part, and then add back the motion component and decode the final stylized video. In addition, we propose a dataset, curated from existing datasets, to train video stylization networks. We also provide an independently collected test set to study the generalization of video stylization methods. We provide results on this test dataset comparing the proposed method with 2D stylization methods applied frame by frame. We show successful stylization with 3D CNN for the first time, and obtain better stylization in terms of texture cf.\ the existing 2D methods.</details>

### Video Dynamics Prior: An Internal Learning Approach for Robust Video Enhancements

**Authors:** Gaurav Shrivastava, Ser Nam Lim, Abhinav Shrivastava

**<details><summary>Abstract**</summary>

In this paper, we present a novel robust framework for low-level vision tasks, including denoising, object removal, frame interpolation, and super-resolution, that does not require any external training data corpus. Our proposed approach directly learns the weights of neural modules by optimizing over the corrupted test sequence, leveraging the spatio-temporal coherence and internal statistics of videos. Furthermore, we introduce a novel spatial pyramid loss that leverages the property of spatio-temporal patch recurrence in a video across the different scales of the video. This loss enhances robustness to unstructured noise in both the spatial and temporal domains. This further results in our framework being highly robust to degradation in input frames and yields state-of-the-art results on downstream tasks such as denoising, object removal, and frame interpolation. To validate the effectiveness of our approach, we conduct qualitative and quantitative evaluations on standard video datasets such as DAVIS, UCF-101, and VIMEO90K-T.</details>

### Volume Feature Rendering for Fast Neural Radiance Field Reconstruction

**Authors:** Kang Han, Wei Xiang, Lu Yu

**<details><summary>Abstract**</summary>

Neural radiance fields (NeRFs) are able to synthesize realistic novel views from multi-view images captured from distinct positions and perspectives. In NeRF's rendering pipeline, neural networks are used to represent a scene independently or transform queried learnable feature vector of a point to the expected color or density. With the aid of geometry guides either in the form of occupancy grids or proposal networks, the number of color neural network evaluations can be reduced from hundreds to dozens in the standard volume rendering framework. However, many evaluations of the color neural network are still a bottleneck for fast NeRF reconstruction. This paper revisits volume feature rendering (VFR) for the purpose of fast NeRF reconstruction. The VFR integrates the queried feature vectors of a ray into one feature vector, which is then transformed to the final pixel color by a color neural network. This fundamental change to the standard volume rendering framework requires only one single color neural network evaluation to render a pixel, which substantially lowers the high computational complexity of the rendering framework attributed to a large number of color neural network evaluations. Consequently, we can use a comparably larger color neural network to achieve a better rendering quality while maintaining the same training and rendering time costs. This approach achieves the state-of-the-art rendering quality on both synthetic and real-world datasets while requiring less training time compared with existing methods.</details>

### Waypoint Transformer: Reinforcement Learning via Supervised Learning with Intermediate Targets

**Authors:** Anirudhan Badrinath, Yannis Flet-Berliac, Allen Nie, Emma Brunskill

**<details><summary>Abstract**</summary>

Despite the recent advancements in offline reinforcement learning via supervised learning (RvS) and the success of the decision transformer (DT) architecture in various domains, DTs have fallen short in several challenging benchmarks. The root cause of this underperformance lies in their inability to seamlessly connect segments of suboptimal trajectories. To overcome this limitation, we present a novel approach to enhance RvS methods by integrating intermediate targets. We introduce the Waypoint Transformer (WT), using an architecture that builds upon the DT framework and  conditioned on automatically-generated waypoints. The results show a significant increase in the final return compared to existing RvS methods, with performance on par or greater than existing state-of-the-art temporal difference learning-based methods. Additionally, the performance and stability improvements are largest in the most challenging environments and data configurations, including AntMaze Large Play/Diverse and Kitchen Mixed/Partial.</details>

### What Do Deep Saliency Models Learn about Visual Attention?

**Authors:** Shi Chen, Ming Jiang, Qi Zhao

**<details><summary>Abstract**</summary>

In recent years, deep saliency models have made significant progress in predicting human visual attention. However, the mechanisms behind their success remain largely unexplained due to the opaque nature of deep neural networks. In this paper, we present a novel analytic framework that sheds light on the implicit features learned by saliency models and provides principled interpretation and quantification of their contributions to saliency prediction. Our approach decomposes these implicit features into interpretable bases that are explicitly aligned with semantic attributes and reformulates saliency prediction as a weighted combination of probability maps connecting the bases and saliency. By applying our framework, we conduct extensive analyses from various perspectives, including the positive and negative weights of semantics, the impact of training data and architectural designs, the progressive influences of fine-tuning, and common error patterns of state-of-the-art deep saliency models. Additionally, we demonstrate the effectiveness of our framework by exploring visual attention characteristics in various application scenarios, such as the atypical attention of people with autism spectrum disorder, attention to emotion-eliciting stimuli, and attention evolution over time. Our code is publicly available at \url{https://github.com/szzexpoi/saliency_analysis}.</details>

### [Spotlight] What Makes Data Suitable for a Locally Connected Neural Network? A Necessary and Sufficient Condition Based on Quantum Entanglement.

**Authors:** ‪Yotam Alexander‬‏, Nimrod De La Vega, Noam Razin, Nadav Cohen

**<details><summary>Abstract**</summary>

The question of what makes a data distribution suitable for deep learning is a fundamental open problem. Focusing on locally connected neural networks (a prevalent family of architectures that includes convolutional and recurrent neural networks as well as local self-attention models), we address this problem by adopting theoretical tools from quantum physics. Our main theoretical result states that a certain locally connected neural network is capable of accurate prediction over a data distribution if and only if the data distribution admits low quantum entanglement under certain canonical partitions of features. As a practical application of this result, we derive a preprocessing method for enhancing the suitability of a data distribution to locally connected neural networks. Experiments with widespread models over various datasets demonstrate our findings. We hope that our use of quantum entanglement will encourage further adoption of tools from physics for formally reasoning about the relation between deep learning and real-world data.</details>

### What is the Inductive Bias of Flatness Regularization? A Study of Deep Matrix Factorization Models

**Authors:** Khashayar Gatmiry, Zhiyuan Li, Tengyu Ma, Sashank Reddi, Stefanie Jegelka, Ching-Yao Chuang

**<details><summary>Abstract**</summary>

Recent works on over-parameterized neural networks have shown that  the stochasticity in optimizers has the implicit regularization effect of minimizing the sharpness of the loss function (in particular, the trace of its Hessian) over the family zero-loss solutions. More explicit forms of flatness regularization also empirically improve the generalization performance. However, it remains unclear why and when flatness regularization leads to better generalization. This work takes the first step towards understanding the inductive bias of the minimum trace of the Hessian solutions in an important setting: learning deep linear networks from linear measurements, also known as \emph{deep matrix factorization}. We show that with the standard Restricted Isometry Property (RIP) on the measurements, minimizing the trace of Hessian is approximately equivalent to minimizing the Schatten 1-norm of the corresponding end-to-end matrix parameters (i.e., the product of all layer matrices), which in turn leads to better generalization.</details>

### [Spotlight] When Does Optimizing a Proper Loss Yield Calibration?

**Authors:** Jaroslaw Blasiok, Parikshit Gopalan, Lunjia Hu, Preetum Nakkiran

**<details><summary>Abstract**</summary>

Optimizing proper loss functions is popularly believed to yield predictors with good calibration properties; the intuition being that for such losses, the global optimum is to predict the ground-truth probabilities, which is indeed calibrated. However, typical machine learning models are trained to approximately minimize loss over restricted families of predictors, that are unlikely to contain the ground truth. Under what circumstances does optimizing proper loss  over a restricted family yield calibrated models? What precise calibration guarantees does it give? In this work, we provide a rigorous answer to these questions. We replace the global optimality with a local optimality condition stipulating that the (proper) loss of the predictor cannot be reduced much by post-processing its predictions with a certain family of Lipschitz functions. We show that any predictor with this local optimality satisfies smooth calibration as defined in [Kakade and Foster, 2008, Błasiok et al., 2023]. Local optimality is plausibly satisfied by well-trained DNNs, which suggests an explanation for why they are calibrated from proper loss minimization alone. Finally, we show that the connection between local optimality and calibration error goes both ways: nearly calibrated predictors are also nearly locally optimal.</details>

### When Visual Prompt Tuning Meets Source-Free Domain Adaptive Semantic Segmentation

**Authors:** Xinhong Ma, Yiming Wang, Hao Liu, Tianyu Guo, Yunhe Wang

**<details><summary>Abstract**</summary>

Source-free domain adaptive semantic segmentation aims to adapt a pre-trained source model to the unlabeled target domain without accessing the private source data. Previous methods usually fine-tune the entire network, which suffers from expensive parameter tuning. To avoid this problem, we propose to utilize visual prompt tuning for parameter-efficient adaptation. However, the existing visual prompt tuning methods are unsuitable for source-free domain adaptive semantic segmentation due to the following two reasons: (1) Commonly used visual prompts like input tokens or pixel-level perturbations cannot reliably learn informative knowledge beneficial for semantic segmentation. (2) Visual prompts require sufficient labeled data to fill the gap between the pre-trained model and downstream tasks. To alleviate these problems, we propose a universal unsupervised visual prompt tuning (Uni-UVPT) framework, which is applicable to various transformer-based backbones. Specifically, we first divide the source pre-trained backbone with frozen parameters into multiple stages, and propose a lightweight prompt adapter for progressively encoding informative knowledge into prompts and enhancing the generalization of target features between adjacent backbone stages. Cooperatively, a novel adaptive pseudo-label correction strategy with a multiscale consistency loss is designed to alleviate the negative effect of target samples with noisy pseudo labels and raise the capacity of visual prompts to spatial perturbations. Extensive experiments demonstrate that Uni-UVPT achieves state-of-the-art performance on GTA5 $	o$ Cityscapes and SYNTHIA $	o$ Cityscapes tasks and can serve as a universal and parameter-efficient framework for large-model unsupervised knowledge transfer. Code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/uni-uvpt and https://github.com/huawei-noah/noah-research/tree/master/uni-uvpt.</details>

### Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?

**Authors:** Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre Berges, Tingfan Wu, Jay Vakil, Pieter Abbeel, Jitendra Malik, Dhruv Batra, Yixin Lin, Oleksandr Maksymets, Aravind Rajeswaran, Franziska Meier

**<details><summary>Abstract**</summary>

We present the largest and most comprehensive empirical study of pre-trained visual representations (PVRs) or visual ‘foundation models’ for Embodied AI. First, we curate CortexBench, consisting of 17 different tasks spanning locomotion, navigation, dexterous, and mobile manipulation. Next, we systematically evaluate existing PVRs and find that none are universally dominant. To study the effect of pre-training data size and diversity, we combine over 4,000 hours of egocentric videos from 7 different sources (over 4.3M images) and ImageNet to train different-sized vision transformers using Masked Auto-Encoding (MAE) on slices of this data. Contrary to inferences from prior work, we find that scaling dataset size and diversity does not improve performance universally (but does so on average). Our largest model, named VC-1, outperforms all prior PVRs on average but does not universally dominate either. Next, we show that task- or domain-specific adaptation of VC-1 leads to substantial gains, with VC-1 (adapted) achieving competitive or superior performance than the best known results on all of the benchmarks in CortexBench. Finally, we present real-world hardware experiments, in which VC-1 and VC-1 (adapted) outperform the strongest pre-existing PVR. Overall, this paper presents no new techniques but a rigorous systematic evaluation, a broad set of findings about PVRs (that in some cases, refute those made in narrow domains in prior work), and open-sourced code and models (that required over 10,000 GPU-hours to train) for the benefit of the research community.</details>

### You Only Condense Once: Two Rules for Pruning Condensed Datasets

**Authors:** Yang He, Lingao Xiao, Joey Tianyi Zhou

**<details><summary>Abstract**</summary>

Dataset condensation is a crucial tool for enhancing training efficiency by reducing the size of the training dataset, particularly in on-device scenarios. However, these scenarios have two significant challenges: 1) the varying computational resources available on the devices require a dataset size different from the pre-defined condensed dataset, and 2) the limited computational resources often preclude the possibility of conducting additional condensation processes. We introduce You Only Condense Once (YOCO) to overcome these limitations. On top of one condensed dataset, YOCO produces smaller condensed datasets with two embarrassingly simple dataset pruning rules: Low LBPE Score and Balanced Construction. YOCO offers two key advantages: 1) it can flexibly resize the dataset to fit varying computational constraints, and 2) it eliminates the need for extra condensation processes, which can be computationally prohibitive. Experiments validate our findings on networks including ConvNet, ResNet and DenseNet, and datasets including CIFAR-10, CIFAR-100 and ImageNet. For example, our YOCO surpassed various dataset condensation and dataset pruning methods on CIFAR-10 with ten Images Per Class (IPC), achieving 6.98-8.89% and 6.31-23.92% accuracy gains, respectively. The code is available at: [https://github.com/he-y/you-only-condense-once](https://github.com/he-y/you-only-condense-once).</details>

### [Spotlight] Zero-shot causal learning

**Authors:** Hamed Nilforoshan, Michael Moor, Yusuf Roohani, Yining Chen, Anja Šurina, Michihiro Yasunaga, Sara Oblak, Jure Leskovec

**<details><summary>Abstract**</summary>

Predicting how different interventions will causally affect a specific individual is important in a variety of domains such as personalized medicine, public policy, and online marketing. There are a large number of methods to predict the effect of an existing intervention based on historical data from individuals who received it. However, in many settings it is important to predict the effects of novel interventions (e.g., a newly invented drug), which these methods do not address.Here, we consider zero-shot causal learning: predicting the personalized effects of a novel intervention. We propose CaML, a causal meta-learning framework which formulates the personalized prediction of each intervention's effect as a task. CaML trains a single meta-model across thousands of tasks, each constructed by sampling an intervention, its recipients, and its nonrecipients. By leveraging both intervention information (e.g., a drug's attributes) and individual features (e.g., a patient's history), CaML is able to predict the personalized effects of novel interventions that do not exist at the time of training. Experimental results on real world datasets in large-scale medical claims and cell-line perturbations demonstrate the effectiveness of our approach. Most strikingly, CaML's zero-shot predictions outperform even strong baselines trained directly on data from the test interventions.</details>

### k-Median Clustering via Metric Embedding: Towards Better Initialization with Differential Privacy

**Authors:** Chenglin Fan, Ping Li, Xiaoyun Li

**<details><summary>Abstract**</summary>

In clustering algorithms, the choice of initial centers is crucial for the quality of the learned clusters. We propose a new initialization scheme for the $k$-median problem in the general metric space (e.g., discrete space induced by graphs), based on the construction of metric embedding tree structure of the data. We propose a novel and efficient search algorithm, for good initial centers that can be used subsequently for the local search algorithm. The so-called HST initialization method can produce initial centers achieving lower error than those from another popular method $k$-median++, also with higher efficiency when $k$ is not too small. Our HST initialization can also be easily extended to the setting of differential privacy (DP) to generate private initial centers. We show that the error of applying DP local search followed by our private HST initialization improves previous results on the approximation error, and approaches the lower bound within a small factor. Experiments demonstrate the effectiveness of our proposed methods.</details>

