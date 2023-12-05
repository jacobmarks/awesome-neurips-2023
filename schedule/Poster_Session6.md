## Poster Session 6: Thursday, Dec 14, 15:00 CT

### $S^3$: Increasing GPU Utilization during Generative Inference for Higher Throughput

**Authors:** Yunho Jin, Chun-Feng Wu, David Brooks, Gu-Yeon Wei

**<details><summary>Abstract**</summary>

Generating texts with a large language model (LLM) consumes massive amounts of memory. Apart from the already-large model parameters, the key/value (KV) cache that holds information about previous tokens in a sequence can grow to be even larger than the model itself. This problem is exacerbated in one of the current LLM serving frameworks which reserves the maximum sequence length of memory for the KV cache to guarantee generating a complete sequence as they do not know the output sequence length. This restricts us to use a smaller batch size leading to lower GPU utilization and above all, lower throughput. We argue that designing a system with a priori knowledge of the output sequence can mitigate this problem. To this end, we propose $S^3$, which predicts the output sequence length, schedules generation queries based on the prediction to increase device resource utilization and throughput, and handle mispredictions. Our proposed method achieves 6.49× throughput over those systems that assume the worst case for the output sequence length.</details>

### $\mathbf{\mathbb{E}^{FWI}}$: Multiparameter Benchmark Datasets for Elastic Full Waveform Inversion of Geophysical Properties

**Authors:** Shihang Feng, Hanchen Wang, Chengyuan Deng, Yinan Feng, Yanhua Liu, Min Zhu, Peng Jin, Yinpeng Chen, Youzuo Lin

**<details><summary>Abstract**</summary>

Elastic geophysical properties (such as P- and S-wave velocities) are of great importance to various subsurface applications like CO$_2$ sequestration and energy exploration (e.g., hydrogen and geothermal). Elastic full waveform inversion (FWI) is widely applied for characterizing reservoir properties. In this paper, we introduce $\mathbf{\mathbb{E}^{FWI}}$, a comprehensive benchmark dataset that is specifically designed for elastic FWI. $\mathbf{\mathbb{E}^{FWI}}$ encompasses 8 distinct datasets that cover diverse subsurface geologic structures (flat, curve, faults, etc). The benchmark results produced by three different deep learning methods are provided. In contrast to our previously presented dataset (pressure recordings) for acoustic FWI (referred to as OpenFWI), the seismic dataset in $\mathbf{\mathbb{E}^{FWI}}$ has both vertical and horizontal components. Moreover, the velocity maps in $\mathbf{\mathbb{E}^{FWI}}$ incorporate both P- and S-wave velocities. While the multicomponent data and the added S-wave velocity make the data more realistic, more challenges are introduced regarding the convergence and computational cost of the inversion. We conduct comprehensive numerical experiments to explore the relationship between P-wave and S-wave velocities in seismic data. The relation between P- and S-wave velocities provides crucial insights into the subsurface properties such as lithology, porosity, fluid content, etc. We anticipate that $\mathbf{\mathbb{E}^{FWI}}$ will facilitate future research on multiparameter inversions and stimulate endeavors in several critical research topics of carbon-zero and new energy exploration. All datasets, codes and relevant information can be accessed through our website at https://efwi-lanl.github.io/</details>

### $\mathcal{M}^4$: A Unified XAI Benchmark for Faithfulness Evaluation of Feature Attribution Methods across Metrics, Modalities and Models

**Authors:** Xuhong Li, Mengnan Du, Jiamin Chen, Yekun Chai, Himabindu Lakkaraju, Haoyi Xiong

**<details><summary>Abstract**</summary>

While Explainable Artificial Intelligence (XAI) techniques have been widely studied to explain predictions made by deep neural networks, the way to evaluate the faithfulness of explanation results remains challenging, due to the heterogeneity of explanations for various models and the lack of ground-truth explanations. This paper introduces an XAI benchmark named $\mathcal{M}^4$, which allows evaluating various input feature attribution methods using the same set of faithfulness metrics across multiple data modalities (images and texts) and network structures (ResNets, MobileNets, Transformers). A taxonomy for the metrics has been proposed as well. We first categorize commonly used XAI evaluation metrics into three groups based on the ground truth they require. We then implement classic and state-of-the-art feature attribution methods using InterpretDL and conduct extensive experiments to compare methods and gain insights. Extensive experiments have been conducted to provide holistic evaluations as benchmark baselines. Several interesting observations are noticed for designing attribution algorithms. The implementation of state-of-the-art explanation methods and evaluation metrics of $\mathcal{M}^4$ is publicly available at \url{https://github.com/PaddlePaddle/InterpretDL}.</details>

### (Amplified) Banded Matrix Factorization: A unified approach to private training

**Authors:** Christopher A. Choquette-Choo, Arun Ganesh, Ryan McKenna, H. Brendan McMahan, John Rush, Abhradeep Guha Thakurta, Zheng Xu

**<details><summary>Abstract**</summary>

Matrix factorization (MF) mechanisms for differential privacy (DP) have substantially improved the state-of-the-art in privacy-utility-computation tradeoffs for ML applications in a variety of scenarios, but in both the centralized and federated settings there remain instances where either MF cannot be easily applied, or other algorithms provide better tradeoffs (typically, as $\epsilon$ becomes small).In this work, we show how MF can subsume prior state-of-the-art algorithms in both federated and centralized training settings, across all privacy budgets. The key technique throughout is the construction of MF mechanisms with banded matrices (lower-triangular matrices with at most $\hat{b}$ nonzero bands including the main diagonal). For cross-device federated learning (FL), this enables multiple-participations with a relaxed device participation schema compatible with practical FL infrastructure (as demonstrated by a production deployment).  In the centralized setting, we prove that banded matrices enjoy the same privacy amplification results as the ubiquitous DP-SGD algorithm, but can provide strictly better performance  in most scenarios---this lets us always at least match DP-SGD, and often outperform it</details>

### A Bounded Ability Estimation for Computerized Adaptive Testing

**Authors:** Yan Zhuang, Qi Liu, Guanhao Zhao, Zhenya Huang, Weizhe Huang, Zachary Pardos, Enhong Chen, Jinze Wu, Xin Li

**<details><summary>Abstract**</summary>

Computerized adaptive testing (CAT), as a tool that can efficiently measure student's ability, has been widely used in various standardized tests (e.g., GMAT and GRE). The adaptivity of CAT refers to the selection of the most informative questions for each student, reducing test length. Existing CAT methods do not explicitly target ability estimation accuracy since there is no student's true ability as ground truth; therefore, these methods cannot be guaranteed to make the estimate converge to the true with such limited responses. In this paper, we analyze the statistical properties of estimation and find a theoretical approximation of the true ability: the ability estimated by full responses to question bank. Based on this, a Bounded Ability Estimation framework for CAT (BECAT) is proposed in a data-summary manner, which selects a question subset that closely matches the gradient of the full responses. Thus, we develop an expected gradient difference approximation to design a simple greedy selection algorithm, and show the rigorous theoretical and error upper-bound guarantees of its ability estimate. Experiments on both real-world and synthetic datasets, show that it can reach the same estimation accuracy using 15\% less questions on average, significantly reducing test length.</details>

### A Closer Look at the Robustness of Contrastive Language-Image Pre-Training (CLIP)

**Authors:** Weijie Tu, Weijian Deng, Tom Gedeon

**<details><summary>Abstract**</summary>

Contrastive Language-Image Pre-training (CLIP) models have demonstrated remarkable generalization capabilities across multiple challenging distribution shifts. However, there is still much to be explored in terms of their robustness to the variations of specific visual factors. In real-world applications, reliable and safe systems must consider other safety measures beyond classification accuracy, such as predictive uncertainty. Yet, the effectiveness of CLIP models on such safety-related objectives is less-explored. Driven by the above, this work comprehensively investigates the safety measures of CLIP models, specifically focusing on three key properties: resilience to visual factor variations, calibrated uncertainty estimations, and the ability to detect anomalous inputs. To this end, we study $83$ CLIP models and $127$ ImageNet classifiers. They are diverse in architecture (pre)training distribution and training strategies. We consider $10$ visual factors (\emph{e.g.}, shape and pattern), $5$ types of out-of-distribution data, and $8$ natural and challenging test conditions with different shift types, such as texture, style, and perturbation shifts. Our study has unveiled several previously unknown insights into CLIP models. For instance, they are not consistently more calibrated than other ImageNet models, which contradicts existing findings. Additionally, our analysis underscores the significance of training source design by showcasing its profound influence on the three key properties. We believe our comprehensive study can shed light on and help guide the development of more robust and reliable CLIP models.</details>

### A Comprehensive Benchmark for Neural Human Radiance Fields

**Authors:** Kenkun Liu, Derong Jin, Ailing Zeng, Xiaoguang Han, Lei Zhang

**<details><summary>Abstract**</summary>

The past two years have witnessed a significant increase in interest concerning NeRF-based human body rendering. While this surge has propelled considerable advancements, it has also led to an influx of methods and datasets. This explosion complicates experimental settings and makes fair comparisons challenging. In this work, we design and execute thorough studies into unified evaluation settings and metrics to establish a fair and reasonable benchmark for human NeRF models. To reveal the effects of extant models, we benchmark them against diverse and hard scenes. Additionally, we construct a cross-subject benchmark pre-trained on large-scale datasets to assess generalizable methods. Finally, we analyze the essential components for animatability and generalizability, and make HumanNeRF from monocular videos generalizable, as the inaugural baseline. We hope these benchmarks and analyses could serve the community.</details>

### A Dataset for Analyzing Streaming Media Performance over HTTP/3 Browsers

**Authors:** Sapna Chaudhary, Mukulika Maity, Sandip Chakraborty, Naval Shukla

**<details><summary>Abstract**</summary>

HTTP/3 is a new application layer protocol supported by most browsers. It uses QUIC as an underlying transport protocol. QUIC provides multiple benefits, like faster connection establishment, reduced latency, and improved connection migration. Hence, most popular browsers like Chrome/Chromium, Microsoft Edge, Apple Safari, and Mozilla Firefox have started supporting it. In this paper, we present an HTTP/3-supported browser dataset collection tool named H3B. It collects the application and network-level logs during YouTube streaming. We consider YouTube, as it  the most popular video streaming application supporting QUIC. Using this tool, we collected a dataset of over 5936 YouTube sessions covering 5464 hours of streaming over 5 different geographical locations and 5 different bandwidth patterns. We believe our tool and as well as the dataset could be used in multiple applications such as a better configuration of application/transport protocols based on the network conditions, intelligent integration of network and application, predicting YouTube's QoE etc. We analyze the dataset and observe that during an HTTP/3 streaming not all requests are served by HTTP/3. Instead whenever the network condition is not favorable the browser chooses to fallback, and the application requests are transmitted using HTTP/2 over the old-standing transport protocol TCP. We observe that such switching of protocols impacts the performance of video streaming applications.</details>

### A Dataset of Relighted 3D Interacting Hands

**Authors:** Gyeongsik Moon, Shunsuke Saito, Weipeng Xu, Rohan Joshi, Julia Buffalini, Harley Bellan, Nicholas Rosen, Jesse Richardson, Mallorie Mize, Philippe De Bree, Tomas Simon, Bo Peng, Shubham Garg, Kevyn McPhail, Takaaki Shiratori

**<details><summary>Abstract**</summary>

The two-hand interaction is one of the most challenging signals to analyze due to the self-similarity, complicated articulations, and occlusions of hands. Although several datasets have been proposed for the two-hand interaction analysis, all of them do not achieve 1) diverse and realistic image appearances and 2) diverse and large-scale groundtruth (GT) 3D poses at the same time. In this work, we propose Re:InterHand, a dataset of relighted 3D interacting hands that achieve the two goals. To this end, we employ a state-of-the-art hand relighting network with our accurately tracked two-hand 3D poses. We compare our Re:InterHand with existing 3D interacting hands datasets and show the benefit of it. Our Re:InterHand is available in https://mks0601.github.io/ReInterHand/</details>

### A Logic for Expressing Log-Precision Transformers

**Authors:** William Merrill, Ashish Sabharwal

**<details><summary>Abstract**</summary>

One way to interpret the reasoning power of transformer-based language models is to describe the types of logical rules they can resolve over some input text. Recently, Chiang et al. (2023) showed that finite-precision transformer classifiers can be equivalently expressed in a generalization of first-order logic. However, finite-precision transformers are a weak transformer variant because, as we show, a single head can only attend to a constant number of tokens and, in particular, cannot represent uniform attention. Since attending broadly is a core capability for transformers, we ask whether a minimally more expressive model that can attend universally can also be characterized in logic. To this end, we analyze transformers whose forward pass is computed in $\log n$ precision on contexts of length $n$. We prove any log-precision transformer classifier can be equivalently expressed as a first-order logic sentence that, in addition to standard universal and existential quantifiers, may also contain majority-vote quantifiers. This is the tightest known upper bound and first logical characterization of log-precision transformers.</details>

### A Massive Scale Semantic Similarity Dataset of Historical English

**Authors:** Emily Silcock, Abhishek Arora, Melissa Dell

**<details><summary>Abstract**</summary>

A diversity of tasks use language models trained on semantic similarity data. While there are a variety of datasets that capture semantic similarity, they are either constructed from modern web data or are relatively small datasets created in the past decade by human annotators. This study utilizes a novel source, newly digitized articles from off-copyright, local U.S. newspapers, to assemble a massive-scale semantic similarity dataset spanning 70 years from 1920 to 1989 and containing nearly 400M positive semantic similarity pairs. Historically, around half of articles in U.S. local newspapers came from newswires like the Associated Press. While local papers reproduced articles from the newswire, they wrote their own headlines, which form abstractive summaries of the associated articles. We associate articles and their headlines by exploiting document layouts and language understanding. We then use deep neural methods to detect which articles are from the same underlying source, in the presence of substantial noise and abridgement. The headlines of reproduced articles form positive semantic similarity pairs. The resulting publicly available HEADLINES dataset is significantly larger than most existing semantic similarity datasets and covers a much longer span of time. It will facilitate the application of contrastively trained semantic similarity models to a variety of tasks, including the study of semantic change across space and time.</details>

### A Neural Collapse Perspective on Feature Evolution in Graph Neural Networks

**Authors:** Vignesh Kothapalli, Tom Tirer, Joan Bruna

**<details><summary>Abstract**</summary>

Graph neural networks (GNNs) have become increasingly popular for classification tasks on graph-structured data. Yet, the interplay between graph topology and feature evolution in GNNs is not well understood. In this paper, we focus on node-wise classification, illustrated with community detection on stochastic block model graphs, and explore the feature evolution through the lens of the "Neural Collapse" (NC) phenomenon. When training instance-wise deep classifiers (e.g. for image classification) beyond the zero training error point, NC demonstrates a reduction in the deepest features' within-class variability and an increased alignment of their class means to certain symmetric structures. We start with an empirical study that shows that a decrease in within-class variability is also prevalent in the node-wise classification setting, however, not to the extent observed in the instance-wise case. Then, we theoretically study this distinction. Specifically, we show that even an "optimistic" mathematical model requires that the graphs obey a strict structural condition in order to possess a minimizer with exact collapse. Furthermore, by studying the gradient dynamics of this model, we provide reasoning for the partial collapse observed empirically. Finally, we present a study on the evolution of within- and between-class feature variability across layers of a well-trained GNN and contrast the behavior with spectral methods.</details>

### A Novel Approach for Effective Multi-View Clustering with Information-Theoretic Perspective

**Authors:** Chenhang Cui, Yazhou Ren, Jingyu Pu, Jiawei Li, Xiaorong Pu, Tianyi Wu, Yutao Shi, Lifang He

**<details><summary>Abstract**</summary>

Multi-view clustering (MVC) is a popular technique for improving clustering performance using various data sources. However, existing methods primarily focus on acquiring consistent information while often neglecting the issue of redundancy across multiple views.This study presents a new approach called Sufficient Multi-View Clustering (SUMVC) that examines the multi-view clustering framework from an information-theoretic standpoint. Our proposed method consists of two parts. Firstly, we develop a simple and reliable multi-view clustering method SCMVC (simple consistent multi-view clustering) that employs variational analysis to generate consistent information. Secondly, we propose a sufficient representation lower bound to enhance consistent information and minimise unnecessary information among views. The proposed SUMVC method offers a promising solution to the problem of multi-view clustering and provides a new perspective for analyzing multi-view data.  To verify the effectiveness of our model, we conducted a theoretical analysis based on the Bayes Error Rate, and experiments on multiple multi-view datasets demonstrate the superior performance of SUMVC.</details>

### A Novel Framework for Policy Mirror Descent with General Parameterization and Linear Convergence

**Authors:** Carlo Alfano, Rui Yuan, Patrick Rebeschini

**<details><summary>Abstract**</summary>

Modern policy optimization methods in reinforcement learning, such as TRPO and PPO, owe their success to the use of parameterized policies. However, while theoretical guarantees have been established for this class of algorithms, especially in the tabular setting, the use of general parameterization schemes remains mostly unjustified. In this work, we introduce a novel framework for policy optimization based on mirror descent that naturally accommodates general parameterizations. The policy class induced by our scheme recovers known classes, e.g., softmax, and generates new ones depending on the choice of mirror map. Using our framework, we obtain the first result that guarantees linear convergence for a policy-gradient-based method involving general parameterization. To demonstrate the ability of our framework to accommodate general parameterization schemes, we provide its sample complexity when using shallow neural networks, show that it represents an improvement upon the previous best results, and empirically validate the effectiveness of our theoretical claims on classic control tasks.</details>

### A Regularized Conditional GAN for Posterior Sampling in Image Recovery Problems

**Authors:** Matthew Bendel, Rizwan Ahmad, Philip Schniter

**<details><summary>Abstract**</summary>

In image recovery problems, one seeks to infer an image from distorted, incomplete, and/or noise-corrupted measurements.Such problems arise in magnetic resonance imaging (MRI), computed tomography, deblurring, super-resolution, inpainting, phase retrieval, image-to-image translation, and other applications. Given a training set of signal/measurement pairs, we seek to do more than just produce one good image estimate. Rather, we aim to rapidly and accurately sample from the posterior distribution. To do this,we propose a regularized conditional Wasserstein GAN that generates dozens of high-quality posterior samples per second. Our regularization comprises an $\ell_1$ penalty and an adaptively weighted standard-deviation reward. Using quantitative evaluation metrics like conditional Fréchet inception distance, we demonstrate that our method produces state-of-the-art posterior samples in both multicoil MRI and large-scale inpainting applications. The code for our model can be found here: https://github.com/matt-bendel/rcGAN.</details>

### [Spotlight] A Robust and Opponent-Aware League Training Method for StarCraft II

**Authors:** Ruozi Huang, Xipeng Wu, Hongsheng Yu, Zhong Fan, Haobo Fu, Qiang Fu, Wei Yang

**<details><summary>Abstract**</summary>

It is extremely difficult to train a superhuman Artificial Intelligence (AI) for games of similar size to StarCraft II. AlphaStar is the first AI that beat human professionals in the full game of StarCraft II, using a league training framework that is inspired by a game-theoretic approach. In this paper, we improve AlphaStar's league training in two significant aspects.  We train goal-conditioned exploiters, whose abilities of spotting weaknesses in the main agent and the entire league are greatly improved compared to the unconditioned exploiters in AlphaStar. In addition, we endow the agents in the league with the new ability of opponent modeling, which makes the agent more responsive to the opponent's real-time strategy. Based on these improvements, we train a better and superhuman AI with orders of magnitude less resources than AlphaStar (see Table 1 for a full comparison). Considering the iconic role of StarCraft II in game AI research, we believe our method and results on StarCraft II provide valuable design principles on how one would utilize the general league training framework for obtaining a least-exploitable strategy in various, large-scale, real-world games.</details>

### A Step Towards Worldwide Biodiversity Assessment:  The BIOSCAN-1M Insect Dataset

**Authors:** Zahra Gharaee, ZeMing Gong, Nicholas Pellegrino, Iuliia Zarubiieva, Joakim Bruslund Haurum, Scott Lowe, Jaclyn McKeown, Chris Ho, Joschka McLeod, Yi-Yun Wei, Jireh Agda, Sujeevan Ratnasingham, Dirk Steinke, Angel Chang, Graham Taylor, Paul Fieguth

**<details><summary>Abstract**</summary>

In an effort to catalog insect biodiversity, we propose a new large dataset of hand-labelled insect images, the BIOSCAN-1M Insect Dataset. Each record is taxonomically classified by an expert, and also has associated genetic information including raw nucleotide barcode sequences and assigned barcode index numbers, which are genetic-based proxies for species classification. This paper presents a curated million-image dataset, primarily to train computer-vision models capable of providing image-based taxonomic assessment, however, the dataset also presents compelling characteristics, the study of which would be of interest to the broader machine learning community. Driven by the biological nature inherent to the dataset, a characteristic long-tailed class-imbalance distribution is exhibited. Furthermore, taxonomic labelling is a hierarchical classification scheme, presenting a highly fine-grained classification problem at lower levels. Beyond spurring interest in biodiversity research within the machine learning community, progress on creating an image-based taxonomic classifier will also further the ultimate goal of all BIOSCAN research: to lay the foundation for a comprehensive survey of global biodiversity. This paper introduces the dataset and explores the classification task through the implementation and analysis of a baseline classifier. The code repository of the BIOSCAN-1M-Insect dataset is available at https://github.com/zahrag/BIOSCAN-1M</details>

### A Toolkit for Reliable Benchmarking and Research in Multi-Objective Reinforcement Learning

**Authors:** Florian Felten, Lucas N. Alegre, Ann Nowe, Ana Bazzan, El Ghazali Talbi, Grégoire Danoy, Bruno da Silva

**<details><summary>Abstract**</summary>

Multi-objective reinforcement learning algorithms (MORL) extend standard reinforcement learning (RL) to scenarios where agents must optimize multiple---potentially conflicting---objectives, each represented by a distinct reward function. To facilitate and accelerate research and benchmarking in multi-objective RL problems, we introduce a comprehensive collection of software libraries that includes: (i) MO-Gymnasium, an easy-to-use and flexible API enabling the rapid construction of novel MORL environments. It also includes more than 20 environments under this API. This allows researchers to effortlessly evaluate any algorithms on any existing domains; (ii) MORL-Baselines, a collection of reliable and efficient implementations of state-of-the-art MORL algorithms, designed to provide a solid foundation for advancing research. Notably, all algorithms are inherently compatible with MO-Gymnasium; and(iii) a thorough and robust set of benchmark results and comparisons of MORL-Baselines algorithms, tested across various challenging MO-Gymnasium environments. These benchmarks were constructed to serve as guidelines for the research community, underscoring the properties, advantages, and limitations of each particular state-of-the-art method.</details>

### A Unified Framework for Uniform Signal Recovery in Nonlinear Generative Compressed Sensing

**Authors:** Junren Chen, Jonathan Scarlett, Michael Ng, Zhaoqiang Liu

**<details><summary>Abstract**</summary>

In generative compressed sensing (GCS), we want to recover a signal $\mathbf{x^*}\in\mathbb{R}^n$ from $m$ measurements ($m\ll n$) using a generative prior $\mathbf{x^*}\in G(\mathbb{B}_2^k(r))$, where $G$ is typically an $L$-Lipschitz continuous generative model and $\mathbb{B}_2^k(r)$ represents the radius-$r$ $\ell_2$-ball in $\mathbb{R}^k$. Under nonlinear measurements, most prior results are non-uniform, i.e., they hold with high probability for a fixed $\mathbf{x^*}$ rather than for all $\mathbf{x^*}$ simultaneously. In this paper, we build a unified framework to derive uniform recovery guarantees for nonlinear GCS where the observation model is nonlinear and possibly discontinuous or unknown. Our framework accommodates GCS with 1-bit/uniformly quantized observations and single index model as canonical examples. Specifically, using a single   realization of the sensing ensemble and generalized Lasso,   all $\mathbf{x^*}\in G(\mathbb{B}_2^k(r))$ can be recovered up to an $\ell_2$-error at most $\epsilon$ using roughly $\tilde{O}({k}/{\epsilon^2})$ samples, with omitted logarithmic factors typically being dominated by $\log L$.  Notably, this almost coincides with existing non-uniform guarantees up to logarithmic factors, hence the uniformity costs very little.    As part of our technical contributions, we introduce Lipschitz approximation to handle discontinuous observation models.    We also develop a concentration inequality  that produces tighter bound for product process whose index sets have low metric entropy.  Experimental results are presented to corroborate our theory.</details>

### A Unified, Scalable Framework for Neural Population Decoding

**Authors:** Mehdi Azabou, Vinam Arora, Venkataramana Ganesh, Ximeng Mao, Santosh Nachimuthu, Michael Mendelson, Blake Richards, Matthew Perich, Guillaume Lajoie, Eva Dyer

**<details><summary>Abstract**</summary>

Our ability to use deep learning approaches to decipher neural activity would likely benefit from greater scale, in terms of both the model size and the datasets. However, the integration of many neural recordings into one unified model is challenging, as each recording contains the activity of different neurons from different individual animals. In this paper, we introduce a training framework and architecture designed to model the population dynamics of neural activity across diverse, large-scale neural recordings. Our method first tokenizes individual spikes within the dataset to build an efficient representation of neural events that captures the fine temporal structure of neural activity. We then employ cross-attention and a PerceiverIO backbone to further construct a latent tokenization of neural population activities. Utilizing this architecture and training framework, we construct a large-scale multi-session model trained on large datasets from seven nonhuman primates, spanning over 158 different sessions of recording from over 27,373 neural units and over 100 hours of recordings. In a number of different tasks, we demonstrate that our pretrained model can be rapidly adapted to new, unseen sessions with unspecified neuron correspondence, enabling few-shot performance with minimal labels. This work presents a powerful new approach for building deep learning tools to analyze neural data and stakes out a clear path to training at scale for neural decoding models.</details>

### A benchmark of categorical encoders for binary classification

**Authors:** Federico Matteucci, Vadim Arzamasov, Klemens Böhm

**<details><summary>Abstract**</summary>

Categorical encoders transform categorical features into numerical representations that are indispensable for a wide range of machine learning models.Existing encoder benchmark studies lack generalizability because of their limited choice of (1) encoders, (2) experimental factors, and (3) datasets. Additionally, inconsistencies arise from the adoption of varying aggregation strategies.This paper is the most comprehensive benchmark of categorical encoders to date, including an extensive evaluation of 32 configurations of encoders from diverse families, with 36 combinations of experimental factors, and on 50 datasets.The study shows the profound influence of dataset selection, experimental factors, and aggregation strategies on the benchmark's conclusions~---~aspects disregarded in previous encoder benchmarks.Our code is available at \url{https://github.com/DrCohomology/EncoderBenchmarking}.</details>

### A generative model of the hippocampal formation trained with theta driven local learning rules

**Authors:** Tom M George, Kimberly Stachenfeld, Caswell Barry, Claudia Clopath, Tomoki Fukai

**<details><summary>Abstract**</summary>

Advances in generative models have recently revolutionised machine learning. Meanwhile, in neuroscience, generative models have long been thought fundamental to animal intelligence. Understanding the biological mechanisms that support these processes promises to shed light on the relationship between biological and artificial intelligence. In animals, the hippocampal formation is thought to learn and use a generative model to support its role in spatial and non-spatial memory. Here we introduce a biologically plausible model of the hippocampal formation tantamount to a Helmholtz machine that we apply to a temporal stream of inputs. A novel component of our model is that fast theta-band oscillations (5-10 Hz) gate the direction of information flow throughout the network, training it akin to a high-frequency wake-sleep algorithm. Our model accurately infers the latent state of high-dimensional sensory environments and generates realistic sensory predictions. Furthermore, it can learn to path integrate by developing a ring attractor connectivity structure matching previous theoretical proposals and flexibly transfer this structure between environments. Whereas many models trade-off biological plausibility with generality, our model captures a variety of hippocampal cognitive functions under one biologically plausible local learning rule.</details>

### ADGym: Design Choices for Deep Anomaly Detection

**Authors:** Minqi Jiang, Chaochuan Hou, Ao Zheng, Songqiao Han, Hailiang Huang, Qingsong Wen, Xiyang Hu, Yue Zhao

**<details><summary>Abstract**</summary>

Deep learning (DL) techniques have recently found success in anomaly detection (AD) across various fields such as finance, medical services, and cloud computing. However, most of the current research tends to view deep AD algorithms as a whole, without dissecting the contributions of individual design choices like loss functions and network architectures. This view tends to diminish the value of preliminary steps like data preprocessing, as more attention is given to newly designed loss functions, network architectures, and learning paradigms. In this paper, we aim to bridge this gap by asking two key questions: (i) Which design choices in deep AD methods are crucial for detecting anomalies? (ii) How can we automatically select the optimal design choices for a given AD dataset, instead of relying on generic, pre-existing solutions? To address these questions, we introduce ADGym, a platform specifically crafted for comprehensive evaluation and automatic selection of AD design elements in deep methods. Our extensive experiments reveal that relying solely on existing leading methods is not sufficient. In contrast, models developed using ADGym significantly surpass current state-of-the-art techniques.</details>

### AQuA: A Benchmarking Tool for Label Quality Assessment

**Authors:** Mononito Goswami, Vedant Sanil, Arjun Choudhry, Arvind Srinivasan, Chalisa Udompanyawit, Artur Dubrawski

**<details><summary>Abstract**</summary>

Machine learning (ML) models are only as good as the data they are trained on. But recent studies have found datasets widely used to train and evaluate ML models, e.g. _ImageNet_, to have pervasive labeling errors. Erroneous labels on the train set hurt ML models' ability to generalize, and they impact evaluation and model selection using the test set. Consequently, learning in the presence of labeling errors is an active area of research, yet this field lacks a comprehensive benchmark to evaluate these methods. Most of these methods are evaluated on a few computer vision datasets with significant variance in the experimental protocols. With such a large pool of methods and inconsistent evaluation, it is also unclear how ML practitioners can choose the right models to assess label quality in their data. To this end, we propose a benchmarking environment _AQuA_ to rigorously evaluate methods that enable machine learning in the presence of label noise. We also introduce a design space to delineate concrete design choices of label error detection models. We hope that our proposed design space and benchmark enable practitioners to choose the right tools to improve their label quality and that our benchmark enables objective and rigorous evaluation of machine learning tools facing mislabeled data.</details>

### AV-NeRF: Learning Neural Fields for Real-World Audio-Visual Scene Synthesis

**Authors:** Susan Liang, Chao Huang, Yapeng Tian, Anurag Kumar, Chenliang Xu

**<details><summary>Abstract**</summary>

Can machines recording an audio-visual scene produce realistic, matching audio-visual experiences at novel positions and novel view directions? We answer it by studying a new task---real-world audio-visual scene synthesis---and a first-of-its-kind NeRF-based approach for multimodal learning. Concretely, given a video recording of an audio-visual scene, the task is to synthesize new videos with spatial audios along arbitrary novel camera trajectories in that scene. We propose an acoustic-aware audio generation module that integrates prior knowledge of audio propagation into NeRF, in which we implicitly associate audio generation with the 3D geometry and material properties of a visual environment. Furthermore, we present a coordinate transformation module that expresses a view direction relative to the sound source, enabling the model to learn sound source-centric acoustic fields. To facilitate the study of this new task, we collect a high-quality Real-World Audio-Visual Scene (RWAVS) dataset. We demonstrate the advantages of our method on this real-world dataset and the simulation-based SoundSpaces dataset. Notably, we refer readers to view our demo videos for convincing comparisons.</details>

### AVOIDDS: Aircraft Vision-based Intruder Detection Dataset and Simulator

**Authors:** Elysia Smyers, Sydney Katz, Anthony Corso, Mykel J Kochenderfer

**<details><summary>Abstract**</summary>

Designing robust machine learning systems remains an open problem, and there is a need for benchmark problems that cover both environmental changes and evaluation on a downstream task. In this work, we introduce AVOIDDS, a realistic object detection benchmark for the vision-based aircraft detect-and-avoid problem. We provide a labeled dataset consisting of 72,000 photorealistic images of intruder aircraft with various lighting conditions, weather conditions, relative geometries, and geographic locations. We also provide an interface that evaluates trained models on slices of this dataset to identify changes in performance with respect to changing environmental conditions. Finally, we implement a fully-integrated, closed-loop simulator of the vision-based detect-and-avoid problem to evaluate trained models with respect to the downstream collision avoidance task. This benchmark will enable further research in the design of robust machine learning systems for use in safety-critical applications. The AVOIDDS dataset and code are publicly available at https://purl.stanford.edu/hj293cv5980 and https://github.com/sisl/VisionBasedAircraftDAA, respectively.</details>

### [Spotlight] Accelerated Quasi-Newton Proximal Extragradient: Faster Rate for Smooth Convex Optimization

**Authors:** Ruichen Jiang, Aryan Mokhtari

**<details><summary>Abstract**</summary>

In this paper, we propose an accelerated quasi-Newton proximal extragradient method for solving unconstrained smooth convex optimization problems. With access only to the gradients of the objective, we prove that our method can achieve a convergence rate of $\mathcal{O}\bigl(\min\\{\frac{1}{k^2}, \frac{\sqrt{d\log k}}{k^{2.5}}\\}\bigr)$, where $d$ is the problem dimension and $k$ is the number of iterations. In particular, in the regime where $k = \mathcal{O}(d)$, our method matches the _optimal rate_ of $\mathcal{O}(\frac{1}{k^2})$ by Nesterov's accelerated gradient (NAG). Moreover, in the the regime where $k = \Omega(d \log d)$, it outperforms NAG and converges at a _faster rate_ of $\mathcal{O}\bigl(\frac{\sqrt{d\log k}}{k^{2.5}}\bigr)$. To the best of our knowledge, this result is the first to demonstrate a provable gain for a quasi-Newton-type method over NAG in the convex setting.  To achieve such results, we build our method on a recent variant of the Monteiro-Svaiter acceleration framework and adopt an online learning perspective to update the Hessian approximation matrices, in which we relate the convergence rate of our method to the dynamic regret of a specific online convex optimization problem in the space of matrices.</details>

### Achieving Cross Modal Generalization with Multimodal Unified Representation

**Authors:** Yan Xia, Hai Huang, Jieming Zhu, Zhou Zhao

**<details><summary>Abstract**</summary>

This paper introduces a novel task called Cross Modal Generalization (CMG), which addresses the challenge of learning a unified discrete representation from paired multimodal data during pre-training. Then in downstream tasks, the model can achieve zero-shot generalization ability in other modalities when only one modal is labeled. Existing approaches in multimodal representation learning focus more on coarse-grained alignment or rely on the assumption that information from different modalities is completely aligned, which is impractical in real-world scenarios. To overcome this limitation, we propose 	extbf{Uni-Code}, which contains two key contributions: the Dual Cross-modal Information Disentangling (DCID) module and the Multi-Modal Exponential Moving Average (MM-EMA). These methods facilitate bidirectional supervision between modalities and align semantically equivalent information in a shared discrete latent space, enabling fine-grained unified representation of multimodal sequences. During pre-training, we investigate various modality combinations, including audio-visual, audio-text, and the tri-modal combination of audio-visual-text. Extensive experiments on various downstream tasks, i.e., cross-modal event classification, localization, cross-modal retrieval, query-based video segmentation, and cross-dataset event localization, demonstrate the effectiveness of our proposed methods. The code is available at https://github.com/haihuangcode/CMG.</details>

### Act As You Wish: Fine-Grained Control of Motion Diffusion Model with Hierarchical Semantic Graphs

**Authors:** Peng Jin, Yang Wu, Yanbo Fan, Zhongqian Sun, Wei Yang, Li Yuan

**<details><summary>Abstract**</summary>

Most text-driven human motion generation methods employ sequential modeling approaches, e.g., transformer, to extract sentence-level text representations automatically and implicitly for human motion synthesis. However, these compact text representations may overemphasize the action names at the expense of other important properties and lack fine-grained details to guide the synthesis of subtly distinct motion. In this paper, we propose hierarchical semantic graphs for fine-grained control over motion generation. Specifically, we disentangle motion descriptions into hierarchical semantic graphs including three levels of motions, actions, and specifics. Such global-to-local structures facilitate a comprehensive understanding of motion description and fine-grained control of motion generation. Correspondingly, to leverage the coarse-to-fine topology of hierarchical semantic graphs, we decompose the text-to-motion diffusion process into three semantic levels, which correspond to capturing the overall motion, local actions, and action specifics. Extensive experiments on two benchmark human motion datasets, including HumanML3D and KIT, with superior performances, justify the efficacy of our method. More encouragingly, by modifying the edge weights of hierarchical semantic graphs, our method can continuously refine the generated motion, which may have a far-reaching impact on the community. Code and pre-training weights are available at https://github.com/jpthu17/GraphMotion.</details>

### Active Learning for Semantic Segmentation with Multi-class Label Query

**Authors:** Sehyun Hwang, Sohyun Lee, Hoyoung Kim, Minhyeon Oh, Jungseul Ok, Suha Kwak

**<details><summary>Abstract**</summary>

This paper proposes a new active learning method for semantic segmentation. The core of our method lies in a new annotation query design. It samples informative local image regions ($	extit{e.g.}$, superpixels), and for each of such regions, asks an oracle for a multi-hot vector indicating all classes existing in the region. This multi-class labeling strategy is substantially more efficient than existing ones like segmentation, polygon, and even dominant class labeling in terms of annotation time per click. However, it introduces the class ambiguity issue in training as it assigns partial labels ($	extit{i.e.}$, a set of candidate classes) to individual pixels. We thus propose a new algorithm for learning semantic segmentation while disambiguating the partial labels in two stages. In the first stage, it trains a segmentation model directly with the partial labels through two new loss functions motivated by partial label learning and multiple instance learning. In the second stage, it disambiguates the partial labels by generating pixel-wise pseudo labels, which are used for supervised learning of the model. Equipped with a new acquisition function dedicated to the multi-class labeling, our method outperforms previous work on Cityscapes and PASCAL VOC 2012 while spending less annotation cost. Our code and results are available at [https://github.com/sehyun03/MulActSeg](https://github.com/sehyun03/MulActSeg).</details>

### Adaptive Algorithms for Relaxed Pareto Set Identification

**Authors:** Cyrille KONE, Emilie Kaufmann, Laura Richert

**<details><summary>Abstract**</summary>

In this paper we revisit the fixed-confidence identification of the Pareto optimal set in a multi-objective multi-armed bandit model. As the sample complexity to identify the exact Pareto set can be very large, a relaxation allowing to output some additional near-optimal arms has been studied. In this work we also tackle alternative relaxations that allow instead to identify a relevant \emph{subset} of the Pareto set. Notably, we propose a single sampling strategy, called Adaptive Pareto Exploration, that can be used in conjunction with different stopping rules to take into account different relaxations of the Pareto Set Identification problem. We analyze the sample complexity of these different combinations, quantifying in particular the reduction in sample complexity that occurs when one seeks to identify at most $k$ Pareto optimal arms. We showcase the good practical performance of Adaptive Pareto Exploration on a real-world scenario, in which we adaptively explore several vaccination strategies against Covid-19 in order to find the optimal ones when multiple immunogenicity criteria are taken into account.</details>

### Adaptive Linear Estimating Equations

**Authors:** Mufang Ying, Koulik Khamaru, Cun-Hui Zhang

**<details><summary>Abstract**</summary>

Sequential data collection has emerged as a widely adopted technique for enhancing the efficiency of data gathering processes. Despite its advantages, such data collection mechanism often introduces complexities to the statistical inference procedure. For instance, the ordinary least squares (OLS) estimator in an adaptive linear regression model can exhibit non-normal asymptotic behavior, posing challenges for accurate inference and interpretation. In this paper, we propose a general method for constructing debiased estimator which remedies this issue. It makes use of the idea of adaptive linear estimating equations, and we establish theoretical guarantees of asymptotic normality, supplemented by discussions on achieving near-optimal asymptotic variance. A salient feature of our estimator is that in the context of multi-armed bandits,  our estimator retains the non-asymptotic performance of the least square estimator while obtaining asymptotic normality property. Consequently, this work helps connect two fruitful paradigms of adaptive inference: a) non-asymptotic inference using concentration inequalities  and b) asymptotic inference via asymptotic normality.</details>

### [Spotlight] Adaptive whitening with fast gain modulation and slow synaptic plasticity

**Authors:** Lyndon Duong, Eero Simoncelli, Dmitri Chklovskii, David Lipshutz

**<details><summary>Abstract**</summary>

Neurons in early sensory areas rapidly adapt to changing sensory statistics, both by normalizing the variance of their individual responses and by reducing correlations between their responses. Together, these transformations may be viewed as an adaptive form of statistical whitening. Existing mechanistic models of adaptive whitening exclusively use either synaptic plasticity or gain modulation as the biological substrate for adaptation; however, on their own, each of these models has significant limitations. In this work, we unify these approaches in a normative multi-timescale mechanistic model that adaptively whitens its responses with complementary computational roles for synaptic plasticity and gain modulation. Gains are modified on a fast timescale to adapt to the current statistical context, whereas synapses are modified on a slow timescale to match structural properties of the input statistics that are invariant across contexts. Our model is derived from a novel multi-timescale whitening objective that factorizes the inverse whitening matrix into basis vectors, which correspond to synaptic weights, and a diagonal matrix, which corresponds to neuronal gains. We test our model on synthetic and natural datasets and find that the synapses learn optimal configurations over long timescales that enable adaptive whitening on short timescales using gain modulation.</details>

### Adversarial Resilience in Sequential Prediction via Abstention

**Authors:** Surbhi Goel, Steve Hanneke, Shay Moran, Abhishek Shetty

**<details><summary>Abstract**</summary>

We study the problem of sequential prediction in the stochastic setting with an adversary that is allowed to inject clean-label adversarial (or out-of-distribution) examples. Algorithms designed to handle purely stochastic data tend to fail in the presence of such adversarial examples, often leading to erroneous predictions. This is undesirable in many high-stakes applications such as medical recommendations, where abstaining from predictions on adversarial examples is preferable to misclassification. On the other hand, assuming fully adversarial data leads to very pessimistic bounds that are often vacuous in practice.     To move away from these pessimistic guarantees, we propose a new model of sequential prediction that sits between the purely stochastic and fully adversarial settings by allowing the learner to abstain from making a prediction at no cost on adversarial examples, thereby asking the learner to make predictions with certainty. Assuming access to the marginal distribution on the non-adversarial examples, we design a learner whose error scales with the VC dimension (mirroring the stochastic setting) of the hypothesis class, as opposed to the Littlestone dimension which characterizes the fully adversarial setting. Furthermore, we design learners for VC dimension~1 classes and the class of axis-aligned rectangles, which work even in the absence of access to the marginal distribution. Our key technical contribution is a novel measure for quantifying uncertainty for learning VC classes, which may be of independent interest.</details>

### [Spotlight] Adversarial Robustness in Graph Neural Networks: A Hamiltonian Approach

**Authors:** Kai Zhao, Qiyu Kang, Yang Song, Rui She, Sijie Wang, Wee Peng Tay

**<details><summary>Abstract**</summary>

Graph neural networks (GNNs) are vulnerable to adversarial perturbations, including those that affect both node features and graph topology. This paper investigates GNNs derived from diverse neural flows, concentrating on their connection to various stability notions such as BIBO stability, Lyapunov stability, structural stability, and conservative stability. We argue that Lyapunov stability, despite its common use, does not necessarily ensure adversarial robustness. Inspired by physics principles, we advocate for the use of conservative Hamiltonian neural flows to construct GNNs that are robust to adversarial attacks. The adversarial robustness of different neural flow GNNs is empirically compared on several benchmark datasets under a variety of adversarial attacks. Extensive numerical experiments demonstrate that GNNs leveraging conservative Hamiltonian flows with Lyapunov stability substantially improve robustness against adversarial perturbations. The implementation code of experiments is available at \url{https://github.com/zknus/NeurIPS-2023-HANG-Robustness}.</details>

### Alexa Arena: A User-Centric Interactive Platform for Embodied AI

**Authors:** Qiaozi Gao, Govindarajan Thattai, Suhaila Shakiah, Xiaofeng Gao, Shreyas Pansare, Vasu Sharma, Gaurav Sukhatme, Hangjie Shi, Bofei Yang, Desheng Zhang, Lucy Hu, Karthika Arumugam, Shui Hu, Matthew Wen, Dinakar Guthy, Shunan Chung, Rohan Khanna, Osman Ipek, Leslie Ball, Kate Bland, Heather Rocker, Michael Johnston, Reza Ghanadan, Dilek Hakkani-Tur, Prem Natarajan

**<details><summary>Abstract**</summary>

We introduce Alexa Arena, a user-centric simulation platform to facilitate research in building assistive conversational embodied agents. Alexa Arena features multi-room layouts and an abundance of interactable objects. With user-friendly graphics and control mechanisms, the platform supports the development of gamified robotic tasks readily accessible to general human users, allowing high-efficiency data collection and EAI system evaluation. Along with the platform, we introduce a dialog-enabled task completion benchmark with online human evaluations.</details>

### Algorithm Selection for Deep Active Learning with Imbalanced Datasets

**Authors:** Jifan Zhang, Shuai Shao, Saurabh Verma, Robert Nowak

**<details><summary>Abstract**</summary>

Label efficiency has become an increasingly important objective in deep learning applications. Active learning aims to reduce the number of labeled examples needed to train deep networks, but the empirical performance of active learning algorithms can vary dramatically across datasets and applications. It is difficult to know in advance which active learning strategy will perform well or best in a given application. To address this, we propose the first adaptive algorithm selection strategy for deep active learning. For any unlabeled dataset, our (meta) algorithm TAILOR (Thompson ActIve Learning algORithm selection) iteratively and adaptively chooses among a set of candidate active learning algorithms. TAILOR uses novel reward functions aimed at gathering class-balanced examples. Extensive experiments in multi-class and multi-label applications demonstrate TAILOR's effectiveness in achieving accuracy comparable or better than that of the best of the candidate algorithms. Our implementation of TAILOR is open-sourced at https://github.com/jifanz/TAILOR.</details>

### All Points Matter: Entropy-Regularized Distribution Alignment for Weakly-supervised 3D Segmentation

**Authors:** Liyao Tang, Zhe Chen, Shanshan Zhao, Chaoyue Wang, Dacheng Tao

**<details><summary>Abstract**</summary>

Pseudo-labels are widely employed in weakly supervised 3D segmentation tasks where only sparse ground-truth labels are available for learning.Existing methods often rely on empirical label selection strategies, such as confidence thresholding, to generate beneficial pseudo-labels for model training.This approach may, however, hinder the comprehensive exploitation of unlabeled data points.We hypothesize that this selective usage arises from the noise in pseudo-labels generated on unlabeled data. The noise in pseudo-labels may result in significant discrepancies between pseudo-labels and model predictions, thus confusing and affecting the model training greatly.To address this issue, we propose a novel learning strategy to regularize the generated pseudo-labels and effectively narrow the gaps between pseudo-labels and model predictions.More specifically, our method introduces an Entropy Regularization loss and a Distribution Alignment loss for weakly supervised learning in 3D segmentation tasks, resulting in an ERDA learning strategy.Interestingly, by using KL distance to formulate the distribution alignment loss, it reduces to a deceptively simple cross-entropy-based loss which optimizes both the pseudo-label generation network and the 3D segmentation network simultaneously.Despite the simplicity, our method promisingly improves the performance.We validate the effectiveness through extensive experiments on various baselines and large-scale datasets.Results show that ERDA effectively enables the effective usage of all unlabeled data points for learning and achieves state-of-the-art performance under different settings.Remarkably, our method can outperform fully-supervised baselines using only 1\% of true annotations.Code and model will be made publicly available at https://github.com/LiyaoTang/ERDA.</details>

### Alternating Gradient Descent and Mixture-of-Experts for Integrated Multimodal Perception

**Authors:** Hassan Akbari, Dan Kondratyuk, Yin Cui, Rachel Hornung, Huisheng Wang, Hartwig Adam

**<details><summary>Abstract**</summary>

We present Integrated Multimodal Perception (IMP), a simple and scalable multimodal multi-task training and modeling approach. IMP integrates multimodal inputs including image, video, text, and audio into a single Transformer encoder with minimal modality-specific components. IMP makes use of a novel design that combines Alternating Gradient Descent (AGD) and Mixture-of-Experts (MoE) for efficient model & task scaling. We conduct extensive empirical studies and reveal the following key insights:    1) performing gradient descent updates by alternating on diverse modalities, loss functions, and tasks, with varying input resolutions, efficiently improves the model.    2) sparsification with MoE on a single modality-agnostic encoder substantially improves the performance, outperforming dense models that use modality-specific encoders or additional fusion layers and greatly mitigating the conflicts between modalities. IMP achieves competitive performance on a wide range of downstream tasks including video classification, image classification, image-text, and video-text retrieval. Most notably, we train a sparse IMP-MoE-L focusing on video tasks that achieves new state-of-the-art in zero-shot video classification: 77.0% on Kinetics-400, 76.8% on Kinetics-600, and 68.3% on Kinetics-700, improving the previous state-of-the-art by +5%, +6.7%, and +5.8%, respectively, while using only 15% of their total training computational cost.</details>

### Ambient Diffusion: Learning Clean Distributions from Corrupted Data

**Authors:** Giannis Daras, Kulin Shah, Yuval Dagan, Aravind Gollakota, Alex Dimakis, Adam Klivans

**<details><summary>Abstract**</summary>

We present the first diffusion-based framework that can learn an unknown distribution using only highly-corrupted samples. This problem arises in scientific applications where access to uncorrupted samples is impossible or expensive to acquire. Another benefit of our approach is the ability to train generative models that are less likely to memorize any individual training sample, since they never observe clean training data. Our main idea is to introduce additional measurement distortion during the diffusion process and require the model to predict the original corrupted image from the further corrupted image.  We prove that our method leads to models that learn the conditional expectation of the full uncorrupted image given this additional measurement corruption.  This holds for any corruption process that satisfies some technical conditions (and in particular includes inpainting and compressed sensing).  We train models on standard benchmarks (CelebA, CIFAR-10 and AFHQ) and show that we can learn the distribution even when all the training samples have 90\% of their pixels missing. We also show that we can finetune foundation models on small corrupted datasets (e.g. MRI scans with block corruptions) and learn the clean distribution without memorizing the training set.</details>

### American Stories: A Large-Scale Structured Text Dataset of Historical U.S. Newspapers

**Authors:** Melissa Dell, Jacob Carlson, Tom Bryan, Emily Silcock, Abhishek Arora, Zejiang Shen, Luca D'Amico-Wong, Quan Le, Pablo Querubin, Leander Heldring

**<details><summary>Abstract**</summary>

Existing full text datasets of U.S. public domain newspapers do not recognize the often complex layouts of newspaper scans, and as a result the digitized content scrambles texts from articles, headlines, captions, advertisements, and other layout regions. OCR quality can also be low. This study develops a novel, deep learning pipeline for extracting full article texts from newspaper images and applies it to the nearly 20 million scans in Library of Congress's public domain Chronicling America collection. The pipeline includes layout detection, legibility classification, custom OCR, and association of article texts spanning multiple bounding boxes. To achieve high scalability, it is built with efficient architectures designed for mobile phones. The resulting American Stories dataset provides high quality data that could be used for pre-training a large language model to achieve better understanding of historical English and historical world knowledge. The dataset could also be added to the external database of a retrieval-augmented language model to make historical information - ranging from interpretations of political events to minutiae about the lives of people's ancestors - more widely accessible. Furthermore, structured article texts facilitate using transformer-based methods for popular social science applications like topic classification, detection of reproduced content, and news story clustering.  Finally, American Stories provides a massive silver quality dataset for innovating multimodal layout analysis models and other multimodal applications.</details>

### An Alternating Optimization Method for Bilevel Problems under the Polyak-Łojasiewicz Condition

**Authors:** Quan Xiao, Songtao Lu, Tianyi Chen

**<details><summary>Abstract**</summary>

Bilevel optimization has recently regained interest owing to its applications in emerging machine learning fields such as hyperparameter optimization, meta-learning, and reinforcement learning. Recent results have shown that simple alternating (implicit) gradient-based algorithms can match the convergence rate of single-level gradient descent (GD) when addressing bilevel problems with a strongly convex lower-level objective. However, it remains unclear whether this result can be generalized to bilevel problems beyond this basic setting. In this paper, we first introduce a stationary metric for the considered bilevel problems, which generalizes the existing metric, for a nonconvex lower-level objective that satisfies the Polyak-Łojasiewicz (PL) condition. We then propose a Generalized ALternating mEthod for bilevel opTimization (GALET) tailored to BLO with convex PL LL problem and establish that GALET achieves an $\epsilon$-stationary point for the considered problem within $\tilde{\cal O}(\epsilon^{-1})$ iterations, which matches the iteration complexity of GD for single-level smooth nonconvex problems.</details>

### An Efficient and Robust Framework for Approximate Nearest Neighbor Search with Attribute Constraint

**Authors:** Mengzhao Wang, Lingwei Lv, Xiaoliang Xu, Yuxiang Wang, Qiang Yue, Jiongkang Ni

**<details><summary>Abstract**</summary>

This paper introduces an efficient and robust framework for hybrid query (HQ) processing, which combines approximate nearest neighbor search (ANNS) with attribute constraint. HQ aims to find objects that are similar to a feature vector and match some structured attributes. Existing methods handle ANNS and attribute filtering separately, leading to inefficiency and inaccuracy. Our framework, called native hybrid query (NHQ), builds a composite index based on proximity graph (PG) and applies joint pruning for HQ. We can easily adapt existing PGs to this framework for efficient HQ processing. We also propose two new navigable PGs (NPGs) with optimized edge selection and routing, which improve the overall ANNS performance. We implement five HQ methods based on the proposed NPGs and existing PGs in NHQ, and show that they outperform the state-of-the-art methods on 10 real-world datasets (up to 315$\times$ faster with the same accuracy).</details>

### An Empirical Investigation of the Role of Pre-training in Lifelong Learning

**Authors:** Sanket Vaibhav Mehta, Darshan Patil, Sarath Chandar, Emma Strubell

**<details><summary>Abstract**</summary>

The lifelong learning paradigm in machine learning is an attractive alternative to the more prominent isolated learning scheme not only due to its resemblance to biological learning but also its potential to reduce energy waste by obviating excessive model re-training. A key challenge to this paradigm is the phenomenon of catastrophic forgetting. With the increasing popularity and success of pre-trained models in machine learning, we pose the question: What role does pre-training play in lifelong learning, specifically with respect to catastrophic forgetting? We investigate existing methods in the context of large, pre-trained models and evaluate their performance on a variety of text and image classification tasks, including a large-scale study using a novel data set of 15 diverse NLP tasks. Across all settings, we observe that generic pre-training implicitly alleviates the effects of catastrophic forgetting when learning multiple tasks sequentially compared to randomly initialized models. We then further investigate why pre-training alleviates forgetting in this setting. We study this phenomenon by analyzing the loss landscape, finding that pre-trained weights appear to ease forgetting by leading to wider minima. Based on this insight, we propose jointly optimizing for current task loss and loss basin sharpness to explicitly encourage wider basins during sequential fine-tuning. We show that this optimization approach outperforms several state-of-the-art task-sequential continual learning algorithms across multiple settings, occasionally even without retaining a memory that scales in size with the number of tasks.</details>

### An Exploration-by-Optimization Approach to Best of Both Worlds in Linear Bandits

**Authors:** Shinji Ito, Kei Takemura

**<details><summary>Abstract**</summary>

In this paper, we consider how to construct best-of-both-worlds linear bandit algorithms that achieve nearly optimal performance for both stochastic and adversarial environments.  For this purpose, we show that a natural approach referred to as exploration by optimization [Lattimore and Szepesvári, 2020] works well. Specifically, an algorithm constructed using this approach achieves $O(d \sqrt{ T \log{T}})$-regret in adversarial environments and $O(\frac{d^2 \log T}{\Delta_{\min}} )$-regret in stochastic environments.  Symbols $d$, $T$ and $\Delta_{\min}$ here represent the dimensionality of the action set, the time horizon, and the minimum sub-optimality gap, respectively.  We also show that this algorithm has even better theoretical guarantees for important special cases including the multi-armed bandit problem and multitask bandits.</details>

### An NLP Benchmark Dataset for Assessing Corporate Climate Policy Engagement

**Authors:** Gaku Morio, Christopher D Manning

**<details><summary>Abstract**</summary>

As societal awareness of climate change grows, corporate climate policy engagements are attracting attention.We propose a dataset to estimate corporate climate policy engagement from various PDF-formatted documents.Our dataset comes from LobbyMap (a platform operated by global think tank InfluenceMap) that provides engagement categories and stances on the documents.To convert the LobbyMap data into the structured dataset, we developed a pipeline using text extraction and OCR.Our contributions are: (i) Building an NLP dataset including 10K documents on corporate climate policy engagement. (ii) Analyzing the properties and challenges of the dataset. (iii) Providing experiments for the dataset using pre-trained language models.The results show that while Longformer outperforms baselines and other pre-trained models, there is still room for significant improvement.We hope our work begins to bridge research on NLP and climate change.</details>

### An Optimization-based Approach To Node Role Discovery in Networks: Approximating Equitable Partitions

**Authors:** Michael Scholkemper, Michael T Schaub

**<details><summary>Abstract**</summary>

Similar to community detection, partitioning the nodes of a complex network according to their structural roles aims to identify fundamental building blocks of a network, which can be used, e.g., to find simplified descriptions of the network connectivity, to derive reduced order models for dynamical processes unfolding on processes, or as ingredients for various network analysis and graph mining tasks. In this work, we offer a fresh look on the problem of role extraction and its differences to community detection and present a definition of node roles and two associated optimization problems (cost functions) grounded in ideas related to graph-isomorphism tests, the Weisfeiler-Leman algorithm and equitable partitions. We present theoretical guarantees and validate our approach via a novel “role-infused partition benchmark”, a network model from which we can sample networks in which nodes are endowed with different roles in a stochastic way.</details>

### AndroidInTheWild: A Large-Scale Dataset For Android Device Control

**Authors:** Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, Timothy Lillicrap

**<details><summary>Abstract**</summary>

There is a growing interest in device-control systems that can interpret human natural language instructions and execute them on a digital device by directly controlling its user interface. We present a dataset for device-control research, Android in the Wild (AitW), which is orders of magnitude larger than current datasets. The dataset contains human demonstrations of device interactions, including the screens and actions, and corresponding natural language instructions. It consists of 715k episodes spanning 30k unique instructions, four versions of Android (v10–13), and eight device types (Pixel 2 XL to Pixel 6) with varying screen resolutions. It contains multi-step tasks that require semantic understanding of language and visual context. This dataset poses a new challenge: actions available through the user interface must be inferred from their visual appearance, and, instead of simple UI element-based actions, the action space consists of precise gestures (e.g., horizontal scrolls to operate carousel widgets). We organize our dataset to encourage robustness analysis of device-control systems, i.e., how well a system performs in the presence of new task descriptions, new applications, or new platform versions. We develop two agents and report performance across the dataset. The dataset is available at https://github.com/google-research/google-research/tree/master/android_in_the_wild.</details>

### Annotator: A Generic Active Learning Baseline for LiDAR Semantic Segmentation

**Authors:** Binhui Xie, Shuang Li, Qingju Guo, Chi Liu, Xinjing Cheng

**<details><summary>Abstract**</summary>

Active learning, a label-efficient paradigm, empowers models to interactively query an oracle for labeling new data. In the realm of LiDAR semantic segmentation, the challenges stem from the sheer volume of point clouds, rendering annotation labor-intensive and cost-prohibitive. This paper presents Annotator, a general and efficient active learning baseline, in which a voxel-centric online selection strategy is tailored to efficiently probe and annotate the salient and exemplar voxel girds within each LiDAR scan, even under distribution shift. Concretely, we first execute an in-depth analysis of several common selection strategies such as Random, Entropy, Margin, and then develop voxel confusion degree (VCD) to exploit the local topology relations and structures of point clouds. Annotator excels in diverse settings, with a particular focus on active learning (AL), active source-free domain adaptation (ASFDA), and active domain adaptation (ADA). It consistently delivers exceptional performance across LiDAR semantic segmentation benchmarks, spanning both simulation-to-real and real-to-real scenarios. Surprisingly, Annotator exhibits remarkable efficiency, requiring significantly fewer annotations, e.g., just labeling five voxels per scan in the SynLiDAR → SemanticKITTI task. This results in impressive performance, achieving 87.8% fully-supervised performance under AL, 88.5% under ASFDA, and 94.4% under ADA. We envision that Annotator will offer a simple, general, and efficient solution for label-efficient 3D applications.</details>

### [Oral] Are Emergent Abilities of Large Language Models a Mirage?

**Authors:** Rylan Schaeffer, Brando Miranda, Sanmi Koyejo

**Oral Presentation:** Th, Dec 14, 13:20 -- Oral 6A

**<details><summary>Abstract**</summary>

Recent work claims that large language models display \textit{emergent abilities}, abilities not present in smaller-scale models that are present in larger-scale models.What makes emergent abilities intriguing is two-fold: their \textit{sharpness}, transitioning seemingly instantaneously from not present to present, and their \textit{unpredictability}, appearing at seemingly unforeseeable model scales.Here, we present an alternative explanation for emergent abilities: that for a particular task and model family, when analyzing fixed model outputs, emergent abilities appear due the researcher’s choice of metric rather than due to fundamental changes in model behavior with scale. Specifically, nonlinear or discontinuous metrics produce apparent emergent abilities, whereas linear or continuous metrics produce smooth, continuous, predictable changes in model performance.We present our alternative explanation in a simple mathematical model, then test it in three complementary ways: we (1) make, test and confirm three predictions on the effect of metric choice using the InstructGPT/GPT-3 family on tasks with claimed emergent abilities, (2) make, test and confirm two predictions about metric choices in a meta-analysis of emergent abilities on BIG-Bench; and (3) show how to choose metrics to produce never-before-seen seemingly emergent abilities in multiple vision tasks across diverse deep networks.Via all three analyses, we provide evidence that alleged emergent abilities evaporate with different metrics or with better statistics, and may not be a fundamental property of scaling AI models.</details>

### Asymptotics of Bayesian Uncertainty Estimation in Random Features Regression

**Authors:** Youngsoo Baek, Samuel Berchuck, Sayan Mukherjee

**<details><summary>Abstract**</summary>

In this paper we compare and contrast the behavior of the posterior predictive distribution to the risk of the the maximum a posteriori estimator for the random features regression model in the overparameterized regime.  We will focus on the variance of the  posterior predictive distribution (Bayesian model average) and compare its asymptotics to that of the risk of the MAP estimator. In the regime where the model dimensions grow faster than any constant multiple of the number of samples, asymptotic agreement between these two quantities is governed by the phase transition in the signal-to-noise ratio. They also asymptotically agree with each other when the number of samples grow faster than any constant multiple of model dimensions. Numerical simulations illustrate finer distributional properties of the two quantities for finite dimensions. We conjecture they have Gaussian fluctuations and exhibit similar properties as found by previous authors in a Gaussian sequence model, this is of independent theoretical interest.</details>

### Augmented Memory Replay-based Continual Learning Approaches for Network Intrusion Detection

**Authors:** suresh kumar amalapuram, Sumohana Channappayya, Bheemarjuna Reddy Tamma

**<details><summary>Abstract**</summary>

Intrusion detection is a form of anomalous activity detection in communication network traffic. Continual learning (CL) approaches to the intrusion detection task accumulate old knowledge while adapting to the latest threat knowledge. Previous works have shown the effectiveness of memory replay-based CL approaches for this task. In this work, we present two novel contributions to improve the performance of CL-based network intrusion detection in the context of class imbalance and scalability. First, we extend class balancing reservoir sampling (CBRS), a memory-based CL method, to address the problems of severe class imbalance for large datasets. Second, we propose a novel approach titled perturbation assistance for parameter approximation (PAPA) based on the Gaussian mixture model to reduce the number of \textit{virtual stochastic gradient descent (SGD) parameter} computations needed to discover maximally interfering samples for CL. We demonstrate that the proposed approaches perform remarkably better than the baselines on standard intrusion detection benchmarks created over shorter periods (KDDCUP'99, NSL-KDD, CICIDS-2017/2018, UNSW-NB15, and CTU-13) and a longer period with distribution shift (AnoShift). We also validated proposed approaches on standard continual learning benchmarks (SVHN, CIFAR-10/100, and CLEAR-10/100) and anomaly detection benchmarks (SMAP, SMD, and MSL). Further, the proposed PAPA approach significantly lowers the number of virtual SGD update operations, thus resulting in training time savings in the range of 12 to 40\% compared to the maximally interfered samples retrieval algorithm.</details>

### Auslan-Daily: Australian Sign Language Translation for Daily Communication and News

**Authors:** Xin Shen, Shaozu Yuan, Hongwei Sheng, Heming Du, Xin Yu

**<details><summary>Abstract**</summary>

Sign language translation (SLT) aims to convert a continuous sign language video clip into a spoken language. Considering different geographic regions generally have their own native sign languages, it is valuable to establish corresponding SLT datasets to support related communication and research. Auslan, as a sign language specific to Australia, still lacks a dedicated large-scale dataset for SLT.To fill this gap, we curate an Australian Sign Language translation dataset, dubbed Auslan-Daily, which is collected from the Auslan educational TV series and Auslan TV programs. The former involves daily communications among multiple signers in the wild, while the latter comprises sign language videos for up-to-date news, weather forecasts, and documentaries. In particular, Auslan-Daily has two main features: (1) the topics are diverse and signed by multiple signers, and (2) the scenes in our dataset are more complex, e.g., captured in various environments, gesture interference during multi-signers' interactions and various camera positions. With a collection of more than 45 hours of high-quality Auslan video materials, we invite Auslan experts to align different fine-grained visual and language pairs, including video $\leftrightarrow$ fingerspelling, video $\leftrightarrow$ gloss, and video $\leftrightarrow$ sentence. As a result, Auslan-Daily contains multi-grained annotations that can be utilized to accomplish various fundamental sign language tasks, such as signer detection, sign spotting, fingerspelling detection, isolated sign language recognition, sign language translation and alignment. Moreover, we benchmark results with state-of-the-art models for each task in Auslan-Daily. Experiments indicate that Auslan-Daily is a highly challenging SLT dataset, and we hope this dataset will contribute to the development of Auslan and the advancement of sign languages worldwide in a broader context. All datasets and benchmarks are available at Auslan-Daily.</details>

### Automated Classification of Model Errors on ImageNet

**Authors:** Momchil Peychev, Mark Müller, Marc Fischer, Martin Vechev

**<details><summary>Abstract**</summary>

While the ImageNet dataset has been driving computer vision research over the past decade, significant label noise and ambiguity have made top-1 accuracy an insufficient measure of further progress. To address this, new label-sets and evaluation protocols have been proposed for ImageNet showing that state-of-the-art models already achieve over 95% accuracy and shifting the focus on investigating why the remaining errors persist.Recent work in this direction employed a panel of experts to manually categorize all remaining classification errors for two selected models. However, this process is time-consuming, prone to inconsistencies, and requires trained experts, making it unsuitable for regular model evaluation thus limiting its utility. To overcome these limitations, we propose the first automated error classification framework, a valuable tool to study how modeling choices affect error distributions. We use our framework to comprehensively evaluate the error distribution of over 900 models. Perhaps surprisingly, we find that across model architectures, scales, and pre-training corpora, top-1 accuracy is a strong predictor for the *portion* of all error types. In particular, we observe that the portion of severe errors drops significantly with top-1 accuracy indicating that, while it underreports a model's true performance, it remains a valuable performance metric.We release all our code at https://github.com/eth-sri/automated-error-analysis.</details>

### Balance, Imbalance, and Rebalance: Understanding Robust Overfitting from a Minimax Game Perspective

**Authors:** Yifei Wang, Liangchen Li, Jiansheng Yang, Zhouchen Lin, Yisen Wang

**<details><summary>Abstract**</summary>

Adversarial Training (AT) has become arguably the state-of-the-art algorithm for extracting robust features. However, researchers recently notice that AT suffers from severe robust overfitting problems, particularly after learning rate (LR) decay. In this paper, we explain this phenomenon by viewing adversarial training as a dynamic minimax game between the model trainer and the attacker. Specifically, we analyze how LR decay breaks the balance between the minimax game by empowering the trainer with a stronger memorization ability, and show such imbalance induces robust overfitting as a result of memorizing non-robust features. We validate this understanding with extensive experiments, and provide a holistic view of robust overfitting from the dynamics of both the two game players. This understanding further inspires us to alleviate robust overfitting by rebalancing the two players by either regularizing the trainer's capacity or improving the attack strength. Experiments show that the proposed ReBalanced Adversarial Training (ReBAT) can attain good robustness and does not suffer from robust overfitting even after very long training. Code is available at https://github.com/PKU-ML/ReBAT.</details>

### Balanced Training for Sparse GANs

**Authors:** Yite Wang, Jing Wu, NAIRA HOVAKIMYAN, Ruoyu Sun

**<details><summary>Abstract**</summary>

Over the past few years, there has been growing interest in developing larger and deeper neural networks, including deep generative models like generative adversarial networks (GANs). However, GANs typically come with high computational complexity,  leading researchers to explore methods for reducing the training and inference costs.  One such approach gaining popularity in supervised learning is dynamic sparse training (DST), which maintains good performance while enjoying excellent training efficiency.  Despite its potential benefits, applying DST to GANs presents challenges due to the adversarial nature of the training process. In this paper, we propose a novel metric called the balance ratio (BR) to study the balance between the sparse generator and discriminator. We also introduce a new method called balanced dynamic sparse training (ADAPT), which seeks to control the BR during GAN training to achieve a good trade-off between performance and computational cost. Our proposed method shows promising results on multiple datasets, demonstrating its effectiveness.</details>

### BasisFormer: Attention-based Time Series Forecasting with Learnable and Interpretable Basis

**Authors:** Zelin Ni, Hang Yu, Shizhan Liu, Jianguo Li, Weiyao Lin

**<details><summary>Abstract**</summary>

Bases have become an integral part of modern deep learning-based models for time series forecasting due to their ability to act as feature extractors or future references. To be effective, a basis must be tailored to the specific set of time series data and exhibit distinct correlation with each time series within the set. However, current state-of-the-art methods are limited in their ability to satisfy both of these requirements simultaneously. To address this challenge, we propose BasisFormer, an end-to-end time series forecasting architecture that leverages learnable and interpretable bases. This architecture comprises three components: First, we acquire bases through adaptive self-supervised learning, which treats the historical and future sections of the time series as two distinct views and employs contrastive learning. Next, we design a Coef module that calculates the similarity coefficients between the time series and bases in the historical view via bidirectional cross-attention. Finally, we present a Forecast module that selects and consolidates the bases in the future view based on the similarity coefficients, resulting in accurate future predictions. Through extensive experiments on six datasets, we demonstrate that BasisFormer outperforms previous state-of-the-art methods by 11.04% and 15.78% respectively for univariate and multivariate forecasting tasks. Code isavailable at: https://github.com/nzl5116190/Basisformer.</details>

### Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across Computer Vision Tasks

**Authors:** Micah Goldblum, Hossein Souri, Renkun Ni, Manli Shu, Viraj Prabhu, Gowthami Somepalli, Prithvijit Chattopadhyay, Mark Ibrahim, Adrien Bardes, Judy Hoffman, Rama Chellappa, Andrew Wilson, Tom Goldstein

**<details><summary>Abstract**</summary>

Neural network based computer vision systems are typically built on a backbone, a pretrained or randomly initialized feature extractor. Several years ago, the default option was an ImageNet-trained convolutional neural network. However, the recent past has seen the emergence of countless backbones pretrained using various algorithms and datasets. While this abundance of choice has led to performance increases for a range of systems, it is difficult for practitioners to make informed decisions about which backbone to choose. Battle of the Backbones (BoB) makes this choice easier by benchmarking a diverse suite of pretrained models, including vision-language models, those trained via self-supervised learning, and the Stable Diffusion backbone, across a diverse set of computer vision tasks ranging from classification to object detection to OOD generalization and more. Furthermore, BoB sheds light on promising directions for the research community to advance computer vision by illuminating strengths and weakness of existing approaches through a comprehensive analysis conducted on more than 1500 training runs. While vision transformers (ViTs) and self-supervised learning (SSL) are increasingly popular, we find that convolutional neural networks pretrained in a supervised fashion on large training sets still perform best on most tasks among the models we consider. Moreover, in apples-to-apples comparisons on the same architectures and similarly sized pretraining datasets, we find that SSL backbones are highly competitive, indicating that future works should perform SSL pretraining with advanced architectures and larger pretraining datasets. We release the raw results of our experiments along with code that allows researchers to put their own backbones through the gauntlet here: https://github.com/hsouri/Battle-of-the-Backbones.</details>

### BayesDAG: Gradient-Based Posterior Inference for Causal Discovery

**Authors:** Yashas Annadani, Nick Pawlowski, Joel Jennings, Stefan Bauer, Cheng Zhang, Wenbo Gong

**<details><summary>Abstract**</summary>

Bayesian causal discovery aims to infer the posterior distribution over causal models from observed data, quantifying epistemic uncertainty and benefiting downstream tasks. However, computational challenges arise due to joint inference over combinatorial space of Directed Acyclic Graphs (DAGs) and nonlinear functions. Despite recent progress towards efficient posterior inference over DAGs,  existing methods are either limited to variational inference on node permutation matrices for linear causal models, leading to compromised inference accuracy, or continuous relaxation of adjacency matrices constrained by a DAG regularizer, which cannot ensure resulting graphs are DAGs. In this work, we introduce a scalable Bayesian causal discovery framework based on a combination of stochastic gradient Markov Chain Monte Carlo (SG-MCMC) and Variational Inference (VI) that overcomes these limitations. Our approach directly samples DAGs from the posterior without requiring any DAG regularization, simultaneously draws function parameter samples and is applicable to both linear and nonlinear causal models. To enable our approach, we derive a novel equivalence to the permutation-based DAG learning, which opens up possibilities of using any relaxed gradient estimator defined over permutations. To our knowledge, this is the first framework applying gradient-based MCMC sampling for causal discovery. Empirical evaluation on synthetic and real-world datasets demonstrate our approach's effectiveness compared to state-of-the-art baselines.</details>

### BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset

**Authors:** Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, Yaodong Yang

**<details><summary>Abstract**</summary>

In this paper, we introduce the BeaverTails dataset, aimed at fostering research on safety alignment in large language models (LLMs). This dataset uniquely separates annotations of helpfulness and harmlessness for question-answering pairs, thus offering distinct perspectives on these crucial attributes. In total, we have gathered safety meta-labels for 333,963 question-answer (QA) pairs and 361,903 pairs of expert comparison data for both the helpfulness and harmlessness metrics. We further showcase applications of BeaverTails in content moderation and reinforcement learning with human feedback (RLHF), emphasizing its potential for practical safety measures in LLMs. We believe this dataset provides vital resources for the community, contributing towards the safe development and deployment of LLMs. Our project page is available at the following URL: https://sites.google.com/view/pku-beavertails.</details>

### Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback

**Authors:** Jangwon Kim, Hangyeol Kim, Jiwook Kang, Jongchan Baek, Soohee Han

**<details><summary>Abstract**</summary>

We present a novel actor-critic algorithm for an environment with delayed feedback, which addresses the state-space explosion problem of conventional approaches. Conventional approaches use an augmented state constructed from the last observed state and actions executed since visiting the last observed state. Using the augmented state space, the correct Markov decision process for delayed environments can be constructed; however, this causes the state space to explode as the number of delayed timesteps increases, leading to slow convergence. Our proposed algorithm, called Belief-Projection-Based Q-learning (BPQL), addresses the state-space explosion problem by evaluating the values of the critic for which the input state size is equal to the original state-space size rather than that of the augmented one. We compare BPQL to traditional approaches in continuous control tasks and demonstrate that it significantly outperforms other algorithms in terms of asymptotic performance and sample efficiency. We also show that BPQL solves long-delayed environments, which conventional approaches are unable to do.</details>

### BenchCLAMP: A Benchmark for Evaluating Language Models on Syntactic and Semantic Parsing

**Authors:** Subhro Roy, Samuel Thomson, Tongfei Chen, Richard Shin, Adam Pauls, Jason Eisner, Benjamin Van Durme

**<details><summary>Abstract**</summary>

Recent work has shown that generation from a prompted or fine-tuned language model can perform well at semantic parsing when the output is constrained to be a valid semantic representation. We introduce BenchCLAMP, a Benchmark to evaluate Constrained LAnguage Model Parsing, that includes context-free grammars for seven semantic parsing datasets and two syntactic parsing datasets with varied output meaning representations, as well as a constrained decoding interface to generate only valid outputs covered by these grammars. We provide low, medium, and high resource splits for each dataset, allowing accurate comparison of various language models under different data regimes. Our benchmark supports evaluation of language models using prompt-based learning as well as fine-tuning. We benchmark seven language models, including two GPT-3 variants available only through an API. Our experiments show that encoder-decoder pretrained language models can achieve similar performance or even surpass state-of-the-art methods for both syntactic and semantic parsing when the model output is constrained to be valid.</details>

### Benchmark of Machine Learning Force Fields for Semiconductor Simulations: Datasets, Metrics, and Comparative Analysis

**Authors:** Geonu Kim, Byunggook Na, Gunhee Kim, Hyuntae Cho, Seungjin Kang, Hee Sun Lee, Saerom Choi, Heejae Kim, Seungwon Lee, Yongdeok Kim

**<details><summary>Abstract**</summary>

As semiconductor devices become miniaturized and their structures become more complex, there is a growing need for large-scale atomic-level simulations as a less costly alternative to the trial-and-error approach during development.Although machine learning force fields (MLFFs) can meet the accuracy and scale requirements for such simulations, there are no open-access benchmarks for semiconductor materials.Hence, this study presents a comprehensive benchmark suite that consists of two semiconductor material datasets and ten MLFF models with six evaluation metrics. We select two important semiconductor thin-film materials silicon nitride and hafnium oxide, and generate their datasets using computationally expensive density functional theory simulations under various scenarios at a cost of 2.6k GPU days.Additionally, we provide a variety of architectures as baselines: descriptor-based fully connected neural networks and graph neural networks with rotational invariant or equivariant features.We assess not only the accuracy of energy and force predictions but also five additional simulation indicators to determine the practical applicability of MLFF models in molecular dynamics simulations.To facilitate further research, our benchmark suite is available at https://github.com/SAITPublic/MLFF-Framework.</details>

### Benchmarking Distribution Shift in Tabular Data with TableShift

**Authors:** Josh Gardner, Zoran Popovic, Ludwig Schmidt

**<details><summary>Abstract**</summary>

Robustness to distribution shift has become a growing concern for text and image models as they transition from research subjects to deployment in the real world. However, high-quality benchmarks for distribution shift in tabular machine learning tasks are still lacking despite the widespread real-world use of tabular data and differences in the models used for tabular data in comparison to text and images. As a consequence, the robustness of tabular models to distribution shift is poorly understood. To address this issue, we introduce TableShift, a distribution shift benchmark for tabular data. TableShift contains 15 binary classification tasks in total, each with an associated shift, and includes a diverse set of data sources, prediction targets, and distribution shifts. The benchmark covers domains including finance, education, public policy, healthcare, and civic participation, and is accessible using only a few lines of Python code via the TableShift API. We conduct a large-scale study comparing several state-of-the-art tabular data models alongside robust learning and domain generalization methods on the benchmark tasks. Our study demonstrates (1) a linear trend between in-distribution (ID) and out-of-distribution (OOD) accuracy; (2) domain robustness methods can reduce shift gaps but at the cost of reduced ID accuracy; (3) a strong relationship between shift gap (difference between ID and OOD performance) and shifts in the label distribution. The benchmark data, Python package, model implementations, and more information about TableShift are available at https://github.com/mlfoundations/tableshift and https://tableshift.org .</details>

### Benchmarking Encoder-Decoder Architectures for Biplanar X-ray to 3D Bone Shape Reconstruction

**Authors:** Mahesh Shakya, Bishesh Khanal

**<details><summary>Abstract**</summary>

Various deep learning models have been proposed for 3D bone shape reconstruction from two orthogonal (biplanar) X-ray images.However, it is unclear how these models compare against each other since they are evaluated on different anatomy, cohort and (often privately held) datasets.Moreover, the impact of the commonly optimized image-based segmentation metrics such as dice score on the estimation of clinical parameters relevant in 2D-3D bone shape reconstruction is not well known.To move closer toward clinical translation, we propose a benchmarking framework that evaluates tasks relevant to real-world clinical scenarios, including reconstruction of fractured bones, bones with implants, robustness to population shift, and error in estimating clinical parameters.Our open-source platform provides reference implementations of 8 models (many of whose implementations were not publicly available), APIs to easily collect and preprocess 6 public datasets, and the implementation of automatic clinical parameter and landmark extraction methods. We present an extensive evaluation of 8 2D-3D models on equal footing using 6 public datasets comprising images for four different anatomies.Our results show that attention-based methods that capture global spatial relationships tend to perform better across all anatomies and datasets; performance on clinically relevant subgroups may be overestimated without disaggregated reporting; ribs are substantially more difficult to reconstruct compared to femur, hip and spine; and the dice score improvement does not always bring corresponding improvement in the automatic estimation of clinically relevant parameters.</details>

### Benchmarking Foundation Models with Language-Model-as-an-Examiner

**Authors:** Yushi Bai, Jiahao Ying, Yixin Cao, Xin Lv, Yuze He, Xiaozhi Wang, Jifan Yu, Kaisheng Zeng, Yijia Xiao, Haozhe Lyu, Jiayin Zhang, Juanzi Li, Lei Hou

**<details><summary>Abstract**</summary>

Numerous benchmarks have been established to assess the performance of foundation models on open-ended question answering, which serves as a comprehensive test of a model's ability to understand and generate language in a manner similar to humans.Most of these works focus on proposing new datasets, however, we see two main issues within previous benchmarking pipelines, namely testing leakage and evaluation automation. In this paper, we propose a novel benchmarking framework, Language-Model-as-an-Examiner, where the LM serves as a knowledgeable examiner that formulates questions based on its knowledge and evaluates responses in a reference-free manner. Our framework allows for effortless extensibility as various LMs can be adopted as the examiner, and the questions can be constantly updated given more diverse trigger topics. For a more comprehensive and equitable evaluation, we devise three strategies: (1) We instruct the LM examiner to generate questions across a multitude of domains to probe for a broad acquisition, and raise follow-up questions to engage in a more in-depth assessment. (2) Upon evaluation, the examiner combines both scoring and ranking measurements, providing a reliable result as it aligns closely with human annotations. (3) We additionally propose a decentralized Peer-examination method to address the biases in a single examiner. Our data and benchmarking results are available at: http://lmexam.xlore.cn.</details>

### Benchmarking Large Language Models on CMExam - A comprehensive Chinese Medical Exam Dataset

**Authors:** Junling Liu, Peilin Zhou, Yining Hua, Dading Chong, Zhongyu Tian, Andrew Liu, Helin Wang, Chenyu You, Zhenhua Guo, LEI ZHU, Michael Lingzhi Li

**<details><summary>Abstract**</summary>

Recent advancements in large language models (LLMs) have transformed the field of question answering (QA). However, evaluating LLMs in the medical field is challenging due to the lack of standardized and comprehensive datasets. To address this gap, we introduce CMExam, sourced from the Chinese National Medical Licensing Examination. CMExam consists of 60K+ multiple-choice questions for standardized and objective evaluations, as well as solution explanations for model reasoning evaluation in an open-ended manner. For in-depth analyses of LLMs, we invited medical professionals to label five additional question-wise annotations, including disease groups, clinical departments, medical disciplines, areas of competency, and question difficulty levels. Alongside the dataset, we further conducted thorough experiments with representative LLMs and QA algorithms on CMExam. The results show that GPT-4 had the best accuracy of 61.6% and a weighted F1 score of 0.617. These results highlight a great disparity when compared to human accuracy, which stood at 71.6%. For explanation tasks, while LLMs could generate relevant reasoning and demonstrate improved performance after finetuning, they fall short of a desired standard, indicating ample room for improvement. To the best of our knowledge, CMExam is the first Chinese medical exam dataset to provide comprehensive medical annotations. The experiments and findings of LLM evaluation also provide valuable insights into the challenges and potential solutions in developing Chinese medical QA systems and LLM evaluation pipelines.</details>

### Benchmarking Robustness of Adaptation Methods on Pre-trained Vision-Language Models

**Authors:** Shuo Chen, Jindong Gu, Zhen Han, Yunpu Ma, Philip Torr, Volker Tresp

**<details><summary>Abstract**</summary>

Various adaptation methods, such as LoRA, prompts, and adapters, have been proposed to enhance the performance of pre-trained vision-language models in specific domains. As test samples in real-world applications usually differ from adaptation data, the robustness of these adaptation methods against distribution shifts are essential. In this study, we assess the robustness of 11 widely-used adaptation methods across 4 vision-language datasets under multimodal corruptions. Concretely, we introduce 7 benchmark datasets, including 96 visual and 87 textual corruptions, to investigate the robustness of different adaptation methods, the impact of available adaptation examples, and the influence of trainable parameter size during adaptation. Our analysis reveals that: 1) Adaptation methods are more sensitive to text corruptions than visual corruptions. 2) Full fine-tuning does not consistently provide the highest robustness; instead, adapters can achieve better robustness with comparable clean performance. 3) Contrary to expectations, our findings indicate that increasing the number of adaptation data and parameters does not guarantee enhanced robustness; instead, it results in even lower robustness. We hope this study could benefit future research in the development of robust multimodal adaptation methods. The benchmark, code, and dataset used in this study can be accessed at https://adarobustness.github.io.</details>

### Benchmarking Robustness to Adversarial Image Obfuscations

**Authors:** Florian Stimberg, Ayan Chakrabarti, Chun-Ta Lu, Hussein Hazimeh, Otilia Stretcu, Wei Qiao, Yintao Liu, Merve Kaya, Cyrus Rashtchian, Ariel Fuxman, Mehmet Tek, Sven Gowal

**<details><summary>Abstract**</summary>

Automated content filtering and moderation is an important tool that allows online platforms to build striving user communities that facilitate cooperation and prevent abuse. Unfortunately, resourceful actors try to bypass automated filters in a bid to post content that violate platform policies and codes of conduct. To reach this goal, these malicious actors may obfuscate policy violating images (e.g., overlay harmful images by carefully selected benign images or visual patterns) to prevent machine learning models from reaching the correct decision. In this paper, we invite researchers to tackle this specific issue and present a new image benchmark. This benchmark, based on ImageNet, simulates the type of obfuscations created by malicious actors. It goes beyond Image-Net-C and ImageNet-C-bar by proposing general, drastic, adversarial modifications that preserve the original content intent. It aims to tackle a more common adversarial threat than the one considered by lp-norm bounded adversaries. We evaluate 33 pretrained models on the benchmark and train models with different augmentations, architectures and training methods on subsets of the obfuscations to measure generalization. Our hope is that this benchmark will encourage researchers to test their models and methods and try to find new approaches that are more robust to these obfuscations.</details>

### Benchmarking and Analyzing 3D-aware Image Synthesis with a Modularized Codebase

**Authors:** Qiuyu Wang, Zifan Shi, Kecheng Zheng, Yinghao Xu, Sida Peng, Yujun Shen

**<details><summary>Abstract**</summary>

Despite the rapid advance of 3D-aware image synthesis, existing studies usually adopt a mixture of techniques and tricks, leaving it unclear how each part contributes to the final performance in terms of generality. Following the most popular and effective paradigm in this field, which incorporates a neural radiance field (NeRF) into the generator of a generative adversarial network (GAN), we builda well-structured codebase through modularizing the generation process. Such a design allows researchers to develop and replace each module independently, and hence offers an opportunity to fairly compare various approaches and recognize their contributions from the module perspective. The reproduction of a range of cutting-edge algorithms demonstrates the availability of our modularized codebase. We also perform a variety of in-depth analyses, such as the comparison across different types of point feature, the necessity of the tailing upsampler in the generator, the reliance on the camera pose prior, etc., which deepen our understanding of existing methods and point out some further directions of the research work. Code and models will be made publicly available to facilitate the development and evaluation of this field.</details>

### Beyond MLE: Convex Learning for Text Generation

**Authors:** Chenze Shao, Zhengrui Ma, Min Zhang, Yang Feng

**<details><summary>Abstract**</summary>

Maximum likelihood estimation (MLE) is a statistical method used to estimate the parameters of a probability distribution that best explain the observed data. In the context of text generation, MLE is often used to train generative language models, which can then be used to generate new text. However, we argue that MLE is not always necessary and optimal, especially for closed-ended text generation tasks like machine translation. In these tasks, the goal of model is to generate the most appropriate response, which does not necessarily require it to estimate the entire data distribution with MLE. To this end, we propose a novel class of training objectives based on convex functions, which enables text generation models to focus on highly probable outputs without having to estimate the entire data distribution. We investigate the theoretical properties of the optimal predicted distribution when applying convex functions to the loss, demonstrating that convex functions can sharpen the optimal distribution, thereby enabling the model to better capture outputs with high probabilities. Experiments on various text generation tasks and models show the effectiveness of our approach. It enables autoregressive models to bridge the gap between greedy and beam search, and facilitates the learning of non-autoregressive models with a maximum improvement of 9+ BLEU points. Moreover, our approach also exhibits significant impact on large language models (LLMs), substantially enhancing their generative capability on various tasks. Source code is available at \url{https://github.com/ictnlp/Convex-Learning}.</details>

### Beyond Unimodal: Generalising Neural Processes for Multimodal Uncertainty Estimation

**Authors:** Myong Chol Jung, He Zhao, Joanna Dipnall, Lan Du

**<details><summary>Abstract**</summary>

Uncertainty estimation is an important research area to make deep neural networks (DNNs) more trustworthy. While extensive research on uncertainty estimation has been conducted with unimodal data, uncertainty estimation for multimodal data remains a challenge. Neural processes (NPs) have been demonstrated to be an effective uncertainty estimation method for unimodal data by providing the reliability of Gaussian processes with efficient and powerful DNNs. While NPs hold significant potential for multimodal uncertainty estimation, the adaptation of NPs for multimodal data has not been carefully studied. To bridge this gap, we propose Multimodal Neural Processes (MNPs) by generalising NPs for multimodal uncertainty estimation. Based on the framework of NPs, MNPs consist of several novel and principled mechanisms tailored to the characteristics of multimodal data. In extensive empirical evaluation, our method achieves state-of-the-art multimodal uncertainty estimation performance, showing its appealing robustness against noisy samples and reliability in out-of-distribution detection with faster computation time compared to the current state-of-the-art multimodal uncertainty estimation method.</details>

### Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start

**Authors:** Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**<details><summary>Abstract**</summary>

We analyse a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, equilibrium models, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower-level problem, i.e. they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. However, there are situations, e.g., meta learning and equilibrium models, in which the warm-start procedure is not well-suited or ineffective. In this work we show that without warm-start, it is still possible to achieve order-wise (near) optimal sample complexity. In particular, we propose a simple method which uses (stochastic) fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Finally, compared to methods using warm-start, our approach yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates.</details>

### BioMassters: A Benchmark Dataset for Forest Biomass Estimation using Multi-modal Satellite Time-series

**Authors:** Andrea Nascetti, Ritu Yadav, Kirill Brodt, Qixun Qu, Hongwei Fan, Yuri Shendryk, Isha Shah, Christine Chung

**<details><summary>Abstract**</summary>

Above Ground Biomass is an important variable as forests play a crucial role in mitigating climate change as they act as an efficient, natural and cost-effective carbon sink. Traditional field and airborne LiDAR measurements have been proven to provide reliable estimations of forest biomass. Nevertheless, the use of these techniques at a large scale can be challenging and expensive. Satellite data have been widely used as a valuable tool in estimating biomass on a global scale. However, the full potential of dense multi-modal satellite time series data, in combination with modern deep learning approaches, has yet to be fully explored. The aim of the "BioMassters" data challenge and benchmark dataset is to investigate the potential of multi-modal satellite data (Sentinel-1 SAR and Sentinel-2 MSI) to estimate forest biomass at a large scale using the Finnish Forest Centre's open forest and nature airborne LiDAR data as a reference. The performance of the top three baseline models shows the potential of deep learning to produce accurate and higher-resolution biomass maps. Our benchmark dataset is publically available at https://huggingface.co/datasets/nascetti-a/BioMassters (doi:10.57967/hf/1009) and the implementation of the top three winning models are available at https://github.com/drivendataorg/the-biomassters.</details>

### Bitstream-Corrupted Video Recovery: A Novel Benchmark Dataset and Method

**Authors:** Tianyi Liu, Kejun Wu, Yi Wang, Wenyang Liu, Kim-Hui Yap, Lap-Pui Chau

**<details><summary>Abstract**</summary>

The past decade has witnessed great strides in video recovery by specialist technologies, like video inpainting, completion, and error concealment. However, they typically simulate the missing content by manual-designed error masks, thus failing to fill in the realistic video loss in video communication (e.g., telepresence, live streaming, and internet video) and multimedia forensics. To address this, we introduce the bitstream-corrupted video (BSCV) benchmark, the first benchmark dataset with more than 28,000 video clips, which can be used for bitstream-corrupted video recovery in the real world. The BSCV is a collection of 1) a proposed three-parameter corruption model for video bitstream, 2) a large-scale dataset containing rich error patterns, multiple corruption levels, and flexible dataset branches, and 3) a new video recovery framework that serves as a benchmark. We evaluate state-of-the-art video inpainting methods on the BSCV dataset, demonstrating existing approaches' limitations and our framework's advantages in solving the bitstream-corrupted video recovery problem. The benchmark and dataset are released at https://github.com/LIUTIGHE/BSCV-Dataset.</details>

### Boosting Spectral Clustering on Incomplete Data via Kernel Correction and Affinity Learning

**Authors:** Fangchen Yu, Runze Zhao, Zhan Shi, Yiwen Lu, Jicong Fan, Yicheng Zeng, Jianfeng Mao, Wenye Li

**<details><summary>Abstract**</summary>

Spectral clustering has gained popularity for clustering non-convex data due to its simplicity and effectiveness. It is essential to construct a similarity graph using a high-quality affinity measure that models the local neighborhood relations among the data samples. However, incomplete data can lead to inaccurate affinity measures, resulting in degraded clustering performance. To address these issues, we propose an imputation-free framework with two novel approaches to improve spectral clustering on incomplete data. Firstly, we introduce a new kernel correction method that enhances the quality of the kernel matrix estimated on incomplete data with a theoretical guarantee, benefiting classical spectral clustering on pre-defined kernels. Secondly, we develop a series of affinity learning methods that equip the self-expressive framework with $\ell_p$-norm to construct an intrinsic affinity matrix with an adaptive extension. Our methods outperform existing data imputation and distance calibration techniques on benchmark datasets, offering a promising solution to spectral clustering on incomplete data in various real-world applications.</details>

### Breadcrumbs to the Goal: Supervised Goal Selection from Human-in-the-Loop Feedback

**Authors:** Marcel Torne Villasevil, Max Balsells I Pamies, Zihan Wang, Samedh Desai, Tao Chen, Pulkit Agrawal, Abhishek Gupta

**<details><summary>Abstract**</summary>

Exploration and reward specification are fundamental and intertwined challenges for reinforcement learning. Solving sequential decision making tasks with a non-trivial element of exploration requires either specifying carefully designed reward functions or relying on indiscriminate, novelty seeking exploration bonuses. Human supervisors can provide effective guidance in the loop to direct the exploration process, but prior methods to leverage this guidance require constant synchronous high-quality human feedback, which is expensive and impractical to obtain. In this work, we propose a technique - Human Guided Exploration (HUGE), that is able to leverage low-quality feedback from non-expert users, which is infrequent, asynchronous and noisy, to guide exploration for reinforcement learning, without requiring careful reward specification. The key idea is to separate the challenges of directed exploration and policy learning - human feedback is used to direct exploration, while self-supervised policy learning is used to independently learn unbiased behaviors from the collected data. We show that this procedure can leverage noisy, asynchronous human feedback to learn tasks with no hand-crafted reward design or exploration bonuses. We show that HUGE is able to learn a variety of challenging multi-stage robotic navigation and manipulation tasks in simulation using crowdsourced feedback from non-expert users. Moreover, this paradigm can be scaled to learning directly on real-world robots.</details>

### [Oral] Bridging RL Theory and Practice with the Effective Horizon

**Authors:** Cassidy Laidlaw, Stuart J Russell, Anca Dragan

**Oral Presentation:** Th, Dec 14, 13:35 -- Oral 6B

**<details><summary>Abstract**</summary>

Deep reinforcement learning (RL) works impressively in some environments and fails catastrophically in others. Ideally, RL theory should be able to provide an understanding of why this is, i.e. bounds predictive of practical performance. Unfortunately, current theory does not quite have this ability. We compare standard deep RL algorithms to prior sample complexity bounds by introducing a new dataset, BRIDGE. It consists of 155 MDPs from common deep RL benchmarks, along with their corresponding tabular representations, which enables us to exactly compute instance-dependent bounds. We find that prior bounds do not correlate well with when deep RL succeeds vs. fails, but discover a surprising property that does. When actions with the highest Q-values under the *random* policy also have the highest Q-values under the *optimal* policy—i.e., when it is optimal to act greedily with respect to the random's policy Q function—deep RL tends to succeed; when they don't, deep RL tends to fail. We generalize this property into a new complexity measure of an MDP that we call the *effective horizon*, which roughly corresponds to how many steps of lookahead search would be needed in that MDP in order to identify the next optimal action, when leaf nodes are evaluated with random rollouts. Using BRIDGE, we show that the effective horizon-based bounds are more closely reflective of the empirical performance of PPO and DQN than prior sample complexity bounds across four metrics. We also show that, unlike existing bounds, the effective horizon can predict the effects of using reward shaping or a pre-trained exploration policy.</details>

### Building Socio-culturally Inclusive Stereotype Resources with Community Engagement

**Authors:** Sunipa Dev, Jaya Goyal, Dinesh Tewari, Shachi Dave, Vinodkumar Prabhakaran

**<details><summary>Abstract**</summary>

With rapid development and deployment of generative language models in global settings, there is an urgent need to also scale our measurements of harm, not just in the number and types of harms covered, but also how well they account for local cultural contexts, including marginalized identities and the social biases experienced by them.Current evaluation paradigms are limited in their abilities to address this, as they are not representative of diverse, locally situated but global, socio-cultural perspectives. It is imperative that our evaluation resources are enhanced and calibrated by including people and experiences from different cultures and societies worldwide, in order to prevent gross underestimations or skews in measurements of harm. In this work, we demonstrate a socio-culturally aware expansion of evaluation resources in the Indian societal context, specifically for the harm of stereotyping. We devise a community engaged effort to build a resource which contains stereotypes for axes of disparity that are uniquely present in India. The resultant resource increases the number of stereotypes known for and in the Indian context by over 1000 stereotypes across many unique identities. We also demonstrate the utility and effectiveness of such expanded resources for evaluations of language models.CONTENT WARNING: This paper contains examples of stereotypes that may be offensive.</details>

### Building the Bridge of Schrödinger: A Continuous Entropic Optimal Transport Benchmark

**Authors:** Nikita Gushchin, Alexander Kolesov, Petr Mokrov, Polina Karpikova, Andrei Spiridonov, Evgeny Burnaev, Alexander Korotin

**<details><summary>Abstract**</summary>

ver the last several years, there has been significant progress in developing neural solvers for the Schrödinger Bridge (SB) problem and applying them to generative modelling. This new research field is justifiably fruitful as it is interconnected with the practically well-performing diffusion models and theoretically grounded entropic optimal transport (EOT). Still, the area lacks non-trivial tests allowing a researcher to understand how well the methods solve SB or its equivalent continuous EOT problem. We fill this gap and propose a novel way to create pairs of probability distributions for which the ground truth OT solution is known by the construction. Our methodology is generic and works for a wide range of OT formulations, in particular, it covers the EOT which is equivalent to SB (the main interest of our study). This development allows us to create continuous benchmark distributions with the known EOT and SB solutions on high-dimensional spaces such as spaces of images. As an illustration, we use these benchmark pairs to test how well existing neural EOT/SB solvers actually compute the EOT solution. Our code for constructing benchmark pairs under different setups is available at: https://github.com/ngushchin/EntropicOTBenchmark</details>

### BuildingsBench: A Large-Scale Dataset of 900K Buildings and Benchmark for Short-Term Load Forecasting

**Authors:** Patrick Emami, Abhijeet Sahu, Peter Graf

**<details><summary>Abstract**</summary>

Short-term forecasting of residential and commercial building energy consumption is widely used in power systems and continues to grow in importance. Data-driven short-term load forecasting (STLF), although promising, has suffered from a lack of open, large-scale datasets with high building diversity. This has hindered exploring the pretrain-then-fine-tune paradigm for STLF. To help address this, we present BuildingsBench, which consists of: 1) Buildings-900K, a large-scale dataset of 900K simulated buildings representing the U.S. building stock; and 2) an evaluation platform with over 1,900 real residential and commercial buildings from 7 open datasets. BuildingsBench benchmarks two under-explored tasks: zero-shot STLF, where a pretrained model is evaluated on unseen buildings without fine-tuning, and transfer learning, where a pretrained model is fine-tuned on a target building. The main finding of our benchmark analysis is that synthetically pretrained models generalize surprisingly well to real commercial buildings. An exploration of the effect of increasing dataset size and diversity on zero-shot commercial building performance reveals a power-law with diminishing returns. We also show that fine-tuning pretrained models on real commercial and residential buildings improves performance for a majority of target buildings. We hope that BuildingsBench encourages and facilitates future research on generalizable STLF. All datasets and code can be accessed from https://github.com/NREL/BuildingsBench.</details>

### Bullying10K: A Large-Scale Neuromorphic Dataset towards Privacy-Preserving Bullying Recognition

**Authors:** Yiting Dong, Yang Li, Dongcheng Zhao, Guobin Shen, Yi Zeng

**<details><summary>Abstract**</summary>

The prevalence of violence in daily life poses significant threats to individuals' physical and mental well-being. Using surveillance cameras in public spaces has proven effective in proactively deterring and preventing such incidents. However, concerns regarding privacy invasion have emerged due to their widespread deployment.To address the problem, we leverage Dynamic Vision Sensors (DVS) cameras to detect violent incidents and preserve privacy since it captures pixel brightness variations instead of static imagery. We introduce the Bullying10K dataset, encompassing various actions, complex movements, and occlusions from real-life scenarios. It provides three benchmarks for evaluating different tasks: action recognition, temporal action localization, and pose estimation. With 10,000 event segments, totaling 12 billion events and 255 GB of data, Bullying10K contributes significantly by balancing violence detection and personal privacy persevering. And it also poses a challenge to the neuromorphic dataset. It will serve as a valuable resource for training and developing privacy-protecting video systems. The Bullying10K opens new possibilities for innovative approaches in these domains.</details>

### [Spotlight] Bypassing spike sorting: Density-based decoding using spike localization from dense multielectrode probes

**Authors:** Yizi Zhang, Tianxiao He, Julien Boussard, Charles Windolf, Olivier Winter, Eric Trautmann, Noam Roth, Hailey Barrell, Mark Churchland, Nicholas A Steinmetz, Erdem Varol, Cole Hurwitz, Liam Paninski

**<details><summary>Abstract**</summary>

Neural decoding and its applications to brain computer interfaces (BCI) are essential for understanding the association between neural activity and behavior. A prerequisite for many decoding approaches is spike sorting, the assignment of action potentials (spikes) to individual neurons. Current spike sorting algorithms, however, can be inaccurate and do not properly model uncertainty of spike assignments, therefore discarding information that could potentially improve decoding performance. Recent advances in high-density probes (e.g., Neuropixels) and computational methods now allow for extracting a rich set of spike features from unsorted data; these features can in turn be used to directly decode behavioral correlates. To this end, we propose a spike sorting-free decoding method that directly models the distribution of extracted spike features using a mixture of Gaussians (MoG) encoding the uncertainty of spike assignments, without aiming to solve the spike clustering problem explicitly. We allow the mixing proportion of the MoG to change over time in response to the behavior and develop variational inference methods to fit the resulting model and to perform decoding. We benchmark our method with an extensive suite of recordings from different animals and probe geometries, demonstrating that our proposed decoder can consistently outperform current methods based on thresholding (i.e. multi-unit activity) and spike sorting. Open source code is available at https://github.com/yzhang511/density_decoding.</details>

### CAPP-130: A Corpus of Chinese Application Privacy Policy Summarization and Interpretation

**Authors:** pengyun zhu, Long Wen, Jinfei Liu, Feng Xue, Jian Lou, Zhibo Wang, Kui Ren

**<details><summary>Abstract**</summary>

A privacy policy serves as an online internet protocol crafted by service providers, which details how service providers collect, process, store, manage, and use personal information when users engage with applications. However, these privacy policies are often filled with technobabble and legalese, making them "incomprehensible''. As a result, users often agree to all terms unknowingly, even some terms may conflict with the law, thereby posing a considerable risk to personal privacy information. One potential solution to alleviate this challenge is to automatically summarize privacy policies using NLP techniques. However, existing techniques primarily focus on extracting key sentences, resulting in comparatively shorter agreements, but failing to address the poor readability caused by the "incomprehensible'' of technobabble and legalese. Moreover, research on Chinese application privacy policy summarization is currently almost nonexistent, and there is a lack of a high-quality corpus suitable for addressing readability issues. To tackle these challenges, we introduce a fine-grained CAPP-130 corpus and a TCSI-pp framework. CAPP-130 contains 130 Chinese privacy policies from popular applications that have been carefully annotated and interpreted by legal experts, resulting in 52,489 annotations and 20,555 rewritten sentences. TCSI-pp first extracts sentences related to the topic specified by users and then uses a generative model to rewrite the sentences into comprehensible summarization. Built upon TSCI-pp, we construct a summarization tool TSCI-pp-zh by selecting RoBERTa from six classification models for sentence extraction and selecting mT5 from five generative models for sentence rewriting. Experimental results show that TCSI-pp-zh outperforms GPT-4 and other baselines in Chinese application privacy policy summarization, demonstrating exceptional readability and reliability. Our data, annotation guidelines, benchmark models, and source code are publicly available at https://github.com/EnlightenedAI/CAPP-130.</details>

### CARE-MI: Chinese Benchmark for Misinformation Evaluation in Maternity and Infant Care

**Authors:** Tong Xiang, Liangzhi Li, Wangyue Li, Mingbai Bai, Lu Wei, Bowen Wang, Noa Garcia

**<details><summary>Abstract**</summary>

The recent advances in natural language processing (NLP), have led to a new trend of applying large language models (LLMs) to real-world scenarios. While the latest LLMs are astonishingly fluent when interacting with humans, they suffer from the misinformation problem by unintentionally generating factually false statements. This can lead to harmful consequences, especially when produced within sensitive contexts, such as healthcare. Yet few previous works have focused on evaluating misinformation in the long-form (LF) generation of LLMs, especially for knowledge-intensive topics. Moreover, although LLMs have been shown to perform well in different languages, misinformation evaluation has been mostly conducted in English. To this end, we present a benchmark, CARE-MI, for evaluating LLM misinformation in: 1) a sensitive topic, specifically the maternity and infant care domain; and 2) a language other than English, namely Chinese. Most importantly, we provide an innovative paradigm for building LF generation evaluation benchmarks that can be transferred to other knowledge-intensive domains and low-resourced languages. Our proposed benchmark fills the gap between the extensive usage of LLMs and the lack of datasets for assessing the misinformation generated by these models. It contains 1,612 expert-checked questions, accompanied with human-selected references. Using our benchmark, we conduct extensive experiments and found that current Chinese LLMs are far from perfect in the topic of maternity and infant care. In an effort to minimize the reliance on human resources for performance evaluation, we offer off-the-shelf judgment models for automatically assessing the LF output of LLMs given benchmark questions. Moreover, we compare potential solutions for LF generation evaluation and provide insights for building better automated metrics.</details>

### CEIL: Generalized Contextual Imitation Learning

**Authors:** Jinxin Liu, Li He, Yachen Kang, Zifeng Zhuang, Donglin Wang, Huazhe Xu

**<details><summary>Abstract**</summary>

In this paper, we present ContExtual Imitation Learning (CEIL), a general and broadly applicable algorithm for imitation learning (IL). Inspired by the formulation of hindsight information matching, we derive CEIL by explicitly learning a hindsight embedding function together with a contextual policy using the hindsight embeddings. To achieve the expert matching objective for IL, we advocate for optimizing a contextual variable such that it biases the contextual policy towards mimicking expert behaviors. Beyond the typical learning from demonstrations (LfD) setting, CEIL is a generalist that can be effectively applied to multiple settings including: 1) learning from observations (LfO), 2) offline IL, 3) cross-domain IL (mismatched experts), and 4) one-shot IL settings. Empirically, we evaluate CEIL on the popular MuJoCo tasks (online) and the D4RL dataset (offline). Compared to prior state-of-the-art baselines, we show that CEIL is more sample-efficient in most online IL tasks and achieves better or competitive performances in offline tasks.</details>

### CHAMMI: A benchmark for channel-adaptive models in microscopy imaging

**Authors:** Zitong Sam Chen, Chau Pham, Siqi Wang, Michael Doron, Nikita Moshkov, Bryan Plummer, Juan C. Caicedo

**<details><summary>Abstract**</summary>

Most neural networks assume that input images have a fixed number of channels (three for RGB images). However, there are many settings where the number of channels may vary, such as microscopy images where the number of channels changes depending on instruments and experimental goals. Yet, there has not been a systemic attempt to create and evaluate neural networks that are invariant to the number and type of channels. As a result, trained models remain specific to individual studies and are hardly reusable for other microscopy settings. In this paper, we present a benchmark for investigating channel-adaptive models in microscopy imaging, which consists of 1) a dataset of varied-channel single-cell images, and 2) a biologically relevant evaluation framework. In addition, we adapted several existing techniques to create channel-adaptive models and compared their performance on this benchmark to fixed-channel, baseline models. We find that channel-adaptive models can generalize better to out-of-domain tasks and can be computationally efficient. We contribute a curated dataset and an evaluation API to facilitate objective comparisons in future research and applications.</details>

### CL-NeRF: Continual Learning of Neural Radiance Fields for Evolving Scene Representation

**Authors:** Xiuzhe Wu, Peng Dai, Weipeng DENG, Handi Chen, Yang Wu, Yan-Pei Cao, Ying Shan, Xiaojuan Qi

**<details><summary>Abstract**</summary>

Existing methods for adapting Neural Radiance Fields (NeRFs) to scene changes require extensive data capture and model retraining, which is both time-consuming and labor-intensive. In this paper, we tackle the challenge of efficiently adapting NeRFs to real-world scene changes over time using a few new images while retaining the memory of unaltered areas, focusing on the continual learning aspect of NeRFs. To this end, we propose CL-NeRF, which consists of two key components: a lightweight expert adaptor for adapting to new changes and evolving scene representations and a conflict-aware knowledge distillation learning objective for memorizing unchanged parts. We also present a new benchmark for evaluating Continual Learning of NeRFs with comprehensive metrics. Our extensive experiments demonstrate that CL-NeRF can synthesize high-quality novel views of both changed and unchanged regions with  high training efficiency, surpassing existing methods in terms of reducing forgetting and adapting to changes. Code and benchmark will be made available.</details>

### CMMA: Benchmarking Multi-Affection Detection in Chinese Multi-Modal Conversations

**Authors:** Yazhou Zhang, Yang Yu, Qing Guo, Benyou Wang, Dongming Zhao, Sagar Uprety, Dawei Song, Qiuchi Li, Jing Qin

**<details><summary>Abstract**</summary>

Human communication has a multi-modal and multi-affection nature. The inter-relatedness of different emotions and sentiments poses a challenge to jointly detect multiple human affections with multi-modal clues. Recent advances in this field employed multi-task learning paradigms to render the inter-relatedness across tasks, but the scarcity of publicly available resources sets a limit to the potential of works. To fill this gap, we build the first Chinese Multi-modal Multi-Affection conversation (CMMA) dataset, which contains 3,000 multi-party conversations and 21,795 multi-modal utterances collected from various styles of TV-series. CMMA contains a wide variety of affection labels, including sentiment, emotion, sarcasm and humor, as well as the novel inter-correlations values between certain pairs of tasks. Moreover, it provides the topic and speaker information in conversations, which promotes better modeling of conversational context. On the dataset, we empirically analyze the influence of different data modalities and conversational contexts on different affection analysis tasks, and exhibit the practical benefit of inter-task correlations. The full dataset will be publicly available for researchootnote{https://github.com/annoymity2022/Chinese-Dataset}</details>

### COCO-Counterfactuals: Automatically Constructed Counterfactual Examples for Image-Text Pairs

**Authors:** Tiep Le, VASUDEV LAL, Phillip Howard

**<details><summary>Abstract**</summary>

Counterfactual examples have proven to be valuable in the field of natural language processing (NLP) for both evaluating and improving the robustness of language models to spurious correlations in datasets. Despite their demonstrated utility for NLP, multimodal counterfactual examples have been relatively unexplored due to the difficulty of creating paired image-text data with minimal counterfactual changes. To address this challenge, we introduce a scalable framework for automatic generation of counterfactual examples using text-to-image diffusion models. We use our framework to create COCO-Counterfactuals, a multimodal counterfactual dataset of paired image and text captions based on the MS-COCO dataset. We validate the quality of COCO-Counterfactuals through human evaluations and show that existing multimodal models are challenged by our counterfactual image-text pairs. Additionally, we demonstrate the usefulness of COCO-Counterfactuals for improving out-of-domain generalization of multimodal vision-language models via training data augmentation. We make our code and the COCO-Counterfactuals dataset publicly available.</details>

### CQM: Curriculum Reinforcement Learning with a Quantized World Model

**Authors:** Seungjae Lee, Daesol Cho, Jonghae Park, H. Jin Kim

**<details><summary>Abstract**</summary>

Recent curriculum Reinforcement Learning (RL) has shown notable progress in solving complex tasks by proposing sequences of surrogate tasks. However, the previous approaches often face challenges when they generate curriculum goals in a high-dimensional space. Thus, they usually rely on manually specified goal spaces. To alleviate this limitation and improve the scalability of the curriculum, we propose a novel curriculum method that automatically defines the semantic goal space which contains vital information for the curriculum process, and suggests curriculum goals over it. To define the semantic goal space, our method discretizes continuous observations via vector quantized-variational autoencoders (VQ-VAE) and restores the temporal relations between the discretized observations by a graph. Concurrently, ours suggests uncertainty and temporal distance-aware curriculum goals that converges to the final goals over the automatically composed goal space. We demonstrate that the proposed method allows efficient explorations in an uninformed environment with raw goal examples only. Also, ours outperforms the state-of-the-art curriculum RL methods on data efficiency and performance, in various goal-reaching tasks even with ego-centric visual inputs.</details>

### CSMeD: Bridging the Dataset Gap in Automated Citation Screening for Systematic Literature Reviews

**Authors:** Wojciech Kusa, Oscar E. Mendoza, Matthias Samwald, Petr Knoth, Allan Hanbury

**<details><summary>Abstract**</summary>

Systematic literature reviews (SLRs) play an essential role in summarising, synthesising and validating scientific evidence. In recent years, there has been a growing interest in using machine learning techniques to automate the identification of relevant studies for SLRs. However, the lack of standardised evaluation datasets makes comparing the performance of such automated literature screening systems difficult. In this paper, we analyse the citation screening evaluation datasets, revealing that many of the available datasets are either too small, suffer from data leakage or have limited applicability to systems treating automated literature screening as a classification task, as opposed to, for example, a retrieval or question-answering task. To address these challenges, we introduce CSMED, a meta-dataset consolidating nine publicly released collections, providing unified access to 325 SLRs from the fields of medicine and computer science. CSMED serves as a comprehensive resource for training and evaluating the performance of automated citation screening models. Additionally, we introduce CSMED-FT, a new dataset designed explicitly for evaluating the full text publication screening task. To demonstrate the utility of CSMED, we conduct experiments and establish baselines on new datasets.</details>

### CaMP: Causal Multi-policy Planning for Interactive Navigation in  Multi-room Scenes

**Authors:** Xiaohan Wang, Yuehu Liu, Xinhang Song, Beibei Wang, Shuqiang Jiang

**<details><summary>Abstract**</summary>

Visual navigation has been widely studied under the assumption that there may be several clear routes to reach the goal. However, in more practical scenarios such as a house with several messy rooms, there may not. Interactive Navigation (InterNav) considers agents navigating to their goals more effectively with object interactions, posing new challenges of learning interaction dynamics and extra action space. Previous works learn single vision-to-action policy with the guidance of designed representations. However, the causality between actions and outcomes is prone to be confounded when the attributes of obstacles are diverse and hard to measure. Learning policy for long-term action planning in complex scenes also leads to extensive inefficient exploration. In this paper, we introduce a causal diagram of InterNav clarifying the confounding bias caused by obstacles. To address the problem, we propose a multi-policy model that enables the exploration of counterfactual interactions as well as reduces unnecessary exploration. We develop a large-scale dataset containing 600k task episodes in 12k multi-room scenes based on the ProcTHOR simulator and showcase the effectiveness of our method with the evaluations on our dataset.</details>

### Calibrating Neural Simulation-Based Inference with Differentiable Coverage Probability

**Authors:** Maciej Falkiewicz, Naoya Takeishi, Imahn Shekhzadeh, Antoine Wehenkel, Arnaud Delaunoy, Gilles Louppe, Alexandros Kalousis

**<details><summary>Abstract**</summary>

Bayesian inference allows expressing the uncertainty of posterior belief under a probabilistic model given prior information and the likelihood of the evidence. Predominantly, the likelihood function is only implicitly established by a simulator posing the need for simulation-based inference (SBI). However, the existing algorithms can yield overconfident posteriors (Hermans *et al.*, 2022) defeating the whole purpose of credibility if the uncertainty quantification is inaccurate. We propose to include a calibration term directly into the training objective of the neural model in selected amortized SBI techniques. By introducing a relaxation of the classical formulation of calibration error we enable end-to-end backpropagation. The proposed method is not tied to any particular neural model and brings moderate computational overhead compared to the profits it introduces. It is directly applicable to existing computational pipelines allowing reliable black-box posterior inference. We empirically show on six benchmark problems that the proposed method achieves competitive or better results in terms of coverage and expected posterior density than the previously existing approaches.</details>

### Calibrating “Cheap Signals” in Peer Review without a Prior

**Authors:** Yuxuan Lu, Yuqing Kong

**<details><summary>Abstract**</summary>

Peer review lies at the core of the academic process, but even well-intentioned reviewers can still provide noisy ratings. While ranking papers by average ratings may reduce noise, varying noise levels and systematic biases stemming from ``cheap'' signals (e.g. author identity, proof length) can lead to unfairness. Detecting and correcting bias is challenging, as ratings are subjective and unverifiable. Unlike previous works relying on prior knowledge or historical data, we propose a one-shot noise calibration process without any prior information. We ask reviewers to predict others' scores and use these predictions for calibration. Assuming reviewers adjust their predictions according to the noise, we demonstrate that the calibrated score results in a more robust ranking compared to average ratings, even with varying noise levels and biases.In detail, we show that the error probability of the calibrated score approaches zero as the number of reviewers increases and is significantly lower compared to average ratings when the number of reviewers is small.</details>

### Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs

**Authors:** Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, Xuanhe Zhou, Ma Chenhao, Guoliang Li, Kevin Chang, Fei Huang, Reynold Cheng, Yongbin Li

**<details><summary>Abstract**</summary>

Text-to-SQL parsing, which aims at converting natural language instructions into executable SQLs, has gained increasing attention in recent years. In particular, GPT-4 and Claude-2 have shown impressive results in this task. However, most of the prevalent benchmarks, i.e., Spider, and WikiSQL, focus on database schema with few rows of database contents leaving the gap between academic study and real-world applications. To mitigate this gap, we present BIRD, a BIg benchmark for laRge-scale Database grounded in text-to-SQL tasks, containing 12,751 pairs of text-to-SQL data and 95 databases with a total size of 33.4 GB, spanning 37 professional domains. Our emphasis on database values highlights the new challenges of dirty database contents, external knowledge between NL questions and database contents, and SQL efficiency, particularly in the context of massive databases. To solve these problems, text-to-SQL models must feature database value comprehension in addition to semantic parsing. The experimental results demonstrate the significance of database values in generating accurate text-to-SQLs for big databases. Furthermore, even the most popular and effective text-to-SQL models, i.e. GPT-4, only achieve 54.89% in execution accuracy, which is still far from the human result of 92.96%, proving that challenges still stand. We also provide an efficiency analysis to offer insights into generating text-to-efficient-SQLs that are beneficial to industries. We believe that BIRD will contribute to advancing real-world applications of text-to-SQL research.The leaderboard and source code are available: https://bird-bench.github.io/.</details>

### Can You Rely on Your Model Evaluation? Improving Model Evaluation with Synthetic Test Data

**Authors:** Boris van Breugel, Nabeel Seedat, Fergus Imrie, Mihaela van der Schaar

**<details><summary>Abstract**</summary>

Evaluating the performance of machine learning models on diverse and underrepresented subgroups is essential for ensuring fairness and reliability in real-world applications. However, accurately assessing model performance becomes challenging due to two main issues: (1) a scarcity of test data, especially for small subgroups, and (2) possible distributional shifts in the model's deployment setting, which may not align with the available test data.  In this work, we introduce 3S Testing, a deep generative modeling framework to facilitate model evaluation by generating synthetic test sets for small subgroups and simulating distributional shifts. Our experiments demonstrate that 3S-Testing outperforms traditional baselines---including real test data alone---in estimating model performance on minority subgroups and under plausible distributional shifts. In addition, 3S offers intervals around its performance estimates, exhibiting superior coverage of the ground truth compared to existing approaches.  Overall, these results raise the question of whether we need a paradigm shift away from limited real test data towards synthetic test data.</details>

### Canonical normalizing flows for manifold learning

**Authors:** Kyriakos Flouris, Ender Konukoglu

**<details><summary>Abstract**</summary>

Manifold learning flows are a class of generative modelling techniques that assume a low-dimensional manifold description of the data. The embedding of such a manifold into the high-dimensional space of the data is achieved via learnable invertible transformations. Therefore, once the manifold is properly aligned via a reconstruction loss, the probability density is tractable on the manifold and maximum likelihood can be used to optimize the network parameters. Naturally, the lower-dimensional representation of the data requires an injective-mapping. Recent approaches were able to enforce that the density aligns with the modelled manifold, while efficiently calculating the density volume-change term when embedding to the higher-dimensional space. However, unless the injective-mapping is analytically predefined, the learned manifold is not necessarily an \emph{efficient representation} of the data. Namely, the latent dimensions of such models frequently learn an entangled intrinsic basis, with degenerate information being stored in each dimension. Alternatively, if a locally orthogonal and/or sparse basis is to be learned, here coined canonical intrinsic basis, it can serve in learning a more compact latent space representation. Toward this end, we propose a canonical manifold learning flow method, where a novel optimization objective enforces the transformation matrix to have few prominent and non-degenerate basis functions. We demonstrate that by minimizing the off-diagonal manifold metric elements $\ell_1$-norm, we can achieve such a basis, which is simultaneously sparse and/or orthogonal. Canonical manifold flow yields a more efficient use of the latent space, automatically generating fewer prominent and distinct dimensions to represent data, and consequently a better approximation of target distributions than other manifold flow methods in most experiments we conducted, resulting in lower FID scores.</details>

### Characterization and Learning of Causal Graphs with Small Conditioning Sets

**Authors:** Murat Kocaoglu

**<details><summary>Abstract**</summary>

Constraint-based causal discovery algorithms learn part of the causal graph structure by systematically testing conditional independences  observed in the data. These algorithms, such as the PC algorithm and its variants, rely on graphical characterizations of the so-called equivalence class of causal graphs proposed by Pearl. However, constraint-based causal discovery algorithms struggle when data is limited since conditional independence tests quickly lose their statistical power, especially when the conditioning set is large. To address this, we propose using conditional independence tests where the size of the conditioning set is upper bounded by some integer k for robust causal discovery. The existing graphical characterizations of the equivalence classes of causal graphs are not applicable when we cannot leverage all the conditional independence statements. We first define the notion of k-Markov equivalence: Two causal graphs are k-Markov equivalent if they entail the same conditional independence constraints where the conditioning set size is upper bounded by k. We propose a novel representation that allows us to graphically characterize k-Markov equivalence between two causal graphs. We propose a sound constraint-based algorithm called the k-PC algorithm for learning this equivalence class. Finally, we conduct synthetic, and semi-synthetic experiments to demonstrate that the k-PC algorithm enables more robust causal discovery in the small sample regime compared to the baseline algorithms.</details>

### ClimateLearn: Benchmarking Machine Learning for Weather and Climate Modeling

**Authors:** Tung Nguyen, Jason Jewik, Hritik Bansal, Prakhar Sharma, Aditya Grover

**<details><summary>Abstract**</summary>

Modeling weather and climate is an essential endeavor to understand the near- and long-term impacts of climate change, as well as to inform technology and policymaking for adaptation and mitigation efforts. In recent years, there has been a surging interest in applying data-driven methods based on machine learning for solving core problems such as weather forecasting and climate downscaling. Despite promising results, much of this progress has been impaired due to the lack of large-scale, open-source efforts for reproducibility, resulting in the use of inconsistent or underspecified datasets, training setups, and evaluations by both domain scientists and artificial intelligence researchers. We introduce ClimateLearn, an open-source PyTorch library that vastly simplifies the training and evaluation of machine learning models for data-driven climate science. ClimateLearn consists of holistic pipelines for dataset processing (e.g., ERA5, CMIP6, PRISM), implementing state-of-the-art deep learning models (e.g., Transformers, ResNets), and quantitative and qualitative evaluation for standard weather and climate modeling tasks. We supplement these functionalities with extensive documentation, contribution guides, and quickstart tutorials to expand access and promote community growth. We have also performed comprehensive forecasting and downscaling experiments to showcase the capabilities and key features of our library. To our knowledge, ClimateLearn is the first large-scale, open-source effort for bridging research in weather and climate modeling with modern machine learning systems. Our library is available publicly at https://github.com/aditya-grover/climate-learn.</details>

### Closing the gap between the upper bound and lower bound of Adam's iteration complexity

**Authors:** Bohan Wang, Jingwen Fu, Huishuai Zhang, Nanning Zheng, Wei Chen

**<details><summary>Abstract**</summary>

Recently,  Arjevani et al. [1]  establish a lower bound of iteration complexity for the first-order optimization under an $L$-smooth condition and a bounded noise variance assumption. However, a thorough review of existing literature on Adam's convergence reveals a noticeable gap: none of them meet the above lower bound. In this paper, we close the gap by deriving a new convergence guarantee of Adam, with only an $L$-smooth condition and a bounded noise variance assumption. Our results remain valid across a broad spectrum of hyperparameters. Especially with properly chosen hyperparameters, we derive an upper bound of the iteration complexity of Adam and show that it meets the lower bound for first-order optimizers. To the best of our knowledge, this is the first to establish such a tight upper bound for Adam's convergence. Our proof utilizes novel techniques to handle the entanglement between momentum and adaptive learning rate and to convert the first-order term in the Descent Lemma to the gradient norm, which may be of independent interest.</details>

### Cocktail: Mixing Multi-Modality Control for Text-Conditional Image Generation

**Authors:** Minghui Hu, Jianbin Zheng, Daqing Liu, Chuanxia Zheng, Chaoyue Wang, Dacheng Tao, Tat-Jen Cham

**<details><summary>Abstract**</summary>

Text-conditional diffusion models are able to generate high-fidelity images with diverse contents.However, linguistic representations frequently exhibit ambiguous descriptions of the envisioned objective imagery, requiring the incorporation of additional control signals to bolster the efficacy of text-guided diffusion models. In this work, we propose Cocktail, a pipeline to mix various modalities into one embedding, amalgamated with a generalized ControlNet (gControlNet), a controllable normalisation (ControlNorm), and a spatial guidance sampling method, to actualize multi-modal and spatially-refined control for text-conditional diffusion models. Specifically, we introduce a hyper-network gControlNet, dedicated to the alignment and infusion of the control signals from disparate modalities into the pre-trained diffusion model. gControlNet is capable of accepting flexible modality signals, encompassing the simultaneous reception of any combination of modality signals, or the supplementary fusion of multiple modality signals. The control signals are then fused and injected into the backbone model according to our proposed ControlNorm.Furthermore, our advanced spatial guidance sampling methodology proficiently incorporates the control signal into the designated region, thereby circumventing the manifestation of undesired objects within the generated image.We demonstrate the results of our method in controlling various modalities, proving high-quality synthesis and fidelity to multiple external signals.</details>

### Collaboratively Learning Linear Models with Structured Missing Data

**Authors:** Chen Cheng, Gary Cheng, John Duchi

**<details><summary>Abstract**</summary>

We study the problem of collaboratively learning least squares estimates for $m$ agents. Each agent observes a different subset of the features---e.g., containing data collected from sensors of varying resolution. Our goal is to determine how to coordinate the agents in order to produce the best estimator for each agent. We propose a distributed, semi-supervised algorithm Collab, consisting of three steps: local training, aggregation, and distribution. Our procedure does not require communicating the labeled data, making it communication efficient and useful in settings where the labeled data is inaccessible. Despite this handicap, our procedure is nearly asymptotically, local-minimax optimal---even among estimators allowed to communicate the labeled data such as imputation methods. We test our method on US Census data. We also discuss generalizations of our method to non-Gaussian feature settings, non-linear settings, and Federated Learning.</details>

### Color Equivariant Convolutional Networks

**Authors:** Attila Lengyel, Ombretta Strafforello, Robert-Jan Bruintjes, Alexander Gielisse, Jan van Gemert

**<details><summary>Abstract**</summary>

Color is a crucial visual cue readily exploited by Convolutional Neural Networks (CNNs) for object recognition. However, CNNs struggle if there is data imbalance between color variations introduced by accidental recording conditions. Color invariance addresses this issue but does so at the cost of removing all color information, which sacrifices discriminative power. In this paper, we propose Color Equivariant Convolutions (CEConvs), a novel deep learning building block that enables shape feature sharing across the color spectrum while retaining important color information. We extend the notion of equivariance from geometric to photometric transformations by incorporating parameter sharing over hue-shifts in a neural network. We demonstrate the benefits of CEConvs in terms of downstream performance to various tasks and improved robustness to color changes, including train-test distribution shifts. Our approach can be seamlessly integrated into existing architectures, such as ResNets, and offers a promising solution for addressing color-based domain shifts in CNNs.</details>

### Combinatorial Optimization with Policy Adaptation using Latent Space Search

**Authors:** Felix Chalumeau, Shikha Surana, Clément Bonnet, Nathan Grinsztajn, Arnu Pretorius, Alexandre Laterre, Tom Barrett

**<details><summary>Abstract**</summary>

Combinatorial Optimization underpins many real-world applications and yet, designing performant algorithms to solve these complex, typically NP-hard, problems remains a significant research challenge. Reinforcement Learning (RL) provides a versatile framework for designing heuristics across a broad spectrum of problem domains. However, despite notable progress, RL has not yet supplanted industrial solvers as the go-to solution. Current approaches emphasize pre-training heuristics that construct solutions, but often rely on search procedures with limited variance, such as stochastically sampling numerous solutions from a single policy, or employing computationally expensive fine-tuning of the policy on individual problem instances. Building on the intuition that performant search at inference time should be anticipated during pre-training, we propose COMPASS, a novel RL approach that parameterizes a distribution of diverse and specialized policies conditioned on a continuous latent space. We evaluate COMPASS across three canonical problems - Travelling Salesman, Capacitated Vehicle Routing, and Job-Shop Scheduling - and demonstrate that our search strategy (i) outperforms state-of-the-art approaches in 9 out of 11 standard benchmarking tasks and (ii) generalizes better, surpassing all other approaches on a set of 18 procedurally transformed instance distributions.</details>

### Compact Neural Volumetric Video Representations with Dynamic Codebooks

**Authors:** Haoyu Guo, Sida Peng, Yunzhi Yan, Linzhan Mou, Yujun Shen, Hujun Bao, Xiaowei Zhou

**<details><summary>Abstract**</summary>

This paper addresses the challenge of representing high-fidelity volumetric videos with low storage cost. Some recent feature grid-based methods have shown superior performance of fast learning implicit neural representations from input 2D images. However, such explicit representations easily lead to large model sizes when modeling dynamic scenes. To solve this problem, our key idea is reducing the spatial and temporal redundancy of feature grids, which intrinsically exist due to the self-similarity of scenes. To this end, we propose a novel neural representation, named dynamic codebook, which first merges similar features for the model compression and then compensates for the potential decline in rendering quality by a set of dynamic codes. Experiments on the NHR and DyNeRF datasets demonstrate that the proposed approach achieves state-of-the-art rendering quality, while being able to achieve more storage efficiency. The source code is available at https://github.com/zju3dv/compact_vv.</details>

### Comparing Apples to Oranges: Learning Similarity Functions for Data Produced by Different Distributions

**Authors:** Leonidas Tsepenekas, Ivan Brugere, Freddy Lecue, Daniele Magazzeni

**<details><summary>Abstract**</summary>

Similarity functions measure how comparable pairs of elements are, and play a key role in a wide variety of applications, e.g., notions of Individual Fairness abiding by the seminal paradigm of Dwork et al., as well as Clustering problems. However, access to an accurate similarity function should not always be considered guaranteed, and this point was even raised by Dwork et al. For instance, it is reasonable to assume that when the elements to be compared are produced by different distributions, or in other words belong to different ``demographic'' groups, knowledge of their true similarity might be very difficult to obtain. In this work, we present an efficient sampling framework that learns these across-groups similarity functions, using only a limited amount of experts' feedback. We show analytical results with rigorous theoretical bounds, and empirically validate our algorithms via a large suite of experiments.</details>

### Complementary Benefits of Contrastive Learning and Self-Training Under Distribution Shift

**Authors:** Saurabh Garg, Amrith Setlur, Zachary Lipton, Sivaraman Balakrishnan, Virginia Smith, Aditi Raghunathan

**<details><summary>Abstract**</summary>

Self-training and contrastive learning have emerged as leading techniques for incorporating unlabeled data, both under distribution shift (unsupervised domain adaptation) and when it is absent (semi-supervised learning). However, despite the popularity and compatibility of these techniques, their efficacy in combination remains surprisingly unexplored. In this paper, we first undertake a systematic empirical investigation of this combination, finding (i) that in domain adaptation settings, self-training and contrastive learning offer significant complementary gains; and (ii) that in semi-supervised learning settings, surprisingly, the benefits are not synergistic. Across eight distribution shift datasets (e.g., BREEDs, WILDS), we demonstrate that the combined method obtains 3--8\% higher accuracy than either approach independently. Finally, we theoretically analyze these techniques in a simplified model of distribution shift demonstrating scenarios under which the features produced by contrastive learning can yield a good initialization for self-training to further amplify gains and achieve optimal performance, even when either method alone would fail.</details>

### Complexity of Derivative-Free Policy Optimization for Structured $\mathcal{H}_\infty$ Control

**Authors:** Xingang Guo, Darioush Keivan, Geir Dullerud, Peter Seiler, Bin Hu

**<details><summary>Abstract**</summary>

The applications of direct policy search in reinforcement learning and continuous control have received increasing attention.In this work, we present novel theoretical results on the complexity of derivative-free policy optimization on an important class of robust control tasks, namely the structured $H_\infty$ synthesis with static output feedback. Optimal $H_\infty$ synthesis under structural constraints leads to a constrained nonconvex nonsmooth problem and is typicallyaddressed using subgradient-based policy search techniques that are built upon the concept of Goldstein subdifferential or other notions of enlarged subdifferential.  In this paper, we study the complexity of finding $(\delta,\epsilon)$-stationary points for such nonsmooth robust control design tasks using policy optimization methods which can only access the zeroth-order oracle (i.e. the $H_\infty$ norm of the closed-loop system). First, we study the exact oracle setting and identify the coerciveness of the cost function to prove high-probability feasibility/complexity bounds for derivative-free policy optimization on this problem. Next, we derive a sample complexity result for the multi-input multi-output (MIMO)  $H_\infty$-norm estimation. We combine this with our analysis to obtain the first sample complexity of model-free, trajectory-based, zeroth-order policy optimization on finding $(\delta,\epsilon)$-stationary points for structured $H_\infty$ control. Numerical results are also provided to demonstrate our theory.</details>

### Compositional Sculpting of Iterative Generative Processes

**Authors:** Timur Garipov, Sebastiaan De Peuter, Ge Yang, Vikas Garg, Samuel Kaski, Tommi Jaakkola

**<details><summary>Abstract**</summary>

High training costs of generative models and the need to fine-tune them for specific tasks have created a strong interest in model reuse and composition.A key challenge in composing iterative generative processes, such as GFlowNets and diffusion models, is that to realize the desired target distribution, all steps of the generative process need to be coordinated, and satisfy delicate balance conditions.In this work, we propose Compositional Sculpting: a general approach for defining compositions of iterative generative processes. We then introduce a method for sampling from these compositions built on classifier guidance.We showcase ways to accomplish compositional sculpting in both GFlowNets and diffusion models. We highlight two binary operations $\unicode{x2014}$ the $\textit{harmonic mean}\unicode{x00A0}(p_1 \otimes p_2$) and the $\textit{contrast}\unicode{x00A0}(p_1 \,\unicode{x25D1}\,\, p_2$) between pairs, and the generalization of these operations to multiple component distributions.We offer empirical results on image and molecular generation tasks. Project codebase: https://github.com/timgaripov/compositional-sculpting.</details>

### Computing Optimal Equilibria and Mechanisms via Learning in Zero-Sum Extensive-Form Games

**Authors:** Brian Zhang, Gabriele Farina, Ioannis Anagnostides, Federico Cacciamani, Stephen McAleer, Andreas Haupt, Andrea Celli, Nicola Gatti, Vincent Conitzer, Tuomas Sandholm

**<details><summary>Abstract**</summary>

We introduce a new approach for computing optimal equilibria via learning in games. It applies to extensive-form settings with any number of players, including mechanism design, information design, and solution concepts such as correlated, communication, and certification equilibria. We observe that optimal equilibria are minimax equilibrium strategies of a player in an extensive-form zero-sum game. This reformulation allows to apply techniques for learning in zero-sum games, yielding the first learning dynamics that converge to optimal equilibria, not only in empirical averages, but also in iterates. We demonstrate the practical scalability and flexibility of our approach by attaining state-of-the-art performance in benchmark tabular games, and by computing an optimal mechanism for a sequential auction design problem using deep reinforcement learning.</details>

### Concentration analysis of multivariate elliptic diffusions

**Authors:** Lukas Trottner, Cathrine Aeckerle-Willems, Claudia Strauch

**<details><summary>Abstract**</summary>

We prove concentration inequalities and associated PAC bounds  for both continuous- and discrete-time additive functionals for possibly unbounded functions of multivariate, nonreversible diffusion processes. Our analysis relies on an approach via the Poisson equation allowing us to consider a very broad class of subexponentially ergodic, multivariate diffusion processes. These results add to existing concentration inequalities for additive functionals of diffusion processes which have so far been only available for either bounded functions or for unbounded functions of processes from a significantly smaller class. We demonstrate the power of these exponential inequalities by two examples of very different areas. Considering a possibly high-dimensional, parametric, nonlinear drift model under sparsity constraints we apply the continuous-time concentration results to validate the restricted eigenvalue condition for Lasso estimation, which is fundamental for the derivation of oracle inequalities. The results for discrete additive functionals are applied for an investigation of the unadjusted Langevin MCMC algorithm for sampling of moderately heavy tailed densities $\pi$. In particular, we provide PAC bounds for the sample Monte Carlo estimator of integrals $\pi(f)$ for polynomially growing functions $f$ that quantify sufficient  sample and step sizes for approximation within a prescribed margin with high probability.</details>

### Conditional Distribution Function Estimation Using Neural Networks for Censored and Uncensored Data

**Authors:** Bingqing Hu, Bin Nan

**<details><summary>Abstract**</summary>

Most work in neural networks focuses on estimating the conditional mean of a continuous response variable given a set of covariates. In this article, we consider estimating the conditional distribution function using neural networks for both censored and uncensored data. The algorithm is built upon the data structure particularly constructed for the Cox regression with time-dependent covariates. Without imposing any model assumptions, we consider a loss function that is based on the full likelihood where the conditional hazard function is the only unknown nonparametric parameter, for which unconstrained optimization methods can be applied. Through simulation studies, we show that the proposed method possesses desirable performance, whereas the partial likelihood method and the traditional neural networks with $L_2$ loss yields biased estimates when model assumptions are violated. We further illustrate the proposed method with several real-world data sets.</details>

### Conditional score-based diffusion models for Bayesian inference in infinite dimensions

**Authors:** Lorenzo Baldassari, Ali Siahkoohi, Josselin Garnier, Knut Solna, Maarten V. de Hoop

**<details><summary>Abstract**</summary>

Since their initial introduction, score-based diffusion models (SDMs) have been successfully applied to solve a variety of linear inverse problems in finite-dimensional vector spaces due to their ability to efficiently approximate the posterior distribution. However, using SDMs for inverse problems in infinite-dimensional function spaces has only been addressed recently, primarily through methods that learn the unconditional score. While this approach is advantageous for some inverse problems, it is mostly heuristic and involves numerous computationally costly forward operator evaluations during posterior sampling. To address these limitations, we propose a theoretically grounded method for sampling from the posterior of infinite-dimensional Bayesian linear inverse problems based on amortized conditional SDMs. In particular, we prove that one of the most successful approaches for estimating the conditional score in finite dimensions—the conditional denoising estimator—can also be applied in infinite dimensions. A significant part of our analysis is dedicated to demonstrating that extending infinite-dimensional SDMs to the conditional setting requires careful consideration, as the conditional score typically blows up for small times, contrarily to the unconditional score. We conclude by presenting stylized and large-scale numerical examples that validate our approach, offer additional insights, and demonstrate that our method enables large-scale, discretization-invariant Bayesian inference.</details>

### Conformal PID Control for Time Series Prediction

**Authors:** Anastasios Angelopoulos, Emmanuel Candes, Ryan Tibshirani

**<details><summary>Abstract**</summary>

We study the problem of uncertainty quantification for time series prediction, with the goal of providing easy-to-use algorithms with formal guarantees. The algorithms we present build upon ideas from conformal prediction and control theory, are able to prospectively model conformal scores in an online setting, and adapt to the presence of systematic errors due to seasonality, trends, and general distribution shifts. Our theory both simplifies and strengthens existing analyses in online conformal prediction. Experiments on 4-week-ahead forecasting of statewide COVID-19 death counts in the U.S. show an improvement in coverage over the ensemble forecaster used inofficial CDC communications. We also run experiments on predicting electricity demand, market returns, and temperature using autoregressive, Theta, Prophet, and Transformer models. We provide an extendable codebase for testing our methods and for the integration of new algorithms, data sets, and forecasting rules at [this link](http://github.com/aangelopoulos/conformal-time-series).</details>

### Consensus and Subjectivity of Skin Tone Annotation for ML Fairness

**Authors:** Candice Schumann, Gbolahan Olanubi, Auriel Wright, Ellis Monk, Courtney Heldreth, Susanna Ricco

**<details><summary>Abstract**</summary>

Understanding different human attributes and how they affect model behavior may become a standard need for all model creation and usage, from traditional computer vision tasks to the newest multimodal generative AI systems. In computer vision specifically, we have relied on datasets augmented with perceived attribute signals (eg, gender presentation, skin tone, and age) and benchmarks enabled by these datasets. Typically labels for these tasks come from human annotators. However, annotating attribute signals, especially skin tone, is a difficult and subjective task. Perceived skin tone is affected by technical factors, like lighting conditions, and social factors that shape an annotator's lived experience.This paper examines the subjectivity of skin tone annotation through a series of annotation experiments using the Monk Skin Tone (MST) scale~\cite{Monk2022Monk}, a small pool of professional photographers, and a much larger pool of trained crowdsourced annotators. Along with this study we release the Monk Skin Tone Examples (MST-E) dataset, containing 1515 images and 31 videos spread across the full MST scale. MST-E is designed to help train human annotators to annotate MST effectively.Our study shows that annotators can reliably annotate skin tone in a way that aligns with an expert in the MST scale, even under challenging environmental conditions. We also find evidence that annotators from different geographic regions rely on different mental models of MST categories resulting in annotations that systematically vary across regions. Given this, we advise practitioners to use a diverse set of annotators and a higher replication count for each image when annotating skin tone for fairness research.</details>

### Consistent Aggregation of Objectives with Diverse Time Preferences Requires Non-Markovian Rewards

**Authors:** Silviu Pitis

**<details><summary>Abstract**</summary>

As the capabilities of artificial agents improve, they are being increasingly deployed to service multiple diverse objectives and stakeholders. However, the composition of these objectives is often performed ad hoc, with no clear justification. This paper takes a normative approach to multi-objective agency: from a set of intuitively appealing axioms, it is shown that Markovian aggregation of Markovian reward functions is not possible when the time preference (discount factor) for each objective may vary. It follows that optimal multi-objective agents must admit rewards that are non-Markovian with respect to the individual objectives. To this end, a practical non-Markovian aggregation scheme is proposed, which overcomes the impossibility with only one additional parameter for each objective. This work offers new insights into sequential, multi-objective agency and intertemporal choice, and has practical implications for the design of AI systems deployed to serve multiple generations of principals with varying time preference.</details>

### ContinuAR: Continuous Autoregression For Infinite-Fidelity Fusion

**Authors:** WEI XING, Yuxin Wang, Zheng Xing

**<details><summary>Abstract**</summary>

Multi-fidelity fusion has become an important surrogate technique, which provides insights into expensive computer simulations and effectively improves decision-making, e.g., optimization, with less computational cost. Multi-fidelity fusion is much more computationally efficient compared to traditional single-fidelity surrogates. Despite the fast advancement of multi-fidelity fusion techniques, they lack a systematic framework to make use of the fidelity indicator, deal with high-dimensional and arbitrary data structure, and scale well to infinite-fidelity problems. In this work, we first generalize the popular autoregression (AR) to derive a novel linear fidelity differential equation (FiDE), paving the way to tractable infinite-fidelity fusion. We generalize FiDE to a high-dimensional system, which also provides a unifying framework to seemly bridge the gap between many multi- and single-fidelity GP-based models. We then propose ContinuAR, a rank-1 approximation solution to FiDEs, which is tractable to train, compatible with arbitrary multi-fidelity data structure, linearly scalable to the output dimension, and most importantly, delivers consistent SOTA performance with a significant margin over the baseline methods. Compared to the SOTA infinite-fidelity fusion, IFC, ContinuAR achieves up to 4x improvement in accuracy and 62,500x speedup in training time.</details>

### [Spotlight] Convergence of Adam Under Relaxed Assumptions

**Authors:** Haochuan Li, Alexander Rakhlin, Ali Jadbabaie

**<details><summary>Abstract**</summary>

In this paper, we provide a rigorous proof of convergence of the Adaptive Moment Estimate (Adam) algorithm for a wide class of optimization objectives. Despite the popularity and efficiency of the Adam algorithm in training deep neural networks, its theoretical properties are not yet fully understood, and existing convergence proofs require unrealistically strong assumptions, such as globally bounded gradients, to show the convergence to stationary points. In this paper, we show that Adam provably converges to $\epsilon$-stationary points with $\mathcal{O}(\epsilon^{-4})$ gradient complexity under far more realistic conditions. The key to our analysis is a new proof of boundedness of gradients along the optimization trajectory of Adam, under a generalized smoothness assumption according to which the local smoothness (i.e., Hessian norm when it exists) is bounded by a sub-quadratic function of the gradient norm. Moreover, we propose a variance-reduced version of Adam with an accelerated gradient complexity of $\mathcal{O}(\epsilon^{-3})$.</details>

### Counting Distinct Elements Under Person-Level Differential Privacy

**Authors:** Thomas Steinke, Alexander Knop

**<details><summary>Abstract**</summary>

We study the problem of counting the number of distinct elements in a dataset subject to the constraint of differential privacy. We consider the challenging setting of person-level DP (a.k.a. user-level DP) where each person may contribute an unbounded number of items and hence the sensitivity is unbounded.Our approach is to compute a bounded-sensitivity version of this query, which reduces to solving a max-flow problem. The sensitivity bound is optimized to balance the noise we must add to privatize the answer against the error of the approximation of the bounded-sensitivity query to the true number of unique elements.</details>

### Covariance-adaptive best arm identification

**Authors:** El Mehdi Saad, Gilles Blanchard, Nicolas Verzelen

**<details><summary>Abstract**</summary>

We consider the problem of best arm identification in the multi-armed bandit model, under fixed confidence. Given a confidence input $\delta$, the goal is to identify the arm with the highest mean reward with a probability of at least $1 - \delta$, while minimizing the number of arm pulls. While the literature provides solutions to this problem under the assumption of independent arms distributions, we propose a more flexible scenario where arms can be dependent and rewards can be sampled simultaneously. This framework allows the learner to estimate the covariance among the arms distributions, enabling a more efficient identification of the best arm. The relaxed setting we propose is relevant in various applications, such as clinical trials, where similarities between patients or drugs suggest underlying correlations in the outcomes. We introduce new algorithms that adapt to the unknown covariance of the arms and demonstrate through theoretical guarantees that substantial improvement can be achieved over the standard setting. Additionally, we provide new lower bounds for the relaxed setting and present numerical simulations that support their theoretical findings.</details>

### Credal Marginal MAP

**Authors:** Radu Marinescu, Debarun Bhattacharjya, Junkyu Lee, Fabio Cozman, Alexander Gray

**<details><summary>Abstract**</summary>

Credal networks extend Bayesian networks to allow for imprecision in probability values. Marginal MAP is a widely applicable mixed inference task that identifies the most likely assignment for a subset of variables (called MAP variables). However, the  task is extremely difficult to solve in credal networks particularly because the evaluation of each complete MAP assignment involves exact likelihood computations (combinatorial sums) over the vertices of a complex joint credal set representing the space of all possible marginal distributions of the MAP variables. In this paper, we explore Credal Marginal MAP inference and develop new exact methods based on variable elimination and depth-first search as well as several approximation schemes based on the mini-bucket partitioning and stochastic local search. An extensive empirical evaluation demonstrates the effectiveness of our new methods on random as well as real-world benchmark problems.</details>

### CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement

**Authors:** Qihe Huang, Lei Shen, Ruixin Zhang, Shouhong Ding, Binwu Wang, Zhengyang Zhou, Yang Wang

**<details><summary>Abstract**</summary>

Recently, multivariate time series (MTS) forecasting techniques have seen rapid development and widespread applications across various fields. Transformer-based and GNN-based methods have shown promising potential due to their strong ability to model interaction of time and variables. However, by conducting a comprehensive analysis of the real-world data, we observe that the temporal fluctuations and heterogeneity between variables are not well handled by existing methods. To address the above issues, we propose CrossGNN, a linear complexity GNN model to refine the cross-scale and cross-variable interaction for MTS. To deal with the unexpected noise in time dimension, an adaptive multi-scale identifier (AMSI) is leveraged to construct multi-scale time series with reduced noise. A Cross-Scale GNN is proposed to extract the scales with clearer trend and weaker noise. Cross-Variable GNN is proposed to utilize the homogeneity and heterogeneity between different variables. By simultaneously focusing on edges with higher saliency scores and constraining those edges with lower scores, the time and space complexity (i.e., $O(L)$) of CrossGNN can be linear with the input sequence length $L$. Extensive experimental results on 8 real-world MTS datasets demonstrate the effectiveness of CrossGNN compared with state-of-the-art methods.</details>

### [Spotlight] Curriculum Learning With Infant Egocentric Videos

**Authors:** Saber Sheybani, Himanshu Hansaria, Justin Wood, Linda Smith, Zoran Tiganj

**<details><summary>Abstract**</summary>

Infants possess a remarkable ability to rapidly learn and process visual inputs. As an infant's mobility increases, so does the variety and dynamics of their visual inputs. Is this change in the properties of the visual inputs beneficial or even critical for the proper development of the visual system? To address this question, we used video recordings from infants wearing head-mounted cameras to train a variety of self-supervised learning models. Critically, we separated the infant data by age group and evaluated the importance of training with a curriculum aligned with developmental order. We found that initiating learning with the data from the youngest age group provided the strongest learning signal and led to the best learning outcomes in terms of downstream task performance. We then showed that the benefits of the data from the youngest age group are due to the slowness and simplicity of the visual experience. The results provide strong empirical evidence for the importance of the properties of the early infant experience and developmental progression in training. More broadly, our approach and findings take a noteworthy step towards reverse engineering the learning mechanisms in newborn brains using image-computable models from artificial intelligence.</details>

### CycleNet: Rethinking Cycle Consistency in Text-Guided Diffusion for Image Manipulation

**Authors:** Sihan Xu, Ziqiao Ma, Yidong Huang, Honglak Lee, Joyce Chai

**<details><summary>Abstract**</summary>

Diffusion models (DMs) have enabled breakthroughs in image synthesis tasks but lack an intuitive interface for consistent image-to-image (I2I) translation. Various methods have been explored to address this issue, including mask-based methods, attention-based methods, and image-conditioning. However, it remains a critical challenge to enable unpaired I2I translation with pre-trained DMs while maintaining satisfying consistency. This paper introduces Cyclenet, a novel but simple method that incorporates cycle consistency into DMs to regularize image manipulation. We validate Cyclenet on unpaired I2I tasks of different granularities. Besides the scene and object level translation, we additionally contribute a multi-domain I2I translation dataset to study the physical state changes of objects. Our empirical studies show that Cyclenet is superior in translation consistency and quality, and can generate high-quality images for out-of-domain distributions with a simple change of the textual prompt. Cyclenet is a practical framework, which is robust even with very limited training data (around 2k) and requires minimal computational resources (1 GPU) to train. Project homepage: https://cyclenetweb.github.io/</details>

### D$^2$CSG: Unsupervised Learning of Compact CSG Trees with Dual Complements and Dropouts

**Authors:** Fenggen Yu, Qimin Chen, Maham Tanveer, Ali Mahdavi Amiri, Hao Zhang

**<details><summary>Abstract**</summary>

We present D$^2$CSG, a neural model composed of two dual and complementary network branches, with dropouts, for unsupervised learning of compact constructive solid geometry (CSG) representations of 3D CAD shapes. Our network is trained to reconstruct a 3D shape by a fixed-order assembly of quadric primitives, with both branches producing a union of primitive intersections or inverses. A key difference between D$^2$CSG and all prior neural CSG models is its dedicated residual branch to assemble the potentially complex shape complement, which is subtracted from an overall shape modeled by the cover branch. With the shape complements, our network is provably general, while the weight dropout further improves compactness of the CSG tree by removing redundant primitives. We demonstrate both quantitatively and qualitatively that D$^2$CSG produces compact CSG reconstructions with superior quality and more natural primitives than all existing alternatives, especially over complex and high-genus CAD shapes.</details>

### D-Separation for Causal Self-Explanation

**Authors:** Wei Liu, Jun Wang, Haozhao Wang, Ruixuan Li, Zhiying Deng, YuanKai Zhang, Yang Qiu

**<details><summary>Abstract**</summary>

Rationalization aims to strengthen the interpretability of NLP models by extracting a subset of human-intelligible pieces of their inputting texts. Conventional works generally employ the maximum mutual information (MMI) criterion to find the rationale that is most indicative of the target label. However, this criterion can be influenced by spurious features that correlate with the causal rationale or the target label. Instead of attempting to rectify the issues of the MMI criterion, we propose a novel criterion to uncover the causal rationale, termed the Minimum Conditional Dependence (MCD) criterion, which is grounded on our finding that the non-causal features and the target label are \emph{d-separated} by the causal rationale. By minimizing the dependence between the non-selected parts of the input and the target label conditioned on the selected rationale candidate, all the causes of the label are compelled to be selected. In this study, we employ a simple and practical measure for dependence, specifically the KL-divergence, to validate our proposed MCD criterion. Empirically, we demonstrate that MCD improves the F1 score by up to 13.7% compared to previous state-of-the-art MMI-based methods.Our code is in an anonymous repository: https://anonymous.4open.science/r/MCD-CE88.</details>

### D4: Improving LLM Pretraining via Document De-Duplication and Diversification

**Authors:** Kushal Tirumala, Daniel Simig, Armen Aghajanyan, Ari Morcos

**<details><summary>Abstract**</summary>

Over recent years, an increasing amount of compute and data has been poured into training large language models (LLMs), usually by doing one-pass learning on as many tokens as possible randomly selected from large-scale web corpora. While training on ever-larger portions of the internet leads to consistent performance improvements, the size of these improvements diminishes with scale, and there has been little work exploring the effect of data selection on pre-training and downstream performance beyond simple de-duplication methods such as MinHash. Here, we show that careful data selection (on top of de-duplicated data) via pre-trained model embeddings can speed up training (20% efficiency gains) and improves average downstream accuracy on 16 NLP tasks (up to 2%) at the 6.7B model scale. Furthermore, we show that repeating data intelligently consistently outperforms baseline training (while repeating random data performs worse than baseline training). Our results indicate that clever data selection can significantly improve LLM pre-training, calls into question the common practice of training for a single epoch on as much data as possible, and demonstrates a path to keep improving our models past the limits of randomly sampling web data.</details>

### DICES Dataset: Diversity in Conversational AI Evaluation for Safety

**Authors:** Lora Aroyo, Alex Taylor, Mark Díaz, Christopher Homan, Alicia Parrish, Gregory Serapio-García, Vinodkumar Prabhakaran, Ding Wang

**<details><summary>Abstract**</summary>

Machine learning approaches often require training and evaluation datasets with a clear separation between positive and negative examples. This requirement overly simplifies the natural subjectivity present in many tasks, and obscures the inherent diversity in human perceptions and opinions about many content items. Preserving the variance in content and diversity in human perceptions in datasets is often quite expensive and laborious. This is especially troubling when building safety datasets for conversational AI systems, as safety is socio-culturally situated in this context. To demonstrate this crucial aspect of conversational AI safety, and to facilitate in-depth model performance analyses, we introduce the DICES (Diversity In Conversational AI Evaluation for Safety) dataset that contains fine-grained demographics information about raters, high replication of ratings per item to ensure statistical power for analyses, and encodes rater votes as distributions across different demographics to allow for in-depth explorations of different aggregation strategies. The DICES dataset enables the observation and measurement of variance, ambiguity, and diversity in the context of safety for conversational AI. We further describe a set of metrics that show how rater diversity influences safety perception across different geographic regions, ethnicity groups, age groups, and genders. The goal of the DICES dataset is to be used as a shared resource and benchmark that respects diverse perspectives during safety evaluation of conversational AI systems.</details>

### DISCO-10M: A Large-Scale Music Dataset

**Authors:** Luca Lanzendörfer, Florian Grötschla, Emil Funke, Roger Wattenhofer

**<details><summary>Abstract**</summary>

Music datasets play a crucial role in advancing research in machine learning for music. However, existing music datasets suffer from limited size, accessibility, and lack of audio resources. To address these shortcomings, we present DISCO-10M, a novel and extensive music dataset that surpasses the largest previously available music dataset by an order of magnitude. To ensure high-quality data, we implement a multi-stage filtering process. This process incorporates similarities based on textual descriptions and audio embeddings. Moreover, we provide precomputed CLAP embeddings alongside DISCO-10M, facilitating direct application on various downstream tasks. These embeddings enable efficient exploration of machine learning applications on the provided data. With DISCO-10M, we aim to democratize and facilitate new research to help advance the development of novel machine learning models for music: https://huggingface.co/DISCOX</details>

### DISCOVER: Making Vision Networks Interpretable via Competition and Dissection

**Authors:** Konstantinos Panousis, Sotirios Chatzis

**<details><summary>Abstract**</summary>

Modern deep networks are highly complex and their inferential outcome very hard to interpret. This is a serious obstacle to their transparent deployment in safety-critical or bias-aware applications. This work contributes to *post-hoc* interpretability, and specifically Network Dissection. Our goal is to present a framework that makes it easier to *discover* the individual functionality of each neuron in a network trained on a vision task; discovery is performed in terms of textual description generation. To achieve this objective, we leverage: (i) recent advances in multimodal vision-text models and (ii) network layers founded upon the novel concept of stochastic local competition between linear units. In this setting, only a *small subset* of layer neurons are activated *for a given input*, leading to extremely high activation sparsity (as low as only $\approx 4\%$). Crucially, our proposed method infers (sparse) neuron activation patterns that enables the neurons to activate/specialize to inputs with specific characteristics, diversifying their individual functionality. This capacity of our method supercharges the potential of dissection processes: human understandable descriptions are generated only for the very few active neurons, thus facilitating the direct investigation of the network's decision process. As we experimentally show, our approach: (i) yields Vision Networks that retain or improve classification performance, and (ii) realizes a principled framework for text-based description and examination of the generated neuronal representations.</details>

### Data Portraits: Recording Foundation Model Training Data

**Authors:** Marc Marone, Benjamin Van Durme

**<details><summary>Abstract**</summary>

Foundation models are trained on increasingly immense and opaque datasets. Even while these models are now key in AI system building, it can be difficult to answer the straightforward question: has the model already encountered a given example during training? We therefore propose a widespread adoption of Data Portraits: artifacts that record training data and allow for downstream inspection. First we outline the properties of such an artifact and discuss how existing solutions can be used to increase transparency. We then propose and implement a solution based on data sketching, stressing fast and space efficient querying. Using our tools, we document a popular language modeling corpus (The Pile) and a recently released code modeling dataset (The Stack). We show that our solution enables answering questions about test set leakage and model plagiarism. Our tool is lightweight and fast, costing only 3% of the dataset size in overhead. We release a live interface of our tools at https://dataportraits.org/ and call on dataset and model creators to release Data Portraits as a complement to current documentation practices.</details>

### Data Pruning via Moving-one-Sample-out

**Authors:** Haoru Tan, Sitong Wu, Fei Du, Yukang Chen, Zhibin Wang, Fan Wang, Xiaojuan Qi

**<details><summary>Abstract**</summary>

In this paper, we propose a novel data-pruning approach called moving-one-sample-out (MoSo), which aims to identify and remove the least informative samples from the training set. The core insight behind MoSo is to determine the importance of each sample by assessing its impact on the optimal empirical risk. This is achieved by measuring the extent to which the empirical risk changes when a particular sample is excluded from the training set. Instead of using the computationally expensive leaving-one-out-retraining procedure, we propose an efficient first-order approximator that only requires gradient information from different training stages. The key idea behind our approximation is that samples with gradients that are consistently aligned with the average gradient of the training set are more informative and should receive higher scores, which could be intuitively understood as follows: if the gradient from a specific sample is consistent with the average gradient vector, it implies that optimizing the network using the sample will yield a similar effect on all remaining samples.  Experimental results demonstrate that MoSo effectively mitigates severe performance degradation at high pruning ratios and achieves satisfactory performance across various settings. Experimental results demonstrate that MoSo effectively mitigates severe performance degradation at high pruning ratios and outperforms state-of-the-art methods by a large margin across various settings.</details>

### Data Selection for Language Models via Importance Resampling

**Authors:** Sang Michael Xie, Shibani Santurkar, Tengyu Ma, Percy Liang

**<details><summary>Abstract**</summary>

Selecting a suitable pretraining dataset is crucial for both general-domain (e.g., GPT-3) and domain-specific (e.g., Codex) language models (LMs). We formalize this problem as selecting a subset of a large raw unlabeled dataset to match a desired target distribution given some unlabeled target samples. Due to the large scale and dimensionality of the raw text data, existing methods use simple heuristics or use experts to manually curate data. Instead, we extend the classic importance resampling approach used in low-dimensions for LM data selection. We propose Data Selection with Importance Resampling (DSIR), an efficient and scalable framework that estimates importance weights in a reduced feature space for tractability and selects data with importance resampling according to these weights. To determine an appropriate feature space, we show that KL reduction, a data metric that measures the proximity between selected pretraining data and the target in a feature space, has high correlation with average downstream accuracy (r=0.89) when computed with simple n-gram features. This motivates our instantiation of DSIR using n-gram features. When performing continued pretraining towards a specific domain, DSIR performs comparably to expert curation across 8 target distributions. When pretraining general-domain models (target is Wikipedia + books), DSIR improves over random selection and heuristic filtering baselines by 2-2.5% on the GLUE benchmark.</details>

### Datasets and Benchmarks for Nanophotonic Structure and Parametric Design Simulations

**Authors:** Jungtaek Kim, Mingxuan Li, Oliver Hinder, Paul Leu

**<details><summary>Abstract**</summary>

Nanophotonic structures have versatile applications including solar cells, anti-reflective coatings, electromagnetic interference shielding, optical filters, and light emitting diodes. To design and understand these nanophotonic structures, electrodynamic simulations are essential. These simulations enable us to model electromagnetic fields over time and calculate optical properties. In this work, we introduce frameworks and benchmarks to evaluate nanophotonic structures in the context of parametric structure design problems. The benchmarks are instrumental in assessing the performance of optimization algorithms and identifying an optimal structure based on target optical properties. Moreover, we explore the impact of varying grid sizes in electrodynamic simulations, shedding light on how evaluation fidelity can be strategically leveraged in enhancing structure designs.</details>

### Decision Stacks: Flexible Reinforcement Learning via Modular Generative Models

**Authors:** Siyan Zhao, Aditya Grover

**<details><summary>Abstract**</summary>

Reinforcement learning presents an attractive paradigm to reason about several distinct aspects of sequential decision making, such as specifying complex goals, planning future observations and actions, and critiquing their utilities. However, the combined integration of these capabilities poses competing algorithmic challenges in retaining maximal expressivity while allowing for flexibility in modeling choices for efficient learning and inference. We present Decision Stacks, a generative framework that decomposes goal-conditioned policy agents into 3 generative modules. These modules simulate the temporal evolution of observations, rewards, and actions via independent generative models that can be learned in parallel via teacher forcing. Our framework guarantees both expressivity and flexibility in designing individual modules to account for key factors such as architectural bias, optimization objective and dynamics, transferrability across domains, and inference speed. Our empirical results demonstrate the effectiveness of Decision Stacks for offline policy optimization for several MDP and POMDP environments, outperforming existing methods and enabling flexible generative decision making.</details>

### Decoding the Enigma: Benchmarking Humans and AIs on the Many Facets of Working Memory

**Authors:** Ankur Sikarwar, Mengmi Zhang

**<details><summary>Abstract**</summary>

Working memory (WM), a fundamental cognitive process facilitating the temporary storage, integration, manipulation, and retrieval of information, plays a vital role in reasoning and decision-making tasks. Robust benchmark datasets that capture the multifaceted nature of WM are crucial for the effective development and evaluation of AI WM models. Here, we introduce a comprehensive Working Memory (WorM) benchmark dataset for this purpose. WorM comprises 10 tasks and a total of 1 million trials, assessing 4 functionalities, 3 domains, and 11 behavioral and neural characteristics of WM. We jointly trained and tested state-of-the-art recurrent neural networks and transformers on all these tasks. We also include human behavioral benchmarks as an upper bound for comparison. Our results suggest that AI models replicate some characteristics of WM in the brain, most notably primacy and recency effects, and neural clusters and correlates specialized for different domains and functionalities of WM. In the experiments, we also reveal some limitations in existing models to approximate human behavior. This dataset serves as a valuable resource for communities in cognitive psychology, neuroscience, and AI, offering a standardized framework to compare and enhance WM models, investigate WM's neural underpinnings, and develop WM models with human-like capabilities. Our source code and data are available at: https://github.com/ZhangLab-DeepNeuroCogLab/WorM</details>

### Deep Equilibrium Based Neural Operators for Steady-State PDEs

**Authors:** Tanya Marwah, Ashwini Pokle, J. Zico Kolter, Zachary Lipton, Jianfeng Lu, Andrej Risteski

**<details><summary>Abstract**</summary>

Data-driven machine learning approaches are being increasingly used to solve partial differential equations (PDEs). They have shown particularly striking successes when training an operator, which takes as input a PDE in some family, and outputs its solution. However, the architectural design space, especially given structural knowledge of the PDE family of interest, is still poorly understood. We seek to remedy this gap by studying the benefits of weight-tied neural network architectures for steady-state PDEs. To achieve this, we first demonstrate that the solution of most steady-state PDEs can be expressed as a fixed point of a non-linear operator. Motivated by this observation, we propose FNO-DEQ, a deep equilibrium variant of the FNO architecture that directly solves for the solution of a steady-state PDE as the infinite-depth fixed point of an implicit operator layer using a black-box root solver and differentiates analytically through this fixed point resulting in $\mathcal{O}(1)$ training memory. Our experiments indicate that FNO-DEQ-based architectures outperform FNO-based baselines with $4\times$ the number of parameters in predicting the solution to steady-state PDEs such as Darcy Flow and steady-state incompressible Navier-Stokes. Finally, we show FNO-DEQ is more robust when trained with datasets with more noisy observations than the FNO-based baselines, demonstrating the benefits of using appropriate inductive biases in architectural design for different neural network based PDE solvers. Further, we show a universal approximation result that demonstrates that FNO-DEQ can approximate the solution to any steady-state PDE that can be written as a fixed point equation.</details>

### Deep Momentum Multi-Marginal Schrödinger Bridge

**Authors:** Tianrong Chen, Guan-Horng Liu, Molei Tao, Evangelos Theodorou

**<details><summary>Abstract**</summary>

It is a crucial challenge to reconstruct population dynamics using unlabeled samples from distributions at coarse time intervals. Recent approaches such as flow-based models or Schrödinger Bridge (SB) models have demonstrated appealing performance, yet the inferred sample trajectories either fail to account for the underlying stochasticity or are unnecessarily rigid. In this article, we extend SB into phase space and propose $\underline{D}$eep $\underline{M}$omentum Multi-Marginal $\underline{S}$chrödinger $\underline{B}$ridge (DMSB), a novel computational framework that learns the smooth measure-valued spline for stochastic systems that satisfy position marginal constraints across time. By tailoring the celebrated Bregman Iteration and extending the Iteration Proportional Fitting to phase space, we manage to handle high-dimensional multi-marginal trajectory inference tasks efficiently. Our algorithm outperforms baselines significantly, as evidenced by experiments for synthetic datasets and a real-world single-cell RNA sequence dataset. Additionally, the proposed approach can reasonably reconstruct the evolution of velocity distribution, from position snapshots only, when there is a ground truth velocity that is nevertheless inaccessible.</details>

### [Spotlight] Deep Neural Collapse Is Provably Optimal for the Deep Unconstrained Features Model

**Authors:** Peter Súkeník, Marco Mondelli, Christoph Lampert

**<details><summary>Abstract**</summary>

Neural collapse (NC) refers to the surprising structure of the last layer of deep neural networks in the terminal phase of gradient descent training. Recently, an increasing amount of experimental evidence has pointed to the propagation of NC to earlier layers of neural networks. However, while the NC in the last layer is well studied theoretically, much less is known about its multi-layered counterpart - deep neural collapse (DNC). In particular, existing work focuses either on linear layers or only on the last two layers at the price of an extra assumption. Our work fills this gap by generalizing the established analytical framework for NC - the unconstrained features model - to multiple non-linear layers. Our key technical contribution is to show that, in a deep unconstrained features model, the unique global optimum for binary classification exhibits all the properties typical of DNC. This explains the existing experimental evidence of DNC. We also empirically show that (i) by optimizing deep unconstrained features models via gradient descent, the resulting solution agrees well with our theory, and (ii) trained networks recover the unconstrained features suitable for the occurrence of DNC, thus supporting the validity of this modeling principle.</details>

### DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection

**Authors:** Zhiyuan Yan, Yong Zhang, Xinhang Yuan, Siwei Lyu, Baoyuan Wu

**<details><summary>Abstract**</summary>

A critical yet frequently overlooked challenge in the field of deepfake detection is the lack of a standardized, unified, comprehensive benchmark. This issue leads to unfair performance comparisons and potentially misleading results. Specifically, there is a lack of uniformity in data processing pipelines, resulting in inconsistent data inputs for detection models. Additionally, there are noticeable differences in experimental settings, and evaluation strategies and metrics lack standardization. To fill this gap, we present the first comprehensive benchmark for deepfake detection, called \textit{DeepfakeBench}, which offers three key contributions: 1) a unified data management system to ensure consistent input across all detectors, 2) an integrated framework for state-of-the-art methods implementation, and 3) standardized evaluation metrics and protocols to promote transparency and reproducibility. Featuring an extensible, modular-based codebase, \textit{DeepfakeBench} contains 15 state-of-the-art detection methods, 9 deepfake datasets, a series of deepfake detection evaluation protocols and analysis tools, as well as comprehensive evaluations. Moreover, we provide new insights based on extensive analysis of these evaluations from various perspectives (\eg, data augmentations, backbones). We hope that our efforts could facilitate future research and foster innovation in this increasingly critical domain. All codes, evaluations, and analyses of our benchmark are publicly available at \url{https://github.com/SCLBD/DeepfakeBench}.</details>

### Derandomized novelty detection with FDR control via conformal e-values

**Authors:** Meshi Bashari, Amir Epstein, Yaniv Romano, Matteo Sesia

**<details><summary>Abstract**</summary>

Conformal inference provides a general distribution-free method to rigorously calibrate the output of any machine learning algorithm for novelty detection. While this approach has many strengths, it has the limitation of being randomized, in the sense that it may lead to different results when analyzing twice the same data and this can hinder the interpretation of any findings. We propose to make conformal inferences more stable by leveraging suitable conformal e-values instead of p-values to quantify statistical significance. This solution allows the evidence gathered from multiple analyses of the same data to be aggregated effectively while provably controlling the false discovery rate. Further, we show that the proposed method can reduce randomness without much loss of power compared to standard conformal inference, partly thanks to an innovative way of weighting conformal e-values based on additional side information carefully extracted from the same data. Simulations with synthetic and real data confirm this solution can be effective at eliminating random noise in the inferences obtained with state-of-the-art alternative techniques, sometimes also leading to higher power.</details>

### DiffInfinite: Large Mask-Image Synthesis via Parallel Random Patch Diffusion in Histopathology

**Authors:** Marco Aversa, Gabriel Nobis, Miriam Hägele, Kai Standvoss, Mihaela Chirica, Roderick Murray-Smith, Ahmed Alaa, Lukas Ruff, Daniela Ivanova, Wojciech Samek, Frederick Klauschen, Bruno Sanguinetti, Luis Oala

**<details><summary>Abstract**</summary>

We present DiffInfinite, a hierarchical diffusion model that generates arbitrarily large histological images while preserving long-range correlation structural information. Our approach first generates synthetic segmentation masks, subsequently used as conditions for the high-fidelity generative diffusion process. The proposed sampling method can be scaled up to any desired image size while only requiring small patches for fast training. Moreover, it can be parallelized more efficiently than previous large-content generation methods while avoiding tiling artifacts. The training leverages classifier-free guidance to augment a small, sparsely annotated dataset with unlabelled data. Our method alleviates unique challenges in histopathological imaging practice: large-scale information, costly manual annotation, and protective data handling. The biological plausibility of DiffInfinite data is evaluated in a survey by ten experienced pathologists as well as a downstream classification and segmentation task. Samples from the model score strongly on anti-copying metrics which is relevant for the protection of patient data.</details>

### DiffPack: A Torsional Diffusion Model for Autoregressive Protein Side-Chain Packing

**Authors:** Yangtian Zhang, Zuobai Zhang, Bozitao Zhong, Sanchit Misra, Jian Tang

**<details><summary>Abstract**</summary>

Proteins play a critical role in carrying out biological functions, and their 3D structures are essential in determining their functions. Accurately predicting the conformation of protein side-chains given their backbones is important for applications in protein structure prediction, design and protein-protein interactions. Traditional methods are computationally intensive and have limited accuracy, while existing machine learning methods treat the problem as a regression task and overlook the restrictions imposed by the constant covalent bond lengths and angles. In this work, we present DiffPack, a torsional diffusion model that learns the joint distribution of side-chain torsional angles, the only degrees of freedom in side-chain packing, by diffusing and denoising on the torsional space. To avoid issues arising from simultaneous perturbation of all four torsional angles, we propose autoregressively generating the four torsional angles from $\chi_1$ to $\chi_4$ and training diffusion models for each torsional angle. We evaluate the method on several benchmarks for protein side-chain packing and show that our method achieves improvements of 11.9% and 13.5% in angle accuracy on CASP13 and CASP14, respectively, with a significantly smaller model size ($60\times$ fewer parameters). Additionally, we show the effectiveness of our method in enhancing side-chain predictions in the AlphaFold2 model. Code is available at https://github.com/DeepGraphLearning/DiffPack.</details>

### DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model

**Authors:** Yuanshao Zhu, Yongchao Ye, Shiyao Zhang, Xiangyu Zhao, James Yu

**<details><summary>Abstract**</summary>

Pervasive integration of GPS-enabled devices and data acquisition technologies has led to an exponential increase in GPS trajectory data, fostering advancements in spatial-temporal data mining research. Nonetheless, GPS trajectories contain personal geolocation information, rendering serious privacy concerns when working with raw data. A promising approach to address this issue is trajectory generation, which involves replacing original data with generated, privacy-free alternatives. Despite the potential of trajectory generation, the complex nature of human behavior and its inherent stochastic characteristics pose challenges in generating high-quality trajectories. In this work, we propose a spatial-temporal diffusion probabilistic model for trajectory generation (DiffTraj). This model effectively combines the generative abilities of diffusion models with the spatial-temporal features derived from real trajectories. The core idea is to reconstruct and synthesize geographic trajectories from white noise through a reverse trajectory denoising process. Furthermore, we propose a Trajectory UNet (Traj-UNet) deep neural network to embed conditional information and accurately estimate noise levels during the reverse process. Experiments on two real-world datasets show that DiffTraj can be intuitively applied to generate high-fidelity trajectories while retaining the original distributions. Moreover, the generated results can support downstream trajectory analysis tasks and significantly outperform other methods in terms of geo-distribution evaluations.</details>

### Digital Typhoon: Long-term Satellite Image Dataset for the Spatio-Temporal Modeling of Tropical Cyclones

**Authors:** Asanobu Kitamoto, Jared Hwang, Bastien Vuillod, Lucas Gautier, Yingtao Tian, Tarin Clanuwat

**<details><summary>Abstract**</summary>

This paper presents the official release of the Digital Typhoon dataset, the longest typhoon satellite image dataset for 40+ years aimed at benchmarking machine learning models for long-term spatio-temporal data. To build the dataset, we developed a workflow to create an infrared typhoon-centered image for cropping using Lambert azimuthal equal-area projection referring to the best track data. We also address data quality issues such as inter-satellite calibration to create a homogeneous dataset. To take advantage of the dataset, we organized machine learning tasks by the types and targets of inference, with other tasks for meteorological analysis, societal impact, and climate change. The benchmarking results on the analysis, forecasting, and reanalysis for the intensity suggest that the dataset is challenging for recent deep learning models, due to many choices that affect the performance of various models. This dataset reduces the barrier for machine learning researchers to meet large-scale real-world events called tropical cyclones and develop machine learning models that may contribute to advancing scientific knowledge on tropical cyclones as well as solving societal and sustainability issues such as disaster reduction and climate change. The dataset is publicly available at http://agora.ex.nii.ac.jp/digital-typhoon/dataset/ and https://github.com/kitamoto-lab/digital-typhoon/.</details>

### DinoSR: Self-Distillation and Online Clustering for Self-supervised Speech Representation Learning

**Authors:** Alexander Liu, Heng-Jui Chang, Michael Auli, Wei-Ning Hsu, Jim Glass

**<details><summary>Abstract**</summary>

In this paper, we introduce self-distillation and online clustering for self-supervised speech representation learning (DinoSR) which combines masked language modeling, self-distillation, and online clustering. We show that these concepts complement each other and result in a strong representation learning model for speech. DinoSR first extracts contextualized embeddings from the input audio with a teacher network, then runs an online clustering system on the embeddings to yield a machine-discovered phone inventory, and finally uses the discretized tokens to guide a student network. We show that DinoSR surpasses previous state-of-the-art performance in several downstream tasks, and provide a detailed analysis of the model and the learned discrete units. The source code will be made available after the anonymity period.</details>

### [Oral] Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**Authors:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, Chelsea Finn

**Oral Presentation:** Th, Dec 14, 13:50 -- Oral 6B

**<details><summary>Abstract**</summary>

While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper, we leverage a mapping between reward functions and optimal policies to show that this constrained reward maximization problem can be optimized exactly with a single stage of policy training, essentially solving a classification problem on the human preference data. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for fitting a reward model, sampling from the LM during fine-tuning, or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds RLHF's ability to control sentiment of generations and improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.</details>

### Discover and Align Taxonomic Context Priors  for Open-world Semi-Supervised Learning

**Authors:** Yu Wang, Zhun Zhong, Pengchong Qiao, Xuxin Cheng, Xiawu Zheng, Chang Liu, Nicu Sebe, Rongrong Ji, Jie Chen

**<details><summary>Abstract**</summary>

Open-world Semi-Supervised Learning (OSSL) is a realistic and challenging task, aiming to classify unlabeled samples from both seen and novel classes using partially labeled samples from the seen classes. Previous works typically explore the relationship of samples as priors on the pre-defined single-granularity labels to help novel class recognition. In fact, classes follow a taxonomy and samples can be classified at multiple levels of granularity, which contains more underlying relationships for supervision. We thus argue that learning with single-granularity labels results in sub-optimal representation learning and inaccurate pseudo labels, especially with unknown classes. In this paper, we take the initiative to explore and propose a uniformed framework, called Taxonomic context prIors Discovering and Aligning (TIDA), which exploits the relationship of samples under various granularity. It allows us to discover multi-granularity semantic concepts as taxonomic context priors (i.e., sub-class, target-class, and super-class), and then collaboratively leverage them to enhance representation learning and improve the quality of pseudo labels.Specifically, TIDA comprises two components: i) A taxonomic context discovery module that constructs a set of hierarchical prototypes in the latent space to discover the underlying taxonomic context priors; ii) A taxonomic context-based prediction alignment module that enforces consistency across hierarchical predictions to build the reliable relationship between classes among various granularity and provide additions supervision. We demonstrate that these two components are mutually beneficial for an effective OSSL framework, which is theoretically explained from the perspective of the EM algorithm. Extensive experiments on seven commonly used datasets show that TIDA can significantly improve the performance and achieve a new state of the art. The source codes are publicly available at https://github.com/rain305f/TIDA.</details>

### Dissecting Chain-of-Thought: Compositionality through In-Context Filtering and Learning

**Authors:** Yingcong Li, Kartik Sreenivasan, Angeliki Giannou, Dimitris Papailiopoulos, Samet Oymak

**<details><summary>Abstract**</summary>

Chain-of-thought (CoT) is a method that enables language models to handle complex reasoning tasks by decomposing them into simpler steps. Despite its success, the underlying mechanics of CoT are not yet fully understood. In an attempt to shed light on this, our study investigates the impact of CoT on the ability of transformers to in-context learn a simple to study, yet general family of compositional functions: multi-layer perceptrons (MLPs). In this setting, we find that the success of CoT can be attributed to breaking down in-context learning of a compositional function into two distinct phases: focusing on and filtering data related to each step of the composition and in-context learning the single-step composition function. Through both experimental and theoretical evidence, we demonstrate how CoT significantly reduces the sample complexity of in-context learning (ICL) and facilitates the learning of complex functions that non-CoT methods struggle with. Furthermore, we illustrate how transformers can transition from vanilla in-context learning to mastering a compositional function with CoT by simply incorporating additional layers that perform the necessary data-filtering for CoT via the attention mechanism. In addition to these test-time benefits, we show CoT helps accelerate pretraining by learning shortcuts to represent complex functions and filtering plays an important role in this process. These findings collectively provide insights into the mechanics of CoT, inviting further investigation of its role in complex reasoning tasks.</details>

### Distribution-Free Model-Agnostic Regression Calibration via Nonparametric Methods

**Authors:** Shang Liu, Zhongze Cai, Xiaocheng Li

**<details><summary>Abstract**</summary>

In this paper, we consider the uncertainty quantification problem for regression models. Specifically, we consider an individual calibration objective for characterizing the quantiles of the prediction model. While such an objective is well-motivated from downstream tasks such as newsvendor cost, the existing methods have been largely heuristic and lack of statistical guarantee in terms of individual calibration. We show via simple examples that the existing methods focusing on population-level calibration guarantees such as average calibration or sharpness can lead to harmful and unexpected results. We propose simple nonparametric calibration methods that are agnostic of the underlying prediction model and enjoy both computational efficiency and statistical consistency. Our approach enables a better understanding of the possibility of individual calibration, and we establish matching upper and lower bounds for the calibration error of our proposed methods. Technically, our analysis combines the nonparametric analysis with a covering number argument for parametric analysis, which advances the existing theoretical analyses in the literature of nonparametric density estimation and quantile bandit problems. Importantly, the nonparametric perspective sheds new theoretical insights into regression calibration in terms of the curse of dimensionality and reconciles the existing results on the impossibility of individual calibration. To our knowledge, we make the first effort to reach both individual calibration and finite-sample guarantee with minimal assumptions in terms of conformal prediction. Numerical experiments show the advantage of such a simple approach under various metrics, and also under covariates shift. We hope our work provides a simple benchmark and a starting point of theoretical ground for future research on regression calibration.</details>

### Diverse Community Data for Benchmarking Data Privacy Algorithms

**Authors:** Aniruddha Sen, Christine Task, Dhruv Kapur, Gary Howarth, Karan Bhagat

**<details><summary>Abstract**</summary>

The Collaborative Research Cycle (CRC) is a National Institute of Standards and Technology (NIST) benchmarking program intended to strengthen understanding of tabular data deidentification technologies. Deidentification algorithms are vulnerable to the same bias and privacy issues that impact other data analytics and machine learning applications, and it can even amplify those issues by contaminating downstream applications. This paper summarizes four CRC contributions: theoretical work on the relationship between diverse populations and challenges for equitable deidentification; public benchmark data focused on diverse populations and challenging features; a comprehensive open source suite of evaluation metrology for deidentified datasets; and an archive of more than 450 deidentified data samples from a broad range of techniques. The initial set of evaluation results demonstrate the value of the CRC tools for investigations in this field.</details>

### Diverse Shape Completion via Style Modulated Generative Adversarial Networks

**Authors:** Wesley Khademi, Fuxin Li

**<details><summary>Abstract**</summary>

Shape completion aims to recover the full 3D geometry of an object from a partial observation. This problem is inherently multi-modal since there can be many ways to plausibly complete the missing regions of a shape. Such diversity would be indicative of the underlying uncertainty of the shape and could be preferable for downstream tasks such as planning. In this paper, we propose a novel conditional generative adversarial network that can produce many diverse plausible completions of a partially observed point cloud. To enable our network to produce multiple completions for the same partial input, we introduce stochasticity into our network via style modulation. By extracting style codes from complete shapes during training, and learning a distribution over them, our style codes can explicitly carry shape category information leading to better completions. We further introduce diversity penalties and discriminators at multiple scales to prevent conditional mode collapse and to train without the need for multiple ground truth completions for each partial input. Evaluations across several synthetic and real datasets demonstrate that our method achieves significant improvements in respecting the partial observations while obtaining greater diversity in completions.</details>

### Do Not Marginalize Mechanisms, Rather Consolidate!

**Authors:** Moritz Willig, Matej Zečević, Devendra Dhami, Kristian Kersting

**<details><summary>Abstract**</summary>

Structural causal models (SCMs) are a powerful tool for understanding the complex causal relationships that underlie many real-world systems. As these systems grow in size, the number of variables and complexity of interactions between them does, too. Thus, becoming convoluted and difficult to analyze. This is particularly true in the context of machine learning and artificial intelligence, where an ever increasing amount of data demands for new methods to simplify and compress large scale SCM. While methods for marginalizing and abstracting SCM already exist today, they may destroy the causality of the marginalized model. To alleviate this, we introduce the concept of consolidating causal mechanisms to transform large-scale SCM while preserving consistent interventional behaviour. We show consolidation is a powerful method for simplifying SCM, discuss reduction of computational complexity and give a perspective on generalizing abilities of consolidated SCM.</details>

### Does Continual Learning Meet Compositionality? New Benchmarks and An Evaluation Framework

**Authors:** Weiduo Liao, Ying Wei, Mingchen Jiang, Qingfu Zhang, Hisao Ishibuchi

**<details><summary>Abstract**</summary>

Compositionality facilitates the comprehension of novel objects using acquired concepts and the maintenance of a knowledge pool. This is particularly crucial for continual learners to prevent catastrophic forgetting and enable compositionally forward transfer of knowledge. However, the existing state-of-the-art benchmarks inadequately evaluate the capability of compositional generalization, leaving an intriguing question unanswered. To comprehensively assess this capability, we introduce two vision benchmarks, namely Compositional GQA (CGQA) and Compositional OBJects365 (COBJ), along with a novel evaluation framework called Compositional Few-Shot Testing (CFST). These benchmarks evaluate the systematicity, productivity, and substitutivity aspects of compositional generalization. Experimental results on five baselines and two modularity-based methods demonstrate that current continual learning techniques do exhibit somewhat favorable compositionality in their learned feature extractors. Nonetheless, further efforts are required in developing modularity-based approaches to enhance compositional generalization. We anticipate that our proposed benchmarks and evaluation protocol will foster research on continual learning and compositionality.</details>

### Does progress on ImageNet transfer to real-world datasets?

**Authors:** Alex Fang, Simon Kornblith, Ludwig Schmidt

**<details><summary>Abstract**</summary>

Does progress on ImageNet transfer to real-world datasets? We investigate this question by evaluating ImageNet pre-trained models with varying accuracy (57% - 83%) on six practical image classification datasets. In particular, we study datasets collected with the goal of solving real-world tasks (e.g., classifying images from camera traps or satellites), as opposed to web-scraped benchmarks collected for comparing models. On multiple datasets, models with higher ImageNet accuracy do not consistently yield performance improvements. For certain tasks, interventions such as data augmentation improve performance even when architectures do not. We hope that future benchmarks will include more diverse datasets to encourage a more comprehensive approach to improving learning algorithms.</details>

### Doubly Constrained Fair Clustering

**Authors:** John Dickerson, Seyed Esmaeili, Jamie Morgenstern, Claire Jie Zhang

**<details><summary>Abstract**</summary>

The remarkable attention which fair clustering has received in the last few years has resulted in a significant number of different notions of fairness. Despite the fact that these notions are well-justified, they are often motivated and studied in a disjoint manner where one fairness desideratum is considered exclusively in isolation from the others. This leaves the understanding of the relations between different fairness notions as an important open problem in fair clustering. In this paper, we take the first step in this direction. Specifically, we consider the two most prominent demographic representation fairness notions in clustering: (1) Group Fairness ($\textbf{GF}$), where the different demographic groups are supposed to have close to population-level representation in each cluster and (2) Diversity in Center Selection ($\textbf{DS}$), where the selected centers are supposed to have close to population-level representation of each group. We show that given a constant approximation algorithm for one constraint ($\textbf{GF}$ or $\textbf{DS}$ only) we can obtain a constant approximation solution that satisfies both constraints simultaneously. Interestingly, we prove that any given solution that satisfies the $\textbf{GF}$ constraint can always be post-processed at a bounded degradation to the clustering cost to additionally satisfy the $\textbf{DS}$ constraint while the same statement is not true given a solution that satisfies $\textbf{DS}$ instead. Furthermore, we show that both $\textbf{GF}$ and $\textbf{DS}$ are incompatible (having an empty feasibility set in the worst case) with a collection of other distance-based fairness notions. Finally, we carry experiments to validate our theoretical findings.</details>

### Dream the Impossible: Outlier Imagination with Diffusion Models

**Authors:** Xuefeng Du, Yiyou Sun, Jerry Zhu, Yixuan Li

**<details><summary>Abstract**</summary>

Utilizing auxiliary outlier datasets to regularize the machine learning model has demonstrated promise for out-of-distribution (OOD) detection and safe prediction. Due to the labor intensity in data collection and cleaning, automating outlier data generation has been a long-desired alternative. Despite the appeal, generating photo-realistic outliers in the high dimensional pixel space has been an open challenge for the field. To tackle the problem, this paper proposes a new framework Dream-OOD, which enables imagining photo-realistic outliers by way of diffusion models, provided with only the in-distribution (ID) data and classes. Specifically, Dream-OOD learns a text-conditioned latent space based on ID data, and then samples outliers in the low-likelihood region via the latent, which can be decoded into images by the diffusion model. Different from prior works [16, 95], Dream-OOD enables visualizing and understanding the imagined outliers, directly in the pixel space. We conduct comprehensive quantitative and qualitative studies to understand the efficacy of Dream-OOD, and show that training with the samples generated by Dream-OOD can significantly benefit  OOD detection performance.</details>

### [Spotlight] Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers

**Authors:** Sotiris Anagnostidis, Dario Pavllo, Luca Biggio, Lorenzo Noci, Aurelien Lucchi, Thomas Hofmann

**<details><summary>Abstract**</summary>

Autoregressive Transformers adopted in Large Language Models (LLMs) are hard to scale to long sequences. Despite several works trying to reduce their computational cost, most of LLMs still adopt attention layers between all pairs of tokens in the sequence, thus incurring a quadratic cost. In this study, we present a novel approach that dynamically prunes contextual information while preserving the model's expressiveness, resulting in reduced memory and computational requirements during inference. Our method employs a learnable mechanism that determines which uninformative tokens can be dropped from the context at any point across the generation process. By doing so, our approach not only addresses performance concerns but also enhances interpretability, providing valuable insight into the model's decision-making process. Our technique can be applied to existing pre-trained models through a straightforward fine-tuning process, and the pruning strength can be specified by a sparsity parameter. Notably, our empirical findings demonstrate that we can effectively prune up to 80\% of the context without significant performance degradation on downstream tasks, offering a valuable tool for mitigating inference costs. Our reference implementation achieves up to $2\times$ increase in inference throughput and even greater memory savings.</details>

### Dynamically Masked Discriminator for GANs

**Authors:** Wentian Zhang, Haozhe Liu, Bing Li, Jinheng Xie, Yawen Huang, Yuexiang Li, Yefeng Zheng, Bernard Ghanem

**<details><summary>Abstract**</summary>

Training Generative Adversarial Networks (GANs) remains a challenging problem. The discriminator trains the generator by learning the distribution of real/generated data. However, the distribution of generated data changes throughout the training process, which is difficult for the discriminator to learn. In this paper, we propose a novel method for GANs from the viewpoint of online continual learning. We observe that the discriminator model, trained on historically generated data, often slows down its adaptation to the changes in the new arrival generated data, which accordingly decreases the quality of generated results. By treating the generated data in training as a stream, we propose to detect whether the discriminator slows down the learning of new knowledge in generated data. Therefore, we can explicitly enforce the discriminator to learn new knowledge fast. Particularly, we propose a new discriminator, which automatically detects its retardation and then dynamically masks its features, such that the discriminator can adaptively learn the temporally-vary distribution of generated data. Experimental results show our method outperforms the state-of-the-art approaches.</details>

### E2PNet: Event to Point Cloud Registration with Spatio-Temporal Representation Learning

**Authors:** Xiuhong Lin, Changjie Qiu, zhipeng cai, Siqi Shen, Yu Zang, Weiquan Liu, Xuesheng Bian, Matthias Müller, Cheng Wang

**<details><summary>Abstract**</summary>

Event cameras have emerged as a promising vision sensor in recent years due to their unparalleled temporal resolution and dynamic range. While registration of 2D RGB images to 3D point clouds is a long-standing problem in computer vision, no prior work studies 2D-3D registration for event cameras. To this end, we propose E2PNet, the first learning-based method for event-to-point cloud registration.The core of E2PNet is a novel feature representation network called Event-Points-to-Tensor (EP2T), which encodes event data into a 2D grid-shaped feature tensor. This grid-shaped feature enables matured RGB-based frameworks to be easily used for event-to-point cloud registration, without changing hyper-parameters and the training procedure. EP2T treats the event input as spatio-temporal point clouds. Unlike standard 3D learning architectures that treat all dimensions of point clouds equally, the novel sampling and information aggregation modules in EP2T are designed to handle the inhomogeneity of the spatial and temporal dimensions. Experiments on the MVSEC and VECtor datasets demonstrate the superiority of E2PNet over hand-crafted and other learning-based methods. Compared to RGB-based registration, E2PNet is more robust to extreme illumination or fast motion due to the use of event data. Beyond 2D-3D registration, we also show the potential of EP2T for other vision tasks such as flow estimation, event-to-image reconstruction and object recognition. The source code can be found at: https://github.com/Xmu-qcj/E2PNet.</details>

### ECG-QA: A Comprehensive Question Answering Dataset Combined With Electrocardiogram

**Authors:** Jungwoo Oh, Gyubok Lee, Seongsu Bae, Joon-myoung Kwon, Edward Choi

**<details><summary>Abstract**</summary>

Question answering (QA) in the field of healthcare has received much attention due to significant advancements in natural language processing. However, existing healthcare QA datasets primarily focus on medical images, clinical notes, or structured electronic health record tables. This leaves the vast potential of combining electrocardiogram (ECG) data with these systems largely untapped. To address this gap, we present ECG-QA, the first QA dataset specifically designed for ECG analysis. The dataset comprises a total of 70 question templates that cover a wide range of clinically relevant ECG topics, each validated by an ECG expert to ensure their clinical utility. As a result, our dataset includes diverse ECG interpretation questions, including those that require a comparative analysis of two different ECGs. In addition, we have conducted numerous experiments to provide valuable insights for future research directions. We believe that ECG-QA will serve as a valuable resource for the development of intelligent QA systems capable of assisting clinicians in ECG interpretations.</details>

### EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models

**Authors:** Michael Wornow, Rahul Thapa, Ethan Steinberg, Jason Fries, Nigam Shah

**<details><summary>Abstract**</summary>

While the general machine learning (ML) community has benefited from public datasets, tasks, and models, the progress of ML in healthcare has been hampered by a lack of such shared assets. The success of foundation models creates new challenges for healthcare ML by requiring access to shared pretrained models to validate performance benefits. We help address these challenges through three contributions. First, we publish a new dataset, EHRSHOT, which contains de-identified structured data from the electronic health records (EHRs) of 6,739 patients from Stanford Medicine. Unlike MIMIC-III/IV and other popular EHR datasets, EHRSHOT is longitudinal and not restricted to ICU/ED patients. Second, we publish the weights of CLMBR-T-base, a 141M parameter clinical foundation model pretrained on the structured EHR data of 2.57M patients. We are one of the first to fully release such a model for coded EHR data; in contrast, most prior models released for clinical data (e.g. GatorTron, ClinicalBERT) only work with unstructured text and cannot process the rich, structured data within an EHR. We provide an end-to-end pipeline for the community to validate and build upon its performance. Third, we define 15 few-shot clinical prediction tasks, enabling evaluation of foundation models on benefits such as sample efficiency and task adaptation. Our model and dataset are available via a research data use agreement from here: https://stanfordaimi.azurewebsites.net/. Code to reproduce our results is available here: https://github.com/som-shahlab/ehrshot-benchmark.</details>

### EPIC Fields: Marrying 3D Geometry and Video Understanding

**Authors:** Vadim Tschernezki, Ahmad Darkhalil, Zhifan Zhu, David Fouhey, Iro Laina, Diane Larlus, Dima Damen, Andrea Vedaldi

**<details><summary>Abstract**</summary>

Neural rendering is fuelling a unification of learning, 3D geometry and video understanding that has been waiting for more than two decades. Progress, however, is still hampered by a lack of suitable datasets and benchmarks. To address this gap, we introduce EPIC Fields, an augmentation of EPIC-KITCHENS with 3D camera information. Like other datasets for neural rendering, EPIC Fields removes the complex and expensive step of reconstructing cameras using photogrammetry, and allows researchers to focus on modelling problems. We illustrate the challenge of photogrammetry in egocentric videos of dynamic actions and propose innovations to address them. Compared to other neural rendering datasets, EPIC Fields is better tailored to video understanding because it is paired with labelled action segments and the recent VISOR segment annotations. To further motivate the community, we also evaluate two benchmark tasks in neural rendering and segmenting dynamic objects, with strong baselines that showcase what is not possible today. We also highlight the advantage of geometry in semi-supervised video object segmentations on the VISOR annotations. EPIC Fields reconstructs 96\% of videos in EPIC-KITCHENS, registering 19M frames in 99 hours recorded in 45 kitchens, and is available from: http://epic-kitchens.github.io/epic-fields</details>

### Easy Bayesian Transfer Learning with Informative Priors

**Authors:** Martin Špendl, Klementina Pirc

**<details><summary>Abstract**</summary>

REPRODUCIBILITY SUMMARYScope of ReproducibilityIn this work, we study the reproducibility of the paper: Pre-Train Your Loss: Easy Bayesian Transfer Learning with Informative Priors. The paper proposes a three-step pipeline for replacing standard transfer learning with a pre-trained prior. The first step is training a prior, the second is re-scaling of a prior, and the third is inference.The authors claim that increasing the rank and the scaling factor improves performance on the downstream task. They also argue that using Bayesian learning with informative prior leads to a more data-efficient and improved performance compared to standard SGD transfer learning or using non-informative prior. We reproduce the main claims on one of the four data sets in the paper.MethodologyWe used a combination of the authors' and our code. The authors provided a training pipeline for the user but not the code to fully reproduce the paper. We modified the training pipeline to suit our needs and created a testing pipeline to evaluate the models. We reproduced the results for the Oxford-102-Flowers data set on an Nvidia RTX 3070 GPU using approximately 310 GPU hours for the main results.ResultsOur results confirm most of the claims tested, although we could not achieve the exact same accuracy due to missing hyper-parameters. We reproduced the trend in how scaling the prior impacts the performance and how a learned prior outperforms a non-learned prior.On contrary, we could not reproduce the effect of rank in low-rank covariance approximation on model performance, as well as the beneficial boost in performance of Bayesian learning compared to the standard SGD.What was easyThe authors' implementation provides various training and logging parameters. It is also helpful that the authors provided both the learned priors and scripts for the download, split and pre-processing of the data sets used in the study.What was difficultSetting the environment for the used packages to work correctly was difficult. Although many parameters are available for running the pipeline, their descriptions are misguiding, therefore a lot of time went into clarifying the parameter function and debugging different settings. The training also took a while, especially when training 5 models per data point. Communication with original authorsWe contacted the authors via e-mail about their pipeline and their use of hyper-parameters but did not hear back.</details>

### Efficient Robust Bayesian Optimization for Arbitrary Uncertain inputs

**Authors:** Lin Yang, Junlong Lyu, Wenlong Lyu, Zhitang Chen

**<details><summary>Abstract**</summary>

Bayesian Optimization (BO) is a sample-efficient optimization algorithm widely employed across various applications. In some challenging BO tasks, input uncertainty arises due to the inevitable randomness in the optimization process, such as machining errors, execution noise, or contextual variability. This uncertainty deviates the input from the intended value before evaluation, resulting in significant performance fluctuations in the final result. In this paper, we introduce a novel robust Bayesian Optimization algorithm, AIRBO, which can effectively identify a robust optimum that performs consistently well under arbitrary input uncertainty. Our method directly models the uncertain inputs of arbitrary distributions by empowering the Gaussian Process with the Maximum Mean Discrepancy (MMD) and further accelerates the posterior inference via Nystrom approximation. Rigorous theoretical regret bound is established under MMD estimation error and extensive experiments on synthetic functions and real problems demonstrate that our approach can handle various input uncertainties and achieve a state-of-the-art performance.</details>

### Efficient Symbolic Policy Learning with Differentiable Symbolic Expression

**Authors:** Jiaming Guo, Rui Zhang, Shaohui Peng, Qi Yi, Xing Hu, Ruizhi Chen, Zidong Du, xishan zhang, Ling Li, Qi Guo, Yunji Chen

**<details><summary>Abstract**</summary>

Deep reinforcement learning (DRL) has led to a wide range of advances in sequential decision-making tasks. However, the complexity of neural network policies makes it difficult to understand and deploy with limited computational resources. Currently, employing compact symbolic expressions as symbolic policies is a promising strategy to obtain simple and interpretable policies. Previous symbolic policy methods usually involve complex training processes and pre-trained neural network policies, which are inefficient and limit the application of symbolic policies. In this paper, we propose an efficient gradient-based learning method named Efficient Symbolic Policy Learning (ESPL) that learns the symbolic policy from scratch in an end-to-end way. We introduce a symbolic network as the search space and employ a path selector to find the compact symbolic policy. By doing so we represent the policy with a differentiable symbolic expression and train it in an off-policy manner which further improves the efficiency. In addition, in contrast with previous symbolic policies which only work in single-task RL because of complexity, we expand ESPL on meta-RL to generate symbolic policies for unseen tasks. Experimentally, we show that our approach generates symbolic policies with higher performance and greatly improves data efficiency for single-task RL. In meta-RL, we demonstrate that compared with neural network policies the proposed symbolic policy achieves higher performance and efficiency and shows the potential to be interpretable.</details>

### Ego4D Goal-Step: Toward Hierarchical Understanding of Procedural Activities

**Authors:** Yale Song, Eugene Byrne, Tushar Nagarajan, Huiyu Wang, Miguel Martin, Lorenzo Torresani

**<details><summary>Abstract**</summary>

Human activities are goal-oriented and hierarchical, comprising primary goals at the top level, sequences of steps and substeps in the middle, and atomic actions at the lowest level. Recognizing human activities thus requires relating atomic actions and steps to their functional objectives (what the actions contribute to) and modeling their sequential and hierarchical dependencies towards achieving the goals. Current activity recognition research has primarily focused on only the lowest levels of this hierarchy, i.e., atomic or low-level actions, often in trimmed videos with annotations spanning only a few seconds. In this work, we introduce Ego4D Goal-Step, a new set of annotations on the recently released Ego4D with a novel hierarchical taxonomy of goal-oriented activity labels. It provides dense annotations for 48K procedural step segments (430 hours) and high-level goal annotations for 2,807 hours of Ego4D videos. Compared to existing procedural video datasets, it is substantially larger in size, contains hierarchical action labels (goals - steps - substeps), and provides goal-oriented auxiliary information including natural language summary description, step completion status, and step-to-goal relevance information. We take a data-driven approach to build our taxonomy, resulting in dense step annotations that do not suffer from poor label-data alignment issues resulting from a taxonomy defined a priori. Through comprehensive evaluations and analyses, we demonstrate how Ego4D Goal-Step supports exploring various questions in procedural activity understanding, including goal inference, step prediction, hierarchical relation learning, and long-term temporal modeling.</details>

### EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding

**Authors:** Karttikeya Mangalam, Raiymbek Akshulakov, Jitendra Malik

**<details><summary>Abstract**</summary>

We introduce EgoSchema, a very long-form video question-answering dataset, and benchmark to evaluate long video understanding capabilities of modern vision and language systems. Derived from Ego4D, EgoSchema consists of over 5000 human curated multiple choice question answer pairs, spanning over 250 hours of real video data, covering a very broad range of natural human activity and behavior. For each question, EgoSchema requires the correct answer to be selected between five given options based on a three-minute-long video clip. While some prior works have proposed video datasets with long clip lengths, we posit that merely the length of the video clip does not truly capture the temporal difficulty of the video task that is being considered. To remedy this, we introduce temporal certificate sets, a general notion for capturing the intrinsic temporal understanding length associated with a broad range of video understanding tasks & datasets. Based on this metric, we find EgoSchema to have intrinsic temporal lengths over 5.7x longer than the second closest dataset and 10x to 100x longer than any other video understanding dataset. Further, our evaluation of several current state-of-the-art video and language models shows them to be severely lacking in long-term video understanding capabilities. Even models with several billions of parameters achieve QA accuracy less than 33% (random is 20%) on the EgoSchema multi-choice question answering task, while humans achieve about 76% accuracy. We posit that EgoSchema, with its long intrinsic temporal structures and diverse complexity, would serve as a valuable evaluation probe for developing effective long-term video understanding systems in the future. Data and Zero-shot model evaluation code will all be open-sourced under the Ego4D license at http://egoschema.github.io.</details>

### EgoTracks: A Long-term Egocentric Visual Object Tracking Dataset

**Authors:** Hao Tang, Kevin J Liang, Kristen Grauman, Matt Feiszli, Weiyao Wang

**<details><summary>Abstract**</summary>

Visual object tracking is a key component to many egocentric vision problems. However, the full spectrum of challenges of egocentric tracking faced by an embodied AI is underrepresented in many existing datasets; these tend to focus on relatively short, third-person videos. Egocentric video has several distinguishing characteristics from those commonly found in past datasets: frequent large camera motions and hand interactions with objects commonly lead to occlusions or objects exiting the frame, and object appearance can change rapidly due to widely different points of view, scale, or object states. Embodied tracking is also naturally long-term, and being able to consistently (re-)associate objects to their appearances and disappearances over as long as a lifetime is critical. Previous datasets under-emphasize this re-detection problem, and their "framed" nature has led to adoption of various spatiotemporal priors that we find do not necessarily generalize to egocentric video. We thus introduce EgoTracks, a new dataset for long-term egocentric visual object tracking. Sourced from the Ego4D dataset, this new dataset presents a significant challenge to recent state-of-the-art single-object tracking models, which we find score poorly on traditional tracking metrics for our new dataset, compared to popular benchmarks. We further show improvements that can be made to a STARK tracker to significantly increase its performance on egocentric data, resulting in a baseline model we call EgoSTARK. We publicly release our annotations and benchmark, hoping our dataset leads to further advancements in tracking.</details>

### Emergent Communication in Interactive Sketch Question Answering

**Authors:** Zixing Lei, Yiming Zhang, Yuxin Xiong, Siheng Chen

**<details><summary>Abstract**</summary>

Vision-based emergent communication (EC) aims to learn to communicate through sketches and demystify the evolution of human communication. Ironically, previous works neglect multi-round interaction, which is indispensable in human communication. To fill this gap, we first introduce a novel Interactive Sketch Question Answering (ISQA) task, where two collaborative players are interacting through sketches to answer a question about an image. To accomplish this task, we design a new and efficient interactive EC system, which can achieve an effective balance among three evaluation factors, including the question answering accuracy, drawing complexity and human interpretability. Our experimental results demonstrate that multi-round interactive mechanism facilitates tar- geted and efficient communication between intelligent agents. The code will be released.</details>

### Emergent Correspondence from Image Diffusion

**Authors:** Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, Bharath Hariharan

**<details><summary>Abstract**</summary>

Finding correspondences between images is a fundamental problem in computer vision. In this paper, we show that correspondence emerges in image diffusion models without any explicit supervision. We propose a simple strategy to extract this implicit knowledge out of diffusion networks as image features, namely DIffusion FeaTures (DIFT), and use them to establish correspondences between real images. Without any additional fine-tuning or supervision on the task-specific data or annotations, DIFT is able to outperform both weakly-supervised methods and competitive off-the-shelf features in identifying semantic, geometric, and temporal correspondences. Particularly for semantic correspondence, DIFT from Stable Diffusion is able to outperform DINO and OpenCLIP by 19 and 14 accuracy points respectively on the challenging SPair-71k benchmark. It even outperforms the state-of-the-art supervised methods on 9 out of 18 categories while remaining on par for the overall performance. Project page: https://diffusionfeatures.github.io.</details>

### [Spotlight] Encoding Time-Series Explanations through Self-Supervised Model Behavior Consistency

**Authors:** Owen Queen, Tom Hartvigsen, Teddy Koker, Huan He, Theodoros Tsiligkaridis, Marinka Zitnik

**<details><summary>Abstract**</summary>

Interpreting time series models is uniquely challenging because it requires identifying both the location of time series signals that drive model predictions and their matching to an interpretable temporal pattern. While explainers from other modalities can be applied to time series, their inductive biases do not transfer well to the inherently challenging interpretation of time series. We present TimeX, a time series consistency model for training explainers. TimeX trains an interpretable surrogate to mimic the behavior of a pretrained time series model. It addresses the issue of model faithfulness by introducing model behavior consistency, a novel formulation that preserves relations in the latent space induced by the pretrained model with relations in the latent space induced by TimeX. TimeX provides discrete attribution maps and, unlike existing interpretability methods, it learns a latent space of explanations that can be used in various ways, such as to provide landmarks to visually aggregate similar explanations and easily recognize temporal patterns. We evaluate TimeX on eight synthetic and real-world datasets and compare its performance against state-of-the-art interpretability methods. We also conduct case studies using physiological time series. Quantitative evaluations demonstrate that TimeX achieves the highest or second-highest performance in every metric compared to baselines across all datasets. Through case studies, we show that the novel components of TimeX show potential for training faithful, interpretable models that capture the behavior of pretrained time series models.</details>

### Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization

**Authors:** Xilie Xu, Jingfeng ZHANG, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**<details><summary>Abstract**</summary>

Adversarial contrastive learning (ACL) is a technique that enhances standard contrastive learning (SCL) by incorporating adversarial data to learn a robust representation that can withstand adversarial attacks and common corruptions without requiring costly annotations. To improve transferability, the existing work introduced the standard invariant regularization (SIR) to impose style-independence property to SCL, which can exempt the impact of nuisance style factors in the standard representation. However, it is unclear how the style-independence property benefits ACL-learned robust representations. In this paper, we leverage the technique of causal reasoning to interpret the ACL and propose adversarial invariant regularization (AIR) to enforce independence from style factors. We regulate the ACL using both SIR and AIR to output the robust representation. Theoretically, we show that AIR implicitly encourages the representational distance between different views of natural data and their adversarial variants to be independent of style factors. Empirically, our experimental results show that invariant regularization significantly improves the performance of state-of-the-art ACL methods in terms of both standard generalization and robustness on downstream tasks. To the best of our knowledge, we are the first to apply causal reasoning to interpret ACL and develop AIR for enhancing ACL-learned robust representations. Our source code is at https://github.com/GodXuxilie/Enhancing_ACL_via_AIR.</details>

### Enhancing Knowledge Transfer for Task Incremental Learning with Data-free Subnetwork

**Authors:** Qiang Gao, Xiaojun Shan, Yuchen Zhang, Fan Zhou

**<details><summary>Abstract**</summary>

As there exist competitive subnetworks within a dense network in concert with Lottery Ticket Hypothesis, we introduce a novel neuron-wise task incremental learning method, namely Data-free Subnetworks (DSN), which attempts to enhance the elastic knowledge transfer across the tasks that sequentially arrive. Specifically, DSN primarily seeks to transfer knowledge to the new coming task from the learned tasks by selecting the affiliated weights of a small set of neurons to be activated, including the reused neurons from prior tasks via neuron-wise masks. And it also transfers possibly valuable knowledge to the earlier tasks via data-free replay. Especially, DSN inherently relieves the catastrophic forgetting and the unavailability of past data or possible privacy concerns. The comprehensive experiments conducted on four benchmark datasets demonstrate the effectiveness of the proposed DSN in the context of task-incremental learning by comparing it to several state-of-the-art baselines. In particular, DSN enables the knowledge transfer to the earlier tasks, which is often overlooked by prior efforts.</details>

### Epidemic Learning: Boosting Decentralized Learning with Randomized Communication

**Authors:** Martijn De Vos, Sadegh Farhadkhani, Rachid Guerraoui, Anne-marie Kermarrec, Rafael Pires, Rishi Sharma

**<details><summary>Abstract**</summary>

We present Epidemic Learning (EL), a simple yet powerful decentralized learning (DL) algorithm that leverages changing communication topologies to achieve faster model convergence compared to conventional DL approaches. At each round of EL, each node sends its model updates to a random sample of $s$ other nodes (in a system of $n$ nodes). We provide an extensive theoretical analysis of EL, demonstrating that its changing topology culminates in superior convergence properties compared to the state-of-the-art (static and dynamic) topologies. Considering smooth non-convex loss functions, the number of transient iterations for EL, i.e., the rounds required to achieve asymptotic linear speedup, is in $O(n^3/s^2)$ which outperforms the best-known bound $O(n^3)$ by a factor of $s^2$, indicating the benefit of randomized communication for DL. We empirically evaluate EL in a 96-node network and compare its performance with state-of-the-art DL approaches. Our results illustrate that EL converges up to  $ 1.7\times$ quicker than baseline DL algorithms and attains $2.2 $\% higher accuracy for the same communication volume.</details>

### Error Discovery By Clustering Influence Embeddings

**Authors:** Fulton Wang, Julius Adebayo, Sarah Tan, Diego Garcia-Olano, Narine Kokhlikyan

**<details><summary>Abstract**</summary>

We present a method for identifying groups of test examples---slices---on which a model under-performs, a task now known as slice discovery. We formalize coherence---a requirement that erroneous predictions, within a slice, should be wrong for the same reason---as a key property that any slice discovery method should satisfy.  We then use influence functions to derive a new slice discovery method, InfEmbed, which satisfies coherence by returning slices whose examples are influenced similarly by the training data.  InfEmbed is simple, and consists of applying K-Means clustering to a novel representation we deem influence embeddings. We show InfEmbed outperforms current state-of-the-art methods on 2 benchmarks, and is effective for model debugging across several case studies.</details>

### Estimating Generic 3D Room Structures from 2D Annotations

**Authors:** Denys Rozumnyi, Stefan Popov, Kevis-kokitsi Maninis, Matthias Niessner, Vittorio Ferrari

**<details><summary>Abstract**</summary>

Indoor rooms are among the most common use cases in 3D scene understanding. Current state-of-the-art methods for this task are driven by large annotated datasets. Room layouts are especially important, consisting of structural elements in 3D, such as wall, floor, and ceiling. However, they are difficult to annotate, especially on pure RGB video. We propose a novel method to produce generic 3D room layouts just from 2D segmentation masks, which are easy to annotate for humans. Based on these 2D annotations, we automatically reconstruct 3D plane equations for the structural elements and their spatial extent in the scene, and connect adjacent elements at the appropriate contact edges. We annotate and publicly release 2246 3D room layouts on the RealEstate10k dataset, containing YouTube videos. We demonstrate the high quality of these 3D layouts annotations with extensive experiments.</details>

### Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking

**Authors:** Juanhui Li, Harry Shomer, Haitao Mao, Shenglai Zeng, Yao Ma, Neil Shah, Jiliang Tang, Dawei Yin

**<details><summary>Abstract**</summary>

Link prediction attempts to predict whether an unseen edge exists based on only a portion of the graph. A flurry of methods has been created in recent years that attempt to make use of graph neural networks (GNNs) for this task. Furthermore, new and diverse datasets have also been created to better evaluate the effectiveness of these new models. However, multiple limitations currently exist that hinders our ability to properly evaluate these new methods. This includes, but is not limited to: (1) The underreporting of performance on multiple baselines, (2) A lack of a unified data split and evaluation metric on some datasets, (3) An unrealistic evaluation setting that produces negative samples that are easy to classify. To overcome these challenges we first conduct a fair comparison across prominent methods and datasets, utilizing the same dataset settings and hyperparameter settings. We then create a new real-world evaluation setting that samples difficult negative samples via multiple heuristics. The new evaluation setting helps promote new challenges and opportunities in link prediction by aligning the evaluation with real-world situations.</details>

### Evaluating Open-QA Evaluation

**Authors:** Cunxiang Wang, Sirui Cheng, Qipeng Guo, Yuanhao Yue, Bowen Ding, Zhikun Xu, Yidong Wang, Xiangkun Hu, Zheng Zhang, Yue Zhang

**<details><summary>Abstract**</summary>

This study focuses on the evaluation of the Open Question Answering (Open-QA) task, which can directly estimate the factuality of large language models (LLMs). Current automatic evaluation methods have shown limitations, indicating that human evaluation still remains the most reliable approach. We introduce a new task, QA Evaluation (QA-Eval) and the corresponding dataset EVOUNA, designed to assess the accuracy of AI-generated answers in relation to standard answers within Open-QA. Our evaluation of these methods utilizes human-annotated results to measure their performance. Specifically, the work investigates methods that show high correlation with human evaluations, deeming them more reliable. We also discuss the pitfalls of current methods and methods to improve LLM-based evaluators. We believe this new QA-Eval task and corresponding dataset EVOUNA will facilitate the development of more effective automatic evaluation tools and prove valuable for future research in this area. All resources are available at https://github.com/wangcunxiang/QA-Eval and it is under the Apache-2.0 License.</details>

### Evaluating and Improving Tool-Augmented Computation-Intensive Math Reasoning

**Authors:** Beichen Zhang, Kun Zhou, Xilin Wei, Xin Zhao, Jing Sha, Shijin Wang, Ji-Rong Wen

**<details><summary>Abstract**</summary>

Chain-of-thought prompting (CoT) and tool augmentation have been validated in recent work as effective practices for improving large language models (LLMs) to perform step-by-step reasoning on complex math-related tasks.However, most existing math reasoning datasets may not be able to fully evaluate and analyze the ability of LLMs in manipulating tools and performing reasoning, as they often only require very few invocations of tools or miss annotations for evaluating intermediate reasoning steps, thus supporting only outcome evaluation.To address the issue, we construct **CARP**, a new Chinese dataset consisting of 4,886 computation-intensive algebra problems with formulated annotations on intermediate steps, facilitating the evaluation of the intermediate reasoning process.In CARP, we test four LLMs with CoT prompting, and find that they are all prone to make mistakes at the early steps of the solution, leading to incorrect answers.Based on this finding, we propose a new approach that can facilitate the deliberation on reasoning steps with tool interfaces, namely **DELI**.In DELI, we first initialize a step-by-step solution based on retrieved exemplars, then iterate two deliberation procedures that check and refine the intermediate steps of the generated solution, from both tool manipulation and natural language reasoning perspectives, until solutions converge or the maximum iteration is achieved.Experimental results on CARP and six other datasets show that the proposed DELI mostly outperforms competitive baselines, and can further boost the performance of existing CoT methods.Our data and code are available at https://github.com/RUCAIBox/CARP.</details>

### Event Stream GPT: A Data Pre-processing and Modeling Library for Generative, Pre-trained Transformers over Continuous-time Sequences of Complex Events

**Authors:** Matthew McDermott, Bret Nestor, Peniel Argaw, Isaac S Kohane

**<details><summary>Abstract**</summary>

Generative, pre-trained transformers (GPTs, a type of "Foundation Models") have reshaped natural language processing (NLP) through their versatility in diverse downstream tasks. However, their potential extends far beyond NLP. This paper provides a software utility to help realize this potential, extending the applicability of GPTs to continuous-time sequences of complex events with internal dependencies, such as medical record datasets. Despite their potential, the adoption of foundation models in these domains has been hampered by the lack of suitable tools for model construction and evaluation. To bridge this gap, we introduce Event Stream GPT (ESGPT), an open-source library designed to streamline the end-to-end process for building GPTs for continuous-time event sequences. ESGPT allows users to (1) build flexible, foundation-model scale input datasets by specifying only a minimal configuration file, (2) leverage a Hugging Face compatible modeling API for GPTs over this modality that incorporates intra-event causal dependency structures and autoregressive generation capabilities, and (3) evaluate models via standardized processes that can assess few and even zero-shot performance of pre-trained models on user-specified fine-tuning tasks.</details>

### Exploiting Correlated Auxiliary Feedback in Parameterized Bandits

**Authors:** Arun Verma, Zhongxiang Dai, YAO SHU, Bryan Kian Hsiang Low

**<details><summary>Abstract**</summary>

We study a novel variant of the parameterized bandits problem in which the learner can observe additional auxiliary feedback that is correlated with the observed reward. The auxiliary feedback is readily available in many real-life applications, e.g., an online platform that wants to recommend the best-rated services to its users can observe the user's rating of service (rewards) and collect additional information like service delivery time (auxiliary feedback). In this paper, we first develop a method that exploits auxiliary feedback to build a reward estimator with tight confidence bounds, leading to a smaller regret. We then characterize the regret reduction in terms of the correlation coefficient between reward and its auxiliary feedback. Experimental results in different settings also verify the performance gain achieved by our proposed method.</details>

### [Spotlight] Exploring Geometry of Blind Spots in Vision models

**Authors:** Sriram Balasubramanian, Gaurang Sriramanan, Vinu Sankar Sadasivan, Soheil Feizi

**<details><summary>Abstract**</summary>

Despite the remarkable success of deep neural networks in a myriad of settings, several works have demonstrated their overwhelming sensitivity to near-imperceptible perturbations, known as adversarial attacks. On the other hand, prior works have also observed that deep networks can be under-sensitive, wherein large-magnitude perturbations in input space do not induce appreciable changes to network activations. In this work, we study in detail the phenomenon of under-sensitivity in vision models such as CNNs and Transformers, and present techniques to study the geometry and extent of “equi-confidence” level sets of such networks. We propose a Level Set Traversal algorithm that iteratively explores regions of high confidence with respect to the input space using orthogonal components of the local gradients. Given a source image, we use this algorithm to identify inputs that lie in the same equi-confidence level set as the source image despite being perceptually similar to arbitrary images from other classes. We further observe that the source image is linearly connected by a high-confidence path to these inputs, uncovering a star-like structure for level sets of deep networks. Furthermore, we attempt to identify and estimate the extent of these connected higher-dimensional regions over which the model maintains a high degree of confidence.</details>

### Exponential Lower Bounds for Fictitious Play in Potential Games

**Authors:** Ioannis Panageas, Nikolas Patris, Stratis Skoulakis, Volkan Cevher

**<details><summary>Abstract**</summary>

Fictitious Play (FP) is a simple and natural dynamic for repeated play with many applications in game theory and multi-agent reinforcement learning. It was introduced by Brown and its convergence properties for two-player zero-sum games was established later by Robinson. Potential games [Monderer and Shapley 1996] is another class of games which exhibit the FP property [Monderer and Shapley 1996], i.e., FP dynamics converges to a Nash equilibrium if all agents follows it. Nevertheless, except for two-player zero-sum games and for specific instances of payoff matrices [Abernethy et. al. 2021] or for adversarial tie-breaking rules [Daskalakis and Pan, 2014], the \textit{convergence rate} of FP is unknown. In this work, we focus on the rate of convergence of FP when applied to potential games and more specifically identical payoff games. We prove that FP can take exponential time (in the number of strategies) to reach a Nash equilibrium, even if the game is restricted to \textit{two agents}. To prove this, we recursively construct a two-player coordination game with a unique Nash equilibrium. Moreover, every approximate Nash equilibrium in the constructed game must be close to the pure Nash equilibrium in $\ell_1$-distance.</details>

### Expressivity-Preserving GNN Simulation

**Authors:** Fabian Jogl, Maximilian Thiessen, Thomas Gärtner

**<details><summary>Abstract**</summary>

We systematically investigate graph transformations that enable standard message passing to simulate state-of-the-art graph neural networks (GNNs) without loss of expressivity. Using these, many state-of-the-art GNNs can be implemented with message passing operations from standard libraries, eliminating many sources of implementation issues and allowing for better code optimization. We distinguish between weak and strong simulation: weak simulation achieves the same expressivity only after several message passing steps while strong simulation achieves this after every message passing step. Our contribution leads to a direct way to translate common operations of non-standard GNNs to graph transformations that allow for strong or weak simulation. Our empirical evaluation shows competitive predictive performance of message passing on transformed graphs for various molecular benchmark datasets, in several cases surpassing the original GNNs.</details>

### Extremal Domain Translation with Neural Optimal Transport

**Authors:** Milena Gazdieva, Alexander Korotin, Daniil Selikhanovych, Evgeny Burnaev

**<details><summary>Abstract**</summary>

In many unpaired image domain translation problems, e.g., style transfer or super-resolution, it is important to keep the translated image similar to its respective input image. We propose the extremal transport (ET) which is a mathematical formalization of the theoretically best possible unpaired translation between a pair of domains w.r.t. the given similarity function. Inspired by the recent advances in neural optimal transport (OT), we propose a scalable algorithm to approximate ET maps as a limit of partial OT maps. We test our algorithm on toy examples and on the unpaired image-to-image translation task. The code is publicly available at https://github.com/milenagazdieva/ExtremalNeuralOptimalTransport</details>

### FACE: Evaluating Natural Language Generation with Fourier Analysis of Cross-Entropy

**Authors:** Zuhao Yang, Yingfang Yuan, Yang Xu, SHUO ZHAN, Huajun Bai, Kefan Chen

**<details><summary>Abstract**</summary>

Measuring the distance between machine-produced and human language is a critical open problem. Inspired by empirical findings from psycholinguistics on the periodicity of entropy in language, we propose FACE, a set of metrics based on Fourier Analysis of the estimated Cross-Entropy of language, for measuring the similarity between model-generated and human-written languages. Based on an open-ended generation task and the experimental data from previous studies, we find that FACE can effectively identify the human-model gap, scales with model size, reflects the outcomes of different sampling methods for decoding, correlates well with other evaluation metrics and with human judgment scores.</details>

### FAMO: Fast Adaptive Multitask Optimization

**Authors:** Bo Liu, Yihao Feng, Peter Stone, Qiang Liu

**<details><summary>Abstract**</summary>

One of the grand enduring goals of AI is to create generalist agents that can learn multiple different tasks from diverse data via multitask learning (MTL). However, in practice, applying gradient descent (GD) on the average loss across all tasks may yield poor multitask performance due to severe under-optimization of certain tasks. Previous approaches that manipulate task gradients for a more balanced loss decrease require storing and computing all task gradients ($\mathcal{O}(k)$ space and time where $k$ is the number of tasks), limiting their use in large-scale scenarios. In this work, we introduce Fast Adaptive Multitask Optimization (FAMO), a dynamic weighting method that decreases task losses in a balanced way using $\mathcal{O}(1)$ space and time. We conduct an extensive set of experiments covering multi-task supervised and reinforcement learning problems. Our results indicate that FAMO achieves comparable or superior performance to state-of-the-art gradient manipulation techniques while offering significant improvements in space and computational efficiency. Code is available at \url{https://github.com/Cranial-XIX/FAMO}.</details>

### FAST: a Fused and Accurate Shrinkage Tree for Heterogeneous Treatment Effects Estimation

**Authors:** Jia Gu, Caizhi Tang, Han Yan, Qing Cui, Longfei Li, Jun Zhou

**<details><summary>Abstract**</summary>

This paper proposes a novel strategy for estimating the heterogeneous treatment effect  called the Fused and Accurate Shrinkage Tree ($\mathrm{FAST}$). Our approach utilizes both trial and observational data to improve the accuracy and robustness of the estimator. Inspired by the concept of shrinkage estimation in statistics, we develop an optimal weighting scheme and a corresponding estimator that balances the unbiased estimator based on the trial data with the potentially biased estimator based on the observational data. Specifically, combined with tree-based techniques, we introduce a new split criterion that utilizes both trial data and observational data to more accurately estimate the treatment effect. Furthermore, we confirm the consistency of our proposed tree-based estimator and demonstrate the effectiveness of our criterion in reducing prediction error through theoretical analysis.  The advantageous  finite sample performance of the $\mathrm{FAST}$ and its ensemble version over existing methods is demonstrated via  simulations and real data analysis.</details>

### FD-Align: Feature Discrimination Alignment for Fine-tuning Pre-Trained Models in Few-Shot Learning

**Authors:** Kun Song, Huimin Ma, Bochao Zou, Huishuai Zhang, Weiran Huang

**<details><summary>Abstract**</summary>

Due to the limited availability of data, existing few-shot learning methods trained from scratch fail to achieve satisfactory performance. In contrast, large-scale pre-trained models such as CLIP demonstrate remarkable few-shot and zero-shot capabilities. To enhance the performance of pre-trained models for downstream tasks, fine-tuning the model on downstream data is frequently necessary. However, fine-tuning the pre-trained model leads to a decrease in its generalizability in the presence of distribution shift, while the limited number of samples in few-shot learning makes the model highly susceptible to overfitting. Consequently, existing methods for fine-tuning few-shot learning primarily focus on fine-tuning the model's classification head or introducing additional structure. In this paper, we introduce a fine-tuning approach termed Feature Discrimination Alignment (FD-Align). Our method aims to bolster the model's generalizability by preserving the consistency of spurious features across the fine-tuning process. Extensive experimental results validate the efficacy of our approach for both ID and OOD tasks. Once fine-tuned, the model can seamlessly integrate with existing methods, leading to performance improvements. Our code can be found in https://github.com/skingorz/FD-Align.</details>

### FELM: Benchmarking Factuality Evaluation of Large Language Models

**Authors:** shiqi chen, Yiran Zhao, Jinghan Zhang, I-Chun Chern, Siyang Gao, Pengfei Liu, Junxian He

**<details><summary>Abstract**</summary>

Assessing factuality of text generated by large language models (LLMs) is an emerging yet crucial research area, aimed at alerting users to potential errors and guiding the development of more reliable LLMs. Nonetheless, the evaluators assessing factuality necessitate suitable evaluation themselves to gauge progress and foster advancements. This direction remains under-explored, resulting in substantial impediments to the progress of factuality evaluators. To mitigate this issue, we introduce a benchmark for Factuality Evaluation of large Language Models, referred to as FELM. In this benchmark, we collect responses generated from LLMs and annotate factuality labels in a fine-grained manner. Contrary to previous studies that primarily concentrate on the factuality of world knowledge (e.g. information from Wikipedia), FELM focuses on factuality across diverse domains, spanning from world knowledge to math and reasoning. Our annotation is based on text segments, which can help pinpoint specific factual errors. The factuality annotations are further supplemented by predefined error types and reference links that either support or contradict the statement. In our experiments, we investigate the performance of several LLM-based factuality evaluators on FELM, including both vanilla LLMs and those augmented with retrieval mechanisms and chain-of-thought processes. Our findings reveal that while retrieval aids factuality evaluation, current LLMs are far from satisfactory to faithfully detect factual errors.</details>

### FLAIR : a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery

**Authors:** Anatol Garioud, Nicolas Gonthier, Loic Landrieu, Apolline De Wit, Marion Valette, Marc Poupée, Sebastien Giordano, boris Wattrelos

**<details><summary>Abstract**</summary>

We introduce the French Land cover from Aerospace ImageRy (FLAIR), an extensive dataset from the French National Institute of Geographical and Forest Information (IGN) that provides a unique and rich resource for large-scale geospatial analysis. FLAIR contains high-resolution aerial imagery with a ground sample distance of 20 cm and over 20 billion individually labeled pixels for precise land-cover classification. The dataset also integrates temporal and spectral data from optical satellite time series. FLAIR thus combines data with varying spatial, spectral, and temporal resolutions across over 817 km² of acquisitions representing the full landscape diversity of France. This diversity makes FLAIR a valuable resource for the development and evaluation of novel methods for large-scale land-cover semantic segmentation and raises significant challenges in terms of computer vision, data fusion, and geospatial analysis. We also provide powerful uni- and multi-sensor baseline models that can be employed to assess algorithm's performance and for downstream applications.</details>

### FORB: A Flat Object Retrieval Benchmark for Universal Image Embedding

**Authors:** Pengxiang Wu, Siman Wang, Kevin Dela Rosa, Derek Hu

**<details><summary>Abstract**</summary>

Image retrieval is a fundamental task in computer vision. Despite recent advances in this field, many techniques have been evaluated on a limited number of domains, with a small number of instance categories. Notably, most existing works only consider domains like 3D landmarks, making it difficult to generalize the conclusions made by these works to other domains, e.g., logo and other 2D flat objects. To bridge this gap, we introduce a new dataset for benchmarking visual search methods on flat images with diverse patterns. Our flat object retrieval benchmark (FORB) supplements the commonly adopted 3D object domain, and more importantly, it serves as a testbed for assessing the image embedding quality on out-of-distribution domains. In this benchmark we investigate the retrieval accuracy of representative methods in terms of candidate ranks, as well as matching score margin, a viewpoint which is largely ignored by many works. Our experiments not only highlight the challenges and rich heterogeneity of FORB, but also reveal the hidden properties of different retrieval strategies. The proposed benchmark is a growing project and we expect to expand in both quantity and variety of objects. The dataset and supporting codes are available at https://github.com/pxiangwu/FORB/.</details>

### Fair Graph Distillation

**Authors:** Qizhang Feng, Zhimeng Jiang, Ruiquan Li, Yicheng Wang, Na Zou, Jiang Bian, Xia Hu

**<details><summary>Abstract**</summary>

As graph neural networks (GNNs) struggle with large-scale graphs due to high computational demands, data distillation for graph data promises to alleviate this issue by distilling a large real graph into a smaller distilled graph while maintaining comparable prediction performance for GNNs trained on both graphs. However, we observe that GNNs trained on distilled graphs may exhibit more severe group fairness problems than those trained on real graphs. Motivated by this observation, we propose \textit{fair graph distillation} (\Algnameabbr), an approach for generating small distilled \textit{fair and informative} graphs based on the graph distillation method. The challenge lies in the deficiency of sensitive attributes for nodes in the distilled graph, making most debiasing methods (e.g., regularization and adversarial debiasing) intractable for distilled graphs. We develop a simple yet effective bias metric, called coherence, for distilled graphs. Based on the proposed coherence metric, we introduce a framework for fair graph distillation using a bi-level optimization algorithm. Extensive experiments demonstrate that the proposed algorithm can achieve better prediction performance-fairness trade-offs across various datasets and GNN architectures.</details>

### FairLISA: Fair User Modeling with Limited Sensitive Attributes Information

**Authors:** zheng zhang, Qi Liu, Hao Jiang, Fei Wang, Yan Zhuang, Le Wu, Weibo Gao, Enhong Chen

**<details><summary>Abstract**</summary>

User modeling techniques profile users' latent characteristics (e.g., preference) from their observed behaviors, and play a crucial role in decision-making. Unfortunately, traditional user models may unconsciously capture biases related to sensitive attributes (e.g., gender) from behavior data, even when this sensitive information is not explicitly provided. This can lead to unfair issues and discrimination against certain groups based on these sensitive attributes.  Recent studies have been proposed to improve fairness by explicitly decorrelating user modeling results and sensitive attributes. However, most existing approaches assume that fully sensitive attribute labels are available in the training set, which is unrealistic due to collection limitations like privacy concerns, and hence bear the limitation of performance. In this paper, we focus on a practical situation with limited sensitive data and propose a novel FairLISA framework, which can efficiently utilize data with known and unknown sensitive attributes to facilitate fair model training. We first propose a novel theoretical perspective to build the relationship between data with both known and unknown sensitive attributes with the fairness objective.  Then, based on this, we provide a general adversarial framework to effectively leverage the whole user data for fair user modeling. We conduct experiments on representative user modeling tasks including recommender system and cognitive diagnosis. The results demonstrate that our FairLISA can effectively improve fairness while retaining high accuracy in scenarios with different ratios of missing sensitive attributes.</details>

### Fast Online Changepoint Detection via Functional Pruning CUSUM Statistics

**Authors:** Gaetano Romano, Idris A. Eckley, Paul Fearnhead, Guillem Rigaill

**<details><summary>Abstract**</summary>

Many modern applications of online changepoint detection require the ability to process high-frequency observations, sometimes with limited available computational resources. Online algorithms for detecting a change in mean often involve using a moving window, or specifying the expected size of change. Such choices affect which changes the algorithms have most power to detect. We introduce an algorithm, Functional Online CuSUM (FOCuS), which is equivalent to running these earlier methods simultaneously for all sizes of windows, or all possible values for the size of change. Our theoretical results give tight bounds on the expected computational cost per iteration of FOCuS, with this being logarithmic in the number of observations. We show how FOCuS can be applied to a number of different changes in mean scenarios, and demonstrate its practical utility through its state-of-the-art performance at detecting anomalous behaviour in computer server data.</details>

### Faster Discrete Convex Function Minimization with Predictions: The M-Convex Case

**Authors:** Taihei Oki, Shinsaku Sakaue

**<details><summary>Abstract**</summary>

Recent years have seen a growing interest in accelerating optimization algorithms with machine-learned predictions. Sakaue and Oki (NeurIPS 2022) have developed a general framework that warm-starts the *L-convex function minimization* method with predictions, revealing the idea's usefulness for various discrete optimization problems. In this paper, we present a framework for using predictions to accelerate *M-convex function minimization*, thus complementing previous research and extending the range of discrete optimization algorithms that can benefit from predictions. Our framework is particularly effective for an important subclass called *laminar convex minimization*, which appears in many operations research applications. Our methods can improve time complexity bounds upon the best worst-case results by using predictions and even have potential to go beyond a lower-bound result.</details>

### Feature learning via mean-field Langevin dynamics: classifying sparse parities and beyond

**Authors:** Taiji Suzuki, Denny Wu, Kazusato Oko, Atsushi Nitanda

**<details><summary>Abstract**</summary>

Neural network in the mean-field regime is known to be capable of \textit{feature learning}, unlike the kernel (NTK) counterpart. Recent works have shown that mean-field neural networks can be globally optimized by a noisy gradient descent update termed the \textit{mean-field Langevin dynamics} (MFLD). However, all existing guarantees for MFLD only considered the \textit{optimization} efficiency, and it is unclear if this algorithm leads to improved \textit{generalization} performance and sample complexity due to the presence of feature learning. To fill this gap, in this work we study the statistical and computational complexity of MFLD in learning a class of binary classification problems. Unlike existing margin bounds for neural networks, we avoid the typical norm control by utilizing the perspective that MFLD optimizes the \textit{distribution} of parameters rather than the parameter itself; this leads to an improved analysis of the sample complexity and convergence rate. We apply our general framework to the learning of $k$-sparse parity functions, where we prove that unlike kernel methods, two-layer neural networks optimized by MFLD achieves a sample complexity where the degree $k$ is ``decoupled'' from the exponent in the dimension dependence.</details>

### Fed-FA: Theoretically Modeling Client Data Divergence for Federated Language Backdoor Defense

**Authors:** Zhiyuan Zhang, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, Xu Sun

**<details><summary>Abstract**</summary>

Federated learning algorithms enable neural network models to be trained across multiple decentralized edge devices without sharing private data. However, they are susceptible to backdoor attacks launched by malicious clients. Existing robust federated aggregation algorithms heuristically detect and exclude suspicious clients based on their parameter distances, but they are ineffective on Natural Language Processing (NLP) tasks. The main reason is that, although text backdoor patterns are obvious at the underlying dataset level, they are usually hidden at the parameter level, since injecting backdoors into texts with discrete feature space has less impact on the statistics of the model parameters. To settle this issue, we propose to identify backdoor clients by explicitly modeling the data divergence among clients in federated NLP systems. Through theoretical analysis, we derive the f-divergence indicator to estimate the client data divergence with aggregation updates and Hessians. Furthermore, we devise a dataset synthesization method with a Hessian reassignment mechanism guided by the diffusion theory to address the key challenge of inaccessible datasets in calculating clients' data Hessians.We then present the novel Federated F-Divergence-Based Aggregation~(\textbf{Fed-FA}) algorithm, which leverages the f-divergence indicator to detect and discard suspicious clients. Extensive empirical results show that Fed-FA outperforms all the parameter distance-based methods in defending against backdoor attacks among various natural language backdoor attack scenarios.</details>

### FedNAR: Federated Optimization with Normalized Annealing Regularization

**Authors:** Junbo Li, Ang Li, Chong Tian, Qirong Ho, Eric Xing, Hongyi Wang

**<details><summary>Abstract**</summary>

Weight decay is a standard technique to improve generalization performance in modern deep neural network optimization, and is also widely adopted in federated learning (FL) to prevent overfitting in local clients. In this paper, we first explore the choices of weight decay and identify that weight decay value appreciably influences the convergence of existing FL algorithms. While preventing overfitting is crucial, weight decay can introduce a different optimization goal towards the global objective, which is further amplified in FL due to multiple local updates and heterogeneous data distribution.To address this challenge, we develop {\it Federated optimization with Normalized Annealing Regularization} (FedNAR), a simple yet effective and versatile algorithmic plug-in that can be seamlessly integrated into any existing FL algorithms. Essentially, we regulate the magnitude of each update by performing co-clipping of the gradient and weight decay.We provide a comprehensive theoretical analysis of FedNAR's convergence rate and conduct extensive experiments on both vision and language datasets with different backbone federated optimization algorithms. Our experimental results consistently demonstrate that incorporating FedNAR into existing FL algorithms leads to accelerated convergence and heightened model accuracy. Moreover, FedNAR exhibits resilience in the face of various hyperparameter configurations. Specifically, FedNAR has the ability to self-adjust the weight decay when the initial specification is not optimal, while the accuracy of traditional FL algorithms would markedly decline. Our codes are released at \href{https://anonymous.4open.science/r/fednar-BE8F}{https://anonymous.4open.science/r/fednar-BE8F}.</details>

### Federated Compositional Deep AUC Maximization

**Authors:** Xinwen Zhang, Yihan Zhang, Tianbao Yang, Richard Souvenir, Hongchang Gao

**<details><summary>Abstract**</summary>

Federated learning has attracted increasing attention due to the promise of balancing privacy and large-scale learning; numerous approaches have been proposed. However, most existing approaches focus on problems with balanced data, and prediction performance is far from satisfactory for many real-world applications where the number of samples in different classes is highly imbalanced. To address this challenging problem, we developed a novel federated learning method for imbalanced data by directly optimizing the area under curve (AUC) score. In particular, we formulate the AUC maximization problem as a federated compositional minimax optimization problem, develop a local stochastic compositional gradient descent ascent with momentum algorithm, and provide bounds on the computational and communication complexities of our algorithm. To the best of our knowledge, this is the first work to achieve such favorable theoretical results. Finally, extensive experimental results confirm the efficacy of our method.</details>

### Federated Learning with Bilateral Curation for Partially Class-Disjoint Data

**Authors:** Ziqing Fan, ruipeng zhang, Jiangchao Yao, Bo Han, Ya Zhang, Yanfeng Wang

**<details><summary>Abstract**</summary>

Partially class-disjoint data (PCDD), a common yet under-explored data formation where each client contributes a part of classes (instead of all classes) of samples, severely challenges the performance of federated algorithms. Without full classes, the local objective will contradict the global objective, yielding the angle collapse problem for locally missing classes and the space waste problem for locally existing classes. As far as we know, none of the existing methods can intrinsically mitigate PCDD challenges to achieve holistic improvement in the bilateral views (both global view and local view) of federated learning. To address this dilemma, we are inspired by the strong generalization of simplex Equiangular Tight Frame (ETF) on the imbalanced data, and propose a novel approach called FedGELA where the classifier is globally fixed as a simplex ETF while locally adapted to the personal distributions. Globally, FedGELA provides fair and equal discrimination for all classes and avoids inaccurate updates of the classifier, while locally it utilizes the space of locally missing classes for locally existing classes. We conduct extensive experiments on a range of datasets to demonstrate that our FedGELA achieves promising performance (averaged improvement of 3.9% to FedAvg and 1.5% to best baselines) and provide both local and global convergence guarantees.</details>

### FiGURe: Simple and Efficient Unsupervised Node Representations with Filter Augmentations

**Authors:** Chanakya Ekbote, Ajinkya Deshpande, Arun Iyer, SUNDARARAJAN SELLAMANICKAM, Ramakrishna Bairi

**<details><summary>Abstract**</summary>

Unsupervised node representations learnt using contrastive learning-based methods have shown good performance on downstream tasks. However, these methods rely on augmentations that mimic low-pass filters, limiting their performance on tasks requiring different eigen-spectrum parts. This paper presents a simple filter-based augmentation method to capture different parts of the eigen-spectrum. We show significant improvements using these augmentations. Further, we show that sharing the same weights across these different filter augmentations is possible, reducing the computational load. In addition, previous works have shown that good performance on downstream tasks requires high dimensional representations. Working with high dimensions increases the computations, especially when multiple augmentations are involved. We mitigate this problem and recover good performance through lower dimensional embeddings using simple random Fourier feature projections. Our method, FiGURe, achieves an average gain of up to 4.4\%, compared to the state-of-the-art unsupervised models, across all datasets in consideration, both homophilic and heterophilic. Our code can be found at: https://github.com/Microsoft/figure.</details>

### Fitting trees to $\ell_1$-hyperbolic distances

**Authors:** Joon-Hyeok Yim, Anna Gilbert

**<details><summary>Abstract**</summary>

Building trees to represent or to fit distances is a critical component of phylogenetic analysis, metric embeddings, approximation algorithms, geometric graph neural nets, and the analysis of hierarchical data. Much of the previous algorithmic work, however, has focused on generic metric spaces (i.e., those with no \emph{a priori} constraints). Leveraging several ideas from the mathematical analysis of hyperbolic geometry and geometric group theory, we study the tree fitting problem as finding the relation between the hyperbolicity (ultrametricity) vector and the error of tree (ultrametric) embedding. That is, we define a vector of hyperbolicity (ultrametric) values over all triples of points and compare the $\ell_p$ norms of this vector with the $\ell_q$ norm of the distortion of the best tree fit to the distances. This formulation allows us to define the average hyperbolicity (ultrametricity) in terms of a normalized $\ell_1$ norm of the hyperbolicity vector. Furthermore, we can interpret the classical tree fitting result of Gromov as a $p = q = \infty$ result. We present an algorithm \textsc{HCCRootedTreeFit} such that the $\ell_1$ error of the output embedding is analytically bounded in terms of the $\ell_1$-norm of the hyperbolicity vector (i.e., $p = q = 1$) and that this result is tight. Furthermore, this algorithm has significantly different theoretical and empirical performance as compared to Gromov's result and related algorithms. Finally, we show using \textsc{HCCRootedTreeFit} and related tree fitting algorithms, that supposedly standard data sets for hierarchical data analysis and geometric graph neural networks have radically different tree fits than those of synthetic, truly tree-like data sets, suggesting that a much more refined analysis of these standard data sets is called for.</details>

### Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models

**Authors:** Haonan Duan, Adam Dziedzic, Nicolas Papernot, Franziska Boenisch

**<details><summary>Abstract**</summary>

Large language models (LLMs) are excellent in-context learners. However, the sensitivity of data contained in prompts raises privacy concerns. Our work first shows that these concerns are valid: we instantiate a simple but highly effective membership inference attack against the data used to prompt LLMs. To address this vulnerability, one could forego prompting and resort to fine-tuning LLMs with known algorithms for private gradient descent. However, this comes at the expense of the practicality and efficiency offered by prompting. Therefore, we propose to privately learn to prompt. We first show that soft prompts can be obtained privately through gradient descent on downstream data. However, this is not the case for discrete prompts. Thus, we orchestrate a noisy vote among an ensemble of LLMs presented with different prompts, i.e., a flock of stochastic parrots. The vote privately transfers the flock's knowledge into a single public prompt. We show that LLMs prompted with our private algorithms closely match the non-private baselines. For example, using GPT3 as the base model, we achieve a downstream accuracy of 92.7% on the sst2 dataset with $(\varepsilon=0.147, \delta=10^{-6})$-differential privacy vs. 95.2% for the non-private baseline. Through our experiments, we also show that our prompt-based approach is easily deployed with existing commercial~APIs.</details>

### Focus Your Attention when Few-Shot Classification

**Authors:** Haoqing Wang, Shibo Jie, Zhihong Deng

**<details><summary>Abstract**</summary>

Since many pre-trained vision transformers emerge and provide strong representation for various downstream tasks, we aim to adapt them to few-shot image classification tasks in this work. The input images typically contain multiple entities. The model may not focus on the class-related entities for the current few-shot task, even with fine-tuning on support samples, and the noise information from the class-independent ones harms performance. To this end, we first propose a method that uses the attention and gradient information to automatically locate the positions of key entities, denoted as position prompts, in the support images. Then we employ the cross-entropy loss between their many-hot presentation and the attention logits to optimize the model to focus its attention on the key entities during fine-tuning. This ability then can generalize to the query samples. Our method is applicable to different vision transformers (e.g., columnar or pyramidal ones), and also to different pre-training ways (e.g., single-modal or vision-language pre-training). Extensive experiments show that our method can improve the performance of full or parameter-efficient fine-tuning methods on few-shot tasks. Code is available at https://github.com/Haoqing-Wang/FORT.</details>

### ForecastPFN: Synthetically-Trained Zero-Shot Forecasting

**Authors:** Samuel Dooley, Gurnoor Singh Khurana, Chirag Mohapatra, Siddartha V Naidu, Colin White

**<details><summary>Abstract**</summary>

The vast majority of time-series forecasting approaches require a substantial training dataset. However, many real-life forecasting applications have very little initial observations, sometimes just 40 or fewer. Thus, the applicability of most forecasting methods is restricted in data-sparse commercial applications. While there is recent work in the setting of very limited initial data (so-called `zero-shot' forecasting), its performance is inconsistent depending on the data used for pretraining. In this work, we take a different approach and devise ForecastPFN, the first zero-shot forecasting model trained purely on a novel synthetic data distribution. ForecastPFN is a prior-data fitted network, trained to approximate Bayesian inference, which can make predictions on a new time series dataset in a single forward pass. Through extensive experiments, we show that zero-shot predictions made by ForecastPFN are more accurate and faster compared to state-of-the-art forecasting methods, even when the other methods are allowed to train on hundreds of additional in-distribution data points.</details>

### Fundamental Limits and Tradeoffs in Invariant Representation Learning

**Authors:** Han Zhao, Chen Dan, Bryon Aragam, Tommi Jaakkola, Geoffrey Gordon, Pradeep Ravikumar

**<details><summary>Abstract**</summary>

A wide range of machine learning applications such as privacy-preserving learning, algorithmic fairness, and domain adaptation/generalization among others, involve learning invariant representations of the data that aim to achieve two competing goals: (a) maximize information or accuracy with respect to a target response, and (b) maximize invariance or independence with respect to a set of protected features (e.g.\ for fairness, privacy, etc). Despite their wide applicability, theoretical understanding of the optimal tradeoffs --- with respect to accuracy, and invariance --- achievable by invariant representations is still severely lacking. In this paper, we provide an information theoretic analysis of such tradeoffs under both classification and regression settings. More precisely, we provide a geometric characterization of the accuracy and invariance achievable by any representation of the data; we term this feasible region the information plane. We provide an inner bound for this feasible region for the classification case, and an exact characterization for the regression case, which allows us to either bound or exactly characterize the Pareto optimal frontier between accuracy and invariance. Although our contributions are mainly theoretical, a key practical application of our results is in certifying the potential sub-optimality of any given representation learning algorithm for either classification or regression tasks. Our results shed new light on the fundamental interplay between accuracy and invariance, and may be useful in guiding the design of future representation learning algorithms.</details>

### Fused Gromov-Wasserstein Graph Mixup for Graph-level Classifications

**Authors:** Xinyu Ma, Xu Chu, Yasha Wang, Yang Lin, Junfeng Zhao, Liantao Ma, Wenwu Zhu

**<details><summary>Abstract**</summary>

Graph data augmentation has shown superiority in enhancing generalizability and robustness of GNNs in graph-level classifications. However, existing methods primarily focus on the augmentation in the graph signal space and the graph structure space independently, neglecting the joint interaction between them. In this paper, we address this limitation by formulating the problem as an optimal transport problem that aims to find an optimal inter-graph node matching strategy considering the interactions between graph structures and signals. To solve this problem, we propose a novel graph mixup algorithm called FGWMixup, which seeks a "midpoint" of source graphs in the Fused Gromov-Wasserstein (FGW) metric space. To enhance the scalability of our method, we introduce a relaxed FGW solver that accelerates FGWMixup by improving the convergence rate from $\mathcal{O}(t^{-1})$ to $\mathcal{O}(t^{-2})$. Extensive experiments conducted on five datasets using both classic (MPNNs) and advanced (Graphormers) GNN backbones demonstrate that \mname\xspace effectively improves the generalizability and robustness of GNNs. Codes are available at https://github.com/ArthurLeoM/FGWMixup.</details>

### GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection

**Authors:** Jianheng Tang, Fengrui Hua, Ziqi Gao, Peilin Zhao, Jia Li

**<details><summary>Abstract**</summary>

With a long history of traditional Graph Anomaly Detection (GAD) algorithms and recently popular Graph Neural Networks (GNNs), it is still not clear (1) how they perform under a standard comprehensive setting, (2) whether GNNs can outperform traditional algorithms such as tree ensembles, and (3) how about their efficiency on large-scale graphs. In response, we introduce GADBench---a benchmark tool dedicated to supervised anomalous node detection in static graphs. GADBench facilitates a detailed comparison across 29 distinct models on ten real-world GAD datasets, encompassing thousands to millions (~6M) nodes. Our main finding is that tree ensembles with simple neighborhood aggregation can outperform the latest GNNs tailored for the GAD task. We shed light on the current progress of GAD, setting a robust groundwork for subsequent investigations in this domain. GADBench is open-sourced at https://github.com/squareRoot3/GADBench.</details>

### GAUCHE: A Library for Gaussian Processes in Chemistry

**Authors:** Ryan-Rhys Griffiths, Leo Klarner, Henry Moss, Aditya Ravuri, Sang Truong, Yuanqi Du, Samuel Stanton, Gary Tom, Bojana Rankovic, Arian Jamasb, Aryan Deshwal, Julius Schwartz, Austin Tripp, Gregory Kell, Simon Frieder, Anthony Bourached, Alex Chan, Jacob Moss, Chengzhi Guo, Johannes Peter Dürholt, Saudamini Chaurasia, Ji Won Park, Felix Strieth-Kalthoff, Alpha Lee, Bingqing Cheng, Alan Aspuru-Guzik, Philippe Schwaller, Jian Tang

**<details><summary>Abstract**</summary>

We introduce GAUCHE, an open-source library for GAUssian processes in CHEmistry. Gaussian processes have long been a cornerstone of probabilistic machine learning, affording particular advantages for uncertainty quantification and Bayesian optimisation. Extending Gaussian processes to molecular representations, however, necessitates kernels defined over structured inputs such as graphs, strings and bit vectors. By providing such kernels in a modular, robust and easy-to-use framework, we seek to enable expert chemists and materials scientists to make use of state-of-the-art black-box optimization techniques. Motivated by scenarios frequently encountered in practice, we showcase applications for GAUCHE in molecular discovery, chemical reaction optimisation and protein design. The codebase is made available at https://github.com/leojklarner/gauche.</details>

### GEO-Bench: Toward Foundation Models for Earth Monitoring

**Authors:** Alexandre Lacoste, Nils Lehmann, Pau Rodriguez, Evan Sherwin, Hannah Kerner, Björn Lütjens, Jeremy Irvin, David Dao, Hamed Alemohammad, Alexandre Drouin, Mehmet Gunturkun, Gabriel Huang, David Vazquez, Dava Newman, Yoshua Bengio, Stefano Ermon, Xiaoxiang Zhu

**<details><summary>Abstract**</summary>

Recent progress in self-supervision has shown that pre-training large neural networks on vast amounts of unsupervised data can lead to substantial increases in generalization to downstream tasks. Such models, recently coined foundation models, have been transformational to the field of natural language processing.Variants have also been proposed for image data, but their applicability to remote sensing tasks is limited.To stimulate the development of foundation models for Earth monitoring, we propose a benchmark comprised of six classification and six segmentation tasks, which were carefully curated and adapted to be both relevant to the field and well-suited for model evaluation. We accompany this benchmark with a robust methodology for evaluating models and reporting aggregated results to enable a reliable assessment of progress. Finally, we report results for 20 baselines to gain information about the performance of existing models.We believe that this benchmark will be a driver of progress across a variety of Earth monitoring tasks.</details>

### GEX: A flexible method for approximating influence via Geometric Ensemble

**Authors:** SungYub Kim, Kyungsu Kim, Eunho Yang

**<details><summary>Abstract**</summary>

Through a deeper understanding of predictions of neural networks, Influence Function (IF) has been applied to various tasks such as detecting and relabeling mislabeled samples, dataset pruning, and separation of data sources in practice. However, we found standard approximations of IF suffer from performance degradation due to oversimplified influence distributions caused by their bilinear approximation, suppressing the expressive power of samples with a relatively strong influence. To address this issue, we propose a new interpretation of existing IF approximations as an average relationship between two linearized losses over parameters sampled from the Laplace approximation (LA). In doing so, we highlight two significant limitations of current IF approximations: the linearity of gradients and the singularity of Hessian. Accordingly, by improving each point, we introduce a new IF approximation method with the following features: i) the removal of linearization to alleviate the bilinear constraint and ii) the utilization of Geometric Ensemble (GE) tailored for non-linear losses. Empirically, our approach outperforms existing IF approximations for downstream tasks with lighter computation, thereby providing new feasibility of low-complexity/nonlinear-based IF design.</details>

### GLEMOS: Benchmark for Instantaneous Graph Learning Model Selection

**Authors:** Namyong Park, Ryan Rossi, Xing Wang, Antoine Simoulin, Nesreen K. Ahmed, Christos Faloutsos

**<details><summary>Abstract**</summary>

The choice of a graph learning (GL) model (i.e., a GL algorithm and its hyperparameter settings) has a significant impact on the performance of downstream tasks. However, selecting the right GL model becomes increasingly difficult and time consuming as more and more GL models are developed. Accordingly, it is of great significance and practical value to equip users of GL with the ability to perform a near-instantaneous selection of an effective GL model without manual intervention. Despite the recent attempts to tackle this important problem, there has been no comprehensive benchmark environment to evaluate the performance of GL model selection methods. To bridge this gap, we present GLEMOS in this work, a comprehensive benchmark for instantaneous GL model selection that makes the following contributions. (i) GLEMOS provides extensive benchmark data for fundamental GL tasks, i.e., link prediction and node classification, including the performances of 366 models on 457 graphs on these tasks. (ii) GLEMOS designs multiple evaluation settings, and assesses how effectively representative model selection techniques perform in these different settings. (iii) GLEMOS is designed to be easily extended with new models, new graphs, and new performance records. (iv) Based on the experimental results, we discuss the limitations of existing approaches and highlight future research directions. To promote research on this significant problem, we make the benchmark data and code publicly available at https://namyongpark.github.io/glemos.</details>

### GenEval: An object-focused framework for evaluating text-to-image alignment

**Authors:** Dhruba Ghosh, Hannaneh Hajishirzi, Ludwig Schmidt

**<details><summary>Abstract**</summary>

Recent breakthroughs in diffusion models, multimodal pretraining, and efficient finetuning have led to an explosion of text-to-image generative models. Given human evaluation is expensive and difficult to scale, automated methods are critical for evaluating the increasingly large number of new models. However, most current automated evaluation metrics like FID or CLIPScore only offer a distribution-level measure of image quality or image-text alignment, and are unsuited for fine-grained or instance-level analysis. In this paper, we introduce GenEval, an object-focused framework to evaluate compositional image properties such as object co-occurrence, position, count, and color. We show that current object detection models can be leveraged to evaluate text-to-image models on a variety of generation tasks with strong human agreement, and that other discriminative vision models can be linked to this pipeline to further verify properties like object color. We then evaluate several open-source text-to-image models and analyze their relative reasoning capabilities on our benchmark. We find that recent models demonstrate significant improvement on these tasks, though they are still lacking in complex capabilities such as spatial relations and attribute binding. Finally, we demonstrate how GenEval might be used to help discover existing failure modes, in order to inform development of the next generation of text-to-image models. Our code to run the GenEval framework will be made publicly available at https://github.com/djghosh13/geneval.</details>

### Generalized Information-theoretic Multi-view Clustering

**Authors:** Weitian Huang, Sirui Yang, Hongmin Cai

**<details><summary>Abstract**</summary>

In an era of more diverse data modalities, multi-view clustering has become a fundamental tool for comprehensive data analysis and exploration. However, existing multi-view unsupervised learning methods often rely on strict assumptions on semantic consistency among samples. In this paper, we reformulate the multi-view clustering problem from an information-theoretic perspective and propose a general theoretical model. In particular, we define three desiderata under multi-view unsupervised learning in terms of mutual information, namely, comprehensiveness, concentration, and cross-diversity. The multi-view variational lower bound is then obtained by approximating the samples' high-dimensional mutual information. The Kullback–Leibler divergence is utilized to deduce sample assignments. Ultimately the information-based multi-view clustering model leverages deep neural networks and Stochastic Gradient Variational Bayes to achieve representation learning and clustering simultaneously. Extensive experiments on both synthetic and real datasets with wide types demonstrate that the proposed method exhibits a more stable and superior clustering performance than state-of-the-art algorithms.</details>

### Generalized Weighted Path Consistency for Mastering Atari Games

**Authors:** Dengwei Zhao, Shikui Tu, Lei Xu

**<details><summary>Abstract**</summary>

Reinforcement learning with the help of neural-guided search consumes huge computational resources to achieve remarkable performance. Path consistency (PC), i.e., $f$ values on one optimal path should be identical, was previously imposed on MCTS by PCZero to improve the learning efficiency of AlphaZero. Not only  PCZero still lacks a theoretical support but also considers  merely board games. In this paper,  PCZero is  generalized into GW-PCZero for  real applications with non-zero immediate reward. A weighting mechanism is introduced to reduce the variance caused by scouting's uncertainty on the $f$ value estimation. For the first time, it is theoretically proved that neural-guided MCTS is guaranteed to find the optimal solution under the constraint of PC. Experiments are conducted on the Atari $100$k benchmark with $26$ games and GW-PCZero achieves $198\%$ mean human performance, higher than the state-of-the-art EfficientZero's $194\\%$, while consuming only $25\\%$ of the computational resources consumed by EfficientZero.</details>

### Generating QM1B with PySCF$_{\text{IPU}}$

**Authors:** Alexander Mathiasen, Hatem Helal, Kerstin Klaser, Paul Balanca, Josef Dean, Carlo Luschi, Dominique Beaini, Andrew Fitzgibbon, Dominic Masters

**<details><summary>Abstract**</summary>

The emergence of foundation models in Computer Vision and Natural Language Processing have resulted in immense progress on downstream tasks.  This progress was enabled by datasets with billions of training examples. Similar benefits are yet to be unlocked for quantum chemistry, where the potential of deep learning is constrained by comparatively small datasets with 100k to 20M training examples. These datasets are limited in size because the labels are computed using the accurate (but computationally demanding) predictions of Density Functional Theory (DFT). Notably, prior DFT datasets were created using CPU supercomputers without leveraging hardware acceleration.  In this paper, we take a first step towards utilising hardware accelerators by introducing the data generator PySCF$_{\text{IPU}}$ using Intelligence Processing Units (IPUs). This allows us to create the dataset QM1B with one billion training examples containing 9-11 heavy atoms. We demonstrate that a simple baseline neural network (SchNet 9M) improves its performance by simply increasing the amount of training data without additional inductive biases. To encourage future researchers to use QM1B responsibly, we highlight several limitations of QM1B and emphasise the low resolution of our DFT options, which also serves as motivation for even larger, more accurate datasets.</details>

### Generator Identification for Linear SDEs with Additive and Multiplicative Noise

**Authors:** Yuanyuan Wang, Xi Geng, Wei Huang, Biwei Huang, Mingming Gong

**<details><summary>Abstract**</summary>

In this paper, we present conditions for identifying the generator of a linear stochastic differential equation (SDE) from the distribution of its solution process with a given fixed initial state. These identifiability conditions are crucial in causal inference using linear SDEs as they enable the identification of the post-intervention distributions from its observational distribution. Specifically, we derive a sufficient and necessary condition for identifying the generator of linear SDEs with additive noise, as well as a sufficient condition for identifying the generator of linear SDEs with multiplicative noise. We show that the conditions derived for both types of SDEs are generic. Moreover, we offer geometric interpretations of the derived identifiability conditions to enhance their understanding. To validate our theoretical results, we perform a series of simulations, which support and substantiate the established findings.</details>

### GeoDE: a Geographically Diverse Evaluation Dataset for Object Recognition

**Authors:** Vikram V. Ramaswamy, Sing Yu Lin, Dora Zhao, Aaron Adcock, Laurens van der Maaten, Deepti Ghadiyaram, Olga Russakovsky

**<details><summary>Abstract**</summary>

Current dataset collection methods typically scrape large amounts of data from the web. While this technique is extremely scalable, data collected in this way tends to reinforce stereotypical biases, can contain personally identifiable information, and typically originates from Europe and North America. In this work, we rethink the dataset collection paradigm and introduce GeoDE, a geographically diverse dataset with 61,940 images from 40 classes and 6 world regions, and no personally identifiable information, collected by soliciting images from people across the world. We analyse GeoDE to understand differences in images collected in this manner compared to web-scraping. Despite the smaller size of this dataset, we demonstrate its use as both an evaluation and training dataset, allowing us to highlight shortcomings in current models, as well as demonstrate improved performance even when training on this small dataset. We release the full dataset and code at https://geodiverse-data-collection.cs.princeton.edu/</details>

### Gigastep - One Billion Steps per Second Multi-agent Reinforcement Learning

**Authors:** Mathias Lechner, lianhao yin, Tim Seyde, Tsun-Hsuan Johnson Wang, Wei Xiao, Ramin Hasani, Joshua Rountree, Daniela Rus

**<details><summary>Abstract**</summary>

Multi-agent reinforcement learning (MARL) research is faced with a trade-off: it either uses complex environments requiring large compute resources, which makes it inaccessible to researchers with limited resources, or relies on simpler dynamics for faster execution, which makes the transferability of the results to more realistic tasks challenging. Motivated by these challenges, we present Gigastep, a fully vectorizable, MARL environment implemented in JAX, capable of executing up to one billion environment steps per second on consumer-grade hardware. Its design allows for comprehensive MARL experimentation, including a complex, high-dimensional space defined by 3D dynamics, stochasticity, and partial observations. Gigastep supports both collaborative and adversarial tasks, continuous and discrete action spaces, and provides RGB image and feature vector observations, allowing the evaluation of a wide range of MARL algorithms. We validate Gigastep's usability through an extensive set of experiments, underscoring its role in widening participation and promoting inclusivity in the MARL research community.</details>

### Going Beyond Linear Mode Connectivity: The Layerwise Linear Feature Connectivity

**Authors:** Zhanpeng Zhou, Yongyi Yang, Xiaojiang Yang, Junchi Yan, Wei Hu

**<details><summary>Abstract**</summary>

Recent work has revealed many intriguing empirical phenomena in neural network training, despite the poorly understood and highly complex loss landscapes and training dynamics. One of these phenomena, Linear Mode Connectivity (LMC), has gained considerable attention due to the intriguing observation that different solutions can be connected by a linear path in the parameter space while maintaining near-constant training and test losses. In this work, we introduce a stronger notion of linear connectivity, Layerwise Linear Feature Connectivity (LLFC), which says that the feature maps of every layer in different trained networks are also linearly connected. We provide comprehensive empirical evidence for LLFC across a wide range of settings, demonstrating that whenever two trained networks satisfy LMC (via either spawning or permutation methods), they also satisfy LLFC in nearly all the layers. Furthermore, we delve deeper into the underlying factors contributing to LLFC, which reveal new insights into the permutation approaches. The study of LLFC transcends and advances our understanding of LMC by adopting a feature-learning perspective.</details>

### Granger Components Analysis: Unsupervised learning of latent temporal dependencies

**Authors:** Jacek Dmochowski

**<details><summary>Abstract**</summary>

A new technique for unsupervised learning of time series data based on the notion of Granger causality is presented. The technique learns pairs of projections of a multivariate data set such that the resulting components -- "driving" and "driven" -- maximize the strength of the Granger causality between the latent time series (how strongly the past of the driving signal predicts the present of the driven signal). A coordinate descent algorithm that learns pairs of coefficient vectors in an alternating fashion is developed and shown to blindly identify the underlying sources (up to scale) on simulated vector autoregressive (VAR) data. The technique is tested on scalp electroencephalography (EEG) data from a motor imagery experiment where the resulting components lateralize with the side of the cued hand, and also on functional magnetic resonance imaging (fMRI) data, where the recovered components express previously reported resting-state networks.</details>

### Graph Clustering with Graph Neural Networks

**Authors:** Anton Tsitsulin, John Palowitch, Bryan Perozzi, Emmanuel Müller

**<details><summary>Abstract**</summary>

Graph Neural Networks (GNNs) have achieved state-of-the-art results on many graph analysis tasks such as node classification and link prediction. However, important unsupervised problems on graphs, such as graph clustering, have proved more resistant to advances in GNNs. Graph clustering has the same overall goal as node pooling in GNNs—does this mean that GNN pooling methods do a good job at clustering graphs? Surprisingly, the answer is no—current GNN pooling methods often fail to recover the cluster structure in cases where simple baselines, such as k-means applied on learned representations, work well. We investigate further by carefully designing a set of experiments to study different signal-to-noise scenarios both in graph structure and attribute data. To address these methods' poor performance in clustering, we introduce Deep Modularity Networks (DMoN), an unsupervised pooling method inspired by the modularity measure of clustering quality, and show how it tackles recovery of the challenging clustering structure of real-world graphs. Similarly, on real-world data, we show that DMoN produces high quality clusters which correlate strongly with ground truth labels, achieving state-of-the-art results with over 40% improvement over other pooling methods across different metrics.</details>

### Graph Denoising Diffusion for Inverse Protein Folding

**Authors:** Kai Yi, Bingxin Zhou, Yiqing Shen, Pietro Lió, Yuguang Wang

**<details><summary>Abstract**</summary>

Inverse protein folding is challenging due to its inherent one-to-many mapping characteristic, where numerous possible amino acid sequences can fold into a single, identical protein backbone. This task involves not only identifying viable sequences but also representing the sheer diversity of potential solutions. However, existing discriminative models, such as transformer-based auto-regressive models, struggle to encapsulate the diverse range of plausible solutions. In contrast, diffusion probabilistic models, as an emerging genre of generative approaches, offer the potential to generate a diverse set of sequence candidates for determined protein backbones. We propose a novel graph denoising diffusion model for inverse protein folding, where a given protein backbone guides the diffusion process on the corresponding amino acid residue types. The model infers the joint distribution of amino acids conditioned on the nodes' physiochemical properties and local environment. Moreover, we utilize amino acid replacement matrices for the diffusion forward process, encoding the biologically-meaningful prior knowledge of amino acids from their spatial and sequential neighbors as well as themselves, which reduces the sampling space of the generative process. Our model achieves state-of-the-art performance over a set of popular baseline methods in sequence recovery and exhibits great potential in generating diverse protein sequences for a determined protein backbone structure.</details>

### Graph Neural Networks for Road Safety Modeling: Datasets and Evaluations for Accident Analysis

**Authors:** Abhinav Nippani, Dongyue Li, Haotian Ju, Haris Koutsopoulos, Hongyang Zhang

**<details><summary>Abstract**</summary>

We consider the problem of traffic accident analysis on a road network based on road network connections and traffic volume. Previous works have designed various deep-learning methods using historical records to predict traffic accident occurrences. However, there is a lack of consensus on how accurate existing methods are, and a fundamental issue is the lack of public accident datasets for comprehensive evaluations. This paper constructs a large-scale, unified dataset of traffic accident records from official reports of various states in the US, totaling 9 million records, accompanied by road networks and traffic volume reports. Using this new dataset, we evaluate existing deep-learning methods for predicting the occurrence of accidents on road networks. Our main finding is that graph neural networks such as GraphSAGE can accurately predict the number of accidents on roads with less than 22% mean absolute error (relative to the actual count) and whether an accident will occur or not with over 87% AUROC, averaged over states. We achieve these results by using multitask learning to account for cross-state variabilities (e.g., availability of accident labels) and transfer learning to combine traffic volume with accident prediction. Ablation studies highlight the importance of road graph-structural features, amongst other features. Lastly, we discuss the implications of the analysis and develop a package for easily using our new dataset.</details>

### H2RBox-v2: Incorporating Symmetry for Boosting Horizontal Box Supervised Oriented Object Detection

**Authors:** Yi Yu, Xue Yang, Qingyun Li, Yue Zhou, Feipeng Da, Junchi Yan

**<details><summary>Abstract**</summary>

With the rapidly increasing demand for oriented object detection, e.g. in autonomous driving and remote sensing, the recently proposed paradigm involving weakly-supervised detector H2RBox for learning rotated box (RBox) from the more readily-available horizontal box (HBox) has shown promise. This paper presents H2RBox-v2, to further bridge the gap between HBox-supervised and RBox-supervised oriented object detection. Specifically, we propose to leverage the reflection symmetry via flip and rotate consistencies, using a weakly-supervised network branch similar to H2RBox, together with a novel self-supervised branch that learns orientations from the symmetry inherent in visual objects. The detector is further stabilized and enhanced by practical techniques to cope with peripheral issues e.g. angular periodicity. To our best knowledge, H2RBox-v2 is the first symmetry-aware self-supervised paradigm for oriented object detection. In particular, our method shows less susceptibility to low-quality annotation and insufficient training data compared to H2RBox. Specifically, H2RBox-v2 achieves very close performance to a rotation annotation trained counterpart -- Rotated FCOS: 1) DOTA-v1.0/1.5/2.0: 72.31%/64.76%/50.33% vs. 72.44%/64.53%/51.77%; 2) HRSC: 89.66% vs. 88.99%; 3) FAIR1M: 42.27% vs. 41.25%.</details>

### HASSOD: Hierarchical Adaptive Self-Supervised Object Detection

**Authors:** Shengcao Cao, Dhiraj Joshi, Liangyan Gui, Yu-Xiong Wang

**<details><summary>Abstract**</summary>

The human visual perception system demonstrates exceptional capabilities in learning without explicit supervision and understanding the part-to-whole composition of objects. Drawing inspiration from these two abilities, we propose Hierarchical Adaptive Self-Supervised Object Detection (HASSOD), a novel approach that learns to detect objects and understand their compositions without human supervision. HASSOD employs a hierarchical adaptive clustering strategy to group regions into object masks based on self-supervised visual representations, adaptively determining the number of objects per image. Furthermore, HASSOD identifies the hierarchical levels of objects in terms of composition, by analyzing coverage relations between masks and constructing tree structures. This additional self-supervised learning task leads to improved detection performance and enhanced interpretability. Lastly, we abandon the inefficient multi-round self-training process utilized in prior methods and instead adapt the Mean Teacher framework from semi-supervised learning, which leads to a smoother and more efficient training process. Through extensive experiments on prevalent image datasets, we demonstrate the superiority of HASSOD over existing methods, thereby advancing the state of the art in self-supervised object detection. Notably, we improve Mask AR from 20.2 to 22.5 on LVIS, and from 17.0 to 26.0 on SA-1B. Code is available at https://github.com/Shengcao-Cao/HASSOD.</details>

### HOH: Markerless Multimodal Human-Object-Human Handover Dataset with Large Object Count

**Authors:** Noah Wiederhold, Ava Megyeri, DiMaggio Paris, Sean Banerjee, Natasha Banerjee

**<details><summary>Abstract**</summary>

We present the HOH (Human-Object-Human) Handover Dataset, a large object count dataset with 136 objects, to accelerate data-driven research on handover studies, human-robot handover implementation, and artificial intelligence (AI) on handover parameter estimation from 2D and 3D data of two-person interactions. HOH contains multi-view RGB and depth data, skeletons, fused point clouds, grasp type and handedness labels, object, giver hand, and receiver hand 2D and 3D segmentations, giver and receiver comfort ratings, and paired object metadata and aligned 3D models for 2,720 handover interactions spanning 136 objects and 20 giver-receiver pairs—40 with role-reversal—organized from 40 participants. We also show experimental results of neural networks trained using HOH to perform grasp, orientation, and trajectory prediction. As the only fully markerless handover capture dataset, HOH represents natural human-human handover interactions, overcoming challenges with markered datasets that require specific suiting for body tracking, and lack high-resolution hand tracking. To date, HOH is the largest handover dataset in terms of object count, participant count, pairs with role reversal accounted for, and total interactions captured.</details>

### Hierarchical Multi-Agent Skill Discovery

**Authors:** Mingyu Yang, Yaodong Yang, Zhenbo Lu, Wengang Zhou, Houqiang Li

**<details><summary>Abstract**</summary>

Skill discovery has shown significant progress in unsupervised reinforcement learning. This approach enables the discovery of a wide range of skills without any extrinsic reward, which can be effectively combined to tackle complex tasks. However, such unsupervised skill learning has not been well applied to multi-agent reinforcement learning (MARL) due to two primary challenges. One is how to learn skills not only for the individual agents but also for the entire team, and the other is how to coordinate the skills of different agents to accomplish multi-agent tasks. To address these challenges, we present Hierarchical Multi-Agent Skill Discovery (HMASD), a two-level hierarchical algorithm for discovering both team and individual skills in MARL. The high-level policy employs a transformer structure to realize sequential skill assignment, while the low-level policy learns to discover valuable team and individual skills. We evaluate HMASD on sparse reward multi-agent benchmarks, and the results show that HMASD achieves significant performance improvements compared to strong MARL baselines.</details>

### [Spotlight] High-dimensional Asymptotics of Denoising Autoencoders

**Authors:** Hugo Cui, Lenka Zdeborová

**<details><summary>Abstract**</summary>

We address the problem of denoising data from a Gaussian mixture using a two-layer non-linear autoencoder with tied weights and a skip connection. We consider the high-dimensional limit where the number of training samples and the input dimension jointly tend to infinity while the number of hidden units remains bounded. We provide closed-form expressions for the denoising mean-squared test error. Building on this result, we quantitatively characterize the advantage of the considered architecture over the autoencoder without the skip connection that relates closely to principal component analysis. We further show that our results capture accurately the learning curves on a range of real datasets.</details>

### Higher-Order Uncoupled Dynamics Do Not Lead to Nash Equilibrium - Except When They Do

**Authors:** Sarah Toonsi, Jeff Shamma

**<details><summary>Abstract**</summary>

The framework of multi-agent learning explores the dynamics of how an agent's strategies evolve in response to the evolving strategies of other agents. Of particular interest is whether or not agent strategies converge to well known solution concepts such as Nash Equilibrium (NE). In "higher order'' learning, agent dynamics include auxiliary states that can capture phenomena such as path dependencies. We introduce higher-order gradient play dynamics that resemble projected gradient ascent with auxiliary states. The dynamics are "payoff based'' and "uncoupled'' in that each agent's dynamics depend on its own evolving payoff and has no explicit dependence on the utilities of other agents. We first show that for any specific game with an isolated completely mixed-strategy NE, there exist higher-order gradient play dynamics that lead (locally) to that NE, both for the specific game and nearby games with perturbed utility functions. Conversely, we show that for any higher-order gradient play dynamics, there exists a game with a unique isolated completely mixed-strategy NE for which the dynamics do not lead to NE. Finally, we show that convergence to the mixed-strategy equilibrium in coordination games, comes at the expense of the dynamics being inherently internally unstable.</details>

### Hokoff: Real Game Dataset from Honor of Kings and its Offline Reinforcement Learning Benchmarks

**Authors:** Yun Qu, Boyuan Wang, Jianzhun Shao, Yuhang Jiang, Chen Chen, Zhenbin Ye, Liu Linc, Yang Feng, Lin Lai, Hongyang Qin, Minwen Deng, Juchao Zhuo, Deheng Ye, Qiang Fu, YANG GUANG, Wei Yang, Lanxiao Huang, Xiangyang Ji

**<details><summary>Abstract**</summary>

The advancement of Offline Reinforcement Learning (RL) and Offline Multi-Agent Reinforcement Learning (MARL) critically depends on the availability of high-quality, pre-collected offline datasets that represent real-world complexities and practical applications. However, existing datasets often fall short in their simplicity and lack of realism. To address this gap, we propose Hokoff, a comprehensive set of pre-collected datasets that covers both offline RL and offline MARL, accompanied by a robust framework, to facilitate further research. This data is derived from Honor of Kings, a recognized Multiplayer Online Battle Arena (MOBA) game known for its intricate nature, closely resembling real-life situations. Utilizing this framework, we benchmark a variety of offline RL and offline MARL algorithms. We also introduce a novel baseline algorithm tailored for the inherent hierarchical action space of the game. We reveal the incompetency of current offline RL approaches in handling task complexity, generalization and multi-task learning.</details>

### How hard are computer vision datasets? Calibrating dataset difficulty to viewing time

**Authors:** David Mayo, Jesse Cummings, Xinyu Lin, Dan Gutfreund, Boris Katz, Andrei Barbu

**<details><summary>Abstract**</summary>

Humans outperform object recognizers despite the fact that models perform well on current datasets, including those explicitly designed to challenge machines with debiased images or distribution shift. This problem persists, in part, because we have no guidance on the absolute difficulty of an image or dataset making it hard to objectively assess progress toward human-level performance, to cover the range of human abilities, and to increase the challenge posed by a dataset. We develop a dataset difficulty metric MVT, Minimum Viewing Time, that addresses these three problems. Subjects view an image that flashes on screen and then classify the object in the image. Images that require brief flashes to recognize are easy, those which require seconds of viewing are hard. We compute the ImageNet and ObjectNet image difficulty distribution, which we find significantly undersamples hard images. Nearly 90% of current benchmark performance is derived from images that are easy for humans. Rather than hoping that we will make harder datasets, we can for the first time objectively guide dataset difficulty during development. We can also subset recognition performance as a function of difficulty: model performance drops precipitously while human performance remains stable. Difficulty provides a new lens through which to view model performance, one which uncovers new scaling laws: vision-language models stand out as being the most robust and human-like while all other techniques scale poorly. We release tools to automatically compute MVT, along with image sets which are tagged by difficulty. Objective image difficulty has practical applications – one can measure how hard a test set is before deploying a real-world system – and scientific applications such as discovering the neural correlates of image difficulty and enabling new object recognition techniques that eliminate the benchmark-vs- real-world performance gap.</details>

### How to Data in Datathons

**Authors:** Carlos Mougan, Richard Plant, Clare Teng, Marya Bazzi, Alvaro Cabrejas Egea, Ryan Chan, David Salvador Jasin, Martin Stoffel, Kirstie Whitaker, JULES MANSER

**<details><summary>Abstract**</summary>

The rise of datathons, also known as data or data science hackathons, has provided a platform to collaborate, learn, and innovate quickly. Despite their significant potential benefits, organizations often struggle to effectively work with data due to a lack of clear guidelines and best practices for potential issues that might arise. Drawing on our own experiences and insights from organizing +80 datathon challenges with +60 partnership organizations since 2016, we provide a guide that serves as a resource for organizers to navigate the data-related complexities of datathons. We apply our proposed framework to 10 case studies.</details>

### HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face

**Authors:** Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, Yueting Zhuang

**<details><summary>Abstract**</summary>

Solving complicated AI tasks with different domains and modalities is a key step toward artificial general intelligence. While there are numerous AI models available for various domains and modalities, they cannot handle complicated AI tasks autonomously. Considering large language models (LLMs) have exhibited exceptional abilities in language understanding, generation, interaction, and reasoning, we advocate that LLMs could act as a controller to manage existing AI models to solve complicated AI tasks, with language serving as a generic interface to empower this. Based on this philosophy, we present HuggingGPT, an LLM-powered agent that leverages LLMs (e.g., ChatGPT) to connect various AI models in machine learning communities (e.g., Hugging Face) to solve AI tasks. Specifically, we use ChatGPT to conduct task planning when receiving a user request, select models according to their function descriptions available in Hugging Face, execute each subtask with the selected AI model, and summarize the response according to the execution results. By leveraging the strong language capability of ChatGPT and abundant AI models in Hugging Face, HuggingGPT can tackle a wide range of sophisticated AI tasks spanning different modalities and domains and achieve impressive results in language, vision, speech, and other challenging tasks, which paves a new way towards the realization of artificial general intelligence.</details>

### Human-Guided Complexity-Controlled Abstractions

**Authors:** Andi Peng, Mycal Tucker, Eoin Kenny, Noga Zaslavsky, Pulkit Agrawal, Julie A Shah

**<details><summary>Abstract**</summary>

Neural networks often learn task-specific latent representations that fail to generalize to novel settings or tasks. Conversely, humans learn discrete representations (i.e., concepts or words) at a variety of abstraction levels (e.g., "bird" vs. "sparrow'") and use the appropriate abstraction based on tasks. Inspired by this, we train neural models to generate a spectrum of discrete representations, and control the complexity of the representations (roughly, how many bits are allocated for encoding inputs) by tuning the entropy of the distribution over representations. In finetuning experiments, using only a small number of labeled examples for a new task, we show that (1) tuning the representation to a task-appropriate complexity level supports the greatest finetuning performance, and (2) in a human-participant study, users were able to identify the appropriate complexity level for a downstream task via visualizations of discrete representations. Our results indicate a promising direction for rapid model finetuning by leveraging human insight.</details>

### Humans in Kitchens: A Dataset for Multi-Person Human Motion Forecasting with Scene Context

**Authors:** Julian Tanke, Oh-Hun Kwon, Felix B Mueller, Andreas Doering, Jürgen Gall

**<details><summary>Abstract**</summary>

Forecasting human motion of multiple persons is very challenging. It requires to model the interactions between humans and the interactions with objects and the environment. For example, a person might want to make a coffee, but if the coffee machine is already occupied the person will haveto wait. These complex relations between scene geometry and persons ariseconstantly in our daily lives, and models that wish to accurately forecasthuman behavior will have to take them into consideration. To facilitate research in this direction, we propose Humans in Kitchens, alarge-scale multi-person human motion dataset with annotated 3D human poses, scene geometry and activities per person and frame.Our dataset consists of over 7.3h recorded data of up to 16 persons at the same time in four kitchen scenes, with more than 4M annotated human poses, represented by a parametric 3D body model. In addition, dynamic scene geometry and objects like chair or cupboard are annotated per frame. As first benchmarks, we propose two protocols for short-term and long-term human motion forecasting.</details>

### Hyper-HMM: aligning human brains and semantic features in a common latent event space

**Authors:** Caroline Lee, Jane Han, Ma Feilong, Guo Jiahui, James Haxby, Christopher Baldassano

**<details><summary>Abstract**</summary>

Naturalistic stimuli evoke complex neural responses with spatial and temporal properties that differ across individuals. Current alignment methods focus on either spatial hyperalignment (assuming exact temporal correspondence) or temporal alignment (assuming exact spatial correspondence). Here, we propose a hybrid model, the Hyper-HMM, that simultaneously aligns both temporal and spatial features across brains. The model learns to linearly project voxels to a reduced-dimension latent space, in which timecourses are segmented into corresponding temporal events. This approach allows tracking of each individual's mental trajectory through an event sequence, and also allows for alignment with other feature spaces such as stimulus content. Using an fMRI dataset in which students watch videos of class lectures, we demonstrate that the Hyper-HMM can be used to map all participants and the semantic content of the videos into a common low-dimensional space, and that these mappings generalize to held-out data. Our model provides a new window into individual cognitive dynamics evoked by complex naturalistic stimuli.</details>

### Hyper-Skin: A Hyperspectral Dataset for Reconstructing Facial Skin-Spectra from RGB Images

**Authors:** Pai Chet Ng, Zhixiang Chi, Yannick Verdie, Juwei Lu, Konstantinos N Plataniotis

**<details><summary>Abstract**</summary>

We introduce Hyper-Skin, a hyperspectral dataset covering wide range of wavelengths from visible (VIS) spectrum (400nm - 700nm) to near-infrared (NIR) spectrum (700nm - 1000nm), uniquely designed to facilitate research on facial skin-spectra reconstruction.By reconstructing skin spectra from RGB images, our dataset enables the study of hyperspectral skin analysis, such as melanin and hemoglobin concentrations, directly on the consumer device. Overcoming limitations of existing datasets, Hyper-Skin consists of diverse facial skin data collected with a pushbroom hyperspectral camera. With 330 hyperspectral cubes from 51 subjects, the dataset covers the facial skin from different angles and facial poses.Each hyperspectral cube has dimensions of 1024$	imes$1024$	imes$448, resulting in millions of spectra vectors per image. The dataset, carefully curated in adherence to ethical guidelines, includes paired hyperspectral images and synthetic RGB images generated using real camera responses. We demonstrate the efficacy of our dataset by showcasing skin spectra reconstruction using state-of-the-art models on 31 bands of hyperspectral data resampled in the VIS and NIR spectrum. This Hyper-Skin dataset would be a valuable resource to NeurIPS community, encouraging the development of novel algorithms for skin spectral reconstruction while fostering interdisciplinary collaboration in hyperspectral skin analysis related to cosmetology and skin's well-being. Instructions to request the data and the related benchmarking codes are publicly available at: https://github.com/hyperspectral-skin/Hyper-Skin-2023.</details>

### INSPECT: A Multimodal Dataset for Patient Outcome Prediction of Pulmonary Embolisms

**Authors:** Shih-Cheng Huang, Zepeng Huo, Ethan Steinberg, Chia-Chun Chiang, Curtis Langlotz, Matthew Lungren, Serena Yeung, Nigam Shah, Jason Fries

**<details><summary>Abstract**</summary>

Synthesizing information from various data sources plays a crucial role in the practice of modern medicine. Current applications of artificial intelligence in medicine often focus on single-modality data due to a lack of publicly available, multimodal medical datasets. To address this limitation, we introduce INSPECT, which contains de-identified longitudinal records from a large cohort of pulmonary embolism (PE) patients, along with ground truth labels for multiple outcomes. INSPECT contains data from 19,402 patients, including CT images, sections of radiology reports, and structured electronic health record (EHR) data (including demographics, diagnoses, procedures, and vitals). Using our provided dataset, we develop and release a benchmark for evaluating several baseline modeling approaches on a variety of important PE related tasks. We evaluate image-only, EHR-only, and fused models. Trained models and the de-identified dataset are made available for non-commercial use under a data use agreement. To the best our knowledge, INSPECT is the largest multimodal dataset for enabling reproducible research on strategies for integrating 3D medical imaging and EHR data.</details>

### Ignorance is Bliss: Robust Control via Information Gating

**Authors:** Manan Tomar, Riashat Islam, Matthew Taylor, Sergey Levine, Philip Bachman

**<details><summary>Abstract**</summary>

Informational parsimony provides a useful inductive bias for learning representations that achieve better generalization by being robust to noise and spurious correlations. We propose *information gating* as a way to learn parsimonious representations that identify the minimal information required for a task. When gating information, we can learn to reveal as little information as possible so that a task remains solvable, or hide as little information as possible so that a task becomes unsolvable. We gate information using a differentiable parameterization of the signal-to-noise ratio, which can be applied to arbitrary values in a network, e.g., erasing pixels at the input layer or activations in some intermediate layer. When gating at the input layer, our models learn which visual cues matter for a given task. When gating intermediate layers, our models learn which activations are needed for subsequent stages of computation. We call our approach *InfoGating*. We apply InfoGating to various objectives such as multi-step forward and inverse dynamics models, Q-learning, and behavior cloning, highlighting how InfoGating can naturally help in discarding information not relevant for control. Results show that learning to identify and use minimal information can improve generalization in downstream tasks. Policies based on InfoGating are considerably more robust to irrelevant visual features, leading to improved pretraining and finetuning of RL models.</details>

### [Oral] Image Captioners Are Scalable Vision Learners Too

**Authors:** Michael Tschannen, Manoj Kumar, Andreas Steiner, Xiaohua Zhai, Neil Houlsby, Lucas Beyer

**Oral Presentation:** Th, Dec 14, 13:35 -- Oral 6C

**<details><summary>Abstract**</summary>

Contrastive pretraining on image-text pairs from the web is one of the most popular large-scale pretraining strategies for vision backbones, especially in the context of large multimodal models. At the same time, image captioning on this type of data is commonly considered an inferior pretraining strategy. In this paper, we perform a fair comparison of these two pretraining strategies, carefully matching training data, compute, and model capacity. Using a standard encoder-decoder transformer, we find that captioning alone is surprisingly effective: on classification tasks, captioning produces vision encoders competitive with contrastively pretrained encoders, while surpassing them on vision & language tasks. We further analyze the effect of the model architecture and scale, as well as the pretraining data on the representation quality, and find that captioning exhibits the same or better scaling behavior along these axes. Overall our results show that plain image captioning is a more powerful pretraining strategy than was previously believed.</details>

### ImageNet-Hard: The Hardest Images Remaining from a Study of the Power of Zoom and Spatial Biases in Image Classification

**Authors:** Mohammad Reza Taesiri, Giang Nguyen, Sarra Habchi, Cor-Paul Bezemer, Anh Nguyen

**<details><summary>Abstract**</summary>

Image classifiers are information-discarding machines, by design. Yet, how these models discard information remains mysterious. We hypothesize that one way for image classifiers to reach high accuracy is to first zoom to the most discriminative region in the image and then extract features from there to predict image labels, discarding the rest of the image. Studying six popular networks ranging from AlexNet to CLIP, we find that proper framing of the input image can lead to the correct classification of 98.91% of ImageNet images. Furthermore, we  uncover positional biases in various datasets, especially a strong center bias in two popular datasets: ImageNet-A and ObjectNet. Finally, leveraging our insights into the potential of zooming, we propose a test-time augmentation (TTA) technique that improves classification accuracy by forcing models to explicitly perform zoom-in operations before making predictions.Our method is more interpretable, accurate, and faster than MEMO, a state-of-the-art (SOTA) TTA method. We introduce ImageNet-Hard, a new benchmark that challenges SOTA classifiers including large vision-language models even when optimal zooming is allowed.</details>

### Imagine the Unseen World: A Benchmark for Systematic Generalization in Visual World Models

**Authors:** Yeongbin Kim, Gautam Singh, Junyeong Park, Caglar Gulcehre, Sungjin Ahn

**<details><summary>Abstract**</summary>

Systematic compositionality, or the ability to adapt to novel situations by creating a mental model of the world using reusable pieces of knowledge, remains a significant challenge in machine learning. While there has been considerable progress in the language domain, efforts towards systematic visual imagination, or envisioning the dynamical implications of a visual observation, are in their infancy. We introduce the Systematic Visual Imagination Benchmark (SVIB), the first benchmark designed to address this problem head-on. SVIB offers a novel framework for a minimal world modeling problem, where models are evaluated based on their ability to generate one-step image-to-image transformations under a latent world dynamics. The framework provides benefits such as the possibility to jointly optimize for systematic perception and imagination, a range of difficulty levels, and the ability to control the fraction of possible factor combinations used during training. We provide a comprehensive evaluation of various baseline models on SVIB, offering insight into the current state-of-the-art in systematic visual imagination. We hope that this benchmark will help advance visual systematic compositionality.</details>

### Implicit variance regularization in non-contrastive SSL

**Authors:** Manu Srinath Halvagal, Axel Laborieux, Friedemann Zenke

**<details><summary>Abstract**</summary>

Non-contrastive SSL methods like BYOL and SimSiam rely on asymmetric predictor networks to avoid representational collapse without negative samples. Yet, how predictor networks facilitate stable learning is not fully understood. While previous theoretical analyses assumed Euclidean losses, most practical implementations rely on cosine similarity. To gain further theoretical insight into non-contrastive SSL, we analytically study learning dynamics in conjunction with Euclidean and cosine similarity in the eigenspace of closed-form linear predictor networks. We show that both avoid collapse through implicit variance regularization albeit through different dynamical mechanisms. Moreover, we find that the eigenvalues act as effective learning rate multipliers and propose a family of isotropic loss functions (IsoLoss) that equalize convergence rates across eigenmodes. Empirically, IsoLoss speeds up the initial learning dynamics and increases robustness, thereby allowing us to dispense with the EMA target network typically used with non-contrastive methods. Our analysis sheds light on the variance regularization mechanisms of non-contrastive SSL and lays the theoretical grounds for crafting novel loss functions that shape the learning dynamics of the predictor's spectrum.</details>

### [Oral] Improved Algorithms for Stochastic Linear Bandits Using Tail Bounds for Martingale Mixtures

**Authors:** Hamish Flynn, David Reeb, Melih Kandemir, Jan Peters

**Oral Presentation:** Th, Dec 14, 14:05 -- Oral 6D

**<details><summary>Abstract**</summary>

We present improved algorithms with worst-case regret guarantees for the stochastic linear bandit problem. The widely used "optimism in the face of uncertainty" principle reduces a stochastic bandit problem to the construction of a confidence sequence for the unknown reward function. The performance of the resulting bandit algorithm depends on the size of the confidence sequence, with smaller confidence sets yielding better empirical performance and stronger regret guarantees. In this work, we use a novel tail bound for adaptive martingale mixtures to construct confidence sequences which are suitable for stochastic bandits. These confidence sequences allow for efficient action selection via convex programming. We prove that a linear bandit algorithm based on our confidence sequences is guaranteed to achieve competitive worst-case regret. We show that our confidence sequences are tighter than competitors, both empirically and theoretically. Finally, we demonstrate that our tighter confidence sequences give improved performance in several hyperparameter tuning tasks.</details>

### Improving *day-ahead* Solar Irradiance Time Series Forecasting by Leveraging Spatio-Temporal Context

**Authors:** Oussama Boussif, Ghait Boukachab, Dan Assouline, Stefano Massaroli, Tianle Yuan, Loubna Benabbou, Yoshua Bengio

**<details><summary>Abstract**</summary>

Solar power harbors immense potential in mitigating climate change by substantially reducing CO$_{2}$ emissions. Nonetheless, the inherent variability of solar irradiance poses a significant challenge for seamlessly integrating solar power into the electrical grid. While the majority of prior research has centered on employing purely time series-based methodologies for solar forecasting, only a limited number of studies have taken into account factors such as cloud cover or the surrounding physical context.In this paper, we put forth a deep learning architecture designed to harness spatio-temporal context using satellite data, to attain highly accurate day-ahead time-series forecasting for any given station, with a particular emphasis on forecasting Global Horizontal Irradiance (GHI). We also suggest a methodology to extract a distribution for each time step prediction, which can serve as a very valuable measure of uncertainty attached to the forecast. When evaluating models, we propose a testing scheme in which we separate particularly difficult examples from easy ones, in order to capture the model performances in crucial situations, which in the case of this study are the days suffering from varying cloudy conditions. Furthermore, we present a new multi-modal dataset gathering satellite imagery over a large zone and time series for solar irradiance and other related physical variables from multiple geographically diverse solar stations. Our approach exhibits robust performance in solar irradiance forecasting, including zero-shot generalization tests at unobserved solar stations, and holds great promise in promoting the effective integration of solar power into the grid.</details>

### Improving Self-supervised Molecular Representation Learning using Persistent Homology

**Authors:** Yuankai Luo, Lei Shi, Veronika Thost

**<details><summary>Abstract**</summary>

Self-supervised learning (SSL) has great potential for molecular representation learning given the complexity of molecular graphs, the large amounts of unlabelled data available, the considerable cost of obtaining labels experimentally, and the hence often only small training datasets. The importance of the topic is reflected in the variety of paradigms and architectures that have been investigated recently. Yet the differences in performance seem often minor and are barely understood to date. In this paper, we study SSL based on persistent homology (PH), a mathematical tool for modeling topological features of data that persist across multiple scales. It has several unique features which particularly suit SSL, naturally offering: different views of the data, stability in terms of distance preservation, and the opportunity to flexibly incorporate domain knowledge.We (1) investigate an autoencoder, which shows the general representational power of PH, and (2) propose a contrastive loss that complements existing approaches. We rigorously evaluate our approach for molecular property prediction and demonstrate its particular features in improving the embedding space:after SSL, the representations are better and offer considerably more predictive power than the baselines over different probing tasks; our loss increases baseline performance, sometimes largely; and we often obtain substantial improvements over very small datasets, a common scenario in practice.</details>

### Inference for Gaussian Processes with Matern Covariogram on Compact Riemannian Manifolds

**Authors:** Didong Li, Wenpin Tang, Sudipto Banerjee

**<details><summary>Abstract**</summary>

Gaussian processes are widely employed as versatile modelling and predictive tools in spatial statistics, functional data analysis, computer modelling and diverse applications of machine learning. They have been widely studied over Euclidean spaces, where they are specified using covariance functions or covariograms for modelling complex dependencies. There is a growing literature on Gaussian processes over Riemannian manifolds in order to develop richer and more flexible inferential frameworks for non-Euclidean data. While numerical approximations through graph representations have been well studied for the Matern covariogram and heat kernel, the behaviour of asymptotic inference on the parameters of the covariogram has received relatively scant attention. We focus on asymptotic behaviour for Gaussian processes constructed over compact Riemannian manifolds. Building upon a recently introduced Matern covariogram on a compact Riemannian manifold, we employ formal notions and conditions for the equivalence of two Matern Gaussian random measures on compact manifolds to derive the parameter that is identifiable, also known as the microergodic parameter, and formally establish the consistency of the maximum likelihood estimate and the asymptotic optimality of the best linear unbiased predictor. The circle is studied as a specific example of compact Riemannian manifolds with numerical experiments to illustrate and corroborate the theory.</details>

### InfoCD: A Contrastive Chamfer Distance Loss for Point Cloud Completion

**Authors:** Fangzhou Lin, Yun Yue, Ziming Zhang, Songlin Hou, Kazunori Yamada, Vijaya Kolachalama, Venkatesh Saligrama

**<details><summary>Abstract**</summary>

A point cloud is a discrete set of data points sampled from a 3D geometric surface. Chamfer distance (CD) is a popular metric and training loss to measure the distances between point clouds, but also well known to be sensitive to outliers. To address this issue, in this paper we propose InfoCD, a novel contrastive Chamfer distance loss to learn to spread the matched points for better distribution alignments between point clouds as well as accounting for a surface similarity estimator. We show that minimizing InfoCD is equivalent to maximizing a lower bound of the mutual information between the underlying geometric surfaces represented by the point clouds, leading to a regularized CD metric which is robust and computationally efficient for deep learning. We conduct comprehensive experiments for point cloud completion using InfoCD and observe significant improvements consistently over all the popular baseline networks trained with CD-based losses, leading to new state-of-the-art results on several benchmark datasets. Demo code is available at https://github.com/Zhang-VISLab/NeurIPS2023-InfoCD.</details>

### InstanT: Semi-supervised Learning with Instance-dependent Thresholds

**Authors:** Muyang Li, Runze Wu, Haoyu Liu, Jun Yu, Xun Yang, Bo Han, Tongliang Liu

**<details><summary>Abstract**</summary>

Semi-supervised learning (SSL) has been a fundamental challenge in machine learning for decades. The primary family of SSL algorithms, known as pseudo-labeling, involves assigning pseudo-labels to confident unlabeled instances and incorporating them into the training set. Therefore, the selection criteria of confident instances are crucial to the success of SSL. Recently, there has been growing interest in the development of SSL methods that use dynamic or adaptive thresholds. Yet, these methods typically apply the same threshold to all samples, or use class-dependent thresholds for instances belonging to a certain class, while neglecting instance-level information. In this paper, we propose the study of instance-dependent thresholds, which has the highest degree of freedom compared with existing methods. Specifically, we devise a novel instance-dependent threshold function for all unlabeled instances by utilizing their instance-level ambiguity and the instance-dependent error rates of pseudo-labels, so instances that are more likely to have incorrect pseudo-labels will have higher thresholds. Furthermore, we demonstrate that our instance-dependent threshold function provides a bounded probabilistic guarantee for the correctness of the pseudo-labels it assigns.</details>

### InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning

**Authors:** Wenliang Dai, Junnan Li, DONGXU LI, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, Steven Hoi

**<details><summary>Abstract**</summary>

Large-scale pre-training and instruction tuning have been successful at creating general-purpose language models with broad competence. However, building general-purpose vision-language models is challenging due to the rich input distributions and task diversity resulting from the additional visual input. Although vision-language pretraining has been widely studied, vision-language instruction tuning remains under-explored. In this paper, we conduct a systematic and comprehensive study on vision-language instruction tuning based on the pretrained BLIP-2 models. We gather 26 publicly available datasets, covering a wide variety of tasks and capabilities, and transform them into instruction tuning format. Additionally, we introduce an instruction-aware Query Transformer, which extracts informative features tailored to the given instruction. Trained on 13 held-in datasets, InstructBLIP attains state-of-the-art zero-shot performance across all 13 held-out datasets, substantially outperforming BLIP-2 and larger Flamingo models. Our models also lead to state-of-the-art performance when finetuned on individual downstream tasks (e.g., 90.7% accuracy on ScienceQA questions with image contexts). Furthermore, we qualitatively demonstrate the advantages of InstructBLIP over concurrent multimodal models. All InstructBLIP models are open-source.</details>

### Intelligent Knee Sleeves: A Real-time Multimodal Dataset for 3D Lower Body Motion Estimation Using Smart Textile

**Authors:** Wenwen Zhang, Arvin Tashakori, Zenan Jiang, Amir Servati, Harishkumar Narayana, Saeid Soltanian, Rou Yi Yeap, Menghan Ma, Lauren Toy, Peyman Servati

**<details><summary>Abstract**</summary>

The kinematics of human movements and locomotion are closely linked to the activation and contractions of muscles. To investigate this, we present a multimodal dataset with benchmarks collected using a novel pair of Intelligent Knee Sleeves (Texavie MarsWear Knee Sleeves) for human pose estimation. Our system utilizes synchronized datasets that comprise time-series data from the Knee Sleeves and the corresponding ground truth labels from visualized motion capture camera system. We employ these to generate 3D human models solely based on the wearable data of individuals performing different activities. We demonstrate the effectiveness of this camera-free system and machine learning algorithms in the assessment of various movements and exercises, including extension to unseen exercises and individuals. The results show an average error of 7.21 degrees across all eight lower body joints when compared to the ground truth, indicating the effectiveness and reliability of the Knee Sleeve system for the prediction of different lower body joints beyond knees. The results enable human pose estimation in a seamless manner without being limited by visual occlusion or the field of view of cameras. Our results show the potential of multimodal wearable sensing in a variety of applications from home fitness to sports, healthcare, and physical rehabilitation focusing on pose and movement estimation.</details>

### InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback

**Authors:** John Yang, Akshara Prabhakar, Karthik Narasimhan, Shunyu Yao

**<details><summary>Abstract**</summary>

Humans write code in a fundamentally interactive manner and rely on constant execution feedback to correct errors, resolve ambiguities, and decompose tasks. While LLMs have recently exhibited promising coding capabilities, current coding benchmarks mostly consider a static instruction-to-code sequence transduction process, which has the potential for error propagation and a disconnect between the generated code and its final execution environment. To address this gap, we introduce InterCode, a lightweight, flexible, and easy-to-use framework of interactive coding as a standard reinforcement learning (RL) environment, with code as actions and execution feedback as observations. Our framework is language and platform agnostic, uses self-contained Docker environments to provide safe and reproducible execution, and is compatible out-of-the-box with traditional seq2seq coding methods, while enabling the development of new methods for interactive code generation. We use InterCode to create three interactive code environments with Bash, SQL, and Python as action spaces, leveraging data from the static NL2Bash, Spider, and MBPP datasets. We demonstrate InterCode’s viability as a testbed by evaluating multiple state-of-the-art LLMs configured with different prompting strategies such as ReAct and Plan & Solve. Our results showcase the benefits of interactive code generation and demonstrate that InterCode can serve as a challenging benchmark for advancing code understanding and generation capabilities. InterCode is designed to be easily extensible and can even be used to create new tasks such as Capture the Flag, a popular coding puzzle that is inherently multi-step and involves multiple programming languages.</details>

### Interactive Visual Reasoning under Uncertainty

**Authors:** Manjie Xu, Guangyuan Jiang, Wei Liang, Chi Zhang, Yixin Zhu

**<details><summary>Abstract**</summary>

One of the fundamental cognitive abilities of humans is to quickly resolve uncertainty by generating hypotheses and testing them via active trials. Encountering a novel phenomenon accompanied by ambiguous cause-effect relationships, humans make hypotheses against data, conduct inferences from observation, test their theory via experimentation, and correct the proposition if inconsistency arises. These iterative processes persist until the underlying mechanism becomes clear. In this work, we devise the  **IVRE** (pronounced as *"ivory"*) environment for evaluating artificial agents' reasoning ability under uncertainty. **IVRE** is an interactive environment featuring rich scenarios centered around *Blicket* detection. Agents in **IVRE** are placed into environments with various ambiguous action-effect pairs and asked to determine each object's role. They are encouraged to propose effective and efficient experiments to validate their hypotheses based on observations and actively gather new information. The game ends when all uncertainties are resolved or the maximum number of trials is consumed. By evaluating modern artificial agents in **IVRE**, we notice a clear failure of today's learning methods compared to humans. Such inefficacy in interactive reasoning ability under uncertainty calls for future research in building human-like intelligence.</details>

### Interpreting Unsupervised Anomaly Detection in Security via Rule Extraction

**Authors:** Ruoyu Li, Qing Li, Yu Zhang, Dan Zhao, Yong Jiang, Yong Yang

**<details><summary>Abstract**</summary>

Many security applications require unsupervised anomaly detection, as malicious data are extremely rare and often only unlabeled normal data are available for training (i.e., zero-positive). However, security operators are concerned about the high stakes of trusting black-box models due to their lack of interpretability. In this paper, we propose a post-hoc method to globally explain a black-box unsupervised anomaly detection model via rule extraction.First, we propose the concept of distribution decomposition rules that decompose the complex distribution of normal data into multiple compositional distributions. To find such rules, we design an unsupervised Interior Clustering Tree that incorporates the model prediction into the splitting criteria. Then, we propose the Compositional Boundary Exploration (CBE) algorithm to obtain the boundary inference rules that estimate the decision boundary of the original model on each compositional distribution. By merging these two types of rules into a rule set, we can present the inferential process of the unsupervised black-box model in a human-understandable way, and build a surrogate rule-based model for online deployment at the same time. We conduct comprehensive experiments on the explanation of four distinct unsupervised anomaly detection models on various real-world datasets. The evaluation shows that our method outperforms existing methods in terms of diverse metrics including fidelity, correctness and robustness.</details>

### Into the LAION’s Den: Investigating Hate in Multimodal Datasets

**Authors:** Abeba Birhane, vinay prabhu, Sanghyun Han, Vishnu Boddeti, Sasha Luccioni

**<details><summary>Abstract**</summary>

Scale the model, scale the data, scale the compute' is the reigning sentiment in the world of generative AI today. While the impact of model scaling has been extensively studied, we are only beginning to scratch the surface of data scaling and its consequences. This is especially of critical importance in the context of vision-language datasets such as LAION. These datasets are continually growing in size and are built based on large-scale internet dumps such as the Common Crawl, which is known to have numerous drawbacks ranging from quality, legality, and content. The datasets then serve as the backbone for large generative models, contributing to the operationalization and perpetuation of harmful societal and historical biases and stereotypes. In this paper, we investigate the effect of scaling datasets on hateful content through a comparative audit of two datasets: LAION-400M and LAION-2B. Our results show that hate content increased by nearly **12%** with dataset scale, measured both qualitatively and quantitatively using a metric that we term as Hate Content Rate (HCR). We also found that filtering dataset contents based on Not Safe For Work (NSFW) values calculated based on images alone does not exclude all the harmful content in alt-text. Instead, we found that trace amounts of hateful, targeted, and aggressive text remain even when carrying out conservative filtering. We end with a reflection and a discussion of the significance of our results for dataset curation and usage in the AI community.Code and the meta-data assets curated in this paper are publicly available at https://github.com/vinayprabhu/hate_scaling. Content warning: This paper contains examples of hateful text that might be disturbing, distressing, and/or offensive.</details>

### Into the Single Cell Multiverse: an End-to-End Dataset for Procedural Knowledge Extraction in Biomedical Texts

**Authors:** Ruth Dannenfelser, Jeffrey Zhong, Ran Zhang, Vicky Yao

**<details><summary>Abstract**</summary>

Many of the most commonly explored natural language processing (NLP) information extraction tasks can be thought of as evaluations of declarative knowledge, or fact-based information extraction. Procedural knowledge extraction, i.e., breaking down a described process into a series of steps, has received much less attention, perhaps in part due to the lack of structured datasets that capture the knowledge extraction process from end-to-end. To address this unmet need, we present FlaMBé (Flow annotations for Multiverse Biological entities), a collection of expert-curated datasets across a series of complementary tasks that capture procedural knowledge in biomedical texts. This dataset is inspired by the observation that one ubiquitous source of procedural knowledge that is described as unstructured text is within academic papers describing their methodology. The workflows annotated in FlaMBé are from texts in the burgeoning field of single cell research, a research area that has become notorious for the number of software tools and complexity of workflows used. Additionally, FlaMBé provides, to our knowledge, the largest manually curated named entity recognition (NER) and disambiguation (NED) datasets for tissue/cell type, a fundamental biological entity that is critical for knowledge extraction in the biomedical research domain. Beyond providing a valuable dataset to enable further development of NLP models for procedural knowledge extraction, automating the process of workflow mining also has important implications for advancing reproducibility in biomedical research.</details>

### Intrinsic Gaussian Process on Unknown Manifolds with Probabilistic Metrics

**Authors:** mu niu, Zhenwen Dai, Pokman Cheung, Yizhu Wang

**<details><summary>Abstract**</summary>

This article presents a novel approach to construct Intrinsic Gaussian Processes for regression on unknown manifolds with probabilistic metrics (GPUM) in point clouds. In many real world applications, one often encounters high dimensional data (e.g.‘point cloud data’) centered around some lower dimensional unknown manifolds. The geometry of manifold is in general different from the usual Euclidean geometry. Naively applying traditional smoothing methods such as Euclidean Gaussian Processes (GPs) to manifold-valued data and so ignoring the geometry of the space can potentially lead to highly misleading predictions and inferences. A manifold embedded in a high dimensional Euclidean space can be well described by a probabilistic mapping function and the corresponding latent space. We investigate the geometrical structure of the unknown manifolds using the Bayesian Gaussian Processes latent variable models(B-GPLVM) and Riemannian geometry. The distribution of the metric tensor is learned using B-GPLVM. The boundary of the resulting manifold is defined based on the uncertainty quantification of the mapping. We use the probabilistic metric tensor to simulate Brownian Motion paths on the unknown manifold. The heat kernel is estimated as the transition density of Brownian Motion and used as the covariance functions of GPUM. The applications of GPUM are illustrated in the simulation studies on the Swiss roll, high dimensional real datasets of WiFi signals and image data examples. Its performance is compared with the Graph Laplacian GP, Graph Mat\'{e}rn GP and Euclidean GP.</details>

### Is RLHF More Difficult than Standard RL? A Theoretical Perspective

**Authors:** Yuanhao Wang, Qinghua Liu, Chi Jin

**<details><summary>Abstract**</summary>

Reinforcement learning from Human Feedback (RLHF) learns from preference signals, while standard Reinforcement Learning (RL) directly learns from reward signals. Preferences arguably contain less information than rewards, which makes preference-based RL seemingly more difficult. This paper theoretically proves that, for a wide range of preference models, we can solve preference-based RL directly using existing algorithms and techniques for reward-based RL, with small or no extra costs. Specifically, (1) for preferences that are drawn from reward-based probabilistic models, we reduce the problem to robust reward-based RL that can tolerate small errors in rewards; (2) for general arbitrary preferences where the objective is to find the von Neumann winner, we reduce the problem to multiagent reward-based RL which finds Nash equilibria for factored Markov games under a restricted set of policies. The latter case can be further reduce to adversarial MDP when preferences only depend on the final state. We instantiate all reward-based RL subroutines by concrete provable algorithms, and apply our theory to a large class of models including tabular MDPs and MDPs with generic function approximation. We further provide guarantees when K-wise comparisons are available.</details>

### [Oral] Jailbroken: How Does LLM Safety Training Fail?

**Authors:** Alexander Wei, Nika Haghtalab, Jacob Steinhardt

**Oral Presentation:** Th, Dec 14, 14:05 -- Oral 6A

**<details><summary>Abstract**</summary>

Large language models trained for safety and harmlessness remain susceptible to adversarial misuse, as evidenced by the prevalence of “jailbreak” attacks on early releases of ChatGPT that elicit undesired behavior. Going beyond recognition of the issue, we investigate why such attacks succeed and how they can be created. We hypothesize two failure modes of safety training: competing objectives and mismatched generalization. Competing objectives arise when a model’s capabilities and safety goals conflict, while mismatched generalization occurs when safety training fails to generalize to a domain for which capabilities exist. We use these failure modes to guide jailbreak design and then evaluate state-of-the-art models, including OpenAI’s GPT-4 and Anthropic’s Claude v1.3, against both existing and newly designed attacks. We find that vulnerabilities persist despite the extensive red-teaming and safety-training efforts behind these models. Notably, new attacks utilizing our failure modes succeed on every prompt in a collection of unsafe requests from the models’ red-teaming evaluation sets and outperform existing ad hoc jailbreaks. Our analysis emphasizes the need for safety-capability parity—that safety mechanisms should be as sophisticated as the underlying model—and argues against the idea that scaling alone can resolve these safety failure modes.</details>

### Knowledge Distillation Performs Partial Variance Reduction

**Authors:** Mher Safaryan, Alexandra Peste, Dan Alistarh

**<details><summary>Abstract**</summary>

Knowledge distillation is a popular approach for enhancing the performance of "student" models, with lower representational capacity, by taking advantage of more powerful "teacher" models. Despite its apparent simplicity, the underlying mechanics behind knowledge distillation (KD) are not yet fully understood. In this work, we shed new light on the inner workings of this method, by examining it from an optimization perspective. Specifically, we show that, in the context of linear and deep linear models, KD can be interpreted as a novel type of stochastic variance reduction mechanism. We provide a detailed convergence analysis of the resulting dynamics, which hold under standard assumptions for both strongly-convex and non-convex losses, showing that KD acts as a form of \emph{partial variance reduction}, which can reduce the stochastic gradient noise, but may not eliminate it completely, depending on the properties of the ``teacher'' model. Our analysis puts further emphasis on the need for careful parametrization of KD, in particular w.r.t. the weighting of the distillation loss, and is validated empirically on both linear models and deep neural networks.</details>

### Knowledge-Augmented Reasoning Distillation for Small Language Models in Knowledge-Intensive Tasks

**Authors:** Minki Kang, Seanie Lee, Jinheon Baek, Kenji Kawaguchi, Sung Ju Hwang

**<details><summary>Abstract**</summary>

Large Language Models (LLMs) have shown promising performance in knowledge-intensive reasoning tasks that require a compound understanding of knowledge. However, deployment of the LLMs in real-world applications can be challenging due to their high computational requirements and concerns on data privacy.Previous studies have focused on building task-specific small Language Models (LMs) by fine-tuning them with labeled data or distilling LLMs. However, these approaches are ill-suited for knowledge-intensive reasoning tasks due to the limited capacity of small LMs in memorizing the knowledge required.Motivated by our theoretical analysis on memorization, we propose Knowledge-Augmented Reasoning Distillation (KARD), a novel method that fine-tunes small LMs to generate rationales obtained from LLMs with augmented knowledge retrieved from an external knowledge base. Moreover, we further propose a neural reranker to obtain documents relevant to rationale generation. We empirically show that KARD significantly improves the performance of small T5 and GPT models on the challenging knowledge-intensive reasoning datasets, namely MedQA-USMLE, StrategyQA, and OpenbookQA.Notably, our method makes the 250M T5 models achieve superior performance against the fine-tuned 3B models, having 12 times larger parameters, on both MedQA-USMLE and StrategyQA benchmarks.</details>

### Knowledge-based in silico models and dataset for the comparative evaluation of mammography AI for a range of breast characteristics, lesion conspicuities and doses

**Authors:** Elena Sizikova, Niloufar Saharkhiz, Diksha Sharma, Miguel Lago, Berkman Sahiner, Jana Delfino, Aldo Badano

**<details><summary>Abstract**</summary>

To generate evidence regarding the safety and efficacy of artificial intelligence (AI) enabled medical devices, AI models need to be evaluated on a diverse population of patient cases, some of which may not be readily available. We propose an evaluation approach for testing medical imaging AI models that relies on in silico imaging pipelines in which stochastic digital models of human anatomy (in object space) with and without pathology are imaged using a digital replica imaging acquisition system to generate realistic synthetic image datasets. Here, we release M-SYNTH, a dataset of cohorts with four breast fibroglandular density distributions imaged at different exposure levels using Monte Carlo x-ray simulations with the publicly available Virtual Imaging Clinical Trial for Regulatory Evaluation (VICTRE) toolkit. We utilize the synthetic dataset to analyze AI model performance and find that model performance decreases with increasing breast density and increases with higher mass density, as expected. As exposure levels decrease, AI model performance drops with the highest performance achieved at exposure levels lower than the nominal recommended dose for the breast type.</details>

### Kullback-Leibler Maillard Sampling for Multi-armed Bandits with Bounded Rewards

**Authors:** Hao Qin, Kwang-Sung Jun, Chicheng Zhang

**<details><summary>Abstract**</summary>

We study $K$-armed bandit problems where the reward distributions of the arms are all supported on the $[0,1]$ interval. Maillard sampling\cite{maillard13apprentissage}, an attractive alternative to Thompson sampling, has recently been shown to achieve competitive regret guarantees in the sub-Gaussian reward setting\cite{bian2022maillard} while maintaining closed-form action probabilities, which is useful for offline policy evaluation. In this work, we analyze the Kullback-Leibler Maillard Sampling (KL-MS) algorithm, a natural extension of Maillard sampling {and a special case of Minimum Empirical Divergence (MED)~\cite{honda2011asymptotically}} for achieving a KL-style finite-time gap-dependent regret bound. We show that KL-MS enjoys the asymptotic optimality when the rewards are Bernoulli and has an {adaptive} worst-case regret bound of the form $O(\sqrt{\mu^*(1-\mu^*) K T \ln K} + K \ln T)$, where $\mu^*$ is the expected reward of the optimal arm, and $T$ is the time horizon length; {this is the first time such adaptivity is reported in the literature for an algorithm with asymptotic optimality guarantees.}</details>

### LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark

**Authors:** Zhenfei Yin, Jiong Wang, Jianjian Cao, Zhelun Shi, Dingning Liu, Mukai Li, Xiaoshui Huang, Zhiyong Wang, Lu Sheng, LEI BAI, Jing Shao, Wanli Ouyang

**<details><summary>Abstract**</summary>

Large language models have emerged as a promising approach towards achieving general-purpose AI agents. The thriving open-source LLM community has greatly accelerated the development of agents that support human-machine dialogue interaction through natural language processing. However, human interaction with the world extends beyond only text as a modality, and other modalities such as vision are also crucial. Recent works on multi-modal large language models, such as GPT-4V and Bard, have demonstrated their effectiveness in handling visual modalities. However, the transparency of these works is limited and insufficient to support academic research. To the best of our knowledge, we present one of the very first open-source endeavors in the field, LAMM, encompassing a Language-Assisted Multi-Modal instruction tuning dataset, framework, and benchmark. Our aim is to establish LAMM as a growing ecosystem for training and evaluating MLLMs, with a specific focus on facilitating AI agents capable of bridging the gap between ideas and execution, thereby enabling seamless human-AI interaction. Our main contribution is three-fold: 1) We present a comprehensive dataset and benchmark, which cover a wide range of vision tasks for 2D and 3D vision. Extensive experiments validate the effectiveness of our dataset and benchmark. 2) We outline the detailed methodology of constructing multi-modal instruction tuning datasets and benchmarks for MLLMs, enabling rapid scaling and extension of MLLM research to diverse domains, tasks, and modalities. 3) We provide a primary but potential MLLM training framework optimized for modality extension. We also provide baseline models, comprehensive experimental observations, and analysis to accelerate future research. Our baseline model is trained within 24 A100 GPU hours, framework supports training with V100 and RTX3090 is available thanks to the open-source society. Codes and data are now available at https://openlamm.github.io.</details>

### LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day

**Authors:** Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, Jianfeng Gao

**<details><summary>Abstract**</summary>

Conversational generative AI has demonstrated remarkable promise for empowering biomedical practitioners, but current investigations focus on unimodal text. Multimodal conversational AI has seen rapid progress by leveraging billions of image-text pairs from the public web, but such general-domain vision-language models still lack sophistication in understanding and conversing about biomedical images. In this paper, we propose a cost-efficient approach for training a vision-language conversational assistant that can answer open-ended research questions of biomedical images. The key idea is to leverage a large-scale, broad-coverage biomedical figure-caption dataset extracted from PubMed Central, use GPT-4 to self-instruct open-ended instruction-following data from the captions, and then fine-tune a large general-domain vision-language model using a novel curriculum learning method. Specifically, the model first learns to align biomedical vocabulary using the figure-caption pairs as is, then learns to master open-ended conversational semantics using GPT-4 generated instruction-following data, broadly mimicking how a layperson gradually acquires biomedical knowledge. This enables us to train a Large Language and Vision Assistant for BioMedicine (LLaVA-Med) in less than 15 hours (with eight A100s). LLaVA-Med exhibits excellent multimodal conversational capability and can follow open-ended instruction to assist with inquiries about a biomedical image. On three standard biomedical visual question answering datasets, LLaVA-Med outperforms previous supervised state-of-the-art on certain metrics. To facilitate biomedical multimodal research, we will release our instruction-following data and the LLaVA-Med model.</details>

### LOVM: Language-Only Vision Model Selection

**Authors:** Orr Zohar, Shih-Cheng Huang, Kuan-Chieh Wang, Serena Yeung

**<details><summary>Abstract**</summary>

Pre-trained multi-modal vision-language models (VLMs) are becoming increasingly popular due to their exceptional performance on downstream vision applications, particularly in the few- and zero-shot settings. However, selecting the best-performing VLM for some downstream applications is non-trivial, as it is dataset and task-dependent. Meanwhile, the exhaustive evaluation of all available VLMs on a novel application is not only time and  computationally demanding but also necessitates the collection of a labeled dataset for evaluation. As the number of open-source VLM variants increases, there is a need for an efficient model selection strategy that does not require access to a curated evaluation dataset. This paper proposes a novel task and benchmark for efficiently evaluating VLMs' zero-shot performance on downstream applications without access to the downstream task dataset. Specifically, we introduce a new task LOVM: **L**anguage-**O**nly  **V**ision  **M**odel Selection , where methods are expected to perform both model selection and performance prediction based solely on a text description of the desired downstream application. We then introduced an extensive LOVM benchmark consisting of ground-truth evaluations of 35 pre-trained VLMs and 23 datasets, where methods are expected to rank the pre-trained VLMs and predict their zero-shot performance.</details>

### LVM-Med: Learning Large-Scale Self-Supervised Vision Models for Medical Imaging via Second-order Graph Matching

**Authors:** Duy M. H. Nguyen, Hoang Nguyen, Nghiem Diep, Tan Ngoc Pham, Tri Cao, Binh Nguyen, Paul Swoboda, Nhat Ho, Shadi Albarqouni, Pengtao Xie, Daniel Sonntag, Mathias Niepert

**<details><summary>Abstract**</summary>

Obtaining large pre-trained models that can be fine-tuned to new tasks with limited annotated samples has remained an open challenge for medical imaging data. While pre-trained networks on ImageNet and vision-language foundation models trained on web-scale data are the prevailing approaches, their effectiveness on medical tasks is limited due to the significant domain shift between natural and medical images. To bridge this gap, we introduce LVM-Med, the first family of deep networks trained on large-scale medical datasets. We have collected approximately 1.3 million medical images from 55 publicly available datasets, covering a large number of organs and modalities such as CT, MRI, X-ray, and Ultrasound. We benchmark several state-of-the-art self-supervised algorithms on this dataset and propose a novel self-supervised contrastive learning algorithm using a graph-matching formulation. The proposed approach makes three contributions: (i) it integrates prior pair-wise image similarity metrics based on local and global information; (ii) it captures the structural constraints of feature embeddings through a loss function constructed through a combinatorial graph-matching objective, and (iii) it can be trained efficiently end-to-end using modern gradient-estimation techniques for black-box solvers. We thoroughly evaluate the proposed LVM-Med on 15 downstream medical tasks ranging from segmentation and classification to object detection, and both for the in and out-of-distribution settings. LVM-Med empirically outperforms a number of state-of-the-art supervised, self-supervised, and foundation models. For challenging tasks such as Brain Tumor Classification or Diabetic Retinopathy Grading, LVM-Med improves previous vision-language models trained on 1 billion masks by 6-7%  while using only a ResNet-50.</details>

### LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite

**Authors:** Artur Toshev, Gianluca Galletti, Fabian Fritz, Stefan Adami, Nikolaus Adams

**<details><summary>Abstract**</summary>

Machine learning has been successfully applied to grid-based PDE modeling in various scientific applications. However, learned PDE solvers based on Lagrangian particle discretizations, which are the preferred approach to problems with free surfaces or complex physics, remain largely unexplored. We present LagrangeBench, the first benchmarking suite for Lagrangian particle problems, focusing on temporal coarse-graining. In particular, our contribution is: (a) seven new fluid mechanics datasets (four in 2D and three in 3D) generated with the Smoothed Particle Hydrodynamics (SPH) method including the Taylor-Green vortex, lid-driven cavity, reverse Poiseuille flow, and dam break, each of which includes different physics like solid wall interactions or free surface, (b) efficient JAX-based API with various recent training strategies and three neighbor search routines, and (c) JAX implementation of established Graph Neural Networks (GNNs) like GNS and SEGNN with baseline results. Finally, to measure the performance of learned surrogates we go beyond established position errors and introduce physical metrics like kinetic energy MSE and Sinkhorn distance for the particle distribution. Our codebase is available under the URL: [https://github.com/tumaer/lagrangebench](https://github.com/tumaer/lagrangebench).</details>

### Language-driven Scene Synthesis using Multi-conditional Diffusion Model

**Authors:** An Dinh Vuong, Minh Nhat VU, Toan Nguyen, Baoru Huang, Dzung Nguyen, Thieu Vo, Anh Nguyen

**<details><summary>Abstract**</summary>

Scene synthesis is a challenging problem with several industrial applications. Recently, substantial efforts have been directed to synthesize the scene using human motions, room layouts, or spatial graphs as the input. However, few studies have addressed this problem from multiple modalities, especially combining text prompts. In this paper, we propose a language-driven scene synthesis task, which is a new task that integrates text prompts, human motion, and existing objects for scene synthesis. Unlike other single-condition synthesis tasks, our problem involves multiple conditions and requires a strategy for processing and encoding them into a unified space. To address the challenge, we present a multi-conditional diffusion model, which differs from the implicit unification approach of other diffusion literature by explicitly predicting the guiding points for the original data distribution. We demonstrate that our approach is theoretically supportive. The intensive experiment results illustrate that our method outperforms state-of-the-art benchmarks and enables natural scene editing applications. The source code and dataset can be accessed at https://lang-scene-synth.github.io/.</details>

### Large Language Model as Attributed Training Data Generator: A Tale of Diversity and Bias

**Authors:** Yue Yu, Yuchen Zhuang, Jieyu Zhang, Yu Meng, Alexander Ratner, Ranjay Krishna, Jiaming Shen, Chao Zhang

**<details><summary>Abstract**</summary>

Large language models (LLMs) have been recently leveraged as training data generators for various natural language processing (NLP) tasks. While previous research has explored different approaches to training models using generated data, they generally rely on simple class-conditional prompts, which may limit the diversity of the generated data and inherit systematic biases of LLM. Thus, we investigate training data generation with diversely attributed prompts (e.g., specifying attributes like length and style), which have the potential to yield diverse and attributed generated data. Our investigation focuses on datasets with high cardinality and diverse domains, wherein we demonstrate that attributed prompts outperform simple class-conditional prompts in terms of the resulting model's performance. Additionally, we present a comprehensive empirical study on data generation encompassing vital aspects like bias, diversity, and efficiency, and highlight three key observations: firstly, synthetic datasets generated by simple prompts exhibit significant biases, such as regional bias; secondly, attribute diversity plays a pivotal role in enhancing model performance; lastly, attributed prompts achieve the performance of simple class-conditional prompts while utilizing only 5\% of the querying cost of ChatGPT associated with the latter. The data and code are available on {\url{https://github.com/yueyu1030/AttrPrompt}}.</details>

### Large Language Models are Fixated by Red Herrings: Exploring Creative Problem Solving and Einstellung Effect using the Only Connect Wall Dataset

**Authors:** Saeid Alavi Naeini, Raeid Saqur, Mozhgan Saeidi, John Giorgi, Babak Taati

**<details><summary>Abstract**</summary>

The quest for human imitative AI has been an enduring topic in AI research since inception. The technical evolution and emerging capabilities of the latest cohort of large language models (LLMs) have reinvigorated the subject beyond academia to cultural zeitgeist. While recent NLP evaluation benchmark tasks test some aspects of human-imitative behaviour (e.g., BIG-bench's `human-like behavior' tasks), few, if not none, examine *creative problem solving* abilities. Creative problem solving in humans is a well-studied topic in cognitive neuroscience with standardized tests that predominantly use ability to associate (heterogeneous) connections among clue words as a metric for creativity. Exposure to misleading stimuli --- distractors dubbed *red herrings* --- impede human performance in such tasks via the *fixation effect* and Einstellung paradigm. In cognitive neuroscience studies, such fixations are experimentally induced by pre-exposing participants to orthographically similar incorrect words to subsequent word-fragments or clues. The popular British quiz show Only Connect's *Connecting Wall* segment essentially mimics Mednick's Remote Associates Test (RAT) formulation with built-in, deliberate red herrings, that makes it an ideal proxy dataset to explore and study fixation effect and Einstellung paradigm from cognitive neuroscience in LLMs. In addition to presenting the novel Only Connect Wall (OCW) dataset, we also report results from our evaluation of selected pre-trained language models and LLMs (including OpenAI's GPT series) on creative problem solving tasks like grouping clue words by heterogeneous connections, and identifying correct open knowledge domain connections in respective groups. We synthetically generate two additional datasets: OCW-Randomized, OCW-WordNet to further analyze our red-herrings hypothesis in language models.The code and link to the dataset is available at [url](https://github.com/TaatiTeam/OCW).</details>

### Latent Graph Inference with Limited Supervision

**Authors:** Jianglin Lu, Yi Xu, Huan Wang, Yue Bai, Yun Fu

**<details><summary>Abstract**</summary>

Latent graph inference (LGI) aims to jointly learn the underlying graph structure and node representations from data features. However, existing LGI methods commonly suffer from the issue of supervision starvation, where massive edge weights are learned without semantic supervision and do not contribute to the training loss. Consequently, these supervision-starved weights, which determine the predictions of testing samples, cannot be semantically optimal, resulting in poor generalization. In this paper, we observe that this issue is actually caused by the graph sparsification operation, which severely destroys the important connections established between pivotal nodes and labeled ones. To address this, we propose to restore the corrupted affinities and replenish the missed supervision for better LGI. The key challenge then lies in identifying the critical nodes and recovering the corrupted affinities. We begin by defining the pivotal nodes as k-hop starved nodes, which can be identified based on a given adjacency matrix. Considering the high computational burden, we further present a more efficient alternative inspired by CUR matrix decomposition. Subsequently, we eliminate the starved nodes by reconstructing the destroyed connections. Extensive experiments on representative benchmarks demonstrate that reducing the starved nodes consistently improves the performance of state-of-the-art LGI methods, especially under extremely limited supervision (6.12% improvement on Pubmed with a labeling rate of only 0.3%).</details>

### Learning Causal Models under Independent Changes

**Authors:** Sarah Mameche, David Kaltenpoth, Jilles Vreeken

**<details><summary>Abstract**</summary>

In many scientific applications, we observe a system in different conditions in which its components may change, rather than in isolation. In our work, we are interested in explaining the generating process of such a multi-context system using a finite mixture of causal mechanisms. Recent work shows that this causal model is identifiable from data, but is limited to settings where the sparse mechanism shift hypothesis holds and only a subset of the causal conditionals change. As this assumption is not easily verifiable in practice, we study the more general principle that mechanism shifts are independent, which we formalize using the algorithmic notion of independence. We introduce an approach for causal discovery beyond partially directed graphs using Gaussian Process models, and give conditions under which we provably identify the correct causal model. In our experiments, we show that our method performs well in a range of synthetic settings, on realistic gene expression simulations, as well as on real-world cell signaling data.</details>

### Learning Curves for Noisy Heterogeneous Feature-Subsampled Ridge Ensembles

**Authors:** Ben Ruben, Cengiz Pehlevan

**<details><summary>Abstract**</summary>

Feature bagging is a well-established ensembling method which aims to reduceprediction variance by combining predictions of many estimators trained on subsetsor projections of features. Here, we develop a theory of feature-bagging in noisyleast-squares ridge ensembles and simplify the resulting learning curves in the specialcase of equicorrelated data. Using analytical learning curves, we demonstratethat subsampling shifts the double-descent peak of a linear predictor. This leadsus to introduce heterogeneous feature ensembling, with estimators built on varyingnumbers of feature dimensions, as a computationally efficient method to mitigatedouble-descent. Then, we compare the performance of a feature-subsamplingensemble to a single linear predictor, describing a trade-off between noise amplificationdue to subsampling and noise reduction due to ensembling. Our qualitativeinsights carry over to linear classifiers applied to image classification tasks withrealistic datasets constructed using a state-of-the-art deep learning feature map.</details>

### Learning Human Action Recognition Representations Without Real Humans

**Authors:** Howard Zhong, Samarth Mishra, Donghyun Kim, SouYoung Jin, Rameswar Panda, Hilde Kuehne, Leonid Karlinsky, Venkatesh Saligrama, Aude Oliva, Rogerio Feris

**<details><summary>Abstract**</summary>

Pre-training on massive video datasets has become essential to achieve high action recognition performance on smaller downstream datasets. However, most large-scale video datasets contain images of people and hence are accompanied with issues related to privacy, ethics, and data protection, often preventing them from being publicly shared for reproducible research. Existing work has attempted to alleviate these problems by blurring faces, downsampling videos, or training on synthetic data. On the other hand, analysis on the {\em transferability} of privacy-preserving pre-trained models to downstream tasks has been limited. In this work, we study this problem by first asking the question: can we pre-train models for human action recognition with data that does not include real humans? To this end, we present, for the first time, a benchmark that leverages real-world videos with {\em humans removed} and synthetic data containing virtual humans to pre-train a model. We then evaluate the transferability of the representation learned on this data to a diverse set of downstream action recognition benchmarks. Furthermore, we propose a novel pre-training strategy, called Privacy-Preserving MAE-Align, to effectively combine synthetic data and human-removed real data. Our approach outperforms previous baselines by up to 5\% and closes the performance gap between human and no-human action recognition representations on downstream tasks, for both linear probing and fine-tuning. Our benchmark, code, and models are available at https://github.com/howardzh01/PPMA.</details>

### Learning a 1-layer conditional generative model in total variation

**Authors:** Ajil Jalal, Justin Kang, Ananya Uppal, Kannan Ramchandran, Eric Price

**<details><summary>Abstract**</summary>

A conditional generative model is a method for sampling from a conditional distribution $p(y \mid x)$.  For example, one may want to sample an image of a cat given the label ``cat''.  A feed-forward conditional generative model is a function $g(x, z)$ that takes the input $x$ and a random seed $z$, and outputs a sample $y$ from $p(y \mid x)$.  Ideally the distribution of outputs $(x, g(x, z))$ would be close in total variation to the ideal distribution $(x, y)$.Generalization bounds for other learning models require assumptions on the distribution of $x$, even in simple settings like linear regression with Gaussian noise. We show these assumptions are unnecessary in our model, for both linear regression and single-layer ReLU networks.  Given samples $(x, y)$, we show how to learn a 1-layer ReLU conditional generative model in total variation. As our result has no assumption on the distribution of inputs $x$, if we are given access to the internal activations of a deep generative model, we can compose our 1-layer guarantee to progressively learn the deep model using a near-linear number of samples.</details>

### Learning a Neuron by a Shallow ReLU Network: Dynamics and Implicit Bias for Correlated Inputs

**Authors:** Dmitry Chistikov, Matthias Englert, Ranko Lazic

**<details><summary>Abstract**</summary>

We prove that, for the fundamental regression task of learning a single neuron, training a one-hidden layer ReLU network of any width by gradient flow from a small initialisation converges to zero loss and is implicitly biased to minimise the rank of network parameters.  By assuming that the training points are correlated with the teacher neuron, we complement previous work that considered orthogonal datasets.  Our results are based on a detailed non-asymptotic analysis of the dynamics of each hidden neuron throughout the training.  We also show and characterise a surprising distinction in this setting between interpolator networks of minimal rank and those of minimal Euclidean norm.  Finally we perform a range of numerical experiments, which corroborate our theoretical findings.</details>

### Learning from Rich Semantics and Coarse Locations for Long-tailed Object Detection

**Authors:** Lingchen Meng, Xiyang Dai, Jianwei Yang, Dongdong Chen, Yinpeng Chen, Mengchen Liu, Yi-Ling Chen, Zuxuan Wu, Lu Yuan, Yu-Gang Jiang

**<details><summary>Abstract**</summary>

Long-tailed object detection (LTOD) aims to handle the extreme data imbalance in real-world datasets, where many tail classes have scarce instances. One popular strategy is to explore extra data with image-level labels, yet it produces limited results due to (1) semantic ambiguity---an image-level label only captures a salient part of the image, ignoring the remaining rich semantics within the image; and (2) location sensitivity---the label highly depends on the locations and crops of the original image, which may change after data transformations like random cropping.To remedy this, we propose RichSem, a simple but effective method, which is robust to learn rich semantics from coarse locations without the need of accurate bounding boxes. RichSem leverages rich semantics from images, which are then served as additional ``soft supervision'' for training detectors. Specifically, we add a semantic branch to our detector to learn these soft semantics and enhance feature representations for long-tailed object detection. The semantic branch is only used for training and is removed during inference. RichSem achieves consistent improvements on both overall and rare-category of LVIS under different backbones and detectors. Our method achieves state-of-the-art performance without requiring complex training and testing procedures. Moreover, we show the effectiveness of our method on other long-tailed datasets with additional experiments.</details>

### Learning to Taste: A Multimodal Wine Dataset

**Authors:** Thoranna Bender, Simon Sørensen, Alireza Kashani, Kristjan Hjorleifsson, Grethe Hyldig, Søren Hauberg, Serge Belongie, Frederik Warburg

**<details><summary>Abstract**</summary>

We present WineSensed, a large multimodal wine dataset for studying the relations between visual perception, language, and flavor. The dataset encompasses 897k images of wine labels and 824k reviews of wines curated from the Vivino platform. It has over 350k unique vintages, annotated with year, region, rating, alcohol percentage, price, and grape composition. We obtained fine-grained flavor annotations on a subset by conducting a wine-tasting experiment with 256 participants who were asked to rank wines based on their similarity in flavor, resulting in more than 5k pairwise flavor distances. We propose a low-dimensional concept embedding algorithm that combines human experience with automatic machine similarity kernels. We demonstrate that this shared concept embedding space improves upon separate embedding spaces for coarse flavor classification  (alcohol percentage, country, grape, price, rating) and representing human perception of flavor.</details>

### Lending Interaction Wings to Recommender Systems with Conversational Agents

**Authors:** Jiarui Jin, Xianyu Chen, Fanghua Ye, Mengyue Yang, Yue Feng, Weinan Zhang, Yong Yu, Jun Wang

**<details><summary>Abstract**</summary>

An intelligent conversational agent (a.k.a., chat-bot) could embrace conversational technologies to obtain user preferences online, to overcome inherent limitations of recommender systems trained over the offline historical user behaviors. In this paper, we propose CORE, a new offline-training and online-checking framework to plug a COnversational agent into REcommender systems. Unlike most prior conversational recommendation approaches that systemically combine conversational and recommender parts through a reinforcement learning framework, CORE bridges the conversational agent and recommender system through a unified uncertainty minimization framework, which can be easily applied to any existing recommendation approach. Concretely, CORE treats a recommender system as an offline estimator to produce an estimated relevance score for each item, while CORE regards a conversational agent as an online checker that checks these estimated scores in each online session. We define uncertainty as the sum of unchecked relevance scores. In this regard, the conversational agent acts to minimize uncertainty via querying either attributes or items. Towards uncertainty minimization, we derive the certainty gain of querying each attribute and item, and develop a novel online decision tree algorithm to decide what to query at each turn. Our theoretical analysis reveals the bound of the expected number of turns of CORE in a cold-start setting. Experimental results demonstrate that CORE can be seamlessly employed on a variety of recommendation approaches, and can consistently bring significant improvements in both hot-start and cold-start settings.</details>

### [Spotlight] Let the Flows Tell:  Solving Graph Combinatorial Problems with GFlowNets

**Authors:** Dinghuai Zhang, Hanjun Dai, Nikolay Malkin, Aaron Courville, Yoshua Bengio, Ling Pan

**<details><summary>Abstract**</summary>

Combinatorial optimization (CO) problems are often NP-hard and thus out of reach for exact algorithms, making them a tempting domain to apply machine learning methods. The highly structured constraints in these problems can hinder either optimization or sampling directly in the solution space.On the other hand, GFlowNets have recently emerged as a powerful machinery to efficiently sample from composite unnormalized densities sequentially and have the potential to amortize such solution-searching processes in CO, as well as generate diverse solution candidates.In this paper, we design Markov decision processes (MDPs) for different combinatorial problems and propose to train conditional GFlowNets to sample from the solution space. Efficient training techniques are also developed to benefit long-range credit assignment.Through extensive experiments on a variety of different CO tasks with synthetic and realistic data, we demonstrate that GFlowNet policies can efficiently find high-quality solutions.Our implementation is open-sourced at https://github.com/zdhNarsil/GFlowNet-CombOpt.</details>

### Leveraging Early-Stage Robustness in Diffusion Models for Efficient and High-Quality Image Synthesis

**Authors:** Yulhwa Kim, Dongwon Jo, Hyesung Jeon, Taesu Kim, Daehyun Ahn, Hyungjun Kim, jae-joon kim

**<details><summary>Abstract**</summary>

While diffusion models have demonstrated exceptional image generation capabilities, the iterative noise estimation process required for these models is compute-intensive and their practical implementation is limited by slow sampling speeds. In this paper, we propose a novel approach to speed up the noise estimation network by leveraging the robustness of early-stage diffusion models. Our findings indicate that inaccurate computation during the early-stage of the reverse diffusion process has minimal impact on the quality of generated images, as this stage primarily outlines the image while later stages handle the finer details that require more sensitive information. To improve computational efficiency, we combine our findings with post-training quantization (PTQ) to introduce a method that utilizes low-bit activation for the early reverse diffusion process while maintaining high-bit activation for the later stages. Experimental results show that the proposed method can accelerate the early-stage computation without sacrificing the quality of the generated images.</details>

### LithoBench: Benchmarking AI Computational Lithography for Semiconductor Manufacturing

**Authors:** Su Zheng, Haoyu Yang, Binwu Zhu, Bei Yu, Martin Wong

**<details><summary>Abstract**</summary>

Computational lithography provides algorithmic and mathematical support for resolution enhancement in optical lithography, which is the critical step in semiconductor manufacturing. The time-consuming lithography simulation and mask optimization processes limit the practical application of inverse lithography technology (ILT), a promising solution to the challenges of advanced-node lithography. Although various machine learning methods for ILT have shown promise for reducing the computational burden, this field is in lack of a dataset that can train the models thoroughly and evaluate the performance comprehensively. To boost the development of AI-driven computational lithography, we present the LithoBench dataset, a collection of circuit layout tiles for deep-learning-based lithography simulation and mask optimization. LithoBench consists of more than 120k tiles that are cropped from real circuit designs or synthesized according to the layout topologies of famous ILT testcases. The ground truths are generated by a famous lithography model in academia and an advanced ILT method. Based on the data, we provide a framework to design and evaluate deep neural networks (DNNs) with the data. The framework is used to benchmark state-of-the-art models on lithography simulation and mask optimization. We hope LithoBench can promote the research and development of computational lithography. LithoBench is available at https://anonymous.4open.science/r/lithobench-APPL.</details>

### Lo-Hi: Practical ML Drug Discovery Benchmark

**Authors:** Simon Steshin

**<details><summary>Abstract**</summary>

Finding new drugs is getting harder and harder. One of the hopes of drug discovery is to use machine learning models to predict molecular properties. That is why models for molecular property prediction are being developed and tested on benchmarks such as MoleculeNet. However, existing benchmarks are unrealistic and are too different from applying the models in practice. We have created a new practical \emph{Lo-Hi} benchmark consisting of two tasks: Lead Optimization (Lo) and Hit Identification (Hi), corresponding to the real drug discovery process. For the Hi task, we designed a novel molecular splitting algorithm that solves the Balanced Vertex Minimum $k$-Cut problem. We tested state-of-the-art and classic ML models, revealing which works better under practical settings. We analyzed modern benchmarks and showed that they are unrealistic and overoptimistic.Review: https://openreview.net/forum?id=H2Yb28qGLVLo-Hi benchmark: https://github.com/SteshinSS/lohi_neurips2023Lo-Hi splitter library: https://github.com/SteshinSS/lohi_splitter</details>

### Lung250M-4B: A Combined 3D Dataset for CT- and Point Cloud-Based Intra-Patient Lung Registration

**Authors:** Fenja Falta, Christoph Großbröhmer, Alessa Hering, Alexander Bigalke, Mattias Heinrich

**<details><summary>Abstract**</summary>

A popular benchmark for intra-patient lung registration is provided by the DIR-LAB COPDgene dataset consisting of large-motion in- and expiratory breath-hold CT pairs. This dataset alone, however, does not provide enough samples to properly train state-of-the-art deep learning methods. Other public datasets often also provide only small sample sizes or include primarily small motions between scans that do not translate well to larger deformations. For point-based geometric registration, the PVT1010 dataset provides a large number of vessel point clouds without any correspondences and a labeled test set corresponding to the COPDgene cases. However, the absence of correspondences for supervision complicates training, and a fair comparison with image-based algorithms is infeasible, since CT scans for the training data are not publicly available.We here provide a combined benchmark for image- and point-based registration approaches. We curated a total of 248 public multi-centric in- and expiratory lung CT scans from 124 patients, which show large motion between scans, processed them to ensure sufficient homogeneity between the data and generated vessel point clouds that are well distributed even deeper inside the lungs. For supervised training, we provide vein and artery segmentations of the vessels and multiple thousand image-derived keypoint correspondences for each pair. For validation, we provide multiple scan pairs with manual landmark annotations. Finally, as first baselines on our new benchmark, we evaluate several image and point cloud registration methods on the dataset.</details>

### M$^2$Hub: Unlocking the Potential of Machine Learning for Materials Discovery

**Authors:** Yuanqi Du, Yingheng Wang, Yining Huang, Jianan Canal Li, Yanqiao Zhu, Tian Xie, Chenru Duan, John Gregoire, Carla Gomes

**<details><summary>Abstract**</summary>

We introduce M$^2$Hub, a toolkit for advancing machine learning in materials discovery. Machine learning has achieved remarkable progress in modeling molecular structures, especially biomolecules for drug discovery. However, the development of machine learning approaches for modeling materials structures lag behind, which is partly due to the lack of an integrated platform that enables access to diverse tasks for materials discovery. To bridge this gap, M$^2$Hub will enable easy access to materials discovery tasks, datasets, machine learning methods, evaluations, and benchmark results that cover the entire workflow. Specifically, the first release of M$^2$Hub focuses on three key stages in materials discovery: virtual screening, inverse design, and molecular simulation, including 9 datasets that covers 6 types of materials with 56 tasks across 8 types of material properties. We further provide 2 synthetic datasets for the purpose of generative tasks on materials. In addition to random data splits, we also provide 3 additional data partitions to reflect the real-world materials discovery scenarios. State-of-the-art machine learning methods (including those are suitable for materials structures but never compared in the literature) are benchmarked on representative tasks. Our codes and library are publicly available at \url{https://github.com/yuanqidu/M2Hub}.</details>

### M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models

**Authors:** Wenxuan Zhang, Mahani Aljunied, Chang Gao, Yew Ken Chia, Lidong Bing

**<details><summary>Abstract**</summary>

Despite the existence of various benchmarks for evaluating natural language processing models, we argue that human exams are a more suitable means of evaluating general intelligence for large language models (LLMs), as they inherently demand a much wider range of abilities such as language understanding, domain knowledge, and problem-solving skills. To this end, we introduce M3Exam, a novel benchmark sourced from real and official human exam questions for evaluating LLMs in a multilingual, multimodal, and multilevel context. M3Exam exhibits three unique characteristics: (1) multilingualism, encompassing questions from multiple countries that require strong multilingual proficiency and cultural knowledge; (2) multimodality, accounting for the multimodal nature of many exam questions to test the model's multimodal understanding capability; and (3) multilevel structure, featuring exams from three critical educational periods to comprehensively assess a model's proficiency at different levels. In total, M3Exam contains 12,317 questions in 9 diverse languages with three educational levels, where about 23\% of the questions require processing images for successful solving. We assess the performance of top-performing LLMs on M3Exam and find that current models, including GPT-4, still struggle with multilingual text, particularly in low-resource and non-Latin script languages. Multimodal LLMs also perform poorly with complex multimodal questions. We believe that M3Exam can be a valuable resource for comprehensively evaluating LLMs by examining their multilingual and multimodal abilities and tracking their development. Data and evaluation code is available at \url{https://github.com/DAMO-NLP-SG/M3Exam}.</details>

### MADG: Margin-based Adversarial Learning for Domain Generalization

**Authors:** Aveen Dayal, Vimal K B, Linga Reddy Cenkeramaddi, C Mohan, Abhinav Kumar, Vineeth N Balasubramanian

**<details><summary>Abstract**</summary>

Domain Generalization (DG) techniques have emerged as a popular approach to address the challenges of domain shift in Deep Learning (DL), with the goal of generalizing well to the target domain unseen during the training. In recent years, numerous methods have been proposed to address the DG setting, among which one popular approach is the adversarial learning-based methodology. The main idea behind adversarial DG methods is to learn domain-invariant features by minimizing a discrepancy metric. However, most adversarial DG methods use 0-1 loss based $\mathcal{H}\Delta\mathcal{H}$ divergence metric. In contrast, the margin loss-based discrepancy metric has the following advantages: more informative, tighter, practical, and efficiently optimizable. To mitigate this gap, this work proposes a novel adversarial learning DG algorithm, $\textbf{MADG}$, motivated by a margin loss-based discrepancy metric. The proposed $\textbf{MADG}$ model learns domain-invariant features across all source domains and uses adversarial training to generalize well to the unseen target domain. We also provide a theoretical analysis of the proposed $\textbf{MADG}$ model based on the unseen target error bound. Specifically, we construct the link between the source and unseen domains in the real-valued hypothesis space and derive the generalization bound using margin loss and Rademacher complexity. We extensively experiment with the $\textbf{MADG}$ model on popular real-world DG datasets, VLCS, PACS, OfficeHome, DomainNet, and TerraIncognita. We evaluate the proposed algorithm on DomainBed's benchmark and observe consistent performance across all the datasets.</details>

### MADLAD-400: A Multilingual And Document-Level Large Audited Dataset

**Authors:** Sneha Kudugunta, Isaac Caswell, Biao Zhang, Xavier Garcia, Derrick Xin, Aditya Kusupati, Romi Stella, Ankur Bapna, Orhan Firat

**<details><summary>Abstract**</summary>

We introduce MADLAD-400, a manually audited, general domain 3T token monolingual dataset based on CommonCrawl, spanning 419 languages. We discuss the limitations revealed by self-auditing MADLAD-400, and the role data auditing had in the dataset creation process. We then train and release a 10.7B-parameter multilingual machine translation model on 250 billion tokens covering over 450 languages using publicly available data, and find that it is competitive with models that are significantly larger, and report the results on different domains. In addition, we train a 8B-parameter language model, and assess the results on few-shot translation. We make the baseline models available to the research community.</details>

### MARBLE: Music Audio Representation Benchmark for Universal Evaluation

**Authors:** Ruibin Yuan, Yinghao Ma, Yizhi Li, Ge Zhang, Xingran Chen, Hanzhi Yin, zhuo le, Yiqi Liu, Jiawen Huang, Zeyue Tian, Binyue Deng, Ningzhi Wang, Chenghua Lin, Emmanouil Benetos, Anton Ragni, Norbert Gyenge, Roger Dannenberg, wenhu chen, Gus Xia, Wei Xue, Si Liu, Shi Wang, Ruibo Liu, Yike Guo, Jie Fu

**<details><summary>Abstract**</summary>

In the era of extensive intersection between art and Artificial Intelligence (AI), such as image generation and fiction co-creation, AI for music remains relatively nascent, particularly in music understanding. This is evident in the limited work on deep music representations, the scarcity of large-scale datasets, and the absence of a universal and community-driven benchmark. To address this issue, we introduce the Music Audio Representation Benchmark for universaL Evaluation, termed MARBLE. It aims to provide a benchmark for various Music Information Retrieval (MIR) tasks by defining a comprehensive taxonomy with four hierarchy levels, including acoustic, performance, score, and high-level description. We then establish a unified protocol based on 18 tasks on 12 public-available datasets, providing a fair and standard assessment of representations of all open-sourced pre-trained models developed on music recordings as baselines. Besides, MARBLE offers an easy-to-use, extendable, and reproducible suite for the community, with a clear statement on copyright issues on datasets. Results suggest recently proposed large-scale pre-trained musical language models perform the best in most tasks, with room for further improvement. The leaderboard and toolkit repository are published to promote future music AI research.</details>

### MLFMF: Data Sets for Machine Learning for Mathematical Formalization

**Authors:** Andrej Bauer, Matej Petković, Ljupco Todorovski

**<details><summary>Abstract**</summary>

We introduce MLFMF, a collection of data sets for benchmarking recommendation systems used to support formalization of mathematics with proof assistants. These systems help humans identify which previous entries (theorems, constructions, datatypes, and postulates) are relevant in proving a new theorem or carrying out a new construction. Each data set is derived from a library of formalized mathematics written in proof assistants Agda or Lean. The collection includes the largest Lean 4 library Mathlib, and some of the largest Agda libraries: the standard library, the library of univalent mathematics Agda-unimath, and the TypeTopology library. Each data set represents the corresponding library in two ways: as a heterogeneous network, and as a list of s-expressions representing the syntax trees of all the entries in the library. The network contains the (modular) structure of the library and the references between entries, while the s-expressions give complete and easily parsed information about every entry.We report baseline results using standard graph and word embeddings, tree ensembles, and instance-based learning algorithms. The MLFMF data sets provide solid benchmarking support for further investigation of the numerous machine learning approaches to formalized mathematics. The methodology used to extract the networks and the s-expressions readily applies to other libraries, and is applicable to other proof assistants. With more than $250\,000$ entries in total, this is currently the largest collection of formalized mathematical knowledge in machine learnable format.</details>

### MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing

**Authors:** Jianfei Yang, He Huang, Yunjiao Zhou, Xinyan Chen, Yuecong Xu, Shenghai Yuan, Han Zou, Chris Xiaoxuan Lu, Lihua Xie

**<details><summary>Abstract**</summary>

4D human perception plays an essential role in a myriad of applications, such as home automation and metaverse avatar simulation. However, existing solutions which mainly rely on cameras and wearable devices are either privacy intrusive or inconvenient to use. To address these issues, wireless sensing has emerged as a promising alternative, leveraging LiDAR, mmWave radar, and WiFi signals for device-free human sensing. In this paper, we propose MM-Fi, the first multi-modal non-intrusive 4D human dataset with 27 daily or rehabilitation action categories, to bridge the gap between wireless sensing and high-level human perception tasks. MM-Fi consists of over 320k synchronized frames of five modalities from 40 human subjects. Various annotations are provided to support potential sensing tasks, e.g., human pose estimation and action recognition. Extensive experiments have been conducted to compare the sensing capacity of each or several modalities in terms of multiple tasks. We envision that MM-Fi can contribute to wireless sensing research with respect to action recognition, human pose estimation, multi-modal learning, cross-modal supervision, and interdisciplinary healthcare research.</details>

### MMD Aggregated Two-Sample Test

**Authors:** Antonin Schrab, Ilmun Kim, Mélisande Albert, Béatrice Laurent, Benjamin Guedj, Arthur Gretton

**<details><summary>Abstract**</summary>

We propose two novel nonparametric two-sample kernel tests based on the Maximum Mean Discrepancy (MMD). First, for a fixed kernel, we construct an MMD test using either permutations or a wild bootstrap, two popular numerical procedures to determine the test threshold. We prove that this test controls the probability of type I error non-asymptotically. Hence, it can be used reliably even in settings with small sample sizes as it remains well-calibrated, which differs from previous MMD tests which only guarantee correct test level asymptotically. When the difference in densities lies in a Sobolev ball, we prove minimax optimality of our MMD test with a specific kernel depending on the smoothness parameter of the Sobolev ball. In practice, this parameter is unknown and, hence, the optimal MMD test with this particular kernel cannot be used. To overcome this issue, we construct an aggregated test, called MMDAgg, which is adaptive to the smoothness parameter. The test power is maximised over the collection of kernels used, without requiring held-out data for kernel selection (which results in a loss of test power), or arbitrary kernel choices such as the median heuristic. We prove that MMDAgg still controls the level non-asymptotically, and achieves the minimax rate over Sobolev balls, up to an iterated logarithmic term. Our guarantees are not restricted to a specific type of kernel, but hold for any product of one-dimensional translation invariant characteristic kernels. We provide a user-friendly parameter-free implementation of MMDAgg using an adaptive collection of bandwidths. We demonstrate that MMDAgg significantly outperforms alternative state-of-the-art MMD-based two-sample tests on synthetic data satisfying the Sobolev smoothness assumption, and that, on real-world image data, MMDAgg closely matches the power of tests leveraging the use of models such as neural networks.</details>

### [Spotlight] MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion

**Authors:** Shitao Tang, Fuyang Zhang, Jiacheng Chen, Peng Wang, Yasutaka Furukawa

**<details><summary>Abstract**</summary>

This paper introduces MVDiffusion, a simple yet effective multi-view image generation method for scenarios where pixel-to-pixel correspondences are available, such as perspective crops from panorama or multi-view images given geometry (depth maps and poses). Unlike prior methods that rely on iterative image warping and inpainting, MVDiffusion concurrently generates all images with a global awareness, encompassing high resolution and rich content, effectively addressing the error accumulation prevalent in preceding models. MVDiffusion specifically incorporates a correspondence-aware attention mechanism, enabling effective cross-view interaction. This mechanism underpins three pivotal modules: 1) a generation module that produces low-resolution images while maintaining global correspondence, 2) an interpolation module that densifies spatial coverage between images, and 3) a super-resolution module that upscales into high-resolution images. In terms of panoramic imagery, MVDiffusion generates high-resolution photorealistic images up to 1024*1024 pixels. For geometry-conditioned multi-view image generation, MVDiffusion demonstrates state-of-the-art performance on texture-map generation for a given scene mesh. We recommend referring to our Arxiv version at https://arxiv.org/pdf/2307.01097.pdf for the latest update. The project page is at https://mvdiffusion.github.io/.</details>

### MVDoppler: Unleashing the Power of Multi-View Doppler for MicroMotion-based Gait Classification

**Authors:** Soheil Hor, Shubo Yang, Jaeho Choi, Amin Arbabian

**<details><summary>Abstract**</summary>

Modern perception systems rely heavily on high-resolution cameras, LiDARs, and advanced deep neural networks, enabling exceptional performance across various applications. However, these optical systems predominantly depend on geometric features and shapes of objects, which can be challenging to capture in long-range perception applications. To overcome this limitation, alternative approaches such as Doppler-based perception using high-resolution radars have been proposed. Doppler-based systems are capable of measuring micro-motions of targets remotely and with very high precision. When compared to geometric features, the resolution of micro-motion features exhibits significantly greater resilience to the influence of distance. However, the true potential of Doppler-based perception has yet to be fully realized due to several factors. These include the unintuitive nature of Doppler signals, the limited availability of public Doppler datasets, and the current datasets' inability to capture the specific co-factors that are unique to Doppler-based perception, such as the effect of the radar's observation angle and the target's motion trajectory.This paper introduces a new large multi-view Doppler dataset together with baseline perception models for micro-motion-based gait analysis and classification. The dataset captures the impact of the subject's walking trajectory and radar's observation angle on the classification performance. Additionally, baseline multi-view data fusion techniques are provided to mitigate these effects. This work demonstrates that sub-second micro-motion snapshots can be sufficient for reliable detection of hand movement patterns and even changes in a pedestrian's walking behavior when distracted by their phone. Overall, this research not only showcases the potential of Doppler-based perception, but also offers valuable solutions to tackle its fundamental challenges.</details>

### MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing

**Authors:** Kai Zhang, Lingbo Mo, wenhu chen, Huan Sun, Yu Su

**<details><summary>Abstract**</summary>

Text-guided image editing is widely needed in daily life, ranging from personal use to professional applications such as Photoshop.However, existing methods are either zero-shot or trained on an automatically synthesized dataset, which contains a high volume of noise.Thus, they still require lots of manual tuning to produce desirable outcomes in practice.To address this issue, we introduce MagicBrush, the first large-scale, manually annotated dataset for instruction-guided real image editing that covers diverse scenarios: single-turn, multi-turn, mask-provided, and mask-free editing.MagicBrush comprises over 10K manually annotated triplets (source image, instruction, target image), which supports trainining large-scale text-guided image editing models.We fine-tune InstructPix2Pix on MagicBrush and show that the new model can produce much better images according to human evaluation.We further conduct extensive experiments to evaluate current image editing baselines from multiple dimensions including quantitative, qualitative, and human evaluations.The results reveal the challenging nature of our dataset and the gap between current baselines and real-world editing needs.</details>

### Massively Multilingual Corpus of Sentiment Datasets and Multi-faceted Sentiment Classification Benchmark

**Authors:** Lukasz Augustyniak, Szymon Woźniak, Marcin Gruza, Piotr Gramacki, Krzysztof Rajda, Mikołaj Morzy, Tomasz Kajdanowicz

**<details><summary>Abstract**</summary>

Despite impressive advancements in multilingual corpora collection and model training, developing large-scale deployments of multilingual models still presents a significant challenge. This is particularly true for language tasks that are culture-dependent. One such example is the area of multilingual sentiment analysis, where affective markers can be subtle and deeply ensconced in culture.This work presents the most extensive open massively multilingual corpus of datasets for training sentiment models. The corpus consists of 79 manually selected datasets from over 350 datasets reported in the scientific literature based on strict quality criteria. The corpus covers 27 languages representing 6 language families. Datasets can be queried using several linguistic and functional features. In addition, we present a multi-faceted sentiment classification benchmark summarizing hundreds of experiments conducted on different base models, training objectives, dataset collections, and fine-tuning strategies.</details>

### [Spotlight] Maximization of Average Precision for Deep Learning with Adversarial Ranking Robustness

**Authors:** Gang Li, Wei Tong, Tianbao Yang

**<details><summary>Abstract**</summary>

This paper seeks to address a gap in optimizing Average Precision (AP) while ensuring adversarial robustness, an area that has not been extensively explored to the best of our knowledge. AP maximization for deep learning has widespread applications, particularly when there is a significant imbalance between positive and negative examples. Although numerous studies have been conducted on adversarial training, they primarily focus on robustness concerning accuracy, ensuring that the average accuracy on adversarially perturbed examples is well maintained. However, this type of adversarial robustness is insufficient for many applications, as minor perturbations on a single example can significantly impact AP  while not greatly influencing the accuracy of the prediction system. To tackle this issue, we introduce a novel formulation that combines an AP surrogate loss with a regularization term representing adversarial ranking robustness, which maintains the consistency between ranking of clean data and that of perturbed data. We then devise an efficient stochastic optimization algorithm to optimize the resulting objective. Our empirical studies, which compare our method to current leading adversarial training baselines and other robust AP maximization strategies, demonstrate the effectiveness of the proposed approach. Notably, our methods outperform a state-of-the-art method (TRADES) by more than 4\% in terms of robust AP against PGD attacks while achieving 7\% higher AP on clean data simultaneously on CIFAR10 and CIFAR100.</details>

### Mechanic: A Learning Rate Tuner

**Authors:** Ashok Cutkosky, Aaron Defazio, Harsh Mehta

**<details><summary>Abstract**</summary>

We introduce a technique for tuning the learning rate scale factor of any base optimization algorithm and schedule automatically, which we call Mechanic. Our method provides a practical realization of recent theoretical reductions for accomplishing a similar goal in online convex optimization. We rigorously evaluate Mechanic on a range of large scale deep learning tasks with varying batch sizes, schedules, and base optimization algorithms. These experiments demonstrate that depending on the problem, Mechanic either comes very close to, matches or even improves upon manual tuning of learning rates.</details>

### [Spotlight] Mechanism Design for Collaborative Normal Mean Estimation

**Authors:** Yiding Chen, Jerry Zhu, Kirthevasan Kandasamy

**<details><summary>Abstract**</summary>

We study collaborative normal mean estimation, where $m$ strategic agents collect i.i.d samples from a normal distribution $\mathcal{N}(\mu, \sigma^2)$ at a cost. They all wish to estimate the mean $\mu$. By sharing data with each other, agents can obtain better estimates while keeping the cost of data collection small. To facilitate this collaboration, we wish to design mechanisms that encourage agents to collect a sufficient amount of data and share it truthfully, so that they are all better off than working alone. In naive mechanisms, such as simply pooling and sharing all the data, an individual agent might find it beneficial to under-collect and/or fabricate data, which can lead to poor social outcomes. We design a novel mechanism that overcomes these challenges via two key techniques: first, when sharing the others' data with an agent, the mechanism corrupts this dataset proportional to how much the data reported by the agent differs from the others; second, we design minimax optimal estimators for the corrupted dataset. Our mechanism, which is Nash incentive compatible and individually rational, achieves a social penalty (sum of all agents' estimation errors and data collection costs) that is at most a factor 2 of the global minimum. When applied to high dimensional (non-Gaussian) distributions with bounded variance, this mechanism retains these three properties, but with slightly weaker results. Finally, in two special cases where we restrict the strategy space of the agents, we design mechanisms that essentially achieve the global minimum.</details>

### MedSat: A Public Health Dataset for England Featuring Medical Prescriptions and Satellite Imagery

**Authors:** Sanja Scepanovic, Ivica Obadic, Sagar Joglekar, Laura GIUSTARINI, Cristiano Nattero, Daniele Quercia, Xiaoxiang Zhu

**<details><summary>Abstract**</summary>

As extreme weather events become more frequent, understanding their impact on human health becomes increasingly crucial. However, the utilization of Earth Observation to effectively analyze the environmental context in relation to health remains limited. This limitation is primarily due to the lack of fine-grained spatial and temporal data in public and population health studies, hindering a comprehensive understanding of health outcomes. Additionally, obtaining appropriate environmental indices across different geographical levels and timeframes poses a challenge. For the years 2019 (pre-COVID) and 2020 (COVID), we collected spatio-temporal indicators for all Lower Layer Super Output Areas in England. These indicators included: i) 111 sociodemographic features linked to health in existing literature, ii) 43 environmental point features (e.g., greenery and air pollution levels), iii) 4 seasonal composite satellite images each with 11 bands, and iv) prescription prevalence associated with five medical conditions (depression, anxiety, diabetes, hypertension, and asthma), opioids and total prescriptions. We combined these indicators into a single MedSat dataset, the availability of which presents an opportunity for the machine learning community to develop new techniques specific to public health. These techniques would address challenges such as handling large and complex data volumes, performing effective feature engineering on environmental and sociodemographic factors, capturing spatial and temporal dependencies in the models, addressing imbalanced data distributions, developing novel computer vision methods for health modeling based on satellite imagery, ensuring model explainability, and achieving generalization beyond the specific geographical region.</details>

### Meta-in-context learning in large language models

**Authors:** Julian Coda-Forno, Marcel Binz, Zeynep Akata, Matt Botvinick, Jane Wang, Eric Schulz

**<details><summary>Abstract**</summary>

Large language models have shown tremendous performance in a variety of tasks.  In-context learning  -- the ability to improve at a task after being provided with a number of demonstrations -- is seen as one of the main contributors to their success. In the present paper, we demonstrate that the in-context learning abilities of large language models can be recursively improved via in-context learning itself. We coin this phenomenon meta-in-context learning. Looking at two idealized domains, a one-dimensional regression task and a two-armed bandit task, we show that meta-in-context learning adaptively reshapes a large language model's priors over expected tasks. Furthermore, we find that meta-in-context learning modifies the in-context learning strategies of such models. Finally, we broaden the scope of our investigation to encompass two diverse benchmarks: one focusing on real-world regression problems and the other encompassing multiple NLP tasks. In both cases, we observe competitive performance comparable to that of traditional learning algorithms. Taken together, our work improves our understanding of in-context learning and paves the way toward adapting large language models to the environment they are applied purely through meta-in-context learning rather than traditional finetuning.</details>

### [Oral] MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning

**Authors:** Zeyuan Ma, Hongshu Guo, Jiacheng Chen, Zhenrui Li, Guojun Peng, Yue-Jiao Gong, Yining Ma, Zhiguang Cao

**Oral Presentation:** Th, Dec 14, 14:05 -- Oral 6B

**<details><summary>Abstract**</summary>

Recently, Meta-Black-Box Optimization with Reinforcement Learning (MetaBBO-RL) has showcased the power of leveraging RL at the meta-level to mitigate manual fine-tuning of low-level black-box optimizers. However, this field is hindered by the lack of a unified benchmark. To fill this gap, we introduce MetaBox, the first benchmark platform expressly tailored for developing and evaluating MetaBBO-RL methods. MetaBox offers a flexible algorithmic template that allows users to effortlessly implement their unique designs within the platform. Moreover, it provides a broad spectrum of over 300 problem instances, collected from synthetic to realistic scenarios, and an extensive library of 19 baseline methods, including both traditional black-box optimizers and recent MetaBBO-RL methods. Besides, MetaBox introduces three standardized performance metrics, enabling a more thorough assessment of the methods. In a bid to illustrate the utility of MetaBox for facilitating rigorous evaluation and in-depth analysis, we carry out a wide-ranging benchmarking study on existing MetaBBO-RL methods. Our MetaBox is open-source and accessible at: https://github.com/GMC-DRL/MetaBox.</details>

### MiliPoint: A Point Cloud Dataset for mmWave Radar

**Authors:** Han Cui, Shu Zhong, Jiacheng Wu, Zichao Shen, Naim Dahnoun, Yiren Zhao

**<details><summary>Abstract**</summary>

Millimetre-wave (mmWave) radar has emerged as an attractive and cost-effective alternative for human activity sensing compared to traditional camera-based systems. mmWave radars are also non-intrusive, providing better protection for user privacy. However, as a Radio Frequency based technology, mmWave radars rely on capturing reflected signals from objects, making them more prone to noise compared to cameras. This raises an intriguing question for the deep learning community: Can we develop more effective point set-based deep learning methods for such attractive sensors?   To answer this question, our work, termed MiliPoint, delves into this idea by providing a large-scale, open dataset for the community to explore how mmWave radars can be utilised for human activity recognition. Moreover, MiliPoint stands out as it is larger in size than existing datasets, has more diverse human actions represented, and encompasses all three key tasks in human activity recognition. We have also established a range of point-based deep neural networks such as DGCNN, PointNet++ and PointTransformer, on MiliPoint, which can serve to set the ground baseline for further development.</details>

### Mind the spikes: Benign overfitting of kernels and neural networks in fixed dimension

**Authors:** Moritz Haas, David Holzmüller, Ulrike Luxburg, Ingo Steinwart

**<details><summary>Abstract**</summary>

The success of over-parameterized neural networks trained to near-zero training error has caused great interest in the phenomenon of benign overfitting, where estimators are statistically consistent even though they interpolate noisy training data. While benign overfitting in fixed dimension has been established for some learning methods, current literature suggests that for regression with typical kernel methods and wide neural networks, benign overfitting requires a high-dimensional setting, where the dimension grows with the sample size. In this paper, we show that the smoothness of the estimators, and not the dimension, is the key: benign overfitting is possible if and only if the estimator's derivatives are large enough. We generalize existing inconsistency results to non-interpolating models and more kernels to show that benign overfitting with moderate derivatives is impossible in fixed dimension. Conversely, we show that benign overfitting is possible for regression with a sequence of spiky-smooth kernels with large derivatives. Using neural tangent kernels, we translate our results to wide neural networks. We prove that while infinite-width networks do not overfit benignly with the ReLU activation, this can be fixed by adding small high-frequency fluctuations to the activation function. Our experiments verify that such neural networks, while overfitting, can indeed generalize well even on low-dimensional data sets.</details>

### Mind2Web: Towards a Generalist Agent for the Web

**Authors:** Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, Yu Su

**<details><summary>Abstract**</summary>

We introduce Mind2Web, the first dataset for developing and evaluating generalist agents for the web that can follow language instructions to complete complex tasks on any website. Existing datasets for web agents either use simulated websites or only cover a limited set of websites and tasks, thus not suitable for generalist web agents. With over 2,000 open-ended tasks collected from 137 websites spanning 31 domains and crowdsourced action sequences for the tasks, Mind2Web provides three necessary ingredients for building generalist web agents: 1) diverse domains, websites, and tasks, 2) use of real-world websites instead of simulated and simplified ones, and 3) a broad spectrum of user interaction patterns. Based on Mind2Web, we conduct an initial exploration of using large language models (LLMs) for building generalist web agents. While the raw HTML of real-world websites are often too large to be fed to LLMs, we show that first filtering it with a small LM significantly improves the effectiveness and efficiency of LLMs. Our solution demonstrates a decent level of performance, even on websites or entire domains the model has never seen before, but there is still a substantial room to improve towards truly generalizable agents. We open-source our dataset, model implementation, and trained models (https://osu-nlp-group.github.io/Mind2Web) to facilitate further research on building a generalist agent for the web.</details>

### Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks

**Authors:** Maxime Chevalier-Boisvert, Bolun Dai, Mark Towers, Rodrigo Perez-Vicente, Lucas Willems, Salem Lahlou, Suman Pal, Pablo Samuel Castro, J Terry

**<details><summary>Abstract**</summary>

We present the Minigrid and Miniworld libraries which provide a suite of goal-oriented 2D and 3D environments. The libraries were explicitly created with a minimalistic design paradigm to allow users to rapidly develop new environments for a wide range of research-specific needs. As a result, both have received widescale adoption by the RL community, facilitating research in a wide range of areas. In this paper, we outline the design philosophy, environment details, and their world generation API. We also showcase the additional capabilities brought by the unified API between Minigrid and Miniworld through case studies on transfer learning (for both RL agents and humans) between the different observation spaces. The source code of Minigrid and Miniworld can be found at https://github.com/Farama-Foundation/Minigrid and https://github.com/Farama-Foundation/Miniworld along with their documentation at https://minigrid.farama.org/ and https://miniworld.farama.org/.</details>

### Minimax Forward and Backward Learning of Evolving Tasks with Performance Guarantees

**Authors:** Veronica Alvarez, Santiago Mazuelas, Jose A. Lozano

**<details><summary>Abstract**</summary>

For a sequence of classification tasks that arrive over time, it is common that tasks are evolving in the sense that consecutive tasks often have a higher similarity. The incremental learning of a growing sequence of tasks holds promise to enable accurate classification even with few samples per task by leveraging information from all the tasks in the sequence (forward and backward learning). However, existing techniques developed for continual learning and concept drift adaptation are either designed for tasks with time-independent similarities or only aim to learn the last task in the sequence. This paper presents incremental minimax risk classifiers (IMRCs) that effectively exploit forward and backward learning and account for evolving tasks. In addition, we analytically characterize the performance improvement provided by forward and backward learning in terms of the tasks’ expected quadratic change and the number of tasks. The experimental evaluation shows that IMRCs can result in a significant performance improvement, especially for reduced sample sizes.</details>

### Minimax Optimal Rate for Parameter Estimation in Multivariate Deviated Models

**Authors:** Dat Do, Huy Nguyen, Khai Nguyen, Nhat Ho

**<details><summary>Abstract**</summary>

We study the maximum likelihood estimation (MLE) in the multivariate deviated model where the data are generated from the density function $(1-\lambda^{\ast})h_{0}(x)+\lambda^{\ast}f(x|\mu^{\ast}, \Sigma^{\ast})$ in which $h_{0}$ is a known function, $\lambda^{\ast} \in [0,1]$ and $(\mu^{\ast}, \Sigma^{\ast})$ are unknown parameters to estimate. The main challenges in deriving the convergence rate of the MLE mainly come from two issues: (1) The interaction between the function $h_{0}$ and the density function $f$; (2) The deviated proportion $\lambda^{\ast}$ can go to the extreme points of $[0,1]$ as the sample size tends to infinity. To address these challenges, we develop the \emph{distinguishability condition} to capture the linear independent relation between the function $h_{0}$ and the density function $f$. We then provide comprehensive convergence rates of the MLE via the vanishing rate of $\lambda^{\ast}$ to zero as well as the distinguishability of two functions $h_{0}$ and $f$.</details>

### Modeling Human Visual Motion Processing with Trainable Motion Energy Sensing and a Self-attention Network

**Authors:** Zitang Sun, Yen-Ju Chen, Yung-Hao Yang, Shin'ya Nishida

**<details><summary>Abstract**</summary>

Visual motion processing is essential for humans to perceive and interact with dynamic environments. Despite extensive research in cognitive neuroscience, image-computable models that can extract informative motion flow from natural scenes in a manner consistent with human visual processing have yet to be established. Meanwhile, recent advancements in computer vision (CV), propelled by deep learning, have led to significant progress in optical flow estimation, a task closely related to motion perception. Here we propose an image-computable model of human motion perception by bridging the gap between biological and CV models. Specifically, we introduce a novel two-stages approach that combines trainable motion energy sensing with a recurrent self-attention network for adaptive motion integration and segregation. This model architecture aims to capture the computations in V1-MT, the core structure for motion perception in the biological visual system, while providing the ability to derive informative motion flow for a wide range of stimuli, including complex natural scenes. In silico neurophysiology reveals that our model's unit responses are similar to mammalian neural recordings regarding motion pooling and speed tuning. The proposed model can also replicate human responses to a range of stimuli examined in past psychophysical studies. The experimental results on the Sintel benchmark demonstrate that our model predicts human responses better than the ground truth, whereas the state-of-the-art CV models show the opposite. Our study provides a computational architecture consistent with human visual motion processing, although the physiological correspondence may not be exact.</details>

### Molecule Joint Auto-Encoding: Trajectory Pretraining with 2D and 3D Diffusion

**Authors:** weitao Du, Jiujiu Chen, Xuecang Zhang, Zhi-Ming Ma, Shengchao Liu

**<details><summary>Abstract**</summary>

Recently, artificial intelligence for drug discovery has raised increasing interest in both machine learning and chemistry domains. The fundamental building block for drug discovery is molecule geometry and thus, the molecule's geometrical representation is the main bottleneck to better utilize machine learning techniques for drug discovery. In this work, we propose a pretraining method for molecule joint auto-encoding (MoleculeJAE). MoleculeJAE can learn both the 2D bond (topology) and 3D conformation (geometry) information, and a diffusion process model is applied to mimic the augmented trajectories of such two modalities, based on which, MoleculeJAE will learn the inherent chemical structure in a self-supervised manner. Thus, the pretrained geometrical representation in MoleculeJAE is expected to benefit downstream geometry-related tasks. Empirically, MoleculeJAE proves its effectiveness by reaching state-of-the-art performance on 15 out of 20 tasks by comparing it with 12 competitive baselines.</details>

### Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset

**Authors:** Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, Lei Zhang

**<details><summary>Abstract**</summary>

In this paper, we present Motion-X, a large-scale 3D expressive whole-body motion dataset. Existing motion datasets predominantly contain body-only poses, lacking facial expressions, hand gestures, and fine-grained pose descriptions. Moreover, they are primarily collected from limited laboratory scenes with textual descriptions manually labeled, which greatly limits their scalability. To overcome these limitations, we develop a whole-body motion and text annotation pipeline, which can automatically annotate motion from either single- or multi-view videos and provide comprehensive semantic labels for each video and fine-grained whole-body pose descriptions for each frame. This pipeline is of high precision, cost-effective, and scalable for further research. Based on it, we construct Motion-X, which comprises 15.6M precise 3D whole-body pose annotations (i.e., SMPL-X) covering 81.1K motion sequences from massive scenes. Besides, Motion-X provides 15.6M frame-level whole-body pose descriptions and 81.1K sequence-level semantic labels. Comprehensive experiments demonstrate the accuracy of the annotation pipeline and the significant benefit of Motion-X in enhancing expressive, diverse, and natural motion generation, as well as 3D whole-body human mesh recovery.</details>

### Multi-scale Diffusion Denoised Smoothing

**Authors:** Jongheon Jeong, Jinwoo Shin

**<details><summary>Abstract**</summary>

Along with recent diffusion models, randomized smoothing has become one of a few tangible approaches that offers adversarial robustness to models at scale, e.g., those of large pre-trained models. Specifically, one can perform randomized smoothing on any classifier via a simple "denoise-and-classify" pipeline, so-called denoised smoothing, given that an accurate denoiser is available - such as diffusion model. In this paper, we present scalable methods to address the current trade-off between certified robustness and accuracy in denoised smoothing. Our key idea is to "selectively" apply smoothing among multiple noise scales, coined multi-scale smoothing, which can be efficiently implemented with a single diffusion model. This approach also suggests a new objective to compare the collective robustness of multi-scale smoothed classifiers, and questions which representation of diffusion model would maximize the objective. To address this, we propose to further fine-tune diffusion model (a) to perform consistent denoising whenever the original image is recoverable, but (b) to generate rather diverse outputs otherwise. Our experiments show that the proposed multi-scale smoothing scheme, combined with diffusion fine-tuning, not only allows strong certified robustness at high noise scales but also maintains accuracy close to non-smoothed classifiers. Code is available at https://github.com/jh-jeong/smoothing-multiscale.</details>

### MultiVENT: Multilingual Videos of Events and Aligned Natural Text

**Authors:** Kate Sanders, David Etter, Reno Kriz, Benjamin Van Durme

**<details><summary>Abstract**</summary>

Everyday news coverage has shifted from traditional broadcasts towards a wide range of presentation formats such as first-hand, unedited video footage. Datasets that reflect the diverse array of multimodal, multilingual news sources available online could be used to teach models to benefit from this shift, but existing news video datasets focus on traditional news broadcasts produced for English-speaking audiences. We address this limitation by constructing MultiVENT, a dataset of multilingual, event-centric videos grounded in text documents across five target languages. MultiVENT includes both news broadcast videos and non-professional event footage, which we use to analyze the state of online news videos and how they can be leveraged to build robust, factually accurate models. Finally, we provide a  model for complex, multilingual video retrieval to serve as a baseline for information retrieval using MultiVENT.</details>

### Multimodal Clinical Benchmark for Emergency Care (MC-BEC): A Comprehensive Benchmark for Evaluating Foundation Models in Emergency Medicine

**Authors:** Emma Chen, Aman Kansal, Julie Chen, Boyang Tom Jin, Julia Reisler, David Kim, Pranav Rajpurkar

**<details><summary>Abstract**</summary>

We propose the Multimodal Clinical Benchmark for Emergency Care (MC-BEC), a comprehensive benchmark for evaluating foundation models in Emergency Medicine using a dataset of 100K+ continuously monitored Emergency Department visits from 2020-2022. MC-BEC focuses on clinically relevant prediction tasks at timescales from minutes to days, including predicting patient decompensation, disposition, and emergency department (ED) revisit, and includes a standardized evaluation framework with train-test splits and evaluation metrics. The multimodal dataset includes a wide range of detailed clinical data, including triage information, prior diagnoses and medications, continuously measured vital signs, electrocardiogram and photoplethysmograph waveforms, orders placed and medications administered throughout the visit, free-text reports of imaging studies, and information on ED diagnosis, disposition, and subsequent revisits. We provide performance baselines for each prediction task to enable the evaluation of multimodal, multitask models. We believe that MC-BEC will encourage researchers to develop more effective, generalizable, and accessible foundation models for multimodal clinical data.</details>

### NAVI: Category-Agnostic Image Collections with High-Quality 3D Shape and Pose Annotations

**Authors:** Varun Jampani, Kevis-kokitsi Maninis, Andreas Engelhardt, Arjun Karpur, Karen Truong, Kyle Sargent, Stefan Popov, Andre Araujo, Ricardo Martin Brualla, Kaushal Patel, Daniel Vlasic, Vittorio Ferrari, Ameesh Makadia, Ce Liu, Yuanzhen Li, Howard Zhou

**<details><summary>Abstract**</summary>

Recent advances in neural reconstruction enable high-quality 3D object reconstruction from casually captured image collections. Current techniques mostly analyze their progress on relatively simple image collections where SfM techniques can provide ground-truth (GT) camera poses. We note that SfM techniques tend to fail on in-the-wild image collections such as image search results with varying backgrounds and illuminations. To enable systematic research progress on 3D reconstruction from casual image captures, we propose `NAVI': a new dataset of category-agnostic image collections of objects with high-quality 3D scans along with per-image 2D-3D alignments providing near-perfect GT camera parameters. These 2D-3D alignments allow us to extract accurate derivative annotations such as dense pixel correspondences, depth and segmentation maps. We demonstrate the use of NAVI image collections on different problem settings and show that NAVI enables more thorough evaluations that were not possible with existing datasets. We believe NAVI is beneficial for systematic research progress on 3D reconstruction and correspondence estimation.</details>

### NIS3D: A Completely Annotated Benchmark for Dense 3D Nuclei Image Segmentation

**Authors:** Wei Zheng, Cheng Peng, Zeyuan Hou, Boyu Lyu, Mengfan Wang, Xuelong Mi, Shuoxuan Qiao, Yinan Wan, Guoqiang Yu

**<details><summary>Abstract**</summary>

3D segmentation of nuclei images is a fundamental task for many biological studies. Despite the rapid advances of large-volume 3D imaging acquisition methods and the emergence of sophisticated algorithms to segment the nuclei in recent years, a benchmark with all cells completely annotated is still missing, making it hard to accurately assess and further improve the performance of the algorithms. The existing nuclei segmentation benchmarks either worked on 2D only or annotated a small number of 3D cells, perhaps due to the high cost of 3D annotation for large-scale data. To fulfill the critical need, we constructed NIS3D, a 3D, high cell density, large-volume, and completely annotated Nuclei Image Segmentation benchmark, assisted by our newly designed semi-automatic annotation software. NIS3D provides more than 22,000 cells across multiple most-used species in this area. Each cell is labeled by three independent annotators, so we can measure the variability of each annotation. A confidence score is computed for each cell, allowing more nuanced testing and performance comparison. A comprehensive review on the methods of segmenting 3D dense nuclei was conducted. The benchmark was used to evaluate the performance of several selected state-of-the-art segmentation algorithms. The best of current methods is still far away from human-level accuracy, corroborating the necessity of generating such a benchmark. The testing results also demonstrated the strength and weakness of each method and pointed out the directions of further methodological development. The dataset can be downloaded here: https://github.com/yu-lab-vt/NIS3D.</details>

### Natural Actor-Critic for Robust Reinforcement Learning with Function Approximation

**Authors:** Ruida Zhou, Tao Liu, Min Cheng, Dileep Kalathil, P. R. Kumar, Chao Tian

**<details><summary>Abstract**</summary>

We study robust reinforcement learning (RL) with the goal of determining a well-performing policy that is robust against model mismatch between the training simulator and the testing environment. Previous policy-based robust RL algorithms mainly focus on the tabular setting under uncertainty sets that facilitate robust policy evaluation, but are no longer tractable when the number of states scales up. To this end, we propose two novel uncertainty set formulations, one based on double sampling and the other on an integral probability metric. Both make large-scale robust RL tractable even when one only has access to a simulator. We propose a robust natural actor-critic (RNAC) approach that incorporates the new uncertainty sets and employs function approximation. We provide finite-time convergence guarantees for the proposed RNAC algorithm to the optimal robust policy within the function approximation error. Finally, we demonstrate the robust performance of the policy learned by our proposed RNAC approach in multiple  MuJoCo environments and a real-world TurtleBot navigation task.</details>

### NetHack is Hard to Hack

**Authors:** Ulyana Piterbarg, Lerrel Pinto, Rob Fergus

**<details><summary>Abstract**</summary>

Neural policy learning methods have achieved remarkable results in various control problems, ranging from Atari games to simulated locomotion. However, these methods struggle in long-horizon tasks, especially in open-ended environments with multi-modal observations, such as the popular dungeon-crawler game, NetHack. Intriguingly, the NeurIPS 2021 NetHack Challenge revealed that symbolic agents outperformed neural approaches by over four times in median game score. In this paper, we delve into the reasons behind this performance gap and present an extensive study on neural policy learning for NetHack. To conduct this study, we analyze the winning symbolic agent, extending its codebase to track internal strategy selection in order to generate one of the largest available demonstration datasets. Utilizing this dataset, we examine (i) the advantages of an action hierarchy; (ii) enhancements in neural architecture; and (iii) the integration of reinforcement learning with imitation learning. Our investigations produce a state-of-the-art neural agent that surpasses previous fully neural policies by 127% in offline settings and 25% in online settings on median game score. However, we also demonstrate that mere scaling is insufficient to bridge the performance gap with the best symbolic models or even the top human players.</details>

### Network Regression with Graph Laplacians

**Authors:** Yidong Zhou, Hans-Georg Müller

**<details><summary>Abstract**</summary>

Network data are increasingly available in various research fields, motivating statistical analysis for populations of networks, where a network as a whole is viewed as a data point. The study of how a network changes as a function of covariates is often of paramount interest. However, due to the non-Euclidean nature of networks, basic statistical tools available for scalar and vector data are no longer applicable. This motivates an extension of the notion of regression to the case where responses are network data. Here we propose to adopt conditional Fréchet means implemented as M-estimators that depend on weights derived from both global and local least squares regression, extending the Fréchet regression framework to networks that are quantified by their graph Laplacians. The challenge is to characterize the space of graph Laplacians to justify the application of Fréchet regression. This characterization then leads to asymptotic rates of convergence for the corresponding M-estimators by applying empirical process methods. We demonstrate the usefulness and good practical performance of the proposed framework with simulations and with network data arising from resting-state fMRI in neuroimaging, as well as New York taxi records.</details>

### Neural Algorithmic Reasoning Without Intermediate Supervision

**Authors:** Gleb Rodionov, Liudmila Prokhorenkova

**<details><summary>Abstract**</summary>

Neural algorithmic reasoning is an emerging area of machine learning focusing on building models that can imitate the execution of classic algorithms, such as sorting, shortest paths, etc. One of the main challenges is to learn algorithms that are able to generalize to out-of-distribution data, in particular with significantly larger input sizes. Recent work on this problem has demonstrated the advantages of learning algorithms step-by-step, giving models access to all intermediate steps of the original algorithm. In this work, we instead focus on learning neural algorithmic reasoning only from the input-output pairs without appealing to the intermediate supervision. We propose simple but effective architectural improvements and also build a self-supervised objective that can regularise intermediate computations of the model without access to the algorithm trajectory. We demonstrate that our approach is competitive to its trajectory-supervised counterpart on tasks from the CLRS Algorithmic Reasoning Benchmark and achieves new state-of-the-art results for several problems, including sorting, where we obtain significant improvements. Thus, learning without intermediate supervision is a promising direction for further research on neural reasoners.</details>

### Neural Ideal Large Eddy Simulation: Modeling Turbulence with Neural Stochastic Differential Equations

**Authors:** Anudhyan Boral, Zhong Yi Wan, Leonardo Zepeda-Núñez, James Lottes, Qing Wang, Yi-Fan Chen, John Anderson, Fei Sha

**<details><summary>Abstract**</summary>

We introduce a data-driven learning framework that assimilates two powerful ideas: ideal large eddy simulation (LES) from turbulence closure modeling and neural stochastic differential equations (SDE) for stochastic modeling. The ideal LES models the LES flow by treating each full-order trajectory as a random realization of the underlying dynamics, as such, the effect of small-scales is marginalized to obtain the deterministic evolution of the LES state. However, ideal LES is analytically intractable. In our work, we use a latent neural SDE to model the evolution of the stochastic process and an encoder-decoder pair for transforming between the latent space and the desired ideal flow field. This stands in sharp contrast to other types of neural parameterization of closure models where each trajectory is treated as a deterministic realization of the dynamics. We show the effectiveness of our approach (niLES – neural ideal LES) on two challenging chaotic dynamical systems: Kolmogorov flow at a Reynolds number of 20,000 and flow past a cylinder at Reynolds number 500. Compared to competing methods, our method can handle non-uniform geometries using unstructured meshes seamlessly. In particular, niLES leads to trajectories with more accurate statistics and enhances stability, particularly for long-horizon rollouts. (Source codes and datasets will be made publicly available.)</details>

### Neural MMO 2.0: A Massively Multi-task Addition to Massively Multi-agent Learning

**Authors:** Joseph Suarez, David Bloomin, Kyoung Whan Choe, Hao Xiang Li, Ryan Sullivan, Nishaanth Kanna, Daniel Scott, Rose Shuman, Herbie Bradley, Louis Castricato, Phillip Isola, Chenghui Yu, Yuhao Jiang, Qimai Li, Jiaxin Chen, Xiaolong Zhu

**<details><summary>Abstract**</summary>

Neural MMO 2.0 is a massively multi-agent and multi-task environment for reinforcement learning research. This version features a novel task-system that broadens the range of training settings and poses a new challenge in generalization: evaluation on and against tasks, maps, and opponents never seen during training. Maps are procedurally generated with 128 agents in the standard setting and 1-1024 supported overall. Version 2.0 is a complete rewrite of its predecessor with three-fold improved performance, effectively addressing simulation bottlenecks in online training. Enhancements to compatibility enable training with standard reinforcement learning frameworks designed for much simpler environments. Neural MMO 2.0 is free and open-source with comprehensive documentation available at neuralmmo.github.io and an active community Discord. To spark initial research on this new platform, we are concurrently running a competition at NeurIPS 2023.</details>

### NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics

**Authors:** Anwar Said, Roza Bayrak, Tyler Derr, Mudassir Shabbir, Daniel Moyer, Catie Chang, Xenofon Koutsoukos

**<details><summary>Abstract**</summary>

Machine learning provides a valuable tool for analyzing high-dimensional functional neuroimaging data, and is proving effective in predicting various neurological conditions, psychiatric disorders, and cognitive patterns. In functional magnetic resonance imaging (MRI) research, interactions between brain regions are commonly modeled using graph-based representations. The potency of graph machine learning methods has been established across myriad domains, marking a transformative step in data interpretation and predictive modeling. Yet, despite their promise, the transposition of these techniques to the neuroimaging domain has been  challenging due to the expansive number of potential preprocessing pipelines and the large parameter search space for graph-based dataset construction. In this paper, we introduce NeuroGraph, a collection of graph-based neuroimaging datasets, and demonstrated its utility for predicting multiple categories of behavioral and cognitive traits. We delve deeply into the dataset generation search space by crafting 35 datasets that encompass static and dynamic brain connectivity, running in excess of 15 baseline methods for benchmarking. Additionally, we provide generic frameworks for learning on both static and dynamic graphs. Our extensive experiments lead to several key observations. Notably, using correlation vectors as node features, incorporating larger number of regions of interest, and employing sparser graphs lead to improved performance. To foster further advancements in graph-based data driven neuroimaging analysis, we offer a comprehensive open-source Python package that includes the benchmark datasets, baseline implementations, model training, and standard evaluation.</details>

### New Complexity-Theoretic Frontiers of Tractability for Neural Network Training

**Authors:** Cornelius Brand, Robert Ganian, Mathis Rocton

**<details><summary>Abstract**</summary>

In spite of the fundamental role of neural networks in contemporary machine learning research, our understanding of the computational complexity of optimally training neural networks remains limited even when dealing with the simplest kinds of activation functions. Indeed, while there has been a number of very recent results that establish ever-tighter lower bounds for the problem under linear and ReLU activation functions, little progress has been made towards the identification of novel polynomial-time tractable network architectures. In this article we obtain novel algorithmic upper bounds for training linear- and ReLU-activated neural networks to optimality which push the boundaries of tractability for these problems beyond the previous state of the art.</details>

### No Train No Gain: Revisiting Efficient Training Algorithms For Transformer-based Language Models

**Authors:** Jean Kaddour, Oscar Key, Piotr Nawrot, Pasquale Minervini, Matt Kusner

**<details><summary>Abstract**</summary>

The computation necessary for training Transformer-based language models has skyrocketed in recent years.This trend has motivated research on efficient training algorithms designed to improve training, validation, and downstream performance faster than standard training. In this work, we revisit three categories of such algorithms: dynamic architectures (layer stacking, layer dropping), batch selection (selective backprop., RHO-loss), and efficient optimizers (Lion, Sophia). When pre-training BERT and T5 with a fixed computation budget using such methods, we find that their training, validation, and downstream gains vanish compared to a baseline with a fully-decayed learning rate. We define an evaluation protocol that enables computation to be done on arbitrary machines by mapping all computation time to a reference machine which we call reference system time. We discuss the limitations of our proposed protocol and release our code to encourage rigorous research in efficient training procedures: https://github.com/JeanKaddour/NoTrainNoGain.</details>

### Non-Stationary Bandits with Auto-Regressive Temporal Dependency

**Authors:** Qinyi Chen, Negin Golrezaei, Djallel Bouneffouf

**<details><summary>Abstract**</summary>

Traditional multi-armed bandit (MAB) frameworks, predominantly examined under stochastic or adversarial settings, often overlook the temporal dynamics inherent in many real-world applications such as recommendation systems and online advertising. This paper introduces a novel non-stationary MAB framework that captures the temporal structure of these real-world dynamics through an auto-regressive (AR) reward structure. We propose an algorithm that integrates two key mechanisms: (i) an alternation mechanism adept at leveraging temporal dependencies to dynamically balance exploration and exploitation, and (ii) a restarting mechanism designed to discard out-of-date information. Our algorithm achieves a regret upper bound that nearly matches the lower bound, with regret measured against a robust dynamic benchmark. Finally, via a real-world case study on tourism demand prediction, we demonstrate both the efficacy of our algorithm and the broader applicability of our techniques to more complex, rapidly evolving time series.</details>

### Not All Neuro-Symbolic Concepts Are Created Equal: Analysis and Mitigation of Reasoning Shortcuts

**Authors:** Emanuele Marconato, Stefano Teso, Antonio Vergari, Andrea Passerini

**<details><summary>Abstract**</summary>

Neuro-Symbolic (NeSy) predictive models hold the promise of improved compliance with given constraints, systematic generalization, and interpretability, as they allow to infer labels that are consistent with some prior knowledge by reasoning over high-level concepts extracted from sub-symbolic inputs. It was recently shown that NeSy predictors are affected by *reasoning shortcuts*: they can attain high accuracy but by leveraging concepts with \textit{unintended semantics}, thus coming short of their promised advantages. Yet, a systematic characterization of reasoning shortcuts and of potential mitigation strategies is missing. This work fills this gap by characterizing them as unintended optima of the learning objective and identifying four key conditions behind their occurrence. Based on this, we derive several natural mitigation strategies, and analyze their efficacy both theoretically and empirically. Our analysis shows reasoning shortcuts are difficult to deal with, casting doubts on the trustworthiness and interpretability of existing NeSy solutions.</details>

### OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents

**Authors:** Hugo Laurençon, Lucile Saulnier, Leo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, Matthieu Cord, Victor Sanh

**<details><summary>Abstract**</summary>

Large multimodal models trained on natural documents, which interleave images and text, outperform models trained on image-text pairs on various multimodal benchmarks. However, the datasets used to train these models have not been released, and the collection process has not been fully specified.  We introduce the OBELICS dataset, an open web-scale filtered dataset of interleaved image-text documents comprising 141 million web pages extracted from Common Crawl, 353 million associated images, and 115 billion text tokens. We describe the dataset creation process, present comprehensive filtering rules, and provide an analysis of the dataset's content. To show the viability of OBELICS, we train on the dataset vision and language models of 9 and 80 billion parameters, IDEFICS-9B and IDEFICS, and obtain competitive performance on different multimodal benchmarks. We release our dataset, models and code.</details>

### OBJECT 3DIT: Language-guided 3D-aware Image Editing

**Authors:** Oscar Michel, Anand Bhattad, Eli VanderBilt, Ranjay Krishna, Aniruddha Kembhavi, Tanmay Gupta

**<details><summary>Abstract**</summary>

Existing image editing tools, while powerful, typically disregard the underlying 3D geometry from which the image is projected. As a result, edits made using these tools may become detached from the geometry and lighting conditions that are at the foundation of the image formation process; such edits break the portrayal of a coherent 3D world. 3D-aware generative models are a promising solution, but currently only succeed on small datasets or at the level of a single object. In this work, we formulate the new task of language-guided 3D-aware editing, where objects in an image should be edited according to a language instruction while remaining consistent with the underlying 3D scene. To promote progress towards this goal, we release OBJect: a benchmark dataset of 400K editing examples created from procedurally generated 3D scenes. Each example consists of an input image, editing instruction in language, and the edited image. We also introduce 3DIT: single and multi-task models for four editing tasks. Our models show impressive abilities to understand the 3D composition of entire scenes, factoring in surrounding objects, surfaces, lighting conditions, shadows, and physically-plausible object configurations. Surprisingly, training on only synthetic scenes from \dataset, editing capabilities of 3DIT generalize to real-world images.</details>

### OFCOURSE: A Multi-Agent Reinforcement Learning Environment for Order Fulfillment

**Authors:** Yiheng Zhu, Yang Zhan, Xuankun Huang, Yuwei Chen, yujie Chen, Jiangwen Wei, Wei Feng, Yinzhi Zhou, Haoyuan Hu, Jieping Ye

**<details><summary>Abstract**</summary>

The dramatic growth of global e-commerce has led to a surge in demand for efficient and cost-effective order fulfillment which can increase customers' service levels and sellers' competitiveness. However, managing order fulfillment is challenging due to a series of interdependent online sequential decision-making problems. To clear this hurdle, rather than solving the problems separately as attempted in some recent researches, this paper proposes a method based on multi-agent reinforcement learning to integratively solve the series of interconnected problems, encompassing order handling, packing and pickup, storage, order consolidation, and last-mile delivery. In particular, we model the integrated problem as a Markov game, wherein a team of agents learns a joint policy via interacting with a simulated environment. Since no simulated environment supporting the complete order fulfillment problem exists, we devise Order Fulfillment COoperative mUlti-agent Reinforcement learning Scalable Environment (OFCOURSE) in the OpenAI Gym style, which allows reproduction and re-utilization to build customized applications. By constructing the fulfillment system in OFCOURSE, we optimize a joint policy that solves the integrated problem, facilitating sequential order-wise operations across all fulfillment units and minimizing the total cost of fulfilling all orders within the promised time. With OFCOURSE, we also demonstrate that the joint policy learned by multi-agent reinforcement learning outperforms the combination of locally optimal policies. The source code of OFCOURSE is available at: https://github.com/GitYiheng/ofcourse.</details>

### OV-PARTS: Towards Open-Vocabulary Part Segmentation

**Authors:** Meng Wei, Xiaoyu Yue, Wenwei Zhang, Shu Kong, Xihui Liu, Jiangmiao Pang

**<details><summary>Abstract**</summary>

Segmenting and recognizing diverse object parts is a crucial ability in applications spanning various computer vision and robotic tasks. While significant progress has been made in object-level Open-Vocabulary Semantic Segmentation (OVSS), i.e., segmenting objects with arbitrary text, the corresponding part-level research poses additional challenges. Firstly, part segmentation inherently involves intricate boundaries, while limited annotated data compounds the challenge. Secondly, part segmentation introduces an open granularity challenge due to the diverse and often ambiguous definitions of parts in the open world. Furthermore, the large-scale vision and language models, which play a key role in the open vocabulary setting, struggle to recognize parts as effectively as objects. To comprehensively investigate and tackle these challenges, we propose an Open-Vocabulary Part Segmentation (OV-PARTS) benchmark. OV-PARTS includes refined versions of two publicly available datasets: Pascal-Part-116 and ADE20K-Part-234. And it covers three specific tasks: Generalized Zero-Shot Part Segmentation, Cross-Dataset Part Segmentation, and Few-Shot Part Segmentation, providing insights into analogical reasoning, open granularity and few-shot adapting abilities of models. Moreover, we analyze and adapt two prevailing paradigms of existing object-level OVSS methods for OV-PARTS. Extensive experimental analysis is conducted to inspire future research in leveraging foundational models for OV-PARTS. The code and dataset are available at https://github.com/kellyiss/OV_PARTS.</details>

### Objaverse-XL: A Universe of 10M+ 3D Objects

**Authors:** Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, Ali Farhadi

**<details><summary>Abstract**</summary>

Natural language processing and 2D vision models have attained remarkable proficiency on many tasks primarily by escalating the scale of training data. However, 3D vision tasks have not seen the same progress, in part due to the challenges of acquiring high-quality 3D data. In this work, we present Objaverse-XL, a dataset of over 10 million 3D objects. Our compilation comprises deduplicated 3D objects from a diverse set of sources, including manually designed objects, photogrammetry scans of landmarks and everyday items, and professional scans of historic and antique artifacts. Representing the largest scale and diversity in the realm of 3D datasets, Objaverse-XL enables significant new possibilities for 3D vision. Our experiments demonstrate the vast improvements enabled with the scale provided by Objaverse-XL. We show that by training Zero123 on novel view synthesis, utilizing over 100 million multi-view rendered images, we achieve strong zero-shot generalization abilities. We hope that releasing Objaverse-XL will enable further innovations in the field of 3D vision at scale.</details>

### Object Reprojection Error (ORE): Camera pose benchmarks from lightweight tracking annotations

**Authors:** Xingyu Chen, Weiyao Wang, Hao Tang, Matt Feiszli

**<details><summary>Abstract**</summary>

3D spatial understanding is highly valuable in the context of semantic modeling of environments, agents, and their relationships.  Semantic modeling approaches employed on monocular video often ingest outputs from off-the-shelf SLAM/SfM pipelines, which are anecdotally observed to perform poorly or fail completely on some fraction of the videos of interest.  These target videos may vary widely in complexity of scenes, activities, camera trajectory, etc.  Unfortunately, such semantically-rich video data often comes with no ground-truth 3D information, and in practice it is prohibitively costly or impossible to obtain ground truth reconstructions or camera pose post-hoc.  This paper proposes a novel evaluation protocol, Object Reprojection Error (ORE) to benchmark camera trajectories; ORE computes reprojection error for static objects within the video and requires only lightweight object tracklet annotations.  These annotations are easy to gather on new or existing video, enabling ORE to be calculated on essentially arbitrary datasets.  We show that ORE maintains high rank correlation with standard metrics based on groundtruth.  Leveraging ORE, we source videos and annotations from Ego4D-EgoTracks, resulting in EgoStatic, a large-scale diverse dataset for evaluating camera trajectories in-the-wild.</details>

### Object-Centric Learning for Real-World Videos by Predicting Temporal Feature Similarities

**Authors:** Andrii Zadaianchuk, Maximilian Seitzer, Georg Martius

**<details><summary>Abstract**</summary>

Unsupervised video-based object-centric learning is a promising avenue to learn structured representations from large, unlabeled video collections, but previous approaches have only managed to scale to real-world datasets in restricted domains.Recently, it was shown that the reconstruction of pre-trained self-supervised features leads to object-centric representations on unconstrained real-world image datasets.Building on this approach, we propose a novel way to use such pre-trained features in the form of a temporal feature similarity loss.This loss encodes semantic and temporal correlations between image patches and is a natural way to introduce a motion bias for object discovery.We demonstrate that this loss leads to state-of-the-art performance on the challenging synthetic MOVi datasets.When used in combination with the feature reconstruction loss, our model is the first object-centric video model that scales to unconstrained video datasets such as YouTube-VIS.https://martius-lab.github.io/videosaur/</details>

### Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving

**Authors:** Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, Hang Zhao

**<details><summary>Abstract**</summary>

Robotic perception requires the modeling of both 3D geometry and semantics. Existing methods typically focus on estimating 3D bounding boxes, neglecting finer geometric details and struggling to handle general, out-of-vocabulary objects. 3D occupancy prediction, which estimates the detailed occupancy states and semantics of a scene, is an emerging task to overcome these limitations.To support 3D occupancy prediction, we develop a label generation pipeline that produces dense, visibility-aware labels for any given scene. This pipeline comprises three stages: voxel densification, occlusion reasoning, and image-guided voxel refinement. We establish two benchmarks, derived from the Waymo Open Dataset and the nuScenes Dataset, namely Occ3D-Waymo and Occ3D-nuScenes benchmarks. Furthermore, we provide an extensive analysis of the proposed dataset with various baseline models. Lastly, we propose a new model, dubbed Coarse-to-Fine Occupancy (CTF-Occ) network, which demonstrates superior performance on the Occ3D benchmarks.The code, data, and benchmarks are released at \url{https://tsinghua-mars-lab.github.io/Occ3D/}.</details>

### Offline RL with Discrete Proxy Representations for Generalizability in POMDPs

**Authors:** Pengjie Gu, Xinyu Cai, Dong Xing, Xinrun Wang, Mengchen Zhao, Bo An

**<details><summary>Abstract**</summary>

Offline Reinforcement Learning (RL) has demonstrated promising results in various applications by learning policies from previously collected datasets, reducing the need for online exploration and interactions. However, real-world scenarios usually involve partial observability, which brings crucial challenges of the deployment of offline RL methods: i) the policy trained on data with full observability is not robust against the masked observations during execution, and ii) the information of which parts of observations are masked is usually unknown during training. In order to address these challenges, we present Offline RL with DiscrEte pRoxy representations (ORDER), a probabilistic framework which leverages novel state representations to improve the robustness against diverse masked observabilities. Specifically, we propose a discrete representation of the states and use a proxy representation to recover the states from masked partial observable trajectories. The training of ORDER can be compactly described as the following three steps. i) Learning the discrete state representations on data with full observations, ii) Training the decision module based on the discrete representations, and iii) Training the proxy discrete representations on the data with various partial observations, aligning with the discrete representations. We conduct extensive experiments to evaluate ORDER, showcasing its effectiveness in offline RL for diverse partially observable scenarios and highlighting the significance of discrete proxy representations in  generalization performance.ORDER is a flexible framework to employ any offline RL algorithms and we hope that ORDER can pave the way for the deployment of RL policy against various partial observabilities in the real world.</details>

### On Class Distributions Induced by Nearest Neighbor Graphs for Node Classification of Tabular Data

**Authors:** Federico Errica

**<details><summary>Abstract**</summary>

Researchers have used nearest neighbor graphs to transform classical machine learning problems on tabular data into node classification tasks to solve with graph representation learning methods. Such artificial structures often reflect the homophily assumption, believed to be a key factor in the performances of deep graph networks. In light of recent results demystifying these beliefs, we introduce a theoretical framework to understand the benefits of Nearest Neighbor (NN) graphs when a graph structure is missing. We formally analyze the Cross-Class Neighborhood Similarity (CCNS), used to empirically evaluate the usefulness of structures, in the context of nearest neighbor graphs. Moreover, we study the class separability induced by deep graph networks on a k-NN graph. Motivated by the theory, our quantitative experiments demonstrate that, under full supervision, employing a k-NN graph offers no benefits compared to a structure-agnostic baseline. Qualitative analyses suggest that our framework is good at estimating the CCNS and hint at k-NN graphs never being useful for such classification tasks under full supervision, thus advocating for the study of alternative graph construction techniques in combination with deep graph networks.</details>

### On Evaluating Adversarial Robustness of Large Vision-Language Models

**Authors:** Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Chongxuan LI, Ngai-Man (Man) Cheung, Min Lin

**<details><summary>Abstract**</summary>

Large vision-language models (VLMs) such as GPT-4 have achieved unprecedented performance in response generation, especially with visual inputs, enabling more creative and adaptable interaction than large language models such as ChatGPT. Nonetheless, multimodal generation exacerbates safety concerns, since adversaries may successfully evade the entire system by subtly manipulating the most vulnerable modality (e.g., vision). To this end, we propose evaluating the robustness of open-source large VLMs in the most realistic and high-risk setting, where adversaries have only black-box system access and seek to deceive the model into returning the targeted responses. In particular, we first craft targeted adversarial examples against pretrained models such as CLIP and BLIP, and then transfer these adversarial examples to other VLMs such as MiniGPT-4, LLaVA, UniDiffuser, BLIP-2, and Img2Prompt. In addition, we observe that black-box queries on these VLMs can further improve the effectiveness of targeted evasion, resulting in a surprisingly high success rate for generating targeted responses. Our findings provide a quantitative understanding regarding the adversarial vulnerability of large VLMs and call for a more thorough examination of their potential security flaws before deployment in practice. Our project page: https://yunqing-me.github.io/AttackVLM/.</details>

### On Sample-Efficient Offline Reinforcement Learning: Data Diversity, Posterior Sampling and Beyond

**Authors:** Thanh Nguyen-Tang, Raman Arora

**<details><summary>Abstract**</summary>

We seek to understand what facilitates sample-efficient learning from historical datasets for sequential decision-making, a problem that is popularly known as offline reinforcement learning (RL). Further, we are interested in algorithms that enjoy sample efficiency while leveraging (value) function approximation. In this paper, we address these fundamental questions by (i) proposing a notion of data diversity that subsumes the previous notions of coverage measures in offline RL and (ii) using this notion to \emph{unify} three distinct classes of offline RL algorithms based on version spaces (VS), regularized optimization (RO), and posterior sampling (PS). We establish that VS-based, RO-based, and PS-based algorithms, under standard assumptions, achieve \emph{comparable} sample efficiency, which recovers the state-of-the-art sub-optimality bounds for finite and linear model classes with the standard assumptions. This result is surprising, given that the prior work suggested an unfavorable sample complexity of the RO-based algorithm compared to the VS-based algorithm, whereas posterior sampling is rarely considered in offline RL due to its explorative nature. Notably, our proposed model-free PS-based algorithm for offline RL is \emph{novel}, with sub-optimality bounds that are \emph{frequentist} (i.e., worst-case) in nature.</details>

### On the Importance of Exploration for Generalization in Reinforcement Learning

**Authors:** Yiding Jiang, J. Zico Kolter, Roberta Raileanu

**<details><summary>Abstract**</summary>

Existing approaches for improving generalization in deep reinforcement learning (RL) have mostly focused on representation learning, neglecting RL-specific aspects such as exploration. We hypothesize that the agent's exploration strategy plays a key role in its ability to generalize to new environments.Through a series of experiments in a tabular contextual MDP, we show that exploration is helpful not only for efficiently finding the optimal policy for the training environments but also for acquiring knowledge that helps decision making in unseen environments. Based on these observations, we propose EDE: Exploration via Distributional Ensemble, a method that encourages the exploration of states with high epistemic uncertainty through an ensemble of Q-value distributions. The proposed algorithm is the first value-based approach to achieve strong performance on both Procgen and Crafter, two benchmarks for generalization in RL with high-dimensional observations. The open-sourced implementation can be found at https://github.com/facebookresearch/ede.</details>

### On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets

**Authors:** Jiashuo Liu, Tianyu Wang, Peng Cui, Hongseok Namkoong

**<details><summary>Abstract**</summary>

Different distribution shifts require different algorithmic and operational  interventions. Methodological research must be grounded by the specific  shifts they address.  Although nascent benchmarks provide a promising  empirical foundation, they \emph{implicitly} focus on covariate  shifts, and the validity of empirical findings depends on the type of shift,   e.g., previous observations on algorithmic performance can fail to be valid when  the $Y|X$ distribution changes.  We conduct a thorough investigation of  natural shifts in 5 tabular datasets over 86,000 model configurations, and  find that $Y|X$-shifts are most prevalent.  To encourage researchers to  develop a refined language for distribution shifts, we build ``WhyShift``, an empirical testbed of curated real-world shifts where  we characterize the type of shift we benchmark performance over.  Since  $Y|X$-shifts are prevalent in tabular settings, we \emph{identify covariate  regions} that suffer the biggest $Y|X$-shifts and discuss implications for  algorithmic and data-based interventions.  Our testbed highlights the  importance of future research that builds an understanding of why  distributions differ.</details>

### On the Overlooked Pitfalls of Weight Decay and How to Mitigate Them: A Gradient-Norm Perspective

**Authors:** Zeke Xie, Zhiqiang Xu, Jingzhao Zhang, Issei Sato, Masashi Sugiyama

**<details><summary>Abstract**</summary>

Weight decay is a simple yet powerful regularization technique that has been very widely used in training of deep neural networks (DNNs). While weight decay has attracted much attention, previous studies fail to discover some overlooked pitfalls on large gradient norms resulted by weight decay. In this paper, we discover that, weight decay can unfortunately lead to large gradient norms at the final phase (or the terminated solution) of training, which often indicates bad convergence and poor generalization. To mitigate the gradient-norm-centered pitfalls, we present the first practical scheduler for weight decay, called the Scheduled Weight Decay (SWD) method that can dynamically adjust the weight decay strength according to the gradient norm and significantly penalize large gradient norms during training. Our experiments also support that SWD indeed mitigates large gradient norms and often significantly outperforms the conventional constant weight decay strategy for Adaptive Moment Estimation (Adam).</details>

### One-Line-of-Code Data Mollification Improves Optimization of Likelihood-based Generative Models

**Authors:** Ba-Hien Tran, Giulio Franzese, Pietro Michiardi, Maurizio Filippone

**<details><summary>Abstract**</summary>

Generative Models (GMs) have attracted considerable attention due to their tremendous success in various domains, such as computer vision where they are capable to generate impressive realistic-looking images. Likelihood-based GMs are attractive due to the possibility to generate new data by a single model evaluation. However, they typically achieve lower sample quality compared to state-of-the-art score-based Diffusion Models (DMs). This paper provides a significant step in the direction of addressing this limitation. The idea is to borrow one of the strengths of score-based DMs, which is the ability to perform accurate density estimation in low-density regions and to address manifold overfitting by means of data mollification. We propose a view of data mollification within likelihood-based GMs as a continuation method, whereby the optimization objective smoothly transitions from simple-to-optimize to the original target. Crucially, data mollification can be implemented by adding one line of code in the optimization loop, and we demonstrate that this provides a boost in generation quality of likelihood-based GMs, without computational overheads. We report results on real-world image data sets and UCI benchmarks with popular likelihood-based GMs, including variants of variational autoencoders and normalizing flows, showing large improvements in FID score and density estimation.</details>

### [Spotlight] Online Constrained Meta-Learning: Provable Guarantees for Generalization

**Authors:** Siyuan Xu, Minghui Zhu

**<details><summary>Abstract**</summary>

Meta-learning has attracted attention due to its strong ability to learn experiences from known tasks, which can speed up and enhance the learning process for new tasks. However, most existing meta-learning approaches only can learn from tasks without any constraint. This paper proposes an online constrained meta-learning framework, which continuously learns meta-knowledge from sequential learning tasks, and the learning tasks are subject to hard constraints. Beyond existing meta-learning analyses, we provide the upper bounds of optimality gaps and constraint violations produced by the proposed framework, which considers the dynamic regret of online learning, as well as the generalization ability of the task-specific models. Moreover, we provide a practical algorithm for the framework, and validate its superior effectiveness through experiments conducted on meta-imitation learning and few-shot image classification.</details>

### Online Corrupted User Detection and Regret Minimization

**Authors:** Zhiyong Wang, Jize Xie, Tong Yu, Shuai Li, John C.S. Lui

**<details><summary>Abstract**</summary>

In real-world online web systems, multiple users usually arrive sequentially into the system. For applications like click fraud and fake reviews, some users can maliciously perform corrupted (disrupted) behaviors to trick the system. Therefore, it is crucial to design efficient online learning algorithms to robustly learn from potentially corrupted user behaviors and accurately identify the corrupted users in an online manner.  Existing works propose bandit algorithms robust to adversarial corruption. However, these algorithms are designed for a single user, and cannot leverage the implicit social relations among multiple users for more efficient learning. Moreover, none of them consider how to detect corrupted users online in the multiple-user scenario. In this paper, we present an important online learning problem named LOCUD to learn and utilize unknown user relations from disrupted behaviors to speed up learning, and identify the corrupted users in an online setting. To robustly learn and utilize the unknown relations among potentially corrupted users, we propose a novel bandit algorithm RCLUB-WCU. To detect the corrupted users, we devise a novel online detection algorithm OCCUD based on RCLUB-WCU's inferred user relations. We prove a regret upper bound for RCLUB-WCU, which asymptotically matches the lower bound with respect to $T$ up to logarithmic factors, and matches the state-of-the-art results in degenerate cases. We also give a theoretical guarantee for the detection accuracy of OCCUD. With extensive experiments, our methods achieve superior performance over previous bandit algorithms and high corrupted user detection accuracy.</details>

### [Spotlight] Online List Labeling with Predictions

**Authors:** Samuel McCauley, Ben Moseley, Aidin Niaparast, Shikha Singh

**<details><summary>Abstract**</summary>

A growing line of work shows how learned predictions can be used to break through worst-case barriers to improve the running time of an algorithm. However, incorporating predictions into data structures with strong theoretical guarantees remains underdeveloped.  This paper takes a step in this direction by showing that predictions can be leveraged in the fundamental online list labeling problem. In the problem, $n$ items arrive over time and must be stored in sorted order in an array of size $\Theta(n)$.  The array slot of an element is its label and the goal is to maintain sorted order while minimizing the total number of elements moved (i.e., relabeled). We design a new list labeling data structure and bound its performance in two models.  In the worst-case learning-augmented model, we give guarantees in terms of the error in the predictions.  Our data structure provides strong guarantees: it is optimal for any prediction error and guarantees the best-known worst-case bound even when the predictions are entirely erroneous. We also consider a stochastic error model and bound the performance in terms of the expectation and variance of the error. Finally, the theoretical results are demonstrated empirically.  In particular, we show that our data structure has strong performance on real temporal data sets where predictions are constructed from elements that arrived in the past, as is typically done in a practical use case.</details>

### Online robust non-stationary estimation

**Authors:** Abishek Sankararaman, Balakrishnan Narayanaswamy

**<details><summary>Abstract**</summary>

The real-time estimation of time-varying parameters from high-dimensional, heavy-tailed and corrupted data-streams is a common sub-routine in systems ranging from those for network monitoring and anomaly detection to those for traffic scheduling in data-centers. For estimation tasks that can be cast as minimizing a strongly convex loss function, we prove that an appropriately tuned version of the {\ttfamily clipped Stochastic Gradient Descent} (SGD) is simultaneously {\em(i)} adaptive to drift, {\em (ii)} robust to heavy-tailed inliers and arbitrary corruptions,  {\em(iii)} requires no distributional knowledge and {\em (iv)} can be implemented in an online streaming fashion. All prior estimation algorithms have only been proven to posses a subset of these practical desiderata. A observation we make is that, neither the $\mathcal{O}\left(\frac{1}{t}\right)$ learning rate for {\ttfamily clipped SGD} known to be optimal for strongly convex loss functions of a \emph{stationary} data-stream, nor the $\mathcal{O}(1)$ learning rate known to be optimal for being adaptive to drift in a \emph{noiseless} environment can be used. Instead, a learning rate of $T^{-\alpha}$ for $ \alpha < 1$ where $T$ is the stream-length is needed to balance adaptivity to potential drift and to combat noise. We develop a new inductive argument and combine it with a martingale concentration result to derive high-probability under \emph{any learning rate} on data-streams exhibiting \emph{arbitrary distribution shift} - a proof strategy that may be of independent interest. Further, using the classical doubling-trick, we relax the knowledge of the stream length $T$. Ours is the first online estimation algorithm that is provably robust to heavy-tails, corruptions and distribution shift simultaneously. We complement our theoretical results empirically on synthetic and real data.</details>

### OpenAGI: When LLM Meets Domain Experts

**Authors:** Yingqiang Ge, Wenyue Hua, Kai Mei, jianchao ji, Juntao Tan, Shuyuan Xu, Zelong Li, Yongfeng Zhang

**<details><summary>Abstract**</summary>

Human Intelligence (HI) excels at combining basic skills to solve complex tasks. This capability is vital for Artificial Intelligence (AI) and should be embedded in comprehensive AI Agents, enabling them to harness expert models for complex task-solving towards Artificial General Intelligence (AGI). Large Language Models (LLMs) show promising learning and reasoning abilities, and can effectively use external models, tools, plugins, or APIs to tackle complex problems. In this work, we introduce OpenAGI, an open-source AGI research and development platform designed for solving multi-step, real-world tasks. Specifically, OpenAGI uses a dual strategy, integrating standard benchmark tasks for benchmarking and evaluation, and open-ended tasks including more expandable models, tools, plugins, or APIs for creative problem-solving. Tasks are presented as natural language queries to the LLM, which then selects and executes appropriate models. We also propose a Reinforcement Learning from Task Feedback (RLTF) mechanism that uses task results to improve the LLM's task-solving ability, which creates a self-improving AI feedback loop. While we acknowledge that AGI is a broad and multifaceted research challenge with no singularly defined solution path, the integration of LLMs with domain-specific expert models, inspired by mirroring the blend of general and specialized intelligence in humans, offers a promising approach towards AGI. We are open-sourcing the OpenAGI project's code, dataset, benchmarks, evaluation methods, and the UI demo to foster community involvement in AGI advancement: https://github.com/agiresearch/OpenAGI.</details>

### OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping

**Authors:** Huijie Wang, Tianyu Li, Yang Li, Li Chen, Chonghao Sima, Zhenbo Liu, Bangjun Wang, Peijin Jia, Yuting Wang, Shengyin Jiang, Feng Wen, Hang Xu, Ping Luo, Junchi Yan, Wei Zhang, Hongyang Li

**<details><summary>Abstract**</summary>

Accurately depicting the complex traffic scene is a vital component for autonomous vehicles to execute correct judgments. However, existing benchmarks tend to oversimplify the scene by solely focusing on lane perception tasks. Observing that human drivers rely on both lanes and traffic signals to operate their vehicles safely, we present OpenLane-V2, the first dataset on topology reasoning for traffic scene structure. The objective of the presented dataset is to advance research in understanding the structure of road scenes by examining the relationship between perceived entities, such as traffic elements and lanes. Leveraging existing datasets, OpenLane-V2 consists of 2,000 annotated road scenes that describe traffic elements and their correlation to the lanes. It comprises three primary sub-tasks, including the 3D lane detection inherited from OpenLane, accompanied by corresponding metrics to evaluate the model’s performance. We evaluate various state-of-the-art methods, and present their quantitative and qualitative results on OpenLane-V2 to indicate future avenues for investigating topology reasoning in traffic scenes.</details>

### OpenProteinSet: Training data for structural biology at scale

**Authors:** Gustaf Ahdritz, Nazim Bouatta, Sachin Kadyan, Lukas Jarosch, Dan Berenberg, Ian Fisk, Andrew Watkins, Stephen Ra, Richard Bonneau, Mohammed AlQuraishi

**<details><summary>Abstract**</summary>

Multiple sequence alignments (MSAs) of proteins encode rich biological information and have been workhorses in bioinformatic methods for tasks like protein design and protein structure prediction for decades. Recent breakthroughs like AlphaFold2 that use transformers to attend directly over large quantities of raw MSAs have reaffirmed their importance. Generation of MSAs is highly computationally intensive, however, and no datasets comparable to those used to train AlphaFold2 have been made available to the research community, hindering progress in machine learning for proteins. To remedy this problem, we introduce OpenProteinSet, an open-source corpus of more than 16 million MSAs, associated structural homologs from the Protein Data Bank, and AlphaFold2 protein structure predictions. We have previously demonstrated the utility of OpenProteinSet by successfully retraining AlphaFold2 on it. We expect OpenProteinSet to be broadly useful as training and validation data for 1) diverse tasks focused on protein structure, function, and design and 2) large-scale multimodal machine learning research.</details>

### OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning

**Authors:** Cheng Tan, Siyuan Li, Zhangyang Gao, Wenfei Guan, Zedong Wang, Zicheng Liu, Lirong Wu, Stan Z. Li

**<details><summary>Abstract**</summary>

Spatio-temporal predictive learning is a learning paradigm that enables models to learn spatial and temporal patterns by predicting future frames from given past frames in an unsupervised manner. Despite remarkable progress in recent years, a lack of systematic understanding persists due to the diverse settings, complex implementation, and difficult reproducibility. Without standardization, comparisons can be unfair and insights inconclusive. To address this dilemma, we propose OpenSTL, a comprehensive benchmark for spatio-temporal predictive learning that categorizes prevalent approaches into recurrent-based and recurrent-free models. OpenSTL provides a modular and extensible framework implementing various state-of-the-art methods. We conduct standard evaluations on datasets across various domains, including synthetic moving object trajectory, human motion, driving scenes, traffic flow, and weather forecasting. Based on our observations, we provide a detailed analysis of how model architecture and dataset properties affect spatio-temporal predictive learning performance. Surprisingly, we find that recurrent-free models achieve a good balance between efficiency and performance than recurrent models. Thus, we further extend the common MetaFormers to boost recurrent-free spatial-temporal predictive learning. We open-source the code and models at https://github.com/chengtan9907/OpenSTL.</details>

### Operation-Level Early Stopping for Robustifying Differentiable NAS

**Authors:** Shen Jiang, Zipeng Ji, Guanghui Zhu, Chunfeng Yuan, Yihua Huang

**<details><summary>Abstract**</summary>

Differentiable NAS (DARTS) is a simple and efficient neural architecture search method that has been extensively adopted in various machine learning tasks.% Nevertheless, DARTS still encounters several robustness issues, mainly the domination of skip connections.% The resulting architectures are full of parametric-free operations, leading to performance collapse.% Existing methods suggest that the skip connection has additional advantages in optimization compared to other parametric operations and propose to alleviate the domination of skip connections by eliminating these additional advantages.% In this paper, we analyze this issue from a simple and straightforward perspective and propose that the domination of skip connections results from parametric operations overfitting the training data while architecture parameters are trained on the validation data, leading to undesired behaviors.% Based on this observation, we propose the operation-level early stopping (OLES) method to overcome this issue and robustify DARTS without introducing any computation overhead.% Extensive experimental results can verify our hypothesis and the effectiveness of OLES.</details>

### Optimal Algorithms for the Inhomogeneous Spiked Wigner Model

**Authors:** Aleksandr Pak, Justin Ko, Florent Krzakala

**<details><summary>Abstract**</summary>

We study a spiked Wigner problem with an inhomogeneous noise profile. Our aim in this problem is to recover the signal passed through an inhomogeneous low-rank matrix channel. While the information-theoretic performances are well-known, we focus on the algorithmic problem. First, we derive an approximate message-passing algorithm (AMP) for the inhomogeneous problem and show that its rigorous state evolution coincides with the information-theoretic optimal Bayes fixed-point equations. Second, we deduce a simple and efficient spectral method that outperforms PCA and is shown to match the information-theoretic transition.</details>

### Optimal Block-wise Asymmetric Graph Construction for Graph-based Semi-supervised Learning

**Authors:** Zixing Song, Yifei Zhang, Irwin King

**<details><summary>Abstract**</summary>

Graph-based semi-supervised learning (GSSL) serves as a powerful tool to model the underlying manifold structures of samples in high-dimensional spaces. It involves two phases: constructing an affinity graph from available data and inferring labels for unlabeled nodes on this graph. While numerous algorithms have been developed for label inference, the crucial graph construction phase has received comparatively less attention, despite its significant influence on the subsequent phase. In this paper, we present an optimal asymmetric graph structure for the label inference phase with theoretical motivations. Unlike existing graph construction methods, we differentiate the distinct roles that labeled nodes and unlabeled nodes could play. Accordingly, we design an efficient block-wise graph learning algorithm with a global convergence guarantee. Other benefits induced by our method, such as enhanced robustness to noisy node features, are explored as well. Finally, we perform extensive experiments on synthetic and real-world datasets to demonstrate its superiority to the state-of-the-art graph construction methods in GSSL.</details>

### [Oral] Optimal Learners for Realizable Regression: PAC Learning and Online Learning

**Authors:** Idan Attias, Steve Hanneke, Alkis Kalavasis, Amin Karbasi, Grigoris Velegkas

**Oral Presentation:** Th, Dec 14, 13:20 -- Oral 6D

**<details><summary>Abstract**</summary>

In this work, we aim to characterize the statistical complexity of realizable regression both in the PAC learning setting and the online learning setting. Previous work had established the sufficiency of finiteness of the fat shattering dimension for PAC learnability and the necessity of finiteness of the scaled Natarajan dimension, but little progress had been made towards a more complete characterization since  the work of Simon 1997 (SICOMP '97). To this end,  we first introduce a minimax instance optimal learner for realizable regression and propose a novel dimension that both qualitatively and quantitatively characterizes which classes of real-valued predictors are learnable.  We then identify a combinatorial dimension related to the graph dimension that characterizes ERM learnability in the realizable setting. Finally, we establish a necessary condition for learnability based on a combinatorial dimension related to the DS dimension, and conjecture that it may also be sufficient in this context. Additionally, in the context of online learning we provide a dimension that characterizes the minimax instance optimal cumulative loss up to a constant factor and design an optimal online learner for realizable regression, thus resolving an open question raised by Daskalakis and Golowich in STOC '22.</details>

### Optimal Regret Is Achievable with Bounded Approximate Inference Error: An Enhanced Bayesian Upper Confidence Bound Framework

**Authors:** Ziyi Huang, Henry Lam, Amirhossein Meisami, Haofeng Zhang

**<details><summary>Abstract**</summary>

Bayesian bandit algorithms with approximate Bayesian inference have been widely used in real-world applications. However, there is a large discrepancy between the superior practical performance of these approaches and their theoretical justification. Previous research only indicates a negative theoretical result: Thompson sampling could have a worst-case linear regret $\Omega(T)$ with a constant threshold on the inference error measured by one $\alpha$-divergence. To bridge this gap, we propose an Enhanced Bayesian Upper Confidence Bound (EBUCB) framework that can efficiently accommodate bandit problems in the presence of approximate inference. Our theoretical analysis demonstrates that for Bernoulli multi-armed bandits, EBUCB can achieve the optimal regret order $O(\log T)$ if the inference error measured by two different $\alpha$-divergences is less than a constant, regardless of how large this constant is. To our best knowledge, our study provides the first theoretical regret bound that is better than $o(T)$ in the setting of constant approximate inference error. Furthermore, in concordance with the negative results in previous studies, we show that only one bounded $\alpha$-divergence is insufficient to guarantee a sub-linear regret.</details>

### Optimal Treatment Regimes for Proximal Causal Learning

**Authors:** Tao Shen, Yifan Cui

**<details><summary>Abstract**</summary>

A common concern when a policymaker draws causal inferences from and makes decisions based on observational data is that the measured covariates are insufficiently rich to account for all sources of confounding, i.e., the standard no confoundedness assumption fails to hold. The recently proposed proximal causal inference framework shows that proxy variables that abound in real-life scenarios can be leveraged to identify causal effects and therefore facilitate decision-making. Building upon this line of work, we propose a novel optimal individualized treatment regime based on so-called outcome and treatment confounding bridges. We then show that the value function of this new optimal treatment regime is superior to that of existing ones in the literature. Theoretical guarantees, including identification, superiority, excess value bound, and consistency of the estimated regime, are established. Furthermore, we demonstrate the proposed optimal regime via numerical experiments and a real data application.</details>

### Optimal approximation using complex-valued neural networks

**Authors:** Paul Geuchen, Felix Voigtlaender

**<details><summary>Abstract**</summary>

Complex-valued neural networks (CVNNs) have recently shown promising empirical success, for instance for increasing the stability of recurrent neural networks and for improving the performance in tasks with complex-valued inputs, such as MRI fingerprinting. While the overwhelming success of Deep Learning in the real-valued case is supported by a growing mathematical foundation, such a foundation is still largely lacking in the complex-valued case. We thus analyze the expressivity of CVNNs by studying their approximation properties. Our results yield the first quantitative approximation bounds for CVNNs that apply to a wide class of activation functions including the popular modReLU and complex cardioid activation functions. Precisely, our results apply to any activation function that is smooth but not polyharmonic on some non-empty open set; this is the natural generalization of the class of smooth and non-polynomial activation functions to the complex setting. Our main result shows that the approximation error scales as $m^{-k/(2n)}$ for $m \to \infty$ where $m$ is the number of neurons, $k$ the smoothness of the target function and $n$ is the (complex) input dimension. Under a natural continuity assumption, we show that this rate is optimal; we further discuss the optimality when dropping this assumption. Moreover, we prove that the problem of approximating $C^k$-functions using continuous approximation methods unavoidably suffers from the curse of dimensionality.</details>

### Optimality of Message-Passing Architectures for Sparse Graphs

**Authors:** Aseem Baranwal, Kimon Fountoulakis, Aukosh Jagannath

**<details><summary>Abstract**</summary>

We study the node classification problem on feature-decorated graphs in the sparse setting, i.e., when the expected degree of a node is $O(1)$ in the number of nodes, in the fixed-dimensional asymptotic regime, i.e., the dimension of the feature data is fixed while the number of nodes is large. Such graphs are typically known to be locally tree-like. We introduce a notion of Bayes optimality for node classification tasks, called asymptotic local Bayes optimality, and compute the optimal classifier according to this criterion for a fairly general statistical data model with arbitrary distributions of the node features and edge connectivity. The optimal classifier is implementable using a message-passing graph neural network architecture. We then compute the generalization error of this classifier and compare its performance against existing learning methods theoretically on a well-studied statistical model with naturally identifiable signal-to-noise ratios (SNRs) in the data. We find that the optimal message-passing architecture interpolates between a standard MLP in the regime of low graph signal and a typical convolution in the regime of high graph signal. Furthermore, we prove a corresponding non-asymptotic result.</details>

### [Spotlight] Optimistic Natural Policy Gradient: a Simple Efficient Policy Optimization Framework  for Online RL

**Authors:** Qinghua Liu, Gellert Weisz, András György, Chi Jin, Csaba Szepesvari

**<details><summary>Abstract**</summary>

While policy optimization algorithms have played an important role in recent empirical success of Reinforcement Learning (RL), the existing theoretical understanding of policy optimization remains rather limited---they are either restricted to tabular MDPs or suffer from highly suboptimal sample complexity, especial in online RL where exploration is necessary. This paper proposes a simple efficient policy optimization framework---Optimistic NPG for online RL. Optimistic NPG can be viewed as simply combining of the classic natural policy gradient (NPG) algorithm [Kakade, 2001]  with optimistic policy evaluation subroutines to encourage exploration. For $d$-dimensional linear MDPs, Optimistic NPG is computationally efficient, and learns an $\epsilon$-optimal policy within  $\tilde{\mathcal{O}}(d^2/\epsilon^3)$ samples, which is the first computationally efficient algorithm whose sample complexity has the optimal dimension dependence $\tilde{\Theta}(d^2)$. It also improves over state-of-the-art results of policy optimization algorithms [Zanette et al., 2021] by a factor of $d$. For general function approximation that subsumes linear MDPs, Optimistic NPG, to our best knowledge, is also the first policy optimization algorithm that achieves the polynomial sample complexity for learning near-optimal policies.</details>

### Optimized Covariance Design for AB Test on Social Network under Interference

**Authors:** Qianyi Chen, Bo Li, Lu Deng, Yong Wang

**<details><summary>Abstract**</summary>

Online A/B tests have become increasingly popular and important for social platforms. However, accurately estimating the global average treatment effect (GATE) has proven to be challenging due to network interference, which violates the Stable Unit Treatment Value Assumption (SUTVA) and poses great challenge to experimental design. Existing network experimental design research was mostly based on the unbiased Horvitz-Thompson (HT) estimator with substantial data trimming to ensure unbiasedness at the price of high resultant estimation variance. In this paper, we strive to balance the bias and variance in designing randomized network experiments.  Under a potential outcome model with 1-hop interference, we derive the bias and variance of the standard HT estimator and reveal their relation to the network topological structure and the covariance of the treatment assignment vector. We then propose to formulate the experimental design problem as to optimize the covariance matrix of the treatment assignment vector to achieve the bias and variance balance by minimizing the mean squared error (MSE) of the estimator. An efficient projected gradient descent algorithm is presented to the implement of the desired randomization scheme. Finally, we carry out extensive  simulation studies to demonstrate the advantages of our proposed method over other existing methods in many settings, with different levels of model misspecification.</details>

### Outlier-Robust Wasserstein DRO

**Authors:** Sloan Nietert, Ziv Goldfeld, Soroosh Shafiee

**<details><summary>Abstract**</summary>

Distributionally robust optimization (DRO) is an effective approach for data-driven decision-making in the presence of uncertainty. Geometric uncertainty due to~sampling or localized perturbations of data points is captured by Wasserstein DRO (WDRO), which seeks to learn a model that performs uniformly well over a Wasserstein ball centered around the observed data distribution. However, WDRO fails to account for non-geometric perturbations such as adversarial outliers, which can greatly distort the Wasserstein distance measurement and impede the learned model. We address this gap by proposing a novel outlier-robust WDRO framework for decision-making under both geometric (Wasserstein) perturbations and non-geometric (total variation (TV)) contamination that allows an $\varepsilon$-fraction of data to be arbitrarily corrupted. We design an uncertainty set using a certain robust Wasserstein ball that accounts for both perturbation types and derive minimax optimal excess risk bounds for this procedure that explicitly capture the Wasserstein and TV risks. We prove a strong duality result that enables tractable convex reformulations and efficient computation of our outlier-robust WDRO problem. When the loss function depends only on low-dimensional features of the data, we eliminate certain dimension dependencies from the risk bounds that are unavoidable in the general setting. Finally, we present experiments validating our theory on standard regression and classification tasks.</details>

### P-Flow: A Fast and Data-Efficient Zero-Shot TTS through Speech Prompting

**Authors:** Sungwon Kim, Kevin Shih, rohan badlani, Joao Felipe Santos, Evelina Bakhturina, Mikyas Desta, Rafael Valle, Sungroh Yoon, Bryan Catanzaro

**<details><summary>Abstract**</summary>

While recent large-scale neural codec language models have shown significant improvement in zero-shot TTS by training on thousands of hours of data, they suffer from drawbacks such as a lack of robustness, slow sampling speed similar to previous autoregressive TTS methods, and reliance on pre-trained neural codec representations. Our work proposes P-Flow, a fast and data-efficient zero-shot TTS model that uses speech prompts for speaker adaptation. P-Flow comprises a speech-prompted text encoder for speaker adaptation and a flow matching generative decoder for high-quality and fast speech synthesis. Our speech-prompted text encoder uses speech prompts and text input to generate speaker-conditional text representation. The flow matching generative decoder uses the speaker-conditional output to synthesize high-quality personalized speech significantly faster than in real-time. Unlike the neural codec language models, we specifically train P-Flow on LibriTTS dataset using a continuous mel-representation. Through our training method using continuous speech prompts, P-Flow matches the speaker similarity performance of the large-scale zero-shot TTS models with two orders of magnitude less training data and has more than 20$	imes$ faster sampling speed. Our results show that P-Flow has better pronunciation and is preferred in human likeness and speaker similarity to its recent state-of-the-art counterparts, thus defining P-Flow as an attractive and desirable alternative. We provide audio samples on our demo page: \url{https://research.nvidia.com/labs/adlr/pflow}</details>

### PAC-Bayesian Spectrally-Normalized Bounds for Adversarially Robust Generalization

**Authors:** Jiancong Xiao, Ruoyu Sun, Zhi-Quan Luo

**<details><summary>Abstract**</summary>

Deep neural networks (DNNs) are vulnerable to adversarial attacks. It is found empirically that adversarially robust generalization is crucial in establishing defense algorithms against adversarial attacks. Therefore, it is interesting to study the theoretical guarantee of robust generalization. This paper focuses on norm-based complexity, based on a PAC-Bayes approach (Neyshabur et al., 2017). The main challenge lies in extending the key ingredient, which is a weight perturbation bound in standard settings, to the robust settings. Existing attempts heavily rely on additional strong assumptions, leading to loose bounds. In this paper, we address this issue and provide a spectrally-normalized robust generalization bound for DNNs. Compared to existing bounds, our bound offers two significant advantages: Firstly, it does not depend on additional assumptions. Secondly, it is considerably tighter, aligning with the bounds of standard generalization. Therefore, our result provides a different perspective on understanding robust generalization: The mismatch terms between standard and robust generalization bounds shown in previous studies do not contribute to the poor robust generalization. Instead, these disparities solely due to mathematical issues. Finally, we extend the main result to adversarial robustness against general non-$\ell_p$ attacks and other neural network architectures.</details>

### PAD: A Dataset and Benchmark for Pose-agnostic Anomaly Detection

**Authors:** Qiang Zhou, Weize Li, Lihan Jiang, Guoliang Wang, Guyue Zhou, Shanghang Zhang, Hao Zhao

**<details><summary>Abstract**</summary>

Object anomaly detection is an important problem in the field of machine vision and has seen remarkable progress recently. However, two significant challenges hinder its research and application. First, existing datasets lack comprehensive visual information from various pose angles. They usually have an unrealistic assumption that the anomaly-free training dataset is pose-aligned, and the testing samples have the same pose as the training data. However, in practice, anomaly may exist in any regions on a object, the training and query samples may have different poses, calling for the study on pose-agnostic anomaly detection. Second, the absence of a consensus on experimental protocols for pose-agnostic anomaly detection leads to unfair comparisons of different methods, hindering the research on pose-agnostic anomaly detection. To address these issues, we develop Multi-pose Anomaly Detection (MAD) dataset and Pose-agnostic Anomaly Detection (PAD) benchmark, which takes the first step to address the pose-agnostic anomaly detection problem. Specifically, we build MAD using 20 complex-shaped LEGO toys including 4K views with various poses, and high-quality and diverse 3D anomalies in both simulated and real environments. Additionally, we propose a novel method OmniposeAD, trained using MAD, specifically designed for pose-agnostic anomaly detection. Through comprehensive evaluations, we demonstrate the relevance of our dataset and method. Furthermore, we provide an open-source benchmark library, including dataset and baseline methods that cover 8 anomaly detection paradigms, to facilitate future research and application in this domain. Code, data, and models are publicly available at https://github.com/EricLee0224/PAD.</details>

### PGDiff: Guiding Diffusion Models for Versatile Face Restoration via Partial Guidance

**Authors:** Peiqing Yang, Shangchen Zhou, Qingyi Tao, Chen Change Loy

**<details><summary>Abstract**</summary>

Exploiting pre-trained diffusion models for restoration has recently become a favored alternative to the traditional task-specific training approach. Previous works have achieved noteworthy success by limiting the solution space using explicit degradation models. However, these methods often fall short when faced with complex degradations as they generally cannot be precisely modeled. In this paper, we introduce $\textit{partial guidance}$, a fresh perspective that is more adaptable to real-world degradations compared to existing works. Rather than specifically defining the degradation process, our approach models the desired properties, such as image structure and color statistics of high-quality images, and applies this guidance during the reverse diffusion process. These properties are readily available and make no assumptions about the degradation process. When combined with a diffusion prior, this partial guidance can deliver appealing results across a range of restoration tasks. Additionally, our method can be extended to handle composite tasks by consolidating multiple high-quality image properties, achieved by integrating the guidance from respective tasks. Experimental results demonstrate that our method not only outperforms existing diffusion-prior-based approaches but also competes favorably with task-specific models.</details>

### PPi: Pretraining Brain Signal Model for Patient-independent Seizure Detection

**Authors:** Zhizhang Yuan, Daoze Zhang, YANG YANG, Junru Chen, Yafeng Li

**<details><summary>Abstract**</summary>

Automated seizure detection is of great importance to epilepsy diagnosis and treatment. An emerging method used in seizure detection, stereoelectroencephalography (SEEG), can provide detailed and stereoscopic brainwave information. However, modeling SEEG in clinical scenarios will face challenges like huge domain shift between different patients and dramatic pattern evolution among different brain areas. In this study, we propose a Pretraining-based model for Patient-independent seizure detection (PPi) to address these challenges. Firstly, we design two novel self-supervised tasks which can extract rich information from abundant SEEG data while preserving the unique characteristics between brain signals recorded from different brain areas. Then two techniques channel background subtraction and brain region enhancement are proposed to effectively tackle the domain shift problem. Extensive experiments show that PPi outperforms the SOTA baselines on two public datasets and a real-world clinical dataset collected by ourselves, which demonstrates the effectiveness and practicability of PPi. Finally, visualization analysis illustrates the rationality of the two domain generalization techniques.</details>

### PTADisc: A Cross-Course Dataset Supporting Personalized Learning in Cold-Start Scenarios

**Authors:** Liya Hu, Zhiang Dong, Jingyuan Chen, Guifeng Wang, Zhihua Wang, Zhou Zhao, Fei Wu

**<details><summary>Abstract**</summary>

The focus of our work is on diagnostic tasks in personalized learning, such as cognitive diagnosis and knowledge tracing. The goal of these tasks is to assess students' latent proficiency on knowledge concepts through analyzing their historical learning records. However, existing research has been limited to single-course scenarios; cross-course studies have not been explored due to a lack of dataset. We address this issue by constructing PTADisc, a Diverse, Immense, Student-centered dataset that emphasizes its sufficient Cross-course information for personalized learning. PTADisc includes 74 courses, 1,530,100 students, 4,054 concepts, 225,615 problems, and over 680 million student response logs. Based on PTADisc, we developed a model-agnostic Cross-Course Learner Modeling Framework (CCLMF) which utilizes relationships between students' proficiency across courses to alleviate the difficulty of diagnosing student knowledge state in cold-start scenarios. CCLMF uses a meta network to generate personalized mapping functions between courses. The experimental results on PTADisc verify the effectiveness of CCLMF with an average improvement of 4.2% on AUC. We also report the performance of baseline models for cognitive diagnosis and knowledge tracing over PTADisc, demonstrating that our dataset supports a wide scope of research in personalized learning. Additionally, PTADisc contains valuable programming logs and student-group information that are worth exploring in the future.</details>

### PUG: Photorealistic and Semantically Controllable Synthetic Data for Representation Learning

**Authors:** Florian Bordes, Shashank Shekhar, Mark Ibrahim, Diane Bouchacourt, Pascal Vincent, Ari Morcos

**<details><summary>Abstract**</summary>

Synthetic image datasets offer unmatched advantages for designing and evaluating deep neural networks: they make it possible to (i) render as many data samples as needed, (ii) precisely control each scene and yield granular ground truth labels (and captions), (iii) precisely control distribution shifts between training and testing to isolate variables of interest for sound experimentation.Despite such promise, the use of synthetic image data is still limited -- and often played down -- mainly due to their lack of realism. Most works therefore rely on datasets of real images, which have often been scraped from public images on the internet, and may have issues with regards to privacy, bias, and copyright, while offering little control over how objects precisely appear.In this work, we present a path to democratize the use of photorealistic synthetic data: we develop a new generation of interactive environments for representation learning research, that offer both controllability and realism. We use the Unreal Engine, a powerful game engine well known in the entertainment industry, to produce PUG (Photorealistic Unreal Graphics) environments and datasets for representation learning. Using PUG for evaluation and fine-tuning, we demonstrate the potential of PUG to both enable more rigorous evaluations and to improve model training.</details>

### Pairwise GUI Dataset Construction Between Android Phones and Tablets

**Authors:** han hu, Haolan Zhan, Yujin Huang, Di Liu

**<details><summary>Abstract**</summary>

In the current landscape of pervasive smartphones and tablets, apps frequently exist across both platforms.Although apps share most graphic user interfaces (GUIs) and functionalities across phones and tablets, developers often rebuild from scratch for tablet versions, escalating costs and squandering existing design resources.Researchers are attempting to collect data and employ deep learning in automated GUIs development to enhance developers' productivity.There are currently several publicly accessible GUI page datasets for phones, but none for pairwise GUIs between phones and tablets.This poses a significant barrier to the employment of deep learning in automated GUI development.In this paper, we introduce the Papt dataset, a pioneering pairwise GUI dataset tailored for Android phones and tablets, encompassing 10,035 phone-tablet GUI page pairs sourced from 5,593 unique app pairs.We propose novel pairwise GUI collection approaches for constructing this dataset and delineate its advantages over currently prevailing datasets in the field.Through preliminary experiments on this dataset, we analyze the present challenges of utilizing deep learning in automated GUI development.</details>

### Parallel-mentoring for Offline Model-based Optimization

**Authors:** Can Chen, Christopher Beckham, Zixuan Liu, Xue (Steve) Liu, Chris Pal

**<details><summary>Abstract**</summary>

We study offline model-based optimization to maximize a black-box objective function with a static dataset of designs and scores. These designs encompass a variety of domains, including materials, robots, DNA sequences, and proteins. A common approach trains a proxy on the static dataset and performs gradient ascent to obtain new designs. However, this often results in poor designs due to the proxy inaccuracies for out-of-distribution designs. Recent studies indicate that (a) gradient ascent with a mean ensemble of proxies generally outperforms simple gradient ascent, and (b) a trained proxy provides weak ranking supervision signals for design selection. Motivated by (a) and (b), we propose $\textit{parallel-mentoring}$ as an effective and novel method that facilitates mentoring among proxies, creating a more robust ensemble to mitigate the out-of-distribution issue. We focus on the three-proxy case in the main paper and our method consists of two modules. The first module, $\textit{voting-based pairwise supervision}$, operates on three parallel proxies and captures their ranking supervision signals as pairwise comparison labels. These labels are combined through majority voting to generate consensus labels, which incorporates ranking supervision signals from all proxies and enables mutual mentoring. Yet, label noise arises due to possible incorrect consensus. To alleviate this, we introduce an $\textit{adaptive soft-labeling}$ module with soft-labels initialized as consensus labels. Based on bi-level optimization, this module fine-tunes proxies in the inner level and learns more accurate labels in the outer level to adaptively mentor proxies, resulting in a more robust ensemble. Experiments validate the effectiveness of our method. Our code is available here.</details>

### Perception Test: A Diagnostic Benchmark for Multimodal Video Models

**Authors:** Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, joseph heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alexandre Fréchette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, Joao Carreira

**<details><summary>Abstract**</summary>

We propose a novel multimodal video benchmark - the Perception Test - to evaluate the perception and reasoning skills of pre-trained multimodal models (e.g. Flamingo, BEiT-3, or GPT-4). Compared to existing benchmarks that focus on computational tasks (e.g. classification, detection or tracking), the Perception Test focuses on skills (Memory, Abstraction, Physics, Semantics) and types of reasoning (descriptive, explanatory, predictive, counterfactual) across video, audio, and text modalities, to provide a comprehensive and efficient evaluation tool. The benchmark probes pre-trained models for their transfer capabilities, in a zero-shot / few-shot or limited finetuning regime. For these purposes, the Perception Test introduces 11.6k real-world videos, 23s average length, designed to show perceptually interesting situations, filmed by around 100 participants worldwide. The videos are densely annotated with six types of labels (multiple-choice and grounded video question-answers, object and point tracks, temporal action and sound segments), enabling both language and non-language evaluations. The fine-tuning and validation splits of the benchmark are publicly available (CC-BY license), in addition to a challenge server with a held-out test split. Human baseline results compared to state-of-the-art video QA models show a significant gap in performance (91.4% vs 45.8%), suggesting that there is significant room for improvement in multimodal video understanding.Dataset, baselines code, and challenge server are available at https://github.com/deepmind/perception_test</details>

### Personalized Dictionary Learning for Heterogeneous Datasets

**Authors:** Geyu Liang, Naichen Shi, Raed AL Kontar, Salar Fattahi

**<details><summary>Abstract**</summary>

We introduce a relevant yet challenging problem named Personalized Dictionary Learning (PerDL), where the goal is to learn sparse linear representations from heterogeneous datasets that share some commonality. In PerDL, we model each dataset's shared and unique features as global and local dictionaries. Challenges for PerDL not only are inherited from classical dictionary learning(DL), but also arise due to the unknown nature of the shared and unique features. In this paper, we rigorously formulate this problem and provide conditions under which the global and local dictionaries can be provably disentangled. Under these conditions, we provide a meta-algorithm called Personalized Matching and Averaging (PerMA) that can recover both global and local dictionaries from heterogeneous datasets. PerMA is highly efficient; it converges to the ground truth at a linear rate under suitable conditions. Moreover, it automatically borrows strength from strong learners to improve the prediction of weak learners. As a general framework for extracting global and local dictionaries, we show the application of PerDL in different learning tasks, such as training with imbalanced datasets and video surveillance.</details>

### Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning

**Authors:** Sotetsu Koyamada, Shinri Okano, Soichiro Nishimori, Yu Murata, Keigo Habara, Haruka Kita, Shin Ishii

**<details><summary>Abstract**</summary>

We propose Pgx, a suite of board game reinforcement learning (RL) environments written in JAX and optimized for GPU/TPU accelerators. By leveraging JAX's auto-vectorization and parallelization over accelerators, Pgx can efficiently scale to thousands of simultaneous simulations over accelerators. In our experiments on a DGX-A100 workstation, we discovered that Pgx can simulate RL environments 10-100x faster than existing implementations available in Python. Pgx includes RL environments commonly used as benchmarks in RL research, such as backgammon, chess, shogi, and Go. Additionally, Pgx offers miniature game sets and baseline models to facilitate rapid research cycles. We demonstrate the efficient training of the Gumbel AlphaZero algorithm with Pgx environments. Overall, Pgx provides high-performance environment simulators for researchers to accelerate their RL experiments. Pgx is available at https://github.com/sotetsuk/pgx.</details>

### Physics-Informed Bayesian Optimization of Variational Quantum Circuits

**Authors:** Kim Nicoli, Christopher J. Anders, Lena Funcke, Tobias Hartung, Karl Jansen, Stefan Kühn, Klaus-Robert Müller, Paolo Stornati, Pan Kessel, Shinichi Nakajima

**<details><summary>Abstract**</summary>

In this paper, we propose a novel and powerful method to harness Bayesian optimization for variational quantum eigensolvers (VQEs) - a hybrid quantum-classical protocol used to approximate the ground state of a quantum Hamiltonian. Specifically, we derive a *VQE-kernel* which incorporates important prior information about quantum circuits: the kernel feature map of the VQE-kernel exactly matches the known functional form of the VQE's objective function and thereby significantly reduces the posterior uncertainty.Moreover, we propose a novel acquisition function for Bayesian optimization called \emph{Expected Maximum Improvement over Confident Regions} (EMICoRe) which can actively exploit the inductive bias of the VQE-kernel by treating regions with low predictive uncertainty as indirectly "observed". As a result, observations at as few as three points in the search domain are sufficient to determine the complete objective function along an entire one-dimensional subspace of the optimization landscape. Our numerical experiments demonstrate that our approach improves over state-of-the-art baselines.</details>

### Physion++: Evaluating Physical Scene Understanding that Requires Online Inference of Different Physical Properties

**Authors:** Hsiao-Yu Tung, Mingyu Ding, Zhenfang Chen, Daniel Bear, Chuang Gan, Josh Tenenbaum, Dan Yamins, Judith Fan, Kevin Smith

**<details><summary>Abstract**</summary>

General physical scene understanding requires more than simply localizing and recognizing objects -- it requires knowledge that objects can have different latent properties (e.g., mass or elasticity), and that those properties affect the outcome of physical events. While there has been great progress in physical and video prediction models in recent years, benchmarks to test their performance typically do not require an understanding that objects have individual physical properties, or at best test only those properties that are directly observable (e.g., size or color). This work proposes a novel dataset and benchmark, termed Physion++, that rigorously evaluates visual physical prediction in artificial systems under circumstances where those predictions rely on accurate estimates of the latent physical properties of objects in the scene. Specifically, we test scenarios where accurate prediction relies on estimates of properties such as mass, friction, elasticity, and deformability, and where the values of those properties can only be inferred by observing how objects move and interact with other objects or fluids. We evaluate the performance of a number of state-of-the-art prediction models that span a variety of levels of learning vs. built-in knowledge, and compare that performance to a set of human predictions. We find that models that have been trained using standard regimes and datasets do not spontaneously learn to make inferences about latent properties, but also that models that encode objectness and physical states tend to make better predictions. However, there is still a huge gap between all models and human performance, and all models' predictions correlate poorly with those made by humans, suggesting that no state-of-the-art model is learning to make physical predictions in a human-like way. These results show that current deep learning models that succeed in some settings nevertheless fail to achieve human-level physical prediction in other cases, especially those where latent property inference is required. Project page: https://dingmyu.github.io/physion_v2/</details>

### Pitfall of Optimism: Distributional Reinforcement Learning by Randomizing Risk Criterion

**Authors:** Taehyun Cho, Seungyub Han, Heesoo Lee, Kyungjae Lee, Jungwoo Lee

**<details><summary>Abstract**</summary>

Distributional reinforcement learning algorithms have attempted to utilize estimated uncertainty for exploration, such as optimism in the face of uncertainty. However, using the estimated variance for optimistic exploration may cause biased data collection and hinder convergence or performance. In this paper, we present a novel distributional reinforcement learning that selects actions by randomizing risk criterion without losing the risk-neutral objective. We provide a perturbed distributional Bellman optimality operator by distorting the risk measure. Also,we prove the convergence and optimality of the proposed method with the weaker contraction property. Our theoretical results support that the proposed method does not fall into biased exploration and is guaranteed to converge to an optimal return. Finally, we empirically show that our method outperforms other existing distribution-based algorithms in various environments including Atari 55 games.</details>

### PlanE: Representation Learning over Planar Graphs

**Authors:** Radoslav Dimitrov, Zeyang Zhao, Ralph Abboud, Ismail Ceylan

**<details><summary>Abstract**</summary>

Graph neural networks are prominent models for representation learning over graphs, where the idea is to iteratively compute representations of nodes of an input graph through a series of transformations in such a way that the learned graph function is isomorphism-invariant on graphs, which makes the learned representations graph invariants. On the other hand, it is well-known that graph invariants learned by these class of models are incomplete: there are pairs of non-isomorphic graphs which cannot be distinguished by standard graph neural networks. This is unsurprising given the computational difficulty of graph isomorphism testing on general graphs, but the situation begs to differ for special graph classes, for which efficient graph isomorphism testing algorithms are known, such as planar graphs. The goal of this work is to design architectures for efficiently learning complete invariants of planar graphs. Inspired by the classical planar graph isomorphism algorithm of Hopcroft and Tarjan, we propose PlanE as a framework for planar representation learning. PlanE includes architectures which can learn complete invariants over planar graphs while remaining practically scalable. We empirically validate the strong performance of the resulting model architectures on well-known planar graph benchmarks, achieving multiple state-of-the-art results.</details>

### Practical Equivariances via Relational Conditional Neural Processes

**Authors:** Daolang Huang, Manuel Haussmann, Ulpu Remes, ST John, Grégoire Clarté, Kevin Sebastian Luck, Samuel Kaski, Luigi Acerbi

**<details><summary>Abstract**</summary>

Conditional Neural Processes (CNPs) are a class of metalearning models popular for combining the runtime efficiency of amortized inference with reliable uncertainty quantification. Many relevant machine learning tasks, such as in spatio-temporal modeling, Bayesian Optimization and continuous control, inherently contain equivariances – for example to translation – which the model can exploit for maximal performance. However, prior attempts to include equivariances in CNPs do not scale effectively beyond two input dimensions. In this work, we propose Relational Conditional Neural Processes (RCNPs), an effective approach to incorporate equivariances into any neural process model. Our proposed method extends the applicability and impact of equivariant neural processes to higher dimensions. We empirically demonstrate the competitive performance of RCNPs on a large array of tasks naturally containing equivariances.</details>

### [Spotlight] Pre-RMSNorm and Pre-CRMSNorm Transformers: Equivalent and Efficient Pre-LN Transformers

**Authors:** Zixuan Jiang, Jiaqi Gu, Hanqing Zhu, David Pan

**<details><summary>Abstract**</summary>

Transformers have achieved great success in machine learning applications.Normalization techniques, such as Layer Normalization (LayerNorm, LN) and Root Mean Square Normalization (RMSNorm), play a critical role in accelerating and stabilizing the training of Transformers.While LayerNorm recenters and rescales input vectors, RMSNorm only rescales the vectors by their RMS value.Despite being more computationally efficient, RMSNorm may compromise the representation ability of Transformers.There is currently no consensus regarding the preferred normalization technique, as some models employ LayerNorm while others utilize RMSNorm, especially in recent large language models.It is challenging to convert Transformers with one normalization to the other type.While there is an ongoing disagreement between the two normalization types,we propose a solution to unify two mainstream Transformer architectures, Pre-LN and Pre-RMSNorm Transformers.By removing the inherent redundant mean information in the main branch of Pre-LN Transformers, we can reduce LayerNorm to RMSNorm, achieving higher efficiency.We further propose the Compressed RMSNorm (CRMSNorm) and Pre-CRMSNorm Transformer based on a lossless compression of the zero-mean vectors.We formally establish the equivalence of Pre-LN, Pre-RMSNorm, and Pre-CRMSNorm Transformer variants in both training and inference.It implies that Pre-LN Transformers can be substituted with Pre-(C)RMSNorm counterparts at almost no cost, offering the same arithmetic functionality along with free efficiency improvement.Experiments demonstrate that we can reduce the training and inference time of Pre-LN Transformers by 1% - 10%.</details>

### PriorBand: Practical Hyperparameter Optimization in the Age of Deep Learning

**Authors:** Neeratyoy Mallik, Edward Bergman, Carl Hvarfner, Danny Stoll, Maciej Janowski, Marius Lindauer, Luigi Nardi, Frank Hutter

**<details><summary>Abstract**</summary>

Hyperparameters of Deep Learning (DL) pipelines are crucial for their downstream performance. While a large number of methods for Hyperparameter Optimization (HPO) have been developed, their incurred costs are often untenable for modern DL.Consequently, manual experimentation is still the most prevalent approach to optimize hyperparameters, relying on the researcher's intuition, domain knowledge, and cheap preliminary explorations.To resolve this misalignment between HPO algorithms and DL researchers, we propose PriorBand, an HPO algorithm tailored to DL, able to utilize both expert beliefs and cheap proxy tasks. Empirically, we demonstrate PriorBand's efficiency across a range of DL benchmarks and show its gains under informative expert input and robustness against poor expert beliefs.</details>

### Probabilistic inverse optimal control for non-linear partially observable systems disentangles perceptual uncertainty and behavioral costs

**Authors:** Dominik Straub, Matthias Schultheis, Heinz Koeppl, Constantin Rothkopf

**<details><summary>Abstract**</summary>

Inverse optimal control can be used to characterize behavior in sequential decision-making tasks. Most existing work, however, is limited to fully observable or linear systems, or requires the action signals to be known. Here, we introduce a probabilistic approach to inverse optimal control for partially observable stochastic non-linear systems with unobserved action signals, which unifies previous approaches to inverse optimal control with maximum causal entropy formulations. Using an explicit model of the noise characteristics of the sensory and motor systems of the agent in conjunction with local linearization techniques, we derive an approximate likelihood function for the model parameters, which can be computed within a single forward pass. We present quantitative evaluations on stochastic and partially observable versions of two classic control tasks and two human behavioral tasks. Importantly, we show that our method can disentangle perceptual factors and behavioral costs despite the fact that epistemic and pragmatic actions are intertwined in sequential decision-making under uncertainty, such as in active sensing and active learning. The proposed method has broad applicability, ranging from imitation learning to sensorimotor neuroscience.</details>

### Progressive Ensemble Distillation: Building Ensembles for Efficient Inference

**Authors:** Don Dennis, Abhishek Shetty, Anish Prasad Sevekari, Kazuhito Koishida, Virginia Smith

**<details><summary>Abstract**</summary>

Knowledge distillation is commonly used to compress an ensemble of models into a single model. In this work we study the problem of progressive ensemble distillation: Given a large, pretrained teacher model , we seek to decompose the model into an ensemble of smaller, low-inference cost student models . The resulting ensemble allows for flexibly tuning accuracy vs. inference cost, which can be useful for a multitude of applications in efficient inference. Our method, B-DISTIL, uses a boosting procedure that allows function composition based aggregation rules to construct expressive ensembles with similar performance as using much smaller student models. We demonstrate the effectiveness of B-DISTIL by decomposing pretrained models across a variety of image, speech, and sensor datasets. Our method comes with strong theoretical guarantees in terms of convergence as well as generalization.</details>

### ProteinInvBench: Benchmarking Protein Inverse Folding on Diverse Tasks, Models, and Metrics

**Authors:** Zhangyang Gao, Cheng Tan, Yijie Zhang, Xingran Chen, Lirong Wu, Stan Z. Li

**<details><summary>Abstract**</summary>

Protein inverse folding has attracted increasing attention in recent years. However, we observe that current methods are usually limited to the CATH dataset and the recovery metric. The lack of a unified framework for ensembling and comparing different methods hinders the comprehensive investigation. In this paper, we propose ProteinBench, a new benchmark for protein design, which comprises extended protein design tasks, integrated models, and diverse evaluation metrics. We broaden the application of methods originally designed for single-chain protein design to new scenarios of multi-chain and 	extit{de novo} protein design. Recent impressive methods, including GraphTrans, StructGNN, GVP, GCA, AlphaDesign, ProteinMPNN, PiFold and KWDesign are integrated into our framework. In addition to the recovery, we also evaluate the confidence, diversity, sc-TM, efficiency, and robustness to thoroughly revisit current protein design approaches and inspire future work. As a result, we establish the first comprehensive benchmark for protein design, which is publicly available at \url{https://github.com/A4Bio/OpenCPD}.</details>

### ProteinShake: Building datasets and benchmarks for deep learning on protein structures

**Authors:** Tim Kucera, Carlos Oliver, Dexiong Chen, Karsten Borgwardt

**<details><summary>Abstract**</summary>

We present ProteinShake, a Python software package that simplifies datasetcreation and model evaluation for deep learning on protein structures. Users cancreate custom datasets or load an extensive set of pre-processed datasets fromthe Protein Data Bank (PDB) and AlphaFoldDB. Each dataset is associated withprediction tasks and evaluation functions covering a broad array of biologicalchallenges. A benchmark on these tasks shows that pre-training almost alwaysimproves performance, the optimal data modality (graphs, voxel grids, or pointclouds) is task-dependent, and models struggle to generalize to new structures.ProteinShake makes protein structure data easily accessible and comparisonamong models straightforward, providing challenging benchmark settings withreal-world implications.ProteinShake is available at: https://proteinshake.ai</details>

### [Spotlight] Provable benefits of score matching

**Authors:** Chirag Pabbaraju, Dhruv Rohatgi, Anish Prasad Sevekari, Holden Lee, Ankur Moitra, Andrej Risteski

**<details><summary>Abstract**</summary>

Score matching is an alternative to maximum likelihood (ML) for estimating a probability distribution parametrized up to a constant of proportionality. By fitting the ''score'' of the distribution, it sidesteps the need to compute this constant of proportionality (which is often intractable).While score matching and variants thereof are popular in practice, precise theoretical understanding of the benefits and tradeoffs with maximum likelihood---both computational and statistical---are not well understood. In this work, we give the first example of a natural exponential family of distributions such that the score matching loss is computationally efficient to optimize, and has a comparable statistical efficiency to ML, while the ML loss is intractable to optimize using a gradient-based method. The family consists of exponentials of polynomials of fixed degree, and our result can be viewed as a continuous analogue of recent developments in the discrete setting. Precisely, we show: (1) Designing a zeroth-order or first-order oracle for optimizing the maximum likelihood loss is NP-hard. (2) Maximum likelihood has a statistical efficiency polynomial in the ambient dimension and the radius of the parameters of the family. (3) Minimizing the score matching loss is both computationally and statistically efficient, with complexity polynomial in the ambient dimension.</details>

### [Spotlight] Proximity-Informed Calibration for Deep Neural Networks

**Authors:** Miao Xiong, Ailin Deng, Pang Wei Koh, Jiaying Wu, Shen Li, Jianqing Xu, Bryan Hooi

**<details><summary>Abstract**</summary>

Confidence calibration is central to providing accurate and interpretable uncertainty estimates, especially under safety-critical scenarios. However, we find that existing calibration algorithms often overlook the issue of proximity bias, a phenomenon where models tend to be more overconfident in low proximity data (i.e., data lying in the sparse region of the data distribution) compared to high proximity samples, and thus suffer from inconsistent miscalibration across different proximity samples. We examine the problem over $504$ pretrained ImageNet models and observe that: 1) Proximity bias exists across a wide variety of model architectures and sizes; 2) Transformer-based models are relatively more susceptible to proximity bias than CNN-based models; 3) Proximity bias persists even after performing popular calibration algorithms like temperature scaling; 4) Models tend to overfit more heavily on low proximity samples than on high proximity samples. Motivated by the empirical findings, we propose ProCal, a plug-and-play algorithm with a theoretical guarantee to adjust sample confidence based on proximity. To further quantify the effectiveness of calibration algorithms in mitigating proximity bias, we introduce proximity-informed expected calibration error (PIECE) with theoretical analysis. We show that ProCal is effective in addressing proximity bias and improving calibration on balanced, long-tail, and distribution-shift settings under four metrics over various model architectures. We believe our findings on proximity bias will guide the development of fairer and better-calibrated} models, contributing to the broader pursuit of trustworthy AI.</details>

### QATCH: Benchmarking SQL-centric tasks with Table Representation Learning Models on Your Data

**Authors:** Simone Papicchio, Paolo Papotti, Luca Cagliero

**<details><summary>Abstract**</summary>

Table Representation Learning (TRL) models are commonly pre-trained on large open-domain datasets comprising millions of tables and then used to address downstream tasks. Choosing the right TRL model to use on proprietary data can be challenging, as the best results depend on the content domain, schema, and data quality. Our purpose is to support end-users in testing TRL models on proprietary data in two established SQL-centric tasks, i.e., Question Answering (QA) and Semantic Parsing (SP). We present QATCH (Query-Aided TRL Checklist), a toolbox to highlight TRL models’ strengths and weaknesses on relational tables unseen at training time. For an input table, QATCH automatically generates a testing checklist tailored to QA and SP. Checklist generation is driven by a SQL query engine that crafts tests of different complexity. This design facilitates inherent portability, allowing the checks to be used by alternative models. We also introduce a set of cross-task performance metrics evaluating the TRL model’s performance over its output. Finally, we show how QATCH automatically generates tests for proprietary datasets to evaluate various state-of-the-art models including TAPAS, TAPEX, and CHATGPT.</details>

### [Spotlight] QuIP: 2-Bit Quantization of Large Language Models With Guarantees

**Authors:** Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa

**<details><summary>Abstract**</summary>

This work studies post-training parameter quantization in large language models (LLMs). We introduce quantization with incoherence processing (QuIP), a new method based on the insight that quantization benefits from incoherent weight and Hessian matrices, i.e., from the weights being even in magnitude and the directions in which it is important to round them accurately being unaligned with the coordinate axes. QuIP consists of two steps: (1) an adaptive rounding procedure minimizing a quadratic proxy objective; (2) efficient pre- and post-processing that ensures weight and Hessian incoherence via multiplication by random orthogonal matrices. We complement QuIP with the first theoretical analysis for an LLM-scale quantization algorithm, and show that our theory also applies to an existing method, OPTQ. Empirically, we find that our incoherence preprocessing improves several existing quantization algorithms and yields the first LLM quantization methods that produce viable results using only two bits per weight. Our code can be found at https://github.com/jerry-chee/QuIP.</details>

### QuadAttac$K$: A Quadratic Programming Approach to Learning Ordered Top-$K$ Adversarial Attacks

**Authors:** Thomas Paniagua, Ryan Grainger, Tianfu Wu

**<details><summary>Abstract**</summary>

The adversarial vulnerability of Deep Neural Networks (DNNs) has been well-known and widely concerned, often under the context of learning top-$1$ attacks (e.g., fooling a DNN to classify a cat image as dog). This paper shows that the concern is much more serious by learning significantly more aggressive ordered top-$K$ clear-box targeted attacks proposed in~\citep{zhang2020learning}. We propose a novel and rigorous quadratic programming (QP) method of learning ordered top-$K$ attacks with low computing cost, dubbed as \textbf{QuadAttac$K$}. Our QuadAttac$K$ directly solves the QP to satisfy the attack constraint in the feature embedding space (i.e., the input space to the final linear classifier), which thus exploits the semantics of the feature embedding space (i.e., the principle of class coherence). With the optimized feature embedding  vector perturbation, it then computes the adversarial perturbation in the data space via the vanilla one-step back-propagation. In experiments, the proposed QuadAttac$K$ is tested in the ImageNet-1k  classification using ResNet-50, DenseNet-121, and Vision Transformers (ViT-B and DEiT-S). It successfully pushes the boundary of successful ordered top-$K$ attacks from $K=10$ up to $K=20$ at a cheap budget ($1\times 60$) and further improves attack success rates for $K=5$ for all tested models, while retaining the performance for $K=1$.</details>

### Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing

**Authors:** Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort

**<details><summary>Abstract**</summary>

Transformer models have been widely adopted in various domains over the last years and especially large language models have advanced the field of AI significantly. Due to their size, the capability of these networks has increased tremendously, but this has come at the cost of a significant increase in necessary compute. Quantization is one of the most effective ways for reducing the computational time and memory consumption of neural networks. Many studies have shown, however, that modern transformer models tend to learn strong outliers in their activations, making them difficult to quantize. To retain acceptable performance, the existence of these outliers requires activations to be in higher-bitwidth or the use of different numeric formats, extra fine-tuning, or other workarounds. We show that strong outliers are related to very specific behavior of attention heads that try to learn a "no-op", or just a partial update of the residual. To achieve the exact zeros needed in the attention matrix for a no-update, the input to the softmax is pushed to be larger and larger during training, causing outliers in other parts of the network. Based on these observations, we propose two simple (independent) modifications to the attention mechanism - _clipped softmax_ and _gated attention_. We empirically show that models pre-trained using our methods learn significantly smaller outliers while maintaining and sometimes even improving the floating-point task performance. This enables us to quantize transformers to full INT8 quantization of the activations without any additional effort. We demonstrate the effectiveness of our methods on both language models (BERT, OPT) and vision transformers.</details>

### Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations and Beyond

**Authors:** Anna Hedström, Leander Weber, Daniel Krakowczyk, Dilyara Bareeva, Franz Motzkus, Wojciech Samek, Sebastian Lapuschkin, Marina Höhne

**<details><summary>Abstract**</summary>

The evaluation of explanation methods is a research topic that has not yet been explored deeply, however, since explainability is supposed to strengthen trust in artificial intelligence, it is necessary to systematically review and compare explanation methods in order to confirm their correctness. Until now, no tool with focus on XAI evaluation exists that exhaustively and speedily allows researchers to evaluate the performance of explanations of neural network predictions. To increase transparency and reproducibility in the field, we therefore built Quantus—a comprehensive, evaluation toolkit in Python that includes a growing, well-organised collection of evaluation metrics and tutorials for evaluating explainable methods. The toolkit has been thoroughly tested and is available under an open-source license on PyPi (or on https://github.com/understandable-machine-intelligence-lab/Quantus/).</details>

### [Spotlight] Quasi-Monte Carlo Graph Random Features

**Authors:** Isaac Reid, Adrian Weller, Krzysztof M Choromanski

**<details><summary>Abstract**</summary>

We present a novel mechanism to improve the accuracy of the recently-introduced class of graph random features (GRFs). Our method induces negative correlations between the lengths of the algorithm's random walks by imposing antithetic termination: a procedure to sample more diverse random walks which may be of independent interest. It has a trivial drop-in implementation. We derive strong theoretical guarantees on the properties of these quasi-Monte Carlo GRFs (q-GRFs), proving that they yield lower-variance estimators of the $2$-regularised Laplacian kernel under mild conditions. Remarkably, our results hold for any graph topology. We demonstrate empirical accuracy improvements on a variety of tasks including a new practical application: time-efficient approximation of the graph diffusion process. To our knowledge, q-GRFs constitute the first rigorously studied quasi-Monte Carlo scheme for kernels defined on combinatorial objects, inviting new research on correlations between graph random walks.</details>

### RELIC: Reproducibility and Extension on LIC metric in quantifying bias in captioning models

**Authors:** Martijn van Raaphorst, Egoitz Gonzalez, Marta Grasa, Paula Antequera Hernández

**<details><summary>Abstract**</summary>

Scope of ReproducibilityIn this work we reproduce and extend the results presented in “Quantifying Societal Bias Amplification in Image Captioning” by Hirota et al. This paper introduces LIC, a metric to quantify bias amplification by image captioning models, which is tested for gender and racial bias amplification. The original paper claims that this metric is robust, and that all models amplify both gender and racial bias. It also claims that gender bias is more apparent than racial bias, and the Equalizer variation of the NIC+ model increases gender but not racial bias. We repeat the measurements to confirm these claims. We extend the analysis to whether the method can be generalized to other attributes such as bias in age.MethodologyThe authors of the paper provided a repository containing the necessary code. We had to modify it and add several scripts to be able to run all the experiments. The results were reproduced using the same subset of COCO [3] as in the original paper. Additionally, we manually labeled images according to age for our specific experiments. All experiments were ran on GPUs for a total of approximately 100 hours.ResultsAll claims made by the paper seem to hold, as the results we obtained follow the same trends as those presented in the original paper even if they do not match exactly. However, the same cannot always be said of the additional experiments.What was easyThe paper was clear and matched the implementation. The code was well organized and was easy to run using the command interface provided by the authors. This also made it easy to replicate and expand upon it by adding our own new features. The data was also readily available and could be easily downloaded with no need for preprocessing.What was difficultWe had to run several iterations of the same code, using different seeds and models, to get the results with the same conditions as in the original paper, which made use of time and resources. Our own experiments required additional time to hand-annotate data due to lack of data for new features.Communication with original authorsThere was no contact with the authors, since the code and the experiments were clear and did not need any additional explanation.</details>

### RIO: A Benchmark for Reasoning Intention-Oriented Objects in Open Environments

**Authors:** Mengxue Qu, Yu Wu, Wu Liu, Xiaodan Liang, Jingkuan Song, Yao Zhao, Yunchao Wei

**<details><summary>Abstract**</summary>

Intention-oriented object detection aims to detect desired objects based on specific intentions or requirements. For instance, when we desire to "lie down and rest", we instinctively seek out a suitable option such as a "bed" or a "sofa" that can fulfill our needs. Previous work in this area is limited either by the number of intention descriptions or by the affordance vocabulary available for intention objects. These limitations make it challenging to handle intentions in open environments effectively. To facilitate this research, we construct a comprehensive dataset called Reasoning Intention-Oriented Objects (RIO). In particular, RIO is specifically designed to incorporate diverse real-world scenarios and a wide range of object categories. It offers the following key features: 1) intention descriptions in RIO are represented as natural sentences rather than a mere word or verb phrase, making them more practical and meaningful; 2) the intention descriptions are contextually relevant to the scene, enabling a broader range of potential functionalities associated with the objects; 3) the dataset comprises a total of 40,214 images and 130,585 intention-object pairs. With the proposed RIO, we evaluate the ability of some existing models to reason intention-oriented objects in open environments.</details>

### RL-ViGen: A Reinforcement Learning Benchmark for Visual Generalization

**Authors:** Zhecheng Yuan, Sizhe Yang, Pu Hua, Can Chang, Kaizhe Hu, Huazhe Xu

**<details><summary>Abstract**</summary>

Visual Reinforcement Learning (Visual RL), coupled with high-dimensional observations, has consistently confronted the long-standing challenge of out-of-distribution generalization. Despite the focus on algorithms aimed at resolving visual generalization problems, we argue that the devil is in the existing benchmarks as they are restricted to isolated tasks and generalization categories, undermining a comprehensive evaluation of agents' visual generalization capabilities. To bridge this gap, we introduce RL-ViGen: a novel **R**einforcement **L**earning Benchmark for **Vi**sual **Gen**eralization, which contains diverse tasks and a wide spectrum of generalization types, thereby facilitating the derivation of more reliable conclusions. Furthermore, RL-ViGen incorporates the latest generalization visual RL algorithms into a unified framework, under which the experiment results indicate that no single existing algorithm has prevailed universally across tasks. Our aspiration is that Rl-ViGen will serve as a catalyst in this area, and lay a foundation for the future creation of universal visual generalization RL agents suitable for real-world scenarios. Access to our code and implemented algorithms is provided at https://gemcollector.github.io/RL-ViGen/.</details>

### RL-based Stateful Neural Adaptive Sampling and Denoising for Real-Time Path Tracing

**Authors:** Antoine Scardigli, Lukas Cavigelli, Lorenz K. Müller

**<details><summary>Abstract**</summary>

Monte-Carlo path tracing is a powerful technique for realistic image synthesis but suffers from high levels of noise at low sample counts, limiting its use in real-time applications. To address this, we propose a framework with end-to-end training of a sampling importance network, a latent space encoder network, and a denoiser network. Our approach uses reinforcement learning to optimize the sampling importance network, thus avoiding explicit numerically approximated gradients. Our method does not aggregate the sampled values per pixel by averaging but keeps all sampled values which are then fed into the latent space encoder. The encoder replaces handcrafted spatiotemporal heuristics by learned representations in a latent space. Finally, a neural denoiser is trained to refine the output image. Our approach increases visual quality on several challenging datasets and reduces rendering times for equal quality by a factor of 1.6x compared to the previous state-of-the-art, making it a promising solution for real-time applications.</details>

### RVD: A Handheld Device-Based Fundus Video Dataset for Retinal Vessel Segmentation

**Authors:** MD WAHIDUZZAMAN KHAN, Hongwei Sheng, Hu Zhang, Heming Du, Sen Wang, Minas Coroneo, Farshid Hajati, Sahar Shariflou, Michael Kalloniatis, Jack Phu, Ashish Agar, Zi Huang, S.Mojtaba Golzan, Xin Yu

**<details><summary>Abstract**</summary>

Retinal vessel segmentation is generally grounded in image-based datasets collected with bench-top devices. The static images naturally lose the dynamic characteristics of retina fluctuation, resulting in diminished dataset richness, and the usage of bench-top devices further restricts dataset scalability due to its limited accessibility. Considering these limitations, we introduce the first video-based retinal dataset by employing handheld devices for data acquisition. The dataset comprises 635 smartphone-based fundus videos collected from four different clinics, involving 415 patients from 50 to 75 years old. It delivers comprehensive and precise annotations of retinal structures in both spatial and temporal dimensions, aiming to advance the landscape of vasculature segmentation. Specifically, the dataset provides three levels of spatial annotations: binary vessel masks for overall retinal structure delineation, general vein-artery masks for distinguishing the vein and artery, and fine-grained vein-artery masks for further characterizing the granularities of each artery and vein. In addition, the dataset offers temporal annotations that capture the vessel pulsation characteristics, assisting in detecting ocular diseases that require fine-grained recognition of hemodynamic fluctuation. In application, our dataset exhibits a significant domain shift with respect to data captured by bench-top devices, thus posing great challenges to existing methods. Thanks to rich annotations and data scales, our dataset potentially paves the path for more advanced retinal analysis and accurate disease diagnosis. In the experiments, we provide evaluation metrics and benchmark results on our dataset, reflecting both the potential and challenges it offers for vessel segmentation tasks. We hope this challenging dataset would significantly contribute to the development of eye disease diagnosis and early prevention.</details>

### [Oral] Random Cuts are Optimal for Explainable k-Medians

**Authors:** Konstantin Makarychev, Liren Shan

**Oral Presentation:** Th, Dec 14, 13:35 -- Oral 6D

**<details><summary>Abstract**</summary>

We show that the RandomCoordinateCut algorithm gives the optimal competitive ratio for explainable $k$-medians in $\ell_1$. The problem of explainable $k$-medians was introduced by Dasgupta, Frost, Moshkovitz, and Rashtchian in 2020. Several groups of authors independently proposed a simple polynomial-time randomized algorithm for the problem and showed that this algorithm is $O(\log k \log\log k)$ competitive.  We provide a tight analysis of the algorithm and prove that its competitive ratio is upper bounded by $2\ln k+2$. This bound matches the $\Omega(\log k)$ lower bound by Dasgupta et al (2020).</details>

### ReDS: Offline RL With Heteroskedastic Datasets via Support Constraints

**Authors:** Anikait Singh, Aviral Kumar, Quan Vuong, Yevgen Chebotar, Sergey Levine

**<details><summary>Abstract**</summary>

Offline reinforcement learning (RL) learns policies entirely from static datasets. Practical applications of offline RL will inevitably require learning from datasets where the variability of demonstrated behaviors changes non-uniformly across the state space. For example, at a red light, nearly all human drivers behave similarly by stopping, but when merging onto a highway, some drivers merge quickly, efficiently, and safely, while many hesitate or merge dangerously. Both theoretically and empirically, we show that typical offline RL methods, which are based on distribution constraints fail to learn from data with such non-uniform variability, due to the requirement to stay close to the behavior policy **to the same extent** across the state space. Ideally, the learned policy should be free to choose **per state** how closely to follow the behavior policy to maximize long-term return, as long as the learned policy stays within the support of the behavior policy. To instantiate this principle, we reweight the data distribution in conservative Q-learning (CQL) to obtain an approximate support constraint formulation. The reweighted distribution is a mixture of the current policy and an additional policy trained to mine poor actions that are likely under the behavior policy. Our method, CQL (ReDS), is theoretically motivated, and improves performance across a wide range of offline RL problems in games, navigation, and pixel-based manipulation.</details>

### [Spotlight] Real-World Image Variation by Aligning Diffusion Inversion Chain

**Authors:** Yuechen Zhang, Jinbo Xing, Eric Lo, Jiaya Jia

**<details><summary>Abstract**</summary>

Recent diffusion model advancements have enabled high-fidelity images to be generated using text prompts. However, a domain gap exists between generated images and real-world images, which poses a challenge in generating high-quality variations of real-world images. Our investigation uncovers that this domain gap originates from a latents' distribution gap in different diffusion processes. To address this issue, we propose a novel inference pipeline called Real-world Image Variation by ALignment (RIVAL) that utilizes diffusion models to generate image variations from a single image exemplar. Our pipeline enhances the generation quality of image variations by aligning the image generation process to the source image's inversion chain. Specifically, we demonstrate that step-wise latent distribution alignment is essential for generating high-quality variations. To attain this, we design a cross-image self-attention injection for feature interaction and a step-wise distribution normalization to align the latent features. Incorporating these alignment processes into a diffusion model allows RIVAL to generate high-quality image variations without further parameter optimization. Our experimental results demonstrate that our proposed approach outperforms existing methods concerning semantic similarity and perceptual quality. This generalized inference pipeline can be easily applied to other diffusion-based generation tasks, such as image-conditioned text-to-image generation and stylization. Project page: https://rival-diff.github.io</details>

### RealTime QA: What's the Answer Right Now?

**Authors:** Jungo Kasai, Keisuke Sakaguchi, yoichi takahashi, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir Radev, Noah Smith, Yejin Choi, Kentaro Inui

**<details><summary>Abstract**</summary>

We introduce RealTime QA, a dynamic question answering (QA) platform that announces questions and evaluates systems on a regular basis (weekly in this version). RealTime QA inquires about the current world, and QA systems need to answer questions about novel events or information. It therefore challenges static, conventional assumptions in open-domain QA datasets and pursues instantaneous applications. We build strong baseline models upon large pretrained language models, including GPT-3 and T5. Our benchmark is an ongoing effort, and this paper presents real-time evaluation results over the past year. Our experimental results show that GPT-3 can often properly update its generation results, based on newly-retrieved documents, highlighting the importance of up-to-date information retrieval. Nonetheless, we find that GPT-3 tends to return outdated answers when retrieved documents do not provide sufficient information to find an answer. This suggests an important avenue for future research: can an open-domain QA system identify such unanswerable cases and communicate with the user or even the retrieval module to modify the retrieval results? We hope that RealTime QA will spur progress in instantaneous applications of question answering and beyond.</details>

### Realistic Synthetic Financial Transactions for Anti-Money Laundering Models

**Authors:** Erik Altman, Jovan Blanuša, Luc von Niederhäusern, Beni Egressy, Andreea Anghel, Kubilay Atasu

**<details><summary>Abstract**</summary>

With the widespread digitization of finance and the increasing popularity of cryptocurrencies, the sophistication of fraud schemes devised by cybercriminals is growing. Money laundering -- the movement of illicit funds to conceal their origins -- can cross bank and national boundaries, producing complex transaction patterns. The UN estimates 2-5\% of global GDP or \$0.8 - \$2.0 trillion dollars are laundered globally each year.  Unfortunately, real data to train machine learning models to detect laundering is generally not available, and previous synthetic data generators have had significant shortcomings. A realistic, standardized, publicly-available benchmark is needed for comparing models and for the advancement of the area.To this end, this paper contributes a synthetic financial transaction dataset generator and a set of synthetically generated AML (Anti-Money Laundering) datasets.  We have calibrated this agent-based generator to match real transactions as closely as possible and made the datasets public. We describe the generator in detail and demonstrate how the datasets generated can help compare different machine learning models in terms of their AML abilities. In a key way, using synthetic data in these comparisons can be even better than using real data: the ground truth labels are complete, whilst many laundering transactions in real data are never detected.</details>

### Recasting Continual Learning as Sequence Modeling

**Authors:** Soochan Lee, Jaehyeon Son, Gunhee Kim

**<details><summary>Abstract**</summary>

In this work, we aim to establish a strong connection between two significant bodies of machine learning research: continual learning and sequence modeling.That is, we propose to formulate continual learning as a sequence modeling problem, allowing advanced sequence models to be utilized for continual learning.Under this formulation, the continual learning process becomes the forward pass of a sequence model.By adopting the meta-continual learning (MCL) framework, we can train the sequence model at the meta-level, on multiple continual learning episodes.As a specific example of our new formulation, we demonstrate the application of Transformers and their efficient variants as MCL methods.Our experiments on seven benchmarks, covering both classification and regression, show that sequence models can be an attractive solution for general MCL.</details>

### Reflexion: language agents with verbal reinforcement learning

**Authors:** Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao

**<details><summary>Abstract**</summary>

Large language models (LLMs) have been increasingly used to interact with external environments (e.g., games, compilers, APIs) as goal-driven agents. However, it remains challenging for these language agents to quickly and efficiently learn from trial-and-error as traditional reinforcement learning methods require extensive training samples and expensive model fine-tuning. We propose \emph{Reflexion}, a novel framework to reinforce language agents not by updating weights, but instead through linguistic feedback. Concretely, Reflexion agents verbally reflect on task feedback signals, then maintain their own reflective text in an episodic memory buffer to induce better decision-making in subsequent trials. Reflexion is flexible enough to incorporate various types (scalar values or free-form language) and sources (external or internally simulated) of feedback signals, and obtains significant improvements over a baseline agent across diverse tasks (sequential decision-making, coding, language reasoning). For example, Reflexion achieves a 91\% pass@1 accuracy on the HumanEval coding benchmark, surpassing the previous state-of-the-art GPT-4 that achieves 80\%. We also conduct ablation and analysis studies using different feedback signals, feedback incorporation methods, and agent types, and provide insights into how they affect performance. We release all code, demos, and datasets at \url{https://github.com/noahshinn024/reflexion}.</details>

### Reimagining Synthetic Tabular Data Generation through Data-Centric AI: A Comprehensive Benchmark

**Authors:** Lasse Hansen, Nabeel Seedat, Mihaela van der Schaar, Andrija Petrovic

**<details><summary>Abstract**</summary>

Synthetic data serves as an alternative in training machine learning models, particularly when real-world data is limited or inaccessible. However, ensuring that synthetic data mirrors the complex nuances of real-world data is a challenging task. This paper addresses this issue by exploring the potential of integrating data-centric AI techniques which profile the data to guide the synthetic data generation process. Moreover, we shed light on the often ignored consequences of neglecting these data profiles during synthetic data generation --- despite seemingly high statistical fidelity. Subsequently, we propose a novel framework to evaluate the integration of data profiles to guide the creation of more representative synthetic data. In an empirical study, we evaluate the performance of five state-of-the-art models for tabular data generation on eleven distinct tabular datasets. The findings offer critical insights into the successes and limitations of current synthetic data generation techniques. Finally, we provide practical recommendations for integrating data-centric insights into the synthetic data generation process, with a specific focus on classification performance, model selection, and feature selection. This study aims to reevaluate conventional approaches to synthetic data generation and promote the application of data-centric AI techniques in improving the quality and effectiveness of synthetic data.</details>

### Replicable Reinforcement Learning

**Authors:** Eric Eaton, Marcel Hussing, Michael Kearns, Jessica Sorrell

**<details><summary>Abstract**</summary>

The replicability crisis in the social, behavioral, and data sciences has led to the formulation of algorithm frameworks for replicability --- i.e., a requirement that an algorithm produce identical outputs (with high probability) when run on two different samples from the same underlying distribution. While still in its infancy, provably replicable algorithms have been developed for many fundamental tasks in machine learning and statistics, including statistical query learning, the heavy hitters problem, and distribution testing. In this work we initiate the study of replicable reinforcement learning, providing a provably replicable algorithm for parallel value iteration, and a provably replicable version of R-Max in the episodic setting. These are the first formal replicability results for control problems, which present different challenges for replication than batch learning settings.</details>

### Reproducibility Study of “Quantifying Societal Bias Amplification in Image Captioning”

**Authors:** Farrukh Baratov, Goksenin Yuksel, Darie Petcu, Jan Bakker

**<details><summary>Abstract**</summary>

Scope of reproducibility - We study the reproducibility of the paper "Quantifying Societal Bias Amplification in Image Captioning" by Hirota et al. In this paper, the authors propose a new metric to measure bias amplification, called LIC, and evaluate it on multiple image captioning models. Based on this evaluation, they make the following main claims which we aim to verify: (1) all models amplify gender bias, (2) all models amplify racial bias, (3) LIC is robust against encoders, and (4) the NIC+Equalizer model increases gender bias with respect to the baseline. We also extend upon the original work by evaluating LIC for age bias.Methodology - For our reproduction, we were able to run the code provided by the authors without any modifications. For our extension, we automatically labelled the images in the dataset with age annotations and adjusted the code to work with this dataset. In total, 38 GPU hours were needed to perform all experiments.Results - The reproduced results are close to the original results and support all four main claims. Furthermore, our additional results show that only a subset of the models amplifies age bias, while they strengthen the claim that LIC is robust against encoders. However, we acknowledge that our extension to age bias has its limitations.What was easy - The author's code and the data needed to run it are publicly available. The code required no modification to run and the scripts were provided with an extensive argument parser, allowing us to quickly set up our experiments. Moreover, the details of the original experiments were clearly stated in the appendix.What was difficult - We found that it was difficult to interpret the author's code as the provided documentation contained room for improvement. Also, the scripts contained repetitive code. While the authors retrained all image captioning models, they did not share the model weights, making it difficult to extend upon their work.Communication with original authors - No (attempt at) communication with the original authors was performed.</details>

### Reproducibility Study of ”Label-Free Explainability for Unsupervised Models”

**Authors:** Julius Wagenbach, Gergely Papp, Niklas Mather, Laurens de Vries

**<details><summary>Abstract**</summary>

In this work, we present our reproducibility study of "Label-Free Explainability for Unsupervised Models", a paper that introduces two post‐hoc explanation techniques for neural networks: (1) label‐free feature importance and (2) label‐free example importance. Our study focuses on the reproducibility of the authors’ most important claims: (i) perturbing features with the highest importance scores causes higherlatent shift than perturbing random pixels, (ii) label‐free example importance scores help to identify training examples that are highly related to a given test example, (iii) unsupervised models trained on different tasks show moderate correlation among the highest scored features and (iv) low correlation in example scores measured on a fixed set of data points, and (v) increasing the disentanglement with β in a β‐VAE does not imply that latent units will focus on more different features. We reviewed the authors’ code, checked if the implementation of experiments matched with the paper, and also ran all experiments. The results are shown to be reproducible. Moreover, we extended the codebase in order to run the experiments on more datasets, and to test the claims with other experiments.</details>

### Reproducibility study of the Fairness-enhanced Node Representation Learning

**Authors:** Gijs Moens, Job De Witte, Tobias Gobel, Meggie Van den Oever

**<details><summary>Abstract**</summary>

"CrossWalk: Fairness-Enhanced Node Representation Learning" is set to be reproduced and reviewed. It presents an extension to existing graph algorithms that incorporate the idea of biased random walks for obtaining node embeddings. CrossWalk incorporates fairness by up-weighting edges of nodes located near group boundaries. The authors claim that their approach outperforms baseline algorithms, such as DeepWalk and FairWalk, in terms of reducing the disparity between different classes within a graph network. The authors accompanied their paper with the publication of an open GitHub page, which includes the source code and relevant data sets. The limited size of the data sets in combination with the efficient algorithms enables the experiments to be conducted without significant difficulties and is computable on standard CPUs without the need for additional resources.In this reproducibility report, the outcomes of the experiments are in agreement with the results presented in the original paper. However, the inherent randomness of the random walks makes it difficult to quantify the extent of similarity between the reproduced results and the results as stated in the original paper. However, it can be concluded that CrossWalk results in a decreased disparity between groups in graph networks.The authors effectively conveyed the underlying concept of their proposed method, rendering it both intriguing and straightforward to comprehend the key ideas. Furthermore, the authors successfully incorporated a range of methods and baseline algorithms into the paper.In contrast, the source code may not have been optimally constructed with reproducibility in mind. Certain sections of the code appear to be unfinished or inadequately executed. Additionally, the authors neglected to specify key hyperparameters, resulting in the unidentifiability of certain results. This presents challenges in drawing conclusions based on the available sources.The authors were unable to respond in time for elaborating on certain implementation details. However, we did receive additional data which was crucial to obtaining certain results.</details>

### Residual Alignment: Uncovering the Mechanisms of Residual Networks

**Authors:** Jianing Li, Vardan Papyan

**<details><summary>Abstract**</summary>

The ResNet architecture has been widely adopted in deep learning due to its significant boost to performance through the use of simple skip connections, yet the underlying mechanisms leading to its success remain largely unknown. In this paper, we conduct a thorough empirical study of the ResNet architecture in classification tasks by linearizing its constituent residual blocks using Residual Jacobians and measuring their singular value decompositions. Our measurements ([code](https://colab.research.google.com/drive/1yKjEg2yF616tnZFAfuN0aQ-E9v3JmyjN?usp=sharing)) reveal a process called Residual Alignment (RA) characterized by four properties:- **(RA1):** intermediate representations of a given input are *equispaced* on a *line*, embedded in high dimensional space, as observed by Gai and Zhang [2021];- **(RA2):** top left and right singular vectors of Residual Jacobians align with each other and across different depths;- **(RA3):** Residual Jacobians are at most rank $C$ for fully-connected ResNets, where $C$ is the number of classes; and- **(RA4):** top singular values of Residual Jacobians scale inversely with depth.RA consistently occurs in models that generalize well, in both fully-connected and convolutional architectures, across various depths and widths, for varying numbers of classes, on all tested benchmark datasets, but ceases to occur once the skip connections are removed. It also provably occurs in a novel mathematical model we propose. This phenomenon reveals a strong alignment between residual branches of a ResNet (RA2+4), imparting a highly rigid geometric structure to the intermediate representations as they progress *linearly* through the network (RA1) up to the final layer, where they undergo Neural Collapse.</details>

### Restart Sampling for Improving Generative Processes

**Authors:** Yilun Xu, Mingyang Deng, Xiang Cheng, Yonglong Tian, Ziming Liu, Tommi Jaakkola

**<details><summary>Abstract**</summary>

Generative processes that involve solving differential equations, such as diffusion models, frequently necessitate balancing speed and quality. ODE-based samplers are fast but plateau in performance while SDE-based samplers deliver higher sample quality at the cost of increased sampling time. We attribute this difference to sampling errors: ODE-samplers involve smaller discretization errors while stochasticity in SDE contracts accumulated errors. Based on these findings, we propose a novel sampling algorithm called 	extit{Restart} in order to better balance discretization errors and contraction. The sampling method alternates between adding substantial noise in additional forward steps and strictly following a backward ODE. Empirically, Restart sampler surpasses previous SDE and ODE samplers in both speed and accuracy. Restart not only outperforms the previous best SDE results, but also accelerates the sampling speed by 10-fold / 2-fold on CIFAR-10 / ImageNet $64{	imes} 64$. In addition, it attains significantly better sample quality than ODE samplers within comparable sampling times. Moreover, Restart better balances text-image alignment/visual quality versus diversity than previous samplers in the large-scale text-to-image Stable Diffusion model pre-trained on LAION $512{	imes} 512$. Code is available at https://github.com/Newbeeer/diffusion_restart_sampling</details>

### Rethinking Incentives in Recommender Systems: Are Monotone Rewards Always Beneficial?

**Authors:** Fan Yao, Chuanhao Li, Karthik Abinav Sankararaman, Yiming Liao, Yan Zhu, Qifan Wang, Hongning Wang, Haifeng Xu

**<details><summary>Abstract**</summary>

The past decade has witnessed the flourishing of a new profession as media content creators, who rely on revenue streams from online content recommendation platforms. The reward mechanism employed by these platforms creates a competitive environment among creators which affects their production choices and, consequently, content distribution and system welfare. It is thus crucial to design the platform's reward mechanism in order to steer the creators' competition towards a desirable welfare outcome in the long run. This work makes two major contributions in this regard: first, we uncover a fundamental limit about a class of widely adopted mechanisms, coined \emph{Merit-based Monotone Mechanisms}, by showing that they inevitably lead to a constant fraction loss of the optimal welfare. To circumvent this limitation, we introduce \emph{Backward Rewarding Mechanisms} (BRMs) and show that the competition game resultant from BRMs possesses a potential game structure. BRMs thus naturally induce strategic creators' collective behaviors towards optimizing the potential function, which can be designed to match any given welfare metric. In addition, the class of BRM can be parameterized so that it allows the platform to directly optimize welfare within the feasible mechanism space even when the welfare metric is not explicitly defined.</details>

### Retrieval-Augmented Multiple Instance Learning

**Authors:** Yufei CUI, Ziquan Liu, Yixin Chen, Yuchen Lu, Xinyue Yu, Xue (Steve) Liu, Tei-Wei Kuo, Miguel Rodrigues, Chun Jason XUE, Antoni Chan

**<details><summary>Abstract**</summary>

Multiple Instance Learning (MIL) is a crucial weakly supervised learning method applied across various domains, e.g., medical diagnosis based on whole slide images (WSIs). Recent advancements in MIL algorithms have yielded exceptional performance when the training and test data originate from the same domain, such as WSIs obtained from the same hospital. However, this paper reveals a performance deterioration of MIL models when tested on an out-of-domain test set, exemplified by WSIs sourced from a novel hospital. To address this challenge, this paper introduces the Retrieval-AugMented MIL (RAM-MIL) framework, which integrates Optimal Transport (OT) as the distance metric for nearest neighbor retrieval. The development of RAM-MIL is driven by two key insights. First, a theoretical discovery indicates that reducing the input's intrinsic dimension can minimize the approximation error in attention-based MIL. Second, previous studies highlight a link between input intrinsic dimension and the feature merging process with the retrieved data. Empirical evaluations conducted on WSI classification demonstrate that the proposed RAM-MIL framework achieves state-of-the-art performance in both in-domain scenarios, where the training and retrieval data are in the same domain, and more crucially, in out-of-domain scenarios, where the (unlabeled) retrieval data originates from a different domain. Furthermore, the use of the transportation matrix derived from OT renders the retrieval results interpretable at the instance level, in contrast to the vanilla $l_2$ distance, and allows for visualization for human experts. *Code can be found at \url{https://github.com/ralphc1212/ram-mil*.</details>

### Revealing the unseen: Benchmarking video action recognition under occlusion

**Authors:** Shresth Grover, Vibhav Vineet, Yogesh Rawat

**<details><summary>Abstract**</summary>

In this work, we study the effect of occlusion on video action recognition. Tofacilitate this study, we propose three benchmark datasets and experiment withseven different video action recognition models. These datasets include two synthetic benchmarks, UCF-101-O and K-400-O, which enabled understanding the effects of fundamental properties of occlusion via controlled experiments. We also propose a real-world occlusion dataset, UCF-101-Y-OCC, which helps in further validating the findings of this study. We find several interesting insights such as 1) transformers are more robust than CNN counterparts, 2) pretraining make modelsrobust against occlusions, and 3) augmentation helps, but does not generalize well to real-world occlusions. In addition, we propose a simple transformer based compositional model, termed as CTx-Net, which generalizes well under this distribution shift. We observe that CTx-Net outperforms models which are trained using occlusions as augmentation, performing significantly better under natural occlusions. We believe this benchmark will open up interesting future research in robust video action recognition</details>

### Revisiting Adversarial Robustness Distillation from the Perspective of Robust Fairness

**Authors:** Xinli Yue, Mou Ningping, Qian Wang, Lingchen Zhao

**<details><summary>Abstract**</summary>

Adversarial Robustness Distillation (ARD) aims to transfer the robustness of large teacher models to small student models, facilitating the attainment of robust performance on resource-limited devices. However, existing research on ARD primarily focuses on the overall robustness of student models, overlooking the crucial aspect of $	extit{robust fairness}$. Specifically, these models may demonstrate strong robustness on some classes of data while exhibiting high vulnerability on other classes. Unfortunately, the "buckets effect" implies that the robustness of the deployed model depends on the classes with the lowest level of robustness. In this paper, we first investigate the inheritance of robust fairness during ARD and reveal that student models only partially inherit robust fairness from teacher models. We further validate this issue through fine-grained experiments with various model capacities and find that it may arise due to the gap in capacity between teacher and student models, as well as the existing methods treating each class equally during distillation. Based on these observations, we propose $	extbf{Fair}$ $	extbf{A}$dversarial $	extbf{R}$obustness $	extbf{D}$istillation (Fair-ARD), a novel framework for enhancing the robust fairness of student models by increasing the weights of difficult classes, and design a geometric perspective-based method to quantify the difficulty of different classes for determining the weights. Extensive experiments show that Fair-ARD surpasses both state-of-the-art ARD methods and existing robust fairness algorithms in terms of robust fairness (e.g., the worst-class robustness under AutoAttack is improved by at most 12.3\% and 5.3\% using ResNet18 on CIFAR10, respectively), while also slightly improving overall robustness. Our code is available at: [https://github.com/NISP-official/Fair-ARD](https://github.com/NISP-official/Fair-ARD).</details>

### Revisiting Evaluation Metrics for Semantic Segmentation: Optimization and Evaluation of Fine-grained Intersection over Union

**Authors:** Zifu Wang, Maxim Berman, Amal Rannen-Triki, Philip Torr, Devis Tuia, Tinne Tuytelaars, Luc V Gool, Jiaqian Yu, Matthew Blaschko

**<details><summary>Abstract**</summary>

Semantic segmentation datasets often exhibit two types of imbalance: 	extit{class imbalance}, where some classes appear more frequently than others and 	extit{size imbalance}, where some objects occupy more pixels than others. This causes traditional evaluation metrics to be biased towards 	extit{majority classes} (e.g. overall pixel-wise accuracy) and 	extit{large objects} (e.g. mean pixel-wise accuracy and per-dataset mean intersection over union). To address these shortcomings, we propose the use of fine-grained mIoUs along with corresponding worst-case metrics, thereby offering a more holistic evaluation of segmentation techniques. These fine-grained metrics offer less bias towards large objects, richer statistical information, and valuable insights into model and dataset auditing. Furthermore, we undertake an extensive benchmark study, where we train and evaluate 15 modern neural networks with the proposed metrics on 12 diverse natural and aerial segmentation datasets. Our benchmark study highlights the necessity of not basing evaluations on a single metric and confirms that fine-grained mIoUs reduce the bias towards large objects. Moreover, we identify the crucial role played by architecture designs and loss functions, which lead to best practices in optimizing fine-grained metrics. The code is available at \href{https://github.com/zifuwanggg/JDTLosses}{https://github.com/zifuwanggg/JDTLosses}.</details>

### Revisiting the Evaluation of Image Synthesis with GANs

**Authors:** mengping yang, Ceyuan Yang, Yichi Zhang, Qingyan Bai, Yujun Shen, Bo Dai

**<details><summary>Abstract**</summary>

A good metric, which promises a reliable comparison between solutions, is essential for any well-defined task. Unlike most vision tasks that have per-sample ground-truth, image synthesis tasks target generating unseen data and hence are usually evaluated through a distributional distance between one set of real samples and another set of generated samples. This study presents an empirical investigation into the evaluation of synthesis performance, with generative adversarial networks (GANs) as a representative of generative models. In particular, we make in-depth analyses of various factors, including how to represent a data point in the representation space, how to calculate a fair distance using selected samples, and how many instances to use from each set. Extensive experiments conducted on multiple datasets and settings reveal several important findings. Firstly, a group of models that include both CNN-based and ViT-based architectures serve as reliable and robust feature extractors for measurement evaluation. Secondly, Centered Kernel Alignment (CKA) provides a better comparison across various extractors and hierarchical layers in one model. Finally, CKA is more sample-efficient and enjoys better agreement with human judgment in characterizing the similarity between two internal data correlations. These findings contribute to the development of a new measurement system, which enables a consistent and reliable re-evaluation of current state-of-the-art generative models.</details>

### Revisiting the Minimalist Approach to Offline Reinforcement Learning

**Authors:** Denis Tarasov, Vladislav Kurenkov, Alexander Nikulin, Sergey Kolesnikov

**<details><summary>Abstract**</summary>

Recent years have witnessed significant advancements in offline reinforcement learning (RL), resulting in the development of numerous algorithms with varying degrees of complexity. While these algorithms have led to noteworthy improvements, many incorporate seemingly minor design choices that impact their effectiveness beyond core algorithmic advances. However, the effect of these design choices on established baselines remains understudied. In this work, we aim to bridge this gap by conducting a retrospective analysis of recent works in offline RL and propose ReBRAC, a minimalistic algorithm that integrates such design elements built on top of the TD3+BC method. We evaluate ReBRAC on 51 datasets with both proprioceptive and visual state spaces using D4RL and V-D4RL benchmarks, demonstrating its state-of-the-art performance among ensemble-free methods in both offline and offline-to-online settings. To further illustrate the efficacy of these design choices, we perform a large-scale ablation study and hyperparameter sensitivity analysis on the scale of thousands of experiments.</details>

### [Spotlight] Rewiring Neurons in Non-Stationary Environments

**Authors:** Zhicheng Sun, Yadong Mu

**<details><summary>Abstract**</summary>

The human brain rewires itself for neuroplasticity in the presence of new tasks. We are inspired to harness this key process in continual reinforcement learning, prioritizing adaptation to non-stationary environments. In distinction to existing rewiring approaches that rely on pruning or dynamic routing, which may limit network capacity and plasticity, this work presents a novel rewiring scheme by permuting hidden neurons. Specifically, the neuron permutation is parameterized to be end-to-end learnable and can rearrange all available synapses to explore a large span of weight space, thereby promoting adaptivity. In addition, we introduce two main designs to steer the rewiring process in continual reinforcement learning: first, a multi-mode rewiring strategy is proposed which diversifies the policy and encourages exploration when encountering new environments. Secondly, to ensure stability on history tasks, the network is devised to cache each learned wiring while subtly updating its weights, allowing for retrospective recovery of any previous state appropriate for the task. Meanwhile, an alignment mechanism is curated to achieve better plasticity-stability tradeoff by jointly optimizing cached wirings and weights. Our proposed method is comprehensively evaluated on 18 continual reinforcement learning scenarios ranging from locomotion to manipulation, demonstrating its advantages over state-of-the-art competitors in performance-efficiency tradeoffs. Code is available at https://github.com/feifeiobama/RewireNeuron.</details>

### Riemannian SAM: Sharpness-Aware Minimization on Riemannian Manifolds

**Authors:** Jihun Yun, Eunho Yang

**<details><summary>Abstract**</summary>

Contemporary advances in the field of deep learning have embarked upon an exploration of the underlying geometric properties of data, thus encouraging the investigation of techniques that consider general manifolds, for example, hyperbolic or orthogonal neural networks. However, the optimization algorithms for training such geometric deep learning models still remain highly under-explored. In this paper, we introduce Riemannian SAM by generalizing conventional Euclidean SAM to Riemannian manifolds. We successfully formulate the sharpness-aware minimization on Riemannian manifolds, leading to one of a novel instantiation, Lorentz SAM. In addition, SAM variants proposed in previous studies such as Fisher SAM can be derived as special examples under our Riemannian SAM framework. We provide the convergence analysis of Riemannian SAM under a less aggressively decaying ascent learning rate than Euclidean SAM. Our analysis serves as a theoretically sound contribution encompassing a diverse range of manifolds, also providing the guarantees for SAM variants such as Fisher SAM, whose convergence analyses are absent. Lastly, we illustrate the superiority of Riemannian SAM in terms of generalization over previous Riemannian optimization algorithms through experiments on knowledge graph completion and machine translation tasks.</details>

### RoboHive: A Unified Framework for Robot Learning

**Authors:** Vikash Kumar, Rutav Shah, Gaoyue Zhou, Vincent Moens, Vittorio Caggiano, Abhishek Gupta, Aravind Rajeswaran

**<details><summary>Abstract**</summary>

We present RoboHive, a comprehensive software platform and ecosystem for research in the field of Robot Learning and Embodied Artificial Intelligence. Our platform encompasses a diverse range of pre-existing and novel environments, including dexterous manipulation with the Shadow Hand, whole-arm manipulation tasks with Franka and Fetch robots, quadruped locomotion, among others. Included environments are organized within and cover multiple domains such as hand manipulation, locomotion, multi-task, multi-agent, muscles, etc. In comparison to prior works, RoboHive offers a streamlined and unified task interface taking dependency on only a minimal set of well-maintained packages, features tasks with high physics fidelity and rich visual diversity, and supports common hardware drivers for real-world deployment. The unified interface of RoboHive offers a convenient and accessible abstraction for algorithmic research in imitation, reinforcement, multi-task, and hierarchical learning. Furthermore, RoboHive includes expert demonstrations and baseline results for most environments, providing a standard for benchmarking and comparisons. Details: https://sites.google.com/view/robohive</details>

### Robust Bayesian Satisficing

**Authors:** Artun Saday, Y. Cahit Yıldırım, Cem Tekin

**<details><summary>Abstract**</summary>

Distributional shifts pose a significant challenge to achieving robustness in contemporary machine learning. To overcome this challenge, robust satisficing (RS) seeks a robust solution to an unspecified distributional shift while achieving a utility above a desired threshold. This paper focuses on the problem of RS in contextual Bayesian optimization when there is a discrepancy between the true and reference distributions of the context. We propose a novel robust Bayesian satisficing algorithm called RoBOS for noisy black-box optimization. Our algorithm guarantees sublinear lenient regret under certain assumptions on the amount of distribution shift. In addition, we define a weaker notion of regret called robust satisficing regret, in which our algorithm achieves a sublinear upper bound independent of the amount of distribution shift. To demonstrate the effectiveness of our method, we apply it to various learning problems and compare it to other approaches, such as distributionally robust optimization.</details>

### Robust Data Valuation with Weighted Banzhaf Values

**Authors:** Weida Li, Yaoliang Yu

**<details><summary>Abstract**</summary>

Data valuation, a principled way to rank the importance of each training datum, has become increasingly important. However, existing value-based approaches (e.g., Shapley) are known to suffer from the stochasticity inherent in utility functions that render consistent and reliable ranking difficult. Recently, Wang and Jia (2023) proposed the noise-structure-agnostic framework to advocate the Banzhaf value for its robustness against such stochasticity as it achieves the largest safe margin among many alternatives. Surprisingly, our empirical study shows that the Banzhaf value is not always the most robust when compared with a broader family: weighted Banzhaf values. To analyze this scenario, we introduce the concept of Kronecker noise to parameterize stochasticity, through which we prove that the uniquely robust semi-value, which can be analytically derived from the underlying Kronecker noise, lies in the family of weighted Banzhaf values while minimizing the worst-case entropy. In addition, we adopt the maximum sample reuse principle to design an estimator to efficiently approximate weighted Banzhaf values, and show that it enjoys the best time complexity in terms of achieving an $(\epsilon, \delta)$-approximation. Our theory is verified under both synthetic and authentic noises. For the latter, we fit a Kronecker noise to the inherent stochasticity, which is then plugged in to generate the predicted most robust semi-value. Our study suggests that weighted Banzhaf values are promising when facing undue noises in data valuation.</details>

### [Spotlight] Robust Model Reasoning and Fitting via Dual Sparsity Pursuit

**Authors:** Xingyu Jiang, Jiayi Ma

**<details><summary>Abstract**</summary>

In this paper, we contribute to solving a threefold problem: outlier rejection, true model reasoning and parameter estimation with a unified optimization modeling. To this end, we first pose this task as a sparse subspace recovering problem, to search a maximum of independent bases under an over-embedded data space. Then we convert the objective into a continuous optimization paradigm that estimates sparse solutions for both bases and errors. Wherein a fast and robust solver is proposed to accurately estimate the sparse subspace parameters and error entries, which is implemented by a proximal approximation method under the alternating optimization framework with the ``optimal'' sub-gradient descent. Extensive experiments regarding known and unknown model fitting on synthetic and challenging real datasets have demonstrated the superiority of our method against the state-of-the-art. We also apply our method to multi-class multi-model fitting and loop closure detection, and achieve promising results both in accuracy and efficiency. Code is released at: https://github.com/StaRainJ/DSP.</details>

### SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model

**Authors:** Di Wang, Jing Zhang, Bo Du, Minqiang Xu, Lin Liu, Dacheng Tao, Liangpei Zhang

**<details><summary>Abstract**</summary>

The success of the Segment Anything Model (SAM) demonstrates the significance of data-centric machine learning. However, due to the difficulties and high costs associated with annotating Remote Sensing (RS) images, a large amount of valuable RS data remains unlabeled, particularly at the pixel level. In this study, we leverage SAM and existing RS object detection datasets to develop an efficient pipeline for generating a large-scale RS segmentation dataset, dubbed SAMRS. SAMRS totally possesses 105,090 images and 1,668,241 instances, surpassing existing high-resolution RS segmentation datasets in size by several orders of magnitude. It provides object category, location, and instance information that can be used for semantic segmentation, instance segmentation, and object detection, either individually or in combination. We also provide a comprehensive analysis of SAMRS from various aspects. Moreover, preliminary experiments highlight the importance of conducting segmentation pre-training with SAMRS to address task discrepancies and alleviate the limitations posed by limited training data during fine-tuning. The code and dataset will be available at https://github.com/ViTAE-Transformer/SAMRS</details>

### SARAMIS: Simulation Assets for Robotic Assisted and Minimally Invasive Surgery

**Authors:** Nina Montana-Brown, Shaheer U. Saeed, Ahmed Abdulaal, Thomas Dowrick, Yakup Kilic, Sophie Wilkinson, Jack Gao, Meghavi Mashar, Chloe He, Alkisti Stavropoulou, Emma Thomson, Zachary MC Baum, Simone Foti, Brian Davidson, Yipeng Hu, Matthew Clarkson

**<details><summary>Abstract**</summary>

Minimally-invasive surgery (MIS) and robot-assisted minimally invasive (RAMIS) surgery offer well-documented benefits to patients such as reduced post-operative pain and shorter hospital stays.However, the automation of MIS and RAMIS through the use of AI has been slow due to difficulties in data acquisition and curation, partially caused by the ethical considerations of training, testing and deploying AI models in medical environments.We introduce 	exttt{SARAMIS}, the first large-scale dataset of anatomically derived 3D rendering assets of the human abdominal anatomy.Using previously existing, open-source CT datasets of the human anatomy, we derive novel 3D meshes, tetrahedral volumes, textures and diffuse maps for over 104 different anatomical targets in the human body, representing the largest, open-source dataset of 3D rendering assets for synthetic simulation of vision tasks in MIS+RAMIS, increasing the availability of openly available 3D meshes in the literature by three orders of magnitude.We supplement our dataset with a series of GPU-enabled rendering environments, which can be used to generate datasets for realistic MIS/RAMIS tasks.Finally, we present an example of the use of 	exttt{SARAMIS} assets for an autonomous navigation task in colonoscopy from CT abdomen-pelvis scans for the first time in the literature.	exttt{SARAMIS} is publically made available at https://github.com/NMontanaBrown/saramis/, with assets released under a CC-BY-NC-SA license.</details>

### SEVA: Leveraging sketches to evaluate alignment between human and machine visual abstraction

**Authors:** Kushin Mukherjee, Holly Huey, Xuanchen Lu, Yael Vinker, Rio Aguina-Kang, Ariel Shamir, Judith Fan

**<details><summary>Abstract**</summary>

Sketching is a powerful tool for creating abstract images that are sparse but meaningful. Sketch understanding poses fundamental challenges for general-purpose vision algorithms because it requires robustness to the sparsity of sketches relative to natural visual inputs and because it demands tolerance for semantic ambiguity, as sketches can reliably evoke multiple meanings. While current vision algorithms have achieved high performance on a variety of visual tasks, it remains unclear to what extent they understand sketches in a human-like way. Here we introduce $\texttt{SEVA}$, a new benchmark dataset containing approximately 90K human-generated sketches of 128 object concepts produced under different time constraints, and thus systematically varying in sparsity. We evaluated a suite of state-of-the-art vision algorithms on their ability to correctly identify the target concept depicted in these sketches and to generate responses that are strongly aligned with human response patterns on the same sketch recognition task. We found that vision algorithms that better predicted human sketch recognition performance also better approximated human uncertainty about sketch meaning, but there remains a sizable gap between model and human response patterns. To explore the potential of models that emulate human visual abstraction in generative tasks, we conducted further evaluations of a recently developed sketch generation algorithm (Vinker et al., 2022) capable of generating sketches that vary in sparsity. We hope that public release of this dataset and evaluation protocol will catalyze progress towards algorithms with enhanced capacities for human-like visual abstraction.</details>

### SG×P : A Sorghum Genotype × Phenotype Prediction Dataset and Benchmark

**Authors:** Zeyu Zhang, Robert Pless, Nadia Shakoor, Austin Carnahan, Abby Stylianou

**<details><summary>Abstract**</summary>

Large scale field-phenotyping approaches have the potential to solve important questions about the relationship of plant genotype to plant phenotype. Computational approaches to measuring the phenotype (the observable plant features) are required to address the problem at a large scale, but machine learning approaches to extract phenotypes from sensor data have been hampered by limited access to (a) sufficiently large, organized multi-sensor datasets, (b) field trials that have a large scale and significant number of genotypes, (c) full genetic sequencing of those phenotypes, and (d) datasets sufficiently organized so that algorithm centered researchers can directly address the real biological problems. To address this, we present SGxP, a novel benchmark dataset from a large-scale field trial consisting of the complete genotype of over 300 sorghum varieties, and time sequences of imagery from several field plots growing each variety, taken with RGB and laser 3D scanner imaging. To lower the barrier to entry and facilitate further developments, we provide a set of well organized, multi-sensor imagery and corresponding genomic data. We implement baseline deep learning based phenotyping approaches to create baseline results for individual sensors and multi-sensor fusion for detecting genetic mutations with known impacts. We also provide and support an open-ended challenge by identifying thousands of genetic mutations whose phenotypic impacts are currently unknown. A web interface for machine learning researchers and practitioners to share approaches, visualizations and hypotheses supports engagement with plant biologists to further the understanding of the sorghum genotype x phenotype relationship. The full dataset, leaderboard (including baseline results) and discussion forums can be found at http://sorghumsnpbenchmark.com.</details>

### SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning

**Authors:** Benjamin Ellis, Jonathan Cook, Skander Moalla, Mikayel Samvelyan, Mingfei Sun, Anuj Mahajan, Jakob Foerster, Shimon Whiteson

**<details><summary>Abstract**</summary>

The availability of challenging benchmarks has played a key role in the recent progress of machine learning. In cooperative multi-agent reinforcement learning, the StarCraft Multi-Agent Challenge (SMAC) has become a popular testbed for centralised training with decentralised execution. However, after years of sustained improvement on SMAC, algorithms now achieve near-perfect performance. In this work, we conduct new analysis demonstrating that SMAC lacks the stochasticity and partial observability to require complex *closed-loop* policies. In particular, we show that an *open-loop* policy conditioned only on the timestep can achieve non-trivial win rates for many SMAC scenarios. To address this limitation, we introduce SMACv2, a new version of the benchmark where scenarios are procedurally generated and require agents to generalise to previously unseen settings (from the same distribution) during evaluation. We also introduce the extended partial observability challenge (EPO), which augments SMACv2 to ensure meaningful partial observability. We show that these changes ensure the benchmarkrequires the use of *closed-loop* policies. We evaluate state-of-the-art algorithms on SMACv2 and show that it presents significant challenges not present in the original benchmark. Our analysis illustrates that SMACv2 addresses the discovered deficiencies of SMAC and can help benchmark the next generation of MARL methods. Videos of training are available on our [website](https://sites.google.com/view/smacv2).</details>

### SODA: Robust Training of Test-Time Data Adaptors

**Authors:** Zige Wang, Yonggang Zhang, Zhen Fang, Long Lan, Wenjing Yang, Bo Han

**<details><summary>Abstract**</summary>

Adapting models deployed to test distributions can mitigate the performance degradation caused by distribution shifts. However, privacy concerns may render model parameters inaccessible. One promising approach involves utilizing zeroth-order optimization (ZOO) to train a data adaptor to adapt the test data to fit the deployed models. Nevertheless, the data adaptor trained with ZOO typically brings restricted improvements due to the potential corruption of data features caused by the data adaptor. To address this issue, we revisit ZOO in the context of test-time data adaptation. We find that the issue directly stems from the unreliable estimation of the gradients used to optimize the data adaptor, which is inherently due to the unreliable nature of the pseudo-labels assigned to the test data. Based on this observation, we propose pseudo-label-robust data adaptation (SODA) to improve the performance of data adaptation. Specifically, SODA leverages high-confidence predicted labels as reliable labels to optimize the data adaptor with ZOO for label prediction. For data with low-confidence predictions, SODA encourages the adaptor to preserve data information to mitigate data corruption. Empirical results indicate that SODA can significantly enhance the performance of deployed models in the presence of distribution shifts without requiring access to model parameters.</details>

### SPACE: Single-round Participant Amalgamation for Contribution Evaluation in Federated Learning

**Authors:** Yi-Chung Chen, Hsi-Wen Chen, Shun-Gui Wang, Ming-syan Chen

**<details><summary>Abstract**</summary>

The evaluation of participant contribution in federated learning (FL) has recently gained significant attention due to its applicability in various domains, such as incentive mechanisms, robustness enhancement, and client selection. Previous approaches have predominantly relied on the widely adopted Shapley value for participant evaluation. However, the computation of the Shapley value is expensive, despite using techniques like gradient-based model reconstruction and truncating unnecessary evaluations. Therefore, we present an efficient approach called Single-round Participants Amalgamation for Contribution Evaluation (SPACE). SPACE incorporates two novel components, namely Federated Knowledge Amalgamation and Prototype-based Model Evaluation to reduce the evaluation effort by eliminating the dependence on the size of the validation set and enabling participant evaluation within a single communication round. Experimental results demonstrate that SPACE outperforms state-of-the-art methods in terms of both running time and Pearson’s Correlation Coefficient (PCC). Furthermore, extensive experiments conducted on applications, client reweighting, and client selection highlight the effectiveness of SPACE. The code is available at https://github.com/culiver/SPACE.</details>

### STARSS23: An Audio-Visual Dataset of Spatial Recordings of Real Scenes with Spatiotemporal Annotations of Sound Events

**Authors:** Kazuki Shimada, Archontis Politis, Parthasaarathy Sudarsanam, Daniel A. Krause, Kengo Uchida, Sharath Adavanne, Aapo Hakala, Yuichiro Koyama, Naoya Takahashi, Shusuke Takahashi, Tuomas Virtanen, Yuki Mitsufuji

**<details><summary>Abstract**</summary>

While direction of arrival (DOA) of sound events is generally estimated from multichannel audio data recorded in a microphone array, sound events usually derive from visually perceptible source objects, e.g., sounds of footsteps come from the feet of a walker. This paper proposes an audio-visual sound event localization and detection (SELD) task, which uses multichannel audio and video information to estimate the temporal activation and DOA of target sound events. Audio-visual SELD systems can detect and localize sound events using signals from a microphone array and audio-visual correspondence. We also introduce an audio-visual dataset, Sony-TAu Realistic Spatial Soundscapes 2023 (STARSS23), which consists of multichannel audio data recorded with a microphone array, video data, and spatiotemporal annotation of sound events. Sound scenes in STARSS23 are recorded with instructions, which guide recording participants to ensure adequate activity and occurrences of sound events. STARSS23 also serves human-annotated temporal activation labels and human-confirmed DOA labels, which are based on tracking results of a motion capture system. Our benchmark results demonstrate the benefits of using visual object positions in audio-visual SELD tasks. The data is available at https://zenodo.org/record/7880637.</details>

### SafeDICE: Offline Safe Imitation Learning with Non-Preferred Demonstrations

**Authors:** Youngsoo Jang, Geon-Hyeong Kim, Jongmin Lee, Sungryull Sohn, Byoungjip Kim, Honglak Lee, Moontae Lee

**<details><summary>Abstract**</summary>

We consider offline safe imitation learning (IL), where the agent aims to learn the safe policy that mimics preferred behavior while avoiding non-preferred behavior from non-preferred demonstrations and unlabeled demonstrations. This problem setting corresponds to various real-world scenarios, where satisfying safety constraints is more important than maximizing the expected return. However, it is very challenging to learn the policy to avoid constraint-violating (i.e. non-preferred) behavior, as opposed to standard imitation learning which learns the policy to mimic given demonstrations. In this paper, we present a hyperparameter-free offline safe IL algorithm, SafeDICE, that learns safe policy by leveraging the non-preferred demonstrations in the space of stationary distributions. Our algorithm directly estimates the stationary distribution corrections of the policy that imitate the demonstrations excluding the non-preferred behavior. In the experiments, we demonstrate that our algorithm learns a more safe policy that satisfies the cost constraint without degrading the reward performance, compared to baseline algorithms.</details>

### Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark

**Authors:** Jiaming Ji, Borong Zhang, Jiayi Zhou, Xuehai Pan, Weidong Huang, Ruiyang Sun, Yiran Geng, Yifan Zhong, Josef Dai, Yaodong Yang

**<details><summary>Abstract**</summary>

Artificial intelligence (AI) systems possess significant potential to drive societal progress. However, their deployment often faces obstacles due to substantial safety concerns. Safe reinforcement learning (SafeRL) emerges as a solution to optimize policies while simultaneously adhering to multiple constraints, thereby addressing the challenge of integrating reinforcement learning in safety-critical scenarios. In this paper, we present an environment suite called Safety-Gymnasium, which encompasses safety-critical tasks in both single and multi-agent scenarios, accepting vector and vision-only input. Additionally, we offer a library of algorithms named Safe Policy Optimization (SafePO), comprising 16 state-of-the-art SafeRL algorithms. This comprehensive library can serve as a validation tool for the research community. By introducing this benchmark, we aim to facilitate the evaluation and comparison of safety performance, thus fostering the development of reinforcement learning for safer, more reliable, and responsible real-world applications. The website of this project can be accessed at https://sites.google.com/view/safety-gymnasium.</details>

### Sample Complexity Bounds for Score-Matching: Causal Discovery and Generative Modeling

**Authors:** Zhenyu Zhu, Francesco Locatello, Volkan Cevher

**<details><summary>Abstract**</summary>

This paper provides statistical sample complexity bounds for score-matching and its applications in causal discovery. We demonstrate that accurate estimation of the score function is achievable by training a standard deep ReLU neural network using stochastic gradient descent. We establish bounds on the error rate of recovering causal relationships using the score-matching-based causal discovery method of Rolland et al. [2022], assuming a sufficiently good estimation of the score function. Finally, we analyze the upper bound of score-matching estimation within the score-based generative modeling, which has been applied for causal discovery but is also of independent interest within the domain of generative models.</details>

### Sample-efficient Multi-objective Molecular Optimization with GFlowNets

**Authors:** Yiheng Zhu, Jialu Wu, Chaowen Hu, Jiahuan Yan, kim hsieh, Tingjun Hou, Jian Wu

**<details><summary>Abstract**</summary>

Many crucial scientific problems involve designing novel molecules with desired properties, which can be formulated as a black-box optimization problem over the *discrete* chemical space. In practice, multiple conflicting objectives and costly evaluations (e.g., wet-lab experiments) make the *diversity* of candidates paramount. Computational methods have achieved initial success but still struggle with considering diversity in both objective and search space. To fill this gap, we propose a multi-objective Bayesian optimization (MOBO) algorithm leveraging the hypernetwork-based GFlowNets (HN-GFN) as an acquisition function optimizer, with the purpose of sampling a diverse batch of candidate molecular graphs from an approximate Pareto front. Using a single preference-conditioned hypernetwork, HN-GFN learns to explore various trade-offs between objectives. We further propose a hindsight-like off-policy strategy to share high-performing molecules among different preferences in order to speed up learning for HN-GFN. We empirically illustrate that HN-GFN has adequate capacity to generalize over preferences. Moreover, experiments in various real-world MOBO settings demonstrate that our framework predominantly outperforms existing methods in terms of candidate quality and sample efficiency. The code is available at https://github.com/violet-sto/HN-GFN.</details>

### Sampling weights of deep neural networks

**Authors:** Erik L Bolager, Iryna Burak, Chinmay Datar, Qing Sun, Felix Dietrich

**<details><summary>Abstract**</summary>

We introduce a probability distribution, combined with an efficient sampling algorithm, for weights and biases of fully-connected neural networks. In a supervised learning context, no iterative optimization or gradient computations of internal network parameters are needed to obtain a trained network. The sampling is based on the idea of random feature models. However, instead of a data-agnostic distribution, e.g., a normal distribution, we use both the input and the output training data to sample shallow and deep networks. We prove that sampled networks are universal approximators. For Barron functions, we show that the $L^2$-approximation error of sampled shallow networks decreases with the square root of the number of neurons. Our sampling scheme is invariant to rigid body transformations and scaling of the input data, which implies many popular pre-processing techniques are not required. In numerical experiments, we demonstrate that sampled networks achieve accuracy comparable to iteratively trained ones, but can be constructed orders of magnitude faster. Our test cases involve a classification benchmark from OpenML, sampling of neural operators to represent maps in function spaces, and transfer learning using well-known architectures.</details>

### SatBird: a Dataset for Bird Species Distribution Modeling using Remote Sensing and Citizen Science Data

**Authors:** Mélisande Teng, Amna Elmustafa, Benjamin Akera, Yoshua Bengio, Hager Radi, Hugo Larochelle, David Rolnick

**<details><summary>Abstract**</summary>

Biodiversity is declining at an unprecedented rate, impacting ecosystem services necessary to ensure food, water, and human health and well-being. Understanding the distribution of species and their habitats is crucial for conservation policy planning. However, traditional methods in ecology for species distribution models (SDMs) generally focus either on narrow sets of species or narrow geographical areas and there remain significant knowledge gaps about the distribution of species. A major reason for this is the limited availability of data traditionally used, due to the prohibitive amount of effort and expertise required for traditional field monitoring. The wide availability of remote sensing data and the growing adoption of citizen science tools to collect species observations data at low cost offer an opportunity for improving biodiversity monitoring and enabling the modelling of complex ecosystems. We introduce a novel task for mapping bird species to their habitats by predicting species encounter rates from satellite images, and present SatBird, a satellite dataset of locations in the USA with labels derived from presence-absence observation data from the citizen science database eBird, considering summer (breeding) and winter seasons. We also provide a dataset in Kenya representing low-data regimes. We additionally provide environmental data and species range maps for each location.  We benchmark a set of baselines on our dataset, including SOTA models for remote sensing tasks. SatBird opens up possibilities for scalably modelling properties of ecosystems worldwide.</details>

### Scalable 3D Captioning with Pretrained Models

**Authors:** Tiange Luo, Chris Rockwell, Honglak Lee, Justin Johnson

**<details><summary>Abstract**</summary>

We introduce Cap3D, an automatic approach for generating descriptive text for 3D objects. This approach utilizes pretrained models from image captioning, image-text alignment, and LLM to consolidate captions from multiple views of a 3D asset, completely side-stepping the time-consuming and costly process of manual annotation. We apply Cap3D to the recently introduced large-scale 3D dataset, Objaverse, resulting in 660k 3D-text pairs. Our evaluation, conducted using 41k human annotations from the same dataset, demonstrates that Cap3D surpasses human-authored descriptions in terms of quality, cost, and speed. Through effective prompt engineering, Cap3D rivals human performance in generating geometric descriptions on 17k collected annotations from the ABO dataset. Finally, we finetune Text-to-3D models on Cap3D and human captions, and show Cap3D outperforms; and benchmark the SOTA including Point·E, Shape·E, and DreamFusion.</details>

### Scalable Fair Influence Maximization

**Authors:** Xiaobin Rui, Zhixiao Wang, Jiayu Zhao, Lichao Sun, Wei Chen

**<details><summary>Abstract**</summary>

Given a graph $G$, a community structure $\mathcal{C}$, and a budget $k$, the fair influence maximization problem aims to select a seed set $S$ ($|S|\leq k$) that maximizes the influence spread while narrowing the influence gap between different communities. While various fairness notions exist, the welfare fairness notion, which balances fairness level and influence spread, has shown promising effectiveness. However, the lack of efficient algorithms for optimizing the welfare fairness objective function restricts its application to small-scale networks with only a few hundred nodes. In this paper, we adopt the objective function of welfare fairness to maximize the exponentially weighted summation over the influenced fraction of all communities. We first introduce an unbiased estimator for the fractional power of the arithmetic mean. Then, by adapting the reverse influence sampling (RIS) approach, we convert the optimization problem to a weighted maximum coverage problem. We also analyze the number of reverse reachable sets needed to approximate the fair influence at a high probability. Further, we present an efficient algorithm that guarantees $1-1/e - \varepsilon$ approximation.</details>

### Scalable Membership Inference Attacks via Quantile Regression

**Authors:** Martin Bertran, Shuai Tang, Aaron Roth, Michael Kearns, Jamie Morgenstern, Steven Wu

**<details><summary>Abstract**</summary>

Membership inference attacks are designed to determine, using black box access to trained models, whether a particular example was used in training or not. Membership inference can be formalized as a hypothesis testing problem. The most effective existing attacks estimate the distribution of some test statistic (usually the model's confidence on the true label) on points that were (and were not) used in training by training many \emph{shadow models}---i.e. models of the same architecture as the model being attacked, trained on a random subsample of data. While effective, these attacks are extremely computationally expensive, especially when the model under attack is large. \footnotetext[0]{Martin and Shuai are the lead authors, and other authors are ordered alphabetically. \{maberlop,shuat\}@amazon.com}We introduce a new class of attacks based on performing quantile regression on the distribution of confidence scores induced by the model under attack on points that are not used in training. We show that our method is competitive with state-of-the-art shadow model attacks, while requiring substantially less compute because our attack requires training only a single model. Moreover, unlike shadow model attacks, our proposed attack does not require any knowledge of the architecture of the model under attack and is therefore truly ``black-box". We show the efficacy of this approach in an extensive series of experiments on various datasets and model architectures. Our code is available at \href{https://github.com/amazon-science/quantile-mia}{github.com/amazon-science/quantile-mia.}</details>

### Scientific Document Retrieval using Multi-level Aspect-based Queries

**Authors:** Jianyou (Andre) Wang, Kaicheng Wang, Xiaoyue Wang, Prudhviraj Naidu, Leon Bergen, Ramamohan Paturi

**<details><summary>Abstract**</summary>

In scientific research, the ability to effectively retrieve relevant documents based on complex, multifaceted queries is critical. Existing evaluation datasets for this task are limited, primarily due to the high costs and effort required to annotate resources that effectively represent complex queries. To address this, we propose a novel task, $	extbf{S}$cientific $	extbf{Do}$cument $	extbf{R}$etrieval using $	extbf{M}$ulti-level $	extbf{A}$spect-based qu$	extbf{E}$ries (DORIS-MAE), which is designed to handle the complex nature of user queries in scientific research. We developed a benchmark dataset within the field of computer science, consisting of 100 human-authored complex query cases. For each complex query, we assembled a collection of 100 relevant documents and produced annotated relevance scores for ranking them. Recognizing the significant labor of expert annotation, we also introduce Anno-GPT, a scalable framework for evaluating the viability of Large Language Models (LLMs) such as ChatGPT-3.5 for expert-level dataset annotation tasks. The application of Anno-GPT to annotate the DORIS-MAE dataset resulted in a 500x reduction in cost, without compromising quality. Furthermore, due to the multi-tiered structure of these complex queries, our DORIS-MAE dataset can be extended to over 4,000 sub-query test cases without requiring additional annotation. We evaluated 17 recent retrieval methods on DORIS-MAE, observing notable performance drops compared to traditional datasets. This highlights DORIS-MAE's challenges and the need for better approaches to handle complex, multifaceted queries in scientific research. Our dataset and codebase are available at https://github.com/Real-Doris-Mae/Doris-Mae-Dataset .</details>

### Searching for Optimal Per-Coordinate Step-sizes with Multidimensional Backtracking

**Authors:** Frederik Kunstner, Victor Sanches Portella, Mark Schmidt, Nicholas Harvey

**<details><summary>Abstract**</summary>

The backtracking line-search is an effective technique to automatically tune the step-size in smooth optimization. It guarantees similar performance to using the theoretically optimal step-size. Many approaches have been developed to instead tune per-coordinate step-sizes, also known as diagonal preconditioners, but none of the existing methods are provably competitive with the optimal per-coordinate step-sizes. We propose multidimensional backtracking, an extension of the backtracking line-search to find good diagonal preconditioners for smooth convex problems. Our key insight is that the gradient with respect to the step-sizes, also known as hyper-gradients, yields separating hyperplanes that let us search for good preconditioners using cutting-plane methods. As black-box cutting-plane approaches like the ellipsoid method are computationally prohibitive, we develop an efficient algorithm tailored to our setting. Multidimensional backtracking is provably competitive with the best diagonal preconditioner and requires no manual tuning.</details>

### Secure Out-of-Distribution Task Generalization with Energy-Based Models

**Authors:** Shengzhuang Chen, Long-Kai Huang, Jonathan Richard Schwarz, Yilun Du, Ying Wei

**<details><summary>Abstract**</summary>

The success of meta-learning on out-of-distribution (OOD) tasks in the wild has proved to be hit-and-miss.To safeguard the generalization capability of the meta-learned prior knowledge to OOD tasks, in particularly safety-critical applications, necessitates detection of an OOD task followed by adaptation of the task towards the prior. Nonetheless, the reliability of estimated uncertainty on OOD tasks by existing Bayesian meta-learning methods is restricted by incomplete coverage of the feature distribution shift and insufficient expressiveness of the meta-learned prior. Besides, they struggle to adapt an OOD task, running parallel to the line of cross-domain task adaptation solutions which are vulnerable to overfitting.To this end, we build a single coherent framework that supports both detection and adaptation of OOD tasks, while remaining compatible with off-the-shelf meta-learning backbones. The proposed Energy-Based Meta-Learning (EBML) framework learns to characterize any arbitrary meta-training task distribution with the composition of two expressive neural-network-based energy functions. We deploy the sum of the two energy functions, being proportional to the joint distribution of a task, as a reliable score for detecting OOD tasks; during meta-testing, we adapt the OOD task to in-distribution tasks by energy minimization.Experiments on four regression and classification datasets  demonstrate the effectiveness of our proposal.</details>

### Seeing is not always believing: Benchmarking Human and Model Perception of AI-Generated Images

**Authors:** Zeyu Lu, Di Huang, LEI BAI, Jingjing Qu, Chengyue Wu, Xihui Liu, Wanli Ouyang

**<details><summary>Abstract**</summary>

Photos serve as a way for humans to record what they experience in their daily lives, and they are often regarded as trustworthy sources of information. However, there is a growing concern that the advancement of artificial intelligence (AI) technology may produce fake photos, which can create confusion and diminish trust in photographs. This study aims to comprehensively evaluate agents for distinguishing state-of-the-art AI-generated visual content. Our study benchmarks both human capability and cutting-edge fake image detection AI algorithms, using a newly collected large-scale fake image dataset Fake2M. In our human perception evaluation, titled HPBench, we discovered that humans struggle significantly to distinguish real photos from AI-generated ones, with a misclassification rate of 38.7\%. Along with this, we conduct the model capability of AI-Generated images detection evaluation MPBench and the top-performing model from MPBench achieves a 13\% failure rate under the same setting used in the human evaluation.We hope that our study can raise awareness of the potential risks of AI-generated images and facilitate further research to prevent the spread of false information. More information can refer to https://github.com/Inf-imagine/Sentry.</details>

### Self-Supervised Reinforcement Learning that Transfers using Random Features

**Authors:** Boyuan Chen, Chuning Zhu, Pulkit Agrawal, Kaiqing Zhang, Abhishek Gupta

**<details><summary>Abstract**</summary>

Model-free reinforcement learning algorithms have exhibited great potential in solving single-task sequential decision-making problems with high-dimensional observations and long horizons, but are known to be hard to generalize across tasks. Model-based RL, on the other hand, learns task-agnostic models of the world that naturally enables transfer across different reward functions, but struggles to scale to complex environments due to the compounding error. To get the best of both worlds, we propose a self-supervised reinforcement learning method that enables the transfer of behaviors across tasks with different rewards, while circumventing the challenges of model-based RL. In particular, we show self-supervised pre-training of model-free reinforcement learning with a number of random features as rewards allows implicit modeling of long-horizon environment dynamics. Then, planning techniques like model-predictive control using these implicit models enable fast adaptation to problems with new reward functions. Our method is self-supervised in that it can be trained on offline datasets without reward labels, but can then be quickly deployed on new tasks. We validate that our proposed method enables transfer across tasks on a variety of manipulation and locomotion domains in simulation, opening the door to generalist decision-making agents.</details>

### Self-Weighted Contrastive Learning among Multiple Views for Mitigating Representation Degeneration

**Authors:** Jie Xu, Shuo Chen, Yazhou Ren, Xiaoshuang Shi, Hengtao Shen, Gang Niu, Xiaofeng Zhu

**<details><summary>Abstract**</summary>

Recently, numerous studies have demonstrated the effectiveness of contrastive learning (CL), which learns feature representations by pulling in positive samples while pushing away negative samples. Many successes of CL lie in that there exists semantic consistency between data augmentations of the same instance. In multi-view scenarios, however, CL might cause representation degeneration when the collected multiple views inherently have inconsistent semantic information or their representations subsequently do not capture sufficient discriminative information. To address this issue, we propose a novel framework called SEM: SElf-weighted Multi-view contrastive learning with reconstruction regularization. Specifically, SEM is a general framework where we propose to first measure the discrepancy between pairwise representations and then minimize the corresponding self-weighted contrastive loss, and thus making SEM adaptively strengthen the useful pairwise views and also weaken the unreliable pairwise views. Meanwhile, we impose a self-supervised reconstruction term to regularize the hidden features of encoders, to assist CL in accessing sufficient discriminative information of data. Experiments on public multi-view datasets verified that SEM can mitigate representation degeneration in existing CL methods and help them achieve significant performance improvements. Ablation studies also demonstrated the effectiveness of SEM with different options of weighting strategies and reconstruction terms.</details>

### Self-supervised video pretraining yields robust and more human-aligned visual representations

**Authors:** Nikhil Parthasarathy, S. M. Ali Eslami, Joao Carreira, Olivier Henaff

**<details><summary>Abstract**</summary>

Humans learn powerful representations of objects and scenes by observing how they evolve over time. Yet, outside of specific tasks that require explicit temporal understanding, static image pretraining remains the dominant paradigm for learning visual foundation models. We question this mismatch, and ask whether video pretraining can yield visual representations that bear the hallmarks of human perception: generalisation across tasks, robustness to perturbations, and consistency with human judgements. To that end we propose a novel procedure for curating videos, and develop a contrastive framework which learns from the complex transformations therein. This simple paradigm for distilling knowledge from videos, called VITO, yields general representations that far outperform prior video pretraining methods on image understanding tasks, and image pretraining methods on video understanding tasks. Moreover, VITO representations are significantly more robust to natural and synthetic deformations than image-, video-, and adversarially-trainedones. Finally, VITO’s predictions are strongly aligned with human judgements, surpassing models that were specifically trained for that purpose. Together, these results suggest that video pretraining could be a simple way of learning unified, robust, and human-aligned representations of the visual world.</details>

### Semantic HELM: A Human-Readable Memory for Reinforcement Learning

**Authors:** Fabian Paischer, Thomas Adler, Markus Hofmarcher, Sepp Hochreiter

**<details><summary>Abstract**</summary>

Reinforcement learning agents deployed in the real world often have to cope with partially observable environments. Therefore, most agents employ memory mechanisms to approximate the state of the environment. Recently, there have been impressive success stories in mastering partially observable environments, mostly in the realm of computer games like Dota 2, StarCraft II, or MineCraft. However, existing methods lack interpretability in the sense that it is not comprehensible for humans what the agent stores in its memory.In this regard, we propose a novel memory mechanism that represents past events in human language.Our method uses CLIP to associate visual inputs with language tokens. Then we feed these tokens to a pretrained language model that serves the agent as memory and provides it with a coherent and human-readable representation of the past.We train our memory mechanism on a set of partially observable environments and find that it excels on tasks that require a memory component, while mostly attaining performance on-par with strong baselines on tasks that do not. On a challenging continuous recognition task, where memorizing the past is crucial, our memory mechanism converges two orders of magnitude faster than prior methods.Since our memory mechanism is human-readable, we can peek at an agent's memory and check whether crucial pieces of information have been stored.This significantly enhances troubleshooting and paves the way toward more interpretable agents.</details>

### Shared Adversarial Unlearning: Backdoor Mitigation by Unlearning Shared Adversarial Examples

**Authors:** Shaokui Wei, Mingda Zhang, Hongyuan Zha, Baoyuan Wu

**<details><summary>Abstract**</summary>

Backdoor attacks are serious security threats to machine learning models where an adversary can inject poisoned samples into the training set, causing a backdoored model which predicts poisoned samples with particular triggers to particular target classes, while behaving normally on benign samples. In this paper, we explore the task of purifying a backdoored model using a small clean dataset. By establishing the connection between backdoor risk and adversarial risk, we derive a novel upper bound for backdoor risk, which mainly captures the risk on the shared adversarial examples (SAEs) between the backdoored model and the purified model. This upper bound further suggests a novel bi-level optimization problem for mitigating backdoor using adversarial training techniques. To solve it, we propose Shared Adversarial Unlearning (SAU). Specifically, SAU first generates SAEs, and then, unlearns the generated SAEs such that they are either correctly classified by the purified model and/or differently classified by the two models, such that the backdoor effect in the backdoored model will be mitigated in the purified model. Experiments on various benchmark datasets and network architectures show that our proposed method achieves state-of-the-art performance for backdoor defense. The code is available at https://github.com/SCLBD/BackdoorBench (PyTorch) and https://github.com/shawkui/MindTrojan (MindSpore).</details>

### Sharp Calibrated Gaussian Processes

**Authors:** Alexandre Capone, Sandra Hirche, Geoff Pleiss

**<details><summary>Abstract**</summary>

While Gaussian processes are a mainstay for various engineering and scientific applications, the uncertainty estimates don't satisfy frequentist guarantees and can be miscalibrated in practice. State-of-the-art approaches for designing calibrated models rely on inflating the Gaussian process posterior variance, which yields confidence intervals that are potentially too coarse. To remedy this, we present a calibration approach that generates predictive quantiles using a computation inspired by the vanilla Gaussian process posterior variance but using a different set of hyperparameters chosen to satisfy an empirical calibration constraint. This results in a calibration approach that is considerably more flexible than existing approaches, which we optimize to yield tight predictive quantiles. Our approach is shown to yield a calibrated model under reasonable assumptions. Furthermore, it outperforms existing approaches in sharpness when employed for calibrated regression.</details>

### SiT Dataset: Socially Interactive Pedestrian Trajectory Dataset for Social Navigation Robots

**Authors:** Jong Wook Bae, Jungho Kim, Junyong Yun, Changwon Kang, Jeongseon Choi, Chanhyeok Kim, Junho Lee, Jungwook Choi, Jun Won Choi

**<details><summary>Abstract**</summary>

To ensure secure and dependable mobility in environments shared by humans and robots, social navigation robots should possess the capability to accurately perceive and predict the trajectories of nearby pedestrians. In this paper, we present a novel dataset of pedestrian trajectories, referred to as Social Interactive Trajectory (SiT) dataset, which can be used to train pedestrian detection, tracking, and trajectory prediction models needed to design social navigation robots. Our dataset includes sequential raw data captured by two 3D LiDARs and five cameras covering a 360-degree view, two inertial measurement unit (IMU) sensors, and real-time kinematic positioning (RTK), as well as annotations including 2D & 3D boxes, object classes, and object IDs. Thus far, various human trajectory datasets have been introduced to support the development of pedestrian motion forecasting models. Our SiT dataset differs from these datasets in the following two respects. First, whereas the pedestrian trajectory data in other datasets was obtained from static scenes, our data was collected while the robot navigates in a crowded environment, capturing human-robot interactive scenarios in motion. Second, our dataset has been carefully organized to facilitate training and evaluation of end-to-end prediction models encompassing 3D detection, 3D multi-object tracking, and trajectory prediction. This design allows for an end-to-end unified modular approach across different tasks. We have introduced a comprehensive benchmark for assessing models across all aforementioned tasks, and have showcased the performance of multiple baseline models as part of our evaluation. Our dataset provides a strong foundation for future research in pedestrian trajectory prediction, which could expedite the development of safe and agile social navigation robots. The SiT dataset, devkit, and pre-trained models are publicly released at: https://spalaboratory.github.io/SiT</details>

### [Oral] Siamese Masked Autoencoders

**Authors:** Agrim Gupta, Jiajun Wu, Jia Deng, Fei-Fei Li

**Oral Presentation:** Th, Dec 14, 13:20 -- Oral 6C

**<details><summary>Abstract**</summary>

Establishing correspondence between images or scenes is a significant challenge in computer vision, especially given occlusions, viewpoint changes, and varying object appearances. In this paper, we present Siamese Masked Autoencoders (SiamMAE), a simple extension of Masked Autoencoders (MAE) for learning visual correspondence from videos. SiamMAE operates on pairs of randomly sampled video frames and asymmetrically masks them. These frames are processed independently by an encoder network, and a decoder composed of a sequence of cross-attention layers is tasked with predicting the missing patches in the future frame. By masking a large fraction (95%) of patches in the future frame while leaving the past frame unchanged, SiamMAE encourages the network to focus on object motion and learn object-centric representations. Despite its conceptual simplicity, features learned via SiamMAE outperform state-of-the-art self-supervised methods on video object segmentation, pose keypoint propagation, and semantic part propagation tasks. SiamMAE achieves competitive results without relying on data augmentation, handcrafted tracking-based pretext tasks, or other techniques to prevent representational collapse.</details>

### Single-Call Stochastic Extragradient Methods for Structured Non-monotone Variational Inequalities: Improved Analysis under Weaker Conditions

**Authors:** Sayantan Choudhury, Eduard Gorbunov, Nicolas Loizou

**<details><summary>Abstract**</summary>

Single-call stochastic extragradient methods, like stochastic past extragradient (SPEG) and stochastic optimistic gradient (SOG),  have gained a lot of interest in recent years and are one of the most efficient algorithms for solving large-scale min-max optimization and variational inequalities problems (VIP) appearing in various machine learning tasks. However, despite their undoubted popularity, current convergence analyses of SPEG and SOG require strong assumptions like bounded variance or growth conditions. In addition, several important questions regarding the convergence properties of these methods are still open, including mini-batching, efficient step-size selection, and convergence guarantees under different sampling strategies. In this work, we address these questions and provide convergence guarantees for two large classes of structured non-monotone VIPs: (i) quasi-strongly monotone problems (a generalization of strongly monotone problems) and (ii) weak Minty variational inequalities (a generalization of monotone and Minty VIPs). We introduce the expected residual condition, explain its benefits, and show how it allows us to obtain a strictly weaker bound than previously used growth conditions, expected co-coercivity, or bounded variance assumptions. Finally, our convergence analysis holds under the arbitrary sampling paradigm, which includes importance sampling and various mini-batching strategies as special cases.</details>

### Small Total-Cost Constraints in Contextual Bandits with Knapsacks, with Application to Fairness

**Authors:** Evgenii Chzhen, Christophe Giraud, Zhen LI, Gilles Stoltz

**<details><summary>Abstract**</summary>

We consider contextual bandit problems with knapsacks [CBwK], a problem where at each round, a scalar reward is obtained and vector-valued costs are suffered. The learner aims to maximize the cumulative rewards while ensuring that the cumulative costs are lower than some predetermined cost constraints. We assume that contexts come from a continuous set, that costs can be signed, and that the expected reward and cost functions, while unknown, may be uniformly estimated---a typical assumption in the literature. In this setting, total cost constraints had so far to be at least of order $T^{3/4}$, where $T$ is the number of rounds, and were even typically assumed to depend linearly on $T$. We are however motivated to use CBwK to impose a fairness constraint of equalized average costs between groups: the budget associated with the corresponding cost constraints should be as close as possible to the natural deviations, of order $\sqrt{T}$. To that end, we introduce a dual strategy based on projected-gradient-descent updates, that is able to deal with total-cost constraints of the order of $\sqrt{T}$ up to poly-logarithmic terms. This strategy is more direct and simpler than existing strategies in the literature. It relies on a careful, adaptive, tuning of the step size.</details>

### Small batch deep reinforcement learning

**Authors:** Johan Obando Ceron, Marc Bellemare, Pablo Samuel Castro

**<details><summary>Abstract**</summary>

In value-based deep reinforcement learning with replay memories, the batch size parameter specifies how many transitions to sample for each gradient update. Although critical to the learning process, this value is typically not adjusted when proposing new algorithms. In this work we present a broad empirical study that suggests reducing the batch size can result in a number of significant performance gains; this is surprising, as the general tendency when training neural networks is towards larger batch sizes for improved performance. We complement our experimental findings with a set of empirical analyses towards better understanding this phenomenon.</details>

### SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds

**Authors:** Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, Jian Ren

**<details><summary>Abstract**</summary>

Text-to-image diffusion models can create stunning images from natural language descriptions that rival the work of professional artists and photographers. However, these models are large, with complex network architectures and tens of denoising iterations, making them computationally expensive and slow to run. As a result, high-end GPUs and cloud-based inference are required to run diffusion models at scale. This is costly and has privacy implications, especially when user data is sent to a third party. To overcome these challenges, we present a generic approach that, for the first time, unlocks running text-to-image diffusion models on mobile devices in **less than 2 seconds**.  We achieve so by introducing efficient network architecture and improving step distillation. Specifically, we propose an efficient UNet by identifying the redundancy of the original model and reducing the computation of the image decoder via data distillation. Further, we enhance the step distillation by exploring training strategies and introducing regularization from classifier-free guidance. Our extensive experiments on MS-COCO show that our model with $8$ denoising steps achieves better FID and CLIP scores than Stable Diffusion v$1.5$ with $50$ steps. Our work democratizes content creation by bringing powerful text-to-image diffusion models to the hands of users.</details>

### [Spotlight] Sounding Bodies: Modeling 3D Spatial Sound of Humans Using Body Pose and Audio

**Authors:** Xudong XU, Dejan Markovic, Jacob Sandakly, Todd Keebler, Steven Krenn, Alexander Richard

**<details><summary>Abstract**</summary>

While 3D human body modeling has received much attention in computer vision, modeling the acoustic equivalent, i.e. modeling 3D spatial audio produced by body motion and speech, has fallen short in the community. To close this gap, we present a model that can generate accurate 3D spatial audio for full human bodies. The system consumes, as input, audio signals from headset microphones and body pose, and produces, as output, a 3D sound field surrounding the transmitter's body, from which spatial audio can be rendered at any arbitrary position in the 3D space. We collect a first-of-its-kind multimodal dataset of human bodies, recorded with multiple cameras and a spherical array of 345 microphones. In an empirical evaluation, we demonstrate that our model can produce accurate body-induced sound fields when trained with a suitable loss. Dataset and code are available online.</details>

### Sparse Graph Learning from Spatiotemporal Time Series

**Authors:** Andrea Cini, Daniele Zambon, Cesare Alippi

**<details><summary>Abstract**</summary>

Outstanding achievements of graph neural networks for spatiotemporal time series analysis show that relational constraints introduce an effective inductive bias into neural forecasting architectures. Often, however, the relational information characterizing the underlying data-generating process is unavailable and the practitioner is left with the problem of inferring from data which relational graph to use in the subsequent processing stages. We propose novel, principled - yet practical - probabilistic score-based methods that learn the relational dependencies as distributions over graphs while maximizing end-to-end the performance at task. The proposed graph learning framework is based on consolidated variance reduction techniques for Monte Carlo score-based gradient estimation, is theoretically grounded, and, as we show, effective in practice. In this paper, we focus on the time series forecasting problem and show that, by tailoring the gradient estimators to the graph learning problem, we are able to achieve state-of-the-art performance while controlling the sparsity of the learned graph and the computational scalability. We empirically assess the effectiveness of the proposed method on synthetic and real-world benchmarks, showing that the proposed solution can be used as a stand-alone graph identification procedure as well as a graph learning component of an end-to-end forecasting architecture.</details>

### SparseProp: Efficient Event-Based Simulation and Training of Sparse Recurrent Spiking Neural Networks

**Authors:** Rainer Engelken

**<details><summary>Abstract**</summary>

Spiking Neural Networks (SNNs) are biologically-inspired models that are capable of processing information in streams of action potentials. However, simulating and training SNNs is computationally expensive due to the need to solve large systems of coupled differential equations. In this paper, we propose a novel event-based algorithm called SparseProp for simulating and training sparse SNNs. Our algorithm reduces the computational cost of both forward pass and backward pass operations from O(N) to O(log(N)) per network spike, enabling numerically exact simulations of large spiking networks and their efficient training using backpropagation through time. By exploiting the sparsity of the network, SparseProp avoids iterating through all neurons at every spike and uses efficient state updates. We demonstrate the effectiveness of SparseProp for several classical integrate-and-fire neuron models, including simulating a sparse SNN with one million LIF neurons, which is sped up by more than four orders of magnitude compared to previous implementations. Our work provides an efficient and exact solution for training large-scale spiking neural networks and opens up new possibilities for building more sophisticated brain-inspired models.</details>

### Sparsity-Preserving Differentially Private Training of Large Embedding Models

**Authors:** Badih Ghazi, Yangsibo Huang, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Amer Sinha, Chiyuan Zhang

**<details><summary>Abstract**</summary>

As the use of large embedding models in recommendation systems and language applications increases, concerns over user data privacy have also risen.  DP-SGD, a training algorithm that combines differential privacy with stochastic gradient descent, has been the workhorse in protecting user privacy without compromising model accuracy by much. However, applying DP-SGD naively to embedding models can destroy gradient sparsity, leading to reduced training efficiency. To address this issue, we present two new algorithms, DP-FEST and DP-AdaFEST, that preserve gradient sparsity during the private training of large embedding models.  Our algorithms achieve substantial reductions ($10^6 \times$) in gradient size, while maintaining comparable levels of accuracy, on benchmark real-world datasets.</details>

### [Oral] Spatial-frequency channels, shape bias, and adversarial robustness

**Authors:** Ajay Subramanian, Elena Sizikova, Najib Majaj, Denis Pelli

**Oral Presentation:** Th, Dec 14, 14:05 -- Oral 6C

**<details><summary>Abstract**</summary>

What spatial frequency information do humans and neural networks use to recognize objects? In neuroscience, critical band masking is an established tool that can reveal the frequency-selective filters used for object recognition. Critical band masking measures the sensitivity of recognition performance to noise added at each spatial frequency. Existing critical band masking studies show that humans recognize periodic patterns (gratings) and letters by means of a spatial-frequency filter (or "channel") that has a frequency bandwidth of one octave (doubling of frequency). Here, we introduce critical band masking as a task for network-human comparison and test 14 humans and 76 neural networks on 16-way ImageNet categorization in the presence of narrowband noise. We find that humans recognize objects in natural images using the same one-octave-wide channel that they use for letters and gratings, making it a canonical feature of human object recognition. Unlike humans, the neural network channel is very broad, 2-4 times wider than the human channel. This means that the network channel extends to frequencies higher and lower than those that humans are sensitive to. Thus, noise at those frequencies will impair network performance and spare human performance. Adversarial and augmented-image training are commonly used to increase network robustness and shape bias. Does this training align network and human object recognition channels? Three network channel properties (bandwidth, center frequency, peak noise sensitivity) correlate positively with shape bias (53% variance explained) but negatively with robustness of adversarially-trained networks (74% variance explained). Adversarial training increases robustness but expands the channel bandwidth even further beyond the human bandwidth. Thus, critical band masking reveals that the network channel is more than twice as wide as the human channel, and that adversarial training only makes it worse. Networks with narrower channels might be more robust.</details>

### Stability of Random Forests and Coverage of Random-Forest Prediction Intervals

**Authors:** Yan Wang, Huaiqing Wu, Dan Nettleton

**<details><summary>Abstract**</summary>

We establish stability of random forests under the mild condition that the squared response ($Y^2$) does not have a heavy tail. In particular, our analysis holds for the practical version of random forests that is implemented in popular packages like \texttt{randomForest} in \texttt{R}. Empirical results show that stability may persist even beyond our assumption and hold for heavy-tailed $Y^2$. Using the stability property, we prove a non-asymptotic lower bound for the coverage probability of prediction intervals constructed from the out-of-bag error of random forests. With another mild condition that is typically satisfied when $Y$ is continuous, we also establish a complementary upper bound, which can be similarly established for the jackknife prediction interval constructed from an arbitrary stable algorithm. We also discuss the asymptotic coverage probability under assumptions weaker than those considered in previous literature. Our work implies that random forests, with its stability property, is an effective machine learning method that can provide not only satisfactory point prediction but also justified interval prediction at almost no extra computational cost.</details>

### [Spotlight] Stable Diffusion is Unstable

**Authors:** Chengbin Du, Yanxi Li, Zhongwei Qiu, Chang Xu

**<details><summary>Abstract**</summary>

Recently, text-to-image models have been thriving. Despite their powerful generative capacity, our research has uncovered a lack of robustness in this generation process. Specifically, the introduction of small perturbations to the text prompts can result in the blending of primary subjects with other categories or their complete disappearance in the generated images. In this paper, we propose **Auto-attack on Text-to-image Models (ATM)**, a gradient-based approach, to effectively and efficiently generate such perturbations. By learning a Gumbel Softmax distribution, we can make the discrete process of word replacement or extension continuous, thus ensuring the differentiability of the perturbation generation. Once the distribution is learned, ATM can sample multiple attack samples simultaneously. These attack samples can prevent the generative model from generating the desired subjects without tampering with the category keywords in the prompt. ATM has achieved a 91.1\% success rate in short-text attacks and an 81.2\% success rate in long-text attacks. Further empirical analysis revealed three attack patterns based on: 1) variability in generation speed, 2) similarity of coarse-grained characteristics, and 3) polysemy of words. The code is available at https://github.com/duchengbin8/Stable_Diffusion_is_Unstable</details>

### Stable and low-precision training for large-scale vision-language models

**Authors:** Mitchell Wortsman, Tim Dettmers, Luke Zettlemoyer, Ari Morcos, Ali Farhadi, Ludwig Schmidt

**<details><summary>Abstract**</summary>

We introduce new methods for 1) accelerating and 2) stabilizing training for large language-vision models. 1) For acceleration, we introduce SwitchBack, a linear layer for int8 quantized training which provides a speed-up of 13-25% while matching the performance of bfloat16 training within 0.1 percentage points for the 1B parameter CLIP ViT-Huge---the largest int8 training to date. Our main focus is int8 as GPU support for float8 is rare, though we also analyze float8 training through simulation. While SwitchBack proves effective for float8, we show that standard techniques are also successful if the network is trained and initialized so that large feature magnitudes are discouraged, which we accomplish via layer-scale initialized with zeros. 2) For stability, we analyze loss spikes and find they consistently occur 1-8 iterations after the squared gradients become under-estimated by their AdamW second moment estimator. As a result, we recommend an AdamW-Adafactor hybrid which avoids loss spikes when training a CLIP ViT-Huge model and outperforms gradient clipping at the scales we test.</details>

### StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners

**Authors:** Yonglong Tian, Lijie Fan, Phillip Isola, Huiwen Chang, Dilip Krishnan

**<details><summary>Abstract**</summary>

We investigate the potential of learning visual representations using synthetic images generated by text-to-image models. This is a natural question in the light of the excellent performance of such models in generating high-quality images. We consider specifically the Stable Diffusion, one of the leading open source text-to-image models. We show that (1) when the generative model is properly configured, training self-supervised methods on synthetic images can match or beat the real image counterpart;(2) by treating the multiple images generated from the same text prompt as positives for each other, we develop a multi-positive contrastive learning method, which we call StableRep. With solely synthetic images, the representations learned by StableRep surpass the performance of representations learned by SimCLR and CLIP using the same set of text prompts and corresponding real images, on large scale datasets. When we further add language supervision, \name~trained with 20M synthetic images (10M captions) achieves better accuracy than CLIP trained with 50M real images (50M captions).</details>

### Star-Shaped Denoising Diffusion Probabilistic Models

**Authors:** Andrey Okhotin, Dmitry Molchanov, Arkhipkin Vladimir, Grigory Bartosh, Viktor Ohanesian, Aibek Alanov, Dmitry Vetrov

**<details><summary>Abstract**</summary>

Denoising Diffusion Probabilistic Models (DDPMs) provide the foundation for the recent breakthroughs in generative modeling.Their Markovian structure makes it difficult to define DDPMs with distributions other than Gaussian or discrete.In this paper, we introduce Star-Shaped DDPM (SS-DDPM).Its *star-shaped diffusion process* allows us to bypass the need to define the transition probabilities or compute posteriors.We establish duality between star-shaped and specific Markovian diffusions for the exponential family of distributions and derive efficient algorithms for training and sampling from SS-DDPMs.In the case of Gaussian distributions, SS-DDPM is equivalent to DDPM.However, SS-DDPMs provide a simple recipe for designing diffusion models with distributions such as Beta, von Mises–Fisher, Dirichlet, Wishart and others, which can be especially useful when data lies on a constrained manifold.We evaluate the model in different settings and find it competitive even on image data, where Beta SS-DDPM achieves results comparable to a Gaussian DDPM.Our implementation is available at https://github.com/andrey-okhotin/star-shaped</details>

### Static and Sequential Malicious Attacks in the Context of Selective Forgetting

**Authors:** Chenxu Zhao, Wei Qian, Rex Ying, Mengdi Huai

**<details><summary>Abstract**</summary>

With the growing demand for the right to be forgotten, there is an increasing need for machine learning models to forget sensitive data and its impact. To address this, the paradigm of selective forgetting (a.k.a machine unlearning) has been extensively studied, which aims to remove the impact of requested data from a well-trained model without retraining from scratch. Despite its significant success, limited attention has been given to the security vulnerabilities of the unlearning system concerning malicious data update requests. Motivated by this, in this paper, we explore the possibility and feasibility of malicious data update requests during the unlearning process. Specifically, we first propose a new class of malicious selective forgetting attacks, which involves a static scenario where all the malicious data update requests are provided by the adversary at once. Additionally, considering the sequential setting where the data update requests arrive sequentially, we also design a novel framework for sequential forgetting attacks, which is formulated as a stochastic optimal control problem. We also propose novel optimization algorithms that can find the effective malicious data update requests. We perform theoretical analyses for the proposed selective forgetting attacks, and extensive experimental results validate the effectiveness of our proposed selective forgetting attacks. The source code is available in the supplementary material.</details>

### Statistical Limits of Adaptive Linear Models: Low-Dimensional Estimation and Inference

**Authors:** Licong Lin, Mufang Ying, Suvrojit Ghosh, Koulik Khamaru, Cun-Hui Zhang

**<details><summary>Abstract**</summary>

Estimation and inference in statistics pose significant challenges when data are collected adaptively. Even in linear models, the Ordinary Least Squares (OLS) estimator may fail to exhibit asymptotic normality for single coordinate estimation and have inflated error. This issue is highlighted by a recent minimax lower bound, which shows that the error of estimating a  single coordinate can be enlarged by a multiple of $\sqrt{d}$ when data are allowed to be arbitrarily adaptive, compared with the case when they are i.i.d. Our work explores this striking difference in estimation performance between utilizing i.i.d. and adaptive data. We investigate how the degree of adaptivity in data collection impacts the performance of estimating a low-dimensional parameter component in high-dimensional linear models. We identify conditions on the data collection mechanism under which the estimation error for a low-dimensional parameter component matches its counterpart in the i.i.d. setting, up to a factor that depends on the degree of adaptivity. We show that OLS or OLS on centered data can achieve this matching error. In addition, we propose a novel estimator for single coordinate inference via solving a Two-stage Adaptive Linear Estimating equation (TALE). Under a weaker form of adaptivity in data collection, we establish an asymptotic normality property of the proposed estimator.</details>

### [Spotlight] Stein $\Pi$-Importance Sampling

**Authors:** Congye Wang, Ye Chen, Heishiro Kanagawa, Chris Oates

**<details><summary>Abstract**</summary>

Stein discrepancies have emerged as a powerful tool for retrospective improvement of Markov chain Monte Carlo output.  However, the question of how to design Markov chains that are well-suited to such post-processing has yet to be addressed.  This paper studies Stein importance sampling, in which weights are assigned to the states visited by a $\Pi$-invariant Markov chain to obtain a consistent approximation of $P$, the intended target.  Surprisingly, the optimal choice of $\Pi$ is not identical to the target $P$; we therefore propose an explicit construction for $\Pi$ based on a novel variational argument.  Explicit conditions for convergence of Stein $\Pi$-Importance Sampling are established.  For $\approx 70$% of tasks in the PosteriorDB benchmark, a significant improvement over the analogous post-processing of $P$-invariant Markov chains is reported.</details>

### Stochastic Distributed Optimization under Average Second-order Similarity: Algorithms and Analysis

**Authors:** Dachao Lin, Yuze Han, Haishan Ye, Zhihua Zhang

**<details><summary>Abstract**</summary>

We study finite-sum distributed optimization problems involving a master node and $n-1$ local nodes under the popular $\delta$-similarity and $\mu$-strong convexity conditions. We propose two new algorithms, SVRS and AccSVRS, motivated by previous works. The non-accelerated SVRS method combines the techniques of gradient sliding and variance reduction and achieves a better communication complexity of $\tilde{\mathcal{O}}(n {+} \sqrt{n}\delta/\mu)$ compared to existing non-accelerated algorithms. Applying the framework proposed in Katyusha X, we also develop a directly accelerated version named AccSVRS with the $\tilde{\mathcal{O}}(n {+} n^{3/4}\sqrt{\delta/\mu})$ communication complexity. In contrast to existing results, our complexity bounds are entirely smoothness-free and exhibit superiority in ill-conditioned cases. Furthermore, we establish a nearly matched lower bound to verify the tightness of our AccSVRS method.</details>

### Structured Voronoi Sampling

**Authors:** Afra Amini, Li Du, Ryan Cotterell

**<details><summary>Abstract**</summary>

Gradient-based sampling algorithms have demonstrated their effectiveness in text generation, especially in the context of controlled text generation. However, there exists a lack of theoretically grounded and principled approaches for this task. In this paper, we take an important step toward building a principled approach for sampling from language models with gradient-based methods. We use discrete distributions given by language models to define densities and develop an algorithm based on Hamiltonian Monte Carlo to sample from them. We name our gradient-based technique Structured Voronoi Sampling (SVS). In an experimental setup where the reference distribution is known, we show that the empirical distribution of SVS samples is closer to the reference distribution compared to alternative sampling schemes. Furthermore, in a controlled generation task, SVS is able to generate fluent and diverse samples while following the control targets significantly better than other methods.</details>

### SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality

**Authors:** Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, Ranjay Krishna

**<details><summary>Abstract**</summary>

In the last year alone, a surge of new benchmarks to measure $	extit{compositional}$ understanding of vision-language models have permeated the machine learning ecosystem.Given an image, these benchmarks probe a model's ability to identify its associated caption amongst a set of compositional distractors.Surprisingly, we find significant biases in $	extit{all}$ these benchmarks rendering them hackable. This hackability is so dire that blind models with no access to the image outperform state-of-the-art vision-language models.To remedy this rampant vulnerability, we introduce $	extit{SugarCrepe}$, a new benchmark for vision-language compositionality evaluation.We employ large language models, instead of rule-based templates used in previous benchmarks, to generate fluent and sensical hard negatives, and utilize an adversarial refinement mechanism to maximally reduce biases. We re-evaluate state-of-the-art models and recently proposed compositionality inducing strategies, and find that their improvements were hugely overestimated, suggesting that more innovation is needed in this important direction.We release $	extit{SugarCrepe}$ and the code for evaluation at: https://github.com/RAIVNLab/sugar-crepe.</details>

### Suggesting Variable Order for Cylindrical Algebraic Decomposition via Reinforcement Learning

**Authors:** Fuqi Jia, Yuhang Dong, Minghao Liu, Pei Huang, Feifei Ma, Jian Zhang

**<details><summary>Abstract**</summary>

Cylindrical Algebraic Decomposition (CAD) is one of the pillar algorithms of symbolic computation, and its worst-case complexity is double exponential to the number of variables. Researchers found that variable order dramatically affects efficiency and proposed various heuristics. The existing learning-based methods are all supervised learning methods that cannot cope with diverse polynomial sets.This paper proposes two Reinforcement Learning (RL) approaches combined with Graph Neural Networks (GNN) for Suggesting Variable Order (SVO). One is GRL-SVO(UP), a branching heuristic integrated with CAD. The other is GRL-SVO(NUP), a fast heuristic providing a total order directly. We generate a random dataset and collect a real-world dataset from SMT-LIB. The experiments show that our approaches outperform state-of-the-art learning-based heuristics and are competitive with the best expert-based heuristics. Interestingly, our models show a strong generalization ability, working well on various datasets even if they are only trained on a 3-var random dataset. The source code and data are available at https://github.com/dongyuhang22/GRL-SVO.</details>

### [Spotlight] Survival Instinct in Offline Reinforcement Learning

**Authors:** Anqi Li, Dipendra Misra, Andrey Kolobov, Ching-An Cheng

**<details><summary>Abstract**</summary>

We present a novel observation about the behavior of offline reinforcement learning (RL) algorithms: on many benchmark datasets, offline RL can produce well-performing and safe policies even when trained with "wrong" reward labels, such as those that are zero everywhere or are negatives of the true rewards. This phenomenon cannot be easily explained by offline RL's return maximization objective. Moreover, it gives offline RL a degree of robustness that is uncharacteristic of its online RL counterparts, which are known to be sensitive to reward design. We demonstrate that this surprising robustness property is attributable to an interplay between the notion of *pessimism* in offline RL algorithms and certain implicit biases in common data collection practices. As we prove in this work, pessimism endows the agent with a *survival instinct*, i.e., an incentive to stay within the data support in the long term, while the limited and biased data coverage further constrains the set of survival policies. Formally, given a reward class -- which may not even contain the true reward -- we identify conditions on the training data distribution that enable offline RL to learn a near-optimal and safe policy from any reward within the class. We argue that the survival instinct should be taken into account when interpreting results from existing offline RL benchmarks and when creating future ones. Our empirical and theoretical results suggest a new paradigm for offline RL, whereby an agent is "nudged" to learn a desirable behavior with imperfect reward but purposely biased data coverage. Please visit our website [https://survival-instinct.github.io](https://survival-instinct.github.io) for accompanied code and videos.</details>

### TIES-Merging: Resolving Interference When Merging Models

**Authors:** Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, Mohit Bansal

**<details><summary>Abstract**</summary>

Transfer learning – i.e., further fine-tuning a pre-trained model on a downstream task – can confer significant advantages, including improved downstream performance, faster convergence, and better sample efficiency. These advantages have led to a proliferation of task-specific fine-tuned models, which typically can only perform a single task and do not benefit from one another. Recently, model merging techniques have emerged as a solution to combine multiple task-specific models into a single multitask model without performing additional training. However, existing merging methods often ignore the interference between parameters of different models, resulting in large performance drops when merging multiple models. In this paper, we demonstrate that prior merging techniques inadvertently lose valuable information due to two major sources of interference: (a) interference due to redundant parameter values and (b) disagreement on the sign of a given parameter’s values across models. To address this, we propose our method, TrIm, Elect Sign & Merge (TIES-Merging), which introduces three novel steps when merging models: (1) resetting parameters that only changed a small amount during fine-tuning, (2) resolving sign conflicts, and (3) merging only the parameters that are in alignment with the final agreed-upon sign. We find that TIES-Merging outperforms existing methods in diverse settings covering a range of modalities, domains, number of tasks, model sizes, architectures, and fine-tuning settings. We further analyze the impact of different types of interference on model parameters, highlight the importance of signs, and show that estimating the signs using the validation data could further improve performance.</details>

### TOA: Task-oriented Active VQA

**Authors:** xiaoying xing, Mingfu Liang, Ying Wu

**<details><summary>Abstract**</summary>

Knowledge-based visual question answering (VQA) requires external knowledge to answer the question about an image. Early methods explicitly retrieve knowledge from external knowledge bases, which often introduce noisy information. Recently large language models like GPT-3 have shown encouraging performance as implicit knowledge source and revealed planning abilities. However, current large language models can not effectively understand image inputs, thus it remains an open problem to extract the image information and input to large language models. Prior works have used image captioning and object descriptions to represent the image. However, they may either drop the essential visual information to answer the question correctly or involve irrelevant objects to the task-of-interest. To address this problem, we propose to let large language models make an initial hypothesis according to their knowledge, then actively collect the visual evidence required to verify the hypothesis. In this way, the model can attend to the essential visual information in a task-oriented manner. We leverage several vision modules from the perspectives of spatial attention (i.e., Where to look) and attribute attention (i.e., What to look), which is similar to human cognition. The experiments show that our proposed method outperforms the baselines on open-ended knowledge-based VQA datasets and presents clear reasoning procedure with better interpretability.</details>

### Tailoring Self-Attention for Graph via Rooted Subtrees

**Authors:** Siyuan Huang, Yunchong Song, Jiayue Zhou, Zhouhan Lin

**<details><summary>Abstract**</summary>

Attention mechanisms have made significant strides in graph learning, yet they still exhibit notable limitations: local attention faces challenges in capturing long-range information due to the inherent problems of the message-passing scheme, while global attention cannot reflect the hierarchical neighborhood structure and fails to capture fine-grained local information. In this paper, we propose a novel multi-hop graph attention mechanism, named Subtree Attention (STA), to address the aforementioned issues. STA seamlessly bridges the fully-attentional structure and the rooted subtree, with theoretical proof that STA approximates the global attention under extreme settings. By allowing direct computation of attention weights among multi-hop neighbors, STA mitigates the inherent problems in existing graph attention mechanisms. Further we devise an efficient form for STA by employing kernelized softmax, which yields a linear time complexity. Our resulting GNN architecture, the STAGNN, presents a simple yet performant STA-based graph neural network leveraging a hop-aware attention strategy. Comprehensive evaluations on ten node classification datasets demonstrate that STA-based models outperform existing graph transformers and mainstream GNNs. The codeis available at https://github.com/LUMIA-Group/SubTree-Attention.</details>

### Tartarus: A Benchmarking Platform for Realistic And Practical Inverse Molecular Design

**Authors:** AkshatKumar Nigam, Robert Pollice, Gary Tom, Kjell Jorner, John Willes, Luca Thiede, Anshul Kundaje, Alan Aspuru-Guzik

**<details><summary>Abstract**</summary>

The efficient exploration of chemical space to design molecules with intended properties enables the accelerated discovery of drugs, materials, and catalysts, and is one of the most important outstanding challenges in chemistry. Encouraged by the recent surge in computer power and artificial intelligence development, many algorithms have been developed to tackle this problem. However, despite the emergence of many new approaches in recent years, comparatively little progress has been made in developing realistic benchmarks that reflect the complexity of molecular design for real-world applications. In this work, we develop a set of practical benchmark tasks relying on physical simulation of molecular systems mimicking real-life molecular design problems for materials, drugs, and chemical reactions. Additionally, we demonstrate the utility and ease of use of our new benchmark set by demonstrating how to compare the performance of several well-established families of algorithms. Overall, we believe that our benchmark suite will help move the field towards more realistic molecular design benchmarks, and move the development of inverse molecular design algorithms closer to the practice of designing molecules that solve existing problems in both academia and industry alike.</details>

### [Oral] Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models

**Authors:** Guillermo Ortiz-Jimenez, Alessandro Favero, Pascal Frossard

**Oral Presentation:** Th, Dec 14, 13:35 -- Oral 6A

**<details><summary>Abstract**</summary>

Task arithmetic has recently emerged as a cost-effective and scalable approach to edit pre-trained models directly in weight space: By adding the fine-tuned weights of different tasks, the model's performance can be improved on these tasks, while negating them leads to task forgetting. Yet, our understanding of the effectiveness of task arithmetic and its underlying principles remains limited. We present a comprehensive study of task arithmetic in vision-language models and show that weight disentanglement is the crucial factor that makes it effective. This property arises during pre-training and manifests when distinct directions in weight space govern separate, localized regions in function space associated with the tasks. Notably, we show that fine-tuning models in their tangent space by linearizing them amplifies weight disentanglement. This leads to substantial performance improvements across multiple task arithmetic benchmarks and diverse models. Building on these findings, we provide theoretical and empirical analyses of the neural tangent kernel (NTK) of these models and establish a compelling link between task arithmetic and the spatial localization of the NTK eigenfunctions. Overall, our work uncovers novel insights into the fundamental mechanisms of task arithmetic and offers a more reliable and effective approach to edit pre-trained models through the NTK linearization.</details>

### Temporal Causal Mediation through a Point Process: Direct and Indirect Effects of Healthcare Interventions

**Authors:** Çağlar Hızlı, ST John, Anne Juuti, Tuure Saarinen, Kirsi Pietiläinen, Pekka Marttinen

**<details><summary>Abstract**</summary>

Deciding on an appropriate intervention requires a causal model of a treatment, the outcome, and potential mediators. Causal mediation analysis lets us distinguish between direct and indirect effects of the intervention, but has mostly been studied in a static setting. In healthcare, data come in the form of complex, irregularly sampled time-series, with dynamic interdependencies between a treatment, outcomes, and mediators across time. Existing approaches to dynamic causal mediation analysis are limited to regular measurement intervals, simple parametric models, and disregard long-range mediator--outcome interactions. To address these limitations, we propose a non-parametric mediator--outcome model where the mediator is assumed to be a temporal point process that interacts with the outcome process. With this model, we estimate the direct and indirect effects of an external intervention on the outcome, showing how each of these affects the whole future trajectory. We demonstrate on semi-synthetic data that our method can accurately estimate direct and indirect effects. On real-world healthcare data, our model infers clinically  meaningful direct and indirect effect trajectories for blood glucose after a surgery.</details>

### Temporal Continual Learning with Prior Compensation for Human Motion Prediction

**Authors:** Jianwei Tang, Jiangxin Sun, Xiaotong Lin, lifang zhang, Wei-Shi Zheng, Jian-Fang Hu

**<details><summary>Abstract**</summary>

Human Motion Prediction (HMP) aims to predict future poses at different moments according to past motion sequences. Previous approaches have treated the prediction of various moments equally, resulting in two main limitations: the learning of short-term predictions is hindered by the focus on long-term predictions, and the incorporation of prior information from past predictions into subsequent predictions is limited. In this paper, we introduce a novel multi-stage training framework called Temporal Continual Learning (TCL) to address the above challenges. To better preserve prior information, we introduce the Prior Compensation Factor (PCF). We incorporate it into the model training to compensate for the lost prior information. Furthermore, we derive a more reasonable optimization objective through theoretical derivation. It is important to note that our TCL framework can be easily integrated with different HMP backbone models and adapted to various datasets and applications. Extensive experiments on four HMP benchmark datasets demonstrate the effectiveness and flexibility of TCL. The code is available at https://github.com/hyqlat/TCL.</details>

### [Oral] Tester-Learners for Halfspaces: Universal Algorithms

**Authors:** Aravind Gollakota, Adam Klivans, Konstantinos Stavropoulos, Arsen Vasilyan

**Oral Presentation:** Th, Dec 14, 13:50 -- Oral 6D

**<details><summary>Abstract**</summary>

We give the first tester-learner for halfspaces that succeeds universally over a wide class of structured distributions. Our universal tester-learner runs in fully polynomial time and has the following guarantee: the learner achieves error $O(\mathrm{opt}) + \epsilon$ on any labeled distribution that the tester accepts, and moreover, the tester accepts whenever the marginal is any distribution that satisfies a Poincare inequality. In contrast to prior work on testable learning, our tester is not tailored to any single target distribution but rather succeeds for an entire target class of distributions. The class of Poincare distributions includes all strongly log-concave distributions, and, assuming the Kannan--Lovasz--Simonovits (KLS) conjecture, includes all log-concave distributions. In the special case where the label noise is known to be Massart, our tester-learner achieves error $\mathrm{opt} + \epsilon$ while accepting all log-concave distributions unconditionally (without assuming KLS).Our tests rely on checking hypercontractivity of the unknown distribution using a sum-of-squares (SOS) program, and crucially make use of the fact that Poincare distributions are certifiably hypercontractive in the SOS framework.</details>

### Text Alignment Is An Efficient Unified Model for Massive NLP Tasks

**Authors:** Yuheng Zha, Yichi Yang, Ruichen Li, Zhiting Hu

**<details><summary>Abstract**</summary>

Large language models (LLMs), typically designed as a function of next-word prediction, have excelled across extensive NLP tasks. Despite the generality, next-word prediction is often not an efficient formulation for many of the tasks, demanding an extreme scale of model parameters (10s or 100s of billions) and sometimes yielding suboptimal performance.In practice, it is often desirable to build more efficient models---despite being less versatile, they still apply to a substantial subset of problems, delivering on par or even superior performance with much smaller model sizes.In this paper, we propose text alignment as an efficient unified model for a wide range of crucial tasks involving text entailment, similarity, question answering (and answerability), factual consistency, and so forth. Given a pair of texts, the model measures the degree of alignment between their information. We instantiate an alignment model through lightweight finetuning of RoBERTa (355M parameters) using 5.9M examples from 28 datasets. Despite its compact size, extensive experiments show the model's efficiency and strong performance: (1) On over 20 datasets of aforementioned diverse tasks, the model matches or surpasses FLAN-T5 models that have around 2x or 10x more parameters; the single unified model also outperforms task-specific models finetuned on individual datasets; (2) When applied to evaluate factual consistency of language generation on 23 datasets, our model improves over various baselines, including the much larger GPT-3.5 (ChatGPT) and sometimes even GPT-4; (3) The lightweight model can also serve as an add-on component for LLMs such as GPT-3.5 in question answering tasks, improving the average exact match (EM) score by 17.94 and F1 score by 15.05 through identifying unanswerable questions.</details>

### The Bayesian Stability Zoo

**Authors:** Shay Moran, Hilla Schefler, Jonathan Shafer

**<details><summary>Abstract**</summary>

We show that many definitions of stability found in the learning theory literature are equivalent to one another. We distinguish between two families of definitions of stability: distribution-dependent and distribution-independent Bayesian stability. Within each family, we establish equivalences between various definitions, encompassing approximate differential privacy, pure differential privacy, replicability, global stability, perfect generalization, TV stability, mutual information stability, KL-divergence stability, and Rényi-divergence stability. Along the way, we prove boosting results that enable the amplification of the stability of a learning rule. This work is a step towards a more systematic taxonomy of stability notions in learning theory,  which can promote clarity and an improved understanding of an array of stability concepts that have emerged in recent years.</details>

### [Oral] The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks

**Authors:** Ziqian Zhong, Ziming Liu, Max Tegmark, Jacob Andreas

**Oral Presentation:** Th, Dec 14, 13:50 -- Oral 6A

**<details><summary>Abstract**</summary>

Do neural networks, trained on well-understood algorithmic tasks, reliably rediscover known algorithms? Several recent studies, on tasks ranging from group operations to in-context linear regression, have suggested that the answer is yes. Using modular addition as a prototypical problem, we show that algorithm discovery in neural networks is sometimes more complex: small changes to model hyperparameters and initializations can induce discovery of qualitatively different algorithms from a fixed training set, and even learning of multiple different solutions in parallel. In modular addition, we specifically show that models learn a known *Clock* algorithm, a previously undescribed, less intuitive, but comprehensible procedure we term the *Pizza* algorithm, and a variety of even more complex procedures. Our results show that even simple learning problems can admit a surprising diversity of solutions, motivating the development of new tools for mechanistically characterizing the behavior of neural networks across the algorithmic phase space.</details>

### The Drunkard’s Odometry: Estimating Camera Motion in Deforming Scenes

**Authors:** David Recasens Lafuente, Martin R. Oswald, Marc Pollefeys, Javier Civera

**<details><summary>Abstract**</summary>

Estimating camera motion in deformable scenes poses a complex and open research challenge. Most existing non-rigid structure from motion techniques assume to observe also static scene parts besides deforming scene parts in order to establish an anchoring reference. However, this assumption does not hold true in certain relevant application cases such as endoscopies. Deformable odometry and SLAM pipelines, which tackle the most challenging scenario of exploratory trajectories, suffer from a lack of robustness and proper quantitative evaluation methodologies. To tackle this issue with a common benchmark, we introduce the Drunkard's Dataset, a challenging collection of synthetic data targeting visual navigation and reconstruction in deformable environments. This dataset is the first large set of exploratory camera trajectories with ground truth inside 3D scenes where every surface exhibits non-rigid deformations over time. Simulations in realistic 3D buildings lets us obtain a vast amount of data and ground truth labels, including camera poses, RGB images and depth, optical flow and normal maps at high resolution and quality. We further present a novel deformable odometry method, dubbed the Drunkard’s Odometry, which decomposes optical flow estimates into rigid-body camera motion and non-rigid scene deformations. In order to validate our data, our work contains an evaluation of several baselines as well as a novel tracking error metric which does not require ground truth data. Dataset and code: https://davidrecasens.github.io/TheDrunkard'sOdometry/</details>

### [Spotlight] The Geometry of Neural Nets' Parameter Spaces Under Reparametrization

**Authors:** Agustinus Kristiadi, Felix Dangel, Philipp Hennig

**<details><summary>Abstract**</summary>

Model reparametrization, which follows the change-of-variable rule of calculus, is a popular way to improve the training of neural nets. But it can also be problematic since it can induce inconsistencies in, e.g., Hessian-based flatness measures, optimization trajectories, and modes of probability densities. This complicates downstream analyses: e.g. one cannot definitively relate flatness with generalization since arbitrary reparametrization changes their relationship. In this work, we study the invariance of neural nets under reparametrization from the perspective of Riemannian geometry. From this point of view, invariance is an inherent property of any neural net _if_ one explicitly represents the metric and uses the correct associated transformation rules. This is important since although the metric is always present, it is often implicitly assumed as identity, and thus dropped from the notation, then lost under reparametrization. We discuss implications for measuring the flatness of minima, optimization, and for probability-density maximization. Finally, we explore some interesting directions where invariance is useful.</details>

### The Graph Pencil Method: Mapping Subgraph Densities to Stochastic Block Models

**Authors:** Lee Gunderson, Gecia Bravo-Hermsdorff, Peter Orbanz

**<details><summary>Abstract**</summary>

In this work, we describe a method that determines an exact map from a finite set of subgraph densities to the parameters of a stochastic block model (SBM) matching these densities. Given a number K of blocks, the subgraph densities of a finite number of stars and bistars uniquely determines a single element of the class of all degree-separated stochastic block models with K blocks. Our method makes it possible to translate estimates of these subgraph densities into model parameters, and hence to use subgraph densities directly for inference. The computational overhead is negligible; computing the translation map is polynomial in K, but independent of the graph size once the subgraph densities are given.</details>

### [Spotlight] The Pursuit of Human Labeling: A New Perspective on Unsupervised Learning

**Authors:** Artyom Gadetsky, Maria Brbic

**<details><summary>Abstract**</summary>

We present HUME, a simple model-agnostic framework for inferring human labeling of a given dataset without any external supervision. The key insight behind our approach is that classes defined by many human labelings are linearly separable regardless of the representation space used to represent a dataset. HUME utilizes this insight to guide the search over all possible labelings of a dataset to discover an underlying human labeling. We show that the proposed optimization objective is strikingly well-correlated with the ground truth labeling of the dataset. In effect, we only train linear classifiers on top of pretrained representations that remain fixed during training, making our framework compatible with any large pretrained and self-supervised model. Despite its simplicity, HUME outperforms a supervised linear classifier on top of self-supervised representations on the STL-10 dataset by a large margin and achieves comparable performance on the CIFAR-10 dataset. Compared to the existing unsupervised baselines, HUME achieves state-of-the-art performance on four benchmark image classification datasets including the large-scale ImageNet-1000 dataset. Altogether, our work provides a fundamentally new view to tackle unsupervised learning by searching for consistent labelings between different representation spaces.</details>

### The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data Only

**Authors:** Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, Julien Launay

**<details><summary>Abstract**</summary>

Large language models are commonly trained on a mixture of filtered web data and curated ``high-quality'' corpora, such as social media conversations, books, or technical papers. This curation process is believed to be necessary to produce performant models with broad zero-shot generalization abilities. However, as larger models requiring pretraining on trillions of tokens are considered, it is unclear how scalable is curation, and whether we will run out of unique high-quality data soon.  At variance with previous beliefs, we show that properly filtered and deduplicated web data alone can lead to powerful models; even significantly outperforming models trained on The Pile. Despite extensive filtering, the high-quality data we extract from the web is still plentiful, and we are able to obtain five trillion tokens from CommonCrawl. We publicly release an extract of 500 billion tokens from our RefinedWeb dataset, and 1.3/7.5B parameters language models trained on it.</details>

### The Separation Capacity of Random Neural Networks

**Authors:** Sjoerd Dirksen, Martin Genzel, Laurent Jacques, Alexander Stollenwerk

**<details><summary>Abstract**</summary>

Neural networks with random weights appear in a variety of machine learning applications, most prominently as the initialization of many deep learning algorithms and as a computationally cheap alternative to fully learned neural networks. In the present article, we enhance the theoretical understanding of random neural networks by addressing the following data separation problem: under what conditions can a random neural network make two classes $\mathcal{X}^-, \mathcal{X}^+ \subset \mathbb{R}^d$ (with positive distance) linearly separable? We show that a sufficiently large two-layer ReLU-network with standard Gaussian weights and uniformly distributed biases can solve this problem with high probability. Crucially, the number of required neurons is explicitly linked to geometric properties of the underlying sets $\mathcal{X}^-, \mathcal{X}^+$ and their mutual arrangement. This instance-specific viewpoint allows us to overcome the usual curse of dimensionality (exponential width of the layers) in non-pathological situations where the data carries low-complexity structure. We quantify the relevant structure of the data in terms of a novel notion of mutual complexity (based on a localized version of Gaussian mean width), which leads to sound and informative separation guarantees. We connect our result with related lines of work on approximation, memorization, and generalization.</details>

### The Simplicity Bias in Multi-Task RNNs: Shared Attractors, Reuse of Dynamics, and Geometric Representation

**Authors:** Elia Turner, Omri Barak

**<details><summary>Abstract**</summary>

How does a single interconnected neural population perform multiple tasks, each with its own dynamical requirements? The relation between task requirements and neural dynamics in Recurrent Neural Networks (RNNs) has been investigated for single tasks. The forces shaping joint dynamics of multiple tasks, however, are largely unexplored. In this work, we first construct a systematic framework to study multiple tasks in RNNs, minimizing interference from input and output correlations with the hidden representation. This allows us to reveal how RNNs tend to share attractors and reuse dynamics, a tendency we define as the "simplicity bias".We find that RNNs develop attractors sequentially during training, preferentially reusing existing dynamics and opting for simple solutions when possible. This sequenced emergence and preferential reuse encapsulate the simplicity bias. Through concrete examples, we demonstrate that new attractors primarily emerge due to task demands or architectural constraints, illustrating a balance between simplicity bias and external factors.We examine the geometry of joint representations within a single attractor, by constructing a family of tasks from a set of functions. We show that the steepness of the associated functions controls their alignment within the attractor. This arrangement again highlights the simplicity bias, as points with similar input spacings undergo comparable transformations to reach the shared attractor.Our findings propose compelling applications. The geometry of shared attractors might allow us to infer the nature of unknown tasks. Furthermore, the simplicity bias implies that without specific incentives, modularity in RNNs may not spontaneously emerge, providing insights into the conditions required for network specialization.</details>

### [Oral] The Surprising Effectiveness of Diffusion Models for Optical Flow and Monocular Depth Estimation

**Authors:** Saurabh Saxena, Charles Herrmann, Junhwa Hur, Abhishek Kar, Mohammad Norouzi, Deqing Sun, David Fleet

**Oral Presentation:** Th, Dec 14, 13:50 -- Oral 6C

**<details><summary>Abstract**</summary>

Denoising diffusion probabilistic models have transformed image generation with their impressive fidelity and diversity.We show that they also excel in estimating optical flow and monocular depth, surprisingly without task-specific architectures and loss functions that are predominant for these tasks. Compared to the point estimates of conventional regression-based methods, diffusion models also enable Monte Carlo inference, e.g., capturing uncertainty and ambiguity in flow and depth.With self-supervised pre-training, the combined use of synthetic and real data for supervised training, and technical innovations (infilling and step-unrolled denoising diffusion training) to handle noisy-incomplete training data, one can train state-of-the-art diffusion models for depth and optical flow estimation, with additional zero-shot coarse-to-fine refinement for high resolution estimates. Extensive experiments focus on quantitative performance against benchmarks, ablations, and the model's ability to capture uncertainty and multimodality, and impute missing values. Our model obtains a state-of-the-art relative depth error of 0.074 on the indoor NYU benchmark and an Fl-all score of 3.26\% on the KITTI  optical flow benchmark, about 25\% better than the best published method.</details>

### The ToMCAT Dataset

**Authors:** Adarsh Pyarelal, Eric Duong, Caleb Shibu, Paulo Soares, Savannah Boyd, Payal Khosla, Valeria A. Pfeifer, Diheng Zhang, Eric Andrews, Rick Champlin, Vincent Raymond, Meghavarshini Krishnaswamy, Clayton Morrison, Emily Butler, Kobus Barnard

**<details><summary>Abstract**</summary>

We present a rich, multimodal dataset consisting of data from 40 teams of three humans conducting simulated urban search-and-rescue (SAR) missions in a Minecraft-based testbed, collected for the Theory of Mind-based Cognitive Architecture for Teams (ToMCAT) project. Modalities include two kinds of brain scan data---functional near-infrared spectroscopy (fNIRS) and electroencephalography (EEG), as well as skin conductance, heart rate, eye tracking, face images, spoken dialog audio data with automatic speech recognition (ASR) transcriptions, game screenshots, gameplay data, game performance data, demographic data, and self-report questionnaires. Each team undergoes up to six consecutive phases: three behavioral tasks, one mission training session, and two collaborative SAR missions. As time-synchronized multimodal data collected under a variety of circumstances, this dataset will support studying a large variety of research questions on topics including teamwork, coordination, plan recognition, affective computing, physiological linkage, entrainment, and dialog understanding.  We provide an initial public release of the de-identified data, along with analyses illustrating the utility of this dataset to both computer scientists and social scientists.</details>

### The Tunnel Effect: Building Data Representations in Deep Neural Networks

**Authors:** Wojciech Masarczyk, Mateusz Ostaszewski, Ehsan Imani, Razvan Pascanu, Piotr Miłoś, Tomasz Trzcinski

**<details><summary>Abstract**</summary>

Deep neural networks are widely known for their remarkable effectiveness across various tasks, with the consensus that deeper networks implicitly learn more complex data representations. This paper shows that sufficiently deep networks trained for supervised image classification split into two distinct parts that contribute to the resulting data representations differently. The initial layers create linearly-separable representations, while the subsequent layers, which we refer to as \textit{the tunnel}, compress these representations and have a minimal impact on the overall performance. We explore the tunnel's behavior through comprehensive empirical studies, highlighting that it emerges early in the training process. Its depth depends on the relation between the network's capacity and task complexity. Furthermore, we show that the tunnel degrades out-of-distribution generalization and discuss its implications for continual learning.</details>

### The Waymo Open Sim Agents Challenge

**Authors:** Nico Montali, John Lambert, Paul Mougin, Alex Kuefler, Nicholas Rhinehart, Michelle Li, Cole Gulino, Tristan Emrich, Zoey Yang, Shimon Whiteson, Brandyn White, Dragomir Anguelov

**<details><summary>Abstract**</summary>

Simulation with realistic, interactive agents represents a key task for autonomous vehicle software development. In this work, we introduce the Waymo Open Sim Agents Challenge (WOSAC). WOSAC is the first public challenge to tackle this task and propose corresponding metrics. The goal of the challenge is to stimulate the design of realistic simulators that can be used to evaluate and train a behavior model for autonomous driving. We outline our evaluation methodology, present results for a number of different baseline simulation agent methods, and analyze several submissions to the 2023 competition which ran from March 16, 2023 to May 23, 2023. The WOSAC evaluation server remains open for submissions and we discuss open problems for the task.</details>

### The expressive power of pooling in Graph Neural Networks

**Authors:** Filippo Maria Bianchi, Veronica Lachi

**<details><summary>Abstract**</summary>

In Graph Neural Networks (GNNs), hierarchical pooling operators generate local summaries of the data by coarsening the graph structure and the vertex features. Considerable attention has been devoted to analyzing the expressive power of message-passing (MP) layers in GNNs, while a study on how graph pooling affects the expressiveness of a GNN is still lacking. Additionally, despite the recent advances in the design of pooling operators, there is not a principled criterion to compare them. In this work, we derive sufficient conditions for a pooling operator to fully preserve the expressive power of the MP layers before it. These conditions serve as a universal and theoretically-grounded criterion for choosing among existing pooling operators or designing new ones. Based on our theoretical findings, we analyze several existing pooling operators and identify those that fail to satisfy the expressiveness conditions. Finally, we introduce an experimental setup to verify empirically the expressive power of a GNN equipped with pooling layers, in terms of its capability to perform a graph isomorphism test.</details>

### Thin and deep Gaussian processes

**Authors:** Daniel Augusto de Souza, Alexander Nikitin, ST John, Magnus Ross, Mauricio A Álvarez, Marc Deisenroth, João Paulo Gomes, Diego Mesquita, César Lincoln Mattos

**<details><summary>Abstract**</summary>

Gaussian processes (GPs) can provide a principled approach to uncertainty quantification with easy-to-interpret kernel hyperparameters, such as the lengthscale, which controls the correlation distance of function values.However, selecting an appropriate kernel can be challenging.Deep GPs avoid manual kernel engineering by successively parameterizing kernels with GP layers, allowing them to learn low-dimensional embeddings of the inputs that explain the output data.Following the architecture of deep neural networks, the most common deep GPs warp the input space layer-by-layer but lose all the interpretability of shallow GPs. An alternative construction is to successively parameterize the lengthscale of a kernel, improving the interpretability but ultimately giving away the notion of learning lower-dimensional embeddings. Unfortunately, both methods are susceptible to particular pathologies which may hinder fitting and limit their interpretability.This work proposes a novel synthesis of both previous approaches: {Thin and Deep GP} (TDGP). Each TDGP layer defines locally linear transformations of the original input data maintaining the concept of latent embeddings while also retaining the interpretation of lengthscales of a kernel. Moreover, unlike the prior solutions, TDGP induces non-pathological manifolds that admit learning lower-dimensional representations.We show with theoretical and experimental results that i) TDGP is, unlike previous models, tailored to specifically discover lower-dimensional manifolds in the input data, ii) TDGP behaves well when increasing the number of layers, and iii) TDGP performs well in standard benchmark datasets.</details>

### Three-Way Trade-Off in Multi-Objective Learning: Optimization, Generalization and Conflict-Avoidance

**Authors:** Lisha Chen, Heshan Fernando, Yiming Ying, Tianyi Chen

**<details><summary>Abstract**</summary>

Multi-objective learning (MOL) often arises in emerging machine learning problems when multiple learning criteria or tasks need to be addressed. Recent works have developed various _dynamic weighting_ algorithms for MOL, including MGDA and its variants, whose central idea is to find an update direction that _avoids conflicts_ among objectives. Albeit its appealing intuition, empirical studies show that dynamic weighting methods may not always outperform static alternatives. To bridge this gap between theory and practice, we focus on a new variant of stochastic MGDA - the Multi-objective gradient with Double sampling (MoDo) algorithm and study its generalization performance and the interplay with optimization through the lens of algorithm stability. We find that the rationale behind MGDA -- updating along conflict-avoidant direction - may \emph{impede} dynamic weighting algorithms from achieving the optimal ${\cal O}(1/\sqrt{n})$ population risk, where $n$ is the number of training samples. We further highlight the variability of dynamic weights and their impact on the three-way trade-off among optimization, generalization, and conflict avoidance that is unique in MOL. Code is available at https://github.com/heshandevaka/Trade-Off-MOL.</details>

### Time-Reversed Dissipation Induces Duality Between Minimizing Gradient Norm and Function Value

**Authors:** Jaeyeon Kim, Asuman Ozdaglar, Chanwoo Park, Ernest Ryu

**<details><summary>Abstract**</summary>

In convex optimization, first-order optimization methods efficiently minimizing function values have been a central subject study since Nesterov's seminal work of 1983. Recently, however, Kim and Fessler's OGM-G and Lee et al.'s FISTA-G have been presented as alternatives that efficiently minimize the gradient magnitude instead. In this paper, we present H-duality, which represents a surprising one-to-one correspondence between methods efficiently minimizing function values and methods efficiently minimizing gradient magnitude. In continuous-time formulations, H-duality corresponds to reversing the time dependence of the dissipation/friction term. To the best of our knowledge, H-duality is different from Lagrange/Fenchel duality and is distinct from any previously known duality or symmetry relations. Using H-duality, we obtain a clearer understanding of the symmetry between Nesterov's method and OGM-G, derive a new class of methods efficiently reducing gradient magnitudes of smooth convex functions, and find a new composite minimization method that is simpler and faster than FISTA-G.</details>

### Toolbox for Multimodal Learn (scikit-multimodallearn)

**Authors:** Dominique Benielli, Baptiste Bauvin, Sokol Koço, Riikka Huusari, Cécile Capponi, Hachem Kadri, François Laviolette

**<details><summary>Abstract**</summary>

scikit-multimodallearn is a Python library for multimodal supervised learning, licensed under Free BSD, and compatible with the well-known scikit-learn toolbox (Fabian Pedregosa, 2011). This paper details the content of the library, including a specific multimodal data formatting and classification and regression algorithms. Use cases and examples are also provided.</details>

### Towards Better Dynamic Graph Learning: New Architecture and Unified Library

**Authors:** Le Yu, Leilei Sun, Bowen Du, Weifeng Lv

**<details><summary>Abstract**</summary>

We propose DyGFormer, a new Transformer-based architecture for dynamic graph learning. DyGFormer is conceptually simple and only needs to learn from nodes' historical first-hop interactions by: (1) a neighbor co-occurrence encoding scheme that explores the correlations of the source node and destination node based on their historical sequences; (2) a patching technique that divides each sequence into multiple patches and feeds them to Transformer, allowing the model to effectively and efficiently benefit from longer histories. We also introduce DyGLib, a unified library with standard training pipelines, extensible coding interfaces, and comprehensive evaluating protocols to promote reproducible, scalable, and credible dynamic graph learning research. By performing exhaustive experiments on thirteen datasets for dynamic link prediction and dynamic node classification tasks, we find that DyGFormer achieves state-of-the-art performance on most of the datasets, demonstrating its effectiveness in capturing nodes' correlations and long-term temporal dependencies. Moreover, some results of baselines are inconsistent with previous reports, which may be caused by their diverse but less rigorous implementations, showing the importance of DyGLib. All the used resources are publicly available at https://github.com/yule-BUAA/DyGLib.</details>

### Towards Data-Agnostic Pruning At Initialization: What Makes a Good Sparse Mask?

**Authors:** Hoang Pham, The Anh Ta, Shiwei Liu, Lichuan Xiang, Dung Le, Hongkai Wen, Long Tran-Thanh

**<details><summary>Abstract**</summary>

Pruning at initialization (PaI) aims to remove weights of neural networks before training in pursuit of training efficiency besides the inference. While off-the-shelf PaI methods manage to find trainable subnetworks that  outperform random pruning, their performance in terms of both accuracy and computational reduction is far from satisfactory compared to post-training pruning and the understanding of PaI is missing.  For instance, recent studies show that existing PaI methods only able to find good layerwise sparsities not weights, as the discovered subnetworks are surprisingly resilient against layerwise random mask shuffling and weight re-initialization.In this paper, we study PaI from a brand-new perspective -- the topology of subnetworks. In particular, we propose a principled framework for analyzing the performance of Pruning and Initialization (PaI) methods with two quantities, namely, the number of effective paths and effective nodes. These quantities allow for a more comprehensive understanding of PaI methods, giving us an accurate assessment of different subnetworks at initialization. We systematically analyze the behavior of various PaI methods through our framework and observe a guiding principle for constructing effective subnetworks: *at a specific sparsity, the top-performing subnetwork always presents a good balance between the number of effective nodes and the number of effective paths.*Inspired by this observation, we present a novel data-agnostic pruning method by solving a multi-objective optimization problem. By conducting extensive experiments across different architectures and datasets, our results demonstrate that our approach outperforms state-of-the-art PaI methods while it is able to discover subnetworks that have much lower inference FLOPs (up to 3.4$\times$). Code will be fully released.</details>

### Towards Federated Foundation Models: Scalable Dataset Pipelines for Group-Structured Learning

**Authors:** Zachary Charles, Nicole Mitchell, Krishna Pillutla, Michael Reneer, Zachary Garrett

**<details><summary>Abstract**</summary>

We introduce Dataset Grouper, a library to create large-scale group-structured (e.g., federated) datasets, enabling federated learning simulation at the scale of foundation models. This library facilitates the creation of group-structured versions of existing datasets based on user-specified partitions, and directly leads to a variety of useful heterogeneous datasets that can be plugged into existing software frameworks. Dataset Grouper offers three key advantages. First, it scales to settings where even a single group's dataset is too large to fit in memory. Second, it provides flexibility, both in choosing the base (non-partitioned) dataset and in defining partitions. Finally, it is framework-agnostic. We empirically demonstrate that Dataset Grouper enables large-scale federated language modeling simulations on datasets that are orders of magnitude larger than in previous work, allowing for federated training of language models with hundreds of millions, and even billions, of parameters. Our experimental results show that algorithms like FedAvg operate more as meta-learning methods than as empirical risk minimization methods at this scale, suggesting their utility in downstream personalization and task-specific adaptation. Dataset Grouper is available at https://github.com/google-research/dataset_grouper.</details>

### Towards Hybrid-grained Feature Interaction Selection for Deep Sparse Network

**Authors:** Fuyuan Lyu, Xing Tang, Dugang Liu, Chen Ma, Weihong Luo, Liang Chen, xiuqiang He, Xue (Steve) Liu

**<details><summary>Abstract**</summary>

Deep sparse networks are widely investigated as a neural network architecture for prediction tasks with high-dimensional sparse features, with which feature interaction selection is a critical component. While previous methods primarily focus on how to search feature interaction in a coarse-grained space, less attention has been given to a finer granularity. In this work, we introduce a hybrid-grained feature interaction selection approach that targets both feature field and feature value for deep sparse networks. To explore such expansive space, we propose a decomposed space which is calculated on the fly. We then develop a selection algorithm called OptFeature, which efficiently selects the feature interaction from both the feature field and the feature value simultaneously. Results from experiments on three large real-world benchmark datasets demonstrate that OptFeature performs well in terms of accuracy and efficiency. Additional studies support the feasibility of our method. All source code are publicly availableootnote{https://anonymous.4open.science/r/OptFeature-Anonymous}.</details>

### Towards Personalized Federated Learning via Heterogeneous Model Reassembly

**Authors:** Jiaqi Wang, Xingyi Yang, Suhan Cui, Liwei Che, Lingjuan Lyu, Dongkuan (DK) Xu, Fenglong Ma

**<details><summary>Abstract**</summary>

This paper focuses on addressing the practical yet challenging problem of model heterogeneity in federated learning, where clients possess models with different network structures. To track this problem, we propose a novel framework called pFedHR, which leverages heterogeneous model reassembly to achieve personalized federated learning. In particular, we approach the problem of heterogeneous model personalization as a model-matching optimization task on the server side. Moreover, pFedHR automatically and dynamically generates informative and diverse personalized candidates with minimal human intervention. Furthermore, our proposed heterogeneous model reassembly technique mitigates the adverse impact introduced by using public data with different distributions from the client data to a certain extent. Experimental results demonstrate that pFedHR outperforms baselines on three datasets under both IID and Non-IID settings. Additionally, pFedHR effectively reduces the adverse impact of using different public data and dynamically generates diverse personalized models in an automated manner.</details>

### Towards a Comprehensive Benchmark for High-Level Synthesis Targeted to FPGAs

**Authors:** Yunsheng Bai, Atefeh Sohrabizadeh, Zongyue Qin, Ziniu Hu, Yizhou Sun, Jason Cong

**<details><summary>Abstract**</summary>

High-level synthesis (HLS) aims to raise the abstraction layer in hardware design, enabling the design of domain-specific accelerators (DSAs) like field-programmable gate arrays (FPGAs) using C/C++ instead of hardware description languages (HDLs). Compiler directives in the form of pragmas play a crucial role in modifying the microarchitecture within the HLS framework. However, the space of possible microarchitectures grows exponentially with the number of pragmas. Moreover, the evaluation of each candidate design using the HLS tool consumes significant time, ranging from minutes to hours, leading to a time-consuming optimization process. To accelerate this process, machine learning models have been used to predict design quality in milliseconds. However, existing open-source datasets for training such models are limited in terms of design complexity and available optimizations. In this paper, we present HLSyn, the first benchmark that addresses these limitations. It contains more complex programs with a wider range of optimization pragmas, making it a comprehensive dataset for training and evaluating design quality prediction models. The HLSyn benchmark consists of 42 unique programs/kernels, resulting in over 42,000 labeled designs. We conduct an extensive comparison of state-of-the-art baselines to assess their effectiveness in predicting design quality. As an ongoing project, we anticipate expanding the HLSyn benchmark in terms of both quantity and variety of programs to further support the development of this field.</details>

### [Spotlight] Tracr: Compiled Transformers as a Laboratory for Interpretability

**Authors:** David Lindner, Janos Kramar, Sebastian Farquhar, Matthew Rahtz, Tom McGrath, Vladimir Mikulik

**<details><summary>Abstract**</summary>

We show how to "compile" human-readable programs into standard decoder-only transformer models. Our compiler, Tracr, generates models with known structure. This structure can be used to design experiments. For example, we use it to study "superposition" in transformers that execute multi-step algorithms. Additionally, the known structure of Tracr-compiled models can serve as _ground-truth_ for evaluating interpretability methods. Commonly, because the "programs" learned by transformers are unknown it is unclear whether an interpretation succeeded. We demonstrate our approach by implementing and examining programs including computing token frequencies, sorting, and parenthesis checking. We provide an open-source implementation of Tracr at https://github.com/google-deepmind/tracr.</details>

### Training Your Image Restoration Network Better with  Random Weight Network as Optimization Function

**Authors:** man zhou, Naishan Zheng, Yuan Xu, Chun-Le Guo, Chongyi Li

**<details><summary>Abstract**</summary>

The blooming progress made in deep learning-based image restoration has been largely attributed to the availability of high-quality, large-scale datasets and advanced network structures. However, optimization functions such as L_1 and L_2 are still de facto. In this study, we propose to  investigate new optimization functions to improve image restoration performance. Our key insight is  that ``random weight network can be acted as a constraint for training better image restoration networks''. However, not all random weight networks are suitable as constraints. We draw inspiration from Functional theory and show that alternative random weight networks should be represented in the form of a strict mathematical manifold. We explore the potential of our random weight network prototypes that satisfy this requirement:  Taylor's unfolding network, invertible neural network, central difference convolution, and zero-order filtering. We investigate these prototypes from four aspects: 1)   random weight strategies, 2)  network architectures, 3)   network depths, and 4) combinations of random weight networks. Furthermore, we devise the random weight in two variants:  the weights are randomly initialized only once during the entire training procedure, and  the weights are randomly initialized in each training epoch. Our approach can be directly integrated into existing networks without incurring additional training and testing computational costs. We perform extensive experiments across multiple image restoration tasks, including image denoising, low-light image enhancement, and guided image super-resolution to demonstrate the consistent performance gains achieved by our method.  Upon acceptance of this paper, we will release the code.</details>

### [Spotlight] Trans-Dimensional Generative Modeling via Jump Diffusion Models

**Authors:** Andrew Campbell, William Harvey, Christian Weilbach, Valentin De Bortoli, Thomas Rainforth, Arnaud Doucet

**<details><summary>Abstract**</summary>

We propose a new class of generative model that naturally handles data of varying dimensionality by jointly modeling the state and dimension of each datapoint. The generative process is formulated as a jump diffusion process that makes jumps between different dimensional spaces. We first define a dimension destroying forward noising process, before deriving the dimension creating time-reversed generative process along with a novel evidence lower bound training objective for learning to approximate it.Simulating our learned approximation to the time-reversed generative process then provides an effective way of sampling data of varying dimensionality by jointly generating state values and dimensions. We demonstrate our approach on molecular and video datasets of varying dimensionality, reporting better compatibility with test-time diffusion guidance imputation tasks and improved interpolation capabilities versus fixed dimensional models that generate state values and dimensions separately.</details>

### TransHP: Image Classification with Hierarchical Prompting

**Authors:** Wenhao Wang, Yifan Sun, Wei Li, Yi Yang

**<details><summary>Abstract**</summary>

This paper explores a hierarchical prompting mechanism for the hierarchical image classification (HIC) task. Different from prior HIC methods, our hierarchical prompting is the first to explicitly inject ancestor-class information as a tokenized hint that benefits the descendant-class discrimination. We think it well imitates human visual recognition, i.e., humans may use the ancestor class as a prompt to draw focus on the subtle differences among descendant classes. We model this prompting mechanism into a Transformer with Hierarchical Prompting (TransHP). TransHP consists of three steps: 1) learning a set of prompt tokens to represent the coarse (ancestor) classes, 2) on-the-fly predicting the coarse class of the input image at an intermediate block, and 3) injecting the prompt token of the predicted coarse class into the intermediate feature. Though the parameters of TransHP maintain the same for all input images, the injected coarse-class prompt conditions (modifies) the subsequent feature extraction and encourages a dynamic focus on relatively subtle differences among the descendant classes. Extensive experiments show that TransHP improves image classification on accuracy (e.g., improving ViT-B/16 by +2.83% ImageNet classification accuracy), training data efficiency (e.g., +12.69% improvement under 10% ImageNet training data), and model explainability. Moreover, TransHP also performs favorably against prior HIC methods, showing that TransHP well exploits the hierarchical information.</details>

### Transformer as a hippocampal memory consolidation model based on NMDAR-inspired nonlinearity

**Authors:** Dong Kyum Kim, Jea Kwon, Meeyoung Cha, C. Lee

**<details><summary>Abstract**</summary>

The hippocampus plays a critical role in learning, memory, and spatial representation, processes that depend on the NMDA receptor (NMDAR). Inspired by recent findings that compare deep learning models to the hippocampus, we propose a new nonlinear activation function that mimics NMDAR dynamics. NMDAR-like nonlinearity has a beneficial role in shifting short-term working memory into long-term reference memory in transformers, thus enhancing a process that is similar to memory consolidation in the mammalian brain. We design a navigation task assessing these two memory functions and show that manipulating the activation function (i.e., mimicking the Mg$^{2+}$-gating of NMDAR) disrupts long-term memory processes. Our experiments suggest that place cell-like functions and reference memory reside in the feed-forward network layer of transformers and that nonlinearity drives these processes. We discuss the role of NMDAR-like nonlinearity in establishing this striking resemblance between transformer architecture and hippocampal spatial representation.</details>

### Transitivity Recovering Decompositions: Interpretable and Robust Fine-Grained Relationships

**Authors:** ABHRA CHAUDHURI, Massimiliano Mancini, Zeynep Akata, Anjan Dutta

**<details><summary>Abstract**</summary>

Recent advances in fine-grained representation learning leverage local-to-global (emergent) relationships for achieving state-of-the-art results. The relational representations relied upon by such methods, however, are abstract. We aim to deconstruct this abstraction by expressing them as interpretable graphs over image views. We begin by theoretically showing that abstract relational representations are nothing but a way of recovering transitive relationships among local views. Based on this, we design Transitivity Recovering Decompositions (TRD), a graph-space search algorithm that identifies interpretable equivalents of abstract emergent relationships at both instance and class levels, and with no post-hoc computations. We additionally show that TRD is provably robust to noisy views, with empirical evidence also supporting this finding. The latter allows TRD to perform at par or even better than the state-of-the-art, while being fully interpretable. Implementation is available at https://github.com/abhrac/trd.</details>

### UDC-SIT: A Real-World Dataset for Under-Display Cameras

**Authors:** Kyusu Ahn, Byeonghyun Ko, HyunGyu Lee, Chanwoo Park, Jaejin Lee

**<details><summary>Abstract**</summary>

Under Display Camera (UDC) is a novel imaging system that mounts a digital camera lens beneath a display panel with the panel covering the camera. However, the display panel causes severe degradation to captured images, such as low transmittance, blur, noise, and flare. The restoration of UDC-degraded images is challenging because of the unique luminance and diverse patterns of flares. Existing UDC dataset studies focus on unrealistic or synthetic UDC degradation rather than real-world UDC images. In this paper, we propose a real-world UDC dataset called UDC-SIT. To obtain the non-degraded and UDC-degraded images for the same scene, we propose an image-capturing system and an image alignment technique that exploits discrete Fourier transform (DFT) to align a pair of captured images. UDC-SIT also includes comprehensive annotations missing from other UDC datasets, such as light source, day/night, indoor/outdoor, and flare components (e.g., shimmers, streaks, and glares). We compare UDC-SIT with four existing representative UDC datasets and present the problems with existing UDC datasets. To show UDC-SIT's effectiveness, we compare UDC-SIT and a representative synthetic UDC dataset using four representative learnable image restoration models. The result indicates that the models trained with the synthetic UDC dataset are impractical because the synthetic UDC dataset does not reflect the actual characteristics of UDC-degraded images. UDC-SIT can enable further exploration in the UDC image restoration area and provide better insights into the problem. UDC-SIT is available at: https://github.com/mcrl/UDC-SIT.</details>

### URL: A Representation Learning Benchmark for Transferable Uncertainty Estimates

**Authors:** Michael Kirchhof, Bálint Mucsányi, Seong Joon Oh, Dr. Enkelejda Kasneci

**<details><summary>Abstract**</summary>

Representation learning has significantly driven the field to develop pretrained models that can act as a valuable starting point when transferring to new datasets. With the rising demand for reliable machine learning and uncertainty quantification, there is a need for pretrained models that not only provide embeddings but also transferable uncertainty estimates. To guide the development of such models, we propose the Uncertainty-aware Representation Learning (URL) benchmark. Besides the transferability of the representations, it also measures the zero-shot transferability of the uncertainty estimate using a novel metric. We apply URL to evaluate ten uncertainty quantifiers that are pretrained on ImageNet and transferred to eight downstream datasets. We find that approaches that focus on the uncertainty of the representation itself or estimate the prediction risk directly outperform those that are based on the probabilities of upstream classes. Yet, achieving transferable uncertainty quantification remains an open challenge. Our findings indicate that it is not necessarily in conflict with traditional representation learning goals. Code is available at [https://github.com/mkirchhof/url](https://github.com/mkirchhof/url).</details>

### Uncertainty Estimation for Safety-critical Scene Segmentation via Fine-grained Reward Maximization

**Authors:** Hongzheng Yang, Cheng Chen, Yueyao CHEN, Scheppach, Hon Chi Yip, DOU QI

**<details><summary>Abstract**</summary>

Uncertainty estimation plays an important role for future reliable deployment of deep segmentation models in safety-critical scenarios such as medical applications. However, existing methods for uncertainty estimation have been limited by the lack of explicit guidance for calibrating the prediction risk and model confidence. In this work, we propose a novel fine-grained reward maximization (FGRM) framework, to address uncertainty estimation by directly utilizing an uncertainty metric related reward function with a reinforcement learning based model tuning algorithm. This would benefit the model uncertainty estimation with direct optimization guidance for model calibration. Specifically, our method designs a new uncertainty estimation reward function using the calibration metric, which is maximized to fine-tune an evidential learning pre-trained segmentation model for calibrating prediction risk. Importantly, we innovate an effective fine-grained parameter update scheme, which imposes fine-grained reward-weighting of each network parameter according to the parameter importance quantified by the fisher information matrix. To the best of our knowledge, this is the first work exploring reward optimization for model uncertainty estimation in safety-critical vision tasks. The effectiveness of our method is demonstrated on two large safety-critical surgical scene segmentation datasets under two different uncertainty estimation settings. With real-time one forward pass at inference, our method outperforms state-of-the-art methods by a clear margin on all the calibration metrics of uncertainty estimation, while maintaining a high task accuracy for the segmentation results. Code is available at https://github.com/med-air/FGRM.</details>

### [Spotlight] Uncertainty Quantification over Graph with Conformalized Graph Neural Networks

**Authors:** Kexin Huang, Ying Jin, Emmanuel Candes, Jure Leskovec

**<details><summary>Abstract**</summary>

Graph Neural Networks (GNNs) are powerful machine learning prediction models on graph-structured data. However, GNNs lack rigorous uncertainty estimates, limiting their reliable deployment in settings where the cost of errors is significant. We propose conformalized GNN (CF-GNN), extending conformal prediction (CP) to graph-based models for guaranteed uncertainty estimates. Given an entity in the graph, CF-GNN produces a prediction set/interval that provably contains the true label with pre-defined coverage probability (e.g. 90%). We establish a permutation invariance condition that enables the validity of CP on graph data and provide an exact characterization of the test-time coverage. Moreover, besides valid coverage, it is crucial to reduce the prediction set size/interval length for practical use. We observe a key connection between non-conformity scores and network structures, which motivates us to develop a topology-aware output correction model that learns to update the prediction and produces more efficient prediction sets/intervals. Extensive experiments show that CF-GNN achieves any pre-defined target marginal coverage while significantly reducing the prediction set/interval size by up to 74% over the baselines. It also empirically achieves satisfactory conditional coverage over various raw and network features.</details>

### Uncovering Meanings of Embeddings via Partial Orthogonality

**Authors:** Yibo Jiang, Bryon Aragam, Victor Veitch

**<details><summary>Abstract**</summary>

Machine learning tools often rely on embedding text as vectors of real numbers.In this paper, we study how the semantic structure of language is encoded in the algebraic structure of such embeddings.Specifically, we look at a notion of  "semantic independence" capturing the idea that, e.g., "eggplant" and "tomato" are independent given "vegetable". Although such examples are intuitive, it is difficult to formalize such a notion of semantic independence. The key observation here is that any sensible formalization should obey a set of so-called independence axioms, and thus any algebraic encoding of this structure should also obey these axioms. This leads us naturally to use partial orthogonality as the relevant algebraic structure. We develop theory and methods that allow us to demonstrate that partial orthogonality does indeed capture semantic independence.Complementary to this, we also introduce the concept of independence preserving embeddings where embeddings preserve the conditional independence structures of a distribution, and we prove the existence of such embeddings and approximations to them.</details>

### Uncovering Neural Scaling Laws in Molecular Representation Learning

**Authors:** Dingshuo Chen, Yanqiao Zhu, Jieyu Zhang, Yuanqi Du, Zhixun Li, Qiang Liu, Shu Wu, Liang Wang

**<details><summary>Abstract**</summary>

Molecular Representation Learning (MRL) has emerged as a powerful tool for drug and materials discovery in a variety of tasks such as virtual screening and inverse design. While there has been a surge of interest in advancing model-centric techniques, the influence of both data quantity and quality on molecular representations is not yet clearly understood within this field. In this paper, we delve into the neural scaling behaviors of MRL from a data-centric viewpoint, examining four key dimensions: (1) data modalities, (2) dataset splitting, (3) the role of pre-training, and (4) model capacity.Our empirical studies confirm a consistent power-law relationship between data volume and MRL performance across these dimensions. Additionally, through detailed analysis, we identify potential avenues for improving learning efficiency.To challenge these scaling laws, we adapt seven popular data pruning strategies to molecular data and benchmark their performance. Our findings underline the importance of data-centric MRL and highlight possible directions for future research.</details>

### Understanding Deep Gradient Leakage via Inversion Influence Functions

**Authors:** Haobo Zhang, Junyuan Hong, Yuyang Deng, Mehrdad Mahdavi, Jiayu Zhou

**<details><summary>Abstract**</summary>

Deep Gradient Leakage (DGL) is a highly effective attack that recovers private training images from gradient vectors. This attack casts significant privacy challenges on distributed learning from clients with sensitive data, where clients are required to share gradients. Defending against such attacks requires but lacks an understanding of when and how privacy leakage happens, mostly because of the black-box nature of deep networks. In this paper, we propose a novel Inversion Influence Function (I$^2$F) that establishes a closed-form connection between the recovered images and the private gradients by implicitly solving the DGL problem. Compared to directly solving DGL, I$^2$F is scalable for analyzing deep networks, requiring only oracle access to gradients and Jacobian-vector products. We empirically demonstrate that I$^2$F effectively approximated the DGL generally on different model architectures, datasets, attack implementations, and noise-based defenses. With this novel tool, we provide insights into effective gradient perturbation directions, the unfairness of privacy protection, and privacy-preferred model initialization. Our codes are provided in https://github.com/illidanlab/inversion-influence-function.</details>

### Understanding How Consistency Works in Federated Learning via Stage-wise Relaxed Initialization

**Authors:** Yan Sun, Li Shen, Dacheng Tao

**<details><summary>Abstract**</summary>

Federated learning (FL) is a distributed paradigm that coordinates massive local clients to collaboratively train a global model via stage-wise local training processes on the heterogeneous dataset.  Previous works have implicitly studied that FL suffers from the "client-drift" problem, which is caused by the inconsistent optimum across local clients. However, till now it still lacks solid theoretical analysis to explain the impact of this local inconsistency.   To alleviate the negative impact of the "client drift" and explore its substance in FL, in this paper, we first design an efficient FL algorithm FedInit, which allows employing the personalized relaxed initialization state at the beginning of each local training stage. Specifically, FedInit initializes the local state by moving away from the current global state towards the reverse direction of the latest local state. This relaxed initialization helps to revise the local divergence and enhance the local consistency level.  Moreover, to further understand how inconsistency disrupts performance in FL, we introduce the excess risk analysis and study the divergence term to investigate the test error of the proposed FedInit method. Our studies show that on the non-convex objectives, optimization error is not sensitive to this local inconsistency, while it mainly affects the generalization error bound in FedInit.   Extensive experiments are conducted to validate this conclusion. Our proposed FedInit could achieve state-of-the-art (SOTA) results compared to several advanced benchmarks without any additional costs. Meanwhile, stage-wise relaxed initialization could also be incorporated into the current advanced algorithms to achieve higher performance in the FL paradigm.</details>

### Understanding Social Reasoning in Language Models with Language Models

**Authors:** Kanishk Gandhi, Jan-Philipp Fraenken, Tobias Gerstenberg, Noah Goodman

**<details><summary>Abstract**</summary>

As Large Language Models (LLMs) become increasingly integrated into our everyday lives, understanding their ability to comprehend human mental states becomes critical for ensuring effective interactions. However, despite the recent attempts to assess the Theory-of-Mind (ToM) reasoning capabilities of LLMs, the degree to which these models can align with human ToM remains a nuanced topic of exploration. This is primarily due to two distinct challenges: (1) the presence of inconsistent results from previous evaluations, and (2) concerns surrounding the validity of existing evaluation methodologies. To address these challenges, we present a novel framework for procedurally generating evaluations with LLMs by populating causal templates. Using our framework, we create a new social reasoning benchmark (BigToM) for LLMs which consists of 25 controls and 5,000 model-written evaluations. We find that human participants rate the quality of our benchmark higher than previous crowd-sourced evaluations and comparable to expert-written evaluations. Using BigToM, we evaluate the social reasoning capabilities of a variety of LLMs and compare model performances with human performance. Our results suggest that GPT4 has ToM capabilities that mirror human inference patterns, though less reliable, while other LLMs struggle.</details>

### Understanding and Addressing the Pitfalls of Bisimulation-based Representations in Offline Reinforcement Learning

**Authors:** Hongyu Zang, Xin Li, Leiji Zhang, Yang Liu, Baigui Sun, Riashat Islam, Remi Tachet des Combes, Romain Laroche

**<details><summary>Abstract**</summary>

While bisimulation-based approaches hold promise for learning robust state representations for Reinforcement Learning (RL) tasks, their efficacy in offline RL tasks has not been up to par. In some instances, their performance has even significantly underperformed alternative methods. We aim to understand why bisimulation methods succeed in online settings, but falter in offline tasks. Our analysis reveals that missing transitions in the dataset are particularly harmful to the bisimulation principle, leading to ineffective estimation. We also shed light on the critical role of reward scaling in bounding the scale of bisimulation measurements and of the value error they induce. Based on these findings, we propose to apply the expectile operator for representation learning to our offline RL setting, which helps to prevent overfitting to incomplete data. Meanwhile, by introducing an appropriate reward scaling strategy, we avoid the risk of feature collapse in representation space. We implement these recommendations on two state-of-the-art bisimulation-based algorithms, MICo and SimSR, and demonstrate performance gains on two benchmark suites: D4RL and Visual D4RL. Codes are provided at \url{https://github.com/zanghyu/Offline_Bisimulation}.</details>

### Understanding the detrimental class-level effects of data augmentation

**Authors:** Polina Kirichenko, Mark Ibrahim, Randall Balestriero, Diane Bouchacourt, Shanmukha Ramakrishna Vedantam, Hamed Firooz, Andrew Wilson

**<details><summary>Abstract**</summary>

Data augmentation (DA) encodes invariance and provides implicit regularization critical to a model's performance in image classification tasks. However, while DA improves average accuracy, recent studies have shown that its impact can be highly class dependent: achieving optimal average accuracy comes at the cost of significantly hurting individual class accuracy by as much as 20% on ImageNet. There has been little progress in resolving class-level accuracy drops due to a limited understanding of these effects. In this work, we present a framework for understanding how DA interacts with class-level learning dynamics. Using higher-quality multi-label annotations on ImageNet, we systematically categorize the affected classes and find that the majority are inherently ambiguous, co-occur, or involve fine-grained distinctions, while DA controls the model's bias towards one of the closely related classes. While many of the previously reported performance drops are explained by multi-label annotations, we identify other sources of accuracy degradations by analyzing class confusions. We show that simple class-conditional augmentation strategies informed by our framework improve performance on the negatively affected classes.</details>

### UniControl: A Unified Diffusion Model for Controllable Visual Generation In the Wild

**Authors:** Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, Stefano Ermon, Yun Fu, Ran Xu

**<details><summary>Abstract**</summary>

Achieving machine autonomy and human control often represent divergent objectives in the design of interactive AI systems. Visual generative foundation models such as Stable Diffusion show promise in navigating these goals, especially when prompted with arbitrary languages. However, they often fall short in generating images with spatial, structural, or geometric controls. The integration of such controls, which can accommodate various visual conditions in a single unified model, remains an unaddressed challenge. In response, we introduce UniControl, a new generative foundation model that consolidates a wide array of controllable condition-to-image (C2I) tasks within a singular framework, while still allowing for arbitrary language prompts. UniControl enables pixel-level-precise image generation, where visual conditions primarily influence the generated structures and language prompts guide the style and context. To equip UniControl with the capacity to handle diverse visual conditions, we augment pretrained text-to-image diffusion models and introduce a task-aware HyperNet to modulate the diffusion models, enabling the adaptation to different C2I tasks simultaneously. Trained on nine unique C2I tasks, UniControl demonstrates impressive zero-shot generation abilities with unseen visual conditions. Experimental results show that UniControl often surpasses the performance of single-task-controlled methods of comparable model sizes. This control versatility positions UniControl as a significant advancement in the realm of controllable visual generation.</details>

### Universality and Limitations of Prompt Tuning

**Authors:** Yihan Wang, Jatin Chauhan, Wei Wang, Cho-Jui Hsieh

**<details><summary>Abstract**</summary>

Despite the demonstrated empirical efficacy of prompt tuning to adapt a pretrained language model for a new task, the theoretical underpinnings of the difference between "tuning parameters before the input" against "the tuning of model weights" are limited. We thus take one of the first steps to understand the role of soft-prompt tuning for transformer-based architectures. By considering a general purpose architecture, we analyze prompt tuning from the lens of both: universal approximation and limitations with finite-depth fixed-weight pretrained transformers for continuous-valued functions. Our universality result guarantees the existence of a strong transformer with a prompt to approximate any sequence-to-sequence function in the set of Lipschitz functions. The limitations of prompt tuning for limited-depth transformers are first proved by constructing a set of datasets, that cannot be memorized by a prompt of any length for a given single encoder layer. We also provide a lower bound on the required number of tunable prompt parameters and compare the result with the number of parameters required for a low-rank update (based on LoRA) for a single-layer setting. We finally extend our analysis to multi-layer settings by providing sufficient conditions under which the transformer can at best learn datasets from invertible functions only. Our theoretical claims are also corroborated by empirical results.</details>

### Unleashing the Full Potential of Product Quantization for Large-Scale Image Retrieval

**Authors:** Yu Liang, Shiliang Zhang, Li Ken Li, Xiaoyu Wang

**<details><summary>Abstract**</summary>

Due to its promising performance, deep hashing has become a prevalent method for approximate nearest neighbors search (ANNs). However, most of current deep hashing methods are validated on relatively small-scale datasets, leaving potential threats when are applied to large-scale real-world scenarios. Specifically, they can be constrained either by the computational cost due to the large number of training categories and samples, or unsatisfactory accuracy. To tackle those issues, we propose a novel deep hashing framework based on product quantization (PQ). It uses a softmax-based differentiable PQ branch to learn a set of predefined PQ codes of the classes. Our method is easy to implement, does not involve large-scale matrix operations, and learns highly discriminate compact codes. We validate our method on multiple large-scaled datasets, including ImageNet100, ImageNet1K, and Glint360K, where the category size scales from 100 to 360K and sample number scales from 10K to 17 million, respectively. Extensive experiments demonstrate the superiority of our method. Code is available at https://github.com/yuleung/FPPQ.</details>

### [Spotlight] Unpaired Multi-Domain Causal Representation Learning

**Authors:** Nils Sturma, Chandler Squires, Mathias Drton, Caroline Uhler

**<details><summary>Abstract**</summary>

The goal of causal representation learning is to find a representation of data that consists of causally related latent variables. We consider a setup where one has access to data from multiple domains that potentially share a causal representation. Crucially, observations in different domains are assumed to be unpaired, that is, we only observe the marginal distribution in each domain but not their joint distribution. In this paper, we give sufficient conditions for identifiability of the joint distribution and the shared causal graph in a linear setup. Identifiability holds if we can uniquely recover the joint distribution and the shared causal representation from the marginal distributions in each domain. We transform our results into a practical method to recover the shared latent causal graph.</details>

### Unsupervised Semantic Correspondence Using Stable Diffusion

**Authors:** Eric Hedlin, Gopal Sharma, Shweta Mahajan, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, Kwang Moo Yi

**<details><summary>Abstract**</summary>

Text-to-image diffusion models are now capable of generating images that are often indistinguishable from real images. To generate such images, these models must understand the semantics of the objects they are asked to generate. In this work we show that, without any training, one can leverage this semantic knowledge within diffusion models to find semantic correspondences – locations in multiple images that have the same semantic meaning. Specifically, given an image, we optimize the prompt embeddings of these models for maximum attention on the regions of interest. These optimized embeddings capture semantic information about the location, which can then be transferred to another image. By doing so we obtain results on par with the strongly supervised state of the art on the PF-Willow dataset and significantly outperform (20.9% relative for the SPair-71k dataset) any existing weakly- or unsupervised method on PF-Willow, CUB-200 and SPair-71k datasets.</details>

### VAST: A Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset

**Authors:** Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, Jing Liu

**<details><summary>Abstract**</summary>

Vision and text have been fully explored  in contemporary video-text foundational models, while other modalities such as audio and subtitles in videos have not received sufficient attention. In this paper, we resort to establish connections between multi-modality video tracks, including Vision, Audio, and Subtitle, and Text by exploring an automatically generated large-scale omni-modality video caption dataset called VAST-27M. Specifically, we first collect 27 million open-domain video clips and separately train a vision and an audio captioner to generate vision and audio captions. Then, we employ an off-the-shelf Large Language Model (LLM) to integrate the generated captions, together with subtitles and instructional prompts into omni-modality captions. Based on the proposed VAST-27M dataset, we train an omni-modality video-text foundational model named VAST, which can perceive and process vision, audio, and subtitle modalities from video, and better support various tasks including  vision-text, audio-text, and multi-modal video-text tasks (retrieval, captioning and QA). Extensive experiments have been conducted to demonstrate the effectiveness of our proposed VAST-27M corpus and VAST foundation model. VAST achieves 22 new state-of-the-art results on various cross-modality benchmarks.</details>

### Variational Gibbs Inference for Statistical Model Estimation from Incomplete Data

**Authors:** Vaidotas Simkus, Benjamin Rhodes, Michael Gutmann

**<details><summary>Abstract**</summary>

Statistical models are central to machine learning with broad applicability across a range of downstream tasks. The models are controlled by free parameters that are typically estimated from data by maximum-likelihood estimation or approximations thereof. However, when faced with real-world data sets many of the models run into a critical issue: they are formulated in terms of fully-observed data, whereas in practice the data sets are plagued with missing data. The theory of statistical model estimation from incomplete data is conceptually similar to the estimation of latent-variable models, where powerful tools such as variational inference (VI) exist. However, in contrast to standard latent-variable models, parameter estimation with incomplete data often requires estimating exponentially-many conditional distributions of the missing variables, hence making standard VI methods intractable. We address this gap by introducing variational Gibbs inference (VGI), a new general-purpose method to estimate the parameters of statistical models from incomplete data. We validate VGI on a set of synthetic and real-world estimation tasks, estimating important machine learning models such as variational autoencoders and normalising flows from incomplete data. The proposed method, whilst general-purpose, achieves competitive or better performance than existing model-specific estimation methods.</details>

### VidChapters-7M: Video Chapters at Scale

**Authors:** Antoine Yang, Arsha Nagrani, Ivan Laptev, Josef Sivic, Cordelia Schmid

**<details><summary>Abstract**</summary>

Segmenting untrimmed videos into chapters enables users to quickly navigate to the information of their interest. This important topic has been understudied due to the lack of publicly released datasets. To address this issue, we present VidChapters-7M, a dataset of 817K user-chaptered videos including 7M chapters in total. VidChapters-7M is automatically created from videos online in a scalable manner by scraping user-annotated chapters and hence without any additional manual annotation. We introduce the following three tasks based on this data. First, the video chapter generation task consists of temporally segmenting the video and generating a chapter title for each segment. To further dissect the problem, we also define two variants of this task: video chapter generation given ground-truth boundaries, which requires generating a chapter title given an annotated video segment, and video chapter grounding, which requires temporally localizing a chapter given its annotated title. We benchmark both simple baselines as well as state-of-the-art video-language models on these three tasks. We also show that pretraining on VidChapters-7M transfers well to dense video captioning tasks, largely improving the state of the art on the YouCook2 and ViTT benchmarks. Finally, our experiments reveal that downstream performance scales well with the size of the pretraining dataset.</details>

### VillanDiffusion: A Unified Backdoor Attack Framework for Diffusion Models

**Authors:** Sheng-Yen Chou, Pin-Yu Chen, Tsung-Yi Ho

**<details><summary>Abstract**</summary>

Diffusion Models (DMs) are state-of-the-art generative models that learn a reversible corruption process from iterative noise addition and denoising. They are the backbone of many generative AI applications, such as text-to-image conditional generation. However, recent studies have shown that basic unconditional DMs (e.g., DDPM and DDIM) are vulnerable to backdoor injection, a type of output manipulation attack triggered by a maliciously embedded pattern at model input. This paper presents a unified backdoor attack framework (VillanDiffusion) to expand the current scope of backdoor analysis for DMs. Our framework covers mainstream unconditional and conditional DMs (denoising-based and score-based) and various training-free samplers for holistic evaluations. Experiments show that our unified framework facilitates the backdoor analysis of different DM configurations and provides new insights into caption-based backdoor attacks on DMs.</details>

### VisAlign: Dataset for Measuring the Alignment between AI and Humans in Visual Perception

**Authors:** Jiyoung Lee, Seungho Kim, Seunghyun Won, Joonseok Lee, Marzyeh Ghassemi, James Thorne, Jaeseok Choi, O-Kil Kwon, Edward Choi

**<details><summary>Abstract**</summary>

AI alignment refers to models acting towards human-intended goals, preferences, or ethical principles. Analyzing the similarity between models and humans can be a proxy measure for ensuring AI safety. In this paper, we focus on the models' visual perception alignment with humans, further referred to as AI-human visual alignment. Specifically, we propose a new dataset for measuring AI-human visual alignment in terms of image classification. In order to evaluate AI-human visual alignment, a dataset should encompass samples with various scenarios and have gold human perception labels. Our dataset consists of three groups of samples, namely Must-Act (i.e., Must-Classify), Must-Abstain, and Uncertain, based on the quantity and clarity of visual information in an image and further divided into eight categories. All samples have a gold human perception label; even Uncertain (e.g., severely blurry) sample labels were obtained via crowd-sourcing. The validity of our dataset is verified by sampling theory, statistical theories related to survey design, and experts in the related fields. Using our dataset, we analyze the visual alignment and reliability of five popular visual perception models and seven abstention methods. Our code and data is available at https://github.com/jiyounglee-0523/VisAlign.</details>

### VisoGender:  A dataset for benchmarking gender bias in image-text pronoun resolution

**Authors:** Siobhan Mackenzie Hall, Fernanda Gonçalves Abrantes, Hanwen Zhu, Grace Sodunke, Aleksandar Shtedritski, Hannah Rose Kirk

**<details><summary>Abstract**</summary>

We introduce VisoGender, a novel dataset for benchmarking gender bias in vision-language models. We focus on occupation-related biases within a hegemonic system of binary gender, inspired by Winograd and Winogender schemas, where each image is associated with a caption containing a pronoun relationship of subjects and objects in the scene. VisoGender is balanced by gender representation in professional roles, supporting bias evaluation in two ways: i) resolution bias, where we evaluate the difference between pronoun resolution accuracies for image subjects with gender presentations perceived as masculine versus feminine by human annotators and ii) retrieval bias, where we compare ratios of professionals perceived to have masculine and feminine gender presentations retrieved for a gender-neutral search query. We benchmark several state-of-the-art vision-language models and find that they demonstrate bias in resolving binary gender in complex scenes. While the direction and magnitude of gender bias depends on the task and the model being evaluated, captioning models are generally less biased than Vision-Language Encoders.</details>

### Visual Programming for Step-by-Step Text-to-Image Generation and Evaluation

**Authors:** Jaemin Cho, Abhay Zala, Mohit Bansal

**<details><summary>Abstract**</summary>

As large language models have demonstrated impressive performance in many domains, recent works have adopted language models (LMs) as controllers of visual modules for vision-and-language tasks. While existing work focuses on equipping LMs with visual understanding, we propose two novel interpretable/explainable visual programming frameworks for text-to-image (T2I) generation and evaluation. First, we introduce VPGen, an interpretable step-by-step T2I generation framework that decomposes T2I generation into three steps: object/count generation, layout generation, and image generation. We employ an LM to handle the first two steps (object/count generation and layout generation), by finetuning it on text-layout pairs. Our step-by-step T2I generation framework provides stronger spatial control than end-to-end models, the dominant approach for this task. Furthermore, we leverage the world knowledge of pretrained LMs, overcoming the limitation of previous layout-guided T2I works that can only handle predefined object classes. We demonstrate that our VPGen has improved control in counts/spatial relations/scales of objects than state-of-the-art T2I generation models. Second, we introduce VPEval, an interpretable and explainable evaluation framework for T2I generation based on visual programming. Unlike previous T2I evaluations with a single scoring model that is accurate in some skills but unreliable in others, VPEval produces evaluation programs that invoke a set of visual modules that are experts in different skills, and also provides visual+textual explanations of the evaluation results. Our analysis shows that VPEval provides a more human-correlated evaluation for skill-specific and open-ended prompts than widely used single model-based evaluation. We hope that our work encourages future progress on interpretable/explainable generation and evaluation for T2I models.</details>

### WBCAtt: A White Blood Cell Dataset Annotated with Detailed Morphological Attributes

**Authors:** Satoshi Tsutsui, Winnie Pang, Bihan Wen

**<details><summary>Abstract**</summary>

The examination of blood samples at a microscopic level plays a fundamental role in clinical diagnostics. For instance, an in-depth study of White Blood Cells (WBCs), a crucial component of our blood, is essential for diagnosing blood-related diseases such as leukemia and anemia. While multiple datasets containing WBC images have been proposed, they mostly focus on cell categorization, often lacking the necessary morphological details to explain such categorizations, despite the importance of explainable artificial intelligence (XAI) in medical domains. This paper seeks to address this limitation by introducing comprehensive annotations for WBC images. Through collaboration with pathologists, a thorough literature review, and manual inspection of microscopic images, we have identified 11 morphological attributes associated with the cell and its components (nucleus, cytoplasm, and granules). We then annotated ten thousand WBC images with these attributes, resulting in 113k labels (11 attributes x 10.3k images). Annotating at this level of detail and scale is unprecedented, offering unique value to AI in pathology. Moreover, we conduct experiments to predict these attributes from cell images, and also demonstrate specific applications that can benefit from our detailed annotations. Overall, our dataset paves the way for interpreting WBC recognition models, further advancing XAI in the fields of pathology and hematology.</details>

### [Spotlight] WITRAN: Water-wave Information Transmission and Recurrent Acceleration Network for Long-range Time Series Forecasting

**Authors:** Yuxin Jia, Youfang Lin, Xinyan Hao, Yan Lin, Shengnan Guo, Huaiyu Wan

**<details><summary>Abstract**</summary>

Capturing semantic information is crucial for accurate long-range time series forecasting, which involves modeling global and local correlations, as well as discovering long- and short-term repetitive patterns. Previous works have partially addressed these issues separately, but have not been able to address all of them simultaneously. Meanwhile, their time and memory complexities are still not sufficiently low for long-range forecasting. To address the challenge of capturing different types of semantic information, we propose a novel Water-wave Information Transmission (WIT) framework. This framework captures both long- and short-term repetitive patterns through bi-granular information transmission. It also models global and local correlations by recursively fusing and selecting information using Horizontal Vertical Gated Selective Unit (HVGSU). In addition, to improve the computing efficiency, we propose a generic Recurrent Acceleration Network (RAN) which reduces the time complexity to $\mathcal{O}(\sqrt{L})$ while maintaining the memory complexity at $\mathcal{O}(L)$. Our proposed method, called Water-wave Information Transmission and Recurrent Acceleration Network (WITRAN), outperforms the state-of-the-art methods by 5.80% and 14.28% on long-range and ultra-long-range time series forecasting tasks respectively, as demonstrated by experiments on four benchmark datasets. The code is available at: https://github.com/Water2sea/WITRAN.</details>

### Weakly-Supervised Audio-Visual Segmentation

**Authors:** Shentong Mo, Bhiksha Raj

**<details><summary>Abstract**</summary>

Audio-visual segmentation is a challenging task that aims to predict pixel-level masks for sound sources in a video.  Previous work applied a comprehensive manually designed architecture with countless pixel-wise accurate masks as supervision.  However, these pixel-level masks are expensive and not available in all cases.  In this work, we aim to simplify the supervision as the instance-level annotation, $\textit{i.e.}$, weakly-supervised audio-visual segmentation.  We present a novel Weakly-Supervised Audio-Visual Segmentation framework, namely WS-AVS, that can learn multi-scale audio-visual alignment with multi-scale multiple-instance contrastive learning for audio-visual segmentation.  Extensive experiments on AVSBench demonstrate the effectiveness of our WS-AVS in the weakly-supervised audio-visual segmentation of single-source and multi-source scenarios.</details>

### What Can We Learn from Unlearnable Datasets?

**Authors:** Pedro Sandoval-Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein

**<details><summary>Abstract**</summary>

In an era of widespread web scraping, unlearnable dataset methods have the potential to protect data privacy by preventing deep neural networks from generalizing. But in addition to a number of practical limitations that make their use unlikely, we make a number of findings that call into question their ability to safeguard data. First, it is widely believed that neural networks trained on unlearnable datasets only learn shortcuts, simpler rules that are not useful for generalization. In contrast, we find that networks actually can learn useful features that can be reweighed for high test performance, suggesting that image protection is not assured. Unlearnable datasets are also believed to induce learning shortcuts through linear separability of added perturbations. We provide a counterexample, demonstrating that linear separability of perturbations is not a necessary condition. To emphasize why linearly separable perturbations should not be relied upon, we propose an orthogonal projection attack which allows learning from unlearnable datasets published in ICML 2021 and ICLR 2023. Our proposed attack is significantly less complex than recently proposed techniques.</details>

### What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks

**Authors:** Taicheng Guo, kehan Guo, Bozhao Nan, Zhenwen Liang, Zhichun Guo, Nitesh Chawla, Olaf Wiest, Xiangliang Zhang

**<details><summary>Abstract**</summary>

Large Language Models (LLMs) with strong abilities in natural language processing tasks have emerged and have been applied in various kinds of areas such as science, finance and software engineering. However, the capability of LLMs to advance the field of chemistry remains unclear. In this paper, rather than pursuing state-of-the-art performance, we aim to evaluate capabilities of LLMs in a wide range of tasks across the chemistry domain. We identify three key chemistry-related capabilities including understanding, reasoning and explaining to explore in LLMs and establish a benchmark containing eight chemistry tasks. Our analysis draws on widely recognized datasets facilitating a broad exploration of the capacities of LLMs within the context of practical chemistry. Five LLMs (GPT-4,GPT-3.5, Davinci-003, Llama and Galactica) are evaluated for each chemistry task in zero-shot and few-shot in-context learning settings with carefully selected demonstration examples and specially crafted prompts. Our investigation found that GPT-4 outperformed other models and LLMs exhibit different competitive levels in eight chemistry tasks. In addition to the key findings from the comprehensive benchmark analysis, our work provides insights into the limitation of current LLMs and the impact of in-context learning settings on LLMs’ performance across various chemistry tasks. The code and datasets used in this study are available at https://github.com/ChemFoundationModels/ChemLLMBench.</details>

### What can a Single Attention Layer Learn? A Study Through the Random Features Lens

**Authors:** Hengyu Fu, Tianyu Guo, Yu Bai, Song Mei

**<details><summary>Abstract**</summary>

Attention layers---which map a sequence of inputs to a sequence of outputs---are core building blocks of the Transformer architecture which has achieved significant breakthroughs in modern artificial intelligence. This paper presents a rigorous theoretical study on the learning and generalization of a single multi-head attention layer, with a sequence of key vectors and a separate query vector as input. We consider the random feature setting where the attention layer has a large number of heads, with randomly sampled frozen query and key matrices, and trainable value matrices. We show that such a random-feature attention layer can express a broad class of target functions that are permutation invariant to the key vectors.  We further provide quantitative excess risk bounds for learning these target functions from finite samples, using random feature attention with finitely many heads.Our results feature several implications unique to the attention structure compared with existing random features theory for neural networks, such as (1) Advantages in the sample complexity over standard two-layer random-feature networks;  (2) Concrete and natural classes of functions that can be learned efficiently by a random-feature attention layer; and (3) The effect of the sampling distribution of the query-key weight matrix (the product of the query and key matrix), where Gaussian random weights with a non-zero mean result in better sample complexities over the zero-mean counterpart for learning certain natural target functions. Experiments on simulated data corroborate our theoretical findings and further illustrate the interplay between the sample size and the complexity of the target function.</details>

### What is Flagged in Uncertainty Quantification?  Latent Density Models for Uncertainty Categorization

**Authors:** Hao Sun, Boris van Breugel, Jonathan Crabbé, Nabeel Seedat, Mihaela van der Schaar

**<details><summary>Abstract**</summary>

Uncertainty quantification (UQ) is essential for creating trustworthy machine learning models. Recent years have seen a steep rise in UQ methods that can flag suspicious examples, however, it is often unclear what exactly these methods identify. In this work, we propose a framework for categorizing uncertain examples flagged by UQ methods. We introduce the confusion density matrix---a kernel-based approximation of the misclassification density---and use this to categorize suspicious examples identified by a given uncertainty method into three classes: out-of-distribution (OOD) examples, boundary (Bnd) examples, and examples in regions of high in-distribution misclassification (IDM). Through extensive experiments, we show that our framework provides a new and distinct perspective for assessing differences between uncertainty quantification methods, thereby forming a valuable assessment benchmark.</details>

### When Do Neural Nets Outperform Boosted Trees on Tabular Data?

**Authors:** Duncan McElfresh, Sujay Khandagale, Jonathan Valverde, Vishak Prasad C, Ganesh Ramakrishnan, Micah Goldblum, Colin White

**<details><summary>Abstract**</summary>

Tabular data is one of the most commonly used types of data in machine learning. Despite recent advances in neural nets (NNs) for tabular data, there is still an active discussion on whether or not NNs generally outperform gradient-boosted decision trees (GBDTs) on tabular data, with several recent works arguing either that GBDTs consistently outperform NNs on tabular data, or vice versa. In this work, we take a step back and question the importance of this debate. To this end, we conduct the largest tabular data analysis to date, comparing 19 algorithms across 176 datasets, and we find that the 'NN vs. GBDT' debate is overemphasized: for a surprisingly high number of datasets, either the performance difference between GBDTs and NNs is negligible, or light hyperparameter tuning on a GBDT is more important than choosing between NNs and GBDTs. Next, we analyze dozens of metafeatures to determine what \emph{properties} of a dataset make NNs or GBDTs better-suited to perform well. For example, we find that GBDTs are much better than NNs at handling skewed or heavy-tailed feature distributions and other forms of dataset irregularities. Our insights act as a guide for practitioners to determine which techniques may work best on their dataset. Finally, with the goal of accelerating tabular data research, we release the TabZilla Benchmark Suite: a collection of the 36 'hardest' of the datasets we study. Our benchmark suite, codebase, and all raw results are available at https://github.com/naszilla/tabzilla.</details>

### [Oral] When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment

**Authors:** Tianwei Ni, Michel Ma, Benjamin Eysenbach, Pierre-Luc Bacon

**Oral Presentation:** Th, Dec 14, 13:20 -- Oral 6B

**<details><summary>Abstract**</summary>

Reinforcement learning (RL) algorithms face two distinct challenges: learning effective representations of past and present observations, and determining how actions influence future returns. Both challenges involve modeling long-term dependencies. The Transformer architecture has been very successful to solve problems that involve long-term dependencies, including in the RL domain. However, the underlying reason for the strong performance of Transformer-based RL methods remains unclear: is it because they learn effective memory, or because they perform effective credit assignment? After introducing formal definitions of memory length and credit assignment length, we design simple configurable tasks to measure these distinct quantities. Our empirical results reveal that Transformers can enhance the memory capability of RL algorithms, scaling up to tasks that require memorizing observations $1500$ steps ago. However, Transformers do not improve long-term credit assignment. In summary, our results provide an explanation for the success of Transformers in RL, while also highlighting an important area for future research and benchmark design. Our code is open-sourced at https://github.com/twni2016/Memory-RL.</details>

### WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction

**Authors:** Sebastian Gerard, Yu Zhao, Josephine Sullivan

**<details><summary>Abstract**</summary>

We present a multi-temporal, multi-modal remote-sensing dataset for predicting how active wildfires will spread at a resolution of 24 hours. The dataset consists of 13607 images across 607 fire events in the United States from January 2018 to October 2021. For each fire event, the dataset contains a full time series of daily observations, containing detected active fires and variables related to fuel, topography and weather conditions. The dataset is challenging due to: a) its inputs being multi-temporal, b) the high number of 23 multi-modal input channels, c) highly imbalanced labels and d) noisy labels, due to smoke, clouds, and inaccuracies in the active fire detection. The underlying complexity of the physical processes adds to these challenges. Compared to existing public datasets in this area, WildfireSpreadTS allows for multi-temporal modeling of spreading wildfires, due to its time series structure. Furthermore, we provide additional input modalities and a high spatial resolution of 375m for the active fire maps. We publish this dataset to encourage further research on this important task with multi-temporal, noise-resistant or generative methods, uncertainty estimation or advanced optimization techniques that deal with the high-dimensional input space.</details>

### Winner Takes It All: Training Performant RL Populations for Combinatorial Optimization

**Authors:** Nathan Grinsztajn, Daniel Furelos-Blanco, Shikha Surana, Clément Bonnet, Tom Barrett

**<details><summary>Abstract**</summary>

Applying reinforcement learning (RL) to combinatorial optimization problems is attractive as it removes the need for expert knowledge or pre-solved instances. However, it is unrealistic to expect an agent to solve these (often NP-)hard problems in a single shot at inference due to their inherent complexity. Thus, leading approaches often implement additional search strategies, from stochastic sampling and beam-search to explicit fine-tuning. In this paper, we argue for the benefits of learning a population of complementary policies, which can be simultaneously rolled out at inference. To this end, we introduce Poppy, a simple training procedure for populations. Instead of relying on a predefined or hand-crafted notion of diversity, Poppy induces an unsupervised specialization targeted solely at maximizing the performance of the population. We show that Poppy produces a set of complementary policies, and obtains state-of-the-art RL results on three popular NP-hard problems: traveling salesman, capacitated vehicle routing, and job-shop scheduling.</details>

### [Spotlight] Would I have gotten that reward? Long-term credit assignment by counterfactual contribution analysis

**Authors:** Alexander Meulemans, Simon Schug, Seijin Kobayashi, nathaniel daw, Gregory Wayne

**<details><summary>Abstract**</summary>

To make reinforcement learning more sample efficient, we need better credit assignment methods that measure an action’s influence on future rewards. Building upon Hindsight Credit Assignment (HCA), we introduce Counterfactual Contribution Analysis (COCOA), a new family of model-based credit assignment algorithms. Our algorithms achieve precise credit assignment by measuring the contribution of actions upon obtaining subsequent rewards, by quantifying a counterfactual query: ‘Would the agent still have reached this reward if it had taken another action?’. We show that measuring contributions w.r.t. rewarding _states_, as is done in HCA, results in spurious estimates of contributions, causing HCA to degrade towards the high-variance REINFORCE estimator in many relevant environments. Instead, we measure contributions w.r.t. rewards or learned representations of the rewarding objects, resulting in gradient estimates with lower variance. We run experiments on a suite of problems specifically designed to evaluate long-term credit assignment capabilities. By using dynamic programming, we measure ground-truth policy gradients and show that the improved performance of our new model-based credit assignment methods is due to lower bias and variance compared to HCA and common baselines. Our results demonstrate how modeling action contributions towards rewarding outcomes can be leveraged for credit assignment, opening a new path towards sample-efficient reinforcement learning.</details>

### XES3G5M: A Knowledge Tracing Benchmark Dataset with Auxiliary Information

**Authors:** Zitao Liu, Qiongqiong Liu, Teng Guo, Jiahao Chen, Shuyan Huang, Xiangyu Zhao, Jiliang Tang, Weiqi Luo, Jian Weng

**<details><summary>Abstract**</summary>

Knowledge tracing (KT) is a task that predicts students' future performance based on their historical learning interactions. With the rapid development of deep learning techniques, existing KT approaches follow a data-driven paradigm that uses massive problem-solving records to model students' learning processes. However, although the educational contexts contain various factors that may have an influence on student learning outcomes, existing public KT datasets mainly consist of anonymized ID-like features, which may hinder the research advances towards this field. Therefore, in this work, we present, \emph{XES3G5M}, a large-scale dataset with rich auxiliary information about questions and their associated knowledge components (KCs)\footnote{\label{ft:kc}A KC is a generalization of everyday terms like concept, principle, fact, or skill.}. The XES3G5M dataset is collected from a real-world online math learning platform, which contains 7,652 questions, and 865 KCs with 5,549,635 interactions from 18,066 students. To the best of our knowledge, the XES3G5M dataset not only has the largest number of KCs in math domain but contains the richest contextual information including tree structured KC relations, question types, textual contents and analysis and student response timestamps. Furthermore, we build a comprehensive benchmark on 19 state-of-the-art deep learning based knowledge tracing (DLKT) models. Extensive experiments demonstrate the effectiveness of leveraging the auxiliary information in our XES3G5M with DLKT models. We hope the proposed dataset can effectively facilitate the KT research work.</details>

### YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus

**Authors:** Dave Uthus, Garrett Tanzer, Manfred Georg

**<details><summary>Abstract**</summary>

Machine learning for sign languages is bottlenecked by data. In this paper, we present YouTube-ASL, a large-scale, open-domain corpus of American Sign Language (ASL) videos and accompanying English captions drawn from YouTube. With ~1000 hours of videos and >2500 unique signers, YouTube-ASL is ~3x as large and has ~10x as many unique signers as the largest prior ASL dataset. We train baseline models for ASL to English translation on YouTube-ASL and evaluate them on How2Sign, where we achieve a new fine-tuned state of the art of 12.397 BLEU and, for the first time, nontrivial zero-shot results.</details>

### YouTubePD: A Multimodal Benchmark for Parkinson’s Disease Analysis

**Authors:** Andy Zhou, Samuel Li, Pranav Sriram, Xiang Li, Jiahua Dong, Ansh Sharma, Yuanyi Zhong, Shirui Luo, Volodymyr Kindratenko, George Heintz, Christopher Zallek, Yu-Xiong Wang

**<details><summary>Abstract**</summary>

The healthcare and AI communities have witnessed a growing interest in the development of AI-assisted systems for automated diagnosis of Parkinson's Disease (PD), one of the most prevalent neurodegenerative disorders. However, the progress in this area has been significantly impeded by the absence of a unified, publicly available benchmark, which prevents comprehensive evaluation of existing PD analysis methods and the development of advanced models. This work overcomes these challenges by introducing YouTubePD -- the first publicly available multimodal benchmark designed for PD analysis. We crowd-source existing videos featured with PD from YouTube, exploit multimodal information including in-the-wild videos, audio data, and facial landmarks across 200+ subject videos, and provide dense and diverse annotations from clinical expert. Based on our benchmark, we propose three challenging and complementary tasks encompassing both discriminative and generative tasks, along with a comprehensive set of corresponding baselines. Experimental evaluation showcases the potential of modern deep learning and computer vision techniques, in particular the generalizability of the models developed on YouTubePD to real-world clinical settings, while revealing their limitations. We hope our work paves the way for future research in this direction.</details>

### [Re] $\mathcal{G}$-Mixup: Graph Data Augmentation for Graph Classification

**Authors:** Ermin Omeragic, Vuk Đuranović

**<details><summary>Abstract**</summary>

Scope of ReproducibilityThis paper presents a novel augmentation method that can be used for graph classification tasks: $\mathcal{G}$-Mixup. Our goal is to reproduce eight claims that the authors make in their paper. The first two claims relate to the properties of graphons estimated from graphs, which are the main components of the method. Claims three to eight relate to the superior performance of the method compared to other augmentation strategies.MethodologyTo reproduce the results, we use the open-source implementation of the method provided by the authors, with a few modifications. We write from scratch all the experiments and pipelines needed to defend the claims of the paper. Additionally, we implement three out of four baseline augmentation methods that are compared to the novel method. For one part of the experiments, we use a local computer and run the experiments on a CPU, with a total of 31.7 CPU hours, while for other more demanding experiments, we use a GPU-accelerated machine for a total of 157.3 GPU hours.ResultsDue to many missing implementation details, we were not able to reproduce all of the original results. Some claims can be supported by our results, but most results are very vague. Even though the new method outperforms the baselines in certain scenarios, we find that the superiority of the method is not as strong as presented in the original paper.What was easyThe novel augmentation method and its theoretical justification are presented intuitively in the paper, and it was easy to grasp the main ideas of the paper.What was difficultWhile the code with the method implementation was given, the reproduction of all the results required much more code and details than what was provided in the paper. This means that we had to make a lot of educated guesses about the experimental settings, choices of hyperparameters, and details about the models used. Communication with original authorsWe contacted the authors on two occasions. The first time was very early in the reproduction process, when we asked about the details of the graph estimation methods that they used and which weren't explained in the paper, to which they responded swiftly. On the second occasion, we asked about the experimental settings and hyperparameter details, but we did not receive a reply.</details>

### [Re] Bandit Theory and Thompson Sampling-guided Directed Evolution for Sequence Optimization

**Authors:** Luka Žontar

**<details><summary>Abstract**</summary>

The paper presents a novel DE approach using Thompson Sampling and Bandit theory, TS-DE. Our reproducibility study aims to confirm the 5 main claims of the original paper, including sublinear Bayesian regret, improved performance compared to basic DE, robustness to mutation rate changes, initial diversification, and the concentration of the population to the optimal value in later iterations, and iterative distribution shift towards optimal population fitness. Finally, we provide a reproducible environment to support the main claims of the original paper along with the source code of the proposed approach and all experiments, comprehensive documentation, and unit tests.No code was available beforehand for this article, thus we re-implemented the proposed approach by meticulously following the comprehensive explanations of the process in the original article. The experiments were run on a personal computer.We managed to reproduce all the experiments supporting the main claims of the original article. Additionally, we add uncertainty quantification to the results as we believe this is a crucial part to confirm any of the claims. Finally, we present the exploration-exploitation trade-off experiment in a more robust manner leveraging the nucleotide diversity metric to gain additional insight into how the proposed algorithm works.With comprehensive explanations in the original article, it was relatively easy to rewrite the main concepts from pseudo-code to Python code. Experiments were clearly explained with all the necessary hyperparameters.Since no source code was available, every detail missing from the original article resulted in additional research and trial and error experimentation. To conclude, we recommend several improvements to the authors of the studied paper that could additionally improve the quality of their outstanding contribution. Some of the recommendations include better explanations of how the optimal solution is calculated, on which population the PCA is fitted, how to use $\theta^*$ and $\Tilde{\theta}$, and lastly improved documentation on how the basic DE was implemented. We contacted the original authors on multiple occasions during the development of our reproducibility study but got no response.</details>

### [Re] CrossWalk: Fairness-enhanced Node Representation Learning

**Authors:** Luca Pantea, Andrei-Eusebiu Blahovici

**<details><summary>Abstract**</summary>

Scope of ReproducibilityThis work aims to reproduce the findings of the paper “CrossWalk: Fairness-enhanced Node Representation Learning” by investigating the two main claims made by the authors about CrossWalk, which suggest that (i) CrossWalk enhances fairness in three graph algorithms, while only suffering from small decreases in performance, and that (ii) CrossWalk preserves the necessary structural properties of the graph while reducing disparity.MethodologyThe authors made the CrossWalk repository available, which contained most of the datasets used for their experimentation, and the scripts needed to run the experiments. However, the codebase lacked documentation and was missing logic for running all experiments and visualizing the results. We, therefore, re-implement their code from scratch and deploy it as a python package which can be run to obtain all the showcased results. ResultsOur work suggests that the first claim of the paper, which states that Crosswalk minimizes disparity and thus enhances fairness is partially reproducible, and only for the tasks of Node classification and Influence maximization as the parameters specified in the paper do not always yield similar results. Then, the second claim of the paper which states that Crosswalk attains the necessary structural properties of the graph is fully reproducible through our experiments.What was easyThe original paper contained the necessary information about hyperparameters, which coupled with the publicly available repository made it straightforward to refactor the code and understand the idea of the proposed method. What was difficultThe difficulty stems from the lack of structure and documentation in the provided code which made the original experiments hard to reproduce. Furthermore, there were missing files in the provided datasets. Also, some experiments were not reproducible at all through the provided code. One more important aspect is that the experiments are CPU intensive which made the reproducibility even harder.Communication with original authorsAlbeit rather late, the authors provided meaningful feedback on our questions about implementation details and initial results.</details>

### [Re] End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking

**Authors:** Sean McLeish, Long Tran-Thanh

**<details><summary>Abstract**</summary>

Scope of Reproducibility:In this report, we aim to validate the claims of Bansal et al. These are that the recurrent architecture presented, with skip connections and a progressive loss function, prevent the original problem being forgotten or corrupted during processing allowing for the recurrent module to be applied indefinitely and that this architecture avoids the overthinking trap. We use both code released by the authors and newly developed to recreate many results presented in the paper. Additionally, we present analysis of the newly introduced alpha hyperparameter and investigate interesting perturbation behaviour of prefix sums models. Further, we conduct a hyperparameter search and provide an analysis of the Asymptotic Alignment scores of the models presented.Methodology:We use the PyTorch code released by the authors to replicate accuracy experiments. We then, independently, develop our own PyTorchFI to replicate perturbation experiments presented by Bansal et al. Overall, providing a replication of all results shown in the main body of the paper. We then extend these results, providing an analysis of the alpha hyperparameter, analysis of perturbation recovery, Asymptotic Alignment scores and a hyperparameter search. We used both a Nvidia RTX 2080Ti GPU and sets of three NVIDIA Quadro RTX6000 GPUs, taking a total of 982.2 GPU hours to create all results presented in the main body of this report.Results:We verify the authors' claims by replicating the experiments presented by Bansal et al. All of our experiments show identical results to the ones presented in the paper, apart from perturbation testing for which we provide an additional in depth analysis. We also provide an analysis of the new alpha hyperparameter and a hyperparameter search.What was easy:The code provided by Bansal et al gave clear instructions on how to use it, along with pretrained models being available for all problems.What was difficult:Chess models required a considerable amount of time to train, putting a drain on resources. Also, code for reproducing perturbation results was not available so this had to developed from scratch.Communication with original authors:We had good communication with the original authors, both emailing and meeting online.</details>

### [Re] Exploring the Role of Grammar and Word Choice in Bias Toward African American English (AAE) in Hate Speech Classification

**Authors:** Priyanka Bose, Chandra shekhar Pandey, Fraida Fund

**<details><summary>Abstract**</summary>

Reproducibility SummaryScope of Reproducibility — We aim to reproduce a result from the paper “Exploring the Role of Grammar and Word Choice in Bias Toward African American English (AAE) in Hate Speech Classification” [1]. Our study is restricted specifically to the claim that the use of swear words impacts hate speech classification of AAE text. We were able to broadly validate the claim of the paper, however, the magnitude of the effect was dependent on the word replacement strategy, which was somewhat ambiguous in the original paper.Methodology — The authors did not publish source code. Therefore, we reproduce the experiments by following the methodology described in the paper. We train BERT models from TensorFlow Hub [2] to classify hate speech using the DWMW17[3] and FDCL18[4]Twitter datasets. Then, we compile a dictionary of swear words and replacement words with comparable meaning, and we use this to create “censored” versions of samples in Blodgett et al.’s[5] AAE Twitter dataset. Using the BERT models, we evaluate the hate speech classification of the original data and the censored data. Our experiments are conducted on an open‐access research testbed, Chameleon [6], and we make available both our code and instructions for reproducing the result on the shared facility.Results — Our results are consistent with the claim that the censored text (without swear words) is less often classified as hate speech, offensive, or abusive than the same text with swear words. However, we find the classification is very sensitive to the word replacement dictionary being used.What was easy — The authors used three well known datasets which were easy to obtain. They also used well‐known widely available models, BERT and Word2Vec.What was difficult — Some of the details of model training were not fully specified. Also, we were not able to exactly re‐create a comparable dictionary of swear words and their replacement terms of similar meanings, following their methodology.Communication with original authors — We reached out to the authors, but were not able to communicate with them before the submission of this report.</details>

### [Re] FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles

**Authors:** Kyosuke Morita

**<details><summary>Abstract**</summary>

Scope of ReproducibilityThis study aims to reproduce the results of the paper 'FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles' by Lucic et al.The main claims of the original paper are that FOCUS is able to (i) generate counterfactual explanations for all the instances in a dataset; and (ii) find counterfactual explanations that are closer to the original input for tree-based algorithms than existing methods.MethodologyThis study replicates the original experiments using the code, data, and models provided by the authors. Additionally, this study re-implements code and retrains the models to evaluate the robustness and generality of FOCUS.All the experiments were conducted on a personal laptop with a quad-core CPU with 8GB of RAM and it approximately took 33 hours in total.ResultsThis study was able to replicate the results of the original paper in terms of finding counterfactual explanations for all instances in datasets. Additional experiments were conducted to validate the robustness and generality of the conclusion. While there were slight deviations in terms of generating smaller mean distances, half of the models still outperformed the results of the existing method.What was easyThe implementation of the original paper is publicly available on GitHub. The repository contains the models and data used in the original experiments. Also, the authors provided a technical appendix, which includes all the hyperparameters that were used for the experiments for reproduction upon request.What was difficultAlthough the implementation code was available, it employs outdated packages and the code structure is complex. Also, the comments in the functions and the documentation of the code are sparse or nonexistent, which made it difficult to follow the code.Communication with original authorsI reached out to the authors to obtain the hyperparameters used in the experiments. The authors responded promptly with a detailed technical appendix of the original paper.</details>

### [Re] Fairness Guarantees under Demographic Shift

**Authors:** Valentin Buchner, Philip Schutte, Yassin Ben Allal, Hamed Ahadi

**<details><summary>Abstract**</summary>

Scope of Reproducibility: The original authors'  main contribution is the family of Shifty algorithms, which can guarantee that certain fairness constraints will hold with high confidence even after a demographic shift in the deployment population occurs. They claim that Shifty provides these high-confidence fairness guarantees without a loss in model performance, given enough training data.Methodology: The code provided by the original paper was used, and only some small adjustments needed to be made in order to reproduce the experiments. All model specifications and hyperparameters from the original implementation were used. Extending beyond reproducing the original paper, we investigated the sensibility of Shifty to the size of the bounding intervals limiting the possible demographic shift, and ran shifty with an additional optimization method.Results: Our results approached the results reported in the original paper. They supported the claim that \textit{Shifty} reliably guarantees fairness under demographic shift, but could not verify that Shifty performs at no loss of accuracy. What was easy: The theoretical framework laid out in the original paper was well explained and supported by additional formulas and proofs in the appendix. Further, the authors provided clear instructions on how to run the experiments and provided necessary hyperparameters.What was difficult: While an open-source implementation of Shifty was provided and was debugged with relatively low time investment, the code did not contain extensive documentation and was complex to understand. It was therefore difficult to verify that each part of the code functions as expected and to expand upon the existing experiments. Further, certain hyperparameter and model specifications deviated between the provided code and the original paper, which made it challenging to know which specifications to apply when reproducing.Communication with original authors: The first author of the original paper was contacted, but we have yet to receive a reply.</details>

### [Re] Numerical influence of ReLU'(0) on backpropagation

**Authors:** Tommaso Martorella, Daniel Garcia

**<details><summary>Abstract**</summary>

Neural networks have become very common in machine learning, and new problems and trends arise as the trade-off between theory, computational tools and real-world problems become more narrow and complex. We decided to retake the influence of the ReLU'(0) on the backpropagation as it has become more common to use lower floating point precisions in the GPUs so that more tasks can run in parallel and make training and inference more efficient. As opposed to what theory suggests, the original authors shown that when using 16- and 32-bit precision, the value of ReLU'(0) may influence the result. In this work we extended some experiments to see how the training and test loss are affected in simple and more complex models.</details>

### [Re] On Explainability of Graph Neural Networks via Subgraph Explorations

**Authors:** Yannik Mahlau, Lukas Berg, Leonie Kayser

**<details><summary>Abstract**</summary>

Yuan et al. claim their proposed method SubgraphX achieves (i) higher fidelity in explaining models for graph- and node classification tasks compared to other explanation techniques, namely GNNExplainer. Additionally, (ii) the computational effort of SubgraphX is at a 'reasonable level', which is not further specified by the original authors. We define this as at most ten times slower than GNNExplainer. We reimplemented the proposed algorithm in PyTorch. Then, we replicated the experiments performed by the authors on a smaller scale due to resource constraints. Additionally, we checked the performance on a new dataset and investigated the influence of hyperparameters. Lastly, we improved SubgraphX using greedy initialization and utilizing fidelity as a score function. We were able to reproduce the main claims on the MUTAG dataset, where SubgraphX has a better performance than GNNExplainer. Furthermore, SubgraphX has a reasonable runtime of about seven times longer than GNNExplainer. We successfully employed SubgraphX on the Karate Club dataset, where it outperforms GNNExplainer as well. The hyperparameter study revealed that the number of Monte-Carlo Tree search iterations and Monte-Carlo sampling steps are the most important hyperparameters and directly trade performance for runtime. Lastly, we show that our proposed improvements to SubgraphX significantly enhance fidelity and runtime. The authors' description of the algorithm was clear and concise. The original implementation is available in the DIG-library as a reference to our implementation. The authors performed extensive experiments, which we could not replicate in their full scale due to resource constraints. However, we were able to achieve similar results on a subset of the datasets used. Another issue was that despite the original code of the authors and datasets being publicly available, there were many compatibility issues. The original authors briefly reviewed our work and agreed with the findings.</details>

### [Re] On the Reproducibility of CartoonX

**Authors:** Robin Sasse, Aniek Eijpe, Jona Ruthardt, Elias Dubbeldam

**<details><summary>Abstract**</summary>

Scope of Reproducibility — CartoonX [1] is a novel explanation method for image classifiers. In this reproducibility study, we examine the claims of the original authors of CartoonX that it: (i) extracts relevant piece‐wise smooth parts of the image, resulting in explanations which are more straightforward to interpret for humans; (ii) achieves lower distortion in the model output, using fewer coefficients than other state-of‐the‐art methods; (iii) is model‐agnostic. Finally, we examine how to reduce the runtime.Methodology — The original authors’ open‐sourced implementation has been used to examine (i). We implemented the code to examine (ii), as there was no public code available for this. We tested claim (iii) by performing the same experiments with a Vision Transformer instead of a CNN. To reduce the runtime, we extended the existing implementation with multiple enhanced initialization techniques. All experiments took approximately 38.4 hours on a single NVIDIA Titan RTX.Results — Our results support the claims made by the original authors. (i) We observe that CartoonX produces piece‐wise smooth explanations. Most of the explanations give valuable insights. (ii) Most experiments, that show how CartoonX achieves lower distortion outputs compared to other methods, have been reproduced. In the cases where exact reproducibility has not been achieved, claim (ii) of the author still holds. (iii) The model‐agnosticism claim still holds as the overall quality of the ViT‐based explanations almost matches that of the CNN‐based explanations. Finally, simple heuristical initializations did not improve the runtime.What was easy — The mathematical background and intuition of CartoonX were clearly explained by the original authors. Moreover, the original author’s code was well structured and documented, which made it straightforward to run and extend.What was difficult — Some hyperparameter settings and implementation details needed to reproduce the experiments were not clear or transparent from the original paper or code. This made it difficult to implement and reproduce these experiments.Communication with original authors — We have been in brief communication with the original authors. They were able to address most of our points, providing us with some additional clarifications about the exact implementation and hyperparameter settings.</details>

### [Re] On the Reproducibility of “FairCal: Fairness Calibration for Face Verification”

**Authors:** Marga Don, Satchit Chatterji, Milena Kapralova, Ryan Amaudruz

**<details><summary>Abstract**</summary>

Scope of Reproducibility — This paper aims to reproduce the study FairCal: Fairness Calibration for Face Verification by Salvador et al., focused on verifying three main claims: FairCal (introduced by the authors) achieves state‐of‐the‐art (i) global accuracy, (ii) fairness-calibrated probabilities and (iii) equality in false positive rates across sensitive attributes (i.e. predictive equality). The sensitive attribute taken into account is ethnicity.Methodology — Salvador et al. provide partial code via a GitHub repository. Additional code to generate image embeddings from three pretrained neural network models were based on existing repositories. All code was refactored to fit our needs, keeping extendability and readability in mind. Two datasets were used, namely, Balanced Faces in the Wild (BFW) and Racial Faces in the Wild (RFW). Additional experiments using Gaussian mixture models instead of K‐means clustering for FairCal validate the use of unsupervised clus‐ tering methods. The code was run on an AMD Ryzen 7 2700X CPU and NVIDIA GeForce GTX1080Ti GPU with a total runtime of around 3 hours for all experiments.Results — In most cases, we were able to reproduce results from the original paper to within 1 standard deviation, and observe similar trends. However, due to missing information about image pre‐processing, we were unable to reproduce the results exactly.What was easy — The original paper is clear and understandable. Furthermore, the authors provided a mostly working version of the code. Though the datasets are not freely available to the public, their authors supplied these to us swiftly after contacting them.What was difficult — While most of the code worked with slight changes, it was assumed there were files containing image embeddings available for both datasets, which the authors neither provided nor gave details about. We therefore pre‐processed and generated embeddings independent of the authors, which makes it more difficult to judge the overall reproducibility of their method. Additionally, we encountered difficulties while improving the efficiency and extendability of the code.Communication with original authors — We emailed the first author of the paper twice. First at the beginning of our undertaking, they were enthusiastic about our attempt, and clarified a few initial doubts about their implementation, the embeddings, and missing files. As per the writing of this paper, they have not responded to the second email.</details>

### [Re] Pure Noise to the Rescue of Insufficient Data

**Authors:** Ryan Lee, Seungmin Lee

**<details><summary>Abstract**</summary>

Scope of Reproducibility — We examine the main claims of the original paper [1], whichstates that in an image classification task with imbalanced training data, (i) using purenoise to augment minority‐class images encourages generalization by improving minority‐class accuracy. This method is paired with (ii) a new batch normalization layer thatnormalizes noise images using affine parameters learned from natural images, whichimproves the model’s performance. Moreover, (iii) this improvement is robust to vary‐ing levels of data augmentation. Finally, the authors propose that (iv) adding pure noiseimages can improve classification even on balanced training data.Methodology — We implemented the training pipeline from the description of the paperusing PyTorch and integrated authors’ code snippets for sampling pure noise imagesand batch normalizing noise and natural images separately. All of our experiments wererun on a machine from a cloud computing service with one NVIDIA RTX A5000 GraphicsCard and had a total computational time of approximately 432 GPU hours.Results — We reproduced the main claims that (i) oversampling with pure noise improvesgeneralization by improving the minority‐class accuracy, (ii) the proposed batch nor‐malization (BN) method outperforms baselines, (iii) and this improvement is robustacross data augmentations. Our results also support that (iv) adding pure noise imagescan improve classification on balanced training data. However, additional experimentssuggest that the performance improvement from OPeN may be more orthogonal to theimprovement caused by a bigger network or more complex data augmentation.What was easy — The code snippet in the original paper was thoroughly documented andwas easy to use. The authors also clearly documented most of the hyperparameters thatwere used in the main experiments.What was difficult — The repo linked in the original paper was not populated yet. As a re‐sult, we had to retrieve the CIFAR‐10‐LT dataset from previous works [2, 3], re‐implementWideResNet [4], and the overall training pipeline.Communication with original authors — We contacted the authors for clarifications on theimplementation details of the algorithm. Prior works had many important implemen‐tation details such as linear learning rate warmup or deferred oversampling, so we con‐firmed with the authors on whether these methods were used.</details>

### [Re] Variational Neural Cellular Automata

**Authors:** Albert Sund Aillet, Simon Sondén

**<details><summary>Abstract**</summary>

The main claim of the paper being reproduced is that the proposed Variational Neural Cellular Automata (VNCA) architecture, composed of a convolutional encoder and a Neural Cellular Automata (NCA)-based decoder, is able to generate high-quality samples.The paper presents two variants of this VNCA decoder: the doubling VNCA variant that is claimed to have a simple latent space, and the non-doubling VNCA variant that is claimed to be optimized for damage recovery and stability over many steps.To reproduce the results, we re-implemented all of the VNCA models and a fully-convolutional baseline in JAX, by using the descriptions given in the paper. We then followed the same experimental setup and hyperparameter choices as in the original paper. All of the models were trained on a TPU v3-8 provided by Kaggle, with a total budget of around 4 TPU hours, not counting unreported experiments.All but one of the figures and results from the original study were possible to reproduce. The obtained Evidence Lower Bound (ELBO) of the doubling VNCA was within $0.3\%$ of the stated and for the non-doubling VNCA the ELBO was within $1.8\%$ and the observed damage recovery was similar. We were however not able to reproduce the t-SNE reduction experiment for the baseline and were therefore not able to show the VNCA decoder having a cleaner t-SNE separation than the baseline.The implementation of the convolutional baseline and most parts of the NCA-based decoder were straightforward to re-implement based on the description provided in the paper.One of the difficulties faced in the reproduction study was obtaining the labels of the binarized MNIST dataset used in the original paper since it officially is provided without labels. This made it unclear how the original paper got the labels for the latent space visualization. Additionally, implementing the pool for the non-doubling version of the VNCA model with a distributed training setup was challenging as it required operations between TPU cores.No communication was made with the original authors.</details>

### trajdata: A Unified Interface to Multiple Human Trajectory Datasets

**Authors:** Boris Ivanovic, Guanyu Song, Igor Gilitschenski, Marco Pavone

**<details><summary>Abstract**</summary>

The field of trajectory forecasting has grown significantly in recent years, partially owing to the release of numerous large-scale, real-world human trajectory datasets for autonomous vehicles (AVs) and pedestrian motion tracking. While such datasets have been a boon for the community, they each use custom and unique data formats and APIs, making it cumbersome for researchers to train and evaluate methods across multiple datasets. To remedy this, we present trajdata: a unified interface to multiple human trajectory datasets. At its core, trajdata provides a simple, uniform, and efficient representation and API for trajectory and map data. As a demonstration of its capabilities, in this work we conduct a comprehensive empirical evaluation of existing trajectory datasets, providing users with a rich understanding of the data underpinning much of current pedestrian and AV motion forecasting research, and proposing suggestions for future datasets from these insights. trajdata is permissively licensed (Apache 2.0) and can be accessed online at https://github.com/NVlabs/trajdata.</details>

